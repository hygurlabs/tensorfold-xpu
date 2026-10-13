//! DFlash2 drafter for the Qwen3.8-27B XPU runner (after tensorfold/families/qwen3_5/cuda/dflash2.py): its bf16
//! projections quantized to affine 4/64 at load and run on the target's int4 GEMM, its context the target taps of
//! the committed rows (K/V per layer at absolute positions), a block of L rows proposing L − 1 tokens, and the
//! candidate selector (top 16 a depth, rank-256 edge codebooks) choosing one chain. Drafts only change acceptance:
//! the target verifies every token.
const std = @import("std");
const ze = @import("ze.zig");
const kb = @import("xpu_qwen27_kernels");
const q = @import("qwen27.zig");

const alloc = q.alloc;
const D = q.D;
const GS = 64; // the drafter's projections: affine 4/64 whatever the target's format
const HIDDEN_NORM_EPS: f32 = 1e-6;
const LAYERS = 5;
const NH = 32;
const NKV = 8;
const HD = 128;
const FF = 17408;
pub const WIN = 2047; // context rows a block row sees (sliding_window − 1)
const MASK_ID = 248070;
const MAXL = 16; // block rows at most
pub const MAXS = 16; // blocks (sequences) a batched lattice or absorb at most
const MAXB = 128; // rows a batched lattice at most
const SPAN_LO = 98304; // the drafter's head: target vocab [0, 98 304) ∪ [248 032, 248 320)
const SPAN_HI0 = 248032;
const HEAD_N = SPAN_LO + (q.VOCAB - SPAN_HI0); // 98 592
const RANK = 256;
pub const TOPK = 16;
pub const DEPTHS = MAXL;
// the tree policy (tensorfold's Metal values): edge weight, weight of the target's noise, temperature of the pops,
// children a node pushes
const EDGE = 0.6;
const NOISE = 0.7;
const SCALE = 1.5;
const BRANCH = 4;

const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    kp_attn: q.Proj,
    base_attn: *anyopaque,
    qkv: q.Proj,
    kv: q.Proj,
    qn: *anyopaque,
    kn: *anyopaque,
    o: q.Proj,
    kp_mlp: q.Proj,
    base_mlp: *anyopaque,
    gu: q.Proj,
    down: q.Proj,
};

/// A sequence's context: each layer's K, V (bf16 [NKV][CAP][HD]) in a ring of CAP positions (a block row reads the
/// WIN before it), the target rows from `start` on absorbed up to `end`.
pub const Ctx = struct {
    kc: [LAYERS]*anyopaque,
    vc: [LAYERS]*anyopaque,
    start: u32 = 0,
    end: u32 = 0,
};
const CAP = WIN + MAXL + 17; // 2 080

/// One block's lattice: per depth the top 16 candidates (target ids), their logits, the selector projection.
const Lat = struct {
    cand: [MAXL][TOPK]u32 = undefined,
    unary: [MAXL][TOPK]f64 = undefined,
    sp: [MAXL * RANK]f16 = undefined,
    depths: u32 = 0,
};

/// A block of a batched lattice: a slot's context and its pending token.
pub const Block = struct { slot: u32, pending: u32 };
/// Rows to absorb into a slot's context (target taps bf16 [n, 5 D] on the device).
pub const Absorb = struct { slot: u32, taps: *anyopaque, n: u32 };

/// A device-side MLX affine part (words [n, k/8], scales and biases bf16 [n, k/64]) to repack at column n0.
const Mlx = struct { w: *anyopaque, s: *anyopaque, b: *anyopaque, n: u32 };

/// The tree search's frontier: a few dozen items, popped smallest (score, parent, token, depth) first like heapq.
const Heap = struct {
    const Item = struct { score: f64, parent: i32, token: u32, depth: u32 };
    items: [MAXL * BRANCH + BRANCH]Item = undefined,
    n: usize = 0,
    fn push(h: *Heap, it: Item) void {
        if (h.n < h.items.len) {
            h.items[h.n] = it;
            h.n += 1;
        }
    }
    fn before(a: Item, b: Item) bool {
        if (a.score != b.score) return a.score < b.score;
        if (a.parent != b.parent) return a.parent < b.parent;
        if (a.token != b.token) return a.token < b.token;
        return a.depth < b.depth;
    }
    fn pop(h: *Heap) ?Item {
        if (h.n == 0) return null;
        var best: usize = 0;
        for (1..h.n) |i| if (before(h.items[i], h.items[best])) {
            best = i;
        };
        const it = h.items[best];
        h.n -= 1;
        h.items[best] = h.items[h.n];
        return it;
    }
};

pub const Drafter = struct {
    r: *q.Runner,
    k: struct { dconv: ze.H, dprep: ze.H, dattn: ze.H, rowsums_rows: ze.H, taps_in: ze.H, to_res: ze.H, topk_part: ze.H, quantize4: ze.H },
    layers: [LAYERS]Layer,
    fc: q.Proj,
    hidden_norm: *anyopaque,
    norm: *anyopaque,
    sel: q.Proj,
    head: q.Proj,
    invf: *anyopaque,
    pred: []const u8, // codebooks bf16 [vocab, 256], on the host (mmap)
    succ: []const u8,
    ctxs: [q.MAX_SLOTS]?Ctx = @splat(null), // a context a runner slot, on its first use
    cur: u32 = 0, // the context every call works on (use)
    // the last block's lattice: per depth the top 16 candidates (target ids), their logits, the selector projection
    cand: [MAXL][TOPK]u32 = undefined,
    unary: [MAXL][TOPK]f64 = undefined,
    sp: [MAXL * RANK]f16 = undefined,
    depths: u32 = 0,
    multi: [MAXS]Lat = undefined, // a batched lattice's blocks (take() makes one the current)
    rows_host: [MAXB][4]i32 = undefined, // a batched pass's rows: position, context start, block start, block
    ptrs_host: [LAYERS][MAXS][2]u64 = undefined, // and each block's K, V caches by layer
    rows_dev: *anyopaque,
    ptrs_dev: *anyopaque,
    tapsbuf: *anyopaque, // a batched absorb's rows gathered, bf16 [MAXB, 5 D]
    pv_host: []f32,
    pi_host: []i32,
    // buffers
    x: *anyopaque, // fp32 [rows, D]
    h: *anyopaque, // fp16 [rows, 25 600]
    h2: *anyopaque, // fp16 [MAXL, FF]
    ahi: *anyopaque,
    alo: *anyopaque,
    p: *anyopaque, // fp16 [rows, 2 FF]
    r16: *anyopaque, // fp16 [rows, D]
    dyn: *anyopaque, // fp16 [MAXL, 1 280]
    qb: *anyopaque, // bf16 [MAXL, 32, 128]
    o: *anyopaque, // fp16 [MAXL, 4 096]
    logits: *anyopaque, // fp16 [MAXL, head.n]
    selp: *anyopaque, // fp16 [MAXL, 256]
    pv: *anyopaque,
    pi: *anyopaque,
    ids: *anyopaque,
    pscr: *anyopaque,

    fn pack(d: *q.Dev, parts: []const Mlx, k: u32, wide: bool) !q.Proj {
        var n_real: u32 = 0;
        for (parts) |p| n_real += p.n;
        const n = (n_real + 255) / 256 * 256;
        const out: q.Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n * 4), .s = try d.alloc_(@as(usize, k / GS) * n * 2), .bg = try d.alloc_(@as(usize, k / GS) * n * 2), .n = n, .k = k, .wide = wide };
        if (n != n_real) {
            try d.fill(out.s, @as(u16, 0), @as(usize, k / GS) * n * 2);
            try d.fill(out.bg, @as(u16, 0), @as(usize, k / GS) * n * 2);
        }
        var n0: u32 = 0;
        for (parts) |p| {
            try d.launch(d.k.repack, .{ .x = (p.n + 15) / 16, .y = k / 8 }, .{ p.w, p.s, p.b, out.w, out.s, out.bg, @as(i32, @intCast(p.n)), @as(i32, @intCast(k)), @as(i32, @intCast(n)), @as(i32, @intCast(n0)), @as(i32, 0) });
            n0 += p.n;
        }
        try d.sync();
        return out;
    }

    /// bf16 weights [n_i, k] stacked on N, quantized to affine 4/64 on the device, then repacked.
    fn quantized(dr: *Drafter, c: *const q.Checkpoint, names: []const []const u8, wide: bool, stage: *anyopaque) !q.Proj {
        const d = &dr.r.d;
        var parts: [4]Mlx = undefined;
        var k: u32 = 0;
        var at: usize = 0;
        for (names, 0..) |name, i| {
            const t = try c.get(name);
            const n: u32 = @intCast(t.shape[0]);
            k = @intCast(t.shape[1]);
            const raw = q.off(stage, at);
            try d.copyIn(raw, t.bytes);
            at += t.bytes.len;
            const mw = q.off(stage, at);
            at += @as(usize, n) * (k / 8) * 4;
            const ms = q.off(stage, at);
            at += @as(usize, n) * (k / GS) * 2;
            const mb = q.off(stage, at);
            at += @as(usize, n) * (k / GS) * 2;
            at = (at + 4095) / 4096 * 4096;
            try d.launch(dr.k.quantize4, .{ .x = (n + 15) / 16, .y = k / GS }, .{ raw, @as(i32, @intCast(n)), @as(i32, @intCast(k)), mw, ms, mb });
            parts[i] = .{ .w = mw, .s = ms, .b = mb, .n = n };
        }
        try d.sync();
        return pack(d, parts[0..names.len], k, wide);
    }

    pub fn init(r: *q.Runner, dir: []const u8) !*Drafter {
        const d = &r.d;
        var c: q.Checkpoint = .{};
        try c.addShard(try std.fmt.allocPrintSentinel(alloc, "{s}/model.safetensors", .{dir}, 0));
        const dr = try alloc.create(Drafter);
        const m = try d.g.module(&kb.drafter);
        const kern = struct {
            fn f(dd: *q.Dev, mm: ze.H, name: [*:0]const u8, local: u32) !ze.H {
                return dd.kernel(mm, name, local);
            }
        }.f;
        const rows = @max(q.CHUNK, MAXB);
        dr.* = .{
            .r = r,
            .k = .{ .dconv = try kern(d, m, "dconv", 512), .dprep = try kern(d, m, "dprep", 16), .dattn = try kern(d, m, "dattn", 16), .rowsums_rows = try kern(d, m, "rowsums_rows", 256), .taps_in = try kern(d, m, "taps_in", 256), .to_res = try kern(d, m, "to_res", 256), .topk_part = try kern(d, m, "topk_part", 16), .quantize4 = try kern(d, m, "quantize4", 16) },
            .layers = undefined,
            .fc = undefined,
            .hidden_norm = undefined,
            .norm = undefined,
            .sel = undefined,
            .head = undefined,
            .invf = undefined,
            .pred = (try c.get("candidate_selector.predecessor_codebook")).bytes,
            .succ = (try c.get("candidate_selector.successor_codebook")).bytes,
            .x = try d.alloc_(rows * D * 4),
            .h = try d.alloc_(rows * 25600 * 2),
            .h2 = try d.alloc_(MAXB * FF * 2),
            .ahi = try d.alloc_(rows * 400 * 2),
            .alo = try d.alloc_(rows * 400 * 2),
            .p = try d.alloc_(rows * 2 * FF * 2),
            .r16 = try d.alloc_(rows * D * 2),
            .dyn = try d.alloc_(MAXB * 1280 * 2),
            .qb = try d.alloc_(MAXB * NH * HD * 2),
            .o = try d.alloc_(MAXB * NH * HD * 2),
            .logits = undefined,
            .selp = try d.alloc_(MAXB * RANK * 2),
            .pv = try d.alloc_(MAXB * 256 * 16 * 4),
            .pi = try d.alloc_(MAXB * 256 * 16 * 4),
            .ids = try d.alloc_(MAXB * 4),
            .rows_dev = try d.alloc_(MAXB * 16),
            .ptrs_dev = try d.alloc_(LAYERS * MAXS * 16),
            .tapsbuf = try d.alloc_(MAXB * 5 * D * 2),
            .pv_host = try alloc.alloc(f32, MAXB * 256 * 16),
            .pi_host = try alloc.alloc(i32, MAXB * 256 * 16),
            .pscr = try d.alloc_(rows * D * 4),
        };
        const stage = try d.alloc_(800 << 20);
        defer d.g.free(stage);
        const bf = struct {
            fn f(dd: *q.Dev, cc: *const q.Checkpoint, name: []const u8) !*anyopaque {
                return dd.upload((try cc.get(name)).bytes);
            }
        }.f;
        const nm = struct {
            fn f(l: usize, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "layers.{d}.{s}", .{ l, s }) catch unreachable;
            }
        }.f;
        dr.fc = try dr.quantized(&c, &.{"fc.weight"}, false, stage);
        dr.hidden_norm = try bf(d, &c, "hidden_norm.weight");
        dr.norm = try bf(d, &c, "norm.weight");
        dr.sel = try dr.quantized(&c, &.{"candidate_selector.hidden_projection.weight"}, false, stage);
        for (&dr.layers, 0..) |*l, i| l.* = .{
            .in_norm = try bf(d, &c, nm(i, "input_layernorm.weight")),
            .post_norm = try bf(d, &c, nm(i, "post_attention_layernorm.weight")),
            .kp_attn = try dr.quantized(&c, &.{nm(i, "attention_conv.kernel_projection.weight")}, false, stage),
            .base_attn = try bf(d, &c, nm(i, "attention_conv.base_kernel")),
            .qkv = try dr.quantized(&c, &.{ nm(i, "self_attn.q_proj.weight"), nm(i, "self_attn.k_proj.weight"), nm(i, "self_attn.v_proj.weight") }, false, stage),
            .kv = try dr.quantized(&c, &.{ nm(i, "self_attn.k_proj.weight"), nm(i, "self_attn.v_proj.weight") }, false, stage),
            .qn = try bf(d, &c, nm(i, "self_attn.q_norm.weight")),
            .kn = try bf(d, &c, nm(i, "self_attn.k_norm.weight")),
            .o = try dr.quantized(&c, &.{nm(i, "self_attn.o_proj.weight")}, false, stage),
            .kp_mlp = try dr.quantized(&c, &.{nm(i, "mlp_conv.kernel_projection.weight")}, false, stage),
            .base_mlp = try bf(d, &c, nm(i, "mlp_conv.base_kernel")),
            .gu = try dr.quantized(&c, &.{ nm(i, "mlp.gate_proj.weight"), nm(i, "mlp.up_proj.weight") }, true, stage),
            .down = try dr.quantized(&c, &.{nm(i, "mlp.down_proj.weight")}, false, stage),
        };
        // the head: the target's lm_head rows of the two spans (its own 4-bit rows, repacked; from a GPTQ target, its
        // fp16 rows quantized as the target's head is, to the same values)
        if (!r.ckpt.has("lm_head.scales")) {
            const w = try r.ckpt.get("lm_head.weight");
            const kk: u32 = @intCast(w.shape[1]);
            const raw = try d.alloc_(32768 * @as(usize, kk) * 2);
            defer d.g.free(raw);
            var parts: [2]Mlx = undefined;
            var at: usize = 0;
            for ([_][2]u32{ .{ 0, SPAN_LO }, .{ SPAN_HI0, q.VOCAB } }, 0..) |span, j| {
                const n = span[1] - span[0];
                const pw = q.off(stage, at);
                at += @as(usize, n) * (kk / 8) * 4;
                const ps = q.off(stage, at);
                at += @as(usize, n) * (kk / GS) * 2;
                const pb = q.off(stage, at);
                at += @as(usize, n) * (kk / GS) * 2;
                at = (at + 4095) / 4096 * 4096;
                var r0: u32 = 0;
                while (r0 < n) : (r0 += 32768) {
                    const nr = @min(32768, n - r0);
                    const src = w.bytes[@as(usize, span[0] + r0) * kk * 2 ..][0 .. @as(usize, nr) * kk * 2];
                    try d.copyIn(raw, src);
                    try d.launch(d.k.quantize_mlx, .{ .x = (nr + 15) / 16, .y = kk / 64 }, .{ raw, @as(i32, @intCast(nr)), @as(i32, @intCast(kk)), q.off(pw, @as(usize, r0) * (kk / 8) * 4), q.off(ps, @as(usize, r0) * (kk / GS) * 2), q.off(pb, @as(usize, r0) * (kk / GS) * 2) });
                    try d.sync();
                }
                parts[j] = .{ .w = pw, .s = ps, .b = pb, .n = n };
            }
            dr.head = try pack(d, &parts, kk, true);
        } else {
            const w = try r.ckpt.get("lm_head.weight");
            const s = try r.ckpt.get("lm_head.scales");
            const b = try r.ckpt.get("lm_head.biases");
            const kk: u32 = @intCast(w.shape[1] * 8);
            const wr = kk / 8 * 4;
            const sr = kk / GS * 2;
            var parts: [2]Mlx = undefined;
            var at: usize = 0;
            for ([_][2]u32{ .{ 0, SPAN_LO }, .{ SPAN_HI0, q.VOCAB } }, 0..) |span, j| {
                const n = span[1] - span[0];
                const pw = q.off(stage, at);
                try d.copyIn(pw, w.bytes[span[0] * wr .. span[1] * wr]);
                at += n * wr;
                const ps = q.off(stage, at);
                try d.copyIn(ps, s.bytes[span[0] * sr .. span[1] * sr]);
                at += n * sr;
                const pb = q.off(stage, at);
                try d.copyIn(pb, b.bytes[span[0] * sr .. span[1] * sr]);
                at += n * sr;
                at = (at + 4095) / 4096 * 4096;
                parts[j] = .{ .w = pw, .s = ps, .b = pb, .n = n };
            }
            dr.head = try pack(d, &parts, kk, true);
        }
        dr.logits = try d.alloc_(MAXB * @as(usize, dr.head.n) * 2);
        var invf: [64]f32 = undefined;
        for (&invf, 0..) |*f, i| f.* = @floatCast(1.0 / std.math.pow(f64, 1e7, @as(f64, @floatFromInt(2 * i)) / 128.0));
        dr.invf = try d.upload(std.mem.sliceAsBytes(&invf));
        try dr.use(0);
        return dr;
    }

    /// Makes slot `slot`'s context the one every call works on (allocated on first use: ~42 MB).
    pub fn use(dr: *Drafter, slot: u32) !void {
        if (slot >= q.MAX_SLOTS) return error.NoSuchSlot;
        if (dr.ctxs[slot] == null) {
            const d = &dr.r.d;
            var x: Ctx = undefined;
            for (&x.kc, &x.vc) |*k, *v| {
                k.* = try d.alloc_(@as(usize, NKV) * CAP * HD * 2);
                v.* = try d.alloc_(@as(usize, NKV) * CAP * HD * 2);
            }
            x.start = 0;
            x.end = 0;
            dr.ctxs[slot] = x;
        }
        dr.cur = slot;
    }

    fn ctx(dr: *Drafter) *Ctx {
        return &dr.ctxs[dr.cur].?;
    }

    /// The position after the context's last absorbed row.
    pub fn end(dr: *Drafter) u32 {
        return dr.ctx().end;
    }

    /// A fresh sequence: no context rows.
    pub fn reset(dr: *Drafter) void {
        dr.restart(0);
    }

    /// A context from position `at` on (the target's rows before it unseen): addTaps appends from there.
    pub fn restart(dr: *Drafter, at: u32) void {
        dr.ctx().start = at;
        dr.ctx().end = at;
    }

    /// Appends `n` committed target rows (taps bf16 [n, 5 D] on the device) to the context: fc, its norm, then each
    /// layer's k (normed, rotated) and v at positions end .. end + n.
    pub fn addTaps(dr: *Drafter, taps: *anyopaque, n: u32) !void {
        const d = &dr.r.d;
        const nul: ?*anyopaque = null;
        const base = dr.ctx().end;
        // a block row reads the WIN rows before it: of more, only the last WIN are stored (the ring holds CAP)
        var done: u32 = n -| WIN;
        if (done > 0) dr.ctx().start = @max(dr.ctx().start, base + done);
        while (done < n) {
            const rows = @min(n - done, q.CHUNK);
            const src = q.off(taps, @as(usize, done) * 5 * D * 2);
            try d.launch(dr.k.taps_in, .{ .x = rows }, .{ src, dr.h, dr.ahi, dr.alo });
            try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.fc, dr.r16, rows);
            try d.launch(dr.k.to_res, .{ .x = rows * D / 256 }, .{ dr.r16, dr.x });
            try d.launch(d.k.add_rmsnorm, .{ .x = rows }, .{ dr.x, dr.r16, dr.hidden_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0), @as(i32, GS), q.no_q8, q.no_q8, @as(i32, 0), @as(i32, 0) });
            const c = dr.ctx();
            for (&dr.layers, c.kc, c.vc) |*l, kc, vc| {
                try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kv, dr.p, rows);
                try d.launch(dr.k.dprep, .{ .x = rows, .y = 16 }, .{ dr.p, @as(i32, @intCast(l.kv.n)), @as(i32, -1), l.qn, l.kn, dr.invf, dr.qb, kc, vc, @as(i32, CAP), @as(i32, @intCast(base + done)), HIDDEN_NORM_EPS, nul, nul });
            }
            done += rows;
        }
        dr.ctx().end = base + n; // no wait: the next block queues behind on the in-order list
    }

    /// Moves the context past `n` target rows it will never read (a long prompt's first rows, further than WIN from
    /// its end).
    pub fn skip(dr: *Drafter, n: u32) void {
        dr.ctx().end += n;
        dr.ctx().start = dr.ctx().end;
    }

    /// Proposes `count` (≤ MAXL − 1) drafts after `pending`, the token at position `end`: one chain, each depth's
    /// best child under the selector's score (unary + edge).
    pub fn propose(dr: *Drafter, pending: u32, count: u32, out: []u32) !void {
        try dr.lattice(pending, count);
        dr.chain(pending, count, out);
    }

    /// The chain over the current lattice (propose's choice).
    pub fn chain(dr: *const Drafter, pending: u32, count: u32, out: []u32) void {
        var parent = pending;
        for (0..count) |depth| {
            var best: usize = 0;
            var best_score: f64 = -std.math.inf(f64);
            for (0..TOPK) |j| {
                const tok = dr.cand[depth][j];
                const score = dr.unary[depth][j] + EDGE * dr.edge(parent, depth, tok);
                if (score > best_score or (score == best_score and tok < dr.cand[depth][best])) {
                    best_score = score;
                    best = j;
                }
            }
            out[depth] = dr.cand[depth][best];
            parent = out[depth];
        }
    }

    fn edge(dr: *const Drafter, parent: u32, depth: usize, tok: u32) f64 {
        var e: f64 = 0;
        for (0..RANK) |rr| e += code(dr.pred, parent, rr) * @as(f64, dr.sp[depth * RANK + rr]) * code(dr.succ, tok, rr);
        return e;
    }

    /// A tree of at most `max_nodes` drafts over the last lattice (tensorfold's DFlash2 best-first policy, after
    /// families/qwen3_5/cuda/draft_tree.py): nodes in pop order, parents first, `parents[i]` −1 at the root (the
    /// pending token). `noise` (sampled replies): the target's keyed Gumbel noise of each candidate, `temp` its
    /// temperature. Returns the node count.
    pub fn tree(dr: *Drafter, pending: u32, max_nodes: u32, noise: ?*const [MAXL][TOPK]f64, temp: f64, tokens: []u32, parents: []i32) u32 {
        var heap: Heap = .{};
        dr.expand(&heap, noise, @max(temp, 1e-6), pending, 0, -1, 0);
        var n: u32 = 0;
        while (n < max_nodes) {
            const it = heap.pop() orelse break;
            tokens[n] = it.token;
            parents[n] = it.parent;
            const me: i32 = @intCast(n);
            n += 1;
            if (it.depth + 1 < dr.depths) dr.expand(&heap, noise, @max(temp, 1e-6), it.token, it.depth + 1, me, it.score);
        }
        return n;
    }

    /// Pushes the BRANCH best children of `token` at `depth`, scored by path − log p under the policy.
    fn expand(dr: *const Drafter, heap: *Heap, noise: ?*const [MAXL][TOPK]f64, temp: f64, token: u32, depth: u32, parent: i32, path: f64) void {
        var v: [TOPK]f64 = undefined;
        for (0..TOPK) |j| {
            v[j] = (dr.unary[depth][j] + EDGE * dr.edge(token, depth, dr.cand[depth][j])) / temp;
            if (noise) |nz| v[j] += NOISE * nz[depth][j];
            v[j] /= SCALE;
        }
        var mx = v[0];
        for (v) |x| mx = @max(mx, x);
        var sum: f64 = 0;
        for (&v) |*x| {
            x.* -= mx;
            sum += @exp(x.*);
        }
        const lse = @log(sum);
        var taken: [TOPK]bool = @splat(false);
        for (0..BRANCH) |_| { // the BRANCH largest, ties to the lower index (numpy's argsort of −logp)
            var j: usize = TOPK;
            for (0..TOPK) |i| if (!taken[i] and (j == TOPK or v[i] > v[j])) {
                j = i;
            };
            taken[j] = true;
            const lp = v[j] - lse;
            if (!std.math.isFinite(lp)) break;
            heap.push(.{ .score = path - lp, .parent = parent, .token = dr.cand[depth][j], .depth = depth });
        }
    }

    /// The drafter's block after `pending`: per depth 1 .. count the top 16 candidates and the selector projection.
    pub fn lattice(dr: *Drafter, pending: u32, count: u32) !void {
        try dr.run(&.{.{ .slot = dr.cur, .pending = pending }}, count);
        dr.take(0);
    }

    /// One block of `count` + 1 rows a slot (its context, its pending token), all in one pass; take(i) makes block
    /// i's lattice the current one.
    pub fn latticeMulti(dr: *Drafter, items: []const Block, count: u32) !void {
        try dr.run(items, count);
    }

    pub fn take(dr: *Drafter, i: usize) void {
        const x = &dr.multi[i];
        dr.cand = x.cand;
        dr.unary = x.unary;
        dr.sp = x.sp;
        dr.depths = x.depths;
    }

    /// Each item's rows into its slot's context, in one pass (fc, its norm, each layer's k and v).
    pub fn absorbMulti(dr: *Drafter, items: []const Absorb) !void {
        if (items.len == 0) return;
        if (items.len == 1) {
            try dr.use(items[0].slot);
            return dr.addTaps(items[0].taps, items[0].n);
        }
        if (items.len > MAXS) return error.TooManyBlocks;
        const d = &dr.r.d;
        var R: u32 = 0;
        for (items, 0..) |it, b| {
            try dr.use(it.slot);
            if (R + it.n > MAXB or it.n > WIN) return error.TooManyRows;
            try q.copy(d, q.off(dr.tapsbuf, @as(usize, R) * 5 * D * 2), it.taps, @as(usize, it.n) * 5 * D * 2);
            const c = dr.ctx();
            for (0..it.n) |j| dr.rows_host[R + j] = .{ @intCast(c.end + j), 0, 0, @intCast(b) };
            for (0..LAYERS) |li| dr.ptrs_host[li][b] = .{ @intFromPtr(c.kc[li]), @intFromPtr(c.vc[li]) };
            R += it.n;
        }
        try dr.tables(R);
        try d.launch(dr.k.taps_in, .{ .x = R }, .{ dr.tapsbuf, dr.h, dr.ahi, dr.alo });
        try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.fc, dr.r16, R);
        try d.launch(dr.k.to_res, .{ .x = R * D / 256 }, .{ dr.r16, dr.x });
        try d.launch(d.k.add_rmsnorm, .{ .x = R }, .{ dr.x, dr.r16, dr.hidden_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0), @as(i32, GS), q.no_q8, q.no_q8, @as(i32, 0), @as(i32, 0) });
        for (&dr.layers, 0..) |*l, li| {
            try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kv, dr.p, R);
            try d.launch(dr.k.dprep, .{ .x = R, .y = 16 }, .{ dr.p, @as(i32, @intCast(l.kv.n)), @as(i32, -1), l.qn, l.kn, dr.invf, dr.qb, dr.ptrs_host[li][0][0], dr.ptrs_host[li][0][1], @as(i32, CAP), @as(i32, 0), HIDDEN_NORM_EPS, dr.rows_dev, q.off(dr.ptrs_dev, li * MAXS * 16) });
        }
        for (items) |it| {
            try dr.use(it.slot);
            dr.ctx().end += it.n;
        }
        try d.sync(); // the tables are read from the host
    }

    /// The batched pass's row and cache tables to the device.
    fn tables(dr: *Drafter, rows: u32) !void {
        const d = &dr.r.d;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dr.rows_dev, &dr.rows_host, @as(usize, rows) * 16, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dr.ptrs_dev, &dr.ptrs_host, LAYERS * MAXS * 16, null, 0, null));
    }

    fn run(dr: *Drafter, items: []const Block, count: u32) !void {
        const d = &dr.r.d;
        const L = count + 1;
        const S: u32 = @intCast(items.len);
        const R = S * L;
        if (S == 0 or S > MAXS or R > MAXB or L > MAXL) return error.TooManyRows;
        const one = S == 1;
        var tokens: [MAXB]i32 = undefined;
        for (items, 0..) |it, b| {
            tokens[b * L] = @intCast(it.pending);
            for (1..L) |i| tokens[b * L + i] = MASK_ID;
            if (one) continue;
            try dr.use(it.slot);
            const c = dr.ctx();
            const c0: i32 = @intCast(@max(c.end -| WIN, c.start));
            for (0..L) |i| dr.rows_host[b * L + i] = .{ @intCast(c.end + i), c0, @intCast(c.end), @intCast(b) };
            for (0..LAYERS) |li| dr.ptrs_host[li][b] = .{ @intFromPtr(c.kc[li]), @intFromPtr(c.vc[li]) };
        }
        if (one) try dr.use(items[0].slot) else try dr.tables(R);
        const nul: ?*anyopaque = null;
        const rows_t: ?*anyopaque = if (one) nul else dr.rows_dev;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dr.ids, &tokens, R * 4, null, 0, null));
        const m = &dr.r.model;
        try q.embedRows(d, m, dr.ids, dr.x, R);
        const l32: i32 = @intCast(L);
        const c = dr.ctx(); // a single block's (S = 1)
        const end_: i32 = @intCast(c.end);
        const c0: i32 = @intCast(@max(c.end -| WIN, c.start));
        for (&dr.layers, c.kc, c.vc, 0..) |*l, kc, vc, li| {
            const ptrs_t: ?*anyopaque = if (one) nul else q.off(dr.ptrs_dev, li * MAXS * 16);
            try d.launch(d.k.add_rmsnorm, .{ .x = R }, .{ dr.x, dr.r16, l.in_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0), @as(i32, GS), q.no_q8, q.no_q8, @as(i32, 0), @as(i32, 0) });
            try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kp_attn, dr.dyn, R);
            try d.launch(dr.k.dconv, .{ .x = R, .y = D / 512 }, .{ dr.h, dr.dyn, @as(i32, @intCast(l.kp_attn.n)), l.base_attn, @as(i32, 0), dr.x, @as(i32, 0), dr.h2, dr.ahi, dr.alo, l32 });
            try q.gemm(d, dr.pscr, dr.h2, dr.ahi, dr.alo, l.qkv, dr.p, R);
            try d.launch(dr.k.dprep, .{ .x = R, .y = 48 }, .{ dr.p, @as(i32, @intCast(l.qkv.n)), @as(i32, 0), l.qn, l.kn, dr.invf, dr.qb, kc, vc, @as(i32, CAP), end_, HIDDEN_NORM_EPS, rows_t, ptrs_t });
            try d.launch(dr.k.dattn, .{ .x = R, .y = NKV }, .{ dr.qb, kc, vc, @as(i32, CAP), c0, end_, l32, @as(i32, WIN), @as(f32, 1.0 / @sqrt(@as(f32, HD))), dr.o, rows_t, ptrs_t });
            try d.launch(dr.k.rowsums_rows, .{ .x = R }, .{ dr.o, @as(i32, NH * HD), dr.ahi, dr.alo });
            try q.gemm(d, dr.pscr, dr.o, dr.ahi, dr.alo, l.o, dr.r16, R);
            try d.launch(dr.k.dconv, .{ .x = R, .y = D / 512 }, .{ dr.r16, dr.dyn, @as(i32, @intCast(l.kp_attn.n)), l.base_attn, @as(i32, 1), dr.x, @as(i32, 1), dr.h2, dr.ahi, dr.alo, l32 });
            try d.launch(d.k.add_rmsnorm, .{ .x = R }, .{ dr.x, dr.r16, l.post_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0), @as(i32, GS), q.no_q8, q.no_q8, @as(i32, 0), @as(i32, 0) });
            try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kp_mlp, dr.dyn, R);
            try d.launch(dr.k.dconv, .{ .x = R, .y = D / 512 }, .{ dr.h, dr.dyn, @as(i32, @intCast(l.kp_mlp.n)), l.base_mlp, @as(i32, 0), dr.x, @as(i32, 0), dr.h2, dr.ahi, dr.alo, l32 });
            try q.gemm(d, dr.pscr, dr.h2, dr.ahi, dr.alo, l.gu, dr.p, R);
            try d.launch(d.k.swiglu, .{ .x = R, .y = 17 }, .{ dr.p, dr.h2, dr.ahi, dr.alo, @as(i32, GS), @as(i32, 0), @as(i32, 0) });
            try q.gemm(d, dr.pscr, dr.h2, dr.ahi, dr.alo, l.down, dr.r16, R);
            try d.launch(dr.k.dconv, .{ .x = R, .y = D / 512 }, .{ dr.r16, dr.dyn, @as(i32, @intCast(l.kp_mlp.n)), l.base_mlp, @as(i32, 1), dr.x, @as(i32, 1), dr.h2, dr.ahi, dr.alo, l32 });
        }
        // a block's rows 1 .. L − 1: final norm, selector projection, head over the two spans, top 16 (one block: only
        // those rows; several: every row, row 0 of each unread)
        const first: u32 = if (one) 1 else 0;
        const rows = if (one) count else R;
        try d.launch(d.k.add_rmsnorm, .{ .x = rows }, .{ q.off(dr.x, first * D * 4), dr.r16, dr.norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0), @as(i32, GS), q.no_q8, q.no_q8, @as(i32, 0), @as(i32, 0) });
        try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.sel, dr.selp, rows);
        try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.head, dr.logits, rows);
        try d.launch(dr.k.topk_part, .{ .x = 256, .y = rows }, .{ dr.logits, @as(i32, HEAD_N), @as(i32, @intCast(dr.head.n)), dr.pv, dr.pi });
        const pv = dr.pv_host;
        const pi = dr.pi_host;
        var sp: [MAXB * RANK]f16 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, pv.ptr, dr.pv, rows * 256 * 16 * 4, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, pi.ptr, dr.pi, rows * 256 * 16 * 4, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &sp, dr.selp, rows * RANK * 2, null, 0, null));
        try d.sync();
        for (0..S) |b| {
            const x = &dr.multi[b];
            for (0..count) |depth| {
                const row = if (one) depth else b * L + 1 + depth;
                var bv: [TOPK]f32 = @splat(-std.math.inf(f32));
                var bi: [TOPK]i32 = @splat(std.math.maxInt(i32));
                for (0..256 * 16) |j| {
                    const v = pv[row * 4096 + j];
                    const ix = pi[row * 4096 + j];
                    if (ix == std.math.maxInt(i32)) continue;
                    if (v < bv[TOPK - 1] or (v == bv[TOPK - 1] and ix > bi[TOPK - 1])) continue;
                    var k: usize = TOPK - 1;
                    while (k > 0 and (v > bv[k - 1] or (v == bv[k - 1] and ix < bi[k - 1]))) : (k -= 1) {
                        bv[k] = bv[k - 1];
                        bi[k] = bi[k - 1];
                    }
                    bv[k] = v;
                    bi[k] = ix;
                }
                for (0..TOPK) |j| {
                    const local: u32 = @intCast(bi[j]);
                    x.cand[depth][j] = if (local < SPAN_LO) local else SPAN_HI0 + (local - SPAN_LO);
                    x.unary[depth][j] = bv[j];
                }
                @memcpy(x.sp[depth * RANK ..][0..RANK], sp[row * RANK ..][0..RANK]);
            }
            x.depths = count;
        }
    }

    fn code(t: []const u8, row: u32, col: usize) f64 {
        const at = (@as(usize, row) * RANK + col) * 2;
        const bits = std.mem.readInt(u16, t[at..][0..2], .little);
        return @as(f32, @bitCast(@as(u32, bits) << 16));
    }
};

/// How a round draws the target's tokens: greedy (each verified row's argmax) without one; with one, `row` draws
/// row `row` of the verified block at `position` and `gumbel` gives the target's keyed noise of a candidate there.
pub const Draw = struct {
    ctx: *anyopaque,
    row: *const fn (*anyopaque, row: u32, position: u64) anyerror!u32,
    gumbel: *const fn (*anyopaque, position: u64, id: u64) f64,
    temperature: f64,
};

pub const Round = struct { tokens: []const u32, terminal: u32, drafted: u32, accepted: u32 };

/// One tree round from `pending` (the token at the target's position): drafts a tree of at most `nodes`, verifies it,
/// walks the drawn tokens down the tree and keeps that path. Returns the pending token and the accepted drafts; the
/// terminal draw (the next pending token) is also `r.last` when greedy.
pub fn treeRound(dr: *Drafter, pending: u32, depths: u32, nodes: u32, draw: ?Draw, out: *[MAXL]u32) !Round {
    const r = dr.r;
    const p0: u64 = r.pos;
    try dr.lattice(pending, depths);
    var noise: [MAXL][TOPK]f64 = undefined;
    if (draw) |dw| for (0..depths) |dd| for (0..TOPK) |j| {
        noise[dd][j] = dw.gumbel(dw.ctx, p0 + 1 + dd, dr.cand[dd][j]);
    };
    var ids: [MAXL]u32 = undefined;
    var par: [MAXL]i32 = undefined;
    ids[0] = pending;
    const n = dr.tree(pending, @min(nodes, MAXL - 1), if (draw != null) &noise else null, if (draw) |dw| dw.temperature else 1.0, ids[1..], par[1..]);
    var rows: [MAXL]i32 = undefined; // the tree in verified rows: row 0 the pending token
    rows[0] = 0;
    for (1..n + 1) |i| rows[i] = if (par[i] < 0) 0 else par[i] + 1;
    const greedy = try r.verifyTree(ids[0 .. n + 1], rows[0 .. n + 1]);
    var path: [MAXL]u32 = undefined;
    path[0] = 0;
    var len: u32 = 1;
    var terminal = if (draw) |dw| try dw.row(dw.ctx, 0, p0 + 1) else greedy[0];
    walk: while (true) {
        for (1..n + 1) |c| if (rows[c] == path[len - 1] and ids[c] == terminal) {
            path[len] = @intCast(c);
            len += 1;
            terminal = if (draw) |dw| try dw.row(dw.ctx, @intCast(c), p0 + len) else greedy[c];
            continue :walk;
        };
        break;
    }
    try r.commitPath(path[0..len]);
    try dr.addTaps(r.acts.taps, len);
    for (0..len) |i| out[i] = ids[path[i]];
    return .{ .tokens = out[0..len], .terminal = terminal, .drafted = n, .accepted = len - 1 };
}
