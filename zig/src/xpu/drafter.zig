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
const GS = q.GS;
const HIDDEN_NORM_EPS: f32 = 1e-6;
const LAYERS = 5;
const NH = 32;
const NKV = 8;
const HD = 128;
const FF = 17408;
const WIN = 2047; // context rows a block row sees (sliding_window − 1)
const MASK_ID = 248070;
const MAXL = 16; // block rows at most
const SPAN_LO = 98304; // the drafter's head: target vocab [0, 98 304) ∪ [248 032, 248 320)
const SPAN_HI0 = 248032;
const HEAD_N = SPAN_LO + (q.VOCAB - SPAN_HI0); // 98 592
const RANK = 256;
const TOPK = 16;

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
    kc: *anyopaque, // bf16 [NKV][cap][HD]
    vc: *anyopaque,
};

/// A device-side MLX affine part (words [n, k/8], scales and biases bf16 [n, k/64]) to repack at column n0.
const Mlx = struct { w: *anyopaque, s: *anyopaque, b: *anyopaque, n: u32 };

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
    cap: u32,
    end: u32 = 0, // context rows held (absolute positions 0 .. end)
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
            try d.launch(d.k.repack, .{ .x = (p.n + 15) / 16, .y = k / 8 }, .{ p.w, p.s, p.b, out.w, out.s, out.bg, @as(i32, @intCast(p.n)), @as(i32, @intCast(k)), @as(i32, @intCast(n)), @as(i32, @intCast(n0)) });
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
        const rows = @max(q.CHUNK, MAXL);
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
            .cap = r.context + MAXL,
            .x = try d.alloc_(rows * D * 4),
            .h = try d.alloc_(rows * 25600 * 2),
            .h2 = try d.alloc_(MAXL * FF * 2),
            .ahi = try d.alloc_(rows * 400 * 2),
            .alo = try d.alloc_(rows * 400 * 2),
            .p = try d.alloc_(rows * 2 * FF * 2),
            .r16 = try d.alloc_(rows * D * 2),
            .dyn = try d.alloc_(MAXL * 1280 * 2),
            .qb = try d.alloc_(MAXL * NH * HD * 2),
            .o = try d.alloc_(MAXL * NH * HD * 2),
            .logits = undefined,
            .selp = try d.alloc_(MAXL * RANK * 2),
            .pv = try d.alloc_(MAXL * 256 * 16 * 4),
            .pi = try d.alloc_(MAXL * 256 * 16 * 4),
            .ids = try d.alloc_(MAXL * 4),
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
            .kc = try d.alloc_(@as(usize, NKV) * dr.cap * HD * 2),
            .vc = try d.alloc_(@as(usize, NKV) * dr.cap * HD * 2),
        };
        // the head: the target's lm_head rows of the two spans (its own 4-bit rows, repacked)
        {
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
        dr.logits = try d.alloc_(MAXL * @as(usize, dr.head.n) * 2);
        var invf: [64]f32 = undefined;
        for (&invf, 0..) |*f, i| f.* = @floatCast(1.0 / std.math.pow(f64, 1e7, @as(f64, @floatFromInt(2 * i)) / 128.0));
        dr.invf = try d.upload(std.mem.sliceAsBytes(&invf));
        return dr;
    }

    /// A fresh sequence: no context rows.
    pub fn reset(dr: *Drafter) void {
        dr.end = 0;
    }

    /// Appends `n` committed target rows (taps bf16 [n, 5 D] on the device) to the context: fc, its norm, then each
    /// layer's k (normed, rotated) and v at positions end .. end + n.
    pub fn addTaps(dr: *Drafter, taps: *anyopaque, n: u32) !void {
        const d = &dr.r.d;
        var done: u32 = 0;
        while (done < n) {
            const rows = @min(n - done, q.CHUNK);
            const src = q.off(taps, @as(usize, done) * 5 * D * 2);
            try d.launch(dr.k.taps_in, .{ .x = rows }, .{ src, dr.h, dr.ahi, dr.alo });
            try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.fc, dr.r16, rows);
            try d.launch(dr.k.to_res, .{ .x = rows * D / 256 }, .{ dr.r16, dr.x });
            try d.launch(d.k.add_rmsnorm, .{ .x = rows }, .{ dr.x, dr.r16, dr.hidden_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0) });
            for (&dr.layers) |*l| {
                try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kv, dr.p, rows);
                try d.launch(dr.k.dprep, .{ .x = rows, .y = 16 }, .{ dr.p, @as(i32, @intCast(l.kv.n)), @as(i32, -1), l.qn, l.kn, dr.invf, dr.qb, l.kc, l.vc, @as(i32, @intCast(dr.cap)), @as(i32, @intCast(dr.end + done)), HIDDEN_NORM_EPS });
            }
            done += rows;
        }
        dr.end += n; // no wait: the next block queues behind on the in-order list
    }

    /// Proposes `count` (≤ MAXL − 1) drafts after `pending`, the token at position `end`.
    pub fn propose(dr: *Drafter, pending: u32, count: u32, out: []u32) !void {
        const d = &dr.r.d;
        const L = count + 1;
        var tokens: [MAXL]i32 = undefined;
        tokens[0] = @intCast(pending);
        for (1..L) |i| tokens[i] = MASK_ID;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dr.ids, &tokens, L * 4, null, 0, null));
        const m = &dr.r.model;
        try d.launch(d.k.embed, .{ .x = L }, .{ dr.ids, m.embed_w, m.embed_s, m.embed_b, dr.x, @as(i32, D) });
        const l32: i32 = @intCast(L);
        const end: i32 = @intCast(dr.end);
        const c0: i32 = @intCast(dr.end -| WIN);
        for (&dr.layers) |*l| {
            try d.launch(d.k.add_rmsnorm, .{ .x = L }, .{ dr.x, dr.r16, l.in_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0) });
            try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kp_attn, dr.dyn, L);
            try d.launch(dr.k.dconv, .{ .x = L, .y = D / 512 }, .{ dr.h, dr.dyn, @as(i32, @intCast(l.kp_attn.n)), l.base_attn, @as(i32, 0), dr.x, @as(i32, 0), dr.h2, dr.ahi, dr.alo });
            try q.gemm(d, dr.pscr, dr.h2, dr.ahi, dr.alo, l.qkv, dr.p, L);
            try d.launch(dr.k.dprep, .{ .x = L, .y = 48 }, .{ dr.p, @as(i32, @intCast(l.qkv.n)), @as(i32, 0), l.qn, l.kn, dr.invf, dr.qb, l.kc, l.vc, @as(i32, @intCast(dr.cap)), end, HIDDEN_NORM_EPS });
            try d.launch(dr.k.dattn, .{ .x = L, .y = NKV }, .{ dr.qb, l.kc, l.vc, @as(i32, @intCast(dr.cap)), c0, end, l32, @as(i32, WIN), @as(f32, 1.0 / @sqrt(@as(f32, HD))), dr.o });
            try d.launch(dr.k.rowsums_rows, .{ .x = L }, .{ dr.o, @as(i32, NH * HD), dr.ahi, dr.alo });
            try q.gemm(d, dr.pscr, dr.o, dr.ahi, dr.alo, l.o, dr.r16, L);
            try d.launch(dr.k.dconv, .{ .x = L, .y = D / 512 }, .{ dr.r16, dr.dyn, @as(i32, @intCast(l.kp_attn.n)), l.base_attn, @as(i32, 1), dr.x, @as(i32, 1), dr.h2, dr.ahi, dr.alo });
            try d.launch(d.k.add_rmsnorm, .{ .x = L }, .{ dr.x, dr.r16, l.post_norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0) });
            try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, l.kp_mlp, dr.dyn, L);
            try d.launch(dr.k.dconv, .{ .x = L, .y = D / 512 }, .{ dr.h, dr.dyn, @as(i32, @intCast(l.kp_mlp.n)), l.base_mlp, @as(i32, 0), dr.x, @as(i32, 0), dr.h2, dr.ahi, dr.alo });
            try q.gemm(d, dr.pscr, dr.h2, dr.ahi, dr.alo, l.gu, dr.p, L);
            try d.launch(d.k.swiglu, .{ .x = L, .y = 17 }, .{ dr.p, dr.h2, dr.ahi, dr.alo });
            try q.gemm(d, dr.pscr, dr.h2, dr.ahi, dr.alo, l.down, dr.r16, L);
            try d.launch(dr.k.dconv, .{ .x = L, .y = D / 512 }, .{ dr.r16, dr.dyn, @as(i32, @intCast(l.kp_mlp.n)), l.base_mlp, @as(i32, 1), dr.x, @as(i32, 1), dr.h2, dr.ahi, dr.alo });
        }
        // rows 1 .. L − 1: final norm, selector projection, head over the two spans, top 16
        const rows = count;
        try d.launch(d.k.add_rmsnorm, .{ .x = rows }, .{ q.off(dr.x, D * 4), dr.r16, dr.norm, dr.h, dr.ahi, dr.alo, @as(i32, D), HIDDEN_NORM_EPS, @as(i32, 0) });
        try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.sel, dr.selp, rows);
        try q.gemm(d, dr.pscr, dr.h, dr.ahi, dr.alo, dr.head, dr.logits, rows);
        try d.launch(dr.k.topk_part, .{ .x = 256, .y = rows }, .{ dr.logits, @as(i32, HEAD_N), @as(i32, @intCast(dr.head.n)), dr.pv, dr.pi });
        var pv: [MAXL * 256 * 16]f32 = undefined;
        var pi: [MAXL * 256 * 16]i32 = undefined;
        var sp: [MAXL * RANK]f16 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &pv, dr.pv, rows * 256 * 16 * 4, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &pi, dr.pi, rows * 256 * 16 * 4, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &sp, dr.selp, rows * RANK * 2, null, 0, null));
        try d.sync();
        // lattice: per depth the top 16 (global ids), their logits, and the selector projection
        var cand: [MAXL][TOPK]u32 = undefined;
        var unary: [MAXL][TOPK]f64 = undefined;
        for (0..rows) |row| {
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
                cand[row][j] = if (local < SPAN_LO) local else SPAN_HI0 + (local - SPAN_LO);
                unary[row][j] = bv[j];
            }
        }
        // chain: from the pending token, each depth's best child under the selector's score (unary + edge)
        var parent = pending;
        for (0..rows) |depth| {
            var best: usize = 0;
            var best_score: f64 = -std.math.inf(f64);
            for (0..TOPK) |j| {
                const tok = cand[depth][j];
                var e: f64 = 0;
                for (0..RANK) |rr| e += code(dr.pred, parent, rr) * @as(f64, sp[depth * RANK + rr]) * code(dr.succ, tok, rr);
                const score = unary[depth][j] + 0.6 * e;
                if (score > best_score or (score == best_score and tok < cand[depth][best])) {
                    best_score = score;
                    best = j;
                }
            }
            out[depth] = cand[depth][best];
            parent = out[depth];
        }
    }

    fn code(t: []const u8, row: u32, col: usize) f64 {
        const at = (@as(usize, row) * RANK + col) * 2;
        const bits = std.mem.readInt(u16, t[at..][0..2], .little);
        return @as(f32, @bitCast(@as(u32, bits) << 16));
    }
};
