//! Qwen3.8-27B forward on an Intel Xe2 GPU (Arc Pro B70) over Level Zero: MLX affine 4-bit g64 checkpoint, prompt in
//! chunks (XPU_CHUNK rows, 1 024 by default), serial decode. `Runner` is what the native serial host drives
//! (native/xpu.zig) and what the development runner (qwen27_main.zig) times.
const std = @import("std");
const ze = @import("ze.zig");
const kb = @import("xpu_qwen27_kernels");

pub const alloc = std.heap.c_allocator;

// Qwen3.8-27B shape (checked against config.json at load).
pub const D = 5120;
const FF = 17408;
const LAYERS = 64;
pub const VOCAB = 248320;
pub const GS = 64;
const GDN_IN = 16640; // qkv 10240 | z 6144 | a 48 | b 48 = 16480, padded to the GEMM tile (256)
const ATT_IN = 14336; // (q 256 | gate 256) × 24 | k 1024 | v 1024
pub var CHUNK: u32 = 1024; // prompt rows per forward (XPU_CHUNK, a multiple of 64)
var MAX_CTX: u32 = 4096; // KV cache rows (XPU_CTX)
const SPLIT = 32; // keys per attention split (decode.cl)
pub const EPS: f32 = 1e-6;

// Layout of a GDN layer's stash (bytes): q, k, v, g, β, then the qkv rows of P.
const ST_Q = 0;
const ST_K = ST_Q + 16 * 16 * 128 * 2;
const ST_V = ST_K + 16 * 16 * 128 * 2;
const ST_G = ST_V + 16 * 48 * 128 * 2;
const ST_B = ST_G + 16 * 48 * 4;
const ST_P = ST_B + 16 * 48 * 4;
const STASH = ST_P + 16 * GDN_IN * 2;
/// Target layers whose hidden states feed the DFlash2 drafter (its config's target_layer_ids).
const TAP_LAYERS = [5]usize{ 5, 19, 33, 47, 61 };

fn isAttention(i: usize) bool {
    return (i + 1) % 4 == 0;
}

// ---------------------------------------------------------------- safetensors (mmap through libc)

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;

pub const Tensor = struct { name: []const u8, dtype: []const u8, shape: [4]u64, rank: usize, bytes: []const u8 };

pub const Checkpoint = struct {
    tensors: std.ArrayList(Tensor) = .empty,

    pub fn addShard(c: *Checkpoint, path: [:0]const u8) !void {
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        defer _ = close(fd);
        const size: usize = @intCast(lseek(fd, 0, 2));
        const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed; // PROT_READ, MAP_PRIVATE
        const bytes: [*]const u8 = @ptrCast(base);
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        const header = bytes[8 .. 8 + n];
        const data = bytes[8 + n .. size];
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = e.value_ptr.object;
            var t: Tensor = .{ .name = e.key_ptr.*, .dtype = o.get("dtype").?.string, .shape = @splat(1), .rank = 0, .bytes = &.{} };
            for (o.get("shape").?.array.items, 0..) |d, i| t.shape[i] = @intCast(d.integer);
            t.rank = o.get("shape").?.array.items.len;
            const span = o.get("data_offsets").?.array.items;
            t.bytes = data[@intCast(span[0].integer)..@intCast(span[1].integer)];
            try c.tensors.append(alloc, t);
        }
    }

    pub fn get(c: *const Checkpoint, name: []const u8) !Tensor {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name) and
            (t.name.len == name.len or std.mem.startsWith(u8, t.name, "language_model."))) return t;
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    }
};

// ---------------------------------------------------------------- device

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    staging: []u8 = &.{}, // USM host memory: the driver cannot copy from file-backed mmap pages directly
    k: struct {
        embed: ze.H, add_rmsnorm: ze.H, gdn_pre: ze.H, gdn_pre_rows: ze.H, gated_norm: ze.H, attn_prep: ze.H, attn_gate: ze.H,
        swiglu: ze.H, argmax: ze.H, argmax_part: ze.H, tap: ze.H, conv_commit: ze.H, repack: ze.H, gdn_step: ze.H, attn_decode: ze.H, attn_merge: ze.H,
        dec_wide: ze.H, dec_wide64: ze.H, dec_narrow: ze.H, gdn_prep: ze.H, gdn_state: ze.H, gdn_out: ze.H, ngen_wide: ze.H,
        ngen_narrow: ze.H, prompt_attn: ze.H,
    },

    pub fn kernel(d: *Dev, m: ze.H, name: [*:0]const u8, local: u32) !ze.H {
        const k = try d.g.kernel(m, name);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, local, 1, 1));
        return k;
    }

    pub fn alloc_(d: *Dev, bytes: usize) !*anyopaque {
        return d.g.deviceAlloc(@max(bytes, 64));
    }

    pub fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.alloc_(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    /// Host bytes (any memory, mmap included) to the device, through the USM staging buffer.
    pub fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
        if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(d.staging.len, bytes.len - done);
            @memcpy(d.staging[0..n], bytes[done .. done + n]);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(dst, done), d.staging.ptr, n, null, 0, null));
            try d.sync();
            done += n;
        }
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        var ev: ze.H = null;
        if (prof.on) {
            if (prof.used == prof.events.len) try d.sync();
            ev = prof.events[prof.used];
            prof.kernel[prof.used] = k;
            prof.used += 1;
        }
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, ev, 0, null));
    }

    pub fn sync(d: *Dev) !void {
        try ze.check(d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64)));
        if (prof.on) {
            for (prof.events[0..prof.used], prof.kernel[0..prof.used]) |e, k| {
                const t = try d.g.seconds(e);
                try ze.check(d.g.api.zeEventHostReset(e));
                for (&prof.names, &prof.total) |n, *tot| {
                    if (n.k == k) {
                        tot.* += t;
                        break;
                    }
                }
            }
            prof.used = 0;
        }
    }

    pub fn fill(d: *Dev, p: *anyopaque, pattern: anytype, bytes: usize) !void {
        const T = @TypeOf(pattern);
        const host = try alloc.alloc(T, bytes / @sizeOf(T));
        defer alloc.free(host);
        @memset(host, pattern);
        try d.copyIn(p, std.mem.sliceAsBytes(host));
    }
};

pub fn copy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, n, null, 0, null));
}

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

/// XPU_PROFILE=1: GPU time per kernel, summed over the run.
const Prof = struct {
    on: bool = false,
    events: [4096]ze.H = @splat(null),
    kernel: [4096]ze.H = @splat(null),
    used: usize = 0,
    names: [25]struct { k: ze.H = null, name: []const u8 = "" } = @splat(.{}),
    total: [25]f64 = @splat(0),
};
pub var prof: Prof = .{};

pub fn profReport(label: []const u8) void {
    if (!prof.on) return;
    var sum: f64 = 0;
    for (prof.total) |t| sum += t;
    std.debug.print("GPU time, {s}: {d:.1} ms\n", .{ label, sum * 1e3 });
    for (prof.names, &prof.total) |n, *t| {
        if (t.* > 0) std.debug.print("  {s:<14} {d:8.2} ms {d:5.1} %\n", .{ n.name, t.* * 1e3, t.* / sum * 100 });
        t.* = 0;
    }
}

/// An int4 projection in the GEMM layout: words [K/8, N] (nibble pairs interleaved), scales and biases × 64 [K/64, N]
/// fp16. `wide`: no K slices (family dpb), else 4 slices (family s4dpb, narrow shapes).
pub const Proj = struct { w: *anyopaque, s: *anyopaque, bg: *anyopaque, n: u32, k: u32, wide: bool };

const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // gdn.in or attn.qkv
    mixer_out: Proj, // out_proj or o_proj
    gu: Proj,
    down: Proj,
    // GDN
    conv_w: ?*anyopaque = null,
    alog: ?*anyopaque = null,
    dt: ?*anyopaque = null,
    gnorm: ?*anyopaque = null,
    rec: ?*anyopaque = null, // fp32 [48, 128, 128]
    conv: [2]?*anyopaque = .{ null, null }, // fp16 [3, 10240], ping-pong
    conv_cur: u1 = 0,
    // a verified block's GDN inputs and its entry state, to replay the kept rows (Runner.commit)
    rec_save: ?*anyopaque = null,
    stash: ?*anyopaque = null, // q, k [16, 16, 128] bf16 | v [16, 48, 128] bf16 | g, β [16, 48] fp32 | qkv rows [16, 16 640] fp16
    // attention
    qn: ?*anyopaque = null,
    kn: ?*anyopaque = null,
    kc: ?*anyopaque = null, // bf16 [MAX_CTX, 4, 256]
    vc: ?*anyopaque = null,
};

const Model = struct {
    embed_w: *anyopaque,
    embed_s: *anyopaque,
    embed_b: *anyopaque,
    layers: [LAYERS]Layer,
    norm: *anyopaque,
    head: Proj,
    invf: *anyopaque,
};

fn bf16Tensor(d: *Dev, c: *const Checkpoint, name: []const u8) !*anyopaque {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
    return d.upload(t.bytes);
}

/// Fused projection from MLX parts stacked on N, repacked on the GPU through a raw staging area.
pub fn projection(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        n_total += @intCast(w.shape[0]);
        k = @intCast(w.shape[1] * 8);
    }
    // N padded to the GEMM tile: the padding columns have zero scales and biases, so they come out as zeros.
    const n_real = n_total;
    n_total = (n_total + 255) / 256 * 256;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / GS) * n_total * 2), .bg = try d.alloc_(@as(usize, k / GS) * n_total * 2), .n = n_total, .k = k, .wide = wide };
    if (n_real != n_total) {
        try d.fill(out.s, @as(u16, 0), @as(usize, k / GS) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / GS) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        const s = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        const b = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.biases", .{ prefix, p }));
        const n: u32 = @intCast(w.shape[0]);
        const sw = stage;
        const ss = off(stage, w.bytes.len);
        const sb = off(ss, s.bytes.len);
        try d.copyIn(sw, w.bytes);
        try d.copyIn(ss, s.bytes);
        try d.copyIn(sb, b.bytes);
        try d.launch(d.k.repack, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ sw, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)) });
        try d.sync();
        n0 += n;
    }
    return out;
}

fn load(d: *Dev, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e_w = try c.get("model.embed_tokens.weight");
    m.embed_w = try d.upload(e_w.bytes);
    m.embed_s = try bf16Tensor(d, c, "model.embed_tokens.scales");
    m.embed_b = try bf16Tensor(d, c, "model.embed_tokens.biases");
    m.norm = try bf16Tensor(d, c, "model.norm.weight");
    m.head = try projection(d, c, "lm_head", &.{""}, true, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, s }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try bf16Tensor(d, c, f(p, "input_layernorm.weight")), .post_norm = try bf16Tensor(d, c, f(p, "post_attention_layernorm.weight")), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projection(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, stage), .down = try projection(d, c, f(p, "mlp."), &.{"down_proj"}, false, stage) };
        if (l.attn) {
            l.mixer_in = try projection(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, stage);
            l.mixer_out = try projection(d, c, f(p, "self_attn."), &.{"o_proj"}, false, stage);
            l.qn = try bf16Tensor(d, c, f(p, "self_attn.q_norm.weight"));
            l.kn = try bf16Tensor(d, c, f(p, "self_attn.k_norm.weight"));
            l.kc = try d.alloc_(MAX_CTX * 4 * 256 * 2);
            l.vc = try d.alloc_(MAX_CTX * 4 * 256 * 2);
        } else {
            l.mixer_in = try projection(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b" }, true, stage);
            l.mixer_out = try projection(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, stage);
            l.conv_w = try bf16Tensor(d, c, f(p, "linear_attn.conv1d.weight"));
            l.alog = try bf16Tensor(d, c, f(p, "linear_attn.A_log"));
            l.dt = try bf16Tensor(d, c, f(p, "linear_attn.dt_bias"));
            l.gnorm = try bf16Tensor(d, c, f(p, "linear_attn.norm.weight"));
            l.rec = try d.alloc_(48 * 128 * 128 * 4);
            l.rec_save = try d.alloc_(48 * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    return m;
}

// ---------------------------------------------------------------- activations and the forward

const Acts = struct {
    x: *anyopaque, // fp32 [CHUNK, D]
    h: *anyopaque, // fp16 [CHUNK, FF]
    ahi: *anyopaque, // fp16 [CHUNK, FF / 64]
    alo: *anyopaque,
    r: *anyopaque, // fp16 [CHUNK, D]
    p: *anyopaque, // fp16 [CHUNK, 2 FF]
    q: *anyopaque, // bf16 [CHUNK, 16, 128] (GDN) / [CHUNK, 24, 256] (attention)
    k: *anyopaque, // bf16 [CHUNK, 16, 128]
    v: *anyopaque, // bf16 [CHUNK, 48, 128]
    g: *anyopaque, // fp32 [CHUNK, 48]
    beta: *anyopaque,
    y: *anyopaque, // bf16 [CHUNK, 6144] (GDN out, attention out)
    la: *anyopaque,
    wb: *anyopaque,
    u0: *anyopaque,
    hs: *anyopaque,
    vn: *anyopaque,
    pm: *anyopaque,
    pl: *anyopaque,
    po: *anyopaque,
    pscr: *anyopaque, // fp32 [CHUNK, D] (slice totals of the narrow prompt GEMM)
    logits: *anyopaque, // fp16 [16, VOCAB]
    taps: *anyopaque, // bf16 [CHUNK, 5 D]: hidden states after TAP_LAYERS, when Mode.taps
    tok: *anyopaque, // i32
    pv: *anyopaque, // argmax partials: f32 [256], i32 [256]
    pi: *anyopaque,
    ids: *anyopaque, // i32 [CHUNK]
    zeros: *anyopaque, // padding rows of the GDN chunks: k = v = 0, β = 0 …
    ones: *anyopaque, // … and g = 1

    fn init(d: *Dev) !Acts {
        const nc = CHUNK / 64;
        const ns = MAX_CTX / SPLIT;
        return .{
            .x = try d.alloc_(CHUNK * D * 4), .h = try d.alloc_(CHUNK * FF * 2), .ahi = try d.alloc_(CHUNK * (FF / GS) * 2),
            .alo = try d.alloc_(CHUNK * (FF / GS) * 2), .r = try d.alloc_(CHUNK * D * 2), .p = try d.alloc_(CHUNK * 2 * FF * 2),
            .q = try d.alloc_(CHUNK * 24 * 256 * 2), .k = try d.alloc_(CHUNK * 16 * 128 * 2), .v = try d.alloc_(CHUNK * 48 * 128 * 2),
            .g = try d.alloc_(CHUNK * 48 * 4), .beta = try d.alloc_(CHUNK * 48 * 4), .y = try d.alloc_(CHUNK * 6144 * 2),
            .la = try d.alloc_(48 * nc * 64 * 4), .wb = try d.alloc_(nc * 48 * 64 * 128 * 2), .u0 = try d.alloc_(nc * 48 * 64 * 128 * 4),
            .hs = try d.alloc_(nc * 48 * 128 * 128 * 2), .vn = try d.alloc_(nc * 48 * 128 * 64 * 2),
            // decode attention partials: at most 16 rows (the prompt kernel needs none)
            .pm = try d.alloc_(16 * 24 * ns * 4), .pl = try d.alloc_(16 * 24 * ns * 4), .po = try d.alloc_(16 * 24 * ns * 256 * 4),
            .pscr = try d.alloc_(CHUNK * D * 4), .logits = try d.alloc_(16 * VOCAB * 2), .tok = try d.alloc_(16 * 4), .pv = try d.alloc_(16 * 1024), .pi = try d.alloc_(16 * 1024), .taps = try d.alloc_(CHUNK * 5 * D * 2), .ids = try d.alloc_(CHUNK * 4),
            .zeros = try d.alloc_(CHUNK * 48 * 128 * 2), .ones = try d.alloc_(CHUNK * 48 * 4),
        };
    }
};

/// out[M, N] fp16 = h[M, K] · Wᵀ : decode kernels up to 16 rows, the nGEN prompt GEMMs above.
pub fn gemm(d: *Dev, pscr: *anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
    const M: i32 = @intCast(m);
    const N: i32 = @intCast(w.n);
    const K: i32 = @intCast(w.k);
    const G: i32 = @intCast(w.k / GS);
    const mp: i32 = @intCast((m + 7) / 8 * 8);
    if (m <= 16) {
        if (w.wide)
            // 64 columns per subgroup for the widest shapes (gate|up, head), 32 below (measured end to end)
            if (w.n >= 30000)
                try d.launch(d.k.dec_wide64, .{ .x = 1, .y = w.n / 256 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else
                try d.launch(d.k.dec_wide, .{ .x = 1, .y = w.n / 128 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(d.k.dec_narrow, .{ .x = 1, .y = w.n / 32 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
    } else {
        const tiles = (m + 63) / 64;
        const groups: ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
        if (w.wide)
            try d.launch(d.k.ngen_wide, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(d.k.ngen_narrow, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G, pscr });
    }
}

/// XPU_DUMP=DIR: the first forward writes DIR/xpu_x_embed.f32 and DIR/xpu_x_{i}.f32 (x + pending residual, fp32
/// [T, D]) after each of the first four layers, to compare with tools/xpu-bench/integ/layers_ref.py.
pub var dump_dir: ?[]const u8 = null;
pub var verbose = false;

extern "c" fn creat(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

fn dump(d: *Dev, a: *Acts, T: u32, name: []const u8, with_r: bool) !void {
    const dir = dump_dir orelse return;
    try d.sync();
    const x = try alloc.alloc(f32, T * D);
    defer alloc.free(x);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, x.ptr, a.x, T * D * 4, null, 0, null));
    if (with_r) {
        const r = try alloc.alloc(f16, T * D);
        defer alloc.free(r);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.ptr, a.r, T * D * 2, null, 0, null));
        try d.sync();
        for (x, r) |*xv, rv| xv.* += @floatCast(rv);
    } else try d.sync();
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_x_{s}.f32", .{ dir, name }, 0);
    const fd = creat(path.ptr, 0o644);
    _ = write(fd, @ptrCast(x.ptr), x.len * 4);
    _ = close(fd);
}

/// A device buffer of `n` values (2 bytes: fp16 or bf16, 4: fp32) as fp32 in DIR/xpu_{name}.f32.
fn dumpBuf(d: *Dev, name: []const u8, p: *anyopaque, n: usize, kind: enum { f16, bf16, f32 }) !void {
    const dir = dump_dir orelse return;
    try d.sync();
    const x = try alloc.alloc(f32, n);
    defer alloc.free(x);
    const raw = try alloc.alloc(u16, n * 2);
    defer alloc.free(raw);
    const bytes: usize = if (kind == .f32) n * 4 else n * 2;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, raw.ptr, p, bytes, null, 0, null));
    try d.sync();
    for (x, 0..) |*v, i| v.* = switch (kind) {
        .f16 => @floatCast(@as(f16, @bitCast(raw[i]))),
        .bf16 => @bitCast(@as(u32, raw[i]) << 16),
        .f32 => @bitCast(@as(u32, raw[2 * i]) | (@as(u32, raw[2 * i + 1]) << 16)),
    };
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_{s}.f32", .{ dir, name }, 0);
    const fd = creat(path.ptr, 0o644);
    _ = write(fd, @ptrCast(x.ptr), x.len * 4);
    _ = close(fd);
}

/// What a forward leaves: nothing, the last row's logits and greedy token, or every row's (verification, ≤ 16 rows).
pub const Want = enum { none, last, rows };
pub const Mode = struct { want: Want = .last, stash: bool = false, taps: bool = false };

/// One forward over `ids` (≤ CHUNK rows) at positions p0..; the greedy token of the last row (or rows: a.tok).
/// Up to 16 rows the GDN runs serially (gdn_step), as decode does, so a verified row equals the serial one.
fn forward(d: *Dev, m: *Model, a: *Acts, ids: []const i32, p0: u32, mode: Mode) !?i32 {
    const want = mode.want != .none;
    const T: u32 = @intCast(ids.len);
    const t32: i32 = @intCast(T);
    const tp = (T + 63) / 64 * 64; // GDN chunk rows (padded)
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
    try d.launch(d.k.embed, .{ .x = T }, .{ a.ids, m.embed_w, m.embed_s, m.embed_b, a.x, @as(i32, D) });
    if (p0 == 0) try dump(d, a, T, "embed", false);
    for (&m.layers, 0..) |*l, i| {
        try d.launch(d.k.add_rmsnorm, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, if (i == 0) 0 else 1) });
        const probe = p0 == 0 and i == 0;
        if (probe) try dumpBuf(d, "h0", a.h, T * D, .f16);
        try gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_in, a.p, T);
        if (probe) try dumpBuf(d, "p0", a.p, T * GDN_IN, .f16);
        if (l.attn) {
            const ctx = p0 + T;
            const ns = (ctx + SPLIT - 1) / SPLIT;
            try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, @as(i32, @intCast(p0)), EPS });
            if (T > 16) {                                // the port's Triton prompt kernel: 256 rows a program
                const nul: ?*anyopaque = null;
                try d.launch(d.k.prompt_attn, .{ .x = (T + 255) / 256, .y = 24 }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(i32, @intCast(ctx)), nul, nul });
            } else {
                try d.launch(d.k.attn_decode, .{ .x = ns, .y = 4, .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), @as(f32, 1.0 / 16.0) });
                try d.launch(d.k.attn_merge, .{ .x = 24, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)) });
            }
            try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo });
        } else {
            const cin = l.conv[l.conv_cur].?;
            const cout = l.conv[1 - l.conv_cur].?;
            l.conv_cur = 1 - l.conv_cur;
            if (T > 16)
                try d.launch(d.k.gdn_pre_rows, .{ .x = T }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32 })
            else
                try d.launch(d.k.gdn_pre, .{ .x = T, .y = 40 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32 });
            if (T > 16 and tp > T) {                     // neutral padding rows: the state does not move
                const pad = tp - T;
                const cp = struct {
                    fn f(dd: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
                        try ze.check(dd.g.api.zeCommandListAppendMemoryCopy(dd.list, dst, src, n, null, 0, null));
                    }
                }.f;
                try cp(d, off(a.k, T * 16 * 128 * 2), a.zeros, pad * 16 * 128 * 2);
                try cp(d, off(a.v, T * 48 * 128 * 2), a.zeros, pad * 48 * 128 * 2);
                try cp(d, off(a.beta, T * 48 * 4), a.zeros, pad * 48 * 4);
                try cp(d, off(a.g, T * 48 * 4), a.ones, pad * 48 * 4);
            }
            if (T <= 16) {
                if (mode.stash) {                        // entry state and inputs, for Runner.commit's replay
                    const st = l.stash.?;
                    try copy(d, l.rec_save.?, l.rec.?, 48 * 128 * 128 * 4);
                    try copy(d, off(st, ST_Q), a.q, T * 16 * 128 * 2);
                    try copy(d, off(st, ST_K), a.k, T * 16 * 128 * 2);
                    try copy(d, off(st, ST_V), a.v, T * 48 * 128 * 2);
                    try copy(d, off(st, ST_G), a.g, T * 48 * 4);
                    try copy(d, off(st, ST_B), a.beta, T * 48 * 4);
                    try copy(d, off(st, ST_P), a.p, T * GDN_IN * 2);
                }
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = 48 }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, a.y, t32 });
            } else {
                const nc: i32 = @intCast(tp / 64);
                const ncu: u32 = tp / 64;
                const hk: i32 = 16;
                const hv: i32 = 48;
                try d.launch(d.k.gdn_prep, .{ .x = ncu, .y = 48 }, .{ a.k, a.v, a.g, a.beta, a.la, a.wb, a.u0, hk, hv, nc, @as(i32, 0) });
                try d.launch(d.k.gdn_state, .{ .x = 48 }, .{ a.k, a.wb, a.u0, a.la, l.rec.?, l.rec.?, a.hs, a.vn, hk, hv, nc });
                try d.launch(d.k.gdn_out, .{ .x = ncu, .y = 48 }, .{ a.q, a.k, a.la, a.hs, a.vn, a.y, hk, hv, nc });
            }
            if (probe) {
                try dumpBuf(d, "q0", a.q, T * 2048, .bf16);
                try dumpBuf(d, "k0", a.k, T * 2048, .bf16);
                try dumpBuf(d, "v0", a.v, T * 6144, .bf16);
                try dumpBuf(d, "g0", a.g, T * 48, .f32);
                try dumpBuf(d, "beta0", a.beta, T * 48, .f32);
                try dumpBuf(d, "y0", a.y, T * 6144, .bf16);
            }
            try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, a.p, l.gnorm.?, a.h, a.ahi, a.alo, EPS });
            if (probe) try dumpBuf(d, "o0", a.h, T * 6144, .f16);
        }
        try gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_out, a.r, T);
        if (probe) try dumpBuf(d, "r0", a.r, T * D, .f16);
        try d.launch(d.k.add_rmsnorm, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1) });
        try gemm(d, a.pscr, a.h, a.ahi, a.alo, l.gu, a.p, T);
        try d.launch(d.k.swiglu, .{ .x = T, .y = 17 }, .{ a.p, a.h, a.ahi, a.alo });
        try gemm(d, a.pscr, a.h, a.ahi, a.alo, l.down, a.r, T);
        if (p0 == 0 and i < 4) try dump(d, a, T, try std.fmt.allocPrint(alloc, "{d}", .{i}), true);
        if (mode.taps) for (TAP_LAYERS, 0..) |tl, slot| if (tl == i)
            try d.launch(d.k.tap, .{ .x = D / 256, .y = T }, .{ a.x, a.r, a.taps, @as(i32, D), @as(i32, @intCast(slot)) });
    }
    if (!want) {                                     // x is per token: nothing carries to the next chunk
        try d.sync();
        return null;
    }
    try d.launch(d.k.add_rmsnorm, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1) });
    const first = if (mode.want == .rows) 0 else T - 1;
    const rows = T - first;
    try gemm(d, a.pscr, off(a.h, first * D * 2), off(a.ahi, first * (D / GS) * 2), off(a.alo, first * (D / GS) * 2), m.head, a.logits, rows);
    try d.launch(d.k.argmax_part, .{ .x = 256, .y = rows }, .{ a.logits, @as(i32, VOCAB), a.pv, a.pi });
    try d.launch(d.k.argmax, .{ .x = rows }, .{ a.pv, a.pi, a.tok });
    var toks: [16]i32 = undefined;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, a.tok, rows * 4, null, 0, null));
    try d.sync();
    return toks[rows - 1];
}

extern "c" fn clock_gettime(clk: c_int, ts: *[2]i64) c_int;
pub fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

/// The whole model on the device and one sequence's state: what a serial host drives.
pub const Runner = struct {
    d: Dev,
    model: Model,
    acts: Acts,
    pos: u32 = 0,
    last: u32 = 0, // greedy next token of the last forward
    context: u32,
    ckpt: Checkpoint = .{},
    host_logits: []f32 = &.{},
    block: [16]u32 = @splat(0), // the last verified block's greedy tokens
    block_rows: u32 = 0,
    /// Target hidden states for a drafter: when on, every forward fills acts.taps and calls `on_taps` with its rows.
    taps_on: bool = false,
    on_taps: ?struct { ctx: *anyopaque, f: *const fn (*anyopaque, *anyopaque, u32) anyerror!void } = null,

    /// Loads `dir` (MLX affine 4/64) onto the first B70 with room for `context` positions.
    pub fn init(dir: []const u8, context: u32) !*Runner {
        if (std.c.getenv("XPU_DUMP")) |v| dump_dir = std.mem.span(v);
        if (std.c.getenv("XPU_CHUNK")) |v| CHUNK = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        MAX_CTX = context;
        var ckpt: Checkpoint = .{};
        {
            const index_path = try std.fmt.allocPrintSentinel(alloc, "{s}/model.safetensors.index.json", .{dir}, 0);
            const fd = open(index_path.ptr, 0);
            if (fd < 0) return error.FileNotFound;
            const size: usize = @intCast(lseek(fd, 0, 2));
            const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
            const parsed = try std.json.parseFromSlice(std.json.Value, alloc, text[0..size], .{});
            var seen: std.ArrayList([]const u8) = .empty;
            var it = parsed.value.object.get("weight_map").?.object.iterator();
            while (it.next()) |e| {
                const f = e.value_ptr.string;
                for (seen.items) |s| {
                    if (std.mem.eql(u8, s, f)) break;
                } else {
                    try seen.append(alloc, f);
                    try ckpt.addShard(try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, f }, 0));
                }
            }
        }
        const r = try alloc.create(Runner);
        r.* = .{ .d = .{ .g = try ze.Gpu.open(0xe223), .list = undefined, .k = undefined }, .model = undefined, .acts = undefined, .context = context, .ckpt = ckpt };
        const d = &r.d;
        d.list = try d.g.immediate(0, 0);
        const mg = try d.g.module(&kb.glue);
        const md = try d.g.module(&kb.decode);
        const mm = try d.g.module(&kb.gemm);
        const mgd = try d.g.module(&kb.gdn);
        const mw = try d.g.module(&kb.ngen_wide);
        const mn = try d.g.module(&kb.ngen_narrow);
        const mpa = try d.g.module(&kb.prompt_attention);
        d.k = .{
            .embed = try d.kernel(mg, "embed", 256), .add_rmsnorm = try d.kernel(mg, "add_rmsnorm", 512),
            .gdn_pre = try d.kernel(mg, "gdn_pre", 256), .gdn_pre_rows = try d.kernel(mg, "gdn_pre_rows", 256), .gated_norm = try d.kernel(mg, "gated_norm", 256),
            .attn_prep = try d.kernel(mg, "attn_prep", 256), .attn_gate = try d.kernel(mg, "attn_gate", 256),
            .swiglu = try d.kernel(mg, "swiglu", 256), .argmax = try d.kernel(mg, "argmax", 256), .argmax_part = try d.kernel(mg, "argmax_part", 256), .tap = try d.kernel(mg, "tap", 256), .conv_commit = try d.kernel(mg, "conv_commit", 256),
            .repack = try d.kernel(mg, "repack", 16), .gdn_step = try d.kernel(md, "gdn_step", 16),
            .attn_decode = try d.kernel(md, "attn_decode", 16), .attn_merge = try d.kernel(md, "attn_merge", 16),
            .dec_wide = try d.kernel(mm, "w4k1_decode", 64), .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64), .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
            .gdn_prep = try d.kernel(mgd, "gdn_prep", 64), .gdn_state = try d.kernel(mgd, "gdn_state", 256),
            .gdn_out = try d.kernel(mgd, "gdn_out", 128), .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
            .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256), .prompt_attn = try d.kernel(mpa, "_prompt", 512),
        };
        if (std.c.getenv("XPU_PROFILE") != null) {
            prof.on = true;
            prof.events = try d.g.events(4096);
            inline for (@typeInfo(@TypeOf(d.k)).@"struct".field_names, 0..) |name, i| prof.names[i] = .{ .k = @field(d.k, name), .name = name };
        }
        const prof_on = prof.on;
        prof.on = false; // not during the load
        r.model = try load(d, &ckpt);
        prof.on = prof_on;
        r.acts = try Acts.init(d);
        try d.fill(r.acts.zeros, @as(u16, 0), CHUNK * 48 * 128 * 2);
        try d.fill(r.acts.ones, @as(f32, 1.0), CHUNK * 48 * 4);
        r.host_logits = try alloc.alloc(f32, VOCAB);
        return r;
    }

    /// A fresh sequence: GDN and convolution states to zero (the KV cache is overwritten position by position).
    pub fn reset(r: *Runner) !void {
        for (&r.model.layers) |*l| if (!l.attn) {
            try r.d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (l.conv) |cv| try r.d.fill(cv.?, @as(u16, 0), 3 * 10240 * 2);
            l.conv_cur = 0;
        };
        r.pos = 0;
    }

    /// Feeds prompt tokens in chunks; `last`: the final chunk, whose last row's logits are kept.
    pub fn prompt(r: *Runner, ids: []const u32, last: bool) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + CHUNK);
            const want = last and end == ids.len;
            if (r.pos + (end - at) > r.context) return error.ContextFull;
            if (try forward(&r.d, &r.model, &r.acts, @ptrCast(ids[at..end]), r.pos, .{ .want = if (want) .last else .none, .taps = r.taps_on })) |t| r.last = @intCast(t);
            if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, @intCast(end - at));
            r.pos += @intCast(end - at);
            at = end;
        }
    }

    /// One decoded token in; the next one's logits kept.
    pub fn step(r: *Runner, token: u32) !void {
        if (r.pos + 1 > r.context) return error.ContextFull;
        const id: i32 = @intCast(token);
        r.last = @intCast((try forward(&r.d, &r.model, &r.acts, &.{id}, r.pos, .{ .taps = r.taps_on })).?);
        r.pos += 1;
        if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, 1);
    }

    /// Verifies a block: `ids` (≤ 16 rows: the pending token and the drafts) run from `pos`, every row's greedy token
    /// returned (and its logits kept for sampling). The state moves by all rows until `commit` keeps a prefix.
    pub fn verify(r: *Runner, ids: []const u32) ![]const u32 {
        if (ids.len == 0 or ids.len > 16) return error.BadBlock;
        if (r.pos + ids.len > r.context) return error.ContextFull;
        _ = try forward(&r.d, &r.model, &r.acts, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on });
        var toks: [16]i32 = undefined;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, &toks, r.acts.tok, ids.len * 4, null, 0, null));
        try r.d.sync();
        for (r.block[0..ids.len], toks[0..ids.len]) |*o, t| o.* = @intCast(t);
        r.block_rows = @intCast(ids.len);
        return r.block[0..ids.len];
    }

    /// Keeps the first `keep` rows of the last verified block: the GDN and convolution states are rebuilt from the
    /// block's entry by replaying those rows (the attention caches need nothing: later positions overwrite them).
    pub fn commit(r: *Runner, keep: u32) !void {
        if (keep == 0 or keep > r.block_rows) return error.BadCommit;
        if (keep < r.block_rows) for (&r.model.layers) |*l| if (!l.attn) {
            const st = l.stash.?;
            try copy(&r.d, l.rec.?, l.rec_save.?, 48 * 128 * 128 * 4);
            try r.d.launch(r.d.k.gdn_step, .{ .x = 16, .y = 48 }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.rec.?, r.acts.y, @as(i32, @intCast(keep)) });
            // the block's forward flipped conv_cur: conv[cur] is the state after all rows, conv[1 - cur] the entry
            try r.d.launch(r.d.k.conv_commit, .{ .x = 40 }, .{ l.conv[1 - l.conv_cur].?, off(st, ST_P), l.conv[l.conv_cur].?, @as(i32, @intCast(keep)) });
        };
        r.pos += keep;
        r.last = r.block[keep - 1];
        r.block_rows = 0;
        try r.d.sync();
    }

    /// The logits of row `row` of the last verified block (or of the last forward's row, row 0), fp32 on the host.
    pub fn rowLogits(r: *Runner, row: u32) ![]const f32 {
        const raw = try alloc.alloc(f16, VOCAB);
        defer alloc.free(raw);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, off(r.acts.logits, row * VOCAB * 2), VOCAB * 2, null, 0, null));
        try r.d.sync();
        for (r.host_logits, raw) |*o, v| o.* = v;
        return r.host_logits;
    }

    /// The kept row's logits, fp32 on the host (for sampling).
    pub fn logits(r: *Runner) ![]const f32 {
        const raw = try alloc.alloc(f16, VOCAB);
        defer alloc.free(raw);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, r.acts.logits, VOCAB * 2, null, 0, null));
        try r.d.sync();
        for (r.host_logits, raw) |*o, v| o.* = v;
        return r.host_logits;
    }

    /// The device's name, as the server reports the chip.
    pub fn deviceName(r: *Runner) []const u8 {
        return std.mem.sliceTo(&r.d.g.props.name, 0);
    }
};
