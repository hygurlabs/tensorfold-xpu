//! Qwen3.8-27B forward on an Intel Xe2 GPU (Arc Pro B70) over Level Zero: MLX affine 4-bit g64 checkpoint, prompt in
//! chunks (XPU_CHUNK rows, 1 024 by default), serial decode. `Runner` is what the native serial host drives
//! (native/xpu.zig) and what the development runner (qwen27_main.zig) times.
const std = @import("std");
pub const ze = @import("ze.zig");
const kb = @import("xpu_qwen27_kernels");

pub const alloc = std.heap.c_allocator;

// Qwen3.8-27B shape (checked against config.json at load).
pub const D = 5120;
const FF = 17408;
const LAYERS = 64;
pub const VOCAB = 248320;
/// The layers' quantization group (64: MLX affine, 128: GPTQ), set at load; the head and the drafter keep 64.
pub var GS: u32 = 64;
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

pub const Tensor = struct { name: []const u8, dtype: []const u8, shape: [8]u64, rank: usize, bytes: []const u8 };

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
            if (o.get("shape").?.array.items.len > t.shape.len) return error.TensorRank; // the vision tower's conv is 5-D
            for (o.get("shape").?.array.items, 0..) |d, i| t.shape[i] = @intCast(d.integer);
            t.rank = o.get("shape").?.array.items.len;
            const span = o.get("data_offsets").?.array.items;
            t.bytes = data[@intCast(span[0].integer)..@intCast(span[1].integer)];
            try c.tensors.append(alloc, t);
        }
    }

    pub fn get(c: *const Checkpoint, name: []const u8) !Tensor {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name) and
            (t.name.len == name.len or std.mem.startsWith(u8, t.name, "language_model.") or std.mem.startsWith(u8, t.name, "model.language_model."))) return t;
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    }
    pub fn has(c: *const Checkpoint, name: []const u8) bool {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name)) return true;
        return false;
    }
};

// ---------------------------------------------------------------- device

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    staging: []u8 = &.{}, // USM host memory: the driver cannot copy from file-backed mmap pages directly
    allocated: usize = 0, // device bytes allocated (frees not counted: buffers live as long as the runner)
    k: struct {
        embed: ze.H, add_rmsnorm_dec: ze.H, add_rmsnorm: ze.H, gdn_pre: ze.H, gdn_pre_rows: ze.H, gated_norm: ze.H, attn_prep: ze.H, attn_gate: ze.H,
        swiglu: ze.H, argmax: ze.H, argmax_part: ze.H, tap: ze.H, conv_commit: ze.H, repack: ze.H, gdn_step: ze.H, attn_decode: ze.H, attn_merge: ze.H, kv_move: ze.H, exl3_gemv: ze.H, rot_heads: ze.H, attn_prompt: ze.H, attn_prompt8: ze.H,
        embed_dense: ze.H, dense_rows: ze.H, repack_gptq: ze.H, quantize_mlx: ze.H,
        dec_wide_g128: ze.H, dec_wide64_g128: ze.H, dec_narrow_g128: ze.H, ngen_wide_g128: ze.H, ngen_narrow_g128: ze.H,
        ngen_swiglu: ze.H, ngen_swiglu_g128: ze.H, rowsums_ff: ze.H, fakequant_rows: ze.H,
        a8: ze.H, a8_swiglu: ze.H, quant_a8: ze.H,
        dec_wide: ze.H, dec_wide64: ze.H, dec_narrow: ze.H, gdn_prep: ze.H, gdn_state: ze.H, gdn_out: ze.H, ngen_wide: ze.H,
        ngen_narrow: ze.H, prompt_attn: ze.H,
    },

    pub fn kernel(d: *Dev, m: ze.H, name: [*:0]const u8, local: u32) !ze.H {
        const k = try d.g.kernel(m, name);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, local, 1, 1));
        return k;
    }

    pub fn alloc_(d: *Dev, bytes: usize) !*anyopaque {
        d.allocated += @max(bytes, 64);
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
    names: [64]struct { k: ze.H = null, name: []const u8 = "" } = @splat(.{}),
    total: [64]f64 = @splat(0),
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

/// An int4 projection in the GEMM layout: words [K/8, N] (nibble pairs interleaved), scales and biases × gs [K/gs, N]
/// fp16. `wide`: no K slices (family dpb), else 4 slices (family s4dpb, narrow shapes).
/// `ilv`: gate|up interleaved by blocks of 16 columns (the prompt GEMM's SwiGLU epilogue reads both halves of a column).
pub const Proj = struct { w: *anyopaque, s: *anyopaque, bg: *anyopaque, n: u32, k: u32, wide: bool, gs: u32 = 64, ilv: bool = false };

const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // gdn.in or attn.qkv
    mixer_out: Proj, // out_proj or o_proj
    gu: Proj,
    down: Proj,
    // GDN
    ab: ?*anyopaque = null, // GPTQ: the a and b gates, fp16 [96, 5120] (dense, as stored), into P columns 16 384 …
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
    // rec, conv, conv_cur, kc and vc above are the selected slot's (Runner.select); each slot's own are kept here
    slot_rec: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_save: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_conv: [MAX_SLOTS][2]?*anyopaque = @splat(.{ null, null }),
    slot_cur: [MAX_SLOTS]u1 = @splat(0),
    slot_kc: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_vc: [MAX_SLOTS]?*anyopaque = @splat(null),
};

/// Sequences a runner holds at most (one a stream of the lane host).
pub const MAX_SLOTS = 16;

pub const Model = struct {
    embed_w: *anyopaque, // MLX: words [V, D/8] ; GPTQ: the fp16 table [V, D]
    embed_s: ?*anyopaque = null, // MLX only
    embed_b: ?*anyopaque = null,
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
pub fn projection(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, ilv: bool, stage: *anyopaque) !Proj {
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
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = wide, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.s, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
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
        try d.launch(d.k.repack, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ sw, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// Fused projection from GPTQ symmetric 4-bit parts (qweight [K/8, N], scales fp16 [K/G, N]) stacked on N, at least
/// `min_n` columns (the extra ones zero).
fn projectionGptq(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, min_n: u32, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        n_total += @intCast(w.shape[1]);
        k = @intCast(w.shape[0] * 8);
    }
    const n_real = n_total;
    n_total = (@max(n_total, min_n) + 255) / 256 * 256;
    const g = GS;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / g) * n_total * 2), .bg = try d.alloc_(@as(usize, k / g) * n_total * 2), .n = n_total, .k = k, .wide = wide, .gs = g, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.w, @as(u32, 0), @as(usize, k / 8) * n_total * 4);
        try d.fill(out.s, @as(u16, 0), @as(usize, k / g) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / g) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        const sc = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        if (!std.mem.eql(u8, sc.dtype, "F16") or sc.shape[0] * g != k) return error.UnexpectedGptqLayout;
        const n: u32 = @intCast(w.shape[1]);
        const ss = off(stage, w.bytes.len);
        try d.copyIn(stage, w.bytes);
        try d.copyIn(ss, sc.bytes);
        try d.launch(d.k.repack_gptq, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ stage, ss, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, @intCast(g)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// An fp16 [N, K] weight quantized at load to MLX affine 4/64 (MLX's algorithm) in the GEMM layout, in row blocks
/// through the staging area (the GPTQ head: its error then matches the MLX checkpoint's own 4-bit head).
fn projectionQuantized(d: *Dev, c: *const Checkpoint, name: []const u8, wide: bool, stage: *anyopaque) !Proj {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n: u32 = @intCast(t.shape[0]);
    const k: u32 = @intCast(t.shape[1]);
    if (n % 256 != 0) return error.UnexpectedShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n * 4), .s = try d.alloc_(@as(usize, k / 64) * n * 2), .bg = try d.alloc_(@as(usize, k / 64) * n * 2), .n = n, .k = k, .wide = wide };
    const rows: u32 = 32768;
    var n0: u32 = 0;
    while (n0 < n) : (n0 += rows) {
        const r = @min(rows, n - n0);
        const raw = stage;
        const mw = off(raw, @as(usize, r) * k * 2);
        const ms = off(mw, @as(usize, r) * (k / 8) * 4);
        const mb = off(ms, @as(usize, r) * (k / 64) * 2);
        try d.copyIn(raw, t.bytes[@as(usize, n0) * k * 2 ..][0 .. @as(usize, r) * k * 2]);
        try d.launch(d.k.quantize_mlx, .{ .x = (r + 15) / 16, .y = k / 64 }, .{ raw, @as(i32, @intCast(r)), @as(i32, @intCast(k)), mw, ms, mb });
        try d.launch(d.k.repack, .{ .x = (r + 15) / 16, .y = k / 8 }, .{ mw, ms, mb, out.w, out.s, out.bg, @as(i32, @intCast(r)), @as(i32, @intCast(k)), @as(i32, @intCast(n)), @as(i32, @intCast(n0)), @as(i32, 0) });
        try d.sync();
    }
    return out;
}

/// A small weight to bf16 on the device: BF16 as stored (MLX), F16 converted (GPTQ); `centred`: a GPTQ RMSNorm weight
/// stored as gamma − 1, given its 1 back as the MLX converter writes it.
fn vecTensor(d: *Dev, c: *const Checkpoint, name: []const u8, centred: bool) !*anyopaque {
    const t = try c.get(name);
    if (std.mem.eql(u8, t.dtype, "BF16")) return d.upload(t.bytes);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n = t.bytes.len / 2;
    const out = try alloc.alloc(u16, n);
    defer alloc.free(out);
    for (out, 0..) |*o, i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
        var v: f32 = h;
        if (centred) v += 1.0;
        const u: u32 = @bitCast(v);
        o.* = @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    return d.upload(std.mem.sliceAsBytes(out));
}

/// Qwen3.8-27B from a symmetric 4-bit GPTQ checkpoint (groups of GS = 128): the layers' projections int4 g128, the
/// GDN a and b gates fp16 dense as stored, the embedding fp16 dense, the head quantized at load to affine 4/64.
fn loadGptq(d: *Dev, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e = try c.get("model.embed_tokens.weight");
    if (!std.mem.eql(u8, e.dtype, "F16")) return error.UnexpectedDtype;
    m.embed_w = try d.alloc_(e.bytes.len);
    try d.copyIn(m.embed_w, e.bytes);
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try vecTensor(d, c, "model.norm.weight", true);
    m.head = try projectionQuantized(d, c, "lm_head.weight", true, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, sx: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, sx }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try vecTensor(d, c, f(p, "input_layernorm.weight"), true), .post_norm = try vecTensor(d, c, f(p, "post_attention_layernorm.weight"), true), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projectionGptq(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, 0, true, stage), .down = try projectionGptq(d, c, f(p, "mlp."), &.{"down_proj"}, false, 0, false, stage) };
        if (l.attn) {
            l.mixer_in = try projectionGptq(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, 0, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "self_attn."), &.{"o_proj"}, false, 0, false, stage);
            l.qn = try vecTensor(d, c, f(p, "self_attn.q_norm.weight"), true);
            l.kn = try vecTensor(d, c, f(p, "self_attn.k_norm.weight"), true);
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projectionGptq(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z" }, true, GDN_IN, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, 0, false, stage);
            const ta = try c.get(f(p, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(p, "linear_attn.in_proj_b.weight"));
            if (!std.mem.eql(u8, ta.dtype, "F16") or ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) return error.UnexpectedShape;
            l.ab = try d.alloc_(96 * D * 2);
            try d.copyIn(l.ab.?, ta.bytes);
            try d.copyIn(off(l.ab.?, 48 * D * 2), tb.bytes);
            l.conv_w = try vecTensor(d, c, f(p, "linear_attn.conv1d.weight"), false);
            l.alog = try vecTensor(d, c, f(p, "linear_attn.A_log"), false);
            l.dt = try vecTensor(d, c, f(p, "linear_attn.dt_bias"), false);
            l.gnorm = try vecTensor(d, c, f(p, "linear_attn.norm.weight"), false);
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

fn load(d: *Dev, c: *const Checkpoint) !Model {
    if (c.has("in_proj_qkv.qweight")) {
        GS = 128;
        return loadGptq(d, c);
    }
    GS = 64;
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e_w = try c.get("model.embed_tokens.weight");
    m.embed_w = try d.upload(e_w.bytes);
    m.embed_s = try bf16Tensor(d, c, "model.embed_tokens.scales");
    m.embed_b = try bf16Tensor(d, c, "model.embed_tokens.biases");
    m.norm = try bf16Tensor(d, c, "model.norm.weight");
    m.head = try projection(d, c, "lm_head", &.{""}, true, false, stage);
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
        l.* = .{ .in_norm = try bf16Tensor(d, c, f(p, "input_layernorm.weight")), .post_norm = try bf16Tensor(d, c, f(p, "post_attention_layernorm.weight")), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projection(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, true, stage), .down = try projection(d, c, f(p, "mlp."), &.{"down_proj"}, false, false, stage) };
        if (l.attn) {
            l.mixer_in = try projection(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "self_attn."), &.{"o_proj"}, false, false, stage);
            l.qn = try bf16Tensor(d, c, f(p, "self_attn.q_norm.weight"));
            l.kn = try bf16Tensor(d, c, f(p, "self_attn.k_norm.weight"));
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projection(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, false, stage);
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

pub const Acts = struct {
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
    chain: *anyopaque, // i32 [16, 17]: the tree table of a chain (row t at depth t), for the decode attention
    tree: *anyopaque, // i32 [16, 17]: the verified block's tree table (decode.cl, DEPTH and ANC)
    path: *anyopaque, // i32 [16]: the rows of the kept path
    sl: *anyopaque, // fp32 [16, 48, 128, 128]: GDN state after each row of a verified tree
    mt: *anyopaque, // u64 [LAYERS, 16, 8]: stepMulti's per-row state addresses
    mpos: *anyopaque, // i32 [16]: and positions
    a8: *anyopaque, // W4A8 prompt: activations int8 [CHUNK, FF] (permuted by blocks of 8) …
    sa: *anyopaque, // … and their scales fp32 [FF / 128, CHUNK + 32]

    fn init(d: *Dev) !Acts {
        const nc = CHUNK / 64;
        const ns = MAX_CTX / SPLIT;
        return .{
            .x = try d.alloc_(CHUNK * D * 4), .h = try d.alloc_(CHUNK * FF * 2), .ahi = try d.alloc_(CHUNK * (FF / 64) * 2),
            .alo = try d.alloc_(CHUNK * (FF / 64) * 2), .r = try d.alloc_(CHUNK * D * 2), .p = try d.alloc_(CHUNK * 2 * FF * 2),
            .q = try d.alloc_(CHUNK * 24 * 256 * 2), .k = try d.alloc_(CHUNK * 16 * 128 * 2), .v = try d.alloc_(CHUNK * 48 * 128 * 2),
            .g = try d.alloc_(CHUNK * 48 * 4), .beta = try d.alloc_(CHUNK * 48 * 4), .y = try d.alloc_(CHUNK * 6144 * 2),
            .la = try d.alloc_(48 * nc * 64 * 4), .wb = try d.alloc_(nc * 48 * 64 * 128 * 2), .u0 = try d.alloc_(nc * 48 * 64 * 128 * 4),
            .hs = try d.alloc_(nc * 48 * 128 * 128 * 2), .vn = try d.alloc_(nc * 48 * 128 * 64 * 2),
            // decode attention partials: at most 16 rows (the prompt kernel needs none)
            .pm = try d.alloc_(16 * 24 * ns * 4), .pl = try d.alloc_(16 * 24 * ns * 4), .po = try d.alloc_(16 * 24 * ns * 256 * 4),
            .pscr = try d.alloc_(CHUNK * D * 4), .logits = try d.alloc_(16 * VOCAB * 2), .tok = try d.alloc_(16 * 4), .pv = try d.alloc_(16 * 1024), .pi = try d.alloc_(16 * 1024), .taps = try d.alloc_(CHUNK * 5 * D * 2), .ids = try d.alloc_(CHUNK * 4),
            .zeros = try d.alloc_(CHUNK * 48 * 128 * 2), .ones = try d.alloc_(CHUNK * 48 * 4),
            .chain = try d.upload(std.mem.sliceAsBytes(&chainTable())), .tree = try d.alloc_(16 * 17 * 4), .path = try d.alloc_(16 * 4),
            .sl = try d.alloc_(16 * 48 * 128 * 128 * 4),
            .mt = try d.alloc_(LAYERS * 16 * 8 * 8), .mpos = try d.alloc_(16 * 4),
            .a8 = try d.alloc_(CHUNK * FF), .sa = try d.alloc_((FF / 128) * (CHUNK + 32) * 4),
        };
    }
};

/// Row t at depth t, its ancestors rows 0 .. t: a chain's tree table.
fn chainTable() [16 * 17]i32 {
    var t: [16 * 17]i32 = @splat(0);
    for (0..16) |r| {
        t[r * 17] = @intCast(r);
        for (0..r + 1) |dd| t[r * 17 + 1 + dd] = @intCast(dd);
    }
    return t;
}

/// gate|up and SwiGLU: the activation (fp16 [T, 17 408]) and its row sums; returns the buffer holding it. Prompt rows:
/// the GEMM's SwiGLU epilogue writes it into P; decode rows: the GEMM into P, then `swiglu` into H.
fn mlpIn(d: *Dev, a: *Acts, w: Proj, T: u32, gs: i32, pre: bool) !*anyopaque {
    if (prefill_int8 and T > 16 and w.ilv and w.gs == 128) { // W4A8: SwiGLU epilogue; down quantizes its own input
        if (!pre) try quantA8(d, a, a.h, w.k, T);
        try d.launch(d.k.a8_swiglu, a8Groups(w, T), a8Args(a, w, a.p, T));
        return a.p;
    }
    if (T > 16 and w.ilv) {
        const M: i32 = @intCast(T);
        const tiles = (T + 63) / 64;
        const k = if (w.gs == 128) d.k.ngen_swiglu_g128 else d.k.ngen_swiglu;
        try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, .{ a.h, w.w, w.s, w.bg, a.p, M, @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast((T + 7) / 8 * 8)), @as(i32, @intCast(w.k / w.gs)) });
        try d.launch(d.k.rowsums_ff, .{ .x = T }, .{ a.p, a.ahi, a.alo, gs });
        return a.p;
    }
    try gemm(d, a.pscr, a.h, a.ahi, a.alo, w, a.p, T);
    try d.launch(d.k.swiglu, .{ .x = T, .y = 17 }, .{ a.p, a.h, a.ahi, a.alo, gs, @as(i32, @intFromBool(w.ilv)) });
    return a.h;
}

/// The rows' embeddings: x fp32 [T, D] (bf16 values) for the token ids at `ids` on the device.
pub fn embedRows(d: *Dev, m: *const Model, ids: *anyopaque, x: *anyopaque, rows: u32) !void {
    if (m.embed_s) |es|
        try d.launch(d.k.embed, .{ .x = rows }, .{ ids, m.embed_w, es, m.embed_b.?, x, @as(i32, D) })
    else
        try d.launch(d.k.embed_dense, .{ .x = rows }, .{ ids, m.embed_w, x, @as(i32, D) });
}

/// out[M, N] fp16 = h[M, K] · Wᵀ : decode kernels up to 16 rows, the nGEN prompt GEMMs above.
pub fn gemm(d: *Dev, pscr: *anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
    const M: i32 = @intCast(m);
    const N: i32 = @intCast(w.n);
    const K: i32 = @intCast(w.k);
    const G: i32 = @intCast(w.k / w.gs);
    const g128 = w.gs == 128;
    const k = d.k;
    const dec_wide = if (g128) k.dec_wide_g128 else k.dec_wide;
    const dec_wide64 = if (g128) k.dec_wide64_g128 else k.dec_wide64;
    const dec_narrow = if (g128) k.dec_narrow_g128 else k.dec_narrow;
    const ngen_wide = if (g128) k.ngen_wide_g128 else k.ngen_wide;
    const ngen_narrow = if (g128) k.ngen_narrow_g128 else k.ngen_narrow;
    const mp: i32 = @intCast((m + 7) / 8 * 8);
    if (m <= 16) {
        if (w.wide)
            // 64 columns per subgroup for the widest shapes (gate|up, head), 32 below (measured end to end)
            if (w.n >= 30000)
                try d.launch(dec_wide64, .{ .x = 1, .y = w.n / 256 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
            else
                try d.launch(dec_wide, .{ .x = 1, .y = w.n / 128 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(dec_narrow, .{ .x = 1, .y = w.n / 32 }, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G });
    } else {
        const tiles = (m + 63) / 64;
        const groups: ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
        if (w.wide)
            try d.launch(ngen_wide, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G })
        else
            try d.launch(ngen_narrow, groups, .{ h, w.w, w.s, w.bg, out, M, N, K, ahi, alo, mp, G, pscr });
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
var dump_h: ?c_int = null;
/// W4A8 study (XPU_A8 = 128: int8 activations by groups of 128; "token": one scale a row): prompt GEMM inputs quantized
/// and dequantized in place before each projection. 0: off.
var a8_group: u32 = 0;
/// K/V cache study (XPU_KV8): 1 q, k, v rotated by Hadamard (the output rotated back), 2 k, v quantized to int8 by
/// (token, head) and back, 3 both.
var kv8: i32 = 0;
/// The prompt attention: our DPAS kernel (attn_prompt), or the port's Triton one with XPU_ATTN_TRITON=1.
var attn_own = true;
/// The K/V cache in int8 (XPU_KV_INT8=1): per position and K/V head, 256 int8 then, after all MAX_CTX positions, a
/// float scale (max |x| / 127); half the bf16 cache's bytes. The kernels get MAX_CTX as `cap` (0: bf16).
var kv_i8 = false;
fn kvBytes() usize {
    return @as(usize, MAX_CTX) * 4 * @as(usize, if (kv_i8) 256 + 4 else 256 * 2);
}
fn kvCap() i32 {
    return if (kv_i8) @intCast(MAX_CTX) else 0;
}
/// W4A8 prompt GEMMs (XPU_PREFILL_INT8, local option, GPTQ g128 only): int8 activations by (row, group of 128).
pub var prefill_int8 = false;
/// add_rmsnorm without the fused W4A8 quantization (its A8 and SA arguments).
pub const no_q8: ?*anyopaque = null;

/// Prompt GEMM: W4A8 when the option is on and the rows are prompt rows, else the fp16 GEMM.
/// `pre`: x is already quantized into acts.a8 (by the norm that produced it).
fn pgemm(d: *Dev, a: *Acts, x: *anyopaque, w: Proj, out: *anyopaque, T: u32, pre: bool) !void {
    if (!prefill_int8 or T <= 16 or w.gs != 128) return gemm(d, a.pscr, x, a.ahi, a.alo, w, out, T);
    if (!pre) try quantA8(d, a, x, w.k, T);
    try d.launch(d.k.a8, a8Groups(w, T), a8Args(a, w, out, T));
}
fn quantA8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32) !void {
    try d.launch(d.k.quant_a8, .{ .x = T }, .{ x, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(CHUNK + 32)) });
}
fn a8Groups(w: Proj, T: u32) ze.GroupCount {
    const tiles = (T + 63) / 64;
    return .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
}
fn a8Args(a: *Acts, w: Proj, out: *anyopaque, T: u32) struct { *anyopaque, *anyopaque, *anyopaque, *anyopaque, *anyopaque, i32, i32, i32, *anyopaque, *anyopaque, i32, i32, *anyopaque } {
    return .{ a.a8, w.w, w.s, w.bg, out, @intCast(T), @intCast(w.n), @intCast(w.k), a.ahi, a.alo, @intCast(CHUNK + 32), @intCast(w.k / 128), a.sa };
}
var a8_sites: u32 = 15; // XPU_A8_SITES: 1 mixer in, 2 mixer out, 4 gate|up, 8 down

fn a8(d: *Dev, a: *Acts, x: *anyopaque, K: u32, T: u32, site: u32) !void {
    if (a8_group == 0 or T <= 16 or a8_sites & site == 0) return;
    const qg = if (a8_group == std.math.maxInt(u32)) K else a8_group;
    try d.launch(d.k.fakequant_rows, .{ .x = T }, .{ x, a.ahi, a.alo, @as(i32, @intCast(K)), @as(i32, @intCast(qg)), @as(i32, @intCast(GS)) });
}
/// tree: the rows form a tree (acts.tree), each seeing its ancestors only; their GDN states land in acts.sl.
/// multi: each row a different slot's next token (Runner.stepMulti): `mt` its states' addresses by layer, `pos` its
/// position.
pub const Mode = struct { want: Want = .last, stash: bool = false, taps: bool = false, tree: bool = false, multi: ?struct { mt: *anyopaque, pos: *anyopaque, max_pos: u32 } = null };

/// One forward over `ids` (≤ CHUNK rows) at positions p0..; the greedy token of the last row (or rows: a.tok).
/// Up to 16 rows the GDN runs serially (gdn_step), as decode does, so a verified row equals the serial one.
fn forward(d: *Dev, m: *Model, a: *Acts, ids: []const i32, p0: u32, mode: Mode) !?i32 {
    const want = mode.want != .none;
    const T: u32 = @intCast(ids.len);
    const t32: i32 = @intCast(T);
    const tp = (T + 63) / 64 * 64; // GDN chunk rows (padded)
    const nul: ?*anyopaque = null;
    const gs: i32 = @intCast(GS);
    const tree: ?*anyopaque = if (mode.tree) a.tree else null;
    // W4A8 prompt: the norms before mixer in and gate|up quantize h themselves (not with the study mode)
    const q8on = prefill_int8 and T > 16 and GS == 128 and a8_group == 0;
    const q8a: ?*anyopaque = if (q8on) a.a8 else null;
    const q8s: ?*anyopaque = if (q8on) a.sa else null;
    const q8ms: i32 = @intCast(CHUNK + 32);
    const table = if (mode.tree) a.tree else a.chain;
    const mpos: ?*anyopaque = if (mode.multi) |mm| mm.pos else null;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
    try embedRows(d, m, a.ids, a.x, T);
    if (p0 == 0) try dump(d, a, T, "embed", false);
    for (&m.layers, 0..) |*l, i| {
        try d.launch(if (T > 16) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, if (i == 0) 0 else 1), gs, q8a, q8s, q8ms });
        const probe = p0 == 0 and i == 0;
        if (probe) try dumpBuf(d, "h0", a.h, T * D, .f16);
        // W4A8 study: the GEMM reads a quantized copy (in r, free until mixer_out); the dense a/b gates read h as is
        const min_in = if (a8_group != 0 and T > 16 and a8_sites & 1 != 0) a.r else a.h;
        if (min_in != a.h) try copy(d, a.r, a.h, T * D * 2);
        try a8(d, a, min_in, D, T, 1);
        try pgemm(d, a, min_in, l.mixer_in, a.p, T, q8on);
        if (l.ab) |ab| try d.launch(d.k.dense_rows, .{ .x = (T + 15) / 16, .y = 96 / 16 }, .{ a.h, ab, a.p, t32, @as(i32, GDN_IN), @as(i32, 16384) });
        if (probe) try dumpBuf(d, "p0", a.p, T * GDN_IN, .f16);
        if (l.attn) {
            const ctx = if (mode.multi) |mm| mm.max_pos + 1 else p0 + T; // a tree's rows see fewer: the slices past their depth are empty
            const mt: ?*anyopaque = if (mode.multi) |mm| off(mm.mt, i * 16 * 8 * 8) else null;
            const ns = (ctx + SPLIT - 1) / SPLIT;
            try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, @as(i32, @intCast(p0)), EPS, tree, mt, mpos, kv8, kvCap() });
            if (T > 16 and kv_i8) {
                try d.launch(d.k.attn_prompt8, .{ .x = (T + 7) / 8, .y = 4 }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(f32, 1.0 / 16.0), kvCap() });
            } else if (T > 16 and attn_own) {
                try d.launch(d.k.attn_prompt, .{ .x = (T + 7) / 8, .y = 4 }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(f32, 1.0 / 16.0) });
            } else if (T > 16) {                         // the port's Triton prompt kernel: 256 rows a program
                try d.launch(d.k.prompt_attn, .{ .x = (T + 255) / 256, .y = 24 }, .{ a.q, l.kc.?, l.vc.?, a.y, @as(i32, @intCast(p0)), t32, @as(i32, @intCast(ctx)), nul, nul });
            } else {
                try d.launch(d.k.attn_decode, .{ .x = ns, .y = 4, .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), @as(f32, 1.0 / 16.0), table, mt, mpos, kvCap() });
                try d.launch(d.k.attn_merge, .{ .x = 24, .y = T }, .{ a.pm, a.pl, a.po, a.y, @as(i32, @intCast(p0)), @as(i32, @intCast(ns)), table, mpos });
            }
            if (kv8 & 1 != 0) try d.launch(d.k.rot_heads, .{ .x = T, .y = 24 }, .{a.y});
            try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs });
        } else {
            const cin = l.conv[l.conv_cur].?;
            const cout = l.conv[1 - l.conv_cur].?;
            const gmt: ?*anyopaque = if (mode.multi) |mm| off(mm.mt, i * 16 * 8 * 8) else null;
            if (mode.multi == null) l.conv_cur = 1 - l.conv_cur; // multi: each slot's flips after the forward
            if (T > 16)
                try d.launch(d.k.gdn_pre_rows, .{ .x = (T + 7) / 8, .y = 5 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32 })
            else
                try d.launch(d.k.gdn_pre, .{ .x = T, .y = 40 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32, tree, gmt });
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
                if (mode.stash) {                        // inputs, for Runner.commit's replay
                    const st = l.stash.?;
                    try copy(d, off(st, ST_Q), a.q, T * 16 * 128 * 2);
                    try copy(d, off(st, ST_K), a.k, T * 16 * 128 * 2);
                    try copy(d, off(st, ST_V), a.v, T * 48 * 128 * 2);
                    try copy(d, off(st, ST_G), a.g, T * 48 * 4);
                    try copy(d, off(st, ST_B), a.beta, T * 48 * 4);
                    try copy(d, off(st, ST_P), a.p, T * GDN_IN * 2);
                }
                // verification: the state after the block goes to rec_save, rec keeps the entry for the replay
                // a tree: each row's state to acts.sl, rec untouched
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = 48, .z = if (mode.multi != null) T else 1 }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, if (mode.stash) l.rec_save.? else l.rec.?, a.y, t32, nul, tree, a.sl, gmt });
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
            try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, a.p, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs });
            if (probe) try dumpBuf(d, "o0", a.h, T * 6144, .f16);
        }
        try a8(d, a, a.h, 6144, T, 2);
        try pgemm(d, a, a.h, l.mixer_out, a.r, T, false);
        if (probe) try dumpBuf(d, "r0", a.r, T * D, .f16);
        try d.launch(if (T > 16) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), gs, q8a, q8s, q8ms });
        try a8(d, a, a.h, D, T, 4);
        const act = try mlpIn(d, a, l.gu, T, gs, q8on);
        try a8(d, a, act, 17408, T, 8);
        try pgemm(d, a, act, l.down, a.r, T, false);
        if (p0 == 0 and i < 4) try dump(d, a, T, try std.fmt.allocPrint(alloc, "{d}", .{i}), true);
        if (mode.taps) for (TAP_LAYERS, 0..) |tl, slot| if (tl == i)
            try d.launch(d.k.tap, .{ .x = D / 256, .y = T }, .{ a.x, a.r, a.taps, @as(i32, D), @as(i32, @intCast(slot)) });
    }
    if (!want and dump_h == null) {                  // x is per token: nothing carries to the next chunk
        try d.sync();
        return null;
    }
    const hgs: i32 = @intCast(m.head.gs);
    try d.launch(if (T > 16) d.k.add_rmsnorm else d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), hgs, no_q8, no_q8, @as(i32, 0) });
    if (dump_h) |fd| {                               // XPU_DUMP_H: every row's final normed hidden state, appended
        const host = try alloc.alloc(u16, T * D);
        defer alloc.free(host);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, host.ptr, a.h, T * D * 2, null, 0, null));
        try d.sync();
        _ = write(fd, @ptrCast(host.ptr), T * D * 2);
        if (!want) return null;
    }
    const first = if (mode.want == .rows) 0 else T - 1;
    const rows = T - first;
    try gemm(d, a.pscr, off(a.h, first * D * 2), off(a.ahi, first * (D / m.head.gs) * 2), off(a.alo, first * (D / m.head.gs) * 2), m.head, a.logits, rows);
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
    tree_on: bool = false, // the last verified block was a tree (commitPath)
    tree_host: [16 * 17]i32 = @splat(0),
    path_host: [16]i32 = @splat(0),
    slot: u32 = 0, // the selected sequence (select)
    mt_host: [LAYERS * 16 * 8]u64 = @splat(0), // verifyMulti's per-row state addresses, by layer
    pos_host: [16]i32 = @splat(0),
    win_host: [32]i32 = @splat(0), // and each window's first row and width (keepMulti)
    ids_host: [16]u32 = @splat(0),
    multi_slots: [16]u32 = @splat(0), // the last verifyMulti's slots, by window
    multi_count: u32 = 0,
    seqs: [MAX_SLOTS]struct { pos: u32 = 0, last: u32 = 0 } = @splat(.{}),
    /// Target hidden states for a drafter: when on, every forward fills acts.taps and calls `on_taps` with its rows.
    taps_on: bool = false,
    on_taps: ?struct { ctx: *anyopaque, f: *const fn (*anyopaque, *anyopaque, u32) anyerror!void } = null,

    /// Loads `dir` (MLX affine 4/64) onto the first B70 with room for `context` positions.
    pub fn init(dir: []const u8, context: u32) !*Runner {
        if (std.c.getenv("XPU_DUMP")) |v| dump_dir = std.mem.span(v);
        if (std.c.getenv("XPU_PREFILL_INT8") != null) prefill_int8 = true;
        if (std.c.getenv("XPU_ATTN_TRITON") != null) attn_own = false;
        if (std.c.getenv("XPU_KV_INT8") != null) kv_i8 = true;
        if (std.c.getenv("XPU_KV8")) |v| kv8 = try std.fmt.parseInt(i32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8_SITES")) |v| a8_sites = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8")) |v| a8_group = if (std.mem.eql(u8, std.mem.span(v), "token")) std.math.maxInt(u32) else try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_DUMP_H")) |v| dump_h = open(v, 0o2101, @as(c_uint, 0o644)); // O_WRONLY | O_CREAT | O_APPEND
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
        const mm128 = try d.g.module(&kb.gemm128);
        const mw128 = try d.g.module(&kb.ngen_wide128);
        const mn128 = try d.g.module(&kb.ngen_narrow128);
        const mpa = try d.g.module(&kb.prompt_attention);
        d.k = .{
            .embed = try d.kernel(mg, "embed", 256), .add_rmsnorm = try d.kernel(mg, "add_rmsnorm", 128), .add_rmsnorm_dec = try d.kernel(mg, "add_rmsnorm_dec", 512),
            .gdn_pre = try d.kernel(mg, "gdn_pre", 256), .gdn_pre_rows = try d.kernel(mg, "gdn_pre_rows", 256), .gated_norm = try d.kernel(mg, "gated_norm", 256),
            .attn_prep = try d.kernel(mg, "attn_prep", 256), .attn_gate = try d.kernel(mg, "attn_gate", 256),
            .swiglu = try d.kernel(mg, "swiglu", 256), .argmax = try d.kernel(mg, "argmax", 256), .argmax_part = try d.kernel(mg, "argmax_part", 256), .tap = try d.kernel(mg, "tap", 256), .conv_commit = try d.kernel(mg, "conv_commit", 256),
            .repack = try d.kernel(mg, "repack", 16), .gdn_step = try d.kernel(md, "gdn_step", 16),
            .attn_decode = try d.kernel(md, "attn_decode", 16), .attn_merge = try d.kernel(md, "attn_merge", 16), .kv_move = try d.kernel(md, "kv_move", 16), .exl3_gemv = try d.kernel(md, "exl3_gemv", 16), .rot_heads = try d.kernel(mg, "rot_heads", 16), .attn_prompt = try d.kernel(md, "attn_prompt", 128), .attn_prompt8 = try d.kernel(md, "attn_prompt8", 128),
            .embed_dense = try d.kernel(mg, "embed_dense", 256), .dense_rows = try d.kernel(mg, "dense_rows", 256),
            .repack_gptq = try d.kernel(mg, "repack_gptq", 16), .quantize_mlx = try d.kernel(mg, "quantize_mlx", 16),
            .dec_wide_g128 = try d.kernel(mm128, "w4k1_decode", 64), .dec_wide64_g128 = try d.kernel(mm128, "w4k1_d64", 64), .dec_narrow_g128 = try d.kernel(mm128, "w4k_dsplit", 64),
            .ngen_wide_g128 = try d.kernel(mw128, "w4g_dpb", 256), .ngen_narrow_g128 = try d.kernel(mn128, "w4g_s4dpb", 256),
            .ngen_swiglu = try d.kernel(try d.g.module(&kb.ngen_swiglu), "w4g_dpb_swiglu", 256),
            .ngen_swiglu_g128 = try d.kernel(try d.g.module(&kb.ngen_swiglu128), "w4g_dpb_swiglu", 256),
            .rowsums_ff = try d.kernel(mg, "rowsums_ff", 128), .fakequant_rows = try d.kernel(mg, "fakequant_rows", 256),
            .a8 = try d.kernel(try d.g.module(&kb.ngen_a8), "w4a8", 256), .a8_swiglu = try d.kernel(try d.g.module(&kb.ngen_a8_swiglu), "w4a8_swiglu", 256),
            .quant_a8 = try d.kernel(mg, "quant_a8", 256),
            .dec_wide = try d.kernel(mm, "w4k1_decode", 64), .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64), .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
            .gdn_prep = try d.kernel(mgd, "gdn_prep", 64), .gdn_state = try d.kernel(mgd, "gdn_state", 256),
            .gdn_out = try d.kernel(mgd, "gdn_out", 128), .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
            .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256), .prompt_attn = try d.kernel(mpa, "_prompt", 512),
        };
        // the shared rounds' kernels reach each slot's states through the per-row table (acts.mt)
        for ([_]ze.H{ d.k.attn_prep, d.k.attn_decode, d.k.gdn_pre, d.k.gdn_step }) |k| try d.g.indirect(k);
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
        r.store(); // slot 0: the states the load allocated
        try d.fill(r.acts.zeros, @as(u16, 0), CHUNK * 48 * 128 * 2);
        try d.fill(r.acts.ones, @as(f32, 1.0), CHUNK * 48 * 4);
        r.host_logits = try alloc.alloc(f32, VOCAB);
        return r;
    }

    /// Makes `slot` the one every call works on (its states allocated on first use); the previous slot's states and
    /// position are kept for it.
    pub fn select(r: *Runner, slot: u32) !void {
        if (slot >= MAX_SLOTS) return error.NoSuchSlot;
        if (slot == r.slot) return;
        try r.ensure(slot);
        r.store();
        r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
        for (&r.model.layers) |*l| {
            if (l.attn) {
                l.kc = l.slot_kc[slot];
                l.vc = l.slot_vc[slot];
            } else {
                l.rec = l.slot_rec[slot];
                l.rec_save = l.slot_save[slot];
                l.conv = l.slot_conv[slot];
                l.conv_cur = l.slot_cur[slot];
            }
        }
        r.pos = r.seqs[slot].pos;
        r.last = r.seqs[slot].last;
        r.slot = slot;
        r.block_rows = 0;
        r.tree_on = false;
    }

    /// A window of a shared verify: `ids` (the slot's pending token, then its drafts as a chain) from its position.
    pub const Win = struct { slot: u32, ids: []const u32 };

    /// Every window in one forward (16 rows at most, one window a slot): each row's greedy token in `block`, its
    /// logits row for `rowLogits`; nothing moves until `keepMulti`. Bits per row equal the slot's one-token steps.
    pub fn verifyMulti(r: *Runner, wins: []const Win) !void {
        if (wins.len == 0 or wins.len > 16) return error.BadBlock;
        r.store();
        var T: usize = 0;
        var max_pos: u32 = 0;
        for (wins, 0..) |w, i| {
            for (wins[0..i]) |o| if (o.slot == w.slot) return error.SlotTwice;
            if (w.ids.len == 0 or T + w.ids.len > 16) return error.BadBlock;
            const p = r.seqs[w.slot].pos;
            if (p + w.ids.len > r.context) return error.ContextFull;
            r.win_host[2 * i] = @intCast(T);
            r.win_host[2 * i + 1] = @intCast(w.ids.len);
            for (w.ids, 0..) |id, j| {
                r.pos_host[T + j] = @intCast(p + j);
                r.ids_host[T + j] = id;
            }
            max_pos = @max(max_pos, p + @as(u32, @intCast(w.ids.len)) - 1);
            T += w.ids.len;
        }
        for (&r.model.layers, 0..) |*l, i| for (wins, 0..) |w, wi| {
            const first: usize = @intCast(r.win_host[2 * wi]);
            for (0..w.ids.len) |j| {
                const e = r.mt_host[(i * 16 + first + j) * 8 ..][0..8];
                e.* = @splat(0);
                e[6] = first;
                e[7] = w.ids.len;
                if (l.attn) {
                    e[0] = @intFromPtr(l.slot_kc[w.slot].?);
                    e[1] = @intFromPtr(l.slot_vc[w.slot].?);
                } else {
                    const cur = l.slot_cur[w.slot];
                    e[2] = @intFromPtr(l.slot_conv[w.slot][cur].?);
                    e[3] = @intFromPtr(l.slot_conv[w.slot][1 - cur].?);
                    e[4] = @intFromPtr(l.slot_rec[w.slot].?);
                    e[5] = @intFromPtr(l.slot_save[w.slot].?);
                }
            }
        };
        const d = &r.d;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mt, &r.mt_host, LAYERS * 16 * 8 * 8, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mpos, &r.pos_host, T * 4, null, 0, null));
        _ = try forward(d, &r.model, &r.acts, @ptrCast(r.ids_host[0..T]), 0, .{ .want = .rows, .stash = true, .multi = .{ .mt = r.acts.mt, .pos = r.acts.mpos, .max_pos = max_pos } });
        var toks: [16]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, r.acts.tok, T * 4, null, 0, null));
        try d.sync();
        for (r.block[0..T], toks[0..T]) |*o, t| o.* = @intCast(t);
        for (wins, 0..) |w, i| r.multi_slots[i] = w.slot;
        r.multi_count = @intCast(wins.len);
    }

    /// Keeps the first `keep[i]` rows of window i of the last verifyMulti: a whole window takes the state its forward
    /// wrote, a shorter one replays its kept rows from the stash; each slot moves on by its kept rows.
    pub fn keepMulti(r: *Runner, keep: []const u32) !void {
        if (keep.len != r.multi_count) return error.BadCommit;
        const d = &r.d;
        const nul: ?*anyopaque = null;
        for (keep, 0..) |k, i| {
            const sl = r.multi_slots[i];
            const first: u32 = @intCast(r.win_host[2 * i]);
            const width: u32 = @intCast(r.win_host[2 * i + 1]);
            if (k == 0 or k > width) return error.BadCommit;
            for (&r.model.layers) |*l| if (!l.attn) {
                const cur = l.slot_cur[sl];
                if (k == width) {
                    std.mem.swap(?*anyopaque, &l.slot_rec[sl], &l.slot_save[sl]);
                } else {
                    const st = l.stash.?;
                    try d.launch(d.k.gdn_step, .{ .x = 16, .y = 48 }, .{ off(st, ST_Q + first * 16 * 128 * 2), off(st, ST_K + first * 16 * 128 * 2), off(st, ST_V + first * 48 * 128 * 2), off(st, ST_G + first * 48 * 4), off(st, ST_B + first * 48 * 4), l.slot_rec[sl].?, l.slot_rec[sl].?, r.acts.y, @as(i32, @intCast(k)), nul, nul, nul, nul });
                    try d.launch(d.k.conv_commit, .{ .x = 40 }, .{ l.slot_conv[sl][cur].?, off(st, ST_P + first * GDN_IN * 2), l.slot_conv[sl][1 - cur].?, @as(i32, @intCast(k)), nul });
                }
                l.slot_cur[sl] = 1 - cur;
            };
            r.seqs[sl].pos += k;
            r.seqs[sl].last = r.block[first + k - 1];
        }
        r.multi_count = 0;
        // the selected slot's views follow its arrays
        for (&r.model.layers) |*l| if (!l.attn) {
            l.rec = l.slot_rec[r.slot];
            l.rec_save = l.slot_save[r.slot];
            l.conv_cur = l.slot_cur[r.slot];
        };
        r.pos = r.seqs[r.slot].pos;
        r.last = r.seqs[r.slot].last;
    }

    /// One token for each of several slots: a verifyMulti of one-row windows, all kept.
    pub fn stepMulti(r: *Runner, slots: []const u32, tokens: []const u32) !void {
        var wins: [16]Win = undefined;
        if (slots.len > 16 or tokens.len != slots.len) return error.BadBlock;
        for (slots, tokens, 0..) |sl, *t, i| wins[i] = .{ .slot = sl, .ids = t[0..1] };
        try r.verifyMulti(wins[0..slots.len]);
        const ones: [16]u32 = @splat(1);
        try r.keepMulti(ones[0..slots.len]);
    }

    /// The selected slot's state pointers back into its slot arrays (a verify's commit may have swapped them).
    pub fn store(r: *Runner) void {
        for (&r.model.layers) |*l| {
            if (l.attn) {
                l.slot_kc[r.slot] = l.kc;
                l.slot_vc[r.slot] = l.vc;
            } else {
                l.slot_rec[r.slot] = l.rec;
                l.slot_save[r.slot] = l.rec_save;
                l.slot_conv[r.slot] = l.conv;
                l.slot_cur[r.slot] = l.conv_cur;
            }
        }
        r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
    }

    /// Bytes a slot's states take: GDN and convolution states, and the K/V cache of `context` positions.
    pub fn slotBytes(r: *const Runner) usize {
        _ = r;
        return 48 * (2 * 48 * 128 * 128 * 4 + 2 * 3 * 10240 * 2) + 16 * 2 * kvBytes();
    }

    /// `slot`'s states, allocated (zero) on its first use.
    pub fn ensure(r: *Runner, slot: u32) !void {
        const d = &r.d;
        for (&r.model.layers) |*l| {
            if (l.attn) {
                if (l.slot_kc[slot] != null) continue;
                l.slot_kc[slot] = try d.alloc_(kvBytes());
                l.slot_vc[slot] = try d.alloc_(kvBytes());
            } else {
                if (l.slot_rec[slot] != null) continue;
                l.slot_rec[slot] = try d.alloc_(48 * 128 * 128 * 4);
                l.slot_save[slot] = try d.alloc_(48 * 128 * 128 * 4);
                try d.fill(l.slot_rec[slot].?, @as(u32, 0), 48 * 128 * 128 * 4);
                for (&l.slot_conv[slot]) |*cv| {
                    cv.* = try d.alloc_(3 * 10240 * 2);
                    try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
                }
                l.slot_cur[slot] = 0;
            }
        }
        try d.sync();
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
        if (keep == 0 or keep > r.block_rows or r.tree_on) return error.BadCommit;
        const nul: ?*anyopaque = null;
        for (&r.model.layers) |*l| if (!l.attn) {
            if (keep == r.block_rows) { // every row kept: the block's final state is the state
                std.mem.swap(?*anyopaque, &l.rec, &l.rec_save);
                continue;
            }
            const st = l.stash.?;
            try r.d.launch(r.d.k.gdn_step, .{ .x = 16, .y = 48 }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.rec.?, l.rec.?, r.acts.y, @as(i32, @intCast(keep)), nul, nul, nul, nul });
            // the block's forward flipped conv_cur: conv[cur] is the state after all rows, conv[1 - cur] the entry
            try r.d.launch(r.d.k.conv_commit, .{ .x = 40 }, .{ l.conv[1 - l.conv_cur].?, off(st, ST_P), l.conv[l.conv_cur].?, @as(i32, @intCast(keep)), nul });
        };
        r.pos += keep;
        r.last = r.block[keep - 1];
        r.block_rows = 0; // no wait: the next commands queue behind these on the in-order list
    }

    /// Verifies a tree: row 0 the pending token, `parents[t]` the row below row t (t ≥ 1, parents first). Every row's
    /// greedy token returned; `commitPath` then keeps one root path.
    pub fn verifyTree(r: *Runner, ids: []const u32, parents: []const i32) ![]const u32 {
        if (ids.len == 0 or ids.len > 16 or parents.len != ids.len) return error.BadBlock;
        var depth_max: u32 = 0;
        for (0..ids.len) |t| {
            const row = r.tree_host[t * 17 ..][0..17];
            if (t == 0) {
                row[0] = 0;
                row[1] = 0;
                continue;
            }
            const p: usize = @intCast(parents[t]);
            if (p >= t) return error.BadBlock;
            const pr = r.tree_host[p * 17 ..][0..17];
            row[0] = pr[0] + 1;
            @memcpy(row[1 .. 1 + @as(usize, @intCast(pr[0])) + 1], pr[1 .. 1 + @as(usize, @intCast(pr[0])) + 1]);
            row[1 + @as(usize, @intCast(row[0]))] = @intCast(t);
            depth_max = @max(depth_max, @as(u32, @intCast(row[0])));
        }
        for (1..ids.len) |t| { // LOAD: the parent is not the row before ; SAVE: a child is not the row after
            const p: usize = @intCast(parents[t]);
            if (p + 1 != t) {
                r.tree_host[t * 17] |= 512;
                r.tree_host[p * 17] |= 256;
            }
        }
        if (r.pos + depth_max + 1 > r.context or r.pos + ids.len > r.context) return error.ContextFull;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, r.acts.tree, &r.tree_host, ids.len * 17 * 4, null, 0, null));
        _ = try forward(&r.d, &r.model, &r.acts, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on, .tree = true });
        var toks: [16]i32 = undefined;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, &toks, r.acts.tok, ids.len * 4, null, 0, null));
        try r.d.sync();
        for (r.block[0..ids.len], toks[0..ids.len]) |*o, t| o.* = @intCast(t);
        r.block_rows = @intCast(ids.len);
        r.tree_on = true;
        return r.block[0..ids.len];
    }

    /// Keeps the root path `path` (rows of the last verified tree, path[0] = 0, each the parent of the next): GDN and
    /// convolution states replayed along it, its K/V moved to their positions, its taps gathered to the front.
    pub fn commitPath(r: *Runner, path: []const u32) !void {
        if (!r.tree_on or path.len == 0 or path.len > r.block_rows or path[0] != 0) return error.BadCommit;
        var straight = true;
        for (path, 0..) |row, dd| {
            r.path_host[dd] = @intCast(row);
            straight = straight and row == dd;
        }
        const keep: i32 = @intCast(path.len);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, r.acts.path, &r.path_host, path.len * 4, null, 0, null));
        const nul: ?*anyopaque = null;
        for (&r.model.layers) |*l| {
            if (l.attn) {
                if (!straight) try r.d.launch(r.d.k.kv_move, .{ .x = 4 }, .{ l.kc.?, l.vc.?, @as(i32, @intCast(r.pos)), r.acts.path, keep, kvCap() });
                continue;
            }
            const st = l.stash.?;
            try r.d.launch(r.d.k.gdn_step, .{ .x = 16, .y = 48 }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.rec.?, l.rec.?, r.acts.y, keep, r.acts.path, nul, nul, nul });
            try r.d.launch(r.d.k.conv_commit, .{ .x = 40 }, .{ l.conv[1 - l.conv_cur].?, off(st, ST_P), l.conv[l.conv_cur].?, keep, r.acts.path });
        }
        if (r.taps_on and !straight) for (path, 0..) |row, dd| if (row != dd)
            try copy(&r.d, off(r.acts.taps, dd * 5 * D * 2), off(r.acts.taps, row * 5 * D * 2), 5 * D * 2);
        r.pos += @intCast(path.len);
        r.last = r.block[path[path.len - 1]];
        r.block_rows = 0;
        r.tree_on = false;
        try r.d.sync(); // path_host is read by the queued copy
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

pub const drafter = @import("drafter.zig");
