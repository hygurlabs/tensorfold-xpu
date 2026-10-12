//! Development runner of the Qwen3.8-27B XPU forward: greedy tokens and timings, to check against the fp32 torch
//! reference of the Python port (tools/xpu-bench/integ/oracle.py).
//!   xpu-qwen27 MODEL_DIR IDS_FILE [STEPS]      IDS_FILE: prompt token ids separated by spaces, or --bench=N for
//!                                              N synthetic prompt tokens ; XPU_CTX: context (4 096)
const std = @import("std");
const q = @import("qwen27.zig");
const Drafter = @import("drafter.zig").Drafter;

const alloc = std.heap.c_allocator;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: xpu-qwen27 MODEL_DIR IDS_FILE [STEPS]\n", .{});
        return;
    }
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    const ctx: u32 = if (std.c.getenv("XPU_CTX")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 4096;
    var ids: std.ArrayList(u32) = .empty;
    if (std.mem.startsWith(u8, argv[2], "--bench=")) {
        const n = try std.fmt.parseInt(usize, argv[2][8..], 10);
        for (0..n) |i| try ids.append(alloc, @intCast((i * 7919 + 13) % 200000));
    } else {
        const path = try std.fmt.allocPrintSentinel(alloc, "{s}", .{argv[2]}, 0);
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        const size: usize = @intCast(lseek(fd, 0, 2));
        const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
        var it = std.mem.tokenizeAny(u8, text[0..size], " \n,[]");
        while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
    }
    q.verbose = true;
    const t_load = q.now();
    const r = try q.Runner.init(argv[1], ctx);
    std.debug.print("loaded in {d:.1} s\n", .{q.now() - t_load});
    const t0 = q.now();
    try r.prompt(ids.items, true);
    const t1 = q.now();
    q.profReport("prompt");
    var out: std.ArrayList(u32) = .empty;
    try out.append(alloc, r.last);
    for (1..steps) |_| {
        try r.step(r.last);
        try out.append(alloc, r.last);
    }
    const t2 = q.now();
    q.profReport("decode");
    if (std.c.getenv("XPU_VERIFY_TEST") != null) try verifyTest(r, ids.items, out.items);
    if (std.c.getenv("XPU_DRAFTER")) |dir| try drafted(r, std.mem.span(dir), ids.items, out.items);
    std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.0} tok/s)\n", .{ ids.items.len, t1 - t0, @as(f64, @floatFromInt(ids.items.len)) / (t1 - t0) });
    std.debug.print("decode: {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, t2 - t1, @as(f64, @floatFromInt(steps - 1)) / (t2 - t1) });
    std.debug.print("tokens:", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}

/// The drafted-equals-serial contract with stand-in drafts: blocks of 8 verified rows, with perfect drafts (the serial
/// reply itself) and with wrong ones (token 0); both replies must equal the serial one.
fn verifyTest(r: *q.Runner, prompt: []const u32, serial: []const u32) !void {
    for ([_]bool{ true, false }) |perfect| {
        try r.reset();
        try r.prompt(prompt, true);
        var got: std.ArrayList(u32) = .empty;
        try got.append(alloc, r.last);
        var rounds: usize = 0;
        while (got.items.len < serial.len) {
            var block: [8]u32 = undefined;
            block[0] = got.items[got.items.len - 1];
            for (1..8) |j| {
                const at = got.items.len - 1 + j;
                block[j] = if (perfect and at < serial.len) serial[at] else 0;
            }
            const greedy = try r.verify(&block);
            var n: u32 = 0; // accepted drafts
            while (n + 1 < 8 and greedy[n] == block[n + 1]) n += 1;
            try r.commit(n + 1);
            for (greedy[0 .. n + 1]) |t| try got.append(alloc, t);
            rounds += 1;
        }
        const same = std.mem.eql(u32, got.items[0..serial.len], serial);
        std.debug.print("verify test ({s} drafts): {d} rounds, equal to serial: {}\n", .{ if (perfect) "perfect" else "wrong", rounds, same });
        if (!same) {
            std.debug.print("  got:", .{});
            for (got.items[0..serial.len]) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\n", .{});
        }
    }
}

fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) !void {
    const dr: *Drafter = @ptrCast(@alignCast(ctx));
    try dr.addTaps(taps, n);
}

/// Greedy decode with DFlash2 drafts (blocks of XPU_BLOCK rows, 8 by default): must equal the serial reply.
fn drafted(r: *q.Runner, dir: []const u8, prompt: []const u32, serial: []const u32) !void {
    const t_load = q.now();
    const dr = try Drafter.init(r, dir);
    std.debug.print("drafter loaded in {d:.1} s\n", .{q.now() - t_load});
    const L: u32 = if (std.c.getenv("XPU_BLOCK")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 8;
    try r.reset();
    dr.reset();
    r.taps_on = true;
    r.on_taps = .{ .ctx = dr, .f = onTaps };
    try r.prompt(prompt, true);
    var got: std.ArrayList(u32) = .empty;
    try got.append(alloc, r.last);
    var rounds: usize = 0;
    q.profReport("drafter prompt (discarded)");
    var t_draft: f64 = 0;
    var t_verify: f64 = 0;
    const t0 = q.now();
    while (got.items.len < serial.len) {
        var block: [16]u32 = undefined;
        block[0] = r.last;
        const a = q.now();
        try dr.propose(r.last, L - 1, block[1..L]);
        const b = q.now();
        const greedy = try r.verify(block[0..L]);
        var n: u32 = 0;
        while (n + 1 < L and greedy[n] == block[n + 1]) n += 1;
        try r.commit(n + 1);
        try dr.addTaps(r.acts.taps, n + 1);
        t_verify += q.now() - b;
        t_draft += b - a;
        for (greedy[0 .. n + 1]) |t| try got.append(alloc, t);
        rounds += 1;
    }
    const dt = q.now() - t0;
    q.profReport("drafted rounds");
    const len = @min(got.items.len, serial.len);
    const same = std.mem.eql(u32, got.items[0..len], serial[0..len]);
    std.debug.print("drafted: {d} tokens in {d} rounds ({d:.2} a round), {d:.1} tok/s, draft {d:.1} ms + verify {d:.1} ms a round, equal to serial: {}\n", .{ got.items.len - 1, rounds, @as(f64, @floatFromInt(got.items.len - 1)) / @as(f64, @floatFromInt(rounds)), @as(f64, @floatFromInt(got.items.len - 1)) / dt, t_draft * 1e3 / @as(f64, @floatFromInt(rounds)), t_verify * 1e3 / @as(f64, @floatFromInt(rounds)), same });
}
