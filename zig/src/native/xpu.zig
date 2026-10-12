//! XPU engines (Intel Xe2 over Level Zero): Qwen3.8-27B, MLX affine 4/64, behind the shared native serial host.
const std = @import("std");
const api = @import("engine_api");
const xq = @import("xpu_qwen27");
const Drafter = xq.drafter.Drafter;
const Serial = api.serial_host;
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"xpu"};
pub const families: []const api.Family = &.{.{ .model_type = "qwen3_5", .formats = &.{"mlx-q4g64"} }};

pub fn chip(_: Allocator) ?[]const u8 {
    return "Intel Xe2 (Level Zero)";
}

const Host = struct {
    gpa: Allocator,
    runner: *xq.Runner,
    serial: *Serial.Host,
    drafter: ?*Drafter = null,
    block: u32 = 8,
    out: [16]u32 = undefined,

    fn self(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        const h = self(ptr);
        try h.runner.reset();
        if (h.drafter) |dr| dr.reset();
    }
    fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) !void {
        const dr: *Drafter = @ptrCast(@alignCast(ctx));
        try dr.addTaps(taps, n);
    }
    /// One drafted round: the pending token and the accepted drafts (the bonus stays pending for the next round).
    fn batch(ptr: *anyopaque, budget: u32, eos: []const u32) !Serial.Batch {
        const h = self(ptr);
        const r = h.runner;
        const dr = h.drafter orelse return error.DraftNotAttached;
        const L = h.block;
        var block: [16]u32 = undefined;
        block[0] = r.last;
        try dr.propose(r.last, L - 1, block[1..L]);
        const greedy = try r.verify(block[0..L]);
        var n: u32 = 0;
        while (n + 1 < L and greedy[n] == block[n + 1]) n += 1;
        try r.commit(n + 1);
        try dr.addTaps(r.acts.taps, n + 1);
        h.out[0] = block[0];
        @memcpy(h.out[1 .. n + 1], greedy[0..n]);
        var count = @min(n + 1, budget);
        for (h.out[0..count], 0..) |t, i| if (std.mem.indexOfScalar(u32, eos, t) != null) {
            count = @intCast(i + 1);
            break;
        };
        return .{ .tokens = h.out[0..count], .stats = .{ .rounds = 1, .drafted = L - 1, .accepted = n, .min_rows = L } };
    }
    fn chunk(ptr: *anyopaque, ids: []const u32, last: bool) !void {
        try self(ptr).runner.prompt(ids, last);
    }
    fn draw(ptr: *anyopaque, sampling: ?api.Sampling) !u32 {
        const h = self(ptr);
        if (h.runner.pos == 0) return error.RoundNotReady;
        if (sampling) |s| return Serial.choose(h.gpa, .{ .f32 = try h.runner.logits() }, h.runner.pos, s);
        return h.runner.last;
    }
    fn advance(ptr: *anyopaque, token: u32) !void {
        try self(ptr).runner.step(token);
    }
    fn fed(ptr: *anyopaque) u64 {
        return self(ptr).runner.pos;
    }
};

fn close(ptr: *anyopaque) void {
    const h = Host.self(ptr);
    h.serial.deinit();
    h.gpa.destroy(h);
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "qwen3_5")) {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine serves Qwen3.8-27B (qwen3_5, MLX affine 4/64) only, not {s}", .{o.model_type});
        return null;
    }
    const context: u32 = if (o.context) |c| @intCast(c) else 32768;
    const runner = xq.Runner.init(o.dir, context) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine could not load {s}: {s}", .{ o.dir, @errorName(err) });
        return null;
    };
    const h = try gpa.create(Host);
    h.* = .{ .gpa = gpa, .runner = runner, .serial = undefined };
    h.serial = try Serial.Host.init(gpa, io, .{
        .ctx = h,
        .info = .{ .name = "qwen27-xpu", .lanes = 1, .context_window = context, .prefill_step = 1024 },
        .reset = Host.reset,
        .prompt_chunk = Host.chunk,
        .draw = Host.draw,
        .advance = Host.advance,
        .fed = Host.fed,
    });
    if (o.drafts) if (o.drafter) |dir| {
        const dr = Drafter.init(runner, dir) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "the XPU engine could not load the drafter {s}: {s}", .{ dir, @errorName(err) });
            return null;
        };
        h.drafter = dr;
        if (std.c.getenv("XPU_BLOCK")) |v| h.block = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 8;
        runner.taps_on = true;
        runner.on_taps = .{ .ctx = dr, .f = Host.onTaps };
        h.serial.driver.draft_batch = Host.batch;
    };
    return .{ .engine = h.serial.engine(), .close = close, .ctx = h };
}
