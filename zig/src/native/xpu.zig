//! XPU engines (Intel Xe2 over Level Zero): Qwen3.8-27B, MLX affine 4/64, behind the shared native serial host.
const std = @import("std");
const api = @import("engine_api");
const xq = @import("xpu_qwen27");
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

    fn self(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        try self(ptr).runner.reset();
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
    return .{ .engine = h.serial.engine(), .close = close, .ctx = h };
}
