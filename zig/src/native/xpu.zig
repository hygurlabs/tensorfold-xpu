//! XPU engines (Intel Xe2 over Level Zero): Qwen3.8-27B (MLX affine 4/64, GPTQ 4/128) behind the native serial host
//! (`--parallel 1`) or the lane host (`--parallel N`: shared rounds across N streams, xpu_lanes.zig).
const std = @import("std");
const api = @import("engine_api");
const xq = @import("xpu_qwen27");
const Drafter = xq.drafter.Drafter;
const Serial = api.serial_host;
const lanes = @import("lanes");
const sampling = lanes.sampling;
const XpuLanes = @import("xpu_lanes.zig").Lanes;
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"xpu"};
pub const families: []const api.Family = &.{.{ .model_type = "qwen3_5", .formats = &.{ "mlx-q4g64", "gptq-int4-g128" } }};

pub fn chip(_: Allocator) ?[]const u8 {
    return "Intel Xe2 (Level Zero)";
}

const Host = struct {
    gpa: Allocator,
    runner: *xq.Runner,
    serial: *Serial.Host,
    drafter: ?*Drafter = null,
    block: u32 = 8,
    nodes: u32 = 15, // drafts a tree round verifies (XPU_TREE; 0: one chain of block − 1)
    out: [16]u32 = undefined,
    pending: ?u32 = null, // a sampled reply's next token, drawn at the end of the previous round

    fn self(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        const h = self(ptr);
        h.pending = null;
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
        if (h.nodes > 0) {
            const round = try xq.drafter.treeRound(dr, r.last, L - 1, h.nodes, null, h.out[0..16]);
            return h.cut(round, budget, eos);
        }
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
        const h = self(ptr);
        h.pending = null;
        try h.runner.prompt(ids, last);
    }
    /// The serial draw of `row`'s logits at `position`: `Serial.choose` over the top `top_k` candidates only (the
    /// draw reads no other: same candidates, same order by value then id, same keyed noise), not the whole vocabulary.
    fn pick(h: *Host, row: []const f32, position: u64, s: api.Sampling) !u32 {
        return pickRow(h.gpa, row, position, s);
    }
    fn pickCtx(ctx: *anyopaque, row: []const f32, position: u64, s: api.Sampling) !u32 {
        const gpa: *const Allocator = @ptrCast(@alignCast(ctx));
        return pickRow(gpa.*, row, position, s);
    }
    fn pickRow(gpa: Allocator, row: []const f32, position: u64, s: api.Sampling) !u32 {
        if (s.temperature <= 0 or s.top_k == 0 or s.top_k > 256) return Serial.choose(gpa, .{ .f32 = row }, position, s);
        const k: usize = s.top_k;
        var vals: [256]f64 = undefined;
        var ids: [256]u64 = undefined;
        var n: usize = 0;
        for (row, 0..) |v32, i| {
            const v: f64 = v32;
            if (!std.math.isFinite(v)) return error.NonfiniteLogits;
            if (n == k and !(v > vals[k - 1])) continue; // equal values keep the lower id, already held
            var j = if (n < k) n else k - 1;
            if (n < k) n += 1;
            while (j > 0 and v > vals[j - 1]) : (j -= 1) {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
            }
            vals[j] = v;
            ids[j] = i;
        }
        return @intCast(try sampling.choose(gpa, vals[0..n], ids[0..n], position, s));
    }
    fn draw(ptr: *anyopaque, s: ?api.Sampling) !u32 {
        const h = self(ptr);
        if (h.runner.pos == 0) return error.RoundNotReady;
        if (s) |sp| return h.pick(try h.runner.logits(), h.runner.pos, sp);
        return h.runner.last;
    }
    /// A drafted round for a sampled reply: every verified row is drawn at its position as the serial reply would,
    /// and a draft counts only while it equals that draw; the first differing draw is the round's bonus.
    /// A tree round's tokens cut at the budget and the first stop token.
    fn cut(h: *Host, round: xq.drafter.Round, budget: u32, eos: []const u32) Serial.Batch {
        _ = h;
        var count: u32 = @min(@as(u32, @intCast(round.tokens.len)), budget);
        for (round.tokens[0..count], 0..) |t, i| if (std.mem.indexOfScalar(u32, eos, t) != null) {
            count = @intCast(i + 1);
            break;
        };
        return .{ .tokens = round.tokens[0..count], .stats = .{ .rounds = 1, .drafted = round.drafted, .accepted = round.accepted, .min_rows = round.drafted + 1 } };
    }
    const Sampled = struct {
        h: *Host,
        s: api.Sampling,
        fn row(ctx: *anyopaque, row_: u32, position: u64) !u32 {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            return c.h.pick(try c.h.runner.rowLogits(row_), position, c.s);
        }
        fn gumbel(ctx: *anyopaque, position: u64, id: u64) f64 {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            return -@log(-@log(sampling.uniform(c.s.seed, position, id)));
        }
    };
    fn batchSampled(ptr: *anyopaque, budget: u32, eos: []const u32, s: api.Sampling) !Serial.Batch {
        const h = self(ptr);
        const r = h.runner;
        const dr = h.drafter orelse return error.DraftNotAttached;
        const L = h.block;
        const p0 = r.pos;
        const pending = h.pending orelse try h.pick(try r.rowLogits(0), p0, s);
        if (h.nodes > 0) {
            var ctx: Sampled = .{ .h = h, .s = s };
            const round = try xq.drafter.treeRound(dr, pending, L - 1, h.nodes, .{ .ctx = &ctx, .row = Sampled.row, .gumbel = Sampled.gumbel, .temperature = s.temperature }, h.out[0..16]);
            h.pending = round.terminal;
            return h.cut(round, budget, eos);
        }
        var block: [16]u32 = undefined;
        block[0] = pending;
        try dr.propose(pending, L - 1, block[1..L]);
        _ = try r.verify(block[0..L]);
        var n: u32 = 0;
        var bonus: u32 = 0;
        while (true) {
            const t = try h.pick(try r.rowLogits(n), p0 + 1 + n, s);
            if (n + 1 < L and t == block[n + 1]) {
                n += 1;
                continue;
            }
            bonus = t;
            break;
        }
        try r.commit(n + 1);
        try dr.addTaps(r.acts.taps, n + 1);
        h.pending = bonus;
        @memcpy(h.out[0 .. n + 1], block[0 .. n + 1]);
        var count = @min(n + 1, budget);
        for (h.out[0..count], 0..) |t, i| if (std.mem.indexOfScalar(u32, eos, t) != null) {
            count = @intCast(i + 1);
            break;
        };
        return .{ .tokens = h.out[0..count], .stats = .{ .rounds = 1, .drafted = L - 1, .accepted = n, .min_rows = L } };
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

/// `--parallel N`: the lane host over N runner slots.
const LaneServed = struct {
    gpa: Allocator,
    back: XpuLanes,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ptr: *anyopaque) void {
        const h: *LaneServed = @ptrCast(@alignCast(ptr));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        for (h.back.first_logits) |l| if (l) |x| h.gpa.free(x);
        h.gpa.destroy(@as(*Allocator, @ptrCast(@alignCast(h.back.pick_ctx))));
        h.gpa.destroy(h);
    }
};

/// Slots the card holds beside what is loaded: each a GDN state and a K/V cache of `context` tokens.
fn slotsThatFit(runner: *xq.Runner, want: u32) u32 {
    const card: usize = 31 << 30; // the B70's 32 GiB, less the driver's share
    const margin: usize = 2 << 30; // activations of later chunks, the drafter's growth
    const free = card -| runner.d.allocated -| margin;
    const more: u32 = @intCast(@min(free / runner.slotBytes(), xq.MAX_SLOTS - 1));
    return @max(1, @min(want, 1 + more));
}

fn openLanes(gpa: Allocator, io: std.Io, runner: *xq.Runner, drafter: ?*Drafter, o: api.Open, context: u32) !api.Opened {
    const slots = slotsThatFit(runner, @min(o.lanes, xq.MAX_SLOTS));
    const h = try gpa.create(LaneServed);
    errdefer gpa.destroy(h);
    const gpa_box = try gpa.create(Allocator);
    gpa_box.* = gpa;
    h.gpa = gpa;
    h.back = .{ .gpa = gpa, .runner = runner, .drafter = drafter, .pick = Host.pickCtx, .pick_ctx = gpa_box, .slots = slots };
    if (std.c.getenv("XPU_BLOCK")) |v| h.back.block = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 8;
    if (std.c.getenv("XPU_TREE")) |v| h.back.nodes = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 15;
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), 16, 15);
    errdefer h.cfg.deinit(gpa);
    h.wall = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "qwen27-xpu", .lanes = slots, .context_window = context, .prefill_step = 1024, .structures = true });
    if (drafter != null) {
        h.back.initDrafts();
        h.host.lone = .{ .ctx = &h.back, .run = XpuLanes.lone, .@"resume" = XpuLanes.resumeLone, .sampled = true };
    }
    try h.host.start();
    std.log.info("qwen27-xpu: {d} streams of {d} tokens{s}", .{ slots, context, if (drafter != null) ", DFlash2 trees for a lone stream" else "" });
    return .{ .engine = h.host.engine(), .close = LaneServed.close, .ctx = h };
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "qwen3_5")) {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine serves Qwen3.8-27B (qwen3_5, MLX affine 4/64 or symmetric GPTQ 4/128) only, not {s}", .{o.model_type});
        return null;
    }
    const context: u32 = if (o.context) |c| @intCast(c) else 32768;
    const runner = xq.Runner.init(o.dir, context) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine could not load {s}: {s}", .{ o.dir, @errorName(err) });
        return null;
    };
    if (o.lanes > 1) {
        var dr: ?*Drafter = null;
        if (o.drafts) if (o.drafter) |dir| {
            dr = Drafter.init(runner, dir) catch |err| {
                problem.* = try std.fmt.allocPrint(a, "the XPU engine could not load the drafter {s}: {s}", .{ dir, @errorName(err) });
                return null;
            };
        };
        return try openLanes(gpa, io, runner, dr, o, context);
    }
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
        if (std.c.getenv("XPU_TREE")) |v| h.nodes = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 15;
        runner.taps_on = true;
        runner.on_taps = .{ .ctx = dr, .f = Host.onTaps };
        h.serial.driver.draft_batch = Host.batch;
        h.serial.driver.draft_sampled = Host.batchSampled;
    };
    return .{ .engine = h.serial.engine(), .close = close, .ctx = h };
}
