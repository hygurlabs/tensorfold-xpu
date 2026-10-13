//! The XPU 27B on the lane host: a stream a runner slot, shared rounds across slots (Runner.verifyMulti: each
//! stream's pending token and its copied drafts as a chain, 16 rows a forward), and a lone drafted stream on DFlash2
//! trees (the host's `lone` driver) until another stream arrives; when the others end, the one left resumes alone, its
//! drafter context rebuilt from the taps of its last rows (each slot keeps RING of them).
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const be = lanes.backend;
const Drafter = xq.drafter.Drafter;
const Allocator = std.mem.Allocator;

const ring = 64;
const RING = 256; // a slot's last rows whose taps are kept: the drafter's context when the stream resumes alone
const TAP_ROW = 5 * xq.D * 2; // bf16 [5 D]

pub const Lanes = struct {
    gpa: Allocator,
    runner: *xq.Runner,
    drafter: ?*Drafter,
    pick: *const fn (ctx: *anyopaque, row: []const f32, position: u64, s: api.Sampling) anyerror!u32,
    pick_ctx: *anyopaque,
    slots: u32,
    block: u32 = 8,
    nodes: u32 = 15,
    owners: [xq.MAX_SLOTS]?*lanes.Stream = @splat(null),
    first_logits: [xq.MAX_SLOTS]?[]f32 = @splat(null), // a sampled stream's prompt-row logits, for `first`
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    ids: [16]u32 = undefined, // the last verify's rows, window after window
    widths: [16]u32 = undefined, // and each window's rows
    count: usize = 0,
    wslots: [16]u32 = undefined, // and each window's slot
    taps_ring: ?*anyopaque = null, // bf16 [slots][RING][5 D], when a drafter is attached
    ring_end: [xq.MAX_SLOTS]u64 = @splat(0), // rows a slot pushed: its position while its taps are kept
    feed: bool = false, // the prompt's taps go to the drafter too (a lone stream's prefill)

    fn self(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn backend(b: *Lanes) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    /// What the round loop reads: chains of up to 16 rows (a window's rows equal one-token steps bit for bit), every
    /// stream's window in one shared forward of 16 rows at most; drafts are the core's copies (no head in the rounds).
    pub fn facts(b: *const Lanes) lanes.Model {
        return .{ .exact_width = 16, .mtp = false, .speculate = false, .speculate_early = false, .drafts = 0, .hidden_rows = true, .streams_exact = true, .batch_rows = 16, .max_streams = b.slots, .lone_copies = false };
    }

    /// The taps ring (and the runner's taps callback): with a drafter attached.
    pub fn initRing(b: *Lanes) !void {
        b.taps_ring = try b.runner.d.alloc_(@as(usize, b.slots) * RING * TAP_ROW);
        b.runner.on_taps = .{ .ctx = b, .f = onTaps };
    }

    fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) anyerror!void {
        const b = self(ctx);
        try b.ringPush(b.runner.slot, taps, n);
        if (b.feed) try b.drafter.?.addTaps(taps, n);
    }

    /// `slot`'s next `n` rows' taps (rows of `taps` in order) into its ring: the last RING stay.
    fn ringPush(b: *Lanes, slot: u32, taps: *anyopaque, n: u32) !void {
        const tr = b.taps_ring orelse return;
        const d = &b.runner.d;
        const skip = n -| RING;
        var done: u32 = skip;
        while (done < n) {
            const at: u32 = @intCast((b.ring_end[slot] + done) % RING);
            const rows = @min(n - done, RING - at);
            try xq.copy(d, xq.off(tr, (@as(usize, slot) * RING + at) * TAP_ROW), xq.off(taps, @as(usize, done) * TAP_ROW), @as(usize, rows) * TAP_ROW);
            done += rows;
        }
        b.ring_end[slot] += n;
    }

    /// Keeps the first `keep[i]` rows of the last verify's window i; their taps into the windows' rings.
    fn keepRows(b: *Lanes, keep: []const u32) !void {
        try b.runner.keepMulti(keep);
        var first: usize = 0;
        for (keep, 0..) |k, i| {
            try b.ringPush(b.wslots[i], xq.off(b.runner.acts.taps, first * TAP_ROW), k);
            first += b.widths[i];
        }
    }

    /// A verify whose rows all stayed: the core calls keep only to drop rows, so the next call keeps them all first.
    fn settle(b: *Lanes) !void {
        if (b.count == 0) return;
        const n = b.count;
        b.count = 0;
        try b.keepRows(b.widths[0..n]);
    }

    fn slotOf(b: *const Lanes, s: *const lanes.Stream) ?u32 {
        for (b.owners[0..b.slots], 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    fn push(b: *Lanes, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    fn value(b: *const Lanes, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| if (h < b.next and b.next - h <= ring) b.drawn[h % ring] else error.NoSuchToken,
        };
    }

    /// `slot`'s next greedy token (the runner keeps it per slot).
    fn lastOf(b: *const Lanes, slot: u32) u32 {
        const r = b.runner;
        return if (r.slot == slot) r.last else r.seqs[slot].last;
    }

    /// A stream's prompt into its slot (a free one, slot 0 last: the lone driver's); its last row's logits kept when
    /// it samples.
    fn prefillSlot(b: *Lanes, s: *lanes.Stream, slot: u32, taps: bool) !void {
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + 17 > b.runner.context) return error.PromptTooLong;
        const r = b.runner;
        try r.select(slot);
        try r.reset();
        r.taps_on = taps or b.taps_ring != null;
        b.feed = taps;
        defer {
            r.taps_on = false;
            b.feed = false;
        }
        b.ring_end[slot] = 0;
        if (taps) if (b.drafter) |d| d.reset();
        s.cached = 0;
        try r.prompt(ids, true);
        if (s.sampling != null) {
            const row = try r.logits();
            const keep = b.first_logits[slot] orelse try b.gpa.alloc(f32, row.len);
            @memcpy(keep, row);
            b.first_logits[slot] = keep;
        }
    }

    fn freeSlot(b: *const Lanes) ?u32 {
        var k: u32 = b.slots;
        while (k > 0) {
            k -= 1;
            if (b.owners[k] == null) return k;
        }
        return null;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        try b.settle();
        const slot = b.slotOf(s) orelse (b.freeSlot() orelse return error.NoFreeSlot);
        b.owners[slot] = s;
        errdefer b.owners[slot] = null;
        try b.prefillSlot(s, slot, false);
    }

    fn draw(b: *Lanes, s: *const lanes.Stream, slot: u32, row: ?[]const f32, position: u64) !u32 {
        const sp = s.sampling orelse return b.lastOf(slot);
        return b.pick(b.pick_ctx, row orelse return error.NoLogits, position, sp);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        return b.push(try b.draw(s, slot, b.first_logits[slot], position));
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        try b.settle();
        const r = b.runner;
        try r.select(slot);
        r.taps_on = b.taps_ring != null;
        defer r.taps_on = false;
        try r.step(try b.value(feed));
        return b.push(try b.draw(s, slot, if (s.sampling != null) try r.logits() else null, position));
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).value(.{ .handle = handle });
    }

    /// Every stream's window (its pending token, then its copied drafts as a chain) in one forward, each row drawn
    /// with its stream's sampling at its position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > 16 or windows.len != out.len) return error.BadWindows;
        try b.settle();
        var wins: [16]xq.Runner.Win = undefined;
        var total: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.held > 0) return error.HeldDrafts;
            if (w.parents) |p| for (p, 0..) |x, j| if (x != @as(i32, @intCast(j)) - 1) return error.TreeWindow;
            const rows = w.rows();
            if (total + rows > 16) return error.WindowTooWide;
            const ids = b.ids[total..][0..rows];
            ids[0] = w.pending;
            @memcpy(ids[1..], w.tokens);
            wins[i] = .{ .slot = b.slotOf(w.stream) orelse return error.UnknownStream, .ids = ids };
            b.wslots[i] = wins[i].slot;
            b.widths[i] = @intCast(rows);
            total += rows;
        }
        const r = b.runner;
        r.taps_on = b.taps_ring != null;
        defer r.taps_on = false;
        try r.verifyMulti(wins[0..windows.len]);
        var row: u32 = 0;
        for (windows, out, 0..) |w, o, i| {
            for (0..b.widths[i]) |j| {
                const logits = if (w.stream.sampling != null) try r.rowLogits(row) else null;
                o.sampled[j] = if (w.stream.sampling) |sp| try b.pick(b.pick_ctx, logits.?, w.positions[j], sp) else r.block[row];
                row += 1;
            }
            @memcpy(o.drafts, w.tokens);
        }
        b.count = windows.len;
    }

    /// Each window keeps a prefix of its chain.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != b.count or paths.len != b.count) return error.NothingToKeep;
        var keep: [16]u32 = undefined;
        for (paths, 0..) |p, i| {
            if (p.len == 0) return error.NothingToKeep;
            for (p, 0..) |row, j| if (row != j) return error.TreePath;
            keep[i] = @intCast(p.len);
        }
        b.count = 0;
        try b.keepRows(keep[0..paths.len]);
    }

    fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        b.settle() catch {};
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
    }

    /// A lone drafted stream in slot 0 on DFlash2 trees until it ends (false) or another stream arrives and the lane
    /// core takes it over (true; its state stands in slot 0, its pending token the last it committed).
    pub fn lone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const b = self(ctx);
        if (b.drafter == null) return error.DraftNotAttached;
        try b.settle();
        if (b.owners[0] != null) return error.StreamBusy;
        b.owners[0] = s;
        var handed = false;
        defer if (!handed) {
            b.owners[0] = null;
        };
        try b.prefillSlot(s, 0, true);
        const pending = try b.draw(s, 0, b.first_logits[0], s.prompt_len);
        _ = try s.commit(b.gpa, &.{pending}, &.{});
        hooks.committed(hooks.ctx);
        handed = try b.alone(s, 0, pending, hooks);
        return handed;
    }

    /// The stream the lane core was left with, alone again in its slot: the drafter's context restarts on the taps of
    /// its last rows (RING at most), then DFlash2 trees as `lone`.
    pub fn resumeLone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const b = self(ctx);
        const d = b.drafter orelse return error.DraftNotAttached;
        try b.settle();
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const r = b.runner;
        try r.select(slot);
        var handed = false;
        defer if (!handed) {
            b.owners[slot] = null;
        };
        const n: u32 = if (b.taps_ring != null and b.ring_end[slot] == r.pos) @min(r.pos, RING) else 0;
        d.restart(r.pos - n);
        if (n > 0) {
            const at: u32 = @intCast((r.pos - n) % RING);
            const first = @min(n, RING - at);
            const tr = b.taps_ring.?;
            try xq.copy(&r.d, r.acts.taps, xq.off(tr, (@as(usize, slot) * RING + at) * TAP_ROW), @as(usize, first) * TAP_ROW);
            if (first < n) try xq.copy(&r.d, xq.off(r.acts.taps, @as(usize, first) * TAP_ROW), xq.off(tr, @as(usize, slot) * RING * TAP_ROW), @as(usize, n - first) * TAP_ROW);
            try d.addTaps(r.acts.taps, n);
        }
        handed = try b.alone(s, slot, s.pending orelse return error.NoPending, hooks);
        return handed;
    }

    /// DFlash2 trees for `s` in `slot` from `pending` (committed, not fed) until it ends (false) or another stream
    /// arrives (true: its cache settled, `pending` and `cache_len` set for the lane core).
    fn alone(b: *Lanes, s: *lanes.Stream, slot: u32, pending0: u32, hooks: api.LoneHooks) !bool {
        const d = b.drafter.?;
        const r = b.runner;
        var pending = pending0;
        var out: [16]u32 = undefined;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                s.cache_len = r.pos;
                s.pending = pending;
                return true;
            }
            if (s.isCancelled()) return error.Cancelled;
            r.taps_on = true;
            defer r.taps_on = false;
            var sampled: Sampled = .{ .b = b, .s = s.sampling orelse .{ .seed = 0 } };
            const drawer: ?xq.drafter.Draw = if (s.sampling) |sp| .{ .ctx = &sampled, .row = Sampled.row, .gumbel = Sampled.gumbel, .temperature = sp.temperature } else null;
            const round = try xq.drafter.treeRound(d, pending, b.block - 1, b.nodes, drawer, &out);
            try b.ringPush(slot, r.acts.taps, @intCast(round.tokens.len));
            s.rounds += 1;
            s.drafted += round.drafted;
            s.accepted += round.accepted;
            var landed: [17]u32 = undefined;
            const n = round.tokens.len - 1;
            @memcpy(landed[0..n], round.tokens[1..]);
            landed[n] = round.terminal;
            _ = try s.commit(b.gpa, landed[0 .. n + 1], &.{});
            pending = round.terminal;
            hooks.committed(hooks.ctx);
        }
        return false;
    }

    const Sampled = struct {
        b: *Lanes,
        s: api.Sampling,
        fn row(c_: *anyopaque, row_: u32, position: u64) !u32 {
            const c: *@This() = @ptrCast(@alignCast(c_));
            return c.b.pick(c.b.pick_ctx, try c.b.runner.rowLogits(row_), position, c.s);
        }
        fn gumbel(c_: *anyopaque, position: u64, id: u64) f64 {
            const c: *@This() = @ptrCast(@alignCast(c_));
            return -@log(-@log(lanes.sampling.uniform(c.s.seed, position, id)));
        }
    };
};
