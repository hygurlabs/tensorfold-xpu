//! The XPU 27B on the lane host: a stream a runner slot, shared rounds across slots (Runner.verifyMulti: each
//! stream's pending token and its drafts as a chain, 16 rows a forward), and a lone drafted stream on DFlash2 trees
//! (the host's `lone` driver) until another stream arrives; the one left when the others end resumes alone. With the
//! drafter each slot has its own DFlash2 context (every kept row's taps), so in shared rounds every stream holds a
//! DFlash2 chain for its next window (the core weighs it against copies and plain rows).
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const be = lanes.backend;
const Drafter = xq.drafter.Drafter;
const Allocator = std.mem.Allocator;

const ring = 64;
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
    wfirst: [16]u32 = undefined, // and its first row
    absorbed: [16]bool = @splat(false), // the window's kept rows already in its slot's drafter context
    chain: [xq.MAX_SLOTS][16]u32 = undefined, // each slot's DFlash2 chain for its next window
    chain_n: [xq.MAX_SLOTS]u32 = @splat(0),
    prompt_skip: u32 = 0, // a prefill's first rows the drafter never reads (further than WIN from the prompt's end)

    fn self(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn backend(b: *Lanes) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .tree = treeFn, .release = releaseFn } };
    }

    /// What the round loop reads: chains of up to 16 rows (a window's rows equal one-token steps bit for bit), every
    /// stream's window in one shared forward of 16 rows at most; with the drafter, DFlash2 chains as the head's drafts.
    pub fn facts(b: *const Lanes) lanes.Model {
        const head = b.drafter != null;
        return .{ .exact_width = 16, .mtp = head, .speculate = head, .speculate_early = false, .drafts = if (head) b.block - 1 else 0, .mtp_step_ms = if (head) 0.6 else 0, .draft_streams = head, .hidden_rows = true, .streams_exact = true, .batch_rows = 16, .max_streams = b.slots, .lone_copies = false };
    }

    /// Every forward's taps reach the drafter context of the slot it ran (with a drafter attached).
    pub fn initDrafts(b: *Lanes) void {
        b.runner.on_taps = .{ .ctx = b, .f = onTaps };
    }

    fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) anyerror!void {
        const b = self(ctx);
        const d = b.drafter.?;
        try d.use(b.runner.slot);
        const k = @min(n, b.prompt_skip);
        b.prompt_skip -= k;
        d.skip(k);
        if (k < n) try d.addTaps(xq.off(taps, @as(usize, k) * TAP_ROW), n - k);
    }

    /// The last verify's windows `which`: each one's first `keep[i]` rows' taps into its slot's drafter context, in
    /// one pass.
    fn absorb(b: *Lanes, which: []const usize, keep: []const u32) !void {
        const d = b.drafter orelse return;
        var items: [16]xq.drafter.Absorb = undefined;
        for (which, keep, items[0..which.len]) |i, k, *it| {
            it.* = .{ .slot = b.wslots[i], .taps = xq.off(b.runner.acts.taps, @as(usize, b.wfirst[i]) * TAP_ROW), .n = k };
            b.absorbed[i] = true;
        }
        try d.absorbMulti(items[0..which.len]);
    }

    /// Keeps the first `keep[i]` rows of the last verify's window i (their taps to the drafter, unless taken already).
    fn keepRows(b: *Lanes, keep: []const u32) !void {
        var which: [16]usize = undefined;
        var ks: [16]u32 = undefined;
        var n: usize = 0;
        for (keep, 0..) |k, i| if (!b.absorbed[i]) {
            which[n] = i;
            ks[n] = k;
            n += 1;
        };
        try b.absorb(which[0..n], ks[0..n]);
        @memset(&b.absorbed, false);
        try b.runner.keepMulti(keep);
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
    fn prefillSlot(b: *Lanes, s: *lanes.Stream, slot: u32) !void {
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + 17 > b.runner.context) return error.PromptTooLong;
        const r = b.runner;
        try r.select(slot);
        try r.reset();
        b.chain_n[slot] = 0;
        if (b.drafter) |d| {
            try d.use(slot);
            d.reset();
            r.taps_on = true;
            b.prompt_skip = @intCast(ids.len -| xq.drafter.WIN);
        }
        defer {
            r.taps_on = false;
            b.prompt_skip = 0;
        }
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
        try b.prefillSlot(s, slot);
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
        r.taps_on = b.drafter != null;
        defer r.taps_on = false;
        b.chain_n[slot] = 0;
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
            b.wfirst[i] = @intCast(total);
            b.widths[i] = @intCast(rows);
            total += rows;
        }
        const r = b.runner;
        r.taps_on = b.drafter != null;
        defer r.taps_on = false;
        @memset(&b.absorbed, false);
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

    /// Each asked stream's DFlash2 chain for its next window, all in one drafter pass: in a shared round before the
    /// keep, the kept rows' taps first (from this verify's), so each context ends at its stream's pending token.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        const d = b.drafter orelse return;
        var which: [16]usize = undefined;
        var ks: [16]u32 = undefined;
        var na: usize = 0;
        var items: [16]xq.drafter.Block = undefined;
        var asked: [16]u32 = undefined;
        var ni: usize = 0;
        var depth: u32 = 0;
        for (requests) |q| {
            const slot = b.slotOf(q.stream) orelse return error.UnknownStream;
            b.chain_n[slot] = 0;
            if (q.rows) |rows| if (b.count > 0) {
                const i = std.mem.indexOfScalar(u32, b.wslots[0..b.count], slot) orelse return error.UnknownStream;
                for (rows, 0..) |row, j| if (row != j) return error.TreePath;
                if (!b.absorbed[i]) {
                    which[na] = i;
                    ks[na] = @intCast(rows.len);
                    na += 1;
                }
            };
            if (q.depth == 0 or ni == items.len) continue;
            items[ni] = .{ .slot = slot, .pending = if (q.rows != null) q.follow[q.follow.len - 1] else try b.value(q.first orelse return error.NoFirstToken) };
            asked[ni] = @min(q.depth, b.block - 1);
            depth = @max(depth, asked[ni]);
            ni += 1;
        }
        try b.absorb(which[0..na], ks[0..na]);
        if (ni == 0) return;
        for (requests) |q| if (q.depth > 0) {
            try d.use(b.slotOf(q.stream).?);
            if (d.end() + 1 != q.position) return error.BadDraftPosition;
        };
        try d.latticeMulti(items[0..ni], depth);
        for (items[0..ni], asked[0..ni], 0..) |it, n, i| {
            d.take(i);
            d.chain(it.pending, n, b.chain[it.slot][0..n]);
            b.chain_n[it.slot] = n;
        }
    }

    /// A slot's held chain as host tokens (taken: the next window spends it).
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return null;
        const n = b.chain_n[slot];
        b.chain_n[slot] = 0;
        const tokens = try gpa.dupe(u32, b.chain[slot][0..n]);
        errdefer gpa.free(tokens);
        const parents = try gpa.alloc(i32, n);
        for (parents, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
        return .{ .count = n, .tokens = tokens, .parents = parents };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        b.settle() catch {};
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
        b.chain_n[slot] = 0;
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
        try b.prefillSlot(s, 0);
        const pending = try b.draw(s, 0, b.first_logits[0], s.prompt_len);
        _ = try s.commit(b.gpa, &.{pending}, &.{});
        hooks.committed(hooks.ctx);
        handed = try b.alone(s, 0, pending, hooks);
        return handed;
    }

    /// The stream the lane core was left with, alone again in its slot (its drafter context current), on DFlash2 trees
    /// as `lone`.
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
        try d.use(slot);
        if (d.end() != r.pos) d.restart(r.pos); // its context should end there: a cold one otherwise
        handed = try b.alone(s, slot, s.pending orelse return error.NoPending, hooks);
        return handed;
    }

    /// DFlash2 trees for `s` in `slot` from `pending` (committed, not fed) until it ends (false) or another stream
    /// arrives (true: its cache settled, `pending` and `cache_len` set for the lane core).
    fn alone(b: *Lanes, s: *lanes.Stream, slot: u32, pending0: u32, hooks: api.LoneHooks) !bool {
        const d = b.drafter.?;
        const r = b.runner;
        try d.use(slot);
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
