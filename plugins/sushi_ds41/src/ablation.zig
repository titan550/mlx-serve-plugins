//! Directional ablation on the plugin side: the request's direction from `abi.Forward.ablation` armed on the model for
//! one call. The calibration tap's files are `tap.zig`.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = @import("sushi/mlx.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");

pub const Refusal = error{ AblationStrength, AblationSlots };

/// Whether the kit this plugin is built against has `Forward.ablation`; without it the plugin serves none.
pub const supported = @hasField(abi.Forward, "ablation");

/// One slot's ablation as the forward hands it; `name` is the profile's directory name (empty on a host that predates
/// the field), which a per-layer bank is keyed by.
pub const Spec = struct { directions: abi.Array, strength: f32, name: []const u8 };

/// The host null-terminates the name in a fixed buffer, so the plain `*const u8` is a C string; span it at the NUL.
fn cstr(p: *const u8) []const u8 {
    return std.mem.span(@as([*:0]const u8, @ptrCast(p)));
}

/// The forward's ablation, or null: a host older than `Forward.ablation` passes a shorter struct, and a strength of 0
/// is no ablation.
pub fn of(f: *const abi.Forward) Refusal!?Spec {
    if (comptime !supported) return null else {
        if (f.size < @offsetOf(abi.Forward, "ablation") + @sizeOf(?*const abi.Ablation)) return null;
        const a = f.ablation orelse return null;
        if (a.strengths.len != 1 or a.strengths.len != f.slots.len) return error.AblationSlots;
        const s = a.strengths.slice()[0];
        if (!(s >= 0 and s <= 30)) return error.AblationStrength; // CR-E: >1 needed to override a strong built-in refusal
        // The profile name, only where the host's `Ablation` is wide enough to carry it; an older host gets "".
        const name = if (a.size >= @offsetOf(abi.Ablation, "name") + @sizeOf(?*const u8)) if (a.name) |n| cstr(n) else "" else "";
        return if (s == 0) null else .{ .directions = a.directions, .strength = s, .name = name };
    }
}

/// Arms `a` on the model for one call. The bank `r`/`col` (one direction per capture site) is loaded from the
/// profile's `bank.f32` on first use and cached on the model; a profile that ships none falls back to the single
/// shared direction broadcast to every site. Only `s` is per-call and freed by `disarm`.
pub fn arm(mdl: *dsv.Model, a: Spec, dim: usize, s: mlx.mlx_stream) (error{AblationShape} || anyerror)!void {
    const d: mlx.mlx_array = .{ .ctx = a.directions.ctx };
    if (d.ctx == null) return error.AblationShape;
    const sh = mlx.getShape(d);
    if (sh.len != 2 or sh[0] != 1 or sh[1] != dim or mlx.mlx_array_dtype(d) != .float32) return error.AblationShape;
    const n_sites: usize = mdl.cfg.layers + 1;
    const bank = try bankFor(mdl, a.name, d, n_sites, dim, s);
    mdl.ablation = .{ .r = bank.r, .col = bank.col, .s = mlx.mlx_array_new_float(a.strength) };
}

/// The bank a profile names: its per-layer `bank.f32` if one is present beside the profile, else the single shared
/// direction broadcast to every site. Built once per name and kept on the model (freed in `Model.deinit`).
fn bankFor(mdl: *dsv.Model, name: []const u8, d: mlx.mlx_array, n_sites: usize, dim: usize, s: mlx.mlx_stream) !dsv.AblationBank {
    if (mdl.ablation_banks.get(name)) |b| return b;
    const key = try mdl.gpa.dupe(u8, name);
    errdefer mdl.gpa.free(key);
    const b = try loadBank(mdl, key, d, n_sites, dim, s);
    mdl.ablation_banks.put(mdl.gpa, key, b) catch |e| {
        _ = mlx.mlx_array_free(b.r);
        _ = mlx.mlx_array_free(b.col);
        return e;
    };
    return mdl.ablation_banks.get(name).?;
}

/// Reads `<profiles-dir>/<name>/bank.f32` (`[n_sites, dim]` f32) into MLX arrays; a profile with no bank file gets
/// the single shared direction on every site instead.
fn loadBank(mdl: *dsv.Model, name: []const u8, d: mlx.mlx_array, n_sites: usize, dim: usize, s: mlx.mlx_stream) !dsv.AblationBank {
    const path = bankPath(name);
    const r = try bankArray(mdl, path, n_sites, dim, d, s);
    errdefer _ = mlx.mlx_array_free(r);
    var col = try bankArray(mdl, path, n_sites, dim, d, s);
    errdefer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, r, &[_]c_int{ @intCast(n_sites), @intCast(dim), 1 }, 3, s));
    return .{ .r = r, .col = col, .n_sites = @intCast(n_sites) };
}

/// The bank's `r` (or `col` view of it) from `path`; the single direction broadcast to every site when there is none.
fn bankArray(mdl: *dsv.Model, path: []const u8, n_sites: usize, dim: usize, d: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, mdl.gpa, .limited(n_sites * dim * 4)) catch return broadcastBank(d, n_sites, dim, s);
    defer mdl.gpa.free(data);
    if (data.len != n_sites * dim * 4) return broadcastBank(d, n_sites, dim, s);
    const r = mlx.mlx_array_new_data(data.ptr, &[_]c_int{ @intCast(n_sites), @intCast(dim) }, 2, .float32);
    errdefer _ = mlx.mlx_array_free(r);
    return r;
}

/// No bank file (or an unreadable one): the original single-direction method, the one direction on every site.
fn broadcastBank(d: mlx.mlx_array, n_sites: usize, dim: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_reshape(&row, d, &[_]c_int{@intCast(dim)}, 1, s));
    try mlx.check(mlx.mlx_broadcast_to(&r, row, &[_]c_int{ @intCast(n_sites), @intCast(dim) }, 2, s));
    return r;
}

/// `<HOME>/.mlx-serve/ablation/<name>/bank.f32`, the directory a profile's manifest and direction live in.
fn bankPath(name: []const u8) []const u8 {
    const home = std.c.getenv("HOME") orelse return "";
    return std.fmt.allocPrintSentinel(std.heap.c_allocator, "{s}/.mlx-serve/ablation/{s}/bank.f32", .{ home, name }, 0) catch "";
}

/// Whether a forward of `mode` feeds the calibration tap: prompt passes only, never a one-token decode step.
pub fn tapsMode(mode: abi.Mode) bool {
    return mode == .prefill;
}

/// Drops the call's ablation; the graphs built under it keep their own references. The bank `r`/`col` are the model's
/// (freed in `Model.deinit`), so only the per-call `s` is freed here.
pub fn disarm(mdl: *dsv.Model) void {
    const ab = mdl.ablation orelse return;
    _ = mlx.mlx_array_free(ab.s);
    mdl.ablation = null;
}

test "sushi_ds41 ablation: the tap reads prompt passes only" {
    try std.testing.expect(tapsMode(.prefill));
    for ([_]abi.Mode{ .decode, .verify, .batched_rows }) |mode| try std.testing.expect(!tapsMode(mode));
}

test "sushi_ds41 ablation: a host's forward without the field, a null one, or strength 0 is no ablation; bad tables refuse" {
    if (comptime supported) try ofCases() else return error.SkipZigTest;
}

fn ofCases() !void {
    const dir: abi.Array = .{};
    const one = [_]u32{0};
    var f: abi.Forward = .{ .mode = .decode, .rows = 1, .tokens = .{}, .slots = .of(&one) };
    try std.testing.expect((try of(&f)) == null);
    var zero = [_]f32{0};
    var ab: abi.Ablation = .{ .directions = dir, .strengths = .of(&zero) };
    f.ablation = &ab;
    try std.testing.expect((try of(&f)) == null);
    var half = [_]f32{0.5};
    ab.strengths = .of(&half);
    try std.testing.expectEqual(@as(f32, 0.5), (try of(&f)).?.strength);
    var mid = [_]f32{2.0};
    ab.strengths = .of(&mid);
    try std.testing.expectEqual(@as(f32, 2), (try of(&f)).?.strength);
    var big = [_]f32{31};
    ab.strengths = .of(&big);
    try std.testing.expectError(error.AblationStrength, of(&f));
    var two = [_]f32{ 1, 1 };
    ab.strengths = .of(&two);
    try std.testing.expectError(error.AblationSlots, of(&f));
    f.size = @offsetOf(abi.Forward, "ablation");
    try std.testing.expect((try of(&f)) == null);
}
