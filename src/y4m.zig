// Copyright © 2026, Halide Compression, LLC.
// All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
const std = @import("std");

pub const Chroma = enum {
    yuv420, // 4:2:0
};

pub const BitDepth = enum(u8) {
    b8 = 8,
    b10 = 10,
};

pub const Frame = struct {
    width: usize,
    height: usize,
    chroma: Chroma,
    bit_depth: BitDepth,

    /// Luma plane. For 8-bit: []u8 sample-per-byte. For 10-bit: []u8 containing little-endian u16 words.
    y: []u8,
    /// Chroma U plane. Same packing rules as `y`.
    u: []u8,
    /// Chroma V plane. Same packing rules as `y`.
    v: []u8,

    pub fn deinit(self: *Frame, allocator: std.mem.Allocator) void {
        allocator.free(self.y);
        allocator.free(self.u);
        allocator.free(self.v);
        self.* = undefined;
    }

    pub fn yAsU16LE(self: Frame) ![]align(1) const u16 {
        if (self.bit_depth != .b10) return error.WrongBitDepth;
        if ((self.y.len & 1) != 0) return error.BadPlaneSize;
        const ptr: [*]align(1) const u16 = @ptrCast(self.y.ptr);
        return ptr[0 .. self.y.len / 2];
    }
    pub fn uAsU16LE(self: Frame) ![]align(1) const u16 {
        if (self.bit_depth != .b10) return error.WrongBitDepth;
        if ((self.u.len & 1) != 0) return error.BadPlaneSize;
        const ptr: [*]align(1) const u16 = @ptrCast(self.u.ptr);
        return ptr[0 .. self.u.len / 2];
    }
    pub fn vAsU16LE(self: Frame) ![]align(1) const u16 {
        if (self.bit_depth != .b10) return error.WrongBitDepth;
        if ((self.v.len & 1) != 0) return error.BadPlaneSize;
        const ptr: [*]align(1) const u16 = @ptrCast(self.v.ptr);
        return ptr[0 .. self.v.len / 2];
    }
};

pub const Header = struct {
    width: usize,
    height: usize,
    fps_num: u32 = 0,
    fps_den: u32 = 1,
    chroma: Chroma = .yuv420,
    bit_depth: BitDepth = .b8,
};

pub const Decoder = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    offset: u64,

    header: Header,
    /// Bytes per frame payload (excluding "FRAME\n" line).
    frame_bytes: usize,

    /// Scratch buffer for reading header/frame lines.
    line_buf: [4096]u8,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File) !Decoder {
        var d: Decoder = .{
            .allocator = allocator,
            .io = io,
            .file = file,
            .offset = 0,
            .header = .{ .width = 0, .height = 0 },
            .frame_bytes = 0,
            .line_buf = undefined,
        };

        try d.readStreamHeader();
        d.frame_bytes = computeFrameBytes(d.header);

        return d;
    }

    pub fn deinit(self: *Decoder) void {
        // Decoder does not own the file; caller closes it.
        self.* = undefined;
    }

    /// Returns null on clean EOF (no partial frame).
    pub fn readFrame(self: *Decoder) !?Frame {
        // Read and validate the FRAME header line.
        const line = try self.readLineOrEof();
        if (line == null) return null;

        if (!std.mem.startsWith(u8, line.?, "FRAME")) return error.BadFrameMarker;
        // Y4M frame line is "FRAME" optionally followed by space-separated tags, then '\n'.
        // We ignore tags, but we must have ended at '\n' already via readLineOrEof.

        // Read planes
        const w = self.header.width;
        const h = self.header.height;

        const y_bytes = planeBytes(self.header.bit_depth, w, h);
        const cw = chromaWidth(w, self.header.chroma);
        const ch = chromaHeight(h, self.header.chroma);
        const c_bytes = planeBytes(self.header.bit_depth, cw, ch);

        const y = try self.allocator.alloc(u8, y_bytes);
        errdefer self.allocator.free(y);
        const u = try self.allocator.alloc(u8, c_bytes);
        errdefer self.allocator.free(u);
        const v = try self.allocator.alloc(u8, c_bytes);
        errdefer self.allocator.free(v);

        try self.readExact(y);
        try self.readExact(u);
        try self.readExact(v);

        return Frame{
            .width = w,
            .height = h,
            .chroma = self.header.chroma,
            .bit_depth = self.header.bit_depth,
            .y = y,
            .u = u,
            .v = v,
        };
    }

    fn readStreamHeader(self: *Decoder) !void {
        const line_opt = try self.readLineOrEof();
        if (line_opt == null) return error.UnexpectedEof;

        const line = line_opt.?;
        if (!std.mem.startsWith(u8, line, "YUV4MPEG2")) return error.NotY4M;

        // Tokenize by spaces
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        const magic = it.next() orelse return error.NotY4M;
        if (!std.mem.eql(u8, magic, "YUV4MPEG2")) return error.NotY4M;

        var have_w = false;
        var have_h = false;

        // Defaults (per y4m convention)
        self.header.fps_num = 0;
        self.header.fps_den = 1;
        self.header.chroma = .yuv420;
        self.header.bit_depth = .b8;

        while (it.next()) |tok| {
            if (tok.len == 0) continue;

            switch (tok[0]) {
                'W' => {
                    self.header.width = try parseUsize(tok[1..]);
                    have_w = true;
                },
                'H' => {
                    self.header.height = try parseUsize(tok[1..]);
                    have_h = true;
                },
                'F' => {
                    // F<num>:<den>
                    const frac = tok[1..];
                    const colon = std.mem.indexOfScalar(u8, frac, ':') orelse return error.BadFps;
                    self.header.fps_num = try parseU32(frac[0..colon]);
                    self.header.fps_den = try parseU32(frac[colon + 1 ..]);
                    if (self.header.fps_den == 0) return error.BadFps;
                },
                'I' => {
                    // Interlace: p/t/b/m ?
                    // Only progressive supported for now.
                    if (tok.len < 2) return error.BadInterlace;
                    if (tok[1] != 'p') return error.UnsupportedInterlace;
                },
                'C' => {
                    try self.parseChromaToken(tok[1..]);
                },
                else => {
                    // A= (aspect), X* (extended tags), etc: ignore.
                },
            }
        }

        if (!have_w or !have_h) return error.MissingDimensions;
        if (self.header.width == 0 or self.header.height == 0) return error.InvalidDimensions;

        // 4:2:0 requires even dims for clean subsampling; allow odd but compute floor/ceil?
        // Most y4m expects even; we enforce even to avoid ambiguity.
        if ((self.header.width & 1) != 0 or (self.header.height & 1) != 0) {
            return error.UnsupportedOddDimensions;
        }
    }

    fn parseChromaToken(self: *Decoder, ctoken: []const u8) !void {
        // Examples:
        //   "420jpeg"
        //   "420p10"
        //   "420"
        // There are many y4m chroma strings; we accept 420* and determine bit-depth.
        if (!std.mem.startsWith(u8, ctoken, "420")) return error.UnsupportedChroma;

        self.header.chroma = .yuv420;

        // Determine bit depth.
        // Common patterns:
        // - 8-bit: "420", "420jpeg", "420mpeg2", "420paldv", etc
        // - 10-bit: "420p10"
        // We'll search for "p10" (case-insensitive).
        if (indexOfAsciiNoCase(ctoken, "p10") != null) {
            self.header.bit_depth = .b10;
        } else {
            self.header.bit_depth = .b8;
        }
    }

    fn readLineOrEof(self: *Decoder) !?[]const u8 {
        var index: usize = 0;

        while (true) {
            var byte_buf: [1]u8 = undefined;
            const n = try self.file.readPositionalAll(self.io, &byte_buf, self.offset);
            if (n == 0) {
                if (index == 0) return null;
                // EOF mid-line: treat as error (header/frame lines must end with '\n')
                return error.UnexpectedEof;
            }
            self.offset += 1;
            const b = byte_buf[0];

            if (b == '\n') break;
            // Spec: lines are ASCII, but we keep raw bytes.
            if (index >= self.line_buf.len) return error.LineTooLong;
            self.line_buf[index] = b;
            index += 1;
        }

        return self.line_buf[0..index];
    }

    fn readExact(self: *Decoder, buf: []u8) !void {
        var total: usize = 0;
        while (total < buf.len) {
            const n = try self.file.readPositionalAll(self.io, buf[total..], self.offset + total);
            if (n == 0) return error.UnexpectedEof;
            total += n;
        }
        self.offset += total;
    }
};

fn computeFrameBytes(h: Header) usize {
    const yb = planeBytes(h.bit_depth, h.width, h.height);
    const cw = chromaWidth(h.width, h.chroma);
    const ch = chromaHeight(h.height, h.chroma);
    const cb = planeBytes(h.bit_depth, cw, ch);
    return yb + cb + cb;
}

fn chromaWidth(width: usize, chroma: Chroma) usize {
    return switch (chroma) {
        .yuv420 => width / 2,
    };
}

fn chromaHeight(height: usize, chroma: Chroma) usize {
    return switch (chroma) {
        .yuv420 => height / 2,
    };
}

fn bytesPerSample(bit_depth: BitDepth) usize {
    return switch (bit_depth) {
        .b8 => 1,
        .b10 => 2, // stored as 16-bit words
    };
}

fn planeBytes(bit_depth: BitDepth, width: usize, height: usize) usize {
    return width * height * bytesPerSample(bit_depth);
}

fn parseUsize(s: []const u8) !usize {
    if (s.len == 0) return error.BadNumber;
    return std.fmt.parseInt(usize, s, 10) catch return error.BadNumber;
}

fn parseU32(s: []const u8) !u32 {
    if (s.len == 0) return error.BadNumber;
    return std.fmt.parseInt(u32, s, 10) catch return error.BadNumber;
}

fn indexOfAsciiNoCase(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return null;

    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var ok = true;
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            const a = std.ascii.toLower(haystack[i + j]);
            const b = std.ascii.toLower(needle[j]);
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) return i;
    }
    return null;
}
