// Copyright © 2025, Halide Compression, LLC.
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
const m = @import("main.zig");

const Image = m.Image;

pub const PPMLoadError = error{
    FileOpenFailed,
    ReadFailed,
    InvalidFormat,
    InvalidDimensions,
    UnsupportedMaxValue,
    UnexpectedEof,
    OutOfMemory,
};

fn skipWhitespaceAndComments(data: []const u8, pos: *usize) void {
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (c == '#') {
            while (pos.* < data.len and data[pos.*] != '\n')
                pos.* += 1;
            if (pos.* < data.len) pos.* += 1;
        } else if (std.ascii.isWhitespace(c)) {
            pos.* += 1;
        } else break;
    }
}

fn readToken(data: []const u8, pos: *usize) ?[]const u8 {
    skipWhitespaceAndComments(data, pos);

    const start = pos.*;
    while (pos.* < data.len) {
        const c = data[pos.*];
        if (std.ascii.isWhitespace(c)) break;
        pos.* += 1;
    }

    if (pos.* == start) return null;
    return data[start..pos.*];
}

pub fn loadPPM(allocator: std.mem.Allocator, io: std.Io, path: []const u8) PPMLoadError!Image {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.FileOpenFailed;
    defer file.close(io);

    const file_size = file.length(io) catch return error.ReadFailed;
    const file_buffer = allocator.alloc(u8, file_size) catch return error.OutOfMemory;
    defer allocator.free(file_buffer);

    const bytes_read = file.readPositionalAll(io, file_buffer, 0) catch return error.ReadFailed;
    if (bytes_read != file_size) return error.ReadFailed;

    var pos: usize = 0;

    const magic = readToken(file_buffer, &pos) orelse return error.InvalidFormat;
    if (magic.len != 2 or magic[0] != 'P' or (magic[1] != '5' and magic[1] != '6')) {
        return error.InvalidFormat;
    }
    const is_grayscale = magic[1] == '5';
    const channels: u8 = if (is_grayscale) 1 else 3;

    const width_token = readToken(file_buffer, &pos) orelse return error.InvalidFormat;
    const width = std.fmt.parseInt(usize, width_token, 10) catch return error.InvalidFormat;
    if (width == 0) return error.InvalidDimensions;

    const height_token = readToken(file_buffer, &pos) orelse return error.InvalidFormat;
    const height = std.fmt.parseInt(usize, height_token, 10) catch return error.InvalidFormat;
    if (height == 0) return error.InvalidDimensions;

    const maxval_token = readToken(file_buffer, &pos) orelse return error.InvalidFormat;
    const maxval = std.fmt.parseInt(usize, maxval_token, 10) catch return error.InvalidFormat;
    if (maxval == 0 or maxval > 65535) return error.UnsupportedMaxValue;

    if (pos >= file_buffer.len) return error.UnexpectedEof;
    if (!std.ascii.isWhitespace(file_buffer[pos])) return error.InvalidFormat;
    pos += 1;

    const bytes_per_sample: usize = if (maxval <= 255) 1 else 2;
    const expected_size = width * height * channels * bytes_per_sample;

    if (pos + expected_size > file_buffer.len) return error.UnexpectedEof;

    const raw_data = file_buffer[pos..][0..expected_size];

    const output_size = width * height * channels;
    const output = allocator.alloc(u8, output_size) catch return error.OutOfMemory;
    errdefer allocator.free(output);

    if (maxval == 255) {
        @memcpy(output, raw_data);
    } else if (maxval == 65535) {
        for (0..output_size) |i| {
            const high = @as(u16, raw_data[i * 2]);
            const low = @as(u16, raw_data[i * 2 + 1]);
            const value16 = (high << 8) | low;
            output[i] = m.requantize16to8(value16);
        }
    } else for (0..output_size) |i| {
        const high = @as(u16, raw_data[i * 2]);
        const low = @as(u16, raw_data[i * 2 + 1]);
        const value16 = (high << 8) | low;
        const normalized = (@as(u32, value16) * 65535 + @as(u32, @intCast(maxval)) / 2) / @as(u32, @intCast(maxval));
        output[i] = m.requantize16to8(@intCast(normalized));
    }

    return .{
        .width = width,
        .height = height,
        .channels = channels,
        .data = output,
    };
}

pub fn isPPM(path: []const u8) bool {
    if (path.len < 4) return false;
    const ext = path[path.len - 4 ..];
    return std.ascii.eqlIgnoreCase(ext, ".ppm") or std.ascii.eqlIgnoreCase(ext, ".pgm");
}
