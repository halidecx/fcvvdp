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
const m = @import("main.zig");

const Image = m.Image;

pub const PAMLoadError = error{
    FileOpenFailed,
    ReadFailed,
    InvalidFormat,
    InvalidDimensions,
    UnsupportedMaxValue,
    HeaderNotFound,
    InsufficientData,
    UnknownTupltype,
    OutOfMemory,
};

pub fn loadPAM(allocator: std.mem.Allocator, io: std.Io, path: []const u8) PAMLoadError!Image {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.FileOpenFailed;
    defer file.close(io);

    const file_size = file.length(io) catch return error.ReadFailed;

    const file_buffer = allocator.alloc(u8, file_size) catch return error.OutOfMemory;
    defer allocator.free(file_buffer);

    const bytes_read = file.readPositionalAll(io, file_buffer, 0) catch return error.ReadFailed;
    if (bytes_read != file_size) return error.ReadFailed;

    const end_header_idx = std.mem.indexOf(u8, file_buffer, "ENDHDR\n") orelse return error.HeaderNotFound;
    const header_data = file_buffer[0..end_header_idx];
    const data_start = end_header_idx + 7; // "ENDHDR\n" is 7 bytes

    if (!std.mem.startsWith(u8, header_data, "P7"))
        return error.InvalidFormat;

    var width: usize = 0;
    var height: usize = 0;
    var depth: u8 = 0;
    var maxval: usize = 0;
    var tupltype: ?[]const u8 = null;

    var lines = std.mem.tokenizeAny(u8, header_data, "\r\n");
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (std.mem.startsWith(u8, line, "#")) continue;
        if (std.mem.startsWith(u8, line, "WIDTH")) {
            var value_it = std.mem.tokenizeAny(u8, line[5..], " \t");
            if (value_it.next()) |value|
                width = std.fmt.parseInt(usize, value, 10) catch return error.InvalidDimensions;
        } else if (std.mem.startsWith(u8, line, "HEIGHT")) {
            var value_it = std.mem.tokenizeAny(u8, line[6..], " \t");
            if (value_it.next()) |value|
                height = std.fmt.parseInt(usize, value, 10) catch return error.InvalidDimensions;
        } else if (std.mem.startsWith(u8, line, "DEPTH")) {
            var value_it = std.mem.tokenizeAny(u8, line[5..], " \t");
            if (value_it.next()) |value|
                depth = std.fmt.parseInt(u8, value, 10) catch return error.InvalidDimensions;
        } else if (std.mem.startsWith(u8, line, "MAXVAL")) {
            var value_it = std.mem.tokenizeAny(u8, line[6..], " \t");
            if (value_it.next()) |value|
                maxval = std.fmt.parseInt(usize, value, 10) catch return error.UnsupportedMaxValue;
        } else if (std.mem.startsWith(u8, line, "TUPLTYPE")) {
            const value = std.mem.trim(u8, line[8..], " \t");
            if (value.len > 0) tupltype = value;
        }
    }

    if (width == 0 or height == 0 or depth == 0 or maxval == 0)
        return error.InvalidDimensions;

    if (maxval > 65535)
        return error.UnsupportedMaxValue;

    const channels = depth;

    const bytes_per_sample: usize = if (maxval <= 255) 1 else 2;
    const expected_data_size = @as(usize, width) * @as(usize, height) * @as(usize, channels) * bytes_per_sample;

    if (data_start + expected_data_size > file_buffer.len)
        return error.InsufficientData;

    const raw_data = file_buffer[data_start..][0..expected_data_size];

    const output_size = @as(usize, width) * @as(usize, height) * @as(usize, channels);
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

pub fn isPAM(path: []const u8) bool {
    if (path.len < 4) return false;
    const ext = path[path.len - 4 ..];
    return std.ascii.eqlIgnoreCase(ext, ".pam");
}
