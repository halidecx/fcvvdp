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
const y4m = @import("y4m.zig");
const ppm = @import("ppm.zig");
const pam = @import("pam.zig");
const c = @cImport({
    @cInclude("third-party/spng.h");
    @cInclude("src/cvvdp.h");
});

const print = std.debug.print;

pub const Image = struct {
    width: usize,
    height: usize,
    channels: u8, // 1=Gray, 3=RGB, 4=RGBA
    data: []u8, // interleaved, row-major

    pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

pub inline fn requantize16to8(x: u16) u8 {
    const t: u32 = @as(u32, x) * 255 + 32768;
    return @intCast((t + (t >> 16)) >> 16);
}

pub fn loadPNG(allocator: std.mem.Allocator, path: []const u8) !Image {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const size = try file.getEndPos();
    const buf = try allocator.alloc(u8, size);
    defer allocator.free(buf);
    _ = try file.readAll(buf);

    const ctx = c.spng_ctx_new(0);
    if (ctx == null) return error.FailedCreateContext;
    defer c.spng_ctx_free(ctx);

    if (c.spng_set_png_buffer(ctx, buf.ptr, buf.len) != 0)
        return error.SetBufferFailed;

    var ihdr: c.struct_spng_ihdr = undefined;
    if (c.spng_get_ihdr(ctx, &ihdr) != 0)
        return error.GetHeaderFailed;

    // always decode to RGBA8
    const fmt: c_int = c.SPNG_FMT_RGBA8;
    var out_size: usize = 0;
    if (c.spng_decoded_image_size(ctx, fmt, &out_size) != 0) return error.ImageSizeFailed;

    const out_buf = try allocator.alloc(u8, out_size);
    errdefer allocator.free(out_buf);

    if (c.spng_decode_image(ctx, out_buf.ptr, out_size, fmt, 0) != 0) return error.DecodeFailed;

    return .{
        .width = ihdr.width,
        .height = ihdr.height,
        .channels = 4,
        .data = out_buf,
    };
}

fn hasExtension(path: []const u8, ext: []const u8) bool {
    if (path.len < ext.len) return false;
    const tail = path[path.len - ext.len ..];
    return std.ascii.eqlIgnoreCase(tail, ext);
}

fn yuv420ToRgb8FromFrame(allocator: std.mem.Allocator, frame: y4m.Frame) ![]u8 {
    if (frame.chroma != .yuv420) return error.UnsupportedY4MChroma;

    const width = frame.width;
    const height = frame.height;

    // Convert to RGB8 using a simple full-range BT.601-like YUV->RGB.
    // This is intended for metric input, not broadcast-accurate color management.
    const rgb = try allocator.alloc(u8, width * height * 3);
    errdefer allocator.free(rgb);

    const cw = width / 2;

    const clampU8 = struct {
        fn f(x: i32) u8 {
            if (x < 0) return 0;
            if (x > 255) return 255;
            return @intCast(x);
        }
    }.f;

    if (frame.bit_depth == .b8) {
        const y_plane: []const u8 = frame.y;
        const u_plane: []const u8 = frame.u;
        const v_plane: []const u8 = frame.v;

        for (0..height) |yy| {
            for (0..width) |xx| {
                const yv: i32 = y_plane[yy * width + xx];
                const uv: i32 = u_plane[(yy / 2) * cw + (xx / 2)];
                const vv: i32 = v_plane[(yy / 2) * cw + (xx / 2)];

                const u_off = uv - 128;
                const v_off = vv - 128;

                const r = yv + ((359 * v_off) >> 8);
                const g = yv - ((88 * u_off + 183 * v_off) >> 8);
                const b = yv + ((454 * u_off) >> 8);

                const i = (yy * width + xx) * 3;
                rgb[i + 0] = clampU8(r);
                rgb[i + 1] = clampU8(g);
                rgb[i + 2] = clampU8(b);
            }
        }
    } else if (frame.bit_depth == .b10) {
        // Y4M 10-bit is commonly stored as 16-bit little-endian words with values in [0, 1023].
        const y16 = try frame.yAsU16LE();
        const u_plane16 = try frame.uAsU16LE();
        const v_plane16 = try frame.vAsU16LE();

        for (0..height) |yy| {
            for (0..width) |xx| {
                // Downshift from 10-bit to 8-bit by dropping low bits.
                const yv: i32 = @intCast(y16[yy * width + xx] >> 2);
                const uv: i32 = @intCast(u_plane16[(yy / 2) * cw + (xx / 2)] >> 2);
                const vv: i32 = @intCast(v_plane16[(yy / 2) * cw + (xx / 2)] >> 2);

                const u_off = uv - 128;
                const v_off = vv - 128;

                const r = yv + ((359 * v_off) >> 8);
                const g = yv - ((88 * u_off + 183 * v_off) >> 8);
                const b = yv + ((454 * u_off) >> 8);

                const i = (yy * width + xx) * 3;
                rgb[i + 0] = clampU8(r);
                rgb[i + 1] = clampU8(g);
                rgb[i + 2] = clampU8(b);
            }
        }
    } else {
        return error.UnsupportedY4MBitDepth;
    }

    return rgb;
}

pub fn loadY4MFirstFrameAsRGB(allocator: std.mem.Allocator, path: []const u8) !Image {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var dec = try y4m.Decoder.init(allocator, file);
    defer dec.deinit();

    const frame_opt = try dec.readFrame();
    if (frame_opt == null) return error.EmptyY4M;

    var frame = frame_opt.?;
    defer frame.deinit(allocator);

    const rgb = try yuv420ToRgb8FromFrame(allocator, frame);

    return .{
        .width = frame.width,
        .height = frame.height,
        .channels = 3,
        .data = rgb,
    };
}

pub fn toRGB8(allocator: std.mem.Allocator, img: Image) ![]u8 {
    const pixels = img.width * img.height;
    const rgb = try allocator.alloc(u8, pixels * 3);

    switch (img.channels) {
        3 => { // direct copy
            @memcpy(rgb, img.data);
        },
        4 => { // strip alpha
            for (0..pixels) |i| {
                rgb[i * 3 + 0] = img.data[i * 4 + 0];
                rgb[i * 3 + 1] = img.data[i * 4 + 1];
                rgb[i * 3 + 2] = img.data[i * 4 + 2];
            }
        },
        1 => { // grayscale to RGB
            for (0..pixels) |i| {
                const g = img.data[i];
                rgb[i * 3 + 0] = g;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = g;
            }
        },
        else => return error.UnsupportedChannelCount,
    }
    return rgb;
}

fn parseDisplayModel(name: []const u8) c.FcvvdpDisplayModel {
    if (std.mem.eql(u8, name, "fhd")) {
        return c.CVVDP_DISPLAY_STANDARD_FHD;
    } else if (std.mem.eql(u8, name, "4k")) {
        return c.CVVDP_DISPLAY_STANDARD_4K;
    } else if (std.mem.eql(u8, name, "hdr_pq")) {
        return c.CVVDP_DISPLAY_STANDARD_HDR_PQ;
    } else if (std.mem.eql(u8, name, "hdr_hlg")) {
        return c.CVVDP_DISPLAY_STANDARD_HDR_HLG;
    } else if (std.mem.eql(u8, name, "hdr_linear")) {
        return c.CVVDP_DISPLAY_STANDARD_HDR_LINEAR;
    } else if (std.mem.eql(u8, name, "hdr_dark")) {
        return c.CVVDP_DISPLAY_STANDARD_HDR_DARK;
    } else if (std.mem.eql(u8, name, "hdr_zoom")) {
        return c.CVVDP_DISPLAY_STANDARD_HDR_LINEAR_ZOOM;
    }
    return c.CVVDP_DISPLAY_STANDARD_FHD; // default
}

fn displayModelName(model: c.FcvvdpDisplayModel) []const u8 {
    return switch (model) {
        c.CVVDP_DISPLAY_STANDARD_FHD => "Standard FHD (24\", 200 cd/m², office lighting)",
        c.CVVDP_DISPLAY_STANDARD_4K => "Standard 4K (30\", 200 cd/m², office lighting)",
        c.CVVDP_DISPLAY_STANDARD_HDR_PQ => "Standard HDR PQ (30\" 4K, 1500 cd/m², low light)",
        c.CVVDP_DISPLAY_STANDARD_HDR_HLG => "Standard HDR HLG (30\" 4K, 1500 cd/m², low light)",
        c.CVVDP_DISPLAY_STANDARD_HDR_LINEAR => "Standard HDR Linear (30\" 4K, 1500 cd/m², low light)",
        c.CVVDP_DISPLAY_STANDARD_HDR_DARK => "Standard HDR Dark (30\" 4K, 1500 cd/m², dark room)",
        c.CVVDP_DISPLAY_STANDARD_HDR_LINEAR_ZOOM => "Standard HDR Zoom (30\" 4K, 10000 cd/m², close viewing)",
        else => "Unknown",
    };
}

fn printUsage() void {
    print("\n", .{});
    print(
        \\usage: fcvvdp [options] <reference> <distorted>
        \\
        \\compare two images/videos using the CVVDP perceptual quality metric
        \\
        \\options:
        \\  -m, --model <name>
        \\      display model to use (fhd, 4k, hdr_pq, hdr_hlg, hdr_linear,
        \\      hdr_dark, hdr_zoom); default: fhd
        \\  -v, --verbose
        \\      show verbose output with display parameters
        \\  -j, --json
        \\      output result as JSON
        \\  -h, --help
        \\      show this help message
    , .{});
    print("\n\n\x1b[37msRGB PNG, PPM, PGM, PAM, or Y4M input expected\x1b[0m\n", .{});
}

fn loadImage(allocator: std.mem.Allocator, path: []const u8) !Image {
    if (hasExtension(path, ".png"))
        return loadPNG(allocator, path)
    else if (ppm.isPPM(path))
        return try ppm.loadPPM(allocator, path)
    else if (pam.isPAM(path))
        return try pam.loadPAM(allocator, path)
    else
        return error.UnsupportedFileFormat;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    print("\x1b[38;5;208mfcvvdp\x1b[0m by Halide Compression, LLC | {s}\n", .{c.cvvdp_version_string()});

    var args = std.process.args();
    _ = args.next();

    var ref_filename: ?[]const u8 = null;
    var dis_filename: ?[]const u8 = null;
    var display_model: c_uint = @intCast(c.CVVDP_DISPLAY_STANDARD_FHD);
    var verbose = false;
    var json_output = false;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose"))
            verbose = true
        else if (std.mem.eql(u8, arg, "-j") or std.mem.eql(u8, arg, "--json"))
            json_output = true
        else if ((std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model"))) {
            if (args.next()) |model_arg|
                display_model = parseDisplayModel(model_arg)
            else {
                print("Error: Missing argument for --model\n", .{});
                printUsage();
                return error.InvalidArguments;
            }
        } else if (arg[0] != '-') {
            if (ref_filename == null)
                ref_filename = arg
            else if (dis_filename == null)
                dis_filename = arg
            else {
                print("Error: Too many input files specified\n", .{});
                return error.TooManyFiles;
            }
        } else {
            print("Error: Unknown option '{s}'\n", .{arg});
            printUsage();
            return error.UnknownOption;
        }
    }

    if (ref_filename == null or dis_filename == null) {
        print("Error: Two input image files are required\n", .{});
        printUsage();
        return error.MissingFiles;
    }

    const ref_is_y4m = hasExtension(ref_filename.?, ".y4m");
    const dis_is_y4m = hasExtension(dis_filename.?, ".y4m");

    if (ref_is_y4m != dis_is_y4m) {
        print("Error: Both inputs must be Y4M if one is\n", .{});
        return error.MismatchedInputTypes;
    }

    if (!ref_is_y4m) {
        if (verbose and !json_output)
            print("Loading reference: {s}\n", .{ref_filename.?});

        var ref_img = try loadImage(allocator, ref_filename.?);
        defer ref_img.deinit(allocator);

        if (verbose and !json_output)
            print("Loading distorted: {s}\n", .{dis_filename.?});

        var dis_img = try loadImage(allocator, dis_filename.?);
        defer dis_img.deinit(allocator);

        if (ref_img.width != dis_img.width or ref_img.height != dis_img.height) {
            print("Error: Image dimensions do not match\n", .{});
            print("  Reference: {d}x{d}\n", .{ ref_img.width, ref_img.height });
            print("  Distorted: {d}x{d}\n", .{ dis_img.width, dis_img.height });
            return error.DimensionMismatch;
        }

        const ref_rgb = try toRGB8(allocator, ref_img);
        defer allocator.free(ref_rgb);

        const dis_rgb = try toRGB8(allocator, dis_img);
        defer allocator.free(dis_rgb);

        if (verbose and !json_output) {
            print("Image size: {d}x{d}\n", .{ ref_img.width, ref_img.height });
            print("Display model: {s}\n", .{displayModelName(display_model)});
        }

        var ref_cvvdp = c.FcvvdpImage{
            .data = ref_rgb.ptr,
            .width = @intCast(ref_img.width),
            .height = @intCast(ref_img.height),
            .stride = @intCast(ref_img.width * 3),
            .format = c.CVVDP_PIXEL_FORMAT_RGB_UINT8,
            .colorspace = c.CVVDP_COLORSPACE_SRGB,
        };

        var dis_cvvdp = c.FcvvdpImage{
            .data = dis_rgb.ptr,
            .width = @intCast(dis_img.width),
            .height = @intCast(dis_img.height),
            .stride = @intCast(dis_img.width * 3),
            .format = c.CVVDP_PIXEL_FORMAT_RGB_UINT8,
            .colorspace = c.CVVDP_COLORSPACE_SRGB,
        };

        if (verbose and !json_output)
            print("Computing CVVDP metric...\n", .{});

        var result: c.FcvvdpResult = undefined;
        const err = c.cvvdp_compare_images(&ref_cvvdp, &dis_cvvdp, display_model, null, &result);

        if (err != c.CVVDP_OK) {
            print("Error: CVVDP comparison failed: {s}\n", .{c.cvvdp_error_string(err)});
            return error.CVVDPError;
        }

        if (json_output) {
            print(
                \\{{
                \\  "jod": {d:.6},
                \\  "quality": {d:.6},
                \\  "reference": "{s}",
                \\  "distorted": "{s}",
                \\  "width": {d},
                \\  "height": {d}
                \\}}
            , .{ result.jod, result.quality, ref_filename.?, dis_filename.?, ref_cvvdp.width, ref_cvvdp.height });
        } else {
            if (verbose) {
                print("\n", .{});
                print("Results:\n", .{});
                print("  JOD Score:    {d:.4}\n", .{result.jod});
                print("  Quality (Q):  {d:.4}\n", .{result.quality});
                print("\n", .{});

                print("Interpretation: ", .{});
                if (result.jod >= 9.5)
                    print("Images are virtually identical\n", .{})
                else if (result.jod >= 9.0)
                    print("Barely visible difference\n", .{})
                else if (result.jod >= 8.0)
                    print("Slight visible difference\n", .{})
                else if (result.jod >= 7.0)
                    print("Noticeable but acceptable difference\n", .{})
                else if (result.jod >= 5.0)
                    print("Clearly visible, somewhat annoying difference\n", .{})
                else if (result.jod >= 3.0)
                    print("Very visible, annoying difference\n", .{})
                else
                    print("Large, unacceptable difference\n", .{});
            } else print("JOD: {d:.4}\n", .{result.jod});
        }
        return;
    }

    if (verbose and !json_output)
        print("Opening reference video: {s}\n", .{ref_filename.?});

    const ref_file = try std.fs.cwd().openFile(ref_filename.?, .{});
    defer ref_file.close();
    var ref_dec = try y4m.Decoder.init(allocator, ref_file);
    defer ref_dec.deinit();

    if (verbose and !json_output)
        print("Opening distorted video: {s}\n", .{dis_filename.?});

    const dis_file = try std.fs.cwd().openFile(dis_filename.?, .{});
    defer dis_file.close();
    var dis_dec = try y4m.Decoder.init(allocator, dis_file);
    defer dis_dec.deinit();

    if (ref_dec.header.width != dis_dec.header.width or ref_dec.header.height != dis_dec.header.height) {
        print("Error: Video dimensions do not match\n", .{});
        print("  Reference: {d}x{d}\n", .{ ref_dec.header.width, ref_dec.header.height });
        print("  Distorted: {d}x{d}\n", .{ dis_dec.header.width, dis_dec.header.height });
        return error.DimensionMismatch;
    }

    if (verbose and !json_output) {
        print("Video size: {d}x{d}\n", .{ ref_dec.header.width, ref_dec.header.height });
        print("Display model: {s}\n", .{displayModelName(display_model)});
    }

    const fps: f32 = if (ref_dec.header.fps_num != 0)
        @as(f32, @floatFromInt(ref_dec.header.fps_num)) / @as(f32, @floatFromInt(ref_dec.header.fps_den))
    else
        0.0;

    var ctx_ptr: ?*c.FcvvdpCtx = null;
    const create_err = c.cvvdp_create(
        @intCast(ref_dec.header.width),
        @intCast(ref_dec.header.height),
        fps,
        display_model,
        null,
        &ctx_ptr,
    );
    if (create_err != c.CVVDP_OK or ctx_ptr == null) {
        print("Error: CVVDP context creation failed: {s}\n", .{c.cvvdp_error_string(create_err)});
        return error.CVVDPError;
    }
    defer c.cvvdp_destroy(ctx_ptr.?);

    if (verbose and !json_output)
        print("Processing frames...\n", .{});

    var frame_index: usize = 0;
    var result: c.FcvvdpResult = undefined;

    while (true) {
        const ref_frame_opt = try ref_dec.readFrame();
        const dis_frame_opt = try dis_dec.readFrame();

        if (ref_frame_opt == null and dis_frame_opt == null) break;
        if (ref_frame_opt == null or dis_frame_opt == null) {
            print("Error: Video frame count does not match\n", .{});
            return error.FrameCountMismatch;
        }

        var ref_frame = ref_frame_opt.?;
        defer ref_frame.deinit(allocator);
        var dis_frame = dis_frame_opt.?;
        defer dis_frame.deinit(allocator);

        // Convert both frames to RGB8
        const ref_rgb = try yuv420ToRgb8FromFrame(allocator, ref_frame);
        defer allocator.free(ref_rgb);
        const dis_rgb = try yuv420ToRgb8FromFrame(allocator, dis_frame);
        defer allocator.free(dis_rgb);

        var ref_cvvdp = c.FcvvdpImage{
            .data = ref_rgb.ptr,
            .width = @intCast(ref_frame.width),
            .height = @intCast(ref_frame.height),
            .stride = @intCast(ref_frame.width * 3),
            .format = c.CVVDP_PIXEL_FORMAT_RGB_UINT8,
            .colorspace = c.CVVDP_COLORSPACE_SRGB,
        };

        var dis_cvvdp = c.FcvvdpImage{
            .data = dis_rgb.ptr,
            .width = @intCast(dis_frame.width),
            .height = @intCast(dis_frame.height),
            .stride = @intCast(dis_frame.width * 3),
            .format = c.CVVDP_PIXEL_FORMAT_RGB_UINT8,
            .colorspace = c.CVVDP_COLORSPACE_SRGB,
        };

        const proc_err = c.cvvdp_process_frame(ctx_ptr.?, &ref_cvvdp, &dis_cvvdp, &result);
        if (proc_err != c.CVVDP_OK) {
            print("Error: CVVDP frame processing failed at frame {d}: {s}\n", .{ frame_index, c.cvvdp_error_string(proc_err) });
            return error.CVVDPError;
        }

        frame_index += 1;
    }

    if (frame_index == 0) return error.EmptyY4M;

    if (json_output) {
        print(
            \\{{
            \\  "jod": {d:.6},
            \\  "quality": {d:.6},
            \\  "reference": "{s}",
            \\  "distorted": "{s}",
            \\  "width": {d},
            \\  "height": {d},
            \\  "frames": {d}
            \\}}
        , .{ result.jod, result.quality, ref_filename.?, dis_filename.?, ref_dec.header.width, ref_dec.header.height, frame_index });
    } else {
        if (verbose) {
            print("\n", .{});
            print("Results:\n", .{});
            print("  Frames:       {d}\n", .{frame_index});
            print("  JOD Score:    {d:.4}\n", .{result.jod});
            print("  Quality (Q):  {d:.4}\n", .{result.quality});
            print("\n", .{});

            print("Interpretation: ", .{});
            if (result.jod >= 9.5)
                print("Videos are virtually identical\n", .{})
            else if (result.jod >= 9.0)
                print("Barely visible difference\n", .{})
            else if (result.jod >= 8.0)
                print("Slight visible difference\n", .{})
            else if (result.jod >= 7.0)
                print("Noticeable but acceptable difference\n", .{})
            else if (result.jod >= 5.0)
                print("Clearly visible, somewhat annoying difference\n", .{})
            else if (result.jod >= 3.0)
                print("Very visible, annoying difference\n", .{})
            else
                print("Large, unacceptable difference\n", .{});
        } else print("JOD: {d:.4}\n", .{result.jod});
    }
    return;
}
