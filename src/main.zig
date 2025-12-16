const std = @import("std");
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
    print(
        \\fcvvdp | {s}
        \\
        \\usage: fcvvdp [options] <reference.png> <distorted.png>
        \\
        \\compare two PNG images using the CVVDP perceptual quality metric
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
        \\
    , .{
        c.cvvdp_version_string(),
    });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

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
        print("Error: Two input PNG files are required\n", .{});
        printUsage();
        return error.MissingFiles;
    }

    if (verbose and !json_output)
        print("Loading reference: {s}\n", .{ref_filename.?});

    var ref_img = try loadPNG(allocator, ref_filename.?);
    defer ref_img.deinit(allocator);

    if (verbose and !json_output)
        print("Loading distorted: {s}\n", .{dis_filename.?});

    var dis_img = try loadPNG(allocator, dis_filename.?);
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
}
