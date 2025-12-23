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

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "strip symbols from the binary, defaults to false") orelse false;
    const flto = b.option(bool, "flto", "enable Link Time Optimization, defaults to false") orelse false;
    const options = b.addOptions();

    // 'libcvvdp.a' static lib
    const cvvdp = b.addLibrary(.{
        .name = "cvvdp",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .strip = strip,
        }),
    });
    const cvvdp_sources = [_][]const u8{
        "src/cvvdp.c",
    };
    const cvvdp_flags = [_][]const u8{
        "-std=c23",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-O3",
    };
    cvvdp.root_module.addCSourceFiles(.{
        .files = &cvvdp_sources,
        .flags = if (flto) &cvvdp_flags ++ &[_][]const u8{"-flto=thin"} else &cvvdp_flags,
    });
    cvvdp.root_module.addIncludePath(b.path("."));
    b.installArtifact(cvvdp);

    // cvvdp.h
    cvvdp.installHeader(b.path("src/cvvdp.h"), "cvvdp.h");

    // libspng
    const spng = b.addLibrary(.{
        .name = "spng",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .strip = strip,
        }),
    });
    const spng_sources = [_][]const u8{
        "third-party/spng.c",
    };
    spng.root_module.addCSourceFiles(.{
        .files = &spng_sources,
        .flags = if (flto) &cvvdp_flags ++ &[_][]const u8{"-flto=thin"} else &cvvdp_flags,
    });
    spng.root_module.addIncludePath(b.path("third-party/"));

    // 'fcvvdp' executable
    const cvvdpenc = b.addExecutable(.{
        .name = "fcvvdp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .strip = strip,
        }),
    });
    cvvdpenc.root_module.addOptions("build_opts", options);
    cvvdpenc.root_module.addIncludePath(b.path("."));
    cvvdpenc.root_module.linkLibrary(cvvdp);
    cvvdpenc.root_module.linkLibrary(spng);
    cvvdpenc.root_module.linkSystemLibrary("z_rs", .{ .preferred_link_mode = .static });
    cvvdpenc.root_module.linkSystemLibrary("unwind", .{ .preferred_link_mode = .static });
    b.installArtifact(cvvdpenc);
}
