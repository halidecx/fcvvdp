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

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "strip symbols from the binary, defaults to false") orelse false;
    const flto = b.option(bool, "flto", "enable Link Time Optimization, defaults to false") orelse false;
    const options = b.addOptions();

    // libspng
    const spng_dep = b.dependency("spng", .{
        .target = target,
        .optimize = optimize,
    });
    const spng = spng_dep.artifact("spng");

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("c_imports.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    translate_c.addIncludePath(b.path("."));
    translate_c.addIncludePath(spng.getEmittedIncludeTree());
    const c_module = translate_c.createModule();

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
        .flags = if (flto) &cvvdp_flags ++ &[_][]const u8{"-flto"} else &cvvdp_flags,
    });
    cvvdp.root_module.addIncludePath(b.path("."));
    b.installArtifact(cvvdp);

    // cvvdp.h
    cvvdp.installHeader(b.path("src/cvvdp.h"), "cvvdp.h");

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
    cvvdpenc.root_module.addImport("c", c_module);
    cvvdpenc.root_module.addIncludePath(b.path("."));
    cvvdpenc.root_module.linkLibrary(cvvdp);
    cvvdpenc.root_module.linkLibrary(spng);
    b.installArtifact(cvvdpenc);
}
