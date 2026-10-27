const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "strip symbols from the binary, defaults to false") orelse false;
    const flto = b.option(bool, "flto", "enable Link Time Optimization on ELF targets, defaults to false") orelse false;
    const enable_checkasm = b.option(bool, "checkasm", "fetch checkasm and install the checkasm binary, defaults to false") orelse false;
    const checkasm_step = b.step("checkasm", "Build and run checkasm with -Dcheckasm, pass arguments after --");
    if (flto and target.result.ofmt != .elf) {
        const unsupported = b.addFail("Zig 0.17 supports this project's LTO build only on ELF targets; omit -Dflto=true for macOS");
        b.default_step.dependOn(&unsupported.step);
        checkasm_step.dependOn(&unsupported.step);
        return;
    }
    const options = b.addOptions();

    const simpleimgio_dep = b.dependency("simpleimgio", .{
        .target = target,
        .optimize = optimize,
    });
    const simpleimgio = simpleimgio_dep.module("simpleimgio");

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

    const cvvdp = b.addLibrary(.{
        .name = "cvvdp",
        .use_lld = if (flto) true else null,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .strip = strip,
        }),
    });
    const cvvdp_sources = [_][]const u8{
        "src/cvvdp.c",
        "src/cvvdp_c.c",
        "src/cpu.c",
    };
    const cvvdp_flags = [_][]const u8{
        "-std=c23",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-O3",
        "-ffast-math",
    };
    const target_result = target.result;
    cvvdp.root_module.addCSourceFiles(.{
        .files = &cvvdp_sources,
        .flags = &cvvdp_flags,
    });

    if (target_result.cpu.arch == .x86_64) {
        var avx2_query = target.query;
        avx2_query.cpu_features_sub.removeFeature(@backingInt(std.Target.x86.Feature.avx2));
        avx2_query.cpu_features_sub.removeFeature(@backingInt(std.Target.x86.Feature.fma));
        avx2_query.cpu_features_add.addFeature(@backingInt(std.Target.x86.Feature.avx2));
        avx2_query.cpu_features_add.addFeature(@backingInt(std.Target.x86.Feature.fma));
        const avx2_module = b.createModule(.{
            .target = b.resolveTargetQuery(avx2_query),
            .optimize = optimize,
            .link_libc = true,
            .strip = strip,
        });
        avx2_module.addIncludePath(b.path("."));
        avx2_module.addCSourceFile(.{
            .file = b.path("src/cvvdp_avx2.c"),
            .flags = &cvvdp_flags,
        });
        const avx2 = b.addObject(.{
            .name = "cvvdp-avx2",
            .root_module = avx2_module,
            .use_lld = if (flto) true else null,
        });
        avx2.lto = if (flto) .full else null;
        cvvdp.root_module.addObject(avx2);
    } else if (target_result.cpu.arch == .aarch64) {
        cvvdp.root_module.addCSourceFile(.{
            .file = b.path("src/cvvdp_neon.c"),
            .flags = &cvvdp_flags,
        });
    }
    cvvdp.root_module.addIncludePath(b.path("."));
    cvvdp.lto = if (flto) .full else null;
    b.installArtifact(cvvdp);

    cvvdp.installHeader(b.path("src/cvvdp.h"), "cvvdp.h");

    const cvvdpenc = b.addExecutable(.{
        .name = "fcvvdp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .strip = strip,
        }),
        .use_llvm = true,
        .use_lld = if (flto) true else null,
    });
    cvvdpenc.root_module.addOptions("build_opts", options);
    cvvdpenc.root_module.addImport("c", c_module);
    cvvdpenc.root_module.addImport("simpleimgio", simpleimgio);
    cvvdpenc.root_module.addIncludePath(b.path("."));
    cvvdpenc.root_module.linkLibrary(cvvdp);
    cvvdpenc.root_module.linkLibrary(spng);
    cvvdpenc.lto = if (flto) .full else null;
    b.installArtifact(cvvdpenc);

    if (!enable_checkasm) {
        checkasm_step.dependOn(&b.addFail("the checkasm step requires -Dcheckasm").step);
    } else if (target_result.cpu.arch != .x86_64 and target_result.cpu.arch != .aarch64) {
        const unsupported = b.addFail("checkasm supports only x86_64 and aarch64 targets");
        b.default_step.dependOn(&unsupported.step);
        checkasm_step.dependOn(&unsupported.step);
    } else if (b.lazyDependency("checkasm", .{})) |checkasm_dep| {
        const checkasm_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        checkasm_module.addIncludePath(checkasm_dep.path("include"));
        checkasm_module.addIncludePath(checkasm_dep.path("src"));
        checkasm_module.addIncludePath(b.path("."));
        checkasm_module.addCSourceFiles(.{
            .root = checkasm_dep.path("src"),
            .files = &.{
                "checkasm.c",
                "cpu.c",
                "function.c",
                "perf.c",
                "perf/linux.c",
                "perf/macos_kperf.c",
                "signal.c",
                "stackguard.c",
                "stats.c",
                "utils.c",
                "arm/cpu.c",
                "perf/arm.c",
                "x86/cpu.c",
            },
            .flags = &.{"-std=gnu11"},
        });
        checkasm_module.addCSourceFiles(.{
            .files = &.{
                "tests/checkasm/checkasm.c",
                "tests/checkasm/color.c",
                "tests/checkasm/contrast.c",
                "tests/checkasm/masking.c",
                "tests/checkasm/pyramid.c",
                "tests/checkasm/temporal.c",
            },
            .flags = &.{ "-std=c23", "-Wall", "-Wextra", "-Wno-missing-field-initializers" },
        });
        if (target_result.cpu.arch == .x86_64) {
            const nasm_dep = b.lazyDependency("nasm", .{
                .target = b.graph.host,
                .optimize = .ReleaseFast,
            }) orelse return;
            const nasm = b.addRunArtifact(nasm_dep.artifact("nasm"));
            nasm.addArg("-f");
            nasm.addArg(switch (target_result.ofmt) {
                .macho => "macho64",
                .coff => "win64",
                else => "elf64",
            });
            nasm.addPrefixedDirectoryArg("-I", checkasm_dep.path("src"));
            nasm.addArg("-o");
            checkasm_module.addObjectFile(nasm.addOutputFileArg("checkasm_x86.o"));
            nasm.addFileArg(checkasm_dep.path("src/x86/checkasm.asm"));
        } else {
            checkasm_module.addCSourceFile(.{
                .file = checkasm_dep.path("src/arm/checkasm_64.S"),
            });
        }
        checkasm_module.linkLibrary(cvvdp);

        const checkasm = b.addExecutable(.{
            .name = "checkasm",
            .root_module = checkasm_module,
        });
        b.installArtifact(checkasm);
        const checkasm_run = b.addRunArtifact(checkasm);
        checkasm_run.addPassthruArgs();
        checkasm_step.dependOn(&checkasm_run.step);
    }
}
