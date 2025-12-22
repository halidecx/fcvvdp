# fcvvdp

A fast C implementation of the [CVVDP](https://github.com/gfxdisp/colorvideovdp)
metric ([arXiv](https://arxiv.org/html/2401.11485)) from the University of
Cambridge. More information about how CVVDP works according to this
implementation is provided [here](./doc/cvvdp.md).

## Benchmarks

Benchmarked using [`poop`](https://github.com/andrewrk/poop) on Linux, Core i7
13700k. Note that fcvvdp runs with one CPU thread here while cvvdp uses multiple
threads. This is a current limitation of fcvvdp, which does not yet support
multithreading.

```
poop "cvvdp -r fm360p.y4m -t fm360p_x264.y4m --display standard_fhd" "./fcvvdp -m fhd fm360p.y4m fm360p_x264.y4m"
Benchmark 1 (3 runs): cvvdp -r fm360p.y4m -t fm360p_x264.y4m --display standard_fhd
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          19.4s  ±  604ms    18.9s  … 20.1s           0 ( 0%)        0%
  peak_rss            984MB ± 14.1MB     975MB … 1.00GB          0 ( 0%)        0%
  cpu_cycles          766G  ± 17.0G      755G  …  785G           0 ( 0%)        0%
  instructions        363G  ± 1.03G      362G  …  364G           0 ( 0%)        0%
  cache_references   2.75G  ± 19.2M     2.74G  … 2.77G           0 ( 0%)        0%
  cache_misses        908M  ± 21.9M      888M  …  932M           0 ( 0%)        0%
  branch_misses       106M  ±  389K      105M  …  106M           0 ( 0%)        0%
Benchmark 2 (3 runs): ./fcvvdp -m fhd fm360p.y4m fm360p_x264.y4m
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          17.7s  ± 70.2ms    17.6s  … 17.8s           0 ( 0%)        ⚡-  8.7% ±  5.0%
  peak_rss           86.5MB ±  110KB    86.3MB … 86.5MB          0 ( 0%)        ⚡- 91.2% ±  2.3%
  cpu_cycles         91.4G  ± 60.5M     91.3G  … 91.4G           0 ( 0%)        ⚡- 88.1% ±  3.6%
  instructions        302G  ± 29.1M      302G  …  302G           0 ( 0%)        ⚡- 16.8% ±  0.5%
  cache_references   1.49G  ± 5.36M     1.49G  … 1.50G           0 ( 0%)        ⚡- 45.7% ±  1.2%
  cache_misses        369M  ± 4.11M      366M  …  374M           0 ( 0%)        ⚡- 59.3% ±  3.9%
  branch_misses      8.67M  ± 55.4K     8.62M  … 8.73M           0 ( 0%)        ⚡- 91.8% ±  0.6%
```

fcvvdp uses 91% less RAM, 88% fewer CPU cycles, and is almost 9% faster in terms
of wall clock time. In terms of user time, fcvvdp is ~15x more efficient.

## Usage

Compilation requires:

- [zlib-rs](https://github.com/trifectatechfoundation/zlib-rs)
- [libunwind](https://github.com/libunwind/libunwind)
- [Zig](https://ziglang.org/) 0.15.2

### Binary

0. Ensure all dependencies are installed
1. Run `zig build --release=fast` (add `-Dflto=true` for FLTO)
2. Your `fcvvdp` binary will be in `zig-out/bin/`

```sh
fcvvdp by Halide Compression, LLC | [version]

usage: fcvvdp [options] <reference.(png|y4m)> <distorted.(png|y4m)>

compare two images/videos using the CVVDP perceptual quality metric

options:
  -m, --model <name>
      display model to use (fhd, 4k, hdr_pq, hdr_hlg, hdr_linear,
      hdr_dark, hdr_zoom); default: fhd
  -v, --verbose
      show verbose output with display parameters
  -j, --json
      output result as JSON
  -h, --help
      show this help message
```

### Library

0. Ensure all dependencies are installed
1. Run `zig build --release=fast` (add `-Dflto=true` for FLTO)
2. The `libcvvdp` library will be in `zig-out/lib/`, alongside `cvvdp.h` in
   `zig-out/include/`.

Library usage is clearly defined in `cvvdp.h`.

## Credits

`fcvvdp` is under the [Apache 2.0 License](LICENSE). `fcvvdp` is developed by
[Halide Compression](https://halide.cx).

Special thanks to [Vship](https://github.com/Line-fr/Vship/releases), from which
this implementation was derived. Vship is under the
[MIT license](https://github.com/Line-fr/Vship#MIT-1-ov-file).
