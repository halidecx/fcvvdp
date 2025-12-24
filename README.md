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
  wall_time          19.1s  ±  250ms    18.9s  … 19.4s           0 ( 0%)        0%
  peak_rss           1.00GB ± 44.0MB     974MB … 1.05GB          0 ( 0%)        0%
  cpu_cycles          752G  ± 10.9G      741G  …  763G           0 ( 0%)        0%
  instructions        360G  ±  962M      360G  …  361G           0 ( 0%)        0%
  cache_references   2.75G  ± 25.4M     2.72G  … 2.77G           0 ( 0%)        0%
  cache_misses        897M  ± 13.6M      884M  …  911M           0 ( 0%)        0%
  branch_misses       105M  ± 1.62M      103M  …  106M           0 ( 0%)        0%
Benchmark 2 (3 runs): ./fcvvdp -m fhd fm360p.y4m fm360p_x264.y4m
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          16.3s  ± 45.3ms    16.3s  … 16.4s           0 ( 0%)        ⚡- 14.7% ±  2.1%
  peak_rss           86.5MB ±  217KB    86.3MB … 86.7MB          0 ( 0%)        ⚡- 91.4% ±  7.0%
  cpu_cycles         84.2G  ± 89.6M     84.1G  … 84.2G           0 ( 0%)        ⚡- 88.8% ±  2.3%
  instructions        262G  ± 5.82M      262G  …  262G           0 ( 0%)        ⚡- 27.3% ±  0.4%
  cache_references   1.49G  ± 5.25M     1.48G  … 1.49G           0 ( 0%)        ⚡- 45.8% ±  1.5%
  cache_misses        370M  ± 1.90M      368M  …  372M           0 ( 0%)        ⚡- 58.7% ±  2.4%
  branch_misses      8.48M  ± 49.3K     8.43M  … 8.53M           0 ( 0%)        ⚡- 91.9% ±  2.5%
```

fcvvdp uses 91% less RAM, 89% fewer CPU cycles, and is almost 15% faster in
terms of wall clock time. In terms of user time, fcvvdp is ~15x more efficient.

## Usage

Compilation requires:

- [zlib-rs](https://github.com/trifectatechfoundation/zlib-rs)
- [libunwind](https://github.com/libunwind/libunwind)
- [Zig](https://ziglang.org/) 0.15.x

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
