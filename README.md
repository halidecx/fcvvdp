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
  wall_time          19.6s  ±  568ms    19.2s  … 20.2s           0 ( 0%)        0%
  peak_rss           1.00GB ± 28.1MB     979MB … 1.03GB          0 ( 0%)        0%
  cpu_cycles          747G  ± 8.54G      741G  …  757G           0 ( 0%)        0%
  instructions        362G  ± 1.20G      361G  …  363G           0 ( 0%)        0%
  cache_references   2.77G  ± 46.9M     2.71G  … 2.81G           0 ( 0%)        0%
  cache_misses        899M  ± 11.7M      890M  …  912M           0 ( 0%)        0%
  branch_misses       107M  ± 1.80M      105M  …  109M           0 ( 0%)        0%
Benchmark 2 (3 runs): ./fcvvdp -m fhd fm360p.y4m fm360p_x264.y4m
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          16.1s  ± 56.2ms    16.0s  … 16.1s           0 ( 0%)        ⚡- 17.9% ±  4.7%
  peak_rss           86.7MB ±  109KB    86.6MB … 86.8MB          0 ( 0%)        ⚡- 91.4% ±  4.5%
  cpu_cycles         82.8G  ± 80.9M     82.8G  … 82.9G           0 ( 0%)        ⚡- 88.9% ±  1.8%
  instructions        255G  ± 30.0M      255G  …  255G           0 ( 0%)        ⚡- 29.6% ±  0.5%
  cache_references   1.49G  ± 6.43M     1.49G  … 1.50G           0 ( 0%)        ⚡- 46.1% ±  2.7%
  cache_misses        369M  ± 2.84M      365M  …  371M           0 ( 0%)        ⚡- 59.0% ±  2.2%
  branch_misses      8.50M  ± 62.3K     8.45M  … 8.57M           0 ( 0%)        ⚡- 92.1% ±  2.7%
```

fcvvdp uses 91% less RAM, 88% fewer CPU cycles, and is almost 18% faster in
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
