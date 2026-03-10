# fcvvdp

A fast C implementation of the [CVVDP](https://github.com/gfxdisp/colorvideovdp)
metric ([arXiv](https://arxiv.org/html/2401.11485)) from the University of
Cambridge. More information about how CVVDP works according to this
implementation is provided [here](./doc/cvvdp.md).

## Benchmarks

Benchmarked using [`poop`](https://github.com/andrewrk/poop) on Linux, Core i7
13700k. We compare cvvdp (multithreaded) to fcvvdp with & without task threading on 302 frames of 360p video.

```
poop "cvvdp -r fm360p.y4m -t fm360p_x264.y4m --display standard_fhd" "./fcvvdp -t 1 -m fhd fm360p.y4m fm360p_x264.y4m" "./fcvvdp -m fhd fm360p.y4m fm360p_x264.y4m"
Benchmark 1 (3 runs): cvvdp -r fm360p.y4m -t fm360p_x264.y4m --display standard_fhd
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          19.5s  ±  338ms    19.3s  … 19.9s           0 ( 0%)        0%
  peak_rss           1.04GB ± 6.86MB    1.03GB … 1.05GB          0 ( 0%)        0%
  cpu_cycles          793G  ± 8.80G      786G  …  803G           0 ( 0%)        0%
  instructions        358G  ± 3.58G      354G  …  360G           0 ( 0%)        0%
  cache_references   2.62G  ± 28.6M     2.59G  … 2.65G           0 ( 0%)        0%
  cache_misses        859M  ± 6.18M      855M  …  866M           0 ( 0%)        0%
  branch_misses      87.9M  ± 1.59M     86.0M  … 88.9M           0 ( 0%)        0%
Benchmark 2 (3 runs): ./fcvvdp -t 1 -m fhd fm360p.y4m fm360p_x264.y4m
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          16.5s  ± 28.3ms    16.4s  … 16.5s           0 ( 0%)        ⚡- 15.7% ±  2.8%
  peak_rss           89.7MB ±  236KB    89.5MB … 89.9MB          0 ( 0%)        ⚡- 91.4% ±  1.1%
  cpu_cycles         81.2G  ± 71.1M     81.2G  … 81.3G           0 ( 0%)        ⚡- 89.8% ±  1.8%
  instructions        246G  ± 32.7M      246G  …  246G           0 ( 0%)        ⚡- 31.2% ±  1.6%
  cache_references   1.21G  ± 4.15M     1.21G  … 1.22G           0 ( 0%)        ⚡- 53.6% ±  1.8%
  cache_misses        267M  ± 1.56M      266M  …  269M           0 ( 0%)        ⚡- 68.9% ±  1.2%
  branch_misses      9.23M  ± 52.4K     9.19M  … 9.29M           0 ( 0%)        ⚡- 89.5% ±  2.9%
Benchmark 3 (3 runs): ./fcvvdp -m fhd fm360p.y4m fm360p_x264.y4m
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          6.89s  ±  266ms    6.58s  … 7.06s           0 ( 0%)        ⚡- 64.7% ±  3.5%
  peak_rss           88.7MB ±  187KB    88.5MB … 88.8MB          0 ( 0%)        ⚡- 91.5% ±  1.1%
  cpu_cycles          173G  ± 3.57G      170G  …  177G           0 ( 0%)        ⚡- 78.1% ±  1.9%
  instructions        345G  ± 5.23G      340G  …  350G           0 ( 0%)          -  3.6% ±  2.8%
  cache_references   1.12G  ± 14.0M     1.11G  … 1.13G           0 ( 0%)        ⚡- 57.4% ±  2.0%
  cache_misses        300M  ± 15.6M      290M  …  318M           0 ( 0%)        ⚡- 65.1% ±  3.1%
  branch_misses      19.3M  ±  299K     19.0M  … 19.5M           0 ( 0%)        ⚡- 78.0% ±  2.9%
```

fcvvdp is almost 65% faster, and uses over 91% less memory.

## Usage

Compilation requires:

- [zlib-rs](https://github.com/trifectatechfoundation/zlib-rs)
- [libunwind](https://github.com/libunwind/libunwind)
- [Zig](https://ziglang.org/) 0.15.x
- macOS, Linux, or Unix-like operating system

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
