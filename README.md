# fcvvdp

A fast C implementation of the [CVVDP](https://github.com/gfxdisp/colorvideovdp)
metric ([arXiv](https://arxiv.org/html/2401.11485)) from the University of
Cambridge. More information about how CVVDP works according to this
implementation is provided [here](./doc/cvvdp.md).

## Benchmarks

Benchmarked using [`poop`](https://github.com/andrewrk/poop) on Linux, Core i7
13700k. We compare cvvdp (multithreaded) to fcvvdp with & without task threading
on 302 frames of 360p video.

```
poop "cvvdp -r fm360p.y4m -t fm360p_x264.y4m --display standard_fhd" "./fcvvdp -t 1 -m fhd fm360p.y4m fm360p_x264.y4m" "./fcvvdp -t 8 -m fhd fm360p.y4m fm360p_x264.y4m"
Benchmark 1 (3 runs): cvvdp -r fm360p.y4m -t fm360p_x264.y4m --display standard_fhd
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          19.1s  ± 52.5ms    19.0s  … 19.1s           0 ( 0%)        0%
  peak_rss           1.03GB ± 11.1MB    1.02GB … 1.04GB          0 ( 0%)        0%
  cpu_cycles          777G  ± 8.37G      768G  …  784G           0 ( 0%)        0%
  instructions        357G  ± 1.18G      355G  …  358G           0 ( 0%)        0%
  cache_references   2.64G  ± 40.0M     2.60G  … 2.67G           0 ( 0%)        0%
  cache_misses        859M  ± 16.5M      844M  …  877M           0 ( 0%)        0%
  branch_misses      88.8M  ± 2.63M     86.7M  … 91.7M           0 ( 0%)        0%
Benchmark 2 (3 runs): ./fcvvdp -t 1 -m fhd fm360p.y4m fm360p_x264.y4m
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          14.9s  ± 10.3ms    14.9s  … 14.9s           0 ( 0%)        ⚡- 21.7% ±  0.5%
  peak_rss           90.4MB ±  159KB    90.3MB … 90.6MB          0 ( 0%)        ⚡- 91.2% ±  1.7%
  cpu_cycles         79.2G  ± 57.1M     79.1G  … 79.2G           0 ( 0%)        ⚡- 89.8% ±  1.7%
  instructions        246G  ± 32.1M      246G  …  246G           0 ( 0%)        ⚡- 30.9% ±  0.5%
  cache_references   1.29G  ±  138K     1.29G  … 1.29G           0 ( 0%)        ⚡- 51.2% ±  2.4%
  cache_misses        340M  ± 1.94M      339M  …  342M           0 ( 0%)        ⚡- 60.4% ±  3.1%
  branch_misses      9.06M  ± 71.5K     9.01M  … 9.14M           0 ( 0%)        ⚡- 89.8% ±  4.8%
Benchmark 3 (3 runs): ./fcvvdp -t 8 -m fhd fm360p.y4m fm360p_x264.y4m
  measurement          mean ± σ            min … max           outliers         delta
  wall_time          5.94s  ± 36.3ms    5.89s  … 5.96s           0 ( 0%)        ⚡- 68.9% ±  0.5%
  peak_rss           89.7MB ± 75.9KB    89.6MB … 89.7MB          0 ( 0%)        ⚡- 91.3% ±  1.7%
  cpu_cycles          126G  ±  753M      126G  …  127G           0 ( 0%)        ⚡- 83.7% ±  1.7%
  instructions        434G  ± 1.99G      432G  …  436G           0 ( 0%)        💩+ 21.6% ±  1.0%
  cache_references   1.21G  ± 7.24M     1.21G  … 1.22G           0 ( 0%)        ⚡- 54.1% ±  2.5%
  cache_misses        379M  ± 1.92M      378M  …  381M           0 ( 0%)        ⚡- 55.9% ±  3.1%
  branch_misses      13.8M  ±  101K     13.7M  … 13.9M           0 ( 0%)        ⚡- 84.4% ±  4.8%
```

fcvvdp uses over 91% less memory. With one thread, it is over 28% faster; with
multiple threads, fcvvdp is over 221% faster than the reference implementation.

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

### FFmpeg Patch

0. Clone FFmpeg:
```sh
git clone https://code.ffmpeg.org/FFmpeg/FFmpeg.git ffmpeg
cd ffmpeg
git checkout n8.0
```

1. Apply patch (change path to point to your cloned copy of fcvvdp):
```sh
git apply ~/fcvvdp/patches/0001-feat-fcvvdp-support.patch
```

2. Configure & build FFmpeg:
```sh
./configure --enable-fcvvdp
make -j$(nproc)
```

Example usage:
```
./ffmpeg -i src -i dst -lavfi "fcvvdp" -f null -
```

For more help, see `./ffmpeg --help filter=fcvvdp`.

## Credits

`fcvvdp` is under the [Apache 2.0 License](LICENSE). `fcvvdp` is developed by
[Halide Compression](https://halide.cx).

Special thanks to [Vship](https://github.com/Line-fr/Vship/releases), from which
this implementation was derived. Vship is under the
[MIT license](https://github.com/Line-fr/Vship#MIT-1-ov-file).
