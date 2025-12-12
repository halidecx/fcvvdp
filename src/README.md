# CVVDP - C Implementation

A standalone C implementation of the ColorVideoVDP (CVVDP) perceptual video/image quality metric.

## Overview

CVVDP is a perceptual quality metric that estimates the probability that a human observer would notice differences between two images or video frames. Unlike simple metrics like PSNR or SSIM, CVVDP models the human visual system including:

- Contrast sensitivity function (CSF) - how sensitivity varies with spatial frequency and luminance
- Temporal filtering - how the eye responds to changes over time in video
- Masking - how one pattern can hide another
- Display model - how content appears on a specific display

This C implementation is based on the GPU-accelerated version in Vship but runs entirely on the CPU without any GPU dependencies.

## Features

- Pure C99 implementation with no GPU requirements
- Single image or video sequence comparison
- Multiple display model presets (SDR and HDR)
- Custom display parameter support
- JOD (Just-Objectionable-Difference) quality score output
- PNG comparison command-line utility

## Building

### Requirements

- C99 compatible compiler (GCC, Clang, MSVC)
- libpng (optional, for PNG comparison tool)
- Math library (-lm)

### Build Commands

```bash
# Build library and tools
make

# Build without libpng dependency (uses stb_image)
make stb

# Build with debug symbols
make debug

# Install to system
sudo make install

# Install to user directory
make install PREFIX=~/.local
```

### Manual Build

```bash
# Build library
gcc -O2 -std=c99 -c cvvdp.c -o cvvdp.o
ar rcs libcvvdp.a cvvdp.o

# Build PNG comparison tool
gcc -O2 -std=c99 -o cvvdp_png_compare cvvdp_png_compare.c -L. -lcvvdp -lpng -lm
```

## Usage

### Command Line Tool

```bash
# Compare two PNG images
./cvvdp_png_compare reference.png distorted.png

# With verbose output
./cvvdp_png_compare reference.png distorted.png --verbose

# Using a specific display model
./cvvdp_png_compare reference.png distorted.png --model hdr_pq

# JSON output
./cvvdp_png_compare reference.png distorted.png --json
```

### Display Models

| Model | Description |
|-------|-------------|
| `fhd` | 24" FullHD monitor, 200 cd/m², office lighting (default) |
| `4k` | 30" 4K monitor, 200 cd/m², office lighting |
| `hdr_pq` | 30" 4K HDR, 1500 cd/m², low light |
| `hdr_hlg` | 30" 4K HDR HLG, 1500 cd/m², low light |
| `hdr_linear` | 30" 4K HDR linear, 1500 cd/m², low light |
| `hdr_dark` | 30" 4K HDR, 1500 cd/m², dark room |
| `hdr_zoom` | 30" 4K HDR, 10000 cd/m², close viewing |

### Library API

```c
#include "cvvdp.h"

// Single image comparison
FcvvdpImage reference = {
    .data = ref_pixels,
    .width = 1920,
    .height = 1080,
    .stride = 1920 * 3,
    .format = CVVDP_PIXEL_FORMAT_RGB_UINT8,
    .colorspace = CVVDP_COLORSPACE_SRGB
};

FcvvdpImage distorted = {
    .data = dis_pixels,
    .width = 1920,
    .height = 1080,
    .stride = 1920 * 3,
    .format = CVVDP_PIXEL_FORMAT_RGB_UINT8,
    .colorspace = CVVDP_COLORSPACE_SRGB
};

FcvvdpResult result;
FcvvdpError err = cvvdp_compare_images(&reference, &distorted,
                                          CVVDP_DISPLAY_STANDARD_FHD,
                                          NULL, &result);

if (err == CVVDP_OK) {
    printf("JOD Score: %.4f\n", result.jod);
}
```

### Video Sequence Comparison

```c
// Create context for video processing
FcvvdpCtx* ctx;
FcvvdpError err = cvvdp_create(1920, 1080, 30.0f, // width, height, fps
                                  CVVDP_DISPLAY_STANDARD_FHD,
                                  NULL, &ctx);

// Process each frame
for (int frame = 0; frame < num_frames; frame++) {
    FcvvdpImage ref_frame = /* load reference frame */;
    FcvvdpImage dis_frame = /* load distorted frame */;

    FcvvdpResult result;
    cvvdp_process_frame(ctx, &ref_frame, &dis_frame, &result);

    // result.jod contains cumulative score up to this frame
    // Only the final frame's score should be used for video
}

// The last result.jod is the final video quality score
cvvdp_destroy(ctx);
```

## Interpreting Results

CVVDP returns a JOD (Just-Objectionable-Difference) score on a 0-10 scale:

| Score | Interpretation |
|-------|----------------|
| 10.0 | Images are identical |
| 9.0 - 10.0 | Barely visible difference |
| 8.0 - 9.0 | Slight visible difference |
| 7.0 - 8.0 | Noticeable but acceptable |
| 5.0 - 7.0 | Clearly visible, somewhat annoying |
| 3.0 - 5.0 | Very visible, annoying difference |
| < 3.0 | Large, unacceptable difference |

## Pixel Formats

The library supports several input pixel formats:

- `CVVDP_PIXEL_FORMAT_RGB_FLOAT` - 32-bit float per channel, [0,1] range for SDR
- `CVVDP_PIXEL_FORMAT_RGB_UINT8` - 8-bit unsigned int per channel, [0,255]
- `CVVDP_PIXEL_FORMAT_RGB_UINT16` - 16-bit unsigned int per channel, [0,65535]

## Custom Display Parameters

```c
FcvvdpDisplayParams custom = {
    .resolution_width = 3840,
    .resolution_height = 2160,
    .viewing_distance_meters = 0.8f,
    .diagonal_size_inches = 32.0f,
    .max_luminance = 400.0f,
    .contrast = 2000.0f,
    .ambient_light = 100.0f,
    .reflectivity = 0.01f,
    .is_hdr = false
};

FcvvdpResult result;
cvvdp_compare_images(&ref, &dis, CVVDP_DISPLAY_CUSTOM, &custom, &result);
```

## Differences from GPU Implementation

This C implementation closely follows the GPU version but with some simplifications:

1. **Single-threaded**: No GPU parallelism; suitable for batch processing or integration into larger pipelines
2. **Simplified masking**: Uses a simplified cross-channel masking model
3. **Approximate CSF**: Uses a parametric CSF model with luminance adaptation

The results should be comparable to the GPU version for most use cases, but exact numerical equivalence is not guaranteed.

## Files

| File | Description |
|------|-------------|
| `cvvdp.h` | Public API header |
| `cvvdp_internal.h` | Internal structures and constants |
| `cvvdp_csf_lut.h` | Contrast sensitivity function lookup tables |
| `cvvdp.c` | Main implementation |
| `cvvdp_png_compare.c` | PNG comparison utility |
| `Makefile` | Build system |

## License

This implementation is part of the Vship project. See the main project license for terms.

## References

- Mantiuk, R., et al. "ColorVideoVDP: A visual difference predictor for image, video and display distortions"
- Original CVVDP: https://github.com/gfxdisp/ColorVideoVDP
