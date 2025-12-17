/*
 * Copyright © 2025, Halide Compression, LLC.
 * All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FCVVDP_INTERNAL_H
#define FCVVDP_INTERNAL_H

#include <stdlib.h>
#include "cvvdp.h"
#include "lut.h"

#define M_PI 3.14159265358979323846f
#define TAU (2.0 * M_PI)

#define CVVDP_MAX_LEVELS 14 // maximum pyramid levels
#define CVVDP_LUT_SIZE 32 // CSF LUT size
#define CVVDP_GAUSSIAN_SIZE 8 // gaussian kernel size
#define CVVDP_NUM_CHANNELS 4 // number of color channels
#define CVVDP_GAUSSIAN_SIZE 8 // Gaussian blur radius

static const float GAUSS_PYR_KERNEL[5] = {
    0.25f - 0.4f/2.0f, 0.25f, 0.4f, 0.25f, 0.25f - 0.4f/2.0f
};

// display model
typedef struct Display {
    unsigned resolution_width;
    unsigned resolution_height;
    float viewing_distance_meters;
    float diagonal_size_inches;
    float max_luminance;
    float contrast;
    float ambient_light;
    float reflectivity;
    float exposure;
    bool is_hdr;

    // computed: pixels per degree
    float ppd;

    // computed: black level
    float black_level;

    // computed: reflectivity level
    float refl_level;
} Display;

// temporal filter
typedef struct TemporalFilter {
    // convolution kernels for 4 channels
    float* kernel[4];
    // kernel size
    int size;
} TemporalFilter;

// temporal ring buffer for frame storage
typedef struct TemporalRingBuf {
    int width, height;

    // buffer for frames (3 planes per frame)
    float* data;

    // maximum number of frames that can be stored
    int max_frames;

    // current number of frames stored
    int num_frames;

    // index of most recent frame
    int current_index;

    // tf
    TemporalFilter filter;
} TemporalRingBuf;

// csf
typedef struct Csf {
    // sensitivity LUT: [num_bands * 4 channels * 32]
    float* log_S_LUT;

    // number of bands
    int num_bands;
} Csf;

// laplacian pyramid level
typedef struct PyramidLvl {
    int width, height;
    float frequency;

    // 4 channels per level
    float* data[4];
} PyramidLvl;

// laplacian pyramid
typedef struct Pyramid {
    PyramidLvl* levels;
    int num_levels;
    int base_width;
    int base_height;
    float ppd;
} Pyramid;

// gaussian blur handle
typedef struct Gaussian {
    float kernel[2 * CVVDP_GAUSSIAN_SIZE + 1];
    float kernel_integral[2 * CVVDP_GAUSSIAN_SIZE + 2];
} Gaussian;

// cvvdp context
typedef struct FcvvdpCtx {
    // configuration
    int width, height;
    float fps;
    Display display;

    // state
    TemporalRingBuf ring_ref;
    TemporalRingBuf ring_dis;
    Csf csf;
    Gaussian gaussian;

    // pyramid
    int num_bands;
    float* band_frequencies;

    // score accum
    int num_frames;
    double score_square_sum;

    // work buffers
    float* work_buffer;
    size_t work_buffer_size;

    // temp channel buffers
    float* Y_sustained;
    float* RG_sustained;
    float* YV_sustained;
    float* Y_transient;
} FcvvdpCtx;

#endif /* FCVVDP_INTERNAL_H */
