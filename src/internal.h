/*
 * Copyright © 2026, Halide Compression, LLC.
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
#include <pthread.h>
#include <stdatomic.h>
#include "cvvdp.h"
#include "lut.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#define TAU (2.0 * M_PI)

#define CVVDP_MAX_LEVELS 14 // maximum pyramid levels
#define CVVDP_LUT_SIZE 32 // CSF LUT size
#define CVVDP_GAUSSIAN_SIZE 8 // gaussian kernel size
#define CVVDP_NUM_CHANNELS 4 // number of color channels
#define CVVDP_GAUSSIAN_SIZE 8 // Gaussian blur radius
#define CVVDP_MAX_THREADS 32 // max thread cnt

static const float GAUSS_PYR_KERNEL[5] = {
    0.25f - 0.4f/2.0f, 0.25f, 0.4f, 0.25f, 0.25f - 0.4f/2.0f
};

typedef void (*CvvdpTaskFn)(void* user_data, int start, int end);

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

// thread pool
typedef struct CvvdpThreadPool {
    int worker_count;
    pthread_t threads[CVVDP_MAX_THREADS - 1];
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    CvvdpTaskFn task;
    void* task_data;
    atomic_int next_index;
    int item_count;
    int chunk_size;
    int generation;
    int remaining;
    bool stop;
} CvvdpThreadPool;

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

    // pyramid
    int num_bands;
    float* band_frequencies;

    // score accum
    int num_frames;
    double score_square_sum;

    // work buffers
    float* work_buffer;

    // task-thread pool
    CvvdpThreadPool* thread_pool;

    // temp channel buffers
    float* Y_sustained;
    float* RG_sustained;
    float* YV_sustained;
    float* Y_transient;
    float* dst_Y_sustained;
    float* dst_RG_sustained;
    float* dst_YV_sustained;
    float* dst_Y_transient;

    // reusable pyramid scratch buffers
    float* pyr_ref[CVVDP_MAX_LEVELS][CVVDP_NUM_CHANNELS];
    float* pyr_dst[CVVDP_MAX_LEVELS][CVVDP_NUM_CHANNELS];
    float* pyr_L_bkg[CVVDP_MAX_LEVELS];
    float* pyr_temp;
    float* pyr_reduced;
    float* pyr_expanded;
    float* pyr_min_abs[CVVDP_NUM_CHANNELS];
    float* pyr_blurred_min_abs[CVVDP_NUM_CHANNELS];
    float* pyr_tmp_blur;
    float* pyr_d;
} FcvvdpCtx;

// -- task threading structs --

typedef struct {
    float* plane;
    const Display* display;
    bool is_hdr;
} CvvdpApplyDisplayTaskData;

typedef struct {
    float* x;
    float* y;
    float* z;
} CvvdpColorTransformTaskData;

typedef struct CvvdpTemporalChannelsTaskData {
    const TemporalRingBuf* ring;
    float* Y_sus;
    float* RG_sus;
    float* YV_sus;
    float* Y_trans;
    size_t plane_size;
} CvvdpTemporalChannelsTaskData;

typedef struct CvvdpReduceTaskData {
    const float* src;
    float* dst;
    int src_w;
    int src_h;
} CvvdpReduceTaskData;

typedef struct CvvdpExpandTaskData {
    const float* src;
    float* dst;
    int dst_w;
    int dst_h;
} CvvdpExpandTaskData;

typedef struct CvvdpContrastTaskData {
    const float* src;
    const float* expanded;
    float* L_bkg;
    float* dst;
    float contrast_scale;
} CvvdpContrastTaskData;

typedef struct CvvdpNormalizeTaskData {
    const float* src;
    float* dst;
    float denom;
} CvvdpNormalizeTaskData;

typedef struct CvvdpCsfWeightTaskData {
    const Csf* csf;
    const float* L_bkg;
    float* const* ref_level;
    float* const* dst_level;
    int lev;
    size_t lev_size;
    const float* ch_gain;
} CvvdpCsfWeightTaskData;

typedef struct CvvdpMinAbsTaskData {
    const float* ref;
    const float* dst;
    float* out;
} CvvdpMinAbsTaskData;

typedef struct CvvdpBlurTaskData {
    const float* src;
    float* dst;
    int width;
    int height;
    const float* kernel;
    int radius;
} CvvdpBlurTaskData;

typedef struct CvvdpMaskedDiffTaskData {
    float* const* blurred_min_abs;
    float* const* ref_level;
    float* const* dst_level;
    float* d;
    size_t lev_size;
    float max_v;
    float pow_mask_c;
} CvvdpMaskedDiffTaskData;

typedef struct CvvdpBasebandDiffTaskData {
    const float* const* ref_level;
    const float* const* dst_level;
    float* d;
    size_t lev_size;
    float sensitivity[4];
} CvvdpBasebandDiffTaskData;

typedef struct CvvdpNormTaskData {
    const float* data;
    double* partials;
    int count;
    int participant_count;
    int power;
} CvvdpNormTaskData;

#endif /* FCVVDP_INTERNAL_H */
