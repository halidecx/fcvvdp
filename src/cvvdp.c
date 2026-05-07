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
#include "cvvdp.h"
#include "lut.h"
#include "util.h"
#include "internal.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

static bool cvvdp_thread_pool_is_active(const CvvdpThreadPool* const pool) {
    return pool != NULL && pool->worker_count > 0;
}

static void cvvdp_run_task_range(CvvdpThreadPool* const pool,
                                 CvvdpTaskFn const task,
                                 void* const task_data,
                                 const int item_count,
                                 const int chunk_size)
{
    for (;;) {
        const int start = atomic_fetch_add_explicit(&pool->next_index,
                                                    chunk_size,
                                                    memory_order_relaxed);
        if (start >= item_count) break;
        const int end = imin(start + chunk_size, item_count);
        task(task_data, start, end);
    }
}

static void* cvvdp_thread_pool_worker(void* arg) {
    CvvdpThreadPool* const pool = (CvvdpThreadPool*)arg;
    int generation = 0;

    pthread_mutex_lock(&pool->mutex);
    for (;;) {
        while (!pool->stop && pool->generation == generation)
            pthread_cond_wait(&pool->work_cond, &pool->mutex);

        if (pool->stop) break;

        generation = pool->generation;
        CvvdpTaskFn const task = pool->task;
        void* const task_data = pool->task_data;
        const int item_count = pool->item_count;
        const int chunk_size = pool->chunk_size;

        pthread_mutex_unlock(&pool->mutex);
        cvvdp_run_task_range(pool, task, task_data, item_count, chunk_size);
        pthread_mutex_lock(&pool->mutex);

        pool->remaining--;
        if (pool->remaining == 0)
            pthread_cond_signal(&pool->done_cond);
    }
    pthread_mutex_unlock(&pool->mutex);

    return NULL;
}

static unsigned get_threadcnt(const unsigned threads) {
    return threads ? threads : (unsigned)sysconf(_SC_NPROCESSORS_ONLN);
}

static CvvdpThreadPool* cvvdp_thread_pool_create(const unsigned total_threads) {
    if (total_threads <= 1) return NULL;

    CvvdpThreadPool* const pool =
        (CvvdpThreadPool*)calloc(1, sizeof(CvvdpThreadPool));
    if (!pool) return NULL;

    pool->worker_count =
        imin(total_threads - 1, CVVDP_MAX_THREADS - 1);

    bool mutex_ready = false;
    bool work_cond_ready = false;
    bool done_cond_ready = false;

    if (pthread_mutex_init(&pool->mutex, NULL) == 0)
        mutex_ready = true;
    if (mutex_ready && pthread_cond_init(&pool->work_cond, NULL) == 0)
        work_cond_ready = true;
    if (work_cond_ready && pthread_cond_init(&pool->done_cond, NULL) == 0)
        done_cond_ready = true;

    if (!done_cond_ready) {
        if (work_cond_ready) pthread_cond_destroy(&pool->work_cond);
        if (mutex_ready) pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }

    for (int i = 0; i < pool->worker_count; i++)
        if (pthread_create(&pool->threads[i], NULL,
                           cvvdp_thread_pool_worker, pool) != 0)
        {
            pthread_mutex_lock(&pool->mutex);
            pool->stop = true;
            pthread_cond_broadcast(&pool->work_cond);
            pthread_mutex_unlock(&pool->mutex);
            for (int j = 0; j < i; j++)
                pthread_join(pool->threads[j], NULL);
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->work_cond);
            pthread_cond_destroy(&pool->done_cond);
            free(pool);
            return NULL;
        }

    return pool;
}

static void cvvdp_thread_pool_destroy(CvvdpThreadPool* const pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    pool->stop = true;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mutex);

    for (int i = 0; i < pool->worker_count; i++)
        pthread_join(pool->threads[i], NULL);

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_cond);
    pthread_cond_destroy(&pool->done_cond);
    free(pool);
}

static void cvvdp_parallel_for(CvvdpThreadPool* const pool,
                               const int item_count,
                               const int min_chunk_size,
                               CvvdpTaskFn const task,
                               void* const task_data)
{
    if (item_count <= 0) return;

    const int participant_count = pool ? pool->worker_count + 1 : 1;
    const int chunk_size = imax(
        min_chunk_size,
        (item_count + participant_count * 4 - 1) / (participant_count * 4));

    if (!pool || pool->worker_count == 0 || item_count <= chunk_size) {
        task(task_data, 0, item_count);
        return;
    }

    pthread_mutex_lock(&pool->mutex);
    pool->task = task;
    pool->task_data = task_data;
    pool->item_count = item_count;
    pool->chunk_size = chunk_size;
    atomic_store_explicit(&pool->next_index, 0, memory_order_relaxed);
    pool->remaining = pool->worker_count + 1;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mutex);

    cvvdp_run_task_range(pool, task, task_data, item_count, chunk_size);

    pthread_mutex_lock(&pool->mutex);
    pool->remaining--;
    if (pool->remaining == 0)
        pthread_cond_signal(&pool->done_cond);
    while (pool->remaining != 0)
        pthread_cond_wait(&pool->done_cond, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

static float cvvdp_csf_sensitivity(const Csf* const csf,
                                   const float L_bkg_val,
                                   const int band,
                                   const int channel);

const char* cvvdp_error_string(const FcvvdpError error) {
    switch (error) {
        case CVVDP_OK: return "Success";
        case CVVDP_ERROR_NULL_POINTER: return "Null pointer";
        case CVVDP_ERROR_INVALID_DIMENSIONS: return "Invalid dimensions";
        case CVVDP_ERROR_INVALID_FORMAT: return "Invalid format";
        case CVVDP_ERROR_INVALID_MODEL: return "Invalid display model";
        case CVVDP_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case CVVDP_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case CVVDP_ERROR_NOT_INITIALIZED: return "Not initialized";
        default: return "Unknown error";
    }
}

static float cvvdp_compute_ppd(const Display* display) {
    const float ar =
        (float)display->resolution_width / (float)display->resolution_height;
    const float diag_mm = display->diagonal_size_inches * 25.4f;
    const float height_m =
        sqrtf(diag_mm * diag_mm / (1.0f + ar * ar)) / 1000.0f;
    const float width_m = ar * height_m;

    const float pix_deg =
        2.0f * 180.0f *
            atanf(0.5f * width_m /
                  (float)display->resolution_width
                  / display->viewing_distance_meters)
                 / (float)M_PI;
    return 1.0f / pix_deg;
}

static void cvvdp_init_display(Display* const display,
                               const FcvvdpDisplayModel model,
                               const FcvvdpDisplayParams* custom)
{
    display->exposure = 1.0f;

    if (model == CVVDP_DISPLAY_CUSTOM && custom != NULL) {
        display->resolution_width = custom->resolution_width;
        display->resolution_height = custom->resolution_height;
        display->viewing_distance_meters = custom->viewing_distance_meters;
        display->diagonal_size_inches = custom->diagonal_size_inches;
        display->max_luminance = custom->max_luminance;
        display->contrast = custom->contrast;
        display->ambient_light = custom->ambient_light;
        display->reflectivity = custom->reflectivity;
        display->is_hdr = custom->is_hdr;
    } else {
        switch (model) {
            case CVVDP_DISPLAY_STANDARD_4K:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.7472f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 200.0f;
                display->contrast = 1000.0f;
                display->ambient_light = 250.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = false;
                break;

            case CVVDP_DISPLAY_STANDARD_HDR_PQ:
            case CVVDP_DISPLAY_STANDARD_HDR_HLG:
            case CVVDP_DISPLAY_STANDARD_HDR_LINEAR:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.7472f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 1500.0f;
                display->contrast = 1000000.0f;
                display->ambient_light = 10.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = true;
                break;

            case CVVDP_DISPLAY_STANDARD_HDR_DARK:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.7472f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 1500.0f;
                display->contrast = 1000000.0f;
                display->ambient_light = 0.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = true;
                break;

            case CVVDP_DISPLAY_STANDARD_HDR_LINEAR_ZOOM:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.25f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 10000.0f;
                display->contrast = 1000000.0f;
                display->ambient_light = 10.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = true;
                break;

            case CVVDP_DISPLAY_STANDARD_FHD:
            default:
                display->resolution_width = 1920;
                display->resolution_height = 1080;
                display->viewing_distance_meters = 0.6f;
                display->diagonal_size_inches = 24.0f;
                display->max_luminance = 200.0f;
                display->contrast = 1000.0f;
                display->ambient_light = 250.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = false;
                break;
        }
    }

    display->ppd = cvvdp_compute_ppd(display);
    display->black_level = display->max_luminance / display->contrast;
    display->refl_level =
        display->ambient_light * display->reflectivity / (float)M_PI;
}

FcvvdpError cvvdp_get_display_params(const FcvvdpDisplayModel model,
                                     FcvvdpDisplayParams* const out_params)
{
    if (out_params == NULL) return CVVDP_ERROR_NULL_POINTER;

    Display display;
    cvvdp_init_display(&display, model, NULL);

    out_params->resolution_width = display.resolution_width;
    out_params->resolution_height = display.resolution_height;
    out_params->viewing_distance_meters = display.viewing_distance_meters;
    out_params->diagonal_size_inches = display.diagonal_size_inches;
    out_params->max_luminance = display.max_luminance;
    out_params->contrast = display.contrast;
    out_params->ambient_light = display.ambient_light;
    out_params->reflectivity = display.reflectivity;
    out_params->is_hdr = display.is_hdr;

    return CVVDP_OK;
}

// inverse real FFT for temporal filter generation
static void cvvdp_inverse_rfft(const float* const input,
                               const int input_size,
                               float* const output,
                               const int size)
{
    float* inp = cvvdp_alloc_float(input_size);
    if (!inp) return;

    memcpy(inp, input, input_size * sizeof(float));

    if (input_size > size / 2)
        inp[size / 2] /= 2.0f;
    inp[0] /= 2.0f;

    for (int i = 0; i < size; i++) {
        output[i] = 0.0f;
        for (int k = 0; k < input_size; k++)
            output[(i + input_size - 1) % size] +=
                2.0f * inp[k] * cosf(2.0f * (float)M_PI * k * i / size) / size;
    }

    free(inp);
}

FcvvdpError cvvdp_temporal_filter_init(TemporalFilter *const filter,
                                       const float fps)
{
    if (fps <= 0)
        filter->size = 1;
    else {
        filter->size = (int)(ceilf(0.25f * fps / 2.0f) * 2) + 1;
        if (filter->size < 1) filter->size = 1;
    }

    int fft_size = filter->size / 2 + 1;

    for (int j = 0; j < 4; j++) {
        filter->kernel[j] = cvvdp_alloc_float(filter->size);
        if (!filter->kernel[j]) {
            for (int k = 0; k < j; k++)
                free(filter->kernel[k]);
            return CVVDP_ERROR_OUT_OF_MEMORY;
        }
    }

    float* freq = cvvdp_alloc_float(fft_size);
    float* fft_domain = cvvdp_alloc_float(fft_size);
    if (!freq || !fft_domain) {
        for (int j = 0; j < 4; j++) free(filter->kernel[j]);
        free(freq);
        free(fft_domain);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < fft_size; i++)
        freq[i] = fps * i / (2.0f * fft_size);

    for (int j = 0; j < 4; j++) {
        if (j < 3) {
            for (int i = 0; i < fft_size; i++)
                fft_domain[i] = expf(-powf(freq[i], CVVDP_BETA_TF[j]) / CVVDP_SIGMA_TF[j]);
        } else {
            for (int i = 0; i < fft_size; i++) {
                float temp = powf(freq[i], CVVDP_BETA_TF[3]) - powf(5.0f, CVVDP_BETA_TF[3]);
                temp *= temp;
                fft_domain[i] = expf(-temp / CVVDP_SIGMA_TF[3]);
            }
        }

        cvvdp_inverse_rfft(fft_domain, fft_size, filter->kernel[j], filter->size);
    }

    free(freq);
    free(fft_domain);

    return CVVDP_OK;
}

void cvvdp_temporal_filter_destroy(TemporalFilter* const filter) {
    for (int j = 0; j < 4; j++)
        if (filter->kernel[j]) {
            free(filter->kernel[j]);
            filter->kernel[j] = NULL;
        }
}

FcvvdpError cvvdp_temporal_ring_init(TemporalRingBuf* const ring,
                                     const int width,
                                     const int height,
                                     const float fps)
{
    ring->width = width;
    ring->height = height;
    ring->num_frames = 0;
    ring->current_index = 0;

    FcvvdpError err = cvvdp_temporal_filter_init(&ring->filter, fps);
    if (err != CVVDP_OK) return err;

    ring->max_frames = ring->filter.size;

    size_t frame_size = (size_t)width * height * 3;
    ring->data = cvvdp_alloc_float(frame_size * ring->max_frames);
    if (!ring->data) {
        cvvdp_temporal_filter_destroy(&ring->filter);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    return CVVDP_OK;
}

void cvvdp_temporal_ring_destroy(TemporalRingBuf* const ring) {
    if (ring->data) {
        free(ring->data);
        ring->data = NULL;
    }
    cvvdp_temporal_filter_destroy(&ring->filter);
}

void cvvdp_temporal_ring_reset(TemporalRingBuf* const ring) {
    ring->num_frames = 0;
    ring->current_index = 0;
}

const float* cvvdp_temporal_ring_get_frame(const TemporalRingBuf* const ring,
                                           int age)
{
    if (ring->num_frames == 0) return NULL;

    if (age >= ring->num_frames) age = ring->num_frames - 1;

    const size_t plane_size = (size_t)ring->width * ring->height * 3;
    const int idx = (ring->current_index + age) % ring->max_frames;
    return ring->data + idx * plane_size;
}

void cvvdp_temporal_ring_push(TemporalRingBuf* ring, const float* frame) {
    ring->num_frames = imin(ring->max_frames, ring->num_frames + 1);
    ring->current_index =
        (ring->current_index + ring->max_frames - 1) % ring->max_frames;

    size_t plane_size = (size_t)ring->width * ring->height * 3;
    float* dst = ring->data + ring->current_index * plane_size;
    memcpy(dst, frame, plane_size * sizeof(float));
}

static void cvvdp_apply_display_task(void* user_data,
                                     const int start,
                                     const int end)
{
    CvvdpApplyDisplayTaskData* const data =
        (CvvdpApplyDisplayTaskData*)user_data;
    const float Y_peak = data->display->max_luminance;
    const float Y_black = data->display->black_level;
    const float Y_refl = data->display->refl_level;
    const float exposure = data->display->exposure;

    for (int i = start; i < end; i++) {
        float val = data->plane[i];
        if (data->is_hdr) {
            val *= 100.0f;
            val = fmaxf(fmaxf(0.005f, Y_black),
                        fminf(Y_peak, val * exposure)) + Y_black + Y_refl;
        } else {
            val = fclip(val * exposure, 0.0f, 1.0f);
            val = (Y_peak - Y_black) * val + Y_black + Y_refl;
        }
        data->plane[i] = val;
    }
}

static void cvvdp_rgb_to_xyz_task(void* user_data,
                                  const int start,
                                  const int end)
{
    CvvdpColorTransformTaskData* const data =
        (CvvdpColorTransformTaskData*)user_data;
    for (int i = start; i < end; i++) {
        const float ri = data->x[i];
        const float gi = data->y[i];
        const float bi = data->z[i];

        data->x[i] = 0.4124564f * ri + 0.3575761f * gi + 0.1804375f * bi;
        data->y[i] = 0.2126729f * ri + 0.7151522f * gi + 0.0721750f * bi;
        data->z[i] = 0.0193339f * ri + 0.1191920f * gi + 0.9503041f * bi;
    }
}

static void cvvdp_xyz_to_dkl_task(void* user_data,
                                  const int start,
                                  const int end)
{
    CvvdpColorTransformTaskData* const data =
        (CvvdpColorTransformTaskData*)user_data;
    for (int i = start; i < end; i++) {
        const float xi = data->x[i];
        const float yi = data->y[i];
        const float zi = data->z[i];

        const float L =
            0.187596268556126f * xi + 0.585168649077728f * yi -
            0.026384263306304f * zi;
        const float M =
            -0.133397430663221f * xi + 0.405505777260049f * yi +
            0.034502127690364f * zi;
        const float S =
            0.000244379021663f * xi - 0.000542995890619f * yi +
            0.019406849066323f * zi;

        const float lum = L + M;
        data->x[i] = lum;
        data->y[i] = lum - 3.311130179947035f * M;
        data->z[i] = 50.977571328718781f * S - lum;
    }
}

static void cvvdp_compute_temporal_channels_task(void* user_data,
                                                 const int start,
                                                 const int end)
{
    CvvdpTemporalChannelsTaskData* const data =
        (CvvdpTemporalChannelsTaskData*)user_data;
    for (int i = start; i < end; i++) {
        float Y_sus = 0.0f;
        float RG_sus = 0.0f;
        float YV_sus = 0.0f;
        float Y_trans = 0.0f;

        for (int k = 0; k < data->ring->filter.size; k++) {
            const float* const frame =
                cvvdp_temporal_ring_get_frame(data->ring, k);
            if (!frame) continue;

            const float y = frame[i];
            Y_sus += y * data->ring->filter.kernel[0][k];
            Y_trans += y * data->ring->filter.kernel[3][k];
            RG_sus += frame[i + data->plane_size] * data->ring->filter.kernel[1][k];
            YV_sus += frame[i + 2 * data->plane_size] *
                data->ring->filter.kernel[2][k];
        }

        data->Y_sus[i] = Y_sus;
        data->RG_sus[i] = RG_sus;
        data->YV_sus[i] = YV_sus;
        data->Y_trans[i] = Y_trans;
    }
}

static void cvvdp_gauss_pyr_reduce_task(void* user_data,
                                        const int start,
                                        const int end)
{
    CvvdpReduceTaskData* const data = (CvvdpReduceTaskData*)user_data;
    const int dst_w = (data->src_w + 1) >> 1;
    const int dst_h = (data->src_h + 1) >> 1;

    for (int dy = start; dy < end && dy < dst_h; dy++) {
        float* const dst_row = &data->dst[dy * dst_w];
        const int dy2 = dy << 1;
        for (int dx = 0; dx < dst_w; dx++) {
            float val = 0.0f;
            const int dx2 = dx << 1;
            for (int ky = -2; ky <= 2; ky++) {
                int sy = dy2 + ky;
                if (sy < 0) sy = -sy - 1;
                if (sy >= data->src_h) sy = 2 * data->src_h - sy - 2;
                const float* const src_row = &data->src[sy * data->src_w];
                for (int kx = -2; kx <= 2; kx++) {
                    int sx = dx2 + kx;
                    if (sx < 0) sx = -sx - 1;
                    if (sx >= data->src_w) sx = 2 * data->src_w - sx - 2;
                    val += GAUSS_PYR_KERNEL[kx + 2] * GAUSS_PYR_KERNEL[ky + 2] *
                        src_row[sx];
                }
            }
            dst_row[dx] = val;
        }
    }
}

static void cvvdp_gauss_pyr_expand_task(void* user_data,
                                        const int start,
                                        const int end)
{
    CvvdpExpandTaskData* const data = (CvvdpExpandTaskData*)user_data;
    const int src_w = (data->dst_w + 1) >> 1;
    const int src_h = (data->dst_h + 1) >> 1;

    for (int dy = start; dy < end && dy < data->dst_h; dy++) {
        float* const dst_row = &data->dst[dy * data->dst_w];
        const int parity_y = dy % 2;
        for (int dx = 0; dx < data->dst_w; dx++) {
            float val = 0.0f;
            const int parity_x = dx % 2;
            for (int ky = -2 + parity_y; ky <= 2; ky += 2) {
                int sy = (dy + ky) >> 1;
                if (sy < 0) sy = -sy - 1;
                if (sy >= src_h) sy = 2 * src_h - sy - 2;
                const float* const src_row = &data->src[sy * src_w];
                for (int kx = -2 + parity_x; kx <= 2; kx += 2) {
                    int sx = (dx + kx) >> 1;
                    if (sx < 0) sx = -sx - 1;
                    if (sx >= src_w) sx = 2 * src_w - sx - 2;
                    val += 4.0f * GAUSS_PYR_KERNEL[kx + 2] *
                        GAUSS_PYR_KERNEL[ky + 2] * src_row[sx];
                }
            }
            dst_row[dx] = val;
        }
    }
}

static void cvvdp_contrast_task(void* user_data,
                                const int start,
                                const int end)
{
    CvvdpContrastTaskData* const data = (CvvdpContrastTaskData*)user_data;
    for (int i = start; i < end; i++) {
        data->dst[i] =
            ((data->src[i] - data->expanded[i]) / fmaxf(0.01f, data->L_bkg[i])) *
            data->contrast_scale;
    }
}

static void cvvdp_luma_contrast_task(void* user_data,
                                     const int start,
                                     const int end)
{
    CvvdpContrastTaskData* const data = (CvvdpContrastTaskData*)user_data;
    for (int i = start; i < end; i++) {
        const float L_bkg = fmaxf(0.01f, data->expanded[i]);
        data->L_bkg[i] = L_bkg;
        data->dst[i] =
            ((data->src[i] - data->expanded[i]) / L_bkg) *
            data->contrast_scale;
    }
}

static void cvvdp_normalize_task(void* user_data,
                                 const int start,
                                 const int end)
{
    CvvdpNormalizeTaskData* const data = (CvvdpNormalizeTaskData*)user_data;
    for (int i = start; i < end; i++)
        data->dst[i] = data->src[i] / data->denom;
}

static void cvvdp_csf_weight_task(void* user_data,
                                  const int start,
                                  const int end)
{
    CvvdpCsfWeightTaskData* const data = (CvvdpCsfWeightTaskData*)user_data;
    for (int idx = start; idx < end; idx++) {
        const int ch = idx / (int)data->lev_size;
        const int i = idx - ch * (int)data->lev_size;
        const float s = cvvdp_csf_sensitivity(data->csf, data->L_bkg[i],
                                              data->lev, ch) *
            data->ch_gain[ch];
        data->ref_level[ch][i] *= s;
        data->dst_level[ch][i] *= s;
    }
}

static void cvvdp_min_abs_task(void* user_data,
                               const int start,
                               const int end)
{
    CvvdpMinAbsTaskData* const data = (CvvdpMinAbsTaskData*)user_data;
    for (int i = start; i < end; i++)
        data->out[i] = fminf(fabsf(data->ref[i]), fabsf(data->dst[i]));
}

static void cvvdp_blur_horizontal_task(void* user_data,
                                       const int start,
                                       const int end)
{
    CvvdpBlurTaskData* const data = (CvvdpBlurTaskData*)user_data;
    for (int y = start; y < end && y < data->height; y++) {
        float* const dst_row = &data->dst[y * data->width];
        for (int x = 0; x < data->width; x++) {
            float sum = 0.0f;
            float wsum = 0.0f;
            for (int k = -data->radius; k <= data->radius; k++) {
                const int sx = x + k;
                if (sx >= 0 && sx < data->width) {
                    const float weight = data->kernel[k + data->radius];
                    sum += data->src[y * data->width + sx] * weight;
                    wsum += weight;
                }
            }
            dst_row[x] = sum / wsum;
        }
    }
}

static void cvvdp_blur_vertical_task(void* user_data,
                                     const int start,
                                     const int end)
{
    CvvdpBlurTaskData* const data = (CvvdpBlurTaskData*)user_data;
    for (int y = start; y < end && y < data->height; y++)
        for (int x = 0; x < data->width; x++) {
            float sum = 0.0f;
            float wsum = 0.0f;
            for (int k = -data->radius; k <= data->radius; k++) {
                const int sy = y + k;
                if (sy >= 0 && sy < data->height) {
                    const float weight = data->kernel[k + data->radius];
                    sum += data->src[sy * data->width + x] * weight;
                    wsum += weight;
                }
            }
            data->dst[y * data->width + x] = sum / wsum;
        }
}

static void cvvdp_masked_diff_task(void* user_data,
                                   const int start,
                                   const int end)
{
    CvvdpMaskedDiffTaskData* const data = (CvvdpMaskedDiffTaskData*)user_data;
    for (int i = start; i < end; i++) {
        float cm[4];
        for (int ch = 0; ch < 4; ch++)
            cm[ch] = powf(data->pow_mask_c * data->blurred_min_abs[ch][i],
                          CVVDP_MASK_Q[ch]);
        for (int ch = 0; ch < 4; ch++) {
            const float mask =
                cm[0] * powf(2.0f, CVVDP_XCM_WEIGHTS[0 + ch]) +
                cm[1] * powf(2.0f, CVVDP_XCM_WEIGHTS[4 + ch]) +
                cm[2] * powf(2.0f, CVVDP_XCM_WEIGHTS[8 + ch]) +
                cm[3] * powf(2.0f, CVVDP_XCM_WEIGHTS[12 + ch]);

            const float diff = fabsf(data->ref_level[ch][i] - data->dst_level[ch][i]);
            const float du = powf(diff, CVVDP_MASK_P) / (1.0f + mask);
            const float d_val = data->max_v * du / (data->max_v + du);

            float weight = 1.0f;
            switch (ch) {
                case 1:
                case 2:
                    weight = CVVDP_CH_CHROM_W;
                    break;
                case 3:
                    weight = CVVDP_CH_TRANS_W;
                    break;
            }

            data->d[ch * data->lev_size + i] = d_val * weight;
        }
    }
}

static void cvvdp_baseband_diff_task(void* user_data,
                                     const int start,
                                     const int end)
{
    CvvdpBasebandDiffTaskData* const data =
        (CvvdpBasebandDiffTaskData*)user_data;
    const int lev_size = (int)data->lev_size;
    for (int idx = start; idx < end; idx++) {
        const int ch = idx / lev_size;
        const int i = idx - ch * lev_size;
        const float diff = fabsf(data->ref_level[ch][i] - data->dst_level[ch][i]);
        data->d[idx] = diff * data->sensitivity[ch] * CVVDP_BASEBAND_WEIGHT[ch];
    }
}

static void cvvdp_norm_task(void* user_data,
                            const int start,
                            const int end)
{
    CvvdpNormTaskData* const data = (CvvdpNormTaskData*)user_data;
    for (int idx = start; idx < end; idx++) {
        const int range_start = (idx * data->count) / data->participant_count;
        const int range_end = ((idx + 1) * data->count) / data->participant_count;
        double sum = 0.0;
        if (data->power == 2) {
            for (int i = range_start; i < range_end; i++) {
                const double v = data->data[i];
                sum += v * v;
            }
        } else {
            for (int i = range_start; i < range_end; i++)
                sum += pow(fabs((double)data->data[i]), data->power);
        }
        data->partials[idx] = sum;
    }
}

void cvvdp_rgb_to_xyz(float* r, float* g, float* b, int count) {
    for (int i = 0; i < count; i++) {
        float ri = r[i], gi = g[i], bi = b[i];

        // BT.709 RGB to XYZ
        float x = 0.4124564f * ri + 0.3575761f * gi + 0.1804375f * bi;
        float y = 0.2126729f * ri + 0.7151522f * gi + 0.0721750f * bi;
        float z = 0.0193339f * ri + 0.1191920f * gi + 0.9503041f * bi;

        r[i] = x;
        g[i] = y;
        b[i] = z;
    }
}

static void cvvdp_rgb_to_xyz_interleaved(CvvdpThreadPool* const pool,
                                         float* const frame,
                                         const int count)
{
    if (!cvvdp_thread_pool_is_active(pool)) {
        cvvdp_rgb_to_xyz(frame, frame + count, frame + 2 * count, count);
        return;
    }

    CvvdpColorTransformTaskData task = {
        .x = frame,
        .y = frame + count,
        .z = frame + 2 * count,
    };
    cvvdp_parallel_for(pool, count, 2048, cvvdp_rgb_to_xyz_task, &task);
}

void cvvdp_xyz_to_dkl(float *const x,
                      float *const y,
                      float* const z,
                      const int count)
{
    for (int i = 0; i < count; i++) {
        float xi = x[i], yi = y[i], zi = z[i];

        // XYZ to LMS (CIE 2006)
        float L = 0.187596268556126f * xi + 0.585168649077728f * yi - 0.026384263306304f * zi;
        float M = -0.133397430663221f * xi + 0.405505777260049f * yi + 0.034502127690364f * zi;
        float S = 0.000244379021663f * xi - 0.000542995890619f * yi + 0.019406849066323f * zi;

        // LMS to DKL (D65 adapted)
        float lum = L + M; // luminance
        float rg = lum - 3.311130179947035f * M; // red-green
        float yv = 50.977571328718781f * S - lum; // yellow-violet

        x[i] = lum;
        y[i] = rg;
        z[i] = yv;
    }
}

static void cvvdp_xyz_to_dkl_interleaved(CvvdpThreadPool* const pool,
                                         float* const frame,
                                         const int count)
{
    if (!cvvdp_thread_pool_is_active(pool)) {
        cvvdp_xyz_to_dkl(frame, frame + count, frame + 2 * count, count);
        return;
    }

    CvvdpColorTransformTaskData task = {
        .x = frame,
        .y = frame + count,
        .z = frame + 2 * count,
    };
    cvvdp_parallel_for(pool, count, 2048, cvvdp_xyz_to_dkl_task, &task);
}

void cvvdp_apply_display_model(float* const plane,
                               const int count,
                               const Display* display,
                               const bool is_hdr)
{
    const float Y_peak = display->max_luminance;
    const float Y_black = display->black_level;
    const float Y_refl = display->refl_level;
    const float exposure = display->exposure;

    for (int i = 0; i < count; i++) {
        float val = plane[i];
        if (is_hdr) {
            val *= 100.0f;
            val = fmax(fmax(0.005f, Y_black),
                            fmin(Y_peak, val * exposure)) + Y_black + Y_refl;
        } else {
            val = fclip(val * exposure, 0.0f, 1.0f);
            val = (Y_peak - Y_black) * val + Y_black + Y_refl;
        }
        plane[i] = val;
    }
}

static void cvvdp_apply_display_model_interleaved(CvvdpThreadPool* const pool,
                                                  float* const frame,
                                                  const int count,
                                                  const Display *const display,
                                                  const bool is_hdr)
{
    if (!cvvdp_thread_pool_is_active(pool)) {
        cvvdp_apply_display_model(frame, count, display, is_hdr);
        cvvdp_apply_display_model(frame + count, count, display, is_hdr);
        cvvdp_apply_display_model(frame + 2 * count, count, display, is_hdr);
        return;
    }

    CvvdpApplyDisplayTaskData tasks[3] = {
        {.plane = frame, .display = display, .is_hdr = is_hdr},
        {.plane = frame + count, .display = display, .is_hdr = is_hdr},
        {.plane = frame + 2 * count, .display = display, .is_hdr = is_hdr},
    };
    for (int plane = 0; plane < 3; plane++)
        cvvdp_parallel_for(pool, count, 2048, cvvdp_apply_display_task,
                           &tasks[plane]);
}

FcvvdpError cvvdp_load_image(const FcvvdpImage* const img,
                             float* const frame)
{
    if (!img || !img->data || !frame) return CVVDP_ERROR_NULL_POINTER;

    const int w = img->width, h = img->height;
    const size_t plane_size = (size_t)w * h;
    int stride = img->stride;
    if (stride == 0)
        switch (img->format) {
            case CVVDP_PIXEL_FORMAT_RGB_FLOAT:
                stride = w * 3 * sizeof(float);
                break;
            case CVVDP_PIXEL_FORMAT_RGB_UINT8:
                stride = w * 3 * sizeof(uint8_t);
                break;
            case CVVDP_PIXEL_FORMAT_RGB_UINT16:
                stride = w * 3 * sizeof(uint16_t);
                break;
        }

    float* const r_plane = frame;
    float* const g_plane = frame + plane_size;
    float* const b_plane = frame + 2 * plane_size;

    float r, g, b;
    for (int y = 0; y < h; y++) {
        const int stride_mul = y * stride;
        const int y_w = y * w;
        for (int x = 0; x < w; x++) {
            const int idx = y_w + x;
            switch (img->format) {
                case CVVDP_PIXEL_FORMAT_RGB_FLOAT: {
                    const float* const row =
                        (const float*)((const uint8_t*)img->data + stride_mul);
                    r = row[x * 3 + 0];
                    g = row[x * 3 + 1];
                    b = row[x * 3 + 2];
                    break;
                }
                case CVVDP_PIXEL_FORMAT_RGB_UINT8: {
                    const uint8_t* const row =
                        (const uint8_t*)img->data + stride_mul;
                    r = row[x * 3 + 0] / 255.0f;
                    g = row[x * 3 + 1] / 255.0f;
                    b = row[x * 3 + 2] / 255.0f;
                    r = (r < 0) ? -powf(-r, 2.4f) : powf(r, 2.4f);
                    g = (g < 0) ? -powf(-g, 2.4f) : powf(g, 2.4f);
                    b = (b < 0) ? -powf(-b, 2.4f) : powf(b, 2.4f);
                    break;
                }
                case CVVDP_PIXEL_FORMAT_RGB_UINT16: {
                    const uint16_t* const row =
                        (const uint16_t*)((const uint8_t*)img->data
                                          + stride_mul);
                    r = row[x * 3 + 0] / 65535.0f;
                    g = row[x * 3 + 1] / 65535.0f;
                    b = row[x * 3 + 2] / 65535.0f;
                    break;
                }
                default: return CVVDP_ERROR_INVALID_FORMAT;
            }
            r_plane[idx] = r;
            g_plane[idx] = g;
            b_plane[idx] = b;
        }
    }

    return CVVDP_OK;
}

static void cvvdp_compute_temporal_channels(CvvdpThreadPool* const pool,
                                            TemporalRingBuf* const ring,
                                            float* const Y_sus,
                                            float* const RG_sus,
                                            float* const YV_sus,
                                            float* const Y_trans)
{
    const int w = ring->width, h = ring->height;
    const size_t plane_size = (size_t)w * h;

    if (ring->num_frames == 1) { // 1 frame, no tf
        const float* const frame = cvvdp_temporal_ring_get_frame(ring, 0);
        memcpy(Y_sus, frame, plane_size * sizeof(float));
        memcpy(RG_sus, frame + plane_size, plane_size * sizeof(float));
        memcpy(YV_sus, frame + 2 * plane_size, plane_size * sizeof(float));
        memset(Y_trans, 0, plane_size * sizeof(float));
        return;
    }

    memset(Y_sus, 0, plane_size * sizeof(float));
    memset(RG_sus, 0, plane_size * sizeof(float));
    memset(YV_sus, 0, plane_size * sizeof(float));
    memset(Y_trans, 0, plane_size * sizeof(float));

    if (!cvvdp_thread_pool_is_active(pool)) {
        for (int k = 0; k < ring->filter.size; k++) {
            const float* const frame = cvvdp_temporal_ring_get_frame(ring, k);
            if (!frame) continue;

            const float k0 = ring->filter.kernel[0][k];
            const float k1 = ring->filter.kernel[1][k];
            const float k2 = ring->filter.kernel[2][k];
            const float k3 = ring->filter.kernel[3][k];

            for (size_t i = 0; i < plane_size; i++) {
                const float y = frame[i];
                Y_sus[i] += y * k0;
                Y_trans[i] += y * k3;
                RG_sus[i] += frame[i + plane_size] * k1;
                YV_sus[i] += frame[i + 2 * plane_size] * k2;
            }
        }
        return;
    }

    CvvdpTemporalChannelsTaskData task = {
        .ring = ring,
        .Y_sus = Y_sus,
        .RG_sus = RG_sus,
        .YV_sus = YV_sus,
        .Y_trans = Y_trans,
        .plane_size = plane_size,
    };
    cvvdp_parallel_for(pool, (int)plane_size, 2048,
                       cvvdp_compute_temporal_channels_task, &task);
}

static void cvvdp_gauss_pyr_reduce(CvvdpThreadPool* const pool,
                                   const float* const src,
                                   float* const dst,
                                   const int src_w,
                                   const int src_h)
{
    if (!cvvdp_thread_pool_is_active(pool)) {
        const int dst_w = (src_w + 1) >> 1;
        const int dst_h = (src_h + 1) >> 1;

        for (int dy = 0; dy < dst_h; dy++) {
            float* const dst_row = &dst[dy * dst_w];
            const int dy2 = dy << 1;
            for (int dx = 0; dx < dst_w; dx++) {
                float val = 0.0f;
                const int dx2 = dx << 1;
                for (int ky = -2; ky <= 2; ky++) {
                    int sy = dy2 + ky;
                    if (sy < 0) sy = -sy - 1;
                    if (sy >= src_h) sy = 2 * src_h - sy - 2;
                    const float* const src_row = &src[sy * src_w];
                    for (int kx = -2; kx <= 2; kx++) {
                        int sx = dx2 + kx;
                        if (sx < 0) sx = -sx - 1;
                        if (sx >= src_w) sx = 2 * src_w - sx - 2;
                        val += GAUSS_PYR_KERNEL[kx + 2] *
                            GAUSS_PYR_KERNEL[ky + 2] * src_row[sx];
                    }
                }
                dst_row[dx] = val;
            }
        }
        return;
    }

    CvvdpReduceTaskData task = {
        .src = src,
        .dst = dst,
        .src_w = src_w,
        .src_h = src_h,
    };
    cvvdp_parallel_for(pool, (src_h + 1) >> 1, 4,
                       cvvdp_gauss_pyr_reduce_task, &task);
}

static void cvvdp_gauss_pyr_expand(CvvdpThreadPool* const pool,
                                   const float* const src,
                                   float* const dst,
                                   const int dst_w,
                                   const int dst_h)
{
    if (!cvvdp_thread_pool_is_active(pool)) {
        const int src_w = (dst_w + 1) >> 1;
        const int src_h = (dst_h + 1) >> 1;

        for (int dy = 0; dy < dst_h; dy++) {
            float* const dst_row = &dst[dy * dst_w];
            const int parity_y = dy % 2;
            for (int dx = 0; dx < dst_w; dx++) {
                float val = 0.0f;
                const int parity_x = dx % 2;
                for (int ky = -2 + parity_y; ky <= 2; ky += 2) {
                    int sy = (dy + ky) >> 1;
                    if (sy < 0) sy = -sy - 1;
                    if (sy >= src_h) sy = 2 * src_h - sy - 2;
                    const float* const src_row = &src[sy * src_w];
                    for (int kx = -2 + parity_x; kx <= 2; kx += 2) {
                        int sx = (dx + kx) >> 1;
                        if (sx < 0) sx = -sx - 1;
                        if (sx >= src_w) sx = 2 * src_w - sx - 2;
                        val += 4.0f * GAUSS_PYR_KERNEL[kx + 2] *
                            GAUSS_PYR_KERNEL[ky + 2] * src_row[sx];
                    }
                }
                dst_row[dx] = val;
            }
        }
        return;
    }

    CvvdpExpandTaskData task = {
        .src = src,
        .dst = dst,
        .dst_w = dst_w,
        .dst_h = dst_h,
    };
    cvvdp_parallel_for(pool, dst_h, 4, cvvdp_gauss_pyr_expand_task, &task);
}

static int cvvdp_get_band_frequencies(const int width,
                                      const int height,
                                      const float ppd,
                                      float* const out_freqs)
{
    const float min_freq = 0.2f;
    const int max_level_res =
        lrint((log2f((float)imin(width, height))) - 1);
    const int max_level_ppd =
        lrint(ceilf(-log2f(2.0f * min_freq / 0.3228f / ppd)) + 1);
    const int max_level =
        imin(max_level_res, imin(max_level_ppd, CVVDP_MAX_LEVELS));

    out_freqs[0] = 0.5f * ppd;
    const float ppd_const = 0.3228f * 0.5f * ppd;
    for (int i = 1; i < max_level; i++)
        out_freqs[i] = ppd_const / (float)(1 << (i - 1));

    return max_level;
}

FcvvdpError cvvdp_csf_init(Csf* const csf,
                           const int width,
                           const int height,
                           const float ppd)
{
    float freqs[CVVDP_MAX_LEVELS];
    csf->num_bands = cvvdp_get_band_frequencies(width, height, ppd, freqs);

    csf->log_S_LUT = cvvdp_alloc_float(csf->num_bands * 4 * CVVDP_LUT_SIZE);
    if (!csf->log_S_LUT) return CVVDP_ERROR_OUT_OF_MEMORY;

    for (int band = 0; band < csf->num_bands; band++) {
        const float log_freq = log10f(freqs[band]);

        int rho_idx = 0;
        for (int i = 1; i < CVVDP_LUT_SIZE; i++) {
            if (LOG10_RHO[i] > log_freq) {
                rho_idx = i - 1;
                break;
            }
            rho_idx = i;
        }
        rho_idx = imin(rho_idx, 30);

        const float slope = (log_freq - LOG10_RHO[rho_idx]) /
            (LOG10_RHO[rho_idx + 1] - LOG10_RHO[rho_idx]);

        for (int ch = 0; ch < 4; ch++)
            for (int l = 0; l < CVVDP_LUT_SIZE; l++) {
                const float y0 = D2LUT[ch][l][rho_idx];
                const float y1 = D2LUT[ch][l][rho_idx + 1];
                const float log_S = y0 + (y1 - y0) * slope;
                csf->log_S_LUT[(band * 4 + ch) * CVVDP_LUT_SIZE + l] = log_S;
            }
    }

    return CVVDP_OK;
}

static void cvvdp_csf_destroy(Csf* const csf) {
    if (csf->log_S_LUT) {
        free(csf->log_S_LUT);
        csf->log_S_LUT = NULL;
    }
}

static float cvvdp_csf_sensitivity(const Csf* const csf,
                                   const float L_bkg_val,
                                   const int band,
                                   const int channel)
{
    float frac = 31.0f * (log10f(L_bkg_val) - LOG10_L_BKG[0]) /
        (LOG10_L_BKG[31] - LOG10_L_BKG[0]);

    const int i_min = iclip((int)frac, 0, 30);
    const int i_max = i_min + 1;
    frac = frac - (float)i_min;

    const float* const lut =
        csf->log_S_LUT + (band * 4 + channel) * CVVDP_LUT_SIZE;
    const float log_S = lut[i_min] * frac + lut[i_max] *
        (1.0f - frac) + CVVDP_SENSITIVITY_CORRECTION / 20.0f;

    return powf(10.0f, log_S);
}

static float cvvdp_to_jod(const float quality) {
    if (quality > 0.1f)
        return 10.0f - CVVDP_JOD_A * powf(quality, CVVDP_JOD_EXP);
    else {
        const float jod_a_p = CVVDP_JOD_A * powf(0.1f, CVVDP_JOD_EXP - 1.0f);
        return 10.0f - jod_a_p * quality;
    }
}

static float cvvdp_compute_norm(CvvdpThreadPool* const pool,
                                const float* const data,
                                const int count,
                                const int power)
{
    const int participant_count = pool ? pool->worker_count + 1 : 1;
    if (!pool || participant_count == 1 || count < 8192) {
        double sum = 0.0;
        if (power == 2) {
            for (int i = 0; i < count; i++) {
                const double v = data[i];
                sum += v * v;
            }
        } else {
            for (int i = 0; i < count; i++)
                sum += pow(fabs((double)data[i]), power);
        }
        return (float)pow(sum / count, 1.0 / power);
    }

    double partials[CVVDP_MAX_THREADS] = {0};
    CvvdpNormTaskData task = {
        .data = data,
        .partials = partials,
        .count = count,
        .participant_count = participant_count,
        .power = power,
    };
    cvvdp_parallel_for(pool, participant_count, 1, cvvdp_norm_task, &task);

    double sum = 0.0;
    for (int i = 0; i < participant_count; i++)
        sum += partials[i];
    return (float)pow(sum / count, 1.0 / power);
}

static FcvvdpError cvvdp_process_pyramid_threaded(FcvvdpCtx* const c,
                                                  float* const ref_channels[4],
                                                  float* const dst_channels[4],
                                                  double* const out_score)
{
    const int w = c->width, h = c->height;
    const int num_levels = c->num_bands;
    CvvdpThreadPool* const pool = c->thread_pool;

    size_t total_size = 0;
    int widths[CVVDP_MAX_LEVELS], heights[CVVDP_MAX_LEVELS];
    int tw = w, th = h;
    for (int lev = 0; lev < num_levels; lev++) {
        widths[lev] = tw;
        heights[lev] = th;
        total_size += (size_t)tw * th;
        tw = (tw + 1) / 2;
        th = (th + 1) / 2;
    }

    float* (*ref_pyr)[4] = c->pyr_ref;
    float* (*dst_pyr)[4] = c->pyr_dst;
    float** const L_bkg_pyr = c->pyr_L_bkg;
    float* const temp = c->pyr_temp;
    float* const reduced = c->pyr_reduced;
    float* const expanded = c->pyr_expanded;
    FcvvdpError err = CVVDP_OK;

    float* buf_a = temp;
    float* buf_b = reduced;

    for (int ch = 0; ch < 4; ch++) {
        memcpy(buf_a, ref_channels[ch], (size_t)w * h * sizeof(float));
        int cw = w, ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            const size_t lev_size = (size_t)widths[lev] * heights[lev];
            if (lev < num_levels - 1) {
                cvvdp_gauss_pyr_reduce(pool, buf_a, buf_b, cw, ch_h);
                cvvdp_gauss_pyr_expand(pool, buf_b, expanded, cw, ch_h);
                if (!ch) {
                    CvvdpContrastTaskData task = {
                        .src = buf_a,
                        .expanded = expanded,
                        .L_bkg = L_bkg_pyr[lev],
                        .dst = ref_pyr[lev][ch],
                        .contrast_scale = lev == 0 ? 1.0f : 2.0f,
                    };
                    cvvdp_parallel_for(pool, (int)lev_size, 2048,
                                       cvvdp_luma_contrast_task, &task);
                } else {
                    CvvdpContrastTaskData task = {
                        .src = buf_a,
                        .expanded = expanded,
                        .L_bkg = L_bkg_pyr[lev],
                        .dst = ref_pyr[lev][ch],
                        .contrast_scale = lev == 0 ? 1.0f : 2.0f,
                    };
                    cvvdp_parallel_for(pool, (int)lev_size, 2048,
                                       cvvdp_contrast_task, &task);
                }
                float* const t = buf_a;
                buf_a = buf_b;
                buf_b = t;
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else {
                if (!ch) {
                    double mean = 0.0;
                    for (size_t i = 0; i < lev_size; i++)
                        mean += buf_a[i];
                    mean /= (double)lev_size;
                    L_bkg_pyr[lev][0] = fmaxf(0.01f, (float)mean);
                }
                CvvdpNormalizeTaskData task = {
                    .src = buf_a,
                    .dst = ref_pyr[lev][ch],
                    .denom = fmaxf(0.01f, L_bkg_pyr[lev][0]),
                };
                cvvdp_parallel_for(pool, (int)lev_size, 1024,
                                   cvvdp_normalize_task, &task);
            }
        }

        buf_a = temp;
        buf_b = reduced;
        memcpy(buf_a, dst_channels[ch], (size_t)w * h * sizeof(float));
        cw = w; ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            const size_t lev_size = (size_t)widths[lev] * heights[lev];
            if (lev < num_levels - 1) {
                cvvdp_gauss_pyr_reduce(pool, buf_a, buf_b, cw, ch_h);
                cvvdp_gauss_pyr_expand(pool, buf_b, expanded, cw, ch_h);
                CvvdpContrastTaskData task = {
                    .src = buf_a,
                    .expanded = expanded,
                    .L_bkg = L_bkg_pyr[lev],
                    .dst = dst_pyr[lev][ch],
                    .contrast_scale = lev == 0 ? 1.0f : 2.0f,
                };
                cvvdp_parallel_for(pool, (int)lev_size, 2048,
                                   cvvdp_contrast_task, &task);
                float* const t = buf_a;
                buf_a = buf_b;
                buf_b = t;
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else {
                CvvdpNormalizeTaskData task = {
                    .src = buf_a,
                    .dst = dst_pyr[lev][ch],
                    .denom = fmaxf(0.01f, L_bkg_pyr[lev][0]),
                };
                cvvdp_parallel_for(pool, (int)lev_size, 1024,
                                   cvvdp_normalize_task, &task);
            }
        }
    }

    const float blur_sigma = 3.0f;
    const int blur_radius = CVVDP_GAUSSIAN_SIZE;
    const float ch_gain[4] = {1.0f, 1.45f, 1.0f, 1.0f};
    double total_score = 0.0;
    float blur_kernel_sum = 0.0f;
    float blur_kernel[17];
    for (int i = 0; i < 17; i++) {
        const float d = (float)(i - 8);
        blur_kernel[i] = expf(-d * d / (2.0f * blur_sigma * blur_sigma));
        blur_kernel_sum += blur_kernel[i];
    }
    for (int i = 0; i < 17; i++)
        blur_kernel[i] /= blur_kernel_sum;

    for (int lev = 0; lev < num_levels; lev++) {
        const size_t lev_size = (size_t)widths[lev] * heights[lev];
        const int lev_w = widths[lev], lev_h = heights[lev];
        const int is_baseband = (lev == num_levels - 1);

        if (!is_baseband) {
            float* ref_level[4] = {
                ref_pyr[lev][0], ref_pyr[lev][1], ref_pyr[lev][2], ref_pyr[lev][3]
            };
            float* dst_level[4] = {
                dst_pyr[lev][0], dst_pyr[lev][1], dst_pyr[lev][2], dst_pyr[lev][3]
            };
            CvvdpCsfWeightTaskData csf_task = {
                .csf = &c->csf,
                .L_bkg = L_bkg_pyr[lev],
                .ref_level = ref_level,
                .dst_level = dst_level,
                .lev = lev,
                .lev_size = lev_size,
                .ch_gain = ch_gain,
            };
            cvvdp_parallel_for(pool, (int)(lev_size * 4), 2048,
                               cvvdp_csf_weight_task, &csf_task);

            float** const min_abs = c->pyr_min_abs;
            float** const blurred_min_abs = c->pyr_blurred_min_abs;
            float* const tmp_blur = c->pyr_tmp_blur;
            for (int ch = 0; ch < 4; ch++) {
                CvvdpMinAbsTaskData min_task = {
                    .ref = ref_pyr[lev][ch],
                    .dst = dst_pyr[lev][ch],
                    .out = min_abs[ch],
                };
                cvvdp_parallel_for(pool, (int)lev_size, 2048,
                                   cvvdp_min_abs_task, &min_task);

                CvvdpBlurTaskData blur_h = {
                    .src = min_abs[ch],
                    .dst = tmp_blur,
                    .width = lev_w,
                    .height = lev_h,
                    .kernel = blur_kernel,
                    .radius = blur_radius,
                };
                CvvdpBlurTaskData blur_v = {
                    .src = tmp_blur,
                    .dst = blurred_min_abs[ch],
                    .width = lev_w,
                    .height = lev_h,
                    .kernel = blur_kernel,
                    .radius = blur_radius,
                };
                cvvdp_parallel_for(pool, lev_h, 2,
                                   cvvdp_blur_horizontal_task, &blur_h);
                cvvdp_parallel_for(pool, lev_h, 2,
                                   cvvdp_blur_vertical_task, &blur_v);
            }

            float* const d = c->pyr_d;

            CvvdpMaskedDiffTaskData mask_task = {
                .blurred_min_abs = blurred_min_abs,
                .ref_level = ref_level,
                .dst_level = dst_level,
                .d = d,
                .lev_size = lev_size,
                .max_v = powf(10.0f, CVVDP_D_MAX),
                .pow_mask_c = powf(10.0f, CVVDP_MASK_C),
            };
            cvvdp_parallel_for(pool, (int)lev_size, 1024,
                               cvvdp_masked_diff_task, &mask_task);

            for (int ch = 0; ch < 4; ch++) {
                const float norm = cvvdp_compute_norm(pool, d + ch * lev_size,
                                                      (int)lev_size, 2);
                total_score += powf(norm, 4.0f);
            }
        } else {
            float* const d = c->pyr_d;
            const float* ref_level[4] = {
                ref_pyr[lev][0], ref_pyr[lev][1], ref_pyr[lev][2], ref_pyr[lev][3]
            };
            const float* dst_level[4] = {
                dst_pyr[lev][0], dst_pyr[lev][1], dst_pyr[lev][2], dst_pyr[lev][3]
            };
            CvvdpBasebandDiffTaskData base_task = {
                .ref_level = ref_level,
                .dst_level = dst_level,
                .d = d,
                .lev_size = lev_size,
                .sensitivity = {
                    cvvdp_csf_sensitivity(&c->csf, L_bkg_pyr[lev][0], lev, 0),
                    cvvdp_csf_sensitivity(&c->csf, L_bkg_pyr[lev][0], lev, 1),
                    cvvdp_csf_sensitivity(&c->csf, L_bkg_pyr[lev][0], lev, 2),
                    cvvdp_csf_sensitivity(&c->csf, L_bkg_pyr[lev][0], lev, 3),
                },
            };
            cvvdp_parallel_for(pool, (int)(lev_size * 4), 2048,
                               cvvdp_baseband_diff_task, &base_task);
            for (int ch = 0; ch < 4; ch++) {
                const float norm =
                    cvvdp_compute_norm(pool, d + ch * lev_size,
                                       (int)lev_size, 2);
                total_score += powf(norm, 4.0f);
            }
        }
    }

    *out_score = pow(total_score, 0.25);

    return err;
}

static FcvvdpError cvvdp_process_pyramid_serial(FcvvdpCtx* const c,
                                                float* const ref_channels[4],
                                                float* const dst_channels[4],
                                                double* const out_score)
{
    const int w = c->width, h = c->height;
    const int num_levels = c->num_bands;

    size_t total_size = 0;
    int widths[CVVDP_MAX_LEVELS], heights[CVVDP_MAX_LEVELS];
    int tw = w, th = h;
    for (int lev = 0; lev < num_levels; lev++) {
        widths[lev] = tw;
        heights[lev] = th;
        total_size += (size_t)tw * th;
        tw = (tw + 1) / 2;
        th = (th + 1) / 2;
    }

    float* (*ref_pyr)[4] = c->pyr_ref;
    float* (*dst_pyr)[4] = c->pyr_dst;
    float** const L_bkg_pyr = c->pyr_L_bkg;
    float* const temp = c->pyr_temp;
    float* const reduced = c->pyr_reduced;
    float* const expanded = c->pyr_expanded;

    float* buf_a = temp;
    float* buf_b = reduced;

    for (int ch = 0; ch < 4; ch++) {
        memcpy(buf_a, ref_channels[ch], (size_t)w * h * sizeof(float));
        int cw = w, ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            const size_t lev_size = (size_t)widths[lev] * heights[lev];
            if (lev < num_levels - 1) {
                cvvdp_gauss_pyr_reduce(NULL, buf_a, buf_b, cw, ch_h);
                cvvdp_gauss_pyr_expand(NULL, buf_b, expanded, cw, ch_h);
                if (!ch) {
                    for (size_t i = 0; i < lev_size; i++)
                        L_bkg_pyr[lev][i] = fmaxf(0.01f, expanded[i]);
                }
                for (size_t i = 0; i < lev_size; i++) {
                    const float contrast =
                        (buf_a[i] - expanded[i]) / fmaxf(0.01f, L_bkg_pyr[lev][i]);
                    ref_pyr[lev][ch][i] = contrast * (lev == 0 ? 1.0f : 2.0f);
                }
                float* const t = buf_a;
                buf_a = buf_b;
                buf_b = t;
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else {
                if (!ch) {
                    double mean = 0.0;
                    for (size_t i = 0; i < lev_size; i++)
                        mean += buf_a[i];
                    mean /= (double)lev_size;
                    L_bkg_pyr[lev][0] = fmaxf(0.01f, (float)mean);
                }
                const float denom = fmaxf(0.01f, L_bkg_pyr[lev][0]);
                for (size_t i = 0; i < lev_size; i++)
                    ref_pyr[lev][ch][i] = buf_a[i] / denom;
            }
        }

        buf_a = temp;
        buf_b = reduced;
        memcpy(buf_a, dst_channels[ch], (size_t)w * h * sizeof(float));
        cw = w;
        ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            const size_t lev_size = (size_t)widths[lev] * heights[lev];
            if (lev < num_levels - 1) {
                cvvdp_gauss_pyr_reduce(NULL, buf_a, buf_b, cw, ch_h);
                cvvdp_gauss_pyr_expand(NULL, buf_b, expanded, cw, ch_h);
                for (size_t i = 0; i < lev_size; i++) {
                    const float contrast =
                        (buf_a[i] - expanded[i]) / fmaxf(0.01f, L_bkg_pyr[lev][i]);
                    dst_pyr[lev][ch][i] = contrast * (lev == 0 ? 1.0f : 2.0f);
                }
                float* const t = buf_a;
                buf_a = buf_b;
                buf_b = t;
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else {
                const float denom = fmaxf(0.01f, L_bkg_pyr[lev][0]);
                for (size_t i = 0; i < lev_size; i++)
                    dst_pyr[lev][ch][i] = buf_a[i] / denom;
            }
        }
    }

    const float blur_sigma = 3.0f;
    const int blur_radius = CVVDP_GAUSSIAN_SIZE;
    const float ch_gain[4] = {1.0f, 1.45f, 1.0f, 1.0f};
    double total_score = 0.0;
    float blur_kernel_sum = 0.0f;
    float blur_kernel[17];
    for (int i = 0; i < 17; i++) {
        const float d = (float)(i - 8);
        blur_kernel[i] = expf(-d * d / (2.0f * blur_sigma * blur_sigma));
        blur_kernel_sum += blur_kernel[i];
    }
    for (int i = 0; i < 17; i++)
        blur_kernel[i] /= blur_kernel_sum;

    for (int lev = 0; lev < num_levels; lev++) {
        const size_t lev_size = (size_t)widths[lev] * heights[lev];
        const int lev_w = widths[lev], lev_h = heights[lev];
        const int is_baseband = (lev == num_levels - 1);

        if (!is_baseband) {
            for (int ch = 0; ch < 4; ch++)
                for (size_t i = 0; i < lev_size; i++) {
                    const float s = cvvdp_csf_sensitivity(&c->csf,
                                                          L_bkg_pyr[lev][i],
                                                          lev, ch);
                    ref_pyr[lev][ch][i] *= s * ch_gain[ch];
                    dst_pyr[lev][ch][i] *= s * ch_gain[ch];
                }

            float** const min_abs = c->pyr_min_abs;
            float** const blurred_min_abs = c->pyr_blurred_min_abs;
            float* const tmp_blur = c->pyr_tmp_blur;
            for (int ch = 0; ch < 4; ch++) {
                for (size_t i = 0; i < lev_size; i++) {
                    const float r = fabsf(ref_pyr[lev][ch][i]);
                    const float d_val = fabsf(dst_pyr[lev][ch][i]);
                    min_abs[ch][i] = fminf(r, d_val);
                }

                for (int y = 0; y < lev_h; y++) {
                    float* const tmp_blur_row = &tmp_blur[y * lev_w];
                    for (int x = 0; x < lev_w; x++) {
                        float sum = 0.0f;
                        float wsum = 0.0f;
                        for (int k = -blur_radius; k <= blur_radius; k++) {
                            const int sx = x + k;
                            if (sx >= 0 && sx < lev_w) {
                                sum += min_abs[ch][y * lev_w + sx] *
                                    blur_kernel[k + blur_radius];
                                wsum += blur_kernel[k + blur_radius];
                            }
                        }
                        tmp_blur_row[x] = sum / wsum;
                    }
                }

                for (int y = 0; y < lev_h; y++)
                    for (int x = 0; x < lev_w; x++) {
                        float sum = 0.0f;
                        float wsum = 0.0f;
                        for (int k = -blur_radius; k <= blur_radius; k++) {
                            const int sy = y + k;
                            if (sy >= 0 && sy < lev_h) {
                                sum += tmp_blur[sy * lev_w + x] *
                                    blur_kernel[k + blur_radius];
                                wsum += blur_kernel[k + blur_radius];
                            }
                        }
                        blurred_min_abs[ch][y * lev_w + x] = sum / wsum;
                    }
            }

            float* const d = c->pyr_d;

            const float max_v = powf(10.0f, CVVDP_D_MAX);
            const float pow_mask_c = powf(10.0f, CVVDP_MASK_C);

            for (size_t i = 0; i < lev_size; i++) {
                float cm[4];
                for (int ch = 0; ch < 4; ch++)
                    cm[ch] = powf(pow_mask_c * blurred_min_abs[ch][i],
                                  CVVDP_MASK_Q[ch]);
                for (int ch = 0; ch < 4; ch++) {
                    const float mask =
                        cm[0] * powf(2.0f, CVVDP_XCM_WEIGHTS[0 + ch]) +
                        cm[1] * powf(2.0f, CVVDP_XCM_WEIGHTS[4 + ch]) +
                        cm[2] * powf(2.0f, CVVDP_XCM_WEIGHTS[8 + ch]) +
                        cm[3] * powf(2.0f, CVVDP_XCM_WEIGHTS[12 + ch]);

                    const float diff =
                        fabsf(ref_pyr[lev][ch][i] - dst_pyr[lev][ch][i]);
                    const float du = powf(diff, CVVDP_MASK_P) / (1.0f + mask);
                    const float d_val = max_v * du / (max_v + du);

                    float weight = 1.0f;
                    switch (ch) {
                        case 1:
                        case 2:
                            weight = CVVDP_CH_CHROM_W;
                            break;
                        case 3:
                            weight = CVVDP_CH_TRANS_W;
                            break;
                    }

                    d[ch * lev_size + i] = d_val * weight;
                }
            }

            for (int ch = 0; ch < 4; ch++) {
                const float norm =
                    cvvdp_compute_norm(NULL, d + ch * lev_size, (int)lev_size, 2);
                total_score += powf(norm, 4.0f);
            }
        } else {
            float* const d = c->pyr_d;

            for (int ch = 0; ch < 4; ch++) {
                const float s =
                    cvvdp_csf_sensitivity(&c->csf, L_bkg_pyr[lev][0], lev, ch);
                float* const drow = &d[ch * lev_size];
                for (size_t i = 0; i < lev_size; i++) {
                    const float diff =
                        fabsf(ref_pyr[lev][ch][i] - dst_pyr[lev][ch][i]);
                    drow[i] = diff * s * CVVDP_BASEBAND_WEIGHT[ch];
                }
            }

            for (int ch = 0; ch < 4; ch++) {
                const float norm =
                    cvvdp_compute_norm(NULL, d + ch * lev_size, (int)lev_size, 2);
                total_score += powf(norm, 4.0f);
            }
        }
    }

    *out_score = pow(total_score, 0.25);
    return CVVDP_OK;
}

static FcvvdpError cvvdp_process_pyramid(FcvvdpCtx* const c,
                                         float* const ref_channels[4],
                                         float* const dst_channels[4],
                                         double* const out_score)
{
    if (!cvvdp_thread_pool_is_active(c->thread_pool))
        return cvvdp_process_pyramid_serial(c, ref_channels, dst_channels,
                                            out_score);
    return cvvdp_process_pyramid_threaded(c, ref_channels, dst_channels,
                                          out_score);
}

FcvvdpError cvvdp_create(const int width,
                         const int height,
                         const float fps,
                         const FcvvdpDisplayModel display_model,
                         const unsigned threads,
                         const FcvvdpDisplayParams* const custom_params,
                         FcvvdpCtx** const out_c)
{
    if (!out_c) return CVVDP_ERROR_NULL_POINTER;
    if (width <= 0 || height <= 0) return CVVDP_ERROR_INVALID_DIMENSIONS;

    FcvvdpCtx* const c = (FcvvdpCtx*)calloc(1, sizeof(FcvvdpCtx));
    if (!c) return CVVDP_ERROR_OUT_OF_MEMORY;

    c->width = width;
    c->height = height;
    c->fps = fps > 0 ? fps : 0;
    c->num_frames = 0;
    c->score_square_sum = 0.0;
    c->thread_pool = cvvdp_thread_pool_create(
        get_threadcnt(threads));

    cvvdp_init_display(&c->display, display_model, custom_params);

    c->band_frequencies = cvvdp_alloc_float(CVVDP_MAX_LEVELS);
    if (!c->band_frequencies) {
        free(c);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }
    c->num_bands = cvvdp_get_band_frequencies(width, height,
                                                c->display.ppd,
                                                c->band_frequencies);

    FcvvdpError err = cvvdp_csf_init(&c->csf, width, height,
                                     c->display.ppd);
    if (err != CVVDP_OK) {
        free(c->band_frequencies);
        free(c);
        return err;
    }

    err = cvvdp_temporal_ring_init(&c->ring_ref, width, height, fps);
    if (err != CVVDP_OK) {
        cvvdp_csf_destroy(&c->csf);
        free(c->band_frequencies);
        free(c);
        return err;
    }

    err = cvvdp_temporal_ring_init(&c->ring_dis, width, height, fps);
    if (err != CVVDP_OK) {
        cvvdp_temporal_ring_destroy(&c->ring_ref);
        cvvdp_csf_destroy(&c->csf);
        free(c->band_frequencies);
        free(c);
        return err;
    }

    const size_t plane_size = (size_t)width * height;
    c->Y_sustained = cvvdp_alloc_float(plane_size);
    c->RG_sustained = cvvdp_alloc_float(plane_size);
    c->YV_sustained = cvvdp_alloc_float(plane_size);
    c->Y_transient = cvvdp_alloc_float(plane_size);
    c->dst_Y_sustained = cvvdp_alloc_float(plane_size);
    c->dst_RG_sustained = cvvdp_alloc_float(plane_size);
    c->dst_YV_sustained = cvvdp_alloc_float(plane_size);
    c->dst_Y_transient = cvvdp_alloc_float(plane_size);
    c->work_buffer = cvvdp_alloc_float(plane_size * 3);

    size_t total_size = 0;
    int tw = width;
    int th = height;
    size_t max_level_size = plane_size;
    for (int lev = 0; lev < c->num_bands; lev++) {
        const size_t lev_size = (size_t)tw * th;
        total_size += lev_size;
        if (lev_size > max_level_size)
            max_level_size = lev_size;
        tw = (tw + 1) / 2;
        th = (th + 1) / 2;
    }

    c->pyr_temp = cvvdp_alloc_float(plane_size);
    c->pyr_reduced = cvvdp_alloc_float(total_size);
    c->pyr_expanded = cvvdp_alloc_float(plane_size);
    c->pyr_tmp_blur = cvvdp_alloc_float(max_level_size);
    c->pyr_d = cvvdp_alloc_float(max_level_size * CVVDP_NUM_CHANNELS);

    for (int ch = 0; ch < CVVDP_NUM_CHANNELS; ch++) {
        c->pyr_min_abs[ch] = cvvdp_alloc_float(max_level_size);
        c->pyr_blurred_min_abs[ch] = cvvdp_alloc_float(max_level_size);
    }

    tw = width;
    th = height;
    for (int lev = 0; lev < c->num_bands; lev++) {
        const size_t lev_size = (size_t)tw * th;
        for (int ch = 0; ch < CVVDP_NUM_CHANNELS; ch++) {
            c->pyr_ref[lev][ch] = cvvdp_alloc_float(lev_size);
            c->pyr_dst[lev][ch] = cvvdp_alloc_float(lev_size);
        }
        c->pyr_L_bkg[lev] = cvvdp_alloc_float(lev_size);
        tw = (tw + 1) / 2;
        th = (th + 1) / 2;
    }

    if (!c->Y_sustained || !c->RG_sustained || !c->YV_sustained ||
        !c->Y_transient || !c->dst_Y_sustained || !c->dst_RG_sustained ||
        !c->dst_YV_sustained || !c->dst_Y_transient || !c->work_buffer ||
        !c->pyr_temp || !c->pyr_reduced || !c->pyr_expanded ||
        !c->pyr_tmp_blur || !c->pyr_d)
    {
        cvvdp_destroy(c);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    for (int ch = 0; ch < CVVDP_NUM_CHANNELS; ch++) {
        if (!c->pyr_min_abs[ch] || !c->pyr_blurred_min_abs[ch]) {
            cvvdp_destroy(c);
            return CVVDP_ERROR_OUT_OF_MEMORY;
        }
    }

    for (int lev = 0; lev < c->num_bands; lev++) {
        if (!c->pyr_L_bkg[lev]) {
            cvvdp_destroy(c);
            return CVVDP_ERROR_OUT_OF_MEMORY;
        }
        for (int ch = 0; ch < CVVDP_NUM_CHANNELS; ch++) {
            if (!c->pyr_ref[lev][ch] || !c->pyr_dst[lev][ch]) {
                cvvdp_destroy(c);
                return CVVDP_ERROR_OUT_OF_MEMORY;
            }
        }
    }

    *out_c = c;
    return CVVDP_OK;
}

void cvvdp_destroy(FcvvdpCtx* const c) {
    if (!c) return;

    cvvdp_thread_pool_destroy(c->thread_pool);
    cvvdp_temporal_ring_destroy(&c->ring_ref);
    cvvdp_temporal_ring_destroy(&c->ring_dis);
    cvvdp_csf_destroy(&c->csf);

    free(c->band_frequencies);
    free(c->Y_sustained);
    free(c->RG_sustained);
    free(c->YV_sustained);
    free(c->Y_transient);
    free(c->dst_Y_sustained);
    free(c->dst_RG_sustained);
    free(c->dst_YV_sustained);
    free(c->dst_Y_transient);
    free(c->work_buffer);

    free(c->pyr_temp);
    free(c->pyr_reduced);
    free(c->pyr_expanded);
    free(c->pyr_tmp_blur);
    free(c->pyr_d);

    for (int ch = 0; ch < CVVDP_NUM_CHANNELS; ch++) {
        free(c->pyr_min_abs[ch]);
        free(c->pyr_blurred_min_abs[ch]);
    }

    for (int lev = 0; lev < c->num_bands; lev++) {
        for (int ch = 0; ch < CVVDP_NUM_CHANNELS; ch++) {
            free(c->pyr_ref[lev][ch]);
            free(c->pyr_dst[lev][ch]);
        }
        free(c->pyr_L_bkg[lev]);
    }

    free(c);
}

FcvvdpError cvvdp_reset(FcvvdpCtx* const c) {
    if (!c)
        return CVVDP_ERROR_NULL_POINTER;

    c->num_frames = 0;
    c->score_square_sum = 0.0;
    cvvdp_temporal_ring_reset(&c->ring_ref);
    cvvdp_temporal_ring_reset(&c->ring_dis);

    return CVVDP_OK;
}

FcvvdpError cvvdp_process_frame(FcvvdpCtx* const c,
                                const FcvvdpImage* const reference,
                                const FcvvdpImage* const distorted,
                                FcvvdpResult* const result)
{
    if (!c || !reference || !distorted || !result)
        return CVVDP_ERROR_NULL_POINTER;

    if (reference->width != c->width || reference->height != c->height ||
        distorted->width != c->width || distorted->height != c->height)
    {
        return CVVDP_ERROR_DIMENSION_MISMATCH;
    }

    const size_t plane_size = (size_t)c->width * c->height;
    float* const frame = c->work_buffer;

    FcvvdpError err = cvvdp_load_image(reference, frame);
    if (err != CVVDP_OK) return err;

    const bool is_hdr = c->display.is_hdr;
    cvvdp_apply_display_model_interleaved(c->thread_pool, frame,
                                          (int)plane_size, &c->display,
                                          is_hdr);
    cvvdp_rgb_to_xyz_interleaved(c->thread_pool, frame, (int)plane_size);
    cvvdp_xyz_to_dkl_interleaved(c->thread_pool, frame, (int)plane_size);
    cvvdp_temporal_ring_push(&c->ring_ref, frame);

    err = cvvdp_load_image(distorted, frame);
    if (err != CVVDP_OK) return err;

    cvvdp_apply_display_model_interleaved(c->thread_pool, frame,
                                          (int)plane_size, &c->display,
                                          is_hdr);
    cvvdp_rgb_to_xyz_interleaved(c->thread_pool, frame, (int)plane_size);
    cvvdp_xyz_to_dkl_interleaved(c->thread_pool, frame, (int)plane_size);
    cvvdp_temporal_ring_push(&c->ring_dis, frame);

    cvvdp_compute_temporal_channels(c->thread_pool, &c->ring_ref,
                                    c->Y_sustained, c->RG_sustained,
                                    c->YV_sustained, c->Y_transient);
    cvvdp_compute_temporal_channels(c->thread_pool, &c->ring_dis,
                                    c->dst_Y_sustained, c->dst_RG_sustained,
                                    c->dst_YV_sustained, c->dst_Y_transient);

    float* const ref_channels[4] =
        {c->Y_sustained, c->RG_sustained, c->YV_sustained, c->Y_transient};
    float* const dst_channels[4] =
        {c->dst_Y_sustained, c->dst_RG_sustained,
         c->dst_YV_sustained, c->dst_Y_transient};

    double current_score;
    err = cvvdp_process_pyramid(c, ref_channels, dst_channels,
                                &current_score);

    if (err != CVVDP_OK) return err;

    c->num_frames++;
    c->score_square_sum += pow(current_score, CVVDP_BETA_T);

    double resQ;
    if (c->num_frames == 1)
        resQ = current_score * CVVDP_IMAGE_INT;
    else
        resQ = pow(c->score_square_sum /
            (double)c->num_frames, 1.0 / CVVDP_BETA_T);

    result->quality = resQ;
    result->jod = cvvdp_to_jod((float)resQ);

    return CVVDP_OK;
}

FcvvdpError cvvdp_compare_images(const FcvvdpImage* const reference,
                                 const FcvvdpImage* const distorted,
                                 FcvvdpDisplayModel display_model,
                                 const unsigned threads,
                                 const FcvvdpDisplayParams* const custom_params,
                                 FcvvdpResult* const result)
{
    if (!reference || !distorted || !result) return CVVDP_ERROR_NULL_POINTER;

    if (reference->width != distorted->width || reference->height != distorted->height)
        return CVVDP_ERROR_DIMENSION_MISMATCH;

    FcvvdpCtx* c;
    FcvvdpError err = cvvdp_create(reference->width, reference->height, 0,
                                   display_model, threads, custom_params,
                                   &c);
    if (err != CVVDP_OK) return err;

    err = cvvdp_process_frame(c, reference, distorted, result);
    cvvdp_destroy(c);

    return err;
}
