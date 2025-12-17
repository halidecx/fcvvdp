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
#include "cvvdp.h"
#include "lut.h"
#include "util.h"
#include "internal.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static void cvvdp_gaussian_init(Gaussian* const gaussian) {
    const float sigma = (float)CVVDP_PU_DILATE;
    const unsigned size = 2 * CVVDP_GAUSSIAN_SIZE + 1;

    gaussian->kernel_integral[0] = 0.0f;

    for (unsigned i = 0; i < size; i++) {
        float x = (float)(i - CVVDP_GAUSSIAN_SIZE);
        gaussian->kernel[i] = expf(-x * x / (2.0f * sigma * sigma)) /
                              sqrtf((float)(TAU) * sigma * sigma);
        gaussian->kernel_integral[i + 1] =
            gaussian->kernel_integral[i] + gaussian->kernel[i];
    }
}

static void cvvdp_gaussian_blur(const Gaussian* gaussian, const float* src,
                         float* dst, int width, int height) {
    int ksize = CVVDP_GAUSSIAN_SIZE;
    float* temp = cvvdp_alloc_float((size_t)width * height);
    if (!temp) return;

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -ksize; k <= ksize; k++) {
                int sx = x + k;
                if (sx < 0) sx = -sx - 1;
                if (sx >= width) sx = 2 * width - sx - 2;

                float w = gaussian->kernel[k + ksize];
                sum += src[y * width + sx] * w;
                weight_sum += w;
            }
            temp[y * width + x] = sum / weight_sum;
        }

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -ksize; k <= ksize; k++) {
                int sy = y + k;
                if (sy < 0) sy = -sy - 1;
                if (sy >= height) sy = 2 * height - sy - 2;

                float w = gaussian->kernel[k + ksize];
                sum += temp[sy * width + x] * w;
                weight_sum += w;
            }
            dst[y * width + x] = sum / weight_sum;
        }

    free(temp);
}

// inverse real FFT for temporal filter generation
static void cvvdp_inverse_rfft(const float* input, int input_size, float* output, int size) {
    float* inp = cvvdp_alloc_float(input_size);
    if (!inp) return;

    memcpy(inp, input, input_size * sizeof(float));

    if (input_size > size / 2)
        inp[size / 2] /= 2.0f;
    inp[0] /= 2.0f;

    for (int i = 0; i < size; i++) {
        output[i] = 0.0f;
        for (int k = 0; k < input_size; k++) {
            int idx = (i + input_size - 1) % size;
            output[idx] += 2.0f * inp[k] * cosf(2.0f * (float)M_PI * k * i / size) / size;
        }
    }

    free(inp);
}

FcvvdpError cvvdp_temporal_filter_init(TemporalFilter* filter, float fps) {
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
        freq[i] = fps * i / 2.0f;

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

void cvvdp_temporal_filter_destroy(TemporalFilter* filter) {
    for (int j = 0; j < 4; j++)
        if (filter->kernel[j]) {
            free(filter->kernel[j]);
            filter->kernel[j] = NULL;
        }
}

FcvvdpError cvvdp_temporal_ring_init(TemporalRingBuf* ring, int width, int height, float fps) {
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

void cvvdp_temporal_ring_destroy(TemporalRingBuf* ring) {
    if (ring->data) {
        free(ring->data);
        ring->data = NULL;
    }
    cvvdp_temporal_filter_destroy(&ring->filter);
}

void cvvdp_temporal_ring_reset(TemporalRingBuf* ring) {
    ring->num_frames = 0;
    ring->current_index = 0;
}

float* cvvdp_temporal_ring_get_frame(TemporalRingBuf* ring, int age) {
    if (ring->num_frames == 0) return NULL;

    if (age >= ring->num_frames) {
        age = ring->num_frames - 1;
    }

    size_t plane_size = (size_t)ring->width * ring->height * 3;
    int idx = (ring->current_index + age) % ring->max_frames;
    return ring->data + idx * plane_size;
}

void cvvdp_temporal_ring_push(TemporalRingBuf* ring, const float* frame) {
    ring->num_frames = imin(ring->max_frames, ring->num_frames + 1);
    ring->current_index = (ring->current_index + ring->max_frames - 1) % ring->max_frames;

    size_t plane_size = (size_t)ring->width * ring->height * 3;
    float* dst = ring->data + ring->current_index * plane_size;
    memcpy(dst, frame, plane_size * sizeof(float));
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

void cvvdp_xyz_to_dkl(float* x, float* y, float* z, int count) {
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

FcvvdpError cvvdp_load_image(const FcvvdpImage* const img,
                             float* const out_planes[3])
{
    if (!img || !img->data || !out_planes) return CVVDP_ERROR_NULL_POINTER;

    const int w = img->width, h = img->height;
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

    float r, g, b;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const int idx = y * w + x;
            switch (img->format) {
                case CVVDP_PIXEL_FORMAT_RGB_FLOAT: {
                    const float* row =
                        (const float*)((const uint8_t*)img->data + y * stride);
                    r = row[x * 3 + 0];
                    g = row[x * 3 + 1];
                    b = row[x * 3 + 2];
                    break;
                }
                case CVVDP_PIXEL_FORMAT_RGB_UINT8: {
                    const uint8_t* row =
                        (const uint8_t*)img->data + y * stride;
                    r = row[x * 3 + 0] / 255.0f;
                    g = row[x * 3 + 1] / 255.0f;
                    b = row[x * 3 + 2] / 255.0f;
                    // sRGB gamma
                    r = (r < 0) ? -powf(-r, 2.4f) : powf(r, 2.4f);
                    g = (g < 0) ? -powf(-g, 2.4f) : powf(g, 2.4f);
                    b = (b < 0) ? -powf(-b, 2.4f) : powf(b, 2.4f);
                    break;
                }
                case CVVDP_PIXEL_FORMAT_RGB_UINT16: {
                    const uint16_t* row =
                        (const uint16_t*)((const uint8_t*)img->data
                                          + y * stride);
                    r = row[x * 3 + 0] / 65535.0f;
                    g = row[x * 3 + 1] / 65535.0f;
                    b = row[x * 3 + 2] / 65535.0f;
                    break;
                }
                default: return CVVDP_ERROR_INVALID_FORMAT;
            }
            out_planes[0][idx] = r;
            out_planes[1][idx] = g;
            out_planes[2][idx] = b;
        }

    return CVVDP_OK;
}

static void cvvdp_compute_temporal_channels(TemporalRingBuf* const ring,
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

    for (int k = 0; k < ring->filter.size; k++) {
        const float* const frame = cvvdp_temporal_ring_get_frame(ring, k);
        if (!frame) continue;

        const float k0 = ring->filter.kernel[0][k]; // Y sustained
        const float k1 = ring->filter.kernel[1][k]; // RG sustained
        const float k2 = ring->filter.kernel[2][k]; // YV sustained
        const float k3 = ring->filter.kernel[3][k]; // Y transient

        for (size_t i = 0; i < plane_size; i++) {
            const float y = frame[i];
            Y_sus[i] += y * k0;
            Y_trans[i] += y * k3;
            RG_sus[i] += frame[i + plane_size] * k1;
            YV_sus[i] += frame[i + 2 * plane_size] * k2;
        }
    }
}

static void cvvdp_gauss_pyr_reduce(const float* const src,
                                   float* const dst,
                                   const int src_w,
                                   const int src_h)
{
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
                    val +=
                        GAUSS_PYR_KERNEL[kx + 2] * GAUSS_PYR_KERNEL[ky + 2] *
                           src_row[sx];
                }
            }
            dst_row[dx] = val;
        }
    }
}

static void cvvdp_gauss_pyr_expand(const float* const src,
                                   float* const dst,
                                   const int dst_w,
                                   const int dst_h)
{
    const int src_w = (dst_w + 1) >> 1;
    const int src_h = (dst_h + 1) >> 1;

    for (int dy = 0; dy < dst_h; dy++) {
        float *const dst_row = &dst[dy * dst_w];
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
                    val +=
                        2.0f * GAUSS_PYR_KERNEL[kx + 2] *
                            2.0f * GAUSS_PYR_KERNEL[ky + 2] *
                                src_row[sx];
                }
            }
            dst_row[dx] = val;
        }
    }
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

static float cvvdp_compute_norm(const float* const data,
                                const int count,
                                const int power)
{
    double sum = 0.0;
    for (int i = 0; i < count; i++)
        sum += pow(fabs((double)data[i]), power);
    return (float)pow(sum / count, 1.0 / power);
}

static FcvvdpError cvvdp_process_pyramid(FcvvdpCtx* const c,
                                         float* const ref_channels[4],
                                         float* const dst_channels[4],
                                         double* const out_score) {
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

    float* ref_pyr[CVVDP_MAX_LEVELS][4];
    float* dst_pyr[CVVDP_MAX_LEVELS][4];
    float* L_bkg_pyr[CVVDP_MAX_LEVELS];
    float* temp = cvvdp_alloc_float((size_t)w * h);
    float* reduced = cvvdp_alloc_float(total_size);
    float* expanded = cvvdp_alloc_float((size_t)w * h);

    if (!temp || !reduced || !expanded) {
        free(temp);
        free(reduced);
        free(expanded);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    for (int lev = 0; lev < num_levels; lev++) {
        const size_t lev_size = (size_t)widths[lev] * heights[lev];
        for (int ch = 0; ch < 4; ch++) {
            ref_pyr[lev][ch] = cvvdp_alloc_float(lev_size);
            dst_pyr[lev][ch] = cvvdp_alloc_float(lev_size);
            if (!ref_pyr[lev][ch] || !dst_pyr[lev][ch]) {
                for (int l = 0; l <= lev; l++) {
                    for (int c = 0; c < 4; c++) {
                        free(ref_pyr[l][c]);
                        free(dst_pyr[l][c]);
                    }
                    free(L_bkg_pyr[l]);
                }
                free(temp);
                free(reduced);
                free(expanded);
                return CVVDP_ERROR_OUT_OF_MEMORY;
            }
        }
        L_bkg_pyr[lev] = cvvdp_alloc_float(lev_size);
        if (!L_bkg_pyr[lev]) {
            for (int l = 0; l <= lev; l++) {
                for (int c = 0; c < 4; c++) {
                    free(ref_pyr[l][c]);
                    free(dst_pyr[l][c]);
                }
                free(L_bkg_pyr[l]);
            }
            free(temp);
            free(reduced);
            free(expanded);
            return CVVDP_ERROR_OUT_OF_MEMORY;
        }
    }

    for (int ch = 0; ch < 4; ch++) {
        memcpy(temp, ref_channels[ch], (size_t)w * h * sizeof(float));
        int cw = w, ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            const size_t lev_size = (size_t)widths[lev] * heights[lev];
            if (lev < num_levels - 1) {
                cvvdp_gauss_pyr_reduce(temp, reduced, cw, ch_h);
                if (!ch) {
                    cvvdp_gauss_pyr_expand(reduced, expanded, cw, ch_h);
                    for (size_t i = 0; i < lev_size; i++)
                        L_bkg_pyr[lev][i] = fmax(0.01f, expanded[i]);
                }
                cvvdp_gauss_pyr_expand(reduced, expanded, cw, ch_h);
                for (size_t i = 0; i < lev_size; i++) {
                    const float contrast = (temp[i] - expanded[i]) /
                        fmax(0.01f, L_bkg_pyr[lev][i]);
                    ref_pyr[lev][ch][i] = contrast * (lev == 0 ? 1.0f : 2.0f);
                }
                memcpy(temp, reduced,
                       (size_t)((cw+1)/2) * ((ch_h+1)/2) * sizeof(float));
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else {
                if (!ch) {
                    float mean = 0.0f;
                    for (size_t i = 0; i < lev_size; i++)
                        mean += temp[i];
                    mean /= lev_size;
                    L_bkg_pyr[lev][0] = fmax(0.01f, mean);
                }
                for (size_t i = 0; i < lev_size; i++)
                    ref_pyr[lev][ch][i] = temp[i] /
                        fmax(0.01f, L_bkg_pyr[lev][0]);
            }
        }

        memcpy(temp, dst_channels[ch], (size_t)w * h * sizeof(float));
        cw = w; ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            const size_t lev_size = (size_t)widths[lev] * heights[lev];
            if (lev < num_levels - 1) {
                cvvdp_gauss_pyr_reduce(temp, reduced, cw, ch_h);
                cvvdp_gauss_pyr_expand(reduced, expanded, cw, ch_h);
                for (size_t i = 0; i < lev_size; i++) {
                    const float contrast = (temp[i] - expanded[i]) /
                        fmax(0.01f, L_bkg_pyr[lev][i]);
                    dst_pyr[lev][ch][i] = contrast * (lev == 0 ? 1.0f : 2.0f);
                }
                memcpy(temp, reduced,
                       (size_t)((cw+1)/2) * ((ch_h+1)/2) * sizeof(float));
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else
                for (size_t i = 0; i < lev_size; i++)
                    dst_pyr[lev][ch][i] = temp[i] /
                        fmax(0.01f, L_bkg_pyr[lev][0]);
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

            float* min_abs[4];
            float* blurred_min_abs[4];
            for (int ch = 0; ch < 4; ch++) {
                min_abs[ch] = cvvdp_alloc_float(lev_size);
                blurred_min_abs[ch] = cvvdp_alloc_float(lev_size);
                if (!min_abs[ch] || !blurred_min_abs[ch]) {
                    for (int c = 0; c <= ch; c++) {
                        free(min_abs[c]);
                        free(blurred_min_abs[c]);
                    }
                    continue;
                }

                for (size_t i = 0; i < lev_size; i++) {
                    float r = fabsf(ref_pyr[lev][ch][i]);
                    float d = fabsf(dst_pyr[lev][ch][i]);
                    min_abs[ch][i] = fmin(r, d);
                }

                float* const tmp_blur = cvvdp_alloc_float(lev_size);
                if (tmp_blur) {
                    for (int y = 0; y < lev_h; y++) {
                        float* const tmp_blur_row = &tmp_blur[y * lev_w];
                        for (int x = 0; x < lev_w; x++) {
                            float sum = 0.0f, wsum = 0.0f;
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
                            float sum = 0.0f, wsum = 0.0f;
                            for (int k = -blur_radius; k <= blur_radius; k++) {
                                int sy = y + k;
                                if (sy >= 0 && sy < lev_h) {
                                    sum += tmp_blur[sy * lev_w + x] *
                                        blur_kernel[k + blur_radius];
                                    wsum += blur_kernel[k + blur_radius];
                                }
                            }
                            blurred_min_abs[ch][y * lev_w + x] = sum / wsum;
                        }
                    free(tmp_blur);
                } else memcpy(blurred_min_abs[ch], min_abs[ch],
                              lev_size * sizeof(float));
            }

            float* const d = cvvdp_alloc_float(lev_size * 4);
            if (!d) {
                for (int ch = 0; ch < 4; ch++) {
                    free(min_abs[ch]);
                    free(blurred_min_abs[ch]);
                }
                continue;
            }

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
                free(min_abs[ch]);
                free(blurred_min_abs[ch]);
            }

            for (int ch = 0; ch < 4; ch++) {
                const float norm = cvvdp_compute_norm(d + ch * lev_size,
                                                      (int)lev_size, 2);
                total_score += pow(norm, 4);
            }

            free(d);
        } else {
            float* const d = cvvdp_alloc_float(lev_size * 4);
            if (!d) continue;
            for (int ch = 0; ch < 4; ch++) {
                const float s =
                    cvvdp_csf_sensitivity(&c->csf, L_bkg_pyr[lev][0],
                                          lev, ch);
                float* const drow = &d[ch * lev_size];
                for (size_t i = 0; i < lev_size; i++) {
                    const float diff =
                        fabsf(ref_pyr[lev][ch][i] - dst_pyr[lev][ch][i]);
                    drow[i] = diff * s * CVVDP_BASEBAND_WEIGHT[ch];
                }
            }
            for (int ch = 0; ch < 4; ch++) {
                const float norm =
                    cvvdp_compute_norm(d + ch * lev_size,
                                       (int)lev_size, 2);
                total_score += pow(norm, 4);
            }
            free(d);
        }
    }

    for (int lev = 0; lev < num_levels; lev++) {
        for (int ch = 0; ch < 4; ch++) {
            free(ref_pyr[lev][ch]);
            free(dst_pyr[lev][ch]);
        }
        free(L_bkg_pyr[lev]);
    }
    free(temp);
    free(reduced);
    free(expanded);

    *out_score = pow(total_score, 0.25);
    return CVVDP_OK;
}

FcvvdpError cvvdp_create(const int width,
                         const int height,
                         const float fps,
                         const FcvvdpDisplayModel display_model,
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

    cvvdp_init_display(&c->display, display_model, custom_params);

    cvvdp_gaussian_init(&c->gaussian);

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
    c->work_buffer = cvvdp_alloc_float(plane_size * 3);

    if (!c->Y_sustained || !c->RG_sustained || !c->YV_sustained ||
        !c->Y_transient || !c->work_buffer)
    {
        cvvdp_destroy(c);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    *out_c = c;
    return CVVDP_OK;
}

void cvvdp_destroy(FcvvdpCtx* const c) {
    if (!c) return;

    cvvdp_temporal_ring_destroy(&c->ring_ref);
    cvvdp_temporal_ring_destroy(&c->ring_dis);
    cvvdp_csf_destroy(&c->csf);

    free(c->band_frequencies);
    free(c->Y_sustained);
    free(c->RG_sustained);
    free(c->YV_sustained);
    free(c->Y_transient);
    free(c->work_buffer);

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
    float* const ref_planes[3] = {
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size)
    };
    float* const dst_planes[3] = {
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size)
    };

    if (!ref_planes[0] || !ref_planes[1] || !ref_planes[2] ||
        !dst_planes[0] || !dst_planes[1] || !dst_planes[2])
    {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dst_planes[i]);
        }
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    FcvvdpError err = cvvdp_load_image(reference, ref_planes);
    if (err != CVVDP_OK) {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dst_planes[i]);
        }
        return err;
    }

    err = cvvdp_load_image(distorted, dst_planes);
    if (err != CVVDP_OK) {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dst_planes[i]);
        }
        return err;
    }

    const bool is_hdr = c->display.is_hdr;
    for (int i = 0; i < 3; i++) {
        cvvdp_apply_display_model(ref_planes[i], (int)plane_size,
                                  &c->display, is_hdr);
        cvvdp_apply_display_model(dst_planes[i], (int)plane_size,
                                  &c->display, is_hdr);
    }

    cvvdp_rgb_to_xyz(ref_planes[0], ref_planes[1], ref_planes[2],
                     (int)plane_size);
    cvvdp_rgb_to_xyz(dst_planes[0], dst_planes[1], dst_planes[2],
                     (int)plane_size);
    cvvdp_xyz_to_dkl(ref_planes[0], ref_planes[1], ref_planes[2],
                     (int)plane_size);
    cvvdp_xyz_to_dkl(dst_planes[0], dst_planes[1], dst_planes[2],
                     (int)plane_size);

    float* const ref_frame = cvvdp_alloc_float(plane_size * 3);
    float* const dst_frame = cvvdp_alloc_float(plane_size * 3);
    if (!ref_frame || !dst_frame) {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dst_planes[i]);
        }
        free(ref_frame);
        free(dst_frame);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    memcpy(ref_frame, ref_planes[0], plane_size * sizeof(float));
    memcpy(ref_frame + plane_size, ref_planes[1], plane_size * sizeof(float));
    memcpy(ref_frame + 2 * plane_size, ref_planes[2],
           plane_size * sizeof(float));
    memcpy(dst_frame, dst_planes[0], plane_size * sizeof(float));
    memcpy(dst_frame + plane_size, dst_planes[1], plane_size * sizeof(float));
    memcpy(dst_frame + 2 * plane_size, dst_planes[2],
           plane_size * sizeof(float));

    cvvdp_temporal_ring_push(&c->ring_ref, ref_frame);
    cvvdp_temporal_ring_push(&c->ring_dis, dst_frame);

    free(ref_frame);
    free(dst_frame);
    for (int i = 0; i < 3; i++) {
        free(ref_planes[i]);
        free(dst_planes[i]);
    }

    float* const ref_Y_sus = cvvdp_alloc_float(plane_size);
    float* const ref_RG_sus = cvvdp_alloc_float(plane_size);
    float* const ref_YV_sus = cvvdp_alloc_float(plane_size);
    float* const ref_Y_trans = cvvdp_alloc_float(plane_size);
    float* const dst_Y_sus = cvvdp_alloc_float(plane_size);
    float* const dst_RG_sus = cvvdp_alloc_float(plane_size);
    float* const dst_YV_sus = cvvdp_alloc_float(plane_size);
    float* const dst_Y_trans = cvvdp_alloc_float(plane_size);

    if (!ref_Y_sus || !ref_RG_sus || !ref_YV_sus || !ref_Y_trans ||
        !dst_Y_sus || !dst_RG_sus || !dst_YV_sus || !dst_Y_trans)
    {
        free(ref_Y_sus);
        free(ref_RG_sus);
        free(ref_YV_sus);
        free(ref_Y_trans);
        free(dst_Y_sus);
        free(dst_RG_sus);
        free(dst_YV_sus);
        free(dst_Y_trans);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    cvvdp_compute_temporal_channels(&c->ring_ref, ref_Y_sus, ref_RG_sus,
                                    ref_YV_sus, ref_Y_trans);
    cvvdp_compute_temporal_channels(&c->ring_dis, dst_Y_sus, dst_RG_sus,
                                    dst_YV_sus, dst_Y_trans);

    float* const ref_channels[4] =
        {ref_Y_sus, ref_RG_sus, ref_YV_sus, ref_Y_trans};
    float* const dst_channels[4] =
        {dst_Y_sus, dst_RG_sus, dst_YV_sus, dst_Y_trans};

    double current_score;
    err = cvvdp_process_pyramid(c, ref_channels, dst_channels,
                                &current_score);

    free(ref_Y_sus);
    free(ref_RG_sus);
    free(ref_YV_sus);
    free(ref_Y_trans);
    free(dst_Y_sus);
    free(dst_RG_sus);
    free(dst_YV_sus);
    free(dst_Y_trans);

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
                                 const FcvvdpDisplayParams* const custom_params,
                                 FcvvdpResult* const result)
{
    if (!reference || !distorted || !result) return CVVDP_ERROR_NULL_POINTER;

    if (reference->width != distorted->width || reference->height != distorted->height)
        return CVVDP_ERROR_DIMENSION_MISMATCH;

    FcvvdpCtx* c;
    FcvvdpError err = cvvdp_create(reference->width, reference->height, 0,
                                      display_model, custom_params, &c);
    if (err != CVVDP_OK) return err;

    err = cvvdp_process_frame(c, reference, distorted, result);
    cvvdp_destroy(c);

    return err;
}
