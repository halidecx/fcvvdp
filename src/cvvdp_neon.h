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
#ifndef FCVVDP_NEON_H
#define FCVVDP_NEON_H

#include <arm_neon.h>

#include "internal.h"
#include "util.h"

static inline float32x4_t cvvdp_neon_madd_n(const float32x4_t acc,
                                            const float32x4_t v,
                                            const float scale)
{
    return vaddq_f32(acc, vmulq_n_f32(v, scale));
}

static inline float32x4_t cvvdp_neon_load_even4(const float* const p)
{
    return vld2q_f32(p).val[0];
}

static inline float cvvdp_neon_kernel_sum(const float* const kernel,
                                          const int radius)
{
    float sum = 0.0f;
    for (int k = -radius; k <= radius; k++)
        sum += kernel[k + radius];
    return sum;
}

static inline void cvvdp_apply_display_impl(
    const CvvdpApplyDisplayTaskData* const data,
    const int start,
    const int end)
{
    const float Y_peak = data->display->max_luminance;
    const float Y_black = data->display->black_level;
    const float Y_refl = data->display->refl_level;
    const float exposure = data->display->exposure;
    const float offset = Y_black + Y_refl;
    int i = start;

    if (data->is_hdr) {
        const float clamp_min = fmaxf(0.005f, Y_black);
        const float scale = 100.0f * exposure;
        const float32x4_t vscale = vdupq_n_f32(scale);
        const float32x4_t vmin = vdupq_n_f32(clamp_min);
        const float32x4_t vmax = vdupq_n_f32(Y_peak);
        const float32x4_t voffset = vdupq_n_f32(offset);
        for (; i + 4 <= end; i += 4) {
            float32x4_t val = vmulq_f32(vld1q_f32(data->plane + i), vscale);
            val = vmaxq_f32(vmin, vminq_f32(vmax, val));
            vst1q_f32(data->plane + i, vaddq_f32(val, voffset));
        }
    } else {
        const float scale = Y_peak - Y_black;
        const float32x4_t vexposure = vdupq_n_f32(exposure);
        const float32x4_t vzero = vdupq_n_f32(0.0f);
        const float32x4_t vone = vdupq_n_f32(1.0f);
        const float32x4_t vscale = vdupq_n_f32(scale);
        const float32x4_t voffset = vdupq_n_f32(offset);
        for (; i + 4 <= end; i += 4) {
            float32x4_t val = vmulq_f32(vld1q_f32(data->plane + i), vexposure);
            val = vmaxq_f32(vzero, vminq_f32(vone, val));
            vst1q_f32(data->plane + i,
                      vaddq_f32(vmulq_f32(val, vscale), voffset));
        }
    }

    for (; i < end; i++) {
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

static inline void cvvdp_rgb_to_xyz_impl(
    const CvvdpColorTransformTaskData* const data,
    const int start,
    const int end)
{
    int i = start;
    for (; i + 4 <= end; i += 4) {
        const float32x4_t r = vld1q_f32(data->x + i);
        const float32x4_t g = vld1q_f32(data->y + i);
        const float32x4_t b = vld1q_f32(data->z + i);

        float32x4_t x = vmulq_n_f32(r, 0.4124564f);
        x = cvvdp_neon_madd_n(x, g, 0.3575761f);
        x = cvvdp_neon_madd_n(x, b, 0.1804375f);

        float32x4_t y = vmulq_n_f32(r, 0.2126729f);
        y = cvvdp_neon_madd_n(y, g, 0.7151522f);
        y = cvvdp_neon_madd_n(y, b, 0.0721750f);

        float32x4_t z = vmulq_n_f32(r, 0.0193339f);
        z = cvvdp_neon_madd_n(z, g, 0.1191920f);
        z = cvvdp_neon_madd_n(z, b, 0.9503041f);

        vst1q_f32(data->x + i, x);
        vst1q_f32(data->y + i, y);
        vst1q_f32(data->z + i, z);
    }

    for (; i < end; i++) {
        const float ri = data->x[i];
        const float gi = data->y[i];
        const float bi = data->z[i];

        data->x[i] = 0.4124564f * ri + 0.3575761f * gi + 0.1804375f * bi;
        data->y[i] = 0.2126729f * ri + 0.7151522f * gi + 0.0721750f * bi;
        data->z[i] = 0.0193339f * ri + 0.1191920f * gi + 0.9503041f * bi;
    }
}

static inline void cvvdp_xyz_to_dkl_impl(
    const CvvdpColorTransformTaskData* const data,
    const int start,
    const int end)
{
    int i = start;
    for (; i + 4 <= end; i += 4) {
        const float32x4_t x = vld1q_f32(data->x + i);
        const float32x4_t y = vld1q_f32(data->y + i);
        const float32x4_t z = vld1q_f32(data->z + i);

        float32x4_t L = vmulq_n_f32(x, 0.187596268556126f);
        L = cvvdp_neon_madd_n(L, y, 0.585168649077728f);
        L = cvvdp_neon_madd_n(L, z, -0.026384263306304f);

        float32x4_t M = vmulq_n_f32(x, -0.133397430663221f);
        M = cvvdp_neon_madd_n(M, y, 0.405505777260049f);
        M = cvvdp_neon_madd_n(M, z, 0.034502127690364f);

        float32x4_t S = vmulq_n_f32(x, 0.000244379021663f);
        S = cvvdp_neon_madd_n(S, y, -0.000542995890619f);
        S = cvvdp_neon_madd_n(S, z, 0.019406849066323f);

        const float32x4_t lum = vaddq_f32(L, M);
        const float32x4_t rg =
            vsubq_f32(lum, vmulq_n_f32(M, 3.311130179947035f));
        const float32x4_t yv =
            vsubq_f32(vmulq_n_f32(S, 50.977571328718781f), lum);

        vst1q_f32(data->x + i, lum);
        vst1q_f32(data->y + i, rg);
        vst1q_f32(data->z + i, yv);
    }

    for (; i < end; i++) {
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

static inline void cvvdp_contrast_impl(
    const CvvdpContrastTaskData* const data,
    const int start,
    const int end)
{
    const float32x4_t vfloor = vdupq_n_f32(0.01f);
    const float32x4_t vscale = vdupq_n_f32(data->contrast_scale);
    int i = start;
    for (; i + 4 <= end; i += 4) {
        const float32x4_t src = vld1q_f32(data->src + i);
        const float32x4_t expanded = vld1q_f32(data->expanded + i);
        const float32x4_t L_bkg =
            vmaxq_f32(vfloor, vld1q_f32(data->L_bkg + i));
        const float32x4_t contrast =
            vdivq_f32(vsubq_f32(src, expanded), L_bkg);
        vst1q_f32(data->dst + i, vmulq_f32(contrast, vscale));
    }

    for (; i < end; i++) {
        data->dst[i] =
            ((data->src[i] - data->expanded[i]) / fmaxf(0.01f, data->L_bkg[i])) *
            data->contrast_scale;
    }
}

static inline void cvvdp_luma_contrast_impl(
    const CvvdpContrastTaskData* const data,
    const int start,
    const int end)
{
    const float32x4_t vfloor = vdupq_n_f32(0.01f);
    const float32x4_t vscale = vdupq_n_f32(data->contrast_scale);
    int i = start;
    for (; i + 4 <= end; i += 4) {
        const float32x4_t src = vld1q_f32(data->src + i);
        const float32x4_t expanded = vld1q_f32(data->expanded + i);
        const float32x4_t L_bkg = vmaxq_f32(vfloor, expanded);
        const float32x4_t contrast =
            vdivq_f32(vsubq_f32(src, expanded), L_bkg);
        vst1q_f32(data->L_bkg + i, L_bkg);
        vst1q_f32(data->dst + i, vmulq_f32(contrast, vscale));
    }

    for (; i < end; i++) {
        const float L_bkg = fmaxf(0.01f, data->expanded[i]);
        data->L_bkg[i] = L_bkg;
        data->dst[i] =
            ((data->src[i] - data->expanded[i]) / L_bkg) *
            data->contrast_scale;
    }
}

static inline void cvvdp_normalize_impl(
    const CvvdpNormalizeTaskData* const data,
    const int start,
    const int end)
{
    const float32x4_t denom = vdupq_n_f32(data->denom);
    int i = start;
    for (; i + 4 <= end; i += 4) {
        vst1q_f32(data->dst + i,
                  vdivq_f32(vld1q_f32(data->src + i), denom));
    }

    for (; i < end; i++)
        data->dst[i] = data->src[i] / data->denom;
}

static inline void cvvdp_min_abs_impl(
    const CvvdpMinAbsTaskData* const data,
    const int start,
    const int end)
{
    int i = start;
    for (; i + 4 <= end; i += 4) {
        const float32x4_t ref = vabsq_f32(vld1q_f32(data->ref + i));
        const float32x4_t dst = vabsq_f32(vld1q_f32(data->dst + i));
        vst1q_f32(data->out + i, vminq_f32(ref, dst));
    }

    for (; i < end; i++)
        data->out[i] = fminf(fabsf(data->ref[i]), fabsf(data->dst[i]));
}

static inline void cvvdp_baseband_diff_impl(
    const CvvdpBasebandDiffTaskData* const data,
    const int start,
    const int end)
{
    const int lev_size = (int)data->lev_size;
    int idx = start;

    while (idx < end) {
        const int ch = idx / lev_size;
        const int ch_end = imin(end, (ch + 1) * lev_size);
        const float scale = data->sensitivity[ch] * CVVDP_BASEBAND_WEIGHT[ch];
        const float32x4_t vscale = vdupq_n_f32(scale);
        int i = idx - ch * lev_size;

        for (; idx + 4 <= ch_end; idx += 4, i += 4) {
            const float32x4_t ref = vld1q_f32(data->ref_level[ch] + i);
            const float32x4_t dst = vld1q_f32(data->dst_level[ch] + i);
            const float32x4_t diff = vabsq_f32(vsubq_f32(ref, dst));
            vst1q_f32(data->d + idx, vmulq_f32(diff, vscale));
        }

        for (; idx < ch_end; idx++, i++) {
            const float diff =
                fabsf(data->ref_level[ch][i] - data->dst_level[ch][i]);
            data->d[idx] = diff * scale;
        }
    }
}

static inline const float* cvvdp_temporal_ring_frame_neon(
    const TemporalRingBuf* const ring,
    int age)
{
    if (!ring->num_frames) return NULL;

    if (age >= ring->num_frames) age = ring->num_frames - 1;

    const size_t frame_size = (size_t)ring->width * ring->height * 3;
    const int idx = (ring->current_index + age) % ring->max_frames;
    return ring->data + idx * frame_size;
}

static inline void cvvdp_compute_temporal_channels_impl(
    const CvvdpTemporalChannelsTaskData* const data,
    const int start,
    const int end)
{
    const size_t plane_size = data->plane_size;
    int i = start;

    for (; i + 4 <= end; i += 4) {
        float32x4_t Y_sus = vdupq_n_f32(0.0f);
        float32x4_t RG_sus = vdupq_n_f32(0.0f);
        float32x4_t YV_sus = vdupq_n_f32(0.0f);
        float32x4_t Y_trans = vdupq_n_f32(0.0f);

        for (int k = 0; k < data->ring->filter.size; k++) {
            const float* const frame =
                cvvdp_temporal_ring_frame_neon(data->ring, k);
            if (!frame) continue;

            const float32x4_t y = vld1q_f32(frame + i);
            const float32x4_t rg = vld1q_f32(frame + plane_size + i);
            const float32x4_t yv = vld1q_f32(frame + 2 * plane_size + i);

            Y_sus = cvvdp_neon_madd_n(Y_sus, y,
                                      data->ring->filter.kernel[0][k]);
            Y_trans = cvvdp_neon_madd_n(Y_trans, y,
                                        data->ring->filter.kernel[3][k]);
            RG_sus = cvvdp_neon_madd_n(RG_sus, rg,
                                       data->ring->filter.kernel[1][k]);
            YV_sus = cvvdp_neon_madd_n(YV_sus, yv,
                                       data->ring->filter.kernel[2][k]);
        }

        vst1q_f32(data->Y_sus + i, Y_sus);
        vst1q_f32(data->RG_sus + i, RG_sus);
        vst1q_f32(data->YV_sus + i, YV_sus);
        vst1q_f32(data->Y_trans + i, Y_trans);
    }

    for (; i < end; i++) {
        float Y_sus = 0.0f;
        float RG_sus = 0.0f;
        float YV_sus = 0.0f;
        float Y_trans = 0.0f;

        for (int k = 0; k < data->ring->filter.size; k++) {
            const float* const frame =
                cvvdp_temporal_ring_frame_neon(data->ring, k);
            if (!frame) continue;

            const float y = frame[i];
            Y_sus += y * data->ring->filter.kernel[0][k];
            Y_trans += y * data->ring->filter.kernel[3][k];
            RG_sus += frame[i + plane_size] * data->ring->filter.kernel[1][k];
            YV_sus += frame[i + 2 * plane_size] *
                data->ring->filter.kernel[2][k];
        }

        data->Y_sus[i] = Y_sus;
        data->RG_sus[i] = RG_sus;
        data->YV_sus[i] = YV_sus;
        data->Y_trans[i] = Y_trans;
    }
}

static inline void cvvdp_blur_horizontal_impl(
    const CvvdpBlurTaskData* const data,
    const int start,
    const int end)
{
    const int width = data->width;
    const int height = data->height;
    const int radius = data->radius;
    const int row_end = imin(end, height);
    const float full_wsum = cvvdp_neon_kernel_sum(data->kernel, radius);
    const float32x4_t inv_full_wsum = vdupq_n_f32(1.0f / full_wsum);

    for (int y = start; y < row_end; y++) {
        const float* const src_row = data->src + (size_t)y * width;
        float* const dst_row = data->dst + (size_t)y * width;
        int x = 0;

        for (; x < width && x < radius; x++) {
            float sum = 0.0f;
            float wsum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                const int sx = x + k;
                if (sx >= 0 && sx < width) {
                    const float weight = data->kernel[k + radius];
                    sum += src_row[sx] * weight;
                    wsum += weight;
                }
            }
            dst_row[x] = sum / wsum;
        }

        const int interior_end = width - radius;
        for (; x + 4 <= interior_end; x += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int k = -radius; k <= radius; k++) {
                const float32x4_t src = vld1q_f32(src_row + x + k);
                sum = cvvdp_neon_madd_n(sum, src, data->kernel[k + radius]);
            }
            vst1q_f32(dst_row + x, vmulq_f32(sum, inv_full_wsum));
        }

        for (; x < width; x++) {
            float sum = 0.0f;
            float wsum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                const int sx = x + k;
                if (sx >= 0 && sx < width) {
                    const float weight = data->kernel[k + radius];
                    sum += src_row[sx] * weight;
                    wsum += weight;
                }
            }
            dst_row[x] = sum / wsum;
        }
    }
}

static inline void cvvdp_blur_vertical_impl(
    const CvvdpBlurTaskData* const data,
    const int start,
    const int end)
{
    const int width = data->width;
    const int height = data->height;
    const int radius = data->radius;
    const int row_end = imin(end, height);
    const float full_wsum = cvvdp_neon_kernel_sum(data->kernel, radius);
    const float32x4_t inv_full_wsum = vdupq_n_f32(1.0f / full_wsum);

    for (int y = start; y < row_end; y++) {
        float* const dst_row = data->dst + (size_t)y * width;

        if (y < radius || y >= height - radius) {
            for (int x = 0; x < width; x++) {
                float sum = 0.0f;
                float wsum = 0.0f;
                for (int k = -radius; k <= radius; k++) {
                    const int sy = y + k;
                    if (sy >= 0 && sy < height) {
                        const float weight = data->kernel[k + radius];
                        sum += data->src[(size_t)sy * width + x] * weight;
                        wsum += weight;
                    }
                }
                dst_row[x] = sum / wsum;
            }
            continue;
        }

        int x = 0;
        for (; x + 4 <= width; x += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int k = -radius; k <= radius; k++) {
                const float* const src_row =
                    data->src + (size_t)(y + k) * width;
                const float32x4_t src = vld1q_f32(src_row + x);
                sum = cvvdp_neon_madd_n(sum, src, data->kernel[k + radius]);
            }
            vst1q_f32(dst_row + x, vmulq_f32(sum, inv_full_wsum));
        }

        for (; x < width; x++) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; k++)
                sum += data->src[(size_t)(y + k) * width + x] *
                    data->kernel[k + radius];
            dst_row[x] = sum / full_wsum;
        }
    }
}

static inline float cvvdp_gauss_pyr_reduce_scalar_neon(
    const CvvdpReduceTaskData* const data,
    const int dx,
    const int dy)
{
    float val = 0.0f;
    const int dx2 = dx << 1;
    const int dy2 = dy << 1;
    for (int ky = -2; ky <= 2; ky++) {
        int sy = dy2 + ky;
        if (sy < 0) sy = -sy - 1;
        if (sy >= data->src_h) sy = 2 * data->src_h - sy - 2;
        const float* const src_row = data->src + (size_t)sy * data->src_w;
        for (int kx = -2; kx <= 2; kx++) {
            int sx = dx2 + kx;
            if (sx < 0) sx = -sx - 1;
            if (sx >= data->src_w) sx = 2 * data->src_w - sx - 2;
            val += GAUSS_PYR_KERNEL[kx + 2] * GAUSS_PYR_KERNEL[ky + 2] *
                src_row[sx];
        }
    }
    return val;
}

static inline void cvvdp_gauss_pyr_reduce_impl(
    const CvvdpReduceTaskData* const data,
    const int start,
    const int end)
{
    const int dst_w = (data->src_w + 1) >> 1;
    const int dst_h = (data->src_h + 1) >> 1;
    const int row_end = imin(end, dst_h);
    const int interior_y_end = imin(dst_h, ((data->src_h - 3) / 2) + 1);
    const int interior_x_end = imin(dst_w, ((data->src_w - 3) / 2) + 1);
    const int vector_x_end = imin(interior_x_end,
                                  ((data->src_w - 10) / 2) + 1);

    for (int dy = start; dy < row_end; dy++) {
        float* const dst_row = data->dst + (size_t)dy * dst_w;
        int dx = 0;

        if (dy < 1 || dy >= interior_y_end) {
            for (; dx < dst_w; dx++)
                dst_row[dx] = cvvdp_gauss_pyr_reduce_scalar_neon(data, dx, dy);
            continue;
        }

        for (; dx < dst_w && dx < 1; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_reduce_scalar_neon(data, dx, dy);

        for (; dx + 4 <= vector_x_end; dx += 4) {
            const int sx2 = dx << 1;
            const int sy2 = dy << 1;
            float32x4_t val = vdupq_n_f32(0.0f);

            for (int ky = -2; ky <= 2; ky++) {
                const float wy = GAUSS_PYR_KERNEL[ky + 2];
                const float* const src_row =
                    data->src + (size_t)(sy2 + ky) * data->src_w;
                for (int kx = -2; kx <= 2; kx++) {
                    const float32x4_t src =
                        cvvdp_neon_load_even4(src_row + sx2 + kx);
                    const float weight = wy * GAUSS_PYR_KERNEL[kx + 2];
                    val = cvvdp_neon_madd_n(val, src, weight);
                }
            }

            vst1q_f32(dst_row + dx, val);
        }

        for (; dx < dst_w; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_reduce_scalar_neon(data, dx, dy);
    }
}

static inline float cvvdp_gauss_pyr_expand_scalar_neon(
    const CvvdpExpandTaskData* const data,
    const int dx,
    const int dy)
{
    const int src_w = (data->dst_w + 1) >> 1;
    const int src_h = (data->dst_h + 1) >> 1;
    const int parity_y = dy & 1;
    const int parity_x = dx & 1;
    float val = 0.0f;

    for (int ky = -2 + parity_y; ky <= 2; ky += 2) {
        int sy = (dy + ky) >> 1;
        if (sy < 0) sy = -sy - 1;
        if (sy >= src_h) sy = 2 * src_h - sy - 2;
        const float* const src_row = data->src + (size_t)sy * src_w;
        for (int kx = -2 + parity_x; kx <= 2; kx += 2) {
            int sx = (dx + kx) >> 1;
            if (sx < 0) sx = -sx - 1;
            if (sx >= src_w) sx = 2 * src_w - sx - 2;
            val += 4.0f * GAUSS_PYR_KERNEL[kx + 2] *
                GAUSS_PYR_KERNEL[ky + 2] * src_row[sx];
        }
    }

    return val;
}

static inline bool cvvdp_gauss_pyr_expand_row_interior_neon(
    const int dy,
    const int src_h)
{
    const int parity_y = dy & 1;
    for (int ky = -2 + parity_y; ky <= 2; ky += 2) {
        const int sy = (dy + ky) >> 1;
        if (sy < 0 || sy >= src_h) return false;
    }
    return true;
}

static inline void cvvdp_gauss_pyr_expand_impl(
    const CvvdpExpandTaskData* const data,
    const int start,
    const int end)
{
    const int dst_w = data->dst_w;
    const int dst_h = data->dst_h;
    const int src_w = (dst_w + 1) >> 1;
    const int src_h = (dst_h + 1) >> 1;
    const int row_end = imin(end, dst_h);

    for (int dy = start; dy < row_end; dy++) {
        float* const dst_row = data->dst + (size_t)dy * dst_w;
        int dx = 0;

        if (!cvvdp_gauss_pyr_expand_row_interior_neon(dy, src_h)) {
            for (; dx < dst_w; dx++)
                dst_row[dx] = cvvdp_gauss_pyr_expand_scalar_neon(data, dx, dy);
            continue;
        }

        for (; dx < dst_w && dx < 2; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_expand_scalar_neon(data, dx, dy);

        int n = 1;
        const int n_end_for_src = src_w - 4;
        const int n_end_for_dst = ((dst_w - 8) / 2) + 1;
        const int n_end = imin(n_end_for_src, n_end_for_dst);

        for (; n + 4 <= n_end; n += 4) {
            const int parity_y = dy & 1;
            float32x4_t even = vdupq_n_f32(0.0f);
            float32x4_t odd = vdupq_n_f32(0.0f);

            for (int ky = -2 + parity_y; ky <= 2; ky += 2) {
                const int sy = (dy + ky) >> 1;
                const float wy = 4.0f * GAUSS_PYR_KERNEL[ky + 2];
                const float* const src_row = data->src + (size_t)sy * src_w;
                float32x4_t h_even = vdupq_n_f32(0.0f);
                float32x4_t h_odd = vdupq_n_f32(0.0f);

                h_even = cvvdp_neon_madd_n(h_even, vld1q_f32(src_row + n - 1),
                                           GAUSS_PYR_KERNEL[0]);
                h_even = cvvdp_neon_madd_n(h_even, vld1q_f32(src_row + n),
                                           GAUSS_PYR_KERNEL[2]);
                h_even = cvvdp_neon_madd_n(h_even, vld1q_f32(src_row + n + 1),
                                           GAUSS_PYR_KERNEL[4]);
                h_odd = cvvdp_neon_madd_n(h_odd, vld1q_f32(src_row + n),
                                          GAUSS_PYR_KERNEL[1]);
                h_odd = cvvdp_neon_madd_n(h_odd, vld1q_f32(src_row + n + 1),
                                          GAUSS_PYR_KERNEL[3]);

                even = cvvdp_neon_madd_n(even, h_even, wy);
                odd = cvvdp_neon_madd_n(odd, h_odd, wy);
            }

            const float32x4x2_t zipped = vzipq_f32(even, odd);
            vst1q_f32(dst_row + (n << 1), zipped.val[0]);
            vst1q_f32(dst_row + (n << 1) + 4, zipped.val[1]);
        }

        dx = n << 1;
        for (; dx < dst_w; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_expand_scalar_neon(data, dx, dy);
    }
}

#endif /* FCVVDP_NEON_H */
