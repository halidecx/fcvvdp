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
#ifndef FCVVDP_AVX2_H
#define FCVVDP_AVX2_H

#include <immintrin.h>

#include "internal.h"
#include "util.h"

static inline __m256 cvvdp_avx2_madd_n(const __m256 acc,
                                       const __m256 v,
                                       const float scale)
{
    return _mm256_add_ps(acc, _mm256_mul_ps(v, _mm256_set1_ps(scale)));
}

static inline __m256 cvvdp_avx2_abs_ps(const __m256 v)
{
    const __m256 sign = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign, v);
}

static inline __m256 cvvdp_avx2_pack_even8(const __m256 lo,
                                           const __m256 hi)
{
    const __m256 shuffled = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
    return _mm256_permutevar8x32_ps(
        shuffled, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
}

static inline __m256 cvvdp_avx2_gauss5_even8(const float* const p) {
    const __m256 w0 = _mm256_set1_ps(GAUSS_PYR_KERNEL[0]);
    const __m256 w1 = _mm256_set1_ps(GAUSS_PYR_KERNEL[1]);
    const __m256 w2 = _mm256_set1_ps(GAUSS_PYR_KERNEL[2]);

    const __m256 lo_outer = _mm256_add_ps(_mm256_loadu_ps(p),
                                           _mm256_loadu_ps(p + 4));
    const __m256 lo_inner = _mm256_add_ps(_mm256_loadu_ps(p + 1),
                                           _mm256_loadu_ps(p + 3));
    __m256 lo = _mm256_mul_ps(_mm256_loadu_ps(p + 2), w2);
    lo = _mm256_fmadd_ps(lo_outer, w0, lo);
    lo = _mm256_fmadd_ps(lo_inner, w1, lo);

    const __m256 hi_outer = _mm256_add_ps(_mm256_loadu_ps(p + 8),
                                           _mm256_loadu_ps(p + 12));
    const __m256 hi_inner = _mm256_add_ps(_mm256_loadu_ps(p + 9),
                                           _mm256_loadu_ps(p + 11));
    __m256 hi = _mm256_mul_ps(_mm256_loadu_ps(p + 10), w2);
    hi = _mm256_fmadd_ps(hi_outer, w0, hi);
    hi = _mm256_fmadd_ps(hi_inner, w1, hi);

    return cvvdp_avx2_pack_even8(lo, hi);
}

static inline void cvvdp_avx2_store_interleaved8(float* const p,
                                                 const __m256 even,
                                                 const __m256 odd)
{
    const __m256 lo = _mm256_unpacklo_ps(even, odd);
    const __m256 hi = _mm256_unpackhi_ps(even, odd);
    _mm256_storeu_ps(p, _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(p + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

static inline float cvvdp_avx2_kernel_sum(const float* const kernel,
                                          const int radius)
{
    float sum = 0.0f;
    for (int k = -radius; k <= radius; k++)
        sum += kernel[k + radius];
    return sum;
}

static inline bool cvvdp_avx2_is_symmetric8(const float* const kernel,
                                            const int radius)
{
    if (radius != 8) return false;
    for (int k = 0; k < 8; k++)
        if (kernel[k] != kernel[16 - k]) return false;
    return true;
}

static inline __m256 cvvdp_avx2_blur_h_symmetric8(
    const float* const p,
    const float* const kernel)
{
    __m256 sum0 = _mm256_mul_ps(_mm256_loadu_ps(p),
                                 _mm256_set1_ps(kernel[8]));
    __m256 sum1 = _mm256_mul_ps(
        _mm256_add_ps(_mm256_loadu_ps(p - 2), _mm256_loadu_ps(p + 2)),
        _mm256_set1_ps(kernel[10]));
    __m256 sum2 = _mm256_mul_ps(
        _mm256_add_ps(_mm256_loadu_ps(p - 3), _mm256_loadu_ps(p + 3)),
        _mm256_set1_ps(kernel[11]));

    sum0 = cvvdp_avx2_madd_n(
        sum0,
        _mm256_add_ps(_mm256_loadu_ps(p - 1), _mm256_loadu_ps(p + 1)),
        kernel[9]);
    sum0 = cvvdp_avx2_madd_n(
        sum0,
        _mm256_add_ps(_mm256_loadu_ps(p - 4), _mm256_loadu_ps(p + 4)),
        kernel[12]);
    sum0 = cvvdp_avx2_madd_n(
        sum0,
        _mm256_add_ps(_mm256_loadu_ps(p - 7), _mm256_loadu_ps(p + 7)),
        kernel[15]);
    sum1 = cvvdp_avx2_madd_n(
        sum1,
        _mm256_add_ps(_mm256_loadu_ps(p - 5), _mm256_loadu_ps(p + 5)),
        kernel[13]);
    sum1 = cvvdp_avx2_madd_n(
        sum1,
        _mm256_add_ps(_mm256_loadu_ps(p - 8), _mm256_loadu_ps(p + 8)),
        kernel[16]);
    sum2 = cvvdp_avx2_madd_n(
        sum2,
        _mm256_add_ps(_mm256_loadu_ps(p - 6), _mm256_loadu_ps(p + 6)),
        kernel[14]);

    return _mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2);
}

static inline __m256 cvvdp_avx2_blur_v_symmetric8(
    const float* const p,
    const size_t stride,
    const float* const kernel)
{
    __m256 sum0 = _mm256_mul_ps(_mm256_loadu_ps(p),
                                 _mm256_set1_ps(kernel[8]));
    __m256 sum1 = _mm256_mul_ps(
        _mm256_add_ps(_mm256_loadu_ps(p - 2 * stride),
                      _mm256_loadu_ps(p + 2 * stride)),
        _mm256_set1_ps(kernel[10]));
    __m256 sum2 = _mm256_mul_ps(
        _mm256_add_ps(_mm256_loadu_ps(p - 3 * stride),
                      _mm256_loadu_ps(p + 3 * stride)),
        _mm256_set1_ps(kernel[11]));

    sum0 = cvvdp_avx2_madd_n(
        sum0,
        _mm256_add_ps(_mm256_loadu_ps(p - stride),
                      _mm256_loadu_ps(p + stride)), kernel[9]);
    sum0 = cvvdp_avx2_madd_n(
        sum0,
        _mm256_add_ps(_mm256_loadu_ps(p - 4 * stride),
                      _mm256_loadu_ps(p + 4 * stride)), kernel[12]);
    sum0 = cvvdp_avx2_madd_n(
        sum0,
        _mm256_add_ps(_mm256_loadu_ps(p - 7 * stride),
                      _mm256_loadu_ps(p + 7 * stride)), kernel[15]);
    sum1 = cvvdp_avx2_madd_n(
        sum1,
        _mm256_add_ps(_mm256_loadu_ps(p - 5 * stride),
                      _mm256_loadu_ps(p + 5 * stride)), kernel[13]);
    sum1 = cvvdp_avx2_madd_n(
        sum1,
        _mm256_add_ps(_mm256_loadu_ps(p - 8 * stride),
                      _mm256_loadu_ps(p + 8 * stride)), kernel[16]);
    sum2 = cvvdp_avx2_madd_n(
        sum2,
        _mm256_add_ps(_mm256_loadu_ps(p - 6 * stride),
                      _mm256_loadu_ps(p + 6 * stride)), kernel[14]);

    return _mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2);
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
        const __m256 vscale = _mm256_set1_ps(scale);
        const __m256 vmin = _mm256_set1_ps(clamp_min);
        const __m256 vmax = _mm256_set1_ps(Y_peak);
        const __m256 voffset = _mm256_set1_ps(offset);
        for (; i + 8 <= end; i += 8) {
            __m256 val = _mm256_mul_ps(_mm256_loadu_ps(data->plane + i), vscale);
            val = _mm256_max_ps(vmin, _mm256_min_ps(vmax, val));
            _mm256_storeu_ps(data->plane + i, _mm256_add_ps(val, voffset));
        }
    } else {
        const float scale = Y_peak - Y_black;
        const __m256 vexposure = _mm256_set1_ps(exposure);
        const __m256 vzero = _mm256_set1_ps(0.0f);
        const __m256 vone = _mm256_set1_ps(1.0f);
        const __m256 vscale = _mm256_set1_ps(scale);
        const __m256 voffset = _mm256_set1_ps(offset);
        for (; i + 8 <= end; i += 8) {
            __m256 val = _mm256_mul_ps(_mm256_loadu_ps(data->plane + i), vexposure);
            val = _mm256_max_ps(vzero, _mm256_min_ps(vone, val));
            _mm256_storeu_ps(data->plane + i,
                      _mm256_add_ps(_mm256_mul_ps(val, vscale), voffset));
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
    for (; i + 8 <= end; i += 8) {
        const __m256 r = _mm256_loadu_ps(data->x + i);
        const __m256 g = _mm256_loadu_ps(data->y + i);
        const __m256 b = _mm256_loadu_ps(data->z + i);

        __m256 x = _mm256_mul_ps(r, _mm256_set1_ps(0.4124564f));
        x = cvvdp_avx2_madd_n(x, g, 0.3575761f);
        x = cvvdp_avx2_madd_n(x, b, 0.1804375f);

        __m256 y = _mm256_mul_ps(r, _mm256_set1_ps(0.2126729f));
        y = cvvdp_avx2_madd_n(y, g, 0.7151522f);
        y = cvvdp_avx2_madd_n(y, b, 0.0721750f);

        __m256 z = _mm256_mul_ps(r, _mm256_set1_ps(0.0193339f));
        z = cvvdp_avx2_madd_n(z, g, 0.1191920f);
        z = cvvdp_avx2_madd_n(z, b, 0.9503041f);

        _mm256_storeu_ps(data->x + i, x);
        _mm256_storeu_ps(data->y + i, y);
        _mm256_storeu_ps(data->z + i, z);
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
    for (; i + 8 <= end; i += 8) {
        const __m256 x = _mm256_loadu_ps(data->x + i);
        const __m256 y = _mm256_loadu_ps(data->y + i);
        const __m256 z = _mm256_loadu_ps(data->z + i);

        __m256 L = _mm256_mul_ps(x, _mm256_set1_ps(0.187596268556126f));
        L = cvvdp_avx2_madd_n(L, y, 0.585168649077728f);
        L = cvvdp_avx2_madd_n(L, z, -0.026384263306304f);

        __m256 M = _mm256_mul_ps(x, _mm256_set1_ps(-0.133397430663221f));
        M = cvvdp_avx2_madd_n(M, y, 0.405505777260049f);
        M = cvvdp_avx2_madd_n(M, z, 0.034502127690364f);

        __m256 S = _mm256_mul_ps(x, _mm256_set1_ps(0.000244379021663f));
        S = cvvdp_avx2_madd_n(S, y, -0.000542995890619f);
        S = cvvdp_avx2_madd_n(S, z, 0.019406849066323f);

        const __m256 lum = _mm256_add_ps(L, M);
        const __m256 rg =
            _mm256_sub_ps(lum, _mm256_mul_ps(M, _mm256_set1_ps(3.311130179947035f)));
        const __m256 yv =
            _mm256_sub_ps(_mm256_mul_ps(S, _mm256_set1_ps(50.977571328718781f)), lum);

        _mm256_storeu_ps(data->x + i, lum);
        _mm256_storeu_ps(data->y + i, rg);
        _mm256_storeu_ps(data->z + i, yv);
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
    const __m256 vfloor = _mm256_set1_ps(0.01f);
    const __m256 vscale = _mm256_set1_ps(data->contrast_scale);
    int i = start;
    for (; i + 8 <= end; i += 8) {
        const __m256 src = _mm256_loadu_ps(data->src + i);
        const __m256 expanded = _mm256_loadu_ps(data->expanded + i);
        const __m256 L_bkg =
            _mm256_max_ps(vfloor, _mm256_loadu_ps(data->L_bkg + i));
        const __m256 contrast =
            _mm256_div_ps(_mm256_sub_ps(src, expanded), L_bkg);
        _mm256_storeu_ps(data->dst + i, _mm256_mul_ps(contrast, vscale));
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
    const __m256 vfloor = _mm256_set1_ps(0.01f);
    const __m256 vscale = _mm256_set1_ps(data->contrast_scale);
    int i = start;
    for (; i + 8 <= end; i += 8) {
        const __m256 src = _mm256_loadu_ps(data->src + i);
        const __m256 expanded = _mm256_loadu_ps(data->expanded + i);
        const __m256 L_bkg = _mm256_max_ps(vfloor, expanded);
        const __m256 contrast =
            _mm256_div_ps(_mm256_sub_ps(src, expanded), L_bkg);
        _mm256_storeu_ps(data->L_bkg + i, L_bkg);
        _mm256_storeu_ps(data->dst + i, _mm256_mul_ps(contrast, vscale));
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
    const __m256 denom = _mm256_set1_ps(data->denom);
    int i = start;
    for (; i + 8 <= end; i += 8) {
        _mm256_storeu_ps(data->dst + i,
                  _mm256_div_ps(_mm256_loadu_ps(data->src + i), denom));
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
    for (; i + 8 <= end; i += 8) {
        const __m256 ref = cvvdp_avx2_abs_ps(_mm256_loadu_ps(data->ref + i));
        const __m256 dst = cvvdp_avx2_abs_ps(_mm256_loadu_ps(data->dst + i));
        _mm256_storeu_ps(data->out + i, _mm256_min_ps(ref, dst));
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
        const __m256 vscale = _mm256_set1_ps(scale);
        int i = idx - ch * lev_size;

        for (; idx + 8 <= ch_end; idx += 8, i += 8) {
            const __m256 ref = _mm256_loadu_ps(data->ref_level[ch] + i);
            const __m256 dst = _mm256_loadu_ps(data->dst_level[ch] + i);
            const __m256 diff = cvvdp_avx2_abs_ps(_mm256_sub_ps(ref, dst));
            _mm256_storeu_ps(data->d + idx, _mm256_mul_ps(diff, vscale));
        }

        for (; idx < ch_end; idx++, i++) {
            const float diff =
                fabsf(data->ref_level[ch][i] - data->dst_level[ch][i]);
            data->d[idx] = diff * scale;
        }
    }
}

static inline const float* cvvdp_temporal_ring_frame_avx2(
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

    for (; i + 8 <= end; i += 8) {
        __m256 Y_sus = _mm256_set1_ps(0.0f);
        __m256 RG_sus = _mm256_set1_ps(0.0f);
        __m256 YV_sus = _mm256_set1_ps(0.0f);
        __m256 Y_trans = _mm256_set1_ps(0.0f);

        for (int k = 0; k < data->ring->filter.size; k++) {
            const float* const frame =
                cvvdp_temporal_ring_frame_avx2(data->ring, k);
            if (!frame) continue;

            const __m256 y = _mm256_loadu_ps(frame + i);
            const __m256 rg = _mm256_loadu_ps(frame + plane_size + i);
            const __m256 yv = _mm256_loadu_ps(frame + 2 * plane_size + i);

            Y_sus = cvvdp_avx2_madd_n(Y_sus, y,
                                      data->ring->filter.kernel[0][k]);
            Y_trans = cvvdp_avx2_madd_n(Y_trans, y,
                                        data->ring->filter.kernel[3][k]);
            RG_sus = cvvdp_avx2_madd_n(RG_sus, rg,
                                       data->ring->filter.kernel[1][k]);
            YV_sus = cvvdp_avx2_madd_n(YV_sus, yv,
                                       data->ring->filter.kernel[2][k]);
        }

        _mm256_storeu_ps(data->Y_sus + i, Y_sus);
        _mm256_storeu_ps(data->RG_sus + i, RG_sus);
        _mm256_storeu_ps(data->YV_sus + i, YV_sus);
        _mm256_storeu_ps(data->Y_trans + i, Y_trans);
    }

    for (; i < end; i++) {
        float Y_sus = 0.0f;
        float RG_sus = 0.0f;
        float YV_sus = 0.0f;
        float Y_trans = 0.0f;

        for (int k = 0; k < data->ring->filter.size; k++) {
            const float* const frame =
                cvvdp_temporal_ring_frame_avx2(data->ring, k);
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
    const float full_wsum = cvvdp_avx2_kernel_sum(data->kernel, radius);
    const __m256 inv_full_wsum = _mm256_set1_ps(1.0f / full_wsum);
    const bool symmetric8 = cvvdp_avx2_is_symmetric8(data->kernel, radius);

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
        for (; x + 8 <= interior_end; x += 8) {
            __m256 sum;
            if (symmetric8) {
                sum = cvvdp_avx2_blur_h_symmetric8(src_row + x, data->kernel);
            } else {
                sum = _mm256_setzero_ps();
                for (int k = -radius; k <= radius; k++) {
                    const __m256 src = _mm256_loadu_ps(src_row + x + k);
                    sum = cvvdp_avx2_madd_n(
                        sum, src, data->kernel[k + radius]);
                }
            }
            _mm256_storeu_ps(dst_row + x, _mm256_mul_ps(sum, inv_full_wsum));
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
    const float full_wsum = cvvdp_avx2_kernel_sum(data->kernel, radius);
    const __m256 inv_full_wsum = _mm256_set1_ps(1.0f / full_wsum);
    const bool symmetric8 = cvvdp_avx2_is_symmetric8(data->kernel, radius);

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
        for (; x + 8 <= width; x += 8) {
            __m256 sum;
            if (symmetric8) {
                sum = cvvdp_avx2_blur_v_symmetric8(
                    data->src + (size_t)y * width + x,
                    (size_t)width, data->kernel);
            } else {
                sum = _mm256_setzero_ps();
                for (int k = -radius; k <= radius; k++) {
                    const float* const src_row =
                        data->src + (size_t)(y + k) * width;
                    const __m256 src = _mm256_loadu_ps(src_row + x);
                    sum = cvvdp_avx2_madd_n(
                        sum, src, data->kernel[k + radius]);
                }
            }
            _mm256_storeu_ps(dst_row + x, _mm256_mul_ps(sum, inv_full_wsum));
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

static inline float cvvdp_gauss_pyr_reduce_scalar_avx2(
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
                dst_row[dx] = cvvdp_gauss_pyr_reduce_scalar_avx2(data, dx, dy);
            continue;
        }

        for (; dx < dst_w && dx < 1; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_reduce_scalar_avx2(data, dx, dy);

        for (; dx + 8 <= vector_x_end; dx += 8) {
            const int sx2 = dx << 1;
            const int sy2 = dy << 1;
            const float* const src0 =
                data->src + (size_t)(sy2 - 2) * data->src_w + sx2 - 2;
            const float* const src1 = src0 + data->src_w;
            const float* const src2 = src1 + data->src_w;
            const float* const src3 = src2 + data->src_w;
            const float* const src4 = src3 + data->src_w;
            const __m256 outer = _mm256_add_ps(
                cvvdp_avx2_gauss5_even8(src0),
                cvvdp_avx2_gauss5_even8(src4));
            const __m256 inner = _mm256_add_ps(
                cvvdp_avx2_gauss5_even8(src1),
                cvvdp_avx2_gauss5_even8(src3));
            __m256 val = _mm256_mul_ps(
                cvvdp_avx2_gauss5_even8(src2),
                _mm256_set1_ps(GAUSS_PYR_KERNEL[2]));
            val = _mm256_fmadd_ps(
                outer, _mm256_set1_ps(GAUSS_PYR_KERNEL[0]), val);
            val = _mm256_fmadd_ps(
                inner, _mm256_set1_ps(GAUSS_PYR_KERNEL[1]), val);

            _mm256_storeu_ps(dst_row + dx, val);
        }

        for (; dx < dst_w; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_reduce_scalar_avx2(data, dx, dy);
    }
}

static inline float cvvdp_gauss_pyr_expand_scalar_avx2(
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

static inline bool cvvdp_gauss_pyr_expand_row_interior_avx2(
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

        if (!cvvdp_gauss_pyr_expand_row_interior_avx2(dy, src_h)) {
            for (; dx < dst_w; dx++)
                dst_row[dx] = cvvdp_gauss_pyr_expand_scalar_avx2(data, dx, dy);
            continue;
        }

        for (; dx < dst_w && dx < 2; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_expand_scalar_avx2(data, dx, dy);

        int n = 1;
        const int n_end_for_src = src_w - 4;
        const int n_end_for_dst = ((dst_w - 8) / 2) + 1;
        const int n_end = imin(n_end_for_src, n_end_for_dst);

        for (; n + 8 <= n_end; n += 8) {
            const int parity_y = dy & 1;
            __m256 left;
            __m256 center;
            __m256 right;

            if (parity_y) {
                const float* const src0 =
                    data->src + (size_t)(dy >> 1) * src_w + n;
                const float* const src1 = src0 + src_w;
                left = _mm256_add_ps(_mm256_loadu_ps(src0 - 1),
                                     _mm256_loadu_ps(src1 - 1));
                center = _mm256_add_ps(_mm256_loadu_ps(src0),
                                       _mm256_loadu_ps(src1));
                right = _mm256_add_ps(_mm256_loadu_ps(src0 + 1),
                                      _mm256_loadu_ps(src1 + 1));
            } else {
                const float* const src0 =
                    data->src + (size_t)((dy >> 1) - 1) * src_w + n;
                const float* const src1 = src0 + src_w;
                const float* const src2 = src1 + src_w;
                const __m256 vouter = _mm256_set1_ps(4.0f * GAUSS_PYR_KERNEL[0]);
                const __m256 vcenter = _mm256_set1_ps(4.0f * GAUSS_PYR_KERNEL[2]);
                left = _mm256_mul_ps(
                    _mm256_add_ps(_mm256_loadu_ps(src0 - 1),
                                  _mm256_loadu_ps(src2 - 1)), vouter);
                left = _mm256_fmadd_ps(_mm256_loadu_ps(src1 - 1),
                                       vcenter, left);
                center = _mm256_mul_ps(
                    _mm256_add_ps(_mm256_loadu_ps(src0),
                                  _mm256_loadu_ps(src2)), vouter);
                center = _mm256_fmadd_ps(_mm256_loadu_ps(src1),
                                         vcenter, center);
                right = _mm256_mul_ps(
                    _mm256_add_ps(_mm256_loadu_ps(src0 + 1),
                                  _mm256_loadu_ps(src2 + 1)), vouter);
                right = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 1),
                                        vcenter, right);
            }

            const __m256 side = _mm256_add_ps(left, right);
            __m256 even = _mm256_mul_ps(
                center, _mm256_set1_ps(GAUSS_PYR_KERNEL[2]));
            even = _mm256_fmadd_ps(
                side, _mm256_set1_ps(GAUSS_PYR_KERNEL[0]), even);
            const __m256 odd = _mm256_mul_ps(
                _mm256_add_ps(center, right),
                _mm256_set1_ps(GAUSS_PYR_KERNEL[1]));

            cvvdp_avx2_store_interleaved8(dst_row + (n << 1), even, odd);
        }

        dx = n << 1;
        for (; dx < dst_w; dx++)
            dst_row[dx] = cvvdp_gauss_pyr_expand_scalar_avx2(data, dx, dy);
    }
}

#endif /* FCVVDP_AVX2_H */
