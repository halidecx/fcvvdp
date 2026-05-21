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
