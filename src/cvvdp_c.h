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
#ifndef FCVVDP_C_H
#define FCVVDP_C_H

#include "internal.h"
#include "util.h"

static inline void cvvdp_gauss_pyr_reduce_impl(
    const CvvdpReduceTaskData* const data,
    const int start,
    const int end)
{
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
                    val += GAUSS_PYR_KERNEL[kx + 2] *
                        GAUSS_PYR_KERNEL[ky + 2] * src_row[sx];
                }
            }
            dst_row[dx] = val;
        }
    }
}

static inline void cvvdp_gauss_pyr_expand_impl(
    const CvvdpExpandTaskData* const data,
    const int start,
    const int end)
{
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

static inline void cvvdp_blur_horizontal_impl(
    const CvvdpBlurTaskData* const data,
    const int start,
    const int end)
{
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

static inline void cvvdp_blur_vertical_impl(
    const CvvdpBlurTaskData* const data,
    const int start,
    const int end)
{
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

#endif /* FCVVDP_C_H */
