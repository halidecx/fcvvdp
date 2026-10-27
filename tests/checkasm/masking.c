/*
 * Copyright © 2026, Sungjoon Moon
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
#include <math.h>

#include "tests/checkasm/internal.h"

#define N 4096
#define MAX_W 128
#define MAX_H 64
#define MAX_SIZE (MAX_W * MAX_H)
#define RADIUS CVVDP_GAUSSIAN_SIZE
#define TAPS (2 * RADIUS + 1)

static void check_min_abs(const CvvdpSimdDispatch* const dsp) {
    CHECKASM_ALIGN(float ref[N]);
    CHECKASM_ALIGN(float dst[N]);
    CHECKASM_ALIGN(float out_c[N]);
    CHECKASM_ALIGN(float out_a[N]);

    checkasm_declare(void, const CvvdpMinAbsTaskData*, int, int);

    if (checkasm_check_func(dsp->min_abs, "min_abs")) {
        const CvvdpMinAbsTaskData data_c = {
            .ref = ref, .dst = dst, .out = out_c,
        };
        const CvvdpMinAbsTaskData data_a = {
            .ref = ref, .dst = dst, .out = out_a,
        };

        checkasm_randomize_normf(ref, N);
        checkasm_randomize_normf(dst, N);
        CLEAR_BUF(out_c);
        CLEAR_BUF(out_a);

        int start, end;
        rand_range(N, &start, &end);
        checkasm_call_ref(&data_c, start, end);
        checkasm_call_new(&data_a, start, end);
        checkasm_check1d(float_ulp, out_c, out_a, N, "out", 0);

        checkasm_bench_new(&data_a, 0, N);
    }

    checkasm_report("min_abs");
}

// the asymmetric kernel is left unnormalized so that the division by the
// kernel sum is tested
static void init_kernel(float* const kernel, const int symmetric) {
    float sum = 0.0f;
    for (int i = 0; i < TAPS; i++) {
        const int d = i - RADIUS;
        kernel[i] = symmetric ? expf(-(float)(d * d) / 8.0f)
                              : 0.01f + (float)checkasm_randf();
        sum += kernel[i];
    }
    if (symmetric)
        for (int i = 0; i < TAPS; i++)
            kernel[i] /= sum;
}

static void check_blur(const CvvdpSimdDispatch* const dsp) {
    static CHECKASM_ALIGN(float src[MAX_SIZE]);
    static CHECKASM_ALIGN(float dst_c[MAX_SIZE]);
    static CHECKASM_ALIGN(float dst_a[MAX_SIZE]);
    CHECKASM_ALIGN(float kernel[TAPS]);

    checkasm_declare(void, const CvvdpBlurTaskData*, int, int);

    static const char* const names[] = { "blur_horizontal", "blur_vertical" };
    void (*const funcs[])(const CvvdpBlurTaskData*, int, int) = {
        dsp->blur_horizontal, dsp->blur_vertical,
    };

    for (int f = 0; f < 2; f++) {
        for (int sym = 0; sym <= 1; sym++) {
            if (!checkasm_check_func(funcs[f], "%s_%s", names[f],
                                     sym ? "sym" : "asym"))
                continue;

            init_kernel(kernel, sym);
            for (int i = 0; i < 16; i++) {
                const int w = 4 + checkasm_rand() % (MAX_W - 3);
                const int h = 4 + checkasm_rand() % (MAX_H - 3);
                const CvvdpBlurTaskData data_c = {
                    .src = src, .dst = dst_c, .width = w, .height = h,
                    .kernel = kernel, .radius = RADIUS,
                };
                const CvvdpBlurTaskData data_a = {
                    .src = src, .dst = dst_a, .width = w, .height = h,
                    .kernel = kernel, .radius = RADIUS,
                };

                checkasm_randomize_intervalf(src, w * h, 0.0f, 10.0f);
                CLEAR_BUF(dst_c);
                CLEAR_BUF(dst_a);

                const int start = checkasm_rand() % h;
                const int end = start + 1 + checkasm_rand() % (h - start);
                checkasm_call_ref(&data_c, start, end);
                checkasm_call_new(&data_a, start, end);
                check_float_near(dst_c, dst_a, MAX_SIZE, "dst", 1e-5f, 8);
            }

            const CvvdpBlurTaskData data = {
                .src = src, .dst = dst_a, .width = MAX_W, .height = MAX_H,
                .kernel = kernel, .radius = RADIUS,
            };
            checkasm_randomize_intervalf(src, MAX_SIZE, 0.0f, 10.0f);
            checkasm_bench_new(&data, 0, MAX_H);
        }
    }

    checkasm_report("blur");
}

static void check_baseband_diff(const CvvdpSimdDispatch* const dsp) {
    CHECKASM_ALIGN(float ref[4][N]);
    CHECKASM_ALIGN(float dst[4][N]);
    CHECKASM_ALIGN(float d_c[4 * N]);
    CHECKASM_ALIGN(float d_a[4 * N]);

    checkasm_declare(void, const CvvdpBasebandDiffTaskData*, int, int);

    if (checkasm_check_func(dsp->baseband_diff, "baseband_diff")) {
        const float* const ref_level[4] = { ref[0], ref[1], ref[2], ref[3] };
        const float* const dst_level[4] = { dst[0], dst[1], dst[2], dst[3] };
        const size_t lev_size = N - checkasm_rand() % 64;
        CvvdpBasebandDiffTaskData data_c = {
            .ref_level = ref_level, .dst_level = dst_level,
            .d = d_c, .lev_size = lev_size,
        };
        for (int ch = 0; ch < 4; ch++)
            data_c.sensitivity[ch] = 1.0f + (float)checkasm_randf() * 100.0f;
        CvvdpBasebandDiffTaskData data_a = data_c;
        data_a.d = d_a;

        checkasm_randomize_normf(&ref[0][0], 4 * N);
        checkasm_randomize_normf(&dst[0][0], 4 * N);
        CLEAR_BUF(d_c);
        CLEAR_BUF(d_a);

        const int total = 4 * (int)lev_size;
        int start, end;
        rand_range(total, &start, &end);
        checkasm_call_ref(&data_c, start, end);
        checkasm_call_new(&data_a, start, end);
        check_float_near(d_c, d_a, 4 * N, "d", 1e-6f, 4);

        checkasm_bench_new(&data_a, 0, total);
    }

    checkasm_report("baseband_diff");
}

void checkasm_check_masking(void) {
    const CvvdpSimdDispatch* const dsp = cvvdp_resolve_dispatch();

    check_min_abs(dsp);
    check_blur(dsp);
    check_baseband_diff(dsp);
}
