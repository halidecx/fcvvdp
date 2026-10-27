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
#include <string.h>

#include "tests/checkasm/internal.h"

#define N 4096

// spans the 0.01 floor and the 1000 clamp
static void randomize_log(float* const buf, const int len) {
    for (int i = 0; i < len; i++)
        buf[i] = powf(10.0f, -3.0f + 5.0f * (float)checkasm_randf());
}

static void check_contrast(const CvvdpSimdDispatch* const dsp) {
    CHECKASM_ALIGN(float src[N]);
    CHECKASM_ALIGN(float expanded[N]);
    CHECKASM_ALIGN(float L_bkg_c[N]);
    CHECKASM_ALIGN(float L_bkg_a[N]);
    CHECKASM_ALIGN(float dst_c[N]);
    CHECKASM_ALIGN(float dst_a[N]);

    checkasm_declare(void, const CvvdpContrastTaskData*, int, int);

    static const char* const names[] = { "contrast", "luma_contrast" };
    void (*const funcs[])(const CvvdpContrastTaskData*, int, int) = {
        dsp->contrast, dsp->luma_contrast,
    };

    for (int f = 0; f < 2; f++) {
        if (checkasm_check_func(funcs[f], "%s", names[f])) {
            const float scale = checkasm_rand() & 1 ? 1.0f : 2.0f;
            const CvvdpContrastTaskData data_c = {
                .src = src, .expanded = expanded, .L_bkg = L_bkg_c,
                .dst = dst_c, .contrast_scale = scale,
            };
            const CvvdpContrastTaskData data_a = {
                .src = src, .expanded = expanded, .L_bkg = L_bkg_a,
                .dst = dst_a, .contrast_scale = scale,
            };

            randomize_log(src, N);
            randomize_log(expanded, N);
            randomize_log(L_bkg_c, N);
            memcpy(L_bkg_a, L_bkg_c, sizeof(L_bkg_c));
            CLEAR_BUF(dst_c);
            CLEAR_BUF(dst_a);

            int start, end;
            rand_range(N, &start, &end);
            checkasm_call_ref(&data_c, start, end);
            checkasm_call_new(&data_a, start, end);
            checkasm_check1d(float_ulp, L_bkg_c, L_bkg_a, N, "L_bkg", 0);
            check_float_near(dst_c, dst_a, N, "dst", 1e-5f, 4);

            checkasm_bench_new(&data_a, 0, N);
        }
    }

    checkasm_report("contrast");
}

static void check_normalize(const CvvdpSimdDispatch* const dsp) {
    CHECKASM_ALIGN(float src[N]);
    CHECKASM_ALIGN(float dst_c[N]);
    CHECKASM_ALIGN(float dst_a[N]);

    checkasm_declare(void, const CvvdpNormalizeTaskData*, int, int);

    if (checkasm_check_func(dsp->normalize, "normalize")) {
        // small enough for the 1000 clamp to hit part of src
        const float denom = 0.01f + (float)checkasm_randf() * 0.09f;
        const CvvdpNormalizeTaskData data_c = {
            .src = src, .dst = dst_c, .denom = denom,
        };
        const CvvdpNormalizeTaskData data_a = {
            .src = src, .dst = dst_a, .denom = denom,
        };

        checkasm_randomize_intervalf(src, N, -100.0f, 100.0f);
        CLEAR_BUF(dst_c);
        CLEAR_BUF(dst_a);

        int start, end;
        rand_range(N, &start, &end);
        checkasm_call_ref(&data_c, start, end);
        checkasm_call_new(&data_a, start, end);
        check_float_near(dst_c, dst_a, N, "dst", 1e-6f, 4);

        checkasm_bench_new(&data_a, 0, N);
    }

    checkasm_report("normalize");
}

void checkasm_check_contrast(void) {
    const CvvdpSimdDispatch* const dsp = cvvdp_resolve_dispatch();

    check_contrast(dsp);
    check_normalize(dsp);
}
