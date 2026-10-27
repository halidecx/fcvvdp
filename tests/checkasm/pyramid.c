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
#include "tests/checkasm/internal.h"

#define MAX_W 256
#define MAX_H 128
#define MAX_SIZE (MAX_W * MAX_H)

// smallest pyramid level is at least 4 pixels per side
static int rand_dim(void) {
    return 4 + checkasm_rand() % 77;
}

static void check_gauss_pyr_reduce(const CvvdpSimdDispatch* const dsp) {
    static CHECKASM_ALIGN(float src[MAX_SIZE]);
    static CHECKASM_ALIGN(float dst_c[MAX_SIZE]);
    static CHECKASM_ALIGN(float dst_a[MAX_SIZE]);

    checkasm_declare(void, const CvvdpReduceTaskData*, int, int);

    if (checkasm_check_func(dsp->gauss_pyr_reduce, "gauss_pyr_reduce")) {
        for (int i = 0; i < 16; i++) {
            const int src_w = rand_dim(), src_h = rand_dim();
            const int dst_h = (src_h + 1) >> 1;
            const CvvdpReduceTaskData data_c = {
                .src = src, .dst = dst_c, .src_w = src_w, .src_h = src_h,
            };
            const CvvdpReduceTaskData data_a = {
                .src = src, .dst = dst_a, .src_w = src_w, .src_h = src_h,
            };

            checkasm_randomize_intervalf(src, src_w * src_h, 0.0f, 1000.0f);
            CLEAR_BUF(dst_c);
            CLEAR_BUF(dst_a);

            const int start = checkasm_rand() % dst_h;
            const int end = start + 1 + checkasm_rand() % (dst_h - start);
            checkasm_call_ref(&data_c, start, end);
            checkasm_call_new(&data_a, start, end);
            check_float_near(dst_c, dst_a, MAX_SIZE, "dst", 1e-3f, 8);
        }

        const CvvdpReduceTaskData data = {
            .src = src, .dst = dst_a, .src_w = MAX_W, .src_h = MAX_H,
        };
        checkasm_randomize_intervalf(src, MAX_SIZE, 0.0f, 1000.0f);
        checkasm_bench_new(&data, 0, MAX_H / 2);
    }

    checkasm_report("gauss_pyr_reduce");
}

static void check_gauss_pyr_expand(const CvvdpSimdDispatch* const dsp) {
    static CHECKASM_ALIGN(float src[MAX_SIZE]);
    static CHECKASM_ALIGN(float dst_c[MAX_SIZE]);
    static CHECKASM_ALIGN(float dst_a[MAX_SIZE]);

    checkasm_declare(void, const CvvdpExpandTaskData*, int, int);

    if (checkasm_check_func(dsp->gauss_pyr_expand, "gauss_pyr_expand")) {
        for (int i = 0; i < 16; i++) {
            const int dst_w = rand_dim(), dst_h = rand_dim();
            const int src_w = (dst_w + 1) >> 1, src_h = (dst_h + 1) >> 1;
            const CvvdpExpandTaskData data_c = {
                .src = src, .dst = dst_c, .dst_w = dst_w, .dst_h = dst_h,
            };
            const CvvdpExpandTaskData data_a = {
                .src = src, .dst = dst_a, .dst_w = dst_w, .dst_h = dst_h,
            };

            checkasm_randomize_intervalf(src, src_w * src_h, 0.0f, 1000.0f);
            CLEAR_BUF(dst_c);
            CLEAR_BUF(dst_a);

            const int start = checkasm_rand() % dst_h;
            const int end = start + 1 + checkasm_rand() % (dst_h - start);
            checkasm_call_ref(&data_c, start, end);
            checkasm_call_new(&data_a, start, end);
            check_float_near(dst_c, dst_a, MAX_SIZE, "dst", 1e-3f, 8);
        }

        const CvvdpExpandTaskData data = {
            .src = src, .dst = dst_a, .dst_w = MAX_W, .dst_h = MAX_H,
        };
        checkasm_randomize_intervalf(src, MAX_SIZE / 4, 0.0f, 1000.0f);
        checkasm_bench_new(&data, 0, MAX_H);
    }

    checkasm_report("gauss_pyr_expand");
}

void checkasm_check_pyramid(void) {
    const CvvdpSimdDispatch* const dsp = cvvdp_resolve_dispatch();

    check_gauss_pyr_reduce(dsp);
    check_gauss_pyr_expand(dsp);
}
