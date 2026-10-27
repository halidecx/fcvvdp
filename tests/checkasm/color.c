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
#include <string.h>

#include "tests/checkasm/internal.h"

#define N 4096

static void check_apply_display(const CvvdpSimdDispatch* const dsp) {
    CHECKASM_ALIGN(float src[N]);
    CHECKASM_ALIGN(float plane_c[N]);
    CHECKASM_ALIGN(float plane_a[N]);

    checkasm_declare(void, const CvvdpApplyDisplayTaskData*, int, int);

    for (int hdr = 0; hdr <= 1; hdr++) {
        if (checkasm_check_func(dsp->apply_display, "apply_display_%s",
                                hdr ? "hdr" : "sdr"))
        {
            Display display = {
                .max_luminance = hdr ? 1000.0f : 200.0f,
                .exposure = 1.0f + (float)checkasm_randf(),
                .refl_level = (float)checkasm_randf() * 0.05f,
            };
            CvvdpApplyDisplayTaskData data_c = {
                .plane = plane_c, .display = &display, .is_hdr = hdr,
            };
            CvvdpApplyDisplayTaskData data_a = {
                .plane = plane_a, .display = &display, .is_hdr = hdr,
            };

            // HDR presets have black levels on both sides of the 0.005 floor
            for (int i = 0; i < 2; i++) {
                display.black_level = hdr
                    ? (i ? 0.005f : 0.0005f) + (float)checkasm_randf() * 0.0045f
                    : 0.05f + (float)checkasm_randf() * 0.5f;

                // HDR inputs below zero exercise the black level floor
                checkasm_randomize_intervalf(src, N, hdr ? -12.0f : -0.1f,
                                             hdr ? 12.0f : 1.1f);
                int start, end;
                rand_range(N, &start, &end);
                // the last element is in the scalar tail
                src[end - 1] = -(float)checkasm_randf();
                memcpy(plane_c, src, sizeof(src));
                memcpy(plane_a, src, sizeof(src));
                checkasm_call_ref(&data_c, start, end);
                checkasm_call_new(&data_a, start, end);
                check_float_near(plane_c, plane_a, N, "plane", 1e-6f, 4);
            }

            checkasm_bench_new(&data_a, 0, N);
        }
    }

    checkasm_report("apply_display");
}

static void check_color_transform(const CvvdpSimdDispatch* const dsp) {
    CHECKASM_ALIGN(float src[3][N]);
    CHECKASM_ALIGN(float dst_c[3][N]);
    CHECKASM_ALIGN(float dst_a[3][N]);

    checkasm_declare(void, const CvvdpColorTransformTaskData*, int, int);

    static const char* const names[] = { "rgb_to_xyz", "xyz_to_dkl" };
    void (*const funcs[])(const CvvdpColorTransformTaskData*, int, int) = {
        dsp->rgb_to_xyz, dsp->xyz_to_dkl,
    };

    for (int f = 0; f < 2; f++) {
        if (checkasm_check_func(funcs[f], "%s", names[f])) {
            CvvdpColorTransformTaskData data_c = {
                .x = dst_c[0], .y = dst_c[1], .z = dst_c[2],
            };
            CvvdpColorTransformTaskData data_a = {
                .x = dst_a[0], .y = dst_a[1], .z = dst_a[2],
            };

            checkasm_randomize_intervalf(&src[0][0], 3 * N, 0.0f, 1000.0f);
            int start, end;
            rand_range(N, &start, &end);
            memcpy(dst_c, src, sizeof(src));
            memcpy(dst_a, src, sizeof(src));
            checkasm_call_ref(&data_c, start, end);
            checkasm_call_new(&data_a, start, end);
            check_float_near(&dst_c[0][0], &dst_a[0][0], 3 * N, "xyz",
                                 1e-3f, 4);

            checkasm_bench_new(&data_a, 0, N);
        }
    }

    checkasm_report("color_transform");
}

void checkasm_check_color(void) {
    const CvvdpSimdDispatch* const dsp = cvvdp_resolve_dispatch();

    check_apply_display(dsp);
    check_color_transform(dsp);
}
