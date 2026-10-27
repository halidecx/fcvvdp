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

#define W 64
#define H 32
#define PLANE (W * H)
#define MAX_TAPS 17

static void check_compute_temporal_channels(
    const CvvdpSimdDispatch* const dsp)
{
    static CHECKASM_ALIGN(float frames[MAX_TAPS * 3 * PLANE]);
    CHECKASM_ALIGN(float kernel[4][MAX_TAPS]);
    CHECKASM_ALIGN(float out_c[4][PLANE]);
    CHECKASM_ALIGN(float out_a[4][PLANE]);

    checkasm_declare(void, const CvvdpTemporalChannelsTaskData*, int, int);

    // filter sizes for fps <= 0, 30 and 60
    static const int taps[] = { 1, 9, 17 };
    for (int t = 0; t < 3; t++) {
        if (checkasm_check_func(dsp->compute_temporal_channels,
                                "compute_temporal_channels_%d", taps[t]))
        {
            const int size = taps[t];
            TemporalRingBuf ring = {
                .width = W,
                .height = H,
                .data = frames,
                .max_frames = size,
                .num_frames = 1 + checkasm_rand() % size,
                .current_index = checkasm_rand() % size,
                .filter = {
                    .kernel = { kernel[0], kernel[1], kernel[2], kernel[3] },
                    .size = size,
                },
            };
            CvvdpTemporalChannelsTaskData data_c = {
                .ring = &ring,
                .Y_sus = out_c[0], .RG_sus = out_c[1],
                .YV_sus = out_c[2], .Y_trans = out_c[3],
                .plane_size = PLANE,
            };
            CvvdpTemporalChannelsTaskData data_a = {
                .ring = &ring,
                .Y_sus = out_a[0], .RG_sus = out_a[1],
                .YV_sus = out_a[2], .Y_trans = out_a[3],
                .plane_size = PLANE,
            };

            checkasm_randomize_intervalf(frames, size * 3 * PLANE,
                                         -100.0f, 1000.0f);
            checkasm_randomize_intervalf(&kernel[0][0], 4 * MAX_TAPS,
                                         -0.5f, 0.5f);
            CLEAR_BUF(out_c);
            CLEAR_BUF(out_a);

            int start, end;
            rand_range(PLANE, &start, &end);
            checkasm_call_ref(&data_c, start, end);
            checkasm_call_new(&data_a, start, end);
            check_float_near(&out_c[0][0], &out_a[0][0], 4 * PLANE, "out",
                                 1e-2f, 16);

            checkasm_bench_new(&data_a, 0, PLANE);
        }
    }

    checkasm_report("compute_temporal_channels");
}

void checkasm_check_temporal(void) {
    check_compute_temporal_channels(cvvdp_resolve_dispatch());
}
