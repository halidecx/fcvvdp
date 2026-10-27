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
#include <checkasm/checkasm.h>

#include "src/cpu.h"
#include "tests/checkasm/internal.h"

static const CheckasmTest tests[] = {
    { "color", checkasm_check_color },
    { "temporal", checkasm_check_temporal },
    { "pyramid", checkasm_check_pyramid },
    { "contrast", checkasm_check_contrast },
    { "masking", checkasm_check_masking },
    { 0 }
};

static const CheckasmCpuInfo flags[] = {
#if defined(__x86_64__)
    { "AVX2", "avx2", CVVDP_CPU_FLAG_AVX2 },
#elif defined(__aarch64__)
    { "NEON", "neon", CVVDP_CPU_FLAG_NEON },
#endif
    { 0 }
};

static void set_cpu_flags(const CheckasmCpu flags) {
    cvvdp_set_cpu_flags_mask((unsigned)flags);
}

int main(const int argc, const char* argv[]) {
    CheckasmConfig cfg = {
        .cpu_flags     = flags,
        .tests         = tests,
        .set_cpu_flags = set_cpu_flags,
    };

    cvvdp_init_cpu();
    cfg.cpu = cvvdp_get_cpu_flags();

    return checkasm_main(&cfg, argc, argv);
}
