/*
 * Copyright © 2026, Halide Compression, LLC.
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
#include "cpu.h"

unsigned cvvdp_cpu_flags = 0U;
unsigned cvvdp_cpu_flags_mask = ~0U;

#if defined(__x86_64__)
static int cvvdp_x86_has_avx2_fma(void) {
    unsigned int eax, ebx, ecx, edx, max_leaf;

    __asm__ volatile("cpuid"
        : "=a"(max_leaf), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0));
    if (max_leaf < 7) return 0;

    __asm__ volatile("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(1));
    if (!(ecx & (1u << 26))) return 0;
    if (!(ecx & (1u << 27))) return 0;
    if (!(ecx & (1u << 28))) return 0;
    if (!(ecx & (1u << 12))) return 0;

    unsigned int xcr0_lo, xcr0_hi;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    (void)xcr0_hi;
    if ((xcr0_lo & 0x6u) != 0x6u) return 0;

    __asm__ volatile("cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0));
    if (!(ebx & (1u << 5))) return 0;

    return 1;
}
#endif

void cvvdp_init_cpu(void) {
#if defined(__x86_64__)
    if (cvvdp_x86_has_avx2_fma())
        cvvdp_cpu_flags |= CVVDP_CPU_FLAG_AVX2;
#elif defined(__aarch64__)
    cvvdp_cpu_flags |= CVVDP_CPU_FLAG_NEON;
#endif
}

void cvvdp_set_cpu_flags_mask(const unsigned mask) {
    cvvdp_cpu_flags_mask = mask;
}
