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
#ifndef FCVVDP_TESTS_CHECKASM_INTERNAL_H
#define FCVVDP_TESTS_CHECKASM_INTERNAL_H

#include <stdio.h>

#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "src/internal.h"

void checkasm_check_color(void);
void checkasm_check_temporal(void);
void checkasm_check_pyramid(void);
void checkasm_check_contrast(void);
void checkasm_check_masking(void);

// SIMD paths use FMA and a different summation order, so outputs are not
// bitexact; eps covers results near zero where ULP distance is meaningless.
static inline void check_float_impl(const char* const file, const int line,
                                    const float* const ref,
                                    const float* const new, const int len,
                                    const char* const name, const float eps,
                                    const unsigned max_ulp)
{
    for (int i = 0; i < len; i++) {
        if (checkasm_float_near_abs_eps_ulp(ref[i], new[i], eps, max_ulp))
            continue;
        if (checkasm_fail_func("%s:%d %s", file, line, name))
            fprintf(stderr, "%s[%d]: ref %.9g new %.9g\n",
                    name, i, ref[i], new[i]);
        return;
    }
}

// unaligned start, and a length that is never a multiple of 4 so the 8- and
// 4-wide SIMD loops always leave a scalar tail
static inline void rand_range(const int len, int* const start, int* const end) {
    *start = checkasm_rand() % 64;
    *end = len - checkasm_rand() % 64;
    if (!((*end - *start) & 3))
        (*end)--;
}

#define check_float_near(...) check_float_impl(__FILE__, __LINE__, __VA_ARGS__)

#endif /* FCVVDP_TESTS_CHECKASM_INTERNAL_H */
