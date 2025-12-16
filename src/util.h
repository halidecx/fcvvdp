/*
 * Copyright © 2025, Halide Compression, LLC.
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
#ifndef FCVVDP_UTIL_H
#define FCVVDP_UTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

static inline int imin(const int a, const int b) { return a < b ? a : b; }

static inline int iclip(const int v, const int min, const int max) {
    return v < min ? min : v > max ? max : v;
}

static inline float fclip(const float v, const float min, const float max) {
    return fmin(fmax(v, min), max);
}

static inline float* cvvdp_alloc_float(const size_t count) {
    return (float*)calloc(count, sizeof(float));
}

#endif /* FCVVDP_UTIL_H */
