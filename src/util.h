// © 2025 Halide Compression, LLC. All Rights Reserved.
#ifndef FCVVDP_UTIL_H
#define FCVVDP_UTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

static inline int imax(const int a, const int b) { return a > b ? a : b; }

static inline int imin(const int a, const int b) { return a < b ? a : b; }

static inline unsigned umax(const unsigned a, const unsigned b) {
    return a > b ? a : b;
}

static inline unsigned umin(const unsigned a, const unsigned b) {
    return a < b ? a : b;
}

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
