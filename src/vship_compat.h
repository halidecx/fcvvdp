/*
 * Copyright © 2026, Halide Compression, LLC.
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

/*
 * Vship-C-API compatibility layer for fcvvdp.
 *
 * The color types below are ABI-compatible with Vship 5.x VshipColor.h;
 * field-for-field identical (same order, same widths, same enum values),
 * renamed with an FcvvdpVship prefix.
 *
 * The compute entry point takes Vship-style planar frames: 3 byte pointers
 * with byte strides, sample values interpreted exactly like Vship's GPU
 * converter (u8 for UINT8; native-endian u16 containers masked to the
 * sample bit depth for UINT9..UINT16; native f32 for FLOAT).
 *
 * Primaries are not used: fcvvdp has no primaries input, so the primaries
 * field is accepted but dropped.
 */
#ifndef FCVVDP_VSHIP_COMPAT_H
#define FCVVDP_VSHIP_COMPAT_H

#include <stdint.h>

#include "cvvdp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FcvvdpVshipSample_t {
    FCVVDP_VSHIP_SAMPLE_FLOAT = 0,
    FCVVDP_VSHIP_SAMPLE_HALF = 1,
    FCVVDP_VSHIP_SAMPLE_UINT8 = 2,
    FCVVDP_VSHIP_SAMPLE_UINT9 = 3,
    FCVVDP_VSHIP_SAMPLE_UINT10 = 5,
    FCVVDP_VSHIP_SAMPLE_UINT12 = 7,
    FCVVDP_VSHIP_SAMPLE_UINT14 = 9,
    FCVVDP_VSHIP_SAMPLE_UINT16 = 11
} FcvvdpVshipSample_t;

typedef enum FcvvdpVshipRange_t {
    FCVVDP_VSHIP_RANGE_LIMITED = 0,
    FCVVDP_VSHIP_RANGE_FULL = 1
} FcvvdpVshipRange_t;

typedef struct FcvvdpVshipChromaSubsample_t {
    // log2 horizontal/vertical subsampling factors. The mirrored Vship ABI
    // permits 0..2, but this shim only accepts 0 (4:4:4) and 1 (4:2:2/4:2:0);
    // values >= 2 return CVVDP_ERROR_INVALID_FORMAT.
    int subw;
    int subh;
} FcvvdpVshipChromaSubsample_t;

typedef enum FcvvdpVshipChromaLocation_t {
    FCVVDP_VSHIP_CHROMA_LEFT = 0,
    FCVVDP_VSHIP_CHROMA_CENTER = 1,
    FCVVDP_VSHIP_CHROMA_TOPLEFT = 2,
    FCVVDP_VSHIP_CHROMA_TOP = 3
} FcvvdpVshipChromaLocation_t;

typedef enum FcvvdpVshipColorFamily_t {
    FCVVDP_VSHIP_COLOR_YUV = 0,
    FCVVDP_VSHIP_COLOR_RGB = 1
} FcvvdpVshipColorFamily_t;

typedef enum FcvvdpVshipYUVMatrix_t {
    FCVVDP_VSHIP_MATRIX_RGB = 0,
    FCVVDP_VSHIP_MATRIX_BT709 = 1,
    FCVVDP_VSHIP_MATRIX_BT470_BG = 5,
    FCVVDP_VSHIP_MATRIX_ST170_M = 6, //same as 5
    FCVVDP_VSHIP_MATRIX_YCGCO = 8,
    FCVVDP_VSHIP_MATRIX_BT2020_NCL = 9,
    FCVVDP_VSHIP_MATRIX_BT2020_CL = 10,
    FCVVDP_VSHIP_MATRIX_BT2100_ICTCP = 14,
    FCVVDP_VSHIP_MATRIX_YCGCO_RE = 16,
    FCVVDP_VSHIP_MATRIX_YCGCO_RO = 17
} FcvvdpVshipYUVMatrix_t;

typedef enum FcvvdpVshipTransferFunction_t {
    FCVVDP_VSHIP_TRC_BT709 = 1,
    FCVVDP_VSHIP_TRC_BT470_M = 4,
    FCVVDP_VSHIP_TRC_BT470_BG = 5,
    FCVVDP_VSHIP_TRC_BT601 = 6, //same as 5
    FCVVDP_VSHIP_TRC_ST240_M = 7,
    FCVVDP_VSHIP_TRC_LINEAR = 8,
    FCVVDP_VSHIP_TRC_SRGB = 13,
    FCVVDP_VSHIP_TRC_PQ = 16,
    FCVVDP_VSHIP_TRC_ST428 = 17,
    FCVVDP_VSHIP_TRC_HLG = 18
} FcvvdpVshipTransferFunction_t;

typedef enum FcvvdpVshipPrimaries_t {
    FCVVDP_VSHIP_PRIMARIES_INTERNAL = -1, //corresponds to XYZ really
    FCVVDP_VSHIP_PRIMARIES_BT709 = 1,
    FCVVDP_VSHIP_PRIMARIES_BT470_M = 4,
    FCVVDP_VSHIP_PRIMARIES_BT470_BG = 5,
    FCVVDP_VSHIP_PRIMARIES_ST170_M = 6,
    FCVVDP_VSHIP_PRIMARIES_ST240_M = 7, // Equivalent to 6.
    FCVVDP_VSHIP_PRIMARIES_BT2020 = 9,
    FCVVDP_VSHIP_PRIMARIES_DISPLAYP3 = 12
} FcvvdpVshipPrimaries_t;

typedef struct FcvvdpVshipCropRectangle_t {
    int top;
    int bottom;
    int left;
    int right;
} FcvvdpVshipCropRectangle_t;

//commentary defines the most classic YUV420P BT709
//target_width/height of -1 mean same as width/height; resize is applied
//before cropping, final size is target - crop on each axis.
typedef struct FcvvdpVshipColorspace_t {
    int64_t width;
    int64_t height;
    int64_t target_width;
    int64_t target_height;
    FcvvdpVshipSample_t sample;
    FcvvdpVshipRange_t range;
    FcvvdpVshipChromaSubsample_t subsampling;
    FcvvdpVshipChromaLocation_t chromaLocation;
    FcvvdpVshipColorFamily_t colorFamily; //now unused, matrix 0 is RGB and others are YUV
    FcvvdpVshipYUVMatrix_t YUVMatrix;
    FcvvdpVshipTransferFunction_t transferFunction;
    FcvvdpVshipPrimaries_t primaries; //dropped, fcvvdp has no primaries input
    FcvvdpVshipCropRectangle_t crop;
} FcvvdpVshipColorspace_t;

/* Opaque shim context */
typedef struct FcvvdpVshipCtx FcvvdpVshipCtx;

/**
 * Create a shim context wrapping a fcvvdp context.
 *
 * @param out_ctx         Output pointer to created context
 * @param src_colorspace  Colorspace of the reference frames
 * @param dis_colorspace  Colorspace of the distorted frames
 * @param fps             Frames per second (passed to cvvdp_create)
 * @param model_key       Display model key, one of "standard_4k",
 *                        "standard_fhd", "standard_hdr_pq",
 *                        "standard_hdr_hlg", "standard_hdr_dark",
 *                        "standard_hdr_linear", "standard_hdr_linear_zoom"
 * @param threads         Worker threads, 0 = auto
 *
 * @return CVVDP_OK on success, error code otherwise
 */
FcvvdpError fcvvdp_vship_create(FcvvdpVshipCtx** const out_ctx,
                                const FcvvdpVshipColorspace_t* const src_colorspace,
                                const FcvvdpVshipColorspace_t* const dis_colorspace,
                                const float fps,
                                const char* const model_key,
                                const unsigned threads);

/**
 * Compare one frame pair, Vship-style.
 *
 * Planes are byte pointers with byte strides, exactly like Vship's
 * ComputeHandler. Chroma plane dimensions are derived from the colorspace
 * subsampling (ceil of the luma dimensions). Both colorspaces are processed
 * independently, so src and dis may differ in format.
 *
 * @param ctx        Shim context
 * @param jod_out    Written with the running temporally-pooled JOD
 * @param srcp       Reference planes (plane 0 = Y or R)
 * @param disp       Distorted planes (plane 0 = Y or R)
 * @param src_stride Reference plane strides in bytes
 * @param dis_stride Distorted plane strides in bytes
 *
 * @return CVVDP_OK on success, error code otherwise
 */
FcvvdpError fcvvdp_vship_compute(FcvvdpVshipCtx* const ctx,
                                 double* const jod_out,
                                 const uint8_t* const srcp[3],
                                 const uint8_t* const disp[3],
                                 const int64_t src_stride[3],
                                 const int64_t dis_stride[3]);

/**
 * Destroy a shim context and free all resources
 *
 * @param ctx Context to destroy
 */
void fcvvdp_vship_destroy(FcvvdpVshipCtx* const ctx);

#ifdef __cplusplus
}
#endif

#endif /* FCVVDP_VSHIP_COMPAT_H */
