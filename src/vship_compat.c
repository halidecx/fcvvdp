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
 * Vship-C-API compatibility layer: converts Vship-style planar frames to
 * fcvvdp's interleaved linear RGB (float) and drives an inner FcvvdpCtx.
 *
 * The pipeline mirrors Vship's GPU CVVDP converter (see
 * Vship/src/HIP/gpuColorToLinear/vshipColor.hpp Converter::convert):
 * range expansion -> chroma upsampling -> YUV matrix -> transfer
 * linearization -> resize/crop. Sample semantics mirror
 * anyDepthToFloat.hpp + rangeToFull.hpp: 2-byte samples are native-endian
 * u16 containers masked to the sample bit depth; full range divides by
 * (2^bits - 1) and centers chroma at -0.5; limited range normalizes to
 * 8-bit units then expands with the studio swing ((v-16)/219 luma,
 * (v-128)/224 chroma); FLOAT limited scales by 256 first, FLOAT full is
 * [0,1] with chroma centered at -0.5. Chroma upsampling is the bicubic
 * Hermite spline of chromaUpsample.hpp with Vship's chroma siting.
 *
 * The output is linear RGB consumed as-is by cvvdp_load_image's
 * RGB_FLOAT path. Per-TRC output scale, matching the linear RGB that
 * Vship's GPU CVVDP core consumes (YUVToLinRGBPipeline feeding
 * linRGB_to_dkl_DisplayEncode, cvvdp/colors.hpp):
 *  - SDR TRCs (linear, sRGB, BT709 pure gamma 2.4, gamma 2.2, gamma 2.8,
 *    ST240, ST428): linear [0,1] (ST428 reaches ~1.091)
 *  - PQ: linear [0,100] (Lmax = 100)
 *  - HLG: inverse OETF + OOTF gamma 1.2, linear [0,1]
 * fcvvdp's display model (cvvdp_apply_display_impl in cvvdp_c.h)
 * multiplies input by 100 on HDR display models and clamps to Y_peak;
 * Vship instead routes every non-PQ transfer through the SDR display
 * encode formula, scaling by (Y_peak - Y_black). On HDR display models,
 * non-PQ values are therefore pre-scaled by (Y_peak - Y_black) / 100 so
 * both chains land on identical display-encoded nits; on SDR display
 * models both apply the same clamp-and-scale, so values pass through
 * unscaled. PQ passes through unscaled either way ([0,100] hits fcvvdp's
 * HDR x100 exactly like Vship's PQ display encode).
 */
#include "vship_compat.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

typedef struct {
    int64_t width, height; // input dimensions
    int64_t target_width, target_height;
    int final_width, final_height;
    int crop_left, crop_top;
    int subw, subh;
    int chroma_location;
    int range;
    int matrix;
    int trc;
    int is_yuv; // derived from matrix, like Vship's Converter::init
    int bits; // 8..16, 1 for float (Vship's bitprecisionSample)
    int bytes_per_sample; // 1, 2 or 4
    int is_float;
} SideCfg;

typedef struct {
    SideCfg cfg;
    float* conv; // width*height*3 interleaved linear RGB, per-TRC scale
    float* out; // final_width*final_height*3 interleaved linear RGB
    float out_scale;
    FcvvdpImage img;
} Side;

struct FcvvdpVshipCtx {
    FcvvdpCtx* inner;
    Side src;
    Side dis;
};

static FcvvdpError model_from_key(const char* const key,
                                  FcvvdpDisplayModel* const out) {
    static const struct {
        const char* key;
        FcvvdpDisplayModel model;
    } table[] = {
        {"standard_4k", CVVDP_DISPLAY_STANDARD_4K},
        {"standard_fhd", CVVDP_DISPLAY_STANDARD_FHD},
        {"standard_hdr_pq", CVVDP_DISPLAY_STANDARD_HDR_PQ},
        {"standard_hdr_hlg", CVVDP_DISPLAY_STANDARD_HDR_HLG},
        {"standard_hdr_dark", CVVDP_DISPLAY_STANDARD_HDR_DARK},
        {"standard_hdr_linear", CVVDP_DISPLAY_STANDARD_HDR_LINEAR},
        {"standard_hdr_linear_zoom", CVVDP_DISPLAY_STANDARD_HDR_LINEAR_ZOOM},
    };
    if (!key) return CVVDP_ERROR_INVALID_MODEL;
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        if (strcmp(table[i].key, key) == 0) {
            *out = table[i].model;
            return CVVDP_OK;
        }
    }
    return CVVDP_ERROR_INVALID_MODEL;
}

/*
 * the image data is already linear, so the colorspace field only states
 * that no further transfer decoding is needed (fcvvdp currently ignores
 * the field for RGB_FLOAT input)
 */
static FcvvdpColorspace trc_to_colorspace(const int trc) {
    (void)trc;
    return CVVDP_COLORSPACE_LINEAR;
}

/*
 * pre-scale for fcvvdp's display model, see the file header comment;
 * Y_black = max_luminance / contrast like Vship's DisplayModel
 * (cvvdp/display_models.hpp getBlackLevel) and fcvvdp's cvvdp_init_display
 */
static float trc_output_scale(const SideCfg* const cfg,
                              const FcvvdpDisplayParams* const dp) {
    if (cfg->trc == FCVVDP_VSHIP_TRC_PQ || !dp->is_hdr) return 1.0f;
    return (dp->max_luminance - dp->max_luminance / dp->contrast) / 100.0f;
}

static FcvvdpError side_init(Side* const s,
                             const FcvvdpVshipColorspace_t* const cs) {
    SideCfg* const cfg = &s->cfg;

    if (cs->width <= 0 || cs->height <= 0)
        return CVVDP_ERROR_INVALID_DIMENSIONS;

    const int64_t tw =
        cs->target_width == -1 ? cs->width : cs->target_width;
    const int64_t th =
        cs->target_height == -1 ? cs->height : cs->target_height;
    if (tw <= 0 || th <= 0) return CVVDP_ERROR_INVALID_DIMENSIONS;

    const FcvvdpVshipCropRectangle_t* const cr = &cs->crop;
    if (cr->top < 0 || cr->bottom < 0 || cr->left < 0 || cr->right < 0)
        return CVVDP_ERROR_INVALID_DIMENSIONS;

    const int64_t fw = tw - cr->left - cr->right;
    const int64_t fh = th - cr->top - cr->bottom;
    if (fw <= 0 || fh <= 0 || fw > INT_MAX / 6 || fh > INT_MAX)
        return CVVDP_ERROR_INVALID_DIMENSIONS;

    switch (cs->sample) {
        case FCVVDP_VSHIP_SAMPLE_UINT8:
            cfg->bits = 8;
            cfg->bytes_per_sample = 1;
            break;
        case FCVVDP_VSHIP_SAMPLE_UINT9:
            cfg->bits = 9;
            cfg->bytes_per_sample = 2;
            break;
        case FCVVDP_VSHIP_SAMPLE_UINT10:
            cfg->bits = 10;
            cfg->bytes_per_sample = 2;
            break;
        case FCVVDP_VSHIP_SAMPLE_UINT12:
            cfg->bits = 12;
            cfg->bytes_per_sample = 2;
            break;
        case FCVVDP_VSHIP_SAMPLE_UINT14:
            cfg->bits = 14;
            cfg->bytes_per_sample = 2;
            break;
        case FCVVDP_VSHIP_SAMPLE_UINT16:
            cfg->bits = 16;
            cfg->bytes_per_sample = 2;
            break;
        case FCVVDP_VSHIP_SAMPLE_FLOAT:
            cfg->bits = 1; // Vship's bitprecisionSample default
            cfg->bytes_per_sample = 4;
            cfg->is_float = 1;
            break;
        default: // HALF unsupported
            return CVVDP_ERROR_INVALID_FORMAT;
    }

    switch (cs->YUVMatrix) {
        case FCVVDP_VSHIP_MATRIX_RGB:
        case FCVVDP_VSHIP_MATRIX_BT709:
        case FCVVDP_VSHIP_MATRIX_BT470_BG:
        case FCVVDP_VSHIP_MATRIX_ST170_M:
        case FCVVDP_VSHIP_MATRIX_YCGCO:
        case FCVVDP_VSHIP_MATRIX_BT2020_NCL:
            cfg->matrix = (int)cs->YUVMatrix;
            break;
        default: // BT2020_CL, BT2100_ICTCP, YCGCO_RE/RO unsupported
            return CVVDP_ERROR_INVALID_FORMAT;
    }
    cfg->is_yuv = cfg->matrix != FCVVDP_VSHIP_MATRIX_RGB;

    if (cs->subsampling.subw < 0 || cs->subsampling.subw > 1 ||
        cs->subsampling.subh < 0 || cs->subsampling.subh > 1)
        return CVVDP_ERROR_INVALID_FORMAT;
    if (cs->chromaLocation < FCVVDP_VSHIP_CHROMA_LEFT ||
        cs->chromaLocation > FCVVDP_VSHIP_CHROMA_TOP)
        return CVVDP_ERROR_INVALID_FORMAT;
    if (cs->range != FCVVDP_VSHIP_RANGE_LIMITED &&
        cs->range != FCVVDP_VSHIP_RANGE_FULL)
        return CVVDP_ERROR_INVALID_FORMAT;

    switch (cs->transferFunction) {
        case FCVVDP_VSHIP_TRC_BT709:
        case FCVVDP_VSHIP_TRC_BT470_M:
        case FCVVDP_VSHIP_TRC_BT470_BG:
        case FCVVDP_VSHIP_TRC_BT601:
        case FCVVDP_VSHIP_TRC_ST240_M:
        case FCVVDP_VSHIP_TRC_LINEAR:
        case FCVVDP_VSHIP_TRC_SRGB:
        case FCVVDP_VSHIP_TRC_PQ:
        case FCVVDP_VSHIP_TRC_ST428:
        case FCVVDP_VSHIP_TRC_HLG:
            cfg->trc = (int)cs->transferFunction;
            break;
        default:
            return CVVDP_ERROR_INVALID_FORMAT;
    }

    cfg->width = cs->width;
    cfg->height = cs->height;
    cfg->target_width = tw;
    cfg->target_height = th;
    cfg->final_width = (int)fw;
    cfg->final_height = (int)fh;
    cfg->crop_left = cr->left;
    cfg->crop_top = cr->top;
    cfg->subw = cs->subsampling.subw;
    cfg->subh = cs->subsampling.subh;
    cfg->chroma_location = (int)cs->chromaLocation;
    cfg->range = (int)cs->range;

    return CVVDP_OK;
}

static FcvvdpError side_alloc(Side* const s) {
    const SideCfg* const cfg = &s->cfg;
    const uint64_t npix = (uint64_t)cfg->width * (uint64_t)cfg->height;
    if (npix > (uint64_t)(SIZE_MAX / (3 * sizeof(float))))
        return CVVDP_ERROR_OUT_OF_MEMORY;

    const uint64_t fpx =
        (uint64_t)cfg->final_width * (uint64_t)cfg->final_height;
    if (fpx > (uint64_t)(SIZE_MAX / (3 * sizeof(float))))
        return CVVDP_ERROR_OUT_OF_MEMORY;

    s->conv = calloc((size_t)(npix * 3), sizeof(float));
    s->out = calloc((size_t)(fpx * 3), sizeof(float));
    if (!s->conv || !s->out) return CVVDP_ERROR_OUT_OF_MEMORY;

    s->img.width = cfg->final_width;
    s->img.height = cfg->final_height;
    s->img.stride = cfg->final_width * 3 * (int)sizeof(float);
    s->img.data = s->out;
    s->img.format = CVVDP_PIXEL_FORMAT_RGB_FLOAT;
    s->img.colorspace = trc_to_colorspace(cfg->trc);

    return CVVDP_OK;
}

/* mirrors Vship's FullRange (rangeToFull.hpp) */
static float expand_range(const SideCfg* const cfg, float v, const int chroma) {
    if (cfg->range == FCVVDP_VSHIP_RANGE_FULL) {
        if (!cfg->is_float) v /= (float)((1 << cfg->bits) - 1);
        if (chroma) v -= 0.5f;
        return v;
    }
    if (cfg->is_float) {
        v *= 256.0f;
    } else {
        v /= (float)(1 << (cfg->bits - 8));
    }
    return chroma ? (v - 128.0f) / 224.0f : (v - 16.0f) / 219.0f;
}

static float fetch_plane_value(const SideCfg* const cfg,
                               const uint8_t* const plane,
                               const int64_t stride, const int64_t x,
                               const int64_t y) {
    const uint8_t* const p = plane + y * stride + x * cfg->bytes_per_sample;
    float v;
    switch (cfg->bytes_per_sample) {
        case 1:
            v = (float)p[0];
            break;
        case 2: {
            uint16_t raw;
            memcpy(&raw, p, 2);
            // bitmask like Vship's PickValue: garbage high bits are not 0
            v = (float)(raw & ((1u << cfg->bits) - 1u));
            break;
        }
        default: {
            memcpy(&v, p, 4);
            break;
        }
    }
    return v;
}

/* cubic Hermite spline with Catmull-Rom tangents and Vship's Horner
   evaluation, mirrors CubicHermitSplineInterpolator
   (gpuColorToLinear/resize.hpp) */
static float hermit_eval(const float pm1, const float p0, const float p1,
                         const float p2, const float t) {
    const float m0 = (p1 - pm1) * 0.5f;
    const float m1 = (p2 - p0) * 0.5f;
    const float c3 = 2.0f * p0 + m0 - 2.0f * p1 + m1;
    const float c2 = -3.0f * p0 + 3.0f * p1 - 2.0f * m0 - m1;
    float res = c3;
    res *= t;
    res += c2;
    res *= t;
    res += m0;
    res *= t;
    res += p0;
    return res;
}

/* bicubic Hermit chroma upsampling at the correct chroma site, mirroring
   Vship's chromaUpsample.hpp: co-sited locations (Left/Top) hit samples
   directly or mid-segment (t = 0.5), Center locations evaluate the
   neighbouring segments at t = 0.25 / 0.75; taps are edge-replicated like
   getHorizontalInterpolator_device/getVerticalInterpolator_device */
static float sample_plane(const SideCfg* const cfg,
                          const uint8_t* const plane,
                          const int64_t stride, const int pw, const int ph,
                          const float sx, const float sy, const int chroma) {
    const int x0 = (int)floorf(sx);
    const int y0 = (int)floorf(sy);
    const float tx = sx - (float)x0;
    const float ty = sy - (float)y0;

    float rows[4];
    for (int j = 0; j < 4; j++) {
        const int yy = iclip(y0 + j - 1, 0, ph - 1);
        const float t0 = expand_range(
            cfg, fetch_plane_value(cfg, plane, stride,
                                   iclip(x0 - 1, 0, pw - 1), yy), chroma);
        const float t1 = expand_range(
            cfg, fetch_plane_value(cfg, plane, stride,
                                   iclip(x0, 0, pw - 1), yy), chroma);
        const float t2 = expand_range(
            cfg, fetch_plane_value(cfg, plane, stride,
                                   iclip(x0 + 1, 0, pw - 1), yy), chroma);
        const float t3 = expand_range(
            cfg, fetch_plane_value(cfg, plane, stride,
                                   iclip(x0 + 2, 0, pw - 1), yy), chroma);
        rows[j] = hermit_eval(t0, t1, t2, t3, tx);
    }
    return hermit_eval(rows[0], rows[1], rows[2], rows[3], ty);
}

/* mirrors Vship's ncl_yuv_to_rgb_from_kr_kb (YUVToLinRGB.hpp) */
static void ncl_to_rgb(const float y, const float u, const float v,
                       const float kr, const float kb, float* const r,
                       float* const g, float* const b) {
    const float kg = 1.0f - kr - kb;
    const float rr = y + v * (2.0f * (1.0f - kr));
    const float bb = y + u * (2.0f * (1.0f - kb));
    *r = rr;
    *b = bb;
    *g = (y - kr * rr - kb * bb) / kg;
}

static void matrix_to_rgb(const int matrix, const float y, const float u,
                          const float v, float* const r, float* const g,
                          float* const b) {
    switch (matrix) {
        case FCVVDP_VSHIP_MATRIX_BT709:
            ncl_to_rgb(y, u, v, 0.2126f, 0.0722f, r, g, b);
            break;
        case FCVVDP_VSHIP_MATRIX_BT470_BG:
        case FCVVDP_VSHIP_MATRIX_ST170_M:
            ncl_to_rgb(y, u, v, 0.299f, 0.114f, r, g, b);
            break;
        case FCVVDP_VSHIP_MATRIX_BT2020_NCL:
            ncl_to_rgb(y, u, v, 0.2627f, 0.0593f, r, g, b);
            break;
        case FCVVDP_VSHIP_MATRIX_YCGCO:
            *r = y + u - v;
            *g = y + v;
            *b = y - u - v;
            break;
        default: // MATRIX_RGB: plane order R,G,B (as Vship's FFVship unpack)
            *r = y;
            *g = u;
            *b = v;
            break;
    }
}

/* per-component transfer linearization, mirrors Vship's
   transferToLinear.hpp transferLinearize<float> specializations, including
   the signed handling of out-of-gamut values; output scales: SDR TRCs
   [0,1] (ST428 ~1.091), PQ [0,100] (Lmax = 100) */
static float linearize_component(const int trc, float a) {
    switch (trc) {
        case FCVVDP_VSHIP_TRC_LINEAR:
            return a;
        case FCVVDP_VSHIP_TRC_SRGB:
            if (a < 0.0f) {
                if (a < -0.04045f)
                    a = -powf((-a + 0.055f) * (1.0f / 1.055f), 2.4f);
                else
                    a *= 1.0f / 12.92f;
            } else {
                if (a > 0.04045f)
                    a = powf((a + 0.055f) * (1.0f / 1.055f), 2.4f);
                else
                    a *= 1.0f / 12.92f;
            }
            return a;
        case FCVVDP_VSHIP_TRC_BT709: // pure gamma 2.4, signed
            return a < 0.0f ? -powf(-a, 2.4f) : powf(a, 2.4f);
        case FCVVDP_VSHIP_TRC_BT470_M:
            return a < 0.0f ? -powf(-a, 2.2f) : powf(a, 2.2f);
        case FCVVDP_VSHIP_TRC_BT470_BG:
        case FCVVDP_VSHIP_TRC_BT601:
            return a < 0.0f ? -powf(-a, 2.8f) : powf(a, 2.8f);
        case FCVVDP_VSHIP_TRC_ST240_M:
            if (a >= 0.0f) {
                if (a < 0.0228f)
                    a = 4.0f * a;
                else
                    a = 1.1115f * powf(a, 0.45f) - 0.1115f;
            } else {
                if (a > -0.0228f)
                    a = 4.0f * a;
                else
                    a = -1.1115f * powf(-a, 0.45f) + 0.1115f;
            }
            return a;
        case FCVVDP_VSHIP_TRC_ST428: {
            const float v = a < 0.0f ? -powf(-a, 2.6f) : powf(a, 2.6f);
            return (float)(v * (52.37 / 48.0));
        }
        case FCVVDP_VSHIP_TRC_PQ: {
            const float n = 0.15930175781250000f;
            const float m = 78.843750000000000f;
            const float c1 = 0.83593750000000000f;
            const float c2 = 18.851562500000000f;
            const float c3 = 18.687500000000000f;
            a = fmaxf(a, 0.0f);
            a = powf(a, 1.0f / m);
            return 100.0f * powf(fmaxf(0.0f, a - c1) / (c2 - c3 * a),
                                 1.0f / n);
        }
        default: // unreachable, side_init validated the transfer function
            return a;
    }
}

/* HLG linearization on the linear-light float3 (inverse OETF then OOTF
   gamma 1.2), mirrors transferLinearizeHLG (transferToLinear.hpp);
   output is linear [0,1] */
static void linearize_hlg(float rgb[3]) {
    const float a = 0.17883277f;
    const float b = 1.0f - 4.0f * a;
    const float c = 0.5f - a * -0.3350097945111627f;

    // inverse OETF
    for (int i = 0; i < 3; i++) {
        const float v = rgb[i];
        rgb[i] = v <= 0.5f ? v * v / 3.0f
                           : (expf((v - c) / a) + b) / 12.0f;
    }

    // OOTF
    const float ys =
        0.2627f * rgb[0] + 0.6780f * rgb[1] + 0.0593f * rgb[2];
    const float ootf = powf(ys, 1.2f - 1.0f);
    rgb[0] *= ootf;
    rgb[1] *= ootf;
    rgb[2] *= ootf;
}

static void convert_frame(const Side* const s, const uint8_t* const planes[3],
                          const int64_t strides[3]) {
    const SideCfg* const cfg = &s->cfg;
    const int w = (int)cfg->width;
    const int h = (int)cfg->height;
    // chroma plane dims, mirrors Vship's Converter::convert plane_widths
    const int cw = (int)(((cfg->width - 1) >> cfg->subw) + 1);
    const int ch = (int)(((cfg->height - 1) >> cfg->subh) + 1);
    const float inv_sw = 1.0f / (float)(1 << cfg->subw);
    const float inv_sh = 1.0f / (float)(1 << cfg->subh);
    const int h_co_sited =
        cfg->chroma_location == FCVVDP_VSHIP_CHROMA_LEFT ||
        cfg->chroma_location == FCVVDP_VSHIP_CHROMA_TOPLEFT;
    const int v_co_sited =
        cfg->chroma_location == FCVVDP_VSHIP_CHROMA_TOPLEFT ||
        cfg->chroma_location == FCVVDP_VSHIP_CHROMA_TOP;
    const int chroma_expand = cfg->is_yuv;

    for (int y = 0; y < h; y++) {
        const float cy = v_co_sited ? (float)y * inv_sh
                                    : ((float)y + 0.5f) * inv_sh - 0.5f;
        float* const row = s->conv + (size_t)y * w * 3;
        for (int x = 0; x < w; x++) {
            const float cx = h_co_sited ? (float)x * inv_sw
                                        : ((float)x + 0.5f) * inv_sw - 0.5f;
            const float p0 = expand_range(
                cfg, fetch_plane_value(cfg, planes[0], strides[0], x, y), 0);
            const float p1 = sample_plane(cfg, planes[1], strides[1], cw, ch,
                                          cx, cy, chroma_expand);
            const float p2 = sample_plane(cfg, planes[2], strides[2], cw, ch,
                                          cx, cy, chroma_expand);
            float rgb[3];
            matrix_to_rgb(cfg->matrix, p0, p1, p2, &rgb[0], &rgb[1], &rgb[2]);
            if (cfg->trc == FCVVDP_VSHIP_TRC_HLG) {
                linearize_hlg(rgb);
            } else {
                for (int c = 0; c < 3; c++)
                    rgb[c] = linearize_component(cfg->trc, rgb[c]);
            }
            row[(size_t)x * 3 + 0] = rgb[0];
            row[(size_t)x * 3 + 1] = rgb[1];
            row[(size_t)x * 3 + 2] = rgb[2];
        }
    }
}

static void sample_conv3(const float* const conv, const int w, const int h,
                         const float sx, const float sy, float* const out3) {
    const float xc = sx < 0.0f ? 0.0f
                               : sx > (float)(w - 1) ? (float)(w - 1) : sx;
    const float yc = sy < 0.0f ? 0.0f
                               : sy > (float)(h - 1) ? (float)(h - 1) : sy;
    const int x0 = (int)xc;
    const int y0 = (int)yc;
    const int x1 = x0 + 1 < w ? x0 + 1 : x0;
    const int y1 = y0 + 1 < h ? y0 + 1 : y0;
    const float fx = xc - (float)x0;
    const float fy = yc - (float)y0;

    const float* const p00 = conv + ((size_t)y0 * w + x0) * 3;
    const float* const p01 = conv + ((size_t)y0 * w + x1) * 3;
    const float* const p10 = conv + ((size_t)y1 * w + x0) * 3;
    const float* const p11 = conv + ((size_t)y1 * w + x1) * 3;
    for (int c = 0; c < 3; c++) {
        const float top = p00[c] + (p01[c] - p00[c]) * fx;
        const float bottom = p10[c] + (p11[c] - p10[c]) * fx;
        out3[c] = top + (bottom - top) * fy;
    }
}

/* resize to target then crop, matching Vship's resize-then-crop order;
   the crop offset is folded into the sampling coordinates. Runs in linear
   light like Vship, which resizes after YUVToLinRGBPipeline */
static void finalize_frame(const Side* const s) {
    const SideCfg* const cfg = &s->cfg;
    const int w = (int)cfg->width;
    const int h = (int)cfg->height;
    const int fw = cfg->final_width;
    const int fh = cfg->final_height;
    const float xw = (float)w / (float)cfg->target_width;
    const float xh = (float)h / (float)cfg->target_height;

    for (int y = 0; y < fh; y++) {
        const float sy =
            ((float)y + (float)cfg->crop_top + 0.5f) * xh - 0.5f;
        float* const row = s->out + (size_t)y * fw * 3;
        for (int x = 0; x < fw; x++) {
            const float sx =
                ((float)x + (float)cfg->crop_left + 0.5f) * xw - 0.5f;
            float rgb[3];
            sample_conv3(s->conv, w, h, sx, sy, rgb);
            row[(size_t)x * 3 + 0] = rgb[0] * s->out_scale;
            row[(size_t)x * 3 + 1] = rgb[1] * s->out_scale;
            row[(size_t)x * 3 + 2] = rgb[2] * s->out_scale;
        }
    }
}

static FcvvdpError validate_strides(const SideCfg* const cfg,
                                    const int64_t strides[3]) {
    // Vship reads lineSize*(height-1) + bytesize*width bytes per plane
    const int64_t cw = ((cfg->width - 1) >> cfg->subw) + 1;
    const int64_t min_strides[3] = {
        cfg->bytes_per_sample * cfg->width,
        cfg->bytes_per_sample * cw,
        cfg->bytes_per_sample * cw,
    };
    for (int i = 0; i < 3; i++) {
        if (strides[i] < min_strides[i]) return CVVDP_ERROR_INVALID_FORMAT;
    }
    return CVVDP_OK;
}

FcvvdpError fcvvdp_vship_create(FcvvdpVshipCtx** const out_ctx,
                                const FcvvdpVshipColorspace_t* const src_colorspace,
                                const FcvvdpVshipColorspace_t* const dis_colorspace,
                                const float fps, const char* const model_key,
                                const unsigned threads) {
    if (!out_ctx || !src_colorspace || !dis_colorspace)
        return CVVDP_ERROR_NULL_POINTER;
    *out_ctx = NULL;

    FcvvdpDisplayModel model;
    FcvvdpError err = model_from_key(model_key, &model);
    if (err != CVVDP_OK) return err;

    FcvvdpDisplayParams display_params;
    err = cvvdp_get_display_params(model, &display_params);
    if (err != CVVDP_OK) return err;

    FcvvdpVshipCtx* const vc = calloc(1, sizeof(*vc));
    if (!vc) return CVVDP_ERROR_OUT_OF_MEMORY;

    err = side_init(&vc->src, src_colorspace);
    if (err == CVVDP_OK) err = side_init(&vc->dis, dis_colorspace);
    if (err == CVVDP_OK &&
        (vc->src.cfg.final_width != vc->dis.cfg.final_width ||
         vc->src.cfg.final_height != vc->dis.cfg.final_height))
        err = CVVDP_ERROR_DIMENSION_MISMATCH;
    if (err == CVVDP_OK) {
        vc->src.out_scale = trc_output_scale(&vc->src.cfg, &display_params);
        vc->dis.out_scale = trc_output_scale(&vc->dis.cfg, &display_params);
    }
    if (err == CVVDP_OK) err = side_alloc(&vc->src);
    if (err == CVVDP_OK) err = side_alloc(&vc->dis);
    if (err == CVVDP_OK)
        err = cvvdp_create(vc->src.cfg.final_width, vc->src.cfg.final_height,
                           fps, model, threads, NULL, &vc->inner);
    if (err != CVVDP_OK) {
        fcvvdp_vship_destroy(vc);
        return err;
    }

    *out_ctx = vc;
    return CVVDP_OK;
}

FcvvdpError fcvvdp_vship_compute(FcvvdpVshipCtx* const ctx,
                                 double* const jod_out,
                                 const uint8_t* const srcp[3],
                                 const uint8_t* const disp[3],
                                 const int64_t src_stride[3],
                                 const int64_t dis_stride[3]) {
    if (!ctx || !jod_out || !srcp || !disp || !src_stride || !dis_stride)
        return CVVDP_ERROR_NULL_POINTER;
    for (int i = 0; i < 3; i++) {
        if (!srcp[i] || !disp[i]) return CVVDP_ERROR_NULL_POINTER;
    }

    FcvvdpError err = validate_strides(&ctx->src.cfg, src_stride);
    if (err != CVVDP_OK) return err;
    err = validate_strides(&ctx->dis.cfg, dis_stride);
    if (err != CVVDP_OK) return err;

    convert_frame(&ctx->src, srcp, src_stride);
    finalize_frame(&ctx->src);
    convert_frame(&ctx->dis, disp, dis_stride);
    finalize_frame(&ctx->dis);

    FcvvdpResult result;
    err = cvvdp_process_frame(ctx->inner, &ctx->src.img, &ctx->dis.img,
                              &result);
    if (err != CVVDP_OK) return err;

    *jod_out = result.jod;
    return CVVDP_OK;
}

void fcvvdp_vship_destroy(FcvvdpVshipCtx* const ctx) {
    if (!ctx) return;
    if (ctx->inner) cvvdp_destroy(ctx->inner);
    free(ctx->src.conv);
    free(ctx->src.out);
    free(ctx->dis.conv);
    free(ctx->dis.out);
    free(ctx);
}
