// © 2025 Halide Compression, LLC. All Rights Reserved.
#ifndef FCVVDP_H
#define FCVVDP_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version */
static const char* CVVDP_VERSION = "0.0.0";

/* Error codes */
typedef enum FcvvdpError {
    CVVDP_OK = 0,
    CVVDP_ERROR_NULL_POINTER = -1,
    CVVDP_ERROR_INVALID_DIMENSIONS = -2,
    CVVDP_ERROR_INVALID_FORMAT = -3,
    CVVDP_ERROR_INVALID_MODEL = -4,
    CVVDP_ERROR_OUT_OF_MEMORY = -5,
    CVVDP_ERROR_DIMENSION_MISMATCH = -6,
    CVVDP_ERROR_NOT_INITIALIZED = -7
} FcvvdpError;

/* Pixel formats */
typedef enum FcvvdpPixFmt {
    // RGB, f32 per channel; [0,1] range (SDR)
    CVVDP_PIXEL_FORMAT_RGB_FLOAT,

    // RGB, u8 per channel; [0,255]
    CVVDP_PIXEL_FORMAT_RGB_UINT8,

    // RGB, u16 per channel; [0,65535]
    CVVDP_PIXEL_FORMAT_RGB_UINT16
} FcvvdpPixFmt;

/* Color space / transfer function */
typedef enum FcvvdpColorspace {
    // sRGB
    CVVDP_COLORSPACE_SRGB,

    // linear RGB
    CVVDP_COLORSPACE_LINEAR,

    // PQ (HDR)
    CVVDP_COLORSPACE_PQ,

    // HLG (HDR)
    CVVDP_COLORSPACE_HLG
} FcvvdpColorspace;

/* Display model presets */
typedef enum FcvvdpDisplayModel {
    // 24" FullHD, 200 nit, office lighting
    CVVDP_DISPLAY_STANDARD_FHD,

    // 30" 4K, 200 nit, office lighting
    CVVDP_DISPLAY_STANDARD_4K,

    // 30" 4K HDR, 1500 nit, low light
    CVVDP_DISPLAY_STANDARD_HDR_PQ,
    CVVDP_DISPLAY_STANDARD_HDR_HLG,

    // 30" 4K HDR linear
    CVVDP_DISPLAY_STANDARD_HDR_LINEAR,

    // 30" 4K HDR dark room
    CVVDP_DISPLAY_STANDARD_HDR_DARK,

    // 30" 4K HDR linear (close viewing)
    CVVDP_DISPLAY_STANDARD_HDR_LINEAR_ZOOM,

    // use user-defined params
    CVVDP_DISPLAY_CUSTOM
} FcvvdpDisplayModel;

/* Custom display parameters (when model == CVVDP_DISPLAY_CUSTOM) */
typedef struct FcvvdpDisplayParams {
    int resolution_width, resolution_height;
    float viewing_distance_meters;
    float diagonal_size_inches;

    // nits
    float max_luminance;

    // contrast ratio
    float contrast;

    // ambient light level in lux
    float ambient_light;

    // reflectivity index (k_refl)
    float reflectivity;

    // 0=SDR, 1=HDR
    bool is_hdr;
} FcvvdpDisplayParams;

/* Image descriptor */
typedef struct FcvvdpImage {
    int width, height, stride;
    const void* data;
    FcvvdpPixFmt format;
    FcvvdpColorspace colorspace;
} FcvvdpImage;

/* CVVDP context (opaque handle) */
typedef struct FcvvdpCtx FcvvdpCtx;

/* Results */
typedef struct FcvvdpResult {
    // Just-Objectionable-Difference score (0-10)
    double jod;

    // Quality score (internal, before JOD transform)
    double quality;
} FcvvdpResult;

/**
 * Create a new CVVDP context
 *
 * @param width         Image width in pixels
 * @param height        Image height in pixels
 * @param fps           Frames per second (use 0 for single images)
 * @param display_model Display model preset
 * @param custom_params Custom display parameters (if CVVDP_DISPLAY_CUSTOM)
 * @param out_c         Output pointer to created context
 *
 * @return CVVDP_OK on success, error code otherwise
 */
FcvvdpError cvvdp_create(const int width,
                         const int height,
                         const float fps,
                         const FcvvdpDisplayModel display_model,
                         const FcvvdpDisplayParams* const custom_params,
                         FcvvdpCtx** const out_c);

/**
 * Destroy a CVVDP context and free all resources
 *
 * @param c Context to destroy
 */
void cvvdp_destroy(FcvvdpCtx* const c);

/**
 * Compare two images/frames and compute quality score
 *
 * For video sequences, call this function for each frame pair in order.
 * The temporal filtering will accumulate the results.
 *
 * @param c          CVVDP context
 * @param reference  Reference image
 * @param distorted  Distorted image
 * @param result     Output result structure
 *
 * @return CVVDP_OK on success, error code otherwise
 */
FcvvdpError cvvdp_process_frame(FcvvdpCtx* const c,
                                const FcvvdpImage* const reference,
                                const FcvvdpImage* const distorted,
                                FcvvdpResult* const result);

/**
 * Reset the temporal accumulator
 *
 * Call this between separate video sequences when reusing a context.
 *
 * @param c CVVDP context
 *
 * @return CVVDP_OK on success, error code otherwise
 */
FcvvdpError cvvdp_reset(FcvvdpCtx* const c);

/**
 * Compare two single images (convenience function)
 *
 * Creates a temporary context, processes the images, and destroys the context.
 * More efficient to use cvvdp_create/process_frame/destroy for multiple comparisons.
 *
 * @param reference     Reference image
 * @param distorted     Distorted image
 * @param display_model Display model preset
 * @param custom_params Custom display parameters (only used if display_model is CVVDP_DISPLAY_CUSTOM)
 * @param result        Output result structure
 *
 * @return CVVDP_OK on success, error code otherwise
 */
FcvvdpError cvvdp_compare_images(const FcvvdpImage* const reference,
                                 const FcvvdpImage* const distorted,
                                 FcvvdpDisplayModel display_model,
                                 const FcvvdpDisplayParams* const custom_params,
                                 FcvvdpResult* const result);

/**
 * Get default display parameters for a given model
 *
 * @param model      Display model preset
 * @param out_params Output parameter structure
 *
 * @return CVVDP_OK on success, error code otherwise
 */
FcvvdpError cvvdp_get_display_params(const FcvvdpDisplayModel model,
                                     FcvvdpDisplayParams* const out_params);

/**
 * Get error message string
 *
 * @param error Error code
 *
 * @return Human-readable error message
 */
const char* cvvdp_error_string(FcvvdpError error);

/**
 * Get version string
 *
 * @return Version string (e.g., "1.0.0")
 */
static const char* cvvdp_version_string(void) {
    return CVVDP_VERSION;
}

#ifdef __cplusplus
}
#endif

#endif /* FCVVDP_H */
