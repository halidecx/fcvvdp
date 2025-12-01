/*
 * CVVDP - ColorVideoVDP C Implementation
 *
 * A standalone C implementation of the ColorVideoVDP perceptual video quality metric.
 * Based on the GPU implementation in Vship.
 *
 * This implementation supports:
 * - Single image comparison (returns JOD score)
 * - Video sequence comparison (temporal integration)
 * - Various display models
 */

#ifndef CVVDP_H
#define CVVDP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
typedef enum {
    CVVDP_OK = 0,
    CVVDP_ERROR_NULL_POINTER = -1,
    CVVDP_ERROR_INVALID_DIMENSIONS = -2,
    CVVDP_ERROR_INVALID_FORMAT = -3,
    CVVDP_ERROR_INVALID_MODEL = -4,
    CVVDP_ERROR_OUT_OF_MEMORY = -5,
    CVVDP_ERROR_DIMENSION_MISMATCH = -6,
    CVVDP_ERROR_NOT_INITIALIZED = -7
} cvvdp_error_t;

/* Pixel format enumeration */
typedef enum {
    CVVDP_PIXEL_FORMAT_RGB_FLOAT,    /* RGB, 32-bit float per channel, [0,1] range for SDR */
    CVVDP_PIXEL_FORMAT_RGB_UINT8,    /* RGB, 8-bit unsigned int per channel, [0,255] */
    CVVDP_PIXEL_FORMAT_RGB_UINT16    /* RGB, 16-bit unsigned int per channel, [0,65535] */
} cvvdp_pixel_format_t;

/* Color space / transfer function */
typedef enum {
    CVVDP_COLORSPACE_SRGB,           /* Standard sRGB */
    CVVDP_COLORSPACE_LINEAR,         /* Linear RGB */
    CVVDP_COLORSPACE_PQ,             /* PQ (Perceptual Quantizer) for HDR */
    CVVDP_COLORSPACE_HLG             /* Hybrid Log-Gamma for HDR */
} cvvdp_colorspace_t;

/* Display model presets */
typedef enum {
    CVVDP_DISPLAY_STANDARD_FHD,      /* 24" FullHD, 200 cd/m², office lighting */
    CVVDP_DISPLAY_STANDARD_4K,       /* 30" 4K, 200 cd/m², office lighting */
    CVVDP_DISPLAY_STANDARD_HDR_PQ,   /* 30" 4K HDR, 1500 cd/m², low light */
    CVVDP_DISPLAY_STANDARD_HDR_HLG,  /* Same as HDR_PQ */
    CVVDP_DISPLAY_STANDARD_HDR_LINEAR, /* 30" 4K HDR linear */
    CVVDP_DISPLAY_STANDARD_HDR_DARK, /* 30" 4K HDR, dark room */
    CVVDP_DISPLAY_STANDARD_HDR_LINEAR_ZOOM, /* Close viewing distance */
    CVVDP_DISPLAY_CUSTOM             /* Use custom display parameters */
} cvvdp_display_model_t;

/* Custom display parameters (used when model is CVVDP_DISPLAY_CUSTOM) */
typedef struct {
    int resolution_width;
    int resolution_height;
    float viewing_distance_meters;
    float diagonal_size_inches;
    float max_luminance;             /* cd/m² */
    float contrast;                  /* Contrast ratio */
    float ambient_light;             /* E_ambient in lux */
    float reflectivity;              /* k_refl */
    int is_hdr;                      /* 0 for SDR, 1 for HDR */
} cvvdp_display_params_t;

/* Image descriptor */
typedef struct {
    const void* data;                /* Pointer to pixel data */
    int width;
    int height;
    int stride;                      /* Bytes per row (0 = tightly packed) */
    cvvdp_pixel_format_t format;
    cvvdp_colorspace_t colorspace;
} cvvdp_image_t;

/* CVVDP context (opaque handle) */
typedef struct cvvdp_context cvvdp_context_t;

/* Results structure */
typedef struct {
    double jod;                      /* Just-Objectionable-Difference score (0-10) */
    double quality;                  /* Quality score (internal, before JOD transform) */
} cvvdp_result_t;

/*
 * Create a new CVVDP context
 *
 * @param width         Image width in pixels
 * @param height        Image height in pixels
 * @param fps           Frames per second (use 0 for single images)
 * @param display_model Display model preset
 * @param custom_params Custom display parameters (only used if display_model is CVVDP_DISPLAY_CUSTOM)
 * @param out_ctx       Output pointer to created context
 *
 * @return CVVDP_OK on success, error code otherwise
 */
cvvdp_error_t cvvdp_create(
    int width,
    int height,
    float fps,
    cvvdp_display_model_t display_model,
    const cvvdp_display_params_t* custom_params,
    cvvdp_context_t** out_ctx
);

/*
 * Destroy a CVVDP context and free all resources
 *
 * @param ctx Context to destroy
 */
void cvvdp_destroy(cvvdp_context_t* ctx);

/*
 * Compare two images/frames and compute quality score
 *
 * For video sequences, call this function for each frame pair in order.
 * The temporal filtering will accumulate the results.
 *
 * @param ctx        CVVDP context
 * @param reference  Reference image
 * @param distorted  Distorted image
 * @param result     Output result structure
 *
 * @return CVVDP_OK on success, error code otherwise
 */
cvvdp_error_t cvvdp_process_frame(
    cvvdp_context_t* ctx,
    const cvvdp_image_t* reference,
    const cvvdp_image_t* distorted,
    cvvdp_result_t* result
);

/*
 * Reset the temporal accumulator
 *
 * Call this between separate video sequences when reusing a context.
 *
 * @param ctx CVVDP context
 *
 * @return CVVDP_OK on success, error code otherwise
 */
cvvdp_error_t cvvdp_reset(cvvdp_context_t* ctx);

/*
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
cvvdp_error_t cvvdp_compare_images(
    const cvvdp_image_t* reference,
    const cvvdp_image_t* distorted,
    cvvdp_display_model_t display_model,
    const cvvdp_display_params_t* custom_params,
    cvvdp_result_t* result
);

/*
 * Get default display parameters for a given model
 *
 * @param model      Display model preset
 * @param out_params Output parameter structure
 *
 * @return CVVDP_OK on success, error code otherwise
 */
cvvdp_error_t cvvdp_get_display_params(
    cvvdp_display_model_t model,
    cvvdp_display_params_t* out_params
);

/*
 * Get error message string
 *
 * @param error Error code
 *
 * @return Human-readable error message
 */
const char* cvvdp_error_string(cvvdp_error_t error);

/*
 * Get version string
 *
 * @return Version string (e.g., "1.0.0")
 */
const char* cvvdp_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* CVVDP_H */
