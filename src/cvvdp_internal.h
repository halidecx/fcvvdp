// © 2025 Halide Compression, LLC. All Rights Reserved.
#ifndef CVVDP_INTERNAL_H
#define CVVDP_INTERNAL_H

#include <math.h>
#include <stdlib.h>
#include "cvvdp.h"
#include "lut.h"

/* Mathematical constants */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TAU (2.0 * M_PI)

/* Maximum pyramid levels */
#define CVVDP_MAX_LEVELS 14

/* CSF LUT size */
#define CVVDP_LUT_SIZE 32

/* Gaussian kernel size */
#define CVVDP_GAUSSIAN_SIZE 8

/* Number of color channels (Y_sustained, RG_sustained, YV_sustained, Y_transient) */
#define CVVDP_NUM_CHANNELS 4

/* Display model structure */
typedef struct {
    int resolution_width;
    int resolution_height;
    float viewing_distance_meters;
    float diagonal_size_inches;
    float max_luminance;
    float contrast;
    float ambient_light;
    float reflectivity;
    float exposure;
    int is_hdr;

    /* Computed values */
    float ppd;                       /* Pixels per degree */
    float black_level;
    float refl_level;
} cvvdp_display_t;

/* Temporal filter structure */
typedef struct {
    float* kernel[4];                /* Convolution kernels for 4 channels */
    int size;                        /* Kernel size */
} cvvdp_temporal_filter_t;

/* Temporal ring buffer for frame storage */
typedef struct {
    float* data;                     /* Buffer for frames (3 planes per frame) */
    int max_frames;                  /* Maximum number of frames that can be stored */
    int num_frames;                  /* Current number of frames stored */
    int current_index;               /* Index of most recent frame */
    int width;
    int height;
    cvvdp_temporal_filter_t filter;
} cvvdp_temporal_ring_t;

/* CSF (Contrast Sensitivity Function) handler */
typedef struct {
    float log_L_bkg[CVVDP_LUT_SIZE]; /* Log of background luminance LUT indices */
    float* log_S_LUT;                /* Sensitivity LUT: [num_bands * 4 channels * 32] */
    int num_bands;
} cvvdp_csf_t;

/* Laplacian pyramid level */
typedef struct {
    float* data[4];                  /* 4 channels per level */
    float* L_bkg;                    /* Background luminance */
    int width;
    int height;
    float frequency;
} cvvdp_pyramid_level_t;

/* Laplacian pyramid */
typedef struct {
    cvvdp_pyramid_level_t* levels;
    int num_levels;
    int base_width;
    int base_height;
    float ppd;
} cvvdp_pyramid_t;

/* Gaussian blur handle */
typedef struct {
    float kernel[2 * CVVDP_GAUSSIAN_SIZE + 1];
    float kernel_integral[2 * CVVDP_GAUSSIAN_SIZE + 2];
} cvvdp_gaussian_t;

/* Main CVVDP context structure */
struct cvvdp_context {
    /* Configuration */
    int width;
    int height;
    float fps;
    cvvdp_display_t display;

    /* Processing state */
    cvvdp_temporal_ring_t ring_ref;
    cvvdp_temporal_ring_t ring_dis;
    cvvdp_csf_t csf;
    cvvdp_gaussian_t gaussian;

    /* Pyramid info */
    int num_bands;
    float* band_frequencies;

    /* Score accumulation */
    int num_frames;
    double score_square_sum;

    /* Work buffers */
    float* work_buffer;              /* General purpose work buffer */
    size_t work_buffer_size;

    /* Temporary channel buffers */
    float* Y_sustained;
    float* RG_sustained;
    float* YV_sustained;
    float* Y_transient;
};

/* Display model functions */
void cvvdp_init_display(cvvdp_display_t* display, cvvdp_display_model_t model,
                        const cvvdp_display_params_t* custom);
float cvvdp_compute_ppd(const cvvdp_display_t* display);

/* Temporal filter functions */
cvvdp_error_t cvvdp_temporal_filter_init(cvvdp_temporal_filter_t* filter, float fps);
void cvvdp_temporal_filter_destroy(cvvdp_temporal_filter_t* filter);

/* Temporal ring buffer functions */
cvvdp_error_t cvvdp_temporal_ring_init(cvvdp_temporal_ring_t* ring, int width, int height, float fps);
void cvvdp_temporal_ring_destroy(cvvdp_temporal_ring_t* ring);
void cvvdp_temporal_ring_push(cvvdp_temporal_ring_t* ring, const float* frame);
void cvvdp_temporal_ring_reset(cvvdp_temporal_ring_t* ring);
float* cvvdp_temporal_ring_get_frame(cvvdp_temporal_ring_t* ring, int age);

/* CSF functions */
cvvdp_error_t cvvdp_csf_init(cvvdp_csf_t* csf, int width, int height, float ppd);
void cvvdp_csf_destroy(cvvdp_csf_t* csf);
float cvvdp_csf_sensitivity(const cvvdp_csf_t* csf, float L_bkg, int band, int channel);

/* Gaussian blur functions */
void cvvdp_gaussian_init(cvvdp_gaussian_t* gaussian);
void cvvdp_gaussian_blur(const cvvdp_gaussian_t* gaussian, const float* src,
                         float* dst, int width, int height);

/* Laplacian pyramid functions */
cvvdp_error_t cvvdp_pyramid_init(cvvdp_pyramid_t* pyr, int width, int height, float ppd);
void cvvdp_pyramid_destroy(cvvdp_pyramid_t* pyr);
void cvvdp_pyramid_build(cvvdp_pyramid_t* pyr, const float* Y_sus, const float* RG_sus,
                         const float* YV_sus, const float* Y_trans, float* work);
void cvvdp_gauss_pyr_reduce(const float* src, float* dst, int src_width, int src_height);
void cvvdp_gauss_pyr_expand(const float* src, float* dst, int dst_width, int dst_height);

/* Color conversion functions */
void cvvdp_rgb_to_xyz(float* r, float* g, float* b, int count);
void cvvdp_xyz_to_dkl(float* x, float* y, float* z, int count);
void cvvdp_apply_display_model(float* plane, int count, const cvvdp_display_t* display, int is_hdr);

/* Temporal channel computation */
void cvvdp_compute_temporal_channels(cvvdp_temporal_ring_t* ring,
                                     float* Y_sus, float* RG_sus,
                                     float* YV_sus, float* Y_trans);

/* Masking model functions */
void cvvdp_apply_csf(cvvdp_csf_t* csf, float* L_bkg, float* ref, float* dis,
                     int width, int height, int channel, int band);
void cvvdp_compute_masking(float* ref[4], float* dis[4], int width, int height,
                           const cvvdp_gaussian_t* gaussian);
void cvvdp_compute_distortion(float* ref[4], float* dis[4], float* D_out[4],
                              int width, int height, const cvvdp_gaussian_t* gaussian);

/* Pooling functions */
float cvvdp_compute_norm(const float* data, int count, int power);
float cvvdp_compute_mean(const float* data, int count);

/* JOD transform */
float cvvdp_to_jod(float quality);

/* Image loading utilities */
cvvdp_error_t cvvdp_load_image(const cvvdp_image_t* img, float* out_planes[3]);

/* Utility functions */
static inline float cvvdp_maxf(float a, float b) { return a > b ? a : b; }
static inline float cvvdp_minf(float a, float b) { return a < b ? a : b; }
static inline float cvvdp_clampf(float x, float lo, float hi) {
    return cvvdp_minf(cvvdp_maxf(x, lo), hi);
}
static inline int cvvdp_maxi(int a, int b) { return a > b ? a : b; }
static inline int cvvdp_mini(int a, int b) { return a < b ? a : b; }

/* Memory helpers */
static inline float* cvvdp_alloc_float(size_t count) {
    return (float*)calloc(count, sizeof(float));
}

#endif /* CVVDP_INTERNAL_H */
