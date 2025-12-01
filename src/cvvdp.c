/*
 * CVVDP - ColorVideoVDP C Implementation
 *
 * A standalone C implementation of the ColorVideoVDP perceptual video quality metric.
 * Based on the GPU implementation in Vship.
 */

#include "cvvdp.h"
#include "cvvdp_internal.h"
#include "lut.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char* CVVDP_VERSION = "0.0.0";

const char* cvvdp_version_string(void) {
    return CVVDP_VERSION;
}

const char* cvvdp_error_string(cvvdp_error_t error) {
    switch (error) {
        case CVVDP_OK: return "Success";
        case CVVDP_ERROR_NULL_POINTER: return "Null pointer";
        case CVVDP_ERROR_INVALID_DIMENSIONS: return "Invalid dimensions";
        case CVVDP_ERROR_INVALID_FORMAT: return "Invalid format";
        case CVVDP_ERROR_INVALID_MODEL: return "Invalid display model";
        case CVVDP_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case CVVDP_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case CVVDP_ERROR_NOT_INITIALIZED: return "Not initialized";
        default: return "Unknown error";
    }
}

void cvvdp_init_display(cvvdp_display_t* display, cvvdp_display_model_t model,
                        const cvvdp_display_params_t* custom) {
    /* Set defaults */
    display->exposure = 1.0f;

    if (model == CVVDP_DISPLAY_CUSTOM && custom != NULL) {
        display->resolution_width = custom->resolution_width;
        display->resolution_height = custom->resolution_height;
        display->viewing_distance_meters = custom->viewing_distance_meters;
        display->diagonal_size_inches = custom->diagonal_size_inches;
        display->max_luminance = custom->max_luminance;
        display->contrast = custom->contrast;
        display->ambient_light = custom->ambient_light;
        display->reflectivity = custom->reflectivity;
        display->is_hdr = custom->is_hdr;
    } else {
        switch (model) {
            case CVVDP_DISPLAY_STANDARD_4K:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.7472f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 200.0f;
                display->contrast = 1000.0f;
                display->ambient_light = 250.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = 0;
                break;

            case CVVDP_DISPLAY_STANDARD_HDR_PQ:
            case CVVDP_DISPLAY_STANDARD_HDR_HLG:
            case CVVDP_DISPLAY_STANDARD_HDR_LINEAR:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.7472f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 1500.0f;
                display->contrast = 1000000.0f;
                display->ambient_light = 10.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = 1;
                break;

            case CVVDP_DISPLAY_STANDARD_HDR_DARK:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.7472f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 1500.0f;
                display->contrast = 1000000.0f;
                display->ambient_light = 0.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = 1;
                break;

            case CVVDP_DISPLAY_STANDARD_HDR_LINEAR_ZOOM:
                display->resolution_width = 3840;
                display->resolution_height = 2160;
                display->viewing_distance_meters = 0.25f;
                display->diagonal_size_inches = 30.0f;
                display->max_luminance = 10000.0f;
                display->contrast = 1000000.0f;
                display->ambient_light = 10.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = 1;
                break;

            case CVVDP_DISPLAY_STANDARD_FHD:
            default:
                display->resolution_width = 1920;
                display->resolution_height = 1080;
                display->viewing_distance_meters = 0.6f;
                display->diagonal_size_inches = 24.0f;
                display->max_luminance = 200.0f;
                display->contrast = 1000.0f;
                display->ambient_light = 250.0f;
                display->reflectivity = 0.005f;
                display->is_hdr = 0;
                break;
        }
    }

    /* Compute derived values */
    display->ppd = cvvdp_compute_ppd(display);
    display->black_level = display->max_luminance / display->contrast;
    display->refl_level = display->ambient_light * display->reflectivity / (float)M_PI;
}

float cvvdp_compute_ppd(const cvvdp_display_t* display) {
    float ar = (float)display->resolution_width / (float)display->resolution_height;
    float diag_mm = display->diagonal_size_inches * 25.4f;
    float height_m = sqrtf(diag_mm * diag_mm / (1.0f + ar * ar)) / 1000.0f;
    float width_m = ar * height_m;

    float pix_deg = 2.0f * 180.0f * atanf(0.5f * width_m /
                    (float)display->resolution_width / display->viewing_distance_meters) / (float)M_PI;
    return 1.0f / pix_deg;
}

cvvdp_error_t cvvdp_get_display_params(cvvdp_display_model_t model,
                                       cvvdp_display_params_t* out_params) {
    if (out_params == NULL) {
        return CVVDP_ERROR_NULL_POINTER;
    }

    cvvdp_display_t display;
    cvvdp_init_display(&display, model, NULL);

    out_params->resolution_width = display.resolution_width;
    out_params->resolution_height = display.resolution_height;
    out_params->viewing_distance_meters = display.viewing_distance_meters;
    out_params->diagonal_size_inches = display.diagonal_size_inches;
    out_params->max_luminance = display.max_luminance;
    out_params->contrast = display.contrast;
    out_params->ambient_light = display.ambient_light;
    out_params->reflectivity = display.reflectivity;
    out_params->is_hdr = display.is_hdr;

    return CVVDP_OK;
}

void cvvdp_gaussian_init(cvvdp_gaussian_t* gaussian) {
    float sigma = (float)CVVDP_PU_DILATE;
    int size = 2 * CVVDP_GAUSSIAN_SIZE + 1;

    gaussian->kernel_integral[0] = 0.0f;

    for (int i = 0; i < size; i++) {
        float x = (float)(i - CVVDP_GAUSSIAN_SIZE);
        gaussian->kernel[i] = expf(-x * x / (2.0f * sigma * sigma)) /
                              sqrtf((float)(TAU) * sigma * sigma);
        gaussian->kernel_integral[i + 1] = gaussian->kernel_integral[i] + gaussian->kernel[i];
    }
}

void cvvdp_gaussian_blur(const cvvdp_gaussian_t* gaussian, const float* src,
                         float* dst, int width, int height) {
    int ksize = CVVDP_GAUSSIAN_SIZE;
    float* temp = cvvdp_alloc_float((size_t)width * height);
    if (!temp) return;

    /* Horizontal pass */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -ksize; k <= ksize; k++) {
                int sx = x + k;
                /* Mirror boundary */
                if (sx < 0) sx = -sx - 1;
                if (sx >= width) sx = 2 * width - sx - 2;

                float w = gaussian->kernel[k + ksize];
                sum += src[y * width + sx] * w;
                weight_sum += w;
            }
            temp[y * width + x] = sum / weight_sum;
        }
    }

    /* Vertical pass */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -ksize; k <= ksize; k++) {
                int sy = y + k;
                /* Mirror boundary */
                if (sy < 0) sy = -sy - 1;
                if (sy >= height) sy = 2 * height - sy - 2;

                float w = gaussian->kernel[k + ksize];
                sum += temp[sy * width + x] * w;
                weight_sum += w;
            }
            dst[y * width + x] = sum / weight_sum;
        }
    }

    free(temp);
}

/* Inverse real FFT for temporal filter generation */
static void cvvdp_inverse_rfft(const float* input, int input_size, float* output, int size) {
    float* inp = cvvdp_alloc_float(input_size);
    if (!inp) return;

    memcpy(inp, input, input_size * sizeof(float));

    /* Normalization for FFT inversion */
    if (input_size > size / 2) {
        inp[size / 2] /= 2.0f;
    }
    inp[0] /= 2.0f;

    for (int i = 0; i < size; i++) {
        output[i] = 0.0f;
        for (int k = 0; k < input_size; k++) {
            int idx = (i + input_size - 1) % size;
            output[idx] += 2.0f * inp[k] * cosf(2.0f * (float)M_PI * k * i / size) / size;
        }
    }

    free(inp);
}

cvvdp_error_t cvvdp_temporal_filter_init(cvvdp_temporal_filter_t* filter, float fps) {
    if (fps <= 0) {
        filter->size = 1;
    } else {
        filter->size = (int)(ceilf(0.25f * fps / 2.0f) * 2) + 1;
        if (filter->size < 1) filter->size = 1;
    }

    int fft_size = filter->size / 2 + 1;

    /* Allocate kernels */
    for (int j = 0; j < 4; j++) {
        filter->kernel[j] = cvvdp_alloc_float(filter->size);
        if (!filter->kernel[j]) {
            for (int k = 0; k < j; k++) {
                free(filter->kernel[k]);
            }
            return CVVDP_ERROR_OUT_OF_MEMORY;
        }
    }

    /* Compute frequency bins */
    float* freq = cvvdp_alloc_float(fft_size);
    float* fft_domain = cvvdp_alloc_float(fft_size);
    if (!freq || !fft_domain) {
        for (int j = 0; j < 4; j++) free(filter->kernel[j]);
        free(freq);
        free(fft_domain);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < fft_size; i++) {
        freq[i] = fps * i / 2.0f;
    }

    /* Generate filters for each channel */
    for (int j = 0; j < 4; j++) {
        if (j < 3) {
            /* Sustained channels (low-pass) */
            for (int i = 0; i < fft_size; i++) {
                fft_domain[i] = expf(-powf(freq[i], CVVDP_BETA_TF[j]) / CVVDP_SIGMA_TF[j]);
            }
        } else {
            /* Transient channel (band-pass, peak at 5 Hz) */
            for (int i = 0; i < fft_size; i++) {
                float temp = powf(freq[i], CVVDP_BETA_TF[3]) - powf(5.0f, CVVDP_BETA_TF[3]);
                temp *= temp;
                fft_domain[i] = expf(-temp / CVVDP_SIGMA_TF[3]);
            }
        }

        cvvdp_inverse_rfft(fft_domain, fft_size, filter->kernel[j], filter->size);
    }

    free(freq);
    free(fft_domain);

    return CVVDP_OK;
}

void cvvdp_temporal_filter_destroy(cvvdp_temporal_filter_t* filter) {
    for (int j = 0; j < 4; j++) {
        if (filter->kernel[j]) {
            free(filter->kernel[j]);
            filter->kernel[j] = NULL;
        }
    }
}

cvvdp_error_t cvvdp_temporal_ring_init(cvvdp_temporal_ring_t* ring, int width, int height, float fps) {
    ring->width = width;
    ring->height = height;
    ring->num_frames = 0;
    ring->current_index = 0;

    cvvdp_error_t err = cvvdp_temporal_filter_init(&ring->filter, fps);
    if (err != CVVDP_OK) return err;

    ring->max_frames = ring->filter.size;

    /* Allocate frame buffer (3 planes per frame) */
    size_t frame_size = (size_t)width * height * 3;
    ring->data = cvvdp_alloc_float(frame_size * ring->max_frames);
    if (!ring->data) {
        cvvdp_temporal_filter_destroy(&ring->filter);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    return CVVDP_OK;
}

void cvvdp_temporal_ring_destroy(cvvdp_temporal_ring_t* ring) {
    if (ring->data) {
        free(ring->data);
        ring->data = NULL;
    }
    cvvdp_temporal_filter_destroy(&ring->filter);
}

void cvvdp_temporal_ring_reset(cvvdp_temporal_ring_t* ring) {
    ring->num_frames = 0;
    ring->current_index = 0;
}

float* cvvdp_temporal_ring_get_frame(cvvdp_temporal_ring_t* ring, int age) {
    if (ring->num_frames == 0) return NULL;

    if (age >= ring->num_frames) {
        age = ring->num_frames - 1;
    }

    size_t plane_size = (size_t)ring->width * ring->height * 3;
    int idx = (ring->current_index + age) % ring->max_frames;
    return ring->data + idx * plane_size;
}

void cvvdp_temporal_ring_push(cvvdp_temporal_ring_t* ring, const float* frame) {
    ring->num_frames = cvvdp_mini(ring->max_frames, ring->num_frames + 1);
    ring->current_index = (ring->current_index + ring->max_frames - 1) % ring->max_frames;

    size_t plane_size = (size_t)ring->width * ring->height * 3;
    float* dst = ring->data + ring->current_index * plane_size;
    memcpy(dst, frame, plane_size * sizeof(float));
}

void cvvdp_rgb_to_xyz(float* r, float* g, float* b, int count) {
    for (int i = 0; i < count; i++) {
        float ri = r[i], gi = g[i], bi = b[i];

        /* BT.709 RGB to XYZ */
        float x = 0.4124564f * ri + 0.3575761f * gi + 0.1804375f * bi;
        float y = 0.2126729f * ri + 0.7151522f * gi + 0.0721750f * bi;
        float z = 0.0193339f * ri + 0.1191920f * gi + 0.9503041f * bi;

        r[i] = x;
        g[i] = y;
        b[i] = z;
    }
}

void cvvdp_xyz_to_dkl(float* x, float* y, float* z, int count) {
    for (int i = 0; i < count; i++) {
        float xi = x[i], yi = y[i], zi = z[i];

        /* XYZ to LMS (CIE 2006) */
        float L = 0.187596268556126f * xi + 0.585168649077728f * yi - 0.026384263306304f * zi;
        float M = -0.133397430663221f * xi + 0.405505777260049f * yi + 0.034502127690364f * zi;
        float S = 0.000244379021663f * xi - 0.000542995890619f * yi + 0.019406849066323f * zi;

        /* LMS to DKL (D65 adapted) */
        float lum = L + M;                      /* Luminance */
        float rg = lum - 3.311130179947035f * M; /* Red-green */
        float yv = 50.977571328718781f * S - lum; /* Yellow-violet */

        x[i] = lum;
        y[i] = rg;
        z[i] = yv;
    }
}

void cvvdp_apply_display_model(float* plane, int count, const cvvdp_display_t* display, int is_hdr) {
    float Y_peak = display->max_luminance;
    float Y_black = display->black_level;
    float Y_refl = display->refl_level;
    float exposure = display->exposure;

    for (int i = 0; i < count; i++) {
        float val = plane[i];

        if (is_hdr) {
            /* HDR: linear scaling with clamping */
            val *= 100.0f; /* Convert normalized to cd/m² */
            val = cvvdp_maxf(cvvdp_maxf(0.005f, Y_black),
                            cvvdp_minf(Y_peak, val * exposure)) + Y_black + Y_refl;
        } else {
            /* SDR: gamma/sRGB model */
            val = cvvdp_clampf(val * exposure, 0.0f, 1.0f);
            val = (Y_peak - Y_black) * val + Y_black + Y_refl;
        }

        plane[i] = val;
    }
}

cvvdp_error_t cvvdp_load_image(const cvvdp_image_t* img, float* out_planes[3]) {
    if (!img || !img->data || !out_planes) {
        return CVVDP_ERROR_NULL_POINTER;
    }

    int w = img->width;
    int h = img->height;
    int stride = img->stride;
    if (stride == 0) {
        switch (img->format) {
            case CVVDP_PIXEL_FORMAT_RGB_FLOAT:
                stride = w * 3 * sizeof(float);
                break;
            case CVVDP_PIXEL_FORMAT_RGB_UINT8:
                stride = w * 3 * sizeof(uint8_t);
                break;
            case CVVDP_PIXEL_FORMAT_RGB_UINT16:
                stride = w * 3 * sizeof(uint16_t);
                break;
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float r, g, b;
            int idx = y * w + x;

            switch (img->format) {
                case CVVDP_PIXEL_FORMAT_RGB_FLOAT: {
                    const float* row = (const float*)((const uint8_t*)img->data + y * stride);
                    r = row[x * 3 + 0];
                    g = row[x * 3 + 1];
                    b = row[x * 3 + 2];
                    break;
                }
                case CVVDP_PIXEL_FORMAT_RGB_UINT8: {
                    const uint8_t* row = (const uint8_t*)img->data + y * stride;
                    r = row[x * 3 + 0] / 255.0f;
                    g = row[x * 3 + 1] / 255.0f;
                    b = row[x * 3 + 2] / 255.0f;
                    /* Apply sRGB gamma */
                    r = (r < 0) ? -powf(-r, 2.4f) : powf(r, 2.4f);
                    g = (g < 0) ? -powf(-g, 2.4f) : powf(g, 2.4f);
                    b = (b < 0) ? -powf(-b, 2.4f) : powf(b, 2.4f);
                    break;
                }
                case CVVDP_PIXEL_FORMAT_RGB_UINT16: {
                    const uint16_t* row = (const uint16_t*)((const uint8_t*)img->data + y * stride);
                    r = row[x * 3 + 0] / 65535.0f;
                    g = row[x * 3 + 1] / 65535.0f;
                    b = row[x * 3 + 2] / 65535.0f;
                    break;
                }
                default:
                    return CVVDP_ERROR_INVALID_FORMAT;
            }

            out_planes[0][idx] = r;
            out_planes[1][idx] = g;
            out_planes[2][idx] = b;
        }
    }

    return CVVDP_OK;
}

void cvvdp_compute_temporal_channels(cvvdp_temporal_ring_t* ring,
                                     float* Y_sus, float* RG_sus,
                                     float* YV_sus, float* Y_trans) {
    int w = ring->width;
    int h = ring->height;
    size_t plane_size = (size_t)w * h;

    if (ring->num_frames == 1) {
        /* Single frame: no temporal filtering */
        float* frame = cvvdp_temporal_ring_get_frame(ring, 0);
        memcpy(Y_sus, frame, plane_size * sizeof(float));
        memcpy(RG_sus, frame + plane_size, plane_size * sizeof(float));
        memcpy(YV_sus, frame + 2 * plane_size, plane_size * sizeof(float));
        memset(Y_trans, 0, plane_size * sizeof(float));
        return;
    }

    /* Temporal convolution */
    memset(Y_sus, 0, plane_size * sizeof(float));
    memset(RG_sus, 0, plane_size * sizeof(float));
    memset(YV_sus, 0, plane_size * sizeof(float));
    memset(Y_trans, 0, plane_size * sizeof(float));

    for (int k = 0; k < ring->filter.size; k++) {
        float* frame = cvvdp_temporal_ring_get_frame(ring, k);
        if (!frame) continue;

        float k0 = ring->filter.kernel[0][k]; /* Y sustained */
        float k1 = ring->filter.kernel[1][k]; /* RG sustained */
        float k2 = ring->filter.kernel[2][k]; /* YV sustained */
        float k3 = ring->filter.kernel[3][k]; /* Y transient */

        for (size_t i = 0; i < plane_size; i++) {
            float Y = frame[i];
            Y_sus[i] += Y * k0;
            Y_trans[i] += Y * k3;
            RG_sus[i] += frame[i + plane_size] * k1;
            YV_sus[i] += frame[i + 2 * plane_size] * k2;
        }
    }
}

static const float GAUSS_PYR_KERNEL[5] = {
    0.25f - 0.4f/2.0f, 0.25f, 0.4f, 0.25f, 0.25f - 0.4f/2.0f
};

void cvvdp_gauss_pyr_reduce(const float* src, float* dst, int src_w, int src_h) {
    int dst_w = (src_w + 1) / 2;
    int dst_h = (src_h + 1) / 2;

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            float val = 0.0f;

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int sx = 2 * dx + kx;
                    int sy = 2 * dy + ky;

                    /* Symmetric padding */
                    if (sx < 0) sx = -sx - 1;
                    if (sx >= src_w) sx = 2 * src_w - sx - 2;
                    if (sy < 0) sy = -sy - 1;
                    if (sy >= src_h) sy = 2 * src_h - sy - 2;

                    val += GAUSS_PYR_KERNEL[kx + 2] * GAUSS_PYR_KERNEL[ky + 2] *
                           src[sy * src_w + sx];
                }
            }

            dst[dy * dst_w + dx] = val;
        }
    }
}

void cvvdp_gauss_pyr_expand(const float* src, float* dst, int dst_w, int dst_h) {
    int src_w = (dst_w + 1) / 2;
    int src_h = (dst_h + 1) / 2;

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            float val = 0.0f;
            int parity_x = dx % 2;
            int parity_y = dy % 2;

            for (int ky = -2 + parity_y; ky <= 2; ky += 2) {
                for (int kx = -2 + parity_x; kx <= 2; kx += 2) {
                    int sx = (dx + kx) / 2;
                    int sy = (dy + ky) / 2;

                    /* Symmetric padding */
                    if (sx < 0) sx = -sx - 1;
                    if (sx >= src_w) sx = 2 * src_w - sx - 2;
                    if (sy < 0) sy = -sy - 1;
                    if (sy >= src_h) sy = 2 * src_h - sy - 2;

                    val += 2.0f * GAUSS_PYR_KERNEL[kx + 2] * 2.0f * GAUSS_PYR_KERNEL[ky + 2] *
                           src[sy * src_w + sx];
                }
            }

            dst[dy * dst_w + dx] = val;
        }
    }
}

/* Get pyramid band frequencies */
static int cvvdp_get_band_frequencies(int width, int height, float ppd, float* out_freqs) {
    float min_freq = 0.2f;
    int max_level_res = (int)(log2f((float)cvvdp_mini(width, height))) - 1;
    int max_level_ppd = (int)ceilf(-log2f(2.0f * min_freq / 0.3228f / ppd)) + 1;
    int max_level = cvvdp_mini(max_level_res, cvvdp_mini(max_level_ppd, CVVDP_MAX_LEVELS));

    out_freqs[0] = 0.5f * ppd;
    for (int i = 1; i < max_level; i++) {
        out_freqs[i] = 0.3228f * 0.5f * ppd / (float)(1 << (i - 1));
    }

    return max_level;
}

cvvdp_error_t cvvdp_csf_init(cvvdp_csf_t* csf, int width, int height, float ppd) {
    float freqs[CVVDP_MAX_LEVELS];
    csf->num_bands = cvvdp_get_band_frequencies(width, height, ppd, freqs);

    /* Precompute log L_bkg values from the GPU LUT */
    for (int i = 0; i < CVVDP_LUT_SIZE; i++) {
        csf->log_L_bkg[i] = log10f(L_bkg[i]);
    }

    /* Precompute log rho values from the GPU LUT */
    float log_rho[CVVDP_LUT_SIZE];
    for (int i = 0; i < CVVDP_LUT_SIZE; i++) {
        log_rho[i] = log10f(rho[i]);
    }

    /* Allocate sensitivity LUT */
    csf->log_S_LUT = cvvdp_alloc_float(csf->num_bands * 4 * CVVDP_LUT_SIZE);
    if (!csf->log_S_LUT) {
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    /* Precompute sensitivity values for each band/channel combination
     * This matches the GPU CSF_Handler::init() logic exactly:
     * For each band frequency, interpolate the D2LUT along the rho dimension
     */
    for (int band = 0; band < csf->num_bands; band++) {
        float log_freq = log10f(freqs[band]);

        /* Find frequency index for interpolation (same logic as GPU indexSearch) */
        int rho_idx = 0;
        for (int i = 1; i < CVVDP_LUT_SIZE; i++) {
            if (log_rho[i] > log_freq) {
                rho_idx = i - 1;
                break;
            }
            rho_idx = i;
        }
        if (rho_idx > 30) rho_idx = 30;

        float slope = (log_freq - log_rho[rho_idx]) / (log_rho[rho_idx + 1] - log_rho[rho_idx]);

        for (int channel = 0; channel < 4; channel++) {
            /* For each luminance level, interpolate from D2LUT along rho dimension */
            for (int l = 0; l < CVVDP_LUT_SIZE; l++) {
                /* D2LUT is indexed as [channel][L_bkg_idx][rho_idx] */
                float y0 = D2LUT[channel][l][rho_idx];
                float y1 = D2LUT[channel][l][rho_idx + 1];
                float log_S = y0 + (y1 - y0) * slope;

                csf->log_S_LUT[(band * 4 + channel) * CVVDP_LUT_SIZE + l] = log_S;
            }
        }
    }

    return CVVDP_OK;
}

void cvvdp_csf_destroy(cvvdp_csf_t* csf) {
    if (csf->log_S_LUT) {
        free(csf->log_S_LUT);
        csf->log_S_LUT = NULL;
    }
}

float cvvdp_csf_sensitivity(const cvvdp_csf_t* csf, float L_bkg_val, int band, int channel) {
    float log_L = log10f(L_bkg_val);

    /* Find luminance index - matches GPU computeSensitivityGPU logic */
    float frac = 31.0f * (log_L - csf->log_L_bkg[0]) / (csf->log_L_bkg[31] - csf->log_L_bkg[0]);
    int imin = (int)frac;
    if (imin < 0) imin = 0;
    if (imin > 30) imin = 30;
    int imax = imin + 1;
    if (imax > 31) imax = 31;
    frac = frac - (float)imin;

    /* Interpolate - NOTE: GPU code has: LUTy[imin] * frac + LUTy[imax] * (1-frac)
     * This appears to be intentionally backwards in the original implementation.
     * We match the GPU behavior exactly for consistency. */
    const float* lut = csf->log_S_LUT + (band * 4 + channel) * CVVDP_LUT_SIZE;
    float log_S = lut[imin] * frac + lut[imax] * (1.0f - frac) + CVVDP_SENSITIVITY_CORRECTION / 20.0f;

    return powf(10.0f, log_S);
}

float cvvdp_to_jod(float quality) {
    if (quality > 0.1f) {
        return 10.0f - CVVDP_JOD_A * powf(quality, CVVDP_JOD_EXP);
    } else {
        float jod_a_p = CVVDP_JOD_A * powf(0.1f, CVVDP_JOD_EXP - 1.0f);
        return 10.0f - jod_a_p * quality;
    }
}

float cvvdp_compute_norm(const float* data, int count, int power) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        double val = (double)data[i];
        sum += pow(fabs(val), power);
    }
    return (float)pow(sum / count, 1.0 / power);
}

static cvvdp_error_t cvvdp_process_pyramid(cvvdp_context_t* ctx,
                                           float* ref_channels[4],
                                           float* dis_channels[4],
                                           double* out_score) {
    int w = ctx->width;
    int h = ctx->height;

    int num_levels = ctx->num_bands;

    /* Allocate pyramid storage */
    size_t total_size = 0;
    int widths[CVVDP_MAX_LEVELS], heights[CVVDP_MAX_LEVELS];
    int tw = w, th = h;
    for (int lev = 0; lev < num_levels; lev++) {
        widths[lev] = tw;
        heights[lev] = th;
        total_size += (size_t)tw * th;
        tw = (tw + 1) / 2;
        th = (th + 1) / 2;
    }

    /* Allocate work buffers for pyramid bands */
    float* ref_pyr[CVVDP_MAX_LEVELS][4];
    float* dis_pyr[CVVDP_MAX_LEVELS][4];
    float* L_bkg_pyr[CVVDP_MAX_LEVELS];
    float* temp = cvvdp_alloc_float((size_t)w * h);
    float* reduced = cvvdp_alloc_float(total_size);
    float* expanded = cvvdp_alloc_float((size_t)w * h);

    if (!temp || !reduced || !expanded) {
        free(temp);
        free(reduced);
        free(expanded);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    /* Allocate pyramid levels */
    for (int lev = 0; lev < num_levels; lev++) {
        size_t lev_size = (size_t)widths[lev] * heights[lev];
        for (int ch = 0; ch < 4; ch++) {
            ref_pyr[lev][ch] = cvvdp_alloc_float(lev_size);
            dis_pyr[lev][ch] = cvvdp_alloc_float(lev_size);
            if (!ref_pyr[lev][ch] || !dis_pyr[lev][ch]) {
                /* Cleanup on failure */
                for (int l = 0; l <= lev; l++) {
                    for (int c = 0; c < 4; c++) {
                        free(ref_pyr[l][c]);
                        free(dis_pyr[l][c]);
                    }
                    free(L_bkg_pyr[l]);
                }
                free(temp);
                free(reduced);
                free(expanded);
                return CVVDP_ERROR_OUT_OF_MEMORY;
            }
        }
        L_bkg_pyr[lev] = cvvdp_alloc_float(lev_size);
        if (!L_bkg_pyr[lev]) {
            for (int l = 0; l <= lev; l++) {
                for (int c = 0; c < 4; c++) {
                    free(ref_pyr[l][c]);
                    free(dis_pyr[l][c]);
                }
                free(L_bkg_pyr[l]);
            }
            free(temp);
            free(reduced);
            free(expanded);
            return CVVDP_ERROR_OUT_OF_MEMORY;
        }
    }

    /* Build Laplacian pyramid for each channel */
    for (int ch = 0; ch < 4; ch++) {
        /* Reference */
        memcpy(temp, ref_channels[ch], (size_t)w * h * sizeof(float));

        int cw = w, ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            size_t lev_size = (size_t)widths[lev] * heights[lev];

            if (lev < num_levels - 1) {
                /* Reduce */
                cvvdp_gauss_pyr_reduce(temp, reduced, cw, ch_h);

                /* Expand and compute L_bkg (only from Y channel) */
                if (ch == 0) {
                    cvvdp_gauss_pyr_expand(reduced, expanded, cw, ch_h);
                    for (size_t i = 0; i < lev_size; i++) {
                        L_bkg_pyr[lev][i] = cvvdp_maxf(0.01f, expanded[i]);
                    }
                }

                /* Laplacian = current - expanded */
                cvvdp_gauss_pyr_expand(reduced, expanded, cw, ch_h);
                for (size_t i = 0; i < lev_size; i++) {
                    float contrast = (temp[i] - expanded[i]) / cvvdp_maxf(0.01f, L_bkg_pyr[lev][i]);
                    ref_pyr[lev][ch][i] = contrast * (lev == 0 ? 1.0f : 2.0f);
                }

                /* Prepare for next level */
                memcpy(temp, reduced, (size_t)((cw+1)/2) * ((ch_h+1)/2) * sizeof(float));
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else {
                /* Baseband: compute mean for L_bkg */
                if (ch == 0) {
                    float mean = 0.0f;
                    for (size_t i = 0; i < lev_size; i++) {
                        mean += temp[i];
                    }
                    mean /= lev_size;
                    /* For baseband, store mean in first element (GPU uses single mean value) */
                    L_bkg_pyr[lev][0] = cvvdp_maxf(0.01f, mean);
                }

                /* Baseband contrast - divide by mean L_bkg */
                for (size_t i = 0; i < lev_size; i++) {
                    ref_pyr[lev][ch][i] = temp[i] / cvvdp_maxf(0.01f, L_bkg_pyr[lev][0]);
                }
            }
        }

        /* Distorted - same process */
        memcpy(temp, dis_channels[ch], (size_t)w * h * sizeof(float));

        cw = w; ch_h = h;
        for (int lev = 0; lev < num_levels; lev++) {
            size_t lev_size = (size_t)widths[lev] * heights[lev];

            if (lev < num_levels - 1) {
                cvvdp_gauss_pyr_reduce(temp, reduced, cw, ch_h);
                cvvdp_gauss_pyr_expand(reduced, expanded, cw, ch_h);

                for (size_t i = 0; i < lev_size; i++) {
                    float contrast = (temp[i] - expanded[i]) / cvvdp_maxf(0.01f, L_bkg_pyr[lev][i]);
                    dis_pyr[lev][ch][i] = contrast * (lev == 0 ? 1.0f : 2.0f);
                }

                memcpy(temp, reduced, (size_t)((cw+1)/2) * ((ch_h+1)/2) * sizeof(float));
                cw = (cw + 1) / 2;
                ch_h = (ch_h + 1) / 2;
            } else {
                /* Baseband contrast - divide by mean L_bkg */
                for (size_t i = 0; i < lev_size; i++) {
                    dis_pyr[lev][ch][i] = temp[i] / cvvdp_maxf(0.01f, L_bkg_pyr[lev][0]);
                }
            }
        }
    }

    /* Apply CSF and compute distortion */
    double total_score = 0.0;

    /* Precompute Gaussian kernel for masking blur (sigma = pu_dilate = 3) */
    const float blur_sigma = 3.0f;
    const int blur_radius = 8;  /* GAUSSIANSIZE from GPU code */
    float blur_kernel[17];
    float blur_kernel_sum = 0.0f;
    for (int i = 0; i < 17; i++) {
        float d = (float)(i - 8);
        blur_kernel[i] = expf(-d * d / (2.0f * blur_sigma * blur_sigma));
        blur_kernel_sum += blur_kernel[i];
    }
    for (int i = 0; i < 17; i++) {
        blur_kernel[i] /= blur_kernel_sum;
    }

    for (int lev = 0; lev < num_levels; lev++) {
        size_t lev_size = (size_t)widths[lev] * heights[lev];
        int lev_w = widths[lev];
        int lev_h = heights[lev];
        int is_baseband = (lev == num_levels - 1);

        /* Channel gain - matches GPU preGaussianPreCompute_kernel */
        float ch_gain[4] = {1.0f, 1.45f, 1.0f, 1.0f};

        if (!is_baseband) {
            /* Non-baseband: Apply CSF, then masking, then compute D */
            for (int ch = 0; ch < 4; ch++) {
                /* Apply CSF */
                for (size_t i = 0; i < lev_size; i++) {
                    float S = cvvdp_csf_sensitivity(&ctx->csf, L_bkg_pyr[lev][i], lev, ch);
                    ref_pyr[lev][ch][i] *= S * ch_gain[ch];
                    dis_pyr[lev][ch][i] *= S * ch_gain[ch];
                }
            }

            /* Compute min(|ref|, |dis|) for each channel and apply Gaussian blur */
            float* min_abs[4];
            float* blurred_min_abs[4];
            for (int ch = 0; ch < 4; ch++) {
                min_abs[ch] = cvvdp_alloc_float(lev_size);
                blurred_min_abs[ch] = cvvdp_alloc_float(lev_size);
                if (!min_abs[ch] || !blurred_min_abs[ch]) {
                    for (int c = 0; c <= ch; c++) {
                        free(min_abs[c]);
                        free(blurred_min_abs[c]);
                    }
                    continue;
                }

                /* Compute min(|ref|, |dis|) */
                for (size_t i = 0; i < lev_size; i++) {
                    float r = fabsf(ref_pyr[lev][ch][i]);
                    float d = fabsf(dis_pyr[lev][ch][i]);
                    min_abs[ch][i] = cvvdp_minf(r, d);
                }

                /* Apply separable Gaussian blur - horizontal pass */
                float* temp_blur = cvvdp_alloc_float(lev_size);
                if (temp_blur) {
                    for (int y = 0; y < lev_h; y++) {
                        for (int x = 0; x < lev_w; x++) {
                            float sum = 0.0f;
                            float wsum = 0.0f;
                            for (int k = -blur_radius; k <= blur_radius; k++) {
                                int sx = x + k;
                                if (sx >= 0 && sx < lev_w) {
                                    sum += min_abs[ch][y * lev_w + sx] * blur_kernel[k + blur_radius];
                                    wsum += blur_kernel[k + blur_radius];
                                }
                            }
                            temp_blur[y * lev_w + x] = sum / wsum;
                        }
                    }

                    /* Vertical pass */
                    for (int y = 0; y < lev_h; y++) {
                        for (int x = 0; x < lev_w; x++) {
                            float sum = 0.0f;
                            float wsum = 0.0f;
                            for (int k = -blur_radius; k <= blur_radius; k++) {
                                int sy = y + k;
                                if (sy >= 0 && sy < lev_h) {
                                    sum += temp_blur[sy * lev_w + x] * blur_kernel[k + blur_radius];
                                    wsum += blur_kernel[k + blur_radius];
                                }
                            }
                            blurred_min_abs[ch][y * lev_w + x] = sum / wsum;
                        }
                    }
                    free(temp_blur);
                } else {
                    memcpy(blurred_min_abs[ch], min_abs[ch], lev_size * sizeof(float));
                }
            }

            /* Compute distortion with masking - matches GPU computeD_Kernel */
            float* D = cvvdp_alloc_float(lev_size * 4);
            if (!D) {
                for (int ch = 0; ch < 4; ch++) {
                    free(min_abs[ch]);
                    free(blurred_min_abs[ch]);
                }
                continue;
            }

            float max_v = powf(10.0f, CVVDP_D_MAX);
            float pow_mask_c = powf(10.0f, CVVDP_MASK_C);

            for (size_t i = 0; i < lev_size; i++) {
                /* Compute Cm values from blurred min(|ref|, |dis|) */
                float Cm[4];
                for (int ch = 0; ch < 4; ch++) {
                    Cm[ch] = powf(pow_mask_c * blurred_min_abs[ch][i], CVVDP_MASK_Q[ch]);
                }

                /* Cross-channel masking - matches GPU computeD_Kernel Cmask computation */
                for (int ch = 0; ch < 4; ch++) {
                    float mask = Cm[0] * powf(2.0f, CVVDP_XCM_WEIGHTS[0 + ch]) +
                                 Cm[1] * powf(2.0f, CVVDP_XCM_WEIGHTS[4 + ch]) +
                                 Cm[2] * powf(2.0f, CVVDP_XCM_WEIGHTS[8 + ch]) +
                                 Cm[3] * powf(2.0f, CVVDP_XCM_WEIGHTS[12 + ch]);

                    float diff = fabsf(ref_pyr[lev][ch][i] - dis_pyr[lev][ch][i]);
                    float Du = powf(diff, CVVDP_MASK_P) / (1.0f + mask);
                    float D_val = max_v * Du / (max_v + Du);

                    /* Channel weights - matches GPU computeD_Kernel */
                    float weight = 1.0f;
                    if (ch == 1 || ch == 2) weight = CVVDP_CH_CHROM_W;
                    if (ch == 3) weight = CVVDP_CH_TRANS_W;

                    D[ch * lev_size + i] = D_val * weight;
                }
            }

            /* Free temporary buffers */
            for (int ch = 0; ch < 4; ch++) {
                free(min_abs[ch]);
                free(blurred_min_abs[ch]);
            }

            /* Pool distortion values - 2-norm then power-4 */
            for (int ch = 0; ch < 4; ch++) {
                float norm = cvvdp_compute_norm(D + ch * lev_size, (int)lev_size, 2);
                double contrib = pow(norm, 4);
                total_score += contrib;
            }

            free(D);
        } else {
            /* Baseband: Different processing - no masking, just abs diff * S * baseband_weight
             * Matches GPU computeD_baseband_kernel */
            float* D = cvvdp_alloc_float(lev_size * 4);
            if (!D) continue;

            /* Use mean L_bkg for baseband (stored in first element) */
            float L_bkg_mean = L_bkg_pyr[lev][0];

            for (int ch = 0; ch < 4; ch++) {
                float S = cvvdp_csf_sensitivity(&ctx->csf, L_bkg_mean, lev, ch);

                for (size_t i = 0; i < lev_size; i++) {
                    /* Baseband: simple abs difference * sensitivity * baseband_weight */
                    float diff = fabsf(ref_pyr[lev][ch][i] - dis_pyr[lev][ch][i]);
                    D[ch * lev_size + i] = diff * S * CVVDP_BASEBAND_WEIGHT[ch];
                }
            }

            /* Pool distortion values - 2-norm then power-4 */
            for (int ch = 0; ch < 4; ch++) {
                float norm = cvvdp_compute_norm(D + ch * lev_size, (int)lev_size, 2);
                double contrib = pow(norm, 4);
                total_score += contrib;
            }

            free(D);
        }
    }

    /* Final score */
    *out_score = pow(total_score, 0.25);

    /* Cleanup */
    for (int lev = 0; lev < num_levels; lev++) {
        for (int ch = 0; ch < 4; ch++) {
            free(ref_pyr[lev][ch]);
            free(dis_pyr[lev][ch]);
        }
        free(L_bkg_pyr[lev]);
    }
    free(temp);
    free(reduced);
    free(expanded);

    return CVVDP_OK;
}

cvvdp_error_t cvvdp_create(int width, int height, float fps,
                           cvvdp_display_model_t display_model,
                           const cvvdp_display_params_t* custom_params,
                           cvvdp_context_t** out_ctx) {
    if (!out_ctx) {
        return CVVDP_ERROR_NULL_POINTER;
    }

    if (width <= 0 || height <= 0) {
        return CVVDP_ERROR_INVALID_DIMENSIONS;
    }

    cvvdp_context_t* ctx = (cvvdp_context_t*)calloc(1, sizeof(cvvdp_context_t));
    if (!ctx) {
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    ctx->width = width;
    ctx->height = height;
    ctx->fps = fps > 0 ? fps : 0;
    ctx->num_frames = 0;
    ctx->score_square_sum = 0.0;

    /* Initialize display model */
    cvvdp_init_display(&ctx->display, display_model, custom_params);

    /* Initialize Gaussian filter */
    cvvdp_gaussian_init(&ctx->gaussian);

    /* Get band frequencies */
    ctx->band_frequencies = cvvdp_alloc_float(CVVDP_MAX_LEVELS);
    if (!ctx->band_frequencies) {
        free(ctx);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }
    ctx->num_bands = cvvdp_get_band_frequencies(width, height, ctx->display.ppd,
                                                 ctx->band_frequencies);

    /* Initialize CSF */
    cvvdp_error_t err = cvvdp_csf_init(&ctx->csf, width, height, ctx->display.ppd);
    if (err != CVVDP_OK) {
        free(ctx->band_frequencies);
        free(ctx);
        return err;
    }

    /* Initialize temporal rings */
    err = cvvdp_temporal_ring_init(&ctx->ring_ref, width, height, fps);
    if (err != CVVDP_OK) {
        cvvdp_csf_destroy(&ctx->csf);
        free(ctx->band_frequencies);
        free(ctx);
        return err;
    }

    err = cvvdp_temporal_ring_init(&ctx->ring_dis, width, height, fps);
    if (err != CVVDP_OK) {
        cvvdp_temporal_ring_destroy(&ctx->ring_ref);
        cvvdp_csf_destroy(&ctx->csf);
        free(ctx->band_frequencies);
        free(ctx);
        return err;
    }

    /* Allocate work buffers */
    size_t plane_size = (size_t)width * height;
    ctx->Y_sustained = cvvdp_alloc_float(plane_size);
    ctx->RG_sustained = cvvdp_alloc_float(plane_size);
    ctx->YV_sustained = cvvdp_alloc_float(plane_size);
    ctx->Y_transient = cvvdp_alloc_float(plane_size);
    ctx->work_buffer = cvvdp_alloc_float(plane_size * 3);

    if (!ctx->Y_sustained || !ctx->RG_sustained || !ctx->YV_sustained ||
        !ctx->Y_transient || !ctx->work_buffer) {
        cvvdp_destroy(ctx);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    *out_ctx = ctx;
    return CVVDP_OK;
}

void cvvdp_destroy(cvvdp_context_t* ctx) {
    if (!ctx) return;

    cvvdp_temporal_ring_destroy(&ctx->ring_ref);
    cvvdp_temporal_ring_destroy(&ctx->ring_dis);
    cvvdp_csf_destroy(&ctx->csf);

    free(ctx->band_frequencies);
    free(ctx->Y_sustained);
    free(ctx->RG_sustained);
    free(ctx->YV_sustained);
    free(ctx->Y_transient);
    free(ctx->work_buffer);

    free(ctx);
}

cvvdp_error_t cvvdp_reset(cvvdp_context_t* ctx) {
    if (!ctx) {
        return CVVDP_ERROR_NULL_POINTER;
    }

    ctx->num_frames = 0;
    ctx->score_square_sum = 0.0;
    cvvdp_temporal_ring_reset(&ctx->ring_ref);
    cvvdp_temporal_ring_reset(&ctx->ring_dis);

    return CVVDP_OK;
}

cvvdp_error_t cvvdp_process_frame(cvvdp_context_t* ctx,
                                  const cvvdp_image_t* reference,
                                  const cvvdp_image_t* distorted,
                                  cvvdp_result_t* result) {
    if (!ctx || !reference || !distorted || !result) {
        return CVVDP_ERROR_NULL_POINTER;
    }

    if (reference->width != ctx->width || reference->height != ctx->height ||
        distorted->width != ctx->width || distorted->height != ctx->height) {
        return CVVDP_ERROR_DIMENSION_MISMATCH;
    }

    size_t plane_size = (size_t)ctx->width * ctx->height;

    /* Allocate temporary planes */
    float* ref_planes[3] = {
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size)
    };
    float* dis_planes[3] = {
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size),
        cvvdp_alloc_float(plane_size)
    };

    if (!ref_planes[0] || !ref_planes[1] || !ref_planes[2] ||
        !dis_planes[0] || !dis_planes[1] || !dis_planes[2]) {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dis_planes[i]);
        }
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    /* Load images */
    cvvdp_error_t err = cvvdp_load_image(reference, ref_planes);
    if (err != CVVDP_OK) {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dis_planes[i]);
        }
        return err;
    }

    err = cvvdp_load_image(distorted, dis_planes);
    if (err != CVVDP_OK) {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dis_planes[i]);
        }
        return err;
    }

    /* Apply display model to Y channel */
    int is_hdr = ctx->display.is_hdr;
    for (int i = 0; i < 3; i++) {
        /* For simplicity, apply to all planes (should be luminance only) */
        cvvdp_apply_display_model(ref_planes[i], (int)plane_size, &ctx->display, is_hdr);
        cvvdp_apply_display_model(dis_planes[i], (int)plane_size, &ctx->display, is_hdr);
    }

    /* Convert to XYZ then DKL */
    cvvdp_rgb_to_xyz(ref_planes[0], ref_planes[1], ref_planes[2], (int)plane_size);
    cvvdp_rgb_to_xyz(dis_planes[0], dis_planes[1], dis_planes[2], (int)plane_size);
    cvvdp_xyz_to_dkl(ref_planes[0], ref_planes[1], ref_planes[2], (int)plane_size);
    cvvdp_xyz_to_dkl(dis_planes[0], dis_planes[1], dis_planes[2], (int)plane_size);

    /* Store in temporal ring (interleaved) */
    float* ref_frame = cvvdp_alloc_float(plane_size * 3);
    float* dis_frame = cvvdp_alloc_float(plane_size * 3);
    if (!ref_frame || !dis_frame) {
        for (int i = 0; i < 3; i++) {
            free(ref_planes[i]);
            free(dis_planes[i]);
        }
        free(ref_frame);
        free(dis_frame);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    memcpy(ref_frame, ref_planes[0], plane_size * sizeof(float));
    memcpy(ref_frame + plane_size, ref_planes[1], plane_size * sizeof(float));
    memcpy(ref_frame + 2 * plane_size, ref_planes[2], plane_size * sizeof(float));
    memcpy(dis_frame, dis_planes[0], plane_size * sizeof(float));
    memcpy(dis_frame + plane_size, dis_planes[1], plane_size * sizeof(float));
    memcpy(dis_frame + 2 * plane_size, dis_planes[2], plane_size * sizeof(float));

    cvvdp_temporal_ring_push(&ctx->ring_ref, ref_frame);
    cvvdp_temporal_ring_push(&ctx->ring_dis, dis_frame);

    free(ref_frame);
    free(dis_frame);
    for (int i = 0; i < 3; i++) {
        free(ref_planes[i]);
        free(dis_planes[i]);
    }

    /* Compute temporal channels */
    float* ref_Y_sus = cvvdp_alloc_float(plane_size);
    float* ref_RG_sus = cvvdp_alloc_float(plane_size);
    float* ref_YV_sus = cvvdp_alloc_float(plane_size);
    float* ref_Y_trans = cvvdp_alloc_float(plane_size);
    float* dis_Y_sus = cvvdp_alloc_float(plane_size);
    float* dis_RG_sus = cvvdp_alloc_float(plane_size);
    float* dis_YV_sus = cvvdp_alloc_float(plane_size);
    float* dis_Y_trans = cvvdp_alloc_float(plane_size);

    if (!ref_Y_sus || !ref_RG_sus || !ref_YV_sus || !ref_Y_trans ||
        !dis_Y_sus || !dis_RG_sus || !dis_YV_sus || !dis_Y_trans) {
        free(ref_Y_sus); free(ref_RG_sus); free(ref_YV_sus); free(ref_Y_trans);
        free(dis_Y_sus); free(dis_RG_sus); free(dis_YV_sus); free(dis_Y_trans);
        return CVVDP_ERROR_OUT_OF_MEMORY;
    }

    cvvdp_compute_temporal_channels(&ctx->ring_ref, ref_Y_sus, ref_RG_sus, ref_YV_sus, ref_Y_trans);
    cvvdp_compute_temporal_channels(&ctx->ring_dis, dis_Y_sus, dis_RG_sus, dis_YV_sus, dis_Y_trans);

    /* Process pyramid and compute score */
    float* ref_channels[4] = {ref_Y_sus, ref_RG_sus, ref_YV_sus, ref_Y_trans};
    float* dis_channels[4] = {dis_Y_sus, dis_RG_sus, dis_YV_sus, dis_Y_trans};

    double current_score;
    err = cvvdp_process_pyramid(ctx, ref_channels, dis_channels, &current_score);

    free(ref_Y_sus); free(ref_RG_sus); free(ref_YV_sus); free(ref_Y_trans);
    free(dis_Y_sus); free(dis_RG_sus); free(dis_YV_sus); free(dis_Y_trans);

    if (err != CVVDP_OK) {
        return err;
    }

    /* Accumulate score over frames */
    ctx->num_frames++;
    ctx->score_square_sum += pow(current_score, CVVDP_BETA_T);

    /* For single image comparison, apply image integration factor */
    double resQ;
    if (ctx->num_frames == 1) {
        /* Single frame: apply image_int scaling factor */
        resQ = current_score * CVVDP_IMAGE_INT;
    } else {
        /* Multi-frame: temporal pooling */
        resQ = pow(ctx->score_square_sum / (double)ctx->num_frames, 1.0 / CVVDP_BETA_T);
    }

    result->quality = resQ;
    result->jod = cvvdp_to_jod((float)resQ);

    return CVVDP_OK;
}

cvvdp_error_t cvvdp_compare_images(const cvvdp_image_t* reference,
                                   const cvvdp_image_t* distorted,
                                   cvvdp_display_model_t display_model,
                                   const cvvdp_display_params_t* custom_params,
                                   cvvdp_result_t* result) {
    if (!reference || !distorted || !result)
        return CVVDP_ERROR_NULL_POINTER;

    if (reference->width != distorted->width || reference->height != distorted->height)
        return CVVDP_ERROR_DIMENSION_MISMATCH;

    cvvdp_context_t* ctx;
    cvvdp_error_t err = cvvdp_create(reference->width, reference->height, 0,
                                      display_model, custom_params, &ctx);
    if (err != CVVDP_OK)
        return err;

    err = cvvdp_process_frame(ctx, reference, distorted, result);
    cvvdp_destroy(ctx);

    return err;
}
