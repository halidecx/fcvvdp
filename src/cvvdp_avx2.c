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
#include "cvvdp_avx2.h"

const CvvdpSimdDispatch cvvdp_dispatch_avx2 = {
    .apply_display           = cvvdp_apply_display_impl,
    .rgb_to_xyz              = cvvdp_rgb_to_xyz_impl,
    .xyz_to_dkl              = cvvdp_xyz_to_dkl_impl,
    .compute_temporal_channels = cvvdp_compute_temporal_channels_impl,
    .gauss_pyr_reduce        = cvvdp_gauss_pyr_reduce_impl,
    .gauss_pyr_expand        = cvvdp_gauss_pyr_expand_impl,
    .contrast                = cvvdp_contrast_impl,
    .luma_contrast           = cvvdp_luma_contrast_impl,
    .normalize               = cvvdp_normalize_impl,
    .min_abs                 = cvvdp_min_abs_impl,
    .blur_horizontal         = cvvdp_blur_horizontal_impl,
    .blur_vertical           = cvvdp_blur_vertical_impl,
    .baseband_diff           = cvvdp_baseband_diff_impl,
};
