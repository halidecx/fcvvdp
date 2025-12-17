# fcvvdp

A fast C implementation of the [CVVDP](https://github.com/gfxdisp/colorvideovdp)
metric ([arXiv](https://arxiv.org/html/2401.11485)) from the University of
Cambridge.

Special thanks to [Vship](https://github.com/Line-fr/Vship/releases), from which
this implementation was derived. Vship is under the
[MIT license](https://github.com/Line-fr/Vship#MIT-1-ov-file).

## Benchmarks

_correlation & speed – soon_

## Metric Process

CVVDP is a full-reference video & image metric meant to simulate the human
visual system to predict the perceived difference between two sources.

Visualization:

![CVVDP](https://mermaid.ink/svg/pako:eNp9VNty2jAQ_ZUdPXSaGZIJDiSBh84EiCkJJATTS2LyoNoLaGpLrizT0JB_71q-5NILLyBz9uw5Z1d-ZIEKkXXZMlI_gzXXBuaDhQT6pNm3lebJGkZSGMEj8YsboWTx55k_EGkS8S14CQbpPezvf9j1eRRkETeY7qDnT6cDeAc9LkNwNf7IUAYC0_uivpcXQN_PuaHvuYScY5wozSNwRWRQV0iU4UK-UTRFve9qHiNMtQowTYVcFZABMSaZgRkuiZI0GhjFfIWFQjj3hzyOOQwwt10grIsJHaOy47k1MxYSuYaxWK3NDlx_NuyBUfD19i7_GlyOS7RboBswGzbg9jP011xKjCiCoV9bmpFA6GXLJeqy7I2jGvnW0NDSu6NZGcsOPvpelhpO8sK6GdxW_e__VTby55rLVKA0VRnc_idjL-H51P8Q9JFSG9kwLyjMjP7hEqZbmoYIba5kJBX5ppTkFxZ86dfbAWMVEPEX_IYa-koazVNTgi8teOyfJUm0tYvhoczpNsJsqSSfBkn5j-4JT7_naeezzfOmtcMCMrbcEyvkwG7IJi1W5Bl5_xJ6VUIrynOJerUtIVc24TqBXpRRyNd-FZuXaOQhqGVVXZZdFzdFqzTdr8YwEQ8E2MG0tF31czMZvAhyQp6mVtiNn0No_s_CqSh5mcpfdmyqVFSP8aZY2xZcKR3vYFYLL1El18zCvCwGtaFpjXFjV9vzi-vnBUpXmXkWOhf0eCQNUs9c-w7mz9fgNffcevlEXhK4yegFQxOmq3VxPXjFW9t5Me1-sSbF4ZM9fH7_3hWSmlRUlmNvjzXYSouQdY3OsMFi1DHPj-wxr14ws8YYF6xLP0Ouvy_YQj5RTcLlnVJxVaZVtlqz7pJHKZ2yJKQ9HghOJuP6Kc0hRN1XmTSs2-50HMvCuo_sgc6HzYOjpnN0eOy0O8edTrvVYFvWPekcHLVO2i2nedpuNZ1W86nBftm-hwenbcdpN0-PnY5z2Dk6bjYYhsIoPSne2PbF_fQbjzvS_g)

### Initialization and Display Modeling

Before processing pixels, the metric models the viewing environment through a
specified display's angular resolution (pixels per degree).

- Using specified display parameters (resolution, diagonal size, viewing
  distance), CVVDP calculates Pixels Per Degree (PPD). This determines how large
  a pixel appears to the eye.
- The Contrast Sensitivity Function (CSF) determines how sensitive the eye is to
  specific spatial frequencies. It maps the display's frequency bands (derived
  from PPD) to sensitivity values.

Available display model presets include:

| Model        | Description                                              |
| ------------ | -------------------------------------------------------- |
| `fhd`        | 24" FullHD monitor, 200 cd/m², office lighting (default) |
| `4k`         | 30" 4K monitor, 200 cd/m², office lighting               |
| `hdr_pq`     | 30" 4K HDR, 1500 cd/m², low light                        |
| `hdr_hlg`    | 30" 4K HDR HLG, 1500 cd/m², low light                    |
| `hdr_linear` | 30" 4K HDR linear, 1500 cd/m², low light                 |
| `hdr_dark`   | 30" 4K HDR, 1500 cd/m², dark room                        |
| `hdr_zoom`   | 30" 4K HDR, 10000 cd/m², close viewing                   |

### Input Loading and Display Mapping

- Input images (uint8, uint16, or float) are converted to linear float RGB. If
  the input is integer-based, sRGB gamma decoding (approx. 2.4 power) is
  applied.
- The linear RGB values are converted into absolute physical light units (nits)
  based on the display model.
- SDR: clips values between 0 and 1, scales by max luminance, adds black level &
  reflected ambient light.
- HDR: Performs tone mapping (PQ/HLG), clips to the display's peak luminance,
  adds black level and reflections.

### Color Space Conversion

1. Linear RGB is converted to the CIE XYZ color space
2. XYZ is transformed to DKL (Derrington-Krauskopf-Lennie), an opponent color
   space that models what is used by the human brain.
   - L: Luminance, a.k.a. achromatic brightness (L+M cones)
   - RG: Chromatic difference (L-M cones)
   - YV: S-cone opponent channel (S - (L+M))

### Temporal Decomposition

If the input is video (FPS > 0), the metric analyzes how pixel values change
over time. It maintains `TemporalRingBuf` to store previous frames.

- The code applies Finite Impulse Response (FIR) filters to the history of DKL
  frames (so, temporal filtering)
- Low temporal frequency information (static or slow-moving) is stored in the
  _sustained channels_. Calculated for Luminance (Y), Red-Green (RG), and
  Yellow-Violet (YV).
- High temporal frequency information (flicker or fast motion) is in the
  _transient channel_. Only calculated for luminance.

We now have 4 channels to process spatially: Y_{sus}, RG_{sus}, YV_{sus},
Y_{trans}.

### Spatial Decomposition

The visual system processes different sizes of features (frequencies)
independently. CVVDP implements a _Gaussian pyramid_ to simulate this.

1. The image is repeatedly downscaled (blurred and subsampled).
2. At each level in the pyramid, local contrast is computed.
3. The upscaled version of the next lower level is subtracted from from the
   current level (Gaussian difference)
4. This difference is normalized by the local background luminance (`L_BKG`) to
   get _Weber Contrast_.

### CSF Weighting & Difference Calculation

- For every pixel at every pyramid level, the contrast is multiplied by the CSF
  sensitivity. This scaling depends on:
  - Spatial Frequency, which is determined by the pyramid level
  - Background Luminance, where brighter areas generally have lower sensitivity
    to absolute differences
  - Channel, because the eye is less sensitive to chroma changes (RG/YV) than
    luma changes.
- The absolute difference between the Reference and Distorted contrast values is
  calculated.

### Visual Masking

This is the most complex step. It is designed to account for the fact that
artifacts are harder to see in textured areas.

- The code computes the minimum activity between the reference and distorted
  signals
- A Gaussian blur is applied to this activity map to simulate the spatial extent
  of masking
- Activity in one channel can mask errors in another. The code computes a
  masking denominator using weighted sums of activity from all 4 channels
- The final difference d is compressed using a non-linear sigmoid-like function
  (not going to explain it here, probably best to read the code for more
  details)

### Pooling & Scoring

- The masked differences are aggregated across the image using a Minkowski norm
  (Power of 4), then averaged
- Scores from all pyramid levels and all four channels are summed
- Scores are accumulated over frames using a power sum
- The final raw quality metric (Q) is mapped to _JOD (Just Objectionable
  Difference)_, which is a more meaningful perceptual score. 10.0 is a perfect
  match (no visible difference), and lower scores mean the quality is worse

| Score      | Interpretation                     |
| ---------- | ---------------------------------- |
| 10.0       | Images are identical               |
| 9.0 - 10.0 | Barely visible difference          |
| 8.0 - 9.0  | Slight visible difference          |
| 7.0 - 8.0  | Noticeable but acceptable          |
| 5.0 - 7.0  | Clearly visible, somewhat annoying |
| 3.0 - 5.0  | Very visible, annoying difference  |
| < 3.0      | Large, unacceptable difference     |

## Credits

`fcvvdp` is under the [Apache 2.0 License](LICENSE). `fcvvdp` is developed by
[Halide Compression](https://halide.cx).
