#ifndef HI_CORE_H
#define HI_CORE_H

#include <stddef.h>
#include "hi_config.h"

enum {
    HI_RMS, HI_STD, HI_PEAK_TO_PEAK, HI_CREST_FACTOR, HI_KURTOSIS, HI_SKEWNESS,
    HI_BAND_0, HI_BAND_1, HI_BAND_2, HI_BAND_3, HI_FEATURE_COUNT
};

typedef struct { float value[HI_FEATURE_COUNT]; } hi_features_t;

/* All buffers are caller-owned. No dynamic allocation is used. */
int hi_compute_features(const float window[HI_WINDOW_SIZE],
                        float fft_real[HI_WINDOW_SIZE], float fft_imag[HI_WINDOW_SIZE],
                        hi_features_t *features);
float hi_zscore_mean(const hi_features_t *features,
                     const float mean[HI_FEATURE_COUNT], const float std[HI_FEATURE_COUNT]);
float hi_ema(float current, float previous, int *initialized);

#endif
