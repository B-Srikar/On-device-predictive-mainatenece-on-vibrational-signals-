#include "hi_core.h"
#include <math.h>

#define HI_PI 3.14159265358979323846f
#define HI_EPSILON 1.19209290e-7f

static void fft(float real[HI_WINDOW_SIZE], float imag[HI_WINDOW_SIZE]) {
    unsigned int i, j, bit, length, half, k;
    for (i = 1u, j = 0u; i < HI_WINDOW_SIZE; ++i) {
        bit = HI_WINDOW_SIZE >> 1u;
        while (j & bit) { j ^= bit; bit >>= 1u; }
        j ^= bit;
        if (i < j) {
            float temporary = real[i]; real[i] = real[j]; real[j] = temporary;
            temporary = imag[i]; imag[i] = imag[j]; imag[j] = temporary;
        }
    }
    for (length = 2u; length <= HI_WINDOW_SIZE; length <<= 1u) {
        half = length >> 1u;
        for (k = 0u; k < half; ++k) {
            float angle = -2.0f * HI_PI * (float)k / (float)length;
            float wr = cosf(angle), wi = sinf(angle);
            for (i = k; i < HI_WINDOW_SIZE; i += length) {
                j = i + half;
                float tr = wr * real[j] - wi * imag[j];
                float ti = wr * imag[j] + wi * real[j];
                real[j] = real[i] - tr; imag[j] = imag[i] - ti;
                real[i] += tr; imag[i] += ti;
            }
        }
    }
}

int hi_compute_features(const float window[HI_WINDOW_SIZE],
                        float real[HI_WINDOW_SIZE], float imag[HI_WINDOW_SIZE],
                        hi_features_t *out) {
    unsigned int i, band;
    float sum = 0.0f, sum_sq = 0.0f, minimum = window[0], maximum = window[0];
    float m2 = 0.0f, m3 = 0.0f, m4 = 0.0f, absolute_peak = 0.0f;
    for (i = 0u; i < HI_WINDOW_SIZE; ++i) {
        float x = window[i];
        if (!isfinite(x)) return -1;
        sum += x; sum_sq += x * x;
        if (x < minimum) minimum = x;
        if (x > maximum) maximum = x;
        if (fabsf(x) > absolute_peak) absolute_peak = fabsf(x);
        real[i] = x; imag[i] = 0.0f;
    }
    {
        float n = (float)HI_WINDOW_SIZE, mean = sum / n;
        out->value[HI_RMS] = sqrtf(sum_sq / n);
        for (i = 0u; i < HI_WINDOW_SIZE; ++i) {
            float d = window[i] - mean;
            m2 += d * d; m3 += d * d * d; m4 += d * d * d * d;
        }
        m2 /= n; m3 /= n; m4 /= n;
        out->value[HI_STD] = sqrtf(m2);
        out->value[HI_PEAK_TO_PEAK] = maximum - minimum;
        out->value[HI_CREST_FACTOR] = out->value[HI_RMS] > HI_EPSILON ? absolute_peak / out->value[HI_RMS] : 0.0f;
        if (m2 <= HI_EPSILON) {
            out->value[HI_KURTOSIS] = 0.0f; out->value[HI_SKEWNESS] = 0.0f;
        } else {
            float g1 = m3 / powf(m2, 1.5f);
            out->value[HI_SKEWNESS] = sqrtf(n * (n - 1.0f)) * g1 / (n - 2.0f);
            out->value[HI_KURTOSIS] = ((n * n - 1.0f) * m4 / (m2 * m2) - 3.0f * (n - 1.0f) * (n - 1.0f)) / ((n - 2.0f) * (n - 3.0f)) + 3.0f;
        }
    }
    fft(real, imag);
    for (band = 0u; band < HI_BAND_COUNT; ++band) out->value[HI_BAND_0 + band] = 0.0f;
    for (i = 0u; i <= HI_WINDOW_SIZE / 2u; ++i) {
        float frequency = (float)i * HI_SAMPLE_RATE_HZ / (float)HI_WINDOW_SIZE;
        float power = (real[i] * real[i] + imag[i] * imag[i]) / ((float)HI_WINDOW_SIZE * (float)HI_WINDOW_SIZE);
        for (band = 0u; band < HI_BAND_COUNT; ++band) {
            int final_band = band == HI_BAND_COUNT - 1u;
            if (frequency >= HI_BANDS[band].low_hz && (frequency < HI_BANDS[band].high_hz || (final_band && frequency <= HI_BANDS[band].high_hz)))
                out->value[HI_BAND_0 + band] += power;
        }
    }
    return 0;
}

float hi_zscore_mean(const hi_features_t *features,
                     const float mean[HI_FEATURE_COUNT], const float std[HI_FEATURE_COUNT]) {
    unsigned int i; float total = 0.0f;
    for (i = 0u; i < HI_FEATURE_COUNT; ++i) {
        float scale = fabsf(std[i]) > HI_EPSILON ? std[i] : 1.0f;
        total += (features->value[i] - mean[i]) / scale;
    }
    return total / (float)HI_FEATURE_COUNT;
}

float hi_ema(float current, float previous, int *initialized) {
    if (!*initialized) { *initialized = 1; return current; }
    return HI_EMA_ALPHA * current + (1.0f - HI_EMA_ALPHA) * previous;
}
