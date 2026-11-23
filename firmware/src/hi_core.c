#include "hi_core.h"
#ifdef HI_CONFIG_IMS
#include "hi_baseline_ims.h"
#else
#include "hi_baseline.h"
#endif
#include <math.h>
#include <string.h>

#define HI_PI 3.14159265358979323846f
#define HI_EPSILON 1.19209290e-7f
#define HI_BAND_FLOOR 1.0e-30f

static void fft(float real[HI_WINDOW_SIZE], float imag[HI_WINDOW_SIZE]) {
    unsigned int i, j, bit, length, half, k;
    for (i = 1u, j = 0u; i < HI_WINDOW_SIZE; ++i) {
        bit = HI_WINDOW_SIZE >> 1u;
        while (j & bit) {
            j ^= bit;
            bit >>= 1u;
        }
        j ^= bit;
        if (i < j) {
            float temporary = real[i];
            real[i] = real[j];
            real[j] = temporary;
            temporary = imag[i];
            imag[i] = imag[j];
            imag[j] = temporary;
        }
    }
    for (length = 2u; length <= HI_WINDOW_SIZE; length <<= 1u) {
        half = length >> 1u;
        for (k = 0u; k < half; ++k) {
            float angle = -2.0f * HI_PI * (float)k / (float)length;
            float wr = cosf(angle);
            float wi = sinf(angle);
            for (i = k; i < HI_WINDOW_SIZE; i += length) {
                float tr, ti;
                j = i + half;
                tr = wr * real[j] - wi * imag[j];
                ti = wr * imag[j] + wi * real[j];
                real[j] = real[i] - tr;
                imag[j] = imag[i] - ti;
                real[i] += tr;
                imag[i] += ti;
            }
        }
    }
}

int hi_compute_features(const float window[HI_WINDOW_SIZE],
                        float fft_real[HI_WINDOW_SIZE],
                        float fft_imag[HI_WINDOW_SIZE],
                        hi_features_t *features) {
    unsigned int i, band;
    float sum = 0.0f;
    float n = (float)HI_WINDOW_SIZE;
    float inv_n2 = 1.0f / (n * n);
    float mean;
    float sum_sq = 0.0f;
    float m4 = 0.0f;
    float minimum = 0.0f;
    float maximum = 0.0f;
    float absolute_peak = 0.0f;
    float band_energy[HI_BAND_COUNT];

    /* Pass 1: validate finiteness and compute window mean. */
    for (i = 0u; i < HI_WINDOW_SIZE; ++i) {
        float x = window[i];
        if (!isfinite(x)) {
            return -1;
        }
        sum += x;
    }
    mean = sum / n;

    /* Pass 2: remove mean and compute time-domain moments on mean-centered window. */
    for (i = 0u; i < HI_WINDOW_SIZE; ++i) {
        float d = window[i] - mean;
        float d2 = d * d;
        float abs_d = fabsf(d);
        if (i == 0u) {
            minimum = d;
            maximum = d;
        } else {
            if (d < minimum) {
                minimum = d;
            }
            if (d > maximum) {
                maximum = d;
            }
        }
        if (abs_d > absolute_peak) {
            absolute_peak = abs_d;
        }
        sum_sq += d2;
        m4 += d2 * d2;
        fft_real[i] = d;
        fft_imag[i] = 0.0f;
    }

    {
        float m2 = sum_sq / n;
        float m4_mean = m4 / n;
        float rms = sqrtf(m2);
        features->value[HI_RMS] = rms;
        features->value[HI_PEAK_TO_PEAK] = maximum - minimum;
        features->value[HI_CREST_FACTOR] = (rms > HI_EPSILON) ? (absolute_peak / rms) : 0.0f;
        if (m2 <= HI_EPSILON) {
            features->value[HI_KURTOSIS] = 0.0f;
        } else {
            /* Unbiased Pearson kurtosis: matches scipy.stats.kurtosis(fisher=False, bias=False). */
            float kurt = ((n * n - 1.0f) * (m4_mean / (m2 * m2)) - 3.0f * (n - 1.0f) * (n - 1.0f)) /
                         ((n - 2.0f) * (n - 3.0f)) + 3.0f;
            features->value[HI_KURTOSIS] = kurt;
        }
    }

    fft(fft_real, fft_imag);

    for (band = 0u; band < HI_BAND_COUNT; ++band) {
        band_energy[band] = 0.0f;
    }
    for (i = 0u; i <= HI_WINDOW_SIZE / 2u; ++i) {
        float frequency = (float)i * HI_SAMPLE_RATE_HZ / n;
        float power = (fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i]) * inv_n2;
        for (band = 0u; band < HI_BAND_COUNT; ++band) {
            int final_band = (band == (HI_BAND_COUNT - 1u));
            if (frequency >= HI_BANDS[band].low_hz &&
                (frequency < HI_BANDS[band].high_hz || (final_band && frequency <= HI_BANDS[band].high_hz))) {
                band_energy[band] += power;
            }
        }
    }
    for (band = 0u; band < HI_BAND_COUNT; ++band) {
        float clamped = (band_energy[band] > HI_BAND_FLOOR) ? band_energy[band] : HI_BAND_FLOOR;
        features->value[HI_LOG_BAND_0 + band] = logf(clamped);
    }
    return 0;
}

#ifdef HI_CONFIG_IMS
static void median_across_windows(const float win_scratch[HI_SNAPSHOT_WINDOWS][HI_FEATURE_COUNT],
                                  hi_features_t *out) {
    unsigned int f, w;
    for (f = 0u; f < HI_FEATURE_COUNT; ++f) {
        float col[HI_SNAPSHOT_WINDOWS];
        for (w = 0u; w < HI_SNAPSHOT_WINDOWS; ++w) {
            col[w] = win_scratch[w][f];
        }
        for (w = 1u; w < HI_SNAPSHOT_WINDOWS; ++w) {
            float key = col[w];
            unsigned int j = w;
            while (j > 0u && col[j - 1u] > key) {
                col[j] = col[j - 1u];
                --j;
            }
            col[j] = key;
        }
        out->value[f] = col[HI_SNAPSHOT_WINDOWS / 2u];
    }
}

int hi_compute_snapshot_features(const float snapshot[HI_SNAPSHOT_SAMPLES],
                                 float fft_real[HI_WINDOW_SIZE],
                                 float fft_imag[HI_WINDOW_SIZE],
                                 float win_scratch[HI_SNAPSHOT_WINDOWS][HI_FEATURE_COUNT],
                                 hi_features_t *out) {
    unsigned int w, f;
    hi_features_t tmp;

    for (w = 0u; w < HI_SNAPSHOT_WINDOWS; ++w) {
        unsigned int offset = w * HI_SNAPSHOT_HOP;
        if (hi_compute_features(&snapshot[offset], fft_real, fft_imag, &tmp) != 0) {
            return -1;
        }
        for (f = 0u; f < HI_FEATURE_COUNT; ++f) {
            win_scratch[w][f] = tmp.value[f];
        }
    }
    median_across_windows(win_scratch, out);
    return 0;
}
#endif

float hi_compute_score(const hi_features_t *features,
                       const float mean[HI_FEATURE_COUNT],
                       const float std[HI_FEATURE_COUNT]) {
    unsigned int i;
    float sum_sq = 0.0f;
    for (i = 0u; i < HI_FEATURE_COUNT; ++i) {
        float scale = (std[i] > HI_SIGMA_FLOOR) ? std[i] : HI_SIGMA_FLOOR;
        float z = (features->value[i] - mean[i]) / scale;
        float c;
        if (z < 0.0f) {
            z = 0.0f;
        }
        c = log1pf(z);
        sum_sq += c * c;
    }
    return sqrtf(sum_sq / (float)HI_FEATURE_COUNT);
}

void hi_monitor_init(hi_monitor_t *mon) {
    unsigned int i;
    mon->ema_initialized = 0;
    mon->ema_hi = 0.0f;
    for (i = 0u; i < HI_SLOPE_WINDOW; ++i) {
        mon->ema_ring[i] = 0.0f;
    }
    mon->ring_pos = 0u;
    mon->ring_count = 0u;
    mon->fixed_consec = 0u;
    mon->slope_consec = 0u;
}

void hi_monitor_update(hi_monitor_t *mon,
                       const hi_monitor_config_t *cfg,
                       const hi_features_t *features,
                       hi_monitor_output_t *out) {
    float rms_gate = cfg->machine_off_rms_ratio * cfg->baseline_median_rms;

    if (features->value[HI_RMS] < rms_gate) {
        out->gated = 1;
        out->raw_hi = 0.0f;
        out->ema_hi = mon->ema_hi;
        out->slope_valid = 0;
        out->slope = 0.0f;
        out->fixed_alarm = 0;
        out->slope_alarm = 0;
        mon->fixed_consec = 0u;
        mon->slope_consec = 0u;
        return;
    }

    out->gated = 0;
    out->raw_hi = hi_compute_score(features, cfg->baseline_mean, cfg->baseline_std);
    if (!mon->ema_initialized) {
        mon->ema_initialized = 1;
        mon->ema_hi = out->raw_hi;
    } else {
        mon->ema_hi = cfg->ema_alpha * out->raw_hi + (1.0f - cfg->ema_alpha) * mon->ema_hi;
    }
    out->ema_hi = mon->ema_hi;

    mon->ema_ring[mon->ring_pos] = mon->ema_hi;
    mon->ring_pos = (mon->ring_pos + 1u) % HI_SLOPE_WINDOW;
    if (mon->ring_count < HI_SLOPE_WINDOW) {
        mon->ring_count += 1u;
    }

    if (mon->ring_count == HI_SLOPE_WINDOW) {
        /* OLS slope over W=12 chronological samples: x = 0..11, mean=5.5, sum((x-5.5)^2)=143.0. */
        const float x_mean = 0.5f * (float)(HI_SLOPE_WINDOW - 1u);
        const float denom = ((float)HI_SLOPE_WINDOW * ((float)HI_SLOPE_WINDOW * (float)HI_SLOPE_WINDOW - 1.0f)) / 12.0f;
        float y_sum = 0.0f;
        float y_mean;
        float num = 0.0f;
        unsigned int i;
        for (i = 0u; i < HI_SLOPE_WINDOW; ++i) {
            unsigned int idx = (mon->ring_pos + i) % HI_SLOPE_WINDOW;
            y_sum += mon->ema_ring[idx];
        }
        y_mean = y_sum / (float)HI_SLOPE_WINDOW;
        for (i = 0u; i < HI_SLOPE_WINDOW; ++i) {
            unsigned int idx = (mon->ring_pos + i) % HI_SLOPE_WINDOW;
            num += ((float)i - x_mean) * (mon->ema_ring[idx] - y_mean);
        }
        out->slope_valid = 1;
        out->slope = num / denom;
    } else {
        out->slope_valid = 0;
        out->slope = 0.0f;
    }

    if (out->ema_hi > cfg->hi_threshold) {
        if (mon->fixed_consec < HI_CONSEC_ALARMS) {
            mon->fixed_consec += 1u;
        }
    } else {
        mon->fixed_consec = 0u;
    }

    if (out->slope_valid && (out->slope > cfg->slope_threshold) && (out->ema_hi > cfg->hi_baseline_mean)) {
        if (mon->slope_consec < HI_CONSEC_ALARMS) {
            mon->slope_consec += 1u;
        }
    } else {
        mon->slope_consec = 0u;
    }

    out->fixed_alarm = (mon->fixed_consec >= HI_CONSEC_ALARMS) ? 1 : 0;
    out->slope_alarm = (mon->slope_consec >= HI_CONSEC_ALARMS) ? 1 : 0;
}

size_t hi_stream_sizeof(void) {
    return sizeof(hi_stream_t);
}

void hi_stream_init(hi_stream_t *st, const hi_monitor_config_t *cfg) {
    memset(st, 0, sizeof(*st));
    st->cfg = *cfg;
    hi_monitor_init(&st->mon);
}

void hi_stream_init_default(hi_stream_t *st) {
    hi_monitor_config_t cfg;
    unsigned int i;
#ifdef HI_CONFIG_IMS
    for (i = 0u; i < HI_FEATURE_COUNT; ++i) {
        cfg.baseline_mean[i] = HI_IMS_BASELINE_MEAN[i];
        cfg.baseline_std[i] = HI_IMS_BASELINE_STD[i];
    }
    cfg.baseline_median_rms = HI_IMS_BASELINE_MEDIAN_RMS;
    cfg.machine_off_rms_ratio = HI_MACHINE_OFF_RMS_RATIO;
    cfg.ema_alpha = HI_EMA_ALPHA;
    cfg.hi_threshold = HI_IMS_THRESHOLD;
    cfg.hi_baseline_mean = HI_IMS_BASELINE_HI_MEAN;
    cfg.slope_threshold = HI_IMS_SLOPE_THRESHOLD;
#else
    for (i = 0u; i < HI_FEATURE_COUNT; ++i) {
        cfg.baseline_mean[i] = HI_BASELINE_MEAN[i];
        cfg.baseline_std[i] = HI_BASELINE_STD[i];
    }
    cfg.baseline_median_rms = HI_BASELINE_MEDIAN_RMS;
    cfg.machine_off_rms_ratio = HI_MACHINE_OFF_RMS_RATIO;
    cfg.ema_alpha = HI_EMA_ALPHA;
    cfg.hi_threshold = HI_THRESHOLD;
    cfg.hi_baseline_mean = HI_BASELINE_HI_MEAN;
    cfg.slope_threshold = HI_SLOPE_THRESHOLD;
#endif
    hi_stream_init(st, &cfg);
}

static int run_stream_window(hi_stream_t *st) {
    unsigned int i;
    unsigned int start = (st->ring_filled == HI_WINDOW_SIZE) ? st->ring_write : 0u;
    hi_features_t win_feats;

    for (i = 0u; i < HI_WINDOW_SIZE; ++i) {
        st->fft_real[i] = st->ring[(start + i) % HI_WINDOW_SIZE];
    }
    if (hi_compute_features(st->fft_real, st->fft_real, st->fft_imag, &win_feats) != 0) {
        st->error_count += 1u;
        return 0;
    }
    st->windows_processed += 1u;

#ifdef HI_CONFIG_IMS
    for (i = 0u; i < HI_FEATURE_COUNT; ++i) {
        st->win_scratch[st->snap_win_count][i] = win_feats.value[i];
    }
    st->snap_win_count += 1u;
    if (st->snap_win_count == HI_SNAPSHOT_WINDOWS) {
        median_across_windows(st->win_scratch, &st->latest_features);
        hi_monitor_update(&st->mon, &st->cfg, &st->latest_features, &st->latest_output);
        st->has_output = 1;
        st->snapshots_processed += 1u;
        st->snap_win_count = 0u;
        st->ring_write = 0u;
        st->ring_filled = 0u;
        st->hop_accum = 0u;
        return 1;
    }
    return 0;
#else
    st->latest_features = win_feats;
    hi_monitor_update(&st->mon, &st->cfg, &st->latest_features, &st->latest_output);
    st->has_output = 1;
    return 1;
#endif
}

int hi_stream_push(hi_stream_t *st, const float *samples, size_t n) {
    size_t s;
    int produced = 0;

    if (!st || (!samples && n > 0u)) {
        return -1;
    }
    for (s = 0u; s < n; ++s) {
        int trigger = 0;
        st->ring[st->ring_write] = samples[s];
        st->ring_write = (st->ring_write + 1u) % HI_WINDOW_SIZE;

        if (st->ring_filled < HI_WINDOW_SIZE) {
            st->ring_filled += 1u;
            if (st->ring_filled == HI_WINDOW_SIZE) {
                trigger = 1;
                st->hop_accum = 0u;
            }
        } else {
            st->hop_accum += 1u;
            if (st->hop_accum >= HI_HOP_SIZE) {
                trigger = 1;
                st->hop_accum = 0u;
            }
        }

        if (trigger) {
            produced += run_stream_window(st);
        }
    }
    return produced;
}

int hi_stream_latest(const hi_stream_t *st, hi_features_t *features_out, hi_monitor_output_t *out) {
    if (!st || !st->has_output) {
        return 0;
    }
    if (features_out) {
        *features_out = st->latest_features;
    }
    if (out) {
        *out = st->latest_output;
    }
    return 1;
}
