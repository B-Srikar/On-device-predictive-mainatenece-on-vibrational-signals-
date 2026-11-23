#ifndef HI_CORE_H
#define HI_CORE_H

#include <stddef.h>
#include "hi_config.h"

/* Locked Stage 3b/3c/4 feature vector: excludes std and skewness; band energies are log-transformed. */
enum {
    HI_RMS = 0,
    HI_PEAK_TO_PEAK,
    HI_CREST_FACTOR,
    HI_KURTOSIS,
    HI_LOG_BAND_0,
    HI_LOG_BAND_1,
    HI_LOG_BAND_2,
    HI_LOG_BAND_3,
#if HI_BAND_COUNT >= 5u
    HI_LOG_BAND_4,
#endif
    HI_FEATURE_COUNT
};

typedef struct {
    float value[HI_FEATURE_COUNT];
} hi_features_t;

typedef struct {
    float baseline_mean[HI_FEATURE_COUNT];
    float baseline_std[HI_FEATURE_COUNT];
    float baseline_median_rms;
    float machine_off_rms_ratio;
    float ema_alpha;
    float hi_threshold;
    float hi_baseline_mean;
    float slope_threshold;
} hi_monitor_config_t;

typedef struct {
    int ema_initialized;
    float ema_hi;
    float ema_ring[HI_SLOPE_WINDOW];
    unsigned int ring_pos;
    unsigned int ring_count;
    unsigned int fixed_consec;
    unsigned int slope_consec;
} hi_monitor_t;

typedef struct {
    int gated;
    float raw_hi;
    float ema_hi;
    int slope_valid;
    float slope;
    int fixed_alarm;
    int slope_alarm;
} hi_monitor_output_t;

/* Caller-owned streaming state. No dynamic allocation; IMS mode omits 20480-sample buffer. */
typedef struct {
    hi_monitor_config_t cfg;
    hi_monitor_t mon;
    float ring[HI_WINDOW_SIZE];
    float fft_real[HI_WINDOW_SIZE];
    float fft_imag[HI_WINDOW_SIZE];
    unsigned int ring_write;
    unsigned int ring_filled;
    unsigned int hop_accum;
#ifdef HI_CONFIG_IMS
    float win_scratch[HI_SNAPSHOT_WINDOWS][HI_FEATURE_COUNT];
    unsigned int snap_win_count;
#endif
    hi_features_t latest_features;
    hi_monitor_output_t latest_output;
    int has_output;
    unsigned int windows_processed;
    unsigned int snapshots_processed;
    unsigned int error_count;
} hi_stream_t;

/* All buffers are caller-owned float32 arrays. No dynamic allocation is used. */
int hi_compute_features(const float window[HI_WINDOW_SIZE],
                        float fft_real[HI_WINDOW_SIZE],
                        float fft_imag[HI_WINDOW_SIZE],
                        hi_features_t *features);

#ifdef HI_CONFIG_IMS
int hi_compute_snapshot_features(const float snapshot[HI_SNAPSHOT_SAMPLES],
                                 float fft_real[HI_WINDOW_SIZE],
                                 float fft_imag[HI_WINDOW_SIZE],
                                 float win_scratch[HI_SNAPSHOT_WINDOWS][HI_FEATURE_COUNT],
                                 hi_features_t *out);
#endif

float hi_compute_score(const hi_features_t *features,
                       const float mean[HI_FEATURE_COUNT],
                       const float std[HI_FEATURE_COUNT]);

void hi_monitor_init(hi_monitor_t *mon);

void hi_monitor_update(hi_monitor_t *mon,
                       const hi_monitor_config_t *cfg,
                       const hi_features_t *features,
                       hi_monitor_output_t *out);

size_t hi_stream_sizeof(void);
void hi_stream_init(hi_stream_t *st, const hi_monitor_config_t *cfg);
void hi_stream_init_default(hi_stream_t *st);
int hi_stream_push(hi_stream_t *st, const float *samples, size_t n);
int hi_stream_latest(const hi_stream_t *st, hi_features_t *features_out, hi_monitor_output_t *out);

#endif
