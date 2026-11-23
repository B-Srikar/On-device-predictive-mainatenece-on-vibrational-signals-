#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hi_core.h"
#include "hi_baseline.h"
#include "hi_test_vectors.h"

#define BENCH_WARMUP 100u
#define BENCH_ITERS 2000u
#define REL_TOL 1.0e-3f

static const char *FEATURE_NAMES[HI_FEATURE_COUNT] = {
    "rms",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "log_band_energy_0_400hz",
    "log_band_energy_400_800hz",
    "log_band_energy_800_1200hz",
    "log_band_energy_1200_1600hz"
};

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static hi_monitor_config_t make_config(float slope_thr) {
    hi_monitor_config_t cfg;
    unsigned int i;
    for (i = 0u; i < HI_FEATURE_COUNT; ++i) {
        cfg.baseline_mean[i] = HI_BASELINE_MEAN[i];
        cfg.baseline_std[i] = HI_BASELINE_STD[i];
    }
    cfg.baseline_median_rms = HI_BASELINE_MEDIAN_RMS;
    cfg.machine_off_rms_ratio = HI_MACHINE_OFF_RMS_RATIO;
    cfg.ema_alpha = HI_EMA_ALPHA;
    cfg.hi_threshold = HI_THRESHOLD;
    cfg.hi_baseline_mean = HI_BASELINE_HI_MEAN;
    cfg.slope_threshold = slope_thr;
    return cfg;
}

int main(void) {
    static float window_buf[HI_WINDOW_SIZE];
    static float fft_real[HI_WINDOW_SIZE];
    static float fft_imag[HI_WINDOW_SIZE];
    hi_features_t feats;
    hi_monitor_t mon;
    hi_monitor_config_t mon_cfg = make_config(HI_SEQ_SLOPE_THR);
    hi_monitor_output_t out;

    float max_abs_feat[HI_FEATURE_COUNT];
    float max_rel_feat[HI_FEATURE_COUNT];
    float max_abs_hi = 0.0f;
    float max_rel_hi = 0.0f;
    int all_pass = 1;
    unsigned int i, j;

    for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
        max_abs_feat[j] = 0.0f;
        max_rel_feat[j] = 0.0f;
    }

    printf("=== 1. C VS PYTHON CONFORMANCE (%u stored CWRU windows @ 3200 Hz) ===\n", HI_TEST_CASE_COUNT);
    for (i = 0u; i < HI_TEST_CASE_COUNT; ++i) {
        const hi_test_case_t *tc = &HI_TEST_CASES[i];
        float c_hi;
        float abs_hi, rel_hi;

        for (j = 0u; j < HI_WINDOW_SIZE; ++j) {
            window_buf[j] = tc->samples[j];
        }
        if (hi_compute_features(window_buf, fft_real, fft_imag, &feats) != 0) {
            printf("FAIL: hi_compute_features returned error at case %u (%s)\n", i, tc->label);
            return 1;
        }
        c_hi = hi_compute_score(&feats, HI_BASELINE_MEAN, HI_BASELINE_STD);

        for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
            float diff = fabsf(feats.value[j] - tc->expected_features[j]);
            float denom = fmaxf(fabsf(tc->expected_features[j]), 1.0e-12f);
            float rel = diff / denom;
            if (diff > max_abs_feat[j]) {
                max_abs_feat[j] = diff;
            }
            if (rel > max_rel_feat[j]) {
                max_rel_feat[j] = rel;
            }
        }
        abs_hi = fabsf(c_hi - tc->expected_hi);
        rel_hi = abs_hi / fmaxf(fabsf(tc->expected_hi), 1.0e-12f);
        if (abs_hi > max_abs_hi) {
            max_abs_hi = abs_hi;
        }
        if (rel_hi > max_rel_hi) {
            max_rel_hi = rel_hi;
        }
    }

    printf("  %-30s %14s %14s %8s\n", "Quantity", "MaxAbsDiff", "MaxRelDiff", "Status");
    printf("  --------------------------------------------------------------------\n");
    for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
        int ok = (max_rel_feat[j] <= REL_TOL);
        if (!ok) {
            all_pass = 0;
        }
        printf("  %-30s %14.6e %14.6e %8s\n",
               FEATURE_NAMES[j], (double)max_abs_feat[j], (double)max_rel_feat[j], ok ? "PASS" : "FAIL");
    }
    {
        int ok = (max_rel_hi <= REL_TOL);
        if (!ok) {
            all_pass = 0;
        }
        printf("  %-30s %14.6e %14.6e %8s\n",
               "log1p_HI", (double)max_abs_hi, (double)max_rel_hi, ok ? "PASS" : "FAIL");
    }

    printf("\n=== 2. 200-WINDOW EMA + SLOPE (W=12) + 3-CONSECUTIVE + GATE SEQUENCE ===\n");
    {
        float max_abs_ema = 0.0f, max_rel_ema = 0.0f;
        float max_abs_slp = 0.0f, max_rel_slp = 0.0f;
        unsigned int gated_matches = 0u, slp_valid_matches = 0u;
        unsigned int fix_matches = 0u, slp_alarm_matches = 0u;
        int first_fix = -1, first_slp = -1;

        hi_monitor_init(&mon);
        for (i = 0u; i < HI_SEQ_LEN; ++i) {
            for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
                feats.value[j] = HI_SEQ_FEATURES[i][j];
            }
            hi_monitor_update(&mon, &mon_cfg, &feats, &out);

            {
                float d_ema = fabsf(out.ema_hi - HI_SEQ_EXPECTED_EMA[i]);
                float r_ema = d_ema / fmaxf(fabsf(HI_SEQ_EXPECTED_EMA[i]), 1.0e-12f);
                if (d_ema > max_abs_ema) {
                    max_abs_ema = d_ema;
                }
                if (r_ema > max_rel_ema) {
                    max_rel_ema = r_ema;
                }
            }
            if (out.slope_valid && HI_SEQ_EXPECTED_SLOPE_VALID[i]) {
                float d_slp = fabsf(out.slope - HI_SEQ_EXPECTED_SLOPE[i]);
                float r_slp = d_slp / fmaxf(fabsf(HI_SEQ_EXPECTED_SLOPE[i]), 1.0e-12f);
                if (d_slp > max_abs_slp) {
                    max_abs_slp = d_slp;
                }
                if (r_slp > max_rel_slp) {
                    max_rel_slp = r_slp;
                }
            }
            if (out.gated == HI_SEQ_EXPECTED_GATED[i]) {
                gated_matches += 1u;
            }
            if (out.slope_valid == HI_SEQ_EXPECTED_SLOPE_VALID[i]) {
                slp_valid_matches += 1u;
            }
            if (out.fixed_alarm == HI_SEQ_EXPECTED_FIXED_ALARM[i]) {
                fix_matches += 1u;
            }
            if (out.slope_alarm == HI_SEQ_EXPECTED_SLOPE_ALARM[i]) {
                slp_alarm_matches += 1u;
            }
            if (out.fixed_alarm && first_fix < 0) {
                first_fix = (int)i;
            }
            if (out.slope_alarm && first_slp < 0) {
                first_slp = (int)i;
            }
        }
        {
            int seq_ok = (max_rel_ema <= REL_TOL) && (max_rel_slp <= REL_TOL) &&
                         (gated_matches == HI_SEQ_LEN) && (slp_valid_matches == HI_SEQ_LEN) &&
                         (fix_matches == HI_SEQ_LEN) && (slp_alarm_matches == HI_SEQ_LEN);
            if (!seq_ok) {
                all_pass = 0;
            }
            printf("  EMA         : max_abs=%.6e, max_rel=%.6e\n", (double)max_abs_ema, (double)max_rel_ema);
            printf("  Slope (W=12): max_abs=%.6e, max_rel=%.6e\n", (double)max_abs_slp, (double)max_rel_slp);
            printf("  Exact flags : gated=%u/%u, slope_valid=%u/%u, fixed_alarm=%u/%u, slope_alarm=%u/%u\n",
                   gated_matches, HI_SEQ_LEN, slp_valid_matches, HI_SEQ_LEN,
                   fix_matches, HI_SEQ_LEN, slp_alarm_matches, HI_SEQ_LEN);
            printf("  First alarms: fixed_idx=%d, slope_idx=%d (injected machine-off gate at idx 100)\n",
                   first_fix, first_slp);
            printf("  Sequence status: %s\n", seq_ok ? "PASS" : "FAIL");
        }
    }

    printf("\n=== 3. LAPTOP BENCHMARK (1024 samples @ 3200 Hz = 0.320 s/window) ===\n");
    {
        static double elapsed_us[BENCH_ITERS];
        volatile float sink = 0.0f;
        double sum_us = 0.0, mean_us, p99_us, rtf;
        struct timespec t0, t1;

        hi_monitor_init(&mon);
        for (i = 0u; i < BENCH_WARMUP; ++i) {
            const hi_test_case_t *tc = &HI_TEST_CASES[i % HI_TEST_CASE_COUNT];
            hi_compute_features(tc->samples, fft_real, fft_imag, &feats);
            hi_monitor_update(&mon, &mon_cfg, &feats, &out);
            sink += out.ema_hi;
        }
        for (i = 0u; i < BENCH_ITERS; ++i) {
            const hi_test_case_t *tc = &HI_TEST_CASES[i % HI_TEST_CASE_COUNT];
            clock_gettime(CLOCK_MONOTONIC, &t0);
            hi_compute_features(tc->samples, fft_real, fft_imag, &feats);
            hi_monitor_update(&mon, &mon_cfg, &feats, &out);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            sink += out.ema_hi;
            elapsed_us[i] = ((double)(t1.tv_sec - t0.tv_sec) * 1.0e6) +
                            ((double)(t1.tv_nsec - t0.tv_nsec) * 1.0e-3);
            sum_us += elapsed_us[i];
        }
        (void)sink;
        qsort(elapsed_us, BENCH_ITERS, sizeof(double), cmp_double);
        mean_us = sum_us / (double)BENCH_ITERS;
        p99_us = elapsed_us[(size_t)((0.99 * (double)BENCH_ITERS) - 0.5)];
        rtf = (mean_us * 1.0e-6) / ((double)HI_WINDOW_SIZE / (double)HI_SAMPLE_RATE_HZ);
        printf("  Mean latency   : %.2f us (%.4f ms)\n", mean_us, mean_us * 1.0e-3);
        printf("  p99 latency    : %.2f us (%.4f ms)\n", p99_us, p99_us * 1.0e-3);
        printf("  Real-time frac : %.6e (compute_time / 0.32 s; speedup = %.0fx real-time)\n", rtf, 1.0 / rtf);
    }

    printf("\n=== 4. MEMORY FOOTPRINT (sizeof) ===\n");
    {
        size_t ram_total = sizeof(window_buf) + sizeof(fft_real) + sizeof(fft_imag) +
                           sizeof(feats) + sizeof(mon) + sizeof(out);
        size_t flash_consts = sizeof(HI_BANDS) + sizeof(HI_BASELINE_MEAN) +
                              sizeof(HI_BASELINE_STD) + sizeof(mon_cfg);
        printf("  Static RAM buffers:\n");
        printf("    window_buf[1024]  : %zu bytes\n", sizeof(window_buf));
        printf("    fft_real[1024]    : %zu bytes\n", sizeof(fft_real));
        printf("    fft_imag[1024]    : %zu bytes\n", sizeof(fft_imag));
        printf("    hi_features_t     : %zu bytes\n", sizeof(feats));
        printf("    hi_monitor_t      : %zu bytes\n", sizeof(mon));
        printf("    hi_monitor_out_t  : %zu bytes\n", sizeof(out));
        printf("    TOTAL static RAM  : %zu bytes (%.2f KiB)\n", ram_total, (double)ram_total / 1024.0);
        printf("  Flash constant arrays:\n");
        printf("    HI_BANDS          : %zu bytes\n", sizeof(HI_BANDS));
        printf("    HI_BASELINE_MEAN  : %zu bytes\n", sizeof(HI_BASELINE_MEAN));
        printf("    HI_BASELINE_STD   : %zu bytes\n", sizeof(HI_BASELINE_STD));
        printf("    hi_monitor_cfg_t  : %zu bytes\n", sizeof(mon_cfg));
        printf("    Total flash const : %zu bytes (deduplicated config: %zu bytes)\n",
               flash_consts, sizeof(HI_BANDS) + sizeof(mon_cfg));
    }

    if (!all_pass) {
        printf("\nOVERALL RESULT: FAIL\n");
        return 1;
    }
    printf("\nOVERALL RESULT: PASS\n");
    return 0;
}
