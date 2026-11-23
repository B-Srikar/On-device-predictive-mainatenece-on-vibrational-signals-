#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hi_core.h"
#include "hi_baseline.h"

#define REL_TOL 1.0e-3f
#define WORST_K 3u

static const char *QTY_NAMES[HI_FEATURE_COUNT + 1u] = {
    "rms",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "log_band_energy_0_400hz",
    "log_band_energy_400_800hz",
    "log_band_energy_800_1200hz",
    "log_band_energy_1200_1600hz",
    "log1p_HI"
};

typedef struct {
    char label[32];
    char file[32];
    unsigned int window_idx;
    unsigned int qty_idx;
    float py_val;
    float c_val;
    float abs_diff;
    float rel_diff;
} worst_rec_t;

static void insert_worst(worst_rec_t worst[WORST_K], const worst_rec_t *cand) {
    unsigned int i, j;
    for (i = 0u; i < WORST_K; ++i) {
        if (cand->rel_diff > worst[i].rel_diff) {
            for (j = WORST_K - 1u; j > i; --j) {
                worst[j] = worst[j - 1u];
            }
            worst[i] = *cand;
            break;
        }
    }
}

static hi_monitor_config_t make_cwru_cfg(void) {
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
    cfg.slope_threshold = HI_SLOPE_THRESHOLD;
    return cfg;
}

static int run_edge_case(const char *name, const char *expected, const float win[HI_WINDOW_SIZE],
                         float fft_real[HI_WINDOW_SIZE], float fft_imag[HI_WINDOW_SIZE],
                         const hi_monitor_config_t *cfg) {
    hi_features_t feats;
    hi_monitor_t mon;
    hi_monitor_output_t out;
    int rc;
    unsigned int j;

    hi_monitor_init(&mon);
    rc = hi_compute_features(win, fft_real, fft_imag, &feats);
    if (rc != 0) {
        printf("  %-18s | exp: %s | rc=%d (error handled safely, monitor untouched)\n", name, expected, rc);
        return 0;
    }
    for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
        if (!isfinite(feats.value[j])) {
            printf("  %-18s | FAIL: non-finite feature %u\n", name, j);
            return 1;
        }
    }
    hi_monitor_update(&mon, cfg, &feats, &out);
    if (!isfinite(out.raw_hi) || !isfinite(out.ema_hi)) {
        printf("  %-18s | FAIL: non-finite HI\n", name);
        return 1;
    }
    printf("  %-18s | exp: %s | rc=%d gated=%d rms=%.3e ptp=%.3e crest=%.3f kurt=%.3f b0=%.2f rawHI=%.3f\n",
           name, expected, rc, out.gated, (double)feats.value[HI_RMS],
           (double)feats.value[HI_PEAK_TO_PEAK], (double)feats.value[HI_CREST_FACTOR],
           (double)feats.value[HI_KURTOSIS], (double)feats.value[HI_LOG_BAND_0], (double)out.raw_hi);
    return 0;
}

int main(void) {
    static float window_buf[HI_WINDOW_SIZE];
    static float reuse_buf[HI_WINDOW_SIZE];
    static float fft_real[HI_WINDOW_SIZE];
    static float fft_imag[HI_WINDOW_SIZE];
    static float py_feats[HI_FEATURE_COUNT];
    hi_features_t feats, feats_reuse;
    hi_monitor_config_t cfg = make_cwru_cfg();

    float max_abs[HI_FEATURE_COUNT + 1u];
    float max_rel[HI_FEATURE_COUNT + 1u];
    float max_reuse_feat_diff = 0.0f;
    float max_reuse_hi_diff = 0.0f;
    worst_rec_t worst[WORST_K];
    uint32_t n_records = 0u, idx = 0u;
    unsigned int j;
    int all_pass = 1;
    FILE *fp;

    for (j = 0u; j <= HI_FEATURE_COUNT; ++j) {
        max_abs[j] = 0.0f;
        max_rel[j] = 0.0f;
    }
    memset(worst, 0, sizeof(worst));
    for (j = 0u; j < WORST_K; ++j) {
        worst[j].rel_diff = -1.0f;
    }

    fp = fopen("data/processed/cwru_3200_conformance.bin", "rb");
    if (!fp) {
        printf("ERROR: failed to open data/processed/cwru_3200_conformance.bin\n");
        return 1;
    }
    if (fread(&n_records, sizeof(uint32_t), 1u, fp) != 1u) {
        fclose(fp);
        return 1;
    }

    for (idx = 0u; idx < n_records; ++idx) {
        char label[32], file[32];
        uint32_t win_idx = 0u;
        float py_hi = 0.0f, c_hi = 0.0f, c_hi_reuse = 0.0f;
        worst_rec_t win_worst;

        if (fread(label, 1u, 32u, fp) != 32u ||
            fread(file, 1u, 32u, fp) != 32u ||
            fread(&win_idx, sizeof(uint32_t), 1u, fp) != 1u ||
            fread(window_buf, sizeof(float), HI_WINDOW_SIZE, fp) != HI_WINDOW_SIZE ||
            fread(py_feats, sizeof(float), HI_FEATURE_COUNT, fp) != HI_FEATURE_COUNT ||
            fread(&py_hi, sizeof(float), 1u, fp) != 1u) {
            fclose(fp);
            printf("ERROR: truncated binary record %u\n", idx);
            return 1;
        }
        label[31] = '\0';
        file[31] = '\0';

        for (j = 0u; j < HI_WINDOW_SIZE; ++j) {
            reuse_buf[j] = window_buf[j];
        }
        if (hi_compute_features(window_buf, fft_real, fft_imag, &feats) != 0 ||
            hi_compute_features(reuse_buf, reuse_buf, fft_imag, &feats_reuse) != 0) {
            fclose(fp);
            printf("ERROR: hi_compute_features failed at record %u\n", idx);
            return 1;
        }
        c_hi = hi_compute_score(&feats, HI_BASELINE_MEAN, HI_BASELINE_STD);
        c_hi_reuse = hi_compute_score(&feats_reuse, HI_BASELINE_MEAN, HI_BASELINE_STD);
        if (fabsf(c_hi - c_hi_reuse) > max_reuse_hi_diff) {
            max_reuse_hi_diff = fabsf(c_hi - c_hi_reuse);
        }

        win_worst.rel_diff = -1.0f;
        for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
            float diff = fabsf(feats.value[j] - py_feats[j]);
            float rel = diff / fmaxf(fabsf(py_feats[j]), 1.0e-12f);
            float rdiff = fabsf(feats.value[j] - feats_reuse.value[j]);
            if (rdiff > max_reuse_feat_diff) {
                max_reuse_feat_diff = rdiff;
            }
            if (diff > max_abs[j]) {
                max_abs[j] = diff;
            }
            if (rel > max_rel[j]) {
                max_rel[j] = rel;
            }
            if (rel > win_worst.rel_diff) {
                snprintf(win_worst.label, sizeof(win_worst.label), "%s", label);
                snprintf(win_worst.file, sizeof(win_worst.file), "%s", file);
                win_worst.window_idx = win_idx;
                win_worst.qty_idx = j;
                win_worst.py_val = py_feats[j];
                win_worst.c_val = feats.value[j];
                win_worst.abs_diff = diff;
                win_worst.rel_diff = rel;
            }
        }
        {
            float diff = fabsf(c_hi - py_hi);
            float rel = diff / fmaxf(fabsf(py_hi), 1.0e-12f);
            if (diff > max_abs[HI_FEATURE_COUNT]) {
                max_abs[HI_FEATURE_COUNT] = diff;
            }
            if (rel > max_rel[HI_FEATURE_COUNT]) {
                max_rel[HI_FEATURE_COUNT] = rel;
            }
            if (rel > win_worst.rel_diff) {
                snprintf(win_worst.label, sizeof(win_worst.label), "%s", label);
                snprintf(win_worst.file, sizeof(win_worst.file), "%s", file);
                win_worst.window_idx = win_idx;
                win_worst.qty_idx = HI_FEATURE_COUNT;
                win_worst.py_val = py_hi;
                win_worst.c_val = c_hi;
                win_worst.abs_diff = diff;
                win_worst.rel_diff = rel;
            }
        }
        insert_worst(worst, &win_worst);
    }
    fclose(fp);

    printf("=== 1. ALL %u CWRU 3200 Hz WINDOWS CONFORMANCE (including normal_1-3) ===\n", n_records);
    printf("  %-30s %14s %14s %8s\n", "Quantity", "MaxAbsDiff", "MaxRelDiff", "Status");
    printf("  --------------------------------------------------------------------\n");
    for (j = 0u; j <= HI_FEATURE_COUNT; ++j) {
        int ok = (max_rel[j] <= REL_TOL);
        if (!ok) {
            all_pass = 0;
        }
        printf("  %-30s %14.6e %14.6e %8s\n",
               QTY_NAMES[j], (double)max_abs[j], (double)max_rel[j], ok ? "PASS" : "FAIL");
    }
    printf("\n  Worst 3 windows by max relative error:\n");
    for (j = 0u; j < WORST_K; ++j) {
        printf("    #%u: label=%-16s file=%-16s win=%3u qty=%-14s py=%.7g c=%.7g abs=%.3e rel=%.3e\n",
               j + 1u, worst[j].label, worst[j].file, worst[j].window_idx,
               QTY_NAMES[worst[j].qty_idx], (double)worst[j].py_val, (double)worst[j].c_val,
               (double)worst[j].abs_diff, (double)worst[j].rel_diff);
    }

    printf("\n=== 2. EDGE CASES ===\n");
    for (j = 0u; j < HI_WINDOW_SIZE; ++j) {
        window_buf[j] = 0.3f;
    }
    if (run_edge_case("constant_0.3", "rc=0 gated=1 rms~0 ptp=0 kurt=0 bands=-69.08", window_buf, fft_real, fft_imag, &cfg)) {
        all_pass = 0;
    }
    for (j = 0u; j < HI_WINDOW_SIZE; ++j) {
        window_buf[j] = 0.0f;
    }
    if (run_edge_case("all_zero", "rc=0 gated=1 rms=0 ptp=0 kurt=0 bands=-69.08", window_buf, fft_real, fft_imag, &cfg)) {
        all_pass = 0;
    }
    for (j = 0u; j < HI_WINDOW_SIZE; ++j) {
        float v = 10.0f * sinf(0.19634954f * (float)j);
        window_buf[j] = (v > 5.0f) ? 5.0f : ((v < -5.0f) ? -5.0f : v);
    }
    if (run_edge_case("clipped_pm5", "rc=0 gated=0 finite ptp=10.0", window_buf, fft_real, fft_imag, &cfg)) {
        all_pass = 0;
    }
    window_buf[512] = NAN;
    if (run_edge_case("one_nan", "rc=-1 early reject", window_buf, fft_real, fft_imag, &cfg)) {
        all_pass = 0;
    }

    printf("\n=== 3. IN-PLACE WINDOW_BUF -> FFT_REAL REUSE CHECK ===\n");
    printf("  Max abs diff (separate vs aliased window_buf/fft_real): feats=%.6e, HI=%.6e\n",
           (double)max_reuse_feat_diff, (double)max_reuse_hi_diff);
    printf("  RAM saved if aliased: %zu bytes (4.00 KiB); caveat: mutates window_buf in place (overlap requires raw tail save)\n",
           sizeof(fft_real));

    printf("\n=== CONSTANT ARRAYS (3200 Hz CWRU) ===\n");
    printf("  sizeof(HI_BANDS)=%zu B, sizeof(HI_BASELINE_MEAN)=%zu B, sizeof(HI_BASELINE_STD)=%zu B, sizeof(hi_monitor_config_t)=%zu B\n",
           sizeof(HI_BANDS), sizeof(HI_BASELINE_MEAN), sizeof(HI_BASELINE_STD), sizeof(cfg));

    printf("\nCWRU STAGE 6B STATUS: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

