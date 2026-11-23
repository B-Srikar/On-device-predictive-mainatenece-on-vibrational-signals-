#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "hi_core.h"
#include "hi_baseline_ims.h"

#define REL_TOL 1.0e-3f

static const char *IMS_FEAT_NAMES[HI_FEATURE_COUNT] = {
    "rms",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "log_band_energy_0_500hz",
    "log_band_energy_500_1500hz",
    "log_band_energy_1500_3000hz",
    "log_band_energy_3000_6000hz",
    "log_band_energy_6000_10240hz"
};

int main(void) {
    static float snap_buf[HI_SNAPSHOT_SAMPLES];
    static float fft_real[HI_WINDOW_SIZE];
    static float fft_imag[HI_WINDOW_SIZE];
    static float win_scratch[HI_SNAPSHOT_WINDOWS][HI_FEATURE_COUNT];
    static float py_feats[HI_FEATURE_COUNT];

    hi_monitor_config_t cfg;
    hi_monitor_t mon;
    hi_features_t feats;
    hi_monitor_output_t out;

    uint32_t n_wav = 0u, n_samp = 0u, n_exp = 0u, t;
    float exp_slope_thr = 0.0f, exp_hi_thr = 0.0f, exp_hi_base = 0.0f;
    float max_abs_feat[HI_FEATURE_COUNT], max_rel_feat[HI_FEATURE_COUNT];
    float max_abs_raw = 0.0f, max_rel_raw = 0.0f;
    float max_abs_ema = 0.0f, max_rel_ema = 0.0f;
    float max_abs_slp = 0.0f, max_rel_slp = 0.0f;
    unsigned int slp_valid_matches = 0u, fix_matches = 0u, slp_matches = 0u;
    unsigned int hlth_fix = 0u, hlth_slp = 0u;
    int first_fix = -1, first_slp = -1;
    unsigned int j;
    int all_pass = 1;
    FILE *fw, *fe;

    for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
        cfg.baseline_mean[j] = HI_IMS_BASELINE_MEAN[j];
        cfg.baseline_std[j] = HI_IMS_BASELINE_STD[j];
        max_abs_feat[j] = 0.0f;
        max_rel_feat[j] = 0.0f;
    }
    cfg.baseline_median_rms = HI_IMS_BASELINE_MEDIAN_RMS;
    cfg.machine_off_rms_ratio = HI_MACHINE_OFF_RMS_RATIO;
    cfg.ema_alpha = HI_EMA_ALPHA;
    cfg.hi_threshold = HI_IMS_THRESHOLD;
    cfg.hi_baseline_mean = HI_IMS_BASELINE_HI_MEAN;
    cfg.slope_threshold = HI_IMS_SLOPE_THRESHOLD;

    fw = fopen("data/processed/ims_ch1_waveforms.bin", "rb");
    fe = fopen("data/processed/ims_ch1_expected.bin", "rb");
    if (!fw || !fe) {
        if (fw) fclose(fw);
        if (fe) fclose(fe);
        printf("ERROR: failed to open IMS conformance binary files\n");
        return 1;
    }
    if (fread(&n_wav, sizeof(uint32_t), 1u, fw) != 1u ||
        fread(&n_samp, sizeof(uint32_t), 1u, fw) != 1u ||
        fread(&n_exp, sizeof(uint32_t), 1u, fe) != 1u ||
        fread(&exp_slope_thr, sizeof(float), 1u, fe) != 1u ||
        fread(&exp_hi_thr, sizeof(float), 1u, fe) != 1u ||
        fread(&exp_hi_base, sizeof(float), 1u, fe) != 1u) {
        fclose(fw);
        fclose(fe);
        printf("ERROR: failed reading IMS binary headers\n");
        return 1;
    }
    if (n_wav != n_exp || n_samp != HI_SNAPSHOT_SAMPLES) {
        fclose(fw);
        fclose(fe);
        printf("ERROR: header mismatch (%u vs %u, samp=%u)\n", n_wav, n_exp, n_samp);
        return 1;
    }

    hi_monitor_init(&mon);
    for (t = 0u; t < n_wav; ++t) {
        float py_raw = 0.0f, py_ema = 0.0f, py_slp = 0.0f;
        int32_t py_svalid = 0, py_fix = 0, py_slp_al = 0;

        if (fread(snap_buf, sizeof(float), HI_SNAPSHOT_SAMPLES, fw) != HI_SNAPSHOT_SAMPLES ||
            fread(py_feats, sizeof(float), HI_FEATURE_COUNT, fe) != HI_FEATURE_COUNT ||
            fread(&py_raw, sizeof(float), 1u, fe) != 1u ||
            fread(&py_ema, sizeof(float), 1u, fe) != 1u ||
            fread(&py_slp, sizeof(float), 1u, fe) != 1u ||
            fread(&py_svalid, sizeof(int32_t), 1u, fe) != 1u ||
            fread(&py_fix, sizeof(int32_t), 1u, fe) != 1u ||
            fread(&py_slp_al, sizeof(int32_t), 1u, fe) != 1u) {
            fclose(fw);
            fclose(fe);
            printf("ERROR: truncated IMS record %u\n", t);
            return 1;
        }
        if (hi_compute_snapshot_features(snap_buf, fft_real, fft_imag, win_scratch, &feats) != 0) {
            fclose(fw);
            fclose(fe);
            printf("ERROR: hi_compute_snapshot_features failed at %u\n", t);
            return 1;
        }
        hi_monitor_update(&mon, &cfg, &feats, &out);

        for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
            float d = fabsf(feats.value[j] - py_feats[j]);
            float r = d / fmaxf(fabsf(py_feats[j]), 1.0e-12f);
            if (d > max_abs_feat[j]) max_abs_feat[j] = d;
            if (r > max_rel_feat[j]) max_rel_feat[j] = r;
        }
        {
            float d = fabsf(out.raw_hi - py_raw);
            float r = d / fmaxf(fabsf(py_raw), 1.0e-12f);
            if (d > max_abs_raw) max_abs_raw = d;
            if (r > max_rel_raw) max_rel_raw = r;
        }
        {
            float d = fabsf(out.ema_hi - py_ema);
            float r = d / fmaxf(fabsf(py_ema), 1.0e-12f);
            if (d > max_abs_ema) max_abs_ema = d;
            if (r > max_rel_ema) max_rel_ema = r;
        }
        if (out.slope_valid && py_svalid) {
            float d = fabsf(out.slope - py_slp);
            float r = d / fmaxf(fabsf(py_slp), 1.0e-6f);
            if (d > max_abs_slp) max_abs_slp = d;
            if (r > max_rel_slp) max_rel_slp = r;
        }
        if (out.slope_valid == py_svalid) slp_valid_matches += 1u;
        if (out.fixed_alarm == py_fix) fix_matches += 1u;
        if (out.slope_alarm == py_slp_al) slp_matches += 1u;

        if (t >= 196u && t <= 491u) {
            if (out.fixed_alarm) hlth_fix += 1u;
            if (out.slope_alarm) hlth_slp += 1u;
        }
        if (t >= 196u) {
            if (out.fixed_alarm && first_fix < 0) first_fix = (int)t;
            if (out.slope_alarm && first_slp < 0) first_slp = (int)t;
        }
    }
    fclose(fw);
    fclose(fe);

    printf("=== 4. IMS CH 1 (20480 Hz) ALL %u SNAPSHOTS C VS PYTHON CONFORMANCE ===\n", n_wav);
    printf("  Thresholds: HI=%.6f (py=%.6f), slope=%.6f (py=%.6f)\n",
           (double)HI_IMS_THRESHOLD, (double)exp_hi_thr, (double)HI_IMS_SLOPE_THRESHOLD, (double)exp_slope_thr);
    printf("  %-30s %14s %14s %8s\n", "Quantity", "MaxAbsDiff", "MaxRelDiff", "Status");
    printf("  --------------------------------------------------------------------\n");
    for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
        int ok = (max_rel_feat[j] <= REL_TOL);
        if (!ok) all_pass = 0;
        printf("  %-30s %14.6e %14.6e %8s\n",
               IMS_FEAT_NAMES[j], (double)max_abs_feat[j], (double)max_rel_feat[j], ok ? "PASS" : "FAIL");
    }
    {
        int ok_raw = (max_rel_raw <= REL_TOL), ok_ema = (max_rel_ema <= REL_TOL);
        if (!ok_raw || !ok_ema) all_pass = 0;
        printf("  %-30s %14.6e %14.6e %8s\n", "HI_raw", (double)max_abs_raw, (double)max_rel_raw, ok_raw ? "PASS" : "FAIL");
        printf("  %-30s %14.6e %14.6e %8s\n", "HI_ema", (double)max_abs_ema, (double)max_rel_ema, ok_ema ? "PASS" : "FAIL");
        printf("  %-30s %14.6e %14.6e %8s\n", "slope (W=12, where >=1e-6)",
               (double)max_abs_slp, (double)max_rel_slp, (max_abs_slp <= 1.0e-5f) ? "PASS" : "FAIL");
        if (max_abs_slp > 1.0e-5f) all_pass = 0;
    }
    printf("  Exact boolean matches: slope_valid=%u/%u, fixed_alarm=%u/%u, slope_alarm=%u/%u\n",
           slp_valid_matches, n_wav, fix_matches, n_wav, slp_matches, n_wav);
    printf("  Presumed-healthy alarms (196..491): fixed=%u, slope=%u\n", hlth_fix, hlth_slp);
    printf("  C First alarm indices (>=196): fixed=%d (Stage4=527), slope=%d (Stage4=538)\n",
           first_fix, first_slp);
    if (slp_valid_matches != n_wav || fix_matches != n_wav || slp_matches != n_wav ||
        first_fix != 527 || first_slp != 538) {
        all_pass = 0;
    }
    printf("\n  IMS static RAM (sizeof): snap=%zu + fft_r=%zu + fft_i=%zu + scratch=%zu + feats=%zu + mon=%zu + out=%zu = %zu B\n",
           sizeof(snap_buf), sizeof(fft_real), sizeof(fft_imag), sizeof(win_scratch),
           sizeof(feats), sizeof(mon), sizeof(out),
           sizeof(snap_buf) + sizeof(fft_real) + sizeof(fft_imag) + sizeof(win_scratch) + sizeof(feats) + sizeof(mon) + sizeof(out));
    printf("  IMS flash consts (sizeof): HI_BANDS=%zu B, HI_IMS_BASELINE_MEAN=%zu B, HI_IMS_BASELINE_STD=%zu B, cfg=%zu B\n",
           sizeof(HI_BANDS), sizeof(HI_IMS_BASELINE_MEAN), sizeof(HI_IMS_BASELINE_STD), sizeof(cfg));

    printf("\nIMS STAGE 6B STATUS: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

