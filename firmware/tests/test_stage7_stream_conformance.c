#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hi_core.h"

#define REL_TOL 1.0e-3f

#ifdef HI_CONFIG_IMS
static const char *FEAT_NAMES[HI_FEATURE_COUNT] = {
    "rms", "peak_to_peak", "crest_factor", "kurtosis",
    "log_band_0_500", "log_band_500_1500", "log_band_1500_3000", "log_band_3000_6000", "log_band_6000_10240"
};

int main(void) {
    static hi_stream_t st;
    static float snap_buf[HI_SNAPSHOT_SAMPLES];
    static float py_feats[HI_FEATURE_COUNT];
    uint32_t n_wav = 0u, n_samp = 0u, n_exp = 0u, t;
    float exp_slope_thr = 0.0f, exp_hi_thr = 0.0f, exp_hi_base = 0.0f;
    float max_abs_feat[HI_FEATURE_COUNT], max_rel_feat[HI_FEATURE_COUNT];
    float max_abs_raw = 0.0f, max_rel_raw = 0.0f;
    float max_abs_ema = 0.0f, max_rel_ema = 0.0f;
    float max_abs_slp = 0.0f;
    unsigned int slp_valid_matches = 0u, fix_matches = 0u, slp_matches = 0u;
    unsigned int hlth_fix = 0u, hlth_slp = 0u;
    int first_fix = -1, first_slp = -1;
    unsigned int j;
    int all_pass = 1;
    FILE *fw, *fe;

    for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
        max_abs_feat[j] = 0.0f;
        max_rel_feat[j] = 0.0f;
    }

    fw = fopen("data/processed/ims_ch1_waveforms.bin", "rb");
    fe = fopen("data/processed/ims_ch1_expected.bin", "rb");
    if (!fw || !fe) {
        if (fw) fclose(fw);
        if (fe) fclose(fe);
        printf("ERROR: failed to open IMS conformance binaries\n");
        return 1;
    }
    if (fread(&n_wav, sizeof(uint32_t), 1u, fw) != 1u ||
        fread(&n_samp, sizeof(uint32_t), 1u, fw) != 1u ||
        fread(&n_exp, sizeof(uint32_t), 1u, fe) != 1u ||
        fread(&exp_slope_thr, sizeof(float), 1u, fe) != 1u ||
        fread(&exp_hi_thr, sizeof(float), 1u, fe) != 1u ||
        fread(&exp_hi_base, sizeof(float), 1u, fe) != 1u ||
        n_wav != n_exp || n_samp != HI_SNAPSHOT_SAMPLES) {
        fclose(fw);
        fclose(fe);
        printf("ERROR: bad IMS binary headers\n");
        return 1;
    }

    hi_stream_init_default(&st);
    for (t = 0u; t < n_wav; ++t) {
        float py_raw = 0.0f, py_ema = 0.0f, py_slp = 0.0f;
        int32_t py_svalid = 0, py_fix = 0, py_slp_al = 0;
        hi_features_t feats;
        hi_monitor_output_t out;
        unsigned int off;
        int produced = 0;

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
            return 1;
        }
        for (off = 0u; off < HI_SNAPSHOT_SAMPLES; off += HI_HOP_SIZE) {
            produced += hi_stream_push(&st, &snap_buf[off], HI_HOP_SIZE);
        }
        if (produced != 1 || hi_stream_latest(&st, &feats, &out) != 1) {
            fclose(fw);
            fclose(fe);
            printf("ERROR: snapshot %u produced=%d\n", t, produced);
            return 1;
        }
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
            if (d > max_abs_slp) max_abs_slp = d;
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

    printf("=== STREAMING CONFORMANCE: IMS 20480 Hz (%u snapshots, 512-sample chunks) ===\n", n_wav);
    printf("  Static RAM sizeof(hi_stream_t) = %zu bytes (%.2f KiB; NO 20480-sample buffer)\n",
           hi_stream_sizeof(), (double)hi_stream_sizeof() / 1024.0);
    for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
        if (max_rel_feat[j] > REL_TOL) all_pass = 0;
        printf("  %-24s max_abs=%.6e max_rel=%.6e %s\n",
               FEAT_NAMES[j], (double)max_abs_feat[j], (double)max_rel_feat[j],
               (max_rel_feat[j] <= REL_TOL) ? "PASS" : "FAIL");
    }
    if (max_rel_ema > REL_TOL || max_abs_slp > 1.0e-5f ||
        slp_valid_matches != n_wav || fix_matches != n_wav || slp_matches != n_wav ||
        first_fix != 527 || first_slp != 538) {
        all_pass = 0;
    }
    printf("  %-24s max_abs=%.6e max_rel=%.6e (baseline near-zero idx86 ref)\n",
           "HI_raw", (double)max_abs_raw, (double)max_rel_raw);
    printf("  %-24s max_abs=%.6e max_rel=%.6e %s\n",
           "HI_ema", (double)max_abs_ema, (double)max_rel_ema, (max_rel_ema <= REL_TOL) ? "PASS" : "FAIL");
    printf("  %-24s max_abs=%.6e PASS\n", "slope (W=12)", (double)max_abs_slp);
    printf("  Exact alarms: slope_valid=%u/%u fixed=%u/%u slope=%u/%u | 196..491: %u/%u | first=(%d, %d)\n",
           slp_valid_matches, n_wav, fix_matches, n_wav, slp_matches, n_wav,
           hlth_fix, hlth_slp, first_fix, first_slp);
    printf("  IMS STREAM STATUS: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

#else
static const char *FEAT_NAMES[HI_FEATURE_COUNT + 1u] = {
    "rms", "peak_to_peak", "crest_factor", "kurtosis",
    "log_band_0_400", "log_band_400_800", "log_band_800_1200", "log_band_1200_1600", "log1p_HI"
};

int main(void) {
    static hi_stream_t st;
    static float window_buf[HI_WINDOW_SIZE];
    static float py_feats[HI_FEATURE_COUNT];
    char prev_file[32] = "";
    uint32_t n_records = 0u, idx = 0u;
    float max_abs[HI_FEATURE_COUNT + 1u], max_rel[HI_FEATURE_COUNT + 1u];
    unsigned int j;
    int all_pass = 1;
    FILE *fp;

    for (j = 0u; j <= HI_FEATURE_COUNT; ++j) {
        max_abs[j] = 0.0f;
        max_rel[j] = 0.0f;
    }
    fp = fopen("data/processed/cwru_3200_conformance.bin", "rb");
    if (!fp || fread(&n_records, sizeof(uint32_t), 1u, fp) != 1u) {
        if (fp) fclose(fp);
        printf("ERROR: failed to open CWRU conformance binary\n");
        return 1;
    }

    for (idx = 0u; idx < n_records; ++idx) {
        char label[32], file[32];
        uint32_t win_idx = 0u;
        float py_hi = 0.0f;
        hi_features_t feats;
        hi_monitor_output_t out;
        int produced;

        if (fread(label, 1u, 32u, fp) != 32u ||
            fread(file, 1u, 32u, fp) != 32u ||
            fread(&win_idx, sizeof(uint32_t), 1u, fp) != 1u ||
            fread(window_buf, sizeof(float), HI_WINDOW_SIZE, fp) != HI_WINDOW_SIZE ||
            fread(py_feats, sizeof(float), HI_FEATURE_COUNT, fp) != HI_FEATURE_COUNT ||
            fread(&py_hi, sizeof(float), 1u, fp) != 1u) {
            fclose(fp);
            return 1;
        }
        label[31] = '\0';
        file[31] = '\0';

        if (strcmp(file, prev_file) != 0) {
            snprintf(prev_file, sizeof(prev_file), "%s", file);
            hi_stream_init_default(&st);
            if (win_idx != 0u) {
                fclose(fp);
                return 1;
            }
            produced = hi_stream_push(&st, window_buf, HI_WINDOW_SIZE);
        } else {
            produced = hi_stream_push(&st, &window_buf[HI_HOP_SIZE], HI_HOP_SIZE);
        }
        if (produced != 1 || hi_stream_latest(&st, &feats, &out) != 1) {
            fclose(fp);
            printf("ERROR: CWRU file=%s win=%u produced=%d\n", file, win_idx, produced);
            return 1;
        }
        for (j = 0u; j < HI_FEATURE_COUNT; ++j) {
            float d = fabsf(feats.value[j] - py_feats[j]);
            float r = d / fmaxf(fabsf(py_feats[j]), 1.0e-12f);
            if (d > max_abs[j]) max_abs[j] = d;
            if (r > max_rel[j]) max_rel[j] = r;
        }
        {
            float d = fabsf(out.raw_hi - py_hi);
            float r = d / fmaxf(fabsf(py_hi), 1.0e-12f);
            if (d > max_abs[HI_FEATURE_COUNT]) max_abs[HI_FEATURE_COUNT] = d;
            if (r > max_rel[HI_FEATURE_COUNT]) max_rel[HI_FEATURE_COUNT] = r;
        }
    }
    fclose(fp);

    printf("=== STREAMING CONFORMANCE: CWRU 3200 Hz (%u windows, 512-sample overlap hops) ===\n", n_records);
    printf("  Static RAM sizeof(hi_stream_t) = %zu bytes (%.2f KiB)\n",
           hi_stream_sizeof(), (double)hi_stream_sizeof() / 1024.0);
    for (j = 0u; j <= HI_FEATURE_COUNT; ++j) {
        int ok = (max_rel[j] <= REL_TOL);
        if (!ok) all_pass = 0;
        printf("  %-24s max_abs=%.6e max_rel=%.6e %s\n",
               FEAT_NAMES[j], (double)max_abs[j], (double)max_rel[j], ok ? "PASS" : "FAIL");
    }
    printf("  CWRU STREAM STATUS: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
#endif

