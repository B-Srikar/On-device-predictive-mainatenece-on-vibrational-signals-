# Stage 1 — CWRU Data Inspection Report

**Inspection date:** 2026-11-22  
**Raw-data path:** `data/raw/cwru/`  
**Assumed sampling rate:** 12,000 Hz (from `config.yaml`)

---

## Local-file availability — all 10 files present ✅

| File | MAT var (DE) | Samples | Duration | RPM | File # |
|------|-------------|--------:|---------:|----:|-------:|
| `normal_0.mat` | `X097_DE_time` | 243,938 | 20.33 s | 1796 | 97 |
| `IR007_0.mat` | `X105_DE_time` | 121,265 | 10.11 s | 1797 | 105 |
| `B007_0.mat` | `X118_DE_time` | 122,571 | 10.21 s | 1796 | 118 |
| `OR007@6_0.mat` | `X130_DE_time` | 121,991 | 10.17 s | 1796 | 130 |
| `IR014_0.mat` | `X169_DE_time` | 121,846 | 10.15 s | 1796 | 169 |
| `B014_0.mat` | `X185_DE_time` | 121,846 | 10.15 s | 1796 | 185 |
| `OR014@6_0.mat` | `X197_DE_time` | 121,846 | 10.15 s | 1796 | 197 |
| `IR021_0.mat` | `X209_DE_time` | 122,136 | 10.18 s | 1797 | 209 |
| `B021_0.mat` | `X222_DE_time` | 121,991 | 10.17 s | 1796 | 222 |
| `OR021@6_0.mat` | `X234_DE_time` | 122,426 | 10.20 s | 1796 | 234 |

> File numbers match the user's expected list exactly (97, 105, 118, 130, 169, 185, 197, 209, 222, 234).

---

## DE-signal statistics

| File | RMS | Peak | Kurtosis |
|------|----:|-----:|---------:|
| `normal_0.mat` | 0.07376 | 0.311 | 2.76 |
| `IR007_0.mat` | 0.29153 | 1.739 | 5.40 |
| `B007_0.mat` | 0.13923 | 0.607 | 2.99 |
| `OR007@6_0.mat` | 0.66951 | 3.630 | 7.65 |
| `IR014_0.mat` | 0.19783 | 2.011 | **21.96** |
| `B014_0.mat` | 0.15271 | 2.278 | **17.77** |
| `OR014@6_0.mat` | 0.10068 | 0.551 | 3.06 |
| `IR021_0.mat` | 0.52537 | 3.788 | 7.45 |
| `B021_0.mat` | 0.13561 | 1.660 | 8.55 |
| `OR021@6_0.mat` | 0.58317 | 6.653 | **21.01** |

Kurtosis for a Gaussian signal ≈ 3. Elevated values (>3) are characteristic of impulsive bearing faults.

---

## MAT-file variable inventory

All fault files contain 4 channels; the normal baseline has 3 (no BA channel):

| File | Variables |
|------|-----------|
| `normal_0.mat` | `X097_DE_time`, `X097_FE_time`, `X097RPM` |
| `IR007_0.mat` | `X105_DE_time`, `X105_FE_time`, `X105_BA_time`, `X105RPM` |
| `B007_0.mat` | `X118_DE_time`, `X118_FE_time`, `X118_BA_time`, `X118RPM` |
| `OR007@6_0.mat` | `X130_DE_time`, `X130_FE_time`, `X130_BA_time`, `X130RPM` |
| `IR014_0.mat` | `X169_DE_time`, `X169_FE_time`, `X169_BA_time`, `X169RPM` |
| `B014_0.mat` | `X185_DE_time`, `X185_FE_time`, `X185_BA_time`, `X185RPM` |
| `OR014@6_0.mat` | `X197_DE_time`, `X197_FE_time`, `X197_BA_time`, `X197RPM` |
| `IR021_0.mat` | `X209_DE_time`, `X209_FE_time`, `X209_BA_time`, `X209RPM` |
| `B021_0.mat` | `X222_DE_time`, `X222_FE_time`, `X222_BA_time`, `X222RPM` |
| `OR021@6_0.mat` | `X234_DE_time`, `X234_FE_time`, `X234_BA_time`, `X234RPM` |

> `preprocess.py` looks for the `*_DE_time` key — confirmed present in all files.

---

## Sampling-rate consistency check ✅

- **Normal baseline** (`97.mat`): 243,938 samples ÷ 12,000 Hz = **20.33 s** — consistent with 12 kHz.  
  (The CWRU Normal Baseline page does not explicitly print the rate, but 243,938/12000 = 20.33 s is a natural recording length; 243,938/48000 = 5.08 s would be atypically short for a baseline.)
- **All 12k fault files**: ~121,000–122,600 samples ÷ 12,000 Hz ≈ **10.1–10.2 s** — consistent with 12 kHz.
- **Conclusion:** `sampling_rate: 12000` in `config.yaml` is correct for all 10 files. No mismatch.

---

## Outputs

- `config.yaml` updated with `source_sampling_rates` for all 10 files at 12,000 Hz.
- All 10 `.mat` files present in `data/raw/cwru/`.
- Feature extraction (`preprocess.py`) has **not** started — awaiting your confirmation.

---

## IMS status (as of inspection)

Download of `4.+Bearings.zip` from `phm-datasets.s3.amazonaws.com` is **in progress**.  
Will unzip and configure `ims_snapshot_directory` and `ims_channel` in `config.yaml` once complete.
