# On-Device Vibration Health Index — Consolidated Results (`final_results.md`)

**Generated from:** clean end-to-end execution via [tools/run_all_stages.sh](tools/run_all_stages.sh)  
**Locked configuration:** [config.yaml](config.yaml)  
**Core C implementation:** [firmware/src/hi_core.c](firmware/src/hi_core.c), [firmware/include/hi_core.h](firmware/include/hi_core.h), [firmware/include/hi_config.h](firmware/include/hi_config.h)

---

## 1. Locked Method Summary

Every stage below uses the locked health index (HI) definition without post-hoc retuning ([config.yaml](config.yaml), [firmware/src/hi_core.c](firmware/src/hi_core.c)):

- **Framing:** `window_size = 1024`, `overlap = 0.5` (`hop = 512`), per-window mean removal.
- **Features per window:**
  - Time domain (4): `rms`, `peak_to_peak`, `crest_factor`, unbiased Pearson `kurtosis` (`fisher=False, bias=False`).
  - Frequency domain: FFT power spectrum log-band energies (`ln(max(sum_k |X_k|^2 / N^2, 1e-30))`) over contiguous bands `[0, 400, 800, 1200, 1600]` Hz at `3200 Hz` (`F = 8`), `+[1600, 3000, 6000]` Hz at `12,000 Hz` (`F = 10`), and `[0, 500, 1500, 3000, 6000, 10240]` Hz at `20,480 Hz` (`F = 9`).
- **Scoring & Tracking:** One-sided z-score `z_j = max((x_j - mu_j) / max(sigma_j, 1e-4), 0)`, compressed score `HI_raw = sqrt(mean(log1p(z_j)^2))`, EMA `alpha = 0.1` initialized with `HI_raw[0]`, `k = 3` (`mean + 3 * std` healthy holdout/calibration threshold), rolling `W = 12` linear-regression slope (gated by `HI_ema > baseline_mean`), `3`-consecutive-sample trigger, and machine-off gate skipping EMA/alarm updates when `rms < 0.10 * baseline_median_rms`.
- **Train/Test discipline:** All CWRU splits are by whole recording (zero cross-recording windows and 1-window gap before holdouts); IMS splits are chronological (`0..97` baseline, `98..195` calibration, `196+` monitor).

---

## 2. Dataset & Sampling-Rate Verification

- **Producing scripts / outputs:** [src/preprocess.py](src/preprocess.py) ([outputs/stage2_3200_summary.txt](outputs/stage2_3200_summary.txt), [outputs/stage2_12000_summary.txt](outputs/stage2_12000_summary.txt)), [src/stage3c_lolo_log1p.py](src/stage3c_lolo_log1p.py) ([outputs/stage3c_results.txt](outputs/stage3c_results.txt)), [src/stage5b_ims_diagnostics.py](src/stage5b_ims_diagnostics.py) ([outputs/stage5b_ims_diagnostics.txt](outputs/stage5b_ims_diagnostics.txt)).

| Dataset / Subset | Verified Structure & Rate Check | Key Finding | Output File |
| :--- | :--- | :--- | :--- |
| **CWRU `normal_1.mat` vs `normal_2.mat`** | `X098_DE_time` inside `99.mat` vs `98.mat` (`483,903` samples) | Exact bit match (`max \|diff\| = 0.000e+00`); `preprocess.py` loads `X099_DE_time` (`485,063` samples) for `normal_2` | [outputs/stage3c_results.txt](outputs/stage3c_results.txt) |
| **CWRU shaft-frequency check @ 48 kHz** | `X098` peak `29.56 Hz` (`1773.6 RPM`, ~1 HP); `X099` peak `29.19 Hz` (`1751.5 RPM`, ~2 HP) | Confirms `normal_0..3.mat` are `48,000 Hz` (`118, 235, 235, 236` windows @ 12k; `30, 62, 62, 62` @ 3200 Hz); 9 fault `.mat` files are `12,000 Hz` (`235–238` @ 12k; `62` @ 3200 Hz) | [outputs/stage3c_results.txt](outputs/stage3c_results.txt), [outputs/stage2_3200_summary.txt](outputs/stage2_3200_summary.txt) |
| **IMS Set No. 2 (`2nd_test`)** | `984` chronological snapshots (`2004.02.12.10.32.39` to `2004.02.19.06.22.39`, `163.83 h`, `20,480` samples @ `20,480 Hz`, `39` windows/snapshot median) | Ch 1 (`idx 0`, Bearing 1) failed outer race; Ch 2–4 (`idx 1..3`) non-failing. Snapshots `982–983` are shaft-stopped (`RMS 0.0012–0.0040`, near-zero), excluded in Stage 4–8 (`982` snapshots `0..981`, `163.50 h`) | [outputs/stage5_ims_results.txt](outputs/stage5_ims_results.txt), [outputs/stage5b_ims_diagnostics.txt](outputs/stage5b_ims_diagnostics.txt) |

---

## 3. CWRU Leave-One-Load-Out (LOLO) — `log1p_HI` vs Baselines (`B1–B3`)

- **Producing scripts / outputs:** [src/stage4_baselines.py](src/stage4_baselines.py) ([outputs/stage4_results.txt](outputs/stage4_results.txt), [outputs/stage4_cwru_results.csv](outputs/stage4_cwru_results.csv)) and [src/stage3c_lolo_log1p.py](src/stage3c_lolo_log1p.py) ([outputs/stage3c_results.txt](outputs/stage3c_results.txt)).
- **Associated figures:** [outputs/plots/hi_per_window_12000hz.png](outputs/plots/hi_per_window_12000hz.png), [outputs/plots/hi_per_window_3200hz.png](outputs/plots/hi_per_window_3200hz.png).

### Table 3a — 4-Fold Mean & Worst-Case Across All 4 LOLO Folds (12 kHz and 3200 Hz)

*(Thresholds = `mean + 3 std` of pooled healthy holdout; 1%/5% FPR calibrated across healthy holdout + unseen load. `IForest` fixed at `n_estimators=100, random_state=0`, `max_samples=min(256, n_base)`.)*

| Rate | Detector | Model Size (4 folds) | `FPR_H` (mean) | `FPR_U` (mean / max) | `AUROC` (mean / min) | `Rec@thr` (mean / min) | `Rec@1%FPR` (mean) | `Rec@5%FPR` (mean) | Source |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **12 kHz** | `B1_rms` | 3 floats | 0.001 | 0.250 / 1.000 | 1.0000 / 0.9999 | 0.997 / 0.966 | 1.000 | 1.000 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **12 kHz** | `B1_kurtosis` | 3 floats | 0.010 | 0.010 / 0.034 | 0.8743 / 0.4127 | 0.714 / 0.004 | 0.718 | 0.757 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **12 kHz** | `B2_any_feature` | 21 floats | 0.015 | 0.263 / 1.000 | 0.9999 / 0.9960 | 1.000 / 0.992 | 0.999 | 1.000 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **12 kHz** | `B3_iforest` | 13,744–14,080 nodes (mean 13,809.0) + 21 floats | 0.013 | 0.253 / 0.975 | 0.9975 / 0.9624 | 0.996 / 0.877 | 0.977 | 0.988 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **12 kHz** | **`log1p_HI`** | **21 floats** | **0.002** | **0.049 / 0.186** | **1.0000 / 1.0000** | **1.000 / 1.000** | **1.000** | **1.000** | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **3200 Hz** | `B1_rms` | 3 floats | 0.000 | 0.000 / 0.000 | 0.8185 / 0.2079 | 0.606 / 0.000 | 0.665 | 0.683 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **3200 Hz** | `B1_kurtosis` | 3 floats | 0.000 | 0.017 / 0.067 | 0.8574 / 0.6367 | 0.408 / 0.065 | 0.474 | 0.602 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **3200 Hz** | `B2_any_feature` | 17 floats | 0.011 | 0.116 / 0.433 | 0.9796 / 0.8587 | 0.823 / 0.113 | 0.809 | 0.893 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **3200 Hz** | `B3_iforest` | 7,478–12,064 nodes (mean 8,706.0) + 17 floats | 0.005 | 0.150 / 0.600 | 0.9576 / 0.7195 | 0.511 / 0.000 | 0.685 | 0.779 | [outputs/stage4_results.txt](outputs/stage4_results.txt) |
| **3200 Hz** | **`log1p_HI`** | **17 floats** | **0.000** | **0.020 / 0.081** | **0.9705 / 0.7648** | **0.732 / 0.097** | **0.789** | **0.852** | [outputs/stage4_results.txt](outputs/stage4_results.txt) |

### Table 3b — `log1p_HI` Per-Fold Detail & Compression Reference ([outputs/stage3c_results.txt](outputs/stage3c_results.txt), [outputs/stage4_results.txt](outputs/stage4_results.txt))

| Rate | Unseen Load (`B/H/U`) | `Thr` (`log1p`) | `FPR_H` | `FPR_U` (`n`) | `MinAUC` | `MeanAUC` | `MinRec@thr` | `MeanRec@thr` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12 kHz** | `normal_3` (410/175/236) | 0.9579 | 0.000 | 0.008 (2/236) | 1.0000 | 1.0000 | 1.000 | 1.000 |
| **12 kHz** | `normal_2` (411/175/235) | 0.9652 | 0.000 | 0.000 (0/235) | 1.0000 | 1.0000 | 1.000 | 1.000 |
| **12 kHz** | `normal_1` (411/175/235) | 0.9898 | 0.000 | 0.000 (0/235) | 1.0000 | 1.0000 | 1.000 | 1.000 |
| **12 kHz** | **`normal` (493/210/118)** | 1.0608 | 0.010 | **0.186 (22/118)** | 1.0000 | 1.0000 | 1.000 | 1.000 |
| **3200 Hz** | `normal_3` (107/44/62) | 1.0119 | 0.000 | 0.000 (0/62) | 0.9002 | 0.9799 | 0.161 | 0.710 |
| **3200 Hz** | `normal_2` (107/44/62) | 0.9348 | 0.000 | 0.000 (0/62) | 0.9168 | 0.9817 | 0.161 | 0.711 |
| **3200 Hz** | `normal_1` (107/44/62) | 0.8363 | 0.000 | **0.081 (5/62)** | 0.7648 | 0.9539 | 0.371 | 0.806 |
| **3200 Hz** | `normal` (129/54/30) | 1.1339 | 0.000 | 0.000 (0/30) | 0.8564 | 0.9663 | 0.097 | 0.701 |

- **Compression effect ([outputs/stage3c_results.txt](outputs/stage3c_results.txt)):** Under hard `Z_CAP=6` on the `normal_3`-unseen fold at `12 kHz`, up to `81.2%` of fault features hit `z >= 6` (`32.2%–81.2%`, `max_z = 346.9`), compressing all fault medians into `[3.542, 5.587]`; `log1p(z)` eliminates the hard saturation wall (`[1.339, 3.385]`) while uncapped reference spans `[3.963, 84.753]`.

### Table 3c — Per-Class 4-Fold Mean at `3200 Hz` Showing Weak Fault Degradation ([outputs/stage4_results.txt](outputs/stage4_results.txt))

*(Entries formatted as `AUROC / Rec@thr / Rec@1%FPR / Rec@5%FPR`. At `12 kHz`, `log1p_HI` achieves `1.000/1.00/1.00/1.00` on all 9 classes.)*

| Fault Class | `B1_rms` | `B1_kurtosis` | `B2_any_feature` | `B3_iforest` | `log1p_HI` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`ball_007` (weak @ 3200 Hz)** | 0.527/0.00/0.04/0.06 | 0.734/0.12/0.16/0.30 | 0.965/0.57/0.54/0.80 | 0.903/0.13/0.33/0.52 | **0.943/0.25/0.47/0.69** |
| `ball_014` | 1.000/0.98/1.00/1.00 | 0.822/0.48/0.54/0.60 | 0.999/1.00/0.98/1.00 | 0.979/0.61/0.81/0.90 | 1.000/0.99/1.00/1.00 |
| **`ball_021` (weak @ 3200 Hz)** | 0.338/0.00/0.03/0.03 | 0.756/0.08/0.17/0.33 | 0.944/0.34/0.31/0.67 | 0.897/0.14/0.30/0.41 | **0.937/0.21/0.29/0.49** |
| `inner_race_007` | 1.000/1.00/1.00/1.00 | 0.998/0.98/0.94/1.00 | 1.000/1.00/1.00/1.00 | 0.997/0.98/1.00/1.00 | 1.000/1.00/1.00/1.00 |
| `inner_race_014` | 0.938/0.44/0.75/0.83 | 0.942/0.55/0.64/0.83 | 0.990/0.89/0.85/0.91 | 0.982/0.69/0.85/0.92 | 0.994/0.90/0.96/0.96 |
| `inner_race_021` | 1.000/1.00/1.00/1.00 | 0.939/0.17/0.38/0.65 | 1.000/1.00/1.00/1.00 | 0.988/0.33/0.73/0.92 | 1.000/1.00/1.00/1.00 |
| `outer_race_007` | 1.000/1.00/1.00/1.00 | 0.835/0.19/0.31/0.49 | 1.000/1.00/1.00/1.00 | 0.989/0.55/0.86/0.92 | 1.000/1.00/1.00/1.00 |
| **`outer_race_014` (weak @ 3200 Hz)** | 0.564/0.04/0.17/0.22 | 0.691/0.09/0.13/0.22 | 0.918/0.60/0.58/0.67 | 0.883/0.17/0.28/0.43 | **0.860/0.24/0.38/0.52** |
| `outer_race_021` | 1.000/1.00/1.00/1.00 | 1.000/1.00/1.00/1.00 | 1.000/1.00/1.00/1.00 | 1.000/1.00/1.00/1.00 | 1.000/1.00/1.00/1.00 |

---

## 4. IMS Set No. 2 Run-to-Failure Lead Times & Sensitivity

- **Producing scripts / outputs:** [src/stage4_baselines.py](src/stage4_baselines.py) ([outputs/stage4_results.txt](outputs/stage4_results.txt), [outputs/stage4_ims_results.csv](outputs/stage4_ims_results.csv)) and [src/stage5_ims_trend.py](src/stage5_ims_trend.py) ([outputs/stage5_ims_results.txt](outputs/stage5_ims_results.txt)).
- **Associated figure:** [outputs/plots/ims_2nd_test_hi_trend.png](outputs/plots/ims_2nd_test_hi_trend.png).
- **Note on end-of-test span:** Stage 5 includes all `984` snapshots (`0..983`, ending `163.83 h`, giving lead `76.00 h` / `74.17 h`). Stage 4–8 exclude the two shaft-stopped final snapshots `982–983` (`982` snapshots `0..981`, ending `163.50 h`), giving `75.67 h` / `73.83 h` (`-0.33 h`) at identical alarm indices `527` and `538`.

### Table 4a — Bearing 1 (Channel 1, Failed Outer Race) All 7 Stage 4 Detectors ([outputs/stage4_results.txt](outputs/stage4_results.txt))

*(Snapshots kept = `982` (`0..981`); baseline `0..97`, calibration `98..195`, presumed healthy `196..491` (`n=296`), monitor `196..981`; `3`-consecutive-snapshot rule; `rms_off_thr = 0.00775`, `0` gated in `0..981`, `[982, 983]` gated if retained; `B3_iforest` = `6736` nodes.)*

| Detector | Threshold | `Cal (98–195)` | `Hlth (196–491)` | `Post (196–981)` | `FirstIdx (>=196)` | Timestamp | Lead Time (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `B1_rms` | 1.78402 | 0 | 0 | 448 | 534 | `2004.02.16.03.32.39` | 74.50 |
| `B1_kurtosis` | 2.75867 | 0 | 0 | 290 | 649 | `2004.02.16.22.42.39` | 55.33 |
| `B2_any_feature` | 3.14935 | 0 | 0 | 448 | 534 | `2004.02.16.03.32.39` | 74.50 |
| `B3_iforest` | 0.10013 | 0 | 0 | 369 | 561 | `2004.02.16.08.02.39` | 70.00 |
| `HI_raw` | 0.83792 | 0 | 0 | 445 | 534 | `2004.02.16.03.32.39` | 74.50 |
| **`HI_ema`** | **0.39277** | **0** | **0** | **455** | **527** | `2004.02.16.02.22.39` | **75.67** |
| **`HI_ema_slope`** | **0.01779** | **0** | **0** | **68** | **538** | `2004.02.16.04.12.39` | **73.83** *(later than `HI_ema` by 1.83 h)* |

### Table 4b — Stage 5 Sensitivity Table (`alpha` × `baseline_fraction`, `cal = 10%` fixed, all `984` snapshots) ([outputs/stage5_ims_results.txt](outputs/stage5_ims_results.txt))

*(`*` marks the locked primary [config.yaml](config.yaml) setting. Stage 5 counts `base` + `cal` together in `Hlth(F/S)`; note `cal=0` everywhere and base alarms arise because Stage 5 threshold was calibrated on `98..195`.)*

| `alpha` | `Base%` | B1 `Hlth(F/S)` (`0..195`) | B1 `Fixed (idx / lead)` | B1 `Slope (idx / lead)` | B1 `rho` (full `0..983`) | B4 `Hlth(F/S)` | B4 `Post(F/S)` | B4 `rho` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.05 | 5% | 43 / 0 | 535 / 74.7 h | 538 / 74.2 h | 0.7730 | 0 / 0 | 443 / 141 | 0.8512 |
| 0.10 | 5% | 38 / 0 | 532 / 75.2 h | 537 / 74.3 h | 0.7924 | 0 / 0 | 413 / 53 | 0.8300 |
| 0.20 | 5% | 14 / 0 | 534 / 74.8 h | 538 / 74.2 h | 0.8026 | 0 / 0 | 387 / 25 | 0.8136 |
| 0.05 | 10% | 65 / 0 | 526 / 76.2 h | 537 / 74.3 h | 0.7728 | 20 / 0 | 448 / 130 | 0.8356 |
| **0.10** | **10%\*** | **43 / 0** | **527 / 76.0 h\*** | **538 / 74.2 h\*** | **0.7894** | **14 / 0** | **423 / 45** | **0.8182** |
| 0.20 | 10% | 22 / 0 | 534 / 74.8 h | 539 / 74.0 h | 0.7983 | 1 / 0 | 394 / 26 | 0.8055 |
| 0.05 | 15% | 99 / 0 | 461 / 87.0 h | 534 / 74.8 h | 0.7688 | 12 / 0 | 444 / 95 | 0.8281 |
| 0.10 | 15% | 67 / 0 | 461 / 87.0 h | 536 / 74.5 h | 0.7848 | 2 / 0 | 408 / 39 | 0.8142 |
| 0.20 | 15% | 41 / 0 | 526 / 76.2 h | 537 / 74.3 h | 0.7933 | 0 / 0 | 382 / 24 | 0.8038 |

---

## 5. IMS All-Four-Bearing Diagnostics

- **Producing scripts / outputs:** [src/stage5b_ims_diagnostics.py](src/stage5b_ims_diagnostics.py) ([outputs/stage5b_ims_diagnostics.txt](outputs/stage5b_ims_diagnostics.txt)) and [src/stage4_baselines.py](src/stage4_baselines.py) ([outputs/stage4_results.txt](outputs/stage4_results.txt)).
- **Associated figures:** [outputs/plots/ims_stage5b_all4_hi_with_last.png](outputs/plots/ims_stage5b_all4_hi_with_last.png), [outputs/plots/ims_stage5b_all4_hi_without_last.png](outputs/plots/ims_stage5b_all4_hi_without_last.png), and exploratory kurtosis plot [outputs/plots/ims_stage4_exploratory_kurtosis_z.png](outputs/plots/ims_stage4_exploratory_kurtosis_z.png).

### Table 5a — All Four Bearings Through Locked Pipeline & Baselines ([outputs/stage4_results.txt](outputs/stage4_results.txt), [outputs/stage5b_ims_diagnostics.txt](outputs/stage5b_ims_diagnostics.txt))

*(All 4 bearings show `0` false alarms in `Cal (98–195)` and `0` in `Hlth (196–491)` in Stage 4 across all detectors; `MaxHI` & `EndHI` are from Stage 5b `0..983`.)*

| Bearing (Channel) | `HI_ema` FirstIdx / Lead (`0..981`) | `HI_ema_slope` FirstIdx / Lead | `B1_rms` FirstIdx | `B1_kurtosis` FirstIdx | `B2_any` FirstIdx | `B3_iforest` FirstIdx | `MaxHI` / `EndHI` (Stage 5b) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bearing 1 (Ch 1 — Failed OR)** | **527 / 75.67 h** | **538 / 73.83 h** | 534 | **649** | 534 | 561 | 4.083 / 3.416 |
| Bearing 2 (Ch 2 — Normal) | 710 / 45.17 h | 706 / 45.83 h | 705 | none | 708 | 781 | 2.770 / 2.244 |
| Bearing 3 (Ch 3 — Normal) | 750 / 38.50 h | none | 875 | none | 750 | 887 | 1.890 / 1.531 |
| **Bearing 4 (Ch 4 — Normal control)** | **539 / 73.67 h** | **620 / 60.17 h** | 657 | 918 | 649 | 661 | 3.259 / 2.649 |

### Table 5b — Bearing 1 vs Bearing 4 Feature Decomposition, Correlation & Exploratory Diagnostics ([outputs/stage5b_ims_diagnostics.txt](outputs/stage5b_ims_diagnostics.txt), [outputs/stage4_results.txt](outputs/stage4_results.txt))

| Diagnostic | Bearing 1 (Failed OR) | Bearing 4 (Control) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Feature share @ snap 450 (`t=75.00h`)** | `HI_ema=0.2914`: time `99.7%`, FFT `0.3%` (`crest_factor 79.5%`) | `HI_ema=0.3715`: time `19.2%`, FFT `80.8%` | Pre-fault fluctuations only |
| **Feature share @ snap 600 (`t=100.00h`)** | `HI_ema=1.3421`: time `52.4%`, FFT `47.6%` (`rms 36.7%`, `3–6k 40.9%`) | `HI_ema=0.4896`: time `45.6%`, FFT `54.4%` (`3–6k 31.0%`) | Initial `3–6 kHz` resonance + RMS rise |
| **Feature share @ snap 800 (`t=133.33h`)** | `HI_ema=2.2404`: time `42.5%`, FFT `57.5%` (`1.5–3k 19.8%`, `3–6k 16.9%`, `6–10.24k 13.0%`) | `HI_ema=1.4376`: time `28.9%`, FFT `71.1%` (`0.5–1.5k 20.5%`, `1.5–3k 33.3%`, `6–10.24k 10.4%`) | **Broadband** multi-band energy growth |
| **Feature share @ snap 950 (`t=158.33h`)** | `HI_ema=3.1387`: time `40.9%`, FFT `59.1%` (`0.5–10.24k` all `>=7.6%`, `kurt 4.4%`) | `HI_ema=2.6418`: time `38.5%`, FFT `61.5%` (`0.5–1.5k 20.4%`, `0–0.5k 15.9%`, `kurt 0.0%`) | Broadband in both; impulsiveness (`kurtosis` `z=6.62`) concentrated in Ch 1 |
| **B1–B4 correlation (`196..983`, `n=788`)** | `rms`: `r=0.9211, rho=0.8979`; `HI_raw`: `r=0.9278, rho=0.8978`; **`HI_ema`: `r=0.9414, rho=0.9427`** | — | Strong structural housing coupling across shafts |
| **B1–B4 correlation (`525..983`, `n=459`)** | `rms`: `r=0.9106, rho=0.9366`; `HI_raw`: `r=0.9412, rho=0.9591`; **`HI_ema`: `r=0.9483, rho=0.9824`** | — | Near-collinear post-onset monotonic energy transmission |
| **Spearman `rho` vs time** | `196..983`: EMA `0.9300` (raw `0.9060`); `525..983`: EMA `0.9494` (raw `0.9051`) | `196..983`: EMA `0.9100` (raw `0.8499`); `525..983`: EMA `0.9568` (raw `0.9135`) | Shows strong monotone trend post-onset; **does not isolate which bearing failed** |
| **Raw tail inspection (`979..983`) & `0..59`** | `979`: Ch 1 `[-5.000, +4.998]`, `RMS=0.7250, Kurt=15.58`; `982–983`: **ANOMALOUS** (`RMS 0.0012–0.0040`); dropping `983`: Ch 1 `end_HI 3.4157 -> 3.7952`; `0..59` normal (`len 20480`) | `982–983` also near-zero stopped | Led to pre-specified exclusion of `982–983` & RMS gate in Stage 4+ |
| **[Exploratory post-hoc] Cross-bearing relative `z_rel` (Stage 5b)** | `med_z(525+) = 19.40`, `max = 28.09`, `frac>3 = 0.993` | `med_z(525+) = 5.99`, `max = 8.65` (`Ch 2/3 < 0`) | Exploratory only; Ch 1 stands out relative to peers |
| **[Exploratory post-hoc] Full kurtosis `z` (Stage 4)** | `525–981` `med/p95/max = 5.93 / 26.44 / 161.58`, `end(981) = 51.53` | `525–981` `med/p95/max = 0.04 / 4.57 / 18.09`, `end(981) = 11.16` | Exploratory only; localized high-kurtosis impacting on Ch 1 |

---

## 6. C Implementation Conformance, Edge Cases & No-`malloc` Verification

- **Producing scripts / tests / outputs:** [firmware/src/hi_core.c](firmware/src/hi_core.c), [firmware/tests/test_health_index.c](firmware/tests/test_health_index.c) ([outputs/stage6_c_conformance.txt](outputs/stage6_c_conformance.txt)), [firmware/tests/test_stage6b_cwru.c](firmware/tests/test_stage6b_cwru.c) ([outputs/stage6b_cwru_conformance.txt](outputs/stage6b_cwru_conformance.txt)), [firmware/tests/test_stage6b_ims.c](firmware/tests/test_stage6b_ims.c) ([outputs/stage6b_ims_conformance.txt](outputs/stage6b_ims_conformance.txt)), [firmware/tests/test_stage7_stream_conformance.c](firmware/tests/test_stage7_stream_conformance.c) ([outputs/stage7_stream_conformance.txt](outputs/stage7_stream_conformance.txt)).
- **Compiler hygiene & memory safety:** All C binaries compile cleanly with `-std=c99 -O2 -Wall -Wextra -Werror` and execute cleanly under `-fsanitize=address,undefined` (`ASan` + `UBSan`). `grep -rn "malloc\|calloc\|realloc\|free" firmware/src/ firmware/include/` returns zero matches ([tools/run_all_stages.sh](tools/run_all_stages.sh)).

### Table 6 — C vs Python Conformance Matrix

| Test Suite | Scope | Max Abs Diff | Max Rel Diff | Boolean / Alarm Parity | Status & Notes | Output File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 6 Batch CWRU (24 windows)** | 2 windows/class across 12 groups @ `3200 Hz` | `kurtosis`: `2.57e-5`; `log1p_HI`: `1.55e-6` | `kurtosis`: `3.44e-6`; `log1p_HI`: `2.55e-6` | — | **PASS** (`<= 1e-3` rel tol) | [outputs/stage6_c_conformance.txt](outputs/stage6_c_conformance.txt) |
| **Stage 6 Synthetic 200-win Sequence** | Concatenated CWRU test constants (`100` healthy `normal_0..2` + `1` synthetic zero gate at `idx 100` + `99` fault `IR007`/`OR021` windows) | `EMA`: `9.54e-7`; `Slope`: `4.47e-8` | `EMA`: `7.46e-7`; `Slope`: `4.71e-5` | `200/200` on `gated`, `slope_valid`, `fixed_alarm`, `slope_alarm` | **PASS**; test alarms at `fixed=106, slope=105` are **synthetic step-transition test constants, NOT real lead-time results** | [outputs/stage6_c_conformance.txt](outputs/stage6_c_conformance.txt), [tools/export_c_test_vectors.py](tools/export_c_test_vectors.py) |
| **Stage 6b Batch CWRU (774 windows)** | Every window in `features_3200.csv` (`normal_0..3` + all 9 faults) | `kurtosis`: `2.57e-5`; `log1p_HI`: `7.79e-6` | `kurtosis`: `3.80e-6`; `log1p_HI`: `8.74e-5` | Worst 3 rel errors in low-HI `normal_2` (`win 21`: `py=0.089212` vs `c=0.089205`) | **PASS** | [outputs/stage6b_cwru_conformance.txt](outputs/stage6b_cwru_conformance.txt) |
| **Stage 6b CWRU Edge Cases (4 cases)** | `constant_0.3`, `all_zero`, `clipped_pm5`, `one_nan` | — | — | `constant_0.3` & `all_zero` gated (`HI=0`); `clipped_pm5` finite (`HI=4.015`); `one_nan` rejected (`rc=-1`, monitor untouched) | **PASS** (no crash / UB) | [outputs/stage6b_cwru_conformance.txt](outputs/stage6b_cwru_conformance.txt) |
| **Stage 6b Batch IMS (982 snapshots)** | IMS Ch 1 (`0..981`, `39`-window median per snapshot @ `20,480 Hz`) | `kurtosis`: `3.81e-5`; `HI_raw`: `4.84e-5`; `HI_ema`: `5.30e-6`; `slope`: `4.70e-7` | Features `<= 7.33e-6`; `HI_ema`: `2.27e-5`; **`HI_raw`: `1.46e-3`** | `982/982` exact (`slope_valid`, `fixed_alarm`, `slope_alarm`); `196..491` = `0/0`; first `=(527, 538)` | **Flagged FAIL** on `HI_raw` strict `1e-3` rel check at near-zero baseline snapshot `idx 86` (`abs 4.84e-5`, `HI_ema` & all alarms match cleanly) | [outputs/stage6b_ims_conformance.txt](outputs/stage6b_ims_conformance.txt) |
| **Stage 7 Streaming API (CWRU + IMS)** | `hi_stream_push` ring buffer (`512` hop): `774` CWRU windows & `982` IMS snapshots (`39`-win scratch) | Identical to Stage 6b (`CWRU HI 7.79e-6`; `IMS raw 4.84e-5 / ema 5.30e-6`) | `CWRU <= 8.74e-5`; `IMS ema 2.27e-5` | `982/982` IMS exact (`first=(527, 538)`, `196..491: 0/0`) | **PASS** (IMS `idx 86` near-zero `HI_raw` noted) | [outputs/stage7_stream_conformance.txt](outputs/stage7_stream_conformance.txt) |

---

## 7. Static RAM, Flash Constants, Host Code Size & ESP32 Estimate

- **Producing scripts / outputs:** `size` on compiled objects ([outputs/stage6b_host_size.txt](outputs/stage6b_host_size.txt)), `sizeof` probes ([outputs/stage6_c_conformance.txt](outputs/stage6_c_conformance.txt), [outputs/stage6b_cwru_conformance.txt](outputs/stage6b_cwru_conformance.txt), [outputs/stage6b_ims_conformance.txt](outputs/stage6b_ims_conformance.txt), [outputs/stage7_stream_conformance.txt](outputs/stage7_stream_conformance.txt)).

| Configuration | Static RAM (`sizeof`) | Flash Constants (`sizeof`) | Host Object Code Size (`macOS arm64` **PROXY — NOT ESP32 flash**) |
| :--- | :--- | :--- | :--- |
| **CWRU `3200 Hz` Batch** | `12,420 B` (`12.13 KiB`: `window_buf` + `fft_real` + `fft_imag` = `3 x 4096 B` + `132 B` state; aliasing `window_buf`/`fft_real` saves `4,096 B` with `0.00e+00` diff, unapplied) | `184 B` (`120 B` deduplicated: `HI_BANDS 32 B`, `MEAN 32 B`, `STD 32 B`, `hi_monitor_config_t 88 B`) | `__TEXT = 2,968 B`, `__DATA = 0 B`, `others = 288 B`, total `3,256 B` |
| **CWRU `3200 Hz` Streaming (`hi_stream_t`)** | **`12,536 B` (`12.24 KiB`)** (`ring[1024] + fft_real[1024] + fft_imag[1024] + config/monitor/output`) | Same (`120 B` deduplicated) | — |
| **IMS `20,480 Hz` Batch** | `91,652 B` (`snap[20480] = 81,920 B` + `8,192 B` FFT + `win_scratch[39][9] = 1,404 B` + `136 B` state) | `HI_BANDS 40 B`, `MEAN 36 B`, `STD 36 B`, `cfg 96 B` (`172 B` deduplicated) | `__TEXT = 4,672 B`, `__DATA = 0 B`, `others = 320 B`, total `4,992 B` |
| **IMS `20,480 Hz` Streaming (`hi_stream_t`)** | **`13,956 B` (`13.63 KiB`)** — `win_scratch[39][9]` eliminates the `81,920 B` snapshot buffer | Same (`172 B` deduplicated) | — |

- **Analytical ESP32 timing estimate *(analytical bound — NOT measured on ESP32 hardware)*:** Per 1024-sample window, `hi_compute_features` performs two linear passes over 1024 floats plus a radix-2 1024-point complex Cooley-Tukey FFT (`10 x 512 = 5,120` butterflies, ~100k–200k FP ops; note [firmware/src/hi_core.c](firmware/src/hi_core.c) currently calls `cosf`/`sinf` inside the `k` loop across `1,023` twiddles per FFT rather than a lookup table). On the Apple `arm64` laptop host benchmark ([outputs/stage6_c_conformance.txt](outputs/stage6_c_conformance.txt)), batch execution takes `17.60 us` mean (`44.00 us` p99) per `0.320 s` window (`18,181x` real-time). On a 240 MHz ESP32 with single-precision hardware FPU, `3200 Hz` (`160 ms` hop deadline -> ~`38.4M` cycles available per hop) leaves orders-of-magnitude margin even with un-cached `cosf`/`sinf`; at `20,480 Hz` (`25 ms` hop = `6.0M` cycles per hop), precomputing the 512-entry twiddle table in flash would still be recommended to guarantee deterministic headroom without hardware benchmark verification.

---

## 8. Streaming Simulator Timing & Scenarios *(Laptop Simulation — NOT ESP32 Measurements)*

- **Producing script / outputs:** [tools/stream_simulator.py](tools/stream_simulator.py), [outputs/stage7_simulator_results.txt](outputs/stage7_simulator_results.txt).
- **Associated figures:** [outputs/plots/stage7_cwru_ball007.png](outputs/plots/stage7_cwru_ball007.png) ([.gif](outputs/plots/stage7_cwru_ball007.gif)), [outputs/plots/stage7_cwru_outer_race_021.png](outputs/plots/stage7_cwru_outer_race_021.png) ([.gif](outputs/plots/stage7_cwru_outer_race_021.gif)), [outputs/plots/stage7_ims_ch1.png](outputs/plots/stage7_ims_ch1.png) ([.gif](outputs/plots/stage7_ims_ch1.gif)), [outputs/plots/stage7_dropout.png](outputs/plots/stage7_dropout.png) ([.gif](outputs/plots/stage7_dropout.gif)).
- **CWRU slope threshold source:** `HI_SLOPE_THRESHOLD = 0.021213` computed strictly from pooled healthy holdout `normal_0-2` rolling 12-window EMA slopes (`n = 14` finite slopes, `mean + 3 * std`) in [tools/export_baseline_header.py](tools/export_baseline_header.py).

| Scenario | Mode & Wall Time | Hop Latency `mean/p99/max` (`us`) | Output-Window Latency `mean/p99/max` (`us`) | Deadline (`us`) / Misses / Drops | First Alarms & Behaviour | C vs Python Parity |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **A1: CWRU `normal_3` (`62w`) -> `ball_007` (`62w`)** *(fault injection from separate recordings)* | `fast` (`0.002s`) / `paced 10x` (`2.019s`) | `15.7 / 18.1 / 18.5` (`fast`), `82.2 / 134.5 / 144.4` (`10x`) | `15.8 / 18.1 / 18.5` (`fast`) | `160,000` (`fast`), `16,000` (`10x`) / **`0/126`** / `0` | **`first(fix=none, slp=none)`** (`normal_3`: `0/0`; `ball_007 max_ema=0.879 < thr 1.012`) | `raw 4.50e-6`, `ema 8.52e-7`, `125/125` exact booleans |
| **A2: CWRU `normal_3` (`62w`) -> `outer_race_021` (`62w`)** *(fault injection from separate recordings)* | `fast` (`0.002s`) / `paced 10x` (`2.020s`) | `15.7 / 17.9 / 23.0` (`fast`), `78.8 / 132.8 / 136.4` (`10x`) | `15.8 / 17.9 / 23.0` (`fast`) | `160,000` / `16,000` / **`0/126`** / `0` | `normal_3`: `0/0`; `OR021`: **`first(fix=66, slp=65)`**, `fixed=59/62`, `max_ema=3.033` | `raw 4.50e-6`, `ema 1.13e-6`, `125/125` exact booleans |
| **B: IMS Ch 1 (`982` snaps = `982s`; + `5`-snap `1x`)** | `fast` (`0.646s`), `paced 100x` (`9.820s`), `1x` (`5.005s`) | `15.7 / 21.2 / 105.3` (`fast`), `29.5 / 63.5 / 166.1` (`100x`), `62.8 / 151.2 / 277.0` (`1x`) | `19.7 / 23.7 / 71.4` (`fast`), `85.4 / 113.0 / 113.6` (`1x`) | `25,000` / `250` (`100x`) -> **`0/39,280`** (`fast` & `100x`) & `0/200` (`1x`) / `0` drop | **`0..195` baseline/cal not evaluated**; **`>=196: fixed=527, slope=538`**; `196..491`: `0/0` | `raw 4.84e-5`, `ema 5.30e-6`, `982/982` exact booleans |
| **C: Dropout (`30` `n3` + `20` zero + `62` `OR021`)** | `fast` (`0.002s`) / `paced 10x` (`1.843s`) | `15.8 / 17.9 / 18.3` (`fast`), `65.5 / 163.5 / 528.9` (`10x`) | `15.9 / 17.9 / 18.3` (`fast`) | `160,000` / `16,000` / **`0/115`** / `0` | **`20` gated windows (`31..50`)**, EMA frozen at `0.5506`, `0` dropout alarms; `win 30` straddles seam (`512` signal + `512` zero, not gated); post-dropout `OR021` `first fixed=54` | `raw 9.46e-6`, `ema 9.89e-7`, `114/114` exact booleans |

---

## 9. Stage 8 Simulated Sensor Resolution & Noise Sweep *(Simulated — NOT Measured Sensor)*

- **Producing script / outputs:** [src/stage8_sensor_sweep.py](src/stage8_sensor_sweep.py), [outputs/stage8_results.txt](outputs/stage8_results.txt), [outputs/stage8_cwru_sweep.csv](outputs/stage8_cwru_sweep.csv), [outputs/stage8_ims_sweep.csv](outputs/stage8_ims_sweep.csv).
- **Associated figures:** [outputs/plots/stage8_cwru_heatmap.png](outputs/plots/stage8_cwru_heatmap.png), [outputs/plots/stage8_ims_heatmap.png](outputs/plots/stage8_ims_heatmap.png).
- **Explicit degradation point:** For CWRU, Gaussian noise + mid-tread uniform quantization are applied directly on the **3200 Hz resampled stream** (`cwru_3200_conformance.bin`); for IMS Ch 1, degradation is applied on the **native 20,480 Hz** waveforms (`ims_ch1_waveforms.bin`) in multiples of baseline RMS. Baseline `mean`, `std`, and thresholds are refitted on the degraded signal (`10` seeds `0..9` when noise > 0). Batch verifiers match Python reference (`CWRU max_diff = 7.28e-7`, `IMS max_diff = 4.88e-7`).

### Table 9a — CWRU `3200 Hz` Stream Sweep Across 4 LOLO Folds (`10` seeds for `Noise > 0`, `mean ± std`) ([outputs/stage8_results.txt](outputs/stage8_results.txt))

| LSB (`mg`) | Noise (`mg`) | `MeanAUC` | `MinAUC` | `Rec@thr` | `Rec@1%FPR` | `FPR_U` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ref | 0.0 | 0.9705 ± 0.0000 | 0.7648 ± 0.0000 | 0.732 ± 0.000 | 0.789 ± 0.000 | 0.020 ± 0.000 |
| 3.9 | 0.0 | 0.9705 ± 0.0000 | 0.7673 ± 0.0000 | 0.732 ± 0.000 | 0.786 ± 0.000 | 0.016 ± 0.000 |
| 7.8 | 0.0 | 0.9706 ± 0.0000 | 0.7532 ± 0.0000 | 0.734 ± 0.000 | 0.800 ± 0.000 | 0.008 ± 0.000 |
| 15.6 | 0.0 | 0.9728 ± 0.0000 | 0.7766 ± 0.0000 | 0.729 ± 0.000 | 0.767 ± 0.000 | 0.016 ± 0.000 |
| 31.2 | 0.0 | 0.9686 ± 0.0000 | 0.7568 ± 0.0000 | 0.703 ± 0.000 | 0.778 ± 0.000 | 0.008 ± 0.000 |
| Ref | 5.0 | 0.9705 ± 0.0010 | 0.7626 ± 0.0059 | 0.726 ± 0.009 | 0.785 ± 0.011 | 0.013 ± 0.005 |
| 3.9 | 5.0 | 0.9705 ± 0.0009 | 0.7627 ± 0.0064 | 0.727 ± 0.008 | 0.786 ± 0.011 | 0.013 ± 0.003 |
| 7.8 | 5.0 | 0.9707 ± 0.0013 | 0.7646 ± 0.0089 | 0.726 ± 0.011 | 0.787 ± 0.011 | 0.013 ± 0.007 |
| 15.6 | 5.0 | 0.9705 ± 0.0016 | 0.7625 ± 0.0056 | 0.720 ± 0.012 | 0.778 ± 0.011 | 0.014 ± 0.007 |
| 31.2 | 5.0 | 0.9718 ± 0.0016 | 0.7655 ± 0.0121 | 0.718 ± 0.011 | 0.791 ± 0.014 | 0.012 ± 0.006 |
| Ref | 10.0 | 0.9710 ± 0.0019 | 0.7630 ± 0.0107 | 0.720 ± 0.015 | 0.789 ± 0.027 | 0.012 ± 0.010 |
| 3.9 | 10.0 | 0.9708 ± 0.0018 | 0.7622 ± 0.0106 | 0.719 ± 0.014 | 0.788 ± 0.027 | 0.012 ± 0.009 |
| 7.8 | 10.0 | 0.9710 ± 0.0018 | 0.7635 ± 0.0106 | 0.720 ± 0.012 | 0.786 ± 0.028 | 0.010 ± 0.009 |
| 15.6 | 10.0 | 0.9707 ± 0.0015 | 0.7643 ± 0.0075 | 0.719 ± 0.015 | 0.781 ± 0.023 | 0.014 ± 0.010 |
| 31.2 | 10.0 | 0.9730 ± 0.0016 | 0.7713 ± 0.0142 | 0.719 ± 0.012 | 0.790 ± 0.021 | 0.013 ± 0.006 |

### Table 9b — IMS Ch 1 Native `20,480 Hz` Sweep (`10` seeds for `Noise > 0`, units in multiples of baseline RMS) ([outputs/stage8_results.txt](outputs/stage8_results.txt))

| LSB (`× rms`) | Noise (`× rms`) | Fixed `idx` (`m ± sd`) | Fixed Lead (`h`) | Slope `idx` (`m ± sd`) | Slope Lead (`h`) | `196..491` Alarms (`F / S`) | Gated |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ref | `0 × rms` | 527.0 ± 0.0 | 75.67 ± 0.00 | 538.0 ± 0.0 | 73.83 ± 0.00 | 0.0 / 0.0 | 0 |
| `rms/16` (`0.0625×`) | `0 × rms` | 527.0 ± 0.0 | 75.67 ± 0.00 | 538.0 ± 0.0 | 73.83 ± 0.00 | 0.0 / 0.0 | 0 |
| `rms/8` (`0.125×`) | `0 × rms` | 527.0 ± 0.0 | 75.67 ± 0.00 | 538.0 ± 0.0 | 73.83 ± 0.00 | 0.0 / 0.0 | 0 |
| `rms/4` (`0.25×`) | `0 × rms` | 527.0 ± 0.0 | 75.67 ± 0.00 | 538.0 ± 0.0 | 73.83 ± 0.00 | 0.0 / 0.0 | 0 |
| `rms/2` (`0.5×`) | `0 × rms` | 534.0 ± 0.0 | 74.50 ± 0.00 | 538.0 ± 0.0 | 73.83 ± 0.00 | 0.0 / 0.0 | 0 |
| Ref | `0.125 × rms` | 521.8 ± 21.9 | 76.53 ± 3.65 | 537.8 ± 0.8 | 73.87 ± 0.13 | 0.4 / 0.0 | 0 |
| `rms/16` (`0.0625×`) | `0.125 × rms` | 522.0 ± 22.0 | 76.50 ± 3.66 | 537.9 ± 0.3 | 73.85 ± 0.05 | 0.4 / 0.0 | 0 |
| `rms/8` (`0.125×`) | `0.125 × rms` | 522.5 ± 21.1 | 76.42 ± 3.52 | 537.7 ± 0.5 | 73.88 ± 0.08 | 0.1 / 0.0 | 0 |
| `rms/4` (`0.25×`) | `0.125 × rms` | 529.5 ± 3.2 | 75.25 ± 0.53 | 538.1 ± 0.6 | 73.82 ± 0.09 | 0.0 / 0.0 | 0 |
| `rms/2` (`0.5×`) | `0.125 × rms` | 522.2 ± 23.4 | 76.47 ± 3.90 | 537.9 ± 0.3 | 73.85 ± 0.05 | 0.5 / 0.0 | 0 |
| Ref | `0.25 × rms` | 517.5 ± 29.4 | 77.25 ± 4.90 | 537.8 ± 0.8 | 73.87 ± 0.13 | 0.4 / 0.0 | 0 |
| `rms/16` (`0.0625×`) | `0.25 × rms` | 523.7 ± 21.6 | 76.22 ± 3.60 | 537.8 ± 0.8 | 73.87 ± 0.13 | 0.1 / 0.0 | 0 |
| `rms/8` (`0.125×`) | `0.25 × rms` | 516.2 ± 28.7 | 77.47 ± 4.79 | 537.8 ± 0.8 | 73.87 ± 0.13 | 0.4 / 0.0 | 0 |
| `rms/4` (`0.25×`) | `0.25 × rms` | 522.8 ± 21.3 | 76.37 ± 3.54 | 538.0 ± 0.8 | 73.83 ± 0.14 | 0.1 / 0.0 | 0 |
| `rms/2` (`0.5×`) | `0.25 × rms` | 513.5 ± 29.0 | 77.92 ± 4.84 | 537.9 ± 1.0 | 73.85 ± 0.17 | 0.9 / 0.0 | 0 |

### Table 9c — Baseline `mean`/`sigma` Storage Formats (`float16` & `int16` Fixed Point vs `float32`) ([outputs/stage8_results.txt](outputs/stage8_results.txt))

| Dataset | Format | Bytes vs `float32` | `HI` Difference vs `float32` | Downstream Impact |
| :--- | :--- | :---: | :--- | :--- |
| **CWRU `3200 Hz` (`F=8`)** | `float16` | 32 B vs 64 B | `max_abs = 2.376e-03`, `med_abs = 7.757e-04`, `max_rel = 2.143e-02` | `min_AUROC 0.9002 -> 0.9005`, `max_rec_delta = 0.0000` |
| **CWRU `3200 Hz` (`F=8`)** | `int16` (`per-vec scale`) | 40 B vs 64 B | `max_abs = 5.555e-03`, `med_abs = 3.462e-04`, `max_rel = 2.416e-02` | `min_AUROC 0.9002 -> 0.9014`, `max_rec_delta = 0.0000` |
| **IMS Ch 1 (`F=9`)** | `float16` | 36 B vs 72 B | `raw max_abs = 2.003e-02`, `EMA max_abs = 8.119e-03`, `med_abs = 1.383e-03` | Alarms unchanged (`527, 538`) |
| **IMS Ch 1 (`F=9`)** | `int16` (`per-vec scale`) | 44 B vs 72 B | `raw max_abs = 1.138e-02`, `EMA max_abs = 3.209e-03`, `med_abs = 9.765e-04` | Alarms unchanged (`527, 538`) |

---

## 10. Consolidated Claims, Evidence & Limitations Table

| Claim | Evidence (`script` -> `output file`) | Limitation |
| :--- | :--- | :--- |
| **`log1p_HI` separates all 9 CWRU faults at `12 kHz` (`AUROC 1.0000`, `Rec@thr 1.000`, `17–21` floats vs `~13.8k` IForest nodes)** | [src/stage3c_lolo_log1p.py](src/stage3c_lolo_log1p.py), [src/stage4_baselines.py](src/stage4_baselines.py) -> [outputs/stage3c_results.txt](outputs/stage3c_results.txt), [outputs/stage4_results.txt](outputs/stage4_results.txt) | **Unseen 0 HP false-positive rate (`normal` left out) is `0.186` (`22/118`) at `12 kHz`**, because `normal_0` has higher 3000–6000 Hz energy than 1–3 HP; only 1 recording per fault class at 0 HP is evaluated |
| **`log1p` removes hard `Z_CAP=6` saturation while keeping small healthy spread** | [src/stage3c_lolo_log1p.py](src/stage3c_lolo_log1p.py) -> [outputs/stage3c_results.txt](outputs/stage3c_results.txt) | No CWRU fault severity monotonic ordering is claimed or expected (`OR014@6` is physically lower amplitude than `007`) |
| **Negative: `3200 Hz` downsampling degrades weak faults (`ball_007`, `ball_021`, `outer_race_014`)** | [src/stage4_baselines.py](src/stage4_baselines.py), [tools/stream_simulator.py](tools/stream_simulator.py) -> [outputs/stage4_results.txt](outputs/stage4_results.txt), [outputs/stage7_simulator_results.txt](outputs/stage7_simulator_results.txt) | Dropping `1.6–6 kHz` loses fault carrier energy: `Rec@thr` falls to `0.25 / 0.21 / 0.24` (`AUROC 0.943 / 0.937 / 0.860`), and streamed `ball_007` never reaches threshold (`max_ema=0.879 < 1.012`) |
| **IMS Ch 1 `HI_ema` gives `75.67 h` lead (`idx 527`) with zero false alarms in `98..491`** | [src/stage4_baselines.py](src/stage4_baselines.py), [src/stage5_ims_trend.py](src/stage5_ims_trend.py) -> [outputs/stage4_results.txt](outputs/stage4_results.txt), [outputs/stage5_ims_results.txt](outputs/stage5_ims_results.txt) | **Single run-to-failure recording** (`Set No. 2`, outer race only); `presumed_healthy = 196..491` was informed by Stage 5 plots; Stage 5 baseline-block alarms (`0..97`, `43` fixed) reflect `98..195` calibration shift |
| **Negative: Slope detector (`idx 538`) is NOT earlier than fixed EMA threshold (`idx 527`)** | [src/stage4_baselines.py](src/stage4_baselines.py) -> [outputs/stage4_results.txt](outputs/stage4_results.txt), [outputs/stage5_ims_results.txt](outputs/stage5_ims_results.txt) | Requires `12` windows (`+` `HI > baseline_mean` `+` 3-consec), triggering `11` snapshots (`1.83 h`) **after** fixed `HI_ema` |
| **Negative: Non-failing IMS bearings (`Ch 2–4`) also trend upward and trigger alarms (`Ch 4 idx 539`)** | [src/stage5b_ims_diagnostics.py](src/stage5b_ims_diagnostics.py), [src/stage4_baselines.py](src/stage4_baselines.py) -> [outputs/stage5b_ims_diagnostics.txt](outputs/stage5b_ims_diagnostics.txt), [outputs/stage4_results.txt](outputs/stage4_results.txt) | Broadband mechanical coupling yields `r=0.9414 / rho=0.9427` between Ch 1 and Ch 4 `HI_ema`; HI shows monotonic housing degradation, **not fault localization** (localization in Stages 4/5b is post-hoc exploratory only) |
| **Shaft-stopped tail (`982–983`) triggers false "drop back to zero" without gating** | [src/stage5b_ims_diagnostics.py](src/stage5b_ims_diagnostics.py), [tools/stream_simulator.py](tools/stream_simulator.py) -> [outputs/stage5b_ims_diagnostics.txt](outputs/stage5b_ims_diagnostics.txt), [outputs/stage7_simulator_results.txt](outputs/stage7_simulator_results.txt) | Machine-off RMS gate cleanly freezes EMA (`20/20` zero windows in Scenario C), but in streaming (`512` hop), **transition window `30` straddles the seam (`512` signal + `512` zeros)** and is not gated |
| **C99 static `float32` implementation matches Python across `774` CWRU windows (`<= 8.74e-5` rel) and `982` IMS snapshots (`982/982` boolean parity, zero `malloc`)** | [firmware/tests/test_stage6b_cwru.c](firmware/tests/test_stage6b_cwru.c), [firmware/tests/test_stage6b_ims.c](firmware/tests/test_stage6b_ims.c), [firmware/tests/test_stage7_stream_conformance.c](firmware/tests/test_stage7_stream_conformance.c) -> [outputs/stage6b_cwru_conformance.txt](outputs/stage6b_cwru_conformance.txt), [outputs/stage6b_ims_conformance.txt](outputs/stage6b_ims_conformance.txt), [outputs/stage7_stream_conformance.txt](outputs/stage7_stream_conformance.txt) | `test_stage6b_ims.c` reports `FAIL` strictly on near-zero baseline snapshot `idx 86` `HI_raw` relative diff (`1.462e-3` > `1e-3`, while `max_abs = 4.84e-5`); Stage 6 200-window alarm indices (`105, 106`) test an artificially stitched file sequence |
| **Streaming static RAM fits ESP32 budget (`12.24 KiB` CWRU, `13.63 KiB` IMS)** | `sizeof(hi_stream_t)` & `size` -> [outputs/stage7_stream_conformance.txt](outputs/stage7_stream_conformance.txt), [outputs/stage6b_host_size.txt](outputs/stage6b_host_size.txt) | `size /tmp/hi_core_*.o` (`3,256 B` / `4,992 B`) is a **macOS arm64 host proxy (not ESP32 flash)**; streaming timing (`15.7–82.2 us/hop`) is a **laptop `ctypes` simulation (not measured ESP32 hardware execution)** |
| **Simulated quantization (`<= 0.5 x rms` / `31.2 mg`) and AWGN preserve IMS slope (`537.7–538.1`) & CWRU ranking (`MeanAUC >= 0.9686`); `float16`/`int16` weights match alarms** | [src/stage8_sensor_sweep.py](src/stage8_sensor_sweep.py) -> [outputs/stage8_results.txt](outputs/stage8_results.txt), [outputs/plots/stage8_cwru_heatmap.png](outputs/plots/stage8_cwru_heatmap.png), [outputs/plots/stage8_ims_heatmap.png](outputs/plots/stage8_ims_heatmap.png) | **Simulated sensor sweep on digitized streams, NOT a real MEMS accelerometer**: omits analog anti-aliasing, sensor resonance, temperature drift, and quantization before resampling (`48k->3.2k`); IMS `0.125/0.25 x rms` noise injects `0.0–0.9` fixed alarms in `196..491` |

