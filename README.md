# On-Device Predictive Maintenance on Vibrational Signals

Lightweight `float32` C99 + Python pipeline for vibration health-index (HI) monitoring targeting resource-constrained microcontrollers (ESP32 class), evaluated on the Case Western Reserve University (CWRU) bearing fault dataset and the IMS Set No. 2 run-to-failure dataset.

All consolidated metrics, figures, and limitations are documented in [final_results.md](final_results.md). All locked algorithm parameters and fixed seeds live in [config.yaml](config.yaml).

---

## 1. Environment Setup

Tested environment:
- **OS / Architecture:** macOS (Apple `arm64`)
- **C Compiler:** Apple `clang` (`-std=c99`, `-Wall -Wextra -Werror`, `ASan`/`UBSan`) and `size`
- **Python:** `3.9.6` with:
  - `numpy` (`2.3.5`)
  - `scipy` (`1.16.3`)
  - `matplotlib` (`3.10.6`)
  - `scikit-learn` (`1.7.2`)
  - `Pillow` (`12.0.0`)
  - `PyYAML` (`6.0.3`)

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install numpy scipy matplotlib scikit-learn pillow pyyaml
```

Verify the toolchain:

```bash
clang --version
python3 -c "import numpy, scipy, matplotlib, sklearn, PIL, yaml; print('OK')"
```

---

## 2. Raw Data Download & Layout

Do not modify files under `data/raw/`. Place the raw datasets in the exact structure expected by [config.yaml](config.yaml):

### A. CWRU Bearing Data Center (`data/raw/cwru/`)

Download from the **Case Western Reserve University Bearing Data Center** (Normal Baseline Data + 12k Drive End Bearing Fault Data, **0 HP motor load** for faults; normal baselines across 0–3 HP) and save/rename as:

```text
data/raw/cwru/
  normal_0.mat     # CWRU file 97  (0 HP, 1796 RPM, 48 kHz; var X097_DE_time)
  normal_1.mat     # CWRU file 98  (1 HP, 1772 RPM, 48 kHz; var X098_DE_time)
  normal_2.mat     # CWRU file 99  (2 HP, 1750 RPM, 48 kHz; var X099_DE_time — contains both X098 & X099)
  normal_3.mat     # CWRU file 100 (3 HP, 1730 RPM, 48 kHz; var X100_DE_time)
  IR007_0.mat      # CWRU file 105 (12 kHz)
  B007_0.mat       # CWRU file 118 (12 kHz)
  OR007@6_0.mat    # CWRU file 130 (12 kHz)
  IR014_0.mat      # CWRU file 169 (12 kHz)
  B014_0.mat       # CWRU file 185 (12 kHz)
  OR014@6_0.mat    # CWRU file 197 (12 kHz)
  IR021_0.mat      # CWRU file 209 (12 kHz)
  B021_0.mat       # CWRU file 222 (12 kHz)
  OR021@6_0.mat    # CWRU file 234 (12 kHz)
```

> **Sampling-rate note:** CWRU `.mat` files do not store sampling rates internally. Source rates (`48000` Hz for `normal_0..3.mat`, `12000` Hz for the 9 fault `.mat` files) are explicitly configured in `source_sampling_rates` in [config.yaml](config.yaml) and verified via shaft rotational peak checks in Stage 3c.

### B. IMS Bearing Dataset — Set No. 2 (`data/raw/ims/2nd_test/`)

Download the **IMS Bearing Dataset** (NSF I/UCRC Center for Intelligent Maintenance Systems / NASA Prognostics Data Repository, `4.+Bearings.zip`) and extract `2nd_test.rar` (`2nd_test` folder) into `data/raw/ims/2nd_test/`:

```text
data/raw/ims/
  Readme Document for IMS Bearing Data.pdf
  2nd_test/
    2004.02.12.10.32.39
    ...                 # 984 chronological 4-channel ASCII snapshots (20,480 Hz, 20,480 samples/file)
    2004.02.19.06.22.39
```

Channel mapping documented in `Readme Document for IMS Bearing Data.pdf` (Test No. 2):
- **Channel 1 (`idx 0`, Bearing 1):** failed with an outer-race defect at end of test.
- **Channels 2–4 (`idx 1..3`, Bearings 2–4):** did not fail (`Channel 4` is used as the primary far-end control).

---

## 3. Reproducing `final_results.md` from Scratch

### Full One-Shot Clean Reproduction

Run the full clean pipeline (Stages 2 through 8, C builds, ASan/UBSan checks, host object size, `ctypes` streaming simulation, and 10-seed sensor sweeps):

```bash
bash tools/run_all_stages.sh
```

> Deterministic seeds live in [config.yaml](config.yaml): `stage4_iforest_random_state: 0`, `stage8_seed: 0`, and `stage8_num_seeds: 10` (using seeds `0..9` when noise > 0).

---

### Individual Stage Commands

All commands run from the repository root. Each stage writes to `data/processed/`, `firmware/`, or `outputs/`.

#### Stage 2 — Resampling, Windowing & Feature Extraction (`src/preprocess.py`)
```bash
python3 src/preprocess.py --rate-mode configured > outputs/stage2_3200_summary.txt
python3 src/preprocess.py --rate-mode original   > outputs/stage2_12000_summary.txt
```

#### Stage 3b — Downsampling & Filter Confound Checks (`src/stage3b_confounds.py`)
```bash
python3 src/stage3b_confounds.py > outputs/stage3b_results.txt
```

#### Stage 3c — Leave-One-Load-Out (LOLO) & `log1p` Compression (`src/stage3c_lolo_log1p.py`)
```bash
python3 src/stage3c_lolo_log1p.py > outputs/stage3c_results.txt
```

#### Stage 5 — IMS Set No. 2 Run-to-Failure Trend & Sensitivity (`src/stage5_ims_trend.py`)
```bash
python3 src/stage5_ims_trend.py > outputs/stage5_ims_results.txt
```

#### Stage 5b — IMS All-Four-Bearing Diagnostics & Tail Audit (`src/stage5b_ims_diagnostics.py`)
```bash
python3 src/stage5b_ims_diagnostics.py > outputs/stage5b_ims_diagnostics.txt
```

#### Stage 4 — CWRU & IMS Baselines (`B1–B3`) & HI Ablations (`src/stage4_baselines.py`)
*(Uses `stage4_iforest_random_state: 0` from [config.yaml](config.yaml).)*
```bash
python3 src/stage4_baselines.py > outputs/stage4_results.txt
```

#### Stage 6 & Stage 6b — C Baseline Export, Static Conformance, Edge Cases, ASan/UBSan & Host Object Size
1. Export headers, test vectors, and float32 waveform conformance binaries:
```bash
python3 tools/export_baseline_header.py --mode cwru
python3 tools/export_baseline_header.py --mode ims
python3 tools/export_c_test_vectors.py
python3 tools/export_conformance_bins.py
```

2. Verify zero dynamic allocation in C core:
```bash
! grep -rn "malloc\|calloc\|realloc\|free" firmware/src/ firmware/include/
```

3. Build and run Stage 6 conformance (24 stored windows + 200-window sequence + laptop latency) + ASan/UBSan + macOS host `size` proxy:
```bash
clang -std=c99 -O2 -Wall -Wextra -Werror -Ifirmware/include -Ifirmware/tests \
  firmware/src/hi_core.c firmware/tests/test_health_index.c -lm -o /tmp/test_hi
/tmp/test_hi > outputs/stage6_c_conformance.txt

clang -std=c99 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -Ifirmware/include -Ifirmware/tests \
  firmware/src/hi_core.c firmware/tests/test_health_index.c -lm -o /tmp/test_hi_asan
/tmp/test_hi_asan > /dev/null

clang -std=c99 -O2 -Wall -Wextra -Werror -c -Ifirmware/include firmware/src/hi_core.c -o /tmp/hi_core_cwru.o
clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -c -Ifirmware/include firmware/src/hi_core.c -o /tmp/hi_core_ims.o
size /tmp/hi_core_cwru.o /tmp/hi_core_ims.o > outputs/stage6b_host_size.txt
```

4. Run Stage 6b full CWRU (774 windows + 4 edge cases) and IMS (982 snapshots) conformance:
```bash
clang -std=c99 -O2 -Wall -Wextra -Werror -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage6b_cwru.c -lm -o /tmp/test_stage6b_cwru
/tmp/test_stage6b_cwru > outputs/stage6b_cwru_conformance.txt

clang -std=c99 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage6b_cwru.c -lm -o /tmp/test_stage6b_cwru_asan
/tmp/test_stage6b_cwru_asan > /dev/null

clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage6b_ims.c -lm -o /tmp/test_stage6b_ims
/tmp/test_stage6b_ims > outputs/stage6b_ims_conformance.txt || true
```
*(Note: `test_stage6b_ims` exits `1` strictly on the baseline near-zero `idx 86` `HI_raw` relative check `1.462e-3 > 1e-3`, while `HI_ema` passes at `2.267e-5` and all `982/982` alarms match identically.)*

#### Stage 7 — C Streaming API (`hi_stream_push`), Shared Libraries & `ctypes` Laptop Simulator
```bash
clang -std=c99 -O2 -Wall -Wextra -Werror -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_cwru
clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_ims
/tmp/test_stream_cwru >  outputs/stage7_stream_conformance.txt
/tmp/test_stream_ims  >> outputs/stage7_stream_conformance.txt

clang -std=c99 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_cwru_asan
clang -std=c99 -O1 -g -Wall -Wextra -Werror -DHI_CONFIG_IMS -fsanitize=address,undefined -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_ims_asan
/tmp/test_stream_cwru_asan > /dev/null
/tmp/test_stream_ims_asan  > /dev/null

clang -std=c99 -O2 -Wall -Wextra -Werror -shared -fPIC -Ifirmware/include firmware/src/hi_core.c -lm -o outputs/libhi_cwru.dylib
clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -shared -fPIC -Ifirmware/include firmware/src/hi_core.c -lm -o outputs/libhi_ims.dylib

python3 tools/stream_simulator.py > outputs/stage7_simulator_results.txt
```
*(Interactive live viewer for a single scenario: `python3 tools/stream_simulator.py --live --scenario ims --speed 10`.)*

#### Stage 8 — Simulated Sensor Resolution & 10-Seed Noise Sweep + `float16`/`int16` Weights (`src/stage8_sensor_sweep.py`)
*(Uses `stage8_seed: 0` and `stage8_num_seeds: 10` from [config.yaml](config.yaml).)*
```bash
python3 src/stage8_sensor_sweep.py > outputs/stage8_results.txt
```

---

## 4. Repository Structure

```text
config.yaml                  # Locked pipeline, threshold, IMS, and sweep parameters
final_results.md             # Consolidated results, plots index, and claims/limitations matrix
data/raw/                    # Unmodified CWRU (.mat) and IMS (2nd_test/) inputs
data/processed/              # Generated CSV features and binary waveform test streams
firmware/
  include/                   # hi_config.h, hi_core.h, generated hi_baseline*.h
  src/hi_core.c              # Static-memory C99 streaming & snapshot health index core
  tests/                     # C unit, edge-case, batch, and streaming conformance tests
  arduino/health_monitor.ino # Microcontroller integration sketch wrapper
src/                         # Python evaluation modules (Stages 2–5b, Stage 8)
tools/                       # Header/vector exporters, ctypes simulator, clean runner script
outputs/                     # Stage logs (.txt), metrics (.csv), dylibs, and plots (.png, .gif)
```
