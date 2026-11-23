"""Export 24 stored CWRU windows and a 200-window consecutive sequence to firmware/tests/hi_test_vectors.h."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]

from export_baseline_header import compute_pooled_normal_0_2_baseline  # noqa: E402
from preprocess import (  # noqa: E402
    RECORDINGS,
    de_signal_and_metadata,
    extract_features,
    load_config,
    matched_rate_signal,
    window_signal,
)
from stage3_hi import load_rows  # noqa: E402
from stage3b_confounds import add_logs, mat  # noqa: E402

EPSILON = np.finfo(float).eps
RAW_BAND_COLS = [
    "band_energy_0_400hz",
    "band_energy_400_800hz",
    "band_energy_800_1200hz",
    "band_energy_1200_1600hz",
]
TEST_SELECTIONS: list[tuple[str, str, list[int]]] = [
    ("normal_0.mat", "normal_holdout", [22, 29]),
    ("normal_1.mat", "normal_1_holdout", [44]),
    ("normal_2.mat", "normal_2_holdout", [44]),
    ("normal_3.mat", "normal_3_unseen", [0, 30]),
    ("IR007_0.mat", "inner_race_007", [0, 30]),
    ("B007_0.mat", "ball_007", [0, 30]),
    ("OR007@6_0.mat", "outer_race_007", [0, 30]),
    ("IR014_0.mat", "inner_race_014", [0, 30]),
    ("B014_0.mat", "ball_014", [0, 30]),
    ("OR014@6_0.mat", "outer_race_014", [0, 30]),
    ("IR021_0.mat", "inner_race_021", [0, 30]),
    ("B021_0.mat", "ball_021", [0, 30]),
    ("OR021@6_0.mat", "outer_race_021", [0, 30]),
]


def c_float(value: float) -> str:
    s = f"{float(value):.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def c_array(values: np.ndarray) -> str:
    return ", ".join(c_float(v) for v in values)


def py_hi(feat_vec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    z = np.maximum((feat_vec - mean) / std, 0.0)
    return float(np.sqrt(np.mean(np.log1p(z) ** 2)))


def build_sequence(
    rows: list[dict],
    feats: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    median_rms: float,
    hi_threshold: float,
    hi_base_mean: float,
    config: dict,
) -> dict[str, object]:
    """Build 200 consecutive 3200 Hz windows exercising EMA, 12-sample slope, 3-consec rule, and machine-off gate."""
    n0 = sorted([r for r in rows if r["label"] == "normal"], key=lambda r: r["window_idx"])
    n1 = sorted([r for r in rows if r["label"] == "normal_1"], key=lambda r: r["window_idx"])
    n2 = sorted([r for r in rows if r["label"] == "normal_2"], key=lambda r: r["window_idx"])
    ir7 = sorted([r for r in rows if r["label"] == "inner_race_007"], key=lambda r: r["window_idx"])
    or21 = sorted([r for r in rows if r["label"] == "outer_race_021"], key=lambda r: r["window_idx"])

    healthy_100 = mat([*n0, *n1, *n2[:8]], feats)
    off_win = np.array([[0.0, 0.0, 0.0, 0.0, -69.07755, -69.07755, -69.07755, -69.07755]], dtype=float)
    fault_99 = mat([*ir7, *or21[:37]], feats)
    seq_X = np.vstack([healthy_100, off_win, fault_99])
    assert seq_X.shape == (200, len(feats))

    alpha = float(config["ema_alpha"])
    off_ratio = float(config["ims_machine_off_rms_ratio"])
    w_slope = int(config["ims_slope_window"])
    consec_req = int(config["ims_consecutive_alarms"])
    k_slope = float(config["ims_slope_threshold_sigma"])
    rms_gate = off_ratio * median_rms

    # Pre-pass on the initial healthy segment (indices 11..99) to fix slope threshold cleanly.
    ema_tmp: list[float] = []
    for t in range(100):
        raw_hi = py_hi(seq_X[t], mean, std)
        ema_tmp.append(raw_hi if t == 0 else alpha * raw_hi + (1.0 - alpha) * ema_tmp[-1])
    x_idx = np.arange(w_slope, dtype=float)
    x_c = x_idx - x_idx.mean()
    denom = float(np.sum(x_c ** 2))
    cal_slopes = []
    for t in range(w_slope - 1, 100):
        y = np.array(ema_tmp[t - w_slope + 1 : t + 1], dtype=float)
        cal_slopes.append(float(np.sum(x_c * (y - y.mean())) / denom))
    slope_arr = np.array(cal_slopes, dtype=float)
    slope_thr = float(slope_arr.mean() + k_slope * slope_arr.std(ddof=1))

    # Full sequential simulation matching C hi_monitor_update step-by-step.
    ema_initialized = False
    ema_hi = 0.0
    ring: list[float] = []
    fixed_consec = 0
    slope_consec = 0
    exp_gated, exp_ema, exp_slope_valid, exp_slope, exp_fixed, exp_slope_alarm = [], [], [], [], [], []
    for t in range(len(seq_X)):
        vec = seq_X[t]
        if vec[0] < rms_gate:
            exp_gated.append(1)
            exp_ema.append(ema_hi)
            exp_slope_valid.append(0)
            exp_slope.append(0.0)
            fixed_consec = 0
            slope_consec = 0
            exp_fixed.append(0)
            exp_slope_alarm.append(0)
            continue
        exp_gated.append(0)
        raw_hi = py_hi(vec, mean, std)
        if not ema_initialized:
            ema_initialized = True
            ema_hi = raw_hi
        else:
            ema_hi = alpha * raw_hi + (1.0 - alpha) * ema_hi
        exp_ema.append(ema_hi)
        ring.append(ema_hi)
        if len(ring) > w_slope:
            ring.pop(0)
        if len(ring) == w_slope:
            y = np.array(ring, dtype=float)
            slp = float(np.sum(x_c * (y - y.mean())) / denom)
            s_valid = 1
        else:
            slp = 0.0
            s_valid = 0
        exp_slope_valid.append(s_valid)
        exp_slope.append(slp)

        fixed_consec = min(fixed_consec + 1, consec_req) if ema_hi > hi_threshold else 0
        slope_consec = min(slope_consec + 1, consec_req) if (s_valid and slp > slope_thr and ema_hi > hi_base_mean) else 0
        exp_fixed.append(1 if fixed_consec >= consec_req else 0)
        exp_slope_alarm.append(1 if slope_consec >= consec_req else 0)

    return {
        "features": seq_X,
        "slope_thr": slope_thr,
        "gated": exp_gated,
        "ema": exp_ema,
        "slope_valid": exp_slope_valid,
        "slope": exp_slope,
        "fixed_alarm": exp_fixed,
        "slope_alarm": exp_slope_alarm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cwru"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_3200.csv"))
    parser.add_argument("--output", type=Path, default=Path("firmware/tests/hi_test_vectors.h"))
    args = parser.parse_args()

    config = load_config(args.config)
    stats = compute_pooled_normal_0_2_baseline(args.config, args.features)
    mean: np.ndarray = stats["mean"]  # type: ignore[assignment]
    std: np.ndarray = stats["std"]  # type: ignore[assignment]
    feats: list[str] = stats["feats"]  # type: ignore[assignment]
    median_rms = float(stats["median_rms"])
    hi_thr = float(stats["hi_threshold"])
    hi_base_mean = float(stats["hi_baseline_mean"])

    base_rate = int(config["sampling_rate"])
    target_rate = int(config["target_rate"])
    win_size = int(config["window_size"])
    overlap = float(config["overlap"])
    bands = config["fft_bands_hz"]
    source_rates = config["source_sampling_rates"]
    label_map = dict(RECORDINGS)

    cases = []
    for filename, case_label, indices in TEST_SELECTIONS:
        raw_sig, _ = de_signal_and_metadata(args.raw_dir / filename)
        sig, rate = matched_rate_signal(
            raw_sig,
            filename,
            label_map[filename],
            int(source_rates[filename]),
            base_rate,
            target_rate,
        )
        windows = list(window_signal(sig, win_size, overlap))
        for idx in indices:
            w = windows[idx]
            raw_f = extract_features(w, rate, bands)
            fvec = np.array(
                [raw_f["rms"], raw_f["peak_to_peak"], raw_f["crest_factor"], raw_f["kurtosis"]]
                + [float(np.log(max(raw_f[c], EPSILON))) for c in RAW_BAND_COLS],
                dtype=float,
            )
            cases.append((case_label, idx, w, fvec, py_hi(fvec, mean, std)))

    rows, _ = load_rows(args.features)
    add_logs(rows)
    seq = build_sequence(rows, feats, mean, std, median_rms, hi_thr, hi_base_mean, config)

    lines = [
        "#ifndef HI_TEST_VECTORS_H",
        "#define HI_TEST_VECTORS_H",
        "",
        '#include "hi_core.h"',
        "",
        f"#define HI_TEST_CASE_COUNT {len(cases)}u",
        "#define HI_SEQ_LEN 200u",
        f"#define HI_SEQ_SLOPE_THR {c_float(seq['slope_thr'])}",
        "",
        "typedef struct {",
        "    const char *label;",
        "    unsigned int window_idx;",
        "    float samples[HI_WINDOW_SIZE];",
        "    float expected_features[HI_FEATURE_COUNT];",
        "    float expected_hi;",
        "} hi_test_case_t;",
        "",
        "static const hi_test_case_t HI_TEST_CASES[HI_TEST_CASE_COUNT] = {",
    ]
    for lbl, widx, win, fvec, hi_val in cases:
        lines.append(
            f'    {{"{lbl}", {widx}u, {{{c_array(win)}}}, {{{c_array(fvec)}}}, {c_float(hi_val)}}},'
        )
    lines.extend(["};", ""])

    seq_feats: np.ndarray = seq["features"]  # type: ignore[assignment]
    lines.append("static const float HI_SEQ_FEATURES[HI_SEQ_LEN][HI_FEATURE_COUNT] = {")
    for row in seq_feats:
        lines.append("    {" + c_array(row) + "},")
    lines.extend(
        [
            "};",
            "static const float HI_SEQ_EXPECTED_EMA[HI_SEQ_LEN] = {"
            + c_array(np.asarray(seq["ema"]))
            + "};",
            "static const float HI_SEQ_EXPECTED_SLOPE[HI_SEQ_LEN] = {"
            + c_array(np.asarray(seq["slope"]))
            + "};",
            "static const int HI_SEQ_EXPECTED_SLOPE_VALID[HI_SEQ_LEN] = {"
            + ", ".join(str(v) for v in seq["slope_valid"])  # type: ignore[attr-defined]
            + "};",
            "static const int HI_SEQ_EXPECTED_GATED[HI_SEQ_LEN] = {"
            + ", ".join(str(v) for v in seq["gated"])  # type: ignore[attr-defined]
            + "};",
            "static const int HI_SEQ_EXPECTED_FIXED_ALARM[HI_SEQ_LEN] = {"
            + ", ".join(str(v) for v in seq["fixed_alarm"])  # type: ignore[attr-defined]
            + "};",
            "static const int HI_SEQ_EXPECTED_SLOPE_ALARM[HI_SEQ_LEN] = {"
            + ", ".join(str(v) for v in seq["slope_alarm"])  # type: ignore[attr-defined]
            + "};",
            "",
            "#endif",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(cases)} test windows + {len(seq_feats)}-step sequence to {args.output}")


if __name__ == "__main__":
    main()
