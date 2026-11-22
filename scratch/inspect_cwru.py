"""Stage 1 CWRU inspection: load every .mat file, report shape, DE variable name,
sample count, inferred sampling rate from config, and basic DE statistics."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

RAW_DIR = Path("data/raw/cwru")

EXPECTED = [
    ("normal_0.mat",   "normal",           "0.007 in — normal baseline, 0 HP"),
    ("IR007_0.mat",    "inner_race_007",   "0.007 in inner-race fault, 0 HP"),
    ("B007_0.mat",     "ball_007",         "0.007 in ball fault, 0 HP"),
    ("OR007@6_0.mat",  "outer_race_007",   "0.007 in OR@6:00 fault, 0 HP"),
    ("IR014_0.mat",    "inner_race_014",   "0.014 in inner-race fault, 0 HP"),
    ("B014_0.mat",     "ball_014",         "0.014 in ball fault, 0 HP"),
    ("OR014@6_0.mat",  "outer_race_014",   "0.014 in OR@6:00 fault, 0 HP"),
    ("IR021_0.mat",    "inner_race_021",   "0.021 in inner-race fault, 0 HP"),
    ("B021_0.mat",     "ball_021",         "0.021 in ball fault, 0 HP"),
    ("OR021@6_0.mat",  "outer_race_021",   "0.021 in OR@6:00 fault, 0 HP"),
]

ASSUMED_RATE = 12000  # from config.yaml

rows = []
for fname, label, description in EXPECTED:
    path = RAW_DIR / fname
    if not path.exists():
        rows.append((fname, "MISSING", "", "", "", "", "", "", ""))
        continue
    mat = loadmat(path)
    public = {k: v for k, v in mat.items() if not k.startswith("__")}
    de_keys = [k for k in public if k.lower().endswith("_de_time")]
    fe_keys = [k for k in public if k.lower().endswith("_fe_time")]
    ba_keys = [k for k in public if k.lower().endswith("_ba_time")]
    rpm_keys= [k for k in public if "rpm" in k.lower()]
    
    if len(de_keys) != 1:
        rows.append((fname, f"ERROR: DE keys={de_keys}", "", "", "", "", "", "", ""))
        continue
    de = np.asarray(public[de_keys[0]], dtype=float).squeeze()
    n = de.size
    duration_s = n / ASSUMED_RATE
    rms = float(np.sqrt(np.mean(de**2)))
    peak = float(np.max(np.abs(de)))
    kurt = float(np.mean((de - de.mean())**4) / (de.std()**4 + 1e-30))
    rpm_val = ""
    if rpm_keys:
        rpm_val = str(int(np.asarray(public[rpm_keys[0]]).flat[0]))
    
    all_vars = list(public.keys())
    rows.append((fname, de_keys[0], n, f"{duration_s:.2f}s", rpm_val, f"{rms:.5f}", f"{peak:.5f}", f"{kurt:.3f}", all_vars))

print(f"\n{'='*90}")
print(f"CWRU Stage-1 Inspection  |  Assumed rate: {ASSUMED_RATE} Hz  |  Dir: {RAW_DIR}")
print(f"{'='*90}")
print(f"{'File':<20} {'DE variable':<22} {'Samples':>9} {'Duration':>9} {'RPM':>5} {'RMS':>10} {'Peak':>10} {'Kurtosis':>10}")
print(f"{'-'*20} {'-'*22} {'-'*9} {'-'*9} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
for r in rows:
    fname, de_var, n, dur, rpm, rms, peak, kurt, allv = r
    print(f"{fname:<20} {str(de_var):<22} {str(n):>9} {str(dur):>9} {str(rpm):>5} {str(rms):>10} {str(peak):>10} {str(kurt):>10}")

print(f"\n{'All MAT variables per file':}")
print(f"{'-'*60}")
for r in rows:
    fname, de_var, n, dur, rpm, rms, peak, kurt, allv = r
    if isinstance(allv, list):
        print(f"  {fname}: {allv}")

print(f"\n{'Sampling-rate consistency note':}")
print(f"  - normal_0.mat (97.mat) is from the Normal Baseline page.")
print(f"  - All fault files are from the 12k Drive End Bearing Fault Data page.")
print(f"  - Expected rate for all 10 files: 12,000 Hz.")
print(f"  - normal_0.mat samples={rows[0][2]}  →  {rows[0][2]}/{ASSUMED_RATE} = {rows[0][2]/ASSUMED_RATE:.2f}s at 12 kHz")
print(f"\nDone.")

