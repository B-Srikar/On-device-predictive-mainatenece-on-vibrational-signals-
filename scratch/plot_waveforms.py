"""Generate 0.2-second waveform plots for the four starter CWRU DE recordings."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

RAW_DIR = Path("data/raw/cwru")
OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATE = 12000
WINDOW_S = 0.2
N = int(RATE * WINDOW_S)

FILES = [
    ("normal_0.mat",  "Normal (0 HP)",          "steelblue"),
    ("IR007_0.mat",   "IR Fault 0.007 in (0 HP)","darkorange"),
    ("B007_0.mat",    "Ball Fault 0.007 in (0 HP)","forestgreen"),
    ("OR007@6_0.mat", "OR@6 Fault 0.007 in (0 HP)","crimson"),
]

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
fig.suptitle("CWRU Drive-End Vibration — first 0.2 s at 12 kHz", fontsize=13, fontweight="bold")

for ax, (fname, title, color) in zip(axes, FILES):
    path = RAW_DIR / fname
    mat = loadmat(path)
    de_key = next(k for k in mat if not k.startswith("__") and k.lower().endswith("_de_time"))
    sig = np.asarray(mat[de_key], dtype=float).squeeze()[:N]
    t = np.arange(len(sig)) / RATE * 1000  # milliseconds
    ax.plot(t, sig, color=color, linewidth=0.6)
    ax.set_title(title, fontsize=10, loc="left", pad=2)
    ax.set_ylabel("Amplitude (g)", fontsize=8)
    ax.set_xlim(0, WINDOW_S * 1000)
    rms = float(np.sqrt(np.mean(sig**2)))
    kurt = float(np.mean((sig - sig.mean())**4) / (sig.std()**4 + 1e-30))
    ax.text(0.98, 0.88, f"RMS={rms:.4f}  Kurt={kurt:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.grid(True, alpha=0.3, linewidth=0.4)

axes[-1].set_xlabel("Time (ms)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT_DIR / "cwru_waveforms_0007_0hp.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Also plot severity comparison for IR fault
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig2.suptitle("IR Fault — severity progression (0.007 / 0.014 / 0.021 in, 0 HP)", fontsize=12, fontweight="bold")
for ax, (fname, sev, color) in zip(axes2, [
    ("IR007_0.mat", "0.007 in", "gold"),
    ("IR014_0.mat", "0.014 in", "darkorange"),
    ("IR021_0.mat", "0.021 in", "firebrick"),
]):
    path = RAW_DIR / fname
    mat = loadmat(path)
    de_key = next(k for k in mat if not k.startswith("__") and k.lower().endswith("_de_time"))
    sig = np.asarray(mat[de_key], dtype=float).squeeze()[:N]
    t = np.arange(len(sig)) / RATE * 1000
    ax.plot(t, sig, color=color, linewidth=0.6)
    ax.set_title(f"IR {sev}", fontsize=10, loc="left", pad=2)
    ax.set_ylabel("Amplitude (g)", fontsize=8)
    rms = float(np.sqrt(np.mean(sig**2)))
    kurt = float(np.mean((sig - sig.mean())**4) / (sig.std()**4 + 1e-30))
    ax.text(0.98, 0.88, f"RMS={rms:.4f}  Kurt={kurt:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.grid(True, alpha=0.3, linewidth=0.4)
axes2[-1].set_xlabel("Time (ms)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out2 = OUT_DIR / "cwru_waveforms_IR_severity.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

