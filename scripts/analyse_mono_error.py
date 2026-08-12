"""Error as a function of true depth — the median alone hides the structure."""
import numpy as np, yaml, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# re-run the pairing quickly by importing the calibration script's logic
exec(open("/home/inoruske/surgical_twin_ws/calibrate_mono_depth.py")
     .read().split("Z=np.array(Z_stereo)")[0])

Z = np.array(Z_stereo); d = np.array(d_rel_v)
A = np.stack([d, np.ones_like(d)], 1)
(a, b), *_ = np.linalg.lstsq(A, 1.0/Z, rcond=None)
Zhat = 1.0/(a*d + b)
err  = (Zhat - Z) * 1000          # signed, mm

bins = [(50,65),(65,80),(80,95),(95,120),(120,150)]
print(f"\n{'depth range':>14s} {'n':>5s} {'bias':>8s} {'|err| med':>10s} {'|err| p90':>10s}")
print("-"*52)
for lo,hi in bins:
    m = (Z*1000 >= lo) & (Z*1000 < hi)
    if m.sum() < 5: continue
    e = err[m]
    print(f"{lo:5d}-{hi:<4d} mm {m.sum():5d} {e.mean():+8.1f} "
          f"{np.median(np.abs(e)):10.1f} {np.percentile(np.abs(e),90):10.1f}")

print(f"\nSpearman rank corr (ordering preserved?): "
      f"{np.corrcoef(np.argsort(np.argsort(Z)), np.argsort(np.argsort(Zhat)))[0,1]:.3f}")
print(f"Pearson on depth:  {np.corrcoef(Z, Zhat)[0,1]:.3f}")

fig, ax = plt.subplots(figsize=(7,4))
ax.scatter(Z*1000, err, s=14, alpha=.5, color="#028090")
ax.axhline(0, color='k', lw=1)
ax.set_xlabel("true (stereo) depth, mm"); ax.set_ylabel("mono error, mm")
ax.set_title("Monocular depth error vs true depth"); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("/home/inoruske/surgical_twin_ws/results/mono_error_vs_depth.png", dpi=150)
print("\nsaved -> results/mono_error_vs_depth.png")
