"""
Cross-procedure assessment on EndoVis test datasets 9 and 10.

These are procedures the model has never seen. Ground-truth masks were
withheld by the challenge, so mAP cannot be computed. What we CAN measure,
and what is still diagnostic of domain shift:

  * detection rate      - fraction of frames with any instrument found
  * confidence          - distribution vs the validation set
  * instruments/frame   - vs the 2-3 typically present
  * class distribution  - implausible predictions indicate confusion
  * UNKNOWN rate        - fraction below the 0.55 identity threshold

A large confidence drop against validation is evidence of domain shift
even without labels.
"""
import cv2, numpy as np, json, sys
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/home/inoruske/surgical_twin_ws")
model = YOLO(str(WS/"models/best_temporal.pt"))
KNOWN_CONF = 0.55

def run(name, folder, limit=None):
    files = sorted(Path(folder).glob("*.png"))
    if limit: files = files[:limit]
    confs, per_frame, classes, n_empty = [], [], Counter(), 0
    for f in files:
        r = model(str(f), conf=0.25, verbose=False)[0]
        n = 0
        if r.boxes is not None:
            for b in r.boxes:
                c = float(b.conf[0])
                confs.append(c)
                classes[model.names[int(b.cls[0])]] += 1
                n += 1
        per_frame.append(n)
        if n == 0: n_empty += 1
    return dict(name=name, n_frames=len(files), confs=np.array(confs),
                per_frame=np.array(per_frame), classes=classes, n_empty=n_empty)

sets = [
    ("val (seen procedures)", WS/"demo_data"),
    ("ds9 (UNSEEN)",  WS/"data/endovis_test/instrument_dataset_9/left_frames"),
    ("ds10 (UNSEEN)", WS/"data/endovis_test/instrument_dataset_10/left_frames"),
]

results = []
for name, folder in sets:
    if not Path(folder).exists():
        print(f"missing: {folder}"); continue
    print(f"running {name} ...")
    results.append(run(name, folder, limit=300))

print(f"\n{'dataset':24s} {'frames':>7s} {'empty':>7s} {'det/frame':>10s} "
      f"{'conf med':>9s} {'conf mean':>10s} {'>0.55':>7s}")
print("-"*82)
for r in results:
    c = r['confs']
    print(f"{r['name']:24s} {r['n_frames']:7d} "
          f"{r['n_empty']/r['n_frames']*100:6.1f}% "
          f"{r['per_frame'].mean():10.2f} "
          f"{np.median(c) if c.size else 0:9.3f} "
          f"{c.mean() if c.size else 0:10.3f} "
          f"{(c>=KNOWN_CONF).mean()*100 if c.size else 0:6.1f}%")

print(f"\nclass distribution:")
for r in results:
    tot = sum(r['classes'].values()) or 1
    print(f"\n  {r['name']}")
    for k, v in r['classes'].most_common():
        print(f"    {k:30s} {v:5d}  {v/tot*100:5.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for r in results:
    if r['confs'].size:
        ax[0].hist(r['confs'], bins=30, alpha=.5, density=True, label=r['name'])
ax[0].axvline(KNOWN_CONF, c='r', ls='--', lw=1)
ax[0].set_xlabel("detection confidence"); ax[0].set_ylabel("density")
ax[0].set_title("Confidence: seen vs unseen procedures")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

names = [r['name'] for r in results]
meds  = [np.median(r['confs']) if r['confs'].size else 0 for r in results]
ax[1].bar(range(len(names)), meds, color=["#028090","#D97706","#CC3300"][:len(names)])
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels([n.split()[0] for n in names])
ax[1].axhline(KNOWN_CONF, c='r', ls='--', lw=1)
ax[1].set_ylabel("median confidence"); ax[1].grid(alpha=.3, axis='y')
ax[1].set_title("Median confidence by dataset")
plt.tight_layout(); plt.savefig(WS/"results/cross_procedure.png", dpi=150)
print(f"\nsaved -> results/cross_procedure.png")
