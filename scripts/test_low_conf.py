"""
Do the missed instruments produce ANY response below the 0.25 floor?

ds10 frame241 and ds9 frame032 both contain a metallic serrated-jaw
instrument (likely Vessel Sealer or Probe) that goes entirely undetected.
If it activates weakly, lowering the floor surfaces it for UNKNOWN
handling. If it produces nothing even at 0.05, only a class-agnostic
detector will catch it.
"""
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/home/inoruske/surgical_twin_ws")
m  = YOLO(str(WS/"models/best_temporal.pt"))

targets = [
    ("ds10 frame241", WS/"data/endovis_test/instrument_dataset_10/left_frames/frame241.png"),
    ("ds9 frame032",  WS/"data/endovis_test/instrument_dataset_9/left_frames/frame032.png"),
    ("ds9 frame253",  WS/"data/endovis_test/instrument_dataset_9/left_frames/frame253.png"),
]

for name, path in targets:
    if not path.exists():
        print(f"{name}: missing"); continue
    print(f"\n{'='*60}\n{name}")
    for thr in (0.25, 0.15, 0.10, 0.05, 0.02):
        r = m(str(path), conf=thr, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        cls = ([] if r.boxes is None else
               [f"{m.names[int(b.cls[0])]}:{float(b.conf[0]):.2f}" for b in r.boxes])
        print(f"  conf>={thr:.2f}: {n:2d} det   {', '.join(cls[:6])}")

# visualise the lowest threshold
fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 5))
if len(targets) == 1: axes = [axes]
for ax, (name, path) in zip(axes, targets):
    if not path.exists(): continue
    r = m(str(path), conf=0.05, verbose=False)[0]
    ax.imshow(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name} @ conf>=0.05", fontsize=10); ax.axis('off')
plt.tight_layout()
plt.savefig(WS/"results/low_conf_probe.png", dpi=110, bbox_inches='tight')
print(f"\nsaved -> results/low_conf_probe.png")
