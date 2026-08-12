"""
Visual check on unseen procedures. Confidence statistics cannot separate
'confidently correct' from 'confidently wrong' -- only looking can.
"""
import cv2, numpy as np, random
from pathlib import Path
from ultralytics import YOLO
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/home/inoruske/surgical_twin_ws")
m  = YOLO(str(WS/"models/best_temporal.pt"))
random.seed(1)

for ds in (9, 10):
    files = sorted((WS/f"data/endovis_test/instrument_dataset_{ds}/left_frames").glob("*.png"))
    picks = random.sample(files, 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, f in zip(axes.ravel(), picks):
        r = m(str(f), conf=0.25, verbose=False)[0]
        ax.imshow(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        n = 0 if r.boxes is None else len(r.boxes)
        ax.set_title(f"{f.stem}  ({n} det)", fontsize=9); ax.axis('off')
    plt.suptitle(f"EndoVis test dataset {ds} — UNSEEN procedure", fontsize=13)
    plt.tight_layout()
    plt.savefig(WS/f"results/unseen_ds{ds}.png", dpi=110, bbox_inches='tight')
    print(f"saved -> results/unseen_ds{ds}.png")
