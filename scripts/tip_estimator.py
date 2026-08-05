"""
Geometric tip estimation, validated against ground-truth jaw labels.

At inference the model outputs a whole-instrument mask with no part
labels. We estimate the jaws location geometrically:
  1. PCA on mask pixels -> principal (shaft) axis
  2. Project pixels onto that axis
  3. The distal end is the one FURTHER from the image border (trocar entry)
  4. Tip = centroid of the distal TAIL_FRAC of projected pixels

Validation: run on ground-truth instrument masks (union of 10,20,30),
compare estimated tip against the true jaws (30) centroid. This isolates
estimator error from detection error.
"""
import cv2, numpy as np
from pathlib import Path

RAW    = Path("/mnt/d/surgical-twin/data/raw")
ACTIVE = dict(x=328, y=32, w=1272, h=1016)
FX     = 1125.55        # rectified focal length
Z_NOM  = 0.055          # measured median working distance (m)

def border_dist(p):
    x, y = p
    return min(x-ACTIVE['x'], ACTIVE['x']+ACTIVE['w']-x,
               y-ACTIVE['y'], ACTIVE['y']+ACTIVE['h']-y)

def estimate_tip(mask, tail_frac=0.15):
    ys, xs = np.nonzero(mask)
    if xs.size < 100:
        return None, None
    pts  = np.stack([xs, ys], 1).astype(np.float64)
    mean = pts.mean(0)
    c    = pts - mean
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    axis = Vt[0]                                   # principal direction

    t    = c @ axis
    pos_end = mean + axis * t.max()
    neg_end = mean + axis * t.min()

    # distal = further from the border the instrument entered through
    if border_dist(pos_end) >= border_dist(neg_end):
        sel = t >= np.quantile(t, 1 - tail_frac)
    else:
        sel = t <= np.quantile(t, tail_frac)

    tip = pts[sel].mean(0)
    return tip, mean            # tip, whole-instrument centroid

rows = []
for tf in ("instrument_1_4_training", "instrument_5_8_training"):
    for ds in sorted((RAW/tf).glob("instrument_dataset_*")):
        gt = ds/"ground_truth"
        if not gt.exists(): continue
        for inst in sorted(gt.iterdir()):
            if not inst.is_dir(): continue
            for mp in sorted(inst.glob("*.png")):
                a = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if a is None: continue
                jaws = (a == 30)
                if jaws.sum() < 150: continue
                whole = np.isin(a, (10, 20, 30))
                if whole.sum() < 500: continue

                M = cv2.moments(jaws.astype(np.uint8))
                true_tip = np.array([M['m10']/M['m00'], M['m01']/M['m00']])

                est_tip, centroid = estimate_tip(whole.astype(np.uint8))
                if est_tip is None: continue

                rows.append((np.linalg.norm(est_tip - true_tip),
                             np.linalg.norm(centroid - true_tip),
                             ds.name, inst.name))

err_tip = np.array([r[0] for r in rows])
err_cen = np.array([r[1] for r in rows])

def mm(px): return px * Z_NOM / FX * 1000

print(f"n = {len(rows)} instrument instances\n")
print(f"{'method':26s} {'mean px':>9s} {'median px':>10s} "
      f"{'p90 px':>8s} {'median mm':>10s}")
print("-"*68)
print(f"{'whole-mask centroid (old)':26s} {err_cen.mean():9.1f} "
      f"{np.median(err_cen):10.1f} {np.percentile(err_cen,90):8.1f} "
      f"{mm(np.median(err_cen)):10.1f}")
print(f"{'PCA distal tip (new)':26s} {err_tip.mean():9.1f} "
      f"{np.median(err_tip):10.1f} {np.percentile(err_tip,90):8.1f} "
      f"{mm(np.median(err_tip)):10.1f}")

imp = (1 - np.median(err_tip)/np.median(err_cen)) * 100
print(f"\nmedian error reduction: {imp:.1f}%")
print(f"  {mm(np.median(err_cen)):.1f} mm -> {mm(np.median(err_tip)):.1f} mm "
      f"(at Z={Z_NOM*1000:.0f} mm)")

print(f"\nper-instrument-type median tip error (px):")
from collections import defaultdict
by = defaultdict(list)
for e, _, _, inst in rows: by[inst].append(e)
for k in sorted(by):
    v = np.array(by[k])
    print(f"  {k:36s} n={len(v):4d}  {np.median(v):6.1f} px  "
          f"{mm(np.median(v)):5.1f} mm")
