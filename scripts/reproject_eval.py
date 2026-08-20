"""
End-to-end tip accuracy on PREDICTED masks, against ground-truth claspers.

WHAT THIS MEASURES
  For each validation frame: YOLO predicts an instrument mask, the PCA
  estimator locates the tip, and that tip is compared against the nearest
  ground-truth clasper centroid (part label 30). Error is reported in
  pixels and converted to millimetres at the estimated stereo depth.

  This is the end-to-end figure. The existing 2.81mm was measured on
  ground-truth masks, isolating estimator error; the difference between
  the two is the segmentation's contribution.

WHAT THIS DOES NOT MEASURE
  It does not validate depth or back-projection. Back-projecting a pixel
  to 3D and projecting it back is identity, so those stages cancel and
  cannot be tested this way. Depth enters only in the px->mm conversion,
  scaling the result rather than being checked by it.

MATCHING
  Each predicted tip is matched to the NEAREST ground-truth clasper within
  MATCH_PX, without regard to class. Left/right classification is unstable
  for identical instrument pairs (measured separately: 48% flip rate on
  ds10) and folding it in would conflate classification with geometry.
"""
import sys, cv2, yaml, json
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

WS    = Path("/home/inoruske/surgical_twin_ws")
RAW   = Path("/mnt/d/surgical-twin/data/raw")
VAL   = WS/"data/processed_temporal/val/images"
MODEL = WS/"models/best_temporal.pt"
OUT   = WS/"results/reprojection_wide.csv"

MATCH_PX  = 400      # widened to characterise the tail; report at 150
CLASPER   = 30       # part label: 10=shaft, 20=wrist, 30=claspers
MIN_AREA  = 80       # px, minimum clasper region to trust a centroid
CONF      = 0.25
# TAIL_FRAC imported from perception_node below -- not duplicated


def tf_for(ds):
    return "instrument_1_4_training" if ds <= 4 else "instrument_5_8_training"


def load_calib(ds):
    c = yaml.safe_load(open(WS/f"config/camera_calib_ds{ds}.yaml"))
    L, R_ = c['left'], c['right']
    K1 = np.array([[L['fx'],0,L['cx']],[0,L['fy'],L['cy']],[0,0,1]], float)
    K2 = np.array([[R_['fx'],0,R_['cx']],[0,R_['fy'],R_['cy']],[0,0,1]], float)
    D1 = np.array(L['dist'], float); D2 = np.array(R_['dist'], float)
    R  = np.array(c['stereo']['R'], float)
    T  = np.array(c['stereo']['T_m'], float).reshape(3,1)
    size = (c['image_width'], c['image_height'])
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K1,D1,K2,D2,size,R,T,
                                          flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1 = cv2.initUndistortRectifyMap(K1,D1,R1,P1,size,cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2,D2,R2,P2,size,cv2.CV_32FC1)
    ar = c['active_region']
    mask = np.zeros((size[1],size[0]), np.uint8)
    mask[ar['y']:ar['y']+ar['h'], ar['x']:ar['x']+ar['w']] = 255
    valid = cv2.remap(mask, m1[0], m1[1], cv2.INTER_NEAREST) > 127
    return dict(K1=K1,D1=D1,R1=R1,P1=P1,m1=m1,m2=m2,valid=valid,
                fx=float(P1[0,0]), baseline=float(abs(P2[0,3]/P2[0,0])))


def gt_claspers(ds, frame):
    """All ground-truth clasper centroids in this frame."""
    gt = RAW/tf_for(ds)/f"instrument_dataset_{ds}"/"ground_truth"
    out = []
    if not gt.exists():
        return out
    for inst in sorted(gt.iterdir()):
        if not inst.is_dir():
            continue
        p = inst/f"{frame}.png"
        if not p.exists():
            continue
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        m = (a == CLASPER).astype(np.uint8)
        if m.sum() < MIN_AREA:
            continue
        M = cv2.moments(m)
        out.append((M['m10']/M['m00'], M['m01']/M['m00'], inst.name))
    return out


# ---- tip estimator: reuse the deployed one, do not reimplement ----------
sys.path.insert(0, str(WS/"src/surgical_perception/surgical_perception"))
try:
    from perception_node import PerceptionNode
    _tip = PerceptionNode.estimate_tip
    from perception_node import TAIL_FRAC
    print(f"using PerceptionNode.estimate_tip, tail_frac={TAIL_FRAC}")

    class _Shim:
        """Carries only what estimate_tip reads off self.

        The deployed node hardcodes one active region; here it is set per
        dataset from that dataset's own calibration, since the regions
        differ (1272-1280 wide, origin x 320-328).
        """
        def __init__(self, ar, tail_frac):
            self.active = ar
            self.tail_frac = tail_frac
        _border_dist_pt = PerceptionNode._border_dist_pt
except Exception as e:
    print(f"!! could not import the deployed tip estimator: {e}")
    print("!! STOP -- reimplementing it would not be comparable to 2.81mm")
    raise SystemExit(1)


def main():
    model = YOLO(str(MODEL))
    files = sorted(VAL.glob("*.png"))
    print(f"{len(files)} validation frames")

    by_ds = defaultdict(list)
    for f in files:
        ds = int(f.stem.split("_frame")[0].replace("instrument_dataset_",""))
        by_ds[ds].append(f)

    rows = []
    skip = defaultdict(int)

    for ds in sorted(by_ds):
        cal = load_calib(ds)
        _c = yaml.safe_load(open(WS/f"config/camera_calib_ds{ds}.yaml"))
        shim = _Shim(_c['active_region'], TAIL_FRAC)
        src = RAW/tf_for(ds)/f"instrument_dataset_{ds}"
        sg = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=96, blockSize=7,
            P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=1, uniquenessRatio=12,
            speckleWindowSize=150, speckleRange=2, preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        for f in by_ds[ds]:
            frame = f.stem.split("_frame")[1]
            frame = f"frame{frame}"
            lp = src/"left_frames"/f"{frame}.png"
            rp = src/"right_frames"/f"{frame}.png"
            if not lp.exists() or not rp.exists():
                skip['noframe'] += 1; continue

            gts = gt_claspers(ds, frame)
            if not gts:
                skip['nogt'] += 1; continue

            li = cv2.imread(str(lp)); ri = cv2.imread(str(rp))
            H, W = li.shape[:2]

            lr = cv2.remap(li, cal['m1'][0], cal['m1'][1], cv2.INTER_LINEAR)
            rr = cv2.remap(ri, cal['m2'][0], cal['m2'][1], cv2.INTER_LINEAR)
            d = sg.compute(cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0
            d[~cal['valid']] = np.nan
            d[d <= 0.5] = np.nan

            res = model(str(lp), conf=CONF, verbose=False)[0]
            if res.masks is None:
                skip['nodet'] += 1; continue

            for box, mk in zip(res.boxes, res.masks):
                cls  = int(box.cls[0]); conf = float(box.conf[0])
                name = model.names[cls]
                m = cv2.resize(mk.data[0].cpu().numpy(), (W, H)) > 0.4
                if m.sum() < 50:
                    skip['tinymask'] += 1; continue
                tip, cen, yaw, alen = _tip(shim, m.astype(np.uint8))
                if tip is None:
                    skip['tipfail'] += 1; continue
                tx, ty = float(tip[0]), float(tip[1])

                # nearest ground-truth clasper, no class matching
                dists = [((tx-gx)**2 + (ty-gy)**2)**0.5 for gx, gy, _ in gts]
                j = int(np.argmin(dists)); err_px = dists[j]
                if err_px > MATCH_PX:
                    skip['nomatch'] += 1; continue

                # depth, sampled as stereo_depth_node does
                pt = cv2.undistortPoints(np.array([[[tx,ty]]]), cal['K1'],
                                         cal['D1'], R=cal['R1'], P=cal['P1'])
                ur, vr = int(round(pt[0,0,0])), int(round(pt[0,0,1]))
                z = None
                if 0 <= ur < W and 0 <= vr < H:
                    patch = d[max(0,vr-7):vr+8, max(0,ur-7):ur+8]
                    v = patch[~np.isnan(patch)]
                    if v.size >= 12 and float(np.std(v)) <= 6.0:
                        z = cal['fx']*cal['baseline']/float(np.median(v))
                        if not (0.02 <= z <= 0.15):
                            z = None

                err_mm = err_px * z / cal['fx'] * 1000 if z else ''
                rows.append([ds, frame, name, f"{conf:.3f}",
                             f"{tx:.1f}", f"{ty:.1f}",
                             f"{gts[j][0]:.1f}", f"{gts[j][1]:.1f}", gts[j][2],
                             f"{err_px:.2f}",
                             f"{z*1000:.1f}" if z else '',
                             f"{err_mm:.2f}" if err_mm != '' else ''])

        print(f"  ds{ds}: {len(rows)} matches so far")

    OUT.parent.mkdir(exist_ok=True)
    import csv
    with open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset','frame','pred_class','conf','tip_x','tip_y',
                    'gt_x','gt_y','gt_instrument','err_px','depth_mm','err_mm'])
        w.writerows(rows)

    print(f"\n{len(rows)} matched tips -> {OUT}")
    print(f"excluded: {dict(skip)}")

    e = np.array([float(r[9]) for r in rows])
    mm = np.array([float(r[11]) for r in rows if r[11] != ''])
    print(f"\nerror px : median {np.median(e):.1f}  IQR {np.percentile(e,25):.1f}-{np.percentile(e,75):.1f}")
    if mm.size:
        print(f"error mm : median {np.median(mm):.2f}  IQR {np.percentile(mm,25):.2f}-{np.percentile(mm,75):.2f}  (n={mm.size})")


if __name__ == "__main__":
    main()
