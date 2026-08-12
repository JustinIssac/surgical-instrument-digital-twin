"""
Does the model distinguish Left/Right from instrument APPEARANCE or from
IMAGE POSITION?

Mirroring the active image region moves a left-side instrument to the right.
A model reading appearance keeps its label; one reading position flips it.

Note fliplr=0.5 was used during training with UNCHANGED labels, which
should have taught position-invariance -- or taught that side is noise.
This has never been tested.
"""
import cv2, numpy as np, sys
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

WS   = Path("/home/inoruske/surgical_twin_ws")
MODEL = str(WS/"models/best_temporal.pt")

NAMES = ['Large_Needle_Driver_Left','Large_Needle_Driver_Right',
         'Prograsp_Forceps_Left','Prograsp_Forceps_Right',
         'Maryland_Bipolar_Forceps','Bipolar_Forceps',
         'Monopolar_Curved_Scissors','Grasping_Retractor_Right']
MIRROR = {0:1, 1:0, 2:3, 3:2, 4:4, 5:5, 6:6, 7:7}
SIDED  = {0,1,2,3}

def mirror_active(img, ar):
    out = img.copy()
    x,y,w,h = ar['x'], ar['y'], ar['w'], ar['h']
    out[y:y+h, x:x+w] = img[y:y+h, x:x+w][:, ::-1]
    return out

def detect(model, img):
    r = model(img, conf=0.5, verbose=False)[0]
    if r.boxes is None: return []
    return [(int(b.cls[0]), float(b.conf[0]),
             float((b.xyxy[0][0]+b.xyxy[0][2])/2)) for b in r.boxes]

def active_region(files, n=25):
    st = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
                   .astype(np.float32) for f in files[:n]])
    live = st.var(axis=0) > 0.5
    c, r = np.where(live.any(axis=0))[0], np.where(live.any(axis=1))[0]
    return dict(x=int(c.min()), y=int(r.min()),
                w=int(c.max()-c.min()+1), h=int(r.max()-r.min()+1))

model = YOLO(MODEL)

for label, folder, lim in [
        ("SEEN (demo_data)", WS/"demo_data", None),
        ("UNSEEN ds9", WS/"data/endovis_test/instrument_dataset_9/left_frames", 150),
        ("UNSEEN ds10", WS/"data/endovis_test/instrument_dataset_10/left_frames", 150)]:
    files = sorted(Path(folder).glob("*.png"))
    if not files: 
        print(f"{label}: no frames"); continue
    if lim: files = files[:lim]
    ar = active_region(files)

    same = flip = other = 0
    per  = defaultdict(lambda: {'same':0,'flip':0,'other':0})
    pos  = defaultdict(list)

    for fp in files:
        img = cv2.imread(str(fp))
        do, dm = detect(model, img), detect(model, mirror_active(img, ar))
        for cid, _, cx in do:
            if cid in SIDED:
                pos[cid].append((cx - ar['x']) / ar['w'])
        for cid, _, cx in do:
            cxm = 2*ar['x'] + ar['w'] - cx
            best, bd = None, 1e9
            for cid2, _, cx2 in dm:
                if abs(cx2 - cxm) < bd: best, bd = cid2, abs(cx2 - cxm)
            if best is None or bd > 120: continue
            if best == cid:            same += 1;  per[cid]['same'] += 1
            elif best == MIRROR[cid]:  flip += 1;  per[cid]['flip'] += 1
            else:                      other += 1; per[cid]['other'] += 1

    tot = same + flip + other
    print(f"\n{'='*66}\n{label}   ({len(files)} frames, {tot} matched detections)")
    if tot == 0: continue
    print(f"  label UNCHANGED under mirror : {same:5d}  ({same/tot*100:5.1f}%)")
    print(f"  label FLIPPED  L<->R         : {flip:5d}  ({flip/tot*100:5.1f}%)")
    print(f"  other class                  : {other:5d}  ({other/tot*100:5.1f}%)")
    print(f"\n  {'class':28s} {'same':>5s} {'flip':>5s} {'other':>6s}  reading")
    for cid in sorted(per):
        d = per[cid]; t = sum(d.values())
        if not t: continue
        fr = d['flip']/t
        tag = ("POSITION-dependent" if fr > 0.6 else
               "appearance-dependent" if fr < 0.25 else "MIXED/unstable")
        print(f"  {NAMES[cid]:28s} {d['same']:5d} {d['flip']:5d} "
              f"{d['other']:6d}  {tag}")

    if pos:
        print(f"\n  where sided classes appear (0=left edge, 1=right edge):")
        for cid in sorted(pos):
            v = np.array(pos[cid])
            print(f"    {NAMES[cid]:28s} n={len(v):4d}  mean {v.mean():.3f}  "
                  f"[{np.percentile(v,10):.2f}, {np.percentile(v,90):.2f}]")
