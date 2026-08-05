"""
Paired-correspondence epipolar test.

Match features ONCE in the raw pair, then push those SAME points through
the rectification transform. Identical sample set on both sides, so the
before/after comparison is valid. RANSAC removes mismatches.
"""
import cv2, numpy as np, yaml
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS  = Path("/home/inoruske/surgical_twin_ws")
cfg = yaml.safe_load(open(WS/"config/camera_calib.yaml"))
W,H = cfg['image_width'], cfg['image_height']
L,R_= cfg['left'], cfg['right']

K1 = np.array([[L['fx'],0,L['cx']],[0,L['fy'],L['cy']],[0,0,1]])
K2 = np.array([[R_['fx'],0,R_['cx']],[0,R_['fy'],R_['cy']],[0,0,1]])
D1 = np.array(L['dist']); D2 = np.array(R_['dist'])
Rm = np.array(cfg['stereo']['R']); Tv = np.array(cfg['stereo']['T_m']).reshape(3,1)

R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K1,D1,K2,D2,(W,H),Rm,Tv,
                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
ar = cfg['active_region']
def in_active(p):
    return (ar['x'] <= p[0] < ar['x']+ar['w']) and (ar['y'] <= p[1] < ar['y']+ar['h'])

sift = cv2.SIFT_create(nfeatures=4000)
bf   = cv2.BFMatcher(cv2.NORM_L2)

raw_e, rect_e = [], []
lefts  = sorted((WS/"test_data").glob("*.png"))[:25]
rights = sorted((WS/"test_data_right").glob("*.png"))[:25]

for lf, rf in zip(lefts, rights):
    lg = cv2.cvtColor(cv2.imread(str(lf)), cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(cv2.imread(str(rf)), cv2.COLOR_BGR2GRAY)
    k1,d1 = sift.detectAndCompute(lg, None)
    k2,d2 = sift.detectAndCompute(rg, None)
    if d1 is None or d2 is None: continue

    good=[]
    for pair in bf.knnMatch(d1,d2,k=2):
        if len(pair)<2: continue
        m,n = pair
        if m.distance < 0.65*n.distance:      # strict Lowe
            good.append(m)
    if len(good) < 20: continue

    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])

    keep = np.array([in_active(a) and in_active(b) for a,b in zip(p1,p2)])
    p1,p2 = p1[keep], p2[keep]
    if len(p1) < 20: continue

    # RANSAC on fundamental matrix -> geometric inliers only
    _, inl = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0, 0.99)
    if inl is None: continue
    inl = inl.ravel().astype(bool)
    p1,p2 = p1[inl], p2[inl]
    if len(p1) < 10: continue

    raw_e.append(np.abs(p1[:,1]-p2[:,1]))

    # SAME points -> rectified coordinates
    q1 = cv2.undistortPoints(p1.reshape(-1,1,2), K1, D1, R=R1, P=P1).reshape(-1,2)
    q2 = cv2.undistortPoints(p2.reshape(-1,1,2), K2, D2, R=R2, P=P2).reshape(-1,2)
    rect_e.append(np.abs(q1[:,1]-q2[:,1]))

raw  = np.concatenate(raw_e)
rect = np.concatenate(rect_e)

print(f"Paired correspondences: {raw.size}  (identical set both sides)\n")
print(f"{'':12s} {'mean':>8s} {'median':>8s} {'p90':>8s} {'p99':>8s} {'<1px':>7s}")
for lbl,a in [("RAW",raw),("RECTIFIED",rect)]:
    print(f"{lbl:12s} {a.mean():8.3f} {np.median(a):8.3f} "
          f"{np.percentile(a,90):8.3f} {np.percentile(a,99):8.3f} "
          f"{(a<1).mean()*100:6.1f}%")

print(f"\nMedian: {np.median(raw):.3f} -> {np.median(rect):.3f} px "
      f"({(1-np.median(rect)/np.median(raw))*100:+.1f}%)")

print(f"\n--- why the effect is small ---")
print(f"  cy_left - cy_right   = {L['cy']-R_['cy']:+.3f} px")
print(f"  rotation |omega|     = {np.linalg.norm(cv2.Rodrigues(Rm)[0]):.6f} rad")
print(f"  cx_left - cx_right   = {L['cx']-R_['cx']:+.3f} px  (horizontal)")

fig,ax = plt.subplots(figsize=(7,4))
bins = np.linspace(0,6,61)
ax.hist(raw,  bins=bins, alpha=.6, label=f"Raw (median {np.median(raw):.2f} px)",
        color="#CC3300")
ax.hist(rect, bins=bins, alpha=.6, label=f"Rectified (median {np.median(rect):.2f} px)",
        color="#02C39A")
ax.set_xlabel("Vertical epipolar error |$y_L-y_R$| (px)")
ax.set_ylabel("Correspondences"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Epipolar error, paired correspondences (RANSAC inliers)")
plt.tight_layout(); plt.savefig(WS/"results/epipolar_paired.png", dpi=150)
print(f"\nSaved -> results/epipolar_paired.png")
