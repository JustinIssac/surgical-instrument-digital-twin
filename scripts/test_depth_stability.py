"""
Decisive test: does corrected geometry produce temporally stable depth?

Three configurations, same frames, same detections:
  A  OLD  : raw frames, fx=1602.59, numDisp=64   (original code)
  B  MID  : raw frames, corrected fx             (intrinsics only)
  C  NEW  : rectified frames, corrected fx       (current code)

Metric = frame-to-frame depth change. A real instrument at 10 FPS
cannot move more than ~1 cm axially per frame.
"""
import cv2, numpy as np, yaml, json
from pathlib import Path
from ultralytics import YOLO
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS  = Path("/home/inoruske/surgical_twin_ws")
cfg = yaml.safe_load(open(WS/"config/camera_calib.yaml"))
W,H = cfg['image_width'], cfg['image_height']
L,R_= cfg['left'], cfg['right']

K1=np.array([[L['fx'],0,L['cx']],[0,L['fy'],L['cy']],[0,0,1]])
K2=np.array([[R_['fx'],0,R_['cx']],[0,R_['fy'],R_['cy']],[0,0,1]])
D1=np.array(L['dist']); D2=np.array(R_['dist'])
Rm=np.array(cfg['stereo']['R']); Tv=np.array(cfg['stereo']['T_m']).reshape(3,1)
B = cfg['stereo']['baseline_m']

R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K1,D1,K2,D2,(W,H),Rm,Tv,
                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
m1x,m1y = cv2.initUndistortRectifyMap(K1,D1,R1,P1,(W,H),cv2.CV_32FC1)
m2x,m2y = cv2.initUndistortRectifyMap(K2,D2,R2,P2,(W,H),cv2.CV_32FC1)
fx_rect = P1[0,0]; B_rect = abs(P2[0,3]/P2[0,0])
print(f"fx_rect={fx_rect:.2f}  B_rect={B_rect*1000:.4f} mm")
print(f"cx1_rect={P1[0,2]:.2f}  cx2_rect={P2[0,2]:.2f}  "
      f"(equal => zero-disparity enforced)\n")

def sgbm(nd):
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=7,
        P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=1,
        uniquenessRatio=12, speckleWindowSize=150, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

sg64, sg192 = sgbm(64), sgbm(192)

def sample(disp, u, v, win=7):
    h,w = disp.shape
    ui,vi = int(round(u)), int(round(v))
    if not (0<=ui<w and 0<=vi<h): return None
    p = disp[max(0,vi-win):min(h,vi+win+1), max(0,ui-win):min(w,ui+win+1)]
    p = p[np.isfinite(p) & (p>0.5)]
    if p.size < 12 or np.std(p) > 6.0: return None
    return float(np.median(p))

model  = YOLO(str(WS/"models/best.pt"))
lefts  = sorted((WS/"test_data").glob("*.png"))[:80]
rights = sorted((WS/"test_data_right").glob("*.png"))[:80]

series = {"A_old":{}, "B_mid":{}, "C_new":{}}

for i,(lf,rf) in enumerate(zip(lefts,rights)):
    li, ri = cv2.imread(str(lf)), cv2.imread(str(rf))
    res = model(li, conf=0.5, verbose=False)[0]
    if res.masks is None: continue

    lg, rg = cv2.cvtColor(li,cv2.COLOR_BGR2GRAY), cv2.cvtColor(ri,cv2.COLOR_BGR2GRAY)
    d_raw = sg64.compute(lg,rg).astype(np.float32)/16.0
    d_raw[d_raw<=0]=np.nan

    lr = cv2.remap(li,m1x,m1y,cv2.INTER_LINEAR)
    rr = cv2.remap(ri,m2x,m2y,cv2.INTER_LINEAR)
    d_rect = sg192.compute(cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0
    d_rect[d_rect<=0]=np.nan

    for box,mask in zip(res.boxes,res.masks):
        cid = int(box.cls[0]); name = model.names[cid]
        mk  = cv2.resize(mask.data[0].cpu().numpy(),(W,H))
        M   = cv2.moments((mk>0.5).astype(np.uint8))
        if M['m00']==0: continue
        u,v = M['m10']/M['m00'], M['m01']/M['m00']

        dr = sample(d_raw,u,v)
        if dr:
            series["A_old"].setdefault(name,[]).append((i, 1602.59*B/dr))
            series["B_mid"].setdefault(name,[]).append((i, L['fx']*B/dr))

        pt = cv2.undistortPoints(np.array([[[u,v]]]),K1,D1,R=R1,P=P1)[0,0]
        dc = sample(d_rect, pt[0], pt[1])
        if dc:
            series["C_new"].setdefault(name,[]).append((i, fx_rect*B_rect/dc))

print(f"{'config':8s} {'class':28s} {'N':>4s} {'mean_d':>8s} {'med|Δ|':>8s} {'p95|Δ|':>8s}")
print("-"*70)
for cfgname in ("A_old","B_mid","C_new"):
    for name,pts in series[cfgname].items():
        if len(pts)<10: continue
        z = np.array([p[1] for p in pts])
        dz= np.abs(np.diff(z))
        print(f"{cfgname:8s} {name:28s} {len(z):4d} {z.mean():8.4f} "
              f"{np.median(dz):8.4f} {np.percentile(dz,95):8.4f}")

fig,axes = plt.subplots(3,1,figsize=(10,9),sharex=True)
titles = {"A_old":"A — original (raw frames, fx=1602.59, numDisp=64)",
          "B_mid":"B — corrected intrinsics only",
          "C_new":"C — rectified + corrected (current)"}
for ax,(k,t) in zip(axes, titles.items()):
    for name,pts in series[k].items():
        if len(pts)<10: continue
        ax.plot([p[0] for p in pts],[p[1] for p in pts],'.-',ms=3,lw=.8,label=name)
    ax.axhspan(0.03,0.20,color='green',alpha=.07)
    ax.set_title(t,fontsize=10); ax.set_ylabel("depth (m)"); ax.grid(alpha=.3)
    ax.legend(fontsize=6,loc='upper right')
axes[-1].set_xlabel("frame")
plt.tight_layout(); plt.savefig(WS/"results/depth_stability.png",dpi=150)
print(f"\nSaved -> results/depth_stability.png")
