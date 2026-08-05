"""
Two checks on config-C depth:
 1. Is the flat region a stationary instrument or a stuck matcher?
    -> compare depth variance against 2D centroid motion
 2. Is depth sampling the INSTRUMENT or the tissue behind it?
    -> compare depth inside the mask vs. in a ring outside it
"""
import cv2, numpy as np, yaml
from pathlib import Path
from ultralytics import YOLO

WS  = Path("/home/inoruske/surgical_twin_ws")
cfg = yaml.safe_load(open(WS/"config/camera_calib.yaml"))
W,H = cfg['image_width'], cfg['image_height']
L,R_= cfg['left'], cfg['right']
K1=np.array([[L['fx'],0,L['cx']],[0,L['fy'],L['cy']],[0,0,1]])
K2=np.array([[R_['fx'],0,R_['cx']],[0,R_['fy'],R_['cy']],[0,0,1]])
D1=np.array(L['dist']); D2=np.array(R_['dist'])
Rm=np.array(cfg['stereo']['R']); Tv=np.array(cfg['stereo']['T_m']).reshape(3,1)

R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K1,D1,K2,D2,(W,H),Rm,Tv,
                                     flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
m1x,m1y=cv2.initUndistortRectifyMap(K1,D1,R1,P1,(W,H),cv2.CV_32FC1)
m2x,m2y=cv2.initUndistortRectifyMap(K2,D2,R2,P2,(W,H),cv2.CV_32FC1)
fx_r=P1[0,0]; B_r=abs(P2[0,3]/P2[0,0])

sg=cv2.StereoSGBM_create(minDisparity=0,numDisparities=192,blockSize=7,
    P1=8*3*7**2,P2=32*3*7**2,disp12MaxDiff=1,uniquenessRatio=12,
    speckleWindowSize=150,speckleRange=2,preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

model=YOLO(str(WS/"models/best.pt"))
lefts=sorted((WS/"test_data").glob("*.png"))[:80]
rights=sorted((WS/"test_data_right").glob("*.png"))[:80]

rows=[]
for i,(lf,rf) in enumerate(zip(lefts,rights)):
    li,ri=cv2.imread(str(lf)),cv2.imread(str(rf))
    res=model(li,conf=0.5,verbose=False)[0]
    if res.masks is None: continue
    lr=cv2.remap(li,m1x,m1y,cv2.INTER_LINEAR); rr=cv2.remap(ri,m2x,m2y,cv2.INTER_LINEAR)
    d=sg.compute(cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY),
                 cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0
    d[d<=0.5]=np.nan

    for box,mask in zip(res.boxes,res.masks):
        name=model.names[int(box.cls[0])]
        mk=(cv2.resize(mask.data[0].cpu().numpy(),(W,H))>0.5).astype(np.uint8)
        M=cv2.moments(mk)
        if M['m00']==0: continue
        u,v=M['m10']/M['m00'], M['m01']/M['m00']

        mk_r=cv2.remap(mk,m1x,m1y,cv2.INTER_NEAREST)
        ring=cv2.dilate(mk_r,np.ones((41,41),np.uint8))-cv2.dilate(mk_r,np.ones((9,9),np.uint8))

        din = d[mk_r>0];  din = din[np.isfinite(din)]
        dout= d[ring>0];  dout= dout[np.isfinite(dout)]
        if din.size<50 or dout.size<50: continue

        z_in  = fx_r*B_r/np.median(din)
        z_out = fx_r*B_r/np.median(dout)
        rows.append((i,name,u,v,z_in,z_out,din.size))

import collections
by=collections.defaultdict(list)
for r in rows: by[r[1]].append(r)

print(f"{'class':28s} {'N':>4s} {'z_mask':>8s} {'z_ring':>8s} {'Δ(mm)':>7s}")
print("-"*62)
for name,rs in by.items():
    zi=np.array([r[4] for r in rs]); zo=np.array([r[5] for r in rs])
    print(f"{name:28s} {len(rs):4d} {zi.mean():8.4f} {zo.mean():8.4f} "
          f"{(zo.mean()-zi.mean())*1000:7.1f}")

print(f"\n--- flat-region check: 2D motion vs depth change ---")
print(f"{'class':28s} {'frames':>12s} {'2D px/frame':>12s} {'Δz mm/frame':>12s}")
print("-"*70)
for name,rs in by.items():
    rs=sorted(rs)
    for lo,hi in [(0,20),(20,32),(32,47),(47,80)]:
        seg=[r for r in rs if lo<=r[0]<hi]
        if len(seg)<5: continue
        p=np.array([[r[2],r[3]] for r in seg])
        z=np.array([r[4] for r in seg])
        d2=np.linalg.norm(np.diff(p,axis=0),axis=1).mean()
        dz=np.abs(np.diff(z)).mean()*1000
        print(f"{name:28s} {f'{lo}-{hi}':>12s} {d2:12.2f} {dz:12.2f}")
