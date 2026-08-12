"""
Relative-anchoring: instead of mapping the tip's absolute d_rel to metres,
measure the tip's OFFSET from the surrounding tissue in relative space and
convert only that offset. The background plane provides a per-frame anchor,
removing scale drift and using the model where it is strongest (local
ordering) rather than weakest (absolute value).

    Z_tip = Z_bg_ref  -  k * (d_tip - d_bg)

Z_bg_ref is the median stereo depth of the tissue ring (a stable, textured
surface SGBM handles well); k is fitted once.
"""
import cv2, numpy as np, yaml, torch
from pathlib import Path
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

WS  = Path("/home/inoruske/surgical_twin_ws")
cfg = yaml.safe_load(open(WS/"config/camera_calib.yaml"))
W,H = cfg['image_width'], cfg['image_height']
L,R_= cfg['left'], cfg['right']; AR = cfg['active_region']

K1=np.array([[L['fx'],0,L['cx']],[0,L['fy'],L['cy']],[0,0,1]])
K2=np.array([[R_['fx'],0,R_['cx']],[0,R_['fy'],R_['cy']],[0,0,1]])
D1=np.array(L['dist']); D2=np.array(R_['dist'])
Rm=np.array(cfg['stereo']['R']); Tv=np.array(cfg['stereo']['T_m']).reshape(3,1)
R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K1,D1,K2,D2,(W,H),Rm,Tv,
                                     flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
m1x,m1y=cv2.initUndistortRectifyMap(K1,D1,R1,P1,(W,H),cv2.CV_32FC1)
m2x,m2y=cv2.initUndistortRectifyMap(K2,D2,R2,P2,(W,H),cv2.CV_32FC1)
fx_r,B_r=P1[0,0],abs(P2[0,3]/P2[0,0])

sg=cv2.StereoSGBM_create(minDisparity=0,numDisparities=96,blockSize=7,
   P1=8*3*7**2,P2=32*3*7**2,disp12MaxDiff=1,uniquenessRatio=12,
   speckleWindowSize=150,speckleRange=2,preFilterCap=63,
   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def bd(p):
    x,y=p
    return min(x-AR['x'],AR['x']+AR['w']-x,y-AR['y'],AR['y']+AR['h']-y)

def tip_of(mb):
    ys,xs=np.nonzero(mb)
    if xs.size<100: return None
    pts=np.stack([xs,ys],1).astype(float); mean=pts.mean(0); c=pts-mean
    _,_,Vt=np.linalg.svd(c,full_matrices=False); ax=Vt[0]; t=c@ax
    distal=bd(mean+ax*t.max())>=bd(mean+ax*t.min())
    sel=t>=np.quantile(t,0.92) if distal else t<=np.quantile(t,0.08)
    return pts[sel].mean(0) if sel.sum()>10 else None

yolo=YOLO(str(WS/"models/best_temporal.pt"))
proc=AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
dam=AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf").to("cuda").eval()

lefts=sorted((WS/"demo_data").glob("*.png"))
rights=sorted((WS/"demo_data_right").glob("*.png"))

Z_true, dd_rel, Zbg = [], [], []
for lf,rf in zip(lefts,rights):
    li,ri=cv2.imread(str(lf)),cv2.imread(str(rf))
    res=yolo(li,conf=0.5,verbose=False)[0]
    if res.masks is None: continue
    lr=cv2.remap(li,m1x,m1y,cv2.INTER_LINEAR); rr=cv2.remap(ri,m2x,m2y,cv2.INTER_LINEAR)
    disp=sg.compute(cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0
    disp[disp<=0.5]=np.nan

    with torch.no_grad():
        o=dam(**proc(images=cv2.cvtColor(li,cv2.COLOR_BGR2RGB),
                     return_tensors="pt").to("cuda"))
    dmap=cv2.resize(o.predicted_depth[0].cpu().numpy().astype(np.float32),(W,H))

    for box,mask in zip(res.boxes,res.masks):
        mb=(cv2.resize(mask.data[0].cpu().numpy(),(W,H))>0.5).astype(np.uint8)
        tp=tip_of(mb)
        if tp is None: continue
        u,v=tp

        # stereo depth at tip (ground truth)
        pr=cv2.undistortPoints(np.array([[[u,v]]]),K1,D1,R=R1,P=P1)[0,0]
        ui,vi=int(round(pr[0])),int(round(pr[1]))
        if not (0<=ui<W and 0<=vi<H): continue
        pt=disp[max(0,vi-7):vi+8, max(0,ui-7):ui+8]; pt=pt[np.isfinite(pt)]
        if pt.size<12 or pt.std()>6.0: continue
        Z=fx_r*B_r/float(np.median(pt))
        if not (0.02<=Z<=0.16): continue

        # background ring: dilate the mask, exclude the instrument
        ring=cv2.dilate(mb,np.ones((61,61),np.uint8))-cv2.dilate(mb,np.ones((15,15),np.uint8))
        ring_r=cv2.remap(ring,m1x,m1y,cv2.INTER_NEAREST)
        bgd=disp[ring_r>0]; bgd=bgd[np.isfinite(bgd)]
        if bgd.size<200: continue
        Zb=fx_r*B_r/float(np.median(bgd))

        d_tip=float(np.median(dmap[max(0,int(v)-7):int(v)+8,
                                   max(0,int(u)-7):int(u)+8]))
        d_bg =float(np.median(dmap[ring>0]))

        Z_true.append(Z); dd_rel.append(d_tip-d_bg); Zbg.append(Zb)

Z=np.array(Z_true); dd=np.array(dd_rel); Zb=np.array(Zbg)
print(f"\n{len(Z)} samples")
print(f"tip-vs-background relative offset: mean {dd.mean():.3f}  sd {dd.std():.3f}")

# Z_tip = Zb - k*dd
k,*_ = np.linalg.lstsq(dd.reshape(-1,1), (Zb-Z), rcond=None)
Zhat = Zb - k[0]*dd
err  = np.abs(Zhat-Z)*1000
print(f"\nfit: Z_tip = Z_bg - {k[0]:.5f} * (d_tip - d_bg)")
print(f"  corr(dd, Zb-Z) = {np.corrcoef(dd, Zb-Z)[0,1]:.4f}")
print(f"  median err {np.median(err):.2f} mm   p90 {np.percentile(err,90):.2f} mm")
print(f"  (global absolute fit was 5.53 mm median / 18.3 p90)")
print(f"\nrecovered depth range: {Zhat.min()*1000:.0f}..{Zhat.max()*1000:.0f} mm")
print(f"true range:             {Z.min()*1000:.0f}..{Z.max()*1000:.0f} mm")
