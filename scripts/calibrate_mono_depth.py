"""
Fit relative monocular depth to metric stereo depth.

Depth Anything V2 outputs inverse relative depth d_rel. The metric
relationship is:
        1/Z  =  a * d_rel  +  b
so we fit (a, b) by least squares against SGBM stereo depth sampled at
the same instrument tip locations, then report the residual error.

Only stereo samples that pass the pipeline's own quality gates are used
as reference, so the fit is against depths the system would actually trust.
"""
import cv2, numpy as np, yaml, torch
from pathlib import Path
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS  = Path("/home/inoruske/surgical_twin_ws")
cfg = yaml.safe_load(open(WS/"config/camera_calib.yaml"))
W,H = cfg['image_width'], cfg['image_height']
L,R_= cfg['left'], cfg['right']
ACTIVE, TAIL = cfg['active_region'], 0.08

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
    return min(x-ACTIVE['x'],ACTIVE['x']+ACTIVE['w']-x,
               y-ACTIVE['y'],ACTIVE['y']+ACTIVE['h']-y)

def tip_of(mb):
    ys,xs=np.nonzero(mb)
    if xs.size<100: return None
    pts=np.stack([xs,ys],1).astype(float); mean=pts.mean(0); c=pts-mean
    _,_,Vt=np.linalg.svd(c,full_matrices=False); ax=Vt[0]; t=c@ax
    distal=bd(mean+ax*t.max())>=bd(mean+ax*t.min())
    sel=t>=np.quantile(t,1-TAIL) if distal else t<=np.quantile(t,TAIL)
    return pts[sel].mean(0) if sel.sum()>10 else None

yolo=YOLO(str(WS/"models/best_temporal.pt"))
proc=AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
dam=AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf").to("cuda").eval()

lefts=sorted((WS/"demo_data").glob("*.png"))
rights=sorted((WS/"demo_data_right").glob("*.png"))
print(f"{len(lefts)} stereo pairs\n")

Z_stereo, d_rel_v, labels = [], [], []
for lf,rf in zip(lefts,rights):
    li,ri=cv2.imread(str(lf)),cv2.imread(str(rf))
    res=yolo(li,conf=0.5,verbose=False)[0]
    if res.masks is None: continue

    lr=cv2.remap(li,m1x,m1y,cv2.INTER_LINEAR); rr=cv2.remap(ri,m2x,m2y,cv2.INTER_LINEAR)
    disp=sg.compute(cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0
    disp[disp<=0.5]=np.nan

    rgb=cv2.cvtColor(li,cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        o=dam(**proc(images=cv2.resize(rgb,(518,518)),return_tensors="pt").to("cuda"))
    dmap=cv2.resize(o.predicted_depth[0].cpu().numpy().astype(np.float32),(W,H))

    for box,mask in zip(res.boxes,res.masks):
        mb=(cv2.resize(mask.data[0].cpu().numpy(),(W,H))>0.5).astype(np.uint8)
        tp=tip_of(mb)
        if tp is None: continue
        u,v=tp
        pr=cv2.undistortPoints(np.array([[[u,v]]]),K1,D1,R=R1,P=P1)[0,0]
        ui,vi=int(round(pr[0])),int(round(pr[1]))
        if not (0<=ui<W and 0<=vi<H): continue
        patch=disp[max(0,vi-7):vi+8, max(0,ui-7):ui+8]
        patch=patch[np.isfinite(patch)]
        if patch.size<12 or patch.std()>6.0: continue
        Z=fx_r*B_r/float(np.median(patch))
        if not (0.02<=Z<=0.15): continue

        dpatch=dmap[max(0,int(v)-7):int(v)+8, max(0,int(u)-7):int(u)+8]
        if dpatch.size<20: continue

        Z_stereo.append(Z)
        d_rel_v.append(float(np.median(dpatch)))
        labels.append(yolo.names[int(box.cls[0])])

Z=np.array(Z_stereo); d=np.array(d_rel_v)
print(f"paired samples: {len(Z)}")
if len(Z)<30:
    raise SystemExit("too few samples for a reliable fit")

# 1/Z = a*d + b
A=np.stack([d,np.ones_like(d)],1)
(a,b),*_=np.linalg.lstsq(A,1.0/Z,rcond=None)
Zhat=1.0/(a*d+b)
err=np.abs(Zhat-Z)*1000

print(f"\nfit:  1/Z = {a:.6f} * d_rel + {b:.6f}")
print(f"      corr(d_rel, 1/Z) = {np.corrcoef(d,1.0/Z)[0,1]:.4f}")
print(f"\nresidual error (mm):")
print(f"  mean {err.mean():.2f}   median {np.median(err):.2f}   "
      f"p90 {np.percentile(err,90):.2f}   max {err.max():.2f}")
print(f"  stereo range {Z.min()*1000:.1f}..{Z.max()*1000:.1f} mm")

yaml.safe_dump({'model':'depth-anything/Depth-Anything-V2-Small-hf',
                'a':float(a),'b':float(b),
                'n_samples':int(len(Z)),
                'median_err_mm':float(np.median(err)),
                'note':'1/Z_metres = a*d_rel + b, fitted against SGBM stereo'},
               open(WS/"config/mono_depth_calib.yaml","w"),
               default_flow_style=False, sort_keys=False)

fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].scatter(d,1.0/Z,s=12,alpha=.5,color="#028090")
xs=np.linspace(d.min(),d.max(),50)
ax[0].plot(xs,a*xs+b,'r-',lw=2)
ax[0].set_xlabel("relative depth (model output)"); ax[0].set_ylabel("1/Z stereo (1/m)")
ax[0].set_title(f"linear fit, r={np.corrcoef(d,1.0/Z)[0,1]:.3f}"); ax[0].grid(alpha=.3)
ax[1].scatter(Z*1000,Zhat*1000,s=12,alpha=.5,color="#02C39A")
lim=[Z.min()*1000,Z.max()*1000]
ax[1].plot(lim,lim,'k--',lw=1)
ax[1].set_xlabel("stereo depth (mm)"); ax[1].set_ylabel("mono depth, calibrated (mm)")
ax[1].set_title(f"median error {np.median(err):.1f} mm"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(WS/"results/mono_depth_calib.png",dpi=150)
print(f"\nsaved -> config/mono_depth_calib.yaml, results/mono_depth_calib.png")
