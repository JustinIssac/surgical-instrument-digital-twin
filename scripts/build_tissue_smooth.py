"""
Smooth tissue surface from the stereo disparity field.

Meshing the raw disparity produced a fragmented surface: the field is too
speckled at vertex scale even after median and bilateral filtering. This
instead fits a low-order polynomial to the disparity, giving the overall
cavity shape without the noise, and applies the endoscopic image as
per-vertex colour rather than a texture map.

The result is a smoothed representation of the observed surface, not a
per-pixel reconstruction, and should be described as such.
"""
import cv2, numpy as np, yaml, sys
from pathlib import Path

WS  = Path("/home/inoruske/surgical_twin_ws")
SEQ = sys.argv[1] if len(sys.argv) > 1 else "instrument_dataset_8"
OUT = WS/"models/tissue"; (OUT/"meshes").mkdir(parents=True, exist_ok=True)
STEP, ORDER = 7, 4

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
fx,fy,cx,cy = P1[0,0],P1[1,1],P1[0,2],P1[1,2]
B = abs(P2[0,3]/P2[0,0])

lefts = sorted((WS/"demo_data").glob(f"{SEQ}*.png"))
lf = lefts[len(lefts)//2]; rf = WS/"demo_data_right"/lf.name
print(f"frame: {lf.name}")
li, ri = cv2.imread(str(lf)), cv2.imread(str(rf))
lr = cv2.remap(li,m1x,m1y,cv2.INTER_LINEAR)
rr = cv2.remap(ri,m2x,m2y,cv2.INTER_LINEAR)

sg = cv2.StereoSGBM_create(minDisparity=0,numDisparities=192,blockSize=11,
     P1=8*3*11**2,P2=32*3*11**2,disp12MaxDiff=1,uniquenessRatio=12,
     speckleWindowSize=300,speckleRange=2,preFilterCap=63,
     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp = sg.compute(cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY),
                  cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0

mask_raw = np.zeros((H,W),np.uint8)
mask_raw[AR['y']:AR['y']+AR['h'], AR['x']:AR['x']+AR['w']] = 255
active = cv2.erode(cv2.remap(mask_raw,m1x,m1y,cv2.INTER_NEAREST),
                   np.ones((55,55),np.uint8)) > 127

ok = active & (disp > 1.0) & np.isfinite(disp)
Zall = fx*B/np.where(ok, disp, np.nan)
ok &= (Zall > 0.035) & (Zall < 0.14)
print(f"usable disparity over {ok.sum()/active.sum()*100:.0f}% of the active region")

# robust polynomial fit of inverse depth over image coordinates
ys,xs = np.nonzero(ok)
zs = 1.0/Zall[ok]
u = (xs-cx)/fx; v = (ys-cy)/fy
terms = [(i,j) for i in range(ORDER+1) for j in range(ORDER+1) if i+j <= ORDER]
A = np.stack([u**i * v**j for i,j in terms], 1)
coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
for _ in range(3):                     # reject outliers, refit
    r = np.abs(A@coef - zs)
    keep = r < 2.5*np.std(r)
    coef, *_ = np.linalg.lstsq(A[keep], zs[keep], rcond=None)
resid = np.abs(A@coef - zs)
print(f"fit residual: median {np.median(1/(A@coef)[resid<1]*0+resid.mean())*0+np.median(resid):.3f} 1/m")

def Zsurf(px, py):
    uu, vv = (px-cx)/fx, (py-cy)/fy
    inv = sum(c * uu**i * vv**j for c,(i,j) in zip(coef, terms))
    return 1.0/np.clip(inv, 1/0.20, 1/0.03)

gy = np.arange(0,H,STEP); gx = np.arange(0,W,STEP)
gi = -np.ones((len(gy),len(gx)),int)
verts, cols = [], []
for iy,y in enumerate(gy):
    for ix,x in enumerate(gx):
        if not active[y,x]: continue
        Z = float(Zsurf(x,y))
        X = (x-cx)*Z/fx; Y = (y-cy)*Z/fy
        gi[iy,ix] = len(verts)
        verts.append((Z, -X, 0.5-Y))
        b,g,r = lr[y,x]
        cols.append((r/255.0, g/255.0, b/255.0))

faces=[]
for iy in range(len(gy)-1):
    for ix in range(len(gx)-1):
        a,b_,c,d = gi[iy,ix],gi[iy,ix+1],gi[iy+1,ix],gi[iy+1,ix+1]
        if min(a,b_,c) >= 0: faces.append((a,b_,c))
        if min(b_,d,c) >= 0: faces.append((b_,d,c))

print(f"vertices {len(verts)}, faces {len(faces)}")
with open(OUT/"meshes/tissue.obj","w") as f:
    f.write("# smoothed tissue surface, per-vertex colour\n")
    for (vx,vy,vz),(r,g,b) in zip(verts,cols):
        f.write(f"v {vx:.5f} {vy:.5f} {vz:.5f} {r:.4f} {g:.4f} {b:.4f}\n")
    for t in faces:
        f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

(OUT/"model.sdf").write_text(f'''<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="tissue">
    <static>true</static>
    <link name="surface">
      <visual name="v">
        <geometry><mesh><uri>file://{OUT}/meshes/tissue.obj</uri></mesh></geometry>
        <material>
          <ambient>0.42 0.26 0.25 1</ambient>
          <diffuse>0.78 0.50 0.47 1</diffuse>
          <specular>0.18 0.14 0.14 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
''')
Z = np.array([v[0] for v in verts])
print(f"depth {Z.min()*1000:.0f}..{Z.max()*1000:.0f} mm  ->  models/tissue/")
