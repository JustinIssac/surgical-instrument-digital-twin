"""
Tissue surface as a grid of shaded primitive tiles.

Gazebo Harmonic would not apply materials to the reconstructed mesh under
any of the approaches tried (MTL reference, SDF material block, PBR albedo
map, per-vertex colour), rendering it flat white. Primitives accept SDF
materials reliably, so the fitted surface is instead tiled with thin boxes.

Depth is still the quartic fit to the measured stereo disparity; only the
rendering changes. Tiles are shaded by depth so the surface reads as
three-dimensional under flat lighting.
"""
import cv2, numpy as np, yaml, sys
from pathlib import Path

WS  = Path("/home/inoruske/surgical_twin_ws")
SEQ = sys.argv[1] if len(sys.argv) > 1 else "instrument_dataset_8"
NX, NY, ORDER = 22, 18, 4
OUT = WS/"models/tissue"; OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(WS/"config/camera_calib.yaml"))
W,H = cfg['image_width'], cfg['image_height']
L,R_= cfg['left'], cfg['right']; AR = cfg['active_region']
K1=np.array([[L['fx'],0,L['cx']],[0,L['fy'],L['cy']],[0,0,1]])
K2=np.array([[R_['fx'],0,R_['cx']],[0,R_['fy'],R_['cy']],[0,0,1]])
D1=np.array(L['dist']); D2=np.array(R_['dist'])
Rm=np.array(cfg['stereo']['R']); Tv=np.array(cfg['stereo']['T_m']).reshape(3,1)
R1,R2,P1,P2,_,_,_=cv2.stereoRectify(K1,D1,K2,D2,(W,H),Rm,Tv,
                                     flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
m1x,m1y=cv2.initUndistortRectifyMap(K1,D1,R1,P1,(W,H),cv2.CV_32FC1)
m2x,m2y=cv2.initUndistortRectifyMap(K2,D2,R2,P2,(W,H),cv2.CV_32FC1)
fx,fy,cx,cy = P1[0,0],P1[1,1],P1[0,2],P1[1,2]
B = abs(P2[0,3]/P2[0,0])

lefts = sorted((WS/"demo_data").glob(f"{SEQ}*.png"))
lf = lefts[len(lefts)//2]
li = cv2.imread(str(lf)); ri = cv2.imread(str(WS/"demo_data_right"/lf.name))
lr = cv2.remap(li,m1x,m1y,cv2.INTER_LINEAR)
rr = cv2.remap(ri,m2x,m2y,cv2.INTER_LINEAR)
sg = cv2.StereoSGBM_create(minDisparity=0,numDisparities=192,blockSize=11,
     P1=8*3*11**2,P2=32*3*11**2,disp12MaxDiff=1,uniquenessRatio=12,
     speckleWindowSize=300,speckleRange=2,preFilterCap=63,
     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp = sg.compute(cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY),
                  cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0

mask = np.zeros((H,W),np.uint8)
mask[AR['y']:AR['y']+AR['h'], AR['x']:AR['x']+AR['w']] = 255
active = cv2.erode(cv2.remap(mask,m1x,m1y,cv2.INTER_NEAREST),
                   np.ones((60,60),np.uint8)) > 127
Z = fx*B/np.where(active & (disp>1.0), disp, np.nan)
ok = active & np.isfinite(Z) & (Z>0.035) & (Z<0.14)

ys,xs = np.nonzero(ok)
u=(xs-cx)/fx; v=(ys-cy)/fy
terms=[(i,j) for i in range(ORDER+1) for j in range(ORDER+1) if i+j<=ORDER]
A=np.stack([u**i*v**j for i,j in terms],1)
coef,*_=np.linalg.lstsq(A,1.0/Z[ok],rcond=None)
for _ in range(3):
    r=np.abs(A@coef-1.0/Z[ok]); k=r<2.5*r.std()
    coef,*_=np.linalg.lstsq(A[k],(1.0/Z[ok])[k],rcond=None)

def Zs(px,py):
    uu,vv=(px-cx)/fx,(py-cy)/fy
    return 1.0/np.clip(sum(c*uu**i*vv**j for c,(i,j) in zip(coef,terms)),1/0.18,1/0.035)

ay,ax = np.nonzero(active)
x0,x1,y0,y1 = ax.min(),ax.max(),ay.min(),ay.max()
gx = np.linspace(x0,x1,NX+1); gy = np.linspace(y0,y1,NY+1)

def world(px,py):
    Zp=float(Zs(px,py)); X=(px-cx)*Zp/fx; Y=(py-cy)*Zp/fy
    return np.array([Zp,-X,0.5-Y])

tiles=[]
depths=[]
for iy in range(NY):
    for ix in range(NX):
        pxc,pyc = (gx[ix]+gx[ix+1])/2, (gy[iy]+gy[iy+1])/2
        if not active[int(pyc),int(pxc)]: continue
        c  = world(pxc,pyc)
        e1 = world(gx[ix+1],pyc) - world(gx[ix],pyc)
        e2 = world(pxc,gy[iy+1]) - world(pxc,gy[iy])
        w_,h_ = np.linalg.norm(e1), np.linalg.norm(e2)
        # Build an orthonormal frame from the tile's own edges and convert
        # to RPY. Deriving the three angles independently (as before) does
        # not yield a consistent rotation and left tiles standing on edge.
        ex = e1/ (np.linalg.norm(e1)+1e-9)
        n  = np.cross(e1, e2); n /= (np.linalg.norm(n)+1e-9)
        if n[2] < 0: n = -n
        ey = np.cross(n, ex); ey /= (np.linalg.norm(ey)+1e-9)
        Rt = np.column_stack([ex, ey, n])          # box x,y,z -> world
        sy = np.hypot(Rt[0,0], Rt[1,0])
        if sy > 1e-6:
            roll  = np.arctan2(Rt[2,1], Rt[2,2])
            pitch = np.arctan2(-Rt[2,0], sy)
            yaw   = np.arctan2(Rt[1,0], Rt[0,0])
        else:
            roll  = np.arctan2(-Rt[1,2], Rt[1,1]); pitch = np.arctan2(-Rt[2,0], sy); yaw = 0.0
        tiles.append((c, w_*1.12, h_*1.12, roll, pitch, yaw))
        depths.append(c[0])

d = np.array(depths); dn = (d-d.min())/max(d.ptp(),1e-6)
links=[]
for i,((c,w_,h_,ro,pi,ya),t) in enumerate(zip(tiles,dn)):
    # nearer tissue lighter and warmer, deeper tissue darker
    r_ = 0.62-0.30*t; g_ = 0.30-0.17*t; b_ = 0.28-0.16*t
    links.append(f'''    <link name="t{i}">
      <pose>{c[0]:.5f} {c[1]:.5f} {c[2]:.5f} {ro:.4f} {pi:.4f} {ya:.4f}</pose>
      <visual name="v">
        <geometry><box><size>{w_:.5f} {h_:.5f} 0.0025</size></box></geometry>
        <material>
          <ambient>{r_*0.55:.3f} {g_*0.55:.3f} {b_*0.55:.3f} 1</ambient>
          <diffuse>{r_:.3f} {g_:.3f} {b_:.3f} 1</diffuse>
          <specular>0.10 0.07 0.07 1</specular>
        </material>
      </visual>
    </link>''')

(OUT/"model.sdf").write_text(
 '<?xml version="1.0" ?>\n<sdf version="1.8">\n  <model name="tissue">\n'
 '    <static>true</static>\n' + "\n".join(links) + '\n  </model>\n</sdf>\n')
(OUT/"model.config").write_text(
 '<?xml version="1.0"?>\n<model><name>tissue</name><version>1.0</version>'
 '<sdf version="1.8">model.sdf</sdf></model>\n')
print(f"{len(tiles)} tiles, depth {d.min()*1000:.0f}..{d.max()*1000:.0f} mm")
print(f"-> models/tissue/model.sdf")
