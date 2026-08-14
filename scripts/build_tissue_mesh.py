"""
Reconstruct the tissue surface from the stereo disparity field.

The pipeline already computes dense disparity across the whole frame but
samples it only at instrument tips. Back-projecting the full field gives
the actual observed tissue surface, textured with the endoscopic frame.

This is pipeline output rather than scenery: the geometry is derived from
the same stereo computation the depth measurements use, so it also serves
as a visual check that the disparity field is coherent away from the
sampled points.

Writes an OBJ + MTL + texture into models/tissue/.
"""
import cv2, numpy as np, yaml, sys
from pathlib import Path

WS   = Path("/home/inoruske/surgical_twin_ws")
SEQ  = sys.argv[1] if len(sys.argv) > 1 else "instrument_dataset_8"
FR   = sys.argv[2] if len(sys.argv) > 2 else None
OUT  = WS/"models/tissue"
STEP = 5            # mesh vertex spacing in pixels
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"meshes").mkdir(exist_ok=True)

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
fx,fy = P1[0,0], P1[1,1]
cx,cy = P1[0,2], P1[1,2]
B     = abs(P2[0,3]/P2[0,0])

lefts = sorted((WS/"demo_data").glob(f"{SEQ}*.png"))
if not lefts:
    raise SystemExit(f"no frames for {SEQ} in demo_data")
lf = next((f for f in lefts if FR and FR in f.name), lefts[len(lefts)//2])
rf = WS/"demo_data_right"/lf.name
print(f"frame: {lf.name}")

li, ri = cv2.imread(str(lf)), cv2.imread(str(rf))
lr = cv2.remap(li, m1x, m1y, cv2.INTER_LINEAR)
rr = cv2.remap(ri, m2x, m2y, cv2.INTER_LINEAR)

sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=192, blockSize=9,
     P1=8*3*9**2, P2=32*3*9**2, disp12MaxDiff=1, uniquenessRatio=10,
     speckleWindowSize=200, speckleRange=2, preFilterCap=63,
     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp = sg.compute(cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY),
                  cv2.cvtColor(rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0
disp[disp <= 1.0] = np.nan

# smooth the field; the surface is continuous even where matching is noisy
valid = np.isfinite(disp)
filled = np.where(valid, disp, 0).astype(np.float32)
sm = cv2.blur(filled, (15,15)) / np.maximum(cv2.blur(valid.astype(np.float32),(15,15)), 1e-6)
disp = np.where(valid, disp, np.nan)
# median filter first: box blur smears speckle rather than removing it,
# and the residual is what shreds the mesh
# medianBlur with kernel > 5 requires 8-bit input, so work in a scaled
# fixed-point representation. Median removes speckle that box blur only
# smears, and the residual speckle is what shreds the mesh.
dmax = np.nanmax(disp)
d8 = np.clip(np.nan_to_num(disp, nan=0) / dmax * 255.0, 0, 255).astype(np.uint8)
for k in (9, 15, 21):
    d8 = cv2.medianBlur(d8, k)
d8 = cv2.bilateralFilter(d8, 15, 45, 45)
d0 = d8.astype(np.float32) / 255.0 * dmax
disp = np.where(np.isfinite(disp) & (d0 > 1.0), d0, np.nan)

# active region in rectified coordinates
mask_raw = np.zeros((H,W), np.uint8)
mask_raw[AR['y']:AR['y']+AR['h'], AR['x']:AR['x']+AR['w']] = 255
active = cv2.remap(mask_raw, m1x, m1y, cv2.INTER_NEAREST) > 127
# erode to avoid rectification border artefacts
active = cv2.erode(active.astype(np.uint8), np.ones((45,45),np.uint8)) > 0

ys = np.arange(0, H, STEP); xs = np.arange(0, W, STEP)
gi = -np.ones((len(ys), len(xs)), int)
verts, uvs = [], []
ZMIN, ZMAX = 0.040, 0.130

for iy, y in enumerate(ys):
    for ix, x in enumerate(xs):
        if not active[y, x]: continue
        d = disp[y, x]
        if not np.isfinite(d) or d <= 1.0: continue
        Z = fx*B/d
        if not (ZMIN <= Z <= ZMAX): continue
        X = (x-cx)*Z/fx; Y = (y-cy)*Z/fy
        gi[iy, ix] = len(verts)
        # optical -> world (REP-103), camera at z=0.5
        verts.append((Z, -X, 0.5-Y))
        uvs.append((x/W, 1.0 - y/H))

faces = []
MAXEDGE = 0.006          # drop triangles spanning a depth discontinuity
Varr = np.array(verts) if verts else np.zeros((0,3))
for iy in range(len(ys)-1):
    for ix in range(len(xs)-1):
        a,b,c,d = gi[iy,ix], gi[iy,ix+1], gi[iy+1,ix], gi[iy+1,ix+1]
        for tri in ((a,b,c), (b,d,c)):
            if min(tri) < 0: continue
            p = Varr[list(tri)]
            if max(np.linalg.norm(p[0]-p[1]), np.linalg.norm(p[1]-p[2]),
                   np.linalg.norm(p[2]-p[0])) > MAXEDGE: continue
            faces.append(tri)

print(f"vertices {len(verts)}, faces {len(faces)}")
if len(faces) < 100:
    raise SystemExit("too few faces - check disparity coverage")

cv2.imwrite(str(OUT/"meshes/tissue.png"), lr)
with open(OUT/"meshes/tissue.mtl","w") as f:
    f.write("newmtl tissue\nKa 0.35 0.30 0.30\nKd 1.0 1.0 1.0\n"
            "Ks 0.12 0.12 0.12\nNs 12\nmap_Kd tissue.png\n")
with open(OUT/"meshes/tissue.obj","w") as f:
    f.write("mtllib tissue.mtl\nusemtl tissue\n")
    for v in verts: f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
    for u in uvs:   f.write(f"vt {u[0]:.5f} {u[1]:.5f}\n")
    for t in faces: f.write(f"f {t[0]+1}/{t[0]+1} {t[1]+1}/{t[1]+1} {t[2]+1}/{t[2]+1}\n")

open(OUT/"model.config","w").write(
 '<?xml version="1.0"?>\n<model><name>tissue</name><version>1.0</version>'
 '<sdf version="1.8">model.sdf</sdf>'
 '<description>Tissue surface reconstructed from the stereo disparity field.'
 '</description></model>\n')
open(OUT/"model.sdf","w").write(
 '<?xml version="1.0" ?>\n<sdf version="1.8">\n  <model name="tissue">\n'
 '    <static>true</static>\n    <link name="surface">\n'
 '      <visual name="v">\n        <geometry><mesh>\n'
 '          <uri>model://tissue/meshes/tissue.obj</uri>\n'
 '        </mesh></geometry>\n      </visual>\n'
 '    </link>\n  </model>\n</sdf>\n')

Z = Varr[:,0]
print(f"depth range {Z.min()*1000:.0f}..{Z.max()*1000:.0f} mm")
print(f"-> models/tissue/")
