"""
Generate a camera_calib.yaml from any EndoVis camera_calibration.txt.

The pipeline previously baked in dataset 1's parameters. Datasets 9/10
use a different scope (fx 1068->1080, cy 501->514, baseline 4.277->4.365mm),
so a single hardcoded file gives ~3% depth error on other sequences and
degrades rectification.

Usage:
  python3 make_calib.py <camera_calibration.txt> <frames_dir> <out.yaml>
"""
import sys, re, cv2, yaml
import numpy as np
from pathlib import Path

txt, frames_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]

vals = {}
for line in open(txt):
    m = re.match(r'([\w-]+):\s*([-\d.\s]+)', line)
    if m:
        vals[m.group(1)] = [float(x) for x in m.group(2).split()]

CW, CH = int(vals['Width'][0]), int(vals['Height'][0])
f0, c0 = vals['Camera-0-F'], vals['Camera-0-C']
f1, c1 = vals['Camera-1-F'], vals['Camera-1-C']
k0, k1 = vals['Camera-0-K'], vals['Camera-1-K']
omega  = np.array(vals['Extrinsic-Omega'])
T_mm   = np.array(vals['Extrinsic-T'])

# active region from temporal variance (padding has zero variance)
files = sorted(Path(frames_dir).glob('*.png'))[:40]
stack = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
                  .astype(np.float32) for f in files])
live = stack.var(axis=0) > 0.5
cols, rows = np.where(live.any(axis=0))[0], np.where(live.any(axis=1))[0]
ox, oy = int(cols.min()), int(rows.min())
aw, ah = int(cols.max()-ox+1), int(rows.max()-oy+1)
H, W = stack.shape[1], stack.shape[2]
s = ((aw/CW) + (ah/CH)) / 2.0

R, _ = cv2.Rodrigues(omega)
cfg = {
  'source_calibration': str(txt),
  'image_width': W, 'image_height': H,
  'active_region': {'x': ox, 'y': oy, 'w': aw, 'h': ah},
  'scale': float(round(s, 6)),
  'left':  {'fx': round(f0[0]*s,3), 'fy': round(f0[1]*s,3),
            'cx': round(c0[0]*s+ox,3), 'cy': round(c0[1]*s+oy,3),
            'dist': k0},
  'right': {'fx': round(f1[0]*s,3), 'fy': round(f1[1]*s,3),
            'cx': round(c1[0]*s+ox,3), 'cy': round(c1[1]*s+oy,3),
            'dist': k1},
  'stereo': {'R': R.tolist(), 'T_m': (T_mm/1000.0).tolist(),
             'baseline_m': float(abs(T_mm[0])/1000.0)},
}
yaml.safe_dump(cfg, open(out,'w'), default_flow_style=False, sort_keys=False)

print(f"{Path(txt).parent.name}:")
print(f"  frame {W}x{H}, active {aw}x{ah} at ({ox},{oy}), scale {s:.5f}")
print(f"  fx {cfg['left']['fx']:.2f}  fy {cfg['left']['fy']:.2f}  "
      f"ratio {cfg['left']['fx']/cfg['left']['fy']:.5f}")
print(f"  cx {cfg['left']['cx']:.2f}  cy {cfg['left']['cy']:.2f}")
print(f"  baseline {cfg['stereo']['baseline_m']*1000:.4f} mm")
print(f"  -> {out}")
