"""
Is the depth-range compression real, or an artefact of squashing
1920x1080 into a square 518x518 input?
"""
import cv2, numpy as np, torch, yaml
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

WS = Path("/home/inoruske/surgical_twin_ws")
cfg = yaml.safe_load(open(WS/"config/camera_calib.yaml"))
AR  = cfg['active_region']

proc  = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf").to("cuda").eval()

def infer(img):
    with torch.no_grad():
        o = model(**proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         return_tensors="pt").to("cuda"))
    return o.predicted_depth[0].cpu().numpy()

variants = {
    "square 518 (current)": lambda im: cv2.resize(im, (518, 518)),
    "aspect-preserved pad": lambda im: _pad(im, 518),
    "active region only":   lambda im: cv2.resize(
        im[AR['y']:AR['y']+AR['h'], AR['x']:AR['x']+AR['w']], (518, 518)),
    "active + aspect pad":  lambda im: _pad(
        im[AR['y']:AR['y']+AR['h'], AR['x']:AR['x']+AR['w']], 518),
    "native 700 wide":      lambda im: cv2.resize(im, (700, 392)),
}

def _pad(im, size):
    h, w = im.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(im, (int(w*s), int(h*s)))
    out = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size-r.shape[0])//2, (size-r.shape[1])//2
    out[y0:y0+r.shape[0], x0:x0+r.shape[1]] = r
    return out

frames = sorted((WS/"demo_data").glob("*.png"))[:25]
print(f"{len(frames)} frames\n")
print(f"{'preprocessing':24s} {'d_rel range':>16s} {'ratio':>7s} {'IQR':>7s}")
print("-"*60)

for name, fn in variants.items():
    lo, hi, iqr = [], [], []
    for f in frames:
        d = infer(fn(cv2.imread(str(f))))
        d = d[d > 0]
        if d.size == 0: continue
        lo.append(np.percentile(d, 2)); hi.append(np.percentile(d, 98))
        iqr.append(np.percentile(d, 75) - np.percentile(d, 25))
    lo, hi = np.mean(lo), np.mean(hi)
    print(f"{name:24s} {lo:7.2f} .. {hi:6.2f} {hi/lo:7.2f} {np.mean(iqr):7.2f}")

print("\nratio = 98th/2nd percentile of predicted inverse depth.")
print("Higher ratio => model is discriminating a wider depth span.")
print("Stereo ground truth spans 50-148mm, i.e. a 1/Z ratio of ~2.96.")
