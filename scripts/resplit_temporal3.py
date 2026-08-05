"""
Temporal split, v3 — balanced.

Rationale:
  Maryland_Bipolar_Forceps (ds1, frames 0-75) and Grasping_Retractor_Right
  (ds8, frames 0-157) each occur in ONE sequence in ONE contiguous block.
  Forcing them into val costs ~30% of all training data and inverts the
  train/val ratio for unrelated classes.

  Instead: apply a uniform 75/25 temporal cut everywhere. The two rare
  classes remain training-only and are reported as NOT TEMPORALLY
  EVALUABLE -- an honest, documented dataset limitation rather than a
  statistically worthless 18-sample mAP.
"""
import re, shutil, yaml
from pathlib import Path
from collections import defaultdict

WS  = Path("/home/inoruske/surgical_twin_ws")
SRC = WS/"data/processed_v2"
DST = WS/"data/processed_temporal"
CUT, GUARD = 160, 8            # train 0-160, guard 161-168, val 169-224

CLASS_NAMES = ["Large_Needle_Driver_Left","Large_Needle_Driver_Right",
               "Prograsp_Forceps_Left","Prograsp_Forceps_Right",
               "Maryland_Bipolar_Forceps","Bipolar_Forceps",
               "Monopolar_Curved_Scissors","Grasping_Retractor_Right"]
pat = re.compile(r"(instrument_dataset_\d+)_frame(\d+)")

if DST.exists(): shutil.rmtree(DST)
for sp in ("train","val"):
    (DST/sp/"images").mkdir(parents=True); (DST/sp/"labels").mkdir(parents=True)

cnt = defaultdict(int)
cc  = {"train": defaultdict(int), "val": defaultdict(int)}

for sp in ("train","val"):
    for lbl in (SRC/sp/"labels").glob("*.txt"):
        m = pat.match(lbl.stem)
        if not m: continue
        f = int(m.group(2))
        if   f <= CUT:         split = "train"
        elif f >  CUT + GUARD: split = "val"
        else:
            cnt["guard"] += 1; continue
        img = SRC/sp/"images"/f"{lbl.stem}.png"
        if not img.exists(): continue
        shutil.copy(img, DST/split/"images"/img.name)
        shutil.copy(lbl, DST/split/"labels"/lbl.name)
        cnt[split] += 1
        for line in lbl.read_text().splitlines():
            if line.strip(): cc[split][int(line.split()[0])] += 1

print(f"Protocol: train frames 0-{CUT}, guard {CUT+1}-{CUT+GUARD}, "
      f"val {CUT+GUARD+1}-224")
print(f"train {cnt['train']}   val {cnt['val']}   guard-dropped {cnt['guard']}")
print(f"split ratio: {cnt['train']/(cnt['train']+cnt['val'])*100:.0f}% / "
      f"{cnt['val']/(cnt['train']+cnt['val'])*100:.0f}%")

print(f"\n{'class':30s} {'train':>7s} {'val':>7s}   status")
print("-"*70)
evaluable = []
for i,n in enumerate(CLASS_NAMES):
    t,v = cc['train'][i], cc['val'][i]
    if   v == 0: st = "TRAIN-ONLY (not temporally evaluable)"
    elif v < 30: st = "sparse - caveat required"
    else:        st = "evaluable"; evaluable.append(i)
    print(f"{n:30s} {t:7d} {v:7d}   {st}")

print(f"\n{len(evaluable)}/8 classes temporally evaluable: "
      f"{[CLASS_NAMES[i] for i in evaluable]}")

yaml.safe_dump({"path": str(DST), "train":"train/images", "val":"val/images",
                "nc":8, "names":CLASS_NAMES},
               open(WS/"data/surgical_temporal.yaml","w"),
               default_flow_style=False, sort_keys=False)
print("config -> data/surgical_temporal.yaml")
