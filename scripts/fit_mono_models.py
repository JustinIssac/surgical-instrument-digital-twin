"""
Compare candidate mono->metric depth models.

The linear-in-inverse-depth fit shows monotonic bias (+4mm near, -48mm far),
so it is misspecified. Try richer models, but validate with 5-fold CV --
with only 19 samples above 95mm, a flexible model can trivially overfit
the sparse far range and look better than it is.
"""
import numpy as np, yaml, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/home/inoruske/surgical_twin_ws")

# reuse the pairing code, stop before the fit
exec(open(WS/"calibrate_mono_depth.py").read().split("Z=np.array(Z_stereo)")[0])
Z = np.array(Z_stereo); d = np.array(d_rel_v)
print(f"\n{len(Z)} paired samples\n")

def m_linear_inv(dtr, Ztr):
    A = np.stack([dtr, np.ones_like(dtr)], 1)
    c, *_ = np.linalg.lstsq(A, 1.0/Ztr, rcond=None)
    return lambda x: 1.0/(c[0]*x + c[1]), f"1/Z={c[0]:.4f}d+{c[1]:.4f}"

def m_linear_z(dtr, Ztr):
    A = np.stack([dtr, np.ones_like(dtr)], 1)
    c, *_ = np.linalg.lstsq(A, Ztr, rcond=None)
    return lambda x: c[0]*x + c[1], f"Z={c[0]:.4f}d+{c[1]:.4f}"

def m_poly2_inv(dtr, Ztr):
    c = np.polyfit(dtr, 1.0/Ztr, 2)
    return lambda x: 1.0/np.polyval(c, x), "1/Z=quad(d)"

def m_poly3_inv(dtr, Ztr):
    c = np.polyfit(dtr, 1.0/Ztr, 3)
    return lambda x: 1.0/np.clip(np.polyval(c, x), 1e-3, None), "1/Z=cubic(d)"

def m_loglog(dtr, Ztr):
    A = np.stack([np.log(np.clip(dtr,1e-3,None)), np.ones_like(dtr)], 1)
    c, *_ = np.linalg.lstsq(A, np.log(Ztr), rcond=None)
    return lambda x: np.exp(c[0]*np.log(np.clip(x,1e-3,None)) + c[1]), \
           f"Z={np.exp(c[1]):.4f}*d^{c[0]:.3f}"

def m_isotonic(dtr, Ztr):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
    ir.fit(dtr, Ztr)
    return lambda x: ir.predict(x), "isotonic (monotone, nonparametric)"

def m_piecewise(dtr, Ztr, knot=2.6):
    """Separate linear-inverse fits either side of a knot in d_rel."""
    lo, hi = dtr <= knot, dtr > knot
    out = []
    for m in (lo, hi):
        if m.sum() >= 5:
            A = np.stack([dtr[m], np.ones(m.sum())], 1)
            c, *_ = np.linalg.lstsq(A, 1.0/Ztr[m], rcond=None)
        else:
            A = np.stack([dtr, np.ones_like(dtr)], 1)
            c, *_ = np.linalg.lstsq(A, 1.0/Ztr, rcond=None)
        out.append(c)
    def f(x):
        x = np.atleast_1d(x)
        r = np.where(x <= knot,
                     1.0/(out[0][0]*x + out[0][1]),
                     1.0/(out[1][0]*x + out[1][1]))
        return r
    return f, f"piecewise @ d={knot}"

MODELS = [("linear (1/Z)", m_linear_inv), ("linear (Z)", m_linear_z),
          ("quadratic (1/Z)", m_poly2_inv), ("cubic (1/Z)", m_poly3_inv),
          ("power law", m_loglog), ("piecewise", m_piecewise)]
try:
    import sklearn; MODELS.append(("isotonic", m_isotonic))
except ImportError:
    print("(sklearn absent -- skipping isotonic)\n")

rng = np.random.default_rng(0)
idx = rng.permutation(len(Z))
folds = np.array_split(idx, 5)

print(f"{'model':22s} {'CV med':>8s} {'CV p90':>8s} "
      f"{'bias<80':>9s} {'bias>95':>9s}  form")
print("-"*78)
results = {}
for name, fn in MODELS:
    errs, preds = [], np.full(len(Z), np.nan)
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        try:
            f, form = fn(d[tr], Z[tr])
            p = np.asarray(f(d[te])).ravel()
        except Exception as e:
            form = f"FAILED ({e})"; break
        preds[te] = p
        errs.append(np.abs(p - Z[te]) * 1000)
    if not errs: 
        print(f"{name:22s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>9s}  {form}")
        continue
    e = np.concatenate(errs)
    sg = (preds - Z) * 1000
    near = sg[Z*1000 < 80]; far = sg[Z*1000 > 95]
    results[name] = (np.median(e), np.percentile(e,90), preds)
    print(f"{name:22s} {np.median(e):8.2f} {np.percentile(e,90):8.2f} "
          f"{np.nanmean(near):+9.1f} {np.nanmean(far):+9.1f}  {form}")

best = min(results, key=lambda k: results[k][0])
print(f"\nlowest CV median error: {best} ({results[best][0]:.2f} mm)")

fig, ax = plt.subplots(1, 2, figsize=(12,4.5))
for name in results:
    p = results[name][2]
    ax[0].scatter(Z*1000, (p-Z)*1000, s=8, alpha=.35, label=name)
ax[0].axhline(0, c='k', lw=1); ax[0].set_ylim(-70,45)
ax[0].set_xlabel("true depth (mm)"); ax[0].set_ylabel("error (mm)")
ax[0].set_title("Cross-validated error vs depth"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

names = list(results); meds = [results[n][0] for n in names]
ax[1].barh(names, meds, color="#028090")
ax[1].set_xlabel("CV median |error| (mm)"); ax[1].grid(alpha=.3, axis='x')
ax[1].set_title("Model comparison (5-fold CV)")
plt.tight_layout(); plt.savefig(WS/"results/mono_model_comparison.png", dpi=150)
print(f"saved -> results/mono_model_comparison.png")
