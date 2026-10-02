"""Where does the exact binned density give large ratios? seed 9000, in sample (no learner)."""
import sys, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from cleverly.datasets import make_shift_dose, shift_dgp
from cleverly.learners.density import bin_edges, fit_conditional_density
from cleverly.learners.crossfit import Folds
from tests.studies.canonical_shift_policies import OracleShiftDensity
from scipy.stats import norm
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 40
frame, _ = make_shift_dose(n=3000, seed=seed)
a = np.asarray(frame["A"]); w = np.column_stack([np.asarray(frame[c]) for c in ("W1","W2","W3")])
mu = 2.0 + 0.7*w[:,0] - 0.3*w[:,1]
edges = bin_edges(a, nb)
# exact binned density computed directly from the normal law
lo = np.where(np.arange(nb)==0, -np.inf, edges[:-1]); hi = np.where(np.arange(nb)==nb-1, np.inf, edges[1:])
P = norm.cdf(hi[None,:]-mu[:,None]) - norm.cdf(lo[None,:]-mu[:,None])
width = np.diff(edges)
def dens(x):
    b = np.clip(np.digitize(x, edges)-1, 0, nb-1); out = (x<edges[0])|(x>edges[-1])
    return np.where(out, 0.0, P[np.arange(x.size), b]/width[b]), b
for delta in (0.5, 1.0):
    num, bn = dens(a-delta); den, bd = dens(a)
    h = num/den; ht = np.exp(delta*(a-mu)-delta**2/2)
    print(f"delta={delta} bins={nb}: mean h {h.mean():.3f} (true {ht.mean():.3f}) mean h^2 {np.mean(h**2):.3f} (true {np.mean(ht**2):.3f}, pop {np.exp(delta**2):.3f}) max {h.max():.1f} (true max {ht.max():.1f}) corr log {np.corrcoef(np.log(np.maximum(h,1e-300)), np.log(ht))[0,1]:.3f}")
    top = np.argsort(h)[::-1][:6]
    for i in top:
        print(f"   a={a[i]:.3f} mu={mu[i]:.3f} bin(a)={bd[i]} [{edges[bd[i]]:.2f},{edges[bd[i]+1]:.2f}] bin(a-d)={bn[i]} h={h[i]:.1f} true={ht[i]:.2f}")
print("edge widths first/last 3:", np.round(width[:3],3), np.round(width[-3:],3))
