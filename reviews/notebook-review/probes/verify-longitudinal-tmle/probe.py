"""Verifier probe for LT-01 and LT-02. Own code; reads only make_longitudinal and numpy/sklearn."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, LinearRegression
import cleverly
from cleverly.datasets import make_longitudinal

print("cleverly at", cleverly.__file__)


def cal_slope(p, y):
    """Unpenalized logistic recalibration slope of y on logit(p), one intercept."""
    m = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-12)
    m.fit(logit(np.clip(p, 1e-12, 1 - 1e-12)).reshape(-1, 1), y)
    return float(m.coef_[0, 0])


f, _ = make_longitudinal(n=8000, seed=41, cluster_size=20)
obs = (f["C1"] == 1) & (f["C2"] == 1)
d = f[obs]
y = d["Y"].to_numpy()

designs = {
    "notebook form W1 W2 A1 L2 A2": ["W1", "W2", "A1", "L2", "A2"],
    "wrong: W1 only": ["W1"],
    "wrong: A2 only": ["A2"],
}
for name, cols in designs.items():
    for pen, kw in (("default C=1", {}), ("unpenalized", {"C": np.inf})):
        m = LogisticRegression(max_iter=5000, tol=1e-12, **kw).fit(d[cols], y)
        p = m.predict_proba(d[cols])[:, 1]
        print(f"LT-02 {name:32s} {pen:12s} in-sample slope {cal_slope(p, y):.4f}")

# LT-01 nonzero witness: best main-term logistic vs true outcome probability on a big draw.
big, _ = make_longitudinal(n=400_000, seed=7, censoring=False)
X = big[["W1", "W2", "A1", "L2", "A2"]].to_numpy()
l2 = big["L2"].to_numpy()
true_logit = (-0.4 + 0.5 * big["A1"] + 0.8 * big["A2"] + 0.4 * l2 + 0.3 * big["W1"]
              - 0.2 * big["W2"] + 0.5 * np.tanh(l2)).to_numpy()
ptrue = expit(true_logit)
yb = big["Y"].to_numpy()
mt = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-10).fit(X, yb)
pm = mt.predict_proba(X)[:, 1]
print("LT-01 check Y probability formula: mean |y-p_true| residual mean", float(np.mean(yb - ptrue)))
print("LT-01 main-term logistic vs truth: max |p - p_true|", float(np.max(np.abs(pm - ptrue))),
      " RMS", float(np.sqrt(np.mean((pm - ptrue) ** 2))))
Xt = np.column_stack([X, np.tanh(l2)])
mtt = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-10).fit(Xt, yb)
print("LT-01 with tanh(L2) term: coef", np.round(mtt.coef_[0], 3), "intercept", np.round(mtt.intercept_, 3))
# Node-1 pseudo-outcome under 'always': E[Qbar2(W,A1=1,L2,A2=1) | W, A1=1] -- linearity check
a1 = big["A1"].to_numpy() == 1
q2 = expit(-0.4 + 0.5 + 0.8 + 0.4 * l2 + 0.3 * big["W1"].to_numpy() - 0.2 * big["W2"].to_numpy()
           + 0.5 * np.tanh(l2))
W = big[["W1", "W2"]].to_numpy()[a1]
lin = LinearRegression().fit(W, q2[a1])
quad = LinearRegression().fit(np.column_stack([W, W ** 2, W[:, 0] * W[:, 1]]), q2[a1])
print("LT-01 node-1 pseudo-outcome (always): R2 linear", round(lin.score(W, q2[a1]), 4),
      " R2 quadratic", round(quad.score(np.column_stack([W, W ** 2, W[:, 0] * W[:, 1]]), q2[a1]), 4))
