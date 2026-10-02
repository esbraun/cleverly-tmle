"""Independent truths from the structural equations (no library truth helper)."""
import numpy as np
from scipy.special import expit

rng = np.random.default_rng(12345)
N = 4_000_000
W = rng.normal(size=(N, 3))
w1, w2, w3 = W.T

def q(a, s):
    e = s - 1.0
    lin = 1.0 + 1.2 * a + 0.9 * w1 + 0.6 * w2 - 0.4 * w3
    return lin + e * (1.1 * np.tanh(1.5 * w1) + 0.8 * w2**2 - 0.9 * a * w1)

def pi(a, s):
    e = s - 1.0
    return expit(1.2 + 0.6 * a - (0.8 + 0.6 * e) * w1 + 0.3 * w3)

g = expit(0.4 * w1 - 0.3 * w2)
for s in (1.0, 2.0):
    cate = q(1, s) - q(0, s)
    p_resp = g * pi(1, s) + (1 - g) * pi(0, s)  # P(Delta=1|W)
    ate = cate.mean()
    resp_eff = np.sum(p_resp * cate) / np.sum(p_resp)
    ew1 = np.sum(p_resp * w1) / np.sum(p_resp)
    print(f"strength {s}: ATE={ate:.4f}  P(Delta=1)={p_resp.mean():.4f}  E[W1|D=1]={ew1:.4f}  "
          f"respondent-standardized effect={resp_eff:.4f}  "
          f"EY1={q(1,s).mean():.4f} EY0={q(0,s).mean():.4f}")
    # response by arm marginal
    print("  P(Delta=1|A=0)=", np.sum((1-g)*pi(0,s))/np.sum(1-g), " P(Delta=1|A=1)=", np.sum(g*pi(1,s))/np.sum(g))
    # min of product
    print("  1st pct of P(Delta=1|A,W):", np.quantile(np.where(rng.random(N)<g, pi(1,s), pi(0,s))[:400000], 0.01))

# binary law
gb = expit(0.4 * w1 - 0.3 * w2 + 0.2 * w3)
qb = lambda a: expit(-0.6 + 0.9 * a + 0.7 * w1 - 0.5 * w2 + 0.4 * w3)
e1, e0 = qb(1).mean(), qb(0).mean()
print(f"binary: ey1={e1:.4f} ey0={e0:.4f} ate={e1-e0:.4f} rr={e1/e0:.4f} or={(e1/(1-e1))/(e0/(1-e0)):.4f}")
