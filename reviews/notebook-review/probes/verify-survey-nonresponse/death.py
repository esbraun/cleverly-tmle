import numpy as np, cleverly
from scipy.special import expit
assert "bridge-cse" in cleverly.__file__, cleverly.__file__
from cleverly.datasets import missing_outcome_dgp
law = missing_outcome_dgp(strength=2.0)
rng = np.random.default_rng(1)
w = rng.normal(size=(4_000_000, 3))
worst = -7.0
res = {}
for a in (0.0, 1.0):
    m = law.outcome_mean(w, a, None); pi = law.missingness(w, a)
    pd_ = expit(-3.5 + 0.8*w[:,0] - 0.5*a)
    truth = np.mean(pd_*worst + (1-pd_)*m)
    obs = pd_ + (1-pd_)*pi
    shipped = np.mean((pd_*worst + (1-pd_)*pi*m)/obs)
    res[a] = (truth, shipped, pd_.mean())
    print(a, "truth", round(truth,4), "shipped", round(shipped,4), "bias", round(shipped-truth,4), "mort", round(pd_.mean(),4))
print("ATE truth", res[1][0]-res[0][0], "shipped", res[1][1]-res[0][1], "bias", (res[1][1]-res[0][1])-(res[1][0]-res[0][0]))
# also: survivor-MAR two-part formula equals truth by construction; check with no deaths shipped==truth
