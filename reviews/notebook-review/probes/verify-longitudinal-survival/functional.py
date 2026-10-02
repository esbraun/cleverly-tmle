"""Independent death-as-censoring functional, by quadrature and by Monte Carlo of the recoded data."""
import numpy as np
from scipy.special import expit
import cleverly
from cleverly.datasets import longitudinal as L

print(cleverly.__file__)

pts, wts = np.polynomial.hermite_e.hermegauss(60)
wts = wts / np.sqrt(2 * np.pi)
w1 = pts.reshape(-1, 1, 1); w2 = pts.reshape(1, -1, 1); z = pts.reshape(1, 1, -1)
mass = wts.reshape(-1, 1, 1) * wts.reshape(1, -1, 1) * wts.reshape(1, 1, -1)


def functional(a1, a2, order="death_first"):
    h1 = L._hazard_one(w1, w2, a1); s1 = L._relapse_share_one(w1, w2, a1)
    l2 = L._L2["w1"] * w1 + L._L2["a1"] * a1 + z
    h2 = L._hazard_two(w1, w2, l2, a1, a2); s2 = L._relapse_share_two(w1, l2, a1, a2)
    if order == "death_first":
        lam1 = h1 * s1 / (1 - h1 * (1 - s1)); lam2 = h2 * s2 / (1 - h2 * (1 - s2))
    else:  # readmission hazard is the raw cause-specific hazard
        lam1 = h1 * s1; lam2 = h2 * s2
    return float(np.sum(mass * (lam1 + (1 - lam1) * lam2)))


def total(a1, a2):
    h1 = L._hazard_one(w1, w2, a1); s1 = L._relapse_share_one(w1, w2, a1)
    l2 = L._L2["w1"] * w1 + L._L2["a1"] * a1 + z
    h2 = L._hazard_two(w1, w2, l2, a1, a2); s2 = L._relapse_share_two(w1, l2, a1, a2)
    return float(np.sum(mass * (h1 * s1 + (1 - h1) * h2 * s2)))


for order in ("death_first", "readmission_first"):
    nv, al = functional(0, 0, order), functional(1, 1, order)
    print(order, "never", round(nv, 4), "always", round(al, 4), "diff", round(al - nv, 4))
print("total never", round(total(0, 0), 4), "always", round(total(1, 1), 4), "diff", round(total(1, 1) - total(0, 0), 4))
