"""Truth and nu^2 for the strong-positivity law; smoke-test one fit per learner."""
import sys
import os

import numpy as np
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run2  # noqa: E402

frame, truth = run2.strong_navigation_data(3000, 21)
print(list(frame.columns))
print({k: round(v, 6) for k, v in truth.items() if not k.startswith("sample_")})
from cleverly.datasets import navigation_data  # noqa: E402

_, old = navigation_data(n=3000, seed=21)
print("old truth ate", old["ate"], "new", truth["ate"], "diff", truth["ate"] - old["ate"])

rng = np.random.default_rng(777)
acc = []
gmin = 1.0
for _ in range(10):
    w = rng.normal(size=(1_000_000, 4))
    g = run2.g_nav(w)
    gmin = min(gmin, g.min())
    acc.append((1 / g + 1 / (1 - g)).mean())
print(f"true nu^2 = E[1/g + 1/(1-g)] = {np.mean(acc):.4f} (MC SE {np.std(acc, ddof=1) / np.sqrt(10):.4f}); min g in 1e7 draws {gmin:.4f}")
print("P(A=1)", frame["transition_navigation"].mean())
if "--smoke" in sys.argv:
    for l in ["a_hgb_default", "b_hgb_regularised", "c_hgb_early_stop", "e2_logit_poly2"]:
        print(run2.one((l, 3000, 21)))
