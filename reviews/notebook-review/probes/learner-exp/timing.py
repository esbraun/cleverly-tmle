"""Idle single-process timing per fit (n=3000, point config), 3 seeds each."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run  # noqa: E402
import run2  # noqa: E402

for learner in ["a_hgb_default", "b_hgb_regularised", "c_hgb_early_stop", "d_logit_main", "e_logit_true"]:
    t = [run.one(("point", learner, 3000, s))["fit_s"] for s in (5000, 5001, 5002)]
    print(f"{learner:20s} navigation_data fit s: " + " ".join(f"{x:.2f}" for x in t))
t = [run2.one(("e2_logit_poly2", 3000, s))["fit_s"] for s in (5000, 5001, 5002)]
print(f"{'e2_logit_poly2':20s} strong law fit s:      " + " ".join(f"{x:.2f}" for x in t))
