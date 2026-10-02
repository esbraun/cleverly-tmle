"""Re-measure the held-out-scale and scale-only probes of cv-tmle.md on either law.

Usage: ``python probe.py {old,new} <fold seed>``.  ``old`` restores the unbounded
propensity ``expit(nonlinear_logit(W))``; ``new`` is the shipped ``[0.05, 0.95]`` law.

The original probes ran at commit ``1cf6628`` and no script was kept.  This reconstruction,
at fold seed 0 on the old law, reproduces the scale (-3.37, 10.39) -> (-9.11, 73.49) and the
fold-0 prediction 0.450 -> 0.145 exactly.  Its ATE moves (-1.4e-4 and +3.0e-4) differ from the
recorded -1.7e-4 and +3.2e-4 because the targeting code changed after ``1cf6628``.
Outputs: ``old-law-seed0.txt`` and ``new-law-seed0.txt``.
"""

import dataclasses
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

import cleverly.datasets.synthetic as synthetic
import cleverly.learners.crossfit as crossfit
from cleverly.estimators import TMLE
from cleverly.utils.bounds import OutcomeScaler, expit

LAW = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if LAW == "old":
    shipped = synthetic.nonlinear_dgp

    def raw() -> synthetic.DGP:
        return dataclasses.replace(
            shipped(), propensity=lambda w: expit(synthetic.nonlinear_logit(w))
        )

    synthetic.nonlinear_dgp = raw

frame, _ = synthetic.make_nonlinear_ate(n=400, seed=11)
y = frame["Y"].to_numpy()
a = frame["A"].to_numpy()
x = frame[["A", "W1", "W2", "W3", "W4"]].to_numpy()

base = OutcomeScaler.from_outcome(y)
raised = y.copy()
row = int(np.argmax(y))
raised[row] = y[row] + 5.0 * (y.max() - y.min())
moved = OutcomeScaler.from_outcome(raised)
print(f"scale ({base.lower:.2f}, {base.upper:.2f}) -> ({moved.lower:.2f}, {moved.upper:.2f})")

folds = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(x, a))
for k, (train, valid) in enumerate(folds):
    train = train[train != row]
    fit_base = LinearRegression().fit(x[train], base.scale(y[train]))
    fit_moved = LinearRegression().fit(x[train], moved.scale(y[train]))
    print(
        k,
        "raised row held out" if row in valid else "",
        f"{fit_base.predict(x[valid]).mean():.3f} -> {fit_moved.predict(x[valid]).mean():.3f}",
    )

# The scale-only probe needs the policies the package now refuses: treatment-stratified folds
# and a scale read from every row.  Both reservations are bypassed here, as the original
# probe did for strata.
crossfit.fold_strata_refusal = lambda *args, **kwargs: None
TMLE._outcome_scale_refusal = lambda self, data: None

for name, learner in (
    ("linear", LinearRegression()),
    ("forest", RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=0)),
):
    common = dict(
        outcome_learner=learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=10,
        stratify_folds="treatment",
        random_state=SEED,
        estimands=("ate",),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = TMLE(**common).fit(frame, outcome="Y", treatment="A").single()
        fixed = (
            TMLE(**common, q_bounds=(moved.lower, moved.upper))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
    d = fixed.estimates["ate"].psi - free.estimates["ate"].psi
    se = free.estimates["ate"].std_error
    print(name, f"ATE moves by {d:+.3e} ({abs(d) / se:.4f} SE)")
