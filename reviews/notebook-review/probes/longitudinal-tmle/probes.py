"""Small probes of individual notebook claims (seed 41 frame, plus oracle quantities)."""

import warnings

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

import cleverly
from cleverly import (
    CausalStudy, CrossFitting, LongitudinalTreatment, ModelSpec, RegimeContrast, Runtime, TMLEMethod,
)
from cleverly.datasets import make_longitudinal

warnings.filterwarnings("ignore")
assert "bridge-cse" in cleverly.__file__

frame, _ = make_longitudinal(n=8_000, seed=41, cluster_size=20)

# 1. In-sample calibration slope of a deliberately WRONG logistic outcome model.
obs = frame.dropna(subset=["Y"])
y = obs["Y"].to_numpy()
for label, cols, C in (
    ("wrong form (W only, no A, no L2), default C=1", ["W1", "W2"], 1.0),
    ("wrong form, unpenalized", ["W1", "W2"], np.inf),
    ("notebook form (W, A1, A2, L2), default C=1", ["W1", "W2", "A1", "A2", "L2"], 1.0),
):
    X = obs[cols].to_numpy()
    m = LogisticRegression(max_iter=1000, C=C if np.isfinite(C) else 1e12).fit(X, y)
    lp = m.decision_function(X).reshape(-1, 1)
    slope = LogisticRegression(max_iter=1000, C=1e12).fit(lp, y).coef_[0, 0]
    print(f"in-sample calibration slope, {label}: {slope:.4f}  auc {roc_auc_score(y, lp.ravel()):.3f}")

# 2. Oracle AUC of the day-30 tracking model, on rows at risk (tracked at day 7)
rng = np.random.default_rng(7)
n = 2_000_000
w1 = rng.standard_normal(n); w2 = rng.standard_normal(n)
a1 = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * w1 - 0.4 * w2))))
c1 = rng.binomial(1, 1 / (1 + np.exp(-(2.2 + 0.3 * w1 - 0.3 * a1))))
l2 = 0.6 * w1 + 0.9 * a1 + rng.standard_normal(n)
p2 = 1 / (1 + np.exp(-(2.4 + 0.2 * l2)))
c2 = rng.binomial(1, p2)
risk = c1 == 1
print(f"oracle AUC of true P(tracked_day30 | history) among tracked at day 7: {roc_auc_score(c2[risk], p2[risk]):.4f}")
pc1 = 1 / (1 + np.exp(-(2.2 + 0.3 * w1 - 0.3 * a1)))
print(f"oracle AUC of true P(tracked_day7 | history): {roc_auc_score(c1, pc1):.4f}")

# 3. Does engagement drive day-seven navigation WITHIN discharge-navigation strata?
tracked = frame[frame["C1"] == 1]
print("share A2=1 by (A1, L2>0) on seed 41:")
print(tracked.groupby(["A1", tracked["L2"] > 0])["A2"].mean().round(3).to_string())
print("share of A1=1 among L2>0 vs L2<=0:",
      tracked.groupby(tracked["L2"] > 0)["A1"].mean().round(3).to_dict())

# 4. Cross-fitting with a cluster: refused?  And without a cluster: which score kinds?
logistic = LogisticRegression(max_iter=1000, random_state=41)
models = ModelSpec(outcome_learner=logistic, pseudo_learner=LinearRegression(),
                   treatment_learner=logistic, censoring_learner=logistic)


def design(cluster):
    return LongitudinalTreatment(
        outcome="Y", treatment=("A1", "A2"), baseline=("W1", "W2"),
        time_varying=((), ("L2",)), censoring=("C1", "C2"),
        **({"cluster": "id"} if cluster else {}))


plan = RegimeContrast({"always": 1, "never": 0}, reference="never")
cf = TMLEMethod(models=models, cross_fitting=CrossFitting(enabled=True, n_folds=5),
                runtime=Runtime(random_state=41, n_jobs=1))
try:
    CausalStudy(frame, design=design(True)).identify(plan).estimate(method=cf)
    print("cross-fit + cluster: NOT refused")
except Exception as exc:  # noqa: BLE001
    print("cross-fit + cluster refused:", type(exc).__name__, str(exc)[:400])
r = CausalStudy(frame.drop(columns=["id"]), design=design(False)).identify(plan).estimate(method=cf)
sc = r.assess().report("score_equations").to_frame()
print("cross-fitted (no cluster) score kinds:", sorted(sc["kind"].unique()), "all passed:", bool(sc["passed"].all()))
print("cross-fitted psi", round(r["ate_regimen[always vs never]"].psi, 4))
