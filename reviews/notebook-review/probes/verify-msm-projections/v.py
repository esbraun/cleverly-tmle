"""Independent verifier sweep for msm-projections (MS-01, MS-02)."""
import os
import sys
import warnings

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import CausalStudy, CounterfactualMean, CrossFitting, ModelSpec, MSMProjection, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_multi_arm
from cleverly.msm import MSM

warnings.filterwarnings("ignore")
here = os.path.dirname(os.path.abspath(__file__))
wt = os.path.abspath(os.path.join(here, "..", "..", ".."))
assert os.path.abspath(cleverly.__file__).startswith(os.path.join(wt, "src")), cleverly.__file__

ARMS = ("low", "medium", "high")
C = {"low": 1.0, "medium": 2.0, "high": 6.0}
X3 = np.column_stack([np.ones(3), [1.0, 2.0, 6.0]])
pop = np.array([0.0, 0.6, 1.44])  # 0.6 * step, covariate terms mean zero
proj = np.linalg.lstsq(X3, pop, rcond=None)[0]
hat = X3 @ np.linalg.pinv(X3)
mw = hat[1] - np.eye(3)[1]


class OracleMulti(BaseEstimator):
    """True softmax arm probabilities from the generator's logits."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        w = np.asarray(X, dtype=float)
        lg = np.column_stack([np.zeros(len(w)), 0.8 * w[:, 0] - 0.4 * w[:, 1], -0.5 * w[:, 0] + 0.8 * w[:, 1]])
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        order = {lab: i for i, lab in enumerate(ARMS)}
        cols = []
        for c in self.classes_:
            key = c if isinstance(c, str) else sorted(ARMS)[int(c)]
            cols.append(order[key])
        return p[:, cols]


def design(a, data):
    return np.column_stack([np.ones(len(data)), np.full(len(data), C[a])])


def run(seed, n, oracle):
    frame, _ = make_multi_arm(n=n, seed=seed)
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")))
    g = OracleMulti() if oracle else LogisticRegression(max_iter=1000, random_state=61)
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=LinearRegression(), treatment_learner=g),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=61, n_jobs=1),
    )
    tr = study.identify(MSMProjection(MSM(design=design, terms=("i", "s"), design_kind="known"))).estimate(method=method)
    out = dict(seed=seed, n=n, oracle=oracle, slope=tr["msm[s]"].psi, se=tr["msm[s]"].std_error)
    out["cov"] = tr["msm[s]"].ci[0] <= proj[1] <= tr["msm[s]"].ci[1]
    if not oracle:
        arm = study.identify(CounterfactualMean()).estimate(method=method)
        means = np.array([arm[f"ey[{a}]"].psi for a in ARMS])
        mis = arm.contrast(lambda m: float(mw @ m), [f"ey[{a}]" for a in ARMS], name="m", gradient=lambda m: mw)
        line = X3 @ np.array([tr["msm[i]"].psi, tr["msm[s]"].psi])
        out["mis"] = mis.psi
        out["step7"] = line[1] - means[1]
        gp = np.asarray(tr.nuisance.propensity.values)
        out["gmin"] = float(gp.min())
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check61":
        print(run(61, 3000, False))
        print(run(61, 3000, True))
        sys.exit()
    n = int(sys.argv[2]); s0 = int(sys.argv[3]); s1 = int(sys.argv[4])
    oracle = mode == "oracle"
    rows = [run(s, n, oracle) for s in range(s0, s1)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(here, f"{mode}_{n}_{s0}_{s1}.csv"), index=False)
    k = len(df)
    print(mode, "n", n, "seeds", k)
    print("slope coverage", df["cov"].mean(), "MC SE", np.sqrt(0.95 * 0.05 / k))
    print("mean SE", df["se"].mean(), "emp SD", df["slope"].std(), "ratio", df["se"].mean() / df["slope"].std())
    print("bias", df["slope"].mean() - proj[1])
    if not oracle:
        d = (df["mis"] - df["step7"]).abs()
        print("|mis-step7| <5e-4", (d < 5e-4).mean(), "<1e-3", (d < 1e-3).mean(), "median", d.median(), "max", d.max())
        lo = df["gmin"] <= df["gmin"].median()
        print("coverage gmin<=median", df.loc[lo, "cov"].mean(), "gmin>median", df.loc[~lo, "cov"].mean())
