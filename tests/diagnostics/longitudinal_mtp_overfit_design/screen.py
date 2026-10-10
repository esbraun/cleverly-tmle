"""The design screen behind the third declaration of the ``longitudinal-mtp`` overfitting pair.

Each design fits both arms on fresh draws.  ``screen.jsonl`` holds one line per run, in run
order.  The screen chose the design; ``run.py`` is the committed probe of the chosen one.

The screen draws its noise columns from its own generator, not from
``sample_continuous(noise=True)``, so its draws are not the study's; ``run.py`` reads the
study's draw path.  ``tree_u1_step1`` moves the policy step from 0.5 to 1.0, which changes the
estimand, so the screen rejected it whatever it read; its truth comes from a 768-point
quadrature.  A step of 0.75 failed the quadrature's stability check and has no line.

    python tests/diagnostics/longitudinal_mtp_overfit_design/screen.py <design> <draws> <jobs>
"""

import json
import sys
import time
import warnings
from pathlib import Path

WT = str(Path(__file__).resolve().parents[3])
sys.path[:0] = [WT + "/src", WT]

import numpy as np  # noqa: E402

DESIGNS = {
    # name: (learner kind, noise columns, n)
    "tree_u1": ("tree", 1, 1000),
    "tree_u5": ("tree", 5, 1000),
    "tree_u10": ("tree", 10, 1000),
    "nn_u1": ("nn", 1, 1000),
    "nn_u3": ("nn", 3, 1000),
    "nn_u5": ("nn", 5, 1000),
    "tree_u1_n500": ("tree", 1, 500),
    "tree_u1_n250": ("tree", 1, 250),
    "nnraw_u1": ("nnraw", 1, 1000),
    "nnraw_u0": ("nnraw", 0, 1000),
    "tree_u1_step1": ("tree", 1, 1000),
    "et50_u1": ("et50", 1, 1000),
}
STEPS = {"tree_u1_step1": 1.0}


def plans(design):
    from cleverly.interventions import Shift
    from tests.studies import longitudinal_mtp_common as common
    from tests.studies import longitudinal_mtp_properties as props

    if design not in STEPS:
        return props._up_plans(), float(props.TRUTH[props.CONTINUOUS][props.UP]), props.UP
    step = STEPS[design]
    label = f"up{step}"

    def move(a, step=step):
        a = np.asarray(a, dtype=float)
        return np.where(a + step > common.CAP, a, a + step)

    common.CONTINUOUS_MAPS[label] = (move, lambda a2, a1: move(a2))
    truth = common._continuous_mean(label, 768) - common.continuous_truth("natural")
    return {"natural": common.NATURAL, "up": Shift(step, cap=common.CAP)}, truth, props.UP


BASE = 515_000_000


def learners(kind):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if kind == "tree":
        return (
            DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
            DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
        )
    if kind == "et50":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        settings = {
            "n_estimators": 50,
            "min_samples_leaf": 1,
            "bootstrap": False,
            "random_state": 0,
        }
        return ExtraTreesClassifier(**settings), ExtraTreesRegressor(**settings)
    if kind == "nnraw":
        return KNeighborsClassifier(n_neighbors=1), KNeighborsRegressor(n_neighbors=1)
    return (
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1)),
    )


def sample(n, seed, k):
    from tests.studies import longitudinal_mtp_common as common

    frame = common.sample_continuous(n, seed)
    rng = np.random.default_rng([seed, 7])
    for j in range(k):
        frame.insert(1 + j, f"U{j}", rng.normal(size=n))
    return frame


def fit(frame, kind, k, folds, design="tree_u1"):
    from sklearn.base import clone

    from cleverly.longitudinal import LTMLE
    from tests.studies import canonical_longitudinal_mtp as study
    from tests.studies import longitudinal_mtp_common as common
    from tests.studies import longitudinal_mtp_properties as props
    from tests.studies.oracle_density_bins import oracle_bins

    outcome, pseudo = learners(kind)
    hazard = common.OracleDoseHazard(
        *(tuple(float(v) for v in e) for e in study.edges_of(frame)), extra=k
    )
    noise = [f"U{j}" for j in range(k)]
    return LTMLE(
        plans(design)[0],
        reference=study.REFERENCE,
        outcome_learner=clone(outcome),
        pseudo_learner=clone(pseudo),
        treatment_learner=hazard,
        n_folds=folds,
        learner_folds=study.LEARNER_FOLDS,
        g_bounds=study.G_BOUNDS,
        density_bins=oracle_bins(len(frame)),
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W", *noise],
        time_varying=[[], ["L2"]],
        continuous_treatment=["A1", "A2"],
    )[props.UP]


def one(args):
    design, d = args
    warnings.simplefilter("ignore")
    kind, k, n = DESIGNS[design]
    frame = sample(n, BASE + d, k)
    a = fit(frame, kind, k, 5, design)
    b = fit(frame, kind, k, 1, design)
    return float(a.psi), float(a.std_error), float(b.psi), float(b.std_error)


if __name__ == "__main__":
    from joblib import Parallel, delayed

    design, draws, jobs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    truth = plans(design)[1]
    start = time.time()
    rows = np.array(Parallel(n_jobs=jobs)(delayed(one)((design, d)) for d in range(draws)))
    z = 1.959963984540054
    out = {"design": design, "draws": draws, "seconds": round(time.time() - start, 1)}
    covers = {}
    for name, (pi, si) in (("crossfit", (0, 1)), ("control", (2, 3))):
        psi, se = rows[:, pi], rows[:, si]
        covers[name] = np.abs(psi - truth) <= z * se
        out[name] = {
            "bias": float(psi.mean() - truth),
            "se_ratio": float(np.nanmean(se) / np.std(psi, ddof=1)),
            "coverage": float(covers[name].mean()),
            "finite_se": float(np.isfinite(se).mean()),
        }
    gain = covers["crossfit"].astype(float) - covers["control"].astype(float)
    out["gain"] = float(gain.mean())
    out["gain_se"] = float(gain.std(ddof=1) / np.sqrt(draws))
    print(json.dumps(out))
