"""Seed sweep of the regime and incremental fits of docs/examples/interventions.ipynb.

Each seed ``s`` draws ``navigation_data(n=3000, seed=s)`` and runs the notebook's Step 5 regime
fit and Step 13 incremental fit with each of two treatment learners, every ``random_state`` set
to ``s``:

- ``booster``: ``HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200,
  l2_regularization=1.0)``, the shallow, regularized booster of the point-treatment page.
- ``poly2``: ``make_pipeline(StandardScaler(), PolynomialFeatures(2),
  LogisticRegression(C=1000.0, max_iter=5000))``, a degree-2 logistic model. The law's logit
  has a step in W4 and the propensity is squeezed into [0.05, 0.95], so this model is flexible
  parametric, not correctly specified.

The outcome learner, the three folds, ``q_bounds=(0.0, 1.0)``, and ``n_jobs=1`` are the
notebook's.  The truths come from the package's quadrature (``nonlinear_bounded_dgp``); the
independent recomputation is ``truth.py``.

Usage: ``python sweep_binary.py <first seed> <count> <workers>``.  The page cites
``python sweep_binary.py 9000 120 12``.  Outputs: ``sweep_binary.csv`` and ``sweep_binary.log``,
which ``summarize.py`` reads.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")
OFFER_ALL = "ate_regime[offer to all vs offer to none]"
SCREEN = "ate_regime[screen on risk vs offer to none]"
IPSI = "ate_ipsi[double odds vs current odds]"


def treatment_learner(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if kind == "booster":
        return HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=seed,
        )
    return make_pipeline(
        StandardScaler(), PolynomialFeatures(2), LogisticRegression(C=1000.0, max_iter=5000)
    )


def truths() -> dict[str, float]:
    from cleverly.datasets import nonlinear_bounded_dgp

    law = nonlinear_bounded_dgp()

    def effect_of_offer(latent):
        return law.outcome_mean(latent, 1.0, None) - law.outcome_mean(latent, 0.0, None)

    screen = law.expectation(lambda latent: effect_of_offer(latent) * (latent[:, 0] > 0))
    ipsi = law.incremental_truth((1.0, 2.0))["ate_ipsi[odds x2 vs natural course]"]
    return {"screen": float(screen), "ipsi": float(ipsi)}


def one_seed(seed: int) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    from cleverly import (
        CausalStudy,
        CrossFitting,
        IncrementalEffect,
        ModelSpec,
        PointTreatment,
        RegimeContrast,
        Runtime,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import navigation_data
    from cleverly.interventions import Incremental, Rule, Static

    frame, truth = navigation_data(n=3_000, seed=seed)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=COVARIATES,
        ),
    )
    plans = (
        Static(0, name="offer to none"),
        Static(1, name="offer to all"),
        Rule(
            lambda data: (data["discharge_risk"] > 0).astype(float),
            name="screen on risk",
            rule_kind="known",
        ),
    )
    regimes = study.identify(RegimeContrast(plans, reference="offer to none"))
    incremental = study.identify(
        IncrementalEffect(
            (Incremental(1.0, name="current odds"), Incremental(2.0, name="double odds"))
        )
    )
    rows = []
    for kind in ("booster", "poly2"):
        start = time.perf_counter()
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=HistGradientBoostingRegressor(random_state=seed),
                treatment_learner=treatment_learner(kind, seed),
            ),
            cross_fitting=CrossFitting(n_folds=3),
            targeting=Targeting(q_bounds=(0.0, 1.0)),
            runtime=Runtime(random_state=seed, n_jobs=1),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regime_result = regimes.estimate(method=method)
            incremental_result = incremental.estimate(method=method)
        row: dict = {"seed": seed, "learner": kind, "ate_truth": truth["ate"]}
        row["n_warnings"] = len(caught)
        row["warning_kinds"] = "|".join(sorted({w.category.__name__ for w in caught}))
        for label, fitted, name in (
            ("all", regime_result, OFFER_ALL),
            ("screen", regime_result, SCREEN),
            ("ipsi", incremental_result, IPSI),
        ):
            point = fitted[name]
            row |= {
                f"{label}_psi": point.psi,
                f"{label}_se": point.std_error,
                f"{label}_lo": point.ci[0],
                f"{label}_hi": point.ci[1],
            }
        regime_assessment = regime_result.assess()
        incremental_assessment = incremental_result.assess()
        row["regime_attention"] = "|".join(i.name for i in regime_assessment.attention)
        row["ipsi_attention"] = "|".join(i.name for i in incremental_assessment.attention)
        nuisance = regime_assessment.report("nuisance_models")
        propensity = next(m for m in nuisance.models if m.name == "propensity")
        row["cal_slope"] = propensity.metrics.get("calibration_slope", np.nan)
        row["auc"] = propensity.metrics.get("auc", np.nan)
        support = regime_assessment.report("support").regimes
        for plan, key in (
            ("offer to none", "none"),
            ("offer to all", "all"),
            ("screen on risk", "screen"),
        ):
            row[f"{key}_min_g"] = support[plan].min_support_propensity
            row[f"{key}_max_ratio"] = support[plan].max_ratio
            row[f"{key}_ess"] = support[plan].effective_sample_size
            row[f"{key}_load"] = support[plan].score_load["effective"]
        tilts = incremental_assessment.report("support")
        row["ipsi_ess"] = tilts["double odds"].ess_ratio
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
    return rows


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    import cleverly

    assert Path(cleverly.__file__).resolve().is_relative_to(HERE.parents[3] / "src")
    seeds = range(first, first + count)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for chunk in pool.map(one_seed, seeds) for row in chunk]
    frame = pd.DataFrame(rows).sort_values(["learner", "seed"])
    frame.to_csv(HERE / "sweep_binary.csv", index=False, float_format="%.10g")
    t = truths()
    print(f"{count} seeds, {first} to {first + count - 1}; cleverly {cleverly.__file__}")
    print(f"truths: screen {t['screen']:.7f}, ipsi {t['ipsi']:.7f}")
    print(f"median seconds per seed and learner: {np.median(frame['seconds']):.1f}")


if __name__ == "__main__":
    main()
