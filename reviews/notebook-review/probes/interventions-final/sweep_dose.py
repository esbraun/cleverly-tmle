"""Seed sweep of the modified-treatment-policy fit of docs/examples/interventions.ipynb.

Each seed ``s`` draws ``make_shift_dose(n=3000, seed=s, shifts=...)`` with the page's four
policies, and runs the Step 9 fit and the Step 10 cap gap for each configuration below, every
``random_state`` set to ``s``.  All fits are in sample (``CrossFitting(enabled=False)``), as on
the page.

| configuration | outcome learner | density learner | bins |
| --- | --- | --- | --- |
| ``boosted40`` | ``HistGradientBoostingRegressor`` | ``HistGradientBoostingClassifier`` | 40 |
| ``quad40`` | degree-2 polynomial least squares | ``HistGradientBoostingClassifier`` | 40 |
| ``quad80`` | degree-2 polynomial least squares | ``HistGradientBoostingClassifier`` | 80 |
| ``quad80logit`` | degree-2 polynomial least squares | ``LogisticRegression`` (pooled hazard) | 80 |

``boosted40`` is the configuration the page showed before this review.  The degree-2 outcome
model holds ``[a, a^2, W]`` and so represents the law's outcome mean exactly (``shift_dgp``).

Usage: ``python sweep_dose.py <first seed> <count> <workers> <configuration> [...]``.  The page
cites ``python sweep_dose.py 9000 120 12 boosted40 quad40 quad80 quad80logit``.  Outputs:
``sweep_dose.csv`` and ``sweep_dose.log``, which ``summarize.py`` reads.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHIFTS = (
    (0.0, None, "current practice"),
    (0.5, 5.0, "+0.5 capped at 5"),
    (0.5, None, "+0.5 uncapped"),
    (1.0, None, "+1.0 uncapped"),
)
CAPPED = "ate_shift[+0.5 capped at 5 vs current practice]"
UNCAPPED = "ate_shift[+0.5 uncapped vs current practice]"
ONE = "ate_shift[+1.0 uncapped vs current practice]"


def learners(configuration: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    if configuration == "boosted40":
        return (
            HistGradientBoostingRegressor(random_state=seed),
            HistGradientBoostingClassifier(random_state=seed),
            40,
        )
    quadratic = make_pipeline(PolynomialFeatures(2), LinearRegression())
    if configuration == "quad40":
        return quadratic, HistGradientBoostingClassifier(random_state=seed), 40
    if configuration == "quad80":
        return quadratic, HistGradientBoostingClassifier(random_state=seed), 80
    if configuration == "quad80logit":
        return quadratic, LogisticRegression(C=1000.0, max_iter=5000), 80
    raise ValueError(configuration)


def one_seed(seed: int, configurations: tuple[str, ...]) -> list[dict]:
    from cleverly import (
        CausalStudy,
        CrossFitting,
        ModelSpec,
        ModifiedTreatmentPolicyEffect,
        PointTreatment,
        PositivityWarning,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_shift_dose
    from cleverly.interventions import Shift

    frame, truth = make_shift_dose(n=3_000, seed=seed, shifts=SHIFTS)
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), treatment_kind="continuous"
        ),
    ).identify(
        ModifiedTreatmentPolicyEffect(
            tuple(Shift(delta, cap=cap, name=name) for delta, cap, name in SHIFTS)
        )
    )
    rows = []
    for configuration in configurations:
        start = time.perf_counter()
        outcome, density, bins = learners(configuration, seed)
        method = TMLEMethod(
            models=ModelSpec(outcome_learner=outcome, treatment_learner=density, density_bins=bins),
            cross_fitting=CrossFitting(enabled=False),
            runtime=Runtime(random_state=seed, n_jobs=1),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = effect.estimate(method=method)
        row: dict = {"seed": seed, "configuration": configuration}
        row["positivity_warnings"] = sum(w.category is PositivityWarning for w in caught)
        row["other_warnings"] = "|".join(
            sorted({w.category.__name__ for w in caught if w.category is not PositivityWarning})
        )
        for label, name in (("capped", CAPPED), ("uncapped", UNCAPPED), ("one", ONE)):
            point = result[name]
            row |= {
                f"{label}_psi": point.psi,
                f"{label}_se": point.std_error,
                f"{label}_lo": point.ci[0],
                f"{label}_hi": point.ci[1],
                f"{label}_truth": truth[name],
            }
        gap = result.contrast(lambda p: p[0] - p[1], [UNCAPPED, CAPPED], name="gap")
        row |= {
            "gap_psi": gap.psi,
            "gap_se": gap.std_error,
            "gap_lo": gap.ci[0],
            "gap_hi": gap.ci[1],
            "gap_truth": truth[UNCAPPED] - truth[CAPPED],
        }
        assessment = result.assess()
        row["attention"] = "|".join(item.name for item in assessment.attention)
        support = assessment.report("support")
        for label, name in (
            ("capped", "+0.5 capped at 5"),
            ("uncapped", "+0.5 uncapped"),
            ("one", "+1.0 uncapped"),
        ):
            row[f"{label}_ess"] = support[name].ess_ratio
            row[f"{label}_max_ratio"] = support[name].max_ratio
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
    return rows


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    configurations = tuple(sys.argv[4:])
    import cleverly

    assert Path(cleverly.__file__).resolve().is_relative_to(HERE.parents[3] / "src")
    seeds = range(first, first + count)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [
            row
            for chunk in pool.map(partial(one_seed, configurations=configurations), seeds)
            for row in chunk
        ]
    frame = pd.DataFrame(rows).sort_values(["configuration", "seed"])
    frame.to_csv(HERE / "sweep_dose.csv", index=False, float_format="%.10g")
    print(f"{count} seeds, {first} to {first + count - 1}; cleverly {cleverly.__file__}")
    for configuration, group in frame.groupby("configuration"):
        print(f"{configuration}: median seconds per fit {np.median(group['seconds']):.1f}")


if __name__ == "__main__":
    main()
