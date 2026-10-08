"""Three checks behind the third declaration of the ``longitudinal-mtp`` overfitting pair.

``checks.jsonl`` holds one line per result.

=============  ===============================================================================
check          question
=============  ===============================================================================
hazard         Can the run-2 pair learn the dose hazard with its own fully grown tree?  One
               draw per leaf size; the outcome learners are run 2's single trees
folds          Does ten-fold cross-fitting move the redesigned cross-fitted arm's SE ratio?
max_features   Do fully random splits (``max_features=1``) move either arm?
=============  ===============================================================================

``folds`` and ``max_features`` read the 400 draws of ``run.py`` and ran on 2026-10-07 at the
declaration's code; their lines in ``checks.jsonl`` are that run's output.  ``hazard`` ran
again at the record commit and reproduces its first run.

    python -m tests.diagnostics.longitudinal_mtp_overfit_design.checks hazard
    python -m tests.diagnostics.longitudinal_mtp_overfit_design.checks folds --jobs 4
    python -m tests.diagnostics.longitudinal_mtp_overfit_design.checks max_features --jobs 4
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.utils.parallel import map_parallel
from tests.diagnostics.longitudinal_mtp_overfit_design import run as probe
from tests.studies import longitudinal_mtp_common as common
from tests.studies import longitudinal_mtp_properties as properties

#: The two draws the hazard check read, outside every declared stream.
HAZARD_SEEDS = (917_000_000, 917_000_001)
HAZARD_LEAVES = (1, 5, 20, 50, 200)


class TreeDoseHazard(DecisionTreeClassifier):
    """A tree on the pooled dose hazard that reads the bin index as one ordered column."""

    bin_design = "index"


def _hazard_row(seed: int, leaf: int) -> dict[str, Any]:
    original = common.continuous_learners

    def run_two(configuration: str, edges: Any, *, extra: int = 0) -> tuple[Any, Any, Any]:
        del configuration, edges, extra
        return (
            DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
            DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            TreeDoseHazard(min_samples_leaf=leaf, random_state=0),
        )

    common.continuous_learners = run_two  # type: ignore[assignment]
    try:
        frame = common.sample_continuous(1_000, seed, noise=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crossed = properties.continuous_fit(frame, "overfit_crossfit", n_folds=5)[properties.UP]
            control = properties.continuous_fit(frame, "overfit_control", n_folds=1)[properties.UP]
    finally:
        common.continuous_learners = original
    return {
        "check": "hazard",
        "seed": seed,
        "leaf": leaf,
        "cross_fitted_std_error": float(crossed.std_error),
        "in_sample_std_error": float(control.std_error),
    }


def _arm(replicate: int, folds: int, configuration: str, max_features: Any) -> tuple[float, float]:
    if max_features is not None:
        common.OVERFIT_ENSEMBLE["max_features"] = max_features
    frame = common.sample_continuous(properties.OVERFIT_N, probe._seed(replicate), noise=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate = properties.continuous_fit(frame, configuration, n_folds=folds)[properties.UP]
    return float(estimate.psi), float(estimate.std_error)


def _ratio(rows: list[tuple[float, float]]) -> float:
    values = np.asarray(rows)
    return float(values[:, 1].mean() / values[:, 0].std(ddof=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=("hazard", "folds", "max_features"))
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)
    replicates = list(range(probe.REPLICATES))
    if arguments.check == "hazard":
        lines = [_hazard_row(seed, leaf) for seed in HAZARD_SEEDS for leaf in HAZARD_LEAVES]
    elif arguments.check == "folds":
        rows = map_parallel(
            _arm, [(k, 10, "overfit_crossfit", None) for k in replicates], n_jobs=arguments.jobs
        )
        lines = [{"check": "folds", "folds": 10, "cross_fitted_se_ratio": _ratio(rows)}]
    else:
        crossed = map_parallel(
            _arm, [(k, 5, "overfit_crossfit", 1) for k in replicates], n_jobs=arguments.jobs
        )
        control = map_parallel(
            _arm, [(k, 1, "overfit_control", 1) for k in replicates], n_jobs=arguments.jobs
        )
        lines = [
            {
                "check": "max_features",
                "max_features": 1,
                "cross_fitted_se_ratio": _ratio(crossed),
                "in_sample_se_ratio": _ratio(control),
            }
        ]
    for line in lines:
        print(json.dumps(line).replace("NaN", "null"), flush=True)


if __name__ == "__main__":
    main()
