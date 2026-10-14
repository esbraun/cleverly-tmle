"""Registered evidence for C-TMLE0, ``ctmle-derivative``.

The subject is ``CTMLE(strategy="derivative")``: one TMLE update whose submodel adds, for each
arm, the derivative of the clever covariate along the lasso penalty.  The update keeps the
clever covariates in its span, so it solves the efficient influence curve equation and takes
the ordinary TMLE status.  The study measures that interval.

**The laws** are those of :mod:`tests.studies.ctmle_derivative_laws`: a main-terms logistic
treatment law, so the internal lasso is consistent for it, and a GLM-correct binary outcome.

**The comparator** is R ``ctmle`` 0.1.2 at ``18de559``, ``ctmletype = 3``, through
``tests/canonical/ctmle_derivative/run_ctmle_derivative.R`` in the image that
``tests/canonical/ctmle_selector`` builds.  Both scenarios read the primary law at
``n = 2,000``.

=========================  ==================================================================
scenario                   both sides
=========================  ==================================================================
``binary_declared_pair``   the declared pair of ``penalties.csv``.  The package passes
                           ``lambdas=PAIR``; R fits ``glmnet(lambda = PAIR, thresh = 1e-14)``
                           and hands both predictions to ``ctmleGlmnet``.  Both sides fit the
                           same estimator, so the paired rows are parity at scale
``binary_pipeline``        each side's own pipeline.  The package passes ``lambdas=None`` with
                           ``learner_folds=10``; R runs ``cv.glmnet(nfolds = 10)``, takes
                           ``lambda.min`` and the partner at ``1.05 lambda.min``, as the
                           ``ctmle`` README does
=========================  ==================================================================

The pipelines differ in three declared ways: the cross-validation folds; the partner (R
interpolates a larger penalty, and the package takes the nearest grid neighbour, which is the
next smaller penalty); and the grid (the package appends the penalty 0, and R does not).

**The declared pair** is fixed before any run by :func:`penalty_design`.  It fits the package
pipeline on 50 design draws of the primary law and takes the lower median ``lambda_cv``.  The
partner is that draw's own grid neighbour.  ``penalty-design.csv`` keeps every draw, and
``tests/unit/test_ctmle_derivative_study_design.py`` refits the median draw.

**What the Python rows also carry.**  Each primary row carries the ordinary ``TMLE`` on the same
initial fit, mechanism and bounds, with only the derivative columns removed
(:func:`paired_tmle`), and the penalty pair the fit used.  ``paired-tmle.csv`` and
``chosen-penalties.csv`` publish them.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import ctmle_derivative_laws as laws
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

CTMLE_COMMIT = "18de559f47dc1286617350a0668391e80e1dbf7c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20263001
RESAMPLING_SEED = 2026300101
IMPLEMENTATION = "cleverly-ctmle-derivative"
REFERENCE = "r-ctmle"
STRATIFY_FOLDS = "none"
#: The targeting shrink of both sides: R ``ctmle``'s ``alpha`` default.
ALPHA = 0.995
#: The folds of the internal penalty cross-validation, and of ``cv.glmnet``.
LEARNER_FOLDS = 10

ARTIFACTS = ROOT / "tests" / "canonical" / "ctmle_derivative"
#: The declared pair, read by both sides.
PENALTIES = ARTIFACTS / "penalties.csv"
#: The design draws the pair was chosen from.
PENALTY_DESIGN = ARTIFACTS / "penalty-design.csv"
#: The design draws, their size and their seeds.
PENALTY_DRAWS = 50
PENALTY_SEED = 20263100

SCENARIO_ESTIMANDS: Mapping[str, tuple[str, ...]] = {
    "binary_declared_pair": ("ate",),
    "binary_pipeline": ("ate",),
}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        "in_sample",
        "cross_fitted",
        "multi_arm",
        "missing_outcome",
        "clustered",
    ),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "double_robustness": ("treatment_correct", "outcome_correct", "both_wrong"),
    "targeting_necessity": ("ate__targeted", "ate__untargeted"),
}

#: The owner of every red cell, by its ``id`` in the roadmap's "Red-cell owners" table.  The
#: declared reading of :mod:`tests.studies.ctmle_derivative_properties` assigns each red cell.
RED_CELL_OWNER = "X30-derivative-reading"

STUDY = StudyRecord(
    name="C-TMLE0 derivative covariate",
    slug="ctmle-derivative",
    artifacts=ARTIFACTS,
    document="docs/technical-reference/method-evidence/ctmle-derivative.md",
    anchor="c-tmle0-derivative-covariate",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    extra_artifacts=("paired-tmle.csv", "chosen-penalties.csv"),
    modules=(
        "tests/studies/canonical_ctmle_derivative.py",
        "tests/studies/ctmle_derivative_properties.py",
        "tests/studies/ctmle_derivative_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/ctmle_selector/Dockerfile",
        "tests/canonical/ctmle_derivative/run_ctmle_derivative.R",
        "tests/canonical/ctmle_derivative/penalties.csv",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_ctmle_derivative",
    properties_module="tests.studies.ctmle_derivative_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)


def declared_pair() -> tuple[float, float]:
    """The ``(mechanism, partner)`` penalties of ``binary_declared_pair``."""
    frame = pd.read_csv(PENALTIES, float_precision="round_trip")
    values = dict(zip(frame["role"], frame["lambda"], strict=True))
    return float(values["mechanism"]), float(values["partner"])


REFERENCE_METADATA = {
    "ctmle_commit": CTMLE_COMMIT,
    "ctmle_version": "0.1.2",
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "Q: glm(Y ~ A + W1 + W2 + W3 + W4, binomial). binary_declared_pair: glmnet(W, "
        "factor(A), binomial, lambda = sort(PAIR, decreasing = TRUE), thresh = 1e-14, "
        "maxit = 1e7), g1W and g1WPrev by predict(type = 'response') at the mechanism and the "
        "partner penalty. binary_pipeline: cv.glmnet(W, factor(A), binomial, nfolds = 10, "
        "nlambda = 100), g1W at lambda.min and g1WPrev at 1.05 lambda.min. Both: ctmleGlmnet("
        "ctmletype = 3, gbound = 0.025, alpha = 0.995, family = 'binomial'). glmnet.control("
        "fdev = 0, devmax = 1)"
    ),
    "red_cell_owner": RED_CELL_OWNER,
}

CONFIGURATION = {
    "strategy": "derivative",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(laws.G_BOUNDS),
    "alpha": ALPHA,
    "learner_folds": LEARNER_FOLDS,
    "stratify_folds": STRATIFY_FOLDS,
    "outcome_learner": "unpenalized logistic regression on [A, W1, W2, W3, W4]",
    "declared_pair": (
        "tests/canonical/ctmle_derivative/penalties.csv: the lower median lambda_cv of "
        f"{PENALTY_DRAWS} design draws of the primary law at n = {PRIMARY_N:,} (seeds "
        f"{PENALTY_SEED} onward), and that draw's grid neighbour"
    ),
    "pipeline_differences": (
        "the cross-validation folds; the partner (R: 1.05 lambda.min; package: the nearest grid "
        "neighbour, the next smaller penalty); the package appends the penalty 0 to its grid"
    ),
    "paired_tmle": (
        "every primary and calibration-family fit also reports the ordinary TMLE on the same "
        "initial fit, mechanism and g_bounds, with only the derivative columns removed"
    ),
}

CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def outcome_learner() -> LogisticRegression:
    """The study's outcome learner: an unpenalized main-terms logistic GLM."""
    return LogisticRegression(C=1e6, max_iter=2000, tol=1e-10)


def estimator(
    *,
    lambdas: tuple[float, float] | None,
    outcome: Any = None,
    cross_fit: bool = False,
    three_arm: bool = False,
    missingness: Any = None,
    random_state: int = 0,
) -> CTMLE:
    """The study's subject with every declared setting.

    Parameters
    ----------
    lambdas : tuple of float or None
        The declared pair, or ``None`` for the internal penalty cross-validation.
    outcome : estimator or None, optional
        The outcome learner.  ``None`` is :func:`outcome_learner`.
    cross_fit : bool, optional
        Whether the nuisances are cross-fitted over five folds.
    three_arm : bool, optional
        Whether the treatment has three arms; the contrasts then read arm 0.
    missingness : estimator or None, optional
        The response learner of a fit with missing outcomes.
    random_state : int, optional
        The estimator's seed.

    Returns
    -------
    CTMLE
        The configured, unfitted estimator.
    """
    return CTMLE(
        strategy="derivative",
        lambdas=lambdas,
        learner_folds=LEARNER_FOLDS,
        outcome_learner=outcome_learner() if outcome is None else outcome,
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=missingness,
        cross_fit=cross_fit,
        n_folds=5,
        estimands=("ate",),
        reference=0.0 if three_arm else None,
        simultaneous=False,
        g_bounds=laws.G_BOUNDS,
        alpha=ALPHA,
        stratify_folds=STRATIFY_FOLDS,
        max_iter=100,
        tol=1e-10,
        random_state=random_state,
    )


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    """Fit the subject on one primary sample of ``scenario``."""
    lambdas = declared_pair() if scenario == "binary_declared_pair" else None
    return (
        estimator(lambdas=lambdas)
        .fit(frame, outcome="Y", treatment="A", covariates=list(laws.COVARIATES))
        .single()
    )


def paired_tmle(result: Any, estimands: tuple[str, ...] = ("ate",)) -> dict[str, Any]:
    """The ordinary TMLE of ``result``'s fit: the same initial fit, mechanism and bounds.

    The partner and the derivative fit's targeted regression are removed, so the retarget runs
    the plain ``K``-column update from the initial outcome regression.

    Parameters
    ----------
    result : TMLEResult
        One fitted C-TMLE0 result.
    estimands : tuple of str, optional
        The estimands to report.

    Returns
    -------
    dict
        Each estimand's retargeted estimate, which carries its own interval.
    """
    plain = replace(result.nuisance, treatment_partner=None, targeting_outcome=None)
    estimates, _ = result.estimator.retarget(
        result.data,
        plain,
        estimands=sorted({name.split("[", 1)[0] for name in estimands}),
        g_bounds=result.config.g_bounds,
    )
    return {name: estimates[name] for name in estimands}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIO_ESTIMANDS:
        raise KeyError(scenario)
    truth = laws.PRIMARY.truth()
    return laws.PRIMARY.sample(n, seed), {name: truth[name] for name in STUDY.scenarios[scenario]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def initials(result: Any) -> dict[str, float]:
    """The untargeted plug-in ``ate``, as the R runner forms it from its own ``Q``."""
    arms = result.nuisance.outcome.arms
    codes = list(result.data.arm_codes)
    return {"ate": float(np.mean(arms[codes[1]]) - np.mean(arms[codes[0]]))}


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    plain = paired_tmle(result)["ate"]
    fit = result.extra["ctmle"]
    ((mechanism, partner),) = fit.penalties
    sample = frame.loc[:, [*laws.COVARIATES, "A", "Y"]].copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    rows = primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=STUDY.scenarios[scenario],
        initials=initials(result),
    )
    low, high = plain.ci
    for row in rows:
        row["tmle_estimate"] = float(plain.psi)
        row["tmle_std_error"] = float(plain.std_error)
        row["tmle_covered"] = int(low <= row["truth"] <= high)
        row["mechanism_penalty"] = float(mechanism)
        row["partner_penalty"] = float(partner)
    return sample, truth_rows, rows


#: The columns the Python rows carry beside the published schema, read by
#: :func:`extra_artifacts`.
EXTRA_COLUMNS = (
    "tmle_estimate",
    "tmle_std_error",
    "tmle_covered",
    "mechanism_penalty",
    "partner_penalty",
)
#: The columns the R rows carry beside the published schema.
REFERENCE_COLUMNS = ("mechanism_penalty", "partner_penalty")


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [
            ((scenario, replicate, n),)
            for scenario in SCENARIO_ESTIMANDS
            for replicate in range(replicates)
        ],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]]


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The paired ordinary TMLE, and the penalty pair of every primary fit.

    ``paired-tmle.csv`` publishes, per scenario, the mean C-TMLE0 minus TMLE difference, its
    standard deviation, and each estimator's coverage and mean standard error over the primary
    replications.  The numbers are descriptive.  ``chosen-penalties.csv`` publishes the
    ``(mechanism, partner)`` pair of each side, per scenario and replication.
    """
    subject = rows.loc[rows["implementation"] == IMPLEMENTATION].copy()
    subject["difference"] = subject["estimate"] - subject["tmle_estimate"]
    paired = (
        subject.groupby(["scenario", "estimand"], sort=True)
        .agg(
            mean_difference=("difference", "mean"),
            sd_difference=("difference", "std"),
            derivative_coverage=("covered", "mean"),
            tmle_coverage=("tmle_covered", "mean"),
            derivative_mean_std_error=("std_error", "mean"),
            tmle_mean_std_error=("tmle_std_error", "mean"),
        )
        .reset_index()
    )
    penalties = rows.loc[
        :, ["implementation", "scenario", "replicate", "mechanism_penalty", "partner_penalty"]
    ].sort_values(["scenario", "replicate", "implementation"], ignore_index=True)
    return {"paired-tmle.csv": paired, "chosen-penalties.csv": penalties}


def _penalty_draw(draw: int) -> dict[str, Any]:
    seed = PENALTY_SEED + draw
    frame = laws.PRIMARY.sample(PRIMARY_N, seed)
    result = (
        estimator(lambdas=None, random_state=draw)
        .fit(frame, outcome="Y", treatment="A", covariates=list(laws.COVARIATES))
        .single()
    )
    ((mechanism, partner),) = result.extra["ctmle"].penalties
    return {"draw": draw, "seed": seed, "lambda_cv": mechanism, "partner": partner}


def penalty_design(*, draws: int = PENALTY_DRAWS, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """The package pipeline's ``(lambda_cv, partner)`` on each design draw of the primary law.

    Parameters
    ----------
    draws : int, optional
        The number of design draws.
    n_jobs : int, optional
        Worker processes.

    Returns
    -------
    DataFrame
        One row per draw: ``draw``, ``seed``, ``lambda_cv`` and ``partner``.
    """
    rows = map_parallel(_penalty_draw, [(draw,) for draw in range(draws)], n_jobs=n_jobs)
    return pd.DataFrame(rows)


def median_draw(design: pd.DataFrame) -> pd.Series:
    """The design draw at the lower median ``lambda_cv``; its pair is the declared pair."""
    ordered = design.sort_values(["lambda_cv", "draw"], ignore_index=True)
    return ordered.iloc[(len(ordered) - 1) // 2]


def write_penalties(design: pd.DataFrame, directory: Path = ARTIFACTS) -> tuple[float, float]:
    """Write ``penalty-design.csv`` and the declared pair to ``penalties.csv``."""
    chosen = median_draw(design)
    pair = (float(chosen["lambda_cv"]), float(chosen["partner"]))
    design.to_csv(
        directory / PENALTY_DESIGN.name, index=False, float_format="%.17g", lineterminator="\n"
    )
    pd.DataFrame({"role": ["mechanism", "partner"], "lambda": list(pair)}).to_csv(
        directory / PENALTIES.name, index=False, float_format="%.17g", lineterminator="\n"
    )
    return pair


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix the declared penalty pair of the study.")
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - the pre-declaration step, run by hand
    arguments = _arguments()
    pair = write_penalties(penalty_design(n_jobs=arguments.jobs))
    print(f"declared pair: mechanism {pair[0]!r}, partner {pair[1]!r}")
    if not math.isfinite(pair[0]):
        raise SystemExit("the penalty design produced a non-finite penalty")
