"""Registered evidence for the default simultaneous band of every shipped fit shape.

``TMLE`` and ``LTMLE`` default to ``simultaneous=True``.  A fit that reports two or more
inferential estimates therefore publishes a max-t multiplier band over all of them.  Four
registered studies measure the band of the families the roadmap names (multi-arm arms,
longitudinal regimens, survival curves and competing-risk incidences) and the strata study
measures it across baseline strata.  The multi-arm missing-outcome DR-TMLE study measures
its own three-arm band in the same way.  This study measures the band of every other shipped
shape that publishes one.

Each shape is one registered source study's subject fit, at that study's own primary law
and primary sample size, refitted with ``simultaneous=True`` and nothing else changed.  The
law, the sampler, the truth oracle and the fit function are the source study's own.  The
samples are this study's: each replication is drawn through the source study's
``draw_from_seed`` at a seed this record owns.

The primary scenario ``default_fit`` is the shipped ``TMLE()`` default on the ordinary
point-treatment binary law: ten-fold pooled cross-fitting, the default binary estimands and
the default band.  Only the two learners and ``random_state`` are set.  Its primary rows are
the pointwise results of that fit, gated against the exact truth.

Declared before any run:

* the reading rule of every joint cell is ``simultaneous_coverage_verdicts`` with the shared
  ``Margins()``: a band's 99% joint-coverage interval inside ``[0.92, 0.98]``, and its
  pointwise control's 99% upper endpoint below ``0.95``;
* :data:`SHAPES` fixes each cell's source, sample size and budget;
* ``publication_policy="reporting"``.  Three shapes carry a pointwise calibration cell that is
  already red under a recorded owner: the static calibration cells of the ordinary and the
  cross-fitted weighted longitudinal studies, and the calibration cell of multi-arm DR-TMLE.
  A red joint cell is diagnosed against the oracle band before it is published, and it
  receives an owner in the roadmap's red-cell table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_tmle
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 1_000
SEED = 20263601
RESAMPLING_SEED = 20263602
SCENARIO = "default_fit"
#: The shipped default estimands of a binary point-treatment fit, in the order it reports them.
ESTIMANDS = ("ate", "att", "atc", "ey1", "ey0", "rr", "or")


@dataclass(frozen=True)
class Shape:
    """One shipped fit shape whose default band a joint cell measures.

    Parameters
    ----------
    label : str
        The joint cell prefix.
    source : str
        The slug of the registered study whose subject fit this is.
    description : str
        What the shape adds to the band's construction.
    n : int
        The sample size, the source study's primary size.
    replicates : int
        The declared replication budget of the cell.
    pointwise_control : bool
        Whether the cell keeps its pointwise joint control.
    design_fits : int
        How many fits the design computation averages the family's curve covariance over.
        Fewer for the four slowest fits, whose ``p0`` sits far from the control's limit.
    """

    label: str
    source: str
    description: str
    n: int
    replicates: int
    pointwise_control: bool = True
    design_fits: int = 10


#: Every shape this study measures.  The budgets come from the design computation in
#: :mod:`tests.studies.default_band_properties`, fixed before any run.
SHAPES: tuple[Shape, ...] = (
    Shape("default_fit", "default-simultaneous-bands", "the shipped TMLE() defaults", 1_000, 2_400),
    Shape(
        "ordinary_binary",
        "canonical-tmle",
        "in-sample TMLE over ten estimands, with log-scale ratio bands and the PAF",
        1_000,
        2_400,
    ),
    Shape(
        "fold_evaluated",
        "fold-evaluated-cvtmle",
        "cross-validated variance beside globally centered multiplier draws",
        1_000,
        2_400,
    ),
    Shape("weighted", "weighted-tmle", "fixed observation weights, with ratios", 2_000, 2_400),
    Shape(
        "learned_weighted",
        "learned-weighted-tmle",
        "weighted nuisances and a linear fluctuation",
        2_000,
        2_400,
    ),
    Shape("missing_outcome", "mar-tmle", "a missing outcome", 2_000, 2_400),
    Shape(
        "missing_outcome_drtmle",
        "mar-drtmle",
        "DR-TMLE with a missing outcome",
        2_000,
        2_400,
        design_fits=3,
    ),
    Shape(
        "drtmle",
        "canonical-drtmle",
        "cross-fitted DR-TMLE corrected curves",
        3_000,
        2_400,
        design_fits=3,
    ),
    Shape(
        "multi_arm_drtmle",
        "canonical-multi-arm-drtmle",
        "multi-arm DR-TMLE with ratios",
        2_000,
        2_400,
        design_fits=3,
    ),
    Shape("cde_z0", "cde-tmle", "the controlled direct effect at Z = 0", 2_000, 2_400),
    Shape("cde_z1", "cde-tmle", "the controlled direct effect at Z = 1", 2_000, 2_400),
    Shape("point_msm", "point-msm", "point-treatment MSM coefficients", 2_000, 2_400),
    Shape(
        "clustered", "clustered-tmle", "grouped cross-fitting and cluster multipliers", 2_000, 2_400
    ),
    Shape(
        "shift_grid",
        "shift-policies",
        "a shift grid and its contrasts",
        2_000,
        2_400,
        design_fits=3,
    ),
    Shape(
        "incremental_grid",
        "incremental-interventions",
        "an incremental odds grid and its contrasts",
        2_000,
        2_400,
    ),
    Shape("stochastic_regimes", "stochastic-regimes", "a known stochastic regime", 2_000, 2_400),
    Shape("deterministic_regimes", "deterministic-regimes", "a dynamic rule", 2_000, 2_400),
    Shape(
        "ltmle_crossfit",
        "canonical-ltmle-crossfit",
        "cross-fitted end-of-study regimens",
        2_000,
        2_400,
    ),
    Shape(
        "survival_crossfit",
        "canonical-ltmle-survival-crossfit",
        "a cross-fitted survival curve with two exact duplicate pairs",
        2_000,
        2_400,
    ),
    Shape(
        "competing_crossfit",
        "canonical-ltmle-competing-crossfit",
        "cross-fitted cumulative incidence over both horizons with four exact duplicates",
        4_000,
        2_400,
    ),
    Shape("weighted_ltmle", "weighted-ltmle", "weighted end-of-study regimens", 2_000, 2_400),
    Shape(
        "weighted_ltmle_crossfit",
        "weighted-ltmle-crossfit",
        "cross-fitted weighted end-of-study regimens",
        2_000,
        2_400,
    ),
    Shape(
        "categorical_ltmle",
        "canonical-categorical-ltmle",
        "five categorical regimens",
        2_000,
        2_400,
    ),
    Shape(
        "categorical_ltmle_crossfit",
        "canonical-categorical-ltmle-crossfit",
        "five cross-fitted categorical regimens",
        2_000,
        2_400,
    ),
    Shape("longitudinal_msm", "longitudinal-msm", "longitudinal MSM coefficients", 2_500, 4_000),
)

LABELS = tuple(shape.label for shape in SHAPES)


def _joint_cells() -> tuple[str, ...]:
    cells: list[str] = []
    for shape in SHAPES:
        cells.append(f"{shape.label}__simultaneous_band")
        if shape.pointwise_control:
            cells.append(f"{shape.label}__pointwise_joint_control")
    return tuple(cells)


STUDY = StudyRecord(
    name="default simultaneous bands across shipped fit shapes",
    slug="default-simultaneous-bands",
    artifacts=ROOT / "tests" / "canonical" / "default_bands",
    document="docs/technical-reference/method-evidence/default-simultaneous-bands.md",
    anchor="default-simultaneous-bands",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-default-bands",
    reference=None,
    modules=(
        "tests/conftest.py",
        "tests/discrete_law.py",
        "tests/discrete_law_cde.py",
        "tests/discrete_law_competing.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/discrete_law_mar.py",
        "tests/incrementals.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/canonical_categorical_ltmle_crossfit.py",
        "tests/studies/canonical_cde_tmle.py",
        "tests/studies/canonical_clustered_tmle.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_deterministic_regimes.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/canonical_incremental_interventions.py",
        "tests/studies/canonical_learned_weighted_tmle.py",
        "tests/studies/canonical_longitudinal_msm.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_ltmle_competing.py",
        "tests/studies/canonical_ltmle_competing_crossfit.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle_survival.py",
        "tests/studies/canonical_ltmle_survival_crossfit.py",
        "tests/studies/canonical_mar_drtmle.py",
        "tests/studies/canonical_mar_tmle.py",
        "tests/studies/canonical_multi_arm_drtmle.py",
        "tests/studies/canonical_point_msm.py",
        "tests/studies/canonical_shift_policies.py",
        "tests/studies/canonical_stochastic_regimes.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/canonical_weighted_ltmle.py",
        "tests/studies/canonical_weighted_ltmle_crossfit.py",
        "tests/studies/canonical_weighted_tmle.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/studies/cde_study_helpers.py",
        "tests/studies/default_band_properties.py",
        "tests/studies/default_bands.py",
        "tests/studies/fold_evaluated_cvtmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/intervention_study_helpers.py",
        "tests/studies/learned_weighted_point_common.py",
        "tests/studies/ltmle_crossfit_properties.py",
        "tests/studies/ltmle_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/weighted_longitudinal_common.py",
        "tests/studies/weighted_longitudinal_properties_common.py",
        "tests/studies/weighted_point_common.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.default_bands",
    properties_module="tests.studies.default_band_properties",
    property_cells={"simultaneous_coverage": _joint_cells()},
    publication_policy="reporting",
)

CONFIGURATION = {
    "primary_fit": (
        "TMLE() with outcome_learner and treatment_learner LogisticRegression(C=1e6, "
        "max_iter=2000) on W1..W3, random_state=0, and every other argument at its shipped "
        "default: cross_fit=True, n_folds=10, pooled targeting, the default binary estimands, "
        "simultaneous=True with 1000 rademacher draws"
    ),
    "primary_law": "the canonical-tmle binary law (binary_outcome_dgp)",
    "simultaneous_intervals": True,
    "band_cells": (
        "each shape refits its source study's subject fit_cleverly with simultaneous=True at "
        "the source study's primary law and size; the band covers every estimate the fit "
        "reports, with the engine's default draws (1000 point, 2000 sequential), rademacher "
        "multipliers and the subject fit's random_state"
    ),
    "shapes": {
        shape.label: {
            "source": shape.source,
            "n": shape.n,
            "replicates": shape.replicates,
            "pointwise_control": shape.pointwise_control,
        }
        for shape in SHAPES
    },
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample of the canonical binary law from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return canonical_tmle.draw_from_seed("binary", n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    """The shipped default fit: only the two learners and the seed are set."""
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(
            outcome_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one replication."""
    result = fit_cleverly(frame)
    if tuple(result.estimates) != ESTIMANDS:
        raise AssertionError(f"the default fit reported {tuple(result.estimates)}")
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=SCENARIO,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def _replicate(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, n = payload
    frame, truth = draw_scenario(SCENARIO, n, replicate)
    return cleverly_rows(frame, truth, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Draw and fit every declared primary replication."""
    payloads = [((index, n),) for index in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    built = pd.DataFrame([row for records in outcomes for row in records])
    return built.loc[:, list(REPLICATE_COLUMNS)]
