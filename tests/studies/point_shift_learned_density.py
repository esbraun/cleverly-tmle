r"""Declared evidence for a point additive shift whose density is learned (X121 part b2).

The question is whether TMLE's interval for a point additive shift of a continuous dose is
calibrated when the dose's conditional density is learned on the binned density route, in sample
and cross-fitted. The registered shift studies (``shift-policies``, ``policy-point-mtp``) give
the method the exact density, so they say nothing about a learned one, and a fit that learns its
density reads ``experimental`` (``density_learner_family`` in ``src/cleverly/_support.py``).

The law, the parameters and the three arms are those of
:mod:`tests.studies._point_shift_learned_density_law`: the interventions page's dose fit
(``in_sample_learned``), the same learners cross-fitted on three folds with the outcome range
declared from the law (``crossfit_learned``), and the exact density on the fit's own grid
(``in_sample_oracle``, diagnostic). Every arm fits the same draw with the same random state.

**Primary.** :data:`PRIMARY_REPLICATES` draws at n = 3,000, one scenario whose 21 labelled
parameters are each arm's three contrasts and four policy means, each held to the shared bias
leg, coverage floor and SE sanity band. The harness records each fit's density-ratio SD, mean and
RMSE against the law's ratio at the observed doses. One draw fits all three arms in one call, so
a fit that raises (for example an outcome outside the cross-fitted arm's declared range, chance
at most 0.014 over the whole budget) loses that draw for every arm, and the failure accounting
counts it.

**``interval_calibration``.** The cross-fitted arm's three contrasts, each with the shared
shrunken-SE control (factor 0.70) and noise control (one efficient-influence-curve SD), on
:data:`~tests.studies.point_shift_learned_density_properties.CALIBRATION_REPLICATES` property
draws. No efficiency band is read: the learned ratio is noisier than the law's, and the claim is
calibration, not efficiency. The in-sample arms are not in this family. The page probe measured
the in-sample learned arm's SE ratio at 0.76 to 0.85 on 120 draws, and the primary rows publish
both in-sample arms' coverage and SE-ratio intervals. A green primary cell of an in-sample arm
does not show calibration: the 0.90 coverage floor and the [0.8, 1.2] SE sanity band pass an
interval whose SE ratio is 0.85.

**Sizing, declared before the pilot.** The 3,200-draw pilot
(:mod:`tests.studies.point_shift_learned_density_pilot`) records each contrast's SE ratio, the
bootstrap SD of that ratio and the spread constant it implies. The calibration count is the
per-cell predictive rule at the larger of the shared constant and the pilot's, capped at the
95th percentile of that rule over ideal pilots at the same constant, and never above a ceiling
of 19,600 fixed now. The run proceeds at the smallest of the rule, the cap and the ceiling. A run
at the ceiling may come back red on a finite-sample basis; such a red is routed as below. A design change after the pilot needs a diagnosed defect, a
written reason, and a fresh pilot on new seeds.

**Reading, declared before any run.**

- If every primary verdict of the cross-fitted arm passes (bias leg, coverage floor and SE sanity
  of its seven parameters) and every ``interval_calibration`` verdict passes (positives and
  controls), the registering change adds :func:`support_row`.
- The in-sample learned arm adds no row whatever it reads. A red primary cell of that arm is a
  finite-sample limit of the in-sample learned density, published under
  ``publication_policy="reporting"`` with owner :data:`RED_CELL_OWNER`, and the method-evidence
  page states the shortfall.
- A red cell of the cross-fitted arm is diagnosed first. A defect keeps the row out. A red with
  no defect keeps the row out too, whatever its direction: a cell whose SE-ratio interval
  contains 1, and a cell that is consistently off (an SE ratio above or below the band, or
  coverage outside [0.92, 0.98]). It is published under ``reporting`` with owner
  :data:`RED_CELL_OWNER`, the page states the direction, and a fit keeps reading
  ``experimental``. No red cell is re-run on new seeds.
- The oracle arm licenses nothing. If its primary cells are red, the reading says that the
  in-sample learned arm's shortfall is not all the learner's.
- A control that does not fail is a design defect.

**Before the declared run.** A smoke of the primary path drew primary replicates 0 to 7 of the
first declared seeds (20261211, 20261212) into scratch, and its rows were discarded. The record's
seeds were then rotated to unused values, so no declared draw has been seen.

There is no comparator: no external package fits this estimator with this density learner, so
``equivalence.csv`` has no rows.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _point_shift_learned_density_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SLUG = "point-shift-learned-density"
#: The record's seeds, checked against every registered study's seeds and the page probe's
#: 9000-range seeds by ``test_point_shift_learned_density_design``.
SEED = 20261221
RESAMPLING_SEED = 20261222
#: The roadmap ask that owns a red cell of this study, named before the run.
RED_CELL_OWNER = "X121-learned-density-finite-sample"

#: The primary draws: the study's floor of 1,000, as a multiple of 400.
PRIMARY_REPLICATES = 1_200
PRIMARY_N = law.N
SCENARIO = "shift_law"


def label(arm: str, parameter: str) -> str:
    """The estimand label of one arm's parameter."""
    return f"{arm}: {law.PARAMETERS[parameter]}"


LABELS = {(arm, p): label(arm, p) for arm in law.ARMS for p in law.PARAMETERS}
SCENARIOS = {SCENARIO: tuple(LABELS.values())}

#: The arm whose calibration the study gates, and the cells that gate it.
GATED_ARM = "crossfit_learned"
POSITIVE = "declared"
CALIBRATION_LABELS = tuple(f"{GATED_ARM}_{tag}" for tag in law.CONTRASTS)
#: The efficient-influence-curve SD of each contrast
#: (:func:`~tests.studies._point_shift_learned_density_law.efficiency_bounds`), which sizes the
#: noise control. ``test_point_shift_learned_density_design`` recomputes them.
TAG_BOUNDS = {"capped": 0.540505, "uncapped": 0.529648, "one": 1.222596}
EFFICIENCY_BOUNDS = {f"{GATED_ARM}_{tag}": TAG_BOUNDS[tag] for tag in law.CONTRASTS}
#: The SD of each contrast's true density ratio at the observed dose, reported beside the fitted
#: ratio's SD (:func:`~tests.studies._point_shift_learned_density_law.true_ratio_sd`).
TRUE_RATIO_SD = {"capped": 0.548355, "uncapped": 0.532941, "one": 1.310856}

PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{cell}__{kind}"
        for cell in CALIBRATION_LABELS
        for kind in (POSITIVE, "shrunken_se_control", "noise_control")
    ),
}

#: What the row added by a passing reading states beyond its key.
ROW_LIMITS = (
    "Measured at n = 3,000 on one law (shift_dgp, a normal dose with a quadratic outcome) for "
    "additive shifts of +0.5 (capped and uncapped) and +1.0, with a regularized "
    "HistGradientBoostingClassifier density (max_depth 2, learning rate 0.05, 200 iterations, "
    "l2 1.0) on 40 core bins, three folds and an outcome range declared from the law. Other "
    "policy kinds, booster settings, bin counts and sample sizes are not measured. An in-sample "
    "learned density is not covered."
)


def support_row() -> Any:
    """The support row a passing reading adds, declared before any run.

    Built from the cross-fitted arm's computed key: stacked splitting, a bounded continuous
    outcome (the declared range), the density route, and the booster class as the treatment
    learner. The cross-fitted key carries no ``density_learner_family``, so the learner class is
    what keeps an oracle or default-learner fit out of the row
    (``test_point_shift_learned_density_design``).
    """
    from dataclasses import replace

    from sklearn.ensemble import HistGradientBoostingClassifier

    from cleverly._support_rows import point_row

    row = point_row(
        "T-point-shift-learned-density",
        [SLUG],
        families=("P7",),
        outcome=("gaussian", "bounded"),
        treatment=("continuous",),
        splitting=("stacked",),
        options={"ratio": "density"},
        cells=list(CALIBRATION_LABELS),
        limits=[ROW_LIMITS],
    )
    return replace(row, learners={"treatment": HistGradientBoostingClassifier})


#: The per-draw columns beside the shared schema.
HARNESS_COLUMNS = ("ratio_sd", "ratio_mean", "ratio_rmse")
#: The harness artifact: one row per primary replication and parameter.
HARNESS_ARTIFACT = "harness.csv.gz"

FAILURE_BOUND = 0.01

STUDY = StudyRecord(
    name="A point additive shift with a learned density",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "point_shift_learned_density",
    document="docs/technical-reference/method-evidence/point-shift-learned-density.md",
    anchor="a-point-additive-shift-with-a-learned-density",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-point-shift-learned-density",
    reference=None,
    failure_bound={"cleverly-point-shift-learned-density": FAILURE_BOUND},
    modules=(
        "tests/studies/point_shift_learned_density.py",
        "tests/studies/point_shift_learned_density_properties.py",
        "tests/studies/point_shift_learned_density_pilot.py",
        "tests/studies/mtp_study_tools.py",
        "tests/studies/_point_shift_learned_density_law.py",
        "tests/studies/canonical_shift_policies.py",
        "tests/studies/point_study_helpers.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/learners/density.py",
        "src/cleverly/interventions/policy.py",
        "src/cleverly/_support.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.point_shift_learned_density",
    properties_module="tests.studies.point_shift_learned_density_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
    extra_artifacts=(HARNESS_ARTIFACT,),
)

CONFIGURATION = {
    "law": "shift_dgp() at its defaults (make_shift_dose), n = 3000",
    "policies": [name for _, _, name in law.POLICIES],
    "parameters": dict(law.PARAMETERS),
    "truths": dict(law.TRUTH),
    "outcome_learner": "make_pipeline(PolynomialFeatures(2), LinearRegression())",
    "density_learner": f"HistGradientBoostingClassifier({law.BOOSTER}, random_state=<fit seed>)",
    "density_bins": law.DENSITY_BINS,
    "in_sample_learned": "TMLE(cross_fit=False, ratio='density')",
    "crossfit_learned": (
        f"TMLE(cross_fit=True, n_folds={law.N_FOLDS}, q_bounds={law.Q_BOUNDS}, ratio='density')"
    ),
    "in_sample_oracle": (
        "TMLE(cross_fit=False, ratio='density') with OracleShiftDensity(shift_dgp(), "
        "bin_edges(A, 40))"
    ),
    "q_bounds": (
        f"the outcome's {law.TAIL_PROBABILITY} and 1 - {law.TAIL_PROBABILITY} quantiles under "
        "the law, rounded outward"
    ),
    "fit_seed": "stream_seed(record, 'fit_seed', 'primary', scenario, replicate)",
    "efficiency_bounds": dict(TAG_BOUNDS),
    "true_ratio_sd": dict(TRUE_RATIO_SD),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed, and every label's truth."""
    if scenario != SCENARIO:
        raise KeyError(f"{SLUG} has no scenario {scenario!r}")
    truths = {name: law.TRUTH[tag] for (_, tag), name in LABELS.items()}
    return law.draw(n, seed), truths


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def initial_estimate(result: Any, parameter: str) -> float:
    """The untargeted plug-in of one parameter: a policy's mean, or a shifted minus the natural."""
    nuisance = result.nuisance
    weights = np.asarray(result.data.weights, dtype=float)

    def mean(index: int) -> float:
        levels = nuisance.scaler.unscale_levels(nuisance.outcome.arms[float(index)])
        return float(np.average(levels, weights=weights))

    index = list(law.SHIFT_OF).index(law.policy_of(parameter))
    if parameter.startswith("ey_"):
        return mean(index)
    return mean(index) - mean(0)


def arm_rows(
    result: Any, arm: str, frame: pd.DataFrame, where: tuple[str, int, int], record: StudyRecord
) -> list[dict[str, Any]]:
    """The primary rows of one arm's fit."""
    scenario, replicate, n = where
    rows = []
    for tag, name in law.PARAMETERS.items():
        estimate = result.estimates[name]
        std_error, low, high = reported_inference(estimate)
        truth = law.TRUTH[tag]
        diagnostics = law.ratio_diagnostics(result, tag, frame)
        rows.append(
            {
                "implementation": record.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": LABELS[arm, tag],
                "truth": truth,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": initial_estimate(result, tag),
                "ratio_sd": diagnostics.ratio_sd,
                "ratio_mean": diagnostics.ratio_mean,
                "ratio_rmse": diagnostics.ratio_rmse,
            }
        )
    return rows


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    seed = stream_seed(record, "fit_seed", "primary", scenario, replicate)
    rows = []
    for arm in law.ARMS:
        rows += arm_rows(law.fit(arm, frame, seed), arm, frame, key, record)
    return rows


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.draw(size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``, with its harness columns."""
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, n)
            for scenario in record.scenarios
            for replicate in range(replicates)
        ],
        partial(_primary, record),
        draw=partial(_sample, record),
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *HARNESS_COLUMNS]])


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication and parameter."""
    columns = ["scenario", "replicate", "estimand", *HARNESS_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "replicate", "estimand"], ignore_index=True
        )
    }
