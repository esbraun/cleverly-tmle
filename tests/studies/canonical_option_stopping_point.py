r"""Declared evidence for the stopping rules of point TMLE (``option-stopping-point``, X76 (a)).

The subject is ``TMLE(stopping="scaled_var")`` and ``TMLE(stopping="sample_size")``, each under
``targeting="iterative"`` and ``targeting="one_step"``.  Under either rule the targeting walk
stops before the score equation is solved, and the reported interval adds the remaining score
``|r_n|`` to its half-width, ``psi +/- (z se + |r_n|)`` (``ParameterEstimate.rule_widened``;
Fix-8 plan section 2.1).  That interval covers whenever ``|Z| <= z``, whatever the residual's
dependence on the error, so its large-sample level holds for any rule that drives
``r_n / se`` to a bounded value.  The study measures it at n = 2,000.

The laws, learners and fits are :mod:`tests.studies._option_stopping_laws` (``L1``, ``L2``,
``LN``).  Every fit is in sample, ``max_iter=100`` (200 for a one-step walk), ``tol=1e-10``,
with ``canonical-tmle``'s ``g_bounds``.  The property families
(:mod:`tests.studies.option_stopping_point_properties`):

==========================  ======================================================================
family                      cells
==========================  ======================================================================
``interval_calibration``    the ATE's widened interval under each rule and targeting on the L1
(K)                         binary and continuous laws: 8 gated positives, each with the derived
                            ``shrunken_se_control`` and ``noise_control``
``residual_witness`` (W)    the share of draws whose residual is at least 0.01 SE, on the same
                            fits: the two ``scaled_var`` one-step cells gate (lower limit at
                            least :data:`WITNESS_SHARE`); the other six report
``residual_necessity`` (N)  LN, one-step ``scaled_var`` with every threshold times 8, EY1: the
                            plain interval's coverage upper limit below
                            :data:`NECESSITY_CEILING` (control) and the widened interval's lower
                            limit at least :data:`NECESSITY_FLOOR` (positive), on the same fits
reported                    the plain interval of every K cell; the relative fit on the same
                            draws; LN at threshold times 1 and under the relative rule; L2's
                            one-step EY1 and ATE under each rule, plain and widened; the
                            ``sample_size`` Newton walk with ``step_bound=0.05`` (the ``tmle3``
                            pairing, which reads experimental)
==========================  ======================================================================

Every cell publishes its rule-met share and the distribution of its absolute residual.

**No comparator.**  No package widens a Wald interval by its remaining score, so the study
reads each cell against its exact truth alone.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/option_stopping_point/regenerate.py`` before any run.

**Declared, not registered.**  The record is not in ``registered()`` until its run is
committed; the ``T52`` support rows land with that registration.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _option_stopping_cells as grid
from tests.studies import _option_stopping_laws as laws
from tests.studies import canonical_tmle
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "option-stopping-point"
SEED = 20267621
RESAMPLING_SEED = 20267622
FAILURE_BOUND = 0.01
PRIMARY_N = laws.N

#: The replications of every property cell.  Plan 5.1's floor is the smallest multiple of 400
#: whose product of single-cell calibration pass probabilities under perfect calibration is at
#: least 0.95 (3,200).  The arm-pooled predictive rule
#: (:func:`tests.studies.repeated_sizing.calibration_replicates`) at the committed 2,000-draw
#: pilot raises it to this count: the cells of one law read the same draws (pilot correlation
#: of their estimates at least 0.998), so the pilot holds one SE-ratio reading per law, and the
#: heterogeneity bound between the laws' readings is wide.  The design test recomputes it.
REPLICATES = 16_400
#: The primary phase's replications: plan 5.1's floor, above the 1,000 a primary needs.
PRIMARY_REPLICATES = 3_200

#: W's gate: the lower limit of the witness share (plan 5.1).
WITNESS_SHARE = 0.5
#: N's gates (plan 5.1): the control's plain upper limit, the positive's widened lower limit.
NECESSITY_CEILING = 0.92
NECESSITY_FLOOR = 0.90
#: The ``tmle3`` pairing's step bound (plan 5.1, ``step_bound`` row).
STEP_BOUND = 0.05

OUTCOMES = ("binary", "continuous")

#: The fits whose W cell gates (plan 5.1); the pre-read keeps both above a pilot share of 0.7.
WITNESS_GATED = ("sv_one_step",)


def _label(rule: str, targeting: str) -> str:
    return f"{laws.RULE_LABELS[rule]}_{laws.TARGETING_LABELS[targeting]}"


def _point_fits(law: str) -> dict[str, Any]:
    fits: dict[str, Any] = {
        "rel_newton": functools.partial(laws.fit_point, law, stopping="relative"),
    }
    for rule in laws.RULES:
        for targeting in laws.TARGETINGS:
            fits[_label(rule, targeting)] = functools.partial(
                laws.fit_point, law, stopping=rule, targeting=targeting
            )
    if law == "binary":
        fits["ss_newton_step_bound"] = functools.partial(
            laws.fit_point, law, stopping="sample_size", step_bound=STEP_BOUND
        )
    return fits


def _draw_l1(outcome: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return laws.draw_l1(outcome, n, seed)


STREAMS: dict[str, grid.Stream] = {
    **{
        outcome: grid.Stream(
            outcome,
            functools.partial(_draw_l1, outcome),
            _point_fits(outcome),
            laws.point_rule_met,
        )
        for outcome in OUTCOMES
    },
    "LN": grid.Stream(
        "LN",
        laws.draw_ln,
        {
            f"sv_one_step_k{scale:g}": functools.partial(
                laws.fit_point,
                "LN",
                stopping="scaled_var",
                targeting="one_step",
                estimands=("ey1",),
                scale=scale,
            )
            for scale in (laws.NECESSITY_SCALE, 1.0)
        }
        | {
            "rel_one_step": functools.partial(
                laws.fit_point, "LN", stopping="relative", targeting="one_step", estimands=("ey1",)
            )
        },
        laws.point_rule_met,
    ),
    "L2": grid.Stream(
        "L2",
        laws.draw_l2,
        {
            _label(rule, "one_step"): functools.partial(
                laws.fit_point, "L2", stopping=rule, targeting="one_step"
            )
            for rule in laws.RULES
        },
        laws.point_rule_met,
    ),
}

#: Each stream's exact truths.
TRUTHS: dict[str, dict[str, float]] = {
    **{
        outcome: {
            name: float(value)
            for name, value in canonical_tmle.truth_for(
                canonical_tmle.scenario_dgp(outcome)
            ).items()
        }
        for outcome in OUTCOMES
    },
    "LN": dict(laws.LN_TRUTH),
    "L2": dict(laws.L2_TRUTH),
}


def _cells() -> tuple[grid.Cell, ...]:
    cells: list[grid.Cell] = []
    for outcome in OUTCOMES:
        for rule in laws.RULES:
            for targeting in laws.TARGETINGS:
                fit = _label(rule, targeting)
                label = f"{outcome}_{fit}"
                cells += [
                    grid.Cell(
                        "interval_calibration",
                        f"{label}__widened",
                        "positive",
                        outcome,
                        fit,
                        "ate",
                        "widened",
                    ),
                    grid.Cell(
                        "plain_interval",
                        f"{label}__plain",
                        "diagnostic",
                        outcome,
                        fit,
                        "ate",
                        "plain",
                    ),
                    grid.Cell(
                        "residual_witness",
                        f"{label}__witness",
                        "positive" if fit in WITNESS_GATED else "diagnostic",
                        outcome,
                        fit,
                        "ate",
                        "witness",
                    ),
                ]
        cells.append(
            grid.Cell(
                "relative_reference",
                f"{outcome}_rel_newton__widened",
                "diagnostic",
                outcome,
                "rel_newton",
                "ate",
                "widened",
            )
        )
    cells.append(
        grid.Cell(
            "step_bound",
            "binary_ss_newton_step_bound__widened",
            "diagnostic",
            "binary",
            "ss_newton_step_bound",
            "ate",
            "widened",
        )
    )
    k8 = f"sv_one_step_k{laws.NECESSITY_SCALE:g}"
    cells += [
        grid.Cell("residual_necessity", "ln_ey1__widened", "positive", "LN", k8, "ey1", "widened"),
        grid.Cell("residual_necessity", "ln_ey1__plain", "control", "LN", k8, "ey1", "plain"),
        grid.Cell(
            "necessity_reference",
            "ln_ey1_k1__widened",
            "diagnostic",
            "LN",
            "sv_one_step_k1",
            "ey1",
            "widened",
        ),
        grid.Cell(
            "necessity_reference",
            "ln_ey1_k1__plain",
            "diagnostic",
            "LN",
            "sv_one_step_k1",
            "ey1",
            "plain",
        ),
        grid.Cell(
            "necessity_reference",
            "ln_ey1_rel__widened",
            "diagnostic",
            "LN",
            "rel_one_step",
            "ey1",
            "widened",
        ),
    ]
    for rule in laws.RULES:
        fit = _label(rule, "one_step")
        for parameter in ("ey1", "ate"):
            for reading in ("widened", "plain"):
                cells.append(
                    grid.Cell(
                        "contrast_residual",
                        f"l2_{fit}_{parameter}__{reading}",
                        "diagnostic",
                        "L2",
                        fit,
                        parameter,
                        reading,
                    )
                )
    return tuple(cells)


CELLS = _cells()
PROPERTY_CELLS = grid.property_cells(CELLS)
CALIBRATION_LABELS = grid.calibration_labels(CELLS)

#: The families that state no verdict.
REPORTED_FAMILIES = (
    "plain_interval",
    "relative_reference",
    "step_bound",
    "necessity_reference",
    "contrast_residual",
)

#: The primary phase: the one-step ``scaled_var`` fit of each L1 law, ATE and EY1.
PRIMARY_FIT = "sv_one_step"
SCENARIOS: dict[str, tuple[str, ...]] = dict.fromkeys(OUTCOMES, ("ate", "ey1"))

STUDY = StudyRecord(
    name="stopping rules on point TMLE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "option_stopping_point",
    document="docs/technical-reference/method-evidence/option-stopping-point.md",
    anchor="option-stopping-point",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    failure_bound={"cleverly": FAILURE_BOUND},
    modules=(
        "tests/studies/canonical_option_stopping_point.py",
        "tests/studies/option_stopping_point_properties.py",
        "tests/studies/_option_stopping_cells.py",
        "tests/studies/_option_stopping_laws.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/repeated_sizing.py",
        "src/cleverly/fluctuation/_score.py",
        "src/cleverly/fluctuation/iterative.py",
        "src/cleverly/inference/influence.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_option_stopping_point",
    properties_module="tests.studies.option_stopping_point_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "in-sample point TMLE under a declared stopping rule, widened interval",
    "laws": "L1 (canonical-tmle binary and continuous), L2 (contrast residual), LN (necessity)",
    "rules": list(laws.RULES),
    "targetings": list(laws.TARGETINGS),
    "max_iter": laws.MAX_ITER,
    "one_step_max_iter": laws.ONE_STEP_MAX_ITER,
    "tol": laws.TOL,
    "necessity_threshold_scale": laws.NECESSITY_SCALE,
    "step_bound": STEP_BOUND,
    "witness_floor_se": grid.WITNESS_FLOOR,
    "outcome_learner": "unpenalized logistic (binary), least squares (continuous, L2), least "
    "squares on A, W1, W2 without V (LN)",
    "treatment_learner": "unpenalized logistic regression on every covariate",
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return STREAMS[scenario].draw(n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def primary_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """One replication's rows: the widened interval of each primary parameter."""
    rows = []
    for name in SCENARIOS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    frame, truth = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = STREAMS[scenario].fits[PRIMARY_FIT](frame)
    return primary_rows(result, truth, scenario, replicate, n)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The exact truth of every primary estimand, by scenario."""
    return {
        scenario: {name: TRUTHS[scenario][name] for name in SCENARIOS[scenario]}
        for scenario in SCENARIOS
    }


__all__ = [
    "CELLS",
    "PROPERTY_CELLS",
    "STREAMS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
]
