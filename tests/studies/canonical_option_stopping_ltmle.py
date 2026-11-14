r"""Declared evidence for the stopping rules of LTMLE (``option-stopping-ltmle``, X76 (a)).

The subject is ``LTMLE(stopping="scaled_var")`` and ``LTMLE(stopping="sample_size")`` on two
laws of :mod:`tests.studies._option_stopping_laws`: ``L3``, the ``canonical-ltmle`` censored
two-node law with that study's registered calibrated configuration (a Newton walk at each
node), and ``L4``, the ``point-survival-curve`` five-visit survival law under the one-step
hazard walk with that study's learners.  Under either rule each walk stops before its score
equation is solved, and the reported interval adds the remaining score to its half-width
(``ParameterEstimate.rule_widened``; Fix-8 plan section 2.1).  The property families
(:mod:`tests.studies.option_stopping_ltmle_properties`):

==========================  ======================================================================
family                      cells
==========================  ======================================================================
``interval_calibration``    the widened interval of L3's ``ate_regimen[always vs never]`` and of
(K)                         L4's ``ate_regimen[arm1 vs arm0 @ t=5]`` under each rule: 4 gated
                            positives, each with the derived controls
``residual_witness`` (W)    the share of draws whose residual is at least 0.01 SE: L4's
                            ``scaled_var`` cell gates (lower limit at least
                            :data:`WITNESS_SHARE`); the other three report
``residual_necessity`` (N)  L4 with the ``mechanism_correct`` learners, ``scaled_var`` with every
                            threshold times 8, ``risk_regimen[arm1 @ t=5]``: the plain interval's
                            upper limit below :data:`NECESSITY_CEILING` (control) and the
                            widened interval's lower limit at least :data:`NECESSITY_FLOOR`
                            (positive), on the same fits
reported                    the plain interval of every K cell; the arm-1 risk at t = 5 under
                            each rule; the relative fits on the same draws; the N fit's contrast
                            at t = 5, and the ``mechanism_correct`` fit at threshold times 1
==========================  ======================================================================

Every cell publishes its rule-met share and the distribution of its absolute residual.

On L3 the per-node Newton walk lands at the default fit: its pilot residual is about
7.6e-11 SE under either rule, and its coverage and SE ratio equal the relative fit's.  The L3
cells therefore show no rule effect, and they license only the Newton walk under a rule, not
a one-step end-of-study walk.  The design test pins that equality.

**No comparator**, for the reason ``option-stopping-point`` gives.  **Publication policy** is
``reporting``; the red-cell reading is declared in
``tests/canonical/option_stopping_ltmle/regenerate.py`` before any run.  **Registered**: the
declared run is committed at 674c4f1, and the record backs the ``G55`` support rows.
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
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "option-stopping-ltmle"
SEED = 20267631
RESAMPLING_SEED = 20267632
FAILURE_BOUND = 0.01
PRIMARY_N = laws.N

#: The replications of every property cell.  Plan 5.2's floor is the smallest multiple of 400
#: whose product of single-cell calibration pass probabilities under perfect calibration is at
#: least 0.95 (2,800).  The arm-pooled predictive rule
#: (:func:`tests.studies.repeated_sizing.calibration_replicates`) at the committed 2,000-draw
#: pilot raises it to this count: the cells of one law read the same draws (pilot correlation
#: of their estimates at least 0.998), so the pilot holds one SE-ratio reading per law, and the
#: heterogeneity bound between the laws' readings is wide.  The design test recomputes it.
REPLICATES = 6_800
#: The primary phase's replications: plan 5.2's floor, above the 1,000 a primary needs.
PRIMARY_REPLICATES = 2_800

WITNESS_SHARE = 0.5
NECESSITY_CEILING = 0.92
NECESSITY_FLOOR = 0.90

#: The fits whose W cell gates (plan 5.2).
WITNESS_GATED = ("L4", "sv_one_step")

_K8 = f"k{laws.NECESSITY_SCALE:g}"

STREAMS: dict[str, grid.Stream] = {
    "L3": grid.Stream(
        "L3",
        laws.draw_l3,
        {
            f"{laws.RULE_LABELS[rule]}_newton": functools.partial(laws.fit_l3, stopping=rule)
            for rule in ("relative", *laws.RULES)
        },
        laws.longitudinal_rule_met,
    ),
    "L4": grid.Stream(
        "L4",
        laws.draw_l4,
        {
            **{
                f"{laws.RULE_LABELS[rule]}_one_step": functools.partial(laws.fit_l4, stopping=rule)
                for rule in ("relative", *laws.RULES)
            },
            **{
                f"mc_sv_one_step_k{scale:g}": functools.partial(
                    laws.fit_l4,
                    stopping="scaled_var",
                    configuration="mechanism_correct",
                    scale=scale,
                )
                for scale in (laws.NECESSITY_SCALE, 1.0)
            },
        },
        laws.longitudinal_rule_met,
    ),
}

#: Each stream's contrast.
CONTRASTS = {"L3": laws.L3_CONTRAST, "L4": laws.L4_CONTRAST}


def _cells() -> tuple[grid.Cell, ...]:
    cells: list[grid.Cell] = []
    for stream, walk in (("L3", "newton"), ("L4", "one_step")):
        contrast = CONTRASTS[stream]
        for rule in laws.RULES:
            fit = f"{laws.RULE_LABELS[rule]}_{walk}"
            label = f"{stream.lower()}_{fit}"
            gated = (stream, fit) == WITNESS_GATED
            cells += [
                grid.Cell(
                    "interval_calibration",
                    f"{label}__widened",
                    "positive",
                    stream,
                    fit,
                    contrast,
                    "widened",
                ),
                grid.Cell(
                    "plain_interval",
                    f"{label}__plain",
                    "diagnostic",
                    stream,
                    fit,
                    contrast,
                    "plain",
                ),
                grid.Cell(
                    "residual_witness",
                    f"{label}__witness",
                    "positive" if gated else "diagnostic",
                    stream,
                    fit,
                    contrast,
                    "witness",
                ),
            ]
            if stream == "L4":
                cells.append(
                    grid.Cell(
                        "reported_calibration",
                        f"{label}_risk__widened",
                        "diagnostic",
                        stream,
                        fit,
                        laws.L4_RISK,
                        "widened",
                    )
                )
        cells.append(
            grid.Cell(
                "relative_reference",
                f"{stream.lower()}_rel_{walk}__widened",
                "diagnostic",
                stream,
                f"rel_{walk}",
                contrast,
                "widened",
            )
        )
    k8 = f"mc_sv_one_step_{_K8}"
    cells += [
        grid.Cell(
            "residual_necessity",
            "l4_mc_risk__widened",
            "positive",
            "L4",
            k8,
            laws.L4_RISK,
            "widened",
        ),
        grid.Cell(
            "residual_necessity", "l4_mc_risk__plain", "control", "L4", k8, laws.L4_RISK, "plain"
        ),
    ]
    for label, fit, name in (
        ("l4_mc_contrast", k8, laws.L4_CONTRAST),
        ("l4_mc_k1_risk", "mc_sv_one_step_k1", laws.L4_RISK),
    ):
        for reading in ("widened", "plain"):
            cells.append(
                grid.Cell(
                    "necessity_reference",
                    f"{label}__{reading}",
                    "diagnostic",
                    "L4",
                    fit,
                    name,
                    reading,
                )
            )
    return tuple(cells)


CELLS = _cells()
PROPERTY_CELLS = grid.property_cells(CELLS)
CALIBRATION_LABELS = grid.calibration_labels(CELLS)

REPORTED_FAMILIES = (
    "plain_interval",
    "reported_calibration",
    "relative_reference",
    "necessity_reference",
)


@functools.cache
def _truths() -> dict[str, dict[str, float]]:
    return {stream: STREAMS[stream].draw(50, 0)[1] for stream in STREAMS}


def truths() -> dict[str, dict[str, float]]:
    """Each stream's exact truths, which do not depend on the draw."""
    return {stream: dict(values) for stream, values in _truths().items()}


#: The primary phase: L4's one-step ``scaled_var`` fit (contrast and arm-1 risk at t = 5) and
#: L3's ``scaled_var`` fit (its contrast).
PRIMARY_FITS = {"L3": "sv_newton", "L4": "sv_one_step"}
SCENARIOS: dict[str, tuple[str, ...]] = {
    "L3": (laws.L3_CONTRAST,),
    "L4": (laws.L4_CONTRAST, laws.L4_RISK),
}

STUDY = StudyRecord(
    name="Stopping rules on LTMLE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "option_stopping_ltmle",
    document="docs/technical-reference/method-evidence/option-stopping-ltmle.md",
    anchor="stopping-rules-on-ltmle",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    nuisance_count=3,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    failure_bound={"cleverly": FAILURE_BOUND},
    modules=(
        "tests/studies/canonical_option_stopping_ltmle.py",
        "tests/studies/option_stopping_ltmle_properties.py",
        "tests/studies/_option_stopping_cells.py",
        "tests/studies/_option_stopping_laws.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_point_survival_curve.py",
        "tests/studies/canonical_point_survival_hazard.py",
        "tests/studies/point_survival_hazard_properties.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/repeated_sizing.py",
        "src/cleverly/fluctuation/_score.py",
        "src/cleverly/longitudinal/hazard.py",
        "src/cleverly/longitudinal/sequential.py",
        "src/cleverly/inference/influence.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_option_stopping_ltmle",
    properties_module="tests.studies.option_stopping_ltmle_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "LTMLE under a declared stopping rule, widened interval, in sample",
    "laws": "L3 (canonical-ltmle censored two-node), L4 (point-survival-curve survival, hazard)",
    "rules": list(laws.RULES),
    "l3_walk": "Newton per node, canonical-ltmle learners",
    "l4_walk": f"one-step hazard walk, step size {laws.L4_STEP_SIZE}",
    "max_iter": laws.MAX_ITER,
    "one_step_max_iter": laws.ONE_STEP_MAX_ITER,
    "tol": laws.TOL,
    "necessity_threshold_scale": laws.NECESSITY_SCALE,
    "necessity_learners": "point-survival-curve mechanism_correct",
    "witness_floor_se": grid.WITNESS_FLOOR,
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
        result = STREAMS[scenario].fits[PRIMARY_FITS[scenario]](frame)
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
    values = truths()
    return {
        scenario: {name: values[scenario][name] for name in SCENARIOS[scenario]}
        for scenario in SCENARIOS
    }


__all__ = [
    "CELLS",
    "PROPERTY_CELLS",
    "STREAMS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truths",
]
