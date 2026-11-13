r"""Declared evidence for a binding ``isotonic=True`` projection (``isotonic-binding``).

The subject is ``LTMLE(isotonic=True)`` with ``survival_model="iterated_mean"`` on a two-node
survival law whose projection binds (X76 part (g), Fix-8 plan sections 2.7, 3.8 and 5.8).  The
projection moves a level by :math:`D^-/2` and keeps its standard error, so the claim under test
is the interval centred at the projected level.  Law, learners and quadrature truths are
:mod:`tests.studies._isotonic_binding_law`; the property families are
:mod:`tests.studies.isotonic_binding_properties`:

========================  ==============================================================
family                    cells
========================  ==============================================================
``interval_calibration``  ``projected_t1``: the never regimen's horizon-1 interval at the
                          projected level, with the derived shrunken-SE and noise controls
``binding_witness``       the never horizon-1 level moved by more than 1e-6 SE
``material_move``         the never horizon-1 level moved by at least 0.1 SE
``reported_levels``       the unprojected horizon-1 interval, horizon 2, the always
                          regimen, the SE shares
``reported_large_move``   the never horizon-1 level on the larger-move law ``LG2-gs2``,
                          projected and unprojected
========================  ==============================================================

The moves arise because the horizon-1 node-1 regression is the logistic outcome learner and
the horizon-2 one is the linear pseudo learner (plan 2.7).  The logistic learner is solved to
``tol=1e-13``, and the design test shows that nothing gated changes at ``1e-10``.

**Primary.**  The never regimen's horizon-1 risk at the projected level on ``LG2``.

**No comparator.**  ``lmtp``'s ``isotonic_projection`` is the transcription the unit tests
compare against (``tests/unit/test_isotonic_projection.py``), and it moves the same level; it
is not a reference for the truth.  The study reads every cell against its quadrature truth.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/isotonic_binding/regenerate.py`` before any run.

**Registered** from its committed run, with the ``G31.isotonic`` evidence, cells and limit and
the evidence page ``isotonic-binding.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _isotonic_binding_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "isotonic-binding"
SEED = 20267741
RESAMPLING_SEED = 20267742
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the never regimen's projected horizon-1 risk on ``LG2`` at n = 2,000.
PRIMARY_SCENARIO = law.LAW
PRIMARY_N = 2_000
PRIMARY_REPLICATES = 1_600
PRIMARY_ESTIMANDS = (law.ESTIMAND,)

#: The calibration label and the cells of every family.  Filled here rather than from
#: :mod:`tests.studies.isotonic_binding_properties`, which this module cannot import; the
#: design test pins the two equal.
CALIBRATION_LABELS = ("projected_t1",)
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        "projected_t1__correctly_specified",
        "projected_t1__shrunken_se_control",
        "projected_t1__noise_control",
    ),
    "binding_witness": ("never_t1",),
    "material_move": ("never_t1",),
    "reported_levels": (
        "unprojected_t1",
        "projected_t2",
        "unprojected_t2",
        "se_present_t2",
        "always_projected_t1",
        "always_binding_t1",
        "always_projected_t2",
        "always_se_present_t2",
        "stalled_fits",
    ),
    "reported_large_move": ("projected_t1", "unprojected_t1", "binding_t1", "stalled_fits"),
}

STUDY = StudyRecord(
    name="Binding isotonic projection of a survival curve",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "isotonic_binding",
    document="docs/technical-reference/method-evidence/isotonic-binding.md",
    anchor="binding-isotonic-projection-of-a-survival-curve",
    scenarios={PRIMARY_SCENARIO: PRIMARY_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/studies/canonical_isotonic_binding.py",
        "tests/studies/isotonic_binding_properties.py",
        "tests/studies/_isotonic_binding_law.py",
        "src/cleverly/longitudinal/estimator.py",
        "src/cleverly/longitudinal/sequential.py",
        "src/cleverly/longitudinal/hazard.py",
        "src/cleverly/inference/influence.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_isotonic_binding",
    properties_module="tests.studies.isotonic_binding_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "in-sample LTMLE, survival_model='iterated_mean', isotonic=True",
    "law": "W ~ N(0, 1); A1 ~ Bern(expit(gs W)); C1 ~ Bern(expit(2.5 - 0.5 W)); "
    "Y1 ~ Bern(expit(-1.5 + 1.6 W - 0.5 A1)); L2 ~ Bern(expit(-0.3 + 0.6 W + 0.4 A1)); "
    "A2 ~ Bern(expit(gs W + 0.8 L2 + 0.5 A1)); C2 ~ Bern(expit(2.0 - 0.5 W + 0.5 L2)); "
    "Y2 ~ Bern(0.01 expit(-0.5 + 1.2 W^2 - 0.4 A2)); gs = 1.5 (LG2), 2.0 (LG2-gs2, reported)",
    "regimens": dict(law.REGIMENS),
    "truths": dict(law.TRUTH),
    "outcome_learner": "LogisticRegression(C=inf, max_iter=100000, tol=1e-13)",
    "pseudo_learner": "LinearRegression",
    "mechanism_learners": "LogisticRegression(C=inf, max_iter=100000, tol=1e-13)",
    "g_bounds": list(law.G_BOUNDS),
    "max_iter": law.MAX_ITER,
    "tol": law.TOL,
    "n_folds": 1,
    "simultaneous_intervals": False,
    "discarded_law": "LG (the two-node discrete survival law with binary W): its node "
    "regressions are saturated, so the projection cannot bind once the node-1 score is "
    "solved; its pilot binding came from sklearn's default tol 1e-4",
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(n, seed), {law.ESTIMAND: law.TRUTH[law.ESTIMAND]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Fit one primary sample and convert it to the shared replication schema."""
    result = law.fit(frame)
    estimate = result.estimates[law.ESTIMAND]
    low, high = (float(value) for value in estimate.ci)
    reference = float(truth[law.ESTIMAND])
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": len(frame),
            "estimand": law.ESTIMAND,
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": law.initial_level(result),
        }
    ]


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(PRIMARY_SCENARIO, replicate, n) for replicate in range(replicates)],
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
    """The quadrature truth of every primary estimand, by scenario."""
    return {PRIMARY_SCENARIO: {law.ESTIMAND: law.TRUTH[law.ESTIMAND]}}


__all__ = [
    "CALIBRATION_LABELS",
    "PROPERTY_CELLS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
