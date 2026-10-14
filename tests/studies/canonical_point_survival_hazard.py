"""Registered evidence for per-horizon hazard targeting (``point-survival-hazard``).

The subject is ``LTMLE`` with ``survival_model="hazard"`` and ``targeting="iterative"`` on a
held design: one baseline treatment held over every node, read from one time and one event
column per unit (``TimeToEvent``).  Each reported horizon targets the discrete cause-specific
hazards of every regimen and cause on its own (Moore and van der Laan 2009, Section 4.1;
:mod:`cleverly.longitudinal.hazard`).  The laws are X13's ``survival`` (S1) and ``competing``
(S2) scenarios of :mod:`tests.studies.point_survival_common`, so every truth is an exact finite
sum and every efficiency bound is exact.

**The comparator** is ``survtmle`` 1.1.1 with ``method="hazard"`` (Benkeser, Carone and Gilbert
2018), on the same realized samples, through ``tests/canonical/point_survival_hazard_runner.R``
on the X13 image:

* the treatment model is the logistic regression on ``W1 + W2`` on both sides;
* the censoring model is X13's: one logistic regression on ``trt + W1 + W2`` per visit;
* the hazard model is the logistic regression on ``trt + W1 + W2`` per visit and cause.  Here it
  is one unpenalized fit per node on the design ``[W1, W2, A]``, and there one pooled ``glm``
  with one intercept and three slopes per visit and no shared term, so the two are one model.
  A later cause regresses on the rows with no event of an earlier cause at the visit on both
  sides (``survtmle``'s ``hazards_estimate.R``).

``survtmle`` runs with ``tol = 1 / n`` and ``maxIter = 100``, with ``gtol = 1e-3``.  Its
exported default ``tol = 1 / sqrt(n)`` is 0.022 at n = 2,000, and the initial
``max |meanIC|`` of the declared learners is at most 0.0096 on these laws, so at the default
the targeting loop never runs.  At ``1 / n`` it runs in 87 to 100 percent of fits.  The
initial plug-in is a second call with ``tol = Inf``, whose targeting loop never runs.  The runner counts the "Fluctuation
convergence failure" warnings per replicate and records the final ``meanIC``.

Declared unmatched settings: ``survtmle`` fluctuates one cause at a time with a logistic
submodel scaled by :math:`1 - \\lambda_{\\neg j}`, and this package fluctuates every cause
jointly in the multinomial logistic submodel; ``gtol = 1e-3``; the ``l_j``/``u_j`` bound
machinery, inactive at ``bounds = NULL``; and the stopping rule (``survtmle`` stops at
``max |meanIC| <= tol``, this package at a relative score of ``1e-10``).  A ``survtmle`` fit
that reaches ``maxIter`` with ``max |meanIC| > 1/n`` stays in the paired rows.  The A2 smoke
had 4 such fits in 105, with a residual of at most three times ``tol``, which moved one
estimate by 0.10 SD.  The run log records each one as ``converged=FALSE``.

**The scenarios.**  ``survival`` (five integer visits; both arms' risks and their difference at
visits 1, 3 and 5) and ``competing`` (four visits, two causes; the incidences and their
differences at visits 2 and 4), each at n = 2,000 with 1,600 replications.  The property cells
are in :mod:`tests.studies.point_survival_hazard_properties`.

Publication policy is ``reporting``.  The red-cell route was declared before any run.  A band
cell goes to ``band-finite-sample`` only if the shortfall-reading diagnostic of
``tests/unit/test_band_shortfall_reading.py`` reproduces for it, and otherwise to
``X14-finite-sample``.  A paired row whose reading names a comparator convention above goes to
``X14-comparator-conventions``, which closes when a re-declared paired cell passes.  Any other
red cell goes to ``X14-finite-sample``, which closes when a re-declared cell passes.  No re-run
uses more replicates, wider margins or a new law.  A replication that raises is never redrawn.
A recordable fit failure, on either side, is recorded and costs its replicate under a zero
failure bound, so its cells read ``bound exceeded``.  Such a cell goes to ``X14-fit-failures``,
with its failure count stated as a page limit, which closes when a re-declared run records no
failure.  The reference fits one replication's two scenarios as one group, so a survtmle failure
in either scenario costs the reference both.  A fatal error or a systematic failure stops the
run.  A fit whose targeting stops short is data: the manifest counts it and the evidence page
reports it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import point_survival_common as common
from tests.studies.canonical_ltmle import R_BASE_IMAGE, regimen_rows
from tests.studies.canonical_point_survival import (
    COMPETING,
    REFERENCE,
    REGIMENS,
    REPORTED,
    SCENARIOS,
    SURVIVAL,
    SURVTMLE_TARBALL_SHA256,
    SURVTMLE_VERSION,
    TRUTH,
)
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261070
RESAMPLING_SEED = 2026107001

#: The targeting settings of every fit of this study.
ALGORITHM = "iterative"
MAX_ITER = 100
TOL = 1e-10

#: The ``survtmle`` settings the runner passes.  ``maxIter = 100`` is ``hazard_tmle``'s default;
#: ``survtmle()`` exports 10.  ``gtol = 1e-3`` is the exported default.
SURVTMLE_MAX_ITER = 100
SURVTMLE_GTOL = 1e-3


def _names(scenario: str) -> tuple[str, ...]:
    law = SCENARIOS[scenario]
    causes: tuple[str | None, ...] = (
        tuple(common.CAUSES[c] for c in (1, 2)) if law.causes == 2 else (None,)
    )
    levels = tuple(
        common.risk_name(label, horizon, cause)
        for label in REGIMENS
        for horizon in REPORTED[scenario]
        for cause in causes
    )
    contrasts = tuple(
        common.contrast_name("arm1", REFERENCE, horizon, cause)
        for horizon in REPORTED[scenario]
        for cause in causes
    )
    return levels + contrasts


ESTIMANDS = {scenario: _names(scenario) for scenario in SCENARIOS}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        "survival_t5__correctly_specified",
        "survival_t5__shrunken_se_control",
        "survival_t5__noise_control",
        "competing_t4__correctly_specified",
    ),
    "double_robustness": tuple(
        f"survival_t5__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "targeting_necessity": ("survival_t5__targeted", "survival_t5__untargeted"),
    "root_n_and_efficiency": tuple(f"survival_t5__n_{size}" for size in (1_000, 2_000, 8_000)),
    "root_n_rate": ("survival_t5__empirical_sd", "survival_t5__reported_se"),
    "power": ("survival_t5__alternative",),
    "simultaneous_coverage": (
        "all_reported__simultaneous_band",
        "all_reported__pointwise_joint_control",
    ),
}

STUDY = StudyRecord(
    name="point-treatment survival, per-horizon hazard targeting",
    slug="point-survival-hazard",
    artifacts=ROOT / "tests" / "canonical" / "point_survival_hazard",
    document="docs/technical-reference/method-evidence/point-survival-hazard.md",
    anchor="point-survival-hazard",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="survtmle",
    modules=(
        "tests/studies/canonical_point_survival_hazard.py",
        "tests/studies/point_survival_hazard_properties.py",
        "tests/studies/canonical_point_survival.py",
        "tests/studies/point_survival_properties.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/canonical_ltmle.py",
        "src/cleverly/datasets/survival_point.py",
        "src/cleverly/longitudinal/hazard.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/survtmle/Dockerfile",
        "tests/canonical/point_survival_hazard_runner.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_point_survival_hazard",
    properties_module="tests.studies.point_survival_hazard_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "survtmle_version": SURVTMLE_VERSION,
    "survtmle_tarball_sha256": SURVTMLE_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "survtmle(method='hazard') per horizon t0 with tol=1/n, maxIter=100 and gtol=1e-3; "
        "glm.trt='W1 + W2'; glm.ftime and glm.ctime with one intercept and "
        "three slopes per visit and no shared term; the initial estimate is a second call with "
        "tol=Inf"
    ),
    "pairing": (
        "read under the default margins; survtmle fluctuates one cause at a time and stops at "
        "max |meanIC| <= 1/n or at maxIter = 100; a fit stopped at maxIter above 1/n stays "
        "in the paired rows"
    ),
}

CONFIGURATION = {
    "construction": "held point-treatment survival, hazard targeting per horizon",
    "survival_model": "hazard",
    "targeting": ALGORITHM,
    "cross_fit": False,
    "outcome_learner": "unpenalized logistic regression on W1, W2, A per node and cause",
    "treatment_learner": "unpenalized logistic regression on W1, W2",
    "censoring_learner": "unpenalized logistic regression on W1, W2, A per node",
    "regimens": list(REGIMENS),
    "reported_nodes": {scenario: list(nodes) for scenario, nodes in REPORTED.items()},
}


def primary_learners() -> dict[str, Any]:
    """The glm learners both sides fit."""
    return {
        "outcome_learner": LogisticRegression(C=np.inf, max_iter=1000),
        "treatment_learner": LogisticRegression(C=np.inf, max_iter=1000),
        "censoring_learner": LogisticRegression(C=np.inf, max_iter=1000),
    }


def fit(scenario: str, frame: pd.DataFrame, *, targeting: str = ALGORITHM) -> Any:
    """The subject's primary fit of one scenario's sample."""
    law = SCENARIOS[scenario]
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        n_folds=1,
        simultaneous=False,
        survival_model="hazard",
        targeting=targeting,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=0,
        **primary_learners(),
    ).fit(law.container(frame))


def initial_estimates(result: Any, names: Sequence[str]) -> dict[str, float]:
    """The plug-in of every reported level and contrast at the initial hazards.

    Each fit's targeting record holds the initial plug-in of every target it solved, in
    ``targets`` order, so a per-horizon and a one-step fit read it the same way.
    """
    index = result.parameter_index or {}
    plug_in: dict[tuple[str, str | None, int], float] = {}
    for fit_record in result.fits.values():
        record = fit_record.hazard
        for target, value in zip(record.targets, record.initial_levels, strict=True):
            plug_in[target] = float(value)
    out: dict[str, float] = {}
    for name in names:
        label, cause, horizon = index[name]
        if " vs " in label:
            left, right = label.split(" vs ")
            out[name] = plug_in[(left, cause, horizon)] - plug_in[(right, cause, horizon)]
        else:
            out[name] = plug_in[(label, cause, horizon)]
    return out


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return SCENARIOS[scenario].draw(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _reference_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """The replicate's sample and truth rows, which need no fit, and no subject rows."""
    scenario, replicate, _ = key
    frame, truth = drawn
    sample = frame.loc[:, ["W1", "W2", "A", "time", "event"]].copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, []


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    result = fit(scenario, frame)
    rows = regimen_rows(
        STUDY,
        result,
        truth,
        initial_estimates(result, ESTIMANDS[scenario]),
        ESTIMANDS[scenario],
        scenario,
        replicate,
        n=len(frame),
    )
    return sample, truth_rows, rows


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of both scenarios, its truth rows and the subject's rows.

    The keys are ordered by replication and then by scenario, so the reference reads one
    replication's two samples as one contiguous group and splits them by scenario.  A sample
    needs no fit, so the reference fits a replicate whose subject fit failed.
    """
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand, by scenario."""
    return TRUTH


__all__ = ["COMPETING", "STUDY", "SURVIVAL", "TRUTH", "draw_and_fit", "fit"]
