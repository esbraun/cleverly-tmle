"""Registered evidence for complete-data DR-TMLE on a declared known treatment mechanism.

The subject is ``DRTMLE`` on data that declares its treatment mechanism.  This is Benkeser,
Carone, van der Laan and Gilbert (2017), Theorem 1, at the degenerate estimator
``g_n = g0``.  Under the ``"Q"`` guard the fit still fluctuates the mechanism along
``Q_r / g`` (the algorithm's step 6), so the targeted ``g*`` moves off the declaration on
purpose, as R ``drtmle`` does with a supplied ``gn``.

**The comparator** is R ``drtmle`` 1.1.2 at ``538a3a2`` with ``gn = list(1 - g0, g0)``,
through ``tests/canonical/known_mechanism_drtmle/run_study.R``.  Both sides read this
package's initial outcome regression as ``Qn``, fit the reduced regressions with the same
families (``glm_Qr = "gn"``, a linear regression on the mechanism, and ``glm_gr = "Qn"``, a
logistic regression on the outcome regression), and fit in sample (``cvFolds = 1``).
``tolg = 0.1`` lies below every declared value, and the runner refuses a sample where it
would bind.

**The scenarios** are the four guards, ``guard_none``, ``guard_q``, ``guard_g`` and
``guard_qg``, each reporting ``ey0``, ``ey1`` and ``ate`` with the wrong outcome regression of
:mod:`tests.studies.known_mechanism_law`.  All four read the same samples.  A fifth,
``three_arm_guard_none``, fits the three-arm law at ``guard=()`` and reports each arm mean
and each contrast against arm 0.  At ``guard=()`` the fit is the plain TMLE at the declared
mechanism, so this scenario is the external comparator of the multi-arm known path.  R
``drtmle`` takes the three levels in ``a_0`` with one ``gn`` per level, as
``canonical_multi_arm_mar_drtmle`` does.

**The targeting witness.**  Each implementation's mean ``|estimate - initial|`` of the
contrast must clear :data:`TARGETING_WITNESS_FLOOR` in every scenario, so the pairing
compares two targeting steps that moved, not two plug-ins.

**Rejected candidates.**  The plan named a ``tmle3`` ``LF_known`` multi-arm pairing.  It was
replaced by the three-arm ``drtmle`` scenario above, which needs no new image and reaches the
same estimator.

Publication policy is ``reporting``, with the red-cell route of
:mod:`tests.studies.canonical_known_mechanism` (owner ``X15-known-mechanism``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import known_mechanism_law as law
from tests.studies.canonical_known_mechanism import (
    COVARIATES,
    G_BOUNDS,
    PROBABILITIES,
    DeclaredOnly,
    OutcomeGLM,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import initial_estimates, primary_rows

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261151
RESAMPLING_SEED = 2026115101
IMPLEMENTATION = "cleverly-known-mechanism-drtmle"
REFERENCE = "drtmle-r-known-g"
ESTIMANDS = ("ey0", "ey1", "ate")
#: Scenario -> the guard both sides solve.
GUARDS: dict[str, tuple[str, ...]] = {
    "guard_none": (),
    "guard_q": ("Q",),
    "guard_g": ("g",),
    "guard_qg": ("Q", "g"),
}
OWNER = "guard_qg"
#: The three-arm scenario and its guard: the plain TMLE at the declared mechanism.
THREE_ARM = "three_arm_guard_none"
THREE_ARM_ESTIMANDS = ("ey[0.0]", "ey[1.0]", "ey[2.0]", "ate[1.0 vs 0.0]", "ate[2.0 vs 0.0]")
#: Scenario -> (guard, arm count).
SCENARIOS: dict[str, tuple[tuple[str, ...], int]] = {
    **{scenario: (guard, 2) for scenario, guard in GUARDS.items()},
    THREE_ARM: ((), 3),
}
MAX_OUTER = 100
#: The floor of each implementation's mean ``|estimate - initial|`` of the contrast, in
#: every scenario.  A 20-replicate smoke run moved ``ate`` by about 0.075 at each guard and
#: ``ate[2.0 vs 0.0]`` by about 0.018 at three arms, on both sides.
TARGETING_WITNESS_FLOOR = 0.005

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "known_mechanism_accuracy": tuple(
        f"ate__drtmle_known__{name}" for name in ("none", "Q", "g", "Qg")
    ),
}

STUDY = StudyRecord(
    name="DR-TMLE on a declared known treatment mechanism",
    slug="known-treatment-mechanism-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "known_mechanism_drtmle",
    document="docs/technical-reference/method-evidence/known-treatment-mechanism.md",
    anchor="dr-tmle-on-a-declared-known-treatment-mechanism",
    scenarios={**dict.fromkeys(GUARDS, ESTIMANDS), THREE_ARM: THREE_ARM_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={scenario: OWNER for scenario in GUARDS if scenario != OWNER},
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    modules=(
        "tests/studies/canonical_known_mechanism_drtmle.py",
        "tests/studies/known_mechanism_drtmle_properties.py",
        "tests/studies/canonical_known_mechanism.py",
        "tests/studies/known_mechanism_properties.py",
        "tests/studies/known_mechanism_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/drtmle/Dockerfile",
        "tests/canonical/known_mechanism_drtmle/run_study.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_known_mechanism_drtmle",
    properties_module="tests.studies.known_mechanism_drtmle_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "drtmle(Qn = list(qn0, qn1), gn = list(1 - g0, g0), glm_Qr = 'gn', glm_gr = 'Qn', "
        "reduction = 'univariate', maxIter = 100, tolIC = 1e-8, tolg = 0.1, cvFolds = 1); Qn is "
        "this package's initial outcome regression; the three-arm scenario passes a_0 = 0:2 "
        "and one Qn and gn per level"
    ),
    "rejected_candidates": (
        "tmle3 LF_known multi-arm pairing: replaced by the three-arm drtmle scenario at "
        "guard=(), which is the plain TMLE at the declared mechanism and needs no new image"
    ),
    "targeting_witness_floor": TARGETING_WITNESS_FLOOR,
}

CONFIGURATION = {
    "construction": "complete-data DR-TMLE dividing by the declared mechanism",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "treatment_mechanism": "known (declared by column)",
    "outcome_learner": "unpenalized logistic regression on A, W1, W3 (omits W2)",
    "reduced_outcome_learner": "linear regression on the mechanism",
    "reduced_treatment_learner": "logistic regression on the outcome regression",
    "reduction": "univariate",
    "max_outer": MAX_OUTER,
}


def fit_cleverly(
    frame: pd.DataFrame, guard: tuple[str, ...], *, outcome: str = "wrong", arms: int = 2
) -> Any:
    """Fit this study's DR-TMLE at one guard on one sample."""
    return (
        DRTMLE(
            guard=guard,
            reduction="univariate",
            outcome_learner=OutcomeGLM(outcome, arms),
            treatment_learner=DeclaredOnly(),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=2_000),
            cross_fit=False,
            estimands=ESTIMANDS if arms == 2 else ("ey", "ate"),
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_outer=MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_probabilities=PROBABILITIES[arms],
        )
        .single()
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIOS:
        raise KeyError(scenario)
    arms = SCENARIOS[scenario][1]
    truth = law.truth(arms)
    return law.sample(n, seed, arms=arms), {name: truth[name] for name in STUDY.scenarios[scenario]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    return primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=STUDY.scenarios[scenario],
        initials=(
            _three_arm_initials(result)
            if scenario == THREE_ARM
            else initial_estimates(result, STUDY.scenarios[scenario])
        ),
    )


def _three_arm_initials(result: Any) -> dict[str, float]:
    """The untargeted arm means and contrasts, as the R runner forms them from ``Qn``."""
    means = {code: float(np.mean(result.nuisance.outcome.arms[float(code)])) for code in range(3)}
    out = {f"ey[{float(code)}]": value for code, value in means.items()}
    for code in (1, 2):
        out[f"ate[{float(code)} vs 0.0]"] = means[code] - means[0]
    return out


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    guard, arms = SCENARIOS[scenario]
    result = fit_cleverly(frame, guard, arms=arms)
    sample = frame.loc[:, ["W1", "W2", "W3", "A", "Y"]].copy()
    for code in range(3):
        sample[f"qn{code}"] = (
            np.asarray(result.nuisance.outcome.arms[float(code)], dtype=float)
            if code < arms
            else np.nan
        )
    for code in range(3):
        sample[f"gn{code}"] = frame[f"p{code}"].to_numpy(dtype=float) if code < arms else np.nan
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, cleverly_rows(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((scenario, replicate, n),) for scenario in SCENARIOS for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]


def targeting_displacement(rows: pd.DataFrame) -> dict[str, float]:
    """Mean ``|estimate - initial_estimate|`` of the contrast, by implementation and scenario.

    The nonzero-targeting witness of ``docs/development/method-benchmarking.md``: each value
    must clear :data:`TARGETING_WITNESS_FLOOR`.  The contrast is ``ate`` at two arms and
    ``ate[2.0 vs 0.0]`` at three.
    """
    contrast = dict.fromkeys(GUARDS, "ate") | {THREE_ARM: "ate[2.0 vs 0.0]"}
    out: dict[str, float] = {}
    for (implementation, scenario), group in rows.groupby(["implementation", "scenario"]):
        selected = group.loc[group["estimand"] == contrast[str(scenario)]]
        out[f"{implementation} {scenario}"] = float(
            np.mean(np.abs(selected["estimate"] - selected["initial_estimate"]))
        )
    return out
