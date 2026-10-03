"""Registered evidence study for randomized multi-arm missing-outcome DR-TMLE.

The law is L3 of :mod:`tests.studies.mar_arm_indexed_laws`, reused unchanged: three arms with
labels ``low``, ``mid`` and ``high``, a binary outcome, and MAR outcomes. Its treatment table
``multi.G`` is read as a known W-stratified randomization and passed to the fit by mapping.
The labels sort to ``high``, ``low``, ``mid``, so ``high`` is arm code 0 and the reference arm.

The comparator is R ``drtmle`` 1.1.2 at the both-correct limit. It fluctuates one joint
treatment-response mechanism with the composite response ``A == a & DeltaY == 1``. That is
the construction Díaz and van der Laan (2017, p. 25) reject for this problem. The pairing is
evidence of the shared limit only.

**Declared before any verdict** (F4 plan, Sections 3.4 and 3.5):

* budget: 800 primary replications at n = 2,000; the property budgets are the constants of
  :mod:`tests.studies.multi_arm_mar_drtmle_properties`. Measured single-thread cost per fit:
  1.8 s at n = 500, 2.2 s at n = 2,000 (2.5 to 2.9 s under drift) and 3.5 s at n = 8,000, so
  about 10 CPU-hours in all. A cost up to twice that is accepted, and it is a cost tolerance,
  not a margin;
* seeds: 20261010 for the samples and 20261011 for resampling;
* red cells: ``publication_policy="reporting"``. A red verdict stays red at its registered
  budget and margin. Raising a budget, moving a margin, and re-declaring a law, a learner or a
  size after a verdict are refused. A red positive cell is diagnosed first as a possible defect
  against the K-arm code; a defect is fixed and the study regenerated once, last. Without a
  defect, the cell gets a red-cells ledger row and an owner row. A red control means the
  instrument cannot discriminate, and it is reported, not re-declared.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20261010
SCENARIO = "three_arm_mar_randomized"
LAW = dataclasses.replace(laws.LAWS["l3"], scenario=SCENARIO)
#: The engine request and the nine names it reports on a binary outcome.
REQUEST = laws.REQUESTS["l3"]
ESTIMANDS = laws.ESTIMANDS["l3"]
G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01
MAX_OUTER = 100
N_MULTIPLIER = 1_000

STUDY = StudyRecord(
    name="randomized multi-arm missing-outcome DR-TMLE",
    slug="multi-arm-mar-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_mar_multi_arm",
    document=(
        "docs/technical-reference/method-evidence/randomized-multi-arm-missing-outcome-dr-tmle.md"
    ),
    anchor="randomized-multi-arm-missing-outcome-dr-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    nuisance_count=3,
    resampling_seed=20261011,
    margins=Margins(),
    implementation="cleverly-multi-arm-mar-drtmle",
    reference="drtmle-r-multi-arm-mar",
    publication_policy="reporting",
    modules=(
        "tests/studies/canonical_multi_arm_mar_drtmle.py",
        "tests/studies/multi_arm_mar_drtmle_properties.py",
        "tests/studies/mar_arm_indexed_laws.py",
        "tests/discrete_law_multi.py",
        "tests/discrete_law_mar.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_multi_arm_mar_drtmle",
    properties_module="tests.studies.multi_arm_mar_drtmle_properties",
    property_cells={
        # ``l3_ate_low`` and ``l3_ate_mid`` are the arm labels of the two contrasts, as the
        # stacked arm-indexed study names them (``mar_arm_indexed_laws.label``).
        "corrected_mar_inference": tuple(
            f"{contrast}__{configuration}"
            for contrast in ("l3_ate_low", "l3_ate_mid")
            for configuration in (
                "both_correct",
                "outcome_drift",
                "observation_drift",
                "both_wrong",
            )
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ate__correctly_specified",
            "ate__shrunken_se_control",
            "ate__noise_control",
        ),
        "type_i_error": ("l3_ate_mid__randomized_sharp_null",),
        "power": ("l3_ate_mid__randomized_alternative",),
        "correction_necessity": (
            "five_reduction_cycle__closed_score",
            "five_reduction_cycle__initial_score_control",
        ),
    },
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "comparator_boundary": (
        "joint treatment-response mechanism (Diaz and van der Laan 2017, p. 25); "
        "both-correct limit only"
    ),
}

CONFIGURATION = {
    "construction": "Diaz and van der Laan randomized MAR DR-TMLE, armwise at three arms",
    "cross_fit": False,
    "simultaneous_intervals": True,
    "n_multiplier": N_MULTIPLIER,
    "known_treatment_probabilities": "W-stratified multi.G, passed by level",
    "g_bounds": list(G_BOUNDS),
    "missingness_bound": NUISANCE_BOUND,
    "guard": ["Q", "g"],
    "reduction": "univariate",
    "fitted_reduction": "missing_outcome",
    "max_outer": MAX_OUTER,
}


def known_probabilities(frame: pd.DataFrame, law: laws.Law = LAW) -> dict[str, np.ndarray]:
    """The design probabilities of each row, keyed by treatment level."""
    w = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    return {label: law.g[w, column] for column, label in enumerate(law.labels)}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return laws.sample(LAW, n, seed), laws.truths(LAW)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    law: laws.Law = LAW,
    mu: np.ndarray | None = None,
    pi: np.ndarray | None = None,
    request: tuple[Any, ...] = REQUEST,
    simultaneous: bool = True,
) -> Any:
    """One fit with law-table primaries; ``mu`` and ``pi`` declare a misspecified table."""
    return (
        DRTMLE(
            randomized=True,
            cross_fit=False,
            outcome_learner=laws.LawOutcome(law, mu=mu),
            treatment_learner=laws.LawTreatment(law),
            missingness_learner=laws.LawResponse(law, pi=pi),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=2_000),
            estimands=request,
            simultaneous=simultaneous,
            n_multiplier=N_MULTIPLIER,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            max_outer=MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            guard=tuple(CONFIGURATION["guard"]),
            reduction=str(CONFIGURATION["reduction"]),
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_probabilities=known_probabilities(frame, law),
        )
        .single()
    )


def initial_estimates(result: Any) -> dict[str, float]:
    """Each reported estimand's untargeted plug-in, on its natural scale."""
    nuisance = result.nuisance
    psi = np.empty(LAW.arms)
    for code in result.data.arm_codes:
        values = nuisance.scaler.unscale_levels(nuisance.outcome.arms[code])
        psi[LAW.column(int(code))] = float(np.mean(np.asarray(values, dtype=float)))
    return laws.values(LAW, psi)


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame)
    return primary_rows(
        result=result,
        truth=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result),
    )


def comparator_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The oracle columns R ``drtmle`` receives, in arm-code order.

    ``A_code`` is the arm code, ``qn<k>`` the outcome regression of arm code ``k`` and
    ``gn<k>`` the joint mechanism ``P(A = k, Delta = 1 | W) = g_k(W) pi_k(W)``.
    """
    w = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    out = frame.copy()
    out["A_code"] = frame["A"].map({label: code for code, label in enumerate(LAW.codes)})
    for code in range(LAW.arms):
        column = LAW.column(code)
        out[f"qn{code}"] = LAW.mu[w, column]
        out[f"gn{code}"] = LAW.g[w, column] * LAW.pi[w, column]
    return out


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    sample = comparator_columns(frame)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference.items()
    ]
    return sample, truth_rows, cleverly_rows(frame, reference, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]
