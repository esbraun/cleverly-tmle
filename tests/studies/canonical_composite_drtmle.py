"""Registered evidence study for the composite-indicator missing-data DR-TMLE.

Three scenarios, each a law of :mod:`tests.studies.mar_arm_indexed_laws` with every table
indexed ``[w, a]``:

=====================================  ====  ==========================  ======================
scenario                               arms  missing                     composite
=====================================  ====  ==========================  ======================
``binary_observational_mar``           2     the outcome                 ``g pi``, min 0.12
``binary_mar_outcome_and_treatment``   2     the outcome and treatment   ``g pi_A pi``, min 0.0825
``three_arm_mar_outcome_and_treatment`` 3    the outcome and treatment   ``g pi_A pi``, min 0.07
=====================================  ====  ==========================  ======================

The treatment is observational in every scenario: its table ``g`` is estimated by the oracle
learner and never passed as known probabilities.  ``P(Delta_A = 1 | A = a, W = w)``
(:data:`PI_TREATMENT_TWO`, :data:`PI_TREATMENT_THREE`) depends on both arguments, so the
recording of the treatment depends on the treatment, and ``P(A | W)`` is not identified.

The comparator is R ``drtmle`` 1.1.2, which implements this construction: an ``NA`` treatment
gives ``DeltaA = 0``, and its fluctuation indicator is ``A == a & DeltaA == 1 & DeltaY == 1``.
``out$drtmle`` pairs with the composite DR-TMLE.  ``out$tmle`` is one logistic fluctuation
along ``C_a / g_c``, which on a binary outcome is the maximum-likelihood fit the composite
TMLE reaches, so its contrast pairs with the composite TMLE under the names
:data:`TMLE_NAMES`.

**Declared before any verdict:**

* budget: 800 primary replications per scenario at n = 2,000; the property budgets are the
  constants of :mod:`tests.studies.composite_drtmle_properties`.  Measured single-thread cost,
  on a loaded machine: R ``drtmle`` 23 s per fit, so about 15.5 CPU-hours for the 2,400
  reference fits; the Python primaries about 1 CPU-hour; the property families about 8
  CPU-hours (0.2 to 4.3 s per fit by cell).  About 25 CPU-hours in all.  A cost up to twice
  that is accepted, and it is a cost tolerance, not a margin;
* seeds: 20262301 for the samples and 20262302 for resampling;
* red cells: ``publication_policy="reporting"``.  A red verdict stays red at its registered
  budget and margin.  Raising a budget, moving a margin, and re-declaring a law, a learner or
  a size after a verdict are refused.  A red positive cell is first diagnosed as a possible
  defect; a defect is fixed and the study regenerated once, last.  Without a defect, the cell
  gets a red-cells ledger row and an owner row.  A red control means the
  instrument cannot discriminate, and it is reported, not re-declared.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
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
SEED = 20262301

#: ``P(Delta_A = 1 | A = a, W = w)`` of the two-arm scenario, columns ``(0, 1)``.
PI_TREATMENT_TWO = np.array([[0.55, 0.70], [0.85, 0.90], [0.95, 0.90]])
#: The same for three arms, columns in :data:`laws.THREE_ARM_LABELS` order (low, mid, high).
PI_TREATMENT_THREE = np.array([[0.55, 0.60, 0.50], [0.80, 0.75, 0.85], [0.95, 0.90, 0.95]])

OBSERVATIONAL = "binary_observational_mar"
BINARY = "binary_mar_outcome_and_treatment"
THREE_ARM = "three_arm_mar_outcome_and_treatment"

LAWS: dict[str, laws.Law] = {
    OBSERVATIONAL: dataclasses.replace(laws.LAWS["l1"], scenario=OBSERVATIONAL),
    BINARY: dataclasses.replace(laws.LAWS["l1"], scenario=BINARY, pi_treatment=PI_TREATMENT_TWO),
    THREE_ARM: dataclasses.replace(
        laws.LAWS["l3"], scenario=THREE_ARM, pi_treatment=PI_TREATMENT_THREE
    ),
}
#: Each scenario's law-module key, for its engine request and its reported names.
LAW_KEYS = {OBSERVATIONAL: "l1", BINARY: "l1", THREE_ARM: "l3"}

#: The composite TMLE's contrast, published beside the DR-TMLE's names and paired with
#: R ``drtmle``'s ``out$tmle``.  The names carry no bracket, so they read as plain keys.
TMLE_NAMES = {BINARY: ("tmle_ate", "ate"), THREE_ARM: ("tmle_ate_mid", "ate[mid vs high]")}


def estimands(scenario: str) -> tuple[str, ...]:
    """The names a scenario reports: the DR-TMLE's, then the composite TMLE's contrast."""
    names = laws.ESTIMANDS[LAW_KEYS[scenario]]
    if scenario in TMLE_NAMES:
        return (*names, TMLE_NAMES[scenario][0])
    return names


G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01
MAX_OUTER = 100
N_MULTIPLIER = 1_000

STUDY = StudyRecord(
    name="observational missing-data DR-TMLE with a composite indicator",
    slug="composite-missing-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_composite",
    document="docs/technical-reference/method-evidence/observational-missing-data-dr-tmle.md",
    anchor="observational-missing-data-dr-tmle",
    scenarios={scenario: estimands(scenario) for scenario in LAWS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    nuisance_count=4,
    resampling_seed=20262302,
    margins=Margins(),
    implementation="cleverly-composite-drtmle",
    reference="drtmle-r-composite",
    publication_policy="reporting",
    extra_artifacts=("fit-exits.csv",),
    modules=(
        "tests/studies/canonical_composite_drtmle.py",
        "tests/studies/composite_drtmle_properties.py",
        "tests/studies/mar_arm_indexed_laws.py",
        "tests/discrete_law_mar.py",
        "tests/discrete_law_multi.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_composite_drtmle",
    properties_module="tests.studies.composite_drtmle_properties",
    property_cells={
        "ordinary_targeting": ("binary", "three_arm", "binary_regime", "binary_msm"),
        # The three-arm ``low`` contrast's both-wrong control runs under a drift of its own;
        # the properties module says why.
        "corrected_mar_inference": tuple(
            f"{prefix}__{configuration}"
            for prefix in (
                "composite_observational_ate",
                "composite_binary_ate",
                "composite_three_arm_ate_low",
                "composite_three_arm_ate_mid",
            )
            for configuration in ("both_correct", "outcome_drift", "mechanism_drift", "both_wrong")
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ate__correctly_specified",
            "ate__shrunken_se_control",
            "ate__noise_control",
        ),
        "simultaneous_coverage": tuple(
            f"{label}__{cell}"
            for label in ("composite_observational", "composite_binary", "composite_three_arm")
            for cell in ("simultaneous_band", "pointwise_joint_control")
        ),
        "type_i_error": ("sharp_null",),
        "power": ("ate",),
        "correction_necessity": (
            "composite_cycle__closed_score",
            "composite_cycle__initial_score_control",
        ),
        "treatment_complete_case": ("drop_unrecorded__control",),
    },
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "comparator_boundary": (
        "the same composite construction: NA treatment read as DeltaA = 0, oracle Qn and the "
        "oracle composite gn supplied, so estimateG is skipped; out$tmle pairs with the "
        "composite TMLE on a binary outcome"
    ),
}

CONFIGURATION = {
    "construction": "composite indicator Delta_A * Delta * 1{A = a}, armwise at every arm count",
    "cross_fit": False,
    "simultaneous_intervals": True,
    "n_multiplier": N_MULTIPLIER,
    "treatment_mechanism": "estimated (oracle table on the recorded rows); never known",
    "g_bounds": list(G_BOUNDS),
    "missingness_bound": NUISANCE_BOUND,
    "composite_bounds_floor": G_BOUNDS[0] * NUISANCE_BOUND * NUISANCE_BOUND,
    "guard": ["Q", "g"],
    "reduction": "univariate",
    "reduced_learners": "LinearRegression, LogisticRegression(C=1e6)",
    "max_outer": MAX_OUTER,
    "tmle_pairing": "guard=() composite TMLE against out$tmle",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    law = LAWS[scenario]
    truths = laws.truths(law)
    if scenario in TMLE_NAMES:
        name, source = TMLE_NAMES[scenario]
        truths[name] = truths[source]
    return laws.sample(law, n, seed), truths


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(
    frame: pd.DataFrame,
    law: laws.Law,
    *,
    mu: np.ndarray | None = None,
    g: np.ndarray | None = None,
    recorded: np.ndarray | None = None,
    guard: tuple[str, ...] = ("Q", "g"),
    request: tuple[Any, ...] | None = None,
    simultaneous: bool = True,
    interventions: Any = None,
    msm: Any = None,
) -> Any:
    """One fit with law-table primaries.

    ``mu``, ``g`` and ``recorded`` declare a misspecified outcome regression, conditional
    treatment table ``P(A | Delta_A = 1, W)`` and treatment observation vector.
    ``guard=()`` fits the composite TMLE, which alone takes ``interventions=`` and ``msm=``.
    """
    settings: dict[str, Any] = {
        "cross_fit": False,
        "outcome_learner": laws.LawOutcome(law, mu=mu),
        "treatment_learner": laws.LawTreatment(law, g=g),
        "missingness_learner": laws.LawResponse(law, recorded=recorded),
        "estimands": laws.REQUESTS[_key(law)] if request is None else request,
        "simultaneous": simultaneous,
        "n_multiplier": N_MULTIPLIER,
        "g_bounds": G_BOUNDS,
        "nuisance_bound": NUISANCE_BOUND,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    if interventions is not None:
        settings["interventions"] = interventions
    if msm is not None:
        settings["msm"] = msm
    estimator: Any
    if guard:
        estimator = DRTMLE(
            guard=guard,
            reduction=str(CONFIGURATION["reduction"]),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=2_000),
            max_outer=MAX_OUTER,
            **settings,
        )
    else:
        estimator = TMLE(**settings)
    roles: dict[str, Any] = {"delta": "Delta"}
    if law.pi_treatment is not None:
        roles["treatment_delta"] = "DeltaA"
    return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], **roles).single()


def fit_exit(result: Any) -> dict[str, Any]:
    """How the DR-TMLE outer loop of ``result`` ended, and after how many rounds.

    A composite TMLE fit has no outer loop, so it records ``"none"`` and 0.  A capped fit
    counts as it is when its scores pass; the record lets a reader find the capped fits.
    """
    reduction = getattr(result.repeats[0].fluctuations.get("mean"), "reduction", None)
    if reduction is None:
        return {"exit_reason": "none", "rounds": 0}
    return {"exit_reason": str(reduction.exit_reason), "rounds": int(reduction.rounds)}


def _key(law: laws.Law) -> str:
    return "l1" if law.arms == 2 else "l3"


def initial_estimates(result: Any, law: laws.Law) -> dict[str, float]:
    """Each reported estimand's untargeted plug-in, on its natural scale."""
    nuisance = result.nuisance
    psi = np.empty(law.arms)
    for code in result.data.arm_codes:
        values = nuisance.scaler.unscale_levels(nuisance.outcome.arms[code])
        psi[law.column(int(code))] = float(np.mean(np.asarray(values, dtype=float)))
    return laws.values(dataclasses.replace(law, key=_key(law)), psi)


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    law = LAWS[scenario]
    result = fit_cleverly(frame, law)
    rows = primary_rows(
        result=result,
        truth=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=laws.ESTIMANDS[LAW_KEYS[scenario]],
        initials=initial_estimates(result, law),
    )
    rows = [{**row, **fit_exit(result)} for row in rows]
    if scenario in TMLE_NAMES:
        name, source = TMLE_NAMES[scenario]
        tmle = fit_cleverly(frame, law, guard=(), request=("ate",), simultaneous=False)
        initials = initial_estimates(tmle, law)
        (row,) = primary_rows(
            result=tmle,
            truth={source: reference[name]},
            implementation=STUDY.implementation,
            scenario=scenario,
            replicate=replicate,
            estimands=(source,),
            initials={source: initials[source]},
        )
        rows.append({**row, "estimand": name, **fit_exit(tmle)})
    return rows


#: The columns of ``fit-exits.csv``: each package fit's outer-loop exit, by estimand.
FIT_EXIT_COLUMNS = ("scenario", "replicate", "estimand", "exit_reason", "rounds")


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``fit-exits.csv``: the exit reason and round count of every package primary fit."""
    package = rows.loc[rows["implementation"].eq(STUDY.implementation), list(FIT_EXIT_COLUMNS)]
    return {
        "fit-exits.csv": package.sort_values(
            ["scenario", "replicate", "estimand"], ignore_index=True
        )
    }


def comparator_columns(frame: pd.DataFrame, law: laws.Law) -> pd.DataFrame:
    """The oracle columns R ``drtmle`` receives, in arm-code order.

    ``A_code`` is the arm code, ``NaN`` where the treatment is unrecorded.  ``qn<k>`` is the
    outcome regression of arm code ``k`` and ``gn<k>`` the composite mechanism
    ``P(A = k, Delta_A = 1, Delta = 1 | W)``.
    """
    w = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    out = frame.copy()
    out["A_code"] = frame["A"].map({label: code for code, label in enumerate(law.codes)})
    if "DeltaA" not in out:
        out["DeltaA"] = 1.0
    composite = law.composite
    for code in range(law.arms):
        column = law.column(code)
        out[f"qn{code}"] = law.mu[w, column]
        out[f"gn{code}"] = composite[w, column]
    return out


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    sample = comparator_columns(frame, LAWS[scenario])
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
        [((scenario, replicate, n),) for scenario in LAWS for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    # The exit columns ride along for :func:`extra_artifacts`; the driver drops them after it.
    return samples, truth_rows, estimates.loc[:, [*REPLICATE_COLUMNS, "exit_reason", "rounds"]]
