"""Declared evidence for a working model over the hazard curve (``hazard-msm``).

The subject is ``LTMLE(msm=MSM(...), survival_model="hazard", targeting="one_step")`` on a held
design: one baseline treatment held over every visit, read from one time and one event column
per unit (``TimeToEvent``).  The walk targets the coefficient scores of a marginal structural
working model over the ``(regimen, visit)`` cells of the risk curve, and each cause's targeted
cell plug-ins are projected onto the model (:func:`cleverly.longitudinal.hazard.target_hazard_msm`;
``docs/technical-reference/longitudinal-tmle.md``, "A working model over the hazard curve").

**The laws** are the study-local ``survival`` (five visits, one cause) and ``competing`` (four
visits, two causes) of :mod:`tests.studies.hazard_msm_law`, which carry every gated family, and
``overlap``, which carries only the reported ``iterated_mean_difference`` family.  They are finite, so every truth is
the exact projection of the law's cell risks and every efficiency bound an exact enumeration of
the coefficients' efficient influence function.  The module docstring there gives the reason
the study does not read the shared point-survival laws: their hazard has no ``a x W1`` term, so
the wrong hazard learner's plug-in coefficient of ``a_W1`` sits 0.07 standard errors from the
truth and no necessity or robustness cell could fail.

**Why two laws** (plan X70 rev 2.4).  On the gated laws overlap is good, and the hazard fit and
the iterated-mean fit of the same working model agree: with correctly specified parametric
learners both are efficient, so their difference is :math:`o_P(n^{-1/2})`.  The ``overlap`` law
keeps the same ``survival`` hazard under a treatment mechanism whose smallest probability is
0.055.  On it the two fits differ by up to about 8% of an efficiency SE (200-draw pilot paired
ratios 0.028, 0.077, 0.035 and 0.054); both are efficient, so they cannot differ by more than
:math:`o_P(n^{-1/2})` on a law where both nuisance fits are correct.  The same two fits with
misspecified regressions on the gated law's ``mechanism_correct`` draws differ by 4 to 7%.  A
Wald SE shortfall on the ``overlap`` law is shared by the two fits and small: 0.92 for the
intercept in both on the 200-draw pilot, and 0.94 (hazard) and 0.95 (iterated mean) for the arm
coefficient in an 800-draw planning run.  It is reported, not gated.

**The working model** is :math:`\\operatorname{logit} m = \\beta_0 + \\beta_1 a + \\beta_2 t +
\\beta_3 a W_1` over the cells ``(a1, t)`` and ``(a0, t)``, ``t = 1..T``, with uniform cell
weights; each cause of the competing law gets its own projection.  The primary rows report
every coefficient of each scenario.  One calibration cell fits the same design under the
identity link (:mod:`tests.studies.hazard_msm_properties`).

**The learners** are parametric, correctly specified and not saturated.  The hazard learner is
the unpenalised logistic regression on ``(W1, W2, A, A W1)`` at each visit and cause
(:class:`~tests.studies.hazard_msm_law.InteractionLogistic`), which contains every pseudo-hazard
of both laws, so targeting moves every fit.  The treatment learner is the logistic regression on
``(W1, W2, W1 W2)`` and the censoring learner the logistic regression on ``(W1, W2, A)`` at each
visit.  A saturated mechanism is not used: with the empirical treatment and retention shares of
every stratum of a finite ``W``, the targeted coefficients equal the nonparametric
maximum-likelihood projection whatever the hazard learner, so no cell could see the hazard fit
or the walk (``tests/unit/test_hazard_msm_study_design.py`` keeps the witness).

**No comparator.**  No maintained package targets the coefficient scores of a working model over
discrete-time hazards.  ``survtmle`` and MOSS target cells, ``ltmle`` and ``lmtp`` fit the
iterated-mean recursion, and ``ltmle``'s working-model path projects iterated-mean regimen fits
(``docs/development/method-benchmarking.md`` lists each).  The record names no reference, and
the run writes the empty ``equivalence.csv``.  The independent evidence that replaces the
comparison is the exact truths and bounds, the property families, and the exact-law tests of
``tests/unit/test_hazard_msm.py``.

Publication policy is ``gated``.  The red-cell readings are declared in
``tests/canonical/hazard_msm/regenerate.py`` before any run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import hazard_msm_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: PROVISIONAL.  A placeholder until the calibration pilot is committed: the shared primary rule
#: (``repeated_sizing.primary_replicates``, read by the design test) then sets this count from
#: the ``survival_logit`` and ``competing_logit`` calibration pilots, in a declaration commit
#: with no skipped design test, before any declared run.
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261117
RESAMPLING_SEED = 2026111701

#: The targeting settings of every hazard fit of this study.
ALGORITHM = "one_step"
TOL = 1e-10
#: The primary working model's link.
LINK = "logit"

#: Scenario -> its law.
SCENARIOS: dict[str, law.Law] = {"survival": law.SURVIVAL, "competing": law.COMPETING}


def coefficient_names(scenario: str) -> tuple[str, ...]:
    """Every coefficient one scenario's fit reports, cause by cause and term by term."""
    current = SCENARIOS[scenario]
    return tuple(
        law.coefficient_name(term, current.cause_label(cause))
        for cause in current.causes
        for term in law.TERMS
    )


def truths(scenario: str, link: str = LINK) -> dict[str, float]:
    """The exact coefficients of one scenario, by reported name."""
    current = SCENARIOS[scenario]
    out: dict[str, float] = {}
    for cause in current.causes:
        beta = law.exact_beta(current, cause, link)
        for term, value in zip(law.TERMS, beta, strict=True):
            out[law.coefficient_name(term, current.cause_label(cause))] = float(value)
    return out


ESTIMANDS = {scenario: coefficient_names(scenario) for scenario in SCENARIOS}
TRUTH = {scenario: truths(scenario) for scenario in SCENARIOS}


def _cells(prefix: str, *suffixes: str) -> tuple[str, ...]:
    return tuple(f"{prefix}__{suffix}" for suffix in suffixes)


def _term_key(term: str) -> str:
    return "intercept" if term == "(intercept)" else term


#: Each overlap-law coefficient's cell label, read only by the reported family.
OVERLAP_LABELS = tuple(f"overlap_logit_{_term_key(term)}" for term in law.TERMS)
#: Each survival-law coefficient's cell label.
SURVIVAL_LABELS = tuple(f"survival_logit_{_term_key(term)}" for term in law.TERMS)
#: Each competing-law coefficient's cell label.
COMPETING_LABELS = tuple(
    f"competing_logit_{_term_key(term)}_{law.CAUSES[cause]}"
    for cause in law.COMPETING.causes
    for term in law.TERMS
)
#: The coefficient every necessity, robustness and power cell reads.
MODIFIER_LABEL = "survival_logit_a_W1"
IDENTITY_LABEL = "survival_identity_a_W1"
#: The competing-law death coefficient of the cross-cause double-robustness cell.
COMPETING_DR_LABEL = "competing_logit_a_W1_death"
JOINT_LABEL = "survival_logit_coefficients"
RATE_SIZES = (1_000, 2_000, 8_000)

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        *(
            cell
            for label in SURVIVAL_LABELS
            for cell in _cells(label, "correctly_specified", "shrunken_se_control", "noise_control")
        ),
        *_cells(IDENTITY_LABEL, "correctly_specified"),
        *(cell for label in COMPETING_LABELS for cell in _cells(label, "correctly_specified")),
    ),
    "double_robustness": (
        *_cells(
            MODIFIER_LABEL, "both_correct", "outcome_correct", "mechanism_correct", "both_wrong"
        ),
        *_cells(COMPETING_DR_LABEL, "mechanism_correct"),
    ),
    "targeting_necessity": _cells(MODIFIER_LABEL, "targeted", "untargeted"),
    "root_n_and_efficiency": tuple(
        cell
        for label in SURVIVAL_LABELS
        for cell in _cells(label, *(f"n_{size}" for size in RATE_SIZES))
    ),
    "root_n_rate": tuple(
        cell for label in SURVIVAL_LABELS for cell in _cells(label, "empirical_sd", "reported_se")
    ),
    "power": _cells(MODIFIER_LABEL, "alternative"),
    "simultaneous_coverage": _cells(JOINT_LABEL, "simultaneous_band", "pointwise_joint_control"),
    "iterated_mean_difference": (
        *(
            cell
            for label in OVERLAP_LABELS
            for cell in _cells(label, "paired_difference", "hazard_fit", "iterated_fit")
        ),
        "overlap_logit__iterated_curve_decrease",
        *(
            cell
            for label in SURVIVAL_LABELS
            for cell in _cells(label, "mechanism_correct_difference")
        ),
    ),
}

STUDY = StudyRecord(
    name="working model over the hazard curve, one-step targeting",
    slug="hazard-msm",
    artifacts=ROOT / "tests" / "canonical" / "hazard_msm",
    document="docs/technical-reference/method-evidence/hazard-msm.md",
    anchor="hazard-msm",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.01},
    modules=(
        "tests/studies/canonical_hazard_msm.py",
        "tests/studies/hazard_msm_properties.py",
        "tests/studies/hazard_msm_law.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/longitudinal/hazard.py",
        "src/cleverly/msm.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_hazard_msm",
    properties_module="tests.studies.hazard_msm_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": "held point-treatment survival, one-step hazard walk on working-model scores",
    "survival_model": "hazard",
    "targeting": ALGORITHM,
    "tol": TOL,
    "cross_fit": False,
    "working_model": "logit m = b0 + b1 a + b2 t + b3 a W1 over (regimen, visit), uniform h",
    "terms": list(law.TERMS),
    "link": LINK,
    "outcome_learner": "unpenalized logistic regression on W1, W2, A, A W1 per visit and cause",
    "treatment_learner": "unpenalized logistic regression on W1, W2, W1 W2",
    "censoring_learner": "unpenalized logistic regression on W1, W2, A per visit",
    "regimens": dict(law.REGIMENS),
}


def learners() -> dict[str, Any]:
    """The correct learners of every primary fit."""
    return {
        "outcome_learner": law.InteractionLogistic(),
        "treatment_learner": law.TreatmentLogistic(),
        "censoring_learner": law.CensoringLogistic(),
    }


def fit(
    scenario: str,
    frame: pd.DataFrame,
    *,
    link: str = LINK,
    simultaneous: bool = False,
    learner_set: Mapping[str, Any] | None = None,
) -> Any:
    """The subject's fit of one scenario's sample under the declared working model."""
    return LTMLE(
        law.REGIMENS,
        n_folds=1,
        simultaneous=simultaneous,
        survival_model="hazard",
        targeting=ALGORITHM,
        msm=law.working_model(link),
        tol=TOL,
        random_state=0,
        **dict(learners() if learner_set is None else learner_set),
    ).fit(law.LAWS[scenario].container(frame))


def initial_estimates(result: Any, names: Sequence[str]) -> dict[str, float]:
    """The coefficients projected from the initial hazards, by reported name.

    The walk record, shared by every cause, holds them in ``(term, cause)`` order.
    """
    out: dict[str, float] = {}
    for msm_fit in result.msm_fits:
        record = msm_fit.hazard
        for (term, cause), value in zip(record.targets, record.initial_levels, strict=True):
            if cause == msm_fit.cause:
                out[law.coefficient_name(term, cause)] = float(value)
    return {name: out[name] for name in names}


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


def subject_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """One replication's primary rows, each coefficient on its own (identity) scale."""
    initials = initial_estimates(result, ESTIMANDS[scenario])
    rows: list[dict[str, Any]] = []
    for name in ESTIMANDS[scenario]:
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
                "initial_estimate": initials[name],
            }
        )
    return rows


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    result = fit(scenario, frame)
    return sample, truth_rows, subject_rows(result, truth, scenario, replicate, len(frame))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of both scenarios, its truth rows and the subject's rows."""
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
    frames = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=frames
    )


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand, by scenario."""
    return TRUTH


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit"]
