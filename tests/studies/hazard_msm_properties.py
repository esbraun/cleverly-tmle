"""Property families of ``hazard-msm``.

Every fit is ``LTMLE`` with ``msm=``, ``survival_model="hazard"`` and ``targeting="one_step"`` on
a law of :mod:`tests.studies.hazard_msm_law`, under the declared working model
:math:`(1, a, t, a W_1)` over the ``(regimen, visit)`` cells with uniform weights.  A cell label
names one coefficient of one fit: ``survival_logit_<term>`` on the five-visit law under the logit
link, ``survival_identity_a_W1`` under the identity link, and
``competing_logit_<term>_<cause>`` on the four-visit law with two causes.  The correct hazard
learner is :class:`~tests.studies.hazard_msm_law.InteractionLogistic`, which contains every
pseudo-hazard of both laws and is not saturated, so targeting moves every fit.  The correct
mechanism learners are the logistic regressions
:class:`~tests.studies.hazard_msm_law.TreatmentLogistic` and
:class:`~tests.studies.hazard_msm_law.CensoringLogistic`, also correct and not saturated.
Every truth is the exact projection and every efficiency bound is exact (:func:`~tests.studies.hazard_msm_law.efficiency_sd`).

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``interval_calibration``        each coefficient of ``survival_logit`` (with the shrunken-SE
                                and noise controls), ``survival_identity_a_W1``, and each
                                coefficient of ``competing_logit``; 5,200 fits per law and
                                link at n = 2,000
``double_robustness``           ``survival_logit_a_W1`` in four nuisance configurations, and
                                ``competing_logit_a_W1_death`` with the wrong hazard learner
                                and the correct mechanism; 1,200 at n = 2,000
``targeting_necessity``         ``survival_logit_a_W1``: the targeted coefficient and the
                                coefficient projected from the initial hazards of the same
                                fit, with the wrong hazard learner and the correct mechanism;
                                1,200 at n = 2,000
``root_n_and_efficiency``       each coefficient of ``survival_logit`` at n = 1,000, 2,000 and
                                8,000; 800 each.  The 1,000 rung is a rate rung that
                                carries no claim (it takes the harness's smallest-rung role)
``power``                       ``survival_logit_a_W1`` at n = 2,000; 800
``simultaneous_coverage``       the band over the four ``survival_logit`` coefficients of the
                                calibration fits, and its pointwise control
``iterated_mean_difference``    reported, no verdict.  On the ``overlap`` law, 800 draws at
                                n = 2,000: the iterated-mean fit of the same working model
                                beside the hazard fit, each fit's coverage and SE ratio, and
                                whether the iterated-mean risk curve decreases.  On the
                                ``double_robustness`` ``mechanism_correct`` draws: the paired
                                difference of the two fits with misspecified regressions
==============================  ===========================================================

The wrong hazard learner is :class:`~tests.studies.hazard_msm_law.NoW2Logistic`, a per-visit
logistic regression on ``(W1, A)``.  Its plug-in coefficient of ``a_W1`` is not zero, varies
from sample to sample, and its limit lies more than three efficiency standard errors from the
truth at n = 2,000 (``tests/unit/test_hazard_msm_study_design.py``), so the ``untargeted``
control can fail.  The wrong mechanism is the prior-share classifier, which ignores ``W``.

The ``iterated_mean_difference`` rows state no verdict and read the ``overlap`` law
(``overlap_logit_<term>``).  Both fits are efficient, so where both nuisance fits are correct
they cannot differ by more than :math:`o_P(n^{-1/2})`; on the ``overlap`` law the 200-draw pilot
separates them by 3 to 8% of an efficiency SE (plan X70 rev 2.4).  The ``hazard_fit`` and
``iterated_fit`` rows of a coefficient are the two fits of the same draw against the exact truth,
so their summaries give coverage and SE ratio side by side.  A Wald SE shortfall on this law is
shared by the two fits and small: the 200-draw pilot reads 0.92 for the intercept in both, and a
planning run of 800 draws read 0.94 (hazard) and 0.95 (iterated mean) for the arm coefficient.
It is reported, not gated.  The ``mechanism_correct_difference`` rows read the gated law's
``double_robustness`` ``mechanism_correct`` draws, as :data:`MECHANISM_DIFFERENCE` describes.  Each
``paired_difference`` row records the hazard coefficient minus the iterated-mean coefficient of
the same draw; its summary adds
``paired_sd_over_bound``, the SD of that difference over the efficiency-bound standard error.
The ``iterated_curve_decrease`` row records the largest decrease of the iterated-mean risk curve
of either regimen between consecutive visits, with the standard error of that difference from
the fit's influence curves; ``rejected`` marks a draw whose curve decreases somewhere, so the
row's rejection rate is the share of such draws.  The curve is read from an iterated-mean fit of
the same draw with no working model.

The pilot readings of the diagnostic are read by the design test from the committed pilot and
reported with no floor: the two fits are efficient, so they differ by o_P(n^-1/2), and X70
reports their agreement instead of requiring a gap (the owner's decision of 2026-11-13).  The
share of decreasing iterated-mean curves is reported with its 99%
Clopper-Pearson interval and carries no floor.

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost more than its bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_hazard_msm as study
from tests.studies import hazard_msm_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT_FAMILY
from tests.studies.evidence.simultaneous import (
    joint_cells,
    joint_coverage_rows,
    joint_property_cells,
)
from tests.studies.fractional_glm import QuasiBinomialGLM

STUDY = study.STUDY
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = study.RATE_SIZES
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = 5_200
POWER_REPLICATES = 800
POWER_N = 2_000
DIFFERENCE_REPLICATES = 800
DIFFERENCE_N = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The declared floor on the targeted arm's displacement from the plug-in, in the targeted
#: arm's empirical standard deviations.
TARGETING_DISPLACEMENT = 0.25
CONFIGURATIONS = ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
DIFFERENCE_FAMILY = "iterated_mean_difference"
DECREASE_CELL = "overlap_logit__iterated_curve_decrease"
#: The fit label the reported family reads.
DIFFERENCE_FIT = "overlap_logit"
MODIFIER = study.MODIFIER_LABEL
JOINT_LABEL = study.JOINT_LABEL

#: Fit label -> ``(scenario, link)``.
FITS: dict[str, tuple[str, str]] = {
    "survival_logit": ("survival", "logit"),
    "survival_identity": ("survival", "identity"),
    "competing_logit": ("competing", "logit"),
    "overlap_logit": ("overlap", "logit"),
}


def _coefficients(fit_label: str) -> tuple[tuple[str, str], ...]:
    """``(cell label, reported name)`` of every coefficient a fit's cells read."""
    scenario, _ = FITS[fit_label]
    current = law.LAWS[scenario]
    pairs = []
    for cause in current.causes:
        cause_name = current.cause_label(cause)
        for term in law.TERMS:
            key = "intercept" if term == "(intercept)" else term
            suffix = "" if cause_name is None else f"_{cause_name}"
            pairs.append((f"{fit_label}_{key}{suffix}", law.coefficient_name(term, cause_name)))
    if fit_label == "survival_identity":
        return tuple(pair for pair in pairs if pair[0] == study.IDENTITY_LABEL)
    return tuple(pairs)


COEFFICIENTS = {fit_label: _coefficients(fit_label) for fit_label in FITS}
#: Cell label -> the reported name it reads.
ESTIMAND = {label: name for pairs in COEFFICIENTS.values() for label, name in pairs}
#: Cell label -> the fit label it belongs to.
FIT_OF = {label: fit_label for fit_label, pairs in COEFFICIENTS.items() for label, _ in pairs}
#: The fit labels of the calibration family; ``overlap_logit`` is read only by the reported
#: ``iterated_mean_difference`` family.
CALIBRATION_FITS = ("survival_logit", "survival_identity", "competing_logit")
#: The labels that carry the two derived calibration controls.
CONTROLLED_LABELS = study.SURVIVAL_LABELS


def _exact(fit_label: str) -> tuple[dict[str, float], dict[str, float]]:
    """The exact truth and efficiency bound of each coefficient of one fit, by cell label."""
    scenario, link = FITS[fit_label]
    current = law.LAWS[scenario]
    truth: dict[str, float] = {}
    bound: dict[str, float] = {}
    labels = dict(COEFFICIENTS[fit_label])
    for cause in current.causes:
        beta = law.exact_beta(current, cause, link)
        sd = law.efficiency_sd(current, cause, link)
        for index, term in enumerate(law.TERMS):
            name = law.coefficient_name(term, current.cause_label(cause))
            for label, reported in labels.items():
                if reported == name:
                    truth[label] = float(beta[index])
                    bound[label] = float(sd[index])
    return truth, bound


_EXACT = {fit_label: _exact(fit_label) for fit_label in FITS}
#: Cell label -> the exact coefficient.
TRUTHS = {label: value for truth, _ in _EXACT.values() for label, value in truth.items()}
#: Cell label -> the exact efficiency bound of the coefficient, as a standard deviation.
EFFICIENCY_SD = {label: value for _, bound in _EXACT.values() for label, value in bound.items()}


def learners(configuration: str = "both_correct") -> dict[str, Any]:
    """The hazard and mechanism learners of one nuisance configuration."""
    hazard_correct = configuration in {"both_correct", "outcome_correct"}
    mechanism_correct = configuration in {"both_correct", "mechanism_correct"}
    return {
        "outcome_learner": law.InteractionLogistic() if hazard_correct else law.NoW2Logistic(),
        "treatment_learner": (
            law.TreatmentLogistic() if mechanism_correct else DummyClassifier(strategy="prior")
        ),
        "censoring_learner": (
            law.CensoringLogistic() if mechanism_correct else DummyClassifier(strategy="prior")
        ),
    }


def fit_label(
    label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
) -> Any:
    """The hazard fit of one fit label's sample."""
    scenario, link = FITS[label]
    return study.fit(
        scenario,
        frame,
        link=link,
        simultaneous=simultaneous,
        learner_set=learners(configuration),
    )


class W1QuasiBinomial(QuasiBinomialGLM):
    """The quasibinomial regression on ``W1`` alone: the iterated-mean design's first column.

    It drops ``W2`` as :class:`~tests.studies.hazard_msm_law.NoW2Logistic` does, so the
    iterated-mean fit it serves is misspecified the same way as the wrong hazard fit.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> W1QuasiBinomial:
        super().fit(np.asarray(X, dtype=float)[:, :1], y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(np.asarray(X, dtype=float)[:, :1])

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(np.asarray(X, dtype=float)[:, :1])


def iterated_fit(
    frame: pd.DataFrame,
    *,
    working_model: bool = True,
    scenario: str = "overlap",
    regression_correct: bool = True,
) -> Any:
    """The iterated-mean fit of one law's sample, with or without the working model.

    Each visit's regression is the unpenalised quasibinomial regression on ``W1`` and ``W2``
    within the followers of a regimen, which is the design an iterated-mean fit hands its
    learners on a held design, or on ``W1`` alone when ``regression_correct`` is false
    (:class:`W1QuasiBinomial`).  The mechanism learners are the hazard fit's correct ones.
    """
    regression = QuasiBinomialGLM if regression_correct else W1QuasiBinomial
    return LTMLE(
        law.REGIMENS,
        n_folds=1,
        simultaneous=False,
        survival_model="iterated_mean",
        isotonic=False,
        targeting="iterative",
        msm=law.working_model("logit") if working_model else None,
        max_iter=100,
        tol=study.TOL,
        random_state=0,
        outcome_learner=regression(),
        pseudo_learner=regression(),
        treatment_learner=law.TreatmentLogistic(),
        censoring_learner=law.CensoringLogistic(),
    ).fit(law.LAWS[scenario].container(frame))


def largest_decrease(result: Any) -> tuple[float, float]:
    """The largest decrease of a fit's risk curve between visits, and its standard error.

    The decrease of regimen ``r`` at visit ``t`` is :math:`\\hat F_r(t - 1) - \\hat F_r(t)`, and
    its standard error is the SD of the difference of the two influence curves over
    :math:`\\sqrt n`.  The largest over every regimen and visit is negative when the curve rises
    everywhere.
    """
    best: tuple[float, float] | None = None
    for label in law.REGIMENS:
        for horizon in range(2, law.SURVIVAL.n_times + 1):
            before = result[law.risk_name(label, horizon - 1)]
            after = result[law.risk_name(label, horizon)]
            drop = float(before.psi - after.psi)
            curve = np.asarray(before.influence_curve) - np.asarray(after.influence_curve)
            error = float(np.std(curve) / np.sqrt(curve.size))
            if best is None or drop > best[0]:
                best = (drop, error)
    assert best is not None
    return best


def draw(fit_label_: str, n: int, seed: int) -> pd.DataFrame:
    scenario, _ = FITS[fit_label_]
    return law.LAWS[scenario].draw(n, seed)


def plug_in(result: Any, name: str) -> float:
    """The coefficient projected from the initial hazards of a fit."""
    return study.initial_estimates(result, (name,))[name]


# ------------------------------------------------------------------ the grid

#: The competing-law coefficient of the cross-cause double-robustness cell: a death coefficient,
#: whose clever covariate carries the relapse hazard's term when the hazards are wrong.
COMPETING_DR_LABEL = study.COMPETING_DR_LABEL
#: The configuration and stream of that cell.
COMPETING_DR_STREAM = "competing_mechanism_correct"
#: The base cell of the reported paired difference on the gated law's ``mechanism_correct``
#: draws: the hazard fit with the wrong hazard learner minus the iterated-mean fit with its
#: regressions misspecified the same way (:class:`W1QuasiBinomial`), both with the correct
#: mechanism.  Both are consistent through the correct mechanism.  The 200-draw
#: ``fits_differ_mechanism_correct`` pilot reads paired ratios of 0.04 to 0.07, the same order
#: as on the ``overlap`` law: the two misspecified regressions are close enough that the fits do
#: not separate at first order on this law.
MECHANISM_DIFFERENCE = "mechanism_correct_difference"

#: Every fit set: ``(family, stream label, configuration, n, replicates, fit label)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        (
            "interval_calibration",
            label,
            "both_correct",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            label,
        )
        for label in CALIBRATION_FITS
    ),
    *(
        ("double_robustness", c, c, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, "survival_logit")
        for c in CONFIGURATIONS
    ),
    (
        "double_robustness",
        COMPETING_DR_STREAM,
        "mechanism_correct",
        DOUBLE_ROBUST_N,
        DOUBLE_ROBUST_REPLICATES,
        "competing_logit",
    ),
    (
        "targeting_necessity",
        "wrong_hazard",
        "mechanism_correct",
        NECESSITY_N,
        NECESSITY_REPLICATES,
        "survival_logit",
    ),
    *(
        (
            "root_n_and_efficiency",
            f"n_{size}",
            "both_correct",
            size,
            RATE_REPLICATES,
            "survival_logit",
        )
        for size in RATE_SIZES
    ),
    ("power", "alternative", "both_correct", POWER_N, POWER_REPLICATES, "survival_logit"),
    (
        DIFFERENCE_FAMILY,
        "overlap",
        "both_correct",
        DIFFERENCE_N,
        DIFFERENCE_REPLICATES,
        DIFFERENCE_FIT,
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


class DeclaredCoefficients:
    """The law and working model a coefficient cell reads, with its exact truths.

    Parameters
    ----------
    fit_label : str
        A key of :data:`FITS`.
    """

    def __init__(self, fit_label: str) -> None:
        self.fit_label = fit_label
        self.name = f"hazard_msm__{fit_label}"

    def truth(self) -> dict[str, float]:
        return {ESTIMAND[label]: TRUTHS[label] for label, _ in COEFFICIENTS[self.fit_label]}


class DeclaredDifference:
    """The zero every ``iterated_mean_difference`` row is recorded against."""

    name = "hazard_msm__iterated_mean_difference"

    def truth(self) -> dict[str, float]:
        return {"paired_difference": 0.0, "largest_decrease": 0.0}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        dgp: Any,
        estimand: str,
        n: int,
        reps: int,
        stream: str,
        draws_of: str | None = None,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=dgp,
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(draws_of or family, stream, 0),
                role=role,
                estimand=estimand,
            )
        )

    for fit in CALIBRATION_FITS:
        dgp = DeclaredCoefficients(fit)
        for label, name in COEFFICIENTS[fit]:
            arms = [("correctly_specified", "positive")]
            if label in CONTROLLED_LABELS:
                arms += [("shrunken_se_control", "control"), ("noise_control", "control")]
            for arm, role in arms:
                add(
                    "interval_calibration",
                    f"{label}__{arm}",
                    role,
                    dgp,
                    name,
                    CALIBRATION_N,
                    CALIBRATION_REPLICATES,
                    fit,
                )
    survival = DeclaredCoefficients("survival_logit")
    for configuration in CONFIGURATIONS:
        add(
            "double_robustness",
            f"{MODIFIER}__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            survival,
            ESTIMAND[MODIFIER],
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
        )
    add(
        "double_robustness",
        f"{COMPETING_DR_LABEL}__mechanism_correct",
        "positive",
        DeclaredCoefficients("competing_logit"),
        ESTIMAND[COMPETING_DR_LABEL],
        DOUBLE_ROBUST_N,
        DOUBLE_ROBUST_REPLICATES,
        COMPETING_DR_STREAM,
    )
    # The reported paired difference reads the survival ``mechanism_correct`` draws.
    for label, _ in COEFFICIENTS["survival_logit"]:
        add(
            DIFFERENCE_FAMILY,
            f"{label}__{MECHANISM_DIFFERENCE}",
            DIAGNOSTIC_ROLE,
            DeclaredDifference(),
            "paired_difference",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            "mechanism_correct",
            draws_of="double_robustness",
        )
    for arm, role in (("targeted", "positive"), ("untargeted", "control")):
        add(
            "targeting_necessity",
            f"{MODIFIER}__{arm}",
            role,
            survival,
            ESTIMAND[MODIFIER],
            NECESSITY_N,
            NECESSITY_REPLICATES,
            "wrong_hazard",
        )
    for size in RATE_SIZES:
        for label, name in COEFFICIENTS["survival_logit"]:
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                "control" if size == min(RATE_SIZES) else "positive",
                survival,
                name,
                size,
                RATE_REPLICATES,
                f"n_{size}",
            )
    add(
        "power",
        f"{MODIFIER}__alternative",
        "positive",
        survival,
        ESTIMAND[MODIFIER],
        POWER_N,
        POWER_REPLICATES,
        "alternative",
    )
    cells.extend(
        joint_property_cells(
            JOINT_LABEL,
            n=CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES,
            seed=_seed("interval_calibration", "survival_logit", 0),
        )
    )
    difference = DeclaredDifference()
    overlap = DeclaredCoefficients(DIFFERENCE_FIT)
    for label, name in COEFFICIENTS[DIFFERENCE_FIT]:
        add(
            DIFFERENCE_FAMILY,
            f"{label}__paired_difference",
            DIAGNOSTIC_ROLE,
            difference,
            "paired_difference",
            DIFFERENCE_N,
            DIFFERENCE_REPLICATES,
            "overlap",
        )
        for arm in ("hazard_fit", "iterated_fit"):
            add(
                DIFFERENCE_FAMILY,
                f"{label}__{arm}",
                DIAGNOSTIC_ROLE,
                overlap,
                name,
                DIFFERENCE_N,
                DIFFERENCE_REPLICATES,
                "overlap",
            )
    add(
        DIFFERENCE_FAMILY,
        DECREASE_CELL,
        DIAGNOSTIC_ROLE,
        difference,
        "largest_decrease",
        DIFFERENCE_N,
        DIFFERENCE_REPLICATES,
        "overlap",
    )
    return tuple(cells)


# ------------------------------------------------------------------ rows


#: The calibration positive arm of each controlled label to the controls derived from it.
DERIVED_CELLS = calibration_derived(CONTROLLED_LABELS)

_Payload = tuple[str, str, str, int, int, int, str]
_Key = tuple[_Payload]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    family, stream, _, replicate, n, _, label = key[0]
    return draw(label, n, _seed(family, stream, replicate))


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the fit set's rows and derived controls fill."""
    family, _, configuration, _, n, _, label = key[0]
    if family == "interval_calibration":
        names: list[str] = []
        for cell_label, _ in COEFFICIENTS[label]:
            name = f"{family}/{cell_label}__correctly_specified"
            names += [name, *DERIVED_CELLS.get(name, ())]
        if label == "survival_logit":
            names += [f"{JOINT_FAMILY}/{joint}" for joint in joint_cells(JOINT_LABEL)]
        return names
    if family == "double_robustness":
        if label == "competing_logit":
            return [f"{family}/{COMPETING_DR_LABEL}__{configuration}"]
        names = [f"{family}/{MODIFIER}__{configuration}"]
        if configuration == "mechanism_correct":
            names += [
                f"{DIFFERENCE_FAMILY}/{cell_label}__{MECHANISM_DIFFERENCE}"
                for cell_label, _ in COEFFICIENTS[label]
            ]
        return names
    if family == "targeting_necessity":
        return [f"{family}/{MODIFIER}__targeted", f"{family}/{MODIFIER}__untargeted"]
    if family == "root_n_and_efficiency":
        return [f"{family}/{cell_label}__n_{n}" for cell_label, _ in COEFFICIENTS[label]]
    if family == DIFFERENCE_FAMILY:
        return [
            *(
                f"{family}/{cell_label}__{arm}"
                for cell_label, _ in COEFFICIENTS[label]
                for arm in ("paired_difference", "hazard_fit", "iterated_fit")
            ),
            f"{family}/{DECREASE_CELL}",
        ]
    return [f"{family}/{MODIFIER}__alternative"]


def cell_of(key: _Key) -> str:
    return f"{key[0][0]}/{key[0][1]}"


def replicate_of(key: _Key) -> int:
    return key[0][3]


def fit_set_rows(payload: _Payload) -> list[dict[str, Any]]:
    """The replicate rows of one fit of one fit set, drawn and fitted in one call."""
    return fit_drawn(payload, _draw((payload,)))


def _difference_rows(
    frame: pd.DataFrame, common_row: dict[str, Any], shared: dict[str, Any], label: str
) -> list[dict[str, Any]]:
    scenario, _ = FITS[label]
    hazard = fit_label(label, frame)
    iterated = iterated_fit(frame, scenario=scenario)
    curve = iterated_fit(frame, working_model=False, scenario=scenario)
    rows: list[dict[str, Any]] = []
    for cell_label, name in COEFFICIENTS[label]:
        for arm, result in (("hazard_fit", hazard), ("iterated_fit", iterated)):
            rows.append(
                replicate_row(
                    property_name=DIFFERENCE_FAMILY,
                    cell=f"{cell_label}__{arm}",
                    role=DIAGNOSTIC_ROLE,
                    truth=TRUTHS[cell_label],
                    estimate=result[name],
                    alpha=STUDY.margins.alpha,
                    **shared,
                )
            )
        targeted = hazard[name]
        low, high = targeted.ci
        other = float(iterated[name].psi)
        covered = int(low <= other <= high)
        rows.append(
            {
                "property": DIFFERENCE_FAMILY,
                "cell": f"{cell_label}__paired_difference",
                "role": DIAGNOSTIC_ROLE,
                **common_row,
                "failed_replicates": 0,
                "truth": 0.0,
                "estimate": float(targeted.psi) - other,
                "std_error": float(targeted.std_error),
                "covered": covered,
                "rejected": 1 - covered,
            }
        )
    drop, error = largest_decrease(curve)
    decreases = int(drop > 0.0)
    rows.append(
        {
            "property": DIFFERENCE_FAMILY,
            "cell": DECREASE_CELL,
            "role": DIAGNOSTIC_ROLE,
            **common_row,
            "failed_replicates": 0,
            "truth": 0.0,
            "estimate": drop,
            "std_error": error,
            "covered": 1 - decreases,
            "rejected": decreases,
        }
    )
    return rows


def mechanism_difference_rows(
    frame: pd.DataFrame, hazard: Any, common_row: dict[str, Any], label: str
) -> list[dict[str, Any]]:
    """The paired difference of the two consistent, misspecified-regression fits of one draw.

    ``hazard`` is the draw's ``mechanism_correct`` hazard fit (wrong hazard learner, correct
    mechanism).  The iterated-mean fit of the same working model uses
    :class:`W1QuasiBinomial` regressions and the same correct mechanism.
    """
    scenario, _ = FITS[label]
    iterated = iterated_fit(frame, scenario=scenario, regression_correct=False)
    rows: list[dict[str, Any]] = []
    for cell_label, name in COEFFICIENTS[label]:
        targeted = hazard[name]
        low, high = targeted.ci
        other = float(iterated[name].psi)
        covered = int(low <= other <= high)
        rows.append(
            {
                "property": DIFFERENCE_FAMILY,
                "cell": f"{cell_label}__{MECHANISM_DIFFERENCE}",
                "role": DIAGNOSTIC_ROLE,
                **common_row,
                "failed_replicates": 0,
                "truth": 0.0,
                "estimate": float(targeted.psi) - other,
                "std_error": float(targeted.std_error),
                "covered": covered,
                "rejected": 1 - covered,
            }
        )
    return rows


def fit_drawn(payload: _Payload, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The replicate rows of one fit of one fit set on its drawn sample."""
    family, _, configuration, replicate, n, requested, label = payload
    common_row = {"replicate": replicate, "n": n, "requested_replicates": requested}
    shared: dict[str, Any] = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell_label: str, cell: str, role: str, estimate: Any) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=TRUTHS[cell_label],
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **shared,
        )

    if family == DIFFERENCE_FAMILY:
        return _difference_rows(frame, common_row, shared, label)
    if family == "interval_calibration":
        joint = label == "survival_logit"
        result = fit_label(label, frame, simultaneous=joint)
        rows = [
            row(cell_label, f"{cell_label}__correctly_specified", "positive", result[name])
            for cell_label, name in COEFFICIENTS[label]
        ]
        if joint:
            rows.extend(
                joint_coverage_rows(
                    result,
                    {
                        ESTIMAND[cell_label]: TRUTHS[cell_label]
                        for cell_label, _ in COEFFICIENTS[label]
                    },
                    tuple(result.estimates),
                    label=JOINT_LABEL,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        return rows
    result = fit_label(label, frame, configuration=configuration)
    if family == "root_n_and_efficiency":
        role = "control" if n == min(RATE_SIZES) else "positive"
        return [
            row(cell_label, f"{cell_label}__n_{n}", role, result[name])
            for cell_label, name in COEFFICIENTS[label]
        ]
    if family == "double_robustness" and label == "competing_logit":
        return [
            row(
                COMPETING_DR_LABEL,
                f"{COMPETING_DR_LABEL}__{configuration}",
                "positive",
                result[ESTIMAND[COMPETING_DR_LABEL]],
            )
        ]
    estimate = result[ESTIMAND[MODIFIER]]
    if family == "double_robustness":
        role = "control" if configuration == "both_wrong" else "positive"
        rows = [row(MODIFIER, f"{MODIFIER}__{configuration}", role, estimate)]
        if configuration == "mechanism_correct":
            rows += mechanism_difference_rows(frame, result, common_row, label)
        return rows
    if family == "targeting_necessity":
        return [
            row(MODIFIER, f"{MODIFIER}__targeted", "positive", estimate),
            control_row(
                property_name=family,
                cell=f"{MODIFIER}__untargeted",
                truth=TRUTHS[MODIFIER],
                estimate=plug_in(result, ESTIMAND[MODIFIER]),
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
                **shared,
            ),
        ]
    return [row(MODIFIER, f"{MODIFIER}__alternative", "positive", estimate)]


def _payloads(
    fit_sets: Sequence[tuple[str, str, str, int, int, str]], budget: int | None
) -> list[_Key]:
    out: list[_Key] = []
    for family, stream, configuration, n, replicates, label in fit_sets:
        requested = replicates if budget is None else budget
        out.extend(
            ((family, stream, configuration, r, n, requested, label),) for r in range(requested)
        )
    return out


#: The share of a cell's requested replications that the accounting lets a run lose.  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_drawn(key[0], frame)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = fit_properties(
        _payloads(FIT_SETS, budget),
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CONTROLLED_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def synthetic_scale(cell: PropertyCell) -> float:
    """The standard deviation of a synthetic cell's estimates for the property differential.

    A coefficient cell carries an exact efficiency band, so its synthetic estimates spread at
    the coefficient's efficiency bound over ``sqrt(n)``.  A cell that names no coefficient
    keeps the differential's default ``n ** -0.5``.
    """
    label = cell.cell.split("__", 1)[0]
    sd = EFFICIENCY_SD.get(label, 1.0)
    return float(sd / np.sqrt(cell.n))


#: The ``targeting_necessity`` family's declaration, read by its verdict and its rule.
TARGETING_NECESSITY = property_verdicts.Necessity(
    family="targeting_necessity",
    labels=(MODIFIER,),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
)


def difference_readings(summary: pd.DataFrame) -> None:
    """Publish the ``iterated_mean_difference`` rows with no verdict.

    Each ``paired_difference`` row gains ``paired_sd_over_bound``: the empirical SD of the paired
    difference over the coefficient's efficiency-bound standard error at the row's ``n``.  The
    decrease row's share of decreasing curves is its ``rejection_rate``, with the exact interval
    the summary already holds.  Every row reads ``passed`` and ``property_passed`` as true, and
    its ``diagnostic`` role is what says that no margin was read.
    """
    mask = summary["property"] == DIFFERENCE_FAMILY
    for index in summary.index[mask]:
        cell = str(summary.loc[index, "cell"])
        if cell.endswith(("__paired_difference", f"__{MECHANISM_DIFFERENCE}")):
            label = cell.split("__", 1)[0]
            bound = EFFICIENCY_SD[label] / np.sqrt(float(summary.loc[index, "n"]))
            summary.loc[index, "paired_sd_over_bound"] = (
                float(summary.loc[index, "empirical_se"]) / bound
            )
        summary.loc[index, "passed"] = True
        summary.loc[index, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the study's declared rules, and the reported difference rows."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "paired_sd_over_bound"),
        rate_labels=CONTROLLED_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    difference_readings(summary)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration rule takes this study's efficiency
#: band as an explicit argument.  ``iterated_mean_difference`` states no verdict, so no rule
#: reads it.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
