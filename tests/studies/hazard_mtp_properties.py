"""Property families of ``hazard-mtp``.

Every fit is ``LTMLE`` with ``survival_model="hazard"`` and ``targeting="one_step"`` on a law of
:mod:`tests.studies.hazard_mtp_law`, with the regimens ``shift`` (the policy) and ``natural``
(the identity policy) and the natural course as reference.  A cell label names one estimate of
one fit label of :data:`~tests.studies.canonical_hazard_mtp.FITS`: ``<fit>_contrast`` is the
difference ``shift`` minus ``natural`` in the risk by the last visit, and ``<fit>_risk`` the risk
of ``shift`` by the last visit.  A joint label ``<fit>_curve`` reads every estimate of one fit
(both risk curves and the contrast at each visit) under the simultaneous band.  Every truth is
an exact sum or a quadrature to ``1e-10`` and every efficiency bound is exact
(:func:`~tests.studies.hazard_mtp_law.efficiency_sd`).

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``interval_calibration``        the contrast and the risk of each of the five fit labels,
                                with the shrunken-SE and noise controls on
                                ``levels_contrast`` and ``gaussian_density_contrast``; the
                                calibration count of fits per fit label at n = 2,000
``double_robustness``           ``levels_contrast`` in four nuisance configurations, and the
                                contrast of each Gaussian ratio route with the wrong hazard
                                learner and the correct mechanism; 1,200 at n = 2,000
``targeting_necessity``         ``levels_contrast``: the targeted contrast and the contrast
                                of the initial hazards of the same fit, with the wrong hazard
                                learner and the correct mechanism; 1,200 at n = 2,000
``root_n_and_efficiency``       ``levels_contrast`` and ``levels_risk`` at n = 1,000, 2,000
                                and 8,000; 800 each.  The 1,000 rung is a rate rung that
                                carries no claim (it takes the harness's smallest-rung role)
``power``                       ``levels_contrast`` at n = 2,000; 800
``simultaneous_coverage``       the band over every estimate of the ``levels`` and the
                                ``gaussian_density`` calibration fits, and its pointwise
                                control
``iterated_mean_difference``    reported, no verdict: :data:`DIFFERENCE_REPLICATES` draws at
                                n = 2,000 of ``levels`` and ``gaussian_density``.  Each draw
                                fits the hazard fit, the iterated-mean fit of the same plans
                                without and with ``isotonic=True``, and records the paired
                                differences, each fit's coverage and SE ratio, and whether the
                                iterated-mean risk curve decreases.  On the
                                ``double_robustness`` ``mechanism_correct`` draws of
                                ``levels``: the paired difference of the two fits with
                                regressions that omit ``W``
==============================  ===========================================================

The wrong hazard learner is :class:`~tests.studies.hazard_mtp_law.NoWLogistic`, a per-visit
logistic regression on the dose with no ``W``.  ``W`` confounds the dose, so its dose slope is
biased, and its plug-in contrast lies more than three efficiency standard errors from the truth
at n = 2,000 on every law (``tests/unit/test_hazard_mtp_study_design.py``).  The wrong
mechanism is the prior share for the levels, a Gaussian density with no covariates on the
density route, a classifier on the dose alone on the classifier route, and the prior share for
the censoring.

**The fits-differ reading** (plan X70 section 5 and rev 2.4) lives in one place,
:func:`fits_differ_reading`, with its floors as the named constants
:data:`PAIRED_DIFFERENCE_FLOOR` and :data:`NON_MONOTONE_FLOOR`.  Both fits are efficient when
their nuisance fits are correct, so on the gated laws they differ by :math:`o_P(n^{-1/2})`; the
design test reads the committed pilot against these floors, and a decision on the reading is
pending with the project owner.  Each ``paired_difference`` row records the hazard estimate
minus the iterated-mean estimate of the same draw (``isotonic_difference`` the same against the
isotonic iterated-mean fit); its summary adds ``paired_sd_over_bound``, the SD of that
difference over the efficiency-bound standard error.  The ``iterated_curve_decrease`` row
records the largest decrease of the iterated-mean risk curve of either regimen between
consecutive visits; ``rejected`` marks a draw whose curve decreases somewhere.

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost more than its bound.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_hazard_mtp as study
from tests.studies import hazard_mtp_law as law
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
#: PROVISIONAL until the calibration pilot is committed: the arm-pooled rule over the ten
#: calibration arms at a pilot ratio of one (``tests/unit/test_hazard_mtp_study_design.py``).
CALIBRATION_REPLICATES = 4_000
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
#: The fits-differ floors of plan X70 section 5, read by :func:`fits_differ_reading` alone.
#: The paired-difference floor is the SD of hazard minus iterated-mean over the efficiency-bound
#: standard error, on the largest estimate's ratio and on the gated contrast.  The non-monotone
#: floor is the share of draws whose iterated-mean curve (without isotonic) decreases.  A
#: decision on this reading is pending with the project owner.
PAIRED_DIFFERENCE_FLOOR = 0.05
NON_MONOTONE_FLOOR = 0.10
CONFIGURATIONS = ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
DIFFERENCE_FAMILY = "iterated_mean_difference"
MECHANISM_DIFFERENCE = "mechanism_correct_difference"
CONTRAST = study.CONTRAST_LABEL
FITS = study.FITS
ARMS = ("contrast", "risk")


def _labels(fit_label: str) -> tuple[tuple[str, str], ...]:
    """``(cell label, reported name)`` of the contrast and the risk of one fit label."""
    current = study.law_of(fit_label)
    last = current.n_times
    return (
        (f"{fit_label}_contrast", study.contrast_name(last)),
        (f"{fit_label}_risk", study.risk_name(study.SHIFT, last)),
    )


COEFFICIENTS = {fit_label: _labels(fit_label) for fit_label in FITS}
#: Cell label -> the reported name it reads.
ESTIMAND = {label: name for pairs in COEFFICIENTS.values() for label, name in pairs}
#: Cell label -> the fit label it belongs to.
FIT_OF = {label: fit_label for fit_label, pairs in COEFFICIENTS.items() for label, _ in pairs}
CALIBRATION_FITS = study.CALIBRATION_FITS
CONTROLLED_LABELS = study.CONTROLLED_LABELS
JOINT_LABELS = study.JOINT_LABELS
#: Joint label -> the fit label whose calibration fits carry it.
JOINT_FIT = {label: label.removesuffix("_curve") for label in JOINT_LABELS}


def target(label: str) -> law.Target:
    """The linear target a cell label reads."""
    current = study.law_of(FIT_OF[label])
    last = current.n_times
    if label.endswith("_contrast"):
        return law.Target(((1.0, last, False), (-1.0, last, True)))
    return law.Target(((1.0, last, False),))


def _exact() -> tuple[dict[str, float], dict[str, float]]:
    """The exact truth and efficiency bound of every cell label, each law computed once."""
    truth: dict[str, float] = {}
    bound: dict[str, float] = {}
    cache: dict[tuple[str, str], tuple[float, float]] = {}
    for label, fit_label in FIT_OF.items():
        law_name = FITS[fit_label][0]
        arm = label.rsplit("_", 1)[1]
        if (law_name, arm) not in cache:
            current, aim = law.LAWS[law_name], target(label)
            cache[(law_name, arm)] = (aim.value(current), law.efficiency_sd(current, aim))
        truth[label], bound[label] = cache[(law_name, arm)]
    return truth, bound


TRUTHS, EFFICIENCY_SD = _exact()


def fit_label(
    label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
) -> Any:
    """The hazard fit of one fit label's sample."""
    return study.fit(label, frame, configuration=configuration, simultaneous=simultaneous)


class NoWQuasiBinomial(QuasiBinomialGLM):
    """The quasibinomial regression on the dose block alone: the design without ``W``.

    It drops ``W`` as :class:`~tests.studies.hazard_mtp_law.NoWLogistic` does, so the
    iterated-mean fit it serves is misspecified the same way as the wrong hazard fit.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NoWQuasiBinomial:
        super().fit(np.asarray(X, dtype=float)[:, 2:], y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(np.asarray(X, dtype=float)[:, 2:])

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(np.asarray(X, dtype=float)[:, 2:])


def iterated_fit(
    label: str,
    frame: pd.DataFrame,
    *,
    isotonic: bool = False,
    regression_correct: bool = True,
) -> Any:
    """The iterated-mean fit of the same plans on one fit label's sample.

    Each visit's regression is the unpenalised quasibinomial regression on the design an
    iterated-mean fit hands its learners (``W1``, ``W2`` and the dose block), or on the dose
    block alone when ``regression_correct`` is false (:class:`NoWQuasiBinomial`).  The mechanism
    learners are the hazard fit's correct ones.
    """
    law_name, ratio, folds = FITS[label]
    regression = QuasiBinomialGLM if regression_correct else NoWQuasiBinomial
    mechanism = study.learners(label, frame)
    del mechanism["outcome_learner"]
    settings: dict[str, Any] = {"ratio": ratio}
    if ratio == "density" and law.LAWS[law_name].continuous:
        settings["density_bins"] = study.oracle_bins(len(frame))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            study.regimens(law_name),
            reference=study.NATURAL,
            n_folds=folds,
            simultaneous=False,
            survival_model="iterated_mean",
            isotonic=isotonic,
            targeting="iterative",
            max_iter=100,
            tol=study.TOL,
            random_state=0,
            outcome_learner=regression(),
            pseudo_learner=regression(),
            **settings,
            **mechanism,
        ).fit(law.LAWS[law_name].container(frame))


def largest_decrease(result: Any, n_times: int) -> tuple[float, float]:
    """The largest decrease of a fit's risk curves between visits, and its standard error.

    The decrease of regimen ``r`` at visit ``t`` is :math:`\\hat F_r(t - 1) - \\hat F_r(t)`, and
    its standard error is the SD of the difference of the two influence curves over
    :math:`\\sqrt n`.  The largest over both regimens and every visit is negative when both
    curves rise everywhere.
    """
    best: tuple[float, float] | None = None
    for label in (study.SHIFT, study.NATURAL):
        for horizon in range(2, n_times + 1):
            before = result[study.risk_name(label, horizon - 1)]
            after = result[study.risk_name(label, horizon)]
            drop = float(before.psi - after.psi)
            curve = np.asarray(before.influence_curve) - np.asarray(after.influence_curve)
            error = float(np.std(curve) / np.sqrt(curve.size))
            if best is None or drop > best[0]:
                best = (drop, error)
    assert best is not None
    return best


def draw(fit_label_: str, n: int, seed: int) -> pd.DataFrame:
    return study.law_of(fit_label_).draw(n, seed)


def plug_in(result: Any, name: str) -> float:
    """The estimate of the initial hazards of a fit."""
    return study.initial_estimates(result, (name,))[name]


# ------------------------------------------------------------------ the fits-differ reading


def fits_differ_reading(
    paired_ratios: dict[str, float], decreasing: int, draws: int, level: float = 0.99
) -> dict[str, Any]:
    """The one place the fits-differ floors are read (plan X70 section 5, rev 2.4).

    ``paired_ratios`` maps each cell label to its paired-difference SD over the efficiency-bound
    standard error; ``decreasing`` counts the draws whose iterated-mean curve without isotonic
    decreases somewhere, out of ``draws``.  The reading passes when the largest ratio and the
    gated contrast's ratio reach :data:`PAIRED_DIFFERENCE_FLOOR` and the decreasing share
    reaches :data:`NON_MONOTONE_FLOOR`.  The share's Clopper-Pearson interval at ``level`` is
    reported beside it.
    """
    tail = (1.0 - level) / 2.0
    low = float(beta.ppf(tail, decreasing, draws - decreasing + 1)) if decreasing else 0.0
    high = float(beta.ppf(1.0 - tail, decreasing + 1, draws - decreasing)) if decreasing < draws else 1.0
    largest = max(paired_ratios.values())
    contrasts = [value for label, value in paired_ratios.items() if label.endswith("_contrast")]
    share = decreasing / draws
    return {
        "largest_ratio": largest,
        "contrast_ratio": min(contrasts) if contrasts else float("nan"),
        "share": share,
        "share_interval": (low, high),
        "paired_passes": largest >= PAIRED_DIFFERENCE_FLOOR
        and bool(contrasts)
        and min(contrasts) >= PAIRED_DIFFERENCE_FLOOR,
        "monotone_passes": share >= NON_MONOTONE_FLOOR,
    }


# ------------------------------------------------------------------ the grid

#: Every fit set: ``(family, stream label, configuration, n, replicates, fit label)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("interval_calibration", fit, "both_correct", CALIBRATION_N, CALIBRATION_REPLICATES, fit)
        for fit in CALIBRATION_FITS
    ),
    *(
        ("double_robustness", c, c, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, "levels")
        for c in CONFIGURATIONS
    ),
    *(
        (
            "double_robustness",
            f"{FIT_OF[label]}_mechanism_correct",
            "mechanism_correct",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            FIT_OF[label],
        )
        for label in study.ROUTE_DR_LABELS
    ),
    (
        "targeting_necessity",
        "wrong_hazard",
        "mechanism_correct",
        NECESSITY_N,
        NECESSITY_REPLICATES,
        "levels",
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "levels")
        for size in RATE_SIZES
    ),
    ("power", "alternative", "both_correct", POWER_N, POWER_REPLICATES, "levels"),
    *(
        (DIFFERENCE_FAMILY, fit, "both_correct", DIFFERENCE_N, DIFFERENCE_REPLICATES, fit)
        for fit in study.DIFFERENCE_FITS
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


class DeclaredEstimates:
    """The law a fit label's cells read, with its exact truths.

    Parameters
    ----------
    fit_label : str
        A key of :data:`FITS`.
    """

    def __init__(self, fit_label: str) -> None:
        self.fit_label = fit_label
        self.name = f"hazard_mtp__{fit_label}"

    def truth(self) -> dict[str, float]:
        return {ESTIMAND[label]: TRUTHS[label] for label, _ in COEFFICIENTS[self.fit_label]}


class DeclaredDifference:
    """The zero every ``iterated_mean_difference`` difference row is recorded against."""

    name = "hazard_mtp__iterated_mean_difference"

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
        dgp = DeclaredEstimates(fit)
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
    levels = DeclaredEstimates("levels")
    for configuration in CONFIGURATIONS:
        add(
            "double_robustness",
            f"{CONTRAST}__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            levels,
            ESTIMAND[CONTRAST],
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
        )
    for label in study.ROUTE_DR_LABELS:
        add(
            "double_robustness",
            f"{label}__mechanism_correct",
            "positive",
            DeclaredEstimates(FIT_OF[label]),
            ESTIMAND[label],
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            f"{FIT_OF[label]}_mechanism_correct",
        )
    # The reported paired difference reads the levels ``mechanism_correct`` draws.
    for label, _ in COEFFICIENTS["levels"]:
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
            f"{CONTRAST}__{arm}",
            role,
            levels,
            ESTIMAND[CONTRAST],
            NECESSITY_N,
            NECESSITY_REPLICATES,
            "wrong_hazard",
        )
    for size in RATE_SIZES:
        for label, name in COEFFICIENTS["levels"]:
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                "control" if size == min(RATE_SIZES) else "positive",
                levels,
                name,
                size,
                RATE_REPLICATES,
                f"n_{size}",
            )
    add(
        "power",
        f"{CONTRAST}__alternative",
        "positive",
        levels,
        ESTIMAND[CONTRAST],
        POWER_N,
        POWER_REPLICATES,
        "alternative",
    )
    for joint in JOINT_LABELS:
        cells.extend(
            joint_property_cells(
                joint,
                n=CALIBRATION_N,
                replicates=CALIBRATION_REPLICATES,
                seed=_seed("interval_calibration", JOINT_FIT[joint], 0),
            )
        )
    difference = DeclaredDifference()
    for fit in study.DIFFERENCE_FITS:
        dgp = DeclaredEstimates(fit)
        for label, name in COEFFICIENTS[fit]:
            for arm in study.DIFFERENCE_ARMS:
                paired = arm in ("paired_difference", "isotonic_difference")
                add(
                    DIFFERENCE_FAMILY,
                    f"{label}__{arm}",
                    DIAGNOSTIC_ROLE,
                    difference if paired else dgp,
                    "paired_difference" if paired else name,
                    DIFFERENCE_N,
                    DIFFERENCE_REPLICATES,
                    fit,
                )
        add(
            DIFFERENCE_FAMILY,
            f"{fit}__iterated_curve_decrease",
            DIAGNOSTIC_ROLE,
            difference,
            "largest_decrease",
            DIFFERENCE_N,
            DIFFERENCE_REPLICATES,
            fit,
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
        joint = f"{label}_curve"
        if joint in JOINT_LABELS:
            names += [f"{JOINT_FAMILY}/{cell}" for cell in joint_cells(joint)]
        return names
    if family == "double_robustness":
        if label != "levels":
            return [f"{family}/{label}_contrast__{configuration}"]
        names = [f"{family}/{CONTRAST}__{configuration}"]
        if configuration == "mechanism_correct":
            names += [
                f"{DIFFERENCE_FAMILY}/{cell_label}__{MECHANISM_DIFFERENCE}"
                for cell_label, _ in COEFFICIENTS[label]
            ]
        return names
    if family == "targeting_necessity":
        return [f"{family}/{CONTRAST}__targeted", f"{family}/{CONTRAST}__untargeted"]
    if family == "root_n_and_efficiency":
        return [f"{family}/{cell_label}__n_{n}" for cell_label, _ in COEFFICIENTS[label]]
    if family == DIFFERENCE_FAMILY:
        return [
            *(
                f"{family}/{cell_label}__{arm}"
                for cell_label, _ in COEFFICIENTS[label]
                for arm in study.DIFFERENCE_ARMS
            ),
            f"{family}/{label}__iterated_curve_decrease",
        ]
    return [f"{family}/{CONTRAST}__alternative"]


def cell_of(key: _Key) -> str:
    return f"{key[0][0]}/{key[0][1]}"


def replicate_of(key: _Key) -> int:
    return key[0][3]


def fit_set_rows(payload: _Payload) -> list[dict[str, Any]]:
    """The replicate rows of one fit of one fit set, drawn and fitted in one call."""
    return fit_drawn(payload, _draw((payload,)))


def _paired_row(
    cell: str, hazard: Any, other: float, common_row: dict[str, Any]
) -> dict[str, Any]:
    low, high = hazard.ci
    covered = int(low <= other <= high)
    return {
        "property": DIFFERENCE_FAMILY,
        "cell": cell,
        "role": DIAGNOSTIC_ROLE,
        **common_row,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": float(hazard.psi) - other,
        "std_error": float(hazard.std_error),
        "covered": covered,
        "rejected": 1 - covered,
    }


def _difference_rows(
    frame: pd.DataFrame, common_row: dict[str, Any], shared: dict[str, Any], label: str
) -> list[dict[str, Any]]:
    hazard = fit_label(label, frame)
    iterated = iterated_fit(label, frame)
    isotonic = iterated_fit(label, frame, isotonic=True)
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
        rows.append(
            _paired_row(
                f"{cell_label}__paired_difference",
                hazard[name],
                float(iterated[name].psi),
                common_row,
            )
        )
        rows.append(
            _paired_row(
                f"{cell_label}__isotonic_difference",
                hazard[name],
                float(isotonic[name].psi),
                common_row,
            )
        )
    drop, error = largest_decrease(iterated, study.law_of(label).n_times)
    decreases = int(drop > 0.0)
    rows.append(
        {
            "property": DIFFERENCE_FAMILY,
            "cell": f"{label}__iterated_curve_decrease",
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
    mechanism).  The iterated-mean fit of the same plans uses :class:`NoWQuasiBinomial`
    regressions and the same correct mechanism.
    """
    iterated = iterated_fit(label, frame, regression_correct=False)
    return [
        _paired_row(
            f"{cell_label}__{MECHANISM_DIFFERENCE}",
            hazard[name],
            float(iterated[name].psi),
            common_row,
        )
        for cell_label, name in COEFFICIENTS[label]
    ]


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
        joint = f"{label}_curve"
        banded = joint in JOINT_LABELS
        result = fit_label(label, frame, simultaneous=banded)
        rows = [
            row(cell_label, f"{cell_label}__correctly_specified", "positive", result[name])
            for cell_label, name in COEFFICIENTS[label]
        ]
        if banded:
            truth = study.truths(FITS[label][0])
            rows.extend(
                joint_coverage_rows(
                    result,
                    truth,
                    tuple(result.estimates),
                    label=joint,
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
    if family == "double_robustness" and label != "levels":
        cell_label = f"{label}_contrast"
        return [
            row(
                cell_label,
                f"{cell_label}__{configuration}",
                "positive",
                result[ESTIMAND[cell_label]],
            )
        ]
    estimate = result[ESTIMAND[CONTRAST]]
    if family == "double_robustness":
        role = "control" if configuration == "both_wrong" else "positive"
        rows = [row(CONTRAST, f"{CONTRAST}__{configuration}", role, estimate)]
        if configuration == "mechanism_correct":
            rows += mechanism_difference_rows(frame, result, common_row, label)
        return rows
    if family == "targeting_necessity":
        return [
            row(CONTRAST, f"{CONTRAST}__targeted", "positive", estimate),
            control_row(
                property_name=family,
                cell=f"{CONTRAST}__untargeted",
                truth=TRUTHS[CONTRAST],
                estimate=plug_in(result, ESTIMAND[CONTRAST]),
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
                **shared,
            ),
        ]
    return [row(CONTRAST, f"{CONTRAST}__alternative", "positive", estimate)]


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

    A cell that names a cell label spreads at that estimate's efficiency bound over
    ``sqrt(n)``; any other keeps the differential's default ``n ** -0.5``.
    """
    label = cell.cell.split("__", 1)[0]
    sd = EFFICIENCY_SD.get(label, 1.0)
    return float(sd / np.sqrt(cell.n))


#: The ``targeting_necessity`` family's declaration, read by its verdict and its rule.
TARGETING_NECESSITY = property_verdicts.Necessity(
    family="targeting_necessity",
    labels=(CONTRAST,),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
)


def difference_readings(summary: pd.DataFrame) -> None:
    """Publish the ``iterated_mean_difference`` rows with no verdict.

    Each paired row gains ``paired_sd_over_bound``: the empirical SD of the paired difference
    over the estimate's efficiency-bound standard error at the row's ``n``.  The decrease row's
    share of decreasing curves is its ``rejection_rate``, with the exact interval the summary
    already holds.  Every row reads ``passed`` and ``property_passed`` as true, and its
    ``diagnostic`` role is what says that no margin was read.
    """
    mask = summary["property"] == DIFFERENCE_FAMILY
    paired = ("__paired_difference", "__isotonic_difference", f"__{MECHANISM_DIFFERENCE}")
    for index in summary.index[mask]:
        cell = str(summary.loc[index, "cell"])
        if cell.endswith(paired):
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
        rate_labels=study.LADDER_LABELS,
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
