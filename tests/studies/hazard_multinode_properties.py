"""Property families of ``hazard-multinode``.

Every fit is ``LTMLE`` with ``survival_model="hazard"`` on a law of
:mod:`tests.studies.hazard_multinode_law`, with a binary treatment at every node: the per-horizon
walk for the regimen fits and the one-step walk for the working model
(:func:`~tests.studies.canonical_hazard_multinode.algorithm`).  A cell label
names one estimate of one fit label of :data:`~tests.studies.canonical_hazard_multinode.FITS`:
``base_contrast`` and ``five_contrast`` are the difference ``always`` minus ``never`` in the risk
by the last node; ``base_rule``, ``base_policy``, ``base_stay`` and ``five_policy`` the risk of
that regimen by the last node; ``msm_dose`` and ``msm_dose_W1`` the coefficients of ``d`` and
``d W1`` of the working model over the static paths.  A joint label ``<fit>_curve`` reads every
estimate of one fit (each regimen's risk curve and its contrast with ``never`` at each node)
under the simultaneous band.  Every truth is an exact sum over the paths and every efficiency
bound an exact sum over the support of the observed law
(:func:`~tests.studies.hazard_multinode_law.efficiency_sd`,
:func:`~tests.studies.hazard_multinode_law.beta_sd`).

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``interval_calibration``        the eight arms of the three fit labels
                                (:data:`~tests.studies.canonical_hazard_multinode.ARMS`), with
                                the shrunken-SE and noise controls on ``base_contrast``;
                                :data:`CALIBRATION_REPLICATES` fits per fit label at
                                n = 2,000
``double_robustness``           ``base_contrast`` in four nuisance configurations, and every
                                other calibration arm with the wrong hazard learner and the
                                correct mechanism (the ``base`` arms from the ``base``
                                ``mechanism_correct`` fits, the ``five`` and ``msm`` arms from
                                fits of their own); :data:`DOUBLE_ROBUST_REPLICATES` at
                                n = 2,000, or the wrong-learner count of
                                :data:`MECHANISM_REPLICATES`
``targeting_necessity``         ``base_contrast``: the targeted contrast and the contrast of
                                the initial hazards of the same fit, with the wrong hazard
                                learner and the correct mechanism; :data:`NECESSITY_REPLICATES`
                                at n = 2,000
``root_n_and_efficiency``       ``base_contrast`` and ``base_policy`` at n = 1,000, 2,000 and
                                8,000; 800 each.  The 1,000 rung is a rate rung that carries
                                no claim (it takes the harness's smallest-rung role)
``power``                       ``base_contrast`` at n = 2,000; 800
``simultaneous_coverage``       the band over every estimate of the ``base`` and the ``five``
                                calibration fits, and its pointwise control
``iterated_mean_difference``    reported, no verdict: :data:`DIFFERENCE_REPLICATES` draws at
                                n = 2,000 of ``base`` and ``five``.  Each draw fits the hazard
                                fit, the iterated-mean fit of the same regimens without and
                                with ``isotonic=True``, and records the paired differences,
                                each fit's coverage and SE ratio, and whether an iterated-mean
                                risk curve decreases.  On the ``double_robustness``
                                ``mechanism_correct`` draws of ``base``: the paired difference
                                of the two fits with regressions that omit ``W``
==============================  ===========================================================

The wrong hazard learner is :class:`~tests.studies.hazard_multinode_law.LastArmLogistic`, a
per-node logistic regression on the node's own arm with no earlier arm and no ``W`` (plan X70
section 5).  ``W`` and the previous arm confound the arm, so its arm effect is biased, and its
plug-in ``base_contrast`` lies 3.54 efficiency standard errors from the truth at n = 2,000
(``tests/unit/test_hazard_multinode_study_design.py``).  The wrong mechanism is the prior share
of the treatment and of the retention at each node.

**The agreement reading** is reported with no floor.  :func:`fits_differ_reading` states the
paired ratios and the share of decreasing iterated-mean curves with its interval.  Both fits are
efficient, so where both nuisance fits are correct they differ by :math:`o_P(n^{-1/2})`, and X70
reports their agreement instead of requiring a gap (the owner's decision of 2026-11-13).  Each
``paired_difference`` row records the hazard estimate minus the iterated-mean estimate of the
same draw (``isotonic_difference`` the same against the isotonic iterated-mean fit); its summary
adds ``paired_sd_over_bound``, the SD of that difference over the efficiency-bound standard
error.  The ``iterated_curve_decrease`` row records the largest decrease of an iterated-mean
risk curve between consecutive nodes over every regimen; ``rejected`` marks a draw whose curve
decreases somewhere.

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost more than its bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_hazard_multinode as study
from tests.studies import hazard_multinode_law as law
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
#: The replicates of each ``mechanism_correct`` fit set, by fit label: the larger of
#: :data:`DOUBLE_ROBUST_REPLICATES` and the wrong-learner pilot's need over the fit set's arms
#: (plan X70 rev 2.7 rule, read by the design test).  Provisional until the wrong-learner pilot
#: is committed.
MECHANISM_REPLICATES = {"base": 1_200, "five": 1_200, "msm": 1_200}
#: Provisional until the wrong-learner pilot is committed, as :data:`MECHANISM_REPLICATES`.
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = study.RATE_SIZES
CALIBRATION_N = 2_000
#: The arm-pooled rule over the eight calibration arms.  Until the calibration pilot is
#: committed it is the rule read at a pilot of the declared size whose SE ratio is one
#: (``tests/unit/test_hazard_multinode_study_design.py``); the committed pilot then sets it.
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
CONFIGURATIONS = ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
DIFFERENCE_FAMILY = "iterated_mean_difference"
MECHANISM_DIFFERENCE = "mechanism_correct_difference"
CONTRAST = study.CONTRAST_LABEL
FITS = study.FITS
#: The per-row record of each fit, and the summary columns that count it
#: (:func:`convergence_reading`).  No verdict reads them.
CONVERGENCE_COLUMNS = ("converged", "final_score", "unexpected_warnings")
CONVERGENCE_SUMMARY = (
    "nonconverged_fits",
    "max_final_score",
    "unexpected_warning_fits",
    "convergence_defect_suspect",
)
#: A fit that does not converge and whose final relative score is above this is a recordable
#: failure under :data:`FAILURE_BOUND` (:func:`stalled_fit`).  A miss at or below it is the
#: walk's reporting floor near ``1e-8`` (plan X70 rev 2.5), and the fit keeps its rows.
STALLED_SCORE = 1e-6
#: A cell whose non-converged fits, whatever their scores, exceed this share of its requested
#: replicates is a defect suspect (:func:`convergence_reading`).
NONCONVERGED_SUSPECT_SHARE = 0.01


def _name(fit_label: str, arm: str) -> str:
    """The reported name a calibration arm reads."""
    if FITS[fit_label][2]:
        return law.coefficient_name(arm)
    last = study.law_of(fit_label).n_times
    if arm == "contrast":
        return study.contrast_name("always", last)
    return study.risk_name(arm, last)


#: Fit label -> ``(cell label, reported name)`` of each of its arms.
COEFFICIENTS = {
    fit: tuple((f"{fit}_{arm}", _name(fit, arm)) for arm in study.ARMS[fit]) for fit in FITS
}
#: Cell label -> the reported name it reads.
ESTIMAND = {label: name for pairs in COEFFICIENTS.values() for label, name in pairs}
#: Cell label -> the fit label it belongs to.
FIT_OF = {label: fit_label for fit_label, pairs in COEFFICIENTS.items() for label, _ in pairs}
CALIBRATION_FITS = study.CALIBRATION_FITS
CONTROLLED_LABELS = study.CONTROLLED_LABELS
JOINT_LABELS = study.JOINT_LABELS
#: Joint label -> the fit label whose calibration fits carry it.
JOINT_FIT = {label: label.removesuffix("_curve") for label in JOINT_LABELS}
#: The fit labels whose ``mechanism_correct`` fits carry robustness cells beyond ``base``.
ROUTE_FITS = tuple(fit for fit in FITS if fit != "base")


def target(label: str) -> law.Target:
    """The linear target a regimen cell label reads; a coefficient label has none."""
    fit_label = FIT_OF[label]
    if FITS[fit_label][2]:
        raise ValueError(f"{label} is a working-model coefficient, not a linear target")
    last = study.law_of(fit_label).n_times
    arm = label.removeprefix(f"{fit_label}_")
    if arm == "contrast":
        return law.Target(((1.0, "always", last), (-1.0, "never", last)))
    return law.Target(((1.0, arm, last),))


def _exact() -> tuple[dict[str, float], dict[str, float]]:
    """The exact truth and efficiency bound of every cell label."""
    truth: dict[str, float] = {}
    bound: dict[str, float] = {}
    for label, fit_label in FIT_OF.items():
        current = study.law_of(fit_label)
        if FITS[fit_label][2]:
            continue
        aim = target(label)
        truth[label], bound[label] = aim.value(current), law.efficiency_sd(current, aim)
    coefficients = dict(zip(law.TERMS, law.exact_beta(law.BASE), strict=True))
    spreads = dict(zip(law.TERMS, law.beta_sd(law.BASE), strict=True))
    for label, _ in COEFFICIENTS["msm"]:
        term = label.removeprefix("msm_")
        truth[label], bound[label] = float(coefficients[term]), float(spreads[term])
    return truth, bound


TRUTHS, EFFICIENCY_SD = _exact()


def wrong_limit(label: str) -> float:
    """The plug-in value of a cell label at the limit of the wrong hazard learner."""
    fit_label = FIT_OF[label]
    current = study.law_of(fit_label)
    if FITS[fit_label][2]:
        coefficients = dict(zip(law.TERMS, law.wrong_beta(current), strict=True))
        return float(coefficients[label.removeprefix("msm_")])
    return law.wrong_hazard_limit(current, target(label))


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
    """The quasibinomial regression on the arm columns alone: the design without ``W``.

    An iterated-mean regression design is ``[W1, W2]`` followed by the arms it conditions on:
    none for a static path or the rule, whose regressions run on the units that follow them,
    and the arms so far for the policy and "stay".  This drops ``W``, so a static regression is
    an intercept, and the iterated-mean fit it serves omits ``W`` as
    :class:`~tests.studies.hazard_multinode_law.LastArmLogistic` does.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NoWQuasiBinomial:
        super().fit(np.asarray(X, dtype=float)[:, 2:], y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        # The parent's ``predict_proba`` calls this ``predict``, so it reads the slice too.
        return super().predict(np.asarray(X, dtype=float)[:, 2:])


def iterated_fit(
    label: str,
    frame: pd.DataFrame,
    *,
    isotonic: bool = False,
    regression_correct: bool = True,
) -> Any:
    """The iterated-mean fit of the same regimens on one fit label's sample.

    Each node's regression is the unpenalised quasibinomial regression on the design an
    iterated-mean fit hands its learners (``W1``, ``W2`` and the arms it conditions on), or on
    that design without ``W`` when ``regression_correct`` is false (:class:`NoWQuasiBinomial`).
    The mechanism learners are the hazard fit's correct ones.  It filters no warning;
    :meth:`FitRecords.iterated` records them.
    """
    regression = QuasiBinomialGLM if regression_correct else NoWQuasiBinomial
    mechanism = study.learners()
    del mechanism["outcome_learner"]
    return LTMLE(
        study.regimens(label),
        reference=study.REFERENCE,
        n_folds=1,
        simultaneous=False,
        survival_model="iterated_mean",
        isotonic=isotonic,
        targeting="iterative",
        max_iter=100,
        tol=study.TOL,
        g_bounds=study.G_BOUNDS,
        random_state=0,
        outcome_learner=regression(),
        pseudo_learner=regression(),
        **mechanism,
    ).fit(frame, **study.law_of(label).fit_columns())


def largest_decrease(result: Any, fit_label_: str) -> tuple[float, float]:
    """The largest decrease of a fit's risk curves between nodes, and its standard error.

    The decrease of regimen ``r`` at node ``t`` is :math:`\\hat F_r(t - 1) - \\hat F_r(t)`, and
    its standard error is the SD of the difference of the two influence curves over
    :math:`\\sqrt n`.  The largest over every regimen and every node is negative when every
    curve rises everywhere.
    """
    _, labels, _ = FITS[fit_label_]
    best: tuple[float, float] | None = None
    for label in labels:
        for horizon in range(2, study.law_of(fit_label_).n_times + 1):
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


# ------------------------------------------------------------------ the agreement reading


def fits_differ_reading(
    paired_ratios: dict[str, float], decreasing: int, draws: int, level: float = 0.99
) -> dict[str, Any]:
    """The reported agreement of the hazard and iterated-mean fits; it carries no floor.

    ``paired_ratios`` maps each cell label to its paired-difference SD over the efficiency-bound
    standard error; ``decreasing`` counts the draws whose iterated-mean curve without isotonic
    decreases somewhere, out of ``draws``.  The reading returns the ratios, the largest of them
    and the smallest contrast ratio, the decreasing share with its Clopper-Pearson interval at
    ``level``, and ``text``, the sentence that states them.  Both fits are efficient, so with
    correct nuisance fits they differ by :math:`o_P(n^{-1/2})`, and X70 reports their agreement
    instead of requiring a gap (the owner's decision of 2026-11-13).
    """
    tail = (1.0 - level) / 2.0
    low = float(beta.ppf(tail, decreasing, draws - decreasing + 1)) if decreasing else 0.0
    high = (
        float(beta.ppf(1.0 - tail, decreasing + 1, draws - decreasing))
        if decreasing < draws
        else 1.0
    )
    largest = max(paired_ratios.values())
    contrasts = [value for label, value in paired_ratios.items() if "_contrast" in label]
    contrast = min(contrasts) if contrasts else float("nan")
    share = decreasing / draws
    ratios = ", ".join(f"{label} {value:.3f}" for label, value in sorted(paired_ratios.items()))
    text = (
        f"Paired-difference SD over the efficiency SE: {ratios}. Decreasing iterated-mean "
        f"curves: {decreasing} of {draws}, {share:.3f} with {level:.0%} interval "
        f"[{low:.3f}, {high:.3f}]. Reported, with no floor."
    )
    return {
        "ratios": dict(paired_ratios),
        "largest_ratio": largest,
        "contrast_ratio": contrast,
        "share": share,
        "share_interval": (low, high),
        "text": text,
    }


# ------------------------------------------------------------------ the grid

#: Every fit set: ``(family, stream label, configuration, n, replicates, fit label)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("interval_calibration", fit, "both_correct", CALIBRATION_N, CALIBRATION_REPLICATES, fit)
        for fit in CALIBRATION_FITS
    ),
    *(
        (
            "double_robustness",
            c,
            c,
            DOUBLE_ROBUST_N,
            MECHANISM_REPLICATES["base"] if c == "mechanism_correct" else DOUBLE_ROBUST_REPLICATES,
            "base",
        )
        for c in CONFIGURATIONS
    ),
    *(
        (
            "double_robustness",
            f"{fit}_mechanism_correct",
            "mechanism_correct",
            DOUBLE_ROBUST_N,
            MECHANISM_REPLICATES[fit],
            fit,
        )
        for fit in ROUTE_FITS
    ),
    (
        "targeting_necessity",
        "wrong_hazard",
        "mechanism_correct",
        NECESSITY_N,
        NECESSITY_REPLICATES,
        "base",
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "base")
        for size in RATE_SIZES
    ),
    ("power", "alternative", "both_correct", POWER_N, POWER_REPLICATES, "base"),
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
        self.name = f"hazard_multinode__{fit_label}"

    def truth(self) -> dict[str, float]:
        return {ESTIMAND[label]: TRUTHS[label] for label, _ in COEFFICIENTS[self.fit_label]}


class DeclaredDifference:
    """The zero every ``iterated_mean_difference`` difference row is recorded against."""

    name = "hazard_multinode__iterated_mean_difference"

    def truth(self) -> dict[str, float]:
        return {"paired_difference": 0.0, "largest_decrease": 0.0}


def _dr_replicates(fit: str, configuration: str) -> int:
    if configuration == "mechanism_correct":
        return MECHANISM_REPLICATES[fit]
    return DOUBLE_ROBUST_REPLICATES


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
    base = DeclaredEstimates("base")
    for configuration in CONFIGURATIONS:
        add(
            "double_robustness",
            f"{CONTRAST}__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            base,
            ESTIMAND[CONTRAST],
            DOUBLE_ROBUST_N,
            _dr_replicates("base", configuration),
            configuration,
        )
    for label in study.ROUTE_DR_LABELS:
        fit = FIT_OF[label]
        add(
            "double_robustness",
            f"{label}__mechanism_correct",
            "positive",
            DeclaredEstimates(fit),
            ESTIMAND[label],
            DOUBLE_ROBUST_N,
            MECHANISM_REPLICATES[fit],
            "mechanism_correct" if fit == "base" else f"{fit}_mechanism_correct",
        )
    # The reported paired difference reads the base ``mechanism_correct`` draws.
    for label, _ in COEFFICIENTS["base"]:
        add(
            DIFFERENCE_FAMILY,
            f"{label}__{MECHANISM_DIFFERENCE}",
            DIAGNOSTIC_ROLE,
            DeclaredDifference(),
            "paired_difference",
            DOUBLE_ROBUST_N,
            MECHANISM_REPLICATES["base"],
            "mechanism_correct",
            draws_of="double_robustness",
        )
    for arm, role in (("targeted", "positive"), ("untargeted", "control")):
        add(
            "targeting_necessity",
            f"{CONTRAST}__{arm}",
            role,
            base,
            ESTIMAND[CONTRAST],
            NECESSITY_N,
            NECESSITY_REPLICATES,
            "wrong_hazard",
        )
    for size in RATE_SIZES:
        for label in study.LADDER_LABELS:
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                "control" if size == min(RATE_SIZES) else "positive",
                base,
                ESTIMAND[label],
                size,
                RATE_REPLICATES,
                f"n_{size}",
            )
    add(
        "power",
        f"{CONTRAST}__alternative",
        "positive",
        base,
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
        if label != "base":
            return [f"{family}/{cell_label}__{configuration}" for cell_label, _ in COEFFICIENTS[label]]
        names = [f"{family}/{CONTRAST}__{configuration}"]
        if configuration == "mechanism_correct":
            names += [
                f"{family}/{cell_label}__{configuration}"
                for cell_label, _ in COEFFICIENTS[label]
                if cell_label != CONTRAST
            ]
            names += [
                f"{DIFFERENCE_FAMILY}/{cell_label}__{MECHANISM_DIFFERENCE}"
                for cell_label, _ in COEFFICIENTS[label]
            ]
        return names
    if family == "targeting_necessity":
        return [f"{family}/{CONTRAST}__targeted", f"{family}/{CONTRAST}__untargeted"]
    if family == "root_n_and_efficiency":
        return [f"{family}/{cell_label}__n_{n}" for cell_label in study.LADDER_LABELS]
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


def _paired_row(cell: str, hazard: Any, other: float, common_row: dict[str, Any]) -> dict[str, Any]:
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


class FitRecords:
    """The :class:`~tests.studies.canonical_hazard_multinode.FitRecord` of every fit one key runs."""

    def __init__(self) -> None:
        self.records: list[study.FitRecord] = []

    def hazard(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        result, record = study.fit_recorded(label, frame, **settings)
        self.records.append(record)
        return result

    def iterated(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        """An iterated-mean fit: its warnings are recorded, its convergence is not the subject's."""
        result, record = study.recorded(lambda: iterated_fit(label, frame, **settings))
        self.records.append(study.FitRecord(True, record.unexpected_warnings))
        return result

    def columns(self) -> dict[str, float]:
        return {
            "converged": int(all(record.converged for record in self.records)),
            "final_score": max((r.final_score for r in self.records), default=0.0),
            "unexpected_warnings": sum(len(r.unexpected_warnings) for r in self.records),
        }


def stalled_fit(rows: pd.DataFrame) -> str | None:
    """The failure rule of a key whose fit stalled: non-converged above :data:`STALLED_SCORE`.

    Passed to the shared harness as its ``extra`` row rule, so such a key is an
    ``invalid_result`` record, a lost replicate charged to the cell's failure bound.
    """
    if not set(CONVERGENCE_COLUMNS) <= set(rows.columns):
        return None
    stalled = (rows["converged"] == 0) & (rows["final_score"] > STALLED_SCORE)
    if not stalled.any():
        return None
    worst = float(rows.loc[stalled, "final_score"].max())
    return (
        f"a hazard walk stopped short of tol with a final relative score of {worst:.3g}, "
        f"above {STALLED_SCORE:g}"
    )


def _difference_rows(
    frame: pd.DataFrame,
    common_row: dict[str, Any],
    shared: dict[str, Any],
    label: str,
    records: FitRecords,
) -> list[dict[str, Any]]:
    hazard = records.hazard(label, frame)
    iterated = records.iterated(label, frame)
    isotonic = records.iterated(label, frame, isotonic=True)
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
    drop, error = largest_decrease(iterated, label)
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
    frame: pd.DataFrame, hazard: Any, common_row: dict[str, Any], label: str, records: FitRecords
) -> list[dict[str, Any]]:
    """The paired difference of the two consistent, misspecified-regression fits of one draw.

    ``hazard`` is the draw's ``mechanism_correct`` hazard fit (wrong hazard learner, correct
    mechanism).  The iterated-mean fit of the same regimens uses :class:`NoWQuasiBinomial`
    regressions and the same correct mechanism.
    """
    iterated = records.iterated(label, frame, regression_correct=False)
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
    """The replicate rows of one fit of one fit set on its drawn sample.

    Every row carries :data:`CONVERGENCE_COLUMNS` of the fits the key ran: ``converged``, 1
    when every hazard walk met its tolerance, and ``unexpected_warnings``, the count of
    warnings outside the expected classes.  A nonconverged fit keeps its rows.
    """
    records = FitRecords()
    rows = _fit_drawn(payload, frame, records)
    columns = records.columns()
    return [{**row, **columns} for row in rows]


def _fit_drawn(payload: _Payload, frame: pd.DataFrame, records: FitRecords) -> list[dict[str, Any]]:
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
        return _difference_rows(frame, common_row, shared, label, records)
    if family == "interval_calibration":
        joint = f"{label}_curve"
        banded = joint in JOINT_LABELS
        result = records.hazard(label, frame, simultaneous=banded)
        rows = [
            row(cell_label, f"{cell_label}__correctly_specified", "positive", result[name])
            for cell_label, name in COEFFICIENTS[label]
        ]
        if banded:
            rows.extend(
                joint_coverage_rows(
                    result,
                    study.truths(label),
                    tuple(result.estimates),
                    label=joint,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        return rows
    result = records.hazard(label, frame, configuration=configuration)
    if family == "root_n_and_efficiency":
        role = "control" if n == min(RATE_SIZES) else "positive"
        return [
            row(cell_label, f"{cell_label}__n_{n}", role, result[ESTIMAND[cell_label]])
            for cell_label in study.LADDER_LABELS
        ]
    if family == "double_robustness" and label != "base":
        return [
            row(cell_label, f"{cell_label}__{configuration}", "positive", result[name])
            for cell_label, name in COEFFICIENTS[label]
        ]
    estimate = result[ESTIMAND[CONTRAST]]
    if family == "double_robustness":
        role = "control" if configuration == "both_wrong" else "positive"
        rows = [row(CONTRAST, f"{CONTRAST}__{configuration}", role, estimate)]
        if configuration == "mechanism_correct":
            rows += [
                row(cell_label, f"{cell_label}__{configuration}", "positive", result[name])
                for cell_label, name in COEFFICIENTS[label]
                if cell_label != CONTRAST
            ]
            rows += mechanism_difference_rows(frame, result, common_row, label, records)
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
        extra=stalled_fit,
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
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *CONVERGENCE_COLUMNS]).sort_values(
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


def convergence_reading(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Count each cell's non-converged fits and other warnings, and flag a defect suspect.

    ``nonconverged_fits`` counts the kept rows whose fit set ``converged`` to 0 (each at or below
    :data:`STALLED_SCORE`, the others being lost replicates by :func:`stalled_fit`),
    ``max_final_score`` is the largest final relative score, and ``unexpected_warning_fits``
    counts the rows whose fits raised a warning outside the expected classes.
    ``convergence_defect_suspect`` is true when the kept non-converged fits plus the cell's lost
    replicates exceed :data:`NONCONVERGED_SUSPECT_SHARE` of its requested replicates.  The lost
    replicates are counted whatever their cause, so the share can only over-state the
    non-converged fits.  A suspect gated cell does not pass.  A derived control's rows carry no
    record and leave the columns empty.

    Raises
    ------
    ValueError
        When a kept row records a non-converged fit above :data:`STALLED_SCORE`, which the
        failure rule should have recorded as a lost replicate.
    """
    for index in summary.index:
        group = rows.loc[
            (rows["property"] == summary.loc[index, "property"])
            & (rows["cell"] == summary.loc[index, "cell"])
        ]
        kept = group.loc[group["converged"].notna()] if "converged" in group else group.iloc[:0]
        if kept.empty:
            continue
        stalled = (kept["converged"] == 0) & (kept["final_score"] > STALLED_SCORE)
        if stalled.any():
            raise ValueError(
                f"{summary.loc[index, 'cell']}: a kept row records a non-converged fit above "
                f"{STALLED_SCORE:g}; such a fit is a recordable failure"
            )
        nonconverged = int((kept["converged"] == 0).sum())
        lost = int(group["failed_replicates"].max())
        requested = int(group["requested_replicates"].max())
        suspect = nonconverged + lost > NONCONVERGED_SUSPECT_SHARE * requested
        summary.loc[index, "nonconverged_fits"] = float(nonconverged)
        summary.loc[index, "max_final_score"] = float(kept["final_score"].max())
        summary.loc[index, "unexpected_warning_fits"] = float(
            (kept["unexpected_warnings"] > 0).sum()
        )
        summary.loc[index, "convergence_defect_suspect"] = float(suspect)
        if suspect and summary.loc[index, "role"] != DIAGNOSTIC_ROLE:
            summary.loc[index, "passed"] = False


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the study's declared rules, and the reported rows."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "paired_sd_over_bound", *CONVERGENCE_SUMMARY),
        rate_labels=study.LADDER_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    difference_readings(summary)
    convergence_reading(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration rule takes this study's efficiency
#: band as an explicit argument.  ``iterated_mean_difference`` states no verdict, so no rule
#: reads it.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
