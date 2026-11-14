"""Property families of ``hazard-time-varying``.

Every fit is ``LTMLE`` with ``survival_model="hazard"`` and ``targeting="one_step"`` on a law of
:mod:`tests.studies.hazard_time_varying_law`, a baseline treatment held over the nodes with a
binary covariate :math:`L_k` at every node after the first.  A cell label names one estimate of
one fit label of :data:`~tests.studies.canonical_hazard_time_varying.FITS`: ``<fit>_contrast``
is the regimen minus the reference in the risk of the first cause by the last node, and
``<fit>_risk`` the regimen's risk.  A joint label ``<fit>_curve`` reads every estimate of one fit
under the simultaneous band.  Every truth is an exact sum over the law's support and every
efficiency bound is exact (:func:`~tests.studies.hazard_time_varying_law.efficiency_sd`).

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``interval_calibration``        the contrast and the risk of each of the four fit labels,
                                with the shrunken-SE and noise controls on
                                ``one_cause_contrast``; the calibration count of fits per
                                fit label at n = 2,000
``double_robustness``           ``one_cause_contrast`` in five nuisance configurations, and
                                the ``mechanism_correct`` configuration on
                                ``competing_contrast`` and ``three_arm_contrast``; 1,200 at
                                n = 2,000, each arm with a wrong learner as its pilot sizes it
``targeting_necessity``         ``one_cause_contrast``: the targeted contrast of the
                                ``double_robustness`` ``mechanism_correct`` fits, and the
                                untargeted backward plug-in of the same draws
``root_n_and_efficiency``       ``one_cause_contrast`` and ``one_cause_risk`` at n = 1,000,
                                2,000 and 8,000; 800 each.  The 1,000 rung is a rate rung
                                that carries no claim
``power``                       ``one_cause_contrast`` against zero, read on the
                                ``one_cause`` calibration fits
``simultaneous_coverage``       the band over every estimate of the ``one_cause`` and the
                                ``competing`` calibration fits, and its pointwise control
``iterated_mean_difference``    reported, no verdict: :data:`DIFFERENCE_REPLICATES` draws at
                                n = 2,000 of ``one_cause`` and ``competing``.  Each draw fits
                                the hazard fit and the iterated-mean fit of the same regimens
                                without and with ``isotonic=True``, and records the paired
                                differences, each fit's coverage and SE ratio, whether the
                                iterated-mean curve decreases and, on ``competing``, each
                                fit's largest sum of cumulative incidences minus one.  On the
                                ``double_robustness`` ``mechanism_correct`` draws: the paired
                                difference of the two fits with regressions that omit ``L``
==============================  ===========================================================

The wrong hazard and continuation learners omit ``L`` (:class:`~tests.studies.
hazard_time_varying_law.NoL`).  The drop-out reads :math:`L_k`, so their plug-in treats it as
ignorable given :math:`(W, A)`, and its contrast lies 4.2 efficiency standard errors from the
truth at n = 2,000 on ``one_cause``, 3.2 on ``competing`` and 3.6 on ``three_arm``
(``tests/unit/test_hazard_time_varying_study_design.py``).  The wrong mechanism is the
censoring model without :math:`L_k`; the treatment is baseline and its model is correct in
every configuration.

**What each robustness arm witnesses.**  With a hazard that omits :math:`L`, every fitted
continuation is a function of :math:`(W, A)` and the covariate term of the walk is zero before
targeting, so the ``mechanism_correct`` arms and the targeted necessity arm witness the hazard
fluctuation, whose weights read :math:`L_k`.  With the hazard correct the hazard term is zero,
so the ``continuation_wrong`` arm (hazard correct, continuation without :math:`L`, mechanism
correct; plug-in 1.37 efficiency SEs from the truth) witnesses the covariate term alone.  The
``both_wrong`` control and the untargeted plug-in read one bias, the bias of treating the
drop-out as ignorable given :math:`(W, A)`: with every factor a function of :math:`(W, A)` the
walk barely moves, so they are not two independent witnesses.  The ``competing`` and
``three_arm`` arms witness the two-cause soft-label panel and the policy ratio times the
retention that reads :math:`L_k`.

**Shared fits.**  The necessity cells read the ``one_cause`` ``mechanism_correct`` robustness
fits, adding only the untargeted pass of the same draw, and the power cell reads the
``one_cause`` calibration fits, as ``hazard-multinode`` does.

**The agreement reading** is reported with no floor (plan X70 rev 2.6 and open question 11).
Both fits are efficient, so where both nuisance fits are correct they differ by
:math:`o_P(n^{-1/2})`, and X70 reports their agreement instead of requiring a gap.  Each
``paired_difference`` row records the hazard estimate minus the iterated-mean estimate of the
same draw (``isotonic_difference`` the same against the isotonic fit); its summary adds
``paired_sd_over_bound``.  The ``iterated_curve_decrease`` row records the largest decrease of
an iterated-mean curve between consecutive nodes.

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost more than its bound.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_hazard_time_varying as study
from tests.studies import hazard_time_varying_law as law
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

STUDY = study.STUDY
#: The replicates of a positive robustness arm, and of each wrong-learner arm until its
#: 800-draw pilot sizes it under the rule of plan X70 rev 2.7 (the design test recomputes it
#: once the pilot is committed).
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
DOUBLE_ROBUST_ARMS = study.DOUBLE_ROBUST_ARMS
#: The arms that fit a wrong outcome learner with the correct mechanism, each with its pilot
#: label and its replicates: provisional until the pilot label is committed.
WRONG_LEARNER_ARMS: dict[tuple[str, str], str] = {
    ("one_cause", "mechanism_correct"): "wrong_learner_one_cause",
    ("one_cause", "continuation_wrong"): "wrong_learner_continuation",
    ("competing", "mechanism_correct"): "wrong_learner_competing",
    ("three_arm", "mechanism_correct"): "wrong_learner_three_arm",
}
WRONG_LEARNER_REPLICATES: dict[tuple[str, str], int] = dict.fromkeys(WRONG_LEARNER_ARMS, 1_200)
#: The robustness fits the necessity family reads.
NECESSITY_ARM = ("one_cause", "mechanism_correct")
NECESSITY_N = DOUBLE_ROBUST_N
RATE_REPLICATES = 800
RATE_SIZES = study.RATE_SIZES
CALIBRATION_N = 2_000
#: The arm-pooled rule over the eight calibration arms, read from the committed calibration
#: pilot (the design test).
CALIBRATION_REPLICATES = 4_800
#: The power cell reads the ``one_cause`` calibration fits.
POWER_REPLICATES = CALIBRATION_REPLICATES
POWER_N = CALIBRATION_N
DIFFERENCE_REPLICATES = 800
DIFFERENCE_N = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The declared floor on the targeted arm's displacement from the plug-in, in the targeted
#: arm's empirical standard deviations.
TARGETING_DISPLACEMENT = 0.25
#: The influence-curve gap of a fit label: the SD of its influence curve minus the exact
#: per-unit efficient influence function, over the efficiency bound.  The claim is the halving
#: rule: the large-size gap is at most :data:`IC_GAP_FALL` times the median over
#: :data:`IC_GAP_DRAWS` draws at the small size.  :data:`IC_GAP_BOUND` holds loose sanity
#: bounds, taken from ``hazard-mtp`` and fixed before this study's pilot.
IC_GAP_SIZES = (2_000, 20_000)
IC_GAP_DRAWS = 5
IC_GAP_BOUND = {2_000: 0.30, 20_000: 0.10}
IC_GAP_FALL = 0.5
CONFIGURATIONS = tuple(study.CONFIGURATIONS)
DIFFERENCE_FAMILY = "iterated_mean_difference"
MECHANISM_DIFFERENCE = "mechanism_correct_difference"
CONTRAST = study.CONTRAST_LABEL
FITS = study.FITS
ARMS = ("contrast", "risk")
#: The per-row record of each fit, and the summary columns that count it.  The influence
#: measures are the smallest Kish count, largest relative SD and largest Edgeworth loss over
#: the fit's estimates (:class:`~tests.studies.canonical_hazard_time_varying.FitRecord`).
INFLUENCE_COLUMNS = ("smallest_kish", "largest_relative_sd", "largest_edgeworth_loss")
CONVERGENCE_COLUMNS = ("converged", "final_score", "unexpected_warnings", *INFLUENCE_COLUMNS)
CONVERGENCE_SUMMARY = (
    "nonconverged_fits",
    "max_final_score",
    "unexpected_warning_fits",
    "convergence_defect_suspect",
)
#: A fit that does not converge and whose final relative score is above this is a recordable
#: failure (:func:`stalled_fit`); a miss at or below it is the walk's reporting floor.
STALLED_SCORE = 1e-6
#: A cell whose non-converged fits exceed this share of its requested replicates is a defect
#: suspect (:func:`convergence_reading`).
NONCONVERGED_SUSPECT_SHARE = 0.01


def _labels(fit_label: str) -> tuple[tuple[str, str], ...]:
    """``(cell label, reported name)`` of the contrast and the risk of one fit label."""
    law_name = FITS[fit_label][0]
    last = law.LAWS[law_name].n_times
    return (
        (f"{fit_label}_contrast", study.contrast_name(law_name, last)),
        (f"{fit_label}_risk", study.risk_name(law_name, study.labels(law_name)[0], last)),
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
    return study.target_of(FITS[FIT_OF[label]][0], ESTIMAND[label])


def _exact() -> tuple[dict[str, float], dict[str, float]]:
    """The exact truth and efficiency bound of every cell label, each law read once."""
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


def iterated_fit(
    label: str,
    frame: pd.DataFrame,
    *,
    isotonic: bool = False,
    regression_correct: bool = True,
) -> Any:
    """The iterated-mean fit of the same regimens on one fit label's sample.

    Each node's regression is the cell mean over :math:`(W, L_k)` among the regimen's followers
    (the design an iterated-mean fit hands its learners holds no treatment column), or over
    :math:`W` alone when ``regression_correct`` is false.  On these Markov laws the first is
    the true regression.  The mechanism learners are the hazard fit's correct ones.
    """
    law_name, folds = FITS[label]
    regression = law.continuation_learner(regression_correct)
    mechanism = study.learners("both_correct")
    del mechanism["outcome_learner"], mechanism["pseudo_learner"]
    return LTMLE(
        study.regimens(law_name),
        reference=study.labels(law_name)[1],
        n_folds=folds,
        simultaneous=False,
        survival_model="iterated_mean",
        isotonic=isotonic,
        targeting="iterative",
        max_iter=100,
        tol=study.TOL,
        random_state=0,
        outcome_learner=regression,
        pseudo_learner=law.continuation_learner(regression_correct),
        **mechanism,
    ).fit(frame, **law.LAWS[law_name].columns())


def _curve_names(law_name: str, label: str, cause: int) -> list[str]:
    return [
        study.risk_name(law_name, label, t, cause) for t in range(1, law.LAWS[law_name].n_times + 1)
    ]


def largest_decrease(result: Any, law_name: str) -> tuple[float, float]:
    """The largest decrease of a fit's risk curves between nodes, and its standard error.

    Over both regimens, every cause and every node; negative when every curve rises.
    """
    best: tuple[float, float] | None = None
    current = law.LAWS[law_name]
    for label in study.labels(law_name):
        for cause in range(1, current.causes + 1):
            names = _curve_names(law_name, label, cause)
            for before_name, after_name in itertools.pairwise(names):
                before, after = result[before_name], result[after_name]
                drop = float(before.psi - after.psi)
                curve = np.asarray(before.influence_curve) - np.asarray(after.influence_curve)
                error = float(np.std(curve) / np.sqrt(curve.size))
                if best is None or drop > best[0]:
                    best = (drop, error)
    assert best is not None
    return best


def largest_excess(result: Any, law_name: str) -> tuple[float, float]:
    """The largest sum of the cumulative incidences minus one, and its standard error.

    Over both regimens and every horizon; negative when the curves are coherent with room.
    """
    best: tuple[float, float] | None = None
    current = law.LAWS[law_name]
    for label in study.labels(law_name):
        for t in range(1, current.n_times + 1):
            risks = [
                result[study.risk_name(law_name, label, t, j)] for j in range(1, current.causes + 1)
            ]
            excess = sum(float(risk.psi) for risk in risks) - 1.0
            curve = np.sum([np.asarray(risk.influence_curve) for risk in risks], axis=0)
            error = float(np.std(curve) / np.sqrt(curve.size))
            if best is None or excess > best[0]:
                best = (excess, error)
    assert best is not None
    return best


def draw(fit_label_: str, n: int, seed: int) -> pd.DataFrame:
    return study.law_of(fit_label_).draw(n, seed)


# ------------------------------------------------------------------ the agreement reading


def fits_differ_reading(
    paired_ratios: dict[str, float], decreasing: int, draws: int, level: float = 0.99
) -> dict[str, Any]:
    """The reported agreement of the hazard and iterated-mean fits; it carries no floor.

    ``paired_ratios`` maps each cell label to its paired-difference SD over the efficiency-bound
    standard error; ``decreasing`` counts the draws whose iterated-mean curve without isotonic
    decreases somewhere, out of ``draws``.  The reading returns the ratios, the largest of them,
    the decreasing share with its Clopper-Pearson interval at ``level``, and ``text``.
    """
    tail = (1.0 - level) / 2.0
    low = float(beta.ppf(tail, decreasing, draws - decreasing + 1)) if decreasing else 0.0
    high = (
        float(beta.ppf(1.0 - tail, decreasing + 1, draws - decreasing))
        if decreasing < draws
        else 1.0
    )
    share = decreasing / draws
    ratios = ", ".join(f"{label} {value:.3f}" for label, value in sorted(paired_ratios.items()))
    text = (
        f"Paired-difference SD over the efficiency SE: {ratios}. Decreasing iterated-mean "
        f"curves: {decreasing} of {draws}, {share:.3f} with {level:.0%} interval "
        f"[{low:.3f}, {high:.3f}]. Reported, with no floor."
    )
    return {
        "ratios": dict(paired_ratios),
        "largest_ratio": max(paired_ratios.values()),
        "share": share,
        "share_interval": (low, high),
        "text": text,
    }


# ------------------------------------------------------------------ the grid


def dr_stream(fit: str, configuration: str) -> str:
    """The stream of one robustness arm: the configuration on ``one_cause``, else prefixed."""
    return configuration if fit == "one_cause" else f"{fit}_{configuration}"


def dr_replicates(fit: str, configuration: str) -> int:
    """The replicates of one robustness arm (:data:`WRONG_LEARNER_REPLICATES` or the default)."""
    return WRONG_LEARNER_REPLICATES.get((fit, configuration), DOUBLE_ROBUST_REPLICATES)


#: Every fit set: ``(family, stream label, configuration, n, replicates, fit label)``.  The
#: necessity cells ride on the ``one_cause`` ``mechanism_correct`` robustness fits and the
#: power cell on the ``one_cause`` calibration fits, so neither has a fit set of its own.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("interval_calibration", fit, "both_correct", CALIBRATION_N, CALIBRATION_REPLICATES, fit)
        for fit in CALIBRATION_FITS
    ),
    *(
        ("double_robustness", dr_stream(fit, c), c, DOUBLE_ROBUST_N, dr_replicates(fit, c), fit)
        for fit, c in DOUBLE_ROBUST_ARMS
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "one_cause")
        for size in RATE_SIZES
    ),
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
        self.name = f"hazard_time_varying__{fit_label}"

    def truth(self) -> dict[str, float]:
        return {ESTIMAND[label]: TRUTHS[label] for label, _ in COEFFICIENTS[self.fit_label]}


class DeclaredDifference:
    """The zero every ``iterated_mean_difference`` difference row is recorded against."""

    name = "hazard_time_varying__iterated_mean_difference"

    def truth(self) -> dict[str, float]:
        return {"paired_difference": 0.0, "largest_decrease": 0.0, "largest_excess": 0.0}


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
    one_cause = DeclaredEstimates("one_cause")
    for fit, configuration in DOUBLE_ROBUST_ARMS:
        add(
            "double_robustness",
            f"{fit}_contrast__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            DeclaredEstimates(fit),
            ESTIMAND[f"{fit}_contrast"],
            DOUBLE_ROBUST_N,
            dr_replicates(fit, configuration),
            dr_stream(fit, configuration),
        )
    necessity_replicates = dr_replicates(*NECESSITY_ARM)
    for label, _ in COEFFICIENTS["one_cause"]:
        add(
            DIFFERENCE_FAMILY,
            f"{label}__{MECHANISM_DIFFERENCE}",
            DIAGNOSTIC_ROLE,
            DeclaredDifference(),
            "paired_difference",
            DOUBLE_ROBUST_N,
            necessity_replicates,
            dr_stream(*NECESSITY_ARM),
            draws_of="double_robustness",
        )
    for arm, role in (("targeted", "positive"), ("untargeted", "control")):
        add(
            "targeting_necessity",
            f"{CONTRAST}__{arm}",
            role,
            one_cause,
            ESTIMAND[CONTRAST],
            NECESSITY_N,
            necessity_replicates,
            dr_stream(*NECESSITY_ARM),
            draws_of="double_robustness",
        )
    for size in RATE_SIZES:
        for label, name in COEFFICIENTS["one_cause"]:
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                "control" if size == min(RATE_SIZES) else "positive",
                one_cause,
                name,
                size,
                RATE_REPLICATES,
                f"n_{size}",
            )
    add(
        "power",
        f"{CONTRAST}__alternative",
        "positive",
        one_cause,
        ESTIMAND[CONTRAST],
        POWER_N,
        POWER_REPLICATES,
        "one_cause",
        draws_of="interval_calibration",
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
        if fit in study.COHERENCE_FITS:
            for arm in study.COHERENCE_ARMS:
                add(
                    DIFFERENCE_FAMILY,
                    f"{fit}__{arm}",
                    DIAGNOSTIC_ROLE,
                    difference,
                    "largest_excess",
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
        if label == "one_cause":
            names.append(f"power/{CONTRAST}__alternative")
        return names
    if family == "double_robustness":
        names = [f"{family}/{label}_contrast__{configuration}"]
        if (label, configuration) == NECESSITY_ARM:
            names += [
                f"{DIFFERENCE_FAMILY}/{cell_label}__{MECHANISM_DIFFERENCE}"
                for cell_label, _ in COEFFICIENTS[label]
            ]
            names += [
                f"targeting_necessity/{CONTRAST}__{arm}" for arm in ("targeted", "untargeted")
            ]
        return names
    if family == "root_n_and_efficiency":
        return [f"{family}/{cell_label}__n_{n}" for cell_label, _ in COEFFICIENTS[label]]
    if family == DIFFERENCE_FAMILY:
        names = [
            *(
                f"{family}/{cell_label}__{arm}"
                for cell_label, _ in COEFFICIENTS[label]
                for arm in study.DIFFERENCE_ARMS
            ),
            f"{family}/{label}__iterated_curve_decrease",
        ]
        if label in study.COHERENCE_FITS:
            names += [f"{family}/{label}__{arm}" for arm in study.COHERENCE_ARMS]
        return names
    raise ValueError(f"no fit set of family {family!r}")


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


def _reading_row(
    cell: str, value: float, error: float, common_row: dict[str, Any]
) -> dict[str, Any]:
    """A reported reading recorded against zero: ``rejected`` marks a positive value."""
    positive = int(value > 0.0)
    return {
        "property": DIFFERENCE_FAMILY,
        "cell": cell,
        "role": DIAGNOSTIC_ROLE,
        **common_row,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": value,
        "std_error": error,
        "covered": 1 - positive,
        "rejected": positive,
    }


class FitRecords:
    """The :class:`~tests.studies.canonical_hazard_time_varying.FitRecord` of every fit a key runs."""

    def __init__(self) -> None:
        self.records: list[study.FitRecord] = []

    def hazard(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        result, record = study.fit_recorded(label, frame, **settings)
        self.records.append(record)
        return result

    def untargeted(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        """The untargeted backward pass: its warnings are recorded; it has no walk to converge.

        Its influence measures are not the subject's, so they are not recorded.
        """
        result, record = study.recorded(lambda: study.untargeted_fit(label, frame, **settings))
        self.records.append(study.FitRecord(True, record.unexpected_warnings))
        return result

    def iterated(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        """An iterated-mean fit: its warnings are recorded, its convergence is not the subject's."""
        result, record = study.recorded(lambda: iterated_fit(label, frame, **settings))
        self.records.append(study.FitRecord(True, record.unexpected_warnings))
        return result

    def columns(self) -> dict[str, float]:
        def extreme(values: list[float], pick: Any) -> float:
            finite = [value for value in values if np.isfinite(value)]
            return float(pick(finite)) if finite else float("nan")

        hazards = [r for r in self.records if np.isfinite(r.smallest_kish)]
        return {
            "converged": int(all(record.converged for record in self.records)),
            "final_score": max((r.final_score for r in self.records), default=0.0),
            "unexpected_warnings": sum(len(r.unexpected_warnings) for r in self.records),
            "smallest_kish": extreme([r.smallest_kish for r in hazards], min),
            "largest_relative_sd": extreme([r.largest_relative_sd for r in hazards], max),
            "largest_edgeworth_loss": extreme([r.largest_edgeworth_loss for r in hazards], max),
        }


def stalled_fit(rows: pd.DataFrame) -> str | None:
    """The failure rule of a key whose fit stalled: non-converged above :data:`STALLED_SCORE`."""
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
    law_name = FITS[label][0]
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
    drop, error = largest_decrease(iterated, law_name)
    rows.append(_reading_row(f"{label}__iterated_curve_decrease", drop, error, common_row))
    if label in study.COHERENCE_FITS:
        for arm, result in zip(study.COHERENCE_ARMS, (hazard, iterated, isotonic), strict=True):
            excess, error = largest_excess(result, law_name)
            rows.append(_reading_row(f"{label}__{arm}", excess, error, common_row))
    return rows


def mechanism_difference_rows(
    frame: pd.DataFrame, hazard: Any, common_row: dict[str, Any], label: str, records: FitRecords
) -> list[dict[str, Any]]:
    """The paired difference of the two consistent, misspecified-regression fits of one draw.

    ``hazard`` is the draw's ``mechanism_correct`` hazard fit (hazard and continuation without
    ``L``, correct mechanism).  The iterated-mean fit of the same regimens uses regressions
    without ``L`` and the same correct mechanism.
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

    Every row carries :data:`CONVERGENCE_COLUMNS` of the fits the key ran.
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
        if label == "one_cause":
            rows.append(
                replicate_row(
                    property_name="power",
                    cell=f"{CONTRAST}__alternative",
                    role="positive",
                    truth=TRUTHS[CONTRAST],
                    estimate=result[ESTIMAND[CONTRAST]],
                    alpha=STUDY.margins.alpha,
                    **shared,
                )
            )
        return rows
    result = records.hazard(label, frame, configuration=configuration)
    if family == "root_n_and_efficiency":
        role = "control" if n == min(RATE_SIZES) else "positive"
        return [
            row(cell_label, f"{cell_label}__n_{n}", role, result[name])
            for cell_label, name in COEFFICIENTS[label]
        ]
    if family != "double_robustness":
        raise ValueError(f"no fit set of family {family!r}")
    contrast = f"{label}_contrast"
    estimate = result[ESTIMAND[contrast]]
    role = "control" if configuration == "both_wrong" else "positive"
    rows = [row(contrast, f"{contrast}__{configuration}", role, estimate)]
    if (label, configuration) == NECESSITY_ARM:
        rows += mechanism_difference_rows(frame, result, common_row, label, records)
        plug_in = records.untargeted(label, frame, configuration=configuration)
        necessity = {**shared}
        rows += [
            replicate_row(
                property_name="targeting_necessity",
                cell=f"{CONTRAST}__targeted",
                role="positive",
                truth=TRUTHS[CONTRAST],
                estimate=estimate,
                alpha=STUDY.margins.alpha,
                **necessity,
            ),
            control_row(
                property_name="targeting_necessity",
                cell=f"{CONTRAST}__untargeted",
                truth=TRUTHS[CONTRAST],
                estimate=float(plug_in[ESTIMAND[CONTRAST]].psi),
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
                **necessity,
            ),
        ]
    return rows


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


#: The share of a cell's requested replications that the accounting lets a run lose.
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
    """The standard deviation of a synthetic cell's estimates for the property differential."""
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
    over the estimate's efficiency-bound standard error at the row's ``n``.  Every row reads
    ``passed`` and ``property_passed`` as true; its ``diagnostic`` role says no margin was read.
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

    As in ``hazard-mtp``: ``convergence_defect_suspect`` is true when the kept non-converged
    fits plus the cell's lost replicates exceed :data:`NONCONVERGED_SUSPECT_SHARE` of its
    requested replicates, and a suspect gated cell does not pass.

    Raises
    ------
    ValueError
        When a kept row records a non-converged fit above :data:`STALLED_SCORE`.
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


#: The failure accounting's declared rules.  ``iterated_mean_difference`` states no verdict,
#: so no rule reads it.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
