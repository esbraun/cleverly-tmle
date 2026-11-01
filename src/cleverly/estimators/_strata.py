"""Shared pieces of the baseline-strata targeting.

A stratum parameter is the marginal parameter of the law of ``O`` given ``S = s``, for a fixed
finite partition ``S`` of the adjustment columns with ``P(S = s) > 0``.  Every target group
solves one score block ``I(S = s) H_s / P_n(S = s)`` per stratum.  The helpers here read the
stratum masses and refuse, before any learner, a stratum whose score equations have no finite
root.  The fold-evaluated report with strata (``cv_evaluation=True``, X28) reads its target list,
its data refusals, its fluctuation weights and its marginal mixture from here too.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .._refusal import RefusalReason
from .._typing import BoolArray, FloatArray, IntArray
from ..data.causal_data import CausalData, arm_share
from ..exceptions import CapabilityError, DataError
from ..inference.influence import ParameterEstimate

__all__ = [
    "FOLD_EVALUATED_STRATA_TARGETS",
    "absent_arm_error",
    "check_stratified_msm_design",
    "check_stratified_targets",
    "companion_stratum_codes",
    "embed_stratum_curve",
    "fold_evaluated_strata_refusal",
    "fold_evaluated_stratum_weights",
    "refuse_absent_tilt_arms",
    "refuse_empty_stratum_folds",
    "refuse_untrainable_stratum_folds",
    "stratum_mixture",
    "stratum_probabilities",
    "stratum_source",
    "unscale_block",
]


def stratum_probabilities(data: CausalData) -> list[float]:
    """``P_n(S = s)`` under the fit's weights, for each stratum code in order."""
    assert data.strata is not None
    return [
        float(np.average(data.strata == code, weights=data.weights))
        for code in range(data.n_strata)
    ]


def unscale_block(values: FloatArray, probability: float) -> FloatArray:
    """Undo a block's ``1 / P_n(S = s)`` scale, so the block is the stratum's own covariate.

    The ordinary target builders then read the block on the stratum's rows as the marginal
    covariate of the stratum's law.
    """
    return values * probability


def embed_stratum_curve(curve: FloatArray, index: IntArray, n: int) -> FloatArray:
    """Put a stratum curve on all ``n`` rows: ``I(S = s) D_s / P_n(S = s)``.

    ``curve`` is the influence curve of the stratum's law on its ``n_s`` rows, whose
    empirical scale is ``n_s``.  Multiplying by ``n / n_s`` gives the full-law gradient,
    including the mass term, on the full sample's scale.  Rows outside the stratum are zero.
    """
    out = np.zeros(n, dtype=float)
    out[index] = curve * (n / index.size)
    return out


def companion_stratum_codes(data: CausalData, companion: CausalData) -> IntArray:
    """The fit's stratum code of each companion row, matched by stratum label.

    The companion is prepared with the fit's stratum columns, but its own codes follow its
    own rows.  This maps each companion row's stratum levels to the fit's code for them.

    Parameters
    ----------
    data : CausalData
        The fitting data, with baseline strata.
    companion : CausalData
        The evaluation rows, prepared with the same stratum columns.

    Returns
    -------
    IntArray
        One fit code per companion row.

    Raises
    ------
    DataError
        If the companion has no strata, or holds a stratum the fit does not.
    """
    if companion.strata is None or companion.strata_names != data.strata_names:
        raise DataError(
            "the evaluation companion of a stratified fit must be prepared with the fit's "
            f"baseline strata {list(data.strata_names)}"
        )
    lookup = {levels: code for code, levels in enumerate(data.strata_levels)}
    absent = [levels for levels in companion.strata_levels if levels not in lookup]
    if absent:
        raise DataError(
            f"the evaluation companion holds baseline strata {absent} that the fitting rows "
            "do not; a stratum's reduced regressions cannot predict outside their stratum"
        )
    mapped = np.array([lookup[levels] for levels in companion.strata_levels], dtype=np.int64)
    return np.asarray(mapped[np.asarray(companion.strata)], dtype=np.int64)


def stratum_source(fluctuation: Any) -> Any:
    """The fluctuation a group's stratum estimates read.

    A linked working model nests its stratum fluctuation on the marginal one
    (:attr:`~cleverly.fluctuation.Fluctuation.stratified`).  Every other group solves its
    stratum blocks in the group's one fluctuation.
    """
    return fluctuation if fluctuation.stratified is None else fluctuation.stratified


def absent_arm_error(
    data: CausalData,
    arms: Sequence[float],
    fractions: FloatArray,
    code: int,
    group: str,
) -> DataError:
    """The refusal of a stratum that holds no positive-weight row of some arm.

    Parameters
    ----------
    data : CausalData
        The data of the fit.
    arms : sequence of float
        The arm codes, in the order of ``fractions``.
    fractions : FloatArray
        The within-stratum share of each arm.
    code : int
        The stratum code.
    group : str
        The target group.  The ``ipsi`` group names its treatment-mechanism equation.

    Returns
    -------
    DataError
        The refusal.
    """
    absent = [
        data.arm_label(arm)
        for arm, fraction in zip(arms, fractions, strict=True)
        if fraction <= 0.0
    ]
    label = data.stratum_label(code)
    if group == "ipsi":
        return DataError(
            f"baseline stratum {label} has no positive-weight rows of arm(s) {absent}, so the "
            "stratum's treatment-mechanism score equation may have no finite root"
        )
    return DataError(
        f"baseline stratum {label} contains no positive-weight observations from treatment "
        f"arm(s) {absent}; its empirical targeting score is unidentified"
    )


def refuse_absent_tilt_arms(data: CausalData) -> None:
    """Refuse an incremental fit with a stratum that lacks a treatment arm.

    Within a stratum the mechanism step maximises the concave log-likelihood
    :math:`\\sum_i w_i \\log g_\\epsilon(A_i \\mid W_i)`, with :math:`g_\\epsilon(a \\mid W)
    \\propto \\hat g(a \\mid W)\\exp\\{\\epsilon^\\top H(a, W)\\}`.  Its score equation has
    no finite root only if some direction :math:`d \\ne 0` separates: every observed level
    maximises :math:`d^\\top H(\\cdot, W_i)` in its row.  An absent level is neither necessary
    nor sufficient for that.  With a single tilt the condition reads: every observed level is
    the arg-max, or every one is the arg-min, of :math:`H(\\cdot, W_i)`.  :math:`H` holds the
    fitted outcome regression, so the condition cannot be read before the learners run.  The
    check reads the data only and refuses a stratum without a level, as the binary fit always
    did, and the refusal says the equation *may* have no root.  A relaxed rule would need the
    separation check after the outcome fit and a defined estimate when it fails.

    Parameters
    ----------
    data : CausalData
        The data of the fit.  The check returns at once without baseline strata.

    Raises
    ------
    DataError
        If a stratum holds no positive-weight row of some arm.
    """
    if not data.has_strata or data.is_continuous_treatment:
        return
    assert data.strata is not None
    arms = tuple(data.arm_codes)
    for code in range(data.n_strata):
        mask = data.strata == code
        fractions = np.array(
            [arm_share(data.treatment, data.weights, arm, mask=mask) for arm in arms], dtype=float
        )
        if np.any(fractions <= 0.0):
            raise absent_arm_error(data, arms, fractions, code, "ipsi")


#: One requirement of an outcome-adaptive design family: the rows that must exist in every
#: training complement of a stratum, and the clause that states their absence in sample.
Requirement = tuple[BoolArray, str]


def refuse_untrainable_stratum_folds(
    data: CausalData,
    folds: Any,
    *,
    family: str = "reduced regression",
    requirements: Callable[[float], Sequence[Requirement]] | None = None,
) -> None:
    """Refuse a fit whose stratum-local regressions cannot train in some stratum and fold.

    A stratified ``DRTMLE`` fits each reduced regression inside each stratum, on the rows
    of that stratum that its family trains on and that each fold's training complement
    holds.  The narrowest family trains on the rows of one arm with an observed outcome:
    ``Q_r`` on the complete-data and composite routes, and ``e`` on the missing-outcome
    route.  A per-arm outcome-adaptive :class:`~cleverly.CTMLE` fits each factor of its
    design inside each stratum in the same way, and inside the whole sample without
    strata, so it passes its own ``family`` and ``requirements``.  The check reads the
    data and the drawn folds only, so it runs after the fold draw and before any learner.

    Parameters
    ----------
    data : CausalData
        The data of the fit.  For the default family, the check returns at once without
        baseline strata.
    folds : Folds
        The drawn split.
    family : str
        ``"reduced regression"`` (DR-TMLE, the default), ``"treatment design"`` or
        ``"observation design"``.
    requirements : callable or None
        For a family other than the default: maps an arm code to the ``(rows, absence)``
        pairs that must each hold a row in every training complement.  ``absence`` is the
        clause the in-sample message states, for example ``"no recorded row is in arm
        '1'"``.

    Raises
    ------
    ValueError
        If some stratum, arm and training complement hold no row of a requirement.  The
        default family raises the shipped no-trainable-rows refusal with the stratum named
        first.
    """
    from ..learners.crossfit import _POST_DRAW_REMEDY

    default = family == "reduced regression"
    if data.is_continuous_treatment or (default and data.strata is None):
        return
    treatment = np.asarray(data.treatment, dtype=float)
    observed = np.asarray(data.observed, dtype=bool)
    every = np.arange(data.n)
    complements = [every] if folds.is_single else [train for train, _ in folds]
    if data.strata is None:
        regions = [("the sample", np.ones(data.n, dtype=bool))]
    else:
        regions = [
            (f"baseline stratum {data.stratum_label(code)}", data.strata == code)
            for code in range(data.n_strata)
        ]
    stratified = data.strata is not None
    for where, inside in regions:
        for arm in data.arm_codes:
            if default:
                trainable = inside & (treatment == float(arm)) & observed
                if any(not np.any(trainable[train]) for train in complements):
                    raise ValueError(
                        f"inside {where}: a cross-fitting "
                        "fold has no trainable rows for a reduced regression of arm "
                        f"{data.arm_label(arm)!r}. "
                        + _POST_DRAW_REMEDY.format(
                            remedy="declare fewer baseline strata, or fit with cross_fit=False"
                        )
                    )
                continue
            assert requirements is not None
            for rows, absence in requirements(arm):
                trainable = inside & np.asarray(rows, dtype=bool)
                if all(np.any(trainable[train]) for train in complements):
                    continue
                if folds.is_single:
                    message = (
                        f"inside {where}: {absence}, so the outcome-adaptive {family} of arm "
                        f"{data.arm_label(arm)!r} cannot be fitted."
                    )
                    if stratified:
                        message += " Declare fewer baseline strata."
                    raise ValueError(message)
                remedy = (
                    "declare fewer baseline strata, or fit with cross_fit=False"
                    if stratified
                    else "fit with cross_fit=False"
                )
                raise ValueError(
                    f"inside {where}: a cross-fitting fold has no trainable rows for the "
                    f"outcome-adaptive {family} of arm {data.arm_label(arm)!r}. "
                    + _POST_DRAW_REMEDY.format(remedy=remedy)
                )


def refuse_untrainable_leave_two_out(data: CausalData, folds: Any) -> None:
    """Refuse a foldwise fit whose leave-two-out complements cannot train every arm.

    The foldwise DR-TMLE slabs fit the primary regressions once for each unordered pair of
    folds ``{k, m}``, on the rows outside both.  Each of those fits trains on the whole
    sample outside the pair, so it needs a row of every arm with an observed outcome there.
    Baseline strata do not enter: the primary regressions are not fitted inside a stratum,
    and a stratum's reduced regressions for fold ``k`` train on the rows of the fold-``k``
    complement, which :func:`refuse_untrainable_stratum_folds` checks.  A cluster split
    removes whole clusters, so a few-cluster split can empty an arm here while every
    one-fold complement still holds it.  The check reads the data and the drawn folds only,
    so it runs after the fold draw and before any learner.

    Parameters
    ----------
    data : CausalData
        The data of the fit.
    folds : Folds
        The drawn split, with at least three folds.

    Raises
    ------
    ValueError
        If some arm and pair of folds leave no trainable row outside the pair.
    """
    from ..learners.crossfit import _POST_DRAW_REMEDY

    if data.is_continuous_treatment or folds.is_single:
        return
    treatment = np.asarray(data.treatment, dtype=float)
    observed = np.asarray(data.observed, dtype=bool)
    assignment = np.asarray(folds.assignment)
    pairs = [
        (first, second)
        for first in range(folds.n_folds)
        for second in range(first + 1, folds.n_folds)
    ]
    for arm in data.arm_codes:
        trainable = (treatment == float(arm)) & observed
        for first, second in pairs:
            outside = (assignment != first) & (assignment != second)
            if np.any(trainable & outside):
                continue
            raise ValueError(
                f"folds {first} and {second} together hold every trainable row of arm "
                f"{data.arm_label(arm)!r} in the sample, so the leave-two-out primary "
                "regressions of reduced_crossfit='foldwise' cannot be fitted. "
                + _POST_DRAW_REMEDY.format(remedy="fit with reduced_crossfit='pooled'")
            )


def check_stratified_msm_design(msm: object, data: CausalData) -> None:
    """Refuse a working model whose projection is singular inside a baseline stratum.

    A stratum coefficient vector is the projection on the stratum's rows, so its Gram must
    have full rank there.  A term that is constant in a stratum, such as the stratum
    column itself, or one that equals another there, such as ``a:S`` in the stratum
    ``S = 1``, makes it singular.  The check evaluates the model on the fit's rows and
    restricts it to each stratum, which reruns the shipped rank rule
    (:func:`~cleverly.msm.check_projection_rank`).  A link does not change the rank,
    because ``dm/deta > 0``.  The check reads the declaration and the data only, so it
    runs before any learner.

    Parameters
    ----------
    msm : MSM
        The declared working model.
    data : CausalData
        The data of the fit.  The check returns at once without baseline strata.

    Raises
    ------
    DataError
        If the projection is singular inside a stratum.  The message is the shipped rank
        message with the stratum named first.
    """
    from ..msm import MSM, MSMSet

    if not data.has_strata:
        return
    assert isinstance(msm, MSM)
    assert data.strata is not None
    evaluated = MSMSet.evaluate(msm, data)
    for code in range(data.n_strata):
        rows = np.flatnonzero(data.strata == code).astype(np.int64)
        try:
            evaluated.subset(rows)
        except DataError as error:
            inner = error.refusal
            raise DataError.from_refusal(
                replace(
                    inner,
                    condition=f"inside baseline stratum {data.stratum_label(code)}: "
                    f"{inner.condition}",
                )
            ) from error


def check_stratified_targets(data: CausalData, *, incremental: bool, msm: object | None) -> None:
    """Refuse, before any learner, a stratified target whose stratum equations cannot be solved.

    Every target group fits baseline strata.  Two data conditions still refuse: an
    incremental fit with a stratum that lacks a treatment arm
    (:func:`refuse_absent_tilt_arms`), and a working model singular inside a stratum
    (:func:`check_stratified_msm_design`).  :meth:`~cleverly.TMLE.fit` and
    :meth:`~cleverly.CausalStudy.identify` run it.

    Parameters
    ----------
    data : CausalData
        The data of the fit.  The check returns at once without baseline strata.
    incremental : bool
        Whether the fit targets an incremental intervention.
    msm : MSM or None
        The working model of the fit, or ``None``.

    Raises
    ------
    DataError
        If either data condition fails.
    """
    if not data.has_strata:
        return
    if incremental:
        refuse_absent_tilt_arms(data)
    if msm is not None:
        check_stratified_msm_design(msm, data)


# ------------------------------------------------------------------ fold evaluation (X28)

#: The targets a fold-evaluated fit with baseline strata reports: the arm means and their
#: linear contrasts, the regime means and their contrasts, and the learned-rule value
#: (``learned_rule=``). Each is a mean of the targeted regression over the stratum's rows, so its
#: fold plug-in, its fold curve and its solving weights follow one construction. The
#: learned-rule value reads each fold's own rule as the one-hot regime density of the fold's
#: rows, so stratum :math:`s` of fold :math:`v` evaluates the fixed rule :math:`d_v`. Every
#: other target keeps the refusal of :func:`fold_evaluated_strata_refusal`.
FOLD_EVALUATED_STRATA_TARGETS: frozenset[str] = frozenset(
    {"ate", "ey", "ey1", "ey0", "ey_regime", "ate_regime", "ey_learned_rule"}
)

_STACKED_REPORT = "the stacked report (cv_evaluation=False, CrossFitting(fold_evaluation=False))"


def _x28_gap(what: str, why: str, alternative: str) -> CapabilityError:
    return CapabilityError(
        "cv_evaluation=True with baseline strata reports the arm means and their linear "
        "contrasts (ey, ate), the regime means and their contrasts (ey_regime, ate_regime), "
        "and the learned-rule value (ey_learned_rule), "
        f"on complete data with rows as the independent unit. {what} is not implemented: "
        f"{why}.",
        reason=RefusalReason.IMPLEMENTATION_GAP,
        alternative=alternative,
        roadmap="X28",
    )


def fold_evaluated_strata_refusal(
    data: CausalData, estimands: Sequence[str]
) -> CapabilityError | None:
    """The refusal of a fold-evaluated stratified request outside the X28 construction.

    ``cv_evaluation=True`` with baseline strata averages each stratum's fold plug-ins and
    builds each stratum curve with the fold's own stratum share. The construction is derived
    and checked for :data:`FOLD_EVALUATED_STRATA_TARGETS` on complete data with rows as the
    unit. A missing outcome or treatment, an intermediate variable, clusters, and every other
    target return a refusal citing X28. Ratios and MSM coefficients never reach it: the fit
    refuses them under ``cv_evaluation=True`` first, citing X61, with or without strata. The
    check reads the declaration and the data only,
    so :meth:`~cleverly.TMLE.fit` raises it before any learner and
    :meth:`~cleverly.TMLE.retarget` raises it before any targeting step.

    Parameters
    ----------
    data : CausalData
        The prepared data. Without baseline strata the check returns ``None`` at once.
    estimands : sequence of str
        The resolved target names.

    Returns
    -------
    CapabilityError or None
        The refusal to raise, or ``None`` when the request is inside the construction.
    """
    if not data.has_strata:
        return None
    stacked = f"Use {_STACKED_REPORT}."
    if data.has_missing_outcome:
        return _x28_gap(
            "A missing outcome (delta=) beside them",
            "each stratum's fold estimate would also read the observation mechanism inside "
            "every validation fold and stratum, and no witness covers that construction",
            stacked,
        )
    if data.has_missing_treatment:
        return _x28_gap(
            "A missing treatment beside them",
            "each stratum's fold estimate would read the composite mechanism inside every "
            "validation fold and stratum, and no witness covers that construction",
            stacked,
        )
    if data.has_intermediate:
        return _x28_gap(
            "An intermediate variable (intermediate=) beside them",
            "the controlled direct effect inside every validation fold and stratum has no witness",
            stacked,
        )
    if data.cluster is not None:
        return _x28_gap(
            "Clusters (id=) beside them",
            "each stratum's fold variance would compare the cluster totals of its stratum "
            "inside every validation fold, and neither that variance nor its Student t "
            "reference has been checked",
            stacked,
        )
    other = sorted({name for name in estimands if name not in FOLD_EVALUATED_STRATA_TARGETS})
    if other:
        return _x28_gap(
            f"Fold evaluation of {other}",
            "each needs its own fold-evaluated stratum construction (a fold-local arm share "
            "or a second mechanism equation), and none has been derived and checked",
            "Request the arm or regime means and their contrasts (estimands=('ate', 'ey') on "
            f"the arm axis), or use {_STACKED_REPORT}.",
        )
    return None


def refuse_empty_stratum_folds(data: CausalData, folds: Sequence[Any]) -> None:
    """Refuse a fold-evaluated stratified split with an empty stratum or arm in some fold.

    The fold-evaluated stratum estimate is the ``1/V`` average of plug-ins over the rows of
    each validation fold in the stratum, and its curve divides by the fold's own stratum
    share. A fold that holds no positive-weight row of a stratum has no plug-in and no share.
    A fold whose stratum holds no positive-weight row of some arm leaves that arm's stratum
    score without a contribution from the fold, as the whole-sample check of
    :func:`absent_arm_error` refuses for a stratum. The stratum targets read every arm, so
    every arm is required. The check reads the data and every drawn split only, so it runs
    after the fold draw and before any learner.

    Parameters
    ----------
    data : CausalData
        The data of the fit. Without baseline strata the check returns at once.
    folds : sequence of Folds
        The drawn or supplied split of every repeat.

    Raises
    ------
    DataError
        If some validation fold holds no positive-weight row of a stratum, or none of an arm
        inside a stratum.
    """
    from ..learners.crossfit import _POST_DRAW_REMEDY

    if not data.has_strata:
        return
    assert data.strata is not None
    strata = np.asarray(data.strata)
    weights = np.asarray(data.weights, dtype=float)
    treatment = np.asarray(data.treatment, dtype=float)
    arms = tuple(data.arm_codes) if data.has_arms else ()
    positive = " positive-weight" if data.is_weighted else ""
    remedy = _POST_DRAW_REMEDY.format(
        remedy=f"use {_STACKED_REPORT}, declare fewer baseline strata"
    )
    draws = [draw for draw in folds if not draw.is_single]
    # An empty stratum is the more basic shortfall, so every fold is read for it first.
    for draw in draws:
        for fold, (_, test) in enumerate(draw):
            for code in range(data.n_strata):
                if float(np.sum(weights[test[strata[test] == code]])) <= 0.0:
                    raise DataError(
                        "cv_evaluation=True with baseline strata evaluates every stratum "
                        f"inside every validation fold, and validation fold {fold} holds no"
                        f"{positive} row of baseline stratum {data.stratum_label(code)}, so "
                        f"the stratum has no plug-in and no share in that fold. {remedy}"
                    )
    for draw in draws:
        for fold, (_, test) in enumerate(draw):
            for code in range(data.n_strata):
                rows = test[strata[test] == code]
                label = data.stratum_label(code)
                absent = [
                    data.arm_label(arm)
                    for arm in arms
                    if float(np.sum(weights[rows][treatment[rows] == float(arm)])) <= 0.0
                ]
                if absent:
                    raise DataError(
                        "cv_evaluation=True with baseline strata evaluates every stratum "
                        f"inside every validation fold, and validation fold {fold} holds no"
                        f"{positive} row of treatment arm(s) {absent} inside baseline stratum "
                        f"{label}, so the fold adds nothing to that arm's stratum score. "
                        f"{remedy}"
                    )


def fold_evaluated_stratum_weights(data: CausalData, validation: Sequence[IntArray]) -> FloatArray:
    r"""The fluctuation weights that solve every fold-evaluated stratum score exactly.

    The reported stratum estimate is :math:`V^{-1}\sum_v \psi_{s,v}`, with
    :math:`\psi_{s,v}` the weighted plug-in over the rows of fold :math:`v` in stratum
    :math:`s`. Its score is :math:`\sum_v \sum_{i \in v, S_i = s} w_i H_i (Y_i - \bar Q^*_i)
    / (V W_{v,s})`, with :math:`W_{v,s} = \sum_{j \in v, S_j = s} w_j`. Row :math:`i` of fold
    :math:`v` and stratum :math:`s` therefore takes

    .. math::

        \omega_i = w_i \, \frac{n \, P_n(S = s)}{V \, W_{v,s}} .

    The factor :math:`n P_n(S = s)` is constant inside the stratum, and each stratum block
    of the stratified submodel is zero off its stratum, so it moves no root. It keeps the
    weights summing to :math:`n`, as the fold weights of the unstratified fold-evaluated
    update do. The one-step walk, a step bound and the declared stopping thresholds read the
    scale, and this one matches the unstratified fold weights (it equals them at one
    stratum). The block's own :math:`1 / P_n(S = s)` cancels the share, so each block's
    equation is :math:`\sum_i w_i I(S_i = s) H_i (Y_i - \bar Q^*_i) \, n / (V W_{v,s}) = 0`.

    Parameters
    ----------
    data : CausalData
        The data of the fit, with baseline strata.
    validation : sequence of IntArray
        The validation rows of each fold. They partition the sample.

    Returns
    -------
    FloatArray
        ``(n,)`` fluctuation weights.

    Raises
    ------
    ValueError
        If a fold holds no weight mass of some stratum, which
        :func:`refuse_empty_stratum_folds` refuses before any learner.
    """
    assert data.strata is not None
    strata = np.asarray(data.strata)
    weights = np.array(data.weights, dtype=float, copy=True)
    probabilities = stratum_probabilities(data)
    n_folds = len(validation)
    for test in validation:
        for code, probability in enumerate(probabilities):
            rows = test[strata[test] == code]
            mass = float(np.sum(weights[rows]))
            if mass <= 0.0:
                raise ValueError(
                    "each validation fold must have positive observation-weight mass in "
                    "every baseline stratum"
                )
            weights[rows] *= data.n * probability / (n_folds * mass)
    return weights


def stratum_mixture(
    data: CausalData,
    marginal: str,
    parts: Sequence[ParameterEstimate],
    fold_values: Sequence[Sequence[float]],
    *,
    alpha: float,
) -> tuple[ParameterEstimate, tuple[float, ...]]:
    r"""The marginal estimate of a fold-evaluated stratified fit, and its fold values.

    The marginal estimate is :math:`\sum_s P_n(S = s) \hat\psi_s`, the weighted stratum
    shares times the fold-evaluated stratum estimates. Its curve is the delta-method curve of
    that sum,

    .. math::

        \phi_i = \sum_s P_n(S = s)\,\phi_{s,i}
            + \frac{w_i}{\bar w}\bigl(\hat\psi_{S_i} - \hat\psi\bigr),

    with :math:`\phi_s` the stored fold-evaluated stratum curve and the second term the
    curve of the shares. Each :math:`\phi_s` has mean zero because each stratum score is
    solved, so :math:`\phi` has mean zero too, and its variance is
    :math:`n^{-2}\sum_i \phi_i^2`. That is the cross-validated variance on each stratum part,
    since :math:`\phi_s` is stored as :math:`n / (V n_v)` times the fold curve. The marginal
    score itself is not solved: the update solves each stratum block under its fold-local
    weights, so the marginal plug-in average of the folds would carry a residual, and the
    report does not use it.

    Parameters
    ----------
    data : CausalData
        The data of the fit, with baseline strata.
    marginal : str
        The marginal parameter name.
    parts : sequence of ParameterEstimate
        The fold-evaluated estimate of each stratum, in stratum-code order.
    fold_values : sequence of sequence of float
        The fold plug-ins of each stratum, in stratum-code order.
    alpha : float
        The significance level of the estimate's interval.

    Returns
    -------
    tuple of ParameterEstimate and tuple of float
        The marginal estimate, and its value in each fold, :math:`\sum_s P_n(S = s)
        \psi_{s,v}`, whose ``1/V`` average is the estimate.

    Raises
    ------
    RuntimeError
        If a part is on the ratio scale, which the target list of
        :data:`FOLD_EVALUATED_STRATA_TARGETS` never produces.
    """
    assert data.strata is not None
    if any(part.scale == "ratio" for part in parts):
        raise RuntimeError("a fold-evaluated stratum mixture is linear and takes no ratio")
    probabilities = np.asarray(stratum_probabilities(data), dtype=float)
    stratum_psi = np.array([part.psi for part in parts], dtype=float)
    psi = float(probabilities @ stratum_psi)
    weights = np.asarray(data.weights, dtype=float)
    shares = weights / weights.mean() * (stratum_psi[np.asarray(data.strata)] - psi)
    curve = sum(
        (
            probability * part.influence_curve
            for probability, part in zip(probabilities, parts, strict=True)
        ),
        start=shares,
    )
    curve = np.asarray(curve, dtype=float)
    folds = tuple(
        float(probabilities @ np.asarray(values, dtype=float))
        for values in zip(*fold_values, strict=True)
    )
    estimate = ParameterEstimate(
        name=marginal,
        psi=psi,
        influence_curve=curve,
        variance=float(np.sum(np.square(curve)) / curve.size**2),
        n=data.n,
        n_clusters=data.n,
        scale=parts[0].scale,
        alpha=alpha,
        covariance_rule="centered",
    )
    return estimate, folds
