"""Shared pieces of the baseline-strata targeting.

A stratum parameter is the marginal parameter of the law of ``O`` given ``S = s``, for a fixed
finite partition ``S`` of the adjustment columns with ``P(S = s) > 0``.  Every target group
solves one score block ``I(S = s) H_s / P_n(S = s)`` per stratum.  The helpers here read the
stratum masses and refuse, before any learner, a stratum whose score equations have no finite
root.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..data.causal_data import CausalData, arm_share
from ..exceptions import DataError

__all__ = [
    "absent_arm_error",
    "check_stratified_msm_design",
    "check_stratified_targets",
    "companion_stratum_codes",
    "embed_stratum_curve",
    "refuse_absent_tilt_arms",
    "refuse_untrainable_stratum_folds",
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
            "stratum's treatment-mechanism score equation has no finite root"
        )
    return DataError(
        f"baseline stratum {label} contains no positive-weight observations from treatment "
        f"arm(s) {absent}; its empirical targeting score is unidentified"
    )


def refuse_absent_tilt_arms(data: CausalData) -> None:
    """Refuse an incremental fit with a stratum that lacks a treatment arm.

    In a stratum where every row has ``A = 0``, the stratum's mechanism equation
    ``P_n[I(S = s) H_g (0 - g*)] = 0`` has no finite root wherever the blip keeps one sign:
    the logistic tilt separates.  The check reads the data only, so it runs before any
    learner.

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
            raise DataError(
                f"inside baseline stratum {data.stratum_label(code)}: {error}"
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
