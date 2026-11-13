r"""Targeted estimation of the missingness-tilted parameters, with their own curves.

For arm :math:`a` with tilt :math:`\gamma_a`, on the outcome scaled to ``[0, 1]``, write
:math:`\pi_a(W) = P(\Delta = 1 \mid A = a, W)`,
:math:`\bar Q_a(W) = E[Y \mid \Delta = 1, A = a, W]` and
:math:`h_a = \operatorname{expit}(\operatorname{logit} \bar Q_a + \gamma_a)`.  The tilted
regression is :math:`m_a = \pi_a \bar Q_a + (1 - \pi_a) h_a`, and every tilted parameter is
a functional of the :math:`m_a`.  Differentiating :math:`m_a` in :math:`\bar Q_a` and in
:math:`\pi_a` gives the residual block of every tilted curve,

.. math::

    R_a = \frac{\Delta\,\kappa_a\,(Y - \bar Q_a)}{\pi_a} + (\bar Q_a - h_a)(\Delta - \pi_a),
    \qquad
    \kappa_a = \pi_a + (1 - \pi_a)\,\frac{h_a (1 - h_a)}{\bar Q_a (1 - \bar Q_a)}.

At :math:`\gamma_a = 0`, :math:`\kappa_a = 1` and :math:`h_a = \bar Q_a`, so the block is the
missing-at-random residual.

The estimator is a TMLE of the tilted parameter.  It starts from the targeted regression,
the full-covariate missingness mechanism and the bounded treatment mechanism of one
cross-fitting draw, and fluctuates :math:`\bar Q` and :math:`\pi` together, one coefficient
per column of the target's arm weights :math:`b_{j,a}(W)`:

.. math::

    \operatorname{logit} \bar Q_a(\varepsilon) = \operatorname{logit} \bar Q_a
      + \sum_j \varepsilon_j\, b_{j,a} \kappa_a / \pi_a,
    \qquad
    \operatorname{logit} \pi_a(\varepsilon) = \operatorname{logit} \pi_a
      + \sum_j \varepsilon_j\, b_{j,a} (\bar Q_a - h_a).

The two submodels share one coefficient per column, so the score of a coefficient is the
sum of the two halves of the residual block, and solving it solves the tilted estimating
equation.  :math:`\kappa_a` and :math:`h_a` read :math:`\bar Q_a`, so the covariates are
recomputed after every Newton step until each column's relative score meets the fit's
tolerance.  At :math:`\gamma = 0` the mechanism covariate is zero and the outcome covariate is
the missing-at-random one, which the fit's targeted regression already solves, so nothing
moves and the estimate is the reported one.  The tilted parameter is not doubly robust in
:math:`\bar Q`, because :math:`h_a` is nonlinear in it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from .._warnings import warn as _warn
from ..data.causal_data import arm_share
from ..estimators._nuisance import tilt_missingness
from ..exceptions import DataWarning
from ..inference.delta import log_odds_ratio_influence, log_ratio_influence
from ..inference.influence import ParameterEstimate, Scale, make_estimate, unscale
from ..utils.bounds import expit, g_bounds_for, logit

#: Predictions are kept this far inside ``(0, 1)`` while the fluctuation moves them, the
#: guard the package's logistic submodel applies.
_EDGE = 1e-9

#: The most Newton steps one solve may take.
_MAX_ITER = 200


@dataclass(frozen=True)
class TiltSolve:
    """One solved tilt fluctuation.

    Parameters
    ----------
    outcome : ndarray
        ``(n, K)`` the fluctuated outcome regression at every arm.
    mechanism : ndarray
        ``(n, K)`` the fluctuated missingness mechanism at every arm.
    converged : bool
        Whether both relative scores reached the tolerance.
    n_iter : int
        Newton steps taken.
    score : float
        The largest relative score at the end.
    """

    outcome: FloatArray
    mechanism: FloatArray
    converged: bool
    n_iter: int
    score: float


def tilted_regression(outcome: FloatArray, mechanism: FloatArray, gamma: FloatArray) -> FloatArray:
    """``m_a = pi_a Q_a + (1 - pi_a) h_a`` for every arm column."""
    h = expit(logit(outcome) + gamma)
    return np.asarray(mechanism * outcome + (1.0 - mechanism) * h, dtype=float)


def residual_block(
    scaled: FloatArray,
    observed: BoolArray,
    outcome: FloatArray,
    mechanism: FloatArray,
    gamma: FloatArray,
) -> FloatArray:
    r"""``R_a`` for every row and arm column, ``(n, K)``.

    The row's own arm is the only column a curve reads; the others are returned so the
    caller can index by the received arm without a loop.
    """
    h = expit(logit(outcome) + gamma)
    kappa = mechanism + (1.0 - mechanism) * h * (1.0 - h) / (outcome * (1.0 - outcome))
    delta = np.asarray(observed, dtype=float)[:, None]
    y = np.where(np.asarray(observed, dtype=bool), scaled, 0.0)[:, None]
    return np.asarray(
        delta * kappa * (y - outcome) / mechanism + (outcome - h) * (delta - mechanism),
        dtype=float,
    )


def solve_tilt(
    scaled: FloatArray,
    observed: BoolArray,
    received: IntArray,
    weights: FloatArray,
    outcome: FloatArray,
    mechanism: FloatArray,
    gamma: FloatArray,
    arm_weights: FloatArray,
    *,
    tol: float = 1e-10,
) -> TiltSolve:
    r"""Fluctuate ``Qbar`` and ``pi`` together until the tilted score is solved.

    Parameters
    ----------
    scaled : ndarray
        ``(n,)`` the outcome on the ``[0, 1]`` scale; read only where ``observed``.
    observed : ndarray of bool
        ``(n,)`` the outcome indicator ``Delta``.
    received : ndarray of int
        ``(n,)`` each row's arm, as a column index.
    weights : ndarray
        ``(n,)`` observation weights.
    outcome : ndarray
        ``(n, K)`` the starting ``Qbar_a(W)``.
    mechanism : ndarray
        ``(n, K)`` the starting ``pi_a(W)``.
    gamma : ndarray
        ``(K,)`` the tilt of each arm.
    arm_weights : ndarray
        ``(n, K, J)`` the target's arm weight :math:`b_{j,a}(W)` of each column ``j``, which
        already divides by the treatment mechanism where the target's curve does.
    tol : float
        The relative score each column must reach, the fit's ``targeting_spec.tol``.

    Returns
    -------
    TiltSolve
        The fluctuated regressions and how the solve ended.
    """
    q = np.clip(np.asarray(outcome, dtype=float), _EDGE, 1.0 - _EDGE)
    p = np.clip(np.asarray(mechanism, dtype=float), _EDGE, 1.0 - _EDGE)
    b = np.asarray(arm_weights, dtype=float)
    g = np.asarray(gamma, dtype=float)[None, :]
    a = np.asarray(received, dtype=int)
    w = np.asarray(weights, dtype=float)
    d = np.asarray(observed, dtype=bool)
    y = np.where(d, np.asarray(scaled, dtype=float), 0.0)
    m = q.shape[0]
    at = np.arange(m)
    delta = d.astype(float)

    def covariates(q: FloatArray, p: FloatArray) -> tuple[FloatArray, FloatArray]:
        h = expit(logit(q) + g)
        kappa = p + (1.0 - p) * h * (1.0 - h) / (q * (1.0 - q))
        # (m, K, J): the outcome and the mechanism clever covariates of every column.
        return b * (kappa / p)[:, :, None], b * (q - h)[:, :, None]

    def score(q: FloatArray, p: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        # The total score of each column, the outcome half plus the mechanism half, and its
        # largest value relative to the most it could be.
        hq, hp = covariates(q, p)
        own_q, own_p = hq[at, a], hp[at, a]
        total = (w * delta * (y - q[at, a])) @ own_q / m + (w * (delta - p[at, a])) @ own_p / m
        scale = np.maximum(
            (np.abs(w * delta) @ np.abs(own_q) + np.abs(w) @ np.abs(own_p)) / m, 1e-300
        )
        return total, scale, float(np.max(np.abs(total) / scale, initial=0.0))

    converged = False
    iteration = 0
    _, _, relative = score(q, p)
    while iteration < _MAX_ITER:
        if relative <= tol:
            converged = True
            break
        iteration += 1
        # One Newton step of the summed weighted log likelihood of the two submodels, with
        # one coefficient per column shared by both.  Sharing it is what keeps the mechanism
        # still as gamma goes to zero, where its covariate vanishes: a coefficient of its
        # own would solve the mechanism score along that vanishing direction at any gamma,
        # and the estimate would jump at gamma = 0.
        hq, hp = covariates(q, p)
        own_q, own_p = hq[at, a], hp[at, a]
        vq = w * delta * q[at, a] * (1.0 - q[at, a])
        vp = w * p[at, a] * (1.0 - p[at, a])
        gram = ((own_q * vq[:, None]).T @ own_q + (own_p * vp[:, None]).T @ own_p) / m
        total, _, _ = score(q, p)
        step = np.asarray(np.linalg.lstsq(gram, total, rcond=None)[0], dtype=float)
        # The updated mechanism is not clipped back to ``missingness_bound``.  That floor
        # guards the initial fit; the update is a score-solving submodel step, and clipping
        # its result would break the estimating equation the step solves.  The departure from
        # the bounded start is of the order of the coefficient, which shrinks with n.
        q = np.clip(expit(logit(q) + hq @ step), _EDGE, 1.0 - _EDGE)
        p = np.clip(expit(logit(p) + hp @ step), _EDGE, 1.0 - _EDGE)
        _, _, relative = score(q, p)
    else:
        converged = relative <= tol

    return TiltSolve(q, p, converged, iteration, relative)


def arm_columns(received: IntArray, propensity: FloatArray) -> FloatArray:
    """The arm weights of the arm means: column ``j`` is ``1 / g_j`` on arm ``j``.

    Parameters
    ----------
    received : ndarray of int
        ``(n,)`` each row's arm column.
    propensity : ndarray
        ``(n, K)`` the bounded treatment mechanism.

    Returns
    -------
    ndarray
        ``(n, K, K)``.
    """
    n, k = propensity.shape
    del received
    out = np.zeros((n, k, k))
    for j in range(k):
        out[:, j, j] = 1.0 / propensity[:, j]
    return out


def contrast_columns(propensity: FloatArray, pairs: Sequence[tuple[int, int, int]]) -> FloatArray:
    """The arm weights of conditional contrasts, one column per ``(arm, reference, on)``.

    Column ``j`` for the contrast of arm ``k`` against ``r`` among arm ``c`` is
    ``g_c / g_k`` on arm ``k`` and ``-g_c / g_r`` on arm ``r``.

    Parameters
    ----------
    propensity : ndarray
        ``(n, K)`` the bounded treatment mechanism.
    pairs : sequence of tuple of int
        ``(arm, reference, conditioning)`` column indices per contrast.

    Returns
    -------
    ndarray
        ``(n, K, J)``.
    """
    n, k = propensity.shape
    out = np.zeros((n, k, len(pairs)))
    for j, (arm, reference, on) in enumerate(pairs):
        out[:, arm, j] = propensity[:, on] / propensity[:, arm]
        out[:, reference, j] = -propensity[:, on] / propensity[:, reference]
    return out


def natural_course_columns(n: int, k: int) -> FloatArray:
    """The arm weights of the natural-course mean: one column, ``1`` at every arm."""
    return np.ones((n, k, 1))


def arm_mean_curves(
    solve: TiltSolve,
    scaled: FloatArray,
    observed: BoolArray,
    received: IntArray,
    weights: FloatArray,
    propensity: FloatArray,
    gamma: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Every tilted arm mean and its weighted curve on the scaled outcome.

    Returns
    -------
    tuple of ndarray
        ``(K,)`` the means and ``(n, K)`` the curves, ``w (m_a - psi_a + 1{A = a} R_a / g_a)``.
    """
    w = np.asarray(weights, dtype=float)
    regression = tilted_regression(solve.outcome, solve.mechanism, gamma[None, :])
    psi = np.average(regression, axis=0, weights=w)
    block = residual_block(scaled, observed, solve.outcome, solve.mechanism, gamma[None, :])
    n, k = regression.shape
    own = np.zeros((n, k))
    own[np.arange(n), received] = 1.0
    curves = w[:, None] * (regression - psi + own * block / propensity)
    return np.asarray(psi, dtype=float), np.asarray(curves, dtype=float)


def contrast_curves(
    solve: TiltSolve,
    scaled: FloatArray,
    observed: BoolArray,
    received: IntArray,
    weights: FloatArray,
    propensity: FloatArray,
    gamma: FloatArray,
    pairs: Sequence[tuple[int, int, int]],
) -> tuple[FloatArray, FloatArray]:
    r"""Each tilted conditional contrast and its weighted curve on the scaled outcome.

    For arm ``k`` against ``r`` among arm ``c`` with share ``p``,

    .. math::

        D = \frac{w}{p}\Bigl[1\{A = c\}(m_k - m_r - \psi)
          + g_c\Bigl(\frac{1\{A = k\} R_k}{g_k} - \frac{1\{A = r\} R_r}{g_r}\Bigr)\Bigr].

    Returns
    -------
    tuple of ndarray
        ``(J,)`` the contrasts and ``(n, J)`` the curves.
    """
    w = np.asarray(weights, dtype=float)
    regression = tilted_regression(solve.outcome, solve.mechanism, gamma[None, :])
    block = residual_block(scaled, observed, solve.outcome, solve.mechanism, gamma[None, :])
    n = regression.shape[0]
    rows = np.arange(n)
    own_block = block[rows, received]
    psi = np.zeros(len(pairs))
    curves = np.zeros((n, len(pairs)))
    for j, (arm, reference, on) in enumerate(pairs):
        indicator = (received == on).astype(float)
        share = float(np.average(indicator, weights=w))
        contrast = regression[:, arm] - regression[:, reference]
        psi[j] = float(np.average(contrast, weights=w * indicator))
        residual = np.where(
            received == arm,
            own_block / propensity[:, arm],
            np.where(received == reference, -own_block / propensity[:, reference], 0.0),
        )
        curves[:, j] = w * (indicator * (contrast - psi[j]) + propensity[:, on] * residual) / share
    return psi, curves


def natural_course_curve(
    solve: TiltSolve,
    scaled: FloatArray,
    observed: BoolArray,
    received: IntArray,
    weights: FloatArray,
    gamma: FloatArray,
) -> tuple[float, FloatArray]:
    """The tilted natural-course mean and its weighted curve, ``w (m_A - psi + R_A)``."""
    w = np.asarray(weights, dtype=float)
    rows = np.arange(received.shape[0])
    regression = tilted_regression(solve.outcome, solve.mechanism, gamma[None, :])[rows, received]
    block = residual_block(scaled, observed, solve.outcome, solve.mechanism, gamma[None, :])
    psi = float(np.average(regression, weights=w))
    return psi, np.asarray(w * (regression - psi + block[rows, received]), dtype=float)


# ----------------------------------------------------------------- reading a fitted result


#: The target kinds whose tilted curve is derived here.  A regime mean and an MSM
#: coefficient keep the diagnostic spread: their tilted curves need a derivation of their
#: own (the regime density and the working-model projection enter the curve).
INFERENTIAL_KINDS = frozenset({"arm", "ratio", "natural_course", "par", "paf"})


def tilt_interval_exclusion(result: Any) -> str | None:
    """Why no tilted parameter of this fit gets an interval, or ``None`` when they can.

    Read after the fit-wide tilt refusals, so the result is a point-treatment fit with arms
    and missing outcomes.  Each sentence names what the tilted interval would need.

    Parameters
    ----------
    result : Any
        A fitted result that the fit-wide tilt rules admit.

    Returns
    -------
    str or None
        The reason, or ``None``.
    """
    from ..estimators.tmle import TMLE

    estimator = getattr(result, "estimator", None)
    name = type(estimator).__name__
    if name == "DRTMLE" or getattr(result, "solved_corrections", False):
        return (
            "the tilted parameter is not doubly robust in the outcome regression, so no "
            "tilted interval can inherit this fit's one-wrong-nuisance guarantee, and a "
            "plain tilted TMLE from its nuisances would claim less than the fit's label implies"
        )
    if name == "CTMLE":
        return (
            "a collaborative fit estimates its treatment mechanism only as far as the "
            "missing-at-random parameter needs, and that this suffices for the tilted "
            "parameter, whose clever covariates read the tilt, is not shown"
        )
    if type(estimator) is not TMLE:
        return f"no targeted tilt is derived from the nuisances of a {name} fit"
    if result.extra.get("missing_data", "missing_outcome") != "missing_outcome":
        return (
            "the fit reads a composite observation mechanism, and the tilt is defined on the "
            "outcome-missingness mechanism alone"
        )
    if result.data.has_intermediate:
        return "no targeted tilt is derived for a controlled direct effect"
    return None


def targeted_tilt_estimates(
    result: Any,
    repeat: Any,
    targets: Sequence[Any],
    gamma: float,
    direction: Mapping[float, float],
) -> tuple[dict[str, ParameterEstimate], bool]:
    """The targeted tilted estimate of every target, from one cross-fitting draw.

    Parameters
    ----------
    result : Any
        The fitted result.
    repeat : RepeatFit
        The draw whose targeted regressions and mechanisms start the fluctuation.
    targets : sequence of _TiltTarget
        The targets to estimate, each of an :data:`INFERENTIAL_KINDS` kind.
    gamma : float
        The tilt magnitude.
    direction : mapping of float to float
        The per-arm multiplier, by arm code.

    Returns
    -------
    tuple of dict and bool
        The estimates by name, on the reported scale, and whether every solve converged.
    """
    data = result.data
    config = result.config
    nuisance = repeat.nuisance
    scaler = nuisance.scaler
    arms = tuple(nuisance.arms)
    column = {arm: arms.index(arm) for arm in arms}
    treatment = np.asarray(data.treatment, dtype=float)
    received = np.array([column[float(code)] for code in treatment], dtype=int)
    observed = np.asarray(data.observed, dtype=bool)
    raw = np.asarray(data.outcome, dtype=float)
    scaled = np.where(observed, scaler.scale(np.where(observed, raw, 0.0)), 0.0)
    weights = np.asarray(data.weights, dtype=float)
    mechanism = tilt_missingness(nuisance, config.missingness_bound)
    assert mechanism is not None
    tilt = np.array([gamma * direction[arm] for arm in arms], dtype=float)
    tol = float(config.targeting_spec.tol)
    n = data.n
    alpha = float(config.alpha_sig)
    folds = (
        [np.asarray(test, dtype=int) for _, test in nuisance.folds]
        if config.cv_evaluation and not nuisance.folds.is_single
        else None
    )
    # The fit's own update weights: under the fold-evaluated construction they normalise
    # the weight mass inside each validation fold, and solving with any other weights would
    # move the fluctuation at gamma = 0 away from the fit's.
    solve_weights = np.asarray(result.estimator._validation_weights(data, nuisance), dtype=float)
    converged = True
    solved: dict[Any, TiltSolve] = {}

    def solve(key: Any, columns: FloatArray, start: FloatArray) -> TiltSolve:
        # Fold-specific targeting with missing outcomes is refused by the fit, so every fit
        # reaching here fluctuates once on the pooled sample.
        nonlocal converged
        if key not in solved:
            solved[key] = solve_tilt(
                scaled, observed, received, solve_weights, start, mechanism, tilt, columns, tol=tol
            )
            converged = converged and solved[key].converged
            lowest = float(np.min(solved[key].mechanism[np.arange(n), received]))
            if lowest < config.missingness_bound:
                # Re-clipping would break the solved equation, so the update keeps its value
                # and says that 1 / pi now exceeds the fit's bound on it.
                _warn(
                    f"the tilt update at gamma = {gamma:g} moved the response probability "
                    f"below the fit's missingness_bound {config.missingness_bound:g}: its "
                    f"smallest value is {lowest:.3g}, so 1 / pi in the correction and the "
                    "influence curve exceeds the bound the fit declared.",
                    DataWarning,
                    stacklevel=2,
                )
        return solved[key]

    def regression(group: str) -> FloatArray:
        targeted = repeat.fluctuations[group].targeted
        if group == "natural_course":
            # Only the received arm is read, by the curve and by the plug-in.
            out = np.full((n, len(arms)), 0.5)
            out[np.arange(n), received] = np.asarray(targeted.observed, dtype=float)
            return out
        return np.column_stack([np.asarray(targeted.arms[arm], dtype=float) for arm in arms])

    def propensity(group: str) -> FloatArray:
        bounds = g_bounds_for(group, config.g_bounds, config.g_bounds_conditional)
        return np.asarray(nuisance.bounded_propensity(bounds), dtype=float)

    def report(rows: IntArray | None) -> dict[str, ParameterEstimate]:
        # Every target on the rows ``rows`` selects, one validation fold or (``None``) the
        # whole sample, from the pooled fluctuations.  A fold's weights are renormalised to
        # mean one, so its estimate and curve are those of a standalone fit on its rows.
        at = slice(None) if rows is None else rows
        size = n if rows is None else rows.size
        y, d, a = scaled[at], observed[at], received[at]
        w = weights[at] / (1.0 if rows is None else float(np.mean(weights[at])))
        cluster = None if data.cluster is None else np.asarray(data.cluster)[at]

        def cut(out: TiltSolve) -> TiltSolve:
            if rows is None:
                return out
            return TiltSolve(
                out.outcome[rows], out.mechanism[rows], out.converged, out.n_iter, out.score
            )

        means: dict[str, tuple[FloatArray, FloatArray]] = {}

        def arm_means(group: str) -> tuple[FloatArray, FloatArray]:
            if group not in means:
                g = propensity(group)
                out = solve(("arm", group), arm_columns(received, g), regression(group))
                means[group] = arm_mean_curves(cut(out), y, d, a, w, g[at], tilt)
            return means[group]

        observed_mean: list[tuple[float, FloatArray]] = []

        def natural_course() -> tuple[float, FloatArray]:
            if not observed_mean:
                columns = natural_course_columns(n, len(arms))
                out = solve("natural_course", columns, regression("natural_course"))
                observed_mean.append(natural_course_curve(cut(out), y, d, a, w, tilt))
            return observed_mean[0]

        def finish(target: Any, psi: float, curve: FloatArray, scale: Scale) -> ParameterEstimate:
            reported = result[target.name]
            log_psi = None
            if scale == "fraction":
                value, ic = psi, curve
            else:
                value, ic = unscale(psi, curve, scaler, scale)
            if scale == "ratio":
                log_psi, value = psi, float(np.exp(psi))
            return make_estimate(
                target.name,
                value,
                ic,
                n=size,
                cluster=cluster,
                scale=scale,
                alpha=alpha,
                log_psi=log_psi,
                covariance_rule=reported.covariance_rule,
                inference=reported.inference,
                reference_df=reported.reference_df,
                # The tilted refit solves its own score to tolerance (``solve_tilt``).
                rule_widened=False,
            )

        out: dict[str, ParameterEstimate] = {}
        for target in targets:
            kind = target.kind
            if kind == "arm":
                parameter = target.parameter
                conditioning = parameter.conditions_on
                if conditioning is None:
                    psi_arms, curves = arm_means(parameter.group)
                    i = column[parameter.arm]
                    if parameter.versus is None:
                        out[target.name] = finish(target, psi_arms[i], curves[:, i], "level")
                    else:
                        r = column[parameter.versus]
                        out[target.name] = finish(
                            target,
                            psi_arms[i] - psi_arms[r],
                            curves[:, i] - curves[:, r],
                            "difference",
                        )
                    continue
                g = propensity(parameter.group)
                pair = [(column[parameter.arm], column[parameter.versus], column[conditioning])]
                key = ("contrast", parameter.group, pair[0])
                columns = contrast_columns(g, pair)
                if folds is not None:
                    # The fit rebuilds this covariate with each validation fold's share of
                    # the conditioning arm, so the common update solves the equal average of
                    # the fold scores; a common share would target a different equation.
                    for fold in folds:
                        columns[fold] /= arm_share(
                            data.treatment, data.weights, conditioning, mask=fold
                        )
                fitted = solve(key, columns, regression(parameter.group))
                psi_c, curve_c = contrast_curves(cut(fitted), y, d, a, w, g[at], tilt, pair)
                out[target.name] = finish(target, float(psi_c[0]), curve_c[:, 0], "difference")
            elif kind == "natural_course":
                psi_o, curve_o = natural_course()
                out[target.name] = finish(target, psi_o, curve_o, "level")
            elif kind in {"par", "paf"}:
                psi_o, curve_o = natural_course()
                psi_arms, curves = arm_means(target.group)
                r = column[target.code]
                if kind == "par":
                    out[target.name] = finish(
                        target, psi_o - psi_arms[r], curve_o - curves[:, r], "difference"
                    )
                else:
                    # The fraction reads the two means on the outcome's own scale.  The fit
                    # admits paf only on a binary outcome, whose scaler is the identity, so
                    # this is the scale the fit's own paf target reads.
                    whole, whole_curve = unscale(psi_o, curve_o, scaler, "level")
                    base, base_curve = unscale(psi_arms[r], curves[:, r], scaler, "level")
                    psi = 1.0 - base / whole
                    curve = -base_curve / whole + base * whole_curve / whole**2
                    out[target.name] = finish(target, psi, curve, "fraction")
            elif kind == "ratio":
                psi_arms, curves = arm_means(target.group)
                i, r = column[target.code], column[target.versus]
                ratio = log_ratio_influence if target.stem == "rr" else log_odds_ratio_influence
                log_psi, curve = ratio(psi_arms[i], curves[:, i], psi_arms[r], curves[:, r])
                out[target.name] = finish(target, log_psi, curve, "ratio")
            else:  # pragma: no cover - the caller passes only the inferential kinds
                raise ValueError(f"no targeted tilt for the {kind!r} kind")
        return out

    if folds is None:
        return report(None), converged
    from ..estimators.tmle import _average_over_folds

    # The fold-evaluated construction: the equal 1/V average of the fold plug-ins, with the
    # cross-validated variance, assembled exactly as the fit assembles its own report.
    per_fold = [report(rows) for rows in folds]
    names = [target.name for target in targets]
    averaged = _average_over_folds(per_fold, names, folds, n=n, cluster=data.cluster, alpha=alpha)
    return averaged, converged
