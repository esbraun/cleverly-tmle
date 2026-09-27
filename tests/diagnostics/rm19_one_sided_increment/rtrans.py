"""A transcription of R ``drtmle`` 1.1.2's binary cross-validated loop, with four switches.

RM19 of ``docs/roadmap.md`` declares the design that reads it.  The transcription follows the
pinned source at commit ``538a3a26`` (``R/drtmle.R``, ``R/fluctuate.R``, ``R/estimate.R`` and
``R/inf_functions.R``) at the settings of ``tests/canonical/drtmle/run_drtmle.R``: univariate
reduction, ``guard = c("Q", "g")``, ``Qsteps = 2``, ``glm_Qr = "gn"``, ``glm_gr = "Qn"``,
``tolg = 0.01``, ``tolIC = 1e-8``, ``maxIter = 100``, supplied ``Qn`` and ``gn``, and the study's
fold vector.  With every switch off it is R.  Each switch moves one named step to the choice that
``cleverly``'s registered fit makes:

========== ============================================ ==========================================
switch     R ``drtmle``                                 ``cleverly``
========== ============================================ ==========================================
``J``      two one-column tilts of ``g_0`` and ``g_1``  one tilt of ``g_1`` along the two columns
           (``fluctuate.R:93-126``)                     ``(-Qr_0/g_0, Qr_1/g_1)``, ``g_0 = 1 - g_1``
``P``      the loop starts with the ``g`` step          one equation-(8) step of ``Qbar^0`` first
``S``      stop when every mean is at most ``tolIC``    the per-equation exit, the stall rule and
           or at ``maxIter`` (``drtmle.R:607``)         the closing pass of ``solve_with_reduction``
``G``      skip a step on ``all(H < 1e-7)``, and keep   no skip and no fallback
           the current fit when ``glm`` fails or
           ``abs(coef) > 1e3``
========== ============================================ ==========================================

The README of this directory gives each resolution the declaration leaves open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The runner's settings (``run_drtmle.R:33-54``) and R's defaults that it leaves alone.
TOLG = 0.01
TOL_IC = 1e-8
MAX_ITER = 100
COEF_TOL = 1e3
#: ``SuperLearner::trimLogit``'s default trim, which ``fluctuateQ1`` and ``fluctuateQ2`` use.
TRIM = 1e-5
#: ``stats::glm.control()``: ``epsilon = 1e-8``, ``maxit = 25``.
GLM_EPSILON = 1e-8
GLM_MAXIT = 25
#: A skip in ``fluctuateQ1`` and ``fluctuateQ2`` fires when every covariate value is below this.
SKIP = 1e-7
#: R's logit link clamps its linear predictor at these (``THRESH``, ``DOUBLE_EPS``).
THRESH = 30.0
EPS = float(np.finfo(float).eps)

#: ``cleverly``'s side of switch S: ``fit_cleverly``'s ``tol``, ``solve_with_reduction``'s
#: absolute bar ``1e-3 / n`` (``targeting.py``, ``_negligible_bar``), its stall factor, and the
#: closing pass's step cap (``_close_at_frozen_reductions``).
CLEVERLY_TOL = 1e-10
NEGLIGIBLE = 1e-3
STALL_FACTOR = 0.95
CLOSING_STEPS = 20
#: The precision of the Newton solves that switch J and the closing pass take in place of
#: ``cleverly``'s own solvers.
FINE_EPSILON = 1e-14
FINE_MAXIT = 100
#: ``cleverly``'s logit guard on the mechanism offset (``fluctuation/mechanism.py``).
LOGIT_GUARD = 1e-12

ARMS = (0.0, 1.0)


@dataclass(frozen=True)
class Switches:
    """Which steps take ``cleverly``'s choice.  All ``False`` is R ``drtmle``."""

    joint_tilt: bool = False
    prime: bool = False
    cleverly_exit: bool = False
    no_guard: bool = False

    @property
    def label(self) -> str:
        """``T`` and the three factor levels, with ``+G`` when the guards are removed."""
        levels = f"{int(self.joint_tilt)}{int(self.prime)}{int(self.cleverly_exit)}"
        return f"T{levels}" + ("+G" if self.no_guard else "")


@dataclass
class _Log:
    """What one fit records beside its estimates."""

    guard_events: int = 0


@dataclass(frozen=True)
class Transcribed:
    """One transcribed fit: the three estimates, their standard errors and the instruments."""

    estimates: dict[str, float]
    std_errors: dict[str, float]
    iterations: int
    exit: str
    closing: int
    guard_events: int
    at_bound: int
    score_max: float


class TranscriptionFailure(RuntimeError):
    """A step that R would not survive either: a non-finite coefficient with the guards off."""


# ----------------------------------------------------------------------------- R numerics


def expit(x: np.ndarray) -> np.ndarray:
    """The logistic function of ``cleverly``'s side: switch J and the closing pass."""
    with np.errstate(over="ignore"):
        return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=float)


def linkinv(eta: np.ndarray) -> np.ndarray:
    """``binomial()$linkinv`` (R's ``logit_linkinv`` in ``src/library/stats/src/family.c``).

    ``exp(eta) / (1 + exp(eta))``, with ``exp(eta)`` replaced by ``DBL_EPSILON`` below
    ``eta = -30`` and by ``1 / DBL_EPSILON`` above ``30``.  ``glm.fit`` and ``predict.glm``
    both read it, so a fit whose linear predictor leaves ``[-30, 30]`` needs the clamp.
    """
    eta = np.asarray(eta, dtype=float)
    with np.errstate(over="ignore"):
        tmp = np.where(eta < -THRESH, EPS, np.where(eta > THRESH, 1.0 / EPS, np.exp(eta)))
    return np.asarray(tmp / (1.0 + tmp), dtype=float)


def mu_eta(eta: np.ndarray) -> np.ndarray:
    """``binomial()$mu.eta`` (R's ``logit_mu_eta``): ``DBL_EPSILON`` outside ``[-30, 30]``."""
    eta = np.asarray(eta, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        opexp = 1.0 + np.exp(eta)
        inside = np.exp(eta) / (opexp * opexp)
    return np.asarray(np.where(np.abs(eta) > THRESH, EPS, inside), dtype=float)


def trim_logit(x: np.ndarray, trim: float = TRIM) -> np.ndarray:
    """``SuperLearner::trimLogit``: clip to ``[trim, 1 - trim]``, then the logit."""
    clipped = np.clip(x, trim, 1.0 - trim)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(y > 0, y * np.log(y / mu), 0.0)
        second = np.where(y < 1, (1.0 - y) * np.log((1.0 - y) / (1.0 - mu)), 0.0)
    return float(np.sum(2.0 * (first + second)))


def glm_binomial(
    y: np.ndarray,
    design: np.ndarray,
    offset: np.ndarray,
    start: np.ndarray | None = None,
    *,
    maxit: int = GLM_MAXIT,
    epsilon: float = GLM_EPSILON,
) -> tuple[np.ndarray, bool]:
    """``stats::glm.fit`` for ``binomial(logit)`` with unit prior weights.

    The IRLS iteration of R 4.5's ``glm.fit``: the ``binomial`` initialiser
    ``mustart = (y + 0.5) / 2`` without ``start``, the working response
    ``(eta - offset) + (y - mu) / mu.eta`` with weights ``sqrt(mu.eta^2 / (mu (1 - mu)))``, the
    clamped link of :func:`linkinv` and :func:`mu_eta`, the convergence test
    ``abs(dev - devold) / (abs(dev) + 0.1) < epsilon``, and step halving towards the previous
    coefficients on a non-finite deviance or an invalid mean.  Returns the coefficients and
    ``converged``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(design, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if start is None:
        mustart = (y + 0.5) / 2.0
        eta = np.log(mustart / (1.0 - mustart))
        coef = np.zeros(x.shape[1])
    else:
        coef = np.asarray(start, dtype=float)
        eta = offset + x @ coef
    mu = linkinv(eta)
    devold = _deviance(y, mu)
    converged = False
    for _ in range(maxit):
        derivative = mu_eta(eta)
        good = derivative != 0
        working = (eta - offset)[good] + (y - mu)[good] / derivative[good]
        root = np.sqrt(derivative[good] ** 2 / (mu * (1.0 - mu))[good])
        new, *_ = np.linalg.lstsq(x[good] * root[:, None], working * root, rcond=None)
        if not np.all(np.isfinite(new)):
            break
        eta_new = offset + x @ new
        mu_new = linkinv(eta_new)
        dev = _deviance(y, mu_new)
        halvings = 0
        while (not np.isfinite(dev) or not np.all((mu_new > 0) & (mu_new < 1))) and halvings < 25:
            new = (new + coef) / 2.0
            eta_new = offset + x @ new
            mu_new = linkinv(eta_new)
            dev = _deviance(y, mu_new)
            halvings += 1
        coef, eta, mu = new, eta_new, mu_new
        if abs(dev - devold) / (abs(dev) + 0.1) < epsilon:
            converged = True
            break
        devold = dev
    return coef, converged


def ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``glm(y ~ x, gaussian)``: intercept and slope, or the mean alone on a constant ``x``."""
    if np.unique(x).size == 1:
        return np.array([float(np.mean(y)), 0.0])
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(coef, dtype=float)


# --------------------------------------------------------------- reduced regressions (R)


def fold_rows(folds: np.ndarray) -> list[np.ndarray]:
    """``make_validRows`` on the supplied fold vector: the rows of each fold, in fold order."""
    return [np.flatnonzero(folds == fold) for fold in np.unique(folds)]


def estimate_qrn(
    y: np.ndarray, a: np.ndarray, qn: list[np.ndarray], gn: list[np.ndarray], valid: list
) -> list[np.ndarray]:
    """``estimateQrn`` (``estimate.R:1053-1190``): ``Y - Qn_a`` on ``gn_a`` over ``A = a``."""
    n = y.size
    out = []
    for arm, q, g in zip(ARMS, qn, gn, strict=True):
        estimate = np.empty(n)
        for rows in valid:
            train = np.ones(n, dtype=bool)
            train[rows] = False
            selected = train & (a == arm)
            intercept, slope = ols((y - q)[selected], g[selected])
            estimate[rows] = intercept + slope * g[rows]
        out.append(estimate)
    return out


def estimate_grn(
    a: np.ndarray, qn: list[np.ndarray], gn: list[np.ndarray], valid: list
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``estimategrn`` (``estimate.R:1470-1520``), univariate: R's ``(grn1, grn2)`` per arm.

    ``grn1`` is the Gaussian fit of ``(1_a - g)/g`` on ``Qn_a`` and ``grn2`` the logistic fit of
    ``1_a`` on ``Qn_a``, bounded below at ``tolg``.  ``cleverly`` names them ``gr2`` and ``gr1``.
    """
    n = a.size
    out = []
    for arm, q, g in zip(ARMS, qn, gn, strict=True):
        signed, probability = np.empty(n), np.empty(n)
        indicator = (a == arm).astype(float)
        for rows in valid:
            train = np.ones(n, dtype=bool)
            train[rows] = False
            intercept, slope = ols(((indicator - g) / g)[train], q[train])
            signed[rows] = intercept + slope * q[rows]
            if np.unique(q[train]).size == 1:
                probability[rows] = float(np.mean(indicator[train]))
                continue
            design = np.column_stack([np.ones(int(train.sum())), q[train]])
            coef, _ = glm_binomial(indicator[train], design, np.zeros(int(train.sum())))
            probability[rows] = linkinv(coef[0] + coef[1] * q[rows])
        probability[probability < TOLG] = TOLG
        out.append((signed, probability))
    return out


# --------------------------------------------------------------------------- the steps


def fluctuate_q(
    y: np.ndarray,
    covariate: np.ndarray,
    counterfactual: np.ndarray,
    q: np.ndarray,
    switches: Switches,
    log: _Log,
) -> np.ndarray:
    """``fluctuateQ1`` or ``fluctuateQ2`` for one arm (``fluctuate.R:1-60``, ``150-215``).

    ``covariate`` is the observed column, zero off the arm, and ``counterfactual`` the column
    the update is applied at.  A skip or a fallback is a guard event.
    """
    offset = trim_logit(q)
    guarded = not switches.no_guard
    if guarded and np.all(covariate < SKIP):
        log.guard_events += 1
        return q
    coef, converged = glm_binomial(y, covariate, offset, np.zeros(1))
    if guarded and (not converged or abs(float(np.max(coef))) > COEF_TOL):
        log.guard_events += 1
        coef, converged = glm_binomial(y, covariate, offset)
        if not converged or abs(float(np.max(coef))) > COEF_TOL:
            return q
    if not np.all(np.isfinite(coef)):
        raise TranscriptionFailure("an outcome fluctuation returned a non-finite coefficient")
    return linkinv(offset + coef[0] * counterfactual)


def fluctuate_g_armwise(
    a: np.ndarray,
    gn: list[np.ndarray],
    qrn: list[np.ndarray],
    switches: Switches,
    log: _Log,
    *,
    guarded: bool = True,
) -> list[np.ndarray]:
    """``fluctuateG`` (``fluctuate.R:93-126``): one tilt per arm, then ``pred[pred < tolg]``."""
    out = []
    use_guard = guarded and not switches.no_guard
    for arm, g, qr in zip(ARMS, gn, qrn, strict=True):
        covariate = qr / g
        offset = trim_logit(g, TOLG)
        indicator = (a == arm).astype(float)
        coef, converged = glm_binomial(indicator, covariate, offset, np.zeros(1))
        failed = not np.all(np.isfinite(coef)) or not converged or abs(float(coef[0])) > COEF_TOL
        if use_guard and failed:
            log.guard_events += 1
            coef, converged = glm_binomial(indicator, covariate, offset)
            if not np.all(np.isfinite(coef)) or not converged or abs(float(coef[0])) > COEF_TOL:
                coef = np.zeros(1)
        if not np.all(np.isfinite(coef)):
            raise TranscriptionFailure("a mechanism fluctuation returned a non-finite coefficient")
        prediction = linkinv(offset + coef[0] * covariate)
        prediction[prediction < TOLG] = TOLG
        out.append(prediction)
    return out


def fluctuate_g_joint(
    a: np.ndarray, gn: list[np.ndarray], qrn: list[np.ndarray]
) -> list[np.ndarray]:
    """Switch J: ``cleverly``'s binary tilt of ``g_1`` along ``(-Qr_0/g_0, Qr_1/g_1)``.

    One logistic fit of ``1(A = 1)`` with offset ``logit(g_1)``, solved to ``FINE_EPSILON``, and
    ``g_1`` clipped to ``[tolg, 1 - tolg]`` afterwards.  ``g_0`` is ``1 - g_1``.
    """
    upper = gn[1]
    design = np.column_stack([-qrn[0] / (1.0 - upper), qrn[1] / upper])
    offset = trim_logit(upper, LOGIT_GUARD)
    coef, _ = glm_binomial(
        (a == 1.0).astype(float),
        design,
        offset,
        np.zeros(2),
        maxit=FINE_MAXIT,
        epsilon=FINE_EPSILON,
    )
    if not np.all(np.isfinite(coef)):
        raise TranscriptionFailure("the joint mechanism tilt returned a non-finite coefficient")
    tilted = np.clip(expit(offset + design @ coef), TOLG, 1.0 - TOLG)
    return [1.0 - tilted, tilted]


def _tilt_g(
    a: np.ndarray,
    gn: list[np.ndarray],
    qrn: list[np.ndarray],
    switches: Switches,
    log: _Log,
    *,
    guarded: bool = True,
) -> list[np.ndarray]:
    if switches.joint_tilt:
        return fluctuate_g_joint(a, gn, qrn)
    return fluctuate_g_armwise(a, gn, qrn, switches, log, guarded=guarded)


def _joint_outcome_step(
    y: np.ndarray,
    a: np.ndarray,
    qn: list[np.ndarray],
    gn: list[np.ndarray],
    grn: list[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    """The closing pass's outcome stage: equations (8) and (10) in one solve per arm.

    ``cleverly`` solves the four columns at once.  The two arms' columns are zero on each
    other's rows, so the four-column likelihood separates into one two-column fit per arm.
    """
    out = []
    for arm, q, g, (signed, probability) in zip(ARMS, qn, gn, grn, strict=True):
        on = (a == arm).astype(float)
        counterfactual = np.column_stack([1.0 / g, signed / probability])
        offset = trim_logit(q)
        coef, _ = glm_binomial(
            y,
            on[:, None] * counterfactual,
            offset,
            np.zeros(2),
            maxit=FINE_MAXIT,
            epsilon=FINE_EPSILON,
        )
        if not np.all(np.isfinite(coef)):
            raise TranscriptionFailure("the closing outcome solve returned a non-finite value")
        out.append(expit(offset + counterfactual @ coef))
    return out


# ------------------------------------------------------------------------- the scores


@dataclass(frozen=True)
class _Scores:
    """The three equations, per arm, at one state: the score and its scale."""

    outcome: tuple[np.ndarray, np.ndarray]
    reduced: tuple[np.ndarray, np.ndarray]
    mechanism: tuple[np.ndarray, np.ndarray]

    def relative(self, which: tuple[np.ndarray, np.ndarray]) -> float:
        score, scale = which
        return float(np.max(np.abs(score) / np.maximum(scale, 1e-300)))

    @property
    def worst_relative(self) -> float:
        return max(
            self.relative(self.outcome), self.relative(self.reduced), self.relative(self.mechanism)
        )

    @property
    def largest(self) -> float:
        """R's exit statistic: the largest absolute mean of the six (``drtmle.R:607``)."""
        return float(
            max(np.max(np.abs(pair[0])) for pair in (self.outcome, self.reduced, self.mechanism))
        )

    def solved(self, n: int) -> bool:
        """``cleverly``'s exit: every equation meets ``tol`` relatively or ``1e-3/n`` absolutely."""
        bar = NEGLIGIBLE / float(n)
        return all(
            self.relative(pair) <= CLEVERLY_TOL or float(np.max(np.abs(pair[0]))) <= bar
            for pair in (self.outcome, self.reduced, self.mechanism)
        )


def _column(score: list[float], scale: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(score, dtype=float), np.asarray(scale, dtype=float)


def scores(
    y: np.ndarray,
    a: np.ndarray,
    qn: list[np.ndarray],
    gn: list[np.ndarray],
    grn: list[tuple[np.ndarray, np.ndarray]],
    qrn: list[np.ndarray],
) -> _Scores:
    """The three equations' means per arm, as R evaluates them after a round.

    Equation (8) is ``eval_Dstar``'s mean without its ``Q - psi`` term, which has mean zero by
    the definition of ``psi``; (10) is ``eval_Dstar_Q``'s and (9) ``eval_Dstar_g``'s
    (``inf_functions.R``).  Each scale is ``mean(abs(H))``, as ``cleverly``'s ``score_scale``.
    """
    outcome: tuple[list[float], list[float]] = ([], [])
    reduced: tuple[list[float], list[float]] = ([], [])
    mechanism: tuple[list[float], list[float]] = ([], [])
    for arm, q, g, (signed, probability), qr in zip(ARMS, qn, gn, grn, qrn, strict=True):
        on = (a == arm).astype(float)
        for (score, scale), h, residual in (
            (outcome, on / g, y - q),
            (reduced, on * signed / probability, y - q),
            (mechanism, qr / g, on - g),
        ):
            score.append(float(np.mean(h * residual)))
            scale.append(float(np.mean(np.abs(h))))
    return _Scores(_column(*outcome), _column(*reduced), _column(*mechanism))


def _loglik(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(np.sum(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def joint_loglik(
    y: np.ndarray, a: np.ndarray, qn: list[np.ndarray], gn: list[np.ndarray], switches: Switches
) -> float:
    """The stall rule's objective: the outcome log-likelihood plus the mechanism's.

    The outcome term reads ``Q*(A, W)``.  The mechanism term is the binomial log-likelihood of
    ``1(A = 1)`` at ``g_1`` under switch J, and the sum over arms of ``1(A = a)`` at ``g_a``
    otherwise, as ``solve_armwise_bounded_mechanism`` sums it.
    """
    observed = np.where(a == 1.0, qn[1], qn[0])
    outcome = _loglik(y, observed)
    if switches.joint_tilt:
        return outcome + _loglik((a == 1.0).astype(float), gn[1])
    return outcome + sum(
        _loglik((a == arm).astype(float), g) for arm, g in zip(ARMS, gn, strict=True)
    )


# ---------------------------------------------------------------------- the estimator


def standard_errors(
    y: np.ndarray,
    a: np.ndarray,
    qn: list[np.ndarray],
    gn: list[np.ndarray],
    grn: list[tuple[np.ndarray, np.ndarray]],
    qrn: list[np.ndarray],
) -> dict[str, float]:
    """R's ``targeted_se`` covariance (``drtmle.R:787-818``).

    Per arm ``DnoStar - DnQoStar - DngoStar`` at the final state, ``stats::cov(...) / n``, and
    the contrast's variance ``c11 + c22 - 2 c12``, as ``run_drtmle.R`` forms it.
    """
    n = y.size
    columns = []
    for arm, q, g, (signed, probability), qr in zip(ARMS, qn, gn, grn, qrn, strict=True):
        on = (a == arm).astype(float)
        psi = float(np.mean(q))
        efficient = on / g * (y - q) + q - psi
        missing_g = on / probability * signed * (y - q)
        missing_q = qr / g * (on - g)
        columns.append(efficient - missing_g - missing_q)
    covariance = np.cov(np.column_stack(columns), rowvar=False, ddof=1) / n
    return {
        "ey0": float(np.sqrt(covariance[0, 0])),
        "ey1": float(np.sqrt(covariance[1, 1])),
        "ate": float(np.sqrt(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])),
    }


def at_bound(gn: list[np.ndarray]) -> int:
    """Rows on which some arm's mechanism is at or beyond ``tolg`` or ``1 - tolg``."""
    stacked = np.column_stack(gn)
    return int(np.sum(np.any((stacked <= TOLG) | (stacked >= 1.0 - TOLG), axis=1)))


def transcribe(payload: pd.DataFrame, switches: Switches = Switches()) -> Transcribed:
    """One fit of the transcription on a study payload.

    ``payload`` carries ``Y``, ``A``, ``fold`` and the shared initial arrays ``qn0``, ``qn1`` and
    ``gn1``, exactly as ``canonical_drtmle._replicate`` hands them to R.
    """
    y = payload["Y"].to_numpy(dtype=float)
    a = payload["A"].to_numpy(dtype=float)
    n = y.size
    valid = fold_rows(payload["fold"].to_numpy())
    qn = [payload["qn0"].to_numpy(dtype=float).copy(), payload["qn1"].to_numpy(dtype=float).copy()]
    upper = payload["gn1"].to_numpy(dtype=float)
    # drtmle.R:413, the lower bound on a supplied gn.
    gn = [np.where(g < TOLG, TOLG, g) for g in (1.0 - upper, upper.copy())]
    log = _Log()
    qrn = estimate_qrn(y, a, qn, gn, valid)
    grn = estimate_grn(a, qn, gn, valid)
    if switches.prime:
        qn = [
            fluctuate_q(y, (a == arm) / g, 1.0 / g, q, switches, log)
            for arm, q, g in zip(ARMS, qn, gn, strict=True)
        ]

    iterations = 0
    exit_reason = "cap"
    previous: float | None = None
    previous_joint: float | None = None
    current: _Scores | None = None
    while iterations < MAX_ITER:
        if not switches.cleverly_exit and current is not None and current.largest <= TOL_IC:
            exit_reason = "tolIC"
            break
        iterations += 1
        gn = _tilt_g(a, gn, qrn, switches, log)
        grn = estimate_grn(a, qn, gn, valid)
        updated = []
        for arm, q, g, (signed, probability) in zip(ARMS, qn, gn, grn, strict=True):
            on = (a == arm).astype(float)
            q = fluctuate_q(y, on / probability * signed, signed / probability, q, switches, log)
            q = fluctuate_q(y, on / g, 1.0 / g, q, switches, log)
            updated.append(q)
        qn = updated
        qrn = estimate_qrn(y, a, qn, gn, valid)
        current = scores(y, a, qn, gn, grn, qrn)
        if switches.cleverly_exit:
            if current.solved(n):
                exit_reason = "tolerance"
                break
            joint = joint_loglik(y, a, qn, gn, switches)
            climbing = previous_joint is None or joint > previous_joint + CLEVERLY_TOL * (
                1.0 + abs(joint)
            )
            worst = current.worst_relative
            improving = previous is None or worst <= STALL_FACTOR * previous
            if not climbing and not improving:
                exit_reason = "stall"
                break
            previous = worst if previous is None else min(previous, worst)
            previous_joint = joint
    else:
        if not switches.cleverly_exit and current is not None and current.largest <= TOL_IC:
            exit_reason = "tolIC"

    closing = 0
    if switches.cleverly_exit:
        # _close_at_frozen_reductions: equation (9) at the frozen Qr until it meets tol on the
        # relative ruler, then equations (8) and (10) together at the frozen gr.  No guard.
        for _ in range(CLOSING_STEPS):
            closing += 1
            gn = _tilt_g(a, gn, qrn, switches, log, guarded=False)
            state = scores(y, a, qn, gn, grn, qrn)
            if state.relative(state.mechanism) <= CLEVERLY_TOL:
                break
        closing += 1
        qn = _joint_outcome_step(y, a, qn, gn, grn)

    final = scores(y, a, qn, gn, grn, qrn)
    means = [float(np.mean(q)) for q in qn]
    return Transcribed(
        estimates={"ey0": means[0], "ey1": means[1], "ate": means[1] - means[0]},
        std_errors=standard_errors(y, a, qn, gn, grn, qrn),
        iterations=iterations,
        exit=exit_reason,
        closing=closing,
        guard_events=log.guard_events,
        at_bound=at_bound(gn),
        score_max=final.largest,
    )
