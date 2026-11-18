r"""Hazard learners for the continuous-time survival engine.

A hazard learner fits one cause-specific hazard from ``(X, time, event)``, where ``event`` is a
0/1 indicator of the modelled cause and every other unit is censored for it, and returns
cumulative hazards :math:`\Lambda(t \mid x)` at requested times.  The first column of ``X`` is
the treatment by contract: the engine builds ``X = [A, L]``.

:class:`CoxHazard` is the proportional-hazards model fitted by Newton's method on the Breslow
partial likelihood, with the Breslow baseline hazard (Breslow 1972; Cox 1972).
:class:`StratifiedNelsonAalen` is the Nelson-Aalen estimator within each cell of discrete
columns, which is saturated on discrete covariates.  :func:`select_hazard` chooses among a
library of Cox models by the cross-validated partial likelihood of Verweij and van Houwelingen
(1993), as ``concrete`` selects its hazards, and refits the winner on every row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .._refusal import NO_ALTERNATIVE, RefusalReason
from .._typing import FloatArray, IntArray
from .._warnings import warn as _warn
from ..exceptions import CapabilityError, ConvergenceWarning, DataError

__all__ = [
    "CoxHazard",
    "HazardSelection",
    "StratifiedNelsonAalen",
    "partial_log_likelihood",
    "select_hazard",
]


def _gap(text: str) -> CapabilityError:
    """A refusal of a composition the continuous-time engine has no source or study for."""
    return CapabilityError(
        text, reason=RefusalReason.IMPLEMENTATION_GAP, alternative=NO_ALTERNATIVE, roadmap="X6"
    )


CoxTerms = Literal["treatment", "main", "interaction"]

#: Newton stops when the largest absolute score entry, on the standardized scale, is below
#: this value.
_SCORE_TOL: Final = 1e-10
_MAX_NEWTON: Final = 100
#: A Newton step below this share of the coefficients' size is below rounding: converged.
_STEP_TOL: Final = 1e-12
#: The widest spread (largest minus smallest) the standardized linear predictor may reach on
#: the fitting rows.  Within it every shifted risk in a risk set is at least
#: ``exp(-_SPREAD_MAX)``, a normal double, so the Breslow sums keep full relative precision.  The
#: design is centred, so the predictor spans zero and the unshifted risks of the baseline stay
#: within ``exp(+-_SPREAD_MAX)``, and a fitting row's cumulative hazard stays finite.  Fits
#: running to a monotone-likelihood boundary end at it; a regular maximum reaches it only with
#: hazard ratios of about ``exp(568)`` between two fitting rows.
_SPREAD_MAX: Final = 0.8 * float(np.log(np.finfo(float).max))
#: The infinite-coefficient test of ``survival::coxph.fit`` (Therneau): coefficient ``j`` runs
#: to infinity when the Newton step at the final fit exceeds ``_INF_EPS`` and
#: ``sqrt(_INF_EPS) * |beta_j|``.
_INF_EPS: Final = 1e-9


def _newton_step(info: FloatArray, grad: FloatArray) -> FloatArray:
    """The Newton step ``info^{-1} grad``, by least squares where ``info`` is singular."""
    try:
        return np.asarray(cho_solve(cho_factor(info), grad))
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(info, grad, rcond=None)[0])


def _column_names(terms: CoxTerms, p: int) -> list[str]:
    """The design columns in words: the treatment, covariates by their ``X`` column."""
    names = ["the treatment"]
    if terms != "treatment":
        names += [f"covariate {k}" for k in range(1, p)]
    if terms == "interaction":
        names += [f"treatment x covariate {k}" for k in range(1, p)]
    return names


def _design(X: FloatArray, terms: CoxTerms) -> FloatArray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("a hazard learner reads X as a 2-D array whose first column is A")
    if terms == "treatment":
        return X[:, :1]
    if terms == "main":
        return X
    return np.hstack([X, X[:, :1] * X[:, 1:]])


@dataclass(frozen=True)
class _RiskSets:
    """Event times and the rows at risk at each, for the Breslow partial likelihood."""

    times: FloatArray  # distinct event times, ascending
    start: IntArray  # first row (in time order) still at risk at each event time
    deaths: FloatArray  # number of events at each event time
    order: IntArray  # rows sorted by time ascending
    event_sum_rows: IntArray  # rows with an event


def _risk_sets(time: FloatArray, event: FloatArray) -> _RiskSets:
    order = np.argsort(time, kind="stable")
    sorted_time = time[order]
    events = event > 0
    times, counts = np.unique(time[events], return_counts=True)
    start = np.searchsorted(sorted_time, times, side="left")
    # Breslow tie counts, one grouped count instead of a scan of every row per event time.
    # A NaN time counts no deaths, as an equality test would find none.
    deaths = np.where(np.isnan(times), 0, counts).astype(float)
    return _RiskSets(times, start, deaths, order, np.flatnonzero(events))


def _tail(values: FloatArray, start: IntArray) -> FloatArray:
    """Sums of ``values[start:]`` along axis 0, for each start index."""
    cum = np.concatenate([np.cumsum(values[::-1], axis=0)[::-1], np.zeros((1, *values.shape[1:]))])
    return cum[start]


#: Bytes one block of the risk second moments may take: the ``(n, b, q)`` products, their
#: reversed running sums and the ``(M, b, q)`` sums at the event times are each about this size.
#: A block holds at least one row, so when ``8 n q`` exceeds this budget a block is ``8 n q``.
_BLOCK_BYTES: Final = 32 * 2**20


def _loglik_terms(
    beta: FloatArray, Z: FloatArray, sets: _RiskSets
) -> tuple[float, FloatArray, FloatArray]:
    """The log partial likelihood, the risks in time order, and their sums ``s0``."""
    eta = Z @ beta
    shift = float(np.max(eta)) if eta.size else 0.0
    risk = np.exp(eta - shift)
    rs = risk[sets.order]
    s0 = _tail(rs, sets.start)
    loglik = float(np.sum(eta[sets.event_sum_rows]) - np.sum(sets.deaths * (np.log(s0) + shift)))
    return loglik, rs, s0


def _loglik(beta: FloatArray, Z: FloatArray, sets: _RiskSets) -> float:
    """Breslow log partial likelihood alone, which is all a step-halving trial reads."""
    return _loglik_terms(beta, Z, sets)[0]


def _loglik_parts(
    beta: FloatArray, Z: FloatArray, sets: _RiskSets
) -> tuple[float, FloatArray, FloatArray]:
    """Breslow log partial likelihood, its gradient and its negative Hessian.

    The risk second moments are built a block of Hessian rows at a time, so no ``(n, q, q)``
    array exists. Each entry is the same product, running sum and sum over event times that
    one full array would hold, so the result does not depend on the block size.
    """
    loglik, rs, s0 = _loglik_terms(beta, Z, sets)
    zs = Z[sets.order]
    s1 = _tail(rs[:, None] * zs, sets.start)
    zbar = s1 / s0[:, None]
    grad = Z[sets.event_sum_rows].sum(axis=0) - (sets.deaths[:, None] * zbar).sum(axis=0)
    q = zs.shape[1]
    info = np.full((q, q), np.nan)  # a row no block writes stays visible
    block = max(1, _BLOCK_BYTES // (8 * max(1, zs.shape[0] * q)))
    for first in range(0, q, block):
        rows = slice(first, min(q, first + block))
        s2 = _tail(rs[:, None, None] * zs[:, rows, None] * zs[:, None, :], sets.start)
        info[rows] = (
            sets.deaths[:, None, None]
            * (s2 / s0[:, None, None] - zbar[:, rows, None] * zbar[:, None, :])
        ).sum(axis=0)
    return loglik, grad, info


class CoxHazard:
    r"""A Cox proportional-hazards model of one cause-specific hazard.

    Parameters
    ----------
    terms : {"treatment", "main", "interaction"}
        The linear predictor: the treatment alone, the treatment and the covariates, or those
        and every treatment-by-covariate product.
    penalty : float
        A ridge penalty :math:`\tfrac{1}{2}\,\text{penalty}\,\lVert\beta\rVert^2` on the
        standardized coefficients.  Zero fits the maximum partial-likelihood estimate.

    Notes
    -----
    Ties use the Breslow partial likelihood, and the baseline is the Breslow estimate
    :math:`d\Lambda_0(u) = d(u) / \sum_{i:\, T_i \ge u} e^{x_i^\top\beta}`, so
    ``CoxHazard("treatment")`` at :math:`\beta = 0` is the Nelson-Aalen estimator.

    Newton steps are halved until the likelihood does not fall and the standardized linear
    predictor's spread over the fitting rows stays within :math:`0.8 \log(\text{max double})`,
    about 568, where the Breslow sums keep full precision. A regular fit ends inside that range.
    On separated data (a cause with no events in one arm, or a covariate that orders the event
    times) the partial likelihood keeps rising along one direction toward a limit it never
    reaches, so no maximum exists. The fit then either ends pinned at the range's edge, or meets
    the score tolerance with the coefficients still running, which the infinite-coefficient test
    of R's ``survival::coxph`` usually detects (a heuristic: a small separated sample can pass
    it). A detected case sets ``monotone_likelihood_`` and warns with a
    :class:`~cleverly.exceptions.ConvergenceWarning` naming the coefficients that may be
    infinite. That fit is not a maximum-likelihood estimate: it is the point its Newton path
    reached, where the hazards the separating direction sends toward zero or infinity are
    effectively at their limit. A positive ``penalty`` gives the likelihood a finite maximum.
    ``converged_`` says whether the score met its tolerance. A regular maximum with hazard ratios
    of about :math:`e^{568}` between two fitting rows is beyond the range and is not reached.
    """

    def __init__(self, terms: CoxTerms = "main", penalty: float = 0.0) -> None:
        if terms not in ("treatment", "main", "interaction"):
            raise ValueError("CoxHazard(terms=) is 'treatment', 'main' or 'interaction'")
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("CoxHazard(penalty=) is a finite number at or above zero")
        self.terms: CoxTerms = terms
        self.penalty = float(penalty)

    def __repr__(self) -> str:
        extra = f", penalty={self.penalty:g}" if self.penalty else ""
        return f"CoxHazard({self.terms!r}{extra})"

    @property
    def label(self) -> str:
        """The model in words, as a summary prints it."""
        terms = {"treatment": "treatment only", "main": "main terms", "interaction": "interactions"}
        extra = f", penalty {self.penalty:g}" if self.penalty else ""
        return f"Cox, {terms[self.terms]}{extra}"

    def _standardize(self, Z: FloatArray) -> FloatArray:
        return (Z - self.center_) / self.scale_

    def fit(self, X: Any, time: Any, event: Any) -> CoxHazard:
        """Fit the coefficients and the Breslow baseline.

        Parameters
        ----------
        X : array-like
            ``(n, p)`` with the treatment in the first column.
        time : array-like
            ``(n,)`` follow-up times.
        event : array-like
            ``(n,)`` 0/1 indicators of the modelled cause.

        Returns
        -------
        CoxHazard
            This learner, fitted.
        """
        Z = _design(X, self.terms)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=float)
        self.center_ = Z.mean(axis=0)
        scale = Z.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        Zs = self._standardize(Z)
        sets = _risk_sets(time, event)
        beta = np.zeros(Zs.shape[1])
        converged = sets.times.size == 0
        # The last full Newton step left the resolved range (or read a non-finite value).
        unresolved_step = False
        info = np.zeros((beta.size, beta.size))
        grad = np.zeros(beta.size)
        for _ in range(_MAX_NEWTON):
            if converged:
                break
            loglik, grad, info = _loglik_parts(beta, Zs, sets)
            grad = grad - self.penalty * beta
            info = info + self.penalty * np.eye(beta.size)
            if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(info))):
                # Inside the resolved range only a non-finite input reaches this.
                break
            if np.max(np.abs(grad)) < _SCORE_TOL:
                converged = True
                break
            step = _newton_step(info, grad)
            # At n in the thousands the summed score can stall just above its tolerance while
            # the Newton step is already below rounding; that is convergence, not a boundary,
            # where the step grows instead.
            if np.max(np.abs(step)) <= _STEP_TOL * (1.0 + np.max(np.abs(beta))):
                converged = True
                break
            # Halve until the penalized likelihood is resolved and does not fall.  A trial whose
            # linear predictor leaves the resolved range is halved without being evaluated: its
            # Breslow sums would lose precision or underflow, and an underflowed risk set reads
            # as +inf.  A regular fit's Newton step can leave the range too (one heavy-tailed
            # covariate), so leaving it is halved like a fall, never taken as a boundary.
            target = loglik - 0.5 * self.penalty * beta @ beta
            accepted = False
            for halving in range(30):
                trial = beta + step
                resolved = bool(np.ptp(Zs @ trial) <= _SPREAD_MAX)
                value = (
                    _loglik(trial, Zs, sets) - 0.5 * self.penalty * trial @ trial
                    if resolved
                    else np.nan
                )
                if halving == 0:
                    unresolved_step = not np.isfinite(value)
                if np.isfinite(value) and value >= target - 1e-12:
                    accepted = True
                    break
                step = step / 2
            if not accepted:
                # No step keeps the likelihood resolved and not lower: the fit stays here.
                break
            beta = beta + step
        reach = float(np.ptp(Zs @ beta)) if beta.size else 0.0
        if converged and beta.size:
            # coxph's infinite-coefficient test, on the standardized scale where its relative
            # criterion is scale-free: at a finite maximum the Newton step is at rounding; along
            # a separating direction the score and the information decay together and the step
            # stays near the inverse of the separation's gap.
            final_step = np.abs(np.linalg.pinv(info) @ grad)
            runaway = (final_step > _INF_EPS) & (final_step > np.sqrt(_INF_EPS) * np.abs(beta))
            # Columns coupled to the run through the inverse information can pass the test with a
            # step orders of magnitude below the run's own; they are not named.
            runaway &= final_step >= 0.1 * np.max(final_step)
            boundary = bool(np.any(runaway))
        else:
            # Pinned at the edge of the resolved range by steps that keep leaving it.
            boundary = unresolved_step and reach > 0.5 * _SPREAD_MAX and not self.penalty
            runaway = (
                np.abs(beta) >= 0.5 * np.max(np.abs(beta))
                if boundary
                else np.zeros(beta.size, bool)
            )
        if boundary:
            names = _column_names(self.terms, np.asarray(X).shape[1])
            diverging = ", ".join(n for n, r in zip(names, runaway, strict=True) if r)
            _warn(
                f"{self!r} has a monotone partial likelihood: the coefficients of {diverging} may "
                "be infinite. The fit stops along its Newton path where the likelihood is still "
                "resolved in double precision, so the hazards it sends toward zero or infinity "
                "are effectively at their limit; it is not a maximum-likelihood estimate. A ridge "
                "penalty, such as CoxHazard(..., penalty=1.0), has a finite maximum; a "
                "StratifiedNelsonAalen on a separating discrete column fits the limit exactly.",
                ConvergenceWarning,
            )
        elif not converged:
            _warn(
                f"{self!r} did not reach a score below {_SCORE_TOL:g} in {_MAX_NEWTON} Newton "
                "steps; the coefficients are the last fit whose likelihood was resolved.",
                ConvergenceWarning,
            )
        self.monotone_likelihood_ = boundary
        self.converged_ = converged
        self.coef_ = beta / self.scale_
        self.intercept_shift_ = float(self.center_ / self.scale_ @ beta)
        risk = np.exp(Zs @ beta)
        order = sets.order
        s0 = _tail(risk[order], sets.start)
        self.baseline_times_ = sets.times
        self.baseline_jumps_ = sets.deaths / s0
        self._zs_beta = beta
        return self

    def linear_predictor(self, X: Any) -> FloatArray:
        """The fitted :math:`x^\\top\\beta` on the standardized scale the baseline reads.

        Parameters
        ----------
        X : array-like
            ``(n, p)`` with the treatment in the first column.

        Returns
        -------
        ndarray
            ``(n,)`` linear predictors.
        """
        return self._standardize(_design(X, self.terms)) @ self._zs_beta

    def cumulative_hazard(self, X: Any, times: Any) -> FloatArray:
        """:math:`\\Lambda(t \\mid x)` at each requested time, right-continuous.

        Parameters
        ----------
        X : array-like
            ``(n, p)`` with the treatment in the first column.
        times : array-like
            ``(m,)`` times.

        Returns
        -------
        ndarray
            ``(n, m)`` cumulative hazards.
        """
        scale, base = self.hazard_factors(X, times)
        return scale[:, None] * base[None, :]

    def hazard_factors(self, X: Any, times: Any) -> tuple[FloatArray, FloatArray]:
        """The two factors of :meth:`cumulative_hazard`: one scale per row, one baseline per time.

        ``cumulative_hazard(X, times)`` is ``scale[:, None] * base[None, :]`` bit for bit.  The
        scale :math:`e^{x^\\top\\beta}` of a row depends on the rows it is computed with in its
        last bits (a matrix-vector product), so a caller that rebuilds the hazards of some rows
        later keeps the scales of one call rather than recomputing them on a subset.

        Parameters
        ----------
        X : array-like
            ``(n, p)`` with the treatment in the first column.
        times : array-like
            ``(m,)`` times.

        Returns
        -------
        scale : ndarray
            ``(n,)`` :math:`e^{x_i^\\top\\beta}`.
        base : ndarray
            ``(m,)`` the Breslow cumulative baseline hazard at each time, right-continuous.
        """
        times = np.asarray(times, dtype=float)
        cum = np.concatenate([[0.0], np.cumsum(self.baseline_jumps_)])
        base = cum[np.searchsorted(self.baseline_times_, times, side="right")]
        return np.exp(self.linear_predictor(X)), base


def partial_log_likelihood(model: CoxHazard, X: Any, time: Any, event: Any) -> float:
    """The Breslow log partial likelihood of fitted coefficients on a data set.

    Parameters
    ----------
    model : CoxHazard
        A fitted model.
    X : array-like
        ``(n, p)`` with the treatment in the first column.
    time : array-like
        ``(n,)`` follow-up times.
    event : array-like
        ``(n,)`` 0/1 indicators.

    Returns
    -------
    float
        The log partial likelihood at the model's coefficients.
    """
    Zs = model._standardize(_design(X, model.terms))
    sets = _risk_sets(np.asarray(time, dtype=float), np.asarray(event, dtype=float))
    if sets.times.size == 0:
        return 0.0
    return _loglik(model._zs_beta, Zs, sets)


class StratifiedNelsonAalen:
    """The Nelson-Aalen cumulative hazard within each cell of discrete columns.

    Parameters
    ----------
    by : sequence of int
        The columns of ``X`` that define the cells (``0`` is the treatment).

    Notes
    -----
    Saturated on discrete covariates, so with every column in ``by`` the fit is the
    nonparametric maximum-likelihood hazard.  A cell absent from the fitting rows is refused
    at prediction.
    """

    def __init__(self, by: Sequence[int] = (0,)) -> None:
        self.by = tuple(int(c) for c in by)

    def __repr__(self) -> str:
        return f"StratifiedNelsonAalen(by={self.by})"

    @property
    def label(self) -> str:
        """The model in words, as a summary prints it."""
        return "Nelson-Aalen by treatment" if self.by == (0,) else "Nelson-Aalen by stratum"

    def _keys(self, X: Any) -> list[tuple[float, ...]]:
        X = np.asarray(X, dtype=float)
        return [tuple(row) for row in X[:, list(self.by)]]

    def fit(self, X: Any, time: Any, event: Any) -> StratifiedNelsonAalen:
        """Fit the cell hazards.

        Parameters
        ----------
        X : array-like
            ``(n, p)``.
        time : array-like
            ``(n,)`` follow-up times.
        event : array-like
            ``(n,)`` 0/1 indicators.

        Returns
        -------
        StratifiedNelsonAalen
            This learner, fitted.
        """
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=float)
        keys = self._keys(X)
        self.cells_: dict[tuple[float, ...], tuple[FloatArray, FloatArray]] = {}
        for key in sorted(set(keys)):
            rows = np.array([k == key for k in keys])
            t, e = time[rows], event[rows]
            jump_times = np.unique(t[e > 0])
            at_risk = np.array([np.count_nonzero(t >= u) for u in jump_times], dtype=float)
            deaths = np.array([np.count_nonzero((t == u) & (e > 0)) for u in jump_times], float)
            self.cells_[key] = (jump_times, deaths / at_risk if jump_times.size else deaths)
        return self

    def cumulative_hazard(self, X: Any, times: Any) -> FloatArray:
        """:math:`\\Lambda(t \\mid x)` at each requested time, right-continuous.

        Parameters
        ----------
        X : array-like
            ``(n, p)``.
        times : array-like
            ``(m,)`` times.

        Returns
        -------
        ndarray
            ``(n, m)`` cumulative hazards.
        """
        index, curves = self.hazard_cells(X, times)
        return curves[index]

    def hazard_cells(self, X: Any, times: Any) -> tuple[IntArray, FloatArray]:
        """The cell of each row and each cell's cumulative hazard.

        These are the parts of :meth:`cumulative_hazard`: ``cumulative_hazard(X, times)`` is
        ``curves[index]`` bit for bit.

        Parameters
        ----------
        X : array-like
            ``(n, p)``.
        times : array-like
            ``(m,)`` times.

        Returns
        -------
        index : ndarray
            ``(n,)`` the row of ``curves`` that holds each row's cell.
        curves : ndarray
            ``(cells, m)`` the cumulative hazard of each cell the rows fall in, in order of
            first appearance.
        """
        times = np.asarray(times, dtype=float)
        keys = self._keys(X)
        index = np.empty(len(keys), dtype=int)
        position: dict[tuple[float, ...], int] = {}
        rows: list[FloatArray] = []
        for i, key in enumerate(keys):
            if key not in position:
                if key not in self.cells_:
                    raise _gap(
                        f"{self!r} has no fitted cell {key}; a saturated hazard predicts only "
                        "the cells it saw."
                    )
                jump_times, jumps = self.cells_[key]
                cum = np.concatenate([[0.0], np.cumsum(jumps)])
                position[key] = len(rows)
                rows.append(cum[np.searchsorted(jump_times, times, side="right")])
            index[i] = position[key]
        curves = np.vstack(rows) if rows else np.empty((0, times.size))
        return index, curves


@dataclass(frozen=True)
class HazardSelection:
    """The cross-validated choice among a hazard library.

    Parameters
    ----------
    names : tuple of str
        The library members.
    losses : tuple of float
        Each member's cross-validated negative log partial likelihood.
    chosen : int
        The index of the selected member.
    model : object
        The selected member refitted on every row.
    """

    names: tuple[str, ...]
    losses: tuple[float, ...]
    chosen: int
    model: Any

    @property
    def chosen_name(self) -> str:
        """The selected member's name."""
        return self.names[self.chosen]


def _clone(learner: Any) -> Any:
    if isinstance(learner, CoxHazard):
        return CoxHazard(learner.terms, learner.penalty)
    if isinstance(learner, StratifiedNelsonAalen):
        return StratifiedNelsonAalen(learner.by)
    from sklearn.base import clone

    return clone(learner)


def select_hazard(
    library: Sequence[Any],
    X: Any,
    time: Any,
    event: Any,
    folds: IntArray | None,
) -> HazardSelection:
    """Select one hazard learner by cross-validated partial likelihood and refit it.

    Parameters
    ----------
    library : sequence
        Hazard learners.  More than one member must be a :class:`CoxHazard`, whose partial
        likelihoods compare over the same risk sets.
    X : array-like
        ``(n, p)`` with the treatment in the first column.
    time : array-like
        ``(n,)`` follow-up times.
    event : array-like
        ``(n,)`` 0/1 indicators of the modelled cause.
    folds : ndarray or None
        ``(n,)`` fold labels for the cross-validation, or ``None`` with a one-member library.

    Returns
    -------
    HazardSelection
        The losses, the choice and the refitted model.

    Notes
    -----
    The loss of a member on fold :math:`v` is
    :math:`-\\{\\ell(\\hat\\beta_{-v}) - \\ell_{-v}(\\hat\\beta_{-v})\\}`, the log partial
    likelihood of all rows at the coefficients fitted without fold :math:`v` minus that of the
    training rows (Verweij and van Houwelingen 1993).  It compares members over the same risk
    sets, which a stratified model or a nonparametric learner would not.
    """
    library = list(library)
    names = tuple(repr(member) for member in library)
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if len(library) == 1:
        return HazardSelection(names, (float("nan"),), 0, _clone(library[0]).fit(X, time, event))
    if not all(isinstance(member, CoxHazard) for member in library):
        raise _gap(
            "A hazard library of more than one member holds CoxHazard models only: the "
            "cross-validated partial likelihood compares members over the same risk sets, which "
            "a stratified or nonparametric learner does not share. A comparable loss for mixed "
            "libraries is planned (roadmap X6)."
        )
    if folds is None:
        raise ValueError("select_hazard needs fold labels for a library of several members")
    losses = []
    for member in library:
        total = 0.0
        for fold in np.unique(folds):
            train = folds != fold
            fitted = _clone(member).fit(X[train], time[train], event[train])
            total -= partial_log_likelihood(fitted, X, time, event) - partial_log_likelihood(
                fitted, X[train], time[train], event[train]
            )
        losses.append(total)
    # A member whose held-out likelihood is not finite (coefficients at a monotone-likelihood
    # boundary) cannot win; np.argmin would pick a nan.
    ranked = np.where(np.isfinite(losses), losses, np.inf)
    if not np.isfinite(ranked).any():
        raise DataError(
            "Every member of the hazard library has a non-finite cross-validated loss: each one's "
            "coefficients run to a monotone-likelihood boundary on some training fold. Add a "
            "penalized member, such as CoxHazard('main', penalty=1.0)."
        )
    chosen = int(np.argmin(ranked))
    model = _clone(library[chosen]).fit(X, time, event)
    return HazardSelection(names, tuple(float(v) for v in losses), chosen, model)


def default_library() -> tuple[CoxHazard, ...]:
    """The default hazard library: three unstratified Cox models.

    Returns
    -------
    tuple of CoxHazard
        ``CoxHazard("treatment")``, ``CoxHazard("main")`` and ``CoxHazard("interaction")``.
    """
    return (CoxHazard("treatment"), CoxHazard("main"), CoxHazard("interaction"))
