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
    times = np.unique(time[events])
    start = np.searchsorted(sorted_time, times, side="left")
    deaths = np.array([np.count_nonzero(events & (time == t)) for t in times], dtype=float)
    return _RiskSets(times, start, deaths, order, np.flatnonzero(events))


def _tail(values: FloatArray, start: IntArray) -> FloatArray:
    """Sums of ``values[start:]`` along axis 0, for each start index."""
    cum = np.concatenate([np.cumsum(values[::-1], axis=0)[::-1], np.zeros((1, *values.shape[1:]))])
    return cum[start]


def _loglik_parts(
    beta: FloatArray, Z: FloatArray, sets: _RiskSets
) -> tuple[float, FloatArray, FloatArray]:
    """Breslow log partial likelihood, its gradient and its negative Hessian."""
    eta = Z @ beta
    shift = float(np.max(eta)) if eta.size else 0.0
    risk = np.exp(eta - shift)
    zs = Z[sets.order]
    rs = risk[sets.order]
    s0 = _tail(rs, sets.start)
    s1 = _tail(rs[:, None] * zs, sets.start)
    s2 = _tail(rs[:, None, None] * zs[:, :, None] * zs[:, None, :], sets.start)
    zbar = s1 / s0[:, None]
    loglik = float(np.sum(eta[sets.event_sum_rows]) - np.sum(sets.deaths * (np.log(s0) + shift)))
    grad = Z[sets.event_sum_rows].sum(axis=0) - (sets.deaths[:, None] * zbar).sum(axis=0)
    info = (
        sets.deaths[:, None, None] * (s2 / s0[:, None, None] - zbar[:, :, None] * zbar[:, None, :])
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
        for _ in range(_MAX_NEWTON):
            if converged:
                break
            loglik, grad, info = _loglik_parts(beta, Zs, sets)
            grad = grad - self.penalty * beta
            info = info + self.penalty * np.eye(beta.size)
            if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(info))):
                # A risk set's weights underflowed: the same monotone-likelihood boundary.
                break
            if np.max(np.abs(grad)) < _SCORE_TOL:
                converged = True
                break
            try:
                step = cho_solve(cho_factor(info), grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(info, grad, rcond=None)[0]
            # At n in the thousands the summed score can stall just above its tolerance while
            # the Newton step is already below rounding; that is convergence, not a boundary,
            # where the step grows instead.
            if np.max(np.abs(step)) <= _STEP_TOL * (1.0 + np.max(np.abs(beta))):
                converged = True
                break
            # Halve until the penalized likelihood does not fall.
            target = loglik - 0.5 * self.penalty * beta @ beta
            accepted = False
            for _halving in range(30):
                trial = beta + step
                value = _loglik_parts(trial, Zs, sets)[0] - 0.5 * self.penalty * trial @ trial
                if value >= target - 1e-12:  # False for a non-finite value
                    accepted = True
                    break
                step = step / 2
            if not accepted:
                # No step keeps the likelihood finite and not lower: the coefficients run to a
                # monotone-likelihood boundary, so they stay at the last finite fit.
                break
            beta = beta + step
        if not converged:
            _warn(
                f"{self!r} did not reach a score below {_SCORE_TOL:g} in {_MAX_NEWTON} Newton "
                "steps; the coefficients may sit on a monotone-likelihood boundary.",
                ConvergenceWarning,
            )
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
        times = np.asarray(times, dtype=float)
        cum = np.concatenate([[0.0], np.cumsum(self.baseline_jumps_)])
        base = cum[np.searchsorted(self.baseline_times_, times, side="right")]
        return np.exp(self.linear_predictor(X))[:, None] * base[None, :]


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
    return _loglik_parts(model._zs_beta, Zs, sets)[0]


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
        times = np.asarray(times, dtype=float)
        keys = self._keys(X)
        out = np.empty((len(keys), times.size))
        cache: dict[tuple[float, ...], FloatArray] = {}
        for i, key in enumerate(keys):
            if key not in cache:
                if key not in self.cells_:
                    raise _gap(
                        f"{self!r} has no fitted cell {key}; a saturated hazard predicts only "
                        "the cells it saw."
                    )
                jump_times, jumps = self.cells_[key]
                cum = np.concatenate([[0.0], np.cumsum(jumps)])
                cache[key] = cum[np.searchsorted(jump_times, times, side="right")]
            out[i] = cache[key]
        return out


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
