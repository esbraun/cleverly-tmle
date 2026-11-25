r"""Segments, folds and nuisance fits of the continuous-time longitudinal TMLE (X6 part (e)).

**Segments.**  Segment :math:`k` of a unit runs from its visit :math:`k` at :math:`s_k` to its
next exit: the next visit, a death or censoring.  Its history is fixed on the segment and enters
every model through the window

.. math::

    x_k = \bigl(A(s_k), W, L(s_k), A(s_{k-1}), k, s_k\bigr),

with :math:`A(s_{-1})` read as zero, the treatment first (the hazard learners' contract).  Every
intensity is a calendar-time model on the segment rows with delayed entry at :math:`s_k` (the
Andersen-Gill form), pooled over :math:`k`.  The window is a restriction of the observed-data
model (:data:`HISTORY_RESTRICTION`).

**The visit cap.**  For a regime, :math:`K^*` is the largest :math:`K` at which every outer
training fold holds at least :data:`VISIT_CAP_MIN_FOLLOWERS` units that followed the regime
through visit :math:`K - 1` and visit a :math:`K`-th time before the last target time.  Visits
after :math:`K^*` are ignored: their rows merge into segment :math:`K^*`, which runs to the
unit's exit with the design :math:`x_{K^*}`, so its only exits are deaths.  Past the cap,
censoring and the hazards read the window of visit :math:`K^*` (:data:`HISTORY_RESTRICTION`).
:math:`K^*` is chosen once on the full sample, so it reads every fold's held-out units too: a
discrete tuning choice, not fitted out of fold.

**Ties.**  Within one time the order is death, visit, censoring.  The visit model moves each
death below the visit times it ties with, the censoring model moves each death and each visit
below the censoring times they tie with, and a censoring at a unit's own visit belongs to the
segment the visit starts, by half the smallest gap between distinct times (the shift of the
baseline engine).

**Folds** are drawn per unit and broadcast to the unit's segment and visit rows, in the outer
cross-fitting and in every library's inner cross-validation, so no unit's rows straddle a fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .._typing import FloatArray, IntArray
from ..exceptions import DataError
from ..learners._fitting import seed_unset
from .cox import HazardSelection, _clone, select_hazard
from .engine import PropensitySelection, _fit_propensity, _stratified_folds

__all__ = [
    "HISTORY_RESTRICTION",
    "VISIT_CAP_MIN_FOLLOWERS",
    "ContinuationModel",
    "Segments",
    "visit_cap",
]

#: The fewest units, in every outer training fold, that must follow a regime through visit
#: :math:`K - 1` and visit a :math:`K`-th time before the last target time for visit :math:`K`
#: to be modelled (the visit cap :math:`K^*`).
VISIT_CAP_MIN_FOLLOWERS: Final = 5

#: The model restriction the window states, recorded on every fit (X70's
#: ``markov_restriction`` in continuous time).
HISTORY_RESTRICTION: Final = (
    "Every intensity, the treatment model and the continuation read a unit's history only "
    "through the window (W, L(s_k), A(s_k), A(s_{k-1}), k, s_k) at its last visit s_k, the "
    "treatment model without A(s_k): a statistical-model restriction, not a property of the "
    "estimator. Add summaries of a longer history (lags of L, cumulative treatment, the time "
    "since the last switch) as time_varying columns to widen it. Past the visit cap K*, "
    "censoring and the hazards read the window of visit K*, not of the unit's later visits, so "
    "the fit identifies the capped risk only when censoring after visit K* + 1 does not depend "
    "on the later L or A given that window."
)


@dataclass(frozen=True)
class Segments:
    """The segment rows of every unit under one visit cap.

    Parameters
    ----------
    unit : ndarray
        ``(R,)`` each row's unit.
    visit : ndarray
        ``(R,)`` the position, in the data's visit arrays, of the visit that starts the row.
    k : ndarray
        ``(R,)`` the visit count.
    entry : ndarray
        ``(R,)`` :math:`s_k`.
    stop : ndarray
        ``(R,)`` the next exit's time.
    exit : ndarray
        ``(R,)`` ``-1`` for a next visit, a cause code for a death, ``0`` for censoring (or
        follow-up beyond the data).
    next_visit : ndarray
        ``(R,)`` the visit position of the next visit, ``-1`` when the row does not end in one.
    cap : int
        The visit cap.
    """

    unit: IntArray
    visit: IntArray
    k: IntArray
    entry: FloatArray
    stop: FloatArray
    exit: IntArray
    next_visit: IntArray
    cap: int

    @property
    def size(self) -> int:
        """The number of rows."""
        return int(self.unit.size)


def kept_visits(data: Any, last: float) -> np.ndarray:
    """``(V,)`` the visits before the last target time: the later ones no target reads.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.
    last : float
        The last target time.

    Returns
    -------
    ndarray
        The mask.
    """
    return (data.visit_index == 0) | (data.visit_time < last)


def segments(data: Any, cap: int, last: float) -> Segments:
    """The segment rows of every unit, visits after ``cap`` or at or after ``last`` ignored.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.
    cap : int
        The visit cap.
    last : float
        The last target time.

    Returns
    -------
    Segments
        The rows, unit-major and in time order.
    """
    kept = kept_visits(data, last) & (data.visit_index <= cap)
    position = np.flatnonzero(kept)
    unit = data.unit[position]
    following = np.r_[unit[1:] == unit[:-1], False]
    next_visit = np.where(following, np.r_[position[1:], -1], -1)
    stop = np.where(following, data.visit_time[np.maximum(next_visit, 0)], data.time[unit])
    exit_code = np.where(following, -1, data.event[unit])
    return Segments(
        unit=unit,
        visit=position,
        k=data.visit_index[position],
        entry=data.visit_time[position],
        stop=stop.astype(float),
        exit=exit_code.astype(int),
        next_visit=next_visit.astype(int),
        cap=int(cap),
    )


def shift(data: Any) -> float:
    """Half the smallest gap between distinct observed times, the tie-breaking shift.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.

    Returns
    -------
    float
        The shift.
    """
    times = np.unique(np.concatenate([data.visit_time, data.time]))
    gap = np.diff(times)
    return 0.5 * float(gap.min()) if gap.size else 0.5


def design(
    data: Any, coded: IntArray, visits: IntArray, slot: IntArray | None = None
) -> FloatArray:
    r"""``(R, p)`` the window :math:`x_k` at the given visits, the treatment first.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.
    coded : ndarray
        ``(V,)`` the treatment coded 0/1.
    visits : ndarray
        ``(R,)`` visit positions.
    slot : ndarray or None
        ``(R,)`` the coded treatment to place at :math:`A(s_k)`, the regime's slot; ``None``
        places the observed one.

    Returns
    -------
    ndarray
        ``[A(s_k), W, L(s_k), A(s_{k-1}), k, s_k]``.
    """
    current = coded[visits] if slot is None else np.asarray(slot)
    return np.column_stack([current.astype(float), treatment_design(data, coded, visits)])


def treatment_design(data: Any, coded: IntArray, visits: IntArray) -> FloatArray:
    """``(R, p - 1)`` the window without :math:`A(s_k)`: the treatment model's design.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.
    coded : ndarray
        ``(V,)`` the treatment coded 0/1.
    visits : ndarray
        ``(R,)`` visit positions.

    Returns
    -------
    ndarray
        ``[W, L(s_k), A(s_{k-1}), k, s_k]``.
    """
    index = data.visit_index[visits]
    previous = np.where(index > 0, coded[np.maximum(visits - 1, 0)], 0)
    return np.column_stack(
        [
            data.baseline[data.unit[visits]],
            data.covariates[visits],
            previous.astype(float),
            index.astype(float),
            data.visit_time[visits],
        ]
    )


def unit_folds(event: IntArray, n_folds: int, rng: np.random.Generator) -> IntArray:
    """Seeded fold labels per unit, stratified by event code (the baseline engine's draw).

    Parameters
    ----------
    event : ndarray
        ``(n,)`` event codes.
    n_folds : int
        Folds.
    rng : Generator
        The fit's generator.

    Returns
    -------
    ndarray
        ``(n,)`` labels.
    """
    return _stratified_folds(event, n_folds, rng)


def visit_cap(
    data: Any,
    followed: np.ndarray,
    folds: IntArray,
    last: float,
    cross_fit: bool,
    minimum: int | None = None,
) -> tuple[int, int, int]:
    """The visit cap :math:`K^*` of a regime, and the binding fold and count past it.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.
    followed : ndarray
        ``(V,)`` whether the unit followed the regime at every visit up to and including this
        one.
    folds : ndarray
        ``(n,)`` the outer fold of each unit.
    last : float
        The last target time.
    cross_fit : bool
        Whether the folds are training folds; otherwise the full sample is the one fold.
    minimum : int or None
        The fewest followers; ``None`` reads :data:`VISIT_CAP_MIN_FOLLOWERS` now, so a test can
        patch it.

    Returns
    -------
    cap : int
        :math:`K^*`, ``0`` when no follower visits after baseline before ``last``.
    fold : int
        The training fold that fails at :math:`K^* + 1` (``-1`` with none).
    count : int
        Its count of followers with a visit :math:`K^* + 1`.
    """
    m = VISIT_CAP_MIN_FOLLOWERS if minimum is None else int(minimum)
    index, time, unit = data.visit_index, data.visit_time, data.unit
    labels = np.unique(folds) if cross_fit else np.array([0])
    K = 0
    while True:
        K += 1
        at = np.flatnonzero((index == K) & (time < last))
        # The visit K of a unit that followed through visit K - 1.
        through = followed[at - 1]
        units = unit[at[through]]
        if units.size == 0:
            return K - 1, -1, 0
        counts = []
        for fold in labels:
            train = folds[units] != fold if cross_fit else np.ones(units.size, dtype=bool)
            counts.append(int(np.count_nonzero(train)))
        worst = int(np.argmin(counts))
        if counts[worst] < m:
            return K - 1, int(labels[worst]), counts[worst]


# ---------------------------------------------------------------------- fitting


def fit_hazard(
    library: Sequence[Any],
    X: FloatArray,
    time: FloatArray,
    event: FloatArray,
    entry: FloatArray,
    folds: IntArray,
    weight: FloatArray | None,
) -> HazardSelection:
    """Select and fit one hazard with delayed entry; under the weight seam, fit one member.

    Parameters
    ----------
    library : sequence
        Hazard learners.
    X : ndarray
        ``(R, p)`` designs.
    time : ndarray
        ``(R,)`` stops.
    event : ndarray
        ``(R,)`` 0/1 indicators.
    entry : ndarray
        ``(R,)`` entries.
    folds : ndarray
        ``(R,)`` inner fold labels, by unit.
    weight : ndarray or None
        ``(R,)`` the private row weights.

    Returns
    -------
    HazardSelection
        The fitted model.
    """
    if weight is None:
        return select_hazard(library, X, time, event, folds, entry=entry)
    if len(library) != 1:
        raise ValueError("the private weight seam fits one-member libraries only")
    model = _clone(library[0]).fit(X, time, event, entry=entry, weight=weight)
    return HazardSelection((repr(library[0]),), (float("nan"),), 0, model)


def fit_treatment(
    learner: Any,
    X: FloatArray,
    A: IntArray,
    train: np.ndarray,
    predict: np.ndarray,
    folds: IntArray,
    seed: int,
    weight: FloatArray | None,
) -> tuple[FloatArray, PropensitySelection | None]:
    """``P(A = 1 | x)`` at the prediction rows from a model fitted on the training rows.

    Parameters
    ----------
    learner : classifier or None
        The treatment learner, ``None`` for the default selection.
    X : ndarray
        ``(V, p)`` designs.
    A : ndarray
        ``(V,)`` coded treatment.
    train : ndarray
        ``(V,)`` training rows.
    predict : ndarray
        ``(V,)`` prediction rows.
    folds : ndarray
        Inner labels of the training rows.
    seed : int
        The fit's seed.
    weight : ndarray or None
        ``(V,)`` the private row weights.

    Returns
    -------
    tuple
        The probabilities and the selection record.
    """
    if not np.any(predict):
        return np.zeros(0), None
    At = A[train]
    if np.all(At == At[0]) if At.size else True:
        # One level among the training visits: its probability is one there.
        level = float(At[0]) if At.size else 0.5
        return np.full(int(np.count_nonzero(predict)), level), None
    if weight is not None:
        from sklearn.base import clone

        model = clone(seed_unset(learner, seed)).fit(X[train], At, sample_weight=weight[train])
        return model.predict_proba(X[predict])[:, 1], None
    return _fit_propensity(learner, X, A, train, predict, folds, seed)


class _SoftLogistic:
    """A logistic model of a fraction in ``[0, 1]``, fitted on the soft labels.

    Each row enters twice, as an event with weight ``w * y`` and as a non-event with weight
    ``w * (1 - y)``, so the fit maximizes the soft log-likelihood.
    """

    def __init__(self, C: float, random_state: int | None) -> None:
        self.C = C
        self.random_state = random_state

    def __repr__(self) -> str:
        return "logistic" if not np.isfinite(self.C) else f"lasso C={self.C:g}"

    def fit(self, X: FloatArray, y: FloatArray, sample_weight: FloatArray) -> _SoftLogistic:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if np.isfinite(self.C):
            model = LogisticRegression(
                C=self.C,
                l1_ratio=1.0,
                solver="saga",
                max_iter=5000,
                random_state=self.random_state,
            )
        else:
            model = LogisticRegression(C=np.inf, max_iter=2000)
        self.pipeline_ = Pipeline([("scale", StandardScaler()), ("model", model)])
        stacked = np.vstack([X, X])
        labels = np.r_[np.ones(len(y)), np.zeros(len(y))]
        weights = np.r_[sample_weight * y, sample_weight * (1.0 - y)]
        keep = weights > 0
        self.pipeline_.fit(stacked[keep], labels[keep], model__sample_weight=weights[keep])
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.pipeline_.predict_proba(X)[:, 1], dtype=float)


#: The inverse L1 penalty of the lasso member of the default continuation library.
CONTINUATION_LASSO_C: Final = 0.1


def _soft_loss(p: FloatArray, y: FloatArray, w: FloatArray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -float(np.sum(w * (y * np.log(p) + (1.0 - y) * np.log1p(-p))))


class _DefaultContinuation:
    """The default continuation learner: a main-terms and an L1 soft logistic model, chosen
    by cross-validated soft log loss over the inner folds."""

    def __init__(self, folds: IntArray, random_state: int | None) -> None:
        self.folds = folds
        self.random_state = random_state

    def fit(self, X: FloatArray, y: FloatArray, sample_weight: FloatArray) -> _DefaultContinuation:
        members = [
            _SoftLogistic(np.inf, self.random_state),
            _SoftLogistic(CONTINUATION_LASSO_C, self.random_state),
        ]
        losses = []
        labels = np.unique(self.folds)
        for member in members:
            if labels.size < 2:
                losses.append(0.0)
                continue
            loss = 0.0
            for fold in labels:
                held = self.folds == fold
                fitted = _SoftLogistic(member.C, self.random_state).fit(
                    X[~held], y[~held], sample_weight[~held]
                )
                loss += _soft_loss(fitted.predict(X[held]), y[held], sample_weight[held])
            losses.append(loss)
        self.chosen_ = int(np.argmin(losses))
        self.names_ = tuple(repr(m) for m in members)
        self.losses_ = tuple(losses)
        self.model_ = members[self.chosen_].fit(X, y, sample_weight)
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        return self.model_.predict(X)


@dataclass
class ContinuationModel:
    r"""The binned law :math:`\nu_k(b, l \mid u, x_k)` of the event time after a visit at
    :math:`u`, held by its bin hazards.

    The bins are cut at the target times: bin :math:`b` is :math:`(\tau_{b-1}, \tau_b]`, and a
    visit at :math:`u` starts its first bin at :math:`u`, so the bins ending at or before
    :math:`u` hold nothing.  Cause :math:`j` regresses the fraction
    :math:`p(b, j) / (\rho(b) - \sum_{l < j} p(b, l))` with weight
    :math:`\rho(b) - \sum_{l < j} p(b, l)` on the window, the visit time and one indicator per
    bin, and :math:`\lambda_j = \hat f_j (1 - \sum_{l < j} \lambda_l)`, as X70's
    ``fit_continuation`` builds its continuation; the soft-label hazard loss is linear in
    :math:`p`, so its minimizer is the hazard of the mean sub-distribution.

    Parameters
    ----------
    models : list
        Per cause, a fitted regressor, or a float for a constant fraction.
    bins : ndarray
        ``(B,)`` the bin ends.
    """

    models: list[Any]
    bins: FloatArray

    @staticmethod
    def stack(
        x: FloatArray, u: FloatArray, bins: FloatArray
    ) -> tuple[FloatArray, IntArray, IntArray]:
        """Rows ``(i, b)`` with ``bins[b] > u_i``, and their features ``[x_i, u_i, e_b]``."""
        B = bins.size
        live = bins[None, :] > u[:, None]
        rows, cols = np.nonzero(live)
        onehot = np.zeros((rows.size, B))
        onehot[np.arange(rows.size), cols] = 1.0
        return np.column_stack([x[rows], u[rows], onehot]), rows, cols

    @classmethod
    def fit(
        cls,
        learner: Any,
        x: FloatArray,
        u: FloatArray,
        labels: FloatArray,
        weight: FloatArray,
        bins: FloatArray,
        folds: IntArray,
        seed: int,
    ) -> ContinuationModel:
        """Fit on the visits of the followers, with their soft labels.

        Parameters
        ----------
        learner : regressor or None
            ``fit(X, y, sample_weight)`` and ``predict(X)``; ``None`` for the default.
        x : ndarray
            ``(c, p)`` the window at the earlier visit (the regime's slot).
        u : ndarray
            ``(c,)`` the observed visit time.
        labels : ndarray
            ``(c, B, J)`` the soft sub-distribution of the event time after the visit.
        weight : ndarray
            ``(c,)`` row weights (one, or the private seam's).
        bins : ndarray
            ``(B,)`` the bin ends.
        folds : ndarray
            ``(c,)`` inner fold labels, by unit.
        seed : int
            The fit's seed.

        Returns
        -------
        ContinuationModel
            The fitted model.
        """
        from sklearn.base import clone

        features, rows, cols = cls.stack(x, u, bins)
        J = labels.shape[2]
        p = labels[rows, cols]  # (S, J)
        spent = np.cumsum(labels.sum(axis=2), axis=1)
        before = np.concatenate([np.zeros((labels.shape[0], 1)), spent[:, :-1]], axis=1)
        rho = np.clip(1.0 - before, 0.0, None)[rows, cols]
        earlier = np.zeros(rows.size)
        models: list[Any] = []
        for j in range(J):
            remaining = rho - earlier
            positive = remaining > 1e-15
            target = np.where(positive, p[:, j] / np.where(positive, remaining, 1.0), 0.0)
            target = np.clip(target, 0.0, 1.0)
            w = np.where(positive, weight[rows] * remaining, 0.0)
            fitted = w > 0
            if not np.any(target[fitted] > 0):
                models.append(0.0)
            elif np.all(target[fitted] >= 1.0):
                models.append(1.0)
            else:
                if learner is None:
                    model = _DefaultContinuation(folds[rows][fitted], seed)
                else:
                    model = clone(seed_unset(learner, seed))
                model.fit(features[fitted], target[fitted], sample_weight=w[fitted])
                models.append(model)
            earlier = earlier + p[:, j]
        return cls(models, np.asarray(bins, dtype=float))

    def hazards(self, x: FloatArray, u: FloatArray) -> FloatArray:
        """``(J, c, B)`` the bin hazards after a visit at ``u`` with window ``x``.

        Parameters
        ----------
        x : ndarray
            ``(c, p)`` windows.
        u : ndarray
            ``(c,)`` visit times.

        Returns
        -------
        ndarray
            Zero in the bins ending at or before ``u``.
        """
        features, rows, cols = self.stack(x, u, self.bins)
        J, c, B = len(self.models), x.shape[0], self.bins.size
        out = np.zeros((J, c, B))
        taken = np.zeros(rows.size)
        for j, model in enumerate(self.models):
            if isinstance(model, float):
                fraction = np.full(rows.size, model)
            else:
                fraction = np.clip(np.asarray(model.predict(features), dtype=float), 0.0, 1.0)
            value = fraction * (1.0 - taken)
            out[j, rows, cols] = value
            taken = taken + value
        return out


def check_levels(levels: np.ndarray) -> tuple[list[Any], str | None]:
    """The sorted treatment levels, and the kind of treatment C7 or C14 refuses, if any.

    Parameters
    ----------
    levels : ndarray
        The treatment values at every visit.

    Returns
    -------
    tuple
        The levels and ``None``, ``"continuous"`` or ``"levels"``.
    """
    observed = list(dict.fromkeys(sorted(np.unique(levels).tolist())))
    try:
        numeric = np.asarray(observed, dtype=float)
    except (TypeError, ValueError):
        numeric = None
    if len(observed) > 2:
        if numeric is not None and np.any(numeric % 1):
            return observed, "continuous"
        return observed, "levels"
    if len(observed) < 2:
        raise DataError(
            "the treatment column holds one level; ContinuousLongitudinalTMLE contrasts "
            "regimes of a binary treatment"
        )
    return observed, None
