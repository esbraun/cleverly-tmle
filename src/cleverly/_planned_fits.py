"""The learner fits an ``estimate`` call would make, counted before any learner runs.

:meth:`cleverly.IdentifiedEffect.planned_fits` returns a :class:`PlannedFits`: one
:class:`PlannedFitRow` per nuisance role and scope, with the attempted top-level fits and
the candidate fits inside them.  Nothing here fits, clones into a fit, or predicts.

Two units are counted.

- A **fit** is one attempted call of ``fit`` on a role learner or its clone, made by the
  estimator.
- A **candidate fit** is one attempted ``fit`` of a base estimator.  A plain learner's fit
  is one candidate fit.  A :class:`~cleverly.learners.SuperLearner` fit is
  ``C * (V + 1)``: ``C`` library members, each fitted on ``V`` inner training folds and once
  on every row, with ``V`` the inner fold count :func:`~cleverly.learners.make_folds`
  resolves on that fit's training rows (capped by the rarer class of a classification
  target and by the cluster count).  A classification target with more than two classes is
  fitted one-vs-rest, which is the sum over the classes of ``C * (V_k + 1)``, each member
  capping its own ``V_k``.  A pipeline counts its final step, a Super Learner inside a
  library recurses, and any other meta-estimator (a grid search, a stacking or bagging
  ensemble) counts as one candidate fit with a note.

A candidate that cross-validates internally reports its fold count as ``internal_cv`` and
is not multiplied by it: a regularization path or a grid is a different unit, and its size
floats with library versions.

The engine side lives beside each engine's ``fit``, as its ``_planned_fits``, so a change to
fitting code and its count sit in one diff.  This module holds what every engine shares: the
record, the counting of one learner object, and the counting of one cross-fitted role.
"""

from __future__ import annotations

import inspect
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._summary import Summary
    from ._typing import BoolArray, FloatArray, IntArray, Learner
    from .learners._fitting import Task
    from .learners.crossfit import Folds

__all__ = ["PlannedFitRow", "PlannedFits"]

Bound = Literal["exact", "upper"]
FoldState = Literal["realized", "outer realized", "one draw"]

#: scikit-learn's default fold count of an estimator whose ``cv`` is ``None``.
SKLEARN_DEFAULT_CV = 5

#: The note a meta-estimator other than a Super Learner adds, by its class name.
META_NOTE = (
    "{name} is counted as one candidate fit per fit; the fits it makes inside "
    "(a search grid, the members of an ensemble) are not counted."
)


@dataclass(frozen=True)
class PlannedFitRow:
    """The fits of one nuisance role in one scope.

    Parameters
    ----------
    role : str
        The method field the role reads, such as ``"outcome_learner"``, or an engine role
        that no field names, such as ``"tuner"``.
    scope : str
        What the fit is and how often it repeats, in plain words, such as
        ``"outcome regression, per fold and repeat"``.
    learner : str
        The role's learner, named as a result summary names it.
    fits : int
        The attempted top-level fits.  For an upper row, the most the fit can make.
    candidate_fits : int
        The attempted fits of base estimators inside those fits.
    bound : {"exact", "upper"}
        ``"exact"`` when the counts are the ones the fit makes, assuming no fit raises;
        ``"upper"`` when they are the most it can make.
    why : str or None
        For an upper row, what can make the realized count smaller.
    min_fits : int or None
        For an upper row, the fewest top-level fits the fit makes when no replicate is
        dropped.  ``None`` for an exact row.
    by_candidate : tuple of tuple
        ``(name, candidate_fits)`` per base estimator, in first-seen order.
    """

    role: str
    scope: str
    learner: str
    fits: int
    candidate_fits: int
    bound: Bound = "exact"
    why: str | None = None
    min_fits: int | None = None
    by_candidate: tuple[tuple[str, int], ...] = ()

    def __repr__(self) -> str:
        from ._summary import short_repr

        return short_repr(
            "PlannedFitRow",
            (
                ("", self.role),
                ("fits", self.fits if self.bound == "exact" else f"at most {self.fits}"),
                ("candidate fits", self.candidate_fits),
            ),
        )


@dataclass(frozen=True)
class PlannedFits:
    """The learner fits an ``estimate`` call would make, counted before any learner runs.

    :meth:`cleverly.IdentifiedEffect.planned_fits` returns one.  It holds names and counts,
    no learner objects.

    Parameters
    ----------
    rows : tuple of PlannedFitRow
        One row per nuisance role and scope.
    candidates : tuple of tuple
        ``(name, candidate_fits, internal_cv)`` per base estimator, across rows.
        ``internal_cv`` is the fold count of a candidate that cross-validates internally,
        read from its final step with ``cv=None`` read as scikit-learn's 5, else ``None``.
    multipliers : tuple of tuple
        ``(name, value)`` per factor the counts multiply by, such as outer folds, repeats
        and bootstrap replicates.
    searches : tuple of tuple
        ``(name, maximum, stops_early, learner_fits)`` per candidate search a method runs.
        Empty for a method that searches nothing.
    folds : {"realized", "outer realized", "one draw"}
        ``"realized"`` when an integer ``random_state`` fixes every draw the counts read,
        so they are the folds ``estimate`` uses.  ``"outer realized"`` when a split plan
        fixes the outer folds and the seeds below them are drawn fresh.  ``"one draw"``
        when the folds are drawn fresh on every fit, so fold-dependent counts describe one
        draw.
    unpreviewed : tuple of str
        The checks ``estimate`` still runs after the point where the count stops, by
        function name.  Empty for most calls.
    notes : tuple of str
        Plain sentences about what the counts include and leave out.

    Attributes
    ----------
    total_fits : int
    total_candidate_fits : int
    exact : bool

    See Also
    --------
    cleverly.IdentifiedEffect.planned_fits : What returns this record.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> frame, _ = make_linear_ate(n=200, seed=1)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(
    ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
    ...     ),
    ... )
    >>> plan = study.planned_fits(
    ...     ATE(),
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(),
    ...     n_folds=5,
    ...     random_state=0,
    ...     q_bounds=(-20.0, 20.0),
    ... )
    >>> plan.total_fits, plan.exact
    (10, True)
    """

    rows: tuple[PlannedFitRow, ...]
    candidates: tuple[tuple[str, int, int | None], ...] = ()
    multipliers: tuple[tuple[str, int], ...] = ()
    searches: tuple[tuple[str, int, bool, int], ...] = ()
    folds: FoldState = "realized"
    unpreviewed: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def total_fits(self) -> int:
        """The attempted top-level fits over every row, an upper bound when not :attr:`exact`."""
        return sum(row.fits for row in self.rows)

    @property
    def total_candidate_fits(self) -> int:
        """The attempted base-estimator fits over every row."""
        return sum(row.candidate_fits for row in self.rows)

    @property
    def exact(self) -> bool:
        """Whether every row is exact, so the totals are the fits ``estimate`` makes."""
        return all(row.bound == "exact" for row in self.rows)

    @property
    def min_fits(self) -> int:
        """The fewest top-level fits: an exact row's count plus an upper row's minimum."""
        return sum(row.fits if row.min_fits is None else row.min_fits for row in self.rows)

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Summary:
        """Return the printed table of the planned fits.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            ``"full"`` adds the multipliers, the notes and any check left to the fit.

        Returns
        -------
        Summary
            A ``str`` whose text is the printed table; a notebook renders it as HTML.
        """
        from ._summary._setup import planned_fits_summary

        return planned_fits_summary(self, detail)

    def to_frame(self, backend: str | None = None) -> Any:
        """Return one row per nuisance role and scope.

        Parameters
        ----------
        backend : {"pandas", "polars", "pyarrow"} or None, default=None
            Dataframe backend to return. ``None`` uses pandas when installed, then the
            first available backend.

        Returns
        -------
        dataframe
            Columns ``role``, ``scope``, ``learner``, ``fits``, ``candidate_fits``,
            ``bound``, ``min_fits`` and ``why``.
        """
        from .utils.frames import frame_from_dict

        return frame_from_dict(
            {
                "role": [row.role for row in self.rows],
                "scope": [row.scope for row in self.rows],
                "learner": [row.learner for row in self.rows],
                "fits": [row.fits for row in self.rows],
                "candidate_fits": [row.candidate_fits for row in self.rows],
                "bound": [row.bound for row in self.rows],
                "min_fits": [row.min_fits for row in self.rows],
                "why": [row.why for row in self.rows],
            },
            backend=backend,
        )

    def __repr__(self) -> str:
        from ._summary import short_repr

        fits = self.total_fits
        return short_repr(
            "PlannedFits",
            (
                ("fits", fits if self.exact else f"at most {fits}"),
                ("candidate fits", self.total_candidate_fits),
                ("", "exact" if self.exact else "upper bound"),
            ),
        )

    def _repr_html_(self) -> str:
        return self.summary().as_html()


# ----------------------------------------------------------------- counting one learner


@dataclass
class Count:
    """The fits and candidate fits of one role, accumulated as it is counted.

    Parameters
    ----------
    fits : int
        Attempted top-level fits.
    candidates : Counter
        Candidate fits per base-estimator name.
    internal_cv : dict
        The internal fold count of each candidate name, or ``None``.
    notes : set of str
        Sentences the record carries.
    upper : list of str
        Why a count is an upper bound; empty for an exact count.
    """

    fits: int = 0
    candidates: Counter[str] = field(default_factory=Counter)
    internal_cv: dict[str, int | None] = field(default_factory=dict)
    notes: set[str] = field(default_factory=set)
    upper: list[str] = field(default_factory=list)

    def add(self, other: Count, times: int = 1) -> None:
        """Add ``times`` copies of ``other``'s counts to this one.

        Parameters
        ----------
        other : Count
            The counts to add.
        times : int
            How many copies.
        """
        self.fits += other.fits * times
        for name, value in other.candidates.items():
            self.candidates[name] += value * times
        for name, folds in other.internal_cv.items():
            self.internal_cv.setdefault(name, folds)
        self.notes |= other.notes
        for reason in other.upper:
            if reason not in self.upper:
                self.upper.append(reason)

    def scaled(self, times: int) -> Count:
        """A copy with every count multiplied by ``times``.

        Parameters
        ----------
        times : int
            The factor.

        Returns
        -------
        Count
            The copy.
        """
        out = Count()
        out.add(self, times)
        return out

    @property
    def candidate_fits(self) -> int:
        """Every candidate fit."""
        return int(sum(self.candidates.values()))


def _final_step(learner: Learner) -> Learner:
    """The estimator that a pipeline's fit ends in, through nested pipelines."""
    from sklearn.pipeline import Pipeline

    while isinstance(learner, Pipeline):
        learner = learner.steps[-1][1]
    return learner


def internal_cv(learner: Learner) -> int | None:
    """The internal fold count of a candidate's final step, or ``None`` if it has none.

    Parameters
    ----------
    learner : Learner
        A candidate.

    Returns
    -------
    int or None
        ``cv`` as an integer, ``None`` read as scikit-learn's 5, a splitter's ``n_splits``,
        or ``None`` for an estimator with no ``cv`` parameter or an iterable of splits.
    """
    final = _final_step(learner)
    try:
        params = final.get_params(deep=False)
    except (AttributeError, TypeError):  # pragma: no cover - not an estimator
        return None
    if "cv" not in params:
        return None
    cv = params["cv"]
    if cv is None:
        return SKLEARN_DEFAULT_CV
    if isinstance(cv, (int, np.integer)):
        return int(cv)
    splits = getattr(cv, "n_splits", None)
    return int(splits) if isinstance(splits, (int, np.integer)) else None


def _is_meta(learner: Learner) -> bool:
    """Whether a candidate wraps other estimators that it fits inside its own fit."""
    from sklearn.base import MetaEstimatorMixin

    return isinstance(learner, MetaEstimatorMixin) or any(
        hasattr(learner, name) for name in ("estimator", "estimators", "base_estimator")
    )


def needs_target(learner: Learner) -> bool:
    """Whether counting ``learner`` reads its training target, which only a Super Learner does.

    Parameters
    ----------
    learner : Learner
        A role learner.

    Returns
    -------
    bool
        ``True`` when the final step is a :class:`~cleverly.learners.SuperLearner`.
    """
    from .learners.super_learner import SuperLearner

    return isinstance(_final_step(learner), SuperLearner)


def inner_fold_count(
    n_folds: int, n: int, stratify: FloatArray | None, cluster: IntArray | None
) -> int:
    """The fold count :func:`~cleverly.learners.make_folds` resolves, without its warnings.

    Parameters
    ----------
    n_folds : int
        Folds requested.
    n : int
        Rows to split.
    stratify : ndarray or None
        The labels a stratified split balances.
    cluster : ndarray or None
        Cluster codes.

    Returns
    -------
    int
        ``min(n_folds, n or the rarer class count, the cluster count)``, at least 2.  A
        smaller cap makes the split raise, which the record notes rather than counts.
    """
    cap = n
    if stratify is not None:
        cap = int(np.unique(np.asarray(stratify), return_counts=True)[1].min())
    resolved = int(min(n_folds, cap))
    if cluster is not None:
        resolved = min(resolved, int(np.unique(cluster).size))
    return max(resolved, 2)


def count_learner(
    learner: Learner,
    target: FloatArray | None,
    *,
    groups: IntArray | None = None,
    seed_fixed: bool = True,
    nominal: bool = False,
    attempts: int = 1,
) -> Count:
    """The candidate fits of one fit of ``learner`` on a training target.

    Parameters
    ----------
    learner : Learner
        The learner as the estimator fits it: resolved and seeded.
    target : ndarray or None
        The training target, as the fit passes it.  Read only by a Super Learner; ``None``
        is allowed for any other learner.
    groups : ndarray or None
        The cluster codes the fit routes to the learner, or ``None``.
    seed_fixed : bool
        Whether the seed a nested Super Learner draws its inner folds from is the one the
        fit will use.  Without it a nested ensemble is counted at its nominal fold count
        and the count is an upper bound.
    nominal : bool
        Count every Super Learner at its declared ``n_folds``, with no cap and no
        realized draw: the upper bound of a fit on data not yet drawn, such as a bootstrap
        replicate.
    attempts : int
        Attempted fits of this one learner, 2 for a fit retried without weights.

    Returns
    -------
    Count
        One fit and its candidate fits, times ``attempts``.
    """
    count = Count(fits=1)
    y = None if target is None else np.asarray(target, dtype=float).reshape(-1)
    _count_into(count, learner, y, groups, None, seed_fixed, nominal, name=None, seed=None)
    return count.scaled(attempts) if attempts != 1 else count


def _count_into(
    count: Count,
    learner: Learner,
    y: FloatArray | None,
    groups: IntArray | None,
    prefix: str | None,
    seed_fixed: bool,
    nominal: bool,
    *,
    name: str | None,
    seed: Any,
) -> None:
    from .learners.super_learner import SuperLearner

    final = _final_step(learner)
    if isinstance(final, SuperLearner):
        _count_super_learner(count, final, y, groups, prefix, seed_fixed, nominal, seed=seed)
        return
    label = name or type(final).__name__
    if prefix:
        label = f"{prefix}/{label}"
    count.candidates[label] += 1
    count.internal_cv.setdefault(label, internal_cv(final))
    if _is_meta(final):
        count.notes.add(META_NOTE.format(name=type(final).__name__))


def _count_super_learner(
    count: Count,
    ensemble: Any,
    y: FloatArray | None,
    groups: IntArray | None,
    prefix: str | None,
    seed_fixed: bool,
    nominal: bool,
    *,
    seed: Any,
) -> None:
    """One :meth:`SuperLearner.fit`, as ``super_learner.py`` runs it."""
    from .learners._fitting import infer_task
    from .learners.crossfit import make_folds
    from .learners.library import _resolve_library

    if y is None:  # pragma: no cover - callers pass the target of a Super Learner
        raise ValueError("a Super Learner's count reads its training target")
    own_seed = ensemble.random_state if ensemble.random_state is not None else seed
    task: Task = ensemble.task or infer_task(y)
    levels = np.unique(y)
    if task == "classification" and levels.size > 2:
        # ``_fit_one_vs_rest``: one binary ensemble per class, each capping its own folds.
        for level in levels:
            _count_super_learner(
                count,
                _with_task(ensemble, "classification"),
                (y == level).astype(float),
                groups,
                prefix,
                seed_fixed,
                nominal,
                seed=own_seed,
            )
        return
    library = _resolve_library(ensemble.library, task, random_state=ensemble.random_state)
    stratify = y if task == "classification" else None
    n_inner = (
        int(ensemble.n_folds)
        if nominal
        else inner_fold_count(int(ensemble.n_folds), y.size, stratify, groups)
    )
    nested = [entry for entry in library if needs_target(entry[1])]
    folds: Folds | None = None
    nested_nominal = nominal
    if nested and not nominal:
        if seed_fixed and isinstance(own_seed, (int, np.integer)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                folds = make_folds(
                    y.size,
                    int(ensemble.n_folds),
                    stratify=stratify,
                    cluster=groups,
                    random_state=int(own_seed),
                )
        else:
            nested_nominal = True
            count.upper.append(
                "a Super Learner inside a library draws its inner folds from a seed that is "
                "drawn afresh on every fit, so it is counted at its declared fold count"
            )
    for entry_name, estimator in library:
        if not needs_target(estimator):
            label = entry_name if not prefix else f"{prefix}/{entry_name}"
            count.candidates[label] += n_inner + 1
            count.internal_cv.setdefault(label, internal_cv(estimator))
            if _is_meta(_final_step(estimator)):
                count.notes.add(META_NOTE.format(name=type(_final_step(estimator)).__name__))
            continue
        inner_prefix = entry_name if not prefix else f"{prefix}/{entry_name}"
        if folds is not None:
            for train, _ in folds:
                _count_into(
                    count,
                    estimator,
                    y[train],
                    None,
                    inner_prefix,
                    seed_fixed,
                    False,
                    name=None,
                    seed=own_seed,
                )
        else:
            for _ in range(n_inner):
                _count_into(
                    count,
                    estimator,
                    y,
                    None,
                    inner_prefix,
                    seed_fixed,
                    nested_nominal,
                    name=None,
                    seed=own_seed,
                )
        # The refit on every row of the ensemble's training set.
        _count_into(
            count,
            estimator,
            y,
            None,
            inner_prefix,
            seed_fixed,
            nested_nominal,
            name=None,
            seed=own_seed,
        )


def _with_task(ensemble: Any, task: Task) -> Any:
    """A one-vs-rest member: the ensemble's settings with the task fixed."""
    from .learners.super_learner import SuperLearner

    return SuperLearner(
        library=ensemble.library,
        task=task,
        meta_learner=ensemble.meta_learner,
        n_folds=ensemble.n_folds,
        clip=ensemble.clip,
        random_state=ensemble.random_state,
        n_jobs=ensemble.n_jobs,
    )


# ----------------------------------------------------------------- counting one role


def _routed_groups(learner: Learner, groups: IntArray | None) -> IntArray | None:
    """The cluster codes :func:`~cleverly.learners.fit_learner` hands ``learner``."""
    from .learners._fitting import accepts_groups

    if groups is None or not accepts_groups(learner):
        return None
    return np.asarray(groups)


def _one_value(target: FloatArray, weights: FloatArray, rows: IntArray) -> bool:
    """Whether ``rows`` with positive weight hold one target value (``cross_fit_companion``)."""
    informative = rows[np.asarray(weights, dtype=float)[rows] > 0.0]
    return bool(np.unique(np.asarray(target, dtype=float)[informative]).size == 1)


def count_cross_fit(
    learner: Learner,
    target: FloatArray,
    folds: Folds,
    *,
    task: Task,
    weights: FloatArray,
    fit_mask: BoolArray | None = None,
    groups: IntArray | None = None,
    constant_skip: bool = False,
    seed_fixed: bool = True,
) -> Count:
    """The fits of one nuisance cross-fitted as ``cross_fit_companion`` fits it.

    Parameters
    ----------
    learner : Learner
        The resolved learner.
    target : ndarray
        ``(n,)`` the target before :func:`~cleverly.learners._fitting.as_target`.
    folds : Folds
        The outer split.
    task : {"regression", "classification"}
        The nuisance task.
    weights : ndarray
        ``(n,)`` the fit weights, which the constant-fold rule reads.
    fit_mask : ndarray or None
        The rows eligible for training.
    groups : ndarray or None
        Cluster codes.
    constant_skip : bool
        Whether a training fold whose positive-weight rows hold one target value takes a
        ``ConstantFit`` and fits no learner, as ``constant_folds`` does.
    seed_fixed : bool
        As :func:`count_learner`.

    Returns
    -------
    Count
        The fits over the folds.
    """
    from .learners._fitting import as_target

    n = np.asarray(target).shape[0]
    mask = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    routed = _routed_groups(learner, groups)
    reads = needs_target(learner)
    out = Count()
    trains = (
        [np.flatnonzero(mask)] if folds.is_single else [train[mask[train]] for train, _ in folds]
    )
    for rows in trains:
        if constant_skip and _one_value(target, weights, rows):
            continue
        y = as_target(np.asarray(target)[rows], task) if reads else None
        out.add(
            count_learner(
                learner,
                y,
                groups=None if routed is None else routed[rows],
                seed_fixed=seed_fixed,
            )
        )
    return out


def count_nominal(learner: Learner, target: FloatArray, n_fits: int, *, task: Task) -> Count:
    """``n_fits`` fits of ``learner`` at its nominal counts, with no constant-fold skip.

    A bootstrap replicate's resample and folds are not drawn ahead, so each fit's inner
    folds are counted at the Super Learner's declared ``n_folds`` and a one-vs-rest target
    at the classes of ``target``.

    Parameters
    ----------
    learner : Learner
        The resolved learner.
    target : ndarray
        The target on the original rows, which fixes the task and the classes.
    n_fits : int
        The fits to count.
    task : {"regression", "classification"}
        The nuisance task.

    Returns
    -------
    Count
        The nominal counts.
    """
    from .learners._fitting import as_target

    if n_fits <= 0:
        return Count()
    y = as_target(np.asarray(target), task) if needs_target(learner) else None
    return count_learner(learner, y, nominal=True).scaled(n_fits)


@dataclass
class PlanItem:
    """One row's worth of fits that a variant's search adds, before its scope is set.

    Parameters
    ----------
    role : str
        The row's role.
    what : str
        What the fits are, in plain words; the caller adds how often.
    learner : str
        The learner's printed name.
    count : Count
        The fits.
    upper : str or None
        Why the count is an upper bound, or ``None`` when it is exact.
    min_fits : int or None
        The fewest fits an upper count can realize.
    """

    role: str
    what: str
    learner: str
    count: Count
    upper: str | None = None
    min_fits: int | None = None


class FitCounter:
    """Counts the fits of a construction on its own rows, realized or at nominal counts.

    The searches of the collaborative, doubly robust and adaptive engines fit learners on
    rows they choose themselves.  One counter serves the original draw, where each fit
    reads its training rows, and a bootstrap replicate, where each fit is counted at the
    learner's nominal counts because the resample is not drawn ahead.

    Parameters
    ----------
    weights : ndarray
        ``(n,)`` the fit weights, which the constant-fold rule reads.
    groups : ndarray or None
        Cluster codes.
    seed_fixed : bool
        As :func:`count_learner`.
    nominal : bool
        Count every fit at the learner's nominal counts, with no constant-fold skip.
    """

    def __init__(
        self,
        *,
        weights: FloatArray,
        groups: IntArray | None,
        seed_fixed: bool = True,
        nominal: bool = False,
    ) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.groups = groups
        self.seed_fixed = seed_fixed
        self.nominal = nominal

    def one(
        self,
        learner: Learner,
        target: FloatArray | None,
        rows: IntArray,
        *,
        task: Task,
        route_groups: bool = True,
    ) -> Count:
        """One fit of ``learner`` on ``rows``.

        Parameters
        ----------
        learner : Learner
            The resolved learner.
        target : ndarray or None
            ``(n,)`` the target on every row; ``None`` when the learner reads none.
        rows : ndarray of int
            The training rows.
        task : {"regression", "classification"}
            The fit's task.
        route_groups : bool
            Whether the fit hands the cluster codes to the learner.

        Returns
        -------
        Count
            The fit and its candidate fits.
        """
        from .learners._fitting import as_target

        reads = needs_target(learner) and target is not None
        if self.nominal:
            y = as_target(np.asarray(target), task) if reads else None
            return count_learner(learner, y, nominal=True)
        index = np.asarray(rows)
        y = as_target(np.asarray(target)[index], task) if reads else None
        routed = _routed_groups(learner, self.groups) if route_groups else None
        return count_learner(
            learner,
            y,
            groups=None if routed is None else routed[index],
            seed_fixed=self.seed_fixed,
        )

    def cross(
        self,
        learner: Learner,
        target: FloatArray,
        folds: Folds,
        *,
        task: Task,
        fit_mask: BoolArray | None = None,
        constant_skip: bool = False,
    ) -> Count:
        """The fits of one nuisance cross-fitted on ``folds``, as :func:`count_cross_fit`.

        Parameters
        ----------
        learner : Learner
            The resolved learner.
        target : ndarray
            ``(n,)`` the target.
        folds : Folds
            The split.
        task : {"regression", "classification"}
            The fit's task.
        fit_mask : ndarray or None
            The rows eligible for training.
        constant_skip : bool
            As :func:`count_cross_fit`.

        Returns
        -------
        Count
            The fits over the folds.
        """
        if self.nominal:
            fits = 1 if folds.is_single else folds.n_folds
            return count_nominal(learner, target, fits, task=task)
        return count_cross_fit(
            learner,
            target,
            folds,
            task=task,
            weights=self.weights,
            fit_mask=fit_mask,
            groups=self.groups,
            constant_skip=constant_skip,
            seed_fixed=self.seed_fixed,
        )


# ----------------------------------------------------------------- building the record


@dataclass
class _Entry:
    role: str
    scope: str
    learner: str
    count: Count
    upper: list[str] = field(default_factory=list)
    min_fits: int | None = None


class PlanBuilder:
    """Collects the rows of one preview and builds the :class:`PlannedFits` record."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._multipliers: dict[str, int] = {}
        self._notes: list[str] = []
        self._unpreviewed: list[str] = []
        self._searches: dict[str, tuple[int, bool, int]] = {}

    def add(
        self,
        role: str,
        scope: str,
        learner: str,
        count: Count,
        *,
        upper: str | None = None,
        min_fits: int | None = None,
    ) -> None:
        """Add ``count`` to the row of ``(role, scope)``.

        Parameters
        ----------
        role : str
            The row's role.
        scope : str
            The row's scope.
        learner : str
            The learner's printed name.
        count : Count
            The counts to add.  A count with no fit adds nothing.
        upper : str or None
            Why the added counts are an upper bound, or ``None`` when they are exact.
        min_fits : int or None
            The fewest fits the added counts can realize, for an upper bound.
        """
        if count.fits == 0 and not count.candidates:
            return
        entry = self._entries.setdefault((role, scope), _Entry(role, scope, learner, Count()))
        entry.count.add(count)
        reasons = [*count.upper, *([] if upper is None else [upper])]
        for reason in reasons:
            if reason not in entry.upper:
                entry.upper.append(reason)
        entry.min_fits = (entry.min_fits or 0) + (count.fits if min_fits is None else min_fits)

    def multiply(self, name: str, value: int) -> None:
        """Record a multiplier, keeping the first value under each name.

        Parameters
        ----------
        name : str
            The factor, in plain words.
        value : int
            Its value.
        """
        self._multipliers.setdefault(name, int(value))

    def note(self, text: str) -> None:
        """Add a note once.

        Parameters
        ----------
        text : str
            The sentence.
        """
        if text not in self._notes:
            self._notes.append(text)

    def search(self, name: str, maximum: int, stops_early: bool, learner_fits: int) -> None:
        """Record a candidate search, adding to an earlier entry under the same name.

        Parameters
        ----------
        name : str
            The search, in plain words.
        maximum : int
            The most steps it can take, summed over every run.
        stops_early : bool
            Whether it can stop before its maximum.
        learner_fits : int
            The learner fits its steps make, already counted in the rows.
        """
        steps, early, fits = self._searches.get(name, (0, False, 0))
        self._searches[name] = (steps + int(maximum), early or stops_early, fits + learner_fits)

    def absorb(
        self,
        plan: PlannedFits,
        scope: Callable[[str], str],
        *,
        upper: str | None = None,
        continuous: Callable[[PlannedFitRow], bool] | None = None,
    ) -> None:
        """Add every row of another preview, under a new scope.

        Parameters
        ----------
        plan : PlannedFits
            A preview of a fit that runs inside this one.
        scope : callable
            Maps a row's scope to the scope it takes here.
        upper : str or None
            Why the rows are an upper bound here though exact there, or ``None``.
        continuous : callable or None
            With ``upper``, whether a row keeps its fits as its minimum; a row it rejects
            has minimum 0.  ``None`` keeps every row's own minimum.
        """
        cv = {name: folds for name, _, folds in plan.candidates}
        for row in plan.rows:
            count = Count(fits=row.fits, candidates=Counter(dict(row.by_candidate)))
            count.internal_cv = {name: cv.get(name) for name in count.candidates}
            if row.why:
                count.upper.append(row.why)
            low = row.fits if row.min_fits is None else row.min_fits
            if upper is not None and continuous is not None and not continuous(row):
                low = 0
            self.add(row.role, scope(row.scope), row.learner, count, upper=upper, min_fits=low)
        for text in plan.notes:
            self.note(text)
        self.defer(*plan.unpreviewed)
        for name, maximum, early, fits in plan.searches:
            self.search(name, maximum, early, fits)

    def defer(self, *names: str) -> None:
        """Name checks ``estimate`` runs after the point where the preview stops.

        Parameters
        ----------
        *names : str
            The checks' function names.
        """
        for name in names:
            if name not in self._unpreviewed:
                self._unpreviewed.append(name)

    def build(self, folds: FoldState) -> PlannedFits:
        """The record.

        Parameters
        ----------
        folds : {"realized", "outer realized", "one draw"}
            What the fold-dependent counts describe.

        Returns
        -------
        PlannedFits
            The record.
        """
        rows: list[PlannedFitRow] = []
        totals: Counter[str] = Counter()
        cv: dict[str, int | None] = {}
        notes = list(self._notes)
        for entry in self._entries.values():
            count = entry.count
            for text in sorted(count.notes):
                if text not in notes:
                    notes.append(text)
            for name, value in count.candidates.items():
                totals[name] += value
            for name, inner in count.internal_cv.items():
                cv.setdefault(name, inner)
            upper = bool(entry.upper)
            rows.append(
                PlannedFitRow(
                    role=entry.role,
                    scope=entry.scope,
                    learner=entry.learner,
                    fits=count.fits,
                    candidate_fits=count.candidate_fits,
                    bound="upper" if upper else "exact",
                    why="; ".join(entry.upper) if upper else None,
                    min_fits=entry.min_fits if upper else None,
                    by_candidate=tuple(
                        (name, int(value)) for name, value in count.candidates.items()
                    ),
                )
            )
        return PlannedFits(
            rows=tuple(rows),
            candidates=tuple((name, int(value), cv.get(name)) for name, value in totals.items()),
            multipliers=tuple(self._multipliers.items()),
            searches=tuple(
                (name, steps, early, fits) for name, (steps, early, fits) in self._searches.items()
            ),
            folds=folds,
            unpreviewed=tuple(self._unpreviewed),
            notes=tuple(notes),
        )


def sample_weight_attempts(learner: Learner, weights: FloatArray | None) -> int:
    """Attempted fits of a tuner's boundary learner, which ``fit_tuner`` retries unweighted.

    ``fit_tuner`` calls ``fit`` with ``sample_weight`` and, on a ``TypeError``, calls it
    again without weights when the weights are absent or all one.  A learner other than a
    pipeline whose ``fit`` names neither ``sample_weight`` nor ``**kwargs`` raises that
    error, so its fit is two attempts.  A pipeline raises scikit-learn's ``ValueError``
    instead, which ``estimate`` raises.

    Parameters
    ----------
    learner : Learner
        The boundary learner.
    weights : ndarray or None
        The fit's observation weights.

    Returns
    -------
    int
        1 or 2.
    """
    from sklearn.pipeline import Pipeline

    if isinstance(learner, Pipeline):
        return 1
    try:
        signature = inspect.signature(learner.fit)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return 1
    if "sample_weight" in signature.parameters or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    ):
        return 1
    if weights is None:
        return 2
    w = np.asarray(weights, dtype=float)
    return 2 if np.allclose(w / w.mean(), 1.0) else 1


def refit_plan(result: Any) -> PlannedFits:
    """The fits of one ``refit()`` of a fitted result on its own data.

    The unit a refutation multiplies by its draw count.  It reads the result's recorded
    splits and seeds, so its folds are ``"realized"``.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.

    Returns
    -------
    PlannedFits
        The fits of one refit.

    Raises
    ------
    NotImplementedError
        Always, in this version.
    """
    # TODO(X106 phase 4): read the recorded splits and seeds through each engine's preview.
    raise NotImplementedError("TODO(X106 phase 4): refit_plan is not priced yet")
