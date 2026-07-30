"""Fold policy and fold construction for cross-fitting.

Nuisance models are fit out-of-fold so that no observation contributes to the
model used to predict it.  Two structural constraints matter and are enforced
here rather than left to the caller:

* **stratification** on the binary label keeps rare-treatment folds usable; a
  fold with no treated units cannot produce a propensity score;
* **cluster integrity** keeps every observation of a cluster in the same fold,
  otherwise cross-fitting leaks information between correlated rows and the
  out-of-fold predictions are optimistic.

The second is *checked* here rather than assumed, which is the difference between a split
that happens to be sound and one that is known to be.  What can be checked depends on
what the caller has in hand, so the checks come in two kinds:
:meth:`Folds.__post_init__` covers what an assignment alone can be wrong about, and
:func:`check_integrity` covers what needs the cluster vector beside it.  Two prohibitions
are on neither list, for opposite reasons.  No row can appear in two validation folds,
because an assignment holds one fold index per row and two-fold membership has no
representation -- worth stating, not worth checking.  And every stratum appearing in
every fold is not checkable as a *guarantee*: :func:`resolve_n_folds` caps the fold count
to make it achievable, but a cluster is atomic, so ``StratifiedGroupKFold`` cannot always
deliver it and an imbalanced split is still a usable one.

Three objects, and the distinction between the first two is the point of the module.
:class:`CrossFitPlan` is what a caller *declares*: a policy, made of numbers, that says
nothing about any particular dataset.  :class:`Folds` is what that policy *realises* on
one: an actual assignment of rows to folds, which depends on the row order and on the
scikit-learn version that made it as much as on the seed -- which is why
:mod:`cleverly.provenance` fingerprints the realisation separately from the seed, and why
a fit records the plan it declared beside the fold count it got.  :func:`make_folds` is
the map from one to the other, and calls :func:`check_integrity` on its way out, so every
split this library builds -- the outer cross-fitting folds, Super Learner's inner folds,
C-TMLE's selection folds -- is checked at construction without any of the three knowing
about it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from .._typing import FloatArray, IntArray
from ..exceptions import DataError

__all__ = [
    "CrossFitPlan",
    "Folds",
    "check_integrity",
    "make_folds",
    "refuse_scheme",
    "resolve_n_folds",
]


@dataclass(frozen=True)
class Folds:
    """A cross-fitting partition.

    ``assignment[i]`` is the index of the fold that holds out observation ``i``.
    Iterating yields ``(train_index, test_index)`` pairs like scikit-learn.
    """

    assignment: IntArray
    n_folds: int

    def __post_init__(self) -> None:
        """Check what an assignment on its own can be wrong about.

        Two things, and no more.  A fold index outside ``[0, n_folds)`` names a fold that
        does not exist, and an empty fold produces no out-of-fold predictions for anyone.
        Cluster integrity needs a vector this object does not hold, and lives in
        :func:`check_integrity`.

        Note what is *not* checked, on purpose.  "Every row is held out exactly once" has
        no check because it has no counterexample: an assignment holds one fold index per
        row, and two-fold membership is unrepresentable.  And an empty *training* fold is
        only reachable at ``n_folds == 1``, which is exactly what :meth:`single` builds --
        the ``cross_fit=False`` path, R's ``cvQinit = FALSE``, where the initial fit is
        deliberately evaluated on the rows that produced it.  Above one fold, every fold
        being non-empty already makes every complement non-empty.
        """
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1; got {self.n_folds}")
        assignment = np.asarray(self.assignment)
        if assignment.ndim != 1:
            raise DataError(
                f"fold assignment must be one index per row; got an array of shape "
                f"{assignment.shape}"
            )
        if assignment.size == 0:
            raise DataError("fold assignment is empty; there is nothing to cross-fit")
        low, high = int(assignment.min()), int(assignment.max())
        if low < 0 or high >= self.n_folds:
            raise DataError(
                f"fold assignment holds index/indices outside [0, {self.n_folds}): "
                f"the range present is [{low}, {high}]"
            )
        counts = np.bincount(assignment, minlength=self.n_folds)
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            raise DataError(
                f"fold(s) {empty.tolist()} hold no rows, so they can produce no "
                f"out-of-fold predictions; {self.n_folds} folds were declared but only "
                f"{int((counts > 0).sum())} are populated"
            )

    @property
    def n(self) -> int:
        return int(self.assignment.shape[0])

    def __len__(self) -> int:
        return self.n_folds

    def __iter__(self):  # type: ignore[no-untyped-def]
        # Recomputed on every pass rather than cached, and measured before being left
        # that way: 40 full iterations of a 10-fold split -- a generous count for one
        # fit, which iterates once per nuisance plus once per targeting solve -- cost
        # 2.5 ms at n=1000 and 6.8 ms at n=5000, against ~1.2 s for the whole fit with
        # the *cheapest* nuisance library (glm). That is under 1% of a fit, and an
        # order of magnitude less with the default SuperLearner. Caching it would mean
        # mutable state on a frozen dataclass for no measurable gain.
        for fold in range(self.n_folds):
            test = np.flatnonzero(self.assignment == fold)
            train = np.flatnonzero(self.assignment != fold)
            yield train, test

    def test_index(self, fold: int) -> IntArray:
        return np.flatnonzero(self.assignment == fold)

    @classmethod
    def single(cls, n: int) -> Folds:
        """A degenerate partition that trains and predicts on all rows.

        This is the ``cross_fit=False`` path -- it reproduces R's
        ``cvQinit = FALSE`` behaviour, where the initial fit is evaluated on the
        same data that produced it.
        """
        return cls(np.zeros(n, dtype=np.int64), 1)

    @property
    def is_single(self) -> bool:
        return self.n_folds == 1


def check_integrity(folds: Folds, *, cluster: IntArray | None = None) -> None:
    """Check the prohibition that needs a vector beside the assignment.

    Every row of a cluster must land in the same fold.  This is the prohibition that
    actually leaks: correlated rows split across the boundary let a nuisance model see,
    in training, rows that stand in for the ones it is about to predict, and the
    out-of-fold predictions come back optimistic.  An unclustered split passes trivially
    and pays nothing.

    Called as a post-condition of :func:`make_folds`, so every split this library builds
    is checked at construction.  It should never fire -- ``GroupKFold`` and
    ``StratifiedGroupKFold`` both guarantee it -- which is the point: the one place the
    guarantee is this library's own is the ``_grouped_splitter`` fallback for
    scikit-learn before 1.6, and a post-condition is what turns that from a hope into a
    checked claim.  Also exposed for the two holders of a :class:`Folds` that
    :func:`make_folds` never saw: a result reloaded from disk, and a caller who built one
    by hand.

    Stratum coverage is deliberately *not* checked here.  Every stratum in every fold is
    guaranteed only when the strata alone constrain the split -- ``resolve_n_folds`` caps
    the count to make it so -- and ``StratifiedGroupKFold`` cannot promise it, because a
    cluster is atomic and a rare class concentrated in few clusters has nowhere else to
    go.  Asserting it would refuse splits that are imbalanced but perfectly usable.

    Runs once per split rather than on the iteration path: one ``np.unique`` over ``n``
    rows, against the many nuisance fits a split goes on to serve.
    :meth:`Folds.__iter__` records that iterating a split at all is under 1% of a fit,
    and this is well below that.
    """
    if cluster is None:
        return
    assignment = np.asarray(folds.assignment)
    codes = np.asarray(cluster).reshape(-1)
    if codes.shape[0] != assignment.shape[0]:
        raise DataError(
            f"cluster has {codes.shape[0]} row(s) but the fold assignment has {assignment.shape[0]}"
        )
    # The distinct (cluster, fold) pairs: a cluster that stayed intact contributes
    # exactly one.  One sort beats an unbuffered scatter -- the same reason
    # inference.cluster.cluster_sums uses np.bincount rather than np.add.at.
    unique, inverse = np.unique(codes, return_inverse=True)
    pairs = np.unique(np.stack([inverse.reshape(-1), assignment], axis=1), axis=0)
    per_cluster = np.bincount(pairs[:, 0], minlength=unique.size)
    split = np.flatnonzero(per_cluster > 1)
    if split.size:
        raise DataError(
            f"{split.size} cluster(s) have rows in more than one fold, which is the "
            f"leakage grouped cross-fitting exists to prevent; the first is cluster code "
            f"{int(unique[split[0]])}, spread over {int(per_cluster[split[0]])} folds"
        )


def resolve_n_folds(n_folds: int, n: int, stratify: FloatArray | None = None) -> int:
    """Cap the requested number of folds at what the data can support.

    A stratified split needs at least one member of the rarer class per fold, so
    the cap is the rarer class count.  Silently exceeding it would raise deep
    inside scikit-learn, or worse, produce a fold with a single treatment arm.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2 for cross-fitting; got {n_folds}")
    cap = n
    if stratify is not None:
        counts = np.unique(np.asarray(stratify), return_counts=True)[1]
        cap = int(counts.min())
    resolved = int(min(n_folds, cap))
    if resolved < 2:
        raise ValueError(
            "cannot cross-fit: the rarer class has fewer than 2 members, so no "
            "stratified split exists"
        )
    if resolved < n_folds:
        warnings.warn(
            f"reducing n_folds from {n_folds} to {resolved}: the rarer class has only "
            f"{cap} member(s)",
            UserWarning,
            stacklevel=2,
        )
    return resolved


def make_folds(
    n: int,
    n_folds: int = 10,
    *,
    stratify: FloatArray | None = None,
    cluster: IntArray | None = None,
    random_state: int | np.random.Generator | None = None,
) -> Folds:
    """Build a cross-fitting partition of ``n`` observations.

    Parameters
    ----------
    stratify:
        Labels to balance across folds -- pass the treatment indicator.
    cluster:
        Cluster codes; every row of a cluster lands in the same fold.
    random_state:
        Seed or generator.  With ``cluster`` and ``stratify`` both given the
        split is deterministic given the seed, so a fit is reproducible.
    """
    if n < 2:
        raise ValueError(f"need at least 2 observations to cross-fit; got {n}")
    seed = _as_seed(random_state)
    resolved = resolve_n_folds(n_folds, n, stratify)
    x = np.zeros((n, 1))
    assignment = np.empty(n, dtype=np.int64)

    if cluster is not None:
        n_groups = int(np.unique(cluster).size)
        if n_groups < resolved:
            warnings.warn(
                f"reducing n_folds from {resolved} to {n_groups}: only {n_groups} clusters",
                UserWarning,
                stacklevel=2,
            )
            resolved = n_groups
        if resolved < 2:
            raise ValueError("cluster-respecting cross-fitting needs at least 2 clusters")
        if stratify is not None:
            splitter = StratifiedGroupKFold(n_splits=resolved, shuffle=True, random_state=seed)
            iterator = splitter.split(x, np.asarray(stratify), groups=cluster)
        else:
            # GroupKFold gained shuffle support in scikit-learn 1.6; fall back to
            # a seeded permutation of the group labels when it is unavailable.
            splitter, groups = _grouped_splitter(resolved, cluster, seed)
            iterator = splitter.split(x, None, groups=groups)
    elif stratify is not None:
        splitter = StratifiedKFold(n_splits=resolved, shuffle=True, random_state=seed)
        iterator = splitter.split(x, np.asarray(stratify))
    else:
        splitter = KFold(n_splits=resolved, shuffle=True, random_state=seed)
        iterator = splitter.split(x)

    for fold, (_, test) in enumerate(iterator):
        assignment[test] = fold
    folds = Folds(assignment, resolved)
    check_integrity(folds, cluster=cluster)
    return folds


def _grouped_splitter(
    n_splits: int, cluster: IntArray, seed: int | None
) -> tuple[GroupKFold, IntArray]:
    """A shuffled group split that works across scikit-learn versions.

    ``GroupKFold`` gained ``shuffle`` in scikit-learn 1.6 and ``pyproject.toml`` declares
    ``scikit-learn>=1.3``, so the fallback is supported rather than vestigial.  What it
    gives up is worth being precise about: older ``GroupKFold`` assigns groups to folds
    greedily by descending size, so permuting the group *labels* only reorders groups of
    equal size.  The shuffling is therefore weaker than 1.6's -- but cluster integrity,
    the prohibition that leaks, is a property of ``GroupKFold`` itself and is untouched.
    :func:`check_integrity` is what holds that claim to account on both paths.
    """
    try:
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), cluster
    except TypeError:
        rng = np.random.default_rng(seed)
        unique = np.unique(cluster)
        relabel = rng.permutation(unique.size)
        lookup = dict(zip(unique.tolist(), relabel.tolist(), strict=True))
        shuffled = np.array([lookup[int(c)] for c in cluster], dtype=np.int64)
        return GroupKFold(n_splits=n_splits), shuffled


@dataclass(frozen=True)
class CrossFitPlan:
    """The fold policy a fit declared, as distinct from the split it got.

    Every field is a number or a string, so a plan is comparable, hashable and
    serialisable, and says nothing about any particular dataset.  What a plan realises on
    one is a :class:`Folds`, and the two can differ: :func:`resolve_n_folds` caps
    ``n_folds`` at the rarest stratum and :func:`make_folds` caps it again at the cluster
    count, both with a warning at fit time and no trace afterwards.  Recording the plan
    beside the realised count is what makes "why did my 10-fold fit run 3 folds?"
    answerable from a saved result.

    Built from the estimator's own keyword arguments by
    :meth:`~cleverly.estimators.tmle.TMLE.crossfit_plan` and held on
    :class:`~cleverly.estimators.base.TMLEConfig`, exactly as
    :class:`~cleverly.estimators.targeting.TargetingSpec` is -- so the settings appear
    once and cannot drift.

    Attributes
    ----------
    n_folds:
        Outer folds, the ones that make the nuisance predictions out of fold.  ``1``
        means ``cross_fit=False``.
    learner_folds:
        Inner folds, which score Super Learner's candidates *inside* one outer training
        fold.  A separate declaration rather than the same one: this is model selection,
        not what makes a prediction out of fold, and its stratum is the learner's own
        target -- the treatment for the mechanism, the outcome for the regression.
    scheme:
        Which family of split was built, resolved from what the data declared rather
        than chosen: ``"grouped"`` whenever ``id=`` named clusters, otherwise
        ``"stratified"`` or ``"vfold"``.
    stratify_by:
        What the outer folds were balanced on, as user-facing names.  Empty when nothing
        was -- a continuous dose has no strata to balance.
    random_state:
        Seed for the split.  Not enough to reproduce it on its own; see
        :mod:`cleverly.provenance`.
    """

    n_folds: int = 10
    learner_folds: int = 5
    scheme: str = "stratified"
    stratify_by: tuple[str, ...] = ()
    random_state: int | None = None

    @property
    def cross_fit(self) -> bool:
        """Whether the outer nuisance fits are cross-fitted at all."""
        return self.n_folds > 1

    def describe(self) -> str:
        by = f" stratified on {', '.join(self.stratify_by)}" if self.stratify_by else ""
        if not self.cross_fit:
            return "declared: no cross-fitting (cross_fit=False)"
        return f"declared: {self.n_folds}-fold {self.scheme}{by}"


def refuse_scheme(kind: str) -> None:
    """Raise for a fold scheme this package will not fake.

    Four names, and four different reasons -- which is why they are spelled out rather
    than collected under one "not implemented".  Two of them are missing a derivation or
    a data layer; one of them is *incompatible with the storage contract* and would still
    be after both arrived, which is the one worth reading.
    """
    if kind == "blocked":
        raise NotImplementedError(
            "blocked temporal folds are not implemented. A contiguous-in-time split is "
            "perfectly expressible as a fold assignment -- what is missing is the "
            "ordering it would need. CausalData declares an outcome, a treatment, "
            "baseline covariates and a cluster, and no node carries a time index; id= "
            "is the independent sampling unit, not a row or time key. The scheme arrives "
            "with the longitudinal data layer, not before it."
        )
    if kind == "rolling_origin":
        raise NotImplementedError(
            "rolling-origin folds are not implemented, and not for want of a time index. "
            "Their training sets are nested prefixes: some rows are never held out and "
            "others fall in several evaluation windows, so there is no one fold that "
            "holds out each row. Cross-fitting here rests on exactly that -- Folds is an "
            "assignment of one fold per row, and NuisanceEstimates stores one "
            "out-of-fold prediction per row because of it. A rolling origin needs a "
            "different storage contract, not a different splitter."
        )
    if kind in {"repeated", "repeats"}:
        raise NotImplementedError(
            "repeated cross-fitting is not implemented. The aggregation is not the "
            "obstacle: every row is out of fold in every repeat, so averaging gives "
            "psi_bar = mean_r psi_r with influence curve mean_r IC_r, which stays "
            "coherent for the variance, the delta method and the simultaneous bands. "
            "What is missing is that R repeats mean R sets of nuisance fits, and "
            "TMLEResult.nuisance is one NuisanceEstimates that retarget -- and so every "
            "sensitivity analysis -- takes as a single object."
        )
    if kind == "row_within_cluster":
        raise ValueError(
            "splitting a cluster across folds is refused. id= declares the independent "
            "sampling unit, and it deliberately fuses the variance unit with the fold "
            "unit: dealing a cluster's correlated rows into different folds lets a "
            "nuisance model train on rows that stand in for the ones it predicts, which "
            "is the leakage grouped cross-fitting exists to prevent. Too few clusters to "
            "split is a reason to reduce n_folds, not to unfuse them."
        )
    raise ValueError(f"unknown cross-fitting scheme {kind!r}")


def _as_seed(random_state: int | np.random.Generator | None) -> int | None:
    """Turn a generator or seed into an int scikit-learn accepts."""
    if random_state is None:
        return None
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**31 - 1))
    return int(random_state)
