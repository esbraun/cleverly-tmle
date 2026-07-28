"""Fold construction for cross-fitting.

Nuisance models are fit out-of-fold so that no observation contributes to the
model used to predict it.  Two structural constraints matter and are enforced
here rather than left to the caller:

* **stratification** on the binary label keeps rare-treatment folds usable; a
  fold with no treated units cannot produce a propensity score;
* **cluster integrity** keeps every observation of a cluster in the same fold,
  otherwise cross-fitting leaks information between correlated rows and the
  out-of-fold predictions are optimistic.
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

__all__ = ["Folds", "make_folds", "resolve_n_folds"]


@dataclass(frozen=True)
class Folds:
    """A cross-fitting partition.

    ``assignment[i]`` is the index of the fold that holds out observation ``i``.
    Iterating yields ``(train_index, test_index)`` pairs like scikit-learn.
    """

    assignment: IntArray
    n_folds: int

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1; got {self.n_folds}")

    @property
    def n(self) -> int:
        return int(self.assignment.shape[0])

    def __len__(self) -> int:
        return self.n_folds

    def __iter__(self):  # type: ignore[no-untyped-def]
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
    return Folds(assignment, resolved)


def _grouped_splitter(
    n_splits: int, cluster: IntArray, seed: int | None
) -> tuple[GroupKFold, IntArray]:
    """A shuffled group split that works across scikit-learn versions."""
    try:
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), cluster
    except TypeError:
        rng = np.random.default_rng(seed)
        unique = np.unique(cluster)
        relabel = rng.permutation(unique.size)
        lookup = dict(zip(unique.tolist(), relabel.tolist(), strict=True))
        shuffled = np.array([lookup[int(c)] for c in cluster], dtype=np.int64)
        return GroupKFold(n_splits=n_splits), shuffled


def _as_seed(random_state: int | np.random.Generator | None) -> int | None:
    """Turn a generator or seed into an int scikit-learn accepts."""
    if random_state is None:
        return None
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**31 - 1))
    return int(random_state)
