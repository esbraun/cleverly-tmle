r"""Cluster aggregation: indexed accumulation, which is the shape numpy is worst at.

``cluster_sums`` densifies the labels with ``np.unique`` and then calls ``np.bincount``
once per estimand.  Both halves are worth naming, because they fail for different reasons
and only one of them is about compilation:

* ``np.unique`` **sorts**.  It is ``O(n log n)`` on the row count for a job whose answer
  needs only a hash, and it is paid on every call -- including inside a cluster bootstrap,
  where the same labels are densified once per replicate.  (The package already found and
  fixed the worse version of this: the bootstrap used to rebuild its membership index per
  draw.  The sort survived.)
* ``np.bincount`` **per column** re-reads the ``(n,)`` index vector once per estimand and
  writes ``m`` separate output arrays.  With ``m = 20`` that is twenty passes over the
  labels where one would do, and the measured cost is very nearly linear in ``m``:
  at ``n = 1e6`` this module measures 78 ms at ``m = 1`` and 468 ms at ``m = 20``.

A compiled kernel does one pass, reading each row's label once and adding its whole
``m``-vector -- so the label traffic is ``1/m`` of numpy's and the sort is gone.

**The parallel version is the interesting one, and not because it is obviously right.**
Rows map to clusters many-to-one, so a ``prange`` over rows writes to shared slots.  Two
ways out, and the benchmark runs both because which wins depends on a dimension a reader
would not think to vary:

``thread-local``
    each thread accumulates into its own ``(C, m)`` block and the blocks are summed.  No
    contention at all, at a cost of ``threads x C x m`` doubles -- which at ``C = 100,000``
    and ``m = 20`` is 128 MB *per thread*, so this is not free and the memory column in the
    report is the point.
``by-cluster``
    ``prange`` over clusters after a counting sort, so each thread owns its outputs
    outright and the memory is ``O(n)`` once.  Perfectly balanced when the clusters are,
    and badly balanced when they are not -- which is why :func:`~..fixtures.make_cluster`
    takes a ``shape`` and why ``skewed`` is in the sweep.

Atomics are deliberately absent.  With a skewed design most rows target a handful of
slots and the threads serialise on exactly those, which is the worst of both.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import make_cluster
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_arrays
from . import KernelSpec, register

__all__ = [
    "build",
    "numba_cluster_sums",
    "numba_cluster_sums_by_cluster",
    "numba_cluster_sums_threadlocal",
    "numpy_cluster_sums",
    "numpy_cluster_sums_sorted",
]


def build(
    n: int = 100_000,
    n_clusters: int = 1000,
    n_estimands: int = 5,
    shape: str = "balanced",
    seed: int = 20260803,
) -> dict[str, Any]:
    fixture = make_cluster(
        n, n_clusters=n_clusters, n_estimands=n_estimands, shape=shape, seed=seed
    )
    return {
        "influence": fixture.influence,
        "cluster": fixture.cluster,
        "n_clusters": fixture.n_clusters,
        "shape": shape,
    }


# --------------------------------------------------------------------------- numpy


def numpy_cluster_sums(inputs: dict[str, Any]) -> np.ndarray:
    """The shipped path: ``np.unique`` then one ``np.bincount`` per column."""
    ic = inputs["influence"]
    codes = inputs["cluster"]
    unique, inverse = np.unique(codes, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_clusters = unique.size
    if ic.ndim == 1:
        return np.bincount(inverse, weights=ic, minlength=n_clusters).astype(float)
    return np.column_stack(
        [
            np.bincount(inverse, weights=ic[:, column], minlength=n_clusters)
            for column in range(ic.shape[1])
        ]
    ).astype(float)


def numpy_cluster_sums_sorted(inputs: dict[str, Any]) -> np.ndarray:
    """A numpy alternative: sort once, then ``np.add.reduceat`` over the whole matrix.

    Here as the plan's "improve the numpy instead" arm.  It still sorts, but it sorts
    *once* and then reduces all ``m`` columns in one call rather than ``m`` times, so it
    separates the two costs the shipped path pays together -- and if it closes most of the
    gap on its own, the compiled kernels have to beat *it* rather than the version that
    does the same work twenty times.
    """
    ic = np.atleast_2d(inputs["influence"].T).T
    codes = np.asarray(inputs["cluster"]).reshape(-1)
    order = np.argsort(codes, kind="stable")
    ordered_codes = codes[order]
    boundaries = np.flatnonzero(np.r_[True, ordered_codes[1:] != ordered_codes[:-1]])
    summed = np.add.reduceat(ic[order], boundaries, axis=0)
    return np.asarray(summed, dtype=float)


# --------------------------------------------------------------------------- numba


@njit()
def _densify(codes: np.ndarray, table_size: int) -> tuple[np.ndarray, int]:
    """Map arbitrary integer labels onto ``0..C-1`` with an open-addressed hash table.

    This is what replaces ``np.unique``'s sort.  The labels are assigned dense indices in
    **first-appearance order**, which is *not* the sorted order numpy produces -- so the
    rows of the two results are permutations of each other and the comparison has to sort
    both before checking them.  Returning first-appearance order rather than paying for a
    sort is the whole saving; pretending the orders match would be the bug.
    """
    mask = table_size - 1
    keys = np.full(table_size, -1, dtype=np.int64)
    slots = np.empty(table_size, dtype=np.int64)
    dense = np.empty(codes.shape[0], dtype=np.int64)
    count = 0
    for i in range(codes.shape[0]):
        key = codes[i]
        # Fibonacci hashing: multiply by the 64-bit golden ratio and take the high bits,
        # which mixes the low-entropy labels a real `id` column has (contiguous integers,
        # or contiguous integers times a stride) far better than a modulo would.
        slot = int((np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) & mask
        while True:
            existing = keys[slot]
            if existing == -1:
                keys[slot] = key
                slots[slot] = count
                dense[i] = count
                count += 1
                break
            if existing == key:
                dense[i] = slots[slot]
                break
            slot = (slot + 1) & mask
    return dense, count


@njit()
def _table_size(n: int) -> int:
    size = 16
    while size < 2 * n:
        size *= 2
    return size


@njit()
def _sums_serial(influence: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """One pass: each row's label is read once and its whole ``m``-vector added."""
    dense, count = _densify(codes, _table_size(codes.shape[0]))
    columns = influence.shape[1]
    out = np.zeros((count, columns))
    for i in range(influence.shape[0]):
        target = dense[i]
        for j in range(columns):
            out[target, j] += influence[i, j]
    return out


@pjit()
def _sums_threadlocal(influence: np.ndarray, codes: np.ndarray, n_threads: int) -> np.ndarray:
    """Each thread owns a private ``(C, m)`` block; the blocks are reduced at the end.

    Costs ``n_threads x C x m`` doubles, which the report carries in its memory column
    because at a large cluster count it is the reason to prefer the by-cluster kernel.
    """
    dense, count = _densify(codes, _table_size(codes.shape[0]))
    rows = influence.shape[0]
    columns = influence.shape[1]
    partial = np.zeros((n_threads, count, columns))
    span = (rows + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * span
        stop = min(rows, start + span)
        for i in range(start, stop):
            target = dense[i]
            for j in range(columns):
                partial[t, target, j] += influence[i, j]
    out = np.zeros((count, columns))
    for t in range(n_threads):
        for c in range(count):
            for j in range(columns):
                out[c, j] += partial[t, c, j]
    return out


@njit()
def _counting_sort(dense: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows grouped by cluster: ``starts`` of length ``count + 1`` and the row order.

    A counting sort rather than a comparison sort -- the keys are already dense integers
    in ``0..C-1``, so this is two linear passes where ``np.argsort`` is ``n log n``.
    """
    starts = np.zeros(count + 1, dtype=np.int64)
    for i in range(dense.shape[0]):
        starts[dense[i] + 1] += 1
    for c in range(count):
        starts[c + 1] += starts[c]
    cursor = starts.copy()
    order = np.empty(dense.shape[0], dtype=np.int64)
    for i in range(dense.shape[0]):
        target = dense[i]
        order[cursor[target]] = i
        cursor[target] += 1
    return starts, order


@pjit()
def _sums_by_cluster(influence: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """``prange`` over clusters after a counting sort: each thread owns its output rows.

    Memory is ``O(n)`` once rather than ``O(threads x C x m)``, which is why this is the
    kernel to reach for at a large cluster count.  What it gives up is load balance: with
    a skewed design a handful of clusters hold most of the rows, and the thread that draws
    one of them runs long after the others are idle.
    """
    dense, count = _densify(codes, _table_size(codes.shape[0]))
    starts, order = _counting_sort(dense, count)
    columns = influence.shape[1]
    out = np.zeros((count, columns))
    for c in prange(count):
        for position in range(starts[c], starts[c + 1]):
            row = order[position]
            for j in range(columns):
                out[c, j] += influence[row, j]
    return out


def numba_cluster_sums(inputs: dict[str, Any]) -> np.ndarray:
    return _sums_serial(inputs["influence"], inputs["cluster"])


def numba_cluster_sums_threadlocal(inputs: dict[str, Any]) -> np.ndarray:
    from ..implementations.numba_parallel import effective_threads

    return _sums_threadlocal(inputs["influence"], inputs["cluster"], effective_threads())


def numba_cluster_sums_by_cluster(inputs: dict[str, Any]) -> np.ndarray:
    return _sums_by_cluster(inputs["influence"], inputs["cluster"])


def _compare_unordered(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Compare two cluster-sum matrices whose rows are in different label orders.

    ``np.unique`` produces sorted labels and the hash densifier produces first-appearance
    order, so the two results are row permutations of one another.  Sorting both by their
    own rows before comparing is the correct reconciliation *here* because a cluster sum is
    a set of per-cluster totals and nothing downstream reads them by position -- the
    variance is a sum of squares over the rows.  It would not be correct for a kernel whose
    output is indexed, and none of those use it.
    """
    a = np.atleast_2d(np.asarray(reference, dtype=float).T).T
    b = np.atleast_2d(np.asarray(candidate, dtype=float).T).T
    if a.shape != b.shape:
        return float("inf"), float("inf")
    a = a[np.lexsort(a.T[::-1])]
    b = b[np.lexsort(b.T[::-1])]
    return compare_arrays(a, b)


_IMPLEMENTATIONS: dict[str, Any] = {
    "numpy": numpy_cluster_sums,
    "numpy_sorted": numpy_cluster_sums_sorted,
}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_cluster_sums
    _IMPLEMENTATIONS["numba_parallel"] = numba_cluster_sums_threadlocal
    _IMPLEMENTATIONS["numba_parallel_by_cluster"] = numba_cluster_sums_by_cluster

register(
    KernelSpec(
        name="cluster_sums",
        estimator="inference",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=_compare_unordered,
        # A sum of a few thousand doubles in a different order: the disagreement is
        # accumulated rounding and grows with the largest cluster, so the bar is relative.
        tolerance=(1e-9, 1e-12),
        parallel_axis="clusters",
        note="indexed accumulation, m passes over the labels, and a sort that buys nothing",
        dimensions={
            "n": 100_000,
            "n_clusters": 1000,
            "n_estimands": 5,
            "shape": "balanced",
            "seed": 20260803,
        },
        amortise=True,
    )
)
