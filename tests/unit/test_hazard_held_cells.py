"""The held cells of a hazard fit, counted by key (X106 J2g).

``_held_cells`` counts each cell's followers with ``bincount`` over the (node, key) index
instead of building the ``(n, T, K)`` follower indicator.  On a path fit K is the number of
distinct observed treatment prefixes, which grows with n and T, so the dense indicator grew
faster than linearly in n.  The cells must equal the dense ones exactly.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from cleverly.longitudinal.hazard import HazardBounds, _held_cells


def _dense(
    at_risk: np.ndarray, events: np.ndarray, weights: np.ndarray, observed: np.ndarray, n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """The cells from the dense ``(n, T, K)`` follower indicator (the implementation before J2g)."""
    keys = observed[:, :, None] if observed.ndim == 2 else observed[:, None, None]
    followers = (at_risk & (weights > 0.0)[:, None])[:, :, None] & (
        keys == np.arange(n_cells)[None, None, :]
    )
    present = followers.any(axis=0)
    happened = events > 0.0
    with_event = np.einsum("nkl,nkj->klj", followers, happened)
    without_event = np.einsum("nkl,nk->kl", followers, ~happened.any(axis=-1))
    return present[:, :, None] & (with_event == 0), present & (without_event == 0)


def _inputs(rng: np.random.Generator, node_keys: bool) -> tuple[np.ndarray, ...]:
    n, n_times, causes, n_cells = (int(v) for v in (rng.integers(1, 80), rng.integers(1, 6),
                                                    rng.integers(1, 3), rng.integers(1, 9)))
    at_risk = rng.random((n, n_times)) < 0.7
    events = (rng.random((n, n_times, causes)) < rng.random()) * rng.random((n, n_times, causes))
    weights = np.where(rng.random(n) < 0.2, 0.0, rng.random(n))
    # Keys outside [0, K) (a missing arm, a key no cell owns) follow no cell.
    observed = rng.integers(-1, n_cells + 1, size=(n, n_times) if node_keys else n)
    return at_risk, events, weights, observed, np.int64(n_cells)


@pytest.mark.parametrize("node_keys", [False, True], ids=["level key", "prefix key"])
def test_the_counted_cells_equal_the_dense_cells(node_keys: bool) -> None:
    rng = np.random.default_rng(20261119)
    for _ in range(200):
        at_risk, events, weights, observed, n_cells = _inputs(rng, node_keys)
        empty, full, excluded = _held_cells(at_risk, events, weights, observed, int(n_cells), None)
        dense_empty, dense_full = _dense(at_risk, events, weights, observed, int(n_cells))
        assert excluded is None
        assert empty.dtype == dense_empty.dtype == bool and full.dtype == dense_full.dtype == bool
        np.testing.assert_array_equal(empty, dense_empty)
        np.testing.assert_array_equal(full, dense_full)


def test_a_bounded_node_holds_the_counted_cells_as_the_dense_ones() -> None:
    rng = np.random.default_rng(7)
    at_risk, events, weights, observed, n_cells = _inputs(rng, True)
    while events.shape[1] < 2:
        at_risk, events, weights, observed, n_cells = _inputs(rng, True)
    n_times, causes = events.shape[1], events.shape[2]
    bounded = np.zeros(n_times, dtype=bool)
    bounded[0] = True
    lower = np.zeros((n_times, causes))
    lower[0, 0] = 0.1
    bounds = HazardBounds(bounded=bounded, lower=lower, scale=np.ones((n_times, causes)))
    empty, full, excluded = _held_cells(at_risk, events, weights, observed, int(n_cells), bounds)
    dense_empty, dense_full = _dense(at_risk, events, weights, observed, int(n_cells))
    held = dense_empty & ~(bounded[:, None, None] & (lower > 0.0)[:, None, :])
    np.testing.assert_array_equal(empty, held)
    np.testing.assert_array_equal(full, dense_full & ~bounded[:, None])
    assert excluded is not None


def test_the_cells_never_build_an_n_by_t_by_k_array() -> None:
    # A path fit at T = 12 holds about 1,500 distinct prefixes at 5,000 units; the dense
    # indicator was n * T * K booleans twice.  Here n * T * K is 60 MB.
    rng = np.random.default_rng(3)
    n, n_times, n_cells = 20_000, 6, 500
    at_risk = rng.random((n, n_times)) < 0.8
    events = (rng.random((n, n_times, 1)) < 0.05).astype(float)
    weights = np.ones(n)
    observed = rng.integers(0, n_cells, size=(n, n_times))
    tracemalloc.start()
    try:
        _held_cells(at_risk, events, weights, observed, n_cells, None)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < n * n_times * n_cells / 10
