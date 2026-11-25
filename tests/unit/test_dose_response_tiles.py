"""The dose-response node pass reads row tiles and gives the untiled numbers (X106 D2).

The fit keeps the outcome regression at the quadrature doses per fold and reads it a row tile
at a time, so no ``(n, K)`` matrix is held.  :func:`cleverly.dose_response.target_pair` is the
untiled construction; at one BLAS thread every tiled number equals it bitwise.  A deliberate
row mix-up must fail the comparison, and the refusals keep the order of a rule-by-rule pass.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.estimators.dose_response as engine
from cleverly._support import UNCLASSIFIED
from cleverly.dose_response import DoseResponse, target_pair
from cleverly.estimators import DoseResponseTMLE
from cleverly.estimators._nuisance import NodePredictions, fit_nuisances
from cleverly.exceptions import DataError
from cleverly.learners._threads import thread_limit
from cleverly.learners.density import bin_edges, default_density_bins

CURVE = DoseResponse(doses=(0.8, 1.5, 2.2), bandwidths=(0.5, 0.8), support=(0.3, 2.7))


def _frame(n: int, seed: int = 3, v_share: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = np.clip(1.5 + 0.6 * w[:, 0] + rng.normal(size=n), 0.0, 3.0)
    mu = 0.3 + 0.2 * a - 0.04 * a * a + 0.2 * w[:, 1]
    frame = pd.DataFrame(w, columns=["W0", "W1"])
    frame["A"] = a
    frame["Y"] = mu + rng.normal(size=n)
    frame["V"] = (rng.uniform(size=n) < v_share).astype(float)
    frame["wt"] = rng.uniform(0.5, 2.0, n)
    return frame


def _nuisance(n: int, v_share: float = 0.4) -> tuple[DoseResponseTMLE, Any, Any, list[Any]]:
    estimator = DoseResponseTMLE(
        CURVE,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        q_bounds=(-6.0, 7.0),
        fluctuation="linear",
        random_state=0,
    )
    engine_ = estimator._engine()
    data = engine_._prepare(
        _frame(n, v_share=v_share),
        outcome="Y",
        treatment="A",
        covariates=["W0", "W1", "V"],
        delta=None,
        weights="wt",
        id=None,
        intermediate=None,
        strata=["V"],
        treatment_kind="continuous",
    )
    bins = default_density_bins(data.n)
    edges = bin_edges(data.treatment, bins, name="A")
    breaks = np.concatenate([edges, 0.5 * (edges[:-1] + edges[1:])])
    rules = [CURVE.quadrature(breaks), CURVE.quadrature(breaks, refine=2)]
    nodes = np.concatenate([rule.nodes for rule in rules])
    nuisance = fit_nuisances(
        data,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=None,
        intermediate_learner=None,
        folds=engine_._folds(data, 0),
        scaler=engine_._scaler(data),
        density_bins=bins,
        dose_nodes=tuple(float(dose) for dose in nodes),
    )
    return estimator, data, nuisance, rules


def _untiled(estimator: DoseResponseTMLE, data: Any, nuisance: Any, rule: Any, offset: int) -> Any:
    """Every cell by ``target_pair`` on whole ``(n, K)`` matrices: the reference."""
    n = data.n
    q_obs = nuisance.outcome.observed
    q_nodes = nuisance.node_predictions.block(np.arange(n), offset, offset + rule.n_nodes)
    g_obs = nuisance.density.density_at(data.treatment)
    g_nodes = np.column_stack([nuisance.density.density_at(np.full(n, d)) for d in rule.nodes])
    y = nuisance.scaler.scale(data.outcome)
    out = {}
    for code in range(data.n_strata):
        mask = (data.strata == code).astype(float)
        for p, (h, a0) in enumerate(CURVE.pairs):
            weight_obs = CURVE.weight(h, a0, data.treatment)
            # Written out here rather than read from the engine, so a change there is caught.
            h_obs = np.divide(
                weight_obs, g_obs, out=np.zeros_like(weight_obs), where=weight_obs != 0.0
            )
            node_weight = CURVE.weight(h, a0, rule.nodes)
            h_nodes = np.divide(
                np.broadcast_to(node_weight, g_nodes.shape),
                g_nodes,
                out=np.zeros_like(g_nodes),
                where=(node_weight != 0.0)[None, :] & (g_nodes > 0.0),
            )
            fitted = target_pair(
                y,
                q_obs,
                q_nodes,
                h_obs,
                h_nodes,
                rule.weights[p],
                data.weights * mask,
                fluctuation="linear",
            )
            name = engine._pair_cell_name(h, a0, data.stratum_label(code))
            out[name] = (fitted.psi, fitted.influence_curve)
    return out


@pytest.mark.parametrize("n", [1_001, 1_025])
@pytest.mark.parametrize("tile_bytes", [1, 10**6, engine.TILE_BYTES])
def test_tiles_give_the_untiled_numbers_bitwise(
    monkeypatch: pytest.MonkeyPatch, tile_bytes: int, n: int
) -> None:
    # n is not a multiple of 64, so the kernel's tail rows sit in the last tile (one tail row
    # at n = 1,025, which a one-row tile would hand to a dot product); the smallest budget
    # cuts 64-row tiles, which cross fold boundaries.
    monkeypatch.setattr(engine, "TILE_BYTES", tile_bytes)
    with thread_limit(1):
        estimator, data, nuisance, rules = _nuisance(n)
        found = estimator._target(
            data, nuisance, [(rules[0], 0), (rules[1], rules[0].n_nodes)], UNCLASSIFIED
        )
        offsets = [0, rules[0].n_nodes]
        for rule, offset, (estimates, _, _) in zip(rules, offsets, found, strict=True):
            reference = _untiled(estimator, data, nuisance, rule, offset)
            assert set(reference) == set(estimates)
            scale = nuisance.scaler
            for name, (psi, curve) in reference.items():
                assert estimates[name].psi == scale.unscale_level(psi), name
                expected = (curve * scale.range).tobytes()
                assert np.asarray(estimates[name].influence_curve).tobytes() == expected


def test_tiles_stay_within_the_contract_on_four_blas_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A multithreaded BLAS splits the untiled product its own way, so the contract is 1e-12
    # relative there.  Forced to four threads, which works under OPENBLAS_NUM_THREADS=1 too.
    import threadpoolctl

    monkeypatch.setattr(engine, "TILE_BYTES", 10**6)
    with threadpoolctl.threadpool_limits(4, user_api="blas"):
        estimator, data, nuisance, rules = _nuisance(3_001)
        found = estimator._target(
            data, nuisance, [(rules[0], 0), (rules[1], rules[0].n_nodes)], UNCLASSIFIED
        )
        offsets = [0, rules[0].n_nodes]
        for rule, offset, (estimates, _, _) in zip(rules, offsets, found, strict=True):
            reference = _untiled(estimator, data, nuisance, rule, offset)
            scale = nuisance.scaler
            for name, (psi, curve) in reference.items():
                want = scale.unscale_level(psi)
                assert abs(estimates[name].psi - want) <= 1e-12 * abs(want), name
                np.testing.assert_allclose(
                    estimates[name].influence_curve, curve * scale.range, rtol=1e-12, atol=1e-15
                )


@pytest.mark.parametrize("n", [1_025, 1_087, 4_097])
@pytest.mark.parametrize("k", [7, 61, 333])
def test_the_matrix_vector_product_is_bitwise_on_the_tile_rule(n: int, k: int) -> None:
    """The kernel fact the node pass rests on, checked on the BLAS this test runs against.

    Rows gathered in permuted fold order into fresh arrays, tiles of 64-row multiples and the
    data's last ``64 + n mod 64`` rows last, give each row the value the untiled product gives,
    at one thread.
    """
    rng = np.random.default_rng(n + k)
    matrix = rng.standard_normal((n, k))
    vector = rng.standard_normal(k)
    order = rng.permutation(n)
    folds = tuple(np.sort(part) for part in np.array_split(order, 5))
    stored = NodePredictions(folds, tuple(matrix[part] for part in folds))
    with thread_limit(1):
        whole = matrix @ vector
        for height in (64, 192, 1_024):
            tiled = np.empty(n)
            for rows in engine._tiles(stored, n, height):
                tiled[rows] = stored.block(rows, 0, k) @ vector
            assert tiled.tobytes() == whole.tobytes(), height


def test_a_row_mix_up_in_the_tiles_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberate mutation: each tile reads its predictions one row over.
    block = NodePredictions.block

    def shifted(self: NodePredictions, rows: np.ndarray, start: int, stop: int) -> np.ndarray:
        return block(self, np.roll(rows, 1), start, stop)

    monkeypatch.setattr(engine, "TILE_BYTES", 1)
    with thread_limit(1):
        estimator, data, nuisance, rules = _nuisance(400)
        monkeypatch.setattr(NodePredictions, "block", shifted)
        estimates, _, _ = estimator._target(data, nuisance, [(rules[0], 0)], UNCLASSIFIED)[0]
        monkeypatch.setattr(NodePredictions, "block", block)
        reference = _untiled(estimator, data, nuisance, rules[0], 0)
    moved = [
        name
        for name, (psi, _) in reference.items()
        if estimates[name].psi != nuisance.scaler.unscale_level(psi)
    ]
    assert moved


def test_the_tiles_cover_every_row_once_with_the_tail_last() -> None:
    n = 200
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    folds = (np.sort(order[:70]), np.sort(order[70:]))
    stored = NodePredictions(folds, (np.zeros((70, 1)), np.zeros((130, 1))))
    tiles = engine._tiles(stored, n, 64)
    assert np.array_equal(np.sort(np.concatenate(tiles)), np.arange(n))
    assert all(tile.size % 64 == 0 for tile in tiles[:-1])
    np.testing.assert_array_equal(tiles[-1], np.arange(128, 200))


# ------------------------------------------------------------------------ refusal order


class _Density:
    """Density one everywhere, except zero at the chosen nodes and observed rows."""

    def __init__(self, n: int, zero_nodes: tuple[float, ...] = ()) -> None:
        self.n = n
        self.zero_nodes = zero_nodes

    def density_at(self, values: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(values).size)

    def _density_on(self, rows: np.ndarray, doses: np.ndarray) -> np.ndarray:
        out = np.ones((rows.size, doses.size))
        out[:, np.isin(np.round(doses, 12), np.round(self.zero_nodes, 12))] = 0.0
        return out


def _refusing_target(
    monkeypatch: pytest.MonkeyPatch, zero_nodes: tuple[float, ...], unsolved_calls: set[int]
) -> str:
    estimator, data, _, rules = _nuisance(300)
    k = rules[0].n_nodes + rules[1].n_nodes
    stored = NodePredictions((np.arange(data.n),), (np.full((data.n, k), 0.5),))
    fake = SimpleNamespace(
        density=_Density(data.n, zero_nodes),
        node_predictions=stored,
        outcome=SimpleNamespace(observed=np.full(data.n, 0.5)),
        scaler=_nuisance_scaler(estimator, data),
    )
    original = engine.solve_fluctuation
    calls = iter(range(10**6))

    def solve(*args: Any, **kwargs: Any) -> tuple[float, float, bool]:
        epsilon, score, converged = original(*args, **kwargs)
        return epsilon, score, converged and next(calls) not in unsolved_calls

    monkeypatch.setattr(engine, "solve_fluctuation", solve)
    with pytest.raises(DataError) as raised:
        estimator._target(data, fake, [(rules[0], 0), (rules[1], rules[0].n_nodes)], UNCLASSIFIED)  # type: ignore[arg-type]
    return str(raised.value)


def _nuisance_scaler(estimator: DoseResponseTMLE, data: Any) -> Any:
    return estimator._engine()._scaler(data)


def _refined_only_node() -> float:
    _, _, _, rules = _nuisance(300)
    only = sorted(set(np.round(rules[1].nodes, 12)) - set(np.round(rules[0].nodes, 12)))
    used = rules[1].weights[0] != 0.0
    inside = [d for d in only if np.isin(d, np.round(rules[1].nodes[used], 12))]
    return float(inside[len(inside) // 2])


def test_a_coarse_failure_comes_before_a_refined_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # The last cell's fluctuation fails on the declared rule, and a node only the doubled
    # rule uses reads a zero density in the first cell: a rule-by-rule pass meets the first.
    last = 2 * len(CURVE.pairs) - 1
    message = _refusing_target(monkeypatch, (_refined_only_node(),), {last})
    assert "did not solve its score" in message


def test_the_first_cell_in_stratum_order_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cells run stratum by stratum, pair by pair; the second cell fails before the fourth.
    message = _refusing_target(monkeypatch, (), {1, 3})
    _, data, _, _ = _nuisance(300)
    h, a0 = CURVE.pairs[1]
    assert message.startswith(engine._pair_cell_name(h, a0, data.stratum_label(0)) + ":")
    assert "did not solve its score" in message


def test_a_zero_density_at_a_refined_node_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    message = _refusing_target(monkeypatch, (_refined_only_node(),), set())
    assert "the fitted density is zero at doses" in message


@pytest.mark.parametrize(
    ("cross_fit", "n_jobs", "outcome"),
    [(True, 1, "linear"), (True, 2, "linear"), (False, 1, "linear"), (True, 1, "logistic")],
)
def test_each_rows_node_prediction_comes_from_its_own_fold(
    cross_fit: bool, n_jobs: int, outcome: str
) -> None:
    """Per-fold storage against an independent path: the prediction at the observed dose.

    A third of the rows have their observed dose at a node, so for those rows the stored
    prediction at that node and the observed prediction come from the same fold's model on
    the same covariate row.  ``outcome.observed`` is predicted apart from the nodes, so a
    row stored under another fold or another position fails here, whatever ``block`` does.
    """
    nodes = (0.6, 1.1, 1.7, 2.3)
    frame = _frame(1_203, seed=11)
    rng = np.random.default_rng(12)
    at_node = rng.uniform(size=len(frame)) < 1 / 3
    frame.loc[at_node, "A"] = rng.choice(nodes, size=int(at_node.sum()))
    if outcome == "logistic":
        frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(float)
    learner = LinearRegression() if outcome == "linear" else LogisticRegression(max_iter=1000)
    estimator = DoseResponseTMLE(
        CURVE,
        outcome_learner=learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        q_bounds=None if outcome == "logistic" else (-6.0, 7.0),
        fluctuation="linear" if outcome == "linear" else "logistic",
        cross_fit=cross_fit,
        random_state=0,
    )
    engine_ = estimator._engine()
    data = engine_._prepare(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W0", "W1", "V"],
        delta=None,
        weights=None,
        id=None,
        intermediate=None,
        strata=None,
        treatment_kind="continuous",
    )
    with thread_limit(1):
        nuisance = fit_nuisances(
            data,
            outcome_learner=learner,
            treatment_learner=LogisticRegression(max_iter=1000),
            missingness_learner=None,
            intermediate_learner=None,
            folds=engine_._folds(data, 0),
            scaler=engine_._scaler(data),
            density_bins=8,
            n_jobs=n_jobs,
            dose_nodes=nodes,
        )
    stored = nuisance.node_predictions
    assert len(stored.tests) == (5 if cross_fit else 1)
    q = stored.block(np.arange(data.n), 0, len(nodes))
    observed = np.asarray(nuisance.outcome.observed, dtype=float)
    rows = np.flatnonzero(at_node)
    column = np.searchsorted(nodes, data.treatment[rows])
    np.testing.assert_allclose(q[rows, column], observed[rows], rtol=0, atol=1e-12)
    # Mutation controls: rows stored under the wrong position, or under another fold of the
    # same size, fail the comparison.
    tests = list(stored.tests)
    tests[0] = tests[0][::-1]
    mutations = [tests]
    if cross_fit:
        swapped = list(stored.tests)
        assert swapped[0].size == swapped[1].size
        swapped[0], swapped[1] = swapped[1], swapped[0]
        mutations.append(swapped)
    for wrong in mutations:
        moved = NodePredictions(tuple(wrong), stored.matrices).block(np.arange(data.n), 0, 4)
        assert np.max(np.abs(moved[rows, column] - observed[rows])) > 1e-6


def test_the_untargeted_observer_sees_each_cells_untiled_plug_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W9: the seam gets, per rule and cell in report order, ``Q(nodes) @ weights`` and weights."""
    monkeypatch.setattr(engine, "TILE_BYTES", 1)
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    monkeypatch.setattr(engine, "_untargeted_observer", lambda q, w: seen.append((q, w)))
    with thread_limit(1):
        estimator, data, nuisance, rules = _nuisance(1_025)
        offsets = [0, rules[0].n_nodes]
        estimator._target(data, nuisance, list(zip(rules, offsets, strict=True)), UNCLASSIFIED)
        expected = []
        for rule, offset in zip(rules, offsets, strict=True):
            q_nodes = nuisance.node_predictions.block(
                np.arange(data.n), offset, offset + rule.n_nodes
            )
            for code in range(data.n_strata):
                weights = data.weights * (data.strata == code).astype(float)
                for p in range(len(CURVE.pairs)):
                    expected.append((q_nodes @ rule.weights[p], weights))
    assert len(seen) == len(expected) == 2 * data.n_strata * len(CURVE.pairs)
    for (q, w), (want_q, want_w) in zip(seen, expected, strict=True):
        assert q.tobytes() == want_q.tobytes()
        assert w.tobytes() == want_w.tobytes()


@pytest.mark.parametrize("strata", [True, False])
def test_the_support_minimum_reads_each_stratums_rows(
    monkeypatch: pytest.MonkeyPatch, strata: bool
) -> None:
    """The smallest density at a used node over a stratum's rows, from 64-row tiles.

    A rare stratum (about 3% of rows) leaves most tiles wholly in the other stratum, some
    mixed and some without it, so every branch of the tile minimum runs; without strata every
    tile is whole.  The reference reads the density at every node over every row.
    """
    monkeypatch.setattr(engine, "TILE_BYTES", 1)
    with thread_limit(1):
        estimator, data, nuisance, rules = _nuisance(1_025, v_share=0.03)
        if not strata:
            data = dataclasses.replace(data, strata=None)
        found = estimator._target(data, nuisance, [(rules[0], 0)], UNCLASSIFIED)
    rule = rules[0]
    g_nodes = np.column_stack([nuisance.density.density_at(np.full(data.n, d)) for d in rule.nodes])
    g_obs = nuisance.density.density_at(data.treatment)
    codes = np.zeros(data.n, dtype=np.int64) if data.strata is None else data.strata
    _, supports, _ = found[0]
    labels = (
        [None] if data.strata is None else [data.stratum_label(c) for c in range(data.n_strata)]
    )
    by_name = {s.name: s for s in supports}
    checked = 0
    for code, label in enumerate(labels):
        rows = codes == code
        assert rows.any()
        for p, (h, a0) in enumerate(CURVE.pairs):
            used = rule.weights[p] != 0.0
            node_min = float(g_nodes[np.ix_(rows, used)].min())
            loaded = rows & (CURVE.weight(h, a0, data.treatment) != 0.0)
            want = min(node_min, float(g_obs[loaded].min())) if loaded.any() else node_min
            got = by_name[engine._pair_cell_name(h, a0, label)].min_density
            assert got == want
            checked += node_min < float(g_obs[loaded].min()) if loaded.any() else 1
    # Nonzero witness: the node minimum decides the reported minimum somewhere.
    assert checked > 0
