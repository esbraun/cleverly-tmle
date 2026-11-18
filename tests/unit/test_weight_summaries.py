"""The weights plot reads per-node summaries that equal what ``node_weights`` gives.

X106 part J3b.  ``plotting.weights`` used to build four ``(n, T)`` arrays per regimen with
``node_weights`` and read them node by node; it now reads ``_weight_summary``, which takes the
same values from the fit one node at a time.  Each summary field is compared bitwise with the
value read off ``NodeWeights`` across the longitudinal routes, and ``node_weights`` itself with
a verbatim copy of the body it had before the selection moved into ``_resolve``.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.data.weighting import effective_sample_size
from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE, NodeWeights, node_weights
from cleverly.longitudinal import weights as module
from tests.unit import _capability_sweep_support as sweep


def _ltmle(*, zero_weights: bool = False, **overrides: Any) -> Any:
    frame, _ = make_longitudinal(n=300, seed=0)
    extra: dict[str, Any] = {}
    if zero_weights:
        # Every seventh unit carries weight zero, the rest 1 or 1.5.
        frame["w"] = np.where(np.arange(300) % 7 == 0, 0.0, 1.0 + 0.5 * (frame["W1"] > 0))
        extra["weights"] = "w"
    settings: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        **overrides,
    }
    return LTMLE({"always": 1, "never": 0}, **settings).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
        **extra,
    )


ROUTES: dict[str, Callable[[], Any]] = {
    "bounded": functools.partial(_ltmle, g_bounds=(0.3, 1.0)),
    "cross-fitted repeats": functools.partial(_ltmle, n_folds=2, repeats=2),
    "zero data weights": functools.partial(_ltmle, zero_weights=True),
    "iterated mean": sweep.fit_ltmle,
    "hazard": sweep.fit_hazard,
    "hazard multinode": sweep.fit_hazard_multinode,
    "odds tilt": sweep.fit_ipsi,
    "sdr modified policy, censored": sweep.fit_sdr_mtp_censored,
    "known sample weights": sweep.fit_ltmle_known_sample_weights,
    "competing": sweep.fit_competing,
    "repeated outcome": sweep.fit_repeated,
    "learned regimen": sweep.fit_learned_regimen,
}


@functools.cache
def _fit(route: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ROUTES[route]()


def _selections(result: Any) -> list[dict[str, Any]]:
    """Every regimen, cause, horizon and repeat the result holds."""
    out = []
    for repeat, draw in enumerate(result.repeats, start=1):
        for fit in draw.fits.values():
            out.append(
                {
                    "regimen": fit.regimen.label,
                    "cause": fit.cause,
                    "horizon": fit.horizon,
                    "repeat": repeat,
                }
            )
    return out


def _baseline_node_weights(result: Any, selection: dict[str, Any]) -> NodeWeights:
    """``node_weights`` as it stood before X106 J3b (main 70ccd44), after its refusals."""
    from cleverly.validation.longitudinal import _node_kind

    repeats = result.repeats
    repeat = selection["repeat"]
    fit = module._select(
        repeats[repeat - 1].fits, selection["regimen"], selection["horizon"], selection["cause"]
    )
    nodes = fit.scored_nodes
    n = fit.obs_weights.shape[0]
    used = np.full((n, len(nodes)), np.nan)
    followed = np.zeros((n, len(nodes)), dtype=bool)
    raw = np.empty((n, len(nodes)))
    bounded = np.empty((n, len(nodes)))
    for index, node in enumerate(nodes):
        weight = fit.obs_weights * node.clever
        used[node.trained_on, index] = weight[node.trained_on]
        followed[:, index] = node.trained_on
        raw[:, index] = fit.cumulative_unbounded[:, node.time - 1]
        bounded[:, index] = fit.cumulative[:, node.time - 1]
    return NodeWeights(
        regimen=fit.regimen.label,
        cause=fit.cause,
        horizon=fit.horizon,
        repeat=repeat,
        n_repeats=len(repeats),
        times=tuple(node.time for node in nodes),
        node_kinds=tuple(_node_kind(fit, node.time) for node in nodes),
        used=used,
        followed=followed,
        cumulative_unbounded=raw,
        cumulative=bounded,
        bounds=(float(result.config.g_bounds[0]), float(result.config.g_bounds[1])),
        unit_weights=bool(np.all(fit.obs_weights == 1.0)),
        cross_fitted=int(getattr(result.config, "n_folds", 1)) > 1,
        reports_curve=bool(result.data.reports_curve),
    )


def _bytes(value: Any) -> bytes:
    return np.asarray(value).tobytes()


@pytest.mark.parametrize("route", list(ROUTES))
def test_node_weights_is_unchanged(route: str) -> None:
    result = _fit(route)
    for selection in _selections(result):
        ours = node_weights(result, **selection)
        theirs = _baseline_node_weights(result, selection)
        for field in ("used", "followed", "cumulative_unbounded", "cumulative"):
            assert _bytes(getattr(ours, field)) == _bytes(getattr(theirs, field)), field
        for field in (
            "regimen",
            "cause",
            "horizon",
            "repeat",
            "n_repeats",
            "times",
            "node_kinds",
            "bounds",
            "unit_weights",
            "cross_fitted",
            "reports_curve",
        ):
            assert getattr(ours, field) == getattr(theirs, field), field


@pytest.mark.parametrize("route", list(ROUTES))
def test_each_summary_is_the_node_weights_value(route: str) -> None:
    result = _fit(route)
    for selection in _selections(result):
        full = node_weights(result, **selection)
        summary = module._weight_summary(result, **selection)
        assert summary.times == full.times
        assert summary.node_kinds == full.node_kinds
        assert (summary.bounds, summary.unit_weights) == (full.bounds, full.unit_weights)
        assert (summary.cross_fitted, summary.reports_curve) == (
            full.cross_fitted,
            full.reports_curve,
        )
        shares = full.share_beyond_bounds
        effective = full.effective_n
        for index, node in enumerate(summary.nodes):
            column = full.followed[:, index]
            assert node.time == full.times[index]
            assert _bytes(node.used) == _bytes(full.used[column, index])
            assert _bytes(node.share) == _bytes(shares[index])
            assert _bytes(node.effective_n) == _bytes(effective[index])
            assert node.followers == int(column.sum())


def test_the_routes_reach_what_the_plot_branches_on() -> None:
    """Nonzero witnesses: the routes hold moved probabilities, non-unit data weights, nodes the
    limit does not cap, several repeats and several causes."""
    summaries = [
        module._weight_summary(_fit(route), **selection)
        for route in ROUTES
        for selection in _selections(_fit(route))
    ]
    assert any(node.share > 0.0 for summary in summaries for node in summary.nodes)
    assert any(not summary.unit_weights for summary in summaries)
    assert any(
        kind not in ("label", "rule", "held")
        for summary in summaries
        for kind in summary.node_kinds
    )
    assert any(summary.n_repeats > 1 for summary in summaries)
    # The zero-weight note and a Kish size below the follower count.
    assert any(np.any(node.used == 0.0) for summary in summaries for node in summary.nodes)
    assert any(
        node.effective_n != node.followers for summary in summaries for node in summary.nodes
    )
    assert len({summary.cause for summary in summaries}) > 2


def test_a_share_read_from_the_wrong_column_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: the share of node t read at node t + 1 differs from node_weights."""
    result = _fit("bounded")
    selection = _selections(result)[0]
    full = node_weights(result, **selection)
    original = module._node_summaries

    def shifted(fit: Any) -> Any:
        found = original(fit)
        shares = [node.share for node in found]
        import dataclasses

        return tuple(
            dataclasses.replace(node, share=shares[(index + 1) % len(found)])
            for index, node in enumerate(found)
        )

    monkeypatch.setattr(module, "_node_summaries", shifted)
    summary = module._weight_summary(result, **selection)
    assert [node.share for node in summary.nodes] != list(full.share_beyond_bounds)


def test_the_kish_size_is_the_shared_helper() -> None:
    result = _fit("known sample weights")
    selection = _selections(result)[0]
    summary = module._weight_summary(result, **selection)
    for node in summary.nodes:
        assert node.effective_n == effective_sample_size(node.used, on_degenerate=0.0)


def test_the_bounded_fixture_moves_different_shares_by_node() -> None:
    """Premise of the wrong-column mutation: shares are nonzero and differ between nodes."""
    result = _fit("bounded")
    shares = list(node_weights(result, **_selections(result)[0]).share_beyond_bounds)
    assert any(share > 0.0 for share in shares) and len(set(shares)) > 1


def test_the_plot_builds_no_unit_by_node_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic guard: the plot never calls node_weights, the four (n, T) arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cleverly import plotting

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("plotting.weights built NodeWeights")

    monkeypatch.setattr(module, "node_weights", refused)
    for route in ("bounded", "hazard", "odds tilt"):
        plt.close(plotting.weights(_fit(route), regimens=[_selections(_fit(route))[0]["regimen"]]))
