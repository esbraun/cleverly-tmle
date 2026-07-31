"""What the longitudinal container accepts, and what it refuses rather than guesses at."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import DataError
from cleverly.longitudinal import LongitudinalData, Regimen, resolve_regimens


def panel(n: int = 40, *, seed: int = 0) -> pd.DataFrame:
    """A small, valid two-time-point panel with monotone censoring."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    a1 = rng.integers(0, 2, n).astype(float)
    c1 = (rng.random(n) < 0.8).astype(float)
    l2 = np.where(c1 == 1, rng.standard_normal(n), np.nan)
    a2 = np.where(c1 == 1, rng.integers(0, 2, n).astype(float), np.nan)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    observed = (c1 == 1) & (c2 == 1)
    y = np.where(observed, rng.integers(0, 2, n).astype(float), np.nan)
    return pd.DataFrame({"W1": w1, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})


def build(frame: pd.DataFrame, **overrides: Any) -> LongitudinalData:
    kwargs: dict[str, Any] = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }
    kwargs.update(overrides)
    return LongitudinalData.from_frame(frame, **kwargs)


def test_reads_the_panel() -> None:
    data = build(panel())
    assert data.n_times == 2
    assert data.baseline_names == ("W1",)
    assert data.time_varying_names == ((), ("L2",))
    assert data.family == "binomial"
    assert data.has_censoring


def test_masks_line_up_with_the_recursion() -> None:
    """``at_risk(t)`` at one node is the set the *previous* node is fitted on.

    That identity is what makes the backward recursion close, so it is checked here
    rather than left as a comment in the estimator.
    """
    data = build(panel())
    plan = (1.0, 1.0)
    for time in range(1, data.n_times):
        np.testing.assert_array_equal(data.at_risk(plan, time + 1), data.following(plan, time))
    np.testing.assert_array_equal(data.at_risk(plan, 1), np.ones(data.n, dtype=bool))


def test_history_grows_one_block_at_a_time() -> None:
    data = build(panel())
    assert data.covariate_history(1).shape == (data.n, 1)
    assert data.covariate_history(2).shape == (data.n, 2)
    # The mechanism at the second node conditions on the first treatment; the censoring
    # model at the same node additionally conditions on the second.
    assert data.history_design(2).shape == (data.n, 3)
    assert data.history_design(2, include_current=True).shape == (data.n, 4)
    at_regimen = data.history_design(2, treatment=(1.0, 0.0), include_current=True)
    np.testing.assert_array_equal(at_regimen[:, -2], np.ones(data.n))
    np.testing.assert_array_equal(at_regimen[:, -1], np.zeros(data.n))


def test_the_fill_for_censored_rows_carries_no_information() -> None:
    """The zero-fill is confined to rows nothing reads, and cannot encode anything.

    The container zero-fills the covariates of a unit censored before the node, so that
    a learner can be *called* on the whole matrix in one pass.  Two things together make
    that safe, and both are checked: the fill lands only where the unit is not at risk,
    and the container refuses a recorded value there in the first place
    (``test_refuses_a_node_recorded_after_censoring``), so there is no number for the
    fill to have replaced.
    """
    data = build(panel())
    reachable = data.uncensored_through(1)
    design = data.covariate_history(2)
    assert np.all(np.isfinite(design))
    np.testing.assert_array_equal(design[~reachable, 1], 0.0)
    np.testing.assert_array_equal(design[reachable, 1], data.time_varying[1][reachable, 0])


def test_refuses_a_unit_that_returns_after_censoring() -> None:
    frame = panel()
    frame.loc[frame.index[0], ["C1", "C2"]] = [0.0, 1.0]
    with pytest.raises(DataError, match="monotone"):
        build(frame)


def test_refuses_a_node_recorded_after_censoring() -> None:
    frame = panel()
    censored = frame.index[frame["C1"] == 0][0]
    frame.loc[censored, "L2"] = 0.3
    with pytest.raises(DataError, match="had already been censored"):
        build(frame)


def test_refuses_a_node_missing_before_censoring() -> None:
    frame = panel()
    present = frame.index[frame["C1"] == 1][0]
    frame.loc[present, "A2"] = np.nan
    with pytest.raises(DataError, match="still under observation"):
        build(frame)


def test_refuses_an_outcome_missing_for_any_other_reason() -> None:
    """A missing outcome on an uncensored unit is a node, not an absence."""
    frame = panel()
    complete = frame.index[(frame["C1"] == 1) & (frame["C2"] == 1)][0]
    frame.loc[complete, "Y"] = np.nan
    with pytest.raises(DataError, match="never censored but have no"):
        build(frame)


def test_refuses_a_non_binary_treatment() -> None:
    frame = panel()
    frame.loc[frame.index[0], "A1"] = 2.0
    with pytest.raises(DataError, match="binary treatment at every node"):
        build(frame)


def test_refuses_a_column_used_twice() -> None:
    with pytest.raises(DataError, match="more than one node"):
        build(panel(), baseline=["W1", "A1"])


def test_refuses_a_censoring_column_per_node_mismatch() -> None:
    with pytest.raises(DataError, match="one censoring column per time point"):
        build(panel(), censoring=["C1"])


def test_works_without_censoring_columns() -> None:
    frame = panel()
    frame = frame[frame["C1"] * frame["C2"] == 1].reset_index(drop=True)
    data = build(frame, censoring=None)
    assert not data.has_censoring
    assert data.uncensored.all()


class TestRegimens:
    def test_scalar_broadcasts_across_the_nodes(self) -> None:
        (always,) = resolve_regimens({"always": 1}, 3)
        assert always.values == (1.0, 1.0, 1.0)

    def test_keeps_declaration_order(self) -> None:
        regimens = resolve_regimens({"b": 0, "a": 1}, 2)
        assert [regimen.label for regimen in regimens] == ["b", "a"]

    def test_refuses_a_plan_of_the_wrong_length(self) -> None:
        with pytest.raises(DataError, match="assigns 3 arm"):
            resolve_regimens({"odd": (1, 0, 1)}, 2)

    def test_refuses_a_non_binary_arm(self) -> None:
        with pytest.raises(DataError, match="must be 0 or 1"):
            Regimen("dose", (0.5, 1.0))

    def test_refuses_a_rule_that_reads_the_history(self) -> None:
        with pytest.raises(DataError, match="reads the history is not supported"):
            resolve_regimens({"dynamic": lambda history: history}, 2)
