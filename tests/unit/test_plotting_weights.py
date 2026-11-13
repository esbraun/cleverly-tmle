"""The weights plot of a longitudinal fit and the covariate-balance plot of a point fit.

Each accessor is checked against the numbers the fit stores and against a hand computation;
each honesty statement of a plot has a mutation control that makes the check fail.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cleverly import ATC, ATE, ATT, CausalStudy, PointTreatment, plotting
from cleverly.data.weighting import effective_sample_size
from cleverly.datasets import make_linear_ate, make_longitudinal, make_multi_arm
from cleverly.exceptions import CapabilityError, DataError
from cleverly.longitudinal import LTMLE, NodeWeights, gcomp, node_weights
from cleverly.plotting import _core, _weights
from cleverly.sensitivity import CovariateBalance, covariate_balance
from cleverly.sensitivity.balance import BalanceRow
from tests.unit._capability_sweep_support import fit_ipsi
from tests.unit._plotting_checks import assert_style, assert_text_inside

LONGITUDINAL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _ltmle(regimens: Any, frame: Any, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        **overrides,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(regimens, **settings).fit(frame, **LONGITUDINAL_COLUMNS)


@pytest.fixture(scope="module")
def longitudinal() -> Any:
    frame, _ = make_longitudinal(n=300, seed=0)
    return _ltmle({"always": 1, "never": 0}, frame)


@pytest.fixture(scope="module")
def bounded() -> Any:
    """A fit whose limit on the cumulative probability moves some units."""
    frame, _ = make_longitudinal(n=300, seed=0)
    return _ltmle({"always": 1, "never": 0}, frame, g_bounds=(0.3, 1.0))


@pytest.fixture(scope="module")
def repeated() -> Any:
    frame, _ = make_longitudinal(n=240, seed=3)
    return _ltmle({"always": 1}, frame, n_folds=2, repeats=2)


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def _close(fig: Any) -> None:
    plt.close(fig)


def _assert_descriptive(fig: Any) -> None:
    """The honesty check of the love plot: its notes say the differences are descriptive."""
    assert "carry no interval or test" in _core.note_text(fig)


def _stage_rows(result: Any) -> dict[tuple[str, int], Any]:
    return {(row.regimen, row.time): row for row in result.diagnostics.support().rows}


class TestNodeWeights:
    def test_the_used_weight_is_the_fits_weight_on_its_followers(self, longitudinal: Any) -> None:
        for label in ("always", "never"):
            weights = node_weights(longitudinal, regimen=label)
            (fit,) = [f for f in longitudinal.fits.values() if f.regimen.label == label]
            assert isinstance(weights, NodeWeights)
            assert weights.times == (1, 2)
            for index, node in enumerate(fit.scored_nodes):
                expected = (fit.obs_weights * node.clever)[node.trained_on]
                np.testing.assert_array_equal(weights.used[node.trained_on, index], expected)
                assert np.all(np.isnan(weights.used[~node.trained_on, index]))
                np.testing.assert_array_equal(weights.followed[:, index], node.trained_on)

    def test_the_kish_effective_n_equals_the_diagnostics_row(self, longitudinal: Any) -> None:
        rows = _stage_rows(longitudinal)
        for label in ("always", "never"):
            weights = node_weights(longitudinal, regimen=label)
            for index, time in enumerate(weights.times):
                row = rows[(label, time)]
                assert weights.effective_n[index] == pytest.approx(row.effective_n)
                assert int(weights.followed[:, index].sum()) == row.n_followed

    def test_a_static_plan_with_unit_weights_divides_by_the_bounded_probability(
        self, bounded: Any
    ) -> None:
        weights = node_weights(bounded, regimen="always")
        assert weights.unit_weights
        for index in range(len(weights.times)):
            mask = weights.followed[:, index]
            np.testing.assert_allclose(
                weights.used[mask, index], 1.0 / weights.cumulative[mask, index]
            )
        # the witness: the limit moved some units, so dividing by the raw probability, the
        # mutated reading, gives other numbers
        moved = weights.cumulative_unbounded != weights.cumulative
        assert (moved & weights.followed).any()
        mutated = 1.0 / weights.cumulative_unbounded[weights.followed[:, 0], 0]
        assert not np.allclose(weights.used[weights.followed[:, 0], 0], mutated)
        assert weights.used[weights.followed].max() <= 1.0 / 0.3 + 1e-9

    def test_the_share_beyond_the_bounds_equals_the_diagnostics_share_truncated(
        self, bounded: Any
    ) -> None:
        rows = _stage_rows(bounded)
        weights = node_weights(bounded, regimen="never")
        for index, time in enumerate(weights.times):
            assert weights.share_beyond_bounds[index] == pytest.approx(
                rows[("never", time)].share_truncated
            )
        assert weights.share_beyond_bounds.max() > 0.0
        assert weights.bounds == (0.3, 1.0)

    def test_it_names_the_horizon_and_repeat(self, repeated: Any) -> None:
        first = node_weights(repeated)
        second = node_weights(repeated, repeat=2)
        assert (first.repeat, second.repeat, first.n_repeats) == (1, 2, 2)
        assert first.horizon == second.horizon == 2
        (fit,) = repeated.repeats[1].fits.values()
        np.testing.assert_array_equal(
            second.cumulative, np.column_stack([fit.cumulative[:, t - 1] for t in (1, 2)])
        )
        assert not np.array_equal(first.cumulative, second.cumulative)

    def test_to_frame_has_one_row_per_unit_and_node(self, longitudinal: Any) -> None:
        weights = node_weights(longitudinal, regimen="always")
        frame = weights.to_frame()
        n, nodes = weights.used.shape
        assert len(frame) == n * nodes
        assert {"regimen", "horizon", "repeat", "time", "weight_used", "cumulative"} <= set(
            frame.columns
        )
        first = frame[frame["time"] == 1]
        np.testing.assert_array_equal(first["weight_used"].to_numpy(), weights.used[:, 0])

    def test_refusals(self, longitudinal: Any) -> None:
        with pytest.raises(DataError, match="several regimens"):
            node_weights(longitudinal)
        with pytest.raises(DataError, match="no regimen 'nope'"):
            node_weights(longitudinal, regimen="nope")
        with pytest.raises(DataError, match="repeat must be"):
            node_weights(longitudinal, regimen="always", repeat=2)
        with pytest.raises(DataError, match="no horizon"):
            node_weights(longitudinal, regimen="always", horizon=9)
        frame, _ = make_longitudinal(n=200, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = gcomp(
                frame,
                regimens={"always": 1},
                n_folds=1,
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                censoring_learner=LogisticRegression(max_iter=1000),
                **LONGITUDINAL_COLUMNS,
            )
        with pytest.raises(CapabilityError, match="no weights"):
            node_weights(fit)


class TestTheWeightsPlot:
    def test_it_draws_a_box_per_regimen_and_node_from_the_used_weights(
        self, longitudinal: Any
    ) -> None:
        fig = plotting.weights(longitudinal)
        top, bottom = fig.axes[0], fig.axes[1]
        assert top.get_yscale() == "log"
        assert top.get_title(loc="left") == "Weights over time"
        assert top.get_xlabel() == "" and bottom.get_xlabel() == "Time point"
        crosses = [c for c in top.collections if c.get_gid() == "cleverly-weight-max"]
        assert len(crosses) == 2
        for cross, label in zip(crosses, ("always", "never"), strict=True):
            weights = node_weights(longitudinal, regimen=label)
            peaks = [
                float(np.nanmax(weights.used[weights.followed[:, i], i]))
                for i in range(len(weights.times))
            ]
            np.testing.assert_allclose(cross.get_offsets()[:, 1], peaks)
        _close(fig)

    def test_the_second_panel_draws_the_kish_effective_n(self, longitudinal: Any) -> None:
        fig = plotting.weights(longitudinal)
        bottom = fig.axes[1]
        lines = [line for line in bottom.lines if line.get_gid() == "cleverly-effective-n"]
        drawn = [line.get_ydata() for line in lines]
        expected = [node_weights(longitudinal, regimen=r).effective_n for r in ("always", "never")]
        for got, want in zip(drawn, expected, strict=True):
            np.testing.assert_allclose(got, want)
        followed = [line for line in bottom.lines if line.get_gid() == "cleverly-followed"]
        assert len(followed) == 2
        # the effective n of a box's weights never exceeds the units followed
        assert all(
            np.all(eff <= f.get_ydata() + 1e-9) for eff, f in zip(drawn, followed, strict=True)
        )
        _close(fig)

    def test_the_declared_limit_is_marked_with_the_share_beyond_it(self, bounded: Any) -> None:
        fig = plotting.weights(bounded, regimens=["never"])
        top = fig.axes[0]
        marks = [line for line in top.lines if line.get_gid() == "cleverly-marker"]
        assert len(marks) == 2
        for mark in marks:
            assert mark.get_ydata()[0] == pytest.approx(1.0 / 0.3)
        texts = [t.get_text() for t in top.texts if t.get_gid() == "cleverly-tail-share"]
        weights = node_weights(bounded, regimen="never")
        expected = [f"{100 * share:.0f}%" for share in weights.share_beyond_bounds]
        assert texts == expected
        key = [text.get_text() for text in top.get_legend().get_texts()]
        assert key == ["Largest weight the limit allows (3.33)"]
        _close(fig)

    def test_a_tilt_node_has_no_limit_line(self) -> None:
        result = fit_ipsi()
        label = next(iter(result.repeats[0].fits.values())).regimen.label
        fig = plotting.weights(result, regimens=[label])
        top = fig.axes[0]
        kinds = node_weights(result, regimen=label).node_kinds
        assert set(kinds) == {"tilt"}
        assert [line for line in top.lines if line.get_gid() == "cleverly-marker"] == []
        notes = _core.note_text(fig)
        assert "an incremental policy that scales the odds of treatment" in notes
        assert "kind" not in notes and "tilt" not in notes
        _close(fig)

    def test_a_mutated_node_kind_changes_the_limit_lines(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # control for the test above: the same fit read as a static plan draws lines
        fig = plotting.weights(longitudinal, regimens=["always"])
        assert [line for line in fig.axes[0].lines if line.get_gid() == "cleverly-marker"], (
            "a static plan with unit weights is capped by the limit"
        )
        _close(fig)
        monkeypatch.setattr("cleverly.validation.longitudinal._node_kind", lambda fit, time: "tilt")
        fig = plotting.weights(longitudinal, regimens=["always"])
        assert [line for line in fig.axes[0].lines if line.get_gid() == "cleverly-marker"] == []
        _close(fig)

    def test_the_note_names_the_horizon_and_repeat(self, repeated: Any) -> None:
        fig = plotting.weights(repeated)
        assert "Final time point 2, cross-fitting repeat 1 of 2." in _core.note_text(fig)
        _close(fig)
        fig = plotting.weights(repeated, repeat=2)
        assert "cross-fitting repeat 2 of 2" in _core.note_text(fig)
        assert "Descriptive" in _core.note_text(fig)
        _close(fig)

    def test_regimens_get_their_palette_slots_and_a_direct_label(self, longitudinal: Any) -> None:
        fig = plotting.weights(longitudinal)
        bottom = fig.axes[1]
        lines = [line for line in bottom.lines if line.get_gid() == "cleverly-effective-n"]
        assert [line.get_color() for line in lines] == list(_core.PALETTE[:2])
        followed = [line for line in bottom.lines if line.get_gid() == "cleverly-followed"]
        # each regimen's follower count wears its own colour and a dash, so the two kinds of
        # line differ by style and the regimens by colour and by their end labels
        assert [line.get_color() for line in followed] == list(_core.PALETTE[:2])
        assert all(line.get_linestyle() == "--" for line in followed)
        assert all(line.get_linestyle() == "-" for line in lines)
        labels = sorted(t.get_text() for t in bottom.texts if t.get_gid() == _core.GID_SERIES_LABEL)
        assert labels == [
            "always: effective sample size",
            "always: units still followed",
            "never: effective sample size",
            "never: units still followed",
        ]
        # the top panel names the regimen under every box, not by colour alone
        names = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == _core.GID_SERIES_LABEL]
        assert names == ["always", "always", "never", "never"]
        # the key holds no series
        key = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert all(text.startswith("Largest weight") for text in key)
        _close(fig)

    def test_one_regimen_labels_both_lines_and_no_box(self, longitudinal: Any) -> None:
        fig = plotting.weights(longitudinal, regimens=["always"])
        labels = sorted(
            t.get_text() for t in fig.axes[1].texts if t.get_gid() == _core.GID_SERIES_LABEL
        )
        assert labels == ["Effective sample size", "Units still followed"]
        assert [t for t in fig.axes[0].texts if t.get_gid() == _core.GID_SERIES_LABEL] == []
        _close(fig)

    def test_render_smoke(self, longitudinal: Any, tmp_path: Any) -> None:
        fig = plotting.weights(longitudinal)
        fig.savefig(tmp_path / "weights.png", dpi=60)
        assert (tmp_path / "weights.png").stat().st_size > 1000
        _close(fig)


def _point_result(
    frame: Any,
    estimand: Any = None,
    adjustment: tuple[str, ...] = ("W1", "W2", "W3", "W4"),
    **design: Any,
) -> Any:
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=adjustment, **design),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(estimand or ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )


@pytest.fixture(scope="module")
def point() -> Any:
    frame, _ = make_linear_ate(n=300, seed=1)
    return _point_result(frame)


def _hand_smd(x: np.ndarray, in_group: np.ndarray, weight: np.ndarray, spread: float) -> float:
    """A difference of weighted group means over a given spread, written out directly."""
    mean_in = sum(w * v for w, v, g in zip(weight, x, in_group, strict=True) if g) / sum(
        w for w, g in zip(weight, in_group, strict=True) if g
    )
    mean_out = sum(w * v for w, v, g in zip(weight, x, in_group, strict=True) if not g) / sum(
        w for w, g in zip(weight, in_group, strict=True) if not g
    )
    return (mean_in - mean_out) / spread


def _hand_balance(result: Any, *, flip: bool = False) -> dict[str, tuple[float, float]]:
    data = result.data
    propensity = result.repeats[0].nuisance.propensity
    low, high = result.config.g_bounds
    treated_probability = np.clip(propensity.values[:, propensity.column_for(1.0)], low, high)
    own = np.where(data.treatment == 1.0, treated_probability, 1.0 - treated_probability)
    weight = own if flip else 1.0 / own
    in_group = data.treatment == 1.0
    out = {}
    for index, name in enumerate(data.covariate_names):
        x = data.covariates[:, index]
        spread = np.sqrt((np.var(x[in_group], ddof=1) + np.var(x[~in_group], ddof=1)) / 2.0)
        out[name] = (
            _hand_smd(x, in_group, np.ones(len(x)), float(spread)),
            _hand_smd(x, in_group, weight, float(spread)),
        )
    return out


class TestCovariateBalance:
    def test_the_smd_matches_a_hand_computation_raw_and_weighted(self, point: Any) -> None:
        balance = covariate_balance(point)
        hand = _hand_balance(point)
        assert [row.covariate for row in balance.rows] == ["W1", "W2", "W3", "W4"]
        for row in balance.rows:
            raw, weighted = hand[row.covariate]
            assert row.smd_unweighted == pytest.approx(raw, abs=1e-12)
            assert row.smd_weighted == pytest.approx(weighted, abs=1e-12)

    def test_weighting_by_the_propensity_removes_the_imbalance_it_models(self, point: Any) -> None:
        balance = covariate_balance(point)
        before = max(abs(row.smd_unweighted) for row in balance.rows)
        after = max(abs(row.smd_weighted) for row in balance.rows)
        assert before > 0.2 > 0.1 > after

    def test_flipping_the_weights_fails_the_hand_check(self, point: Any) -> None:
        balance = covariate_balance(point)
        flipped = _hand_balance(point, flip=True)
        # the mutation control: with the inverse of the weights, the hand value differs
        # from the computed one on every covariate that is imbalanced
        differs = [
            abs(row.smd_weighted - flipped[row.covariate][1]) > 0.05
            for row in balance.rows
            if abs(row.smd_unweighted) > 0.2
        ]
        assert differs and all(differs)

    def test_the_limits_the_fit_used_are_applied(self) -> None:
        frame, _ = make_linear_ate(n=300, seed=1)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
        )
        result = study.identify(ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
            g_bounds=(0.4, 0.6),
        )
        balance = covariate_balance(result)
        assert balance.limits == (0.4, 0.6)
        raw = result.repeats[0].nuisance.propensity.values[:, 1]
        assert ((raw < 0.4) | (raw > 0.6)).mean() > 0.3  # the limits bind on many units
        hand = _hand_balance(result)
        for row in balance.rows:
            assert row.smd_weighted == pytest.approx(hand[row.covariate][1], abs=1e-12)
        # the control: weights that ignore the limits give other numbers
        data = result.data
        in_group = data.treatment == 1.0
        own = np.where(in_group, raw, 1.0 - raw)
        column = list(data.covariate_names).index("W1")
        x = data.covariates[:, column]
        spread = float(np.sqrt((np.var(x[in_group], ddof=1) + np.var(x[~in_group], ddof=1)) / 2))
        ignoring = _hand_smd(x, in_group, 1.0 / own, spread)
        row = next(r for r in balance.rows if r.covariate == "W1")
        assert abs(row.smd_weighted - ignoring) > 1e-3

    def test_a_weighted_data_set_uses_the_data_weights_in_both_columns(self) -> None:
        frame, _ = make_linear_ate(n=300, seed=1)
        frame["w"] = np.random.default_rng(4).uniform(0.4, 2.5, len(frame))
        result = _point_result(frame, weights="w")
        balance = covariate_balance(result)
        assert balance.weighted_data
        data = result.data
        in_group = data.treatment == 1.0
        weight = data.weights
        for row in balance.rows:
            x = data.covariates[:, list(data.covariate_names).index(row.covariate)]

            def moments(mask: np.ndarray, x: np.ndarray = x) -> tuple[float, float]:
                w, v = weight[mask], x[mask]
                mean = float(np.sum(w * v) / np.sum(w))
                correction = np.sum(w) - np.sum(w**2) / np.sum(w)
                return mean, float(np.sum(w * (v - mean) ** 2) / correction)

            (m1, v1), (m0, v0) = moments(in_group), moments(~in_group)
            assert row.mean_group == pytest.approx(m1)
            assert row.mean_rest == pytest.approx(m0)
            assert row.smd_unweighted == pytest.approx((m1 - m0) / np.sqrt((v1 + v0) / 2))

    def test_three_arms_compare_each_arm_with_the_rest_and_expand_categories(self) -> None:
        frame, _ = make_multi_arm(n=400, seed=2)
        frame["G"] = np.random.default_rng(2).choice(["u", "v", "w"], size=len(frame))
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "G")),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = study.identify(ATE()).estimate(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                random_state=0,
            )
        balance = covariate_balance(result)
        assert len(balance.contrasts) == 3
        assert all(contrast.endswith("vs the rest") for contrast in balance.contrasts)
        names = [row.covariate for row in balance.rows_of(balance.contrasts[0])]
        # every level of G has a row, the left-out reference level among them
        assert sorted(name for name in names if name.startswith("G = ")) == [
            "G = u",
            "G = v",
            "G = w",
        ]
        # hand check of one arm against the rest
        data = result.data
        propensity = result.repeats[0].nuisance.propensity
        values = propensity.truncate(result.config.g_bounds).values
        own = np.empty(len(data.treatment))
        for arm in propensity.arms:
            mask = data.treatment == arm
            own[mask] = values[mask, propensity.column_for(arm)]
        arm = float(propensity.arms[1])
        inside = data.treatment == arm
        column = list(data.covariate_names).index("W1")
        x = data.covariates[:, column]
        # one spread per covariate: the root of the average of all three arms' variances
        spread = np.sqrt(np.mean([np.var(x[data.treatment == a], ddof=1) for a in propensity.arms]))
        expected = _hand_smd(x, inside, 1.0 / own, float(spread))
        row = next(r for r in balance.rows_of(balance.contrasts[1]) if r.covariate == "W1")
        assert row.smd_weighted == pytest.approx(expected, abs=1e-12)
        spreads = {r.spread for r in balance.rows if r.covariate == "W1"}
        assert len(spreads) == 1, "every panel divides a covariate by the same spread"
        # the control: the arm-against-the-rest spread, a mixture's, is another number
        rest = np.sqrt((np.var(x[inside], ddof=1) + np.var(x[~inside], ddof=1)) / 2.0)
        assert abs(float(rest) - row.spread) > 1e-3

    def test_to_frame_and_summary(self, point: Any) -> None:
        balance = covariate_balance(point)
        frame = balance.to_frame()
        assert list(frame["covariate"]) == ["W1", "W2", "W3", "W4"]
        np.testing.assert_allclose(
            frame["smd_weighted"].to_numpy(), [row.smd_weighted for row in balance.rows]
        )
        summary = balance.summary()
        text = str(summary)
        assert "Covariate balance before and after weighting" in text
        assert "before weighting" in text and "after weighting" in text
        assert "3 of 4" in summary.reading or "of 4 covariate differences" in summary.reading
        assert {note.code for note in summary.notes} >= {"balance_descriptive", "balance_weights"}
        assert "Group means and spread" in str(balance.summary(detail="full"))
        assert repr(balance).startswith("CovariateBalance(")
        assert isinstance(balance, CovariateBalance)

    def test_refusals(self) -> None:
        frame, _ = make_linear_ate(n=100, seed=0)
        with pytest.raises(ValueError, match="threshold"):
            covariate_balance(_point_result(frame), threshold=0.0)


class TestTheLovePlot:
    def test_it_draws_the_absolute_smds_sorted_by_the_unweighted_one(self, point: Any) -> None:
        balance = covariate_balance(point)
        fig = balance.plot()
        axis = fig.axes[1]
        before = next(c for c in axis.collections if c.get_gid() == "cleverly-balance-before")
        after = next(c for c in axis.collections if c.get_gid() == "cleverly-balance-after")
        order = sorted(balance.rows, key=lambda r: -abs(r.smd_unweighted))
        np.testing.assert_allclose(
            before.get_offsets()[:, 0], [abs(r.smd_unweighted) for r in order]
        )
        np.testing.assert_allclose(after.get_offsets()[:, 0], [abs(r.smd_weighted) for r in order])
        assert [t.get_text() for t in axis.get_yticklabels()] == [r.covariate for r in order]
        # the top row is the most imbalanced covariate
        heights = before.get_offsets()[:, 1]
        assert heights[0] == heights.max()
        _close(fig)

    def test_the_threshold_line_the_labels_and_the_note(self, point: Any) -> None:
        fig = plotting.balance(point)
        head, axis = fig.axes[0], fig.axes[1]
        lines = [line for line in axis.lines if line.get_gid() == "cleverly-reference"]
        assert [line.get_xdata()[0] for line in lines] == [0.1]
        assert head.get_title(loc="left") == "Covariate balance before and after weighting"
        assert axis.get_xlabel() == "Absolute standardized mean difference"
        key = [text.get_text() for text in head.get_legend().get_texts()]
        assert key == ["Threshold, 0.1"]
        subtitle = _core.subtitle_text(head)
        assert "A = 1 vs A = 0" in subtitle and "largest after weighting" in subtitle
        labels = [t.get_text() for t in axis.texts if t.get_gid() == "cleverly-series-label"]
        assert sorted(labels) == ["After weighting", "Before weighting"]
        notes = _core.note_text(fig)
        assert "no interval or test" in notes
        assert "the square root of the average of the two groups' variances" in notes
        _close(fig)

    def test_markers_differ_by_fill_and_colour(self, point: Any) -> None:
        fig = plotting.balance(point)
        axis = fig.axes[1]
        before = next(c for c in axis.collections if c.get_gid() == "cleverly-balance-before")
        after = next(c for c in axis.collections if c.get_gid() == "cleverly-balance-after")
        assert matplotlib.colors.to_hex(before.get_edgecolor()[0]) == _core.PALETTE[1]
        assert matplotlib.colors.to_hex(after.get_facecolor()[0]) == _core.PALETTE[0]
        assert matplotlib.colors.to_hex(before.get_facecolor()[0]) == _core.FACE
        _close(fig)

    def test_a_plot_without_its_descriptive_note_fails_the_check(
        self, point: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _assert_descriptive(plotting.balance(point))
        original = _core.add_notes

        def dropped(target: Any, lines: Any) -> None:
            original(target, list(lines)[1:])

        # the mutation control: the producer loses its first note, and the check fails
        monkeypatch.setattr(_core, "add_notes", dropped)
        mutated = plotting.balance(point)
        with pytest.raises(AssertionError):
            _assert_descriptive(mutated)

    def test_flipped_weights_change_the_drawn_markers(
        self, point: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        genuine = plotting.balance(point)
        before = next(
            c for c in genuine.axes[1].collections if c.get_gid() == "cleverly-balance-after"
        )
        drawn = before.get_offsets()[:, 0].copy()
        _close(genuine)
        import cleverly.sensitivity.balance as module

        original = module._weighted_moments
        flipped = lambda values, weights: original(values, 1.0 / weights)  # noqa: E731
        monkeypatch.setattr(module, "_weighted_moments", flipped)
        mutated = plotting.balance(point)
        after = next(
            c for c in mutated.axes[1].collections if c.get_gid() == "cleverly-balance-after"
        )
        assert not np.allclose(after.get_offsets()[:, 0], drawn)
        _close(mutated)

    def test_three_arms_draw_one_panel_per_arm_on_shared_rows(self) -> None:
        frame, _ = make_multi_arm(n=300, seed=5)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = study.identify(ATE()).estimate(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                random_state=0,
            )
        fig = plotting.balance(result)
        panels = [
            a
            for a in fig.axes
            if any(c.get_gid() == "cleverly-balance-after" for c in a.collections)
        ]
        assert len(panels) == 3
        titles = [a.get_title(loc="left") for a in panels]
        assert all(title.startswith("A = ") and title.endswith(" vs the rest") for title in titles)
        assert "Each panel compares one arm with all the other arms together" in (
            _core.note_text(fig)
        )
        assert "all 3 arms' variances" in _core.note_text(fig)
        _close(fig)

    def test_treatment_label_names_the_arms_of_one_contrast(self, point: Any) -> None:
        fig = plotting.balance(point, treatment_label="Offer")
        subtitle = _core.subtitle_text(fig.axes[0])
        assert "Offer = 1 vs Offer = 0" in subtitle and "A = " not in subtitle
        _close(fig)

    def test_max_covariates_truncates_and_says_so(self, point: Any) -> None:
        fig = plotting.balance(point, max_covariates=2)
        axis = fig.axes[1]
        after = next(c for c in axis.collections if c.get_gid() == "cleverly-balance-after")
        assert len(after.get_offsets()) == 2
        assert "2 covariates are not drawn" in _core.note_text(fig)
        _close(fig)

    def test_render_smoke(self, point: Any, tmp_path: Any) -> None:
        fig = plotting.balance(point)
        fig.savefig(tmp_path / "balance.png", dpi=60)
        assert (tmp_path / "balance.png").stat().st_size > 1000
        _close(fig)


def test_the_kish_helper_agrees_with_the_accessor(longitudinal: Any) -> None:
    weights = node_weights(longitudinal, regimen="always")
    mask = weights.followed[:, 1]
    assert weights.effective_n[1] == pytest.approx(
        effective_sample_size(weights.used[mask, 1], on_degenerate=0.0)
    )


# ---------------------------------------------------------------------------- review fixes


class TestTheWeightsPlotRefusalsAndNotes:
    def test_an_out_of_range_repeat_is_a_data_error(self, longitudinal: Any) -> None:
        with pytest.raises(DataError, match="repeat must be an integer from 1 to 1"):
            plotting.weights(longitudinal, repeat=5)

    def test_cause_is_passed_to_the_accessor(self, longitudinal: Any) -> None:
        with pytest.raises(DataError, match="no cause 'nope'"):
            plotting.weights(longitudinal, regimens=["always"], cause="nope")

    def test_an_uncrossfitted_fit_says_in_sample_and_names_the_final_time_point(
        self, longitudinal: Any
    ) -> None:
        fig = plotting.weights(longitudinal)
        notes = _core.note_text(fig)
        assert "Final time point 2, in-sample probabilities (the fit was not cross-fitted)." in (
            notes
        )
        assert "repeat" not in notes and "Horizon" not in notes
        _close(fig)

    def test_a_policy_before_a_label_node_removes_the_later_limit_line(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def marks() -> list[float]:
            fig = plotting.weights(longitudinal, regimens=["always"])
            found = [
                float(np.mean(line.get_xdata()))
                for line in fig.axes[0].lines
                if line.get_gid() == _core.GID_MARKER
            ]
            _close(fig)
            return found

        # a policy at time 1 carries its factor into time 2, so neither node is capped
        monkeypatch.setattr(
            "cleverly.validation.longitudinal._node_kind",
            lambda fit, time: "policy" if time == 1 else "label",
        )
        assert marks() == []
        # the control: the policy at time 2 leaves time 1 capped
        monkeypatch.setattr(
            "cleverly.validation.longitudinal._node_kind",
            lambda fit, time: "label" if time == 1 else "policy",
        )
        assert marks() == [pytest.approx(1.0)]

    def test_a_node_without_followers_prints_no_share(
        self, bounded: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            NodeWeights, "share_beyond_bounds", property(lambda self: np.array([0.25, np.nan]))
        )
        fig = plotting.weights(bounded, regimens=["never"])
        texts = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "cleverly-tail-share"]
        assert texts == ["25%"]
        _close(fig)

    def test_the_figures_pass_the_style_and_layout_checks(
        self, longitudinal: Any, bounded: Any
    ) -> None:
        for fig in (plotting.weights(longitudinal), plotting.weights(bounded, regimens=["never"])):
            assert_style(fig)
            assert_text_inside(fig)
            _close(fig)


@pytest.fixture(scope="module")
def att_fit() -> Any:
    frame, _ = make_linear_ate(n=300, seed=1)
    return _point_result(frame, ATT())


def _conditional_hand(result: Any, focal: float) -> dict[str, tuple[float, float, float]]:
    """ATT (focal 1) or ATC (focal 0) balance written out: weights, bounds and spread."""
    data = result.data
    propensity = result.repeats[0].nuisance.propensity
    low, high = result.config.g_bounds_conditional
    g1 = np.clip(propensity.values[:, propensity.column_for(1.0)], low, high)
    treated = data.treatment == 1.0
    focal_probability = g1 if focal == 1.0 else 1.0 - g1
    own = np.where(treated, g1, 1.0 - g1)
    in_focal = data.treatment == focal
    weight = np.where(in_focal, 1.0, focal_probability / own)
    out = {}
    for index, name in enumerate(data.covariate_names):
        x = data.covariates[:, index]
        spread = float(np.std(x[in_focal], ddof=1))
        out[name] = (
            spread,
            _hand_smd(x, treated, np.ones(len(x)), spread),
            _hand_smd(x, treated, weight, spread),
        )
    return out


class TestTheEstimandsWeights:
    def test_an_att_fit_uses_its_own_weights_bounds_and_the_treated_spread(
        self, att_fit: Any
    ) -> None:
        balance = covariate_balance(att_fit)
        assert balance.estimand == "att"
        assert balance.limits == att_fit.config.g_bounds_conditional
        assert balance.limits != att_fit.config.g_bounds
        hand = _conditional_hand(att_fit, focal=1.0)
        for row in balance.rows:
            spread, raw, weighted = hand[row.covariate]
            assert row.spread == pytest.approx(spread, abs=1e-12)
            assert row.smd_unweighted == pytest.approx(raw, abs=1e-12)
            assert row.smd_weighted == pytest.approx(weighted, abs=1e-12)
            # the treated group is the target population, so its mean does not move
            assert row.mean_group_weighted == pytest.approx(row.mean_group, abs=1e-12)
        # the control: the ATE's weights and pooled spread give other numbers on this fit
        ate = _hand_balance(att_fit)
        moved = [
            abs(row.smd_weighted - ate[row.covariate][1]) > 1e-3
            for row in balance.rows
            if abs(row.smd_unweighted) > 0.2
        ]
        assert moved and all(moved)

    def test_an_atc_fit_uses_the_untreated_spread(self) -> None:
        frame, _ = make_linear_ate(n=300, seed=1)
        result = _point_result(frame, ATC())
        balance = covariate_balance(result)
        assert balance.estimand == "atc"
        hand = _conditional_hand(result, focal=0.0)
        treated_hand = _conditional_hand(result, focal=1.0)
        for row in balance.rows:
            spread, raw, weighted = hand[row.covariate]
            assert row.spread == pytest.approx(spread, abs=1e-12)
            assert row.smd_unweighted == pytest.approx(raw, abs=1e-12)
            assert row.smd_weighted == pytest.approx(weighted, abs=1e-12)
            assert row.mean_rest_weighted == pytest.approx(row.mean_rest, abs=1e-12)
            # the control: the treated group's spread is another number
            assert abs(row.spread - treated_hand[row.covariate][0]) > 1e-4

    def test_the_subtitle_and_notes_name_the_estimand(self, att_fit: Any) -> None:
        fig = plotting.balance(att_fit)
        subtitle = _core.subtitle_text(fig.axes[0])
        assert "weights for the effect among the treated;" in subtitle
        assert "(att)" not in subtitle  # the alias is a library word
        notes = _core.note_text(fig)
        assert "standard deviation in the first group, the one the effect is among" in notes
        assert "kept within [0.025, 0.975]" in notes
        _close(fig)
        facts = covariate_balance(att_fit).summary().facts
        assert facts["Weights for"] == "the effect among the treated (att)"

    def test_an_estimand_the_fit_did_not_report_is_refused(self, point: Any) -> None:
        with pytest.raises(DataError, match="reports no att estimate"):
            covariate_balance(point, estimand="att")
        with pytest.raises(ValueError, match="estimand must be"):
            covariate_balance(point, estimand="ato")  # type: ignore[arg-type]

    def test_three_arm_att_compares_each_arm_with_the_reference(self) -> None:
        frame, _ = make_multi_arm(n=400, seed=2)
        result = _point_result(frame, ATT(), adjustment=("W1", "W2", "W3"))
        balance = covariate_balance(result)
        data = result.data
        reference = data.arm_label(result.config.reference_arm)
        assert all(c.endswith(f"vs A = {reference}") for c in balance.contrasts)
        assert len(balance.contrasts) == 2
        # each contrast divides by its own focal arm's spread
        arm = float(sorted(set(data.treatment) - {result.config.reference_arm})[0])
        x = data.covariates[:, 0]
        row = balance.rows_of(balance.contrasts[0])[0]
        assert row.spread == pytest.approx(float(np.std(x[data.treatment == arm], ddof=1)))


class TestTheBalanceNotes:
    def test_in_sample_and_out_of_fold_are_named_from_the_fit(self, point: Any) -> None:
        in_sample = covariate_balance(point)
        assert not in_sample.cross_fitted
        text = "in-sample predictions; the fit was not cross-fitted"
        assert any(text in note.text for note in in_sample.summary().notes)
        fig = plotting.balance(in_sample)
        assert "the fit was not cross-fitted" in _core.note_text(fig)
        assert "out-of-fold" not in _core.note_text(fig)
        _close(fig)
        frame, _ = make_linear_ate(n=300, seed=1)
        frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(int)
        study = CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crossed = study.identify(ATE()).estimate(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=True,
                random_state=0,
            )
        balance = covariate_balance(crossed)
        assert balance.cross_fitted
        fig = plotting.balance(balance)
        assert "fold predictions" in _core.note_text(fig)
        assert "in-sample" not in _core.note_text(fig)
        assert not any("in-sample" in note.text for note in balance.summary().notes)
        assert any("out-of-fold" in note.text for note in balance.summary().notes)
        _close(fig)

    def test_weighted_data_say_data_weight_over_probability(self, point: Any) -> None:
        frame, _ = make_linear_ate(n=300, seed=1)
        frame["w"] = np.random.default_rng(4).uniform(0.4, 2.5, len(frame))
        fig = plotting.balance(_point_result(frame, weights="w"))
        notes = _core.note_text(fig)
        assert "its data weight divided by the probability the fit gave its own arm" in notes
        assert "Before weighting, units count as the data's weights say." in notes
        _close(fig)
        # the control: unweighted data say the inverse and count every unit once
        fig = plotting.balance(point)
        notes = _core.note_text(fig)
        assert "data weight" not in notes
        assert "the inverse of the probability the fit gave its own arm" in notes
        assert "every unit counts once" in notes
        _close(fig)

    def test_a_binary_comparison_has_no_rest_note(self, point: Any) -> None:
        fig = plotting.balance(point)
        assert "other arms together" not in _core.note_text(fig)
        _close(fig)

    def test_a_two_level_string_covariate_gets_one_row(self) -> None:
        frame, _ = make_linear_ate(n=300, seed=1)
        frame["sex"] = np.random.default_rng(1).choice(["f", "m"], size=len(frame))
        frame["G"] = np.random.default_rng(2).choice(["u", "v", "w"], size=len(frame))
        result = _point_result(frame, adjustment=("W1", "W2", "sex", "G"))
        names = [row.covariate for row in covariate_balance(result).rows]
        assert [name for name in names if name.startswith("sex")] == ["sex = m"]
        assert sorted(name for name in names if name.startswith("G")) == [
            "G = u",
            "G = v",
            "G = w",
        ]


def _synthetic_balance() -> CovariateBalance:
    """Ten covariates weighting fixed, and one it made worse from a small start."""
    rows = [
        BalanceRow("A = 1 vs A = 0", f"big{i}", 0, 0, 0, 0, 1.0, 0.5 - 0.02 * i, 0.03)
        for i in range(10)
    ]
    rows.append(BalanceRow("A = 1 vs A = 0", "worse", 0, 0, 0, 0, 1.0, 0.01, 0.3))
    return CovariateBalance(
        rows=tuple(rows),
        treatment="A",
        contrasts=("A = 1 vs A = 0",),
        n=100,
        n_repeats=1,
        limits=(0.05, 0.95),
        weighted_data=False,
    )


class TestMaxCovariates:
    @staticmethod
    def _drawn(balance: CovariateBalance, count: int) -> list[str]:
        fig = plotting.balance(balance, max_covariates=count)
        names = [t.get_text() for t in fig.axes[1].get_yticklabels()]
        _close(fig)
        return names

    def test_a_covariate_weighting_made_worse_is_always_drawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        balance = _synthetic_balance()
        drawn = self._drawn(balance, 3)
        assert "worse" in drawn and len(drawn) == 3
        fig = plotting.balance(balance, max_covariates=3)
        assert "largest after weighting worse 0.300" in _core.subtitle_text(fig.axes[0])
        assert "8 covariates are not drawn" in _core.note_text(fig)
        _close(fig)

        # the mutation control: keeping the largest differences before weighting drops it
        def before_only(summary: Any, count: int) -> tuple[list[str], int]:
            rows = sorted(summary.rows, key=lambda row: -abs(row.smd_unweighted))
            names = [row.covariate for row in rows[:count]]
            return names, len(summary.rows) - len(names)

        monkeypatch.setattr(_weights, "_shown_covariates", before_only)
        assert "worse" not in self._drawn(balance, 3)

    def test_max_covariates_below_one_is_refused(self, point: Any) -> None:
        with pytest.raises(ValueError, match="max_covariates"):
            plotting.balance(point, max_covariates=0)


class TestTheBalanceLayout:
    def test_the_figures_pass_the_style_and_layout_checks(self, point: Any, att_fit: Any) -> None:
        frame, _ = make_multi_arm(n=300, seed=5)
        three = _point_result(frame, adjustment=("W1", "W2", "W3"))
        long_names = {name: f"{name} measured at the first clinic visit" for name in ("W1", "W2")}
        figures = [
            plotting.balance(point),
            plotting.balance(att_fit),
            plotting.balance(three),
            plotting.balance(point, labels=long_names),
        ]
        for fig in figures:
            assert_style(fig)
            assert_text_inside(fig)
            _close(fig)

    def test_long_row_labels_leave_the_panel_its_width(self, point: Any) -> None:
        def width(labels: dict[str, str] | None) -> float:
            fig = plotting.balance(point, labels=labels)
            fig.draw_without_rendering()
            inches = fig.axes[1].get_window_extent().width / fig.dpi
            _close(fig)
            return float(inches)

        # each panel is laid out 3.6 inches wide; long labels widen the figure instead
        assert width({"W1": "W1 measured at the first clinic visit, in years"}) > 3.3


# ---------------------------------------------------------------------------- docs-examples fixes


@pytest.fixture(scope="module")
def four_plans() -> Any:
    frame, _ = make_longitudinal(n=300, seed=0)
    return _ltmle({"always": 1, "never": 0, "first": (1, 0), "second": (0, 1)}, frame)


def _with_zeros(
    monkeypatch: pytest.MonkeyPatch, node: int, count: int, *, unit_weights: bool = True
) -> None:
    """Make ``count`` followers' weights zero at ``node``, and the data's weights non-unit
    when ``unit_weights`` is false."""
    import dataclasses

    from cleverly.longitudinal import weights as module

    original = module.node_weights

    def patched(*args: Any, **kwargs: Any) -> NodeWeights:
        found = original(*args, **kwargs)
        used = found.used.copy()
        rows = np.flatnonzero(found.followed[:, node])[:count]
        used[rows, node] = 0.0
        return dataclasses.replace(
            found, used=used, unit_weights=found.unit_weights and unit_weights
        )

    monkeypatch.setattr(module, "node_weights", patched)


def _kinds(monkeypatch: pytest.MonkeyPatch, *kinds: str) -> None:
    """Read the fit's nodes as ``kinds``, one per time point."""
    monkeypatch.setattr(
        "cleverly.validation.longitudinal._node_kind", lambda fit, time: kinds[time - 1]
    )


class TestZeroWeights:
    def test_followers_with_a_zero_weight_are_counted_in_the_note(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fig = plotting.weights(longitudinal, regimens=["always"])
        # the control: an ordinary fit has no zero weight and says nothing about one
        assert "weight of zero" not in _core.note_text(fig)
        _close(fig)
        _with_zeros(monkeypatch, node=1, count=7)
        followed = int(node_weights(longitudinal, regimen="always").followed[:, 1].sum())
        fig = plotting.weights(longitudinal, regimens=["always"])
        notes = _core.note_text(fig)
        assert f"time point 2: 7 of {followed}" in notes
        assert "a log axis cannot show" in notes
        # a static regimen gives a follower no zero factor, so the note names no cause
        assert "modified treatment policy" not in notes and "data weight" not in notes
        # the boxes describe the other weights, and the lower panel still counts every unit
        crosses = [c for c in fig.axes[0].collections if c.get_gid() == "cleverly-weight-max"]
        assert np.all(crosses[0].get_offsets()[:, 1] > 0.0)
        drawn = [line for line in fig.axes[1].lines if line.get_gid() == "cleverly-followed"]
        assert drawn[0].get_ydata()[1] == followed
        _close(fig)

    def test_several_regimens_name_the_regimen_of_each_count(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _with_zeros(monkeypatch, node=0, count=3)
        fig = plotting.weights(longitudinal)
        notes = _core.note_text(fig)
        assert "always, time point 1: 3 of" in notes and "never, time point 1: 3 of" in notes
        _close(fig)


class TestTheLogAxisTicks:
    def _labels(self, low: float, high: float) -> list[str]:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [low, high])
        ax.set_ylim(low, high)
        _core.log_axis(ax, "y")
        fig.canvas.draw()
        found = [
            tick.label1.get_text()
            for tick in ax.yaxis.get_major_ticks()
            if low <= tick.get_loc() <= high and tick.label1.get_text()
        ]
        plt.close(fig)
        return found

    def test_a_span_of_seven_decades_gets_few_readable_ticks(self) -> None:
        # the up_25 rise of the sequential-pricing page: weights from 5e-06 to 20
        labels = self._labels(2e-6, 43.0)
        assert 4 <= len(labels) <= 8, labels
        assert not any("e-" in label or "e+" in label for label in labels), labels
        assert "1" in labels and "10" in labels and "$10^{-5}$" in labels

    def test_a_span_of_two_decades_keeps_the_one_two_five_ticks(self) -> None:
        # the control: a narrower axis still reads at 1, 2 and 5 times each power of ten
        assert self._labels(0.1, 10.0) == ["0.1", "0.2", "0.5", "1", "2", "5", "10"]

    @pytest.mark.parametrize(("low", "high"), [(0.15, 20.0), (0.3, 40.0), (3.0, 400.0)])
    def test_an_axis_just_over_two_decades_keeps_the_one_two_five_ticks(
        self, low: float, high: float
    ) -> None:
        # the boundary: two or three powers of ten inside the limits are too few to tick at
        # powers alone, so these axes keep 1, 2 and 5 times each power
        labels = self._labels(low, high)
        assert len(labels) >= 5, labels
        assert any(label.startswith(("2", "5")) for label in labels), labels

    def test_four_powers_inside_tick_at_powers_alone(self) -> None:
        assert self._labels(0.9, 1200.0) == ["1", "10", "100", "1000"]

    def test_one_axis_keeps_one_style_outside_the_plain_range(self) -> None:
        labels = self._labels(1e-6, 5e-5)
        assert labels == [
            "$10^{-6}$",
            "$2\\times10^{-6}$",
            "$5\\times10^{-6}$",
            "$10^{-5}$",
            "$2\\times10^{-5}$",
            "$5\\times10^{-5}$",
        ]

    def test_the_tick_text(self) -> None:
        assert _core.log_tick_text(0.001) == "0.001"
        assert _core.log_tick_text(2000.0) == "2000"
        assert _core.log_tick_text(1e-6) == "$10^{-6}$"
        assert _core.log_tick_text(1e5) == "$10^{5}$"
        assert _core.log_tick_text(2e-6) == "$2\\times10^{-6}$"
        assert _core.log_tick_text(20000.0) == "$2\\times10^{4}$"

    def test_a_wide_weights_axis_carries_few_ticks(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataclasses

        from cleverly.longitudinal import weights as module

        original = module.node_weights

        def spread(*args: Any, **kwargs: Any) -> NodeWeights:
            found = original(*args, **kwargs)
            used = found.used.copy()
            rows = np.flatnonzero(found.followed[:, 1])
            used[rows, 1] = np.geomspace(5e-6, 20.0, rows.size)
            return dataclasses.replace(found, used=used)

        monkeypatch.setattr(module, "node_weights", spread)
        fig = plotting.weights(longitudinal, regimens=["always"])
        top = fig.axes[0]
        fig.canvas.draw()
        low, high = top.get_ylim()
        shown = [
            tick.label1.get_text()
            for tick in top.yaxis.get_major_ticks()
            if low <= tick.get_loc() <= high and tick.label1.get_text()
        ]
        assert 4 <= len(shown) <= 8, shown
        assert_text_inside(fig)
        _close(fig)


class TestChosenRegimens:
    def test_four_regimens_need_a_choice(self, four_plans: Any) -> None:
        with pytest.raises(DataError, match="regimens="):
            plotting.weights(four_plans)

    def test_the_chosen_regimens_are_drawn_in_order(self, four_plans: Any) -> None:
        fig = plotting.weights(four_plans, regimens=["first", "never"])
        lines = [line for line in fig.axes[1].lines if line.get_gid() == "cleverly-effective-n"]
        assert len(lines) == 2
        for line, name in zip(lines, ("first", "never"), strict=True):
            np.testing.assert_allclose(
                line.get_ydata(), node_weights(four_plans, regimen=name).effective_n
            )
        # the witness: the two chosen regimens differ, so a swapped order would fail
        assert not np.allclose(lines[0].get_ydata(), lines[1].get_ydata())
        subtitle = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == _core.GID_SUBTITLE]
        assert subtitle and subtitle[0].endswith("regimen first, never")
        _close(fig)

    @pytest.mark.parametrize(
        ("chosen", "message"),
        [
            ("always", "takes a list"),
            ([], "names no regimen"),
            (["always", "always"], "repeats a label"),
            (["always", "nope"], "does not hold"),
            (["always", "never", "first", "second"], "at most 3"),
        ],
    )
    def test_a_bad_choice_is_refused(self, four_plans: Any, chosen: Any, message: str) -> None:
        with pytest.raises(DataError, match=message):
            plotting.weights(four_plans, regimens=chosen)


class TestThePolicyWords:
    def test_a_modified_treatment_policy_is_named_in_plain_words(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cleverly.validation.longitudinal._node_kind", lambda fit, time: "mtp")
        fig = plotting.weights(longitudinal, regimens=["always"])
        notes = _core.note_text(fig)
        assert "a modified treatment policy" in notes
        assert "mtp" not in notes and "kind" not in notes
        _close(fig)


class TestTheCauseOfAZeroWeight:
    """The note names a cause only where a modified treatment policy can give a zero."""

    def test_a_modified_treatment_policy_at_or_before_the_time_point_is_named(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the witness: the policy acts at time point 1 and the zeros sit at time point 2
        _kinds(monkeypatch, "mtp", "label")
        _with_zeros(monkeypatch, node=1, count=4)
        fig = plotting.weights(longitudinal, regimens=["always"])
        notes = _core.note_text(fig)
        assert "time point 2: 4 of" in notes
        assert "A modified treatment policy acts at this or an earlier time point;" in notes
        assert "cannot produce the one the unit received." in notes
        assert "data weight" not in notes
        _close(fig)

    def test_a_policy_after_the_time_point_is_not_a_cause(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the control: the policy acts only at time point 2, after the zeros at time point 1
        _kinds(monkeypatch, "label", "mtp")
        _with_zeros(monkeypatch, node=0, count=4)
        notes = _core.note_text(plotting.weights(longitudinal, regimens=["always"]))
        assert "time point 1: 4 of" in notes
        assert "acts at this or an earlier" not in notes

    def test_non_unit_data_weights_add_a_zero_data_weight(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _kinds(monkeypatch, "mtp", "mtp")
        _with_zeros(monkeypatch, node=1, count=4, unit_weights=False)
        notes = _core.note_text(plotting.weights(longitudinal, regimens=["always"]))
        assert "the one the unit received, and so does a data weight of zero." in notes

    def test_only_the_shifted_entries_are_named_when_some_are(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # zeros at both time points; the policy acts only at time point 2
        _kinds(monkeypatch, "label", "mtp")
        _with_zeros(monkeypatch, node=0, count=2)
        notes_one = _core.note_text(plotting.weights(longitudinal, regimens=["always"]))
        assert "acts at this or an earlier" not in notes_one
        monkeypatch.undo()
        _kinds(monkeypatch, "label", "mtp")
        import dataclasses

        from cleverly.longitudinal import weights as module

        original = module.node_weights

        def both(*args: Any, **kwargs: Any) -> NodeWeights:
            found = original(*args, **kwargs)
            used = found.used.copy()
            for node in (0, 1):
                used[np.flatnonzero(found.followed[:, node])[:2], node] = 0.0
            return dataclasses.replace(found, used=used)

        monkeypatch.setattr(module, "node_weights", both)
        notes = _core.note_text(plotting.weights(longitudinal, regimens=["always"]))
        assert "acts at this or an earlier time point (time point 2);" in notes


class TestThePolicySentence:
    def test_several_kinds_are_listed_in_plain_words(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _kinds(monkeypatch, "policy", "mtp")
        notes = _core.note_text(plotting.weights(longitudinal, regimens=["always"]))
        assert (
            "No limit line is drawn where a policy that gives each treatment a set probability "
            "or a modified treatment policy that shifts each unit's treatment acts at that "
            "time point or an earlier one: the weight carries the policy's factor, which the "
            "limit does not cap."
        ) in notes
