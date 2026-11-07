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

from cleverly import ATE, CausalStudy, PointTreatment, plotting
from cleverly.data.weighting import effective_sample_size
from cleverly.datasets import make_linear_ate, make_longitudinal, make_multi_arm
from cleverly.exceptions import CapabilityError, DataError
from cleverly.longitudinal import LTMLE, NodeWeights, gcomp, node_weights
from cleverly.plotting import _core
from cleverly.sensitivity import CovariateBalance, covariate_balance
from tests.unit._capability_sweep_support import fit_ipsi

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


def _close(fig: Any) -> None:
    plt.close(fig)


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
        assert top.get_title(loc="left") == "Where do the weights grow too large?"
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
        fig = plotting.weights(bounded, regimen="never")
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
        assert any(text.startswith("Largest weight the limit allows (3.33333)") for text in key)
        _close(fig)

    def test_a_tilt_node_has_no_limit_line(self) -> None:
        result = fit_ipsi()
        label = next(iter(result.repeats[0].fits.values())).regimen.label
        fig = plotting.weights(result, regimen=label)
        top = fig.axes[0]
        kinds = node_weights(result, regimen=label).node_kinds
        assert set(kinds) == {"tilt"}
        assert [line for line in top.lines if line.get_gid() == "cleverly-marker"] == []
        assert "tilt" in _core.note_text(fig)
        _close(fig)

    def test_a_mutated_node_kind_changes_the_limit_lines(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # control for the test above: the same fit read as a static plan draws lines
        fig = plotting.weights(longitudinal, regimen="always")
        assert [line for line in fig.axes[0].lines if line.get_gid() == "cleverly-marker"], (
            "a static plan with unit weights is capped by the limit"
        )
        _close(fig)
        monkeypatch.setattr("cleverly.validation.longitudinal._node_kind", lambda fit, time: "tilt")
        fig = plotting.weights(longitudinal, regimen="always")
        assert [line for line in fig.axes[0].lines if line.get_gid() == "cleverly-marker"] == []
        _close(fig)

    def test_the_note_names_the_horizon_and_repeat(self, repeated: Any) -> None:
        fig = plotting.weights(repeated)
        assert "Horizon 2, cross-fitting repeat 1 of 2" in _core.note_text(fig)
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
        labels = [t.get_text() for t in bottom.texts if t.get_gid() == "cleverly-series-label"]
        assert labels == ["always", "never"]
        _close(fig)

    def test_render_smoke(self, longitudinal: Any, tmp_path: Any) -> None:
        fig = plotting.weights(longitudinal)
        fig.savefig(tmp_path / "weights.png", dpi=60)
        assert (tmp_path / "weights.png").stat().st_size > 1000
        _close(fig)


def _point_result(frame: Any, **design: Any) -> Any:
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"), **design
        ),
    )
    return study.identify(ATE()).estimate(
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
        assert all(contrast.endswith("vs the other arms") for contrast in balance.contrasts)
        names = [row.covariate for row in balance.rows_of(balance.contrasts[0])]
        # every level of G has a row, the left-out reference level among them
        assert sorted(name for name in names if name.startswith("G__")) == [
            "G__u",
            "G__v",
            "G__w",
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
        spread = np.sqrt((np.var(x[inside], ddof=1) + np.var(x[~inside], ddof=1)) / 2.0)
        expected = _hand_smd(x, inside, 1.0 / own, float(spread))
        row = next(r for r in balance.rows_of(balance.contrasts[1]) if r.covariate == "W1")
        assert row.smd_weighted == pytest.approx(expected, abs=1e-12)

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
        assert "Descriptive" in notes and "no interval or test" in notes
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

    def test_a_plot_without_its_descriptive_note_would_fail(self, point: Any) -> None:
        fig = plotting.balance(point)
        assert "Descriptive" in _core.note_text(fig)
        for text in fig.axes[-1].texts:
            if text.get_gid() == "cleverly-notes":
                text.set_text(text.get_text().replace("Descriptive", "Computed"))
        assert "Descriptive" not in _core.note_text(fig)
        _close(fig)

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
        assert all(title.startswith("A = ") for title in titles)
        _close(fig)

    def test_max_covariates_truncates_and_says_so(self, point: Any) -> None:
        fig = plotting.balance(point, max_covariates=2)
        axis = fig.axes[1]
        after = next(c for c in axis.collections if c.get_gid() == "cleverly-balance-after")
        assert len(after.get_offsets()) == 2
        assert "2 covariates with the smallest differences" in _core.note_text(fig)
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
