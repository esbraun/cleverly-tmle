"""Exact lookups of declared values, and sensitivity curves drawn in x order.

Codex review CDR-004: two declared doses within a relative 1e-5 are different estimands, so a
lookup must not match them by a default tolerance. CDR-006: a sensitivity curve is a function
of its x, so its guide line joins the points in x order, whatever order the grid arrived in.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression

from cleverly.dose_response import DoseResponse, pair_name
from cleverly.estimators.dose_response import DoseResponseTMLE
from cleverly.plotting import _curves, _sensitivity
from cleverly.sensitivity import MissingnessTilt


class TestTheDoseResponseLookupIsExact:
    @pytest.fixture(scope="class")
    def result(self):  # type: ignore[no-untyped-def]
        # The reviewer's probe: doses 100000 and 100000.5 with bandwidth 0.4 on a linear curve.
        rng = np.random.default_rng(7)
        a = np.linspace(99998.0, 100003.0, 1000)
        frame = pd.DataFrame(
            {
                "A": a,
                "W": rng.normal(size=a.size),
                "Y": a - 100000.0 + rng.normal(scale=0.1, size=a.size),
            }
        )
        curve = DoseResponse(
            doses=(100000.0, 100000.5), bandwidths=(0.4,), support=(99999.0, 100002.0)
        )
        return DoseResponseTMLE(
            curve,
            outcome_learner=LinearRegression(),
            treatment_learner=DummyClassifier(strategy="prior"),
            n_folds=2,
            fluctuation="linear",
            q_bounds=(-5, 5),
            n_replicates=100,
            random_state=7,
        ).fit(frame, outcome="Y", treatment="A", covariates=["W"])

    def test_each_declared_dose_returns_its_own_estimate(self, result) -> None:  # type: ignore[no-untyped-def]
        for dose in (100000.0, 100000.5):
            assert result.estimate(0.4, dose) is result[pair_name(0.4, dose)]
        # The two estimands differ by about the dose gap on this linear curve.
        gap = result.estimate(0.4, 100000.5).psi - result.estimate(0.4, 100000.0).psi
        assert gap == pytest.approx(0.5, abs=0.1)

    def test_a_value_with_the_same_declared_name_resolves(self, result) -> None:  # type: ignore[no-untyped-def]
        # 100000.50000000001 is not the declared float, but formats to the declared name.
        nearby = 100000.50000000001
        assert result.estimate(0.4, nearby) is result[pair_name(0.4, 100000.5)]

    def test_an_undeclared_dose_is_refused(self, result) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(KeyError, match="no declared pair"):
            result.estimate(0.4, 100000.25)
        # A relative 1e-9 is inside np.isclose's default tolerance, and outside the name's
        # ten significant digits.
        with pytest.raises(KeyError, match="no declared pair"):
            result.estimate(0.4, 100000.0001)


class TestTheHeldBandValueIsExact:
    @staticmethod
    def band(held_values: list[float]) -> SimpleNamespace:
        x = [0.0, 1.0] * len(held_values)
        z = [value for value in held_values for _ in range(2)]
        return SimpleNamespace(estimate=np.zeros(len(x)), grid={"X": x, "Z": z})

    def test_a_close_but_different_value_is_not_matched(self) -> None:
        band = self.band([100000.0, 100000.5])
        mask = _curves._matching_band_points(band, "X", None, None, {"Z": 100000.5})
        assert mask.tolist() == [False, False, True, True]
        # A default np.isclose would have matched both declared values here.
        assert np.isclose(100000.0, 100000.5)

    def test_a_value_off_by_rounding_names_its_declared_value(self) -> None:
        band = self.band([0.1 + 0.2, 0.5])
        mask = _curves._matching_band_points(band, "X", None, None, {"Z": 0.3})
        assert mask.tolist() == [True, True, False, False]

    def test_an_undeclared_value_matches_nothing(self) -> None:
        band = self.band([1.0, 2.0])
        mask = _curves._matching_band_points(band, "X", None, None, {"Z": 1.00001})
        assert not mask.any()


def _guide(fig) -> tuple[list[float], list[float]]:  # type: ignore[no-untyped-def]
    line = next(a for a in fig.axes[0].lines if a.get_gid() == "cleverly-guide")
    return [float(v) for v in line.get_xdata()], [float(v) for v in line.get_ydata()]


def _points(fig) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    return sorted(
        (float(a.get_xdata()[0]), float(a.get_ydata()[0]))
        for a in fig.axes[0].lines
        if a.get_gid() != "cleverly-guide" and len(a.get_xdata()) == 1
    )


class TestSensitivityCurvesAreDrawnInOrder:
    def test_the_missingness_curve_joins_gamma_in_order(self) -> None:
        # The reviewer's probe: gamma given as (0, 2, 1).
        tilt = MissingnessTilt.build(
            {
                "gamma": [0.0, 2.0, 1.0],
                "estimand": ["ate"] * 3,
                "is_mar": [True, False, False],
                "psi": [0.1, 0.4, 0.2],
                "ci_lower": [0.05, 0.35, 0.15],
                "ci_upper": [0.15, 0.45, 0.25],
            },
            backend="pandas",
            outcome="Y",
            n_missing=100,
            n=500,
            arm_gamma=None,
            level=0.95,
            statuses=(("ate", "influence_curve"),),
            scales=(("ate", "difference"),),
        )
        fig = tilt.plot(estimand="ate")
        assert _guide(fig) == ([0.0, 1.0, 2.0], [0.1, 0.2, 0.4])
        # Each point stays at its own gamma.
        assert (1.0, 0.2) in _points(fig) and (2.0, 0.4) in _points(fig)

    def test_the_truncation_curve_joins_bounds_in_order(self) -> None:
        frame = pd.DataFrame(
            {
                "estimand": ["ate"] * 3,
                "bound": [0.05, 0.01, 0.1],
                "upper_bound": [0.95, 0.99, 0.9],
                "psi": [0.2, 0.1, 0.3],
                "ci_lower": [0.15, 0.05, 0.25],
                "ci_upper": [0.25, 0.15, 0.35],
                "truncated_fraction": [0.02, 0.0, 0.05],
                "is_fitted_bound": [False, True, False],
                "fitted_lower_bound": [0.01] * 3,
                "fitted_upper_bound": [0.99] * 3,
                "scale": ["difference"] * 3,
                "level": [0.95] * 3,
                "support": ["supported"] * 3,
                "support_reason": [""] * 3,
            }
        )
        fig = _sensitivity.truncation(SimpleNamespace(to_frame=lambda: frame))
        assert _guide(fig) == ([0.01, 0.05, 0.1], [0.1, 0.2, 0.3])
        capped = next(a for a in fig.axes[1].lines if a.get_gid() == "cleverly-capped")
        assert [float(v) for v in capped.get_ydata()] == [0.0, 2.0, 5.0]
