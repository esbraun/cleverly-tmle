"""Unit-level checks of the sensitivity formulas.

The E-value and the omitted-variable-bias parameterisation both have published
reference values, so they are checked against those rather than against the
implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.sensitivity import evalue_from_rr
from cleverly.sensitivity.evalue import _evalue_for_limit
from cleverly.sensitivity.omitted_variable import _confounding_strength


class TestEValue:
    @pytest.mark.parametrize(
        "risk_ratio,expected",
        [
            (1.0, 1.0),
            (2.0, 3.4142135624),  # 2 + sqrt(2 * 1)
            (3.0, 5.4494897428),  # 3 + sqrt(3 * 2)
            (1.5, 2.3660254038),  # 1.5 + sqrt(1.5 * 0.5)
        ],
    )
    def test_matches_the_published_formula(self, risk_ratio: float, expected: float) -> None:
        assert evalue_from_rr(risk_ratio) == pytest.approx(expected, rel=1e-8)

    def test_is_symmetric_under_inversion(self) -> None:
        # A protective effect of 1/RR needs a confounder just as strong as a harmful
        # effect of RR, so the E-value must be invariant.
        for rr in (1.2, 2.0, 5.0):
            assert evalue_from_rr(rr) == pytest.approx(evalue_from_rr(1.0 / rr))

    def test_increases_with_effect_size(self) -> None:
        values = [evalue_from_rr(rr) for rr in (1.0, 1.5, 2.0, 4.0, 8.0)]
        assert values == sorted(values)

    def test_a_null_risk_ratio_needs_no_confounding(self) -> None:
        assert evalue_from_rr(1.0) == 1.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_input_returns_nan(self, bad: float) -> None:
        value = evalue_from_rr(bad)
        assert np.isnan(value) or not np.isfinite(value)

    def test_a_limit_crossing_the_null_gives_one(self) -> None:
        # If the interval already includes 1, no unmeasured confounding is needed.
        assert _evalue_for_limit(0.9, above_null=True) == 1.0
        assert _evalue_for_limit(1.1, above_null=False) == 1.0

    def test_a_limit_away_from_the_null_uses_the_formula(self) -> None:
        assert _evalue_for_limit(2.0, above_null=True) == pytest.approx(evalue_from_rr(2.0))


class TestConfoundingStrength:
    def test_matches_the_doubleml_parameterisation(self) -> None:
        # |rho| * sqrt(cf_y * cf_d / (1 - cf_d))
        cf_y, cf_d, rho = 0.04, 0.09, 0.8
        expected = 0.8 * np.sqrt(0.04 * 0.09 / 0.91)
        assert _confounding_strength(cf_y, cf_d, rho) == pytest.approx(expected)

    def test_zero_confounding_gives_zero_bias(self) -> None:
        assert _confounding_strength(0.0, 0.0, 1.0) == 0.0
        assert _confounding_strength(0.5, 0.0, 1.0) == 0.0
        assert _confounding_strength(0.0, 0.5, 1.0) == 0.0

    def test_grows_in_every_argument(self) -> None:
        base = _confounding_strength(0.03, 0.03, 1.0)
        assert _confounding_strength(0.06, 0.03, 1.0) > base
        assert _confounding_strength(0.03, 0.06, 1.0) > base
        assert _confounding_strength(0.03, 0.03, 0.5) < base

    def test_diverges_as_cf_d_approaches_one(self) -> None:
        # A confounder that fully determines treatment can produce unbounded bias: the
        # cf_d factor is sqrt(cf_d / (1 - cf_d)), so going from 0.5 to 0.99 multiplies
        # the bound by sqrt(99).
        near_one = _confounding_strength(0.03, 0.99, 1.0)
        half = _confounding_strength(0.03, 0.5, 1.0)
        assert near_one / half == pytest.approx(np.sqrt(99.0), rel=1e-9)
        assert _confounding_strength(0.03, 0.9999, 1.0) > 30.0 * half

    def test_rho_sign_does_not_matter(self) -> None:
        assert _confounding_strength(0.03, 0.03, -1.0) == pytest.approx(
            _confounding_strength(0.03, 0.03, 1.0)
        )

    @pytest.mark.parametrize(
        "cf_y,cf_d,rho", [(-0.1, 0.03, 1.0), (0.03, 1.0, 1.0), (0.03, 0.03, 1.5)]
    )
    def test_out_of_range_parameters_are_refused(
        self, cf_y: float, cf_d: float, rho: float
    ) -> None:
        with pytest.raises(ValueError):
            _confounding_strength(cf_y, cf_d, rho)
