"""Bounding, link functions and the outcome scaling round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.submodel import SUBMODEL_BUILDERS
from cleverly.utils.bounds import (
    CONDITIONAL_GROUPS,
    OutcomeScaler,
    bound,
    expit,
    g_bounds_for,
    logit,
    resolve_g_bounds,
    shrink_probabilities,
)


class TestLinks:
    def test_logit_expit_round_trip(self) -> None:
        p = np.linspace(1e-6, 1 - 1e-6, 501)
        assert np.allclose(expit(logit(p)), p, atol=1e-12)

    def test_expit_saturates_without_overflow(self) -> None:
        extreme = np.array([-1e6, -800.0, 0.0, 800.0, 1e6])
        values = expit(extreme)
        assert np.all(np.isfinite(values))
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(1.0)
        assert values[2] == pytest.approx(0.5)

    def test_logit_pulls_boundary_values_inside(self) -> None:
        assert np.all(np.isfinite(logit(np.array([0.0, 1.0]))))

    def test_bound_rejects_inverted_interval(self) -> None:
        with pytest.raises(ValueError, match="below lower bound"):
            bound(np.array([0.5]), 0.9, 0.1)


class TestShrinkProbabilities:
    def test_uses_r_tmle_alpha_convention(self) -> None:
        # alpha=0.9995 maps to [0.0005, 0.9995], matching R's tmle package.
        shrunk = shrink_probabilities(np.array([0.0, 0.5, 1.0]), 0.9995)
        assert shrunk[0] == pytest.approx(0.0005)
        assert shrunk[1] == pytest.approx(0.5)
        assert shrunk[2] == pytest.approx(0.9995)

    @pytest.mark.parametrize("alpha", [0.4, 0.5, 1.0, 1.5])
    def test_rejects_alpha_outside_range(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must lie"):
            shrink_probabilities(np.array([0.5]), alpha)


class TestResolveGBounds:
    def test_auto_follows_r_tmle_for_the_ate(self) -> None:
        n = 1000
        expected = 5.0 / (np.sqrt(n) * np.log(n))
        lower, upper = resolve_g_bounds("auto", n, for_att=False)
        assert lower == pytest.approx(expected)
        assert upper == pytest.approx(1.0 - expected)

    def test_auto_is_more_conservative_for_the_att(self) -> None:
        # The ATT reweights controls by the propensity odds, so it needs a tighter
        # bound; R's tmle uses a fixed 0.025.
        ate_lower, _ = resolve_g_bounds("auto", 1000, for_att=False)
        att_lower, _ = resolve_g_bounds("auto", 1000, for_att=True)
        assert att_lower == pytest.approx(0.025)
        assert att_lower > ate_lower

    def test_auto_tightens_as_n_grows(self) -> None:
        small, _ = resolve_g_bounds("auto", 200)
        large, _ = resolve_g_bounds("auto", 20_000)
        assert large < small

    def test_scalar_becomes_a_symmetric_pair(self) -> None:
        assert resolve_g_bounds(0.02, 500) == (0.02, 0.98)

    def test_explicit_pair_passes_through(self) -> None:
        assert resolve_g_bounds((0.01, 0.9), 500) == (0.01, 0.9)

    @pytest.mark.parametrize("spec", [0.0, 0.5, 0.7, -0.1])
    def test_rejects_scalar_outside_the_open_half_interval(self, spec: float) -> None:
        with pytest.raises(ValueError, match="scalar g_bounds"):
            resolve_g_bounds(spec, 500)

    def test_rejects_unordered_pair(self) -> None:
        with pytest.raises(ValueError, match="lower < upper"):
            resolve_g_bounds((0.9, 0.1), 500)

    def test_rejects_unknown_string(self) -> None:
        with pytest.raises(ValueError, match="must be 'auto'"):
            resolve_g_bounds("tight", 500)  # type: ignore[arg-type]


class TestWhichBoundAGroupGets:
    """Only a covariate that is an *odds* needs the tighter bound.

    This used to be spelled ``group == "mean"``, written when ``mean``, ``att`` and
    ``atc`` were the only groups -- so every group added since inherited the ATT bound
    without anything saying so, and so would a group registered through
    ``register_submodel``.  These pin the rule as a statement about the covariate rather
    than about which group happened to be first.
    """

    MEAN = (0.01, 0.99)
    CONDITIONAL = (0.025, 0.975)

    @pytest.mark.parametrize("group", sorted(CONDITIONAL_GROUPS))
    def test_an_odds_covariate_gets_the_tighter_bound(self, group: str) -> None:
        assert g_bounds_for(group, self.MEAN, self.CONDITIONAL) == self.CONDITIONAL

    @pytest.mark.parametrize(
        "group", [*sorted(set(SUBMODEL_BUILDERS) - CONDITIONAL_GROUPS), "a_registered_group"]
    )
    def test_everything_else_divides_by_g_once_and_gets_the_ordinary_bound(
        self, group: str
    ) -> None:
        assert g_bounds_for(group, self.MEAN, self.CONDITIONAL) == self.MEAN

    def test_every_conditional_group_is_binary_only(self) -> None:
        """One direction holds by derivation: an odds needs two arms to be an odds.

        The converse does **not** hold, and it is worth saying why rather than leaving
        the set difference to be rediscovered.  ``mean`` is binary-only only through the
        targets that name particular arms (``rr``, ``or``), and its covariate is an
        inverse probability rather than an odds.  ``ipsi`` is binary-only because an
        *odds multiplier* needs two arms -- but its covariate divides by no mechanism at
        all, so it takes neither bound, and the truncation it does not need is the point
        of the estimand.  "Binary-only" and "needs the tighter bound" are two questions
        that happened to have the same answer while ``att`` and ``atc`` were the only
        binary-only groups with a mechanism in the denominator.
        """
        from cleverly.targets import TARGETS

        binary_only = {t.group for t in TARGETS.values() if t.requires_binary_treatment}
        assert binary_only >= CONDITIONAL_GROUPS
        assert binary_only - CONDITIONAL_GROUPS == {"mean", "ipsi"}


class TestOutcomeScaler:
    def test_scaling_round_trips_a_level(self) -> None:
        y = np.array([-3.0, 0.0, 4.0, 11.5])
        scaler = OutcomeScaler.from_outcome(y)
        scaled = scaler.scale(y)
        assert scaled.min() > 0.0
        assert scaled.max() < 1.0
        for value, original in zip(scaled, y, strict=True):
            assert scaler.unscale_level(float(value)) == pytest.approx(original)

    def test_a_difference_ignores_the_location_shift(self) -> None:
        y = np.array([2.0, 5.0, 9.0])
        scaler = OutcomeScaler.from_outcome(y)
        # A contrast of two levels must equal the unscaled difference: the shift cancels.
        high, low = 0.8, 0.3
        assert scaler.unscale_difference(high - low) == pytest.approx(
            scaler.unscale_level(high) - scaler.unscale_level(low)
        )

    def test_influence_curve_scales_by_range_only(self) -> None:
        scaler = OutcomeScaler(-1.0, 4.0)
        ic = np.array([-0.2, 0.0, 0.35])
        assert np.allclose(scaler.unscale_influence(ic), 5.0 * ic)
        # An influence curve is centred, so a location shift would be wrong here.
        assert np.mean(scaler.unscale_influence(ic - ic.mean())) == pytest.approx(0.0)

    def test_default_padding_widens_the_observed_range(self) -> None:
        y = np.array([0.0, 10.0])
        scaler = OutcomeScaler.from_outcome(y)
        assert scaler.lower == pytest.approx(-1.0)
        assert scaler.upper == pytest.approx(11.0)

    def test_explicit_bounds_are_respected(self) -> None:
        scaler = OutcomeScaler.from_outcome(np.array([1.0, 2.0]), bounds=(0.0, 10.0))
        assert (scaler.lower, scaler.upper) == (0.0, 10.0)

    def test_explicit_bounds_must_contain_the_data(self) -> None:
        with pytest.raises(ValueError, match="outside q_bounds"):
            OutcomeScaler.from_outcome(np.array([1.0, 20.0]), bounds=(0.0, 10.0))

    def test_degenerate_outcome_stays_invertible(self) -> None:
        scaler = OutcomeScaler.from_outcome(np.array([3.0, 3.0, 3.0]))
        assert scaler.range > 0
        assert scaler.unscale_level(float(scaler.scale(np.array([3.0]))[0])) == pytest.approx(3.0)

    def test_identity_scaler_is_flagged(self) -> None:
        assert OutcomeScaler.identity().is_identity
        assert not OutcomeScaler(-1.0, 1.0).is_identity

    def test_rejects_all_missing_outcome(self) -> None:
        with pytest.raises(ValueError, match="no finite outcome"):
            OutcomeScaler.from_outcome(np.array([np.nan, np.nan]))
