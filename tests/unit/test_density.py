"""The binned conditional density: exact invariants, not statistical ones.

The claims worth testing here are arithmetic and fail deterministically -- the hazard
product sums to one, a probability divided by the right bin width is a density, an
evaluation outside the support is zero.  How *well* the density recovers a known law is a
question about the learner, and belongs to the oracle tests where an exact density is
available for comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cleverly.exceptions import DataError
from cleverly.learners.crossfit import Folds, make_folds
from cleverly.learners.density import (
    ConditionalDensity,
    _probabilities_from_hazards,
    bin_edges,
    fit_conditional_density,
    warn_if_unresolved,
)


def _glm() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000))


def _sample(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n, 2))
    treatment = 0.8 * covariates[:, 0] + rng.normal(size=n)
    return covariates, treatment


def _fit(n_bins: int = 10, n: int = 600, **kwargs: object) -> ConditionalDensity:
    covariates, treatment = _sample(n)
    density, _ = fit_conditional_density(
        _glm(),
        covariates,
        treatment,
        np.ones(n),
        make_folds(n, 5, random_state=0),
        n_bins=n_bins,
        **kwargs,  # type: ignore[arg-type]
    )
    return density


class TestTheHazardProduct:
    """``P(bin = b) = lambda_b * prod_{j<b} (1 - lambda_j)``, and the rows sum to one."""

    def test_it_matches_the_formula_written_out(self) -> None:
        hazards = np.array([[0.2, 0.5, 0.25], [0.9, 0.1, 0.5]])
        expected = np.array(
            [
                [0.2, 0.8 * 0.5, 0.8 * 0.5 * 0.25, 0.8 * 0.5 * 0.75],
                [0.9, 0.1 * 0.1, 0.1 * 0.9 * 0.5, 0.1 * 0.9 * 0.5],
            ]
        )
        np.testing.assert_allclose(_probabilities_from_hazards(hazards), expected, atol=1e-14)

    def test_the_last_bin_needs_no_hazard(self) -> None:
        # A unit that survived every modelled bin is in the final one with probability
        # one, which is what makes the rows sum to *exactly* one rather than nearly.
        probabilities = _probabilities_from_hazards(np.array([[0.3, 0.4]]))
        assert probabilities.shape == (1, 3)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-15)

    def test_saturated_hazards_raise_no_runtime_warning(self) -> None:
        # The regression test for the log-space product: a cumulative product of clipped
        # hazards underflows, and ``filterwarnings = ["error::RuntimeWarning"]`` turns the
        # subsequent divide into a build failure rather than a nuisance.
        hazards = np.full((3, 200), 1.0 - 1e-12)
        hazards[:, ::2] = 1e-12
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            probabilities = _probabilities_from_hazards(hazards)
        assert np.all(np.isfinite(probabilities))

    def test_a_fitted_density_integrates_to_one(self) -> None:
        np.testing.assert_allclose(_fit().integrated(), 1.0, atol=1e-12)


class TestEvaluation:
    def test_a_probability_becomes_a_density_by_its_bin_width(self) -> None:
        edges = np.array([0.0, 1.0, 3.0, 4.0])
        probabilities = np.array([[0.5, 0.4, 0.1], [0.2, 0.2, 0.6]])
        density = ConditionalDensity(probabilities, edges)
        np.testing.assert_allclose(density.density_at(np.array([0.5, 2.0])), [0.5, 0.1])

    def test_it_is_constant_within_a_bin(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        density = ConditionalDensity(np.array([[0.3, 0.7], [0.3, 0.7]]), edges)
        first = density.density_at(np.array([0.01, 0.01]))
        second = density.density_at(np.array([0.99, 0.99]))
        np.testing.assert_array_equal(first, second)

    def test_outside_the_support_it_is_exactly_zero(self) -> None:
        density = ConditionalDensity(np.array([[0.5, 0.5]]), np.array([0.0, 1.0, 2.0]))
        assert density.density_at(np.array([5.0]))[0] == 0.0
        assert density.density_at(np.array([-5.0]))[0] == 0.0
        assert density.bin_of(np.array([5.0]))[0] == -1

    def test_every_observation_lands_inside_a_bin(self) -> None:
        # Including the minimum and the maximum, which sit on the outer edges before they
        # are padded -- ``digitize`` would otherwise push the maximum past the last bin.
        _, treatment = _sample()
        edges = bin_edges(treatment, 10)
        assert np.all(np.digitize(treatment, edges) - 1 < edges.size - 1)
        assert np.all(np.digitize(treatment, edges) - 1 >= 0)

    def test_one_value_per_row_is_required(self) -> None:
        density = ConditionalDensity(np.ones((4, 2)) / 2, np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError, match="one value per row"):
            density.density_at(np.array([0.5]))


class TestTheObservedAndShiftedDensityShareAModel:
    """``g(A | W)`` and ``g(A - delta | W)`` are two reads of one stored row.

    The invariant a cross-fitted density has to keep is that both come from the *same*
    out-of-fold model.  Storing evaluated bin probabilities makes that structural rather
    than something to maintain: there is no second model to get wrong.
    """

    def test_both_evaluations_read_the_same_row(self) -> None:
        _, treatment = _sample()
        density = _fit()
        row = 7
        observed_bin = density.bin_of(treatment)[row]
        shifted_bin = density.bin_of(treatment - 0.5)[row]
        expected_observed = (
            density.bin_probabilities[row, observed_bin] / density.widths[observed_bin]
        )
        expected_shifted = density.bin_probabilities[row, shifted_bin] / density.widths[shifted_bin]
        assert density.density_at(treatment)[row] == expected_observed
        assert density.density_at(treatment - 0.5)[row] == expected_shifted

    def test_a_subset_slices_rows_and_keeps_the_grid(self) -> None:
        density = _fit()
        subset = density.subset(np.arange(20))
        assert subset.n == 20
        np.testing.assert_array_equal(subset.edges, density.edges)
        np.testing.assert_array_equal(subset.bin_probabilities, density.bin_probabilities[:20])


class TestBinEdges:
    def test_equal_mass_bins_hold_about_the_same_count(self) -> None:
        _, treatment = _sample(n=600)
        edges = bin_edges(treatment, 10)
        counts = np.bincount(np.digitize(treatment, edges) - 1, minlength=10)
        assert counts.max() - counts.min() <= 1

    def test_supplied_edges_are_used_verbatim(self) -> None:
        edges = np.linspace(-6.0, 6.0, 9)
        np.testing.assert_array_equal(_fit(edges=edges).edges, edges)

    @pytest.mark.parametrize(
        ("values", "n_bins", "match"),
        [
            (np.arange(100.0), 1, "at least 2 bins"),
            (np.zeros(100), 10, "too few distinct values"),
        ],
    )
    def test_a_grid_that_cannot_be_formed_is_refused(
        self, values: np.ndarray, n_bins: int, match: str
    ) -> None:
        with pytest.raises(DataError, match=match):
            bin_edges(values, n_bins)

    def test_non_monotone_supplied_edges_are_refused(self) -> None:
        covariates, treatment = _sample(n=100)
        with pytest.raises(DataError, match="strictly increasing"):
            fit_conditional_density(
                _glm(),
                covariates,
                treatment,
                np.ones(100),
                Folds.single(100),
                edges=[0.0, 2.0, 1.0],
            )


class TestTheResolutionGuard:
    """A shift smaller than a bin width moves nobody, and the fit says so."""

    def test_a_shift_that_moves_nobody_is_reported_and_warned_about(self) -> None:
        _, treatment = _sample()
        density = _fit()
        with pytest.warns(UserWarning, match="across a bin edge"):
            crossing = warn_if_unresolved(density, treatment + 1e-9, treatment)
        assert crossing == pytest.approx(0.0, abs=1e-12)

    def test_a_shift_of_a_bin_width_moves_most_rows(self) -> None:
        _, treatment = _sample()
        density = _fit()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            crossing = warn_if_unresolved(density, treatment + 1.5, treatment)
        assert crossing > 0.8


class TestValidation:
    @pytest.mark.parametrize(
        ("probabilities", "edges", "match"),
        [
            (np.ones(4), np.array([0.0, 1.0]), r"must be \(n, B\)"),
            (np.ones((4, 2)), np.array([0.0, 1.0]), "needs 3 edges"),
            (np.ones((4, 2)), np.array([0.0, 1.0, 0.5]), "strictly increasing"),
        ],
    )
    def test_a_malformed_density_is_refused(
        self, probabilities: np.ndarray, edges: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ConditionalDensity(probabilities, edges)
