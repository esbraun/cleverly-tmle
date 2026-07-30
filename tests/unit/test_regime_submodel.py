"""The ``regime`` clever covariate, checked against arithmetic written out by hand.

The load-bearing claim is the last one: with a static regime, a column of the ``regime``
submodel *is* the corresponding column of ``mean``, entry for entry. That is what makes
the arm-keyed path a special case of this one rather than a parallel implementation of
it, and it is checked with ``array_equal`` rather than ``allclose`` because the two
expressions are meant to be the same arithmetic, not merely the same number.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation import mean_submodel, regime_submodel, submodel_for
from cleverly.fluctuation.submodel import register_submodel


@pytest.fixture
def setup() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 60
    g1 = rng.uniform(0.15, 0.85, n)
    return rng.integers(0, 2, n).astype(float), np.column_stack([1.0 - g1, g1])


def static_density(n: int, k: int, arms: tuple[int, ...]) -> np.ndarray:
    """``(n, K, R)`` densities putting regime ``r`` entirely on ``arms[r]``."""
    values = np.zeros((n, k, len(arms)))
    for r, arm in enumerate(arms):
        values[:, arm, r] = 1.0
    return values


def test_a_static_regime_column_is_the_mean_submodel_column(setup) -> None:
    a, g = setup
    n = a.shape[0]
    mean = mean_submodel(a, g)
    regime = regime_submodel(a, g, regimes=static_density(n, 2, (0, 1)))

    assert np.array_equal(regime.observed[:, 0], mean.observed[:, 0])
    assert np.array_equal(regime.observed[:, 1], mean.observed[:, 1])
    # The counterfactual covariate agrees too: setting everybody to arm 1 makes only the
    # column of the regime that assigns arm 1 non-zero.
    assert np.array_equal(regime.arms[1.0][:, 1], mean.arms[1.0][:, 1])
    assert np.array_equal(regime.arms[1.0][:, 0], np.zeros(n))


def test_the_regime_covariate_is_the_density_ratio(setup) -> None:
    a, g = setup
    n = a.shape[0]
    star = np.column_stack([np.full(n, 0.25), np.full(n, 0.75)])[:, :, None]
    submodel = regime_submodel(a, g, regimes=star)

    observed_g = np.where(a == 1.0, g[:, 1], g[:, 0])
    observed_star = np.where(a == 1.0, 0.75, 0.25)
    assert np.allclose(submodel.observed[:, 0], observed_star / observed_g)
    assert np.allclose(submodel.arms[1.0][:, 0], 0.75 / g[:, 1])
    assert np.allclose(submodel.arms[0.0][:, 0], 0.25 / g[:, 0])


def test_a_regime_equal_to_the_mechanism_has_covariate_one(setup) -> None:
    """``g* = g`` is the observational regime, whose Riesz representer is the constant 1."""
    a, g = setup
    submodel = regime_submodel(a, g, regimes=g[:, :, None])
    assert np.allclose(submodel.observed[:, 0], np.ones(a.shape[0]))


def test_the_missingness_mechanism_joins_the_denominator(setup) -> None:
    a, g = setup
    n = a.shape[0]
    pi = np.column_stack([np.full(n, 0.8), np.full(n, 0.6)])
    star = static_density(n, 2, (1,))
    plain = regime_submodel(a, g, regimes=star)
    with_missing = regime_submodel(a, g, regimes=star, missingness=pi)
    assert np.allclose(with_missing.arms[1.0][:, 0], plain.arms[1.0][:, 0] / 0.6)


def test_the_selection_indicator_multiplies_only_the_observed_covariate(setup) -> None:
    a, g = setup
    n = a.shape[0]
    keep = np.zeros(n)
    keep[::2] = 1.0
    star = static_density(n, 2, (1,))
    submodel = regime_submodel(a, g, regimes=star, selection=keep)
    plain = regime_submodel(a, g, regimes=star)
    assert np.array_equal(submodel.observed[:, 0], plain.observed[:, 0] * keep)
    assert np.array_equal(submodel.arms[1.0], plain.arms[1.0])


def test_no_column_belongs_to_an_arm(setup) -> None:
    a, g = setup
    submodel = regime_submodel(a, g, regimes=static_density(a.shape[0], 2, (0, 1)))
    assert submodel.arm_columns == {}
    with pytest.raises(KeyError, match="no column dedicated to arm"):
        submodel.column_for(1.0)


def test_the_shape_of_the_submodel_follows_the_regime_count(setup) -> None:
    a, g = setup
    n = a.shape[0]
    submodel = regime_submodel(a, g, regimes=static_density(n, 2, (0, 1, 1)))
    assert submodel.observed.shape == (n, 3)
    assert submodel.dim == 3
    assert submodel.names == ("h_regime0", "h_regime1", "h_regime2")
    assert set(submodel.arms) == {0.0, 1.0}
    assert submodel.arms[0.0].shape == (n, 3)


def test_regimes_are_required_and_validated(setup) -> None:
    a, g = setup
    n = a.shape[0]
    with pytest.raises(ValueError, match="needs regimes="):
        regime_submodel(a, g)
    with pytest.raises(ValueError, match=r"shape \(60, 2, R\)"):
        regime_submodel(a, g, regimes=np.zeros((n, 3, 1)))
    with pytest.raises(ValueError, match="at least one regime"):
        regime_submodel(a, g, regimes=np.zeros((n, 2, 0)))
    with pytest.raises(ValueError, match="non-negative"):
        regime_submodel(a, g, regimes=np.full((n, 2, 1), -0.5))


def test_it_dispatches_through_the_registry(setup) -> None:
    a, g = setup
    n = a.shape[0]
    star = static_density(n, 2, (0, 1))
    direct = regime_submodel(a, g, regimes=star)
    viaregistry = submodel_for("regime", a, g, regimes=star)
    assert np.array_equal(direct.observed, viaregistry.observed)
    assert viaregistry.group == "regime"


def test_a_builder_written_before_regimes_is_told_what_to_add(setup) -> None:
    a, g = setup

    def outdated(
        treatment,
        propensity,
        *,
        arms=(0.0, 1.0),
        treated_fraction=None,
        missingness=None,
        intermediate_density=None,
        selection=None,
    ):
        raise AssertionError("should not be reached")

    register_submodel("outdated_for_test", outdated)
    try:
        with pytest.raises(TypeError, match="does not accept 'regimes'"):
            submodel_for("outdated_for_test", a, g)
    finally:
        from cleverly.fluctuation.submodel import SUBMODEL_BUILDERS

        del SUBMODEL_BUILDERS["outdated_for_test"]
