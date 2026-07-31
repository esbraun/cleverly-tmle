"""The ``msm`` clever covariate, checked against arithmetic written out by hand.

The load-bearing claim is the first one: with a **saturated** working model -- one
indicator column per arm and uniform weights -- the ``msm`` submodel *is* ``mean``, entry
for entry.  A working model that summarises the arms with one column per arm is not
summarising anything, so the projection has to reduce to the counterfactual means
themselves; if it does not, the covariate is wrong by a factor that a smoother design
would hide.  It is checked with ``array_equal`` rather than ``allclose`` because the two
expressions are meant to be the same arithmetic, not merely the same number.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation import mean_submodel, msm_submodel, submodel_for
from cleverly.fluctuation.submodel import register_submodel


@pytest.fixture
def setup() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 60
    g1 = rng.uniform(0.15, 0.85, n)
    return rng.integers(0, 2, n).astype(float), np.column_stack([1.0 - g1, g1])


def saturated(n: int, k: int) -> np.ndarray:
    """``(n, K, K)`` weighted design: one indicator term per arm, weights all one."""
    return np.tile(np.eye(k), (n, 1, 1))


def dose_design(n: int, arms: tuple[float, ...]) -> np.ndarray:
    """``(n, K, 2)`` weighted design for ``m(a) = beta0 + beta1 a``, uniform weights."""
    return np.stack([np.column_stack([np.ones(n), np.full(n, float(arm))]) for arm in arms], axis=1)


# ------------------------------------------------------------- the special case


def test_a_saturated_working_model_is_the_mean_submodel(setup) -> None:
    a, g = setup
    n = a.shape[0]
    mean = mean_submodel(a, g)
    msm = msm_submodel(a, g, msm=saturated(n, 2))

    assert np.array_equal(msm.observed, mean.observed)
    for arm in (0.0, 1.0):
        assert np.array_equal(msm.arms[arm], mean.arms[arm])


def test_a_saturated_working_model_is_the_mean_submodel_on_three_arms() -> None:
    """Two arms cannot tell a per-arm design from two columns that happen to be right."""
    rng = np.random.default_rng(11)
    n = 45
    a = rng.integers(0, 3, n).astype(float)
    raw = rng.uniform(0.2, 0.8, (n, 3))
    g = raw / raw.sum(axis=1, keepdims=True)
    arms = (0.0, 1.0, 2.0)

    mean = mean_submodel(a, g, arms=arms)
    msm = msm_submodel(a, g, arms=arms, msm=saturated(n, 3))

    assert np.array_equal(msm.observed, mean.observed)
    for arm in arms:
        assert np.array_equal(msm.arms[arm], mean.arms[arm])


# --------------------------------------------------------------- the covariate


def test_the_covariate_is_the_weighted_design_over_the_mechanism(setup) -> None:
    a, g = setup
    n = a.shape[0]
    submodel = msm_submodel(a, g, msm=dose_design(n, (0.0, 1.0)))

    observed_g = np.where(a == 1.0, g[:, 1], g[:, 0])
    assert np.allclose(submodel.observed[:, 0], 1.0 / observed_g)  # the intercept term
    assert np.allclose(submodel.observed[:, 1], a / observed_g)  # the dose term
    assert np.allclose(submodel.arms[1.0][:, 1], 1.0 / g[:, 1])
    # Arm 0 contributes nothing to the slope's score, because phi's dose column is zero
    # there -- not because the arm is skipped.
    assert np.allclose(submodel.arms[0.0][:, 1], np.zeros(n))
    assert np.allclose(submodel.arms[0.0][:, 0], 1.0 / g[:, 0])


def test_the_weights_scale_the_covariate(setup) -> None:
    a, g = setup
    n = a.shape[0]
    plain = msm_submodel(a, g, msm=dose_design(n, (0.0, 1.0)))
    doubled = msm_submodel(a, g, msm=2.0 * dose_design(n, (0.0, 1.0)))
    assert np.allclose(doubled.observed, 2.0 * plain.observed)


def test_the_missingness_mechanism_joins_the_denominator(setup) -> None:
    a, g = setup
    n = a.shape[0]
    pi = np.column_stack([np.full(n, 0.8), np.full(n, 0.6)])
    plain = msm_submodel(a, g, msm=dose_design(n, (0.0, 1.0)))
    with_pi = msm_submodel(a, g, msm=dose_design(n, (0.0, 1.0)), missingness=pi)
    observed_pi = np.where(a == 1.0, 0.6, 0.8)
    assert np.allclose(with_pi.observed, plain.observed / observed_pi[:, None])


def test_the_selection_indicator_zeroes_unselected_rows(setup) -> None:
    a, g = setup
    n = a.shape[0]
    keep = np.zeros(n)
    keep[::2] = 1.0
    submodel = msm_submodel(a, g, msm=dose_design(n, (0.0, 1.0)), selection=keep)
    assert np.all(submodel.observed[1::2] == 0.0)
    # The counterfactual columns are already evaluated at the targeted level, so the
    # indicator does not touch them -- the same convention mean_submodel follows.
    assert np.all(submodel.arms[1.0][1::2] != 0.0)


# ------------------------------------------------------------------- structure


def test_no_column_belongs_to_one_arm(setup) -> None:
    a, g = setup
    submodel = msm_submodel(a, g, msm=dose_design(a.shape[0], (0.0, 1.0)))
    assert submodel.arm_columns == {}
    with pytest.raises(KeyError, match="no column dedicated to arm"):
        submodel.column_for(1.0)


def test_it_labels_itself_with_its_group_and_names_its_columns(setup) -> None:
    a, g = setup
    submodel = msm_submodel(a, g, msm=dose_design(a.shape[0], (0.0, 1.0)))
    assert submodel.group == "msm"
    assert submodel.names == ("h_msm0", "h_msm1")
    assert submodel.dim == 2


# -------------------------------------------------------------------- refusals


def test_a_missing_working_model_says_how_to_build_one(setup) -> None:
    a, g = setup
    with pytest.raises(ValueError, match=r"MSMSet\.evaluate"):
        msm_submodel(a, g)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.zeros((60, 2)), "rows, arms, terms"),
        (np.zeros((60, 3, 2)), "rows, arms, terms"),
        (np.zeros((60, 2, 0)), "at least one term"),
    ],
)
def test_a_design_of_the_wrong_shape_is_refused(setup, bad, message) -> None:
    a, g = setup
    with pytest.raises(ValueError, match=message):
        msm_submodel(a, g, msm=bad)


def test_a_non_finite_design_is_refused(setup) -> None:
    a, g = setup
    bad = dose_design(a.shape[0], (0.0, 1.0))
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        msm_submodel(a, g, msm=bad)


# -------------------------------------------------------------------- registry


def test_it_is_reachable_through_the_registry(setup) -> None:
    a, g = setup
    n = a.shape[0]
    direct = msm_submodel(a, g, msm=dose_design(n, (0.0, 1.0)))
    dispatched = submodel_for("msm", a, g, msm=dose_design(n, (0.0, 1.0)))
    assert np.array_equal(direct.observed, dispatched.observed)


def test_a_builder_predating_the_msm_keyword_is_told_what_to_add() -> None:
    """The fix-it message, not a bare ``unexpected keyword argument`` from the dispatcher."""

    def outdated(
        treatment,
        propensity,
        *,
        arms=(0.0, 1.0),
        arm_fractions=None,
        reference=None,
        missingness=None,
        intermediate_density=None,
        selection=None,
        regimes=None,
        shifts=None,
    ):
        raise AssertionError("should not be reached")  # pragma: no cover

    register_submodel("outdated_for_msm_test", outdated)
    try:
        with pytest.raises(TypeError, match="does not accept 'msm'"):
            submodel_for("outdated_for_msm_test", np.zeros(3), np.full((3, 2), 0.5))
    finally:
        from cleverly.fluctuation.submodel import SUBMODEL_BUILDERS

        del SUBMODEL_BUILDERS["outdated_for_msm_test"]
