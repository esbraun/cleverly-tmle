"""The ``ipsi`` clever covariate, and the mechanism fluctuation that finishes its targeting.

Two score equations, so two halves to this module.  Both are checked exactly -- the
covariate against the algebra it implements, and the mechanism tilt against a brute-force
grid search over ``epsilon`` -- rather than through a fit that used them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from cleverly.data import CausalData
from cleverly.fluctuation import (
    InitialFit,
    mechanism_covariate,
    needs_mechanism,
    solve_mechanism,
    submodel_for,
)
from cleverly.interventions import Incremental, IPSISet

DELTAS = (Incremental(1.0), Incremental(2.5), Incremental(0.4))


def make_data(n: int = 60, *, seed: int = 0) -> CausalData:
    rng = np.random.default_rng(seed)
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Y": rng.binomial(1, 0.4, n).astype(float),
            "A": rng.binomial(1, 0.5, n).astype(float),
            "W1": rng.normal(size=n),
        }
    )
    return CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W1"])


def make_tilts(data: CausalData, *, seed: int = 1) -> IPSISet:
    rng = np.random.default_rng(seed)
    one = rng.uniform(0.05, 0.95, data.n)
    return IPSISet.evaluate(DELTAS, data, np.column_stack([1.0 - one, one]))


def build(data: CausalData, tilts: IPSISet) -> object:
    # `propensity` is deliberately nonsense: the ipsi builder must not read it.
    return submodel_for(
        "ipsi",
        data.treatment,
        np.full((data.n, 2), 0.5),
        arms=data.arm_codes,
        incremental=tilts.weights,
    )


# ------------------------------------------------------------------ the covariate


def test_the_submodel_has_one_column_per_tilt_and_no_arm_columns() -> None:
    data = make_data()
    submodel = build(data, make_tilts(data))
    assert submodel.observed.shape == (data.n, len(DELTAS))
    assert submodel.names == ("h_ipsi0", "h_ipsi1", "h_ipsi2")
    assert submodel.group == "ipsi"
    # A column targets a tilt, which spreads over the arms rather than naming one.
    assert submodel.arm_columns == {}
    assert set(submodel.arms) == {0.0, 1.0}


def test_the_observed_column_is_the_closed_form_covariate() -> None:
    data = make_data()
    tilts = make_tilts(data)
    submodel = build(data, tilts)
    a = data.treatment
    for index, item in enumerate(DELTAS):
        d = item.delta * tilts.propensity + (1.0 - tilts.propensity)
        expected = (item.delta * a + (1.0 - a)) / d
        np.testing.assert_allclose(submodel.observed[:, index], expected, atol=1e-14, rtol=0)


def test_the_arm_columns_are_delta_over_d_and_one_over_d() -> None:
    data = make_data()
    tilts = make_tilts(data)
    submodel = build(data, tilts)
    for index, item in enumerate(DELTAS):
        d = item.delta * tilts.propensity + (1.0 - tilts.propensity)
        np.testing.assert_allclose(submodel.arms[1.0][:, index], item.delta / d, atol=1e-14, rtol=0)
        np.testing.assert_allclose(submodel.arms[0.0][:, index], 1.0 / d, atol=1e-14, rtol=0)


def test_the_truncated_propensity_cannot_reach_the_covariate() -> None:
    """`build_submodel` hands over a bounded mechanism; on this axis it must do nothing.

    Truncating ``g`` here would move the *estimand*, since ``g`` is inside ``Psi(delta)``.
    The builder ignoring its ``propensity`` argument is what makes that structural rather
    than a convention, so it is checked with two mechanisms that share nothing.
    """
    data = make_data()
    tilts = make_tilts(data)
    loose = submodel_for(
        "ipsi",
        data.treatment,
        np.full((data.n, 2), 0.5),
        arms=data.arm_codes,
        incremental=tilts.weights,
    )
    tight = submodel_for(
        "ipsi",
        data.treatment,
        np.column_stack([np.full(data.n, 0.999), np.full(data.n, 0.001)]),
        arms=data.arm_codes,
        incremental=tilts.weights,
    )
    np.testing.assert_array_equal(loose.observed, tight.observed)
    np.testing.assert_array_equal(loose.arms[1.0], tight.arms[1.0])


def test_the_natural_course_covariate_is_identically_one() -> None:
    data = make_data()
    submodel = build(data, make_tilts(data))
    np.testing.assert_allclose(submodel.observed[:, 0], 1.0, atol=1e-15, rtol=0)


def test_a_missing_covariate_names_the_builder_that_needs_it() -> None:
    data = make_data()
    with pytest.raises(ValueError, match="needs incremental="):
        submodel_for("ipsi", data.treatment, np.full((data.n, 2), 0.5), arms=data.arm_codes)


def test_a_builder_predating_the_keyword_is_told_what_to_add() -> None:
    """The migration message, on the terms `arms` and `regimes` already set."""
    from cleverly.fluctuation import register_submodel
    from cleverly.fluctuation.submodel import SUBMODEL_BUILDERS

    def old_signature(  # type: ignore[no-untyped-def]
        treatment,
        propensity,
        *,
        arms=(0.0, 1.0),
        treated_fraction=None,
        missingness=None,
        intermediate_density=None,
        selection=None,
        regimes=None,
        shifts=None,
        msm=None,
    ):
        raise AssertionError("not reached")

    register_submodel("stale_for_test", old_signature)
    try:
        with pytest.raises(TypeError, match="incremental=None"):
            submodel_for("stale_for_test", np.zeros(3), np.zeros((3, 2)))
    finally:
        del SUBMODEL_BUILDERS["stale_for_test"]


# ------------------------------------------------------------ the mechanism fluctuation


def initial(data: CausalData, *, seed: int = 2) -> InitialFit:
    rng = np.random.default_rng(seed)
    one = rng.uniform(0.2, 0.8, data.n)
    zero = rng.uniform(0.1, 0.5, data.n)
    observed = np.where(data.treatment == 1.0, one, zero)
    return InitialFit(observed, {0.0: zero, 1.0: one})


def test_only_the_ipsi_group_declares_a_mechanism_half() -> None:
    assert needs_mechanism("ipsi")
    assert not needs_mechanism("mean")
    assert not needs_mechanism("regime")


def test_the_mechanism_covariate_is_the_blip_times_the_derivative() -> None:
    data = make_data()
    tilts = make_tilts(data)
    fit = initial(data)
    covariate = mechanism_covariate("ipsi", fit, tilts)
    blip = fit.arms[1.0] - fit.arms[0.0]
    np.testing.assert_allclose(covariate, tilts.derivative * blip[:, None], atol=1e-14, rtol=0)


def test_the_tilt_zeroes_its_own_score() -> None:
    data = make_data()
    tilts = make_tilts(data)
    covariate = mechanism_covariate("ipsi", initial(data), tilts)
    weights = np.ones(data.n)
    result = solve_mechanism(data.treatment, tilts.propensity, covariate, weights)
    assert result.converged
    assert np.max(np.abs(result.score)) < 1e-10
    # ... and it started somewhere else, or the test would pass on a no-op.
    assert np.max(np.abs(result.score_initial)) > 1e-6


def test_the_tilt_matches_a_grid_search_over_epsilon() -> None:
    """One column, so the solution can be found by brute force and compared exactly."""
    data = make_data()
    tilts = IPSISet.evaluate((Incremental(2.5),), data, _mechanism(data))
    covariate = mechanism_covariate("ipsi", initial(data), tilts)
    weights = np.ones(data.n)
    result = solve_mechanism(data.treatment, tilts.propensity, covariate, weights)

    offset = logit(tilts.propensity)
    a = data.treatment

    def score(eps: float) -> float:
        g = expit(offset + eps * covariate[:, 0])
        return float(np.mean(covariate[:, 0] * (a - g)))

    grid = np.linspace(result.epsilon[0] - 0.5, result.epsilon[0] + 0.5, 20001)
    best = grid[np.argmin([abs(score(float(e))) for e in grid])]
    assert result.epsilon[0] == pytest.approx(best, abs=1e-3)


def test_a_zero_covariate_leaves_the_mechanism_alone() -> None:
    data = make_data()
    tilts = make_tilts(data)
    result = solve_mechanism(
        data.treatment, tilts.propensity, np.zeros((data.n, 1)), np.ones(data.n)
    )
    assert result.converged
    np.testing.assert_array_equal(result.epsilon, np.zeros(1))
    np.testing.assert_allclose(result.propensity, tilts.propensity, atol=1e-12, rtol=0)


def test_the_tilted_mechanism_stays_a_probability() -> None:
    data = make_data()
    tilts = make_tilts(data)
    covariate = 50.0 * mechanism_covariate("ipsi", initial(data), tilts)
    result = solve_mechanism(data.treatment, tilts.propensity, covariate, np.ones(data.n))
    assert np.all(result.propensity > 0.0)
    assert np.all(result.propensity < 1.0)


def test_a_boundary_propensity_does_not_put_an_infinity_in_the_solve() -> None:
    """logit(0) is -inf; the guard keeps the offset finite without truncating anything."""
    data = make_data()
    one = np.full(data.n, 0.5)
    one[:3] = 0.0
    one[3:6] = 1.0
    tilts = IPSISet.evaluate((Incremental(2.0),), data, np.column_stack([1.0 - one, one]))
    covariate = mechanism_covariate("ipsi", initial(data), tilts)
    result = solve_mechanism(data.treatment, tilts.propensity, covariate, np.ones(data.n))
    assert np.all(np.isfinite(result.propensity))
    assert np.all(np.isfinite(result.epsilon))


def _mechanism(data: CausalData, *, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    one = rng.uniform(0.05, 0.95, data.n)
    return np.column_stack([1.0 - one, one])
