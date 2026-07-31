"""The incremental propensity-score intervention, before any of it reaches an estimator.

Everything here is exact.  A tilt is a closed-form function of the mechanism, so its
normalisation, its clever covariate and the derivative that the influence curve needs are
checked against the algebra directly rather than inferred from an estimate that used them.

The two claims worth stating separately, because they are the reasons this axis exists:
the tilt at ``delta=1`` *is* the mechanism, and the clever covariate is bounded by ``delta``
however extreme the mechanism gets.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import DataError
from cleverly.interventions import Incremental, IPSISet, check_incremental_support


def make_data(n: int = 40, *, levels: tuple = (0, 1), seed: int = 0) -> CausalData:
    rng = np.random.default_rng(seed)
    k = len(levels)
    frame = {
        "Y": rng.binomial(1, 0.4, n).astype(float),
        "A": np.asarray([levels[i % k] for i in range(n)]),
        "W1": rng.normal(size=n),
    }
    import pandas as pd

    return CausalData.from_frame(pd.DataFrame(frame), outcome="Y", treatment="A", covariates=["W1"])


def mechanism(n: int = 40, *, seed: int = 1) -> np.ndarray:
    """An ``(n, 2)`` mechanism whose columns are in arm-code order and sum to one."""
    rng = np.random.default_rng(seed)
    one = rng.uniform(0.05, 0.95, n)
    return np.column_stack([1.0 - one, one])


# ------------------------------------------------------------------ the declaration


def test_a_tilt_names_itself_by_its_multiplier() -> None:
    assert Incremental(2.5).name == "odds x2.5"
    assert Incremental(0.5).name == "odds x0.5"


def test_the_unit_tilt_is_the_natural_course() -> None:
    # delta=1 leaves the odds alone, so it is the reference the others are read against --
    # the analogue of Shift(0.0, cap=None), and named the same way.
    assert Incremental(1.0).name == "natural course"


def test_a_nonpositive_multiplier_is_refused_by_name() -> None:
    with pytest.raises(DataError, match="strictly positive"):
        Incremental(0.0)
    with pytest.raises(DataError, match="strictly positive"):
        Incremental(-1.0)


def test_an_infinite_multiplier_is_refused() -> None:
    with pytest.raises(DataError, match="finite"):
        Incremental(float("inf"))


# ------------------------------------------------------------------ the arithmetic


def test_the_tilted_density_is_a_density() -> None:
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(2.5), Incremental(0.4)), data, g)
    assert tilts.values.shape == (data.n, 2, 2)
    np.testing.assert_allclose(tilts.values.sum(axis=1), 1.0, atol=1e-15, rtol=0)
    assert np.all(tilts.values >= 0.0)


def test_the_tilt_multiplies_the_odds() -> None:
    """The defining property, stated as odds rather than as the formula that implements it."""
    data, g = make_data(), mechanism()
    delta = 2.5
    tilts = IPSISet.evaluate((Incremental(delta),), data, g)
    one = g[:, 1]
    tilted = tilts.values[:, 1, 0]
    np.testing.assert_allclose(
        tilted / (1.0 - tilted), delta * one / (1.0 - one), atol=1e-12, rtol=0
    )


def test_the_natural_course_reproduces_the_mechanism_exactly() -> None:
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(1.0),), data, g)
    np.testing.assert_allclose(tilts.values[:, :, 0], g, atol=1e-15, rtol=0)
    # ... and its clever covariate is identically one, which is what makes its mean E[Y].
    np.testing.assert_allclose(tilts.weights[:, :, 0], 1.0, atol=1e-15, rtol=0)


def test_the_clever_covariate_is_the_density_ratio() -> None:
    """``h_a = q_a / g_a``, checked against the ratio the code deliberately does not form."""
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(2.5), Incremental(0.4)), data, g)
    np.testing.assert_allclose(tilts.weights, tilts.values / g[:, :, None], atol=1e-12, rtol=0)


def test_the_derivative_is_the_pathwise_one() -> None:
    """``d/dg [ q_1 ] `` folded to what the influence curve multiplies the blip by."""
    data = make_data()
    g = mechanism()
    delta = 2.5
    tilts = IPSISet.evaluate((Incremental(delta),), data, g)
    one = g[:, 1]
    d = delta * one + (1.0 - one)
    np.testing.assert_allclose(tilts.blip_weight(0.0), delta / d**2, atol=1e-14, rtol=0)


def test_the_observed_covariate_reads_the_arm_the_unit_got() -> None:
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(2.5),), data, g)
    observed = tilts.observed(data.treatment)
    expected = np.where(data.treatment == 1.0, tilts.weights[:, 1, 0], tilts.weights[:, 0, 0])
    np.testing.assert_allclose(observed[:, 0], expected, atol=1e-15, rtol=0)


# --------------------------------------------------------- the bound, which is the point


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_the_covariate_is_bounded_by_delta_however_extreme_the_mechanism(delta: float) -> None:
    """No positivity assumption: the bound holds at mechanisms an arm fit could not survive.

    ``g`` runs to 1e-9 here.  An arm-indexed clever covariate would be 1e9; this one stays
    inside ``[min(delta, 1/delta), max(delta, 1/delta)]``, and that is the whole reason the
    axis exists.
    """
    one = np.array(
        [1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-3, 1.0 - 1e-6, 1.0 - 1e-9, 0.25]
    )
    data = make_data(n=one.size)
    g = np.column_stack([1.0 - one, one])
    tilts = IPSISet.evaluate((Incremental(delta),), data, g)
    assert np.all(np.isfinite(tilts.weights))
    assert tilts.weights.min() >= min(delta, 1.0 / delta) - 1e-12
    assert tilts.weights.max() <= max(delta, 1.0 / delta) + 1e-12
    # The arm fit's covariate on the same mechanism, for contrast: nine orders larger.
    assert (1.0 / one).max() > 1e8


# ------------------------------------------------------------------ the carrier


def test_recomputing_at_a_new_mechanism_keeps_the_declaration() -> None:
    """What the mechanism fluctuation calls: the tilt is unchanged, its evaluation moves."""
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(2.5), Incremental(0.4)), data, g, reference="odds x0.4")
    moved = tilts.at(np.clip(g[:, 1] + 0.02, 1e-6, 1 - 1e-6))
    assert moved.names == tilts.names
    assert moved.deltas == tilts.deltas
    assert moved.reference == tilts.reference
    assert not np.allclose(moved.values, tilts.values)
    # `at` and `evaluate` must agree, or the alternating loop would chase two definitions.
    again = IPSISet.evaluate(
        (Incremental(2.5), Incremental(0.4)),
        data,
        np.column_stack([1.0 - moved.propensity, moved.propensity]),
        reference="odds x0.4",
    )
    np.testing.assert_array_equal(again.values, moved.values)
    np.testing.assert_array_equal(again.derivative, moved.derivative)


def test_a_subset_slices_rather_than_re_evaluates() -> None:
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(2.5),), data, g)
    index = np.arange(0, data.n, 2)
    part = tilts.subset(index)
    assert part.n == index.size
    np.testing.assert_array_equal(part.values, tilts.values[index])
    np.testing.assert_array_equal(part.propensity, tilts.propensity[index])
    assert part.names == tilts.names


def test_codes_and_labels_follow_the_declaration_order() -> None:
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(1.0), Incremental(2.0)), data, g)
    assert tilts.codes == (0.0, 1.0)
    assert tilts.labels == {0.0: "natural course", 1.0: "odds x2"}
    assert tilts.label(1.0) == "odds x2"


# ------------------------------------------------------------------ the refusals


def test_a_multi_arm_treatment_is_refused_with_the_derivation_it_would_need() -> None:
    data = make_data(levels=("low", "mid", "high"))
    g = np.full((data.n, 3), 1 / 3)
    with pytest.raises(DataError, match="odds"):
        IPSISet.evaluate((Incremental(2.0),), data, g)


def test_no_interventions_is_refused() -> None:
    with pytest.raises(DataError, match="at least one"):
        IPSISet.evaluate((), make_data(), mechanism())


def test_duplicate_names_are_refused() -> None:
    data, g = make_data(), mechanism()
    with pytest.raises(DataError, match="distinct"):
        IPSISet.evaluate((Incremental(2.0, name="x"), Incremental(3.0, name="x")), data, g)


def test_an_unknown_reference_names_the_ones_there_are() -> None:
    data, g = make_data(), mechanism()
    with pytest.raises(DataError, match="natural course"):
        IPSISet.evaluate((Incremental(1.0),), data, g, reference="nope")


def test_a_mechanism_of_the_wrong_width_is_refused() -> None:
    data = make_data()
    with pytest.raises(DataError, match="arm-code order"):
        IPSISet.evaluate((Incremental(2.0),), data, np.full((data.n, 3), 1 / 3))


# ------------------------------------------------------------------ the support report


def test_the_support_report_states_the_bound_and_the_realised_maximum() -> None:
    data, g = make_data(), mechanism()
    tilts = IPSISet.evaluate((Incremental(1.0), Incremental(3.0)), data, g)
    reports = check_incremental_support(tilts, data.treatment)
    assert set(reports) == {"natural course", "odds x3"}

    natural = reports["natural course"]
    # The natural course weights every unit equally, so it loses no effective sample size
    # at all -- the baseline the other tilts are read against.
    assert natural.guaranteed == (1.0, 1.0)
    assert natural.effective_sample_size == pytest.approx(data.n, abs=1e-9)

    tilted = reports["odds x3"]
    assert tilted.guaranteed == (pytest.approx(1 / 3), 3.0)
    assert tilted.max_ratio <= 3.0 + 1e-12
    assert "by construction" in tilted.summary()
