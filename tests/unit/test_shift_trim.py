"""The shift density ratio is trimmed at targeting time, at a quantile of its own values.

``ShiftSet.ratio`` stays untruncated, so the support report reads what the data support.
The covariate that targeting divides into the score is :attr:`ShiftSet.design`, and that
array carries the trim.  The rule is R ``lmtp``'s ``.trim``: every ratio above the
``shift_trim`` quantile of the ratios at the observed dose becomes that quantile
(``lmtp_control(.trim = 0.999)``; ``trim()`` in ``R/utils.R`` and
``cf_density_ratios()`` in ``R/density_ratios.R``, lmtp 1.5.4).

The witness fit is cross-fitted on a continuous dose, where the ratios are distinct and the
quantile therefore falls below the largest one.  The control is a ratio column whose
largest value is shared by more than one row in a thousand, where the quantile *is* the
largest value and the trim must change nothing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import sklearn.linear_model

from cleverly.datasets import make_shift_dose
from cleverly.estimators import TMLE
from cleverly.interventions import Shift, ShiftSet, check_shift_support

SHIFTS = [Shift(0.0, cap=None, name="natural course"), Shift(1.0, cap=None, name="+1")]
CONTRAST = "ate_shift[+1 vs natural course]"


def _fit(**kwargs):  # type: ignore[no-untyped-def]
    data = make_shift_dose(n=600, seed=9000)[0]
    estimator = TMLE(
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=3,
        q_bounds=(-30.0, 40.0),
        shifts=SHIFTS,
        random_state=0,
        simultaneous=False,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            data, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"]
        ).single()


@pytest.fixture(scope="module")
def trimmed():  # type: ignore[no-untyped-def]
    return _fit()


@pytest.fixture(scope="module")
def untrimmed():  # type: ignore[no-untyped-def]
    return _fit(shift_trim=1.0)


def test_the_default_trim_binds_on_the_largest_held_out_ratio(trimmed) -> None:  # type: ignore[no-untyped-def]
    shifts = trimmed.nuisance.shifts
    ratio = shifts.ratio[:, 1]
    ceiling = float(np.quantile(ratio, 0.999))
    # Nonzero witness: at least one held-out ratio lies above the bound.
    assert int(np.sum(ratio > ceiling)) >= 1
    assert shifts.ceiling is not None
    assert shifts.ceiling[1] == ceiling
    design = shifts.design
    assert float(design[:, :, 1].max()) <= ceiling
    # The observed-dose block is the ratio with only the rows above the bound moved.
    np.testing.assert_array_equal(design[:, 0, 1], np.minimum(ratio, ceiling))
    # The untruncated ratio survives for the support report.
    assert float(ratio.max()) > ceiling


def test_the_trim_reaches_the_estimate(trimmed, untrimmed) -> None:  # type: ignore[no-untyped-def]
    assert trimmed.estimates[CONTRAST].std_error < untrimmed.estimates[CONTRAST].std_error


def test_no_trim_leaves_the_covariate_untouched(untrimmed) -> None:  # type: ignore[no-untyped-def]
    shifts = untrimmed.nuisance.shifts
    assert shifts.ceiling is None
    expected = np.concatenate([shifts.ratio[:, None, :], shifts.ratio_at], axis=1)
    np.testing.assert_array_equal(shifts.design, expected)


def test_the_support_report_counts_the_trimmed_rows(trimmed) -> None:  # type: ignore[no-untyped-def]
    shifts = trimmed.nuisance.shifts
    report = check_shift_support(
        shifts, trimmed.nuisance.density, np.asarray(trimmed.data.treatment)
    )["+1"]
    ratio = shifts.ratio[:, 1]
    assert report.ceiling == shifts.ceiling[1]
    assert report.trimmed == int(np.sum(ratio > shifts.ceiling[1])) >= 1
    assert report.max_ratio == float(ratio.max())
    assert "trimmed" in report.summary()


def _tied_set(top_rows: int) -> ShiftSet:
    n = 1000
    ratio = np.ones((n, 1))
    ratio[:top_rows, 0] = 5.0
    ratio[top_rows : top_rows + 10, 0] = 3.0
    at = np.ones((n, 1, 1)) * 2.0
    return ShiftSet(
        ("+1",), (1.0,), np.zeros((n, 1)), ratio, at, np.zeros((n, 1), dtype=bool)
    ).with_trim(0.999)


def test_a_bound_that_does_not_bind_changes_nothing() -> None:
    """Two of a thousand rows share the largest ratio, so the quantile equals it."""
    shifts = _tied_set(top_rows=2)
    assert shifts.ceiling == (5.0,)
    assert shifts.trimmed == (0,)
    expected = np.concatenate([shifts.ratio[:, None, :], shifts.ratio_at], axis=1)
    np.testing.assert_array_equal(shifts.design, expected)


def test_a_single_largest_ratio_is_moved_to_the_quantile() -> None:
    shifts = _tied_set(top_rows=1)
    assert shifts.ceiling is not None
    assert 3.0 < shifts.ceiling[0] < 5.0
    assert shifts.trimmed == (1,)
    assert float(shifts.design.max()) == shifts.ceiling[0]


def test_the_trim_survives_a_row_subset() -> None:
    """A fold of a fitted set keeps the full-sample bound rather than drawing its own."""
    shifts = _tied_set(top_rows=1)
    fold = shifts.subset(np.arange(500, 1000))
    assert fold.ceiling == shifts.ceiling


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_an_out_of_range_trim_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="shift_trim"):
        TMLE(shifts=SHIFTS, shift_trim=value)


def test_the_method_configuration_reaches_the_estimator() -> None:
    from cleverly import Targeting, TMLEMethod

    assert TMLEMethod().estimator_kwargs()["shift_trim"] == 0.999
    method = TMLEMethod(targeting=Targeting(shift_trim=1.0))
    assert method.estimator_kwargs()["shift_trim"] == 1.0
