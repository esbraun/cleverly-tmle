r"""The longitudinal influence curve, against a numerically differentiated functional.

``tests/discrete_law_longitudinal.py`` holds a two-time-point law that the sample
realises exactly and a closed-form statement of the sequential g-formula.  Its Gateaux
derivative -- taken by complex step, so it is exact to double precision -- *is* the
efficient influence function, and it is derived from the functional alone: no cumulative
product, no clever covariate, nothing the library supplies.  Comparing the reported
influence curve against it is therefore a check of the derivation and not a restatement
of it.

Handed the saturated learner on a law the sample realises exactly, the initial fit is
exactly right *in the sample*, so every targeting step's score is already zero, every
``epsilon`` is zero, and the reported curve is the EIF at :math:`P_0` rather than an
estimate of it.  The assertions can then be exact rather than statistical.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE, LongitudinalError

from .. import discrete_law_longitudinal as law

#: Truncation wide enough never to bind: the law's conditionals all lie in [0.25, 0.75].
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)


@pytest.fixture(scope="module")
def fit() -> object:
    """One fit of the exact law, shared by every test in the module."""
    return LTMLE(
        law.REGIMENS,
        reference=law.REGIMEN_REFERENCE,
        outcome_learner=law.CellMeans(),
        pseudo_learner=law.CellMeans(),
        treatment_learner=law.CellMeans(),
        censoring_learner=law.CellMeans(),
        n_folds=1,
        g_bounds=NO_TRUNCATION,
    ).fit(
        law.frame(),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def test_every_reported_parameter_has_an_oracle_and_no_more(fit: object) -> None:
    """The bidirectional gate, in the shape ``tests/unit/test_registry.py`` uses.

    That gate walks the *target registry*, and a regimen is deliberately not a ``Target``
    -- so the longitudinal parameters sit outside it and need their own. Both directions
    matter and only one was covered: parametrizing over ``law.NAMES`` below catches an
    oracle branch the estimator never reports, but nothing caught the reverse, and the
    reverse is the one that ships a number with no independent check behind it. Adding a
    parameter to ``LTMLE._estimates`` now fails here until a longhand functional for it
    exists in ``tests/discrete_law_longitudinal.py``.
    """
    assert set(fit) == set(law.NAMES)  # type: ignore[call-overload]
    # And the law's own names really are backed by the functional rather than declared:
    # ``functional`` raises on a name it has no branch for, so this is not a tautology.
    assert set(law.TRUTH) == set(law.NAMES)
    for name in law.NAMES:
        assert law.functional(law.PROBS, name) == pytest.approx(law.TRUTH[name], abs=0)


@pytest.mark.parametrize("name", law.NAMES)
def test_point_estimate_is_the_g_formula(fit: object, name: str) -> None:
    """With exact nuisances the estimate is the truth, to the last bit."""
    assert fit.psi(name) == pytest.approx(law.TRUTH[name], abs=1e-12)  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", law.NAMES)
def test_influence_curve_is_the_gateaux_derivative(fit: object, name: str) -> None:
    reported = fit.influence_curves[name][law.first_row_of()]  # type: ignore[attr-defined]
    # ``rtol=0`` as in every sibling module: the curve here reaches order 20, so a
    # default relative tolerance would quietly loosen this to ~1e-6 -- six orders
    # short of what the comparison actually holds to, on the module's central claim.
    np.testing.assert_allclose(reported, law.eif(name), atol=1e-14, rtol=0)


def test_the_curve_is_a_function_of_the_support_point_alone(fit: object) -> None:
    """Two rows with the same observed history must carry the same influence curve.

    A curve that varied within a cell would mean something row-specific had leaked into
    it -- a fold index, a fill value, the row order -- which the cell-by-cell comparison
    above could not see, since it reads one row per cell.
    """
    curve = fit.influence_curves["ey_regimen[always]"]  # type: ignore[attr-defined]
    starts = law.first_row_of()
    for position, start in enumerate(starts):
        stop = start + law.COUNTS[position]
        np.testing.assert_allclose(curve[start:stop], curve[start], atol=1e-12, rtol=0)


def test_targeting_had_nothing_to_do(fit: object) -> None:
    """An exact initial fit already solves every score equation, so ``epsilon`` is zero.

    Worth asserting rather than assuming: an ``epsilon`` that came back non-zero here
    would mean the clever covariate and the regression disagree about which rows the
    score is taken over, which is the commonest way to get a longitudinal fit subtly
    wrong while it still looks convergent.
    """
    for regimen_fit in fit.fits.values():  # type: ignore[attr-defined]
        for step in regimen_fit.steps:
            assert step.fluctuation.converged
            assert abs(float(step.fluctuation.epsilon[0])) < 1e-8
            np.testing.assert_allclose(step.targeted, step.initial, atol=1e-9, rtol=0)


def test_the_contrast_curve_is_the_difference_of_the_two(fit: object) -> None:
    """Exactly, not approximately: a contrast is built from the joint curve."""
    left = fit.influence_curves["ey_regimen[always]"]  # type: ignore[attr-defined]
    right = fit.influence_curves["ey_regimen[never]"]  # type: ignore[attr-defined]
    difference = fit.influence_curves["ate_regimen[always vs never]"]  # type: ignore[attr-defined]
    np.testing.assert_allclose(difference, left - right, atol=1e-14, rtol=0)


def test_dropping_the_censoring_factor_would_be_wrong() -> None:
    """A negative control: the censoring probabilities are load-bearing here.

    Estimating the same law with the censoring nodes declared away -- as though the
    units that left had simply never existed -- moves the estimate off the truth by far
    more than machine precision.  Without this, a fit that silently ignored ``C1`` and
    ``C2`` would pass every assertion above that reads only the treatment mechanism.
    """
    frame = law.frame()
    complete = frame[(frame["C1"] == 1) & (frame["C2"] == 1)].reset_index(drop=True)
    naive = LTMLE(
        law.REGIMENS,
        outcome_learner=law.CellMeans(),
        pseudo_learner=law.CellMeans(),
        treatment_learner=law.CellMeans(),
        n_folds=1,
        g_bounds=NO_TRUNCATION,
    ).fit(
        complete,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
    )
    gap = abs(naive.psi("ey_regimen[always]") - law.TRUTH["ey_regimen[always]"])
    assert gap > 1e-3


def test_refuses_a_regimen_no_unit_followed() -> None:
    """A plan nobody followed has no sample to fit on, and says so."""
    frame = law.frame()
    frame.loc[frame["A1"] == 1, "A1"] = 0.0
    with pytest.raises(LongitudinalError, match="no unit followed regimen"):
        LTMLE(
            {"always": 1},
            outcome_learner=law.CellMeans(),
            pseudo_learner=law.CellMeans(),
            treatment_learner=law.CellMeans(),
            censoring_learner=law.CellMeans(),
            n_folds=1,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
