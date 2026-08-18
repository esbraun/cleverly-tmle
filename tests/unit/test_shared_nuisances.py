"""One set of nuisance fits for both controlled direct effects.

The CDE at ``z = 0`` and at ``z = 1`` are different parameters and get one result
each, but they are estimated from *identical* nuisance models: the propensity, the
missingness mechanism, the intermediate mechanism and the outcome regression are all
level-independent -- the outcome model's design uses the observed ``Z`` -- and the
levels differ only in which counterfactual design the outcome regression is predicted
onto.  Fitting per level therefore refits all four models to obtain two extra
prediction vectors.

The test that matters is :meth:`TestIdentical.test_the_shared_path_changes_no_number`:
the optimisation is only worth having if it is invisible in the output.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import sklearn.linear_model

from cleverly.datasets import GENERATORS
from cleverly.estimators import CTMLE, TMLE

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _frame():  # type: ignore[no-untyped-def]
    frame, _ = GENERATORS["cde"](n=400, seed=5)
    return frame, [c for c in frame.columns if c.startswith("W")]


def _fit(shared: bool):  # type: ignore[no-untyped-def]
    frame, covariates = _frame()
    estimator = TMLE(
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        intermediate_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=4,
        random_state=7,
    )
    if not shared:
        estimator._shares_nuisances_across_levels = lambda: False  # type: ignore[method-assign]
    return estimator.fit(frame, outcome="Y", treatment="A", covariates=covariates, intermediate="Z")


class TestIdentical:
    def test_the_shared_path_changes_no_number(self) -> None:
        """Bit-for-bit, at both levels, for every estimand."""
        shared, separate = _fit(shared=True), _fit(shared=False)
        assert sorted(shared) == sorted(separate) == [0.0, 1.0]
        for level in (0.0, 1.0):
            left, right = shared[level], separate[level]
            assert set(left.estimates) == set(right.estimates)
            for name in left.estimates:
                assert left[name].psi == right[name].psi
                assert left[name].variance == right[name].variance
                np.testing.assert_array_equal(
                    left[name].influence_curve, right[name].influence_curve
                )

    def test_the_shared_nuisances_really_are_shared(self) -> None:
        """The level-independent arrays must be the same object's values, not a refit."""
        result = _fit(shared=True)
        first, second = result[0.0].nuisance, result[1.0].nuisance
        np.testing.assert_array_equal(first.propensity.values, second.propensity.values)
        np.testing.assert_array_equal(first.folds.assignment, second.folds.assignment)
        assert first.intermediate is not None
        np.testing.assert_array_equal(first.intermediate, second.intermediate)

    def test_only_the_counterfactual_predictions_differ(self) -> None:
        """The outcome regression at the observed treatment is level-independent."""
        result = _fit(shared=True)
        first, second = result[0.0].nuisance, result[1.0].nuisance
        np.testing.assert_array_equal(first.outcome.observed, second.outcome.observed)
        # ...and the counterfactual arms are genuinely different parameters.
        assert not np.array_equal(first.outcome.arms[1.0], second.outcome.arms[1.0])

    def test_the_two_levels_are_different_parameters(self) -> None:
        """A guard on the fixture: if these coincided the test above proves nothing."""
        result = _fit(shared=True)
        assert result[0.0]["ate"].psi != pytest.approx(result[1.0]["ate"].psi, rel=1e-3)


class TestOptOut:
    def test_the_base_estimator_shares(self) -> None:
        assert TMLE()._shares_nuisances_across_levels()

    def test_ctmle_refits_per_level(self) -> None:
        """C-TMLE selects *which* propensity to target, scored per level.

        Its selection is not level-independent, so it must not be handed nuisances
        fitted for another level.
        """
        assert not CTMLE()._shares_nuisances_across_levels()


class TestAtLevel:
    def test_a_level_that_is_not_a_level_is_refused_first(self) -> None:
        """``check_level`` guards this before the lookup does, and says more."""
        from cleverly.exceptions import DataError

        result = _fit(shared=True)
        with pytest.raises(DataError, match=r"must be 0\.0 or 1\.0"):
            result[0.0].nuisance.at_level(0.5)

    def test_a_level_that_was_never_computed_is_refused(self) -> None:
        result = _fit(shared=True)
        stripped = replace(result[0.0].nuisance, outcome_by_level={})
        with pytest.raises(KeyError, match="no outcome regression"):
            stripped.at_level(1.0)

    def test_both_levels_round_trip(self) -> None:
        result = _fit(shared=True)
        nuisance = result[0.0].nuisance
        np.testing.assert_array_equal(
            nuisance.at_level(1.0).outcome.arms[1.0], result[1.0].nuisance.outcome.arms[1.0]
        )
        np.testing.assert_array_equal(
            nuisance.at_level(0.0).outcome.arms[1.0], result[0.0].nuisance.outcome.arms[1.0]
        )

    def test_a_plain_fit_carries_no_levels(self) -> None:
        """No intermediate, no per-level bookkeeping."""
        frame, _ = GENERATORS["linear_ate"](n=200, seed=1)
        covariates = [c for c in frame.columns if c.startswith("W")]
        result = (
            TMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=4,
                random_state=7,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()
        )
        assert result.nuisance.outcome_by_level == {}
