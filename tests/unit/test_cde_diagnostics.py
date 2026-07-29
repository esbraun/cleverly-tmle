r"""The controlled-direct-effect diagnostics, and the ``z`` convention they all share.

Every denominator in the clever covariate has a diagnostic attached to it, and for a
controlled direct effect one of those denominators is
:math:`q_z(a, W) = P(Z = z \mid A = a, W)`.  The nuisance array stores
:math:`P(Z = 1 \mid A, W)`, so the quantity the covariate actually divides by is the
array at ``z = 1`` and its *complement* at ``z = 0``.  That is one line of convention
restated at four call sites, and three of them used to get it backwards -- reporting the
overlap of the level that was not being targeted, which is exactly the reading that makes
a fatal positivity violation look immaculate.

The first law below is built so the two levels are as unlike each other as a binary
variable permits: ``P(Z = 1 | A, W)`` is 0.999 everywhere, so ``q_1`` is about as healthy
as a density gets and ``q_0`` is a hard violation on every row.  A diagnostic that reads
the wrong array therefore does not merely lose precision, it inverts: silent at ``z = 0``
where it should shout, and shouting at ``z = 1`` where there is nothing to report.
Restoring any one of the three defects fails a named subset of these tests, and restoring
all three fails nine of them -- five by reporting exactly zero where the answer is one.

That law cannot test everything, and the reason is worth stating rather than working
around silently: its density is the same number on every row, so a Kish effective sample
size taken over *any* set of rows comes out at one, and the claim that only the targeted
rows count has nothing to bite on.  The effective-sample-size tests therefore use
:func:`~cleverly.datasets.make_cde`, whose mechanism varies with both ``A`` and ``W``.

Nothing here is statistical.  The fits use ``glm`` at ``n = 600`` and every assertion is
about which array a report reads, so they are deterministic; three shared fits keep the
module at about four seconds.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from cleverly import TMLE
from cleverly.estimators.direct_effect import (
    check_level,
    clever_covariate_inputs,
    describe,
    targeted_rows,
)
from cleverly.exceptions import DataError, PositivityWarning

#: ``P(Z = 1 | A, W)`` in the process below.  Extreme on purpose: it makes ``q_0`` a
#: violation on every row while leaving ``q_1`` untouched, so a report that reads the
#: wrong one is off by the whole width of the unit interval rather than by a little.
P_Z = 0.999

N = 600


def _frame(seed: int = 0) -> Any:
    """A sample with severe positivity in ``Z = 0`` and none at all in ``Z = 1``.

    ``Z`` is drawn at :data:`P_Z` and then forced to zero on a handful of rows.  Those
    rows are what make the ``z = 0`` fit *possible* -- with no ``Z = 0`` observations at
    all the fluctuation has an empty score and the fit is undefined rather than
    ill-conditioned, which is a different failure and not the one under test.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    w = rng.normal(size=N)
    a = rng.binomial(1, 0.5, size=N).astype(float)
    z = rng.binomial(1, P_Z, size=N).astype(float)
    z[:12] = 0.0
    y = 0.4 + 0.5 * a + 0.8 * z + 0.3 * w + rng.normal(scale=0.5, size=N)
    return pd.DataFrame({"W": w, "A": a, "Z": z, "Y": y})


class _ConstantIntermediate(BaseEstimator):
    """A learner that always predicts :data:`P_Z`.

    A fitted model on this sample would predict close to :data:`P_Z` anyway, but "close"
    is the wrong footing for a test whose assertions are about exact fractions: a single
    fold whose estimate drifted below the bound would move ``clipped_fraction`` off 1.0
    and the failure would read as a bug in the diagnostic rather than as noise.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _ConstantIntermediate:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        p = np.full(np.asarray(X, dtype=float).shape[0], P_Z)
        return np.column_stack([1.0 - p, p])


def _plain_data() -> Any:
    """A minimal :class:`~cleverly.data.CausalData` with no intermediate variable."""
    from cleverly.data import CausalData

    rng = np.random.default_rng(0)
    return CausalData.from_arrays(
        outcome=rng.normal(size=20),
        treatment=np.resize([0.0, 1.0], 20),
        covariates=rng.normal(size=(20, 2)),
    )


def _fit(z: float) -> Any:
    """Fit both levels and return the one at ``z``.

    Note that ``fit`` always targets both levels -- a controlled direct effect is defined
    per level and the API reports every one -- so a single call emits the fit-time
    warnings of *both*.  That is why the warning assertions below are phrased over one
    call's messages rather than as "this level warns, that level does not".
    """
    estimator = TMLE(
        outcome_learner="glm",
        treatment_learner="glm",
        intermediate_learner=_ConstantIntermediate(),
        n_folds=3,
        random_state=0,
        simultaneous=False,
        estimands=("ate",),
    )
    return estimator.fit(_frame(), outcome="Y", treatment="A", covariates=["W"], intermediate="Z")[
        z
    ]


class TestThePositivityWarningReadsTheTargetedLevel:
    """The fit-time warning has to be about the density the covariate divides by.

    One :meth:`~cleverly.TMLE.fit` call targets *both* levels, so the two directions of
    this claim are two assertions about the same set of warnings rather than two fits:
    the ``z = 0`` level must warn and the ``z = 1`` level must not.  Asserting only the
    first would pass on a report that is merely mirror-inverted, which is precisely the
    bug -- so the negative half is the half that has teeth.
    """

    @pytest.fixture(scope="class")
    def messages(self) -> list[str]:
        with pytest.warns(PositivityWarning) as caught:
            _fit(0.0)
        return [str(record.message) for record in caught]

    def test_the_violated_level_warns(self, messages: list[str]) -> None:
        assert any("P(Z = 0 | A, W)" in message for message in messages), messages

    def test_the_healthy_level_does_not(self, messages: list[str]) -> None:
        assert not any("P(Z = 1 | A, W)" in message for message in messages), messages

    def test_it_reports_the_whole_sample(self, messages: list[str]) -> None:
        # q_0 is 0.001 on every row, so the fraction below the 0.01 bound is all of them.
        # A report reading P(Z = 1 | A, W) would have said 0.0% and warned about nothing.
        assert any("100.0% of estimated P(Z = 0 | A, W)" in message for message in messages), (
            messages
        )


@pytest.fixture(scope="module")
def fit_at_zero() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit(0.0)


@pytest.fixture(scope="module")
def fit_at_one() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit(1.0)


class TestTheTruncationCurveReadsTheTargetedLevel:
    """``truncated_fraction`` is the only column on that curve that can see this at all.

    Truncating a density to a constant rescales both clever-covariate columns by the same
    factor, and ``epsilon * h`` is invariant to a constant rescale, so ``psi`` sits flat
    across the whole sweep however hard the bound is binding.  That makes the fraction
    column not a convenience but the entire diagnostic, and it was the broken one.
    """

    @pytest.mark.parametrize(("level", "expected"), [("zero", 1.0), ("one", 0.0)])
    def test_the_fraction_is_reported_for_the_targeted_level(
        self, request: Any, level: str, expected: float
    ) -> None:
        result = request.getfixturevalue(f"fit_at_{level}")
        curve = result.sensitivity.truncation_curve(mechanism=True, bounds=[0.01])
        fraction = np.asarray(curve["truncated_fraction"], dtype=float)
        assert np.allclose(fraction, expected), f"{level}: {fraction}"

    @pytest.mark.parametrize(("level", "z"), [("zero", 0.0), ("one", 1.0)])
    def test_it_agrees_with_the_positivity_report(self, request: Any, level: str, z: float) -> None:
        # Two code paths report the same clipping from the same array; asserting they
        # agree is what survives a refactor of either.
        result = request.getfixturevalue(f"fit_at_{level}")
        curve = result.sensitivity.truncation_curve(mechanism=True, bounds=[0.01])
        reported = result.sensitivity.positivity().mechanisms[f"P(Z={z:.0f}|A,W)"][
            "clipped_fraction"
        ]
        assert np.allclose(np.asarray(curve["truncated_fraction"], dtype=float), reported)


@pytest.fixture(scope="module")
def overlapping_fit() -> Any:
    """A fit on :func:`~cleverly.datasets.make_cde`, whose intermediate mechanism varies.

    The extreme law above cannot test the effective sample size: its density is the same
    number on every row, so the Kish ESS is 1 whichever set of rows it is taken over and
    the claim "only the targeted rows count" has nothing to bite on.  Here ``q_z`` depends
    on both ``A`` and ``W``, and the rows at the other level are systematically the ones
    with the *smaller* density -- so dropping them moves the answer, which is the whole
    point.
    """
    from cleverly.datasets import make_cde

    frame, _ = make_cde(n=600, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            n_folds=3,
            random_state=0,
            simultaneous=False,
            estimands=("ate",),
        ).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"], intermediate="Z")


class TestTheEffectiveSampleSizeCountsOnlyTheRowsTheMechanismWeights:
    """A row at the other level of ``Z`` contributes an exact zero, not a small weight."""

    @pytest.mark.parametrize("z", [0.0, 1.0])
    def test_it_matches_a_hand_count_over_the_targeted_rows(
        self, overlapping_fit: Any, z: float
    ) -> None:
        result = overlapping_fit[z]
        data, nuisance = result.data, result.nuisance
        density = nuisance.intermediate_density(z, 0.0)
        at_arm = np.where(data.treatment == 1.0, density[:, 1], density[:, 0])
        used = np.maximum(at_arm[data.observed & (data.intermediate == z)], 0.01)

        weights = 1.0 / used
        expected = float(weights.sum() ** 2 / (weights.size * (weights**2).sum()))
        reported = result.sensitivity.positivity().mechanisms[f"P(Z={z:.0f}|A,W)"]["ess_ratio"]
        assert reported == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("z", [0.0, 1.0])
    def test_taking_it_over_every_row_would_give_a_different_answer(
        self, overlapping_fit: Any, z: float
    ) -> None:
        # The negative half: without it the test above would pass on the old code, which
        # averaged over the complete cases regardless of the level.
        result = overlapping_fit[z]
        data = result.data
        density = result.nuisance.intermediate_density(z, 0.0)
        at_arm = np.where(data.treatment == 1.0, density[:, 1], density[:, 0])
        every_row = 1.0 / np.maximum(at_arm[data.observed], 0.01)
        wrong = float(every_row.sum() ** 2 / (every_row.size * (every_row**2).sum()))
        reported = result.sensitivity.positivity().mechanisms[f"P(Z={z:.0f}|A,W)"]["ess_ratio"]
        assert abs(reported - wrong) > 1e-3, (reported, wrong)

    def test_the_targeted_row_mask_is_the_intersection(self, fit_at_zero: Any) -> None:
        data = fit_at_zero.data
        mask = targeted_rows(data, 0.0)
        assert np.array_equal(mask, np.asarray(data.observed, bool) & (data.intermediate == 0.0))
        # 12 rows were forced to Z = 0 and nothing is missing here, so the mask is tight.
        assert int(mask.sum()) == 12


class TestTheLevelIsValidated:
    """A level the code does not recognise must raise, never fall through to ``1 - p``."""

    @pytest.mark.parametrize("value", [2.0, -1.0, 0.5])
    def test_check_level_rejects_it(self, value: float) -> None:
        with pytest.raises(DataError, match=r"must be 0\.0 or 1\.0"):
            check_level(value)

    def test_the_density_accessor_rejects_it(self, fit_at_one: Any) -> None:
        with pytest.raises(ValueError, match=r"must be 0\.0 or 1\.0"):
            fit_at_one.nuisance.intermediate_density(2.0, 0.01)

    def test_a_missing_level_raises_rather_than_asserts(self, fit_at_one: Any) -> None:
        # It was an ``assert`` before, which ``python -O`` removes; the difference it
        # guards is between a controlled direct effect and an average treatment effect
        # with an extra covariate.
        with pytest.raises(DataError, match="no intermediate_value was supplied"):
            clever_covariate_inputs(fit_at_one.data, fit_at_one.nuisance, None, 0.01)

    def test_a_level_without_an_intermediate_raises(self) -> None:
        # The mirror of the test above, and the reason ``counterfactual_design`` needed a
        # guard too: it used to append the column regardless of whether the data had one,
        # producing a prediction design one column wider than the model was fitted on.
        data = _plain_data()
        with pytest.raises(DataError, match="no intermediate variable"):
            data.counterfactual_design(1.0, intermediate_value=0.0)

    def test_the_helper_rejects_a_level_the_data_cannot_carry(self, fit_at_one: Any) -> None:
        # The guard fires on the data before the nuisance is consulted, so the nuisance
        # here is only a well-formed stand-in -- pairing it with intermediate-free data
        # is the mismatch under test, not an accident.
        with pytest.raises(DataError, match="no intermediate variable"):
            clever_covariate_inputs(_plain_data(), fit_at_one.nuisance, 0.0, 0.01)


class TestTheSummaryNamesTheEstimand:
    def test_it_reports_the_level(self) -> None:
        assert describe(0.0, "Z") == "controlled direct effect at Z = 0"
        assert describe(1.0, "mediator") == "controlled direct effect at mediator = 1"

    def test_the_fit_summary_carries_it(self, fit_at_one: Any) -> None:
        assert "controlled direct effect at Z = 1" in fit_at_one.summary()
