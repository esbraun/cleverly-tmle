"""The separate randomized missing-outcome DR-TMLE construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model
from sklearn.base import BaseEstimator

from cleverly import AssessmentStatus, CapabilityError, PositivityWarning, load
from cleverly.estimators import DRTMLE
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.reduced import MissingOutcomeReducedSet
from cleverly.estimators.tmle import build_submodel, correction_parts
from cleverly.fluctuation.iterative import InitialFit
from cleverly.inference.influence import missing_outcome_correction_parts
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity._simulated_confounding_request import (
    _MISSING_OUTCOME_REFUSAL,
    _fit_wide_refusal,
)
from cleverly.validation.drtmle import MARGIN_ACTIVE
from tests.unit._confounding_support import forbid_draw_and_refit


def _trial(n: int = 320, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n).astype(float)
    pi = 1.0 / (1.0 + np.exp(-(-0.1 + 0.4 * a + 0.3 * w1)))
    observed = rng.binomial(1, pi, size=n).astype(float)
    y = 0.8 + 1.1 * a + 0.4 * w1 - 0.2 * w2 + rng.normal(scale=0.6, size=n)
    y[observed == 0.0] = np.nan
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Delta": observed, "Y": y})


def _pinched_trial(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """A trial whose *observation* mechanism is pinched and whose randomization is not.

    Three things are arranged, and each is asserted rather than assumed, because a fixture
    where the *wrong* truncation is doing the work would pass while measuring nothing.

    ``A`` is a fair coin independent of ``W``, so ``g-hat`` sits near a half and the
    treatment bounds have nothing to do on either arm.  ``pi`` is driven hard by ``W1``,
    so a fifth of ``P(Delta=1|A,W)`` falls below ``nuisance_bound`` and the observation
    tilt exits against its floor.

    **``Y`` deliberately does not depend on ``W1``.**  ``gamma_Delta`` is a regression of
    the observation indicator on ``Qbar-hat``, so an outcome that tracked ``W1`` would
    carry the pinch straight into the reduction and make ``gr1_margin`` active too -- and
    that column existed before this fixture did, so a bound-active verdict would no longer
    be evidence about the observation mechanism.  With ``W1`` out of the outcome the
    reduction stays well inside its bounds and the only active truncations are the two
    this fixture is for.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n).astype(float)
    pi = 1.0 / (1.0 + np.exp(-(-0.2 + 1.8 * w1)))
    observed = rng.binomial(1, pi, size=n).astype(float)
    y = 0.8 + 1.1 * a - 0.2 * w2 + rng.normal(scale=0.6, size=n)
    y[observed == 0.0] = np.nan
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Delta": observed, "Y": y})


def _estimator(**settings: object) -> DRTMLE:
    return DRTMLE(
        randomized=True,
        cross_fit=False,
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        missingness_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
        reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        estimands=("ate", "ey1", "ey0"),
        simultaneous=False,
        random_state=0,
        **settings,
    )


@pytest.fixture(scope="module")
def randomized_fit():
    return (
        _estimator()
        .fit(
            _trial(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
        )
        .single()
    )


@pytest.fixture(scope="module")
def pinched_observation_fit():
    """A fit whose observation mechanism is bound-active and whose treatment one is not."""
    with pytest.warns(PositivityWarning):
        return (
            _estimator(nuisance_bound=0.15)
            .fit(
                _pinched_trial(),
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                delta="Delta",
            )
            .single()
        )


def test_randomized_missing_outcomes_solve_the_reported_equations(randomized_fit) -> None:
    assert randomized_fit.diagnostics.score_equations().passed
    check = randomized_fit.diagnostics.corrections()
    assert check.passed
    assert {row.equation for row in check.rows} == {"D*_A", "D*_M", "D*_Y"}
    assert len(check.rows) == 6


def test_randomized_missing_outcome_fit_refuses_simulated_confounding_before_work(
    randomized_fit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = _MISSING_OUTCOME_REFUSAL
    forbid_draw_and_refit(monkeypatch, randomized_fit.estimator)
    capability = randomized_fit.sensitivity.capability("simulated_confounding")
    assert not capability.available
    assert capability.status is AssessmentStatus.UNAVAILABLE
    assert capability.reason == reason
    assert _fit_wide_refusal(randomized_fit) == reason
    with pytest.raises(CapabilityError) as refusal:
        simulated_confounding(
            randomized_fit,
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
            benchmark_covariates=("W1",),
            random_state=17,
        )
    assert str(refusal.value) == reason


def test_the_reduced_fit_record_names_the_construction_that_ran(randomized_fit) -> None:
    """The other half of ``test_drtmle_fit.py::test_it_records_what_it_fitted``.

    ``result.extra["drtmle"]`` reported ``reduction="univariate"`` for a fit that ran the
    five-regression construction, because it recorded the constructor argument rather than
    the object.  It also recorded only ``g_bounds``, though ``gamma_m`` and the observation
    mechanism are both formed at ``nuisance_bound`` and neither is reachable from the one
    bound it did carry.
    """
    report = randomized_fit.extra["drtmle"]

    assert report.guard == ("Q", "g")
    assert report.reduction == "missing_outcome"
    assert set(report.diagnostics) == {"gamma_a", "gamma_m", "r_a", "r_m", "e"}
    assert report.missingness_bound == randomized_fit.config.missingness_bound


def test_each_arms_covariate_divides_by_separately_targeted_mechanisms(randomized_fit) -> None:
    data = randomized_fit.data
    repeat = randomized_fit.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    lower, upper = fluctuation.reduction.bounds
    targeted_upper = np.asarray(fluctuation.mechanism.propensity, dtype=float)
    targeted_g = np.column_stack([1.0 - targeted_upper, targeted_upper])
    targeted_m = np.asarray(fluctuation.reduction.observation.propensity, dtype=float)
    current = replace(
        repeat.nuisance,
        propensity=Propensity(targeted_g, data.arm_codes),
        missingness=targeted_m,
    )

    submodel = build_submodel(
        data,
        current,
        "mean",
        bounds=fluctuation.reduction.bounds,
        nuisance_bound=randomized_fit.config.missingness_bound,
    )
    for column, arm in enumerate(data.arm_codes):
        expected = 1.0 / (
            np.clip(targeted_g[:, column], lower, upper)
            * np.clip(targeted_m[:, column], randomized_fit.config.missingness_bound, 1.0)
        )
        np.testing.assert_array_equal(submodel.arms[arm][:, column], expected)
    assert fluctuation.mechanism.propensity.ndim == 1
    assert fluctuation.reduction.observation.propensity.shape == targeted_m.shape


def test_the_five_reductions_and_observation_tilt_survive_serialization(
    randomized_fit, tmp_path
) -> None:
    path = tmp_path / "drtmle-missing.cleverly"
    randomized_fit.save(path)
    restored = load(path)
    before = randomized_fit.repeats[0].fluctuations["mean"].reduction
    after = restored.repeats[0].fluctuations["mean"].reduction
    assert isinstance(after.reduced, MissingOutcomeReducedSet)
    for name in ("gamma_a", "gamma_m", "r_a", "r_m", "e"):
        np.testing.assert_array_equal(getattr(after.reduced, name), getattr(before.reduced, name))
    np.testing.assert_array_equal(after.observation.propensity, before.observation.propensity)
    assert restored.diagnostics.score_equations() == randomized_fit.diagnostics.score_equations()
    # The truncation columns read `nuisance.missingness` and `reduction.missingness_bound`,
    # both of which are persisted -- so a reloaded fit must report the same contract as the
    # fit it came from, rather than one measured on whatever survived the round trip.
    assert restored.diagnostics.corrections().rows == (
        randomized_fit.diagnostics.corrections().rows
    )


def test_the_report_shows_the_product_the_covariate_divides_by(randomized_fit) -> None:
    mechanisms = randomized_fit.diagnostics.support().mechanisms
    joint = randomized_fit.nuisance.propensity.values * randomized_fit.nuisance.missingness
    assert mechanisms["P(A=a,Delta=1|W)"]["min"] == pytest.approx(float(joint.min()))
    assert mechanisms["P(A=a,Delta=1|W)"]["min"] < mechanisms["P(Delta=1|A,W)"]["min"]


def test_treatment_truncation_sweep_counts_only_the_treatment_mechanism(randomized_fit) -> None:
    propensity = randomized_fit.nuisance.propensity.values
    bound = 0.1
    curve = randomized_fit.diagnostics.truncation_curve(bounds=[bound])
    reported = float(np.asarray(curve["truncated_fraction"])[0])
    assert reported == pytest.approx(float(np.mean((propensity < bound) | (propensity > 0.9))))


def test_observation_truncation_sweep_refits_at_and_counts_the_selected_bound(
    randomized_fit,
) -> None:
    missingness = randomized_fit.nuisance.missingness
    assert missingness is not None
    bound = 0.2
    curve = randomized_fit.diagnostics.truncation_curve(bounds=[bound], mechanism=True)
    reported = float(np.asarray(curve["truncated_fraction"])[0])
    assert reported == pytest.approx(float(np.mean(missingness < bound)))

    estimator = randomized_fit.estimator
    assert estimator is not None
    _, fluctuations = estimator.retarget(
        randomized_fit.data,
        randomized_fit.repeats[0].nuisance,
        estimands=("ate",),
        nuisance_bound=bound,
    )
    reduced = fluctuations["mean"].reduction.reduced
    assert isinstance(reduced, MissingOutcomeReducedSet)
    assert reduced.missingness_bound == bound


def test_corrections_keep_the_three_paper_blocks_separate(randomized_fit) -> None:
    repeat = randomized_fit.repeats[0]
    fluctuation = repeat.fluctuations["mean"]
    parts = correction_parts(
        randomized_fit.data,
        repeat.nuisance,
        fluctuation,
        fluctuation.targeted,
        repeat.nuisance.scaler.scale(randomized_fit.data.outcome),
    )
    assert parts.d_a is not None and parts.d_m is not None and parts.d_y is not None
    for arm in randomized_fit.data.arm_codes:
        np.testing.assert_array_equal(parts.d_g[arm], parts.d_a[arm] + parts.d_m[arm])


def test_treatment_correction_has_a_nonzero_independent_witness() -> None:
    reduced = MissingOutcomeReducedSet(
        gamma_a=np.full((4, 2), 0.5),
        gamma_m=np.full((4, 2), 0.7),
        r_a=np.full((4, 2), 0.2),
        r_m=np.full((4, 2), -0.1),
        e=np.array([[0.2, -0.1], [0.3, -0.2], [0.4, -0.3], [0.5, -0.4]]),
        arms=(0.0, 1.0),
        g_bounds=(1e-6, 1 - 1e-6),
        missingness_bound=0.01,
    )
    treatment = np.array([0.0, 0.0, 1.0, 1.0])
    observed = np.array([True, False, True, False])
    targeted = InitialFit(
        np.array([0.2, 0.3, 0.4, 0.5]),
        {0.0: np.zeros(4), 1.0: np.ones(4)},
    )
    parts = missing_outcome_correction_parts(
        np.array([0.1, 0.2, 0.3, 0.4]),
        targeted,
        treatment,
        observed,
        reduced,
        np.array([0.45, 0.55, 0.60, 0.50]),
        np.full((4, 2), 0.7),
        g_bounds=(1e-6, 1 - 1e-6),
        missingness_bound=0.01,
        guard=("Q", "g"),
    )
    assert parts.d_a is not None and parts.d_m is not None
    assert np.max(np.abs(parts.d_a[1.0])) > 0.1
    collapsed = parts.d_a[1.0] + parts.d_m[1.0]
    np.testing.assert_array_equal(parts.d_g[1.0], collapsed)
    assert np.max(np.abs(parts.d_a[1.0] - collapsed)) > 0.01


@pytest.mark.parametrize("guard", [("Q",), ("g",)])
def test_partial_guards_are_refused_for_missing_outcomes(guard) -> None:
    with pytest.raises(NotImplementedError, match="requires guard"):
        _estimator(guard=guard).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


class _FailIfFit(BaseEstimator):
    def fit(self, X, y, sample_weight=None):  # pragma: no cover - failure is the assertion
        raise AssertionError("known randomization probabilities must bypass treatment fitting")

    def predict(self, X):  # pragma: no cover - failure is the assertion
        raise AssertionError("known randomization probabilities must bypass treatment prediction")


def test_known_probabilities_bypass_the_treatment_learner() -> None:
    frame = _trial(n=260, seed=17)
    estimator = DRTMLE(
        randomized=False,
        cross_fit=False,
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=_FailIfFit(),
        missingness_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        reduced_outcome_learner=sklearn.linear_model.LinearRegression(),
        reduced_treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        estimands=("ate",),
        simultaneous=False,
        random_state=0,
    )
    result = estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2"],
        delta="Delta",
        treatment_probabilities=np.full(len(frame), 0.5),
    ).single()
    np.testing.assert_array_equal(result.nuisance.propensity.values, np.full((len(frame), 2), 0.5))
    assert result.diagnostics.score_equations().passed


def test_two_dimensional_known_probabilities_are_copied() -> None:
    frame = _trial(n=140, seed=29)
    supplied = np.full((len(frame), 2), 0.5)
    result = (
        _estimator()
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=supplied,
        )
        .single()
    )
    before = result.nuisance.propensity.values.copy()
    assert not np.shares_memory(supplied, result.nuisance.propensity.values)
    supplied[:, :] = (0.2, 0.8)
    np.testing.assert_array_equal(result.nuisance.propensity.values, before)


def test_known_probabilities_survive_whole_result_persistence(tmp_path) -> None:
    frame = _trial(n=120, seed=31)
    result = (
        _estimator()
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=np.full(len(frame), 0.5),
        )
        .single()
    )
    path = tmp_path / "known-probabilities.cleverly"
    result.save(path)
    restored = load(path)
    assert restored.diagnostics.score_equations() == result.diagnostics.score_equations()
    curve = restored.diagnostics.truncation_curve(bounds=[0.05])
    assert len(curve) == len(result.estimates)
    refitted = restored.estimator.refit(restored.data)
    np.testing.assert_array_equal(
        refitted.nuisance.propensity.values,
        restored.nuisance.propensity.values,
    )


def test_observational_missing_outcomes_are_refused() -> None:
    with pytest.raises(NotImplementedError, match="randomized trial"):
        DRTMLE(cross_fit=False, estimands=("ate",)).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


def test_cross_fitted_missing_outcomes_are_refused() -> None:
    with pytest.raises(NotImplementedError, match="cross-validated extension"):
        DRTMLE(randomized=True, estimands=("ate",)).fit(
            _trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta"
        )


def test_bivariate_missing_outcomes_are_refused_as_a_different_construction() -> None:
    with pytest.raises(NotImplementedError, match="complete-outcome construction"):
        DRTMLE(
            randomized=True,
            cross_fit=False,
            reduction="bivariate",
            estimands=("ate",),
        ).fit(_trial(100), outcome="Y", treatment="A", covariates=["W1", "W2"], delta="Delta")


@pytest.mark.parametrize(
    "probabilities, message",
    [
        (np.full(99, 0.5), "99 rows"),
        (np.full((100, 3), 1 / 3), r"must be \(n,\)"),
        (np.zeros(100), "strictly between"),
        (np.column_stack([np.full(100, 0.5), np.full(100, 0.4)]), "sum to one"),
        ({"nope": np.full(100, 0.5), 1: np.full(100, 0.5)}, "not a level"),
        ({1: np.full(100, 0.6)}, "must name every arm"),
        ({0: np.full(3, 0.4), 1: np.full(100, 0.6)}, r"\[0\] has 3 rows"),
    ],
)
def test_invalid_known_probability_shapes_are_refused(probabilities, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _estimator().fit(
            _trial(100),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=probabilities,
        )


def test_bootstrapping_known_probabilities_is_refused() -> None:
    """A replicate refits on resampled rows the supplied array cannot follow.

    An n-out-of-n resample even passes the length check, so nothing downstream would
    notice -- and ``run_bootstrap`` swallows replicate failures, so raising inside one
    would come back as "the fit is too unstable to bootstrap".
    """
    with pytest.raises(NotImplementedError, match="n_bootstrap"):
        _estimator(n_bootstrap=5).fit(
            _trial(100),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=np.full(100, 0.5),
        )


def test_bootstrapping_known_probabilities_is_refused_without_a_guard() -> None:
    """The control for lifting the unguarded refusal, and the reason it is a pair.

    This refusal used to live *inside* the ``guard``-gated block, so accepting
    ``guard=()`` with ``delta=`` and known probabilities -- which the plumbing already
    supported -- would have walked straight past it. The misalignment it prevents is
    silent, so a fit that merely runs is not evidence that it is right.
    """
    with pytest.raises(NotImplementedError, match="n_bootstrap"):
        _estimator(guard=(), n_bootstrap=5).fit(
            _trial(100),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=np.full(100, 0.5),
        )


def test_known_probabilities_configure_an_unguarded_plain_tmle() -> None:
    """``guard=()`` with ``delta=`` was refused, and the message named ``delta=``.

    It is bit for bit a plain TMLE, and a trial's design probabilities are exactly what
    such a fit should divide by -- so the refusal was both unnecessary and, on a fit that
    *had* passed ``delta=``, false. ``_FailIfFit`` is the witness that the supplied array
    reached the fit rather than the refusal merely being gone: the treatment learner must
    never run, and ``contract == "none"`` is what says the unguarded path was taken.
    """
    frame = _trial(n=260, seed=17)
    estimator = DRTMLE(
        guard=(),
        randomized=False,
        cross_fit=False,
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=_FailIfFit(),
        missingness_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        estimands=("ate",),
        simultaneous=False,
        random_state=0,
    )
    result = estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2"],
        delta="Delta",
        treatment_probabilities=np.full(len(frame), 0.5),
    ).single()

    np.testing.assert_array_equal(result.nuisance.propensity.values, np.full((len(frame), 2), 0.5))
    assert result.extra["drtmle"].guard == ()
    capability = result.diagnostics.capability("corrections")
    assert capability.status is AssessmentStatus.NOT_APPLICABLE
    with pytest.raises(CapabilityError, match="subtracts no correction term"):
        result.diagnostics.corrections()
    assert result.diagnostics.score_equations().passed


def test_known_probabilities_without_delta_are_still_refused() -> None:
    """The half of the old refusal that was true, which had no test of its own."""
    frame = _trial(n=120, seed=3).assign(Y=lambda f: f["Y"].fillna(0.0)).drop(columns=["Delta"])
    with pytest.raises(ValueError, match="only used with delta="):
        _estimator(guard=()).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            treatment_probabilities=np.full(len(frame), 0.5),
        )


def _labelled_trial(n: int = 300, seed: int = 21) -> tuple[pd.DataFrame, np.ndarray]:
    """A stratified trial whose arms are named, so the positional binding is asymmetric.

    ``active`` sorts before ``placebo``, so arm code ``1`` is ``placebo`` and a caller who
    reads the one-dimensional form as "the probability of treatment" inverts the design.
    The allocation is unequal *and* varies with ``W1``: a constant one would make the
    inversion invisible in the point estimate, because rescaling both arms' clever
    covariates by constants is absorbed by the fluctuation's own coefficients.
    """
    frame = _trial(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    active = np.where(frame["W1"].to_numpy() > 0.0, 0.75, 0.55)
    treated = rng.binomial(1, active).astype(float)
    frame["A"] = treated
    frame["arm"] = np.where(treated == 1.0, "active", "placebo")
    return frame, active


def test_known_probabilities_may_be_keyed_by_the_arm_they_belong_to() -> None:
    """The mapping form says which arm each column is, and matches the positional one.

    ``{"active": p_active, "placebo": p_placebo}`` and the ``(n,)`` vector
    ``P(A = 'placebo' | W)`` are the same design stated two ways, so they must give the
    same mechanism.  Writing the vector as ``P(A = 'active' | W)`` instead is a different
    design, and the assertion at the end is that the two really do differ -- otherwise the
    agreement above would be vacuous.
    """
    frame, active = _labelled_trial()
    keyword = {"outcome": "Y", "treatment": "arm", "covariates": ["W1", "W2"], "delta": "Delta"}
    keyed = (
        _estimator()
        .fit(frame, **keyword, treatment_probabilities={"active": active, "placebo": 1.0 - active})
        .single()
    )
    positional = _estimator().fit(frame, **keyword, treatment_probabilities=1.0 - active).single()
    np.testing.assert_array_equal(
        keyed.nuisance.propensity.values, positional.nuisance.propensity.values
    )
    # Column 1 is `placebo`, the second sorted level -- not "the treated arm".
    np.testing.assert_allclose(keyed.nuisance.propensity.arm(1.0), 1.0 - active)
    inverted = _estimator().fit(frame, **keyword, treatment_probabilities=active).single()
    assert abs(inverted.estimates["ate"].psi - keyed.estimates["ate"].psi) > 1e-8


def test_an_unnamed_probability_vector_says_which_arm_it_bound_to() -> None:
    """The message names the resolved level, since the code is what the caller cannot see."""
    frame, _ = _labelled_trial(n=120)
    with pytest.raises(ValueError, match="P\\(arm = 'placebo' \\| W\\)"):
        _estimator().fit(
            frame,
            outcome="Y",
            treatment="arm",
            covariates=["W1", "W2"],
            delta="Delta",
            treatment_probabilities=np.full((len(frame), 3), 1 / 3),
        )


class TestTheContractSeesTheObservationTruncations:
    """Which estimator a missing-outcome fit's numbers are evidence about.

    ``correction_check`` measured the contract on the **treatment** mechanism alone --
    ``reported_mechanism`` returns the targeted treatment propensity by its own docstring,
    and ``initial_clipped`` counted ``propensity.values`` -- while
    ``nuisance.missingness`` and ``reduction.observation`` were both in scope. That is
    blind in exactly the regime this construction is for: a randomized trial's treatment
    mechanism is flat by design and *cannot* clip, so the three columns that existed all
    read "inactive" however much of ``P(Delta=1|A,W)`` was pinned at ``nuisance_bound``,
    and the fit was certified ``"theorem"``.

    **The pair of fits is the test**, as it is for the complete-data label in
    ``test_drtmle_fit.py``: a column that could not disagree is not a witness.
    """

    def test_a_well_behaved_trial_is_still_the_theorems_estimator(self, randomized_fit) -> None:
        check = randomized_fit.diagnostics.corrections()

        assert check.contract == "theorem"
        assert check.truncations_active == ()
        assert check.observation_clip_share == 0.0
        assert check.observation_margin > MARGIN_ACTIVE

    def test_a_pinched_observation_mechanism_is_bound_active(self, pinched_observation_fit) -> None:
        """The nonzero witness. Both observation truncations bite, and the label says so."""
        check = pinched_observation_fit.diagnostics.corrections()

        assert check.observation_clip_share > 0.05
        assert check.observation_margin <= MARGIN_ACTIVE
        assert check.contract == "bound-active"
        assert check.truncations_active == (
            "pi-hat at the initial fit",
            "pi* at the exit",
        )

    def test_every_column_that_existed_before_calls_this_fit_the_theorems(
        self, pinched_observation_fit
    ) -> None:
        """The deliberate-mutation control, written down rather than promised.

        Every truncation column ``correction_check`` carried before this pair is inactive
        on this fixture -- the treatment mechanism at both ends, and the gamma reductions
        the ``gr1_margin`` column reuses. So a check reading only them returns
        ``"theorem"`` here, which is what it did. Reverting the fix therefore *must* turn
        the assertions above red, and this test is also what forbids the fixture from
        drifting into a state where some other truncation is doing the work.
        """
        check = pinched_observation_fit.diagnostics.corrections()

        assert check.initial_clip_share == 0.0
        assert check.margin > MARGIN_ACTIVE
        assert check.gr1_margin > MARGIN_ACTIVE

    def test_bound_active_is_a_scope_label_here_too(self, pinched_observation_fit) -> None:
        """A pinched fit is outside Theorem 1, not broken -- as on the complete-data path."""
        check = pinched_observation_fit.diagnostics.corrections()

        assert check.passed
        assert check.identity_failures() == ()
        assert check.correction_failures() == ()
        assert pinched_observation_fit.diagnostics.score_equations().passed

    def test_the_two_new_columns_are_on_the_face_of_the_check(
        self, pinched_observation_fit
    ) -> None:
        check = pinched_observation_fit.diagnostics.corrections()
        frame = check.to_frame()
        summary = check.summary()

        assert {"observation_clipped", "observation_margin"} <= set(frame.columns)
        assert all(row.observation_clipped > 0 for row in check.rows)
        assert "pi* at the exit" in summary
        assert "none of the three truncations" not in summary


class TestTheJointRowCountsTheTruncationTheEstimatorApplies:
    """``P(A=a,Delta=1|W)`` is a derived denominator, and it has no bound of its own.

    The row counted its ``clipped`` cells against ``g_bounds[0] * nuisance_bound`` -- a
    floor that appears nowhere else in the package, because the estimator truncates the
    two factors **separately** and multiplies them. A raw product only falls below the
    product of the floors when *both* factors are small, so the count was a strict subset
    of the cells truncation altered, and at the shipped defaults the floor is small enough
    that the row reported essentially zero however hard the observation bound was working.
    """

    def test_it_counts_the_cells_either_factors_truncation_altered(
        self, pinched_observation_fit
    ) -> None:
        treatment = pinched_observation_fit.nuisance.propensity.values
        observation = pinched_observation_fit.nuisance.missingness
        g_lower, g_upper = pinched_observation_fit.config.g_bounds
        m_lower = pinched_observation_fit.config.missingness_bound
        altered = (treatment < g_lower) | (treatment > g_upper) | (observation < m_lower)
        stats = pinched_observation_fit.diagnostics.support().mechanisms["P(A=a,Delta=1|W)"]

        assert altered.sum() > 0, "the fixture's precondition: the truncation must bite"
        assert stats["clipped"] == pytest.approx(float(altered.sum()))
        assert stats["clipped_fraction"] == pytest.approx(float(altered.mean()))

        # The control. The product floor this row used to count against is applied by no
        # code, and counts strictly fewer cells -- so the assertions above are red the
        # moment it comes back.
        product_floor = float(((treatment * observation) < g_lower * m_lower).sum())
        assert product_floor < stats["clipped"]

    def test_its_effective_sample_size_uses_the_denominator_the_equation_forms(
        self, pinched_observation_fit
    ) -> None:
        """``clip(g) * clip(pi)``, not ``max(g * pi, g_lo * m_lo)``, which nothing forms."""
        data = pinched_observation_fit.data
        treatment = pinched_observation_fit.nuisance.propensity.values
        observation = pinched_observation_fit.nuisance.missingness
        g_lower, g_upper = pinched_observation_fit.config.g_bounds
        m_lower = pinched_observation_fit.config.missingness_bound
        bounded = np.clip(treatment, g_lower, g_upper) * np.clip(observation, m_lower, 1.0)
        treated = data.treatment == 1.0
        # Rows with no recorded outcome contribute an exact zero to the residual term, so
        # the mechanism never weights them and they do not belong in its ESS.
        contributing = np.asarray(data.observed, dtype=bool)
        used = np.where(treated, bounded[:, 1], bounded[:, 0])[contributing]

        weights = 1.0 / used
        expected = (weights.sum() ** 2 / np.square(weights).sum()) / float(used.size)
        stats = pinched_observation_fit.diagnostics.support().mechanisms["P(A=a,Delta=1|W)"]

        assert stats["ess_ratio"] == pytest.approx(expected)

        # The control, as above: the floored raw product is a third array, neither the one
        # that was fitted nor the one that was divided by.
        raw = treatment * observation
        floored = np.maximum(
            np.where(treated, raw[:, 1], raw[:, 0])[contributing], g_lower * m_lower
        )
        old_weights = 1.0 / floored
        old = (old_weights.sum() ** 2 / np.square(old_weights).sum()) / float(floored.size)
        assert stats["ess_ratio"] != pytest.approx(old)

    def test_it_names_both_bounds_it_was_truncated_at(self, pinched_observation_fit) -> None:
        """Every other row has one bound; quoting it here named one the row never met."""
        report = pinched_observation_fit.diagnostics.support()
        text = report.summary() + report.verdict()

        assert "P(A=a,Delta=1|W) truncated to" in text
        assert f"[{report.bounds[0]:.4g}, {report.bounds[1]:.4g}] x " in text
        assert f"[{report.nuisance_bound:.4g}, 1], factor by factor" in text
