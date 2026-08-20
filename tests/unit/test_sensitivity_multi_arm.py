"""The sensitivity analyses that read a parameter's arms off its name, at three arms.

The omitted-variable bound and the MNAR tilt were refused on a multi-valued treatment and
are now one bound and one tilt *per contrast*; the E-value was not refused but dispatched
on the bare names, so it reached the same wall from the other side.  What each needs
checking for is different, so the instruments differ.

The omitted-variable bound is a closed-form functional of the nuisances, so it is checked
against the closed form: on :mod:`tests.discrete_law_multi`, handed that law's own
nuisances, ``sigma^2`` and ``nu^2`` are exact functions of the eighteen cell
probabilities and this module writes them out longhand.  An implementation that lined the
arms up by position rather than by code produces a perfectly plausible number, and this is
what tells the two apart -- which is why every contrast is checked rather than one.

The MNAR tilt is not a functional of the law but a re-mixing of the *fitted* regression,
so the exact-law instrument has nothing to say about it.  Two properties do: that
``gamma = 0`` reproduces the fit's own report -- for every parameter, including the
conditional effects whose weights are an arm indicator that a three-armed treatment turns
from a 0/1 column into the codes 0, 1, 2 -- and that an arm whose outcomes are never
missing is left exactly where it was however the others are tilted.

The E-value's own arithmetic is checked against published values in
:mod:`tests.unit.test_sensitivity_units`; what is checked here is which parameters it
reaches and which mean its risk-difference conversion divides by.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model
from sklearn.base import BaseEstimator

from cleverly.estimators import TMLE
from cleverly.sensitivity.missingness import missingness_tilt, tipping_gamma
from cleverly.sensitivity.omitted_variable import (
    omitted_variable_bounds,
    robustness_value,
    sensitivity_elements,
)
from tests import discrete_law_multi as law

# --------------------------------------------------------------------------- the bound

#: The reference these fits report against, as a *label*: "low" has arm code 1, since
#: the levels sort to ("high", "low", "mid"). A test whose reference were also code 0
#: could not tell an implementation reading the reference from one assuming it.
REFERENCE = "low"

#: This module's arm index for the reference, for the closed forms below, which are
#: written in the law's own arm order rather than in the library's codes.
REFERENCE_ARM = law.LABELS.index(REFERENCE)

#: ``P(W = w)``, ``P(A = a)``, ``g(a | w)`` and ``Qbar(a, w)`` as the realised sample has
#: them -- taken from the cell probabilities, so nothing here goes through the library.
P_W = law.PROBS.sum(axis=(1, 2))
P_A = law.PROBS.sum(axis=(0, 2))
G = law.G_EXACT
Q = law.Q_EXACT


def _sigma2() -> float:
    r"""``E[(Y - Qbar(A, W))^2]``, which for a binary ``Y`` is ``E[Q(1 - Q)]``."""
    total = 0.0
    for w in range(3):
        for a in range(law.K):
            for y in range(2):
                total += law.PROBS[w, a, y] * (y - Q[w, a]) ** 2
    return float(total)


def _nu2(estimand: str, arm: int) -> float:
    r"""``E[alpha(A, W)^2]`` for one parameter, longhand from its Riesz representer.

    The representers, with ``r`` the reference arm and ``P_a = P(A = a)``:

    ``ey``   ``1{A = a} / g_a``
    ``ate``  ``1{A = a} / g_a - 1{A = r} / g_r``
    ``att``  ``(1{A = a} - 1{A = r} g_a / g_r) / P_a``
    ``atc``  ``(1{A = a} g_r / g_a - 1{A = r}) / P_r``

    Squaring drops the cross terms -- the two indicators are disjoint -- and averaging
    over the arm leaves one factor of ``g`` behind each.
    """
    r = REFERENCE_ARM
    if estimand == "ey":
        return float((P_W / G[:, arm]).sum())
    if estimand == "ate":
        return float((P_W * (1.0 / G[:, arm] + 1.0 / G[:, r])).sum())
    if estimand == "att":
        return float((P_W * G[:, arm] * (1.0 + G[:, arm] / G[:, r])).sum() / P_A[arm] ** 2)
    if estimand == "atc":
        return float((P_W * G[:, r] * (1.0 + G[:, r] / G[:, arm])).sum() / P_A[r] ** 2)
    raise ValueError(estimand)  # pragma: no cover - a typo in a parametrisation


def _name(estimand: str, arm: int) -> str:
    """The reported name for one of this module's (estimand, arm index) pairs."""
    if estimand == "ey":
        return f"ey[{law.LABELS[arm]}]"
    return law.reported_name(f"{estimand}[{arm} vs {REFERENCE_ARM}]")


#: Every parameter the fit below reports, as (estimand, this module's arm index).
PARAMETERS: tuple[tuple[str, int], ...] = tuple(
    (estimand, arm)
    for estimand in ("ey", "ate", "att", "atc")
    for arm in range(law.K)
    if not (estimand != "ey" and arm == REFERENCE_ARM)
)


@pytest.fixture(scope="module")
def exact_fit() -> Any:
    """The three-armed law fitted with its own nuisances, so ``epsilon`` is zero.

    Every group at once -- the conditional effects share the nuisance fits, and asking
    for them separately would fit the same models three times.
    """
    return (
        TMLE(
            outcome_learner=law.OracleMultiOutcome(),
            treatment_learner=law.OracleMultiTreatment(),
            cross_fit=False,
            estimands=("ey", "ate", "att", "atc"),
            reference=REFERENCE,
            simultaneous=False,
            random_state=0,
        )
        .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


class TestTheBoundAtThreeArms:
    def test_the_initial_fit_is_already_targeted(self, exact_fit: Any) -> None:
        """The premise the closed forms below rest on.

        ``sigma^2`` is the residual variance of the *targeted* regression, so comparing
        it against ``E[Q(1 - Q)]`` is only a check on this module's arithmetic if the
        targeting step moved nothing. On this law it does not, because the oracle
        nuisances are exact in the sample; stating it here means the comparison cannot
        quietly become a comparison against a different regression.
        """
        for fluctuation in exact_fit.repeats[0].fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) < 1e-9

    @pytest.mark.parametrize("estimand,arm", PARAMETERS)
    @pytest.mark.parametrize("nu2_estimator", ["plugin", "doubly_robust"])
    def test_the_elements_match_the_closed_form(
        self, exact_fit: Any, estimand: str, arm: int, nu2_estimator: str
    ) -> None:
        """Each parameter's own ``nu^2``, against the representer written out by hand.

        Both estimators of ``nu^2``, because at the truth the Riesz identity makes them
        equal: ``E[m(W, alpha)] = E[alpha^2]``. That they agree here is a check on
        ``_m_alpha``'s arm indexing that no single one of them could be.
        """
        elements = sensitivity_elements(
            exact_fit, _name(estimand, arm), nu2_estimator=nu2_estimator
        )
        assert elements.sigma2 == pytest.approx(_sigma2(), rel=1e-12)
        assert elements.nu2 == pytest.approx(_nu2(estimand, arm), rel=1e-12)
        assert elements.max_bias == pytest.approx(
            np.sqrt(_sigma2() * _nu2(estimand, arm)), rel=1e-12
        )

    def test_the_contrasts_are_not_interchangeable(self, exact_fit: Any) -> None:
        """The parametrisation above is not passing on a coincidence.

        Every parameter's ``nu^2`` differs from every other's on this law, so an
        implementation that answered for the wrong arm would have to be wrong by a
        visible margin rather than by a rounding error.
        """
        values = [
            sensitivity_elements(exact_fit, _name(estimand, arm)).nu2
            for estimand, arm in PARAMETERS
        ]
        assert len(values) == len(PARAMETERS)
        for i, first in enumerate(values):
            for second in values[i + 1 :]:
                assert abs(first - second) > 1e-3

    def test_the_bound_brackets_the_estimate_and_reports_a_robustness_value(
        self, exact_fit: Any
    ) -> None:
        name = _name("ate", law.LABELS.index("high"))
        bounds = omitted_variable_bounds(exact_fit, name, cf_y=0.05, cf_d=0.05)
        assert bounds.estimand == name
        assert bounds.lower < exact_fit.psi(name) < bounds.upper
        assert bounds.ci_lower < bounds.lower and bounds.upper < bounds.ci_upper
        # No confounding is no bias, whatever the arms.
        assert omitted_variable_bounds(exact_fit, name, cf_y=0.0, cf_d=0.0).bias == 0.0
        assert 0.0 < robustness_value(exact_fit, name)["rv"] < 1.0

    def test_a_ratio_is_refused_before_the_fit_is_consulted(self, exact_fit: Any) -> None:
        # Named for arms this fit has, so the refusal has to be about the functional
        # rather than about the parameter being absent.
        with pytest.raises(ValueError, match="applies to"):
            sensitivity_elements(exact_fit, "rr[high vs low]")

    def test_the_bare_name_says_which_contrasts_exist(self, exact_fit: Any) -> None:
        """``"ate"`` is not a parameter of a three-armed fit, and the message says so.

        The default estimand of every entry point is ``"ate"``, so this is the message a
        caller reaches first; listing the contrasts is what turns it into an answer.
        """
        with pytest.raises(ValueError, match=r"available for .*'ate\[high vs low\]'"):
            sensitivity_elements(exact_fit, "ate")


class TestTheRestOfTheFacade:
    """The entry points that reach the bound through a name, rather than computing it.

    Each takes ``estimand="ate"`` by default, so each is a place a ``K``-armed fit could
    have been left answering for a parameter it does not report.
    """

    def test_the_contour_grid_takes_a_contrast(self, exact_fit: Any) -> None:
        grid = exact_fit.sensitivity.contour("ate[high vs low]", grid_size=3)
        assert grid.shape == (9, 3)

    def test_an_ambiguous_default_is_refused_by_name_rather_than_guessed(
        self, exact_fit: Any
    ) -> None:
        """``"ate"`` is not a parameter here, and which one to answer for is not the
        facade's call to make.

        This fit reports seven arm-indexed linear parameters.  Substituting the first --
        which is what filling the gap by position amounts to -- answers about whichever
        one the report happens to order first, with nothing in the returned bound to say
        which.  The refusal instead names every parameter the bound is available for, and
        a combined report carries that same sentence rather than a silent number.
        """
        with pytest.raises(ValueError, match=r"available for .*'ate\[high vs low\]'"):
            exact_fit.sensitivity.omitted_confounding()

        item = exact_fit.sensitivity.run_all()["omitted_confounding"]
        assert item.status == "unavailable"
        assert "ate[high vs low]" in item.detail

    def test_the_benchmark_refits_and_calibrates_for_one_contrast(self, missing_fit: Any) -> None:
        calibrated = missing_fit.sensitivity.benchmark(["W1"], estimand="ate[mid vs low]")
        assert calibrated.estimand == "ate[mid vs low]"
        assert 0.0 <= calibrated.cf_y <= 1.0 and 0.0 <= calibrated.cf_d <= 1.0


class TestTheBoundOnAFitWithNoArmsToContrast:
    def test_a_regime_fit_is_refused_by_name(self) -> None:
        from cleverly.interventions import Static

        result = (
            TMLE(
                outcome_learner=law.OracleMultiOutcome(),
                treatment_learner=law.OracleMultiTreatment(),
                cross_fit=False,
                interventions=(Static("low", name="low"), Static("high", name="high")),
                reference="low",
                simultaneous=False,
                random_state=0,
            )
            .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
            .single()
        )
        with pytest.raises(ValueError, match="reports none"):
            sensitivity_elements(result, "ate")


# ----------------------------------------------------------------------------- the tilt

#: ``P(Delta = 1 | A = a, W)`` by arm *code*, which the stub learner below returns
#: exactly. Code 0 is "high": its outcomes are never missing, which is what makes the
#: invariance test below possible.
OBSERVED_PROBABILITY: tuple[float, ...] = (1.0, 0.75, 0.6)


class ArmMissingness(BaseEstimator):
    """``P(Delta = 1 | A, W)`` depending on the arm alone, and known exactly.

    Fitted on ``[A, W]``, which for a three-armed treatment is ``K - 1`` drop-first
    indicators followed by the covariates -- so the arm is decoded the way
    :class:`tests.discrete_law_multi.OracleMultiOutcome` decodes it, not read off a
    single column.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ArmMissingness:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        indicators = design[:, :2]
        code = np.where(indicators.any(axis=1), indicators.argmax(axis=1) + 1, 0)
        p = np.asarray(OBSERVED_PROBABILITY, dtype=float)[code]
        return np.column_stack([1.0 - p, p])


def _missing_frame(n: int = 900, seed: int = 3) -> pd.DataFrame:
    """A three-armed process whose outcomes go missing at a rate set by the arm alone."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    linear = np.column_stack([np.zeros(n), 0.8 * w[:, 0], -0.7 * w[:, 1]])
    probability = np.exp(linear)
    probability /= probability.sum(axis=1, keepdims=True)
    code = np.array([rng.choice(3, p=row) for row in probability])
    outcome = 0.5 * code + w[:, 0] - 0.3 * w[:, 1] + rng.normal(scale=0.5, size=n)
    observed = rng.random(n) < np.asarray(OBSERVED_PROBABILITY)[code]
    return pd.DataFrame(
        {
            "W1": w[:, 0],
            "W2": w[:, 1],
            "A": np.array(["high", "low", "mid"])[code],
            "Y": np.where(observed, outcome, np.nan),
            "Delta": observed.astype(float),
        }
    )


@pytest.fixture(scope="module")
def missing_fit() -> Any:
    """A three-armed fit with missing outcomes, reporting every tiltable parameter."""
    return (
        TMLE(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            missingness_learner=ArmMissingness(),
            n_folds=5,
            learner_folds=3,
            estimands=("ey", "ate", "att", "atc"),
            reference=REFERENCE,
            simultaneous=False,
            random_state=0,
        )
        .fit(
            _missing_frame(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            delta="Delta",
        )
        .single()
    )


def _curve(result: Any, gamma: list[float], **kwargs: Any) -> dict[tuple[float, str], float]:
    """``{(gamma, estimand): psi}`` from the tidy frame the tilt returns."""
    frame = missingness_tilt(result, gamma, **kwargs)
    return {
        (float(g), str(name)): float(psi)
        for g, name, psi in zip(frame["gamma"], frame["estimand"], frame["psi"], strict=True)
    }


class TestTheTiltAtThreeArms:
    def test_every_reported_parameter_is_tiltable(self, missing_fit: Any) -> None:
        # Including the per-arm means, which are the *only* thing a default multi-arm
        # fit reports: a filter written in terms of ey1/ey0 would leave this fit with
        # nothing to tilt at all.
        curve = _curve(missing_fit, [0.0])
        assert {name for _, name in curve} == set(missing_fit.estimates)

    def test_no_tilt_reproduces_every_reported_estimate(self, missing_fit: Any) -> None:
        """``gamma = 0`` is the MAR analysis, at every arm and every group.

        The conditional effects are where a three-armed treatment bites: their average is
        weighted by ``1{A = c}``, and the two-armed implementation used the treatment
        column itself as that indicator -- which with codes 0, 1, 2 weights the third arm
        twice and conditions on nothing recognisable.
        """
        curve = _curve(missing_fit, [0.0])
        for name in missing_fit.estimates:
            assert curve[(0.0, name)] == pytest.approx(missing_fit.psi(name), rel=1e-12)

    def test_an_arm_with_no_missingness_is_left_exactly_where_it_was(
        self, missing_fit: Any
    ) -> None:
        """``pi_a = 1`` leaves nothing to tilt, whatever gamma the other arms take.

        The sharpest available statement that the mechanism is read at the arm the
        parameter is about: an implementation reading column 0 of ``missingness`` for
        every arm passes every other test in this class and fails this one, because the
        arm that is never missing is code 0 and the others are not.
        """
        gamma = [-1.5, 0.0, 1.5]
        curve = _curve(missing_fit, gamma)
        for value in gamma:
            assert curve[(value, "ey[high]")] == pytest.approx(
                missing_fit.psi("ey[high]"), rel=1e-12
            )
        # And the other two arms do move, so the invariance above is a property of this
        # arm rather than of the tilt.
        for name in ("ey[low]", "ey[mid]"):
            assert curve[(1.5, name)] > curve[(0.0, name)] > curve[(-1.5, name)]

    def test_the_tipping_point_reads_a_multi_arm_name(self, missing_fit: Any) -> None:
        name = "ate[mid vs low]"
        tipping = tipping_gamma(missing_fit, name)
        assert tipping is None or isinstance(tipping, float)


class TestTheDirectionOfTheTilt:
    def test_all_ones_is_the_shared_gamma(self, missing_fit: Any) -> None:
        levels = {"low": 1.0, "mid": 1.0, "high": 1.0}
        shared = _curve(missing_fit, [0.8])
        declared = _curve(missing_fit, [0.8], arm_gamma=levels)
        assert declared == shared

    def test_an_arm_given_zero_stays_at_its_mar_value(self, missing_fit: Any) -> None:
        # The point of the keyword: dropout after one arm need not mean what dropout
        # after another does, and "not at all" is the clearest case of that.
        curve = _curve(missing_fit, [1.0], arm_gamma={"low": 0.0, "mid": 1.0, "high": 1.0})
        assert curve[(1.0, "ey[low]")] == pytest.approx(missing_fit.psi("ey[low]"), rel=1e-12)
        assert curve[(1.0, "ey[mid]")] != pytest.approx(missing_fit.psi("ey[mid]"), rel=1e-12)

    def test_the_sign_is_per_arm(self, missing_fit: Any) -> None:
        """Opposite multipliers move the two arms in opposite directions.

        A shared gamma cannot express this, and it is the case an analyst reaches for:
        the unobserved outcomes are worse than they look in one arm and better in the
        other.
        """
        curve = _curve(missing_fit, [1.0], arm_gamma={"low": 1.0, "mid": -1.0, "high": 1.0})
        assert curve[(1.0, "ey[low]")] > missing_fit.psi("ey[low]")
        assert curve[(1.0, "ey[mid]")] < missing_fit.psi("ey[mid]")

    def test_every_arm_must_be_named(self, missing_fit: Any) -> None:
        with pytest.raises(ValueError, match="must name every arm"):
            missingness_tilt(missing_fit, [1.0], arm_gamma={"low": 1.0, "mid": 1.0})

    def test_an_unknown_level_is_refused(self, missing_fit: Any) -> None:
        with pytest.raises(ValueError, match="not a level of A"):
            missingness_tilt(
                missing_fit, [1.0], arm_gamma={"low": 1.0, "mid": 1.0, "high": 1.0, "nope": 1.0}
            )


class TestTheTiltReportsWhatEachArmReceived:
    def test_a_column_per_arm_carries_its_own_tilt(self, missing_fit: Any) -> None:
        """The direction has to survive into the frame, not just the call.

        A curve read back off disk, or handed to a plot, cannot say what it swept if the
        only record of the direction is the keyword it was passed in.
        """
        levels = {"low": 1.0, "mid": -0.5, "high": 0.0}
        frame = missingness_tilt(missing_fit, [0.0, 2.0], arm_gamma=levels)
        by_row = list(zip(frame["gamma"], frame["gamma[low]"], frame["gamma[mid]"], strict=True))
        assert (2.0, 2.0, -1.0) in by_row
        assert (0.0, 0.0, -0.0) in by_row
        assert set(frame["gamma[high]"]) == {0.0}

    def test_the_default_direction_reports_the_shared_gamma(self, missing_fit: Any) -> None:
        frame = missingness_tilt(missing_fit, [1.5])
        for label in ("low", "mid", "high"):
            assert set(frame[f"gamma[{label}]"]) == {1.5}


class TestTheEValueAtThreeArms:
    @pytest.fixture(scope="class")
    def binary_outcome_fit(self) -> Any:
        """A three-armed fit on the law, whose binary outcome makes a risk ratio real."""
        return (
            TMLE(
                outcome_learner=law.OracleMultiOutcome(),
                treatment_learner=law.OracleMultiTreatment(),
                cross_fit=False,
                estimands=("ey", "ate", "rr", "or"),
                reference=REFERENCE,
                simultaneous=False,
                random_state=0,
            )
            .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
            .single()
        )

    def test_a_ratio_named_for_its_arms_reaches_the_exact_formula(
        self, binary_outcome_fit: Any
    ) -> None:
        from cleverly.sensitivity import evalue_from_rr

        name = "rr[high vs low]"
        report = binary_outcome_fit.sensitivity.evalue(name)
        assert report.estimand == name
        assert not report.approximate
        assert report.point == pytest.approx(evalue_from_rr(binary_outcome_fit.psi(name)))

    def test_the_default_picks_a_reported_ratio(self, binary_outcome_fit: Any) -> None:
        # "rr" is not a parameter of this fit; the *stem* is what the preference order is
        # over, and picking the first bare name that happened to exist found none at all.
        assert binary_outcome_fit.sensitivity.evalue().estimand.startswith("rr[")

    def test_a_risk_difference_divides_by_the_reference_arms_own_mean(
        self, binary_outcome_fit: Any
    ) -> None:
        """The baseline is the arm the contrast is taken against, found by arm.

        Hard-coding ``ey0`` would divide by ``ey[high]`` here -- the arm whose *code* is
        zero -- while the contrast is against ``"low"``, so the conversion would answer
        for a pair of parameters that never appear together in any report.
        """
        report = binary_outcome_fit.sensitivity.evalue("ate[high vs low]")
        baseline = binary_outcome_fit.psi("ey[low]")
        expected = (baseline + binary_outcome_fit.psi("ate[high vs low]")) / baseline
        assert report.risk_ratio == pytest.approx(expected, rel=1e-12)
        assert report.approximate

    def test_a_level_is_refused_by_name(self, binary_outcome_fit: Any) -> None:
        with pytest.raises(ValueError, match="written for a contrast of two arms"):
            binary_outcome_fit.sensitivity.evalue("ey[low]")

    def test_two_arms_divide_by_the_declared_reference_too(self) -> None:
        """The same rule at two arms, where the old constant was right by default only.

        With ``reference=1`` the ``ate`` is ``E[Y^1|A] - E[Y^0|A]`` reversed --
        ``E[Y^0] - E[Y^1]`` -- so the baseline the conversion needs is ``ey1``. Dividing
        by ``ey0``, as the constant did, converts the difference against one arm into a
        ratio against the other.
        """
        from cleverly.datasets import make_binary_outcome

        frame, _ = make_binary_outcome(n=800, seed=5)
        fit = (
            TMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                estimands=("ate", "ey1", "ey0"),
                reference=1,
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )
        report = fit.sensitivity.evalue("ate")
        baseline = fit.psi("ey1")
        assert report.risk_ratio == pytest.approx((baseline + fit.psi("ate")) / baseline, rel=1e-12)

    def test_a_missing_baseline_says_which_mean_it_needs(self) -> None:
        fit = (
            TMLE(
                outcome_learner=law.OracleMultiOutcome(),
                treatment_learner=law.OracleMultiTreatment(),
                cross_fit=False,
                estimands=("ate",),
                reference=REFERENCE,
                simultaneous=False,
                random_state=0,
            )
            .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
            .single()
        )
        with pytest.raises(ValueError, match="is contrasted against"):
            fit.sensitivity.evalue("ate[high vs low]")


class TestTheTiltFollowsTheDeclaredReference:
    """Two arms, reference ``1``: the same rule, where it is easiest to get backwards.

    ``reference=1`` makes ``ate`` the contrast ``E[Y^0] - E[Y^1]`` and ``att`` the effect
    among the units that received arm ``0``. Reading the arms off the parameter is what
    keeps the tilt reporting *those*; the implementation this replaced hard-coded arm 1
    against arm 0 and used the treatment column as the ATT's indicator, so both came back
    for the other contrast with no sign that anything had been substituted.
    """

    @pytest.fixture(scope="class")
    def flipped(self) -> Any:
        from cleverly.datasets import make_missing_outcome

        frame, _ = make_missing_outcome(n=800, seed=11)
        return (
            TMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                missingness_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=5,
                learner_folds=3,
                estimands=("ate", "att", "atc"),
                reference=1,
                simultaneous=False,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
            )
            .single()
        )

    @pytest.mark.parametrize("name", ["ate", "att", "atc"])
    def test_no_tilt_reproduces_the_reported_estimate(self, flipped: Any, name: str) -> None:
        curve = _curve(flipped, [0.0], estimands=[name])
        assert curve[(0.0, name)] == pytest.approx(flipped.psi(name), rel=1e-12)
