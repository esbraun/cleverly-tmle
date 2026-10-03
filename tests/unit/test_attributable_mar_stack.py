r"""The joint natural-course fit is the stack of the two shipped parents, bit for bit.

Decision D1 of the F20 contract targets each parent separately from one initial fit: the
shipped natural-course fluctuation and the shipped ``mean`` fluctuation.  Each coordinate
of the stack is then the shipped estimator, so a joint fit must report the scalar
``ey_obs`` fit's point and curve and the arm-only fit's arm means exactly.  The learners
here are deterministic, the folds are package-generated from one ``random_state``, and
every role draws its own seed, so fitting ``g`` in the joint fit moves neither ``Q`` nor
``pi``.

The stacked joint fit reports every estimate from the ``mean`` context under the centered
covariance rule, where the scalar stacked ``ey_obs`` keeps the raw second moment (D5).
The two variances satisfy an exact algebraic relation, which the last class pins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.inference.influence import ArmMean
from cleverly.targets import TargetContext, targets_for


def _frame(*, arms: int = 2, n: int = 400, seed: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.4, size=n).astype(float)
    if arms == 2:
        a = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * w1)), size=n).astype(float)
    else:
        a = rng.integers(0, arms, size=n).astype(float)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.3 + 0.4 * a + 0.5 * w1 - 0.3 * w2))), size=n)
    delta = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.0 + 0.5 * w1 - 0.4 * a))), size=n)
    outcome = np.where(delta == 1, y.astype(float), np.nan)
    return pd.DataFrame({"Y": outcome, "A": a, "W1": w1, "W2": w2, "Delta": delta.astype(float)})


def _estimator(estimands: tuple[str, ...], *, cross_fit: bool, **extra: Any) -> TMLE:
    settings: dict[str, Any] = {"simultaneous": False, **extra}
    return TMLE(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        missingness_learner=LogisticRegression(),
        estimands=estimands,
        cross_fit=cross_fit,
        n_folds=10,
        stratify_folds="none",
        random_state=11,
        max_iter=100,
        tol=1e-12,
        **settings,
    )


def _fit(estimands: tuple[str, ...], *, cross_fit: bool, arms: int = 2, **extra: Any) -> Any:
    return (
        _estimator(estimands, cross_fit=cross_fit, **extra)
        .fit(
            _frame(arms=arms),
            outcome="Y",
            treatment="A",
            covariates=("W1", "W2"),
            delta="Delta",
        )
        .single()
    )


def _same(left: Any, right: Any) -> None:
    assert left.psi == right.psi
    np.testing.assert_array_equal(left.influence_curve, right.influence_curve)


ROUTES = [pytest.param(False, id="in-sample"), pytest.param(True, id="stacked")]


class TestEachCoordinateIsTheShippedEstimator:
    @pytest.mark.parametrize("cross_fit", ROUTES)
    def test_the_joint_ey_obs_is_the_scalar_fit(self, cross_fit: bool) -> None:
        joint = _fit(("ey_obs", "ey0", "par"), cross_fit=cross_fit)
        scalar = _fit(("ey_obs",), cross_fit=cross_fit)
        _same(joint["ey_obs"], scalar["ey_obs"])

    @pytest.mark.parametrize("cross_fit", ROUTES)
    def test_the_joint_reference_mean_is_the_arm_only_fit(self, cross_fit: bool) -> None:
        joint = _fit(("ey_obs", "ey0", "par"), cross_fit=cross_fit)
        arm_only = _fit(("ey0",), cross_fit=cross_fit)
        _same(joint["ey0"], arm_only["ey0"])
        assert joint["ey0"].variance == arm_only["ey0"].variance

    @pytest.mark.parametrize("cross_fit", ROUTES)
    def test_every_arm_mean_of_a_three_arm_fit_is_the_arm_only_fit(self, cross_fit: bool) -> None:
        joint = _fit(("ey_obs", "ey", "par", "paf"), cross_fit=cross_fit, arms=3, reference=1.0)
        arm_only = _fit(("ey",), cross_fit=cross_fit, arms=3, reference=1.0)
        scalar = _fit(("ey_obs",), cross_fit=cross_fit, arms=3)
        means = [name for name in arm_only.estimates if name.startswith("ey[")]
        assert len(means) == 3
        for name in means:
            _same(joint[name], arm_only[name])
        _same(joint["ey_obs"], scalar["ey_obs"])
        # M8's witness: the reference is the declared arm, not the lowest one.
        assert joint.psi("par[1.0]") == joint.psi("ey_obs") - joint.psi("ey[1.0]")
        assert joint.psi("par[1.0]") != joint.psi("ey_obs") - joint.psi("ey[0.0]")

    def test_att_beside_par_is_the_att_only_fit(self) -> None:
        joint = _fit(("par", "att"), cross_fit=False)
        alone = _fit(("att",), cross_fit=False)
        _same(joint["att"], alone["att"])

    @pytest.mark.parametrize("cross_fit", ROUTES)
    def test_par_is_the_difference_of_the_coordinates(self, cross_fit: bool) -> None:
        joint = _fit(("ey_obs", "ey0", "par", "paf"), cross_fit=cross_fit)
        observed, reference = joint["ey_obs"], joint["ey0"]
        assert joint.psi("par") == observed.psi - reference.psi
        np.testing.assert_array_equal(
            joint["par"].influence_curve, observed.influence_curve - reference.influence_curve
        )
        assert joint.psi("paf") == pytest.approx(1.0 - reference.psi / observed.psi, abs=1e-15)

    def test_the_natural_course_iteration_builds_no_target(self) -> None:
        """M9's witness: one copy of each mean-group target, from the ``mean`` context.

        Away from the truth the natural-course fluctuation's targeted regression differs
        from the ``mean`` group's, so a copy of ``ey0`` built from the natural-course
        context would differ from the arm-only fit.
        """
        joint = _fit(("ey_obs", "ey0"), cross_fit=False)
        arm_only = _fit(("ey0",), cross_fit=False)
        natural = joint.fluctuations["natural_course"].targeted.arms[0.0]
        mean = joint.fluctuations["mean"].targeted.arms[0.0]
        assert np.max(np.abs(np.asarray(natural) - np.asarray(mean))) > 1e-4
        _same(joint["ey0"], arm_only["ey0"])

    def test_the_observed_mean_is_not_read_from_the_mean_group(self) -> None:
        """M7's witness: ``ey_obs`` is not the mean group's targeted regression at A."""
        joint = _fit(("ey_obs", "ey0", "par"), cross_fit=False)
        realised = float(np.mean(joint.fluctuations["mean"].targeted.observed))
        assert abs(joint.psi("ey_obs") - realised) > 1e-5


class TestTheStackedCovarianceRule:
    def test_the_joint_fit_declares_one_centered_rule(self) -> None:
        joint = _fit(("ey_obs", "ey0", "par", "paf"), cross_fit=True)
        assert {estimate.covariance_rule for estimate in joint.estimates.values()} == {"centered"}

    def test_the_scalar_stacked_fit_keeps_the_second_moment(self) -> None:
        scalar = _fit(("ey_obs",), cross_fit=True)
        assert scalar["ey_obs"].covariance_rule == "second_moment"

    def test_the_two_rules_satisfy_the_exact_relation(self) -> None:
        """``var_centered = var_sm * n / (n - 1) - mean(ic)**2 / (n - 1)`` (D5)."""
        joint = _fit(("ey_obs", "ey0", "par", "paf"), cross_fit=True)
        scalar = _fit(("ey_obs",), cross_fit=True)
        curve = np.asarray(scalar["ey_obs"].influence_curve)
        n = curve.size
        expected = scalar["ey_obs"].variance * n / (n - 1) - float(np.mean(curve)) ** 2 / (n - 1)
        assert joint["ey_obs"].variance == pytest.approx(expected, rel=1e-12)

    def test_the_default_band_builds_on_a_stacked_joint_fit(self) -> None:
        """M10's witness: a band refuses the second-moment rule."""
        result = (
            _estimator(("ey_obs", "ey0", "par", "paf"), cross_fit=True, simultaneous=True)
            .fit(_frame(), outcome="Y", treatment="A", covariates=("W1", "W2"), delta="Delta")
            .single()
        )
        assert result.simultaneous is not None


class TestTheCompleteDataReduction:
    @pytest.mark.parametrize("name", ["par", "paf"])
    def test_an_identically_one_response_takes_the_complete_data_path(self, name: str) -> None:
        frame = _frame().assign(Y=lambda f: f["Y"].fillna(0.0), Delta=1.0)
        declared = (
            _estimator((name,), cross_fit=False)
            .fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2"), delta="Delta")
            .single()
        )
        plain = (
            _estimator((name,), cross_fit=False)
            .fit(frame.drop(columns="Delta"), outcome="Y", treatment="A", covariates=("W1", "W2"))
            .single()
        )
        _same(declared[name], plain[name])
        assert tuple(declared.fluctuations) == ("mean",)


class TestTheDenominatorAndTheHook:
    @staticmethod
    def _context(natural_course: ArmMean | None) -> TargetContext:
        result = _fit(("ey0",), cross_fit=False)
        fluctuation = result.fluctuations["mean"]
        submodel = result.estimator._submodel(
            result.data,
            result.nuisance,
            "mean",
            result.config.g_bounds,
            None,
            None,
            None,
            0.0,
        )
        return TargetContext(
            scaled=result.nuisance.scaler.scale(result.data.outcome),
            targeted=fluctuation.targeted,
            submodel=submodel,
            treatment=result.data.treatment,
            weights=result.data.weights,
            observed=result.data.observed,
            scaler=result.nuisance.scaler,
            n=result.data.n,
            alpha_sig=0.05,
            arms=(0.0, 1.0),
            arm_labels={0.0: 0.0, 1.0: 1.0},
            reference=0.0,
            natural_course=natural_course,
        )

    def test_a_zero_natural_course_risk_refuses_paf(self) -> None:
        context = self._context(ArmMean(0.0, np.zeros(400)))
        (target,) = targets_for("mean", ("paf",))
        with pytest.raises(ValueError, match="paf is undefined when the observed outcome risk"):
            target.build(context)
        assert target.undefined_when
        (par,) = targets_for("mean", ("par",))
        assert len(par.build(context)) == 1

    def test_missing_outcomes_without_the_hook_raise_an_internal_error(self) -> None:
        context = self._context(None)
        with pytest.raises(ValueError, match="observed_mean needs the natural-course"):
            _ = context.observed_mean


class TestRetargetReplaysBothFluctuations:
    def test_retarget_at_the_fitted_bounds_reproduces_the_fit(self) -> None:
        result = _fit(("ey_obs", "ey0", "par", "paf"), cross_fit=False)
        estimates, fluctuations = result.estimator.retarget(
            result.data,
            result.nuisance,
            estimands=("ey_obs", "ey0", "par", "paf"),
            g_bounds=result.config.g_bounds,
        )
        assert tuple(fluctuations) == ("natural_course", "mean")
        for name, estimate in result.estimates.items():
            _same(estimates[name], estimate)
