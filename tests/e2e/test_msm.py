"""A working-model fit, end to end.

The exact claims about ``beta`` -- that its influence curve is the efficient one, that its
remainder is second-order -- live in ``tests/unit/test_influence_gateaux_msm.py`` and
``tests/unit/test_remainder_msm.py``, on a law a sample realises exactly.  What is left for
this module is what only a real fit can show: that the estimator recovers the projection
from data, that the score equation it reports having solved is solved, and that the
surrounding machinery -- positivity, contrasts, serialization -- works on this axis without
having been told about it.

``library="glm"`` throughout, per ``CLAUDE.md``: nothing here is about flexible learning,
and the outcome regression is correctly specified for this process anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE, load
from cleverly.datasets import make_multi_arm
from cleverly.interventions import Shift, Static
from cleverly.msm import MSM

#: ``make_multi_arm`` labels its arms, so the dose has to be declared rather than read off
#: the labels -- which is the refusal ``MSM.linear`` makes, taken up.
DOSE = {"low": 0.0, "medium": 1.0, "high": 2.0}

N = 2000
SEED = 0


def dose_response() -> MSM:
    """``m(a) = beta0 + beta1 * dose(a)``, uniform weights."""
    return MSM(
        design=lambda arm, frame: np.column_stack(
            [np.ones(len(frame)), np.full(len(frame), DOSE[arm])]
        ),
        terms=("(intercept)", "dose"),
    )


def saturated() -> MSM:
    """One indicator per arm: the working model that summarises nothing."""
    return MSM(
        design=lambda arm, frame: np.column_stack(
            [np.full(len(frame), float(arm == label)) for label in DOSE]
        ),
        terms=tuple(f"arm[{label}]" for label in DOSE),
    )


def projection(means: dict[str, float]) -> np.ndarray:
    """The population ``beta``: an ordinary least squares of the arm means on the dose.

    Written out here rather than taken from the library, and from the *population* arm
    means the generator reports -- so this is the truth the fit has to find, not a
    restatement of what it found.
    """
    design = np.column_stack([np.ones(len(DOSE)), [DOSE[label] for label in DOSE]])
    target = np.array([means[f"ey[{label}]"] for label in DOSE])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(beta)


@pytest.fixture(scope="module")
def fitted():
    frame, truth = make_multi_arm(n=N, seed=SEED)
    result = (
        TMLE(
            msm=dose_response(),
            outcome_learner="glm",
            treatment_learner="glm",
            random_state=SEED,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )
    return result, truth


class TestItRecoversTheProjection:
    def test_the_reported_parameters_are_the_declared_terms(self, fitted) -> None:
        result, _ = fitted
        assert list(result.estimates) == ["msm[(intercept)]", "msm[dose]"]

    def test_each_coefficient_covers_its_population_value(self, fitted) -> None:
        """Two standard errors of the closed-form projection, on one fit.

        A coverage *claim* would need replications and belongs in the slow tier; this is
        the weaker statement that the point estimate is in the right place at all, which
        is what catches a wrong scale, a dropped weight or a transposed design.
        """
        result, truth = fitted
        expected = projection(truth)
        for value, name in zip(expected, ("msm[(intercept)]", "msm[dose]"), strict=True):
            estimate = result.estimates[name]
            assert abs(estimate.psi - value) < 2.0 * estimate.std_error

    def test_the_working_model_is_wrong_and_that_is_fine(self, fitted) -> None:
        """The process's effect is not linear in the dose -- see ``multi_arm_dgp``.

        Worth asserting rather than assuming, because the recovery check above would be
        much weaker if the model happened to be correct: it is the misspecified case that
        distinguishes a projection from a regression.
        """
        _, truth = fitted
        beta = projection(truth)
        fitted_means = beta[0] + beta[1] * np.array(list(DOSE.values()))
        actual = np.array([truth[f"ey[{label}]"] for label in DOSE])
        assert np.max(np.abs(fitted_means - actual)) > 0.05

    def test_the_score_equation_is_solved(self, fitted) -> None:
        result, _ = fitted
        check = result.validation.score_check()
        assert check.passed, check.summary()

    def test_targeting_moved_the_coefficients(self, fitted) -> None:
        """Otherwise the plug-in would be the untargeted projection wearing its name."""
        result, _ = fitted
        assert np.max(np.abs(result.fluctuations["msm"].epsilon)) > 1e-8


class TestASaturatedModelIsTheArmReport:
    """The end-to-end form of the identity ``tests/unit/test_msm_submodel.py`` pins.

    One indicator per arm summarises nothing, so the projection has to come back as the
    counterfactual means themselves -- point estimate *and* influence curve, against a
    plain fit on the same data with the same seed.  It exercises the whole path at once:
    the design, the weights, ``M^-1``, the unscaling, and the fluctuation.

    ``approx`` rather than exact equality, and only here: the two paths reach the same
    number by different associations of the same floating-point arithmetic -- one averages
    then unscales, the other unscales then averages.
    """

    @pytest.fixture(scope="class")
    def pair(self):
        frame, _ = make_multi_arm(n=N, seed=SEED)
        common = {
            "outcome_learner": "glm",
            "treatment_learner": "glm",
            "random_state": SEED,
            "simultaneous": False,
        }
        arms = TMLE(estimands=("ey",), **common).fit(frame, outcome="Y", treatment="A").single()
        model = TMLE(msm=saturated(), **common).fit(frame, outcome="Y", treatment="A").single()
        return arms, model

    def test_the_point_estimates_agree(self, pair) -> None:
        arms, model = pair
        for label in DOSE:
            assert model.estimates[f"msm[arm[{label}]]"].psi == pytest.approx(
                arms.estimates[f"ey[{label}]"].psi, rel=1e-12
            )

    def test_the_influence_curves_agree(self, pair) -> None:
        arms, model = pair
        for label in DOSE:
            np.testing.assert_allclose(
                model.estimates[f"msm[arm[{label}]]"].influence_curve,
                arms.estimates[f"ey[{label}]"].influence_curve,
                rtol=1e-11,
                atol=1e-13,
            )


class TestTheSurroundingMachineryWorks:
    def test_contrasts_come_from_the_joint_influence_curve(self, fitted) -> None:
        """No refit: a difference of coefficients is a delta-method functional of them."""
        result, _ = fitted
        contrast = result.contrast(lambda psi: psi[0] - psi[1], ["msm[dose]", "msm[(intercept)]"])
        assert contrast.psi == pytest.approx(
            result.estimates["msm[dose]"].psi - result.estimates["msm[(intercept)]"].psi
        )
        assert np.isfinite(contrast.std_error)

    def test_positivity_reports_on_the_arms(self, fitted) -> None:
        """The counterfactuals are still the arms, so the arm-level report still applies."""
        result, _ = fitted
        report = result.sensitivity.positivity()
        assert set(report.propensity_quantiles) >= {f"g[{label}]" for label in DOSE}

    def test_a_truncation_sweep_retargets_without_refitting(self, fitted) -> None:
        result, _ = fitted
        curve = result.sensitivity.truncation_curve(bounds=[0.01, 0.05])
        assert set(curve["estimand"]) == {"msm[(intercept)]", "msm[dose]"}

    def test_a_round_trip_leaves_every_retargeted_analysis_identical(
        self, fitted, tmp_path
    ) -> None:
        """The evaluated design is stored, so a loaded fit targets the same model."""
        result, _ = fitted
        reloaded = load(result.save(tmp_path / "msm.npz"))
        for name, estimate in result.estimates.items():
            assert reloaded.estimates[name].psi == estimate.psi
            np.testing.assert_array_equal(
                reloaded.estimates[name].influence_curve, estimate.influence_curve
            )
        assert reloaded.nuisance.msm is not None
        np.testing.assert_array_equal(reloaded.nuisance.msm.design, result.nuisance.msm.design)
        assert reloaded.validation.score_check().passed

    def test_a_reloaded_fit_cannot_be_refit(self, fitted, tmp_path) -> None:
        """A design is a callable, so ``refute``/``benchmark`` need the estimator back."""
        result, _ = fitted
        reloaded = load(result.save(tmp_path / "msm_refit.npz"))
        with pytest.raises(ValueError, match="working model's design"):
            reloaded.estimator.refit(reloaded.data)


class TestTheAxisIsExclusive:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"interventions": [Static("low")]}, "interventions="),
            ({"shifts": [Shift(0.5, cap=None)]}, "shifts="),
        ],
    )
    def test_msm_cannot_be_combined_with_another_axis(self, kwargs, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            TMLE(msm=dose_response(), **kwargs)

    def test_a_reference_is_refused_rather_than_ignored(self) -> None:
        """Coefficients are not contrasts, so there is nothing to be a reference for."""
        with pytest.raises(ValueError, match="nothing for it to be a reference for"):
            TMLE(msm=dose_response(), reference="low")

    def test_an_arm_indexed_estimand_is_refused_on_a_working_model_fit(self) -> None:
        frame, _ = make_multi_arm(n=200, seed=SEED)
        estimator = TMLE(msm=dose_response(), estimands=("ate",), outcome_learner="glm")
        with pytest.raises(ValueError):
            estimator.fit(frame, outcome="Y", treatment="A")
