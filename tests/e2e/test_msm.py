"""A working-model fit, end to end.

The exact claims about ``beta`` -- that its influence curve is the efficient one, that its
remainder is second-order -- live in ``tests/unit/test_influence_gateaux_msm.py`` and
``tests/unit/test_remainder_msm.py``, on a law a sample realises exactly.  What is left for
this module is what only a real fit can show: that the estimator recovers the projection
from data, that the score equation it reports having solved is solved, and that the
surrounding machinery -- positivity, contrasts, serialization -- works on this axis without
having been told about it.

Direct linear and logistic models are used throughout: nothing here is about flexible
learning, and the outcome regression is correctly specified for this process anyway.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.linear_model
from scipy.special import expit

from cleverly import load
from cleverly.datasets import make_binary_outcome, make_multi_arm
from cleverly.datasets.synthetic import MultiArmDGP
from cleverly.estimators import TMLE
from cleverly.interventions import Shift, Static
from cleverly.msm import MSM
from tests.conftest import FAST_KWARGS

#: ``make_multi_arm`` labels its arms, so the dose has to be declared rather than read off
#: the labels -- which is the refusal ``MSM.linear`` makes, taken up.
DOSE = {"low": 0.0, "medium": 1.0, "high": 2.0}

N = 2000
SEED = 0


def _dose_design(arm, frame):
    return np.column_stack([np.ones(len(frame)), np.full(len(frame), DOSE[arm])])


def _saturated_design(arm, frame):
    return np.column_stack([np.full(len(frame), float(arm == label)) for label in DOSE])


def dose_response(link: str = "identity") -> MSM:
    """``link(m(a)) = beta0 + beta1 * dose(a)``, uniform weights."""
    return MSM(
        design=_dose_design,
        terms=("(intercept)", "dose"),
        link=link,  # type: ignore[arg-type]
    )


def saturated(link: str = "identity") -> MSM:
    """One indicator per arm: the working model that summarises nothing."""
    return MSM(
        design=_saturated_design,
        terms=tuple(f"arm[{label}]" for label in DOSE),
        link=link,  # type: ignore[arg-type]
    )


def make_binary_multi_arm(n: int, seed: int):
    """Three arms and a *binary* outcome, which is what a log or logit model needs.

    ``make_multi_arm`` has a Gaussian outcome that goes negative, so ``exp(beta' phi)``
    could not reach it and ``MSMSet.evaluate`` refuses the link -- correctly.  The process
    here is that one with the mean put through ``expit``: the same confounding, the same
    three labelled arms, the same non-linear step between them, so the working model is
    still wrong and ``beta`` is still a projection rather than a truth.
    """
    step = np.array([0.0, 1.0, 2.4])
    dgp = MultiArmDGP(
        name="multi_arm_binary",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        arm_logits=lambda w: np.column_stack(
            [np.zeros(w.shape[0]), 0.8 * w[:, 0] - 0.4 * w[:, 1], -0.5 * w[:, 0] + 0.8 * w[:, 1]]
        ),
        outcome_mean=lambda w, arm: expit(-0.4 + 0.6 * step[arm] + w[:, 0] - 0.5 * w[:, 1]),
        labels=("low", "medium", "high"),
        family="binomial",
    )
    return dgp.sample(n, seed=seed)


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


#: The fast tier's estimator settings, with this module's seed.  Taken from
#: ``FAST_KWARGS`` rather than re-spelled: writing ``outcome_learner=sklearn.linear_model.LinearRegression()`` by hand and
#: leaving the fold counts out silently takes the ``TMLE`` defaults of ``n_folds=10,
#: learner_folds=5``, which is twice the fast tier's budget on a three-armed process where
#: the propensity is fitted one-vs-rest.  Nothing here turns on the fold count.
SETTINGS: dict[str, object] = {**FAST_KWARGS, "random_state": SEED}


@pytest.fixture(scope="module")
def fitted():
    frame, truth = make_multi_arm(n=N, seed=SEED)
    result = TMLE(msm=dose_response(), **SETTINGS).fit(frame, outcome="Y", treatment="A").single()
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
        check = result.diagnostics.score_equations()
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
        arms = TMLE(estimands=("ey",), **SETTINGS).fit(frame, outcome="Y", treatment="A").single()
        model = TMLE(msm=saturated(), **SETTINGS).fit(frame, outcome="Y", treatment="A").single()
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


class TestALinkChangesTheParameterAndNotTheMachinery:
    """``link="log"`` and ``link="logit"``, end to end on a binary outcome.

    The exact claims are the oracle's, as above.  What only a real fit can show is that
    the alternation between the projection and the fluctuation converges and leaves both
    equations solved -- and that a *saturated* working model still reproduces the arm
    report, now through the link, which is what says the link is a reparameterisation of
    the same counterfactual means rather than a second estimator.
    """

    @pytest.fixture(scope="class")
    def binary(self):
        return make_binary_multi_arm(n=N, seed=SEED)

    @pytest.fixture(scope="class")
    def arms(self, binary):
        frame, _ = binary
        return TMLE(estimands=("ey",), **SETTINGS).fit(frame, outcome="Y", treatment="A").single()

    @pytest.fixture(scope="class")
    def saturated_fits(self, binary):
        frame, _ = binary
        return {
            link: TMLE(msm=saturated(link=link), **SETTINGS)
            .fit(frame, outcome="Y", treatment="A")
            .single()
            for link in ("log", "logit")
        }

    @pytest.mark.parametrize("link", ["log", "logit"])
    def test_a_saturated_model_is_the_arm_report_through_the_link(
        self, saturated_fits, arms, link: str
    ) -> None:
        """``expit(beta_a)`` -- or ``exp`` -- is ``E[Y(a)]``, to machine precision.

        With one indicator per arm the model fits the means exactly whatever the link, so
        this is an *identity* rather than an approximation, and it exercises the whole
        path at once: the alternation, the covariate's ``dm/deta`` factor, the Jacobian's
        curvature term and the projection's Newton solve.
        """
        inverse = np.exp if link == "log" else lambda eta: 1.0 / (1.0 + np.exp(-eta))
        model = saturated_fits[link]
        for label in DOSE:
            assert inverse(model.estimates[f"msm[arm[{label}]]"].psi) == pytest.approx(
                arms.estimates[f"ey[{label}]"].psi, rel=1e-9
            )

    @pytest.mark.parametrize("link", ["log", "logit"])
    def test_the_influence_curve_is_the_arm_report_s_by_the_delta_method(
        self, saturated_fits, arms, link: str
    ) -> None:
        """``beta = link(psi)``, so ``IC_beta = IC_psi / (dm/deta)`` at the fitted mean."""
        model = saturated_fits[link]
        for label in DOSE:
            psi = arms.estimates[f"ey[{label}]"].psi
            slope = psi if link == "log" else psi * (1.0 - psi)
            np.testing.assert_allclose(
                model.estimates[f"msm[arm[{label}]]"].influence_curve,
                arms.estimates[f"ey[{label}]"].influence_curve / slope,
                rtol=1e-7,
                atol=1e-9,
            )

    @pytest.mark.parametrize("link", ["log", "logit"])
    def test_both_equations_are_solved_at_the_reported_coefficients(self, binary, link) -> None:
        frame, _ = binary
        result = (
            TMLE(msm=dose_response(link=link), **SETTINGS)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        check = result.diagnostics.score_equations()
        assert check.passed, check.summary()
        # The plug-in half is zero by construction and the residual half by the
        # fluctuation; together they are the reported curve's mean.
        for estimate in result.estimates.values():
            assert abs(float(np.mean(estimate.influence_curve))) < 5e-10

    @pytest.mark.parametrize("link", ["log", "logit"])
    def test_the_alternation_converges_and_records_how(self, binary, link) -> None:
        frame, _ = binary
        result = (
            TMLE(msm=dose_response(link=link), **SETTINGS)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        projection = result.fluctuations["msm"].projection
        assert projection is not None and projection.converged
        assert projection.failure is None
        # Fast: beta reaches the covariate only through a smooth factor, so the shift
        # contracts by orders of magnitude per round rather than by a constant factor.
        assert projection.n_outer <= 10
        shifts = [row[2] for row in projection.trace]
        assert shifts[-1] < shifts[0]

    def test_an_identity_link_fit_records_no_projection(self, fitted) -> None:
        """The covariate is free of ``beta`` there, so there is nothing to alternate."""
        result, _ = fitted
        assert result.fluctuations["msm"].projection is None

    def test_a_two_armed_logit_msm_is_the_odds_ratio(self) -> None:
        """The saturated case in the form a reader can check by eye, and the README's.

        Two arms and two terms is saturated, so ``exp(beta_a)`` is not an approximation of
        the marginal odds ratio -- it *is* it, interval and all. A link that was subtly
        wrong in its Jacobian, its covariate or its scale would still produce a plausible
        number here and would not produce this one.
        """
        frame, _ = make_binary_outcome(n=N, seed=SEED)
        model = (
            TMLE(msm=MSM.linear(link="logit"), **SETTINGS)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        plain = TMLE(estimands=("or",), **SETTINGS).fit(frame, outcome="Y", treatment="A").single()
        ratios = model.coefficients(scale="ratio")
        assert float(ratios["psi"][1]) == pytest.approx(plain.estimates["or"].psi, rel=1e-9)
        low, high = plain.estimates["or"].ci
        assert float(ratios["ci_lower"][1]) == pytest.approx(low, rel=1e-9)
        assert float(ratios["ci_upper"][1]) == pytest.approx(high, rel=1e-9)


class TestFoldWiseTargetingGivesEachFoldItsOwnBeta:
    """``targeting_scheme="fold"`` under a link, where the covariate is fold-specific.

    Under a link ``beta`` is one of the coefficients the covariate reads, so the
    fold-specific extension solves it on each fold's own rows. Those rows also fit the
    epsilon used on that fold; what is removed is coupling *between* folds. The pooled
    score stays zero because each fold's score is zero at its own ``beta``.
    """

    @pytest.fixture(scope="class")
    def fold_fit(self):
        frame, _ = make_binary_multi_arm(n=N, seed=SEED)
        return (
            TMLE(msm=dose_response(link="logit"), targeting_scheme="fold", **SETTINGS)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    def test_each_fold_carries_its_own_projection(self, fold_fit) -> None:
        projection = fold_fit.fluctuations["msm"].projection
        assert len(projection.folds) == FAST_KWARGS["n_folds"]
        betas = np.vstack([record.beta for record in projection.folds])
        assert np.max(np.ptp(betas, axis=0)) > 1e-6, "the folds should not agree exactly"

    def test_the_stitched_score_is_still_zero(self, fold_fit) -> None:
        for record in fold_fit.fluctuations["msm"].folds:
            assert np.max(np.abs(record.score)) < 1e-9
        assert fold_fit.diagnostics.score_equations().passed
        for estimate in fold_fit.estimates.values():
            assert abs(float(np.mean(estimate.influence_curve))) < 1e-9

    def test_the_reported_beta_is_the_projection_of_the_stitched_fit(self, fold_fit) -> None:
        """No fold's beta is *the* answer; the report's is, and the record says so."""
        projection = fold_fit.fluctuations["msm"].projection
        reported = np.array([fold_fit.estimates[name].psi for name in fold_fit.estimates])
        np.testing.assert_allclose(projection.beta, reported, rtol=1e-10)


class TestTheExponentiatedView:
    @pytest.fixture(scope="class")
    def fits(self):
        frame, _ = make_binary_multi_arm(n=N, seed=SEED)
        return {
            link: TMLE(msm=dose_response(link=link), **SETTINGS)
            .fit(frame, outcome="Y", treatment="A")
            .single()
            for link in ("log", "logit")
        }

    @pytest.mark.parametrize("link", ["log", "logit"])
    def test_it_exponentiates_the_estimate_and_its_interval(self, fits, link: str) -> None:
        result = fits[link]
        frame = result.coefficients(scale="ratio")
        for row, name in enumerate(result.estimates):
            estimate = result.estimates[name]
            assert frame["psi"][row] == pytest.approx(float(np.exp(estimate.psi)))
            assert frame["ci_lower"][row] == pytest.approx(float(np.exp(estimate.ci[0])))
            assert frame["ci_upper"][row] == pytest.approx(float(np.exp(estimate.ci[1])))
            # The p-value is unchanged: the null is beta = 0, which is ratio = 1.
            assert frame["p_value"][row] == pytest.approx(estimate.pvalue)

    def test_it_names_the_ratio_the_link_actually_gives(self, fits) -> None:
        """An odds ratio reported as a risk ratio would be a wrong number, not a wording."""
        assert list(fits["log"].coefficients(scale="ratio")["scale"]) == [
            "baseline",
            "risk ratio",
        ]
        assert list(fits["logit"].coefficients(scale="ratio")["scale"]) == [
            "baseline",
            "odds ratio",
        ]

    def test_the_link_scale_view_is_the_report(self, fits) -> None:
        frame = fits["logit"].coefficients()
        assert list(frame["estimand"]) == list(fits["logit"].estimates)
        assert list(frame["psi"]) == [e.psi for e in fits["logit"].estimates.values()]

    def test_exponentiating_an_identity_link_fit_is_refused(self, fitted) -> None:
        result, _ = fitted
        with pytest.raises(ValueError, match="not a quantity anybody reports"):
            result.coefficients(scale="ratio")

    def test_a_fit_with_no_working_model_has_no_coefficients(self) -> None:
        frame, _ = make_multi_arm(n=200, seed=SEED)
        result = TMLE(estimands=("ey",), **SETTINGS).fit(frame, outcome="Y", treatment="A").single()
        with pytest.raises(ValueError, match="no working model"):
            result.coefficients()

    def test_an_unknown_scale_says_which_two_exist(self, fits) -> None:
        with pytest.raises(ValueError, match="'link' or 'ratio'"):
            fits["log"].coefficients(scale="odds")


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
        report = result.diagnostics.support()
        assert set(report.propensity_quantiles) >= {f"g[{label}]" for label in DOSE}

    def test_a_truncation_sweep_retargets_without_refitting(self, fitted) -> None:
        result, _ = fitted
        curve = result.diagnostics.truncation_curve(bounds=[0.01, 0.05])
        assert set(curve["estimand"]) == {"msm[(intercept)]", "msm[dose]"}

    def test_a_round_trip_leaves_every_retargeted_analysis_identical(
        self, fitted, tmp_path
    ) -> None:
        """The evaluated design is stored, so a loaded fit targets the same model."""
        result, _ = fitted
        reloaded = load(result.save(tmp_path / "msm.joblib"))
        for name, estimate in result.estimates.items():
            assert reloaded.estimates[name].psi == estimate.psi
            np.testing.assert_array_equal(
                reloaded.estimates[name].influence_curve, estimate.influence_curve
            )
        assert reloaded.nuisance.msm is not None
        np.testing.assert_array_equal(reloaded.nuisance.msm.design, result.nuisance.msm.design)
        assert reloaded.diagnostics.score_equations().passed

    def test_a_reloaded_fit_can_be_refit(self, fitted, tmp_path) -> None:
        """Whole-result persistence retains the importable working-model design."""
        result, _ = fitted
        reloaded = load(result.save(tmp_path / "msm_refit.joblib"))
        refitted = reloaded.estimator.refit(reloaded.data)
        assert refitted.diagnostics.score_equations().passed


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
        estimator = TMLE(
            msm=dose_response(),
            estimands=("ate",),
            outcome_learner=sklearn.linear_model.LinearRegression(),
        )
        with pytest.raises(ValueError):
            estimator.fit(frame, outcome="Y", treatment="A")
