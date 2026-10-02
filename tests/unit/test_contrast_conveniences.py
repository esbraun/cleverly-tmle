r"""Ratio contrasts, Wald tests at a declared null, transforms, RMST and RMTL.

Each is an exact delta-method or linear-functional computation on shipped influence
curves, so each test is an identity at ``1e-12`` on an exact law or on synthetic
estimates.  Each transform and gradient also has a mutation control: a deliberately wrong
variant of the arithmetic that the identity must reject.  Exact-law checks are blind to a
term that vanishes at the truth, so every law here keeps the mutated term nonzero.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_clustered, make_longitudinal_survival
from cleverly.estimators import CTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.inference import Transform, make_estimate, simultaneous_bands
from cleverly.inference.delta import log_odds_ratio_influence, log_ratio_influence
from cleverly.inference.results import linear_functional, ratio_contrast, smooth_contrast
from cleverly.longitudinal import LTMLE
from tests import discrete_law_competing as competing_law
from tests import discrete_law_longitudinal as end_law
from tests import discrete_law_survival as survival_law
from tests.studies import survival_grid_law as grid
from tests.studies.canonical_ltmle import declared_regimens
from tests.unit._exact_sensitivity_support import binary_oracle_fit, multi_oracle_fit

NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
EXACT = {"atol": 1e-12, "rtol": 0}


def _cell_means(**settings: Any) -> dict[str, Any]:
    return {
        "outcome_learner": end_law.CellMeans(),
        "pseudo_learner": end_law.CellMeans(),
        "treatment_learner": end_law.CellMeans(),
        "censoring_learner": end_law.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
        **settings,
    }


@pytest.fixture(scope="module")
def binary_fit() -> Any:
    return binary_oracle_fit(estimands=("ey1", "ey0", "ate", "rr", "or"))[0]


@pytest.fixture(scope="module")
def multi_fit() -> Any:
    return multi_oracle_fit(estimands=("ey", "rr", "or"))


@pytest.fixture(scope="module")
def end_fit() -> Any:
    return LTMLE(
        declared_regimens(end_law.REGIMEN_SPEC),
        reference=end_law.REGIMEN_REFERENCE,
        **_cell_means(),
    ).fit(
        end_law.frame(),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


@pytest.fixture(scope="module")
def survival_fit() -> Any:
    return LTMLE(
        declared_regimens(survival_law.REGIMEN_SPEC),
        reference=survival_law.REGIMEN_REFERENCE,
        **_cell_means(),
    ).fit(
        survival_law.frame(),
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


@pytest.fixture(scope="module")
def competing_fit() -> Any:
    return LTMLE(
        declared_regimens(competing_law.REGIMEN_SPEC),
        reference=competing_law.REGIMEN_REFERENCE,
        **_cell_means(),
    ).fit(
        competing_law.frame(),
        outcome=competing_law.outcome_columns(),
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


@pytest.fixture(scope="module")
def grid_fit() -> Any:
    """The four-node law on its weighted support: the weighted empirical law is the law."""
    return LTMLE(
        {label: list(plan) for label, plan in grid.REGIMENS.items()},
        reference="never",
        **_cell_means(),
    ).fit(grid.weighted_frame(), weights="w", **grid.fit_columns())


def _ratio_gradient(p: np.ndarray) -> np.ndarray:
    return np.array([1.0 / p[1], -p[0] / p[1] ** 2])


def _ratio(p: np.ndarray) -> float:
    return float(p[0] / p[1])


# ------------------------------------------------------------- the R1 defect


class TestTheRatioScaleContrast:
    """``contrast(fn, names, scale="ratio")`` reports a log-scale interval."""

    def test_it_equals_the_registered_risk_ratio(self, binary_fit: Any) -> None:
        derived = binary_fit.contrast(
            _ratio, ["ey1", "ey0"], scale="ratio", gradient=_ratio_gradient
        )
        registered = binary_fit["rr"]
        # ``p1 / p0`` against ``exp(log p1 - log p0)``: equal to rounding.
        assert derived.psi == pytest.approx(registered.psi, rel=1e-15)
        assert derived.log_psi == pytest.approx(registered.log_psi, abs=1e-15)
        # The curve is ``(grad . IC) / v``, the registered one ``IC1/p1 - IC0/p0``: the same
        # number by a different order of floating-point operations.
        np.testing.assert_allclose(derived.influence_curve, registered.influence_curve, **EXACT)
        np.testing.assert_allclose(derived.ci, registered.ci, **EXACT)
        assert derived.pvalue == pytest.approx(registered.pvalue, abs=1e-12)
        assert derived.to_dict()["log_psi"] == derived.log_psi

    def test_dropping_the_division_by_the_value_is_caught(self, binary_fit: Any) -> None:
        """The pre-fix curve, ``grad . IC`` with no ``1 / v``, misses the registered one."""
        undivided = binary_fit.contrast(_ratio, ["ey1", "ey0"], gradient=_ratio_gradient)
        gap = np.max(np.abs(undivided.influence_curve - binary_fit["rr"].influence_curve))
        assert gap > 1e-3

    def test_a_non_positive_value_is_refused(self, binary_fit: Any) -> None:
        with pytest.raises(ValueError, match="a ratio-scale contrast needs a positive value"):
            binary_fit.contrast(lambda p: p[0] - p[1] - 1.0, ["ey1", "ey0"], scale="ratio")


# -------------------------------------------------------- part (a): point


class TestPointRatios:
    def test_the_risk_ratio_is_the_registered_one(self, binary_fit: Any) -> None:
        derived = binary_fit.ratio("ey1", "ey0")
        registered = binary_fit["rr"]
        assert derived.name == "rr"
        assert derived.psi == registered.psi
        assert derived.log_psi == registered.log_psi
        np.testing.assert_array_equal(derived.influence_curve, registered.influence_curve)
        assert derived.ci == registered.ci
        assert derived.pvalue == registered.pvalue

    def test_the_odds_ratio_is_the_registered_one(self, binary_fit: Any) -> None:
        derived = binary_fit.ratio("ey1", "ey0", kind="or")
        registered = binary_fit["or"]
        assert derived.name == "or"
        assert derived.psi == registered.psi
        np.testing.assert_array_equal(derived.influence_curve, registered.influence_curve)
        assert derived.ci == registered.ci

    def test_swapping_the_arms_inverts_the_ratio(self, binary_fit: Any) -> None:
        forward, backward = binary_fit.ratio("ey1", "ey0"), binary_fit.ratio("ey0", "ey1")
        assert backward.name == "rr[ey0 vs ey1]"
        assert backward.psi == pytest.approx(1.0 / forward.psi, rel=1e-14)
        np.testing.assert_allclose(backward.influence_curve, -forward.influence_curve, **EXACT)
        # Nonzero at the law, so the sign is a witness.
        assert np.max(np.abs(forward.influence_curve)) > 0.1

    def test_an_odds_ratio_built_with_the_risk_ratio_derivative_is_caught(
        self, binary_fit: Any
    ) -> None:
        one, zero = binary_fit["ey1"], binary_fit["ey0"]
        _, wrong = log_ratio_influence(one.psi, one.influence_curve, zero.psi, zero.influence_curve)
        right = binary_fit.ratio("ey1", "ey0", kind="or").influence_curve
        assert np.max(np.abs(wrong - right)) > 1e-2

    def test_a_multi_arm_ratio_takes_the_registered_name(self, multi_fit: Any) -> None:
        for kind in ("rr", "or"):
            derived = multi_fit.ratio("ey[high]", "ey[low]", kind=kind)
            registered = multi_fit[f"{kind}[high vs low]"]
            assert derived.name == registered.name
            assert derived.psi == registered.psi
            np.testing.assert_array_equal(derived.influence_curve, registered.influence_curve)

    def test_a_multi_arm_ratio_against_another_arm_names_its_arms(self, multi_fit: Any) -> None:
        assert multi_fit.ratio("ey[high]", "ey[mid]").name == "rr[high vs mid]"

    def test_a_contrast_is_not_a_level(self, binary_fit: Any) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "a ratio compares two levels, and 'ate' is reported on the 'difference' "
                "scale. Pass two level estimates, such as two counterfactual means"
            ),
        ):
            binary_fit.ratio("ate", "ey0")

    def test_an_odds_ratio_of_a_continuous_outcome_is_refused(self) -> None:
        frame, _ = make_clustered(n=200, seed=3, cluster_size=5)
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                estimands=("ey1", "ey0"),
                simultaneous=False,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
            .single()
        )
        with pytest.raises(
            CapabilityError,
            match=re.escape(
                "an odds ratio compares two probabilities, and this fit's outcome family is "
                "'gaussian'. Use kind=\"rr\" for a ratio of means"
            ),
        ):
            result.ratio("ey1", "ey0", kind="or")

    def test_a_ratio_after_repeats_is_refused(self) -> None:
        from cleverly.datasets import make_binary_outcome

        frame, _ = make_binary_outcome(n=200, seed=4)
        result = (
            TMLE(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                estimands=("ey1", "ey0"),
                n_folds=2,
                repeats=3,
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
            .single()
        )
        with pytest.raises(CapabilityError, match=r"ratio\(\)"):
            result.ratio("ey1", "ey0")


class TestInheritance:
    def test_a_clustered_ratio_uses_the_cluster_variance(self) -> None:
        frame, _ = make_clustered(n=600, seed=5, cluster_size=10)
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                estimands=("ey1", "ey0"),
                simultaneous=False,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
            .single()
        )
        ratio = result.ratio("ey1", "ey0")
        from cleverly.inference import influence_covariance

        variance = influence_covariance(ratio.influence_curve[:, None], cluster=result.data.cluster)
        assert ratio.std_error == pytest.approx(float(np.sqrt(variance[0, 0])), rel=1e-12)
        assert ratio.n_clusters == result["ey1"].n_clusters

    def test_a_few_cluster_ratio_refuses_its_interval(self) -> None:
        frame, _ = make_clustered(n=200, seed=6, cluster_size=10)
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                estimands=("ey1", "ey0"),
                simultaneous=False,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
            .single()
        )
        ratio = result.ratio("ey1", "ey0")
        assert ratio.inference == "few_cluster_plugin"
        with pytest.raises(CapabilityError):
            _ = ratio.ci
        with pytest.raises(CapabilityError):
            ratio.wald_test(null=2.0)

    def test_a_selector_ctmle_ratio_keeps_its_status(self) -> None:
        from cleverly.datasets import make_binary_outcome

        frame, _ = make_binary_outcome(n=300, seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                CTMLE(
                    outcome_learner=LogisticRegression(max_iter=1000),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    strategy="greedy",
                    cross_fit=False,
                    estimands=("ate", "ey1", "ey0"),
                    simultaneous=False,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
                .single()
            )
        ratio = result.ratio("ey1", "ey0")
        assert ratio.inference == result.inference_status == "working_mechanism_plugin"
        with pytest.raises(CapabilityError):
            _ = ratio.ci
        assert ratio.plugin_interval[0] < ratio.psi < ratio.plugin_interval[1]

    def test_a_mixed_status_selection_is_refused(self) -> None:
        curve = np.array([0.1, -0.2, 0.3, -0.2])
        estimates = {
            "a": make_estimate("a", 0.4, curve, n=4, scale="level"),
            "b": make_estimate(
                "b", 0.3, -curve, n=4, scale="level", inference="few_cluster_plugin"
            ),
        }
        with pytest.raises(ValueError, match="declare different inference statuses"):
            ratio_contrast(estimates, "a", "b", n=4, name="rr")


# ----------------------------------------------------- part (a): longitudinal


class TestLongitudinalRatios:
    def test_an_end_of_study_ratio_is_the_longhand(self, end_fit: Any) -> None:
        a, b = "ey_regimen[always]", "ey_regimen[never]"
        rows = end_law.first_row_of()
        ratio = end_fit.ratio(a, b)
        assert ratio.name == "rr_regimen[always vs never]"
        assert ratio.log_psi == pytest.approx(
            np.log(end_law.TRUTH[a]) - np.log(end_law.TRUTH[b]), abs=1e-12
        )
        expected = end_law.eif(a) / end_law.TRUTH[a] - end_law.eif(b) / end_law.TRUTH[b]
        np.testing.assert_allclose(ratio.influence_curve[rows], expected, **EXACT)

    def test_an_end_of_study_odds_ratio_is_the_longhand(self, end_fit: Any) -> None:
        a, b = "ey_regimen[early]", "ey_regimen[never]"
        rows = end_law.first_row_of()
        ratio = end_fit.ratio(a, b, kind="or")
        assert ratio.name == "or_regimen[early vs never]"
        pa, pb = end_law.TRUTH[a], end_law.TRUTH[b]
        expected = end_law.eif(a) / (pa * (1 - pa)) - end_law.eif(b) / (pb * (1 - pb))
        np.testing.assert_allclose(ratio.influence_curve[rows], expected, **EXACT)

    def test_an_end_of_study_survival_view_is_refused(self, end_fit: Any) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("an end-of-study fit reports means, not a survival curve. Drop view="),
        ):
            end_fit.ratio("ey_regimen[always]", "ey_regimen[never]", view="survival")

    @pytest.mark.parametrize("horizon", survival_law.HORIZONS)
    def test_a_survival_ratio_is_the_longhand(self, survival_fit: Any, horizon: int) -> None:
        a = f"risk_regimen[always @ t={horizon}]"
        b = f"risk_regimen[never @ t={horizon}]"
        rows = survival_law.first_row_of()
        pa, pb = survival_law.TRUTH[a], survival_law.TRUTH[b]
        risk = survival_fit.ratio(a, b)
        assert risk.name == f"rr_regimen[always vs never @ t={horizon}]"
        assert risk.psi == pytest.approx(pa / pb, abs=1e-12)
        expected = survival_law.eif(a) / pa - survival_law.eif(b) / pb
        np.testing.assert_allclose(risk.influence_curve[rows], expected, **EXACT)
        survival = survival_fit.ratio(a, b, view="survival")
        assert survival.name == f"survival_rr_regimen[always vs never @ t={horizon}]"
        assert survival.psi == pytest.approx((1 - pa) / (1 - pb), abs=1e-12)
        expected = -survival_law.eif(a) / (1 - pa) + survival_law.eif(b) / (1 - pb)
        np.testing.assert_allclose(survival.influence_curve[rows], expected, **EXACT)
        # A view that forgot the complement reports the risk ratio, which differs here
        # because no risk on this law is one half.
        assert abs(survival.psi - risk.psi) > 1e-2

    def test_the_survival_odds_ratio_is_the_reciprocal(self, survival_fit: Any) -> None:
        a, b = "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]"
        risk = survival_fit.ratio(a, b, kind="or")
        survival = survival_fit.ratio(a, b, kind="or", view="survival")
        assert survival.name == "survival_or_regimen[always vs never @ t=2]"
        assert survival.psi == pytest.approx(1.0 / risk.psi, rel=1e-13)
        np.testing.assert_allclose(survival.influence_curve, -risk.influence_curve, **EXACT)

    def test_a_ratio_across_horizons_takes_the_generic_name(self, survival_fit: Any) -> None:
        a, b = "risk_regimen[always @ t=2]", "risk_regimen[always @ t=1]"
        assert survival_fit.ratio(a, b).name == f"rr[{a} vs {b}]"

    @pytest.mark.parametrize("cause", ["relapse", "death"])
    def test_a_competing_ratio_is_the_longhand(self, competing_fit: Any, cause: str) -> None:
        a = f"cif_regimen[always, {cause} @ t=2]"
        b = f"cif_regimen[never, {cause} @ t=2]"
        rows = competing_law.first_row_of()
        pa, pb = competing_law.TRUTH[a], competing_law.TRUTH[b]
        ratio = competing_fit.ratio(a, b)
        assert ratio.name == f"rr_regimen[always vs never, {cause} @ t=2]"
        assert ratio.psi == pytest.approx(pa / pb, abs=1e-12)
        expected = competing_law.eif(a) / pa - competing_law.eif(b) / pb
        np.testing.assert_allclose(ratio.influence_curve[rows], expected, **EXACT)

    def test_a_cause_swap_is_caught(self, competing_fit: Any) -> None:
        relapse = competing_fit.ratio(
            "cif_regimen[always, relapse @ t=2]", "cif_regimen[never, relapse @ t=2]"
        )
        death = competing_fit.ratio(
            "cif_regimen[always, death @ t=2]", "cif_regimen[never, death @ t=2]"
        )
        assert abs(relapse.psi - death.psi) > 1e-2

    def test_a_competing_survival_view_is_refused(self, competing_fit: Any) -> None:
        with pytest.raises(ValueError, match="1 - one cause's incidence is not all-cause"):
            competing_fit.ratio(
                "cif_regimen[always, death @ t=2]",
                "cif_regimen[never, death @ t=2]",
                view="survival",
            )

    def test_a_working_model_ratio_is_refused(self) -> None:
        from cleverly.msm import MSM

        frame, _ = make_longitudinal_survival(n=300, seed=1)
        result = LTMLE(
            {"always": 1, "never": 0},
            msm=MSM(
                lambda label, horizon, base: np.column_stack(
                    [np.ones(len(base)), np.full(len(base), float(label == "always"))]
                ),
                terms=("(intercept)", "always"),
                design_kind="known",
            ),
            n_folds=1,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            simultaneous=False,
        ).fit(
            frame,
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
        first, second = list(result.estimates)[:2]
        with pytest.raises(ValueError, match="a ratio of two"):
            result.ratio(first, second)
        with pytest.raises(ValueError, match="RMST sums a survival curve"):
            result.rmst("always", 2)


# --------------------------------------------------------- part (b): Wald test


class TestWaldTest:
    @pytest.mark.parametrize("null", [0.0, 0.1, -0.3])
    def test_a_level_and_a_difference(self, binary_fit: Any, null: float) -> None:
        for name in ("ey1", "ate"):
            estimate = binary_fit[name]
            test = estimate.wald_test(null=null)
            z = (estimate.psi - null) / estimate.std_error
            assert test.statistic == pytest.approx(z, abs=1e-12)
            assert test.pvalue == pytest.approx(2 * stats.norm.sf(abs(z)), abs=1e-12)
            assert test.null == null and test.null_inference_value == null

    def test_a_ratio_logs_the_null(self, binary_fit: Any) -> None:
        estimate = binary_fit["rr"]
        test = estimate.wald_test(null=2.0)
        z = (estimate.log_psi - np.log(2.0)) / estimate.std_error
        assert test.statistic == pytest.approx(z, abs=1e-12)
        # The mutation: a test that compared log psi with the null itself, unlogged.
        unlogged = (estimate.log_psi - 2.0) / estimate.std_error
        assert abs(test.statistic - unlogged) == pytest.approx(
            (2.0 - np.log(2.0)) / estimate.std_error, rel=1e-12
        )

    def test_a_fraction(self) -> None:
        estimate = make_estimate(
            "paf", 0.3, np.array([0.1, -0.2, 0.3, -0.2]), n=4, scale="fraction"
        )
        test = estimate.wald_test(null=0.1)
        assert test.statistic == pytest.approx((0.3 - 0.1) / estimate.std_error, abs=1e-12)

    def test_the_estimate_itself_gives_no_evidence(self, binary_fit: Any) -> None:
        estimate = binary_fit["ate"]
        test = estimate.wald_test(null=estimate.psi)
        assert test.statistic == 0.0 and test.pvalue == 1.0

    def test_the_default_test_is_the_pvalue_bit_for_bit(self, binary_fit: Any) -> None:
        for estimate in binary_fit.estimates.values():
            assert estimate.wald_test().pvalue == estimate.pvalue

    def test_a_non_positive_ratio_null_is_refused(self, binary_fit: Any) -> None:
        with pytest.raises(ValueError, match="a ratio-scale null must be positive; got 0"):
            binary_fit["rr"].wald_test(null=0.0)


# ------------------------------------------------------- part (c): transforms


class TestTransforms:
    def test_the_log_transform_matches_the_ratio_scale_bit_for_bit(self, binary_fit: Any) -> None:
        names = ["ey1", "ey0"]
        ratio = binary_fit.contrast(_ratio, names, scale="ratio", gradient=_ratio_gradient)
        logged = binary_fit.contrast(
            _ratio, names, scale="level", transform=Transform.log(), gradient=_ratio_gradient
        )
        assert logged.std_error == ratio.std_error
        assert logged.ci == ratio.ci
        assert logged.wald_test(null=1.0).pvalue == ratio.pvalue
        assert logged.to_dict()["transform"] == "log"
        assert logged.to_dict()["transform_psi"] == ratio.log_psi

    def test_the_drtmle_list_contrast(self, binary_fit: Any) -> None:
        """``f_inv(f(h) +/- z sqrt(g' S g))`` with ``g = f'(h) grad h``, written longhand."""
        names = ["ey1", "ey0"]
        derived = binary_fit.contrast(
            _ratio, names, scale="level", transform=Transform.log(), gradient=_ratio_gradient
        )
        psi = np.array([binary_fit[n].psi for n in names])
        h = _ratio(psi)
        g = (1.0 / h) * _ratio_gradient(psi)
        sigma = binary_fit.covariance(names)
        se = float(np.sqrt(g @ sigma @ g))
        z = stats.norm.ppf(0.975)
        assert derived.std_error == pytest.approx(se, rel=1e-12)
        np.testing.assert_allclose(
            derived.ci, (np.exp(np.log(h) - z * se), np.exp(np.log(h) + z * se)), **EXACT
        )

    def test_dropping_the_slope_is_caught(self, binary_fit: Any) -> None:
        names = ["ey1", "ey0"]
        derived = binary_fit.contrast(
            _ratio, names, scale="level", transform=Transform.log(), gradient=_ratio_gradient
        )
        no_slope = binary_fit.contrast(_ratio, names, scale="level", gradient=_ratio_gradient)
        assert abs(no_slope.std_error - derived.std_error) > 1e-3

    def test_mapping_back_with_the_forward_map_is_caught(self, binary_fit: Any) -> None:
        names = ["ey1", "ey0"]
        derived = binary_fit.contrast(
            _ratio, names, scale="level", transform=Transform.log(), gradient=_ratio_gradient
        )
        center = derived.inference_value
        z = stats.norm.ppf(0.975)
        wrong = (np.log(center - z * derived.std_error), np.log(center + z * derived.std_error))
        assert not np.allclose(wrong, derived.ci)

    def test_a_decreasing_transform_gives_an_ordered_interval(self, binary_fit: Any) -> None:
        negative_log = Transform(
            "negative log",
            lambda x: -float(np.log(x)),
            lambda y: float(np.exp(-y)),
            lambda x: -1.0 / x,
        )
        derived = binary_fit.contrast(
            _ratio, ["ey1", "ey0"], scale="level", transform=negative_log, gradient=_ratio_gradient
        )
        low, high = derived.ci
        assert low < derived.psi < high
        # Without the sort the inverse map hands the limits back in the wrong order.
        center, se = derived.inference_value, derived.std_error
        z = stats.norm.ppf(0.975)
        unsorted = (np.exp(-(center - z * se)), np.exp(-(center + z * se)))
        assert unsorted[0] > unsorted[1]
        np.testing.assert_allclose(sorted(unsorted), derived.ci, **EXACT)

    def test_a_numeric_slope_agrees_to_finite_difference_precision(self, binary_fit: Any) -> None:
        numeric = Transform("log", np.log, np.exp)
        exact = binary_fit.contrast(
            _ratio, ["ey1", "ey0"], transform=Transform.log(), gradient=_ratio_gradient
        )
        approx = binary_fit.contrast(
            _ratio, ["ey1", "ey0"], transform=numeric, gradient=_ratio_gradient
        )
        assert approx.std_error == pytest.approx(exact.std_error, rel=1e-8)

    def test_the_default_null_outside_the_domain_is_named(self, binary_fit: Any) -> None:
        derived = binary_fit.contrast(
            _ratio, ["ey1", "ey0"], transform=Transform.log(), gradient=_ratio_gradient
        )
        with pytest.raises(
            ValueError,
            match=(
                "the default null 0 has no value under the transform 'log'. Call "
                r"wald_test\(null=...\) with a null inside the transform's domain"
            ),
        ):
            _ = derived.pvalue
        assert np.isnan(derived.to_dict()["p_value"])

    def test_a_transform_with_a_ratio_scale_is_refused(self, binary_fit: Any) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "a ratio-scale estimate already carries the log transform. Pass "
                'scale="ratio" or transform=, not both'
            ),
        ):
            binary_fit.contrast(_ratio, ["ey1", "ey0"], scale="ratio", transform=Transform.log())
        with pytest.raises(ValueError, match="must be 'level' or 'difference'"):
            binary_fit.contrast(_ratio, ["ey1", "ey0"], scale="fraction", transform=Transform.log())

    def test_a_transformed_band_maps_through_the_inverse(self, binary_fit: Any) -> None:
        derived = binary_fit.contrast(
            _ratio,
            ["ey1", "ey0"],
            name="logged",
            transform=Transform.log(),
            gradient=_ratio_gradient,
        )
        other = binary_fit["ate"]
        bands = simultaneous_bands([derived, other], random_state=0, n_replicates=500)
        half = bands.critical_value * derived.std_error
        expected = (np.exp(derived.inference_value - half), np.exp(derived.inference_value + half))
        np.testing.assert_allclose(bands.bands["logged"], expected, **EXACT)
        # The band without the map would be centred on log(psi), far from psi.
        assert bands.bands["logged"][0] < derived.psi < bands.bands["logged"][1]

    def test_a_transformed_estimate_never_reaches_a_fit(
        self, binary_fit: Any, end_fit: Any
    ) -> None:
        for result in (binary_fit, end_fit):
            assert all(e.transform is None for e in result.estimates.values())

    def test_the_built_in_transforms_pickle(self) -> None:
        import pickle

        for transform in (Transform.log(), Transform.logit()):
            restored = pickle.loads(pickle.dumps(transform))
            assert restored.forward(0.5) == transform.forward(0.5)


# ------------------------------------------------------ part (d): RMST, RMTL


class TestLinearFunctional:
    """The algebra on six synthetic estimates with distinct nonzero curves."""

    @pytest.fixture
    def estimates(self) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        return {
            f"F{t}": make_estimate(f"F{t}", 0.1 * t, rng.normal(size=50), n=50, scale="level")
            for t in range(1, 7)
        }

    def test_weights_and_constant(self, estimates: dict[str, Any]) -> None:
        weights = {"F1": -1.0, "F2": -1.0, "F3": -1.0}
        derived = linear_functional(
            estimates, weights, constant=4.0, n=50, name="rmst", scale="level"
        )
        longhand_psi = 4.0 - sum(estimates[k].psi for k in weights)
        longhand_curve = -sum(estimates[k].influence_curve for k in weights)
        assert derived.psi == pytest.approx(longhand_psi, abs=1e-14)
        np.testing.assert_allclose(derived.influence_curve, longhand_curve, **EXACT)


class TestTheGridLaw:
    """The four-node law's two statements of the truth, and its sampler."""

    @pytest.mark.parametrize("regimen", list(grid.REGIMENS))
    def test_the_enumeration_matches_a_monte_carlo_draw(self, regimen: str) -> None:
        times = grid.simulate_event_times(regimen, 1_000_000, np.random.default_rng(7))
        restricted = np.minimum(times, grid.K + 1)
        se = restricted.std(ddof=1) / np.sqrt(times.size)
        assert abs(restricted.mean() - grid.rmst_truth(regimen, grid.K + 1)) < 4 * se

    @pytest.mark.parametrize("regimen", list(grid.REGIMENS))
    def test_the_observed_g_formula_is_the_counterfactual_risk(self, regimen: str) -> None:
        for horizon in range(1, grid.K + 1):
            assert float(grid.functional(grid.PROBS, regimen, horizon)) == pytest.approx(
                grid.risk_truth(regimen, horizon), abs=1e-14
            )

    def test_the_support_is_a_distribution(self) -> None:
        assert grid.PROBS.sum() == pytest.approx(1.0, abs=1e-14)
        assert grid.PROBS.min() > 0

    def test_the_sampler_draws_the_law(self) -> None:
        frame = grid.sample(200_000, np.random.default_rng(3))
        observed = frame["Y1"].eq(1).mean()
        expected = float(
            sum(
                m
                for path, m in zip(grid.SUPPORT, grid.PROBS, strict=True)
                if len(path) > 4 and path[4] == 1
            )
        )
        assert observed == pytest.approx(expected, abs=4 * np.sqrt(expected / 200_000))


class TestRmst:
    @pytest.mark.parametrize("regimen", list(grid.REGIMENS))
    @pytest.mark.parametrize("horizon", range(2, grid.K + 2))
    def test_rmst_and_rmtl_are_the_enumeration_truth(
        self, grid_fit: Any, regimen: str, horizon: int
    ) -> None:
        assert grid_fit.rmst(regimen, horizon).psi == pytest.approx(
            grid.rmst_truth(regimen, horizon), abs=1e-12
        )
        assert grid_fit.rmtl(regimen, horizon).psi == pytest.approx(
            grid.rmtl_truth(regimen, horizon), abs=1e-12
        )

    def test_the_curve_is_the_weighted_eif(self, grid_fit: Any) -> None:
        weights = grid_fit.data.weights
        expected = weights * -sum(grid.eif("always", t) for t in range(1, grid.K + 1))
        curve = grid_fit.rmst("always", grid.K + 1).influence_curve
        np.testing.assert_allclose(curve, expected, atol=1e-11, rtol=0)

    def test_a_contrast_between_regimens(self, grid_fit: Any) -> None:
        derived = grid_fit.rmst("always", 5, versus="never")
        assert derived.name == "rmst_regimen[always vs never @ t=5]"
        assert derived.scale == "difference"
        assert derived.psi == pytest.approx(
            grid.rmst_truth("always", 5) - grid.rmst_truth("never", 5), abs=1e-12
        )
        np.testing.assert_allclose(
            derived.influence_curve,
            grid_fit.rmst("always", 5).influence_curve - grid_fit.rmst("never", 5).influence_curve,
            **EXACT,
        )

    def test_an_off_by_one_range_misses_the_truth(self, grid_fit: Any) -> None:
        def risk(t: int) -> str:
            return f"risk_regimen[always @ t={t}]"

        truth = grid.rmst_truth("always", 4)
        through = linear_functional(
            grid_fit.estimates,
            {risk(t): -1.0 for t in range(1, 5)},
            constant=4.0,
            n=grid_fit.n,
            name="m",
            scale="level",
        )
        from_zero = linear_functional(
            grid_fit.estimates,
            {risk(t): -1.0 for t in range(1, 4)},
            constant=3.0,
            n=grid_fit.n,
            name="m",
            scale="level",
        )
        assert abs(through.psi - truth) > 0.1
        assert abs(from_zero.psi - truth) > 0.1

    def test_the_consistency_identity(self, grid_fit: Any) -> None:
        for horizon in range(2, grid.K + 2):
            total = grid_fit.rmst("early", horizon).psi + grid_fit.rmtl("early", horizon).psi
            assert total == pytest.approx(horizon, abs=1e-12)

    def test_the_two_node_survival_law_at_three(self, survival_fit: Any) -> None:
        for regimen in survival_law.REGIMEN_SPEC:
            risks = {
                t: survival_law.TRUTH[f"risk_regimen[{regimen} @ t={t}]"]
                for t in survival_law.HORIZONS
            }
            truth = grid.enumerate_rmst(grid.event_time_pmf(risks), 3)
            assert survival_fit.rmst(regimen, 3).psi == pytest.approx(truth, abs=1e-12)
        rows = survival_law.first_row_of()
        expected = -(
            survival_law.eif("risk_regimen[always @ t=1]")
            + survival_law.eif("risk_regimen[always @ t=2]")
        )
        np.testing.assert_allclose(
            survival_fit.rmst("always", 3).influence_curve[rows], expected, **EXACT
        )

    def test_the_two_node_competing_law_at_three(self, competing_fit: Any) -> None:
        probs = competing_law.PROBS
        for regimen in competing_law.REGIMEN_SPEC:
            survival = {
                t: float(competing_law.survival_functional(probs, regimen, t)) for t in (1, 2)
            }
            pmf = grid.event_time_pmf({t: 1.0 - s for t, s in survival.items()})
            truth = grid.enumerate_rmst(pmf, 3)
            assert competing_fit.rmst(regimen, 3).psi == pytest.approx(truth, abs=1e-12)
            for cause in ("relapse", "death"):
                incidence = {
                    t: competing_law.TRUTH[f"cif_regimen[{regimen}, {cause} @ t={t}]"]
                    for t in (1, 2)
                }
                cause_pmf = grid.event_time_pmf(incidence)
                lost = sum((3 - t) * cause_pmf[t] for t in (1, 2))
                derived = competing_fit.rmtl(regimen, 3, cause)
                assert derived.name == f"rmtl_regimen[{regimen}, {cause} @ t=3]"
                assert derived.psi == pytest.approx(lost, abs=1e-12)

    def test_dropping_a_cause_from_the_competing_rmst_is_caught(self, competing_fit: Any) -> None:
        full = competing_fit.rmst("always", 3).psi
        one_cause = 3.0 - sum(
            competing_fit[f"cif_regimen[always, relapse @ t={t}]"].psi for t in (1, 2)
        )
        assert abs(full - one_cause) > 1e-2

    def test_the_refusals(self, end_fit: Any, survival_fit: Any, competing_fit: Any) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "RMST sums a survival curve, and this fit reports one end-of-study mean of "
                "'Y'. Declare an outcome sequence to estimate a curve"
            ),
        ):
            end_fit.rmst("always", 2)
        with pytest.raises(
            ValueError,
            match="horizon must be an integer from 2 to 3; RMST up to t=1 is the constant 1",
        ):
            survival_fit.rmst("always", 1)
        with pytest.raises(ValueError, match="horizon must be an integer from 2 to 3"):
            survival_fit.rmst("always", 4)
        with pytest.raises(ValueError, match="name one with cause="):
            competing_fit.rmtl("always", 3)

    def test_a_missing_horizon_is_named(self) -> None:
        result = LTMLE(
            declared_regimens(survival_law.REGIMEN_SPEC),
            reference=survival_law.REGIMEN_REFERENCE,
            horizons=[2],
            **_cell_means(),
        ).fit(
            survival_law.frame(),
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
        with pytest.raises(
            ValueError,
            match=(
                r"RMST up to t=3 needs the risk at every horizon 1 to 2, and this fit "
                r"reports \[2\]. Refit with horizons=None or include \[1\]"
            ),
        ):
            result.rmst("always", 3)


def test_the_ratio_helpers_keep_their_domain_refusals() -> None:
    curve = np.array([0.1, -0.1, 0.2, -0.2])
    estimates = {
        "a": make_estimate("a", 0.0, curve, n=4, scale="level"),
        "b": make_estimate("b", 0.5, curve, n=4, scale="level"),
        "c": make_estimate("c", 1.0, curve, n=4, scale="level"),
    }
    with pytest.raises(ValueError, match="strictly positive"):
        ratio_contrast(estimates, "a", "b", n=4, name="rr")
    with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
        ratio_contrast(estimates, "c", "b", kind="or", n=4, name="or")
    assert log_odds_ratio_influence is not None
    assert smooth_contrast is not None
