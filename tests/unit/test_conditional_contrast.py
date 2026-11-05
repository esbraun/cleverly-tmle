r"""``ConditionalContrast`` through the study layer: refusals, results, curves and scores.

The projection arithmetic is pinned in ``tests/unit/test_effect_modification_exact.py``.
This module pins the contract around it: every refusal of the P1 plan's part (a) raises
before any learner with its text (R2 to R7, R9, R11, R13), the curve result predicts,
bands and persists, the heterogeneity test excludes the intercept, and the assessment of a
fixed external score is exact at the oracle with the influence curve of the plan's 2.3.
Its rev 1 curve, ``+ 2 (c - E s)(Dtilde - c)``, is checked to fail.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import cleverly
from cleverly import ATE, CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Constant, Cosine, Indicators, Linear, Modifiers, Sieve
from cleverly.effect_curves import EffectCurveResult
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.unit._natural_course_support import NeverFit

MODIFIERS = Modifiers(["W"], domains={"W": (0.0, 2.0)})

#: A score of ``W`` that is not the CATE, so the risk difference is not ``-Var(tau)``.
SCORE = np.array([0.10, 0.40, 0.20])


def law_study(**design: Any) -> CausalStudy:
    frame = law.frame()
    frame["S"] = SCORE[frame["W"].astype(int)]
    frame["T"] = (law.Q_EXACT[:, 1] - law.Q_EXACT[:, 0])[frame["W"].astype(int)]
    frame["M"] = (frame["W"] > 0).astype(float)
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), **design)
    )


def oracle() -> dict[str, Any]:
    dgp = law.DiscreteLaw()
    return {
        "outcome_learner": OracleOutcome(dgp),
        "treatment_learner": OracleTreatment(dgp),
        "cross_fit": False,
        "random_state": 0,
    }


def never() -> dict[str, Any]:
    NeverFit.calls = 0
    return {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}


def linear_contrast(scale: str = "difference") -> ConditionalContrast:
    return ConditionalContrast(MODIFIERS, scale=scale, model=Linear(["W"]))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def projection_fit() -> Any:
    return law_study().identify(linear_contrast()).estimate(**oracle())


class TestTheStudyFit:
    def test_the_study_reports_the_engine_projection(self, projection_fit: Any) -> None:
        expected = [law.TRUTH[f"cate_projection[{t}]"] for t in law.EFFECT_MODIFICATION_TERMS]
        np.testing.assert_allclose(
            [estimate.psi for estimate in projection_fit.estimates.values()], expected, atol=1e-12
        )
        keys = projection_fit.parameter_keys
        assert [key.term for key in keys.values()] == ["(intercept)", "W"]
        assert {key.axis for key in keys.values()} == {"effect_modification"}

    def test_the_estimand_names_its_scale(self) -> None:
        assert linear_contrast().name == "cate"
        assert linear_contrast("log_ratio").name == "log_crr"
        assert "projection" in linear_contrast().definition

    def test_only_tmle_is_available(self) -> None:
        methods = {m.name: m for m in law_study().identify(linear_contrast()).available_methods()}
        assert methods["tmle"].available
        assert not any(m.available for name, m in methods.items() if name != "tmle")
        assert methods["ep"].refusal.roadmap == "P1"

    def test_other_estimands_read_the_ep_record(self) -> None:
        methods = {m.name: m for m in law_study().identify(ATE()).available_methods()}
        refusal = methods["ep"].refusal
        assert refusal.reason is RefusalReason.DIFFERENT_ESTIMAND
        assert refusal.condition == "EP estimates a ConditionalContrast."
        assert refusal.alternative == "Request ConditionalContrast(modifiers=...)."


class TestRefusalsBeforeAnyLearner:
    def test_r2_tmle_without_a_model(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS))
        with pytest.raises(CapabilityError, match="not pathwise differentiable"):
            effect.estimate(**never())
        assert NeverFit.calls == 0

    def test_r3_a_basis_that_reads_an_undeclared_domain(self) -> None:
        with pytest.raises(CapabilityError, match="known function of the modifiers"):
            law_study().identify(ConditionalContrast(Modifiers(["W"]), model=Cosine(["W"], 2)))

    def test_r3_indicators_without_declared_levels(self) -> None:
        with pytest.raises((CapabilityError, DataError)):
            law_study().identify(ConditionalContrast(Modifiers(["W"]), model=Indicators("W")))

    def test_r4_a_sieve_sized_by_cross_validation(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS, model=Sieve(Cosine(["W"], 3))))
        with pytest.raises(CapabilityError, match="chosen by cross-validation"):
            effect.estimate(**never())
        assert NeverFit.calls == 0

    def test_r4_a_declared_size_resolves(self) -> None:
        effect = law_study().identify(
            ConditionalContrast(MODIFIERS, model=Sieve(Cosine(["W"], 3), size=1))
        )
        assert effect.functional.effect_modification.terms == ("(intercept)", "cos1(W)")

    def test_r5_a_log_ratio_on_a_shifted_scale(self) -> None:
        frame = law.frame()
        frame["Y"] = 5.0 * frame["Y"] + 1.0
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        effect = study.identify(linear_contrast("log_ratio"))
        with pytest.raises(CapabilityError, match="These q_bounds start at 1"):
            effect.estimate(q_bounds=(1.0, 6.0), cross_fit=False, **never())
        assert NeverFit.calls == 0

    def test_r6_a_modifier_outside_the_adjustment_set(self) -> None:
        with pytest.raises(CapabilityError, match="must be an adjustment column") as caught:
            law_study().identify(ConditionalContrast(Modifiers(["S"]), model=Linear(["S"])))
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    @pytest.mark.parametrize(
        ("design", "axis"),
        [({"strata": ("W",)}, "strata="), ({"intermediate": "M"}, "an intermediate")],
    )
    def test_r7_compositions_outside_p1(self, design: dict, axis: str) -> None:
        with pytest.raises(CapabilityError, match=f"does not yet compose with {axis}") as caught:
            law_study(**design).identify(linear_contrast())
        assert caught.value.refusal.roadmap == "P1"

    @pytest.mark.parametrize(
        ("setting", "axis"),
        [({"repeats": 2}, "repeats above 1"), ({"n_bootstrap": 20}, "the full-refit bootstrap")],
    )
    def test_r7_settings_outside_p1(self, setting: dict, axis: str) -> None:
        effect = law_study().identify(linear_contrast())
        with pytest.raises(CapabilityError, match=f"does not yet compose with {axis}"):
            effect.estimate(**never(), **setting)
        assert NeverFit.calls == 0

    def test_r9_sensitivity_and_refutation_refuse_on_the_projection(
        self, projection_fit: Any
    ) -> None:
        rows = projection_fit.sensitivity.capabilities
        assert rows and not any(row.available for row in rows)
        for row in rows:
            assert row.refusal.condition.startswith(f"{row.operation} is defined for a mean")
            assert row.refusal.alternative == "Run it on ATE() or RiskRatio() on the same study."
        refute = projection_fit.diagnostics.capability("refute")
        assert not refute.available and refute.refusal.roadmap == "P1"
        with pytest.raises(CapabilityError, match="working-model coefficient"):
            projection_fit.sensitivity.evalue()


class TestTheHeterogeneityTest:
    def test_it_tests_the_slope_alone(self, projection_fit: Any) -> None:
        test = projection_fit.heterogeneity_test()
        slope = projection_fit.estimates["cate_projection[W]"]
        assert test.df == 1 and test.terms == ("W",)
        assert test.statistic == pytest.approx((slope.psi / slope.std_error) ** 2, rel=1e-10)

    def test_including_the_intercept_changes_the_statistic(self, projection_fit: Any) -> None:
        """Mutation control: the joint statistic over both terms is a different number."""
        names = list(projection_fit.estimates)
        beta = np.array([projection_fit.estimates[name].psi for name in names])
        joint = float(beta @ np.linalg.solve(projection_fit.covariance(names), beta))
        assert joint - projection_fit.heterogeneity_test().statistic > 1.0

    def test_a_constant_basis_has_nothing_to_test(self) -> None:
        fit = (
            law_study()
            .identify(ConditionalContrast(MODIFIERS, model=Constant()))
            .estimate(**oracle())
        )
        with pytest.raises(CapabilityError, match="no non-constant term"):
            fit.heterogeneity_test()

    def test_an_arm_fit_has_no_test(self) -> None:
        fit = law_study().identify(ATE()).estimate(**oracle())
        with pytest.raises(CapabilityError, match="projects no conditional contrast"):
            fit.heterogeneity_test()


#: The three ``W`` cells, as a grid of modifier values.
GRID = {"W": np.array([0.0, 1.0, 2.0])}


class TestTheCurve:
    def test_predictions_are_the_basis_times_the_coefficients(self, projection_fit: Any) -> None:
        curve = projection_fit.effect_curve()
        beta = np.array([estimate.psi for estimate in projection_fit.estimates.values()])
        np.testing.assert_allclose(curve.predict(GRID), law.EFFECT_MODIFICATION_DESIGN @ beta)
        frame = curve.predict(GRID, interval=True)
        covariance = projection_fit.covariance(list(projection_fit.estimates))
        phi = law.EFFECT_MODIFICATION_DESIGN
        np.testing.assert_allclose(
            frame["std_err"], np.sqrt(np.einsum("ip,pq,iq->i", phi, covariance, phi))
        )
        assert curve.to_frame().columns.tolist() == ["effect_curve:cate"]

    def test_the_band_is_wider_than_the_pointwise_interval(self, projection_fit: Any) -> None:
        curve = projection_fit.effect_curve(GRID, random_state=0)
        assert curve.band is not None and curve.band.critical_value > 1.96
        np.testing.assert_allclose(curve.band.estimate, curve.predict(GRID))
        assert "declared grid points" in str(curve.summary())

    def test_predict_checks_the_modifier_schema(self, projection_fit: Any) -> None:
        curve = projection_fit.effect_curve()
        with pytest.raises(DataError):
            curve.predict({"X": np.array([1.0])})
        with pytest.raises(DataError, match="domain"):
            curve.predict({"W": np.array([3.0])})

    def test_outside_clip_clips_to_the_domain(self) -> None:
        clipped = Modifiers(["W"], domains={"W": (0.0, 2.0)}, outside="clip")
        fit = (
            law_study()
            .identify(ConditionalContrast(clipped, model=Linear(["W"])))
            .estimate(**oracle())
        )
        curve = fit.effect_curve()
        np.testing.assert_allclose(
            curve.predict({"W": np.array([5.0])}), curve.predict({"W": np.array([2.0])})
        )

    def test_save_and_load_give_the_same_predictions(self, projection_fit: Any, tmp_path) -> None:
        curve = projection_fit.effect_curve()
        path = curve.save(tmp_path / "curve.joblib")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = cleverly.load(path)
        assert isinstance(loaded, EffectCurveResult)
        np.testing.assert_array_equal(loaded.predict(GRID), curve.predict(GRID))

    def test_the_outcome_scale_factor_reaches_the_prediction(self) -> None:
        """``q_bounds=(-2, 5)``: the curve is 7 times the binary curve."""

        class Scaled(law.DiscreteLaw):
            def __init__(self) -> None:
                super().__init__()
                self.q = 7.0 * self.q - 2.0

        frame = law.frame()
        frame["Y"] = 7.0 * frame["Y"] - 2.0
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        settings = {**oracle(), "outcome_learner": OracleOutcome(Scaled()), "q_bounds": (-2.0, 5.0)}
        curve = study.identify(linear_contrast()).estimate(**settings).effect_curve()
        binary = law_study().identify(linear_contrast()).estimate(**oracle()).effect_curve()
        np.testing.assert_allclose(curve.predict(GRID), 7.0 * binary.predict(GRID), atol=1e-10)
        assert curve.upper_bound - curve.lower_bound == pytest.approx(7.0)


def score_functional(score: np.ndarray, scale: str) -> Any:
    """``(risk difference, calibration intercept, calibration slope)`` of a fixed score."""
    phi = np.column_stack([np.ones(3), score])

    def functional(probs: Any) -> Any:
        p_w = probs.sum(axis=(1, 2))
        q = probs[:, :, 1] / probs.sum(axis=2)
        if scale == "difference":
            tau = q[:, 1] - q[:, 0]
            ate = (p_w * tau).sum()
            risk = (p_w * (score**2 - 2.0 * score * tau)).sum() + ate**2
            gram = np.einsum("wp,wq,w->pq", phi, phi, p_w)
            beta = np.linalg.solve(gram, np.einsum("wp,w->p", phi, p_w * tau))
            return np.array([risk, beta[0], beta[1]])
        total = q[:, 1] + q[:, 0]
        c = np.log((p_w * q[:, 1]).sum() / (p_w * q[:, 0]).sum())

        def loss(theta: Any) -> Any:
            return (p_w * (total * np.log(1.0 + np.exp(theta)) - q[:, 1] * theta)).sum()

        risk = loss(score) - loss(c + 0.0 * score)
        beta = np.zeros(2, dtype=probs.dtype)
        for _ in range(law.MSM_NEWTON_STEPS):
            e = 1.0 / (1.0 + np.exp(-(phi @ beta)))
            equation = np.einsum("wp,w->p", phi, p_w * (total * e - q[:, 1]))
            jacobian = np.einsum("wp,wq,w->pq", phi, phi, p_w * total * e * (1.0 - e))
            beta = beta - np.linalg.solve(jacobian, equation)
        return np.array([risk, beta[0], beta[1]])

    return functional


SCORE_NAMES = ("risk_difference", "calibration[(intercept)]", "calibration[S]")


def assess(scale: str, column: str = "S") -> Any:
    effect = law_study().identify(linear_contrast(scale))
    return effect.assess_score(
        column, computed_from=["W"], score_kind="external", scale=scale, **oracle()
    )


def score_curves(assessment: Any, names: tuple[str, ...] = SCORE_NAMES) -> np.ndarray:
    rows = law.first_row_of()
    return np.column_stack([np.asarray(assessment[name].influence_curve)[rows] for name in names])


class TestTheScoreAssessment:
    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_the_point_is_the_functional_and_the_curve_its_derivative(self, scale: str) -> None:
        assessment = assess(scale)
        functional = score_functional(SCORE, scale)
        np.testing.assert_allclose(
            [assessment[name].psi for name in SCORE_NAMES], functional(law.PROBS), atol=1e-12
        )
        np.testing.assert_allclose(
            score_curves(assessment), law.gateaux_eif(functional), atol=1e-11, rtol=0
        )

    def test_the_rev_1_risk_curve_misses_the_derivative(self) -> None:
        """Mutation: ``+ 2 (c - E s)(Dtilde - c)`` in place of ``+ 2 ATE (Dtilde - ATE)``."""
        assessment = assess("difference")
        q = law.Q_EXACT
        tau = q[:, 1] - q[:, 0]
        ate = float((law.P_W * tau).sum())
        mean_score = float((law.P_W * SCORE).sum())
        w, a, y = (np.array(column) for column in zip(*law.SUPPORT, strict=True))
        g = np.where(a == 1, law.G_EXACT[w], 1.0 - law.G_EXACT[w])
        dtilde = (2 * a - 1) / g * (y - q[w, a]) + tau[w]
        reported = score_curves(assessment)[:, 0]
        rev1 = reported - 2.0 * ate * (dtilde - ate) + 2.0 * (ate - mean_score) * (dtilde - ate)
        expected = law.gateaux_eif(score_functional(SCORE, "difference"))[:, 0]
        np.testing.assert_allclose(reported, expected, atol=1e-11, rtol=0)
        assert np.max(np.abs(rev1 - expected)) > 0.1

    def test_dropping_the_constant_block_misses_the_log_ratio_derivative(self) -> None:
        """Mutation: the risk of the score alone, without ``- [L(c) + Dlt(c) - R(c)]``."""
        functional = score_functional(SCORE, "log_ratio")

        def score_risk_only(probs: Any) -> Any:
            p_w = probs.sum(axis=(1, 2))
            q = probs[:, :, 1] / probs.sum(axis=2)
            total = q[:, 1] + q[:, 0]
            return (p_w * (total * np.log(1.0 + np.exp(SCORE)) - q[:, 1] * SCORE)).sum()

        full = law.gateaux_eif(functional)[:, 0]
        alone = law.gateaux_eif(score_risk_only)
        assert np.max(np.abs(full - alone)) > 0.05
        np.testing.assert_allclose(score_curves(assess("log_ratio"))[:, 0], full, atol=1e-11)

    def test_the_true_cate_is_calibrated_and_its_risk_is_minus_its_variance(self) -> None:
        assessment = assess("difference", column="T")
        tau = law.Q_EXACT[:, 1] - law.Q_EXACT[:, 0]
        variance = float((law.P_W * tau**2).sum() - (law.P_W * tau).sum() ** 2)
        assert assessment["calibration[(intercept)]"].psi == pytest.approx(0.0, abs=1e-10)
        assert assessment["calibration[T]"].psi == pytest.approx(1.0, abs=1e-10)
        assert assessment["risk_difference"].psi == pytest.approx(-variance, abs=1e-12)


class TestScoreRefusalsBeforeAnyLearner:
    def effect(self) -> Any:
        return law_study().identify(linear_contrast())

    def test_r13_iii_a_score_not_declared_external(self) -> None:
        with pytest.raises(CapabilityError, match="score_kind='external'"):
            self.effect().assess_score("S", computed_from=["W"], **never())
        assert NeverFit.calls == 0

    @pytest.mark.parametrize("inputs", [None, ["Y"], "W"])
    def test_r13_ii_inputs_outside_the_adjustment_set(self, inputs: Any) -> None:
        with pytest.raises(CapabilityError, match="known function of adjustment columns"):
            self.effect().assess_score("S", computed_from=inputs, score_kind="external", **never())
        assert NeverFit.calls == 0

    def test_r13_i_a_curve_fitted_on_these_rows(self) -> None:
        effect = self.effect()
        curve = effect.estimate(**oracle()).effect_curve()
        frame = law.frame()
        frame["fitted"] = curve.in_sample
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        refit = study.identify(linear_contrast())
        refit._register_curve(curve.in_sample)
        with pytest.raises(CapabilityError, match="fitted on the rows it would be assessed on"):
            refit.assess_score("fitted", computed_from=["W"], score_kind="external", **never())
        frame = frame.rename(columns={"fitted": "effect_curve:cate"})
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        with pytest.raises(CapabilityError, match="fitted on the rows it would be assessed on"):
            study.identify(linear_contrast()).assess_score(
                "effect_curve:cate", computed_from=["W"], score_kind="external", **never()
            )
        assert NeverFit.calls == 0

    def test_an_effect_registers_the_curves_fitted_from_it(self) -> None:
        effect = self.effect()
        curve = effect.estimate(**oracle()).effect_curve()
        assert any(np.array_equal(stored, curve.in_sample) for stored in effect._curves)


def test_a_learned_fit_runs_end_to_end() -> None:
    """Cross-fitted GLM nuisances on a continuous modifier, both scales."""
    rng = np.random.default_rng(3)
    n = 600
    w = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3 + 0.2 * a * (w > 0))
    frame = pd.DataFrame({"Y": y, "A": a, "W": w})
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)))
    for scale in ("difference", "log_ratio"):
        fit = study.identify(
            ConditionalContrast(Modifiers(["W"]), scale=scale, model=Linear(["W"]))  # type: ignore[arg-type]
        ).estimate(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
        )
        assert fit.heterogeneity_test().df == 1
        assert np.isfinite(fit.effect_curve().in_sample).all()
