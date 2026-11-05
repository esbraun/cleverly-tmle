r"""The EP learner through the study layer: exactness at the oracle, refusals, folds, records.

``tests/unit/test_ep_exact.py`` pins the debiasing arithmetic on frequency-weighted
populations.  This module runs ``estimate("ep", ...)`` end to end:

- at the oracle learners on :mod:`tests.discrete_law` (and its missing-outcome sibling) with
  a wrong outcome regression, the EP projection equals the truth, and dropping the ``1 / pi``
  or the ``1 / Pi`` the fit reads off the nuisance fit breaks it;
- the curve is ``b - a`` times the scaled contrast, and is clipped to the outcome range with
  ``model=None`` only;
- at a fixed size the EP coefficients solve the projection's efficient estimating equation
  (the replacement for the plan's equality test, which needs a weighted-form projection TMLE
  the engine does not fit with more than one basis column);
- the sieve-size cross-validation is fold-local: flipping the outcomes of one fold leaves that
  fold's curves unchanged, and a mutation that lets the inner refit see the fold moves them;
- outer and inner folds keep clusters whole, and the ensemble's candidates receive the row
  weights in every outer and inner fit;
- every EP refusal raises before any learner runs.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (
    ATE,
    CausalStudy,
    ConditionalContrast,
    EPMethod,
    PointTreatment,
)
from cleverly.bases import Constant, Cosine, Indicators, Linear, Modifiers, Sieve
from cleverly.data.causal_data import CausalData
from cleverly.effect_curves import EffectCurveResult
from cleverly.estimators import ep
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from tests import discrete_law as law
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.unit._natural_course_support import NeverFit

MODIFIERS = Modifiers(["W"], domains={"W": (0.0, 2.0)})
LEVELS = Modifiers(["W"], levels={"W": (0.0, 1.0, 2.0)})

#: A wrong outcome regression, indexed ``[w, a]``, as in the projection's exact tests.
Q_ERROR = np.array([[0.10, -0.05], [0.15, 0.10], [-0.20, 0.05]])


def wrong(base: Any) -> Any:
    out = type(base)()
    out.q = np.clip(base.q + Q_ERROR, 0.05, 0.95)
    return out


def law_study(frame: pd.DataFrame | None = None, **design: Any) -> CausalStudy:
    return CausalStudy(
        law.frame() if frame is None else frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), **design),
    )


def oracle(outcome_law: Any = None, dgp: Any = None) -> dict[str, Any]:
    dgp = law.DiscreteLaw() if dgp is None else dgp
    return {
        "outcome_learner": OracleOutcome(dgp if outcome_law is None else outcome_law),
        "treatment_learner": OracleTreatment(dgp),
        "n_folds": 2,
        "random_state": 0,
    }


def never() -> dict[str, Any]:
    NeverFit.calls = 0
    return {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}


def beta(curve: EffectCurveResult) -> np.ndarray:
    return np.asarray(curve.model.beta) * curve.model.width


def truth(scale: str = "difference") -> np.ndarray:
    q = law.Q_EXACT
    return law.effect_modification_beta(law.P_W, q[:, 1], q[:, 0], scale)


def linear_fit(**overrides: Any) -> EffectCurveResult:
    contrast = ConditionalContrast(MODIFIERS, model=Linear(["W"]))  # type: ignore[arg-type]
    return law_study().identify(contrast).estimate("ep", **overrides)


# ---------------------------------------------------------------- exact at the oracle


class TestExactAtTheOracle:
    def test_the_projection_is_the_truth_at_a_wrong_outcome_regression(self) -> None:
        fit = linear_fit(**oracle(wrong(law.DiscreteLaw())))
        np.testing.assert_allclose(beta(fit), truth(), atol=1e-10)
        assert fit.fitted_method == "ep"
        assert fit.inference_status == "no_interval"

    def test_the_t_learner_at_that_regression_is_off(self) -> None:
        q = wrong(law.DiscreteLaw()).q
        plug_in = law.effect_modification_beta(law.P_W, q[:, 1], q[:, 0], "difference")
        assert np.max(np.abs(plug_in - truth())) > 0.02

    def test_dropping_the_inverse_propensity_breaks_it(self, monkeypatch: Any) -> None:
        pieces = ep._pieces
        monkeypatch.setattr(
            ep, "_pieces", lambda fit: replace(pieces(fit), weight=pieces(fit).rows)
        )
        fit = linear_fit(**oracle(wrong(law.DiscreteLaw())))
        assert np.max(np.abs(beta(fit) - truth())) > 1e-3

    def test_the_saturated_log_ratio_is_the_per_cell_log_crr(self) -> None:
        contrast = ConditionalContrast(LEVELS, scale="log_ratio", model=Indicators("W"))
        fit = law_study().identify(contrast).estimate("ep", **oracle(wrong(law.DiscreteLaw())))
        q = law.Q_EXACT
        np.testing.assert_allclose(
            fit.predict({"W": np.array([0.0, 1.0, 2.0])}), np.log(q[:, 1] / q[:, 0]), atol=1e-10
        )

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_missing_outcomes_read_the_observation_weight(
        self, scale: str, monkeypatch: Any
    ) -> None:
        dgp = mar.DiscreteLaw()
        study = CausalStudy(
            mar.frame(),
            design=PointTreatment(
                outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"
            ),
        )
        basis = Linear(["W"]) if scale == "difference" else Indicators("W")
        modifiers = MODIFIERS if scale == "difference" else LEVELS
        effect = study.identify(ConditionalContrast(modifiers, scale=scale, model=basis))
        kwargs = {**oracle(wrong(dgp), dgp), "missingness_learner": OracleMissingness(dgp)}
        observed = mar.PROBS[:, :, mar.OBSERVED_ONE]
        q = observed / (mar.PROBS[:, :, mar.OBSERVED_ZERO] + observed)
        if scale == "difference":
            expected = law.effect_modification_beta(
                mar.PROBS.sum(axis=(1, 2)), q[:, 1], q[:, 0], scale
            )
            read = beta
        else:
            expected = np.log(q[:, 1] / q[:, 0])

            def read(fit: Any) -> np.ndarray:
                return fit.predict({"W": np.array([0.0, 1.0, 2.0])})

        fit = effect.estimate("ep", **kwargs)
        np.testing.assert_allclose(read(fit), expected, atol=1e-10)
        assert "missing outcomes" in fit.status_reason

        nuisance_type = type(fit.fit.nuisance)
        pieces = ep._pieces

        def without_pi(nuisance_fit: Any) -> Any:
            with monkeypatch.context() as patch:
                patch.setattr(nuisance_type, "bounded_missingness", lambda self, lower: None)
                return pieces(nuisance_fit)

        monkeypatch.setattr(ep, "_pieces", without_pi)
        mutated = effect.estimate("ep", **kwargs)
        if scale == "difference":
            assert np.max(np.abs(read(mutated) - expected)) > 1e-3
        # On the log ratio a basis saturated in W is exact under any weights, so the mutation
        # cannot show there; tests/unit/test_ep_exact.py shows it with V coarser than W.


class TestTheProjectionEquation:
    """Plan 4.3, as the engine allows it (see the module docstring)."""

    @pytest.fixture(scope="class")
    def learned(self) -> tuple[Any, Any]:
        rng = np.random.default_rng(11)
        n = 800
        w = rng.uniform(0, 2, n)
        a = rng.binomial(1, 0.3 + 0.3 * (w > 1))
        y = rng.binomial(1, 0.3 + 0.1 * w + a * (0.1 + 0.15 * np.sin(2 * w)))
        frame = pd.DataFrame({"Y": y, "A": a, "W": w})
        effect = law_study(frame).identify(
            ConditionalContrast(MODIFIERS, model=Linear(["W"]))  # type: ignore[arg-type]
        )
        kwargs = {
            "outcome_learner": LogisticRegression(),
            "treatment_learner": LogisticRegression(),
            "n_folds": 3,
            "random_state": 0,
        }
        return effect.estimate("ep", **kwargs), effect.estimate(**kwargs)

    def test_the_ep_coefficients_solve_the_efficient_estimating_equation(
        self, learned: Any
    ) -> None:
        fit, _ = learned
        pieces = ep._pieces(fit.fit.nuisance_fit)
        phi = fit.model.design({"W": fit.fit.data.covariates[:, 0]})
        debiased = fit.fit.debiasing
        coef = np.asarray(fit.model.beta)
        sign = (pieces.treatment == 1.0).astype(float) - (pieces.treatment == 0.0).astype(float)
        fitted = np.where(pieces.treatment == 1.0, debiased.targeted_t, debiased.targeted_r)
        residual = pieces.weight * sign * (pieces.y - fitted)
        plug_in = debiased.contrast - phi @ coef
        # The two parts of the projection's efficient influence curve, at the EP fit.
        assert np.max(np.abs(phi.T @ residual / pieces.y.size)) < 1e-10
        assert np.max(np.abs(phi.T @ plug_in / pieces.y.size)) < 1e-10

    def test_the_ep_and_tmle_projections_agree_closely(self, learned: Any) -> None:
        fit, tmle = learned
        np.testing.assert_allclose(
            beta(fit), [estimate.psi for estimate in tmle.estimates.values()], atol=0.02
        )

    def test_at_the_oracle_they_agree_exactly(self) -> None:
        contrast = ConditionalContrast(MODIFIERS, model=Linear(["W"]))  # type: ignore[arg-type]
        effect = law_study().identify(contrast)
        dgp = law.DiscreteLaw()
        fit = effect.estimate("ep", **oracle(wrong(dgp)))
        tmle = effect.estimate(
            outcome_learner=OracleOutcome(wrong(dgp)),
            treatment_learner=OracleTreatment(dgp),
            cross_fit=False,
            random_state=0,
        )
        np.testing.assert_allclose(
            beta(fit), [estimate.psi for estimate in tmle.estimates.values()], atol=1e-10
        )


# ---------------------------------------------------------------- scale, clip, records


class ConstantRegressor(RegressorMixin, BaseEstimator):
    """A final learner whose prediction is a declared constant on the scaled outcome."""

    def __init__(self, value: float = 5.0) -> None:
        self.value = value

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantRegressor:
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.value)


class TestTheOutcomeScale:
    def test_the_curve_is_b_minus_a_times_the_scaled_contrast(self) -> None:
        frame = law.frame()
        frame["Y"] = -2.0 + 7.0 * frame["Y"]
        contrast = ConditionalContrast(MODIFIERS, model=Linear(["W"]))  # type: ignore[arg-type]
        fit = (
            law_study(frame)
            .identify(contrast)
            .estimate("ep", q_bounds=(-2.0, 5.0), **oracle(wrong(law.DiscreteLaw())))
        )
        assert fit.model.width == pytest.approx(7.0)
        np.testing.assert_allclose(beta(fit), 7.0 * truth(), atol=1e-9)
        np.testing.assert_allclose(np.asarray(fit.model.beta), truth(), atol=1e-10)
        new = {"W": np.array([0.0, 2.0])}
        np.testing.assert_allclose(
            fit.predict(new), 7.0 * (law.EFFECT_MODIFICATION_DESIGN[[0, 2]] @ truth()), atol=1e-9
        )

    def test_model_none_clips_to_the_outcome_range(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS))
        fit = effect.estimate(
            "ep",
            sieve=Sieve(Linear(["W"]), size=1),
            final_learner=ConstantRegressor(5.0),
            **oracle(),
        )
        np.testing.assert_allclose(fit.predict({"W": np.array([0.5, 1.5])}), 1.0)
        assert fit.clipped_share == 1.0
        assert fit.model.raw({"W": np.array([0.5])})[0] == pytest.approx(5.0)
        assert "final learner" in fit.status_reason

    def test_a_projection_is_not_clipped(self) -> None:
        fit = linear_fit(**oracle())
        assert not fit.model.clip
        assert fit.clipped_share == 0.0


class TestTheRecords:
    @pytest.fixture(scope="class")
    def cv_fit(self) -> EffectCurveResult:
        frame = continuous_frame(400, seed=2)
        effect = law_study(frame).identify(ConditionalContrast(MODIFIERS))
        return effect.estimate(
            "ep",
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            sieve=Sieve(Cosine(["W"], 3)),
            n_folds=3,
            random_state=0,
        )

    def test_the_sieve_record(self, cv_fit: EffectCurveResult) -> None:
        record = cv_fit.sieve
        assert record.mode == "cv"
        assert record.sizes == (1, 2, 3)
        assert len(record.cv_risks) == 3
        assert record.chosen == record.sizes[int(np.argmin(record.cv_risks))]
        assert len(record.inner_seeds) == 3
        assert len(set(record.inner_seeds)) == 3

    def test_the_label_is_experimental_with_the_cv_reason(self, cv_fit: EffectCurveResult) -> None:
        assert cv_fit.status == "experimental"
        assert "cross-validation" in cv_fit.status_reason

    def test_the_curve_refuses_an_interval(self, cv_fit: EffectCurveResult) -> None:
        with pytest.raises(CapabilityError) as caught:
            cv_fit.predict({"W": np.array([1.0])}, interval=True)
        assert caught.value.refusal == ep.EP_INTERVAL_REFUSAL

    def test_the_score_report(self, cv_fit: EffectCurveResult) -> None:
        scores = cv_fit.diagnostics.score_equations()
        assert isinstance(scores, ep.DebiasingScores)
        assert scores.max_after < 1e-8
        assert len(scores.features) == cv_fit.sieve.chosen + 1
        assert list(scores.to_frame().columns) == ["feature", "before", "after"]

    def test_the_summary_names_the_method_and_the_missing_interval(
        self, cv_fit: EffectCurveResult
    ) -> None:
        text = str(cv_fit.summary())
        assert "EP learner" in text
        assert "interval" in text

    def test_persistence(self, cv_fit: EffectCurveResult, tmp_path: Any) -> None:
        path = tmp_path / "curve.joblib"
        cv_fit.save(path)
        loaded = cleverly.load(path)
        new = {"W": np.array([0.1, 0.9, 1.7])}
        np.testing.assert_array_equal(loaded.predict(new), cv_fit.predict(new))


def continuous_frame(n: int, *, seed: int, clusters: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 2, n)
    a = rng.binomial(1, 0.3 + 0.3 * (w > 1))
    y = rng.binomial(1, 0.2 + 0.2 * w + a * 0.2 * np.cos(2 * w))
    frame = pd.DataFrame({"Y": y.astype(float), "A": a, "W": w})
    if clusters is not None:
        frame["cluster"] = rng.integers(0, clusters, n)
    return frame


# ---------------------------------------------------------------- fold locality


class CurveSpy:
    """Records every curve the EP fit computes, in call order."""

    def __init__(self, monkeypatch: Any) -> None:
        self.betas: list[np.ndarray] = []
        self.training: list[Any] = []
        curve_at = ep.EPEstimator._curve_at
        nuisances = ep.EPEstimator._nuisances
        spy = self

        def recording_curve(self: Any, nuisance_fit: Any, size: int) -> Any:
            curve, debiased = curve_at(self, nuisance_fit, size)
            spy.betas.append(np.asarray(curve.beta).copy())
            return curve, debiased

        def recording_nuisances(self: Any, data: CausalData, **changes: Any) -> Any:
            fitted = nuisances(self, data, **changes)
            spy.training.append((data, fitted))
            return fitted

        monkeypatch.setattr(ep.EPEstimator, "_curve_at", recording_curve)
        monkeypatch.setattr(ep.EPEstimator, "_nuisances", recording_nuisances)


def cv_estimate(frame: pd.DataFrame, **design: Any) -> EffectCurveResult:
    effect = law_study(frame, **design).identify(ConditionalContrast(MODIFIERS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(
            "ep",
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            sieve=Sieve(Cosine(["W"], 3)),
            n_folds=3,
            random_state=0,
        )


class TestTheCrossValidationIsFoldLocal:
    SIZES = 3

    def fold_betas(self, frame: pd.DataFrame, monkeypatch: Any) -> tuple[list[Any], np.ndarray]:
        with monkeypatch.context() as patch:
            spy = CurveSpy(patch)
            fit = cv_estimate(frame)
        assignment = np.asarray(fit.fit.nuisance.folds.assignment)
        return spy.betas, assignment

    def flipped(self, frame: pd.DataFrame, assignment: np.ndarray, fold: int) -> pd.DataFrame:
        out = frame.copy()
        at = assignment == fold
        out.loc[at, "Y"] = 1.0 - out.loc[at, "Y"]
        return out

    def test_flipping_a_fold_leaves_that_folds_curves_unchanged(self, monkeypatch: Any) -> None:
        frame = continuous_frame(450, seed=5)
        before, assignment = self.fold_betas(frame, monkeypatch)
        after, again = self.fold_betas(self.flipped(frame, assignment, 0), monkeypatch)
        np.testing.assert_array_equal(assignment, again)
        size = self.SIZES
        for index in range(size):
            np.testing.assert_allclose(after[index], before[index], atol=1e-12)
        moved = [np.max(np.abs(after[i] - before[i])) for i in range(size, 3 * size)]
        assert min(moved) > 1e-4

    def test_an_inner_refit_that_sees_the_fold_moves_them(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(CausalData, "subset", lambda self, index: self)
        frame = continuous_frame(450, seed=5)
        before, assignment = self.fold_betas(frame, monkeypatch)
        after, _ = self.fold_betas(self.flipped(frame, assignment, 0), monkeypatch)
        assert np.max(np.abs(after[0] - before[0])) > 1e-4


class TestClustersStayWhole:
    def clusters_whole(self, assignment: np.ndarray, cluster: np.ndarray) -> bool:
        return all(np.unique(assignment[cluster == c]).size == 1 for c in np.unique(cluster))

    def test_outer_and_inner_folds_keep_clusters_whole(self, monkeypatch: Any) -> None:
        frame = continuous_frame(400, seed=7, clusters=60)
        spy = CurveSpy(monkeypatch)
        cv_estimate(frame, cluster="cluster")
        assert len(spy.training) == 1 + 3
        for data, fitted in spy.training:
            assert data.cluster is not None
            assignment = np.asarray(fitted.nuisance.folds.assignment)
            assert self.clusters_whole(assignment, np.asarray(data.cluster))

    def test_a_row_split_breaks_a_cluster(self, monkeypatch: Any) -> None:
        tmle = importlib.import_module("cleverly.estimators.tmle")
        make_folds = tmle.make_folds

        def by_row(*args: Any, cluster: Any = None, **kwargs: Any) -> Any:
            return make_folds(*args, cluster=None, **kwargs)

        monkeypatch.setattr(tmle, "make_folds", by_row)
        frame = continuous_frame(400, seed=7, clusters=60)
        spy = CurveSpy(monkeypatch)
        cv_estimate(frame, cluster="cluster")
        assert spy.training
        assert not all(
            self.clusters_whole(np.asarray(f.nuisance.folds.assignment), np.asarray(d.cluster))
            for d, f in spy.training
        )


# ---------------------------------------------------------------- ensemble weights


class WeightSpy(RegressorMixin, BaseEstimator):
    """A regression candidate that records the ``sample_weight`` each of its fits receives."""

    seen: ClassVar[list[Any]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WeightSpy:
        WeightSpy.seen.append(None if sample_weight is None else np.array(sample_weight))
        self.fitted_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> Any:
        return self.fitted_.predict(X)


def test_the_row_weights_reach_every_outer_and_inner_fit() -> None:
    frame = continuous_frame(300, seed=9)
    frame["Y"] = np.clip(0.5 * frame["Y"] + 0.5 * frame["W"] / 2.0, 0.0, 1.0)
    frame["w"] = np.random.default_rng(1).uniform(0.5, 2.0, len(frame))
    effect = law_study(frame, weights="w").identify(ConditionalContrast(MODIFIERS))
    WeightSpy.seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = effect.estimate(
            "ep",
            outcome_learner=WeightSpy(),
            treatment_learner=LogisticRegression(),
            q_bounds=(0.0, 1.0),
            sieve=Sieve(Cosine(["W"], 2)),
            n_folds=2,
            random_state=0,
        )
    # Two outer folds, then two training folds each refitted on five inner folds.
    assert len(WeightSpy.seen) == 2 + 2 * 5
    assert all(weights is not None and np.ptp(weights) > 0.5 for weights in WeightSpy.seen)
    assert "observation weights" in fit.status_reason


# ---------------------------------------------------------------- refusals before any learner


class TestRefusalsBeforeAnyLearner:
    def refuses(self, effect: Any, match: str, **overrides: Any) -> CapabilityError:
        with pytest.raises(CapabilityError, match=match) as caught:
            effect.estimate("ep", **never(), **overrides)
        assert NeverFit.calls == 0
        return caught.value

    def test_ep_is_available_for_a_conditional_contrast(self) -> None:
        for contrast in (
            ConditionalContrast(MODIFIERS),
            ConditionalContrast(MODIFIERS, model=Linear(["W"])),
        ):  # type: ignore[arg-type]
            methods = {m.name: m for m in law_study().identify(contrast).available_methods()}
            assert methods["ep"].available

    def test_r11_another_estimand(self) -> None:
        methods = {m.name: m for m in law_study().identify(ATE()).available_methods()}
        assert methods["ep"].refusal.reason is RefusalReason.DIFFERENT_ESTIMAND

    def test_r1_an_interval_setting(self) -> None:
        from cleverly.methods import Inference, MethodConfigurationError

        with pytest.raises(MethodConfigurationError, match="without confidence intervals"):
            EPMethod(inference=replace(Inference(), alpha=0.1))

    def test_a_cv_sized_model(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS, model=Sieve(Cosine(["W"], 3))))
        self.refuses(effect, "cross-validation")

    def test_two_bases(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS, model=Linear(["W"])))  # type: ignore[arg-type]
        error = self.refuses(effect, "two bases", sieve=Sieve(Cosine(["W"], 3), size=2))
        assert error.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_final_learner_with_a_model(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS, model=Linear(["W"])))  # type: ignore[arg-type]
        error = self.refuses(effect, "projection onto that model", final_learner=LinearRegression())
        assert error.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_final_learner_on_the_log_ratio(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS, scale="log_ratio"))
        error = self.refuses(effect, "", final_learner=LinearRegression())
        assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    @pytest.mark.parametrize(
        "setting",
        [
            {"cross_fit": False},
            {"repeats": 2},
            {"cv_evaluation": True},
            {"target_weights": True},
            {"max_iter": 5},
        ],
    )
    def test_settings_the_ep_fit_does_not_read(self, setting: dict[str, Any]) -> None:
        from cleverly.methods import MethodConfigurationError

        effect = law_study().identify(ConditionalContrast(MODIFIERS))
        with pytest.raises(MethodConfigurationError, match="EPMethod does not read") as caught:
            effect.estimate("ep", **never(), **setting)
        assert NeverFit.calls == 0
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_default_sieve_needs_a_domain(self) -> None:
        effect = law_study().identify(ConditionalContrast(Modifiers(["W"])))
        with pytest.raises(CapabilityError) as caught:
            effect.estimate("ep", **never())
        assert NeverFit.calls == 0
        assert "domain" in str(caught.value)

    def test_a_collinear_member(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS))
        with pytest.raises(DataError, match="collinear"):
            effect.estimate("ep", **never(), sieve=Sieve(Cosine(["W"], 4), size=4))
        assert NeverFit.calls == 0

    def test_the_constant_member_is_the_ate(self) -> None:
        contrast = ConditionalContrast(MODIFIERS, model=Constant())  # type: ignore[arg-type]
        fit = law_study().identify(contrast).estimate("ep", **oracle(wrong(law.DiscreteLaw())))
        assert beta(fit)[0] == pytest.approx(law.TRUTH["ate"], abs=1e-10)
