r"""Where the plug-in omitted-variable limits stop, clause by clause, and what the fit records.

Under ``nu2_estimator="plugin"`` the bound's limits open when the fit's treatment mechanism is
declared known or is a qualifying scikit-learn ``LogisticRegression`` (X26).  Every other fit
stores a :class:`~cleverly.sensitivity._mechanism_stack.MechanismRefusal`, and the six limit
accessors raise it.  This module pins the refusal and the capture the rule reads:

*   one full-string test per clause key, with the reason code, the roadmap item and the
    alternative of the :class:`~cleverly.exceptions.CapabilityError` the accessor raises;
*   a qualifying twin of the learner clauses, so that a clause fires on its own condition
    and not on an unrelated one;
*   the capture (:class:`~cleverly.estimators._mechanism_fit.MechanismFit`): the predictions
    do not move, a process pool records the same folds in the same order, and the record
    survives a pickle and ``save``;
*   one plug-in ``benchmark()`` on a qualifying estimated fit;
*   the order of the accessors, where the inference status answers before the mechanism.

The numeric clauses (``score_not_zero``, ``singular``) quote a number that depends on the
solver, so their tests match the number by pattern and the rest of the text exactly.  The
derivation of the curve is tested in ``test_omitted_variable_plugin_limits.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import cleverly
from cleverly._inference_status import STATUS_RECORDS, supplies_inference
from cleverly.datasets import make_linear_ate, make_nonlinear_ate
from cleverly.estimators import TMLE
from cleverly.estimators._mechanism_fit import MechanismFit, ScalerStep, describe_mechanism
from cleverly.estimators._nuisance import ConstantFit, cross_fit_companion
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import Incremental
from cleverly.learners import SuperLearner, random_partition
from cleverly.sensitivity import _second_order, omitted_variable
from cleverly.sensitivity._mechanism_stack import MechanismRefusal, plugin_mechanism_term
from cleverly.sensitivity.omitted_variable import (
    benchmark,
    omitted_variable_bounds,
    resolve_parameter,
    robustness_value,
    sensitivity_elements,
)
from tests.unit._direct_effect_support import cde_frame, fit_cde

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

#: The strength at which each bound is read.
STRENGTH = {"cf_y": 0.3, "cf_d": 0.3}

#: The six accessors that read the bound's curve.
LIMIT_ACCESSORS: tuple[str, ...] = (
    "ci_lower",
    "ci_upper",
    "robustness_value_ci",
    "plugin_interval_lower",
    "plugin_interval_upper",
    "robustness_value_plugin_interval",
)

# ---------------------------------------------------------------------- the exact text

#: The stem every plug-in refusal shares, written out here and not imported, so that an edit
#: to the source sentence fails this module.
STEM = (
    "the plug-in nu^2 has a standard error only through the score of the fit's own treatment "
    "mechanism. E_n[alpha_hat^2] squares the fitted Riesz representer, so it moves at first "
    "order with the fitted mechanism. The limits need that mechanism to be declared known, or "
    "to be a scikit-learn LogisticRegression, binary or multinomial, with no penalty or a "
    "fixed L2 penalty, fitted bare or after StandardScaler. For that model the stacked "
    "estimating equation of the mechanism score and the second moment gives the curve (Saul "
    "and Hudgens 2020). The bias-adjusted bounds, rv, max_bias, benchmark() and contour() "
    "remain available"
)

#: The sentence a learner the rule does not admit adds to the stem.
UNCITED = (
    ". No source this package cites gives the curve for any other mechanism learner. "
    "docs/roadmap.md F26 tracks this stop"
)

#: The alternative of every clause that has no remedy of its own.
DOUBLY_ROBUST = (
    "Fit the bound with nu2_estimator='doubly_robust', which has derived limits for any "
    "mechanism learner."
)


def full_text(clause: str, *, learner: bool) -> str:
    """The text a refusal stores, for a clause and the kind of key it belongs to."""
    stem = STEM + (UNCITED if learner else "")
    return f"nu2_estimator='plugin': {stem}. This fit does not qualify: {clause}."


def full_pattern(before: str, after: str) -> re.Pattern[str]:
    """The text of a fold clause that quotes one number, with the number matched by pattern."""
    prefix = full_text(before, learner=False)[: -len(".")]
    return re.compile(re.escape(prefix) + r"[0-9.e+-]+" + re.escape(after + "."))


# ------------------------------------------------------------------------------ the fits


def binary_frame(n: int = 1000, seed: int = 1) -> pd.DataFrame:
    """A two-arm confounded frame with four covariates, ``W1`` to ``W4``."""
    return make_linear_ate(n=n, seed=seed)[0]


def three_arm_frame(n: int = 1000, seed: int = 5) -> pd.DataFrame:
    """A three-arm frame with a binary outcome, whose mechanism is multinomial logistic."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    eta = np.column_stack([np.zeros(n), 0.5 * w[:, 0], -0.4 * w[:, 1]])
    probabilities = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    arm = np.array([rng.choice(3, p=row) for row in probabilities], dtype=float)
    outcome = rng.random(n) < expit(0.3 * w[:, 0] + 0.4 * (arm == 1) + 0.8 * (arm == 2))
    return pd.DataFrame({"Y": outcome.astype(float), "A": arm, "W1": w[:, 0], "W2": w[:, 1]})


def known_frame(n: int = 300, seed: int = 4) -> pd.DataFrame:
    """A two-arm frame whose true propensity is a column, for a declared known mechanism."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    p1 = expit(1.2 * w[:, 0])
    arm = (rng.random(n) < p1).astype(float)
    outcome = arm + w[:, 0] - 0.5 * w[:, 1] + 0.5 * rng.normal(size=n)
    return pd.DataFrame(
        {"Y": outcome, "A": arm, "W1": w[:, 0], "W2": w[:, 1], "p0": 1.0 - p1, "p1": p1}
    )


def fit_binary(
    treatment_learner: Any = None, frame: pd.DataFrame | None = None, **settings: Any
) -> Any:
    """An in-sample two-arm TMLE with a linear outcome regression and one treatment learner."""
    arguments: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "cross_fit": False,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": 0,
        **settings,
    }
    if treatment_learner is not None:
        arguments["treatment_learner"] = treatment_learner
    return (
        TMLE(**arguments)
        .fit(binary_frame() if frame is None else frame, outcome="Y", treatment="A")
        .single()
    )


def fit_three_arm(**settings: Any) -> Any:
    """A cross-fitted three-arm TMLE with multinomial logistic nuisances, in two folds."""
    arguments: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": True,
        "n_folds": 2,
        "estimands": ("ey", "ate"),
        "simultaneous": False,
        "random_state": 0,
        **settings,
    }
    return TMLE(**arguments).fit(three_arm_frame(), outcome="Y", treatment="A").single()


def fit_known() -> Any:
    """An in-sample two-arm TMLE whose treatment probabilities are declared known."""
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            cross_fit=False,
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
            g_bounds=(0.001, 0.999),
        )
        .fit(
            known_frame(),
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            treatment_probabilities={0.0: "p0", 1.0: "p1"},
        )
        .single()
    )


def bound_of(result: Any, estimand: str = "ate") -> Any:
    """The plug-in bound of ``result`` at :data:`STRENGTH`."""
    return omitted_variable_bounds(result, estimand, nu2_estimator="plugin", **STRENGTH)


def with_nuisance(result: Any, **changes: Any) -> Any:
    """``result`` with fields of its first repeat's nuisance replaced."""
    repeat = result.repeats[0]
    return replace(result, repeats=(replace(repeat, nuisance=replace(repeat.nuisance, **changes)),))


def assert_refusal(
    bound: Any,
    *,
    key: str,
    text: str | re.Pattern[str],
    alternative: str,
    roadmap: str | None,
    reason: RefusalReason = RefusalReason.INFERENCE_GAP,
) -> None:
    """The stored record, and the error, the mapping and the summary that quote it."""
    stored = bound.limit_refusal
    assert isinstance(stored, MechanismRefusal)
    assert stored.key == key
    if isinstance(text, str):
        assert stored.text == text
    else:
        assert text.fullmatch(stored.text), stored.text
    assert stored.alternative == alternative
    assert stored.roadmap == roadmap
    assert stored.reason is reason
    assert bound.plugin_mechanism is None
    for accessor in LIMIT_ACCESSORS:
        with pytest.raises(CapabilityError) as refused:
            getattr(bound, accessor)
        raised = refused.value.refusal
        assert (
            raised.condition == f"SensitivityBounds.{accessor} is not defined under {stored.text}"
        )
        assert raised.reason is reason
        assert raised.alternative == alternative
        assert raised.roadmap == roadmap
    assert bound.to_dict()["limit_refusal"] == stored.text
    assert bound.to_dict()["nu2_estimator"] == "plugin"
    assert stored.text in bound.summary()


# ------------------------------------------------------------------ the learner clauses


class TestEveryLearnerClauseCitesF26:
    """The keys that name a mechanism learner the rule does not admit.

    Each carries the roadmap item F26 and the doubly robust alternative, except where the
    clause has a remedy of its own.
    """

    def test_super_learner_with_a_logistic_member(self) -> None:
        library = [
            ("logistic", LogisticRegression(max_iter=1000)),
            ("prior", DummyClassifier(strategy="prior")),
        ]
        learner = SuperLearner(library=library, task="classification", n_folds=3, random_state=0)
        assert_refusal(
            bound_of(fit_binary(learner)),
            key="super_learner",
            text=full_text(
                "the treatment learner is a SuperLearner. A library that contains "
                "LogisticRegression does not qualify, because the bound reads the ensemble's "
                "fitted mechanism and not one member's",
                learner=True,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap="F26",
        )

    def test_the_default_learner_is_a_super_learner(self) -> None:
        """The one fit here that leaves the treatment learner to the package, at n = 60."""
        result = fit_binary(frame=binary_frame(n=60))
        assert_refusal(
            bound_of(result),
            key="super_learner",
            text=full_text(
                "the treatment learner is a SuperLearner. A library that contains "
                "LogisticRegression does not qualify, because the bound reads the ensemble's "
                "fitted mechanism and not one member's",
                learner=True,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap="F26",
        )

    def test_screened(self) -> None:
        result = fit_binary(LogisticRegression(max_iter=1000), screen_treatment=True)
        nuisance = result.repeats[0].nuisance
        assert nuisance.treatment_covariates == result.data.covariate_names
        assert_refusal(
            bound_of(result),
            key="screened",
            text=full_text(
                "screen_treatment=True selects the covariates of the treatment model inside "
                "each fold, so the mechanism is not one fixed parametric model",
                learner=True,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap="F26",
        )

    def test_cv_penalty(self) -> None:
        assert_refusal(
            bound_of(fit_binary(LogisticRegressionCV(max_iter=1000))),
            key="cv_penalty",
            text=full_text(
                "the treatment learner is LogisticRegressionCV, which selects C by "
                "cross-validation, so the penalty is data-adaptive",
                learner=True,
            ),
            alternative="Pass LogisticRegression with a fixed C.",
            roadmap="F26",
        )

    def test_other_learner(self) -> None:
        learner = RandomForestClassifier(n_estimators=10, random_state=0)
        assert_refusal(
            bound_of(fit_binary(learner)),
            key="other_learner",
            text=full_text(
                "the treatment learner is RandomForestClassifier, not LogisticRegression",
                learner=True,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap="F26",
        )

    def test_pipeline(self) -> None:
        learner = make_pipeline(PolynomialFeatures(2), LogisticRegression(max_iter=1000))
        assert_refusal(
            bound_of(fit_binary(learner)),
            key="pipeline",
            text=full_text(
                "the treatment learner is a Pipeline with the step PolynomialFeatures before "
                "the LogisticRegression. Only StandardScaler steps qualify",
                learner=True,
            ),
            alternative=(
                "Transform the covariates before the fit, or pass the bare LogisticRegression."
            ),
            roadmap="F26",
        )

    def test_pipeline_intercept(self) -> None:
        learner = make_pipeline(
            StandardScaler(), LogisticRegression(fit_intercept=False, max_iter=1000)
        )
        assert_refusal(
            bound_of(fit_binary(learner)),
            key="pipeline_intercept",
            text=full_text(
                "a StandardScaler centers the covariates and the LogisticRegression sets "
                "fit_intercept=False, so the mechanism depends on the estimated column means",
                learner=True,
            ),
            alternative="Set fit_intercept=True.",
            roadmap="F26",
        )

    def test_l1_penalty(self) -> None:
        learner = LogisticRegression(solver="saga", l1_ratio=1, C=1, max_iter=1000)
        assert_refusal(
            bound_of(fit_binary(learner)),
            key="l1_penalty",
            text=full_text(
                "the LogisticRegression has an L1 or elastic-net penalty, whose score has no "
                "derivative where a coefficient is zero",
                learner=True,
            ),
            alternative="Use l1_ratio=0 for an L2 penalty, or C=numpy.inf.",
            roadmap="F26",
        )

    def test_liblinear(self) -> None:
        assert_refusal(
            bound_of(fit_binary(LogisticRegression(solver="liblinear"))),
            key="liblinear",
            text=full_text(
                "the LogisticRegression uses solver='liblinear', which penalizes the intercept",
                learner=True,
            ),
            alternative="Use another solver.",
            roadmap="F26",
        )

    def test_class_weight(self) -> None:
        learner = LogisticRegression(class_weight="balanced", max_iter=1000)
        assert_refusal(
            bound_of(fit_binary(learner)),
            key="class_weight",
            text=full_text(
                "the LogisticRegression sets class_weight='balanced', which changes its score",
                learner=True,
            ),
            alternative="Declare observation weights with weights= instead.",
            roadmap="F26",
        )

    def test_a_cross_fitted_fit_refuses_on_the_learner_of_its_folds(self) -> None:
        result = fit_three_arm(
            treatment_learner=LogisticRegression(class_weight="balanced", max_iter=1000)
        )
        assert len(result.repeats[0].nuisance.treatment_fits) == 2
        assert bound_of(result, "ey[1.0]").limit_refusal.key == "class_weight"


class TestEveryLearnerClauseHasAQualifyingTwin:
    """Each clause above fires on its own condition: the fit with that condition lifted opens."""

    @pytest.mark.parametrize(
        "learner",
        [
            LogisticRegression(max_iter=1000),
            LogisticRegression(C=np.inf, max_iter=1000),
            LogisticRegression(C=0.5, max_iter=1000),
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            LogisticRegression(fit_intercept=False, max_iter=1000),
            LogisticRegression(solver="newton-cholesky", max_iter=1000),
        ],
        ids=[
            "bare",
            "unpenalized",
            "ridge",
            "scaler",
            "no_intercept",
            "newton_cholesky",
        ],
    )
    def test_a_qualifying_logistic_model_opens_the_limits(self, learner: Any) -> None:
        bound = bound_of(fit_binary(learner))
        assert bound.limit_refusal is None
        assert bound.plugin_mechanism == "logistic"
        assert np.isfinite(bound.ci_lower) and bound.ci_lower < bound.lower
        assert np.isfinite(bound.ci_upper) and bound.ci_upper > bound.upper
        assert "limit_refusal" not in bound.to_dict()

    @pytest.mark.parametrize("intercept", [True, False])
    def test_a_scaler_that_does_not_center_qualifies(self, intercept: bool) -> None:
        learner = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(fit_intercept=intercept, max_iter=1000),
        )
        assert bound_of(fit_binary(learner)).limit_refusal is None

    def test_a_multinomial_model_opens_the_limits(self) -> None:
        result = fit_three_arm(cross_fit=False)
        for estimand in ("ey[1.0]", "ate[2.0 vs 0.0]"):
            bound = bound_of(result, estimand)
            assert bound.limit_refusal is None and bound.plugin_mechanism == "logistic"
            assert np.isfinite(bound.ci_lower)

    def test_a_cross_fitted_fit_reaches_the_correction(self) -> None:
        """A cross-fitted fit has no refusal of its own: it meets the correction's checks.

        On this two-fold, 1,000-row fit the ``ey[1.0]`` logit sits near a bound, and the
        ``ate[2.0 vs 0.0]`` limits carry the correction and the plain influence variance.
        """
        result = fit_three_arm()
        assert bound_of(result, "ey[1.0]").limit_refusal.key == "truncation_active"
        bound = bound_of(result, "ate[2.0 vs 0.0]")
        assert bound.limit_refusal is None and bound.nu2_correction is not None
        assert np.isfinite(bound.ci_lower) and bound.ci_lower < bound.lower

    def test_a_correction_beyond_the_moment_is_nonpositive(self) -> None:
        """A correction larger than the moment refuses with ``corrected_nonpositive`` only."""
        original = _second_order.PluginStack.bias

        def huge(self: Any) -> Any:
            return dataclasses.replace(original(self), point=1.0e6)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_second_order.PluginStack, "bias", huge)
            refusal = bound_of(fit_binary(LogisticRegression(max_iter=1000))).limit_refusal
        assert refusal is not None and refusal.key == "corrected_nonpositive"
        assert "corrected variance" not in refusal.text

    def test_the_summary_names_the_stacked_mechanism(self) -> None:
        summary = bound_of(fit_binary(LogisticRegression(max_iter=1000))).summary()
        assert (
            "plug-in nu^2, second-order corrected by 0.021593; its standard error carries the "
            "score of the fitted LogisticRegression mechanism, stacked" in summary
        )

    def test_the_mappings_name_the_estimator_and_no_refusal(self) -> None:
        result = fit_binary(LogisticRegression(max_iter=1000))
        row = bound_of(result).to_dict()
        assert row["nu2_estimator"] == "plugin"
        assert "limit_refusal" not in row
        assert {"ci_lower", "ci_upper", "robustness_value_ci"} <= set(row)
        assert row["nu2_correction"] > 0.0
        report = robustness_value(result, "ate", nu2_estimator="plugin")
        assert list(report) == [
            "nu2_estimator",
            "nu2_correction",
            "rv",
            "rva",
            "max_bias",
        ]
        assert report["nu2_correction"] == row["nu2_correction"]


# ------------------------------------------------------------------------- the fold clauses


class TestEveryFoldClauseCarriesNoRoadmapItem:
    """A declared model that failed a numerical check is not what F26 tracks."""

    def test_score_not_zero_names_the_solver_stop(self) -> None:
        result = fit_binary(LogisticRegression(max_iter=1))
        assert_refusal(
            bound_of(result),
            key="score_not_zero",
            text=full_pattern(
                "the score of the LogisticRegression of fold 0 is not zero at its "
                "coefficients: the largest Newton step is ",
                " standard errors, and the solver stopped at max_iter",
            ),
            alternative="Lower tol, raise max_iter, or use solver='newton-cholesky'.",
            roadmap=None,
        )

    def test_score_not_zero_without_the_solver_stop(self) -> None:
        """A loose ``tol`` stops the solver before ``max_iter``, so the clause adds no remedy."""
        result = fit_binary(LogisticRegression(tol=0.5, max_iter=1000))
        fit = result.repeats[0].nuisance.treatment_fits[0]
        assert fit.n_iter < fit.max_iter
        assert_refusal(
            bound_of(result),
            key="score_not_zero",
            text=full_pattern(
                "the score of the LogisticRegression of fold 0 is not zero at its "
                "coefficients: the largest Newton step is ",
                " standard errors",
            ),
            alternative="Lower tol, raise max_iter, or use solver='newton-cholesky'.",
            roadmap=None,
        )

    def test_the_remedy_it_names_opens_the_limits(self) -> None:
        bound = bound_of(fit_binary(LogisticRegression(solver="newton-cholesky", tol=1e-10)))
        assert bound.limit_refusal is None and np.isfinite(bound.ci_lower)

    def test_singular(self) -> None:
        """A covariate on a scale that saturates every fitted probability at 0 or 1."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(size=n)
        x = np.sign(x) * (np.abs(x) + 1.0)
        z = rng.normal(size=n)
        arm = (x > 0).astype(float)
        frame = pd.DataFrame(
            {"Y": 0.5 * arm + 0.3 * z + 0.5 * rng.normal(size=n), "A": arm, "X": 1000.0 * x, "Z": z}
        )
        result = fit_binary(
            LogisticRegression(C=1e12, max_iter=10000, tol=1e-14), frame, g_bounds=(0.01, 0.99)
        )
        assert_refusal(
            bound_of(result),
            key="singular",
            text=full_pattern(
                "the information matrix of the mechanism in fold 0 is singular (condition number ",
                "): the fitted probabilities reach 0 or 1, so the coefficients are not identified",
            ),
            alternative="Remove the covariates that separate the arms, or use a fixed C.",
            roadmap=None,
        )

    def test_missing_arm(self) -> None:
        """A fold whose training rows hold one arm fewer fits a model of fewer classes.

        The package refuses a declared split with such a fold before the fit, so the capture is
        edited: fold 1 records two classes of the three.
        """
        result = fit_three_arm()
        fits = result.repeats[0].nuisance.treatment_fits
        assert [fit.classes.tolist() for fit in fits] == [[0.0, 1.0, 2.0]] * 2
        edited = (fits[0], replace(fits[1], classes=np.array([0.0, 1.0])))
        assert_refusal(
            bound_of(with_nuisance(result, treatment_fits=edited), "ey[1.0]"),
            key="missing_arm",
            text=full_text(
                "the training rows of fold 1 hold no unit of arm 2, so the LogisticRegression "
                "of that fold is not the 3-arm model",
                learner=False,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap=None,
        )

    def test_a_fold_that_fitted_no_model_is_a_missing_arm(self) -> None:
        result = fit_three_arm()
        fits = result.repeats[0].nuisance.treatment_fits
        constant = describe_mechanism(ConstantFit(value=1.0))
        assert constant.kind == "constant"
        bound = bound_of(with_nuisance(result, treatment_fits=(fits[0], constant)), "ey[1.0]")
        assert bound.limit_refusal.key == "missing_arm"
        assert "the training rows of fold 1 hold no unit of arm 0," in bound.limit_refusal.text

    def test_mismatch(self) -> None:
        """A coefficient moved by 1e-4 passes the root check, which allows 0.05 standard errors.

        The recomputed mechanism then departs from the stored one by about 1e-5, which is what
        the clause compares.
        """
        result = fit_binary(LogisticRegression(max_iter=1000))
        (fit,) = result.repeats[0].nuisance.treatment_fits
        corrupted = replace(fit, coef=fit.coef + 1e-4)
        assert_refusal(
            bound_of(with_nuisance(result, treatment_fits=(corrupted,))),
            key="mismatch",
            text=full_text(
                "the stored treatment mechanism does not equal the LogisticRegression's own "
                "predictions",
                learner=False,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap=None,
        )

    def test_a_large_corruption_fails_the_root_check_first(self) -> None:
        """The order of the fold checks: the score before the recomputed mechanism."""
        result = fit_binary(LogisticRegression(max_iter=1000))
        (fit,) = result.repeats[0].nuisance.treatment_fits
        corrupted = replace(fit, coef=fit.coef + 0.5)
        bound = bound_of(with_nuisance(result, treatment_fits=(corrupted,)))
        assert bound.limit_refusal.key == "score_not_zero"

    def test_a_record_of_the_wrong_length_is_a_mismatch(self) -> None:
        result = fit_three_arm()
        fits = result.repeats[0].nuisance.treatment_fits
        bound = bound_of(with_nuisance(result, treatment_fits=fits[:1]), "ey[1.0]")
        assert bound.limit_refusal.key == "mismatch"
        assert "the fit records 1 treatment models for 2 folds" in bound.limit_refusal.text

    def test_treatment_design(self) -> None:
        """The treatment model read other covariates than the design the stack rebuilds."""
        result = fit_binary(LogisticRegression(max_iter=1000))
        assert result.data.covariate_names == ("W1", "W2", "W3", "W4")
        edited = with_nuisance(result, treatment_covariates=("W1", "W2"))
        assert_refusal(
            bound_of(edited),
            key="treatment_design",
            text=full_text(
                "the treatment model read the covariates ['W1', 'W2'], and the stack rebuilds "
                "its design from ['W1', 'W2', 'W3', 'W4']",
                learner=False,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap=None,
        )

    def test_screening_that_drops_a_covariate_reaches_treatment_design_first(self) -> None:
        """Check 0 runs before the learner rule, so a screen that drops a column names it."""
        result = fit_binary(
            LogisticRegression(max_iter=1000),
            screen_treatment=True,
            screen_threshold=0.2,
            min_retain=1,
        )
        assert result.repeats[0].nuisance.treatment_covariates != result.data.covariate_names
        assert bound_of(result).limit_refusal.key == "treatment_design"

    def test_unrecorded(self) -> None:
        """An estimated mechanism whose fit holds no record of the learner."""
        result = fit_binary(LogisticRegression(max_iter=1000))
        assert result.repeats[0].nuisance.treatment_mechanism != "known"
        assert_refusal(
            bound_of(with_nuisance(result, treatment_fits=None)),
            key="unrecorded",
            text=full_text(
                "the fit records no fitted treatment learner, so the mechanism score is not "
                "available",
                learner=False,
            ),
            alternative=DOUBLY_ROBUST,
            roadmap=None,
        )

    def test_an_empty_record_is_unrecorded_too(self) -> None:
        result = fit_binary(LogisticRegression(max_iter=1000))
        assert bound_of(with_nuisance(result, treatment_fits=())).limit_refusal.key == "unrecorded"


# --------------------------------------------------------------------- a declared mechanism


class TestAKnownMechanismTruncatedByTheBounds:
    """``known_truncated`` is unreachable through the public API, so the bounds are narrowed here.

    The fit refuses a ``g_bounds`` that moves a declared probability.  The stack still asserts
    it, and a declared value moved by the bounds is an invalid declaration, not a gap.
    """

    @pytest.fixture(scope="class")
    def known_fit(self) -> Any:
        return fit_known()

    @staticmethod
    def stack(result: Any, bounds: tuple[float, float]) -> Any:
        elements = sensitivity_elements(result, "ate", nu2_estimator="plugin")
        return plugin_mechanism_term(
            result,
            result.repeats[0],
            resolve_parameter(result, "ate"),
            elements.riesz_representer,
            bounds,
            None,
        )

    def test_a_known_fit_opens_the_limits_with_no_mechanism_term(self, known_fit: Any) -> None:
        bound = bound_of(known_fit)
        assert bound.limit_refusal is None and bound.plugin_mechanism == "known"
        assert np.isfinite(bound.ci_lower)
        assert "the declared known mechanism" in bound.summary()
        term = self.stack(known_fit, (0.0, 1.0))
        assert not isinstance(term, MechanismRefusal)
        assert term.values.shape == (known_fit.n,) and not np.any(term.values)
        # An unweighted ATE of a known mechanism is linear in the sample: no second-order bias.
        assert term.correction == pytest.approx(0.0, abs=1e-12)
        assert bound.nu2_correction == pytest.approx(0.0, abs=1e-12)

    def test_known_truncated(self, known_fit: Any) -> None:
        probabilities = known_fit.repeats[0].nuisance.propensity.values
        assert probabilities.min() < 0.4 and probabilities.max() > 0.6
        refusal = self.stack(known_fit, (0.4, 0.6))
        assert isinstance(refusal, MechanismRefusal)
        assert refusal.key == "known_truncated"
        assert refusal.text == full_text(
            "the truncation bounds move a declared known mechanism, so the representer is not "
            "alpha(g_0)",
            learner=False,
        )
        assert refusal.alternative == (
            "Pass g_bounds=(lower, upper) that contains every known probability."
        )
        assert refusal.reason is RefusalReason.INVALID_DECLARATION
        assert refusal.roadmap is None

    def test_the_accessors_raise_the_stored_record(
        self, known_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        refusal = self.stack(known_fit, (0.4, 0.6))
        monkeypatch.setattr(omitted_variable, "plugin_mechanism_term", lambda *args: refusal)
        assert_refusal(
            bound_of(known_fit),
            key="known_truncated",
            text=refusal.text,
            alternative=refusal.alternative,
            roadmap=None,
            reason=RefusalReason.INVALID_DECLARATION,
        )


# ------------------------------------------------------------------------- the capture


def assert_same_record(first: MechanismFit, second: MechanismFit) -> None:
    """Two records hold the same values, arrays compared exactly."""
    for field in dataclasses.fields(MechanismFit):
        a, b = getattr(first, field.name), getattr(second, field.name)
        if field.name == "steps":
            assert (a is None) == (b is None)
            for step_a, step_b in zip(a or (), b or (), strict=True):
                assert isinstance(step_a, ScalerStep) and step_a.kind == step_b.kind
                for part in ("mean", "scale"):
                    assert np.array_equal(getattr(step_a, part), getattr(step_b, part))
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert np.array_equal(a, b), field.name
        else:
            assert a == b, field.name


def companion_inputs(n: int = 240, folds: int = 3) -> dict[str, Any]:
    """The arguments :func:`cross_fit_companion` takes to cross-fit a treatment mechanism."""
    frame = binary_frame(n=n)
    covariates = frame[["W1", "W2", "W3", "W4"]].to_numpy(dtype=float)
    return {
        "design": covariates,
        "target": frame["A"].to_numpy(dtype=float),
        "weights": np.ones(n),
        "folds": random_partition(n, folds, seed=3),
    }


def run_companion(learner: Any, *, mechanisms: list[Any] | None, n_jobs: int = 1) -> Any:
    """The out-of-fold mechanism of ``learner`` on :func:`companion_inputs`."""
    arguments = companion_inputs()
    predictions, _, _ = cross_fit_companion(
        learner,
        arguments["design"],
        arguments["target"],
        arguments["weights"],
        arguments["folds"],
        task="classification",
        predict_designs={"g": arguments["design"]},
        companion_designs={},
        clip=(0.0, 1.0),
        classes=(0.0, 1.0),
        n_jobs=n_jobs,
        mechanisms=mechanisms,
    )
    return predictions["g"]


class TestTheCaptureRecordsFoldByFold:
    def test_the_default_predictions_are_bit_identical(self) -> None:
        without = run_companion(LogisticRegression(max_iter=1000), mechanisms=None)
        records: list[MechanismFit] = []
        captured = run_companion(LogisticRegression(max_iter=1000), mechanisms=records)
        assert len(records) == 3
        assert np.array_equal(without, captured)

    def test_the_records_come_in_fold_order(self) -> None:
        """Record ``k`` is the model trained on fold ``k``'s complement, refitted here."""
        records: list[MechanismFit] = []
        run_companion(LogisticRegression(max_iter=1000), mechanisms=records)
        arguments = companion_inputs()
        assert len(records) == arguments["folds"].n_folds == 3
        coefficients = []
        for record, (train, _) in zip(records, arguments["folds"], strict=True):
            refit = LogisticRegression(max_iter=1000).fit(
                arguments["design"][train], arguments["target"][train]
            )
            np.testing.assert_allclose(record.coef, refit.coef_, atol=1e-6)
            np.testing.assert_allclose(record.intercept, refit.intercept_, atol=1e-6)
            coefficients.append(record.coef)
        # The folds differ, so a reversed or repeated record would not pass the loop above.
        assert not np.allclose(coefficients[0], coefficients[1], atol=1e-6)
        assert not np.allclose(coefficients[1], coefficients[2], atol=1e-6)

    def test_a_process_pool_records_the_same_folds_in_the_same_order(self) -> None:
        serial: list[MechanismFit] = []
        pooled: list[MechanismFit] = []
        predictions = run_companion(LogisticRegression(max_iter=1000), mechanisms=serial)
        in_pool = run_companion(LogisticRegression(max_iter=1000), mechanisms=pooled, n_jobs=2)
        assert len(pooled) == 3
        assert np.array_equal(predictions, in_pool)
        for first, second in zip(serial, pooled, strict=True):
            assert_same_record(first, second)

    def test_a_scaler_step_records_the_training_mean_and_scale(self) -> None:
        learner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        records: list[MechanismFit] = []
        run_companion(learner, mechanisms=records)
        arguments = companion_inputs()
        for record, (train, _) in zip(records, arguments["folds"], strict=True):
            assert record.step_names == ("StandardScaler",)
            (step,) = record.steps
            np.testing.assert_allclose(step.mean, arguments["design"][train].mean(axis=0))
            np.testing.assert_allclose(step.scale, arguments["design"][train].std(axis=0))

    def test_a_learner_the_rule_does_not_read_is_recorded_by_name(self) -> None:
        records: list[MechanismFit] = []
        run_companion(RandomForestClassifier(n_estimators=5, random_state=0), mechanisms=records)
        assert {(r.kind, r.learner_name, r.coef) for r in records} == {
            ("other", "RandomForestClassifier", None)
        }

    def test_an_in_sample_fit_records_one_model(self) -> None:
        result = fit_binary(LogisticRegression(max_iter=1000))
        (fit,) = result.repeats[0].nuisance.treatment_fits
        assert fit.kind == "logistic" and fit.learner_name == "LogisticRegression"
        assert fit.classes.tolist() == [0.0, 1.0]
        assert fit.coef.shape == (1, 4)

    def test_a_cross_fitted_fit_records_one_model_per_fold(self) -> None:
        result = fit_three_arm()
        fits = result.repeats[0].nuisance.treatment_fits
        assert len(fits) == result.repeats[0].folds.n_folds == 2
        assert all(fit.coef.shape == (3, 2) for fit in fits)

    def test_a_fit_that_estimates_no_mechanism_records_none(self) -> None:
        assert fit_known().repeats[0].nuisance.treatment_fits is None


class TestTheCaptureSurvivesTheRoutesAFitTakes:
    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return fit_binary(LogisticRegression(max_iter=1000))

    def test_a_pickle_round_trip_keeps_the_records_and_the_limits(self, result: Any) -> None:
        restored = pickle.loads(pickle.dumps(result))
        original = result.repeats[0].nuisance.treatment_fits
        again = restored.repeats[0].nuisance.treatment_fits
        assert len(again) == len(original)
        for first, second in zip(original, again, strict=True):
            assert_same_record(first, second)
        assert bound_of(restored).ci_lower == bound_of(result).ci_lower

    def test_save_and_load_keep_the_records_and_the_limits(
        self, result: Any, tmp_path: Path
    ) -> None:
        restored = cleverly.load(result.save(tmp_path / "plugin.joblib"))
        for first, second in zip(
            result.repeats[0].nuisance.treatment_fits,
            restored.repeats[0].nuisance.treatment_fits,
            strict=True,
        ):
            assert_same_record(first, second)
        loaded = bound_of(restored)
        assert loaded.limit_refusal is None
        assert loaded.ci_lower == bound_of(result).ci_lower

    def test_a_refusing_fit_keeps_its_refusal_through_save(self, tmp_path: Path) -> None:
        refusing = fit_binary(LogisticRegression(solver="liblinear"))
        restored = cleverly.load(refusing.save(tmp_path / "refusing.joblib"))
        assert bound_of(restored).limit_refusal == bound_of(refusing).limit_refusal

    def test_a_stored_refusal_survives_a_pickle_of_the_bound(self) -> None:
        bound = bound_of(fit_binary(LogisticRegression(solver="liblinear")))
        restored = pickle.loads(pickle.dumps(bound))
        assert restored == bound
        assert restored.limit_refusal.key == "liblinear"
        with pytest.raises(CapabilityError):
            _ = restored.ci_lower

    def test_at_level_keeps_the_records(self) -> None:
        results = fit_cde(
            cde_frame(),
            ("ate",),
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        )
        nuisance = results[0.0].repeats[0].nuisance
        assert nuisance.treatment_fits is not None
        moved = nuisance.at_level(1.0)
        assert moved.outcome is not nuisance.outcome
        assert moved.treatment_fits is nuisance.treatment_fits

    def test_retilted_keeps_the_records(self) -> None:
        frame, _ = make_nonlinear_ate(n=300, seed=0)
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                incremental=[Incremental(1.0), Incremental(2.0)],
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        nuisance = result.repeats[0].nuisance
        assert nuisance.treatment_fits is not None
        again = nuisance.retilted(nuisance.propensity.values[:, 1])
        assert again.treatment_fits is nuisance.treatment_fits


# ------------------------------------------------------------------------------ benchmark


class TestTheBenchmarkReadsItsOwnShortFit:
    """``benchmark()`` refits without the named covariates, so the short fit has its own record.

    It refuses a known-mechanism fit, so the test runs on an estimated logistic mechanism.
    """

    def test_a_plugin_benchmark_runs_on_a_qualifying_fit(self) -> None:
        result = fit_binary(LogisticRegression(max_iter=1000))
        calibrated = benchmark(result, ["W4"], nu2_estimator="plugin", random_state=0)
        long = sensitivity_elements(result, "ate", nu2_estimator="plugin")
        assert calibrated.nu2_long == long.nu2
        assert calibrated.sigma2_long == long.sigma2
        assert np.isfinite(calibrated.cf_y) and np.isfinite(calibrated.cf_d)
        assert calibrated.nu2_short > 0.0 and calibrated.nu2_short != calibrated.nu2_long

    def test_the_short_refit_records_the_covariates_it_read(self) -> None:
        """The short fit's own design opens the limits, which a stale design would refuse."""
        result = fit_binary(LogisticRegression(max_iter=1000))
        short = result.estimator.refit(result.data.without_covariates(["W4"]), random_state=0)
        nuisance = short.repeats[0].nuisance
        assert nuisance.treatment_covariates == ("W1", "W2", "W3")
        assert nuisance.treatment_fits[0].coef.shape == (1, 3)
        bound = bound_of(short)
        assert bound.limit_refusal is None and np.isfinite(bound.ci_lower)

    def test_the_benchmark_refuses_a_known_mechanism(self) -> None:
        with pytest.raises(CapabilityError, match="declares the mechanism known"):
            benchmark(fit_known(), ["W2"], nu2_estimator="plugin")


# ------------------------------------------------------------------------------ the order


class TestTheInferenceStatusAnswersFirst:
    """A non-inferential status refuses the inferential names before the mechanism does.

    The plug-in names are diagnostics, so they skip the status and meet the mechanism.
    """

    INFERENTIAL = ("ci_lower", "ci_upper", "robustness_value_ci")
    DIAGNOSTIC = (
        "plugin_interval_lower",
        "plugin_interval_upper",
        "robustness_value_plugin_interval",
    )

    @staticmethod
    def withheld(bound: Any) -> Any:
        copied = copy.copy(bound)
        withheld = next(s for s in STATUS_RECORDS if not supplies_inference(s))
        object.__setattr__(copied, "inference", withheld)
        return copied

    @pytest.mark.parametrize("accessor", INFERENTIAL)
    def test_a_refusing_fit_meets_the_status_first(self, accessor: str) -> None:
        bound = self.withheld(bound_of(fit_binary(LogisticRegression(solver="liblinear"))))
        with pytest.raises(CapabilityError) as refused:
            getattr(bound, accessor)
        message = str(refused.value)
        assert "nu2_estimator='plugin'" not in message and "F26" not in message
        assert "This fit does not qualify" not in message

    @pytest.mark.parametrize("accessor", DIAGNOSTIC)
    def test_a_refusing_fit_meets_the_mechanism_at_a_diagnostic(self, accessor: str) -> None:
        bound = self.withheld(bound_of(fit_binary(LogisticRegression(solver="liblinear"))))
        with pytest.raises(CapabilityError) as refused:
            getattr(bound, accessor)
        assert bound.limit_refusal.text in str(refused.value)

    @pytest.mark.parametrize("accessor", INFERENTIAL)
    def test_a_qualifying_fit_meets_the_status_too(self, accessor: str) -> None:
        bound = self.withheld(bound_of(fit_binary(LogisticRegression(max_iter=1000))))
        with pytest.raises(CapabilityError) as refused:
            getattr(bound, accessor)
        assert "nu2_estimator='plugin'" not in str(refused.value)

    def test_a_qualifying_fit_reads_its_diagnostic_limits_at_a_status(self) -> None:
        open_bound = bound_of(fit_binary(LogisticRegression(max_iter=1000)))
        bound = self.withheld(open_bound)
        assert bound.plugin_interval_lower == open_bound.ci_lower
        assert bound.plugin_interval_upper == open_bound.ci_upper
        assert bound.to_dict()["inference"] == bound.inference

    def test_the_mapping_carries_the_status_and_the_refusal_together(self) -> None:
        bound = self.withheld(bound_of(fit_binary(LogisticRegression(solver="liblinear"))))
        row = bound.to_dict()
        assert row["inference"] == bound.inference
        assert row["limit_refusal"] == bound.limit_refusal.text


# ------------------------------------------------------------------- the assessment row


class TestTheAssessmentRowQuotesTheClause:
    def test_a_refusing_fit_names_its_clause(self) -> None:
        result = fit_binary(LogisticRegression(solver="liblinear"))
        report = replace(result).sensitivity.run_all(
            arguments={"robustness_value": {"estimand": "ate", "nu2_estimator": "plugin"}}
        )
        stored = bound_of(result).limit_refusal
        assert report["robustness_value"].detail.endswith(
            f"no confidence-limit value: {stored.text}"
        )

    def test_a_qualifying_fit_reports_its_limit_value(self) -> None:
        result = fit_binary(LogisticRegression(max_iter=1000))
        report = replace(result).sensitivity.run_all(
            arguments={"robustness_value": {"estimand": "ate", "nu2_estimator": "plugin"}}
        )
        assert "no confidence-limit value" not in report["robustness_value"].detail
