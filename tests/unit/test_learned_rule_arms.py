"""The learned rule over more than two arms (X11 (d)).

The plug-in rule of a learned-rule fit is :math:`d_{nv}(w) = \\arg\\max_a \\bar Q_{nv}(a, w)`
over every arm, with a tie going to the lowest arm code (polle, Nordland and Holst 2026).  Every
fold's rule at every row is stored as ``(n, V)`` ``int8`` arm codes, and the regime density of
the learned column is one-hot over the ``K`` arms.  The estimator, the fluctuation and the
variance are the shipped ones: only the rule and its record change.

The witnesses are the plan's (``X11-wave2-plan.md`` rev 2.1, section 5):

- W-d1 (reduction): at two arms every fit is the pre-change binary fit bit for bit (estimates,
  stored fold rules and the omitted-variable bound pinned from commit 854dac1), two levels
  coded as strings give the numeric fit, and the argmax of two arms is the blip sign.
- W-d2 (nonzero): the record's per-arm shares are those of the rule on each fold's validation
  rows, and they match the population quadrature of a stub's argmax regions, which a shift of
  the arm index fails.
- W-d3 (ties, mutation): exact ties between two and between three arms assign the lowest arm
  code on both the pooled and the per-arm regression, and the mutation to the last code
  changes the rule (pooled) or breaks the stored diagonal (per arm).
- W-d4 (reduction): with a stub outcome regression whose argmax is a known function of ``W1``,
  the learned-rule fit equals the shipped fold-evaluated and stacked fits of
  ``Rule(..., rule_kind="known")`` on three arms, on the same folds, with a nonzero ``epsilon``.
  The same reduction holds beside ``Static(2)``, so G's contrasts read the new fold rules.

Two-arm support keys and labels are pinned for ten configurations, and every three-arm fit
reads experimental (``treatment=("arms3",)``) until a study registers one.
"""

from __future__ import annotations

import hashlib
import warnings
from functools import cache
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import CausalStudy, LearnedRuleValue, PointTreatment
from cleverly.data.validate import MAX_TREATMENT_LEVELS
from cleverly.estimators import TMLE
from cleverly.estimators import _nuisance as nuisance_module
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    InSampleLearnerWarning,
    RefusalReason,
)
from cleverly.interventions import LearnedRule, Rule, Static
from cleverly.interventions import learned as learned_module
from cleverly.interventions.learned import (
    FrozenLearnedRule,
    check_fold_rules,
    fold_rules,
    freeze_learned_rule,
    frozen_rule_replay,
    realized_arms,
)
from cleverly.sensitivity import (
    ConfounderStrengthGrid,
    omitted_variable_bounds,
    simulated_confounding,
)
from cleverly.validation import refute
from tests.unit import _learned_rule_support as support

NAME = support.NAME
COVARIATES = ("W1", "W2")
FIT: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": COVARIATES}

#: The population shares of the arms under :class:`ArgmaxStub`'s rule, by quadrature of
#: ``W1 ~ U(-1, 1)``: arm 0 on ``[-0.1, 0.5]``, arm 1 above 0.5, arm 2 below -0.1.
STUB_SHARES = (0.3, 0.25, 0.45)


def arms_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Draw ``n`` rows of a three-arm law whose optimal rule uses every arm.

    ``W1 ~ U(-1, 1)`` and ``W2 ~ Bernoulli(0.5)``.  The treatment is a multinomial logit with
    scores ``(0, 0.3 W1, 0.1 - 0.2 W2)``, every arm probability above 0.2, and
    ``logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + 1{A = 1}(W1 - 0.5) + 1{A = 2}(-W1 - 0.1)``.

    Parameters
    ----------
    n : int
        Rows to draw.
    seed : int
        Seed of the draw.

    Returns
    -------
    pandas.DataFrame
        Columns ``W1``, ``W2``, ``A`` (integer codes 0, 1, 2) and ``Y``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    p = softmax(np.column_stack([np.zeros(n), 0.3 * w1, 0.1 - 0.2 * w2]), axis=1)
    a = (rng.uniform(size=n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    logit = 0.2 + 0.5 * w1 - 0.3 * w2 + (a == 1) * (w1 - 0.5) + (a == 2) * (-w1 - 0.1)
    y = rng.binomial(1, expit(logit)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


def four_arms_frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Draw ``n`` rows of a four-arm law whose per-arm plug-in rules use every arm.

    The law is :func:`arms_frame`'s with a fourth arm: treatment scores
    ``(0, 0.3 W1, 0.1 - 0.2 W2, -0.2 W1)`` and an added blip ``1{A = 3}(0.8 W2 - 0.5)``.

    Parameters
    ----------
    n : int
        Rows to draw.
    seed : int
        Seed of the draw.

    Returns
    -------
    pandas.DataFrame
        Columns ``W1``, ``W2``, ``A`` (integer codes 0 to 3) and ``Y``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    p = softmax(np.column_stack([np.zeros(n), 0.3 * w1, 0.1 - 0.2 * w2, -0.2 * w1]), axis=1)
    a = (rng.uniform(size=n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    logit = (
        0.2
        + 0.5 * w1
        - 0.3 * w2
        + (a == 1) * (w1 - 0.5)
        + (a == 2) * (-w1 - 0.1)
        + (a == 3) * (0.8 * w2 - 0.5)
    )
    y = rng.binomial(1, expit(logit)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


def outcome_learner() -> Pipeline:
    """The outcome learner on ``(1{A=1}, 1{A=2}, W1, W2)``, with every pairwise interaction."""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("fit", LogisticRegression(max_iter=5000)),
        ]
    )


def treatment_learner() -> LogisticRegression:
    """A correct multinomial logistic model of the treatment on ``(W1, W2)``."""
    return LogisticRegression(max_iter=1000)


def known_argmax(w1: np.ndarray) -> np.ndarray:
    """The argmax rule of :class:`ArgmaxStub` as arm codes: 1 above 0.5, 2 below -0.1, else 0."""
    return np.where(w1 > 0.5, 1, np.where(w1 < -0.1, 2, 0)).astype(np.int8)


class ArgmaxStub(ClassifierMixin, BaseEstimator):
    """A fixed outcome regression whose argmax over three arms is :func:`known_argmax`.

    ``logit Qbar(a, w) = 0.4 w1 + 1{a = 1}(w1 - 0.5) + 1{a = 2}(-w1 - 0.1)`` on the design
    ``(1{A=1}, 1{A=2}, W1, W2)``.  It omits the intercept and ``W2`` of the law, so the
    fluctuation moves, and it learns nothing, so the learned-rule fit and the fit of the
    known rule read the same regression.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ArgmaxStub:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        w1 = design[:, 2]
        p = expit(0.4 * w1 + design[:, 0] * (w1 - 0.5) + design[:, 1] * (-w1 - 0.1))
        return np.column_stack([1.0 - p, p])


class TieStub(ClassifierMixin, BaseEstimator):
    """A fixed outcome regression with exact ties between two and between three arms.

    ``logit Qbar(a, w) = 0.4 w1 + 1{a = 1} b1(w) + 1{a = 2} b2(w)``.  At ``W2 = 1`` both blips
    are 0.5, so arms 1 and 2 tie above arm 0.  At ``W2 = 0`` and ``W1 > 0`` both are 0, so all
    three arms tie.  At ``W2 = 0`` and ``W1 <= 0`` arm 2 is best.  The sums are written so
    that each tied logit is the same float.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TieStub:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        w1, w2 = design[:, 2], design[:, 3]
        b1 = np.where(w2 == 1.0, 0.5, np.where(w1 > 0.0, 0.0, -0.2))
        b2 = np.where(w2 == 1.0, 0.5, np.where(w1 > 0.0, 0.0, 0.3))
        p = expit(0.4 * w1 + design[:, 0] * b1 + design[:, 1] * b2)
        return np.column_stack([1.0 - p, p])


class ConstantStub(ClassifierMixin, BaseEstimator):
    """A per-arm outcome model that predicts 0.5 everywhere, so every arm ties at every row."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantStub:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        return np.full((np.asarray(X).shape[0], 2), 0.5)


def tmle(**overrides: Any) -> TMLE:
    """A fold-evaluated three-arm learned-rule estimator with explicit learners and 5 folds."""
    settings: dict[str, Any] = {
        "learned_rule": LearnedRule(),
        "outcome_learner": outcome_learner(),
        "treatment_learner": treatment_learner(),
        "cross_fit": True,
        "cv_evaluation": True,
        "n_folds": 5,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return TMLE(**settings)


def fit(frame: pd.DataFrame | None = None, *, fit_options: Any = None, **overrides: Any) -> Any:
    """Fit :func:`tmle` on ``frame``, which defaults to :func:`arms_frame`."""
    frame = arms_frame() if frame is None else frame
    return tmle(**overrides).fit(frame, **FIT, **(fit_options or {})).single()


@cache
def _real() -> Any:
    return fit()


@cache
def _stub(cv_evaluation: bool = True) -> Any:
    return fit(outcome_learner=ArgmaxStub(), cv_evaluation=cv_evaluation)


def _stored(result: Any) -> np.ndarray:
    rules = fold_rules(result.repeats[0].nuisance)
    assert rules is not None
    return rules


def _designs(result: Any, rows: np.ndarray | None = None) -> list[np.ndarray]:
    """The pooled outcome design at each arm, for every row (or ``rows``)."""
    data = result.data
    covariates = np.asarray(data.covariates, dtype=float)
    if rows is not None:
        covariates = covariates[rows]
    return [
        np.column_stack(
            [np.full(len(covariates), arm == 1.0), np.full(len(covariates), arm == 2.0), covariates]
        ).astype(float)
        for arm in data.arm_codes
    ]


# ------------------------------------------------------------------------------- W-d1

#: The pre-change binary fits at commit 854dac1, on ``law_frame(203, 1)`` with the
#: misspecified outcome learner (a correct logistic model for the per-arm regression): every
#: estimate's ``psi`` and ``variance`` as float hex, and the SHA-256 of the stored fold rules
#: as ``int8`` bytes in C order (the pre-change booleans cast to ``int8``).  ``bound`` is the
#: robustness value at ``cf_y=0.05, cf_d=0.04``.
PRE_CHANGE: dict[str, dict[str, Any]] = {
    "pooled": {
        "estimates": {
            NAME: ("0x1.d5c14b93350a0p-2", "0x1.50e224378e961p-9"),
        },
        "rules": "fe4f43d014c01f8608d6ae95bbc031084a0e8ba68e914d2d278008a358c85b92",
        "bound": "0x1.0869eb5762119p-1",
    },
    "by_arm": {
        "estimates": {
            NAME: ("0x1.2a1d1240854b6p-1", "0x1.4d1dec93f4653p-9"),
        },
        "rules": "b608d199c819d935f994679d3faac29eede0f46837c0f52c8331cecbf5786edc",
    },
    "stacked": {
        "estimates": {
            NAME: ("0x1.d583aae4fe308p-2", "0x1.50b6ad4b2cdbep-9"),
        },
        "rules": "fe4f43d014c01f8608d6ae95bbc031084a0e8ba68e914d2d278008a358c85b92",
    },
    "in_sample": {
        "estimates": {
            NAME: ("0x1.0b4b5dc997b4fp-1", "0x1.6a1482f67e375p-9"),
        },
        "rules": "123c71b340bd2d9c4ea8b95ba2f4fd98f63384f27ace01b8039d44d1b626a55c",
    },
    "blip": {
        "estimates": {
            NAME: ("0x1.2c7d71b7390b0p-1", "0x1.481dd5771aa17p-9"),
        },
        "rules": "899c4559945f2359f67d8381d787a67c968d658f793c0c160efe2f0a30bd418f",
    },
    "contrast": {
        "estimates": {
            NAME: ("0x1.d4fe4fb62a3e0p-2", "0x1.529fccdb0fb40p-9"),
            "ey_known_regime[always 1]": ("0x1.060f1e4202d12p-1", "0x1.78d8258e46e3dp-9"),
            "ate_learned_rule[learned rule vs always 1]": (
                "-0x1.b8ff666edb21ep-5",
                "0x1.166222642ff33p-10",
            ),
        },
        "rules": "fe4f43d014c01f8608d6ae95bbc031084a0e8ba68e914d2d278008a358c85b92",
    },
}

_PRE_CHANGE_FITS: dict[str, dict[str, Any]] = {
    "pooled": {},
    "by_arm": {"outcome_by_arm": True, "outcome_learner": LogisticRegression(max_iter=1000)},
    "stacked": {"cv_evaluation": False},
    "in_sample": {"cross_fit": False, "cv_evaluation": False},
    "blip": {"learned_rule": LearnedRule(loss="blip", learner=LinearRegression())},
    "contrast": {"interventions": [Static(1)]},
}


@cache
def _binary(configuration: str) -> Any:
    overrides = {
        "outcome_learner": support.misspecified_learner(),
        **_PRE_CHANGE_FITS[configuration],
    }
    if configuration != "in_sample":
        return support.fit(support.law_frame(203, 1), **overrides)
    # The helper's outcome learner is a Pipeline, which keys adaptive in sample, and warns.
    with pytest.warns(InSampleLearnerWarning):
        return support.fit(support.law_frame(203, 1), **overrides)


class TestWd1Reduction:
    @pytest.mark.parametrize("configuration", list(PRE_CHANGE))
    def test_a_two_arm_fit_is_the_pre_change_fit_bit_for_bit(self, configuration: str) -> None:
        result = _binary(configuration)
        pinned = PRE_CHANGE[configuration]["estimates"]
        assert list(result.estimates) == list(pinned)
        for name, (psi, variance) in pinned.items():
            assert result.estimates[name].psi == float.fromhex(psi), name
            assert result.estimates[name].variance == float.fromhex(variance), name
        assert _sha(_stored(result)) == PRE_CHANGE[configuration]["rules"]

    def test_the_two_arm_bound_is_the_pre_change_bound(self) -> None:
        bound = omitted_variable_bounds(_binary("pooled"), NAME, cf_y=0.05, cf_d=0.04)
        assert bound.robustness_value == float.fromhex(PRE_CHANGE["pooled"]["bound"])

    @pytest.mark.parametrize("configuration", ["pooled", "by_arm", "blip"])
    def test_the_two_arm_fold_rules_are_the_blip_sign_as_codes(self, configuration: str) -> None:
        """Every stored entry is 0 or 1, and 1 exactly where the fold's rule treats."""
        result = _binary(configuration)
        rules = _stored(result)
        assert rules.dtype == np.int8 and set(np.unique(rules)) == {0, 1}
        own = rules[np.arange(result.data.n), np.asarray(result.nuisance.folds.assignment)]
        np.testing.assert_array_equal(own, result.nuisance.regimes.values[:, 1, 0])

    def test_two_levels_coded_as_strings_give_the_numeric_fit(self) -> None:
        frame = support.law_frame(203, 1)
        labelled = frame.assign(A=np.where(frame["A"] == 1.0, "treated", "control"))
        numeric = _binary("pooled")
        named = support.fit(labelled, outcome_learner=support.misspecified_learner())
        assert tuple(named.data.treatment_levels) == ("control", "treated")
        a, b = numeric.estimates[NAME], named.estimates[NAME]
        assert (a.psi, a.variance) == (b.psi, b.variance)
        np.testing.assert_array_equal(_stored(numeric), _stored(named))
        assert named.extra["learned_rule"].arms == ("control", "treated")

    def test_the_argmax_of_two_arms_is_the_blip_sign(self) -> None:
        rng = np.random.default_rng(0)
        control = rng.uniform(size=500)
        treated = np.where(np.arange(500) % 5 == 0, control, rng.uniform(size=500))
        arm = nuisance_module.plug_in_arm([control, treated])
        assert arm.dtype == np.int8
        np.testing.assert_array_equal(arm, (treated - control > 0.0).astype(np.int8))
        # The mutation: the last maximum in place of the first assigns the higher arm at a tie.
        last = 1 - np.argmax(np.vstack([treated, control]), axis=0)
        assert np.sum(last != arm) == 100


# ---------------------------------------------------------------------- W-d4, the rule


def _known_rule() -> Rule:
    return Rule(
        lambda w: known_argmax(w["W1"].to_numpy()).astype(float), name="w1", rule_kind="known"
    )


@cache
def _reduction(cv_evaluation: bool, contrast: bool = False) -> tuple[Any, Any]:
    """The stub learned-rule fit and the shipped regime fit of its known argmax, on one draw."""
    known = [Static(2)] if contrast else []
    common: dict[str, Any] = {"outcome_learner": ArgmaxStub(), "cv_evaluation": cv_evaluation}
    learned = fit(interventions=known, **common)
    regime = fit(
        learned_rule=None,
        interventions=[_known_rule(), *known],
        reference="always 2" if contrast else None,
        **common,
    )
    return learned, regime


class TestTheArgmaxRule:
    def test_every_fold_rule_is_the_stub_s_argmax_at_every_row(self) -> None:
        result = _stub()
        rules = _stored(result)
        expected = known_argmax(np.asarray(result.data.covariates)[:, 0])
        assert rules.shape == (600, 5) and rules.dtype == np.int8
        for fold in range(5):
            np.testing.assert_array_equal(rules[:, fold], expected)
        np.testing.assert_array_equal(realized_arms(result.nuisance.regimes), expected)
        # The density is one-hot over the three arms.
        values = result.nuisance.regimes.values
        assert values.shape == (600, 3, 1)
        np.testing.assert_array_equal(values[np.arange(600), expected, 0], 1.0)
        np.testing.assert_array_equal(values.sum(axis=1), 1.0)

    def test_each_column_of_a_learned_regression_is_its_fold_s_argmax(self) -> None:
        """Refit fold ``v``'s pooled regression by hand and take its argmax at every row."""
        result = _real()
        frame = arms_frame()
        design = np.column_stack(
            [frame["A"] == 1, frame["A"] == 2, frame[list(COVARIATES)]]
        ).astype(float)
        rules = _stored(result)
        at = _designs(result)
        for fold, (train, _) in enumerate(result.nuisance.folds):
            model = outcome_learner().fit(design[np.asarray(train)], frame["Y"].to_numpy()[train])
            predictions = np.vstack([model.predict_proba(x)[:, 1] for x in at])
            np.testing.assert_array_equal(rules[:, fold], np.argmax(predictions, axis=0))
        # The nonzero witness: every arm is some fold's rule somewhere, and the folds differ.
        assert set(np.unique(rules)) == {0, 1, 2}
        assert np.any(rules != rules[:, :1])

    @pytest.mark.parametrize("n_arms", [3, 4], ids=["arms3", "arms4"])
    def test_each_column_of_a_per_arm_regression_is_its_fold_s_argmax(self, n_arms: int) -> None:
        """Four arms run the per-arm step that carries an index into an arm before the last."""
        learner = LogisticRegression(max_iter=1000)
        frame = arms_frame() if n_arms == 3 else four_arms_frame()
        result = fit(frame, outcome_by_arm=True, outcome_learner=learner)
        covariates = frame[list(COVARIATES)].to_numpy()
        treatment, outcome = frame["A"].to_numpy(), frame["Y"].to_numpy()
        rules = _stored(result)
        for fold, (train, _) in enumerate(result.nuisance.folds):
            rows = np.asarray(train)
            predictions = []
            for arm in range(n_arms):
                own = rows[treatment[rows] == arm]
                model = clone(learner).fit(covariates[own], outcome[own])
                predictions.append(model.predict_proba(covariates)[:, 1])
            np.testing.assert_array_equal(rules[:, fold], np.argmax(predictions, axis=0))
        assert set(np.unique(rules)) == set(range(n_arms))

    @pytest.mark.parametrize("cv_evaluation", [True, False], ids=["fold_evaluated", "stacked"])
    def test_wd4_the_stub_fit_is_the_known_rule_fit(self, cv_evaluation: bool) -> None:
        learned, regime = _reduction(cv_evaluation)
        for (_, a), (_, b) in zip(learned.nuisance.folds, regime.nuisance.folds, strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            learned.nuisance.regimes.values, regime.nuisance.regimes.values
        )
        # The nonzero witness: the comparison runs through a nonzero epsilon.
        assert float(np.max(np.abs(learned.fluctuations["regime"].epsilon))) > 1e-2
        a, b = learned.estimates[NAME], regime.estimates["ey_regime[w1]"]
        assert a.psi == pytest.approx(b.psi, rel=1e-12, abs=1e-15)
        assert a.variance == pytest.approx(b.variance, rel=1e-12)
        np.testing.assert_allclose(a.influence_curve, b.influence_curve, rtol=0, atol=1e-13)

    @pytest.mark.parametrize("cv_evaluation", [True, False], ids=["fold_evaluated", "stacked"])
    def test_wd4_the_contrast_with_a_static_arm_is_the_regime_contrast(
        self, cv_evaluation: bool
    ) -> None:
        """G's path on three arms: the learned column and ``Static(2)`` on the same folds."""
        learned, regime = _reduction(cv_evaluation, contrast=True)
        pairs = {
            NAME: "ey_regime[w1]",
            "ey_known_regime[always 2]": "ey_regime[always 2]",
            "ate_learned_rule[learned rule vs always 2]": "ate_regime[w1 vs always 2]",
        }
        assert list(learned.estimates) == list(pairs)
        for name, other in pairs.items():
            a, b = learned.estimates[name], regime.estimates[other]
            assert a.psi == pytest.approx(b.psi, rel=1e-12, abs=1e-15), name
            assert a.variance == pytest.approx(b.variance, rel=1e-12), name
            np.testing.assert_allclose(a.influence_curve, b.influence_curve, rtol=0, atol=1e-13)
        contrast = learned.estimates["ate_learned_rule[learned rule vs always 2]"]
        assert abs(contrast.psi) > 0.05

    def test_the_contrast_curve_is_the_difference_on_a_learned_regression(self) -> None:
        result = fit(interventions=[Static(0), Static(2)])
        for arm in ("0", "2"):
            contrast = result.estimates[f"ate_learned_rule[learned rule vs always {arm}]"]
            known = result.estimates[f"ey_known_regime[always {arm}]"]
            learned = result.estimates[NAME]
            assert contrast.psi == pytest.approx(learned.psi - known.psi, abs=1e-12)
            difference = np.asarray(learned.influence_curve) - np.asarray(known.influence_curve)
            np.testing.assert_allclose(contrast.influence_curve, difference, rtol=0, atol=1e-14)
            assert np.max(np.abs(difference)) > 1e-2


# ------------------------------------------------------------------------------- W-d3


def _last_code(predictions: Any) -> np.ndarray:
    """The mutation: the last of the tied maxima in place of the first."""
    stacked = np.vstack(predictions)
    return np.asarray(stacked.shape[0] - 1 - np.argmax(stacked[::-1], axis=0), dtype=np.int8)


class TestWd3Ties:
    @staticmethod
    def _regions(result: Any) -> tuple[np.ndarray, np.ndarray]:
        covariates = np.asarray(result.data.covariates)
        two = covariates[:, 1] == 1.0
        three = (covariates[:, 1] == 0.0) & (covariates[:, 0] > 0.0)
        return two, three

    def test_a_tie_assigns_the_lowest_arm_code_at_every_fold(self) -> None:
        result = fit(outcome_learner=TieStub())
        rules = _stored(result)
        two, three = self._regions(result)
        assert two.sum() > 200 and three.sum() > 100
        assert np.all(rules[two] == 1) and np.all(rules[three] == 0)
        assert np.all(rules[~two & ~three] == 2)
        # The predictions do tie exactly: the last-code rule differs on every tied row.
        stub = TieStub().fit(None, None)
        predictions = [stub.predict_proba(x)[:, 1] for x in _designs(result)]
        last = _last_code(predictions)
        np.testing.assert_array_equal(last[two | three], 2)
        np.testing.assert_array_equal(np.argmax(predictions, axis=0), rules[:, 0])
        shares = result.extra["learned_rule"].splits[0].arm_shares
        assert all(min(fold) > 0.1 for fold in shares)

    def test_control_the_last_code_mutation_moves_the_tied_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nuisance_module, "plug_in_arm", _last_code)
        result = fit(outcome_learner=TieStub())
        rules = _stored(result)
        two, three = self._regions(result)
        assert np.all(rules[two | three] == 2)

    def test_a_three_way_tie_of_the_per_arm_regression_assigns_arm_zero(self) -> None:
        result = fit(outcome_by_arm=True, outcome_learner=ConstantStub())
        np.testing.assert_array_equal(_stored(result), 0)
        assert result.extra["learned_rule"].splits[0].gap_quantiles[0] == (0.0,) * 7

    def test_control_a_per_arm_step_that_keeps_the_last_tie_is_a_defect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``>=`` in the running comparison stores the last arm; the realized rule keeps 0."""
        original = nuisance_module._by_arm_rule_step

        def last(fold: int, values: Any, *, best: Any, **kwargs: Any) -> Any:
            return original(fold, values, best=best - 1e-12, **kwargs)

        monkeypatch.setattr(nuisance_module, "_by_arm_rule_step", last)
        with pytest.raises(RuntimeError, match="defect in cleverly"):
            fit(outcome_by_arm=True, outcome_learner=ConstantStub())


# ------------------------------------------------------------------------------- W-d2


class TestWd2TheRecord:
    def test_the_shares_are_the_rule_s_on_each_fold(self) -> None:
        result = _stub()
        record = result.extra["learned_rule"]
        expected = known_argmax(np.asarray(result.data.covariates)[:, 0])
        assert record.arms == tuple(str(level) for level in result.data.treatment_levels)
        assert len(record.arms) == 3
        for (_, test), shares in zip(
            result.nuisance.folds, record.splits[0].arm_shares, strict=True
        ):
            rows = np.asarray(test)
            assert shares == tuple(np.bincount(expected[rows], minlength=3) / rows.size)
            assert sum(shares) == pytest.approx(1.0, abs=1e-15)
            assert min(shares) > 0.1

    @staticmethod
    def _within(shares: np.ndarray, n: int) -> bool:
        se = np.sqrt(np.asarray(STUB_SHARES) * (1.0 - np.asarray(STUB_SHARES)) / n)
        return bool(np.all(np.abs(shares - np.asarray(STUB_SHARES)) < 4.0 * se))

    def test_the_shares_match_the_population_quadrature_and_an_index_shift_fails(self) -> None:
        result = fit(arms_frame(3000, 4), outcome_learner=ArgmaxStub())
        split = result.extra["learned_rule"].splits[0]
        pooled = np.average(np.asarray(split.arm_shares), axis=0, weights=split.fold_sizes)
        assert self._within(pooled, 3000)
        # The mutation: the arm index shifted by one names the shares of the wrong arms.
        assert not self._within(np.roll(pooled, 1), 3000)

    def test_the_gap_quantiles_are_the_top_two_gap_of_each_fold(self) -> None:
        result = _stub()
        stub = ArgmaxStub().fit(None, None)
        predictions = np.sort(
            np.vstack([stub.predict_proba(x)[:, 1] for x in _designs(result)]), axis=0
        )
        gap = predictions[-1] - predictions[-2]
        split = result.extra["learned_rule"].splits[0]
        for (_, test), recorded in zip(result.nuisance.folds, split.gap_quantiles, strict=True):
            np.testing.assert_allclose(
                recorded, np.quantile(gap[np.asarray(test)], split_levels()), rtol=0, atol=1e-12
            )
        # The nonzero witness: the gap is not the blip of the two highest codes.
        blip = predictions[-1] - predictions[0]
        assert np.max(np.abs(blip - gap)) > 1e-2

    def test_the_summary_line_gives_every_arm_s_share(self) -> None:
        result = _real()
        record = result.extra["learned_rule"]
        line = record.describe()
        assert "the arm of 3 with the highest predicted outcome" in line
        for arm in record.arms:
            assert f"A = {arm}: " in line
        assert line in result.summary(detail="full").section("Method details").lines

    def test_the_rule_class_names_the_argmax_and_its_tie(self) -> None:
        text = _real().extra["learned_rule"].rule
        assert "argmax_a Qbar_v(a, W)" in text
        assert "assigns the lowest of their arm codes" in text


def split_levels() -> tuple[float, ...]:
    """The record's quantile levels."""
    return learned_module.BLIP_QUANTILE_LEVELS


# ---------------------------------------------------------------------- the stored rules


class TestTheStoredRules:
    def test_a_bool_matrix_is_a_defect(self) -> None:
        result = _real()
        nuisance = result.repeats[0].nuisance
        rules = _stored(result)
        realized = realized_arms(nuisance.regimes)
        with pytest.raises(RuntimeError, match="int8 arm codes"):
            check_fold_rules(rules.astype(bool), realized, nuisance.folds)

    def test_a_code_outside_the_arms_is_a_defect(self) -> None:
        result = _real()
        nuisance = result.repeats[0].nuisance
        rules = np.array(_stored(result))
        rules[rules == 2] = 3
        realized = rules[np.arange(600), np.asarray(nuisance.folds.assignment)]
        with pytest.raises(RuntimeError, match="outside 0 to 2"):
            check_fold_rules(rules, realized, nuisance.folds, n_arms=3)
        # Without the arm count the same matrix passes the diagonal check.
        check_fold_rules(rules, realized, nuisance.folds)

    def test_the_treatment_level_cap_fits_the_int8_fold_rule_store(self) -> None:
        """The fold rules are stored as ``int8`` arm codes, which hold at most 128 arms.

        The data layer's level cap is what keeps the guard in ``fit_nuisances`` unreachable.
        """
        assert np.iinfo(np.int8).max + 1 >= MAX_TREATMENT_LEVELS

    def test_realized_arms_refuses_a_fit_without_regimes(self) -> None:
        with pytest.raises(ValueError, match="regime set of a learned-rule fit"):
            realized_arms(None)

    def test_a_corrupted_diagonal_is_a_defect(self) -> None:
        from dataclasses import replace

        nuisance = _real().repeats[0].nuisance
        corrupted = np.array(nuisance.regimes.fold_rules)
        fold = int(nuisance.folds.assignment[0])
        corrupted[0, fold] = (corrupted[0, fold] + 1) % 3
        broken = replace(nuisance, regimes=replace(nuisance.regimes, fold_rules=corrupted))
        with pytest.raises(RuntimeError, match="defect in cleverly"):
            fold_rules(broken)


# ------------------------------------------------------------------------ the refusals


class TestTheRefusals:
    @pytest.mark.parametrize(
        "rule",
        [
            LearnedRule(loss="blip", learner=LinearRegression()),
            LearnedRule(loss="weighted_classification", learner=LogisticRegression()),
        ],
        ids=["blip", "weighted_classification"],
    )
    def test_a_rule_learner_refuses_and_its_remedies_fit(self, rule: LearnedRule) -> None:
        with pytest.raises(CapabilityError, match=r"X11 \(d2\)") as caught:
            fit(learned_rule=rule)
        refusal = caught.value.refusal
        assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert "LearnedRule() (loss='plug_in')" in str(caught.value)
        assert "TMLE(interventions=[Rule(..., rule_kind='known')])" in str(caught.value)
        # The remedies, as written: the plug-in rule, and a fixed rule on the same treatment.
        assert np.isfinite(_real().estimates[NAME].psi)
        fixed = fit(learned_rule=None, interventions=[_known_rule()])
        assert np.isfinite(fixed.estimates["ey_regime[w1]"].psi)

    def test_the_bound_refuses_by_name_and_its_remedy_runs(self) -> None:
        result = _real()
        assert not result.sensitivity.capability("omitted_confounding").available
        with pytest.raises(CapabilityError, match=r"over 3 arms is not built.*X11 \(n2\)"):
            omitted_variable_bounds(result, NAME, cf_y=0.05, cf_d=0.04)
        fixed = fit(learned_rule=None, interventions=[_known_rule()])
        bound = omitted_variable_bounds(fixed, "ey_regime[w1]", cf_y=0.05, cf_d=0.04)
        assert bound.max_bias > 0.0

    def test_a_frozen_rule_of_other_levels_refuses(self) -> None:
        result = _real()
        frozen = freeze_learned_rule(result)
        assert frozen.levels == tuple(result.data.treatment_levels)
        relabelled = FrozenLearnedRule(frozen.name, frozen.treat, ("a", "b", "c"))
        with pytest.raises(DataError, match="other arms"):
            frozen_rule_replay(result.estimator, relabelled).refit(
                result.data, random_state=0, frozen_folds=[result.nuisance.folds]
            )
        beyond = FrozenLearnedRule(frozen.name, np.full(600, 3, dtype=np.int8))
        with pytest.raises(DataError, match="above 2"):
            frozen_rule_replay(result.estimator, beyond).refit(
                result.data, random_state=0, frozen_folds=[result.nuisance.folds]
            )


# --------------------------------------------------------------- the replay surfaces


class TestTheReplaySurfaces:
    def test_the_frozen_rule_holds_arm_codes_and_its_replay_reproduces_the_fit(self) -> None:
        result = fit(interventions=[Static(2)])
        frozen = freeze_learned_rule(result)
        assert frozen.treat.dtype == np.int8
        np.testing.assert_array_equal(frozen.treat, realized_arms(result.nuisance.regimes))
        assert set(np.unique(frozen.treat)) == {0, 1, 2}
        replay = frozen_rule_replay(result.estimator, frozen).refit(
            result.data, random_state=0, frozen_folds=[result.nuisance.folds]
        )
        assert list(replay.estimates) == list(result.estimates)
        for name, estimate in result.estimates.items():
            assert replay.estimates[name].psi == estimate.psi, name
        assert fold_rules(replay.repeats[0].nuisance) is None

    def test_a_subset_refit_carries_each_kept_row_s_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _real()
        seen: list[Any] = []
        original = learned_module.frozen_rule_replay

        def recording(estimator: Any, rule: FrozenLearnedRule, rows: Any = None) -> Any:
            seen.append((rows, rule))
            return original(estimator, rule, rows)

        monkeypatch.setattr(learned_module, "frozen_rule_replay", recording)
        report = refute(result, estimand=NAME, tests=("subset",), n_replicates=2)
        assert len(report.tests[0].values) == 2 and len(seen) == 2
        realized = realized_arms(result.nuisance.regimes)
        for rows, rule in seen:
            np.testing.assert_array_equal(rule.rows(rows).treat, realized[rows])
            assert set(np.unique(rule.rows(rows).treat)) == {0, 1, 2}

    def test_simulated_confounding_at_zero_strength_reproduces_the_fit(self) -> None:
        study = CausalStudy(
            arms_frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=COVARIATES)
        )
        result = study.identify(LearnedRuleValue()).estimate(
            outcome_learner=outcome_learner(),
            treatment_learner=treatment_learner(),
            n_folds=5,
            cv_evaluation=True,
            simultaneous=False,
            random_state=0,
        )
        surface = simulated_confounding(
            result,
            estimand=NAME,
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.2)),
            random_state=3,
            arm_pair=(0, 1),
        )
        assert surface.complete
        assert surface.cells[0].estimate == result[NAME].psi
        assert max(abs(cell.displacement) for cell in surface.cells) > 5e-3

    def test_a_round_trip_keeps_the_record_and_the_rules(self, tmp_path: Any) -> None:
        from cleverly import load

        result = _real()
        restored = load(result.save(tmp_path / "arms.joblib"))
        assert restored.extra["learned_rule"] == result.extra["learned_rule"]
        np.testing.assert_array_equal(_stored(restored), _stored(result))
        assert _stored(restored).dtype == np.int8


# ------------------------------------------------------------------------------ the label

#: The label and row of every two-arm configuration at commit 854dac1, on
#: ``law_frame(240, 2)`` with the support helper's learners, and its key's treatment.
TWO_ARM_LABELS: dict[str, tuple[dict[str, Any], dict[str, Any], str | None]] = {
    "fold_evaluated": ({}, {}, "T24"),
    "stacked": ({"cv_evaluation": False}, {}, "T42.rows"),
    "in_sample_parametric": (
        {
            "cross_fit": False,
            "cv_evaluation": False,
            "outcome_learner": LogisticRegression(max_iter=1000),
        },
        {},
        "T43",
    ),
    "in_sample_adaptive": ({"cross_fit": False, "cv_evaluation": False}, {}, None),
    "contrast": ({"interventions": [Static(1)]}, {}, None),
    "weights": ({}, {"weights": "w"}, "T37.weighted"),
    "clusters": ({}, {"id": "c"}, "T37.clustered"),
    "strata": ({}, {"strata": ("W2",)}, "T41.binary"),
    "repeats": ({"repeats": 2}, {}, "T38.split_mean"),
    "fold_targeting": ({"targeting_scheme": "fold"}, {}, "T37.fold_targeted"),
}


def _with_design_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(w=1.0 + 0.5 * frame["W2"], c=np.arange(len(frame)) % 40)


class TestTheLabel:
    @pytest.mark.parametrize("configuration", list(TWO_ARM_LABELS))
    def test_a_two_arm_fit_keeps_its_key_and_its_row(self, configuration: str) -> None:
        estimator, options, row = TWO_ARM_LABELS[configuration]
        frame = _with_design_columns(support.law_frame(240, 2))
        with warnings.catch_warnings():
            # The adaptive in-sample configuration warns, and its label is what is read here.
            warnings.simplefilter("ignore", InSampleLearnerWarning)
            result = (
                support.learned_tmle(**estimator)
                .fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES, **options)
                .single()
            )
        assert result.config.support_key.treatment == ("binary",)
        for name, estimate in result.estimates.items():
            assert estimate.support.row == row, name
            assert estimate.support.label == ("experimental" if row is None else "supported")

    THREE_ARMS: ClassVar[dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {
        "fold_evaluated": ({}, {}),
        "stacked": ({"cv_evaluation": False}, {}),
        "in_sample_parametric": (
            {
                "cross_fit": False,
                "cv_evaluation": False,
                "outcome_learner": LogisticRegression(max_iter=1000),
            },
            {},
        ),
        "contrast": ({"interventions": [Static(2)]}, {}),
        "weights": ({}, {"weights": "w"}),
        "clusters": ({}, {"id": "c"}),
        "strata": ({}, {"strata": ("W2",)}),
        "repeats": ({"repeats": 2}, {}),
        "fold_targeting": ({"targeting_scheme": "fold"}, {}),
    }

    @pytest.mark.parametrize("configuration", list(THREE_ARMS))
    def test_a_three_arm_fit_reads_experimental(self, configuration: str) -> None:
        estimator, options = self.THREE_ARMS[configuration]
        result = fit(_with_design_columns(arms_frame()), fit_options=options, **estimator)
        assert result.config.support_key.treatment == ("arms3",)
        for name, estimate in result.estimates.items():
            assert np.isfinite(estimate.psi), name
            assert estimate.support.label == "experimental", name
            assert estimate.support.row is None, name


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.int8).tobytes()).hexdigest()
