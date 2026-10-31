r"""The omitted-variable point bound of the learned-rule value (X11 part (n)).

With the fold rules :math:`d_1, \dots, d_V` held fixed, the learned-rule value is the linear
functional :math:`\theta(\bar Q) = E[V^{-1}\sum_v \bar Q(d_v(W), W)]`.  It is the regime
functional of the stochastic regime :math:`g^\star(a \mid w) = V^{-1}\sum_v 1\{a = d_v(w)\}`, so
its Riesz representer is :math:`\alpha = g^\star / g` and the bias of the fold-evaluated estimate
is :math:`V^{-1}\sum_v E[\Delta\bar Q\,\Delta\alpha_v] = E[\Delta\bar Q\,\Delta\alpha]`.  The fit
stores every fold's rule at every row, and the bound reads that matrix.

* :class:`TestTheStoredRules` checks the ``(n, V)`` boolean matrix: its diagonal is the realized
  rule, its columns are the fold models' own rules, and a corrupted diagonal is caught.
* :class:`TestEqualRulesGiveTheKnownRegimeBound` is W-n1: when every fold learns ``1{W1 > 0}``,
  the bound equals the bound of that known regime, alone and with fold targeting, known weights
  and clusters.
* :class:`TestTheCombinedRepresenterOnAnExactLaw` is W-n2 and W-n3: an exact discrete law with an
  omitted binary ``U`` and two different frozen fold rules.  The long-minus-short bias equals
  :math:`E[\Delta\bar Q\,\Delta\alpha]` for the combined representer; a one-fold representer
  misses it; :math:`\sum_a g^{\star 2}/g < V^{-1}\sum_v 1/g(d_v)` holds strictly; and the
  diagonal representer, each row's own fold rule only, gives a strictly larger :math:`\nu^2`.
* :class:`TestThePointBoundOnly` checks the withheld limits, the summary notes, the refusals
  and serialization.
"""

from __future__ import annotations

import importlib
import itertools
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
import cleverly.interventions._rule_learners as rule_learners
import cleverly.sensitivity.omitted_variable as ov
from cleverly.estimators import TMLE, serialize
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import LearnedRule, Rule
from cleverly.interventions.learned import (
    LEARNED_RULE_BENCHMARK_REFUSAL,
    LEARNED_RULE_LIMITS,
    fold_rules,
    freeze_learned_rule,
    frozen_rule_replay,
)
from tests.unit import _learned_rule_support as support
from tests.unit.test_learned_rule_compositions import SignOfW1

pytestmark = pytest.mark.xdist_group("learned_rule_bound")

FIT = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}


def _rules(result: Any) -> np.ndarray:
    rules = fold_rules(result.repeats[0].nuisance)
    assert rules is not None
    return rules


def _off_diagonal(result: Any) -> np.ndarray:
    """``(n, V)``: ``True`` where column ``v`` is not the row's own fold."""
    nuisance = result.repeats[0].nuisance
    assignment = np.asarray(nuisance.folds.assignment, dtype=int)
    return assignment[:, None] != np.arange(nuisance.folds.n_folds)[None, :]


def _assert_both_values_off_the_diagonal(result: Any) -> None:
    """The nonzero witness: a column filled only on its own fold's rows cannot pass."""
    rules = _rules(result)
    off = _off_diagonal(result)
    assert np.any(rules[off]) and not np.all(rules[off])


class TieAtW2(ClassifierMixin, BaseEstimator):
    """The outcome learner of :func:`support.outcome_learner`, with a blip of exactly zero.

    At every row with ``W2 = 1`` it predicts one half at both arms, so the plug-in rule there
    is a tie, which assigns the lower arm.  ``column`` is the position of ``W2`` in the
    design: 2 on the pooled design ``(A, W1, W2)``, 1 on the per-arm design ``(W1, W2)``.
    """

    def __init__(self, column: int = 2) -> None:
        self.column = column

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TieAtW2:
        self.model_ = support.outcome_learner().fit(np.asarray(X, dtype=float), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        proba = np.array(self.model_.predict_proba(design), dtype=float)
        proba[design[:, self.column] == 1.0] = 0.5
        return proba


class TieBlip(RegressorMixin, BaseEstimator):
    """A blip learner whose fitted blip is exactly zero, the threshold, wherever ``W2 = 1``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TieBlip:
        self.model_ = LinearRegression().fit(np.asarray(X, dtype=float), y, sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        return np.where(design[:, 1] == 1.0, 0.0, self.model_.predict(design))


def _spy_fold_rules(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every :class:`FoldRule` the rule learner fits, in the order it fits them."""
    recorded: list[Any] = []
    original = rule_learners.fit_fold_rule

    def spy(*args: Any, **kwargs: Any) -> Any:
        rule = original(*args, **kwargs)
        recorded.append((kwargs.get("fold"), rule))
        return rule

    monkeypatch.setattr(rule_learners, "fit_fold_rule", spy)
    return recorded


def _by_hand_by_arm(result: Any, frame: pd.DataFrame, learner: Any) -> np.ndarray:
    """Refit fold ``v``'s two per-arm outcome regressions by hand and read the sign."""
    covariates = frame[list(support.COVARIATES)].to_numpy()
    treatment = frame["A"].to_numpy()
    outcome = frame["Y"].to_numpy()
    folds = result.repeats[0].nuisance.folds
    columns = []
    for train, _ in folds:
        rows = np.asarray(train)
        at = {}
        for arm in (0.0, 1.0):
            own = rows[treatment[rows] == arm]
            model = clone_learner(learner).fit(covariates[own], outcome[own])
            at[arm] = model.predict_proba(covariates)[:, 1]
        columns.append(at[1.0] - at[0.0] > 0.0)
    return np.column_stack(columns)


def clone_learner(learner: Any) -> Any:
    """A fresh copy of ``learner``."""
    from sklearn.base import clone

    return clone(learner)


# --------------------------------------------------------------------- the stored rules


class TestTheStoredRules:
    """The fit stores each fold's rule at every row, as booleans, from the fold's own model."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return support.fit(support.law_frame(400, 1))

    def test_the_matrix_is_n_by_v_booleans_whose_diagonal_is_the_realized_rule(
        self, result: Any
    ) -> None:
        rules = _rules(result)
        nuisance = result.repeats[0].nuisance
        assert rules.shape == (400, 5) and rules.dtype == bool
        # One byte per row and fold: the memory cost of the matrix.
        assert rules.nbytes == 400 * 5
        own = rules[np.arange(400), np.asarray(nuisance.folds.assignment)]
        np.testing.assert_array_equal(own, nuisance.regimes.values[:, 1, 0] > 0.5)
        # The nonzero witness: the fold rules differ, so the off-diagonal entries carry
        # information the realized rule does not.
        assert np.any(rules != rules[:, :1])
        # The nuisances keep no second copy.
        assert nuisance.fold_plug_in_rules is None

    def test_each_column_is_the_plug_in_rule_of_that_fold_s_outcome_regression(
        self, result: Any
    ) -> None:
        """Refit fold ``v``'s outcome regression by hand and read its blip at every row."""
        frame = support.law_frame(400, 1)
        covariates = frame[list(support.COVARIATES)].to_numpy()
        design = np.column_stack([frame["A"].to_numpy(), covariates])
        rules = _rules(result)
        for fold, (train, _) in enumerate(result.repeats[0].nuisance.folds):
            rows = np.asarray(train)
            model = support.outcome_learner().fit(design[rows], frame["Y"].to_numpy()[rows])
            np.testing.assert_array_equal(rules[:, fold], support.blip(model, covariates) > 0.0)

        _assert_both_values_off_the_diagonal(result)

    def test_each_column_of_a_per_arm_regression_is_that_fold_s_two_arm_models(self) -> None:
        """Refit fold ``v``'s per-arm regressions on ``train & (A == a)`` and read the sign."""
        frame = support.law_frame(400, 1)
        result = support.fit(frame, outcome_by_arm=True)
        rules = _rules(result)
        assert rules.shape == (400, 5) and rules.dtype == bool
        np.testing.assert_array_equal(
            rules, _by_hand_by_arm(result, frame, support.outcome_learner())
        )
        _assert_both_values_off_the_diagonal(result)

    @pytest.mark.parametrize("outcome_by_arm", [False, True], ids=["pooled", "by_arm"])
    def test_a_tie_assigns_the_lower_arm_at_every_fold(self, outcome_by_arm: bool) -> None:
        """The tie witness: a blip of exactly zero at ``W2 = 1`` gives ``False`` in every column.

        A matrix built with ``>=`` in place of ``>`` puts ``True`` there.  The realized rule
        keeps ``>``, so the diagonal check would also refuse that fit.
        """
        frame = support.law_frame(400, 1)
        learner = TieAtW2(column=1 if outcome_by_arm else 2)
        result = support.fit(frame, outcome_learner=learner, outcome_by_arm=outcome_by_arm)
        rules = _rules(result)
        tied = frame["W2"].to_numpy() == 1.0
        assert 100 < tied.sum() < 300
        assert not np.any(rules[tied])
        # Away from the tie the rules still take both values off the diagonal.
        off = _off_diagonal(result) & ~tied[:, None]
        assert np.any(rules[off]) and not np.all(rules[off])
        if outcome_by_arm:
            np.testing.assert_array_equal(rules, _by_hand_by_arm(result, frame, learner))

    @pytest.mark.parametrize(
        "rule",
        [
            LearnedRule(loss="blip", learner=LinearRegression()),
            LearnedRule(loss="weighted_classification", learner=LogisticRegression(max_iter=1000)),
        ],
        ids=["blip", "weighted_classification"],
    )
    def test_each_column_of_a_rule_learner_is_that_fold_s_fitted_rule(
        self, rule: LearnedRule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Column ``v`` is fold ``v``'s fitted rule at every row, off its own fold too.

        The spy records each fold's :class:`FoldRule` as the fit returns it.  A matrix that
        evaluates each fold's rule on that fold's own rows only fails here.
        """
        recorded = _spy_fold_rules(monkeypatch)
        result = support.fit(support.law_frame(400, 1), learned_rule=rule)
        rules = _rules(result)
        assert rules.shape == (400, 5) and rules.dtype == bool
        assert [fold for fold, _ in recorded] == list(range(5))
        covariates = result.data.covariates
        for fold, fold_rule in recorded:
            np.testing.assert_array_equal(rules[:, fold], fold_rule.treat(covariates))
        _assert_both_values_off_the_diagonal(result)

    def test_a_rule_learner_tie_assigns_the_lower_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tie witness of a rule learner: a fitted blip of exactly zero treats no one."""
        recorded = _spy_fold_rules(monkeypatch)
        frame = support.law_frame(400, 1)
        result = support.fit(frame, learned_rule=LearnedRule(loss="blip", learner=TieBlip()))
        rules = _rules(result)
        tied = frame["W2"].to_numpy() == 1.0
        assert not np.any(rules[tied])
        for fold, fold_rule in recorded:
            np.testing.assert_array_equal(fold_rule.margin(result.data.covariates)[tied], 0.0)
            np.testing.assert_array_equal(rules[:, fold], fold_rule.treat(result.data.covariates))
        off = _off_diagonal(result) & ~tied[:, None]
        assert np.any(rules[off]) and not np.all(rules[off])

    def test_a_corrupted_diagonal_is_a_defect(self, result: Any) -> None:
        """The mutation: one row's entry at its own fold disagrees with its realized rule."""
        nuisance = result.repeats[0].nuisance
        corrupted = np.array(nuisance.regimes.fold_rules)
        fold = int(nuisance.folds.assignment[0])
        corrupted[0, fold] = not corrupted[0, fold]
        broken = replace(nuisance, regimes=replace(nuisance.regimes, fold_rules=corrupted))
        with pytest.raises(RuntimeError, match="defect in cleverly"):
            fold_rules(broken)

    def test_the_fit_checks_the_diagonal_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same corruption at fit time raises from the fit, before any bound is asked for."""
        tmle_module = importlib.import_module("cleverly.estimators.tmle")
        original = tmle_module.learned_rule_treat

        def corrupting(*args: Any, **kwargs: Any) -> Any:
            treat, margin, rules = original(*args, **kwargs)
            rules = np.array(rules)
            rules[:, :] = ~rules
            return treat, margin, rules

        monkeypatch.setattr(tmle_module, "learned_rule_treat", corrupting)
        with pytest.raises(RuntimeError, match="defect in cleverly"):
            support.fit(support.law_frame(400, 1))

    def test_a_frozen_rule_replay_stores_none_and_the_bound_refuses_it(self, result: Any) -> None:
        estimator = frozen_rule_replay(result.estimator, freeze_learned_rule(result))
        replay = estimator.fit(support.law_frame(400, 1), **FIT).single()
        assert fold_rules(replay.repeats[0].nuisance) is None
        with pytest.raises(CapabilityError, match="FrozenLearnedRule") as caught:
            ov.omitted_variable_bounds(replay, support.NAME)
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert not replay.sensitivity.capability("omitted_confounding").available


# ------------------------------------------------------------------------------- W-n1


def _w1_rule() -> Rule:
    return Rule(lambda w: (w["W1"] > 0).astype(int), name="w1", rule_kind="known")


#: The configurations of W-n1: each one part G's regime bound already admits.
CONFIGURATIONS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "pooled": ({}, {}),
    "fold_targeting": ({"targeting_scheme": "fold"}, {}),
    "weights": ({}, {"weights": "w"}),
    "clusters": ({}, {"id": "cluster"}),
}


class TestEqualRulesGiveTheKnownRegimeBound:
    """W-n1: every fold learns ``1{W1 > 0}``, so ``g*`` is the one-hot density of that rule."""

    @pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
    def test_the_bound_equals_the_known_regime_bound(self, configuration: str) -> None:
        settings, columns = CONFIGURATIONS[configuration]
        frame = support.law_frame(400, 2)
        rng = np.random.default_rng(5)
        frame = frame.assign(w=rng.uniform(0.5, 2.0, 400), cluster=np.arange(400) // 4)
        common = {
            "outcome_learner": SignOfW1(),
            "treatment_learner": support.treatment_learner(),
            "n_folds": 5,
            "simultaneous": False,
            "random_state": 0,
            **support.FOLD_EVALUATED,
            **settings,
        }
        fit_args = {**FIT, **columns}
        learned = TMLE(learned_rule=LearnedRule(), **common).fit(frame, **fit_args).single()
        known = TMLE(interventions=[_w1_rule()], **common).fit(frame, **fit_args).single()
        rules = _rules(learned)
        # Every column is the one rule, so the combined density is its one-hot density.
        np.testing.assert_array_equal(rules, np.repeat(rules[:, :1], 5, axis=1))
        np.testing.assert_array_equal(rules[:, 0], frame["W1"].to_numpy() > 0)
        mine = ov.sensitivity_elements(learned, support.NAME)
        theirs = ov.sensitivity_elements(known, "ey_regime[w1]")
        np.testing.assert_allclose(mine.riesz_representer, theirs.riesz_representer, rtol=1e-12)
        assert mine.nu2 == pytest.approx(theirs.nu2, rel=1e-10)
        assert mine.sigma2 == pytest.approx(theirs.sigma2, rel=1e-10)
        a = ov.omitted_variable_bounds(learned, support.NAME, cf_y=0.05, cf_d=0.04)
        b = ov.omitted_variable_bounds(known, "ey_regime[w1]", cf_y=0.05, cf_d=0.04)
        assert (a.lower, a.upper) == pytest.approx((b.lower, b.upper), rel=1e-10)
        assert a.robustness_value == pytest.approx(b.robustness_value, rel=1e-8)
        assert a.features == ("learned_rule",) and b.features == ("regime",)


# ------------------------------------------------------------------------- W-n2, W-n3

#: The exact law. ``W`` takes three values, ``U`` is binary and omitted from the fit.
P_W = np.array([0.3, 0.4, 0.3])
#: ``P(U = 1 | W = w)``.
P_U = np.array([0.3, 0.5, 0.6])
#: ``g_L(1 | w, u)``, rows ``w``, columns ``u``.
G_LONG = np.array([[0.2, 0.7], [0.25, 0.75], [0.2, 0.65]])
#: ``Qbar_L(a, w, u)``, indexed ``[a, w, u]``.
Q_LONG = np.array(
    [
        [[0.1, 0.6], [0.2, 0.7], [0.15, 0.65]],
        [[0.3, 0.9], [0.25, 0.85], [0.05, 0.55]],
    ]
)


def _law_u() -> np.ndarray:
    """``P(U = u | W = w)``, ``(3, 2)``."""
    return np.column_stack([1.0 - P_U, P_U])


def _g_long(a: int) -> np.ndarray:
    """``g_L(a | w, u)``, ``(3, 2)``."""
    return G_LONG if a == 1 else 1.0 - G_LONG


def g_short(a: int) -> np.ndarray:
    """``g_S(a | w) = sum_u P(u | w) g_L(a | w, u)``, ``(3,)``."""
    return np.sum(_law_u() * _g_long(a), axis=1)


def q_short(a: int) -> np.ndarray:
    """``Qbar_S(a, w) = sum_u P(u | a, w) Qbar_L(a, w, u)``, ``(3,)``."""
    joint = _law_u() * _g_long(a)
    return np.sum(joint * Q_LONG[a], axis=1) / np.sum(joint, axis=1)


class TableTreatment(ClassifierMixin, BaseEstimator):
    """The short treatment mechanism ``g_S(1 | W)`` of the exact law, whatever the rows."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TableTreatment:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        w = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        p = g_short(1)[w]
        return np.column_stack([1.0 - p, p])


class ParityRule(ClassifierMixin, BaseEstimator):
    """An outcome model whose plug-in rule depends on its training rows.

    The blip is ``+0.1`` at ``W = 0`` and ``-0.1`` at ``W = 2``.  At ``W = 1`` it is ``+0.1``
    when the model trained on an odd number of rows and ``-0.1`` otherwise.  Two folds of 24
    and 23 rows give training sets of 23 and 24 rows, so the two fold rules differ at ``W = 1``
    alone.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ParityRule:
        self.classes_ = np.array([0.0, 1.0])
        self.odd_ = bool(np.asarray(X).shape[0] % 2)
        return self

    def blip(self) -> np.ndarray:
        return np.array([0.1, 0.1 if self.odd_ else -0.1, -0.1])

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], np.rint(design[:, 1]).astype(int)
        p = 0.5 + a * self.blip()[w]
        return np.column_stack([1.0 - p, p])


#: Rows of each support point ``(w, a, y)``. One point has one row fewer, so the sample has 47
#: rows and its two folds have different sizes.
REPLICATES = 4


def exact_frame() -> pd.DataFrame:
    """The short law of ``(W, A, Y)``, one weighted row per replicate of each support point.

    The weights of a point's replicates sum to its probability, so every weighted mean is an
    expectation under the law.
    """
    rows = []
    for w, a, y in itertools.product(range(3), range(2), range(2)):
        q = q_short(a)[w]
        p = P_W[w] * g_short(a)[w] * (q if y else 1.0 - q)
        copies = REPLICATES - 1 if (w, a, y) == (0, 0, 0) else REPLICATES
        rows.extend(
            {"W": float(w), "A": float(a), "Y": float(y), "p": p / copies} for _ in range(copies)
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def exact_fit() -> Any:
    return (
        TMLE(
            learned_rule=LearnedRule(),
            outcome_learner=ParityRule(),
            treatment_learner=TableTreatment(),
            n_folds=2,
            simultaneous=False,
            random_state=0,
            g_bounds=(0.01, 0.99),
            **support.FOLD_EVALUATED,
        )
        .fit(exact_frame(), outcome="Y", treatment="A", covariates=["W"], weights="p")
        .single()
    )


def _rule_table(result: Any) -> np.ndarray:
    """``d_v(w)`` as a ``(3, V)`` table, read from the stored matrix, one value per ``w``."""
    rules = _rules(result)
    w = np.rint(result.data.covariates[:, 0]).astype(int)
    table = np.array([rules[w == level][0] for level in range(3)])
    for level in range(3):
        # Each column is a function of W: every row at one level carries one value.
        assert np.all(rules[w == level] == table[level])
    return table.astype(float)


def _g_star(table: np.ndarray, a: int) -> np.ndarray:
    """``g*(a | w) = V^{-1} sum_v 1{a = d_v(w)}``, ``(3,)``."""
    return np.mean(table if a == 1 else 1.0 - table, axis=1)


ELEMENT = support.NAME


class TestTheCombinedRepresenterOnAnExactLaw:
    """W-n2 and W-n3 on the exact law with an omitted binary ``U``."""

    def test_the_two_fold_rules_differ_at_one_level(self, exact_fit: Any) -> None:
        sizes = [np.asarray(test).size for _, test in exact_fit.nuisance.folds]
        assert sorted(sizes) == [23, 24]
        table = _rule_table(exact_fit)
        np.testing.assert_array_equal(table[0], [1.0, 1.0])
        np.testing.assert_array_equal(table[2], [0.0, 0.0])
        assert sorted(table[1]) == [0.0, 1.0]

    def test_the_representer_is_g_star_over_g(self, exact_fit: Any) -> None:
        table = _rule_table(exact_fit)
        elements = ov.sensitivity_elements(exact_fit, ELEMENT)
        w = np.rint(exact_fit.data.covariates[:, 0]).astype(int)
        a = np.asarray(exact_fit.data.treatment, dtype=int)
        star = np.where(a == 1, _g_star(table, 1)[w], _g_star(table, 0)[w])
        g = np.where(a == 1, g_short(1)[w], g_short(0)[w])
        np.testing.assert_allclose(elements.riesz_representer, star / g, rtol=1e-12)

    def test_nu2_is_the_second_moment_of_the_combined_representer(self, exact_fit: Any) -> None:
        """At the law's own mechanism both estimators of ``nu^2`` are exact.

        ``E[alpha^2] = E[sum_a g*(a | W)^2 / g(a | W)]``, strictly below
        ``V^{-1} sum_v E[1 / g(d_v(W) | W)]``, because the two rules differ at ``W = 1``,
        where ``g*`` is one half for each arm.
        """
        table = _rule_table(exact_fit)
        combined = float(np.sum(P_W * sum(_g_star(table, a) ** 2 / g_short(a) for a in (0, 1))))
        per_fold = float(
            np.mean(
                [
                    np.sum(P_W * np.where(table[:, v] == 1.0, 1 / g_short(1), 1 / g_short(0)))
                    for v in range(2)
                ]
            )
        )
        for estimator in ("doubly_robust", "plugin"):
            nu2 = ov.sensitivity_elements(exact_fit, ELEMENT, nu2_estimator=estimator).nu2
            assert nu2 == pytest.approx(combined, rel=1e-12)
        # The gap is P(W = 1) (1 / g(1 | 1) + 1 / g(0 | 1)) / 4, declared from the law.
        gap = P_W[1] * (1 / g_short(1)[1] + 1 / g_short(0)[1]) / 4
        assert per_fold - combined == pytest.approx(gap, rel=1e-12)
        assert gap > 0.3

    def test_the_bias_is_the_product_of_the_two_differences_and_the_bound_reaches_it(
        self, exact_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        r"""W-n2: Theorem 2 of Chernozhukov et al. (2026), for the combined representer.

        With :math:`\alpha_L = g^\star(a \mid w) / g_L(a \mid w, u)` and
        :math:`\alpha_S = g^\star / g_S`, the long-minus-short difference of the value of the
        frozen rules is :math:`E[\Delta\bar Q\,\Delta\alpha]`.  At the law's own strengths
        ``cf_y``, ``cf_d`` and ``rho``, the bound built from the fit's :math:`\nu^2` and the
        short residual variance equals that bias exactly.  ``alpha_S`` is a function of
        ``(A, W)``, and ``Delta Qbar`` is orthogonal to every such function, so the check
        that discriminates is the one on ``g*``: the representer of fold 0's rule alone misses
        the bias, and so does the diagonal ``nu^2`` of W-n3.
        """
        table = _rule_table(exact_fit)
        star = {a: _g_star(table, a) for a in (0, 1)}
        one = {a: (table[:, 0] if a == 1 else 1.0 - table[:, 0]) for a in (0, 1)}
        law = P_W[:, None] * _law_u()

        def theta(weights: dict[int, np.ndarray]) -> tuple[float, float]:
            long = sum(float(np.sum(law * weights[a][:, None] * Q_LONG[a])) for a in (0, 1))
            short = sum(float(np.sum(P_W * weights[a] * q_short(a))) for a in (0, 1))
            return long, short

        def moments(weights: dict[int, np.ndarray]) -> dict[str, float]:
            """``E[dQ dalpha]``, ``E[dQ^2]``, ``E[alpha_L^2]`` and ``E[alpha_S^2]``."""
            out = {"product": 0.0, "dq2": 0.0, "long2": 0.0, "short2": 0.0}
            for a, w, u in itertools.product(range(2), range(3), range(2)):
                mass = law[w, u] * _g_long(a)[w, u]
                dq = Q_LONG[a, w, u] - q_short(a)[w]
                alpha_long = weights[a][w] / _g_long(a)[w, u]
                alpha_short = weights[a][w] / g_short(a)[w]
                out["product"] += mass * dq * (alpha_long - alpha_short)
                out["dq2"] += mass * dq**2
                out["long2"] += mass * alpha_long**2
                out["short2"] += mass * alpha_short**2
            return out

        long, short = theta(star)
        combined = moments(star)
        assert long - short == pytest.approx(combined["product"], rel=1e-12)
        # The nonzero witness: the omitted U moves the value of the frozen rules.  The value
        # is the one docs/technical-reference/validation-methods.md quotes.
        assert abs(long - short) > 0.02
        assert long - short == pytest.approx(-0.0284, abs=5e-5)
        # The mutation: fold 0's rule alone gives another product.
        assert abs(moments(one)["product"] - (long - short)) > 0.01

        nu2 = ov.sensitivity_elements(exact_fit, ELEMENT).nu2
        assert nu2 == pytest.approx(combined["short2"], rel=1e-12)
        assert nu2 == pytest.approx(1.823, abs=5e-4)
        residual = sum(
            float(np.sum(law * _g_long(a) * Q_LONG[a] * (1.0 - Q_LONG[a]))) for a in (0, 1)
        )
        sigma2 = residual + combined["dq2"]
        cf_y = combined["dq2"] / sigma2
        cf_d = 1.0 - combined["short2"] / combined["long2"]
        rho = combined["product"] / np.sqrt(
            combined["dq2"] * (combined["long2"] - combined["short2"])
        )
        assert abs(rho) < 1.0 and 0.0 < cf_y < 1.0 and 0.0 < cf_d < 1.0

        def reach(second_moment: float) -> float:
            strength = ov._confounding_strength(cf_y, cf_d, float(rho))
            return float(strength * np.sqrt(sigma2 * second_moment))

        assert reach(nu2) == pytest.approx(abs(long - short), rel=1e-10)
        # The mutation: with the diagonal representer the bound overshoots the bias.
        monkeypatch.setattr(ov, "_combined_rule_regimes", lambda nuisance: nuisance.regimes)
        diagonal = ov.sensitivity_elements(exact_fit, ELEMENT).nu2
        assert diagonal == pytest.approx(2.078, abs=5e-4)
        assert reach(diagonal) > abs(long - short) * 1.02

    def test_the_diagonal_representer_gives_a_larger_nu2(
        self, exact_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W-n3: each row's own fold rule only, the realized one-hot density, inflates ``nu^2``.

        The mutation replaces the combined density with the realized rule, which is what the
        bound would read without the stored matrix.  On every row the doubly robust term is
        then ``2 / g(d_i | W_i) - 1{A_i = d_i} / g(A_i | W_i)^2``.  The rows at ``W = 1`` split
        between the two folds in the shares the draw gives, not one half each, so the gap is
        below the population gap of the last test.
        """
        combined = ov.sensitivity_elements(exact_fit, ELEMENT).nu2
        monkeypatch.setattr(ov, "_combined_rule_regimes", lambda nuisance: nuisance.regimes)
        diagonal = ov.sensitivity_elements(exact_fit, ELEMENT).nu2
        data = exact_fit.data
        w = np.rint(data.covariates[:, 0]).astype(int)
        a = np.asarray(data.treatment, dtype=int)
        rule = np.asarray(exact_fit.nuisance.regimes.values[:, 1, 0] > 0.5, dtype=int)
        g_rule = np.where(rule == 1, g_short(1)[w], g_short(0)[w])
        g_obs = np.where(a == 1, g_short(1)[w], g_short(0)[w])
        terms = 2.0 / g_rule - (a == rule) / g_obs**2
        assert diagonal == pytest.approx(float(np.average(terms, weights=data.weights)), rel=1e-12)
        assert diagonal > combined + 0.1


# ---------------------------------------------------------------------- the point bound


class TestThePointBoundOnly:
    """The limits are withheld, the summary says what is fixed, and the result round-trips."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return support.fit(support.law_frame(400, 1))

    def test_every_limit_refuses_with_the_learned_rule_reason(self, result: Any) -> None:
        bounds = ov.omitted_variable_bounds(result, support.NAME)
        assert bounds.lower < result[support.NAME].psi < bounds.upper
        assert bounds.robustness_value is not None
        assert bounds.limit_reading is None
        for name in ("ci_lower", "ci_upper", "robustness_value_ci"):
            with pytest.raises(CapabilityError) as caught:
                getattr(bounds, name)
            assert caught.value.refusal.reason is RefusalReason.INFERENCE_GAP
            assert LEARNED_RULE_LIMITS in str(caught.value)
        elements = ov.sensitivity_elements(result, support.NAME)
        assert elements.psi_nu2 is None and elements.psi_max_bias is None
        assert elements.limit_refusal is not None and elements.limit_refusal.key == "learned_rule"
        assert "X11 (n2) in docs/roadmap.md" in LEARNED_RULE_LIMITS

    def test_each_summary_names_the_fixed_rules(self, result: Any) -> None:
        from cleverly._summary.vocabulary import NOTE_TEXTS

        for summary in (
            ov.omitted_variable_bounds(result, support.NAME).summary(),
            ov.sensitivity_elements(result, support.NAME).summary(),
            ov.robustness_value(result, support.NAME).summary(),
        ):
            assert summary.note("learned_rule_bound").text == NOTE_TEXTS["learned_rule_bound"]
            assert "point bound" in summary.note("limits_withheld").text

    def test_the_contour_and_the_facade_run(self, result: Any) -> None:
        assert ov.contour_data(result, support.NAME) is not None
        facade = result.sensitivity.omitted_confounding(support.NAME)
        assert facade.lower == ov.omitted_variable_bounds(result, support.NAME).lower

    def test_the_benchmark_refuses(self, result: Any) -> None:
        with pytest.raises(CapabilityError) as caught:
            ov.benchmark(result, ["W2"], estimand=support.NAME)
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert LEARNED_RULE_BENCHMARK_REFUSAL.condition in str(caught.value)
        row = result.sensitivity.capability("benchmark")
        assert not row.available
        assert LEARNED_RULE_BENCHMARK_REFUSAL.condition in (row.reason or "")

    def test_the_rules_survive_serialization(self, result: Any, tmp_path: Any) -> None:
        expected = ov.omitted_variable_bounds(result, support.NAME)
        path = tmp_path / "learned.cleverly"
        result.save(path)
        for loaded in (serialize.loads(serialize.dumps(result)), cleverly.load(path)):
            np.testing.assert_array_equal(_rules(loaded), _rules(result))
            bounds = ov.omitted_variable_bounds(loaded, support.NAME)
            assert (bounds.lower, bounds.upper) == (expected.lower, expected.upper)
