r"""A declared final learner on the log-ratio EP curve (P1 PR 2, part e).

The log-ratio final stage minimises :math:`\sum_i w_i T_i\,\ell(p_i, \operatorname{expit}
f(V_i))`, :math:`T_i = Q^*_{t,i} + Q^*_{r,i}`, :math:`p_i = Q^*_{t,i}/T_i` (arXiv:2402.01972v2,
Section 3.2).  Algebra makes that risk the weighted log loss of the doubled set
``{(V_i, 1, w_i Q*_t,i), (V_i, 0, w_i Q*_r,i)}``, which the seam hands to a classifier.

- the bridge: an unpenalised logistic regression on the modifier column reproduces the default
  final stage (``logistic_newton`` on a ``Linear`` sieve) at unequal row weights; a classifier
  saturated in the modifier reproduces the default on an ``Indicators`` sieve and is never
  clipped; a logistic line, which is not saturated, can leave the hull of the shares, and the
  clip then moves the curve off the default at that cell;
- the two terms that can vanish at the truth, the total :math:`T_i` and the fractional label
  :math:`p_i`, each carry a witness that they do not vanish on the law the bridge runs on;
- the class-1 column is read through ``classes_``; the pairs share a group code that reaches
  a learner whose ``fit`` names it, and a Super Learner's inner folds keep each pair whole;
- R-ep-learner-api refuses a regressor and a classifier without ``sample_weight`` before any
  learner; the clip moves a probability outside the hull and records its share.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ConditionalContrast, EPMethod
from cleverly.bases import Indicators, Linear, Sieve
from cleverly.estimators import ep
from cleverly.exceptions import CapabilityError
from cleverly.learners import SuperLearner
from tests.unit._natural_course_support import NeverFit
from tests.unit.test_ep_contract import LEVELS, MODIFIERS, law, law_study, oracle, wrong

#: The bridge holds to this tolerance; a mutation must move the curve by more than the floor.
BRIDGE = 1e-8
FLOOR = 1e-3

GRID = {"W": np.array([0.0, 0.5, 1.0, 1.5, 2.0])}
CELLS = {"W": np.array([0.0, 1.0, 2.0])}


def weighted_frame() -> pd.DataFrame:
    """The three-cell law with unequal row weights.

    The weights depend on ``W`` alone, so the weighted law keeps the oracle's ``g`` and
    ``Qbar`` and the debiased regressions stay exact.
    """
    frame = law.frame()
    frame["n"] = 1.0 + 0.75 * frame["W"]
    return frame


def settings() -> dict[str, Any]:
    """The oracle nuisances: the debiased arm regressions are the law's, so ``T`` spans 0.7
    to 1.4 across the cells."""
    return {**oracle(), "g_bounds": (0.01, 0.99)}


def fit(sieve: Sieve, final_learner: Any = None, modifiers: Any = MODIFIERS) -> Any:
    contrast = ConditionalContrast(modifiers, scale="log_ratio")
    effect = law_study(weighted_frame(), weights="n").identify(contrast)
    method = EPMethod(sieve=sieve, final_learner=final_learner).with_overrides(**settings())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(method)


def tight_logistic() -> LogisticRegression:
    """The unpenalised logistic GLM, solved tightly enough for a 1e-8 bridge."""
    return LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000)


class SaturatedClassifier(ClassifierMixin, BaseEstimator):
    """The weighted share of label 1 at each distinct modifier row: saturated in ``V``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SaturatedClassifier:
        x = np.asarray(X, dtype=float)[:, 0]
        w = np.ones(x.size) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        y = np.asarray(y, dtype=float)
        self.levels_ = np.unique(x)
        self.share_ = np.array([np.sum((w * y)[x == v]) / np.sum(w[x == v]) for v in self.levels_])
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        x = np.asarray(X, dtype=float)[:, 0]
        p = self.share_[np.searchsorted(self.levels_, x)]
        return np.column_stack([1.0 - p, p])


class ReversedClasses(ClassifierMixin, BaseEstimator):
    """A logistic regression that reports its classes in the order (1, 0)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ReversedClasses:
        self.inner_ = tight_logistic().fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array([1.0, 0.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.inner_.predict_proba(X)[:, ::-1]


class GroupSpy(ClassifierMixin, BaseEstimator):
    """A logistic regression that records the ``groups`` its fit receives."""

    seen: ClassVar[list[Any]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None, groups: Any = None) -> GroupSpy:
        GroupSpy.seen.append(None if groups is None else np.asarray(groups))
        self.inner_ = tight_logistic().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.inner_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.inner_.predict_proba(X)


class PairSpy(ClassifierMixin, BaseEstimator):
    """A Super Learner candidate that records, per fit, the label counts at each modifier row."""

    seen: ClassVar[list[Any]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> PairSpy:
        x = np.asarray(X, dtype=float)[:, 0]
        y = np.asarray(y, dtype=float)
        PairSpy.seen.append(
            [
                (int(np.sum((x == v) & (y == 1))), int(np.sum((x == v) & (y == 0))))
                for v in CELLS["W"]
            ]
        )
        self.inner_ = LogisticRegression().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.inner_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.inner_.predict_proba(X)


class Unweighted(ClassifierMixin, BaseEstimator):
    """A classifier whose ``fit`` takes no ``sample_weight``: the sentinel fails if called."""

    def fit(self, X: Any, y: Any) -> Unweighted:
        raise AssertionError("R-ep-learner-api must refuse before any learner")

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover - fit must fail first
        raise AssertionError("never predicted")


class Certain(ClassifierMixin, BaseEstimator):
    """A classifier that returns probability 1 at ``W = 2`` and 0.6 elsewhere."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Certain:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        x = np.asarray(X, dtype=float)[:, 0]
        p = np.where(x == 2.0, 1.0, 0.6)
        return np.column_stack([1.0 - p, p])


def debiased_shares() -> tuple[np.ndarray, np.ndarray]:
    """The total ``T`` and the share ``p`` the final stage of the Linear-sieve fit reads."""
    seen: dict[str, np.ndarray] = {}
    stage = ep.final_stage

    def spy(debiased: Any, *args: Any, **kwargs: Any) -> Any:
        seen["t"] = np.asarray(debiased.targeted_t)
        seen["r"] = np.asarray(debiased.targeted_r)
        return stage(debiased, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(ep, "final_stage", spy)
        fit(Sieve(Linear(["W"]), size=1))
    total = seen["t"] + seen["r"]
    return total, seen["t"] / total


class TestTheBridge:
    def test_a_logistic_line_reproduces_the_default_stage_before_the_clip(self) -> None:
        sieve = Sieve(Linear(["W"]), size=1)
        default = fit(sieve)
        learned = fit(sieve, tight_logistic())
        learner = learned.model.learner
        assert isinstance(learner, ep.RatioLearner)
        frame = ep._learner_input(GRID)
        np.testing.assert_allclose(
            logit(learner.probability(frame)), default.predict(GRID), rtol=0, atol=BRIDGE
        )

    def test_inside_the_hull_the_clipped_curve_is_the_default_stage(self) -> None:
        sieve = Sieve(Linear(["W"]), size=1)
        default = fit(sieve)
        learned = fit(sieve, tight_logistic())
        np.testing.assert_allclose(
            learned.predict(GRID), default.predict(GRID), rtol=0, atol=BRIDGE
        )
        assert learned.ep_clipped_share == 0.0

    def test_a_logistic_line_can_leave_the_hull_and_the_clip_then_binds(self) -> None:
        """A smoother class is not saturated: at a wrong outcome regression and row weights
        that move with ``Y``, the line's probability at ``W = 2`` passes the hull."""
        frame = law.frame()
        frame["n"] = 1.0 + 0.5 * frame["Y"] + 0.75 * frame["A"]
        contrast = ConditionalContrast(MODIFIERS, scale="log_ratio")
        effect = law_study(frame, weights="n").identify(contrast)
        overrides = {**oracle(wrong(law.DiscreteLaw())), "g_bounds": (0.01, 0.99)}
        sieve = Sieve(Linear(["W"]), size=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            default = effect.estimate(EPMethod(sieve=sieve).with_overrides(**overrides))
            learned = effect.estimate(
                EPMethod(sieve=sieve, final_learner=tight_logistic()).with_overrides(**overrides)
            )
        learner = learned.model.learner
        cells = ep._learner_input(CELLS)
        np.testing.assert_allclose(
            logit(learner.probability(cells)), default.predict(CELLS), rtol=0, atol=BRIDGE
        )
        inside = ~learner.clipped(cells)
        assert inside.tolist() == [True, True, False]
        at_cells = learned.predict(CELLS)
        np.testing.assert_allclose(at_cells[inside], default.predict(CELLS)[inside], atol=BRIDGE)
        assert abs(at_cells[2] - default.predict(CELLS)[2]) > FLOOR
        share = float(np.mean(frame["W"] == 2.0))
        assert learned.ep_clipped_share == pytest.approx(share, abs=1e-12)

    def test_a_saturated_classifier_reproduces_the_indicators_stage_and_is_never_clipped(
        self,
    ) -> None:
        sieve = Sieve(Indicators("W"), size=1)
        default = fit(sieve, modifiers=LEVELS)
        learned = fit(sieve, SaturatedClassifier(), modifiers=LEVELS)
        np.testing.assert_allclose(
            learned.predict(CELLS), default.predict(CELLS), rtol=0, atol=BRIDGE
        )
        assert learned.ep_clipped_share == 0.0

    def test_the_total_varies_by_a_factor_two_so_a_dropped_total_is_seen(self) -> None:
        """Witness for the weight total: a constant ``T`` would hide ``w`` in place of ``w T``."""
        total, _ = debiased_shares()
        assert total.max() / total.min() >= 2.0 - 1e-12

    def test_the_labels_are_fractional_so_rounding_them_is_seen(self) -> None:
        """Witness for the fractional label: ``p`` in {0, 1} would hide rounding to labels."""
        _, share = debiased_shares()
        assert share.min() > 0.5 and share.max() < 0.75
        assert np.ptp(share) > 0.1

    @pytest.mark.parametrize("mutation", ["weight total", "labels"])
    def test_each_mutation_of_the_doubled_data_breaks_the_bridge(
        self, mutation: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate-mutation controls of the bridge, through the production seam."""
        fit_learner = ep.fit_learner

        def mutated(learner: Any, x: Any, y: Any, sample_weight: Any, **kwargs: Any) -> Any:
            n = y.size // 2
            total = sample_weight[:n] + sample_weight[n:]
            share = sample_weight[:n] / total
            if mutation == "weight total":
                sample_weight = sample_weight / np.concatenate([total, total])
            else:
                hard = (share >= np.median(share)).astype(float)
                sample_weight = np.concatenate([total * hard, total * (1.0 - hard)])
            return fit_learner(learner, x, y, sample_weight, **kwargs)

        sieve = Sieve(Linear(["W"]), size=1)
        default = fit(sieve)
        monkeypatch.setattr(ep, "fit_learner", mutated)
        learner = fit(sieve, tight_logistic()).model.learner
        moved = logit(learner.probability(ep._learner_input(GRID))) - default.predict(GRID)
        assert np.max(np.abs(moved)) > FLOOR


def test_the_class_one_column_is_read_through_classes() -> None:
    sieve = Sieve(Linear(["W"]), size=1)
    reference = fit(sieve, tight_logistic())
    reversed_ = fit(sieve, ReversedClasses())
    np.testing.assert_allclose(
        reversed_.predict(GRID), reference.predict(GRID), rtol=0, atol=BRIDGE
    )


def test_each_pair_shares_a_group_code() -> None:
    GroupSpy.seen = []
    curve = fit(Sieve(Linear(["W"]), size=1), GroupSpy())
    n = curve.fit.data.n
    assert len(GroupSpy.seen) == 1
    np.testing.assert_array_equal(GroupSpy.seen[0], np.concatenate([np.arange(n), np.arange(n)]))


def test_a_super_learner_keeps_each_pair_in_one_inner_fold() -> None:
    """Every candidate fit sees as many label-1 as label-0 rows at each modifier value."""
    PairSpy.seen = []
    learner = SuperLearner([PairSpy()], task="classification", n_folds=3, random_state=0)
    fit(Sieve(Linear(["W"]), size=1), learner)
    assert len(PairSpy.seen) >= 4  # three inner folds and the full fit
    for counts in PairSpy.seen:
        assert all(ones == zeros for ones, zeros in counts)
        assert sum(ones for ones, _ in counts) > 0


class KwargsOnly(ClassifierMixin, BaseEstimator):
    """A classifier whose ``fit(**kwargs)`` would silently drop the weights (review S4)."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> KwargsOnly:
        NeverFit.calls += 1
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


@pytest.mark.parametrize("learner", [LinearRegression(), Unweighted(), KwargsOnly()])
def test_a_learner_without_the_weighted_log_loss_refuses_before_any_learner(learner: Any) -> None:
    NeverFit.calls = 0
    contrast = ConditionalContrast(MODIFIERS, scale="log_ratio")
    effect = law_study().identify(contrast)
    method = EPMethod(final_learner=learner).with_overrides(
        outcome_learner=NeverFit(), treatment_learner=NeverFit()
    )
    with pytest.raises(CapabilityError, match="weighted log loss on two labels") as caught:
        effect.estimate(method)
    assert caught.value.refusal == ep.EP_RATIO_LEARNER_API_REFUSAL
    assert NeverFit.calls == 0


def test_the_clip_moves_a_certain_probability_into_the_hull_and_records_its_share() -> None:
    curve = fit(Sieve(Linear(["W"]), size=1), Certain())
    learner = curve.model.learner
    assert learner.high < 1.0
    values = curve.predict(CELLS)
    assert values[2] == pytest.approx(logit(learner.high), abs=1e-12)
    np.testing.assert_allclose(values[:2], logit(0.6), atol=1e-12)
    share = float(np.mean(weighted_frame()["W"] == 2.0))
    assert share > 0.0
    assert curve.ep_clipped_share == pytest.approx(share, abs=1e-12)


def test_the_planned_fits_count_the_log_ratio_final_learner() -> None:
    contrast = ConditionalContrast(MODIFIERS, scale="log_ratio")
    effect = law_study().identify(contrast)
    method = EPMethod(sieve=Sieve(Linear(["W"]), size=1), final_learner=tight_logistic())
    plan = effect.planned_fits(method.with_overrides(**settings()))
    assert any(row.role == "final_learner" and row.fits == 1 for row in plan.rows)


def test_the_engine_route_refuses_the_same_learner_in_check() -> None:
    """``EPEstimator.check`` refuses as the study route does (review N1)."""
    from cleverly.effect_modification import EffectModification

    estimator = ep.EPEstimator(
        {},
        EffectModification(Linear(["W"]), MODIFIERS, "log_ratio"),
        1,
        target="curve",
        final_learner=KwargsOnly(),
    )
    with pytest.raises(CapabilityError) as caught:
        estimator.check(law_study().data)
    assert caught.value.refusal == ep.EP_RATIO_LEARNER_API_REFUSAL
    difference = ep.EPEstimator(
        {},
        EffectModification(Linear(["W"]), MODIFIERS, "difference"),
        1,
        target="curve",
        final_learner=LinearRegression(),
    )
    difference.check(law_study().data)


def test_a_row_of_weight_zero_does_not_widen_the_hull() -> None:
    """Review N6: the hull is taken over the rows that enter the loss."""
    values = {"W": np.array([0.0, 1.0, 2.0, 2.0])}
    q_t = np.array([0.3, 0.4, 0.5, 0.99])
    q_r = np.array([0.3, 0.3, 0.3, 0.01])
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    fitted = ep.fit_ratio_learner(tight_logistic(), values, q_t, q_r, weights)
    share = q_t / (q_t + q_r)
    assert fitted.low == pytest.approx(float(share[:3].min()))
    assert fitted.high == pytest.approx(float(share[:3].max()))
    assert fitted.high < 0.99
