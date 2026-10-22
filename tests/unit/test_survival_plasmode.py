"""The longitudinal plasmode of a survival or competing-risk fit (X82 part E4b).

The law draws the history from the frozen node models of part E4a, and a declared hazard draws
the event at each node after the censoring. The truth integrates the event out: each history
carries its survival probability. On a two-node panel with a binary ``L_2`` the truth
enumerates every branch, so the tests below compare it with the g-formula worked by hand from
the frozen models. The hazard reads ``L_2`` through ``history_effect``, and ``L_2`` reads
``A_1``. A truth that holds ``L_2`` at its law under the observed ``A_1`` (the naive regression
truth) is the mutation control.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import Regimen
from cleverly.validation import (
    GaussianAdjustmentOutcome,
    LogisticHazardOutcome,
    MultinomialHazardOutcome,
    refute,
)
from cleverly.validation._longitudinal_plasmode import LongitudinalPlasmodeLaw
from cleverly.validation.outcome_processes import standardized_sum
from cleverly.validation.refute import _longitudinal_plasmode_target
from tests.unit.test_longitudinal_plasmode import RULE, _ltmle
from tests.unit.test_longitudinal_refutation import _survival_panel

HAZARD = LogisticHazardOutcome(intercept=-1.5, effect=0.8, adjustment_scale=0.5, history_effect=0.6)
CAUSES = MultinomialHazardOutcome(
    intercepts=(-1.5, -2.0), effects=(0.8, -0.3), adjustment_scale=0.5, history_effect=0.6
)
CONTRAST = "ate_regimen[never vs always @ t=2]"


def _fit(n: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle({"always": 1, "never": 0, "rule": RULE}).fit(
            _survival_panel(n=n),
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


@pytest.fixture(scope="module")
def survival() -> Any:
    return _fit(800)


@pytest.fixture(scope="module")
def law(survival: Any) -> LongitudinalPlasmodeLaw:
    return LongitudinalPlasmodeLaw.fit(survival)


def _competing_panel(n: int = 800) -> pd.DataFrame:
    """The survival panel with each event split into relapse (60%) and death."""
    frame = _survival_panel(n=n)
    relapse = np.random.default_rng(5).random(n) < 0.6
    y1, y2 = frame["Y1"].to_numpy(), frame["Y2"].to_numpy()
    first = np.where(y1 == 1, 1.0, np.where(np.isnan(y2), np.nan, y2))
    for cause, mask in (("R", relapse), ("D", ~relapse)):
        frame[f"{cause}1"] = np.where(np.isnan(y1), np.nan, y1 * mask)
        frame[f"{cause}2"] = np.where(np.isnan(first), np.nan, first * mask)
    return frame.drop(columns=["Y1", "Y2"])


@pytest.fixture(scope="module")
def competing() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle({"always": 1, "never": 0}).fit(
            _competing_panel(),
            outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


def _hand(law: Any, process: Any, a1: Any, a2: Any, cause: int = 0) -> float:
    """The two-node risk of ``cause`` by hand: ``a2`` maps ``L_2`` to the node-2 arm."""
    w = np.asarray(law.support.baseline, dtype=float)
    f = standardized_sum(w, process.adjustment_scale)
    n = w.shape[0]
    first = process.probabilities(f, np.full(n, float(a1)))
    model = law.covariates[1][0]
    p = model.probability(np.column_stack([w, np.broadcast_to(np.asarray(a1, float), n)]))
    centre, scale = law.history_centre[0], law.history_scale[0]
    risk = first[:, cause].copy()
    for level, chance in ((1.0, p), (0.0, 1.0 - p)):
        shift = f + process.history_effect * (level - centre) / scale
        second = process.probabilities(shift, np.full(n, float(a1) + a2(level)))
        risk += (1.0 - first.sum(axis=1)) * chance * second[:, cause]
    return float(np.mean(risk))


class TestTruthByHand:
    """The enumerated truth equals the g-formula worked by hand from the frozen models."""

    def test_the_law_enumerates(self, law: Any) -> None:
        assert law.binary and law.enumerable

    @pytest.mark.parametrize("arms", [(1, 1), (0, 0), (1, 0)])
    def test_a_static_regimen(self, law: Any, arms: tuple[int, int]) -> None:
        expected = _hand(law, HAZARD, arms[0], lambda _level: arms[1])
        value, error = law.truth(HAZARD, Regimen("r", arms), draws=None, horizon=2)
        assert value == pytest.approx(expected, abs=1e-12) and error == 0.0

    def test_the_first_horizon_reads_one_node(self, law: Any) -> None:
        w = np.asarray(law.support.baseline, dtype=float)
        f = standardized_sum(w, HAZARD.adjustment_scale)
        expected = float(np.mean(HAZARD.probabilities(f, np.ones(w.shape[0]))[:, 0]))
        value, _ = law.truth(HAZARD, Regimen("always", (1, 1)), draws=None, horizon=1)
        assert value == pytest.approx(expected, abs=1e-12)

    def test_a_dynamic_rule(self, law: Any) -> None:
        # Treat at node 1, then treat at node 2 exactly when L2 = 1.
        expected = _hand(law, HAZARD, 1, lambda level: level)
        value, _ = law.truth(HAZARD, RULE, draws=None, horizon=2)
        assert value == pytest.approx(expected, abs=1e-12)

    def test_the_naive_truth_is_the_mutation_control(self, survival: Any, law: Any) -> None:
        # Holding L2 at its law under the observed A1 drops the path A1 -> L2 -> event.
        w = np.asarray(law.support.baseline, dtype=float)
        f = standardized_sum(w, HAZARD.adjustment_scale)
        n = w.shape[0]
        observed = np.nan_to_num(np.asarray(survival.data.treatment[:, 0], dtype=float))
        p = law.covariates[1][0].probability(np.column_stack([w, observed]))
        first = HAZARD.probabilities(f, np.ones(n))[:, 0]
        naive = first.copy()
        for level, chance in ((1.0, p), (0.0, 1.0 - p)):
            standard = (level - law.history_centre[0]) / law.history_scale[0]
            shift = f + HAZARD.history_effect * standard
            naive += (1.0 - first) * chance * HAZARD.probabilities(shift, np.full(n, 2.0))[:, 0]
        value, _ = law.truth(HAZARD, Regimen("always", (1, 1)), draws=None, horizon=2)
        assert abs(value - float(np.mean(naive))) > 0.01

    def test_the_monte_carlo_truth_agrees_with_the_enumeration(self, law: Any) -> None:
        exact, _ = law.truth(HAZARD, RULE, draws=None, horizon=2)
        sampled, error = law.truth(HAZARD, RULE, draws=512, seed=4, horizon=2)
        assert error > 0.0 and abs(sampled - exact) < 4.0 * error

    @pytest.mark.parametrize("cause", [0, 1])
    def test_a_competing_cause(self, competing: Any, cause: int) -> None:
        law = LongitudinalPlasmodeLaw.fit(competing)
        expected = _hand(law, CAUSES, 1, lambda _level: 1, cause=cause)
        value, _ = law.truth(CAUSES, Regimen("always", (1, 1)), draws=None, horizon=2, cause=cause)
        assert value == pytest.approx(expected, abs=1e-12)


class TestTheDraw:
    def test_a_drawn_panel_masks_every_exit(self, survival: Any, law: Any) -> None:
        drawn = law.draw(HAZARD, seed=3, name="plasmode")
        event, kept = drawn.event, drawn.uncensored
        # An event at node 1 ends follow-up, so node 2 holds no treatment or covariate.
        assert np.isnan(drawn.treatment[event[:, 0], 1]).all()
        assert np.isnan(drawn.time_varying[1][event[:, 0], 0]).all()
        # A unit censored at node 1 has no event.
        assert not event[~kept[:, 0], 0].any() and not event[~kept[:, 0], 1].any()
        np.testing.assert_array_equal(drawn.outcome, event[:, -1].astype(float))
        assert 0.05 < event[:, 0].mean() < 0.6

    def test_the_node_one_events_follow_the_hazard(self, survival: Any, law: Any) -> None:
        # Pool draws: the first-node event rate among the uncensored is the mean hazard.
        hazard, observed = [], []
        for seed in range(50):
            drawn = law.draw(HAZARD, seed=seed, name="plasmode")
            w = np.asarray(drawn.baseline, dtype=float)
            f = standardized_sum(w, HAZARD.adjustment_scale, centre=law.centre, scale=law.scale)
            kept = drawn.uncensored[:, 0]
            a1 = drawn.treatment[kept, 0]
            hazard.append(HAZARD.probabilities(f[kept], a1)[:, 0].mean())
            observed.append(drawn.event[kept, 0].mean())
        assert abs(np.mean(observed) - np.mean(hazard)) < 0.01

    def test_a_competing_draw_splits_its_events_by_cause(self, competing: Any) -> None:
        law = LongitudinalPlasmodeLaw.fit(competing)
        drawn = law.draw(CAUSES, seed=2, name="plasmode")
        np.testing.assert_array_equal(drawn.cause_event.any(axis=2), drawn.event)
        assert drawn.cause_event[:, -1, 0].any() and drawn.cause_event[:, -1, 1].any()


def _run(result: Any, estimand: str, process: Any, draws: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(
            result,
            estimand=estimand,
            tests=("plasmode",),
            n_replicates=draws,
            random_state=1,
            simulated_outcome=process,
        )["plasmode"]


class TestRefits:
    def test_a_survival_contrast_passes(self) -> None:
        test = _run(_fit(1500), CONTRAST, HAZARD, 400)
        assert test.passed is True and len(test.readings) == 3
        assert "enumerates every branch" in test.scope_sentence

    def test_a_competing_contrast_sits_on_the_truth(self, competing: Any) -> None:
        estimand = "ate_regimen[never vs always, relapse @ t=2]"
        target = _longitudinal_plasmode_target(competing, estimand, CAUSES, "plasmode")
        test = _run(competing, estimand, CAUSES, 40)
        values = np.asarray(test.values)
        error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        assert test.declared_effect == pytest.approx(target.truth)
        assert abs(float(np.mean(values)) - target.truth) < 4.0 * error


class TestRefusals:
    def test_a_gaussian_declaration(self, survival: Any) -> None:
        with pytest.raises(CapabilityError, match="LogisticHazardOutcome"):
            _run(survival, CONTRAST, GaussianAdjustmentOutcome(), 10)

    def test_a_cause_count_mismatch(self, competing: Any) -> None:
        process = MultinomialHazardOutcome(intercepts=(-2.0,), effects=(0.5,))
        with pytest.raises(CapabilityError, match="declares 1 cause"):
            _run(competing, "ate_regimen[never vs always, relapse @ t=2]", process, 10)

    def test_a_history_effect_in_the_held_design(self, survival: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CapabilityError, match="no held-history truth"):
                refute(
                    survival,
                    estimand=CONTRAST,
                    tests=("simulated_outcome",),
                    n_replicates=10,
                    simulated_outcome=HAZARD,
                )
