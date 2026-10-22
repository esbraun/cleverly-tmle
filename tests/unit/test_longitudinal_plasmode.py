"""The longitudinal plasmode (X82 part E4).

The law resamples the baselines and draws each history from models frozen on the fit. The
truth is the sequential g-formula of that law. On a panel whose time-varying column is binary
the truth enumerates every branch, so the tests below compare it with the g-formula worked by
hand from the frozen models. A truth that holds ``L_2`` at its law under the observed ``A_1``
(the naive regression truth) is the mutation control: the outcome reads ``L_2``, and ``L_2``
reads ``A_1``, so the two truths differ.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen, Regimen
from cleverly.validation import GaussianAdjustmentOutcome, refute
from cleverly.validation import _longitudinal_plasmode as plasmode_module
from cleverly.validation._longitudinal_plasmode import LongitudinalPlasmodeLaw
from cleverly.validation.refute import _longitudinal_plasmode_target
from tests.unit.test_longitudinal_generated_events import HAZARD, _survival_panel
from tests.unit.test_longitudinal_generated_outcomes import _fit as _gaussian_history_fit
from tests.unit.test_longitudinal_generated_outcomes import _panel as _gaussian_history_panel

EFFECT = 0.7
HISTORY = 0.5
PROCESS = GaussianAdjustmentOutcome(effect=EFFECT, history_effect=HISTORY)


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _binary_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A two-node panel whose binary ``L2`` reads ``A1``, and whose ``A2`` reads ``L2``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.binomial(1, _expit(w)).astype(float)
    c1 = rng.binomial(1, 0.95, size=n).astype(float)
    l2 = np.where(c1 == 1, rng.binomial(1, _expit(w + 1.5 * a1 - 0.7)), np.nan)
    a2 = np.where(c1 == 1, rng.binomial(1, _expit(np.nan_to_num(l2) - 0.5)), np.nan)
    c2 = np.where(c1 == 1, rng.binomial(1, 0.95, size=n), np.nan)
    y = np.where(c2 == 1, w + np.nan_to_num(l2) + np.nan_to_num(a2) + rng.normal(size=n), np.nan)
    return pd.DataFrame({"W": w, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})


def _rule(history: Any) -> Any:
    return (np.asarray(history["L2"]) > 0).astype(int)


def _coin(frame: Any) -> Any:
    return np.tile([0.7, 0.3], (len(frame), 1))


RULE = DynamicRegimen("rule", (1, _rule), rule_kind="known")
COIN = DynamicRegimen("coin", (1, Stochastic(_coin, "coin", density_kind="known")))


def _ltmle(plans: Any) -> LTMLE:
    return LTMLE(
        plans,
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        censoring_learner=LogisticRegression(),
        n_folds=1,
        simultaneous=False,
        random_state=3,
    )


@pytest.fixture(scope="module")
def binary() -> Any:
    plans = {"always": 1, "never": 0, "rule": RULE, "coin": COIN}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle(plans).fit(
            _binary_panel(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


@pytest.fixture(scope="module")
def law(binary: Any) -> LongitudinalPlasmodeLaw:
    return LongitudinalPlasmodeLaw.fit(binary)


def _run(result: Any, estimand: str, draws: int = 400, process: Any = PROCESS) -> Any:
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


class TestTruthByHand:
    """The enumerated truth equals the g-formula worked by hand from the frozen models."""

    def _parts(self, law: LongitudinalPlasmodeLaw) -> tuple[np.ndarray, Any, Any]:
        w = np.asarray(law.support.baseline, dtype=float)[:, 0]
        f = PROCESS.adjustment_scale * (w - w.mean()) / w.std()
        model = law.covariates[1][0]
        centre, scale = law.history_centre[0], law.history_scale[0]

        def risk(a1: Any) -> np.ndarray:
            # P(L2 = 1 | W, A1 = a1): the node-2 design is [W, A1].
            design = np.column_stack([w, np.broadcast_to(a1, w.shape)])
            return np.asarray(model.probability(design))

        def history(p: np.ndarray) -> np.ndarray:
            return HISTORY * (p * (1.0 - centre) + (1.0 - p) * (0.0 - centre)) / scale

        return f, risk, history

    def test_the_law_is_binary_so_the_truth_enumerates(self, law: Any) -> None:
        assert law.binary and law.enumerable

    def test_a_law_past_the_branch_limit_samples_its_truth(
        self, binary: Any, law: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # One L column and two nodes have 2**3 branches per support point.
        monkeypatch.setattr(plasmode_module, "MAX_BRANCHES", 4)
        assert law.binary and not law.enumerable
        with pytest.raises(ValueError, match="more branches"):
            law.truth(PROCESS, RULE, draws=None)
        target = _longitudinal_plasmode_target(binary, "ey_regimen[rule]", PROCESS, "plasmode")
        assert target.draws is not None and target.monte_carlo_se > 0.0

    @pytest.mark.parametrize("arms", [(1, 1), (0, 0), (1, 0)])
    def test_a_static_regimen(self, law: Any, arms: tuple[int, int]) -> None:
        f, risk, history = self._parts(law)
        expected = np.mean(f + EFFECT * sum(arms) + history(risk(arms[0])))
        value, error = law.truth(PROCESS, Regimen("r", arms), draws=None)
        assert value == pytest.approx(float(expected), abs=1e-12) and error == 0.0

    def test_the_naive_truth_is_the_mutation_control(self, binary: Any, law: Any) -> None:
        # Holding L2 at its law under the observed A1 drops the path A1 -> L2 -> Y.
        f, risk, history = self._parts(law)
        observed = np.nan_to_num(np.asarray(binary.data.treatment[:, 0], dtype=float))
        naive = float(np.mean(f + 2.0 * EFFECT + history(risk(observed))))
        value, _ = law.truth(PROCESS, Regimen("always", (1, 1)), draws=None)
        assert abs(value - naive) > 0.05

    def test_a_dynamic_rule(self, law: Any) -> None:
        f, risk, history = self._parts(law)
        p = risk(1.0)
        # Treat at node 1, then treat at node 2 exactly when L2 = 1.
        expected = np.mean(f + EFFECT * (1.0 + p) + history(p))
        value, _ = law.truth(PROCESS, RULE, draws=None)
        assert value == pytest.approx(float(expected), abs=1e-12)

    def test_a_stochastic_node(self, law: Any) -> None:
        f, risk, history = self._parts(law)
        expected = np.mean(f + EFFECT * 1.3 + history(risk(1.0)))
        value, _ = law.truth(PROCESS, COIN, draws=None)
        assert value == pytest.approx(float(expected), abs=1e-12)

    def test_the_monte_carlo_truth_agrees_with_the_enumeration(self, law: Any) -> None:
        exact, _ = law.truth(PROCESS, RULE, draws=None)
        sampled, error = law.truth(PROCESS, RULE, draws=512, seed=4)
        assert error > 0.0 and abs(sampled - exact) < 4.0 * error


class TestTheDraw:
    def test_a_drawn_panel_masks_every_exit(self, law: Any) -> None:
        drawn = law.draw(PROCESS, seed=5, name="plasmode")
        kept = drawn.uncensored
        assert np.isnan(drawn.treatment[~kept[:, 0], 1]).all()
        assert np.isnan(drawn.time_varying[1][~kept[:, 0], 0]).all()
        assert np.isnan(drawn.outcome[~kept[:, 1]]).all()
        assert np.isfinite(drawn.outcome[kept[:, 1]]).all()
        # The baselines are support rows, and the drawn L2 is binary.
        support = set(np.asarray(law.support.baseline)[:, 0])
        assert set(np.asarray(drawn.baseline)[:, 0]) <= support
        assert set(np.unique(drawn.time_varying[1][kept[:, 0], 0])) <= {0.0, 1.0}

    def test_the_drawn_treatment_follows_the_frozen_mechanism(self, law: Any) -> None:
        drawn = law.draw(PROCESS, seed=6, name="plasmode")
        baseline = np.asarray(drawn.baseline, dtype=float)
        expected = np.clip(law.treatment[0].probability(baseline), law.lower, 1.0 - law.lower)
        assert abs(np.mean(drawn.treatment[:, 0]) - np.mean(expected)) < 0.06


class TestEndToEnd:
    def test_a_static_contrast_on_the_binary_panel(self, binary: Any, law: Any) -> None:
        test = _run(binary, "ate_regimen[never vs always]")
        never, _ = law.truth(PROCESS, Regimen("never", (0, 0)), draws=None)
        always, _ = law.truth(PROCESS, Regimen("always", (1, 1)), draws=None)
        assert test.declared_effect == pytest.approx(never - always, abs=1e-12)
        assert test.passed is True and len(test.readings) == 3
        assert "enumerates every branch" in test.scope_sentence

    def test_a_dynamic_rule_on_the_binary_panel(self, binary: Any) -> None:
        test = _run(binary, "ey_regimen[rule]")
        assert test.passed is True

    def test_a_continuous_history_reads_a_monte_carlo_truth(self) -> None:
        result = _gaussian_history_fit(_gaussian_history_panel())
        test = _run(result, "ate_regimen[never vs always]")
        assert test.passed is True
        assert "simulated histories per support point" in test.scope_sentence
        assert "in quadrature" in test.scope_sentence

    def test_a_capped_monte_carlo_truth_widens_the_bias_reading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _gaussian_history_fit(_gaussian_history_panel())
        estimand = "ate_regimen[never vs always]"
        monkeypatch.setattr(plasmode_module, "MAX_DRAWS", 4)
        target = _longitudinal_plasmode_target(result, estimand, PROCESS, "plasmode")
        share = target.monte_carlo_se / float(result[estimand].plugin_std_error)
        assert target.draws == 4 and share > plasmode_module.MONTE_CARLO_SHARE
        capped = _run(result, estimand, draws=40)
        monkeypatch.setattr(plasmode_module, "MAX_DRAWS", 2048)
        sized = _run(result, estimand, draws=40)
        width = {
            name: next(r.upper - r.lower for r in test.readings if r.name == "bias")
            for name, test in (("capped", capped), ("sized", sized))
        }
        assert width["capped"] > width["sized"]


class _Overridden:
    """Forward to an object, with some attributes replaced."""

    def __init__(self, wrapped: Any, **overrides: Any) -> None:
        self._wrapped = wrapped
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class TestRefusals:
    @pytest.mark.parametrize(
        ("override", "structure"),
        [
            ({"repeated": np.zeros((1, 1))}, "a repeated outcome"),
            ({"treatment_held": True}, "a held baseline decision"),
            ({"n_decisions": 1}, "a node without a treatment decision"),
            ({"known_mechanisms": object()}, "a declared known mechanism"),
            ({"cluster": np.zeros(1)}, "a clustered panel"),
            ({"is_weighted": True}, "a weighted panel"),
            ({"treatment_levels": ((0, 1, 2), (0, 1))}, "a non-binary treatment node"),
            ({"has_mixed_node": True}, "a non-binary treatment node"),
        ],
    )
    def test_a_structure_the_draw_does_not_cover(
        self, binary: Any, override: dict[str, Any], structure: str
    ) -> None:
        result = _Overridden(binary, data=_Overridden(binary.data, **override))
        with pytest.raises(CapabilityError, match=structure) as caught:
            _longitudinal_plasmode_target(
                result, "ate_regimen[never vs always]", PROCESS, "plasmode"
            )
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    def test_a_survival_fit(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _ltmle({"always": 1, "never": 0}).fit(
                _survival_panel(),
                outcome=["Y1", "Y2"],
                treatment=["A1", "A2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
                censoring=["C1", "C2"],
            )
        estimand = next(key for key in result.estimates if key.startswith("ate"))
        with pytest.raises(CapabilityError, match="a survival or competing-risk fit"):
            _run(result, estimand, draws=10, process=HAZARD)

    def test_a_history_effect_in_the_held_design(self, binary: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CapabilityError, match="no held-history truth"):
                refute(
                    binary,
                    estimand="ate_regimen[never vs always]",
                    tests=("simulated_outcome",),
                    n_replicates=10,
                    simulated_outcome=PROCESS,
                )

    def test_a_history_effect_on_a_point_fit(self) -> None:
        rng = np.random.default_rng(1)
        w = rng.normal(size=200)
        frame = pd.DataFrame(
            {"W": w, "A": rng.binomial(1, _expit(w)), "Y": w + rng.normal(size=200)}
        )
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = study.identify(ATE()).estimate(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(),
                cross_fit=False,
                random_state=1,
            )
            with pytest.raises(CapabilityError, match="history_effect"):
                refute(
                    result,
                    estimand="ate",
                    tests=("simulated_outcome",),
                    n_replicates=10,
                    simulated_outcome=PROCESS,
                )
