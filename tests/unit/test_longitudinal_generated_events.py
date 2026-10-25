"""Generated events on the held history of a survival fit (X82 part C3b).

The draw keeps ``(W, L, A, C)``, recodes each original event at node ``t`` as censoring at
``t``, and draws new events from a declared hazard that reads ``W`` and the treatments. The
exact witness enumerates a finite law whose original event reads ``L``. The g-formula of the
recoded, censored law equals the closed-form risk with no censoring. Its mutation control
lets the generated hazard read ``L_2``, and the two values part. The other tests pin the
draw, the per-cause probabilities, the sample-conditional truths and the refusals.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.validation import LogisticHazardOutcome, MultinomialHazardOutcome, refute
from cleverly.validation.refute import (
    GaussianAdjustmentOutcome,
    _longitudinal_generated_target,
    _recoded_uncensored,
    refute_refusal,
)
from tests.unit import _longitudinal_repeats_support as competing_support
from tests.unit.test_longitudinal_refutation import _ltmle, _survival_panel

HAZARD = LogisticHazardOutcome(intercept=-1.5, effect=0.8, adjustment_scale=0.5)
INDEPENDENT = LogisticHazardOutcome(effect=0.0, adjustment_scale=0.0)
CONTRAST = "ate_regimen[never vs always @ t=2]"


def _expit(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


# ------------------------------------------------------------------- the exact witness

Hazard = Callable[[int, int, int, int], float]


def _bernoulli(probability: float, value: int) -> float:
    return probability if value else 1.0 - probability


def _enumerated_law(hazard: Hazard) -> list[dict[str, Any]]:
    """Every path of a two-node survival law, recoded and with generated events.

    The original event reads ``W``, ``A`` and ``L2``. A recoded unit is censored at the node
    of its original event. ``hazard(node, w, dose, l2)`` gives the generated event.
    """
    paths: list[dict[str, Any]] = []
    for w, a1, c1, y1, l2, a2, c2, y2 in itertools.product((0, 1), repeat=8):
        p = 0.5 * _bernoulli(0.3 + 0.4 * w, a1) * _bernoulli(0.85, c1)
        if not c1:
            if (y1, l2, a2, c2, y2) != (0, 0, 0, 0, 0):
                continue
            kept, present2 = (False, False), False
        else:
            p *= _bernoulli(0.1 + 0.2 * w + 0.15 * a1, y1)
            if y1:
                if (l2, a2, c2, y2) != (0, 0, 0, 0):
                    continue
                kept, present2 = (False, False), False
            else:
                p *= _bernoulli(0.3 + 0.4 * a1, l2) * _bernoulli(0.2 + 0.5 * l2, a2)
                p *= _bernoulli(0.9, c2)
                if c2:
                    p *= _bernoulli(0.1 + 0.3 * l2 + 0.2 * a2, y2)
                elif y2:
                    continue
                kept, present2 = (True, bool(c2) and not y2), True
        for g1, g2 in itertools.product((0, 1), repeat=2):
            q = p
            if kept[0]:
                q *= _bernoulli(hazard(1, w, a1, l2), g1)
            elif g1:
                continue
            observed2 = kept[0] and not g1 and present2 and kept[1]
            if observed2:
                q *= _bernoulli(hazard(2, w, a1 + a2, l2), g2)
            elif g2:
                continue
            if q > 0.0:
                paths.append(
                    {
                        "p": q,
                        "w": w,
                        "a1": a1,
                        "l2": l2 if kept[0] else None,
                        "a2": a2,
                        "kept1": kept[0],
                        "observed2": observed2,
                        "g1": g1,
                        "g2": g2,
                    }
                )
    return paths


def _recoded_gformula(paths: list[dict[str, Any]], regimen: tuple[int, int]) -> float:
    """The iterated conditional expectation of the generated risk on the recoded law."""

    def mean(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]) -> float:
        total = sum(row["p"] for row in rows)
        return sum(row["p"] * value(row) for row in rows) / total

    second = {}
    for w, l2 in itertools.product((0, 1), repeat=2):
        cell = [
            row
            for row in paths
            if row["observed2"]
            and (row["w"], row["a1"], row["l2"], row["a2"]) == (w, regimen[0], l2, regimen[1])
        ]
        second[w, l2] = mean(cell, lambda row: row["g2"])
    risk = 0.0
    for w in (0, 1):
        weight = sum(row["p"] for row in paths if row["w"] == w)
        cell = [row for row in paths if row["w"] == w and row["kept1"] and row["a1"] == regimen[0]]
        first = mean(cell, lambda row: 1.0 if row["g1"] else second[row["w"], row["l2"]])
        risk += weight * first
    return risk


def _closed_form(hazard: Hazard, regimen: tuple[int, int]) -> float:
    """The uncensored risk ``1 - mean_W (1 - h_1)(1 - h_2)``, reading no ``L2``."""
    survive = [
        (1.0 - hazard(1, w, regimen[0], 0)) * (1.0 - hazard(2, w, sum(regimen), 0)) for w in (0, 1)
    ]
    return 1.0 - 0.5 * sum(survive)


def _hazard(node: int, w: int, dose: int, l2: int) -> float:
    return _expit(-1.5 + 0.7 * w + 0.8 * dose)


def _leaky(node: int, w: int, dose: int, l2: int) -> float:
    return _expit(-1.5 + 0.7 * w + 0.8 * dose + (1.2 * l2 if node == 2 else 0.0))


class TestTheRecodeCannotLeak:
    @pytest.mark.parametrize("regimen", [(0, 0), (1, 1), (1, 0)])
    def test_the_recoded_gformula_is_the_uncensored_risk(self, regimen: tuple[int, int]) -> None:
        paths = _enumerated_law(_hazard)
        assert sum(row["p"] for row in paths) == pytest.approx(1.0, abs=1e-12)
        assert _recoded_gformula(paths, regimen) == pytest.approx(
            _closed_form(_hazard, regimen), abs=1e-12
        )

    def test_a_hazard_that_reads_the_covariate_breaks_the_closed_form(self) -> None:
        # The mutation: the generated event reads L2, which the original event and the
        # recoded censoring read too. The closed form that reads no L2 is then wrong.
        paths = _enumerated_law(_leaky)
        gap = _recoded_gformula(paths, (1, 1)) - _closed_form(_leaky, (1, 1))
        assert abs(gap) > 1e-2


# ------------------------------------------------------------------- the processes


def test_the_cause_probabilities_and_no_event_sum_to_one() -> None:
    process = MultinomialHazardOutcome(intercepts=(2.0, 3.0, -1.0), effects=(5.0, 4.0, 0.0))
    shift = np.linspace(-30.0, 30.0, 41)
    for dose in (0.0, 1.0, 3.0):
        probabilities = process.probabilities(shift, np.full(shift.size, dose))
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities.sum(axis=1) <= 1.0 + 1e-12)
    # Large linear predictors still give a valid law: per-cause logistic hazards would sum
    # above 1 here.
    top = process.probabilities(np.array([10.0]), np.array([3.0]))
    assert top.sum() == pytest.approx(1.0, abs=1e-6)
    assert (
        np.sum([_expit(10.0 + c + 3.0 * e) for c, e in zip((2, 3, -1), (5, 4, 0), strict=True)])
        > 1.0
    )


def test_a_declaration_refuses_a_cause_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one intercept and one effect per cause"):
        MultinomialHazardOutcome(intercepts=(1.0,), effects=(0.0, 0.0))


# ------------------------------------------------------------------- the draw


@pytest.fixture(scope="module")
def fitted() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle(n_folds=1).fit(
            _survival_panel(n=800),
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


class TestTheDraw:
    def test_an_original_event_censors_its_node(self, fitted: Any) -> None:
        data = fitted.data
        kept = _recoded_uncensored(data)
        first = data.event.copy()
        first[:, 1:] &= ~data.event[:, :-1]
        for node in range(data.n_times):
            assert not kept[first[:, node], node].any()
        # A unit with no original event keeps its censoring.
        free = ~data.event[:, -1]
        np.testing.assert_array_equal(kept[free], data.uncensored[free])

    def test_the_drawn_panel_masks_every_exit(self, fitted: Any) -> None:
        target = _longitudinal_generated_target(fitted, CONTRAST, HAZARD, "simulated_outcome")
        drawn = target.draw(fitted.data, seed=3, name="simulated_outcome")
        kept = _recoded_uncensored(fitted.data)
        # The generated event is seen only where the recoded panel observes the unit.
        assert not (drawn.event[:, 0] & ~kept[:, 0]).any()
        assert drawn.uncensored[:, 1][drawn.event[:, 0]].sum() == 0
        assert np.isnan(drawn.treatment[drawn.event[:, 0], 1]).all()
        # A unit censored at its original event keeps that node's treatment.
        first = fitted.data.event[:, 0]
        assert not np.isnan(drawn.treatment[first, 0]).any()
        np.testing.assert_array_equal(drawn.outcome, drawn.event[:, -1].astype(float))

    def test_the_node_one_events_follow_the_hazard(self, fitted: Any) -> None:
        data = fitted.data
        target = _longitudinal_generated_target(fitted, CONTRAST, HAZARD, "simulated_outcome")
        kept = _recoded_uncensored(data)[:, 0]
        from cleverly.validation.outcome_processes import standardized_sum

        shift = standardized_sum(data.baseline, HAZARD.adjustment_scale)
        expected = HAZARD.probabilities(shift, np.nan_to_num(data.treatment[:, 0]))[:, 0]
        rates = np.mean(
            [
                target.draw(data, seed=seed, name="simulated_outcome").event[kept, 0]
                for seed in range(200)
            ],
            axis=0,
        )
        assert np.mean(np.abs(rates - expected[kept])) < 0.05
        assert abs(float(np.mean(rates - expected[kept]))) < 0.01


# ------------------------------------------------------------------- truths and refits


class TestTheTruth:
    def test_the_independent_hazard_has_the_null(self, fitted: Any) -> None:
        target = _longitudinal_generated_target(fitted, CONTRAST, INDEPENDENT, "dummy_outcome")
        assert target.truth == 0.0 and target.covariate_part

    def test_a_static_contrast_has_its_closed_form(self, fitted: Any) -> None:
        from cleverly.validation.outcome_processes import standardized_sum

        shift = standardized_sum(fitted.data.baseline, HAZARD.adjustment_scale)

        def risk(doses: tuple[float, float]) -> float:
            h1 = HAZARD.probabilities(shift, np.full(shift.size, doses[0]))[:, 0]
            h2 = HAZARD.probabilities(shift, np.full(shift.size, doses[1]))[:, 0]
            return float(np.mean(1.0 - (1.0 - h1) * (1.0 - h2)))

        target = _longitudinal_generated_target(fitted, CONTRAST, HAZARD, "simulated_outcome")
        assert target.truth == pytest.approx(risk((0.0, 0.0)) - risk((1.0, 2.0)))
        assert target.truth < -0.1
        assert not target.covariate_part

    def test_the_refits_sit_on_the_truth(self, fitted: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = refute(
                fitted,
                estimand=CONTRAST,
                tests=("simulated_outcome",),
                n_replicates=40,
                random_state=1,
                simulated_outcome=HAZARD,
            )["simulated_outcome"]
        values = np.asarray(test.values)
        error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        assert abs(float(np.mean(values)) - test.declared_effect) < 4.0 * error
        assert [reading.name for reading in test.readings] == ["bias"]
        assert "uncensored share by node" in test.detail
        assert "censors the unit at t" in test.scope_sentence
        assert test.passed is None
        # No study row covers a survival fit with several nodes, so the evidence must not
        # promise a verdict at the calibrated count.
        assert "no simulation study backs a verdict for this kind of fit" in test.evidence

    def test_a_competing_risk_contrast_runs(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = competing_support.competing(competing_support.competing_frame(), n_folds=1)
            process = MultinomialHazardOutcome(
                intercepts=(-1.5, -2.0), effects=(0.8, -0.3), adjustment_scale=0.5
            )
            estimand = "ate_regimen[always vs never, relapse @ t=2]"
            target = _longitudinal_generated_target(result, estimand, process, "simulated_outcome")
            test = refute(
                result,
                estimand=estimand,
                tests=("simulated_outcome",),
                n_replicates=10,
                random_state=1,
                simulated_outcome=process,
            )["simulated_outcome"]
        assert test.declared_effect == pytest.approx(target.truth)
        assert target.truth > 0.1
        drawn = target.draw(result.data, seed=2, name="simulated_outcome")
        np.testing.assert_array_equal(drawn.cause_event.any(axis=2), drawn.event)


# ------------------------------------------------------------------- refusals


def _refused(result: Any, name: str, **request: Any) -> CapabilityError:
    arguments = {"tests": (name,), "n_replicates": 10, **request}
    refusal = refute_refusal(result, estimand=CONTRAST, **arguments)
    assert refusal is not None
    with pytest.raises(CapabilityError) as error:
        refute(result, estimand=CONTRAST, **arguments)
    assert error.value.refusal.message() == refusal.message()
    return error.value


class TestRefusals:
    def test_a_gaussian_declaration_is_refused(self, fitted: Any) -> None:
        error = _refused(fitted, "simulated_outcome", simulated_outcome=GaussianAdjustmentOutcome())
        assert "LogisticHazardOutcome" in error.refusal.message()

    def test_the_default_dummy_declaration_is_refused(self, fitted: Any) -> None:
        error = _refused(fitted, "dummy_outcome")
        assert error.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_dummy_hazard_must_read_nothing(self, fitted: Any) -> None:
        error = _refused(fitted, "dummy_outcome", dummy_outcome=HAZARD)
        assert "effect 0" in error.refusal.message()
