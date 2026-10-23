"""The longitudinal plasmode of a repeated outcome (X82 part E4c).

The law draws ``Y_t`` at every node from the declared Gaussian process, and that draw opens the
history of node ``t + 1`` as the container's copy of ``Y_t``. The outcome at node ``t`` reads
the time-varying columns of nodes 1 to ``t`` through ``history_effect``. On the two-node panel
below, ``L_2`` is Gaussian linear in ``(W, A_1, Y_1)``, so ``E[Y_2]`` under a static regimen has
a closed form, and the Monte Carlo truth must sit on it. Holding ``Y_1`` at its observed value
(the naive truth) drops the path ``A_1 -> Y_1 -> Y_2`` and is the mutation control.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from cleverly.longitudinal import Regimen
from cleverly.validation import GaussianAdjustmentOutcome, refute
from cleverly.validation._longitudinal_plasmode import LongitudinalPlasmodeLaw
from cleverly.validation.outcome_processes import standardized_sum
from cleverly.validation.refute import _longitudinal_plasmode_target
from tests.unit.test_longitudinal_generated_outcomes import _fit, _panel

PROCESS = GaussianAdjustmentOutcome(effect=0.7, history_effect=0.5)
CONTRAST = "ate_regimen[never vs always @ t=2]"


@pytest.fixture(scope="module")
def repeated() -> Any:
    return _fit(_panel(), repeated=True)


@pytest.fixture(scope="module")
def law(repeated: Any) -> LongitudinalPlasmodeLaw:
    return LongitudinalPlasmodeLaw.fit(repeated)


def _second(law: Any, a1: float, a2: float, y1: Any = None) -> float:
    """``E[Y_2]`` by hand. ``y1`` replaces the mean of ``Y_1`` under ``a1`` (the mutation)."""
    w = np.asarray(law.support.baseline, dtype=float)
    f = PROCESS.adjustment(w, centre=law.centre, scale=law.scale)
    first = PROCESS.noise.mean + f + PROCESS.effect * a1
    copy = first if y1 is None else y1
    model = law.covariates[1][1]
    l2 = model.mean(np.column_stack([w[:, 0], np.full(w.shape[0], a1), copy]))
    history = standardized_sum(
        np.column_stack([copy, l2]),
        PROCESS.history_effect,
        centre=law.history_centre,
        scale=law.history_scale,
    )
    return float(np.mean(PROCESS.noise.mean + f + PROCESS.effect * (a1 + a2) + history))


class TestTheTruth:
    def test_the_copy_is_the_previous_draw(self, law: Any) -> None:
        assert type(law.covariates[1][0]).__name__ == "_OutcomeCopy"
        assert not law.binary and not law.enumerable

    @pytest.mark.parametrize("arms", [(1, 1), (0, 0), (1, 0)])
    def test_the_second_node_has_its_closed_form(self, law: Any, arms: tuple[int, int]) -> None:
        regimen = Regimen("r", arms)
        value, error = law.truth(PROCESS, regimen, draws=1024, seed=2, horizon=2)
        assert error > 0.0
        assert abs(value - _second(law, *arms)) < 4.0 * error

    def test_the_first_node_reads_no_history(self, law: Any) -> None:
        w = np.asarray(law.support.baseline, dtype=float)
        f = PROCESS.adjustment(w, centre=law.centre, scale=law.scale)
        expected = float(np.mean(PROCESS.noise.mean + f + PROCESS.effect))
        value, error = law.truth(PROCESS, Regimen("always", (1, 1)), draws=64, horizon=1)
        assert value == pytest.approx(expected, abs=1e-12) and error < 1e-8

    def test_the_naive_truth_is_the_mutation_control(self, repeated: Any, law: Any) -> None:
        # Holding Y1 at its observed value drops the path A1 -> Y1 -> Y2.
        observed = np.asarray(repeated.data.repeated[:, 0], dtype=float)
        observed = np.where(np.isnan(observed), np.nanmean(observed), observed)
        value, error = law.truth(PROCESS, Regimen("always", (1, 1)), draws=1024, horizon=2)
        assert abs(value - _second(law, 1, 1, y1=observed)) > 0.02 + 4.0 * error


class TestTheDraw:
    def test_a_drawn_panel_carries_its_outcome_into_the_next_history(self, law: Any) -> None:
        drawn = law.draw(PROCESS, seed=3, name="plasmode")
        reached = drawn.uncensored[:, 0]
        np.testing.assert_array_equal(drawn.time_varying[1][reached, 0], drawn.repeated[reached, 0])
        # A unit censored at a node has no outcome there, and no later history.
        assert np.isnan(drawn.repeated[~drawn.uncensored]).all()
        assert np.isnan(drawn.time_varying[1][~reached]).all()
        np.testing.assert_array_equal(drawn.outcome, drawn.repeated[:, -1])


class TestRefits:
    def test_a_repeated_contrast_passes(self, repeated: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = refute(
                repeated,
                estimand=CONTRAST,
                tests=("plasmode",),
                n_replicates=400,
                random_state=1,
                simulated_outcome=PROCESS,
            )["plasmode"]
        target = _longitudinal_plasmode_target(repeated, CONTRAST, PROCESS, "plasmode")
        assert test.declared_effect == pytest.approx(target.truth)
        assert test.passed is True and len(test.readings) == 3
        assert "simulated histories per support point" in test.scope_sentence
