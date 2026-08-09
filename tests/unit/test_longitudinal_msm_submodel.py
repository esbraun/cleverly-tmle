"""What the pooled loss-weighted fluctuation is, at the node where it is built.

Array-level and exact: every claim here is about values a fit hands to
``solve_fluctuation``, so none of it needs a statistical argument or a tolerance. The
central claims pin both parts of the canonical decomposition: the working-model design
is the submodel covariate, while ``h / cumulative_g`` is a loss weight.  Their product is
the EIF numerator, but moving the inverse probability into the submodel changes the
finite-sample targeting path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_longitudinal
from cleverly.fluctuation.submodel import Submodel
from cleverly.learners.crossfit import make_folds
from cleverly.learners.super_learner import resolve_learner
from cleverly.longitudinal import LongitudinalData, resolve_plans, resolve_regimens
from cleverly.longitudinal import msm as longitudinal_msm
from cleverly.longitudinal.sequential import fit_mechanism, fit_regimen
from cleverly.msm import MSM
from cleverly.utils.bounds import OutcomeScaler

from .test_longitudinal_msm import DURATION

LABELS = ("always", "never", "early")
SPEC: dict[str, Any] = {"always": 1, "never": 0, "early": (1, 0)}
BOUNDS = (0.01, 0.99)

#: Large enough that every regimen's follower set carries both outcome classes at every
#: node -- the nuisance learners cross-fit internally and refuse a stratum of one. Nothing
#: here is statistical, so this is the smallest ``n`` that lets the arrays exist.
N = 600


def saturated_design(label: Any, horizon: int, frame: Any) -> np.ndarray:
    """One indicator column per regimen -- the model that must reduce to the plain fit."""
    del horizon
    return np.eye(len(LABELS))[LABELS.index(label)] * np.ones((len(frame), 1))


def flat_design(label: Any, horizon: int, frame: Any) -> np.ndarray:
    """A dose-response with **no effect modifier**: phi is constant down the rows."""
    del horizon
    n = len(frame)
    return np.column_stack([np.ones(n), np.full(n, DURATION[label])])


def setup() -> tuple[LongitudinalData, Any, Any, dict[str, Any]]:
    frame, _ = make_longitudinal(n=N, seed=0)
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    plans = resolve_plans(resolve_regimens(SPEC, data.n_times), data)
    folds = make_folds(data.n, n_folds=2, random_state=0)
    classifier = resolve_learner("glm", task="classification", n_folds=2, random_state=0)
    regressor = resolve_learner("glm", task="regression", n_folds=2, random_state=0)
    mechanism = fit_mechanism(
        data, plans, treatment_learner=classifier, censoring_learner=classifier, folds=folds
    )
    kwargs = {
        "outcome_learner": classifier,
        "pseudo_learner": regressor,
        "folds": folds,
        "scaler": OutcomeScaler.identity(),
        "g_bounds": BOUNDS,
    }
    return data, plans, mechanism, {"mechanism_kwargs": kwargs}


def calls_of(design: Any, terms: tuple[str, ...], monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every pooled fluctuation call, deepest node first."""
    data, plans, mechanism, extra = setup()
    seen: list[tuple[Submodel, np.ndarray, np.ndarray]] = []
    original = longitudinal_msm.solve_fluctuation

    def spy(
        outcome: Any,
        initial: Any,
        submodel: Submodel,
        weights: Any,
        mask: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        seen.append((submodel, np.asarray(weights), np.asarray(mask)))
        return original(outcome, initial, submodel, weights, mask, *args, **kwargs)

    monkeypatch.setattr(longitudinal_msm, "solve_fluctuation", spy)
    model = longitudinal_msm.evaluate_regimen_msm(
        MSM(design=design, terms=terms), data, plans, (data.n_times,)
    )
    longitudinal_msm.fit_regimens_msm(data, plans, mechanism, model, **extra["mechanism_kwargs"])
    return data, plans, mechanism, extra, seen


def submodels_of(design: Any, terms: tuple[str, ...], monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every ``Submodel`` a pooled fit hands to the fluctuation, deepest node first."""
    data, plans, mechanism, extra, seen = calls_of(design, terms, monkeypatch)
    return data, plans, mechanism, extra, [submodel for submodel, _, _ in seen]


class TestPoolingIsWhatGivesTheDesignItsRank:
    def test_a_per_cell_design_with_no_modifier_is_rank_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole reason the fluctuation is pooled across the cells.

        With no effect modifier ``phi(c, V)`` is constant down the rows, so a cell's
        block has rank one, whatever ``p`` is, and its ``p`` score equations collapse
        into one. Stacking cells with distinct ``phi`` separates them again.
        """
        _, _, _, _, seen = submodels_of(flat_design, ("(intercept)", "duration"), monkeypatch)
        assert seen, "the pooled fit fluctuated at no node"
        for submodel in seen:
            rows = submodel.observed.shape[0] // len(LABELS)
            blocks = np.split(submodel.observed, len(LABELS))
            for block in blocks:
                assert block.shape == (rows, 2)
                assert np.linalg.matrix_rank(block, tol=1e-12) == 1
            assert np.linalg.matrix_rank(submodel.observed, tol=1e-12) == 2

    def test_the_pooled_design_has_a_column_per_term(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, _, _, seen = submodels_of(flat_design, ("(intercept)", "duration"), monkeypatch)
        for submodel in seen:
            assert submodel.dim == 2
            assert submodel.group == "sequential"
            assert submodel.names[0].startswith("epsilon[(intercept)")


class TestASaturatedModelIsBlockDiagonal:
    def test_the_off_diagonal_blocks_are_exactly_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Which is what makes each cell's epsilon its own, and the report reduce."""
        _, _, _, _, seen = submodels_of(saturated_design, LABELS, monkeypatch)
        for submodel in seen:
            blocks = np.split(submodel.observed, len(LABELS))
            for row, block in enumerate(blocks):
                for column in range(len(LABELS)):
                    if column != row:
                        np.testing.assert_array_equal(block[:, column], np.zeros(block.shape[0]))

    def test_each_diagonal_block_is_the_plain_recursion_s_design(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The inverse probability belongs to the loss weight, not this design."""
        data, plans, mechanism, extra, seen = calls_of(saturated_design, LABELS, monkeypatch)
        plain = {
            plan.label: fit_regimen(data, plan, mechanism, **extra["mechanism_kwargs"])
            for plan in plans
        }
        # ``seen`` is deepest node first; a step is ascending, so index from the end.
        for depth, (submodel, weights, mask) in enumerate(seen):
            time = data.n_times - depth
            blocks = np.split(submodel.observed, len(LABELS))
            weight_blocks = np.split(weights, len(LABELS))
            mask_blocks = np.split(mask, len(LABELS))
            for index, label in enumerate(LABELS):
                step = plain[label].steps[time - 1]
                assert step.time == time
                np.testing.assert_array_equal(blocks[index][:, index], np.ones(data.n))
                np.testing.assert_array_equal(mask_blocks[index], step.trained_on)
                np.testing.assert_array_equal(
                    weight_blocks[index][step.trained_on],
                    data.weights[step.trained_on] * step.clever[step.trained_on],
                )


class TestTheLossWeightsMultiply:
    def test_the_projection_weight_multiplies_the_loss_weight_not_the_design(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doubling ``h`` doubles the canonical loss weight and leaves phi unchanged."""
        data, plans, mechanism, extra = setup()
        seen: list[tuple[Submodel, np.ndarray]] = []
        original = longitudinal_msm.solve_fluctuation

        def spy(outcome: Any, initial: Any, submodel: Submodel, weights: Any, *a: Any, **k: Any):
            seen.append((submodel, np.asarray(weights)))
            return original(outcome, initial, submodel, weights, *a, **k)

        monkeypatch.setattr(longitudinal_msm, "solve_fluctuation", spy)
        doubled = MSM(
            design=flat_design,
            terms=("(intercept)", "duration"),
            weights=lambda label, horizon, w: np.full(len(w), 2.0),
        )
        plain_msm = MSM(design=flat_design, terms=("(intercept)", "duration"))
        model = longitudinal_msm.evaluate_regimen_msm(plain_msm, data, plans, (data.n_times,))
        longitudinal_msm.fit_regimens_msm(
            data, plans, mechanism, model, **extra["mechanism_kwargs"]
        )
        base = [(submodel.observed.copy(), weights.copy()) for submodel, weights in seen]

        seen.clear()
        model = longitudinal_msm.evaluate_regimen_msm(doubled, data, plans, (data.n_times,))
        longitudinal_msm.fit_regimens_msm(
            data, plans, mechanism, model, **extra["mechanism_kwargs"]
        )

        for (covariate, weights), (base_covariate, base_weights) in zip(seen, base, strict=True):
            np.testing.assert_array_equal(covariate.observed, base_covariate)
            np.testing.assert_allclose(weights, 2.0 * base_weights)
