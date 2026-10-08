"""Targeting, cross-fitting, working models and mutations at longitudinal policy nodes.

The exact-law equalities in ``tests/unit/test_influence_gateaux_longitudinal_mtp.py`` hold
with every fluctuation coefficient at zero, so they cannot see the targeting step.  Here a
misspecified initial regression makes the coefficients nonzero (T5b), the pooled
cross-fitted construction is checked at five folds (T6), a working model over policy cells
is checked against the fixed linear map of the cell means (T15) and against the per-cell
fits it must reproduce when saturated (T16), and one mutation per policy-specific term
must move the estimate or the curve by more than ``1e-4`` (M4 to M7, M9 to M11).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from cleverly.longitudinal import LTMLE, msm, sequential
from cleverly.longitudinal import regimen as regimen_module
from cleverly.msm import MSM

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_mtp as law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import longitudinal_mtp as mtp
from ..studies.canonical_ltmle import QuasiBinomialGLM

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}
ROWS = law.first_row_of()
CAUGHT = 1e-4
CELLS = ("natural", "up at 2", "up")
DOSE = {"natural": 0.0, "up at 2": 0.5, "up": 1.0}
WEIGHT = {"natural": 1.0, "up at 2": 2.0, "up": 1.5}


def _exact(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": multivalue.CellProbabilities(),
        "n_folds": 1,
        "g_bounds": (1e-8, 1.0),
        "simultaneous": False,
        "random_state": 2,
        **overrides,
    }


def _misspecified(**overrides: Any) -> dict[str, Any]:
    return _exact(outcome_learner=QuasiBinomialGLM(), pseudo_learner=DummyRegressor(), **overrides)


def _fit(frame: Any, labels: tuple[str, ...], model: MSM | None = None, **settings: Any) -> Any:
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        if model is None:
            return LTMLE(mtp.regimens(labels), reference=labels[0], **settings).fit(
                frame, **COLUMNS
            )
        return LTMLE(mtp.regimens(labels), msm=model, **settings).fit(frame, **COLUMNS)


def _trend() -> MSM:
    return MSM(
        design=lambda label, horizon, w: np.column_stack(
            [np.ones(len(w)), np.full(len(w), DOSE[label])]
        ),
        terms=("intercept", "dose"),
        design_kind="known",
        weights=lambda label, horizon, w: np.full(len(w), WEIGHT[label]),
        weights_kind="known",
    )


def _saturated() -> MSM:
    return MSM(
        design=lambda label, horizon, w: np.column_stack(
            [np.full(len(w), float(label == item)) for item in CELLS]
        ),
        terms=CELLS,
        design_kind="known",
    )


# ------------------------------------------------------------------ T5b, T6


@pytest.mark.parametrize("folds", [1, 5])
def test_t5b_t6_a_misspecified_fit_targets_and_solves_its_score(folds: int) -> None:
    """Nonzero coefficients at every node, and ``P_n D* = 0`` in sample and pooled."""
    frame = law.sample(900, 11)
    result = _fit(frame, ("natural", "up", "random"), **_misspecified(n_folds=folds))
    for label in ("up", "random"):
        steps = result.fits[label].steps
        assert all(abs(float(step.fluctuation.epsilon[0])) > 0.01 for step in steps), label
        curve = result.influence_curves[f"ey_regimen[{label}]"]
        assert abs(float(np.mean(curve))) < 1e-6, label
        assert all(step.fluctuation.converged for step in steps), label


def test_t5b_the_policy_dose_prediction_moves_by_the_node_coefficient() -> None:
    """The targeted prediction at the policy dose is expit(logit(initial) + epsilon)."""
    from cleverly.utils.bounds import bound, expit, logit, shrink_probabilities

    result = _fit(law.sample(900, 11), ("natural", "up"), **_misspecified())
    for step in result.fits["up"].steps:
        epsilon = float(step.fluctuation.epsilon[0])
        moved = bound(
            expit(logit(shrink_probabilities(step.initial_by_arm, 0.9995)) + epsilon),
            1.0 - 0.9995,
            0.9995,
        )
        rows = step.at_risk
        np.testing.assert_allclose(step.targeted_by_arm[rows], moved[rows], atol=1e-12)


# ------------------------------------------------------------------ T15, T16


def test_t15_a_projection_over_policy_cells_is_the_projection_of_the_cell_means() -> None:
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        result = LTMLE(mtp.regimens(CELLS), msm=_trend(), **_exact()).fit(law.frame(), **COLUMNS)
    design = np.array([[1.0, DOSE[label]] for label in CELLS])
    weights = np.array([WEIGHT[label] for label in CELLS])
    mapping = np.linalg.solve(design.T @ (weights[:, None] * design), design.T * weights)
    means = np.array([law.truth(f"ey_regimen[{label}]") for label in CELLS])
    eifs = np.array([law.eif(f"ey_regimen[{label}]") for label in CELLS])
    for index, term in enumerate(("intercept", "dose")):
        name = f"msm_regimen[{term}]"
        assert result.psi(name) == pytest.approx(float(mapping[index] @ means), abs=1e-12)
        np.testing.assert_allclose(
            result.influence_curves[name][ROWS], mapping[index] @ eifs, atol=1e-10, rtol=0
        )
    assert abs(float(mapping[1] @ means)) > 0.01


@pytest.mark.parametrize("folds", [1, 5])
def test_t16_a_saturated_projection_reproduces_the_policy_fits(folds: int) -> None:
    frame = law.sample(900, 7)
    projected = _fit(frame, CELLS, _saturated(), **_misspecified(n_folds=folds))
    separate = _fit(frame, CELLS, **_misspecified(n_folds=folds))
    for label in CELLS:
        assert projected.psi(f"msm_regimen[{label}]") == pytest.approx(
            separate.psi(f"ey_regimen[{label}]"), abs=1e-10
        )
        np.testing.assert_allclose(
            projected.influence_curves[f"msm_regimen[{label}]"],
            separate.influence_curves[f"ey_regimen[{label}]"],
            atol=1e-10,
        )


@pytest.mark.parametrize("folds", [1, 5])
def test_m11_an_initial_carry_breaks_the_saturated_reproduction(
    folds: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = law.sample(900, 7)
    separate = _fit(frame, CELLS, **_misspecified(n_folds=folds))
    original = msm._cell_policy

    def initial(fluctuation: Any, node: Any, position: int, live: int) -> Any:
        by_arm, _ = original(fluctuation, node, position, live)
        if node.policy is None:
            return by_arm, None
        return by_arm, np.where(
            node.at_risk, np.sum(node.policy * node.initial_by_arm, axis=1), 0.5
        )

    monkeypatch.setattr(msm, "_cell_policy", initial)
    projected = _fit(frame, CELLS, _saturated(), **_misspecified(n_folds=folds))
    gap = abs(projected.psi("msm_regimen[up]") - separate.psi("ey_regimen[up]"))
    assert gap > CAUGHT


def test_m10_an_msm_carry_at_the_observed_dose_breaks_the_reproduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = law.sample(900, 7)
    separate = _fit(frame, CELLS, **_misspecified())
    original = msm._cell_policy

    def observed(fluctuation: Any, node: Any, position: int, live: int) -> Any:
        by_arm, _ = original(fluctuation, node, position, live)
        if node.policy is None:
            return by_arm, None
        targeted = np.split(fluctuation.targeted.arms[sequential._REGIMEN_ARM], live)[position]
        return by_arm, np.where(node.at_risk, targeted, 0.5)

    monkeypatch.setattr(msm, "_cell_policy", observed)
    projected = _fit(frame, CELLS, _saturated(), **_misspecified())
    gap = abs(projected.psi("msm_regimen[up]") - separate.psi("ey_regimen[up]"))
    assert gap > CAUGHT


# ------------------------------------------------------------------ mutations on the exact law


def _gap(labels: tuple[str, ...]) -> float:
    result = _fit(law.frame(), labels, **_exact())
    worst = 0.0
    for label in labels[1:]:
        name = f"ey_regimen[{label}]"
        worst = max(worst, abs(result.psi(name) - law.truth(name)))
        worst = max(
            worst, float(np.max(np.abs(result.influence_curves[name][ROWS] - law.eif(name))))
        )
    return worst


def test_the_mutation_baseline_is_exact() -> None:
    assert _gap(("natural", "up", "up then history")) < 1e-10


def test_m4_a_node_two_ratio_of_one_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original = regimen_module._node_numerator

    def one_at_node_two(plan: Any, data: Any, time: int) -> Any:
        values = original(plan, data, time)
        if plan.is_mtp_node(time) and time == 2:
            return np.where(values > 0.0, 1.0, 0.0)
        return values

    monkeypatch.setattr(regimen_module, "_node_numerator", one_at_node_two)
    assert _gap(("natural", "up")) > CAUGHT


def test_m5_the_node_order_swapped_in_the_cumulative_product_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = regimen_module.Plan.cumulative_numerator

    def reversed_order(self: Any, data: Any) -> Any:
        density = self.intervention_density(data)
        if density is None or not self.has_mtp:
            return original(self, data)
        return np.cumprod(density[:, ::-1], axis=1)

    monkeypatch.setattr(regimen_module.Plan, "cumulative_numerator", reversed_order)
    assert _gap(("natural", "up")) > CAUGHT


def test_m6_the_carried_value_at_the_observed_dose_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original = sequential._targeted_step

    def observed(node: Any, fluctuation: Any, *, regression_target: Any) -> Any:
        step = original(node, fluctuation, regression_target=regression_target)
        if node.policy is None:
            return step
        return dataclasses.replace(step, marginal=step.targeted)

    monkeypatch.setattr(sequential, "_targeted_step", observed)
    assert _gap(("natural", "up")) > CAUGHT


def test_m7_a_policy_frame_without_the_earlier_dose_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cleverly.longitudinal.data import LongitudinalData

    original = LongitudinalData.policy_frame

    def blind(self: Any, time: int) -> Any:
        frame = original(self, time)
        if "A1" in frame.columns:
            frame = frame.assign(A1=0.0)
        return frame

    monkeypatch.setattr(LongitudinalData, "policy_frame", blind)
    assert _gap(("natural", "up then history")) > CAUGHT


def test_m9_a_loss_weight_without_the_ratio_leaves_a_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fluctuation weighted without the policy ratio no longer solves ``P_n D* = 0``."""
    original = sequential._fluctuate_node

    def unweighted(
        pseudo_outcome: Any, initial: Any, loss_weights: Any, fitted_on: Any, **kwargs: Any
    ) -> Any:
        return original(
            pseudo_outcome, initial, np.where(loss_weights > 0.0, 1.0, 0.0), fitted_on, **kwargs
        )

    monkeypatch.setattr(sequential, "_fluctuate_node", unweighted)
    result = _fit(law.sample(900, 11), ("natural", "up"), **_misspecified())
    assert abs(float(np.mean(result.influence_curves["ey_regimen[up]"]))) > CAUGHT
