"""A working model over regimen cells that include known policies.

The projection over cells is a fixed-dimension stacked estimating equation, and a policy
cell's bracket is its own integrated curve.  So on the exact law a non-saturated projection
must equal the projection of the oracle's policy means, and its curve the same linear map of
their Gateaux derivatives (T18).  A saturated design must reproduce the per-cell fits, at one
fold and at five (T19), and the per-arm predictions a policy cell carries must be the ones
the stacked fluctuation moved along that cell's own block of the design.  A carry built from
the *initial* per-arm predictions (mutation M8) breaks the reproduction under a misspecified
initial fit, where the fluctuation coefficients are not zero.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from cleverly.longitudinal import LTMLE, msm
from cleverly.msm import MSM
from cleverly.utils.bounds import bound, expit, logit, shrink_probabilities

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_multivalue as law
from .. import discrete_law_longitudinal_policy as policy
from .. import discrete_law_survival as survival
from .. import longitudinal_policies as policies
from ..studies.canonical_ltmle import QuasiBinomialGLM

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
}
LABELS = ("low", "mix", "taper")
DOSE = {"low": 0.0, "mix": 1.0, "taper": 2.5}
WEIGHT = {"low": 1.0, "mix": 2.0, "taper": 3.5}
ROWS = law.first_row_of()
ALPHA = 0.9995


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
            [np.full(len(w), float(label == item)) for item in LABELS]
        ),
        terms=LABELS,
        design_kind="known",
    )


def _exact(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": law.CellProbabilities(),
        "n_folds": 1,
        "g_bounds": (1e-8, 1.0 - 1e-8),
        "simultaneous": False,
        "random_state": 2,
        **overrides,
    }


def _misspecified(**overrides: Any) -> dict[str, Any]:
    return _exact(outcome_learner=QuasiBinomialGLM(), pseudo_learner=DummyRegressor(), **overrides)


def _fit(frame: Any, model: MSM | None, **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model is None:
            return LTMLE(policies.regimens(LABELS), reference="low", **settings).fit(
                frame, **COLUMNS
            )
        return LTMLE(policies.regimens(LABELS), msm=model, **settings).fit(frame, **COLUMNS)


def _projection_map() -> np.ndarray:
    design = np.array([[1.0, DOSE[label]] for label in LABELS])
    weights = np.array([WEIGHT[label] for label in LABELS])
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T * weights)


def test_a_policy_projection_is_the_projection_of_the_policy_means() -> None:
    """T18: coefficients and curve against the oracle, through the fixed linear map."""
    result = _fit(law.frame(), _trend(), **_exact())
    mapping = _projection_map()
    means = np.array([policy.TRUTH[f"ey_regimen[{label}]"] for label in LABELS])
    eifs = np.array([policy.eif_policy(policy.PROBS, f"ey_regimen[{label}]") for label in LABELS])
    for index, term in enumerate(("intercept", "dose")):
        name = f"msm_regimen[{term}]"
        assert result.psi(name) == pytest.approx(float(mapping[index] @ means), abs=1e-12)
        np.testing.assert_allclose(
            result.influence_curves[name][ROWS], mapping[index] @ eifs, atol=1e-10, rtol=0
        )
    assert abs(float(mapping[1] @ means)) > 0.01


@pytest.mark.parametrize("folds", [1, 5])
def test_a_saturated_projection_reproduces_the_policy_fits(folds: int) -> None:
    """T19: the stacked solve over policy cells is the per-cell recursion, at both fold counts."""
    frame = law.sample(law.PROBS, 900, 7)
    projected = _fit(frame, _saturated(), **_misspecified(n_folds=folds))
    separate = _fit(frame, None, **_misspecified(n_folds=folds))
    for label in LABELS:
        assert projected.psi(f"msm_regimen[{label}]") == pytest.approx(
            separate.psi(f"ey_regimen[{label}]"), abs=1e-10
        )
        np.testing.assert_allclose(
            projected.influence_curves[f"msm_regimen[{label}]"],
            separate.influence_curves[f"ey_regimen[{label}]"],
            atol=1e-10,
        )
    for step in separate.fits["mix"].steps:
        assert abs(float(step.fluctuation.epsilon[0])) > 0.01


@pytest.mark.parametrize("folds", [1, 5])
def test_a_policy_cell_carries_the_arms_moved_along_its_own_design(folds: int) -> None:
    result = _fit(law.sample(law.PROBS, 900, 7), _trend(), **_misspecified(n_folds=folds))
    (fitted,) = result.msm_fits
    position = LABELS.index("mix")
    cell = fitted.fits[position]
    for step in cell.steps:
        epsilon = step.fluctuation.epsilon
        moved = bound(
            expit(
                logit(shrink_probabilities(step.initial_by_arm, ALPHA))
                + (fitted.fluctuation_design[:, position, :] @ epsilon)[:, None]
            ),
            1.0 - ALPHA,
            ALPHA,
        )
        rows = step.at_risk
        np.testing.assert_allclose(step.targeted_by_arm[rows], moved[rows], atol=1e-10)
        expected = np.sum(cell.policy[step.time - 1] * step.targeted_by_arm, axis=1)
        np.testing.assert_allclose(step.value[rows], expected[rows], atol=1e-15)
        assert np.max(np.abs(epsilon)) > 0.01


@pytest.mark.parametrize("folds", [1, 5])
def test_m8_an_initial_carry_breaks_the_saturated_reproduction(
    folds: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = law.sample(law.PROBS, 900, 7)
    separate = _fit(frame, None, **_misspecified(n_folds=folds))
    original = msm._cell_policy

    def initial(fluctuation: Any, node: Any, position: int, live: int) -> Any:
        by_arm, _ = original(fluctuation, node, position, live)
        if node.policy is None:
            return by_arm, None
        return by_arm, np.where(
            node.at_risk, np.sum(node.policy * node.initial_by_arm, axis=1), 0.5
        )

    monkeypatch.setattr(msm, "_cell_policy", initial)
    projected = _fit(frame, _saturated(), **_misspecified(n_folds=folds))
    gap = abs(projected.psi("msm_regimen[mix]") - separate.psi("ey_regimen[mix]"))
    assert gap > 1e-4


def _saturated_logit() -> MSM:
    return MSM(
        design=lambda label, horizon, w: np.column_stack(
            [np.full(len(w), float(label == item)) for item in LABELS]
        ),
        terms=LABELS,
        design_kind="known",
        link="logit",
    )


@pytest.mark.parametrize("folds", [1, 5])
def test_a_saturated_logit_projection_reproduces_the_policy_fits(folds: int) -> None:
    """Under a link the stacked solve alternates; a saturated design still reduces to the cells."""
    frame = law.sample(law.PROBS, 900, 7)
    projected = _fit(frame, _saturated_logit(), **_misspecified(n_folds=folds))
    separate = _fit(frame, None, **_misspecified(n_folds=folds))
    for label in LABELS:
        coefficient = projected.psi(f"msm_regimen[{label}]")
        assert expit(coefficient) == pytest.approx(separate.psi(f"ey_regimen[{label}]"), abs=1e-10)


def test_a_saturated_survival_projection_over_policy_cells_is_the_per_cell_report() -> None:
    """Regimen-and-horizon cells of a survival policy fit, on the exact censored law."""
    cells = [(label, horizon) for label in ("never", "draw") for horizon in (1, 2)]
    model = MSM(
        design=lambda label, horizon, w: np.column_stack(
            [np.full(len(w), float((label, horizon) == cell)) for cell in cells]
        ),
        terms=tuple(f"{label}_{horizon}" for label, horizon in cells),
        design_kind="known",
    )

    def run(msm: MSM | None) -> Any:
        settings: dict[str, Any] = {
            "outcome_learner": survival.CellMeans(),
            "pseudo_learner": survival.CellMeans(),
            "treatment_learner": survival.CellMeans(),
            "censoring_learner": survival.CellMeans(),
            "n_folds": 1,
            "g_bounds": (1e-8, 1.0 - 1e-8),
            "simultaneous": False,
        }
        if msm is not None:
            settings["msm"] = msm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LTMLE({"never": 0, "draw": policies.survival_regimen()}, **settings).fit(
                survival.frame(),
                outcome=["Y1", "Y2"],
                treatment=["A1", "A2"],
                censoring=["C1", "C2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
            )

    separate, projected = run(None), run(model)
    for label, horizon in cells:
        term = f"msm_regimen[{label}_{horizon}]"
        cell = f"risk_regimen[{label} @ t={horizon}]"
        assert projected.psi(term) == pytest.approx(separate.psi(cell), abs=1e-12)
        np.testing.assert_allclose(
            projected.influence_curves[term], separate.influence_curves[cell], atol=1e-12
        )
