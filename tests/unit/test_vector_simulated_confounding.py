"""The simulated common-cause surface at a vector treatment with a dose.

One latent normal ``U`` reaches every component and the outcome.  A binary component is
flipped in the upper tail of ``U`` with the declared probability, and a dose takes
``D' = D + k_D U``: each component keeps the law a one-column treatment of its kind has.  A
flip probability and a linear coefficient are not on one scale, so the grid declares one
strength axis per component, and the surface evaluates every combination.

The witnesses rebuild each cell's data longhand, without the production perturbation
helpers, and refit it, as ``tests/unit/test_simulated_confounding_policies.py`` does for a
one-column treatment.  Two mutation controls break the law the production code applies:
a latent draw per component, and a flipped dose.
"""

from __future__ import annotations

import importlib
import warnings
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import tests.discrete_law_policy_point as law
from cleverly import CausalStudy, ModifiedTreatmentPolicy, ModifiedTreatmentPolicyEffect
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding

from . import test_point_vector_treatment as point_tests
from . import test_vector_study_spine as spine
from ._source_mutation import mutate

pytestmark = pytest.mark.xdist_group("vector_simulated_confounding")

#: The module, which the function ``cleverly.sensitivity.simulated_confounding`` shadows.
surface_module = importlib.import_module("cleverly.sensitivity.simulated_confounding")

ALIAS = "ate_policy[flag gated vs natural]"
GRID = ConfounderStrengthGrid(treatment={"F": (0.0, 0.2), "D": (0.0, 0.5)}, outcome=(0.0, 0.2))


def _fit(estimand: Any = None) -> Any:
    frame = law.mixed_sample(400, 21)
    study = CausalStudy(frame, design=spine.point_design(continuous_treatment=["D"]))
    identified = study.identify(
        estimand
        if estimand is not None
        else ModifiedTreatmentPolicyEffect(
            policies=[point_tests.joint("natural"), point_tests.joint("flag gated")],
            reference="natural",
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return identified.estimate(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            simultaneous=False,
            ratio="classifier",
            random_state=5,
        )


def _longhand(result: Any, surface: Any, flag: float, dose: float, outcome: float) -> Any:
    """The published law, rebuilt without the production perturbation helpers."""
    data = result.data
    latent = np.random.default_rng(surface.latent_seed).normal(size=data.n)
    values = np.asarray(data.treatment_components, dtype=float).copy()
    f, d = data.component_names.index("F"), data.component_names.index("D")
    if flag:
        changed = latent >= NormalDist().inv_cdf(1 - flag)
        values[changed, f] = 1 - values[changed, f]
    values[:, d] += dose * latent
    y = data.outcome.copy()
    if outcome:
        changed = latent >= NormalDist().inv_cdf(1 - outcome)
        y[changed] = 1 - y[changed]
    return data.with_treatment(values).with_outcome(
        y, family=data.family, name="simulated-confounding outcome"
    )


def _surface(result: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulated_confounding(result, estimand=ALIAS, grid=GRID, random_state=5)


def _largest_gap(result: Any, surface: Any) -> float:
    gap = 0.0
    for cell in surface.cells[1:]:
        flag, dose = cell.treatment_strength
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manual = result.estimator.refit(
                _longhand(result, surface, flag, dose, cell.outcome_strength), random_state=5
            )
        gap = max(gap, abs(cell.estimate - manual[ALIAS].psi))
    return gap


def test_each_cell_equals_an_independent_refit_of_the_longhand_law() -> None:
    result = _fit()
    surface = _surface(result)
    assert surface.complete
    assert surface.treatment_family == "vector"
    assert surface.treatment_components == ("F", "D")
    assert len(surface.cells) == 8
    anchor = surface.cells[0]
    assert anchor.treatment_strength == (0.0, 0.0) and anchor.displacement == 0.0
    assert _largest_gap(result, surface) < 1e-12
    # Each axis moves the estimate, and each component's association answers its own law.
    moved = {(*cell.treatment_strength, cell.outcome_strength): cell for cell in surface.cells}
    assert abs(moved[(0.2, 0.0, 0.0)].displacement) > 1e-3
    assert abs(moved[(0.0, 0.5, 0.0)].displacement) > 1e-3
    flag_only = moved[(0.2, 0.0, 0.0)].induced_treatment_association
    dose_only = moved[(0.0, 0.5, 0.0)].induced_treatment_association
    # A tail flip of a near-balanced flag moves its association little; the dose keeps its own.
    assert flag_only[0] != anchor.induced_treatment_association[0]
    assert flag_only[1] == anchor.induced_treatment_association[1]
    assert dose_only[1] - anchor.induced_treatment_association[1] > 0.1
    frame = surface.to_frame()
    assert {"treatment_strength[F]", "treatment_strength[D]"} <= set(frame.columns)
    assert "induced_treatment_association[D]" in frame.columns
    assert "treatment strength (F, D)" in surface.summary()


def test_calibration_reports_each_component_by_its_law() -> None:
    result = _fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surface = simulated_confounding(
            result,
            estimand=ALIAS,
            grid=ConfounderStrengthGrid(treatment={"F": (0.0,), "D": (0.0,)}, outcome=(0.0,)),
            benchmark_covariates=("W",),
            random_state=5,
        )
    rows = {(row.role, row.component): row for row in surface.calibrations}
    assert rows[("treatment", "F")].family == "binomial"
    assert rows[("treatment", "D")].family == "gaussian"
    assert rows[("outcome", None)].family == "binomial"
    assert [row.component for row in surface.calibrations] == ["F", "D", None]
    assert list(surface.calibration_frame()["component"])[:2] == ["F", "D"]


# ------------------------------------------------------------------ mutation controls


def test_mv1_a_latent_draw_per_component_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each component reads its own latent draw, so no one cause reaches both.  Measured: 0.0059."""
    mutate(
        monkeypatch,
        surface_module,
        surface_module,
        "_perturb_components",
        [
            (
                "_perturb_treatment(values[:, index], latent, strengths[name], family)",
                "_perturb_treatment(values[:, index], np.roll(latent, index), "
                "strengths[name], family)",
            )
        ],
    )
    result = _fit()
    assert _largest_gap(result, _surface(result)) > 1e-4


def test_mv2_a_dose_read_by_the_flip_law_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dose is flipped as a binary label would be.  Measured: 0.020."""
    mutate(
        monkeypatch,
        surface_module,
        surface_module,
        "_perturb_components",
        [('"binary" if levels else "continuous"', '"binary"')],
    )
    result = _fit()
    surface = _surface(result)
    gap = max(
        _largest_gap(result, surface),
        max(1.0 if cell.failure is not None else 0.0 for cell in surface.cells),
    )
    assert gap > 1e-4


# ------------------------------------------------------------------ refusals


def test_a_grid_without_one_axis_per_component_is_refused() -> None:
    result = _fit()
    with pytest.raises(ValueError, match=r"treatment=\{'F': \(0.0, ...\), 'D': \(0.0, ...\)\}"):
        simulated_confounding(
            result,
            estimand=ALIAS,
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,)),
        )
    with pytest.raises(ValueError, match="This grid names \\['F'\\]"):
        simulated_confounding(
            result,
            estimand=ALIAS,
            grid=ConfounderStrengthGrid(treatment={"F": (0.0, 0.2)}, outcome=(0.0,)),
        )
    with pytest.raises(ValueError, match="binary treatment component 'F' strengths"):
        simulated_confounding(
            result,
            estimand=ALIAS,
            grid=ConfounderStrengthGrid(treatment={"F": (0.0, 0.7), "D": (0.0,)}, outcome=(0.0,)),
        )


def test_a_component_grid_on_a_one_column_treatment_is_refused() -> None:
    from cleverly import PointTreatment

    frame = law.mixed_sample(300, 22)
    identified = CausalStudy(
        frame, design=PointTreatment("Y", "F", adjustment=("W", "D"))
    ).identify(ModifiedTreatmentPolicy(policies=[point_tests.SET_FLAG]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = identified.estimate(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
        )
    with pytest.raises(ValueError, match="treatment 'F' is one column"):
        simulated_confounding(
            result,
            estimand="ey_policy[set flag]",
            grid=ConfounderStrengthGrid(treatment={"F": (0.0, 0.2)}, outcome=(0.0,)),
        )


def test_the_natural_course_mean_of_a_vector_is_refused() -> None:
    result = _fit(
        ModifiedTreatmentPolicy(
            policies=[point_tests.joint("natural"), point_tests.joint("dose up")]
        )
    )
    with pytest.raises(CapabilityError, match="is the natural course, its mean is E\\[Y\\]"):
        simulated_confounding(result, estimand="ey_policy[natural]", grid=GRID)
    assert simulated_confounding(
        result,
        estimand="ey_policy[dose up]",
        grid=ConfounderStrengthGrid(treatment={"F": (0.0,), "D": (0.0, 0.5)}, outcome=(0.0,)),
        random_state=5,
    ).complete
