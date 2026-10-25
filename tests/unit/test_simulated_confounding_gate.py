"""X82 part F3a: the declared replay gate, any registered MSM link, any known regime class.

The gate admits a binary point fit whose estimator class declares
``simulated_confounding_replay = True``.  The enumeration test collects every such class of
the package and runs one zero-strength reproduction and one nonzero cell for each, so a
class that declares the attribute later enters the test without an edit here.
"""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    AdaptiveIPTWMethod,
    CausalStudy,
    CrossFitting,
    DRTMLEMethod,
    Inference,
    ModelSpec,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Runtime,
)
from cleverly.datasets import make_binary_outcome
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import Rule, Static
from cleverly.msm import LINKS, Link, register_link
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity._simulated_confounding_common import declares_replay, replay_classes
from cleverly.sensitivity._simulated_confounding_fixed import fixed_replay_refusal
from tests.unit._confounding_support import _collaborative_method
from tests.unit._confounding_support import replacement as _replacement
from tests.unit._simulated_confounding_support import (
    _GRID,
    _alias,
    _estimate,
    _fit_msm,
    _model,
    _study,
)

_SEED = 7
_GATE_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))


def _declaring_classes() -> list[type]:
    """Every package class under ``TMLE`` that declares the replay in its own namespace."""
    found: list[type] = []
    pending: list[type] = [TMLE]
    while pending:
        cls = pending.pop(0)
        if cls.__module__.startswith("cleverly.") and vars(cls).get("simulated_confounding_replay"):
            found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def _adaptive_iptw_method() -> Any:
    """The in-sample AdaptiveIPTW method that ``test_adaptive_iptw_refute.py`` fits."""
    return AdaptiveIPTWMethod(
        models=ModelSpec(
            treatment_learner=LogisticRegression(max_iter=1000),
            outcome_learner=LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(enabled=False),
        inference=Inference(simultaneous=False),
        runtime=Runtime(random_state=_SEED),
    )


#: The study method that fits each declaring class, keyed by ``_assessment_method``.  A class
#: that declares the replay and has no entry here fails the enumeration test by name.
_METHODS: dict[str, Any] = {
    "tmle": lambda: "tmle",
    "collaborative_tmle": _collaborative_method,
    "drtmle": lambda: DRTMLEMethod(
        reduced_outcome_learner=LinearRegression(),
        reduced_mechanism_residual_learner=LinearRegression(),
        reduced_treatment_learner=LogisticRegression(max_iter=1000),
        # The shared fit uses two folds, and the fold-wise default needs three.
        reduced_crossfit="pooled",
    ),
    "adaptive_iptw": _adaptive_iptw_method,
}


@cache
def _fit(method_name: str) -> Any:
    frame, _ = make_binary_outcome(n=400, seed=_SEED)
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
    )
    if method_name == "adaptive_iptw":
        return study.identify(ATE()).estimate(method=_adaptive_iptw_method())
    return study.identify(ATE()).estimate(
        method=_METHODS[method_name](),
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=_SEED,
        simultaneous=False,
    )


def _check_replays(result: Any) -> None:
    """The zero-strength reproduction and one nonzero cell that moves as a manual refit does."""
    reproduced = result.estimator.refit(
        result.data, intermediate_value=result.intermediate_value, random_state=_SEED
    )
    assert reproduced["ate"].psi == pytest.approx(result["ate"].psi, abs=1e-10)
    surface = simulated_confounding(result, estimand="ate", grid=_GATE_GRID, random_state=_SEED)
    assert surface.complete
    assert surface.cells[0].estimate == result["ate"].psi
    cell = surface.cells[1]
    manual = result.estimator.refit(
        _replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
        intermediate_value=result.intermediate_value,
        random_state=_SEED,
    )
    assert cell.estimate == pytest.approx(manual["ate"].psi, abs=1e-12)
    # The flip of the upper 10% latent tail moves this law by about -0.05 under each class.
    # A refit that ignores the perturbed treatment leaves the cell at the anchor.
    assert cell.displacement is not None
    assert cell.displacement < -0.02


def test_the_shipped_classes_declare_the_replay() -> None:
    names = {cls.__name__ for cls in _declaring_classes()}
    assert {"TMLE", "CTMLE", "DRTMLE", "AdaptiveIPTW"} <= names
    assert set(replay_classes()) == names


@pytest.mark.parametrize("cls", _declaring_classes(), ids=lambda cls: cls.__name__)
def test_every_declaring_class_reproduces_and_moves(cls: type) -> None:
    method = vars(cls).get("_assessment_method", cls._assessment_method)  # type: ignore[attr-defined]
    assert method in _METHODS, f"add a study method for {cls.__name__} to _METHODS"
    result = _fit(method)
    assert type(result.estimator) is cls
    _check_replays(result)


def test_a_replay_that_ignores_the_perturbed_table_fails_the_nonzero_cell() -> None:
    """Mutation control: a declaring class whose refit reads the original table."""
    result = _fit("tmle")

    class _IgnoresThePerturbation(TMLE):
        simulated_confounding_replay = True

        def refit(self, data: Any, **kwargs: Any) -> Any:
            return TMLE.refit(self, result.data, **kwargs)

    stub = copy(result.estimator)
    stub.__class__ = _IgnoresThePerturbation
    with pytest.raises(AssertionError):
        _check_replays(replace(result, estimator=stub))


def test_an_inherited_declaration_is_not_a_declaration() -> None:
    class _Inherits(TMLE):
        pass

    class _Declares(TMLE):
        simulated_confounding_replay = True

    assert declares_replay(TMLE())
    assert not declares_replay(_Inherits())
    assert declares_replay(_Declares())
    result = _fit("tmle")
    stub = copy(result.estimator)
    stub.__class__ = _Inherits
    with pytest.raises(CapabilityError, match="declares simulated_confounding_replay = True"):
        simulated_confounding(replace(result, estimator=stub), estimand="ate", grid=_GATE_GRID)


@pytest.mark.parametrize(
    ("method", "roadmap"),
    [("drtmle", "X50"), ("collaborative_tmle", "X51"), ("adaptive_iptw", "X81")],
)
@pytest.mark.parametrize("target", ["ey_regime", "ate_regime", "msm"])
def test_dr_and_collaborative_regimes_stay_with_their_rows(
    method: str, roadmap: str, target: str
) -> None:
    estimator = _fit(method).estimator
    refusal = fixed_replay_refusal(estimator, target)
    assert refusal is not None
    assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert refusal.roadmap == roadmap
    assert type(estimator).__name__ in refusal.condition


# ------------------------------------------------------------------ (a) any registered link


@pytest.fixture
def identity_copy() -> Any:
    """Register the identity link under a second name for one test."""
    identity = LINKS["identity"]
    link = register_link(
        Link(
            "identity_copy",
            inverse=identity.inverse,
            slope=identity.slope,
            curvature=identity.curvature,
        )
    )
    yield link
    del LINKS["identity_copy"]


def test_a_registered_user_link_replays_as_the_link_it_equals(identity_copy: Any) -> None:
    """A user link equal to the identity gives the identity fit's cells.

    The copy is not named ``"identity"``, so the projection takes the alternation path that
    every other link takes, and only a replay under the fit's own link agrees.
    """
    reference = _fit_msm(saturated=True)
    user = _estimate(
        _study(),
        MSMProjection(_model(saturated=True, link="identity_copy")),
    )
    assert user.estimator.msm.link == "identity_copy"
    alias = _alias(user)
    expected = simulated_confounding(reference, estimand=alias, grid=_GRID, random_state=31)
    surface = simulated_confounding(user, estimand=alias, grid=_GRID, random_state=31)
    assert surface.cells[0].estimate == pytest.approx(expected.cells[0].estimate, abs=1e-8)
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-3
    assert [cell.estimate for cell in surface.cells] == pytest.approx(
        [cell.estimate for cell in expected.cells], abs=1e-8
    )


# ---------------------------------------------------------- (b) any known regime class


class _KnownThreshold:
    """A user-written Intervention whose density equals ``Rule(W > 0 -> 1, else 0)``."""

    density_kind = "known"
    name = "policy"

    def density(self, data: Any) -> np.ndarray:
        column = data.covariate_names.index("W")
        treated = (data.covariates[:, column] > 0).astype(float)
        return np.column_stack((1.0 - treated, treated))


def _threshold_rule() -> Rule:
    return Rule(lambda w: np.where(np.asarray(w["W"]) > 0, 1, 0), name="policy", rule_kind="known")


@pytest.mark.parametrize("contrast", [False, True])
def test_a_known_intervention_class_replays_as_the_rule_it_equals(contrast: bool) -> None:
    def fit(policy: Any) -> Any:
        reference = Static(0, name="reference")
        target = (
            RegimeContrast((reference, policy), reference="reference")
            if contrast
            else RegimeMean((policy,))
        )
        return _estimate(_study(), target)

    user, rule = fit(_KnownThreshold()), fit(_threshold_rule())
    alias = _alias(user)
    assert alias == _alias(rule)
    surface = simulated_confounding(user, estimand=alias, grid=_GRID, random_state=31)
    expected = simulated_confounding(rule, estimand=alias, grid=_GRID, random_state=31)
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-3
    assert [cell.estimate for cell in surface.cells] == [cell.estimate for cell in expected.cells]


def test_a_regime_outside_the_intervention_protocol_is_refused() -> None:
    """The gate reads the protocol; ``RegimeSet.evaluate`` reads the density declaration.

    ``test_rule_and_intervention_declarations.py::TestTheReplay`` pins the declaration
    refusal of an undeclared regime at replay.
    """
    result = _estimate(_study(), RegimeMean((_KnownThreshold(),)))
    estimator = copy(result.estimator)
    estimator.interventions = (object(),)
    refusal = fixed_replay_refusal(estimator, "ey_regime")
    assert refusal is not None
    assert "implements the Intervention protocol" in refusal.condition
