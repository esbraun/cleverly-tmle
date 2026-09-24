"""The MSM builders that the weight and design declaration tests share.

``tests/unit/test_msm_projection_weights.py`` (RM13) tests ``MSM(weights_kind=...)``, and
``tests/unit/test_msm_design_declaration.py`` (RM27) tests ``MSM(design_kind=...)``.  Both
build the same working models, fit them through the same entries, and replay the same fits,
so those builders live here.  ``tests/unit/test_stochastic_regime_densities.py`` (RM25)
builds an MSM from here to test the declaration that all three share.  The parts that do
not depend on the declared field are in ``tests/unit/_declaration_support.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly.longitudinal.estimator as ltmle_module
import cleverly.longitudinal.msm as regimen_msm_module
import cleverly.msm as msm_module
from cleverly import MSMProjection
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.longitudinal import LTMLE, LongitudinalData, resolve_plans, resolve_regimens
from cleverly.longitudinal.msm import evaluate_regimen_msm
from cleverly.msm import MSM, MSMSet
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from cleverly.sensitivity import simulated_confounding
from tests import discrete_law as law
from tests.conftest import linear_in_sample
from tests.unit._confounding_support import alias_for
from tests.unit._declaration_support import legacy_result, point_entries, tmle_module
from tests.unit._natural_course_support import NeverFit
from tests.unit.test_simulated_confounding_msm import _GRID as DOSE_GRID
from tests.unit.test_simulated_confounding_msm import _fit_continuous

#: Fragments of the unknown-value refusal of each declaration.  A test matches a fragment,
#: not the whole text, so a rewording of the explanation does not break it.
WEIGHTS_UNKNOWN = "weights_kind must be 'known', 'estimated' or None"
DESIGN_UNKNOWN = "design_kind must be 'known', 'estimated' or None"


@dataclass(frozen=True)
class FixedWeight:
    """A known weight, ``1 + a``.  A class rather than a lambda, so a model pickles."""

    def __call__(self, arm: Any, frame: Any) -> np.ndarray:
        return np.full(len(frame), 1.0 + float(arm))


@dataclass(frozen=True)
class CentredDesign:
    """A known design, ``[1, a, W - centre]``.  A class rather than a lambda, so it pickles."""

    centre: float = 0.0

    def __call__(self, arm: Any, frame: Any) -> np.ndarray:
        n = len(frame)
        w = np.asarray(frame["W"], dtype=float)
        return np.column_stack([np.ones(n), np.full(n, float(arm)), w - self.centre])


def linear(**declaration: Any) -> MSM:
    """``MSM.linear`` in ``W`` without an interaction: the shorthand, ``[1, a, W]``."""
    return MSM.linear(modifiers=("W",), interaction=False, **declaration)


def written(design: Any = None, **declaration: Any) -> MSM:
    """A working model with a written design, ``[1, a, W]`` unless ``design`` is given."""
    return MSM(
        design=CentredDesign() if design is None else design, terms=law.MSM_TERMS, **declaration
    )


def duration_design(label: Any, horizon: int, frame: Any) -> np.ndarray:
    """A known regimen design, ``[1, duration]``, with a duration of 2 for ``"always"``."""
    del horizon
    return np.column_stack(
        [np.ones(len(frame)), np.full(len(frame), {"always": 2.0}.get(label, 0.0))]
    )


# ------------------------------------------------------------------ the fit entries

#: Every point-treatment fit entry of an MSM: ``TMLE.fit``, ``CausalStudy.estimate``, and
#: ``TMLE.refit``.  The RM13 file drives ``"fit"`` only.
MSM_ENTRIES = point_entries(lambda model: {"msm": model}, MSMProjection)


def tmle_fit(model: MSM, learners: dict[str, Any]) -> Any:
    """``TMLE.fit`` of ``model`` on the default law."""
    return MSM_ENTRIES["fit"](model, learners)


def panel(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Two nodes, censoring, and a binary end-of-study outcome."""
    rng = np.random.default_rng(seed)
    c1 = (rng.random(n) < 0.9).astype(float)
    c2 = np.where(c1 == 1, (rng.random(n) < 0.9).astype(float), np.nan)
    observed = (c1 == 1) & (c2 == 1)
    return pd.DataFrame(
        {
            "W1": rng.standard_normal(n),
            "A1": rng.integers(0, 2, n).astype(float),
            "C1": c1,
            "A2": np.where(c1 == 1, rng.integers(0, 2, n).astype(float), np.nan),
            "C2": c2,
            "Y": np.where(observed, rng.integers(0, 2, n).astype(float), np.nan),
        }
    )


def ltmle_fit(model: MSM) -> Any:
    """``LTMLE.fit`` of a regimen ``model`` on :func:`panel`, with ``NeverFit`` learners."""
    NeverFit.calls = 0
    estimator = LTMLE(
        {"always": 1, "never": 0},
        msm=model,
        n_folds=1,
        simultaneous=False,
        outcome_learner=NeverFit(),
        pseudo_learner=NeverFit(),
        treatment_learner=NeverFit(),
        censoring_learner=NeverFit(),
    )
    return estimator.fit(
        panel(),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1"],
        censoring=["C1", "C2"],
    )


def in_sample_fit(model: MSM) -> Any:
    """An in-sample fit of ``model`` on the default law, with linear learners."""
    estimator = TMLE(msm=model, **linear_in_sample())
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


def legacy_msm_result(result: Any, field: str) -> Any:
    """``result`` as an artifact written before the MSM field ``field`` existed."""
    return legacy_result(result, field, lambda estimator: [estimator.msm])


# ------------------------------------------------------------------ the evaluators


def evaluate_point(model: MSM) -> MSMSet:
    """``MSMSet.evaluate`` called directly, as a user can call it."""
    return MSMSet.evaluate(model, CausalData.from_frame(law.frame(), outcome="Y", treatment="A"))


def evaluate_regimen(model: MSM) -> Any:
    """``evaluate_regimen_msm`` called directly on the panel of :func:`ltmle_fit`."""
    data = LongitudinalData.from_frame(
        panel(), outcome="Y", treatment=["A1", "A2"], baseline=["W1"], censoring=["C1", "C2"]
    )
    plans = resolve_plans(resolve_regimens({"always": 1, "never": 0}, data.n_times), data)
    return evaluate_regimen_msm(model, data, plans, (2,))


# ------------------------------------------------------------------ mutations


def remove_every_fit_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation M2: the fit layer and both evaluators stop checking the model.

    A point fit evaluates its design before the first learner, and ``MSMSet.evaluate``
    checks the model first.  Removing the fit-layer check alone would still refuse before
    any learner, so the mutation removes the evaluator checks too.
    """
    for module in (tmle_module, ltmle_module, msm_module, regimen_msm_module):
        monkeypatch.setattr(module, "refuse_msm_functions", lambda model: None)


def drop_from_the_replay(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    """Mutation M4: the ``replace`` of ``_freeze_msm`` drops the declaration ``field``."""

    def dropping(model: Any, **changes: Any) -> Any:
        changes.pop(field, None)
        return replace(model, **changes)

    monkeypatch.setattr(replay_module, "replace", dropping)


# ------------------------------------------------------------------ the continuous replay

#: The dose slope: the coefficient of a continuous fit that the replay targets.
DOSE_SLOPE = "a"


def uniform_dose_fit() -> Any:
    """A continuous ``MSM.linear`` fit with uniform weights, cached across both files.

    ``_freeze_msm`` builds a continuous replay with its own ``replace`` call.  Only an
    undeclared weight or a legacy design makes that call supply a declaration, so the RM13
    and RM27 files both replay this fit.
    """
    return _fit_continuous(uniform=True)


def dose_surface(result: Any) -> Any:
    """The simulated-confounding surface of the dose slope, with every cell replayed."""
    alias = alias_for(result, coefficient=DOSE_SLOPE)
    surface = simulated_confounding(result, estimand=alias, grid=DOSE_GRID, random_state=31)
    assert all(cell.failure is None for cell in surface.cells)
    return surface


# ------------------------------------------------------------------ the exact-law witness


def msm_eif(probs: Any, weights: Any, design: Any = None) -> np.ndarray:
    """``(12, p)``: the Gateaux derivative of ``law.msm_beta(P, weights, design)``, per term.

    A ``weights`` or ``design`` that is a function of the cell probabilities is recomputed
    from the perturbed law, so the derivative carries the term through it that a frozen
    array does not.
    """
    return law.gateaux_eif(lambda p: law.msm_beta(p, weights, design=design), probs)


def msm_curve(fit: Any, index: int) -> np.ndarray:
    """The reported influence curve of the working-model coefficient ``index``."""
    return np.asarray(fit.estimates[f"msm[{law.MSM_TERMS[index]}]"].influence_curve)
