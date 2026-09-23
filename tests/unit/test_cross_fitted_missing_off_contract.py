"""Cross-fitted missing-outcome fits outside the two audited contracts are refused.

Two stacked CV-TMLE contracts cover missing outcomes (``delta=``) under cross-fitting: the
natural-course mean and the arm-indexed means and contrasts. A shift, incremental, regime,
or MSM axis, or a declared intermediate, puts a fit outside both. Such a fit used to run and
report an interval that no audit read a source for. A ``Static`` regime or a saturated MSM
also reproduced the arm-indexed fit, and so escaped that contract's refusals of ``repeats``,
fold targeting, ``cv_evaluation``, the linear fluctuation, and ``id=``. F21 in ``docs/roadmap.md`` holds the missing results, and RM20
records the decision to refuse.

Every refusal here must run before any learner, which :class:`NeverFit` enforces. The
controls show what the refusal leaves alone: the in-sample fit of each target, the
arm-indexed contract, and the F20 refusal of the population-intervention targets. A
deliberate-mutation control removes the refusal and requires the witness to fail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import cde_dgp, make_missing_outcome_binary
from cleverly.estimators import TMLE
from cleverly.estimators.tmle import (
    _ARM_INDEXED_CONTRACT,
    _CROSS_FITTED_MISSING_CONTRACTS,
    _IN_SAMPLE_ARM_INDEXED_REMEDY,
)
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Shift, Static
from cleverly.msm import MSM
from cleverly.utils.bounds import expit
from tests.unit._natural_course_support import NeverFit, never_fit_learners

COVARIATES = ["W1", "W2", "W3"]


def binary_missing_frame() -> pd.DataFrame:
    """A binary outcome with missing values, a binary arm, a dose, and a cluster label."""
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    rng = np.random.default_rng(0)
    frame["dose"] = frame["A"] + rng.normal(size=len(frame))
    frame["cid"] = np.arange(len(frame)) // 5
    return frame


#: :func:`~cleverly.datasets.cde_dgp`, with its outcome-mean formula mapped through
#: ``expit`` to a Bernoulli outcome, and 80 percent of outcomes observed.
#: :func:`~cleverly.datasets.make_cde` has no binary variant, and a Gaussian outcome would
#: meet the cross-fitted outcome-scale refusal first.
_CDE = cde_dgp()
BINARY_CDE = replace(
    _CDE,
    name="binary_controlled_direct_effect_missing",
    outcome_mean=lambda w, a, z: expit(_CDE.outcome_mean(w, a, z)),
    family="binomial",
    missingness=lambda w, a: np.full(len(w), 0.8),
)


def binary_cde_frame(n: int = 400, seed: int = 13) -> pd.DataFrame:
    """A binary-outcome controlled direct effect with missing outcomes, from :data:`BINARY_CDE`."""
    frame, _ = BINARY_CDE.sample(n=n, seed=seed)
    return frame


def learners() -> dict[str, Any]:
    """Fresh logistic learners for every role, for the controls that fit."""
    return {
        role: LogisticRegression(max_iter=1000)
        for role in (
            "outcome_learner",
            "treatment_learner",
            "missingness_learner",
            "intermediate_learner",
        )
    }


@dataclass(frozen=True)
class Composition:
    """One target family, fitted with ``delta=``."""

    surface: str
    settings: dict[str, Any]
    frame: Callable[[], pd.DataFrame]
    columns: dict[str, Any] = field(default_factory=dict)

    def estimator(self, cross_fit: bool, learner_roles: dict[str, Any], **extra: Any) -> TMLE:
        return TMLE(
            **{
                "estimands": None,
                "n_folds": 5,
                "random_state": 0,
                "simultaneous": False,
                **learner_roles,
                **self.settings,
                "cross_fit": cross_fit,
                **extra,
            }
        )

    def fit(self, estimator: TMLE, **fit_extra: Any) -> Any:
        columns = {
            "outcome": "Y",
            "treatment": "A",
            "covariates": COVARIATES,
            "delta": "Delta",
            **self.columns,
            **fit_extra,
        }
        return estimator.fit(self.frame(), **columns)


COMPOSITIONS = {
    "shift": Composition(
        "shift",
        {"shifts": [Shift(0.5, cap=None)], "density_bins": 4},
        binary_missing_frame,
        {"treatment": "dose", "treatment_kind": "continuous"},
    ),
    "incremental": Composition(
        "incremental", {"incremental": [Incremental(2.0)]}, binary_missing_frame
    ),
    "regime": Composition(
        "regime", {"interventions": [Static(1), Static(0)]}, binary_missing_frame
    ),
    "msm": Composition("MSM", {"msm": MSM.linear()}, binary_missing_frame),
    "cde": Composition(
        "controlled-direct-effect",
        {"estimands": ["ate"]},
        binary_cde_frame,
        {"intermediate": "Z"},
    ),
}


def assert_refused_before_any_learner(
    composition: Composition, *, extra: dict[str, Any] | None = None, **fit_extra: Any
) -> None:
    """The witness: a cross-fitted fit with ``delta=`` raises the F21 refusal, unfitted."""
    estimator = composition.estimator(True, never_fit_learners(), **(extra or {}))
    with pytest.raises(CapabilityError) as raised:
        composition.fit(estimator, **fit_extra)
    message = str(raised.value)
    assert message.startswith(_CROSS_FITTED_MISSING_CONTRACTS), message
    assert f"no audited result covers {composition.surface} targets" in message
    assert "F21 in docs/roadmap.md" in message
    assert message.endswith(_IN_SAMPLE_ARM_INDEXED_REMEDY), message
    assert NeverFit.calls == 0, f"{NeverFit.calls} learner fit(s) ran before the refusal"


@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_a_cross_fitted_fit_with_missing_outcomes_is_refused_before_any_learner(
    name: str,
) -> None:
    assert_refused_before_any_learner(COMPOSITIONS[name])


@pytest.mark.parametrize(
    ("extra", "fit_extra"),
    [
        ({"repeats": 2}, {}),
        ({"targeting_scheme": "fold"}, {}),
        ({"cv_evaluation": True}, {}),
        ({"fluctuation": "linear"}, {}),
        ({}, {"id": "cid"}),
    ],
    ids=["repeats", "fold-targeting", "fold-evaluation", "linear-fluctuation", "clustered"],
)
def test_a_static_regime_no_longer_bypasses_the_arm_indexed_refusals(
    extra: dict[str, Any], fit_extra: dict[str, Any]
) -> None:
    """The closed bypass. The arm-indexed contract refuses each of these settings.

    Before this refusal, a ``Static`` regime with any of them fitted and reported an
    interval. Without them, its ``ey_regime[always 1]`` equalled the arm-indexed ``ey1``.
    """
    assert_refused_before_any_learner(COMPOSITIONS["regime"], extra=extra, **fit_extra)


@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_removing_the_refusal_makes_the_witness_fail(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deliberate-mutation control. Without the refusal, a learner is fitted."""
    monkeypatch.setattr(TMLE, "_refuse_cross_fitted_missing_off_contract", lambda *a, **k: None)
    with pytest.raises(AssertionError):
        assert_refused_before_any_learner(COMPOSITIONS[name])
    assert NeverFit.calls > 0, "the mutated fit refused before a learner for another reason"


@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_the_in_sample_fit_keeps_its_interval(name: str) -> None:
    """The remedy the refusal names is a fit the package admits."""
    composition = COMPOSITIONS[name]
    results = composition.fit(composition.estimator(False, learners()))
    estimates = [
        estimate for _, result in results.items() for estimate in result.estimates.values()
    ]
    assert estimates
    for estimate in estimates:
        lower, upper = estimate.ci
        assert np.isfinite(lower) and np.isfinite(upper) and lower < upper, estimate.name


#: The arm-indexed contract's own composition, which the refusal leaves alone.
ARM_INDEXED = Composition("arm-indexed", {}, binary_missing_frame)


class TestTheNeighbouringRefusalsKeepTheirOwnSentences:
    """The refusal covers what is left, and each narrower surface keeps its own sentence."""

    def test_a_cross_fitted_ate_reaches_the_arm_indexed_contract(self) -> None:
        estimator = ARM_INDEXED.estimator(True, never_fit_learners(), estimands=["ate"], repeats=2)
        with pytest.raises(CapabilityError) as raised:
            ARM_INDEXED.fit(estimator)
        message = str(raised.value)
        assert message.startswith(_ARM_INDEXED_CONTRACT), message
        assert "Set repeats=1" in message
        assert NeverFit.calls == 0

    def test_the_admitted_cross_fitted_ate_still_fits(self) -> None:
        result = ARM_INDEXED.fit(
            ARM_INDEXED.estimator(True, learners(), estimands=["ate"])
        ).single()
        lower, upper = result.estimates["ate"].ci
        assert lower < result.psi("ate") < upper

    def test_par_keeps_its_f20_refusal(self) -> None:
        estimator = ARM_INDEXED.estimator(True, never_fit_learners(), estimands=["par"])
        with pytest.raises(CapabilityError) as raised:
            ARM_INDEXED.fit(estimator)
        message = str(raised.value)
        assert message.startswith("par does not yet support delta="), message
        assert NeverFit.calls == 0

    def test_par_beside_an_intermediate_keeps_its_f20_refusal(self) -> None:
        """Beside an intermediate, only the F20 guard keeps this sentence.

        Without that guard the fit would name F21 and offer the in-sample remedy, which
        is false for ``par``: it is refused at every setting. On the intermediate path the
        F20 refusal runs after the shared nuisances, in sample and cross-fitted alike, so
        this control fits real learners.
        """
        cde = COMPOSITIONS["cde"]
        with pytest.raises(CapabilityError) as raised:
            cde.fit(cde.estimator(True, learners(), estimands=["par"]))
        message = str(raised.value)
        assert message.startswith("par does not yet support delta="), message
        assert "F21" not in message
