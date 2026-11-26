"""The warning when a g-computation fit's diagnostic mechanism cannot be fitted.

On the classifier route, a plan held inside a declared price band never assigns the lowest
recorded prices, so the stacked observed and policy rows separate on the plan's step column and
the default unpenalized GLM cannot fit them. The warning names that cause, gives the remedy, and
describes the outcome regressions the fit actually used rather than GLMs. The witness passes a
penalized classifier and checks that the diagnostics then compute, so the remedy is real.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    GComputationMethod,
    LongitudinalTreatment,
    ModelSpec,
    PositivityWarning,
    RegimeContrast,
    Runtime,
)
from cleverly.datasets import PRICE_BANDS, PRICE_PLANS, make_sequential_pricing


def _fit(treatment_learner: object | None) -> tuple[list[str], Any]:
    frame, _ = make_sequential_pricing(2_000, seed=7)
    design = LongitudinalTreatment(
        outcome="renewed",
        treatment=["price_1", "price_2", "price_3"],
        baseline=["usage"],
        time_varying=[["engaged_1"], ["engaged_2"], ["engaged_3"]],
        continuous_treatment=PRICE_BANDS,
    )
    extra: dict[str, Any] = (
        {} if treatment_learner is None else {"treatment_learner": treatment_learner}
    )
    method = GComputationMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            ratio="classifier",
            **extra,
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=7, n_jobs=1),
    )
    effect = CausalStudy(frame, design=design).identify(
        RegimeContrast(PRICE_PLANS, reference="natural_prices")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = effect.estimate(method=method)
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, PositivityWarning) and "diagnostic" in str(w.message)
    ]
    return messages, result


def test_the_default_glm_on_banded_prices_warns_with_the_cause_and_the_remedy() -> None:
    messages, result = _fit(None)
    assert messages, "the separated design should still fail the unpenalized GLM"
    text = messages[0]
    assert "On the classifier route" in text
    assert "held inside a declared band" in text
    assert "treatment_learner" in text and "LogisticRegression" in text
    assert "outcome regressions" in text
    assert "outcome GLMs" not in text
    assert "where the fit reports one" in text
    # The point estimates do not read the mechanism, so they stay finite.
    assert all(np.isfinite(e.psi) for e in result.estimates.values())


def test_a_penalized_classifier_computes_the_diagnostics_without_the_warning() -> None:
    messages, _ = _fit(LogisticRegression(max_iter=1000))
    assert messages == []
