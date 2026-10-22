"""The learner fingerprints of a longitudinal fit configuration (X82 parts A and C, SL).

A calibration scope matches a fit only when every fingerprint of a learner the fit used is
covered. A fingerprint of a learner the estimator never fits narrows the scope for no reason,
and a missing one widens it past what a study calibrated. Each baseline therefore records the
learners its estimate reads, and only those.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cleverly.learners import GLM
from cleverly.longitudinal import (
    LTMLE,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.validation._interpretation import fit_configuration

CONTRAST = "ate_regimen[always vs never]"
REGIMENS = {"always": 1, "never": 0}


def _panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.4 * w))).astype(float)
    l2 = 0.5 * w + 0.5 * a1 + rng.normal(size=n)
    a2 = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * l2))).astype(float)
    y = w + a1 + 0.5 * l2 + a2 + rng.normal(size=n)
    return pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y})


def _learners(model: Any) -> tuple[str, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            _panel(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
        )
    return fit_configuration(result, CONTRAST).learners


def test_sequential_g_computation_records_its_outcome_regressions() -> None:
    # The pseudo learner defaults to the outcome learner, and the mechanisms feed only the
    # positivity diagnostics.
    learners = _learners(
        SequentialGComputation(REGIMENS, reference="never", outcome_learner=LinearRegression())
    )
    assert learners == ("LinearRegression()", "LinearRegression()")


def test_sequential_g_computation_records_a_declared_pseudo_learner() -> None:
    learners = _learners(
        SequentialGComputation(
            REGIMENS,
            reference="never",
            outcome_learner=GLM(family="gaussian"),
            pseudo_learner=LinearRegression(),
        )
    )
    assert learners == ("GLM(family='gaussian')", "LinearRegression()")


def test_sequential_ipw_records_only_its_mechanism() -> None:
    learners = _learners(SequentialIPW(REGIMENS, reference="never", treatment_learner=GLM()))
    assert learners == ("GLM()",)


def test_the_parametric_g_formula_records_its_covariate_and_outcome_models() -> None:
    learners = _learners(
        ParametricGFormula(
            REGIMENS,
            reference="never",
            covariate_types={"L2": "normal"},
            outcome_learner=GLM(family="gaussian"),
            treatment_learner=GLM(),
        )
    )
    # The default covariate model is the documented main-terms GLM, not the library.
    assert learners == ("GLM()", "GLM(family='gaussian')", "GLM()")


@pytest.mark.parametrize("pseudo", [None, LinearRegression()])
def test_a_targeted_fit_records_every_regression_and_its_mechanism(pseudo: Any) -> None:
    learners = _learners(
        LTMLE(
            REGIMENS,
            reference="never",
            outcome_learner=GLM(family="gaussian"),
            pseudo_learner=pseudo,
            treatment_learner=GLM(),
            n_folds=1,
        )
    )
    # An unset pseudo learner defaults to the outcome learner, read as a regression.
    expected_pseudo = "GLM(family='gaussian')" if pseudo is None else "LinearRegression()"
    assert learners == ("GLM(family='gaussian')", expected_pseudo, "GLM()")


def test_no_shipped_registry_row_reads_a_learner_fingerprint() -> None:
    # The fingerprint change cannot move a shipped verdict's resolution while every shipped
    # row leaves the learner axis unrestricted. A row that restricts it must be calibrated
    # on the corrected fingerprints.
    from cleverly.validation._interpretation import VERDICT_EVIDENCE

    assert all(row.scope.learners is None for row in VERDICT_EVIDENCE)
