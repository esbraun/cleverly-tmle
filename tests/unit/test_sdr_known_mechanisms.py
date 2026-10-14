"""T1 for declared mechanisms: ``SDR`` reads declared treatment and censoring factors.

The cases are those of :mod:`tests.unit.test_known_node_mechanisms`, with the engine swapped
for :class:`~cleverly.longitudinal.SDR`.  With every factor known, each SDR remainder term is
zero for any outcome regression, so a wrong outcome regression must still return the truth on
the exact SMART law.  A declaration must also give the fit that a learner returning the same
factors gives, with or without cross-fitting, and on a held baseline decision.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.longitudinal import SDR
from tests.unit import _known_mechanism_support as law
from tests.unit import test_known_node_mechanisms as known
from tests.unit._natural_course_support import NeverFit


def _sdr(regimens: Any = None, **settings: Any) -> SDR:
    base: dict[str, Any] = {
        "n_folds": 1,
        "random_state": 0,
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
        "simultaneous": False,
    }
    base.update(settings)
    return SDR(known.REGIMENS if regimens is None else regimens, **base)


@pytest.mark.parametrize("censoring", [False, True], ids=["complete", "known_censoring"])
def test_a_wrong_outcome_regression_with_known_factors_recovers_the_truth(censoring: bool) -> None:
    frame = law.smart_frame(censoring=censoring)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _sdr().fit(frame, **known._columns(censoring), **known._mappings(censoring))
    for name, truth in known.TRUTH.items():
        assert abs(result[name].psi - truth) <= 1e-10, name


@pytest.mark.parametrize("n_folds", [1, 3])
def test_known_factors_equal_a_learner_that_returns_them(n_folds: int) -> None:
    frame = known._finite_smart()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        declared = _sdr(n_folds=n_folds).fit(
            frame, **known._columns(False), **known._mappings(False)
        )
        fitted = _sdr(n_folds=n_folds, treatment_learner=known.SmartMechanism()).fit(
            frame, **known._columns(False)
        )
    for name in declared:
        np.testing.assert_allclose(declared[name].psi, fitted[name].psi, rtol=1e-12)
        np.testing.assert_allclose(
            declared[name].influence_curve, fitted[name].influence_curve, rtol=1e-12, atol=1e-14
        )


@pytest.mark.parametrize("causes", [1, 2], ids=["survival", "competing_risks"])
def test_a_held_declaration_equals_a_learner_that_returns_it(causes: int) -> None:
    from tests.discrete_law_point_survival import CellMeans, CellProbabilities, survival_law

    law_ = survival_law(causes=causes)
    frame = law_.frame()
    matrix = known._held_mechanism(frame)
    regimens: dict[str, Any] = {"a0": 0, "a2": 2}
    declaration = {"treatment_probabilities": {"A": {float(a): matrix[:, a] for a in range(3)}}}
    columns = law_.fit_columns(held=True)
    common: dict[str, Any] = {
        "outcome_learner": CellMeans(),
        "pseudo_learner": CellMeans(),
        "censoring_learner": CellMeans(),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        declared = _sdr(regimens, **common).fit(frame, **columns, **declaration)
        fitted = _sdr(regimens, treatment_learner=CellProbabilities(), **common).fit(
            frame, **columns
        )
    assert declared.mechanism.known_treatment_nodes == (1,)
    known._assert_same(declared, fitted)
