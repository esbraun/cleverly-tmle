r"""T2: sequential double robustness on the exact two-node law, through SDR and LTMLE.

A pattern names, per node, which nuisance is wrong: ``m`` the outcome regression, ``r`` the
treatment-and-censoring mechanism.  A wrong regression at node 2 is least squares on ``L2``
alone, and at node 1 the sample mean.  A wrong mechanism is the marginal class share.  The
right nuisance is the saturated cell mean, which is exact on the law.

Lemma 4 of Díaz, Williams, Hoffman and Schenck (2023) says SDR is consistent when at each
node one of the two is right, in any combination over the nodes.  On the exact law that is
exact: the error is zero in every pattern P_A to P_D.  LTMLE is exact in P_A to P_C and not
in P_D, where node 1's mechanism and node 2's regression are both wrong.  The control is
SDR with both nuisances wrong at node 2, which no robustness covers.  The measured values,
on the law as written, are 0.0221 for LTMLE's ``never`` error in P_D and -0.0311 for its
ATE error, and -0.0154 for the control's ``never`` error.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import LTMLE, SDR

from .. import discrete_law_longitudinal as law
from ..longitudinal_sdr import (
    CENSORING_WIDTH,
    COLUMNS,
    OUTCOME_WIDTH,
    TREATMENT_WIDTH,
    NodeLearner,
)
from ..studies.canonical_ltmle import declared_regimens

EXACT_FLOOR = 1e-12
NAMES = ("ey_regimen[never]", "ey_regimen[always]", "ate_regimen[always vs never]")
PATTERNS = {
    "both_correct": ("", ""),
    "P_A": ("r", "r"),
    "P_B": ("m", "r"),
    "P_C": ("m", "m"),
    "P_D": ("r", "m"),
}
CONTROL = ("", "mr")


def _learners(first: str, second: str) -> dict[str, Any]:
    """The four learner slots for a pattern, each dispatching per node."""

    def mechanism(wrong: bool) -> Any:
        return DummyClassifier(strategy="prior") if wrong else law.CellMeans()

    horizon = OUTCOME_WIDTH[2]
    return {
        "outcome_learner": NodeLearner(
            {horizon: LinearRegression() if "m" in second else law.CellMeans()},
            # The wrong regression drops W, the first column of [W, L2].
            columns={horizon: (1,)} if "m" in second else None,
        ),
        "pseudo_learner": NodeLearner(
            {OUTCOME_WIDTH[1]: DummyRegressor() if "m" in first else law.CellMeans()}
        ),
        "treatment_learner": NodeLearner(
            {
                TREATMENT_WIDTH[1]: mechanism("r" in first),
                TREATMENT_WIDTH[2]: mechanism("r" in second),
            }
        ),
        "censoring_learner": NodeLearner(
            {
                CENSORING_WIDTH[1]: mechanism("r" in first),
                CENSORING_WIDTH[2]: mechanism("r" in second),
            }
        ),
    }


def _fit(engine: type, pattern: tuple[str, str]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine(
            declared_regimens({label: law.REGIMEN_SPEC[label] for label in ("never", "always")}),
            reference="never",
            n_folds=1,
            g_bounds=(1e-8, 1.0),
            simultaneous=False,
            **_learners(*pattern),
        ).fit(law.frame(), **COLUMNS)


def _errors(fit: Any) -> dict[str, float]:
    return {name: fit[name].psi - law.TRUTH[name] for name in NAMES}


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
def test_sdr_is_exact_when_one_nuisance_is_right_at_each_node(pattern: str) -> None:
    for name, error in _errors(_fit(SDR, PATTERNS[pattern])).items():
        assert abs(error) < EXACT_FLOOR, (pattern, name, error)


@pytest.mark.parametrize("pattern", ["both_correct", "P_A", "P_B", "P_C"])
def test_ltmle_is_exact_in_the_patterns_it_covers(pattern: str) -> None:
    for name, error in _errors(_fit(LTMLE, PATTERNS[pattern])).items():
        assert abs(error) < EXACT_FLOOR, (pattern, name, error)


def test_ltmle_is_not_exact_where_sdr_is() -> None:
    """P_D separates the two: the witness that the SDR transform is doing work."""
    errors = _errors(_fit(LTMLE, PATTERNS["P_D"]))
    assert abs(errors["ey_regimen[never]"]) >= 0.01
    assert abs(errors["ate_regimen[always vs never]"]) >= 0.02


def test_the_wrong_regression_is_materially_wrong() -> None:
    """The P_D node-2 fit differs from the truth, so exactness there is not vacuous."""
    right = _fit(SDR, PATTERNS["both_correct"]).fits["never"].steps[1]
    wrong = _fit(SDR, PATTERNS["P_D"]).fits["never"].steps[1]
    rows = right.trained_on
    assert np.max(np.abs(wrong.initial[rows] - right.initial[rows])) >= 0.05


def test_the_control_with_both_wrong_at_one_node_is_biased() -> None:
    errors = _errors(_fit(SDR, CONTROL))
    assert abs(errors["ey_regimen[never]"]) >= 0.01
    assert abs(errors["ey_regimen[always]"]) >= 0.01
