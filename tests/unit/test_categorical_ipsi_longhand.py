r"""C3: the cross-fitted engine against a numpy transcription of the categorical recursion.

:func:`tests.longitudinal_ipsi_categorical.longhand_ipsi_categorical` is Kennedy's Algorithm 1
with the dropout weights of Kim, Kennedy and Naimi (2021) and the categorical tilt
:math:`q_t(a) = \delta_{t,a}\pi_t(a)/D_t`.  The nuisances are a multinomial logistic mechanism, a
logistic censoring model and a linear regression.  None is saturated, so :math:`P_n` of the
mechanism term is not zero and the comparison checks the term's contribution to the estimate
as well as to the curve.  The test runs in sample and at five folds.  The longhand shares no
code with the engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests import discrete_law_longitudinal_categorical_incremental as law
from tests.longitudinal_ipsi_categorical import longhand_ipsi_categorical, sorted_factors
from tests.unit import _categorical_ipsi_support as sup

pytestmark = pytest.mark.xdist_group("categorical_ipsi_longhand")

EXACT = 1e-12
CAUGHT = 1e-4
MEAN_FLOOR = 1e-5
LOGISTIC = LogisticRegression(max_iter=1000)
LINEAR = LinearRegression()
#: The tilts of the plans the test fits: the natural course and one tilt per node.
FACTORS = {
    "natural": ({"high": 1.0}, {"high": 1.0}),
    "per_node": (law.T_A, law.T_B),
    "T_a": (law.T_A, law.T_A),
}
PLANS = {label: (law.tilt(a), law.tilt(b)) for label, (a, b) in FACTORS.items()}
N, SEED = 900, 3


@pytest.fixture(scope="module")
def frame() -> Any:
    sample = law.end_frame().sample(n=N, replace=True, random_state=SEED)
    return sample.reset_index(drop=True)


@pytest.fixture(scope="module", params=[1, 5])
def fitted(request: Any, frame: Any) -> tuple[int, Any]:
    fit = sup.fit_ipsi(
        frame,
        PLANS,
        outcome_learner=LINEAR,
        pseudo_learner=LINEAR,
        treatment_learner=LOGISTIC,
        censoring_learner=LOGISTIC,
        n_folds=request.param,
        g_bounds=(0.01, 1.0),
        random_state=4,
    )
    return request.param, fit


def _longhand(fit: Any, frame: Any, label: str, n_folds: int) -> Any:
    assignment = np.zeros(len(frame), dtype=int) if n_folds == 1 else fit.folds.assignment
    first, second = FACTORS[label]
    return longhand_ipsi_categorical(
        frame,
        (sorted_factors(first), sorted_factors(second)),
        assignment,
        treatment_learner=LOGISTIC,
        censoring_learner=LOGISTIC,
        outcome_learner=LINEAR,
    )


@pytest.mark.parametrize("label", sorted(FACTORS))
def test_the_engine_is_the_longhand_recursion(
    fitted: tuple[int, Any], frame: Any, label: str
) -> None:
    """Estimate, curve, each held-out row's node-1 target and mechanism term."""
    n_folds, fit = fitted
    longhand = _longhand(fit, frame, label, n_folds)
    name = f"ey_regimen[{label}]"
    assert fit.psi(name) == pytest.approx(longhand.psi, abs=EXACT)
    np.testing.assert_allclose(
        fit.influence_curves[name], longhand.phi - longhand.psi, atol=EXACT, rtol=0
    )
    observed = frame["C1"].to_numpy() == 1.0
    first = fit.fits[label].steps[0]
    np.testing.assert_allclose(
        first.pseudo_outcome[observed], longhand.target[observed], atol=EXACT, rtol=0
    )
    np.testing.assert_allclose(
        fit.fits[label].mechanism_term.sum(axis=1), longhand.mechanism_term, atol=EXACT, rtol=0
    )


def test_the_mechanism_term_moves_the_cross_fitted_estimate(frame: Any) -> None:
    """The estimate-side witness: with a non-saturated mechanism the term's mean is not 0.

    The mean of the engine's mechanism term, not only its variance, enters the estimate. The floor
    is seven orders above the tolerance of the longhand comparison, so a dropped term fails it.
    """
    fit = sup.fit_ipsi(
        frame,
        PLANS,
        outcome_learner=LINEAR,
        pseudo_learner=LINEAR,
        treatment_learner=LOGISTIC,
        censoring_learner=LOGISTIC,
        n_folds=5,
        g_bounds=(0.01, 1.0),
        random_state=4,
    )
    means = {
        label: abs(float(np.mean(fit.fits[label].mechanism_term.sum(axis=1)))) for label in PLANS
    }
    assert all(value > MEAN_FLOOR for value in means.values()), means
