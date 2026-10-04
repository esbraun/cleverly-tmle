r"""Is the risk-ratio tilt's influence curve the efficient influence function?

The arm-indexed law of :mod:`tests.discrete_law` states :math:`\Psi(\delta)` longhand for
lmtp's ``ipsi`` rule.  Below one, a treated unit keeps treatment with probability
:math:`\delta`.  Above one, an untreated unit stays untreated with probability
:math:`1 / \delta`.  The mean reads :math:`g`, so the complex step differentiates through
the mechanism as well as through :math:`\bar Q`.  The estimator's curve is compared with
that derivative at every support point.

``tests/unit/test_policy_point_exact.py`` checks the same estimand on its own grid.  This
module is the registry's oracle: ``tests/unit/test_registry.py`` requires every target to
own a branch of a law's ``functional``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.estimators import TMLE
from cleverly.interventions import RiskRatioTilt
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment

MEANS = tuple(law.PER_ARM_NAMES["ey_rr_tilt"])
CONTRASTS = tuple(law.PER_ARM_NAMES["ate_rr_tilt"])
ESTIMANDS = MEANS + CONTRASTS


def tilts() -> list[RiskRatioTilt]:
    return [RiskRatioTilt(delta, name=label) for label, delta in law.RR_TILT_DELTAS.items()]


@pytest.fixture(scope="module")
def exact_fit():
    """A tilt fit on the discrete law with oracle nuisances, so the curve is the EIF."""
    dgp = law.DiscreteLaw()
    estimator = TMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        policies=tilts(),
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


def test_the_deltas_straddle_one() -> None:
    values = list(law.RR_TILT_DELTAS.values())
    assert min(values) < 1.0 < max(values), "each side of one is a different branch"
    assert law.RR_TILT_DELTAS[law.RR_TILT_REFERENCE] == 1.0


def test_targeting_has_nothing_left_to_do(exact_fit) -> None:
    for fluctuation in exact_fit.fluctuations.values():
        assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", ESTIMANDS)
def test_matches_the_numerical_gateaux_derivative(exact_fit, name: str) -> None:
    reported = np.asarray(exact_fit.estimates[name].influence_curve)[law.first_row_of()]
    np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", ESTIMANDS)
def test_the_point_estimate_is_the_functional(exact_fit, name: str) -> None:
    assert exact_fit.estimates[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)


def test_dropping_the_mechanism_term_is_detected() -> None:
    """A nonzero witness: the tilts move the mean, and their curves differ from ``Y - psi``.

    If the tilt's ``g`` term vanished, the curve of ``rr 0.5`` would equal the curve of a
    fixed regime with the same mean.  The witness checks that the oracle's curve carries a
    mechanism component: it is not a function of ``(W, A, Y)`` through ``Y`` alone.
    """
    for label in ("rr 0.5", "rr 2"):
        name = f"ey_rr_tilt[{label}]"
        assert abs(law.TRUTH[name] - law.TRUTH["ey_rr_tilt[natural course]"]) > 1e-3
        eif = law.eif(name)
        fixed_g = law.gateaux_eif(
            lambda p, label=label: _fixed_mechanism(p, law.RR_TILT_DELTAS[label])
        )
        assert np.max(np.abs(eif - fixed_g)) > 1e-3


def _fixed_mechanism(probs, delta: float):
    """The same mean with ``g`` frozen at the truth, the mutation the witness rules out."""
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2))
    p_wa = p.sum(axis=2)
    q = p[:, :, 1] / p_wa
    g = law.G_EXACT
    if delta <= 1.0:
        mean = g * (delta * q[:, 1] + (1.0 - delta) * q[:, 0]) + (1.0 - g) * q[:, 0]
    else:
        keep = 1.0 / delta
        mean = (1.0 - g) * (keep * q[:, 0] + (1.0 - keep) * q[:, 1]) + g * q[:, 1]
    return (p_w * mean).sum()
