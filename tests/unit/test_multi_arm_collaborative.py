r"""Multi-arm evidence for the two collaborative point-treatment estimators.

The exact law fixes the plug-in parameters independently of the implementation.  The
nonzero reduced-set witness below covers equation (9)'s armwise denominator and sign,
which an exact nuisance law cannot see because ``Qr`` vanishes at the truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import CTMLE, DRTMLE
from cleverly.estimators.reduced import ReducedSet
from cleverly.fluctuation.reduced import reduced_mechanism_covariate
from tests import discrete_law_multi as law

COMMON = {
    "outcome_learner": law.OracleMultiOutcome(),
    "treatment_learner": law.OracleMultiTreatment(),
    "cross_fit": False,
    "simultaneous": False,
    "estimands": ("ey", "ate"),
    "random_state": 0,
}


@pytest.fixture(scope="module")
def oat_fit():
    return CTMLE(strategy="oat", **COMMON).fit(law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def dr_fit():
    return (
        DRTMLE(
            reduced_outcome_learner="glm",
            reduced_treatment_learner="glm",
            **COMMON,
        )
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )


@pytest.mark.parametrize(
    "oracle",
    ("ey[0]", "ey[1]", "ey[2]", "ate[0 vs 2]", "ate[1 vs 2]"),
)
def test_oat_recovers_each_exact_law_parameter(oat_fit, oracle: str) -> None:
    reported = law.reported_name(oracle)
    assert oat_fit.estimates[reported].psi == pytest.approx(
        float(law.functional(law.PROBS, oracle)), abs=2e-15
    )


def test_oat_records_the_shared_treatment_model_api(oat_fit) -> None:
    record = oat_fit.extra["ctmle"]
    assert record.strategy == "oat"
    assert record.treatment_features == ("Qbar[high]", "Qbar[low]", "Qbar[mid]")
    assert record.treatment_risk_selected == record.treatment_risk
    assert oat_fit.nuisance.propensity.values.shape == (law.N, law.K)
    np.testing.assert_allclose(oat_fit.nuisance.propensity.values.sum(axis=1), 1.0, atol=1e-15)


def test_oat_refuses_selector_only_controls() -> None:
    with pytest.raises(ValueError, match="do not apply to strategy='oat'"):
        CTMLE(strategy="oat", penalty=False)


@pytest.mark.parametrize(
    "oracle",
    ("ey[0]", "ey[1]", "ey[2]", "ate[0 vs 2]", "ate[1 vs 2]"),
)
def test_drtmle_recovers_each_exact_law_parameter(dr_fit, oracle: str) -> None:
    reported = law.reported_name(oracle)
    assert dr_fit.estimates[reported].psi == pytest.approx(
        float(law.functional(law.PROBS, oracle)), abs=2e-15
    )


def test_drtmle_solves_and_reports_every_armwise_correction(dr_fit) -> None:
    fluctuation = dr_fit.fluctuations["mean"]
    assert fluctuation.mechanism is not None
    assert fluctuation.mechanism.propensity.shape == (law.N, law.K)
    assert fluctuation.mechanism.score.shape == (law.K,)
    assert dr_fit.validation.score_check().passed
    check = dr_fit.validation.correction_check()
    assert check.passed
    assert {row.arm for row in check.rows} == set(dr_fit.nuisance.arms)


def test_multi_arm_single_guard_uses_the_initial_mechanism_matrix() -> None:
    fit = (
        DRTMLE(
            guard=("g",),
            reduced_outcome_learner="glm",
            reduced_treatment_learner="glm",
            **COMMON,
        )
        .fit(law.frame(), outcome="Y", treatment="A")
        .single()
    )
    assert fit.fluctuations["mean"].mechanism is None
    assert fit.validation.correction_check().passed


def test_multi_arm_reduced_mechanism_covariate_has_the_r_formula_and_a_sign_witness() -> None:
    propensity = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])
    qr = np.array([[0.12, -0.06, 0.15], [-0.04, 0.21, -0.08]])
    reduced = ReducedSet(
        qr=qr,
        gr1=np.full_like(qr, 0.5),
        gr2=np.zeros_like(qr),
        arms=(0.0, 1.0, 2.0),
        g_bounds=(0.05, 0.95),
    )
    actual = reduced_mechanism_covariate(reduced, propensity, bounds=(0.05, 0.95))
    expected = qr / propensity
    np.testing.assert_array_equal(actual, expected)

    # A binary-style negative sign on any arm is a plausible but wrong extension.  The
    # nonzero witness makes that mutation observable where an exact-law check cannot.
    mutated = expected.copy()
    mutated[:, 0] *= -1.0
    assert not np.array_equal(actual, mutated)
