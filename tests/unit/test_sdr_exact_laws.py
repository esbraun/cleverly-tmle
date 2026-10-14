r"""T1: the SDR estimator on every exact longitudinal law, against its oracle and LTMLE.

Each law module realises its law exactly in the sample, and each test module beside it fits
LTMLE with saturated learners and compares the fit with a closed-form truth and a
complex-step Gateaux derivative.  With saturated nuisances every node regression is exact,
every residual has mean zero within its cell, and the transform :math:`\varphi_1` averages
to the g-formula.  So an SDR fit must report the same truth and the same efficient
influence function, to the same precision.

The fits reuse each sibling module's own fit helper with its engine swapped for
:class:`~cleverly.longitudinal.SDR`.  The settings, the columns and the oracle are then
those the LTMLE tests already hold to, and none is restated here.  ``SDR`` refuses
``alpha``, ``max_iter`` and ``tol``, and no helper passes them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE, SDR

from .. import discrete_law_competing as competing
from .. import discrete_law_longitudinal as binary
from .. import discrete_law_longitudinal_mtp as mtp_law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import discrete_law_longitudinal_policy as policy_law
from .. import discrete_law_survival as survival
from ..studies.canonical_ltmle import declared_regimens
from . import test_influence_gateaux_competing as competing_tests
from . import test_influence_gateaux_longitudinal as binary_tests
from . import test_influence_gateaux_longitudinal_msm as msm_tests
from . import test_influence_gateaux_longitudinal_mtp as mtp_tests
from . import test_influence_gateaux_longitudinal_policy as policy_tests
from . import test_influence_gateaux_survival as survival_tests
from . import test_weighted_estimand_longitudinal as weighted_tests

# The exact grid laws have recorded point masses at every dose, which R-b4 of X40 refuses;
# the fixture turns that refusal off and leaves the ratio code as shipped.
pytestmark = pytest.mark.usefixtures("exact_grid_law")

#: The window every comparison below is held to.  The LTMLE modules hold the same laws to
#: between ``1e-14`` and ``1e-10``; SDR adds one residual sum per node and no solve.
POINT = 1e-12
CURVE = 1e-10


@contextmanager
def _engine(module: ModuleType) -> Iterator[None]:
    """Run ``module``'s fit helper with :class:`SDR` in place of :class:`LTMLE`."""
    original = module.LTMLE
    module.LTMLE = SDR  # type: ignore[attr-defined]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        module.LTMLE = original  # type: ignore[attr-defined]


def _binary(engine: type = SDR) -> Any:
    """The fit of ``test_influence_gateaux_longitudinal``'s fixture, under ``engine``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine(
            declared_regimens(binary.REGIMEN_SPEC),
            reference=binary.REGIMEN_REFERENCE,
            outcome_learner=binary.CellMeans(),
            pseudo_learner=binary.CellMeans(),
            treatment_learner=binary.CellMeans(),
            censoring_learner=binary.CellMeans(),
            n_folds=1,
            g_bounds=binary_tests.NO_TRUNCATION,
            simultaneous=False,
        ).fit(
            binary.frame(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


def _survival() -> Any:
    with _engine(survival_tests):
        return survival_tests._oracle_fit(survival.frame())


def _competing() -> Any:
    with _engine(competing_tests):
        return competing_tests._oracle_fit(competing.frame())


def _multivalue() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SDR(
            declared_regimens(multivalue.REGIMEN_SPEC),
            reference=multivalue.REFERENCE,
            outcome_learner=binary.CellMeans(),
            pseudo_learner=binary.CellMeans(),
            treatment_learner=multivalue.CellProbabilities(),
            n_folds=1,
            g_bounds=(1e-8, 1.0 - 1e-8),
            simultaneous=False,
        ).fit(
            multivalue.frame(),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
        )


def _policy() -> Any:
    with _engine(policy_tests):
        return policy_tests.fit_plans()


def _mtp() -> Any:
    with _engine(mtp_tests):
        return mtp_tests.fit()


#: ``law -> (fit, [(name, truth, eif at one row per support point, rows)])``.
CASES: dict[str, Callable[[], tuple[Any, list[tuple[str, float, np.ndarray, np.ndarray]]]]] = {}


def _case(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def register(builder: Callable[[], Any]) -> Callable[[], Any]:
        CASES[name] = builder
        return builder

    return register


@_case("end_of_study")
def _end_of_study_case() -> Any:
    rows = binary.first_row_of()
    return _binary(), [(n, binary.TRUTH[n], binary.eif(n), rows) for n in binary.NAMES]


@_case("survival")
def _survival_case() -> Any:
    rows = survival.first_row_of()
    return _survival(), [(n, survival.TRUTH[n], survival.eif(n), rows) for n in survival.NAMES]


@_case("competing")
def _competing_case() -> Any:
    rows = competing.first_row_of()
    return _competing(), [(n, competing.TRUTH[n], competing.eif(n), rows) for n in competing.NAMES]


@_case("multivalue")
def _multivalue_case() -> Any:
    rows = multivalue.first_row_of()
    return _multivalue(), [
        (n, multivalue.TRUTH[n], multivalue.eif_at(multivalue.PROBS, n), rows)
        for n in multivalue.NAMES
    ]


@_case("policy")
def _policy_case() -> Any:
    rows = policy_tests.ROWS
    return _policy(), [
        (n, policy_law.TRUTH[n], policy_tests.eif(n), rows) for n in policy_law.NAMES
    ]


@_case("mtp")
def _mtp_case() -> Any:
    rows = mtp_tests.ROWS
    return _mtp(), [
        (n, mtp_law.truth(n), mtp_law.eif(n), rows) for n in mtp_tests.names(mtp_tests.UNIT_PLANS)
    ]


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request: Any) -> tuple[str, Any, list[tuple[str, float, np.ndarray, np.ndarray]]]:
    fit, expected = CASES[request.param]()
    return request.param, fit, expected


def test_every_reported_parameter_is_checked(case: Any) -> None:
    """The comparison covers what the fit reports, and the fit is an SDR fit."""
    _, fit, expected = case
    assert fit.config.estimator == "sdr"
    assert set(fit.estimates) == {name for name, *_ in expected}


def test_the_estimate_is_the_g_formula(case: Any) -> None:
    _, fit, expected = case
    for name, truth, _, _ in expected:
        assert fit.psi(name) == pytest.approx(truth, abs=POINT), name


def test_the_curve_is_the_gateaux_derivative(case: Any) -> None:
    _, fit, expected = case
    for name, _, eif, rows in expected:
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], eif, atol=CURVE, rtol=0, err_msg=name
        )


def test_no_node_fits_a_fluctuation(case: Any) -> None:
    """Every step records its transform and no fluctuation."""
    _, fit, _ = case
    for regimen_fit in fit.fits.values():
        for step in regimen_fit.steps:
            assert step.fluctuation is None
            assert step.transform is not None
            np.testing.assert_array_equal(step.targeted, step.initial)


def test_the_estimate_is_the_mean_of_the_first_transform(case: Any) -> None:
    """``psi`` is the weighted mean of ``phi_1``, read back off the steps."""
    _, fit, _ = case
    weights = fit.data.weights
    for regimen_fit in fit.fits.values():
        first = regimen_fit.steps[0]
        assert first.time == 1
        assert regimen_fit.psi_scaled == pytest.approx(
            float(np.average(first.transform, weights=weights)), abs=1e-14
        )


# ------------------------------------------------------------------ weighted law


@pytest.mark.parametrize("label", sorted(weighted_tests.WEIGHT_FUNCTIONS))
def test_the_weighted_estimate_and_curve_are_the_tilted_oracle(label: str) -> None:
    """M9's witness: the weights reach the regressions and the mean."""
    cells = binary.cell_weights(weighted_tests.WEIGHT_FUNCTIONS[label])
    with _engine(weighted_tests):
        fit = weighted_tests._fit(binary.row_weights(cells))
    rows = binary.first_row_of()
    for name in weighted_tests.NAMES:
        truth = float(binary.weighted_functional(binary.PROBS, name, cells))
        assert fit.psi(name) == pytest.approx(truth, abs=POINT), name
        np.testing.assert_allclose(
            fit.influence_curves[name][rows],
            binary.weighted_eif(name, cells),
            atol=CURVE,
            rtol=0,
            err_msg=name,
        )


# ------------------------------------------------------------------ working models


@pytest.mark.parametrize("family", ["identity", "saturated", "logit", "log"])
def test_the_working_model_projects_the_transforms(family: str) -> None:
    """Each MSM family's coefficients and curve, from the per-cell transforms.

    On the exact law every cell's transform averages to the cell's truth, so a logit or log
    projection converges, and its curve is the projection's Gateaux derivative.
    """
    with _engine(msm_tests):
        fit = msm_tests.oracle_fit(family)
    rows = binary.first_row_of()
    for name in binary.MSM_NAMES[family]:
        reported = name.replace(f"msm_regimen_{family}[", "msm_regimen[")
        assert fit[reported].psi == pytest.approx(
            binary.MSM_TRUTH[name], abs=msm_tests.POINT_TOLERANCE[family]
        )
        np.testing.assert_allclose(
            fit.influence_curves[reported][rows],
            binary.eif(name),
            atol=msm_tests.CURVE_TOLERANCE[family],
            rtol=0,
        )


# ------------------------------------------------------------------ agreement with LTMLE


def test_in_sample_sdr_equals_ltmle_at_exact_nuisances() -> None:
    """At exact nuisances both estimators are the g-formula, so they agree with each other."""
    sdr = _binary()
    ltmle = _binary(LTMLE)
    assert isinstance(ltmle, type(sdr)) and ltmle.config.estimator == "tmle"
    for name in binary.NAMES:
        assert sdr.psi(name) == pytest.approx(ltmle.psi(name), abs=POINT)
        np.testing.assert_allclose(
            sdr.influence_curves[name], ltmle.influence_curves[name], atol=CURVE, rtol=0
        )
    assert LTMLE is not SDR
