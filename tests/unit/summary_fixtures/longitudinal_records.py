"""Longitudinal and overlap records whose summaries the contract test checks (X89 T2).

The structural nested mean model and its model check, the g-formula simulation report, an
MSM fit's working-model record, an SNMM's positivity record, and the overlap reports of
regimes, policies and tilts. A record carries no estimates frame, so each fixture names a
reader in ``_record_readers`` that checks every printed number against the record's fields.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

from . import Case
from . import _record_readers as read

__all__ = ["FIXTURES"]


def _quiet(build: Any) -> Any:
    @functools.cache
    def run() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return build()

    run.__doc__ = build.__doc__
    return run


@_quiet
def snmm() -> Any:
    """A cross-fitted SNMM with two static plans, inside its validation row."""
    from tests.unit.test_structural_support import fit

    return fit()


@_quiet
def snmm_in_sample() -> Any:
    """The same SNMM fitted in sample, which no validation row covers."""
    from tests.unit.test_structural_support import fit

    return fit(n_folds=1)


@_quiet
def check() -> Any:
    """The SNMM's model check on two fold splits."""
    from tests.unit.test_structural_support import BOUNDS

    return snmm().check_model(q_bounds=BOUNDS, n_splits=2)


@_quiet
def structural_diagnostics() -> Any:
    """The SNMM's positivity record."""
    return snmm().diagnostics


@_quiet
def msm_fit() -> Any:
    """The working-model record of an MSM over plan duration."""
    from tests.unit.test_longitudinal_msm import DOSE, fitted_regimens

    return fitted_regimens(msm=DOSE).msm_fits[0]


@_quiet
def simulation() -> Any:
    """The simulation report of a parametric g-formula with one modelled covariate."""
    from .longitudinal import gformula

    return gformula().simulation


@_quiet
def simulation_set_dose() -> Any:
    """The simulation report of a g-formula that sets a continuous dose."""
    from tests.unit.test_parametric_gformula import (
        BOUNDS_F,
        ROLES_F,
        TestFixedDose,
        fit_with,
        panel_f,
    )

    frame = panel_f(200, seed=23, dose_scale=0.6)
    return fit_with(
        frame,
        TestFixedDose()._plans(),
        ROLES_F,
        covariate_types={"L2": "normal"},
        q_bounds=BOUNDS_F,
        n_draws=20,
        reference="central",
    ).simulation


@functools.cache
def _point(group: str) -> Any:
    from tests.unit.test_intervention_load_diagnostics import _fit_group

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit_group(group, n=300)


def regime_report() -> Any:
    """The overlap of three regimes, one a dynamic rule."""
    return _point("regime").diagnostics.support()


def regime_row() -> Any:
    """One regime's overlap."""
    return regime_report().regimes["screen"]


def policy_row() -> Any:
    """The overlap of a treatment policy that shifts a dose up."""
    return _point("mtp").diagnostics.support()["up"]


def tilt_row() -> Any:
    """The overlap of an odds tilt."""
    return _point("ipsi").diagnostics.support()["down"]


FIXTURES: dict[str, Any] = {
    "snmm": snmm,
    "snmm_in_sample": snmm_in_sample,
    "check": Case(check, fidelity=read.read_check),
    "structural_diagnostics": Case(
        structural_diagnostics, fidelity=read.read_structural_diagnostics
    ),
    "msm_fit": Case(msm_fit, names=("duration",), fidelity=read.read_msm_fit),
    "simulation": Case(simulation, names=("L2", "Y"), fidelity=read.read_gformula_report),
    "simulation_set_dose": Case(
        simulation_set_dose, names=("L2", "Y"), fidelity=read.read_gformula_report
    ),
    "regime_report": Case(regime_report, names=("screen",), fidelity=read.read_support_report),
    "regime_row": Case(regime_row, names=("screen",), fidelity=read.read_regime_support),
    "policy_row": Case(policy_row, names=("up",), fidelity=read.read_policy_support),
    "tilt_row": Case(tilt_row, names=("down",), fidelity=read.read_tilt_support),
}
