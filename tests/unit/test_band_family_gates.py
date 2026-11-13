"""The band-family rule at each fit's own gate (Fix-2 plan 2.5), with stand-in statuses.

No builder sets ``"constant_arm_cell_plugin"`` or ``"out_of_range_plugin"`` yet (Fix-3 adds
them), so each test points the two multiplier constants at statuses that exist and stamps one
reported estimate with them as the fit builds its report.  Each fit's gate must then refuse the
whole band over a constant-cell member, and name it in ``summary()``, and keep an out-of-range
member in the band at its plug-in standard error.  Two mutation controls guard each rule: a gate
that drops the constant member instead builds a band over the others, and a band that reads
``std_error`` instead of ``plugin_std_error`` loses the out-of-range member's slice.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Callable
from functools import cache
from typing import Any

import pandas as pd
import pytest

from cleverly.datasets import make_binary_outcome
from cleverly.estimators import TMLE
from cleverly.inference import multiplier
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import structural as structural_module
from cleverly.study import _narrow_bands
from tests.conftest import FAST_KWARGS
from tests.unit import _longitudinal_repeats_support as support

pytestmark = pytest.mark.xdist_group("band_family_gates")

#: The module, not the ``tmle`` function the package exports under the same name.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

CONSTANT = "constant_node_plugin"
OUTSIDE = "stack_failure_plugin"


@pytest.fixture(autouse=True)
def stand_in_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiplier, "CONSTANT_ARM_CELL_STATUS", CONSTANT)
    monkeypatch.setattr(multiplier, "OUT_OF_RANGE_STATUS", OUTSIDE)


def _stamp(estimates: dict[str, Any], name: str, status: str) -> dict[str, Any]:
    out = dict(estimates)
    out[name] = dataclasses.replace(out[name], inference=status)
    return out


@cache
def _binary() -> pd.DataFrame:
    return make_binary_outcome(n=600, seed=11)[0]


def _tmle(monkeypatch: pytest.MonkeyPatch, name: str, status: str, **settings: Any) -> Any:
    original = TMLE._combine_repeats

    def stamped(self: Any, per_repeat: Any) -> dict[str, Any]:
        return _stamp(original(self, per_repeat), name, status)

    monkeypatch.setattr(TMLE, "_combine_repeats", stamped)
    frame = _binary()
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(**{**FAST_KWARGS, "random_state": 0, "simultaneous": True, **settings})
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def _ltmle(monkeypatch: pytest.MonkeyPatch, name: str, status: str, **settings: Any) -> Any:
    original = estimator_module._estimates

    def stamped(*args: Any, **kwargs: Any) -> Any:
        reported = original(*args, **kwargs)
        return dataclasses.replace(
            reported, estimates=_stamp(dict(reported.estimates), name, status)
        )

    monkeypatch.setattr(estimator_module, "_estimates", stamped)
    return support.ltmle(support.frame(), simultaneous=True, **settings)


def _structural(monkeypatch: pytest.MonkeyPatch, name: str, status: str, **settings: Any) -> Any:
    from tests.unit.test_structural_nested_mean import GLMS, REGIMENS, SIM, _clustered, snmm

    original = structural_module.StructuralNestedMean._estimates

    def stamped(self: Any, *args: Any, **kwargs: Any) -> Any:
        estimates, kinds = original(self, *args, **kwargs)
        return _stamp(dict(estimates), name, status), kinds

    monkeypatch.setattr(structural_module.StructuralNestedMean, "_estimates", stamped)
    configured = {
        **GLMS,
        "n_folds": 1,
        "n_multiplier": 400,
        "multiplier_kind": "rademacher",
        "simultaneous": True,
        **SIM,
        **settings,
    }
    return snmm(_clustered(), regimens=REGIMENS, **configured)


#: Each fit: the builder, the gate module whose ``constant_arm_members`` the mutant replaces,
#: and the band-family member it stamps.
FITS: dict[str, tuple[Callable[..., Any], Any, str]] = {
    "tmle": (_tmle, tmle_module, "ate"),
    "ltmle": (_ltmle, estimator_module, "ey_regimen[always]"),
    "structural": (_structural, structural_module, "ey_regimen[always]"),
}


@pytest.mark.parametrize("fit", sorted(FITS))
def test_a_constant_member_refuses_the_fit_band_and_the_summary_names_it(
    fit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build, _, name = FITS[fit]
    result = build(monkeypatch, name, CONSTANT)
    assert result[name].inference == CONSTANT
    assert result.simultaneous is None
    text = " ".join(str(result.summary(detail="full")).split())
    assert "reads a treatment arm" in text and name in text


@pytest.mark.parametrize("fit", sorted(FITS))
def test_a_gate_that_drops_the_constant_member_builds_another_family(
    fit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: without the gate the fit bands the remaining members."""
    build, module, name = FITS[fit]
    monkeypatch.setattr(module, "constant_arm_members", lambda estimates: ())
    result = build(monkeypatch, name, CONSTANT)
    assert result.simultaneous is not None
    assert name not in result.simultaneous.bands


@pytest.mark.parametrize("fit", sorted(FITS))
def test_an_out_of_range_member_stays_in_the_band_at_its_plugin_se(
    fit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build, _, name = FITS[fit]
    result = build(monkeypatch, name, OUTSIDE, band_scale="identity")
    band = result.simultaneous
    assert band is not None and name in band.bands
    low, high = band.bands[name]
    estimate = result[name]
    assert (high - low) / 2.0 == pytest.approx(
        band.critical_value * estimate.plugin_std_error, rel=1e-9
    )


@pytest.mark.parametrize("fit", sorted(FITS))
def test_a_band_on_std_error_loses_the_out_of_range_slice(
    fit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the withheld ``std_error`` gives the member no finite slice."""
    build, _, name = FITS[fit]
    monkeypatch.setattr(multiplier, "_band_std_error", lambda estimate: float(estimate.std_error))
    try:
        result = build(monkeypatch, name, OUTSIDE, band_scale="identity")
    except (ValueError, TypeError, ArithmeticError):
        return
    band = result.simultaneous
    assert band is None or not all(math.isfinite(value) for value in band.bands.get(name, ()))


def test_a_narrowed_study_band_refuses_a_constant_member() -> None:
    result = support.ltmle(support.frame(), simultaneous=True, band_scale="identity")
    names = list(result.simultaneous.bands)
    retained = {name: result[name] for name in names[:-1]}
    assert _narrow_bands(result, retained, method=None) is not None
    stamped = _stamp(retained, names[0], CONSTANT)
    assert _narrow_bands(result, stamped, method=None) is None


def test_a_narrowed_band_without_the_gate_drops_the_constant_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cleverly import study

    monkeypatch.setattr(study, "constant_arm_members", lambda estimates: ())
    result = support.ltmle(support.frame(), simultaneous=True, band_scale="identity")
    names = list(result.simultaneous.bands)
    stamped = _stamp({name: result[name] for name in names}, names[0], CONSTANT)
    narrowed = _narrow_bands(result, stamped, method=None)
    assert narrowed is not None and names[0] not in narrowed.bands


def test_a_bootstrap_band_refuses_a_constant_member(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _ltmle(monkeypatch, "ey_regimen[always]", CONSTANT, n_bootstrap=20)
    assert result.bootstrap_band is None


def test_a_bootstrap_band_without_the_gate_bands_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(estimator_module, "constant_arm_members", lambda estimates: ())
    result = _ltmle(monkeypatch, "ey_regimen[always]", CONSTANT, n_bootstrap=20)
    assert result.bootstrap_band is not None
