"""Fixed-delta, changing-mechanism witnesses for incremental stress surfaces."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pytest

from cleverly import IncrementalEffect, IncrementalMean
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, IPSISet, Static
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from tests.unit._confounding_support import (
    alias_for,
    confounding_estimate,
    confounding_study,
    forbid_draw_and_refit,
    replacement,
    with_estimator,
    with_functional,
    with_key,
    with_last_nuisance,
    with_typed,
)

_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17))
_TILTS = (
    Incremental(0.5, name="down"),
    Incremental(1.0, name="natural"),
    Incremental(2.0, name="up"),
)


@cache
def _fit(
    *,
    contrast: bool = True,
    binary: bool = False,
    weighted: bool = False,
    strata: bool = False,
    repeats: int = 1,
    backend: str = "pandas",
    labels: bool = False,
) -> Any:
    target = (IncrementalEffect if contrast else IncrementalMean)(_TILTS, reference="natural")
    return confounding_estimate(
        confounding_study(
            binary=binary, weighted=weighted, strata=strata, backend=backend, labels=labels
        ),
        target,
        binary=binary,
        repeats=repeats,
    )


@pytest.mark.parametrize(
    "contrast,binary,weighted,strata,repeats,backend,labels",
    [
        (False, False, False, False, 1, "pandas", False),
        (True, False, True, False, 3, "polars", True),
        (False, True, True, False, 1, "pandas", True),
        (True, True, False, False, 1, "pandas", False),
    ],
)
def test_incremental_cells_equal_complete_fixed_delta_refits(
    contrast: bool,
    binary: bool,
    weighted: bool,
    strata: bool,
    repeats: int,
    backend: str,
    labels: bool,
) -> None:
    result = _fit(
        contrast=contrast,
        binary=binary,
        weighted=weighted,
        strata=strata,
        repeats=repeats,
        backend=backend,
        labels=labels,
    )
    alias = alias_for(result, value="up", stratum=("small",) if strata else None)
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.movement_scale == "estimate_difference"
    assert surface.n_repeats == repeats
    for cell in surface.cells[1:]:
        manual = result.estimator.refit(
            replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
            random_state=31,
        )
        assert cell.failure is None
        assert cell.estimate == pytest.approx(manual[alias].psi, abs=1e-12)
        assert cell.displacement == pytest.approx(manual[alias].psi - result[alias].psi, abs=1e-12)
        for nuisance in manual.nuisances:
            state = nuisance.incremental
            assert state.deltas == (0.5, 1.0, 2.0)
            assert state.names == ("down", "natural", "up")
            assert state.reference == 1.0
            g = nuisance.propensity.values[:, 1]
            for index, delta in enumerate(state.deltas):
                denominator = delta * g + 1 - g
                np.testing.assert_allclose(state.values[:, 1, index], delta * g / denominator)
                np.testing.assert_allclose(state.derivative[:, index], delta / denominator**2)
        assert manual[alias].psi == pytest.approx(
            np.median([draw.psi[alias] for draw in manual.repeats]), abs=1e-12
        )
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-4


def test_incremental_refits_rebuild_the_mechanism_and_reject_frozen_density_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit(contrast=False)
    alias = alias_for(result, value="up")
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    changed = replacement(result, surface, 0.22, 0.17)
    correct = result.estimator.refit(changed, random_state=31)
    assert (
        np.max(np.abs(correct.nuisance.incremental.values - result.nuisance.incremental.values))
        > 0.02
    )
    original_evaluate = IPSISet.evaluate.__func__
    baseline_g = result.nuisance.propensity.values

    def frozen(cls: Any, incrementals: Any, data: Any, propensity: Any, **kwargs: Any) -> Any:
        return original_evaluate(cls, incrementals, data, baseline_g, **kwargs)

    monkeypatch.setattr(IPSISet, "evaluate", classmethod(frozen))
    monkeypatch.setattr(IPSISet, "at", lambda self, _: self)
    wrong = result.estimator.refit(changed, random_state=31)
    assert abs(wrong[alias].psi - correct[alias].psi) > 1e-3


def test_incremental_repeats_persistence_and_surface_cache() -> None:
    result = _fit(repeats=3, weighted=True)
    alias = alias_for(result, value="up")
    expected = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    loaded = loads(dumps(result))
    assert simulated_confounding(loaded, estimand=alias, grid=_GRID, random_state=31) == expected
    kwargs = {"estimand": alias, "grid": _GRID, "random_state": 31}
    report = result.sensitivity.simulated_confounding(**kwargs)
    assert report is result.sensitivity.simulated_confounding(**kwargs)


def test_natural_incremental_mean_refuses_before_draw_and_is_not_advertised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit(contrast=False)
    alias = alias_for(result, value="natural")
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="delta=1 incremental mean"):
        simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    with pytest.raises(ValueError) as caught:
        simulated_confounding(result, estimand="absent", grid=_GRID, random_state=31)
    assert "ey_ipsi[natural]" not in str(caught.value)
    assert "ey_ipsi[up]" in str(caught.value)


def test_incremental_stratified_targeting_is_refused_upstream() -> None:
    with pytest.raises(NotImplementedError, match=r"baseline strata.*alternating targeting"):
        _fit(strata=True)


def test_a_sole_incremental_mean_uses_the_facade_selection() -> None:
    result = confounding_estimate(
        confounding_study(), IncrementalMean((Incremental(2.0, name="up"),))
    )
    surface = result.sensitivity.simulated_confounding(grid=_GRID, random_state=31)
    assert surface.estimand == "ey_ipsi[up]"
    assert all(cell.failure is None for cell in surface.cells)


def test_a_natural_only_fit_advertises_no_supported_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    result = confounding_estimate(
        confounding_study(), IncrementalMean((Incremental(1.0, name="natural"),))
    )
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(ValueError, match="reports none"):
        simulated_confounding(result, estimand="absent", grid=_GRID, random_state=31)


def test_incremental_contrast_preserves_a_non_natural_default_reference() -> None:
    result = confounding_estimate(confounding_study(), IncrementalEffect((_TILTS[0], _TILTS[2])))
    alias = alias_for(result, value="up")
    assert result.parameter_keys[alias].reference == "down"
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    manual = result.estimator.refit(replacement(result, surface, 0.22, 0.17), random_state=31)
    assert surface.cells[-1].estimate == pytest.approx(manual[alias].psi, abs=1e-12)


@pytest.mark.parametrize(
    "field",
    [
        "key-axis",
        "key-value",
        "key-reference",
        "key-alias",
        "key-horizon",
        "key-term",
        "functional-target",
        "functional-reference",
        "functional-declaration",
        "typed-reference",
        "typed-declaration",
        "estimator-reference",
        "estimator-declaration",
        "config-axis",
        "config-reference",
        "estimate-scale",
        "estimate-name",
        "state-values",
        "state-weights",
        "state-derivative",
        "state-propensity",
        "state-deltas",
        "state-names",
        "state-reference",
        "state-missing",
        "mixed-regime",
        "mixed-shift",
    ],
)
def test_incremental_provenance_corruption_refuses_before_draw(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit(repeats=3)
    alias = alias_for(result, value="up")
    part = field.split("-", 1)[1]
    if field.startswith("key-"):
        value: Any = {"horizon": 1, "term": "other"}.get(part, "wrong")
        result = with_key(result, alias, **{part: value})
    elif field.startswith("functional-"):
        result = with_functional(
            result,
            **{
                ("interventions" if part == "declaration" else part): ()
                if part == "declaration"
                else "wrong"
            },
        )
    elif field.startswith("typed-"):
        result = with_typed(
            result,
            **{
                ("interventions" if part == "declaration" else part): ()
                if part == "declaration"
                else "wrong"
            },
        )
    elif field.startswith("estimator-"):
        result = with_estimator(
            result,
            **{
                ("incremental" if part == "declaration" else part): ()
                if part == "declaration"
                else "wrong"
            },
        )
    elif field.startswith("config-"):
        changes = {"parameter_axis": "arm"} if part == "axis" else {"reference_arm": 0.0}
        result = replace(result, config=replace(result.config, **changes))
    elif field.startswith("estimate-"):
        estimate = replace(result[alias], **{part: "ratio" if part == "scale" else "wrong"})
        result = replace(result, estimates={**result.estimates, alias: estimate})
    elif field.startswith("state-"):
        state = result.repeats[-1].nuisance.incremental
        if part == "missing":
            state = None
        elif part in {"values", "weights", "derivative", "propensity"}:
            values = getattr(state, part).copy()
            values.flat[0] += 0.1
            state = replace(state, **{part: values})
        else:
            value = {
                "deltas": (0.7, 1.0, 2.0),
                "names": ("other", "natural", "up"),
                "reference": 0.0,
            }[part]
            state = replace(state, **{part: value})
        result = with_last_nuisance(result, incremental=state)
    elif part == "regime":
        result = with_estimator(result, interventions=(Static(1),))
    else:
        from cleverly.interventions import Shift

        result = with_estimator(result, shifts=(Shift(0.5, cap=3.0),))
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match=r"incremental|parameter"):
        simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)


def test_incremental_refit_failure_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fit()
    alias = alias_for(result, value="up")
    original = TMLE.refit
    calls = 0

    def refit(self: Any, data: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("deliberate incremental replay failure")
        return original(self, data, **kwargs)

    monkeypatch.setattr(TMLE, "refit", refit)
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    assert calls == 3
    assert surface.cells[1].estimate is None
    assert "deliberate incremental replay failure" in surface.cells[1].failure.message
    assert surface.cells[-1].failure is None
