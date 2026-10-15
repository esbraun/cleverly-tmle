"""Mutation controls of the ``IPSI`` repeated-outcome witnesses (X32b, step 4a).

A witness that never fails proves nothing.  Each control below applies one deliberate fault to
the library by ``monkeypatch`` and runs the witnesses that must see it.  The control passes
when each listed witness raises ``AssertionError``.  The same witness, unmutated, passes in
its own test module, so a failure here is the fault and not the witness.

The faults are those of X32a (M1 to M8) as they apply to the ``IPSI`` fit, and the
``IPSI`` half of the X32b controls (N3, N5, N7, N9).  A fault acts on a repeated outcome only:
the truncated end-of-study fit that W1 compares with is the reference and stays unmutated.

* ``M1``  the seed reads ``max_{s <= h} Y_s``, the absorbing reading.
* ``M2``  every horizon is seeded at ``Y_T``.
* ``M3``  ``covariate_history`` omits ``Y_s``.
* ``M5'`` the node-``h`` fit mask reads ``uncensored_through(h - 1)``.
* ``M6``  a working model seeds every cell at the last horizon.  The ``IPSI`` cells come
  from the ``IPSI`` recursion, so this fault and ``M2`` patch the same seed.
* ``M7``  the band treats the horizons as independent.
* ``M8``  block ``t`` includes ``Y_t``, which leaks the outcome into the node-``t`` regression
  and mechanisms.
* ``N3``  the ``IPSI`` mechanism term is dropped.
* ``N5``  the ``IPSI`` weights follow the plan through ``n_times`` and not the horizon.
* ``N7``  ``CausalStudy`` drops ``IncrementalMean.horizons``.
* ``N9``  the dropout factor ``1 / omega_s`` is replaced by one, and the indicator stays.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly import study as study_module
from cleverly.longitudinal import IPSI, sequential
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal.data import LongitudinalData

from .. import discrete_law_repeated as laws
from .test_influence_gateaux_repeated import (
    CONTRASTED,
    MULTIPLIERS,
    BaselineAndArm,
    _mean,
    check_e5,
    check_e6,
    check_e8,
    ipsi_fit,
)
from .test_repeated_law_values import OMEGA_GAP
from .test_repeated_outcome import COMPOSITIONS, TILTED, Composition, check_w1, fit, panel
from .test_repeated_outcome_container import GRIDS, check_w7_saturated, check_w7_trend
from .test_repeated_outcome_study_route import check_wc_incremental_horizons

# ``cleverly.longitudinal.ipsi`` and ``.sdr`` name the one-call functions in the package
# namespace, so the modules are imported by path.
ipsi_module = importlib.import_module("cleverly.longitudinal.ipsi")
sdr_module = importlib.import_module("cleverly.longitudinal.sdr")
msm_module = importlib.import_module("cleverly.longitudinal.msm")

Mutation = Callable[[pytest.MonkeyPatch], None]

# A fault that breaks the weights also trips the positivity warning of the fit it mutates.
pytestmark = pytest.mark.filterwarnings("ignore::cleverly.exceptions.PositivityWarning")

BINARY = laws.BINARY
CONTINUOUS = laws.CONTINUOUS
STATIC = COMPOSITIONS[0]
INCREMENTAL = TILTED[0]
IPSI_GRID = next(grid for grid in GRIDS if grid.kind == "IPSI")


def _patch_everywhere(monkeypatch: pytest.MonkeyPatch, name: str, replacement: Any) -> None:
    """Bind ``replacement`` wherever a module imported ``name``, as the library does."""
    for module in (sequential, ipsi_module, msm_module, sdr_module):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, replacement)


# ---------------------------------------------------------------- the faults


def absorbing_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """M1: the seed of horizon ``h`` is the running maximum of ``Y_1 .. Y_h``."""
    original = sequential.seed_carried

    def mutated(data: Any, scaler: Any, *, horizon: int) -> Any:
        if not data.is_repeated:
            return original(data, scaler, horizon=horizon)
        observed = data.uncensored_through(horizon)
        running = np.max(np.nan_to_num(data.repeated[:, :horizon], nan=-np.inf), axis=1)
        scaled = np.where(
            observed, scaler.scale(np.where(observed, running, 0.0)), sequential._FILLER
        )
        return np.clip(scaled, 0.0, 1.0)

    _patch_everywhere(monkeypatch, "seed_carried", mutated)


def seed_at_the_last_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """M2 and M6: every horizon, and every cell of a working model, is seeded at ``Y_T``."""
    original = sequential.seed_carried

    def mutated(data: Any, scaler: Any, *, horizon: int) -> Any:
        return original(data, scaler, horizon=data.n_times if data.is_repeated else horizon)

    _patch_everywhere(monkeypatch, "seed_carried", mutated)


def history_without_the_earlier_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """M3: ``covariate_history`` drops the columns that hold ``Y_s``."""
    original = LongitudinalData.covariate_history

    def mutated(self: LongitudinalData, time: int) -> Any:
        matrix = original(self, time)
        if not self.is_repeated:
            return matrix
        names = self.history_names(time)
        keep = [index for index, name in enumerate(names) if name not in self.repeated_names]
        return matrix[:, keep]

    monkeypatch.setattr(LongitudinalData, "covariate_history", mutated)


class _MasksReadingTheRiskSet:
    """The masks of one plan, with ``following(time)`` read as ``at_risk(time)``."""

    def __init__(self, masks: Any, time: int) -> None:
        self._masks = masks
        self._time = time

    def following(self, time: int) -> Any:
        return self._masks.at_risk(time) if time == self._time else self._masks.following(time)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._masks, name)


def fit_mask_through_the_previous_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """M5': the node-``h`` regression is fitted on the rows uncensored through ``h - 1``."""
    original = sequential._fit_node_regression

    def mutated(
        data: Any, plan: Any, carried: Any, time: int, horizon: int, *args: Any, **kwargs: Any
    ) -> Any:
        if data.is_repeated and time == horizon:
            masks = kwargs.pop("masks", None) or plan.masks(data)
            kwargs["masks"] = _MasksReadingTheRiskSet(masks, time)
        return original(data, plan, carried, time, horizon, *args, **kwargs)

    _patch_everywhere(monkeypatch, "_fit_node_regression", mutated)


def horizons_treated_as_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """M7: the band is built from curves whose second horizon is permuted across units."""
    original = estimator_module.simultaneous_bands

    def mutated(estimates: Any, **kwargs: Any) -> Any:
        n = len(next(iter(estimates.values())).influence_curve)
        order = np.random.default_rng(1).permutation(n)
        shuffled = {
            name: (
                dataclasses.replace(estimate, influence_curve=estimate.influence_curve[order])
                if "@ t=2" in name
                else estimate
            )
            for name, estimate in estimates.items()
        }
        return original(shuffled, **kwargs)

    monkeypatch.setattr(estimator_module, "simultaneous_bands", mutated)


def block_includes_the_current_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """M8: the history at node ``t`` holds ``Y_t``, so ``Y_t`` leaks into its own regression."""
    original = LongitudinalData.covariate_history

    def mutated(self: LongitudinalData, time: int) -> Any:
        matrix = original(self, time)
        if not self.is_repeated:
            return matrix
        return np.column_stack([matrix, np.nan_to_num(self.repeated[:, time - 1], nan=0.0)])

    monkeypatch.setattr(LongitudinalData, "covariate_history", mutated)


def mechanism_term_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """N3: every treatment node reads as declared known, which removes the mechanism term."""
    original = ipsi_module._recursion

    def mutated(data: Any, plan: Any, mechanism: Any, *args: Any, **kwargs: Any) -> Any:
        known = tuple(range(1, data.n_times + 1))
        return original(
            data, plan, dataclasses.replace(mechanism, known_treatment_nodes=known), *args, **kwargs
        )

    monkeypatch.setattr(ipsi_module, "_recursion", mutated)


class _MasksFollowingThroughTheEnd:
    """The masks of one plan, with ``following(time)`` read at the last node."""

    def __init__(self, masks: Any, last: int) -> None:
        self._masks = masks
        self._last = last

    def following(self, time: int) -> Any:
        return self._masks.following(self._last)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._masks, name)


def weights_through_n_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """N5: the weight at every node needs the unit to follow the plan through ``n_times``."""
    original = ipsi_module._weights

    def mutated(
        data: Any,
        plan: Any,
        mechanism: Any,
        masks: Any,
        propensity: Any,
        horizon: int,
        g_bounds: Any,
    ) -> Any:
        if data.is_repeated:
            masks = _MasksFollowingThroughTheEnd(masks, data.n_times)
        return original(data, plan, mechanism, masks, propensity, horizon, g_bounds)

    monkeypatch.setattr(ipsi_module, "_weights", mutated)


def incremental_horizons_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """N7: the route forgets ``horizons`` on an incremental request."""
    provider = study_module.ExplicitAdjustmentProvider
    original = provider._identify_longitudinal

    def mutated(self: Any, study: Any, estimand: Any) -> Any:
        if isinstance(estimand, (study_module.IncrementalMean, study_module.IncrementalEffect)):
            estimand = dataclasses.replace(estimand, horizons=None)
        return original(self, study, estimand)

    monkeypatch.setattr(provider, "_identify_longitudinal", mutated)


def dropout_factor_replaced_by_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """N9: the retention probability in the cumulative product is one; the indicator stays."""
    original = sequential.Mechanism.cumulative_with_unbounded

    def mutated(self: Any, data: Any, plan: Any, bounds: Any) -> Any:
        ones = tuple(
            {label: np.ones_like(value) for label, value in node.items()} for node in self.censoring
        )
        return original(dataclasses.replace(self, censoring=ones), data, plan, bounds)

    monkeypatch.setattr(sequential.Mechanism, "cumulative_with_unbounded", mutated)


# ---------------------------------------------------------------- the witnesses

Witness = Callable[[], None]


def w1(composition: Composition, horizon: int) -> Witness:
    """W1 on ``IPSI``: the repeated fit at ``horizon`` equals the truncated end-of-study fit."""

    def run() -> None:
        frame = panel(continuous=composition.continuous)
        repeated = fit(IPSI, composition, frame, 3, repeated=True)
        check_w1(IPSI, composition, frame, repeated, horizon)

    return run


def e5() -> None:
    for law in (BINARY, CONTINUOUS):
        check_e5(law, ipsi_fit(law))


def e6() -> None:
    for law in (BINARY, CONTINUOUS):
        check_e6(law, ipsi_fit(law, outcome=BaselineAndArm()))


def e8() -> None:
    check_e8(ipsi_fit(BINARY, simultaneous=True, n_multiplier=MULTIPLIERS, random_state=0))


def w7i() -> None:
    check_w7_saturated(IPSI_GRID)
    check_w7_trend(IPSI_GRID)


#: Each control: the fault, and the witnesses that must fail under it.  A W1 witness is
#: listed for each horizon where the fault acts.  M1 and M3 act where an earlier outcome
#: exists, so not at ``h = 1``.  M2 and N5 act where the horizon is before the last node.
def _both(horizons: tuple[int, ...]) -> dict[str, Witness]:
    return {
        f"W1[{composition.name}@h{horizon}]": w1(composition, horizon)
        for composition in (STATIC, INCREMENTAL)
        for horizon in horizons
    }


CONTROLS: dict[str, tuple[Mutation, dict[str, Witness]]] = {
    "M1": (absorbing_seed, {**_both((2, 3)), "E6": e6}),
    "M2": (seed_at_the_last_node, {**_both((1, 2)), "E6": e6}),
    "M3": (history_without_the_earlier_outcome, {**_both((2, 3)), "E5": e5}),
    "M5'": (fit_mask_through_the_previous_node, {**_both((1, 2, 3)), "E5": e5}),
    "M6": (seed_at_the_last_node, {"W7i": w7i}),
    "M7": (horizons_treated_as_independent, {"E8": e8}),
    "M8": (block_includes_the_current_outcome, {"E5": e5}),
    "N3": (mechanism_term_dropped, {"E5": e5}),
    "N5": (weights_through_n_times, {**_both((1, 2)), "E5": e5}),
    "N7": (incremental_horizons_dropped, {"WC": check_wc_incremental_horizons}),
    "N9": (dropout_factor_replaced_by_one, {"E5": e5, "E6": e6}),
}

PAIRS = [
    pytest.param(apply, witness, id=f"{control}-{label}")
    for control, (apply, witnesses) in CONTROLS.items()
    for label, witness in witnesses.items()
]

UNMUTATED = {
    label: witness for _, witnesses in CONTROLS.values() for label, witness in witnesses.items()
}


@pytest.mark.parametrize("label", sorted(UNMUTATED))
def test_every_witness_passes_on_the_unmutated_library(label: str) -> None:
    """The control of the controls: a witness that fails here would fail under any fault."""
    UNMUTATED[label]()


@pytest.mark.parametrize(("apply", "witness"), PAIRS)
def test_each_fault_fails_each_listed_witness(
    apply: Mutation, witness: Witness, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply(monkeypatch)
    with pytest.raises(AssertionError):
        witness()


# ---------------------------------------------------------------- how far N9 moves


@pytest.mark.parametrize("law", (BINARY, CONTINUOUS), ids=lambda law: law.name)
def test_n9_moves_the_curve_of_every_tilted_cell_by_at_least_one_half(
    law: laws.RepeatedLaw, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dropout factor reaches the curve at every plan and horizon, not at one cell.

    The estimate stays at the truth (single robustness in ``omega``), so only the curve
    can see the fault when ``m`` is saturated.
    """
    dropout_factor_replaced_by_one(monkeypatch)
    fitted = ipsi_fit(law)
    rows = law.first_row_of()
    for delta in CONTRASTED:
        for horizon in laws.HORIZONS:
            name = _mean(delta, horizon)
            assert fitted.psi(name) == pytest.approx(
                law.tilt_functional(law.probs, delta, horizon), abs=1e-12
            )
            move = np.max(
                np.abs(fitted.influence_curves[name][rows] - law.tilt_eif(delta, horizon))
            )
            assert move >= 0.5 - 1e-6, (delta, horizon)


@pytest.mark.parametrize("law", (BINARY, CONTINUOUS), ids=lambda law: law.name)
def test_n9_moves_the_w_only_estimate_at_the_second_horizon_by_the_declared_gap(
    law: laws.RepeatedLaw, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E6 sees N9 in the estimate: the miss equals the law-side gap of ``omega := 1``."""
    dropout_factor_replaced_by_one(monkeypatch)
    fitted = ipsi_fit(law, outcome=BaselineAndArm())
    for delta, gap in OMEGA_GAP[law.name].items():
        truth = law.tilt_functional(law.probs, delta, 2)
        assert fitted.psi(_mean(delta, 2)) - truth == pytest.approx(gap, abs=6e-7)
