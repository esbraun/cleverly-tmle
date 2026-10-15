"""Mutation controls of the ``SDR`` repeated-outcome witnesses (X32b, step 4b).

A witness that never fails proves nothing.  Each control below applies one deliberate fault to
the library by ``monkeypatch`` and runs the witnesses that must see it.  The control passes
when each listed witness raises ``AssertionError`` that names the expected reason.  The same
witness, unmutated, passes in its own test module, so a failure here is the fault and not the
witness.

The faults are those of X32a (M1 to M8) as they apply to the ``SDR`` fit, and the ``SDR``
half of the X32b controls (N4, N8', N9).  A seed or history fault acts on a repeated outcome
only: the truncated end-of-study fit that W1 compares with is the reference and stays
unmutated.

* ``M1``  the seed reads ``max_{s <= h} Y_s``, the absorbing reading.
* ``M2``  every horizon is seeded at ``Y_T``.
* ``M3``  ``covariate_history`` omits ``Y_s``.
* ``M5'`` the node-``h`` fit mask reads ``uncensored_through(h - 1)``.
* ``M6``  the working-model projection takes, for every cell, the fit of the last horizon of
  its label.  The cells come from the ``SDR`` recursion and not from ``msm.py``, so a seed
  fault would repeat ``M2``.  This fault patches ``project_sdr_cells``.
* ``M7``  the band treats the horizons as independent.
* ``M8``  block ``t`` includes ``Y_t``, which leaks the outcome into the node-``t`` regression
  and mechanisms.
* ``N4``  the correction is dropped: the transform of every node is the plug-in.
* ``N8'`` the correction is scaled by 0.9 at every node, so a later term carries 0.9 once for
  each node it passes.
* ``N9``  the censoring factor ``1 / c_s`` of the ratio is replaced by one, and the indicator
  stays.
* ``NS``  node ``t`` is regressed on the plug-in ``v_{t+1}`` and not on the transform
  ``phi_{t+1}``: the doubly robust estimator that is not sequentially so.  The plan lists no
  mutation for E2b.  This control shows that E2b sees the difference.

A fault that acts at every cell must fail the witness at every cell.  The cell witnesses below
check each (law, plan, horizon) cell on its own, and fail the control if one cell escapes.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest

from cleverly.longitudinal import SDR, sequential
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal.data import LongitudinalData

from .. import discrete_law_repeated as laws
from .test_influence_gateaux_repeated import (
    SDR_CELLS,
    SDR_PLANS,
    ArmAtNodeOne,
    Baseline,
    InterceptAtNodeOne,
    check_e1,
    check_e2,
    check_e2b,
    check_e3,
    check_e8,
    known_mechanisms,
    sdr_fit,
)
from .test_repeated_law_values import N8_MOVE, N9_MOVE, NON_SEQUENTIAL_GAP, PLUG_IN_GAP
from .test_repeated_outcome import COMPOSITIONS, Composition, check_w1, fit, panel
from .test_repeated_outcome_container import GRIDS, check_w7_saturated, check_w7_trend

# ``cleverly.longitudinal.sdr`` names the one-call function in the package namespace, so the
# modules are imported by path.
ipsi_module = importlib.import_module("cleverly.longitudinal.ipsi")
sdr_module = importlib.import_module("cleverly.longitudinal.sdr")
msm_module = importlib.import_module("cleverly.longitudinal.msm")

Mutation = Callable[[pytest.MonkeyPatch], None]
Witness = Callable[[], None]

# A fault that breaks the weights also trips the positivity warning of the fit it mutates.
pytestmark = pytest.mark.filterwarnings("ignore::cleverly.exceptions.PositivityWarning")

LAWS = (laws.BINARY, laws.CONTINUOUS)
STATIC = COMPOSITIONS[0]
RULE = next(c for c in COMPOSITIONS if c.name == "rule_reads_y1")
FOLDS = next(c for c in COMPOSITIONS if c.name == "static_folds")
CLUSTERS = next(c for c in COMPOSITIONS if c.name == "clusters")
SDR_GRID = next(grid for grid in GRIDS if grid.kind == "SDR")


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
    """M2: every horizon is seeded at ``Y_T``."""
    original = sequential.seed_carried

    def mutated(data: Any, scaler: Any, *, horizon: int) -> Any:
        return original(data, scaler, horizon=data.n_times if data.is_repeated else horizon)

    _patch_everywhere(monkeypatch, "seed_carried", mutated)


def msm_cells_from_the_last_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """M6: each cell of the working model projects the fit of its label's last horizon."""
    original = msm_module.project_sdr_cells

    def mutated(data: Any, model: Any, fits: Any, scaler: Any, **kwargs: Any) -> Any:
        last: dict[str, Any] = {}
        for cell, cell_fit in zip(model.cells, fits, strict=True):
            if cell.label not in last or cell.horizon > last[cell.label][0]:
                last[cell.label] = (cell.horizon, cell_fit)
        moved = [last[cell.label][1] for cell in model.cells]
        return original(data, model, moved, scaler, **kwargs)

    monkeypatch.setattr(msm_module, "project_sdr_cells", mutated)
    monkeypatch.setattr(estimator_module, "project_sdr_cells", mutated)


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


def _scaled_correction(scale: float) -> Mutation:
    """A fault that multiplies the correction of every node by ``scale``.

    The correction of node ``k`` is the ratio factor ``pi_k / p_k``, so ``W_{t,s}`` carries
    ``scale ** (s - t + 1)``.  That is the scaling of the law-side transform
    ``phi_t = m_t + scale * r_t * (phi_{t+1} - m_t)``, whose moves the tests pin.
    """

    def apply(monkeypatch: pytest.MonkeyPatch) -> None:
        original = sdr_module.partial_weights

        def mutated(*args: Any, **kwargs: Any) -> Any:
            weights, paths, binding = original(*args, **kwargs)
            return [scale ** (k + 1) * w for k, w in enumerate(weights)], paths, binding

        monkeypatch.setattr(sdr_module, "partial_weights", mutated)

    return apply


#: N4: the correction is dropped, so the transform of every node is the plug-in ``v_t``.
correction_dropped = _scaled_correction(0.0)
#: N8': the correction is scaled by 0.9 at every node.
correction_scaled = _scaled_correction(0.9)


def censoring_factor_replaced_by_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """N9: the retention probability in the ratio is one; the indicator ``1{C_s = 1}`` stays."""
    original = sdr_module.node_factors

    def mutated(data: Any, plan: Any, mechanism: Any) -> Any:
        ones = tuple(
            {label: np.ones_like(value) for label, value in node.items()}
            for node in mechanism.censoring
        )
        return original(data, plan, dataclasses.replace(mechanism, censoring=ones))

    monkeypatch.setattr(sdr_module, "node_factors", mutated)


def node_regressed_on_the_plug_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """NS: the regression at node ``t < h`` is fitted to the plug-in ``v_{t+1}``.

    A recursion calls the node regression from ``t = h`` down to ``1``.  The wrapper keeps the
    plug-in of the call it just made and hands it to the next one in place of the transform.
    """
    original = sdr_module._fit_node_regression
    kept: dict[str, Any] = {}

    def mutated(
        data: Any, plan: Any, carried: Any, time: int, horizon: int, *args: Any, **kwargs: Any
    ) -> Any:
        if time < horizon:
            carried = kept["plug_in"]
        regression = original(data, plan, carried, time, horizon, *args, **kwargs)
        value = (
            regression.initial
            if regression.initial_marginal is None
            else regression.initial_marginal
        )
        kept["plug_in"] = np.where(regression.at_risk, value, sequential._FILLER)
        return regression

    monkeypatch.setattr(sdr_module, "_fit_node_regression", mutated)


# ---------------------------------------------------------------- the witnesses


def w1(composition: Composition, horizon: int) -> Witness:
    """W1 on ``SDR``: the repeated fit at ``horizon`` equals the truncated end-of-study fit."""

    def run() -> None:
        frame = panel(continuous=composition.continuous)
        repeated = fit(SDR, composition, frame, 3, repeated=True)
        check_w1(SDR, composition, frame, repeated, horizon)

    return run


def _saturated(law: laws.RepeatedLaw) -> Any:
    return sdr_fit(law)


def _w_only(law: laws.RepeatedLaw) -> Any:
    frame, roles = known_mechanisms(law)
    return sdr_fit(law, frame=frame, roles=roles, outcome=Baseline())


def _mixed(law: laws.RepeatedLaw) -> Any:
    return sdr_fit(
        law, outcome=Baseline(), treatment=InterceptAtNodeOne(), censoring=ArmAtNodeOne()
    )


def e1() -> None:
    for law in LAWS:
        check_e1(law, _saturated(law))


def e1_static() -> None:
    """E1 for the two static plans, whose fit does not read ``Y_s``.

    A rule that reads ``Y_1`` raises when the history omits it (M3), so the plans of this
    witness are the static ones.
    """
    for law in LAWS:
        fitted = sdr_fit(law, regimens={"never": 0, "always": 1})
        check_e1(law, fitted, [n for n in law.names if "treat_if_high" not in n])


def e2() -> None:
    for law in LAWS:
        check_e2(law, _w_only(law))


def e2b() -> None:
    for law in LAWS:
        check_e2b(law, _mixed(law))


def e3() -> None:
    for law in LAWS:
        check_e3(law, _w_only(law))


def _at_every(
    reason: str,
    cells: Iterable[Any],
    build: Callable[[laws.RepeatedLaw], Any],
    check: Callable[[laws.RepeatedLaw, Any, Any], None],
) -> Witness:
    """A witness that fails only when ``check`` fails at each (law, cell) on its own.

    Each failure must name ``reason``.  A cell that passes under the fault escapes, and the
    witness raises an ``AssertionError`` that names no reason, so the control fails.
    """
    cells = tuple(cells)

    def run() -> None:
        escaped = []
        for law in LAWS:
            built = build(law)
            for cell in cells:
                try:
                    check(law, built, cell)
                except AssertionError as error:
                    assert reason in str(error), (law.name, cell, str(error)[:200])
                else:
                    escaped.append((law.name, cell))
        assert not escaped, f"escaped the fault: {escaped}"
        raise AssertionError(f"{reason} at every cell")

    return run


def e1_every_cell(reason: str) -> Witness:
    """E1 at each reported mean and contrast, on its own."""
    assert laws.BINARY.names == laws.CONTINUOUS.names
    return _at_every(
        reason, laws.BINARY.names, _saturated, lambda law, built, name: check_e1(law, built, [name])
    )


def e2_second_horizon(reason: str) -> Witness:
    """E2 at each plan of ``h = 2``, where the ``W``-only regression is wrong."""
    return _at_every(
        reason,
        [cell for cell in SDR_CELLS if cell[1] == 2],
        _w_only,
        lambda law, built, cell: check_e2(law, built, [cell]),
    )


def e2_every_cell(reason: str) -> Witness:
    return _at_every(
        reason, SDR_CELLS, _w_only, lambda law, built, cell: check_e2(law, built, [cell])
    )


def e3_every_cell(reason: str) -> Witness:
    """E3 at each plan and horizon, including ``h = 1``."""
    return _at_every(
        reason, SDR_CELLS, _w_only, lambda law, built, cell: check_e3(law, built, [cell])
    )


def w7s_saturated() -> None:
    check_w7_saturated(SDR_GRID)


def w7s_trend() -> None:
    check_w7_trend(SDR_GRID)


def e8() -> None:
    check_e8(sdr_fit(laws.BINARY, simultaneous=True, n_multiplier=2000, random_state=0))


def _w1_witnesses(
    horizons: tuple[int, ...], compositions: tuple[Composition, ...] = (STATIC, RULE, FOLDS)
) -> dict[str, Witness]:
    return {
        f"W1[{composition.name}@h{horizon}]": w1(composition, horizon)
        for composition in compositions
        for horizon in horizons
    }


#: Each control: the fault, and the witnesses that must fail under it, each with the reason
#: its first failing assertion names.  A witness checks the estimate before the curve, so a
#: fault that leaves the estimate exact fails on the curve.  Pinning the reason keeps a
#: control from passing because of a precondition or a crash that raises ``AssertionError``
#: for another reason.  A W1 witness is listed for each horizon where the fault acts: M1 and
#: M3 act where an earlier outcome exists, so not at ``h = 1``, and M2 acts where the horizon
#: is before the last node.
ESTIMATE, CURVE, BAND = "estimate:", "curve:", "critical value"


def _all(reason: str, witnesses: dict[str, Witness]) -> dict[str, tuple[Witness, str]]:
    return {label: (witness, reason) for label, witness in witnesses.items()}


CONTROLS: dict[str, tuple[Mutation, dict[str, tuple[Witness, str]]]] = {
    "M1": (absorbing_seed, _all(ESTIMATE, {**_w1_witnesses((2, 3)), "E2": e2})),
    "M2": (seed_at_the_last_node, _all(ESTIMATE, {**_w1_witnesses((1, 2)), "E2": e2})),
    "M3": (
        history_without_the_earlier_outcome,
        _all(ESTIMATE, {**_w1_witnesses((2, 3), (STATIC, CLUSTERS)), "E1": e1_static}),
    ),
    "M5'": (
        fit_mask_through_the_previous_node,
        {**_all(ESTIMATE, _w1_witnesses((1, 2, 3))), "E1": (e1, CURVE)},
    ),
    "M6": (
        msm_cells_from_the_last_horizon,
        _all(ESTIMATE, {"W7s-cells": w7s_saturated, "W7s-trend": w7s_trend}),
    ),
    "M7": (horizons_treated_as_independent, _all(BAND, {"E8": e8})),
    "M8": (block_includes_the_current_outcome, _all(ESTIMATE, {"E1": e1})),
    "N4": (
        correction_dropped,
        {"E2": (e2_second_horizon(ESTIMATE), ESTIMATE), "E3": (e3_every_cell(CURVE), CURVE)},
    ),
    "N8'": (
        correction_scaled,
        {"E2": (e2_second_horizon(ESTIMATE), ESTIMATE), "E3": (e3_every_cell(CURVE), CURVE)},
    ),
    "NS": (node_regressed_on_the_plug_in, _all(ESTIMATE, {"E2b": e2b})),
    "N9": (
        censoring_factor_replaced_by_one,
        {
            "E1": (e1_every_cell(CURVE), CURVE),
            "E2": (e2_second_horizon(ESTIMATE), ESTIMATE),
            "E3": (e3_every_cell(CURVE), CURVE),
        },
    ),
}

PAIRS = [
    pytest.param(apply, witness, reason, id=f"{control}-{label}")
    for control, (apply, witnesses) in CONTROLS.items()
    for label, (witness, reason) in witnesses.items()
]

UNMUTATED = {
    label: witness
    for _, witnesses in CONTROLS.values()
    for label, (witness, _) in witnesses.items()
    if not label.startswith(("E1", "E2", "E3"))
} | {"E1": e1, "E1_static": e1_static, "E2": e2, "E2b": e2b, "E3": e3}


@pytest.mark.parametrize("label", sorted(UNMUTATED))
def test_every_witness_passes_on_the_unmutated_library(label: str) -> None:
    """The control of the controls: a witness that fails here would fail under any fault."""
    UNMUTATED[label]()


@pytest.mark.parametrize("cells", [e1_every_cell, e2_second_horizon, e2_every_cell, e3_every_cell])
def test_a_cell_witness_fails_when_the_library_is_unmutated(cells: Any) -> None:
    """Without a fault each cell passes, so the cell witness reports the escape and not a pass."""
    with pytest.raises(AssertionError, match="escaped the fault"):
        cells(ESTIMATE)()


@pytest.mark.parametrize(("apply", "witness", "reason"), PAIRS)
def test_each_fault_fails_each_listed_witness(
    apply: Mutation, witness: Witness, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply(monkeypatch)
    with pytest.raises(AssertionError, match=reason):
        witness()


# ---------------------------------------------------------------- how far each fault moves


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_ns_moves_the_mixed_pattern_estimate_by_the_declared_non_sequential_gap(
    law: laws.RepeatedLaw, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutated fit is the non-sequential estimator of E2b: it misses by the declared gap."""
    node_regressed_on_the_plug_in(monkeypatch)
    fitted = _mixed(law)
    for plan, gap in zip(SDR_PLANS, NON_SEQUENTIAL_GAP[law.name], strict=True):
        name = f"ey_regimen[{plan} @ t=2]"
        assert fitted.psi(name) - law.truth[name] == pytest.approx(gap, abs=6e-7), plan
        assert abs(gap) > 1e-4, plan
        first = f"ey_regimen[{plan} @ t=1]"
        assert fitted.psi(first) == pytest.approx(law.truth[first], abs=1e-12), plan


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_n4_moves_the_w_only_estimate_at_the_second_horizon_by_the_plug_in_gap(
    law: laws.RepeatedLaw, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fit without its correction misses by the law-side gap of the ``W``-only plug-in."""
    correction_dropped(monkeypatch)
    fitted = _w_only(law)
    for plan, gap in zip(SDR_PLANS, PLUG_IN_GAP[law.name], strict=True):
        name = f"ey_regimen[{plan} @ t=2]"
        assert fitted.psi(name) - law.truth[name] == pytest.approx(gap, abs=6e-7), plan
        assert abs(gap) > 1e-3, plan
        first = f"ey_regimen[{plan} @ t=1]"
        assert fitted.psi(first) == pytest.approx(law.truth[first], abs=1e-12), plan


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_n8_moves_the_estimate_by_the_declared_amount_and_every_curve_by_a_floor(
    law: laws.RepeatedLaw, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correction scaled by 0.9 moves the ``h = 2`` estimate, and each curve by 0.13 or more."""
    base = _w_only(law)
    correction_scaled(monkeypatch)
    moved = _w_only(law)
    rows = law.first_row_of()
    for plan, expected in zip(SDR_PLANS, N8_MOVE[law.name], strict=True):
        name = f"ey_regimen[{plan} @ t=2]"
        assert moved.psi(name) - base.psi(name) == pytest.approx(expected, abs=6e-7), plan
        assert abs(expected) > 1e-4, plan
        for horizon in laws.HORIZONS:
            name = f"ey_regimen[{plan} @ t={horizon}]"
            shift = np.max(
                np.abs(moved.influence_curves[name][rows] - base.influence_curves[name][rows])
            )
            assert shift >= 0.13, (plan, horizon)


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_n9_moves_the_estimate_by_the_declared_amount_and_every_curve_by_a_floor(
    law: laws.RepeatedLaw, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The censoring factor reaches the ``h = 2`` estimate and the curve of every cell by 0.33."""
    base = _w_only(law)
    saturated = _saturated(law)
    censoring_factor_replaced_by_one(monkeypatch)
    moved = _w_only(law)
    rows = law.first_row_of()
    for plan, expected in zip(SDR_PLANS, N9_MOVE[law.name], strict=True):
        name = f"ey_regimen[{plan} @ t=2]"
        assert moved.psi(name) - base.psi(name) == pytest.approx(expected, abs=6e-7), plan
        assert abs(expected) > 1e-3, plan
    # The saturated fit keeps its estimate (the correction vanishes in sample), and its curve moves.
    for plan in SDR_PLANS:
        for horizon in laws.HORIZONS:
            name = f"ey_regimen[{plan} @ t={horizon}]"
            now = _saturated(law)
            assert now.psi(name) == pytest.approx(saturated.psi(name), abs=1e-12)
            shift = np.max(
                np.abs(now.influence_curves[name][rows] - saturated.influence_curves[name][rows])
            )
            assert shift >= 0.33, (plan, horizon)
