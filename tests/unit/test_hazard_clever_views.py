"""The hazard fits' clever weight: one read-only base per regimen, one view per record.

The hazard, backward-hazard and drift producers build a clever weight that depends on the
regimen alone, so the records of every cause and horizon hold views ``base[:, :horizon]`` of
one read-only array; a separable fit shares one per level pair, across labels.  A
collaborative fit's weight reads each horizon's own state, so each record holds its own
read-only ``(n, horizon)`` array.  These tests pin the sharing, the width, the read-only flag
and each regimen's values, with witnesses that fail if one regimen's base is given to another.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import make_point_survival
from cleverly.estimators.serialize import dumps, loads
from cleverly.longitudinal import (
    LTMLE,
    DynamicRegimen,
    LongitudinalData,
    Separable,
    SurvivalCTMLE,
    SurvivalDRTMLE,
)
from cleverly.longitudinal import separable as separable_module
from cleverly.longitudinal.sequential import RegimenFit
from tests.discrete_law_hazard_time_varying import CellMeans as HeldCellMeans
from tests.discrete_law_hazard_time_varying import CellProbabilities as HeldCellProbabilities
from tests.discrete_law_hazard_time_varying import HeldLaw
from tests.discrete_law_point_survival import CellMeans, CellProbabilities, survival_law
from tests.unit.test_hazard_multinode import _coin, _random_panel

SEED = 0


def _quiet(call: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _owner(array: np.ndarray | None) -> np.ndarray:
    """The array that owns ``array``'s buffer."""
    assert array is not None
    return array if array.base is None else array.base


def _by_regimen(result: Any) -> dict[str, list[RegimenFit]]:
    groups: dict[str, list[RegimenFit]] = defaultdict(list)
    for fit in result.fits.values():
        assert fit.hazard_clever is not None
        groups[fit.regimen.label].append(fit)
    return groups


def _check_views(result: Any, n: int, n_times: int, *, causes: int) -> None:
    """Each regimen's records view one owned, read-only ``(n, T)`` base; regimens differ."""
    groups = _by_regimen(result)
    owners = []
    for fits in groups.values():
        assert len({fit.cause for fit in fits}) == causes
        assert len({fit.horizon for fit in fits}) > 1
        base = _owner(fits[0].hazard_clever)
        assert base.shape == (n, n_times)
        assert not base.flags.writeable
        for fit in fits:
            clever = fit.hazard_clever
            assert clever.shape == (n, fit.horizon)
            assert _owner(clever) is base
            assert np.shares_memory(clever, base)
            assert not clever.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                clever[0, 0] = 1.0
        owners.append(base)
    assert len({id(base) for base in owners}) == len(groups)
    for i, first in enumerate(owners):
        for second in owners[i + 1 :]:
            assert not np.shares_memory(first, second)
    assert sum(base.nbytes for base in owners) == len(groups) * n * n_times * 8


# ----------------------------------------------------------------- hazard, single node


def _point_ltmle(**kwargs: Any) -> LTMLE:
    settings: dict[str, Any] = {
        "survival_model": "hazard",
        "targeting": "one_step",
        "n_folds": 1,
        "g_bounds": (1e-9, 1.0),
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
        **kwargs,
    }
    return LTMLE(settings.pop("regimens", {"a0": 0, "a1": 1}), **settings)


@pytest.mark.parametrize("causes", [1, 2])
def test_a_single_node_hazard_fit_shares_one_base_per_regimen(causes: int) -> None:
    law = survival_law(causes=causes)
    data = LongitudinalData.from_frame(law.frame(), **law.fit_columns())
    result = _quiet(lambda: _point_ltmle().fit(data))
    _check_views(result, data.n, data.n_times, causes=causes)
    # Witness that each static arm keeps its own weight: it weights only its own arm's rows.
    arm = np.nan_to_num(np.asarray(data.treatment, dtype=float)[:, 0], nan=-1.0)
    for label, level in (("a0", 0), ("a1", 1)):
        base = _owner(_by_regimen(result)[label][0].hazard_clever)
        rows = np.any(base != 0.0, axis=1)
        assert rows.any()
        assert np.all(arm[rows] == level)


def test_a_single_node_hazard_fit_per_horizon_shares_one_base() -> None:
    law = survival_law()
    data = LongitudinalData.from_frame(law.frame(), **law.fit_columns())
    result = _quiet(lambda: _point_ltmle(targeting="iterative").fit(data))
    _check_views(result, data.n, data.n_times, causes=1)


# ----------------------------------------------------------------- hazard, multi-node


def test_a_multi_node_hazard_fit_shares_one_base_per_regimen() -> None:
    n, n_times = 500, 3
    frame, columns = _random_panel(n, n_times)
    coin = DynamicRegimen("coin", tuple(_coin(k) for k in range(1, n_times + 1)))
    estimator = LTMLE(
        {"always": (1,) * n_times, "coin": coin},
        survival_model="hazard",
        n_folds=1,
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
    )
    result = _quiet(lambda: estimator.fit(frame, **columns))
    _check_views(result, n, n_times, causes=1)
    # Witness that each regimen keeps its own weight: the static arm weights only the rows
    # that followed it through the node, and the coin also weights rows that did not.
    arms = np.nan_to_num(frame[columns["treatment"]].to_numpy(), nan=-1.0)
    followed = np.cumprod(arms == 1.0, axis=1).astype(bool)
    groups = _by_regimen(result)
    always = _owner(groups["always"][0].hazard_clever)
    coin_weight = _owner(groups["coin"][0].hazard_clever)
    assert np.all(followed[always != 0.0])
    assert np.any((coin_weight != 0.0) & ~followed)


# ----------------------------------------------------------------- backward hazard


@pytest.mark.parametrize("causes", [1, 2])
def test_a_backward_hazard_fit_shares_one_base_per_regimen(causes: int) -> None:
    law = HeldLaw(causes=causes)
    frame, columns = law.frame(), law.columns()
    estimator = LTMLE(
        {"never": 0, "always": 1},
        survival_model="hazard",
        targeting="one_step",
        n_folds=1,
        g_bounds=(1e-8, 1.0),
        outcome_learner=HeldCellMeans(),
        treatment_learner=HeldCellProbabilities(),
        censoring_learner=HeldCellMeans(),
        pseudo_learner=HeldCellMeans(),
    )
    result = _quiet(lambda: estimator.fit(frame, **columns))
    _check_views(result, len(frame), law.n_times, causes=causes)


# ----------------------------------------------------------------- drift and collaborative


def _point_data(n: int = 600) -> LongitudinalData:
    frame, _ = make_point_survival(n=n, seed=SEED, n_times=4)
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=["W1", "W2"], grid=[1, 2, 3, 4]
    )


def test_a_drift_fit_shares_one_base_per_arm_and_weights_only_its_arm() -> None:
    data = _point_data()
    glm = LogisticRegression(max_iter=1000)
    estimator = SurvivalDRTMLE(
        {"a1": 1, "a0": 0},
        reference="a0",
        outcome_learner=glm,
        treatment_learner=glm,
        censoring_learner=glm,
        n_folds=2,
        random_state=SEED,
        horizons=[1, 2, 3, 4],
    )
    result = _quiet(lambda: estimator.fit(data))
    _check_views(result, data.n, data.n_times, causes=1)
    arm = np.asarray(data.treatment[:, 0])
    for label, level in (("a1", 1), ("a0", 0)):
        fits = _by_regimen(result)[label]
        clever = max(fits, key=lambda fit: fit.horizon).hazard_clever
        rows = np.any(clever != 0.0, axis=1)
        assert rows.any()
        assert np.all(arm[rows] == level)
        assert np.all(clever[clever != 0.0] >= 1.0)


def test_a_collaborative_fit_holds_a_read_only_field_of_its_horizon() -> None:
    data = _point_data()
    estimator = SurvivalCTMLE(
        {"a1": 1, "a0": 0},
        reference="a0",
        outcome_learner=LogisticRegression(max_iter=1000),
        selection_folds=3,
        random_state=SEED,
        horizons=[2, 4],
    )
    result = _quiet(lambda: estimator.fit(data))
    for fit in result.fits.values():
        assert fit.hazard_clever.shape == (data.n, fit.horizon)
        assert not fit.hazard_clever.flags.writeable


# ----------------------------------------------------------------- separable


SEP = Separable(arms={"death": 0, "relapse": 1}, first="death")
PES = Separable(arms={"death": 1, "relapse": 0}, first="death")


def test_a_separable_fit_shares_a_base_per_level_pair_with_the_values_of_its_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Witness: each record equals the clever weight recomputed from the final state at its
    own level pair, on a fit with both pairs ``(0, 1)`` and ``(1, 0)``, which read the same
    rows with different values.  Two labels with equal levels share one base."""
    captured: dict[str, Any] = {}
    walk = separable_module.walk_separable

    def spy(start: Any, inputs: Any, targets: Any, **kwargs: Any) -> Any:
        state, record = walk(start, inputs, targets, **kwargs)
        captured.update(state=state, inputs=inputs, targets=list(targets))
        return state, record

    monkeypatch.setattr(separable_module, "walk_separable", spy)
    law = survival_law(causes=2)
    data = LongitudinalData.from_frame(law.frame(), **law.fit_columns())
    regimens = {"a0": 0, "zero": 0, "sep": SEP, "pes": PES}
    result = _quiet(lambda: _point_ltmle(regimens=regimens).fit(data))
    state, inputs = captured["state"], captured["inputs"]
    levels = {target.label: target.levels for target in captured["targets"]}
    assert levels["sep"] != levels["pes"]
    assert set(levels["sep"]) == set(levels["pes"])

    factual = np.concatenate([np.ones_like(state.survival[:, :1]), state.survival[:, :-1]], axis=1)
    observed = inputs.observed
    groups = _by_regimen(result)
    for label, fits in groups.items():
        a_d, a_y = levels[label]
        _, before, _ = state.composed(levels[label])
        reads = (observed == a_d) | (observed == a_y)
        weight = separable_module._per(
            before * inputs.inverse[inputs.rows, :, observed],
            factual[inputs.rows, :, observed],
        )
        expected = np.where(inputs.at_risk & reads[:, None] & (before > 0.0), weight, 0.0)
        base = _owner(fits[0].hazard_clever)
        assert not base.flags.writeable
        for fit in fits:
            assert _owner(fit.hazard_clever) is base
            np.testing.assert_array_equal(fit.hazard_clever, expected[:, : fit.horizon])
    assert _owner(groups["a0"][0].hazard_clever) is _owner(groups["zero"][0].hazard_clever)
    assert _owner(groups["sep"][0].hazard_clever) is not _owner(groups["pes"][0].hazard_clever)
    assert not np.array_equal(
        _owner(groups["sep"][0].hazard_clever), _owner(groups["pes"][0].hazard_clever)
    )


# ----------------------------------------------------------------- saving


def test_a_loaded_hazard_fit_keeps_the_values_and_widths() -> None:
    """A saved result writes each view as its own array: the values and widths survive, and a
    loaded record no longer shares its regimen's buffer."""
    law = survival_law()
    data = LongitudinalData.from_frame(law.frame(), **law.fit_columns())
    result = _quiet(lambda: _point_ltmle().fit(data))
    loaded = loads(dumps(result))
    for key, fit in result.fits.items():
        again = loaded.fits[key].hazard_clever
        assert again.shape == (data.n, fit.horizon)
        np.testing.assert_array_equal(again, fit.hazard_clever)
    for fits in _by_regimen(loaded).values():
        for i, first in enumerate(fits):
            for second in fits[i + 1 :]:
                assert not np.shares_memory(first.hazard_clever, second.hazard_clever)
