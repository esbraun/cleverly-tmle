"""The boundary-node supplement of a longitudinal estimate (X54 part (i)).

An ``LTMLE`` node regression whose target holds one value on its fitted rows drops that node's
term from the influence curve. Inside the supplement's reach the estimate keeps the fit status
and adds the node's variance at the exact one-sided binomial bound of its hazard over the node's
independent units (:mod:`cleverly.inference.node_variance`). Every check below recomputes the
supplement independently from the fit's own steps, and each guard has a deliberate mutation
that fails it.

The fixtures are time-to-event panels: an end-of-study fit whose final node holds one value has a
regimen mean exactly on its bound (``out_of_range_plugin``), so a supplemented estimate inside
its bounds needs an earlier node with events, as on a survival curve.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._inference_status import BOUNDARY_NODE_OUT_OF_REACH, OUT_OF_RANGE_STATUS
from cleverly.exceptions import CapabilityError, DataWarning
from cleverly.inference import influence_variance
from cleverly.inference import results as results_module
from cleverly.inference.influence import bootstrap_line
from cleverly.inference.multiplier import simultaneous_bands
from cleverly.inference.node_variance import hazard_bound
from cleverly.interventions import Shift
from cleverly.longitudinal import LTMLE, SDR, LongitudinalData
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import sequential as sequential_module
from cleverly.longitudinal.estimator import (
    SequentialGComputation,
    SequentialIPW,
    boundary_node_parameters,
    constant_node_parameters,
)
from tests.discrete_law_point_survival import CellMeans
from tests.unit._source_mutation import mutate

ALPHA = 0.05


def _frame(
    n: int = 600,
    seed: int = 1,
    *,
    quiet: dict[int, tuple[int, ...]] | None = None,
    full: dict[int, tuple[int, ...]] | None = None,
    cluster_size: int | None = None,
    weighted: bool = False,
) -> LongitudinalData:
    """A four-node panel; ``quiet`` removes every event of an arm at the named nodes.

    ``full`` turns every unit of an arm at risk at the named nodes into an event there.
    ``weighted`` gives the rows sampling weights drawn from 0.5 to 2.
    """
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n)
    # Clustered, both arms sit inside every cluster, so a contrast's node terms share units.
    a = rng.binomial(1, 0.5, n) if cluster_size is None else np.tile([0, 1], n // 2 + 1)[:n]
    time = rng.integers(1, 5, n).astype(float)
    event = rng.binomial(1, 0.6, n)
    for arm, nodes in (quiet or {}).items():
        for node in nodes:
            event[(a == arm) & (time == node)] = 0
    for arm, nodes in (full or {}).items():
        for node in nodes:
            at_risk = (a == arm) & (time >= node)
            time[at_risk] = float(node)
            event[at_risk] = 1
    frame = pd.DataFrame({"time": time, "event": event, "A": a, "W": w})
    settings: dict[str, Any] = {}
    if cluster_size is not None:
        frame["id"] = np.arange(n) // cluster_size
        settings["id"] = "id"
    if weighted:
        frame["wt"] = np.random.default_rng(seed + 1).uniform(0.5, 2.0, n)
        settings["weights"] = "wt"
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W"],
        grid=[1, 2, 3, 4],
        **settings,
    )


def _learners(hazard: bool) -> dict[str, Any]:
    learners: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
    }
    if not hazard:
        learners["pseudo_learner"] = LinearRegression()
    return learners


def _fit(data: LongitudinalData, *, hazard: bool = False, **settings: Any) -> Any:
    if hazard:
        settings["survival_model"] = "hazard"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"treated": 1, "control": 0},
            **_learners(hazard),
            n_folds=settings.pop("n_folds", 1),
            random_state=0,
            **settings,
        ).fit(data)


def _units(data: LongitudinalData) -> np.ndarray:
    if data.cluster is None:
        return np.arange(data.n)
    _, codes = np.unique(np.asarray(data.cluster), return_inverse=True)
    return codes


def _loading(result: Any, key: str, time: int) -> np.ndarray:
    """Each unit's loading of one flagged node, rebuilt from the fit's own arrays."""
    fit = result.fits[key]
    data = result.data
    weights = np.asarray(data.weights, dtype=float)
    if fit.hazard_clever is not None:
        coefficient = weights * fit.hazard_clever[:, time - 1]
        rows = np.zeros(data.n, dtype=bool)
        for cell in fit.hazard_boundary:
            if cell.time == time:
                rows |= cell.rows
    else:
        step = next(step for step in fit.steps if step.time == time)
        coefficient = weights * step.clever
        rows = np.asarray(step.trained_on, dtype=bool)
    rows &= weights > 0.0
    units = _units(data)
    m = np.unique(units[rows]).size
    bound = 1.0 - (ALPHA / 2.0) ** (1.0 / m)
    spread = np.sqrt(bound * (1.0 - bound))
    return spread * np.bincount(
        units, weights=np.abs(coefficient) * rows, minlength=units.max() + 1
    )


def _curve_variance(estimate: Any, data: LongitudinalData) -> float:
    return influence_variance(estimate.influence_curve, data.cluster)


# ------------------------------------------------------------------------------- the bound


@pytest.mark.parametrize("units", [1, 5, 200])
def test_the_hazard_bound_solves_the_binomial_tail(units: int) -> None:
    bound = hazard_bound(units, ALPHA)
    assert (1.0 - bound) ** units == pytest.approx(ALPHA / 2.0, rel=1e-12)


def test_the_hazard_bound_needs_a_unit() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        hazard_bound(0, ALPHA)


# --------------------------------------------------------------------- the nonzero witnesses


@pytest.mark.parametrize("weighted", [False, True], ids=["unit_weights", "weighted"])
@pytest.mark.parametrize("hazard", [False, True], ids=["iterated_mean", "hazard"])
@pytest.mark.parametrize("n_folds", [1, 2])
def test_the_supplement_is_the_binomial_bound_over_the_fitted_rows(
    hazard: bool, n_folds: int, weighted: bool
) -> None:
    """Unclustered: the variance is the curve's plus n^-2 s^2 sum C^2 over the node's rows."""
    data = _frame(quiet={1: (2,)}, weighted=weighted)
    result = _fit(data, hazard=hazard, n_folds=n_folds)
    name = "risk_regimen[treated @ t=2]"
    estimate = result[name]
    assert estimate.inference == "influence_curve"
    assert estimate.node is not None
    assert estimate.variance_source == "boundary_node"
    supplement = float(np.sum(_loading(result, "treated @ t=2", 2) ** 2)) / data.n**2
    assert supplement > 0.0
    assert estimate.node.node_variance == pytest.approx(supplement, rel=1e-12)
    assert estimate.variance == pytest.approx(
        _curve_variance(estimate, data) + supplement, rel=1e-12
    )
    assert boundary_node_parameters(result) == tuple(
        key for key, value in result.estimates.items() if value.node is not None
    )
    assert name in boundary_node_parameters(result)
    assert constant_node_parameters(result) == ()


def test_removing_the_supplement_returns_the_plugin_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation (1): without the loadings the estimate reads its curve variance alone."""
    data = _frame(quiet={1: (2,)})
    reported = _fit(data)["risk_regimen[treated @ t=2]"]
    monkeypatch.setattr(estimator_module, "boundary_loadings", lambda *args, **kwargs: None)
    mutated = _fit(data)["risk_regimen[treated @ t=2]"]
    assert mutated.node is None
    assert mutated.variance == pytest.approx(_curve_variance(mutated, data), rel=1e-12)
    assert reported.variance > mutated.variance


def test_counting_every_row_moves_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation (2): m read as every row of the panel, not the node's fitted rows."""
    data = _frame(quiet={1: (2,)})
    reported = _fit(data)["risk_regimen[treated @ t=2]"]
    mutate(
        monkeypatch,
        sequential_module,
        estimator_module,
        "boundary_loadings",
        [("m = int(np.unique(units[rows]).size)", "m = int(n_units)")],
    )
    mutated = _fit(data)["risk_regimen[treated @ t=2]"]
    assert mutated.node is not None
    assert mutated.node.node_variance != pytest.approx(reported.node.node_variance, rel=1e-6)


@pytest.mark.parametrize("weighted", [False, True], ids=["unit_weights", "weighted"])
@pytest.mark.parametrize("hazard", [False, True], ids=["iterated_mean", "hazard"])
def test_a_clustered_node_counts_clusters_and_sums_within_them(
    hazard: bool, weighted: bool
) -> None:
    """m_t is the clusters with a fitted row, and a unit's loading is its summed |C|."""
    data = _frame(n=120, quiet={1: (2,)}, cluster_size=10, weighted=weighted)
    result = _fit(data, hazard=hazard)
    estimate = result["risk_regimen[treated @ t=2]"]
    loading = _loading(result, "treated @ t=2", 2)
    assert loading.size == 12
    assert estimate.node is not None
    assert estimate.node.loadings.nodes[0].units == 12
    assert estimate.node.node_variance == pytest.approx(
        float(np.sum(loading**2)) / data.n**2, rel=1e-12
    )


@pytest.mark.parametrize("power", ["** 2", "* 0.0 + 1.0"], ids=["squared", "omitted"])
@pytest.mark.parametrize("cluster_size", [None, 10], ids=["rows", "clusters"])
def test_a_wrong_weight_in_the_coefficient_is_caught(
    monkeypatch: pytest.MonkeyPatch, power: str, cluster_size: int | None
) -> None:
    """m2 mutation control: a coefficient with w squared or without w misses the witness."""
    data = _frame(n=120, quiet={1: (2,)}, cluster_size=cluster_size, weighted=True)
    mutate(
        monkeypatch,
        sequential_module,
        sequential_module,
        "boundary_coefficient",
        [
            (
                "weights = np.asarray(fit.obs_weights, dtype=float)",
                f"weights = np.asarray(fit.obs_weights, dtype=float) {power}",
            )
        ],
    )
    result = _fit(data)
    estimate = result["risk_regimen[treated @ t=2]"]
    loading = _loading(result, "treated @ t=2", 2)
    assert estimate.node.node_variance != pytest.approx(
        float(np.sum(loading**2)) / data.n**2, rel=1e-6
    )


def test_counting_rows_in_a_clustered_fit_fails_the_clustered_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: m read as rows, and a per-row sum of squares, on a clustered fit."""
    data = _frame(n=120, quiet={1: (2,)}, cluster_size=10)
    expected = _fit(data)["risk_regimen[treated @ t=2]"].node.node_variance
    mutate(
        monkeypatch,
        sequential_module,
        estimator_module,
        "boundary_loadings",
        [("m = int(np.unique(units[rows]).size)", "m = int(rows.sum())")],
    )
    mutated = _fit(data)["risk_regimen[treated @ t=2]"].node.node_variance
    assert mutated != pytest.approx(expected, rel=1e-6)


def test_the_all_ones_mirror_carries_its_term_through_a_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2: every follower at risk at node 2 has the event, so the risk is exactly 1.

    The level is on its bound and takes ``out_of_range_plugin``, and the contrast against the
    interior arm inherits that status by precedence, so neither shows an interval. The
    contrast's node record, read with b = 1, is still nonzero; a shown interval that reads a
    node held at 1 is the competing-risks witness below.
    """
    data = _frame(full={1: (2,)})
    # No treated unit is at risk after node 2, so the curve stops there.
    result = _fit(data, horizons=[1, 2])
    assert result["risk_regimen[treated @ t=2]"].inference == OUT_OF_RANGE_STATUS
    contrast = result["ate_regimen[control vs treated @ t=2]"]
    assert -1.0 < contrast.psi < 1.0
    assert contrast.inference == OUT_OF_RANGE_STATUS
    assert contrast.node is not None
    nodes = [node for node in contrast.node.loadings.nodes if node.level.endswith("t=2]")]
    assert {(node.node, node.value) for node in nodes} == {(1, 1.0), (2, 1.0)}
    # Node 2 holds every event, and node 1's pseudo-outcome is then 1 too: both are flagged.
    supplement = (
        sum(float(np.sum(_loading(result, "treated @ t=2", time) ** 2)) for time in (1, 2))
        / data.n**2
    )
    assert supplement > 0.0
    assert contrast.node.node_variance == pytest.approx(supplement, rel=1e-12)
    # Mutation: a version that flags only all-zero nodes drops the term.
    real = sequential_module.boundary_cells
    monkeypatch.setattr(
        sequential_module,
        "boundary_cells",
        lambda fit: tuple(cell for cell in real(fit) if cell.value == 0.0),
    )
    assert _fit(data, horizons=[1, 2])["ate_regimen[control vs treated @ t=2]"].node is None


def _competing_full(split: bool) -> tuple[LongitudinalData, int]:
    """Two causes; every treated unit at risk at node 2 has an event there.

    The all-cause hazard of the treated cell is held at 1 at node 2. With ``split`` the events
    there take both causes, so the cell drops one residual (the all-cause one). Without it they
    are all cause 2, so cause 1 is held at 0 as well and the cell drops two. The second value
    is that count, read from the frame: causes with no event in the cell, plus one when every
    unit of the cell has an event.
    """
    rng = np.random.default_rng(3)
    n = 600
    w = rng.integers(0, 2, n)
    a = rng.binomial(1, 0.5, n)
    time = rng.integers(1, 4, n).astype(float)
    event = rng.choice([0, 1, 2], n, p=[0.3, 0.35, 0.35])
    at_risk = (a == 1) & (time >= 2)
    time[at_risk] = 2.0
    event[at_risk] = rng.choice([1, 2], int(at_risk.sum())) if split else 2
    frame = pd.DataFrame({"time": time, "event": event, "A": a, "W": w})
    cell = event[(a == 1) & (time == 2.0)]
    dropped = sum(int(not np.any(cell == cause)) for cause in (1, 2)) + int(np.all(cell > 0))
    data = LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W"],
        grid=[1, 2],
        causes={1: "c1", 2: "c2"},
    )
    return data, dropped


@pytest.mark.parametrize("split", [True, False], ids=["causes_split", "one_cause"])
def test_a_shown_incidence_reads_a_node_held_at_one(split: bool) -> None:
    """M2: the all-cause hazard is held at 1 and the cause-specific incidence is interior.

    The shown interval adds the node's loading times the residuals the cell drops, counted
    here from the panel: one when the causes split, two when cause 1 is also held at 0.
    """
    data, terms = _competing_full(split)
    result = _fit(data, hazard=True)
    name = "cif_regimen[treated, c1 @ t=2]"
    estimate = result[name]
    assert estimate.supplies_inference
    assert 0.0 < estimate.psi < 1.0
    assert estimate.node is not None
    assert terms == (1 if split else 2)
    (cell,) = result.fits["treated, c1 @ t=2"].hazard_boundary
    assert (cell.time, cell.terms) == (2, terms)
    supplement = terms**2 * float(np.sum(_loading(result, "treated, c1 @ t=2", 2) ** 2))
    assert supplement > 0.0
    assert estimate.node.node_variance == pytest.approx(supplement / data.n**2, rel=1e-12)


@pytest.mark.parametrize(
    "edits",
    [
        [("if not (empty or full):", "if not empty:"), ("empty + full)", "empty)")],
        [("empty + full)", "1)")],
    ],
    ids=["full_dropped", "one_term"],
)
def test_the_full_node_count_is_caught(
    monkeypatch: pytest.MonkeyPatch, edits: list[tuple[str, str]]
) -> None:
    """Mutations: a cell that drops the full node, or counts one term, moves the variance."""
    from cleverly.longitudinal import hazard as hazard_module

    data, _ = _competing_full(split=False)
    expected = _fit(data, hazard=True)["cif_regimen[treated, c1 @ t=2]"].node.node_variance
    mutate(monkeypatch, hazard_module, hazard_module.HazardInputs, "boundary_cells", edits)
    mutated = _fit(data, hazard=True)["cif_regimen[treated, c1 @ t=2]"].node
    assert mutated is not None
    assert mutated.node_variance != pytest.approx(expected, rel=1e-6)


def test_a_clustered_contrast_sums_both_regimens_inside_each_unit() -> None:
    """m3: 12 clusters with followers of both arms, both flagged at node 2."""
    data = _frame(n=120, quiet={1: (2,), 0: (2,)}, cluster_size=10)
    result = _fit(data)
    contrast = result["ate_regimen[control vs treated @ t=2]"]
    treated = _loading(result, "treated @ t=2", 2)
    control = _loading(result, "control @ t=2", 2)
    formula = float(np.sum((treated + control) ** 2)) / data.n**2
    phantom = float(np.sum(treated**2 + control**2)) / data.n**2
    assert contrast.node is not None
    assert contrast.node.node_variance == pytest.approx(formula, rel=1e-12)
    # The rev 1 independent-phantom version is strictly smaller on this fixture.
    assert phantom < formula * (1.0 - 1e-6)


def test_a_ratio_weights_each_regimen_by_its_absolute_gradient() -> None:
    data = _frame(n=120, quiet={1: (2,), 0: (2,)}, cluster_size=10)
    result = _fit(data)
    top, bottom = "risk_regimen[treated @ t=2]", "risk_regimen[control @ t=2]"
    ratio = result.ratio(top, bottom)
    loadings = ratio.node.loadings
    psi_a, psi_b = result.psi(top), result.psi(bottom)
    # The log-ratio gradient over (top, bottom), checked against a finite difference.
    step = 1e-6
    numeric = np.array(
        [
            (np.log((psi_a + step) / psi_b) - np.log((psi_a - step) / psi_b)) / (2 * step),
            (np.log(psi_a / (psi_b + step)) - np.log(psi_a / (psi_b - step))) / (2 * step),
        ]
    )
    order = [loadings.levels.index(top), loadings.levels.index(bottom)]
    np.testing.assert_allclose(ratio.node.gradient[order], numeric, rtol=1e-7)
    treated = _loading(result, "treated @ t=2", 2)
    control = _loading(result, "control @ t=2", 2)
    formula = float(np.sum((abs(numeric[0]) * treated + abs(numeric[1]) * control) ** 2))
    assert ratio.node.node_variance == pytest.approx(formula / data.n**2, rel=1e-6)


def test_an_unclustered_static_contrast_adds_the_two_supplements() -> None:
    """2.5(3): the followers of the two arms are disjoint, so the bound is exact there."""
    data = _frame(quiet={1: (2,), 0: (2,)})
    result = _fit(data)
    contrast = result["ate_regimen[control vs treated @ t=2]"]
    treated = result["risk_regimen[treated @ t=2]"].node.node_variance
    control = result["risk_regimen[control @ t=2]"].node.node_variance
    assert treated > 0.0 and control > 0.0
    assert contrast.node.node_variance == pytest.approx(treated + control, rel=1e-12)


@pytest.mark.parametrize("clustered", [True, False])
def test_two_flagged_nodes_add_inside_a_cluster_and_apart_on_rows(clustered: bool) -> None:
    """m6: two flagged nodes of one parameter, summed per cluster; per node on rows."""
    data = _frame(n=120, quiet={1: (2, 3)}, cluster_size=10 if clustered else None)
    result = _fit(data)
    estimate = result["risk_regimen[treated @ t=3]"]
    first = _loading(result, "treated @ t=3", 2)
    second = _loading(result, "treated @ t=3", 3)
    assert {node.node for node in estimate.node.loadings.nodes if "t=3" in node.level} == {2, 3}
    together = float(np.sum((first + second) ** 2)) / data.n**2
    apart = float(np.sum(first**2 + second**2)) / data.n**2
    assert apart < together * (1.0 - 1e-6)
    expected = together if clustered else apart
    assert estimate.node.node_variance == pytest.approx(expected, rel=1e-12)


def test_a_continuous_dose_policy_reads_the_shared_cell_of_its_slots() -> None:
    """A policy on a continuous dose: every unit at risk shares the node's one held cell.

    The hazard walk keys a continuous node's cells by slot, not by observed level, so the
    cell's rows are every positive-weight row at risk there, rebuilt here from the panel.
    The coefficient is the walk's loss weight ``w r / G``, the clever weight at the node.
    """
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame({"W": rng.integers(0, 2, n).astype(float), "D": rng.uniform(0, 2, n)})
    frame["C1"] = frame["C2"] = 1.0
    frame["Y1"] = (rng.uniform(size=n) < 0.2).astype(float)
    frame["Y2"] = 0.0
    frame.loc[frame["Y1"] == 1.0, ["C2", "Y2"]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            {"up": Shift(0.5, cap=None)},
            n_folds=1,
            survival_model="hazard",
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            censoring_learner=CellMeans(),
        ).fit(
            frame,
            treatment="D",
            continuous_treatment=["D"],
            outcome=["Y1", "Y2"],
            baseline=["W"],
            censoring=["C1", "C2"],
        )
    estimate = result["risk_regimen[up @ t=2]"]
    assert estimate.node is not None
    rows = (frame["Y1"] == 0.0).to_numpy()
    fit = result.fits["up @ t=2"]
    (cell,) = fit.hazard_boundary
    assert cell.time == 2
    assert np.array_equal(cell.rows, rows)
    coefficient = np.asarray(result.data.weights, dtype=float) * fit.hazard_clever[:, 1]
    bound = hazard_bound(int(rows.sum()), ALPHA)
    supplement = bound * (1.0 - bound) * float(np.sum((coefficient * rows) ** 2)) / n**2
    assert supplement > 0.0
    assert estimate.node.node_variance == pytest.approx(supplement, rel=1e-12)


def test_the_range_logit_interval_carries_the_supplemented_variance() -> None:
    data = _frame(quiet={1: (2,)})
    estimate = _fit(data)["risk_regimen[treated @ t=2]"]
    assert estimate.interval_scale == "range_logit"
    slope = 1.0 / (estimate.psi * (1.0 - estimate.psi))
    center = np.log(estimate.psi / (1.0 - estimate.psi))
    half = stats.norm.ppf(1.0 - ALPHA / 2.0) * slope * np.sqrt(estimate.variance)
    low, high = estimate.ci
    assert np.log(low / (1.0 - low)) == pytest.approx(center - half, rel=1e-10)
    assert np.log(high / (1.0 - high)) == pytest.approx(center + half, rel=1e-10)
    assert estimate.variance > estimate.node.influence_variance


# --------------------------------------------------------------------- reach and refusals


def _out_of_reach(kind: str, data: LongitudinalData) -> Any:
    regimens = {"treated": 1, "control": 0}
    learners = _learners(hazard=False)
    if kind == "sdr":
        estimator: Any = SDR(regimens, **learners, n_folds=1, random_state=0)
    elif kind == "gcomp":
        estimator = SequentialGComputation(regimens, random_state=0)
    elif kind == "ipw":
        estimator = SequentialIPW(regimens, random_state=0)
    elif kind == "repeats":
        estimator = LTMLE(regimens, **learners, n_folds=2, repeats=3, random_state=0)
    elif kind == "isotonic":
        estimator = LTMLE(regimens, **learners, n_folds=1, isotonic=True, random_state=0)
    else:  # pragma: no cover - the parametrization lists every kind
        raise AssertionError(kind)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = estimator.fit(data)
    messages = [str(item.message) for item in caught if issubclass(item.category, DataWarning)]
    return result, messages


@pytest.mark.parametrize("kind", ["sdr", "gcomp", "ipw", "repeats", "isotonic"])
def test_a_fit_outside_the_reach_keeps_the_constant_node_status(kind: str) -> None:
    """R2: the parameter withholds its interval, and the warning names the kind of fit."""
    result, messages = _out_of_reach(kind, _frame(quiet={1: (2,)}))
    estimate = result["risk_regimen[treated @ t=2]"]
    assert estimate.inference == "constant_node_plugin"
    assert estimate.node is None
    assert boundary_node_parameters(result) == ()
    what, why = BOUNDARY_NODE_OUT_OF_REACH[kind]
    warned = [message for message in messages if "'constant_node_plugin' status" in message]
    assert warned, messages
    assert f"This fit is {what}: {why}." in warned[0]
    assert "exact one-sided binomial bound" in warned[0]


def test_the_reach_reads_the_estimator_class_and_the_settings() -> None:
    """Every key the fit can return is a row of the reach table, and LTMLE returns none."""
    data = _frame(quiet={1: (2,)})
    ltmle = LTMLE({"treated": 1, "control": 0}, **_learners(False))
    plans: tuple[Any, ...] = ()
    reach = ltmle._boundary_reach(data, plans, repeated=False, learned=False, separable=False)
    assert reach is None
    for flag, key in (("repeated", "repeats"), ("learned", "learned_rule")):
        settings = {"repeated": False, "learned": False, "separable": False, flag: True}
        assert ltmle._boundary_reach(data, plans, **settings) == key
        assert key in BOUNDARY_NODE_OUT_OF_REACH
    from cleverly.longitudinal import SurvivalCTMLE, SurvivalDRTMLE

    assert SurvivalCTMLE._boundary_kind in BOUNDARY_NODE_OUT_OF_REACH
    assert SurvivalDRTMLE._boundary_kind in BOUNDARY_NODE_OUT_OF_REACH


def test_a_fit_without_a_flagged_node_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off-event identity: no node record, and the fit equals the one without the supplement."""
    data = _frame()
    reported = _fit(data)
    assert boundary_node_parameters(reported) == ()
    monkeypatch.setattr(estimator_module, "boundary_loadings", lambda *args, **kwargs: None)
    plain = _fit(data)
    for name, estimate in reported.estimates.items():
        assert estimate.node is None
        assert estimate.psi == plain[name].psi
        assert estimate.std_error == plain[name].std_error
        assert estimate.ci == plain[name].ci


def test_the_band_leaves_a_supplemented_estimate_out_and_refuses_it_by_name() -> None:
    data = _frame(quiet={1: (2,)})
    result = _fit(data, simultaneous=True)
    supplemented = set(boundary_node_parameters(result))
    assert supplemented
    assert result.simultaneous is not None
    assert supplemented.isdisjoint(result.simultaneous.bands)
    assert "boundary-node variance" in str(result.summary())
    chosen = {name: result[name] for name in sorted(supplemented)}
    with pytest.raises(CapabilityError, match="boundary-node supplement"):
        simultaneous_bands(chosen, random_state=0)
    with pytest.raises(CapabilityError, match="boundary-node variance"):
        result.covariance(sorted(supplemented))


def _bootstrapped(data: LongitudinalData) -> Any:
    return _fit(data, simultaneous=True, n_bootstrap=19, bootstrap_interval="percentile")


def test_the_bootstrap_of_a_supplemented_estimate_is_a_diagnostic_and_leaves_the_band() -> None:
    """A row resample keeps the flagged node's one value, so its spread misses the node term.

    The supplemented estimates publish their bootstrap as a range with that reason and stay
    out of the bootstrap sup-t band; their unflagged neighbours keep an interval and the band.
    """
    result = _bootstrapped(_frame(quiet={1: (2,)}))
    supplemented = set(boundary_node_parameters(result))
    assert supplemented
    for name in supplemented:
        summary = result[name].bootstrap
        assert summary.diagnostic and summary.boundary_node
        assert "bootstrap_range_lower" in result[name].to_dict()
        assert "keeps the flagged node's one value" in bootstrap_line(result[name])
    quiet = "risk_regimen[treated @ t=1]"
    assert not result[quiet].bootstrap.diagnostic
    assert result.bootstrap_band is not None
    assert quiet in result.bootstrap_band.bands
    assert supplemented.isdisjoint(result.bootstrap_band.bands)
    assert "the boundary-node supplement adds" in str(result.summary())


def test_a_bootstrap_that_ignores_the_node_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation controls: an interval or a band that reads the node's resample passes neither."""
    data = _frame(quiet={1: (2,)})
    mutate(
        monkeypatch,
        results_module,
        estimator_module,
        "attach_bootstrap",
        [("diagnostic=diagnostic or estimate.node is not None,", "diagnostic=diagnostic,")],
    )
    mutate(
        monkeypatch,
        estimator_module,
        LTMLE,
        "_attach_bootstrap",
        [("and estimate.node is None\n", "\n")],
    )
    result = _bootstrapped(data)
    name = "risk_regimen[treated @ t=2]"
    assert result[name].node is not None
    assert not result[name].bootstrap.diagnostic
    assert name in result.bootstrap_band.bands


def test_the_frame_reports_both_variance_parts() -> None:
    data = _frame(quiet={1: (2,)})
    estimate = _fit(data)["risk_regimen[treated @ t=2]"]
    row = estimate.to_dict()
    assert row["variance_source"] == "boundary_node"
    assert row["boundary_node_std_err"] == pytest.approx(np.sqrt(estimate.node.node_variance))
    assert row["influence_curve_std_err"] == pytest.approx(
        np.sqrt(estimate.node.influence_variance)
    )


# ------------------------------------------------------- what reads a supplemented estimate


def _supplemented_kind() -> Any:
    """The capability sweep's boundary-node kind: horizons 2 to 4 carry the supplement."""
    from tests.unit._capability_sweep_support import KINDS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KINDS["ltmle_boundary_node"].build()


def test_refute_scales_its_shift_by_the_supplemented_standard_error() -> None:
    """The refutation reads the reported standard error, the curve's plus the node term.

    One scale serves every test of a report. A placebo refit permutes treatment, so its arm can
    hold events at the node and the supplement is part of the spread it compares; for this
    added common cause, which keeps the node's one value, the larger scale is conservative.
    """
    result = _supplemented_kind()
    name = "ate_regimen[a1 vs a0 @ t=2]"
    estimate = result[name]
    assert estimate.node is not None
    curve_only = float(np.sqrt(estimate.node.influence_variance))
    assert estimate.plugin_std_error > curve_only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = result.diagnostics.refute(
            estimand=name, tests=("random_common_cause",), n_replicates=1
        )
    (test,) = report.tests
    shift = test.values[0] - test.original
    assert test.statistic == pytest.approx(shift / estimate.plugin_std_error, rel=1e-9)
    assert test.statistic != pytest.approx(shift / curve_only, rel=1e-6)


def test_the_survival_functionals_carry_the_supplement() -> None:
    """RMST and the E-value's risk ratio read the node term of the risks they combine."""
    from tests.unit.test_plotting_result_matrix import _boundary_node

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _boundary_node()
        rmst = result.rmst("arm1", 3)
        ratio = result.ratio("risk_regimen[arm1 @ t=2]", "risk_regimen[arm0 @ t=2]")
        evalue = result.sensitivity.evalue(("risk_regimen[arm1 @ t=2]", "risk_regimen[arm0 @ t=2]"))
    # RMST(3) = 3 - F(1) - F(2), so only F(2) loads on the flagged node.
    assert rmst.node is not None
    risk = result.estimates["risk_regimen[arm1 @ t=2]"]
    assert rmst.node.node_variance == pytest.approx(risk.node.node_variance, rel=1e-12)
    assert rmst.variance == pytest.approx(
        influence_variance(rmst.influence_curve, None) + rmst.node.node_variance, rel=1e-12
    )
    assert ratio.node is not None
    assert evalue.estimand == ratio.name
    # The E-value's limit is that of the supplemented interval's limit nearer the null.
    assert ratio.psi < 1.0 < 1.0 / ratio.ci[1]
    rr = 1.0 / ratio.ci[1]
    assert evalue.limit == pytest.approx(rr + np.sqrt(rr * (rr - 1.0)), rel=1e-9)


def _without_supplement(result: Any, scale: float = 0.0) -> Any:
    """``result`` with each supplemented variance replaced by curve + ``scale`` x node term."""
    import dataclasses

    estimates = {
        key: value
        if value.node is None
        else dataclasses.replace(
            value,
            node=None,
            variance=value.node.influence_variance + scale * value.node.node_variance,
        )
        for key, value in result.estimates.items()
    }
    return dataclasses.replace(result, estimates=estimates)


def _score_table(result: Any) -> pd.DataFrame:
    report = result.diagnostics.score_equations()
    frame = report.to_frame() if hasattr(report, "to_frame") else pd.DataFrame(report.rows)
    return frame.reset_index(drop=True)


def test_the_score_audit_keeps_the_curve_scale() -> None:
    """A supplemented fit's score audit equals the audit of the same fit without it.

    The longitudinal audit bounds each node's relative score, which the supplement does not
    enter. Mutation control: a copy whose variance carries 100 times the node term reads the
    same table, so no standard error, supplemented or not, scales the audit.
    """
    data = _frame(quiet={1: (2,)})
    result = _fit(data)
    assert result["risk_regimen[treated @ t=2]"].node is not None
    reported = _score_table(result)
    assert len(reported) > 0
    pd.testing.assert_frame_equal(reported, _score_table(_without_supplement(result)))
    pd.testing.assert_frame_equal(reported, _score_table(_without_supplement(result, 100.0)))


def _hazard_witness(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, np.ndarray, np.ndarray]:
    """A hazard fit with node 2 of ``treated`` flagged, its loading, and the exact coefficient.

    The exact coefficient of a residual of the cell is the clever weight times the derivative
    of the risk from node 2 to horizon 3, given at risk at node 2, in the node's hazard. The
    derivative is a central finite difference at the reported hazards, which a wrapper of
    :func:`cleverly.longitudinal.hazard.influence` captures from the fit.
    """
    from cleverly.longitudinal import hazard as hazard_module

    captured: list[Any] = []
    shipped = hazard_module.influence

    def capture(state: Any, inputs: Any, horizons: Any) -> Any:
        levels, curves = shipped(state, inputs, horizons)
        captured.append((state, inputs, levels))
        return levels, curves

    monkeypatch.setattr(hazard_module, "influence", capture)
    data = _frame(quiet={1: (2,)})
    result = _fit(data, hazard=True)
    fit = result.fits["treated @ t=3"]
    (cell,) = [cell for cell in fit.hazard_boundary if cell.time == 2]
    rows = cell.rows
    loading = np.abs(sequential_module.boundary_coefficient(fit, 2)) * rows
    # The state of the report that horizon 3 reads: the one whose level is this fit's.
    (state, inputs) = next(
        (state, inputs)
        for state, inputs, levels in captured
        if np.any(np.isclose(levels, fit.psi_scaled, rtol=0.0, atol=1e-12))
    )
    level = int(np.unique(inputs.observed[rows])[0])
    hazards = state.hazards[:, :, level, 0]  # (n, T): one cause

    def risk(h2: np.ndarray) -> np.ndarray:
        return 1.0 - (1.0 - h2) * (1.0 - hazards[:, 2])

    step = 1e-6
    derivative = (risk(hazards[:, 1] + step) - risk(hazards[:, 1] - step)) / (2.0 * step)
    exact = np.abs(np.asarray(fit.obs_weights) * fit.hazard_clever[:, 1] * derivative) * rows
    return result, loading, exact


def _saturated_hazard_panel() -> pd.DataFrame:
    """A censored three-node panel on binary ``W`` and ``A``; time 4 is event-free follow-up."""
    rng = np.random.default_rng(5)
    n = 160
    time = rng.integers(1, 5, n).astype(float)
    event = rng.binomial(1, 0.6, n)
    event[time == 4.0] = 0
    return pd.DataFrame(
        {"time": time, "event": event, "A": rng.binomial(1, 0.5, n), "W": rng.integers(0, 2, n)}
    )


def _saturated_hazard_fit(frame: pd.DataFrame, weights: np.ndarray) -> Any:
    """The hazard fit with cell-mean learners, so the estimate is the empirical plug-in."""
    data = LongitudinalData.from_time_to_event(
        frame.assign(wt=weights),
        time="time",
        event="event",
        treatment="A",
        baseline=["W"],
        grid=[1, 2, 3],
        weights="wt",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"treated": 1, "control": 0},
            outcome_learner=CellMeans(),
            treatment_learner=CellMeans(),
            censoring_learner=CellMeans(),
            n_folds=1,
            random_state=0,
            survival_model="hazard",
            g_bounds=(1e-8, 1.0),
        ).fit(data)


def _empirical_influence(frame: pd.DataFrame, unit: int, name: str) -> float:
    """``n`` times the central difference of the estimate in one unit's weight."""
    n, step = len(frame), 1e-5
    up, down = np.ones(n), np.ones(n)
    up[unit] += step
    down[unit] -= step
    psi_up = _saturated_hazard_fit(frame, up)[name].psi
    psi_down = _saturated_hazard_fit(frame, down)[name].psi
    return float(n * (psi_up - psi_down) / (2.0 * step))


def _clever_witness() -> list[tuple[float, float, float]]:
    """At the last node of horizons 2 and 3: the coefficient by differences, and the fit's.

    With cell-mean learners the estimate is the plug-in of the empirical law, so its
    derivative in a unit's weight is that unit's influence. Two treated units of one ``W`` cell,
    one with the event at node ``h`` and one event-free past it, share every term of the curve
    of the risk at ``h`` except the node-``h`` residual, which is 1 - h for one and -h for the
    other. Their difference is the node's coefficient, whose survival factor is 1 at the last
    node, with no use of ``hazard_clever``.
    """
    frame = _saturated_hazard_panel()
    result = _saturated_hazard_fit(frame, np.ones(len(frame)))
    treated = (frame["A"] == 1) & (frame["W"] == 0)
    out = []
    for h in (2, 3):
        name = f"risk_regimen[treated @ t={h}]"
        event = int(np.flatnonzero(treated & (frame["time"] == h) & (frame["event"] == 1))[0])
        survivor = int(np.flatnonzero(treated & (frame["time"] > h))[0])
        differenced = _empirical_influence(frame, event, name) - _empirical_influence(
            frame, survivor, name
        )
        curve = result[name].influence_curve
        fit = result.fits[f"treated @ t={h}"]
        clever = float(np.asarray(fit.obs_weights)[event] * fit.hazard_clever[event, h - 1])
        assert fit.hazard_clever[survivor, h - 1] == pytest.approx(clever, rel=1e-12)
        out.append((differenced, clever, float(curve[event] - curve[survivor])))
    return out


def test_the_clever_weight_is_the_curve_coefficient_by_finite_differences() -> None:
    """M3: ``hazard_clever`` is the coefficient read off the estimator itself.

    The panel's censoring puts an inverse weight in the coefficient, so it exceeds 1.
    """
    for differenced, clever, curve in _clever_witness():
        assert clever > 1.0
        assert differenced == pytest.approx(clever, rel=1e-6)
        assert curve == pytest.approx(clever, rel=1e-10)


@pytest.mark.parametrize(
    "new",
    ["q[r][:, None] * inverse * 0.5, 0.0)", "q[r][:, None] * np.ones_like(inverse), 0.0)"],
    ids=["half", "no_censoring_inverse"],
)
def test_a_wrong_clever_weight_scale_is_caught(monkeypatch: pytest.MonkeyPatch, new: str) -> None:
    """Mutation control: a stored clever weight off by a factor misses the differenced one."""
    from cleverly.longitudinal import hazard as hazard_module

    mutate(
        monkeypatch,
        hazard_module,
        hazard_module,
        "_cell_fits",
        [("q[r][:, None] * inverse, 0.0)", new)],
    )
    for differenced, clever, _ in _clever_witness():
        assert differenced != pytest.approx(clever, rel=1e-3)


def test_the_hazard_loading_bounds_each_exact_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """|C_exact| <= the clever-weight loading on every row of the cell, strictly on some."""
    _, loading, exact = _hazard_witness(monkeypatch)
    assert loading.max() > 0.0
    assert np.all(exact <= loading + 1e-9)
    assert np.any(exact < loading - 1e-6)


def test_half_the_clever_weight_violates_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: a loading of half the clever weight falls below an exact coefficient."""
    _, loading, exact = _hazard_witness(monkeypatch)
    assert np.any(exact > 0.5 * loading + 1e-9)
