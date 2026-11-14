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
from cleverly.inference.multiplier import simultaneous_bands
from cleverly.inference.node_variance import hazard_bound
from cleverly.longitudinal import LTMLE, SDR, LongitudinalData
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import sequential as sequential_module
from cleverly.longitudinal.estimator import (
    SequentialGComputation,
    SequentialIPW,
    boundary_node_parameters,
    constant_node_parameters,
)
from tests.unit._source_mutation import mutate

ALPHA = 0.05


def _frame(
    n: int = 600,
    seed: int = 1,
    *,
    quiet: dict[int, tuple[int, ...]] | None = None,
    full: dict[int, tuple[int, ...]] | None = None,
    cluster_size: int | None = None,
) -> LongitudinalData:
    """A four-node panel; ``quiet`` removes every event of an arm at the named nodes.

    ``full`` turns every unit of an arm at risk at the named nodes into an event there.
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


@pytest.mark.parametrize("hazard", [False, True], ids=["iterated_mean", "hazard"])
@pytest.mark.parametrize("n_folds", [1, 2])
def test_the_supplement_is_the_binomial_bound_over_the_fitted_rows(
    hazard: bool, n_folds: int
) -> None:
    """Unclustered: the variance is the curve's plus n^-2 s^2 sum C^2 over the node's rows."""
    data = _frame(quiet={1: (2,)})
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


@pytest.mark.parametrize("hazard", [False, True], ids=["iterated_mean", "hazard"])
def test_a_clustered_node_counts_clusters_and_sums_within_them(hazard: bool) -> None:
    """m_t is the clusters with a fitted row, and a unit's loading is its summed |C|."""
    data = _frame(n=120, quiet={1: (2,)}, cluster_size=10)
    result = _fit(data, hazard=hazard)
    estimate = result["risk_regimen[treated @ t=2]"]
    loading = _loading(result, "treated @ t=2", 2)
    assert loading.size == 12
    assert estimate.node is not None
    assert estimate.node.loadings.nodes[0].units == 12
    assert estimate.node.node_variance == pytest.approx(
        float(np.sum(loading**2)) / data.n**2, rel=1e-12
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

    The level is on its bound and takes ``out_of_range_plugin``. The contrast against the
    interior arm is inside its bounds and its node term, read with b = 1, is nonzero.
    """
    data = _frame(full={1: (2,)})
    # No treated unit is at risk after node 2, so the curve stops there.
    result = _fit(data, horizons=[1, 2])
    assert result["risk_regimen[treated @ t=2]"].inference == OUT_OF_RANGE_STATUS
    contrast = result["ate_regimen[control vs treated @ t=2]"]
    assert -1.0 < contrast.psi < 1.0
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


def test_the_frame_reports_both_variance_parts() -> None:
    data = _frame(quiet={1: (2,)})
    estimate = _fit(data)["risk_regimen[treated @ t=2]"]
    row = estimate.to_dict()
    assert row["variance_source"] == "boundary_node"
    assert row["boundary_node_std_err"] == pytest.approx(np.sqrt(estimate.node.node_variance))
    assert row["influence_curve_std_err"] == pytest.approx(
        np.sqrt(estimate.node.influence_variance)
    )
