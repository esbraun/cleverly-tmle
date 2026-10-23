"""The band-rank placebo on a dose whose declared support is a function (X82 part J).

The witnesses follow ``X82-J-plan.md`` section 6. Each one carries a nonzero term (a band whose
location and width read ``W``, a node-2 band that reads ``A_1``, a component-2 band that reads
component 1), and each has a deliberate-mutation control that fails when the component it
guards is wrong.
"""

from __future__ import annotations

import importlib
import itertools
import warnings
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import Shift, Support
from cleverly.interventions import policy as policy_module
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalData, StructuralNestedMean
from cleverly.validation import refute
from cleverly.validation._band_placebo import (
    BAND_SENTENCE,
    NEAREST_DONORS,
    PanelBands,
    PointBands,
    check_row_wise,
    mapped,
    placebo_bands,
    positions,
)
from cleverly.validation.refute import _permuted_treatment, refute_refusal

from .. import dose_declarations as law
from .test_mtp_declared_support import INSIDE, NATURAL, _fit

refute_module = importlib.import_module("cleverly.validation.refute")
band_module = importlib.import_module("cleverly.validation._band_placebo")

CONTRAST = "ate_policy[inside vs natural course]"


class _Fixed:
    """A generator stand-in whose permutation is fixed, to enumerate the point draw."""

    def __init__(self, order: tuple[int, ...]) -> None:
        self.order = np.asarray(order)

    def permutation(self, size: int) -> np.ndarray:
        assert size == self.order.size
        return self.order


def _stand_in(w: np.ndarray, a: np.ndarray, support: Support, atoms: tuple[float, ...] = ()) -> Any:
    """The fields of a point ``CausalData`` that the band-rank draw reads, at any ``n``."""
    return SimpleNamespace(
        treatment_kind="continuous",
        treatment_support=support,
        treatment_name="A",
        treatment=np.asarray(a, dtype=float),
        has_missing_treatment=False,
        covariates=np.asarray(w, dtype=float).reshape(-1, 1),
        covariate_names=("W",),
        frame_like=pd.DataFrame,
        dose_atoms=atoms,
    )


def _conditional_means(keys: list[tuple[Any, ...]], values: list[float]) -> dict[Any, float]:
    sums: dict[Any, float] = defaultdict(float)
    counts: dict[Any, int] = defaultdict(int)
    for key, value in zip(keys, values, strict=True):
        sums[key] += value
        counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}


def _null_gap(w: np.ndarray, y: np.ndarray, placebos: list[np.ndarray]) -> float:
    """The largest ``|E[Y | A', W] - E[Y | W]|`` of the pooled placebo law."""
    keys = [(float(w[i]), round(float(p[i]), 10)) for p in placebos for i in range(w.size)]
    values = [float(y[i]) for _ in placebos for i in range(w.size)]
    given_w = _conditional_means([(float(v),) for v in w], [float(v) for v in y])
    pooled = _conditional_means(keys, values)
    return max(abs(mean - given_w[(key[0],)]) for key, mean in pooled.items())


# ------------------------------------------------------------------ exact null, point

#: Two covariate values. The band ``[0.5 W, 1 + 2 W]`` reads ``W`` in its location and its
#: width, the bands overlap, and ``Y`` moves with ``A`` within each value of ``W``.
W6 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
A6 = np.array([0.2, 0.6, 0.9, 0.7, 1.5, 2.2])
Y6 = A6 + 10.0 * W6
BAND6 = Support(lambda h: 0.5 * np.asarray(h["W"]), lambda h: 1.0 + 2.0 * np.asarray(h["W"]))


def test_the_pooled_point_placebo_law_is_exactly_null() -> None:
    bands = PointBands.prepare(_stand_in(W6, A6, BAND6))
    assert bands is not None
    data = _stand_in(W6, A6, BAND6)
    placebos = [
        bands.draw(data, _Fixed(order), "permute")  # type: ignore[arg-type]
        for order in itertools.permutations(range(6))
    ]
    lower, upper = 0.5 * W6, 1.0 + 2.0 * W6
    for placebo in placebos:
        assert np.all((placebo >= lower) & (placebo <= upper))
    assert _null_gap(W6, Y6, placebos) < 1e-12


def test_a_draw_that_reads_y_is_not_null() -> None:
    # Mutation: a position chosen by the rank of Y is a function of Y.
    bands = PointBands.prepare(_stand_in(W6, A6, BAND6))
    assert bands is not None
    by_rank = tuple(int(v) for v in np.argsort(np.argsort(Y6)))
    leaked = bands.draw(_stand_in(W6, A6, BAND6), _Fixed(by_rank), "permute")  # type: ignore[arg-type]
    assert _null_gap(W6, Y6, [leaked]) > 0.1


def test_the_restricted_permutation_keeps_a_row_own_dose_more_often_than_one_in_n() -> None:
    # Option B of the plan: the permutations that keep every raw dose inside its new band. On a
    # discrete W it is a stratified permutation and exact, so its defect shows on a continuous
    # W: a narrow band can receive only doses near its own, and its own dose more often than
    # 1 / n. The band rank gives each row its own position with probability exactly 1 / n.
    w = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a = w + 0.5 * np.array([0.9, 0.1, 0.7, 0.3, 0.5, 0.2])
    lower, upper = w, w + 0.5
    feasible = [
        order
        for order in itertools.permutations(range(6))
        if np.all((a[list(order)] >= lower) & (a[list(order)] <= upper))
    ]
    own = np.mean([np.mean(np.asarray(order) == np.arange(6)) for order in feasible])
    assert own > 0.3
    support = Support(lambda h: np.asarray(h["W"]), lambda h: np.asarray(h["W"]) + 0.5)
    bands = PointBands.prepare(_stand_in(w, a, support))
    assert bands is not None
    kept = [
        np.mean(bands.draw(_stand_in(w, a, support), _Fixed(order), "permute") == a)  # type: ignore[arg-type]
        for order in itertools.permutations(range(6))
    ]
    assert abs(float(np.mean(kept)) - 1.0 / 6.0) < 1e-12


# ------------------------------------------------------------------ exact null, longitudinal


def _panel_frame() -> pd.DataFrame:
    """Twelve units, two nodes, one unit censored after node 1.

    ``A_1`` has the band ``[0.5 W, 1 + 2 W]`` and ``A_2`` the band
    ``[A_1 + 0.5 L_2, A_1 + 0.5 L_2 + 1 + W]``, which reads the earlier treatment. ``Y`` is
    ``A_1``, so a placebo dose that carries the observed ``A_1`` carries ``Y``.
    """
    w = np.tile([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
    a1 = np.r_[[0.2, 0.6, 0.9, 0.7, 1.5, 2.2], [0.3, 0.5, 0.8, 1.1, 1.9, 2.9]]
    l2 = np.tile([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 2)
    u2 = np.r_[[0.1, 0.4, 0.8, 0.3, 0.9, 0.6], [0.5, 0.2, 0.7, 0.95, 0.15, 0.45]]
    a2 = a1 + 0.5 * l2 + u2 * (1.0 + w)
    c1 = np.ones(12)
    c1[2] = 0.0
    a2[2] = np.nan
    y = a1.copy()
    y[2] = np.nan
    l2_recorded = l2.copy()
    l2_recorded[2] = np.nan
    c2 = np.where(c1 == 0, np.nan, 1.0)
    return pd.DataFrame({"W": w, "A1": a1, "C1": c1, "L2": l2_recorded, "A2": a2, "C2": c2, "Y": y})


def _low2(h: Any) -> np.ndarray:
    return np.asarray(h["A1"], dtype=float) + 0.5 * np.asarray(h["L2"], dtype=float)


def _high2(h: Any) -> np.ndarray:
    return _low2(h) + 1.0 + np.asarray(h["W"], dtype=float)


PANEL_SUPPORTS = {
    "A1": BAND6,
    "A2": Support(_low2, _high2),
}


def _panel_data(frame: pd.DataFrame | None = None, supports: Any = None) -> LongitudinalData:
    return LongitudinalData.from_frame(
        _panel_frame() if frame is None else frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
        continuous_treatment=PANEL_SUPPORTS if supports is None else supports,
    )


def _window(previous: np.ndarray, value: float, k: int) -> list[int]:
    """The ``k`` donors nearest ``value``, the lower one first on a tie."""
    order = sorted(range(previous.size), key=lambda j: (abs(previous[j] - value), previous[j]))
    return order[:k]


def _panel_pooled(
    data: LongitudinalData, bands: PanelBands, *, observed_a1: bool, stale_a1: bool = False
) -> Any:
    """The whole placebo law: every node-1 position, then every donor of its window.

    Returns the largest gap of ``Q_2``, the count of node-2 doses outside their band, and the
    largest gap of ``Q_1 = E[Q_2 | W, A'_1, C_1 = 1]`` from ``E[Q_2 | W, C_1 = 1]``. A unit
    censored after node 1 makes the ``C_1 = 1`` condition bind. ``stale_a1`` keeps the observed
    ``A_1`` on every unit, the mutation of the node-1 step.
    """
    kernel = bands.kernel
    first = kernel.trajectories[:, 0]
    both = ~np.isnan(kernel.trajectories[:, 0]) & ~np.isnan(kernel.trajectories[:, 1])
    previous, current = kernel.trajectories[both, 0], kernel.trajectories[both, 1]
    observed = np.asarray(data.treatment, dtype=float)
    outcomes = np.asarray(data.outcome, dtype=float)
    w = np.asarray(data.baseline[:, 0], dtype=float)
    l2 = np.asarray(data.time_varying[1][:, 0], dtype=float)
    keys: list[tuple[Any, ...]] = []
    values: list[float] = []
    first_keys: list[tuple[Any, ...]] = []
    first_units: list[int] = []
    outside = 0
    for u1 in first[~np.isnan(first)]:
        window = _window(previous, float(u1), min(NEAREST_DONORS, previous.size))
        for donor in window:
            drawn = np.column_stack([np.full(data.n, u1), np.full(data.n, current[donor])])
            masked = np.where(np.isnan(observed), np.nan, drawn)
            doses = bands._mapped(data, masked, drawn, strict=True)
            if stale_a1:
                doses[:, 0] = observed[:, 0]
            if observed_a1:
                # Mutation: node 2 mapped on the observed A_1 instead of the placebo A'_1.
                low = observed[:, 0] + 0.5 * l2
                doses[:, 1] = np.where(
                    np.isnan(observed[:, 1]), np.nan, low + current[donor] * (1.0 + w)
                )
                band_low = doses[:, 0] + 0.5 * l2
                band_high = band_low + 1.0 + w
                with np.errstate(invalid="ignore"):
                    outside += int(np.sum((doses[:, 1] < band_low) | (doses[:, 1] > band_high)))
            for i in range(data.n):
                if np.isnan(observed[i, 1]):
                    continue
                keys.append((w[i], l2[i], round(doses[i, 0], 10), round(doses[i, 1], 10)))
                values.append(outcomes[i])
                first_keys.append((w[i], round(doses[i, 0], 10)))
                first_units.append(i)
    pooled = _conditional_means(keys, values)
    at_risk = ~np.isnan(observed[:, 1])
    cells = _conditional_means(
        [(w[i], l2[i]) for i in np.flatnonzero(at_risk)],
        [outcomes[i] for i in np.flatnonzero(at_risk)],
    )
    gap = max(abs(mean - cells[key[:2]]) for key, mean in pooled.items())
    # Q_2 is E[Y | W, L_2] exactly, so Q_1 regresses that cell mean on (W, A'_1) among the
    # units uncensored after node 1.
    q2 = [cells[(w[i], l2[i])] for i in first_units]
    first = _conditional_means(first_keys, q2)
    given_w = _conditional_means(
        [(w[i],) for i in np.flatnonzero(at_risk)],
        [cells[(w[i], l2[i])] for i in np.flatnonzero(at_risk)],
    )
    first_gap = max(abs(mean - given_w[(key[0],)]) for key, mean in first.items())
    return gap, outside, first_gap


def test_the_pooled_longitudinal_placebo_law_is_exactly_null() -> None:
    data = _panel_data()
    bands = PanelBands.prepare(data)
    assert bands is not None
    assert bands.banded == (True, True)
    gap, _, first_gap = _panel_pooled(data, bands, observed_a1=False)
    # Q_2 = E[Y | W, A'_1, L_2, A'_2] equals E[Y | W, L_2], and Q_1 = E[Q_2 | W, A'_1, C_1 = 1]
    # equals E[Q_2 | W, C_1 = 1]: both regressions are free of the placebo treatment, so every
    # regimen has one value and every contrast is 0.
    assert gap < 1e-12
    assert first_gap < 1e-12
    assert np.isnan(np.asarray(data.treatment, dtype=float)[2, 1])


def test_keeping_the_observed_a1_breaks_the_node_1_null() -> None:
    data = _panel_data()
    bands = PanelBands.prepare(data)
    assert bands is not None
    _, _, first_gap = _panel_pooled(data, bands, observed_a1=False, stale_a1=True)
    assert first_gap > 0.05


def test_mapping_node_2_on_the_observed_a1_leaves_the_band_and_is_not_null() -> None:
    data = _panel_data()
    bands = PanelBands.prepare(data)
    assert bands is not None
    gap, outside, _ = _panel_pooled(data, bands, observed_a1=True)
    assert outside > 0
    assert gap > 0.1


def test_the_kernel_draws_a_position_from_the_nearest_donors_only() -> None:
    data = _panel_data()
    bands = PanelBands.prepare(data)
    assert bands is not None
    kernel = bands.kernel
    assert kernel.nearest == (1, NEAREST_DONORS)
    both = ~np.isnan(kernel.trajectories[:, 0]) & ~np.isnan(kernel.trajectories[:, 1])
    previous, current = kernel.trajectories[both, 0], kernel.trajectories[both, 1]
    rng = np.random.default_rng(0)
    for value in (0.0, 0.31, 1.0):
        given = np.full(4000, value)
        drawn = kernel._nearest_draw(rng, given, previous, current, current, NEAREST_DONORS)
        window = set(current[_window(previous, value, NEAREST_DONORS)].tolist())
        assert set(drawn.tolist()) == window


# ------------------------------------------------------------------ doses stay in the band


def test_every_point_placebo_dose_lies_in_its_band_and_a_plain_permutation_does_not() -> None:
    frame = law.point_frame()
    result = _fit(frame, "density", treatment_support=Support(0.0, law.upper)).single()
    bands = placebo_bands(result)
    assert isinstance(bands, PointBands)
    upper = law.upper(frame)
    rng = np.random.default_rng(4)
    for mode in ("permute", "draw"):
        for _ in range(5):
            placebo = bands.draw(result.data, rng, mode)
            assert np.all((placebo >= 0.0) & (placebo <= upper))
    # The #428 reason: the plain permutation ignores the receiving row's band.
    plain = _permuted_treatment(result.data, np.random.default_rng(4))
    assert np.mean(plain > upper) > 0.05


def _vector_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    d1 = rng.uniform(0.0, 1.0 + np.abs(w))
    d2 = 2.0 * d1 + rng.uniform(0.0, 1.0 + w**2)
    return pd.DataFrame({"W": w, "D1": d1, "D2": d2, "Y": rng.normal(d1 + d2 + w)})


def _d2_low(h: Any) -> np.ndarray:
    return 2.0 * np.asarray(h["D1"], dtype=float)


def _d2_high(h: Any) -> np.ndarray:
    return _d2_low(h) + 1.0 + np.asarray(h["W"], dtype=float) ** 2


VECTOR_SUPPORTS = {
    "D1": Support(0.0, lambda h: 1.0 + np.abs(np.asarray(h["W"], dtype=float))),
    "D2": Support(_d2_low, _d2_high),
}


def _vector_data() -> CausalData:
    return CausalData.from_frame(
        _vector_frame(),
        outcome="Y",
        treatment=["D1", "D2"],
        covariates=["W"],
        continuous_treatment=["D1", "D2"],
    ).with_dose_declarations(VECTOR_SUPPORTS)


def test_a_vector_maps_component_2_on_the_placebo_component_1() -> None:
    data = _vector_data()
    bands = PointBands.prepare(data)
    assert bands is not None and bands.vector
    w = data.covariates[:, 0]
    rng = np.random.default_rng(1)
    for mode in ("permute", "draw"):
        placebo = bands.draw(data, rng, mode)
        d1, d2 = placebo[:, 0], placebo[:, 1]
        assert np.all((d1 >= 0.0) & (d1 <= 1.0 + np.abs(w)))
        assert np.all((d2 >= 2.0 * d1) & (d2 <= 2.0 * d1 + 1.0 + w**2))
    # Mutation: component 2 mapped on the observed component 1 leaves the band the refit
    # reads at the placebo component 1.
    natural = np.asarray(data.treatment_components, dtype=float)
    placebo = bands.draw(data, np.random.default_rng(1), "permute")
    position = positions(
        natural[:, 1], 2.0 * natural[:, 0], 2.0 * natural[:, 0] + 1 + w**2, "scale"
    )
    donors = np.flatnonzero(np.ones(data.n, dtype=bool))
    order = np.random.default_rng(1).permutation(data.n)
    wrong = mapped(
        position[order], 2.0 * natural[donors, 0], 2.0 * natural[donors, 0] + 1 + w**2, "scale"
    )
    low = 2.0 * placebo[:, 0]
    assert np.mean((wrong < low) | (wrong > low + 1.0 + w**2)) > 0.05


def _cluster_frame(*, empty: bool = False, seed: int = 0) -> pd.DataFrame:
    """Forty clusters of four rows, both doses constant within a cluster.

    ``A_1`` has the band ``[0.2 W, 2 + 0.2 W]`` and ``A_2`` the band
    ``[0.5 A_1 + 0.1 W, 0.5 A_1 + 0.1 W + 2]``, so the rows of one cluster have bands with
    nonzero spread. With ``empty``, ``A_2`` has the band ``[W A_1, W A_1 + 1]`` and half the
    clusters spread ``W`` widely at a small ``A_1``.
    """
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(40), 4)
    if empty:
        spread = clusters % 2 == 0
        w = np.where(spread, np.tile([0.1, 0.4, 0.6, 0.9], 40), 0.5)
        a1_cluster = np.where(
            np.arange(40) % 2 == 0, rng.uniform(0.0, 0.5, 40), rng.uniform(3.0, 5.0, 40)
        )
        a1 = a1_cluster[clusters]
        low = np.zeros(clusters.size)
        high = np.zeros(clusters.size)
        for cluster in range(40):
            rows = clusters == cluster
            low[rows] = np.max(w[rows] * a1[rows])
            high[rows] = np.min(w[rows] * a1[rows] + 1.0)
        a2 = low + rng.uniform(0.0, 1.0, 40)[clusters] * (high - low)
    else:
        w = rng.uniform(0.0, 1.0, clusters.size)
        a1 = np.zeros(clusters.size)
        a2 = np.zeros(clusters.size)
        for cluster in range(40):
            rows = clusters == cluster
            a1[rows] = rng.uniform(0.2 * w[rows].max(), 2.0 + 0.2 * w[rows].min())
            base = 0.5 * a1[rows] + 0.1 * w[rows]
            a2[rows] = rng.uniform(base.max(), base.min() + 2.0)
    return pd.DataFrame(
        {"id": clusters, "W": w, "A1": a1, "A2": a2, "Y": rng.normal(size=clusters.size)}
    )


def _cluster_data(frame: pd.DataFrame, supports: Any) -> LongitudinalData:
    return LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], []],
        id="id",
        continuous_treatment=supports,
    )


CLUSTER_SUPPORTS = {
    "A1": Support(lambda h: 0.2 * np.asarray(h["W"]), lambda h: 2.0 + 0.2 * np.asarray(h["W"])),
    "A2": Support(
        lambda h: 0.5 * np.asarray(h["A1"]) + 0.1 * np.asarray(h["W"]),
        lambda h: 0.5 * np.asarray(h["A1"]) + 0.1 * np.asarray(h["W"]) + 2.0,
    ),
}


def test_a_cluster_dose_stays_constant_and_inside_every_row_band() -> None:
    frame = _cluster_frame()
    data = _cluster_data(frame, CLUSTER_SUPPORTS)
    bands = PanelBands.prepare(data)
    assert bands is not None and bands.kernel.cluster_codes is not None
    clusters = frame["id"].to_numpy()
    w = frame["W"].to_numpy()
    for seed in range(5):
        doses, _ = bands.placebo(data, np.random.default_rng(seed))
        a1, a2 = doses[:, 0], doses[:, 1]
        assert np.all((a1 >= 0.2 * w) & (a1 <= 2.0 + 0.2 * w))
        assert np.all((a2 >= 0.5 * a1 + 0.1 * w) & (a2 <= 0.5 * a1 + 0.1 * w + 2.0))
        for cluster in range(40):
            rows = clusters == cluster
            assert np.all(doses[rows] == doses[rows][0])
    # Mutation: a per-row map of the cluster's position breaks the constancy.
    drawn = bands.kernel.draw(np.random.default_rng(0), 40)[clusters]
    per_row = mapped(drawn[:, 0], 0.2 * w, 2.0 + 0.2 * w, "scale")
    assert any(np.ptp(per_row[clusters == c]) > 0 for c in range(40))


def test_an_empty_cluster_band_on_a_placebo_history_is_refused() -> None:
    supports = {
        "A1": Support(0.0, 6.0),
        "A2": Support(
            lambda h: np.asarray(h["W"]) * np.asarray(h["A1"]),
            lambda h: np.asarray(h["W"]) * np.asarray(h["A1"]) + 1.0,
        ),
    }
    data = _cluster_data(_cluster_frame(empty=True), supports)
    bands = PanelBands.prepare(data)
    assert bands is not None
    with pytest.raises(CapabilityError, match="share no dose") as caught:
        for seed in range(20):
            bands.placebo(data, np.random.default_rng(seed))
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert caught.value.refusal.roadmap == "X82"


# ------------------------------------------------------------------ constant bands


def test_a_constant_support_keeps_the_plain_draw_bit_for_bit() -> None:
    frame = law.point_frame()
    fixed = _fit(frame, "density", treatment_support=Support(0.0, 2.0)).single()
    assert placebo_bands(fixed) is None
    expected = np.random.default_rng(9).permutation(np.asarray(fixed.data.treatment, dtype=float))
    drawn = refute_module._placebo_data(fixed.data, np.random.default_rng(9), None).treatment
    assert np.array_equal(np.asarray(drawn), expected)
    # Mutation control on the same data: the affine round trip through a constant band moves
    # the last bit of some row, so a draw that applied it would fail the equality above.
    a = np.asarray(fixed.data.treatment, dtype=float)
    lower, upper = np.zeros(a.size), np.full(a.size, 2.0)
    round_trip = mapped(
        positions(a, lower + 0.1, upper + 0.1, "scale"), lower + 0.1, upper + 0.1, "scale"
    )
    assert not np.array_equal(round_trip, a)


# ------------------------------------------------------------------ atoms, forms, zero width


def _atom_stand_in(atom: float, *, outside: bool = False) -> Any:
    rng = np.random.default_rng(3)
    w = rng.integers(0, 2, 200).astype(float)
    a = rng.uniform(0.0, 1.0 + w)
    planted = (rng.random(200) < 0.25) & ((w == 1.0) if outside else np.ones(200, dtype=bool))
    a[planted] = atom
    return _stand_in(w, a, Support(0.0, law.upper), atoms=(atom,))


def test_atoms_keep_their_values_and_their_pooled_share() -> None:
    data = _atom_stand_in(0.5)
    bands = PointBands.prepare(data)
    assert bands is not None
    observed = int(np.sum(data.treatment == 0.5))
    rng = np.random.default_rng(0)
    for _ in range(5):
        placebo = bands.draw(data, rng, "permute")
        assert int(np.sum(placebo == 0.5)) == observed
        assert np.all((placebo >= 0.0) & (placebo <= law.upper({"W": data.covariates[:, 0]})))


def test_an_atom_outside_a_band_is_refused_naming_the_atom_and_the_row() -> None:
    with pytest.raises(CapabilityError, match=r"declared atom 1.5 .* of row \d+") as caught:
        PointBands.prepare(_atom_stand_in(1.5, outside=True))
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP


def test_a_one_sided_band_uses_the_offset_and_a_mixed_band_is_refused() -> None:
    rng = np.random.default_rng(5)
    w = rng.integers(0, 2, 200).astype(float)
    a = 3.0 * w + rng.exponential(size=200)
    lower_only = Support(lambda h: 3.0 * np.asarray(h["W"]), np.inf)
    bands = PointBands.prepare(_stand_in(w, a, lower_only))
    assert bands is not None and bands.form == "lower"
    placebo = bands.draw(_stand_in(w, a, lower_only), rng, "permute")
    assert np.all(placebo >= 3.0 * w)
    mixed = Support(0.0, lambda h: np.where(np.asarray(h["W"]) == 1.0, np.inf, 20.0))
    with pytest.raises(CapabilityError, match="finite on some rows and infinite on others"):
        PointBands.prepare(_stand_in(w, a, mixed))


def test_a_zero_width_row_keeps_its_bound_and_gives_no_position() -> None:
    rng = np.random.default_rng(6)
    w = rng.integers(0, 3, 300).astype(float)
    a = np.where(w == 2.0, 1.0, rng.uniform(0.0, 1.0 + w))

    def high(h: Any) -> np.ndarray:
        v = np.asarray(h["W"], dtype=float)
        return np.where(v == 2.0, 1.0, 1.0 + v)

    def low(h: Any) -> np.ndarray:
        return np.where(np.asarray(h["W"], dtype=float) == 2.0, 1.0, 0.0)

    data = _stand_in(w, a, Support(low, high))
    bands = PointBands.prepare(data)
    assert bands is not None
    zero = w == 2.0
    assert not np.isin(np.flatnonzero(zero), bands.pool).any()
    placebo = bands.draw(data, rng, "permute")
    assert np.all(placebo[zero] == 1.0)
    # No row of positive width receives its lower bound: the pool holds no position 0.
    assert np.sum(placebo[~zero] == low({"W": w})[~zero]) == 0
    # Mutation: a zero-width row's position, defined as 0, in the pool gives rows of positive
    # width an edge atom at L, which the undeclared-atom check refuses.
    pool = np.r_[bands.pool, np.flatnonzero(zero)]
    position = np.r_[bands.position, np.zeros(int(zero.sum()))]
    donors = rng.permutation(pool.size)
    wrong = a.copy()
    lower, upper = low({"W": w}), high({"W": w})
    wrong[pool] = mapped(position[donors], lower[pool], upper[pool], "scale")
    edge = (~zero) & (wrong == lower)
    assert edge.sum() > 5
    with pytest.raises(DataError, match="records"):
        policy_module._refuse_undeclared_atoms(
            wrong[~zero], atoms=(), moved=True, bins=None, where="A"
        )


# ------------------------------------------------------------------ the preflight up front

#: ``a + 0.5`` with no cap: inside the band on every observed dose of the law below, and
#: outside it once a ``W = 0`` row receives a position above one half.
UNCAPPED = Shift(0.5, cap=None, name="uncapped")


def _low_doses(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """``W`` binary; ``A ~ U(0, 0.5)`` at ``W = 0`` and ``U(0, 1.5)`` at ``W = 1``."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    w[0] = 1.0
    a = rng.uniform(0.0, 0.5 + w)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * a - 0.4 * w)))
    return pd.DataFrame({"W": w, "A": a, "Y": rng.binomial(1, p).astype(float)})


def _uncapped_fit() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(policies=[NATURAL, UNCAPPED], ratio="density", **law.point_learners())
            .fit(
                _low_doses(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_support=Support(0.0, law.upper),
            )
            .single()
        )


def test_a_policy_that_leaves_the_band_on_a_draw_is_refused_before_any_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _uncapped_fit()
    calls: list[int] = []
    original = type(result.estimator).refit

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(result.estimator), "refit", counted)
    estimand = "ate_policy[uncapped vs natural course]"
    for seed in range(3):
        with pytest.raises(
            DataError, match="band-rank placebo: regimen 'uncapped' is not identified"
        ):
            refute(result, estimand=estimand, tests=("placebo",), n_replicates=3, random_state=seed)
    assert calls == []
    # With the extreme-position check removed, the per-draw preflight still refuses on a draw.
    monkeypatch.setattr(refute_module, "_band_extremes_check", lambda result, bands: None)
    with pytest.raises(DataError, match="on placebo draw 1 of the band-rank placebo"):
        refute(result, estimand=estimand, tests=("placebo",), n_replicates=20, random_state=0)
    assert calls == []


def test_a_capped_policy_runs_and_reads_descriptively() -> None:
    result = _fit(law.point_frame(), "density", treatment_support=Support(0.0, law.upper)).single()
    report = refute(result, estimand=CONTRAST, tests=("placebo",), n_replicates=4, random_state=1)
    test = report["placebo"]
    assert len(test.values) == 4
    assert test.passed is None
    assert BAND_SENTENCE in test.detail
    assert refute_refusal(result, estimand=CONTRAST, tests=("placebo",)) is None


# ------------------------------------------------------------------ the row-wise guard


def test_the_row_wise_guard_admits_a_rank_and_refuses_a_closure_over_the_dose() -> None:
    rng = np.random.default_rng(7)
    w = rng.normal(size=50)
    frame = pd.DataFrame({"W": w})
    rank = Support(0.0, lambda h: 1.0 + pd.Series(np.asarray(h["W"])).rank().to_numpy() / 50)
    check_row_wise(rank, lambda: frame, 50, node="A")
    dose = rng.uniform(size=50)
    closure = Support(0.0, lambda h: dose + 1.0)
    with pytest.raises(CapabilityError, match="not a row-wise function"):
        check_row_wise(closure, lambda: frame, 50, node="A")


def test_an_inverted_band_on_a_placebo_history_is_refused_naming_the_node() -> None:
    frame = _panel_frame()

    # At W = 0 the observed A_1 stays below 0.92, where the upper bound of A_2 is valid. A
    # placebo A'_1 above 0.92 at W = 0 (the position 0.96 of A_1 = 2.9 at W = 1) inverts it.
    def high(h: Any) -> np.ndarray:
        a1, w = np.asarray(h["A1"], dtype=float), np.asarray(h["W"], dtype=float)
        return np.where((w == 0.0) & (a1 > 0.92), -5.0, _high2(h))

    data = _panel_data(frame, {"A1": BAND6, "A2": Support(_low2, high)})
    bands = PanelBands.prepare(data)
    assert bands is not None
    with pytest.raises(CapabilityError, match="node 'A2' gives row") as caught:
        for seed in range(50):
            bands.placebo(data, np.random.default_rng(seed))
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_a_held_node_with_a_callable_band_is_refused() -> None:
    frame = law.panel_frame()
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment="A1",
        baseline=["W"],
        time_varying=[[], ["L2"]],
        continuous_treatment={"A1": Support(0.0, law.upper)},
    )
    assert data.treatment_held and data.n_times == 2
    with pytest.raises(CapabilityError, match="held node 'A1'") as caught:
        PanelBands.prepare(data)
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP


# ------------------------------------------------------------------ end to end, longitudinal


def test_the_longitudinal_placebo_runs_on_a_band_that_reads_the_history() -> None:
    frame = law.panel_frame()
    regimens = {"natural": NATURAL, "up at 2": DynamicRegimen("up at 2", (NATURAL, INSIDE))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(regimens, reference="natural", **law.panel_learners()).fit(
            frame,
            continuous_treatment={"A1": Support(0.0, law.upper), "A2": Support(0.0, law.upper)},
            **law.PANEL,
        )
        report = refute(
            result,
            estimand="ate_regimen[up at 2 vs natural]",
            tests=("placebo",),
            n_replicates=3,
            random_state=2,
        )
    test = report["placebo"]
    assert len(test.values) == 3
    assert test.passed is None
    assert BAND_SENTENCE in test.detail


# ------------------------------------------------------------------ structural fits


def _snmm_fit() -> Any:
    """A structural nested mean fit whose static regimen sits at the observed top of ``A_1``."""
    rng = np.random.default_rng(0)
    n = 400
    w = rng.uniform(-0.5, 0.5, n)
    a1 = w - 1.0 + 2.0 * rng.uniform(size=n)
    l2 = 0.5 * w + 0.3 * a1 + rng.standard_normal(n)
    a2 = w - 1.0 + 2.0 * rng.uniform(size=n)
    y = w + 0.5 * a1 + 0.3 * l2 + 0.4 * a2 + rng.standard_normal(n)
    frame = pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y})
    band = Support(lambda h: np.asarray(h["W"]) - 1.0, lambda h: np.asarray(h["W"]) + 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuralNestedMean(
            {"edge": (float(a1.max()), 0.0), "zero": (0.0, 0.0)},
            reference="zero",
            outcome_learner=GLM(family="gaussian"),
            treatment_learner=LinearRegression(),
            n_folds=1,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment={"A1": band, "A2": band},
        )


def test_a_structural_fit_preflights_each_placebo_panel_before_any_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _snmm_fit()
    calls: list[int] = []
    original = type(result.estimator).refit

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(result.estimator), "refit", counted)
    estimand = "ate_regimen[edge vs zero]"
    # The band-rank placebo pools the positions, so the largest placebo A_1 falls below the
    # observed top, where the static regimen sits.
    with pytest.raises(CapabilityError, match="on placebo draw 1 of the band-rank placebo"):
        refute(result, estimand=estimand, tests=("placebo",), n_replicates=3, random_state=0)
    assert calls == []
    # Mutation: without the structural preflight the same refusal comes from inside a refit.
    monkeypatch.setattr(type(result.estimator), "_preflight_policies", lambda self, prepared: None)
    with pytest.raises(CapabilityError, match="outside the observed range") as caught:
        refute(result, estimand=estimand, tests=("placebo",), n_replicates=3, random_state=0)
    assert "band-rank placebo" not in str(caught.value)
    assert calls != []


# ------------------------------------------------------------------ the extreme corners


def _corner_fit() -> Any:
    """Node 2's band ``[0, 1 + 2 A_1]`` reads ``A_1``, and its shift is held at 1.8.

    A placebo unit with ``A'_1`` near 0 and ``A'_2`` at its top position takes ``A'_2`` near 1
    and the assigned dose 1.5, above the band. Neither uniform corner does: all-low assigns 0.5,
    and all-high holds the dose. The observed node-2 positions stay below one half where
    ``A_1 < 0.5``, so the fit's own check passes.
    """
    rng = np.random.default_rng(5)
    n = 400
    w = rng.integers(0, 2, n).astype(float)
    a1 = rng.uniform(size=n)
    l2 = rng.binomial(1, 0.5, n).astype(float)
    u2 = np.where(a1 < 0.5, 0.5 * rng.uniform(size=n), rng.uniform(size=n))
    a2 = u2 * (1.0 + 2.0 * a1)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 0.4 * a1 + 0.3 * a2 - 0.2 * w)))
    frame = pd.DataFrame(
        {"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": rng.binomial(1, p).astype(float)}
    )
    regimens = {
        "natural": NATURAL,
        "up at 2": DynamicRegimen("up at 2", (NATURAL, Shift(0.5, cap=1.8, name="up"))),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(regimens, reference="natural", **law.panel_learners()).fit(
            frame,
            continuous_treatment={
                # A callable band on A_1 too, so both nodes carry positions.
                "A1": Support(0.0, lambda h: 1.0 + 0.0 * np.asarray(h["W"], dtype=float)),
                "A2": Support(0.0, lambda h: 1.0 + 2.0 * np.asarray(h["A1"], dtype=float)),
            },
            **law.PANEL,
        )


def test_a_mixed_corner_of_the_extremes_is_checked_before_any_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _corner_fit()
    bands = placebo_bands(result)
    assert isinstance(bands, PanelBands)
    assert len(bands.extremes(result.data)) == 4
    with pytest.raises(DataError, match="'up at 2'"):
        refute_module._band_extremes_check(result, bands)
    # Mutation: the two uniform corners alone miss the low-high corner.
    monkeypatch.setattr(band_module, "CORNER_NODES", 0)
    assert len(bands.extremes(result.data)) == 2
    refute_module._band_extremes_check(result, bands)
