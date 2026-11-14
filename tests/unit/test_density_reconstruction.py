"""The binned density's grid and its corrected log-polygon reading (``learners.density``).

Each witness pins one part of the representation against a closed form or an oracle, and
the in-suite controls apply a deliberate mutation by monkeypatch and show the witness
fails under it.  The laws are ``tests/mtp_density_tails_law.py``; an oracle hazard learner
there returns each row's exact bin probabilities, so only the grid and the reading are left
between the fit and the truth.

- W1, W9d: the influence-curve gap ``SD(IC - D0) / SD(D0)`` with the oracle hazard and the
  correct outcome regression falls with ``n``.
- W2: the ratio at the largest doses keeps its tail weight.
- W3a to W3g: segment masses, exponential exactness, no new extremum, segment breaks, one
  representation across readers, the mixed law's partial-bin mass, and the pass cap.
- W4a to W4e: the grid's core, floor, robust width cap, gap bins, explicit edges and the
  preflight.
- W5: the collapse of a vector's last dose onto an atom integrates a multi-bin segment.
- W6: the remainder with a wrong outcome regression.
- W7, W10: the longitudinal fits read the point fit's density; occupancy travels with edges.
- W8: the flat-bin share.
- W9a to W9c, W9e: a covariate-dependent declared support.
- W12: the knots are computed once per density instance and are not pickled.
"""

from __future__ import annotations

import pickle
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.learners.density as density_module
from cleverly.dose_response import DoseResponse
from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_mixed_density
from cleverly.interventions import ModifiedPolicy, Piece, Shift, Support
from cleverly.learners import Folds
from cleverly.learners.density import (
    TAIL_ROWS,
    ConditionalDensity,
    MixedDensity,
    _equal_mass_edges,
    _grid,
    _knots,
    _lookup,
    _lookup_mean,
    _table_size,
    bin_edges,
    bin_occupancy,
    default_density_bins,
    fit_conditional_density,
    preflight_conditional_densities,
)
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, LongitudinalData
from cleverly.longitudinal.regimen import resolve_plans, resolve_regimens
from cleverly.longitudinal.sequential import fit_mechanism

from .. import longitudinal_mtp as mtp
from .. import mtp_density_tails_law as law
from ..studies.oracle_density_bins import oracle_bins

INF = np.inf


# ------------------------------------------------------------------ helpers


def _policy(name: str) -> Any:
    """The law's policy: one more, held at the cap on T and on ``[u - 1, u]`` on B and U."""
    if name == "T":
        return Shift(law.DELTA, cap=law.HIGH)
    if name in law.SUPPORTED:
        return ModifiedPolicy(
            "p2",
            pieces=(
                Piece(-INF, _knee(name), _plus_one, _minus_one, _unit),
                Piece(_knee(name), INF),
            ),
            policy_kind="known",
        )
    return Shift(law.DELTA, cap=None)


def _plus_one(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + law.DELTA


def _minus_one(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - law.DELTA


def _unit(b: Any, h: Any) -> np.ndarray:
    return np.ones_like(np.asarray(b, dtype=float))


class _Upper:
    """``u(h)`` of a supported law, as a picklable callable."""

    def __init__(self, name: str, offset: float = 0.0) -> None:
        self.name, self.offset = name, offset

    def __call__(self, h: Any) -> np.ndarray:
        return law.upper(self.name, h["W1"], h["W2"]) - self.offset


def _knee(name: str) -> _Upper:
    return _Upper(name, law.DELTA)


def _support(name: str) -> Support:
    return Support(0.0, _Upper(name))


def _fit(name: str, frame: pd.DataFrame, *, support: bool = True) -> Any:
    """The point TMLE with the oracle hazard and the correct outcome regression."""
    n = len(frame)
    bins = default_density_bins(n)
    edges = tuple(density_module.bin_edges(frame["A"].to_numpy(), bins))
    declared = {"treatment_support": _support(name)} if name in law.SUPPORTED and support else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                policies=(_policy(name),),
                outcome_learner=law.CorrectOutcome(),
                treatment_learner=law.OracleHazard(edges, name),
                ratio="density",
                density_bins=bins,
                cross_fit=False,
                simultaneous=False,
                max_iter=100,
                tol=1e-10,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], **declared)
            .single()
        )


def _ic_gap(name: str, n: int, seed: int, *, support: bool = True) -> float:
    frame = law.draw(name, n, seed)
    result = _fit(name, frame, support=support)
    (key,) = result.estimates
    curve = np.asarray(result.estimates[key].influence_curve, dtype=float)
    return float(np.std(curve - law.eif(name, frame)) / law.EFFICIENCY_SD[name])


def _exact(
    name: str,
    frame: pd.DataFrame,
    edges: Any = None,
    *,
    support: bool = False,
    clip: bool = False,
) -> ConditionalDensity:
    """Each row's exact bin masses on the shipped grid, read by the shipped reading."""
    w1, w2, a, _ = law.columns(frame)
    e = bin_edges(a, default_density_bins(a.size)) if edges is None else np.asarray(edges)
    p = law.bin_probabilities(name, e, w1, w2)
    if clip:
        p = law.through_clip(p)
    bounds = (np.zeros(a.size), law.upper(name, w1, w2)) if support else None
    return ConditionalDensity(p, e, bin_occupancy(a, e), bounds)


def _ratio(name: str, density: Any, frame: pd.DataFrame) -> np.ndarray:
    w1, w2, a, _ = law.columns(frame)
    return law.ratio_from(name, density, a, w1, w2)


def _histogram_read(knots: Any, values: Any) -> np.ndarray:
    """The mutation: each bin read as its probability over its width."""
    e = knots.edges
    v = np.asarray(values, dtype=float).reshape(knots.value.shape[0], -1)
    index = np.clip(np.searchsorted(e, v, side="right") - 1, 0, e.size - 2)
    rows = np.arange(v.shape[0])[:, None]
    out = knots.probabilities[rows, index] / np.diff(e)[index]
    return np.asarray(np.where((v >= e[0]) & (v <= e[-1]), out, 0.0), dtype=float)


def _equal_mass(values: Any, n_bins: int, *, name: str | None = None) -> np.ndarray:
    return _equal_mass_edges(np.asarray(values, dtype=float), n_bins, name)


def _segments(link: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` bins of each multi-bin segment of one row."""
    out, start = [], None
    for b in range(link.size + 1):
        linked_left = b > 0 and link[b - 1]
        linked_right = b < link.size and link[b]
        if linked_right and not linked_left:
            start = b
        if linked_left and not linked_right:
            assert start is not None
            out.append((start, b))
    return out


def _gauss_legendre(density: ConditionalDensity, nodes: int = 16) -> np.ndarray:
    """``(n, B)`` bin masses of the reading by a rule broken at every edge and centre."""
    e = density.edges
    x, w = np.polynomial.legendre.leggauss(nodes)
    out = np.zeros((density.n, density.n_bins))
    for b in range(density.n_bins):
        centre = 0.5 * (e[b] + e[b + 1])
        for left, right in ((e[b], centre), (centre, e[b + 1])):
            points = 0.5 * (right - left) * x + 0.5 * (right + left)
            for point, weight in zip(points, w, strict=True):
                out[:, b] += (
                    0.5 * (right - left) * weight * density.density_at(np.full(density.n, point))
                )
    return out


def _walk_levels(rng: np.random.Generator, n: int, widths: np.ndarray) -> np.ndarray:
    """Bin probabilities from smooth random log levels: steps of at most one per bin."""
    steps = rng.uniform(-1.0, 1.0, (n, widths.size))
    levels = np.cumsum(steps, axis=1)
    p = np.exp(levels) * widths
    return np.asarray(p / p.sum(axis=1, keepdims=True), dtype=float)


#: Unequal widths, an empty bin (4), and bins 11 and 13 empty so bin 12 is a one-bin segment.
WALK_WIDTHS = np.array([0.6, 1.3, 0.8, 1.9, 0.7, 1.1, 0.5, 1.6, 0.9, 1.2, 0.75, 1.4, 0.65, 1.0])
WALK_EDGES = np.concatenate([[0.0], np.cumsum(WALK_WIDTHS)])
WALK_OCCUPIED = np.ones(WALK_WIDTHS.size, dtype=bool)
WALK_OCCUPIED[[4, 11, 13]] = False
#: Bin 9 holds 1e-12 on the first rows: it breaks only those rows' segment.
WALK_FLOORED = 9


def _walk(n: int = 40, seed: int = 3) -> ConditionalDensity:
    p = _walk_levels(np.random.default_rng(seed), n, WALK_WIDTHS)
    p[: n // 4, WALK_FLOORED] = 1e-12
    p /= p.sum(axis=1, keepdims=True)
    return ConditionalDensity(p, WALK_EDGES, WALK_OCCUPIED)


# ------------------------------------------------------------------ W1


#: The seed offsets of each law in the influence-curve runs.
_LAW_SEED = {"G": 0, "T": 1, "C": 2, "S": 3}


def _ic_seed(name: str, n: int, draw: int) -> int:
    return 3300 * n + draw + _LAW_SEED[name] * 100000


def _gaps(name: str, n: int, draws: range = range(5)) -> list[float]:
    return [_ic_gap(name, n, _ic_seed(name, n, d)) for d in draws]


@pytest.mark.parametrize("name", ["G", "C", "T"])
def test_w1_the_influence_curve_gap_falls_with_n(name: str) -> None:
    """With the oracle hazard and the correct outcome regression, only the grid and reading
    stand between the influence curve and ``D0``.  G's 2,000 median and every 20,000 gap
    are bounded and the median falls to 0.6 of its 2,000 value; C (whose fall at five seeds
    is too close to 0.6 for a fixed-seed test) and T are held to the two bounds.
    """
    small, large = _gaps(name, 2000), _gaps(name, 20000)
    bound_small, bound_large = {"G": (0.10, 0.05), "C": (0.10, 0.05), "T": (0.05, 0.02)}[name]
    assert np.median(small) <= bound_small, small
    assert max(large) <= bound_large, large
    if name == "G":
        assert np.median(large) <= 0.6 * np.median(small), (small, large)


def test_w1_the_histogram_reading_fails_the_small_sample_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the same fits at 2,000 with each bin read flat break G's first clause."""
    monkeypatch.setattr(density_module, "_read", _histogram_read)
    assert np.median(_gaps("G", 2000)) > 0.10


def test_w1_the_equal_mass_grid_fails_a_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: the equal-mass grid (no refinement) breaks the 20,000 bound on G."""
    monkeypatch.setattr(density_module, "bin_edges", _equal_mass)
    assert max(_gaps("G", 20000, range(2))) > 0.05


# ------------------------------------------------------------------ W2


def _top_ratio(name: str, n: int, draw: int, *, read: Any = None) -> tuple[float, float, float]:
    """``r_n / r0`` and ``r_n`` at the largest dose, and the median ``r_n / r0`` over the top 1%."""
    frame = law.draw(name, n, 1000 * n + draw)
    density = _exact(name, frame)
    reader = density.density_at if read is None else read(density)
    w1, w2, a, _ = law.columns(frame)
    ratio = law.ratio_from(name, reader, a, w1, w2)
    truth = law.r0(name, a, w1, w2)
    main = a < 12.0 if name == "C" else np.ones(a.size, dtype=bool)
    order = np.flatnonzero(main)[np.argsort(a[main])]
    top = order[-max(1, order.size // 100) :]
    largest = order[-1]
    return (
        float(ratio[largest] / truth[largest]),
        float(ratio[largest]),
        float(np.median(ratio[top] / truth[top])),
    )


def test_w2_the_ratio_keeps_its_tail_weight_at_the_largest_dose() -> None:
    at_max = [_top_ratio("G", 20000, d) for d in range(5)]
    assert np.median([share for share, _, _ in at_max]) >= 0.4, at_max
    assert all(ratio != 1.0 for _, ratio, _ in at_max)
    for name in ("G", "C", "S"):
        for d in range(5):
            top = _top_ratio(name, 2000, d)[2]
            assert 0.8 <= top <= 1.25, (name, d, top)


def test_w2_the_histogram_reading_reads_one_at_the_largest_dose() -> None:
    """The control: read flat on the same grid, the largest dose's ratio is exactly one."""

    def flat(density: ConditionalDensity) -> Any:
        def read(values: Any) -> np.ndarray:
            return _histogram_read(density._knots(), np.asarray(values)).reshape(-1)

        return read

    at_max = [_top_ratio("G", 20000, d, read=flat) for d in range(5)]
    assert np.median([share for share, _, _ in at_max]) < 0.4 or any(
        ratio == 1.0 for _, ratio, _ in at_max
    )


# ------------------------------------------------------------------ W3a


def test_w3a_segments_keep_their_mass_and_end_bins_their_multiple() -> None:
    density = _walk()
    knots = density._knots()
    p = density.bin_probabilities
    e = density.edges
    seen_floor_break = False
    for row in range(density.n):
        segments = _segments(knots.link[row])
        assert segments
        for first, last in segments:
            mass = density.mass_between(
                np.full(density.n, e[first]), np.full(density.n, e[last + 1])
            )
            np.testing.assert_allclose(
                mass[row], p[row, first : last + 1].sum(), rtol=0, atol=1e-12
            )
            lower, upper = np.full(density.n, e[first]), np.full(density.n, e[first + 1])
            head = density.mass_between(lower, upper)[row] / p[row, first]
            lower, upper = np.full(density.n, e[last]), np.full(density.n, e[last + 1])
            tail = density.mass_between(lower, upper)[row] / p[row, last]
            assert head == pytest.approx(tail, rel=1e-9)
            assert not (first <= WALK_FLOORED <= last and row < density.n // 4)
        seen_floor_break |= row < density.n // 4
    assert seen_floor_break
    # A 16-node rule on every half bin integrates the reading to the closed form.
    by_rule = _gauss_legendre(density)
    closed = np.column_stack(
        [
            density.mass_between(np.full(density.n, e[b]), np.full(density.n, e[b + 1]))
            for b in range(density.n_bins)
        ]
    )
    np.testing.assert_allclose(by_rule, closed, rtol=1e-9, atol=1e-15)
    assert density.knot_passes[1] == 0


# ------------------------------------------------------------------ W3b


W3B_EDGES = np.array([0.0, 0.3, 0.5, 1.2, 1.4, 2.5, 2.7, 4.0])


def _exponential(slope: float, edges: np.ndarray = W3B_EDGES) -> tuple[np.ndarray, Any]:
    total = np.expm1(slope * edges[-1]) / slope

    def pdf(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(slope * x) / total, dtype=float)

    p = (np.exp(slope * edges[1:]) - np.exp(slope * edges[:-1])) / slope / total
    return p[None, :], pdf


def _points(edges: np.ndarray, per_bin: int = 41) -> np.ndarray:
    return np.concatenate(
        [np.linspace(edges[b], edges[b + 1], per_bin) for b in range(edges.size - 1)]
    )


def test_w3b_an_exponential_law_is_read_exactly_on_unequal_widths() -> None:
    p, pdf = _exponential(-1.3)
    occupied = np.ones(p.shape[1], dtype=bool)
    knots = _knots(p, W3B_EDGES, occupied)
    x = _points(W3B_EDGES)
    read = density_module._read(knots, x[None, :])[0]
    np.testing.assert_allclose(read, pdf(x), rtol=1e-12)
    assert knots.capped == 0


def test_w3b_a_steep_exponential_is_read_exactly_without_the_floor() -> None:
    p, pdf = _exponential(-16.0)
    occupied = np.ones(p.shape[1], dtype=bool)
    x = _points(W3B_EDGES)
    knots = _knots(p, W3B_EDGES, occupied, floor=0.0)
    np.testing.assert_allclose(density_module._read(knots, x[None, :])[0], pdf(x), rtol=1e-11)
    assert knots.capped == 0
    # With the shipped floor, a bin holding less than 1e-10 reads its probability flat.
    floored = p[0] < density_module._ACTIVE_FLOOR
    assert floored.any()
    shipped = _knots(p, W3B_EDGES, occupied)
    read = density_module._read(shipped, x[None, :])[0]
    for b in np.flatnonzero(floored):
        inside = (x > W3B_EDGES[b]) & (x < W3B_EDGES[b + 1])
        np.testing.assert_allclose(read[inside], p[0, b] / np.diff(W3B_EDGES)[b], rtol=1e-12)


def test_w3b_four_passes_are_not_exact() -> None:
    """The control: the iteration cut at four passes misses the exponential law."""
    p, pdf = _exponential(-1.3)
    knots = _knots(p, W3B_EDGES, np.ones(p.shape[1], dtype=bool), passes=4)
    x = _points(W3B_EDGES)
    error = np.max(np.abs(density_module._read(knots, x[None, :])[0] / pdf(x) - 1.0))
    assert error > 1e-6


# ------------------------------------------------------------------ W3c, W3d


def test_w3c_the_reading_makes_no_new_extremum_inside_a_segment() -> None:
    density = _walk()
    knots = density._knots()
    e = density.edges
    centres = 0.5 * (e[:-1] + e[1:])
    for row in range(density.n):
        for first, last in _segments(knots.link[row]):
            for b in range(first, last):
                x = np.linspace(centres[b], centres[b + 1], 33)
                values = np.full((density.n, x.size), centres[b])
                values[row] = x
                logs = np.log(density_module._read(knots, values)[row])
                low = min(knots.value[row, b], knots.value[row, b + 1])
                high = max(knots.value[row, b], knots.value[row, b + 1])
                assert np.all(logs >= low - 1e-12) and np.all(logs <= high + 1e-12)


def test_w3d_empty_bins_floors_and_outer_bins_break_segments() -> None:
    density = _walk()
    knots = density._knots()
    p, e = density.bin_probabilities, density.edges
    flat = p / np.diff(e)
    # Bin 12 lies between the empty bins 11 and 13: it reads its probability over its width.
    for point in np.linspace(e[12], e[13], 11)[1:-1]:
        read = density.density_at(np.full(density.n, point))
        np.testing.assert_allclose(read, flat[:, 12], rtol=1e-14)
    # The floored bin breaks only the rows that hold it below the floor.
    floored = np.arange(density.n) < density.n // 4
    assert not knots.link[floored, WALK_FLOORED - 1].any()
    assert not knots.link[floored, WALK_FLOORED].any()
    assert knots.link[~floored, WALK_FLOORED - 1].all()
    assert knots.link[~floored, WALK_FLOORED].all()
    # An isolated outer bin: the first bin with the second empty.
    occupied = np.ones(WALK_WIDTHS.size, dtype=bool)
    occupied[1] = False
    outer = ConditionalDensity(density.bin_probabilities, e, occupied)
    for point in np.linspace(e[0], e[1], 11)[1:-1]:
        read = outer.density_at(np.full(density.n, point))
        np.testing.assert_allclose(read, flat[:, 0], rtol=1e-14)


# ------------------------------------------------------------------ W3e, W3f


def test_w3e_every_reader_reads_one_representation() -> None:
    frame = law.draw("B", 600, 11)
    density = _exact("B", frame, support=True, clip=True)
    a = frame["A"].to_numpy()
    p, e, occupied, bounds = (
        density.bin_probabilities,
        density.edges,
        density.occupied,
        density.support,
    )
    for values in (a, a - 1.0, np.full(a.size, 3.3)):
        np.testing.assert_array_equal(
            _lookup(p, e, occupied, bounds, values)[:, 0], density.density_at(values)
        )
    lower, upper = a - 0.7, a + 0.4
    mean = _lookup_mean(p, e, occupied, bounds, lower, upper)[:, 0]
    np.testing.assert_allclose(
        mean * (upper - lower), density.mass_between(lower, upper), rtol=1e-12, atol=1e-300
    )


def test_w3f_a_mixed_law_integrates_a_partial_bin_in_closed_form() -> None:
    p, _ = _exponential(-1.3)
    n = 4
    rows = np.repeat(p, n, axis=0)
    continuous = ConditionalDensity(rows, W3B_EDGES, np.ones(p.shape[1], dtype=bool))
    atoms = np.array([[0.1], [0.2], [0.3], [0.4]])
    mixed = MixedDensity((0.0,), atoms, continuous)
    # [0.6, 1.0] lies inside the bin [0.5, 1.2], which sits inside one multi-bin segment.
    total = np.expm1(-1.3 * W3B_EDGES[-1]) / -1.3
    exact = (np.exp(-1.3 * 1.0) - np.exp(-1.3 * 0.6)) / -1.3 / total
    np.testing.assert_allclose(
        mixed.mass_between(0.6, 1.0), (1.0 - atoms[:, 0]) * exact, rtol=1e-12
    )
    linear = (1.0 - atoms[:, 0]) * p[0, 2] * 0.4 / 0.7
    assert np.all(np.abs(linear / ((1.0 - atoms[:, 0]) * exact) - 1.0) > 1e-3)


# ------------------------------------------------------------------ W3g


def _dirichlet_steep(n: int = 30, seed: int = 0) -> ConditionalDensity:
    """Raw Dirichlet probabilities over widths differing 50-fold: the iteration hits its cap."""
    rng = np.random.default_rng(seed)
    widths = np.where(np.arange(12) % 2 == 0, 0.05, 2.5)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    p = rng.dirichlet(np.ones(widths.size), n)
    return ConditionalDensity(p, edges, np.ones(widths.size, dtype=bool))


def test_w3g_rows_at_the_pass_cap_keep_one_representation_and_their_masses() -> None:
    density = _dirichlet_steep()
    passes, capped = density.knot_passes
    assert passes == density_module._KNOT_PASSES
    assert capped > 0
    each = [
        _knots(density.bin_probabilities[row : row + 1], density.edges, density.occupied).capped
        for row in range(density.n)
    ]
    assert capped == sum(each)
    p, e, occupied = density.bin_probabilities, density.edges, density.occupied
    x = np.random.default_rng(1).uniform(e[0], e[-1], density.n)
    np.testing.assert_array_equal(_lookup(p, e, occupied, None, x)[:, 0], density.density_at(x))
    lower, upper = x - 1.0, x + 0.5
    mean = _lookup_mean(p, e, occupied, None, lower, upper)[:, 0]
    np.testing.assert_allclose(mean * 1.5, density.mass_between(lower, upper), rtol=1e-12)
    total = density.mass_between(np.full(density.n, e[0]), np.full(density.n, e[-1]))
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-12)


def _numeric_total(density: ConditionalDensity, nodes: int) -> np.ndarray:
    """``int g^`` per row by a ``nodes``-point rule on every half bin, read at the reading."""
    e = density.edges
    x, w = np.polynomial.legendre.leggauss(nodes)
    total = np.zeros(density.n)
    for b in range(density.n_bins):
        centre = 0.5 * (e[b] + e[b + 1])
        for left, right in ((e[b], centre), (centre, e[b + 1])):
            points = 0.5 * (right - left) * x + 0.5 * (right + left)
            read = density_module._read(density._knots(), np.tile(points, (density.n, 1)))
            total += 0.5 * (right - left) * read @ w
    return total


def test_w3g_the_capped_reading_integrates_to_its_bin_masses_numerically() -> None:
    """The evaluated reading, not only the closed form, integrates to one at the cap."""
    density = _dirichlet_steep()
    assert density.knot_passes[1] > 0
    np.testing.assert_allclose(_numeric_total(density, 200), 1.0, rtol=0, atol=4e-13)


#: A support cut leaves a sliver of a bin beside a wide end bin of a two-bin segment, whose
#: secant reaches -3,705: the integral of a half bin taken from the knot overflowed, and the
#: segment kept its unscaled integral, 1.0153.
SLIVER_EDGES = np.array(
    [2.6916884599596416, 4.390315986019158, 4.653913866458826, 5.810636349280241, 7.782916569913741]
)
SLIVER_P = np.array(
    [[0.3526788054780303, 0.6473211942004394, 2.921383534707928e-10, 2.9391921156518804e-11]]
)
SLIVER_SUPPORT = (np.array([4.6492475874067996]), np.array([6.7554954895816355]))


def test_w3a_a_steep_segment_keeps_its_mass() -> None:
    density = ConditionalDensity(SLIVER_P, SLIVER_EDGES, np.ones(4, dtype=bool), SLIVER_SUPPORT)
    assert abs(density._knots().secant[0, 1]) > 3000.0
    e = SLIVER_EDGES
    total = density.mass_between(np.array([e[0] - 1.0]), np.array([e[-1] + 1.0]))
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(density.integrated(), total, rtol=0, atol=1e-15)


# ------------------------------------------------------------------ W4


def _bins_of(a: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(a, edges)
    return counts


@pytest.mark.parametrize(("n", "core"), [(200, 20), (2000, 26), (2000, 320), (20000, 55)])
def test_w4a_the_core_survives_and_every_split_bin_holds_its_rows(n: int, core: int) -> None:
    a = law.draw("G", n, 1).A.to_numpy()
    base = _equal_mass_edges(a, core, None)
    edges = bin_edges(a, core)
    assert np.all(np.isin(base, edges))
    assert base.size - 1 <= edges.size - 1
    cap = density_module._width_cap(base)
    assert edges.size - 1 <= int(np.sum(np.ceil(np.diff(base) / cap)))
    for b in range(base.size - 1):
        inner = edges[(edges >= base[b]) & (edges <= base[b + 1])]
        if inner.size == 2:
            continue
        # A gap bin (no rows) ends a stretch: each stretch's bins hold min(20, m) of its rows.
        counts = _bins_of(a, inner)
        for stretch in np.split(counts, np.flatnonzero(counts == 0)):
            held = stretch[stretch > 0]
            if held.size:
                assert held.min() >= min(TAIL_ROWS, int(held.sum())), (b, counts)


def _upper_tail_bins(a: np.ndarray, clean_max: float) -> int:
    edges = bin_edges(a, default_density_bins(a.size))
    q99 = np.quantile(a[a <= clean_max], 0.99)
    return int(np.sum((edges[:-1] >= q99) & (edges[:-1] <= clean_max)))


def _outlier_counts() -> list[tuple[int, int]]:
    out = []
    for d in range(3):
        a = law.draw("G", 20000, 777 * 20000 + d).A.to_numpy()
        clean_max = float(a.max())
        out.append(
            (
                _upper_tail_bins(a, clean_max),
                _upper_tail_bins(np.append(a, clean_max + 25.0), clean_max),
            )
        )
    return out


def test_w4b_one_outlier_moves_the_upper_tail_by_at_most_one_bin() -> None:
    counts = _outlier_counts()
    assert all(abs(after - before) <= 1 for before, after in counts), counts


def test_w4b_a_range_based_width_cap_is_moved_by_the_outlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: a width cap read from the whole range shrinks the tail at the outlier."""
    monkeypatch.setattr(
        density_module, "_width_cap", lambda core: float((core[-1] - core[0]) / (core.size - 1))
    )
    counts = _outlier_counts()
    assert any(abs(after - before) > 1 for before, after in counts), counts


def test_w4c_a_far_component_keeps_its_gap_and_no_bin_is_tiny() -> None:
    for d in range(5):
        a = law.draw("C", 20000, 9100 * 20000 + d + 2 * 100000).A.to_numpy()
        edges = bin_edges(a, default_density_bins(a.size))
        ordered = np.sort(a)
        widest = int(np.argmax(np.diff(ordered)))
        lo, hi = ordered[widest], ordered[widest + 1]
        counts = _bins_of(a, edges)
        empty = (counts == 0) & (edges[:-1] >= lo) & (edges[1:] <= hi)
        assert empty.sum() == 1, (d, lo, hi)
        spans = (edges[:-1] <= lo) & (edges[1:] >= hi)
        assert not spans.any()
    for d in range(5):
        a = law.draw("G", 20000, 9100 * 20000 + d).A.to_numpy()
        counts = _bins_of(a, bin_edges(a, default_density_bins(a.size)))
        assert counts.min() >= 5, (d, counts.min())


def test_w4d_explicit_edges_are_kept_and_built_edges_are_the_whole_sample_s() -> None:
    frame = law.draw("G", 400, 2)
    w = frame[["W1", "W2"]].to_numpy()
    a = frame["A"].to_numpy()
    explicit = np.linspace(a.min() - 0.1, a.max() + 0.1, 9)
    folds = Folds(np.arange(a.size) % 2, 2)
    density, diagnostics = fit_conditional_density(
        law.OracleHazard(tuple(explicit), "G"),
        w,
        a,
        np.ones(a.size),
        folds,
        edges=explicit.tolist(),
    )
    np.testing.assert_array_equal(density.edges, explicit)
    assert diagnostics.core_bins is None
    built = bin_edges(a, 12)
    density, diagnostics = fit_conditional_density(
        law.OracleHazard(tuple(built), "G"), w, a, np.ones(a.size), folds, n_bins=12
    )
    np.testing.assert_array_equal(density.edges, built)
    np.testing.assert_array_equal(density.occupied, bin_occupancy(a, built))
    assert diagnostics.core_bins == 12


def test_w4e_the_preflight_reads_the_refined_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    from cleverly.exceptions import MethodConfigurationError

    rng = np.random.default_rng(4)
    rng.normal(size=(400, 2))
    dose = rng.normal(size=400)
    learner = LogisticRegression()
    refined = _grid(dose, None, 10, None, None)
    core = _equal_mass_edges(dose, 10, None)
    coarse = _grid(dose, None, None, core.tolist(), None)
    assert refined.n_total > coarse.n_total
    at_refined = np.prod(_table_size(learner, 2, refined, None)[:2]) * 8
    at_core = np.prod(_table_size(learner, 2, coarse, None)[:2]) * 8
    free = int(at_core + (at_refined - at_core) // 2)
    monkeypatch.setattr(density_module, "available_memory", lambda: free)
    with pytest.raises(MethodConfigurationError, match="density_bins="):
        preflight_conditional_densities(learner, [(2, dose)], Folds.single(400), n_bins=10)


# ------------------------------------------------------------------ W5


class _CollapseGrid:
    """Dose 2's bins with the doses 2 and 4 in adjacent bins: one two-bin segment."""

    edges = (0.875, 1.25, 1.5, 3.0, 6.0, 7.0, 10.0)


def _collapse_gaps() -> tuple[float, float]:
    from . import test_mtp_component_atoms as atoms_tests
    from . import test_vector_density_route as route

    grids = {"D1": route._Grid, "D2": _CollapseGrid}
    labels = atoms_tests.COLLAPSE_LAST
    with warnings.catch_warnings(), mtp.exact_bins(grids):
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                policies=[atoms_tests.joint(label) for label in labels],
                **route.density_settings(ratio="density"),
            )
            .fit(route.geo_data(None), dose_atoms=atoms_tests.law.ATOMS)
            .single()
        )
    return atoms_tests._gaps(result, labels, longitudinal=False)


@pytest.mark.usefixtures("exact_grid_law")
def test_w5_a_collapse_onto_an_atom_integrates_a_multi_bin_segment() -> None:
    """The interval ``[1.5, 6)`` is the two-bin segment of the doses 2 and 4, so the
    collapse's mass is that segment's total probability only if each bin's overlap is read
    as the closed-form integral of the reading."""
    psi, curve = _collapse_gaps()
    assert psi < 1e-10
    assert curve < 1e-10


@pytest.mark.usefixtures("exact_grid_law")
def test_w5_a_point_read_at_the_midpoint_misses_the_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def midpoint(p: Any, e: Any, occupied: Any, support: Any, lower: Any, upper: Any) -> Any:
        return _lookup(p, e, occupied, support, 0.5 * (np.asarray(lower) + np.asarray(upper)))

    monkeypatch.setattr(density_module, "_lookup_mean", midpoint)
    assert _collapse_gaps()[1] > 1e-3


# ------------------------------------------------------------------ W6


def _remainder(read: str) -> float:
    n = 8000
    frame = law.draw("G", n, 4242 * n)
    w1, w2, a, _ = law.columns(frame)
    if read == "shipped":
        density: Any = _exact("G", frame).density_at
    else:
        edges = law.equal_mass_edges(a, default_density_bins(n))
        p = law.bin_probabilities("G", edges, w1, w2)

        def density(values: Any) -> np.ndarray:
            return law.histogram(p, edges, values)

    gap = law.ratio_from("G", density, a, w1, w2) - law.r0("G", a, w1, w2)
    wrong = law.q0(a, w1, w2) - law.wrong_q(a, w1, w2, frame)
    return float(np.sqrt(n) * np.mean(gap * wrong) / law.EFFICIENCY_SD["G"])


def test_w6_the_remainder_with_a_wrong_outcome_regression_is_small() -> None:
    assert abs(_remainder("shipped")) <= 0.3


def test_w6_the_pre_fix_reading_leaves_a_large_remainder() -> None:
    assert abs(_remainder("histogram")) > 0.3


# ------------------------------------------------------------------ W7


def _recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    records: list[tuple[Any, ...]] = []
    original = density_module._knots_by_block

    def record(*args: Any) -> Any:
        records.append(tuple(None if x is None else np.array(x) for x in args))
        return original(*args)

    monkeypatch.setattr(density_module, "_knots_by_block", record)
    return records


def _same(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return all(
        (x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y))
        for x, y in zip(left, right, strict=True)
    )


@pytest.mark.parametrize("estimator", [LTMLE, SDR], ids=["ltmle", "sdr"])
def test_w7_a_one_node_longitudinal_fit_reads_the_point_density(
    estimator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _recorded(monkeypatch)
    frame = law.draw("U", 600, 3)
    _fit("U", frame)
    point = list(records)
    records.clear()
    bins = default_density_bins(len(frame))
    edges = tuple(bin_edges(frame["A"].to_numpy(), bins))
    regimens = {
        "natural": DynamicRegimen("natural", (Shift(0.0, cap=None),)),
        "p2": DynamicRegimen("p2", (_policy("U"),)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator(
            regimens,
            reference="natural",
            n_folds=1,
            q_bounds=(-60.0, 60.0),
            outcome_learner=LinearRegression(),
            pseudo_learner=LinearRegression(),
            treatment_learner=law.OracleHazard(edges, "U"),
            density_bins=bins,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A"],
            baseline=["W1", "W2"],
            time_varying=[[]],
            continuous_treatment={"A": _support("U")},
        )
    assert len(point) == len(records) == 1
    assert point[0][4] is not None
    assert _same(point[0], records[0])


# ------------------------------------------------------------------ W8


def test_w8_the_flat_share_counts_rows_kept_in_flat_bins() -> None:
    density = _walk()
    e = density.edges
    observed = np.full(density.n, e[12] + 0.1)
    assert density.flat_fraction(observed + 0.2, observed) == 1.0
    frame = law.draw("G", 2000, 5)
    exact = _exact("G", frame)
    a = frame["A"].to_numpy()
    assert exact.flat_fraction(a + 1e-6, a) == 0.0


# ------------------------------------------------------------------ W9


def _support_reading(
    name: str, n: int, draw: int, *, support: bool = True, core: int | None = None
) -> tuple[float, float, float]:
    frame = law.draw(name, n, 7700 * n + draw + (0 if name == "B" else 50000))
    w1, w2, a, _ = law.columns(frame)
    edges = bin_edges(a, default_density_bins(n) if core is None else core)
    density = _exact(name, frame, edges, support=support, clip=True)
    truth = law.pdf(name, a, w1, w2)
    share = density.density_at(a) / truth
    return (
        float(share.min()),
        float(np.median(np.abs(np.log(share)))),
        law.ratio_error(name, _ratio(name, density.density_at, frame), frame),
    )


@pytest.mark.parametrize("name", ["U", "B"])
def test_w9ab_a_declared_support_is_read_row_by_row(name: str) -> None:
    for d in range(5):
        smallest, median, error = _support_reading(name, 2000, d)
        assert smallest >= 0.85 and error <= 0.02, (name, d, smallest, error)
        if name == "U":
            assert median <= 0.002, (d, median)
    if name == "B":
        for d in range(3):
            smallest, _, error = _support_reading("B", 2000, d, core=oracle_bins(2000))
            assert smallest >= 0.85 and error <= 0.02, (d, smallest, error)


def test_w9ab_ignoring_the_declared_support_fails() -> None:
    assert min(_support_reading("U", 2000, d, support=False)[0] for d in range(5)) < 0.85
    assert min(_support_reading("B", 2000, d, support=False)[0] for d in range(5)) < 0.85


def test_w9c_a_straddling_bin_is_read_over_its_supported_part() -> None:
    slope, stop = -1.3, 2.0
    edges = W3B_EDGES
    total = np.expm1(slope * stop) / slope
    capped = np.minimum(edges, stop)
    p = (np.exp(slope * capped[1:]) - np.exp(slope * capped[:-1])) / slope / total
    occupied = p > 0.0
    rows = np.repeat(p[None, :], 2, axis=0)
    density = ConditionalDensity(rows, edges, occupied, (np.zeros(2), np.full(2, stop)))
    x = _points(edges)
    read = np.array([density.density_at(np.full(2, point))[0] for point in x])
    inside = x <= stop
    np.testing.assert_allclose(read[inside], np.exp(slope * x[inside]) / total, rtol=1e-12)
    assert np.all(read[~inside] == 0.0)


def _icb_seed(name: str, n: int, draw: int) -> int:
    return 3500 * n + draw + (0 if name == "B" else 50000)


def test_w9d_the_influence_curve_gap_falls_under_a_declared_support() -> None:
    small = [_ic_gap("U", 2000, _icb_seed("U", 2000, d)) for d in range(5)]
    large = [_ic_gap("U", 20000, _icb_seed("U", 20000, d)) for d in range(2)]
    assert np.median(small) <= 0.04, small
    assert max(large) <= 0.03, large


def test_w9d_ignoring_the_declared_support_fails() -> None:
    small = [_ic_gap("U", 2000, _icb_seed("U", 2000, d), support=False) for d in range(5)]
    assert np.median(small) > 0.04, small


class _AtomOrHazard(law.OracleHazard):
    """The atom classifier on ``[W1, W2]`` (a constant share) and the oracle hazard."""

    def __init__(self, edges: tuple[float, ...] = (), law: str = "B", share: float = 0.2) -> None:
        super().__init__(edges, law)
        self.share = share

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        self.classes_ = np.unique(np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        if design.shape[1] == 2:
            return np.column_stack(
                [np.full(design.shape[0], 1.0 - self.share), np.full(design.shape[0], self.share)]
            )
        return super().predict_proba(X)


def _mixed_b(support: bool) -> tuple[MixedDensity, pd.DataFrame, np.ndarray]:
    frame = law.draw("B", 2000, 7700 * 2000 + 3)
    w1, w2, a, _ = law.columns(frame)
    at_atom = np.random.default_rng(2).random(a.size) < 0.2
    dose = np.where(at_atom, 0.0, a)
    edges = tuple(bin_edges(a[~at_atom], default_density_bins(int((~at_atom).sum()))))
    bounds = (np.zeros(a.size), law.upper("B", w1, w2)) if support else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed, _ = fit_mixed_density(
            _AtomOrHazard(edges, "B"),
            frame[["W1", "W2"]].to_numpy(),
            dose,
            np.ones(a.size),
            Folds.single(a.size),
            atoms=(0.0,),
            support=bounds,
        )
    return mixed, frame, at_atom


def test_w9e_a_mixed_dose_reads_its_continuous_part_over_the_declared_support() -> None:
    mixed, frame, at_atom = _mixed_b(True)
    w1, w2, a, _ = law.columns(frame)
    assert mixed.continuous is not None
    share = mixed.continuous.density_at(a)[~at_atom] / law.pdf("B", a, w1, w2)[~at_atom]
    assert share.min() >= 0.85
    above = law.upper("B", w1, w2) + 1e-6
    assert np.all(mixed.continuous.density_at(above) == 0.0)


def test_w9e_without_the_support_the_continuous_part_fails() -> None:
    mixed, frame, at_atom = _mixed_b(False)
    w1, w2, a, _ = law.columns(frame)
    assert mixed.continuous is not None
    share = mixed.continuous.density_at(a)[~at_atom] / law.pdf("B", a, w1, w2)[~at_atom]
    assert share.min() < 0.85


# ------------------------------------------------------------------ W10


def _restricted_mechanism(frame: pd.DataFrame, rows: np.ndarray | None) -> Any:
    a = frame["A"].to_numpy()
    bins = default_density_bins(a.size)
    edges = tuple(bin_edges(a, bins))
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A"],
        baseline=["W1", "W2"],
        time_varying=[[]],
        continuous_treatment={"A": None},
    )
    spec = {"shift": DynamicRegimen("shift", (Shift(1.0, cap=None),), rule_kind="known")}
    plans = resolve_plans(resolve_regimens(spec, data.n_times), data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_mechanism(
            data,
            plans,
            treatment_learner=law.OracleHazard(edges, "G"),
            censoring_learner=LogisticRegression(),
            folds=Folds.single(a.size),
            ratio="density",
            density_bins=bins,
            fit_rows=rows,
        )


def test_w10_a_training_fold_reads_the_occupancy_of_every_row_at_risk() -> None:
    frame = law.draw("G", 400, 5)
    frame.loc[0, "A"] = 14.0
    a = frame["A"].to_numpy()
    rows = np.arange(a.size) % 4 != 0
    edges = bin_edges(a, default_density_bins(a.size))
    # The premise: the fold's own rows leave the far dose's one-row bin empty.
    assert not np.array_equal(bin_occupancy(a[rows], edges), bin_occupancy(a, edges))
    restricted = _restricted_mechanism(frame, rows)
    everyone = _restricted_mechanism(frame, None)
    ((key, fold),) = restricted.densities.items()
    whole = everyone.densities[key]
    np.testing.assert_array_equal(fold.occupied, whole.occupied)
    held = ~rows
    far = np.full(a.size, 13.9)
    np.testing.assert_array_equal(fold.density_at(a)[held], whole.density_at(a)[held])
    np.testing.assert_array_equal(fold.density_at(far)[held], whole.density_at(far)[held])
    assert np.all(whole.density_at(far)[held] > 0.0)


# ------------------------------------------------------------------ W12


def test_w12_the_knots_are_computed_once_per_density_and_never_pickled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _recorded(monkeypatch)
    frame = law.draw("G", 20000, _ic_seed("G", 20000, 0))
    start = time.perf_counter()
    result = _fit("G", frame)
    elapsed = time.perf_counter() - start
    assert len(records) == 1
    assert elapsed < 30.0
    density = result.nuisance.density
    assert "_reading" in density.__dict__
    restored = pickle.loads(pickle.dumps(density))
    assert "_reading" not in restored.__dict__
    np.testing.assert_array_equal(
        restored.density_at(frame["A"].to_numpy()), density.density_at(frame["A"].to_numpy())
    )


def test_w12_without_the_memo_the_knots_are_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: a density that forgets its knots recomputes them at every read."""
    records = _recorded(monkeypatch)

    def forgetful(self: ConditionalDensity) -> Any:
        lower, upper = (None, None) if self.support is None else self.support
        return density_module._knots_by_block(
            self.bin_probabilities, self.edges, self.occupied, lower, upper
        )

    monkeypatch.setattr(ConditionalDensity, "_knots", forgetful)
    _fit("G", law.draw("G", 2000, 0))
    assert len(records) > 1


# ------------------------------------------------------------------ the dose-response rule


def _kernel_mass(density: ConditionalDensity, breaks: np.ndarray) -> np.ndarray:
    curve = DoseResponse(doses=(5.0,), bandwidths=(2.0,), support=(0.5, 10.0), kernel="uniform")
    rule = curve.quadrature(breaks)
    out = np.zeros(density.n)
    for node, weight in zip(rule.nodes, rule.weights[0], strict=True):
        out += weight * density.density_at(np.full(density.n, node))
    return out


def test_the_dose_response_rule_breaks_at_the_bin_centres() -> None:
    """The reading is exponential on each half bin, so a rule broken at the edges and the
    centres integrates it to the closed form, and one broken at the edges alone does not."""
    density = _walk()
    e = density.edges
    # The uniform kernel's weight is 1 / (2 h) on [a0 - h, a0 + h] = [3, 7].
    closed = density.mass_between(np.full(density.n, 3.0), np.full(density.n, 7.0)) / 4.0
    centres = 0.5 * (e[:-1] + e[1:])
    with_centres = _kernel_mass(density, np.concatenate([e, centres]))
    np.testing.assert_allclose(with_centres, closed, rtol=1e-10)
    edges_only = _kernel_mass(density, e)
    assert np.max(np.abs(edges_only / closed - 1.0)) > 1e-8
