"""The drift correction's carried products and kernel scan keep the baseline's arithmetic.

X106 part J3a.  ``fit_coefficients`` carries each ``B_k`` and ``D_k`` product forward one factor
per horizon step and releases every fitted regression once it is multiplied in; the kernel
smoother bins with one ``bincount`` and scans the bandwidths with a shared right-bin index.  Each
change is meant to be bitwise, so each is compared here against a verbatim copy of the code it
replaced, on fixtures whose products are nonzero, with a mutation control that shows the
comparison can fail.
"""

from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np
import pytest

from cleverly.longitudinal import SurvivalDRTMLE, _kernel, drift
from cleverly.longitudinal._kernel import BINS
from tests.unit.test_survival_drtmle import ARMS, container, glm, sample_frame

GB = (0.01, 1.0)


# ------------------------------------------------------------------ verbatim baseline code


def _baseline_fit_coefficients(
    reductions: drift._Reductions, tau: int, g_lower: float
) -> drift._Coefficients:
    """``fit_coefficients`` as it stood before X106 J3a (main 70ccd44), drift on."""
    view = reductions.view
    n = view.n
    hits = {"q": 0, "B": 0, "D": 0}
    a = np.zeros((n, tau))
    c = np.zeros((n, tau))
    r = [np.zeros((n, tau)) for _ in range(tau)]
    l_terms = [np.zeros((n, tau)) for _ in range(tau)]
    e_l = {t: reductions.e_l(t) for t in range(1, tau + 1)}
    if np.any([np.any(value != 0.0) for value in e_l.values()]):
        for t in range(1, tau + 1):
            u = [reductions.u(t, m) for m in range(t)]
            v = [reductions.v(t, m) for m in range(1, t)]
            a[:, t - 1] = drift._forward(u, n) * drift._forward(v, n) * e_l[t]
            for k in range(t):
                r[k][:, t - 1] = drift._forward(v[k:], n) * drift._forward(u[k:], n) * e_l[t]
    e_r = {k: reductions.e_r(k) for k in range(tau)}
    e_a = reductions.e_a()
    q_raw = reductions.q(g_lower)
    q = np.clip(q_raw, g_lower, 1.0 - g_lower)
    hits["q"] = int(np.sum(q != q_raw))
    floor = drift.DENOMINATOR_FLOOR
    for t in range(1, tau + 1):
        for k in range(t):
            if not np.any(e_r[k] != 0.0):
                continue
            b_prod = drift._forward([reductions.b(k, m) for m in range(k + 1, t)], n)
            d_prod = drift._forward([reductions.d(k, m) for m in range(k, t)], n)
            hits["B"] += int(np.sum(b_prod < floor))
            hits["D"] += int(np.sum(d_prod < floor))
            l_terms[t - 1][:, k] = -e_r[k] / (np.maximum(b_prod, floor) * np.maximum(d_prod, floor))
        if np.any(e_a != 0.0):
            index = drift._a_term_index(t)
            d_full = drift._forward([reductions.d(index, m) for m in range(t)], n)
            b_full = drift._forward([reductions.b(index, m) for m in range(1, t)], n)
            hits["B"] += int(np.sum(b_full < floor))
            hits["D"] += int(np.sum(d_full < floor))
            c[:, t - 1] = e_a / (q * np.maximum(d_full, floor) * np.maximum(b_full, floor))
    return drift._Coefficients(a, tuple(r), tuple(l_terms), c, hits)


def _baseline_bin(
    x: np.ndarray, y: np.ndarray, low: float, width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = (x - low) / width
    left = np.clip(np.floor(position).astype(np.intp), 0, BINS - 2)
    right_weight = np.clip(position - left, 0.0, 1.0)
    left_weight = 1.0 - right_weight
    counts = np.zeros(BINS)
    sums = np.zeros(BINS)
    np.add.at(counts, left, left_weight)
    np.add.at(counts, left + 1, right_weight)
    np.add.at(sums, left, left_weight * y)
    np.add.at(sums, left + 1, right_weight * y)
    return left, left_weight, right_weight, np.stack([counts, sums])


def _baseline_loo_error(
    y: np.ndarray,
    left: np.ndarray,
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    sums: np.ndarray,
    kernel: np.ndarray,
) -> float:
    near = kernel[BINS - 1]
    step = kernel[BINS]
    own_left = near * left_weight + step * right_weight
    own_right = step * left_weight + near * right_weight
    fits = np.zeros(y.size)
    for bins, own, weight in ((left, own_left, left_weight), (left + 1, own_right, right_weight)):
        denominator = sums[0, bins] - own
        numerator = sums[1, bins] - own * y
        safe = denominator > 1e-300
        ratio = np.where(safe, numerator / np.where(safe, denominator, 1.0), np.nan)
        fits += weight * ratio
    residual = y - fits
    finite = np.isfinite(residual)
    if not finite.any():
        return float("inf")
    worst = float(np.max(np.abs(residual[finite]))) + 1.0
    return float(np.mean(np.where(finite, residual, worst) ** 2))


# ------------------------------------------------------------------------------ fixtures


def _views(data: Any, horizons: list[float]) -> dict[int, drift.PaperView]:
    """The paper views of a fit's own out-of-fold nuisances, by arm code."""
    estimator = SurvivalDRTMLE(
        ARMS,
        reference="a0",
        outcome_learner=glm(),
        treatment_learner=glm(),
        censoring_learner=glm(),
        n_folds=2,
        random_state=0,
        horizons=horizons,
    )
    captured: dict[int, drift.PaperView] = {}

    def spy(prepared: Any, plans: Any, mechanism: Any, initial: Any, **kwargs: Any) -> Any:
        for code in (0, 1):
            captured[code] = drift.paper_view(
                prepared,
                mechanism,
                initial,
                code,
                alpha=kwargs["alpha"],
                g_bounds=kwargs["g_bounds"],
            )
        raise _Captured

    original = drift.run_drift
    drift.run_drift = spy  # type: ignore[assignment]
    try:
        with pytest.raises(_Captured):
            estimator.fit(data)
    finally:
        drift.run_drift = original  # type: ignore[assignment]
    return captured


class _Captured(Exception):
    """Stops the fit once the views are captured."""


@pytest.fixture(scope="module")
def nine_node_views() -> dict[int, drift.PaperView]:
    return _views(container(sample_frame(600, 31, n_times=9)), [9.0])


def _held(view: drift.PaperView, *, treatment: bool, censoring: tuple[int, ...]) -> drift.PaperView:
    held_censoring = np.array(view.held_censoring, dtype=bool, copy=True)
    held_censoring[list(censoring)] = True
    return drift.PaperView(
        **{**view.__dict__, "held_treatment": treatment, "held_censoring": held_censoring}
    )


def _pair(view: drift.PaperView, tau: int) -> tuple[Any, Any, Any, Any]:
    def reductions() -> drift._Reductions:
        state = drift._State(view.g_a.copy(), view.g_r.copy(), view.h.copy())
        return drift._Reductions(view, state, tau, GB)

    ours, theirs = reductions(), reductions()
    new = drift.fit_coefficients(ours, tau, GB[0])
    old = _baseline_fit_coefficients(theirs, tau, GB[0])
    return new, old, ours, theirs


def _assert_same(new: drift._Coefficients, old: drift._Coefficients) -> None:
    assert new.a.tobytes() == old.a.tobytes()
    assert new.c.tobytes() == old.c.tobytes()
    assert [x.tobytes() for x in new.r] == [x.tobytes() for x in old.r]
    assert [x.tobytes() for x in new.l] == [x.tobytes() for x in old.l]
    assert new.hits == old.hits


# ---------------------------------------------------------------------------------- tests


@pytest.mark.parametrize("tau", [1, 2, 5, 8, 9])
@pytest.mark.parametrize("code", [0, 1])
def test_carried_products_are_the_baseline_products(
    nine_node_views: dict[int, drift.PaperView], tau: int, code: int
) -> None:
    new, old, ours, theirs = _pair(nine_node_views[code], tau)
    _assert_same(new, old)
    # Same regressions, fitted once each, in the same order: the record order is public.
    assert ours.records == theirs.records
    # Nothing a later horizon step reads is left cached: only the families read once.
    assert all(key[0] in ("e_L", "e_R", "e_A", "q") for key in ours._cache)
    if tau > 1:
        # Nonzero witness: every product the change touches carries weight on the fixture.
        assert np.any(new.c != 0.0) and any(np.any(x != 0.0) for x in new.l[1:])
        assert any(np.any(x != 0.0) for x in new.r)


@pytest.mark.parametrize(
    ("treatment", "censoring"), [(True, ()), (False, (0, 3)), (True, (1, 2, 4))]
)
def test_held_factors_skip_the_same_branches(
    nine_node_views: dict[int, drift.PaperView], treatment: bool, censoring: tuple[int, ...]
) -> None:
    """A held treatment zeroes ``e_A``; a held censoring node zeroes its ``e_R(k)`` branch."""
    view = _held(nine_node_views[1], treatment=treatment, censoring=censoring)
    new, old, ours, theirs = _pair(view, 7)
    _assert_same(new, old)
    assert ours.records == theirs.records
    if treatment:
        assert not np.any(new.c)
    for k in censoring:
        assert all(not np.any(x[:, k]) for x in new.l)


def test_the_comparison_sees_a_one_ulp_factor(
    nine_node_views: dict[int, drift.PaperView], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a carried factor off by one ulp makes the products differ."""
    view = nine_node_views[1]
    original = drift._Reductions.b

    def nudged(self: drift._Reductions, k: int, m: int) -> np.ndarray:
        value = original(self, k, m)
        return np.nextafter(value, 2.0) if (k, m) == (0, 3) else value

    state = drift._State(view.g_a.copy(), view.g_r.copy(), view.h.copy())
    old = _baseline_fit_coefficients(drift._Reductions(view, state, 6, GB), 6, GB[0])
    monkeypatch.setattr(drift._Reductions, "b", nudged)
    state = drift._State(view.g_a.copy(), view.g_r.copy(), view.h.copy())
    new = drift.fit_coefficients(drift._Reductions(view, state, 6, GB), 6, GB[0])
    assert new.c.tobytes() != old.c.tobytes()


def test_regressions_are_released_after_the_products(monkeypatch: pytest.MonkeyPatch) -> None:
    """Few regressions stay cached, and the peak net of the fixed products is O(tau).

    The baseline cached every regression to the end of the horizon, about 2 tau^2 arrays
    (146 entries at tau = 8); the fixed products ``r`` and ``l`` are another 2 tau^2.
    """
    rng = np.random.default_rng(5)
    n, tau = 4000, 8
    k = tau + 1
    view = drift.PaperView(
        arm=rng.binomial(1, 0.5, n).astype(float),
        g_a=rng.uniform(0.3, 0.7, n),
        g_r=rng.uniform(0.0, 0.1, (n, k)),
        h=rng.uniform(0.05, 0.2, (n, k)),
        at_risk=np.ones((n, k), dtype=bool),
        events=rng.binomial(1, 0.1, (n, k)).astype(float),
        eligible=np.ones((n, k), dtype=bool),
        censored=rng.binomial(1, 0.05, (n, k)).astype(float),
        held_treatment=False,
        held_censoring=np.zeros(k, dtype=bool),
        held_hazard=np.zeros(k, dtype=bool),
    )

    def regress(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return 0.5 * x + 0.01

    def run(function: Any) -> tuple[int, int, int]:
        state = drift._State(view.g_a.copy(), view.g_r.copy(), view.h.copy())
        reductions = drift._Reductions(view, state, tau, GB, regress)
        held = [0]
        original = drift._Reductions.fit

        def counting(self: drift._Reductions, *args: Any) -> np.ndarray:
            value = original(self, *args)
            held[0] = max(held[0], len(self._cache))
            return value

        monkeypatch.setattr(drift._Reductions, "fit", counting)
        tracemalloc.start()
        try:
            function(reductions, tau, GB[0])
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
            monkeypatch.undo()
        return len(reductions._cache), held[0], peak

    array = 8 * n
    fixed = 2 * tau * tau * array
    left, most, peak = run(drift.fit_coefficients)
    theirs_left, _, theirs_peak = run(_baseline_fit_coefficients)
    assert left <= 2 * tau + 2  # e_L, e_R, e_A and q only
    assert theirs_left > 8 * left  # the baseline kept every regression (146 at tau = 8)
    assert most <= 4 * tau + 2  # the e_L, e_R families (2 tau) and one horizon step of u, v
    assert peak - fixed <= 10 * tau * array
    assert theirs_peak - fixed > 2 * (peak - fixed)


def test_the_loo_shortcut_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control for the all-finite early exit: taking it when only some residuals
    are finite changes the error on an input with a gap."""
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.uniform(0.0, 1.0, 3000), [3.0, 7.5]])
    y = rng.binomial(1, 0.5, x.size).astype(float)
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    low, width = float(x[0]), float(x[-1] - x[0]) / (BINS - 1)
    left, left_weight, right_weight, binned = _kernel._bin(x, y, low, width)
    bandwidth = float(_kernel.bandwidth_grid(x, width)[0])
    kernel = _kernel._kernel(bandwidth, width)
    sums = _kernel._convolve(binned, kernel)
    honest = _kernel._loo_error(y, left, left + 1, left_weight, right_weight, sums, kernel)
    assert honest == _baseline_loo_error(y, left, left_weight, right_weight, sums, kernel)
    # The residuals the scan scores, as both bodies compute them.
    near, step = kernel[BINS - 1], kernel[BINS]
    fits = np.zeros(y.size)
    own_l = near * left_weight + step * right_weight
    own_r = step * left_weight + near * right_weight
    for bins, own, weight in ((left, own_l, left_weight), (left + 1, own_r, right_weight)):
        denominator = sums[0, bins] - own
        safe = denominator > 1e-300
        ratio = np.where(safe, (sums[1, bins] - own * y) / np.where(safe, denominator, 1.0), np.nan)
        fits += weight * ratio
    residual = y - fits
    assert np.isfinite(residual).any() and not np.isfinite(residual).all()
    # The early exit taken on ``finite.any()`` would score mean(residual**2): not finite.
    mutated = float(np.mean(residual**2))
    assert np.isfinite(honest) and not np.isfinite(mutated)


def test_bincount_binning_is_the_add_at_binning() -> None:
    rng = np.random.default_rng(3)
    cases = [
        rng.uniform(-1.0, 1.0, 5000),
        np.round(rng.uniform(0.0, 1.0, 5000), 2),  # ties
        np.concatenate([np.full(4000, 0.5), rng.uniform(0.0, 1.0, 1000)]),  # one dense bin
        np.linspace(0.0, 1.0, 20),  # both clip edges, the smallest smoothed subpopulation
    ]
    for x in cases:
        y = rng.binomial(1, 0.4, x.size).astype(float)
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
        low, width = float(x[0]), float(x[-1] - x[0]) / (BINS - 1)
        ours = _kernel._bin(x, y, low, width)
        theirs = _baseline_bin(x, y, low, width)
        for one, two in zip(ours, theirs, strict=True):
            assert one.tobytes() == two.tobytes()


@pytest.mark.parametrize("gap", [False, True])
def test_loo_scan_is_the_baseline_scan(gap: bool) -> None:
    """All residuals finite, and a gap that leaves rows with no leave-one-out fit."""
    rng = np.random.default_rng(9)
    x = rng.uniform(0.0, 1.0, 3000)
    if gap:
        # Isolated rows far from the rest: removing one empties its bins at small bandwidths.
        x = np.concatenate([x, [3.0, 7.5]])
    y = rng.binomial(1, 0.3 + 0.4 * np.clip(x, 0.0, 1.0)).astype(float)
    order = np.lexsort((y, x))
    x, y = x[order], y[order]
    low, width = float(x[0]), float(x[-1] - x[0]) / (BINS - 1)
    left, left_weight, right_weight, binned = _kernel._bin(x, y, low, width)
    masked = []
    for bandwidth in _kernel.bandwidth_grid(x, width):
        kernel = _kernel._kernel(float(bandwidth), width)
        sums = _kernel._convolve(binned, kernel)
        ours = _kernel._loo_error(y, left, left + 1, left_weight, right_weight, sums, kernel)
        theirs = _baseline_loo_error(y, left, left_weight, right_weight, sums, kernel)
        assert np.float64(ours).tobytes() == np.float64(theirs).tobytes()
        own = kernel[BINS - 1] * left_weight + kernel[BINS] * right_weight
        masked.append(bool(np.any(sums[0, left] - own <= 1e-300)))
    # Reach: the gap fixture takes the masked branch at some bandwidths, the other never.
    assert any(masked) == gap


def test_observation_weights_are_still_refused() -> None:
    """The change list leaves the weights refusal in place: weighted reductions are not built."""
    frame = sample_frame(400, 33)
    frame["w"] = np.where(frame["W1"] > 0, 2.0, 1.0)
    data = container(frame, weights="w")
    assert data.is_weighted
    assert drift.drift_design_refusal(data, ARMS) is drift.WEIGHTS_REFUSAL
