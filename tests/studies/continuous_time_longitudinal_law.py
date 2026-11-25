r"""The laws of the ``continuous-time-longitudinal`` study (X6 part (e)), with their truths.

**Law CT** (X6 plan rev 7, section 5.7) is a continuous-time version of the setting of
Rytgaard, Gerds and van der Laan (2022, section 9), with two causes added.  The paper's own
coefficients were not read, so the law is the plan's own.  Time is in days on :math:`[0, 30]`,
and every intensity is constant between visits given the history (:class:`Coefficients`):

* :math:`W_1 \sim` Bernoulli(0.5) and :math:`W_2` uniform on :math:`\{1, \dots, 6\}`;
* :math:`L(0) \sim` Bernoulli(expit(:math:`-0.5 + 0.6 W_1`)) and
  :math:`A(0) \sim` Bernoulli(expit(:math:`-0.3 + 0.8 L(0) + 0.3 W_1`));
* visits at intensity :math:`0.07 \exp(0.4 L - 0.3 A)` per day; at a visit
  :math:`L(s) \sim` Bernoulli(expit(:math:`-0.4 + 1.2 L(s^-) - 0.8 A(s^-) + 0.4 W_1`)), then
  :math:`A(s) \sim` Bernoulli(expit(:math:`-1.5 + 1.0 L(s) + 2.0 A(s^-) + 0.2 W_2 / 6`));
* cause 1 at :math:`0.010 \exp(-0.6 A + 0.8 L + 0.3 W_1)`, cause 2 at
  :math:`0.006 \exp(0.2 A + 0.5 L - 0.3 W_1 + 0.1 W_2)` and censoring at
  :math:`0.008 \exp(0.6 L - 0.4 A)` per day.

Follow-up ends at day 30 (:data:`END`).  :func:`sample_ct` draws the counting-process frame the
engine reads, one row per interval between visits.

**Law CT-day** is the same law on whole days: each day's probability of an exit is
:math:`1 - e^{-\lambda}`, and the day's events are drawn in the order death (the cause in
proportion to its hazard), visit (:math:`L`, then :math:`A`), censoring, so the engine's tie
rule acts on every day (:func:`sample_day`).  A censoring on the day of a visit is the unit's
zero-length last row.  No visit or censoring is drawn on day 30, the end of follow-up.

**Regimes and truths.**  Always treat, never treat, and the rule "treat when :math:`L(s) = 1`"
(:data:`REGIMES`), at baseline and every visit; every visit is a decision visit.  The truth of
regime :math:`r` is :math:`P^{G_r}(T \le \tau, J = j)` at :math:`\tau \in \{10, 20, 30\}`
(:data:`TIMES`), the visit process left as it runs under the regime and censoring removed.
:func:`monte_carlo_risks` simulates it, and :data:`TRUTHS` holds the values from
:data:`TRUTH_UNITS` units per regime and law, drawn in chunks from the stream named
``truth`` (``stream_seed(STUDY, "truth", law, regime, chunk)``), with a Monte Carlo SE of
about :math:`1.4 \times 10^{-4}` at a risk of 0.25, against estimator SEs near 0.015.  The design
test recomputes them on a smaller draw.

**Feasibility** (:func:`feasibility`) reads only simulated event counts and true weights: the
expected events of each cause by day 10 among each regime's baseline followers at
:math:`n = 1{,}000` (at least 30), and the 99th percentile of each regime's true inverse
weight among the units that follow it to their exit (at most 20).  Under the plan's visit base
rate 0.10 and visit treatment intercept :math:`-1.0` the percentile was 41.5 for never and 25.3
for the rule, so the truth-only adjustment the plan allows before the declaration set them to
0.07 and :math:`-1.5`; every other value is the plan's.  The plug-in separation (review S8)
then held at the plan's coefficients of :math:`L`, which stand (``pilot-diagnosis.md``).

**The true-weight IPW witness** (:func:`ipw_contributions`): on a draw of law CT-day, each
unit's contribution :math:`1\{\text{followed } r\}\, 1\{T \le \tau, J = j\} / \{\prod \pi\,
S_c(T^-)\}` under the true treatment and censoring laws.  Its SD over root 1,000 is the SD of
the true-weight IPW estimator at :math:`n = 1{,}000`, which bounds the efficient SD from above,
since knowing :math:`G` leaves the efficiency bound unchanged (plan 5.7, review S8).

**Misspecified learners.**  The engine's hazard and censoring design is
:math:`[A(s_k), W_1, W_2, L(s_k), A(s_{k-1}), k, s_k]`, its treatment design the same without
:math:`A(s_k)`, and its continuation design the hazard design, the visit time and one indicator
per bin.  :class:`CoxWithout` and :class:`OmitColumn` drop :math:`L(s_k)` from these
(:data:`HAZARD_L`, :data:`TREATMENT_L`).  A hazard library of several members holds
:class:`~cleverly.survival.CoxHazard` models only (``select_hazard``), and ``cox._clone`` rebuilds a
``CoxHazard`` from its terms, so a wrong side is one member: the Cox main-terms model without
:math:`L`, the main-terms logistic model without :math:`L` for the treatment, and the main-terms
soft logistic model without :math:`L` for the continuation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin, clone

from cleverly.interventions import Rule
from cleverly.survival import CoxHazard

#: The end of follow-up, in days.
END = 30.0
#: The target times, in days.
TIMES: tuple[float, float, float] = (10.0, 20.0, 30.0)
#: The causes, as the engine names them.
CAUSES: dict[int, str] = {1: "cause1", 2: "cause2"}
#: The comparison regime of every risk difference.
REFERENCE = "never"
#: The column of :math:`L(s_k)` in the hazard, censoring and continuation designs.
HAZARD_L = 3
#: The column of :math:`L(s_k)` in the treatment design.
TREATMENT_L = 2
#: The units per regime and law behind :data:`TRUTHS`.
TRUTH_UNITS = 10_000_000
#: The units of one chunk of a truth simulation.
TRUTH_CHUNK = 1_000_000


@dataclass(frozen=True)
class Coefficients:
    """The design values of law CT (plan 5.7), one tuple per factor.

    Parameters
    ----------
    l0 : tuple of float
        :math:`L(0)`: intercept and :math:`W_1`.
    a0 : tuple of float
        :math:`A(0)`: intercept, :math:`L(0)` and :math:`W_1`.
    visit : tuple of float
        The visit intensity: base rate per day, :math:`L` and :math:`A`.
    l_visit : tuple of float
        :math:`L(s)` at a visit: intercept, :math:`L(s^-)`, :math:`A(s^-)` and :math:`W_1`.
    a_visit : tuple of float
        :math:`A(s)` at a visit: intercept, :math:`L(s)`, :math:`A(s^-)` and :math:`W_2 / 6`.
    cause1 : tuple of float
        Cause 1: base rate per day, :math:`A`, :math:`L` and :math:`W_1`.
    cause2 : tuple of float
        Cause 2: base rate per day, :math:`A`, :math:`L`, :math:`W_1` and :math:`W_2`.
    censoring : tuple of float
        Censoring: base rate per day, :math:`L` and :math:`A`.
    """

    l0: tuple[float, float] = (-0.5, 0.6)
    a0: tuple[float, float, float] = (-0.3, 0.8, 0.3)
    visit: tuple[float, float, float] = (0.07, 0.4, -0.3)
    l_visit: tuple[float, float, float, float] = (-0.4, 1.2, -0.8, 0.4)
    a_visit: tuple[float, float, float, float] = (-1.5, 1.0, 2.0, 0.2)
    cause1: tuple[float, float, float, float] = (0.010, -0.6, 0.8, 0.3)
    cause2: tuple[float, float, float, float, float] = (0.006, 0.2, 0.5, -0.3, 0.1)
    censoring: tuple[float, float, float] = (0.008, 0.6, -0.4)

    def rates(self, s: _State) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cause 1, cause 2, visit and censoring intensities per day at the current state."""
        b1, a1, l1, w1 = self.cause1
        b2, a2, l2, w12, w22 = self.cause2
        bv, lv, av = self.visit
        bc, lc, ac = self.censoring
        return (
            b1 * np.exp(a1 * s.a + l1 * s.l + w1 * s.w1),
            b2 * np.exp(a2 * s.a + l2 * s.l + w12 * s.w1 + w22 * s.w2),
            bv * np.exp(lv * s.l + av * s.a),
            bc * np.exp(lc * s.l + ac * s.a),
        )

    def p_l0(self, w1: np.ndarray) -> np.ndarray:
        return np.asarray(expit(self.l0[0] + self.l0[1] * w1))

    def p_a0(self, l: np.ndarray, w1: np.ndarray) -> np.ndarray:
        return np.asarray(expit(self.a0[0] + self.a0[1] * l + self.a0[2] * w1))

    def p_l(self, l: np.ndarray, a: np.ndarray, w1: np.ndarray) -> np.ndarray:
        c = self.l_visit
        return np.asarray(expit(c[0] + c[1] * l + c[2] * a + c[3] * w1))

    def p_a(self, l: np.ndarray, a: np.ndarray, w2: np.ndarray) -> np.ndarray:
        c = self.a_visit
        return np.asarray(expit(c[0] + c[1] * l + c[2] * a + c[3] * w2 / 6.0))


#: The declared design values.  The plug-in separation of plan 5.7 (review S8) reads them before
#: the declaration (``pilot.plugin_separation``); a separation below 0.5 SD raises :math:`L`'s
#: coefficients in the cause hazards and the :math:`L` transition, and the values are then fixed.
COEFFICIENTS = Coefficients()


@dataclass
class _State:
    """The current state of a batch of units."""

    w1: np.ndarray
    w2: np.ndarray
    l: np.ndarray
    a: np.ndarray

    def take(self, rows: np.ndarray) -> _State:
        return _State(self.w1[rows], self.w2[rows], self.l[rows], self.a[rows])


# ------------------------------------------------------------------------------ regimes


def _when_l(frame: Any) -> Any:
    return (frame["L"] == 1).astype(int)


#: The engine's form of the rule: treat when the current :math:`L` is 1.
RULE = Rule(_when_l, "treat when L", rule_kind="known")

#: ``regime -> engine form``: the regimes of ``day_cf`` in the engine's order, reference first.
REGIMES: dict[str, Any] = {"never": 0, "always": 1, "rule": RULE}
#: The static regimes, which every cell but ``day_cf`` fits.
STATIC: dict[str, Any] = {"never": 0, "always": 1}


def regime_level(regime: str, l: np.ndarray) -> np.ndarray:
    """The treatment regime ``regime`` assigns at a decision with covariate ``l``."""
    if regime == "always":
        return np.ones_like(l)
    if regime == "never":
        return np.zeros_like(l)
    if regime == "rule":
        return (l == 1).astype(l.dtype)
    raise KeyError(regime)


# ------------------------------------------------------------------------------ names


def level_name(regime: str, cause: int, t: float) -> str:
    """The engine's name of a regime's risk of one cause at one time."""
    return f"cif_regimen[{regime}, {CAUSES[cause]} @ t={t:g}]"


def difference_name(regime: str, cause: int, t: float) -> str:
    """The engine's name of a regime's risk difference against :data:`REFERENCE`."""
    return f"ate_regimen[{regime} vs {REFERENCE}, {CAUSES[cause]} @ t={t:g}]"


def targets(regimes: tuple[str, ...], times: tuple[float, ...] = TIMES) -> tuple[str, ...]:
    """Every risk of ``regimes`` and every difference against the reference, at ``times``."""
    levels = tuple(level_name(r, j, t) for r in regimes for j in CAUSES for t in times)
    differences = tuple(
        difference_name(r, j, t) for r in regimes if r != REFERENCE for j in CAUSES for t in times
    )
    return levels + differences


# ------------------------------------------------------------------------- the baseline


def _baseline(n: int, rng: np.random.Generator, c: Coefficients) -> _State:
    w1 = (rng.random(n) < 0.5).astype(float)
    w2 = rng.integers(1, 7, n).astype(float)
    l = (rng.random(n) < c.p_l0(w1)).astype(float)
    a = (rng.random(n) < c.p_a0(l, w1)).astype(float)
    return _State(w1, w2, l, a)


# ---------------------------------------------------------------------- observed draws


def _frame(columns: Mapping[str, list[np.ndarray]]) -> pd.DataFrame:
    frame = pd.DataFrame({name: np.concatenate(parts) for name, parts in columns.items()})
    frame["id"] = frame["id"].astype(int)
    frame["event"] = frame["event"].astype(int)
    frame = frame.sort_values(["id", "start", "stop"], kind="stable", ignore_index=True)
    return frame[["id", "start", "stop", "event", "W1", "W2", "L", "A"]]


def _record(
    columns: dict[str, list[np.ndarray]],
    ids: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    state: _State,
) -> None:
    for name, values in (
        ("id", ids),
        ("start", start),
        ("stop", stop),
        ("event", event),
        ("W1", state.w1),
        ("W2", state.w2),
        ("L", state.l),
        ("A", state.a),
    ):
        columns[name].append(np.asarray(values, dtype=float))


def sample_ct(
    n: int, seed: int | np.random.Generator, coefficients: Coefficients = COEFFICIENTS
) -> pd.DataFrame:
    """A draw of law CT: ``id, start, stop, event, W1, W2, L, A``, one row per interval."""
    rng = np.random.default_rng(seed)
    c = coefficients
    state = _baseline(n, rng, c)
    ids = np.arange(n)
    start = np.zeros(n)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in ("id", "start", "stop", "event")}
    columns.update({name: [] for name in ("W1", "W2", "L", "A")})
    while ids.size:
        r1, r2, rv, rc = c.rates(state)
        total = r1 + r2 + rv + rc
        stop = start + rng.exponential(1.0 / total)
        late = stop >= END
        if late.any():
            _record(
                columns,
                ids[late],
                start[late],
                np.full(late.sum(), END),
                0 * ids[late],
                state.take(late),
            )
        live = ~late
        ids, start, stop, state = ids[live], start[live], stop[live], state.take(live)
        r1, r2, rv, total = r1[live], r2[live], rv[live], total[live]
        u = rng.random(ids.size) * total
        cause = np.where(u < r1, 1, np.where(u < r1 + r2, 2, 0))
        visit = (cause == 0) & (u < r1 + r2 + rv)
        _record(columns, ids, start, stop, cause, state)
        keep = visit
        ids, start, state = ids[keep], stop[keep], state.take(keep)
        state.l = (rng.random(ids.size) < c.p_l(state.l, state.a, state.w1)).astype(float)
        state.a = (rng.random(ids.size) < c.p_a(state.l, state.a, state.w2)).astype(float)
    return _frame(columns)


def sample_day(
    n: int, seed: int | np.random.Generator, coefficients: Coefficients = COEFFICIENTS
) -> pd.DataFrame:
    """A draw of law CT-day, in the frame of :func:`sample_ct`, every time a whole day."""
    rng = np.random.default_rng(seed)
    c = coefficients
    state = _baseline(n, rng, c)
    ids = np.arange(n)
    start = np.zeros(n)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in ("id", "start", "stop", "event")}
    columns.update({name: [] for name in ("W1", "W2", "L", "A")})
    for day in range(1, int(END) + 1):
        r1, r2, _, _ = c.rates(state)
        dead = rng.random(ids.size) < 1.0 - np.exp(-(r1 + r2))
        cause = np.where(rng.random(ids.size) * (r1 + r2) < r1, 1, 2)
        if dead.any():
            _record(
                columns,
                ids[dead],
                start[dead],
                np.full(dead.sum(), float(day)),
                cause[dead],
                state.take(dead),
            )
        ids, start, state = ids[~dead], start[~dead], state.take(~dead)
        if day == int(END):
            _record(columns, ids, start, np.full(ids.size, END), 0 * ids, state)
            break
        _, _, rv, _ = c.rates(state)
        visit = rng.random(ids.size) < 1.0 - np.exp(-rv)
        if visit.any():
            _record(
                columns,
                ids[visit],
                start[visit],
                np.full(visit.sum(), float(day)),
                0 * ids[visit],
                state.take(visit),
            )
            start = np.where(visit, float(day), start)
            l_new = (rng.random(ids.size) < c.p_l(state.l, state.a, state.w1)).astype(float)
            l = np.where(visit, l_new, state.l)
            a_new = (rng.random(ids.size) < c.p_a(l, state.a, state.w2)).astype(float)
            state = _State(state.w1, state.w2, l, np.where(visit, a_new, state.a))
        _, _, _, rc = c.rates(state)
        censored = rng.random(ids.size) < 1.0 - np.exp(-rc)
        if censored.any():
            _record(
                columns,
                ids[censored],
                start[censored],
                np.full(censored.sum(), float(day)),
                0 * ids[censored],
                state.take(censored),
            )
        ids, start, state = ids[~censored], start[~censored], state.take(~censored)
    return _frame(columns)


#: ``law -> sampler``.
SAMPLERS: dict[str, Callable[..., pd.DataFrame]] = {"ct": sample_ct, "ct_day": sample_day}


# ------------------------------------------------------------------------ the truths


def _decide(
    regime: str,
    state: _State,
    visits: np.ndarray,
    rng: np.random.Generator,
    c: Coefficients,
    cap: int | None,
) -> np.ndarray:
    """The treatment after a visit: the regime's through visit ``cap``, the law's after it.

    ``visits`` counts each unit's visits after baseline, this one included.  With no cap no
    draw is made, so the full regime's stream, and :data:`TRUTH_RISKS`, are those of the
    simulator before caps existed.
    """
    level = regime_level(regime, state.l)
    if cap is None:
        return level
    natural = (rng.random(visits.size) < c.p_a(state.l, state.a, state.w2)).astype(float)
    return np.where(visits <= cap, level, natural)


def _regime_events_ct(
    n: int, rng: np.random.Generator, regime: str, c: Coefficients, cap: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Event time and cause per unit under ``regime`` in law CT, censoring removed.

    With ``cap`` the regime acts at baseline and the first ``cap`` visits, and treatment
    follows the law after them (:math:`\psi^{(K)}`, plan 3.5.3).
    """
    state = _baseline(n, rng, c)
    state.a = regime_level(regime, state.l)
    time = np.full(n, np.inf)
    cause = np.zeros(n, dtype=int)
    rows = np.arange(n)
    visits = np.zeros(n, dtype=int)
    t = np.zeros(n)
    while rows.size:
        r1, r2, rv, _ = c.rates(state)
        total = r1 + r2 + rv
        t = t + rng.exponential(1.0 / total)
        live = t < END
        rows, t, state, visits = rows[live], t[live], state.take(live), visits[live]
        r1, r2, total = r1[live], r2[live], total[live]
        u = rng.random(rows.size) * total
        dead = u < r1 + r2
        time[rows[dead]] = t[dead]
        cause[rows[dead]] = np.where(u[dead] < r1[dead], 1, 2)
        rows, t, state, visits = rows[~dead], t[~dead], state.take(~dead), visits[~dead] + 1
        state.l = (rng.random(rows.size) < c.p_l(state.l, state.a, state.w1)).astype(float)
        state.a = _decide(regime, state, visits, rng, c, cap)
    return time, cause


def _regime_events_day(
    n: int,
    rng: np.random.Generator,
    regime: str,
    c: Coefficients,
    cap: int | None = None,
    resets: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Event day and cause per unit under ``regime`` in law CT-day, censoring removed.

    With ``cap`` as in :func:`_regime_events_ct`.  With ``resets`` (and ``cap=0``) treatment
    follows the law at every visit and the regime is set at baseline and after each listed day's
    visit: the risk that a panel whose nodes start on those days identifies (:func:`panel_risks`).
    """
    state = _baseline(n, rng, c)
    state.a = regime_level(regime, state.l)
    time = np.full(n, np.inf)
    cause = np.zeros(n, dtype=int)
    rows = np.arange(n)
    visits = np.zeros(n, dtype=int)
    for day in range(1, int(END) + 1):
        r1, r2, rv, _ = c.rates(state)
        dead = rng.random(rows.size) < 1.0 - np.exp(-(r1 + r2))
        which = np.where(rng.random(rows.size) * (r1 + r2) < r1, 1, 2)
        time[rows[dead]] = day
        cause[rows[dead]] = which[dead]
        rows, state, rv, visits = rows[~dead], state.take(~dead), rv[~dead], visits[~dead]
        visit = rng.random(rows.size) < 1.0 - np.exp(-rv)
        visits = visits + visit
        l_new = (rng.random(rows.size) < c.p_l(state.l, state.a, state.w1)).astype(float)
        state.l = np.where(visit, l_new, state.l)
        state.a = np.where(visit, _decide(regime, state, visits, rng, c, cap), state.a)
        if day in resets:
            state.a = regime_level(regime, state.l)
    return time, cause


_REGIME_EVENTS = {"ct": _regime_events_ct, "ct_day": _regime_events_day}


def monte_carlo_risks(
    law: str,
    regime: str,
    seeds: Iterator[int] | list[int],
    *,
    chunk: int = TRUTH_CHUNK,
    coefficients: Coefficients = COEFFICIENTS,
    cap: int | None = None,
    resets: tuple[int, ...] = (),
) -> dict[tuple[int, float], tuple[float, int]]:
    r"""``(cause, time) -> (risk, units)`` under ``regime``: one chunk of ``chunk`` units per seed.

    ``cap`` gives :math:`\psi^{(K)}` and ``resets`` the panel's risk (CT-day only).
    """
    hits = {(j, t): 0 for j in CAUSES for t in TIMES}
    units = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if resets:
            time, cause = _regime_events_day(chunk, rng, regime, coefficients, 0, resets)
        else:
            time, cause = _REGIME_EVENTS[law](chunk, rng, regime, coefficients, cap)
        units += chunk
        for j in CAUSES:
            for t in TIMES:
                hits[(j, t)] += int(np.count_nonzero((cause == j) & (time <= t)))
    return {key: (count / units, units) for key, count in hits.items()}


def truth_from_risks(
    risks: Mapping[str, Mapping[tuple[int, float], float]],
) -> dict[str, float]:
    """The engine-named truths of ``regime -> (cause, time) -> risk``."""
    out = {
        level_name(r, j, t): float(v) for r, table in risks.items() for (j, t), v in table.items()
    }
    for regime in risks:
        if regime == REFERENCE:
            continue
        for j in CAUSES:
            for t in TIMES:
                out[difference_name(regime, j, t)] = (
                    out[level_name(regime, j, t)] - out[level_name(REFERENCE, j, t)]
                )
    return out


#: ``law -> regime -> (cause, time) -> risk``: :func:`monte_carlo_risks` on
#: :data:`TRUTH_UNITS` units per regime and law, from the ``truth`` stream
#: (``pilot.truth_seeds``), written by ``python -m tests.studies.continuous_time_longitudinal_pilot
#: --kinds truth`` and checked by the design test on a smaller draw.
TRUTH_RISKS: dict[str, dict[str, dict[tuple[int, float], float]]] = {
    "ct": {
        "never": {
            (1, 10.0): 0.1599671,
            (1, 20.0): 0.2801204,
            (1, 30.0): 0.37069,
            (2, 10.0): 0.0843557,
            (2, 20.0): 0.1481245,
            (2, 30.0): 0.1962434,
        },
        "always": {
            (1, 10.0): 0.0875624,
            (1, 20.0): 0.1545173,
            (1, 30.0): 0.207289,
            (2, 10.0): 0.1037302,
            (2, 20.0): 0.1848592,
            (2, 30.0): 0.2498012,
        },
        "rule": {
            (1, 10.0): 0.1149202,
            (1, 20.0): 0.2054889,
            (1, 30.0): 0.2768636,
            (2, 10.0): 0.0965213,
            (2, 20.0): 0.1720986,
            (2, 30.0): 0.231601,
        },
    },
    "ct_day": {
        "never": {
            (1, 10.0): 0.1597085,
            (1, 20.0): 0.2800201,
            (1, 30.0): 0.3706977,
            (2, 10.0): 0.0843523,
            (2, 20.0): 0.1481934,
            (2, 30.0): 0.1965621,
        },
        "always": {
            (1, 10.0): 0.0877375,
            (1, 20.0): 0.154738,
            (1, 30.0): 0.2076752,
            (2, 10.0): 0.1039969,
            (2, 20.0): 0.1852486,
            (2, 30.0): 0.2502683,
        },
        "rule": {
            (1, 10.0): 0.1150987,
            (1, 20.0): 0.2056362,
            (1, 30.0): 0.2769752,
            (2, 10.0): 0.0964539,
            (2, 20.0): 0.1721485,
            (2, 30.0): 0.2317292,
        },
    },
}

#: ``law -> engine name -> truth``.
TRUTHS: dict[str, dict[str, float]] = {
    law: truth_from_risks(table) for law, table in TRUTH_RISKS.items()
}

#: The caps :math:`K` whose capped risks :math:`\psi^{(K)}` are committed.  The engine's
#: :math:`K^*_r` at :math:`n = 1{,}000` lay in 3 to 5 in the design review's reimplementation of
#: ``visit_cap`` (``pilot-diagnosis.md``); a fit outside this range has no capped truth.
CAPS: tuple[int, ...] = tuple(range(1, 9))

#: ``law -> regime -> K -> (cause, time) -> risk``: :math:`\psi^{(K)}`, the regime at baseline
#: and at the first :math:`K` visits and the law's treatment after them, the target of a fit
#: whose cap is :math:`K` (plan 3.5.3).  :func:`monte_carlo_risks` with ``cap=K`` on
#: :data:`TRUTH_UNITS` units each, from the stream ``truth_capped`` (``pilot.capped_seeds``),
#: written by the pilot's kind ``truth_capped``.  A reading, never a gate (review S1).
#: The order of the six risks of a row of :data:`_CAPPED` and :data:`_PANEL`.
_ORDER: tuple[tuple[int, float], ...] = tuple((j, t) for j in CAUSES for t in TIMES)
#: The capped risks as rows ``(law, regime, K) -> risks`` in the order of :data:`_ORDER`.
_CAPPED: dict[tuple[str, str, int], tuple[float, ...]] = {
    ("ct", "never", 1): (0.1574101, 0.2687574, 0.3476422, 0.0849482, 0.1512711, 0.2034292),
    ("ct", "never", 2): (0.1594304, 0.2760978, 0.3599470, 0.0842793, 0.1489400, 0.1993639),
    ("ct", "never", 3): (0.1597438, 0.2787198, 0.3662092, 0.0842239, 0.1482712, 0.1975113),
    ("ct", "never", 4): (0.1598725, 0.2797122, 0.3690066, 0.0842909, 0.1480151, 0.1968018),
    ("ct", "never", 5): (0.1599378, 0.2800605, 0.3701028, 0.0843411, 0.1481825, 0.1965681),
    ("ct", "never", 6): (0.1601144, 0.2803872, 0.3706753, 0.0843059, 0.1479433, 0.1963554),
    ("ct", "never", 7): (0.1598520, 0.2802022, 0.3706052, 0.0842409, 0.1480321, 0.1964947),
    ("ct", "never", 8): (0.1598806, 0.2799771, 0.3704323, 0.0843014, 0.1480079, 0.1963036),
    ("ct", "always", 1): (0.0882898, 0.1589175, 0.2178363, 0.1034975, 0.1834039, 0.2465290),
    ("ct", "always", 2): (0.0877464, 0.1557731, 0.2111885, 0.1037446, 0.1845574, 0.2486923),
    ("ct", "always", 3): (0.0875574, 0.1549501, 0.2087511, 0.1037392, 0.1847162, 0.2493698),
    ("ct", "always", 4): (0.0874780, 0.1544841, 0.2076260, 0.1038931, 0.1849157, 0.2496326),
    ("ct", "always", 5): (0.0874862, 0.1545761, 0.2075462, 0.1037687, 0.1849069, 0.2496663),
    ("ct", "always", 6): (0.0874641, 0.1544222, 0.2073579, 0.1037342, 0.1847484, 0.2496200),
    ("ct", "always", 7): (0.0874885, 0.1544970, 0.2074375, 0.1038374, 0.1849924, 0.2497646),
    ("ct", "always", 8): (0.0876090, 0.1545293, 0.2073514, 0.1035770, 0.1848582, 0.2496094),
    ("ct", "rule", 1): (0.1153935, 0.2070525, 0.2789721, 0.0964212, 0.1716750, 0.2306532),
    ("ct", "rule", 2): (0.1151910, 0.2061568, 0.2779270, 0.0964495, 0.1720760, 0.2313527),
    ("ct", "rule", 3): (0.1151231, 0.2057594, 0.2774825, 0.0963718, 0.1720499, 0.2314051),
    ("ct", "rule", 4): (0.1151294, 0.2057333, 0.2770901, 0.0964730, 0.1722385, 0.2318128),
    ("ct", "rule", 5): (0.1150387, 0.2055929, 0.2769084, 0.0965051, 0.1722704, 0.2317160),
    ("ct", "rule", 6): (0.1151875, 0.2056552, 0.2768631, 0.0964095, 0.1720910, 0.2316178),
    ("ct", "rule", 7): (0.1151676, 0.2058492, 0.2770835, 0.0963461, 0.1719731, 0.2315441),
    ("ct", "rule", 8): (0.1150917, 0.2056285, 0.2769741, 0.0965573, 0.1722517, 0.2318505),
    ("ct_day", "never", 1): (0.1579946, 0.2703319, 0.3499859, 0.0848347, 0.1508539, 0.2028238),
    ("ct_day", "never", 2): (0.1596750, 0.2771614, 0.3618185, 0.0842604, 0.1488118, 0.1989324),
    ("ct_day", "never", 3): (0.1596232, 0.2792770, 0.3671174, 0.0844182, 0.1484742, 0.1974928),
    ("ct_day", "never", 4): (0.1596804, 0.2797649, 0.3694466, 0.0842021, 0.1481294, 0.1967587),
    ("ct_day", "never", 5): (0.1595941, 0.2798880, 0.3702224, 0.0844553, 0.1483190, 0.1966858),
    ("ct_day", "never", 6): (0.1597251, 0.2800238, 0.3705891, 0.0842440, 0.1481134, 0.1965296),
    ("ct_day", "never", 7): (0.1598725, 0.2802944, 0.3708796, 0.0842972, 0.1480813, 0.1963259),
    ("ct_day", "never", 8): (0.1597492, 0.2800869, 0.3707820, 0.0842737, 0.1481785, 0.1964534),
    ("ct_day", "always", 1): (0.0883693, 0.1589272, 0.2175145, 0.1036082, 0.1839016, 0.2471324),
    ("ct_day", "always", 2): (0.0878959, 0.1558395, 0.2110245, 0.1039508, 0.1847700, 0.2490603),
    ("ct_day", "always", 3): (0.0876908, 0.1549272, 0.2085799, 0.1039074, 0.1851632, 0.2498043),
    ("ct_day", "always", 4): (0.0878063, 0.1548443, 0.2079911, 0.1041156, 0.1852883, 0.2500497),
    ("ct_day", "always", 5): (0.0876483, 0.1546611, 0.2075214, 0.1038174, 0.1851416, 0.2499206),
    ("ct_day", "always", 6): (0.0878417, 0.1549982, 0.2079699, 0.1037852, 0.1850311, 0.2499160),
    ("ct_day", "always", 7): (0.0877435, 0.1548576, 0.2077260, 0.1040127, 0.1852493, 0.2502344),
    ("ct_day", "always", 8): (0.0876536, 0.1548582, 0.2078419, 0.1037765, 0.1848557, 0.2498193),
    ("ct_day", "rule", 1): (0.1154382, 0.2071608, 0.2794788, 0.0963121, 0.1716088, 0.2306854),
    ("ct_day", "rule", 2): (0.1150914, 0.2060840, 0.2780118, 0.0965347, 0.1721067, 0.2314162),
    ("ct_day", "rule", 3): (0.1148322, 0.2054976, 0.2771127, 0.0964516, 0.1722153, 0.2317509),
    ("ct_day", "rule", 4): (0.1151300, 0.2057156, 0.2771716, 0.0964216, 0.1723239, 0.2318213),
    ("ct_day", "rule", 5): (0.1151342, 0.2057827, 0.2771684, 0.0963959, 0.1721918, 0.2317810),
    ("ct_day", "rule", 6): (0.1149566, 0.2054390, 0.2768850, 0.0964668, 0.1722525, 0.2317254),
    ("ct_day", "rule", 7): (0.1149124, 0.2055263, 0.2766227, 0.0964200, 0.1720555, 0.2317070),
    ("ct_day", "rule", 8): (0.1150932, 0.2056540, 0.2771386, 0.0964485, 0.1720565, 0.2316180),
}
CAPPED_RISKS: dict[str, dict[str, dict[int, dict[tuple[int, float], float]]]] = {}
for (_law, _regime, _cap), _values in _CAPPED.items():
    CAPPED_RISKS.setdefault(_law, {}).setdefault(_regime, {})[_cap] = dict(
        zip(_ORDER, _values, strict=True)
    )

#: The days after whose visit the X70 panel sets the regime: every node start but day 0
#: (``NODE_ENDS`` of the record without day 30).
PANEL_RESETS: tuple[int, ...] = (3, 6, 10, 13, 16, 20, 23, 26)

#: ``regime -> (cause, time) -> risk`` for always and never: the risk the X70 panel identifies.
#: Inside a node a visit changes :math:`L` and, by the law, :math:`A`, which the panel sees only at
#: the next node start, where the panel's regime resets :math:`A`.  :func:`monte_carlo_risks` with
#: ``resets=PANEL_RESETS`` on :data:`TRUTH_UNITS` units each, from the stream ``truth_panel``,
#: written by the pilot's kind ``truth_panel`` (review S5).
#: The panel's risks as rows ``regime -> risks`` in the order of :data:`_ORDER`.
_PANEL: dict[str, tuple[float, ...]] = {
    "never": (0.1572731, 0.2758497, 0.3653426, 0.0850405, 0.1495908, 0.1985487),
    "always": (0.0890393, 0.1570779, 0.2108247, 0.1036115, 0.1846132, 0.2493196),
}
PANEL_RISKS: dict[str, dict[tuple[int, float], float]] = {
    regime: dict(zip(_ORDER, values, strict=True)) for regime, values in _PANEL.items()
}

#: Engine-named truths of the panel (static rows only).
PANEL_TRUTHS: dict[str, float] = truth_from_risks(PANEL_RISKS)


def capped_truth(law: str, name: str, cap: float, reference_cap: float = np.nan) -> float:
    r""":math:`\psi^{(K^*_r)}` of an engine-named target, or NaN for a cap outside :data:`CAPS`.

    A difference reads each regime's own cap: :math:`\psi^{(K^*_r)}_r - \psi^{(K^*_0)}_0`.
    """
    regime, cause, t = parse(name)
    table = CAPPED_RISKS.get(law, {})

    def risk(which: str, k: float) -> float:
        if not np.isfinite(k) or int(k) not in table.get(which, {}):
            return np.nan
        return float(table[which][int(k)][(cause, t)])

    value = risk(regime, cap)
    if name.startswith("ate_regimen"):
        value -= risk(REFERENCE, reference_cap)
    return value


# ---------------------------------------------------------- weights, feasibility, IPW


@dataclass
class DayPaths:
    """Per unit of an observed law CT-day draw: the exit and each regime's true weight.

    Parameters
    ----------
    time : ndarray
        The exit day.
    event : ndarray
        The exit: a cause, or 0 for censoring or the end of follow-up.
    baseline_follow : dict
        ``regime -> (n,)`` whether :math:`A(0)` is the regime's level at :math:`L(0)`.
    followed : dict
        ``regime -> (n,)`` whether the unit followed the regime at every decision before its
        exit.
    log_pi : dict
        ``regime -> (n,)`` the log of the product of the true probabilities of the observed
        treatment at the unit's decisions.
    log_sc : ndarray
        The log of the true censoring survival through the day before the exit.
    """

    time: np.ndarray
    event: np.ndarray
    baseline_follow: dict[str, np.ndarray]
    followed: dict[str, np.ndarray]
    log_pi: np.ndarray
    log_sc: np.ndarray

    def inverse_weight(self, regime: str) -> np.ndarray:
        """``1 / (prod pi * S_c(T-))`` per unit, ``0`` where the unit did not follow."""
        weight = np.exp(-(self.log_pi + self.log_sc))
        return np.where(self.followed[regime], weight, 0.0)


def day_paths(frame: pd.DataFrame, coefficients: Coefficients = COEFFICIENTS) -> DayPaths:
    r"""The true weights of the units of a law CT-day frame (:func:`sample_day`).

    A row starting at 0 carries the baseline decision, a later row the decision of its visit.
    The censoring draw of a day reads the state after that day's visit, so the censoring
    survival through the day before the exit is :math:`\exp(-\sum_r \lambda_c(L_r, A_r)\,
    d_r)`, with :math:`d_r` the row's whole days before its stop: ``stop - 1`` for the first row
    and ``stop - start`` for a later one.
    """
    c = coefficients
    unit = frame["id"].to_numpy()
    start = frame["start"].to_numpy(dtype=float)
    stop = frame["stop"].to_numpy(dtype=float)
    w1 = frame["W1"].to_numpy(dtype=float)
    w2 = frame["W2"].to_numpy(dtype=float)
    l = frame["L"].to_numpy(dtype=float)
    a = frame["A"].to_numpy(dtype=float)
    first = start == 0
    previous = np.r_[0.0, a[:-1]]
    p1 = np.where(first, c.p_a0(l, w1), c.p_a(l, previous, w2))
    log_p = np.log(np.where(a == 1, p1, 1.0 - p1))
    days = np.where(first, stop - 1.0, stop - start)
    _, _, _, rc = c.rates(_State(w1, w2, l, a))
    codes, index = np.unique(unit, return_inverse=True)
    n = codes.size
    last = np.r_[unit[1:] != unit[:-1], True]
    time = np.zeros(n)
    event = np.zeros(n, dtype=int)
    time[index[last]] = stop[last]
    event[index[last]] = frame["event"].to_numpy()[last]
    log_pi = np.bincount(index, weights=log_p, minlength=n)
    log_sc = -np.bincount(index, weights=rc * days, minlength=n)
    # The exit row's own days count only up to the day before the exit, which ``days`` gives.
    baseline_follow: dict[str, np.ndarray] = {}
    followed: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        match = (a == regime_level(regime, l)).astype(float)
        baseline_follow[regime] = np.zeros(n, dtype=bool)
        baseline_follow[regime][index[first]] = match[first] == 1
        followed[regime] = np.bincount(index, weights=1.0 - match, minlength=n) == 0
    return DayPaths(time, event, baseline_follow, followed, log_pi, log_sc)


def ipw_contributions(paths: DayPaths, regime: str, cause: int, t: float) -> np.ndarray:
    """Each unit's term of the true-weight IPW estimator of a risk (zero where censored)."""
    hit = (paths.event == cause) & (paths.time <= t)
    return np.where(hit, paths.inverse_weight(regime), 0.0)


def ipw_sd(paths: DayPaths, name: str, n: int = 1_000) -> float:
    """The SD of the true-weight IPW estimator of an engine-named target at ``n`` units."""
    regime, cause, t = parse(name)
    values = ipw_contributions(paths, regime, cause, t)
    if name.startswith("ate_regimen"):
        values = values - ipw_contributions(paths, REFERENCE, cause, t)
    return float(np.std(values, ddof=1) / np.sqrt(n))


def parse(name: str) -> tuple[str, int, float]:
    """``(regime, cause, time)`` of an engine name of :func:`level_name` or :func:`difference_name`."""
    inner = name[name.index("[") + 1 : -1]
    head, t = inner.split(" @ t=")
    regime, label = head.split(", ")
    code = {v: k for k, v in CAUSES.items()}[label]
    return regime.split(" vs ")[0], code, float(t)


def feasibility(paths: DayPaths, n: int = 1_000) -> pd.DataFrame:
    """Per regime: the truth-only feasibility readings of plan 5.7.

    ``events_cause<j>_day10`` is the expected count at ``n`` units of cause-``j`` events by
    day 10 among the regime's baseline followers (at least 30), and ``weight_p99`` the 99th
    percentile of the true inverse weight among the units that follow the regime to their exit
    (at most 20).  ``weight_p99_day30``, the same percentile among the followers still at risk
    on day 30, is a report: the plan's bound does not name the population (review m3).
    """
    records = []
    scale = n / paths.time.size
    for regime in REGIMES:
        base = paths.baseline_follow[regime]
        follow = paths.followed[regime]
        record: dict[str, Any] = {"regime": regime}
        for j in CAUSES:
            hit = base & (paths.event == j) & (paths.time <= TIMES[0])
            record[f"events_cause{j}_day10"] = float(scale * np.count_nonzero(hit))
        record["weight_p99"] = float(np.quantile(paths.inverse_weight(regime)[follow], 0.99))
        late = follow & (paths.time >= END)
        record["weight_p99_day30"] = float(np.quantile(paths.inverse_weight(regime)[late], 0.99))
        record["baseline_share"] = float(np.mean(base))
        record["followers_share"] = float(np.mean(follow))
        records.append(record)
    return pd.DataFrame.from_records(records)


#: The smallest expected event count of each cause by day 10 among a regime's baseline
#: followers, and the largest 99th percentile of a follower's true inverse weight (plan 5.7).
FEASIBLE_EVENTS = 30.0
FEASIBLE_WEIGHT = 20.0


# -------------------------------------------------------------------------- learners


class CoxWithout(BaseEstimator):
    """A :class:`~cleverly.survival.CoxHazard` fitted without one design column.

    Parameters
    ----------
    terms : str
        The Cox model's terms.
    column : int
        The design column dropped (:data:`HAZARD_L`).
    """

    def __init__(self, terms: str = "main", column: int = HAZARD_L) -> None:
        self.terms = terms
        self.column = column

    def __repr__(self) -> str:
        return f"CoxWithout({self.terms!r}, {self.column})"

    def _reduce(self, X: Any) -> np.ndarray:
        return np.delete(np.asarray(X, dtype=float), self.column, axis=1)

    def fit(self, X: Any, time: Any, event: Any, entry: Any = None) -> CoxWithout:
        self.model_ = CoxHazard(self.terms).fit(self._reduce(X), time, event, entry=entry)  # type: ignore[arg-type]
        return self

    def cumulative_hazard(self, X: Any, times: Any) -> np.ndarray:
        return self.model_.cumulative_hazard(self._reduce(X), times)


class OmitColumn(BaseEstimator):
    """``base`` fitted without one design column.

    Parameters
    ----------
    base : estimator
        A classifier or a regressor with ``fit(X, y, sample_weight)``.
    column : int
        The design column dropped.
    """

    def __init__(self, base: Any, column: int) -> None:
        self.base = base
        self.column = column

    def _reduce(self, X: Any) -> np.ndarray:
        return np.delete(np.asarray(X, dtype=float), self.column, axis=1)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OmitColumn:
        self.model_ = clone(self.base)
        if sample_weight is None:
            self.model_.fit(self._reduce(X), y)
        else:
            self.model_.fit(self._reduce(X), y, sample_weight=sample_weight)
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._reduce(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._reduce(X)))


class SoftLogistic(RegressorMixin, BaseEstimator):
    """A main-terms logistic model of a fraction in :math:`[0, 1]`, fitted on soft labels.

    The default continuation library's main-terms member: each row enters as an event with
    weight :math:`w y` and as a non-event with weight :math:`w (1 - y)`.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SoftLogistic:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        stacked = np.vstack([X, X])
        labels = np.r_[np.ones(len(y)), np.zeros(len(y))]
        weights = np.r_[w * y, w * (1.0 - y)]
        keep = weights > 0
        self.pipeline_ = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(C=np.inf, max_iter=2000))]
        )
        self.pipeline_.fit(stacked[keep], labels[keep], model__sample_weight=weights[keep])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.pipeline_.predict_proba(np.asarray(X, dtype=float))[:, 1])


def main_logistic() -> Any:
    """The default treatment library's main-terms member: scaled, unpenalized logistic."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(C=np.inf, max_iter=2000))]
    )


def wrong_outcome_side() -> dict[str, Any]:
    """The engine arguments of intensities and continuation that omit :math:`L`."""
    return {
        "hazard_library": (CoxWithout("main", HAZARD_L),),
        "visit_library": (CoxWithout("main", HAZARD_L),),
        "continuation_learner": OmitColumn(SoftLogistic(), HAZARD_L),
    }


def wrong_mechanism() -> dict[str, Any]:
    """The engine arguments of treatment and censoring models that omit :math:`L`."""
    return {
        "treatment_learner": OmitColumn(main_logistic(), TREATMENT_L),
        "censoring_library": (CoxWithout("main", HAZARD_L),),
    }


#: ``configuration -> engine arguments`` beside the defaults (plan 5.7's cells).
CONFIGURATIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "default": dict,
    "mechanism_correct": wrong_outcome_side,
    "outcome_correct": wrong_mechanism,
    "both_wrong": lambda: {**wrong_outcome_side(), **wrong_mechanism()},
}


#: The columns of :func:`sample_ct` and :func:`sample_day` for ``fit``.
COLUMNS: dict[str, Any] = {
    "id": "id",
    "start": "start",
    "stop": "stop",
    "event": "event",
    "treatment": "A",
    "baseline": ["W1", "W2"],
    "time_varying": ["L"],
}
