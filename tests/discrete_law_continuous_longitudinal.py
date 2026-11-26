r"""Laws with a time-varying treatment in continuous time that a weighted sample realises exactly.

The implementation never enters this module.  Each law (PL, PL2) is a finite tree of event
paths on a few times, with every conditional probability a multiple of one quarter (one in 64
for one variant's treatment cell).  Every path enters the fit once, weighted by its probability,
through the private row-weight seam of :class:`~cleverly.survival.ContinuousLongitudinalData`
(as law (M) of X70 is passed by weight), so the weighted empirical law *equals* the law.

The parameter is the g-formula on the full history, computed by the recursion over the tree
with every conditional probability a quotient of path masses.  Complex-step derivatives of it,
one per path, give the efficient influence function, with no clever covariate and no
mechanism product: the check of the engine's influence curve is independent of its formula.

Each factor reads the history only through the engine's window
:math:`x_k = (W, L(s_k), A(s_k), A(s_{k-1}), k, s_k)` (``A(s_{-1})`` reads zero), so saturated
learners on the window are the law's own conditionals, and at the truth the window g-formula
and the full-history g-formula agree.

=========  ====================================================================================
law        schedule
=========  ====================================================================================
PL         ``W, L(0), A(0)``; censoring at 0.5; deaths of two causes at 1; a random visit at 1
           (``L(1)``, then ``A(1)``); censoring at 1, after the visit; deaths at 2.  Targets 1, 2
PL2        ``W, L(0), A(0)``; deaths at 1; a visit possible at 1 (certain where the current
           ``L`` is 1, probability one half where it is 0); censoring at 1, after the visit;
           deaths at 2; a visit possible at 2, likewise; deaths at 3.  Targets 2, 3
=========  ====================================================================================

A unit alive and uncensored after the last death time is censored at ``end``.  Within one time
the order is death, then the visit (``L``, then ``A``), then censoring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

#: A window: ``(w, l, a, aprev, k, s)``.
Window = tuple[int, int, int, int, int, float]
#: A regime: the level at a decision visit from ``(t, k, w, l, aprev)``.
Regime = Callable[[float, int, int, int, int], int]

CAUSES = (1, 2)
_PAIRS = ((0.25, 0.25), (0.5, 0.25), (0.25, 0.5))


def static(level: int) -> Regime:
    """The static regime at ``level``."""
    return lambda t, k, w, l, aprev: level


def treat_when_l(t: float, k: int, w: int, l: int, aprev: int) -> int:
    """The rule of T-e15: treat when the current ``L`` is 1."""
    return int(l == 1)


@dataclass(frozen=True)
class Law:
    """A finite law of event paths.

    Parameters
    ----------
    schedule : tuple of (float, str)
        The event times in forward order, each with its kind: ``"D"`` (deaths), ``"V"`` (a
        possible visit) or ``"C"`` (censoring).
    end : float
        The censoring time of a path still followed after the schedule.
    targets : tuple of float
        The target times.
    p_w : float
        ``P(W = 1)``.
    p_l0 : callable
        ``w -> P(L(0) = 1)``.
    p_a : callable
        ``(t, w, l, aprev, k) -> P(A = 1)`` at a decision visit.
    p_d : callable
        ``(t, window) -> (P(cause 1), P(cause 2))`` given at risk.
    p_v : callable
        ``(t, window) -> P(visit)`` given no death at ``t``.
    p_l : callable
        ``(t, window) -> P(L(t) = 1)`` at a visit, from the window before it.
    p_c : callable
        ``(t, window) -> P(censored)``.
    decision : callable
        ``(t, window) -> bool``: whether a visit at ``t`` from that window (before the visit)
        may change the treatment.  Elsewhere the treatment is carried forward.
    """

    schedule: tuple[tuple[float, str], ...]
    end: float
    targets: tuple[float, ...]
    p_w: float
    p_l0: Callable[[int], float]
    p_a: Callable[[float, int, int, int, int], float]
    p_d: Callable[[float, Window], tuple[float, float]]
    p_v: Callable[[float, Window], float]
    p_l: Callable[[float, Window], float]
    p_c: Callable[[float, Window], float]
    decision: Callable[[float, Window], bool] = field(default=lambda t, x: True)

    # ------------------------------------------------------------------ the paths

    @cached_property
    def paths(self) -> list[tuple[tuple[Any, ...], float]]:
        """Every path with positive probability: its tokens and its probability."""
        out: list[tuple[tuple[Any, ...], float]] = []

        def branch(p: float, token: Any, rest: Callable[[], None]) -> None:
            if p > 0:
                stack.append((token, p))
                rest()
                stack.pop()

        stack: list[tuple[Any, float]] = []

        def emit() -> None:
            tokens = tuple(t for t, _ in stack)
            prob = float(np.prod([p for _, p in stack]))
            out.append((tokens, prob))

        def walk(step: int, x: Window) -> None:
            if step == len(self.schedule):
                emit()
                return
            t, kind = self.schedule[step]
            w, _, a, _, k, _ = x
            if kind == "D":
                p1, p2 = self.p_d(t, x)
                branch(p1, ("D", t, 1), emit)
                branch(p2, ("D", t, 2), emit)
                branch(1.0 - p1 - p2, ("D", t, 0), lambda: walk(step + 1, x))
            elif kind == "C":
                pc = self.p_c(t, x)
                branch(pc, ("C", t, 1), emit)
                branch(1.0 - pc, ("C", t, 0), lambda: walk(step + 1, x))
            else:
                pv = self.p_v(t, x)
                branch(1.0 - pv, ("V", t, 0), lambda: walk(step + 1, x))

                decides = bool(self.decision(t, x))

                def visit() -> None:
                    pl = self.p_l(t, x)
                    for value, pv_l in ((1, pl), (0, 1.0 - pl)):

                        def after(value: int = value) -> None:
                            if decides:
                                pa = self.p_a(t, w, value, a, k + 1)
                                for level, p_level in ((1, pa), (0, 1.0 - pa)):
                                    branch(
                                        p_level,
                                        ("A", t, level, k + 1, True),
                                        lambda level=level: walk(
                                            step + 1, (w, value, level, a, k + 1, t)
                                        ),
                                    )
                            else:
                                branch(
                                    1.0,
                                    ("A", t, a, k + 1, False),
                                    lambda: walk(step + 1, (w, value, a, a, k + 1, t)),
                                )

                        branch(pv_l, ("L", t, value), after)

                branch(pv, ("V", t, 1), visit)

        pw = self.p_w
        for w, p_w in ((1, pw), (0, 1.0 - pw)):
            stack.append((("W", w), p_w))
            for l0 in (1, 0):
                p_l0 = self.p_l0(w) if l0 else 1.0 - self.p_l0(w)
                if p_l0 <= 0:
                    continue
                stack.append((("L", 0.0, l0), p_l0))
                pa = self.p_a(0.0, w, l0, 0, 0)
                for a0, p_a0 in ((1, pa), (0, 1.0 - pa)):
                    if p_a0 <= 0:
                        continue
                    stack.append((("A", 0.0, a0, 0, True), p_a0))
                    walk(0, (w, l0, a0, 0, 0, 0.0))
                    stack.pop()
                stack.pop()
            stack.pop()
        total = sum(p for _, p in out)
        assert abs(total - 1.0) < 1e-12, total
        return out

    @property
    def probs(self) -> np.ndarray:
        """``(S,)`` the path probabilities."""
        return np.array([p for _, p in self.paths])

    # ------------------------------------------------------------------ the data

    def frame(self) -> pd.DataFrame:
        """One unit per path in the counting-process long format, with its weight.

        Columns ``id, start, stop, event, W, L, A, decision, weight``; ``weight`` is the path's
        probability, repeated on its rows.
        """
        records = []
        for unit, (tokens, prob) in enumerate(self.paths):
            w = tokens[0][1]
            visits: list[list[Any]] = [[0.0, tokens[1][2], tokens[2][2], True]]
            exit_time, code = self.end, 0
            pending_l: int | None = None
            for token in tokens[3:]:
                kind = token[0]
                if kind == "D" and token[2] > 0:
                    exit_time, code = token[1], token[2]
                elif kind == "C" and token[2] == 1:
                    exit_time, code = token[1], 0
                elif kind == "L":
                    pending_l = token[2]
                elif kind == "A":
                    visits.append([token[1], pending_l, token[2], token[4]])
            for m, (start, l, a, decision) in enumerate(visits):
                stop = visits[m + 1][0] if m + 1 < len(visits) else exit_time
                records.append(
                    {
                        "id": unit,
                        "start": start,
                        "stop": stop,
                        "event": code if m + 1 == len(visits) else 0,
                        "W": w,
                        "L": l,
                        "A": a,
                        "decision": decision,
                        "weight": prob,
                    }
                )
        return pd.DataFrame.from_records(records)

    def unit_weights(self) -> np.ndarray:
        """``(S,)`` each unit's weight, in unit order."""
        return self.probs

    # ------------------------------------------------------------------ the truth

    def _values(
        self,
        masses: np.ndarray,
        regime: Regime,
        targets: Sequence[tuple[int, float]],
        cap: int | None,
        event: Callable[[tuple[Any, ...]], bool] | None = None,
    ) -> np.ndarray:
        """``(B, P)`` the g-formula of each target under each row of path masses.

        ``event``, when given, replaces every target's outcome with the path's indicator.
        """
        paths = [tokens for tokens, _ in self.paths]

        def value(index: list[int], depth: int, state: tuple[int, int, int, int]) -> np.ndarray:
            # state: (w, l, aprev-of-regime, k)
            groups: dict[Any, list[int]] = {}
            leaf = []
            for i in index:
                tokens = paths[i]
                if depth == len(tokens):
                    leaf.append(i)
                else:
                    groups.setdefault(tokens[depth], []).append(i)
            out = np.zeros((masses.shape[0], len(targets)), dtype=masses.dtype)
            if leaf:
                # Every path through this node ends here: its outcome is its last token.
                last = paths[leaf[0]][-1]
                assert not groups
                if event is not None:
                    out[:, :] = float(event(paths[leaf[0]]))
                elif last[0] == "D" and last[2] > 0:
                    for p, (cause, tau) in enumerate(targets):
                        if last[2] == cause and last[1] <= tau:
                            out[:, p] = 1.0
                return out
            kind = next(iter(groups))[0]
            w, l, aprev, k = state
            if kind == "A":
                visit = next(iter(groups))[3]
                if not next(iter(groups))[4]:
                    # A treatment carried forward at a visit that is not a decision visit.
                    ((token, rows),) = groups.items()
                    return value(rows, depth + 1, (w, l, token[2], visit))
                if cap is None or visit <= cap:
                    t = next(iter(groups))[1]
                    level = regime(t, visit, w, l, aprev)
                    token = next(tok for tok in groups if tok[2] == level)
                    return value(groups[token], depth + 1, (w, l, level, visit))
            if kind == "C":
                token = next(tok for tok in groups if tok[2] == 0)
                return value(groups[token], depth + 1, state)
            total = masses[:, index].sum(axis=1)
            for token, rows in groups.items():
                share = masses[:, rows].sum(axis=1) / total
                if token[0] == "W":
                    child = (token[1], l, aprev, k)
                elif token[0] == "L":
                    child = (w, token[2], aprev, k)
                elif token[0] == "A":
                    child = (w, l, token[2], token[3])
                else:
                    child = state
                out = out + share[:, None] * value(rows, depth + 1, child)
            return out

        return value(list(range(len(paths))), 0, (0, 0, 0, 0))

    def truth(
        self,
        regime: Regime,
        targets: Sequence[tuple[int, float]],
        cap: int | None = None,
    ) -> np.ndarray:
        """``(P,)`` each target's g-formula; ``cap`` leaves treatment natural after visit ``cap``.

        Parameters
        ----------
        regime : callable
            The regime.
        targets : sequence of (int, float)
            ``(cause, tau)`` pairs.
        cap : int or None
            The visit cap ``K*``: the regime holds at baseline and at visits ``1..cap``.

        Returns
        -------
        ndarray
            The parameter.
        """
        return np.real(self._values(self.probs[None, :].astype(float), regime, targets, cap)[0])

    def probability(
        self, regime: Regime, event: Callable[[tuple[Any, ...]], bool], cap: int | None = None
    ) -> float:
        """The g-formula of a path event: its probability under the regime, uncensored.

        Parameters
        ----------
        regime : callable
            The regime.
        event : callable
            Whether a path's tokens hold the event.
        cap : int or None
            As :meth:`truth`.

        Returns
        -------
        float
            The probability.
        """
        masses = self.probs[None, :].astype(float)
        return float(self._values(masses, regime, [(0, 0.0)], cap, event)[0, 0])

    def gateaux(
        self,
        regime: Regime,
        targets: Sequence[tuple[int, float]],
        cap: int | None = None,
    ) -> np.ndarray:
        """``(S, P)`` the influence function at each path, by complex step.

        Parameters
        ----------
        regime : callable
            The regime.
        targets : sequence of (int, float)
            ``(cause, tau)`` pairs.
        cap : int or None
            As :meth:`truth`.

        Returns
        -------
        ndarray
            The Gateaux derivative of the g-formula towards each path's point mass.
        """
        h = 1e-30
        base = self.probs
        S = base.size
        masses = np.tile(base.astype(complex), (S, 1)) * (1 - 1j * h)
        masses[np.arange(S), np.arange(S)] += 1j * h
        return np.imag(self._values(masses, regime, targets, cap)) / h


# ---------------------------------------------------------------------- the laws


def _pair(index: int) -> tuple[float, float]:
    return _PAIRS[index % 3]


def point_law(
    *,
    certain_visits: bool = False,
    l_moves: bool = True,
    rare_cell: bool = False,
    non_decision: bool = False,
    censor_reads_a: bool = True,
) -> Law:
    """Law PL, with the variants the tests name.

    Parameters
    ----------
    certain_visits : bool
        Every unit alive at time 1 visits then (the visit term's control).
    l_moves : bool
        ``False``: ``L(1)`` enters no later factor (the covariate term's control).
    rare_cell : bool
        One baseline cell treats with probability 1/64 (the weight floor's witness).
    non_decision : bool
        A visit at time 1 from ``L(0) = 1`` cannot change the treatment.
    censor_reads_a : bool
        The censoring at time 1 reads the current treatment.

    Returns
    -------
    Law
        The law.
    """

    def p_a(t: float, w: int, l: int, aprev: int, k: int) -> float:
        if k == 0:
            if rare_cell and w == 1 and l == 1:
                return 1 / 64
            return (1 + l + w) / 4
        lv = l if l_moves else 0
        return (1 + lv + aprev) / 4

    def p_d(t: float, x: Window) -> tuple[float, float]:
        w, l, a, aprev, k, _ = x
        lv = l if (l_moves or k == 0) else 0
        return _pair(w + 2 * lv + a + aprev + int(t))

    def p_v(t: float, x: Window) -> float:
        w, l, *_ = x
        return 1.0 if certain_visits else (1 + w + l) / 4

    def p_l(t: float, x: Window) -> float:
        _, l, a, *_ = x
        return (1 + l + a) / 4

    def p_c(t: float, x: Window) -> float:
        w, _, a, *_ = x
        if t < 1:
            return (1 + w) / 4
        return (1 + (a if censor_reads_a else w)) / 4

    def decision(t: float, x: Window) -> bool:
        # The window before the visit: a visit from L(0) = 1 carries the treatment.
        return not non_decision or x[1] == 0

    return Law(
        schedule=((0.5, "C"), (1.0, "D"), (1.0, "V"), (1.0, "C"), (2.0, "D")),
        end=3.0,
        targets=(1.0, 2.0),
        p_w=0.5,
        p_l0=lambda w: (1 + 2 * w) / 4,
        p_a=p_a,
        p_d=p_d,
        p_v=p_v,
        p_l=p_l,
        p_c=p_c,
        decision=decision,
    )


def point_law_2(*, censor_reads_a: bool = True) -> Law:
    """Law PL2: visits possible at times 1 and 2, deaths at 1, 2 and 3.

    Parameters
    ----------
    censor_reads_a : bool
        The censoring at time 1 reads the current treatment.

    Returns
    -------
    Law
        The law.
    """

    def p_a(t: float, w: int, l: int, aprev: int, k: int) -> float:
        if k == 0:
            return (1 + l + w) / 4
        return (1 + l + aprev) / 4

    def p_d(t: float, x: Window) -> tuple[float, float]:
        w, l, a, aprev, k, _ = x
        return _pair(w + 2 * l + a + aprev + k + int(t))

    def p_v(t: float, x: Window) -> float:
        return 1.0 if x[1] == 1 else 0.5

    def p_l(t: float, x: Window) -> float:
        _, l, a, *_ = x
        return (1 + l + a) / 4

    def p_c(t: float, x: Window) -> float:
        w, _, a, *_ = x
        return (1 + (a if censor_reads_a else w)) / 4

    return Law(
        schedule=(
            (1.0, "D"),
            (1.0, "V"),
            (1.0, "C"),
            (2.0, "D"),
            (2.0, "V"),
            (3.0, "D"),
        ),
        end=4.0,
        targets=(2.0, 3.0),
        p_w=0.5,
        p_l0=lambda w: (1 + 2 * w) / 4,
        p_a=p_a,
        p_d=p_d,
        p_v=p_v,
        p_l=p_l,
        p_c=p_c,
    )


def with_targets(law: Law, targets: tuple[float, ...]) -> Law:
    """``law`` with other target times."""
    return replace(law, targets=targets)
