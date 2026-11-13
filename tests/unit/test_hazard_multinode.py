"""Hazard targeting with a treatment decided at several nodes, checked on finite laws (X70 (a)).

Each law below is a finite support passed as weighted data: every support point once, with
its probability as its observation weight (plan section 4).  So the empirical law is the law,
and a fit whose mechanism is right and whose hazard is wrong must return the g-formula to the
tolerance of its score.  The truth is the path enumeration of the law's own tables in this
file and never calls the estimator.

The base law has a binary baseline covariate ``W``, a binary treatment ``A_k`` at each of
three nodes that depends on ``(W, A_{k-1})``, dropout on ``(W, A_k, A_{k-1})``, and an event
hazard on ``(W, A_k, A_{k-1})``.  The wrong hazard learner is saturated in ``(W, A_k)`` and
drops ``A_{k-1}``, so its plug-in misses and the walk has work to do; every witness checks
that the initial plug-in misses.  The regimens are "always", "never", a rule ``d(W)``, a
known policy whose probability of treating depends on the previous arm, and a one-hot policy
"stay on the previous arm", which the shipped plan resolution would collapse at the observed
arms.

Each mutation control breaks one component of the construction (plan 4.3) and must move the
estimate, so an exact witness cannot pass because the component vanishes at the truth.
"""

from __future__ import annotations

import itertools
import re
import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import ModifiedPolicy, Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.longitudinal import hazard as hazard_module
from tests.discrete_law_point_survival import CellMeans, CellProbabilities

#: The first one-step step and the relative-score tolerance of the exact witnesses.
STEP = 2e-2
TOL = 1e-11
#: How close an exact estimate must be to the truth.
EXACT = 1e-8
#: How far a mutation, or a wrong initial plug-in, must move the estimate.
MISSED = 1e-3


def _g1(w: int) -> float:
    """``P(A_1 = 1 | W)``."""
    return (0.3, 0.6)[w]


def _g(node: int, w: int, previous: int) -> float:
    """``P(A_k = 1 | W, A_{k-1})`` at a node after the first."""
    return ((0.25, 0.7), (0.4, 0.8))[w][previous] - 0.05 * (node - 2)


def _retained(node: int, w: int, a: int, previous: int) -> float:
    """``P(C_k = 1 | W, A_k, A_{k-1})``."""
    return 0.9 - 0.1 * ((w + a + 2 * previous + node) % 3)


def _hazard(node: int, w: int, a: int, previous: int) -> float:
    """``P(Y_k = 1 | at risk, W, A_k, A_{k-1})``."""
    return 0.1 + 0.08 * a + 0.12 * previous * (1 - a) + 0.05 * w + 0.03 * node


def _split(node: int, w: int, a: int, previous: int) -> tuple[float, float]:
    """The two cause-specific hazards at risk, ``(relapse, death)``."""
    total = _hazard(node, w, a, previous)
    share = 0.3 + 0.2 * ((w + a + previous + node) % 3)
    return (total * share, total * (1.0 - share))


@dataclass(frozen=True)
class PathLaw:
    """A binary treatment at each node, censoring and a survival outcome, as weighted data.

    Parameters
    ----------
    n_times : int
        The nodes.
    causes : int
        One cause, or two coded ``R`` and ``D`` in the frame.
    g1, g, retained, hazard : callable
        The tables of the law; each later-node table reads ``(node, w, a, previous)``.
    """

    n_times: int = 3
    causes: int = 1
    g1: Callable[[int], float] = _g1
    g: Callable[[int, int, int], float] = _g
    retained: Callable[[int, int, int, int], float] = _retained
    hazard: Callable[[int, int, int, int], float] = _hazard
    support: tuple[tuple[Any, ...], ...] = field(init=False)
    masses: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        points: list[tuple[tuple[Any, ...], float]] = []
        for w in (0, 1):
            self._walk((w,), 0.5, 1, 0, points)
        object.__setattr__(self, "support", tuple(point for point, _ in points))
        object.__setattr__(self, "masses", np.array([mass for _, mass in points]))
        assert abs(self.masses.sum() - 1.0) < 1e-12

    def hazards(self, node: int, w: int, a: int, previous: int) -> tuple[float, ...]:
        """The cause-specific hazards at node ``k`` given at risk."""
        if self.causes == 1:
            return (self.hazard(node, w, a, previous),)
        return _split(node, w, a, previous)

    def treat(self, node: int, w: int, previous: int) -> float:
        """``P(A_k = 1 | W, A_{k-1})``."""
        return self.g1(w) if node == 1 else self.g(node, w, previous)

    def _walk(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        previous: int,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        width = 1 + 3 * self.n_times
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w = prefix[0]
        p = self.treat(node, w, previous)
        for a, pa in ((1, p), (0, 1.0 - p)):
            if pa <= 0.0:
                continue
            c = self.retained(node, w, a, previous)
            here = mass * pa
            if c < 1.0:
                point = (*prefix, a, 0)
                out.append((point + (None,) * (width - len(point)), here * (1.0 - c)))
            lam = self.hazards(node, w, a, previous)
            for code, share in enumerate(lam, start=1):
                if share > 0.0:
                    point = (*prefix, a, 1, code)
                    out.append((point + (None,) * (width - len(point)), here * c * share))
            rest = here * c * (1.0 - sum(lam))
            if rest > 0.0:
                self._walk((*prefix, a, 1, 0), rest, node + 1, a, out)

    # ---------------------------------------------------------------- the sample

    def frame(self) -> pd.DataFrame:
        """The law as data: one row per support point, with its probability in ``wt``."""

        def column(i: int) -> np.ndarray:
            return np.array(
                [np.nan if point[i] is None else float(point[i]) for point in self.support]
            )

        out: dict[str, np.ndarray] = {"W": column(0)}
        for node in range(1, self.n_times + 1):
            base = 1 + 3 * (node - 1)
            out[f"A{node}"] = column(base)
            out[f"C{node}"] = column(base + 1)
            code = column(base + 2)
            if self.causes == 2:
                out[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
                out[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))
            else:
                out[f"Y{node}"] = code
        frame = pd.DataFrame(out)
        frame["wt"] = self.masses * len(self.support)
        return frame

    def columns(self) -> dict[str, Any]:
        """The ``fit`` keywords."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": [f"A{k}" for k in nodes],
            "baseline": ["W"],
            "censoring": [f"C{k}" for k in nodes],
            "weights": "wt",
        }
        if self.causes == 2:
            out["outcome"] = {
                "relapse": [f"R{k}" for k in nodes],
                "death": [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        return out

    # ---------------------------------------------------------------- the truth

    def truth(self, regimen: Callable[[int, int, int], float], horizon: int, cause: int = 1) -> float:
        """``E sum_u q(u | W) F_j(t | u, W)`` by enumerating the paths.

        ``regimen(node, w, previous)`` is the probability of treating at the node.
        """
        total = 0.0
        for w in (0, 1):
            for path in itertools.product((0, 1), repeat=horizon):
                mass = 0.5
                previous = 0
                for node, a in enumerate(path, start=1):
                    p = regimen(node, w, previous)
                    mass *= p if a == 1 else 1.0 - p
                    previous = a
                if mass == 0.0:
                    continue
                survival, incidence, previous = 1.0, 0.0, 0
                for node, a in enumerate(path, start=1):
                    lam = self.hazards(node, w, a, previous)
                    incidence += survival * lam[cause - 1]
                    survival *= 1.0 - sum(lam)
                    previous = a
                total += mass * incidence
        return total


# ------------------------------------------------------------------- the regimens


def _always(node: int, w: int, previous: int) -> float:
    return 1.0


def _never(node: int, w: int, previous: int) -> float:
    return 0.0


def _rule_truth(node: int, w: int, previous: int) -> float:
    """``d(W)``: treat when ``W = 1`` at the odd nodes, and when ``W = 0`` at the even ones."""
    return float((w + node) % 2 == 0)


def _policy_truth(node: int, w: int, previous: int) -> float:
    """The known policy: treat with probability ``0.3 + 0.4 A_{k-1} + 0.1 W``."""
    return 0.3 + 0.4 * previous + 0.1 * w


def _stay_truth(node: int, w: int, previous: int) -> float:
    """Treat at node 1, then stay on the previous arm."""
    return 1.0 if node == 1 else float(previous)


def _rule_node(node: int) -> Callable[[Any], Any]:
    def assign(h: Any) -> Any:
        return ((np.asarray(h["W"], dtype=float) + node) % 2 == 0).astype(int)

    return assign


def _policy_density(node: int, n_times: int) -> Stochastic:
    def density(h: Any) -> np.ndarray:
        previous = (
            np.zeros(len(h)) if node == 1 else np.asarray(h[f"A{node - 1}"], dtype=float)
        )
        p = 0.3 + 0.4 * previous + 0.1 * np.asarray(h["W"], dtype=float)
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, f"q{node}", density_kind="known")


def _stay_density(node: int) -> Stochastic:
    def density(h: Any) -> np.ndarray:
        previous = np.asarray(h[f"A{node - 1}"], dtype=float)
        return np.column_stack([1.0 - previous, previous])

    return Stochastic(density, f"stay{node}", density_kind="known")


def regimens(n_times: int = 3) -> dict[str, Any]:
    """The five regimens of the base law, by label."""
    return {
        "always": (1,) * n_times,
        "never": (0,) * n_times,
        "rule": DynamicRegimen(
            "rule", tuple(_rule_node(k) for k in range(1, n_times + 1)), rule_kind="known"
        ),
        "policy": DynamicRegimen(
            "policy", tuple(_policy_density(k, n_times) for k in range(1, n_times + 1))
        ),
        "stay": DynamicRegimen("stay", (1, *(_stay_density(k) for k in range(2, n_times + 1)))),
    }


TRUTHS: dict[str, Callable[[int, int, int], float]] = {
    "always": _always,
    "never": _never,
    "rule": _rule_truth,
    "policy": _policy_truth,
    "stay": _stay_truth,
}


# -------------------------------------------------------------------- the learners


class WrongHazard(CellMeans):
    """Saturated in ``(W, A_k)``: the first and the last design column, dropping ``A_{k-1}``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WrongHazard:
        super().fit(_ends(X), y, sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(_ends(X))


def _ends(X: Any) -> np.ndarray:
    matrix = np.asarray(X, dtype=float)
    return matrix[:, [0, -1]]


def ltmle(chosen: dict[str, Any], **settings: Any) -> LTMLE:
    """A hazard ``LTMLE`` with the correct mechanism and the wrong hazard learner."""
    options: dict[str, Any] = {
        "n_folds": 1,
        "survival_model": "hazard",
        "targeting": "one_step",
        "step_size": STEP,
        "tol": TOL,
        "g_bounds": (1e-6, 1.0),
        "outcome_learner": WrongHazard(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
    }
    options.update(settings)
    return LTMLE(chosen, **options)
