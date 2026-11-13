"""A modified treatment policy on the hazard path, checked exactly on finite laws (X70 part (c)).

Each law below is a finite support whose conditional probabilities are multiples of one
quarter, laid out in exact proportions, so the empirical law equals the law and a fit with the
correct mechanism and a wrong hazard learner must return the g-formula exactly.  The truth is
computed from the law's tables in this file and never calls the estimator.

The dose ``A`` takes the values ``0, 1, 2``.  A fit reads it as three levels (a categorical
node), as a dose on the unit grid (a continuous node, with the binned density reading the
law's own bins through :func:`tests.longitudinal_mtp.exact_bins`), or beside a binary flag
``F`` as a vector with a dose (a mixed node).  The policy "up" moves the dose one step up and
keeps the top, ``d(a) = min(a + 1, 2)``.

The wrong hazard learner is an unpenalised logistic regression additive in its design: the
law's hazard has interactions, so its plug-in misses the truth and the walk has work to do.
Every witness checks that the initial plug-in misses, so an exact estimate is not an accident
of a correct initial fit.  Each mutation control breaks one component the witness reads and
must move the estimate.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.interventions import JointPolicy, ModifiedPolicy, Piece, Randomizer, Shift
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from tests.discrete_law_point_survival import CellMeans, CellProbabilities
from tests.longitudinal_mtp import exact_bins

pytestmark = pytest.mark.usefixtures("exact_grid_law")

INF = np.inf
#: The first one-step step and the relative-score tolerance of the shipped exact witnesses.
STEP = 2e-2
TOL = 1e-10
EXACT = 1e-8
#: How far a mutation, or a wrong initial plug-in, must move the estimate.
MISSED = 1e-3

#: ``P(A = a | W, F)``, indexed ``[(w + f) % 2, a]``.
DOSE = np.array([[0.25, 0.25, 0.50], [0.50, 0.25, 0.25]])
#: The one-cause hazard by node, indexed ``[node - 1][w][a]``; ``F = 1`` adds a quarter.
HAZARD = (
    ((0.25, 0.50, 0.50), (0.50, 0.75, 0.25)),
    ((0.50, 0.25, 0.75), (0.25, 0.50, 0.25)),
    ((0.25, 0.25, 0.50), (0.50, 0.75, 0.25)),
)
#: The two-cause split ``(P(cause 1), P(cause 2))`` at risk, by ``(w + 2a + f + node) % 3``.
PAIRS = ((0.25, 0.25), (0.50, 0.25), (0.25, 0.50))


@dataclass(frozen=True)
class GridLaw:
    """A held baseline dose ``A``, an optional flag ``F``, and a survival outcome.

    Parameters
    ----------
    n_times : int
        The nodes.  A vector treatment is declared one node at a time, so a held vector is
        not a design the container reads: a law with ``flag`` has one node.
    flag : bool
        Whether the treatment is the vector ``(F, A)``.
    causes : int
        One cause, or two coded ``R`` and ``D`` in the frame.
    censored : bool
        Whether units drop out, with a retention that depends on ``(W, F, A)``.
    empty : Callable or None
        ``(node, w, f, a) -> bool``: the cells at which cause 2 never occurs.  Its mass
        moves to cause 1.
    """

    n_times: int = 2
    flag: bool = False
    causes: int = 1
    censored: bool = True
    empty: Callable[[int, int, int, int], bool] | None = None
    support: tuple[tuple[Any, ...], ...] = field(init=False)
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        assert not self.flag or self.n_times == 1
        points: list[tuple[tuple[Any, ...], float]] = []
        for w in (0, 1):
            for f in (0, 1) if self.flag else (0,):
                pf = (0.25 if f == w else 0.75) if self.flag else 1.0
                for a in range(3):
                    mass = 0.5 * pf * DOSE[(w + f) % 2, a]
                    self._walk((w, f, a), mass, 1, points)
        support = tuple(point for point, _ in points)
        masses = np.array([mass for _, mass in points])
        n = 2 * 4 ** (1 + int(self.flag) + 2 * self.n_times)
        counts = np.rint(masses * n)
        assert np.max(np.abs(masses * n - counts)) < 1e-9
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "counts", counts.astype(int))

    # ---------------------------------------------------------------- the tables

    def retained(self, node: int, w: int, f: int, a: int) -> float:
        """``P(C_k = 1 | W, F, A)``."""
        return 0.75 - 0.25 * ((w + f + a + node) % 2) if self.censored else 1.0

    def hazards(self, node: int, w: int, f: int, a: int) -> tuple[float, ...]:
        """The cause-specific hazards at node ``k`` given at risk."""
        if self.causes == 1:
            return (min(HAZARD[node - 1][w][a] + 0.25 * f, 0.75),)
        first, second = PAIRS[(w + 2 * a + f + node) % 3]
        if self.empty is not None and self.empty(node, w, f, a):
            return (first + second, 0.0)
        return (first, second)

    def _walk(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        width = 3 + 2 * self.n_times
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w, f, a = prefix[:3]
        c = self.retained(node, w, f, a)
        if c < 1.0:
            out.append(((*prefix, 0) + (None,) * (width - len(prefix) - 1), mass * (1.0 - c)))
        lam = self.hazards(node, w, f, a)
        for code, share in enumerate(lam, start=1):
            if share > 0.0:
                point = (*prefix, 1, code)
                out.append((point + (None,) * (width - len(point)), mass * c * share))
        self._walk((*prefix, 1, 0), mass * c * (1.0 - sum(lam)), node + 1, out)

    # ---------------------------------------------------------------- the sample

    def frame(self) -> pd.DataFrame:
        """The exact sample, one row per unit."""
        cells = np.repeat(np.arange(len(self.support)), self.counts)

        def column(i: int) -> np.ndarray:
            raw = [np.nan if point[i] is None else float(point[i]) for point in self.support]
            return np.array(raw)[cells]

        out: dict[str, np.ndarray] = {"W": column(0), "A": column(2)}
        if self.flag:
            out["F"] = column(1)
        for node in range(1, self.n_times + 1):
            out[f"C{node}"] = column(1 + 2 * node)
            code = column(2 + 2 * node)
            if self.causes == 2:
                out[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
                out[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))
            else:
                out[f"Y{node}"] = code
        return pd.DataFrame(out)

    def columns(self, reading: str) -> dict[str, Any]:
        """The ``fit`` keywords for a ``"categorical"``, ``"continuous"`` or ``"mixed"`` node."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "baseline": ["W"],
            "censoring": [f"C{k}" for k in nodes],
            "treatment": [["F", "A"]] if reading == "mixed" else "A",
        }
        if reading != "categorical":
            out["continuous_treatment"] = ["A"]
        if self.causes == 2:
            out["outcome"] = {
                "relapse": [f"R{k}" for k in nodes],
                "death": [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        return out

    # ---------------------------------------------------------------- the truth

    def incidence(self, w: int, f: int, a: int, horizon: int, cause: int = 1) -> float:
        """``F_j(t | a, f, w)`` from the hazard tables."""
        survival, total = 1.0, 0.0
        for node in range(1, horizon + 1):
            lam = self.hazards(node, w, f, a)
            total += survival * lam[cause - 1]
            survival *= 1.0 - sum(lam)
        return total

    def truth(
        self, branches: tuple[tuple[float, Callable[[int], int]], ...], horizon: int, cause: int = 1
    ) -> float:
        """``E sum_b p_b F_j(t | d_b(A), F, W)`` over the law of ``(W, F, A)``."""
        psi = 0.0
        for w in (0, 1):
            for f in (0, 1) if self.flag else (0,):
                pf = (0.25 if f == w else 0.75) if self.flag else 1.0
                for a in range(3):
                    mass = 0.5 * pf * DOSE[(w + f) % 2, a]
                    psi += mass * sum(
                        p * self.incidence(w, f, move(a), horizon, cause) for p, move in branches
                    )
        return psi


def up(a: int) -> int:
    return min(a + 1, 2)


def stay(a: int) -> int:
    return a


UP = ((1.0, up),)
RANDOM = ((0.25, up), (0.75, stay))
NATURAL = ((1.0, stay),)


def _up_apply(a: Any, h: Any) -> Any:
    return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)


def _pieces_up() -> tuple[Piece, ...]:
    return (
        Piece(-INF, 1.5, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
        Piece(1.5, INF),
    )


def policy(reading: str, kind: str) -> Any:
    """The policy ``kind`` ("up", "random" or "natural") on a node of ``reading``."""
    if reading == "mixed":
        if kind == "natural":
            return JointPolicy(kind, components={}, policy_kind="known")
        return JointPolicy(kind, components={"A": policy("continuous", kind)}, policy_kind="known")
    if kind == "natural":
        return ModifiedPolicy("natural", pieces=(Piece(-INF, INF),), policy_kind="known")
    if kind == "up":
        if reading == "categorical":
            return ModifiedPolicy("up", apply=_up_apply, policy_kind="known")
        return Shift(1.0, cap=2.0)
    return ModifiedPolicy(
        "random",
        pieces={"up": _pieces_up(), "stay": (Piece(-INF, INF),)},
        randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
        policy_kind="known",
    )


BRANCHES = {"up": UP, "random": RANDOM, "natural": NATURAL}


class Wrong(LogisticRegression):  # type: ignore[misc]
    """A wrong hazard learner: unpenalised logistic, additive in its design."""

    def __init__(self) -> None:
        super().__init__(C=np.inf, tol=1e-12, max_iter=10_000)


def fit(
    law: GridLaw,
    regimens: dict[str, Any],
    reading: str,
    *,
    learner: Any = None,
    targeting: str = "one_step",
    **settings: Any,
) -> Any:
    """A hazard fit of ``regimens`` on the exact sample of ``law``."""
    with warnings.catch_warnings(), exact_bins():
        warnings.simplefilter("ignore")
        return LTMLE(
            regimens,
            n_folds=settings.pop("n_folds", 1),
            survival_model=settings.pop("survival_model", "hazard"),
            targeting=targeting,
            step_size=STEP,
            tol=TOL,
            g_bounds=(1e-8, 1.0),
            outcome_learner=learner if learner is not None else Wrong(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
            **settings,
        ).fit(law.frame(), **law.columns(reading))


def risk(result: Any, label: str, horizon: int, cause: str | None = None) -> float:
    infix = "" if cause is None else f"{cause} "
    name = (
        f"risk_regimen[{label} @ t={horizon}]"
        if cause is None
        else f"risk_regimen[{label} @ {infix}t={horizon}]"
    )
    if name not in result.estimates:
        matches = [
            key
            for key in result.estimates
            if f"[{label} @" in key and f"t={horizon}]" in key and (cause is None or cause in key)
        ]
        (name,) = matches
    return float(result.estimates[name].psi)


def initial_level(result: Any, label: str, horizon: int) -> float:
    """The plug-in of the initial hazards, read from the walk's record."""
    record = result.fits[f"{label} @ t={horizon}"].hazard
    index = record.targets.index((label, None, horizon))
    return float(record.initial_levels[index])


@contextmanager
def captured(name: str) -> Iterator[list[Any]]:
    """Record every return value of ``hazard_module.<name>`` while the block runs."""
    original = getattr(hazard_module, name)
    calls: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        calls.append(out)
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hazard_module, name, spy)
        from cleverly.longitudinal import estimator as estimator_module

        if hasattr(estimator_module, name):
            patch.setattr(estimator_module, name, spy)
        yield calls
