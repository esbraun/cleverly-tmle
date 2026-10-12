"""The package side of the R ``ctmle`` candidate-sequence witness.

``regenerate.py`` draws the fixture with :func:`draw_inputs` and screens each seed with the
functions here. ``tests/unit/test_ctmle_sequence_witness.py`` reruns them on the committed
inputs and compares them with R's outputs. One module serves both, so the screen and the
comparison read the same package path.

The comparison feeds R's own propensity arrays to the package's
:func:`~cleverly.estimators.ctmle._indexed_path` and its fluctuation, so it checks the
construction rule, the fluctuation and the final step, and not the solver. The solver has
its own rows (``g_tight.csv`` and the files beside it).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE, PropensityArrays
from cleverly.estimators import ctmle as ctmle_module
from cleverly.learners._l1_logistic import default_penalty_grid

N = 1_000
COVARIATES = ("W1", "W2", "W3", "W4", "W5", "W6")
FOLDS = 5
GRID_SIZE = 20
#: The grid spans this share of the unweighted ``lambda_max``, top and bottom. The bottom
#: stays above the end of R's default path, ``lambda_max * 1e-4``, so ``predict(s=)``
#: never clamps (plan 1.4, quirk 4).
GRID_TOP = 0.9
GRID_BOTTOM = 0.002
G_BOUNDS = (0.025, 0.975)
#: R ``ctmle``'s ``alpha``, which bounds ``Q`` into ``[0.005, 0.995]`` (plan finding 7).
ALPHA = 0.995
#: The interior that keeps both sides' ``Q`` bounds no-ops.
Q_INTERIOR = (0.005, 0.995)
#: The range the initial ``Q`` must lie in.
Q_RANGE = (0.02, 0.98)
#: The tolerance of the estimates and losses, which inherit R's ``glm`` fluctuation.
GLM_TOLERANCE = 1e-6
#: R's ``glm.control()$epsilon``, the relative deviance change at which a fluctuation stops.
GLM_EPSILON = 1e-8
#: A deciding score gap, relative to the round's best score, must exceed this. It is 100
#: times the relative error a converged ``glm`` can leave in a score.
MARGIN = 100.0 * GLM_EPSILON
#: An accepted ``ctmleGeneral`` column must lie at this squared distance from every earlier
#: column of its block, so R's ``< 1e-5`` match (plan 1.4, quirk 5) finds it alone.
SEPARATION = 1e-4


def draw_inputs(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """One fixture sample and its penalty grid.

    ``W1`` is an instrument, ``W2`` and ``W3`` are confounders, ``W4`` affects the outcome
    only, and ``W5`` and ``W6`` are noise. ``Q`` is main terms, and the outcome law has a
    ``W2 * W3`` term. ``A3`` and ``w`` serve the multinomial and the
    weighted solver rows only. ``Q0W`` and ``Q1W`` are the package's own initial fit, so the
    test refits them bit for bit.

    Parameters
    ----------
    seed : int
        The draw's seed.

    Returns
    -------
    tuple
        The inputs and the decreasing grid.
    """
    rng = np.random.default_rng(seed)
    w_matrix = rng.normal(size=(N, len(COVARIATES)))
    w1, w2, w3, w4 = (w_matrix[:, j] for j in range(4))
    # A strong instrument makes the least-penalized mechanisms poor fluctuation directions,
    # so the final step's argmin can fall inside the remaining penalties (W3).
    a_logit = 1.5 * w1 + 0.6 * w2 - 0.5 * w3
    a = (rng.uniform(size=N) < 1.0 / (1.0 + np.exp(-a_logit))).astype(int)
    # The ``W2 * W3`` term leaves the main-terms initial fit misspecified, so the
    # targeting steps have confounding to remove and the path accepts more than once.
    y_logit = 0.4 * a + 0.85 * (w2 + w2 * w3 + w3) + 0.3 * w4
    y = (rng.uniform(size=N) < 1.0 / (1.0 + np.exp(-y_logit))).astype(int)
    scores = np.column_stack([np.zeros(N), 0.8 * w1 + 0.4 * w2, -0.6 * w1 + 0.5 * w3])
    p3 = np.exp(scores - scores.max(axis=1, keepdims=True))
    p3 /= p3.sum(axis=1, keepdims=True)
    a3 = (rng.uniform(size=N)[:, None] > np.cumsum(p3, axis=1)).sum(axis=1)
    weights = rng.uniform(0.5, 2.0, size=N)
    fold = rng.permutation(np.arange(N) % FOLDS) + 1
    frame = pd.DataFrame({"Y": y, "A": a, "A3": a3, "w": weights, "fold": fold})
    for j, name in enumerate(COVARIATES):
        frame[name] = w_matrix[:, j]
    top = float(default_penalty_grid(w_matrix, a.astype(float))[0])
    grid = np.geomspace(GRID_TOP * top, GRID_BOTTOM * top, GRID_SIZE)
    initial = initial_outcome(frame)
    frame["Q0W"] = initial.arms[0.0]
    frame["Q1W"] = initial.arms[1.0]
    return frame, grid


def _estimator(frame: pd.DataFrame) -> CTMLE:
    placeholder = np.full(len(frame), float(frame["A"].mean()))
    return CTMLE(
        strategy="learners",
        treatment_candidates=[PropensityArrays(full=placeholder, cv=placeholder)],
        selection_split=frame["fold"].to_numpy() - 1,
        outcome_learner=LogisticRegression(C=1e6, max_iter=5000, tol=1e-12),
        treatment_learner=LogisticRegression(C=1e6, max_iter=5000, tol=1e-12),
        cross_fit=False,
        penalty=True,
        loss="loglik",
        alpha=ALPHA,
        g_bounds=G_BOUNDS,
        estimands=("ate",),
        simultaneous=False,
        random_state=0,
    )


@contextlib.contextmanager
def _captured() -> Iterator[list[Any]]:
    """Record the analysis-sample selector of the fits made inside the block."""
    held: list[Any] = []
    original = ctmle_module._Selector._indexed_candidates_path

    def spy(self: Any, rows: Any, train: Any, tag: str) -> Any:
        if train is None:
            held.append(self)
        return original(self, rows, train, tag)

    with mock.patch.object(ctmle_module._Selector, "_indexed_candidates_path", spy):
        yield held


def selector(frame: pd.DataFrame) -> Any:
    """The package's analysis-sample selector on the fixture, with its initial fit.

    Parameters
    ----------
    frame : pandas.DataFrame
        The fixture inputs.

    Returns
    -------
    Any
        The ``_Selector`` of the analysis sample.
    """
    data = frame[["Y", "A", *COVARIATES]]
    with _captured() as held:
        _estimator(frame).fit(data, outcome="Y", treatment="A", covariates=list(COVARIATES))
    if len(held) != 1:
        raise RuntimeError(f"expected one analysis-sample selector, captured {len(held)}")
    return held[0]


def initial_outcome(frame: pd.DataFrame) -> Any:
    """The package's initial outcome fit on the fixture rows."""
    return selector(frame).base.outcome


@dataclass(frozen=True)
class Path:
    """The package's indexed path on one candidate matrix.

    Parameters
    ----------
    candidates : list
        The ``_Candidate`` at each index.
    accepted : tuple of int
        The accepted indices, from 0.
    stopped : bool
        Whether the stop rule ended the construction.
    rounds : tuple of ndarray
        The scores of each round's tries.
    best : tuple of float
        The best score before each round.
    """

    candidates: list[Any]
    accepted: tuple[int, ...]
    stopped: bool
    rounds: tuple[np.ndarray, ...]
    best: tuple[float, ...]

    @property
    def estimates(self) -> np.ndarray:
        """The ATE of each index, on the outcome's scale."""
        return np.array([ate(candidate.targeted) for candidate in self.candidates])

    def likelihoods(self, y: np.ndarray) -> np.ndarray:
        """R's ``-sum(Y log Q* + (1 - Y) log(1 - Q*))`` at each index."""
        return np.array([negative_loglik(c.targeted.observed, y) for c in self.candidates])


def ate(fit: Any) -> float:
    return float(np.mean(fit.arms[1.0]) - np.mean(fit.arms[0.0]))


def negative_loglik(q: np.ndarray, y: np.ndarray) -> float:
    return float(-np.sum(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def _propensities(chooser: Any, columns: np.ndarray) -> list[Any]:
    return [
        chooser._array_propensity(k, PropensityArrays(full=column, cv=column))
        for k, column in enumerate(np.asarray(columns, dtype=float).T)
    ]


def indexed_path(chooser: Any, columns: np.ndarray) -> Path:
    """Run :func:`~cleverly.estimators.ctmle._indexed_path` on the columns of a matrix.

    Parameters
    ----------
    chooser : Any
        The analysis-sample ``_Selector``.
    columns : ndarray of float
        The ``(n, K)`` candidate propensities ``P(A = 1 | W)``, in index order.

    Returns
    -------
    Path
        The path, its accepted indices and the score of each try.
    """
    propensities = _propensities(chooser, columns)
    calls: list[tuple[int, int, float]] = []

    def fluctuate(index: int, initial: Any, n_steps: int) -> Any:
        candidate = chooser._fluctuate(
            (), propensities[index], initial, chooser.all_rows, n_steps, str(index)
        )
        calls.append((id(initial), index, float(candidate.risk)))
        return candidate

    candidates = ctmle_module._indexed_path(len(propensities), chooser.base.outcome, fluctuate)
    order: list[int] = []
    for key, _, _ in calls:
        if key not in order:
            order.append(key)
    starts = [min(index for key, index, _ in calls if key == each) for each in order]
    rounds = tuple(np.array([risk for key, _, risk in calls if key == each]) for each in order)
    accepted = [start - 1 for start in starts[1:]]
    best_before = [math.inf]
    for previous in rounds[:-1]:
        best_before.append(float(np.min(previous)))
    last = rounds[-1]
    choice = int(np.argmin(last))
    stopped = bool(last[choice] > best_before[-1])
    if not stopped:
        accepted.append(starts[-1] + choice)
    return Path(candidates, tuple(accepted), stopped, rounds, tuple(best_before))


def final_step(
    chooser: Any, path: Path, internal: np.ndarray, final_columns: np.ndarray, selected: int
) -> tuple[int, float]:
    """The package's C-TMLE1 final step on R's explicit-lambda columns.

    Parameters
    ----------
    chooser : Any
        The analysis-sample ``_Selector``.
    path : Path
        The package path on R's internal arrays.
    internal : ndarray of float
        R's internal arrays, the candidates of ``path``.
    final_columns : ndarray of float
        R's explicit-lambda columns for each penalty below the selected one.
    selected : int
        The selected index, from 0.

    Returns
    -------
    tuple of (int, float)
        The final index, from 0, and its ATE.
    """
    count = internal.shape[1]
    if final_columns.shape[1] != count - selected - 1:
        raise ValueError("one final column is needed for each penalty below the selected one")
    columns = np.column_stack([internal[:, : selected + 1], final_columns])
    propensities = _propensities(chooser, columns)
    chooser._indexed = (propensities, [()] * count, [str(k) for k in range(count)])
    outcome = chooser.final_step(path.candidates, selected)
    if outcome is None:
        raise ValueError("the selected index is the last; R's type-1 final step errors there")
    index, candidate = outcome
    return index, ate(candidate.targeted)


def separated(columns: np.ndarray, accepted: Sequence[int]) -> float:
    """The smallest squared distance from an accepted column to an earlier one of its block."""
    smallest = math.inf
    start = 0
    for index in accepted:
        for earlier in range(start, index):
            distance = float(np.sum((columns[:, index] - columns[:, earlier]) ** 2))
            smallest = min(smallest, distance)
        start = index + 1
    return smallest


def smallest_margin(path: Path) -> float:
    """The smallest deciding gap of any round, relative to the round's best score.

    A round's argmin decides against its second-best try. A round decides against the best
    score so far when it accepts or stops.
    """
    smallest = math.inf
    for scores, best in zip(path.rounds, path.best, strict=True):
        ordered = np.sort(scores)
        scale = max(1.0, abs(float(ordered[0])))
        if ordered.size > 1:
            smallest = min(smallest, float(ordered[1] - ordered[0]) / scale)
        if math.isfinite(best):
            smallest = min(smallest, abs(float(ordered[0]) - best) / scale)
    return smallest


def interior(path: Path) -> bool:
    """Whether every targeted ``Q`` lies strictly inside :data:`Q_INTERIOR`."""
    low, high = Q_INTERIOR
    for candidate in path.candidates:
        fit = candidate.targeted
        for values in (fit.observed, *fit.arms.values()):
            if not (np.all(values > low) and np.all(values < high)):
                return False
    return True
