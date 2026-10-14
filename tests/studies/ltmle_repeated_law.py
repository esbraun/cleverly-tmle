r"""The observed-data law of :func:`cleverly.datasets.make_longitudinal_repeated`, enumerated.

The generator states each truth by composing the conditionals under the intervention
(:func:`cleverly.datasets.repeated_truth`).  This module states the same law a second way:
it enumerates every path of the twelve observed nodes

.. code-block:: text

    W  ->  A1  ->  C1  ->  Y1  ->  L2  ->  A2  ->  C2  ->  Y2  ->  L3  ->  A3  ->  C3  ->  Y3

with its probability, censoring included, and reads every functional off the *observed*
law by iterated conditional means.  ``W`` takes three values.  ``Y_t`` reads ``Y_{t-1}``
and ``Y_{t-2}``, and ``L_t`` reads ``L_{t-1}``, so the history has depth two.  The two
computations share only the coefficient tables, and
``tests/unit/test_ltmle_repeated_method_study.py`` asserts that they agree.

Five functionals come off the same enumeration.

* :meth:`RepeatedOutcomeLaw.g_formula` with the full history ``[W, Y1, L2, Y2, L3]`` is the
  truth, because treatment and censoring are missing at random given that history.
* ``absorbing=True`` replaces ``Y_h`` with :math:`\max_{s \le h} Y_s`.  This is the limit of
  a fit that reads the outcome as an absorbing event, the N1 control.
* ``history="free"`` drops every ``Y_s`` from every history.  This is the limit of a fit
  whose nuisances are correct for the coarser history, the N2 control.  Treatment and
  censoring read ``Y_{t-1}``, so this limit differs from the truth.
* ``history="markov"`` keeps ``[W, Y_{t-1}, L_t]`` and drops the second-order terms.  The
  g-formula of that history is the coarse functional.  The N3 control reaches it through
  :meth:`RepeatedOutcomeLaw.tmle_limit`.
* ``history="baseline"`` keeps ``W`` alone.  This is the outcome regression of the
  double-robustness family's ``mechanism_correct``, ``both_wrong`` and ``targeting_necessity``
  arms.

:meth:`RepeatedOutcomeLaw.tmle_limit` is the population limit of a single-fold LTMLE whose
outcome regressions are cell means on a coarse history and whose mechanisms are intercepts or
the truth.  A static plan has a constant clever covariate under intercept mechanisms, so the
fluctuation is inert and the limit equals the g-formula of the history.  The rule's clever
covariate reads ``Y_{t-1}``, which a coarse regression does not, so the fluctuation moves
the estimate.  :func:`tmle_limit_functional` is that limit written over any ``(frame,
weights)`` pair, so one function evaluates the law and a sample.

:meth:`RepeatedOutcomeLaw.eif_sd` is the standard deviation of the efficient influence curve
at the truth.  It sizes the noise control and the design computations.

:data:`BINARY` is the generator's law.  :data:`CONTINUOUS` gives each ``Y_t`` three values
``{0, 0.5, 1}`` from a cumulative logit on the same index, and every later node reads the
values ``Y_{t-1}`` and ``Y_{t-2}`` linearly, so the main-terms logistic mechanisms stay
correctly specified.  A fit reads it as ``gaussian``.

X32b adds two functionals to the same enumeration.

* The incremental functional of Kim, Kennedy and Naimi (2021, Theorem 3.1) at a tilt
  :math:`\delta`: :meth:`RepeatedOutcomeLaw.tilt_functional` reads it off the observed law by
  iterated conditional means, and :meth:`RepeatedOutcomeLaw.tilt_counterfactual` composes the
  tilted conditionals forward.  :meth:`RepeatedOutcomeLaw.ipsi_eif` is its influence curve in
  the telescoped form of ``ipsi.py`` (Theorem 4.1).  :meth:`RepeatedOutcomeLaw.ipsi_limit` is
  the population limit of the one-step estimator under coarse nuisances.
* :meth:`RepeatedOutcomeLaw.sdr_limit` is the population limit of ``SDR`` (Díaz et al. 2023,
  Equation (4)) under coarse nuisances, and :meth:`RepeatedOutcomeLaw.sdr_plug_in_limit` is the
  same sequential regression with no correction.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from cleverly.datasets.longitudinal import (
    _REPEATED_A,
    _REPEATED_A1,
    _REPEATED_C,
    _REPEATED_C1,
    _REPEATED_L,
    _REPEATED_TIMES,
    _REPEATED_W,
    _REPEATED_W_PROBABILITY,
    _repeated_logit,
    _repeated_outcome,
)

__all__ = [
    "BINARY",
    "COLUMNS",
    "CONTINUOUS",
    "PLANS",
    "RULE",
    "TIMES",
    "RepeatedOutcomeLaw",
    "empirical_tmle_limit",
    "tmle_limit_functional",
]

TIMES = _REPEATED_TIMES

#: The observed columns, in time order.
COLUMNS = ("W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2", "L3", "A3", "C3", "Y3")

#: Nodes before the first that every later node may read, all zero by construction.  A
#: second-order term at ``t = 1`` or ``t = 2`` reads these, and so contributes nothing.
_LATENT = ("L1", "Y0", "A0", "Y-1", "L0")

#: The rule's short name inside this module.  The study modules map it to the label a fit
#: reports, :data:`cleverly.datasets.REPEATED_RULE_LABEL`.
RULE = "rule"

#: The plans every functional here accepts.  ``reversed`` treats at node one and then
#: exactly when the earlier outcome was zero: the deliberate-mutation control of the rule.
PLANS = ("never", "always", RULE, "reversed")

#: The plans whose arms read ``Y_{t-1}``.
_READS_OUTCOME = frozenset({RULE, "reversed"})

#: The cumulative-logit offsets of :data:`CONTINUOUS`: ``P(Y >= 1) = expit(index + 0.4)`` and
#: ``P(Y >= 2) = expit(index - 1.0)``.
_CUTS = (0.4, -1.0)


def arm(plan: str, time: int, previous: Any) -> Any:
    """The arm ``plan`` assigns at ``time`` given the outcome at ``time - 1``.

    The rule reads ``Y_{t-1} > 0``, so on :data:`CONTINUOUS` it treats after any nonzero
    outcome.
    """
    previous = np.asarray(previous, dtype=float)
    if plan == "never":
        return np.zeros_like(previous)
    if plan == "always":
        return np.ones_like(previous)
    if time == 1:
        return np.ones_like(previous)
    if plan == RULE:
        return (previous > 0.0).astype(float)
    if plan == "reversed":
        return (previous <= 0.0).astype(float)
    raise ValueError(plan)


def _binary_outcome(w: Any, a: Any, l: Any, y: Any, y2: Any, l1: Any) -> np.ndarray:
    p = np.asarray(_repeated_outcome(w, a, l, y, y2, l1), dtype=float)
    return np.stack([1.0 - p, p], axis=-1)


def _ordinal_outcome(w: Any, a: Any, l: Any, y: Any, y2: Any, l1: Any) -> np.ndarray:
    p = _repeated_outcome(w, a, l, y, y2, l1)
    index = np.log(p / (1.0 - p))
    at_least_one = 1.0 / (1.0 + np.exp(-(index + _CUTS[0])))
    at_least_two = 1.0 / (1.0 + np.exp(-(index + _CUTS[1])))
    return np.stack([1.0 - at_least_one, at_least_one - at_least_two, at_least_two], axis=-1)


# ---------------------------------------------------------------------------- frame functionals
#
# These read a ``(frame, weights)`` pair, so the same code evaluates the enumerated law
# (``paths``, ``p``) and an observed sample (the rows, ``1 / n``).


def with_latent(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the zero nodes before the first (:data:`_LATENT`) added when absent."""
    missing = [name for name in _LATENT if name not in frame]
    if not missing:
        return frame
    return frame.assign(**dict.fromkeys(missing, 0.0))


def history_columns(time: int, kind: str = "full") -> list[str]:
    """The columns of ``H_t``: what a fit's outcome regression at ``time`` reads.

    ``full`` is ``[W, Y1, L2, ..., Y_{t-1}, L_t]``, the history a repeated outcome declares.
    ``markov`` is ``[W, Y_{t-1}, L_t]``, which equals ``full`` through ``t = 2``.  ``free``
    drops every ``Y_s``, and ``baseline`` keeps ``W`` alone.
    """
    if kind not in {"full", "markov", "free", "baseline"}:
        raise ValueError(kind)
    columns = ["W"]
    if kind == "baseline":
        return columns
    if kind == "markov":
        return columns if time == 1 else [*columns, f"Y{time - 1}", f"L{time}"]
    for node in range(2, time + 1):
        if kind == "full":
            columns.append(f"Y{node - 1}")
        columns.append(f"L{node}")
    return columns


def followed_mask(frame: pd.DataFrame, plan: str, time: int) -> np.ndarray:
    """Whether a row follows ``plan`` and stays observed through ``time``."""
    mask = np.ones(len(frame), dtype=bool)
    for node in range(1, time + 1):
        assigned = arm(plan, node, frame[f"Y{node - 1}"].to_numpy())
        mask &= (frame[f"A{node}"].to_numpy() == assigned) & (frame[f"C{node}"].to_numpy() == 1.0)
    return mask


def conditional_mean(
    frame: pd.DataFrame,
    weights: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    columns: Sequence[str],
) -> np.ndarray:
    """``E[values | columns, mask]`` under ``weights``, evaluated on every row."""
    masked = weights * mask
    keys = frame[list(columns)]
    grouped = pd.DataFrame({"num": masked * values, "den": masked}).groupby(
        [keys[name] for name in columns], sort=False
    )
    totals = grouped.transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (totals["num"] / totals["den"]).to_numpy()


def intercept_cumulative(
    frame: pd.DataFrame, weights: np.ndarray, plan: str, time: int
) -> np.ndarray:
    r"""The cumulative product of intercept-only mechanisms at the plan's arms, on every row.

    An intercept-only treatment or censoring model at node ``s`` is the prior of the rows
    still under observation, those with ``C_1 = ... = C_{s-1} = 1``.  The product runs over
    ``s <= time`` and evaluates the treatment prior at the plan's arm.
    """
    product = np.ones(len(frame))
    at_risk = np.ones(len(frame), dtype=bool)
    for node in range(1, time + 1):
        mass = float(np.sum(weights[at_risk]))
        p_a = float(np.sum(weights[at_risk] * frame[f"A{node}"].to_numpy()[at_risk]) / mass)
        p_c = float(np.sum(weights[at_risk] * frame[f"C{node}"].to_numpy()[at_risk]) / mass)
        assigned = arm(plan, node, frame[f"Y{node - 1}"].to_numpy())
        product = product * np.where(assigned == 1.0, p_a, 1.0 - p_a) * p_c
        at_risk = at_risk & (frame[f"C{node}"].to_numpy() == 1.0)
    return product


def tmle_limit_functional(
    frame: pd.DataFrame,
    weights: np.ndarray,
    plan: str,
    horizon: int,
    *,
    q_history: str,
    cumulative: Callable[[int], np.ndarray],
) -> float:
    r"""The one-step functional of a single-fold LTMLE whose regressions are cell means.

    Backward from ``h``: :math:`Q_t = E[Q^*_{t+1} \mid H_t, \text{followed through } t]`
    with ``H_t`` the columns ``q_history`` names.  The weight is
    :math:`w = p\,1\{\text{followed through } t\} / G_t`, and ``cumulative(t)`` supplies
    ``G_t``, the plan-evaluated product of the mechanisms through ``t``.  The intercept
    :math:`\epsilon_t` solves :math:`\sum w\,(Q^*_{t+1} - \text{expit}(\text{logit}\,Q_t +
    \epsilon_t)) = 0`, and :math:`Q^*_t = \text{expit}(\text{logit}\,Q_t + \epsilon_t)`.  The
    functional is :math:`\sum p\,Q^*_1`.

    This mirrors the intercept fluctuation of
    ``cleverly.longitudinal.sequential._fluctuate_node``, which puts the inverse cumulative
    probability in the loss weight, as ``ltmle::UpdateQ`` does.  The step is inert, with
    :math:`\epsilon_t = 0`, when the score at zero is below ``1e-14``.

    Parameters
    ----------
    frame : pandas.DataFrame
        The paths of a law, or the rows of a sample with ``nan`` filled by zero.
    weights : numpy.ndarray
        The probability of each path, or ``1 / n`` for a sample.
    plan : str
        One of :data:`PLANS`.
    horizon : int
        The time point ``h``.
    q_history : str
        The history the outcome regressions read: ``full``, ``markov``, ``free`` or
        ``baseline``.
    cumulative : callable
        ``t -> G_t`` on every row of ``frame``.

    Returns
    -------
    float
        The functional.
    """
    frame = with_latent(frame)
    carried = frame[f"Y{horizon}"].to_numpy().astype(float)
    for time in range(horizon, 0, -1):
        followed = followed_mask(frame, plan, time)
        regression = conditional_mean(
            frame, weights, carried, followed, history_columns(time, q_history)
        )
        weight = weights[followed] / cumulative(time)[followed]
        offset = logit(np.clip(regression, 1e-12, 1.0 - 1e-12))
        target = carried[followed]
        base = offset[followed]

        def score(
            shift: float,
            target: np.ndarray = target,
            base: np.ndarray = base,
            w: np.ndarray = weight,
        ) -> float:
            return float(np.sum(w * (target - expit(base + shift))))

        epsilon = 0.0 if abs(score(0.0)) < 1e-14 else float(brentq(score, -10, 10, xtol=1e-14))
        carried = expit(offset + epsilon)
    return float(np.sum(weights * carried))


def empirical_tmle_limit(rows: pd.DataFrame, plan: str, horizon: int, *, q_history: str) -> float:
    """:func:`tmle_limit_functional` on a sample's empirical measure, intercept mechanisms.

    The rows' censored nodes are filled with zero.  A censored row never follows a plan past
    its censoring node, so the fill reaches no result.  This is the exact value a single-fold
    fit with cell-mean regressions and intercept-only mechanisms returns on the sample.
    """
    filled = with_latent(rows.fillna(0.0).reset_index(drop=True))
    weights = np.full(len(filled), 1.0 / len(filled))
    return tmle_limit_functional(
        filled,
        weights,
        plan,
        horizon,
        q_history=q_history,
        cumulative=lambda time: intercept_cumulative(filled, weights, plan, time),
    )


# ---------------------------------------------------------------------------- X32b helpers


def ipsi_history_columns(time: int, kind: str = "full") -> list[str]:
    """The columns an ``IPSI`` nuisance at ``time`` reads: ``H_t`` with the earlier arms.

    ``full`` is ``[W, A_1, Y_1, L_2, ..., A_{t-1}, Y_{t-1}, L_t]``, Kim's history.  ``free``
    drops every ``Y_s``.  ``markov`` is ``[W, A_{t-1}, Y_{t-1}, L_t]``, ``baseline`` is ``W``
    alone and ``intercept`` reads nothing.
    """
    if kind == "intercept":
        return []
    if kind == "baseline":
        return ["W"]
    if kind == "markov":
        return ["W"] if time == 1 else ["W", f"A{time - 1}", f"Y{time - 1}", f"L{time}"]
    if kind not in {"full", "free"}:
        raise ValueError(kind)
    columns = ["W"]
    for node in range(1, time):
        columns.append(f"A{node}")
        if kind == "full":
            columns.append(f"Y{node}")
        columns.append(f"L{node + 1}")
    return columns


def _risk(frame: pd.DataFrame, time: int) -> np.ndarray:
    """Still observed before node ``time``: ``C_1 = ... = C_{time-1} = 1``."""
    mask = np.ones(len(frame), dtype=bool)
    for node in range(1, time):
        mask &= frame[f"C{node}"].to_numpy() == 1.0
    return mask


def _cell_mean(
    frame: pd.DataFrame,
    weights: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    columns: Sequence[str],
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """The cell mean of ``values`` among ``mask``, on every row, by the cells of ``columns``.

    No column gives the mean over the mask.  An unseen cell takes the mean over ``fallback``
    (default ``mask``), the fallback of ``CellMeans``.
    """
    filled = np.nan_to_num(values)
    if not columns:
        return np.full(len(frame), np.sum(weights * mask * filled) / np.sum(weights * mask))
    out = conditional_mean(frame, weights, filled, mask, columns)
    unseen = np.isnan(out)
    if unseen.any():
        base = mask if fallback is None else fallback
        out = np.where(unseen, np.sum(weights * base * filled) / np.sum(weights * base), out)
    return out


@dataclass(frozen=True)
class RepeatedOutcomeLaw:
    """One ``T = 3`` repeated-outcome law and its enumerated observed paths.

    Parameters
    ----------
    name : str
        A label for identifiers.
    y_values : tuple of float
        The support of each ``Y_t``, in level order.
    outcome_table : callable
        ``(w, a, l, y_prev, y_prev2, l_prev) -> probabilities`` over the levels of ``Y_t``.
    family : str
        The family a fit resolves for this law.
    """

    name: str
    y_values: tuple[float, ...]
    outcome_table: Callable[[Any, Any, Any, Any, Any, Any], np.ndarray]
    family: str

    # ------------------------------------------------------------ the conditionals

    def treatment(self, time: int, w: Any, l: Any, y: Any, a: Any, y2: Any = 0.0) -> Any:
        """``P(A_t = 1 | W, L_t, Y_{t-1}, A_{t-1}, Y_{t-2})``."""
        if time == 1:
            return _repeated_logit(_REPEATED_A1, w=w)
        return _repeated_logit(_REPEATED_A, w=w, l=l, y=y, a=a, y2=y2)

    def stay(self, time: int, w: Any, l: Any, y: Any, a: Any, y2: Any = 0.0) -> Any:
        """``P(C_t = 1 | W, L_t, Y_{t-1}, A_t, Y_{t-2})``; ``a`` is the current arm."""
        if time == 1:
            return _repeated_logit(_REPEATED_C1, w=w, a=a)
        return _repeated_logit(_REPEATED_C, l=l, y=y, a=a, y2=y2)

    def covariate(self, w: Any, a: Any, y: Any, l1: Any = 0.0) -> Any:
        """``P(L_t = 1 | W, A_{t-1}, Y_{t-1}, L_{t-1})`` for ``t >= 2``."""
        return _repeated_logit(_REPEATED_L, w=w, a=a, y=y, l1=l1)

    # ------------------------------------------------------------ the paths

    @cached_property
    def paths(self) -> pd.DataFrame:
        """Every value of the twelve nodes, with its probability under the law.

        A path continues past ``C_t = 0`` with the law's own conditionals, so the nodes a
        censored unit never shows still sum to one.  Every functional here conditions on the
        censoring nodes it needs, so those latent values never reach a result.
        """
        levels = range(len(self.y_values))
        grid = itertools.product(
            range(len(_REPEATED_W)),
            (0, 1),
            (0, 1),
            levels,
            (0, 1),
            (0, 1),
            (0, 1),
            levels,
            (0, 1),
            (0, 1),
            (0, 1),
            levels,
        )
        values = np.array(list(grid), dtype=float)
        frame = pd.DataFrame(values, columns=list(COLUMNS))
        support = np.asarray(self.y_values)
        for time in range(1, TIMES + 1):
            frame[f"Y{time}"] = support[frame[f"Y{time}"].to_numpy(dtype=int)]
        frame["W"] = np.asarray(_REPEATED_W)[frame["W"].to_numpy(dtype=int)]
        frame = with_latent(frame)
        probability = np.full(len(frame), _REPEATED_W_PROBABILITY)
        for time in range(1, TIMES + 1):
            w = frame["W"].to_numpy()
            y_prev = frame[f"Y{time - 1}"].to_numpy()
            y_prev2 = frame[f"Y{time - 2}"].to_numpy()
            a_prev = frame[f"A{time - 1}"].to_numpy()
            l_prev = frame[f"L{time - 1}"].to_numpy()
            l_t = frame[f"L{time}"].to_numpy()
            a_t = frame[f"A{time}"].to_numpy()
            c_t = frame[f"C{time}"].to_numpy()
            if time > 1:
                p_l = self.covariate(w, a_prev, y_prev, l_prev)
                probability = probability * np.where(l_t == 1.0, p_l, 1.0 - p_l)
            p_a = self.treatment(time, w, l_t, y_prev, a_prev, y_prev2)
            probability = probability * np.where(a_t == 1.0, p_a, 1.0 - p_a)
            p_c = self.stay(time, w, l_t, y_prev, a_t, y_prev2)
            probability = probability * np.where(c_t == 1.0, p_c, 1.0 - p_c)
            table = self.outcome_table(w, a_t, l_t, y_prev, y_prev2, l_prev)
            level = np.searchsorted(support, frame[f"Y{time}"].to_numpy())
            probability = probability * table[np.arange(len(frame)), level]
        frame["p"] = probability
        return frame

    # ------------------------------------------------------------ the functionals

    def history(self, time: int, kind: str = "full") -> list[str]:
        """The columns of ``H_t``; see :func:`history_columns`."""
        return history_columns(time, kind)

    def followed(self, plan: str, time: int) -> np.ndarray:
        """Whether a path follows ``plan`` and stays observed through ``time``."""
        return followed_mask(self.paths, plan, time)

    def _conditional(
        self, values: np.ndarray, mask: np.ndarray, columns: Sequence[str]
    ) -> np.ndarray:
        """``E[values | columns, mask]`` under the path probabilities, on every path."""
        return conditional_mean(self.paths, self.paths["p"].to_numpy(), values, mask, columns)

    def regressions(
        self,
        plan: str,
        horizon: int,
        *,
        history: str = "full",
        absorbing: bool = False,
    ) -> list[np.ndarray]:
        """``[Q_1, ..., Q_h, Q_{h+1}]`` on every path; ``Q_{h+1}`` is the outcome itself.

        ``Q_t = E[Q_{t+1} | H_t, plan followed and observed through t]``.  A static plan's
        arms are constants among the followers, so ``H_t`` carries no arm column.  The rule's
        arms are functions of ``Y_{t-1}``, which ``full``, ``markov`` and ``baseline``
        histories either carry or leave to the follower mask.  The ``free`` history drops
        ``Y_{t-1}`` and refuses a rule.
        """
        if history == "free" and plan in _READS_OUTCOME:
            raise ValueError("a history without Y_s cannot evaluate a rule that reads Y_s")
        frame = self.paths
        outcome = frame[f"Y{horizon}"].to_numpy()
        if absorbing:
            outcome = np.max(
                np.column_stack([frame[f"Y{s}"].to_numpy() for s in range(1, horizon + 1)]),
                axis=1,
            )
        sequence = [outcome]
        for time in range(horizon, 0, -1):
            sequence.insert(
                0,
                self._conditional(
                    sequence[0], self.followed(plan, time), self.history(time, history)
                ),
            )
        return sequence

    def g_formula(
        self,
        plan: str,
        horizon: int,
        *,
        history: str = "full",
        absorbing: bool = False,
    ) -> float:
        """The iterated-regression functional :math:`E[Q_1(W)]` on the observed law."""
        first = self.regressions(plan, horizon, history=history, absorbing=absorbing)[0]
        return float(np.sum(self.paths["p"].to_numpy() * first))

    def tmle_limit(
        self, plan: str, horizon: int, *, q_history: str, mechanisms: str = "intercept"
    ) -> float:
        """The population limit of a single-fold LTMLE; see :func:`tmle_limit_functional`.

        Parameters
        ----------
        plan : str
            One of :data:`PLANS`.
        horizon : int
            The time point ``h``.
        q_history : str
            The history the cell-mean outcome regressions read.
        mechanisms : str
            ``"intercept"`` for prior-only treatment and censoring models fitted on the
            rows still under observation, or ``"true"`` for the generating mechanisms.
        """
        frame = self.paths
        weights = frame["p"].to_numpy()
        if mechanisms == "intercept":

            def cumulative(time: int) -> np.ndarray:
                return intercept_cumulative(frame, weights, plan, time)

        elif mechanisms == "true":

            def cumulative(time: int) -> np.ndarray:
                return self.cumulative(plan, time)

        else:
            raise ValueError(mechanisms)
        return tmle_limit_functional(
            frame, weights, plan, horizon, q_history=q_history, cumulative=cumulative
        )

    def counterfactual(self, plan: str, horizon: int) -> float:
        """:math:`E[Y_h^d]` by forward composition under the intervention, censoring off.

        Independent of :meth:`g_formula`: it reads the conditionals, not the observed paths.
        """
        total = 0.0
        support = np.asarray(self.y_values)
        for w in _REPEATED_W:
            # ``state[(y, y2, a, l)]``: the law of ``(Y_{t-1}, Y_{t-2}, A_{t-1}, L_{t-1})``.
            state = {(0.0, 0.0, 0.0, 0.0): 1.0}
            mean = 0.0
            for time in range(1, horizon + 1):
                nxt: dict[tuple[float, float, float, float], float] = {}
                mean = 0.0
                for (y_prev, y_prev2, a_prev, l_prev), mass in state.items():
                    p_l = 0.0 if time == 1 else float(self.covariate(w, a_prev, y_prev, l_prev))
                    a_t = float(arm(plan, time, y_prev))
                    for l_value, p_value in ((0.0, 1.0 - p_l), (1.0, p_l)):
                        if p_value == 0.0:
                            continue
                        table = np.asarray(
                            self.outcome_table(w, a_t, l_value, y_prev, y_prev2, l_prev),
                            dtype=float,
                        )
                        mean += mass * p_value * float(table @ support)
                        for level, p_next in enumerate(table):
                            key = (float(support[level]), y_prev, a_t, l_value)
                            nxt[key] = nxt.get(key, 0.0) + mass * p_value * float(p_next)
                state = nxt
            total += _REPEATED_W_PROBABILITY * mean
        return total

    def contrast(self, functional: Callable[[str], float], plan: str) -> float:
        """``functional(plan) - functional("never")``."""
        return functional(plan) - functional("never")

    def cumulative(self, plan: str, time: int) -> np.ndarray:
        r"""``prod_{s <= t} g_s(d_s | H_s) * P(C_s = 1 | H_s, d_s)`` on every path.

        Evaluated at the plan's arms on each path's own history, which is the weight the
        efficient influence curve divides by on the followers.
        """
        frame = self.paths
        product = np.ones(len(frame))
        for node in range(1, time + 1):
            w = frame["W"].to_numpy()
            y_prev = frame[f"Y{node - 1}"].to_numpy()
            y_prev2 = frame[f"Y{node - 2}"].to_numpy()
            a_prev = frame[f"A{node - 1}"].to_numpy()
            l_t = frame[f"L{node}"].to_numpy()
            assigned = arm(plan, node, y_prev)
            p_a = np.asarray(self.treatment(node, w, l_t, y_prev, a_prev, y_prev2), dtype=float)
            product = product * np.where(assigned == 1.0, p_a, 1.0 - p_a)
            product = product * np.asarray(
                self.stay(node, w, l_t, y_prev, assigned, y_prev2), dtype=float
            )
        return product

    def eif(self, plan: str, horizon: int, *, reference: str | None = None) -> np.ndarray:
        r"""The efficient influence curve of ``E[Y_h^d]``, or of a contrast, on every path.

        .. math::

            D = \sum_{t=1}^{h} \frac{I(\text{followed through } t)}{G_t}
                \bigl(Q_{t+1} - Q_t\bigr) + Q_1 - \psi

        A path that left the plan contributes zero to every later term, so the ``nan`` a
        conditional mean takes off the followers never enters.
        """
        if reference is not None:
            return self.eif(plan, horizon) - self.eif(reference, horizon)
        sequence = self.regressions(plan, horizon)
        psi = float(np.sum(self.paths["p"].to_numpy() * sequence[0]))
        curve = sequence[0] - psi
        for time in range(1, horizon + 1):
            mask = self.followed(plan, time)
            weight = np.where(mask, 1.0 / self.cumulative(plan, time), 0.0)
            step = np.where(mask, sequence[time] - sequence[time - 1], 0.0)
            curve = curve + weight * step
        return curve

    def eif_sd(self, plan: str, horizon: int, *, reference: str | None = None) -> float:
        """``sqrt(E[D^2])`` at the truth: the efficiency bound's standard deviation."""
        curve = self.eif(plan, horizon, reference=reference)
        return float(np.sqrt(np.sum(self.paths["p"].to_numpy() * curve**2)))

    # ------------------------------------------------------------ the incremental functional

    def _ipsi_core(
        self,
        delta: float,
        horizon: int,
        *,
        m_kind: str,
        pi_kind: str,
        omega_one: bool = False,
        absorbing: bool = False,
        true_treatment: bool = False,
    ) -> tuple[float, np.ndarray, dict[int, np.ndarray]]:
        r"""The plug-in functional, the uncentered influence function and its mechanism terms.

        The treatment and censoring mechanisms are cell means on ``pi_kind`` histories, the
        regressions :math:`m_t` are cell means on ``m_kind`` histories, and both are read off
        the observed paths.  The plug-in is :math:`P \bar m_1`.  The one-step function is
        :math:`\varphi = \bar m_1 + \sum_t W_t (Z_{t+1} - m_t) + \sum_t W_{t-1}\,
        \delta (m_t(\cdot, 1) - m_t(\cdot, 0)) (A_t - \pi_t) / (\delta\pi_t + 1 - \pi_t)^2`
        (``ipsi.py``, Kim et al. 2021, Theorem 4.1 telescoped), with :math:`W_t` the product of
        the cancelled tilt factors and of :math:`1\{C_s = 1\} / \omega_s`.
        """
        frame, weights = self.paths, self.paths["p"].to_numpy()
        arms = {t: frame[f"A{t}"].to_numpy() for t in range(1, TIMES + 1)}
        stays = {t: frame[f"C{t}"].to_numpy() for t in range(1, TIMES + 1)}
        pi: dict[int, np.ndarray] = {}
        omega: dict[int, np.ndarray] = {}
        for t in range(1, horizon + 1):
            risk = _risk(frame, t)
            if true_treatment:
                pi[t] = np.asarray(
                    self.treatment(
                        t,
                        frame["W"].to_numpy(),
                        frame[f"L{t}"].to_numpy(),
                        frame[f"Y{t - 1}"].to_numpy(),
                        frame[f"A{t - 1}"].to_numpy(),
                        frame[f"Y{t - 2}"].to_numpy(),
                    ),
                    dtype=float,
                )
            else:
                pi[t] = _cell_mean(frame, weights, arms[t], risk, ipsi_history_columns(t, pi_kind))
            if omega_one:
                omega[t] = np.ones(len(frame))
                continue
            columns = ipsi_history_columns(t, pi_kind)
            omega[t] = np.where(
                arms[t] == 1,
                _cell_mean(frame, weights, stays[t], risk & (arms[t] == 1), columns),
                _cell_mean(frame, weights, stays[t], risk & (arms[t] == 0), columns),
            )
        y = frame[f"Y{horizon}"].to_numpy().astype(float)
        if absorbing:
            y = np.max(
                np.column_stack([frame[f"Y{s}"].to_numpy() for s in range(1, horizon + 1)]), axis=1
            )
        target = y
        m: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        mbar: dict[int, np.ndarray] = {}
        for t in range(horizon, 0, -1):
            observed = _risk(frame, t) & (stays[t] == 1)
            columns = ipsi_history_columns(t, m_kind)
            m1 = _cell_mean(frame, weights, target, observed & (arms[t] == 1), columns, observed)
            m0 = _cell_mean(frame, weights, target, observed & (arms[t] == 0), columns, observed)
            m[t] = (m0, m1)
            q_1 = delta * pi[t] / (delta * pi[t] + 1.0 - pi[t])
            mbar[t] = q_1 * m1 + (1.0 - q_1) * m0
            target = mbar[t]
        plug_in = float(np.sum(weights * mbar[1]))
        phi = mbar[1].copy()
        previous = np.ones(len(frame))
        mechanism: dict[int, np.ndarray] = {}
        for t in range(1, horizon + 1):
            risk = _risk(frame, t)
            previous = np.where(risk, previous, 0.0)
            tilt = (delta * arms[t] + 1.0 - arms[t]) / (delta * pi[t] + 1.0 - pi[t])
            keep = risk & (stays[t] == 1)
            current = np.where(keep, previous * tilt / np.where(keep, omega[t], 1.0), 0.0)
            z = mbar[t + 1] if t < horizon else y
            fitted = np.where(arms[t] == 1, m[t][1], m[t][0])
            phi = phi + current * np.where(current != 0, np.nan_to_num(z) - fitted, 0.0)
            derivative = (
                delta * (m[t][1] - m[t][0]) * (arms[t] - pi[t]) / (delta * pi[t] + 1.0 - pi[t]) ** 2
            )
            mechanism[t] = np.where(risk, previous * derivative, 0.0)
            phi = phi + mechanism[t]
            previous = current
        return plug_in, phi, mechanism

    def tilt_functional(
        self,
        delta: float,
        horizon: int,
        *,
        history: str = "full",
        absorbing: bool = False,
        mechanisms: str = "observed",
    ) -> float:
        r"""Kim et al. (2021, Theorem 3.1): :math:`E[Y_h^{Q(\delta)}]` read off the observed law.

        Iterated conditional means over the paths.  The regression at node :math:`t` is the
        mean of the carried value among the rows observed through :math:`t`, by
        :math:`(H_t, A_t)`.  The carried value at the next node is
        :math:`\bar m_t = \sum_a q_t(a \mid H_t)\, m_t(H_t, a)`.

        Parameters
        ----------
        delta : float
            The tilt :math:`\delta`.
        horizon : int
            The horizon.
        history : str
            ``"full"`` reads the whole history, which is the truth.  ``"free"`` drops every
            :math:`Y_s`: the limit of a fit whose nuisances are correct for that coarser
            history.
        absorbing : bool
            Replace :math:`Y_h` by :math:`\max_{s \le h} Y_s`: the limit of a fit that reads
            the outcome as absorbing.
        mechanisms : str
            ``"observed"`` reads :math:`\pi_t` off the paths.  ``"true"`` reads it from the
            generating treatment conditional, which equals the observed one under ``full``.
        """
        if mechanisms not in {"observed", "true"}:
            raise ValueError(mechanisms)
        plug_in, _, _ = self._ipsi_core(
            delta,
            horizon,
            m_kind=history,
            pi_kind=history,
            absorbing=absorbing,
            true_treatment=mechanisms == "true",
        )
        return plug_in

    def tilt_counterfactual(self, delta: float, horizon: int) -> float:
        r""":math:`E[Y_h^{Q(\delta)}]` by forward composition under the tilt, censoring off.

        Independent of :meth:`tilt_functional`: it draws each arm from
        :math:`q_t(\cdot \mid H_t)` over the units that stay observed, so it reads the
        conditionals and never an observed cell.
        """
        frame = self.paths
        kept = frame[(frame.C1 == 1) & (frame.C2 == 1) & (frame.C3 == 1)]
        probability = np.full(len(kept), _REPEATED_W_PROBABILITY)
        support = np.asarray(self.y_values)
        for t in range(1, TIMES + 1):
            w = kept["W"].to_numpy()
            y_prev, y_prev2 = kept[f"Y{t - 1}"].to_numpy(), kept[f"Y{t - 2}"].to_numpy()
            a_prev, l_prev = kept[f"A{t - 1}"].to_numpy(), kept[f"L{t - 1}"].to_numpy()
            l_t, a_t = kept[f"L{t}"].to_numpy(), kept[f"A{t}"].to_numpy()
            if t > 1:
                p_l = self.covariate(w, a_prev, y_prev, l_prev)
                probability = probability * np.where(l_t == 1.0, p_l, 1.0 - p_l)
            p_a = np.asarray(self.treatment(t, w, l_t, y_prev, a_prev, y_prev2), dtype=float)
            q_a = delta * p_a / (delta * p_a + 1.0 - p_a)
            probability = probability * np.where(a_t == 1.0, q_a, 1.0 - q_a)
            table = self.outcome_table(w, a_t, l_t, y_prev, y_prev2, l_prev)
            level = np.searchsorted(support, kept[f"Y{t}"].to_numpy())
            probability = probability * table[np.arange(len(kept)), level]
        if abs(float(probability.sum()) - 1.0) > 1e-12:  # pragma: no cover - guards the law
            raise AssertionError("the tilted paths do not sum to one")
        return float(np.sum(probability * kept[f"Y{horizon}"].to_numpy()))

    def ipsi_eif(self, delta: float, horizon: int, *, reference: float | None = None) -> np.ndarray:
        r"""The centered influence function of :math:`\psi_h(\delta)` on every path.

        The telescoped form of Kim et al. (2021, Theorem 4.1) that ``ipsi.py`` computes, at the
        true nuisances.  Its mean is zero.  With ``reference`` it is the curve of the contrast
        against the tilt ``reference``.
        """
        _, phi, _ = self._ipsi_core(delta, horizon, m_kind="full", pi_kind="full")
        curve = phi - float(np.sum(self.paths["p"].to_numpy() * phi))
        if reference is None:
            return curve
        return curve - self.ipsi_eif(reference, horizon)

    def ipsi_eif_sd(self, delta: float, horizon: int, *, reference: float | None = None) -> float:
        """The standard deviation of :meth:`ipsi_eif` at the truth."""
        curve = self.ipsi_eif(delta, horizon, reference=reference)
        return float(np.sqrt(np.sum(self.paths["p"].to_numpy() * curve**2)))

    def ipsi_mechanism_share(self, delta: float, horizon: int) -> list[float]:
        """The share of the curve variance that each node's mechanism term carries."""
        _, phi, mechanism = self._ipsi_core(delta, horizon, m_kind="full", pi_kind="full")
        weights = self.paths["p"].to_numpy()
        total = float(np.sum(weights * (phi - np.sum(weights * phi)) ** 2))
        return [float(np.sum(weights * mechanism[t] ** 2) / total) for t in sorted(mechanism)]

    def ipsi_limit(
        self,
        delta: float,
        horizon: int,
        *,
        m_kind: str = "full",
        pi_kind: str = "full",
        omega_one: bool = False,
    ) -> float:
        r"""The population limit of the one-step estimator, :math:`P \varphi`.

        The mechanisms converge to the cell means on ``pi_kind`` histories and the regressions
        to the cell means on ``m_kind`` histories, each fitted to the tilt that the limiting
        mechanism builds.  ``full`` for both gives the truth.  ``omega_one`` sets
        :math:`\omega := 1`: the dropout weights dropped.
        """
        _, phi, _ = self._ipsi_core(
            delta, horizon, m_kind=m_kind, pi_kind=pi_kind, omega_one=omega_one
        )
        return float(np.sum(self.paths["p"].to_numpy() * phi))

    def ipsi_plug_in_limit(self, delta: float, horizon: int, *, m_kind: str = "baseline") -> float:
        r"""The tilt plug-in :math:`P \bar m_1` with the true mechanisms and ``m_kind`` regressions.

        The limit of the plug-in whose correction terms are left out: the ``IPSI``
        ``one_step_necessity`` control.
        """
        plug_in, _, _ = self._ipsi_core(delta, horizon, m_kind=m_kind, pi_kind="full")
        return plug_in

    # ------------------------------------------------------------ the SDR limit

    def _sdr_columns(self, time: int, kind: str) -> list[str]:
        if kind == "outcomes":
            return ["W", *[f"Y{s}" for s in range(1, time)]]
        if kind == "y_only":
            return [f"Y{s}" for s in range(1, time)]
        return self.history(time, kind)

    def _sdr_mechanism(self, plan: str, time: int, kind: str) -> np.ndarray:
        """``g_t(d_t | H_t) c_t`` at the plan's arm on every path: the tables or intercepts."""
        frame, weights = self.paths, self.paths["p"].to_numpy()
        if kind == "true":
            earlier = self.cumulative(plan, time - 1) if time > 1 else np.ones(len(frame))
            return self.cumulative(plan, time) / earlier
        at_risk = _risk(frame, time)
        mass = weights[at_risk].sum()
        p_a = np.sum(weights[at_risk] * frame[f"A{time}"].to_numpy()[at_risk]) / mass
        p_c = np.sum(weights[at_risk] * frame[f"C{time}"].to_numpy()[at_risk]) / mass
        assigned = arm(plan, time, frame[f"Y{time - 1}"].to_numpy())
        return np.where(assigned == 1, p_a, 1.0 - p_a) * p_c

    def sdr_limit(self, plan: str, horizon: int, q: Sequence[str], g: Sequence[str]) -> float:
        r"""The population limit of ``SDR`` (Díaz et al. 2023, Equation (4)) at ``horizon``.

        Parameters
        ----------
        plan : str
            One of :data:`PLANS`.
        horizon : int
            The horizon.
        q : sequence of str
            The history of the regression at each node: ``full``, ``markov``, ``baseline`` or
            ``outcomes`` (``W`` and the earlier outcomes).
        g : sequence of str
            The mechanism at each node: ``true`` or ``intercept``.

        Notes
        -----
        The ratio at node :math:`t` is :math:`1\{\text{followed through } t\} / (g_t c_t)`.
        The regression at node :math:`t` is the mean of the transform among the followers by
        the columns of ``q[t - 1]``, and the transform adds every later weighted residual.
        """
        frame, weights = self.paths, self.paths["p"].to_numpy()
        ratio = {
            t: np.where(followed_mask(frame, plan, t), 1.0, 0.0)
            / self._sdr_mechanism(plan, t, g[t - 1])
            for t in range(1, horizon + 1)
        }
        y = frame[f"Y{horizon}"].to_numpy().astype(float)
        value = {horizon + 1: y}
        m: dict[int, np.ndarray] = {}
        target = y
        for t in range(horizon, 0, -1):
            followed = followed_mask(frame, plan, t)
            regression = conditional_mean(
                frame, weights, np.nan_to_num(target), followed, self._sdr_columns(t, q[t - 1])
            )
            m[t] = np.where(np.isnan(regression), 0.0, regression)
            value[t] = m[t]
            transform = m[t].copy()
            carried_weight = np.ones(len(frame))
            for s in range(t, horizon + 1):
                carried_weight = carried_weight * ratio[s]
                z = value[s + 1] if s < horizon else y
                transform = transform + np.where(
                    carried_weight != 0, carried_weight * (z - m[s]), 0.0
                )
            target = transform
        return float(np.sum(weights * target))

    def sdr_plug_in_limit(self, plan: str, horizon: int, key: str) -> float:
        """The sequential regression of ``SDR`` with no correction, in the limit.

        ``key`` names the history of every regression: ``baseline`` (``W``), ``outcomes``
        (``W`` and the earlier outcomes), ``y_only`` (the earlier outcomes alone, which is the
        pairing learner of the ``SDR`` primary) or any of :meth:`history`.  A regression with
        no column is the mean over the followers.
        """
        frame, weights = self.paths, self.paths["p"].to_numpy()
        carried = frame[f"Y{horizon}"].to_numpy().astype(float)
        for t in range(horizon, 0, -1):
            followed = followed_mask(frame, plan, t)
            columns = self._sdr_columns(t, key)
            if not columns:
                carried = np.full(
                    len(frame),
                    np.sum(weights * followed * np.nan_to_num(carried))
                    / np.sum(weights * followed),
                )
            else:
                regression = conditional_mean(
                    frame, weights, np.nan_to_num(carried), followed, columns
                )
                carried = np.where(np.isnan(regression), 0.0, regression)
        return float(np.sum(weights * carried))

    # ------------------------------------------------------------ sampling

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` observed rows, censored nodes set to ``nan``.

        :data:`BINARY` is sampled by :func:`cleverly.datasets.make_longitudinal_repeated`
        instead.  This sampler draws the property cells of :data:`CONTINUOUS`.
        """
        rng = np.random.default_rng(seed)
        frame = self.paths
        cells = rng.choice(len(frame), size=n, p=frame["p"].to_numpy())
        rows = frame.iloc[cells][list(COLUMNS)].reset_index(drop=True).copy()
        alive = np.ones(n, dtype=bool)
        for time in range(1, TIMES + 1):
            for name in ((f"L{time}",) if time > 1 else ()) + (f"A{time}", f"C{time}"):
                rows.loc[~alive, name] = np.nan
            alive = alive & (rows[f"C{time}"].to_numpy() == 1.0)
            rows.loc[~alive, f"Y{time}"] = np.nan
        return rows


BINARY = RepeatedOutcomeLaw(
    name="binary", y_values=(0.0, 1.0), outcome_table=_binary_outcome, family="binomial"
)

CONTINUOUS = RepeatedOutcomeLaw(
    name="continuous",
    y_values=(0.0, 0.5, 1.0),
    outcome_table=_ordinal_outcome,
    family="gaussian",
)
