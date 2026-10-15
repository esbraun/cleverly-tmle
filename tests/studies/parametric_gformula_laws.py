r"""The laws of the registered study ``parametric-gformula`` (X45), with exact truths.

The module imports nothing from ``cleverly``: every truth here is an independent statement of
the g-formula, so agreement with an estimate is evidence about the estimator.

A law is an ordered list of variables.  Each variable is a main-terms generalized linear model
in the *design prefix* the estimator uses: the baseline column, then for each node the
covariates in their declared order and the treatment (``cleverly.longitudinal.gformula.Layout``).
A covariate of node :math:`t` reads every earlier column of that table, so each covariate law
is exactly the main-terms ``GLM`` in the full history.  Treatment and censoring are main-terms
logistic in the history through the node's own covariates.  The outcome (or each node's event
hazard) reads the history through the node's treatment.

======  ======================================================================================
law     structure
======  ======================================================================================
``G1``  :math:`T = 3`.  Continuous :math:`W \sim N(0, 1)`, one binary :math:`L_t` and a binary
        :math:`A_t` per node, retention :math:`C_t`, and a binary end-of-study :math:`Y`
``G2``  :math:`T = 3`.  Continuous :math:`W`; at each node a Gaussian :math:`Z_t` and then a
        binary :math:`B_t` that reads :math:`Z_t` with a strong slope; binary :math:`A_t`; a
        Gaussian end-of-study :math:`Y`.  No censoring.  The mean of :math:`Y` given the
        history is linear, and the regimen mean is nonlinear in each Gaussian draw because a
        binary draw follows it
``G3``  :math:`T = 4`.  Continuous :math:`W`, a binary :math:`L_t`, a binary :math:`A_t`,
        retention :math:`C_t` and a baseline-category multinomial event at each node over
        {none, relapse, death}
``G4``  :math:`T = 3`.  Continuous :math:`W`, a Gaussian :math:`L_t` and a Gaussian dose
        :math:`A_t` per node, a Gaussian :math:`Y`.  Linear throughout, so every truth is a
        closed form
``G4Q`` ``G4`` with :math:`q A_3^2` in the mean of :math:`Y`: a main-terms fit extrapolates
======  ======================================================================================

**Truths.**  The binary laws sum over every branch, with Gauss-Hermite nodes for a Gaussian
variable (:func:`g_formula`).  The design test shows the nodes converge and that a sampler of
the law under the plan agrees within four Monte Carlo errors.  ``G4`` reads closed forms:
every variable is linear, so a mean passes through the linear recursion (:func:`linear_means`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

#: Gauss-Hermite nodes per Gaussian variable in the sums.  The design test pins the convergence.
GH_ORDER = 12


def _hermite(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite nodes and weights of :math:`N(0, 1)`, weights summing to 1."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    return np.asarray(nodes, dtype=float), np.asarray(weights / weights.sum(), dtype=float)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return np.asarray(values / values.sum(axis=1, keepdims=True), dtype=float)


# --------------------------------------------------------------------------------- variables


@dataclass(frozen=True)
class Var:
    """One variable of a law: a main-terms model in the earlier design columns.

    Parameters
    ----------
    name : str
        The column name.  A multinomial event is named by its node, and writes one column per
        cause (``columns``).
    kind : {"gauss", "bern", "cat"}
        A Gaussian variable (mean linear, residual SD ``sd``), a binary one (logistic), or a
        baseline-category multinomial one over the labels ``0..K``.
    role : {"baseline", "covariate", "treatment", "censor", "event", "outcome"}
        What the variable is.  ``censor`` is a retention indicator (1 retained).
    node : int
        The node, counted from one; 0 for the baseline.
    intercept : float or tuple of float
        The intercept, one per non-reference label for ``cat``.
    coef : mapping of str to float or tuple of float
        The coefficient on each earlier column named; an absent column has coefficient zero.
    sd : float
        The residual standard deviation of a Gaussian variable.
    columns : tuple of str
        The data columns a multinomial event writes, one per non-reference label.
    quad : mapping of str to float
        A squared term :math:`q x^2` of an earlier column.  Only a law the main-terms model
        gets wrong declares one (``G4Q``); the exact sums refuse it.
    """

    name: str
    kind: str
    role: str
    node: int
    intercept: Any
    coef: Mapping[str, Any] = field(default_factory=dict)
    sd: float = 1.0
    columns: tuple[str, ...] = ()
    quad: Mapping[str, float] = field(default_factory=dict)

    def eta(self, table: Mapping[str, np.ndarray], n: int) -> np.ndarray:
        """The linear predictor: ``(n,)``, or ``(n, K)`` with a zero first column for ``cat``."""
        if self.kind == "cat":
            k = len(self.intercept)
            out = np.zeros((n, k))
            for j in range(k):
                out[:, j] = self.intercept[j]
                for parent, value in self.coef.items():
                    out[:, j] += np.asarray(table[parent], dtype=float) * value[j]
            return np.column_stack([np.zeros(n), out])
        eta = np.full(n, float(self.intercept))
        for parent, value in self.coef.items():
            eta = eta + np.asarray(table[parent], dtype=float) * float(value)
        for parent, value in self.quad.items():
            eta = eta + float(value) * np.asarray(table[parent], dtype=float) ** 2
        return eta


# ------------------------------------------------------------------------------- plan nodes


class NodePlan:
    """A treatment node of a regimen, in the form the sums and the sampler read."""

    def branches(self, table: Mapping[str, np.ndarray], var: Var, n: int) -> tuple[Any, Any]:
        """``(values, probs)``, each ``(n, k)``: the assigned values and their probabilities."""
        raise NotImplementedError

    def draw(
        self, table: Mapping[str, np.ndarray], var: Var, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """One assigned value per row, for the Monte Carlo cross-check."""
        values, probs = self.branches(table, var, n)
        cumulative = np.cumsum(probs, axis=1)
        index = (rng.random(n)[:, None] >= cumulative).sum(axis=1)
        index = np.minimum(index, values.shape[1] - 1)
        return np.asarray(values[np.arange(n), index], dtype=float)


@dataclass(frozen=True)
class Fixed(NodePlan):
    """Assign the value ``a`` to everyone."""

    a: float

    def branches(self, table: Mapping[str, np.ndarray], var: Var, n: int) -> tuple[Any, Any]:
        return np.full((n, 1), float(self.a)), np.ones((n, 1))


@dataclass(frozen=True)
class Rule(NodePlan):
    """Assign ``rule(table)``, a binary value read from the simulated history."""

    rule: Callable[[Mapping[str, np.ndarray]], np.ndarray]

    def branches(self, table: Mapping[str, np.ndarray], var: Var, n: int) -> tuple[Any, Any]:
        value = np.asarray(self.rule(table), dtype=float)
        return value.reshape(n, 1), np.ones((n, 1))


@dataclass(frozen=True)
class Policy(NodePlan):
    """Draw the binary treatment with probability ``prob(table)`` (a known stochastic policy)."""

    prob: Callable[[Mapping[str, np.ndarray]], np.ndarray]

    def branches(self, table: Mapping[str, np.ndarray], var: Var, n: int) -> tuple[Any, Any]:
        p = np.asarray(self.prob(table), dtype=float)
        return np.tile([0.0, 1.0], (n, 1)), np.column_stack([1.0 - p, p])


@dataclass(frozen=True)
class Tilt(NodePlan):
    """Draw the binary treatment from the odds-tilted natural law: odds times ``delta``."""

    delta: float

    def branches(self, table: Mapping[str, np.ndarray], var: Var, n: int) -> tuple[Any, Any]:
        g = expit(var.eta(table, n))
        q = self.delta * g / (self.delta * g + 1.0 - g)
        return np.tile([0.0, 1.0], (n, 1)), np.column_stack([1.0 - q, q])


@dataclass(frozen=True)
class ShiftPlan(NodePlan):
    """Assign the natural dose of the law plus ``delta``: a modified treatment policy.

    The natural dose is a draw of the law's own model given the intervened history, so the plan
    is a sampler only.  The closed form of a linear law is :func:`linear_means`.
    """

    delta: float

    def branches(self, table: Mapping[str, np.ndarray], var: Var, n: int) -> tuple[Any, Any]:
        raise NotImplementedError("a shifted dose is continuous: the sums do not enumerate it")

    def draw(
        self, table: Mapping[str, np.ndarray], var: Var, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        natural = var.eta(table, n) + var.sd * rng.standard_normal(n)
        return np.asarray(natural + float(self.delta), dtype=float)


# ------------------------------------------------------------------------------------ the law


@dataclass(frozen=True)
class Law:
    """An ordered list of variables with the estimator's design prefix, and its exact sums.

    Parameters
    ----------
    name : str
        The law's name.
    n_times : int
        The number of nodes.
    variables : tuple of Var
        The variables in data order.
    cause_names : tuple of str
        On a survival law, the cause names in label order (``1..K``); empty otherwise.
    """

    name: str
    n_times: int
    variables: tuple[Var, ...]
    cause_names: tuple[str, ...] = ()

    # -- structure ------------------------------------------------------------------------
    @property
    def table_names(self) -> tuple[str, ...]:
        """The design table's columns: baseline, then each node's covariates and treatment."""
        return tuple(
            v.name for v in self.variables if v.role in ("baseline", "covariate", "treatment")
        )

    def var(self, name: str) -> Var:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def of(self, role: str, node: int | None = None) -> list[Var]:
        return [v for v in self.variables if v.role == role and (node is None or v.node == node)]

    @property
    def is_survival(self) -> bool:
        return bool(self.cause_names)

    @property
    def has_censoring(self) -> bool:
        return bool(self.of("censor"))

    def columns(self) -> list[str]:
        """The frame's columns, in data order."""
        out: list[str] = []
        for v in self.variables:
            out += list(v.columns) if v.kind == "cat" else [v.name]
        return out

    def fit_columns(self, *, continuous_treatment: bool = False) -> dict[str, Any]:
        """The node declaration a fit takes."""
        times = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": [f"A{t}" for t in times],
            "baseline": [v.name for v in self.of("baseline")],
            "time_varying": [[v.name for v in self.of("covariate", t)] for t in times],
        }
        if self.is_survival:
            out["outcome"] = {
                cause: [self.var(f"E{t}").columns[i] for t in times]
                for i, cause in enumerate(self.cause_names)
            }
        else:
            out["outcome"] = self.of("outcome")[0].name
        if self.has_censoring:
            out["censoring"] = [f"C{t}" for t in times]
        if continuous_treatment:
            out["continuous_treatment"] = [f"A{t}" for t in times]
        return out

    def coefficients(self, name: str) -> np.ndarray:
        """The variable's parameter vector in the estimator's design order, intercept first.

        The design is ``[1, earlier table columns]``.  A multinomial variable returns
        ``(K - 1, p)``.  The estimator's active design may drop a column that is constant on
        the fit rows, which the caller handles.
        """
        v = self.var(name)
        parents = self._prefix(v)
        if v.kind == "cat":
            k = len(v.intercept)
            out = np.zeros((k, 1 + len(parents)))
            for j in range(k):
                out[j, 0] = v.intercept[j]
                for i, parent in enumerate(parents):
                    out[j, 1 + i] = v.coef.get(parent, (0.0,) * k)[j]
            return out
        flat = np.zeros(1 + len(parents))
        flat[0] = float(v.intercept)
        for i, parent in enumerate(parents):
            flat[1 + i] = float(v.coef.get(parent, 0.0))
        return flat

    def prefix_names(self, name: str) -> tuple[str, ...]:
        """The design columns the estimator's model of ``name`` reads."""
        return self._prefix(self.var(name))

    def _prefix(self, v: Var) -> tuple[str, ...]:
        names = self.table_names
        if v.role in ("baseline",):
            return ()
        if v.role in ("covariate", "treatment"):
            return names[: names.index(v.name)]
        if v.role == "censor":
            # Retention reads the history through the node's treatment.
            last = self.var(f"A{v.node}").name
            return names[: names.index(last) + 1]
        if v.role == "event":
            last = self.var(f"A{v.node}").name
            return names[: names.index(last) + 1]
        return names  # the end-of-study outcome reads the whole history

    def check(self) -> None:
        """Every coefficient names an earlier column of the variable's own prefix."""
        if any(v.quad for v in self.variables):
            return
        for v in self.variables:
            allowed = set(self._prefix(v))
            extra = set(v.coef) - allowed
            if extra:
                raise ValueError(f"{self.name}: {v.name} reads {sorted(extra)}, not in its prefix")

    # -- sampling -------------------------------------------------------------------------
    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` independent rows; a later node of a censored or failed unit is ``nan``."""
        rng = np.random.default_rng(seed)
        table: dict[str, np.ndarray] = {}
        drawn: dict[str, np.ndarray] = {}
        for v in self.variables:
            eta = v.eta(table, n)
            if v.kind == "gauss":
                value = eta + v.sd * rng.standard_normal(n)
            elif v.kind == "bern":
                value = (rng.random(n) < expit(eta)).astype(float)
            else:
                cumulative = np.cumsum(_softmax(eta), axis=1)
                value = np.minimum(
                    (rng.random(n)[:, None] >= cumulative).sum(axis=1), float(len(v.intercept))
                )
            drawn[v.name] = value
            if v.role in ("baseline", "covariate", "treatment"):
                table[v.name] = value
        return self._frame(drawn)

    def _frame(self, drawn: Mapping[str, np.ndarray]) -> pd.DataFrame:
        n = len(next(iter(drawn.values())))
        at_risk = np.ones(n, dtype=bool)
        start = at_risk.copy()
        retained = np.ones(n, dtype=bool)
        current = 0
        out: dict[str, np.ndarray] = {}
        for v in self.variables:
            if v.node != current:
                current = v.node
                start = at_risk.copy()
                retained = np.ones(n, dtype=bool)
            if v.node == 0:
                visible = np.ones(n, dtype=bool)
            elif v.role in ("covariate", "treatment", "censor"):
                visible = start
            else:
                visible = start & retained
            value = np.asarray(drawn[v.name], dtype=float)
            if v.kind == "cat":
                for index, column in enumerate(v.columns):
                    out[column] = np.where(visible, (value == index + 1).astype(float), np.nan)
            else:
                out[v.name] = np.where(visible, value, np.nan)
            if v.role == "censor":
                retained = value == 1
                at_risk = at_risk & retained
            if v.role == "event":
                at_risk = at_risk & (value == 0)
        return pd.DataFrame(out)[self.columns()]

    # -- the sums -------------------------------------------------------------------------
    def support(self, order: int = GH_ORDER) -> pd.DataFrame:
        """Every observable path with its probability in ``prob``.

        A Gaussian variable takes ``order`` Gauss-Hermite nodes.  A Gaussian end-of-study
        outcome is not expanded: it carries its conditional mean, which gives the same normal
        equations for the outcome regression.  The sum is exact for a binary law up to the
        quadrature of its Gaussian variables.
        """
        nodes, weights = _hermite(order)
        cols: dict[str, np.ndarray] = {}
        w = np.ones(1)
        active = np.ones(1, dtype=bool)
        for v in self.variables:
            n = w.size
            table = {k: cols[k] for k in self.table_names if k in cols}
            eta = v.eta(table, n)
            if v.kind == "gauss" and v.role == "outcome":
                cols[v.name] = np.where(active, eta, np.nan)
                continue
            k = {"gauss": order, "bern": 2, "cat": 3}[v.kind]
            index = np.flatnonzero(active)
            dead = np.flatnonzero(~active)
            reps = np.repeat(index, k)
            if v.kind == "gauss":
                values = (eta[index][:, None] + v.sd * nodes[None, :]).ravel()
                probs = np.tile(weights, index.size)
            elif v.kind == "bern":
                p = expit(eta[index])
                values = np.tile([0.0, 1.0], index.size)
                probs = np.column_stack([1.0 - p, p]).ravel()
            else:
                values = np.tile(np.arange(3, dtype=float), index.size)
                probs = _softmax(eta[index]).ravel()
            take = np.concatenate([dead, reps])
            cols = {name: a[take] for name, a in cols.items()}
            cols[v.name] = np.concatenate([np.full(dead.size, np.nan), values])
            w = np.concatenate([w[dead], w[reps] * probs])
            active = np.concatenate(
                [np.zeros(dead.size, dtype=bool), np.ones(reps.size, dtype=bool)]
            )
            if v.role == "censor":
                active = active & (cols[v.name] == 1)
            if v.role == "event":
                active = active & (cols[v.name] == 0)
        return self._support_frame(cols, w)

    def _support_frame(self, cols: Mapping[str, np.ndarray], w: np.ndarray) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        for v in self.variables:
            value = cols[v.name]
            if v.kind == "cat":
                for index, column in enumerate(v.columns):
                    out[column] = np.where(np.isnan(value), np.nan, (value == index + 1) * 1.0)
            else:
                out[v.name] = value
        frame = pd.DataFrame(out)[self.columns()]
        frame["prob"] = w
        return frame

    def g_formula(
        self,
        plan: Mapping[int, NodePlan],
        *,
        order: int = GH_ORDER,
        chunks: int = 1,
        horizons: Sequence[int] | None = None,
    ) -> Any:
        """The regimen mean (end of study) or the cumulative incidences (survival).

        Parameters
        ----------
        plan : mapping of int to NodePlan
            The node plan at every node.
        order : int
            Gauss-Hermite nodes per Gaussian variable.
        chunks : int
            The number of chunks the nodes of the first Gaussian variable are split into, to
            bound the memory.
        horizons : sequence of int or None
            On a survival law, the horizons to report.

        Returns
        -------
        float or dict
            The mean, or ``{(cause, horizon): incidence}``.
        """
        nodes, weights = _hermite(order)
        total: Any = 0.0 if not self.is_survival else {}
        first = self.of("baseline")[0]
        split = np.array_split(np.arange(order), chunks)
        for part in split:
            result = self._walk(plan, nodes, weights, order, first.name, part, horizons)
            if self.is_survival:
                for key, value in result.items():
                    total[key] = total.get(key, 0.0) + value
            else:
                total += result
        return total

    def _walk(
        self,
        plan: Mapping[int, NodePlan],
        nodes: np.ndarray,
        weights: np.ndarray,
        order: int,
        first: str,
        part: np.ndarray,
        horizons: Sequence[int] | None,
    ) -> Any:
        cols: dict[str, np.ndarray] = {}
        w = np.ones(1)
        n = 1
        for v in self.variables:
            if v.role in ("censor", "event"):
                continue
            table = {k: cols[k] for k in self.table_names if k in cols}
            if v.role == "outcome":
                eta = v.eta(table, n)
                f = expit(eta) if v.kind == "bern" else eta
                return float(w @ f)
            eta = v.eta(table, n)
            if v.role == "treatment":
                values, probs = plan[v.node].branches(table, v, n)
            elif v.kind == "gauss":
                if v.name == first:
                    values = eta[:, None] + v.sd * nodes[part][None, :]
                    probs = np.tile(weights[part], (n, 1))
                else:
                    values = eta[:, None] + v.sd * nodes[None, :]
                    probs = np.tile(weights, (n, 1))
            else:
                p = expit(eta)
                values = np.tile([0.0, 1.0], (n, 1))
                probs = np.column_stack([1.0 - p, p])
            k = values.shape[1]
            reps = np.repeat(np.arange(n), k)
            cols = {name: a[reps] for name, a in cols.items()}
            cols[v.name] = values.ravel()
            w = w[reps] * probs.ravel()
            n = w.size
        return self._incidence(cols, w, n, horizons)

    def _incidence(
        self, cols: Mapping[str, np.ndarray], w: np.ndarray, n: int, horizons: Sequence[int] | None
    ) -> dict[tuple[str, int], float]:
        """The cumulative incidences: ``sum_t S(t-1) h_jt`` along each path, then the mean."""
        hazards = np.zeros((n, self.n_times, len(self.cause_names)))
        for t in range(1, self.n_times + 1):
            ev = self.var(f"E{t}")
            table = {k: cols[k] for k in self.table_names if k in cols}
            hazards[:, t - 1, :] = _softmax(ev.eta(table, n))[:, 1:]
        total = hazards.sum(axis=2)
        survival = np.concatenate([np.ones((n, 1)), np.cumprod(1.0 - total, axis=1)], axis=1)
        incidence = np.cumsum(survival[:, :-1, None] * hazards, axis=1)
        out: dict[tuple[str, int], float] = {}
        for h in horizons if horizons is not None else (self.n_times,):
            for index, cause in enumerate(self.cause_names):
                out[(cause, int(h))] = float(w @ incidence[:, h - 1, index])
        return out

    def monte_carlo(
        self,
        plan: Mapping[int, NodePlan],
        n: int,
        seed: int,
        *,
        horizons: Sequence[int] | None = None,
    ) -> Any:
        """The regimen mean (or incidences) by forward simulation under ``plan``, with its SE.

        Returns ``(estimate, standard_error)``, or a dict of such pairs on a survival law.
        """
        rng = np.random.default_rng(seed)
        table: dict[str, np.ndarray] = {}
        for v in self.variables:
            if v.role in ("censor", "event"):
                continue
            eta = v.eta(table, n)
            if v.role == "outcome":
                f = expit(eta) if v.kind == "bern" else eta
                draw = (rng.random(n) < f).astype(float) if v.kind == "bern" else f
                return float(draw.mean()), float(draw.std() / np.sqrt(n))
            if v.role == "treatment":
                table[v.name] = plan[v.node].draw(table, v, n, rng)
            elif v.kind == "gauss":
                table[v.name] = eta + v.sd * rng.standard_normal(n)
            else:
                table[v.name] = (rng.random(n) < expit(eta)).astype(float)
        # survival: draw the events along the path
        times = range(1, self.n_times + 1)
        k = len(self.cause_names)
        alive = np.ones(n, dtype=bool)
        hits = {(cause, t): np.zeros(n) for cause in self.cause_names for t in times}
        for t in times:
            probs = _softmax(self.var(f"E{t}").eta(table, n))
            cumulative = np.cumsum(probs, axis=1)
            label = np.minimum((rng.random(n)[:, None] >= cumulative).sum(axis=1), k)
            for index, cause in enumerate(self.cause_names):
                happened = alive & (label == index + 1)
                for later in range(t, self.n_times + 1):
                    hits[(cause, later)] += happened
            alive = alive & (label == 0)
        report = horizons if horizons is not None else (self.n_times,)
        return {
            (cause, h): (float(hits[(cause, h)].mean()), float(hits[(cause, h)].std() / np.sqrt(n)))
            for cause in self.cause_names
            for h in report
        }

    def with_variable(self, name: str, **changes: Any) -> Law:
        """A copy of the law with one variable's fields changed."""
        updated = tuple(replace(v, **changes) if v.name == name else v for v in self.variables)
        return replace(self, variables=updated)


# ---------------------------------------------------------------------------------- law G1


def _history(*pairs: tuple[str, float]) -> dict[str, float]:
    return dict(pairs)


def _g1() -> Law:
    """Law ``G1``: binary covariates, censoring and a binary end-of-study outcome."""
    v: list[Var] = [Var("W", "gauss", "baseline", 0, 0.0)]
    # node 1
    v += [
        Var("L1", "bern", "covariate", 1, -0.2, {"W": 0.8}),
        Var("A1", "bern", "treatment", 1, 0.1, {"W": 0.4, "L1": 0.6}),
        Var("C1", "bern", "censor", 1, 2.5, {"W": 0.2, "L1": -0.3, "A1": -0.3}),
    ]
    v += [
        Var("L2", "bern", "covariate", 2, -0.4, {"W": 0.5, "L1": 1.4, "A1": 0.8}),
        Var("A2", "bern", "treatment", 2, 0.0, {"W": 0.3, "L1": 0.3, "A1": 0.5, "L2": 0.6}),
        Var("C2", "bern", "censor", 2, 2.5, {"W": 0.2, "L1": -0.2, "L2": -0.3, "A2": -0.3}),
    ]
    v += [
        Var(
            "L3",
            "bern",
            "covariate",
            3,
            -0.4,
            {"W": 0.4, "L1": 0.6, "A1": 0.3, "L2": 1.4, "A2": 0.8},
        ),
        Var(
            "A3",
            "bern",
            "treatment",
            3,
            -0.1,
            {"W": 0.3, "L1": 0.2, "A1": 0.3, "L2": 0.3, "A2": 0.5, "L3": 0.6},
        ),
        Var("C3", "bern", "censor", 3, 2.5, {"W": 0.2, "L3": -0.3, "A3": -0.3}),
        Var(
            "Y",
            "bern",
            "outcome",
            3,
            -0.8,
            {
                "W": 0.5,
                "L1": 0.6,
                "A1": -0.2,
                "L2": 0.5,
                "A2": -0.3,
                "L3": 1.2,
                "A3": -0.9,
            },
        ),
    ]
    return Law("G1", 3, tuple(v))


G1 = _g1()

# ---------------------------------------------------------------------------------- law G2


def _g2() -> Law:
    """Law ``G2``: a Gaussian then a binary covariate per node, a Gaussian outcome.

    The binary covariate :math:`B_t` reads :math:`Z_t` with slope 3, so the regimen mean is
    steep and nonlinear in each Gaussian draw.  The outcome reads the simulated nodes strongly
    and the observed history weakly, which gives the simulation a visible share of the interval
    at one simulated history per unit.
    """
    v: list[Var] = [Var("W", "gauss", "baseline", 0, 0.0)]
    v += [
        Var("Z1", "gauss", "covariate", 1, 0.0, {"W": 0.5}, sd=0.9),
        Var("B1", "bern", "covariate", 1, -0.2, {"W": 0.4, "Z1": 3.0}),
        Var("A1", "bern", "treatment", 1, 0.0, {"W": 0.4, "Z1": 0.5, "B1": 0.7}),
    ]
    v += [
        Var(
            "Z2",
            "gauss",
            "covariate",
            2,
            0.0,
            {"W": 0.3, "Z1": 0.5, "B1": 0.5, "A1": -0.4},
            sd=0.9,
        ),
        Var(
            "B2",
            "bern",
            "covariate",
            2,
            -0.3,
            {"W": 0.3, "Z1": 0.3, "B1": 0.4, "A1": 1.5, "Z2": 3.0},
        ),
        Var(
            "A2",
            "bern",
            "treatment",
            2,
            0.0,
            {"W": 0.3, "Z1": 0.3, "B1": 0.3, "A1": 0.4, "Z2": 0.5, "B2": 0.7},
        ),
    ]
    v += [
        Var(
            "Z3",
            "gauss",
            "covariate",
            3,
            0.0,
            {"W": 0.3, "Z1": 0.2, "B1": 0.2, "A1": -0.2, "Z2": 0.5, "B2": 0.5, "A2": -0.4},
            sd=0.9,
        ),
        Var(
            "B3",
            "bern",
            "covariate",
            3,
            -0.3,
            {"W": 0.3, "B1": 0.3, "Z2": 0.3, "B2": 0.4, "A2": 1.5, "Z3": 3.0},
        ),
        Var(
            "A3",
            "bern",
            "treatment",
            3,
            0.0,
            {"W": 0.3, "Z2": 0.3, "B2": 0.3, "A2": 0.4, "Z3": 0.5, "B3": 0.7},
        ),
        Var(
            "Y",
            "gauss",
            "outcome",
            3,
            1.0,
            {
                "W": 0.2,
                "Z1": 0.2,
                "B1": 0.3,
                "A1": -0.2,
                "Z2": 1.2,
                "B2": 2.5,
                "A2": -0.2,
                "Z3": 1.5,
                "B3": 3.0,
                "A3": -0.8,
            },
            sd=1.0,
        ),
    ]
    return Law("G2", 3, tuple(v))


G2 = _g2()

#: The known stochastic policy of ``G2``: treat with probability :math:`0.3 + 0.4 B_t`.
G2_POLICY = (0.3, 0.4)

# ---------------------------------------------------------------------------------- law G3

CAUSES = ("relapse", "death")


def _g3() -> Law:
    """Law ``G3``: four nodes, binary covariates, censoring and two competing causes."""
    v: list[Var] = [Var("W", "gauss", "baseline", 0, 0.0)]
    for t in (1, 2, 3, 4):
        # covariate: main terms in the whole earlier table
        if t == 1:
            cov = {"W": 0.7}
        else:
            cov = {"W": 0.5, f"L{t - 1}": 1.2, f"A{t - 1}": 0.4}
            if t >= 3:
                cov["L1"] = 0.3
        v.append(Var(f"L{t}", "bern", "covariate", t, -0.3, cov))
        treat = {"W": 0.3, f"L{t}": 0.7}
        if t >= 2:
            treat[f"A{t - 1}"] = 0.3
        v.append(Var(f"A{t}", "bern", "treatment", t, -0.1, treat))
        keep = {"W": 0.2, f"L{t}": -0.3, f"A{t}": -0.3}
        v.append(Var(f"C{t}", "bern", "censor", t, 3.0, keep))
        # one coefficient pair (relapse, death) per earlier column the event reads
        event: dict[str, tuple[float, float]] = {
            "W": (0.4, 0.3),
            f"L{t}": (0.8, 0.5),
            f"A{t}": (-0.6, -0.2),
        }
        if t >= 2:
            event[f"L{t - 1}"] = (0.3, 0.0)
        v.append(Var(f"E{t}", "cat", "event", t, (-2.2, -3.0), event, columns=(f"R{t}", f"D{t}")))
    return Law("G3", 4, tuple(v), cause_names=CAUSES)


G3 = _g3()

# ---------------------------------------------------------------------------------- law G4


def _g4(quadratic: float = 0.0) -> Law:
    """Law ``G4``: Gaussian covariates and doses, a Gaussian outcome, linear throughout."""
    v: list[Var] = [Var("W", "gauss", "baseline", 0, 0.0)]
    v += [
        Var("L1", "gauss", "covariate", 1, 0.0, {"W": 0.5}, sd=0.9),
        Var("A1", "gauss", "treatment", 1, 0.0, {"W": 0.3, "L1": 0.5}, sd=1.0),
    ]
    v += [
        Var("L2", "gauss", "covariate", 2, 0.0, {"W": 0.3, "L1": 0.4, "A1": 0.5}, sd=0.9),
        Var(
            "A2",
            "gauss",
            "treatment",
            2,
            0.0,
            {"W": 0.2, "L1": 0.3, "A1": 0.4, "L2": 0.5},
            sd=1.0,
        ),
    ]
    v += [
        Var(
            "L3",
            "gauss",
            "covariate",
            3,
            0.0,
            {"W": 0.3, "L1": 0.2, "A1": 0.2, "L2": 0.4, "A2": 0.5},
            sd=0.9,
        ),
        Var(
            "A3",
            "gauss",
            "treatment",
            3,
            0.0,
            {"W": 0.2, "L2": 0.2, "A2": 0.4, "L3": 0.5},
            sd=1.0,
        ),
        Var(
            "Y",
            "gauss",
            "outcome",
            3,
            1.0,
            {
                "W": 0.5,
                "L1": 0.4,
                "A1": 0.2,
                "L2": 0.3,
                "A2": 0.3,
                "L3": 0.4,
                "A3": 0.5,
            },
            sd=1.0,
            quad={"A3": quadratic} if quadratic else {},
        ),
    ]
    return Law("G4Q" if quadratic else "G4", 3, tuple(v))


G4 = _g4()
#: The quadratic-dose law: :math:`q A_3^2` in the mean of :math:`Y`, which a main-terms fit misses.
G4_QUADRATIC = 0.3
G4Q = _g4(G4_QUADRATIC)

# ------------------------------------------------------------------- linear means (law G4)


def linear_means(law: Law, plan: Mapping[int, tuple[str, float]]) -> float:
    """The regimen mean of a linear-Gaussian law by propagating means through the recursion.

    Every conditional mean is linear in the earlier columns, so the mean of a column under a
    plan is its intercept plus the coefficients times the means of its parents.  A plan node is
    ``("dose", d)`` (assign ``d``) or ``("shift", delta)`` (the natural dose plus ``delta``; the
    natural dose is the law's own, given the intervened history).  A squared term of the
    outcome needs a fixed dose, whose square is that of the dose.
    """
    means: dict[str, float] = {}
    for v in law.variables:
        if v.role in ("censor", "event"):
            continue
        base = float(v.intercept) + sum(float(c) * means[parent] for parent, c in v.coef.items())
        if v.role == "treatment":
            kind, value = plan[v.node]
            if kind == "dose":
                means[v.name] = float(value)
            elif kind == "shift":
                means[v.name] = base + float(value)
            else:
                raise ValueError(f"unknown plan kind {kind!r}")
        elif v.role == "outcome":
            for name, q in v.quad.items():
                kind, value = plan[law.var(name).node]
                if kind != "dose":
                    raise ValueError("a squared term needs a fixed dose")
                base += float(q) * float(value) ** 2
            return base
        else:
            means[v.name] = base
    raise ValueError("the law has no outcome")


def linear_gaussian_plan(kind: str, value: float, n_times: int) -> dict[int, tuple[str, float]]:
    """The same ``(kind, value)`` at every node."""
    return dict.fromkeys(range(1, n_times + 1), (kind, value))


__all__ = [
    "GH_ORDER",
    "Fixed",
    "Law",
    "NodePlan",
    "Policy",
    "Rule",
    "ShiftPlan",
    "Tilt",
    "Var",
    "linear_means",
]
