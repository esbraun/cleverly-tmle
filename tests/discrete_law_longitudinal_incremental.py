r"""The laws of the registered incremental studies, with exact truths and influence functions.

``longitudinal-incremental`` (S1) and ``longitudinal-incremental-dropout`` (S2) read the laws
of ``X19-plan.md`` section 3.5.  Every law here has binary or three-level discrete nodes and,
at the end of study, a continuous outcome :math:`Y = \mu(h) + \sigma\varepsilon` with
:math:`\varepsilon \sim N(0, 1)`.  So every truth is a finite sum over the support.

======  ===============================================================================
law     structure
======  ===============================================================================
``A``   :math:`T = 3`, :math:`W, A_1, L_2, A_2, L_3, A_3, Y`; the probe law of the plan
``A0``  ``A`` with :math:`\mu` free of every arm and each :math:`L_t` free of
        :math:`A_{t-1}`: the sharp null, where every tilt has one mean
``Aw``  ``A`` with every arm coefficient of :math:`\mu` and of each :math:`L_t` scaled by
        :data:`POWER_SCALE`
``Ax``  :math:`T = 2`, with exact 0/1 propensity cells and :math:`\pi = 0.02` cells
``C``   ``A`` with dropout after each arm, :math:`\omega_t \in [0.85, 0.97]`
``D``   :math:`T = 3` survival with dropout as ``C``; binary absorbing :math:`Y_t`, hazards
        in :math:`[0.05, 0.30]`
``Dc``  ``D`` with a second, competing cause at each node
``Ck``  ``C`` with a shared cluster effect :math:`b = \pm` :data:`CLUSTER_SHIFT`
======  ===============================================================================

A law is a joint array over its discrete axes.  The axis order at node :math:`t` is
:math:`L_t, A_t, C_t, Y_t`, each present when the law has it.  After a dropout or an event
every later axis sits at level 0, which is the support's structural zero.

:func:`functional` evaluates the g-formula of a plan backwards along the axes.  A tilt node
reads :math:`\pi_t` off the law it is handed, so the complex-step derivative of
:func:`functional` carries the mechanism term (Kennedy 2019, Theorem 2).  :func:`eif_sd` adds
the outcome-noise part :math:`W_T(o)\,(Y - \mu(o))`, whose coefficient is the derivative of the
functional in :math:`\mu(o)` over :math:`p(o)`.

The module imports nothing from ``cleverly``: it is an independent statement of the
estimand (``tests/unit/test_oracle_independence.py``).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

#: The outcome noise of every continuous-outcome law.
SIGMA = 1.0
#: The arm-effect scale of law ``Aw``, chosen in ``tests/unit/test_longitudinal_
#: incremental_design.py`` so that the exact-law power of ``odds x1.5 vs natural course`` at
#: n = 4,000 lies in [0.80, 0.90].  It is 0.882 at this scale.
POWER_SCALE = 0.033
#: The half-range of the binary cluster effect of law ``Ck``.
CLUSTER_SHIFT = 1.0
#: The outcome loading of the cluster effect in law ``Ck``.
CLUSTER_OUTCOME = 1.0
#: The complex step of :func:`eif`.
STEP = 1e-30


def expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


Node = tuple[str, Any]


def tilt(delta: float) -> Node:
    """An odds tilt of the treatment mechanism by ``delta``."""
    return ("tilt", float(delta))


def arm(level: int) -> Node:
    """A label node: every unit at risk takes ``level``."""
    return ("arm", int(level))


@dataclass(frozen=True)
class Axis:
    """One discrete axis of a law.

    Parameters
    ----------
    name : str
        The frame column, or ``""`` for an event axis whose columns :func:`frame` names.
    kind : {"baseline", "covariate", "treatment", "censoring", "event"}
        What the axis is.
    node : int
        The node it belongs to, counted from one; ``0`` for the baseline.
    levels : int
        The number of levels.  An event axis with three levels is a competing-risk node:
        level 1 the cause of interest, level 2 the competing cause.
    """

    name: str
    kind: str
    node: int
    levels: int = 2


Conditional = Callable[[Axis, Mapping[str, int]], Sequence[float]]


@dataclass(frozen=True)
class Law:
    """A discrete law, its outcome mean and its node structure.

    Parameters
    ----------
    name : str
        The law's label.
    axes : tuple of Axis
        The axes in time order.
    conditional : callable
        ``(axis, prefix) -> probabilities of each level`` given the values of the earlier
        axes, keyed by axis name.
    mean : callable or None
        ``prefix -> mu`` on a full uncensored history; ``None`` on a survival law.
    n_times : int
        The number of treatment nodes.
    """

    name: str
    axes: tuple[Axis, ...]
    conditional: Conditional
    mean: Callable[[Mapping[str, int]], float] | None
    n_times: int

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.levels for axis in self.axes)

    @property
    def survival(self) -> bool:
        return any(axis.kind == "event" for axis in self.axes)

    @cached_property
    def cells(self) -> tuple[tuple[int, ...], ...]:
        return tuple(itertools.product(*(range(axis.levels) for axis in self.axes)))

    def _structural(self, values: Mapping[str, int], axis: Axis) -> bool:
        """Whether ``axis`` sits after a dropout or an event in ``values``."""
        del axis
        return any(
            (earlier.kind == "censoring" and values[earlier.name] == 0)
            or (earlier.kind == "event" and values[earlier.name] != 0)
            for earlier in self.axes
            if earlier.name in values
        )

    @cached_property
    def probs(self) -> np.ndarray:
        """The joint array of cell probabilities."""
        out = np.zeros(self.shape)
        for cell in self.cells:
            values: dict[str, int] = {}
            mass = 1.0
            for axis, level in zip(self.axes, cell, strict=True):
                if self._structural(values, axis):
                    mass *= 1.0 if level == 0 else 0.0
                else:
                    mass *= float(self.conditional(axis, values)[level])
                values[axis.name] = level
                if mass == 0.0:
                    break
            out[cell] = mass
        if abs(out.sum() - 1.0) > 1e-12:
            raise AssertionError(f"law {self.name} sums to {out.sum()}")
        return out

    @cached_property
    def mu(self) -> np.ndarray:
        """The outcome mean on every cell, zero where it is undefined."""
        out = np.zeros(self.shape)
        if self.mean is None:
            return out
        for cell in self.cells:
            values = {axis.name: level for axis, level in zip(self.axes, cell, strict=True)}
            if (
                self.probs[cell] > 0
                and not self._structural(values, self.axes[-1])
                and all(values[a.name] == 1 for a in self.axes if a.kind == "censoring")
            ):
                out[cell] = self.mean(values)
        return out

    def axis_of(self, kind: str, node: int) -> int | None:
        for index, axis in enumerate(self.axes):
            if axis.kind == kind and axis.node == node:
                return index
        return None


# ------------------------------------------------------------------------- the functional


def functional(
    law: Law,
    probs: Any,
    plan: Sequence[Node],
    *,
    horizon: int | None = None,
    cause: int = 1,
    known: tuple[int, ...] = (),
    mu: Any = None,
    weights: Any = None,
) -> Any:
    r"""The g-formula of ``plan`` on the law with cell masses ``probs``.

    ``probs`` has the law's shape, optionally with a trailing batch axis.  A node in
    ``known`` reads :math:`\pi_t` off :attr:`Law.probs`, the unperturbed law, so its
    derivative carries no mechanism term.  Every dropout is prevented.  On a survival law
    the value is the risk of ``cause`` by ``horizon``; otherwise it is the mean of
    :math:`Y`, with ``mu`` in place of :attr:`Law.mu` when given.  ``weights`` is a
    weight per cell: the functional is then read on the weighted law
    :math:`dP_w = w\,dP / E_P[w]`.
    """
    p = np.asarray(probs)
    batched = p.ndim == len(law.shape) + 1
    if not batched:
        p = p[..., None]
    if weights is not None:
        cell = np.asarray(weights, dtype=float)[..., None]
        p = cell * p
        p = p / p.reshape(-1, p.shape[-1]).sum(axis=0)
    base = law.probs if weights is None else _weighted_base(law, weights)
    k = len(law.axes)
    # Marginal masses of each prefix: margins[j] has the first j axes.
    margins = [p.sum(axis=tuple(range(j, k))) for j in range(k + 1)]
    base_margins = [base.sum(axis=tuple(range(j, k))) for j in range(k + 1)]
    batch = p.shape[-1]
    if law.survival:
        value: Any = np.zeros(p.shape, dtype=p.dtype)
    else:
        mean = law.mu if mu is None else np.asarray(mu)
        if mean.ndim == len(law.shape):
            mean = mean[..., None]
        value = np.broadcast_to(mean, p.shape).astype(complex if np.iscomplexobj(mean) else p.dtype)
    for j in range(k - 1, -1, -1):
        axis = law.axes[j]
        parent = margins[j][..., None, :]
        reached = base_margins[j][..., None, None] > 0
        conditional = np.where(
            reached, margins[j + 1] / np.where(reached, parent, 1.0), 0.0
        )  # shape: prefix + (levels, batch)
        value = _reduce(law, axis, j, value, conditional, plan, horizon, cause, known)
    out = value.reshape(-1)[0] if not batched else value.reshape(batch)
    return out


def _weighted_base(law: Law, weights: Any) -> np.ndarray:
    tilted = np.asarray(weights, dtype=float) * law.probs
    return tilted / tilted.sum()


def _take(value: np.ndarray, j: int, level: int) -> np.ndarray:
    return np.take(value, level, axis=j)


def _reduce(
    law: Law,
    axis: Axis,
    j: int,
    value: np.ndarray,
    conditional: np.ndarray,
    plan: Sequence[Node],
    horizon: int | None,
    cause: int,
    known: tuple[int, ...],
) -> np.ndarray:
    r"""Integrate axis ``j`` out of ``value``, which holds every later axis summed already.

    ``value`` has shape ``prefix + (levels of j,) + (batch,)`` after the later axes were
    reduced; ``conditional`` has the same shape and holds :math:`P(x_j \mid prefix)`.
    """
    levels = axis.levels
    pieces = [_take(value, j, level) for level in range(levels)]
    probs = [_take(conditional, j, level) for level in range(levels)]
    if axis.kind in {"baseline", "covariate"}:
        return sum(prob * piece for prob, piece in zip(probs, pieces, strict=True))
    if axis.kind == "censoring":
        return pieces[1]
    if axis.kind == "event":
        within = horizon is None or axis.node <= horizon
        out = probs[0] * pieces[0]
        if within:
            out = out + probs[cause]
        return out
    node = plan[axis.node - 1]
    pi = _known_pi(law, axis, j, pieces[1].shape) if axis.node in known else probs[1]
    q1 = _q1(node, pi)
    return q1 * pieces[1] + (1.0 - q1) * pieces[0]


def _known_pi(law: Law, axis: Axis, j: int, shape: tuple[int, ...]) -> np.ndarray:
    base = law.probs
    k = len(law.axes)
    upper = base.sum(axis=tuple(range(j + 1, k)))
    lower = upper.sum(axis=j)
    treated = np.take(upper, 1, axis=j)
    pi = np.where(lower > 0, treated / np.where(lower > 0, lower, 1.0), 0.0)
    return np.broadcast_to(pi[..., None], shape)


def _q1(node: Node, pi: Any) -> Any:
    kind, value = node
    if kind == "tilt":
        return value * pi / (value * pi + 1.0 - pi)
    if kind == "arm":
        return np.ones_like(pi) * float(value)
    raise ValueError(f"unknown node kind {kind!r}")


def eif(
    law: Law,
    plan: Sequence[Node],
    *,
    other: Sequence[Node] | None = None,
    **options: Any,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The cell part of the influence function and the outcome coefficient, at each cell.

    Returns ``(cells, coefficient)``, both of the law's shape.  ``cells`` is the complex-step
    derivative of :func:`functional` toward each cell with positive mass, ``nan`` elsewhere.
    ``coefficient`` is :math:`\partial\psi / \partial\mu(o)` over :math:`p(o)`: the weight
    of the outcome residual :math:`Y - \mu(o)` in the influence function.  With ``other``,
    both are of the contrast ``plan`` minus ``other``.
    """
    probs = law.probs
    flat = probs.reshape(-1)
    positive = np.flatnonzero(flat > 0)
    batch = np.zeros((flat.size, positive.size), dtype=complex)
    batch[:, :] = (1.0 - 1j * STEP) * flat[:, None]
    batch[positive, np.arange(positive.size)] += 1j * STEP
    batch = batch.reshape((*law.shape, positive.size))

    def contrast(argument: Any, **extra: Any) -> Any:
        value = functional(law, argument, plan, **options, **extra)
        if other is not None:
            value = value - functional(law, argument, other, **options, **extra)
        return value

    cells = np.full(flat.size, np.nan)
    cells[positive] = np.imag(np.asarray(contrast(batch))) / STEP
    coefficient = np.zeros(flat.size)
    if not law.survival:
        mu = law.mu.reshape(-1)
        observed = _uncensored_cells(law, positive)
        moved = np.zeros((flat.size, observed.size), dtype=complex)
        moved[:, :] = mu[:, None]
        moved[observed, np.arange(observed.size)] += 1j * STEP
        shape = (*law.shape, observed.size)
        masses = np.broadcast_to(flat[:, None], moved.shape).reshape(shape)
        derivative = np.imag(np.asarray(contrast(masses, mu=moved.reshape(shape)))) / STEP
        coefficient[observed] = derivative / flat[observed]
    return cells.reshape(law.shape), coefficient.reshape(law.shape)


def _uncensored_cells(law: Law, positive: np.ndarray) -> np.ndarray:
    keep = []
    for index in positive:
        cell = np.unravel_index(index, law.shape)
        if all(
            cell[position] == 1
            for position, axis in enumerate(law.axes)
            if axis.kind == "censoring"
        ):
            keep.append(index)
    return np.asarray(keep, dtype=int)


def eif_sd(law: Law, plan: Sequence[Node], **options: Any) -> float:
    r"""The standard deviation of the efficient influence function of ``plan``.

    :math:`\sqrt{\sum_o p(o) D(o)^2 + \sigma^2 \sum_o p(o) c(o)^2}`, with ``D`` and ``c``
    from :func:`eif`.  ``other=`` makes it the contrast's.
    """
    cells, coefficient = eif(law, plan, **options)
    probs = law.probs
    keep = probs > 0
    variance = float(np.sum(probs[keep] * cells[keep] ** 2))
    variance += SIGMA**2 * float(np.sum(probs[keep] * coefficient[keep] ** 2))
    return float(np.sqrt(variance))


def truth(law: Law, plan: Sequence[Node], **options: Any) -> float:
    return float(np.real(functional(law, law.probs, plan, **options)))


# ------------------------------------------------------------------------- the laws


def _bern(p: float) -> tuple[float, float]:
    return (1.0 - p, p)


def _node_axes(n_times: int, *, censoring: bool, survival: int = 0) -> tuple[Axis, ...]:
    axes = [Axis("W", "baseline", 0)]
    for t in range(1, n_times + 1):
        if t > 1:
            axes.append(Axis(f"L{t}", "covariate", t))
        axes.append(Axis(f"A{t}", "treatment", t))
        if censoring:
            axes.append(Axis(f"C{t}", "censoring", t))
        if survival:
            axes.append(Axis(f"Y{t}", "event", t, levels=survival + 1))
    return tuple(axes)


def _probe_conditional(*, null: bool = False, scale: float = 1.0) -> Conditional:
    """The conditionals of law ``A`` and of its variants; ``scale`` multiplies the arm effects."""
    effect = 0.0 if null else 1.2 * scale

    def conditional(axis: Axis, v: Mapping[str, int]) -> Sequence[float]:
        w = v.get("W", 0)
        if axis.kind == "baseline":
            return (0.5, 0.5)
        if axis.name == "A1":
            return _bern(expit(-0.4 - 0.6 * w))
        if axis.name == "L2":
            return _bern(expit(-0.5 + effect * v["A1"] + 0.3 * w))
        if axis.name == "A2":
            return _bern(expit(-0.5 + 1.0 * v["L2"] + 0.8 * v["A1"] - 0.6 * w))
        if axis.name == "L3":
            return _bern(expit(-0.5 + effect * v["A2"] + 0.5 * v["L2"] + 0.3 * w))
        if axis.name == "A3":
            return _bern(expit(-0.5 + 1.0 * v["L3"] + 0.8 * v["A2"] - 0.6 * w))
        if axis.kind == "censoring":
            t = axis.node
            score = 2.6 + 0.5 * v[f"A{t}"] - 0.4 * w + (0.3 * v[f"L{t}"] if t > 1 else 0.0)
            return _bern(min(max(expit(score), 0.85), 0.97))
        raise KeyError(axis.name)

    return conditional


def _probe_mean(scale: float = 1.0, *, null: bool = False) -> Callable[[Mapping[str, int]], float]:
    def mean(v: Mapping[str, int]) -> float:
        if null:
            return 1.0 + v["L3"] + 0.5 * v["W"]
        arms = v["A1"] + 1.5 * v["A2"] + 2.0 * v["A3"] + v["A3"] * v["L3"]
        return 1.0 + scale * arms + v["L3"] + 0.5 * v["W"]

    return mean


def _law_a(name: str, *, scale: float = 1.0, null: bool = False, censoring: bool = False) -> Law:
    return Law(
        name=name,
        axes=_node_axes(3, censoring=censoring),
        conditional=_probe_conditional(null=null, scale=scale),
        mean=_probe_mean(scale, null=null),
        n_times=3,
    )


def _law_ax() -> Law:
    """Two nodes; exact 0/1 propensities in some cells and :math:`\\pi = 0.02` in others."""

    def conditional(axis: Axis, v: Mapping[str, int]) -> Sequence[float]:
        if axis.kind == "baseline":
            return (0.5, 0.5)
        if axis.name == "A1":
            return _bern(0.0 if v["W"] == 0 else 0.5)
        if axis.name == "L2":
            return _bern(expit(-0.3 + 1.0 * v["A1"] + 0.4 * v["W"]))
        if axis.name == "A2":
            table = {(0, 0, 0): 0.02, (0, 0, 1): 1.0, (1, 0, 0): 0.4, (1, 0, 1): 0.02}
            table |= {(1, 1, 0): 0.6, (1, 1, 1): 0.0}
            return _bern(table.get((v["W"], v["A1"], v["L2"]), 0.5))
        raise KeyError(axis.name)

    def mean(v: Mapping[str, int]) -> float:
        return 1.0 + v["A1"] + 1.5 * v["A2"] + v["L2"] + 0.5 * v["W"] + v["A2"] * v["L2"]

    return Law("Ax", _node_axes(2, censoring=False), conditional, mean, 2)


def _survival_conditional(*, competing: bool) -> Conditional:
    base = _probe_conditional()

    def conditional(axis: Axis, v: Mapping[str, int]) -> Sequence[float]:
        if axis.kind != "event":
            return base(axis, v)
        t = axis.node
        w, a = v["W"], v[f"A{t}"]
        l = v.get(f"L{t}", 0)
        hazard = min(max(0.12 + 0.06 * w - 0.05 * a + 0.08 * l + 0.03 * t, 0.05), 0.30)
        if not competing:
            return _bern(hazard)
        other = min(max(0.06 + 0.04 * a - 0.02 * w + 0.02 * t, 0.05), 0.30)
        return (1.0 - hazard - other, hazard, other)

    return conditional


def _law_d(name: str, *, competing: bool = False) -> Law:
    return Law(
        name=name,
        axes=_node_axes(3, censoring=True, survival=2 if competing else 1),
        conditional=_survival_conditional(competing=competing),
        mean=None,
        n_times=3,
    )


def _clustered_conditional(shift: float) -> Conditional:
    base = _probe_conditional()

    def conditional(axis: Axis, v: Mapping[str, int]) -> Sequence[float]:
        if axis.name == "L2":
            return _bern(expit(-0.5 + 1.2 * v["A1"] + 0.3 * v["W"] + shift))
        return base(axis, v)

    return conditional


LAW_A = _law_a("A")
LAW_A0 = _law_a("A0", null=True)
LAW_AW = _law_a("Aw", scale=POWER_SCALE)
LAW_AX = _law_ax()
LAW_C = _law_a("C", censoring=True)
LAW_D = _law_d("D")
LAW_DC = _law_d("Dc", competing=True)


@dataclass(frozen=True)
class ClusteredLaw:
    """Law ``C`` with a shared cluster effect :math:`b = \\pm` :data:`CLUSTER_SHIFT`.

    Every unit of a cluster shares :math:`b`, which shifts the logit of :math:`L_2` by
    :math:`b` and the outcome mean by :data:`CLUSTER_OUTCOME` :math:`b`.  The two halves are
    exact laws, so the marginal unit law is their average and its truth is exact.
    """

    halves: tuple[Law, Law] = field(
        default_factory=lambda: tuple(  # type: ignore[return-value]
            Law(
                f"Ck{sign:+d}",
                _node_axes(3, censoring=True),
                _clustered_conditional(sign * CLUSTER_SHIFT),
                _shifted_mean(sign * CLUSTER_OUTCOME * CLUSTER_SHIFT),
                3,
            )
            for sign in (1, -1)
        )
    )

    @cached_property
    def marginal(self) -> Law:
        """The unit law: the average of the halves, with the mixed outcome mean."""
        upper, lower = self.halves
        probs = 0.5 * upper.probs + 0.5 * lower.probs
        mixed = np.where(
            probs > 0,
            (0.5 * upper.probs * upper.mu + 0.5 * lower.probs * lower.mu)
            / np.where(probs > 0, probs, 1.0),
            0.0,
        )
        law = replace(upper, name="Ck")
        law.__dict__["probs"] = probs
        law.__dict__["mu"] = mixed
        return law


def _shifted_mean(shift: float) -> Callable[[Mapping[str, int]], float]:
    base = _probe_mean()

    def mean(v: Mapping[str, int]) -> float:
        return base(v) + shift

    return mean


LAW_CK = ClusteredLaw()

#: The tilt grid of the primary scenarios and its plan labels.
DELTAS: dict[str, float] = {"odds x0.5": 0.5, "natural course": 1.0, "odds x2": 2.0}
REFERENCE = "natural course"


def constant(delta: float, n_times: int = 3) -> tuple[Node, ...]:
    return tuple(tilt(delta) for _ in range(n_times))


# ------------------------------------------------------------------------- sampling


def _columns(law: Law) -> list[str]:
    return [axis.name for axis in law.axes if axis.kind != "event"]


def sample(law: Law, n: int, seed: int, *, probs: Any = None) -> pd.DataFrame:
    """``n`` units of ``law``; ``nan`` after a dropout or an event, as the estimator reads.

    A survival law writes ``Y1``..``YT`` (event of interest by node) and, with a competing
    cause, ``D1``..``DT`` for the competing cause.  A continuous-outcome law writes ``Y``.
    """
    rng = np.random.default_rng(seed)
    p = law.probs if probs is None else np.asarray(probs)
    flat = p.reshape(-1)
    drawn = rng.choice(flat.size, size=n, p=flat / flat.sum())
    cells = np.array(np.unravel_index(drawn, law.shape)).T
    data: dict[str, np.ndarray] = {}
    gone = np.zeros(n, dtype=bool)
    for position, axis in enumerate(law.axes):
        column = cells[:, position].astype(float)
        if axis.kind == "event":
            values = np.where(gone, np.nan, column)
            data[f"Y{axis.node}"] = np.where(np.isnan(values), np.nan, (values == 1).astype(float))
            if axis.levels == 3:
                data[f"D{axis.node}"] = np.where(
                    np.isnan(values), np.nan, (values == 2).astype(float)
                )
            gone = gone | (column != 0)
            continue
        data[axis.name] = np.where(gone, np.nan, column)
        if axis.kind == "censoring":
            gone = gone | (column == 0)
    frame = pd.DataFrame(data)
    if law.mean is not None:
        mu = law.mu.reshape(-1)[drawn]
        noise = rng.normal(scale=SIGMA, size=n)
        frame["Y"] = np.where(gone, np.nan, mu + noise)
    return frame


def sample_clustered(n: int, cluster_size: int, seed: int) -> pd.DataFrame:
    """Law ``Ck`` in clusters of ``cluster_size``, with an ``id`` column."""
    rng = np.random.default_rng(seed)
    clusters = n // cluster_size
    signs = rng.integers(0, 2, size=clusters)
    parts = []
    for cluster, sign in enumerate(signs):
        half = LAW_CK.halves[int(sign)]
        part = sample(half, cluster_size, int(rng.integers(2**63)))
        part["id"] = cluster
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def columns(law: Law) -> dict[str, Any]:
    """The column declaration of a fit on ``law``."""
    times = law.n_times
    out: dict[str, Any] = {
        "treatment": [f"A{t}" for t in range(1, times + 1)],
        "baseline": ["W"],
        "time_varying": [[]] + [[f"L{t}"] for t in range(2, times + 1)],
    }
    if any(axis.kind == "censoring" for axis in law.axes):
        out["censoring"] = [f"C{t}" for t in range(1, times + 1)]
    events = [axis for axis in law.axes if axis.kind == "event"]
    if not events:
        out["outcome"] = "Y"
    elif events[0].levels == 3:
        out["outcome"] = {
            "event": [f"Y{t}" for t in range(1, times + 1)],
            "competing": [f"D{t}" for t in range(1, times + 1)],
        }
    else:
        out["outcome"] = [f"Y{t}" for t in range(1, times + 1)]
    return out
