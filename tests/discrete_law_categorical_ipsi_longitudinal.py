r"""The longitudinal laws of the study ``categorical-incremental`` (X43, family L).

Two treatment nodes of three levels each, coded 0, 1 and 2 and called ``a``, ``b`` and ``c``
in the study's names, a binary covariate :math:`L_2` between them, a binary baseline
:math:`W`, and a continuous outcome :math:`Y = \mu(h) + \varepsilon` with
:math:`\varepsilon \sim N(0, 1)` at the end of study.  Every truth is a finite sum over the
support.

======  ==============================================================================
law     structure
======  ==============================================================================
``L``   :math:`W, A_1, C_1, L_2, A_2, C_2, Y`: dropout after each node with
        :math:`\omega_t \in [0.85, 0.97]`, a function of the history
``L0``  ``L`` with no dropout: :math:`W, A_1, L_2, A_2, Y`
======  ==============================================================================

A law reuses :class:`tests.discrete_law_longitudinal_incremental.Law`, whose joint array,
outcome table, sampler and column declaration hold for any number of levels per axis.  The
functional here generalises that module's tilt node to :math:`K_t` levels: at a tilt node
the value is :math:`\sum_a q_t(a \mid h)\, v(h, a)` with
:math:`q_t(a \mid h) = \delta_{t,a}\pi_t(a \mid h) / \sum_b \delta_{t,b}\pi_t(b \mid h)`
(``X43-plan.md`` 2.3).  The tilt reads :math:`\pi_t` off the masses it is handed, so the
complex-step derivative of :func:`functional` carries the mechanism term of every node.
:func:`eif` adds the outcome-noise coefficient as the sibling module does.

The module imports nothing from ``cleverly`` (``tests/unit/test_oracle_independence.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tests.discrete_law_longitudinal_incremental import (
    SIGMA,
    STEP,
    Axis,
    Law,
    columns,
    expit,
    sample,
)

__all__ = [
    "LAW_L",
    "LAW_L0",
    "PLANS",
    "REFERENCE",
    "columns",
    "complete_case_limit",
    "eif",
    "eif_sd",
    "functional",
    "sample",
    "truth",
]

#: The level names of every node, by code.
LABELS: tuple[str, ...] = ("a", "b", "c")
K = len(LABELS)

Node = tuple[str, tuple[float, ...]]


def tilt(factors: Mapping[str, float]) -> Node:
    """A categorical odds tilt: the factor of each named level, 1 for the others."""
    return ("tilt", tuple(float(factors.get(label, 1.0)) for label in LABELS))


#: The plans: every node tilted by the same factors.
T_A: dict[str, float] = {"b": 2.0, "c": 0.5}
T_B: dict[str, float] = {"b": 0.5, "c": 2.0}
REFERENCE = "natural course"
PLANS: dict[str, tuple[Node, ...]] = {
    REFERENCE: (tilt({}), tilt({})),
    "T_a": (tilt(T_A), tilt(T_A)),
    "T_b": (tilt(T_B), tilt(T_B)),
}


def _categorical(*logits: float) -> tuple[float, ...]:
    weights = np.exp(np.asarray(logits, dtype=float))
    return tuple(float(x) for x in weights / weights.sum())


def _bern(p: float) -> tuple[float, float]:
    return (1.0 - p, p)


def _conditional(axis: Axis, v: Mapping[str, int]) -> Sequence[float]:
    w = v.get("W", 0)
    if axis.kind == "baseline":
        return (0.5, 0.5)
    if axis.name == "A1":
        return _categorical(0.0, 0.3 - 0.6 * w, -0.2 + 0.5 * w)
    if axis.name == "L2":
        a1 = v["A1"]
        return _bern(expit(-0.5 + 1.0 * (a1 == 1) - 0.8 * (a1 == 2) + 0.3 * w))
    if axis.name == "A2":
        a1, l2 = v["A1"], v["L2"]
        return _categorical(
            0.0,
            0.2 + 0.8 * l2 + 0.6 * (a1 == 1) - 0.4 * w,
            -0.3 - 0.5 * l2 + 0.7 * (a1 == 2) + 0.4 * w,
        )
    if axis.kind == "censoring":
        t = axis.node
        a = v[f"A{t}"]
        score = 2.4 + 1.0 * (a == 1) - 1.0 * (a == 2) - 0.5 * w
        if t > 1:
            score += 1.0 * v[f"L{t}"]
        return _bern(min(max(expit(score), 0.85), 0.97))
    raise KeyError(axis.name)


def _mean(v: Mapping[str, int]) -> float:
    a1, a2, l2 = v["A1"], v["A2"], v["L2"]
    return (
        1.0
        + 1.5 * (a1 == 1)
        - 0.5 * (a1 == 2)
        + 2.0 * (a2 == 1)
        + 1.0 * (a2 == 2)
        - 1.5 * (a2 == 2) * l2
        + 1.0 * l2
        + 0.5 * v["W"]
    )


def _axes(*, censoring: bool) -> tuple[Axis, ...]:
    axes = [Axis("W", "baseline", 0), Axis("A1", "treatment", 1, levels=K)]
    if censoring:
        axes.append(Axis("C1", "censoring", 1))
    axes += [Axis("L2", "covariate", 2), Axis("A2", "treatment", 2, levels=K)]
    if censoring:
        axes.append(Axis("C2", "censoring", 2))
    return tuple(axes)


LAW_L = Law("L", _axes(censoring=True), _conditional, _mean, 2)
LAW_L0 = Law("L0", _axes(censoring=False), _conditional, _mean, 2)


# ------------------------------------------------------------------------- the functional


def functional(
    law: Law,
    probs: Any,
    plan: Sequence[Node],
    *,
    mu: Any = None,
    weights: Any = None,
    frozen_treatment: bool = False,
) -> Any:
    r"""The g-formula of ``plan`` on the law with cell masses ``probs``, dropout prevented.

    ``probs`` has the law's shape, optionally with a trailing batch axis.  ``weights`` is a
    weight per cell: the value is then read on the law :math:`w\,dP / E_P(w)`.
    ``frozen_treatment`` reads every treatment conditional from the law itself, not from
    ``probs``, so the derivative drops the mechanism terms.
    """
    p = np.asarray(probs)
    batched = p.ndim == len(law.shape) + 1
    if not batched:
        p = p[..., None]
    base = law.probs
    if weights is not None:
        cell = np.asarray(weights, dtype=float)
        p = cell[..., None] * p
        p = p / p.reshape(-1, p.shape[-1]).sum(axis=0)
        base = cell * base / (cell * base).sum()
    k = len(law.axes)
    margins = [p.sum(axis=tuple(range(j, k))) for j in range(k + 1)]
    base_margins = [base.sum(axis=tuple(range(j, k))) for j in range(k + 1)]
    mean = law.mu if mu is None else np.asarray(mu)
    if mean.ndim == len(law.shape):
        mean = mean[..., None]
    value: Any = np.broadcast_to(mean, p.shape).astype(
        complex if np.iscomplexobj(mean) or np.iscomplexobj(p) else float
    )
    for j in range(k - 1, -1, -1):
        axis = law.axes[j]
        parent = margins[j][..., None, :]
        reached = base_margins[j][..., None, None] > 0
        conditional = np.where(reached, margins[j + 1] / np.where(reached, parent, 1.0), 0.0)
        pieces = [np.take(value, level, axis=j) for level in range(axis.levels)]
        masses = [np.take(conditional, level, axis=j) for level in range(axis.levels)]
        if axis.kind in {"baseline", "covariate"}:
            value = sum(mass * piece for mass, piece in zip(masses, pieces, strict=True))
        elif axis.kind == "censoring":
            value = pieces[1]
        else:
            if frozen_treatment:
                held = base_margins[j][..., None]
                seen = held > 0
                fixed = np.where(seen, base_margins[j + 1] / np.where(seen, held, 1.0), 0.0)
                masses = [np.take(fixed[..., None], level, axis=j) for level in range(axis.levels)]
            _, factors = plan[axis.node - 1]
            normaliser = sum(f * m for f, m in zip(factors, masses, strict=True))
            value = sum(
                f * m * piece for f, m, piece in zip(factors, masses, pieces, strict=True)
            ) / np.where(normaliser == 0, 1.0, normaliser)
    return value.reshape(-1)[0] if not batched else value.reshape(p.shape[-1])


def truth(law: Law, plan: Sequence[Node], **options: Any) -> float:
    """The exact value of ``plan`` on ``law``."""
    return float(np.real(functional(law, law.probs, plan, **options)))


def _uncensored(law: Law, positive: np.ndarray) -> np.ndarray:
    keep = []
    for index in positive:
        cell = np.unravel_index(index, law.shape)
        if all(cell[i] == 1 for i, axis in enumerate(law.axes) if axis.kind == "censoring"):
            keep.append(index)
    return np.asarray(keep, dtype=int)


def eif(
    law: Law,
    plan: Sequence[Node],
    *,
    other: Sequence[Node] | None = None,
    weights: Any = None,
    frozen_treatment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The cell part of the influence function and the outcome coefficient, at each cell.

    The cell part is the complex-step derivative of :func:`functional` toward each cell with
    positive mass; the coefficient is :math:`\partial\psi/\partial\mu(o)` over :math:`p(o)`.
    With ``other``, both are of the contrast.
    """
    probs = law.probs
    flat = probs.reshape(-1)
    positive = np.flatnonzero(flat > 0)
    batch = np.zeros((flat.size, positive.size), dtype=complex)
    batch[:, :] = (1.0 - 1j * STEP) * flat[:, None]
    batch[positive, np.arange(positive.size)] += 1j * STEP
    batch = batch.reshape((*law.shape, positive.size))

    def contrast(argument: Any, **extra: Any) -> Any:
        options = {"weights": weights, "frozen_treatment": frozen_treatment, **extra}
        value = functional(law, argument, plan, **options)
        if other is not None:
            value = value - functional(law, argument, other, **options)
        return value

    cells = np.full(flat.size, np.nan)
    cells[positive] = np.imag(np.asarray(contrast(batch))) / STEP
    coefficient = np.zeros(flat.size)
    mu = law.mu.reshape(-1)
    observed = _uncensored(law, positive)
    moved = np.zeros((flat.size, observed.size), dtype=complex)
    moved[:, :] = mu[:, None]
    moved[observed, np.arange(observed.size)] += 1j * STEP
    shape = (*law.shape, observed.size)
    masses = np.broadcast_to(flat[:, None], moved.shape).reshape(shape)
    derivative = np.imag(np.asarray(contrast(masses, mu=moved.reshape(shape)))) / STEP
    coefficient[observed] = derivative / flat[observed]
    return cells.reshape(law.shape), coefficient.reshape(law.shape)


def eif_sd(law: Law, plan: Sequence[Node], **options: Any) -> float:
    r""":math:`\sqrt{\sum_o p(o) D(o)^2 + \sigma^2 \sum_o p(o) c(o)^2}` of :func:`eif`."""
    cells, coefficient = eif(law, plan, **options)
    probs = law.probs
    keep = probs > 0
    variance = float(np.sum(probs[keep] * cells[keep] ** 2))
    variance += SIGMA**2 * float(np.sum(probs[keep] * coefficient[keep] ** 2))
    return float(np.sqrt(variance))


def stay_weights(law: Law) -> np.ndarray:
    """One on the cells that never drop out, zero elsewhere."""
    out = np.zeros(law.shape)
    for cell in law.cells:
        if all(cell[i] == 1 for i, axis in enumerate(law.axes) if axis.kind == "censoring"):
            out[cell] = 1.0
    return out


def complete_case_limit(
    law: Law, plan: Sequence[Node], other: Sequence[Node] | None = None
) -> float:
    """The limit of a fit to the rows that never drop out, without a censoring model.

    The g-formula of ``plan`` on the law conditioned on staying to the end.
    """
    value = truth(law, plan, weights=stay_weights(law))
    if other is not None:
        value -= truth(law, other, weights=stay_weights(law))
    return value


def complete_case_sd(law: Law, plan: Sequence[Node], other: Sequence[Node] | None = None) -> float:
    """The influence-function SD of the complete-case fit, per complete row.

    The complete rows are a sample of the conditioned law, and the fit is efficient there, so
    its standard error at ``m`` complete rows is this over :math:`\\sqrt m`.
    """
    conditioned = law_conditioned_on_staying(law)
    return eif_sd(conditioned, plan, other=other)


def law_conditioned_on_staying(law: Law) -> Law:
    """``law`` restricted to the rows that never drop out, with the censoring axes held at 1.

    Returned as a law on the uncensored axes, so its :func:`eif_sd` is the efficient bound
    of a complete-case fit.
    """
    keep = [i for i, axis in enumerate(law.axes) if axis.kind != "censoring"]
    probs = law.probs
    index = tuple(slice(None) if i in keep else 1 for i in range(len(law.axes)))
    conditioned = probs[index] / probs[index].sum()
    mean = law.mu[index]
    axes = tuple(law.axes[i] for i in keep)
    out = Law(f"{law.name}|stay", axes, law.conditional, law.mean, law.n_times)
    out.__dict__["probs"] = conditioned
    out.__dict__["mu"] = np.where(conditioned > 0, mean, 0.0)
    return out


def retention(law: Law) -> float:
    """The probability of never dropping out."""
    return float((stay_weights(law) * law.probs).sum())
