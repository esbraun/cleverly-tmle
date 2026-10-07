r"""A point-treatment survival process with an exact truth.

One baseline treatment :math:`A`, two baseline covariates and a discrete-time event at
:math:`K` grid times.  The process has the visit structure of Benkeser, Carone and Gilbert
(2018, Section 2.1): a unit can drop out at a visit, after the visit confirms it event-free,
and the event of the next interval is then unobserved.

======================  =================================================================
node                    law
======================  =================================================================
:math:`W_1`             Bernoulli(0.5)
:math:`W_2`             uniform on :math:`\{0, 1, 2, 3\}`
:math:`A`               two arms, three arms, or a dose on :math:`\{0, \ldots, 5\}`
:math:`L_k`, k >= 2     binary, from :math:`(W_1, A, L_{k-1})`, on the wide layout only
dropout at node k       :math:`k \ge 2`, probability at most 0.12, from :math:`(A, W, L_k)`
event at node k         hazard from :math:`(k, A, W, L_k)`; two causes split it by
                        :math:`(A, W_1)`
======================  =================================================================

No unit drops out at node 1, so the first censoring column is all ones.  Every truth is an
exact finite sum over :math:`(W_1, W_2)` and the paths of :math:`L`.  With
``continuous_event_time=True`` the event time is exponential given :math:`(A, W)`, the
dropout happens at the grid times, and the risk at a grid time has a closed form.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .._typing import Backend, FloatArray
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = ["make_point_survival", "point_survival_truth"]

#: What separates a regimen from the horizon it is reported at, and a regimen from its
#: cause, mirrored from :mod:`cleverly.longitudinal.estimator` rather than imported.
_HORIZON_INFIX = " @ t="
_CAUSE_INFIX = ", "

#: The labels of the two causes on a competing-risks draw, by event code.
CAUSES: dict[int, str] = {1: "relapse", 2: "death"}

#: The dose levels of ``dose="discrete"``.
DOSES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

_W2 = (0, 1, 2, 3)


def _arm_effect(arm: Any, arms: int, dose: str | None) -> Any:
    """The treatment's term in the log-odds of the all-cause hazard."""
    a = np.asarray(arm, dtype=float)
    if dose is not None:
        return -0.12 * a
    if arms == 3:
        return np.where(a == 1.0, -0.45, np.where(a == 2.0, -0.2, 0.0))
    return -0.45 * a


def _treated(arm: Any, dose: str | None) -> Any:
    """``1`` for a treated arm, the dose over five on a dose."""
    a = np.asarray(arm, dtype=float)
    return a / 5.0 if dose is not None else np.minimum(a, 1.0)


def _hazard(node: int, arm: Any, w1: Any, w2: Any, l: Any, arms: int, dose: str | None) -> Any:
    """``P(event at node k | at risk, A, W, L_k)``, all causes."""
    return expit(
        -2.3
        + 0.08 * (node - 1)
        + _arm_effect(arm, arms, dose)
        + 0.35 * np.asarray(w1, dtype=float)
        + 0.15 * np.asarray(w2, dtype=float)
        + 0.5 * np.asarray(l, dtype=float)
    )


def _first_cause(arm: Any, w1: Any, dose: str | None) -> Any:
    """``P(cause 1 | event, A, W)``.  Treatment shifts the events towards cause 1."""
    return expit(0.1 + 0.8 * _treated(arm, dose) - 0.3 * np.asarray(w1, dtype=float))


def _dropout(arm: Any, w1: Any, w2: Any, l: Any, dose: str | None) -> Any:
    """``P(drop out at node k | at risk entering k, A, W, L_k)``, for ``k >= 2``."""
    return (
        0.03
        + 0.02 * _treated(arm, dose)
        + 0.02 * np.asarray(w1, dtype=float)
        + 0.01 * np.asarray(w2, dtype=float)
        + 0.02 * np.asarray(l, dtype=float)
    )


def _covariate(w1: Any, arm: Any, previous: Any, dose: str | None) -> Any:
    """``P(L_k = 1 | W_1, A, L_{k-1})``."""
    return expit(
        -0.5
        + 0.6 * np.asarray(w1, dtype=float)
        - 0.5 * _treated(arm, dose)
        + 0.8 * np.asarray(previous, dtype=float)
    )


def _rate(arm: Any, w1: Any, w2: Any, arms: int, dose: str | None) -> Any:
    """The exponential event rate of ``continuous_event_time=True``."""
    return np.exp(
        -1.6
        + _arm_effect(arm, arms, dose)
        + 0.35 * np.asarray(w1, dtype=float)
        + 0.15 * np.asarray(w2, dtype=float)
    )


def _end_mean(arm: Any, w1: Any, w2: Any, arms: int, dose: str | None) -> Any:
    """``E[Y | A, W]`` of the end-of-study outcome."""
    return expit(
        -0.4
        - 1.3 * _arm_effect(arm, arms, dose)
        - 0.3 * np.asarray(w1, dtype=float)
        + 0.2 * np.asarray(w2, dtype=float)
    )


def treatment_probabilities(w1: Any, w2: Any, *, arms: int = 2, dose: str | None = None) -> Any:
    """``P(A = a | W)``, an ``(n, levels)`` array in level order.

    Parameters
    ----------
    w1, w2 : array-like
        The baseline covariates.
    arms : {2, 3}
        The number of arms, when ``dose`` is ``None``.
    dose : {"discrete"} or None
        ``"discrete"`` for the dose on :data:`DOSES`.

    Returns
    -------
    ndarray
        One column per level, every row summing to one.
    """
    x1 = np.atleast_1d(np.asarray(w1, dtype=float))
    x2 = np.atleast_1d(np.asarray(w2, dtype=float))
    if dose is not None:
        centre = 2.0 + 0.5 * x1 + 0.25 * x2
        levels = np.asarray(DOSES, dtype=float)
        weight = np.exp(-((levels[None, :] - centre[:, None]) ** 2) / (2.0 * 1.5**2))
        return weight / weight.sum(axis=1, keepdims=True)
    if arms == 3:
        p1 = 0.25 + 0.1 * x1
        p2 = 0.25 + 0.04 * x2
        return np.column_stack([1.0 - p1 - p2, p1, p2])
    p = 0.3 + 0.15 * x1 + 0.075 * x2
    return np.column_stack([1.0 - p, p])


def _one_hot(level: int, arms: int) -> Callable[[int, int], Sequence[float]]:
    """The intervention that sets ``A = level`` for everybody."""
    row = tuple(1.0 if code == level else 0.0 for code in range(arms))
    return lambda w1, w2: row


def _levels(arms: int, dose: str | None) -> tuple[int, ...]:
    return DOSES if dose is not None else tuple(range(arms))


def point_survival_truth(
    assign: Callable[[int, int], Sequence[float]],
    *,
    arms: int = 2,
    causes: int = 1,
    n_times: int = 5,
    time_varying: bool = False,
    dose: str | None = None,
    continuous_event_time: bool = False,
    grid: Sequence[float] | None = None,
    end_of_study: bool = False,
    weight: Callable[[int, int], float] | None = None,
) -> FloatArray:
    r"""The exact truth of :func:`make_point_survival` under an intervention on :math:`A`.

    Parameters
    ----------
    assign : callable
        ``assign(w1, w2)`` returns the intervention's probability of each treatment level,
        in level order.  A static arm is one-hot, a rule is one-hot by covariate, a
        stochastic policy is a density, and a modified treatment policy is the law of the
        shifted treatment.
    arms, causes, n_times, time_varying, dose, continuous_event_time, grid, end_of_study
        As for :func:`make_point_survival`.
    weight : callable or None
        ``weight(w1, w2)``, an observation weight of the baseline stratum.  The truth is then
        the parameter of the tilted law :math:`dP_w = w\,dP / E[w]`.  ``None`` is no tilt.

    Returns
    -------
    ndarray
        ``(n_times, causes)``: the cumulative risk, or the incidence of each cause, at each
        node.  On an end-of-study draw, a one-element array holding the mean outcome.
    """
    levels = _levels(arms, dose)
    total = np.zeros((n_times, causes)) if not end_of_study else np.zeros(1)
    times = _grid(grid, n_times, continuous_event_time)
    tilt = (
        1.0
        if weight is None
        else sum(0.5 * 0.25 * float(weight(w1, w2)) for w1 in (0, 1) for w2 in _W2)
    )
    for w1 in (0, 1):
        for w2 in _W2:
            mass = 0.5 * 0.25 * (1.0 if weight is None else float(weight(w1, w2))) / tilt
            probability = np.asarray(assign(w1, w2), dtype=float)
            for code, level in enumerate(levels):
                share = float(probability[code])
                if share == 0.0:
                    continue
                if end_of_study:
                    total[0] += mass * share * float(_end_mean(level, w1, w2, arms, dose))
                    continue
                if continuous_event_time:
                    rate = float(_rate(level, w1, w2, arms, dose))
                    curve = 1.0 - np.exp(-rate * np.asarray(times))
                    total[:, 0] += mass * share * curve
                    continue
                total += (
                    mass
                    * share
                    * _discrete_curve(level, w1, w2, arms, causes, n_times, time_varying, dose)
                )
    return total


def _discrete_curve(
    level: int,
    w1: int,
    w2: int,
    arms: int,
    causes: int,
    n_times: int,
    time_varying: bool,
    dose: str | None,
) -> FloatArray:
    """``(n_times, causes)`` incidences at one stratum and one assigned level."""
    out = np.zeros((n_times, causes))
    # Mass of the event-free units entering each node, by the current value of L.
    alive = {0: 1.0}
    running = np.zeros(causes)
    for node in range(1, n_times + 1):
        if time_varying and node > 1:
            moved: dict[int, float] = {0: 0.0, 1: 0.0}
            for previous, mass in alive.items():
                p = float(_covariate(w1, level, previous, dose))
                moved[1] += mass * p
                moved[0] += mass * (1.0 - p)
            alive = moved
        survivors: dict[int, float] = {}
        for value, mass in alive.items():
            h = float(_hazard(node, level, w1, w2, value, arms, dose))
            share = float(_first_cause(level, w1, dose))
            split = np.array([share, 1.0 - share]) if causes == 2 else np.array([1.0])
            running = running + mass * h * split
            survivors[value] = mass * (1.0 - h)
        alive = survivors
        out[node - 1] = running
    return out


def _grid(grid: Sequence[float] | None, n_times: int, continuous: bool) -> tuple[float, ...]:
    if grid is not None:
        return tuple(float(g) for g in grid)
    if continuous:
        return (
            (0.5, 1.0, 2.0, 3.0)[:n_times]
            if n_times <= 4
            else tuple(float(k) / 2.0 for k in range(1, n_times + 1))
        )
    return tuple(float(k) for k in range(1, n_times + 1))


def make_point_survival(
    n: int = 2000,
    *,
    seed: int | np.random.Generator | None = None,
    arms: int = 2,
    causes: int = 1,
    n_times: int = 5,
    censoring: bool = True,
    time_varying: bool = False,
    continuous_event_time: bool = False,
    dose: str | None = None,
    grid: Sequence[float] | None = None,
    end_of_study: bool = False,
    layout: str = "long",
    cluster_size: int | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A baseline treatment, a discrete-time event, and dropout at the visits.

    The long layout has one row per unit with columns ``W1``, ``W2``, ``A`` (or ``D`` for a
    dose), ``time`` and ``event``.  ``event`` is ``0`` for censored and the cause code
    otherwise.  A unit that drops out at node :math:`k` is censored at the grid time
    :math:`g_{k-1}`, and a unit event-free at the end is censored at :math:`g_K`.  The wide
    layout has ``C1..CK`` and ``Y1..YK`` (``R1..RK`` and ``D1..DK`` for two causes), and
    ``L2..LK`` with ``time_varying=True``.  ``C1`` is all ones.

    Parameters
    ----------
    n : int
        Number of units.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    arms : {2, 3}
        The number of arms, when ``dose`` is ``None``.
    causes : {1, 2}
        The number of absorbing causes.  Two causes are ``relapse`` (code 1) and
        ``death`` (code 2).
    n_times : int
        The number of nodes :math:`K`.
    censoring : bool
        Whether units drop out at the visits.
    time_varying : bool
        Whether a binary covariate :math:`L_k` enters the hazard at nodes ``k >= 2``.  Wide
        layout only.
    continuous_event_time : bool
        Whether the event time is exponential given :math:`(A, W)`.  One cause, long layout
        only.  The default grid is ``0.5, 1, 2, 3`` for four nodes.
    dose : {"discrete"} or None
        ``"discrete"`` draws a dose on ``0..5`` in place of an arm.
    grid : sequence of float or None
        The grid times of the long layout.  ``None`` uses ``1..K``.
    end_of_study : bool
        Whether the outcome is one binary ``Y`` after ``n_times`` censoring nodes.  Wide
        layout only; every censoring node then has dropout.
    layout : {"long", "wide"}
        The frame layout.
    cluster_size : int or None
        Rows per cluster.  ``W1`` and ``W2`` are then drawn once per cluster and shared by its rows,
        which correlates the rows of a cluster and leaves the law of one row unchanged.  The
        frame gains an ``id`` column.  ``None`` leaves the rows independent.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend.

    Returns
    -------
    dataframe
        The observations.
    truth : dict of str to float
        The risk (``risk_regimen``), the incidence (``cif_regimen``) or the mean
        (``ey_regimen``) of every static arm ``arm0``, ``arm1`` (and ``arm2``) at every
        node, and each arm's contrast with ``arm0``, under the names a fit reports for the
        regimens ``{"arm0": 0, "arm1": 1, ...}``.  Empty on a dose draw.

    Examples
    --------
    >>> from cleverly.datasets import make_point_survival
    >>> frame, truth = make_point_survival(n=200, seed=0, n_times=3)
    >>> list(frame.columns)
    ['W1', 'W2', 'A', 'time', 'event']
    >>> round(truth["risk_regimen[arm1 @ t=3]"], 4)
    0.2596
    """
    if layout not in ("long", "wide"):
        raise ValueError(f"layout must be 'long' or 'wide'; got {layout!r}")
    if (time_varying or end_of_study) and layout != "wide":
        raise ValueError("time_varying=True and end_of_study=True need layout='wide'")
    if continuous_event_time and (layout != "long" or causes != 1 or time_varying):
        raise ValueError("continuous_event_time=True needs layout='long', one cause, no L")
    rng = np.random.default_rng(seed)
    if cluster_size is None:
        w1 = rng.binomial(1, 0.5, n).astype(float)
        w2 = rng.integers(0, 4, n).astype(float)
        ids = None
    else:
        ids = np.arange(n) // cluster_size
        clusters = int(ids[-1]) + 1
        w1 = rng.binomial(1, 0.5, clusters).astype(float)[ids]
        w2 = rng.integers(0, 4, clusters).astype(float)[ids]
    probabilities = treatment_probabilities(w1, w2, arms=arms, dose=dose)
    cumulative = np.cumsum(probabilities, axis=1)
    draw = rng.uniform(size=n)
    code = np.minimum((draw[:, None] > cumulative).sum(axis=1), probabilities.shape[1] - 1)
    levels = np.asarray(_levels(arms, dose), dtype=float)
    a = levels[code]
    treatment_name = "D" if dose is not None else "A"
    payload: dict[str, Any] = {"W1": w1, "W2": w2, treatment_name: a}
    if ids is not None:
        payload["id"] = ids.astype(float)
    times = _grid(grid, n_times, continuous_event_time)

    if end_of_study:
        _end_of_study(rng, payload, a, w1, w2, n_times, censoring, arms, dose)
    elif continuous_event_time:
        _continuous(rng, payload, a, w1, w2, times, censoring, arms, dose)
    else:
        _discrete(
            rng,
            payload,
            a,
            w1,
            w2,
            times,
            n_times,
            censoring,
            time_varying,
            causes,
            arms,
            dose,
            layout,
        )

    truth: dict[str, float] = {}
    if dose is None:
        labels = [f"arm{level}" for level in range(arms)]
        curves = {
            label: point_survival_truth(
                _one_hot(level, arms),
                arms=arms,
                causes=causes,
                n_times=n_times,
                time_varying=time_varying,
                continuous_event_time=continuous_event_time,
                grid=grid,
                end_of_study=end_of_study,
            )
            for level, label in enumerate(labels)
        }
        for label in labels:
            if end_of_study:
                truth[f"ey_regimen[{label}]"] = float(curves[label][0])
                if label != labels[0]:
                    truth[f"ate_regimen[{label} vs {labels[0]}]"] = float(
                        curves[label][0] - curves[labels[0]][0]
                    )
                continue
            for node in range(1, n_times + 1):
                for cause in range(causes):
                    stem = (
                        f"risk_regimen[{label}"
                        if causes == 1
                        else f"cif_regimen[{label}{_CAUSE_INFIX}{CAUSES[cause + 1]}"
                    )
                    truth[f"{stem}{_HORIZON_INFIX}{node}]"] = float(curves[label][node - 1, cause])
                    if label != labels[0]:
                        inside = (
                            f"{label} vs {labels[0]}"
                            if causes == 1
                            else f"{label} vs {labels[0]}{_CAUSE_INFIX}{CAUSES[cause + 1]}"
                        )
                        truth[f"ate_regimen[{inside}{_HORIZON_INFIX}{node}]"] = float(
                            curves[label][node - 1, cause] - curves[labels[0]][node - 1, cause]
                        )
    return frame_from_dict(payload, backend=backend), truth


def _discrete(
    rng: np.random.Generator,
    payload: dict[str, Any],
    a: FloatArray,
    w1: FloatArray,
    w2: FloatArray,
    times: tuple[float, ...],
    n_times: int,
    censoring: bool,
    time_varying: bool,
    causes: int,
    arms: int,
    dose: str | None,
    layout: str,
) -> None:
    n = a.shape[0]
    in_study = np.ones(n, dtype=bool)
    previous = np.zeros(n)
    observed_time = np.full(n, times[-1])
    event_code = np.zeros(n, dtype=np.int64)
    columns: dict[str, FloatArray] = {}
    for node in range(1, n_times + 1):
        if time_varying and node > 1:
            l_value = rng.binomial(1, _covariate(w1, a, previous, dose)).astype(float)
            columns[f"L{node}"] = np.where(in_study, l_value, np.nan)
            previous = l_value
        current = previous if time_varying else np.zeros(n)
        if censoring and node > 1:
            drop = rng.uniform(size=n) < _dropout(a, w1, w2, current, dose)
        else:
            drop = np.zeros(n, dtype=bool)
        columns[f"C{node}"] = np.where(in_study, (~drop).astype(float), np.nan)
        leaving = in_study & drop
        observed_time[leaving] = times[node - 2] if node > 1 else 0.0
        stays = in_study & ~drop
        fired = rng.uniform(size=n) < _hazard(node, a, w1, w2, current, arms, dose)
        first = rng.uniform(size=n) < _first_cause(a, w1, dose)
        cause = np.where(first, 1, 2) if causes == 2 else np.ones(n, dtype=np.int64)
        happened = stays & fired
        if causes == 2:
            columns[f"R{node}"] = np.where(stays, (happened & (cause == 1)).astype(float), np.nan)
            columns[f"D{node}"] = np.where(stays, (happened & (cause == 2)).astype(float), np.nan)
        else:
            columns[f"Y{node}"] = np.where(stays, happened.astype(float), np.nan)
        observed_time[happened] = times[node - 1]
        event_code[happened] = cause[happened]
        in_study = stays & ~fired
    if layout == "wide":
        order = []
        for node in range(1, n_times + 1):
            if time_varying and node > 1:
                order.append(f"L{node}")
            order.append(f"C{node}")
            order.extend([f"R{node}", f"D{node}"] if causes == 2 else [f"Y{node}"])
        if not censoring:
            order = [name for name in order if not name.startswith("C")]
        for name in order:
            payload[name] = columns[name]
        return
    payload["time"] = observed_time
    payload["event"] = event_code


def _continuous(
    rng: np.random.Generator,
    payload: dict[str, Any],
    a: FloatArray,
    w1: FloatArray,
    w2: FloatArray,
    times: tuple[float, ...],
    censoring: bool,
    arms: int,
    dose: str | None,
) -> None:
    n = a.shape[0]
    event_time = rng.exponential(1.0 / _rate(a, w1, w2, arms, dose))
    observed = np.minimum(event_time, times[-1])
    code = (event_time <= times[-1]).astype(np.int64)
    if censoring:
        dropped = np.zeros(n, dtype=bool)
        zero = np.zeros(n)
        for visit in times[:-1]:
            # A unit seen event-free at this visit may drop out after it.
            seen = ~dropped & (event_time > visit)
            leave = seen & (rng.uniform(size=n) < _dropout(a, w1, w2, zero, dose))
            observed = np.where(leave, visit, observed)
            code = np.where(leave, 0, code)
            dropped |= leave
    payload["time"] = observed
    payload["event"] = code


def _end_of_study(
    rng: np.random.Generator,
    payload: dict[str, Any],
    a: FloatArray,
    w1: FloatArray,
    w2: FloatArray,
    n_times: int,
    censoring: bool,
    arms: int,
    dose: str | None,
) -> None:
    n = a.shape[0]
    in_study = np.ones(n, dtype=bool)
    zero = np.zeros(n)
    for node in range(1, n_times + 1):
        if censoring:
            stay = rng.uniform(size=n) >= _dropout(a, w1, w2, zero, dose)
            payload[f"C{node}"] = np.where(in_study, stay.astype(float), np.nan)
            in_study = in_study & stay
    y = rng.binomial(1, _end_mean(a, w1, w2, arms, dose)).astype(float)
    payload["Y"] = np.where(in_study, y, np.nan)
