r"""The laws, truths, learners and efficiency bounds the point-treatment survival studies share.

The three studies ``point-treatment-survival``, ``point-treatment-survival-crossfit`` and
``point-treatment-survival-policies`` draw from :func:`cleverly.datasets.make_point_survival`.
Every scenario has a finite support, so each truth is an exact sum
(:func:`cleverly.datasets.point_survival_truth`) and each efficiency bound is the exact
standard deviation of the efficient influence function, summed over the support by
:func:`efficiency_sd`.  The bound shares no code with the estimator: it enumerates the
histories of the generating law and evaluates

.. math::

    D = \sum_{t \le k} \frac{\rho(A, W)}{\prod_{s \le t} c_s} 1\{\text{followed at } t\}
        (Z_t - Q_t) + \Phi(A, W) - \psi,

with :math:`\rho = \pi(A \mid W) / g(A \mid W)` and :math:`\Phi = \sum_a \pi(a \mid W) Q_1(a, W)`
for a static arm or a known policy, and :math:`\rho = g^m(A \mid W) / g(A \mid W)` and
:math:`\Phi = Q_1(m(A), W)` for a modified treatment policy :math:`m` (Díaz, Williams, Hoffman
and Schenck 2023, Theorem 1 at one decision).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from cleverly.datasets import make_point_survival, point_survival_truth
from cleverly.datasets import survival_point as law_module
from cleverly.longitudinal import LongitudinalData

#: The event codes of the competing scenarios, by cause label.
CAUSES = {1: "relapse", 2: "death"}


@dataclass(frozen=True)
class Scenario:
    """One law of the family, and how a fit reads it.

    Parameters
    ----------
    name : str
        The scenario label.
    n_times : int
        The node count.
    arms : int
        The arm count, when there is no dose.
    causes : int
        The cause count.
    time_varying : bool
        Whether ``L_k`` enters, on the wide layout.
    continuous_event_time : bool
        Whether the event time is exponential.
    dose : str or None
        ``"discrete"`` for the dose on ``0..5``.
    end_of_study : bool
        Whether the outcome is one ``Y`` after the censoring nodes.
    grid : tuple of float
        The grid the long layout is declared on.
    wide : bool
        Whether to draw the wide layout, with one censoring and one event column per node.
    """

    name: str
    n_times: int
    arms: int = 2
    causes: int = 1
    time_varying: bool = False
    continuous_event_time: bool = False
    dose: str | None = None
    end_of_study: bool = False
    grid: tuple[float, ...] = ()
    wide: bool = False

    @property
    def layout(self) -> str:
        return "wide" if self.wide or self.time_varying or self.end_of_study else "long"

    @property
    def treatment(self) -> str:
        return "D" if self.dose is not None else "A"

    @property
    def levels(self) -> tuple[int, ...]:
        return law_module.DOSES if self.dose is not None else tuple(range(self.arms))

    def law_keywords(self) -> dict[str, Any]:
        return {
            "n_times": self.n_times,
            "arms": self.arms,
            "causes": self.causes,
            "time_varying": self.time_varying,
            "continuous_event_time": self.continuous_event_time,
            "dose": self.dose,
            "end_of_study": self.end_of_study,
        }

    def draw(
        self,
        n: int,
        seed: int,
        *,
        cluster_size: int | None = None,
    ) -> pd.DataFrame:
        frame, _ = make_point_survival(
            n=n,
            seed=seed,
            layout=self.layout,
            grid=self.grid if self.continuous_event_time else None,
            cluster_size=cluster_size,
            backend="pandas",
            **self.law_keywords(),
        )
        return frame

    def container(
        self, frame: pd.DataFrame, *, weights: str | None = None, id: str | None = None
    ) -> LongitudinalData:
        """The fit's container: the long converter, or the wide held layout."""
        if self.layout == "long":
            return LongitudinalData.from_time_to_event(
                frame,
                time="time",
                event="event",
                treatment=self.treatment,
                baseline=["W1", "W2"],
                grid=self.grid,
                causes=CAUSES if self.causes == 2 else None,
                weights=weights,
                id=id,
            )
        nodes = range(1, self.n_times + 1)
        if self.end_of_study:
            outcome: Any = "Y"
        elif self.causes == 2:
            outcome = {"relapse": [f"R{k}" for k in nodes], "death": [f"D{k}" for k in nodes]}
        else:
            outcome = [f"Y{k}" for k in nodes]
        return LongitudinalData.from_frame(
            frame,
            outcome=outcome,
            treatment=self.treatment,
            baseline=["W1", "W2"],
            time_varying=(
                [[f"L{k}"] if k > 1 else [] for k in nodes] if self.time_varying else None
            ),
            censoring=[f"C{k}" for k in nodes],
            weights=weights,
            id=id,
        )

    def truth_curve(
        self, assign: Callable[[int, int], Sequence[float]], *, weight: Any = None
    ) -> np.ndarray:
        return point_survival_truth(
            assign,
            grid=self.grid if self.continuous_event_time else None,
            weight=weight,
            **self.law_keywords(),
        )


SURVIVAL = Scenario("survival", 5, grid=(1.0, 2.0, 3.0, 4.0, 5.0))
COMPETING = Scenario("competing", 4, causes=2, grid=(1.0, 2.0, 3.0, 4.0))
THREE_ARM = Scenario("three_arm", 3, arms=3, time_varying=True)
CONTINUOUS = Scenario("continuous", 4, continuous_event_time=True, grid=(0.5, 1.0, 2.0, 3.0))
END_OF_STUDY = Scenario("end_of_study", 2, end_of_study=True)
DOSE = Scenario("dose", 5, dose="discrete", grid=(1.0, 2.0, 3.0, 4.0, 5.0))


# ------------------------------------------------------------------ interventions


def static(level: int, levels: Sequence[int]) -> Callable[[int, int], tuple[float, ...]]:
    """The intervention that sets the treatment to ``level``."""
    return lambda w1, w2: tuple(1.0 if value == level else 0.0 for value in levels)


def policy_probability(w1: Any, w2: Any) -> Any:
    """The known stochastic policy of ``point-treatment-survival-policies``: ``q(1 | W)``.

    ``0.5 + 0.25 W1 + 0.25 1{W2 >= 2}``, a multiple of one quarter, so ``lmtp`` realises it
    exactly with four copies of each unit.  It treats more often than the natural course does.
    """
    return 0.5 + 0.25 * np.asarray(w1, dtype=float) + 0.25 * (np.asarray(w2, dtype=float) >= 2.0)


def known_policy(w1: int, w2: int) -> tuple[float, float]:
    q = float(policy_probability(w1, w2))
    return (1.0 - q, q)


def minus_one(level: Any) -> Any:
    """``lmtp``'s Example 2.1 policy on an ordered dose: ``a - 1`` where ``a - 1 >= 1``."""
    a = np.asarray(level, dtype=float)
    return np.where(a - 1.0 >= 1.0, a - 1.0, a)


def shifted(
    mapping: Callable[[Any], Any],
    levels: Sequence[int],
    *,
    arms: int = 2,
    dose: str | None = "discrete",
) -> Callable[[int, int], tuple[float, ...]]:
    """The law of the shifted treatment ``mapping(A)`` given ``W``, on the dose by default."""

    def assign(w1: int, w2: int) -> tuple[float, ...]:
        g = law_module.treatment_probabilities(w1, w2, arms=arms, dose=dose)[0]
        out = np.zeros(len(levels))
        for code, level in enumerate(levels):
            out[list(levels).index(int(mapping(level)))] += g[code]
        return tuple(float(value) for value in out)

    return assign


# ------------------------------------------------------------------ the efficiency bound


@dataclass(frozen=True)
class Target:
    """One term of a linear target: ``coefficient * F_cause(horizon)`` under an intervention.

    Parameters
    ----------
    coefficient : float
        The term's weight.
    kind : {"static", "policy", "mtp"}
        The intervention kind.
    value : Any
        The arm, the policy ``assign(w1, w2)``, or the level map.
    horizon : int
        The node; ignored on an end-of-study law.
    cause : int
        The cause code, one-based.
    """

    coefficient: float
    kind: str
    value: Any
    horizon: int
    cause: int = 1


def _hazard(scenario: Scenario, node: int, a: int, w1: int, w2: int, l: int) -> float:
    if scenario.continuous_event_time:
        grid = (0.0, *scenario.grid)
        rate = float(law_module._rate(a, w1, w2, scenario.arms, scenario.dose))
        return float(1.0 - np.exp(-rate * (grid[node] - grid[node - 1])))
    return float(law_module._hazard(node, a, w1, w2, l, scenario.arms, scenario.dose))


def _split(scenario: Scenario, a: int, w1: int) -> tuple[float, ...]:
    if scenario.causes == 2:
        share = float(law_module._first_cause(a, w1, scenario.dose))
        return (share, 1.0 - share)
    return (1.0,)


def _retained(scenario: Scenario, node: int, a: int, w1: int, w2: int, l: int) -> float:
    if node == 1 and not scenario.end_of_study:
        return 1.0
    return 1.0 - float(law_module._dropout(a, w1, w2, l, scenario.dose))


def _covariate(scenario: Scenario, w1: int, a: int, previous: int) -> float:
    return float(law_module._covariate(w1, a, previous, scenario.dose))


def _regression(
    scenario: Scenario, a: int, w1: int, w2: int, horizon: int, cause: int
) -> dict[tuple[int, int], float]:
    """``Q_t(a, w, l_t)`` for every node ``t <= horizon`` and value of ``L_t``."""
    out: dict[tuple[int, int], float] = {}
    last = scenario.n_times if scenario.end_of_study else horizon
    for node in range(last, 0, -1):
        for l in (0, 1):
            if scenario.end_of_study:
                if node == last:
                    value = float(law_module._end_mean(a, w1, w2, scenario.arms, scenario.dose))
                else:
                    value = out[(node + 1, 0)]
                out[(node, l)] = value
                continue
            h = _hazard(scenario, node, a, w1, w2, l)
            fired = h * _split(scenario, a, w1)[cause - 1]
            if node == last:
                out[(node, l)] = fired
                continue
            if scenario.time_varying:
                p = _covariate(scenario, w1, a, l)
                later = p * out[(node + 1, 1)] + (1.0 - p) * out[(node + 1, 0)]
            else:
                later = out[(node + 1, 0)]
            out[(node, l)] = fired + (1.0 - h) * later
    return out


def _paths(scenario: Scenario, a: int, w1: int, w2: int) -> list[tuple[float, list[Any]]]:
    """Every history after ``(W, A)``: ``(probability, [(l, retained, event), ...], y)``."""
    out: list[tuple[float, list[Any]]] = []

    def walk(node: int, mass: float, previous: int, path: list[Any]) -> None:
        if node > scenario.n_times:
            if scenario.end_of_study:
                m = float(law_module._end_mean(a, w1, w2, scenario.arms, scenario.dose))
                out.append((mass * m, [*path, ("y", 1)]))
                out.append((mass * (1.0 - m), [*path, ("y", 0)]))
            else:
                out.append((mass, path))
            return
        branches = (
            (
                (1, _covariate(scenario, w1, a, previous)),
                (0, 1 - _covariate(scenario, w1, a, previous)),
            )
            if scenario.time_varying and node > 1
            else ((previous if scenario.time_varying else 0, 1.0),)
        )
        for l, share in branches:
            c = _retained(scenario, node, a, w1, w2, l)
            if c < 1.0:
                out.append((mass * share * (1.0 - c), [*path, (l, 0, 0)]))
            if scenario.end_of_study:
                walk(node + 1, mass * share * c, l, [*path, (l, 1, 0)])
                continue
            h = _hazard(scenario, node, a, w1, w2, l)
            for cause, split in enumerate(_split(scenario, a, w1), start=1):
                out.append((mass * share * c * h * split, [*path, (l, 1, cause)]))
            walk(node + 1, mass * share * c * (1.0 - h), l, [*path, (l, 1, 0)])

    walk(1, 1.0, 0, [])
    return out


def _intervention(
    scenario: Scenario, target: Target, w1: int, w2: int
) -> tuple[np.ndarray, Callable[[int], int] | None]:
    """The intervention's arm law given ``W``, and the level map of an MTP."""
    levels = scenario.levels
    if target.kind == "static":
        return np.array(static(int(target.value), levels)(w1, w2)), None
    if target.kind == "policy":
        return np.array(target.value(w1, w2)), None
    mapping = target.value
    law = shifted(mapping, levels, arms=scenario.arms, dose=scenario.dose)
    return np.array(law(w1, w2)), lambda a: int(mapping(a))


def efficiency_sd(
    scenario: Scenario, targets: Sequence[Target], *, weight: Any = None, between: bool = False
) -> float:
    r"""The standard deviation of the efficient influence function of ``sum(targets)``.

    ``weight(w1, w2)`` tilts the law by an observation weight; the curve is then the
    weighted one, :math:`w D^*(P_w)` with :math:`w` normalised to mean one.  ``between=True``
    returns the standard deviation of :math:`E[D \mid W, A]` instead, the part a cluster that
    shares its ``W`` and ``A`` cannot average away.
    """
    levels = scenario.levels
    strata = [(w1, w2) for w1 in (0, 1) for w2 in (0, 1, 2, 3)]
    base = dict.fromkeys(strata, 0.125)
    if weight is not None:
        total = sum(0.125 * float(weight(*stratum)) for stratum in strata)
        tilted = {stratum: 0.125 * float(weight(*stratum)) / total for stratum in strata}
        scale = {stratum: float(weight(*stratum)) / total for stratum in strata}
    else:
        tilted = base
        scale = dict.fromkeys(strata, 1.0)
    # psi of each target, under the (tilted) law
    psi = []
    for target in targets:
        value = 0.0
        for stratum in strata:
            pi, mapping = _intervention(scenario, target, *stratum)
            g = law_module.treatment_probabilities(
                *stratum, arms=scenario.arms, dose=scenario.dose
            )[0]
            q = {
                a: _regression(scenario, a, *stratum, target.horizon, target.cause)[(1, 0)]
                for a in levels
            }
            if mapping is None:
                inner = sum(pi[i] * q[a] for i, a in enumerate(levels))
            else:
                inner = sum(g[i] * q[mapping(a)] for i, a in enumerate(levels))
            value += tilted[stratum] * inner
        psi.append(value)
    second = 0.0
    first = 0.0
    conditional: dict[tuple[tuple[int, int], int], float] = {}
    for stratum in strata:
        w1, w2 = stratum
        g = law_module.treatment_probabilities(w1, w2, arms=scenario.arms, dose=scenario.dose)[0]
        for code, a in enumerate(levels):
            for mass, path in _paths(scenario, a, w1, w2):
                curve = 0.0
                for target, value in zip(targets, psi, strict=True):
                    curve += target.coefficient * _curve(
                        scenario, target, value, stratum, code, a, g, path
                    )
                # The tilted law's curve, w D*(P_w), with w normalised to mean one under P.
                probability = base[stratum] * g[code] * mass
                contribution = scale[stratum] * curve
                first += probability * contribution
                second += probability * contribution**2
                key = (stratum, code)
                conditional[key] = conditional.get(key, 0.0) + mass * contribution
    if abs(first) > 1e-9:  # pragma: no cover - guards the enumeration
        raise AssertionError(f"the influence function has mean {first}, not zero")
    if between:
        g_of = {
            stratum: law_module.treatment_probabilities(
                *stratum, arms=scenario.arms, dose=scenario.dose
            )[0]
            for stratum in strata
        }
        return float(
            np.sqrt(
                sum(
                    base[stratum] * g_of[stratum][code] * value**2
                    for (stratum, code), value in conditional.items()
                )
            )
        )
    return float(np.sqrt(second))


def _curve(
    scenario: Scenario,
    target: Target,
    psi: float,
    stratum: tuple[int, int],
    code: int,
    a: int,
    g: np.ndarray,
    path: list[Any],
) -> float:
    levels = scenario.levels
    w1, w2 = stratum
    pi, mapping = _intervention(scenario, target, w1, w2)
    q_observed = _regression(scenario, a, w1, w2, target.horizon, target.cause)
    if mapping is None:
        ratio = pi[code] / g[code]
        phi = sum(
            pi[i] * _regression(scenario, level, w1, w2, target.horizon, target.cause)[(1, 0)]
            for i, level in enumerate(levels)
        )
    else:
        ratio = pi[code] / g[code]
        phi = _regression(scenario, mapping(a), w1, w2, target.horizon, target.cause)[(1, 0)]
    last = scenario.n_times if scenario.end_of_study else target.horizon
    total = 0.0
    cumulative = 1.0
    nodes = [step for step in path if step[0] != "y"]
    outcome = next((step[1] for step in path if step[0] == "y"), None)
    for node, (l, retained, event) in enumerate(nodes, start=1):
        if node > last:
            break
        cumulative *= _retained(scenario, node, a, w1, w2, l)
        if not retained:
            break
        if scenario.end_of_study:
            z = float(outcome) if node == last else q_observed[(node + 1, nodes[node][0])]  # type: ignore[arg-type]
        elif event:
            z = 1.0 if event == target.cause else 0.0
        elif node == last:
            z = 0.0
        else:
            z = q_observed[(node + 1, nodes[node][0])]
        total += ratio / cumulative * (z - q_observed[(node, l)])
        if event:
            break
    return total + phi - psi


# ------------------------------------------------------------------ learners


class KnownPointMechanism(BaseEstimator):
    """The law's own treatment or retention probabilities, read off a held design.

    The treatment design is ``[W1, W2]``.  A censoring design is ``[W1, W2]`` followed by the
    node-1 treatment block: one column for two arms, and the five drop-first indicators of a
    six-level dose.  Any other width raises.

    Parameters
    ----------
    kind : {"treatment", "censoring"}
        Which factor to report.
    dose : str or None
        ``"discrete"`` for the dose law.
    """

    def __init__(self, kind: str, dose: str | None = None) -> None:
        self.kind = kind
        self.dose = dose

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownPointMechanism:
        del sample_weight
        self.classes_ = np.unique(np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w1, w2 = matrix[:, 0], matrix[:, 1]
        if self.kind == "treatment":
            if matrix.shape[1] != 2:
                raise ValueError(f"a treatment design has 2 columns, not {matrix.shape[1]}")
            return np.asarray(
                law_module.treatment_probabilities(w1, w2, dose=self.dose), dtype=float
            )
        block = matrix[:, 2:]
        expected = 5 if self.dose is not None else 1
        if block.shape[1] != expected:
            raise ValueError(
                f"a censoring design has {2 + expected} columns, not {matrix.shape[1]}"
            )
        if self.dose is not None:
            arm = np.where(block.any(axis=1), np.argmax(block, axis=1) + 1.0, 0.0)
        else:
            arm = block[:, 0]
        stay = 1.0 - np.asarray(law_module._dropout(arm, w1, w2, 0.0, self.dose), dtype=float)
        return np.column_stack([1.0 - stay, stay])


class CellProbabilities(BaseEstimator):
    """Weighted class probabilities within each distinct design row (a saturated mechanism)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellProbabilities:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = np.ones(target.size) if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(target)
        keys, inverse = np.unique(matrix, axis=0, return_inverse=True)
        self.keys_ = keys
        sums = np.zeros((keys.shape[0], self.classes_.size))
        for column, level in enumerate(self.classes_):
            sums[:, column] = np.bincount(
                inverse, weights=weights * (target == level), minlength=keys.shape[0]
            )
        totals = sums.sum(axis=1, keepdims=True)
        self.probabilities_ = np.where(totals > 0, sums / np.where(totals > 0, totals, 1.0), 0.0)
        marginal = sums.sum(axis=0)
        self.default_ = marginal / marginal.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        out = np.broadcast_to(self.default_, (matrix.shape[0], self.classes_.size)).copy()
        lookup = {tuple(key): row for row, key in enumerate(self.keys_)}
        for index, row in enumerate(map(tuple, matrix)):
            position = lookup.get(row)
            if position is not None:
                out[index] = self.probabilities_[position]
        return out

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.classes_.size == 2:
            return proba[:, 1]
        return self.classes_[np.argmax(proba, axis=1)]


class CellMeans(BaseEstimator):
    """The weighted mean of ``y`` within each distinct design row (a saturated regression)."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeans:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = np.ones(target.size) if sample_weight is None else np.asarray(sample_weight)
        keys, inverse = np.unique(matrix, axis=0, return_inverse=True)
        totals = np.bincount(inverse, weights=weights * target, minlength=keys.shape[0])
        sizes = np.bincount(inverse, weights=weights, minlength=keys.shape[0])
        self.keys_ = keys
        self.means_ = np.where(sizes > 0, totals / np.where(sizes > 0, sizes, 1.0), 0.0)
        self.default_ = float(np.average(target, weights=weights))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.round(np.asarray(X, dtype=float), 9)
        lookup = {tuple(key): row for row, key in enumerate(self.keys_)}
        return np.array(
            [
                self.means_[lookup[row]] if row in lookup else self.default_
                for row in map(tuple, matrix)
            ]
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def risk_name(label: str, horizon: int, cause: str | None = None) -> str:
    """The name a fit reports the risk or the incidence of ``label`` at node ``horizon`` by."""
    if cause is None:
        return f"risk_regimen[{label} @ t={horizon}]"
    return f"cif_regimen[{label}, {cause} @ t={horizon}]"


def contrast_name(left: str, right: str, horizon: int, cause: str | None = None) -> str:
    inside = f"{left} vs {right}" if cause is None else f"{left} vs {right}, {cause}"
    return f"ate_regimen[{inside} @ t={horizon}]"


def truths(
    scenario: Scenario,
    plans: Mapping[str, Callable[[int, int], Sequence[float]]],
    reference: str,
    *,
    weight: Any = None,
) -> dict[str, float]:
    """Every level and every contrast against ``reference``, at every node and cause."""
    curves = {label: scenario.truth_curve(assign, weight=weight) for label, assign in plans.items()}
    out: dict[str, float] = {}
    if scenario.end_of_study:
        for label, curve in curves.items():
            out[f"ey_regimen[{label}]"] = float(curve[0])
            if label != reference:
                out[f"ate_regimen[{label} vs {reference}]"] = float(curve[0] - curves[reference][0])
        return out
    causes: tuple[str | None, ...] = (
        tuple(CAUSES[c] for c in (1, 2)) if scenario.causes == 2 else (None,)
    )
    for label, curve in curves.items():
        for node in range(1, scenario.n_times + 1):
            for index, cause in enumerate(causes):
                out[risk_name(label, node, cause)] = float(curve[node - 1, index])
                if label != reference:
                    out[contrast_name(label, reference, node, cause)] = float(
                        curve[node - 1, index] - curves[reference][node - 1, index]
                    )
    return out
