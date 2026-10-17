r"""Finite laws that satisfy a linear structural nested mean model exactly.

The longitudinal counterpart of :mod:`tests.discrete_law_longitudinal` for the structural nested
mean estimator.  A law lists its variables in time order.  Each variable has a conditional
distribution over a few values, given the earlier ones, and the outcome is a function of the
whole path.  Enumerating the paths gives a table of rows with their exact probabilities.  Fitting
that table with ``weights`` equal to the probabilities and saturated learners makes every
nuisance equal the population conditional mean, so the fit returns the population solution and
an assertion can be exact.

The truths come from :func:`expectation`, an independent g-formula by enumeration.  It reads the
conditional tables of the law and nothing from the estimator.  A blip is a difference of two
such expectations, and :func:`structural_blip_errors` checks that a law satisfies the structural
model it claims.

Every law below follows one pattern, the linear probe laws of the study design:

.. code-block:: text

    S2 = 0.2 + 0.3 S1 + 0.4 A1 (probability)       Y = A1 + 0.5 A2 + 0.7 S2 + 0.3 W + ...

so the effect of the first treatment is the direct effect plus the effect through ``S2``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Path = dict[str, float]
Distribution = Sequence[tuple[float, float]]


def bern(probability: float) -> list[tuple[float, float]]:
    """A 0/1 variable with ``P(1) = probability``."""
    return [(0.0, 1.0 - probability), (1.0, probability)]


@dataclass(frozen=True)
class Var:
    """One variable of a law.

    Parameters
    ----------
    name : str
        The column name.
    role : str
        ``"baseline"``, ``"covariate"`` or ``"treatment"``.
    node : int
        ``0`` for a baseline variable, else the node it is measured at.
    law : callable
        Maps the earlier path to a list of ``(value, probability)``.
    reference : float
        The reference level of a treatment, which the structural blip is measured against.
    """

    name: str
    role: str
    node: int
    law: Callable[[Path], Distribution]
    reference: float = 0.0


@dataclass(frozen=True)
class Law:
    """A finite law with an outcome that is a function of the path."""

    name: str
    variables: tuple[Var, ...]
    outcome: Callable[[Path], float]
    #: The family handed to the fit.
    family: str = "gaussian"
    #: Treatment columns that hold a dose.
    doses: tuple[str, ...] = ()
    notes: str = field(default="", compare=False)

    @property
    def baseline(self) -> list[str]:
        return [v.name for v in self.variables if v.role == "baseline"]

    @property
    def treatments(self) -> list[str]:
        return [v.name for v in self.variables if v.role == "treatment"]

    @property
    def n_nodes(self) -> int:
        return len(self.treatments)

    @property
    def time_varying(self) -> list[list[str]]:
        return [
            [v.name for v in self.variables if v.role == "covariate" and v.node == t]
            for t in range(1, self.n_nodes + 1)
        ]

    def var(self, name: str) -> Var:
        return next(v for v in self.variables if v.name == name)

    def reference_of(self, node: int) -> float:
        return next(v for v in self.variables if v.role == "treatment" and v.node == node).reference

    def history_names(self, node: int) -> list[str]:
        """The variables of the history :math:`H_t`: baseline, covariates through ``node``,
        treatments before ``node``."""
        return [
            v.name
            for v in self.variables
            if v.role == "baseline"
            or (v.role == "covariate" and v.node <= node)
            or (v.role == "treatment" and v.node < node)
        ]


def paths(law: Law) -> list[tuple[Path, float]]:
    """Every path of positive probability, with its probability."""
    out: list[tuple[Path, float]] = [({}, 1.0)]
    for variable in law.variables:
        grown: list[tuple[Path, float]] = []
        for path, mass in out:
            for value, probability in variable.law(path):
                if probability > 0.0:
                    grown.append(({**path, variable.name: value}, mass * probability))
        out = grown
    return out


def frame(law: Law) -> pd.DataFrame:
    """The enumerated table: a row per path, the outcome ``Y`` and the probability ``p``."""
    rows = []
    for path, mass in paths(law):
        rows.append({**path, "Y": law.outcome(path), "p": mass})
    table = pd.DataFrame(rows)
    assert abs(table["p"].sum() - 1.0) < 1e-12
    return table


def expectation(law: Law, fixed: dict[str, float]) -> float:
    """The g-formula mean of the outcome with the variables in ``fixed`` held at their values.

    A variable in ``fixed`` is not drawn.  Its probability factor is one, which is an
    intervention for a treatment and a conditioning for a variable of the history.  Every other
    variable is drawn from its conditional law given the path so far.
    """

    def recurse(index: int, path: Path) -> float:
        if index == len(law.variables):
            return law.outcome(path)
        variable = law.variables[index]
        if variable.name in fixed:
            return recurse(index + 1, {**path, variable.name: fixed[variable.name]})
        return sum(
            probability * recurse(index + 1, {**path, variable.name: value})
            for value, probability in variable.law(path)
            if probability > 0.0
        )

    return recurse(0, {})


def regimen_mean(law: Law, assignment: Sequence[float]) -> float:
    """:math:`E[Y(a)]` for the static regimen ``assignment``, one value per node."""
    return expectation(law, dict(zip(law.treatments, assignment, strict=True)))


def reference_mean(law: Law) -> float:
    """:math:`E[Y(0)]`: every treatment at its reference."""
    return regimen_mean(law, [law.reference_of(t) for t in range(1, law.n_nodes + 1)])


def blip(law: Law, node: int, history: Path, level: float) -> float:
    r"""The structural blip :math:`E[Y(\bar a_{t-1}, a_t, 0) - Y(\bar a_{t-1}, 0, 0) \mid h_t]`.

    The history ``history`` holds the baseline variables, the covariates through ``node`` and
    the earlier treatments.  Later treatments are set to the reference.
    """
    later = {law.treatments[t - 1]: law.reference_of(t) for t in range(node + 1, law.n_nodes + 1)}
    name = law.treatments[node - 1]
    at_level = expectation(law, {**history, **later, name: level})
    at_reference = expectation(law, {**history, **later, name: law.reference_of(node)})
    return at_level - at_reference


def histories(law: Law, node: int) -> list[Path]:
    """The distinct histories :math:`h_t` of positive probability."""
    names = law.history_names(node)
    seen: dict[tuple[float, ...], Path] = {}
    for path, _ in paths(law):
        key = tuple(path[name] for name in names)
        seen.setdefault(key, {name: path[name] for name in names})
    return list(seen.values())


def structural_blip_errors(
    law: Law,
    node: int,
    levels: Sequence[float],
    feature: Callable[[Path], np.ndarray],
    coefficients: np.ndarray,
    coding: Callable[[float], np.ndarray],
) -> float:
    """The largest gap between the g-formula blip and :math:`c(a)^\\top \\Psi \\phi(h)`.

    Parameters
    ----------
    law : Law
        The law.
    node : int
        The node.
    levels : sequence of float
        The levels (or doses) of the node at which to compare.
    feature : callable
        :math:`\\phi_t(h)` as a vector.
    coefficients : ndarray
        :math:`\\Psi_t`, shape ``(p, d)``.
    coding : callable
        :math:`c(a)` as a vector.

    Returns
    -------
    float
        The largest absolute gap over the histories and levels.
    """
    worst = 0.0
    for history in histories(law, node):
        phi = feature(history)
        for level in levels:
            claimed = float(coding(level) @ coefficients.T @ phi)
            worst = max(worst, abs(blip(law, node, history, level) - claimed))
    return worst


# ----------------------------------------------------------------------------- the laws

#: The scale of the outcome noise.  It takes the values -NOISE, 0 and NOISE with probabilities
#: 1/4, 1/2 and 1/4, so it has mean zero and changes no truth.  Without it the outcome is a
#: function of the path and the score of the last node is identically zero at the truth, which
#: hides every term of the influence curve that multiplies it.
NOISE = 0.75


def _law(
    name: str,
    variables: tuple[Var, ...],
    mean: Callable[[Path], float],
    doses: tuple[str, ...] = (),
) -> Law:
    """A law whose outcome is ``mean`` of the path plus the symmetric noise ``E``."""
    noise = Var("E", "noise", 0, lambda p: [(-NOISE, 0.25), (0.0, 0.5), (NOISE, 0.25)])
    return Law(name, (*variables, noise), lambda p: mean(p) + p["E"], doses=doses)


def _v(
    name: str, role: str, node: int, law: Callable[[Path], Distribution], reference: float = 0.0
) -> Var:
    return Var(name, role, node, law, reference)


def law_s2() -> Law:
    """Two binary nodes; the structural model holds with constant blips 1.28 and 0.5."""
    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v("A1", "treatment", 1, lambda p: bern(0.3 + 0.2 * p["S1"] + 0.1 * p["W"])),
        _v("S2", "covariate", 2, lambda p: bern(0.2 + 0.3 * p["S1"] + 0.4 * p["A1"])),
        _v("A2", "treatment", 2, lambda p: bern(0.3 + 0.25 * p["S2"] + 0.15 * p["A1"])),
    )

    def outcome(p: Path) -> float:
        return p["A1"] + 0.5 * p["A2"] + 0.7 * p["S2"] + 0.3 * p["W"] + 0.2 * p["S1"]

    return _law("S2", variables, outcome)


def law_s3() -> Law:
    """Three binary nodes with an A1 -> A2 -> A3 path; every blip is at least 0.5."""
    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v("A1", "treatment", 1, lambda p: bern(0.3 + 0.2 * p["S1"] + 0.1 * p["W"])),
        _v("S2", "covariate", 2, lambda p: bern(0.2 + 0.3 * p["S1"] + 0.3 * p["A1"])),
        _v("A2", "treatment", 2, lambda p: bern(0.25 + 0.25 * p["S2"] + 0.25 * p["A1"])),
        _v("S3", "covariate", 3, lambda p: bern(0.2 + 0.2 * p["S2"] + 0.3 * p["A2"])),
        _v(
            "A3",
            "treatment",
            3,
            lambda p: bern(0.2 + 0.2 * p["S3"] + 0.2 * p["A1"] + 0.25 * p["A2"]),
        ),
    )

    def outcome(p: Path) -> float:
        return (
            1.0 * p["A1"]
            + 0.8 * p["A2"]
            + 0.6 * p["A3"]
            + 0.5 * p["S3"]
            + 0.4 * p["S2"]
            + 0.3 * p["W"]
            + 0.2 * p["S1"]
        )

    return _law("S3", variables, outcome)


#: The effect of each level of the three-level treatments of law K on the second covariate's
#: probability, and on the outcome.  The reference level is 0.
K_SHIFT = (0.0, 0.25, 0.45)
K_EFFECT_1 = (0.0, 1.0, 0.4)
K_EFFECT_2 = (0.0, 0.3, 0.9)


def _categorical(weights: Sequence[float]) -> list[tuple[float, float]]:
    total = float(sum(weights))
    return [(float(i), w / total) for i, w in enumerate(weights)]


def law_k() -> Law:
    """Two nodes of three levels each; the effects are not monotone in the level."""
    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v(
            "A1",
            "treatment",
            1,
            lambda p: _categorical([2.0 + p["S1"], 1.5 + 0.5 * p["W"], 1.0 + p["S1"]]),
        ),
        _v("S2", "covariate", 2, lambda p: bern(0.2 + 0.3 * p["S1"] + K_SHIFT[int(p["A1"])])),
        _v(
            "A2",
            "treatment",
            2,
            lambda p: _categorical(
                [2.0 + p["S2"], 1.0 + 0.5 * p["A1"], 1.0 + p["S2"] + 0.5 * (p["A1"] == 2)]
            ),
        ),
    )

    def outcome(p: Path) -> float:
        return (
            K_EFFECT_1[int(p["A1"])]
            + K_EFFECT_2[int(p["A2"])]
            + 0.7 * p["S2"]
            + 0.3 * p["W"]
            + 0.2 * p["S1"]
        )

    return _law("K", variables, outcome)


def law_c() -> Law:
    """Two continuous doses on a finite support; the blips are linear in the dose."""
    doses = (-1.0, 0.0, 1.0)

    def dose(weights: Sequence[float]) -> list[tuple[float, float]]:
        total = float(sum(weights))
        return [(d, w / total) for d, w in zip(doses, weights, strict=True)]

    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v("A1", "treatment", 1, lambda p: dose([2.0 + p["S1"], 2.0, 1.5 + p["W"]])),
        _v("S2", "covariate", 2, lambda p: bern(0.3 + 0.3 * p["S1"] + 0.1 * p["A1"])),
        _v(
            "A2",
            "treatment",
            2,
            lambda p: dose([2.0 + p["S2"], 1.5 + p["A1"] ** 2, 2.0 + 0.5 * p["S2"] * p["A1"]]),
        ),
    )

    def outcome(p: Path) -> float:
        return 0.9 * p["A1"] + 0.6 * p["A2"] + 0.7 * p["S2"] + 0.3 * p["W"] + 0.2 * p["S1"]

    return _law("C", variables, outcome, doses=("A1", "A2"))


def law_h() -> Law:
    """Two binary nodes; the blips depend on ``W`` and, at node 2, on ``A1``.

    The blips are :math:`A_1 (1.14 + 0.5 W)` and :math:`A_2 (0.5 + 0.4 W + 0.6 A_1)`.
    """
    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v("A1", "treatment", 1, lambda p: bern(0.3 + 0.2 * p["S1"] + 0.1 * p["W"])),
        _v("S2", "covariate", 2, lambda p: bern(0.2 + 0.3 * p["S1"] + 0.2 * p["A1"])),
        _v("A2", "treatment", 2, lambda p: bern(0.3 + 0.25 * p["S2"] + 0.15 * p["A1"])),
    )

    def outcome(p: Path) -> float:
        return (
            p["A1"] * (1.0 + 0.5 * p["W"])
            + p["A2"] * (0.5 + 0.4 * p["W"] + 0.6 * p["A1"])
            + 0.7 * p["S2"]
            + 0.3 * p["W"]
            + 0.2 * p["S1"]
        )

    return _law("H", variables, outcome)


def law_p() -> Law:
    """Two binary nodes with the interaction :math:`1.5 A_1 A_2`.

    The node-2 blip :math:`A_2 (0.5 + 1.5 A_1)` reads the earlier treatment.  With
    :math:`\\phi_2 = (1, A_1)` the structural model holds.  With a baseline-only :math:`\\phi_2`
    it does not, and :func:`law_p` is then the negative control of the study design.
    """
    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v("A1", "treatment", 1, lambda p: bern(0.3 + 0.2 * p["S1"] + 0.1 * p["W"])),
        _v("S2", "covariate", 2, lambda p: bern(0.2 + 0.3 * p["S1"] + 0.4 * p["A1"])),
        _v("A2", "treatment", 2, lambda p: bern(0.3 + 0.25 * p["S2"] + 0.15 * p["A1"])),
    )

    def outcome(p: Path) -> float:
        return (
            p["A1"]
            + 0.5 * p["A2"]
            + 1.5 * p["A1"] * p["A2"]
            + 0.7 * p["S2"]
            + 0.3 * p["W"]
            + 0.2 * p["S1"]
        )

    return _law("P", variables, outcome)


def law_m() -> Law:
    """Two binary nodes whose second treatment and outcome read the first covariate.

    The period-2 state ``(W, S2, A1)`` omits ``S1``, which confounds ``A2``.  Given the full
    history the blips are constant.
    """
    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v("A1", "treatment", 1, lambda p: bern(0.3 + 0.2 * p["S1"] + 0.1 * p["W"])),
        _v("S2", "covariate", 2, lambda p: bern(0.2 + 0.3 * p["S1"] + 0.4 * p["A1"])),
        _v(
            "A2",
            "treatment",
            2,
            lambda p: bern(0.15 + 0.15 * p["S2"] + 0.1 * p["A1"] + 0.5 * p["S1"]),
        ),
    )

    def outcome(p: Path) -> float:
        return (
            p["A1"]
            + 0.5 * p["A2"]
            + 0.7 * p["S2"]
            + 0.3 * p["W"]
            + 1.2 * p["S1"]
            + 0.6 * p["S1"] * p["W"]
        )

    return _law("M", variables, outcome)


#: The size of the dependence of the node-2 blip of law L on the second covariate.
L_SLOPE = 0.8


def law_l() -> Law:
    """Two binary nodes; the node-2 blip :math:`A_2 (0.5 + 0.8 S_2)` reads ``S2``.

    Declared with :math:`\\phi_2 = (1, S_2)` the structural model holds, and the blips and
    ``ey_reference`` are estimated.  Declared with :math:`\\phi_2 = 1` it does not, and the
    law is the negative control.
    """
    variables = (
        _v("W", "baseline", 0, lambda p: bern(0.4)),
        _v("S1", "covariate", 1, lambda p: bern(0.3 + 0.2 * p["W"])),
        _v("A1", "treatment", 1, lambda p: bern(0.3 + 0.2 * p["S1"] + 0.1 * p["W"])),
        _v("S2", "covariate", 2, lambda p: bern(0.2 + 0.3 * p["S1"] + 0.4 * p["A1"])),
        _v("A2", "treatment", 2, lambda p: bern(0.3 + 0.25 * p["S2"] + 0.15 * p["A1"])),
    )

    def outcome(p: Path) -> float:
        return (
            p["A1"]
            + p["A2"] * (0.5 + L_SLOPE * p["S2"])
            + 0.7 * p["S2"]
            + 0.3 * p["W"]
            + 0.2 * p["S1"]
        )

    return _law("L", variables, outcome)


LAWS: dict[str, Callable[[], Law]] = {
    "S2": law_s2,
    "S3": law_s3,
    "K": law_k,
    "C": law_c,
    "H": law_h,
    "P": law_p,
    "M": law_m,
    "L": law_l,
}


def static_assignments(law: Law) -> list[tuple[float, ...]]:
    """Every static regimen over the support of each treatment."""
    supports = []
    for treatment in law.treatments:
        values = sorted({p[treatment] for p, _ in paths(law)})
        supports.append(values)
    return list(itertools.product(*supports))


def moment(law: Law, function: Callable[[Path], float]) -> float:
    """:math:`E[f(\\text{path})]` under the observed law."""
    return float(sum(mass * function(path) for path, mass in paths(law)))


def conditional_mean(
    law: Law, node: int, target: Callable[[Path], float]
) -> Callable[[Path], float]:
    """The function :math:`h \\mapsto E[f \\mid H_t = h]` under the observed law, as a lookup."""
    names = law.history_names(node)
    sums: dict[tuple[float, ...], float] = {}
    masses: dict[tuple[float, ...], float] = {}
    for path, mass in paths(law):
        key = tuple(path[name] for name in names)
        sums[key] = sums.get(key, 0.0) + mass * target(path)
        masses[key] = masses.get(key, 0.0) + mass

    def lookup(path: Path) -> float:
        key = tuple(path[name] for name in names)
        return sums[key] / masses[key]

    return lookup


def treatment_state_mean(law: Law, node: int, state: Sequence[str]) -> Callable[[Path], float]:
    """:math:`E[A_t \\mid \\text{state}]` under the observed law, for a subset of the history."""
    name = law.treatments[node - 1]
    sums: dict[tuple[float, ...], float] = {}
    masses: dict[tuple[float, ...], float] = {}
    for path, mass in paths(law):
        key = tuple(path[s] for s in state)
        sums[key] = sums.get(key, 0.0) + mass * path[name]
        masses[key] = masses.get(key, 0.0) + mass

    def lookup(path: Path) -> float:
        key = tuple(path[s] for s in state)
        return sums[key] / masses[key]

    return lookup
