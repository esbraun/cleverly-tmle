"""The pre-declaration design of the three long-horizon studies, computed from the exact law.

This module holds every declared input of the long-horizon design and the rules that check it:

* the exact-law criterion of each quantity (tail ``n_kappa`` from ``max(kappa_{D*},
  kappa_{D_inf})``, the support minimum, the truth range) and the law rule that assigns each
  quantity to ``L_R`` or to its own fallback law;
* the gating scenario of each primary row, and each scenario's n;
* the gated calibration cells, with their law and cell n;
* each bias family's gated positive and its control, as an estimator :class:`Config` that both
  the exact reading (:mod:`tests.long_horizon_sampling_sd`) and a sample fit with ``cleverly``
  read, with the control's declared exact bias, sampling SD and displacement;
* the closed-form sizing floors (calibration F1, the bias ladder, the truth rows, power);
* the CPU cost model of the pilots.

``tests/unit/test_long_horizon_design.py`` asserts the declared values against the exact law.
``tests/long_horizon_pilots.py`` runs the scans that take longer than the fast tier allows and
the sample pilots.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tests import discrete_law_longitudinal_horizon as law_
from tests import long_horizon_sampling_sd as sampling
from tests.discrete_law_longitudinal_horizon import Law
from tests.studies import (
    canonical_long_horizon_categorical_ltmle as s2,
)
from tests.studies import (
    canonical_long_horizon_ltmle as s1,
)
from tests.studies import (
    canonical_long_horizon_sdr as s3,
)
from tests.studies import long_horizon_common as common
from tests.studies import long_horizon_primary as primary
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The sample-size cap of every law assignment and every cell.
NMAX = 128_000
#: The raised cap of the S2 ``LTMLE`` third-arm control's law scan alone.
NMAX_Q14 = 296_000
#: Every cell n is a multiple of this.
STEP = 4_000
#: The displacement floor of a control, in sampling SDs.
D_FLOOR = 3.0
#: A followed path keeps at least this cumulative probability on every gated cell's law.
SUPPORT_FLOOR = 1e-6
#: The per-replicate scale of one arm's SE ratio at 400 replicates (rev-1 pilot readings).
SE_RATIO_SCALE = 0.0354
#: The seed of every pilot stream; a stream's seed spawns from it by cell and replicate.
PILOT_SEED = 20263931

STUDIES = {"S1": s1, "S2": s2, "S3": s3}


def round_n(n: float) -> int:
    """``n`` rounded up to a multiple of :data:`STEP`, at least :data:`STEP`."""
    return max(STEP, math.ceil(n / STEP) * STEP)


def law_name(law: Law) -> str:
    """``(B, s, c0)``, with ``gamma_C`` when it is not 1."""
    base = f"({law.B:g}, {law.s:g}, {law.c0:g})"
    return base if law.gamma_C == 1.0 else f"{base} gamma_C={law.gamma_C:g}"


# ----------------------------------------------------------------------------- exact pieces
@cache
def frame(K: int, m: int, law: Law) -> pd.DataFrame:
    """The weighted exact frame; callers copy before they change it."""
    return law_.weighted_frame(K, m, law)


@cache
def weight(K: int, m: int, law: Law) -> np.ndarray:
    return np.asarray(law_.masses(K, m, law))


@cache
def eif(K: int, m: int, plan: str, law: Law) -> np.ndarray:
    return law_.eif(K, m, plan, law)


@cache
def glm_curve(K: int, m: int, plan: str, law: Law) -> np.ndarray:
    """``D_inf``: the declared GLM ``LTMLE`` curve per support row, known mechanism."""
    return sampling.Sequential(frame(K, m, law), K, m, plan, law).fit().curve


@cache
def support_minimum(K: int, m: int, plan: str, law: Law) -> float:
    cumulative = law_.cumulative_follow(K, m, plan, law)
    return float(np.nanmin(cumulative))


@cache
def ess_top(K: int, m: int, law: Law) -> float:
    """The node-K effective sample fraction ``1 / E[R_K^2]`` of the static top plan."""
    return law_.diagnostics(K, m, "high" if m == 3 else "always", law)["ess_K"]


# ----------------------------------------------------------------------------- quantities
@dataclass(frozen=True)
class Quantity:
    """A plan mean (``second`` empty) or a contrast ``psi(first) - psi(second)``.

    Parameters
    ----------
    K : int
        Decision nodes.
    m : int
        Arms.
    first : str
        The plan, or the first plan of the contrast.
    second : str
        The second plan of a contrast; empty for a mean.
    """

    K: int
    m: int
    first: str
    second: str = ""

    @property
    def name(self) -> str:
        return self.first if not self.second else f"{self.first}_vs_{self.second}"

    @property
    def plans(self) -> tuple[str, ...]:
        return (self.first,) if not self.second else (self.first, self.second)

    def __str__(self) -> str:
        return f"{self.name} K={self.K} m={self.m}"


@dataclass(frozen=True)
class Tail:
    """The criterion readings of one quantity on one law.

    Parameters
    ----------
    sigma : float
        The SD of ``D*``.
    n_star : int
        ``n_kappa`` of ``D*``.
    n_inf : int
        ``n_kappa`` of ``D_inf``, the declared GLM curve.
    support : float
        The smallest cumulative follow probability over the quantity's plans.
    truths : tuple of float
        The plan means.
    """

    sigma: float
    n_star: int
    n_inf: int
    support: float
    truths: tuple[float, ...]

    @property
    def n_kappa(self) -> int:
        return max(self.n_star, self.n_inf)

    def meets(self, cap: int = NMAX) -> bool:
        """Whether the quantity meets the criterion on this law at ``n <= cap``."""
        return (
            self.n_kappa <= cap
            and self.support >= SUPPORT_FLOOR
            and all(0.15 <= value <= 0.85 for value in self.truths)
        )


def _difference(values: Mapping[str, np.ndarray], q: Quantity) -> np.ndarray:
    return values[q.first] - (values[q.second] if q.second else 0.0)


@cache
def tail(q: Quantity, law: Law, star_only: bool = False) -> Tail:
    """The criterion readings of ``q`` on ``law`` (``star_only`` skips the GLM fit)."""
    w = weight(q.K, q.m, law)
    star = _difference({p: eif(q.K, q.m, p, law) for p in q.plans}, q)
    n_star = sampling.n_kappa(sampling.kappa(w, star))
    n_inf = 0
    if not star_only:
        limit = _difference({p: glm_curve(q.K, q.m, p, law) for p in q.plans}, q)
        n_inf = sampling.n_kappa(sampling.kappa(w, limit))
    return Tail(
        sigma=sampling.sd(w, star),
        n_star=n_star,
        n_inf=n_inf,
        support=min(support_minimum(q.K, q.m, p, law) for p in q.plans),
        truths=tuple(law_.truth(q.K, q.m, p, law) for p in q.plans),
    )


@cache
def msm_tail(K: int, law: Law, term: str) -> Tail:
    """The criterion readings of one duration working-model coefficient (``m = 2``)."""
    plans = [f"dur{j}" for j in range(K + 1)]
    design = np.column_stack([np.ones(K + 1), np.arange(K + 1)])
    operator = np.linalg.solve(design.T @ design, design.T)
    row = operator[("intercept", "duration").index(term)]
    w = weight(K, 2, law)
    star = np.column_stack([eif(K, 2, p, law) for p in plans]) @ row
    limit = np.column_stack([glm_curve(K, 2, p, law) for p in plans]) @ row
    return Tail(
        sigma=sampling.sd(w, star),
        n_star=sampling.n_kappa(sampling.kappa(w, star)),
        n_inf=sampling.n_kappa(sampling.kappa(w, limit)),
        support=min(support_minimum(K, 2, p, law) for p in plans),
        truths=tuple(law_.truth(K, 2, p, law) for p in plans),
    )


#: The quantities that fail on ``L_R`` at :data:`NMAX`, and the fallback law the rule picks.
FALLBACKS: dict[Quantity, Law] = {
    Quantity(4, 3, "high"): Law(0.75, 0.1, 3.5),
    Quantity(4, 3, "high", "low"): Law(1.0, 0.0, 4.5),
    Quantity(5, 3, "high"): Law(0.25, 0.0, 3.5),
    Quantity(5, 3, "high", "low"): Law(0.5, 0.05, 3.5),
    Quantity(5, 3, "switch"): Law(0.5, 0.05, 3.5),
    Quantity(5, 3, "switch", "low"): Law(0.5, 0.1, 3.5),
    Quantity(5, 3, "mix"): Law(1.0, 0.05, 3.5),
    Quantity(5, 3, "taper"): Law(1.0, 0.0, 3.5),
    Quantity(5, 3, "taper", "mix"): Law(1.0, 0.05, 3.5),
}


def law_of(q: Quantity) -> Law:
    """The declared law of ``q``: ``L_R`` unless :data:`FALLBACKS` names another."""
    return FALLBACKS.get(q, law_.L_R)


def fallback_order(K: int, m: int) -> list[Law]:
    """The grid laws with ``B >= 0.25``, most support-depleted first (smallest ``ess_K``)."""
    return sorted(law_.GRID, key=lambda law: ess_top(K, m, law))


def fallback_scan(q: Quantity, cap: int = NMAX) -> tuple[Law | None, list[dict[str, Any]]]:
    """The section 5.1 rule over the whole grid: the most depleted law on which ``q`` meets."""
    rows = []
    pick = None
    for law in fallback_order(q.K, q.m):
        star = tail(q, law, star_only=True)
        row = {"law": law_name(law), "ess_K": ess_top(q.K, q.m, law), "n_star": star.n_star}
        if star.n_star <= cap and pick is None:
            full = tail(q, law)
            row.update(n_inf=full.n_inf, support=full.support, meets=full.meets(cap))
            if full.meets(cap):
                pick = law
        rows.append(row)
    return pick, rows


# ----------------------------------------------------------------------------- primary rows
@dataclass(frozen=True)
class Row:
    """One primary row: a quantity in one study under one estimator configuration.

    Parameters
    ----------
    study : str
        ``S1``, ``S2`` or ``S3``.
    quantity : Quantity
        The row's quantity.
    folds : int
        The outer fold count of the configuration.
    """

    study: str
    quantity: Quantity
    folds: int


def scenarios(study: str) -> tuple[primary.Scenario, ...]:
    return tuple(STUDIES[study].SCENARIOS)


def scenario_quantities(scenario: primary.Scenario) -> list[Quantity]:
    """The quantities a non-MSM scenario reports, in its estimand order."""
    K, m, ref = scenario.K, scenario.m, scenario.reference
    out = [Quantity(K, m, plan) for plan in scenario.plans]
    out += [Quantity(K, m, plan, ref) for plan in scenario.plans[1:]]
    out += [Quantity(K, m, a, b) for a, b in scenario.paired]
    return out


def primary_rows() -> list[Row]:
    """Every primary row of every study, without repeats."""
    seen: dict[Row, None] = {}
    for study in STUDIES:
        for scenario in scenarios(study):
            if scenario.msm:
                continue
            for q in scenario_quantities(scenario):
                seen[Row(study, q, scenario.n_folds)] = None
    return list(seen)


def gating(row: Row) -> primary.Scenario | None:
    """The scenario that gates ``row``: on its law, smallest n at least its ``n_kappa``."""
    law = law_of(row.quantity)
    need = tail(row.quantity, law).n_kappa
    candidates = [
        scenario
        for scenario in scenarios(row.study)
        if not scenario.msm
        and scenario.n_folds == row.folds
        and scenario.law == law
        and row.quantity in scenario_quantities(scenario)
        and scenario.n >= need
    ]
    return min(candidates, key=lambda s: s.n) if candidates else None


# ----------------------------------------------------------------------------- calibration cells
@dataclass(frozen=True)
class CalibrationCell:
    """One gated ``interval_calibration`` positive.

    Parameters
    ----------
    study : str
        ``S1``, ``S2`` or ``S3``.
    name : str
        The cell name.
    quantity : Quantity
        The cell's quantity; the MSM cell names its plans ``dur0`` and reads the duration term.
    law : Law
        The cell's law.
    n : int
        The cell n.
    folds : int
        The outer fold count.
    msm : bool
        The duration working-model coefficient instead of ``quantity``.
    """

    study: str
    name: str
    quantity: Quantity
    law: Law
    n: int
    folds: int = 1
    msm: bool = False

    @property
    def estimator(self) -> str:
        return "SDR" if self.study == "S3" else "LTMLE"

    def tail(self) -> Tail:
        if self.msm:
            return msm_tail(self.quantity.K, self.law, "duration")
        return tail(self.quantity, self.law)

    def config(self) -> Config:
        q = self.quantity
        return Config(
            self.estimator,
            q.K,
            q.m,
            self.law,
            q.first,
            q.second,
            n_folds=self.folds,
            msm=self.msm,
        )


def _calibration_names(m: int) -> dict[str, tuple[str, str]]:
    ref, top, policy, last = (
        ("never", "always", "coin", "coinlast") if m == 2 else ("low", "high", "mix", "mixlast")
    )
    return {
        "respond_vs_ref": ("respond", ref),
        "policy": (policy, ""),
        "policy_vs_ref": (policy, ref),
        "policy_vs_policylast": (policy, last),
        "static_contrast": (top, ref),
    }


#: The calibration-cell n table of section 5.1: ``(m, K) -> {cell: n}``.
CALIBRATION_N = {
    (2, 3): dict.fromkeys(_calibration_names(2), 4_000),
    (2, 4): {
        "respond_vs_ref": 8_000,
        "policy": 4_000,
        "policy_vs_ref": 8_000,
        "policy_vs_policylast": 4_000,
        "static_contrast": 8_000,
    },
    (2, 5): {
        "respond_vs_ref": 40_000,
        "policy": 4_000,
        "policy_vs_ref": 60_000,
        "policy_vs_policylast": 4_000,
        "static_contrast": 36_000,
    },
    (3, 3): {
        "respond_vs_ref": 4_000,
        "policy": 4_000,
        "policy_vs_ref": 4_000,
        "policy_vs_policylast": 4_000,
        "static_contrast": 24_000,
    },
    (3, 4): {
        "respond_vs_ref": 8_000,
        "policy": 24_000,
        "policy_vs_ref": 4_000,
        "policy_vs_policylast": 12_000,
        "static_contrast": 124_000,
    },
    (3, 5): {
        "respond_vs_ref": 48_000,
        "policy": 96_000,
        "policy_vs_ref": 16_000,
        "policy_vs_policylast": 84_000,
        "static_contrast": 124_000,
    },
}


def calibration_cells() -> list[CalibrationCell]:
    """Every gated calibration positive of the three studies, in study order."""
    out = []
    arms = {"S1": (2,), "S2": (3,), "S3": (2, 3)}
    for study, ms in arms.items():
        for m in ms:
            prefix = "" if study != "S3" else ("binary_" if m == 2 else "categorical_")
            for K in (3, 4, 5):
                for cell, (first, second) in _calibration_names(m).items():
                    q = Quantity(K, m, first, second)
                    out.append(
                        CalibrationCell(
                            study,
                            f"{prefix}{cell}__k{K}",
                            q,
                            law_of(q),
                            CALIBRATION_N[(m, K)][cell],
                        )
                    )
        if study == "S1":
            out.append(
                CalibrationCell(
                    study, "msm_duration__k5", Quantity(5, 2, "dur0"), law_.L_R, 28_000, msm=True
                )
            )
        else:
            names = _calibration_names(3)
            for cell, n in (("policy_vs_ref", 16_000), ("policy_vs_policylast", 84_000)):
                prefix = "" if study == "S2" else "categorical_"
                out.append(
                    CalibrationCell(
                        study,
                        f"{prefix}{cell}__k5_crossfit",
                        Quantity(5, 3, *names[cell]),
                        law_.L_R,
                        n,
                        folds=5,
                    )
                )
    return out


# ----------------------------------------------------------------------------- estimator configs
#: The outcome-learner ladder of section 5.3, as the sampling-SD learner names.
RUNGS = {"O1": "glm", "O2": "wblind", "O3": "int", "O4": "lblind"}


@dataclass(frozen=True)
class Config:
    """One estimator configuration of one mean or contrast, exact and on a sample.

    Parameters
    ----------
    estimator : str
        ``"LTMLE"``, ``"SDR"`` or ``"plugin"`` (the unfluctuated sequential regression).
    K : int
        Decision nodes.
    m : int
        Arms.
    law : Law
        The law.
    first : str
        The plan fitted first.
    second : str
        The second plan of a contrast; empty for a mean.
    target : str
        The plan whose truth ``first`` targets, when ``first`` is a mutated plan.
    learners : str or tuple
        A sampling-SD learner name for every node, or ``((node, name), ...)``.
    mechanism : str or tuple
        ``"known"``, a :func:`~tests.studies.long_horizon_common.wrong_mechanism` kind,
        ``"complement"``, or ``((node, kind), ...)``.
    complement_nodes : tuple of int
        The nodes of the two-arm shortcut.
    n_folds : int
        The outer fold count.
    trees : bool
        Fully grown trees for the outcome regressions.
    msm : bool
        The duration working model over ``dur0..durK`` (``first`` is ignored).
    """

    estimator: str
    K: int
    m: int
    law: Law
    first: str
    second: str = ""
    target: str = ""
    learners: str | tuple[tuple[int, str], ...] = "glm"
    mechanism: str | tuple[tuple[int, str], ...] = "known"
    complement_nodes: tuple[int, ...] = ()
    n_folds: int = 1
    trees: bool = False
    msm: bool = False

    # ------------------------------------------------------------------------- identity
    def ident(self) -> str:
        return (
            f"{self.estimator}|K{self.K}|m{self.m}|{law_name(self.law)}|{self.first}"
            f"|{self.second}|{self.target}|{self.learners}|{self.mechanism}"
            f"|{self.complement_nodes}|folds{self.n_folds}|trees{int(self.trees)}"
            f"|msm{int(self.msm)}"
        )

    @property
    def kind(self) -> str:
        return {"LTMLE": "ltmle", "SDR": "sdr", "plugin": "plugin"}[self.estimator]

    @property
    def plans(self) -> tuple[str, ...]:
        """The plans a fit of the configuration reports."""
        return (self.first, self.second) if self.second else (self.first,)

    @property
    def truth(self) -> float:
        if self.msm:
            return float(law_.msm_coefficients(self.K, self.law)["msm_duration"]["psi"])
        value = law_.truth(self.K, self.m, self.target or self.first, self.law)
        if self.second:
            value -= law_.truth(self.K, self.m, self.second, self.law)
        return float(value)

    def node_learners(self) -> dict[int, str]:
        if isinstance(self.learners, str):
            return dict.fromkeys(range(1, self.K + 1), self.learners)
        return {t: dict(self.learners).get(t, "glm") for t in range(1, self.K + 1)}

    def node_mechanisms(self) -> dict[int, str]:
        if isinstance(self.mechanism, str):
            return dict.fromkeys(range(1, self.K + 1), self.mechanism)
        return {t: dict(self.mechanism).get(t, "known") for t in range(1, self.K + 1)}

    # ------------------------------------------------------------------------- mechanism
    def mechanism_frame(self, data: pd.DataFrame, *, population: bool) -> pd.DataFrame:
        """``data`` with each node's ``g`` and ``pc`` columns set to the declared mechanism.

        ``population`` reads a wrong mechanism's values from the law (the weighted exact
        frame) by history, so a sample carries the same fixed mechanism the exact reading
        does.  On the weighted frame itself the two coincide.
        """
        out = data.copy()
        kinds = self.node_mechanisms()
        for kind in sorted(set(kinds.values()) - {"known"}):
            if kind == "complement" and population:
                continue  # a sample fit carries the shortcut in its treatment learner
            if kind == "complement":
                changed = sampling.complement(data, self.complement_nodes)
            elif population:
                changed = _population_mechanism(data, self.K, self.m, self.law, kind)
            else:
                changed = common.wrong_mechanism(data, self.K, self.m, kind)
            for t, node_kind in kinds.items():
                if node_kind == kind:
                    columns = [f"g{t}_{k}" for k in range(self.m)] + [f"pc{t}"]
                    out[columns] = changed[columns]
        return out

    # ------------------------------------------------------------------------- exact
    def exact(self) -> sampling.Reading:
        """The exact reading on the weighted frame."""
        if self.msm or self.trees or self.n_folds != 1:
            raise ValueError(f"no exact reading for {self.ident()}")
        data = self.mechanism_frame(frame(self.K, self.m, self.law), population=False)
        return sampling.reading(
            data,
            self.K,
            self.m,
            self.first,
            self.second or None,
            self.law,
            truth_first=self.target or None,
            kind=self.kind,
            learners=self.node_learners(),
        )

    def positive_tail(self) -> int:
        """``n_kappa`` from ``max(kappa_{D*}, kappa_curve)`` of this fit's reported curve."""
        data = self.mechanism_frame(frame(self.K, self.m, self.law), population=False)
        w = weight(self.K, self.m, self.law)
        settings = {"kind": self.kind, "learners": self.node_learners()}
        curve = sampling.Sequential(data, self.K, self.m, self.first, self.law, **settings)
        values = curve.fit().curve
        star = eif(self.K, self.m, self.target or self.first, self.law)
        if self.second:
            other = sampling.Sequential(data, self.K, self.m, self.second, self.law, **settings)
            values = values - other.fit().curve
            star = star - eif(self.K, self.m, self.second, self.law)
        return max(
            sampling.n_kappa(sampling.kappa(w, star)), sampling.n_kappa(sampling.kappa(w, values))
        )

    # ------------------------------------------------------------------------- a sample
    def _learner(self, name: str, plan: str, *, pseudo: bool) -> Any:
        policy = tuple(sorted(law_.policy_nodes(plan, self.K)))
        linear = pseudo and self.estimator == "SDR"
        if name == "glm":
            return LinearRegression() if linear else QuasiBinomialGLM()
        if name == "int":
            return common.InterceptOnly()
        if name == "offset":
            link = "identity" if linear else "logit"
            return common.OracleOffsetGLM(plan, self.K, self.m, self.law, link=link)
        if linear:
            raise ValueError(f"no least-squares learner {name!r} for the SDR transform")
        return common.ColumnSubsetGLM("W" if name == "wblind" else "L", self.m, self.K, policy)

    def learners_for(self, plan: str) -> tuple[Any, Any]:
        """The outcome and pseudo learners of a ``cleverly`` fit of ``plan``."""
        if self.trees:
            return (
                DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
                DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            )
        nodes = self.node_learners()
        if len(set(nodes.values())) == 1:
            name = nodes[1]
            return self._learner(name, plan, pseudo=False), self._learner(name, plan, pseudo=True)
        policy = tuple(sorted(law_.policy_nodes(plan, self.K)))
        outcome = {t: self._learner(n, plan, pseudo=False) for t, n in nodes.items()}
        pseudo = {t: self._learner(n, plan, pseudo=True) for t, n in nodes.items()}
        return (
            common.NodePattern(outcome, None, self.m, self.K, policy),
            common.NodePattern(pseudo, None, self.m, self.K, policy),
        )

    def _separate(self) -> bool:
        """One plan per call: a learner that reads the plan, or plans with other policy nodes."""
        if self.trees or not self.second:
            return False
        uses_plan = "offset" in self.node_learners().values() or (
            "lblind" in self.node_learners().values()
        )
        differ = law_.policy_nodes(self.first, self.K) != law_.policy_nodes(self.second, self.K)
        return uses_plan or (differ and set(self.node_learners().values()) != {"glm"})

    def fit_sample(self, sample: pd.DataFrame) -> dict[str, float]:
        """One fit of a drawn sample: estimate, standard error and interval."""
        data = self.mechanism_frame(sample, population=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.estimator == "plugin":
                return self._plugin(data)
            if self.msm:
                return self._msm(data)
            if self._separate():
                first = self._call(data, (self.first,), f"ey_regimen[{self.first}]")
                second = self._call(data, (self.second,), f"ey_regimen[{self.second}]")
                nan = math.nan
                return {
                    "estimate": first["estimate"] - second["estimate"],
                    "std_error": nan,
                    "ci_lower": nan,
                    "ci_upper": nan,
                }
            if self.second:
                return self._call(
                    data,
                    (self.second, self.first),
                    f"ate_regimen[{self.first} vs {self.second}]",
                )
            return self._call(data, (self.first,), f"ey_regimen[{self.first}]")

    def _route(self) -> dict[str, Any]:
        """The two-arm shortcut goes through the learner route: its rows do not sum to one."""
        if "complement" not in self.node_mechanisms().values():
            return {}
        if set(self.node_mechanisms().values()) != {"complement"}:
            raise ValueError("the two-arm shortcut composes with the law's mechanism only")
        learner = common.HorizonComplementMechanism(self.K, self.m, self.law, self.complement_nodes)
        return {"treatment_learner": learner, "probabilities": "censoring"}

    def _call(self, data: pd.DataFrame, plans: tuple[str, ...], name: str) -> dict[str, float]:
        outcome, pseudo = self.learners_for(plans[-1])
        result = common.fit(
            data,
            self.K,
            self.m,
            plans,
            estimator=self.estimator,
            outcome_learner=outcome,
            pseudo_learner=pseudo,
            n_folds=self.n_folds,
            **self._route(),
        )
        estimate = result[name]
        low, high = estimate.ci
        return {
            "estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": float(low),
            "ci_upper": float(high),
        }

    def _msm(self, data: pd.DataFrame) -> dict[str, float]:
        plans = tuple(f"dur{j}" for j in range(self.K + 1))
        result = common.fit(
            data, self.K, 2, plans, estimator=self.estimator, msm=primary.duration_msm()
        )
        estimate = result["msm_regimen[duration]"]
        low, high = estimate.ci
        return {
            "estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": float(low),
            "ci_upper": float(high),
        }

    def _plugin(self, data: pd.DataFrame) -> dict[str, float]:
        name = self.node_learners()[1]
        if len(set(self.node_learners().values())) != 1:
            raise ValueError("the plug-in takes one learner for every node")
        value = common.untargeted_plugin(
            data, self.first, self.K, self.m, self._learner(name, self.first, pseudo=False)
        )
        if self.second:
            value -= common.untargeted_plugin(
                data, self.second, self.K, self.m, self._learner(name, self.second, pseudo=False)
            )
        nan = math.nan
        return {"estimate": value, "std_error": nan, "ci_lower": nan, "ci_upper": nan}


@cache
def _mechanism_lookup(
    K: int, m: int, law: Law, kind: str
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per node, the wrong mechanism's ``g`` row and ``pc`` by (history index, arm)."""
    exact = frame(K, m, law)
    wrong = common.wrong_mechanism(exact, K, m, kind)
    out = {}
    for t in range(1, K + 1):
        reached = exact[f"A{t}"].notna().to_numpy()
        index = _history(exact, K, m, t)
        arm = np.nan_to_num(exact[f"A{t}"].to_numpy(float)).astype(int)
        size = law_.histories(K, m).size(t)
        g = np.full((size, m, m), np.nan)
        pc = np.full((size, m), np.nan)
        rows = np.flatnonzero(reached)
        g[index[rows], arm[rows]] = wrong[[f"g{t}_{k}" for k in range(m)]].to_numpy(float)[rows]
        pc[index[rows], arm[rows]] = wrong[f"pc{t}"].to_numpy(float)[rows]
        out[t] = (g, pc)
    return out


def _history(data: pd.DataFrame, K: int, m: int, t: int) -> np.ndarray:
    covs = {s: np.nan_to_num(data[f"L{s}"].to_numpy(float)) for s in range(2, t + 1)}
    arms = {s: np.nan_to_num(data[f"A{s}"].to_numpy(float)).astype(int) for s in range(1, t)}
    return np.asarray(
        law_.history_index(t, m, np.nan_to_num(data["W"].to_numpy(float)), covs, arms)
    )


def _population_mechanism(data: pd.DataFrame, K: int, m: int, law: Law, kind: str) -> pd.DataFrame:
    """``data`` with the law's wrong mechanism ``kind`` looked up by each row's history."""
    out = data.copy()
    lookup = _mechanism_lookup(K, m, law, kind)
    for t in range(1, K + 1):
        reached = data[f"A{t}"].notna().to_numpy()
        index = _history(data, K, m, t)
        arm = np.nan_to_num(data[f"A{t}"].to_numpy(float)).astype(int)
        g, pc = lookup[t]
        rows = np.flatnonzero(reached)
        values = g[index[rows], arm[rows]]
        for k in range(m):
            column = out[f"g{t}_{k}"].to_numpy(float).copy()
            column[rows] = values[:, k]
            out[f"g{t}_{k}"] = column
        column = out[f"pc{t}"].to_numpy(float).copy()
        column[rows] = pc[index[rows], arm[rows]]
        out[f"pc{t}"] = column
    return out


# ----------------------------------------------------------------------------- bias families
@dataclass(frozen=True)
class Control:
    """A declared control with its exact reading.

    Parameters
    ----------
    study : str
        ``S1``, ``S2`` or ``S3``.
    family : str
        The property family.
    cell : str
        The control cell.
    config : Config
        The control estimator.
    n : int
        The control n.
    bias : float
        The declared exact bias.
    sigma : float
        The declared sampling SD per unit.
    monte_carlo : bool
        Whether the design's Monte Carlo check draws it.
    """

    study: str
    family: str
    cell: str
    config: Config
    n: int
    bias: float
    sigma: float
    monte_carlo: bool = False


@dataclass(frozen=True)
class Positive:
    """A gated bias-family positive.

    Parameters
    ----------
    study : str
        ``S1``, ``S2`` or ``S3``.
    family : str
        The property family.
    cell : str
        The positive cell.
    config : Config
        The positive estimator.
    n : int
        The positive's own n.
    """

    study: str
    family: str
    cell: str
    config: Config
    n: int


def _respond_vs_ref(m: int) -> tuple[str, str]:
    return ("respond", "never" if m == 2 else "low")


def _policy(m: int) -> str:
    return "coin" if m == 2 else "mix"


#: The ``targeting_necessity`` picks: ``(m, plan, K, law, rung, positive n', control n,
#: b_c, sigma_c)``.
TARGETING = (
    (2, "coin", 3, law_.L_R, "O3", 4_000, 4_000, -0.0316, 0.534),
    (2, "coin", 4, law_.L_R, "O3", 4_000, 4_000, -0.0382, 0.561),
    (2, "coin", 5, law_.L_R, "O3", 4_000, 4_000, -0.0448, 0.605),
    (2, "respond", 3, law_.L_R, "O3", 4_000, 12_000, -0.0295, 1.007),
    (2, "respond", 4, law_.L_R, "O3", 4_000, 16_000, -0.0324, 1.255),
    (2, "respond", 5, law_.L_R, "O3", 4_000, 16_000, -0.0399, 1.600),
    (3, "mix", 3, law_.L_R, "O3", 4_000, 4_000, -0.0286, 0.533),
    (3, "mix", 4, law_.L_R, "O3", 24_000, 24_000, -0.0239, 0.559),
    (3, "mix", 5, Law(1.0, 0.05, 3.5), "O3", 96_000, 96_000, -0.0238, 0.602),
    (3, "respond", 3, law_.L_R, "O4", 4_000, 80_000, -0.0182, 1.690),
    (3, "respond", 4, law_.L_R, "O3", 16_000, 56_000, 0.0302, 2.309),
    (3, "respond", 5, law_.L_R, "O3", 116_000, 116_000, 0.0550, 3.159),
)

_G2 = Law(1.0, 0.1, 3.5, 2.0)
_S2_CENSORING = Law(0.75, 0.05, 3.5, 2.0)
_MIX_LAW = Law(1.0, 0.05, 3.5)
_Q14_LAW = Law(0.25, 0.05, 5.5)
_RULE_S1 = Law(0.5, 0.0, 3.5)
_RULE_S2 = Law(0.75, 0.1, 3.5)
_EVERY = (1, 2, 3, 4, 5)


def pattern(
    estimator: str,
    plan: str,
    odd_right: bool,
    wrong: Iterable[int] = (),
    *,
    rung: str = "int",
    mechanism: str = "constant",
) -> Config:
    """A multiply robust pattern of section 5.3 at K = 5, m = 3 on ``L_R``.

    ``P_odd`` (``odd_right``) keeps the law's mechanism and the outcome learner ``rung`` at
    nodes 1, 3 and 5, and ``OracleOffsetGLM`` with the wrong ``mechanism`` at nodes 2 and 4;
    ``P_even`` swaps them.  ``wrong`` also gives those nodes the wrong mechanism (a control).
    """
    learners, mechanisms = [], []
    broken = set(wrong)
    for t in range(1, 6):
        right_mechanism = (t % 2 == 1) == odd_right
        learners.append((t, rung if right_mechanism else "offset"))
        wrong_here = not right_mechanism or t in broken
        mechanisms.append((t, mechanism if wrong_here else "known"))
    return Config(
        estimator,
        5,
        3,
        law_.L_R,
        plan,
        "low",
        learners=tuple(learners),
        mechanism=tuple(mechanisms),
    )


def controls() -> list[Control]:
    """Every declared bias-family control, with its exact bias and sampling SD."""
    out = []
    for m, plan, K, law, rung, _, n, bias, sigma in TARGETING:
        study = "S1" if m == 2 else "S2"
        config = Config("plugin", K, m, law, plan, learners=RUNGS[rung])
        monte_carlo = (m, plan, K) == (3, "respond", 3)
        out.append(
            Control(
                study,
                "targeting_necessity",
                f"{plan}__k{K}__untargeted",
                config,
                n,
                bias,
                sigma,
                monte_carlo,
            )
        )
    for study, m, mechanism, rung, n, bias, sigma in (
        ("S1", 2, "constant", "O1", 40_000, 0.0517, 3.137),
        ("S2", 3, "confounder_blind", "O3", 52_000, 0.0894, 6.599),
    ):
        config = Config(
            "LTMLE", 5, m, law_.L_R, *_respond_vs_ref(m), learners=RUNGS[rung], mechanism=mechanism
        )
        out.append(
            Control(
                study,
                "double_robustness",
                "respond_vs_ref__k5__both_wrong",
                config,
                n,
                bias,
                sigma,
                study == "S2",
            )
        )
    censoring = (
        ("S1", "LTMLE", 2, _G2, "O3", 36_000, -0.0173, 1.048, True),
        ("S2", "LTMLE", 3, _S2_CENSORING, "O4", 128_000, -0.0116, 1.369, False),
        ("S3", "SDR", 2, law_.L_R, "O3", 16_000, -0.0188, 0.746, True),
        ("S3", "SDR", 3, _MIX_LAW, "O3", 116_000, -0.0105, 1.188, False),
    )
    for study, estimator, m, law, rung, n, bias, sigma, mc in censoring:
        config = Config(
            estimator, 5, m, law, _policy(m), learners=RUNGS[rung], mechanism="retention_one"
        )
        out.append(
            Control(
                study,
                "censoring_necessity",
                f"{_policy(m)}__k5__ignored",
                config,
                n,
                bias,
                sigma,
                mc,
            )
        )
    third = (
        ("S2", "LTMLE", _Q14_LAW, NMAX_Q14, -0.00503, 0.912),
        ("S3", "SDR", _MIX_LAW, 96_000, -0.0140, 1.415),
    )
    for study, estimator, law, n, bias, sigma in third:
        config = Config(
            estimator,
            5,
            3,
            law,
            "mix",
            learners="int",
            mechanism="complement",
            complement_nodes=_EVERY,
        )
        out.append(
            Control(
                study,
                "categorical_probability_necessity",
                "third_arm__k5__binary_complement",
                config,
                n,
                bias,
                sigma,
                True,
            )
        )
    policy = (
        ("S1", "LTMLE", 2, law_.L_R, "mix_reversed", 16_000, -0.0511, 1.970, True),
        ("S2", "LTMLE", 3, _MIX_LAW, "mix_node1", 96_000, 0.0334, 2.250, False),
        ("S3", "SDR", 2, law_.L_R, "mix_reversed", 16_000, -0.0511, 1.973, True),
        ("S3", "SDR", 3, _MIX_LAW, "mix_node1", 96_000, 0.0334, 2.251, False),
    )
    for study, estimator, m, law, plan, n, bias, sigma, mc in policy:
        config = Config(estimator, 5, m, law, plan, target=_policy(m))
        out.append(
            Control(
                study, "policy_necessity", f"policy__k5__{plan}_control", config, n, bias, sigma, mc
            )
        )
    rules = (
        ("S1", 5, 2, _RULE_S1, 116_000, 0.0437, 4.950, False),
        ("S2", 3, 3, _RULE_S2, 112_000, 0.0418, 4.637, True),
    )
    for study, K, m, law, n, bias, sigma, mc in rules:
        config = Config("LTMLE", K, m, law, "reversed", target="respond")
        out.append(
            Control(
                study, "rule_necessity", f"respond__k{K}__reversed_rule", config, n, bias, sigma, mc
            )
        )
    out.append(
        Control(
            "S3",
            "sequential_double_robustness",
            "odd_nodes_both_wrong_control",
            pattern("SDR", "respond", True, (1, 3, 5)),
            56_000,
            0.2203,
            9.380,
            True,
        )
    )
    out.append(
        Control(
            "S3",
            "sequential_double_robustness",
            "ltmle_P_odd_control",
            pattern("LTMLE", "respond", True),
            72_000,
            0.0616,
            5.511,
            True,
        )
    )
    return out


def positives() -> list[Positive]:
    """Every gated bias-family positive at its own n."""
    out = []
    for m, plan, K, law, rung, n_positive, *_ in TARGETING:
        study = "S1" if m == 2 else "S2"
        config = Config("LTMLE", K, m, law, plan, learners=RUNGS[rung])
        out.append(
            Positive(study, "targeting_necessity", f"{plan}__k{K}__targeted", config, n_positive)
        )
    for study, m, n in (("S1", 2, 40_000), ("S2", 3, 48_000)):
        first, second = _respond_vs_ref(m)
        arms = {
            "both_correct": ("offset", "known"),
            "outcome_correct": ("offset", "constant"),
            "mechanism_correct": ("glm", "known"),
        }
        for arm, (learner, mechanism) in arms.items():
            config = Config(
                "LTMLE", 5, m, law_.L_R, first, second, learners=learner, mechanism=mechanism
            )
            out.append(
                Positive(study, "double_robustness", f"respond_vs_ref__k5__{arm}", config, n)
            )
    for name, odd in (("P_odd", True), ("P_even", False)):
        out.append(
            Positive(
                "S3", "sequential_double_robustness", name, pattern("SDR", "respond", odd), 56_000
            )
        )
    for study, estimator, m, law, rung, n in (
        ("S1", "LTMLE", 2, _G2, "O3", 8_000),
        ("S2", "LTMLE", 3, _S2_CENSORING, "O4", 60_000),
        ("S3", "SDR", 2, law_.L_R, "O3", 4_000),
        ("S3", "SDR", 3, _MIX_LAW, "O3", 96_000),
    ):
        config = Config(estimator, 5, m, law, _policy(m), learners=RUNGS[rung])
        out.append(Positive(study, "censoring_necessity", f"{_policy(m)}__k5__declared", config, n))
    for study, estimator, law, n in (
        ("S2", "LTMLE", _Q14_LAW, 4_000),
        ("S3", "SDR", _MIX_LAW, 96_000),
    ):
        config = Config(estimator, 5, 3, law, "mix", learners="int")
        out.append(
            Positive(
                study,
                "categorical_probability_necessity",
                "third_arm__k5__assigned_probability",
                config,
                n,
            )
        )
    for study, K, m, law, n in (("S1", 5, 2, _RULE_S1, 4_000), ("S2", 3, 3, _RULE_S2, 4_000)):
        config = Config("LTMLE", K, m, law, "respond")
        out.append(Positive(study, "rule_necessity", f"respond__k{K}__declared_rule", config, n))
    for study, estimator, m, law, n in (
        ("S1", "LTMLE", 2, law_.L_R, 4_000),
        ("S2", "LTMLE", 3, _MIX_LAW, 96_000),
        ("S3", "SDR", 2, law_.L_R, 4_000),
        ("S3", "SDR", 3, _MIX_LAW, 96_000),
    ):
        config = Config(estimator, 5, m, law, _policy(m))
        out.append(Positive(study, "policy_necessity", "policy__k5__declared_policy", config, n))
    return out


def overfitting_positives() -> list[Positive]:
    """The ``crossfit_overfitting`` positives: grown trees, five folds, ``L_R`` at 16,000."""
    return [
        Positive(
            study,
            "crossfit_overfitting",
            "policy_vs_ref__k5__cross_fitted",
            Config(estimator, 5, 3, law_.L_R, "mix", "low", n_folds=5, trees=True),
            16_000,
        )
        for study, estimator in (("S2", "LTMLE"), ("S3", "SDR"))
    ]


# ----------------------------------------------------------------------------- shared declarations
#: The root-n ladder of the ``root_n_and_efficiency`` family; the smallest rung is the control
#: rung, below ``n_kappa``.  At R = 1,000 the control rung carries no claim: its 99 percent
#: Clopper-Pearson interval is narrower than the 0.05 gap between the coverage floor and
#: nominal, so the shipped rule passes it at every true coverage.  Its job is the slope.
ROOT_N_SIZES = (4_000, 12_000, 48_000)
#: The two fitted slope rows of the root-n ladder, by cell suffix.
RATE_KINDS = ("empirical_sd", "reported_se")
#: Replicates per rung, in the property phase and in the rung pilots.
ROOT_N_REPLICATES = 1_000
#: The cross-fitted overfitting cell and its in-sample control.
CROSS_FITTED = "policy_vs_ref__k5__cross_fitted"
IN_SAMPLE = "policy_vs_ref__k5__in_sample_control"
#: The simultaneous band's law, n and plans, per arm count (S1 has m = 2, S2 has m = 3).
BAND = {
    2: (law_.L_R, 76_000, ("never", "always", "respond", "switch", "coin", "coinlast")),
    3: (Law(0.25, 0.0, 3.5), 64_000, ("low", "high", "respond", "switch", "mix", "mixlast")),
}
#: The studies that declare the band.
BAND_STUDIES = ("S1", "S2")
#: Pilot replicates per band, the ladder its property R is picked from, and the pass probability
#: the pick must reach under the shipped two-sided rule.
BAND_REPLICATES = 3_200
BAND_LADDER = (2_000, 3_000, 4_000, 6_000, 8_000, 12_000, 16_000)
BAND_TARGET = 0.99


def study_estimator(study: str) -> str:
    """The estimator a study fits."""
    return "SDR" if study == "S3" else "LTMLE"


def root_n_label(study: str) -> str:
    """The root-n ladder's label."""
    return "static_contrast__k4" if study == "S1" else "respond_vs_ref__k4"


def root_n_config(study: str) -> Config:
    """The estimator of every rung of the root-n ladder of ``study``, on ``L_R`` at K = 4."""
    m = 2 if study == "S1" else 3
    first, second = ("always", "never") if m == 2 else ("respond", "low")
    return Config(study_estimator(study), 4, m, law_.L_R, first, second)


def band_config(study: str) -> tuple[Config, tuple[str, ...], int]:
    """``(config of the first plan, plans, n)`` of the simultaneous band of ``study``."""
    m = 2 if study == "S1" else 3
    law, n, plans = BAND[m]
    return Config("LTMLE", 5, m, law, plans[0]), plans, n


def cell_prefix(study: str, m: int) -> str:
    """The cell prefix of a study that runs both arm counts (S3), empty elsewhere."""
    if study != "S3":
        return ""
    return "binary_" if m == 2 else "categorical_"


def policy_cell(cell: str, m: int) -> str:
    """The design's ``coin__``/``mix__`` cell under the plan's ``policy__`` name."""
    head = f"{'coin' if m == 2 else 'mix'}__"
    return f"policy__{cell[len(head) :]}" if cell.startswith(head) else cell


def bias_cell(study: str, family: str, cell: str, m: int) -> str:
    """The property cell name of a bias-family positive or control."""
    cell = policy_cell(cell, m)
    if family in ("censoring_necessity", "policy_necessity"):
        return cell_prefix(study, m) + cell
    return cell


@cache
def gated_rows() -> tuple[tuple[str, str, str], ...]:
    """Every gated primary truth row: ``(study, gating scenario, estimand)``.

    A primary row is gated only in its gating scenario (:func:`gating`); the same quantity in
    another scenario reports.  A duration MSM scenario gates both coefficients.
    """
    out: dict[tuple[str, str, str], None] = {}
    for row in primary_rows():
        scenario = gating(row)
        if scenario is None:
            continue
        q = row.quantity
        name = (
            primary.mean_name(q.first) if not q.second else primary.contrast_name(q.first, q.second)
        )
        out[(row.study, scenario.name, name)] = None
    for study in STUDIES:
        for scenario in scenarios(study):
            if scenario.msm:
                for name in primary.estimands(scenario):
                    out[(study, scenario.name, name)] = None
    return tuple(out)


@cache
def gated_cells() -> tuple[tuple[str, str, str], ...]:
    """Every property positive and control that has a verdict: ``(study, property, cell)``.

    The reporting cells (``support_loss``, ``learned_mechanism``, the 4,000-unit pattern rows)
    state no verdict and are not listed.  The root-n control rung has a verdict by the shipped
    rule, and is listed.
    """
    from tests.studies.evidence.property_verdicts import CALIBRATION_CONTROL_KINDS
    from tests.studies.evidence.simultaneous import joint_cells

    out: list[tuple[str, str, str]] = []
    for study in STUDIES:
        for cell in calibration_cells():
            if cell.study != study:
                continue
            kinds = ("known_mechanism", *CALIBRATION_CONTROL_KINDS)
            out += [(study, "interval_calibration", f"{cell.name}__{kind}") for kind in kinds]
            if "policy_vs_policylast" in cell.name:
                out.append((study, "interval_calibration", f"{cell.name}__inflated_se_control"))
        label = root_n_label(study)
        out += [(study, "root_n_and_efficiency", f"{label}__n_{n}") for n in ROOT_N_SIZES]
        out += [(study, "root_n_rate", f"{label}__{kind}") for kind in RATE_KINDS]
        items: list[Positive | Control] = [*positives(), *controls()]
        for item in items:
            if item.study == study:
                cell_name = bias_cell(study, item.family, item.cell, item.config.m)
                out.append((study, item.family, cell_name))
        if study == "S1":
            out.append((study, "type_i_error", "policy_vs_policylast__k5__sharp_null"))
            out.append((study, "power", "policy_vs_policylast__k5__alternative"))
        if study in BAND_STUDIES:
            out += [(study, "simultaneous_coverage", cell) for cell in joint_cells("k5")]
        if any(p.study == study for p in overfitting_positives()):
            out += [(study, "crossfit_overfitting", cell) for cell in (CROSS_FITTED, IN_SAMPLE)]
    return tuple(out)


def __getattr__(name: str) -> Any:
    """``GATED_ROWS`` and ``GATED_CELLS``: computed on first use, because the rows take the exact law."""
    if name == "GATED_ROWS":
        return gated_rows()
    if name == "GATED_CELLS":
        return gated_cells()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def red_gated(
    study: str,
    implementation: str,
    performance: pd.DataFrame,
    properties: pd.DataFrame,
) -> list[str]:
    """The red or missing members of ``GATED_ROWS`` and ``GATED_CELLS`` in a study's summaries.

    Parameters
    ----------
    study : str
        ``S1``, ``S2`` or ``S3``.
    implementation : str
        The subject's implementation name in ``performance-tests.csv``.
    performance : pandas.DataFrame
        The study's ``performance-tests.csv``.
    properties : pandas.DataFrame
        The study's ``properties.csv``.

    Returns
    -------
    list of str
        One line per gated truth row or property cell that is red or absent; empty when all pass.
    """
    out = []
    subject = performance.loc[performance["implementation"] == implementation]
    truth = {
        (scenario, estimand): bool(passed)
        for scenario, estimand, passed in zip(
            subject["scenario"], subject["estimand"], subject["passed"], strict=True
        )
    }
    for tag, scenario, estimand in gated_rows():
        if tag != study:
            continue
        verdict = truth.get((scenario, estimand))
        if verdict is None:
            out.append(f"truth row {scenario} {estimand}: absent")
        elif not verdict:
            out.append(f"truth row {scenario} {estimand}: red")
    cells = {
        (family, cell): bool(passed)
        for family, cell, passed in zip(
            properties["property"], properties["cell"], properties["passed"], strict=True
        )
    }
    for tag, family, cell in gated_cells():
        if tag != study:
            continue
        verdict = cells.get((family, cell))
        if verdict is None:
            out.append(f"property cell {family}/{cell}: absent")
        elif not verdict:
            out.append(f"property cell {family}/{cell}: red")
    return out


def smallest_n(reading: sampling.Reading, start: int, cap: int | None = None) -> int | None:
    """The smallest multiple of :data:`STEP` at least ``start`` with ``d >= 3``."""
    if reading.bias == 0.0:
        return None
    need = round_n(max(start, (D_FLOOR * reading.sigma / abs(reading.bias)) ** 2))
    return need if cap is None or need <= cap else None


# ----------------------------------------------------------------------------- sizing floors
def _calibration_pass(rho: Any, replicates: int, scale: float, coverage: Any) -> Any:
    """:func:`calibration_pass` over arrays: ``rho`` and ``coverage`` broadcast together."""
    z = norm.ppf(0.995)
    spread = scale * math.sqrt(400 / replicates)
    half = z * spread
    ratio = np.maximum(
        0.0, norm.cdf((1.07 - half - rho) / spread) - norm.cdf((0.93 + half - rho) / spread)
    )
    cover = 2 * norm.cdf(1.96 * rho) - 1 if coverage is None else coverage
    se = np.sqrt(cover * (1 - cover) / replicates)
    width = z * se
    inside = np.maximum(
        0.0, norm.cdf((0.98 - width - cover) / se) - norm.cdf((0.92 + width - cover) / se)
    )
    return ratio * inside


def calibration_pass(
    rho: float,
    replicates: int,
    scale: float = SE_RATIO_SCALE,
    *,
    coverage: float | None = None,
) -> float:
    """The pass probability of one calibration positive at true SE ratio ``rho``.

    The SE-ratio estimate is normal with SD ``scale sqrt(400 / R)``; its 99 percent interval
    must lie in [0.93, 1.07].  The coverage count is binomial; its 99 percent interval must lie
    in [0.92, 0.98].  ``coverage`` is the arm's true coverage.  Without it the Wald model
    ``2 Phi(1.96 rho) - 1`` stands in, which is the F1 reading before any arm has run.  The
    cloud pilots read coverage up to 0.6 points below that model (review minor 2), so F2 passes
    each arm's measured coverage.
    """
    return float(_calibration_pass(rho, replicates, scale, coverage))


def f1_replicates(cells: int) -> int:
    """F1: the smallest multiple of 400 at which every cell passes with probability 0.90 and
    the family of ``cells`` passes with probability 0.95, all true ratios 1."""
    for replicates in range(400, 16_001, 400):
        single = calibration_pass(1.0, replicates)
        if single >= 0.90 and single**cells >= 0.95:
            return replicates
    raise ValueError("no R up to 16,000 meets F1")


def bias_pass(bias_sd: float, replicates: int) -> float:
    """The pass probability of a bias positive: the 99 percent interval inside +-0.25 SD."""
    half = norm.ppf(0.995) / math.sqrt(replicates)
    spread = 1 / math.sqrt(replicates)
    return max(
        0.0,
        norm.cdf((0.25 - half - bias_sd) / spread) - norm.cdf((-0.25 + half - bias_sd) / spread),
    )


def truth_row_pass(coverage: float, bias_sd: float, replicates: int) -> float:
    """A truth row: bias interval inside 0.25 SD and coverage lower limit at least 0.90."""
    se = math.sqrt(coverage * (1 - coverage) / replicates)
    covered = norm.cdf((coverage - norm.ppf(0.995) * se - 0.90) / se)
    return bias_pass(bias_sd, replicates) * covered


def _clopper_pearson(
    counts: np.ndarray, trials: int, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """The exact two-sided binomial limits of every count, as ``inference.clopper_pearson``."""
    from scipy.stats import beta

    tail = (1.0 - level) / 2.0
    with np.errstate(invalid="ignore"):
        low = np.where(counts > 0, beta.ppf(tail, np.maximum(counts, 1), trials - counts + 1), 0.0)
        high = np.where(
            counts < trials,
            beta.ppf(1.0 - tail, counts + 1, np.maximum(trials - counts, 1)),
            1.0,
        )
    return low, high


def band_pass(coverage: float, replicates: int) -> float:
    """The pass probability of the simultaneous band under the shipped two-sided rule.

    ``simultaneous_coverage_verdicts`` passes the band when its 99 percent exact interval of
    joint coverage lies inside ``calibration_coverage`` = (0.92, 0.98).  The count of replicates
    with every truth inside the band is binomial at the band's true joint ``coverage``.  This is
    not the one-sided ``coverage_floor`` rule: at R = 2,000 it passes with probability 0.99 only
    for a true coverage in about [0.946, 0.954].
    """
    from scipy.stats import binom

    from tests.studies.evidence.registry import Margins

    margins = Margins()
    low_edge, high_edge = margins.calibration_coverage
    counts = np.arange(replicates + 1)
    low, high = _clopper_pearson(counts, replicates, margins.confidence_level)
    passes = (low >= low_edge) & (high <= high_edge)
    return float(np.sum(binom.pmf(counts, replicates, coverage)[passes]))


def band_replicates(coverage: float, ladder: Sequence[int] = BAND_LADDER) -> int | None:
    """The first rung of ``ladder`` at which the band passes with probability :data:`BAND_TARGET`.

    ``None`` when no rung reaches it: the row stops and is diagnosed (plan section 5.3).
    """
    return next((R for R in ladder if band_pass(coverage, R) >= BAND_TARGET), None)


def positive_rung_pass(coverage: float, replicates: int) -> float:
    """The coverage clause of a positive root-n rung: the exact lower limit at the floor.

    Plan section 5.3: at R = 1,000 this passes with probability 0.99 when the true coverage is
    at least 0.9394.
    """
    from scipy.stats import binom

    from tests.studies.evidence.registry import Margins

    margins = Margins()
    counts = np.arange(replicates + 1)
    low, _ = _clopper_pearson(counts, replicates, margins.confidence_level)
    return float(np.sum(binom.pmf(counts, replicates, coverage)[low >= margins.coverage_floor]))


def slope_pass(
    rungs: Mapping[int, pd.DataFrame],
    statistic: str,
    replicates: int,
    *,
    bootstrap: int = 2_000,
    draws: int = 200,
    seed: int = 0,
) -> float:
    """The pass probability of a ``root_n_rate`` row, from pilots of each rung.

    Each draw resamples ``replicates`` rows of every rung's pilot, with replacement, and fits the
    slope with the shipped :func:`tests.studies.evidence.properties.rate` and its percentile
    interval.  The row passes when the interval lies within 0.125 of -0.5 and excludes -0.25
    (``property_verdicts``).  The fraction of ``draws`` that pass is the pass probability;
    ``bootstrap`` is the draw count of each percentile interval.

    Parameters
    ----------
    rungs : mapping of int to pandas.DataFrame
        The pilot rows of each rung by n, with ``estimate``, ``std_error`` and ``truth``.
    statistic : str
        ``spread`` (the empirical SD) or ``reported`` (the mean standard error).
    replicates : int
        The property phase's replicates per rung.
    bootstrap : int
        Draws of each slope interval.
    draws : int
        Resampled ladders.
    seed : int
        The seed of the resampling.
    """
    from tests.studies.evidence import property_verdicts as pv
    from tests.studies.evidence.properties import rate
    from tests.studies.evidence.registry import Margins

    margins = Margins()
    rng = np.random.default_rng(seed)
    passed = 0
    for draw in range(draws):
        frames = []
        for n, pilot in sorted(rungs.items()):
            pick = rng.integers(0, len(pilot), size=replicates)
            chosen = pilot.iloc[pick][["estimate", "std_error", "truth"]].reset_index(drop=True)
            frames.append(
                chosen.assign(property="root_n_and_efficiency", cell=f"ladder__n_{n}", n=n)
            )
        fitted = rate(
            pd.concat(frames, ignore_index=True),
            property_name="root_n_and_efficiency",
            statistic=statistic,
            bootstrap_replicates=bootstrap,
            confidence_level=margins.confidence_level,
            seed=int(rng.integers(0, 2**31)) + draw,
        )
        passed += bool(
            fitted.equivalent_to(pv.ROOT_N_SLOPE, pv.ROOT_N_SLOPE_MARGIN)
            and fitted.excludes(pv.EXCLUDED_SLOPE)
        )
    return passed / draws


def f2_family_pass(
    ratios: Sequence[float],
    variances: Sequence[float],
    replicates: int,
    *,
    coverages: Sequence[float] | None = None,
    coverage_errors: Sequence[float] | None = None,
    draws: int = 4_000,
    seed: int = 0,
) -> float:
    """F2: the posterior predictive probability that every arm passes at ``replicates``.

    The random-effects model ``r_j ~ N(rho_j, v_j)``, ``rho_j ~ N(mu, tau^2)`` with REML
    ``tau^2``; ``mu`` normal at its REML estimate.  With ``coverages`` each arm's coverage is
    its measured pilot coverage, drawn normal with the arm's binomial error
    ``coverage_errors`` and clipped to the open unit interval; without it the arm's coverage is
    the Wald model of :func:`calibration_pass`.
    """
    r, v = np.asarray(ratios, float), np.asarray(variances, float)
    tau2 = _reml_tau2(r, v)
    total = 1.0 / np.sum(1.0 / (v + tau2))
    mu_hat = total * np.sum(r / (v + tau2))
    rng = np.random.default_rng(seed)
    mu = rng.normal(mu_hat, math.sqrt(total), size=draws)
    shrink = tau2 / (tau2 + v) if tau2 > 0 else np.zeros_like(v)
    mean = shrink[None, :] * r[None, :] + (1 - shrink)[None, :] * mu[:, None]
    spread = np.sqrt(tau2 * v / (tau2 + v)) if tau2 > 0 else np.zeros_like(v)
    rho = mean + spread[None, :] * rng.standard_normal((draws, r.size))
    if coverages is None:
        passes = _calibration_pass(rho, replicates, SE_RATIO_SCALE, None)
    else:
        if coverage_errors is None or len(coverages) != r.size or len(coverage_errors) != r.size:
            raise ValueError("coverages and coverage_errors need one value per arm")
        cover = np.asarray(coverages, float)[None, :] + np.asarray(coverage_errors, float)[
            None, :
        ] * rng.standard_normal((draws, r.size))
        cover = np.clip(cover, 1e-6, 1 - 1e-6)
        passes = _calibration_pass(rho, replicates, SE_RATIO_SCALE, cover)
    return float(np.mean(np.prod(passes, axis=1)))


def _reml_tau2(r: np.ndarray, v: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def negative(tau2: float) -> float:
        w = 1.0 / (v + tau2)
        mu = np.sum(w * r) / np.sum(w)
        return float(
            0.5 * (np.sum(np.log(v + tau2)) + np.log(np.sum(w)) + np.sum(w * (r - mu) ** 2))
        )

    fit = minimize_scalar(negative, bounds=(0.0, 1.0), method="bounded")
    return float(fit.x) if negative(float(fit.x)) < negative(0.0) else 0.0


#: The n of the S1 ``type_i_error`` and ``power`` cells (``coin`` against ``coinlast``, K = 5,
#: ``L_R``): the power on ``D_inf`` reads 0.899 at 48,000, below the 0.90 the rule requires.
POWER_N = 52_000


def power_n(delta: float, sigma: float, power: float = 0.90) -> float:
    """The n at which a two-sided 5 percent test of ``delta`` has ``power``."""
    return (norm.ppf(0.975) + norm.ppf(power)) ** 2 * (sigma / delta) ** 2


# ----------------------------------------------------------------------------- cost model
#: Per row and node, in microseconds: one fit's data, mechanism columns and reference plan.
COST_BASE = 2.7
#: Per extra plan in one fit.
COST_PLAN = {"static": 0.6, "rule": 1.25, "policy2": 2.1, "policy3": 8.6}
#: The five-fold factor per estimator, and grown trees.
COST_FOLDS = {"LTMLE": 5.0 / 3.4, "SDR": 5.9 / 3.3, "trees": 6.0 / 3.4}
#: Per row of a K = 5 draw.
COST_DRAW = 2.0


def _plan_cost(plan: str, m: int, K: int) -> float:
    if plan in ("mixlast", "coinlast"):
        return 0.8 * COST_PLAN[f"policy{m}"]
    if law_.policy_nodes(plan, K):
        return COST_PLAN[f"policy{m}"]
    if plan in ("respond", "reversed", "lagged"):
        return COST_PLAN["rule"]
    return COST_PLAN["static"]


def fit_cost(
    K: int, m: int, plans: Sequence[str], n: int, *, folds: float = 1.0, calls: int = 1
) -> float:
    """CPU seconds of one replicate: ``1.5 n K (calls f (b + sum u_j) + d / 5)`` microseconds."""
    # ``COST_BASE`` carries one static reference plan; every plan of a call adds its own rate.
    extra = sum(_plan_cost(plan, m, K) for plan in plans) / calls - COST_PLAN["static"]
    per = calls * folds * (COST_BASE + max(extra, 0.0)) + COST_DRAW / 5
    return 1.5 * n * K * per * 1e-6


def config_cost(config: Config, n: int) -> float:
    """CPU seconds of one replicate of ``config`` at ``n``."""
    plans = (config.first, config.second) if config.second else (config.first,)
    if config.msm:
        plans = tuple(f"dur{j}" for j in range(config.K + 1))
    folds = 1.0
    if config.trees:
        folds = COST_FOLDS["trees"]
    elif config.n_folds > 1:
        folds = COST_FOLDS[config.estimator]
    calls = 2 if config._separate() else 1
    return fit_cost(config.K, config.m, plans, n, folds=folds, calls=calls)


# ----------------------------------------------------------------------------- pilot seeds
def stream_seed(stream: str, replicate: int) -> int:
    """The seed of replicate ``replicate`` of the pilot stream ``stream``."""
    digest = int(hashlib.sha256(stream.encode()).hexdigest()[:8], 16)
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(digest, replicate))
    return int(sequence.generate_state(1)[0])


@dataclass(frozen=True)
class Stream:
    """One pilot stream: a configuration at one n, ``replicates`` draws.

    Parameters
    ----------
    family : str
        The pilot family: ``mc``, ``f2``, ``bias``, ``truth``, ``overfit``, ``band`` or ``rung``.
    name : str
        The study and cell.
    config : Config or None
        The estimator; ``None`` for a truth-row stream, which fits a primary scenario.
    n : int
        The sample size.
    replicates : int
        The stream's replicate count.
    scenario : tuple
        ``(study, scenario name)`` of a truth-row stream.
    plans : tuple of str
        The plans of one joint fit, for the ``band`` stream; empty elsewhere.
    """

    family: str
    name: str
    config: Config | None
    n: int
    replicates: int
    scenario: tuple[str, str] = field(default=("", ""))
    plans: tuple[str, ...] = ()

    @property
    def ident(self) -> str:
        """The checkpoint cell: study, cell, law, K, m, n, learner rung and stream."""
        what = self.config.ident() if self.config is not None else "|".join(self.scenario)
        return f"{self.family}|{self.name}|n{self.n}|{what}"

    def cost(self) -> float:
        """CPU seconds of one replicate."""
        if self.plans and self.config is not None:
            return fit_cost(self.config.K, self.config.m, self.plans, self.n)
        if self.config is not None:
            return config_cost(self.config, self.n)
        study, name = self.scenario
        scenario = next(s for s in scenarios(study) if s.name == name)
        folds = COST_FOLDS["LTMLE" if study != "S3" else "SDR"] if scenario.n_folds > 1 else 1.0
        return fit_cost(scenario.K, scenario.m, scenario.plans, self.n, folds=folds)


def streams() -> list[Stream]:
    """Every pilot stream, in the declared order.

    Monte Carlo, F2, bias, truth rows and overfitting come first, in the order of rev 4.1.  The
    band and the root-n rung pilots of rev 4.2 follow.
    """
    out = [
        Stream("mc", f"{c.study}:{c.family}:{c.cell}", c.config, c.n, 400)
        for c in controls()
        if c.monte_carlo
    ]
    out += [
        Stream("f2", f"{c.study}:{c.name}", c.config(), c.n, 3_200) for c in calibration_cells()
    ]
    out += [
        Stream("bias", f"{p.study}:{p.family}:{p.cell}", p.config, p.n, 1_000) for p in positives()
    ]
    for study in STUDIES:
        for scenario in scenarios(study):
            out.append(
                Stream(
                    "truth",
                    f"{study}:{scenario.name}",
                    None,
                    scenario.n,
                    400,
                    (study, scenario.name),
                )
            )
    out += [
        Stream("overfit", f"{p.study}:{p.cell}", p.config, p.n, 3_200)
        for p in overfitting_positives()
    ]
    # Rev 4.2 families.  They come after the rev 4.1 streams, so that no earlier stream moves.
    for study in BAND_STUDIES:
        config, plans, n = band_config(study)
        out.append(Stream("band", f"{study}:band_k5", config, n, BAND_REPLICATES, plans=plans))
    for study in STUDIES:
        label = root_n_label(study)
        config = root_n_config(study)
        out += [
            Stream("rung", f"{study}:{label}:n_{n}", config, n, ROOT_N_REPLICATES)
            for n in ROOT_N_SIZES
        ]
    return out


# ----------------------------------------------------------------------------- oracle curve
@dataclass(frozen=True)
class OracleReading:
    """The oracle-curve prediction of one quantity at one n.

    Parameters
    ----------
    se_ratio : float
        The mean plug-in SE over the SD of the estimate, across the draws.
    coverage : float
        The Wald coverage of a 95 percent interval.
    mc_sd : float
        The normal-theory Monte Carlo SD of ``se_ratio``, ``1 / sqrt(2 * draws)``.
    """

    se_ratio: float
    coverage: float
    mc_sd: float


def limit_curve(q: Quantity, law: Law) -> np.ndarray:
    """``D_inf`` of ``q``: the declared GLM limit curve, per support row."""
    return _difference({p: glm_curve(q.K, q.m, p, law) for p in q.plans}, q)


def oracle_reading(
    mass: np.ndarray, curve: np.ndarray, n: int, draws: int, seed: int = PILOT_SEED
) -> OracleReading:
    """An ideal linear estimator with ``curve``: the SE ratio and coverage it predicts at ``n``.

    Each draw is ``n`` units from the law (a multinomial over the support).  The estimate is
    ``mean(curve)`` and its SE the plug-in ``sd(curve) / sqrt(n)``.  This is a property of the law
    and ``n`` alone: the oracle curve of plan section 5.1, with no fit.
    """
    p = np.asarray(mass, dtype=float) / np.sum(mass)
    rng = np.random.default_rng(seed)
    estimate = np.empty(draws)
    se = np.empty(draws)
    chunk = max(1, int(2.5e7 // p.size))
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        counts = rng.multinomial(n, p, size=count)
        mean = counts @ curve / n
        estimate[start : start + count] = mean
        se[start : start + count] = np.sqrt(np.maximum(counts @ curve**2 / n - mean**2, 0.0) / n)
    return OracleReading(
        se_ratio=float(se.mean() / estimate.std(ddof=1)),
        coverage=float(np.mean(np.abs(estimate) <= norm.ppf(0.975) * se)),
        mc_sd=1.0 / math.sqrt(2 * draws),
    )


def control_rung_pass(coverage: float, replicates: int, *, alpha: float = 0.05) -> float:
    """The pass probability of a root-n control rung under the shipped rule.

    The rule at ``tests/studies/evidence/property_verdicts.py`` (the ``small`` branch of the
    efficiency summary) passes the rung when its exact 99 percent coverage interval lies wholly
    below ``1 - alpha``, or when its lower limit is at least the coverage floor 0.90.  The
    coverage count is binomial at ``coverage``; the interval is Clopper-Pearson.
    """
    from scipy.stats import binom

    from tests.studies.evidence.inference import clopper_pearson
    from tests.studies.evidence.registry import Margins

    margins = Margins()
    counts = np.arange(replicates + 1)
    passes = np.array(
        [
            (
                (interval := clopper_pearson(int(k), replicates, confidence_level=0.99)).high
                < 1.0 - alpha
            )
            or interval.low >= margins.coverage_floor
            for k in counts
        ]
    )
    return float(np.sum(binom.pmf(counts, replicates, coverage)[passes]))


def noise_control_ratio(sigma_eff: float, sigma_inf: float) -> float:
    """The expected SE ratio of ``noise_control``: ``(1 + sigma_eff^2 / sigma_inf^2)^(-1/2)``."""
    return (1.0 + (sigma_eff / sigma_inf) ** 2) ** -0.5


def calibration_sigmas(cell: CalibrationCell) -> tuple[float, float]:
    """``(sigma_eff, sigma_inf)`` of a calibration cell: the SDs of ``D*`` and of ``D_inf``."""
    tail_ = cell.tail()
    q = cell.quantity
    w = weight(q.K, q.m, cell.law)
    if not cell.msm:
        return tail_.sigma, sampling.sd(w, limit_curve(q, cell.law))
    plans = [f"dur{j}" for j in range(q.K + 1)]
    design = np.column_stack([np.ones(q.K + 1), np.arange(q.K + 1)])
    row = np.linalg.solve(design.T @ design, design.T)[1]
    limit = np.column_stack([glm_curve(q.K, 2, p, cell.law) for p in plans]) @ row
    return tail_.sigma, sampling.sd(w, limit)
