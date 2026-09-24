r"""Builders for the tests of a declared treatment rule and a user-written intervention.

``tests/unit/test_rule_and_intervention_declarations.py`` (RM28) tests the rule declaration
of ``Rule`` and the density declaration of a user-written ``Intervention``.
``tests/unit/test_regimen_rule_declarations.py`` (RM28) tests the same rule declaration on
the callable nodes of a ``DynamicRegimen``.  ``tests/unit/test_stochastic_regime_densities.py``
lists all three among the users of the shared check.  This module holds the builders of
those files, and not all of them are shared:

* all three files use the unknown-value fragments, ``threshold_rule``,
  ``threshold_regimen`` and ``DataTilt``;
* both RM28 files use the threshold rules and the grid of the threshold law, after Luedtke
  and van der Laan (2016);
* only the rule file uses the other user-written classes, and the one-node law with its
  closed forms and fit;
* only the regimen file uses the two-node law with its closed forms and fit, and every
  longitudinal fit entry that can reach a regimen, with ``NeverFit`` learners.

The rules and classes are module-level, so each one pickles, and each class counts its
``density`` calls in ``calls``.  Each law sits beside its closed forms, so a pin and the
value it pins are written in one place.

The threshold law is :math:`W \sim U(0, 1)`, :math:`A \mid W \sim \mathrm{Bern}(1/2)`, and
:math:`Y = 3W + A + \varepsilon` with :math:`\varepsilon = \pm 1/4`.  The learned rule is
:math:`d(w) = 1\{w \le c\}`, where :math:`c` is the sample mean of :math:`W`.  The regime
value :math:`\psi(c) = 3/2 + c` has slope :math:`f(c)\{Q(1, c) - Q(0, c)\} = 1`, so the
curve of the population-indexed rule carries the term :math:`T = W - c` that the
fixed-rule curve :math:`D` omits.  The exact moments are :math:`E[D^2] = 3/8`,
:math:`E[T^2] = 1/12` and :math:`E[DT] = 1/8`, so the reported standard error is
:math:`3/\sqrt{17}` of the exact one.  RM28 in ``docs/roadmap.md`` declares these values.

The sample is the midpoint grid :math:`W_j = (j - 1/2)/200`, with four rows at each point:
:math:`A \in \{0, 1\}` and :math:`\varepsilon = \pm 1/4`.  Each :math:`(W, A)` cell balances
its residuals, so the cell means are the true :math:`Q`, :math:`g = 1/2`, and the targeting
step moves nothing.  The tilt law of the user-written class is in
``tests/unit/_tilt_law_support.py``.

The two-node law draws :math:`A_1` and :math:`A_2` from :math:`\mathrm{Bern}(1/2)`, with
:math:`Y = 3W + A_2 + \varepsilon`.  The regimen :math:`(1, d)` has the fixed-rule curve
:math:`D = 4 \cdot 1\{A_1 = 1, A_2 = d(W)\}\varepsilon + 3W + d(W) - \psi`, whose node-one
term is zero, so :math:`E[D^2] = 1/2` and the ratio is :math:`\sqrt{3/5}`.  Its grid holds
eight rows at each point.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad

from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
from cleverly.estimators import TMLE
from cleverly.interventions import Rule
from cleverly.interventions.base import refuse_regime_densities
from cleverly.longitudinal import LTMLE, DynamicRegimen, ltmle
from tests.discrete_law_longitudinal import CellMeans
from tests.unit._declaration_support import PANEL_COLUMNS, never_fit_longitudinal_learners, panel
from tests.unit._tilt_law_support import DGP, odds_tilt

#: Fragments of the two unknown-value refusals.  The density declaration of a user-written
#: class shares its field name with ``Stochastic``, so its fragment is the meaning sentence.
RULE_UNKNOWN = "rule_kind must be 'known', 'estimated' or None"
INTERVENTION_UNKNOWN = "of a user-written Intervention is a fixed function"

# ------------------------------------------------------------------ user-written classes


def strata(data: Any) -> np.ndarray:
    """The stratum ``W`` of each row of the tilt law, from the covariates of the fit."""
    return np.rint(np.asarray(data.covariates, dtype=float)[:, 0]).astype(int)


@dataclass(eq=False)
class KnownUserTilt:
    """A user-written class: the odds tilt of the law's true mechanism, a fixed function of ``W``.

    ``density_kind`` is a field, so a test states the declaration it builds.
    """

    delta: float = 2.0
    density_kind: Any = None
    name: str = "tilt"
    calls: int = 0

    def mechanism(self, data: Any) -> np.ndarray:
        """``P(A = 1 | W = w)`` for ``w = 0, 1, 2``."""
        return np.asarray(DGP.g, dtype=float)

    def density(self, data: Any) -> np.ndarray:
        self.calls += 1
        return odds_tilt(self.mechanism(data), self.delta)[strata(data)]


@dataclass(eq=False)
class DataTilt(KnownUserTilt):
    """The user lie: ``density`` reads the data of the fit and tilts the sample treated share.

    On the exact tilt sample the share equals the true mechanism, so the density equals the
    one of :class:`KnownUserTilt`.  Only the declared status differs.
    """

    def mechanism(self, data: Any) -> np.ndarray:
        w, a = strata(data), np.asarray(data.treatment, dtype=float)
        return np.array([a[w == k].mean() for k in range(3)])


@dataclass(eq=False)
class BareTilt:
    """A user-written class with no ``density_kind`` at all, as one written before RM28."""

    delta: float = 2.0
    name: str = "tilt"
    calls: int = 0

    def density(self, data: Any) -> np.ndarray:
        self.calls += 1
        return odds_tilt(np.asarray(DGP.g, dtype=float), self.delta)[strata(data)]


def checked(item: Any) -> Any:
    """``item`` after :func:`refuse_regime_densities` checks it: a user class has no build check."""
    refuse_regime_densities((item,))
    return item


# ------------------------------------------------------------------ the threshold rules

#: The fixed threshold, and the sample mean of ``W`` on the grid.
CENTRE = 0.5


@dataclass(frozen=True)
class FixedThreshold:
    """``d(w) = 1{w <= centre}``, with the centre fixed before the fit: a known rule.

    ``column`` names the covariate ``w`` it reads: ``W`` on the threshold law, and ``W1``
    on :func:`tests.unit._declaration_support.panel`.
    """

    centre: float = CENTRE
    column: str = "W"

    def __call__(self, frame: Any) -> np.ndarray:
        return np.where(np.asarray(frame[self.column], dtype=float) <= self.centre, 1, 0)


@dataclass(frozen=True)
class SampleThreshold(FixedThreshold):
    """The user lie: the threshold at the sample mean of ``W``, a statistic of the sample."""

    @classmethod
    def of(cls, frame: Any) -> SampleThreshold:
        return cls(float(np.mean(np.asarray(frame["W"], dtype=float))))


def threshold_rule(rule: Any = None, **declaration: Any) -> Rule:
    """A ``Rule`` named ``"thr"``, the fixed threshold unless ``rule`` is given."""
    return Rule(FixedThreshold() if rule is None else rule, "thr", **declaration)


def threshold_regimen(rule: Any = None, **declaration: Any) -> DynamicRegimen:
    """The regimen ``(1, d)`` labelled ``"thr"``, the fixed threshold unless ``rule`` is given."""
    return DynamicRegimen("thr", (1, FixedThreshold() if rule is None else rule), **declaration)


# ------------------------------------------------------------------ the threshold law

GRID_SIZE = 200
NOISE = 0.25
#: The midpoint grid: its sample moments are exact but for ``Var_n(W) = 1/12 - 1/(12 m^2)``.
GRID_W = (np.arange(1, GRID_SIZE + 1) - 0.5) / GRID_SIZE

#: The exact ratio of the reported to the exact standard error, and its pin and bound.  The
#: plan measured 0.7276058 before it chose them (RM28 in docs/roadmap.md).  A curve that
#: carried ``T`` would read 1.  The bound claims an understatement of more than 20 percent
#: and pins no digit.  ``_tilt_law_support.UNDERSTATEMENT_BOUND`` is the bound of another
#: witness, so this one carries the law in its name.
EXACT_RATIO = 3.0 / np.sqrt(17.0)
RATIO_PIN = 0.7276
THRESHOLD_UNDERSTATEMENT_BOUND = 0.8
#: The same three for the regimen ``(1, d)`` on the two-node law.  The plan measured
#: 0.7745976 before it chose them.  The bound claims an understatement of more than 15
#: percent.
REGIMEN_EXACT_RATIO = float(np.sqrt(3.0 / 5.0))
REGIMEN_RATIO_PIN = 0.7746
REGIMEN_UNDERSTATEMENT_BOUND = 0.85
#: The exact value of the regime at the fixed threshold, and of the regimen ``(1, d)``.
PSI = 2.0


def threshold_frame(nodes: int = 1) -> pd.DataFrame:
    """The rows of the threshold law: each arm at each node, and both noise signs, at each point.

    One node gives the 800 rows of the point law, with columns ``W``, ``A`` and ``Y``.  Two
    give the 1600 rows of the two-node law, with columns ``W``, ``A1``, ``A2`` and ``Y``.
    ``Y`` reads the arm of the last node.
    """
    arms = ["A"] if nodes == 1 else [f"A{time}" for time in range(1, nodes + 1)]
    rows = [
        (w, *(float(arm) for arm in path), 3.0 * w + path[-1] + e)
        for w in GRID_W
        for path in itertools.product((0, 1), repeat=nodes)
        for e in (-NOISE, NOISE)
    ]
    return pd.DataFrame(rows, columns=["W", *arms, "Y"])


def regime_value(centre: float) -> float:
    """``psi(c) = int_0^1 Q(1{w <= c}, w) dw``, by quadrature."""
    value, _ = quad(lambda w: 3.0 * w + float(w <= centre), 0.0, 1.0, points=[centre])
    return float(value)


def fixed_rule_curve(frame: pd.DataFrame, centre: float = CENTRE) -> np.ndarray:
    """``D = 2 1{A = d(W)} (Y - Q(A, W)) + Q(d(W), W) - psi``, the fixed-rule curve."""
    w, a, y = (np.asarray(frame[column], dtype=float) for column in ("W", "A", "Y"))
    rule = (w <= centre).astype(float)
    return 2.0 * (a == rule) * (y - (3.0 * w + a)) + 3.0 * w + rule - PSI


def threshold_term(frame: pd.DataFrame, centre: float = CENTRE) -> np.ndarray:
    """``T = (dpsi/dc) IF_c = W - c``, the term the fixed-rule curve omits."""
    return np.asarray(frame["W"], dtype=float) - centre


def exact_rule_curve(frame: pd.DataFrame) -> np.ndarray:
    """``D + T``: the curve of the population-indexed rule, from the closed forms alone."""
    return fixed_rule_curve(frame) + threshold_term(frame)


def plugin_se(curve: np.ndarray) -> float:
    """``sqrt(Var_n(curve) / n)`` with ``ddof=1``: the unclustered plug-in standard error.

    Written out here, not imported, so a witness can tie the reported ``std_error`` to the
    reported curve by a second computation.
    """
    curve = np.asarray(curve, dtype=float)
    return float(np.sqrt(np.var(curve, ddof=1) / curve.shape[0]))


def threshold_fit(rule: Rule) -> Any:
    """``rule`` fitted in sample on the threshold law, with ``CellMeans`` nuisances."""
    estimator = TMLE(
        outcome_learner=CellMeans(),
        treatment_learner=CellMeans(),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
        interventions=(rule,),
    )
    return estimator.fit(threshold_frame(), outcome="Y", treatment="A").single()


def threshold_curve(result: Any) -> np.ndarray:
    return np.asarray(result.estimates["ey_regime[thr]"].influence_curve, dtype=float)


# ------------------------------------------------------------------ the two-node law


def fixed_regimen_curve(frame: pd.DataFrame, centre: float = CENTRE) -> np.ndarray:
    """``D = 4 1{A1 = 1, A2 = d(W)} (Y - Q2) + Q2(1, d(W), W) - psi``, the fixed-rule curve."""
    w, a1, a2, y = (np.asarray(frame[column], dtype=float) for column in ("W", "A1", "A2", "Y"))
    rule = (w <= centre).astype(float)
    followed = (a1 == 1.0) & (a2 == rule)
    return 4.0 * followed * (y - (3.0 * w + a2)) + 3.0 * w + rule - PSI


def exact_regimen_curve(frame: pd.DataFrame) -> np.ndarray:
    """``D + T`` for the regimen ``(1, d)``, from the closed forms alone."""
    return fixed_regimen_curve(frame) + threshold_term(frame)


#: The columns of the two-node law, as ``LTMLE.fit`` reads them.
THRESHOLD_COLUMNS: dict[str, Any] = {"outcome": "Y", "treatment": ["A1", "A2"], "baseline": ["W"]}


def regimen_fit(
    regimens: Any, frame: pd.DataFrame | None = None, *, id: str | None = None, **settings: Any
) -> Any:
    """``regimens`` fitted in sample on ``frame``, with ``CellMeans`` in every learner slot.

    ``regimens`` is any ``regimens=`` value, and ``frame`` is the two-node law unless it is
    given.  ``settings`` replace the ``LTMLE`` settings here, and ``id`` names a cluster
    column of ``frame``.
    """
    estimator = LTMLE(
        regimens,
        **{
            "outcome_learner": CellMeans(),
            "pseudo_learner": CellMeans(),
            "treatment_learner": CellMeans(),
            "censoring_learner": CellMeans(),
            "n_folds": 1,
            "simultaneous": False,
            "max_iter": 100,
            "tol": 1e-10,
            "random_state": 0,
            **settings,
        },
    )
    data = threshold_frame(2) if frame is None else frame
    return estimator.fit(data, **THRESHOLD_COLUMNS, id=id)


def regimen_curve(result: Any) -> np.ndarray:
    return np.asarray(result.estimates["ey_regimen[thr]"].influence_curve, dtype=float)


# ------------------------------------------------------------------ the longitudinal entries


def longitudinal_entries() -> dict[str, Callable[[Any], Any]]:
    """Every longitudinal fit entry that can reach a regimen, with ``NeverFit`` learners.

    Each entry takes the raw ``regimens=`` value and fits :func:`panel` on
    :data:`PANEL_COLUMNS`: ``"fit"`` is ``LTMLE.fit``, ``"study"`` is
    ``CausalStudy.estimate`` on a ``LongitudinalTreatment`` design, and ``"ltmle"`` is the
    one-call :func:`cleverly.longitudinal.ltmle`.  Each entry resets ``NeverFit.calls``.
    """

    def fit(regimens: Any) -> Any:
        estimator = LTMLE(regimens, n_folds=1, **never_fit_longitudinal_learners())
        return estimator.fit(panel(), **PANEL_COLUMNS)

    def study(regimens: Any) -> Any:
        design = LongitudinalTreatment(**PANEL_COLUMNS)
        return (
            CausalStudy(panel(), design=design)
            .identify(RegimeMean(regimens))
            .estimate(cross_fit=False, **never_fit_longitudinal_learners())
        )

    def one_call(regimens: Any) -> Any:
        learners = never_fit_longitudinal_learners()
        return ltmle(panel(), regimens=regimens, n_folds=1, **learners, **PANEL_COLUMNS)

    return {"fit": fit, "study": study, "ltmle": one_call}
