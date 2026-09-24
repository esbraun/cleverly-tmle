r"""Builders for the tests of a declared treatment rule and a user-written intervention.

``tests/unit/test_rule_and_intervention_declarations.py`` (RM28) tests the rule declaration
of ``Rule`` and the density declaration of a user-written ``Intervention``.
``tests/unit/test_stochastic_regime_densities.py`` lists both among the users of the shared
check.  This module holds what those files share:

* module-level rules and user-written classes, so each one pickles, and each class counts
  its ``density`` calls in ``calls``;
* the threshold law of the RM28 witness, after Luedtke and van der Laan (2016), with its
  closed forms.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad

from cleverly.estimators import TMLE
from cleverly.interventions import Rule
from cleverly.interventions.base import refuse_regime_densities
from tests.discrete_law_longitudinal import CellMeans
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
    """``d(w) = 1{w <= centre}``, with the centre fixed before the fit: a known rule."""

    centre: float = CENTRE

    def __call__(self, frame: Any) -> np.ndarray:
        return np.where(np.asarray(frame["W"], dtype=float) <= self.centre, 1, 0)


@dataclass(frozen=True)
class SampleThreshold(FixedThreshold):
    """The user lie: the threshold at the sample mean of ``W``, a statistic of the sample."""

    @classmethod
    def of(cls, frame: Any) -> SampleThreshold:
        return cls(float(np.mean(np.asarray(frame["W"], dtype=float))))


def threshold_rule(rule: Any = None, **declaration: Any) -> Rule:
    """A ``Rule`` named ``"thr"``, the fixed threshold unless ``rule`` is given."""
    return Rule(FixedThreshold() if rule is None else rule, "thr", **declaration)


# ------------------------------------------------------------------ the threshold law

GRID_SIZE = 200
NOISE = 0.25
#: The midpoint grid: its sample moments are exact but for ``Var_n(W) = 1/12 - 1/(12 m^2)``.
GRID_W = (np.arange(1, GRID_SIZE + 1) - 0.5) / GRID_SIZE

#: The exact ratio of the reported to the exact standard error, and its pin and bound.  The
#: plan measured 0.7276058 before it chose them (RM28 in docs/roadmap.md).  A curve that
#: carried ``T`` would read 1.  The bound claims an understatement of more than 20 percent
#: and pins no digit.
EXACT_RATIO = 3.0 / np.sqrt(17.0)
RATIO_PIN = 0.7276
UNDERSTATEMENT_BOUND = 0.8
#: The exact value of the regime at the fixed threshold.
PSI = 2.0


def threshold_frame() -> pd.DataFrame:
    """The 800 rows of the point threshold law: four at each grid point."""
    rows = [(w, float(a), 3.0 * w + a + e) for w in GRID_W for a in (0, 1) for e in (-NOISE, NOISE)]
    return pd.DataFrame(rows, columns=["W", "A", "Y"])


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


def sample_ratio(curve: np.ndarray, term: np.ndarray) -> float:
    """``sqrt(E_n[D^2] / E_n[(D + T)^2])``: the reported SE over the exact SE, in sample."""
    return float(np.sqrt(np.mean(curve**2) / np.mean((curve + term) ** 2)))


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
