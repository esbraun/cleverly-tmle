r"""Finite-covariate MAR laws for the arm-indexed stacked CV-TMLE study.

``docs/roadmap.md`` RM9 ("Registered study") names four laws.  Each draws a baseline
covariate ``W`` with three levels, a treatment ``A``, a response indicator ``Delta`` and,
when ``Delta = 1``, an outcome ``Y``.  Each nuisance is a table indexed ``[w, a]``, so every
truth, efficient influence curve, and misspecified-nuisance limit is a finite sum.

=====  ====  ========================  ======================================================
law    arms  outcome                   tables
=====  ====  ========================  ======================================================
L1     2     binary                    ``P_W``, ``G``, ``PI`` and ``Q`` of
                                       :mod:`tests.discrete_law_mar`
L2     2     bounded continuous        L1's tables; ``Y = a + (b - a) B`` with
                                       ``B ~ Beta(phi mu, phi (1 - mu))`` and ``mu = Q``
L3     3     binary                    ``P_W``, ``G`` and ``Q`` of :mod:`tests.discrete_law_multi`
                                       and :data:`PI_THREE`
L4     3     bounded continuous        L3's tables with L2's scaled Beta outcome
=====  ====  ========================  ======================================================

The known support of a continuous outcome is ``(a, b) = (-2, 8)``, and the fit declares
``q_bounds`` equal to it.  On the ``[0, 1]`` scale the fluctuation uses, the conditional mean
of a continuous outcome is ``mu`` itself, so every scaled quantity below is shared by the
binary and continuous laws.

The three-arm laws keep :mod:`tests.discrete_law_multi`'s labels.  ``CausalData`` sorts them,
so arm code ``0`` is ``"high"``, and ``"high"`` is the reference arm of every contrast.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator

from cleverly._typing import EstimandName
from tests import discrete_law_mar as mar
from tests import discrete_law_multi as multi

#: The known support of a continuous outcome, which the fit declares as ``q_bounds``.
LOWER, UPPER = -2.0, 8.0
#: The Beta precision of a continuous outcome.
PHI = 4.0

#: The three-arm labels in each table's column order, and the order ``CausalData`` codes them.
THREE_ARM_LABELS: tuple[str, ...] = multi.LABELS
THREE_ARM_CODES: tuple[str, ...] = tuple(sorted(THREE_ARM_LABELS))
REFERENCE = "high"

#: ``P(Delta = 1 | A = a, W = w)`` for the three-arm laws, indexed ``[w, a]`` in
#: :data:`THREE_ARM_LABELS` order.  It depends on both arguments, and its smallest product
#: with ``multi.G`` is 0.1125.
PI_THREE = np.array([[0.30, 0.50, 0.70], [0.55, 0.65, 0.45], [0.85, 0.75, 0.60]])

#: The misspecified mechanisms of the robustness cells, indexed ``[w, a]``.  A misspecified
#: outcome regression is ``1 - mu`` on the ``[0, 1]`` scale.
WRONG_G_TWO = np.array([0.65, 0.35, 0.55])
WRONG_PI_TWO = np.array([[0.80, 0.75], [0.55, 0.45], [0.30, 0.25]])
WRONG_G_THREE = np.array([[0.25, 0.35, 0.40], [0.45, 0.20, 0.35], [0.40, 0.40, 0.20]])
WRONG_PI_THREE = np.array([[0.80, 0.45, 0.35], [0.30, 0.40, 0.85], [0.35, 0.60, 0.90]])


@dataclass(frozen=True)
class Law:
    """One finite-covariate MAR law, with every table indexed ``[w, a]``.

    Parameters
    ----------
    key : str
        The short law name, ``"l1"`` to ``"l4"``.
    scenario : str
        The scenario name the primary study publishes the law under.
    p_w : ndarray
        ``P(W = w)``.
    g : ndarray
        ``P(A = a | W = w)``; each row sums to one.
    pi : ndarray
        ``P(Delta = 1 | A = a, W = w)``.
    mu : ndarray
        ``E(Y | A = a, Delta = 1, W = w)`` on the ``[0, 1]`` scale.
    labels : tuple
        The treatment value of each table column.
    continuous : bool
        Whether ``Y`` is the scaled Beta outcome rather than a binary one.
    """

    key: str
    scenario: str
    p_w: np.ndarray
    g: np.ndarray
    pi: np.ndarray
    mu: np.ndarray
    labels: tuple[Any, ...]
    continuous: bool

    @property
    def arms(self) -> int:
        return len(self.labels)

    @property
    def codes(self) -> tuple[Any, ...]:
        """The treatment values in the order ``CausalData`` codes them."""
        return tuple(sorted(self.labels))

    @property
    def reference(self) -> Any:
        return self.codes[0]

    def column(self, code: int) -> int:
        """The table column of the arm ``CausalData`` codes as ``code``."""
        return self.labels.index(self.codes[code])

    @property
    def span(self) -> float:
        return UPPER - LOWER if self.continuous else 1.0

    @property
    def offset(self) -> float:
        return LOWER if self.continuous else 0.0

    @property
    def q_bounds(self) -> tuple[float, float] | None:
        return (LOWER, UPPER) if self.continuous else None

    def mean(self, mu: np.ndarray | None = None) -> np.ndarray:
        """``E(Y | A = a, W = w)`` on the outcome's own scale."""
        return self.offset + self.span * (self.mu if mu is None else mu)

    def variance(self) -> np.ndarray:
        """``Var(Y | A = a, W = w)`` on the outcome's own scale."""
        spread = self.mu * (1.0 - self.mu)
        if self.continuous:
            return self.span**2 * spread / (PHI + 1.0)
        return spread


def _two_arm(g: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 - g, g])


LAWS: dict[str, Law] = {
    "l1": Law(
        "l1",
        "binary_mar_two_arm_stacked",
        mar.P_W,
        _two_arm(mar.G),
        mar.PI,
        mar.Q,
        (0.0, 1.0),
        False,
    ),
    "l2": Law(
        "l2",
        "continuous_mar_two_arm_stacked",
        mar.P_W,
        _two_arm(mar.G),
        mar.PI,
        mar.Q,
        (0.0, 1.0),
        True,
    ),
    "l3": Law(
        "l3",
        "binary_mar_three_arm_stacked",
        multi.P_W,
        multi.G,
        PI_THREE,
        multi.Q,
        THREE_ARM_LABELS,
        False,
    ),
    "l4": Law(
        "l4",
        "continuous_mar_three_arm_stacked",
        multi.P_W,
        multi.G,
        PI_THREE,
        multi.Q,
        THREE_ARM_LABELS,
        True,
    ),
}
BY_SCENARIO: dict[str, Law] = {law.scenario: law for law in LAWS.values()}

#: What each law's fit requests, in the engine's estimand vocabulary.
REQUESTS: dict[str, tuple[EstimandName, ...]] = {
    "l1": ("ey0", "ey1", "ate", "rr", "or"),
    "l2": ("ey0", "ey1", "ate"),
    "l3": ("ey", "ate", "rr", "or"),
    "l4": ("ey", "ate"),
}


def _three_arm_names(*, ratios: bool) -> tuple[str, ...]:
    others = [label for label in THREE_ARM_CODES if label != REFERENCE]
    names = [f"ey[{label}]" for label in THREE_ARM_CODES]
    names += [f"ate[{label} vs {REFERENCE}]" for label in others]
    if ratios:
        names += [f"rr[{label} vs {REFERENCE}]" for label in others]
        names += [f"or[{label} vs {REFERENCE}]" for label in others]
    return tuple(names)


#: The reported estimand names of each law, in publication order.
ESTIMANDS: dict[str, tuple[str, ...]] = {
    "l1": ("ey0", "ey1", "ate", "rr", "or"),
    "l2": ("ey0", "ey1", "ate"),
    "l3": _three_arm_names(ratios=True),
    "l4": _three_arm_names(ratios=False),
}

RATIO_STEMS = frozenset({"rr", "or"})


def stem(name: str) -> str:
    """``"rr"`` for ``"rr"`` and for ``"rr[low vs high]"``."""
    return name.partition("[")[0]


def is_ratio(name: str) -> bool:
    return stem(name) in RATIO_STEMS


def label(law: Law, name: str) -> str:
    """A regex-safe property-cell prefix for one estimand, such as ``l3_ate_low``.

    ``tests/studies/evidence/descriptions.py`` joins the prefixes into one regular
    expression without escaping them, so a prefix carries no bracket or space.
    """
    head, _, rest = name.partition("[")
    if not rest:
        return f"{law.key}_{head}"
    return f"{law.key}_{head}_{rest.split(' vs ')[0].rstrip(']')}"


def _arm_of(law: Law, name: str) -> tuple[int, int | None]:
    """The table columns an estimand reads: its arm, and the reference arm for a contrast."""
    head, _, rest = name.partition("[")
    if law.arms == 2:
        if head in {"ey0", "ey1"}:
            return int(head[-1]), None
        return 1, 0
    if head == "ey":
        return law.labels.index(rest.rstrip("]")), None
    arm = rest.split(" vs ")[0]
    return law.labels.index(arm), law.labels.index(REFERENCE)


def arm_means(law: Law, mu: np.ndarray | None = None) -> np.ndarray:
    """``E(Y(a))`` for each table column, on the outcome's own scale."""
    return np.asarray(law.p_w @ law.mean(mu), dtype=float)


def _contrast(name: str, value: float, reference: float | None) -> float:
    head = stem(name)
    if reference is None:
        return value
    if head == "ate":
        return value - reference
    if head == "rr":
        return value / reference
    if head == "or":
        return (value / (1.0 - value)) / (reference / (1.0 - reference))
    raise ValueError(f"unknown estimand {name!r}")


def values(law: Law, psi: np.ndarray) -> dict[str, float]:
    """Each reported estimand from a vector of arm means, on its natural scale."""
    out: dict[str, float] = {}
    for name in ESTIMANDS[law.key]:
        arm, reference = _arm_of(law, name)
        out[name] = float(
            _contrast(name, float(psi[arm]), None if reference is None else float(psi[reference]))
        )
    return out


def truths(law: Law) -> dict[str, float]:
    """The exact value of each reported estimand, on its natural scale."""
    return values(law, arm_means(law))


def gradient(law: Law, name: str, psi: np.ndarray | None = None) -> np.ndarray:
    """The derivative of one estimand's inference-scale value in the arm means."""
    psi = arm_means(law) if psi is None else psi
    arm, reference = _arm_of(law, name)
    out = np.zeros(law.arms)
    head = stem(name)
    scale = {
        "rr": lambda value: 1.0 / value,
        "or": lambda value: 1.0 / (value * (1.0 - value)),
    }.get(head, lambda value: 1.0)
    out[arm] = scale(psi[arm])
    if reference is not None:
        out[reference] = -scale(psi[reference])
    return out


def arm_covariance(law: Law) -> np.ndarray:
    r"""The covariance of the arm means' efficient influence curves.

    Each arm's curve is
    :math:`\mathbb 1\{A = a\}\Delta (Y - m_a(W)) / (g_a \pi_a) + m_a(W) - \psi_a`.  The
    residual terms of two arms never share a row, so

    .. math::

        \Sigma_{ab} = \mathbb 1\{a = b\}\, E\bigl[\sigma_a^2(W) / (g_a \pi_a)(W)\bigr]
            + \operatorname{Cov}\bigl(m_a(W), m_b(W)\bigr).
    """
    means = law.mean()
    centred = means - arm_means(law)
    between = (centred * law.p_w[:, None]).T @ centred
    within = np.diag(law.p_w @ (law.variance() / (law.g * law.pi)))
    return np.asarray(between + within, dtype=float)


def influence_covariance(law: Law) -> np.ndarray:
    """The inference-scale covariance of every reported estimand's efficient curve."""
    jacobian = np.vstack([gradient(law, name) for name in ESTIMANDS[law.key]])
    return np.asarray(jacobian @ arm_covariance(law) @ jacobian.T, dtype=float)


def efficiency_sd(law: Law, name: str) -> float:
    """The standard deviation of one estimand's efficient influence curve."""
    return float(np.sqrt(gradient(law, name) @ arm_covariance(law) @ gradient(law, name)))


def pointwise_joint_coverage(law: Law, *, draws: int = 1_000_000, seed: int = 0) -> float:
    """The limiting share of samples whose pointwise 95% intervals all cover at once.

    Read from the Gaussian limit of the standardized estimates.  A study that claims the
    simultaneous band does work beside the pointwise intervals needs this below the nominal
    rate, which is what the band's pointwise control asserts.
    """
    covariance = influence_covariance(law)
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    values_, vectors = np.linalg.eigh(correlation)
    factor = vectors * np.sqrt(np.clip(values_, 0.0, None))
    normal = np.random.default_rng(seed).standard_normal((draws, len(scale))) @ factor.T
    return float(np.mean(np.all(np.abs(normal) <= 1.959963984540054, axis=1)))


def sample(law: Law, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` observed-data rows: ``W``, ``A``, ``Y`` and ``Delta``.

    ``Y`` is ``NaN`` wherever ``Delta = 0``.  A three-arm law carries its labels in ``A``.
    """
    rng = np.random.default_rng(seed)
    w = rng.choice(len(law.p_w), size=n, p=law.p_w)
    u = rng.random(n)
    column = (u[:, None] > np.cumsum(law.g[w], axis=1)).sum(axis=1)
    column = np.minimum(column, law.arms - 1)
    observed = rng.random(n) < law.pi[w, column]
    mu = law.mu[w, column]
    if law.continuous:
        scaled = rng.beta(PHI * mu, PHI * (1.0 - mu))
        outcome = LOWER + (UPPER - LOWER) * scaled
    else:
        outcome = (rng.random(n) < mu).astype(float)
    treatment: Any = np.asarray(law.labels, dtype=object)[column]
    if law.arms == 2:
        treatment = treatment.astype(float)
    return pd.DataFrame(
        {
            "W": w.astype(float),
            "A": treatment,
            "Y": np.where(observed, outcome, np.nan),
            "Delta": observed.astype(float),
        }
    )


# ------------------------------------------------------------------------ oracle learners


def _levels(design: np.ndarray, column: int) -> np.ndarray:
    return np.rint(design[:, column]).astype(int)


def _codes(law: Law, design: np.ndarray) -> np.ndarray:
    """The arm code of each row of an ``[A-block, W, ...]`` design."""
    if law.arms == 2:
        return np.rint(design[:, 0]).astype(int)
    block = design[:, : law.arms - 1]
    return np.where(block.any(axis=1), block.argmax(axis=1) + 1, 0)


def _columns(law: Law, codes: np.ndarray) -> np.ndarray:
    return np.array([law.column(int(code)) for code in range(law.arms)])[codes]


class LawOutcome(BaseEstimator):
    """The outcome regression of a law on the ``[0, 1]`` scale, read from ``[A-block, W]``.

    ``mu`` replaces the law's own table, which is how a robustness cell declares a
    misspecified regression.  Any columns after ``W`` are ignored.
    """

    def __init__(self, law: Law, mu: np.ndarray | None = None) -> None:
        self.law = law
        self.mu = mu

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> LawOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def _mean(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        table = self.law.mu if self.mu is None else self.mu
        w = _levels(design, self.law.arms - 1)
        return np.asarray(table[w, _columns(self.law, _codes(self.law, design))], dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return self._mean(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self._mean(X)
        return np.column_stack([1.0 - p, p])


class LawTreatment(BaseEstimator):
    """``g(a | W)`` of a law, read from a design whose first column is ``W``.

    The probability columns follow the arm codes, not the table columns.
    """

    def __init__(self, law: Law, g: np.ndarray | None = None) -> None:
        self.law = law
        self.g = g

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> LawTreatment:
        self.classes_ = np.arange(float(self.law.arms))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        table = self.law.g if self.g is None else self.g
        w = _levels(design, 0)
        return np.column_stack([table[w, self.law.column(code)] for code in range(self.law.arms)])


class LawResponse(BaseEstimator):
    """``P(Delta = 1 | A, W)`` of a law, read from ``[A-block, W]``."""

    def __init__(self, law: Law, pi: np.ndarray | None = None) -> None:
        self.law = law
        self.pi = pi

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> LawResponse:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        table = self.law.pi if self.pi is None else self.pi
        w = _levels(design, self.law.arms - 1)
        p = np.asarray(table[w, _columns(self.law, _codes(self.law, design))], dtype=float)
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------------ misspecified limits


def wrong_tables(law: Law) -> Mapping[str, np.ndarray]:
    """The misspecified outcome, treatment, and response tables of a robustness cell."""
    if law.arms == 2:
        return {"mu": 1.0 - law.mu, "g": _two_arm(WRONG_G_TWO), "pi": WRONG_PI_TWO}
    return {"mu": 1.0 - law.mu, "g": WRONG_G_THREE, "pi": WRONG_PI_THREE}


def targeted_limit(
    law: Law,
    *,
    mu: np.ndarray | None = None,
    g: np.ndarray | None = None,
    pi: np.ndarray | None = None,
) -> dict[str, float]:
    r"""The large-sample limit of the stacked fit under fixed working nuisances.

    Each arm's logistic fluctuation uses the respondents with ``A = a`` and the clever
    covariate :math:`1 / (\tilde g_a \tilde\pi_a)`.  Its limit solves

    .. math::

        \sum_w P(W = w)\, g_a(w) \pi_a(w)\, \tilde H_a(w)
            \bigl\{\mu_a(w) - \operatorname{expit}(\operatorname{logit} \tilde\mu_a(w)
            + \epsilon_a \tilde H_a(w))\bigr\} = 0,

    and the arm mean is :math:`\sum_w P(W = w) \operatorname{expit}(\ldots)` on the
    ``[0, 1]`` scale.  With a correct outcome regression, or with correct treatment and
    response mechanisms, the limit is the truth.
    """
    working_mu = law.mu if mu is None else mu
    working_g = law.g if g is None else g
    working_pi = law.pi if pi is None else pi
    clever = 1.0 / (working_g * working_pi)
    mass = law.p_w[:, None] * law.g * law.pi
    offset = logit(working_mu)
    psi = np.empty(law.arms)
    for arm in range(law.arms):

        def score(epsilon: float, arm: int = arm) -> float:
            fitted = expit(offset[:, arm] + epsilon * clever[:, arm])
            return float(np.sum(mass[:, arm] * clever[:, arm] * (law.mu[:, arm] - fitted)))

        epsilon = brentq(score, -50.0, 50.0, xtol=1e-15, rtol=1e-15)
        scaled = law.p_w @ expit(offset[:, arm] + epsilon * clever[:, arm])
        psi[arm] = law.offset + law.span * scaled
    return values(law, psi)
