r"""The exact laws, working model, truths and efficiency bounds of ``hazard-msm``.

**Why a study-local law.**  The study first read the laws of
:mod:`tests.studies.point_survival_common`.  Their hazard is additive on the logit scale, so the
wrong hazard learner of this study (no ``a x W1`` term, no ``W2``) misses only ``W2``, and its
plug-in coefficient of ``a_W1`` sits 0.07 efficiency standard errors from the truth at
n = 2,000 (:data:`SHARED_SURVIVAL` reproduces that law; ``tests/unit/test_hazard_msm_study_design.py``
records the reading).  A necessity or robustness cell on that law could not fail.  The laws here
keep its covariates and, for the survival law, its visits and the hazard's other coefficients.
They change three things: each cause's hazard gains an ``a x W1`` term and a larger ``W2``
coefficient, and both mechanisms are logistic, so that a logistic learner is correct without
being saturated.

**The laws.**  :math:`W_1 \in \{0, 1\}` and :math:`W_2 \in \{0, \dots, 3\}` are independent and
uniform, :math:`P(A = 1 \mid W) = \operatorname{expit}(-1.7 + 1.5 W_1 + 0.95 W_2 - 0.95 W_1 W_2)`,
and a unit entering visit :math:`k \ge 2` drops out with probability
:math:`\operatorname{expit}(-3.5 + 0.4 A + 0.4 W_1 + 0.2 W_2)` before the visit's event.
At each visit cause :math:`j` occurs, among the units at risk with no event of an
earlier cause at the visit, with probability

.. math::

    \rho_j(k \mid a, W) = \operatorname{expit}\bigl(\alpha_j + \delta_j (k - 1) + \beta_j a
        + \gamma_j W_1 + \zeta_j W_2 + \theta_j\, a W_1\bigr),

so the cause-specific hazards are :math:`\lambda_1 = \rho_1` and
:math:`\lambda_2 = \rho_2 (1 - \rho_1)`.  These are the sequential pseudo-hazards a hazard fit
regresses (``survtmle``'s ``hazards_estimate.R``), so a logistic regression on
``(W1, W2, A, A W1)`` at each visit and cause is correctly specified and not saturated
(:class:`InteractionLogistic`).  :data:`SURVIVAL` has five visits and one cause, and
:data:`COMPETING` four visits and two causes.  :data:`OVERLAP` has the hazard of
:data:`SURVIVAL` and a treatment mechanism with limited overlap
(:math:`\operatorname{expit}(-2.85 + 1.5 W_1 + 1.6 W_2 - 1.6 W_1 W_2)`); it carries only the
reported comparison of the hazard and iterated-mean fits.

**The working model.**  A cell is a regimen :math:`r \in \{a_1, a_0\}` (static arms 1 and 0) and
a visit :math:`t = 1, \dots, T`, with uniform weight :math:`h = 1`.  The design is
:math:`\varphi(c, V) = (1, a_r, t, a_r W_1)` with :math:`V = W_1`, under the logit link (the
primary model) or the identity link.  For each cause :math:`j` the coefficient :math:`\beta_j`
solves

.. math::

    E\Bigl[\sum_c \varphi(c, V)\,\{\Psi_{c,j}(W) - m(c, V; \beta)\}\Bigr] = 0,
    \qquad \Psi_{c,j}(W) = F_j(t \mid a_r, W).

**Truths.**  :func:`exact_beta` computes the projection by its own Newton iteration from the
per-stratum cell risks of :meth:`Law.curve`.  Nothing here calls the estimator or
:func:`cleverly.msm.solve_projection`.

**Efficiency bounds.**  The efficient influence function of :math:`\beta_j` is

.. math::

    D^*_\beta = M^{-1} \sum_c \varphi(c, V)\,\bigl[D_c + \psi_c - m(c, V; \beta)\bigr],
    \qquad M = E\Bigl[\sum_c \tfrac{dm}{d\eta}(c, V)\,\varphi\varphi^\top\Bigr],

with :math:`D_c + \psi_c = D^{(1)}_c + \Psi_{c,j}(W)` the cell's own curve without its centring
(Rosenblum and van der Laan 2010, (17)).  Its coefficients :math:`M^{-1}\varphi(c, V)` vary with
:math:`V`, so :func:`coefficient_sd` enumerates the support with stratum-dependent coefficients.
The cell curve is the iterated-mean form :meth:`Law.cell_curve`, a sum over the visits of the
weighted residuals of the law's own sequential regressions.  The estimator uses the hazard form;
the design test checks the two forms equal at every support point.

**The wrong hazard learner.**  :class:`NoW2Logistic` regresses each visit's event on ``(W1, A)``:
no ``W2`` and no ``a x W1``.  Its limit is the misspecified maximum-likelihood fit on the law,
which :func:`wrong_hazard_limit` computes by Newton's method on the law laid out as weighted data,
and then projects.  It keeps ``W1``, so its plug-in coefficient of ``a_W1`` is not zero by
construction.  A learner with no ``W`` would give that coefficient exactly zero under static
regimens and a uniform weight.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.longitudinal import LongitudinalData
from cleverly.msm import MSM, MSMLink

#: The event codes of the competing law, by cause label.
CAUSES = {1: "relapse", 2: "death"}
#: The static regimens, by label, and the treatment value each sets.
REGIMENS: dict[str, int] = {"a1": 1, "a0": 0}
#: The working model's terms, in design-column order.
TERMS = ("(intercept)", "a", "t", "a_W1")
LINKS = ("logit", "identity")
#: The baseline strata and their mass.
STRATA = tuple((w1, w2) for w1 in (0, 1) for w2 in (0, 1, 2, 3))
MASS = 0.125


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class Cause:
    """One cause's sequential pseudo-hazard, on the logit scale.

    Parameters
    ----------
    intercept : float
        :math:`\\alpha_j`, at visit 1.
    trend : float
        :math:`\\delta_j`, per visit.
    arm : float
        :math:`\\beta_j`, the coefficient of ``A``.
    w1 : float
        :math:`\\gamma_j`, the coefficient of ``W1``.
    w2 : float
        :math:`\\zeta_j`, the coefficient of ``W2``.
    interaction : float
        :math:`\\theta_j`, the coefficient of ``A W1``.
    """

    intercept: float
    trend: float
    arm: float
    w1: float
    w2: float
    interaction: float

    def logit(self, node: int, a: Any, w1: Any, w2: Any) -> Any:
        a, w1, w2 = (np.asarray(value, dtype=float) for value in (a, w1, w2))
        return (
            self.intercept
            + self.trend * (node - 1)
            + self.arm * a
            + self.w1 * w1
            + self.w2 * w2
            + self.interaction * a * w1
        )


@dataclass(frozen=True)
class Law:
    """A finite held-treatment survival law with sequential logistic pseudo-hazards.

    Parameters
    ----------
    name : str
        The law's label.
    n_times : int
        The visit count.
    hazards : tuple of Cause
        One pseudo-hazard per cause, in cause order.
    treatment : tuple of float
        :math:`(c_0, c_1, c_2, c_3)` of :math:`P(A = 1 \\mid W)` on ``(1, W1, W2, W1 W2)``.
    dropout_coefficients : tuple of float
        :math:`(d_0, d_1, d_2, d_3)` of the dropout probability on ``(1, A, W1, W2)``.
    logistic : bool
        Whether both mechanisms are linear on the logit scale (the study laws) or on the
        probability scale (the shared law).
    """

    name: str
    n_times: int
    hazards: tuple[Cause, ...]
    treatment: tuple[float, float, float, float] = (0.3, 0.15, 0.075, 0.0)
    dropout_coefficients: tuple[float, float, float, float] = (0.03, 0.02, 0.02, 0.01)
    logistic: bool = False

    @property
    def causes(self) -> tuple[int, ...]:
        """The one-based cause codes."""
        return tuple(range(1, len(self.hazards) + 1))

    def cause_label(self, cause: int) -> str | None:
        """The label a fit reports a cause by, ``None`` on a one-cause law."""
        return CAUSES[cause] if len(self.hazards) == 2 else None

    @property
    def grid(self) -> tuple[float, ...]:
        return tuple(float(k) for k in range(1, self.n_times + 1))

    # -------------------------------------------------------------- factors

    def _link(self, value: Any) -> Any:
        return _expit(value) if self.logistic else value

    def treated(self, w1: Any, w2: Any) -> Any:
        """:math:`P(A = 1 \\mid W)`."""
        c0, c1, c2, c3 = self.treatment
        x1, x2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
        return self._link(c0 + c1 * x1 + c2 * x2 + c3 * x1 * x2)

    def g(self, w1: int, w2: int) -> np.ndarray:
        """``[P(A = 0 | W), P(A = 1 | W)]``."""
        p = float(self.treated(w1, w2))
        return np.array([1.0 - p, p])

    def dropout(self, a: Any, w1: Any, w2: Any) -> Any:
        """The dropout probability at a visit ``k >= 2``, before its event."""
        d0, d1, d2, d3 = self.dropout_coefficients
        x = (np.asarray(value, dtype=float) for value in (a, w1, w2))
        arm, x1, x2 = x
        return self._link(d0 + d1 * arm + d2 * x1 + d3 * x2)

    def retained(self, node: int, a: int, w1: int, w2: int) -> float:
        """The probability of staying in follow-up at ``node``: one at visit 1."""
        return 1.0 if node == 1 else 1.0 - float(self.dropout(a, w1, w2))

    def pseudo(self, node: int, a: Any, w1: Any, w2: Any) -> np.ndarray:
        """``(causes, ...)`` sequential pseudo-hazards :math:`\\rho_j`."""
        return np.array([_expit(cause.logit(node, a, w1, w2)) for cause in self.hazards])

    def hazard(self, node: int, a: int, w1: int, w2: int) -> np.ndarray:
        """The cause-specific hazards :math:`\\lambda_j(k \\mid a, W)`."""
        rho = self.pseudo(node, a, w1, w2)
        out = np.zeros(len(rho))
        remaining = 1.0
        for index, value in enumerate(rho):
            out[index] = float(value) * remaining
            remaining -= out[index]
        return out

    def curve(self, a: int, w1: int, w2: int) -> np.ndarray:
        """``(T, causes)``: :math:`F_j(t \\mid a, W)` at each visit."""
        out = np.zeros((self.n_times, len(self.hazards)))
        survival = 1.0
        running = np.zeros(len(self.hazards))
        for node in range(1, self.n_times + 1):
            lam = self.hazard(node, a, w1, w2)
            running = running + survival * lam
            survival *= 1.0 - lam.sum()
            out[node - 1] = running
        return out

    # -------------------------------------------------------------- sampling

    def draw(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` units as ``W1, W2, A, time, event``.

        An event at visit ``k`` is ``time = k`` with the cause code, a dropout at visit
        ``k >= 2`` is ``time = k - 1`` with ``event = 0``, and a unit followed through the last
        visit is ``time = T`` with ``event = 0``.
        """
        rng = np.random.default_rng(seed)
        w1 = rng.integers(0, 2, size=n).astype(float)
        w2 = rng.integers(0, 4, size=n).astype(float)
        a = (rng.random(n) < self.treated(w1, w2)).astype(float)
        time = np.full(n, float(self.n_times))
        event = np.zeros(n, dtype=int)
        open_ = np.ones(n, dtype=bool)
        for node in range(1, self.n_times + 1):
            if node > 1:
                lost = open_ & (rng.random(n) < self.dropout(a, w1, w2))
                time[lost] = float(node - 1)
                open_ &= ~lost
            for code, value in enumerate(self.pseudo(node, a, w1, w2), start=1):
                fired = open_ & (rng.random(n) < value)
                time[fired] = float(node)
                event[fired] = code
                open_ &= ~fired
        return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "time": time, "event": event})

    def container(self, frame: pd.DataFrame, *, weights: str | None = None) -> LongitudinalData:
        """The fit's container, through the time-to-event converter."""
        return LongitudinalData.from_time_to_event(
            frame,
            time="time",
            event="event",
            treatment="A",
            baseline=["W1", "W2"],
            grid=self.grid,
            causes=CAUSES if len(self.hazards) == 2 else None,
            weights=weights,
        )

    def outcomes(self, a: int, w1: int, w2: int) -> list[tuple[float, float, int]]:
        """Every outcome after ``(W, A)``: ``(probability, time, event)``, as :meth:`draw` codes it."""
        out: list[tuple[float, float, int]] = []
        alive = 1.0
        for node in range(1, self.n_times + 1):
            stay = self.retained(node, a, w1, w2)
            if stay < 1.0:
                out.append((alive * (1.0 - stay), float(node - 1), 0))
            alive *= stay
            lam = self.hazard(node, a, w1, w2)
            for code, value in enumerate(lam, start=1):
                out.append((alive * float(value), float(node), code))
            alive *= 1.0 - float(lam.sum())
        out.append((alive, float(self.n_times), 0))
        return out

    def exact_frame(self) -> pd.DataFrame:
        """The law laid out as weighted time-to-event rows; ``wt`` is each outcome's mass."""
        rows: list[dict[str, float]] = []
        for w1, w2 in STRATA:
            g = self.g(w1, w2)
            for a in (0, 1):
                for mass, time, event in self.outcomes(a, w1, w2):
                    rows.append(
                        {
                            "W1": float(w1),
                            "W2": float(w2),
                            "A": float(a),
                            "time": time,
                            "event": event,
                            "wt": MASS * g[a] * mass,
                        }
                    )
        return pd.DataFrame(rows)

    # -------------------------------------------------------------- the cell curve

    def regression(self, a: int, w1: int, w2: int, horizon: int, cause: int) -> np.ndarray:
        """``Q_k``, ``k = 1..horizon + 1``: P(cause by ``horizon`` | at risk at ``k``, a, W).

        ``Q_{horizon + 1} = 0``.  This is the law's own sequential regression, computed
        backwards from the hazards.
        """
        out = np.zeros(horizon + 2)
        for node in range(horizon, 0, -1):
            lam = self.hazard(node, a, w1, w2)
            out[node] = lam[cause - 1] + (1.0 - lam.sum()) * out[node + 1]
        return out

    def cell_curve(
        self,
        label: str,
        horizon: int,
        cause: int,
        w1: int,
        w2: int,
        a: int,
        time: float,
        event: int,
    ) -> float:
        """The uncentred curve :math:`D^{(1)}_c + \\Psi_c(W)` of one cell at one outcome.

        The iterated-mean form: :math:`\\sum_{k \\le t} \\frac{1\\{A = a_r\\}}{g(a_r \\mid W)
        G_k}\\, R_k\\, (Z_k - Q_k) + Q_1(a_r, W)`, with :math:`Z_k` the cause indicator at an
        event, zero at an event of another cause and :math:`Q_{k + 1}` otherwise.
        """
        arm = REGIMENS[label]
        q_arm = self.regression(arm, w1, w2, horizon, cause)
        total = float(q_arm[1])
        if a != arm:
            return total
        ratio = 1.0 / self.g(w1, w2)[a]
        last = int(time) if event else int(time) + 1
        cumulative = 1.0
        for node in range(1, horizon + 1):
            if node > last:
                break
            cumulative *= self.retained(node, a, w1, w2)
            if not event and node == last:
                break  # dropped out at this visit
            if event and node == last:
                z = 1.0 if event == cause else 0.0
            else:
                z = float(q_arm[node + 1])
            total += ratio / cumulative * (z - float(q_arm[node]))
        return total


def _law(name: str, n_times: int, hazards: Sequence[Cause], **mechanism: Any) -> Law:
    return Law(name, n_times, tuple(hazards), **mechanism)


#: The mechanisms of both gated study laws, logistic so that a logistic learner is correct and
#: not saturated.  :math:`P(A = 1 \mid W)` rises from 0.15 to 0.76 with ``W2`` when ``W1 = 0`` and
#: is 0.45 when ``W1 = 1``, so ``W2`` confounds the arm in one ``W1`` stratum only, and a fit
#: that ignores ``W2`` in both nuisances is biased in the ``a_W1`` coefficient: the both-wrong
#: control has a gap to fail by.  Dropout runs from 0.03 to 0.11.  Overlap is good, so the hazard
#: fit and the iterated-mean fit of the same working model agree: both are efficient, so they
#: differ by :math:`o_P(n^{-1/2})` (plan X70 rev 2.4).
MECHANISM: dict[str, Any] = {
    "treatment": (-1.7, 1.5, 0.95, -0.95),
    "dropout_coefficients": (-3.5, 0.4, 0.4, 0.2),
    "logistic": True,
}
#: The ``survival`` hazard: five visits, one cause, an ``a x W1`` term.
SURVIVAL_HAZARD = Cause(-2.3, 0.08, -0.45, 0.35, 0.5, -1.2)
#: The primary law: five visits, one cause, an ``a x W1`` term.
SURVIVAL = _law("survival", 5, [SURVIVAL_HAZARD], **MECHANISM)
#: The competing law: four visits, two causes, an ``a x W1`` term in each.
COMPETING = _law(
    "competing",
    4,
    [Cause(-2.8, 0.08, -0.2, 0.35, 0.5, -1.2), Cause(-2.6, 0.05, -0.5, 0.2, 0.3, 1.1)],
    **MECHANISM,
)
#: The mechanism of the ``overlap`` law, which carries only the ``iterated_mean_difference``
#: reporting family.  :math:`P(A = 1 \mid W)` is 0.055, 0.223, 0.587 and 0.875 at ``W2 = 0..3``
#: when ``W1 = 0``, and 0.206 at every ``W2`` when ``W1 = 1``.  The smallest product of a
#: treatment probability and the retention through the last visit is 0.046 (treated at
#: ``W1 = 0, W2 = 0``).  Overlap is limited at the ends of the ``W1 = 0`` stratum, which is where
#: two efficient fits separate in finite samples (plan X70 rev 2.4).
OVERLAP_MECHANISM: dict[str, Any] = {**MECHANISM, "treatment": (-2.85, 1.5, 1.6, -1.6)}
#: The ``survival`` hazard under the limited-overlap mechanism.
OVERLAP = _law("overlap", 5, [SURVIVAL_HAZARD], **OVERLAP_MECHANISM)
#: The shared ``survival`` law of :mod:`tests.studies.point_survival_common`: no interaction,
#: and its mechanisms linear on the probability scale.  It is the witness that its hazard is
#: too close to the wrong learner.
SHARED_SURVIVAL = _law("shared_survival", 5, [Cause(-2.3, 0.08, -0.45, 0.35, 0.15, 0.0)])
LAWS = {"survival": SURVIVAL, "competing": COMPETING, "overlap": OVERLAP}


# ------------------------------------------------------------------ the working model


def coefficient_name(term: str, cause: str | None = None) -> str:
    """The name a fit reports one coefficient by."""
    inside = term if cause is None else f"{term}, {cause}"
    return f"msm_regimen[{inside}]"


def risk_name(label: str, horizon: int) -> str:
    """The name a fit with no working model reports one risk by."""
    return f"risk_regimen[{label} @ t={horizon}]"


def cells(law: Law) -> tuple[tuple[str, int], ...]:
    """Every ``(regimen, visit)`` cell of the working model."""
    return tuple((label, horizon) for horizon in range(1, law.n_times + 1) for label in REGIMENS)


def phi(label: str, horizon: int, w1: float) -> np.ndarray:
    """The design row :math:`\\varphi(c, V)` of one cell at ``W1 = w1``."""
    a = float(REGIMENS[label])
    return np.array([1.0, a, float(horizon), a * float(w1)])


def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
    """The working model's design over a fit's baseline frame."""
    w1 = np.asarray(frame["W1"], dtype=float)
    a = float(REGIMENS[str(label)])
    return np.column_stack(
        [np.ones_like(w1), np.full_like(w1, a), np.full_like(w1, float(horizon)), a * w1]
    )


def working_model(link: str = "logit") -> MSM:
    """The declared working model under ``link``, with uniform cell weights."""
    # The longitudinal path calls ``design(label, horizon, frame)``; the field's annotation names
    # the point-treatment ``design(label, frame)``.
    return MSM(
        design=cast(Callable[[Any, Any], Any], design),
        terms=TERMS,
        link=cast(MSMLink, link),
        design_kind="known",
    )


def _mean(eta: Any, link: str) -> Any:
    return _expit(eta) if link == "logit" else np.asarray(eta, dtype=float)


def _slope(mean: Any, link: str) -> Any:
    return mean * (1.0 - mean) if link == "logit" else np.ones_like(mean)


def _rows(law: Law, w1: int) -> np.ndarray:
    return np.array([phi(label, horizon, w1) for label, horizon in cells(law)])


# ------------------------------------------------------------------ the projection


def stratum_risks(law: Law, cause: int) -> dict[tuple[int, int], np.ndarray]:
    """``W -> (Psi_c(W))_c``, the true cell risks of every stratum, in :func:`cells` order."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for w1, w2 in STRATA:
        curves = {a: law.curve(a, w1, w2) for a in (0, 1)}
        out[(w1, w2)] = np.array(
            [curves[REGIMENS[label]][horizon - 1, cause - 1] for label, horizon in cells(law)]
        )
    return out


def project(
    law: Law,
    risks: Mapping[tuple[int, int], np.ndarray],
    link: str,
    *,
    iterations: int = 100,
) -> np.ndarray:
    """The working-model coefficients of cell risks ``W -> (Psi_c(W))_c``, by Newton's method.

    The law of ``W`` is uniform.  The iteration stops when the step is below ``1e-15``.
    """
    beta = np.zeros(len(TERMS))
    step = np.full(len(TERMS), np.inf)
    for _ in range(iterations):
        score = np.zeros(len(TERMS))
        jacobian = np.zeros((len(TERMS), len(TERMS)))
        for stratum in STRATA:
            x = _rows(law, stratum[0])
            mean = _mean(x @ beta, link)
            score += MASS * x.T @ (risks[stratum] - mean)
            jacobian += MASS * (x.T * _slope(mean, link)) @ x
        step = np.linalg.solve(jacobian, score)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-15:
            break
    if np.max(np.abs(step)) > 1e-12:  # pragma: no cover - the projection is well posed
        raise AssertionError(f"the projection did not converge: last step {step}")
    return beta


def exact_beta(law: Law, cause: int = 1, link: str = "logit") -> np.ndarray:
    """The true coefficients of one cause, the projection of the law's own cell risks."""
    return project(law, stratum_risks(law, cause), link)


def information(law: Law, beta: np.ndarray, link: str) -> np.ndarray:
    """:math:`M = E[\\sum_c \\tfrac{dm}{d\\eta}\\,\\varphi\\varphi^\\top]` at ``beta``."""
    out = np.zeros((len(TERMS), len(TERMS)))
    for w1, _ in STRATA:
        x = _rows(law, w1)
        out += MASS * (x.T * _slope(_mean(x @ beta, link), link)) @ x
    return out


# ------------------------------------------------------------------ the efficiency bound


def coefficient_sd(
    law: Law,
    cause: int,
    coefficients: Callable[[tuple[int, int]], np.ndarray],
    offset: Callable[[tuple[int, int]], np.ndarray],
) -> np.ndarray:
    r"""The SD of :math:`\sum_c b_c(W) (D^{(1)}_c + \Psi_c(W)) + o(W)` over the law's support.

    ``coefficients(stratum)`` is a ``(p, cells)`` matrix :math:`b(W)` over :func:`cells` and
    ``offset(stratum)`` a length-``p`` vector :math:`o(W)`.  Constant coefficients and the
    offset :math:`-\sum_c b_c \psi_c` give the efficiency bound of a linear combination of cell
    risks.  The function raises when the enumerated curve does not have mean zero.
    """
    first = np.zeros(np.asarray(offset(STRATA[0])).shape)
    second = np.zeros_like(first)
    for stratum in STRATA:
        w1, w2 = stratum
        g = law.g(w1, w2)
        b = np.asarray(coefficients(stratum), dtype=float)
        o = np.asarray(offset(stratum), dtype=float)
        for a in (0, 1):
            for mass, time, event in law.outcomes(a, w1, w2):
                pieces = np.array(
                    [
                        law.cell_curve(label, horizon, cause, w1, w2, a, time, event)
                        for label, horizon in cells(law)
                    ]
                )
                curve = b @ pieces + o
                probability = MASS * g[a] * mass
                first += probability * curve
                second += probability * curve**2
    if np.max(np.abs(first)) > 1e-9:  # pragma: no cover - guards the enumeration
        raise AssertionError(f"the influence function has mean {first}, not zero")
    return np.sqrt(second)


def efficiency_sd(law: Law, cause: int = 1, link: str = "logit") -> np.ndarray:
    r"""The exact efficiency bound of each coefficient of one cause, as a standard deviation.

    The coefficients of :func:`coefficient_sd` are :math:`M^{-1}\varphi(c, V)` and the offset
    is :math:`-M^{-1}\sum_c \varphi(c, V)\, m(c, V; \beta)`.
    """
    beta = exact_beta(law, cause, link)
    inverse = np.linalg.inv(information(law, beta, link))

    def coefficients(stratum: tuple[int, int]) -> np.ndarray:
        return np.asarray(inverse @ _rows(law, stratum[0]).T)

    def offset(stratum: tuple[int, int]) -> np.ndarray:
        x = _rows(law, stratum[0])
        return np.asarray(-inverse @ x.T @ _mean(x @ beta, link))

    return coefficient_sd(law, cause, coefficients, offset)


# ------------------------------------------------------------------ learners


class InteractionLogistic(BaseEstimator, ClassifierMixin):
    """The correct hazard learner: unpenalised logistic on ``(W1, W2, A, A W1)``.

    A hazard fit hands its learner the design ``[W1, W2, A]`` at each visit and cause, and any
    other width raises.  Each pseudo-hazard of :data:`SURVIVAL` and :data:`COMPETING` is in
    this model, which is not saturated, so its initial score is not zero and targeting moves
    every fit.
    """

    def _features(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != 3:
            raise ValueError(f"a hazard design has 3 columns [W1, W2, A], not {matrix.shape[1]}")
        return np.column_stack([matrix, matrix[:, 0] * matrix[:, 2]])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> InteractionLogistic:
        self.model_ = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
        self.model_.fit(self._features(X), np.asarray(y, dtype=float), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._features(X)))


class NoW2Logistic(InteractionLogistic):
    """The wrong hazard learner: unpenalised logistic on ``(W1, A)``, no ``W2`` and no ``A W1``.

    The law's hazards depend on ``W2`` and on ``A W1``, and ``A`` depends on ``W2``, so the fit
    is misspecified and confounded by ``W2``.
    """

    def _features(self, X: Any) -> np.ndarray:
        return super()._features(X)[:, [0, 2]]


class TreatmentLogistic(BaseEstimator, ClassifierMixin):
    """The correct treatment learner: unpenalised logistic on ``(W1, W2, W1 W2)``.

    A treatment design is ``[W1, W2]``.  The model holds :math:`P(A = 1 \\mid W)` of both
    study laws with four coefficients over eight strata, so it is correct and not saturated.
    """

    def _features(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != 2:
            raise ValueError(f"a treatment design has 2 columns [W1, W2], not {matrix.shape[1]}")
        return np.column_stack([matrix, matrix[:, 0] * matrix[:, 1]])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TreatmentLogistic:
        self.model_ = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
        self.model_.fit(self._features(X), np.asarray(y, dtype=float), sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._features(X)))


class CensoringLogistic(TreatmentLogistic):
    """The correct censoring learner: unpenalised logistic on ``(W1, W2, A)`` at each visit.

    A censoring design is ``[W1, W2, A]``.  The model holds the retention of both study laws
    with four coefficients over sixteen cells, so it is correct and not saturated.
    """

    def _features(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != 3:
            raise ValueError(f"a censoring design has 3 columns [W1, W2, A], not {matrix.shape[1]}")
        return matrix


def _logistic_limit(rows: np.ndarray, events: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """The weighted logistic MLE of ``events / totals`` on ``rows``, by Newton's method."""
    coefficients = np.zeros(rows.shape[1])
    for _ in range(100):
        p = _expit(rows @ coefficients)
        score = rows.T @ (events - totals * p)
        hessian = (rows.T * (totals * p * (1.0 - p))) @ rows
        step = np.linalg.solve(hessian, score)
        coefficients = coefficients + step
        if np.max(np.abs(step)) < 1e-14:
            break
    return coefficients


def wrong_hazard_curves(law: Law) -> dict[tuple[int, int], np.ndarray]:
    """``(w1, a) -> (T, causes)`` incidences at the limit of :class:`NoW2Logistic`.

    At each visit cause 1 regresses its event on the units at risk, and cause 2 on those with
    no cause-1 event at the visit.  The units at risk are those entering event-free and
    retained, so their mass is :math:`P(W) g(a \\mid W) S(k - 1) G_k`.
    """
    keys = [(w1, w2, a) for w1, w2 in STRATA for a in (0, 1)]
    entering = {key: MASS * law.g(key[0], key[1])[key[2]] for key in keys}
    rows = np.array([[1.0, float(w1), float(a)] for w1, _, a in keys])
    design_rows = {(w1, a): np.array([1.0, float(w1), float(a)]) for w1 in (0, 1) for a in (0, 1)}
    hazards = {key: np.zeros((law.n_times, len(law.causes))) for key in design_rows}
    for node in range(1, law.n_times + 1):
        at_risk = np.array(
            [entering[key] * law.retained(node, key[2], key[0], key[1]) for key in keys]
        )
        lam = np.array([law.hazard(node, a, w1, w2) for w1, w2, a in keys])
        remaining = at_risk.copy()
        earlier = dict.fromkeys(design_rows, 0.0)
        for index in range(len(law.causes)):
            events = at_risk * lam[:, index]
            coefficients = _logistic_limit(rows, events, remaining)
            for cell, row in design_rows.items():
                value = float(_expit(row @ coefficients)) * (1.0 - earlier[cell])
                hazards[cell][node - 1, index] = value
                earlier[cell] += value
            remaining = remaining - events
        for key, total in zip(keys, at_risk, strict=True):
            entering[key] = total * (1.0 - float(lam[keys.index(key)].sum()))
    curves: dict[tuple[int, int], np.ndarray] = {}
    for cell, values in hazards.items():
        survival = 1.0
        running = np.zeros(values.shape[1])
        out = np.zeros_like(values)
        for node in range(law.n_times):
            running = running + survival * values[node]
            survival *= 1.0 - float(values[node].sum())
            out[node] = running
        curves[cell] = out
    return curves


def wrong_hazard_limit(law: Law, cause: int = 1, link: str = "logit") -> np.ndarray:
    """The plug-in coefficients of one cause at the limit of :class:`NoW2Logistic`."""
    curves = wrong_hazard_curves(law)
    risks = {
        (w1, w2): np.array(
            [curves[(w1, REGIMENS[label])][horizon - 1, cause - 1] for label, horizon in cells(law)]
        )
        for w1, w2 in STRATA
    }
    return project(law, risks, link)
