r"""Laws, learners, fits and exact-law limits of ``option-markov-order`` (X76 part (f)).

``LongitudinalData.with_markov_order(k)`` declares that every conditional factor depends on the
history only through ``W`` and a window of the last nodes: at node ``t`` the mechanisms read
``L_{t-k}, ..., L_t`` and ``A_{t-k-1}, ..., A_{t-1}``, and the outcome regression reads
``L_{t-k}, ..., L_t`` and the arms of the rule nodes in the window
(``src/cleverly/longitudinal/data.py``).  Under the restriction the windowed nuisances can be
correct, and the estimator is the ordinary LTMLE with its product remainder (Fix-8 plan
section 2.6).  If the restriction fails on one side only, double robustness keeps the fit
consistent; if it fails on both, the fit is inconsistent.

Two laws (Fix-8 plan section 5.7, rev 2.1), with ``W ~ Uniform(0, 1)``, four nodes of binary
``L_t``, ``A_t`` and retention ``C_t``, ``L_0 = A_0 = 0``:

* ``L_t ~ Bernoulli(0.15 + 0.3 W + 0.3 L_{t-1} - 0.1 A_{t-1})``;
* ``C_t ~ Bernoulli(expit(2.8 + 0.3 W - 0.4 L_t))``, retention;
* ``Y ~ Bernoulli(0.15 + 0.15 W + 0.15 L_4 + 0.35 L_3 - 0.05 A_4 - 0.05 A_3)``;
* ``LF1``: ``A_t ~ Bernoulli(expit(-1.0 + 0.4 W - 0.5 L_t + 2.0 A_{t-1} + 1.5 L_{t-1}))``, Markov
  of order 1 on both sides;
* ``LF2``: the same without ``1.5 L_{t-1}``, so the mechanisms read the order-0 window while
  ``Y`` still reads ``L_3``.

Under a static regimen every iterated regression is linear in ``(W, L_{t-1}, L_t)``, so a
``LinearRegression`` on the order-1 window contains it, and the mechanisms are logistic in the
window.  The truths do not depend on the mechanisms, and everything is linear in ``W``, so the
g-formula at ``W = 0.5`` is exact: :func:`truth`.

**Discarded law (rev 2).**  The rev 2 mechanism ``expit(-0.6 + 0.4 W - 0.5 L_t + 0.3 A_{t-1} +
1.5 L_{t-1})`` made the cumulative ratio heavy-tailed on both static paths.  On 2,200 pilot
draws its order-1 ``EY(always)`` cell covered 0.919 (SE ratio 0.94), and the full-history fit,
which declares no order, read the same on the same draws (0.919, SE ratio 0.93); at n = 8,000
both covered 0.953.  The shortfall is a finite-sample property of LTMLE's weights, not of the
window, so the law changed under the rev 2.1 pre-read rule; :data:`DISCARDED_TREATMENT` keeps
it.

**Exact-law limits.**  :func:`quadrature_frame` writes the law as a weighted finite data set:
Gauss-Legendre nodes in ``W`` times every path of ``(L_t, A_t, C_t)`` and ``Y``, each row
weighted by its probability.  A fit with ``weights=`` on it is the estimator applied to the law
itself, so its estimate is the population limit, and its influence curve, divided by the
relative weight, is the limit of the curve: :func:`exact_limit`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression

T = 4
LAWS = ("LF1", "LF2")
#: ``(intercept, W, L_t, A_{t-1}, L_{t-1})`` of each law's treatment logit.
TREATMENT = {"LF1": (-1.0, 0.4, -0.5, 2.0, 1.5), "LF2": (-1.0, 0.4, -0.5, 2.0, 0.0)}
#: The rev 2 ``LF1`` mechanism, discarded before declaration (module docstring).
DISCARDED_TREATMENT = (-0.6, 0.4, -0.5, 0.3, 1.5)
REGIMENS = {"never": 0, "always": 1}
REFERENCE = "never"
EY_ALWAYS = "ey_regimen[always]"
EY_NEVER = "ey_regimen[never]"
ATE = "ate_regimen[always vs never]"
ESTIMANDS = (EY_ALWAYS, EY_NEVER, ATE)

#: Plan 5.7's fit settings.
G_BOUNDS = (1e-8, 1.0)
MAX_ITER = 100
TOL = 1e-10
#: Gauss-Legendre nodes in ``W`` of the quadrature law (the limits agree to 1e-12 at 20 and 40).
QUADRATURE_NODES = 20
#: The least exact-law limit of a control, in SDs (plan 5.0).
CONTROL_FLOOR_SD = 0.45


# ------------------------------------------------------------------------------- the laws
def p_covariate(w: Any, l_prev: Any, a_prev: Any) -> Any:
    """``P(L_t = 1 | W, L_{t-1}, A_{t-1})``."""
    return 0.15 + 0.3 * np.asarray(w) + 0.3 * np.asarray(l_prev) - 0.1 * np.asarray(a_prev)


def p_treatment(
    law: str, w: Any, l_now: Any, a_prev: Any, l_prev: Any, coefficients: Any = None
) -> Any:
    """``P(A_t = 1 | W, L_t, A_{t-1}, L_{t-1})`` of ``law``."""
    c0, cw, cl, ca, clp = TREATMENT[law] if coefficients is None else coefficients
    return expit(
        c0
        + cw * np.asarray(w)
        + cl * np.asarray(l_now)
        + ca * np.asarray(a_prev)
        + clp * np.asarray(l_prev)
    )


def p_retention(w: Any, l_now: Any) -> Any:
    """``P(C_t = 1 | W, L_t)``, the probability of staying in follow-up."""
    return expit(2.8 + 0.3 * np.asarray(w) - 0.4 * np.asarray(l_now))


def p_outcome(w: Any, l4: Any, l3: Any, a4: Any, a3: Any) -> Any:
    """``P(Y = 1 | W, L_4, L_3, A_4, A_3)``."""
    return (
        0.15
        + 0.15 * np.asarray(w)
        + 0.15 * np.asarray(l4)
        + 0.35 * np.asarray(l3)
        - 0.05 * np.asarray(a4)
        - 0.05 * np.asarray(a3)
    )


def columns() -> list[str]:
    """The wide columns, in node order."""
    return ["W", *(f"{v}{t}" for t in range(1, T + 1) for v in ("L", "A", "C")), "Y"]


def draw(law: str, n: int, seed: int, *, treatment: Any = None) -> pd.DataFrame:
    """One sample of ``law`` in wide form; a censored unit's later columns are missing."""
    if law not in LAWS:
        raise KeyError(law)
    rng = np.random.default_rng(seed)
    w = rng.uniform(size=n)
    out: dict[str, np.ndarray] = {"W": w}
    l_prev, a_prev = np.zeros(n), np.zeros(n)
    alive = np.ones(n, dtype=bool)
    ls, arms = [], []
    for t in range(1, T + 1):
        l_now = rng.binomial(1, p_covariate(w, l_prev, a_prev)).astype(float)
        a_now = rng.binomial(1, p_treatment(law, w, l_now, a_prev, l_prev, treatment)).astype(float)
        c_now = rng.binomial(1, p_retention(w, l_now)).astype(float)
        for name, values in (("L", l_now), ("A", a_now), ("C", c_now)):
            column = values.copy()
            column[~alive] = np.nan
            out[f"{name}{t}"] = column
        alive &= c_now == 1.0
        ls.append(l_now)
        arms.append(a_now)
        l_prev, a_prev = l_now, a_now
    y = rng.binomial(1, p_outcome(w, ls[3], ls[2], arms[3], arms[2])).astype(float)
    y[~alive] = np.nan
    out["Y"] = y
    return pd.DataFrame(out)[columns()]


def truth() -> dict[str, float]:
    """The regimen means and the ATE, by the g-formula at ``W = 0.5`` (exact: all linear in W)."""

    def mean(a: float) -> float:
        w = 0.5
        p = 0.15 + 0.3 * w
        path = [p]
        for _ in range(2, T + 1):
            p = 0.15 + 0.3 * w + 0.3 * p - 0.1 * a
            path.append(p)
        return 0.15 + 0.15 * w + 0.15 * path[3] + 0.35 * path[2] - 0.1 * a

    return {EY_ALWAYS: mean(1.0), EY_NEVER: mean(0.0), ATE: mean(1.0) - mean(0.0)}


def quadrature_frame(law: str, nodes: int = QUADRATURE_NODES) -> pd.DataFrame:
    """``law`` as a weighted finite data set: column ``weight`` holds each row's probability."""
    x, weights = np.polynomial.legendre.leggauss(nodes)
    rows: list[dict[str, float]] = []
    missing = {f"{v}{s}": np.nan for s in range(1, T + 1) for v in ("L", "A", "C")}

    def extend(
        w: float, t: int, prob: float, history: dict[str, float], ls: list[float], arms: list[float]
    ) -> None:
        if t > T:
            py = float(p_outcome(w, ls[3], ls[2], arms[3], arms[2]))
            rows.append({**history, "Y": 1.0, "weight": prob * py})
            rows.append({**history, "Y": 0.0, "weight": prob * (1.0 - py)})
            return
        l_prev, a_prev = (ls[-1], arms[-1]) if ls else (0.0, 0.0)
        for l_now in (0.0, 1.0):
            pl = float(p_covariate(w, l_prev, a_prev))
            pl = pl if l_now else 1.0 - pl
            for a_now in (0.0, 1.0):
                pa = float(p_treatment(law, w, l_now, a_prev, l_prev))
                pa = pa if a_now else 1.0 - pa
                pc = float(p_retention(w, l_now))
                node = {**history, f"L{t}": l_now, f"A{t}": a_now}
                censored = {**missing, **node, f"C{t}": 0.0, "Y": np.nan}
                for s in range(1, t):
                    censored[f"C{s}"] = 1.0
                rows.append({**censored, "weight": prob * pl * pa * (1.0 - pc)})
                extend(
                    w,
                    t + 1,
                    prob * pl * pa * pc,
                    {**node, f"C{t}": 1.0},
                    [*ls, l_now],
                    [*arms, a_now],
                )

    for node, weight in zip((x + 1.0) / 2.0, weights / 2.0, strict=True):
        extend(float(node), 1, float(weight), {"W": float(node)}, [], [])
    frame = pd.DataFrame(rows)
    return frame[[*columns(), "weight"]]


# ---------------------------------------------------------------------------- the learners
def newton_logistic(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted logistic maximum likelihood with an intercept, by Newton, ridge ``1e-12``."""
    x = np.column_stack([np.ones(len(target)), design])
    beta = np.zeros(x.shape[1])
    beta[0] = logit(np.clip(np.average(target, weights=weight), 1e-6, 1.0 - 1e-6))
    for _ in range(100):
        p = expit(x @ beta)
        hessian = (x * (weight * p * (1.0 - p))[:, None]).T @ x
        step = np.linalg.solve(hessian + 1e-12 * np.eye(len(beta)), x.T @ (weight * (target - p)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            return beta
    raise RuntimeError("the logistic fit did not converge")


class Logistic(BaseEstimator, ClassifierMixin):
    """The study's mechanism learner: unpenalized logistic maximum likelihood by Newton.

    It solves the score equation to ``1e-12``, so a correct mechanism is fitted exactly on the
    quadrature law, where a quasi-Newton solver stops at its own tolerance.
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> Logistic:
        y = np.asarray(target, dtype=float)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.beta_ = newton_logistic(np.asarray(design, dtype=float), y, weight)
        self.classes_ = np.array([0.0, 1.0])
        self.n_features_in_ = np.asarray(design).shape[1]
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        p = expit(self.beta_[0] + np.asarray(design, dtype=float) @ self.beta_[1:])
        return np.column_stack([1.0 - p, p])


# --------------------------------------------------------------------------- the settings
@dataclass(frozen=True)
class Setting:
    """One fit configuration.

    Parameters
    ----------
    law : str
        ``"LF1"`` or ``"LF2"``.
    order : int or None
        The declared Markov order, or ``None`` for the full history.
    """

    law: str
    order: int | None


SETTINGS: dict[str, Setting] = {
    "lf1_order1": Setting("LF1", 1),
    "lf1_order2": Setting("LF1", 2),
    "lf1_full": Setting("LF1", None),
    "lf1_order0": Setting("LF1", 0),
    "lf2_order0": Setting("LF2", 0),
}


def longitudinal_data(frame: pd.DataFrame, order: int | None, *, weights: str | None = None) -> Any:
    """The study's ``LongitudinalData``, windowed to ``order``."""
    from cleverly.longitudinal import LongitudinalData

    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=[f"A{t}" for t in range(1, T + 1)],
        baseline=["W"],
        time_varying=[[f"L{t}"] for t in range(1, T + 1)],
        censoring=[f"C{t}" for t in range(1, T + 1)],
        weights=weights,
    )
    return data if order is None else data.with_markov_order(order)


def estimator(**overrides: Any) -> Any:
    """The study's ``LTMLE``: linear iterated regressions and Newton logistic mechanisms."""
    from cleverly.longitudinal import LTMLE

    options: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": Logistic(),
        "censoring_learner": Logistic(),
        "n_folds": 1,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": 0,
    }
    options.update(overrides)
    return LTMLE(dict(REGIMENS), reference=REFERENCE, **options)


def fit(frame: pd.DataFrame, name: str, *, weights: str | None = None, **overrides: Any) -> Any:
    """Fit the named setting on one sample (or on a weighted law with ``weights=``)."""
    setting = SETTINGS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(**overrides).fit(longitudinal_data(frame, setting.order, weights=weights))


# ------------------------------------------------------------------------ exact-law limits
@dataclass(frozen=True)
class ExactLimit:
    """The population limit of one estimand of one setting.

    Parameters
    ----------
    estimate : float
        The limit of the estimate.
    curve_sd : float
        ``sqrt(E[D^2])`` of the limit influence curve, the limit of ``sqrt(n)`` times the
        reported SE.
    """

    estimate: float
    curve_sd: float

    def sd(self, n: int) -> float:
        """The reported SE's limit at size ``n``."""
        return self.curve_sd / float(np.sqrt(n))


def exact_limit(
    name: str, nodes: int = QUADRATURE_NODES, *, fitter: Callable[..., Any] | None = None
) -> dict[str, ExactLimit]:
    """Each estimand's limit for a setting, from its weighted fit on the quadrature law.

    The package returns the weighted curve ``(w_i / mean w) D_i``; ``D_i`` is recovered by
    dividing by the relative weight.
    """
    frame = quadrature_frame(SETTINGS[name].law, nodes)
    result = (fitter or fit)(frame, name, weights="weight")
    weight = frame["weight"].to_numpy()
    relative = weight / weight.mean()
    out = {}
    for estimand in ESTIMANDS:
        estimate = result.estimates[estimand]
        curve = np.asarray(estimate.influence_curve, dtype=float) / relative
        out[estimand] = ExactLimit(
            float(estimate.psi), float(np.sqrt(np.sum(weight * curve**2) / weight.sum()))
        )
    return out
