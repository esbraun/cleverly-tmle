r"""The law, the fits and the closed-form constants of ``point-shift-learned-density`` (X121).

The law is :func:`cleverly.datasets.shift_dgp` at its defaults, the law of
:func:`cleverly.datasets.make_shift_dose` and of the interventions page's dose fit:

.. math::

    W \sim N(0, I_3), \quad A \mid W \sim N(2 + 0.7 W_1 - 0.3 W_2, 1), \quad
    Y = 1 + 0.5 A + 0.25 A^2 + W_1 - 0.5 W_2 + 0.2 W_3 + 0.8 \varepsilon.

The parameters are the page's three contrasts against the natural course, ``+0.5`` with a cap
at 5 (a unit the shift would take past 5 keeps its dose), ``+0.5`` uncapped and ``+1.0``
uncapped, and the four policy means. Their truths do not depend on the sample.

Three arms fit every draw, all with the page's degree-2 least-squares outcome model, which
represents :math:`E[Y \mid A, W]` exactly, and 40 core density bins:

``in_sample_learned``
    The page's fit: the regularized booster density, in sample.
``crossfit_learned``
    The same learners cross-fitted on three folds, with the outcome range declared from the law
    (:data:`Q_BOUNDS`).
``in_sample_oracle``
    The law's exact density on the fit's own grid, in sample: the diagnostic arm that isolates
    the learned density's error. It licenses nothing.

Two closed forms are computed here, and ``test_point_shift_learned_density_design`` recomputes
both:

- :func:`outcome_tail_quantiles`. With :math:`s = A + 1` the outcome is
  :math:`Y = 0.75 + s^2 / 4 + L`, where :math:`(s, L)` is bivariate normal with means
  :math:`(3, 0)`, variances :math:`(1.58, 1.93)` and covariance :math:`0.85`. The CDF of
  :math:`Y` is one integral over :math:`s` of a normal CDF, solved for each tail at
  :data:`TAIL_PROBABILITY`.
- :func:`efficiency_bounds`. The efficient influence curve of a contrast is
  :math:`\bar Q(d(A, W), W) - \bar Q(A, W) - \psi + (r_d(A, W) - 1)(Y - \bar Q(A, W))`, with
  :math:`r_d` the ratio of the post-intervention density to :math:`g`. Its variance is
  :math:`\operatorname{Var}(\bar Q(d) - \bar Q) + 0.64 E[(r_d - 1)^2]`. For an uncapped shift
  :math:`r_d = \exp(\delta (a - \mu(w)) - \delta^2 / 2)`; with the cap :math:`u`,
  :math:`r_d = \exp(\delta (a - \mu(w)) - \delta^2 / 2) 1[a \le u] + 1[a > u - \delta]`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.stats import norm, qmc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly.datasets import ShiftDGP, shift_dgp
from cleverly.estimators import TMLE
from cleverly.exceptions import PositivityWarning
from cleverly.interventions import Shift
from cleverly.learners.density import bin_edges
from tests.studies.canonical_shift_policies import OracleShiftDensity

#: The law of ``make_shift_dose``.
DGP: ShiftDGP = shift_dgp()

#: The page's four policies, ``(delta, cap, name)``; the first is the reference.
POLICIES: tuple[tuple[float, float | None, str], ...] = (
    (0.0, None, "current practice"),
    (0.5, 5.0, "+0.5 capped at 5"),
    (0.5, None, "+0.5 uncapped"),
    (1.0, None, "+1.0 uncapped"),
)
REFERENCE = POLICIES[0][2]

#: The short tag of each contrast, and the name a fit reports it under.
CONTRASTS: dict[str, str] = {
    "capped": "ate_policy[+0.5 capped at 5 vs current practice]",
    "uncapped": "ate_policy[+0.5 uncapped vs current practice]",
    "one": "ate_policy[+1.0 uncapped vs current practice]",
}
#: Each policy's ``(delta, cap)``, by the tag its contrast uses; ``current`` is the reference.
SHIFT_OF: dict[str, tuple[float, float | None]] = {
    "current": (0.0, None),
    "capped": (0.5, 5.0),
    "uncapped": (0.5, None),
    "one": (1.0, None),
}
#: Each policy's name, by tag.
POLICY_NAME: dict[str, str] = {
    tag: name for tag, (_, _, name) in zip(SHIFT_OF, POLICIES, strict=True)
}
#: Every primary parameter: the three contrasts and the four policy means.
PARAMETERS: dict[str, str] = {
    **CONTRASTS,
    **{f"ey_{tag}": f"ey_policy[{name}]" for tag, name in POLICY_NAME.items()},
}


def policy_of(parameter: str) -> str:
    """The tag of the policy a parameter reads: a contrast's shifted policy, or a mean's own."""
    return parameter.removeprefix("ey_")


#: The arms, in the order every cell name reads them.
ARMS = ("in_sample_learned", "crossfit_learned", "in_sample_oracle")

#: The page's sample size and core bin count.
N = 3_000
DENSITY_BINS = 40
#: The cross-fitted arm's fold count.
N_FOLDS = 3

#: Each declared tail probability of the outcome range: with ``n`` rows over ``R`` fits of the
#: cross-fitted arm the chance that any observed outcome falls outside :data:`Q_BOUNDS` is at
#: most ``2 n R TAIL_PROBABILITY``, 0.014 at n = 3,000 over the primaries, the pilot and the
#: calibration cap together. Such a fit counts as a failed replication.
TAIL_PROBABILITY = 1e-10
#: The outcome range the cross-fitted arm declares: :func:`outcome_tail_quantiles` at
#: :data:`TAIL_PROBABILITY`, (-7.8878, 35.9063), rounded outward to two decimals.
Q_BOUNDS = (-7.89, 35.91)

#: The regularized booster of the page (``docs/development/example-notebooks.md``, the learner
#: rule below 10,000 rows) without its seed.
BOOSTER = {"max_depth": 2, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0}

#: Sobol points of the closed-form constants, ``2**22``.
CONSTANT_POINTS_LOG2 = 22
#: The scramble seed of those points.
CONSTANT_SEED = 20261126

#: The true value of each contrast, from the law (``ShiftDGP.truth``).
_LAW_TRUTH = DGP.truth(POLICIES)
TRUTH: dict[str, float] = {tag: _LAW_TRUTH[name] for tag, name in PARAMETERS.items()}


def outcome_learner() -> Any:
    """The page's degree-2 least-squares outcome model."""
    return make_pipeline(PolynomialFeatures(2), LinearRegression())


def density_learner(seed: int) -> HistGradientBoostingClassifier:
    """The page's regularized booster, seeded for its early-stopping split."""
    return HistGradientBoostingClassifier(**BOOSTER, random_state=seed)


def draw(n: int, seed: int) -> pd.DataFrame:
    """One sample of the law."""
    frame, _ = DGP.sample(n, policies=POLICIES, seed=seed, backend="pandas")
    return frame


def estimator(arm: str, frame: pd.DataFrame, seed: int, **override: Any) -> TMLE:
    """The unfitted estimator of one arm on one draw, every random state ``seed``.

    ``override`` replaces constructor arguments, for the design test's support-key checks only.
    """
    policies = [Shift(delta, cap=cap, name=name) for delta, cap, name in POLICIES]
    common: dict[str, Any] = {
        "policies": policies,
        "outcome_learner": outcome_learner(),
        "density_bins": DENSITY_BINS,
        "ratio": "density",
        "random_state": seed,
    }
    if arm == "in_sample_learned":
        return TMLE(treatment_learner=density_learner(seed), cross_fit=False, **common)
    if arm == "crossfit_learned":
        arguments = {
            "treatment_learner": density_learner(seed),
            "cross_fit": True,
            "n_folds": N_FOLDS,
            "q_bounds": Q_BOUNDS,
            **common,
            **override,
        }
        return TMLE(**arguments)
    if arm == "in_sample_oracle":
        edges = tuple(float(v) for v in bin_edges(np.asarray(frame["A"], float), DENSITY_BINS))
        return TMLE(treatment_learner=OracleShiftDensity(DGP, edges), cross_fit=False, **common)
    raise KeyError(f"no arm {arm!r}")


def fit(arm: str, frame: pd.DataFrame, seed: int) -> Any:
    """Fit one arm, containing the uncapped shifts' expected support warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        return (
            estimator(arm, frame, seed)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=list(DGP.covariate_names),
                treatment_kind="continuous",
            )
            .single()
        )


def true_ratio(tag: str, dose: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """The law's density ratio ``r_d(a, w)`` of one policy (1 for ``current``)."""
    delta, cap = SHIFT_OF[tag]
    mean = np.asarray(DGP.dose_mean(latent), dtype=float)
    moved = np.exp(delta * (dose - mean) - 0.5 * delta**2)
    if cap is None:
        return moved
    return moved * (dose <= cap) + (dose > cap - delta)


@dataclass(frozen=True)
class Diagnostics:
    """What one fit's density ratio says, beside the law's (reported, no verdict)."""

    ratio_sd: float
    ratio_mean: float
    ratio_rmse: float


def ratio_diagnostics(result: Any, parameter: str, frame: pd.DataFrame) -> Diagnostics:
    """The fitted ratio's SD and mean at the observed doses, and its RMSE against the law's."""
    tag = policy_of(parameter)
    index = list(SHIFT_OF).index(tag)
    fitted = np.asarray(result.nuisance.policies.ratio, dtype=float)[:, index]
    latent = frame.loc[:, list(DGP.covariate_names)].to_numpy(dtype=float)
    exact = true_ratio(tag, frame["A"].to_numpy(dtype=float), latent)
    return Diagnostics(
        ratio_sd=float(np.std(fitted)),
        ratio_mean=float(np.mean(fitted)),
        ratio_rmse=float(np.sqrt(np.mean((fitted - exact) ** 2))),
    )


def _joint() -> tuple[float, float, float, float]:
    """``(mean of s, var of s, var of L, cov(s, L))`` for ``Y = 0.75 + s^2 / 4 + L``."""
    return 3.0, 0.7**2 + 0.3**2 + 1.0, 1.0 + 0.25 + 0.04 + 0.64, 0.7 + 0.5 * 0.3


def outcome_cdf(y: float, *, upper: bool = False) -> float:
    """``P(Y <= y)``, or ``P(Y > y)`` with ``upper``, by one integral over ``s``."""
    mean, var_s, var_l, cov = _joint()
    slope = cov / var_s
    spread = np.sqrt(var_l - cov**2 / var_s)

    def integrand(s: float) -> float:
        z = (y - 0.75 - 0.25 * s * s - slope * (s - mean)) / spread
        tail = norm.sf(z) if upper else norm.cdf(z)
        return float(norm.pdf(s, mean, np.sqrt(var_s)) * tail)

    value, _ = integrate.quad(integrand, -40.0, 50.0, limit=500, epsabs=1e-16, epsrel=1e-10)
    return float(value)


def outcome_tail_quantiles(probability: float = TAIL_PROBABILITY) -> tuple[float, float]:
    """The outcome's ``probability`` and ``1 - probability`` quantiles."""
    target = np.log(probability)
    low = optimize.brentq(lambda y: np.log(outcome_cdf(y)) - target, -30.0, 0.0, xtol=1e-10)
    high = optimize.brentq(
        lambda y: np.log(outcome_cdf(y, upper=True)) - target, 5.0, 300.0, xtol=1e-10
    )
    return float(low), float(high)


def _constant_points() -> tuple[np.ndarray, np.ndarray]:
    sobol = qmc.Sobol(d=DGP.n_latent + 1, scramble=True, seed=CONSTANT_SEED)
    normal = norm.ppf(sobol.random_base2(CONSTANT_POINTS_LOG2))
    latent = normal[:, : DGP.n_latent]
    dose = np.asarray(DGP.dose_mean(latent), dtype=float) + DGP.dose_scale * normal[:, -1]
    return latent, dose


def efficiency_bounds() -> dict[str, float]:
    """The SD of each contrast's efficient influence curve, by quasi-Monte Carlo."""
    latent, dose = _constant_points()
    natural = np.asarray(DGP.outcome_mean(latent, dose), dtype=float)
    out = {}
    for tag in CONTRASTS:
        delta, cap = SHIFT_OF[tag]
        shifted = np.asarray(DGP.outcome_mean(latent, DGP.shifted(dose, delta, cap)), dtype=float)
        ratio = true_ratio(tag, dose, latent)
        variance = np.var(shifted - natural) + DGP.noise_scale**2 * np.mean((ratio - 1.0) ** 2)
        out[tag] = float(np.sqrt(variance))
    return out


def true_ratio_sd() -> dict[str, float]:
    """The SD of each contrast's true density ratio at the observed dose."""
    latent, dose = _constant_points()
    return {tag: float(np.std(true_ratio(tag, dose, latent))) for tag in CONTRASTS}
