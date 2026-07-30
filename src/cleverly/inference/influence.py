r"""Estimands and their efficient influence curves.

Every estimate the library reports is built the same way: a point estimate as a
weighted mean of targeted predictions, plus an influence curve whose sample
variance divided by :math:`n` gives the variance of the estimate.  Writing the
influence curve out explicitly (rather than only its variance) is deliberate -- it
is what makes cluster-robust variance, simultaneous confidence bands, the delta
method and the score diagnostic all fall out of the same object.

The influence curves, on the ``[0, 1]`` outcome scale, with
:math:`r_i = \Delta_i (Y_i - \bar Q^*(A_i, W_i))`:

.. math::

    \mathrm{IC}^{EY_1}_i &= h_1(A_i, W_i)\, r_i + \bar Q^*(1, W_i) - \psi_1 \\
    \mathrm{IC}^{EY_0}_i &= h_0(A_i, W_i)\, r_i + \bar Q^*(0, W_i) - \psi_0 \\
    \mathrm{IC}^{ATE}_i  &= \mathrm{IC}^{EY_1}_i - \mathrm{IC}^{EY_0}_i \\
    \mathrm{IC}^{ATT}_i  &= h_{\mathrm{att}}(A_i, W_i)\, r_i
        + \frac{A_i}{P(A=1)}\bigl(\bar Q^*(1, W_i) - \bar Q^*(0, W_i) - \psi_{\mathrm{att}}\bigr)

and the ATC mirrors the ATT.  Note that the ATT and ATC influence curves carry an
extra term beyond "clever covariate times residual": the estimand conditions on a
*random* event (``A = 1``), so the uncertainty in ``P(A = 1)`` contributes.  Omitting
it -- a common bug -- understates the standard error.

Observation weights enter as :math:`\mathrm{IC}_i \mapsto \tilde w_i \mathrm{IC}_i` with
:math:`\tilde w` normalised to mean one, so weighted and unweighted variances are
directly comparable.  That row-wise multiplication is not a convenience: with every
nuisance fitted and every average taken under the weighted empirical measure, the
estimand is the same functional evaluated at the tilted law
:math:`dP_w = w\,dP / E[w]`, and its efficient influence function is exactly
:math:`(w / E[w])\, D^*_{P_w}`.  :mod:`cleverly.data.weighting` derives that, states
which weighting problems it does and does not cover, and
``tests/unit/test_weighted_estimand.py`` verifies it numerically against a longhand
statement of :math:`\Psi(P_w)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..fluctuation.iterative import InitialFit
from ..fluctuation.submodel import Submodel
from ..utils.bounds import OutcomeScaler
from .cluster import influence_variance
from .delta import log_odds_ratio_influence, log_ratio_influence, normal_ci, two_sided_pvalue

__all__ = [
    "ICParts",
    "ParameterEstimate",
    "Scale",
    "atc_estimate",
    "att_estimate",
    "counterfactual_mean_parts",
    "counterfactual_means",
    "make_estimate",
    "ratio_estimates",
]

Scale = Literal["level", "difference", "ratio"]


@dataclass(frozen=True)
class ParameterEstimate:
    """A point estimate with everything needed to do inference on it.

    Attributes
    ----------
    psi:
        The estimate, on the outcome's original scale.
    influence_curve:
        Per-observation influence curve on the *inference* scale -- the original
        outcome scale for levels and differences, the log scale for ratios.
    variance, std_error:
        Variance and standard error on the inference scale.
    ci, pvalue:
        Wald interval and two-sided p-value.  For a ratio the interval is built on
        the log scale and exponentiated, so it cannot include a negative value and
        has far better small-sample coverage than a symmetric interval would.
    log_psi:
        Present for ratios only: the estimate on the log scale.
    bootstrap:
        Bootstrap summary, when ``n_bootstrap > 0`` was requested.
    """

    name: str
    psi: float
    influence_curve: FloatArray
    variance: float
    n: int
    n_clusters: int
    scale: Scale = "difference"
    alpha: float = 0.05
    log_psi: float | None = None
    bootstrap: BootstrapSummary | None = None

    @property
    def std_error(self) -> float:
        """Standard error on the inference scale."""
        if not np.isfinite(self.variance) or self.variance < 0:
            return float("nan")
        return float(np.sqrt(self.variance))

    @property
    def ci(self) -> tuple[float, float]:
        """Confidence interval at level ``1 - alpha``."""
        if self.scale == "ratio":
            assert self.log_psi is not None
            low, high = normal_ci(self.log_psi, self.std_error, self.alpha)
            return (float(np.exp(low)), float(np.exp(high)))
        return normal_ci(self.psi, self.std_error, self.alpha)

    @property
    def pvalue(self) -> float:
        """Two-sided p-value against the null of no effect.

        The null is zero for a level or difference and one for a ratio (i.e. zero
        on the log scale).
        """
        if self.scale == "ratio":
            assert self.log_psi is not None
            return two_sided_pvalue(self.log_psi, self.std_error)
        return two_sided_pvalue(self.psi, self.std_error)

    @property
    def score(self) -> float:
        """Mean of the influence curve.

        Targeting is supposed to drive this to zero; :mod:`cleverly.validation.score`
        compares it against the standard error to decide whether it did.
        """
        return float(np.mean(self.influence_curve))

    def with_alpha(self, alpha: float) -> ParameterEstimate:
        return replace(self, alpha=alpha)

    def with_bootstrap(self, summary: BootstrapSummary) -> ParameterEstimate:
        return replace(self, bootstrap=summary)

    def to_dict(self) -> dict[str, Any]:
        low, high = self.ci
        row: dict[str, Any] = {
            "estimand": self.name,
            "psi": self.psi,
            "std_err": self.std_error,
            "ci_lower": low,
            "ci_upper": high,
            "p_value": self.pvalue,
            "scale": self.scale,
        }
        if self.log_psi is not None:
            row["log_psi"] = self.log_psi
        if self.bootstrap is not None:
            row["bootstrap_std_err"] = self.bootstrap.std_error
            row["bootstrap_ci_lower"] = self.bootstrap.ci[0]
            row["bootstrap_ci_upper"] = self.bootstrap.ci[1]
        return row

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        low, high = self.ci
        return (
            f"{self.name}: {self.psi:.5g} (se {self.std_error:.4g}, "
            f"{(1 - self.alpha) * 100:g}% CI [{low:.5g}, {high:.5g}], p={self.pvalue:.3g})"
        )


@dataclass(frozen=True)
class BootstrapSummary:
    """Bootstrap inference for a single estimand."""

    std_error: float
    ci: tuple[float, float]
    ci_one_sided_lower: float
    ci_one_sided_upper: float
    n_replicates: int
    n_failed: int
    draws: FloatArray

    @property
    def variance(self) -> float:
        return float(self.std_error**2)


def make_estimate(
    name: str,
    psi: float,
    influence_curve: FloatArray,
    *,
    n: int,
    cluster: IntArray | None = None,
    scale: Scale = "difference",
    alpha: float = 0.05,
    log_psi: float | None = None,
) -> ParameterEstimate:
    """Assemble a :class:`ParameterEstimate`, computing its variance."""
    ic = np.asarray(influence_curve, dtype=float).reshape(-1)
    variance = influence_variance(ic, cluster)
    n_clusters = n if cluster is None else int(np.unique(cluster).size)
    return ParameterEstimate(
        name=name,
        psi=float(psi),
        influence_curve=ic,
        variance=variance,
        n=n,
        n_clusters=n_clusters,
        scale=scale,
        alpha=alpha,
        log_psi=log_psi,
    )


def _residual(outcome: FloatArray, targeted: InitialFit, observed: BoolArray | None) -> FloatArray:
    r"""``Delta * (Y - Q*(A, W))`` on the scaled outcome scale."""
    y = np.asarray(outcome, dtype=float).reshape(-1)
    residual = y - targeted.observed
    if observed is None:
        return residual
    return np.where(np.asarray(observed, dtype=bool), residual, 0.0)


class ArmMean(NamedTuple):
    r"""One counterfactual mean :math:`E[Y(a)]` and its influence curve.

    Both on the *scaled* outcome scale; :meth:`~cleverly.targets.TargetContext.finish`
    maps back.
    """

    psi: float
    influence_curve: FloatArray


def counterfactual_means(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
) -> dict[float, ArmMean]:
    r"""Every counterfactual mean and its influence curve, keyed by arm.

    .. math::

        \hat\Psi_a = \frac1n \sum_i \bar Q^*(a, W_i),
        \qquad
        D_a^*(O) = h_a(A, W)\,\{Y - \bar Q^*(A, W)\} + \bar Q^*(a, W) - \Psi_a

    A mapping rather than the ``(psi1, IC1, psi0, IC0)`` tuple this used to return: the
    four-tuple could not describe a third arm, and unpacking it was the last thing in the
    estimand layer that counted the arms.

    ``submodel`` must be the *unweighted* mean submodel even when the fluctuation was fit
    in weighted form: the influence curve is defined by the true clever covariate, not by
    the reparameterisation used to fit it.
    """
    if submodel.group != "mean":
        raise ValueError(f"expected the 'mean' submodel; got {submodel.group!r}")
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _residual(outcome, targeted, observed)

    out: dict[float, ArmMean] = {}
    for arm in targeted.levels:
        prediction = targeted.arms[arm]
        psi = float(np.average(prediction, weights=w))
        # Summed in this association deliberately.  Splitting the bracket to reuse
        # ICParts here would be mathematically identical and would change the last bit of
        # every influence curve, because floating-point addition is not associative. The
        # decomposition is a diagnostic (`counterfactual_mean_parts`); the estimation path
        # keeps the arithmetic its regression fixtures were built against.
        out[arm] = ArmMean(psi, w * (submodel.column_for(arm) * residual + prediction - psi))
    return out


class ICParts(NamedTuple):
    r"""The two halves of an influence curve, kept apart.

    .. math::

        D^*(O) = \underbrace{H(A, W)\,\{Y - Q^*(A, W)\}}_{\text{residual}}
                 + \underbrace{Q^*(a, W) - \Psi}_{\text{plug-in}}

    They answer different questions, and summing them immediately -- which this code
    used to do -- throws that away.  A heavy tail in the *residual* term is a
    positivity artefact: one unit with a large inverse-propensity weight dominating
    the estimating equation.  A heavy tail in the *plug-in* term is genuine outcome
    heterogeneity, which no amount of truncation will fix.  Their relative size also
    says how much work targeting is doing.
    """

    residual: FloatArray
    plugin: FloatArray

    @property
    def total(self) -> FloatArray:
        return np.asarray(self.residual + self.plugin, dtype=float)

    def shares(self) -> dict[str, float]:
        """Each term's share of the influence curve's variance."""
        total = float(np.var(self.total))
        if total <= 0:
            return {"residual": float("nan"), "plugin": float("nan")}
        return {
            "residual": float(np.var(self.residual)) / total,
            "plugin": float(np.var(self.plugin)) / total,
        }


def counterfactual_mean_parts(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
) -> dict[float, ICParts]:
    """The decomposed influence curves behind :func:`counterfactual_means`, per arm.

    ``parts.total`` agrees with the summed curve to floating-point rounding rather
    than bit-for-bit: the sum there is bracketed differently, and addition is not
    associative.  The gap is a few ULP and is asserted in
    ``tests/unit/test_ic_parts.py``; use :func:`counterfactual_means` for the
    estimate and this for the diagnostic.
    """
    if submodel.group != "mean":
        raise ValueError(f"expected the 'mean' submodel; got {submodel.group!r}")
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _residual(outcome, targeted, observed)
    return {
        arm: ICParts(
            w * submodel.column_for(arm) * residual,
            w * (targeted.arms[arm] - float(np.average(targeted.arms[arm], weights=w))),
        )
        for arm in targeted.levels
    }


def att_estimate(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    treatment: FloatArray,
    weights: FloatArray,
    observed: BoolArray | None = None,
) -> tuple[float, FloatArray]:
    """ATT point estimate and influence curve, on the scaled outcome scale."""
    if submodel.group != "att":
        raise ValueError(f"expected the 'att' submodel; got {submodel.group!r}")
    return _conditional_effect(outcome, targeted, submodel, treatment, weights, observed, arm=1.0)


def atc_estimate(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    treatment: FloatArray,
    weights: FloatArray,
    observed: BoolArray | None = None,
) -> tuple[float, FloatArray]:
    """ATC point estimate and influence curve, on the scaled outcome scale."""
    if submodel.group != "atc":
        raise ValueError(f"expected the 'atc' submodel; got {submodel.group!r}")
    return _conditional_effect(outcome, targeted, submodel, treatment, weights, observed, arm=0.0)


def _conditional_effect(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    treatment: FloatArray,
    weights: FloatArray,
    observed: BoolArray | None,
    *,
    arm: float,
) -> tuple[float, FloatArray]:
    """Effect conditional on being in ``arm``, i.e. the ATT (1) or ATC (0)."""
    a = np.asarray(treatment, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    indicator = a if arm == 1.0 else 1.0 - a
    share = float(np.average(indicator, weights=w))
    if share <= 0:
        raise ValueError(f"no observations in arm {arm:.0f}: the estimand is undefined")

    contrast = targeted.arms[1.0] - targeted.arms[0.0]
    psi = float(np.average(contrast, weights=w * indicator))
    residual = _residual(outcome, targeted, observed)
    # The sole column, not an arm's: a contrast submodel has no per-arm column, which is
    # why ``arm_columns`` is empty for this group.
    ic = w * (submodel.observed[:, 0] * residual + (indicator / share) * (contrast - psi))
    return psi, ic


def ratio_estimates(
    psi_one: float,
    ic_one: FloatArray,
    psi_zero: float,
    ic_zero: FloatArray,
    *,
    n: int,
    cluster: IntArray | None = None,
    alpha: float = 0.05,
    which: tuple[str, ...] = ("rr", "or"),
) -> dict[str, ParameterEstimate]:
    """Risk ratio and odds ratio, with inference on the log scale.

    Only meaningful for a binary outcome, where the counterfactual means are risks
    in ``(0, 1)``; the caller is responsible for not asking otherwise.
    """
    out: dict[str, ParameterEstimate] = {}
    if "rr" in which:
        log_psi, ic = log_ratio_influence(psi_one, ic_one, psi_zero, ic_zero)
        out["rr"] = make_estimate(
            "rr",
            float(np.exp(log_psi)),
            ic,
            n=n,
            cluster=cluster,
            scale="ratio",
            alpha=alpha,
            log_psi=log_psi,
        )
    if "or" in which:
        log_psi, ic = log_odds_ratio_influence(psi_one, ic_one, psi_zero, ic_zero)
        out["or"] = make_estimate(
            "or",
            float(np.exp(log_psi)),
            ic,
            n=n,
            cluster=cluster,
            scale="ratio",
            alpha=alpha,
            log_psi=log_psi,
        )
    return out


def unscale(
    psi: float, ic: FloatArray, scaler: OutcomeScaler, scale: Scale
) -> tuple[float, FloatArray]:
    """Map an estimate and its influence curve back to the original outcome scale.

    A level picks up the location shift, a difference does not, and an influence
    curve never does -- it is already centred.
    """
    if scale == "ratio":
        if not scaler.is_identity:
            raise ValueError("ratio estimands are only defined for an unscaled (binary) outcome")
        return psi, np.asarray(ic, dtype=float)
    if scaler.is_identity:
        return psi, np.asarray(ic, dtype=float)
    value = scaler.unscale_level(psi) if scale == "level" else scaler.unscale_difference(psi)
    return value, scaler.unscale_influence(ic)
