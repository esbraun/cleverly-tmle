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

import warnings
from collections.abc import Mapping, Sequence
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
    "average_estimates",
    "counterfactual_mean_parts",
    "counterfactual_means",
    "ipsi_means",
    "make_estimate",
    "msm_coefficients",
    "ratio_estimates",
    "regime_means",
    "shift_means",
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


def average_estimates(
    per_repeat: Sequence[Mapping[str, ParameterEstimate]],
    *,
    cluster: IntArray | None = None,
) -> dict[str, ParameterEstimate]:
    r"""Average estimates from repeated cross-fitting into one.

    Each entry of ``per_repeat`` is the report from one independent draw of the whole
    cross-fitting split.  Every row is out of fold in every draw, so the average

    .. math:: \bar\psi = \frac{1}{R}\sum_r \psi_r

    is the same functional of the same data with the fold noise averaged down, and its
    influence curve is :math:`\frac{1}{R}\sum_r \mathrm{IC}_r`.  The variance is then
    recomputed from that averaged curve rather than pooled from the per-draw variances,
    which is what keeps the reported variance the variance *of the reported curve* -- and
    with it the delta method, the cluster-robust variance, the simultaneous bands and the
    score diagnostic, all of which read the curve and not the variance.

    Not to be confused with :func:`~cleverly.estimators.tmle._average_over_folds`, which
    also averages a report but along the other axis and by a different rule.  Folds are
    **stitched** by index -- a row appears in exactly one validation fold, so its curve
    has exactly one fold-specific value.  Repeats are **averaged** elementwise -- a row
    appears in every repeat, so it has :math:`R` of them.  Unifying the two would have to
    get that difference wrong in one direction or the other.

    Ratios average on the log scale, where their influence curve and Wald interval live,
    so ``psi == exp(log_psi)`` holds and :attr:`ParameterEstimate.ci` stays on the
    boundary-respecting scale.

    Parameters
    ----------
    per_repeat:
        One mapping per draw.  A one-element sequence returns its input unchanged, so an
        ordinary fit is untouched by the averaging path.
    cluster:
        Cluster codes, so the variance of the averaged curve is taken at the independent
        sampling unit exactly as a single fit's is.

    The confidence level is not a parameter here: every draw was produced by the same
    ``retarget`` call under the same ``alpha_sig``, so it is carried through from the
    per-draw estimates rather than re-declared at the point of averaging, where a
    disagreeing value could only be a mistake.
    """
    if not per_repeat:
        raise ValueError("average_estimates needs at least one repeat to average")
    if len(per_repeat) == 1:
        return dict(per_repeat[0])

    shared = [name for name in per_repeat[0] if all(name in report for report in per_repeat)]
    dropped = [name for name in per_repeat[0] if name not in shared]
    if dropped:
        # Silence here would report an average over a subset of the draws under the same
        # name as an average over all of them.
        warnings.warn(
            f"{', '.join(dropped)} was estimated in some cross-fitting repeats but not "
            "in all of them -- a draw whose targeting step failed, or a fold with no "
            "units in the conditioning arm -- so it is omitted from the averaged report "
            "rather than averaged over fewer draws than the rest.",
            UserWarning,
            stacklevel=2,
        )

    out: dict[str, ParameterEstimate] = {}
    for name in shared:
        parts = [report[name] for report in per_repeat]
        influence_curve = np.mean(
            [np.asarray(part.influence_curve, dtype=float) for part in parts], axis=0
        )
        scale = parts[0].scale
        log_psi: float | None = None
        if scale == "ratio":
            log_psi = float(np.mean([part.log_psi for part in parts]))
            psi = float(np.exp(log_psi))
        else:
            psi = float(np.mean([part.psi for part in parts]))
        out[name] = make_estimate(
            name,
            psi,
            influence_curve,
            n=parts[0].n,
            cluster=cluster,
            scale=scale,
            alpha=parts[0].alpha,
            log_psi=log_psi,
        )
    return out


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


def shift_means(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
) -> dict[float, ArmMean]:
    r"""Each shift's mean and influence curve, keyed by shift code.

    .. math::

        \hat\Psi_r = \frac1n \sum_i \bar Q^*\bigl(d_r(A_i, W_i), W_i\bigr),
        \qquad
        D_r^*(O) = h_r(A, W)\,\{Y - \bar Q^*(A, W)\}
                   + \bar Q^*\bigl(d_r(A,W), W\bigr) - \Psi_r

    **Why this is not** :func:`regime_means` **at the induced density.**  A shift
    :math:`d` induces the density :math:`g^d(b \mid w) = \sum_{a : d(a,w)=b} g(a \mid w)`,
    and the two parameters are equal:

    .. math::

        E[\bar Q(d(A,W), W)]
          = E_W\Bigl[\sum_a g(a \mid W) \bar Q(d(a,W), W)\Bigr]
          = E_W\Bigl[\sum_b g^d(b \mid W) \bar Q(b, W)\Bigr].

    The clever covariates are equal too, entry for entry.  The **influence curves are
    not**.  A regime's plug-in term is :math:`\sum_b g^d(b \mid W)\,\bar Q^*(b, W)`, a
    function of :math:`W` alone; a shift's is :math:`\bar Q^*(d(A,W), W)`, which reads the
    dose the unit actually received.  The two agree only in conditional expectation given
    :math:`W`, so their difference

    .. math::

        \bar Q^*(d(A,W), W) - E\bigl[\bar Q^*(d(A,W), W) \mid W\bigr]

    is mean zero given :math:`W` and uncorrelated with the regime curve -- the residual
    half of that curve is centred given :math:`(A, W)`, and its plug-in half is
    :math:`W`-measurable.  Hence the exact identity

    .. math::

        \operatorname{Var}(D^*_{\text{mtp}})
          = \operatorname{Var}(D^*_{\text{regime}})
          + \operatorname{Var}\bigl(\bar Q(d(A,W),W) - E[\bar Q(d(A,W),W) \mid W]\bigr).

    An MTP is strictly *harder* to estimate than the known stochastic regime with the same
    mean, by exactly that amount -- the price of an intervention that reads the natural
    value of treatment.  So this must not delegate to :func:`regime_means`, and
    ``tests/unit/test_influence_gateaux_shift.py`` keeps a negative control that fails if
    someone later makes it.

    Arithmetically this *is* :func:`counterfactual_means` with the arm replaced by the
    shift: ``arm_columns`` is populated for the ``mtp`` group, so ``column_for`` answers,
    and ``targeted.arms[r]`` already holds :math:`\bar Q^*` at the shifted dose.  The
    bracket below is summed in the same association for the same reason the comment there
    gives.
    """
    if submodel.group != "mtp":
        raise ValueError(f"expected the 'mtp' submodel; got {submodel.group!r}")
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _residual(outcome, targeted, observed)

    out: dict[float, ArmMean] = {}
    for code in targeted.levels:
        prediction = targeted.arms[code]
        psi = float(np.average(prediction, weights=w))
        out[code] = ArmMean(psi, w * (submodel.column_for(code) * residual + prediction - psi))
    return out


def regime_means(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    regimes: FloatArray,
    weights: FloatArray,
    observed: BoolArray | None = None,
) -> dict[float, ArmMean]:
    r"""Every regime's counterfactual mean and influence curve, keyed by regime code.

    .. math::

        \hat\Psi_r = \frac1n \sum_i \sum_a g^\star_r(a \mid W_i)\, \bar Q^*(a, W_i),
        \qquad
        D_r^*(O) = h_r(A, W)\,\{Y - \bar Q^*(A, W)\}
                   + \sum_a g^\star_r(a \mid W)\,\bar Q^*(a, W) - \Psi_r

    The counterpart of :func:`counterfactual_means` for the ``regime`` fluctuation, and
    the same influence curve with :math:`\mathbb 1\{a = v\}` in place of
    :math:`g^\star_r` -- which is why a static regime reproduces the arm result.

    Two details are not interchangeable with the arm version.  The mixture over arms is
    taken **after** fluctuation, on :math:`\bar Q^*` rather than :math:`\bar Q^0`: a
    plug-in of the *targeted* distribution is what makes this a substitution estimator
    that solves the score equation.  And the residual column is read off the submodel
    directly, because :attr:`~cleverly.fluctuation.submodel.Submodel.arm_columns` is
    empty here -- no column belongs to an arm, so
    :meth:`~cleverly.fluctuation.submodel.Submodel.column_for` has nothing to answer.

    ``regimes`` is the ``(n, K, R)`` density; ``submodel`` must be the *unweighted*
    ``regime`` submodel even when the fluctuation was fit in weighted form, for the
    reason :func:`counterfactual_means` gives.
    """
    if submodel.group != "regime":
        raise ValueError(f"expected the 'regime' submodel; got {submodel.group!r}")
    star = np.asarray(regimes, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _residual(outcome, targeted, observed)
    levels = targeted.levels
    if star.shape[1] != len(levels):
        raise ValueError(
            f"regimes has {star.shape[1]} arm column(s) but the targeted fit has "
            f"{len(levels)} arm(s) {list(levels)}"
        )
    if star.shape[2] != submodel.dim:
        raise ValueError(
            f"regimes describes {star.shape[2]} regime(s) but the submodel has "
            f"{submodel.dim} column(s)"
        )
    # (n, K) of the targeted predictions, columns in the same order the densities use.
    predictions = np.column_stack([targeted.arms[level] for level in levels])

    out: dict[float, ArmMean] = {}
    for index in range(star.shape[2]):
        mixture = np.einsum("ij,ij->i", star[:, :, index], predictions)
        psi = float(np.average(mixture, weights=w))
        out[float(index)] = ArmMean(
            psi, w * (submodel.observed[:, index] * residual + mixture - psi)
        )
    return out


def ipsi_means(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    incremental: Any,
    treatment: FloatArray,
    weights: FloatArray,
    observed: BoolArray | None = None,
) -> dict[float, ArmMean]:
    r"""Every tilt's counterfactual mean and influence curve, keyed by tilt code.

    .. math::

        \hat\Psi_r = \frac1n \sum_i m_r(W_i),
        \qquad
        m_r(W) = \sum_a q_{\delta_r}(a \mid W)\, \bar Q^*(a, W)

        D_r^*(O) = h_r(A, W)\,\{Y - \bar Q^*(A, W)\}
                 + \frac{\delta_r\{\bar Q^*(1,W) - \bar Q^*(0,W)\}}{D_{\delta_r}^2}\,(A - g^*)
                 + m_r(W) - \Psi_r

    **The middle term is the whole reason this is not** :func:`regime_means`.  A
    :class:`~cleverly.interventions.Stochastic` regime evaluated at the very same density
    :math:`q_{\delta_r}` has the same mean and, entry for entry, the same clever
    covariate; its influence curve is this one without the middle term.  Because
    :math:`q_\delta` is built out of :math:`g`, the parameter moves when the mechanism
    does, and that dependence is a pathwise derivative the regime curve has no term for.

    The gap is not a wash.  The extra term is mean zero given :math:`W` and orthogonal to
    both halves of the regime curve -- the residual half is centred given :math:`(A, W)`,
    the plug-in half is :math:`W`-measurable -- so

    .. math::

        \operatorname{Var}(D^*_{\text{ipsi}})
            = \operatorname{Var}(D^*_{\text{regime}})
            + \operatorname{Var}\!\left(
                \frac{\delta(\bar Q(1,W) - \bar Q(0,W))}{D_\delta^2}(A - g)\right),

    an exact decomposition.  Treating an incremental intervention as the regime that
    induces it therefore does not merely report a different quantity: it reports a
    standard error that is too *small*, always.
    ``tests/unit/test_influence_gateaux_ipsi.py`` keeps that identity as a negative
    control, on the same terms ``tests/unit/test_influence_gateaux_shift.py`` does for
    the shift axis.

    ``incremental`` is the :class:`~cleverly.interventions.IPSISet` **as targeted** -- its
    :attr:`~cleverly.interventions.IPSISet.propensity` must be the fluctuated mechanism,
    not the initial one, or the middle term would be evaluated at a :math:`g` the plug-in
    did not use.  Carrying the mechanism on the same object as the density is what makes
    that impossible to get wrong rather than merely documented.
    """
    if submodel.group != "ipsi":
        raise ValueError(f"expected the 'ipsi' submodel; got {submodel.group!r}")
    density = np.asarray(incremental.values, dtype=float)
    derivative = np.asarray(incremental.derivative, dtype=float)
    mechanism = np.asarray(incremental.propensity, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _residual(outcome, targeted, observed)
    levels = targeted.levels
    if density.shape[1] != len(levels):
        raise ValueError(
            f"the tilted density has {density.shape[1]} arm column(s) but the targeted fit "
            f"has {len(levels)} arm(s) {list(levels)}"
        )
    if density.shape[2] != submodel.dim:
        raise ValueError(
            f"the tilt describes {density.shape[2]} intervention(s) but the submodel has "
            f"{submodel.dim} column(s)"
        )
    # (n, K) of the targeted predictions, columns in the same order the densities use.
    predictions = np.column_stack([targeted.arms[level] for level in levels])
    # The blip is a contrast of arms, so it needs the two-arm layout the tilt declares;
    # IPSISet.evaluate has already refused anything else.
    blip = np.asarray(targeted.arms[1.0], dtype=float) - np.asarray(targeted.arms[0.0], dtype=float)
    treated_residual = np.asarray(treatment, dtype=float).reshape(-1) - mechanism

    out: dict[float, ArmMean] = {}
    for index in range(density.shape[2]):
        mixture = np.einsum("ij,ij->i", density[:, :, index], predictions)
        psi = float(np.average(mixture, weights=w))
        curve = (
            submodel.observed[:, index] * residual
            + derivative[:, index] * blip * treated_residual
            + mixture
            - psi
        )
        out[float(index)] = ArmMean(psi, w * curve)
    return out


def _raw_predictions(
    targeted: InitialFit, levels: tuple[float, ...], scaler: OutcomeScaler
) -> FloatArray:
    """``(n, K)`` targeted predictions on the *original* outcome scale, arms in order."""
    stacked = np.column_stack([targeted.arms[level] for level in levels])
    if scaler.is_identity:
        return stacked
    return np.asarray(scaler.lower + scaler.range * stacked, dtype=float)


def msm_coefficients(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    design: FloatArray,
    model_weights: FloatArray,
    weights: FloatArray,
    scaler: OutcomeScaler,
    observed: BoolArray | None = None,
) -> dict[float, ArmMean]:
    r"""A working model's coefficients and their influence curves, keyed by coefficient.

    .. math::

        \hat\beta = M^{-1} P_n \Big[\sum_a h(a, V)\,\varphi(a, V)\,\bar Q^*(a, W)\Big],
        \qquad
        M = P_n\Big[\sum_a h(a, V)\,\varphi(a, V)\varphi(a, V)^\top\Big]

    .. math::

        D_\beta^*(O) = M^{-1}\Big[
            H(A, W)\,\{Y - \bar Q^*(A, W)\}
            + \sum_a h(a, V)\,\varphi(a, V)\,
              \big\{\bar Q^*(a, W) - \varphi(a, V)^\top\hat\beta\big\}
        \Big]

    -- the standard M-estimation form.  :math:`M` carries no further term because
    :math:`U(\beta_0, \bar Q_0) = 0` at the truth, and no nuisance because :math:`h` and
    :math:`\varphi` are known functions; see :mod:`cleverly.msm` for the estimand and for
    why the weights have to be known for that to hold.

    Both terms are zero in the sample after targeting, and for different reasons.  The
    first is the ``msm`` fluctuation's own score equation.  The second is zero *by
    construction*: :math:`\hat\beta` is the weighted least-squares solution against
    :math:`\bar Q^*`, so the residuals it leaves are orthogonal to the design.  That is
    what makes this a one-fluctuation TMLE rather than an iteration between
    :math:`\epsilon` and :math:`\beta`.

    **The projection is solved on the original outcome scale**, unlike every other
    estimand here, which works on the scaled outcome and maps back afterwards.  A
    coefficient vector has no single :class:`Scale` to map back *with*: writing
    :math:`\bar Q^*_{\text{raw}} = \ell + r\,\bar Q^*_{\text{scaled}}` gives
    :math:`\beta_{\text{raw}} = \ell M^{-1} P_n[\sum_a h \varphi] + r\,
    \beta_{\text{scaled}}`, and the first term collapses to "the intercept picks up
    :math:`\ell`, the slopes pick up nothing" *only* when the design happens to contain an
    intercept column.  Solving where the coefficients are reported removes the question
    rather than requiring a design to promise something.  Nothing is lost: the residual
    rescales by the same :math:`r`, so a score that is zero on one scale is zero on the
    other.  The consequence for callers is that these estimates must **not** go through
    :meth:`~cleverly.targets.TargetContext.finish`, which would unscale a second time.

    Parameters
    ----------
    design:
        ``(n, K, p)`` array :math:`\varphi(a, V)`, arms in ``targeted.levels`` order.
    model_weights:
        ``(n, K)`` array :math:`h(a, V)`.
    weights:
        The ``(n,)`` *observation* weights, which are a different thing: they tilt the
        population the projection is taken over, while ``model_weights`` says how the
        arms are traded off within it.
    scaler:
        Maps the targeted predictions back off ``[0, 1]``.

    Both arrays are passed plainly rather than as a
    :class:`~cleverly.msm.MSMSet`, on the same terms as ``regimes`` and ``shifts``: the
    inference layer is written against arrays so that it does not depend on the objects
    that produced them.
    """
    if submodel.group != "msm":
        raise ValueError(f"expected the 'msm' submodel; got {submodel.group!r}")
    phi = np.asarray(design, dtype=float)
    h = np.asarray(model_weights, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    levels = targeted.levels
    if phi.ndim != 3 or phi.shape[1] != len(levels):
        raise ValueError(
            f"the working model's design must have shape (n, {len(levels)}, p) for arms "
            f"{list(levels)}; got {phi.shape}"
        )
    if phi.shape[2] != submodel.dim:
        raise ValueError(
            f"the working model has {phi.shape[2]} term(s) but the submodel has "
            f"{submodel.dim} column(s)"
        )
    if h.shape != phi.shape[:2]:
        raise ValueError(
            f"the working model's weights must have shape {phi.shape[:2]}; got {h.shape}"
        )

    residual = scaler.unscale_influence(_residual(outcome, targeted, observed))
    predictions = _raw_predictions(targeted, levels, scaler)
    mass = float(w.sum())

    weighted_design = phi * h[:, :, None]  # (n, K, p) -- h * phi
    gram = np.einsum("ijp,ijq,ij,i->pq", phi, phi, h, w) / mass
    moment = np.einsum("ijp,ij,i->p", weighted_design, predictions, w) / mass
    beta = np.linalg.solve(gram, moment)

    # (n, p): the plug-in half, sum_a h * phi * (Qbar* - m(a, V; beta)).
    plugin = np.einsum("ijp,ij->ip", weighted_design, predictions - phi @ beta)
    contribution = w[:, None] * (submodel.observed * residual[:, None] + plugin)
    influence = np.linalg.solve(gram, contribution.T).T

    return {
        float(j): ArmMean(float(beta[j]), np.ascontiguousarray(influence[:, j]))
        for j in range(phi.shape[2])
    }


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
