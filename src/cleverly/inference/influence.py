r"""Estimands and the influence curves their estimates are reported from.

Efficient influence curves, everywhere but one: :func:`reduced_corrections` builds the two
terms a doubly-robust fit *subtracts* from :math:`D^*`, and the difference is the
estimator's own asymptotic influence function at the nuisance limits rather than the
canonical gradient at :math:`P_0`.  The distinction is set out there and in
:mod:`cleverly.estimators.drtmle`; it is called out in this first line because a reader who
knows that TMLE's curve is the efficient one has no other reason to think this module holds
anything else.

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
    \mathrm{IC}^{ATT[a]}_i  &= h_a(A_i, W_i)\, r_i
        + \frac{\mathbb 1\{A_i = a\}}{P(A = a)}
          \bigl(\bar Q^*(a, W_i) - \bar Q^*(r, W_i) - \psi_{a}\bigr)

with :math:`r` the reference arm, one such curve per non-reference arm, and the ATC
mirrors it with :math:`\mathbb 1\{A_i = r\} / P(A = r)` in place of the arm's own
indicator -- every one of its parameters conditioning on the same population.  With two
arms this is the classic ATT, :math:`a = 1` and :math:`r = 0`.  Note that these curves
carry an extra term beyond "clever covariate times residual": the estimand conditions on a
*random* event (``A = a``), so the uncertainty in ``P(A = a)`` contributes.  Omitting
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
from ..msm import link_for, solve_projection
from ..utils.bounds import OutcomeScaler, bound
from .cluster import influence_variance
from .delta import log_odds_ratio_influence, log_ratio_influence, normal_ci, two_sided_pvalue

__all__ = [
    "CorrectionParts",
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
    "missing_outcome_correction_parts",
    "msm_coefficients",
    "ratio_estimates",
    "reduced_correction_parts",
    "reduced_corrections",
    "regime_means",
    "shift_means",
]

Scale = Literal["level", "difference", "ratio", "fraction"]


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

        The null is zero for a level, difference or fraction and one for a ratio
        (i.e. zero on the log scale).
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


def _expect(submodel: Submodel, group: str) -> None:
    """Refuse a submodel from the wrong group.

    Every curve here reads a fluctuation built for one score equation, and reading the
    wrong one would not raise on its own -- the shapes line up and the arithmetic would
    quietly answer for a different parameter.  Hence a guard at the top of each, and
    hence one function rather than the eight copies of the same two lines it replaces.
    """
    if submodel.group != group:
        raise ValueError(f"expected the {group!r} submodel; got {submodel.group!r}")


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

    Also the carrier for the other things a fluctuation produces one of per parameter: a
    working model's coefficients (:func:`msm_coefficients`) and the conditional effects
    (:func:`att_estimate`).  What is shared is the shape -- an estimate and the curve it
    is entitled to -- which is all anything downstream reads.
    """

    psi: float
    influence_curve: FloatArray


def counterfactual_means(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
    corrections: Mapping[float, FloatArray] | None = None,
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

    ``corrections`` are :func:`reduced_corrections`' :math:`D^*_Q + D^*_g`, **subtracted**,
    for a fit that solved the doubly-robust equations.  ``None`` -- every fit that did not --
    leaves the expression below untouched character for character, which matters: see the
    comment on the sum.  The subtraction cannot move :math:`\hat\Psi`, since the targeting
    drove all three empirical means to zero; what it moves is the variance, which is the
    whole of what that variant buys.
    """
    _expect(submodel, "mean")
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
        if corrections is None:
            out[arm] = ArmMean(psi, w * (submodel.column_for(arm) * residual + prediction - psi))
        else:
            out[arm] = ArmMean(
                psi,
                w
                * (
                    submodel.column_for(arm) * residual
                    + prediction
                    - psi
                    - np.asarray(corrections[arm], dtype=float)
                ),
            )
    return out


def reduced_corrections(
    outcome: FloatArray,
    targeted: InitialFit,
    treatment: FloatArray,
    reduced: Any,
    propensity: FloatArray,
    *,
    bounds: tuple[float, float],
    guard: tuple[str, ...],
    observed: BoolArray | None = None,
) -> dict[float, FloatArray]:
    r""":math:`D^*_Q + D^*_g` per arm, the terms doubly-robust inference subtracts.

    .. math::

        D^*_g &= \frac{Q_r(a, W)}{g^*(a|W)}\,\{1_a - g^*(a|W)\} \\
        D^*_Q &= 1_a\,\frac{g_{r,2}(a|W)}{g_{r,1}(a|W)}\,\{Y - \bar Q^*(a, W)\}

    Randomized missing-outcome fits do not enter this two-correction function. They use
    :func:`missing_outcome_correction_parts`, which keeps the paper's treatment,
    observation and outcome blocks separate.

    In either setting the reported curve is :math:`D^* - D^*_Q - D^*_g`.  **Minus**, both
    of them: that is
    what ``drtmle`` computes and what Theorem 1 derives (see below), and a sum is the
    plausible transcription error.  Since the
    targeting drove all three empirical means to zero, the combination cannot move the point
    estimate however the signs go -- it moves only the variance, so nothing that reports
    :math:`\hat\Psi` can catch getting this wrong.  ``tests/unit/test_influence_drtmle.py``
    carries the sign as a negative control at deliberately wrong nuisances, which is the
    only place it is visible: at the truth :math:`Q_r` and :math:`g_{r,2}` vanish row by row
    and the reported curve *equals* :math:`D^*` array for array.

    **Validity is not efficiency.**  Under misspecification the canonical gradient at
    :math:`P_0` is still :math:`D^*`; what this subtraction leaves is the **estimator's**
    asymptotic influence function at the nuisance limits, and the estimator is generally not
    efficient there.  So a doubly-robust interval is one entitled to be believed under weaker
    conditions -- not a narrower one, and not an efficient one.  When both nuisances are
    consistent :math:`Q_r` and :math:`g_{r,2}` vanish row by row, the two curves coincide, and
    the fit is the ordinary efficient estimator; that is the case the variant is *not* for.
    Worth saying explicitly because the numbers invite the opposite reading -- in the guide's
    own worked example the corrected standard error is the smaller of the two.

    **The sign was challenged and the challenge is answered.**  The 2016 working paper's §3.1
    *display* defines the mechanism-side correction with a leading minus,
    :math:`D_A = -Q_r/g\,(A - g)`, and Theorem 1 subtracts :math:`D_A` -- which read together
    would make the theorem's curve :math:`D^* + Q_r/g\,(A - g)`, the opposite of this
    function's. The sign discrepancy is **resolved in favour of this implementation** on
    the paper's own appendices: each derives its block as
    :math:`P_0[\text{term}] = -(P_n - P_0)D + P_n D + (\text{second order})`, an identity
    satisfiable only with :math:`D` equal to the **positive** term, so appendix A forces
    :math:`D_A = +Q_r/g\,(A - g)` and appendix B forces :math:`D_Y` to be the positive
    quantity below.  Theorem 1's own :math:`\sigma^2_n = P_n\{D^* - D_A - D_Y\}^2` is then
    exactly what this function's caller computes.  The paper also prints :math:`D_Y` twice
    with two signs, which is the other reason not to settle it from a display.
    ``tests/unit/test_theorem_drtmle.py`` checks the appendix step that fixes the
    orientation, and pins these arrays against the theorem's terms;
    ``docs/drtmle.md``'s *The sign of the mechanism correction* carries the argument.

    The *published* 2017 article remains unread and no longer gates this: the adjudication is
    internal consistency plus exact-law arithmetic, and neither depends on the edition.

    **And the decomposition is now pinned against a perturbation of the law**, which is the
    check every other estimand in this package gets and the one this variant could not have
    until a fixture was made wrong on purpose.  In each half of the union model -- one
    nuisance consistent, the other a declared constant -- with the reduced regressions
    saturated, the corrected curve *is* the efficient influence function row for row:

    .. math::

        \frac{1}{g_1} - \frac{g_{r,2}}{g_{r,1}} = \frac{1}{g_0},
        \qquad
        D^* (\bar Q^*, g_0) - D^*_g = D^*(\bar Q_0, g_0),

    the second because the :math:`\bar Q^*` in the residual and the one inside
    :math:`Q_r = \bar Q_0 - \bar Q^*` cancel.  That is what "an interval valid when only one
    nuisance is consistent" means, written as an identity, and
    ``tests/unit/test_influence_gateaux_drtmle.py`` checks it against
    ``tests/discrete_law.py``'s complex-step derivative -- from a real fit as well as from
    longhand, at ``rtol=0``.  A flipped sign misses by ``0.55`` to ``2.8`` against a
    ``1e-12`` window, which was watched rather than argued.

    And note exactly what the reference to ``drtmle`` is: **no number this package produces
    has been compared against that package's, and none will be.**  There is no cross-language
    test here or in CI and none is planned.  So "what that package computes" is a statement
    about a formula read out of its source -- **provenance** -- and not about agreement with
    anything it returns.  That is a decision and not a gap: both implementations descend from
    one source, so agreement would be evidence about the transcription and blind to exactly
    the error above, which is why the sign was settled by the appendices instead. The roadmap's
    standing decisions carry the reasoning for using independent validation, and
    ``docs/architecture-invariants.md`` records the corresponding development rule.

    **The two terms are built by** :func:`reduced_correction_parts` **and added here**, so
    that :func:`~cleverly.validation.drtmle.correction_check` takes each one's empirical
    mean from this expression rather than from a second copy of it.  A second copy is how
    an identity check comes to agree with a curve neither of them is.

    **One correction per equation the fit actually solved**, which is what ``guard`` selects
    and is the crossing ``guard=`` has everywhere else: :math:`D^*_g` is equation (9)'s, the
    one the ``"Q"`` guard adds, and :math:`D^*_Q` is equation (10)'s, the one ``"g"`` adds.
    A fit guarding one nuisance subtracts one term. Subtracting both would leave the unsolved
    equation's arbitrary mean in the curve, measured at
    :math:`2.8\times10^{-3}` on a ``guard=("g",)`` fit against a :math:`7.7\times10^{-6}` bar
    with **no** row clipped, so it is not the bounded-mechanism centring defect in another
    guise. The derivation was
    already in the repository -- ``tests/unit/test_remainder_drtmle.py`` adds each correction
    only under the guard whose equation removes it, and shows that two guards over-correct on
    an exact law; ``tests/unit/test_drtmle_fit.py`` fits one end to end.  ``DRTMLE``'s own
    default is both guards, so the ordinary fit is unaffected.

    Parameters
    ----------
    targeted, propensity:
        The **targeted** regression and the targeted :math:`g^*(a_1 \mid W)` as an ``(n,)``
        array, both as the alternation left them.  Reading the initial mechanism here would
        report a curve for a fit nobody ran, since the equations were solved at the tilted
        one.
    reduced:
        The :class:`~cleverly.estimators.reduced.ReducedSet` the equations were finally
        solved against -- the refit, not the fit's own.  Typed loosely to keep this module
        free of :mod:`cleverly.estimators`, which imports it.
    bounds:
        The same mechanism truncation the clever covariates divided by.
    guard:
        Which of the two extra equations this fit solved, in ``DRTMLE.guard``'s vocabulary,
        and so which corrections belong in its curve.  **Required, with no default**, which
        is the point: an earlier caller did not pass this, and a default of both would
        make that caller's mistake the fallback for the next one.
    """
    return reduced_correction_parts(
        outcome,
        targeted,
        treatment,
        reduced,
        propensity,
        bounds=bounds,
        observed=observed,
        guard=guard,
    ).total()


@dataclass(frozen=True)
class CorrectionParts:
    r"""The two corrections kept apart, plus what the mechanism truncation absorbed.

    :func:`reduced_corrections` is the sum of the first two and is what the reported curve
    subtracts.  They are built here rather than there so that
    :func:`~cleverly.validation.drtmle.correction_check` can take each one's empirical mean
    **from the same expression the curve carries** -- an identity checked against a second
    implementation of the same formula is not an identity, and this is the one class of
    defect that check exists to catch.

    Attributes
    ----------
    d_g, d_q:
        Rowwise :math:`D^*_g(a)` and :math:`D^*_Q(a)` per arm, on the ``[0, 1]`` scaled
        outcome that :math:`Q_r` and the fluctuation's residual both live on.
    clip_bias:
        Rowwise :math:`Q_r(a, W)/g^b(a|W)\,\{g(a|W) - g^b(a|W)\}` per arm -- the quantity
        called :math:`B_{clip}`, in the orientation that expression defines it in.
        It is **exactly** the difference between the mechanism score the
        alternation solves, at the raw tilted :math:`g^*`, and the mean of the
        :math:`D^*_g` above, which truncates :math:`g^*` in its residual as well as in its
        denominator -- *negated*, since the residual is
        :math:`1_a - g` in one and :math:`1_a - g^b` in the other:

        .. math::

            P_n[w\,D^*_g] - S_g^{\text{stored}} = P_n[w\,B_{clip}]

        Zero on every row the bound leaves alone, so its mean is zero whenever nothing
        clips.  It is a diagnostic and not a correction: nothing subtracts it.

        **It is now zero on every fit**, and that is the intended convention rather than a broken
        column. :func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism` runs at the
        ``DRTMLE``
        call sites, so the mechanism the alternation carries forward is already truncated
        and there is no second array left for this to measure the distance to.  What still
        varies between draws, and what a fixture is chosen on, is
        :attr:`~cleverly.validation.drtmle.CorrectionRow.margin`.
    clipped:
        Which rows the mechanism truncation binds on. Empty on every converged bounded fit for the
        same reason: a converged bounded tilt lies inside the bounds.  On record because
        "the identity holds" was uninformative on a draw where the bound never bit -- the
        degeneracy that hid the centring mismatch -- and because zero here is now the
        assertion that the bounded convention is active.
    guard:
        Which equations the fit solved, and so which of the two terms :meth:`total` puts in
        the curve.  **Both arrays are built whatever it says**, because the term a fit does
        *not* subtract is exactly what
        :func:`~cleverly.validation.drtmle.correction_check` reports as a diagnostic -- and
        it has to come from this expression rather than from a second copy of it, for the
        reason above.  It travels here rather than being read off the fluctuation twice, so
        that :func:`~cleverly.estimators.tmle.correction_parts` is the one place the guard
        is copied off the record and the curve and the check cannot select differently.
    """

    d_g: dict[float, FloatArray]
    d_q: dict[float, FloatArray]
    clip_bias: dict[float, FloatArray]
    clipped: BoolArray
    guard: tuple[str, ...]
    d_a: dict[float, FloatArray] | None = None
    d_m: dict[float, FloatArray] | None = None
    d_y: dict[float, FloatArray] | None = None

    def total(self) -> dict[float, FloatArray]:
        """What the curve subtracts: one correction per equation :attr:`guard` solved.

        :math:`D^*_Q + D^*_g` under both guards, and under both alone -- the ``"Q"`` guard
        is what adds equation (9), so it is what puts :math:`D^*_g` here, and ``"g"`` adds
        equation (10) and puts :math:`D^*_Q` here.  Membership rather than equality, since
        ``DRTMLE`` validates the guard's contents and not its order.

        The both-guards line below is character for character the single expression this
        replaced.  Re-associating it -- through ``sum()``, or a comprehension over a list
        of terms -- moves the last bit of every ordinary doubly-robust curve, for the
        reason :func:`counterfactual_means` records about its own arithmetic.

        An empty guard **raises**.  Such a fit fits no reduced regressions at all and never
        reaches here; returning zeros would make it the plain estimator recovered by a
        branch, which is the thing ``DRTMLE._nuisances``' short circuit exists to avoid.
        """
        has_q, has_g = "Q" in self.guard, "g" in self.guard
        # Aliased rather than copied on the single-guard branches: nothing mutates what
        # this returns -- the curve subtracts it and `_slice_fit`'s sibling indexing copies.
        if has_q and has_g:
            return {arm: np.asarray(self.d_g[arm] + self.d_q[arm], dtype=float) for arm in self.d_g}
        if has_q:
            return {arm: np.asarray(self.d_g[arm], dtype=float) for arm in self.d_g}
        if has_g:
            return {arm: np.asarray(self.d_q[arm], dtype=float) for arm in self.d_g}
        raise ValueError(
            "a doubly-robust curve needs at least one guard; an empty guard fits no "
            "reduced regressions and must not reach the corrections at all"
        )


def reduced_correction_parts(
    outcome: FloatArray,
    targeted: InitialFit,
    treatment: FloatArray,
    reduced: Any,
    propensity: FloatArray,
    *,
    bounds: tuple[float, float],
    guard: tuple[str, ...],
    observed: BoolArray | None = None,
) -> CorrectionParts:
    """:func:`reduced_corrections`' two terms before they are added, and the clipping bias.

    Every argument means what it means there.  Both terms are built whatever ``guard``
    says; it is :meth:`~cleverly.inference.influence.CorrectionParts.total` that selects,
    in the same association the single expression used, so the reported curve is unchanged
    to the last bit.
    """
    y = np.asarray(outcome, dtype=float).reshape(-1)
    a = np.asarray(treatment, dtype=float).reshape(-1)
    raw = np.asarray(propensity, dtype=float)
    if len(reduced.arms) == 2 and raw.ndim == 1:
        raw1 = raw.reshape(-1)
        g1 = bound(raw1, float(bounds[0]), float(bounds[1]))
        mechanism = {reduced.arms[0]: 1.0 - g1, reduced.arms[1]: g1}
        # The complement rather than a separately clipped array, exactly as
        # `Propensity.bounded` and `reduced_mechanism_covariate` take it.
        untruncated = {reduced.arms[0]: 1.0 - raw1, reduced.arms[1]: raw1}
        clipped = np.asarray(raw1 != g1, dtype=bool)
    else:
        if raw.shape != (y.size, len(reduced.arms)):
            raise ValueError(
                f"the targeted mechanism must be ({y.size}, {len(reduced.arms)}); got {raw.shape}"
            )
        bounded = bound(raw, float(bounds[0]), float(bounds[1]))
        mechanism = {arm: bounded[:, j] for j, arm in enumerate(reduced.arms)}
        untruncated = {arm: raw[:, j] for j, arm in enumerate(reduced.arms)}
        # Reduced **over the arms**, so this stays one bit per row as the binary branch's
        # is and as `clipped` is documented and reported: the arms are clipped column by
        # column here, and counting the cells would report "3800 row(s) of 2000" on a
        # three-armed fit whose bound binds on two arms at most rows.
        clipped = np.asarray((raw != bounded).any(axis=1), dtype=bool)
    ratio = np.asarray(reduced.gr2, dtype=float) / reduced.bounded_gr1(bounds)
    keep = np.ones(y.shape[0]) if observed is None else np.asarray(observed, dtype=float)

    d_g: dict[float, FloatArray] = {}
    d_q: dict[float, FloatArray] = {}
    clip_bias: dict[float, FloatArray] = {}
    for j, arm in enumerate(reduced.arms):
        # With missing outcomes the theorem's mechanism is the joint probability
        # P(A=a, Delta=1 | W), so its residual is I(A=a, Delta=1) - g_a.  This is
        # deliberately applied here as well as in the reduced-regression targets: a
        # missing mask on only one side is the canonical-source discrepancy that gated
        # this feature.
        indicator = (a == float(arm)).astype(float) * keep
        qr = np.asarray(reduced.qr, dtype=float)[:, j]
        d_g[arm] = qr / mechanism[arm] * (indicator - mechanism[arm])
        # The outcome residual is at the arm this row took, so the indicator already puts it
        # at `arm`; `keep` is the missing-outcome mask every residual here carries.
        d_q[arm] = indicator * keep * ratio[:, j] * (y - targeted.observed)
        clip_bias[arm] = qr / mechanism[arm] * (untruncated[arm] - mechanism[arm])
    return CorrectionParts(d_g, d_q, clip_bias, clipped, tuple(guard))


def missing_outcome_correction_parts(
    outcome: FloatArray,
    targeted: InitialFit,
    treatment: FloatArray,
    observed: BoolArray,
    reduced: Any,
    propensity: FloatArray,
    missingness: FloatArray,
    *,
    g_bounds: tuple[float, float],
    missingness_bound: float,
    guard: tuple[str, ...],
) -> CorrectionParts:
    r"""The separate treatment, observation and outcome corrections in the paper."""
    y = np.asarray(outcome, dtype=float).reshape(-1)
    a = np.asarray(treatment, dtype=float).reshape(-1)
    delta = np.asarray(observed, dtype=float).reshape(-1)
    raw_g = np.asarray(propensity, dtype=float).reshape(-1)
    bounded_upper = bound(raw_g, float(g_bounds[0]), float(g_bounds[1]))
    g_a = np.column_stack([1.0 - bounded_upper, bounded_upper])
    raw_m = np.asarray(missingness, dtype=float)
    g_m = bound(raw_m, float(missingness_bound), 1.0)
    gamma_a = reduced.bounded_gamma_a(g_bounds)
    gamma_m = reduced.bounded_gamma_m(missingness_bound)
    w2 = np.asarray(reduced.r_a, dtype=float) / (gamma_a * gamma_m)
    w2 += np.asarray(reduced.r_m, dtype=float) / gamma_m
    e = np.asarray(reduced.e, dtype=float)

    d_a: dict[float, FloatArray] = {}
    d_m: dict[float, FloatArray] = {}
    d_y: dict[float, FloatArray] = {}
    d_g: dict[float, FloatArray] = {}
    zeros: dict[float, FloatArray] = {}
    for j, arm in enumerate(reduced.arms):
        indicator = (a == float(arm)).astype(float)
        d_a[arm] = e[:, j] / g_a[:, j] * (indicator - g_a[:, j])
        d_m[arm] = indicator * e[:, j] / (g_a[:, j] * g_m[:, j]) * (delta - g_m[:, j])
        d_y[arm] = indicator * delta * w2[:, j] * (y - np.asarray(targeted.observed, dtype=float))
        d_g[arm] = np.asarray(d_a[arm] + d_m[arm], dtype=float)
        zeros[arm] = np.zeros_like(d_g[arm])
    clipped = np.asarray(
        (raw_g != bounded_upper) | np.any(raw_m != g_m, axis=1),
        dtype=bool,
    )
    return CorrectionParts(
        d_g=d_g,
        d_q=d_y,
        clip_bias=zeros,
        clipped=clipped,
        guard=tuple(guard),
        d_a=d_a,
        d_m=d_m,
        d_y=d_y,
    )


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
    _expect(submodel, "mtp")
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
    _expect(submodel, "regime")
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
    _expect(submodel, "ipsi")
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
    return scaler.unscale_levels(np.column_stack([targeted.arms[level] for level in levels]))


def msm_coefficients(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    design: FloatArray,
    model_weights: FloatArray,
    weights: FloatArray,
    scaler: OutcomeScaler,
    observed: BoolArray | None = None,
    link: str = "identity",
) -> dict[float, ArmMean]:
    r"""A working model's coefficients and their influence curves, keyed by coefficient.

    :math:`\hat\beta` solves the weighted least-squares equation
    :math:`U(\beta, \bar Q^*) = 0` of :func:`~cleverly.msm.solve_projection`, which for
    the identity link is the closed form
    :math:`M^{-1} P_n[\sum_a h\,\varphi\,\bar Q^*]`, and

    .. math::

        D_\beta^*(O) = M^{-1}\Big[
            H(A, W)\,\{Y - \bar Q^*(A, W)\}
            + \sum_a h(a, V)\,\frac{dm}{d\eta}\,\varphi(a, V)\,
              \big\{\bar Q^*(a, W) - m(a, V; \hat\beta)\big\}
        \Big]

    -- the standard M-estimation form, with :math:`M = -\partial U/\partial\beta` the
    matrix :func:`~cleverly.msm.solve_projection` returns.  :math:`M` carries no *further*
    term because :math:`U(\beta_0, \bar Q_0) = 0` at the truth, so the variation of
    :math:`M^{-1}` multiplies zero; that argument is untouched by the link, but what
    :math:`M` **contains** is not -- under a non-identity link it carries a curvature term
    that no saturated working model can exercise.  It carries no nuisance either, because
    :math:`h` and :math:`\varphi` are known functions; see :mod:`cleverly.msm` for the
    estimand and for why the weights have to be known for that to hold.

    Both halves are still zero in the sample after targeting, and still for two different
    reasons -- but under a link the first is zero only because the fluctuation was solved
    at *this* :math:`\hat\beta`, which is what
    :func:`~cleverly.estimators.targeting.solve_with_projection` alternates to achieve.

    The first term is zero because the ``msm`` fluctuation solved its own score equation.
    The second is zero *by construction*: :math:`\hat\beta` is the weighted least-squares
    solution against :math:`\bar Q^*`, so the residuals it leaves are orthogonal to
    :math:`h\,(dm/d\eta)\,\varphi`.  With the identity link the first needs no iteration
    either, since the covariate does not mention :math:`\beta` -- which is what makes that
    case a one-fluctuation TMLE.

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
    link:
        The working model's declared link, by name.  A *string* rather than a
        :class:`~cleverly.msm.Link`, for the same reason the arrays are passed plainly:
        this layer is written against what a saved result carries.

    Both arrays are passed plainly rather than as a
    :class:`~cleverly.msm.MSMSet`, on the same terms as ``regimes`` and ``shifts``: the
    inference layer is written against arrays so that it does not depend on the objects
    that produced them.
    """
    _expect(submodel, "msm")
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

    spec = link_for(str(link))
    fit = solve_projection(phi, h, predictions, w, str(link))
    beta = fit.beta
    fitted = np.asarray(spec.inverse(np.einsum("ijp,p->ij", phi, beta)), dtype=float)
    # (n, K, p) -- h * dm/dbeta, which is h * phi under the identity link and so leaves
    # that path on exactly the arithmetic it was on before links existed.
    weighted_design = phi * (h * np.asarray(spec.slope(fitted), dtype=float))[:, :, None]

    # (n, p): the plug-in half, sum_a h * (dm/dbeta) * (Qbar* - m(a, V; beta)).
    plugin = np.einsum("ijp,ij->ip", weighted_design, predictions - fitted)
    contribution = w[:, None] * (submodel.observed * residual[:, None] + plugin)
    influence = np.linalg.solve(fit.jacobian, contribution.T).T

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
    #: The doubly-robust correction, ``-(D*_Q + D*_g)``, for a fit that solved the extra
    #: score equations -- or the one term of it a single-guard fit solved for, since this
    #: is :meth:`CorrectionParts.total` negated and that selects.  **Not** the ``guard=``
    #: tuple, which shares only the name; zeros for every other fit, which is what keeps
    #: ``total`` and
    #: ``shares()`` reading exactly as they did before that variant existed.  It belongs to
    #: neither half above -- it is neither a positivity artefact nor outcome heterogeneity
    #: but the price of an interval that survives one bad nuisance -- and leaving it out
    #: would make this decomposition disagree with the curve by the whole of what the
    #: variant does.
    guard: FloatArray | None = None

    @property
    def total(self) -> FloatArray:
        summed = np.asarray(self.residual + self.plugin, dtype=float)
        if self.guard is None:
            return summed
        return np.asarray(summed + self.guard, dtype=float)

    def shares(self) -> dict[str, float]:
        """Each term's share of the influence curve's variance."""
        total = float(np.var(self.total))
        if total <= 0:
            out = {"residual": float("nan"), "plugin": float("nan")}
            return out if self.guard is None else {**out, "guard": float("nan")}
        shares = {
            "residual": float(np.var(self.residual)) / total,
            "plugin": float(np.var(self.plugin)) / total,
        }
        if self.guard is None:
            return shares
        return {**shares, "guard": float(np.var(self.guard)) / total}


def counterfactual_mean_parts(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
    corrections: Mapping[float, FloatArray] | None = None,
) -> dict[float, ICParts]:
    """The decomposed influence curves behind :func:`counterfactual_means`, per arm.

    ``parts.total`` agrees with the summed curve to floating-point rounding rather
    than bit-for-bit: the sum there is bracketed differently, and addition is not
    associative.  The gap is a few ULP and is asserted in
    ``tests/unit/test_ic_parts.py``; use :func:`counterfactual_means` for the
    estimate and this for the diagnostic.

    ``corrections`` is what :func:`counterfactual_means` subtracts, and must be passed
    whenever it was: this is meant to decompose *the* curve, and a decomposition missing a
    term the curve has is worse than no decomposition at all.
    """
    _expect(submodel, "mean")
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _residual(outcome, targeted, observed)
    return {
        arm: ICParts(
            w * submodel.column_for(arm) * residual,
            w * (targeted.arms[arm] - float(np.average(targeted.arms[arm], weights=w))),
            None if corrections is None else -w * np.asarray(corrections[arm], dtype=float),
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
    *,
    reference: float = 0.0,
) -> dict[float, ArmMean]:
    """The effect among each non-reference arm's own units, on the scaled outcome scale.

    One entry per non-reference arm, keyed by that arm -- a single entry, the classic
    ATT, when the treatment is binary.
    """
    _expect(submodel, "att")
    return _conditional_effects(
        outcome, targeted, submodel, treatment, weights, observed, reference, conditions_on_arm=True
    )


def atc_estimate(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    treatment: FloatArray,
    weights: FloatArray,
    observed: BoolArray | None = None,
    *,
    reference: float = 0.0,
) -> dict[float, ArmMean]:
    """The mirror image: each contrast among the *reference* arm's units."""
    _expect(submodel, "atc")
    return _conditional_effects(
        outcome,
        targeted,
        submodel,
        treatment,
        weights,
        observed,
        reference,
        conditions_on_arm=False,
    )


def _conditional_effects(
    outcome: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    treatment: FloatArray,
    weights: FloatArray,
    observed: BoolArray | None,
    reference: float,
    *,
    conditions_on_arm: bool,
) -> dict[float, ArmMean]:
    """``E[Y^a - Y^r | A = c]`` per non-reference arm, with ``c`` the group's population.

    ``conditions_on_arm`` picks which: the contrast arm itself for the ATT, the reference
    arm for the ATC, where every parameter conditions on the same population.

    The columns are read by :meth:`~cleverly.fluctuation.submodel.Submodel.contrast_column_for`
    rather than positionally, so an implementation that lined the arms up by order would
    fail on the three-armed law rather than pass on a coincidence.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _residual(outcome, targeted, observed)
    out: dict[float, ArmMean] = {}
    for arm in sorted(submodel.contrast_columns):
        conditioning = arm if conditions_on_arm else reference
        indicator = np.asarray(a == conditioning, dtype=float)
        share = float(np.average(indicator, weights=w))
        if share <= 0:
            raise ValueError(f"no observations in arm {conditioning:g}: the estimand is undefined")

        contrast = targeted.arms[arm] - targeted.arms[reference]
        psi = float(np.average(contrast, weights=w * indicator))
        column = submodel.contrast_column_for(arm)
        out[arm] = ArmMean(psi, w * (column * residual + (indicator / share) * (contrast - psi)))
    return out


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
