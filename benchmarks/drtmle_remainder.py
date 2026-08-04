r"""Item 13: the remainder Theorem 1 assumes negligible, computed rather than asserted.

``docs/roadmap.md``'s **item 13** is the theorem's condition beyond the three score
equations.  Solving equations (8), (9) and (10) is *necessary*; Theorem 1 separately assumes
that what is left over is :math:`o_p(n^{-1/2})`, and nothing in this repository has ever
measured it.  ``docs/drtmle/validation-plan.md`` §5 states the quantity:

.. math::

    R_{\text{remaining}} = \hat\psi - \psi_0 - (P_n - P_0)\hat D_{\text{DR}}

and refuses :math:`P_n\hat D` as a stand-in for :math:`P_0\hat D` **by name** -- that is the
quantity targeting drove to zero, so it answers a different question.  :math:`P_0\hat D`
needs the corrected curve as a *function* of :math:`(W, A, Y)`, evaluated where the fit did
not look, and ``DRTMLE(evaluation=...)`` is what supplies it: every fold's nuisances at an
independent draw, moved by the same targeting steps the fitted arrays took.  This module is
the arithmetic on top.

**The fold convention, which §5 requires be documented rather than discovered.**  A
cross-fitted fit has no single nuisance function -- it has :math:`K` of them, one per outer
fold -- so :math:`P_0\hat D` is the fold-conditional average

.. math::

    P_0\hat D = \sum_k \frac{n_k}{n}\; E_0\bigl[\hat D^{(k)}(O)\bigr],

with :math:`n_k` the rows fold :math:`k` holds out and the expectation taken over the
independent draw.  It is the estimator's **own** fold weighting, not a uniform one, and
:class:`~cleverly.estimators._nuisance.CompanionEstimates` carries the counts for exactly
that reason.  Without this stated, ``R_remaining`` can be an artefact of how fold-specific
fits were extrapolated rather than a property of the estimator.

**Three columns, and they are not the same kind of number.**

``R_remaining``
    The whole of what item 13 asks about, exact given the companion: no limit is
    approximated anywhere in it.

``R2``
    The *plain* second-order remainder
    :math:`P_0[(\hat g_a - g_{0,a})/\hat g_a\,(\hat Q_a - \bar Q_{0,a})]` at the **fitted**
    nuisances, which is the regime-entry column Tier 2 gets in place of Tier 1's quadrature
    over a prescribed sequence.  Also exact -- the DGP knows :math:`\bar Q_0` and
    :math:`g_0`, and the companion knows :math:`\hat Q` and :math:`\hat g` at the same rows.

``R_Q`` and ``R_g``
    The two appendix branches, **approximated**, and what is approximated in them is stated
    below rather than buried.  §5 asks for them apart because a total trending to zero can
    conceal cancellation between them, which is gate 1's clause 4.

**What the branch columns are and are not.**  The 2016 working paper's appendix A gives
:math:`R_{Q,n} = R_{3,n} + R_{4,n} + M_{1,n}` and appendix B gives
:math:`R_{g,n} = \tilde R_{5,n} + \tilde R_{6,n} + \tilde M_{2,n}`
(``docs/drtmle/theorem-concordance.md`` §5).  Two observations make the second-order halves
computable and one keeps the rest honest.

*The sums need fewer limits than the terms do.*  Writing them out,

.. math::

    R_{3,n} + R_{4,n} &= P_0\Bigl[\bigl\{\bar Q_{0n,r}/g_0 - \bar Q_{n,r}/g_n\bigr\}
                                  (g_0 - g_n)\Bigr] \\
    \tilde R_{5,n} + \tilde R_{6,n} &= P_0\Bigl[\bigl\{(1_a/g_{1,0n,r})g_{2,0n,r}
                          - (1_a/g_{1,n,r})g_{2,n,r}\bigr\}(Y - \bar Q_n)\Bigr]

-- the univariate limits :math:`\bar Q_{0,r}`, :math:`g_{1,0,r}` and :math:`g_{2,0,r}`
**cancel** out of both.  What is left is the fitted reductions, which the companion has
exactly, and the ``0n`` limits.

*A ``0n`` limit is a quadrature and not a fit.*  :math:`\bar Q_{0n,r}` is the *population*
conditional mean of a computable quantity given two computable scalars, so it is estimated
here by a binned conditional average over the evaluation draw -- accuracy controlled by the
draw size and the bin count rather than by a model choice.  Both are reported: every branch
is recomputed at a second bin count and the difference travels beside it as the column's own
error.  Where that error exceeds the branch, the branch is reported as **not resolvable at
this DGP**, which is §5's *"where the DGP permits"* said out loud.

*The empirical-process terms are refused by name.*  :math:`M_{1,n}` and
:math:`\tilde M_{2,n}` are :math:`(P_n - P_0)` of a difference of estimated curves, and
under the fold convention above :math:`P_n` and :math:`P_0` are taken at *different*
renderings of the nuisances -- out of fold on the fitting sample, fold-conditional on the
evaluation draw.  There is no single-sample expression that is both, so rather than pick one
and call it the theorem's term, this module reports the second-order halves and says so.
They are the halves clause 4 is about: an empirical-process term is
:math:`o_p(n^{-1/2})` under the Donsker and :math:`L_2` conditions §5 lists and carries no
product of nuisance errors to cancel against.

**Nothing here asserts.**  It is an instrument, like ``benchmarks/bench_drtmle.py``: it
returns numbers a table prints and a human reads against the rules frozen in §5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cleverly.datasets import DGP
from cleverly.inference.influence import reduced_correction_parts

__all__ = [
    "BIN_COUNTS",
    "RemainderRow",
    "branch_products",
    "conditional_mean",
    "corrected_remainder",
    "evaluation_frame",
    "plain_remainder",
    "remainder_rows",
]

#: The two bin counts every branch is computed at.  Not a tuning knob: the pair *is* the
#: reported error of the binned limits, so a branch smaller than the gap between them is a
#: branch this design cannot resolve.  A power of two apart, so the coarse grid is a strict
#: coarsening of the fine one and the difference is discretisation rather than reshuffling.
BIN_COUNTS = (12, 24)

#: The arm each per-arm column is reported at, and the contrast taken between them.
ARMS = (1.0, 0.0)


@dataclass(frozen=True)
class RemainderRow:
    """One estimand's remainder columns at one fit.

    Attributes
    ----------
    estimand:
        ``"ey1"``, ``"ey0"`` or ``"ate"``.
    psi, truth:
        The fit's estimate and the law's value, both on the outcome's own scale.
    p0_curve, pn_curve:
        :math:`P_0\\hat D` under the fold convention, and :math:`P_n\\hat D` off the
        reported curve.  The second is what targeting drove to zero and is here so that a
        reader can see it did -- **not** so that it can stand in for the first.
    remaining:
        :math:`\\hat\\psi - \\psi_0 - (P_n - P_0)\\hat D`.
    root_n_remaining:
        :math:`\\sqrt n` times it, which is the quantity item 13 asks to vanish.
    r2:
        The plain second-order remainder at the fitted nuisances, at the **initial**
        regression -- the plug-in one, which says the nuisances are what the design says.
    r2_targeted:
        The same expression at the **targeted** regression, which is what the fit's bias is
        and what §5's targeted-coefficient clause requires the regime be read off.  The pair
        is the point: C3a's pilot had only the first and read it as the second.
    branch_q, branch_g:
        Appendix A's and appendix B's second-order halves, or ``nan`` where the binned
        limits did not resolve them.
    branch_error:
        The larger of the two branches' bin-count sensitivities, which is what "did not
        resolve" is decided on.
    """

    estimand: str
    psi: float
    truth: float
    p0_curve: float
    pn_curve: float
    remaining: float
    root_n_remaining: float
    r2: float
    r2_targeted: float
    branch_q: float
    branch_g: float
    branch_error: float


def evaluation_frame(dgp: DGP, n: int, seed: int) -> Any:
    """An independent draw from the same law, for the companion to be evaluated at.

    Drawn from a seed stream disjoint from the study's, so that raising ``--evaluation-n``
    cannot change which rows a replicate was *fitted* on -- the same prefix-stability rule
    ``benchmarks/drtmle_coverage.py`` applies to its own two streams.
    """
    return dgp.sample(n, seed=seed)[0]


def conditional_mean(
    target: np.ndarray,
    *designs: np.ndarray,
    mask: np.ndarray | None = None,
    bins: int = BIN_COUNTS[0],
) -> np.ndarray:
    r"""``E_0[target | designs]`` on the draw itself, by an equal-count binned average.

    The reduced regressions' **limits** are population conditional expectations given one or
    two scalars that are themselves computable functions of :math:`W`, so estimating them is
    a quadrature over the evaluation draw rather than a second modelling choice.  Equal-count
    bins rather than equal-width ones because a fitted mechanism piles up: an equal-width
    grid puts most of the draw in two cells and leaves the tails at one row each.

    ``mask`` restricts which rows the average is taken over -- the ``| A = a`` of
    :math:`\bar Q_{0,r}`'s definition -- while every row still receives a value, since the
    branch integrals are taken over the whole draw.  A cell with no eligible row falls back
    to the masked mean, which is the coarsest conditioning available rather than a ``nan``
    that would propagate through an integral silently.
    """
    values = np.asarray(target, dtype=float).reshape(-1)
    eligible = np.ones(values.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    codes = np.zeros(values.size, dtype=np.int64)
    width = 1
    for design in designs:
        column = np.asarray(design, dtype=float).reshape(-1)
        edges = np.quantile(column, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        codes = codes * bins + np.searchsorted(edges, column, side="right")
        width *= bins
    totals = np.bincount(codes[eligible], weights=values[eligible], minlength=width)
    counts = np.bincount(codes[eligible], minlength=width)
    fallback = float(values[eligible].mean()) if eligible.any() else 0.0
    cells = np.where(counts > 0, totals / np.maximum(counts, 1), fallback)
    return cells[codes]


def _fold_average(per_fold: list[float], weights: np.ndarray) -> float:
    """Section 5's fold-conditional average, at the estimator's own fold weights."""
    return float(np.dot(np.asarray(per_fold, dtype=float), weights))


def plain_remainder(result: Any, dgp: DGP, bounds: tuple[float, float]) -> dict[str, float]:
    r"""``R_2`` per arm at the **fitted** nuisances, fold-weighted over the companion draw.

    .. math::

        R_{2,a} = P_0\Bigl[\frac{\hat g_a - g_{0,a}}{\hat g_a}
                           \bigl(\hat Q(a, W) - \bar Q_0(a, W)\bigr)\Bigr]

    at the **initial** regression and mechanism, which is what the plug-in remainder is a
    remainder of.  The mechanism is the **truncated** one, because that is what the clever
    covariate divides by and so what the expansion is taken at; on a cell whose bound never
    binds the two are the same array.

    Tier 1 gets this exactly by quadrature over a prescribed sequence
    (``benchmarks/drtmle_injection.exact_remainder``); Tier 2 cannot, because its nuisances
    are fitted -- so this is the same quantity read off the companion instead, and the two
    are deliberately different code for the same definition at different tiers.
    """
    companion = result.nuisance.companion
    if companion is None:
        raise ValueError("R_2 at fitted nuisances needs a companion; fit with evaluation=")
    scaler = result.nuisance.scaler
    latent = _latent(companion.data, dgp)
    weights = companion.fold_weights

    out: dict[str, float] = {}
    for arm in ARMS:
        per_fold = []
        truth_q = np.asarray(dgp.outcome_mean(latent, arm, None), dtype=float)
        truth_g = _arm_probability(np.asarray(dgp.propensity(latent), dtype=float), arm)
        for fold in range(companion.n_folds):
            estimated_g = companion.propensity[fold].bounded(bounds)[
                :, companion.propensity[fold].column_for(arm)
            ]
            estimated_q = scaler.unscale_levels(companion.outcome[fold].arms[arm])
            per_fold.append(
                float(np.mean((estimated_g - truth_g) / estimated_g * (estimated_q - truth_q)))
            )
        out[f"r2_{int(arm)}"] = _fold_average(per_fold, weights)
    out["r2_ate"] = out["r2_1"] - out["r2_0"]
    return out


def targeted_remainder(result: Any, dgp: DGP, bounds: tuple[float, float]) -> dict[str, float]:
    r"""``R_2`` per arm at the **targeted** regression: what the fit's bias actually is.

    .. math::

        R_{2,a}(\bar Q^*) = P_0\Bigl[\frac{\hat g_a - g_{0,a}}{\hat g_a}
                                     \bigl(\bar Q^*(a, W) - \bar Q_0(a, W)\bigr)\Bigr]

    -- :func:`plain_remainder`'s expression at :math:`\bar Q^*` in place of :math:`\hat Q`,
    which is the pair ``docs/drtmle/validation-plan.md`` §5's targeted-coefficient clause
    requires be reported together.  C3a's pilot had only the first and read it as the second.

    **Taken over the fit's own rows rather than over the companion**, and that departure from
    every other column in this module is deliberate.  The companion holds each fold's
    *initial* arrays; the targeted ones live on ``ReductionFluctuation.evaluation``, which
    exists only on the ``DRTMLE`` path -- and the quantity wanted here is the **plain
    ``TMLE``'s** bias, since that is the estimator whose interval a shortfall is claimed
    against.  So :math:`P_0` is approximated by the sample mean over the fitting rows, which
    carries an :math:`O(n^{-1/2})` quadrature error that averages down over the study's
    replicates rather than biasing any of them.  ``benchmarks/drtmle_tier1_bias.py`` takes it
    the same way, and at Tier 1 the two can be checked against a quadrature that does not
    (``drtmle_injection.exact_targeted_remainder``).

    :math:`\bar Q_0` and :math:`g_0` come from the law, so this needs no companion at all and
    is available on a fit that declared no ``evaluation=``.
    """
    fluctuation = result.repeats[0].fluctuations["mean"]
    scaler = result.nuisance.scaler
    latent = _latent(result.data, dgp)
    bounded = result.nuisance.propensity.bounded(bounds)

    out: dict[str, float] = {}
    for arm in ARMS:
        estimated_g = bounded[:, result.nuisance.propensity.column_for(arm)]
        truth_g = _arm_probability(np.asarray(dgp.propensity(latent), dtype=float), arm)
        targeted_q = scaler.unscale_levels(fluctuation.targeted.arms[arm])
        truth_q = np.asarray(dgp.outcome_mean(latent, arm, None), dtype=float)
        out[f"r2_{int(arm)}"] = float(
            np.mean((estimated_g - truth_g) / estimated_g * (targeted_q - truth_q))
        )
    out["r2_ate"] = out["r2_1"] - out["r2_0"]
    return out


def corrected_remainder(result: Any, dgp: DGP) -> dict[str, float]:
    r""":math:`P_0\hat D` per estimand, fold-weighted, on the outcome's own scale.

    The curve is built by :func:`~cleverly.inference.influence.reduced_correction_parts` and
    the same three-term expression :func:`~cleverly.inference.influence.counterfactual_means`
    uses -- **the same functions the reported curve comes through**, at the companion's
    arrays instead of the fit's.  A second copy of that expression written for this module
    is exactly how a remainder comes to describe a curve nobody reported, which is the class
    of defect ``docs/roadmap.md``'s item 20 was.

    The centring is the *fit's* :math:`\hat\psi`, not the companion's own mean: what is
    wanted is :math:`E_0[\hat D]` at the estimator that was reported, and re-centring at the
    evaluation draw would drive it to zero by construction.
    """
    record = result.repeats[0].fluctuations["mean"].reduction
    evaluation = None if record is None else record.evaluation
    if evaluation is None:
        raise ValueError("P_0 D-hat needs a companion; fit with evaluation=")

    scaler = result.nuisance.scaler
    data = evaluation.data
    scaled = scaler.scale(data.outcome)
    weights = evaluation.fold_weights
    bounds = record.bounds
    guard = tuple(record.guard)

    per_arm: dict[float, list[float]] = {arm: [] for arm in ARMS}
    for fold in range(evaluation.n_folds):
        mechanism = evaluation.propensity[fold].arm(ARMS[0])
        targeted = evaluation.outcome[fold]
        corrections = reduced_correction_parts(
            scaled,
            targeted,
            data.treatment,
            evaluation.reduced[fold],
            mechanism,
            bounds=bounds,
            observed=data.observed,
            guard=guard,
        ).total()
        truncated = evaluation.propensity[fold].bounded(bounds)
        residual = scaled - targeted.observed
        for arm in ARMS:
            column = evaluation.propensity[fold].column_for(arm)
            indicator = (np.asarray(data.treatment, dtype=float) == arm).astype(float)
            covariate = indicator / truncated[:, column]
            psi_scaled = (result.estimates[_name(arm)].psi - scaler.lower) / scaler.range
            per_arm[arm].append(
                float(
                    np.mean(
                        covariate * residual
                        + targeted.arms[arm]
                        - psi_scaled
                        - np.asarray(corrections[arm], dtype=float)
                    )
                )
            )

    means = {arm: _fold_average(values, weights) for arm, values in per_arm.items()}
    return {
        "ey1": float(scaler.unscale_difference(means[1.0])),
        "ey0": float(scaler.unscale_difference(means[0.0])),
        "ate": float(scaler.unscale_difference(means[1.0] - means[0.0])),
    }


def branch_products(result: Any, dgp: DGP, *, bins: int) -> dict[str, float]:
    r"""Appendix A's and appendix B's **second-order halves**, at one bin count.

    ``R_3 + R_4`` and ``R̃_5 + R̃_6`` of
    ``docs/drtmle/theorem-concordance.md`` §5, with the univariate limits cancelled out and
    the two ``0n`` limits estimated by :func:`conditional_mean` on the evaluation draw.  The
    ``M`` terms are not here and are refused in this module's docstring rather than
    approximated.

    Per arm and summed into the ATE's contrast, on the outcome's own scale.  The reductions
    are read at each fold's own slab and averaged with the fold weights, exactly as
    :func:`corrected_remainder` is: the branch is a property of the same fold-conditional
    estimator the curve is.
    """
    record = result.repeats[0].fluctuations["mean"].reduction
    evaluation = None if record is None else record.evaluation
    if evaluation is None:
        raise ValueError("the appendix branches need a companion; fit with evaluation=")

    scaler = result.nuisance.scaler
    data = evaluation.data
    scaled = scaler.scale(data.outcome)
    treatment = np.asarray(data.treatment, dtype=float)
    latent = _latent(data, dgp)
    weights = evaluation.fold_weights
    bounds = record.bounds

    truth_g_one = np.asarray(dgp.propensity(latent), dtype=float)
    per_arm: dict[str, dict[float, list[float]]] = {"q": {}, "g": {}}
    for arm in ARMS:
        indicator = (treatment == arm).astype(float)
        truth_g = _arm_probability(truth_g_one, arm)
        truth_q = scaler.scale(np.asarray(dgp.outcome_mean(latent, arm, None), dtype=float))
        branch_q, branch_g = [], []
        for fold in range(evaluation.n_folds):
            column = evaluation.propensity[fold].column_for(arm)
            estimated_g = evaluation.propensity[fold].bounded(bounds)[:, column]
            estimated_q = evaluation.outcome[fold].arms[arm]
            reduced = evaluation.reduced[fold]
            qr = reduced.qr[:, reduced.column_for(arm)]
            gr1 = reduced.bounded_gr1(bounds)[:, reduced.column_for(arm)]
            gr2 = reduced.gr2[:, reduced.column_for(arm)]

            # Appendix A. `Q_{0n,r}` conditions on the estimated mechanism *and* the true
            # one, which is what makes R_3 an approximation error rather than a fitted one;
            # `Q_{n,r}` is the fitted reduction, which the companion holds exactly.
            qr_limit = conditional_mean(
                scaled - estimated_q, estimated_g, truth_g, mask=indicator == 1.0, bins=bins
            )
            branch_q.append(
                float(np.mean((qr_limit / truth_g - qr / estimated_g) * (truth_g - estimated_g)))
            )

            # Appendix B. Both reduced mechanisms' `0n` limits condition on the estimated
            # outcome regression and the true one, for the same reason and the other way up.
            gr1_limit = conditional_mean(indicator, estimated_q, truth_q, bins=bins)
            gr2_limit = conditional_mean(
                (indicator - estimated_g) / estimated_g, estimated_q, truth_q, bins=bins
            )
            floor, ceiling = bounds
            gr1_limit = np.clip(gr1_limit, floor, ceiling)
            branch_g.append(
                float(
                    np.mean(
                        (indicator * gr2_limit / gr1_limit - indicator * gr2 / gr1)
                        * (scaled - estimated_q)
                    )
                )
            )
        per_arm["q"][arm] = branch_q
        per_arm["g"][arm] = branch_g

    out: dict[str, float] = {}
    for key in ("q", "g"):
        one = _fold_average(per_arm[key][1.0], weights)
        zero = _fold_average(per_arm[key][0.0], weights)
        out[f"branch_{key}_ey1"] = float(scaler.unscale_difference(one))
        out[f"branch_{key}_ey0"] = float(scaler.unscale_difference(zero))
        out[f"branch_{key}_ate"] = float(scaler.unscale_difference(one - zero))
    return out


def remainder_rows(
    result: Any, dgp: DGP, *, n: int, bounds: tuple[float, float]
) -> list[RemainderRow]:
    """Every remainder column for one fit, one row per estimand.

    ``n`` is the **fitting** sample size, which is what the root-``n`` scaling is in: the
    evaluation draw is a quadrature rule and its size is an accuracy knob, not a sample size
    the estimator's rate is stated in.  Reading it off the companion is the mistake this
    argument exists to prevent.
    """
    truth = dgp.truth()
    p0 = corrected_remainder(result, dgp)
    r2 = plain_remainder(result, dgp, bounds)
    targeted = targeted_remainder(result, dgp, bounds)
    coarse = branch_products(result, dgp, bins=BIN_COUNTS[0])
    fine = branch_products(result, dgp, bins=BIN_COUNTS[1])

    rows = []
    for name in ("ate", "ey1", "ey0"):
        estimate = result.estimates[name]
        pn = float(np.mean(estimate.influence_curve))
        remaining = float(estimate.psi) - truth[name] - (pn - p0[name])
        error = max(
            abs(fine[f"branch_q_{name}"] - coarse[f"branch_q_{name}"]),
            abs(fine[f"branch_g_{name}"] - coarse[f"branch_g_{name}"]),
        )
        resolved = error <= max(abs(fine[f"branch_q_{name}"]), abs(fine[f"branch_g_{name}"]))
        rows.append(
            RemainderRow(
                estimand=name,
                psi=float(estimate.psi),
                truth=truth[name],
                p0_curve=p0[name],
                pn_curve=pn,
                remaining=remaining,
                root_n_remaining=float(np.sqrt(n)) * remaining,
                r2=r2[_KEYS[name]],
                r2_targeted=targeted[_KEYS[name]],
                branch_q=fine[f"branch_q_{name}"] if resolved else float("nan"),
                branch_g=fine[f"branch_g_{name}"] if resolved else float("nan"),
                branch_error=error,
            )
        )
    return rows


#: Which remainder key an estimand reads.  One mapping rather than one per call site, since
#: the two remainder columns are indexed the same way and a slip between them would put a
#: contrast's number under an arm's.
_KEYS = {"ate": "r2_ate", "ey1": "r2_1", "ey0": "r2_0"}


def _name(arm: float) -> str:
    return "ey1" if arm == 1.0 else "ey0"


def _arm_probability(one: np.ndarray, arm: float) -> np.ndarray:
    """``P(A = arm | W)`` from the arm-1 column, by complement -- the binary path's rule."""
    return one if arm == 1.0 else 1.0 - one


def _latent(data: Any, dgp: DGP) -> np.ndarray:
    """The latent matrix ``dgp.propensity`` and ``dgp.outcome_mean`` are defined on.

    The observed covariates **are** the first columns of it, and a process with hidden
    columns has no way to hand them back -- so such a law is refused here rather than
    silently evaluated at zeros, which would return a plausible truth for a different
    process.  Every cell of the coverage study is drawn from a fully observed law.
    """
    covariates = np.asarray(data.covariates, dtype=float)
    if covariates.shape[1] != dgp.n_latent:
        raise ValueError(
            f"{dgp.name} has {dgp.n_latent} latent variable(s) and the draw carries "
            f"{covariates.shape[1]} covariate(s); the remainder columns evaluate the law's "
            "own nuisances at the evaluation rows, which a process with hidden variables "
            "cannot supply"
        )
    return covariates
