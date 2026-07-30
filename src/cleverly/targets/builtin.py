"""The estimands a classic point-treatment fit reports.

Each is a thin adapter over the influence-function builders in
:mod:`cleverly.inference.influence`, which are unchanged: this module says *which*
functional and *on what scale*, and the arithmetic stays where its derivation and
its Gateaux tests already are.

Registration order is report order.
"""

from __future__ import annotations

import numpy as np

from ..inference.delta import log_odds_ratio_influence, log_ratio_influence
from ..inference.influence import ParameterEstimate, atc_estimate, att_estimate
from .base import Identification, Target, TargetContext, parameter_name

__all__ = ["BUILTIN_TARGETS"]

_POSITIVITY = (
    "positivity: 0 < P(A = 1 | W) < 1 almost surely, so both counterfactual "
    "means are supported by data at every covariate value",
)
_POINT_TREATMENT = (
    "consistency: Y = Y^a when A = a",
    "no unmeasured confounding: Y^a is independent of A given W",
    *_POSITIVITY,
)

_MEAN_ID = Identification(
    assumptions=_POINT_TREATMENT,
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "consistent if either Qbar(A, W) or g(W) is consistent; with delta= the "
        "mechanism half becomes the product g * P(Delta = 1 | A, W), and with "
        "intermediate= the product g * P(Z = z | A, W) * P(Delta = 1 | A, W)"
    ),
    references=("van der Laan & Rubin (2006)", "Gruber & van der Laan (2010)"),
)

_CONDITIONAL_ID = Identification(
    assumptions=(
        *_POINT_TREATMENT,
        "the conditioning event (being treated, or being untreated) has positive "
        "probability, so the parameter is defined",
    ),
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "consistent if either Qbar(A, W) or g(W) is consistent; the influence curve "
        "carries an extra term for the randomness of the conditioning event"
    ),
    references=("van der Laan (2010)",),
)


def _ey(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y(a)]`` for every arm, always named with its label.

    The arm-general counterpart of ``ey1`` / ``ey0``, which name the two arms of a binary
    fit and cannot name a third.  Deliberately *not* collapsed to a bare stem on two
    arms: the names have to stay distinct from each other, and ``ey1`` / ``ey0`` are what
    a two-armed report uses anyway.
    """
    return [
        ctx.finish(
            parameter_name("ey", arm=ctx.label(arm)), mean.psi, mean.influence_curve, "level"
        )
        for arm, mean in sorted(ctx.means.items())
    ]


def _ey1(ctx: TargetContext) -> list[ParameterEstimate]:
    mean = ctx.means[1.0]
    return [ctx.finish("ey1", mean.psi, mean.influence_curve, "level")]


def _ey0(ctx: TargetContext) -> list[ParameterEstimate]:
    mean = ctx.means[0.0]
    return [ctx.finish("ey0", mean.psi, mean.influence_curve, "level")]


def _ate(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y(a)] - E[Y(ref)]``, once per non-reference arm."""
    reference = ctx.means[ctx.reference]
    return [
        ctx.finish(
            ctx.name_for("ate", arm, versus=ctx.reference),
            ctx.means[arm].psi - reference.psi,
            ctx.means[arm].influence_curve - reference.influence_curve,
            "difference",
        )
        for arm in ctx.contrast_arms
    ]


def _rr(ctx: TargetContext) -> list[ParameterEstimate]:
    reference = ctx.means[ctx.reference]
    out: list[ParameterEstimate] = []
    for arm in ctx.contrast_arms:
        mean = ctx.means[arm]
        log_psi, ic = log_ratio_influence(
            mean.psi, mean.influence_curve, reference.psi, reference.influence_curve
        )
        out.append(
            ctx.finish(
                ctx.name_for("rr", arm, versus=ctx.reference),
                float(np.exp(log_psi)),
                ic,
                "ratio",
                log_psi=log_psi,
            )
        )
    return out


def _or(ctx: TargetContext) -> list[ParameterEstimate]:
    reference = ctx.means[ctx.reference]
    out: list[ParameterEstimate] = []
    for arm in ctx.contrast_arms:
        mean = ctx.means[arm]
        log_psi, ic = log_odds_ratio_influence(
            mean.psi, mean.influence_curve, reference.psi, reference.influence_curve
        )
        out.append(
            ctx.finish(
                ctx.name_for("or", arm, versus=ctx.reference),
                float(np.exp(log_psi)),
                ic,
                "ratio",
                log_psi=log_psi,
            )
        )
    return out


def _att(ctx: TargetContext) -> list[ParameterEstimate]:
    psi, ic = att_estimate(
        ctx.scaled, ctx.targeted, ctx.submodel, ctx.treatment, ctx.weights, ctx.observed
    )
    return [ctx.finish("att", psi, ic, "difference")]


def _atc(ctx: TargetContext) -> list[ParameterEstimate]:
    psi, ic = atc_estimate(
        ctx.scaled, ctx.targeted, ctx.submodel, ctx.treatment, ctx.weights, ctx.observed
    )
    return [ctx.finish("atc", psi, ic, "difference")]


#: In report order.
BUILTIN_TARGETS: tuple[Target, ...] = (
    Target(
        name="ate",
        group="mean",
        scale="difference",
        build=_ate,
        identification=_MEAN_ID,
        in_default_set=True,
        description="average treatment effect, E[Y^1] - E[Y^0]",
    ),
    Target(
        name="att",
        group="att",
        scale="difference",
        build=_att,
        identification=_CONDITIONAL_ID,
        requires_binary_treatment=True,
        in_default_set=True,
        undefined_when="the sample contains no treated units",
        description="average treatment effect on the treated, E[Y^1 - Y^0 | A = 1]",
    ),
    Target(
        name="atc",
        group="atc",
        scale="difference",
        build=_atc,
        identification=_CONDITIONAL_ID,
        requires_binary_treatment=True,
        in_default_set=True,
        undefined_when="the sample contains no untreated units",
        description="average treatment effect on the controls, E[Y^1 - Y^0 | A = 0]",
    ),
    Target(
        name="ey",
        group="mean",
        scale="level",
        build=_ey,
        identification=_MEAN_ID,
        in_default_set=True,
        default_arms="multi",
        description="counterfactual mean under each arm, E[Y^a]",
    ),
    Target(
        name="ey1",
        group="mean",
        scale="level",
        build=_ey1,
        identification=_MEAN_ID,
        requires_binary_treatment=True,
        in_default_set=True,
        description="counterfactual mean under treatment, E[Y^1]",
    ),
    Target(
        name="ey0",
        group="mean",
        scale="level",
        build=_ey0,
        identification=_MEAN_ID,
        requires_binary_treatment=True,
        in_default_set=True,
        description="counterfactual mean under control, E[Y^0]",
    ),
    Target(
        name="rr",
        group="mean",
        scale="ratio",
        build=_rr,
        identification=_MEAN_ID,
        requires_family="binomial",
        # Undefined at zero: the log ratio needs both risks strictly positive.
        parameter_bounds=(0.0, float("inf")),
        undefined_when="a counterfactual risk is zero, leaving the log ratio undefined",
        description="risk ratio, E[Y^1] / E[Y^0]",
    ),
    Target(
        name="or",
        group="mean",
        scale="ratio",
        build=_or,
        identification=_MEAN_ID,
        requires_family="binomial",
        # Both risks must lie strictly inside (0, 1) for the odds to be finite.
        parameter_bounds=(0.0, float("inf")),
        undefined_when="a counterfactual risk is 0 or 1, leaving the odds undefined",
        description="odds ratio of the counterfactual risks",
    ),
)
