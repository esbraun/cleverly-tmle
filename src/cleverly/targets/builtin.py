"""The seven estimands a classic point-treatment fit reports.

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
from .base import Identification, Target, TargetContext

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


def _ey1(ctx: TargetContext) -> ParameterEstimate:
    psi_one, ic_one, _, _ = ctx.means
    return ctx.finish("ey1", psi_one, ic_one, "level")


def _ey0(ctx: TargetContext) -> ParameterEstimate:
    _, _, psi_zero, ic_zero = ctx.means
    return ctx.finish("ey0", psi_zero, ic_zero, "level")


def _ate(ctx: TargetContext) -> ParameterEstimate:
    psi_one, ic_one, psi_zero, ic_zero = ctx.means
    return ctx.finish("ate", psi_one - psi_zero, ic_one - ic_zero, "difference")


def _rr(ctx: TargetContext) -> ParameterEstimate:
    psi_one, ic_one, psi_zero, ic_zero = ctx.means
    log_psi, ic = log_ratio_influence(psi_one, ic_one, psi_zero, ic_zero)
    return ctx.finish("rr", float(np.exp(log_psi)), ic, "ratio", log_psi=log_psi)


def _or(ctx: TargetContext) -> ParameterEstimate:
    psi_one, ic_one, psi_zero, ic_zero = ctx.means
    log_psi, ic = log_odds_ratio_influence(psi_one, ic_one, psi_zero, ic_zero)
    return ctx.finish("or", float(np.exp(log_psi)), ic, "ratio", log_psi=log_psi)


def _att(ctx: TargetContext) -> ParameterEstimate:
    psi, ic = att_estimate(
        ctx.scaled, ctx.targeted, ctx.submodel, ctx.treatment, ctx.weights, ctx.observed
    )
    return ctx.finish("att", psi, ic, "difference")


def _atc(ctx: TargetContext) -> ParameterEstimate:
    psi, ic = atc_estimate(
        ctx.scaled, ctx.targeted, ctx.submodel, ctx.treatment, ctx.weights, ctx.observed
    )
    return ctx.finish("atc", psi, ic, "difference")


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
        in_default_set=True,
        undefined_when="the sample contains no untreated units",
        description="average treatment effect on the controls, E[Y^1 - Y^0 | A = 0]",
    ),
    Target(
        name="ey1",
        group="mean",
        scale="level",
        build=_ey1,
        identification=_MEAN_ID,
        in_default_set=True,
        description="counterfactual mean under treatment, E[Y^1]",
    ),
    Target(
        name="ey0",
        group="mean",
        scale="level",
        build=_ey0,
        identification=_MEAN_ID,
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
