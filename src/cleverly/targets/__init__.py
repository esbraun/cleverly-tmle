"""The registry of target parameters.

Adding an estimand used to mean editing seven places: four module-level constants,
the estimand resolver, the group dispatcher, a clever-covariate builder, an
``if/elif`` chain in the estimator, and the ratio helper's ``which`` argument.  Now
it means constructing one :class:`~cleverly.targets.base.Target` and calling
:func:`register`.

The registry is *not* an invitation to add estimands casually.  Registering one
without a matching branch in the ``functional`` of one of the oracle laws
(``tests/discrete_law.py`` for the arm- and regime-indexed estimands,
``tests/discrete_law_shift.py`` for the shift-indexed ones) is a test failure
(``tests/unit/test_registry.py``): the package's evidence that an influence curve
is right is that it matches a numerically differentiated one on an exactly
representable law, and an estimand with no oracle has no such evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._typing import ParameterAxis
from ..fluctuation.submodel import SUBMODEL_BUILDERS, TargetGroup
from .base import Identification, Target, TargetContext, parameter_name, parameter_stem
from .builtin import BUILTIN_TARGETS

__all__ = [
    "TARGETS",
    "Identification",
    "Target",
    "TargetContext",
    "all_names",
    "default_names",
    "groups_for",
    "parameter_name",
    "parameter_stem",
    "register",
    "resolve_estimands",
    "targets_for",
]

#: Registered targets, in report order.  Insertion order is report order.
TARGETS: dict[str, Target] = {}

#: How an outcome family is described to a user, who passed in a binary column rather
#: than choosing a likelihood.
_FAMILY_WORDS = {"binomial": "binary", "gaussian": "continuous"}


def register(target: Target, *, replace: bool = False) -> Target:
    """Add a target to the registry.

    Raises unless ``replace=True`` when the name is taken, because silently
    shadowing a built-in estimand would change what every existing script reports.

    Also refuses a target whose ``group`` names no registered fluctuation.  That used to
    be a static check -- ``TargetGroup`` was a ``Literal`` of the three built-ins -- but a
    registry the caller can extend cannot have an exhaustive one, so the check happens
    here instead.  Registration is the right moment: the alternative is discovering it
    when the fit tries to build a clever covariate, several steps further on.
    """
    if target.name in TARGETS and not replace:
        raise ValueError(
            f"a target named {target.name!r} is already registered; pass replace=True "
            "to override it deliberately"
        )
    if target.group not in SUBMODEL_BUILDERS:
        raise ValueError(
            f"target {target.name!r} declares group {target.group!r}, for which there is "
            f"no submodel builder; registered groups are {sorted(SUBMODEL_BUILDERS)}. "
            "Register the fluctuation with cleverly.fluctuation.register_submodel first -- "
            "a group is a score equation, and a target cannot be solved without one."
        )
    TARGETS[target.name] = target
    return target


for _target in BUILTIN_TARGETS:
    register(_target)


def all_names(
    family: str | None = None, n_arms: int = 2, *, axis: ParameterAxis = "arm"
) -> tuple[str, ...]:
    """Every registered target the family, arm count and parameter axis support."""
    return tuple(
        name
        for name, target in TARGETS.items()
        if (family is None or target.supported_by(family))
        and target.supports_arms(n_arms)
        and target.matches_axis(axis)
    )


def default_names(family: str, n_arms: int = 2, *, axis: ParameterAxis = "arm") -> tuple[str, ...]:
    """The default report: the default set, plus whatever else the family supports.

    For a binary outcome that adds the risk ratio and odds ratio, which are only
    defined when the counterfactual means are probabilities.

    ``n_arms`` decides what belongs in the report at that arm count.  ``ey1`` and ``ey0``
    name two arms and cannot be reported at three; ``ey`` -- one mean per arm -- joins the
    default there instead, since there is no ``ey1`` to carry the means.  The ATT and ATC
    are *defined* at every arm count but stay out of a multi-arm default, which
    :attr:`~cleverly.targets.Target.default_arms` explains; ``estimands=`` is how to ask.

    ``axis`` switches the report between the arm-indexed estimands, the regime-indexed
    ones and the shift-indexed ones.  It is a switch rather than a widening because
    declaring ``interventions=`` or ``shifts=`` says what the fit's counterfactuals
    *are*; see :attr:`~cleverly.targets.Target.parameter_axis`.
    """
    return tuple(
        name
        for name, target in TARGETS.items()
        if target.supported_by(family)
        and target.supports_arms(n_arms)
        and target.matches_axis(axis)
        and not (target.default_arms == "multi" and n_arms == 2)
        and not (target.default_arms == "binary" and n_arms != 2)
        and (target.in_default_set or family == "binomial")
    )


def groups_for(estimands: Sequence[str]) -> list[TargetGroup]:
    """Which fluctuations must be fit to cover the requested estimands.

    Each group has its own efficient influence function and so its own score
    equation to solve; several estimands can share one.  Ordered by first
    appearance in the registry so the targeting steps run in report order.
    """
    wanted = {TARGETS[name].group for name in estimands}
    ordered: list[TargetGroup] = []
    for target in TARGETS.values():
        if target.group in wanted and target.group not in ordered:
            ordered.append(target.group)
    return ordered


def targets_for(group: TargetGroup, estimands: Sequence[str]) -> tuple[Target, ...]:
    """The requested targets that this fluctuation produces, in report order."""
    requested = set(estimands)
    return tuple(
        target for name, target in TARGETS.items() if target.group == group and name in requested
    )


#: How each parameter axis is *declared*, for the message a caller who crossed two of
#: them will read.  Keyed by axis so the sentence names the keyword rather than the
#: internal word.
_AXIS_DECLARED_BY = {
    "arm": "a fit without interventions=, shifts= or msm=",
    "regime": "a fit that declares interventions=",
    "shift": "a fit that declares shifts=",
    "ipsi": "a fit that declares incremental=",
    "msm": "a fit that declares msm=",
}

#: What each axis indexes its parameters by, for the same message.
_AXIS_INDEXES_BY = {
    "arm": "treatment arm",
    "regime": "declared regime",
    "shift": "declared shift",
    "ipsi": "declared tilt of the treatment mechanism",
    "msm": "working-model coefficient",
}


def resolve_estimands(
    requested: Sequence[str] | str | None,
    family: str,
    n_arms: int = 2,
    *,
    axis: ParameterAxis = "arm",
) -> tuple[str, ...]:
    """Normalise and validate a requested estimand list.

    ``None`` gives the default report for the outcome family and arm count, and ``"all"``
    gives everything they support.  An estimand the family or the arm count cannot support
    is an error rather than a silent drop: asking for a risk ratio of two means that may
    be negative is a mistake worth surfacing, not a preference to be honoured quietly,
    and the same goes for asking a three-armed fit for "the effect on the treated".

    ``axis`` says what the fit's parameters are indexed by.  It selects between the
    arm-, regime-, shift- and coefficient-indexed estimands rather than widening the
    choice; asking across two of them is refused, for the reason
    :attr:`~cleverly.targets.Target.parameter_axis` gives.
    """
    if requested is None:
        names: tuple[str, ...] = default_names(family, n_arms, axis=axis)
    elif isinstance(requested, str):
        names = all_names(family, n_arms, axis=axis) if requested == "all" else (requested,)
    else:
        names = tuple(requested)

    unknown = [name for name in names if name not in TARGETS]
    if unknown:
        raise ValueError(f"unknown estimand(s) {unknown}; choose from {list(TARGETS)}")

    mismatched = [name for name in names if not TARGETS[name].matches_axis(axis)]
    if mismatched:
        available = list(all_names(family, n_arms, axis=axis))
        other_axes = sorted({TARGETS[name].parameter_axis for name in mismatched})
        raise ValueError(
            f"estimand(s) {mismatched} do not belong to {_AXIS_DECLARED_BY[axis]}; they are "
            f"indexed by {' and '.join(_AXIS_INDEXES_BY[other] for other in other_axes)}, and this "
            f"fit's parameters are indexed by {_AXIS_INDEXES_BY[axis]}. Declaring "
            "interventions=, shifts= or incremental= says what the fit's counterfactuals "
            "are, and msm= "
            "says how they are summarised, so reporting across two of them from a single "
            "fluctuation would put two score equations under one heading. Available here: "
            f"{available}."
        )

    wrong_arms = [name for name in names if not TARGETS[name].supports_arms(n_arms)]
    if wrong_arms:
        raise ValueError(
            f"estimand(s) {wrong_arms} are defined for a binary treatment only, but this "
            f"fit has {n_arms} arms. They name one of exactly two arms -- or their "
            "intervention does, as an incremental tilt of the odds of treatment does -- "
            "so with more arms they name no single parameter. Available here: "
            f"{list(all_names(family, n_arms, axis=axis))}."
        )

    unsupported = [name for name in names if not TARGETS[name].supported_by(family)]
    if unsupported:
        needs = sorted({TARGETS[name].requires_family or "?" for name in unsupported})
        # Named the way a user thinks of it, with the internal family in parentheses:
        # "binomial" is what the config records, "binary" is what they passed in.
        wanted = " / ".join(_FAMILY_WORDS.get(name, name) for name in needs)
        raise ValueError(
            f"estimand(s) {unsupported} require a {wanted} outcome "
            f"(family={needs[0]!r}), but this fit has family={family!r}. The risk ratio "
            "and odds ratio are not defined for a continuous outcome. Drop them or "
            "dichotomise the outcome."
        )

    ordered = tuple(name for name in TARGETS if name in set(names))
    if not ordered:
        raise ValueError("no estimands requested")
    return ordered
