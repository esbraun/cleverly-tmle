"""Which arms a reported parameter name refers to.

Two sensitivity analyses -- the omitted-variable bound and the MNAR tilt -- are
functionals of *one* estimand rather than of the whole report, so each has to get from
the name a caller passed in (``"ate"``, or ``"ate[medium vs low]"``) back to the arms
that name is about.  With two arms the answer was a constant; with ``K`` it is a
statement about arm codes, and both modules would otherwise make it separately.

The map is **composed forward** through :func:`~cleverly.targets.parameter_name`, the
one place the naming convention lives, rather than parsed back out of the reported name
-- the rule :mod:`cleverly.longitudinal` follows for the same reason.  A label is the
user's own level, so ``"a vs b"`` is a perfectly legal label and a split on ``" vs "``
would file it under a contrast that does not exist instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..targets import parameter_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult
    from ..fluctuation.submodel import TargetGroup

__all__ = ["ArmParameter", "arm_parameters"]


@dataclass(frozen=True)
class ArmParameter:
    """One reported parameter, and the arms it is a functional of.

    Attributes
    ----------
    name:
        The reported name -- what indexes ``result.estimates``.
    stem:
        The target it came from, for messages.
    group:
        Which fluctuation carries its score equation, so a caller can reach the
        submodel the parameter was targeted with.
    arm:
        The arm the parameter is *about*: the one whose mean a level reports, and the
        non-reference arm of a contrast.
    versus:
        The arm a contrast is taken against, or ``None`` for a level.  For ``att`` and
        ``atc`` this is also the reference the propensity odds are formed against.
    """

    name: str
    stem: str
    group: TargetGroup
    arm: float
    versus: float | None = None

    @property
    def conditions_on(self) -> float | None:
        """The arm this parameter conditions on, or ``None`` when it conditions on none.

        The contrast arm for ``att`` and the reference for ``atc`` -- the same rule
        :func:`~cleverly.inference.influence._conditional_effects` applies, since these
        are the same parameters seen from the sensitivity side.
        """
        if self.group == "att":
            return self.arm
        if self.group == "atc":
            return self.versus
        return None


def arm_parameters(result: TMLEResult) -> dict[str, ArmParameter]:
    """Every arm-indexed *linear* parameter this fit could have reported, by name.

    Linear in the outcome regression, which is what the omitted-variable bound needs and
    what the MNAR tilt can re-mix: the counterfactual means, their contrasts against the
    reference, and the two conditional effects.  Ratios are excluded here rather than
    filtered by each caller -- ``rr`` and ``or`` are not linear functionals, and the
    reason is the same one in both modules.

    Names are generated for every arm whether or not the fit reported them; callers
    intersect with ``result.estimates``.  Returns an empty mapping for a fit with no
    arms at all, so a continuous dose is refused by whatever the caller says rather than
    by an index error here.
    """
    data = result.data
    if data.is_continuous_treatment:
        return {}
    if result.parameter_keys:
        from ..targets import TARGETS

        levels = tuple(data.treatment_levels)

        def code(value: object) -> float:
            try:
                return float(levels.index(value))
            except ValueError as error:
                raise ValueError(
                    f"structured parameter metadata names treatment level {value!r}, "
                    f"which is absent from the fitted levels {list(levels)}"
                ) from error

        structured: dict[str, ArmParameter] = {}
        for alias, key in result.parameter_keys.items():
            if key.axis != "arm" or key.estimand not in {"ey", "ey1", "ey0", "ate", "att", "atc"}:
                continue
            target = TARGETS[key.estimand]
            if key.value is None:
                continue
            structured[alias] = ArmParameter(
                alias,
                key.estimand,
                target.group,
                code(key.value),
                None if key.reference is None else code(key.reference),
            )
        return structured

    reference = result.config.reference_arm
    label = data.arm_label
    binary = data.is_binary_treatment
    out: dict[str, ArmParameter] = {}

    for arm in data.arm_codes:
        # ``ey`` labels its arms even on a two-armed fit -- see ``targets.builtin._ey``,
        # which says why -- so this name is the same at every arm count.
        name = parameter_name("ey", arm=label(arm))
        out[name] = ArmParameter(name, "ey", "mean", arm)
    if binary:
        # The two targets that name one of exactly two arms, and so exist only here.
        out["ey1"] = ArmParameter("ey1", "ey1", "mean", 1.0)
        out["ey0"] = ArmParameter("ey0", "ey0", "mean", 0.0)

    for arm in data.arm_codes:
        if arm == reference:
            continue
        for stem, group in (("ate", "mean"), ("att", "att"), ("atc", "atc")):
            # The two-armed report keeps the bare stems, exactly as
            # ``TargetContext.name_for`` collapses them.
            name = (
                parameter_name(stem)
                if binary
                else parameter_name(stem, arm=label(arm), versus=label(reference))
            )
            out[name] = ArmParameter(name, stem, group, arm, reference)
    return out
