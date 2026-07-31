"""Treatment regimens: what a unit would have been given at every time point.

A *regime* (:mod:`cleverly.interventions`) says what one treatment decision would have
been.  A **regimen** says what the whole sequence would have been --
:math:`\\bar a = (a_1, \\ldots, a_T)` -- and it is the thing a longitudinal fit's
parameters are indexed by.  The two words are one letter apart and name different
objects, which is why this module spells the distinction out rather than reusing
:class:`~cleverly.interventions.base.Intervention`: an intervention is a density over
arms at *one* node, and a regimen is a plan across nodes.

Only *static* regimens live here.  A dynamic rule :math:`d_t(H_t)` -- treat once the
biomarker crosses a threshold, say -- is the natural next step and is refused by name
rather than approximated, because it changes which rows the sequential regression is
fitted on (the followers of the rule, not of a constant) and so has to be checked
against an oracle of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..exceptions import DataError

__all__ = ["Regimen", "resolve_regimens"]


@dataclass(frozen=True)
class Regimen:
    """A static treatment plan: one arm per time point, under the user's own label.

    Attributes
    ----------
    label:
        What the reported parameter is named by -- ``ey_regimen[always]``.  Part of
        the estimand rather than decoration: two regimens are two different
        parameters, and the report has to be able to say which is which.
    values:
        The arm assigned at each time point, ``0.0`` or ``1.0``, one per node.
    """

    label: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise DataError(f"regimen {self.label!r} assigns no treatment at any time point")
        for time, value in enumerate(self.values, start=1):
            if value not in (0.0, 1.0):
                raise DataError(
                    f"regimen {self.label!r} assigns {value!r} at time {time}; a longitudinal "
                    "fit takes a binary treatment at every node, so each entry must be 0 or 1"
                )

    @property
    def n_times(self) -> int:
        return len(self.values)

    def at(self, time: int) -> float:
        """The arm assigned at ``time``, counted from one."""
        return self.values[time - 1]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        plan = "".join(str(int(value)) for value in self.values)
        return f"Regimen({self.label!r}, {plan})"


def resolve_regimens(spec: Any, n_times: int) -> tuple[Regimen, ...]:
    """Turn a user's ``regimens=`` argument into an ordered tuple of :class:`Regimen`.

    Accepts a mapping from label to plan, where a plan is either a sequence of
    ``n_times`` arms or a single arm meaning "that arm at every node".  A sequence of
    :class:`Regimen` objects passes through.

    Order is preserved, because the first regimen is the one contrasts are taken
    against by default and so is part of what the fit reports.
    """
    if spec is None:
        raise DataError(
            "a longitudinal fit needs regimens= : the parameter is the mean outcome under "
            "a treatment plan, and there is no default plan to fall back on. Pass, for "
            "example, regimens={'always': 1, 'never': 0}"
        )
    if isinstance(spec, Regimen):
        spec = (spec,)
    if isinstance(spec, Mapping):
        items: list[tuple[str, Any]] = list(spec.items())
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        items = []
        for entry in spec:
            if not isinstance(entry, Regimen):
                raise DataError(
                    "a sequence of regimens must hold Regimen objects; pass a mapping "
                    "{label: plan} to name plans inline"
                )
            items.append((entry.label, entry.values))
    else:
        raise DataError(f"regimens= must be a mapping or a sequence of Regimen; got {spec!r}")

    if not items:
        raise DataError("regimens= is empty; a fit with no regimen reports no parameter")

    resolved: list[Regimen] = []
    seen: set[str] = set()
    for label, plan in items:
        name = str(label)
        if name in seen:
            raise DataError(f"regimen label {name!r} appears twice; labels name parameters")
        seen.add(name)
        resolved.append(Regimen(name, _plan(name, plan, n_times)))
    return tuple(resolved)


def _plan(label: str, plan: Any, n_times: int) -> tuple[float, ...]:
    """Read one plan, broadcasting a scalar arm across the time points."""
    if isinstance(plan, Regimen):
        plan = plan.values
    if isinstance(plan, (int, float)) and not isinstance(plan, bool):
        return (float(plan),) * n_times
    if isinstance(plan, bool):
        return (float(plan),) * n_times
    if isinstance(plan, str) or not isinstance(plan, Sequence):
        raise DataError(
            f"regimen {label!r} must be an arm (0 or 1) or a sequence of {n_times} arms; "
            f"got {plan!r}. A rule that reads the history is not supported yet -- see "
            "the longitudinal section of the README"
        )
    values = tuple(float(value) for value in plan)
    if len(values) != n_times:
        raise DataError(
            f"regimen {label!r} assigns {len(values)} arm(s) but the data has {n_times} "
            "treatment node(s); a plan must say what happens at every one of them"
        )
    return values
