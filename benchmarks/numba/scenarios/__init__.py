"""Complete post-nuisance pipelines, measured through the package rather than around it.

The kernels answer "is this computation faster compiled".  A scenario answers the question
underneath that one: **how much of a real estimator's post-nuisance work is any of these
kernels**.  Both are needed, and the second is the one that decides whether a 9x kernel is
worth adopting -- a 9x speed-up on 2% of the pipeline is a 1.8% speed-up.

A scenario therefore does the opposite of what the kernels do.  It calls the *shipped*
code, at its real API, with the nuisances cached: fit once outside the timed region, then
time ``retarget`` (or the equivalent) and report it as a share of the fit.  Nothing here is
reimplemented and nothing is compiled; what a scenario produces is a denominator.

**Two denominators are reported, and quoting one without the other is the standard way to
mislead with this measurement.**  A ``library="glm"`` fit is the cheapest preset the package
offers, so a post-nuisance share against it is several-fold larger than against a fit
anybody runs; a ``library="default"`` fit costs roughly 37x more per row, and the same
kernel is then a rounding error.  Both are run, and the summary reports the pair.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = ["REGISTRY", "ScenarioResult", "ScenarioSpec", "register", "resolve"]


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario's measurement."""

    name: str
    library: str
    n: int
    fit_seconds: float
    post_nuisance_seconds: float
    detail: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def share(self) -> float:
        """Post-nuisance work as a share of the whole fit."""
        return self.post_nuisance_seconds / self.fit_seconds if self.fit_seconds else float("nan")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    estimator: str
    run: Callable[..., ScenarioResult]
    note: str = ""


REGISTRY: dict[str, ScenarioSpec] = {}


def register(spec: ScenarioSpec) -> ScenarioSpec:
    if spec.name in REGISTRY:
        raise KeyError(f"scenario {spec.name!r} is already registered")
    REGISTRY[spec.name] = spec
    return spec


def resolve(names: Sequence[str] | None) -> list[ScenarioSpec]:
    _load_all()
    if not names or list(names) == ["all"]:
        return list(REGISTRY.values())
    missing = sorted(set(names) - set(REGISTRY))
    if missing:
        raise KeyError(f"unknown scenario(s) {missing}; have {sorted(REGISTRY)}")
    return [REGISTRY[name] for name in names]


def _load_all() -> None:
    from . import pipelines  # noqa: F401
