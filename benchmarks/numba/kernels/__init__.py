"""The kernels under test, and what qualifies one.

A kernel gets an entry here only on evidence, per the inventory's rule: it takes a
measurable share of the cached-nuisance pipeline, or it is called enough times that its
cumulative share is measurable, or it allocates something large enough to matter, or it
contains a Python loop over rows, times, candidates, arms or clusters.  "It contains
arithmetic" is not a reason, and the kernels that are here *despite* failing that test --
the Newton solve, the Gram contraction -- are labelled negative controls and say so.

Each :class:`KernelSpec` binds together the four things a comparison needs: how to build
the input, the implementations to compare, how to decide they agree, and which dimension
the parallelism would run along.  The runner needs nothing else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["REGISTRY", "KernelSpec", "register", "resolve"]


@dataclass(frozen=True)
class KernelSpec:
    """One comparable computation.

    Attributes
    ----------
    name:
        Stable identifier, written into every result row.
    estimator:
        Which flavour's pipeline it belongs to, so the summary can be sectioned by
        estimator rather than by kernel.
    build:
        ``(**dimensions) -> inputs``.  Called *outside* the timed region.
    implementations:
        ``{implementation_name: callable(inputs) -> output}``.  ``"numpy"`` is required
        and is the reference every other one is judged against.
    compare:
        ``(reference_output, candidate_output) -> (max_abs_error, max_rel_error)``.
    tolerance:
        ``(atol, rtol)`` the kernel declares.  A parallel reduction reassociates, so this
        is per-kernel rather than global; a kernel whose output is a solved coefficient
        needs a different bar from one whose output is a sum of a million terms.
    parallel_axis:
        What the parallel implementations run along -- ``rows``, ``replicates``,
        ``clusters``, ``folds``, ``regimens``, ``candidates``, ``horizons`` -- or ``None``
        where there is no independent axis.  Recorded so the report can say *why* a
        speed-up did or did not appear.
    negative_control:
        True for kernels included to be measured and rejected.  Kept because a suite that
        drops its negative results cannot be used to check that the positive ones are not
        an artefact of the harness.
    note:
        One line for the summary, usually the reason the kernel is here.
    """

    name: str
    estimator: str
    build: Callable[..., Any]
    implementations: Mapping[str, Callable[[Any], Any]]
    compare: Callable[[Any, Any], tuple[float, float]]
    tolerance: tuple[float, float] = (1e-12, 1e-10)
    parallel_axis: str | None = None
    negative_control: bool = False
    note: str = ""
    #: Dimensions this kernel accepts, with the defaults the runner uses when a config
    #: does not name them.  Anything not listed is refused rather than ignored, so a
    #: typo in a YAML file fails instead of quietly benchmarking the default.
    dimensions: Mapping[str, Any] = field(default_factory=dict)
    #: Whether a per-call amortisation curve is worth taking.  True for the kernels a
    #: repeated workload actually calls repeatedly.
    amortise: bool = False

    def inputs(self, **dimensions: Any) -> Any:
        unknown = sorted(set(dimensions) - set(self.dimensions))
        if unknown:
            raise KeyError(
                f"kernel {self.name!r} does not take dimension(s) {unknown}; "
                f"it takes {sorted(self.dimensions)}"
            )
        merged = {**self.dimensions, **dimensions}
        return self.build(**merged)


REGISTRY: dict[str, KernelSpec] = {}


def register(spec: KernelSpec) -> KernelSpec:
    if spec.name in REGISTRY:
        raise KeyError(f"kernel {spec.name!r} is already registered")
    if "numpy" not in spec.implementations:
        raise ValueError(
            f"kernel {spec.name!r} has no 'numpy' implementation; every speed-up here is "
            "a ratio to the numpy reference and every correctness gate is stated against "
            "it, so a kernel without one has nothing to be compared to"
        )
    REGISTRY[spec.name] = spec
    return spec


def resolve(names: Sequence[str] | None) -> list[KernelSpec]:
    """The kernels named, or every registered one.  ``all`` is a synonym for every."""
    _load_all()
    if not names or list(names) == ["all"]:
        return list(REGISTRY.values())
    missing = sorted(set(names) - set(REGISTRY))
    if missing:
        raise KeyError(f"unknown kernel(s) {missing}; have {sorted(REGISTRY)}")
    return [REGISTRY[name] for name in names]


def _load_all() -> None:
    """Import every kernel module, which is what populates the registry."""
    from . import (  # noqa: F401
        bootstrap,
        clustered,
        collaborative,
        cvtmle,
        drtmle,
        influence_curves,
        longitudinal,
        newton,
        one_step,
        survival,
    )
