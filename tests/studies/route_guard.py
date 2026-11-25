"""The route each cell of a regenerated study records, read without fitting a nuisance.

A policy fit records the ratio route it used and the rule that chose it
(``config.ratio_route``).  A study regenerated after ``ratio=None`` became the classifier for a
policy on a continuous dose must still read the route its rows name.  Every registered cell names
its route, by ``ratio=``, ``density_bins``, a ``bin_design`` learner or a declared mechanism, so
none may resolve through the default: a cell that did would move route silently.

:func:`property_routes` and :func:`primary_routes` run a study's own loop with a fit stopped as
soon as its data are prepared, after the estimator has resolved its route and before any learner
runs.  The first key of each cell is read once per routed fit it makes: the read that records
the k-th fit lets the k - 1 fits before it run in full, so a cell whose later fits are costly
can be read on its first ``fits`` only.  :func:`default_resolved` names the cells with a fit that
resolved through the default.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

from cleverly import _ratio_route as route_module
from cleverly._ratio_route import RatioRoute
from cleverly.estimators.tmle import TMLE
from tests.studies.evidence import replication

longitudinal_module = importlib.import_module("cleverly.longitudinal.estimator")


class _Resolved(BaseException):
    """Raised once a fit has resolved its route; a ``BaseException`` so no study catches it."""

    def __init__(self, route: RatioRoute) -> None:
        super().__init__(route)
        self.route = route


@contextlib.contextmanager
def stopping_at_resolution(skip: int = 0) -> Iterator[None]:
    """The routed fit after the first ``skip`` inside raises :class:`_Resolved` with its route.

    The point fit resolves in ``TMLE._resolved`` on every path; the longitudinal fit checks its
    prepared data in ``_refuse_prepared`` on every path, before the resolution.  The first
    ``skip`` routed fits run in full.
    """
    point = TMLE._resolved
    sequential = longitudinal_module._SequentialEstimator._refuse_prepared
    seen = [0]

    def reached(route: RatioRoute) -> bool:
        seen[0] += 1
        if seen[0] > skip:
            raise _Resolved(route)
        return False

    def point_spy(self: Any, data: Any) -> Any:
        reached(self.ratio_route(data))
        return point(self, data)

    def sequential_spy(self: Any, prepared: Any) -> Any:
        reached(self.ratio_route(prepared))
        return sequential(self, prepared)

    TMLE._resolved = point_spy  # type: ignore[method-assign]
    longitudinal_module._SequentialEstimator._refuse_prepared = sequential_spy
    try:
        yield
    finally:
        TMLE._resolved = point  # type: ignore[method-assign]
        longitudinal_module._SequentialEstimator._refuse_prepared = sequential


def route_of(call: Callable[[], Any], skip: int = 0) -> RatioRoute | None:
    """The route of the routed fit of ``call`` after the first ``skip``, or ``None``."""
    with stopping_at_resolution(skip):
        try:
            call()
        except _Resolved as stop:
            return stop.route
    return None


def routes_of(call: Callable[[], Any], fits: int) -> tuple[RatioRoute, ...]:
    """The routes of the first ``fits`` routed fits of ``call``, in order."""
    routes: list[RatioRoute] = []
    for skip in range(fits):
        route = route_of(call, skip)
        if route is None:
            break
        routes.append(route)
    return tuple(routes)


class _Collected(Exception):
    """Stops a study's loop once its keys are read."""


def _first_keys(keys: Any, cell_of: Callable[[Any], str]) -> list[Any]:
    seen: dict[str, Any] = {}
    for key in keys:
        seen.setdefault(cell_of(key), key)
    return list(seen.values())


def property_routes(
    module: ModuleType, generate: Callable[[], Any] | None = None, *, fits: int = 1
) -> dict[str, tuple[RatioRoute, ...]]:
    """The routes of the first ``fits`` fits of each cell's first key, by ``cell_of`` name.

    The module's ``fit_properties`` is replaced, for the call, by a reader that reads the first
    key of each cell through :func:`routes_of` and returns no rows.  ``generate`` runs the
    module's loop; ``module.generate_property_rows(n_jobs=1)`` by default.  What the module does
    with the empty rows afterwards is not run past its first error.
    """
    routes: dict[str, tuple[RatioRoute, ...]] = {}

    def reader(keys: Any, fit_one: Any, *, cell_of: Any, draw: Any = None, **_: Any) -> Any:
        for key in _first_keys(keys, cell_of):
            drawn = None if draw is None else draw(key)
            routes[str(cell_of(key))] = routes_of(
                lambda key=key, drawn=drawn: fit_one(key, drawn), fits
            )
        return [[] for _ in keys]

    original = module.fit_properties
    module.fit_properties = reader
    try:
        (generate or (lambda: module.generate_property_rows(n_jobs=1)))()
    except Exception:  # the module's own code after the loop reads no rows
        pass
    finally:
        module.fit_properties = original
    return routes


def primary_routes(
    module: ModuleType, *, n: int | None = None, fits: int = 1
) -> dict[str, tuple[RatioRoute, ...]]:
    """The routes of the first ``fits`` fits of each primary scenario's first replicate."""
    routes: dict[str, tuple[RatioRoute, ...]] = {}

    def reader(keys: Any, replicate: Any, *, draw: Any, **_: Any) -> Any:
        for key in _first_keys(keys, lambda key: str(key[0])):
            drawn = draw(key)
            routes[str(key[0])] = routes_of(
                lambda key=key, drawn=drawn: replicate(key, drawn), fits
            )
        raise _Collected

    original = replication.fit_primary
    replication.fit_primary = reader
    try:
        module.draw_and_fit(replicates=1, n=n or module.STUDY.n, n_jobs=1)
    except _Collected:
        pass
    finally:
        replication.fit_primary = original
    return routes


def default_resolved(routes: dict[str, tuple[RatioRoute, ...]]) -> list[str]:
    """The cells with a fit whose route came from the policy-dose default, not from the cell."""
    return sorted(
        cell
        for cell, read in routes.items()
        if any(route.reason == route_module.POLICY_DOSE for route in read)
    )


__all__ = ["default_resolved", "primary_routes", "property_routes", "route_of", "routes_of"]
