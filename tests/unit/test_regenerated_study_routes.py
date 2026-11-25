"""Each study the density-ratio change regenerates reads the route its cells name.

``ratio=None`` resolves to the classifier for a policy on a continuous or vector dose.  A
registered cell that set no ``ratio=``, passed no ``density_bins``, used no ``bin_design``
learner and declared no mechanism would move route silently when its study is regenerated.
Every cell of these studies names its route, so none may resolve through that default; the
dose-response studies fit no policy ratio at all.  Each study's own loop runs with every fit
stopped once its route is resolved (:mod:`tests.studies.route_guard`), so the check costs the
draws and the fits a key makes before the last routed one it reads.

``mtp-density-tails`` is the one study whose cells use the default on purpose: its own design
test checks that each cell resolves to the route its fit set declares.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import pytest

from cleverly import _ratio_route as route_module
from cleverly._ratio_route import RatioRoute
from cleverly.estimators.tmle import TMLE
from tests.studies import route_guard

#: Each regenerated study's canonical module and properties module, or ``None`` where the
#: study has none.
STUDIES: dict[str, tuple[str | None, str | None]] = {
    "vector-mtp": ("canonical_vector_mtp", "vector_mtp_properties"),
    "mtp-boundaries": ("canonical_mtp_boundaries", "mtp_boundaries_properties"),
    "mtp-pricing": ("canonical_mtp_pricing", "mtp_pricing_properties"),
    "mtp-tuning": ("canonical_mtp_tuning", "mtp_tuning_properties"),
    "mtp-policy-variance": ("canonical_mtp_policy_variance", "mtp_policy_variance_properties"),
    "policy-point-mtp": ("canonical_policy_point_mtp", "policy_point_mtp_properties"),
    "longitudinal-mtp": ("canonical_longitudinal_mtp", "longitudinal_mtp_properties"),
    "longitudinal-sdr": ("canonical_longitudinal_sdr", "longitudinal_sdr_properties"),
    "shift-policies": ("canonical_shift_policies", "shift_policy_properties"),
    "stratified-incremental-msm": (
        "canonical_stratified_incremental_msm",
        "stratified_incremental_msm_properties",
    ),
    "hazard-mtp": ("canonical_hazard_mtp", "hazard_mtp_properties"),
    "refuter-calibration-band": ("refuter_calibration_band", "refuter_calibration_band_properties"),
    "dose-response": ("dose_response", "dose_response_properties"),
    "dose-response-npcausal": ("dose_response_npcausal", None),
}

#: The cells whose later fits name a route their first does not, and how many routed fits of
#: a key the guard reads there: ``mtp-boundaries``' ratio-necessity pair fits the declared
#: density, then the classifier control.  Every other key's fits share one route (measured
#: reading four fits a key), so one read is enough and keeps the check cheap.
FITS = {("mtp-boundaries", "properties"): 2}

#: The studies that fit no policy ratio, so no fit of theirs resolves a route and nothing
#: stops their fits: their primary draws are read at a small size.
UNROUTED = frozenset({"dose-response", "dose-response-npcausal"})
UNROUTED_N = 300

CASES = [
    pytest.param(study, phase, id=f"{study}-{phase}")
    for study, (canonical, properties) in STUDIES.items()
    for phase, module in (("primary", canonical), ("properties", properties))
    if module is not None
]


def _routes(study: str, phase: str) -> dict[str, tuple[RatioRoute, ...]]:
    canonical, properties = STUDIES[study]
    fits = FITS.get((study, phase), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if phase == "primary":
            module = importlib.import_module(f"tests.studies.{canonical}")
            n = UNROUTED_N if study in UNROUTED else None
            return route_guard.primary_routes(module, n=n, fits=fits)
        module = importlib.import_module(f"tests.studies.{properties}")
        return route_guard.property_routes(module, fits=fits)


@pytest.mark.parametrize(("study", "phase"), CASES)
def test_no_cell_resolves_through_the_default(study: str, phase: str) -> None:
    routes = _routes(study, phase)
    assert routes, f"{study} {phase}: the guard read no cell"
    assert route_guard.default_resolved(routes) == []
    routed = {cell for cell, read in routes.items() if read}
    if study in UNROUTED:
        assert routed == set()
    else:
        assert routed, f"{study} {phase}: no cell reached a routed fit"
    if (study, phase) in FITS:
        # The second read reaches the ratio-necessity control, which the first does not.
        read = routes["ratio_necessity/p2__declared"]
        assert [route.route for route in read] == ["density", "classifier"]


def test_a_cell_on_the_default_fails_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: every point fit resolves as if it had named no route."""

    def default(self: Any, data: Any) -> RatioRoute:
        return RatioRoute("classifier", route_module.POLICY_DOSE)

    monkeypatch.setattr(TMLE, "ratio_route", default)
    routes = _routes("policy-point-mtp", "primary")
    assert route_guard.default_resolved(routes) == sorted(routes)
