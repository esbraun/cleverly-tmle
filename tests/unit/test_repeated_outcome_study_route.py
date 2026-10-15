"""``CausalStudy`` on a repeated outcome follows the allowlist of estimators with a reduction.

WC: the method listing and the fit share one rule.  ``REPEATED_OUTCOME_KINDS`` names the
kinds that fit a repeated outcome.  A kind outside it is unavailable in
:meth:`IdentifiedEffect.available_methods`, and :meth:`IdentifiedEffect.estimate` refuses it
with the reason that the listing gives.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly import CausalStudy, RegimeContrast
from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import estimator as sequential
from cleverly.study import LongitudinalTreatment

from .test_repeated_outcome import panel
from .test_repeated_outcome_container import ROLES

#: The estimator kind of each method name, and the title the refusal gives it.
TITLES = {"tmle": "LTMLE", "sdr": "SDR", "ipsi": "IPSI"}


def _effect(frame: Any = None, roles: dict[str, Any] | None = None) -> Any:
    design = LongitudinalTreatment(**(ROLES if roles is None else roles))
    study = CausalStudy(design.prepare(panel() if frame is None else frame), design=design)
    return study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never"))


def _methods(effect: Any) -> dict[str, Any]:
    return {record.name: record for record in effect.available_methods()}


def test_wc_the_listing_withholds_every_kind_outside_the_allowlist() -> None:
    methods = _methods(_effect())
    assert methods["tmle"].available and methods["tmle"].reason is None
    for name in ("sdr", "ipsi"):
        assert not methods[name].available
        assert methods[name].refusal == sequential.repeated_outcome_refusal(TITLES[name])
    assert "LTMLE fits one" in methods["sdr"].reason


@pytest.mark.parametrize("kinds", [{"tmle"}, {"tmle", "sdr"}, {"tmle", "ipsi"}, {"sdr", "ipsi"}])
def test_wc_the_listing_follows_the_allowlist(
    kinds: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each method is available exactly when its kind is in the set."""
    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", frozenset(kinds))
    methods = _methods(_effect())
    for name in TITLES:
        assert methods[name].available == (name in kinds), name
        if name in kinds:
            assert methods[name].reason is None
        else:
            assert methods[name].refusal == sequential.repeated_outcome_refusal(TITLES[name])


def test_wc_the_fit_refuses_with_the_reason_of_the_listing() -> None:
    effect = _effect()
    for name in ("sdr", "ipsi"):
        reason = _methods(effect)[name].reason
        with pytest.raises(CapabilityError) as raised:
            effect.estimate(method=name)
        assert reason is not None and reason in str(raised.value)


def test_wc_an_end_of_study_design_is_not_restricted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The allowlist reads a repeated outcome only: an end-of-study outcome lists all three."""
    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", frozenset())
    roles = {**ROLES, "outcome": "Y3", "time_varying": [["L1"], ["Y1", "L2"], ["Y2", "L3"]]}
    methods = _methods(_effect(roles=roles))
    assert all(methods[name].available for name in TITLES)
