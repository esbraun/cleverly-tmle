"""``CausalStudy`` on a repeated outcome follows the allowlist of estimators with a reduction.

WC: the method listing and the fit share one rule.  ``REPEATED_OUTCOME_KINDS`` names the
kinds that fit a repeated outcome.  A kind outside it is unavailable in
:meth:`IdentifiedEffect.available_methods`, and :meth:`IdentifiedEffect.estimate` refuses it
with the reason that the listing gives.  ``RegimeMean(..., method="sdr")`` reaches the ``SDR``
fit and equals the direct fit, horizon by horizon.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, RegimeContrast, RegimeMean
from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import SDR
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
    for name in ("tmle", "sdr"):
        assert methods[name].available and methods[name].reason is None
    assert not methods["ipsi"].available
    assert methods["ipsi"].reason == sequential.repeated_outcome_refusal("IPSI")
    assert "LTMLE and SDR fit one" in methods["ipsi"].reason


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
            assert methods[name].reason == sequential.repeated_outcome_refusal(TITLES[name])


def test_wc_the_fit_refuses_with_the_reason_of_the_listing() -> None:
    effect = _effect()
    reason = _methods(effect)["ipsi"].reason
    with pytest.raises(CapabilityError) as raised:
        effect.estimate(method="ipsi")
    assert reason is not None and reason in str(raised.value)


def test_wc_an_end_of_study_design_is_not_restricted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The allowlist reads a repeated outcome only: an end-of-study outcome lists all three."""
    monkeypatch.setattr(sequential, "REPEATED_OUTCOME_KINDS", frozenset())
    roles = {**ROLES, "outcome": "Y3", "time_varying": [["L1"], ["Y1", "L2"], ["Y2", "L3"]]}
    methods = _methods(_effect(roles=roles))
    assert all(methods[name].available for name in TITLES)


# ---------------------------------------------------------------- SDR: the typed request

PLANS = {"always": 1, "never": 0}


def _learners() -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "censoring_learner": LogisticRegression(),
    }


def _by_study(estimand: Any) -> Any:
    design = LongitudinalTreatment(**ROLES)
    study = CausalStudy(design.prepare(panel()), design=design)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(estimand).estimate(
            method="sdr", cross_fit=False, simultaneous=False, random_state=0, **_learners()
        )


def _direct(horizons: list[int] | None) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SDR(
            PLANS,
            reference="never",
            horizons=horizons,
            n_folds=1,
            simultaneous=False,
            random_state=0,
            **_learners(),
        ).fit(panel(), **ROLES)


def check_wc_regime_horizons() -> None:
    """``RegimeMean(..., horizons=[1, 3], method="sdr")`` reports those horizons only."""
    result = _by_study(RegimeMean(PLANS, horizons=[1, 3]))
    direct = _direct([1, 3])
    assert result.config.estimator == "sdr"
    expected = {name for name in direct if name.startswith("ey_regimen[")}
    assert set(result) == expected, "reported names"
    assert {name.rsplit("@ ", 1)[1] for name in result} == {"t=1]", "t=3]"}, "reported horizons"
    for name in result:
        assert result.psi(name) == pytest.approx(direct.psi(name), abs=1e-12), f"estimate: {name}"
        np.testing.assert_allclose(
            result.influence_curves[name],
            direct.influence_curves[name],
            atol=1e-12,
            rtol=0,
            err_msg=f"curve: {name}",
        )


def test_wc_a_regime_mean_by_sdr_equals_the_direct_fit() -> None:
    check_wc_regime_horizons()


def test_wc_a_regime_contrast_by_sdr_reports_the_declared_horizon() -> None:
    result = _by_study(RegimeContrast(PLANS, reference="never", horizons=[2]))
    direct = _direct([2])
    assert set(result) == {name for name in direct if name.startswith("ate_regimen[")}
    assert all(name.endswith("@ t=2]") for name in result)
    for name in result:
        assert result.psi(name) == pytest.approx(direct.psi(name), abs=1e-12), name


def test_wc_no_horizons_reports_every_node() -> None:
    result = _by_study(RegimeMean(PLANS))
    assert set(result) == {name for name in _direct(None) if name.startswith("ey_regimen[")}
    assert {name.rsplit("@ ", 1)[1] for name in result} == {"t=1]", "t=2]", "t=3]"}
