"""The reviewed semantic callback for ``docs/examples/point-treatment-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import navigation_protocol
from cleverly.datasets.synthetic import nonlinear_bounded_dgp
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_plugin_limits_refuse,
    assert_protocol_recorded,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "point-treatment-tmle.ipynb"


def _untreated_selection_bias(n: int = 400_000) -> float:
    """``E[Y^0 | A=1] - E[Y^0 | A=0]`` in the law behind ``navigation_data``, by Monte Carlo."""
    law = nonlinear_bounded_dgp()
    latent = np.random.default_rng(0).standard_normal((n, law.n_latent))
    g = law.propensity(latent)
    untreated = law.outcome_mean(latent, 0.0, None)
    return float(np.average(untreated, weights=g) - np.average(untreated, weights=1.0 - g))


def check(namespace: dict[str, Any]) -> None:
    """The ATE tutorial's protocol, failure mode, warnings, and sensitivity claims hold."""
    summary = namespace["effect"].summary()
    assert "Discharge-home order" in summary
    assert "scores death before day 30 as the worst transition score" in summary
    assert "stacked CV-TMLE" in namespace["result"].summary()
    assert list(namespace["frame"].columns[:2]) == ["transition_score", "transition_navigation"]

    # "The offered patients score ... higher", and the arms differ on discharge risk first.
    by_arm = namespace["by_arm"]
    truth = namespace["truth"]
    unadjusted = namespace["unadjusted"]
    assert unadjusted > truth["ate"]
    assert by_arm.loc[0.0, "discharge_risk"] < 0.0 < by_arm.loc[1.0, "discharge_risk"]
    assert by_arm.loc[1.0, "discharge_risk"] > by_arm.loc[0.0, "discharge_risk"] + 0.3
    # "the ATT exceeds the ATE", and "The offered patients would also score higher than the
    # others under usual support alone": both distortions have the same sign in this law, so
    # both raise the unadjusted difference.  The margins are the retired Gaussian law's, scaled
    # by the ratio of the two ATEs (0.1629 against 1.750).
    assert truth["att"] > truth["ate"] + 0.01
    assert _untreated_selection_bias() > 0.01
    # "On this draw, the unadjusted difference is closer to the ATT than to the ATE."
    assert abs(unadjusted - truth["att"]) < abs(unadjusted - truth["ate"])

    # The protocol step prints the record, and the fit and the restored artifact carry its digest.
    assert namespace["protocol"] == navigation_protocol()
    fingerprint = assert_protocol_recorded(
        NOTEBOOK, "protocol", namespace["protocol"], namespace["result"]
    )
    assert fingerprint in stored_output(NOTEBOOK, "save-and-restore")
    assert {item.name for item in namespace["replayed"].attention} == {
        item.name for item in namespace["assessment"].attention
    }
    flags = namespace["restored"].replayability
    assert (
        flags.summarize_existing_artifacts,
        flags.retarget_cached_nuisances,
        flags.evaluate_stored_representer,
        flags.refit_nuisances,
        flags.evaluate_new_data,
    ) == (True, True, False, True, False)

    # Step 7: the three estimates keep the law's order, and each interval covers its truth.
    # A swap of the ATT and ATC weights fails the order; the truth order alone cannot see it.
    spread_truth = namespace["spread_truth"]
    assert spread_truth["att"] > spread_truth["ate"] > spread_truth["atc"]
    assert spread_truth["atc"] < 0.25 * spread_truth["att"]
    points = namespace["spread_points"]
    assert points["att"].psi > points["ate"].psi > points["atc"].psi
    for key, point in points.items():
        assert covers(point, spread_truth[key]), key

    # Step 8: "misses ... by about three times either fit with one flexible learner"; 2.6x here.
    ate = truth["ate"]
    errors = {label: abs(point.psi - ate) for label, point in namespace["dr_points"].items()}
    one_flexible = max(errors["flexible Q, linear g"], errors["linear Q, flexible g"])
    assert errors["both linear"] > 2.0 * one_flexible
    # "each interval contains the true value", and the both-linear interval excludes it.  The
    # coverage claim is the sharper half of the reading and holds on this draw for all three.
    for label in ("flexible Q, linear g", "linear Q, flexible g"):
        assert covers(namespace["dr_points"][label], ate), label
    assert not covers(namespace["dr_points"]["both linear"], ate)
    assert covers(namespace["estimate"], ate)
    # "The summary prints no `outcome scaled` line": the declared q_bounds are the score's own
    # support, so the scaler is the identity.  The outcome range is the nonzero witness that the
    # declared support really holds the data rather than being padded around it.
    scaler = namespace["result"].nuisance.scaler
    assert (scaler.lower, scaler.upper) == (0.0, 1.0)
    assert scaler.is_identity
    assert "outcome scaled" not in namespace["result"].summary()
    outcome = np.asarray(namespace["frame"]["transition_score"], dtype=float)
    assert scaler.lower < outcome.min() < outcome.max() < scaler.upper

    # Step 9: exactly one warning, and the score-equation check passed.
    assessment = namespace["assessment"]
    assert tuple(item.name for item in assessment.attention) == ("nuisance_models",)
    ledger = assessment.to_frame()
    rows = ledger[ledger["check"] == "nuisance_models"]
    assert rows["detail"].str.contains("propensity is poorly calibrated").any()
    assert set(ledger.loc[ledger["check"] == "score_equations", "status"]) == {"passed"}
    # "a calibration slope below 1": the fitted propensities are more extreme than the rates.
    (propensity,) = (m for m in namespace["nuisance"].models if m.name == "propensity")
    assert propensity.metrics["calibration_slope"] < 1.0
    assert 0.0 < namespace["support"].truncated["fraction"] < 0.05

    # Step 9b: "From the fitted bound onward it falls at every step", "more than a quarter
    # of the patients" at the largest bound, and the fitted bound comes from 5 / (sqrt(n) log n).
    # The two margins are the retired Gaussian law's, scaled by the ratio of the two ATEs.
    curve = namespace["curve"]
    n = len(namespace["frame"])
    fitted_bound = 5.0 / (np.sqrt(n) * np.log(n))
    position = int(curve.index.get_loc((curve["bound"] - fitted_bound).abs().idxmin()))
    assert curve["std_err"].iloc[position:].is_monotonic_decreasing
    assert curve["std_err"].iloc[0] > curve["std_err"].iloc[-1] + 0.003
    assert float(np.ptp(curve["psi"].to_numpy())) > 0.001
    assert np.all(np.diff(curve["truncated_fraction"].to_numpy()) > 0.0)
    assert curve["truncated_fraction"].iloc[-1] > 0.25
    nearest = curve.loc[(curve["bound"] - fitted_bound).abs().idxmin()]
    assert abs(nearest["bound"] - fitted_bound) < 1e-12
    # Nonzero witness: a retarget at the fitted bound reproduces the Step 6 estimate.
    assert abs(nearest["psi"] - namespace["estimate"].psi) < 1e-10

    # Step 10: the default estimator of nu^2 refuses on this fit, and the page says so before
    # it names the plug-in. "The doubly robust estimator returned -7.96655", which the refusal
    # prints. A mutation that restores the silent fallback returns a number here instead.
    result = namespace["result"]
    refusal = namespace["nu2_refusal"]
    assert "'doubly_robust'" in refusal
    assert "-7.96655" in refusal
    assert "not a substitute" in refusal
    assert result.sensitivity.elements(estimand="ate", nu2_estimator="plugin").nu2 > 0.0
    # "equal strengths of 0.233 move the point estimate to zero" at rho = 1.
    robustness = namespace["robustness"]
    assert 0.2 < robustness["rv"] < 0.33
    at_rv = result.sensitivity.omitted_confounding(
        cf_y=robustness["rv"], cf_d=robustness["rv"], nu2_estimator="plugin"
    )
    assert abs(at_rv.lower) < 1e-3
    # "The implied cf_y reaches 1.0000 ... so this covariate calibrates no bound here": the
    # refusal is the bound formula's own, and it is what sends the step to a second covariate.
    strong = namespace["strong"]
    assert strong.covariates == ("discharge_risk",)
    assert strong.cf_y == 1.0
    with pytest.raises(ValueError, match=r"cf_y must lie in \[0, 1\)"):
        result.sensitivity.omitted_confounding(
            cf_y=strong.cf_y, cf_d=strong.cf_d, rho=1.0, nu2_estimator="plugin"
        )
    benchmark = namespace["benchmark"]
    assert benchmark.covariates == ("medication_burden",)
    bounds = namespace["bounds"]
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (benchmark.cf_y, benchmark.cf_d, 1.0)
    # "little of the remaining outcome variation and much of the remaining treatment variation".
    assert benchmark.cf_y < 0.2 < 0.4 < benchmark.cf_d
    # "The range contains zero."
    assert bounds.confounding_strength > at_rv.confounding_strength
    assert bounds.lower < 0.0 < bounds.psi < bounds.upper
    # "The last call refuses": the plug-in bound reports no limit and no confidence-limit
    # value, and the page prints the F26 reason. A mutation that restores the plug-in limits
    # returns a number at the page's ``ci_lower`` instead, and the page raises.
    assert_plugin_limits_refuse(result, bounds, namespace)
