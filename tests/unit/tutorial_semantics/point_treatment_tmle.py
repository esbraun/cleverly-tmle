"""The reviewed semantic callback for ``docs/examples/point-treatment-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cleverly.datasets import navigation_protocol
from cleverly.datasets.synthetic import nonlinear_dgp
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "point-treatment-tmle.ipynb"


def _untreated_selection_bias(n: int = 400_000) -> float:
    """``E[Y^0 | A=1] - E[Y^0 | A=0]`` in the law behind ``navigation_data``, by Monte Carlo."""
    law = nonlinear_dgp()
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
    # "the ATT exceeds the ATE", and "Other terms of the law give the offered patients lower
    # scores without the offer": the two distortions have opposite signs in the law.
    assert truth["att"] > truth["ate"] + 0.1
    assert _untreated_selection_bias() < -0.05
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

    # Step 8: "misses by several times more than either fit with one flexible learner"; about 6x here.
    ate = truth["ate"]
    errors = {label: abs(point.psi - ate) for label, point in namespace["dr_points"].items()}
    one_flexible = max(errors["flexible Q, linear g"], errors["linear Q, flexible g"])
    assert errors["both linear"] > 3.0 * one_flexible
    # "its interval ... excludes the true value", while the Step 6 interval contains it.
    assert not covers(namespace["dr_points"]["both linear"], ate)
    assert covers(namespace["estimate"], ate)
    # The summary prints padded scaling bounds, not the observed outcome range. The signed
    # endpoints witness both the padding and the direction of the narrated negative value.
    scaler = namespace["result"].nuisance.scaler
    assert (round(scaler.lower, 3), round(scaler.upper, 2)) == (-4.372, 14.81)
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

    # Step 9b: "The standard error falls ... as the bound clips more rows", "more than a quarter
    # of the patients" at the largest bound, and the fitted bound comes from 5 / (sqrt(n) log n).
    curve = namespace["curve"]
    assert curve["std_err"].is_monotonic_decreasing
    assert curve["std_err"].iloc[0] > curve["std_err"].iloc[-1] + 0.1
    assert float(np.ptp(curve["psi"].to_numpy())) > 0.02
    assert np.all(np.diff(curve["truncated_fraction"].to_numpy()) > 0.0)
    assert curve["truncated_fraction"].iloc[-1] > 0.25
    n = len(namespace["frame"])
    fitted_bound = 5.0 / (np.sqrt(n) * np.log(n))
    nearest = curve.loc[(curve["bound"] - fitted_bound).abs().idxmin()]
    assert abs(nearest["bound"] - fitted_bound) < 1e-12
    # Nonzero witness: a retarget at the fitted bound reproduces the Step 6 estimate.
    assert abs(nearest["psi"] - namespace["estimate"].psi) < 1e-10

    # Step 10: "equal strengths of 0.264 move the point estimate to zero" at rho = 1.
    result = namespace["result"]
    robustness = namespace["robustness"]
    assert 0.2 < robustness["rv"] < 0.33
    at_rv = result.sensitivity.omitted_confounding(cf_y=robustness["rv"], cf_d=robustness["rv"])
    assert abs(at_rv.lower) < 1e-3
    benchmark = namespace["benchmark"]
    assert benchmark.covariates == ("discharge_risk",)
    bounds = namespace["bounds"]
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (benchmark.cf_y, benchmark.cf_d, 1.0)
    # "much of the remaining outcome variation but little of the remaining treatment variation",
    # so "Its combined strength is therefore below that of equal strengths of 0.264".
    assert benchmark.cf_y > 0.5 > 0.1 > benchmark.cf_d
    assert bounds.confounding_strength < at_rv.confounding_strength
    assert 0.0 < bounds.lower < bounds.psi < bounds.upper
    # "the limits ... still exclude zero", and each one-sided limit lies outside its bound.
    assert 0.0 < bounds.ci_lower < bounds.lower
    assert bounds.ci_upper > bounds.upper
