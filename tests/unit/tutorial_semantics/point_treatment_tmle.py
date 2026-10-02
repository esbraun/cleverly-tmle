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


def _law_moments(n: int = 400_000) -> dict[str, float]:
    """``nu^2 = E[1 / {g (1 - g)}]`` and the share of true ``g`` below 0.1, by Monte Carlo."""
    law = nonlinear_bounded_dgp()
    g = law.propensity(np.random.default_rng(1).standard_normal((n, law.n_latent)))
    return {"nu2": float(np.mean(1.0 / (g * (1.0 - g)))), "below_0.1": float(np.mean(g < 0.1))}


_PROBE = "reviews/notebook-review/probes/point-treatment-tmle-final"
_STUDY = "tests/canonical/tmle3_cvtmle/properties.csv"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.98": f"mean SE over empirical SD of the Step 6 fit, 200 draws: {_PROBE}/summary.log",
    "1.9": f"share (0.0188) of the law's true propensity below 0.1: {_PROBE}/truth.log",
    "4.83": f"the law's nu^2 = E[1/(g(1-g))], 4.8286 by Monte Carlo: {_PROBE}/truth.log",
    "0.456": f"largest implied cf_d of medication_burden over 200 draws: {_PROBE}/summary.log",
    "4.48": f"mean doubly robust nu^2 over 200 draws, 4.479: {_PROBE}/summary.log",
    "0.431": f"RV of the shown draw at the law's nu^2, by summarize.py: {_PROBE}/summary.log",
    "0.142": f"lower bound of the shown draw at the law's nu^2, 0.1422: {_PROBE}/summary.log",
    "0.025": "the study's propensity bounds: tests/studies/canonical_cvtmle.py G_BOUNDS",
    "0.975": "the study's propensity bounds: tests/studies/canonical_cvtmle.py G_BOUNDS",
    "0.945": f"coverage of crossfit_overfitting/stacked_cvtmle: {_STUDY}",
    "0.5225": f"coverage of crossfit_overfitting/in_sample_control: {_STUDY}",
}


def check(namespace: dict[str, Any]) -> None:
    """The ATE tutorial's protocol, failure mode, diagnostics, and sensitivity claims hold."""
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

    # Step 6: the headline interval covers the truth on this draw.  The repeated-draw numbers
    # the reading quotes (189 of 200, SE/SD 0.98) come from the committed probe.
    ate = truth["ate"]
    assert covers(namespace["estimate"], ate)

    # Step 8: "The both-linear fit ... excludes that value", and each fit with one flexible
    # learner contains it.  The probe gives the repeated-draw counts.
    errors = {label: abs(point.psi - ate) for label, point in namespace["dr_points"].items()}
    one_flexible = max(errors["flexible Q, linear g"], errors["linear Q, flexible g"])
    assert errors["both linear"] > one_flexible
    for label in ("flexible Q, linear g", "linear Q, flexible g"):
        assert covers(namespace["dr_points"][label], ate), label
    assert not covers(namespace["dr_points"]["both linear"], ate)
    # "The summary prints no `outcome scaled` line": the declared q_bounds are the score's own
    # support, so the scaler is the identity.  The outcome range is the nonzero witness that the
    # declared support really holds the data rather than being padded around it.
    scaler = namespace["result"].nuisance.scaler
    assert (scaler.lower, scaler.upper) == (0.0, 1.0)
    assert scaler.is_identity
    assert "outcome scaled" not in namespace["result"].summary()
    outcome = np.asarray(namespace["frame"]["transition_score"], dtype=float)
    assert scaler.lower < outcome.min() < outcome.max() < scaler.upper

    # Step 9: no row needs attention, the score-equation check passed, and the shallow booster's
    # calibration slope is near 1.  The verdict the reading quotes is the report's own.
    assessment = namespace["assessment"]
    assert tuple(item.name for item in assessment.attention) == ()
    ledger = assessment.to_frame()
    assert set(ledger.loc[ledger["check"] == "score_equations", "status"]) == {"passed"}
    assert "nuisance fits look reasonable" in namespace["nuisance"].summary()
    (propensity,) = (m for m in namespace["nuisance"].models if m.name == "propensity")
    assert abs(propensity.metrics["calibration_slope"] - 1.0) < 0.1
    # "no unit sits at the truncation bound", and the smallest fitted propensity lies above
    # every bound up to 0.05 in the curve.
    support = namespace["support"]
    assert support.truncated["fraction"] == 0.0
    assert support.truncated["most_extreme"] > 0.05
    # "The fitted model puts a share of 0.0027 of the rows below 0.1 ... In the law, about 1.9%":
    # the fitted tail is nonzero and thinner than the law's.
    fitted_below = support.tail_mass[0.1]["below"]
    law = _law_moments()
    assert 0.0 < fitted_below < 0.5 * law["below_0.1"]
    assert 0.018 < law["below_0.1"] < 0.020

    # Step 9b: the default bound clips no row on this law, and the two tight bounds do.  The
    # fitted bound comes from 5 / (sqrt(n) log n), and a retarget there reproduces Step 6.
    curve = namespace["curve"]
    n = len(namespace["frame"])
    fitted_bound = 5.0 / (np.sqrt(n) * np.log(n))
    nearest = curve.loc[(curve["bound"] - fitted_bound).abs().idxmin()]
    assert abs(nearest["bound"] - fitted_bound) < 1e-12
    assert abs(nearest["psi"] - namespace["estimate"].psi) < 1e-10
    loose = curve[curve["bound"] <= 0.05 + 1e-12]
    assert len(loose) >= 5
    assert (loose["truncated_fraction"] == 0.0).all()
    assert (loose["psi"] == namespace["estimate"].psi).all()
    tight = {
        round(float(row["bound"]), 4): row for _, row in curve.iterrows() if row["bound"] > 0.05
    }
    assert set(tight) == {0.1, 0.2}
    assert tight[0.1]["truncated_fraction"] > 0.0
    assert tight[0.2]["truncated_fraction"] > tight[0.1]["truncated_fraction"]
    # Nonzero witness: "the estimate moves to 0.1674" at the tightest bound.
    assert tight[0.2]["psi"] - namespace["estimate"].psi > 5e-4

    # Step 10: the default nu^2 estimator runs, so the robustness value carries its limit.
    result = namespace["result"]
    robustness = namespace["robustness"]
    assert "nu2_estimator" not in robustness
    assert 0.0 < robustness["rva"] < robustness["rv"] < 1.0
    elements = result.sensitivity.elements(estimand="ate")
    assert elements.nu2_estimator == "doubly_robust"
    # "below the law's 4.83": a positive estimate under the finite population value. The
    # shortfall is the representer's squared error; "slightly" fails loudly above 10%.
    assert 0.0 < elements.nu2 < law["nu2"]
    assert elements.nu2 > 0.9 * law["nu2"]
    assert abs(law["nu2"] - 4.83) < 0.02
    # "equal strengths ... would move the point estimate to zero", and the 95% limit at rva.
    at_rv = result.sensitivity.omitted_confounding(cf_y=robustness["rv"], cf_d=robustness["rv"])
    assert abs(at_rv.lower) < 1e-3
    at_rva = result.sensitivity.omitted_confounding(cf_y=robustness["rva"], cf_d=robustness["rva"])
    assert abs(at_rva.ci_lower) < 1e-3
    # "The residual outcome variance ... more than doubles", so the implied cf_y is a clip.
    # "A bound needs a cf_y below 1" is the bound formula's own requirement.
    strong = namespace["strong"]
    assert strong.covariates == ("discharge_risk",)
    assert strong.sigma2_short > 2.0 * strong.sigma2_long
    assert strong.cf_y == 1.0
    with pytest.raises(ValueError, match=r"cf_y must lie in \[0, 1\)"):
        result.sensitivity.omitted_confounding(cf_y=strong.cf_y, cf_d=strong.cf_d, rho=1.0)
    # "Both lie below the robustness value", and the bounds stay worst-case at rho = 1.
    benchmark = namespace["benchmark"]
    assert benchmark.covariates == ("medication_burden",)
    assert 0.0 < benchmark.cf_d < benchmark.cf_y < robustness["rv"]
    bounds = namespace["bounds"]
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (benchmark.cf_y, benchmark.cf_d, 1.0)
    assert bounds.confounding_strength < at_rv.confounding_strength
    # "the sign of the effect survives": both one-sided limits exclude zero on this draw.
    assert 0.0 < bounds.ci_lower < bounds.lower < bounds.psi < bounds.upper < bounds.ci_upper
    summary = bounds.summary()
    assert "the sign of the effect survives" in summary
    assert "bias scale" in summary
    # "the bias scale is 0.27859, and the strengths limit the bias to 0.023307": the factor is
    # below 1 here, and the module's own example shows it can exceed 1.
    assert 0.0 < bounds.bias < bounds.max_bias
