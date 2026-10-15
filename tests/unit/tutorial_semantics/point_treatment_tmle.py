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
    "-0.018": f"mean error of the linear-Q plug-in over 200 draws: {_PROBE}/plugin.log",
    "-0.0022": f"mean error of TMLE with linear Q and the booster's g: {_PROBE}/summary.log",
    "-0.0006": f"mean error of TMLE with the booster's Q and logistic g: {_PROBE}/summary.log",
    "-0.0218": f"mean error of TMLE with both nuisances linear: {_PROBE}/summary.log",
    "0.05": "the law's propensity bounds: cleverly.datasets.synthetic.nonlinear_dgp",
    "0.95": "the law's propensity bounds: cleverly.datasets.synthetic.nonlinear_dgp",
    "0.431": f"RV of the shown draw at the law's nu^2, by summarize.py: {_PROBE}/summary.log",
    "0.025": "the study's propensity bounds: tests/studies/canonical_cvtmle.py G_BOUNDS",
    "0.975": "the study's propensity bounds: tests/studies/canonical_cvtmle.py G_BOUNDS",
    "0.945": f"coverage of crossfit_overfitting/stacked_cvtmle: {_STUDY}",
    "0.5225": f"coverage of crossfit_overfitting/in_sample_control: {_STUDY}",
}


def check(namespace: dict[str, Any]) -> None:
    """The ATE tutorial's graph, estimand, failure mode, diagnostics, and sensitivity claims hold."""
    summary = namespace["effect"].summary()
    assert "Discharge-home order" in summary
    assert "scores death before day 30 as the worst transition score" in summary
    assert "Analyze the offer regardless of completed contacts" in summary
    assert "stacked CV-TMLE" in namespace["result"].summary()
    assert list(namespace["frame"].columns[:2]) == ["transition_score", "transition_navigation"]

    # Step 2: "all four covariates enter g" and "enter Q".  Moving one covariate at a time
    # moves both functions, so each covariate is a confounder of the law.  A covariate that
    # entered neither would leave both unchanged.
    law = nonlinear_bounded_dgp()
    base = np.random.default_rng(2).standard_normal((2_000, law.n_latent))
    for column in range(law.n_latent):
        moved = base.copy()
        moved[:, column] += 0.5
        assert np.abs(law.propensity(moved) - law.propensity(base)).max() > 1e-3, column
        assert (
            np.abs(law.outcome_mean(moved, 0.0, None) - law.outcome_mean(base, 0.0, None)).max()
            > 1e-3
        ), column
    # "a true propensity between 0.05 and 0.95".
    wide = law.propensity(np.random.default_rng(3).standard_normal((200_000, law.n_latent)) * 3.0)
    assert wide.min() >= 0.05
    assert wide.max() <= 0.95

    # Step 3: "The offered patients score ... higher", and the arms differ on discharge risk first.
    by_arm = namespace["by_arm"]
    truth = namespace["truth"]
    unadjusted = namespace["unadjusted"]
    assert unadjusted > truth["ate"]
    assert by_arm.loc[0.0, "discharge_risk"] < 0.0 < by_arm.loc[1.0, "discharge_risk"]
    assert by_arm.loc[1.0, "discharge_risk"] > by_arm.loc[0.0, "discharge_risk"] + 0.3
    # "Two mechanisms of this law raise the unadjusted difference": the ATT exceeds the ATE,
    # and the offered patients would score higher under usual support alone.  Both have the
    # same sign in this law.  The margins are the retired Gaussian law's, scaled by the ratio
    # of the two ATEs (0.1629 against 1.750).
    assert truth["att"] > truth["ate"] + 0.01
    assert truth["att"] > truth["ate"] > truth["atc"]
    assert _untreated_selection_bias() > 0.01

    # Step 4: the protocol step prints the record, and the fit carries its digest.
    assert namespace["protocol"] == navigation_protocol()
    assert_protocol_recorded(NOTEBOOK, "protocol", namespace["protocol"], namespace["result"])

    # Step 6: the headline interval covers the truth on this draw.  The repeated-draw numbers
    # the reading quotes (189 of 200, SE/SD 0.98) come from the committed probe.  The declared
    # q_bounds are the score's own support: the scaler is the identity, and the outcome range
    # is the nonzero witness that the support holds the data rather than being padded around it.
    ate = truth["ate"]
    assert covers(namespace["estimate"], ate)
    scaler = namespace["result"].nuisance.scaler
    assert (scaler.lower, scaler.upper) == (0.0, 1.0)
    assert scaler.is_identity
    outcome = np.asarray(namespace["frame"]["transition_score"], dtype=float)
    assert scaler.lower < outcome.min() < outcome.max() < scaler.upper

    # Step 7: "that average equals the regression coefficient on the offer".
    linear_q = namespace["linear_q"]
    plug_in = namespace["plug_in"]
    assert linear_q.feature_names_in_[0] == "transition_navigation"
    assert abs(plug_in - float(linear_q.coef_[0])) < 1e-10
    # The plug-in misses by more than TMLE with the same Q class and the booster's g, and that
    # fit and the flexible-Q fit cover the truth.  The both-linear fit misses by more than the
    # plug-in and its interval excludes the truth.  The probes give the repeated-draw numbers.
    errors = {label: abs(point.psi - ate) for label, point in namespace["dr_points"].items()}
    assert abs(plug_in - ate) > 3.0 * errors["linear Q, flexible g"]
    one_flexible = max(errors["flexible Q, linear g"], errors["linear Q, flexible g"])
    assert errors["both linear"] > one_flexible
    assert errors["both linear"] > abs(plug_in - ate)
    for label in ("flexible Q, linear g", "linear Q, flexible g"):
        assert covers(namespace["dr_points"][label], ate), label
    assert not covers(namespace["dr_points"]["both linear"], ate)

    # Step 8: no row needs attention, the score-equation check passed, and the shallow booster's
    # calibration slope is near 1.  The verdict the reading quotes is the report's own.
    assessment = namespace["assessment"]
    assert tuple(item.name for item in assessment.attention) == ()
    ledger = assessment.to_frame()
    assert set(ledger.loc[ledger["check"] == "score_equations", "status"]) == {"passed"}
    assert "nuisance fits look reasonable" in namespace["nuisance"].summary()
    (propensity,) = (m for m in namespace["nuisance"].models if m.name == "propensity")
    assert abs(propensity.metrics["calibration_slope"] - 1.0) < 0.1
    # "no unit sits at the truncation bound".
    support = namespace["support"]
    assert support.truncated["fraction"] == 0.0
    assert support.truncated["most_extreme"] > 0.05
    # "It puts a share of 0.0027 of the rows below 0.1 ... In the law, about 1.9%": the fitted
    # tail is nonzero and thinner than the law's.
    fitted_below = support.tail_mass[0.1]["below"]
    moments = _law_moments()
    assert 0.0 < fitted_below < 0.5 * moments["below_0.1"]
    assert 0.018 < moments["below_0.1"] < 0.020

    # Step 9: the default bound clips no row on this law, and the two tight bounds do.  The
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
    # "the law's value is about 4.83 ... The bounds are then slightly narrow": a positive
    # estimate under the finite population value. "slightly" fails loudly above 10%.
    assert 0.0 < elements.nu2 < moments["nu2"]
    assert elements.nu2 > 0.9 * moments["nu2"]
    assert abs(moments["nu2"] - 4.83) < 0.02
    # "equal strengths ... would move the point estimate to zero", and the 95% limit at rva.
    at_rv = result.sensitivity.omitted_confounding(cf_y=robustness["rv"], cf_d=robustness["rv"])
    assert abs(at_rv.lower) < 1e-3
    at_rva = result.sensitivity.omitted_confounding(cf_y=robustness["rva"], cf_d=robustness["rva"])
    assert abs(at_rva.ci_lower) < 1e-3
    # "its gain exceeds 1. The benchmark clips the gain to 1.0000, and the bound accepts no
    # `cf_y` of 1": the residual variance more than doubles, and the bound refuses the clip.
    strong = namespace["strong"]
    assert strong.covariates == ("discharge_risk",)
    assert strong.sigma2_short > 2.0 * strong.sigma2_long
    assert strong.cf_y == 1.0
    with pytest.raises(ValueError, match=r"cf_y must lie in \[0, 1\)"):
        result.sensitivity.omitted_confounding(cf_y=strong.cf_y, cf_d=strong.cf_d, rho=1.0)
    # "Both lie below the robustness value", and the gains go to the bound unchanged, at rho = 1.
    benchmark = namespace["benchmark"]
    assert benchmark.covariates == ("medication_burden",)
    assert 0.0 < benchmark.cf_d < benchmark.cf_y < robustness["rv"]
    bounds = namespace["bounds"]
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (benchmark.cf_y, benchmark.cf_d, 1.0)
    assert bounds.confounding_strength < at_rv.confounding_strength
    # "Both limits exclude zero on this draw".
    assert 0.0 < bounds.ci_lower < bounds.lower < bounds.psi < bounds.upper < bounds.ci_upper
    assert "the sign of the effect survives" in bounds.summary()
    # "the bias bound would be about 5% larger" with the law's nu^2: the bias scale is
    # proportional to sqrt(nu^2).
    assert 1.03 < np.sqrt(moments["nu2"] / elements.nu2) < 1.07
