"""The reviewed semantic callback for ``docs/examples/cross-fitting``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import navigation_protocol
from cleverly.datasets.synthetic import nonlinear_bounded_dgp
from cleverly.inference import median_estimates
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "cross-fitting.ipynb"

_PROBE = "reviews/notebook-review/probes/cross-fitting-final"
_STUDY = "tests/canonical/tmle3_cvtmle/properties.csv"
_CLUSTERED = "tests/canonical/lmtp_clustered_tmle/properties.csv"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.93": f"cross-fitted mean SE over empirical SD, 200 draws: {_PROBE}/summary.log",
    "0.70": f"in-sample mean SE over empirical SD, 200 draws: {_PROBE}/summary.log",
    "-0.0024": f"in-sample mean estimate minus the truth, 200 draws: {_PROBE}/summary.log",
    "0.0003": f"cross-fitted mean estimate minus the truth, 200 draws: {_PROBE}/summary.log",
    "0.68": f"5% quantile of the in-sample/cross-fitted SE ratio: {_PROBE}/summary.log",
    "0.90": f"5% quantile of the propensity calibration slope: {_PROBE}/summary.log",
    "1.01": f"95% quantile of the propensity calibration slope: {_PROBE}/summary.log",
    "1.9": f"share (0.0188) of the law's true propensity below 0.1: {_PROBE}/truth.log",
    "4.83": f"the law's nu^2 = E[1/(g(1-g))], 4.8287 by Monte Carlo: {_PROBE}/truth.log",
    "4.50": f"mean doubly robust nu^2 over 200 draws, 4.504: {_PROBE}/summary.log",
    "0.409": f"RV of the shown draw at the law's nu^2, by summarize.py: {_PROBE}/summary.log",
    "0.945": f"coverage of crossfit_overfitting/stacked_cvtmle: {_STUDY}",
    "0.5225": f"coverage of crossfit_overfitting/in_sample_control: {_STUDY}",
    "0.9475": f"coverage of clustered_inference/cluster_robust: {_CLUSTERED}",
    "0.84125": f"coverage of clustered_inference/iid_control: {_CLUSTERED}",
}


def _law_nu2(n: int = 400_000) -> float:
    """``nu^2 = E[1 / {g (1 - g)}]`` of the law behind ``navigation_data``, by Monte Carlo."""
    law = nonlinear_bounded_dgp()
    g = law.propensity(np.random.default_rng(1).standard_normal((n, law.n_latent)))
    return float(np.mean(1.0 / (g * (1.0 - g))))


def _teams_spanned(result: Any, teams: pd.Series) -> int:
    """The largest number of outer folds any one team spans in ``result``'s split plan."""
    labels = pd.Series(result.split_plan.assignments[0], index=teams.index)
    return int(labels.groupby(teams).nunique().max())


def _assert_unprinted_sources() -> None:
    """Each repeated-draw and study number the prose quotes equals its committed source."""
    summary = (ROOT / _PROBE / "summary.log").read_text()
    truth = (ROOT / _PROBE / "truth.log").read_text()
    fits = summary[summary.index("  cross-fitted\n") : summary.index("Step 9,")]
    cross_fitted, in_sample = fits.split("  in-sample\n")[:2]
    # Steps 5 and 6: SE/SD 0.93 and 0.70, bias +0.0003 and -0.0024, coverage 188 and 153 of 200,
    # and the in-sample/cross-fitted SE ratio between 0.68 and 0.73 on 90% of draws.
    assert "bias +0.0003 " in cross_fitted and "SE/SD 0.93" in cross_fitted
    assert "188 of 200" in cross_fitted
    assert "bias -0.0024 " in in_sample and "SE/SD 0.70" in in_sample
    assert "153 of 200" in in_sample
    assert "in-sample SE / cross-fitted SE, 5% / 50% / 95%: [0.68, 0.71, 0.73]" in summary
    # Step 9: slope 0.90 to 1.01; fitted tail share under the law's on 196 draws; in-sample
    # minimum ESS ratio above the cross-fitted one on all 200.
    assert "propensity calibration slope, 5% / 50% / 95%: [0.9, 0.96, 1.01]" in summary
    assert re.search(r"under the law's share, 0\.0188 .*196 of 200", summary)
    assert re.search(
        r"in-sample minimum ESS ratio above the cross-fitted one: .*200 of 200", summary
    )
    # Step 10: mean nu^2 4.50, the law's 4.83, and the shown draw's RV 0.409 at the law's nu^2.
    assert "nu^2 (doubly robust) mean 4.504" in summary
    assert re.search(r"law +nu\^2 4\.8287: .*rv 0\.409", summary)
    # The law's quantities: share below 0.1 is 1.9%, nu^2 is 4.83, 46% of teams lose.
    assert re.search(r"navigation nu2 +4\.828", truth)
    assert re.search(r"navigation share g < 0\.10 +0\.0188", truth)
    assert "P(u < -0.1000): 0.4602" in truth
    # The registered cells and their coverage.
    for path, cell, coverage, n, replicates in (
        (_STUDY, "in_sample_control", 0.5225, 500, 400),
        (_STUDY, "stacked_cvtmle", 0.945, 500, 400),
        (_CLUSTERED, "cluster_robust", 0.9475, 2000, 2400),
        (_CLUSTERED, "iid_control", 0.84125, 2000, 2400),
    ):
        table = pd.read_csv(ROOT / path, float_precision="round_trip")
        row = table.loc[table["cell"] == cell].iloc[0]
        assert np.isclose(row["coverage"], coverage, rtol=0.0, atol=1e-12)
        assert (row["n"], row["replicates"]) == (n, replicates)


def check(namespace: dict[str, Any]) -> None:
    """The seeded relations the cross-fitting notebook narrates, at its documented size."""
    _assert_unprinted_sources()
    first = namespace["cross_fitted"]
    cross_fitted = first["ate"]
    in_sample = namespace["in_sample"]["ate"]
    truth = namespace["truth"]["ate"]
    assert np.isclose(truth, 0.1629, atol=1e-4)

    # The data come from the shared helper under the program's names.
    assert list(namespace["frame"].columns) == [
        "transition_score",
        "transition_navigation",
        "discharge_risk",
        "prior_utilization",
        "medication_burden",
        "age",
    ]

    # The protocol is the program's, unchanged, so this page and the point-treatment page print
    # one fingerprint.
    assert namespace["protocol"] == navigation_protocol()
    fingerprint = assert_protocol_recorded(NOTEBOOK, "protocol", namespace["protocol"], first)
    assert fingerprint in stored_output(EXAMPLES / "point-treatment-tmle.ipynb", "protocol")

    # Step 5: the construction line, and one draw whose interval contains the truth.
    assert (
        "stacked CV-TMLE (Levy): nuisances cross-fitted over 5 folds; targeting: pooled"
        in first.summary()
    )
    assert covers(cross_fitted, truth)

    # Step 6: "close" points; the in-sample SE is 0.73 times the cross-fitted one; both cover,
    # and the truth lies 1.9 in-sample and 1.0 cross-fitted standard errors away.  Each window
    # brackets the narrated decimal.  The coverage claims are the probe's, not this draw's.
    assert "in-sample nuisances" in namespace["in_sample"].summary()
    assert abs(in_sample.psi - cross_fitted.psi) < 0.01
    assert 0.725 <= in_sample.std_error / cross_fitted.std_error < 0.735
    assert covers(in_sample, truth)
    assert 1.85 <= abs(truth - in_sample.psi) / in_sample.std_error < 1.95
    assert 0.95 <= abs(truth - cross_fitted.psi) / cross_fitted.std_error < 1.05

    # Step 7: points that "differ only because the team declaration regroups the folds"; "1.54
    # times larger".  The law is the binary clustered one: its outcome support is known, so the
    # team fits cross-fit without a declared support and the fold count has a split to read.  The team
    # protocol changes only the outcome field, and the cell prints its fingerprint.
    assert changed_fields(namespace["team_protocol"], namespace["protocol"]) == {"outcome"}
    assert_protocol_recorded(
        NOTEBOOK,
        "team-clusters",
        namespace["team_protocol"],
        namespace["ignoring"],
        namespace["clustered"],
    )
    ignoring = namespace["ignoring"]
    clustered = namespace["clustered"]
    team_truth = namespace["team_truth"]["ate"]
    assert np.isclose(team_truth, 0.104, atol=1e-3)
    assert abs(ignoring["ate"].psi - clustered["ate"].psi) < 0.005
    assert ignoring["ate"].psi != clustered["ate"].psi
    assert ignoring.provenance.fold_fingerprint != clustered.provenance.fold_fingerprint
    assert 1.4 < clustered["ate"].std_error / ignoring["ate"].std_error < 1.7
    assert covers(ignoring["ate"], team_truth) and covers(clustered["ate"], team_truth)
    assert clustered.data.n_clusters == 200
    # "This fit has 200 teams, so it reports a normal-reference interval."
    assert clustered.inference_status == "influence_curve"

    # "Without the declaration, one team spans 5 outer folds ... With it, each team spans 1 fold."
    # The unclustered fit is the nonzero witness for the grouped split.
    teams = namespace["team_frame"]["navigator_team"]
    assert _teams_spanned(clustered, teams) == 1
    assert _teams_spanned(ignoring, teams) == 5
    assert list(namespace["team_comparison"]["folds one team spans"]) == [5, 1]

    # Step 9: no attention row; slope 0.9611 near 1; no unit truncated; the treated arm has the
    # smaller ratio (82.5%); the fitted share below 0.1 (0.0063) is under the law's 1.9%; the
    # in-sample minimum ratio (89.6%) exceeds the cross-fitted one.
    assessment = namespace["assessment"]
    assert tuple(assessment.attention) == ()
    assert assessment.validation["score_equations"].status.value == "passed"
    nuisance = assessment.report("nuisance_models")
    assert nuisance is namespace["nuisance"]
    assert assessment.report("score_equations") is namespace["scores"]
    assert 0.9 < nuisance["propensity"].metrics["calibration_slope"] < 1.1
    assert "nuisance fits look reasonable" in stored_output(NOTEBOOK, "assessment")
    support = namespace["support"]
    assert assessment.report("support") is support
    assert 0.0 < support.propensity_quantiles["overall"][0.0] < 0.05
    assert support.truncated["count"] == 0
    assert 0.0 < support.tail_mass[0.1]["below"] < 0.0188
    ess = support.effective_sample_size
    assert min(ess, key=lambda arm: ess[arm]["ratio"]) == "treated"
    assert 0.82 < ess["treated"]["ratio"] < 0.83
    in_sample_support = namespace["in_sample"].diagnostics.support()
    in_sample_ratios = [row["ratio"] for row in in_sample_support.effective_sample_size.values()]
    assert min(in_sample_ratios) > ess["treated"]["ratio"]

    # Step 8: reconstruct the median and its within-plus-between variance from the three retained
    # draws. This seed gives nonzero displacement even though the median happens to equal the
    # within-only median, so the controlled mutation below makes that term result-determining.
    # The diagnostic row must describe that same headline estimate.
    repeated = namespace["repeated"]
    assert len(repeated.to_frame()) == 1
    assert "median over 3 independent draws" in repeated.summary()
    draw_estimates = []
    for draw in repeated.repeats:
        estimates, _, _ = repeated.estimator._retarget_detailed(
            repeated.data,
            draw.nuisance,
            estimands=("ate",),
            g_bounds=repeated.config.g_bounds,
            g_bounds_conditional=repeated.config.g_bounds_conditional,
        )
        draw_estimates.append(estimates["ate"])
    points = np.asarray([estimate.psi for estimate in draw_estimates])
    centre = float(np.median(points))
    assert repeated["ate"].psi == centre
    displacements = (points - centre) ** 2
    assert np.count_nonzero(displacements) == 2
    expected_variance = float(
        np.median(
            [
                estimate.variance + displacement
                for estimate, displacement in zip(draw_estimates, displacements, strict=True)
            ]
        )
    )
    assert np.isclose(repeated["ate"].variance, expected_variance, rtol=1e-12, atol=0.0)
    mutation_parts = [
        {"ate": replace(draw_estimates[0], psi=draw_estimates[0].psi + shift)}
        for shift in (-0.1, 0.0, 0.2)
    ]
    mutation_witness = median_estimates(mutation_parts)["ate"]
    assert mutation_witness.psi == draw_estimates[0].psi
    assert np.isclose(
        mutation_witness.variance,
        draw_estimates[0].variance + 0.01,
        rtol=1e-12,
        atol=1e-15,
    )
    assert mutation_witness.variance != draw_estimates[0].variance
    spread = namespace["repeated_nuisance"].repeat_spread
    assert [row.n_repeats for row in spread] == [3]
    assert np.isclose(spread[0].standard_deviation, np.std(points, ddof=1))
    assert spread[0].reported_standard_error == repeated["ate"].std_error
    assert np.isclose(
        spread[0].ratio_to_standard_error,
        spread[0].standard_deviation / repeated["ate"].std_error,
    )
    # "Across these three draws, the estimate moved much less than its standard error."
    assert spread[0].standard_deviation < 0.1 * spread[0].reported_standard_error

    # Step 10: the default (doubly robust) nu^2 runs with no refusal; it is positive, below the
    # law's 4.83, and within 10% of it.  Then the robustness values and the default-strength
    # bounds.  The law's nu^2 is recomputed here, so a drift in the law fails loudly.
    elements = namespace["elements"]
    robustness = namespace["robustness"]
    bounds = namespace["bounds"]
    law_nu2 = _law_nu2()
    assert abs(law_nu2 - 4.83) < 0.01
    assert elements.nu2_estimator == "doubly_robust"
    assert 0.9 * law_nu2 < elements.nu2 < law_nu2
    assert robustness["rva"] < robustness["rv"]
    assert 0.40 < robustness["rv"] < 0.45
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (0.03, 0.03, 1.0)
    assert 0.0 < bounds.ci_lower < bounds.lower < cross_fitted.psi < bounds.upper < bounds.ci_upper
    # "the law's nu^2 gives a robustness value of 0.409 rather than 0.419": a larger nu^2 gives a
    # larger bias scale, so a smaller robustness value.  The formula is the one the fit used,
    # since it reproduces the printed value at the fitted nu^2 (the nonzero witness).
    for nu2, expected in ((elements.nu2, robustness["rv"]), (law_nu2, 0.409)):
        t2 = cross_fitted.psi**2 / (elements.sigma2 * nu2)
        assert abs((np.sqrt(t2**2 + 4.0 * t2) - t2) / 2.0 - expected) < 5e-4
    at_rv = first.sensitivity.omitted_confounding(
        cf_y=robustness["rv"], cf_d=robustness["rv"], rho=1.0
    )
    assert abs(at_rv.lower) < 1e-3
