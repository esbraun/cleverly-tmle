"""The reviewed semantic callback for ``docs/examples/cross-fitting``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from cleverly import DataError
from cleverly.datasets import navigation_protocol
from cleverly.inference import median_estimates
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "cross-fitting.ipynb"

UNPRINTED_DECIMALS = {
    "48.8": "the registered stacked CV-TMLE study's in-sample control coverage, 0.4875 in "
    "tests/canonical/tmle3_cvtmle/properties.csv and 0.4875 on its linked evidence page",
    "93.3": "the registered stacked CV-TMLE study's cross-fitted coverage, 0.9325 in "
    "tests/canonical/tmle3_cvtmle/properties.csv and 0.9325 on its linked evidence page",
}


def _teams_spanned(result: Any, teams: pd.Series) -> int:
    """The largest number of outer folds any one team spans in ``result``'s split plan."""
    labels = pd.Series(result.split_plan.assignments[0], index=teams.index)
    return int(labels.groupby(teams).nunique().max())


def check(namespace: dict[str, Any]) -> None:
    """The seeded relations the cross-fitting notebook narrates, at its documented size."""
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

    # Step 6: "close" points, "less than a third" (0.27), and a miss of 2.5 standard errors.
    # The point-distance margin is the retired Gaussian law's, scaled by the ratio of the two
    # ATEs (0.1629 against 1.750); the miss window brackets the narrated single decimal.
    assert "in-sample nuisances" in namespace["in_sample"].summary()
    assert abs(in_sample.psi - cross_fitted.psi) < 0.01
    assert in_sample.std_error < cross_fitted.std_error / 3
    assert not covers(in_sample, truth)
    assert 2.45 <= abs(truth - in_sample.psi) / in_sample.std_error < 2.55

    # Step 7: "nearly identical" points that differ because the folds regroup; "1.54 times larger".
    # The law is the binary clustered one: a binary outcome needs no declared support, so the
    # team fits keep cross-fitting and Step 8 still has a split to read.
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

    # Step 8: "each team spans at most 1 outer fold". The unclustered fit is the nonzero witness:
    # without the declaration a team does span several folds.
    teams = namespace["team_frame"]["navigator_team"]
    assert int(namespace["folds_per_team"].max()) == 1
    assert _teams_spanned(clustered, teams) == 1
    assert _teams_spanned(ignoring, teams) > 1
    few = namespace["few"]
    assert few.data.n_clusters == 4
    assert few.split_plan.n_folds == 4
    assert any("only 4 clusters" in str(warning.message) for warning in namespace["caught"])

    # Step 9: a new seed draws new folds; the plan reproduces the folds, the point, and the curve;
    # the plan is refused, live, on data with another fingerprint.
    redrawn = namespace["redrawn"]
    reused = namespace["reused"]
    plan = namespace["plan"]
    assert plan.fingerprint == first.provenance.fold_fingerprint
    assert redrawn.provenance.fold_fingerprint != first.provenance.fold_fingerprint
    assert redrawn["ate"].psi != cross_fitted.psi
    assert reused.provenance.fold_fingerprint == first.provenance.fold_fingerprint
    assert reused["ate"].psi == cross_fitted.psi
    assert np.array_equal(reused["ate"].influence_curve, cross_fitted.influence_curve)
    refusal = namespace["refusal"]
    assert isinstance(refusal, DataError)
    assert first.provenance.data_fingerprint in str(refusal)

    # Step 10: equal-size folds make the fold-evaluated point the stacked point; only the variance
    # formula differs. The fold-targeted fit differs in both point and standard error.
    fits = namespace["construction_fits"]
    evaluated = fits["fold-evaluated"]["ate"]
    targeted = fits["fold-targeted"]["ate"]
    assert "fold-evaluated CV-TMLE" in fits["fold-evaluated"].summary()
    assert "fold-specific targeted TMLE" in fits["fold-targeted"].summary()
    assert set(np.unique(plan.assignments[0], return_counts=True)[1]) == {600}
    assert abs(evaluated.psi - cross_fitted.psi) < 1e-10
    assert tuple(evaluated.ci) != tuple(cross_fitted.ci)
    # "Its estimate moves in the fourth decimal, and its standard error is smaller."
    assert targeted.psi != cross_fitted.psi
    assert 1e-5 < abs(targeted.psi - cross_fitted.psi) < 1e-3
    assert targeted.std_error < cross_fitted.std_error - 1e-4

    # Step 11: "one warning, nuisance_models"; slope 0.49 below 1; small g(W); 16 units (0.53%)
    # truncated; the treated arm has the 22.9% ratio; the in-sample ratio is 91.6%.
    assessment = namespace["assessment"]
    warned = [item.name for item in assessment.attention if item.status.value == "warning"]
    assert warned == ["nuisance_models"]
    assert assessment.validation["score_equations"].status.value == "passed"
    nuisance = assessment.report("nuisance_models")
    assert nuisance is namespace["nuisance"]
    assert assessment.report("score_equations") is namespace["scores"]
    assert nuisance["propensity"].metrics["calibration_slope"] < 0.6
    support = namespace["support"]
    assert assessment.report("support") is support
    assert support.propensity_quantiles["overall"][0.0] < 0.01
    assert support.truncated["count"] == 16
    ess = support.effective_sample_size
    assert min(ess, key=lambda arm: ess[arm]["ratio"]) == "treated"
    assert 0.2 < ess["treated"]["ratio"] < 0.3
    in_sample_support = namespace["in_sample"].diagnostics.support()
    in_sample_ratios = [row["ratio"] for row in in_sample_support.effective_sample_size.values()]
    assert min(in_sample_ratios) > 0.85

    # Step 12: reconstruct the median and its within-plus-between variance from the three retained
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
    assert repeated["ate"].std_error < cross_fitted.std_error
    assert redrawn["ate"].std_error < cross_fitted.std_error
    spread = namespace["repeated_nuisance"].repeat_spread
    assert [row.n_repeats for row in spread] == [3]
    assert np.isclose(spread[0].standard_deviation, np.std(points, ddof=1))
    assert spread[0].reported_standard_error == repeated["ate"].std_error
    assert np.isclose(
        spread[0].ratio_to_standard_error,
        spread[0].standard_deviation / repeated["ate"].std_error,
    )
    assert spread[0].standard_deviation < spread[0].reported_standard_error
    # "about a third of this standard deviation" between the Step 5 and Step 9 fold draws.
    assert 0.2 < abs(redrawn["ate"].psi - cross_fitted.psi) / spread[0].standard_deviation < 0.4

    # Step 13: the live assessment retains the ordinary point-treatment sensitivity calculation.
    robustness = namespace["robustness"]
    confounding = namespace["confounding"]
    assert 0.15 < robustness["rv"] < 0.35
    assert (confounding.cf_y, confounding.cf_d, confounding.rho) == (0.03, 0.03, 1.0)
    assert confounding.lower < cross_fitted.psi < confounding.upper
    assert confounding.ci_lower < confounding.lower
    assert confounding.ci_upper > confounding.upper
    at_rv = first.sensitivity.omitted_confounding(
        cf_y=robustness["rv"], cf_d=robustness["rv"], rho=1.0
    )
    assert abs(at_rv.lower) < 1e-3
