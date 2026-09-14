"""The reviewed semantic callback for ``docs/examples/cross-fitting``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleverly import DataError
from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "cross-fitting.ipynb"

UNPRINTED_DECIMALS = {
    "65.0": "the registered stacked CV-TMLE study's in-sample control coverage, 0.65 in "
    "tests/canonical/tmle3_cvtmle/properties.csv and 0.6500 on its linked evidence page",
    "89.5": "the registered stacked CV-TMLE study's cross-fitted coverage, 0.895 in "
    "tests/canonical/tmle3_cvtmle/properties.csv and 0.8950 on its linked evidence page",
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

    # Step 6: "close" points, "less than a third" (0.28), and a miss of 2.6 standard errors.
    assert "in-sample nuisances" in namespace["in_sample"].summary()
    assert abs(in_sample.psi - cross_fitted.psi) < 0.1
    assert in_sample.std_error < cross_fitted.std_error / 3
    assert not covers(in_sample, truth)
    assert 2.5 < abs(truth - in_sample.psi) / in_sample.std_error < 2.7

    # Step 7: "nearly identical" points that differ because the folds regroup; "1.68 times larger".
    ignoring = namespace["ignoring"]
    clustered = namespace["clustered"]
    team_truth = namespace["team_truth"]["ate"]
    assert abs(ignoring["ate"].psi - clustered["ate"].psi) < 0.02
    assert ignoring["ate"].psi != clustered["ate"].psi
    assert ignoring.provenance.fold_fingerprint != clustered.provenance.fold_fingerprint
    assert 1.5 < clustered["ate"].std_error / ignoring["ate"].std_error < 1.9
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
    assert abs(targeted.psi - cross_fitted.psi) > 0.01
    assert abs(targeted.std_error - cross_fitted.std_error) > 0.01

    # Step 11: "one warning, nuisance_models"; slope 0.49 below 1; small g(W); 30 units (1.00%)
    # truncated; the treated arm has the 25.2% ratio; the in-sample ratio is 91.6%.
    diagnostics = namespace["diagnostics"]
    warned = [item.name for item in diagnostics.items if item.status.value == "warning"]
    assert warned == ["nuisance_models"]
    assert diagnostics["score_equations"].status.value == "passed"
    nuisance = diagnostics.report("nuisance_models")
    assert nuisance["propensity"].metrics["calibration_slope"] < 0.6
    support = namespace["support"]
    assert support.propensity_quantiles["overall"][0.0] < 0.01
    assert support.truncated["count"] == 30
    ess = support.effective_sample_size
    assert min(ess, key=lambda arm: ess[arm]["ratio"]) == "treated"
    assert 0.2 < ess["treated"]["ratio"] < 0.3
    in_sample_support = namespace["in_sample"].diagnostics.support()
    in_sample_ratios = [row["ratio"] for row in in_sample_support.effective_sample_size.values()]
    assert min(in_sample_ratios) > 0.85

    # Step 12: one estimate, so no band applies; the median over 3 draws; its standard error is
    # below Step 5's, as the Step 9 redraw's is; the redraw moved the point by about five spreads.
    repeated = namespace["repeated"]
    assert len(repeated.to_frame()) == 1
    assert "median over 3 independent draws" in repeated.summary()
    assert repeated["ate"].std_error < cross_fitted.std_error
    assert redrawn["ate"].std_error < cross_fitted.std_error
    spread = namespace["repeated_nuisance"].repeat_spread
    assert [row.n_repeats for row in spread] == [3]
    assert spread[0].standard_deviation < spread[0].reported_standard_error
    assert 4 < abs(redrawn["ate"].psi - cross_fitted.psi) / spread[0].standard_deviation < 6
