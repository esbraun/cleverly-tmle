"""The reviewed semantic callback for ``docs/examples/cross-fitting``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import EXAMPLES, stored_output

NOTEBOOK = EXAMPLES / "cross-fitting.ipynb"

UNPRINTED_DECIMALS = {
    "65.0": "the registered stacked CV-TMLE study's in-sample control coverage, printed on its "
    "linked evidence page",
    "89.5": "the registered stacked CV-TMLE study's cross-fitted coverage, printed on its linked "
    "evidence page",
}


def _covers(point: Any, target: float) -> bool:
    low, high = point.ci
    return bool(low <= target <= high)


def check(namespace: dict[str, Any]) -> None:
    """The seeded relations the cross-fitting notebook narrates, at its documented size."""
    cross_fitted = namespace["cross_fitted"]["ate"]
    in_sample = namespace["in_sample"]["ate"]
    truth = namespace["truth"]["ate"]

    # The protocol is the program's, unchanged, so this page and the point-treatment page print
    # one fingerprint.
    assert namespace["protocol"] == navigation_protocol()
    fingerprint = namespace["protocol"].fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "protocol")
    assert fingerprint in stored_output(EXAMPLES / "point-treatment-tmle.ipynb", "protocol")
    assert namespace["cross_fitted"].provenance.protocol_fingerprint == fingerprint

    assert "stacked CV-TMLE" in namespace["cross_fitted"].summary()
    assert "in-sample nuisances" in namespace["in_sample"].summary()
    # "the two point estimates are close"; "less than a third" (measured ratio 0.28).
    assert abs(in_sample.psi - cross_fitted.psi) < 0.1
    assert in_sample.std_error < cross_fitted.std_error / 3
    # The comparison table: the in-sample interval excludes the truth, the cross-fitted one does not.
    assert not _covers(in_sample, truth)
    assert _covers(cross_fitted, truth)

    # "nearly identical" points and "1.68 times larger" standard error; both cover on this draw.
    ignoring = namespace["ignoring"]["ate"]
    clustered = namespace["clustered"]["ate"]
    team_truth = namespace["team_truth"]["ate"]
    assert abs(ignoring.psi - clustered.psi) < 0.02
    assert 1.5 < clustered.std_error / ignoring.std_error < 1.9
    assert _covers(ignoring, team_truth) and _covers(clustered, team_truth)
    assert namespace["clustered"].data.n_clusters == 200

    # "Each team spans at most 1 outer fold"; four teams realize four of five requested folds.
    assert int(namespace["folds_per_team"].max()) == 1
    few = namespace["few"]
    assert few.data.n_clusters == 4
    assert few.split_plan.n_folds == 4
    assert any("only 4 clusters" in str(warning.message) for warning in namespace["caught"])

    # A new seed draws new folds; the plan reproduces the folds, the point, and the curve.
    redrawn = namespace["redrawn"]
    reused = namespace["reused"]
    first = namespace["cross_fitted"]
    assert redrawn.provenance.fold_fingerprint != first.provenance.fold_fingerprint
    assert redrawn["ate"].psi != cross_fitted.psi
    assert reused.provenance.fold_fingerprint == first.provenance.fold_fingerprint
    assert reused["ate"].psi == cross_fitted.psi
    assert np.array_equal(reused["ate"].influence_curve, cross_fitted.influence_curve)
    assert "refused on other data" in stored_output(NOTEBOOK, "reuse-split")

    # "agrees with the stacked one to four decimals", "endpoints differ in the fourth decimal".
    fits = namespace["construction_fits"]
    evaluated = fits["fold-evaluated"]["ate"]
    assert "fold-evaluated CV-TMLE" in fits["fold-evaluated"].summary()
    assert "fold-specific targeted TMLE" in fits["fold-targeted"].summary()
    assert abs(evaluated.psi - cross_fitted.psi) < 5e-5
    assert tuple(evaluated.ci) != tuple(cross_fitted.ci)
    assert abs(fits["fold-targeted"]["ate"].psi - cross_fitted.psi) > 0.01

    # "warns that the out-of-fold propensity model is poorly calibrated" (slope 0.49);
    # "1.00%" truncated; "25.2%"; in-sample "91.6%".
    diagnostics = namespace["diagnostics"]
    assert diagnostics["nuisance_models"].status.value == "warning"
    assert diagnostics["score_equations"].status.value == "passed"
    nuisance = diagnostics.report("nuisance_models")
    assert nuisance["propensity"].metrics["calibration_slope"] < 0.6
    support = diagnostics.report("support")
    assert 0.005 < support.truncated["fraction"] < 0.02
    ratios = [ess["ratio"] for ess in support.effective_sample_size.values()]
    assert 0.2 < min(ratios) < 0.3
    in_sample_support = namespace["in_sample"].diagnostics.support()
    in_sample_ratios = [ess["ratio"] for ess in in_sample_support.effective_sample_size.values()]
    assert min(in_sample_ratios) > 0.85

    spread = namespace["repeated_nuisance"].repeat_spread
    assert [row.n_repeats for row in spread] == [3]
