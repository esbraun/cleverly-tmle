"""The reviewed semantic callback for ``docs/examples/longitudinal-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.tutorial_semantics import EXAMPLES, stored_output

NOTEBOOK = EXAMPLES / "longitudinal-tmle.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """The two-decision tutorial keeps its named roles and its seeded narrative."""
    frame = namespace["frame"]
    # The page renames L2 by the signs of the law. Engagement must rise with discharge
    # navigation and with day-seven navigation among tracked patients, or the name is wrong.
    tracked = frame[frame["tracked_day7"] == 1]
    by_first = tracked.groupby("navigation_discharge")["engagement_day7"].mean()
    assert by_first[1.0] - by_first[0.0] > 0.5
    by_second = tracked.groupby("navigation_day7")["engagement_day7"].mean()
    assert by_second[1.0] > by_second[0.0]
    complete = tracked.dropna(subset=["navigation_day7", "transition_top_box"])
    for _, group in complete.groupby(["navigation_discharge", "navigation_day7"]):
        high = group["engagement_day7"] > group["engagement_day7"].median()
        assert (
            group.loc[high, "transition_top_box"].mean()
            > group.loc[~high, "transition_top_box"].mean()
        )
    # "Among engaged patients, a share of ... received day-seven navigation" exceeds the others.
    shares = tracked.groupby(tracked["engagement_day7"] > 0)["navigation_day7"].mean()
    assert shares[True] > shares[False]
    # "so 1476 patients have no outcome": the subset's tracking loss is exactly that set.
    lost = len(frame) - len(namespace["observed"])
    assert lost == int(frame["transition_top_box"].isna().sum())
    assert lost > int((frame["tracked_day7"] == 0).sum())

    summary = namespace["effect"].summary()
    assert "sequential exchangeability" in summary and "sequential positivity" in summary
    assert "The protocol scores death before day 30 as not top box" in summary

    # The protocol step prints the record, and both fits carry its digest.
    fingerprint = namespace["protocol"].fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "protocol")
    assert namespace["result"].provenance.protocol_fingerprint == fingerprint
    assert namespace["rule_result"].provenance.protocol_fingerprint == fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "rule")

    # The top-box shares with no navigation and with both offers are quadrature truths.
    truth = namespace["truth"]
    assert truth["ey_regimen[never]"] == pytest.approx(0.42, abs=0.005)
    assert truth["ey_regimen[always]"] == pytest.approx(0.78, abs=0.005)

    target = namespace["target"]
    assert namespace["crude"] > target
    adjusted, baseline_only = namespace["adjusted"], namespace["baseline_only"]
    sequential = namespace["result"]["ate_regimen[always vs never]"]
    # "the interval ... contains the true contrast" on this draw.
    assert sequential.ci[0] <= target <= sequential.ci[1]
    # "0.137 below", "0.051 above", and "within one standard error" (measured 0.43 SE).
    assert adjusted.psi < target - 0.1
    assert 0.03 < baseline_only.psi - target < 0.08
    assert abs(sequential.psi - target) < sequential.std_error

    rule = namespace["rule_result"]
    assert set(rule.estimates) == {"ate_regimen[continue if engaged vs never]"}
    rule_point = rule["ate_regimen[continue if engaged vs never]"]
    rule_truth = truth["ate_regimen[treat then continue if l2 positive vs never]"]
    assert rule_point.ci[0] <= rule_truth <= rule_point.ci[1]
    assigned = rule.diagnostics.support().to_frame().set_index(["regimen", "time"])
    assert assigned.loc[("continue if engaged", 1), "share_assigned_1"] == 1.0
    assert 0.0 < assigned.loc[("continue if engaged", 2), "share_assigned_1"] < 1.0

    assessment = namespace["assessment"]
    sensitivity = assessment.sensitivity.items
    assert sensitivity and all(item.status.value == "unavailable" for item in sensitivity)
    assert assessment.diagnostics["refute"].status.value == "unavailable"
    assert assessment.diagnostics["corrections"].status.value == "not_applicable"
    refused = {item.name for item in namespace["effect"].available_methods() if not item.available}
    assert {"collaborative_tmle", "drtmle"} <= refused
    support = namespace["support"]
    # "the bound replaces no row": the largest weight (measured 33.3) is well under the cap of 100.
    assert (support["share_truncated"] == 0.0).all()
    assert (support["max_weight"] < 60.0).all()
    assert (support["effective_n"] / support["n_followed"]).min() > 0.75
    curve = namespace["curve"]
    assert list(curve["lower_bound"]) == [0.01, 0.05, 0.1]
    assert curve["truncated_score_cells"].iloc[0] == 0
    assert curve["truncated_score_cells"].iloc[-1] > 0
    # "0.0019, which is 0.11 standard errors".
    movement = float(curve["delta_from_fitted"].abs().max())
    assert 0.0 < movement < 0.005
    assert movement < 0.3 * sequential.std_error

    # "Every score row passed": solver rows at solver precision, stitching rows near zero in SE.
    scores = namespace["scores"]
    assert scores["passed"].all()
    solver = scores[scores["kind"] == "solver"]
    stitching = scores[scores["kind"] == "stitching"]
    assert len(solver) == len(stitching) == 4
    assert (solver["relative_score"] < 1e-10).all()
    assert stitching["z"].abs().max() < 1.0
    # "The lowest calibration slope is ... for the censoring model at node 2", below the ideal 1.
    nuisance = namespace["nuisance"]
    lowest = nuisance.loc[nuisance["calibration_slope"].idxmin()]
    assert (lowest["role"], lowest["time"]) == ("censoring", 2)
    assert lowest["calibration_slope"] < 1.0
    mechanisms = nuisance[nuisance["role"].isin(["treatment", "censoring"])]
    assert mechanisms["regimen"].isna().all()
