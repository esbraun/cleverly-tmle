"""The reviewed semantic callback for ``docs/examples/longitudinal-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

from typing import Any

import pytest


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

    summary = namespace["effect"].summary()
    assert "sequential exchangeability" in summary and "sequential positivity" in summary
    assert "The protocol scores death before day 30 as not top box" in summary

    # "42% with no navigation and 78% with both offers" are quadrature truths, not draws.
    truth = namespace["truth"]
    assert truth["ey_regimen[never]"] == pytest.approx(0.42, abs=0.005)
    assert truth["ey_regimen[always]"] == pytest.approx(0.78, abs=0.005)

    target = namespace["target"]
    adjusted, baseline_only = namespace["adjusted"], namespace["baseline_only"]
    sequential = namespace["result"]["ate_regimen[always vs never]"]
    # "more than 0.1 below" (measured 0.137), "about 0.05 above" (measured 0.051), and
    # "within one standard error" (measured 0.43 standard errors).
    assert adjusted.psi < target - 0.1
    assert 0.03 < baseline_only.psi - target < 0.08
    assert abs(sequential.psi - target) < sequential.std_error

    rule = namespace["rule_result"]
    assert set(rule.estimates) == {"ate_regimen[continue if engaged vs never]"}
    shares = rule.diagnostics.support().to_frame().set_index(["regimen", "time"])
    assert shares.loc[("continue if engaged", 1), "share_assigned_1"] == 1.0
    assert 0.0 < shares.loc[("continue if engaged", 2), "share_assigned_1"] < 1.0

    assessment = namespace["assessment"]
    sensitivity = assessment.sensitivity.items
    assert sensitivity and all(item.status.value == "unavailable" for item in sensitivity)
    assert assessment.diagnostics["refute"].status.value == "unavailable"
    assert assessment.diagnostics["corrections"].status.value == "not_applicable"
    support = namespace["support"]
    # "the bound replaces no row": the largest weight (measured 33) is well under the cap of 100.
    assert (support["share_truncated"] == 0.0).all()
    assert (support["max_weight"] < 60.0).all()
    curve = namespace["curve"]
    assert list(curve["lower_bound"]) == [0.01, 0.05, 0.1]
    assert curve["truncated_score_cells"].iloc[0] == 0
    assert curve["truncated_score_cells"].iloc[-1] > 0
    # "less than 0.005, a small fraction of its standard error" (measured 0.0019 and 0.11 SE).
    movement = float(curve["delta_from_fitted"].abs().max())
    assert 0.0 < movement < 0.005
    assert movement < 0.3 * sequential.std_error
