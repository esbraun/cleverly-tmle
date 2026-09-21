"""The reviewed semantic callback for ``docs/examples/longitudinal-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly.datasets import longitudinal_navigation_protocol
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

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
    # "The 923 patients lost before day seven" and "In total, 1476 patients were lost before
    # day 30 and have no outcome": the subset's tracking loss is exactly the missing outcomes.
    lost = len(frame) - len(namespace["observed"])
    assert int((frame["tracked_day7"] == 0).sum()) == 923
    assert lost == int(frame["transition_top_box"].isna().sum()) == 1476
    # "The subset keeps 4098 of 8000 patients" and "the 2426 tracked patients whose
    # assignments differed".
    assert len(namespace["consistent"]) == 4098
    assert len(namespace["observed"]) - len(namespace["consistent"]) == 2426

    effect = namespace["effect"]
    summary = effect.summary()
    assert "sequential exchangeability" in summary and "sequential positivity" in summary
    assert "The protocol scores death before day 30 as not top box" in summary
    # "The `adjustment/history` line lists only the baseline covariates."
    assert "adjustment/history: ['age', 'baseline_readiness']" in summary

    # The protocol step changes exactly the two fields the reading names, appending the rule.
    protocol, base = namespace["protocol"], longitudinal_navigation_protocol()
    assert changed_fields(protocol, base) == {"treatment_strategies", "treatment_versions"}
    assert protocol.treatment_strategies[:2] == base.treatment_strategies
    assert protocol.treatment_versions[:2] == base.treatment_versions
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 3
    # The protocol step prints the record, and both fits carry its digest.
    fingerprint = assert_protocol_recorded(
        NOTEBOOK, "protocol", protocol, namespace["result"], namespace["rule_result"]
    )
    assert fingerprint != base.fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "rule")

    # The top-box shares with no navigation and with both offers are quadrature truths.
    truth = namespace["truth"]
    assert truth["ey_regimen[never]"] == pytest.approx(0.42, abs=0.005)
    assert truth["ey_regimen[always]"] == pytest.approx(0.78, abs=0.005)

    target = namespace["target"]
    assert namespace["crude"] > target
    result = namespace["result"]
    adjusted, baseline_only = namespace["adjusted"], namespace["baseline_only"]
    sequential = result["ate_regimen[always vs never]"]
    # "the interval ... contains the true contrast" on this draw.
    assert covers(sequential, target)
    # "0.137 below", "0.051 above", and "within one standard error" (measured 0.43 SE).
    assert adjusted.psi < target - 0.1
    assert namespace["adjusted_gap_magnitude"] == pytest.approx(target - adjusted.psi)
    assert round(namespace["adjusted_gap_magnitude"], 3) == 0.137
    assert 0.03 < baseline_only.psi - target < 0.08
    assert abs(sequential.psi - target) < sequential.std_error
    # "400 navigator teams".
    assert frame["navigator_team"].nunique() == 400
    # "with the folds and seed of Step 6": the point method replaces only the models.
    point_method, method = namespace["point_method"], namespace["sequential"]
    assert point_method.cross_fitting == method.cross_fitting
    assert point_method.runtime == method.runtime
    # "The fit runs in sample, and the navigator teams are the reason": the page's own reason
    # for the fold setting, and the fit reports it.
    assert not method.cross_fitting.enabled
    assert "nuisances fitted in sample" in result.summary()

    rule = namespace["rule_result"]
    assert set(rule.estimates) == {
        "ate_regimen[always vs never]",
        "ate_regimen[continue if engaged vs never]",
    }
    # "The `always` row gives 0.369 and the interval (0.335, 0.403), the same as Step 6."
    rule_always = rule["ate_regimen[always vs never]"]
    assert rule_always.psi == pytest.approx(sequential.psi, abs=1e-9)
    assert rule_always.ci == pytest.approx(sequential.ci, abs=1e-9)
    rule_point = rule["ate_regimen[continue if engaged vs never]"]
    rule_truth = truth["ate_regimen[treat then continue if l2 positive vs never]"]
    assert namespace["rule_truth"] == rule_truth
    assert covers(rule_point, rule_truth)
    assigned = namespace["rule_support"].set_index(["regimen", "time"])
    assert assigned.loc[("continue if engaged", 1), "share_assigned_1"] == 1.0
    # A nonzero witness for the rule's mask: the day-seven share is exactly the engaged share
    # among patients at risk under the rule (discharge navigation, tracked at day seven). A
    # sign-flipped rule would give one minus this share, and "0.799" requires it above 0.5.
    at_risk = frame[(frame["navigation_discharge"] == 1) & (frame["tracked_day7"] == 1)]
    expected = float((at_risk["engagement_day7"] > 0).mean())
    assert assigned.loc[("continue if engaged", 2), "share_assigned_1"] == pytest.approx(expected)
    assert expected > 0.5
    # "The `rule minus always` line is -0.038 ... The true difference is -0.040. On this draw
    # the interval excludes zero."
    rule_vs_always = namespace["rule_vs_always"]
    assert rule_vs_always.psi == pytest.approx(rule_point.psi - rule_always.psi, abs=1e-9)
    assert covers(rule_vs_always, rule_truth - target)
    assert rule_vs_always.ci[1] < 0.0
    assert rule_truth - target < 0.0
    # "`result.diagnostics.support()` returns the same support report that
    # `assessment.report("support")` retains."
    support = namespace["support"]
    assert result.diagnostics.support().to_frame().equals(support)

    assessment = namespace["assessment"]
    sensitivity = assessment.sensitivity.items
    assert sensitivity and all(item.status.value == "unavailable" for item in sensitivity)
    # "Six report that no longitudinal derivation is registered. `simulated_confounding` has no
    # time-indexed latent law. `missingness` and `tipping_gamma` have no longitudinal adapter."
    details = {item.name: item.detail for item in sensitivity}
    registered = {name for name, detail in details.items() if "derivation is registered" in detail}
    assert len(registered) == 6
    assert "time-indexed latent law" in details["simulated_confounding"]
    for name in ("missingness", "tipping_gamma"):
        assert "adapter is implemented" in details[name]
    assert set(details) == registered | {"simulated_confounding", "missingness", "tipping_gamma"}
    assert assessment.diagnostics["refute"].status.value == "unavailable"
    assert assessment.diagnostics["corrections"].status.value == "not_applicable"
    # "The `Returned results` section lists the support report."
    assert "Returned results" in stored_output(NOTEBOOK, "assessment")
    refused = {item.name for item in effect.available_methods() if not item.available}
    assert {"collaborative_tmle", "drtmle"} <= refused
    # "the bound replaces no row": the largest weight (measured 33.3) is well under the cap of 100.
    assert (support["share_truncated"] == 0.0).all()
    assert (support["max_weight"] < 60.0).all()
    assert (support["effective_n"] / support["n_followed"]).min() > 0.75
    # "2362 for `always` and 1736 for `never` at node 2".
    followed = support.set_index(["regimen", "time"])["n_followed"]
    assert (followed[("always", 2)], followed[("never", 2)]) == (2362, 1736)
    curve = namespace["curve"]
    assert list(curve["lower_bound"]) == [0.01, 0.05, 0.1]
    assert curve["truncated_score_cells"].iloc[0] == 0
    # "truncates 75 of 11175 score cells ... one followed patient at one node of one plan.
    # An in-sample fit scores each such cell once, so the total is the sum of n_followed".
    assert curve["truncated_score_cells"].iloc[-1] == 75
    assert (curve["evaluated_score_cells"] == 11175).all()
    assert int(support["n_followed"].sum()) == 11175
    # "0.0019, which is 0.11 standard errors".
    movement = float(curve["delta_from_fitted"].abs().max())
    assert 0.0 < movement < 0.005
    assert movement < 0.3 * sequential.std_error

    # "Every score row passed": raw solver residuals are nonzero but far below the report
    # tolerance, while the presentation copy renders only those negligible solver values as zero.
    scores = namespace["scores"]
    assert scores["passed"].all()
    solver = scores[scores["kind"] == "solver"]
    assert len(solver) == 4
    assert solver["relative_score"].gt(0.0).any()
    assert (solver["relative_score"] < namespace["solver_display_floor"]).all()
    score_display = namespace["score_display"]
    displayed_solver = score_display[score_display["kind"] == "solver"]
    assert (displayed_solver["relative_score"] == 0.0).all()
    # "every row is of one kind" and "A cross-fitted fit also reports `solver` rows only".
    # The pooled cross-fitted fit and this in-sample fit both solve one score per node.
    assert namespace["score_kinds"] == ["solver"]
    assert "kinds of score row: ['solver']" in stored_output(NOTEBOOK, "retained-reports")
    # "Every calibration slope here sits within 0.03 of 1": each model is measured on the rows
    # it was fitted on. The auc column still separates the models, and the lowest value is the
    # censoring model at node 2, which "barely separates the patients who stay tracked".
    nuisance = namespace["nuisance"]
    assert nuisance["calibration_slope"].between(0.97, 1.03).all()
    lowest = nuisance.loc[nuisance["auc"].idxmin()]
    assert (lowest["role"], lowest["time"]) == ("censoring", 2)
    assert lowest["auc"] < 0.6
    # "The pseudo-outcome rows have continuous targets, so they have no `auc`."
    assert nuisance.loc[nuisance["role"] == "pseudo_outcome", "auc"].isna().all()
    mechanisms = nuisance[nuisance["role"].isin(["treatment", "censoring"])]
    assert mechanisms["regimen"].isna().all()
