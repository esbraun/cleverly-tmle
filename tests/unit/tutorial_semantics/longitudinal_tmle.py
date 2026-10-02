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

_PROBE = "reviews/notebook-review/probes/longitudinal-tmle-final"
_LTMLE = "tests/canonical/ltmle"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.935": f"Step 6 clustered coverage (374 of 400), seeds 1000-1399: {_PROBE}/summary.log",
    "0.97": f"mean clustered SE over empirical SD (0.969), 400 draws: {_PROBE}/summary.log",
    "0.65": f"bound that both censoring auc values stayed below on 400 of 400 draws "
    f"(maximum 0.6193): {_PROBE}/summary.log",
    "0.9413": f"coverage of ate_regimen[always vs never], cleverly: {_LTMLE}/summary.csv",
    "0.9890": f"se_ratio of ate_regimen[always vs never], cleverly: {_LTMLE}/summary.csv",
    "-0.0036": f"bias_ci_lower of double_robustness/static__mechanism_correct (-0.003633): "
    f"{_LTMLE}/properties.csv",
    "0.0053": f"bias_ci_upper of double_robustness/static__mechanism_correct (0.005264): "
    f"{_LTMLE}/properties.csv",
    "0.0149": f"bias_margin of double_robustness/static__mechanism_correct (0.014934): "
    f"{_LTMLE}/properties.csv",
    "1.0132": f"se_ratio of double_robustness/static__mechanism_correct (1.013217): "
    f"{_LTMLE}/properties.csv",
}


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
    # "Each gap holds discharge navigation fixed, so engagement drives the second decision":
    # within each discharge stratum, engaged patients receive day-seven navigation more often.
    # The review probe found both gaps positive on 400 of 400 draws.
    shares, gap = namespace["day7_shares"], namespace["engagement_gap"]
    assert list(shares.index) == [0.0, 1.0] and list(gap.index) == [0.0, 1.0]
    assert (gap > 0.1).all()
    assert gap.round(3).tolist() == [0.212, 0.213]
    # A nonzero witness that the strata differ: discharge navigation raises the day-seven share
    # at both engagement levels, so a pooled comparison would mix the two decisions.
    assert (shares.loc[1.0] - shares.loc[0.0] > 0.1).all()
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
    # "The two `history at` lines give the history at each navigation node."
    assert "history at navigation_discharge: ['age', 'baseline_readiness']" in summary
    assert (
        "history at navigation_day7: ['age', 'baseline_readiness', 'engagement_day7', "
        "'navigation_discharge']"
    ) in summary

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

    # "The first table lists the four operations that ran on this fit. The score-equation check
    # `passed`. The support report, the nuisance report, and the truncation curve are
    # `completed`."
    ran = namespace["ran"].set_index("check")["status"]
    assert ran.to_dict() == {
        "score_equations": "passed",
        "support": "completed",
        "nuisance_models": "completed",
        "truncation_curve": "completed",
    }
    # "at least 79.5% of the followers, the minimum ratio printed under the table".
    assert namespace["kish_ratio"] == pytest.approx(
        (support["effective_n"] / support["n_followed"]).min()
    )
    # "the bound replaces no row": the largest weight (measured 35.2) is under the cap of 100.
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
    # "Every calibration slope here sits between 0.999 and 1.004, and each regression slope
    # sits between 1.007 and 1.023": each model is measured on the rows it was fitted on. A
    # binary row carries the logistic slope and a pseudo-outcome row the linear one. "The two
    # censoring models have the lowest values: 0.599 at node 1 and 0.557 at node 2." Which of
    # the two is lower varies by draw (node 1 on 77 of 400 in the review probe), so the callback
    # pins both below 0.65 and below every other auc, as on all 400 probe draws.
    nuisance = namespace["nuisance"]
    binary = nuisance["calibration_slope"].notna()
    assert (binary == nuisance["regression_slope"].isna()).all()
    assert nuisance.loc[binary, "calibration_slope"].round(3).between(0.999, 1.004).all()
    assert nuisance.loc[~binary, "regression_slope"].round(3).between(1.007, 1.023).all()
    censoring = nuisance[nuisance["role"] == "censoring"].set_index("time")["auc"]
    assert censoring.round(3).to_dict() == {1: 0.599, 2: 0.557}
    assert (censoring < 0.65).all()
    others = nuisance.loc[nuisance["role"] != "censoring", "auc"].dropna()
    assert censoring.max() < others.min()
    # "The pseudo-outcome rows have continuous targets, so they have no `auc`."
    assert nuisance.loc[nuisance["role"] == "pseudo_outcome", "auc"].isna().all()
    mechanisms = nuisance[nuisance["role"].isin(["treatment", "censoring"])]
    assert mechanisms["regimen"].isna().all()

    # Step 10: "Each row drops one recorded covariate, refits, and divides the move by that
    # standard error." The moves are 2.30, 0.68, and 2.76 standard errors on this draw, all
    # positive, and dropping engagement moves the estimate most.
    benchmark = namespace["benchmark"].set_index("dropped")
    assert list(benchmark.index) == ["age", "baseline_readiness", "engagement_day7"]
    assert benchmark["move_in_se"].tolist() == [2.30, 0.68, 2.76]
    assert (benchmark["move"] > 0.0).all()
    assert benchmark["move_in_se"].idxmax() == "engagement_day7"
    assert ((benchmark["psi"] - benchmark["move"]) - sequential.psi).abs().max() < 2e-4
    # Nonzero witness that each refit drops its covariate: the last refit has no time-varying
    # history, and its estimate moves by more than two standard errors.
    reduced = namespace["reduced"]
    assert reduced.time_varying == ((), ())
    assert reduced.baseline == ("age", "baseline_readiness")
    assert namespace["full_design"].time_varying == ((), ("engagement_day7",))
