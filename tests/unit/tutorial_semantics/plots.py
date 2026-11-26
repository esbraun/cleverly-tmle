"""The reviewed semantic callback for ``docs/examples/plots``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the simulated-confounding cell and calls
:func:`check_refits`.  Each assertion is a claim the prose makes about this draw, with a
tolerance wide enough to survive a library release, and not a statistical guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.unit.tutorial_semantics import covers, figure_title

#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {
    "fit-reading": {"1.000": "forest"},
}

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("benchmarks", "simulated-confounding")

#: Decimals the prose writes that no output prints, each with how it follows from one.
UNPRINTED_DECIMALS: dict[str, str] = {}


def _check_estimate(namespace: dict[str, Any]) -> None:
    """The fit covers the true effect, and the weighting-only estimate has no interval."""
    truth = namespace["truth"]["ate"]
    assert truth == 1.0
    result, weighting = namespace["result"], namespace["weighting"]
    assert covers(result["ate"], truth)
    assert 0.9 < result["ate"].psi < 1.3
    assert result.inference_status == "influence_curve"
    # The second row of the forest plot has a point and no interval, because the fit holds none.
    assert weighting.inference_status == "estimated_mechanism_plugin"
    estimates = namespace["estimates"].set_index("method")
    assert estimates.loc["weighting only", ["ci_lower", "ci_upper"]].isna().all()
    assert estimates.loc["TMLE", ["ci_lower", "ci_upper"]].notna().all()
    assert weighting["ate"].support.label == "experimental"
    assert figure_title(namespace["forest_fig"]) == "Estimates and intervals"


def _check_overlap_and_balance(namespace: dict[str, Any]) -> None:
    """Offered patients have high chances, the capped share is small, and weighting balances."""
    chances = namespace["chances"]
    assert chances.loc["5%", "offered"] > 0.1, "few offered patients have a low chance"
    assert chances.loc["50%", "not offered"] < 0.4 < 0.6 < chances.loc["50%", "offered"]
    support = namespace["support"]
    assert 0.02 < support.truncated["fraction"] < 0.05
    low, high = support.bounds
    assert abs(low - 0.018) < 0.002 and abs(high - 0.982) < 0.002
    assert figure_title(namespace["overlap_fig"]) == "Predicted probability of treatment, by group"
    differences = namespace["differences"].set_index("covariate")
    before, after = differences["smd_unweighted"], differences["smd_weighted"]
    assert after["discharge_risk"] < 0.1 < before["discharge_risk"], "the main difference is gone"
    assert after["discharge_risk"] < 0.1 < after["prior_utilization"] < before["prior_utilization"]
    assert before["medication_burden"] < 0.1 and after["medication_burden"] < 0.1
    # The true chances leave about the same prior-utilization gap as the fitted ones, so the gap
    # is noise from large weights rather than a wrong treatment model.
    true_chances = differences["smd_true_chances"]
    assert abs(true_chances["prior_utilization"] - after["prior_utilization"]) < 0.05
    assert true_chances["prior_utilization"] > 0.1
    assert figure_title(namespace["balance_fig"]) == "Covariate balance before and after weighting"


def _check_truncation(namespace: dict[str, Any]) -> None:
    """The estimate drifts by about a tenth across the caps and stays well above zero."""
    frame = namespace["truncation"].to_frame()
    assert frame["bound"].is_monotonic_increasing
    assert frame["psi"].is_monotonic_decreasing, "a tighter cap lowers this estimate"
    assert 0.05 < frame["psi"].iloc[0] - frame["psi"].iloc[-1] < 0.2
    assert (frame["ci_lower"] > 0.5).all()
    fitted = frame[frame["is_fitted_bound"]]
    assert len(fitted) == 1
    assert abs(fitted["psi"].iloc[0] - namespace["result"]["ate"].psi) < 1e-6
    assert (
        abs(fitted["truncated_fraction"].iloc[0] - namespace["support"].truncated["fraction"])
        < 0.002
    )
    assert frame["truncated_fraction"].iloc[-1] > 0.3
    title = figure_title(namespace["truncation_fig"])
    assert title == "Estimate across probability limits"


def _check_policy(namespace: dict[str, Any]) -> None:
    """The effect falls below zero at half the odds and rises with the multiplier."""
    table = namespace["policy_table"].set_index("estimand")
    half, double, four = (table["psi"].iloc[i] for i in range(3))
    assert -0.2 < half < -0.05 < 0.05 < double < 0.2
    assert double < four < 0.3
    assert (table["ci_lower"] * table["ci_upper"] > 0).all(), "no interval crosses zero"
    assert figure_title(namespace["policy_fig"]) == "Effect by policy"


def _check_bound(namespace: dict[str, Any]) -> None:
    """The robustness value is a small floor, and the contour marks it."""
    robustness = namespace["robustness"]
    assert 0.15 < robustness["rv"] < 0.4
    assert robustness["rva"] < robustness["rv"]
    assert figure_title(namespace["contour_fig"]) == "Sensitivity to hidden confounding"


def _check_models(namespace: dict[str, Any]) -> None:
    """Every calibration group lies near the diagonal, and few rows carry the uncertainty."""
    bins = namespace["calibration_bins"]
    assert len(bins) == 10
    assert (bins["mean_predicted"] - bins["mean_observed"]).abs().max() < 0.07
    assert figure_title(namespace["calibration_fig"]) == "Calibration of predicted probabilities"
    row = namespace["influence_row"]
    assert row["n"] == 1500 and 20 < row["kish"] < 50, "a few dozen rows carry the uncertainty"
    assert figure_title(namespace["influence_fig"]) == "Concentration of the uncertainty"


def _check_learners_and_bootstrap(namespace: dict[str, Any]) -> None:
    """The linear model wins the ensemble, and the bootstrap spreads a little wider."""
    table = namespace["learner_table"]
    assert abs(table["weight"].sum() - 1.0) < 1e-9
    winner = table["cross-validated error"].idxmin()
    assert winner == "LinearRegression" and table.loc[winner, "weight"] > 0.99
    assert figure_title(namespace["learners_fig"]) == "Learner risks and weights"
    result = namespace["bootstrap_result"]
    assert result.bootstrap.n_failed == 0 and namespace["spread"]["n_replicates"] == 200
    row = namespace["row"]
    assert row["bootstrap_ci_lower"] < row["psi"] < row["bootstrap_ci_upper"]
    assert row["std_err"] < namespace["spread"]["sd"] < 1.5 * row["std_err"]
    assert row["bootstrap_support"] == "experimental"
    assert figure_title(namespace["bootstrap_fig"]) == "Bootstrap replicates"


def _check_evalue(namespace: dict[str, Any]) -> None:
    """The drawn curves pass through the E-values on the diagonal, and a wrong curve would not."""
    from cleverly.plotting._evalue import GID_EVALUE_ESTIMATE, GID_EVALUE_LIMIT

    evalue, fig = namespace["evalue"], namespace["evalue_fig"]
    ratio, limit_ratio = evalue.risk_ratio, evalue.risk_ratio_ci[0]
    assert 1.4 < ratio < 1.7
    assert 1.0 < limit_ratio < ratio
    # E-value = RR + sqrt(RR (RR - 1)), where the two links are equal.
    assert abs(evalue.point - (ratio + np.sqrt(ratio * (ratio - 1.0)))) < 1e-9
    assert abs(evalue.limit - (limit_ratio + np.sqrt(limit_ratio * (limit_ratio - 1.0)))) < 1e-9

    def factor(x: Any, y: Any) -> Any:
        """The bounding factor x * y / (x + y - 1) of the two links."""
        return x * y / (x + y - 1.0)

    lines = {line.get_gid(): line.get_xydata() for line in fig.axes[0].lines}
    for gid, ratio_i, value in (
        (GID_EVALUE_ESTIMATE, ratio, evalue.point),
        (GID_EVALUE_LIMIT, limit_ratio, evalue.limit),
    ):
        curve = np.asarray(lines[gid])
        assert np.allclose(factor(curve[:, 0], curve[:, 1]), ratio_i, rtol=1e-9)
        assert abs(factor(value, value) - ratio_i) < 1e-9
        # A pair of links below the curve gives a bias factor under the ratio, so it does not
        # explain the estimate away; one above it does.
        x = value + 0.5
        y = float(np.interp(x, curve[:, 0], curve[:, 1]))
        assert factor(x, y - 0.2) < ratio_i < factor(x, y + 0.2)
    assert evalue.limit < evalue.point
    assert figure_title(fig) == "Confounding that could explain the estimate away"


def _check_gamma(namespace: dict[str, Any]) -> None:
    """The range is one point at gamma 1, widens with gamma, and stays above zero up to 3."""
    frame = namespace["gamma_bounds"].to_frame()
    first, last = frame.iloc[0], frame.iloc[-1]
    assert first["gamma"] == 1.0 and abs(first["upper"] - first["lower"]) < 1e-9
    width = frame["upper"] - frame["lower"]
    assert width.is_monotonic_increasing
    assert last["gamma"] == 3.0 and 0.0 < last["lower_limit"] < last["lower"] < 0.5
    assert last["upper"] > 1.5
    assert figure_title(namespace["gamma_fig"]) == "Bounds under hidden confounding"


def _check_survey(namespace: dict[str, Any]) -> None:
    """About a fifth of the scores are missing, and the tilt lowers the estimate to zero."""
    survey = namespace["survey"]
    share = 1.0 - float(survey["responded"].mean())
    assert 0.2 < share < 0.25
    result = namespace["survey_result"]
    assert covers(result["ate"], namespace["survey_truth"]["ate"])
    frame = namespace["tilt"].to_frame()
    assert frame["psi"].is_monotonic_decreasing
    assert abs(frame["psi"].iloc[0] - result["ate"].psi) < 1e-9, "gamma 0 is the main analysis"
    # The grid reaches 3.0, so the estimate crosses zero at the tipping value between 2.0 and 3.0.
    assert frame["gamma"].iloc[-1] == 3.0 and frame["psi"].iloc[-1] < 0.0
    tipping = namespace["tipping"]
    assert 2.0 < tipping < 3.0
    assert (frame["psi"][frame["gamma"] < tipping] > 0.0).all()
    assert (frame["psi"][frame["gamma"] > tipping] < 0.0).all()
    assert figure_title(namespace["tilt_fig"]) == "Estimate across missingness assumptions"


def _check_zipper(namespace: dict[str, Any]) -> None:
    """Fewer than six in ten intervals cover the truth, and every miss is below it."""
    study = namespace["coverage_study"]
    records = namespace["records"]
    truth = namespace["truth_ate"]
    assert len(records) == 100
    inside = (records["low"] <= truth) & (truth <= records["high"])
    assert (inside == records["covered"]).all(), "each recorded interval agrees with `covered`"
    assert 0.3 < inside.mean() < 0.6
    assert (records["low"] > truth).sum() == 0
    assert (records["high"] < truth).sum() == (~inside).sum() > 30
    assert abs(study.summaries["ate"].truth - truth) < 1e-12
    assert figure_title(namespace["zipper_fig"]) == "Interval coverage"


def _check_visits(namespace: dict[str, Any]) -> None:
    """The largest weight grows and the effective sample size falls from visit 1 to 2."""
    support = namespace["visit_support"].set_index(["regimen", "time"])
    first, second = support.loc[("always", 1)], support.loc[("always", 2)]
    assert first["max_weight"] < second["max_weight"]
    assert first["effective_n"] / first["n_followed"] > second["effective_n"] / second["n_followed"]
    assert (
        first["effective_n"] < first["n_followed"] and second["effective_n"] < second["n_followed"]
    )
    assert len(namespace["node_fig"].axes) == 2, "one panel per visit"
    largest = namespace["balance_by_visit"].groupby("diagnostic")[["before", "after"]].max()
    assert largest.loc["confounding", "after"] < 0.1 < largest.loc["confounding", "before"]
    assert largest.loc["censoring", "after"] < largest.loc["censoring", "before"]
    # The offer changes the later covariate, so the feedback row stays large after weighting.
    assert largest.loc["feedback", "after"] > 0.5
    assert figure_title(namespace["visit_weights_fig"]) == "Weights over time"
    moves = namespace["cap_curve"]["move in SEs"].abs()
    assert moves.max() < 0.05, "no estimate leans on its largest weights"
    kish = namespace["influence_rows"]["kish"]
    assert (
        kish["ey_regimen[never]"]
        < kish["ey_regimen[always]"]
        < kish["ate_regimen[never vs always]"]
    )
    assert figure_title(namespace["node_fig"]) == "Treatment probability at each time point"


def _check_exact(namespace: dict[str, Any]) -> None:
    """The offer lowers the two-year relapse risk; one year reads experimental; no weight capped."""
    result = namespace["exact_result"]
    effect = result["ate_regimen[offered vs not offered, relapse @ t=2]"]
    assert effect.ci[1] < 0.0, "the two-year relapse interval lies below zero"
    for name, estimate in result.items():
        if "@ t=1]" in name and not name.startswith("rr_"):
            assert estimate.support.label == "experimental"
    assert result.simultaneous is not None
    assert figure_title(namespace["exact_curve_fig"]) == "Risk at the target times"
    support = namespace["exact_support"]
    assert support.bounded_share == 0.0, "the reading says no weight was capped"
    assert namespace["retention"].loc["1%", 2.0] > 0.6
    title = figure_title(namespace["exact_overlap_fig"])
    assert title == "Predicted probability of treatment, by arm"
    effective = namespace["effective"]
    assert ((effective["effective"] / effective["n"]).between(0.75, 0.9)).all()
    assert figure_title(namespace["exact_weights_fig"]) == "Weights over time"
    assert figure_title(namespace["exact_learners_fig"]) == "Hazard and treatment model selection"
    lost = namespace["lost"]
    assert len(lost) == 6 and (lost["support"] == "experimental").all()
    relapse = lost.set_index("estimand").loc["rmtl_regimen[offered vs not offered, relapse @ t=2]"]
    assert relapse["ci_upper"] < 0.0, "the offer cuts the years lost to relapse"


def check(namespace: dict[str, Any]) -> None:
    """The plots tutorial's claims hold at the displayed sample size."""
    _check_estimate(namespace)
    _check_overlap_and_balance(namespace)
    _check_truncation(namespace)
    _check_policy(namespace)
    _check_bound(namespace)
    _check_gamma(namespace)
    _check_survey(namespace)
    _check_evalue(namespace)
    _check_models(namespace)
    _check_learners_and_bootstrap(namespace)
    _check_visits(namespace)
    _check_exact(namespace)
    _check_zipper(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The offer drivers pass the floor; the invented confounder never reverses the estimate."""
    floor = namespace["robustness"]["rv"]
    std_error = namespace["result"]["ate"].std_error
    psi = namespace["result"]["ate"].psi
    rows = {row["left out"]: row for row in namespace["rows"]}
    for name in ("discharge_risk", "prior_utilization"):
        row = rows[name]
        assert row["score side"] > floor and row["weight side"] > floor
        assert abs(row["estimate without it"] - psi) > 5 * std_error
    # The witness the contrast needs: a characteristic below the floor on both sides moves the
    # estimate by less than one standard error.
    weak = rows["medication_burden"]
    assert weak["score side"] < floor and weak["weight side"] < floor
    assert abs(weak["estimate without it"] - psi) < std_error
    frame = namespace["cells"].set_index(["treatment_strength", "outcome_strength"])
    estimate = frame["estimate"]
    assert abs(estimate[(0.0, 0.0)] - namespace["result"]["ate"].psi) < 1e-6
    assert (estimate > 0.0).all(), "no cell crosses zero"
    flips = estimate.xs(0.0, level="outcome_strength")
    assert flips.is_monotonic_decreasing and flips.iloc[-1] < 0.5 * flips.iloc[0]
    calibration = namespace["surface"].calibration_frame().set_index("role")
    assert 0.2 < calibration.loc["treatment", "strength"] < 0.4
    title = figure_title(namespace["surface_fig"])
    assert title == "Estimate under a simulated common cause"
