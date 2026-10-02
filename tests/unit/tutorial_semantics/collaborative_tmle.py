"""The reviewed semantic callback for ``docs/examples/collaborative-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

import numpy as np

from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "collaborative-tmle.ipynb"

_PROBE = "reviews/notebook-review/probes/collaborative-tmle-final"
_SELECTOR = "tests/canonical/ctmle_selector/properties.csv"
_OAT = "tests/canonical/ctmle3_oat/properties.csv"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.003": f"largest move of either estimate between in-sample and five-fold fits of this "
    f"draw (0.0020, plain TMLE): {_PROBE}/crossfit_check.log",
    "0.048": f"design-based plain TMLE empirical SD (0.0479), 600 draws: {_PROBE}/summary.log",
    "0.97": f"design-based plain TMLE mean SE over empirical SD (0.973), 600 draws: "
    f"{_PROBE}/summary.log",
    "4.357": f"population nu2 of the design-based set, by quadrature: {_PROBE}/truth.log",
    "4.164": f"smallest design-based nu2 over 600 draws: {_PROBE}/summary.log",
    "4.702": f"largest design-based nu2 over 600 draws: {_PROBE}/summary.log",
    "0.069": f"all-three plain TMLE empirical SD (0.0694), 600 draws: {_PROBE}/summary.log",
    "0.055": f"C-TMLE empirical SD (0.0554), 600 draws: {_PROBE}/summary.log",
    "0.92": f"all-three plain TMLE mean SE over empirical SD (0.919), 600 draws: {_PROBE}/summary.log; "
    f"and coverage of crossfit_overfitting/cross_fitted_oat (0.9200): {_OAT}",
    "0.81": f"C-TMLE mean plug-in SE over empirical SD (0.813), 600 draws: {_PROBE}/summary.log",
    "0.1": f"constant-Q plain TMLE mean error (+0.1109), 600 draws: {_PROBE}/summary.log",
    "0.98": f"mean HC0 SE over the empirical SD of the C-TMLE estimate (0.976), 600 draws: "
    f"{_PROBE}/summary.log",
    "0.241": f"rmse_ratio of selector_necessity/collaborative (0.241034): {_SELECTOR}",
    "0.0037": f"bias_ci_upper of selector_necessity/collaborative (0.003711): {_SELECTOR}",
    "0.0030": f"bias_margin of selector_necessity/collaborative (0.003007): {_SELECTOR}",
    "0.80": f"se_ratio of selector_necessity/collaborative (0.801055): {_SELECTOR}",
    "0.0495": f"bias_ci_lower of selector_necessity/empty_control (0.049506): {_SELECTOR}",
    "0.0511": f"bias_ci_upper of selector_necessity/empty_control (0.051086): {_SELECTOR}",
    "0.0022": f"bias_margin of selector_necessity/empty_control (0.002164): {_SELECTOR}",
    "0.99": f"se_ratio of crossfit_overfitting/cross_fitted_oat (0.991589): {_OAT}",
    "0.54": f"coverage of crossfit_overfitting/in_sample_control (0.54): {_OAT}",
    "0.47": f"se_ratio of crossfit_overfitting/in_sample_control (0.472781): {_OAT}",
}


def check(namespace: dict[str, Any]) -> None:
    """A selector report describes its loss without assigning causal roles to omissions."""
    truth = namespace["truth"]
    # "the population ate, att, and atc all equal 1.000".
    assert truth["ate"] == truth["att"] == truth["atc"] == 1.0

    # The rename is load-bearing: a stale column name would make the exclusion vacuous.
    frame = namespace["frame"]
    assert {"baseline_readiness", "queue_lottery_draw", "social_support"} <= set(frame.columns)
    # "The score of this page has Gaussian noise, so it has no documented range": the draw
    # leaves the program's share scale of 0 to 1 on both sides.
    score = frame["transition_score"]
    assert score.min() < 0.0 and score.max() > 1.0

    # "The arms differ before the offer" on both the confounder and the instrument.
    by_arm = namespace["by_arm"]
    assert namespace["unadjusted"] > truth["ate"] + 0.5
    for column in ("baseline_readiness", "queue_lottery_draw"):
        assert by_arm.loc[1.0, column] > by_arm.loc[0.0, column] + 0.5

    # The protocol step prints the record, and every fit carries its digest.  "This page
    # changes four fields": the outcome, the death rule, time zero, and the rationale.
    base = navigation_protocol()
    protocol = namespace["protocol"]
    assert changed_fields(protocol, base) == {
        "outcome",
        "intercurrent_event_handling",
        "time_zero",
        "assumption_rationale",
    }
    assert "no documented range" in protocol.outcome
    # Only the death entry changes, and it no longer presupposes a worst score.
    assert protocol.intercurrent_event_handling[:2] == base.intercurrent_event_handling[:2]
    assert "worst transition score" in base.intercurrent_event_handling[2]
    assert "hypothetical strategy" in protocol.intercurrent_event_handling[2]
    assert protocol.assumption_rationale[3:] == base.assumption_rationale[1:]
    # "This score has no documented range to declare, so Step 6 fits in sample."
    assert not namespace["collaborative_method"].cross_fitting.enabled
    assert "in-sample nuisances" in namespace["collaborative"].summary()
    assert_protocol_recorded(
        NOTEBOOK,
        "protocol",
        protocol,
        namespace["collaborative"],
        namespace["reported"],
        namespace["plain"],
        namespace["weak_plain"],
        namespace["weak_collaborative"],
    )

    # The reported design leaves the queue draw out by design; C-TMLE searches all three.
    assert namespace["design_set"] == ("baseline_readiness", "social_support")
    assert set(namespace["approved_set"]) == set(namespace["covariates"])

    # "The last line lists `collaborative_tmle` beside `tmle` and `drtmle`."
    assert namespace["available"] == ["tmle", "collaborative_tmle", "drtmle"]
    assert "available=False" not in stored_output(NOTEBOOK, "identify")

    selection = namespace["selection"]
    assert selection.selected_covariates == ("social_support",)
    assert selection.path[-1] == ("social_support", "baseline_readiness", "queue_lottery_draw")
    # "On this draw ... it left out both the confounder and the instrument."
    assert "baseline_readiness" not in selection.selected_covariates
    assert "queue_lottery_draw" not in selection.selected_covariates
    # "The first two candidates are nearly tied", and the second one wins by that margin.
    assert -1e-3 < selection.cv_risk[1] - selection.cv_risk[0] < 0.0
    # "`risk` can rise along the path": the greedy search adds a variable after a step that
    # did not lower the in-sample loss.  Nonzero witness for the forced addition.
    assert any(later > earlier for earlier, later in pairwise(selection.train_risk))
    assert max(selection.n_steps) > 1
    summary = selection.summary().lower()
    assert "cross-validated" in summary and "loss" in summary
    assert "does not determine why" in summary
    # Whole words, because a substring test reads a footer that says "covariance" as a
    # footer that says "variance", and the report would then fail for a word it never used.
    for unsupported in ("bias", "variance", "confounder", "instrument"):
        assert not re.search(rf"\b{unsupported}\b", summary)

    # Step 6: the greedy path reports a point estimate and a diagnostic pair.  "The estimate
    # table has no interval column": the diagnostic column is named for what it is.  A
    # mutation that restored the Wald accessors would print an interval here again.
    plain, collaborative = namespace["plain"]["ate"], namespace["collaborative"]["ate"]
    reported = namespace["reported"]["ate"]
    assert collaborative.inference == "working_mechanism_plugin"
    assert plain.inference == "influence_curve" and reported.inference == "influence_curve"
    estimate_table = stored_output(NOTEBOOK, "collaborative")
    assert "working-mechanism se" in estimate_table
    assert "95% CI" not in estimate_table.split("estimand  psi", 1)[1].split("\n\n", 1)[0]
    assert "refused" not in estimate_table
    # "The interval contains the true value of 1.000 on this draw", and "Its estimate of 0.955
    # lies inside the reported interval": the C-TMLE check agrees with the design-based fit.
    assert covers(reported, truth["ate"])
    assert covers(reported, collaborative.psi)
    assert covers(plain, truth["ate"])
    assert covers(collaborative.plugin_interval, truth["ate"])
    assert collaborative.plugin_std_error < plain.std_error
    # The instrument in g costs precision on this draw.
    assert reported.std_error < plain.std_error

    # The all-three fit's tails against none in the design-based and collaborative fits.
    tails = namespace["tail_table"]
    assert tails.loc["share of g below 0.1", "all-three TMLE"] > 0.05
    assert tails.loc["share of g above 0.9", "all-three TMLE"] > 0.05
    for column in ("design-based TMLE", "collaborative TMLE"):
        assert tails.loc["share of g below 0.1", column] == 0.0
        assert tails.loc["share of g above 0.9", column] == 0.0
    # "because readiness still moves this g": nonzero witness that the design-based g is not
    # an empty g, and that its ratios sit between the all-three fit and the flat C-TMLE g.
    for arm in ("treated ESS / n", "control ESS / n"):
        assert tails.loc[arm, "all-three TMLE"] < tails.loc[arm, "design-based TMLE"] < 0.95
    assert tails.loc["share of g below 0.1", "collaborative TMLE"] == 0.0
    assert tails.loc["share of g above 0.9", "collaborative TMLE"] == 0.0
    assert tails.loc["truncated fraction", "collaborative TMLE"] == 0.0
    # "every row in an arm carries nearly the same weight" under a g that does not move
    # assignment.  The forced control below is the nonzero witness: its g does move, and both
    # ratios fall below 0.95 there.
    assert tails.loc["treated ESS / n", "collaborative TMLE"] > 0.99
    assert tails.loc["control ESS / n", "collaborative TMLE"] > 0.99
    overall = namespace["plain"].diagnostics.support().propensity_quantiles["overall"]
    assert overall[0.05] < 0.1 and overall[0.95] > 0.9
    # "The plain g has an AUC of 0.844, so it predicts the offer well."
    assert namespace["plain_auc"] > 0.8

    weak_selection = namespace["weak_selection"]
    assert weak_selection.selected_covariates == ("baseline_readiness", "social_support")
    weak_plain, weak_collaborative = namespace["weak_plain"], namespace["weak_collaborative"]
    assert weak_collaborative["ate"].plugin_std_error < 0.5 * weak_plain["ate"].std_error
    # "The plain interval and the C-TMLE plug-in spread both contain 1.000 on this draw."
    assert covers(weak_plain["ate"], truth["ate"])
    assert covers(weak_collaborative["ate"].plugin_interval, truth["ate"])
    # Leaving out only the draw removes the tails; readiness still moves g.  The ESS ratios
    # below 1 and the clever covariate above the intercept-only value are the nonzero witness
    # that the remaining weight variation is real, not an empty g in disguise.
    weak_tails = namespace["weak_tail_table"]
    plain_column, collaborative_column = "constant Q, plain", "constant Q, C-TMLE"
    for tail in ("share of g below 0.1", "share of g above 0.9"):
        assert weak_tails.loc[tail, plain_column] == tails.loc[tail, "all-three TMLE"]
        assert weak_tails.loc[tail, collaborative_column] == 0.0
    for arm in ("treated ESS / n", "control ESS / n"):
        assert weak_tails.loc[arm, collaborative_column] < 0.95
    assert (
        weak_tails.loc["max |clever covariate|", collaborative_column]
        > 2 * tails.loc["max |clever covariate|", "collaborative TMLE"]
    )

    assessment = namespace["assessment"]
    assert tuple(assessment.attention) == ()
    assert namespace["nuisance"].treatment_role == "collaborative_working_model"
    # "no row is truncated" in the support report of the collaborative fit.
    assert namespace["support"].truncated["fraction"] == 0.0
    # "its AUC sits near 0.5 and its calibration slope near 1" for a working model built on a
    # variable that does not move assignment.
    metrics = namespace["nuisance"]["propensity"].metrics
    assert 0.45 < metrics["auc"] < 0.55
    assert 0.9 < metrics["calibration_slope"] < 1.1
    # The status table prints only the rows that ran or run on request:
    # "`simulated_confounding`, a `deferred` row here".
    ledger = assessment.to_frame().set_index(["surface", "check"])["status"]
    assert ledger[("validation", "score_equations")] == "passed"
    assert ledger[("validation", "support")] == "completed"
    assert ledger[("validation", "nuisance_models")] == "completed"
    assert ledger[("sensitivity", "simulated_confounding")] == "deferred"
    stored = stored_output(NOTEBOOK, "assessment")
    assert "unavailable" not in stored and "not_applicable" not in stored
    assert re.search(r"simulated_confounding\s+deferred", stored)

    # Step 11: the omitted-variable bound on the reported fit, beside the all-three fit.
    # "`nu2` explains the difference": the draw in g inflates nu2 and lowers the robustness
    # value; sigma2 barely moves, so the witness is the nu2 gap itself.
    bounds = namespace["bound_table"]
    reported_bounds, plain_bounds = bounds["design-based TMLE"], bounds["all-three TMLE"]
    assert plain_bounds["nu2"] > 10.0 and reported_bounds["nu2"] < 0.5 * plain_bounds["nu2"]
    assert abs(reported_bounds["sigma2"] - plain_bounds["sigma2"]) < 0.05
    assert 0.0 < plain_bounds["robustness value"] < reported_bounds["robustness value"] < 1.0
    assert plain_bounds["confidence-limit value"] < reported_bounds["confidence-limit value"]
    sensitivity_output = stored_output(NOTEBOOK, "sensitivity")
    assert sensitivity_output.count("nu2") == 1
    for column in (reported_bounds, plain_bounds):
        assert f"{column['nu2']:.3f}" in sensitivity_output
    assert "refused" not in sensitivity_output

    # Step 11, collaborative fit: the stress surface starts from the C-TMLE estimate, and
    # every cell returns.
    point = namespace["collaborative"]["ate"]
    surface = namespace["surface"]
    cells = {(cell.treatment_strength, cell.outcome_strength): cell for cell in surface.cells}
    assert set(cells) == {(0.0, 0.0), (0.0, 0.5), (0.1, 0.0), (0.1, 0.5)}
    assert all(cell.failure is None for cell in cells.values())
    assert cells[0.0, 0.0].estimate == point.psi and cells[0.0, 0.0].displacement == 0.0
    # "The latent value is drawn independently of the data.  At treatment strength 0, its
    # association of -0.0480 is therefore a chance correlation on this draw": the latent
    # vector redrawn from its seed reproduces the printed association.  Nonzero witness: the
    # association is not zero, which is what lets the outcome axis move the estimate.
    latent = np.random.default_rng(surface.latent_seed).normal(size=len(frame))
    chance = np.corrcoef(latent, frame["transition_navigation"])[0, 1]
    association = cells[0.0, 0.0].induced_treatment_association
    assert association is not None and abs(association - chance) < 1e-9
    assert -0.06 < association < -0.03
    # "The outcome strength alone moves the estimate by +0.068278 through that small
    # correlation": the score loses 0.5 times a latent value that runs slightly against the
    # offer, so the contrast rises.
    assert cells[0.0, 0.5].displacement > 0.0 > association
    # "The treatment flips move the estimate down by about a third."
    for outcome in (0.0, 0.5):
        share = -cells[0.1, outcome].displacement / point.psi
        assert 0.25 < share < 0.4
    assert "It is not a bound" in sensitivity_output

    # "The regression gives the same estimate" with a larger robust standard error on this
    # draw than the plug-in diagnostic of the selected fit.
    assert abs(namespace["coefficient"] - point.psi) < 5e-4
    assert namespace["robust_se"] > 1.1 * point.plugin_std_error
