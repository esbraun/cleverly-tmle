"""The reviewed semantic callback for ``docs/examples/collaborative-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

NOTEBOOK = EXAMPLES / "collaborative-tmle.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """A selector report describes its loss without assigning causal roles to omissions."""
    truth = namespace["truth"]
    # "the population ate, att, and atc all equal 1.000".
    assert truth["ate"] == truth["att"] == truth["atc"] == 1.0

    # The rename is load-bearing: a stale column name would make the exclusion vacuous.
    covariates = set(namespace["frame"].columns)
    assert {"baseline_readiness", "queue_lottery_draw", "social_support"} <= covariates

    # "The arms differ before the offer" on both the confounder and the instrument.
    by_arm = namespace["by_arm"]
    assert namespace["unadjusted"] > truth["ate"] + 0.5
    for column in ("baseline_readiness", "queue_lottery_draw"):
        assert by_arm.loc[1.0, column] > by_arm.loc[0.0, column] + 0.5

    # The protocol step prints the record, and the collaborative fit carries its digest.
    # The reading names the fields this page changes in the program protocol.
    assert changed_fields(namespace["protocol"], navigation_protocol()) == {
        "outcome",
        "time_zero",
        "assumption_rationale",
    }
    # "A standardized score takes its scale from the data, so the analyst can declare no finite
    # support for it": the page's own reason for fitting in sample, and the fit reports it.
    assert not namespace["collaborative_method"].cross_fitting.enabled
    assert "in-sample nuisances" in namespace["collaborative"].summary()
    assert_protocol_recorded(
        NOTEBOOK,
        "protocol",
        namespace["protocol"],
        namespace["collaborative"],
        namespace["plain"],
        namespace["weak_plain"],
        namespace["weak_collaborative"],
    )

    # "Two other methods are not available for the ATE, and each line prints the reason."
    refused = [m for m in namespace["effect"].available_methods() if not m.available]
    assert len(refused) == 2 and all(m.reason for m in refused)
    # "For the ATT, the catalog refuses it with the reason ...".
    att_collaborative = namespace["att_collaborative"]
    assert att_collaborative.available is False
    assert "no collaborative score is evidenced" in att_collaborative.reason

    selection = namespace["selection"]
    assert selection.selected_covariates == ("social_support",)
    assert selection.path[-1] == ("social_support", "baseline_readiness", "queue_lottery_draw")
    # "The search kept it, and it left out both the confounder and the instrument."
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

    # "Both intervals contain the true value" and the collaborative standard error is smaller.
    plain, collaborative = namespace["plain"]["ate"], namespace["collaborative"]["ate"]
    assert covers(plain, truth["ate"]) and covers(collaborative, truth["ate"])
    assert collaborative.std_error < plain.std_error

    # The plain fit's tails against the collaborative fit's none.
    tails = namespace["tail_table"]
    assert tails.loc["share of g below 0.1", "plain TMLE"] > 0.05
    assert tails.loc["share of g above 0.9", "plain TMLE"] > 0.05
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
    # "The plain g has an AUC of 0.843, so it predicts the offer well."
    assert namespace["plain_auc"] > 0.8

    weak_selection = namespace["weak_selection"]
    assert weak_selection.selected_covariates == ("baseline_readiness", "social_support")
    weak_plain, weak_collaborative = namespace["weak_plain"], namespace["weak_collaborative"]
    assert weak_collaborative["ate"].std_error < 0.5 * weak_plain["ate"].std_error
    assert covers(weak_plain["ate"], truth["ate"])
    assert covers(weak_collaborative["ate"], truth["ate"])
    # Leaving out only the draw removes the tails; readiness still moves g.  The ESS ratios
    # below 1 and the clever covariate above the intercept-only value are the nonzero witness
    # that the remaining weight variation is real, not an empty g in disguise.
    weak_tails = namespace["weak_tail_table"]
    plain_column, collaborative_column = "constant Q, plain", "constant Q, C-TMLE"
    for tail in ("share of g below 0.1", "share of g above 0.9"):
        assert weak_tails.loc[tail, plain_column] == tails.loc[tail, "plain TMLE"]
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

    # "The difference comes from `nu2`", because the representer reads the selected near-constant
    # g: "`nu2` lands close to 1/p + 1/(1 - p)" for the treated share p.  The nonzero witness: a
    # representer built from the full mechanism gives 11.963 on the same draw.
    rows = namespace["sensitivity_rows"]
    plain_row, collaborative_row = rows["plain TMLE"], rows["collaborative TMLE"]
    assert collaborative_row["nu2"] < 0.5 * plain_row["nu2"]
    assert collaborative_row["robustness value"] > plain_row["robustness value"]
    share = float(namespace["frame"]["transition_navigation"].mean())
    constant_g_nu2 = 1.0 / share + 1.0 / (1.0 - share)
    assert abs(collaborative_row["nu2"] - constant_g_nu2) < 0.01 * constant_g_nu2
    # "optimistic by construction": a near-constant representer is close to the within-arm mean
    # of the full representer, so its second moment cannot exceed the full one (projection).
    assert collaborative_row["nu2"] <= plain_row["nu2"]

    # "The same representer sets the collaborative standard error" and the regression gives
    # "the same estimate" with a larger robust standard error on this draw.
    point = namespace["collaborative"]["ate"]
    assert abs(namespace["implied_se"] - point.std_error) < 0.01 * point.std_error
    assert abs(namespace["coefficient"] - point.psi) < 5e-4
    assert namespace["robust_se"] > 1.1 * point.std_error
