"""The reviewed semantic callback for ``docs/examples/collaborative-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

import re
from typing import Any

from tests.unit.tutorial_semantics import EXAMPLES, stored_output

NOTEBOOK = EXAMPLES / "collaborative-tmle.ipynb"


def _contains(interval: tuple[float, float], value: float) -> bool:
    return interval[0] <= value <= interval[1]


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
    fingerprint = namespace["protocol"].fingerprint
    assert fingerprint in stored_output(NOTEBOOK, "protocol")
    assert namespace["collaborative"].provenance.protocol_fingerprint == fingerprint

    # "For the ATT, the catalog refuses it with the reason ...".
    att_collaborative = namespace["att_collaborative"]
    assert att_collaborative.available is False
    assert "no collaborative score is evidenced" in att_collaborative.reason

    selection = namespace["selection"]
    assert selection.selected_covariates == ()
    assert selection.path[-1] == ("social_support", "baseline_readiness", "queue_lottery_draw")
    summary = selection.summary().lower()
    assert "cross-validated" in summary and "loss" in summary
    assert "does not determine why" in summary
    # Whole words, because a substring test reads a footer that says "covariance" as a
    # footer that says "variance", and the report would then fail for a word it never used.
    for unsupported in ("bias", "variance", "confounder", "instrument"):
        assert not re.search(rf"\b{unsupported}\b", summary)

    # "Both intervals contain the true value" and the collaborative standard error is smaller.
    plain, collaborative = namespace["plain"]["ate"], namespace["collaborative"]["ate"]
    assert _contains(plain.ci, truth["ate"]) and _contains(collaborative.ci, truth["ate"])
    assert collaborative.std_error < plain.std_error

    # The plain fit's tails against the collaborative fit's none.
    tails = namespace["tail_table"]
    assert tails.loc["share of g below 0.1", "plain TMLE"] > 0.05
    assert tails.loc["share of g above 0.9", "plain TMLE"] > 0.05
    assert tails.loc["share of g below 0.1", "collaborative TMLE"] == 0.0
    assert tails.loc["share of g above 0.9", "collaborative TMLE"] == 0.0
    overall = namespace["plain"].diagnostics.support().propensity_quantiles["overall"]
    assert overall[0.05] < 0.1 and overall[0.95] > 0.9

    weak_selection = namespace["weak_selection"]
    assert "baseline_readiness" in weak_selection.selected_covariates
    assert "queue_lottery_draw" not in weak_selection.selected_covariates
    weak_plain, weak_collaborative = namespace["weak_plain"], namespace["weak_collaborative"]
    assert weak_collaborative["ate"].std_error < 0.5 * weak_plain["ate"].std_error
    assert _contains(weak_plain["ate"].ci, truth["ate"])
    assert _contains(weak_collaborative["ate"].ci, truth["ate"])

    assessment = namespace["assessment"]
    assert tuple(assessment.attention) == ()
    assert namespace["nuisance"].treatment_role == "collaborative_working_model"
    # "its AUC is about 0.5" for the intercept-only working model. The calibration slope of a
    # constant model is undefined noise, so the page makes no claim about it.
    assert 0.45 < namespace["nuisance"]["propensity"].metrics["auc"] < 0.55

    # "nu2 is small and the robustness value is large" because the representer reads the
    # selected constant g.  The nonzero witness: for a constant g equal to the treated share p,
    # nu2 is 1/p + 1/(1 - p), which a representer built from the full mechanism would not give.
    rows = namespace["sensitivity_rows"]
    plain_row, collaborative_row = rows["plain TMLE"], rows["collaborative TMLE"]
    assert collaborative_row["nu2"] < 0.5 * plain_row["nu2"]
    assert collaborative_row["robustness value"] > plain_row["robustness value"]
    share = float(namespace["frame"]["transition_navigation"].mean())
    constant_g_nu2 = 1.0 / share + 1.0 / (1.0 - share)
    assert abs(collaborative_row["nu2"] - constant_g_nu2) < 0.01 * constant_g_nu2
