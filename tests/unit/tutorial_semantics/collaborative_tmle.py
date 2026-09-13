"""The reviewed semantic callback for ``docs/examples/collaborative-tmle``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

import re
from typing import Any


def check(namespace: dict[str, Any]) -> None:
    """A selector report describes its loss without assigning causal roles to omissions."""
    selection = namespace["selection"]
    assert selection.selected_covariates == ()

    summary = selection.summary().lower()
    assert "cross-validated" in summary and "loss" in summary
    assert "does not determine why" in summary
    # Whole words, because a substring test reads a footer that says "covariance" as a
    # footer that says "variance", and the report would then fail for a word it never used.
    for unsupported in ("bias", "variance", "confounder", "instrument"):
        assert not re.search(rf"\b{unsupported}\b", summary)

    # The rename is load-bearing: a stale column name would make the exclusion vacuous.
    covariates = set(namespace["frame"].columns)
    assert {"baseline_readiness", "queue_lottery_draw", "social_support"} <= covariates
    weak_selection = namespace["weak_selection"]
    assert "baseline_readiness" in weak_selection.selected_covariates
    assert "queue_lottery_draw" not in weak_selection.selected_covariates
    assert namespace["weak_collaborative"]["ate"].std_error < (
        0.5 * namespace["weak_plain"]["ate"].std_error
    )
    assert namespace["nuisance"].treatment_role == "collaborative_working_model"

    # "More than 5% ... below 0.1, and more than 5% ... above 0.9" for the plain fit.
    overall = namespace["plain"].diagnostics.support().propensity_quantiles["overall"]
    assert overall[0.05] < 0.1 and overall[0.95] > 0.9
    # "its AUC is about 0.5" for the intercept-only working model. The calibration slope of a
    # constant model is undefined noise, so the page makes no claim about it.
    assert 0.45 < namespace["nuisance"]["propensity"].metrics["auc"] < 0.55
