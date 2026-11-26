"""The reviewed semantic callback for ``docs/examples/two-decision-rules``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  Each assertion pins a reading the prose makes about this
draw; a relation on one draw certifies no method.  The page has no refit cell, because every
refit surface refuses a learned-regimen fit (F27).  The page's data are the law of the registered
``learned-regimen-ltmle`` study, and ``tests/unit/test_datasets_two_decisions.py`` pins the data
and their true values against that study; here the true value of the fit's own rules is checked
once more against the study harness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.unit.tutorial_semantics import EXAMPLES, assert_protocol_recorded, covers, stored_output

#: The refit cells the rule derives from this notebook today, in notebook order: none.
REFIT_CELLS: tuple[str, ...] = ()

NOTEBOOK = EXAMPLES / "two-decision-rules.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """Assert the readings of the page against the executed namespace."""
    frame, truth = namespace["frame"], namespace["truth"]
    learned = namespace["learned"]

    # The data step: the counts the reading quotes, and "no fixed plan does well".
    assert int(frame["tracked_14"].sum()) == 1821
    assert int(frame["home_30"].notna().sum()) == 1691
    fixed = [truth[key] for key in ("ey_neither", "ey_navigation_only")]
    fixed += [truth[key] for key in ("ey_telemonitoring_only", "ey_both")]
    assert max(fixed) == truth["ey_navigation_only"] < truth["ey_best_rules"]

    # "Those who got both offers stayed home more often than both offers would keep at home",
    # the one relation of the association that holds on every swept draw (plan review, finding 3);
    # "more ready than the patients on any other path"; their day-14 score is below average.
    summary = namespace["summary"]
    assert summary.loc["both", "stayed home"] > truth["ey_both"]
    assert summary["readiness"].idxmax() == "both"
    assert summary.loc["both", "self_care_14"] < frame["self_care_14"].mean()

    # The protocol step prints the record, and no result carries it: a learned regimen is fitted
    # with LTMLE directly (example-notebooks.md, the exception beside plots).
    assert_protocol_recorded(NOTEBOOK, "protocol", namespace["protocol"])
    assert "learned" in namespace["protocol"].treatment_strategies[0]

    # The estimate: its interval covers the true value of the ten learned pairs, it reads
    # supported, and the true value sits near the best pair's and above the best fixed plan's.
    value = learned["ey_regimen[learned regimen]"]
    assert value.support.label == "supported"
    true_target = namespace["true_target"]
    assert covers(value, true_target)
    assert true_target < truth["ey_best_rules"]
    assert abs(true_target - truth["ey_best_rules"]) < 0.01
    assert true_target > truth["ey_navigation_only"] + 0.05
    assert learned.learned_regimen.n_folds == 10
    _check_true_target_against_the_study(frame, learned, true_target)

    # The rule check: the rules offer navigation to nobody at readiness 0.2 or below, to most
    # patients above 0.4, and less often to patients living alone; telemonitoring starts at a
    # higher score after navigation.
    rules, reached = namespace["rules"], namespace["reached"]
    first = rules["navigation rule"]
    assert first[frame["readiness"] <= 0.2].sum() == 0
    assert first[frame["readiness"] > 0.4].mean() > 0.5
    above = frame["readiness"] > 0.2
    assert (
        first[above & (frame["lives_alone"] == 1)].mean()
        < first[above & (frame["lives_alone"] == 0)].mean()
    )
    started = namespace["started"].mean().unstack()
    assert (started.iloc[:, 1] <= started.iloc[:, 0]).all()
    nodes = learned.learned_regimen.nodes
    assert round(nodes[0].rule_disagreement, 3) == 0.053
    assert round(nodes[1].rule_disagreement, 3) == 0.079
    # The rules each patient follows are the stored ones of the fold that held them out.
    folds = learned.folds
    for fold, (_, held_out) in enumerate(folds):
        np.testing.assert_array_equal(
            rules.to_numpy()[held_out], learned.learned_regimen.fold_rules[fold][held_out]
        )
    assert reached.sum() > 0

    # The weights check: no weight was cut, and the quoted day-14 numbers.
    support = namespace["support"]
    assert (support["share_truncated"] == 0).all()
    day_14 = support.set_index("time").loc[2]
    assert int(day_14["n_followed"]) == 555
    assert round(float(day_14["max_weight"]), 2) == 6.09
    printed = stored_output(NOTEBOOK, "assessment")
    assert "score_equations    passed" in printed
    # "Refutation checks and the curve of weight caps do not run on this fit": each refits.
    ledger = namespace["ledger"].set_index("check")
    for check_name in ("refute", "truncation_curve"):
        assert ledger.loc[check_name, "status"] == "unavailable"
        assert "relearns the rules" in ledger.loc[check_name, "detail"]
    # "No check of a missing confounder runs on this fit yet".
    for check_name in ("omitted_confounding", "evalue", "simulated_confounding"):
        assert ledger.loc[check_name, "status"] == "unavailable"

    # Why the first rule depends on the second: the cutoff that ignores telemonitoring is far
    # below the best rule's.
    assert truth["navigation_cutoff_without_telemonitoring_with_others"] == pytest.approx(-0.05)
    assert truth["navigation_cutoff_with_others"] > 0.2
    assert round(float(namespace["between"].mean()), 3) == 0.129

    # Balance: weighting shrinks every confounding gap below 0.03 and every censoring gap below
    # 0.08 (absolute), and the feedback row grows, the code's own effect.
    balance = namespace["balance"]
    by_kind = balance.groupby("diagnostic")
    assert by_kind.get_group("confounding")["after"].abs().max() <= 0.03
    assert by_kind.get_group("censoring")["after"].abs().max() <= 0.085
    assert by_kind.get_group("confounding")["before"].abs().max() == pytest.approx(0.50, abs=0.005)
    feedback = by_kind.get_group("feedback")
    assert (feedback["after"].abs() > feedback["before"].abs()).all()

    # Influence: kish counts fewer rows than the fit holds, and fewer than the weights check's
    # effective size of the followers.
    influence = namespace["influence"]
    kish = float(influence["kish"].iloc[0])
    assert kish < float(day_14["effective_n"]) < len(frame)


def _check_true_target_against_the_study(frame: Any, learned: Any, true_target: float) -> None:
    """The page's true value equals the study harness's truth of the same fit."""
    from tests.studies import _learned_regimen_law as study

    names = {
        "readiness": "W1",
        "lives_alone": "W2",
        "navigation": "A1",
        "tracked_14": "C1",
        "self_care_14": "L2",
        "telemonitoring": "A2",
        "tracked_30": "C2",
        "home_30": "Y",
    }
    study_frame = frame.rename(columns=names)
    fitted = study.fit(study_frame, "declared", 11)
    np.testing.assert_array_equal(
        fitted.learned_regimen.fold_rules, learned.learned_regimen.fold_rules
    )
    assert study.harness(fitted, study_frame, "declared").truth == pytest.approx(
        true_target, abs=1e-12
    )
