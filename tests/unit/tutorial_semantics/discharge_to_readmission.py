"""The reviewed semantic callback for ``docs/examples/discharge-to-readmission``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`, and the slow tier adds :func:`check_refits`.
Each assertion pins a reading the prose makes about this draw.  A relation on one draw certifies
no method; the repeated-sampling numbers the prose quotes come from the committed review probe,
which :func:`_check_probe_log` reads.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from tests.unit.tutorial_semantics import (
    EXAMPLES,
    ROOT,
    assert_protocol_recorded,
    changed_fields,
    covers,
)

NOTEBOOK = EXAMPLES / "discharge-to-readmission.ipynb"

_PROBE_NAME = "reviews/notebook-review/probes/discharge-to-readmission/probe.log"
_PROBE = ROOT / _PROBE_NAME

#: Decimals the readings quote from the committed review probe, which no cell prints.
UNPRINTED_DECIMALS = dict.fromkeys(
    ("-0.1294", "-0.1556", "0.9600", "0.9915", "0.7131"),
    f"a value of the review probe log, {_PROBE_NAME}",
)


def _check_probe_log() -> None:
    """Each quoted probe number appears in the committed probe log, in its line."""
    text = _PROBE.read_text(encoding="utf-8")
    lines = {
        "-0.1294": "simple difference: mean -0.1294",
        "-0.1556": "fitted difference: mean -0.1556",
        "0.9600": "coverage 0.9600",
        "0.9915": "teams declared: mean -0.1296, SD 0.0258, mean SE 0.0256, SE over SD 0.9915",
        "0.7131": "SE over SD 0.7131",
    }
    for literal, line in lines.items():
        assert literal in UNPRINTED_DECIMALS
        assert line in text, f"{line!r} is not in {_PROBE_NAME}"
    # "the team interval contained the true value in about 95% of them and the independent one in
    # only 83%".
    assert "SE over SD 0.9915, coverage 0.9525" in text
    assert "SE over SD 0.7131, coverage 0.8275" in text
    assert round(100 * 0.9525) == 95 and round(100 * 0.8275) == 83
    # "about a sixth of the effect short, and more data would not remove that gap": the bias of
    # the simple difference is a sixth of the true value, and the fitted difference has almost none.
    assert "bias 0.0246" in text and "true value -0.1539" in text
    assert 0.14 < 0.0246 / 0.1539 < 0.19
    assert "bias -0.0017" in text


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth, known = namespace["frame"], namespace["truth"], namespace["known"]
    assert frame.shape == (2_000, 5)
    assert list(frame.columns) == ["lives_alone", "prior_admissions", "offer", "readmitted", "day"]
    assert set(frame["prior_admissions"]) == {0.0, 1.0, 2.0, 3.0}
    # "such as 9.3 for row 1"; the other head rows are a dropout after day 30 and two patients
    # who reached day 90 without a readmission.
    assert frame.loc[1, "readmitted"] == 1 and frame.loc[1, "day"] == pytest.approx(9.3, abs=0.05)
    assert frame.loc[2, "readmitted"] == 0 and frame.loc[2, "day"] == 30.0
    for row in (0, 3):
        assert frame.loc[row, "readmitted"] == 0 and frame.loc[row, "day"] == 90.0
    # Every censoring time sits on a check day, which the grid requires.
    censored = frame.loc[frame["readmitted"].eq(0), "day"]
    assert set(censored) <= set(namespace["checks"])
    left = frame["readmitted"].eq(0) & frame["day"].lt(90.0)
    assert int(left.sum()) == 275
    for day, k in namespace["horizons"].items():
        row = known.set_index("day").loc[day]
        assert row["difference"] == truth[f"ate_regimen[arm1 vs arm0 @ t={k}]"]
        assert row["offered"] - row["not_offered"] == pytest.approx(row["difference"], abs=1e-12)


def _check_association(namespace: dict[str, Any]) -> None:
    frame, known, naive = namespace["frame"], namespace["known"], namespace["naive"]
    means = frame.groupby("offer")[["lives_alone", "prior_admissions"]].mean()
    # "Offered patients more often lived alone ... and had more prior admissions".
    assert (means.loc[1.0] > means.loc[0.0]).all()
    # "which pulls the simple difference toward zero" at every day, on this draw.
    for day, value in naive.items():
        assert known.set_index("day").loc[day, "difference"] < value < 0.0


def _check_first_fit(namespace: dict[str, Any]) -> None:
    from cleverly.datasets import navigation_protocol

    protocol, result = namespace["protocol"], namespace["result"]
    assert changed_fields(protocol, navigation_protocol()) == {
        "outcome",
        "horizon",
        "intercurrent_event_handling",
    }
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, result)
    assert namespace["design"].grid == (15.0, 30.0, 60.0, 90.0)
    for name in result.to_frame()["estimand"]:
        support = result[name].support
        assert support.label == "supported" and support.row == "G14.continuous_time"
    assert result.data.n_times == 4 and result.data.treatment_held

    differences, rows = namespace["differences"], namespace["rows"]
    previous = 0.0
    for day, row in zip(namespace["horizons"], rows, strict=True):
        difference = differences[day]
        # "lowers readmission at every check ... and no interval comes near zero".
        assert difference.ci[1] < -0.05
        assert difference.psi < previous
        previous = difference.psi
        # At a glance: "each interval contains its true value".
        assert covers(difference, row["true_difference"])
        names = [namespace["risk"]("offered", day), namespace["risk"]("not offered", day)]
        assert difference.psi == pytest.approx(
            result[names[0]].psi - result[names[1]].psi, abs=1e-12
        )
    # "the fitted day-90 difference lies farther from the truth than the simple one, though inside
    # its interval".
    day90, true90 = differences[90.0], rows[-1]["true_difference"]
    assert abs(day90.psi - true90) > abs(namespace["naive"][90.0] - true90)
    assert covers(day90, true90)


def _check_assessment(namespace: dict[str, Any]) -> None:
    assessment, support = namespace["assessment"], namespace["support"]
    assert not assessment.attention
    ledger = assessment.to_frame().set_index("check")["status"]
    assert ledger["score_equations"] == "passed"
    # "The largest weight is 4.3, among the patients not offered in the last interval".  Node 4
    # is the interval from day 60 to day 90; `n_followed` counts the patients at risk entering it.
    top = support.loc[support["max_weight"].idxmax()]
    assert top["regimen"] == "not offered" and top["time"] == 4
    assert round(float(top["max_weight"]), 1) == 4.3
    assert (support["share_truncated"] == 0.0).all()
    assert namespace["ratio"] >= 0.915
    frame = namespace["frame"]
    # "515 offered patients were still in follow-up and not yet readmitted when the last interval
    # ... began".
    entering = support.set_index(["regimen", "time"]).loc[("offered", 4), "n_followed"]
    assert entering == 515 == int((frame["offer"].eq(1) & frame["day"].gt(60.0)).sum())
    assert bool(support["converged"].all())


def _check_teams(namespace: dict[str, Any]) -> None:
    from cleverly.datasets import longitudinal_navigation_protocol

    teams, team_truth = namespace["teams"], namespace["team_truth"]
    assert teams.shape == (4_000, 9)
    assert teams["team"].nunique() == 100 and teams.groupby("team").size().eq(40).all()
    # "as in row 2"; "Row 3 left follow-up between day 30 and day 90".
    assert teams.loc[2, "readmitted_30"] == 1 and teams.loc[2, "readmitted_90"] == 1
    assert teams.loc[3, "followed_30"] == 1 and teams.loc[3, "followed_90"] == 0
    assert math.isnan(teams.loc[3, "readmitted_90"])

    team_protocol, team_result = namespace["team_protocol"], namespace["team_result"]
    assert changed_fields(team_protocol, longitudinal_navigation_protocol()) == {
        "treatment_strategies",
        "treatment_versions",
        "outcome",
        "horizon",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert_protocol_recorded(NOTEBOOK, "team-identify", team_protocol, team_result)

    clustered, independent = namespace["clustered"], namespace["independent"]
    truth = team_truth["ate_regimen[always vs never]"]
    assert namespace["team_target_truth"] == truth
    assert clustered.support.label == "supported" and clustered.support.row == "G05.cross_fitted"
    assert independent.support.label == "supported" and independent.support.row == "G02"
    assert team_result.data.cluster is not None and len(np.unique(team_result.data.cluster)) == 100
    # "each interval contains its true value", and the effect is clear of zero.
    assert covers(clustered, truth) and clustered.ci[1] < 0.0
    # "nearly the same estimate, but the independent fit's standard error is 0.72 times".
    assert independent.psi == pytest.approx(clustered.psi, abs=0.002)
    assert round(namespace["se_ratio"], 2) == 0.72
    low, high = independent.ci
    assert clustered.ci[0] < low and high < clustered.ci[1]
    # "The weight check still finds nothing that needs attention, no patient at the floor, and
    # every group worth at least 64.3% of its size."
    assert not namespace["team_assessment"].attention
    team_support = namespace["team_support"]
    assert (team_support["share_truncated"] == 0.0).all()
    assert round(namespace["team_ratio"], 3) == 0.643
    # "The largest weight, 58.2 among patients who got neither".
    top = team_support.loc[team_support["max_weight"].idxmax()]
    assert top["regimen"] == "neither" and round(float(top["max_weight"]), 1) == 58.2


def _check_evalue(namespace: dict[str, Any]) -> None:
    bound, result, risk = namespace["bound"], namespace["result"], namespace["risk"]
    offered, not_offered = (result[risk(plan, 90.0)].psi for plan in ("offered", "not offered"))
    # "the offered risk is 0.714 times the not-offered risk"; "2.15 ... and 1.87".
    assert bound.scale == "risk ratio" and not bound.approximate
    assert bound.risk_ratio == pytest.approx(offered / not_offered, rel=1e-9)
    assert round(bound.risk_ratio, 3) == 0.714
    assert round(bound.point, 2) == 2.15 and round(bound.limit, 2) == 1.87
    # The limit is the interval end nearest no effect, which a ratio below 1 puts at the top.
    assert bound.risk_ratio_ci[1] < 1.0 and bound.limit < bound.point
    # "the missing confounder does not change who leaves follow-up"; "read it as a description".
    assert "does not affect censoring" in bound.note
    assert "read it as descriptive" in str(bound.summary())


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    moves = namespace["moves"]
    by_name = {(move["fit"], move["left out"]): move["move in SEs"] for move in moves}
    # "Leaving out discharge_risk moves the team estimate ... 4.1 standard errors"; "Every other
    # move is under one standard error, and in the first program the larger one is ... for
    # prior_admissions".
    assert by_name["team program", "discharge_risk"] > 4.0
    others = [abs(v) for k, v in by_name.items() if k != ("team program", "discharge_risk")]
    assert max(others) < 1.0
    first = {k[1]: v for k, v in by_name.items() if k[0] == "first program"}
    assert max(first, key=lambda name: abs(first[name])) == "prior_admissions"


def check(namespace: dict[str, Any]) -> None:
    """Pin the readings of the two fits on their seeded draws."""
    _check_probe_log()
    _check_data(namespace)
    _check_association(namespace)
    _check_first_fit(namespace)
    _check_assessment(namespace)
    _check_evalue(namespace)
    _check_teams(namespace)
    _check_sensitivity(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The refutation and generated-outcome refits of each fit's day-90 difference."""
    contrast_fit, day90_name = namespace["contrast_fit"], namespace["day90_name"]
    day90 = namespace["day90"]
    # The typed contrast that `refute` reads is the `result.contrast` difference.
    assert contrast_fit[day90_name].psi == pytest.approx(day90.psi, abs=1e-6)
    assert contrast_fit[day90_name].std_error == pytest.approx(day90.std_error, abs=1e-6)
    reports = namespace["refutations"]
    assert list(reports) == ["first program", "team program"]
    for report in reports.values():
        tests = {test.name: test for test in report.tests}
        assert list(tests) == ["placebo", "subset", "random_common_cause"]
        for test in tests.values():
            assert test.verdict == "descriptive" and test.passed is None
            assert test.n_replicates == 20
        # "The random offers give means near zero, and the noise column barely moves either".
        assert abs(tests["placebo"].mean) < 0.01
        assert tests["placebo"].statistic_pvalue > 0.05
        assert abs(tests["random_common_cause"].statistic) < 0.05
        # "spread about as the standard error predicts ... 20 refits leave wide room".
        assert 0.7 < tests["subset"].statistic < 1.4
        # The study behind the verdict used out-of-fold predictions and no teams.
        assert "out-of-fold predictions in 5 folds" in tests["placebo"].evidence

    generated = namespace["generated_tests"]
    assert list(generated) == ["dummy_outcome", "simulated_outcome"]
    for test in generated.values():
        assert test.verdict == "descriptive" and test.n_replicates == 20
        # "Each mean estimate lands near its true value"; "Twenty refits give intervals far
        # wider than that zone".
        bias = test.readings[0]
        assert bias.lower < 0.0 < bias.upper
        assert bias.lower < bias.band[0] and bias.band[1] < bias.upper
        assert abs(test.mean - test.declared_effect) < 0.01
    assert generated["dummy_outcome"].declared_effect == 0.0
    # "A verdict needs at least 400 refits, and no simulation study yet backs one for a fit with
    # several checks, so even 400 would read `descriptive`".
    from cleverly.validation import EquivalenceBands

    assert EquivalenceBands().minimum_draws == 400
    for test in generated.values():
        assert "no simulation study backs a verdict for this kind of fit" in test.evidence
