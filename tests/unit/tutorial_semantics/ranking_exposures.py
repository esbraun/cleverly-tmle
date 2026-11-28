"""The reviewed semantic callback for ``docs/examples/ranking-exposures``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`, and the slow tier passes it to :func:`check_refits` as
well.  Each assertion witnesses one claim the prose makes about the law or about this draw.  A
relation on this draw certifies no method; the registered studies do that, and the page's seed
sweep (``reviews/notebook-review/probes/ranking-exposures``) gives the repeated-draw rates the
chain reading quotes. The sweep's multiplicity rates read the Benjamini-Hochberg adjustment the
page used before the step-down max-T became the default, so the page no longer quotes them.
"""

from __future__ import annotations

import re
from typing import Any

from cleverly.datasets import PRACTICES, discharge_practices_protocol, discharge_practices_truth
from cleverly.variable_importance import PENDING_REASON
from tests.documents import ROOT
from tests.unit.tutorial_semantics import EXAMPLES, assert_protocol_recorded, covers

#: The seed sweep that the chain reading quotes.
_SWEEP = "reviews/notebook-review/probes/ranking-exposures/sweep.log"

#: Decimals the readings write that no stored output prints, each with its reason.
UNPRINTED_DECIMALS: dict[str, str] = {}

#: Decimals a reading quotes from a code cell outside its own section.
NARRATION_SOURCES: dict[str, dict[str, str]] = {}

#: The refit cells the rule derives from this notebook today, in notebook order.
REFIT_CELLS = ("sensitivity", "refutation")

NOTEBOOK = EXAMPLES / "ranking-exposures.ipynb"

NAVIGATION = "transition_navigation"
NULLS = ("teach_back", "home_health_referral")


def _sweep_rate(label: str) -> tuple[int, int]:
    text = (ROOT / _SWEEP).read_text()
    match = re.search(rf"^{re.escape(label)}: (\d+)/(\d+)", text, re.MULTILINE)
    assert match, label
    return int(match.group(1)), int(match.group(2))


def _check_law(namespace: dict[str, Any]) -> None:
    truth = namespace["truth"]
    assert truth == discharge_practices_truth()
    assert namespace["candidates"] == list(PRACTICES)
    # Only navigation's two effects differ: the booking is the one practice another causes.
    for name in PRACTICES:
        gap = truth[f"total[{name}]"] - truth[f"adjusted[{name}]"]
        assert (abs(gap) > 0.03) == (name == NAVIGATION), name
    for name in NULLS:
        assert truth[f"total[{name}]"] == 0.0


def _check_simple_comparison(namespace: dict[str, Any]) -> None:
    simple = namespace["simple"]
    # The referral looks harmful and navigation looks harmful, though neither harms.
    assert simple.loc["home_health_referral", "simple difference"] < -0.1
    assert simple.loc[NAVIGATION, "simple difference"] < 0.0
    for name in ("home_health_referral", NAVIGATION):
        assert (
            simple.loc[name, "mean discharge_risk, got it"]
            > simple.loc[name, "mean discharge_risk, did not"]
        )


def _check_protocol(namespace: dict[str, Any]) -> None:
    # The protocol step prints the record, both rankings record it, and every candidate's fit
    # carries its digest.
    protocol = namespace["protocol"]
    assert protocol == discharge_practices_protocol()
    rankings = (namespace["ranking"], namespace["offer_first"])
    assert all(ranking.protocol is protocol for ranking in rankings)
    fits = [fit for ranking in rankings for fit in ranking.fits.values()]
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, *fits)


def _check_ranking(namespace: dict[str, Any]) -> None:
    ranking = namespace["ranking"]
    table = namespace["table"]
    truth = namespace["truth"]
    assert namespace["planned"].total_fits == 50
    # Each default fit adjusts for the three characteristics and the other four practices.
    covariates = namespace["covariates"]
    for name, adjusted_for in ranking.to_frame().set_index("candidate")["adjustment_set"].items():
        others = {other for other in PRACTICES if other != name}
        assert set(adjusted_for.split(", ")) == set(covariates) | others, name
    frame = ranking.to_frame()
    # Sorted by adjusted p-value, not by size.
    adjusted = list(frame["p_value_adjusted"])
    assert adjusted == sorted(adjusted)
    assert list(frame["candidate"][:2]) == ["followup_booked", "medication_review"]
    # Every row prints without an experimental mark, so it reads supported.
    assert "support" not in frame.columns
    for entry in ranking:
        assert covers(entry.estimate, truth[f"adjusted[{entry.candidate}]"]), entry.candidate
    assert bool(table["covers truth"].all())
    for name in NULLS:
        assert table.loc[name, "p_value_adjusted"] > 0.05
    # The multiplicity witness: navigation's raw p-value is below 0.05 and its adjusted one not.
    assert table.loc[NAVIGATION, "p_value"] < 0.05 < table.loc[NAVIGATION, "p_value_adjusted"]


def _check_chain(namespace: dict[str, Any]) -> None:
    table = namespace["table"]
    second = namespace["second"]
    truth = namespace["truth"]
    total = truth[f"total[{NAVIGATION}]"]
    held = truth[f"adjusted[{NAVIGATION}]"]
    offer = second.loc[NAVIGATION]
    assert offer["ci_lower"] <= total <= offer["ci_upper"]
    assert not offer["ci_lower"] <= held <= offer["ci_upper"]
    assert offer["psi"] > table.loc[NAVIGATION, "psi"]
    # "In this sample the first interval also contains 0.0506".
    first = table.loc[NAVIGATION]
    assert first["ci_lower"] <= total <= first["ci_upper"]
    navigation_set = second.loc[NAVIGATION, "adjustment_set"]
    assert navigation_set.split(", ") == namespace["covariates"]
    booking_set = second.loc["followup_booked", "adjustment_set"]
    assert booking_set.split(", ") == [*namespace["covariates"], NAVIGATION]
    booking = abs(second.loc["followup_booked", "psi"] - table.loc["followup_booked", "psi"])
    assert booking < 0.001
    # "it missed 0.0506 on 254" of the 400 swept draws.
    assert _sweep_rate("default navigation interval excludes the total truth") == (254, 400)
    # Both single-draw relations are qualified "in this sample": they fail on many draws.
    assert _sweep_rate("second-fit navigation interval excludes the adjusted truth")[0] < 360
    assert _sweep_rate("all five default intervals cover their truth")[0] < 360


def _check_assessment(namespace: dict[str, Any]) -> None:
    reports = namespace["reports"]
    smallest = {
        name: report.propensity_quantiles["overall"][0.0] for name, report in reports.items()
    }
    assert min(smallest, key=lambda name: smallest[name]) == "home_health_referral"
    ratios = {
        name: report.effective_sample_size["treated"]["ratio"] for name, report in reports.items()
    }
    assert min(ratios, key=lambda name: ratios[name]) == "home_health_referral"
    # "which its standard error, the largest of the five, already shows".
    table = namespace["table"]
    assert table["std_err"].idxmax() == "home_health_referral"
    # The one check that needs attention, and the simulated-confounder test it skips.
    assessment = namespace["ranking"].assess()
    assert [item.name for item in assessment.attention] == ["omitted_confounding"]
    rows = assessment.to_frame().set_index("check")
    assert rows.loc["simulated_confounding", "status"] == "unavailable"


def _check_multiplicity(namespace: dict[str, Any]) -> None:
    # The default adjustment is the step-down max-T, and its claim reads experimental only
    # because the study that measures it has not run: this run is the study's configuration.
    ranking = namespace["ranking"]
    assert (ranking.method, ranking.error_rate) == ("step-down max-T", "fwer")
    support = ranking.adjustment_support
    assert support.label == "experimental"
    assert support.reason == PENDING_REASON


def check(namespace: dict[str, Any]) -> None:
    """The claims the readings make, on the fast tier."""
    _check_law(namespace)
    _check_simple_comparison(namespace)
    _check_protocol(namespace)
    _check_ranking(namespace)
    _check_chain(namespace)
    _check_assessment(namespace)
    _check_multiplicity(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The claims of the refit cells, on the slow tier."""
    rows = namespace["rows"]
    # The booking and the review keep a positive low end at the strongest recorded
    # characteristic; navigation's total effect does not.
    assert rows["followup_booked"]["bound, low end"] > 0
    assert rows["medication_review"]["bound, low end"] > 0
    assert rows["navigation, booking follows"]["bound, low end"] < 0
    assert rows["navigation, booking follows"]["strongest characteristic"] == "discharge_risk"
    refutation = namespace["refutation"].to_frame()
    assert set(refutation["verdict"]) == {"descriptive"}
    noise = refutation[refutation["test"] == "random_common_cause"]
    assert (noise["refuted_mean"] - noise["original"]).abs().max() < 0.0002
    placebo = refutation[refutation["test"] == "placebo"]
    assert ((placebo["refuted_mean"]).abs() < 2 * placebo["refuted_sd"]).all()
    subset = refutation[refutation["test"] == "subset"]
    for report in namespace["refutation"].values():
        for test in report.tests:
            if test.name == "subset":
                assert 0.75 < test.statistic < 1.25, test.statistic
    assert len(subset) == len(PRACTICES)
