"""The reviewed semantic callback for ``docs/examples/ranking-exposures``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`, and the slow tier passes it to :func:`check_refits` as
well.  Each assertion witnesses one claim the prose makes about the law or about this draw.  A
relation on this draw certifies no method; the registered studies do that, and the page's seed
sweep (``reviews/notebook-review/probes/ranking-exposures``) gives the repeated-draw rates the
readings quote.
"""

from __future__ import annotations

import re
from typing import Any

from cleverly.datasets import PRACTICES, discharge_practices_truth
from tests.documents import ROOT
from tests.unit.tutorial_semantics import EXAMPLES, covers

#: The seed sweep that the multiplicity lesson and the chain reading quote.
_SWEEP = "reviews/notebook-review/probes/ranking-exposures/sweep.log"

#: Decimals the readings write that no stored output prints, each with its reason.
UNPRINTED_DECIMALS = {
    "multiplicity-reading: 0.0158": f"the mean false discovery proportion over 400 draws: {_SWEEP} "
    "(check asserts it)",
    "multiplicity-reading: 0.0035": f"its standard error over 400 draws: {_SWEEP} (check asserts it)",
    "multiplicity-reading: 0.02": "the Benjamini-Hochberg bound q m0 / m = 0.05 x 2 / 5, written "
    f"in the reading and in {_SWEEP}",
}

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


def _check_ranking(namespace: dict[str, Any]) -> None:
    ranking = namespace["ranking"]
    table = namespace["table"]
    truth = namespace["truth"]
    assert namespace["planned"].total_fits == 50
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
    booking = abs(second.loc["followup_booked", "psi"] - table.loc["followup_booked", "psi"])
    assert booking < 0.001
    # "it missed 0.0506 on 254" of the 400 swept draws.
    assert _sweep_rate("default navigation interval excludes the total truth") == (254, 400)


def _check_assessment(namespace: dict[str, Any]) -> None:
    support = namespace["support"]
    smallest = {
        name: report.propensity_quantiles["overall"][0.0] for name, report in support.items()
    }
    assert min(smallest, key=smallest.get) == "home_health_referral"
    ratios = {
        name: report.effective_sample_size["treated"]["ratio"] for name, report in support.items()
    }
    assert min(ratios, key=ratios.get) == "home_health_referral"


def _check_multiplicity() -> None:
    assert _sweep_rate("some no-effect practice at raw p below 0.05") == (33, 400)
    assert _sweep_rate("some no-effect practice at adjusted p below 0.05") == (20, 400)
    text = (ROOT / _SWEEP).read_text()
    assert "mean false discovery proportion: 0.0158 (standard error 0.0035)" in text
    assert "2 of 5 nulls: 0.0200" in text


def check(namespace: dict[str, Any]) -> None:
    """The claims the readings make, on the fast tier."""
    _check_law(namespace)
    _check_simple_comparison(namespace)
    _check_ranking(namespace)
    _check_chain(namespace)
    _check_assessment(namespace)
    _check_multiplicity()


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
    assert (noise["refuted_mean"] - noise["original"]).abs().max() < 0.0003
