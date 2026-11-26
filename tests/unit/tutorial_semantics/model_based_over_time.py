"""The reviewed semantic callback for ``docs/examples/model-based-over-time``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals
are also compared against its stored outputs, and every decimal it writes is printed or listed in
:data:`UNPRINTED_DECIMALS` with its source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cleverly.datasets import (
    make_recovery_score,
    recovery_offer_count,
    recovery_score_protocol,
    recovery_score_truth,
)
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refute",)

NOTEBOOK = EXAMPLES / "model-based-over-time.ipynb"

_PROBE = "reviews/notebook-review/probes/model-based-over-time"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.43": f"the largest gap between the g-formula's and the blip model's gains, 400 draws: "
    f"{_PROBE}/sweep.csv; checked by _check_probe_log",
    "0.96": f"the smallest gap of the simple-model g-formula's gain above the blip model's, 400 "
    f"draws: {_PROBE}/sweep.csv; checked by _check_probe_log",
    "1.165": "half the width of the g-formula's printed interval for the gain, 9.49 to 11.82; "
    "checked by _check_probe_log",
    "1.66": "the simple-model g-formula's gain minus the right fit's, 12.32 - 10.66 (interpret); "
    "checked by _check_failure_mode",
    "3.0": f"the largest simulation share of a plan mean's variance, in percent, 400 draws: "
    f"{_PROBE}/sweep.log",
    "0.7": f"the largest simulation share of the difference's variance, in percent, 400 draws: "
    f"{_PROBE}/sweep.log",
    "2.26": f"the mean error of the wrong models' mean under always, 400 draws: {_PROBE}/sweep.log",
    "0.05": f"the bound on the mean error of every blip-model effect, 400 draws: {_PROBE}/sweep.log",
}

#: Decimals a summary cell quotes from a lesson under "Go deeper", by the cell that prints them.
NARRATION_SOURCES = {"interpret": {"0.55": "targeted", "0.82": "targeted"}}

DRAWS = 400
LEVELS = ("ey_regimen[never]", "ey_regimen[always]")
CONTRAST = "ate_regimen[always vs never]"
TARGETED = "ate_regimen[targeted vs never]"
OFFERS = ("offer_discharge", "offer_day10", "offer_day20")


def check(namespace: dict[str, Any]) -> None:
    """The tutorial keeps its law, its protocol, its fits, and its seeded narrative."""
    _check_data(namespace)
    _check_design(namespace)
    _check_gformula(namespace)
    _check_failure_mode(namespace)
    _check_blips(namespace)
    _check_compare(namespace)
    _check_assessment(namespace)
    _check_sensitivity(namespace)
    _check_targeted(namespace)
    _check_blip_overlap(namespace)
    _check_bootstrap(namespace)
    _check_probe_log(namespace)
    _check_prose_phrases()


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    assert frame.shape == (4_000, 8)
    assert truth == make_recovery_score(10, seed=0)[1]
    assert [round(truth[key], 2) for key in (*LEVELS, CONTRAST)] == [49.24, 60.46, 11.22]
    # "0.624 of the patients who live alone got the discharge offer, against 0.132".
    shares = frame.groupby("lives_alone")["offer_discharge"].mean()
    assert round(shares[1.0], 3) == 0.624 and round(shares[0.0], 3) == 0.132
    # "The simple difference is -1.83 points", with the offered patients unlike the others.
    summary = namespace["summary"]
    assert round(namespace["crude"], 2) == -1.83
    offered, never = summary["offered all three times"], summary["never offered"]
    assert round(offered["lives alone"], 3) == 0.754 and round(never["lives alone"], 3) == 0.110
    assert round(offered["discharge risk"], 3) == 0.638
    assert round(never["discharge risk"], 3) == 0.389
    assert offered["symptoms on day 10"] > never["symptoms on day 10"]
    # The crude difference is the difference of the printed means.
    assert namespace["crude"] == pytest.approx(
        offered["mean score on day 30"] - never["mean score on day 30"]
    )


def _check_design(namespace: dict[str, Any]) -> None:
    protocol, study, result = namespace["protocol"], namespace["study"], namespace["result"]
    assert protocol == recovery_score_protocol()
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, result)
    # The day-10 offer reads the day-10 flag, and the day-20 offer both flags.
    assert "symptoms_day10" in study.data.history_names(2)
    assert {"symptoms_day10", "symptoms_day20"} <= set(study.data.history_names(3))
    assert "symptoms_day20" not in study.data.history_names(2)
    assert "parametric_gformula" in stored_output(NOTEBOOK, "identify")


def _support(estimates: Any) -> set[tuple[str, str | None]]:
    return {(e.support.label, getattr(e.support, "row", None)) for e in estimates}


def _check_gformula(namespace: dict[str, Any]) -> None:
    result, means, truth = namespace["result"], namespace["means"], namespace["truth"]
    gain = result[CONTRAST]
    assert list(result.estimates) == [CONTRAST]
    assert [round(means[key].psi, 2) for key in LEVELS] == [49.37, 60.03]
    assert [round(value, 2) for value in gain.ci] == [9.49, 11.82]
    assert round(gain.psi, 2) == 10.66
    # The gain of the contrast fit is the difference of the plan means of the same simulation.
    assert gain.psi == pytest.approx(means[LEVELS[1]].psi - means[LEVELS[0]].psi, abs=1e-9)
    # "all three intervals hold".
    for key in LEVELS:
        assert covers(means[key], truth[key])
    assert covers(gain, truth[CONTRAST])
    assert _support([*(means[key] for key in LEVELS), gain]) == {("supported", "G29.g2_g5")}
    # The simulation's share of each variance, smallest for the difference.
    shares = result.simulation.monte_carlo_share
    assert [round(100 * shares[key], 1) for key in (*LEVELS, CONTRAST)] == [2.7, 1.2, 0.5]
    assert shares[CONTRAST] < min(shares[key] for key in LEVELS) < 0.10
    printed = stored_output(NOTEBOOK, "gformula")
    assert "never        812 of 4,000" in printed and "always       495 of 4,000" in printed
    # "the summary's note on the natural-treatment model" names a note the summary prints.
    assert "natural-treatment model" in printed


def _check_failure_mode(namespace: dict[str, Any]) -> None:
    wrong, truth, result = namespace["wrong"], namespace["truth"], namespace["result"]
    assert namespace["gformula"].covariate_learner is not None
    gain = wrong[CONTRAST]
    assert gain.psi != result[CONTRAST].psi
    # "the interval for the gain misses: 12.32 (11.30 to 13.34)".
    assert round(gain.psi, 2) == 12.32
    assert [round(value, 2) for value in gain.ci] == [11.30, 13.34]
    # "Simple symptom models moved the answer by 1.66 points".
    assert round(gain.psi - result[CONTRAST].psi, 2) == 1.66
    assert not covers(gain, truth[CONTRAST])
    # "The fit still reads `supported`."
    assert _support([gain]) == {("supported", "G29.g2_g5")}


def _effects(namespace: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(namespace["effect_rows"]).set_index(["offer", "lives_alone"])


def _check_blips(namespace: dict[str, Any]) -> None:
    blips = namespace["blips"]
    effects = _effects(namespace)
    expected = {
        ("offer_discharge", 0): 2.00,
        ("offer_discharge", 1): 13.39,
        ("offer_day10", 0): 2.15,
        ("offer_day10", 1): 3.84,
        ("offer_day20", 0): 2.10,
    }
    for index, value in expected.items():
        assert round(effects.loc[index, "effect"], 2) == value
    # "Four of the five estimates hold their true values"; the day-10 effect when alone misses.
    held = {
        index: bool(row["ci_low"] <= row["truth"] <= row["ci_high"])
        for index, row in effects.iterrows()
    }
    assert [index for index, value in held.items() if not value] == [("offer_day10", 1)]
    assert [round(effects.loc[("offer_day10", 1), c], 2) for c in ("ci_low", "ci_high")] == [
        3.00,
        4.68,
    ]
    assert _support(blips.estimates.values()) == {("supported", "G42.three_nodes")}
    # The blip model's gain is the patients' mean of the summed effects, so the interpret cell's
    # "more than half of the gain comes from the discharge offer, most of that among patients who
    # live alone" reads these rows weighted by the share who live alone.
    share = float(namespace["frame"]["lives_alone"].mean())
    weights = {0: 1.0 - share, 1: share}
    part = {
        index: weights[index[1]] * float(effects.loc[index, "effect"]) for index in effects.index
    }
    gain = blips[CONTRAST].psi
    assert sum(part.values()) == pytest.approx(gain, abs=1e-6)
    discharge = part[("offer_discharge", 0)] + part[("offer_discharge", 1)]
    # "The discharge offer carries 5.98 of the 10.82 points, 4.69 of them from patients who live
    # alone, though they are only 0.350 of the patients."
    assert round(discharge, 2) == 5.98
    assert round(part[("offer_discharge", 1)], 2) == 4.69
    assert round(share, 3) == 0.350
    parts = namespace["parts"]
    assert parts["both groups"].sum() == pytest.approx(gain, abs=1e-6)
    assert discharge / gain > 0.5
    assert part[("offer_discharge", 1)] > part[("offer_discharge", 0)]


def _check_compare(namespace: dict[str, Any]) -> None:
    result, blips, truth = namespace["result"], namespace["blips"], namespace["truth"]
    wrong, tmle = namespace["wrong"], namespace["tmle_fit"]
    # "the g-formula's gain sits 0.16 points below the blip model's; with simple ones it sits 1.50
    # points above".
    assert round(result[CONTRAST].psi - blips[CONTRAST].psi, 2) == -0.16
    assert round(wrong[CONTRAST].psi - blips[CONTRAST].psi, 2) == 1.50
    assert round(blips[CONTRAST].psi, 2) == 10.82
    # LTMLE: in sample, GLM models, the score's known range; "10.42 (9.08 to 11.73), holds the
    # true value" and reads `experimental` beside the G08 row, which differs on outcome.
    ltmle = namespace["ltmle"]
    assert ltmle.cross_fitting.enabled is False
    assert tuple(ltmle.targeting.q_bounds) == (0.0, 100.0)
    gain = tmle[CONTRAST]
    assert round(gain.psi, 2) == 10.42
    assert [round(value, 2) for value in gain.ci] == [9.08, 11.73]
    assert covers(gain, truth[CONTRAST])
    assert gain.support.label == "experimental"
    assert "G08" in gain.support.reason
    assert "outcome" in gain.support.reason and "treatment" in gain.support.reason
    # The figure's companion prints the four intervals it draws.
    assert list(namespace["gains"]) == [
        "g-formula",
        "g-formula, simple symptom models",
        "blip model",
        "LTMLE",
    ]
    printed = stored_output(NOTEBOOK, "compare")
    for label in namespace["gains"]:
        assert label in printed
    check = namespace["check"]
    assert check.names == (CONTRAST,)
    assert round(float(check.structural[0]), 2) == 10.80
    assert round(float(check.nonparametric[0]), 2) == 10.41
    assert round(float(check.pvalue), 2) == 0.29
    # "The check reads `experimental`", printed by the cell; its LTMLE half is off the rows.
    assert check.support.label == "experimental"
    assert "G08" in check.support.reason


def _check_assessment(namespace: dict[str, Any]) -> None:
    assert namespace["attention"] == ()
    printed = stored_output(NOTEBOOK, "assessment")
    assert "always           3        495        63.48       206.5" in printed
    assert "Overlap is thinnest for always at time 3" in " ".join(printed.split())


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    moves = pd.DataFrame(namespace["move_rows"]).set_index("dropped").round(2)
    risk, flags = moves.loc["without discharge risk"], moves.loc["without symptom flags"]
    # "by -6.26 and -6.57 standard errors without the discharge risk, and by -7.36 and -10.83".
    assert risk.tolist() == [-6.26, -6.57]
    assert flags.tolist() == [-7.36, -10.83]
    assert (moves < 0).all().all()


def _check_targeted(namespace: dict[str, Any]) -> None:
    """The targeted plan: both fits hold its exact truth, and LTMLE estimates its offers."""
    rules = namespace["RULES"]
    assert recovery_score_truth(*rules) - recovery_score_truth(0, 0, 0) == pytest.approx(
        6.0765, abs=1e-12
    )
    assert recovery_offer_count(*rules) == pytest.approx(0.82275, abs=1e-12)
    table = namespace["targeted_table"]
    expected = {
        ("g-formula", "gain over never"): (5.86, 5.26, 6.46),
        ("g-formula", "share of the gain of always"): (0.55, 0.49, 0.61),
        ("LTMLE", "gain over never"): (6.00, 5.17, 6.84),
    }
    for index, values in expected.items():
        row = table.loc[index]
        assert tuple(round(float(row[c]), 2) for c in ("estimate", "ci_low", "ci_high")) == values
    assert table["covers"].all()
    assert round(float(table.loc[("g-formula", "gain over never"), "truth"]), 2) == 6.08
    fits = namespace["targeted_fits"]
    # "The g-formula's fits read `supported`": the Gaussian row holds rule cells (G2 and G5).
    assert fits["g-formula"][TARGETED].support.label == "supported"
    assert fits["g-formula"][TARGETED].support.row == "G29.g2_g5"
    # "LTMLE's read `experimental`": no LTMLE study has a continuous score.
    assert fits["LTMLE"][TARGETED].support.label == "experimental"
    # A nonzero witness: the targeted plan is neither fixed plan, so its gain is not always's.
    assert fits["g-formula"][TARGETED].psi < 0.6 * fits["g-formula"][CONTRAST].psi
    offers = namespace["offers"]
    assert [round(value, 2) for value in (offers.psi, *offers.ci)] == [0.82, 0.79, 0.85]
    assert offers.ci[0] <= 0.82275 <= offers.ci[1]


def _check_blip_overlap(namespace: dict[str, Any]) -> None:
    positivity = namespace["blips"].structural_diagnostics
    assert [round(value, 4) for value in positivity.smallest_eigenvalue[:2]] == [0.2758, 0.4277]
    assert all(share == 0.0 for share in positivity.extreme_share)
    quantiles = namespace["quantiles"]
    # "the smallest is 0.075, at the discharge offer": the figure's values, printed beside it.
    assert round(float(quantiles["min"].min()), 3) == 0.075
    assert quantiles["min"].idxmin() == "time 1"


def _check_bootstrap(namespace: dict[str, Any]) -> None:
    boot, derived = namespace["boot"], namespace["derived"]
    assert boot.n_replicates == 100
    assert round(boot.std_error, 2) == 0.58
    assert round(derived.std_error, 2) == 0.59
    assert [round(float(value), 2) for value in boot.ci] == [9.50, 11.86]
    # "The bootstrap interval reads `experimental`".
    assert boot.support.label == "experimental"


def _sweep() -> tuple[str, pd.DataFrame]:
    folder = Path(__file__).resolve().parents[3] / _PROBE
    log = (folder / "sweep.log").read_text()
    frame = pd.read_csv(folder / "sweep.csv", float_precision="round_trip")
    return log, frame


def _check_probe_log(namespace: dict[str, Any]) -> None:
    """The committed sweep holds the repeated-draw numbers the readings quote."""
    log, frame = _sweep()
    assert log.startswith("seeds 3100 to 3499, n = 4000")
    assert len(frame) == DRAWS and frame["seed"].tolist() == list(range(3100, 3100 + DRAWS))
    truth = namespace["truth"]
    shares = [frame[f"gf:{key}:mc_share"].max() for key in LEVELS]
    assert round(100 * max(shares), 1) == 3.0
    assert round(100 * frame[f"gf:{CONTRAST}:mc_share"].max(), 1) == 0.7
    assert round(frame[f"wrong:{LEVELS[1]}"].mean() - truth[LEVELS[1]], 2) == 2.26
    # "differed from the blip model's by at most 0.43 points, while the gain with simple symptom
    # models sat at least 0.96 points above it, and more than 1.165 ... on 380 of 400 draws".
    right = frame[f"gf:{CONTRAST}"] - frame[f"snmm:{CONTRAST}"]
    simple = frame[f"wrong:{CONTRAST}"] - frame[f"snmm:{CONTRAST}"]
    assert round(right.abs().max(), 2) == 0.43
    assert round(simple.min(), 2) == 0.96
    low, high = (round(float(bound), 2) for bound in namespace["result"][CONTRAST].ci)
    half = round((high - low) / 2, 3)
    assert half == 1.165
    assert int((simple > half).sum()) == 380
    for column in frame.columns:
        if column.startswith("snmm:effect:") and not column.endswith(":covers"):
            name = column.removeprefix("snmm:")
            assert abs(frame[column].mean() - frame[f"truth:{name}"].iloc[0]) < 0.05
    # The sweep fits the page: its first row is the draw that the page shows.
    first = frame.iloc[0]
    means, blips = namespace["means"], namespace["blips"]
    for key in LEVELS:
        assert first[f"gf:{key}"] == pytest.approx(means[key].psi, abs=1e-9)
        assert first[f"snmm:{key}"] == pytest.approx(blips[key].psi, abs=1e-9)
    for tag, fit in (("gf", "result"), ("wrong", "wrong"), ("ltmle", "tmle_fit")):
        assert first[f"{tag}:{CONTRAST}"] == pytest.approx(namespace[fit][CONTRAST].psi, abs=1e-9)
    fits = namespace["targeted_fits"]
    assert first[f"gf:{TARGETED}"] == pytest.approx(fits["g-formula"][TARGETED].psi, abs=1e-9)
    assert first[f"ltmle:{TARGETED}"] == pytest.approx(fits["LTMLE"][TARGETED].psi, abs=1e-9)
    assert first["ltmle:ey_regimen[targeted]:offers"] == pytest.approx(
        namespace["offers"].psi, abs=1e-9
    )
    assert first["check:pvalue"] == pytest.approx(namespace["check"].pvalue, abs=1e-9)


def _prose(cell_id: str) -> str:
    """The source of one markdown cell, with each run of whitespace collapsed to a space."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])
    return " ".join(source.split())


def _hits(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].sum())


def _check_prose_phrases() -> None:
    """The readings attach each repeated-draw count to the quantity the sweep gives it.

    A count is an integer, so the narration test does not read it.  This check builds each
    phrase of the readings from the committed sweep, so a count moved to another quantity fails.
    """
    _, frame = _sweep()
    gf = [_hits(frame, f"gf:{key}:covers") for key in (*LEVELS, CONTRAST)]
    assert (
        f"held the true value on {gf[0]} of {DRAWS} draws for `never`, {gf[1]} for `always`, "
        f"and {gf[2]} for the difference"
    ) in _prose("gformula-plans-reading")
    wrong_level = _hits(frame, f"wrong:{LEVELS[1]}:covers")
    wrong_gain = _hits(frame, f"wrong:{CONTRAST}:covers")
    assert wrong_level == 0
    assert (
        f"missed the truth on all {DRAWS} draws, and the interval for the gain held it on only "
        f"{wrong_gain} of {DRAWS}"
    ) in _prose("failure-reading")
    effects = [
        _hits(frame, c)
        for c in frame.columns
        if c.startswith("snmm:effect:") and c.endswith(":covers")
    ]
    assert len(effects) == 6
    assert f"held their true values on {min(effects)} to {max(effects)} of {DRAWS}" in _prose(
        "snmm-reading"
    )
    targeted = [
        _hits(frame, column)
        for column in (
            f"gf:{TARGETED}:covers",
            f"ltmle:{TARGETED}:covers",
            "ltmle:ey_regimen[targeted]:offers:covers",
        )
    ]
    assert (
        f"held the true value on {targeted[0]} of {DRAWS} draws for the g-formula and {targeted[1]} for "
        f"LTMLE, and LTMLE's interval for the offers on {targeted[2]}"
    ) in _prose("targeted-reading")
    ltmle = _hits(frame, f"ltmle:{CONTRAST}:covers")
    assert f"LTMLE's interval held the true gain on {ltmle} of {DRAWS} draws" in _prose(
        "compare-reading"
    )
    rejects = int((frame["check:pvalue"] < 0.05).sum())
    assert f"p below 0.05 on {rejects} of {DRAWS} draws" in _prose("compare-reading")


def check_refits(namespace: dict[str, Any]) -> None:
    """Refutation: the three refits of the g-formula's gain of `always`."""
    result, refutation = namespace["result"], namespace["refutation"]
    assert refutation.estimand == CONTRAST
    tests = {test.name: test for test in refutation.tests}
    assert list(tests) == ["placebo", "subset", "random_common_cause"]
    original = result[CONTRAST].psi
    for test in tests.values():
        # "`descriptive` ... means no simulation study backs a pass or fail".
        assert test.passed is None
        assert len(test.values) == 20 and test.original == pytest.approx(original)
    placebo = tests["placebo"]
    assert round(float(placebo.statistic), 3) == 1.033
    # A nonzero witness: a placebo that kept the observed offers would return the original gain.
    assert abs(sum(placebo.values) / len(placebo.values)) < 0.1 * original
    noise = tests["random_common_cause"]
    # "the noise column leaves the gain unchanged to two decimals".
    assert round(sum(noise.values) / len(noise.values), 2) == round(original, 2) == 10.66
    assert round(float(tests["subset"].statistic), 3) == 1.069
    assert "descriptive" in stored_output(NOTEBOOK, "refute")
