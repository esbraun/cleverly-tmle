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

from cleverly.datasets import make_recovery_score, recovery_score_protocol
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
    "1.155": "half the width of the g-formula's printed interval for the gain, 9.51 to 11.82; "
    "checked by _check_probe_log",
    "1.66": "the simple-model g-formula's gain minus the right fit's, 12.32 - 10.66 (interpret); "
    "checked by _check_failure_mode",
    "3.0": f"the largest simulation share of a plan mean's variance, in percent, 400 draws: "
    f"{_PROBE}/sweep.log",
    "0.7": f"the largest simulation share of the difference's variance, in percent, 400 draws: "
    f"{_PROBE}/sweep.log",
    "2.26": f"the mean error of the wrong models' mean under always, 400 draws: {_PROBE}/sweep.log",
    "0.05": f"the bound on the mean error of every blip-model effect, 400 draws: {_PROBE}/sweep.log",
    "0.3": "the bound on the gap between the two methods' plan means, which the compare cell "
    "prints row by row; check() asserts it",
}

DRAWS = 400
LEVELS = ("ey_regimen[never]", "ey_regimen[always]")
CONTRAST = "ate_regimen[always vs never]"
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
    result, difference, truth = namespace["result"], namespace["difference"], namespace["truth"]
    assert [round(result[key].psi, 2) for key in LEVELS] == [49.37, 60.03]
    assert [round(value, 2) for value in difference.ci] == [9.51, 11.82]
    assert round(difference.psi, 2) == 10.66
    assert difference.psi == pytest.approx(result[LEVELS[1]].psi - result[LEVELS[0]].psi)
    # "all three intervals hold".
    for key in LEVELS:
        assert covers(result[key], truth[key])
    assert covers(difference, truth[CONTRAST])
    assert _support(result[key] for key in LEVELS) == {("supported", "G29.g2_g5")}
    # The simulation's share of each variance, smallest for the difference.
    shares = result.simulation.monte_carlo_share
    assert [round(100 * shares[key], 1) for key in (*LEVELS, CONTRAST)] == [2.7, 1.2, 0.5]
    assert shares[CONTRAST] < min(shares[key] for key in LEVELS) < 0.10
    printed = stored_output(NOTEBOOK, "gformula")
    assert "never        812 of 4,000" in printed and "always       495 of 4,000" in printed


def _check_failure_mode(namespace: dict[str, Any]) -> None:
    wrong, wrong_difference = namespace["wrong"], namespace["wrong_difference"]
    truth, result = namespace["truth"], namespace["result"]
    assert namespace["gformula"].covariate_learner is not None
    assert wrong.estimates[LEVELS[1]].psi != result[LEVELS[1]].psi
    # "all three intervals miss", with the printed values.
    assert [round(wrong[key].psi, 2) for key in LEVELS] == [50.05, 62.38]
    assert [round(value, 2) for value in wrong[LEVELS[1]].ci] == [61.71, 63.04]
    assert round(wrong_difference.psi, 2) == 12.32
    # "Simple symptom models moved the answer by 1.66 points".
    assert round(wrong_difference.psi - namespace["difference"].psi, 2) == 1.66
    for key in LEVELS:
        assert not covers(wrong[key], truth[key])
    assert not covers(wrong_difference, truth[CONTRAST])
    # "The fit still reads `supported`."
    assert _support(wrong[key] for key in LEVELS) == {("supported", "G29.g2_g5")}


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
        4.67,
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
    result, blips, difference = namespace["result"], namespace["blips"], namespace["difference"]
    # "they agree with the g-formula's to within 0.3 points".
    gaps = [abs(blips[key].psi - result[key].psi) for key in LEVELS]
    gaps.append(abs(blips[CONTRAST].psi - difference.psi))
    assert 0.0 < max(gaps) < 0.3
    # "sits well above the blip model (12.32 against 10.82 for the gain)".
    wrong = namespace["wrong"]
    assert round(namespace["wrong_difference"].psi, 2) == 12.32
    assert round(blips[CONTRAST].psi, 2) == 10.82
    assert all(wrong[key].psi - blips[key].psi > 0.5 for key in LEVELS)
    check = namespace["check"]
    assert check.names == (CONTRAST,)
    assert round(float(check.structural[0]), 2) == 10.80
    assert round(float(check.nonparametric[0]), 2) == 10.41
    assert round(float(check.pvalue), 2) == 0.29
    # "The check reads `experimental`", printed by the cell.
    assert check.support.label == "experimental"


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
    # models sat at least 0.96 points above it, and more than 1.155 ... on 382 of 400 draws".
    right = frame[f"gf:{CONTRAST}"] - frame[f"snmm:{CONTRAST}"]
    simple = frame[f"wrong:{CONTRAST}"] - frame[f"snmm:{CONTRAST}"]
    assert round(right.abs().max(), 2) == 0.43
    assert round(simple.min(), 2) == 0.96
    low, high = (round(float(bound), 2) for bound in namespace["difference"].ci)
    half = round((high - low) / 2, 3)
    assert half == 1.155
    assert int((simple > half).sum()) == 382
    for column in frame.columns:
        if column.startswith("snmm:effect:") and not column.endswith(":covers"):
            name = column.removeprefix("snmm:")
            assert abs(frame[column].mean() - frame[f"truth:{name}"].iloc[0]) < 0.05
    # The sweep fits the page: its first row is the draw that the page shows.
    first = frame.iloc[0]
    result, wrong, blips = namespace["result"], namespace["wrong"], namespace["blips"]
    for key in LEVELS:
        assert first[f"gf:{key}"] == pytest.approx(result[key].psi, abs=1e-9)
        assert first[f"wrong:{key}"] == pytest.approx(wrong[key].psi, abs=1e-9)
        assert first[f"snmm:{key}"] == pytest.approx(blips[key].psi, abs=1e-9)
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
    rejects = int((frame["check:pvalue"] < 0.05).sum())
    assert f"p below 0.05 on {rejects} of {DRAWS} draws" in _prose("compare-reading")


def check_refits(namespace: dict[str, Any]) -> None:
    """ "Try to break the result": the three refits of the blip model's gain of `always`."""
    blips, refutation = namespace["blips"], namespace["refutation"]
    assert refutation.estimand == CONTRAST
    tests = {test.name: test for test in refutation.tests}
    assert list(tests) == ["placebo", "subset", "random_common_cause"]
    original = blips[CONTRAST].psi
    for test in tests.values():
        # "`descriptive` ... means no simulation study backs a pass or fail".
        assert test.passed is None
        assert len(test.values) == 20 and test.original == pytest.approx(original)
    placebo = tests["placebo"]
    assert round(float(placebo.statistic), 3) == 1.065
    # A nonzero witness: a placebo that kept the observed offers would return the original gain.
    assert abs(sum(placebo.values) / len(placebo.values)) < 0.1 * original
    noise = tests["random_common_cause"]
    assert round(sum(noise.values) / len(noise.values), 3) == 10.822
    assert round(original, 3) == 10.824
    assert round(float(tests["subset"].statistic), 3) == 1.028
    assert "descriptive" in stored_output(NOTEBOOK, "refute")
