"""The reviewed semantic callback for ``docs/examples/weekly-outcomes``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals
are also compared against its stored outputs, and every decimal it writes is printed or listed in
:data:`UNPRINTED_DECIMALS` with its source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import (
    CapabilityError,
    CausalStudy,
    LongitudinalTreatment,
    RegimeContrast,
    sensitivity,
)
from cleverly.datasets import weekly_orders_protocol, weekly_orders_truth
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refute",)


NOTEBOOK = EXAMPLES / "weekly-outcomes.ipynb"

_PROBE = "reviews/notebook-review/probes/weekly-outcomes"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.0006": f"the largest absolute mean bias of the six LTMLE rows, 400 draws: {_PROBE}/sweep.log",
    "0.071": f"mean amount by which the week-2 separate fit falls below the truth, 400 draws: "
    f"{_PROBE}/sweep.log",
    "0.098": f"mean amount by which the week-3 separate fit falls below the truth, 400 draws: "
    f"{_PROBE}/sweep.log",
}

PLANS = ("always", "win_back")
WEEKS = (1, 2, 3)
DRAWS = 400


def _key(plan: str, week: int) -> str:
    return f"ate_regimen[{plan} vs never @ t={week}]"


def _tilt_key(week: int) -> str:
    return f"ate_regimen[odds x2 vs natural course @ t={week}]"


def check(namespace: dict[str, Any]) -> None:
    """The weekly-orders tutorial keeps its law, its protocol, its fits, and its seeded narrative."""
    _check_data(namespace)
    _check_design(namespace)
    _check_fits(namespace)
    _check_failure_mode(namespace)
    _check_total(namespace)
    _check_ipsi(namespace)
    _check_assessment(namespace)
    _check_sensitivity(namespace)
    _check_sdr(namespace)
    _check_probe_logs(namespace)
    _check_prose_phrases()


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    assert frame.shape == (12_000, 14)
    assert truth == weekly_orders_truth()
    # "the gain of `always` over `never` falls from 0.178 in week 1 to 0.110 in week 3".
    assert [round(truth[_key("always", week)], 3) for week in WEEKS] == [0.178, 0.122, 0.110]
    # "About half of the customers get a code each week."
    for week in WEEKS:
        assert 0.4 < frame[f"offer_{week}"].mean() < 0.6
    # The scoring at work: "twice as often (0.668, against 0.337)".
    week_two = frame[frame["tracked_1"] == 1]
    rates = week_two.groupby("order_1")["offer_2"].mean()
    assert round(rates[0.0], 3) == 0.668 and round(rates[1.0], 3) == 0.337
    # "The simple comparison understates every effect", and "by week 3 it has the wrong sign".
    crude = namespace["crude"]
    assert (crude["crude_vs_never"] < crude["true_vs_never"]).all()
    always = crude.loc["always", "crude_vs_never"]
    assert [round(always[week], 3) for week in WEEKS] == [0.047, -0.041, -0.084]


def _check_design(namespace: dict[str, Any]) -> None:
    protocol, study, result = namespace["protocol"], namespace["study"], namespace["result"]
    assert protocol == weekly_orders_protocol()
    assert_protocol_recorded(NOTEBOOK, "protocol", protocol, result)
    # "the week-3 code [is adjusted] for `order_1` and `order_2`", which Repeated adds.
    history = study.data.history_names(3)
    assert {"order_1", "order_2"} <= set(history)
    assert tuple(study.data.repeated_names) == ("order_1", "order_2", "order_3")


def _check_fits(namespace: dict[str, Any]) -> None:
    result, truth = namespace["result"], namespace["truth"]
    assert set(result.estimates) == {_key(plan, week) for plan in PLANS for week in WEEKS}
    psi = {key: estimate.psi for key, estimate in result.estimates.items()}
    assert [round(psi[_key("always", week)], 3) for week in WEEKS] == [0.168, 0.111, 0.076]
    assert [round(psi[_key("win_back", week)], 3) for week in WEEKS] == [0.098, 0.085, 0.046]
    assert [round(value, 3) for value in result[_key("always", 1)].ci] == [0.151, 0.185]
    assert [round(value, 3) for value in result[_key("always", 3)].ci] == [0.030, 0.122]
    # "the effect fades as customers get codes in a row".
    assert psi[_key("always", 1)] > psi[_key("always", 2)] > psi[_key("always", 3)]
    # "five of the six intervals holding their true values", and win-back's week 3 misses.
    held = {key: covers(estimate, truth[key]) for key, estimate in result.estimates.items()}
    assert [key for key, value in held.items() if not value] == [_key("win_back", 3)]
    assert [round(value, 3) for value in result[_key("win_back", 3)].ci] == [0.012, 0.080]
    # Every row is supported (G27 in-sample repeated LTMLE).
    assert {estimate.support.label for estimate in result.estimates.values()} == {"supported"}
    # The simultaneous bands hold the truth that the pointwise interval misses, marked experimental.
    printed = stored_output(NOTEBOOK, "ltmle")
    assert "ate_regimen[win_back vs never @ t=3]   [0.0000, 0.0915]" in printed
    assert truth[_key("win_back", 3)] <= 0.0915
    assert "cleverly marks them experimental" in " ".join(printed.split())
    # The plan table: followers through week 3 and their weights.
    assert "never     1,623 of 12,000        45.1" in printed
    assert "win_back  3,128 of 12,000         6.0" in printed


def _separate_fit(namespace: dict[str, Any], week: int) -> Any:
    """The page's separate end-of-study fit for one week, which leaves earlier orders out."""
    design = LongitudinalTreatment(
        outcome=f"order_{week}",
        treatment=tuple(f"offer_{t}" for t in range(1, week + 1)),
        baseline=("tenure", "order_0"),
        time_varying=tuple((f"engaged_{t}",) for t in range(1, week + 1)),
        censoring=tuple(f"tracked_{t}" for t in range(1, week + 1)),
    )
    contrast = RegimeContrast({"never": (0,) * week, "always": (1,) * week}, reference="never")
    fit = (
        CausalStudy(namespace["frame"], design=design)
        .identify(contrast)
        .estimate(method=namespace["ltmle"])
    )
    return fit["ate_regimen[always vs never]"]


def _check_failure_mode(namespace: dict[str, Any]) -> None:
    """ "What goes wrong when earlier orders are left out": refit the separate fits."""
    result, truth = namespace["result"], namespace["truth"]
    separate = {week: _separate_fit(namespace, week) for week in WEEKS}
    # "Week 1 agrees (0.168 in both)": the same design, so the same estimate.
    assert round(separate[1].psi, 3) == round(result[_key("always", 1)].psi, 3) == 0.168
    # "0.051 and 0.008 against true gains of 0.122 and 0.110, and their intervals miss".
    assert round(separate[2].psi, 3) == 0.051 and round(separate[3].psi, 3) == 0.008
    for week in (2, 3):
        assert not covers(separate[week], truth[_key("always", week)])
        assert separate[week].psi < result[_key("always", week)].psi
    printed = stored_output(NOTEBOOK, "failure-mode")
    assert "3             0.008        -0.034          0.050         0.076  0.110" in printed


def _check_total(namespace: dict[str, Any]) -> None:
    result, truth = namespace["result"], namespace["truth"]
    expected = {"always": (0.355, 0.294, 0.417, 0.031, 0.029), "win_back": (0.228, 0.183, 0.274)}
    for plan in PLANS:
        keys = [_key(plan, week) for week in WEEKS]
        total = result.contrast(lambda values: values[0] + values[1] + values[2], keys, name=plan)
        unpaired = float(np.sqrt(sum(result[key].std_error ** 2 for key in keys)))
        shown = (total.psi, *total.ci, total.std_error, unpaired)
        assert [round(value, 3) for value in shown[: len(expected[plan])]] == list(expected[plan])
        # "their errors move together": the joint standard error is above the unpaired one.
        assert total.std_error > unpaired
        # "Win-back's interval just misses its true total"; the interval of `always` holds it.
        true_total = sum(truth[key] for key in keys)
        assert covers(total, true_total) == (plan == "always")
        # The total is the sum of the weekly gains.
        assert total.psi == pytest.approx(sum(result[key].psi for key in keys), abs=1e-12)
    totals = [round(sum(truth[_key(plan, week)] for week in WEEKS), 2) for plan in PLANS]
    assert totals == [0.41, 0.28]


def _check_ipsi(namespace: dict[str, Any]) -> None:
    nudge_result, truth = namespace["nudge_result"], namespace["truth"]
    assert set(nudge_result.estimates) == {_tilt_key(week) for week in WEEKS}
    psi = [round(nudge_result[_tilt_key(week)].psi, 4) for week in WEEKS]
    assert psi == [0.0248, 0.0139, 0.0114]
    for week in WEEKS:
        assert covers(nudge_result[_tilt_key(week)], truth[_tilt_key(week)])
    labels = {estimate.support.label for estimate in nudge_result.estimates.values()}
    assert labels == {"supported"}
    # "smaller than those of a code every week".
    result = namespace["result"]
    for week in WEEKS:
        assert nudge_result[_tilt_key(week)].psi < result[_key("always", week)].psi


def _check_assessment(namespace: dict[str, Any]) -> None:
    assert namespace["attention"] == ()
    shown = namespace["shown"]
    assert shown["followers"].to_dict() == {"never": 1623, "always": 1150, "win_back": 3128}
    weights = shown["largest weight"].round(1).to_dict()
    assert weights == {"never": 45.1, "always": 44.2, "win_back": 6.0}
    # "no customer reaches the cap".
    assert (shown["share at the cap"] == 0.0).all()
    # Win-back keeps the most followers and the lightest weights.
    assert (
        shown["followers"].idxmax() == "win_back" and shown["largest weight"].idxmin() == "win_back"
    )
    nuisance = namespace["nuisance"]
    offer = nuisance[nuisance["role"] == "treatment"]
    tracking = nuisance[nuisance["role"] == "censoring"]
    assert len(offer) == len(tracking) == 3
    # "AUC about 0.69" and "about 0.61", and slopes near 1.
    assert offer["auc"].between(0.68, 0.70).all()
    assert tracking["auc"].between(0.60, 0.63).all()
    slopes = nuisance[nuisance["role"].isin(["treatment", "censoring"])]["calibration_slope"]
    assert slopes.between(0.98, 1.02).all()


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    moves = namespace["moves"]
    engagement, tenure = moves["without engagement"], moves["without tenure"]
    # "Dropping engagement moves every effect down, from -0.69 to -2.28 standard errors".
    assert (engagement < 0).all()
    assert round(-engagement.max(), 2) == 0.69 and round(-engagement.min(), 2) == 2.28
    # "Dropping tenure moves the effects up by at most 0.83".
    assert (tenure > 0).all() and round(tenure.max(), 2) == 0.83
    # "Even the largest move leaves every effect positive."
    result = namespace["result"]
    for (plan, week), shift in engagement.items():
        estimate = result[_key(plan, week)]
        assert estimate.psi + shift * estimate.std_error > 0
    # The refusals the Details dropdown names.
    nudge_result = namespace["nudge_result"]
    estimand = _key("always", 3)
    grid = sensitivity.ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.1))
    with pytest.raises(CapabilityError, match="repeated outcome"):
        sensitivity.simulated_confounding(result, grid=grid, estimand=estimand)
    with pytest.raises(CapabilityError, match="IPSI fit"):
        sensitivity.simulated_confounding(nudge_result, grid=grid, estimand=_tilt_key(3))
    with pytest.raises(CapabilityError, match="F16"):
        sensitivity.omitted_variable_bounds(result, estimand=estimand)
    with pytest.raises(CapabilityError, match="F16"):
        sensitivity.robustness_value(result, estimand=estimand)
    with pytest.raises(CapabilityError, match="F16"):
        sensitivity.evalue(result, estimand)


def _check_sdr(namespace: dict[str, Any]) -> None:
    result, sdr_result, truth = namespace["result"], namespace["sdr_result"], namespace["truth"]
    labels = {estimate.support.label for estimate in sdr_result.estimates.values()}
    assert labels == {"supported"}
    # "SDR's estimates match LTMLE's to within a few thousandths in every row".
    gaps = [abs(sdr_result[key].psi - result[key].psi) for key in result.estimates]
    assert 0.0 < max(gaps) < 0.005
    # "SDR's intervals hold the same five true values and miss win-back's week 3".
    held = {key: covers(sdr_result[key], truth[key]) for key in result.estimates}
    assert [key for key, value in held.items() if not value] == [_key("win_back", 3)]


def _sweep() -> tuple[str, pd.DataFrame]:
    folder = Path(__file__).resolve().parents[3] / _PROBE
    log = (folder / "sweep.log").read_text()
    frame = pd.read_csv(folder / "sweep.csv", float_precision="round_trip")
    return log, frame


def _hits(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].sum())


def _check_probe_logs(namespace: dict[str, Any]) -> None:
    """The committed sweep prints the repeated-draw numbers that the readings quote."""
    log, frame = _sweep()
    assert log.startswith(f"draws: {DRAWS} (seeds 3000 to 3399)")
    assert len(frame) == DRAWS and frame["seed"].tolist() == list(range(3000, 3000 + DRAWS))
    truth = weekly_orders_truth()
    # LTMLE: biases within 0.0006 of zero, and every row supported.
    biases = [
        frame[f"ltmle_{plan}_{week}_psi"].mean() - truth[_key(plan, week)]
        for plan in PLANS
        for week in WEEKS
    ]
    assert round(max(abs(value) for value in biases), 4) == 0.0006
    for label in ("ltmle", "sdr"):
        for plan in PLANS:
            for week in WEEKS:
                assert set(frame[f"{label}_{plan}_{week}_label"]) == {"supported"}
    # The separate fits: below the truth on every draw in weeks 2 and 3.
    for week, shortfall in ((2, 0.071), (3, 0.098)):
        bias = frame[f"separate_{week}_psi"] - truth[_key("always", week)]
        assert (bias < 0).all() and round(-bias.mean(), 3) == shortfall
    # The total: the paired standard error is above the unpaired one on every draw.
    for plan in PLANS:
        assert (frame[f"total_{plan}_se"] > frame[f"total_{plan}_unpaired_se"]).all()
    # The sweep fits the page: its first row is the draw that the page shows.
    first = frame.iloc[0]
    result, nudge_result = namespace["result"], namespace["nudge_result"]
    sdr_result = namespace["sdr_result"]
    for plan in PLANS:
        for week in WEEKS:
            key = _key(plan, week)
            assert first[f"ltmle_{plan}_{week}_psi"] == pytest.approx(result[key].psi, abs=1e-9)
            assert first[f"sdr_{plan}_{week}_psi"] == pytest.approx(sdr_result[key].psi, abs=1e-9)
    for week in WEEKS:
        estimate = nudge_result[_tilt_key(week)].psi
        assert first[f"ipsi_{week}_psi"] == pytest.approx(estimate, abs=1e-9)


def _prose(cell_id: str) -> str:
    """The source of one markdown cell, with each run of whitespace collapsed to a space."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])
    return " ".join(source.split())


def _span(counts: list[int]) -> str:
    return f"{min(counts)} to {max(counts)} of {DRAWS} draws"


def _check_prose_phrases() -> None:
    """The readings attach each repeated-draw count to the quantity the sweep gives it.

    A count is an integer, so the narration test does not read it.  This check builds each
    phrase of the readings from the committed sweep, so a count moved to another quantity fails.
    """
    _, frame = _sweep()
    ltmle = [_hits(frame, f"ltmle_{plan}_{week}_covers") for plan in PLANS for week in WEEKS]
    sdr = [_hits(frame, f"sdr_{plan}_{week}_covers") for plan in PLANS for week in WEEKS]
    ipsi = [_hits(frame, f"ipsi_{week}_covers") for week in WEEKS]
    assert f"held its true value on {_span(ltmle)}" in _prose("ltmle-reading")
    assert f"held their true values on {_span(sdr)}" in _prose("sdr-reading")
    assert f"held their true values on {_span(ipsi)}" in _prose("ipsi-reading")
    separate = [
        int((frame[f"separate_{week}_psi"] < weekly_orders_truth()[_key("always", week)]).sum())
        for week in (2, 3)
    ]
    assert separate == [DRAWS, DRAWS]
    assert f"fell below the truth on all {DRAWS} draws in weeks 2 and 3" in _prose(
        "failure-mode-reading"
    )
    paired = _hits(frame, "total_always_covers")
    unpaired = _hits(frame, "total_always_unpaired_covers")
    assert (
        f"held the true total on {paired} of {DRAWS} draws for `always`, and the other interval "
        f"on {unpaired}."
    ) in _prose("total-reading")


def check_refits(namespace: dict[str, Any]) -> None:
    """ "Try to break the result": the three refits of the week-3 effect of a code every week."""
    result, refutation = namespace["result"], namespace["refutation"]
    estimand = _key("always", 3)
    assert refutation.estimand == estimand
    tests = {test.name: test for test in refutation.tests}
    assert list(tests) == ["placebo", "subset", "random_common_cause"]
    original = result[estimand].psi
    for test in tests.values():
        # "`descriptive` ... means no simulation study backs a pass or fail".
        assert test.passed is None
        assert len(test.values) == 20 and test.original == pytest.approx(original)
    assert refutation.passed is None
    # "The placebo refits average -0.0004, and their `t` ... is -0.10".
    placebo = tests["placebo"]
    assert round(float(np.mean(placebo.values)), 4) == -0.0004
    assert round(float(placebo.statistic), 2) == -0.10
    # A nonzero witness: a placebo that kept the observed codes would return the original 0.076.
    assert abs(np.mean(placebo.values)) < 0.1 * original
    # "The noise column leaves the estimate at 0.0762, against 0.0761."
    noise = tests["random_common_cause"]
    assert round(float(np.mean(noise.values)), 4) == 0.0762 and round(original, 4) == 0.0761
    # "The 70% refits spread a little less than predicted (`rho` 0.80, ideal 1)".
    subset = tests["subset"]
    assert round(float(subset.statistic), 2) == 0.80
    assert "descriptive" in stored_output(NOTEBOOK, "refute")
