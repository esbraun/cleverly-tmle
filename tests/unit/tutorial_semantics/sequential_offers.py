"""The reviewed semantic callback for ``docs/examples/sequential-offers``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals
are also compared against its stored outputs, and every decimal it writes is printed or listed in
:data:`UNPRINTED_DECIMALS` with its source.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import offers_protocol, win_back, win_back_then_ship
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "sequential-offers.ipynb"

_PROBE = "reviews/notebook-review/probes/sequential-offers"
_KNOWN = "tests/canonical/known_node_mechanisms"
_BASELINES = "tests/canonical/sequential_baselines"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.0005": f"median of the largest distance of SDR from LTMLE (0.000485), 100 draws, seeds "
    f"1000-1099: {_PROBE}/summary.log",
    "0.0015": f"maximum of the largest distance of SDR from LTMLE (0.0015), 100 draws: "
    f"{_PROBE}/summary.log",
    "0.0036": f"median of the largest distance of g-computation from LTMLE, 100 draws: "
    f"{_PROBE}/summary.log",
    "0.0101": f"maximum of the largest distance of g-computation from LTMLE, 100 draws: "
    f"{_PROBE}/summary.log",
    "0.0126": f"median of the largest distance of IPW from LTMLE, 100 draws: {_PROBE}/summary.log",
    "0.0364": f"maximum of the largest distance of IPW from LTMLE, 100 draws: {_PROBE}/summary.log",
    "0.513": f"largest absolute truncation-curve move in standard errors over 100 draws, the "
    f"bounds 0.02, 0.05 and 0.10, and the four contrasts: {_PROBE}/summary.log",
    "0.973": f"lowest mean-SE over empirical-SD ratio of the four LTMLE contrasts, 100 draws: "
    f"{_PROBE}/summary.log",
    "1.048": f"highest mean-SE over empirical-SD ratio of the four LTMLE contrasts, 100 draws: "
    f"{_PROBE}/summary.log",
    "-0.0063": f"bias of the LTMLE win-back contrast, 100 draws: {_PROBE}/summary.log",
    "0.0020": f"Monte Carlo standard error of that bias, 100 draws: {_PROBE}/summary.log",
    "1.7": "lower end of the standardized bias of the three misspecification_control cells: "
    "docs/technical-reference/method-evidence/sequential-baselines.md, the declared negative "
    "control",
    "0.48": f"lower end of the coverage of the three misspecification_control cells: {_BASELINES}",
    "0.9480": f"coverage of ate_regimen[always vs never], correct outcome regressions, declared "
    f"factors: {_KNOWN}/summary.csv",
    "0.9804": f"se_ratio of the same row: {_KNOWN}/summary.csv",
    "0.9272": f"lower coverage of known_mechanism_accuracy/"
    f"ate_regimen__ltmle_known_treatment__censoring_estimated: {_KNOWN}/properties.csv",
    "0.9644": f"upper coverage of the same cell: {_KNOWN}/properties.csv",
    "0.9974": f"se_ratio of the same cell: {_KNOWN}/properties.csv",
    "0.9500": f"coverage of ate_regimen[static_12 vs static_00], g-computation, law D2: "
    f"{_BASELINES}/summary.csv",
    "0.9590": f"coverage of the same estimand, Hajek IPW, law D2: {_BASELINES}/summary.csv",
    "0.9957": f"se_ratio of the same estimand, g-computation, law D2: {_BASELINES}/summary.csv",
    "1.0054": f"se_ratio of the same estimand, Hajek IPW, law D2: {_BASELINES}/summary.csv",
    "0.1306": f"lower bias of misspecification_control/hajek_x_contrast__main_terms: "
    f"{_BASELINES}/properties.csv",
    "0.1344": f"upper bias of the same cell: {_BASELINES}/properties.csv",
    "0.5072": f"lower coverage of the same cell: {_BASELINES}/properties.csv",
    "0.5348": f"upper coverage of the same cell: {_BASELINES}/properties.csv",
}

CONTRASTS = ("discount", "shipping", "win_back", "win_back_ship")


def _key(contrast: str) -> str:
    return f"ate_regimen[{contrast} vs none]"


def check(namespace: dict[str, Any]) -> None:
    """The offers tutorial keeps its panel, its truth, its protocol, and its seeded narrative."""
    _check_data(namespace)
    _check_design(namespace)
    _check_fits(namespace)
    _check_paired(namespace)
    _check_diagnostics(namespace)
    _check_secondary(namespace)


def _check_data(namespace: dict[str, Any]) -> None:
    panel, truth = namespace["panel"], namespace["truth"]
    # "57079 rows and 10 columns for 20000 customers", up to three rows each.
    assert panel.shape == (57079, 10)
    assert panel["customer"].nunique() == 20_000
    assert panel.groupby("customer").size().max() == 3
    assert list(panel.columns) == [
        "customer",
        "week",
        "tenure",
        "engaged",
        "offer",
        "consent",
        "p_none",
        "p_discount",
        "p_shipping",
        "full_price",
    ]
    # "A total of 2697 customers withdrew consent before week 4, and they have no outcome."
    withdrawn = panel.groupby("customer")["consent"].min() == 0
    assert int(withdrawn.sum()) == namespace["left"] == 2697
    outcome = panel.groupby("customer")["full_price"].first()
    assert outcome[withdrawn].isna().all()
    assert outcome[~withdrawn].notna().all()
    # A nonzero witness for the declared mechanism: the logged probabilities are those the reading
    # states, and each row's three probabilities sum to one.
    first = panel[panel["week"] == 1]
    engaged_rows, other_rows = first[first["engaged"] == 1], first[first["engaged"] == 0]
    assert engaged_rows[["p_none", "p_discount", "p_shipping"]].iloc[0].tolist() == [
        0.4,
        0.25,
        0.35,
    ]
    assert other_rows[["p_none", "p_discount", "p_shipping"]].iloc[0].tolist() == [0.3, 0.45, 0.25]
    np.testing.assert_allclose(panel[["p_none", "p_discount", "p_shipping"]].sum(axis=1), 1.0)

    # "The true full-price rate is 0.5275 ... free shipping beats no offers, and a discount every
    # week loses to no offers. Win-back beats both static policies." The orderings are quadrature
    # truths, and the margins are the design constraints of the generator.
    rates = {name: truth[f"ey_regimen[{name}]"] for name in namespace["plans"]}
    assert {name: round(value, 4) for name, value in rates.items()} == {
        "none": 0.5275,
        "discount": 0.4596,
        "shipping": 0.5865,
        "win_back": 0.5997,
        "win_back_ship": 0.6268,
    }
    assert rates["discount"] < rates["none"] < rates["shipping"] < rates["win_back"]
    assert rates["win_back"] < rates["win_back_ship"]
    assert truth[_key("win_back")] - 0.0 >= 0.05
    assert rates["win_back"] - rates["discount"] >= 0.08
    assert round(truth[_key("win_back")], 4) == 0.0722

    # Step 3: "After a discount, a share of 0.889 ... engaged in week 2".
    moved = namespace["moved"].groupby("offer")["engaged_week2"].mean()
    assert moved.round(3).to_dict() == {"discount": 0.889, "none": 0.550, "shipping": 0.668}
    assert moved["discount"] > moved["shipping"] > moved["none"]
    shares = namespace["week_one"].groupby("engaged")["offer"].value_counts(normalize=True)
    assert round(shares[(1, "none")], 3) == 0.406 and round(shares[(1, "discount")], 3) == 0.248
    assert round(shares[(0, "discount")], 3) == 0.455
    consent = namespace["week_one"].groupby("engaged")["consent"].mean()
    assert consent.round(3).to_dict() == {0: 0.928, 1: 0.967}
    # "It keeps 17303 of 20000 customers ... Only 718 ... and 427".
    crude = namespace["crude"]
    assert len(namespace["kept"]) == 17_303 and len(namespace["wide"]) == 20_000
    assert crude["followers"].tolist() == [718, 427, 525, 1270, 928]
    assert round(crude.loc["win_back", "crude_vs_none"], 3) == 0.039
    assert round(crude.loc["discount", "crude_vs_none"], 3) == -0.084
    assert round(crude.loc["discount", "true_vs_none"], 3) == -0.068
    # A nonzero witness: the crude contrasts differ from the truth by more than 0.01 for every
    # plan, so the comparison cannot be read as the estimand.
    assert ((crude["crude_vs_none"] - crude["true_vs_none"]).abs()[1:] > 0.01).all()
    assert crude.loc["win_back", "crude_vs_none"] < crude.loc["win_back", "true_vs_none"] - 0.03


def _check_design(namespace: dict[str, Any]) -> None:
    protocol, study = namespace["protocol"], namespace["study"]
    # The protocol step uses the shipped helper unchanged.
    assert protocol == offers_protocol()
    assert protocol.fingerprint == "94024dfd011817af"
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 5
    assert "94024dfd011817af" in stored_output(NOTEBOOK, "protocol")
    assert_protocol_recorded(
        NOTEBOOK,
        "protocol",
        protocol,
        namespace["result"],
        namespace["sdr"],
        namespace["gcomp"],
        namespace["ipw"],
        namespace["no_logs"],
        namespace["levels"],
    )
    # "The three `history_names` lines list the columns that the design puts in the history."
    printed = stored_output(NOTEBOOK, "identify")
    for week, names in (
        (1, ("tenure", "engaged@1")),
        (2, ("tenure", "engaged@1", "engaged@2")),
        (3, ("tenure", "engaged@1", "engaged@2", "engaged@3")),
    ):
        assert study.data.history_names(week) == names
        assert f"history_names({week}): {names}" in printed
    # "The `history at offer@3` line lists `engaged@1`, `engaged@2`, `engaged@3`, `offer@1`, and
    # `offer@2`, with `tenure`."
    assert (
        "history at offer@3: ['tenure', 'engaged@1', 'engaged@2', 'engaged@3', "
        "'offer@1', 'offer@2']"
    ) in printed
    assert "history at offer@1: ['tenure', 'engaged@1']" in printed
    design = namespace["offers_design"]()
    assert dict(design.treatment_probabilities) == namespace["LOGGED"]
    assert dict(namespace["LOGGED"]) == {
        "none": "p_none",
        "discount": "p_discount",
        "shipping": "p_shipping",
    }
    assert design.censoring == "consent"
    plans = namespace["plans"]
    assert list(plans) == ["none", "discount", "shipping", "win_back", "win_back_ship"]
    assert plans["none"] == ("none",) * 3 and plans["shipping"] == ("shipping",) * 3
    # The rule plans read the engagement of their own week and send the shipped offers.
    for label, offer_for in (("win_back", win_back), ("win_back_ship", win_back_then_ship)):
        rule = plans[label]
        assert rule.rule_kind == "known"
        engaged = np.array([0, 1, 0, 1])
        expected = offer_for(engaged)
        assert expected[0] == "discount" and expected[1] == (
            "none" if label == "win_back" else "shipping"
        )


def _check_fits(namespace: dict[str, Any]) -> None:
    result, truth = namespace["result"], namespace["truth"]
    sdr, gcomp, ipw = namespace["sdr"], namespace["gcomp"], namespace["ipw"]
    keys = [_key(name) for name in CONTRASTS]
    assert list(result.estimates) == keys
    # Step 6: every learner and the folds are explicit and seeded.
    method = namespace["ltmle"]
    assert method.cross_fitting.n_folds == 2 and method.cross_fitting.enabled
    assert method.runtime.random_state == 42 and method.runtime.n_jobs == 1
    summary = result.summary()
    assert "known factors (declared, no learner): offer@1, offer@2, offer@3" in summary
    assert "no truncation bound moved a declared value" in summary
    assert "cross-fitting: 2 fold(s)" in summary
    assert "g_bounds: fixed [0.01, 1]" in summary
    # "The win-back contrast is 0.0701 ... All four intervals contain their true values on this
    # draw."
    win = result[_key("win_back")]
    assert (round(win.psi, 4), round(win.std_error, 4)) == (0.0701, 0.0195)
    assert [round(value, 4) for value in win.ci] == [0.0318, 0.1083]
    assert [round(result[key].psi, 4) for key in keys] == [-0.0464, 0.0437, 0.0701, 0.0844]
    for key in keys:
        assert covers(result[key], truth[key])
    assert namespace["against_truth"](result)["covers"].all()
    # The interval contains zero for the two weakest contrasts and excludes it for the rules.
    assert result[_key("discount")].ci[0] < 0 < result[_key("discount")].ci[1]
    assert result[_key("win_back")].ci[0] > 0 and result[_key("win_back_ship")].ci[0] > 0
    # "718, 427, and 525 customers followed `none`, `discount`, and `shipping` throughout."
    assert "none: 718 of 20000 units followed it throughout" in summary
    assert "win_back: 1270 of 20000 units followed it throughout" in summary
    assert "critical value 2.451 vs 1.960" in summary

    # Step 7: the other estimators fit the same keys and carry the same protocol.
    for fit in (sdr, gcomp, ipw):
        assert list(fit.estimates) == keys
    assert "sequentially doubly robust" in sdr.summary() and "no fluctuation" in sdr.summary()
    assert "cross-fitting: none -- nuisances fitted in sample" in gcomp.summary()
    assert "an interval assumes every outcome GLM is correctly specified" in gcomp.summary()
    assert "Hajek (normalized by the weight sum)" in ipw.summary()
    assert "an interval assumes every treatment and censoring GLM" in ipw.summary()
    # "The SDR estimates equal the LTMLE estimates within 0.0005", and the win-back contrasts.
    gaps = {
        name: max(abs(fit[key].psi - result[key].psi) for key in keys)
        for name, fit in (("sdr", sdr), ("gcomp", gcomp), ("ipw", ipw))
    }
    assert gaps["sdr"] < 0.0005 and round(gaps["sdr"], 4) == 0.0004
    assert round(gaps["gcomp"], 4) == 0.0015
    assert round(gaps["ipw"], 4) == 0.0196
    assert [round(fit[_key("win_back")].psi, 4) for fit in (sdr, gcomp, ipw)] == [
        0.0696,
        0.0699,
        0.0613,
    ]
    assert round(gcomp[_key("win_back")].std_error, 4) == 0.0219
    assert round(ipw[_key("win_back")].std_error, 4) == 0.0219
    # Step 8: "IPW has a wider standard error than LTMLE on all four contrasts."
    assert all(ipw[key].std_error > result[key].std_error for key in keys)
    for fit in namespace["fits"].values():
        assert fit.inference_status == "influence_curve"
    side = namespace["side"]
    assert list(side.columns) == ["LTMLE", "SDR", "g-computation", "IPW", "truth"]
    assert [round(side.loc["win_back", name], 3) for name in side.columns] == [
        0.070,
        0.070,
        0.070,
        0.061,
        0.072,
    ]
    # A nonzero witness that the comparison reads the fits: IPW is the one that moves.
    assert abs(side.loc["win_back", "IPW"] - side.loc["win_back", "LTMLE"]) > 0.005


def _check_paired(namespace: dict[str, Any]) -> None:
    truth, result = namespace["truth"], namespace["result"]
    comparison = namespace["comparison"]
    assert list(comparison.index) == [
        "win_back - discount",
        "win_back - shipping",
        "win_back_ship - win_back",
    ]
    # "The unpaired number is larger in every row": the two contrasts share their customers, so
    # their estimates correlate positively and the paired variance is smaller.
    assert (comparison["unpaired_se"] > comparison["paired_se"]).all()
    unpaired = comparison["unpaired_se"] ** 2 - comparison["paired_se"] ** 2
    assert (unpaired > 0).all()
    assert comparison["paired_se"].round(3).tolist() == [0.027, 0.025, 0.020]
    assert comparison["unpaired_se"].round(3).tolist() == [0.035, 0.034, 0.030]
    assert comparison["estimate"].round(3).tolist() == [0.117, 0.026, 0.014]
    assert comparison["truth"].round(3).tolist() == [0.140, 0.013, 0.027]
    expected = [
        truth[_key("win_back")] - truth[_key("discount")],
        truth[_key("win_back")] - truth[_key("shipping")],
        truth[_key("win_back_ship")] - truth[_key("win_back")],
    ]
    np.testing.assert_allclose(comparison["truth"], expected)
    # The estimates are the differences of the two reported contrasts.
    np.testing.assert_allclose(
        comparison["estimate"].to_numpy()[0],
        result[_key("win_back")].psi - result[_key("discount")].psi,
    )
    # "The paired interval excludes zero" for the first pair and holds the truth, and the other
    # two pairs hold zero and their truths on this draw.
    low, high = comparison["paired_low"], comparison["paired_high"]
    assert low["win_back - discount"] > 0
    assert (low < comparison["truth"]).all() and (comparison["truth"] < high).all()
    assert low["win_back - shipping"] < 0 < high["win_back - shipping"]
    assert low["win_back_ship - win_back"] < 0 < high["win_back_ship - win_back"]
    assert [round(value, 3) for value in (low.iloc[0], high.iloc[0])] == [0.064, 0.169]


def _check_diagnostics(namespace: dict[str, Any]) -> None:
    result = namespace["result"]
    ran = namespace["ran"].set_index("check")["status"]
    assert ran.to_dict() == {
        "score_equations": "passed",
        "support": "completed",
        "nuisance_models": "completed",
    }
    support = namespace["support"]
    followed = support[support["time"] == 3].set_index("regimen")["n_followed"]
    assert followed.to_dict() == {
        "none": 718,
        "discount": 427,
        "shipping": 525,
        "win_back": 1270,
        "win_back_ship": 928,
    }
    # "Win-back keeps the most customers", and every plan keeps at least 1.5% of the 20000.
    assert followed.idxmax() == "win_back"
    assert (followed / 20_000 >= 0.015).all()
    assert (
        min(followed["none"], followed["discount"], followed["shipping"])
        < followed["win_back_ship"]
    )
    at_end = support[support["time"] == 3].set_index("regimen")
    assert at_end["max_weight"].round(3).to_dict() == {
        "none": 60.032,
        "discount": 75.712,
        "shipping": 92.840,
        "win_back": 19.399,
        "win_back_ship": 27.744,
    }
    assert at_end["effective_n"].round(3)["win_back"] == 1264.961
    assert at_end["max_weight"].idxmax() == "shipping"
    # "No row has a truncated weight at the default bound of (0.01, 1), which caps a weight at
    # 100": the largest weight sits under the cap of 1 / 0.01.
    assert (support["share_truncated"] == 0.0).all()
    assert support["max_weight"].max() < 100.0
    assert round(namespace["kish_ratio"], 3) == 0.861
    np.testing.assert_allclose(
        namespace["kish_ratio"], (support["effective_n"] / support["n_followed"]).min()
    )
    assert result.diagnostics.support().to_frame().equals(support)
    consent = namespace["consent"].set_index("time")
    assert set(consent["evaluation"]) == {"out_of_fold"}
    assert consent["n"].tolist() == [20_000, 18_980, 18_099]
    assert consent["auc"].round(3).tolist() == [0.633, 0.649, 0.651]
    assert consent["calibration_slope"].round(3).tolist() == [0.971, 0.977, 0.923]
    assert consent["calibration_slope"].iloc[-1] < 1.0


def _check_secondary(namespace: dict[str, Any]) -> None:
    truth = namespace["truth"]
    # Step 11: the secondary analysis declares no offer probabilities and estimates the mechanism.
    no_logs = namespace["no_logs"]
    assert namespace["offers_design"](logged=False).treatment_probabilities is None
    assert "known factors" not in no_logs.summary()
    assert namespace["estimated"].models.treatment_learner.random_state == 42
    assert namespace["against_truth"](no_logs)["covers"].all()
    win = no_logs[_key("win_back")]
    assert (round(win.psi, 3), round(win.std_error, 3)) == (0.070, 0.020)
    curve = namespace["curve"]
    assert sorted(curve["lower_bound"].unique()) == [0.01, 0.02, 0.05, 0.10]
    tables = namespace["tables"]
    # "The bound 0.01 is the fit itself": no move and no truncated cell there.
    assert (tables["move"][0.01] == 0.0).all()
    assert (tables["cells"][0.01] == 0).all()
    # "The largest move is 0.101 standard errors, for the discount contrast at the bound 0.02."
    moves = tables["move"].div(namespace["no_logs_se"], axis=0)
    assert round(float(moves.abs().max().max()), 3) == 0.101
    assert moves.abs().stack().idxmax() == ("discount", 0.02)
    # "At the bound 0.10, the discount contrast truncates 2275 of 18368 cells ... 0.071".
    assert namespace["cells"].loc["discount", 0.10] == "2275 of 18368"
    assert round(float(moves.loc["discount", 0.10]), 3) == 0.071
    # More truncation at a higher bound, on every contrast.
    cells = tables["cells"]
    assert (cells[0.02] <= cells[0.05]).all() and (cells[0.05] <= cells[0.10]).all()
    assert (cells[0.10] > 0).all()
    assert (tables["evaluated"].nunique(axis=1) == 1).all()

    # Step 12: the synthetic deployment check.
    pilot, pilot_outcome = namespace["pilot"], namespace["pilot_outcome"]
    assert len(pilot_outcome) == 10_000 and int((pilot["consent"] == 0).sum()) == 0
    # The pilot deploys the win-back rule: each week's offer is the rule's offer for that week's
    # engagement, and the pilot panel is not the estimation sample.
    expected = win_back(pilot["engaged"].to_numpy())
    assert (pilot["offer"].to_numpy() == expected).all()
    assert namespace["panel"].shape != pilot.shape
    fitted = namespace["fitted"]
    assert fitted is namespace["levels"]["ey_regimen[win_back]"]
    assert round(namespace["pilot_mean"], 3) == 0.603
    assert round(namespace["pilot_se"], 4) == 0.0049
    assert (round(fitted.psi, 3), round(fitted.std_error, 4)) == (0.608, 0.0133)
    assert round(namespace["difference"], 4) == 0.0050
    assert namespace["difference_se"] == pytest.approx(
        float(np.hypot(fitted.std_error, namespace["pilot_se"]))
    )
    assert round(namespace["difference"] / namespace["difference_se"], 2) == 0.35
    assert abs(namespace["difference"]) < 2.0 * namespace["difference_se"]
    assert round(truth["ey_regimen[win_back]"], 3) == 0.600
    assert covers(fitted, truth["ey_regimen[win_back]"])

    # Step 13: the benchmark refits the static plans without engagement and keeps the declared
    # offer probabilities.
    benchmark = namespace["benchmark"]
    assert list(benchmark.index) == ["discount", "shipping"]
    assert benchmark["move_in_se"].round(3).tolist() == [-0.058, -0.083]
    assert benchmark["move"].round(3).tolist() == [-0.002, -0.002]
    assert benchmark["with_engagement"].round(3).tolist() == [-0.046, 0.044]
    assert benchmark["without"].round(3).tolist() == [-0.048, 0.041]
    # A nonzero witness that the refit drops the covariate: the reduced history lacks engagement,
    # the full one has it, and the estimates move.
    reduced = namespace["reduced_study"]
    assert reduced.data.history_names(2) == ("tenure",)
    assert namespace["study"].data.history_names(2) == ("tenure", "engaged@1", "engaged@2")
    assert (benchmark["move"].abs() > 0.0).all()
    assert (benchmark["move_in_se"].abs() < 1.0).all()
    assert set(namespace["static"]) == {"none", "discount", "shipping"}
    assert "known factors (declared, no learner)" in namespace["reduced_fit"].summary()
