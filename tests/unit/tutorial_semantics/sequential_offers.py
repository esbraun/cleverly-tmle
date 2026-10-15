"""The reviewed semantic callback for ``docs/examples/sequential-offers``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals
are also compared against its stored outputs, and every decimal it writes is printed or listed in
:data:`UNPRINTED_DECIMALS` with its source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import offers_protocol, win_back
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
    "0.973": f"lowest mean-SE over empirical-SD ratio of the four LTMLE contrasts, 100 draws: "
    f"{_PROBE}/summary.log",
    "1.048": f"highest mean-SE over empirical-SD ratio of the four LTMLE contrasts, 100 draws: "
    f"{_PROBE}/summary.log",
    "-0.0063": f"bias of the LTMLE win-back contrast, 100 draws (sweep.py, seeds 1000-1099): "
    f"{_PROBE}/summary.log",
    "0.0020": f"Monte Carlo standard error of that bias, 100 draws: {_PROBE}/summary.log",
    "-0.0017": f"bias of the LTMLE win-back contrast, 400 draws (bias_check.py, seeds 2000-2399): "
    f"{_PROBE}/bias_check_summary.log",
    "0.0010": f"Monte Carlo standard error of that bias, and the end of the range of LTMLE contrast "
    f"biases, 400 draws: {_PROBE}/bias_check_summary.log",
    "-0.0004": f"bias of the LTMLE win-back level, 400 draws: {_PROBE}/bias_check_summary.log",
    "0.0007": f"Monte Carlo standard error of that bias, 400 draws: {_PROBE}/bias_check_summary.log",
    "+0.0013": f"bias of the LTMLE none level, 400 draws: {_PROBE}/bias_check_summary.log",
    "0.0009": f"Monte Carlo standard error of that bias, 400 draws: {_PROBE}/bias_check_summary.log",
    "-0.0014": f"bias of the IPW win-back contrast, 400 draws: {_PROBE}/bias_check_summary.log",
    "0.0011": f"Monte Carlo standard error of that bias, 400 draws: {_PROBE}/bias_check_summary.log",
    "-0.0029": f"largest LTMLE contrast bias (discount), 400 draws: "
    f"{_PROBE}/bias_check_summary.log",
    "-0.0010": f"smallest LTMLE contrast bias (win-back-then-ship), 400 draws: "
    f"{_PROBE}/bias_check_summary.log",
    "0.968": f"coverage of the LTMLE discount contrast, 400 draws: {_PROBE}/bias_check_summary.log",
    "0.955": f"coverage of the LTMLE shipping contrast, 400 draws: {_PROBE}/bias_check_summary.log",
    "0.953": f"coverage of the LTMLE win-back contrast, 400 draws: {_PROBE}/bias_check_summary.log",
    "0.948": f"coverage of the LTMLE win-back-then-ship contrast, 400 draws "
    f"({_PROBE}/bias_check_summary.log), and the coverage of the cell "
    f"known_mechanism_accuracy/ate_regimen__ltmle_known_treatment__censoring_estimated: "
    f"{_KNOWN}/properties.csv",
    "0.0721": f"Horvitz-Thompson estimate of the win-back contrast, 300 logged draws of 50000 "
    f"customers, true consent probabilities: {_PROBE}/logged_branch_check.log",
    "0.0013": f"Monte Carlo standard error of that estimate: {_PROBE}/logged_branch_check.log",
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
    _check_deployment_and_benchmark(namespace)
    _check_probe_logs()


def _check_data(namespace: dict[str, Any]) -> None:
    panel, truth = namespace["panel"], namespace["truth"]
    # "57079 rows and 10 columns for 20000 customers", up to three rows each.
    assert panel.shape == (57154, 10)
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
    # "A total of 2686 customers withdrew consent before week 4, and they have no outcome."
    withdrawn = panel.groupby("customer")["consent"].min() == 0
    assert int(withdrawn.sum()) == namespace["left"] == 2686
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

    # Step 3: "After a discount, a share of 0.886 ... engaged in week 2".
    moved = namespace["moved"].groupby("offer")["engaged_week2"].mean()
    assert moved.round(3).to_dict() == {"discount": 0.886, "none": 0.551, "shipping": 0.661}
    assert moved["discount"] > moved["shipping"] > moved["none"]
    shares = namespace["week_one"].groupby("engaged")["offer"].value_counts(normalize=True)
    assert round(shares[(1, "none")], 3) == 0.399 and round(shares[(1, "discount")], 3) == 0.246
    assert round(shares[(0, "discount")], 3) == 0.440
    consent = namespace["week_one"].groupby("engaged")["consent"].mean()
    assert consent.round(3).to_dict() == {0: 0.929, 1: 0.969}
    # "It keeps 17314 of 20000 customers ... Only 805 ... and 436".
    crude = namespace["crude"]
    assert len(namespace["kept"]) == 17_314 and len(namespace["wide"]) == 20_000
    assert crude["followers"].tolist() == [805, 436, 550, 1247, 956]
    assert round(crude.loc["win_back", "crude_vs_none"], 3) == 0.049
    assert round(crude.loc["discount", "crude_vs_none"], 3) == -0.110
    assert round(crude.loc["discount", "true_vs_none"], 3) == -0.068
    # A nonzero witness: the crude contrasts of the two plans the reading narrates differ from the
    # truth by more than 0.01, so the comparison cannot be read as the estimand. The shipping
    # contrast happens to land within 0.01 of its truth on this draw, and the reading says only
    # that the sizes of the misses are not general.
    miss = (crude["crude_vs_none"] - crude["true_vs_none"]).abs()
    assert miss["discount"] > 0.01 and miss["win_back"] > 0.01
    assert crude.loc["win_back", "crude_vs_none"] < crude.loc["win_back", "true_vs_none"] - 0.02


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
    # The rule plans are declared known. That each rule reads the engagement of its own week is
    # checked in _check_diagnostics against the follower counts of the fit.
    for label in ("win_back", "win_back_ship"):
        assert plans[label].rule_kind == "known"


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
    # "The win-back contrast is 0.0719 ... All four intervals contain their true values on this
    # draw." The seed follows a fixed rule (the first seed of the swept range), so this relation
    # is a record of the draw and not a selected outcome.
    win = result[_key("win_back")]
    assert (round(win.psi, 4), round(win.std_error, 4)) == (0.0719, 0.0205)
    assert [round(value, 4) for value in win.ci] == [0.0316, 0.1121]
    assert [round(result[key].psi, 4) for key in keys] == [-0.0881, 0.0564, 0.0719, 0.0789]
    for key in keys:
        assert covers(result[key], truth[key])
    assert namespace["against_truth"](result)["covers"].all()
    # The discount interval excludes zero from above, the rules exclude it from below, and the
    # shipping interval starts at 0.0000 to four places: its lower end is positive and below 1e-4.
    assert result[_key("discount")].ci[1] < 0
    assert 0 < result[_key("shipping")].ci[0] < 1e-4
    assert round(result[_key("shipping")].pvalue, 4) == 0.0500
    assert result[_key("win_back")].ci[0] > 0 and result[_key("win_back_ship")].ci[0] > 0
    # "805, 436, and 550 customers followed `none`, `discount`, and `shipping` throughout."
    assert "none: 805 of 20000 units followed it throughout" in summary
    assert "win_back: 1247 of 20000 units followed it throughout" in summary
    assert "win_back_ship: 956 of 20000 units followed it throughout" in summary
    assert "shipping: 550 of 20000 units followed it throughout; max weight 100.0" in summary
    assert "max truncated share 0.2% at t=3" in summary
    assert "critical value 2.412 vs 1.960" in summary

    # Step 7: the other estimators fit the same keys and carry the same protocol.
    for fit in (sdr, gcomp, ipw):
        assert list(fit.estimates) == keys
    assert "sequentially doubly robust" in sdr.summary() and "no fluctuation" in sdr.summary()
    assert "cross-fitting: none -- nuisances fitted in sample" in gcomp.summary()
    assert "an interval assumes every outcome GLM is correctly specified" in gcomp.summary()
    assert "Hajek (normalized by the weight sum)" in ipw.summary()
    assert "an interval assumes every treatment and censoring GLM" in ipw.summary()
    # "The SDR estimates lie within 0.0006 of the LTMLE estimates", and the win-back contrasts.
    gaps = {
        name: max(abs(fit[key].psi - result[key].psi) for key in keys)
        for name, fit in (("sdr", sdr), ("gcomp", gcomp), ("ipw", ipw))
    }
    assert round(gaps["sdr"], 4) == 0.0006
    assert round(gaps["gcomp"], 4) == 0.0030
    assert round(gaps["ipw"], 4) == 0.0112
    # "On this draw, SDR sits above its median, and g-computation and IPW sit below theirs." The
    # medians are those of the 100-draw sweep in UNPRINTED_DECIMALS.
    assert gaps["sdr"] > 0.000485 and gaps["gcomp"] < 0.0036 and gaps["ipw"] < 0.0126
    assert [round(fit[_key("win_back")].psi, 4) for fit in (sdr, gcomp, ipw)] == [
        0.0718,
        0.0739,
        0.0803,
    ]
    assert round(gcomp[_key("win_back")].std_error, 4) == 0.0203
    assert round(ipw[_key("win_back")].std_error, 4) == 0.0213
    # Step 8: "IPW has a wider standard error than LTMLE on all four contrasts."
    assert all(ipw[key].std_error > result[key].std_error for key in keys)
    for fit in namespace["fits"].values():
        assert fit.inference_status == "influence_curve"
    side = namespace["side"]
    assert list(side.columns) == ["LTMLE", "SDR", "g-computation", "IPW", "truth"]
    assert [round(side.loc["win_back", name], 3) for name in side.columns] == [
        0.072,
        0.072,
        0.074,
        0.080,
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
    assert comparison["paired_se"].round(3).tolist() == [0.028, 0.026, 0.020]
    assert comparison["unpaired_se"].round(3).tolist() == [0.037, 0.035, 0.031]
    assert comparison["estimate"].round(3).tolist() == [0.160, 0.016, 0.007]
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
    assert [round(value, 3) for value in (low.iloc[0], high.iloc[0])] == [0.105, 0.214]


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
        "none": 805,
        "discount": 436,
        "shipping": 550,
        "win_back": 1247,
        "win_back_ship": 956,
    }
    # The rule-node witness. The hand count of Step 3 reads each rule at the engagement of its own
    # week, on the customers who stayed consenting to week 3. The fit follows the rule through its
    # own history nodes. The two counts agree for every plan. A rule that read the engagement of
    # the previous week follows a different set of customers, which the second count shows.
    crude = namespace["crude"]
    assert crude["followers"].to_dict() == followed.to_dict()
    kept = namespace["kept"]
    lagged = np.ones(len(kept), dtype=bool)
    for week in (1, 2, 3):
        seen = kept[("engaged", max(week - 1, 1))].astype(int).to_numpy()
        lagged &= kept[("offer", week)].to_numpy() == np.asarray(win_back(seen), dtype=object)
    assert int(lagged.sum()) != int(followed["win_back"])
    # "Win-back keeps the most customers", and every plan keeps at least 1.5% of the 20000.
    assert followed.idxmax() == "win_back"
    assert (followed / 20_000 >= 0.015).all()
    assert (
        min(followed["none"], followed["discount"], followed["shipping"])
        < followed["win_back_ship"]
    )
    at_end = support[support["time"] == 3].set_index("regimen")
    assert at_end["max_weight"].round(3).to_dict() == {
        "none": 60.503,
        "discount": 78.986,
        "shipping": 100.000,
        "win_back": 19.225,
        "win_back_ship": 28.315,
    }
    assert at_end["effective_n"].round(3)["win_back"] == 1241.822
    assert at_end["max_weight"].idxmax() == "shipping"
    # "One row has a truncated weight ... 1 of its 550 followers": the shipping plan at week 3
    # reaches the cap of 1 / 0.01 that the default bound sets. No other row is truncated.
    truncated = support[support["share_truncated"] > 0]
    assert truncated[["regimen", "time"]].values.tolist() == [["shipping", 3]]
    assert round(float(truncated["share_truncated"].iloc[0]) * followed["shipping"]) == 1
    assert round(float(truncated["share_truncated"].iloc[0]), 3) == 0.002
    assert support["max_weight"].max() == pytest.approx(100.0)
    assert round(namespace["kish_ratio"], 3) == 0.863
    np.testing.assert_allclose(
        namespace["kish_ratio"], (support["effective_n"] / support["n_followed"]).min()
    )
    assert result.diagnostics.support().to_frame().equals(support)
    consent = namespace["consent"].set_index("time")
    assert set(consent["evaluation"]) == {"out_of_fold"}
    assert consent["n"].tolist() == [20_000, 19_017, 18_137]
    assert consent["auc"].round(3).tolist() == [0.647, 0.640, 0.638]
    assert consent["calibration_slope"].round(3).tolist() == [1.003, 0.948, 0.892]
    assert consent["calibration_slope"].iloc[-1] < 1.0


def _check_deployment_and_benchmark(namespace: dict[str, Any]) -> None:
    truth = namespace["truth"]
    # Step 11: the synthetic deployment check.
    pilot, pilot_outcome = namespace["pilot"], namespace["pilot_outcome"]
    assert len(pilot_outcome) == 10_000 and int((pilot["consent"] == 0).sum()) == 0
    # The pilot deploys the win-back rule: each week's offer is the rule's offer for that week's
    # engagement, and the pilot panel is not the estimation sample.
    expected = win_back(pilot["engaged"].to_numpy())
    assert (pilot["offer"].to_numpy() == expected).all()
    assert namespace["panel"].shape != pilot.shape
    fitted = namespace["fitted"]
    assert fitted is namespace["levels"]["ey_regimen[win_back]"]
    assert round(namespace["pilot_mean"], 3) == 0.609
    assert round(namespace["pilot_se"], 4) == 0.0049
    assert (round(fitted.psi, 3), round(fitted.std_error, 4)) == (0.595, 0.0133)
    assert round(namespace["difference"], 4) == -0.0143
    assert namespace["difference_se"] == pytest.approx(
        float(np.hypot(fitted.std_error, namespace["pilot_se"]))
    )
    assert round(namespace["difference"] / namespace["difference_se"], 2) == -1.01
    assert abs(namespace["difference"]) < 1.96 * namespace["difference_se"]
    assert round(truth["ey_regimen[win_back]"], 3) == 0.600
    assert covers(fitted, truth["ey_regimen[win_back]"])

    # Step 12: the benchmark refits the static plans without engagement and keeps the declared
    # offer probabilities.
    benchmark = namespace["benchmark"]
    assert list(benchmark.index) == ["discount", "shipping"]
    assert benchmark["move_in_se"].round(3).tolist() == [-0.251, -0.040]
    assert benchmark["move"].round(3).tolist() == [-0.008, -0.001]
    assert benchmark["with_engagement"].round(3).tolist() == [-0.088, 0.056]
    assert benchmark["without"].round(3).tolist() == [-0.096, 0.055]
    # A nonzero witness that the refit drops the covariate: the reduced history lacks engagement,
    # the full one has it, and the estimates move.
    reduced = namespace["reduced_study"]
    assert reduced.data.history_names(2) == ("tenure",)
    assert namespace["study"].data.history_names(2) == ("tenure", "engaged@1", "engaged@2")
    assert (benchmark["move"].abs() > 0.0).all()
    assert (benchmark["move_in_se"].abs() < 1.0).all()
    assert set(namespace["static"]) == {"none", "discount", "shipping"}
    assert "known factors (declared, no learner)" in namespace["reduced_fit"].summary()


def _check_probe_logs() -> None:
    """The committed probe logs print the repeated-draw numbers that the trust step quotes."""
    folder = Path(__file__).resolve().parents[3] / _PROBE
    bias = (folder / "bias_check_summary.log").read_text()
    assert "draws 400, seeds 2000 to 2399" in bias
    ltmle = bias.split("ipw: the four contrasts")[0]
    ipw = bias.split("ipw: the four contrasts")[1]

    def row(text: str, label: str) -> str:
        return next(line for line in text.splitlines() if line.strip().startswith(label + " "))

    assert "bias -0.0017 (MC SE 0.0010" in row(ltmle, "win_back")
    assert "bias -0.0004 (MC SE 0.0007" in row(ltmle.split("the five levels")[1], "win_back")
    assert "bias +0.0013 (MC SE 0.0009" in row(ltmle.split("the five levels")[1], "none")
    assert "bias -0.0014 (MC SE 0.0011" in row(ipw.split("ipw: the five levels")[0], "win_back")
    assert "bias -0.0029" in row(ltmle, "discount") and "bias -0.0010" in row(
        ltmle, "win_back_ship"
    )
    for label, coverage in (
        ("discount", "0.968"),
        ("shipping", "0.955"),
        ("win_back", "0.953"),
        ("win_back_ship", "0.948"),
    ):
        assert f"interval covers {coverage}" in row(ltmle.split("the five levels")[0], label)
    sweep = (folder / "summary.log").read_text()
    assert "ltmle  win_back       coverage 93 of 100 (0.930); bias -0.0063 (MC SE 0.0020)" in sweep
    logged = (folder / "logged_branch_check.log").read_text()
    assert "logged HT-IPW contrast win_back       0.07208 se 0.00132 truth 0.07218" in logged
