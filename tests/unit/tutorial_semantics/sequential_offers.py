"""The reviewed semantic callback for ``docs/examples/sequential-offers``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals
are also compared against its stored outputs, and every decimal it writes is printed or listed in
:data:`UNPRINTED_DECIMALS` with its source.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

import cleverly.datasets.offers as offers_law
from cleverly import CapabilityError, sensitivity
from cleverly.datasets import OFFER_LEVELS, offers_protocol, win_back
from cleverly.utils.bounds import expit
from cleverly.validation import refute
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "sequential-offers.ipynb"

_PROBE = "reviews/notebook-review/probes/sequential-offers"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.0156": "the product of the three smallest logged offer probabilities, 0.25 cubed = 0.015625: "
    "src/cleverly/datasets/offers.py, recomputed in _check_cap",
    "0.0019": f"Monte Carlo standard error of the crude and wrong-model g-computation "
    f"win-back biases, 100 draws: {_PROBE}/summary.log",
    "0.0314": f"mean amount by which the crude win-back contrast, and the wrong-model "
    f"g-computation estimate of it, fall below the truth, 100 draws: {_PROBE}/summary.log",
    "-0.0061": f"bias of the crude shipping contrast, and of the wrong-model g-computation "
    f"estimate of it, 100 draws: {_PROBE}/summary.log",
    "0.0029": f"Monte Carlo standard error of that bias, 100 draws: {_PROBE}/summary.log",
    "-0.0060": f"bias of the wrong-model LTMLE win-back contrast, 100 draws: {_PROBE}/summary.log",
}

CONTRASTS = ("discount", "shipping", "win_back", "win_back_ship")


def _key(contrast: str) -> str:
    return f"ate_regimen[{contrast} vs none]"


def check(namespace: dict[str, Any]) -> None:
    """The offers tutorial keeps its panel, its truth, its protocol, and its seeded narrative."""
    _check_data(namespace)
    _check_graph(namespace)
    _check_design(namespace)
    _check_fits(namespace)
    _check_blind(namespace)
    _check_paired(namespace)
    _check_diagnostics(namespace)
    _check_refutation(namespace)
    _check_deployment_and_benchmark(namespace)
    _check_probe_logs()
    _check_prose_phrases()


def _check_data(namespace: dict[str, Any]) -> None:
    panel, truth = namespace["panel"], namespace["truth"]
    # "The panel has 57154 rows and 10 columns for 20000 customers", up to three rows each.
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

    # "A discount every week ranks last, and win-back then ship ranks first." The orderings are
    # quadrature truths, and the margins are the design constraints of the generator.
    rates = {name: truth[f"ey_regimen[{name}]"] for name in namespace["plans"]}
    assert {name: round(value, 4) for name, value in rates.items()} == {
        "none": 0.5275,
        "discount": 0.4596,
        "shipping": 0.5865,
        "win_back": 0.5997,
        "win_back_ship": 0.6268,
    }
    assert (
        min(rates, key=rates.__getitem__) == "discount"
        and max(rates, key=rates.__getitem__) == "win_back_ship"
    )
    assert rates["discount"] < rates["none"] < rates["shipping"] < rates["win_back"]
    assert truth[_key("win_back")] - 0.0 >= 0.05
    assert rates["win_back"] - rates["discount"] >= 0.08
    assert round(truth[_key("win_back")], 4) == 0.0722

    # Step 3: the three arrows. "After a discount, a share of 0.886 ... engaged in week 2".
    moved = namespace["moved"].groupby("offer")["engaged_week2"].mean()
    assert moved.round(3).to_dict() == {"discount": 0.886, "none": 0.551, "shipping": 0.661}
    assert moved["discount"] > moved["shipping"] > moved["none"]
    shares = namespace["week_one"].groupby("engaged")["offer"].value_counts(normalize=True)
    assert round(shares[(1, "none")], 3) == 0.399 and round(shares[(1, "discount")], 3) == 0.246
    assert round(shares[(0, "discount")], 3) == 0.440
    assert round(shares[(0, "none")], 3) == 0.308 and round(shares[(1, "shipping")], 3) == 0.355
    assert round(shares[(0, "shipping")], 3) == 0.251
    consent = namespace["week_one"].groupby("engaged")["consent"].mean()
    assert consent.round(3).to_dict() == {0: 0.929, 1: 0.969}
    # "It keeps 17314 of 20000 customers".
    crude = namespace["crude"]
    kept = namespace["kept"]
    assert len(kept) == 17_314 and len(namespace["wide"]) == 20_000
    assert crude["followers"].tolist() == [805, 436, 550, 1247, 956]
    assert round(crude.loc["win_back", "crude_vs_none"], 3) == 0.049
    assert round(crude.loc["discount", "crude_vs_none"], 3) == -0.110
    assert round(crude.loc["discount", "true_vs_none"], 3) == -0.068
    assert round(crude.loc["win_back", "true_vs_none"], 3) == 0.072
    # A nonzero witness: the crude contrasts of the two rules and of the discount differ from the
    # truth by more than 0.01, so the comparison cannot be read as the estimand.
    miss = (crude["crude_vs_none"] - crude["true_vs_none"]).abs()
    assert miss["discount"] > 0.01 and miss["win_back"] > 0.01 and miss["win_back_ship"] > 0.01
    assert crude.loc["win_back", "crude_vs_none"] < crude.loc["win_back", "true_vs_none"] - 0.02
    # The mechanism of the reading. The followers of `none` are more often engaged in week 1 than
    # the kept customers, "0.646 against 0.562", because the CRM sends no offer to engaged
    # customers more often. Engaged customers order more, so the reference group is lifted.
    first_engaged = kept[("engaged", 1)].astype(int)
    assert round(first_engaged.mean(), 3) == 0.562
    assert round(crude.loc["none", "engaged_week1"], 3) == 0.646
    assert crude.loc["none", "engaged_week1"] > first_engaged.mean()
    assert shares[(1, "none")] > shares[(0, "none")]
    orders = kept[("full_price", 1)]
    assert orders[first_engaged == 1].mean() > orders[first_engaged == 0].mean() + 0.03
    # Free shipping, at 0.355 against 0.251, also over-selects engaged customers, and its followers
    # are engaged at 0.671 against 0.646 for the followers of `none`. On this draw its crude
    # contrast is within 0.01 of its truth. The reading claims no more: the probe bias of the crude
    # shipping contrast is small and not zero (_check_probe_logs). The rules over-select unengaged
    # customers.
    assert shares[(1, "shipping")] > shares[(0, "shipping")]
    assert round(crude.loc["shipping", "engaged_week1"], 3) == 0.671
    assert crude.loc["shipping", "engaged_week1"] > crude.loc["none", "engaged_week1"]
    assert crude.loc["shipping", "engaged_week1"] > first_engaged.mean()
    assert miss["shipping"] < 0.01 and round(crude.loc["shipping", "crude_vs_none"], 3) == 0.054
    assert crude.loc["win_back", "engaged_week1"] < first_engaged.mean()
    assert crude.loc["win_back_ship", "engaged_week1"] < first_engaged.mean()


def _check_graph(namespace: dict[str, Any]) -> None:
    """The causal graph of the title cell names the arrows that the law and the data carry.

    Every witness is nonzero, so a law that drops an arrow fails the page.
    """
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    title = "".join(next(c for c in notebook["cells"] if c["id"] == "title")["source"])
    fence = title.split("```text")[1].split("```")[0]
    for node in ("W", "L_t", "A_t", "L_{t+1}", "L_4", "C_t", "D", "Y", "U"):
        assert node in fence, node
    assert fence.count("-->") + fence.count("<--") >= 8
    # One table row for each arrow or absent arrow, in the order of the reading.
    rows = [
        line.split("|")[1].strip()
        for line in title.splitlines()
        if line.startswith("| ") and "what it asserts" not in line and "---" not in line
    ]
    arrows = [row for row in rows if row.startswith(("`W`", "`L_", "`A_", "no "))]
    assert arrows == [
        "`W` to `L_1`, `L_2`, `L_3`, `L_4`, `C_t`, and `Y`",
        "`L_t` to `A_t`",
        "`A_t` to `L_{t+1}`",
        "`L_t` to `L_{t+1}`",
        "`L_t` to `C_t`",
        "`L_t` and `A_t` to `D`, and `D` to `Y`",
        "`L_4` to `Y`",
        "no unrecorded cause of `A_t`",
        "no `U` with arrows into `C_t` and `Y`",
        "no `W` to `A_t`",
        "no `A_t` to `C_t`",
    ]
    law, logged = offers_law._LAW, offers_law._LOGGED
    # W to L, C, and Y: every tenure coefficient is nonzero.
    for name in ("e1_tenure", "next_tenure", "consent_tenure", "y_tenure"):
        assert law[name] != 0, name
    # A_t to L_{t+1}: a discount raises the next engagement, and free shipping raises it less.
    assert law["next_discount"] > law["next_shipping"] > 0
    # L_t to L_{t+1}, L_t to C_t, L_4 to Y.
    assert law["next_engaged"] > 0 and law["consent_engaged"] > 0 and law["y_engaged"] > 0
    # D to Y: the fatigue count lowers the order, and it counts discounts sent to engaged customers.
    assert law["y_fatigue"] > 0
    fatigue = offers_law._fatigue_step
    assert bool(fatigue(1, "discount")) and not bool(fatigue(0, "discount"))
    assert not bool(fatigue(1, "none")) and not bool(fatigue(1, "shipping"))
    # No W to A_t, and no unrecorded cause of A_t: the logged probabilities are keyed by the
    # engagement of the week alone, and they are constant within an engagement group.
    assert set(logged) == {0, 1}
    assert list(inspect.signature(offers_law._logged_probabilities).parameters) == ["engaged"]
    probabilities = ["p_none", "p_discount", "p_shipping"]
    panel = namespace["panel"]
    assert (panel.groupby("engaged")[probabilities].nunique() == 1).all().all()
    # No A_t to C_t: no consent coefficient names an offer, and the consent function reads the
    # engagement and the tenure only. The data agree: within an engagement group, consent after
    # week 1 does not move with the week-1 offer by more than four standard errors.
    assert not any(
        name.startswith("consent_") and any(level in name for level in OFFER_LEVELS) for name in law
    )
    assert list(inspect.signature(offers_law._consent).parameters) == ["engaged", "tenure"]
    week_one = namespace["week_one"]
    for engaged, group in week_one.groupby("engaged"):
        rates = group.groupby("offer")["consent"].agg(["mean", "count"])
        spread = rates["mean"].max() - rates["mean"].min()
        error = np.sqrt((rates["mean"] * (1 - rates["mean"]) / rates["count"]).sum())
        assert spread < 4 * error, engaged
    # A_t to L_{t+1} in the data: the same ordering as the law, within each engagement group.
    after = namespace["moved"]
    for engaged, group in after.groupby("engaged"):
        by_offer = group.groupby("offer")["engaged_week2"].mean()
        assert by_offer["discount"] > by_offer["shipping"] > by_offer["none"], engaged


def _check_design(namespace: dict[str, Any]) -> None:
    protocol, study = namespace["protocol"], namespace["study"]
    # The protocol step uses the shipped helper unchanged.
    assert protocol == offers_protocol()
    assert protocol.fingerprint == "91f189a988d70fec"
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 5
    assert "91f189a988d70fec" in stored_output(NOTEBOOK, "protocol")
    assert_protocol_recorded(
        NOTEBOOK,
        "protocol",
        protocol,
        namespace["result"],
        namespace["blind_ltmle"],
        namespace["blind_gcomp"],
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
    # `offer@2`, with `tenure`." No history holds the week-4 engagement.
    assert (
        "history at offer@3: ['tenure', 'engaged@1', 'engaged@2', 'engaged@3', "
        "'offer@1', 'offer@2']"
    ) in printed
    assert "history at offer@1: ['tenure', 'engaged@1']" in printed
    assert "engaged@4" not in printed
    # The four printed assumptions are those of the table in the plan.
    for assumption in (
        "consistency: each observed history equals the potential history",
        "no interference: one unit's potential history",
        "sequential exchangeability given the recorded history at every node",
        "sequential positivity for treatment and remaining under observation",
    ):
        assert assumption in printed
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
    keys = [_key(name) for name in CONTRASTS]
    assert list(result.estimates) == keys
    # Step 6: every learner and the seed are explicit, and the nuisances are fitted in sample.
    method = namespace["ltmle"]
    assert not method.cross_fitting.enabled
    assert method.runtime.random_state == 42 and method.runtime.n_jobs == 1
    assert method.models.treatment_learner is None
    summary = result.summary()
    assert "known factors (declared, no learner): offer@1, offer@2, offer@3" in summary
    assert "no truncation bound moved a declared value" in summary
    assert "cross-fitting: none -- nuisances fitted in sample" in summary
    assert "g_bounds: fixed [0.01, 1] on each cumulative treatment-and-censoring probability" in (
        summary
    )
    # "The win-back contrast is 0.0754 ... All four intervals contain their true values on this
    # draw." The seed follows a fixed rule (the first seed of the swept range), so this relation
    # is a record of the draw and not a selected outcome.
    win = result[_key("win_back")]
    assert (round(win.psi, 4), round(win.std_error, 4)) == (0.0754, 0.0204)
    assert [round(value, 4) for value in win.ci] == [0.0353, 0.1154]
    assert [round(result[key].psi, 4) for key in keys] == [-0.0878, 0.0569, 0.0754, 0.0820]
    for key in keys:
        assert covers(result[key], truth[key])
    assert namespace["against_truth"](result)["covers"].all()
    assert [round(result[key].ci[0], 4) for key in keys] == [-0.1467, 0.0011, 0.0353, 0.0355]
    # The discount interval excludes zero from above, the rules and shipping exclude it from
    # below, and the shipping interval starts at 0.0011: its lower end is positive and small.
    assert result[_key("discount")].ci[1] < 0
    assert 0 < result[_key("shipping")].ci[0] < 2e-3
    assert round(result[_key("shipping")].pvalue, 4) == 0.0456
    assert result[_key("win_back")].ci[0] > 0 and result[_key("win_back_ship")].ci[0] > 0
    # The plan lines of the summary: followers of each policy.
    assert "none: 805 of 20000 units followed it throughout" in summary
    assert "win_back: 1247 of 20000 units followed it throughout" in summary
    assert "win_back_ship: 956 of 20000 units followed it throughout" in summary
    assert "shipping: 550 of 20000 units followed it throughout; max weight 100.0" in summary
    assert "max truncated share 0.2% at t=3" in summary
    # "The bands of discount, win-back, and win-back then ship exclude zero. The band of shipping
    # contains it." The simultaneous bands are printed lines of the summary.
    assert "critical value 2.411 vs 1.960" in summary
    bands = {
        name: (float(low), float(high))
        for name, low, high in re.findall(
            r"^ +ate_regimen\[(\w+) vs none\] +\[(-?[\d.]+), (-?[\d.]+)\]$", summary, re.MULTILINE
        )
    }
    assert set(bands) == set(CONTRASTS)
    assert bands["discount"][1] < 0 < bands["win_back"][0] < bands["win_back_ship"][1]
    assert bands["win_back_ship"][0] > 0
    assert bands["shipping"][0] < 0 < bands["shipping"][1]


def _check_blind(namespace: dict[str, Any]) -> None:
    """Step 7: LTMLE with exact offer probabilities under a wrong outcome model."""
    result, truth = namespace["result"], namespace["truth"]
    blind_ltmle, blind_gcomp = namespace["blind_ltmle"], namespace["blind_gcomp"]
    table = namespace["blind_table"]
    keys = [_key(name) for name in CONTRASTS]
    # The same estimands, so the numbers compare.
    assert blind_ltmle.estimates.keys() == blind_gcomp.estimates.keys() == result.estimates.keys()
    assert "same estimands: True" in stored_output(NOTEBOOK, "blind-outcome")
    # The outcome model is blind: a prior-only classifier and a mean regressor read no history.
    blind = namespace["blind"]
    assert isinstance(blind.outcome_learner, DummyClassifier)
    assert blind.outcome_learner.strategy == "prior"
    assert isinstance(blind.pseudo_learner, DummyRegressor)
    assert blind.censoring_learner is namespace["logistic"]
    assert blind.treatment_learner is None
    assert not namespace["ltmle"].cross_fitting.enabled
    # Both blind fits use the same estimand and identification as the Step 6 fit.
    assert blind_ltmle.provenance.protocol_fingerprint == result.provenance.protocol_fingerprint
    assert list(table.index) == list(CONTRASTS)
    # g-computation reports a point estimate only with these learners.
    for attribute in ("ci", "std_error"):
        with pytest.raises(CapabilityError):
            getattr(blind_gcomp[_key("win_back")], attribute)
    # "The `blind_gcomp` column equals the `crude` column on every row": a constant regression at
    # each node returns the mean outcome of the followers, which is the crude follower mean.
    np.testing.assert_allclose(table["blind_gcomp"], table["crude"], atol=1e-12)
    np.testing.assert_allclose(
        table["crude"], namespace["crude"].loc[list(CONTRASTS), "crude_vs_none"], atol=1e-12
    )
    assert [round(blind_gcomp[key].psi, 3) for key in keys] == [-0.110, 0.054, 0.049, 0.050]
    # "The truth is 0.072. For win-back then ship it is 0.050 against 0.099": g-computation misses
    # both rules by more than 0.02 on this draw.
    for contrast in ("win_back", "win_back_ship"):
        assert table.loc[contrast, "truth"] - table.loc[contrast, "blind_gcomp"] > 0.02
    # "The wrong-model LTMLE estimate for win-back is 0.080, with an interval of (0.038, 0.123)
    # that contains the truth ... All four intervals contain their true values on this draw."
    win = blind_ltmle[_key("win_back")]
    assert round(win.psi, 3) == 0.080 and [round(v, 3) for v in win.ci] == [0.038, 0.123]
    assert table["covers"].all()
    for key in keys:
        assert covers(blind_ltmle[key], truth[key])
    # A nonzero witness that the targeting step moves the blind plug-in: the blind LTMLE estimate
    # differs from the blind g-computation estimate by more than 0.02 for the win-back rule, and
    # it differs from the Step 6 estimate.
    assert abs(table.loc["win_back", "blind_ltmle"] - table.loc["win_back", "blind_gcomp"]) > 0.02
    assert (table["blind_ltmle"] - table["ltmle"]).abs().max() > 0.01
    # "On this draw, the wrong-model LTMLE estimates sit nearer to the truth than the
    # g-computation estimates for discount, win-back, and win-back then ship." Free shipping is the
    # exception, and the reading does not name it. The sentence holds only on this draw: the probe
    # sweep has it for fewer than 90% of the draws for win-back.
    nearer = (table["blind_ltmle"] - table["truth"]).abs() < (
        table["blind_gcomp"] - table["truth"]
    ).abs()
    assert nearer.to_dict() == {
        "discount": True,
        "shipping": False,
        "win_back": True,
        "win_back_ship": True,
    }
    # "The cost is a wider standard error, 0.022 against 0.020 for win-back."
    assert (table["blind_se"] > table["ltmle_se"]).all()
    assert round(table.loc["win_back", "blind_se"], 3) == 0.022
    assert round(table.loc["win_back", "ltmle_se"], 3) == 0.020


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
    assert comparison["paired_se"].round(3).tolist() == [0.027, 0.026, 0.020]
    assert comparison["unpaired_se"].round(3).tolist() == [0.036, 0.035, 0.031]
    assert comparison["estimate"].round(3).tolist() == [0.163, 0.018, 0.007]
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
    assert [round(value, 3) for value in (low.iloc[0], high.iloc[0])] == [0.110, 0.217]
    assert [round(value, 3) for value in (low.iloc[1], high.iloc[1])] == [-0.032, 0.069]
    assert [round(value, 3) for value in (low.iloc[2], high.iloc[2])] == [-0.033, 0.046]


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
    # own history nodes. The two counts agree for every policy. A rule that read the engagement of
    # the previous week follows a different set of customers, which the second count shows.
    crude = namespace["crude"]
    assert crude["followers"].to_dict() == followed.to_dict()
    kept = namespace["kept"]
    lagged = np.ones(len(kept), dtype=bool)
    for week in (1, 2, 3):
        seen = kept[("engaged", max(week - 1, 1))].astype(int).to_numpy()
        lagged &= kept[("offer", week)].to_numpy() == np.asarray(win_back(seen), dtype=object)
    assert int(lagged.sum()) != int(followed["win_back"])
    # "Win-back keeps the most customers", and every policy keeps at least 1.5% of the 20000.
    assert followed.idxmax() == "win_back"
    assert (followed / 20_000 >= 0.015).all()
    assert (
        min(followed["none"], followed["discount"], followed["shipping"])
        < followed["win_back_ship"]
    )
    at_end = support[support["time"] == 3].set_index("regimen")
    assert at_end["max_weight"].round(3).to_dict() == {
        "none": 59.676,
        "discount": 80.877,
        "shipping": 100.000,
        "win_back": 19.651,
        "win_back_ship": 28.592,
    }
    assert at_end["effective_n"].round(3).to_dict() == {
        "none": 708.798,
        "discount": 379.489,
        "shipping": 475.304,
        "win_back": 1241.930,
        "win_back_ship": 935.762,
    }
    assert at_end["max_weight"].idxmax() == "shipping"
    # "One row has a truncated weight ... 1 of its 550 followers": the shipping policy at week 3
    # reaches the cap of 1 / 0.01 that the default bound sets. No other row is truncated.
    truncated = support[support["share_truncated"] > 0]
    assert truncated[["regimen", "time"]].values.tolist() == [["shipping", 3]]
    assert round(float(truncated["share_truncated"].iloc[0]) * followed["shipping"]) == 1
    assert round(float(truncated["share_truncated"].iloc[0]), 3) == 0.002
    assert support["max_weight"].max() == pytest.approx(100.0)
    assert round(namespace["kish_ratio"], 3) == 0.864
    np.testing.assert_allclose(
        namespace["kish_ratio"], (support["effective_n"] / support["n_followed"]).min()
    )
    assert result.diagnostics.support().to_frame().equals(support)
    _check_cap(namespace)
    # The consent models are fitted in sample, so the reading does not call the slopes out of fold.
    consent = namespace["consent"].set_index("time")
    assert set(consent["evaluation"]) == {"in_sample"}
    assert consent["n"].tolist() == [20_000, 19_017, 18_137]
    assert consent["auc"].round(3).tolist() == [0.647, 0.645, 0.650]
    assert consent["calibration_slope"].round(3).tolist() == [1.003, 1.004, 1.000]


def _check_cap(namespace: dict[str, Any]) -> None:
    """The one weight at the cap comes from the consent probabilities, not from the offers.

    The reading says the offer probabilities alone give a product of at least 0.0156, and that the
    customer at the cap has low tenure and was not engaged in any week.  The check recomputes the
    cumulative probability of each shipping follower from the law.
    """
    wide = namespace["wide"]
    law = offers_law._LAW
    followers = np.ones(len(wide), dtype=bool)
    for week in (1, 2, 3):
        engaged = np.nan_to_num(wide[("engaged", week)].to_numpy()).astype(int)
        shipped = wide[("offer", week)].to_numpy() == np.asarray(
            namespace["OFFER_PLANS"]["shipping"](engaged), dtype=object
        )
        followers &= shipped & (wide[("consent", week)].to_numpy() == 1)
    rows = wide[followers]
    assert len(rows) == 550
    offers = np.ones(len(rows))
    cumulative = np.ones(len(rows))
    for week in (1, 2, 3):
        engaged = rows[("engaged", week)].astype(int).to_numpy()
        tenure = rows[("tenure", week)].to_numpy()
        stay = expit(
            law["consent_intercept"]
            + law["consent_engaged"] * engaged
            + law["consent_tenure"] * tenure
        )
        offers = offers * rows[("p_shipping", week)].to_numpy()
        cumulative = cumulative * rows[("p_shipping", week)].to_numpy() * stay
    assert offers.min() == pytest.approx(0.25**3) and round(0.25**3, 4) == 0.0156
    below = np.flatnonzero(cumulative < 0.01)
    assert len(below) == 1
    capped = rows.iloc[below[0]]
    assert all(capped[("engaged", week)] == 0 for week in (1, 2, 3))
    # The tenure at which the three consent probabilities of an unengaged customer multiply to
    # 0.64 = 0.01 / 0.25**3 is -2.566, the value that the comment of the logged table states.
    threshold = (np.log(0.64 ** (1 / 3) / (1 - 0.64 ** (1 / 3))) - law["consent_intercept"]) / law[
        "consent_tenure"
    ]
    assert round(float(threshold), 3) == -2.566
    assert capped[("tenure", 1)] < threshold
    # The truncation that the reading describes comes from the fitted consent model. The fit's own
    # unbounded cumulative probability, restricted to the rows that train the last node, falls below
    # the cap at the same customer, so a change of the consent learner cannot separate the two.
    fit = namespace["result"].fits["shipping"]
    fitted = np.flatnonzero(fit.steps[-1].trained_on & (fit.cumulative_unbounded[:, -1] < 0.01))
    assert fitted.tolist() == [wide.index.get_loc(rows.index[below[0]])]
    assert all(wide.iloc[fitted[0]][("engaged", week)] == 0 for week in (1, 2, 3))


def _check_refutation(namespace: dict[str, Any]) -> None:
    """Step 10: the four refits of the win-back contrast, and the checks that the page states it skips."""
    result, refutation = namespace["result"], namespace["refutation"]
    estimand = _key("win_back")
    assert refutation.estimand == estimand
    tests = {test.name: test for test in refutation.tests}
    assert list(tests) == ["placebo", "subset", "random_common_cause", "bootstrap_stability"]
    original = result[estimand].psi
    for test in tests.values():
        # "No test prints a verdict": no registry row covers a binomial end-of-study fit.
        assert test.passed is None and test.rung == "L0"
        assert len(test.values) == 20 and test.original == pytest.approx(original)
    assert refutation.passed is None
    # The placebo reading: "-0.0023826 ... spread 0.01815 ... t -0.5872 (p = 0.564) ... 1.15".
    placebo = tests["placebo"]
    values = np.asarray(placebo.values)
    assert round(float(values.mean()), 7) == -0.0023826
    assert round(float(values.std(ddof=1)), 5) == 0.01815
    assert round(float(placebo.statistic), 4) == -0.5872
    assert round(float(placebo.statistic_pvalue), 3) == 0.564
    assert round(float(np.mean(placebo.standard_errors) / values.std(ddof=1)), 2) == 1.15
    # A nonzero witness: a placebo that kept the observed offers would return the original
    # estimate, so the null reading would sit at 0.0754 and not near zero.
    assert abs(values.mean()) < 0.1 * original
    # The subset reading: "0.012468 against 0.013387 ... ratio 0.93129 (p = 0.75)".
    subset = tests["subset"]
    scatter = float(np.std(subset.values, ddof=1))
    reference = float(result[estimand].std_error * np.sqrt((20_000 / 14_000 - 1) * 20_000 / 19_999))
    assert round(scatter, 6) == 0.012468 and round(reference, 6) == 0.013387
    assert round(scatter / reference, 5) == 0.93129
    # The random-common-cause reading: the paired shift is nonzero and small.
    noise = tests["random_common_cause"]
    shift = (np.asarray(noise.values) - original) / result[estimand].std_error
    assert 0.0 < abs(shift.mean()) < 0.05 and round(float(shift.mean()), 6) == -0.012488
    assert round(float(np.std(shift, ddof=1)), 5) == 0.06524
    # The stability reading: "-5.321e-05 ... spread 0.01597".
    stability = tests["bootstrap_stability"]
    assert round(float(np.mean(stability.values)) - original, 8) == -5.321e-05
    assert round(float(np.std(stability.values, ddof=1)), 5) == 0.01597
    # The stored output holds the sentence that names the outcome family as the difference.
    assert "This fit differs on outcome family 'binomial'" in stored_output(NOTEBOOK, "refute")

    # The skipped checks of the reading, each refused as the reading states.
    wide = namespace["panel"].pivot(index="customer", columns="week")
    tenure = wide[("tenure", 1)].to_numpy()
    with pytest.raises(CapabilityError, match="equals the adjustment column 'tenure'"):
        refute(
            result,
            estimand=estimand,
            tests=("negative_control_outcome",),
            negative_control_outcome=tenure,
            negative_control_kind="pre_treatment",
            n_replicates=5,
        )
    with pytest.raises(CapabilityError, match="node with more levels"):
        refute(
            result,
            estimand=estimand,
            tests=("future_treatment",),
            future_treatment_control=tenure,
        )
    for name in ("simulated_outcome", "plasmode"):
        with pytest.raises(CapabilityError, match="'binomial'"):
            refute(result, estimand=estimand, tests=(name,), n_replicates=10)
    with pytest.raises(CapabilityError, match="F16"):
        sensitivity.evalue(result, estimand)
    grid = sensitivity.ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.1))
    with pytest.raises(CapabilityError, match="F13"):
        sensitivity.simulated_confounding(result, grid=grid, estimand=estimand)


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
    assert (round(fitted.psi, 3), round(fitted.std_error, 4)) == (0.596, 0.0133)
    assert round(namespace["difference"], 4) == -0.0135
    assert namespace["difference_se"] == pytest.approx(
        float(np.hypot(fitted.std_error, namespace["pilot_se"]))
    )
    assert round(namespace["difference_se"], 4) == 0.0141
    assert round(namespace["difference"] / namespace["difference_se"], 2) == -0.95
    assert abs(namespace["difference"]) < 1.96 * namespace["difference_se"]
    assert round(truth["ey_regimen[win_back]"], 3) == 0.600
    assert covers(fitted, truth["ey_regimen[win_back]"])

    # Step 12: the benchmark refits the static policies without engagement and keeps the declared
    # offer probabilities.
    benchmark = namespace["benchmark"]
    assert list(benchmark.index) == ["discount", "shipping"]
    assert benchmark["move_in_se"].round(3).tolist() == [-0.239, -0.095]
    assert benchmark["move"].round(3).tolist() == [-0.007, -0.003]
    assert benchmark["with_engagement"].round(3).tolist() == [-0.088, 0.057]
    assert benchmark["without"].round(3).tolist() == [-0.095, 0.054]
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
    """The committed probe logs print the repeated-draw numbers that the readings quote."""
    folder = Path(__file__).resolve().parents[3] / _PROBE
    sweep = (folder / "summary.log").read_text()
    assert sweep.startswith("draws 100, seeds 1000 to 1099")

    def line(text: str, *parts: str) -> str:
        return next(row for row in text.splitlines() if all(part in row for part in parts))

    # Step 6: the coverage of the four contrasts.
    coverage = {"discount": 95, "shipping": 95, "win_back": 92, "win_back_ship": 94}
    for contrast, hits in coverage.items():
        row = line(sweep, "  ltmle  ", contrast.ljust(14), "coverage")
        assert f"coverage {hits} of 100" in row
    # Step 3: the crude win-back contrast, and the engagement of the followers of `none`.
    assert "bias -0.0314 (MC SE 0.0019)" in line(sweep, "  crude  ", "win_back       ")
    # The crude shipping contrast: a small bias of about two Monte Carlo standard errors.
    assert "bias -0.0061 (MC SE 0.0029)" in line(sweep, "  crude  ", "shipping       ")
    assert "below the truth on 96 of 100" in line(sweep, "  crude  ", "win_back       ")
    assert "followers of none above kept customers: 100 of 100" in sweep
    # Step 7: the blind fits. The blind g-computation equals the crude contrast on every draw.
    blind = line(sweep, "blind LTMLE   win_back       ")
    assert "coverage 93 of 100" in blind and "bias -0.0060" in blind
    assert "bias -0.0314 (MC SE 0.0019)" in line(sweep, "blind g-comp  win_back       ")
    gaps = [row for row in sweep.splitlines() if "largest gap from the crude contrast" in row]
    assert len(gaps) == 4 and all(row.endswith("0.00e+00") for row in gaps)
    # Step 8: the paired intervals.
    assert sweep.count("paired SE below unpaired SE: 100 of 100") == 3
    exclusions = [
        row.split(": ", 1)[1].split(" (")[0]
        for row in sweep.splitlines()
        if "paired interval excludes zero" in row
    ]
    assert exclusions == ["100 of 100", "8 of 100", "26 of 100"]
    # The probe fits the page: the first row of the sweep is the draw that the page shows.
    first = pd.read_csv(folder / "sweep.csv", float_precision="round_trip").iloc[0]
    assert int(first["seed"]) == 1000
    assert round(first["ltmle_win_back_psi"], 4) == 0.0754
    assert round(first["ltmle_win_back_se"], 4) == 0.0204
    assert round(first["blind_win_back_psi"], 3) == 0.080
    assert round(first["blindg_win_back_psi"], 3) == 0.049
    assert round(first["crude_win_back_psi"], 3) == 0.049


def _prose(cell_id: str) -> str:
    """The source of one markdown cell, with each run of whitespace collapsed to a space."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])
    return " ".join(source.split())


def _check_prose_phrases() -> None:
    """The readings attach each repeated-draw count and decimal to the quantity the log gives it.

    A decimal that a log prints passes the narration test wherever it stands.  This check builds
    each phrase of the readings from the parsed logs, so a count or a value moved to another
    quantity fails.
    """
    root = Path(__file__).resolve().parents[3]
    sweep = (root / _PROBE / "summary.log").read_text()

    def line(text: str, *parts: str) -> str:
        return next(row for row in text.splitlines() if all(part in row for part in parts))

    def bias_of(row: str) -> tuple[str, str]:
        found = re.search(r"bias ([+-]\d\.\d{4}) \(MC SE (\d\.\d{4})", row)
        assert found, row
        return found.group(1), found.group(2)

    def grab(pattern: str, row: str) -> str:
        found = re.search(pattern, row)
        assert found, row
        return found.group(1)

    def count_of(row: str, label: str) -> str:
        found = re.search(rf"{re.escape(label)} (\d+) of 100", row)
        assert found, row
        return found.group(1)

    contrasts = ("discount", "shipping", "win_back", "win_back_ship")
    ltmle = {c: line(sweep, "  ltmle  ", c.ljust(14), "coverage") for c in contrasts}
    crude = {c: line(sweep, "  crude  ", c.ljust(14), "below the truth") for c in contrasts}
    blind_ltmle = line(sweep, "blind LTMLE   win_back       ")
    blind_gcomp = line(sweep, "blind g-comp  win_back       ")
    crude_win = bias_of(crude["win_back"])
    crude_ship = bias_of(crude["shipping"])
    gcomp_win = bias_of(blind_gcomp)
    blind_win = bias_of(blind_ltmle)
    hits = [count_of(ltmle[c], "coverage") for c in contrasts]
    exclusions = [
        grab(r"excludes zero: (\d+) of 100", row)
        for row in sweep.splitlines()
        if "paired interval excludes zero" in row
    ]

    reading_ltmle = _prose("ltmle-reading")
    assert f"covered on {', '.join(hits[:-1])}, and {hits[-1]} of 100 draws" in reading_ltmle

    association = _prose("association-reading")
    assert (
        f"fell below the truth on {count_of(crude['win_back'], 'below the truth on')} of 100 draws, by "
        f"{crude_win[0].lstrip('-')} on average. The Monte Carlo standard error was {crude_win[1]}."
    ) in association
    assert (
        f"mean bias of {crude_ship[0]} (Monte Carlo standard error {crude_ship[1]})" in association
    )

    reading = _prose("blind-outcome-reading")
    assert (
        f"fell below the truth by {gcomp_win[0].lstrip('-')} on average over 100 draws" in reading
    )
    assert (
        f"The wrong-model LTMLE bias was {blind_win[0]}, and its interval covered the truth on "
        f"{count_of(blind_ltmle, 'coverage')} of 100 draws"
    ) in reading

    interpret = _prose("interpret")
    assert f"excluded zero on {exclusions[1]} and {exclusions[2]} of 100 draws" in interpret
