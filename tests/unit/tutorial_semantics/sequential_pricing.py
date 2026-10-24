"""The reviewed semantic callback for ``docs/examples/sequential-pricing``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals
are also compared against its stored outputs, and every decimal it writes is printed or listed in
:data:`UNPRINTED_DECIMALS` with its source.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

import cleverly.datasets.pricing as pricing_law
from cleverly import (
    CapabilityError,
    CausalStudy,
    DataError,
    IPWMethod,
    LongitudinalTreatment,
    ModelSpec,
    RegimeContrast,
    SparseInfluenceWarning,
    TMLEMethod,
)
from cleverly._summary import vocabulary
from cleverly.datasets import (
    PRICE_BANDS,
    PRICE_PLANS,
    make_sequential_pricing,
    price_rise,
    pricing_protocol,
    pricing_truth,
)
from cleverly.interventions import ModifiedPolicy, Piece
from cleverly.longitudinal import DynamicRegimen
from cleverly.sensitivity import evalue
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.utils.bounds import logit
from cleverly.validation import refute
from cleverly.validation.refute import _placebo_null, refute_refusal
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    notebook_code,
    stored_output,
)

NOTEBOOK = EXAMPLES / "sequential-pricing.ipynb"

_PROBE = "reviews/notebook-review/probes/sequential-pricing"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.0056": f"SD of the price_up placebo estimates, 400 draws: {_PROBE}/placebo.log",
    "-0.00013": f"mean of the price_up placebo estimates, 400 draws: {_PROBE}/placebo.log",
    "-0.45": f"t statistic of the price_up placebo, 400 draws: {_PROBE}/placebo.log",
    "0.65": f"two-sided p-value of that t statistic, 400 draws: {_PROBE}/placebo.log",
    "1.01": f"mean reported SE over the SD of the placebo estimates, 400 draws: {_PROBE}/placebo.log",
    "94.5": f"share of the placebo 95% intervals that contain 0, 400 draws: {_PROBE}/placebo.log",
    "-0.0002": f"bias of the LTMLE renewal contrast of price_up, 200 draws: {_PROBE}/summary.log",
    "0.0004": f"bias of the LTMLE renewal contrast of price_down, 200 draws: {_PROBE}/summary.log",
    "0.0005": f"Monte Carlo standard error of both renewal biases, 200 draws: {_PROBE}/summary.log, "
    f"and the paired Monte Carlo standard error of the price_up learner difference: "
    f"{_PROBE}/revenue_learner.log",
    "0.0099": f"bias of the LTMLE revenue contrast of price_up, 200 draws: {_PROBE}/summary.log",
    "0.0094": f"bias of the LTMLE revenue contrast of price_down, 200 draws: {_PROBE}/summary.log",
    "0.0039": f"Monte Carlo standard error of the price_up revenue bias: {_PROBE}/summary.log",
    "0.0038": f"Monte Carlo standard error of the price_down revenue bias: {_PROBE}/summary.log",
    "0.0557": f"SD of the price_up revenue estimates, 200 draws: {_PROBE}/summary.log",
    "0.0537": f"SD of the price_down revenue estimates, 200 draws: {_PROBE}/summary.log",
    "0.0028": f"bias of the price_up revenue contrast, quadratic outcome learner, 60 draws: "
    f"{_PROBE}/revenue_learner.log",
    "0.0132": f"bias of the price_up revenue contrast, linear outcome learner, 60 draws: "
    f"{_PROBE}/revenue_learner.log",
    "0.0075": f"Monte Carlo standard error of the price_up quadratic bias, 60 draws: "
    f"{_PROBE}/revenue_learner.log",
    "0.0074": f"Monte Carlo standard error of the price_up linear bias, 60 draws: "
    f"{_PROBE}/revenue_learner.log",
    "0.88": "bootstrap_stability spread 0.005749 over the renewal standard error 0.0065, both "
    "printed (checked in check_refits)",
    "1.98": "mean reported standard error over the SD of the ten placebo estimates, 1.983 (checked "
    "in check_refits)",
    "10.5": "the printed renewal contrast 0.1050 of the 15% cut, in percentage points (checked in "
    "_check_reading_attribution)",
    "0.31": "the size of the printed revenue contrast -0.3129 of the 12% rise, written as a fall "
    "(checked in _check_reading_attribution)",
    "0.0104": f"mean estimate of the linear learner minus the quadratic learner, price_up revenue, "
    f"60 paired draws: {_PROBE}/revenue_learner.log",
    "0.0076": f"mean estimate of the linear learner minus the quadratic learner, price_down revenue, "
    f"60 paired draws: {_PROBE}/revenue_learner.log",
    "0.0010": f"paired Monte Carlo standard error of the price_down learner difference: "
    f"{_PROBE}/revenue_learner.log",
    "-0.0203": f"limit of the omit regression, price_up, one draw of 400000: {_PROBE}/summary.log",
    "0.0270": f"limit of the omit regression, price_down, one draw of 400000: {_PROBE}/summary.log",
    "14.82": f"mean maximum weight of price_up at month 3, 200 draws: {_PROBE}/summary.log",
    "2.03": f"SD of that maximum weight: {_PROBE}/summary.log",
    "31.76": f"mean maximum weight of price_down at month 3, 200 draws: {_PROBE}/summary.log",
    "5.21": f"SD of that maximum weight: {_PROBE}/summary.log",
    "0.361": f"mean effective n over n of price_up at month 3, 200 draws: {_PROBE}/summary.log",
    "0.154": f"mean effective n over n of price_down at month 3, 200 draws: {_PROBE}/summary.log",
}

FINGERPRINT = "682ab02814f9837b"
PLANS = ("price_up", "price_down")
#: Relative tolerance of the weight-tail quantities, the maximum weight and the effective n, which
#: the fresh run agrees with within the last digits on every platform.
WEIGHT_TAIL_REL = 1e-3
RENEWAL = "ate_regimen"
REVENUE = "ate_revenue"


def _policies(regimen: DynamicRegimen) -> list[Any]:
    """The node policies of a plan, each a ``ModifiedPolicy`` or a ``Shift``."""
    return list(regimen.plan)


def _key(plan: str) -> str:
    return f"ate_regimen[{plan} vs natural_prices]"


def check(namespace: dict[str, Any]) -> None:
    """The pricing tutorial keeps its data, its truth, its protocol, and its seeded narrative."""
    _check_data(namespace)
    _check_graph(namespace)
    _check_design(namespace)
    _check_renewal_fit(namespace)
    _check_revenue_fit(namespace)
    _check_failure_mode(namespace)
    _check_support(namespace)
    _check_assessment(namespace)
    _check_refusals(namespace)
    _check_refit_limits(namespace)
    _check_probe_logs()
    _check_prose_phrases(namespace)


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    fresh, fresh_truth = make_sequential_pricing(10_000, seed=1000)
    pd.testing.assert_frame_equal(frame, fresh)
    assert truth == fresh_truth
    assert frame.shape == (10_000, 9)
    assert list(frame.columns) == [
        "usage",
        "engaged_1",
        "price_1",
        "engaged_2",
        "price_2",
        "engaged_3",
        "price_3",
        "renewed",
        "revenue",
    ]
    # "It has no missing value, because these data have no censoring."
    assert int(frame.isna().sum().sum()) == 0
    assert (frame["revenue"] == frame["renewed"] * frame["price_3"]).all()
    assert frame["usage"].abs().max() <= pricing_law._LAW["usage_bound"] == 2.5
    for month in (1, 2, 3):
        engaged = frame[f"engaged_{month}"]
        assert set(engaged.unique()) == {0.0, 1.0}
        assert (frame[f"price_{month}"] >= 4 + 2 * engaged).all()
        assert (frame[f"price_{month}"] <= 8 + 2 * engaged).all()
    # The true values the page quotes come from the quadrature of the law.
    exact = pricing_truth()
    for key, value in exact.items():
        assert truth[key] == pytest.approx(value, abs=1e-12), key
    levels = {
        plan: (round(truth[f"ey_regimen[{plan}]"], 4), round(truth[f"ey_revenue[{plan}]"], 4))
        for plan in ("natural_prices", "price_up", "price_down")
    }
    assert levels == {
        "natural_prices": (0.7301, 5.3765),
        "price_up": (0.6298, 5.0643),
        "price_down": (0.8450, 5.5019),
    }
    assert [round(truth[_key(plan)], 4) for plan in PLANS] == [-0.1003, 0.1149]
    assert [round(truth[f"ate_revenue[{plan} vs natural_prices]"], 4) for plan in PLANS] == [
        -0.3122,
        0.1254,
    ]
    # Nonzero witnesses: the effects are far from zero, so a sign flip in the law fails here.
    assert truth[_key("price_up")] < -0.05 and truth[_key("price_down")] > 0.05
    assert truth["ate_revenue[price_up vs natural_prices]"] < -0.2
    assert truth["ate_revenue[price_down vs natural_prices]"] > 0.05

    # Step 3: the three arrows of the graph, read in the data.
    gap_1 = frame["price_1"] - (6 + 2 * frame["engaged_1"])
    assert gap_1.equals(namespace["gap_1"])
    moved = frame.groupby(np.where(gap_1 > 0, "above the centre", "at or below"))[
        "engaged_2"
    ].mean()
    assert moved.round(3).to_dict() == {"above the centre": 0.619, "at or below": 0.696}
    assert moved["at or below"] - moved["above the centre"] > 0.05
    by_engagement = pd.DataFrame(
        {f"month {t}": frame.groupby(f"engaged_{t}")[f"price_{t}"].mean() for t in (1, 2, 3)}
    )
    assert by_engagement.round(3).loc[0.0].tolist() == [5.946, 5.973, 5.948]
    assert by_engagement.round(3).loc[1.0].tolist() == [8.033, 8.006, 7.980]
    assert (by_engagement.loc[1.0] - by_engagement.loc[0.0] > 1.9).all()
    renewed = frame.groupby("engaged_3")["renewed"].mean()
    assert renewed.round(3).to_dict() == {0.0: 0.598, 1.0: 0.809}
    # A_{t-1} to A_t and L_{t-1} to L_t, read in the data with a nonzero witness each.
    after = frame.groupby("engaged_1")["engaged_2"].mean()
    assert after[1.0] - after[0.0] > 0.1
    gap_2 = frame["price_2"] - (6 + 2 * frame["engaged_2"])
    design = np.column_stack([np.ones(len(frame)), gap_1, frame["engaged_2"], frame["usage"]])
    slope = np.linalg.lstsq(design, gap_2.to_numpy(), rcond=None)[0][1]
    # The band truncation attenuates the law's 0.3, so the data slope is a positive witness only.
    assert 0.08 < slope < pricing_law._LAW["price_gap"]


#: The arrow rows of the graph in the ``plan`` cell, by parent and child.
_GRAPH_ROWS = (
    ("usage", "every engagement score, every price, and the renewal"),
    ("engaged_t", "price_t"),
    ("price_t-1", "price_t"),
    ("engaged_t-1", "price_t"),
    ("price_t-1", "engaged_t"),
    ("engaged_t-1", "engaged_t"),
    ("engaged_t", "renewed"),
    ("price_t", "renewed"),
)


def _check_graph(namespace: dict[str, Any]) -> None:
    """The causal graph of the question names the arrows that the law and the data carry."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    plan = "".join(next(c for c in notebook["cells"] if c["id"] == "plan")["source"])
    fence = plan.split("```text")[1].split("```")[0]
    rows = [
        tuple(part.strip() for part in line.split("──▶", 1))
        for line in fence.splitlines()
        if "──▶" in line and not line.strip().startswith("not drawn")
    ]
    assert [(parent, child.split("   ")[0].strip()) for parent, child in rows] == list(_GRAPH_ROWS)
    assert "not drawn: an unrecorded cause U with U ──▶ price and U ──▶ renewed" in fence
    law = pricing_law._LAW
    centre = law["centre_engaged"]
    # Each drawn arrow is a nonzero coefficient of the law, with the sign the row states.
    for name in ("engaged_usage", "price_usage", "renew_usage"):
        assert law[name] != 0.0, name
    assert centre > 0.0  # "the band rises with engagement"
    assert law["price_gap"] > 0.0  # "the engine carries part of last month's price gap"
    assert -law["price_gap"] * centre != 0.0  # "it measures that gap from last month's band"
    assert law["engaged_gap"] < 0.0  # "a price above the band's middle lowers engagement"
    assert law["engaged_lag"] > 0.0  # "engagement persists"
    assert law["renew_engaged"] > 0.0
    # "month 3's price matters most": the month-3 gap weighs more than each earlier gap.
    assert law["renew_gap"] < 0.0 and abs(law["renew_gap"]) > abs(law["renew_lag"]) > 0.0
    # "Month-3 revenue is the month-3 price for a renewer and 0 otherwise", which varies.
    frame = namespace["frame"]
    renewers = frame["renewed"] == 1
    assert frame.loc[renewers, "revenue"].equals(frame.loc[renewers, "price_3"])
    assert frame.loc[renewers, "price_3"].std() > 1.0
    assert (frame.loc[~renewers, "revenue"] == 0.0).all()
    assert (law["band_below"], law["band_above"], law["centre_base"]) == (2.0, 2.0, 6.0)
    # The band $[4 + 2e, 8 + 2e]$ and the centre $6 + 2e$ of the situation table.
    engaged = np.array([0.0, 1.0])
    np.testing.assert_allclose(pricing_law._lower(engaged), [4.0, 6.0])
    np.testing.assert_allclose(pricing_law._upper(engaged), [8.0, 10.0])
    np.testing.assert_allclose(pricing_law._centre(engaged), [6.0, 8.0])
    # The correct-specification claim of Step 8: the logit of the renewal law is exactly linear in
    # the usage, the three engagement scores, and the three prices.
    rng = np.random.default_rng(5)
    points = rng.uniform(size=(40, 7)) * [2, 1, 1, 1, 4, 4, 4] + [-1, 0, 0, 0, 4, 4, 4]
    points[:, 1:4] = np.round(points[:, 1:4])

    def renewal_logit(point: np.ndarray) -> float:
        u, a1, a2, a3, p1, p2, p3 = point
        gaps = [p - (6 + 2 * a) for p, a in ((p1, a1), (p2, a2), (p3, a3))]
        return float(logit(pricing_law._renewal(u, a3, gaps[2], gaps[0] + gaps[1])))

    values = np.array([renewal_logit(point) for point in points])
    design = np.column_stack([np.ones(len(points)), points])
    fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    np.testing.assert_allclose(fitted, values, atol=1e-9)


def _check_design(namespace: dict[str, Any]) -> None:
    protocol, study = namespace["protocol"], namespace["study"]
    assert protocol == pricing_protocol()
    assert protocol.fingerprint == FINGERPRINT
    # The eligibility rests on time zero alone: a three-month term with no exit before the month-3
    # renewal.  No event after time zero, such as staying open through month 3, defines it.
    assert (
        "A three-month term that starts at time zero, with no exit before the month-3 renewal"
        in protocol.eligibility
    )
    assert protocol.intercurrent_event_handling[0].startswith("None: the three-month term")
    protocol_text = " ".join(protocol.summary_lines()).lower()
    assert "cancel" not in protocol_text and "stays open" not in protocol_text
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 3
    assert_protocol_recorded(
        NOTEBOOK,
        "protocol",
        protocol,
        namespace["result"],
        namespace["revenue_result"],
        namespace["rise_fit"],
    )
    assert f"protocol: schema 1; {FINGERPRINT}" in stored_output(NOTEBOOK, "protocol")
    printed = stored_output(NOTEBOOK, "identify")
    for month, names in (
        (1, ("usage", "engaged_1")),
        (2, ("usage", "engaged_1", "engaged_2")),
        (3, ("usage", "engaged_1", "engaged_2", "engaged_3")),
    ):
        assert study.data.history_names(month) == names
        assert f"history_names({month}): {names}" in printed
    assert (
        "history at price_3: ['usage', 'engaged_1', 'engaged_2', 'engaged_3', 'price_1', 'price_2']"
    ) in printed
    assert "history at price_1: ['usage', 'engaged_1']" in printed
    # The short summary prints the analysis and leaves the record to the protocol step.
    assert "causal study protocol" not in printed and FINGERPRINT not in printed
    for assumption in (
        "consistency: each observed history equals the potential history",
        "no interference: one unit's potential history",
        "strong sequential randomization (Díaz et al. 2023, Assumption 3) for the modified "
        "treatment policy effect",
        "positivity: the dose a policy assigns",
        "modified treatment policy effect: the mean outcome had each node's natural dose",
        "stochastic intervention (LMTP-SI)",
    ):
        assert assumption in printed
    design = namespace["pricing_design"]("renewed")
    assert isinstance(design, LongitudinalTreatment)
    assert list(design.treatment) == ["price_1", "price_2", "price_3"]
    assert dict(cast(Mapping[str, Any], design.continuous_treatment)) == PRICE_BANDS
    assert list(PRICE_PLANS) == ["natural_prices", "price_up", "price_down"]
    for label in ("price_up", "price_down"):
        assert all(policy.policy_kind == "known" for policy in _policies(PRICE_PLANS[label]))
    # The 12% rise and the 15% cut of the plan.
    frame = namespace["frame"]
    for month, policy in enumerate(_policies(PRICE_PLANS["price_up"]), start=1):
        price = frame[f"price_{month}"].to_numpy()
        moved = np.where(price < (8 + 2 * frame[f"engaged_{month}"]) / 1.12, 1.12 * price, price)
        assert (moved <= 8 + 2 * frame[f"engaged_{month}"] + 1e-12).all()
        assert policy.name == f"price_up at {month}"
    assert price_rise(1.12).label == "price_up"


def _check_renewal_fit(namespace: dict[str, Any]) -> None:
    result, truth = namespace["result"], namespace["truth"]
    assert list(result.estimates) == [_key(plan) for plan in PLANS]
    method = namespace["ltmle"]
    assert isinstance(method, TMLEMethod)
    assert not method.cross_fitting.enabled
    assert method.runtime.random_state == 1000 and method.runtime.n_jobs == 1
    models: ModelSpec = method.models
    assert models.ratio == "classifier"
    assert isinstance(models.outcome_learner, LogisticRegression)
    assert isinstance(models.pseudo_learner, LinearRegression)
    ratio_model = namespace["ratio_model"]
    assert models.treatment_learner is ratio_model and isinstance(ratio_model, Pipeline)
    assert [type(step).__name__ for _, step in ratio_model.steps] == [
        "PolynomialFeatures",
        "StandardScaler",
        "LogisticRegression",
    ]
    assert ratio_model.steps[0][1].degree == 2 and ratio_model.steps[2][1].C == 1e6
    # A tight tolerance makes the lbfgs fit converge. An unconverged fit leaves the largest density
    # ratios dependent on the platform.
    assert ratio_model.steps[2][1].tol == 1e-8 and ratio_model.steps[2][1].max_iter == 100_000
    summary = result.summary()
    assert dict(summary.header)["Out-of-fold predictions"] == "no"
    assert dict(summary.header)["Probability limits"] == "[0.01, 1]"
    # The limits move no probability, so the summary prints no capped note.
    assert "weights_capped" not in summary.note_codes
    # The summary prints the weight tail at one decimal. The stored output fixes the printed text,
    # and the fresh summary agrees with it within the rounding of that decimal.
    printed_tail = stored_output(NOTEBOOK, "ltmle")
    plans_table = summary.table("What each plan assigns and who followed it to the end")
    rows = {row[0].value: row for row in plans_table.rows}
    for plan, shown in (("price_up", 13.7), ("price_down", 31.2)):
        assert rows[plan][1].value == 10000
        assert re.search(rf"^{plan} +10,000 of 10,000 +{shown} ", printed_tail, re.MULTILINE)
        assert rows[plan][2].value == pytest.approx(shown, abs=0.11)
    assert result.simultaneous is not None
    assert round(result.simultaneous.critical_value, 3) == 2.221
    assert round(result.simultaneous.pointwise_critical_value, 3) == 1.960
    assert "experimental" in summary.note("simultaneous_bands").text

    up, down = (result[_key(plan)] for plan in PLANS)
    # "The `price_up` contrast is -0.1025, with a standard error of 0.0065 and a 95% interval of
    # (-0.1152, -0.0899) ... The `price_down` contrast is 0.1050, with an interval of (0.0896,
    # 0.1203)." The seed is the first of the swept range, so the coverage is a record of the draw.
    assert (round(up.psi, 4), round(up.std_error, 4)) == (-0.1025, 0.0065)
    assert [round(value, 4) for value in up.ci] == [-0.1152, -0.0899]
    assert (round(down.psi, 4), round(down.std_error, 4)) == (0.1050, 0.0078)
    assert [round(value, 4) for value in down.ci] == [0.0896, 0.1203]
    for plan in PLANS:
        assert covers(result[_key(plan)], truth[_key(plan)])
    table = namespace["against_truth"](result, "ate_regimen")
    assert table["covers"].all() and table["support"].tolist() == ["supported"] * 2
    assert up.ci[1] < 0 < down.ci[0]
    # "A 12% rise lowers the month-3 renewal share by about ten points."
    assert -0.11 < up.psi < -0.09
    for plan in PLANS:
        estimate = result[_key(plan)]
        assert estimate.support.label == "supported"
        assert estimate.support.reason == "Row G39.ltmle.tau3 covers this configuration."
    # A supported row prints no support column and no experimental note.
    assert vocabulary.SUPPORT not in result.summary().table("estimates").columns
    assert "experimental" not in result.summary().note_codes


def _check_revenue_fit(namespace: dict[str, Any]) -> None:
    result, truth = namespace["revenue_result"], namespace["truth"]
    method = namespace["revenue_method"]
    assert isinstance(method.models.outcome_learner, LinearRegression)
    assert method.targeting.q_bounds == (0.0, 10.0)
    assert not method.cross_fitting.enabled and method.models.ratio == "classifier"
    assert namespace["revenue_study"].data.history_names(3) == (
        "usage",
        "engaged_1",
        "engaged_2",
        "engaged_3",
    )
    up, down = (result[_key(plan)] for plan in PLANS)
    # "The `price_up` revenue contrast is -0.3129, with an interval of (-0.4224, -0.2035). The
    # true contrast is -0.3122." "The `price_down` contrast is 0.0743, with an interval of
    # (-0.0307, 0.1792)... the interval contains zero on this draw."
    assert (round(up.psi, 4), round(up.std_error, 4)) == (-0.3129, 0.0559)
    assert [round(value, 4) for value in up.ci] == [-0.4224, -0.2035]
    assert (round(down.psi, 4), round(down.std_error, 4)) == (0.0743, 0.0536)
    assert [round(value, 4) for value in down.ci] == [-0.0307, 0.1792]
    revenue_truth = {plan: truth[f"ate_revenue[{plan} vs natural_prices]"] for plan in PLANS}
    assert round(revenue_truth["price_up"], 4) == -0.3122
    assert round(revenue_truth["price_down"], 4) == 0.1254
    assert covers(up, revenue_truth["price_up"]) and covers(down, revenue_truth["price_down"])
    assert down.ci[0] < 0 < down.ci[1] and up.ci[1] < 0
    table = namespace["against_truth"](result, "ate_revenue")
    assert table["covers"].all()
    for plan in PLANS:
        estimate = result[_key(plan)]
        assert estimate.support.label == "supported"
        assert estimate.support.reason == "Row G39.ltmle.tau3_revenue covers this configuration."
    # The revenue outcome is the renewal times the month-3 price, in the declared support.
    frame = namespace["frame"]
    assert frame["revenue"].between(0.0, 10.0).all()
    # The linear outcome regression cannot match the law: revenue is zero for a lapsed subscriber
    # and the price for a renewed one, so the fit of the last node leaves a residual structure.
    last = LinearRegression().fit(
        frame[["usage", "engaged_1", "engaged_2", "engaged_3", "price_1", "price_2", "price_3"]],
        frame["revenue"],
    )
    residual = frame["revenue"] - last.predict(
        frame[["usage", "engaged_1", "engaged_2", "engaged_3", "price_1", "price_2", "price_3"]]
    )
    assert residual.std() > 1.5


def _check_failure_mode(namespace: dict[str, Any]) -> None:
    result, truth = namespace["result"], namespace["truth"]
    one_regression = namespace["one_regression"]
    adjust_columns, omit_columns = namespace["adjust_columns"], namespace["omit_columns"]
    assert adjust_columns == [
        "usage",
        "engaged_1",
        "engaged_2",
        "engaged_3",
        "price_1",
        "price_2",
        "price_3",
    ]
    assert omit_columns == ["usage", "price_1", "price_2", "price_3"]
    adjust = {plan: one_regression(adjust_columns, plan) for plan in PLANS}
    omit = {plan: one_regression(omit_columns, plan) for plan in PLANS}
    assert {plan: round(value, 4) for plan, value in adjust.items()} == {
        "price_up": -0.0796,
        "price_down": 0.0938,
    }
    assert {plan: round(value, 4) for plan, value in omit.items()} == {
        "price_up": -0.0202,
        "price_down": 0.0268,
    }
    printed = stored_output(NOTEBOOK, "failure-mode")
    assert "price_up   -0.1003 -0.1025    -0.1152     -0.0899 -0.0796 -0.0202" in printed
    assert "price_down  0.1149  0.1050     0.0896      0.1203  0.0938  0.0268" in printed
    up, down = (result[_key(plan)] for plan in PLANS)
    # price_up: the adjust regression lies outside the LTMLE interval and between LTMLE and omit.
    assert not covers(up, adjust["price_up"])
    assert adjust["price_up"] > up.ci[1]
    assert up.psi < adjust["price_up"] < omit["price_up"]
    # price_down: the adjust regression lies inside the LTMLE interval, and still between.
    assert covers(down, adjust["price_down"])
    assert omit["price_down"] < adjust["price_down"] < down.psi
    assert not covers(down, omit["price_down"]) and not covers(up, omit["price_up"])
    # "Its exact limit ... is further from the truth than the LTMLE estimate is."
    both = {plan: truth[f"ate_both_held[{plan}]"] for plan in PLANS}
    for plan in PLANS:
        key = _key(plan)
        assert abs(both[plan] - truth[key]) > abs(result[key].psi - truth[key])
        assert abs(result[key].psi - truth[key]) < abs(adjust[plan] - truth[key])
        assert abs(adjust[plan] - truth[key]) < abs(omit[plan] - truth[key])
    # The regression with engagement estimates the `both held` value, and the sample value lies
    # within three standard errors of it. A regression has no sampling error of this size.
    assert abs(adjust["price_up"] - both["price_up"]) < 0.01
    assert abs(adjust["price_down"] - both["price_down"]) < 0.01
    paths = namespace["paths"]
    assert {
        plan: {k: round(v, 4) for k, v in column.items()} for plan, column in paths.items()
    } == {
        "price_up": {
            "truth": -0.1003,
            "engagement held": -0.0906,
            "engine held": -0.0900,
            "both held": -0.0804,
        },
        "price_down": {
            "truth": 0.1149,
            "engagement held": 0.1050,
            "engine held": 0.1081,
            "both held": 0.0973,
        },
    }
    # "Each path supplies about half of the gap": the share of the gap that holding one path
    # closes lies between 0.4 and 0.6 for price_up, and each path closes some of it for both plans.
    gap = paths["price_up"]["truth"] - paths["price_up"]["both held"]
    for name in ("engagement held", "engine held"):
        share = (paths["price_up"]["truth"] - paths["price_up"][name]) / gap
        assert 0.4 < share < 0.6, name
    for plan in PLANS:
        assert abs(paths[plan]["both held"] - paths[plan]["truth"]) > 0.015
    # The regression gap is structural: the limit of the regression is far from the truth, and the
    # omit regression is farther.
    assert abs(truth[_key("price_up")] - both["price_up"]) >= 0.015
    # The exact limit of the omit regression, one draw of 400000, is in the committed probe.
    summary = (Path(__file__).resolve().parents[3] / _PROBE / "summary.log").read_text()
    assert "omit price_up: limit -0.0203" in summary and "omit price_down: limit 0.0270" in summary


def _check_support(namespace: dict[str, Any]) -> None:
    truth, frame = namespace["truth"], namespace["frame"]
    rise_fit, result = namespace["rise_fit"], namespace["result"]
    assert list(namespace["rises"]) == ["natural_prices", "up_6", "price_up", "up_25"]
    support = namespace["support"]
    month3 = namespace["month3"].set_index("regimen")
    assert list(month3.index) == ["up_6", "price_up", "up_25"]
    # The weight tail is the one quantity that a platform moves in the last digits.
    assert month3["max_weight"].to_dict() == pytest.approx(
        {"up_6": 5.712, "price_up": 13.704, "up_25": 19.976}, rel=WEIGHT_TAIL_REL
    )
    assert month3["effective_n"].to_dict() == pytest.approx(
        {"up_6": 6629.661, "price_up": 3508.672, "up_25": 1977.153}, rel=WEIGHT_TAIL_REL
    )
    assert (support["share_truncated"] == 0.0).all()
    assert support.equals(rise_fit.diagnostics.support().to_frame())
    # The reading says the weight rises, the effective n falls, and the interval widens with the
    # factor. Each relation is strict, and the estimates move away from zero.
    assert month3["max_weight"].is_monotonic_increasing and month3["max_weight"].is_unique
    assert month3["effective_n"].is_monotonic_decreasing and month3["effective_n"].is_unique
    ceiling = 8 + 2 * frame["engaged_3"]
    held = {
        plan: float((frame["price_3"] >= ceiling / factor).mean())
        for plan, factor in {"up_6": 1.06, "price_up": 1.12, "up_25": 1.25}.items()
    }
    assert {plan: round(value, 4) for plan, value in held.items()} == {
        "up_6": 0.0862,
        "price_up": 0.1916,
        "up_25": 0.4505,
    }
    assert held["up_6"] < held["price_up"] < held["up_25"]
    # The mechanism of Step 9: no plan price lands below the factor times the lower limit, so the
    # exact month-3 ratio is zero for a subscriber whose engine price lies there.
    lowest = 4 + 2 * frame["engaged_3"]
    zero = {
        plan: float((frame["price_3"] < factor * lowest).mean())
        for plan, factor in {"up_6": 1.06, "price_up": 1.12, "up_25": 1.25}.items()
    }
    assert {plan: round(value, 4) for plan, value in zero.items()} == {
        "up_6": 0.0559,
        "price_up": 0.1231,
        "up_25": 0.3044,
    }
    assert zero["up_6"] < zero["price_up"] < zero["up_25"]
    printed_support = stored_output(NOTEBOOK, "support")
    for plan in zero:
        row = rf"^{plan}\s+{zero[plan]:.4f}\s+{held[plan]:.4f}\s"
        assert re.search(row, printed_support, re.MULTILINE), plan
    assert "share_assigned_1" not in printed_support
    _check_exact_node_ratio(frame, zero, held)
    keys = {plan: _key(plan) for plan in ("up_6", "price_up", "up_25")}
    estimates = [rise_fit[key] for key in keys.values()]
    widths = [estimate.ci[1] - estimate.ci[0] for estimate in estimates]
    assert [round(width, 4) for width in widths] == [0.0129, 0.0254, 0.0393]
    assert widths == sorted(widths)
    assert [round(estimate.psi, 4) for estimate in estimates] == [-0.0598, -0.1025, -0.1223]
    assert [round(truth[key], 4) for key in keys.values()] == [-0.0587, -0.1003, -0.1283]
    # The ladder's 12% estimate is the Step 6 estimate: the same data, plan, learners, and seed.
    assert rise_fit[keys["price_up"]].psi == pytest.approx(result[_key("price_up")].psi, abs=1e-9)
    saturation = truth[keys["up_25"]] / truth[keys["up_6"]]
    assert round(saturation, 2) == 2.19 and round(25 / 6, 2) == 4.17
    assert saturation < 0.6 * 25 / 6
    for estimate in estimates:
        assert estimate.support.label == "supported"
        assert estimate.support.reason == "Row G39.ltmle.tau3 covers this configuration."
    printed = stored_output(NOTEBOOK, "support")
    assert printed.count(": supported") == 3
    assert "exact effect of up_25 over up_6: 2.19; a linear response gives 4.17" in printed
    # The label of each factor, and a known policy at each month.
    assert price_rise(1.06).label == "up_6" and price_rise(1.25).label == "up_25"
    for node, policy in enumerate(_policies(namespace["rises"]["up_25"]), start=1):
        assert policy.policy_kind == "known" and policy.name == f"up_25 at {node}"


def _check_exact_node_ratio(
    frame: pd.DataFrame, zero: dict[str, float], held: dict[str, float]
) -> None:
    """The exact month-3 ratio of a rise: zero in the bottom region, largest in the held region.

    For a rise by ``f`` at the engine price ``a``, with ``g`` the engine's price density, the node
    ratio is 0 on ``[l, f l)``, ``g(a / f) / (f g(a))`` on ``[f l, u / f)``, and
    ``1 + g(a / f) / (f g(a))`` on ``[u / f, u]``.  The mean of the ratio under the engine's law is
    one, which checks the formula.  Step 9 narrates the zero region and the held region.
    """
    law = pricing_law
    engaged_2, engaged_3 = frame["engaged_2"], frame["engaged_3"]
    gap_2 = frame["price_2"] - (6 + 2 * engaged_2)
    mean = law._price_mean(frame["usage"], engaged_3, gap_2)
    low, high = law._lower(engaged_3), law._upper(engaged_3)
    price = frame["price_3"].to_numpy()
    for plan, factor in {"up_6": 1.06, "price_up": 1.12, "up_25": 1.25}.items():
        moved = law._density(price / factor, mean, low, high) / (
            factor * law._density(price, mean, low, high)
        )
        below = price < factor * low
        top = price >= high / factor
        ratio = np.where(below, 0.0, np.where(top, 1.0 + moved, moved))
        assert below.mean() == pytest.approx(zero[plan])
        assert top.mean() == pytest.approx(held[plan])
        assert abs(ratio.mean() - 1.0) < 0.05, plan
        # A nonzero witness: the largest ratios all belong to the held region, and the held ratio
        # exceeds one there.  A sign error or a missing "1 +" fails here.
        largest = np.argsort(-ratio)[:100]
        assert top[largest].all(), plan
        assert (ratio[top] >= 1.0).all() and ratio.max() > 2.0


def _check_assessment(namespace: dict[str, Any]) -> None:
    ran = namespace["ran"].set_index("check")["status"]
    assert ran.to_dict() == {
        "score_equations": "passed",
        "support": "completed",
        "nuisance_models": "completed",
    }
    report = namespace["support_report"]
    result = namespace["result"]
    assert report.equals(result.diagnostics.support().to_frame())
    at_end = report[report["time"] == 3].set_index("regimen")
    assert at_end["max_weight"].to_dict() == pytest.approx(
        {"natural_prices": 1.0, "price_up": 13.704, "price_down": 31.240}, rel=WEIGHT_TAIL_REL
    )
    assert at_end["effective_n"].to_dict() == pytest.approx(
        {"natural_prices": 10000.0, "price_up": 3508.672, "price_down": 1554.276},
        rel=WEIGHT_TAIL_REL,
    )
    # "its largest weight grows from about 5 at month 1 to about 31 at month 3"
    down = report[report["regimen"] == "price_down"].set_index("time")["max_weight"]
    assert down.tolist() == pytest.approx([4.646, 13.093, 31.240], rel=WEIGHT_TAIL_REL)
    assert down.is_monotonic_increasing
    assert (report["share_truncated"] == 0.0).all()
    ratio = report["effective_n"] / report["n_followed"]
    assert namespace["ess_ratio"] == pytest.approx(0.1554, rel=WEIGHT_TAIL_REL)
    assert ratio.min() == pytest.approx(namespace["ess_ratio"])
    assert report.loc[ratio.idxmin(), "regimen"] == "price_down"
    assert report.loc[ratio.idxmin(), "time"] == 3
    # "about a third of the sample carries them": 3508.672 of 10000.
    assert 0.3 < at_end.loc["price_up", "effective_n"] / 10_000 < 0.4
    assert at_end["max_weight"].idxmax() == "price_down"
    # "It leaves 19.7% of the month-3 prices at the engine's value."
    held_down = float(namespace["held_down"])
    assert round(held_down, 3) == 0.197
    frame = namespace["frame"]
    floor = 4 + 2 * frame["engaged_3"]
    assert held_down == pytest.approx(float((frame["price_3"] < floor / 0.85).mean()))
    assert "price_down, share of month-3 prices left at the engine's value: 19.7%" in (
        stored_output(NOTEBOOK, "assessment")
    )
    # The nuisance report scores the outcome regression and leaves the density-ratio classifier out.
    outcome = namespace["outcome"]
    assert outcome["time"].tolist() == [3] and outcome["evaluation"].tolist() == ["in_sample"]
    assert round(float(outcome["calibration_slope"].iloc[0]), 3) == 1.004
    assert round(float(outcome["auc"].iloc[0]), 3) == 0.697
    assert namespace["nuisance"].omissions
    assert {o.reason for o in namespace["nuisance"].omissions if o.role == "treatment"} == {
        "continuous node ratio estimated by classification, no density"
    }


class _NeverFit(LogisticRegression):
    """A learner that fails the test if any fit reaches it."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the fit reached a learner")


def _check_own_plan_comment(namespace: dict[str, Any]) -> None:
    """The commented band and rule of the ``identify`` cell reproduce the page's 12% rise.

    The comment is the page's only example of a hand-written plan, so a slip in it would mislead
    a reader with their own data.  Each month's copy of the snippet builds the band and the rule,
    and the fit with them returns the ``price_up`` estimate of the page.
    """
    source = dict(notebook_code(NOTEBOOK))["identify"]
    start = source.index("#   from cleverly.interventions")
    end = source.index("#   ))", start) + len("#   ))")
    snippet = "\n".join(line.removeprefix("#   ") for line in source[start:end].splitlines())
    bands, rules = {}, []
    for month in (1, 2, 3):
        scope: dict[str, Any] = {"np": np}
        exec(
            snippet.replace("engaged_1", f"engaged_{month}").replace("up at 1", f"up at {month}"),
            scope,
        )
        bands[f"price_{month}"] = scope["band_1"]
        rules.append(scope["rise_1"])
    design = LongitudinalTreatment(
        outcome="renewed",
        treatment=["price_1", "price_2", "price_3"],
        baseline=["usage"],
        time_varying=[["engaged_1"], ["engaged_2"], ["engaged_3"]],
        continuous_treatment=bands,
    )
    plans = {
        "natural_prices": PRICE_PLANS["natural_prices"],
        "price_up": DynamicRegimen("price_up", tuple(rules)),
    }
    own = (
        CausalStudy(namespace["frame"], design=design)
        .identify(RegimeContrast(plans, reference="natural_prices"))
        .estimate(method=namespace["ltmle"])
    )
    page = namespace["result"][_key("price_up")]
    assert own[_key("price_up")].psi == pytest.approx(page.psi, abs=1e-9)
    assert own[_key("price_up")].ci == pytest.approx(page.ci, abs=1e-9)


def check_refits(namespace: dict[str, Any]) -> None:
    """Step 11: the refits are descriptive, and print the numbers that the reading quotes."""
    _check_own_plan_comment(namespace)
    report = namespace["report"]
    estimand = namespace["estimand"]
    assert estimand == _key("price_up")
    assert report.estimand == estimand
    assert tuple(test.name for test in report.tests) == (
        "placebo",
        "subset",
        "random_common_cause",
        "bootstrap_stability",
    )
    for test in report.tests:
        # "The `verdict` column reads `descriptive`": no registry row covers the fit.
        assert test.rung == "L0" and test.passed is None and test.verdict == "descriptive"
        assert len(test.values) == 10 and not test.failures
    assert report.passed is None
    summary = report.summary(detail="full")
    compared = summary.section(vocabulary.SECTION_COMPARED).lines
    assert any("another kind of outcome (binary)" in line for line in compared)
    # "From month 2 on, the donor is one of the 5 subscribers whose previous-month value is nearest"
    # and "The placebo prices therefore still depend on engagement through the band".
    note = summary.note("test_placebo").text
    assert "The placebo dose keeps each unit's declared band" in note
    assert "one of the 5 donors whose previous value is nearest" in note
    assert "so the placebo dose depends on the covariates through the band" in note
    placebo = report["placebo"]
    values = np.asarray(placebo.values)
    assert len(values) == 10
    assert float(values.mean()) == pytest.approx(0.00064, abs=0.000005)
    assert placebo.statistic == pytest.approx(0.72, abs=0.005)
    assert placebo.statistic_pvalue == pytest.approx(0.49, abs=0.005)
    assert float(values.std(ddof=1)) == pytest.approx(0.0028, abs=0.00005)
    se_over_sd = float(np.mean(placebo.standard_errors)) / float(values.std(ddof=1))
    assert se_over_sd == pytest.approx(1.98, abs=0.005)
    # The subset ratio, the noise-column shift and the bootstrap spread that the reading quotes.
    subset = report["subset"]
    assert subset.statistic == pytest.approx(0.98, abs=0.005)
    assert subset.statistic_pvalue == pytest.approx(0.93, abs=0.005)
    assert report["random_common_cause"].statistic == pytest.approx(-0.034, abs=0.0005)
    table = report.to_frame().set_index("test")
    spread = float(table.loc["bootstrap_stability", "refuted_sd"])
    assert spread / namespace["result"].estimates[estimand].std_error == pytest.approx(
        0.88, abs=0.01
    )
    # "The revenue contrast takes the same tests": the banded placebo runs on the revenue fit too.
    revenue = namespace["revenue_result"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseInfluenceWarning)
        revenue_placebo = refute(
            revenue, estimand=estimand, tests=("placebo",), n_replicates=2, random_state=1000
        )["placebo"]
    assert len(revenue_placebo.values) == 2 and revenue_placebo.passed is None


def _check_refit_limits(namespace: dict[str, Any]) -> None:
    """Step 11: the stored output, and each check the page skips is refused for its reason."""
    estimand = _key("price_up")
    printed = stored_output(NOTEBOOK, "refute")
    assert printed.count(" descriptive") == 4
    for name in ("placebo", "subset", "random_common_cause", "bootstrap_stability"):
        assert f"{name}: 10 refits, 0 failed" in printed
    # The checks the page does not run are refused before any refit, for the reasons it gives.
    result, revenue = namespace["result"], namespace["revenue_result"]
    # "Every policy contrast is then exactly 0": the placebo reads against 0.
    assert _placebo_null(result, estimand) == 0.0
    # The banded placebo is not refused on the revenue fit; check_refits runs it.
    assert refute_refusal(revenue, estimand=estimand, tests=("placebo",)) is None
    for fit in (result, revenue):
        refusal = refute_refusal(
            fit,
            estimand=estimand,
            tests=("future_treatment",),
            future_treatment_control=np.zeros(len(namespace["frame"])),
        )
        assert refusal is not None and "binary treatment at the later node" in refusal.message()
    refusal = refute_refusal(result, estimand=estimand, tests=("simulated_outcome",))
    assert refusal is not None and "'gaussian' process" in refusal.message()
    assert "family='binomial'" in refusal.message()
    refusal = refute_refusal(revenue, estimand=estimand, tests=("simulated_outcome",))
    assert refusal is not None and "q_bounds=(0.0, 10.0)" in refusal.message()
    refusal = refute_refusal(
        result,
        estimand=estimand,
        tests=("negative_control_outcome",),
        negative_control_outcome=namespace["frame"]["usage"].to_numpy(),
        negative_control_kind="pre_treatment",
    )
    assert refusal is not None and "adjustment" in refusal.message().lower()
    with pytest.raises(CapabilityError, match="F16"):
        evalue(result, estimand)
    # The later covariates `engaged_2` and `engaged_3` are why the simulated common cause refuses.
    refusal = simulated_confounding_refusal(result, estimand)
    assert refusal is not None
    assert "no time-varying covariate after the first treatment node" in refusal.message()
    assert "F13" in refusal.message()


def _check_refusals(namespace: dict[str, Any]) -> None:
    frame, study = namespace["frame"], namespace["study"]
    # "A pure 12% scale ... the fit refuses that plan before any learner."
    forward, inverse, derivative = pricing_law._scaled(1.12)
    raw = DynamicRegimen(
        "raw",
        tuple(
            ModifiedPolicy(
                f"raw at {month}",
                pieces=(Piece(-np.inf, np.inf, forward, inverse, derivative),),
                policy_kind="known",
            )
            for month in (1, 2, 3)
        ),
    )
    assert (1.12 * frame["price_3"] > 8 + 2 * frame["engaged_3"]).any()
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=_NeverFit(),
            pseudo_learner=LinearRegression(),
            treatment_learner=_NeverFit(),
            ratio="classifier",
        ),
        cross_fitting=namespace["ltmle"].cross_fitting,
        runtime=namespace["ltmle"].runtime,
    )
    with pytest.raises(DataError, match="not identified under the declared support"):
        study.identify(
            RegimeContrast(
                {"natural_prices": PRICE_PLANS["natural_prices"], "raw": raw},
                reference="natural_prices",
            )
        ).estimate(method=method)
    # "the fit refuses one on a continuous price, because its mean cannot be estimated at the usual
    # rate"
    static = DynamicRegimen("static", (6.0, 6.0, 6.0))
    with pytest.raises(CapabilityError, match="no regular estimator of it exists"):
        study.identify(
            RegimeContrast(
                {"natural_prices": PRICE_PLANS["natural_prices"], "static": static},
                reference="natural_prices",
            )
        ).estimate(method=method)

    # Step 11: a covariate-drop benchmark cannot run, because a longitudinal fit refuses an empty
    # baseline.  The same call with `usage` builds the study, so the empty baseline is the cause.
    def study_with(baseline: list[str]) -> Any:
        design = LongitudinalTreatment(
            outcome="renewed",
            treatment=["price_1", "price_2", "price_3"],
            baseline=baseline,
            time_varying=[["engaged_1"], ["engaged_2"], ["engaged_3"]],
            continuous_treatment=PRICE_BANDS,
        )
        return CausalStudy(frame, design=design, protocol=namespace["protocol"])

    study_with(["usage"])
    with pytest.raises(DataError, match="at least one baseline covariate"):
        study_with([])
    # "IPWMethod refuses a modified treatment policy."
    effect = namespace["effect"]
    with pytest.raises(CapabilityError, match="IPW of a modified treatment policy"):
        effect.estimate(method=IPWMethod())


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _line(text: str, *parts: str) -> str:
    return next(row for row in _lines(text) if all(part in row for part in parts))


def _check_probe_logs() -> None:
    """The committed probe logs print the repeated-draw numbers that the readings quote."""
    folder = Path(__file__).resolve().parents[3] / _PROBE
    sweep = (folder / "summary.log").read_text()
    assert sweep.startswith("draws 200, seeds 1000 to 1199, n = 10000, complete data")

    # Step 6: coverage and bias of the renewal contrasts.
    assert "coverage 186 of 200" in _line(sweep, "  renewal  price_up")
    assert "coverage 187 of 200" in _line(sweep, "  renewal  price_down")
    assert "bias -0.0002 (MC SE 0.0005)" in _line(sweep, "  renewal  price_up")
    assert "bias +0.0004 (MC SE 0.0005)" in _line(sweep, "  renewal  price_down")
    # Every fit of the sweep carries the label `supported`, and no fit raised another warning.
    labels = [row for row in _lines(sweep) if "_label: supported on 200 of 200" in row]
    assert len(labels) == 7
    assert "renewal_warnings: other warnings []" in sweep
    assert "revenue_warnings: other warnings []" in sweep
    assert "SparseInfluenceWarning on 46 of 200" in _line(sweep, "revenue_warnings:")
    assert "revenue price_down SparseInfluenceWarning, seed 1000: False" in sweep
    # Step 7: the revenue cut, and the linear learner's bias.
    assert "revenue price_down interval contains zero: 60 of 200" in sweep
    assert "revenue price_up interval contains zero: 0 of 200" in sweep
    assert "coverage 184 of 200" in _line(sweep, "  revenue  price_up")
    assert "coverage 191 of 200" in _line(sweep, "  revenue  price_down")
    assert "bias +0.0099 (MC SE 0.0039); SD 0.0557" in _line(sweep, "  revenue  price_up")
    assert "bias +0.0094 (MC SE 0.0038); SD 0.0537" in _line(sweep, "  revenue  price_down")
    # Step 8: the regressions against LTMLE.
    up = sweep.split("  price_up: truth")[1].split("  price_down: truth")[0]
    down = sweep.split("  price_down: truth")[1].split("Node-3 support")[0]
    assert "adjust further from the truth than LTMLE: 193 of 200" in up
    assert "adjust outside the LTMLE interval: 194 of 200" in up
    assert "omit outside the LTMLE interval: 200 of 200" in up
    assert "adjust between LTMLE and omit: 200 of 200" in up
    assert "adjust further from the truth than LTMLE: 185 of 200" in down
    assert "adjust outside the LTMLE interval: 127 of 200" in down
    assert "omit outside the LTMLE interval: 200 of 200" in down
    assert "adjust between LTMLE and omit: 199 of 200" in down
    assert "omit price_up: limit -0.0203, truth -0.1003" in sweep
    assert "omit price_down: limit 0.0270, truth 0.1149" in sweep
    assert "adjust price_up: limit -0.0801, truth -0.1003" in sweep
    # The probe truth is an independent transcription of the law: it agrees to 1e-12.
    difference = _line(sweep, "largest difference")
    assert float(difference.split(":")[1]) < 1e-12
    assert "same keys: True" in sweep
    # Step 9 and 10: the support of the ladder and of the plans.
    assert "width increases with the factor: 200 of 200" in sweep
    assert "effective n decreases with the factor: 200 of 200" in sweep
    assert "max weight increases with the factor: 195 of 200" in sweep
    assert sweep.count("no weight truncated: 200 of 200 (1.000)") == 5
    assert "max weight mean 14.8178, SD 2.0257" in sweep
    assert "max weight mean 31.7571, SD 5.2131" in sweep
    assert "effective n over n mean 0.3607" in sweep and "effective n over n mean 0.1536" in sweep
    # The probe fits the page: the first row of the sweep is the draw that the page shows.
    first = pd.read_csv(folder / "sweep.csv", float_precision="round_trip")
    assert len(first) == 200 and first["seed"].tolist() == list(range(1000, 1200))
    row = first.iloc[0]
    assert round(row["y_price_up_psi"], 4) == -0.1025 and round(row["y_price_up_se"], 4) == 0.0065
    assert round(row["y_price_down_psi"], 4) == 0.1050
    assert round(row["r_price_up_psi"], 4) == -0.3129
    assert round(row["r_price_down_psi"], 4) == 0.0743
    assert round(row["adjust_price_up"], 4) == -0.0796 and round(row["omit_price_up"], 4) == -0.0202
    assert round(row["adjust_price_down"], 4) == 0.0938
    assert row["maxw_price_up"] == pytest.approx(13.704, rel=WEIGHT_TAIL_REL)
    assert row["maxw_price_down"] == pytest.approx(31.240, rel=WEIGHT_TAIL_REL)
    assert round(row["ladder_up_6_psi"], 4) == -0.0598
    assert round(row["ladder_up_25_psi"], 4) == -0.1223
    # The seeds 1000 to 1059 of the revenue-learner probe are the first 60 of the sweep.
    learner = (folder / "revenue_learner.log").read_text()
    assert learner.startswith("draws 60, seeds 1000 to 1059, n = 10000")
    assert "price_up    linear    outcome learner: bias +0.0132 (MC SE 0.0074)" in learner
    assert "price_up    quadratic outcome learner: bias +0.0028 (MC SE 0.0075)" in learner
    # The two learners fit the same draws, so the log prints the paired difference of the estimates.
    assert "linear minus quadratic: mean +0.0104 (paired MC SE 0.0005)" in _line(
        learner, "price_up", "linear minus quadratic"
    )
    assert "linear minus quadratic: mean +0.0076 (paired MC SE 0.0010)" in _line(
        learner, "price_down", "linear minus quadratic"
    )
    # The renewal fit and the revenue fit of the page's draw raised no warning.
    assert pd.isna(row["renewal_warnings"]) and pd.isna(row["revenue_warnings"])
    # Step 11: the placebo of the page's fit at 400 draws.
    placebo = (folder / "placebo.log").read_text()
    assert "draws 400; mean -0.000128; sd 0.005649" in placebo
    assert "t = -0.45274 (two-sided p = 0.651" in placebo
    assert "mean SE over sd 1.006" in placebo
    assert "interval holds 0: 0.9450" in placebo


def _prose(cell_id: str) -> str:
    """The source of one markdown cell, with each run of whitespace collapsed to a space."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])
    return " ".join(source.split())


def _check_prose_phrases(namespace: dict[str, Any]) -> None:
    """The readings attach each repeated-draw count and decimal to the quantity the log gives it.

    A decimal that a log prints passes the narration test wherever it stands.  This check builds
    each phrase of the Details blocks from the parsed logs, so a count or a value moved to another
    quantity fails.
    """
    root = Path(__file__).resolve().parents[3]
    sweep = (root / _PROBE / "summary.log").read_text()
    learner = (root / _PROBE / "revenue_learner.log").read_text()

    def grab(pattern: str, row: str) -> str:
        found = re.search(pattern, row)
        assert found, (pattern, row)
        return found.group(1)

    def stat(label: str, plan: str, pattern: str) -> str:
        return grab(pattern, _line(sweep, f"  {label}  {plan}"))

    hits = {
        (label, plan): stat(label, plan, r"coverage (\d+) of 200")
        for label in ("renewal", "revenue")
        for plan in PLANS
    }
    bias = {
        (label, plan): stat(label, plan, r"bias ([+-]\d\.\d{4})")
        for label in ("renewal", "revenue")
        for plan in PLANS
    }
    error = {
        (label, plan): stat(label, plan, r"MC SE (\d\.\d{4})")
        for label in ("renewal", "revenue")
        for plan in PLANS
    }
    spread = {plan: stat("revenue", plan, r"SD (\d\.\d{4})") for plan in PLANS}

    renewal = _prose("ltmle-reading")
    assert (
        f"contained the truth on {hits['renewal', 'price_up']} and "
        f"{hits['renewal', 'price_down']} of 200 draws"
    ) in renewal
    assert error["renewal", "price_up"] == error["renewal", "price_down"]
    assert (
        f"The biases were {bias['renewal', 'price_up']} and "
        f"{bias['renewal', 'price_down'].lstrip('+')}, each with a Monte Carlo standard error of "
        f"{error['renewal', 'price_up']}"
    ) in renewal
    # The nominal count is 190 of 200.  A count near it has a Monte Carlo standard error of about 3.
    count_error = float(np.sqrt(200 * 0.95 * 0.05))
    assert round(count_error) == 3 and "a Monte Carlo standard error of about 3 draws" in renewal
    for plan in PLANS:
        z = (190 - int(hits["renewal", plan])) / count_error
        assert 0.9 < z < 1.5, plan
    assert "so both lie about one standard error below it" in renewal
    renewal_warned = grab(
        r"SparseInfluenceWarning on (\d+) of 200", _line(sweep, "renewal_warnings:")
    )
    assert (
        f"raised `SparseInfluenceWarning` on {renewal_warned} of 200 renewal fits, and this draw "
        "raised none"
    ) in renewal

    revenue = _prose("revenue-reading")
    up_bias = bias["revenue", "price_up"].lstrip("+")
    down_bias = bias["revenue", "price_down"].lstrip("+")
    assert (
        f"the estimates of the rise and the cut had biases of {up_bias} and {down_bias}. Each is "
        f"about two and a half Monte Carlo standard errors ({error['revenue', 'price_up']} and "
        f"{error['revenue', 'price_down']}) and about a sixth of the spread of the estimates "
        f"(standard deviations {spread['price_up']} and {spread['price_down']})"
    ) in revenue
    for plan, value in (("price_up", up_bias), ("price_down", down_bias)):
        assert 2.2 < float(value) / float(error["revenue", plan]) < 2.8
        assert 5 < float(spread[plan]) / float(value) < 7
    assert (
        f"contained the truth on {hits['revenue', 'price_up']} and "
        f"{hits['revenue', 'price_down']} of 200 draws"
    ) in revenue
    z_low = (190 - int(hits["revenue", "price_up"])) / count_error
    assert 1 < z_low < 2 and int(hits["revenue", "price_down"]) > 190
    assert (
        f"The count {hits['revenue', 'price_up']} lies between one and two standard errors "
        f"below the nominal 190, and {hits['revenue', 'price_down']} lies above it"
    ) in revenue
    zero = grab(r"revenue price_down interval contains zero: (\d+) of 200", sweep)
    assert f"The cut's interval contained zero on {zero} of 200 draws" in revenue
    warned = grab(r"SparseInfluenceWarning on (\d+) of 200", _line(sweep, "revenue_warnings:"))
    assert f"`SparseInfluenceWarning` came on {warned} of 200 revenue fits" in revenue
    quadratic = grab(r"price_up    quadratic outcome learner: bias \+(\d\.\d{4})", learner)
    linear = grab(r"price_up    linear    outcome learner: bias \+(\d\.\d{4})", learner)
    learner_error = grab(
        r"price_up    linear    outcome learner: bias .* \(MC SE (\d\.\d{4})", learner
    )
    quadratic_error = grab(
        r"price_up    quadratic outcome learner: bias .* \(MC SE (\d\.\d{4})", learner
    )
    assert (
        f"For the rise, the bias was {quadratic} with the quadratic model and {linear} with the "
        f"linear one, with Monte Carlo standard errors of {quadratic_error} and {learner_error}"
    ) in revenue
    # The paired line of the log, attributed by plan: the learner change moves the estimate by
    # about the size of the bias, with a paired standard error far below the shift.
    shift, shift_error = {}, {}
    for plan in PLANS:
        paired = _line(learner, plan, "linear minus quadratic")
        shift[plan] = grab(r"mean \+(\d\.\d{4})", paired)
        shift_error[plan] = grab(r"paired MC SE (\d\.\d{4})", paired)
    assert (
        f"the switch moves the rise's estimate by {shift['price_up']} on average (paired standard "
        f"error {shift_error['price_up']}) and the cut's by {shift['price_down']} (paired standard "
        f"error {shift_error['price_down']})"
    ) in revenue
    for plan in PLANS:
        assert float(shift[plan]) > 5 * float(shift_error[plan]), plan
    bias_linear = {
        plan: float(grab(rf"{plan}\s+linear\s+outcome learner: bias \+(\d\.\d{{4}})", learner))
        for plan in PLANS
    }
    # "most of the rise's bias and about half of the cut's"
    assert float(shift["price_up"]) / bias_linear["price_up"] > 0.7
    assert 0.4 < float(shift["price_down"]) / bias_linear["price_down"] < 0.6
    assert "the linear model carries most of the rise's bias and about half of the cut's" in revenue
    # With the quadratic learner the price_up bias is within one Monte Carlo standard error of zero.
    assert float(quadratic) < float(learner_error)
    assert "With the quadratic model, 60 draws cannot tell the rise's bias from zero" in revenue

    failure = _prose("failure-mode-reading")
    up_text = sweep.split("  price_up: truth")[1].split("  price_down: truth")[0]
    down_text = sweep.split("  price_down: truth")[1].split("Node-3 support")[0]

    def count(text: str, label: str) -> str:
        return grab(rf"{re.escape(label)}: (\d+) of 200", text)

    far = "adjust further from the truth than LTMLE"
    out = "adjust outside the LTMLE interval"
    between = "adjust between LTMLE and omit"
    assert (
        f"further from the truth than LTMLE on {count(up_text, far)} of 200 draws for the rise and "
        f"on {count(down_text, far)} for the cut"
    ) in failure
    assert (
        f"outside the LTMLE interval on {count(up_text, out)} draws for the rise and on "
        f"{count(down_text, out)} for the cut, and between the LTMLE and `omit` estimates on "
        f"{count(up_text, between)} and {count(down_text, between)} of 200 draws"
    ) in failure
    assert count(up_text, "omit outside the LTMLE interval") == "200"
    assert count(down_text, "omit outside the LTMLE interval") == "200"
    assert "The `omit` regression lay outside the LTMLE interval on every draw" in failure
    omit_up = grab(r"omit price_up: limit (-\d\.\d{4})", sweep)
    omit_down = grab(r"omit price_down: limit (\d\.\d{4})", sweep)
    assert f"is {omit_up} for the rise and {omit_down} for the cut" in failure

    assessment = _prose("assessment-reading")
    block = sweep.split("Node-3 support of the renewal fit")[1].split("Factor ladder")[0]

    def weights(plan: str) -> tuple[str, str]:
        part = block.split(f"  {plan}: max weight mean")[1].split("\n")[0]
        found = re.match(r" (\d+\.\d+), SD (\d+\.\d+)", part)
        assert found, part
        return found.group(1), found.group(2)

    mean_up, sd_up = weights("price_up")
    mean_down, sd_down = weights("price_down")
    ess_up = grab(r"price_up: effective n over n mean (\d\.\d+)", block)
    ess_down = grab(r"price_down: effective n over n mean (\d\.\d+)", block)
    assert (
        f"the largest month-3 weight of the rise averaged {float(mean_up):.2f}, with a standard "
        f"deviation of {float(sd_up):.2f}. For the cut it averaged {float(mean_down):.2f}, with a "
        f"standard deviation of {float(sd_down):.2f}."
    ) in assessment
    assert (
        f"averaged {float(ess_up):.3f} for the rise and {float(ess_down):.3f} for the cut"
    ) in assessment

    _check_reading_attribution(namespace)
    _check_phrase_rules()


def _interval(estimate: Any, places: int = 4) -> str:
    low, high = estimate.ci
    return f"{low:.{places}f} to {high:.{places}f}"


def _check_reading_attribution(namespace: dict[str, Any]) -> None:
    """Each seed-draw number of a reading sits beside the quantity that the draw gives it.

    The narration test accepts a decimal that any stored output prints.  This check builds the
    phrase of each reading from the namespace, so two numbers swapped between plans, or between a
    truth and an estimate, fail.  A relation word is tied to the ``covers`` relation it states.
    """
    truth, frame = namespace["truth"], namespace["frame"]
    result, revenue_result = namespace["result"], namespace["revenue_result"]
    up, down = (result[_key(plan)] for plan in PLANS)
    r_up, r_down = (revenue_result[_key(plan)] for plan in PLANS)
    t_up, t_down = (truth[_key(plan)] for plan in PLANS)
    rt_up, rt_down = (truth[f"ate_revenue[{plan} vs natural_prices]"] for plan in PLANS)
    one_regression = namespace["one_regression"]
    adjust = {plan: one_regression(namespace["adjust_columns"], plan) for plan in PLANS}
    omit = {plan: one_regression(namespace["omit_columns"], plan) for plan in PLANS}

    # At a glance: the answer, rounded, and the truth.
    title = _prose("title")
    assert f"(estimate {up.psi:.2f}, 95% interval {_interval(up, 3)})" in title
    assert f"by about {-r_up.psi:.2f} per subscriber (interval {_interval(r_up, 2)})" in title
    assert f"the true effects: {t_up:.3f} on renewal and {rt_up:.2f} on revenue" in title
    assert "about 10 percentage points" in title and -0.105 < up.psi < -0.095
    # "understates the renewal effect by about a fifth".
    assert 0.15 < 1 - adjust["price_up"] / t_up < 0.25
    assert "understates the renewal effect by about a fifth" in title

    # The data: the true values, attributed by plan.
    data = _prose("data-reading")
    natural = (truth["ey_regimen[natural_prices]"], truth["ey_revenue[natural_prices]"])
    assert (
        f"(`natural_prices`) the renewal share is {natural[0]:.3f} and mean revenue {natural[1]:.2f}"
    ) in data
    assert (
        f"The 12% rise (`price_up`) changes them by {t_up:.3f} and {rt_up:.2f}, and the 15% cut "
        f"(`price_down`) by {t_down:.3f} and {rt_down:.3f}"
    ) in data

    # The three associations.
    association = _prose("association-reading")
    gap_1 = frame["price_1"] - (6 + 2 * frame["engaged_1"])
    moved = frame.groupby(np.where(gap_1 > 0, "above the middle", "at or below"))[
        "engaged_2"
    ].mean()
    assert (
        f"lowers the share engaged in month 2, from {moved['at or below']:.3f} to "
        f"{moved['above the middle']:.3f}"
    ) in association
    assert moved["above the middle"] < moved["at or below"]
    month_1 = frame.groupby("engaged_1")["price_1"].mean()
    assert f"({month_1[1.0]:.3f} against {month_1[0.0]:.3f} in month 1)" in association
    for t in (1, 2, 3):
        by_month = frame.groupby(f"engaged_{t}")[f"price_{t}"].mean()
        assert 1.8 < by_month[1.0] - by_month[0.0] < 2.2, t
    assert "pay about 2 more each month" in association
    renewed = frame.groupby("engaged_3")["renewed"].mean()
    assert f"renew more often ({renewed[1.0]:.3f} against {renewed[0.0]:.3f})" in association

    # The renewal contrasts, and the truth that each interval contains on this draw.
    renewal = _prose("ltmle-reading")
    assert (
        f"the 12% rise changes the month-3 renewal share by {up.psi:.4f} (95% interval "
        f"{_interval(up)}), which contains the true {t_up:.4f}"
    ) in renewal
    assert covers(up, t_up)
    assert (
        f"The 15% cut changes it by {down.psi:.4f} ({_interval(down)}), which contains the true "
        f"{t_down:.4f}"
    ) in renewal
    assert covers(down, t_down)
    # `supported` is explained once, in the reading of the first fit that prints it.
    explanation = "a simulation study covers this kind of fit"
    assert explanation in renewal
    assert sum(_prose(cell_id).count(explanation) for cell_id in _MARKDOWN_IDS) == 1

    # The revenue contrasts.
    revenue = _prose("revenue-reading")
    assert (
        f"changes mean month-3 revenue by {r_up.psi:.4f} per subscriber (interval "
        f"{_interval(r_up)}), against a true {rt_up:.4f}"
    ) in revenue
    assert covers(r_up, rt_up) and r_up.ci[1] < 0
    assert (
        f"The cut's estimate, {r_down.psi:.4f}, has an interval of {_interval(r_down)}, which "
        f"includes zero"
    ) in revenue
    assert f"Its true effect is {rt_down:.4f}" in revenue
    assert covers(r_down, rt_down) and r_down.ci[0] < 0 < r_down.ci[1]

    # The regressions, the limits, and the interval relation of each plan.
    failure = _prose("failure-mode-reading")
    paths = namespace["paths"]
    held_up = paths["price_up"]
    assert not covers(up, adjust["price_up"])
    assert (
        f"For the 12% rise, LTMLE gives {up.psi:.4f} against the true {t_up:.4f}. The `adjust` "
        f"regression gives {adjust['price_up']:.4f}, outside the LTMLE interval"
    ) in failure
    assert (
        f"It estimates the `both held` value, {held_up['both held']:.4f}. "
        f"Holding only engagement gives {held_up['engagement held']:.4f} and only the engine's "
        f"next price {held_up['engine held']:.4f}, so each path carries about half of the gap"
    ) in failure
    assert f"`omit` gives {omit['price_up']:.4f}" in failure
    assert covers(down, adjust["price_down"])
    assert (
        f"For the 15% cut, `adjust` gives {adjust['price_down']:.4f}, inside the LTMLE interval "
        f"here, but it estimates {paths['price_down']['both held']:.4f}, further from the truth "
        f"({t_down:.4f}) than LTMLE's {down.psi:.4f} is"
    ) in failure

    # The ladder, row by row.
    support = _prose("support-reading")
    rise_fit = namespace["rise_fit"]
    factors = {"up_6": 1.06, "price_up": 1.12, "up_25": 1.25}
    held = {
        plan: float((frame["price_3"] >= (8 + 2 * frame["engaged_3"]) / factor).mean())
        for plan, factor in factors.items()
    }
    zero = {
        plan: float((frame["price_3"] < factor * (4 + 2 * frame["engaged_3"])).mean())
        for plan, factor in factors.items()
    }
    ladder_tail = _stored_weight_tail("support")
    widths = {}
    for plan in held:
        low, high = rise_fit[_key(plan)].ci
        widths[plan] = high - low
        assert covers(rise_fit[_key(plan)], truth[_key(plan)]), plan
    # "the largest month-3 weight goes from 5.7 to 20.0, the effective sample size falls from about
    # 6,630 to about 1,977, a fifth of the sample, and the interval width triples, from 0.0129 to
    # 0.0393"
    weight = {plan: float(ladder_tail[(plan, 3)][0]) for plan in held}
    size = {plan: float(ladder_tail[(plan, 3)][1]) for plan in held}
    assert (
        f"the largest month-3 weight goes from {weight['up_6']:.1f} to {weight['up_25']:.1f}, the "
        f"effective sample size falls from about {round(size['up_6']):,} to about "
        f"{round(size['up_25']):,}, a fifth of the sample, and the interval width triples, from "
        f"{widths['up_6']:.4f} to {widths['up_25']:.4f}"
    ) in support
    assert 0.15 < size["up_25"] / len(frame) < 0.25
    assert 2.8 < widths["up_25"] / widths["up_6"] < 3.3
    assert round(100 * held["up_25"]) == 45 and "leaves 45% of month-3 prices unchanged" in support
    assert all(0.0 < zero[plan] < held[plan] for plan in held)
    assert "Each estimate stays near its truth, and each interval contains it" in support
    assert all(abs(rise_fit[_key(plan)].psi - truth[_key(plan)]) < 0.01 for plan in held)
    assert [round(100 * zero[plan]) for plan in ("up_6", "up_25")] == [6, 30]
    assert "The share of month-3 prices with an exact ratio of zero grows from 6% to 30%" in support
    saturation = truth[_key("up_25")] / truth[_key("up_6")]
    assert (
        f"the true 25% effect is {saturation:.2f} times the 6% effect, against {25 / 6:.2f} for a "
        "straight line"
    ) in support

    # The overlap check of the three plans.
    assessment = _prose("assessment-reading")
    plan_tail = _stored_weight_tail("assessment")
    first, last = (round(float(plan_tail[("price_down", t)][0])) for t in (1, 3))
    assert f"grows from about {first} at month 1 to about {last} at month 3" in assessment
    assert "a check on the arithmetic, not on bias" in assessment
    stored_ratio = re.search(
        r"minimum effective-sample-size ratio: (\d+\.\d%)", stored_output(NOTEBOOK, "assessment")
    )
    assert stored_ratio is not None
    effective_down = float(plan_tail[("price_down", 3)][1])
    assert (
        f"By month 3 its 10,000 subscribers count as about {round(effective_down):,}, the smallest "
        f"share, {stored_ratio.group(1)}"
    ) in assessment
    assert (
        f"It leaves {float(namespace['held_down']):.1%} of month-3 prices unchanged" in assessment
    )
    outcome = namespace["outcome"]
    assert (
        f"calibration slope of {float(outcome['calibration_slope'].iloc[0]):.3f}, where 1 is ideal, "
        f"and an AUC of {float(outcome['auc'].iloc[0]):.3f}"
    ) in assessment

    # The answer: each estimate and interval, attributed.
    interpret = _prose("interpret")
    assert (
        f"would lower the month-3 renewal share by about 10 percentage points (95% interval "
        f"{_interval(up, 3)})"
    ) in interpret
    assert (
        f"lower mean month-3 revenue by about {-r_up.psi:.2f} per subscriber (interval "
        f"{_interval(r_up, 2)})"
    ) in interpret
    assert round(100 * down.psi, 1) == 10.5
    assert f"raises renewals by about 10.5 points ({_interval(down, 3)})" in interpret
    assert "about a fifth of the prices" in interpret
    assert all(0.15 < value < 0.25 for value in (held["price_up"], float(namespace["held_down"])))
    assert "who start a three-month term" in interpret
    assert "cancellation-free" not in interpret


def _stored_weight_tail(cell_id: str) -> dict[tuple[str, int], tuple[str, str]]:
    """The printed maximum weight and effective n of each ``(regimen, time)`` row of a stored table.

    The weight tail is the one printed quantity that a platform moves in the last digit, so the
    prose is tied to the stored text and the fresh run is compared to the stored value by tolerance.
    """
    rows = re.findall(
        r"^\s*(\w+)\s+(\d)\s+\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\d+\.\d+$",
        stored_output(NOTEBOOK, cell_id),
        flags=re.MULTILINE,
    )
    assert rows, cell_id
    return {(regimen, int(time)): (weight, size) for regimen, time, weight, size in rows}


_MARKDOWN_IDS = (
    "title",
    "plan",
    "setup-heading",
    "data-heading",
    "data-reading",
    "association-heading",
    "association-reading",
    "protocol-heading",
    "protocol-reading",
    "identify-heading",
    "identify-reading",
    "ltmle-heading",
    "ltmle-reading",
    "revenue-heading",
    "revenue-reading",
    "failure-mode-heading",
    "failure-mode-reading",
    "support-heading",
    "support-reading",
    "assessment-heading",
    "assessment-reading",
    "refute-heading",
    "refute-reading",
    "sensitivity",
    "interpret",
    "method-evidence",
    "other-estimators",
    "skipped-checks",
    "where-next",
)


def _check_phrase_rules() -> None:
    """The wording rules of the plan: how the policy, the population, and the ladder are named."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = {
        cell["id"]: " ".join("".join(cell["source"]).split())
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    }
    assert set(_MARKDOWN_IDS) == set(markdown)
    # No plan is a cap.  The refused alternative of the `plan` cell is the one place that names a
    # cap, and a floor appears only as the name `l_t` of the band's lower limit.
    for cell_id, text in markdown.items():
        found = re.findall(r"\bcap(?:s|ped)?\b", text, flags=re.IGNORECASE)
        assert len(found) == (1 if cell_id == "plan" else 0), (cell_id, found)
        assert not re.search(r"\bfloors?\b(?! \$l_t\$)", text, flags=re.IGNORECASE), cell_id
    assert "A rule that raised the price and capped it at the ceiling" in markdown["plan"]
    assert "band edge" not in " ".join(markdown.values())
    # The population rests on time zero: a three-month term, in the question, the protocol
    # reading, and the answer.  No reading defines it by an event after time zero.
    assert "Each subscriber starts a three-month term at time zero" in markdown["plan"]
    assert "no one can leave before the term ends" in markdown["plan"]
    assert "among subscribers who start a three-month term at time zero" in markdown["plan"]
    assert (
        "Eligibility is a three-month term that starts at time zero" in markdown["protocol-reading"]
    )
    assert "with no exit before the month-3 renewal" in markdown["protocol-reading"]
    for text in markdown.values():
        assert "stays open" not in text and "cancellation" not in text.lower()
    # The engine reads the last gap from the last month's band.
    assert "it measures that gap from last month's band" in markdown["plan"]
    # The mechanism of the ladder names the zero region and the held region.
    heading = markdown["support-heading"]
    assert (
        "A subscriber whose engine price lies below that value in month 3 has an exact" in heading
    )
    assert "month-3 ratio of zero, because the plan never produces such a price" in heading
    assert "Near the top, the plan's price can be a held or a raised one" in heading
    assert "carry the largest weights" in heading
    assert "moved prices carry a larger density ratio" not in heading
    # The refusal reads the declared band.
    assert "that price lies outside the declared band" in markdown["plan"]
    assert "no subscriber is observed at that price" not in markdown["plan"]
    # The covariate-drop benchmark gives the refusal of an empty baseline as its reason.
    assert "refuses an empty baseline" in markdown["skipped-checks"]
    assert "(../roadmap.md), item X63, part (c)" in markdown["skipped-checks"]
    assert "Dropping either one changes the question" not in markdown["skipped-checks"]
    # The association heading and reading claim an association only.
    assert "These tables show associations only" in markdown["association-reading"]
    assert "Engagement is therefore" not in markdown["association-reading"]
    # Each check the page tries to break the result with is named in plain words.
    refute_text = markdown["refute-heading"]
    for test in ("placebo", "subset", "random_common_cause", "bootstrap_stability"):
        assert f"`{test}`" in refute_text, test
    assert "None can detect a missing confounder" in refute_text
