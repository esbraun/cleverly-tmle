"""The reviewed semantic callback for ``docs/examples/sequential-pricing``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and passes
the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated decimals
are also compared against its stored outputs, and every decimal it writes is printed or listed in
:data:`UNPRINTED_DECIMALS` with its source.
"""

from __future__ import annotations

import json
import re
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
    TMLEMethod,
)
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
from cleverly.utils.bounds import logit
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "sequential-pricing.ipynb"

_PROBE = "reviews/notebook-review/probes/sequential-pricing"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
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


def _graph_rows(title: str) -> dict[str, str]:
    """Each arrow row of the title cell, keyed by its first column."""
    rows = {}
    for line in title.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith(("`W`", "`L_", "`A_", "`Y`", "no ")):
            rows[cells[0]] = line
    return rows


def _check_graph(namespace: dict[str, Any]) -> None:
    """The causal graph of the title cell names the arrows that the law and the data carry."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    title = "".join(next(c for c in notebook["cells"] if c["id"] == "title")["source"])
    fence = title.split("```text")[1].split("```")[0]
    for node in ("W", "L_t", "A_t", "Y", "R", "U"):
        assert node in fence, node
    assert fence.count("-->") >= 8
    # Revenue is the renewal times the month-3 price, so the month-3 price is a direct parent of R.
    assert re.search(r"^\s*Y\s+-->\s+R\s", fence, re.MULTILINE)
    assert re.search(r"^\s*A_3\s+-->\s+R\s", fence, re.MULTILINE)
    rows = _graph_rows(title)
    assert list(rows) == [
        "`W` to `L_t`, `A_t`, and `Y`",
        "`L_t` to `A_t`",
        "`A_{t-1}` and `L_{t-1}` to `A_t`",
        "`A_{t-1}` and `L_{t-1}` to `L_t`",
        "`A_3`, and `A_1` and `A_2`, to `Y`",
        "`L_3`, and `L_1` and `L_2`, to `Y`",
        "`Y` and `A_3` to `R`",
        "no `U` with arrows into `A_t` and `Y`",
        "no arrow between customers",
    ]
    law = pricing_law._LAW
    centre = law["centre_engaged"]
    # Every number the table writes is a coefficient of the law, or a product of two.
    expected = {
        "`W` to `L_t`, `A_t`, and `Y`": (
            (law["engaged_usage"], "+0.6"),
            (law["price_usage"], "+0.3"),
            (law["renew_usage"], "+0.3"),
        ),
        "`L_t` to `A_t`": ((centre, "`2`"),),
        "`A_{t-1}` and `L_{t-1}` to `A_t`": (
            (law["price_gap"], "`0.3`"),
            (-law["price_gap"] * centre, "`-0.6`"),
        ),
        "`A_{t-1}` and `L_{t-1}` to `L_t`": (
            (law["engaged_gap"], "`-0.35`"),
            (law["engaged_lag"] - law["engaged_gap"] * centre, "`1.5`"),
            (law["engaged_lag"], "`0.8`"),
            (-law["engaged_gap"] * centre, "`0.7`"),
        ),
        "`A_3`, and `A_1` and `A_2`, to `Y`": (
            (law["renew_gap"], "`-0.45`"),
            (law["renew_lag"], "`0.1`"),
        ),
        "`L_3`, and `L_1` and `L_2`, to `Y`": (
            (law["renew_engaged"] - law["renew_gap"] * centre, "`1.8`"),
            (law["renew_engaged"], "`0.9`"),
            (-law["renew_gap"] * centre, "`0.9`"),
            (-law["renew_lag"] * centre, "`0.2`"),
        ),
    }
    for name, claims in expected.items():
        for value, written in claims:
            assert written.strip("`").lstrip("+") in rows[name], (name, written)
            assert abs(value) == pytest.approx(abs(float(written.strip("`+"))), abs=1e-12), (
                name,
                written,
            )
    # The arrow A_3 to R: among renewers the revenue is the month-3 price, which varies.
    assert "`revenue = renewed * price_3`" in rows["`Y` and `A_3` to `R`"]
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
    assert printed.rstrip().endswith(f"causal study protocol: schema 1; {FINGERPRINT}")
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
    assert "cross-fitting: none -- nuisances fitted in sample" in summary
    assert "g_bounds: fixed [0.01, 1] on each cumulative treatment-and-censoring probability" in (
        summary
    )
    assert "max truncated share 0.0% at t=1" in summary
    # The summary prints the weight tail at one decimal. The stored output fixes the printed text,
    # and the fresh summary agrees with it within the rounding of that decimal.
    printed_tail = stored_output(NOTEBOOK, "ltmle")
    for plan, shown in (("price_up", 13.7), ("price_down", 31.2)):
        head = f"{plan}: 10000 of 10000 units followed it throughout; max weight "
        assert f"{head}{shown}," in printed_tail
        fresh = re.search(re.escape(head) + r"(\d+\.\d)", summary)
        assert fresh is not None and float(fresh.group(1)) == pytest.approx(shown, abs=0.11)
    assert "critical value 2.221 vs 1.960" in summary and "[experimental]" in summary

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
    printed = stored_output(NOTEBOOK, "ltmle")
    for plan in PLANS:
        assert f"{_key(plan)}: supported. Row G39.ltmle.tau3 covers this configuration." in printed


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
    # "The maximum weight compounds from 4.646 at month 1 to 13.093 at month 2 and 31.240 at month 3."
    down = report[report["regimen"] == "price_down"].set_index("time")["max_weight"]
    assert down.tolist() == pytest.approx([4.646, 13.093, 31.240], rel=WEIGHT_TAIL_REL)
    assert down.is_monotonic_increasing
    assert (report["share_truncated"] == 0.0).all()
    ratio = report["effective_n"] / report["n_followed"]
    assert namespace["kish_ratio"] == pytest.approx(0.1554, rel=WEIGHT_TAIL_REL)
    assert ratio.min() == pytest.approx(namespace["kish_ratio"])
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
    # "The fit refuses a static price on a continuous node, because no regular estimator of its mean
    # exists."
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


def _prose(cell_id: str) -> str:
    """The source of one markdown cell, with each run of whitespace collapsed to a space."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])
    return " ".join(source.split())


def _check_prose_phrases(namespace: dict[str, Any]) -> None:
    """The readings attach each repeated-draw count and decimal to the quantity the log gives it.

    A decimal that a log prints passes the narration test wherever it stands.  This check builds
    each phrase of the readings from the parsed logs, so a count or a value moved to another
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
        f"covered on {hits['renewal', 'price_up']} and {hits['renewal', 'price_down']} of 200 draws"
    ) in renewal
    assert error["renewal", "price_up"] == error["renewal", "price_down"]
    assert (
        f"was {bias['renewal', 'price_up']} and {bias['renewal', 'price_down'].lstrip('+')}, each "
        f"with a Monte Carlo standard error of {error['renewal', 'price_up']}"
    ) in renewal

    revenue = _prose("revenue-reading")
    zero = grab(r"revenue price_down interval contains zero: (\d+) of 200", sweep)
    assert f"contained zero on {zero} of 200 draws" in revenue
    up_bias = bias["revenue", "price_up"].lstrip("+")
    down_bias = bias["revenue", "price_down"].lstrip("+")
    assert (
        f"biases of {up_bias} and {down_bias}, each about two and a half Monte Carlo standard errors "
        f"({error['revenue', 'price_up']} and {error['revenue', 'price_down']})"
    ) in revenue
    for plan, value in (("price_up", up_bias), ("price_down", down_bias)):
        assert 2.2 < float(value) / float(error["revenue", plan]) < 2.8
    assert f"are {spread['price_up']} and {spread['price_down']}, so each bias is" in revenue
    assert 5 < float(spread["price_up"]) / float(up_bias) < 7
    assert 5 < float(spread["price_down"]) / float(down_bias) < 7
    assert (
        f"covered on {hits['revenue', 'price_up']} and {hits['revenue', 'price_down']} of 200 draws"
    ) in revenue
    warned = grab(r"SparseInfluenceWarning on (\d+) of 200", _line(sweep, "revenue_warnings:"))
    assert f"on {warned} of 200 revenue fits" in revenue
    quadratic = grab(r"price_up    quadratic outcome learner: bias \+(\d\.\d{4})", learner)
    linear = grab(r"price_up    linear    outcome learner: bias \+(\d\.\d{4})", learner)
    learner_error = grab(
        r"price_up    linear    outcome learner: bias .* \(MC SE (\d\.\d{4})", learner
    )
    quadratic_error = grab(
        r"price_up    quadratic outcome learner: bias .* \(MC SE (\d\.\d{4})", learner
    )
    assert (
        f"was {quadratic} with the quadratic learner and {linear} with the linear learner, with "
        f"Monte Carlo standard errors of {quadratic_error} and {learner_error}"
    ) in revenue
    # The paired line of the log, attributed by plan: the learner change moves the estimate by
    # about the size of the bias, with a paired standard error far below the shift.
    shift, shift_error = {}, {}
    for plan in PLANS:
        paired = _line(learner, plan, "linear minus quadratic")
        shift[plan] = grab(r"mean \+(\d\.\d{4})", paired)
        shift_error[plan] = grab(r"paired MC SE (\d\.\d{4})", paired)
    assert (
        f"moved the `price_up` estimate by {shift['price_up']} on average, with a paired Monte "
        f"Carlo standard error of {shift_error['price_up']}"
    ) in revenue
    assert (
        f"For `price_down` the move was {shift['price_down']}, with a paired Monte Carlo "
        f"standard error of {shift_error['price_down']}"
    ) in revenue
    for plan in PLANS:
        assert float(shift[plan]) > 5 * float(shift_error[plan]), plan
    bias_linear = {
        plan: float(grab(rf"{plan}\s+linear\s+outcome learner: bias \+(\d\.\d{{4}})", learner))
        for plan in PLANS
    }
    # "most of the `price_up` bias and about half of the `price_down` bias"
    assert float(shift["price_up"]) / bias_linear["price_up"] > 0.7
    assert 0.4 < float(shift["price_down"]) / bias_linear["price_down"] < 0.6
    assert "carries most of the `price_up` bias and about half of the `price_down` bias" in revenue
    assert "does not show that the linear learner carries the bias" not in revenue
    # With the quadratic learner the price_up bias is within one Monte Carlo standard error of zero.
    assert float(quadratic) < float(learner_error)
    assert (
        "With the quadratic learner, 60 draws cannot tell the `price_up` bias from zero"
    ) in revenue

    # The nominal count is 190 of 200.  A count near it has a Monte Carlo standard error of about 3.
    count_error = float(np.sqrt(200 * 0.95 * 0.05))
    assert round(count_error) == 3 and "is about 3 draws" in renewal
    for plan in PLANS:
        z = (190 - int(hits["renewal", plan])) / count_error
        assert 0.9 < z < 1.5, plan
    assert "both counts lie about one standard error below it" in renewal
    z_low = (190 - int(hits["revenue", "price_up"])) / count_error
    assert 1 < z_low < 2 and int(hits["revenue", "price_down"]) > 190
    assert (
        f"The count {hits['revenue', 'price_up']} lies between one and two standard errors "
        f"below the nominal 190, and {hits['revenue', 'price_down']} lies above it"
    ) in revenue
    renewal_warned = grab(
        r"SparseInfluenceWarning on (\d+) of 200", _line(sweep, "renewal_warnings:")
    )
    assert (
        f"raised `SparseInfluenceWarning` on {renewal_warned} of 200 renewal fits. This draw "
        "raised none"
    ) in renewal

    failure = _prose("failure-mode-reading")
    up_text = sweep.split("  price_up: truth")[1].split("  price_down: truth")[0]
    down_text = sweep.split("  price_down: truth")[1].split("Node-3 support")[0]

    def count(text: str, label: str) -> str:
        return grab(rf"{re.escape(label)}: (\d+) of 200", text)

    far = "adjust further from the truth than LTMLE"
    out = "adjust outside the LTMLE interval"
    between = "adjust between LTMLE and omit"
    assert (
        f"than LTMLE on {count(up_text, far)} of 200 draws for `price_up` and on "
        f"{count(down_text, far)} for `price_down`"
    ) in failure
    assert (
        f"interval on {count(up_text, out)} draws for `price_up` and on {count(down_text, out)} "
        "for `price_down`"
    ) in failure
    assert count(up_text, "omit outside the LTMLE interval") == "200"
    assert count(down_text, "omit outside the LTMLE interval") == "200"
    assert (f"on {count(up_text, between)} and {count(down_text, between)} of 200 draws") in failure
    omit_up = grab(r"omit price_up: limit (-\d\.\d{4})", sweep)
    omit_down = grab(r"omit price_down: limit (\d\.\d{4})", sweep)
    assert f"{omit_up} for `price_up` and {omit_down} for `price_down`" in failure

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
        f"averaged {float(mean_up):.2f}, with a standard deviation of {float(sd_up):.2f}. For "
        f"`price_down` it averaged {float(mean_down):.2f}, with a standard deviation of "
        f"{float(sd_down):.2f}."
    ) in assessment
    assert (
        f"averaged {float(ess_up):.3f} for `price_up` and {float(ess_down):.3f} for `price_down`"
    ) in assessment

    _check_reading_attribution(namespace)
    _check_phrase_rules()


def _interval(estimate: Any) -> str:
    low, high = estimate.ci
    return f"({low:.4f}, {high:.4f})"


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

    # Step 2: the true values, attributed by plan.
    data = _prose("data-reading")
    assert f"The frame has {len(frame)} rows and {frame.shape[1]} columns" in data
    level = {
        plan: (truth[f"ey_regimen[{plan}]"], truth[f"ey_revenue[{plan}]"])
        for plan in ("natural_prices", "price_up", "price_down")
    }
    assert (
        f"Under the engine's own prices, the renewal share is {level['natural_prices'][0]:.4f} and "
        f"the mean revenue is {level['natural_prices'][1]:.4f}. "
        f"Under `price_up` they are {level['price_up'][0]:.4f} and {level['price_up'][1]:.4f}. "
        f"Under `price_down` they are {level['price_down'][0]:.4f} and {level['price_down'][1]:.4f}."
    ) in data
    assert (
        f"The true renewal contrast is {t_up:.4f} for `price_up` and {t_down:.4f} for `price_down`. "
        f"The true revenue contrast is {rt_up:.4f} for `price_up` and {rt_down:.4f} for `price_down`."
    ) in data

    # Step 3: the three associations.
    association = _prose("association-reading")
    gap_1 = frame["price_1"] - (6 + 2 * frame["engaged_1"])
    moved = frame.groupby(np.where(gap_1 > 0, "above the centre", "at or below"))[
        "engaged_2"
    ].mean()
    assert (
        f"after a month-1 price above its centre, a share of {moved['above the centre']:.3f} of "
        f"the subscribers are engaged in month 2. After a price at or below the centre the share "
        f"is {moved['at or below']:.3f}"
    ) in association
    month_1 = frame.groupby("engaged_1")["price_1"].mean()
    assert (
        f"the mean month-1 price is {month_1[0.0]:.3f} for subscribers who are not engaged and "
        f"{month_1[1.0]:.3f} for engaged subscribers"
    ) in association
    renewed = frame.groupby("engaged_3")["renewed"].mean()
    assert (
        f"the renewal share is {renewed[0.0]:.3f} for subscribers who are not engaged in month 3 "
        f"and {renewed[1.0]:.3f} for engaged subscribers"
    ) in association

    # Step 6: the renewal contrasts, and the truth that each interval contains on this draw.
    renewal = _prose("ltmle-reading")
    assert (
        f"The `price_up` contrast is {up.psi:.4f}, with a standard error of {up.std_error:.4f} and "
        f"a 95% interval of {_interval(up)}"
    ) in renewal
    assert covers(up, t_up) and f"The interval contains the true contrast of {t_up:.4f}" in renewal
    assert (
        f"The `price_down` contrast is {down.psi:.4f}, with an interval of {_interval(down)}"
    ) in renewal
    assert covers(down, t_down) and f"It contains the true {t_down:.4f}" in renewal
    assert "Both intervals contain their true values on this draw" in renewal
    # `supported` is explained once, in the reading of the first fit that prints it.
    explanation = "a registered row covers this configuration"
    assert explanation in renewal
    assert sum(_prose(cell_id).count(explanation) for cell_id in _MARKDOWN_IDS) == 1

    # Step 7: the revenue contrasts.
    revenue = _prose("revenue-reading")
    assert (
        f"The `price_up` revenue contrast is {r_up.psi:.4f}, with an interval of "
        f"{_interval(r_up)}. The true contrast is {rt_up:.4f}."
    ) in revenue
    assert covers(r_up, rt_up) and r_up.ci[1] < 0
    assert (
        f"The `price_down` revenue contrast is {r_down.psi:.4f}, with an interval of "
        f"{_interval(r_down)}. The true contrast is {rt_down:.4f}."
    ) in revenue
    assert covers(r_down, rt_down) and r_down.ci[0] < 0 < r_down.ci[1]
    assert "The interval contains zero on this draw" in revenue

    # Step 8: the regressions, the limits, and the interval relation of each plan.
    failure = _prose("failure-mode-reading")
    one_regression = namespace["one_regression"]
    adjust = {plan: one_regression(namespace["adjust_columns"], plan) for plan in PLANS}
    omit = {plan: one_regression(namespace["omit_columns"], plan) for plan in PLANS}
    paths = namespace["paths"]
    assert not covers(up, adjust["price_up"])
    assert (
        f"For `price_up`, the truth is {t_up:.4f}, and LTMLE gives {up.psi:.4f}. The `adjust` "
        f"regression gives {adjust['price_up']:.4f}, which lies outside the LTMLE interval "
        f"{_interval(up)} on this draw. The `omit` regression gives {omit['price_up']:.4f}."
    ) in failure
    assert covers(down, adjust["price_down"])
    assert (
        f"For `price_down`, the truth is {t_down:.4f}, and LTMLE gives {down.psi:.4f} with an "
        f"interval of {_interval(down)}. The `adjust` regression gives {adjust['price_down']:.4f}, "
        "and it lies inside that interval on this draw. Its exact limit is "
        f"{paths['price_down']['both held']:.4f}, which is further from the truth than the LTMLE "
        f"estimate is. The `omit` regression gives {omit['price_down']:.4f}."
    ) in failure
    held_up = paths["price_up"]
    assert (
        f"estimates the `both held` value, {held_up['both held']:.4f} for `price_up`, and not the "
        f"truth. Each path supplies about half of the gap. Holding engagement at its natural "
        f"values gives {held_up['engagement held']:.4f}. Holding the engine's next price gives "
        f"{held_up['engine held']:.4f}."
    ) in failure

    # Step 9: the ladder, row by row.
    support = _prose("support-reading")
    rise_fit = namespace["rise_fit"]
    held = {
        plan: float((frame["price_3"] >= (8 + 2 * frame["engaged_3"]) / factor).mean())
        for plan, factor in {"up_6": 1.06, "price_up": 1.12, "up_25": 1.25}.items()
    }
    zero = {
        plan: float((frame["price_3"] < factor * (4 + 2 * frame["engaged_3"])).mean())
        for plan, factor in {"up_6": 1.06, "price_up": 1.12, "up_25": 1.25}.items()
    }
    ladder_tail = _stored_weight_tail("support")
    for plan in held:
        low, high = rise_fit[_key(plan)].ci
        max_weight, effective_n = ladder_tail[(plan, 3)]
        assert (
            f"| `{plan}` | {zero[plan]:.4f} | {held[plan]:.4f} | {max_weight} "
            f"| {effective_n} | {high - low:.4f} |"
        ) in support
    estimates = ", ".join(f"{rise_fit[_key(plan)].psi:.4f}" for plan in held)
    exact = ", ".join(f"{truth[_key(plan)]:.4f}" for plan in held)
    assert f"The estimates are {_and(estimates)}, against the true {_and(exact)}" in support
    saturation = truth[_key("up_25")] / truth[_key("up_6")]
    assert (
        f"The 25% rise moves the renewal {saturation:.2f} times as far as the 6% rise, and a "
        f"linear response would move it {25 / 6:.2f} times as far"
    ) in support
    assert "The column `share_truncated` is 0.0 in every row" in support

    # Step 10: the three plans at month 3.
    assessment = _prose("assessment-reading")
    plan_tail = _stored_weight_tail("assessment")
    for plan in ("natural_prices", "price_up", "price_down"):
        max_weight, effective_n = plan_tail[(plan, 3)]
        assert f"| `{plan}` | {max_weight} | {effective_n} |" in assessment
    assert (
        f"from {plan_tail[('price_down', 1)][0]} at month 1 to {plan_tail[('price_down', 2)][0]} "
        f"at month 2 and {plan_tail[('price_down', 3)][0]} at month 3"
    ) in assessment
    stored_ratio = re.search(
        r"minimum effective-sample-size ratio: (\d+\.\d%)", stored_output(NOTEBOOK, "assessment")
    )
    assert stored_ratio is not None
    assert f"is {stored_ratio.group(1)}, and it belongs to `price_down` at month 3" in assessment
    outcome = namespace["outcome"]
    assert (
        f"Its calibration slope is {float(outcome['calibration_slope'].iloc[0]):.3f}, and its "
        f"`auc` is {float(outcome['auc'].iloc[0]):.3f}"
    ) in assessment
    assert f"It leaves {float(namespace['held_down']):.1%} of the month-3 prices" in assessment

    # The answer: each estimate and interval, attributed.
    interpret = _prose("interpret")
    assert (
        f"the 12% rise moves the month-3 renewal share by {up.psi:.4f} against the engine's own "
        f"prices, with a 95% interval of {_interval(up)}. It moves the mean month-3 revenue by "
        f"{r_up.psi:.4f}, with an interval of {_interval(r_up)}."
    ) in interpret
    assert (
        f"the 15% cut raises the renewal share by {down.psi:.4f}, with an interval of "
        f"{_interval(down)}. Its revenue effect of {r_down.psi:.4f} has an interval of "
        f"{_interval(r_down)}, which contains zero on this draw."
    ) in interpret
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


def _and(values: str) -> str:
    """``"a, b, c"`` as ``"a, b, and c"``."""
    head, last = values.rsplit(", ", 1)
    return f"{head}, and {last}"


_MARKDOWN_IDS = (
    "title",
    "plan",
    "setup-reading",
    "data-reading",
    "association-reading",
    "protocol-reading",
    "identify-reading",
    "ltmle-reading",
    "revenue-reading",
    "failure-mode-reading",
    "support-heading",
    "support-reading",
    "assessment-reading",
    "sensitivity",
    "method-evidence",
    "interpret",
)


def _check_phrase_rules() -> None:
    """The wording rules of the plan: how the policy, the population, and the ladder are named."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = {
        cell["id"]: " ".join("".join(cell["source"]).split())
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    }
    assert set(_MARKDOWN_IDS) <= set(markdown)
    # No plan is a cap.  The refused alternative of the `plan` cell is the one place that names a
    # cap, and a floor appears only as the name `l_t` of the band's lower limit.
    for cell_id, text in markdown.items():
        found = re.findall(r"\bcap(?:s|ped)?\b", text, flags=re.IGNORECASE)
        assert len(found) == (1 if cell_id == "plan" else 0), (cell_id, found)
        assert not re.search(r"\bfloors?\b(?! \$l_t\$)", text, flags=re.IGNORECASE), cell_id
    assert "A rule that raised the price and capped it at the ceiling" in markdown["plan"]
    assert "band edge" not in " ".join(markdown.values())
    # The population rests on time zero: a three-month term, in the situation, the plan, the
    # protocol reading, and the answer.  No reading defines it by an event after time zero.
    assert "on a three-month term" in markdown["title"]
    assert "no one can leave before the term ends" in markdown["title"]
    assert "who start a three-month term at time zero" in markdown["plan"]
    assert (
        "Eligibility is a three-month term that starts at time zero" in markdown["protocol-reading"]
    )
    assert "with no exit before the month-3 renewal" in markdown["protocol-reading"]
    for text in markdown.values():
        assert "stays open" not in text and "cancellation" not in text.lower()
    # The engine reads the last gap from the last month's centre.
    assert "the gap between its last price and the last month's centre" in markdown["title"]
    # The mechanism of the ladder names the zero region and the held region.
    heading = markdown["support-heading"]
    assert (
        "The exact density ratio is zero for a subscriber whose engine price lies below" in heading
    )
    assert "The exact ratio there is one plus the ratio from the raised prices" in heading
    assert "carry the largest ratios" in heading
    assert "moved prices carry a larger density ratio" not in heading
    # The refusal reads the declared band.
    assert "that price lies outside the declared band" in markdown["plan"]
    assert "no subscriber is observed at that price" not in markdown["plan"]
    # The sensitivity step gives the refusal of an empty baseline as its reason.
    assert "refuses an empty baseline" in markdown["sensitivity"]
    assert "../roadmap.md#x63-missing-data-intermediates-and-designs" in markdown["sensitivity"]
    assert "Dropping either one changes the question" not in markdown["sensitivity"]
    # The association heading and reading claim an association only.
    assert "Each table agrees with one arrow of the graph" in markdown["association-reading"]
    assert "Engagement is therefore" not in markdown["association-reading"]
