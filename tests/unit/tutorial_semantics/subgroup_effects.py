"""The reviewed semantic callback for ``docs/examples/subgroup-effects``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the refit cell and calls
:func:`check_refits`.  Each assertion is a claim the prose makes about this draw, not a
statistical guarantee.  The repeated-draw numbers the prose quotes come from the committed seed
sweep, whose summary :func:`_summary` reads; the notebook shows seed 100, the first seed of that
sweep, fixed before any fit was read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    figure_title,
)

NOTEBOOK = EXAMPLES / "subgroup-effects.ipynb"
_PROBE = "reviews/notebook-review/probes/subgroup-effects"

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS = ("refute",)

#: Decimals the prose writes that no output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.95": f"lowest coverage of a direct-effect coefficient at visit yes, 100 draws: {_PROBE}/summary.log",
    "0.96": f"highest coverage of a direct-effect coefficient at visit yes, 100 draws: {_PROBE}/summary.log",
}

#: Decimals a reading quotes from a code cell outside its own section.
NARRATION_SOURCES: dict[str, dict[str, str]] = {}

LEVELS = ("family", "alone", "paid carer")
_TERMS = {
    "family": "(intercept)",
    "alone": "home_support[alone]",
    "paid carer": "home_support[paid carer]",
}


def _summary() -> str:
    return (ROOT / _PROBE / "summary.log").read_text(encoding="utf-8")


def _coefficients_cover(result: Any, truth: dict[str, float], prefix: str) -> dict[str, bool]:
    """Whether each coefficient's interval covers its truth, joined on the estimand name."""
    table = result.to_frame().set_index("estimand")
    out = {}
    for name, row in table.iterrows():
        key = f"{prefix}:{name}"
        if key in truth:
            out[str(name)] = covers((row["ci_lower"], row["ci_upper"]), truth[key])
    return out


def _check_data(namespace: dict[str, Any]) -> None:
    """The 8,000 rows, the three text levels, and the truths recomputed from the row effect."""
    frame, truth = namespace["frame"], namespace["truth"]
    from cleverly.datasets import home_support_effect

    assert frame.shape == (8000, 6)
    assert set(frame["home_support"]) == set(LEVELS)
    assert frame["home_support"].dtype != float
    ages = np.linspace(50, 90, 40_001)
    for level in LEVELS:
        average = float(np.mean(home_support_effect(np.full(ages.size, level), ages)))
        assert abs(average - truth[f"group_effect[{level}]"]) < 1e-9
    assert [round(truth[f"group_effect[{level}]"], 3) for level in LEVELS] == [0.073, 0.193, 0.113]


def _check_association(namespace: dict[str, Any]) -> None:
    """Offered patients have higher risk; the simple difference understates every group."""
    frame = namespace["frame"]
    offered = frame["navigation"] == 1
    assert (
        frame.loc[offered, "discharge_risk"].mean() > frame.loc[~offered, "discharge_risk"].mean()
    )
    naive = namespace["naive"]
    gap = naive["true effect"] - naive["simple difference"]
    assert (gap > 0).all()
    assert "simple difference below the truth in every group: 1.00" in _summary()


def _check_protocol(namespace: dict[str, Any]) -> None:
    """Every fit carries the printed description, which changes three fields of the program."""
    protocol = namespace["protocol"]
    assert changed_fields(protocol, namespace["program"]) == {
        "eligibility",
        "outcome",
        "intercurrent_event_handling",
    }
    assert_protocol_recorded(
        NOTEBOOK,
        "protocol",
        protocol,
        namespace["fit"],
        namespace["per_group"],
        namespace["with_age"],
        namespace["direct"],
        namespace["strata_fit"],
        namespace["ratio_fit"],
    )


def _check_estimate(namespace: dict[str, Any]) -> None:
    """Every row is supported and covers; the band covers; the test rejects equal gains."""
    fit, truth = namespace["fit"], namespace["truth"]
    table = fit.to_frame()
    assert (table["support"] == "supported").all()
    covered = _coefficients_cover(fit, truth, "indicators")
    assert len(covered) == 3 and all(covered.values())
    test = namespace["test"]
    assert test.df == 2 and test.p_value < 0.01
    assert round(test.p_value, 4) == 0.0003
    groups = namespace["by_group_table"].set_index("group")
    effects = groups["true effect"]
    assert ((groups["95% low"] <= effects) & (effects <= groups["95% high"])).all()
    assert ((groups["band low"] <= effects) & (effects <= groups["band high"])).all()
    assert (
        (groups["band low"] < groups["95% low"]) & (groups["band high"] > groups["95% high"])
    ).all()
    error = (groups["effect"] - effects).abs()
    assert error.idxmax() == "paid carer", "the paid-carer estimate is the furthest off"
    assert namespace["group_curve"].band.support.label == "supported"
    # The intercept-free fit draws the same group effects.
    per_group = namespace["per_group"].to_frame()["psi"].to_numpy()
    assert np.allclose(per_group, groups["effect"].to_numpy(), atol=1e-8)
    assert figure_title(namespace["group_fig"])
    summary = _summary()
    assert "every group's pointwise interval covers: 0.88" in summary, "worded 'in this sample'"
    assert "band covers all three groups: 0.96" in summary
    assert "heterogeneity test p < 0.01: 0.93" in summary


def _check_by_age(namespace: dict[str, Any]) -> None:
    """The slope covers 0.003 and excludes 0; two pointwise misses, both paid carer; band holds."""
    with_age, truth = namespace["with_age"], namespace["truth"]
    covered = _coefficients_cover(with_age, truth, "additive")
    assert len(covered) == 4 and all(covered.values())
    slope = namespace["slope"]
    assert slope["ci_lower"] > 0, "in this sample the interval excludes zero"
    assert (with_age.to_frame()["support"] == "supported").all()
    grid, at_ages = namespace["grid"], namespace["at_ages"]
    from cleverly.datasets import home_support_effect

    true = home_support_effect(grid["home_support"], grid["age"])
    misses = ~((at_ages["ci_lower"] <= true) & (true <= at_ages["ci_upper"]))
    missed = [(grid["home_support"][j], grid["age"][j]) for j in np.flatnonzero(misses)]
    assert missed == [("paid carer", 75.0), ("paid carer", 90.0)]
    band = namespace["age_curve"].band
    assert np.all((band.lower <= true) & (true <= band.upper))
    assert band.support.label == "supported"
    assert figure_title(namespace["age_fig"])
    summary = _summary()
    assert "age slope interval excludes 0: 0.81 (MC SE 0.039; 81 of 100)" in summary
    assert "all nine pointwise intervals cover: 0.79 (MC SE 0.041; 79 of 100)" in summary
    assert "band covers all nine points: 0.98 (MC SE 0.014; 98 of 100)" in summary


def _check_assessment(namespace: dict[str, Any]) -> None:
    """The overlap range and the solved equations."""
    chances = namespace["chances"]
    assert 0.1 < chances[0.0] < chances[0.01] < chances[0.99] < chances[1.0] < 0.95
    assert namespace["solved"]


def _check_sensitivity(namespace: dict[str, Any]) -> None:
    """Both floors are positive; the difference survives a cause of strength 0.03."""
    fit = namespace["fit"]
    alone = "cate_projection[home_support[alone]]"
    value = fit.sensitivity.robustness_value(estimand=alone)
    assert 0 < value["rva"] < value["rv"] < 1
    bounds = fit.sensitivity.omitted_confounding(estimand=alone, cf_y=0.03, cf_d=0.03)
    psi = fit.to_frame().set_index("estimand").loc[alone, "psi"]
    assert 0 < bounds.lower < psi < bounds.upper
    assert "bound at strength 0.03 stays above 0: 0.97" in _summary()


def _check_direct(namespace: dict[str, Any]) -> None:
    """Visit no covers; visit yes misses family and paid carer; band experimental."""
    rows = pd.DataFrame(namespace["direct_rows"])
    inside = (rows["95% low"] <= rows["true effect"]) & (rows["true effect"] <= rows["95% high"])
    missed = rows.loc[~inside, ["visit held at", "group"]].values.tolist()
    assert missed == [["yes", "family"], ["yes", "paid carer"]]
    # The 0.04 offer-by-visit witness and the 0.053 / 0.013 gaps, read off the generator.
    truth = namespace["truth"]
    for level in LEVELS:
        total = truth[f"group_effect[{level}]"]
        assert round(total - truth[f"direct_effect[visit=0][{level}]"], 3) == 0.053
        assert round(total - truth[f"direct_effect[visit=1][{level}]"], 3) == 0.013
    direct = namespace["direct"]
    assert (direct.to_frame()["support"] == "supported").all()
    assert namespace["direct_curve"].band.support.label == "experimental"
    summary = _summary()
    yes = summary[summary.index("visit held at yes") :]
    for group, share in (("family", "0.95"), ("alone", "0.96"), ("carer", "0.95")):
        assert f"{group} term interval covers: {share}" in yes
    assert "every group's pointwise interval covers: 0.84 (MC SE 0.037; 84 of 100)" in yes
    no = summary[summary.index("visit held at no") : summary.index("visit held at yes")]
    assert "family term interval covers: 1.00" in no, "covered on every draw"


def _check_strata(namespace: dict[str, Any]) -> None:
    """Each group's slope covers 0.003; the same numbers as a Tensor fit after summing."""
    strata_fit = namespace["strata_fit"]
    rows = strata_fit.to_frame().set_index("estimand")
    for level in LEVELS:
        row = rows.loc[f"cate_projection[age][home_support='{level}']"]
        assert covers((row["ci_lower"], row["ci_upper"]), 0.003)
        assert row["support"] == "supported"
    from sklearn.linear_model import LogisticRegression

    from cleverly import ConditionalContrast
    from cleverly.bases import Indicators, Linear, Modifiers, Tensor

    both = Modifiers(
        ["home_support", "age"], levels={"home_support": LEVELS}, domains={"age": (50, 90)}
    )
    tensor = (
        namespace["study"]
        .identify(
            ConditionalContrast(both, model=Tensor(Indicators("home_support"), Linear(["age"])))
        )
        .estimate(
            outcome_by_arm=True,
            random_state=0,
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
        )
    )
    product = tensor.to_frame().set_index("estimand")["psi"].to_dict()
    for level in LEVELS:
        slope = product["cate_projection[age]"]
        intercept = product["cate_projection[(intercept)]"]
        if level != "family":
            term = _TERMS[level]
            slope += product[f"cate_projection[{term}:age]"]
            intercept += product[f"cate_projection[{term}]"]
        assert abs(rows.loc[f"cate_projection[age][home_support='{level}']", "psi"] - slope) < 1e-6
        assert (
            abs(
                rows.loc[f"cate_projection[(intercept)][home_support='{level}']", "psi"] - intercept
            )
            < 1e-6
        )


def _check_ratio(namespace: dict[str, Any]) -> None:
    """Every ratio covers its truth; living alone has the lowest rate without the offer."""
    ratio_fit, truth = namespace["ratio_fit"], namespace["truth"]
    assert (ratio_fit.to_frame()["support"] == "supported").all()
    covered = _coefficients_cover(ratio_fit, truth, "log_ratio")
    assert len(covered) == 3 and all(covered.values())
    on_log = namespace["on_log"]
    logs = np.log([truth[f"risk_ratio[{level}]"] for level in LEVELS])
    assert np.all((on_log["ci_lower"] <= logs) & (logs <= on_log["ci_upper"]))
    assert min(LEVELS, key=lambda level: truth[f"untreated_rate[{level}]"]) == "alone"
    assert namespace["ratio_test"].p_value < 0.01
    assert "every group's ratio interval covers: 0.91" in _summary()


def check(namespace: dict[str, Any]) -> None:
    """Assert every relation the readings state about this draw."""
    _check_data(namespace)
    _check_association(namespace)
    _check_protocol(namespace)
    _check_estimate(namespace)
    _check_by_age(namespace)
    _check_assessment(namespace)
    _check_sensitivity(namespace)
    _check_direct(namespace)
    _check_strata(namespace)
    _check_ratio(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """The refutation reading: the placebo falls to near zero, the others stay put."""
    tests = namespace["refutation"].to_frame().set_index("test")
    original = float(tests["original"].iloc[0])
    placebo = tests.loc["placebo"]
    assert abs(placebo["refuted_mean"]) < 2 * placebo["refuted_sd"]
    for test in ("random_common_cause", "subset"):
        assert abs(tests.loc[test, "refuted_mean"] - original) < 0.01
    assert (tests["verdict"] == "descriptive").all()
    summary = _summary()
    assert "placebo mean within 2 SDs of 0: 1.00" in summary
    assert "random common cause within 0.01: 1.00" in summary
    assert "subset within 0.01: 0.99" in summary
