"""The reviewed semantic callback for ``docs/examples/longitudinal-survival``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "longitudinal-survival.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """The survival view, delta-method differences, and competing-risk narrative hold."""
    result = namespace["exit_result"]

    # Each protocol step prints its record, and each fit carries the digest of its own protocol.
    fingerprints = {
        assert_protocol_recorded(NOTEBOOK, "protocol", namespace["exit_protocol"], result),
        assert_protocol_recorded(
            NOTEBOOK, "competing-events", namespace["event_protocol"], namespace["event_levels"]
        ),
        assert_protocol_recorded(NOTEBOOK, "failure-mode", namespace["event_protocol"]),
        assert_protocol_recorded(
            NOTEBOOK, "failure-mode", namespace["eliminated_protocol"], namespace["eliminated"]
        ),
    }
    assert len(fingerprints) == 3
    # "The code changes seven fields of the program protocol and keeps three."
    exit_protocol, event_protocol = namespace["exit_protocol"], namespace["event_protocol"]
    assert changed_fields(exit_protocol, namespace["program"]) == {
        "eligibility",
        "treatment_strategies",
        "treatment_versions",
        "outcome",
        "horizon",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    # "Four fields changed: the treatment strategies, the outcome, ... the assumption rationale."
    assert changed_fields(event_protocol, exit_protocol) == {
        "treatment_strategies",
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    # "The death-as-censoring protocol changes three fields."
    assert changed_fields(namespace["eliminated_protocol"], event_protocol) == {
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    # "The fit refuses a horizon outside 1..2 rather than interpolating it."
    retention_output = stored_output(NOTEBOOK, "estimate-retention")
    assert "outside 1..2" in retention_output
    # "reference: never ... A RegimeMean fit reports no contrast, so this line changes no number."
    assert result.config.reference == "never" and "reference: never" in retention_output
    assert all(name.startswith("risk_regimen[") for name in result.estimates)
    from cleverly import RegimeMean

    default_reference = (
        namespace["exit_study"]
        .identify(RegimeMean(namespace["plans"], horizons=(1, 2)))
        .estimate(method=namespace["sequential"])
    )
    # The witness is nonzero: the library default names the other plan.
    assert default_reference.config.reference == "always"
    assert set(default_reference.estimates) == set(result.estimates)
    for name, estimate in result.estimates.items():
        assert default_reference[name].psi == pytest.approx(estimate.psi, abs=1e-12)
        assert default_reference[name].std_error == pytest.approx(estimate.std_error, abs=1e-12)
    # "The adjustment/history line lists the baseline covariates only. The design still adds
    # identified_needs to the history at the second node."
    assert "adjustment/history: ['age', 'baseline_readiness']" in stored_output(
        NOTEBOOK, "identify"
    )
    assert "identified_needs" not in result.data.history_names(1)
    assert result.data.history_names(2)[-1] == "identified_needs"
    # "The simultaneous bands are built to hold all four parameters at once ... wider than the
    # pointwise intervals."
    bands = result.simultaneous
    assert bands is not None and len(bands.bands) == 4
    assert bands.critical_value > bands.pointwise_critical_value
    for name, (low, high) in bands.bands.items():
        assert low < result[name].ci[0] and high > result[name].ci[1]

    # "offered patients are older ... less ready", and the day-31 offer goes to more needs.
    frame = namespace["exit_frame"]
    by_offer = frame.groupby("navigation_p1")[["age", "baseline_readiness"]].mean()
    assert by_offer.loc[1.0, "age"] > by_offer.loc[0.0, "age"]
    assert by_offer.loc[1.0, "baseline_readiness"] < by_offer.loc[0.0, "baseline_readiness"]
    at_risk = frame.loc[frame["plan_exit_p1"] == 0]
    needs = at_risk.groupby("navigation_p2")["identified_needs"].mean()
    assert needs.loc[1.0] > needs.loc[0.0] + 0.5
    # "the naive comparison understates the benefit on this draw", at both horizons.
    for horizon, naive in namespace["naive_differences"].items():
        population = namespace["exit_truth"][f"ate_regimen[always vs never @ t={horizon}]"]
        assert population < naive < 0.0
    risk = result.curve(scale="risk")
    survival = namespace["survival_curve"]
    keys = set(result.estimates)
    assert len(risk) == len(survival) == 4

    # ``estimand`` names the survival quantity and ``parameter`` names the risk behind it.
    assert set(risk["view"]) == {"risk"} and set(survival["view"]) == {"survival"}
    assert set(risk["scale"]) == {"level"} and set(survival["scale"]) == {"level"}
    assert list(risk["parameter"]) == list(survival["parameter"])
    assert set(survival["parameter"]) <= keys
    assert not set(survival["estimand"]) & keys
    assert all(str(name).startswith("survival_regimen[") for name in survival["estimand"])
    np.testing.assert_allclose(survival["psi"], 1.0 - risk["psi"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(survival["ci_lower"], 1.0 - risk["ci_upper"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(survival["ci_upper"], 1.0 - risk["ci_lower"], rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(survival["std_err"], risk["std_err"])

    # "both risk differences are negative, and the 60-day reduction is larger" (ratio 1.37).
    exit_differences = namespace["exit_differences"]
    exit_t1, exit_t2 = exit_differences[1], exit_differences[2]
    assert exit_t1.psi < -0.05
    assert exit_t2.psi < 1.15 * exit_t1.psi
    # "each interval contains its population value" on this draw.
    for horizon, estimate in exit_differences.items():
        assert covers(
            estimate, namespace["exit_truth"][f"ate_regimen[always vs never @ t={horizon}]"]
        )
    retention = survival.set_index(["regimen", "time"])
    assert retention.loc[("always", 2), "psi"] > retention.loc[("never", 2), "psi"] + 0.05
    exit_truth = namespace["exit_truth"]
    assert (
        exit_truth["ate_regimen[always vs never @ t=2]"]
        < exit_truth["ate_regimen[always vs never @ t=1]"]
        < 0.0
    )
    # "26% ... within 30 days and 46% within 60 days under no navigation" are exact truths.
    assert exit_truth["risk_regimen[never @ t=1]"] == pytest.approx(0.26, abs=0.005)
    assert exit_truth["risk_regimen[never @ t=2]"] == pytest.approx(0.46, abs=0.006)

    # "the same estimate and standard error as a separate RegimeContrast fit". The refit reuses
    # the page's clustered study and method, and costs one more backward pass.
    from cleverly import RegimeContrast

    contrast = (
        namespace["exit_study"]
        .identify(RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2)))
        .estimate(method=namespace["sequential"])
    )
    assert result.data.cluster is not None
    for horizon, estimate in exit_differences.items():
        fitted = contrast[f"ate_regimen[always vs never @ t={horizon}]"]
        assert estimate.psi == pytest.approx(fitted.psi, abs=1e-10)
        assert estimate.std_error == pytest.approx(fitted.std_error, rel=1e-8)

    # The renamed censoring columns gate the event in their own period.
    frame = namespace["exit_frame"]
    assert frame.loc[frame["tracked_p1"] == 0, "plan_exit_p1"].isna().all()
    assert frame.loc[frame["tracked_p1"] == 1, "plan_exit_p1"].notna().all()
    at_risk_p2 = (frame["tracked_p1"] == 1) & (frame["plan_exit_p1"] == 0)
    period_two = frame.loc[at_risk_p2]
    assert period_two.loc[period_two["tracked_p2"] == 0, "plan_exit_p2"].isna().all()
    assert period_two.loc[period_two["tracked_p2"] == 1, "plan_exit_p2"].notna().all()

    event_truth = namespace["event_truth"]
    # "60-day death risk is 21% under no navigation and 7% under navigation at both periods."
    assert event_truth["cif_regimen[never, death @ t=2]"] == pytest.approx(0.21, abs=0.006)
    assert event_truth["cif_regimen[always, death @ t=2]"] == pytest.approx(0.07, abs=0.006)
    events = namespace["event_differences"]
    readmission_t1, readmission_t2 = events["readmission", 1], events["readmission", 2]
    # "negative at 30 days and shrinks toward zero by 60 days" (measured ratio 0.21).
    assert readmission_t1.psi < -0.015
    assert abs(readmission_t2.psi) < 0.5 * abs(readmission_t1.psi)
    assert event_truth["ate_regimen[always vs never, relapse @ t=1]"] < 0.0
    assert event_truth["ate_regimen[always vs never, relapse @ t=2]"] > 0.0
    # "negative at both horizons, and the reduction is larger at 60 days" (measured ratio 1.55).
    death_t1, death_t2 = events["death", 1], events["death", 2]
    assert death_t1.psi < -0.03
    assert death_t2.psi < 1.25 * death_t1.psi
    # "Two intervals miss their population values on this draw", out of the 12 the step prints:
    # the 60-day death difference and the never-plan death incidence at 60 days.
    levels = namespace["event_levels"]
    assert len(levels.estimates) == 8 and len(events) == 4
    level_misses = {
        name
        for name, estimate in levels.estimates.items()
        if not covers(estimate, event_truth[name.replace("readmission", "relapse")])
    }
    assert level_misses == {"cif_regimen[never, death @ t=2]"}
    difference_misses = {
        key
        for key, estimate in events.items()
        if not covers(
            estimate,
            event_truth[
                f"ate_regimen[always vs never, "
                f"{'relapse' if key[0] == 'readmission' else key[0]} @ t={key[1]}]"
            ],
        )
    }
    assert difference_misses == {("death", 2)}
    # "Both misses come from one estimate. The never-plan death incidence sits below its
    # population value, and the difference inherits that gap." The always-plan death incidence
    # covers its value, and the difference sits above its (negative) population value.
    never_death = levels["cif_regimen[never, death @ t=2]"]
    assert never_death.psi < event_truth["cif_regimen[never, death @ t=2]"]
    assert covers(
        levels["cif_regimen[always, death @ t=2]"], event_truth["cif_regimen[always, death @ t=2]"]
    )
    assert death_t2.psi > event_truth["ate_regimen[always vs never, death @ t=2]"]
    # The printed distances are 2.9 and 2.3 standard errors.
    never_distance = (never_death.psi - event_truth["cif_regimen[never, death @ t=2]"]) / (
        never_death.std_error
    )
    assert never_distance == pytest.approx(-2.9, abs=0.05)
    death_distance = (
        death_t2.psi - event_truth["ate_regimen[always vs never, death @ t=2]"]
    ) / death_t2.std_error
    assert death_distance == pytest.approx(2.3, abs=0.05)
    # "More patients die under the never plan": the population incidence the reading relies on.
    assert event_truth["ate_regimen[always vs never, death @ t=2]"] < 0.0

    # Death coded as censoring gives a larger reduction than the total effect, and it raises
    # never-plan readmission risk more than always-plan risk.
    eliminated = namespace["eliminated"]
    assert eliminated.data.censoring_names == ("alive_p1", "alive_p2")
    assert namespace["eliminated_t2"].psi < readmission_t2.psi - 0.01
    raised_never = (
        eliminated["risk_regimen[never @ t=2]"].psi
        - levels["cif_regimen[never, readmission @ t=2]"].psi
    )
    raised_always = (
        eliminated["risk_regimen[always @ t=2]"].psi
        - levels["cif_regimen[always, readmission @ t=2]"].psi
    )
    assert raised_never > raised_always + 0.01 and raised_always > 0.0

    for assessment in (namespace["exit_assessment"], namespace["event_assessment"]):
        sensitivity = assessment.sensitivity.items
        assert sensitivity and all(item.status.value == "unavailable" for item in sensitivity)
        assert assessment.diagnostics["refute"].status.value == "unavailable"
        # "Neither fit has a row that needs attention", and no support row reached the bound.
        assert not assessment.attention
        support = assessment.report("support").to_frame()
        assert support["converged"].all() and (support["share_truncated"] == 0.0).all()
    # "The competing-event table adds a cause column"; both tables carry horizon and time.
    exit_support = namespace["exit_assessment"].report("support").to_frame()
    event_support = namespace["event_assessment"].report("support").to_frame()
    assert {"horizon", "time"} <= set(exit_support.columns) and "cause" not in exit_support
    assert set(event_support.columns) == set(exit_support.columns) | {"cause"}
    # "a minimum effective-sample-size ratio of 77.8%. That is the never plan at the second node:
    # an effective sample of about 473 from 608 followers."
    ratios = exit_support["effective_n"] / exit_support["n_followed"]
    weakest = exit_support.loc[ratios.idxmin()]
    assert (weakest["regimen"], weakest["time"], weakest["n_followed"]) == ("never", 2, 608)
    assert ratios.min() == pytest.approx(0.778, abs=0.0005)
    assert "minimum effective-sample-size ratio 77.8%" in stored_output(NOTEBOOK, "assessment")
    # "The largest [epsilon], -0.1324, is on the never-plan death row at the second node."
    largest = event_support.loc[event_support["epsilon"].abs().idxmax()]
    assert (largest["regimen"], largest["cause"], largest["horizon"], largest["time"]) == (
        "never",
        "death",
        2,
        2,
    )
    assert largest["epsilon"] == pytest.approx(-0.1324, abs=5e-5)

    assert len(levels.config.causes) == 2
    with pytest.raises(ValueError, match="not all-cause survival"):
        levels.curve(scale="survival")

    # The reported total standard error is derived from the influence curves alone. This fit
    # declares no cluster, so the independent formula is the iid one.
    totals = namespace["incidence_totals"]
    index = levels.parameter_index or {}
    assert levels.data.cluster is None
    for row in totals.itertuples(index=False):
        names = [
            name
            for name, (regimen, _cause, horizon) in index.items()
            if regimen == row.regimen and horizon == row.time
        ]
        assert len(names) == len(levels.config.causes)
        # "incidence_total() sums the causes ... It does not renormalize them", and "excess ...
        # is 0.0 in every row". A renormalizing mutation would still show a zero excess, so the
        # witness is the sum itself.
        assert row.total == pytest.approx(sum(levels[name].psi for name in names), abs=1e-12)
        assert row.total < 1.0 and row.excess == 0.0
        curve = np.sum(np.column_stack([levels[name].influence_curve for name in names]), axis=1)
        expected = float(np.sqrt(np.var(curve, ddof=1) / levels.data.n))
        old_extra_scaling = expected / np.sqrt(levels.data.n)
        assert row.std_err == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert row.std_err != pytest.approx(old_extra_scaling, rel=1e-6, abs=0.0)
