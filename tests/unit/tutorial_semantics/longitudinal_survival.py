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

_PROBE = "reviews/notebook-review/probes/longitudinal-survival-final"
_SURVIVAL = "tests/canonical/ltmle_survival"
_COMPETING = "tests/canonical/lmtp_ltmle_competing"
_POOLED = f"pooled block of {_PROBE}/summary.log (seeds 1000-1399 and 5000-5399, 800 draws)"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "0.64": f"competing-fit misses per draw of 12 intervals (0.638): {_POOLED}",
    "0.60": "expected misses per draw of twelve 95% intervals, 12 x 0.05",
    "-0.0093": f"readmission-first death-as-censoring functional, always minus never at t=2 "
    f"(-0.009290): {_PROBE}/truth.log",
    "1.03": f"retention 60-day difference, mean SE over empirical SD (1.025): {_POOLED}",
    "0.935": f"lowest coverage of the 12 competing-fit intervals: {_POOLED}",
    "0.961": f"highest coverage of the 12 competing-fit intervals: {_POOLED}",
    "0.927": f"death-as-censoring difference, coverage of the functional (742 of 800): {_POOLED}",
    "0.94": f"death-as-censoring difference, mean SE over empirical SD (0.937): {_POOLED}",
    "0.9406": f"coverage of ate_regimen[always vs never @ t=2], cleverly (0.940625): "
    f"{_SURVIVAL}/summary.csv",
    "0.9938": f"se_ratio of ate_regimen[always vs never @ t=2], cleverly: {_SURVIVAL}/summary.csv",
    "0.9276": f"coverage_ci_lower of interval_calibration/static_t2__correctly_specified: "
    f"{_SURVIVAL}/properties.csv",
    "0.9407": f"coverage_ci_upper of interval_calibration/static_t2__correctly_specified: "
    f"{_SURVIVAL}/properties.csv",
    "0.9411": f"se_ratio_ci_lower of interval_calibration/static_t2__correctly_specified: "
    f"{_SURVIVAL}/properties.csv",
    "0.9774": f"se_ratio_ci_upper of interval_calibration/static_t2__correctly_specified: "
    f"{_SURVIVAL}/properties.csv",
    "0.9519": f"coverage of ate_regimen[always vs never, death @ t=2], cleverly-competing-ltmle "
    f"(0.951875): {_COMPETING}/summary.csv",
    "0.9833": f"se_ratio of ate_regimen[always vs never, death @ t=2], cleverly-competing-ltmle: "
    f"{_COMPETING}/summary.csv",
    "-0.0026": f"bias_ci_lower of double_robustness/death_static_t2__mechanism_correct: "
    f"{_COMPETING}/properties.csv",
    "0.0022": f"bias_ci_upper of double_robustness/death_static_t2__mechanism_correct: "
    f"{_COMPETING}/properties.csv",
    "0.0080": f"bias_margin of double_robustness/death_static_t2__mechanism_correct (0.008019): "
    f"{_COMPETING}/properties.csv",
    "0.9778": f"se_ratio of double_robustness/death_static_t2__mechanism_correct: "
    f"{_COMPETING}/properties.csv",
}


def _death_as_censoring_functional(a: float, *, death_first: bool) -> float:
    """The g-formula with death as a censoring node, from the generator's own equations.

    With death before readmission in each period the readmission hazard among those alive is
    ``h s / (1 - h (1 - s))``; with readmission first it is the cause-specific ``h s``.
    """
    from cleverly.datasets import longitudinal as law

    points, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = points.reshape(-1, 1, 1)
    w2 = points.reshape(1, -1, 1)
    noise = points.reshape(1, 1, -1)
    mass = weights.reshape(-1, 1, 1) * weights.reshape(1, -1, 1) * weights.reshape(1, 1, -1)
    h1, s1 = law._hazard_one(w1, w2, a), law._relapse_share_one(w1, w2, a)
    l2 = law._L2["w1"] * w1 + law._L2["a1"] * a + noise
    h2, s2 = law._hazard_two(w1, w2, l2, a, a), law._relapse_share_two(w1, l2, a, a)
    if death_first:
        first, second = h1 * s1 / (1 - h1 * (1 - s1)), h2 * s2 / (1 - h2 * (1 - s2))
    else:
        first, second = h1 * s1, h2 * s2
    return float(np.sum(mass * (first + (1 - first) * second)))


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
    # The protocol names disenrollment as censoring and the hypothetical of continued enrollment.
    disenrollment = event_protocol.intercurrent_event_handling[1]
    assert "disenrollment" in disenrollment and "stayed enrolled" in disenrollment
    assert "disenrollment" in event_protocol.assumption_rationale[0]
    # "The death-as-censoring protocol changes three fields."
    eliminated_protocol = namespace["eliminated_protocol"]
    assert changed_fields(eliminated_protocol, event_protocol) == {
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert eliminated_protocol.intercurrent_event_handling[1] == disenrollment

    # "`horizons reported: t = 1, 2` ... Each one is a time point of the fit."
    retention_output = stored_output(NOTEBOOK, "estimate-retention")
    assert "horizons reported: t = 1, 2" in retention_output
    assert {key.horizon for key in result.parameter_keys.values()} == {1, 2}
    # "The summary prints no `reference` line, because a `RegimeMean` fit reports no contrast."
    assert result.config.reference == "never" and "reference:" not in retention_output
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
    # "Each line lists the covariates first and the earlier offers last. The second node adds
    # `identified_needs` and the first offer."
    identify_output = stored_output(NOTEBOOK, "identify")
    assert "history at navigation_p1: ['age', 'baseline_readiness']" in identify_output
    assert (
        "history at navigation_p2: ['age', 'baseline_readiness', 'identified_needs', "
        "'navigation_p1']"
    ) in identify_output
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
    assert frame.loc[frame["tracked_p1"] == 0, "plan_exit_p1"].isna().all()
    assert frame.loc[frame["tracked_p1"] == 1, "plan_exit_p1"].notna().all()
    at_risk_p2 = (frame["tracked_p1"] == 1) & (frame["plan_exit_p1"] == 0)
    period_two = frame.loc[at_risk_p2]
    assert period_two.loc[period_two["tracked_p2"] == 0, "plan_exit_p2"].isna().all()
    assert period_two.loc[period_two["tracked_p2"] == 1, "plan_exit_p2"].notna().all()

    # Step 8 declares disenrollment as censoring, and the plan loses patients in both periods.
    events_frame = namespace["event_frame"]
    levels = namespace["event_levels"]
    assert levels.data.censoring_names == ("enrolled_p1", "enrolled_p2")
    assert events_frame["enrolled_p1"].eq(0).sum() == 451
    assert events_frame["enrolled_p2"].eq(0).sum() == 233
    disenrolled = events_frame["enrolled_p1"] == 0
    assert events_frame.loc[disenrolled, ["readmission_p1", "death_p1"]].isna().all().all()

    event_truth = namespace["event_truth"]
    # "60-day death risk is 21% under no navigation and 7% under navigation at both periods."
    assert event_truth["cif_regimen[never, death @ t=2]"] == pytest.approx(0.21, abs=0.006)
    assert event_truth["cif_regimen[always, death @ t=2]"] == pytest.approx(0.07, abs=0.006)
    # "Its readmission and death incidences therefore add up to the plan-exit risk of Step 2",
    # and the 60-day plan-exit difference equals the death plus the readmission difference.
    for plan in ("always", "never"):
        for horizon in (1, 2):
            total = (
                event_truth[f"cif_regimen[{plan}, relapse @ t={horizon}]"]
                + event_truth[f"cif_regimen[{plan}, death @ t={horizon}]"]
            )
            assert total == pytest.approx(
                exit_truth[f"risk_regimen[{plan} @ t={horizon}]"], abs=1e-9
            )
    assert exit_truth["ate_regimen[always vs never @ t=2]"] == pytest.approx(
        event_truth["ate_regimen[always vs never, death @ t=2]"]
        + event_truth["ate_regimen[always vs never, relapse @ t=2]"],
        abs=1e-9,
    )
    events = namespace["event_differences"]
    readmission_t1, readmission_t2 = events["readmission", 1], events["readmission", 2]
    # "negative at 30 days and positive at 60 days" on this draw; the population also crosses.
    assert readmission_t1.psi < -0.01 and readmission_t2.psi > 0.01
    assert event_truth["ate_regimen[always vs never, relapse @ t=1]"] < 0.0
    assert event_truth["ate_regimen[always vs never, relapse @ t=2]"] > 0.0
    # "negative at both horizons, and the reduction is larger at 60 days" (measured ratio 1.70).
    death_t1, death_t2 = events["death", 1], events["death", 2]
    assert death_t1.psi < -0.03
    assert death_t2.psi < 1.25 * death_t1.psi
    # "Each of the 12 intervals contains its population value on this draw. The largest
    # distance is 1.1 standard errors."
    assert len(levels.estimates) == 8 and len(events) == 4
    distances = []
    for name, estimate in levels.estimates.items():
        value = event_truth[name.replace("readmission", "relapse")]
        assert covers(estimate, value)
        distances.append((estimate.psi - value) / estimate.std_error)
    for (cause, horizon), estimate in events.items():
        source = "relapse" if cause == "readmission" else cause
        value = event_truth[f"ate_regimen[always vs never, {source} @ t={horizon}]"]
        assert covers(estimate, value)
        distances.append((estimate.psi - value) / estimate.std_error)
    assert max(abs(d) for d in distances) == pytest.approx(1.1, abs=0.05)
    # "More patients die under the never plan": the population incidence the reading relies on.
    assert event_truth["ate_regimen[always vs never, death @ t=2]"] < 0.0

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
    # "Event-free survival is one minus that sum, with the same standard error", and "Each
    # event-free interval contains its population value on this draw". One minus one cause's
    # incidence is a different number, which is the nonzero witness.
    event_free = namespace["event_free"].set_index(["regimen", "time"])
    for row in totals.itertuples(index=False):
        free = event_free.loc[(row.regimen, row.time)]
        assert free["event_free"] == pytest.approx(1.0 - row.total, abs=1e-15)
        assert free["std_err"] == row.std_err
        assert free["ci_lower"] <= free["population"] <= free["ci_upper"]
        one_cause = 1.0 - levels[f"cif_regimen[{row.regimen}, readmission @ t={row.time}]"].psi
        assert one_cause - free["event_free"] > 0.02
    assert event_free.loc[("always", 2), "population"] == pytest.approx(0.682834, abs=1e-6)

    # Step 10. The censoring indicator is K_k = C_k (1 - D_k): both components remove rows, so
    # dropping either one is a mutation that changes the risk set.
    kept_p1 = namespace["kept_p1"]
    enrolled_died = (events_frame["enrolled_p1"] == 1) & (events_frame["death_p1"] == 1)
    assert (kept_p1[enrolled_died] == 0).all() and enrolled_died.sum() == 253
    assert (kept_p1[disenrolled] == 0).all() and disenrolled.sum() > 0
    assert ((kept_p1 == 1) == ((events_frame["enrolled_p1"] == 1) & ~enrolled_died)).all()
    kept_p2 = namespace["kept_p2"]
    died_p2 = namespace["through_p1"] & (events_frame["death_p2"] == 1)
    left_p2 = namespace["through_p1"] & (events_frame["enrolled_p2"] == 0)
    assert died_p2.sum() > 0 and left_p2.sum() > 0
    assert (kept_p2[died_p2 | left_p2] == 0).all()
    recoded = namespace["death_as_censoring"]
    assert recoded.loc[kept_p1 == 0, "readmission_p1"].isna().all()
    eliminated = namespace["eliminated"]
    assert eliminated.data.censoring_names == ("enrolled_alive_p1", "enrolled_alive_p2")

    # The constant equals the generator's death-first functional; the order matters.
    functional = namespace["functional"]
    for plan, arm in (("never", 0.0), ("always", 1.0)):
        assert functional[plan] == pytest.approx(
            _death_as_censoring_functional(arm, death_first=True), abs=5e-7
        )
    readmission_first = _death_as_censoring_functional(
        1.0, death_first=False
    ) - _death_as_censoring_functional(0.0, death_first=False)
    assert round(readmission_first, 4) == -0.0093
    assert round(functional["always"] - functional["never"], 4) == -0.0403

    eliminated_t2 = namespace["eliminated_t2"]
    # "the censored analysis reports a reduction ..., and the competing-event analysis reports
    # an increase" on this draw; the censored difference is lower on every probe draw.
    assert eliminated_t2.psi < 0.0 < readmission_t2.psi
    assert eliminated_t2.psi < readmission_t2.psi - 0.01
    # "The interval of the censored difference ... contains both population values on this draw."
    assert covers(eliminated_t2, functional["always"] - functional["never"])
    assert covers(eliminated_t2, event_truth["ate_regimen[always vs never, relapse @ t=2]"])
    # "Censoring at death ... raises readmission risk most under that plan."
    raised_never = (
        eliminated["risk_regimen[never @ t=2]"].psi
        - levels["cif_regimen[never, readmission @ t=2]"].psi
    )
    raised_always = (
        eliminated["risk_regimen[always @ t=2]"].psi
        - levels["cif_regimen[always, readmission @ t=2]"].psi
    )
    assert raised_never > raised_always + 0.01 and raised_always > 0.0

    # Step 11 prints only the operations that ran.
    assessment_output = stored_output(NOTEBOOK, "assessment")
    assert "unavailable" not in assessment_output and "deferred" not in assessment_output
    ran = namespace["ran"]
    assert list(zip(ran["check"], ran["status"], strict=True)) == [
        ("score_equations", "passed"),
        ("support", "completed"),
        ("nuisance_models", "completed"),
    ]
    for assessment in (namespace["exit_assessment"], namespace["event_assessment"]):
        # "Neither fit has a row that needs attention", and no support row reached the bound.
        assert not assessment.attention
        support = assessment.report("support").to_frame()
        assert support["converged"].all() and (support["share_truncated"] == 0.0).all()
    # "The competing-event table adds a cause column"; both tables carry horizon and time.
    exit_support = namespace["exit_support"]
    event_support = namespace["event_assessment"].report("support").to_frame()
    assert {"horizon", "time"} <= set(exit_support.columns) and "cause" not in exit_support
    assert set(event_support.columns) == set(exit_support.columns) | {"cause"}
    # "The minimum effective-sample-size ratio is 77.8%. That is the never plan at the second
    # node: an effective sample of about 473 from 608 followers."
    ratios = exit_support["effective_n"] / exit_support["n_followed"]
    weakest = exit_support.loc[ratios.idxmin()]
    assert (weakest["regimen"], weakest["time"], weakest["n_followed"]) == ("never", 2, 608)
    assert ratios.min() == pytest.approx(0.778, abs=0.0005)
    assert "minimum effective-sample-size ratio: 77.8%" in assessment_output
    # "On the competing fit the largest, -0.0679, is on the never-plan readmission row at the
    # second node."
    largest = event_support.loc[event_support["epsilon"].abs().idxmax()]
    assert (largest["regimen"], largest["cause"], largest["horizon"], largest["time"]) == (
        "never",
        "readmission",
        2,
        2,
    )
    assert largest["epsilon"] == pytest.approx(-0.0679, abs=5e-5)

    # Step 12. Each refit drops one recorded covariate from the full design; every move is
    # positive, and the last refit has no time-varying history left.
    benchmark = namespace["benchmark"].set_index("dropped")
    assert list(benchmark.index) == ["age", "baseline_readiness", "identified_needs"]
    assert benchmark["move_in_se"].tolist() == [1.89, 1.35, 1.46]
    assert (benchmark["move"] > 0.0).all()
    assert ((benchmark["psi"] - benchmark["move"]) - exit_t2.psi).abs().max() < 2e-4
    assert namespace["reduced"].time_varying == ((), ())
    assert namespace["reduced"].baseline == ("age", "baseline_readiness")
    assert namespace["full_design"].time_varying == ((), ("identified_needs",))
    # "The full-history difference is about seven standard errors below zero."
    assert 6.5 < -exit_t2.psi / exit_t2.std_error < 7.5
