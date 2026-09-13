"""The reviewed semantic callback for ``docs/examples/longitudinal-survival``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def check(namespace: dict[str, Any]) -> None:
    """The survival view, delta-method differences, and competing-risk narrative hold."""
    result = namespace["exit_result"]
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

    # "both risk differences are negative, and the 60-day difference is larger" (ratio 1.37).
    exit_differences = namespace["exit_differences"]
    exit_t1, exit_t2 = exit_differences[1], exit_differences[2]
    assert exit_t1.psi < -0.05
    assert exit_t2.psi < 1.15 * exit_t1.psi
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
    # "negative at both horizons and larger at 60 days" (measured ratio 1.55).
    death_t1, death_t2 = events["death", 1], events["death", 2]
    assert death_t1.psi < -0.03
    assert death_t2.psi < 1.25 * death_t1.psi

    # Death coded as censoring gives a larger reduction than the total effect, and it raises
    # never-plan readmission risk more than always-plan risk.
    eliminated = namespace["eliminated"]
    assert eliminated.data.censoring_names == ("alive_p1", "alive_p2")
    assert namespace["eliminated_t2"].psi < readmission_t2.psi - 0.01
    levels = namespace["event_levels"]
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
    event_support = namespace["event_assessment"].report("support").to_frame()
    assert {"cause", "horizon"} <= set(event_support.columns)

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
        curve = np.sum(np.column_stack([levels[name].influence_curve for name in names]), axis=1)
        expected = float(np.sqrt(np.var(curve, ddof=1) / levels.data.n))
        old_extra_scaling = expected / np.sqrt(levels.data.n)
        assert row.std_err == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert row.std_err != pytest.approx(old_extra_scaling, rel=1e-6, abs=0.0)
