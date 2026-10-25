"""The reviewed semantic callback for ``docs/examples/schedules-and-intensity``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  The slow tier also runs the refutation and sensitivity cells and
calls :func:`check_refits`.  Each assertion is a claim the prose makes about this draw or about a
committed probe log, not a statistical guarantee.
"""

from __future__ import annotations

import re
from typing import Any

from cleverly.datasets import contact_schedules_truth, navigation_doses_truth
from tests.documents import ROOT
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "schedules-and-intensity.ipynb"
REFIT_CELLS = ("refutation", "sensitivity")
_PROBE = "reviews/notebook-review/probes/schedules-and-intensity"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "1.03": f"mean SE over the SD, medium and high doubled, 400 draws: {_PROBE}/sweep.log",
    "0.99": f"mean SE over the SD, high doubled, 400 draws: {_PROBE}/sweep.log",
    "1.04": f"mean SE over the SD, more navigator time, 400 draws: {_PROBE}/sweep.log",
    "1.02": f"mean SE over the SD, more pharmacist time, 400 draws: {_PROBE}/sweep.log",
    "0.98": f"mean SE over the SD, both raises, 400 draws: {_PROBE}/sweep.log",
    "7.5": "the both-raises estimate +0.0747 in percentage points",
    "6.2": "the lower limit +0.0625 of the both-raises interval in percentage points",
    "4.6": "the printed movement -4.6 standard errors, read as a size without its sign",
}

_STRATEGY_FIELDS = {
    "time_zero",
    "treatment_strategies",
    "treatment_versions",
    "outcome",
    "intercurrent_event_handling",
    "assumption_rationale",
}


def _assert_sweep_log() -> None:
    """Each repeated-draw number of the readings matches the committed sweep log."""
    log = (ROOT / _PROBE / "sweep.log").read_text(encoding="utf-8")
    assert log.startswith("n = 3000; draws = 400")

    def row(name: str) -> str:
        match = re.search(rf"^{name}: .*$", log, flags=re.MULTILINE)
        assert match, name
        return match.group(0)

    expected = {
        "medium and high doubled": ("SE/SD 1.03", "covered 377 of 400", "supported 400 of 400"),
        "high doubled": ("SE/SD 0.99", "covered 378 of 400", "supported 400 of 400"),
        "more navigator time": ("SE/SD 1.04", "covered 384 of 400"),
        "more pharmacist time": ("SE/SD 1.02", "covered 379 of 400"),
        "both": ("SE/SD 0.98", "covered 372 of 400"),
        "together minus added up": ("SE/SD 0.97", "covered 379 of 400"),
    }
    for name, parts in expected.items():
        line = row(name)
        assert all(part in line for part in parts), (name, line)
    assert "below the lower limit of both's interval: 394 of 400" in log


def _check_schedules(namespace: dict[str, Any]) -> None:
    frame, truth, tilts = namespace["frame"], namespace["truth"], namespace["TILTS"]
    assert frame.shape == (3000, 5)
    assert truth == contact_schedules_truth(tilts)
    observed = namespace["observed"]
    # Risk rises with the schedule, so the raw averages are flatter than the true ones.
    risk = observed["discharge_risk"].tolist()
    assert risk == sorted(risk)
    raw, true = observed["average_score"], observed["true_average"]
    assert raw.max() - raw.min() < true.max() - true.min()
    # "about a third and a sixth of the true changes"
    ratios = [
        namespace["simple"][name] / truth[f"ate_ipsi[{name} vs current practice]"]
        for name in list(tilts)[1:]
    ]
    assert 0.28 < ratios[0] < 0.4 and 0.13 < ratios[1] < 0.2
    # "about an eighth of the true 0.17 gap between all-high and all-low"
    gap = truth["ey[high]"] - truth["ey[low]"]
    assert round(gap, 2) == 0.17
    high = namespace["tilt_result"]["ate_ipsi[high doubled vs current practice]"].psi
    assert 0.1 < high / gap < 0.15


def _check_protocols(namespace: dict[str, Any]) -> None:
    program = namespace["program"]
    assert changed_fields(namespace["protocol"], program) == _STRATEGY_FIELDS
    assert changed_fields(namespace["dose_protocol"], program) == _STRATEGY_FIELDS
    assert_protocol_recorded(NOTEBOOK, "protocol", namespace["protocol"], namespace["tilt_result"])
    assert_protocol_recorded(
        NOTEBOOK, "dose-protocol", namespace["dose_protocol"], namespace["dose_result"]
    )


def _check_tilt_fit(namespace: dict[str, Any]) -> None:
    truth, result = namespace["truth"], namespace["tilt_result"]
    for name in list(namespace["TILTS"])[1:]:
        key = f"ate_ipsi[{name} vs current practice]"
        point = result[key]
        assert covers(point, truth[key]), key
        assert point.support.label == "supported", key
        assert point.support.row.startswith("T35"), point.support.row
    summary = stored_output(NOTEBOOK, "estimate")
    assert "cleverly marks them experimental" in summary
    assessment = namespace["tilt_assessment"]
    assert tuple(assessment.attention) == ()
    for name, row in assessment.report("support").items():
        if name == "current practice":
            continue
        assert tuple(row.guaranteed) == (0.5, 2.0)
        assert 2_700 < row.effective_sample_size < 2_800
    aucs = [
        model.metrics["auc"]
        for model in assessment.report("nuisance_models").models
        if model.name.startswith("propensity[")
    ]
    assert len(aucs) == 3 and all(0.6 < auc < 0.73 for auc in aucs)


def _check_dose_fit(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["dose_frame"], namespace["dose_truth"]
    assert truth == navigation_doses_truth()
    assert round(frame["navigator_hours"].mean(), 1) == 4.0
    assert round(frame["pharmacist_minutes"].mean(), 1) == 40.0
    both = namespace["BOTH"]
    # "one fifteenth of the true +0.0731"
    # "one fifteenth of the true change"
    assert 0.055 < namespace["simple_both"] / truth[both] < 0.08
    # Adjusting brings the regression close to the added-up raises, still short of the truth.
    added = truth[namespace["HOURS"]] + truth[namespace["MINUTES"]]
    assert added < namespace["adjusted_both"] < truth[both]
    result = namespace["dose_result"]
    for key in (namespace["HOURS"], namespace["MINUTES"], both):
        point = result[key]
        assert covers(point, truth[key]), key
        assert (point.support.label, point.support.row) == ("supported", "T33.classifier")
    assert tuple(namespace["dose_assessment"].attention) == ()
    support = result.diagnostics.support()
    hours, minutes, joint = (support[plan.name] for plan in namespace["plans"][1:])
    for single in (hours, minutes):
        assert round(single.moved_fraction, 3) == 0.984
        assert round(single.moved_fraction + single.capped_fraction, 6) == 1.0
    assert joint.moved_fraction == 1.0
    assert all(abs(row.mean_ratio - 1.0) < 0.02 for row in (hours, minutes, joint))
    assert joint.max_ratio > max(hours.max_ratio, minutes.max_ratio)
    assert joint.effective_sample_size < min(
        hours.effective_sample_size, minutes.effective_sample_size
    )
    # "That is why its interval is the widest."
    width = {
        key: result[key].ci[1] - result[key].ci[0]
        for key in (namespace["HOURS"], namespace["MINUTES"], both)
    }
    assert max(width, key=width.__getitem__) == both


def _check_failure_mode(namespace: dict[str, Any]) -> None:
    truth, result = namespace["dose_truth"], namespace["dose_result"]
    both, hours, minutes = namespace["BOTH"], namespace["HOURS"], namespace["MINUTES"]
    assert namespace["added_up"] < result[both].ci[0]
    extra = namespace["extra"]
    assert covers(extra, truth[both] - truth[hours] - truth[minutes])
    assert extra.ci[0] > 0.0
    assert extra.support.label == "experimental"
    _assert_sweep_log()


def check(namespace: dict[str, Any]) -> None:
    """Assert each claim the readings make about the stored outputs."""
    _check_schedules(namespace)
    _check_protocols(namespace)
    _check_tilt_fit(namespace)
    _check_dose_fit(namespace)
    _check_failure_mode(namespace)


def check_refits(namespace: dict[str, Any]) -> None:
    """Assert the refutation and sensitivity readings."""
    for report, fit, name, ratio_above_one in (
        (namespace["tilt_refutation"], namespace["tilt_result"], namespace["TILT"], True),
        (namespace["dose_refutation"], namespace["dose_result"], namespace["BOTH"], False),
    ):
        tests = {test.name: test for test in report.tests}
        assert list(tests) == ["placebo", "random_common_cause", "subset"]
        for test in tests.values():
            assert test.verdict == "descriptive" and test.passed is None
            assert test.original == fit[name].psi
        assert abs(tests["placebo"].statistic) < 0.5
        assert round(abs(tests["random_common_cause"].statistic), 2) <= 0.01
        assert (tests["subset"].statistic > 1.0) is ratio_above_one
    tilt = namespace["tilt_surface"].cells[-1]
    assert tilt.displacement / namespace["tilt_result"][namespace["TILT"]].std_error < -4.0
    assert tilt.estimate > 0.0  # "The estimate stays positive"
    dose = namespace["dose_surface"].cells[-1]
    assert abs(dose.displacement / namespace["dose_result"][namespace["BOTH"]].std_error) < 0.5
