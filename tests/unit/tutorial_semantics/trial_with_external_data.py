"""The reviewed semantic callback for ``docs/examples/trial-with-external-data``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`. The notebook has no refit cell. Each assertion is a claim the
prose makes about this draw, not a statistical guarantee.
"""

from __future__ import annotations

from typing import Any

from tests.unit.tutorial_semantics import EXAMPLES, covers, stored_output

#: The refit cells the rule derives from this notebook today, in notebook order.
REFIT_CELLS: tuple[str, ...] = ()

NOTEBOOK = EXAMPLES / "trial-with-external-data.ipynb"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS: dict[str, str] = {
    "0.01": "the floor below which the fit warns about trial coverage, SOURCE_FLOOR in atmle.py",
}


def _check_data(namespace: dict[str, Any]) -> None:
    frame, truth = namespace["frame"], namespace["truth"]
    assert frame.shape == (3000, 6)
    pilot = frame[frame["in_pilot"] == 1]
    assert len(pilot) == 560
    # The effect is larger among the sicker pilot than among everyone.
    assert truth["pilot patients"] > truth["all patients"] == 5.0
    # The pooled difference is off by more than the pilot's own difference.
    assert abs(namespace["pooled_gap"] - truth["pilot patients"]) > 1.0
    assert abs(namespace["pilot_gap"] - truth["pilot patients"]) < 1.0


def _check_estimate(namespace: dict[str, Any]) -> None:
    alone, borrowed, truth = namespace["alone"], namespace["borrowed"], namespace["truth"]
    target = truth["pilot patients"]
    assert covers(alone.estimate, target) and covers(borrowed.estimate, target)
    # "cut the standard error from 0.67 to 0.37": the records narrow the interval.
    assert borrowed.estimate.std_error < 0.6 * alone.estimate.std_error
    assert borrowed.estimate.support.label == "experimental"
    assert borrowed.inference_status == "working_model_projection"
    assert "label on the pilot-with-records interval: experimental" in stored_output(
        NOTEBOOK, "estimate"
    )


def _check_assessment(namespace: dict[str, Any]) -> None:
    report = namespace["report"]
    assert (report.n_trial, report.n_external) == (560, 2440)
    assert report.population == "trial" and report.adequate
    assert not report.controls_only and report.n_external_treated == 1203
    assert report.external_source_min > 0.01
    fold = namespace["fold"]
    assert len(fold.bias.selected) < len(fold.effect.selected)


def _check_comparison(namespace: dict[str, Any]) -> None:
    fit = namespace["pooled_fit"]
    assert covers(fit.estimate, namespace["truth"]["all patients"])
    assert fit.estimate.psi < namespace["borrowed"].estimate.psi
    assert fit.support_report().population == "pooled"
    assert fit.support_report().external_source_below_floor == 0.0


def check(namespace: dict[str, Any]) -> None:
    """Assert each claim the readings make about the stored outputs."""
    _check_data(namespace)
    _check_estimate(namespace)
    _check_assessment(namespace)
    _check_comparison(namespace)
