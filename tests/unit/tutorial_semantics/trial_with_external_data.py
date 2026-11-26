"""The reviewed semantic callback for ``docs/examples/trial-with-external-data``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`. The notebook has no refit cell. Each assertion is a claim the
prose makes about this draw, not a statistical guarantee.
"""

from __future__ import annotations

from typing import Any


from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

#: The refit cells the rule derives from this notebook today, in notebook order.
REFIT_CELLS: tuple[str, ...] = ()

NOTEBOOK = EXAMPLES / "trial-with-external-data.ipynb"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS: dict[str, str] = {
    "0.01": "the floor below which the fit warns about trial coverage, SOURCE_FLOOR in atmle.py",
}

#: Decimals a reading quotes from a code cell outside its own section, by reading cell, then
#: literal, then the code cell that prints it (tests.unit.tutorial_semantics.narration_scopes).
NARRATION_SOURCES: dict[str, dict[str, str]] = {
    "comparison-reading": {"0.014": "overlap"},
}

#: The boosted outcome model sums gradients in float32, which round with the CPU's vector
#: instructions, so its estimate can move in the last printed digit between machines.
HARDWARE_SENSITIVE_CELLS = {
    "flexible": (
        "HistGradientBoostingRegressor sums gradients in float32, which round with the CPU's "
        "vector instructions"
    ),
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
    # "the records offered navigation to about the same share of patients as the pilot did".
    records = frame[frame["in_pilot"] == 0]
    share_gap = records["transition_navigation"].mean() - pilot["transition_navigation"].mean()
    assert abs(share_gap) < 0.05


def _check_protocol(namespace: dict[str, Any]) -> None:
    protocol, program = namespace["protocol"], namespace["program"]
    # "the population, ... the eligibility, the outcome's scale, the handling of deaths, and the
    # first reason for the assumptions".
    assert changed_fields(protocol, program) == {
        "target_population",
        "eligibility",
        "outcome",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert protocol.eligibility[0] == program.eligibility[0]
    assert protocol.intercurrent_event_handling[:2] == program.intercurrent_event_handling[:2]
    assert "hypothetical strategy" in protocol.intercurrent_event_handling[2]
    assert protocol.assumption_rationale[1:] == program.assumption_rationale[1:]
    assert_protocol_recorded(
        NOTEBOOK, "protocol", protocol, namespace["borrowed"], namespace["pooled_fit"]
    )


def _check_estimate(namespace: dict[str, Any]) -> None:
    alone, borrowed, truth = namespace["alone"], namespace["borrowed"], namespace["truth"]
    target = truth["pilot patients"]
    assert covers(alone.estimate, target) and covers(borrowed.estimate, target)
    # "the standard error falls from 0.67 to 0.36".
    assert borrowed.estimate.std_error < 0.6 * alone.estimate.std_error
    assert borrowed.estimate.support.label == "experimental"
    assert borrowed.inference_status == "working_model_projection"
    printed = stored_output(NOTEBOOK, "estimate")
    assert "label on the pilot-with-records interval: experimental" in printed
    # The summary's notes [1] to [3] are the ones the reading names.
    assert "[2] The interval of ate is widened on the side" in printed
    assert "[3] On the external rows" in printed
    assert "Trial effect with external data (adaptive TMLE)" in printed


def _check_assessment(namespace: dict[str, Any]) -> None:
    report = namespace["report"]
    assert (report.n_trial, report.n_external) == (560, 2440)
    assert report.population == "trial" and report.adequate
    assert not report.controls_only and report.n_external_treated == 1203
    assert report.external_source_min > 0.01
    # "All four fit checks ran and passed ... The other twelve ... do not run".
    checks, ran = namespace["checks"], namespace["ran"]
    assert len(ran) == 4 and (ran["status"] == "passed").all()
    assert len(checks) - len(ran) == 12
    score, updates = namespace["score"], namespace["updates"]
    assert score.converged and score.influence < score.tolerance
    assert updates.converged and updates.passes == 1
    # "the formula for how the records differ kept only the constant term `1`" in every fold.
    borrowed = namespace["borrowed"]
    assert len(borrowed.working_models) == 5
    assert all(model.bias_columns == ("1",) for model in borrowed.working_models)
    printed = stored_output(NOTEBOOK, "overlap")
    assert f"lowest chance that a record row is a pilot row: {report.external_source_min:.3f}" in (
        printed
    )
    assert "initial trial-membership probabilities" in printed
    low, high = report.trial_row_treatment_range
    assert low < 0.5 < high


def _check_comparison(namespace: dict[str, Any]) -> None:
    fit = namespace["pooled_fit"]
    assert covers(fit.estimate, namespace["truth"]["all patients"])
    assert fit.estimate.psi < namespace["borrowed"].estimate.psi
    assert fit.support_report().population == "pooled"
    assert fit.support_report().external_source_below_floor == 0.0


def _check_options(namespace: dict[str, Any]) -> None:
    options, alone = namespace["options"], namespace["alone"]
    target = namespace["truth"]["pilot patients"]
    # "Every interval contains the true 6.05 in this draw".
    assert all(covers(fit.estimate, target) for fit in options.values())
    assert covers(alone.estimate, target)
    se = {label: fit.estimate.std_error for label, fit in options.items()}
    # "The forced arm is about as wide as the pilot alone ... The records of controls only are
    # about as wide too": both within a tenth of the pilot-alone standard error.
    for label in ("force_arm=True", "records of controls only"):
        assert abs(se[label] / alone.estimate.std_error - 1) < 0.1
    # "The pooled form and the squares-and-products list land close to the default".
    for label in ('form="pooled"', "squares and products"):
        assert abs(se[label] / se["default (cv form)"] - 1) < 0.1
    assert options["records of controls only"].support_report().controls_only
    roles = dict(options["missing scores"].diagnostics.nuisance_models().learners)
    assert roles["missingness"] == "logistic regression"
    # The missing scores come from their own stream: about a fifth of the pilot's scores.
    gaps = namespace["with_gaps"]
    assert gaps["transition_score"].isna().sum() == (gaps["answered"] == 0).sum() > 0
    assert namespace["frame"]["transition_score"].notna().all()
    # "it refuses the usual missing-outcome tilt checks".
    rows = options["missing scores"].assess().to_frame().set_index("check")["status"]
    assert rows["missingness"] == rows["tipping_gamma"] == "unavailable"


def _check_binary(namespace: dict[str, Any]) -> None:
    fit, truth = namespace["binary_fit"], namespace["binary_truth"]
    # A route that reads simulate() itself: the large draw's pilot randomized the offer, so the
    # difference in the share of scores above 55 between its arms estimates the same effect.
    large = namespace["large"]
    pilot = large[large["in_pilot"] == 1]
    good = pilot["transition_score"] > 55
    offered = pilot["transition_navigation"] == 1
    randomized = float(good[offered].mean() - good[~offered].mean())
    assert abs(randomized - truth) < 0.006
    assert covers(fit.estimate, truth)


def _check_flexible(namespace: dict[str, Any]) -> None:
    boosted, borrowed = namespace["boosted"], namespace["borrowed"]
    target = namespace["truth"]["pilot patients"]
    # "both contain the true 6.05", "with about the same standard error, 0.36".
    assert covers(boosted.estimate, target) and covers(borrowed.estimate, target)
    assert abs(boosted.estimate.std_error - borrowed.estimate.std_error) < 0.02


def check(namespace: dict[str, Any]) -> None:
    """Assert each claim the readings make about the stored outputs."""
    _check_data(namespace)
    _check_protocol(namespace)
    _check_estimate(namespace)
    _check_assessment(namespace)
    _check_comparison(namespace)
    _check_options(namespace)
    _check_binary(namespace)
    _check_flexible(namespace)
