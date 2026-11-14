"""Every plot and every refute, validate and sensitivity row on the P1 PR 1 compositions.

Each composition of PR 1 (the projection with a missing treatment, with ``strata=`` on both
scales, with ``repeats`` above 1 and with the full-refit bootstrap; the EP rule with a
declared sieve or final learner) is a kind of
:data:`tests.unit._capability_sweep_support.KINDS`, so the capability sweep asks every row
about it.  This module pins the other half: each row that refuses refuses with its reason
and alternative before any learner, and each plot of :mod:`cleverly.plotting` either draws
from the fit or refuses with a stated reason.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.unit._capability_sweep_support import KINDS

pytest.importorskip("matplotlib")

PROJECTION_KINDS = (
    "projection",
    "projection+missing_treatment",
    "projection+strata",
    "projection+strata+log_ratio",
    "projection+repeats",
    "projection+bootstrap",
)
EP_KINDS = ("learned_rule_ep_sieve", "learned_rule_ep_learner")

#: The rows that refuse a working-model coefficient of a conditional contrast, by name.
WORKING_MODEL_REFUSED = (
    "refute",
    "omitted_confounding",
    "robustness_value",
    "elements",
    "benchmark",
    "contour",
    "evalue",
    "missingness",
    "tipping_gamma",
    "simulated_confounding",
    "marginal_bounds",
)


def _rows(result: Any) -> dict[str, Any]:
    return {
        row.operation: row
        for facade in (result.diagnostics, result.sensitivity)
        for row in facade.capabilities
    }


@pytest.mark.parametrize("kind", PROJECTION_KINDS)
def test_refute_and_sensitivity_refuse_a_projection_coefficient_by_name(kind: str) -> None:
    rows = _rows(KINDS[kind].build())
    for operation in WORKING_MODEL_REFUSED:
        row = rows[operation]
        assert not row.available, (kind, operation)
        assert row.refusal.condition.startswith(f"{operation} is defined for a mean")
        assert "working-model coefficient of a conditional contrast" in row.refusal.condition
        assert row.refusal.alternative == "Run it on ATE() or RiskRatio() on the same study."
    for operation in ("nuisance_models", "score_equations", "support", "truncation_curve"):
        assert rows[operation].available, (kind, operation)


@pytest.mark.parametrize("kind", EP_KINDS)
def test_the_declared_ep_rule_answers_as_the_learned_rule_does(kind: str) -> None:
    declared, shipped = _rows(KINDS[kind].build()), _rows(KINDS["learned_rule"].build())
    assert {op: row.available for op, row in declared.items()} == {
        op: row.available for op, row in shipped.items()
    }
    for operation, row in declared.items():
        if not row.available and row.refusal is not None:
            assert row.refusal.condition == shipped[operation].refusal.condition


#: What each plot does with a fit of these kinds: ``"draws"``, or the error and the message
#: that names what the plot needs.  ``repeats`` and ``bootstrap`` draw on the kinds that hold
#: draws or replicates.
PLOTS: dict[str, tuple[str, str]] = {
    "forest": ("draws", ""),
    "curve": ("draws", ""),
    "overlap": ("draws", ""),
    "calibration": ("draws", ""),
    "influence": ("draws", ""),
    "balance": ("CapabilityError", "the fit reports no estimand weighted by the probability"),
    "learners": ("ValueError", "no nuisance model here was fitted by a Super Learner"),
    "repeats": ("ValueError", "this fit has one cross-fitting repeat"),
    "bootstrap": ("ValueError", "this fit holds no bootstrap replicates"),
    "sensitivity": ("TypeError", "sensitivity() takes a TruncationCurve"),
    "confounding": ("TypeError", "confounding() takes a SensitivityContour"),
}


def _expected(kind: str, plot: str) -> tuple[str, str]:
    if (kind, plot) in {("projection+repeats", "repeats"), ("projection+bootstrap", "bootstrap")}:
        return ("draws", "")
    return PLOTS[plot]


@pytest.mark.parametrize("kind", PROJECTION_KINDS + EP_KINDS)
@pytest.mark.parametrize("plot", list(PLOTS))
def test_each_plot_draws_or_names_what_it_needs(kind: str, plot: str) -> None:
    import matplotlib.pyplot as plt

    from cleverly import plotting

    result = KINDS[kind].build()
    outcome, message = _expected(kind, plot)
    if outcome == "draws":
        figure = getattr(plotting, plot)(result)
        assert figure.axes
        plt.close("all")
        return
    with pytest.raises(Exception, match=re.escape(message)) as caught:
        getattr(plotting, plot)(result)
    assert type(caught.value).__name__ == outcome


@pytest.mark.parametrize("kind", PROJECTION_KINDS + EP_KINDS)
def test_the_truncation_curve_draws(kind: str) -> None:
    import matplotlib.pyplot as plt

    from cleverly import plotting

    figure = plotting.sensitivity(KINDS[kind].build().diagnostics.truncation_curve())
    assert figure.axes
    plt.close("all")


@pytest.mark.parametrize("kind", PROJECTION_KINDS)
def test_the_effect_curve_draws_with_its_band(kind: str) -> None:
    import matplotlib.pyplot as plt

    from cleverly import plotting

    result = KINDS[kind].build()
    curve = result.effect_curve(grid={"W1": [-1.0, 0.0, 1.0]}, random_state=0)
    assert curve.band is not None
    for plot in ("curve", "calibration"):
        assert getattr(plotting, plot)(curve).axes
    plt.close("all")
    if result.data.has_strata:
        label = result.data.stratum_label(1)
        stratum = result.effect_curve(stratum=label)
        assert stratum.model.beta.shape == curve.model.beta.shape
