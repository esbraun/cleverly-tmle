"""A longitudinal design with no baseline covariate is a named gap, on every layout.

The package fits no intercept-only nuisance design, so ``baseline=()`` is refused as
``implementation_gap``.  X63 part (c) in ``docs/roadmap.md`` tracks the fit with no covariates.
The wide frame, the long panel and the time-to-event layout share one container, so one refusal
covers all three.  A baseline with one column is the control: the same declaration fits.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly import CausalStudy, LongitudinalTreatment, LongPanel, TimeToEvent
from cleverly._refusal import NO_ALTERNATIVE
from cleverly.datasets import make_longitudinal, make_sequential_offers
from cleverly.exceptions import DataError, RefusalReason
from cleverly.longitudinal import LongitudinalData
from cleverly.utils.frames import frame_from_dict


def _wide(baseline: list[str]) -> Any:
    frame, _ = make_longitudinal(300, seed=0)
    design = LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=baseline,
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    return CausalStudy(frame, design=design)


def _panel(baseline: list[str]) -> Any:
    panel, _ = make_sequential_offers(300, seed=1)
    design = LongPanel(
        unit="customer",
        time="week",
        treatment="offer",
        baseline=baseline,
        time_varying=["engaged"],
        censoring="consent",
        outcome="full_price",
    )
    return CausalStudy(panel, design=design)


def _time_to_event(baseline: list[str]) -> Any:
    frame = frame_from_dict(
        {
            "time": [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1],
            "event": [1, 0, 1, 2, 0, 1, 1, 0, 2, 1, 0, 1],
            "arm": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "age": [0.1, 0.5, -0.3, 0.8, -1.2, 0.4, 0.9, -0.6, 0.2, -0.1, 0.7, -0.8],
        }
    )
    design = TimeToEvent(time="time", event="event", treatment="arm", baseline=baseline)
    return CausalStudy(frame, design=design)


@pytest.mark.parametrize("layout", [_wide, _panel, _time_to_event])
def test_an_empty_baseline_is_a_named_implementation_gap(layout: Any) -> None:
    with pytest.raises(DataError, match="baseline= is empty") as caught:
        layout([])
    refusal = caught.value.refusal
    assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert refusal.roadmap == "X63"
    assert refusal.alternative == NO_ALTERNATIVE
    assert "X63 in docs/roadmap.md tracks this" in str(caught.value)


@pytest.mark.parametrize(("layout", "baseline"), [(_wide, ["W1"]), (_panel, ["tenure"])])
def test_a_baseline_with_one_covariate_builds(layout: Any, baseline: list[str]) -> None:
    """The control: the refusal depends on the empty baseline and on nothing else."""
    study = layout(baseline)
    assert isinstance(study.data, LongitudinalData)
    assert study.data.baseline_names == tuple(baseline)


def test_the_roadmap_holds_the_part_the_refusal_names() -> None:
    from pathlib import Path

    roadmap = (Path(__file__).resolve().parents[2] / "docs" / "roadmap.md").read_text()
    assert "### X63. Missing data, intermediates and designs" in roadmap
    assert "| (c) | a fit with no covariates" in roadmap
