"""The reviewed semantic callback for ``docs/examples/interventions``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.unit.tutorial_semantics import EXAMPLES, stored_output

NOTEBOOK = EXAMPLES / "interventions.ipynb"


def _covers(interval: tuple[float, float], value: float) -> bool:
    return interval[0] <= value <= interval[1]


def check(namespace: dict[str, Any]) -> None:
    """The three-axis tutorial's truths, support figures, and draw-specific claims hold."""
    offer_all, screen = namespace["truth"]["ate"], namespace["screen_truth"]
    assert 0.5 * offer_all < screen < offer_all
    assert "positivity *for the shifted dose*" in namespace["shift_effect"].summary()
    assert "*no positivity assumption*" in namespace["incremental_effect"].summary()

    # Each axis has its own protocol, printed in its step and stamped on its result.
    for cell, protocol, result in (
        ("protocol", "regime_protocol", "regime_result"),
        ("dose-protocol", "dose_protocol", "shift_result"),
        ("incremental-protocol", "incremental_protocol", "incremental_result"),
    ):
        fingerprint = namespace[protocol].fingerprint
        assert fingerprint in stored_output(NOTEBOOK, cell)
        assert namespace[result].provenance.protocol_fingerprint == fingerprint
    assert (
        len(
            {
                namespace[name].fingerprint
                for name in ("regime_protocol", "dose_protocol", "incremental_protocol")
            }
        )
        == 3
    )

    # "Current practice already offers navigation more often to flagged patients."
    current = namespace["current"]
    assert (
        current.loc["flagged", "share offered now"]
        > current.loc["not flagged", "share offered now"]
    )

    # "Each interval contains its population value on this draw."
    regimes = namespace["regime_result"]
    assert _covers(regimes["ate_regime[offer to all vs offer to none]"].ci, offer_all)
    assert _covers(regimes["ate_regime[screen on risk vs offer to none]"].ci, screen)

    # Offer to all has the smallest ratio effective n and the largest ratio; the screen's zero
    # load falls exactly on the rows whose observed arm is off the plan.
    rows = namespace["regime_support"].regimes
    assert rows["offer to all"].effective_sample_size == min(
        row.effective_sample_size for row in rows.values()
    )
    assert rows["offer to all"].max_ratio == max(row.max_ratio for row in rows.values())
    assert namespace["screen_load"]["zero_load"] == namespace["off_plan"] > 0

    warned = namespace["positivity_warnings"]
    assert len(warned) == 2
    assert "'+0.5 uncapped'" in warned[0] and "'+1.0 uncapped'" in warned[1]
    support = namespace["shift_assessment"].report("support")
    assert support["current practice"].ess_ratio == pytest.approx(1.0)
    capped = support["+0.5 capped at 5"]
    assert 0.015 < capped.capped_fraction < 0.035
    assert 0.35 < capped.ess_ratio < 0.55
    # "more than three times as wide as either +0.5 interval"; about 4.5x here.
    frame = namespace["shift_result"].to_frame().set_index("estimand")
    width = frame["ci_upper"] - frame["ci_lower"]
    wide = width["ate_shift[+1.0 uncapped vs current practice]"]
    assert wide > 3.0 * width["ate_shift[+0.5 capped at 5 vs current practice]"]
    assert wide > 3.0 * width["ate_shift[+0.5 uncapped vs current practice]"]
    # "a few percent", far below the exp(-1) a true normal density ratio would keep.
    assert 0.01 < support["+1.0 uncapped"].ess_ratio < 0.1
    assert support["+1.0 uncapped"].ess_ratio < 0.5 * np.exp(-1.0)
    # "An empty list is not evidence of support."
    assert not namespace["shift_assessment"].attention
    truth = namespace["dose_truth"]
    capped_name = "ate_shift[+0.5 capped at 5 vs current practice]"
    uncapped_name = "ate_shift[+0.5 uncapped vs current practice]"
    assert 0.02 < truth[uncapped_name] - truth[capped_name] < 0.06
    capped = namespace["shift_result"][capped_name]
    uncapped = namespace["shift_result"][uncapped_name]
    assert abs(uncapped.psi - capped.psi) < 2.0 * max(capped.std_error, uncapped.std_error)
    dose = np.asarray(namespace["dose_frame"]["assigned_navigation_intensity"], dtype=float)
    # "a handful of rows" above the observed maximum: 2 and 3 rows on this draw.
    beyond = {delta: int(np.sum(dose + delta > dose.max())) for delta in (0.5, 1.0)}
    assert 1 <= beyond[0.5] <= 10 and 2 <= beyond[1.0] <= 10

    incremental = namespace["incremental_assessment"]
    assert "nuisance_models" in {item.name for item in incremental.attention}
    ledger = incremental.to_frame()
    rows = ledger[ledger["check"] == "nuisance_models"]
    assert rows["detail"].str.contains("propensity is poorly calibrated").any()
    # "the interval excludes it on this draw".
    tilt = namespace["incremental_result"]["ate_ipsi[double odds vs current odds]"]
    assert not _covers(
        tilt.ci, namespace["incremental_truth"]["ate_ipsi[odds x2 vs natural course]"]
    )
    assert "ipsi (mechanism)" in stored_output(NOTEBOOK, "incremental-fit")

    # One axis per fit: the mixed request and the omitted-variable bound are refused.
    assert "refused: incremental propensity-score interventions" in stored_output(
        NOTEBOOK, "one-axis-per-fit"
    )
    assert "robustness value refused" in stored_output(NOTEBOOK, "sensitivity")
    # "The outcome strength moves the flipped estimate by about as much as it moves the original fit."
    cells = {
        (cell.treatment_strength, cell.outcome_strength): cell
        for cell in namespace["surface"].cells
    }
    assert all(cell.failure is None for cell in cells.values())
    outcome_only = cells[0.0, 0.5].estimate - cells[0.0, 0.0].estimate
    with_flips = cells[0.1, 0.5].estimate - cells[0.1, 0.0].estimate
    assert abs(with_flips - outcome_only) < 0.25 * abs(outcome_only)
    assert abs(cells[0.1, 0.0].induced_treatment_association) < 0.1
