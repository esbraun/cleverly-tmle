"""The reviewed semantic callback for ``docs/examples/interventions``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def check(namespace: dict[str, Any]) -> None:
    """The three-axis tutorial's truths, support figures, and draw-specific claims hold."""
    offer_all, screen = namespace["truth"]["ate"], namespace["screen_truth"]
    assert 0.5 * offer_all < screen < offer_all
    assert "positivity *for the shifted dose*" in namespace["shift_effect"].summary()
    assert "*no positivity assumption*" in namespace["incremental_effect"].summary()
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
