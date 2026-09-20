"""The reviewed semantic callback for ``docs/examples/interventions``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs, and every decimal it writes is printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import navigation_protocol
from tests.unit.tutorial_semantics import (
    EXAMPLES,
    assert_protocol_recorded,
    changed_fields,
    covers,
    stored_output,
)

NOTEBOOK = EXAMPLES / "interventions.ipynb"

#: The propensity bound the regime fit prints, which Step 6 compares with each plan's min g.
TRUNCATION = 0.0114


def check(namespace: dict[str, Any]) -> None:
    """The three-axis tutorial's truths, support figures, and draw-specific claims hold."""
    offer_all, screen = namespace["truth"]["ate"], namespace["screen_truth"]
    assert 0.5 * offer_all < screen < offer_all
    assert "positivity *for the shifted dose*" in namespace["shift_effect"].summary()
    assert "*no positivity assumption*" in namespace["incremental_effect"].summary()

    # Each reading names the fields its protocol changes in the program protocol.
    program = navigation_protocol()
    assert changed_fields(namespace["regime_protocol"], program) == {
        "treatment_strategies",
        "treatment_versions",
        "assumption_rationale",
    }
    assert changed_fields(namespace["dose_protocol"], program) == {
        "outcome",
        "time_zero",
        "treatment_strategies",
        "treatment_versions",
        "intercurrent_event_handling",
        "assumption_rationale",
    }
    assert changed_fields(namespace["incremental_protocol"], program) == {
        "treatment_strategies",
        "treatment_versions",
        "assumption_rationale",
    }
    # "Offer to none reuses the program's version for usual support."
    assert namespace["regime_protocol"].treatment_versions[0] == program.treatment_versions[1]
    # Each axis has its own protocol, printed in its step and stamped on its result.
    for cell, protocol, result in (
        ("protocol", "regime_protocol", "regime_result"),
        ("dose-identify", "dose_protocol", "shift_result"),
        ("incremental-identify", "incremental_protocol", "incremental_result"),
    ):
        assert_protocol_recorded(NOTEBOOK, cell, namespace[protocol], namespace[result])
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
    assert covers(regimes["ate_regime[offer to all vs offer to none]"], offer_all)
    assert covers(regimes["ate_regime[screen on risk vs offer to none]"], screen)

    # "`needs attention` names `nuisance_models`" on the regime fit.
    assert "nuisance_models" in {i.name for i in namespace["regime_assessment"].attention}
    # Offer to all has the smallest ratio effective n and the largest ratio.
    rows = namespace["regime_support"].regimes
    assert rows["offer to all"].effective_sample_size == min(
        row.effective_sample_size for row in rows.values()
    )
    assert rows["offer to all"].max_ratio == max(row.max_ratio for row in rows.values())
    # The support columns use g before truncation, and the score load uses the truncated g.
    # Offer to all's largest ratio is above the 1/0.0114 cap, so truncation raises its load
    # above its ratio ESS. Offer to none's largest ratio is below the cap, so its two numbers
    # barely move: the nonzero witness that truncation, not a different definition, separates
    # the columns.
    assert f"propensity truncated to [{TRUNCATION}, " in stored_output(NOTEBOOK, "regime-fit")
    assert rows["offer to all"].max_ratio > 1 / TRUNCATION
    assert rows["offer to none"].max_ratio < 1 / TRUNCATION
    assert rows["offer to all"].score_load["effective"] > (
        1.02 * rows["offer to all"].effective_sample_size
    )
    # Both print as about 476.5; they agree to about 1e-4 relative, not exactly.
    assert rows["offer to none"].score_load["effective"] == pytest.approx(
        rows["offer to none"].effective_sample_size, rel=1e-3
    )
    # The largest load of offer to all is the truncation cap 1/0.0114, not its untruncated ratio.
    assert rows["offer to all"].score_load["max_load"] == pytest.approx(1 / TRUNCATION, rel=1e-3)
    assert rows["offer to all"].max_ratio > 1.1 * rows["offer to all"].score_load["max_load"]
    assert rows["offer to none"].score_load["max_load"] == pytest.approx(
        rows["offer to none"].max_ratio, rel=1e-6
    )
    # "The screen assigns a different arm from the observed arm in 1173 rows ... the same count."
    assert namespace["screen_load"]["zero_load"] == namespace["off_plan"] > 0

    warned = namespace["positivity_warnings"]
    assert len(warned) == 2
    assert "'+0.5 uncapped'" in warned[0] and "'+1.0 uncapped'" in warned[1]
    support = namespace["shift_assessment"].report("support")
    assert support["current practice"].ess_ratio == pytest.approx(1.0)
    capped = support["+0.5 capped at 5"]
    assert 0.015 < capped.capped_fraction < 0.035
    assert 0.55 < capped.ess_ratio < 0.70
    # "about twice as wide as either +0.5 interval"; about 2.1x here.
    frame = namespace["shift_result"].to_frame().set_index("estimand")
    width = frame["ci_upper"] - frame["ci_lower"]
    wide = width["ate_shift[+1.0 uncapped vs current practice]"]
    assert wide > 1.8 * width["ate_shift[+0.5 capped at 5 vs current practice]"]
    assert wide > 1.8 * width["ate_shift[+0.5 uncapped vs current practice]"]
    # "the one interval that excludes its population value, by a small margin".
    dose_truth = namespace["dose_truth"]
    wide_name = "ate_shift[+1.0 uncapped vs current practice]"
    assert not covers(namespace["shift_result"][wide_name], dose_truth[wide_name])
    for name in (
        "ate_shift[+0.5 capped at 5 vs current practice]",
        "ate_shift[+0.5 uncapped vs current practice]",
    ):
        assert covers(namespace["shift_result"][name], dose_truth[name])
    # The estimated density keeps less than the true density ratio, for both shifts.
    assert 0.15 < support["+1.0 uncapped"].ess_ratio < np.exp(-1.0)
    assert support["+0.5 uncapped"].ess_ratio < np.exp(-0.25)
    # "An empty list is not evidence of support."
    assert not namespace["shift_assessment"].attention

    # The cap gap, read through the joint influence curve.
    truth = namespace["dose_truth"]
    capped_name = "ate_shift[+0.5 capped at 5 vs current practice]"
    uncapped_name = "ate_shift[+0.5 uncapped vs current practice]"
    population_gap = truth[uncapped_name] - truth[capped_name]
    assert 0.02 < population_gap < 0.06
    gap = namespace["gap"]
    capped = namespace["shift_result"][capped_name]
    uncapped = namespace["shift_result"][uncapped_name]
    assert gap.psi == pytest.approx(uncapped.psi - capped.psi)
    # "the interval contains the population gap and excludes zero".
    assert covers(gap, population_gap)
    assert gap.ci[0] > 0.0
    # "far below the standard error of each estimate": the correlation is load-bearing.
    assert gap.std_error < 0.5 * min(capped.std_error, uncapped.std_error)
    dose = np.asarray(namespace["dose_frame"]["assigned_navigation_intensity"], dtype=float)
    # "a handful of rows" above the observed maximum: 2 and 3 rows on this draw.
    beyond = {delta: int(np.sum(dose + delta > dose.max())) for delta in (0.5, 1.0)}
    assert 1 <= beyond[0.5] <= 10 and 2 <= beyond[1.0] <= 10

    incremental = namespace["incremental_assessment"]
    assert "nuisance_models" in {item.name for item in incremental.attention}
    ledger = incremental.to_frame()
    nuisance_rows = ledger[ledger["check"] == "nuisance_models"]
    assert nuisance_rows["detail"].str.contains("propensity is poorly calibrated").any()
    # "the interval excludes it on this draw", at 2.2 standard errors.
    tilt = namespace["incremental_result"]["ate_ipsi[double odds vs current odds]"]
    double_odds = namespace["incremental_truth"]["ate_ipsi[odds x2 vs natural course]"]
    assert not covers(tilt, double_odds)
    assert 2.55 <= (tilt.psi - double_odds) / tilt.std_error < 2.65
    assert "ipsi (mechanism)" in stored_output(NOTEBOOK, "incremental-fit")
    # "the outcome-targeting score weight stays between one half and two"; the load describes
    # the outcome equations only, so the mechanism equation has no row-level load in the report.
    assert "covariate in [0.5, 2]" in stored_output(NOTEBOOK, "incremental-support")
    tilts = incremental.report("support")
    assert tilts["double odds"].guaranteed == pytest.approx((0.5, 2.0))
    assert all(item.score_load["equation"].startswith("h_ipsi") for item in tilts.values())

    # The status table of Step 14, as its reading narrates it.
    status = namespace["status"]
    assert set(status.loc["validation.support"]) == {"completed"}
    assert set(status.loc["validation.score_equations"]) == {"passed"}
    assert status.loc["validation.nuisance_models"].to_dict() == {
        "known regime": "warning",
        "modified treatment policy": "completed",
        "incremental intervention": "warning",
    }
    assert set(status.loc["sensitivity.robustness_value"]) == {"unavailable"}
    assert set(status.loc["sensitivity.simulated_confounding"]) == {"deferred"}

    # The omitted-variable bound is refused for the regime fit.
    assert "robustness value refused" in stored_output(NOTEBOOK, "sensitivity")
    cells = {
        (cell.treatment_strength, cell.outcome_strength): cell
        for cell in namespace["surface"].cells
    }
    # "Two of its four cells return an estimate, and two record a failure": the outcome
    # perturbation sends the score off the support this fit declares, and the surface keeps
    # each failure per cell instead of abandoning the run. The zero-strength column is the
    # nonzero witness that only the outcome axis fails.
    for treatment_strength in (0.0, 0.1):
        assert cells[treatment_strength, 0.0].failure is None
        failed = cells[treatment_strength, 0.5]
        assert failed.estimate is None
        assert failed.failure is not None
        assert "outside q_bounds" in failed.failure.message
    assert namespace["regime_result"].estimator.q_bounds == (0.0, 1.0)
    # "The flipped cell that ran moves the estimate by -0.029598."
    original = cells[0.0, 0.0].estimate
    assert cells[0.1, 0.0].estimate - original == pytest.approx(-0.0296, abs=0.001)
    # "the flips induce an association of only +0.0442", with P(A=1) near one half.
    association = cells[0.1, 0.0].induced_treatment_association
    assert association is not None
    assert 0.0 < association < 0.1
    assert abs(namespace["frame"]["transition_navigation"].mean() - 0.5) < 0.05
    # The flips go in both directions, but they raise the offer rate on this draw.
    offer_rates = namespace["offer_rates"]
    assert offer_rates["after flips"] > offer_rates["before flips"]
