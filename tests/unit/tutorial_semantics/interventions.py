"""The reviewed semantic callback for ``docs/examples/interventions``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs.
"""

from __future__ import annotations

from itertools import pairwise
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

#: The propensity bound the regime fit prints, which Step 6 compares with each plan's ratios.
TRUNCATION = 0.0114

_PROBE = "reviews/notebook-review/probes/interventions-final"
_BINARY = f"{_PROBE}/summary.log, sweep_binary.csv (seeds 9000 to 9119)"
_DOSE = f"{_PROBE}/summary.log, sweep_dose.csv (seeds 9000 to 9119)"

#: Decimals the readings write that no stored output prints, each with its source.
UNPRINTED_DECIMALS = {
    "1.07": f"regime offer-to-all mean SE over empirical SD, degree-2 logistic g: {_BINARY}",
    "1.10": f"regime screen mean SE over empirical SD, degree-2 logistic g: {_BINARY}",
    "5.6": f"the law's share of intensities below zero, 0.055784: {_PROBE}/truth.log",
    "+0.0108": f"+0.5 capped mean error, boosted40: {_DOSE}",
    "+0.0065": f"+0.5 uncapped mean error, boosted40 (+0.00654): {_DOSE}",
    "-0.0155": f"+1.0 mean error, boosted40: {_DOSE}",
    "0.81": f"+0.5 mean SE over empirical SD, boosted40: {_DOSE}",
    "0.61": f"+1.0 mean SE over empirical SD, boosted40: {_DOSE}",
    "+0.0005": f"+1.0 mean error, quad40 (+0.00047): {_DOSE}",
    "0.042": f"+1.0 empirical SD, quad40 (0.04189): {_DOSE}",
    "0.070": f"+1.0 empirical SD, quad80 (0.07001): {_DOSE}",
    "0.024": f"+1.0 mean SE, quad40 and quad80 (0.02395, 0.02392): {_DOSE}",
    "0.0043": f"cap-gap mean error, boosted40 (-0.00425): {_DOSE}",
    "0.024137": f"the double-odds incremental contrast by Monte Carlo: {_PROBE}/truth.log",
    "+0.00011": f"incremental mean error, degree-2 logistic g: {_BINARY}",
    "+0.00066": f"incremental mean error, shallow booster: {_BINARY}",
    "0.00008": f"Monte Carlo SE of each incremental mean error: {_BINARY}",
    "0.93": f"median propensity calibration slope, degree-2 logistic g (0.929): {_BINARY}",
    "0.94": f"median propensity calibration slope, shallow booster (0.939): {_BINARY}",
    "0.9525": "coverage of ate_regime[rule vs never]: tests/canonical/lmtp_regimes/summary.csv",
    "0.9625": (
        "coverage of ate_shift[+0.5 capped vs natural course]: "
        "tests/canonical/lmtp_shift/summary.csv"
    ),
    "0.94625": (
        "coverage of ate_ipsi[odds x2 vs natural course]: "
        "tests/canonical/npcausal_incremental/summary.csv"
    ),
    "0.15": "the shift study's curvature, PRIMARY_CURVATURE: tests/studies/canonical_shift_policies.py",
    "0.25": "the curvature of shift_dgp(): src/cleverly/datasets/synthetic.py",
}

SCREEN = "ate_regime[screen on risk vs offer to none]"
OFFER_ALL = "ate_regime[offer to all vs offer to none]"
CAPPED = "ate_shift[+0.5 capped at 5 vs current practice]"
UNCAPPED = "ate_shift[+0.5 uncapped vs current practice]"
ONE = "ate_shift[+1.0 uncapped vs current practice]"
IPSI = "ate_ipsi[double odds vs current odds]"


def check(namespace: dict[str, Any]) -> None:
    """The three-axis tutorial's truths, support figures, and draw-specific claims hold."""
    offer_all, screen = namespace["truth"]["ate"], namespace["screen_truth"]
    assert 0.5 * offer_all < screen < offer_all
    # The incremental truth carries g, so it moved when the law's g was bounded. truth.py
    # recomputes it from the structural equations as 0.024137 (MC SE 2.4e-06).
    double_odds = namespace["incremental_truth"]["ate_ipsi[odds x2 vs natural course]"]
    assert double_odds == pytest.approx(0.024137, abs=2e-5)
    assert "positivity *for the shifted dose*" in namespace["shift_effect"].summary()
    assert "A cap secures it only when the cap lies inside the conditional support" in (
        namespace["shift_effect"].summary()
    )
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
    # The intensity protocol names an index and a score with no fixed range (IV-04, IV-05).
    dose_protocol = namespace["dose_protocol"]
    assert "standardized" not in dose_protocol.outcome
    assert "no fixed range" in dose_protocol.outcome
    assert not any("contact" in text for text in dose_protocol.treatment_versions)
    assert any("index" in text for text in dose_protocol.assumption_rationale)
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

    # "Each interval contains its population value on this draw", the screen by a narrow
    # margin: its lower end lies within 0.001 below the population value.
    regimes = namespace["regime_result"]
    assert covers(regimes[OFFER_ALL], offer_all)
    assert covers(regimes[SCREEN], screen)
    assert 0.0 < screen - regimes[SCREEN].ci[0] < 0.001

    # "`needs attention` is empty" on the regime fit.
    assert not namespace["regime_assessment"].attention
    rows = namespace["regime_support"].regimes
    # Offer to all has the smallest ratio effective n and the largest ratio.
    assert rows["offer to all"].effective_sample_size == min(
        row.effective_sample_size for row in rows.values()
    )
    assert rows["offer to all"].max_ratio == max(row.max_ratio for row in rows.values())
    # "The largest ratio of each plan is below that cap ... Truncation therefore changes no
    # weight on this draw, and each score load agrees with its ratio effective n."
    assert f"propensity truncated to [{TRUNCATION}, " in stored_output(NOTEBOOK, "regime-fit")
    for row in rows.values():
        assert row.max_ratio < 1 / TRUNCATION
        assert row.score_load["effective"] == pytest.approx(row.effective_sample_size, rel=1e-3)
    # Nonzero witness: the fitted model does put a probability below the bound and below the
    # law's floor of 0.05, so the bound would bind if that row carried weight. The row is
    # off-plan: every on-plan ratio is at most max_ratio, so every on-plan g is at least
    # 1 / max_ratio, which is far above the smallest fitted g.
    smallest = rows["offer to all"].min_support_propensity
    assert smallest < TRUNCATION < 0.05
    assert smallest < 0.01 / rows["offer to all"].max_ratio
    # "The screen assigns a different arm from the observed arm in 1205 rows ... the same count."
    assert namespace["screen_load"]["zero_load"] == namespace["off_plan"] > 0

    # Two warnings, for the uncapped policies; the cap of 5 lies below the largest dose, so
    # the capped policy moves no row past it and raises none.
    warned = namespace["positivity_warnings"]
    assert len(warned) == 2
    assert "'+0.5 uncapped'" in warned[0] and "'+1.0 uncapped'" in warned[1]
    assert not any("capped at 5" in message for message in warned)
    dose = np.asarray(namespace["dose_frame"]["assigned_navigation_intensity"], dtype=float)
    assert dose.max() > 5.0
    support = namespace["shift_assessment"].report("support")
    assert support["current practice"].ess_ratio == pytest.approx(1.0)
    capped = support["+0.5 capped at 5"]
    assert 0.015 < capped.capped_fraction < 0.035
    assert 0.55 < capped.ess_ratio < 0.70
    # The printed widths: the +1.0 interval is wider than either +0.5 interval.
    frame = namespace["shift_result"].to_frame().set_index("estimand")
    width = frame["ci_upper"] - frame["ci_lower"]
    assert width[ONE] > max(width[CAPPED], width[UNCAPPED])
    # "the one interval that excludes its population value, by a small margin".
    dose_truth = namespace["dose_truth"]
    shift_result = namespace["shift_result"]
    assert not covers(shift_result[ONE], dose_truth[ONE])
    assert 0.0 < shift_result[ONE].ci[0] - dose_truth[ONE] < 0.01
    for name in (CAPPED, UNCAPPED):
        assert covers(shift_result[name], dose_truth[name])
    # The estimated density keeps less than the true density ratio, for both shifts.
    assert 0.15 < support["+1.0 uncapped"].ess_ratio < np.exp(-1.0)
    assert support["+0.5 uncapped"].ess_ratio < np.exp(-0.25)
    # "An empty list is not evidence of support."
    assert not namespace["shift_assessment"].attention

    # The cap gap, read as a point contrast through the joint influence curve.
    population_gap = dose_truth[UNCAPPED] - dose_truth[CAPPED]
    assert 0.02 < population_gap < 0.06
    gap = namespace["gap"]
    assert gap.psi == pytest.approx(shift_result[UNCAPPED].psi - shift_result[CAPPED].psi)
    assert gap.psi > 0.0
    assert "95% CI" not in stored_output(NOTEBOOK, "failure-mode")
    # "below the standard error of each estimate": the correlation is load-bearing.
    assert gap.std_error < min(shift_result[CAPPED].std_error, shift_result[UNCAPPED].std_error)
    # "The count is 2 for +0.5 and 3 for +1.0."
    beyond = {delta: int(np.sum(dose + delta > dose.max())) for delta in (0.5, 1.0)}
    assert beyond == {0.5: 2, 1.0: 3}

    # The incremental interval contains its population value on this draw, 0.9 SE away.
    tilt = namespace["incremental_result"][IPSI]
    assert covers(tilt, double_odds)
    assert 0.85 <= (tilt.psi - double_odds) / tilt.std_error < 0.95
    incremental = namespace["incremental_assessment"]
    assert not incremental.attention
    assert "nuisance fits look reasonable" in stored_output(NOTEBOOK, "incremental-fit")
    # "The check marks both rows as solved, with each score below 0.01 of its threshold."
    score_rows = incremental.report("score_equations").rows
    assert {row.name for row in score_rows} >= {"ipsi", "ipsi (mechanism)"}
    assert all(row.passed and row.ratio < 0.01 for row in score_rows)
    # "the outcome-targeting score weight stays between one half and two"; the load describes
    # the outcome equations only, so the mechanism equation has no row-level load in the report.
    assert "covariate in [0.5, 2]" in stored_output(NOTEBOOK, "incremental-support")
    tilts = incremental.report("support")
    assert tilts["double odds"].guaranteed == pytest.approx((0.5, 2.0))
    assert all(item.score_load["equation"].startswith("h_ipsi") for item in tilts.values())
    # "The smallest fitted g is 2.66e-06, the same fitted value as in Step 6."
    assert tilts["double odds"].min_propensity == pytest.approx(smallest, rel=1e-6)

    # The status table of Step 14 prints only the operations that ran or can run.
    status = namespace["status"]
    printed = status[namespace["ran"]]
    assert list(printed.index) == [
        "validation.score_equations",
        "validation.support",
        "validation.nuisance_models",
        "diagnostics.refute",
        "sensitivity.simulated_confounding",
    ]
    assert set(printed.loc["validation.support"]) == {"completed"}
    assert set(printed.loc["validation.score_equations"]) == {"passed"}
    assert set(printed.loc["validation.nuisance_models"]) == {"completed"}
    assert set(printed.loc["sensitivity.simulated_confounding"]) == {"deferred"}
    assert "unavailable" not in stored_output(NOTEBOOK, "assessment")

    # The treatment-only stress surface: every cell returns, and each flip strength moves the
    # screen estimate further down.
    surface = namespace["surface"]
    cells = sorted(surface.cells, key=lambda cell: cell.treatment_strength)
    assert [cell.treatment_strength for cell in cells] == [0.0, 0.05, 0.1, 0.2]
    assert all(cell.outcome_strength == 0.0 and cell.failure is None for cell in cells)
    estimates = [cell.estimate for cell in cells]
    assert all(later < earlier for earlier, later in pairwise(estimates))
    assert cells[2].displacement == pytest.approx(-0.0290, abs=0.001)
    # "The induced association ... stays small in every cell."
    for cell in cells:
        assert cell.induced_treatment_association is not None
        assert abs(cell.induced_treatment_association) < 0.1
    # "A strength of 0.5 sends this score outside the declared support of 0 to 1": the
    # witness for why the grid has no outcome axis.
    assert namespace["regime_result"].estimator.q_bounds == (0.0, 1.0)
    latent = np.random.default_rng(surface.latent_seed).normal(size=len(namespace["frame"]))
    perturbed = namespace["frame"]["transition_score"].to_numpy() - 0.5 * latent
    assert perturbed.min() < 0.0 or perturbed.max() > 1.0
    assert "robustness value refused" not in stored_output(NOTEBOOK, "sensitivity")
