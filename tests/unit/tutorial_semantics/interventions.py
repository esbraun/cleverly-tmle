"""The reviewed semantic callback for ``docs/examples/interventions``.

``tests/unit/test_documentation_runtime.py`` runs the tutorial at its documented size and
passes the resulting namespace to :func:`check`.  The tutorial is a notebook, so its narrated
decimals are also compared against its stored outputs.
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
    # The shown dose configuration, quad_rboost40.
    "-0.0004": f"+0.5 capped mean error, quad_rboost40 (-0.00040): {_DOSE}",
    "+0.0002": f"+0.5 uncapped mean error, quad_rboost40 (+0.00020): {_DOSE}",
    "-0.0005": f"+1.0 mean error, quad_rboost40 (-0.00052): {_DOSE}",
    "0.0010": f"MC SE of the +0.5 mean errors, quad_rboost40 (0.00102, 0.00097): {_DOSE}",
    "0.0025": f"MC SE of the +1.0 mean error, quad_rboost40 (0.00248): {_DOSE}",
    "1.17": f"+0.5 capped mean SE over empirical SD, quad_rboost40: {_DOSE}",
    "1.16": f"+0.5 uncapped mean SE over empirical SD, quad_rboost40: {_DOSE}",
    "0.95": f"+1.0 mean SE over empirical SD, quad_rboost40: {_DOSE}",
    "+0.0006": f"cap-gap mean error, quad_rboost40 (+0.00059): {_DOSE}",
    "0.0005": f"MC SE of the cap-gap mean error, quad_rboost40 (0.00046): {_DOSE}",
    "1.05": f"largest mean of the mean density ratio, quad_rboost40 (+1.0, 1.048): {_DOSE}",
    # The density comparison, same quadratic Q.
    "1.01": f"largest mean SE over empirical SD, quad_oracle320 (+0.5 uncapped): {_DOSE}",
    "1.00": f"mean of the +0.5 capped mean density ratio, quad_oracle320 (1.002): {_DOSE}",
    "0.57": f"+1.0 mean SE over empirical SD, quad40: {_DOSE}",
    "0.73": f"+0.5 capped mean SE over empirical SD, quad40: {_DOSE}",
    # The cross-fitted shift probe that the technical reference cites.
    "3.91": (
        "cross-fitted +1.0 mean SE over empirical SD, default booster density: "
        "reviews/notebook-review/probes/iv-n3/summary_ic.log, point-treatment-tmle.md"
    ),
    "0.98": (
        "cross-fitted +1.0 mean SE over empirical SD, exact density at 320 bins: "
        "reviews/notebook-review/probes/iv-n3/summary_ic.log, point-treatment-tmle.md"
    ),
    "0.024137": f"the double-odds incremental contrast by Monte Carlo: {_PROBE}/truth.log",
    "+0.00011": f"incremental mean error, degree-2 logistic g: {_BINARY}",
    "+0.00066": f"incremental mean error, shallow booster: {_BINARY}",
    "0.00008": f"Monte Carlo SE of each incremental mean error: {_BINARY}",
    "0.93": (
        f"median propensity calibration slope, degree-2 logistic g (0.929): {_BINARY}; "
        f"smallest mean SE over empirical SD, quad_oracle320 (+0.5 capped), and the cap-gap "
        f"ratio, quad_rboost40: {_DOSE}"
    ),
    "0.94": f"median propensity calibration slope, shallow booster (0.939): {_BINARY}",
    "0.9525": "coverage of ate_regime[rule vs never]: tests/canonical/lmtp_regimes/summary.csv",
    "0.9625": (
        "coverage of ate_policy[+0.5 capped vs natural course]: "
        "tests/canonical/lmtp_shift/summary.csv"
    ),
    "0.94625": (
        "coverage of ate_ipsi[odds x2 vs natural course]: "
        "tests/canonical/npcausal_incremental/summary.csv"
    ),
    "3.2": "a section number of Kennedy (2019), docs/references.md#incremental-interventions",
    "3.3": "a section number of Kennedy (2019), docs/references.md#incremental-interventions",
    "4.5": "the capacity of 5 minus the raise of 0.5, the threshold the cap acts on",
    "0.15": "the shift study's curvature, PRIMARY_CURVATURE: tests/studies/canonical_shift_policies.py",
    "0.25": "the curvature of shift_dgp(): src/cleverly/datasets/synthetic.py",
}

SCREEN = "ate_regime[screen on risk vs offer to none]"
OFFER_ALL = "ate_regime[offer to all vs offer to none]"
CAPPED = "ate_policy[+0.5 capped at 5 vs current practice]"
UNCAPPED = "ate_policy[+0.5 uncapped vs current practice]"
ONE = "ate_policy[+1.0 uncapped vs current practice]"
IPSI = "ate_ipsi[double odds vs current odds]"


def check(namespace: dict[str, Any]) -> None:
    """The three-proposal tutorial's truths, support figures, and draw-specific claims hold."""
    offer_all, screen = namespace["truth"]["ate"], namespace["screen_truth"]
    assert 0.5 * offer_all < screen < offer_all
    # The incremental truth carries g, so it moved when the law's g was bounded. truth.py
    # recomputes it from the structural equations as 0.024137 (MC SE 2.4e-06).
    double_odds = namespace["incremental_truth"]["ate_ipsi[odds x2 vs natural course]"]
    assert double_odds == pytest.approx(0.024137, abs=2e-5)
    assert "positivity *for the assigned dose*" in namespace["shift_effect"].summary()
    assert "An upper cap alone does not ensure this for negative shifts or support gaps" in (
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
    # "offer to all minus screen: 0.056 (0.048, 0.065) ... population value 0.057": the gap
    # comes from the joint influence curve, so its interval is narrower than either contrast's.
    given_up = namespace["given_up"]
    assert given_up.psi == pytest.approx(regimes[OFFER_ALL].psi - regimes[SCREEN].psi)
    assert covers(given_up, offer_all - screen)
    assert given_up.std_error < min(regimes[OFFER_ALL].std_error, regimes[SCREEN].std_error)

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
    # "The screen asks for offers where current practice already offers more often, and its
    # effective sample size is the largest."
    assert rows["screen on risk"].effective_sample_size == max(
        row.effective_sample_size for row in rows.values()
    )

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
    assert 0.60 < capped.ess_ratio < 0.75
    # The printed widths: the +1.0 interval is wider than either +0.5 interval.
    frame = namespace["shift_result"].to_frame().set_index("estimand")
    width = frame["ci_upper"] - frame["ci_lower"]
    assert width[ONE] > max(width[CAPPED], width[UNCAPPED])
    # "On this draw each interval contains its population value."
    dose_truth = namespace["dose_truth"]
    shift_result = namespace["shift_result"]
    for name in (CAPPED, UNCAPPED, ONE):
        assert covers(shift_result[name], dose_truth[name])
    # The shown fit is the quadratic Q and the regularized density booster of Step 9.
    models = namespace["shift_method"].models
    assert models.density_bins == 40
    assert models.treatment_learner.get_params()["max_depth"] == 2
    assert models.treatment_learner.get_params()["l2_regularization"] == 1.0
    assert models.outcome_learner.get_params()["polynomialfeatures__degree"] == 2
    # On this draw the estimated density keeps less than the true density ratio, for both
    # shifts.
    assert 0.15 < support["+1.0 uncapped"].ess_ratio < np.exp(-1.0)
    assert support["+0.5 uncapped"].ess_ratio < np.exp(-0.25)
    # The histogram reports zeros at assigned doses outside its fitted bin range.
    # These counts describe this seeded output, not population positivity.
    assert {name: row.unsupported for name, row in support.items()} == {
        "current practice": 0,
        "+0.5 capped at 5": 0,
        "+0.5 uncapped": 2,
        "+1.0 uncapped": 3,
    }
    assert tuple(item.name for item in namespace["shift_assessment"].attention) == ("support",)
    # The mean ratio: reference 1 for every shift of a normal intensity. "The estimated ratio
    # runs 3% to 8% high on average, and most for +1.0." Current practice reads exactly 1
    # (control); each moving shift reads above it (nonzero witness). In sample the one
    # per-fold value equals the overall mean.
    assert support["current practice"].mean_ratio == pytest.approx(1.0)
    means = {name: support[name].mean_ratio for name in ("+0.5 capped at 5", "+0.5 uncapped")}
    means["+1.0 uncapped"] = support["+1.0 uncapped"].mean_ratio
    assert all(1.02 < value < 1.09 for value in means.values())
    assert means["+1.0 uncapped"] == max(means.values())
    for name in support:
        assert support[name].fold_mean_ratio == (support[name].mean_ratio,)

    # The cap gap, read with its interval through the joint influence curve.
    population_gap = dose_truth[UNCAPPED] - dose_truth[CAPPED]
    assert 0.02 < population_gap < 0.06
    gap = namespace["gap"]
    assert gap.psi == pytest.approx(shift_result[UNCAPPED].psi - shift_result[CAPPED].psi)
    assert gap.psi > 0.0
    # "The population gap lies within 1.96 standard errors of the estimate"; the interval
    # also excludes zero, so the gap it covers is a nonzero one.
    assert abs(gap.psi - population_gap) < 1.96 * gap.std_error
    assert covers(gap, population_gap)
    assert gap.ci[0] > 0.0
    # "below the standard error of each estimate": the correlation is load-bearing.
    assert gap.std_error < min(shift_result[CAPPED].std_error, shift_result[UNCAPPED].std_error)

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
    # "the clever covariate stays between one half and two for this multiplier".
    tilts = incremental.report("support")
    assert tilts["double odds"].guaranteed == pytest.approx((0.5, 2.0))

    # The treatment-only stress surface: both cells return, and the flip moves the screen
    # estimate down by 0.029037 (-6.1 standard errors) with a small induced association.
    surface = namespace["surface"]
    cells = sorted(surface.cells, key=lambda cell: cell.treatment_strength)
    assert [cell.treatment_strength for cell in cells] == [0.0, 0.1]
    assert all(cell.outcome_strength == 0.0 and cell.failure is None for cell in cells)
    assert cells[1].estimate < cells[0].estimate
    assert cells[1].displacement == pytest.approx(-0.0290, abs=0.001)
    for cell in cells:
        assert cell.induced_treatment_association is not None
        assert abs(cell.induced_treatment_association) < 0.1
    # "An outcome strength ... moves the score outside its declared range of 0 to 1": the
    # witness for why the grid has no outcome axis.
    assert namespace["regime_result"].estimator.q_bounds == (0.0, 1.0)
    latent = np.random.default_rng(surface.latent_seed).normal(size=len(namespace["frame"]))
    perturbed = namespace["frame"]["transition_score"].to_numpy() - 0.5 * latent
    assert perturbed.min() < 0.0 or perturbed.max() > 1.0
