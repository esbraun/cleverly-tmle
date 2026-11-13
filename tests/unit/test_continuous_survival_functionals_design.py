"""The declaration of ``continuous-survival-functionals`` (X6 plan rev 2, section 5.4)."""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleverly.survival import ContinuousSurvivalTMLE
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_functionals as study
from tests.studies import continuous_survival_functionals_law as functionals
from tests.studies import continuous_survival_functionals_pilot as pilot
from tests.studies import continuous_survival_functionals_properties as properties
from tests.studies import continuous_survival_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import replicate_seed

STUDIES = Path(__file__).resolve().parents[1] / "studies"


def test_the_declaration_is_the_plans() -> None:
    assert study.PILOT_REPLICATES == 800
    assert study.STUDY.replicates == study.PRIMARY_REPLICATES
    assert study.PRIMARY_N == 1_000 == study.STUDY.n
    assert study.TAU == law.TIMES[2] == 2.32
    assert study.MARGINS.standardized_bias == 0.2
    assert study.STUDY.reference is None
    assert properties.CALIBRATION_REPLICATES == 3_200
    assert properties.NECESSITY_REPLICATES == 1_000
    assert properties.SHRUNKEN_SE_FACTOR == 0.85
    assert tuple(study.STUDY.scenarios) == (study.GATED_CELL,)
    assert study.STUDY.property_cells == {
        "interval_calibration": (
            "rmst_rd__correctly_specified",
            "rmst_rd__shrunken_se_control",
            "rmst_rd__noise_control",
        )
    }


def test_the_gated_rows_are_the_plans_25() -> None:
    names = functionals.GATED
    assert len(names) == len(set(names)) == 25
    # Risks and differences at the median and upper quartile: 8 + 4; ratios there: 4.
    assert sum(name.startswith("cif_regimen") for name in names) == 8
    assert sum(name.startswith("ate_regimen") for name in names) == 4
    assert sum(name.startswith("rr_regimen") for name in names) == 4
    assert not any(f"t={law.TIMES[0]:g}]" in name for name in names)
    # RMTL of each arm and cause, their differences; RMST of each arm, its difference.
    assert sum(name.startswith("rmtl_regimen") for name in names) == 6
    assert sum(name.startswith("rmst_regimen") for name in names) == 3


def test_the_seeds_are_used_by_no_other_study() -> None:
    mine = {str(study.SEED), str(study.RESAMPLING_SEED)}
    for path in STUDIES.glob("*.py"):
        if path.name == "continuous_survival_functionals.py":
            continue
        numbers = set(re.findall(r"\d{7,}", path.read_text(encoding="utf-8").replace("_", "")))
        assert not mine & numbers, path.name


def test_each_restricted_truth_is_the_integral_of_the_risks() -> None:
    nodes, weights = np.polynomial.legendre.leggauss(40)
    t = study.TAU / 2.0 * (nodes + 1.0)
    w = study.TAU / 2.0 * weights
    truth = functionals.TRUTH
    for arm, z in law.ARMS.items():
        lost = 0.0
        for code, cause in law.CAUSES.items():
            integral = float(sum(wi * law.risk(z, code, ti) for ti, wi in zip(t, w, strict=True)))
            assert truth[functionals.rmtl_name(arm, cause)] == pytest.approx(integral, abs=1e-9)
            lost += integral
        assert truth[functionals.rmst_name(arm)] == pytest.approx(study.TAU - lost, abs=1e-9)
    difference = truth[functionals.rmst_name("treated")] - truth[functionals.rmst_name("control")]
    assert truth[functionals.rmst_difference_name()] == pytest.approx(difference)
    for cause in law.CAUSES.values():
        for time in functionals.GATED_TIMES:
            ratio = (
                law.TRUTH[law.level_name("treated", cause, time)]
                / law.TRUTH[law.level_name("control", cause, time)]
            )
            assert truth[functionals.ratio_name(cause, time)] == pytest.approx(ratio)


def test_the_noise_control_is_sized_by_a_positive_efficient_spread() -> None:
    sd = properties.EFFICIENCY_SD["rmst_rd"]
    assert np.isfinite(sd) and 1.0 < sd < 4.0


def _frame() -> object:
    return registered.sample(400, 20261113)


def _fit(*, probe: bool) -> object:
    subject = ContinuousSurvivalTMLE(
        law.TIMES,
        restricted_horizons=(study.TAU,),
        arms=law.ARMS,
        reference=law.REFERENCE,
        causes=law.CAUSES,
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if probe:
            with properties.risks_targeted_only(len(law.TIMES)):
                return registered.fit_with(_frame(), subject)
        return registered.fit_with(_frame(), subject)


def test_the_risk_only_walk_leaves_the_restricted_means_untargeted() -> None:
    probed, targeted = _fit(probe=True), _fit(probe=False)
    scores = np.asarray(probed.record.standardized_score)
    cutoff = probed.record.cutoff
    rmtl = {
        (arm, cause): properties._walk_index(arm, cause)
        for arm in law.ARMS
        for cause in law.CAUSES.values()
    }
    risk_rows = np.setdiff1d(np.arange(scores.size), list(rmtl.values()))
    assert probed.record.converged
    assert np.all(scores[risk_rows] <= cutoff)
    # Each RMTL's recorded score is its own: |P_n D| / sqrt(P_n D^2) of its reported curve.
    for (arm, cause), index in rmtl.items():
        curve = np.asarray(probed[functionals.rmtl_name(arm, cause)].influence_curve)
        own = abs(curve.mean()) / np.sqrt(np.mean(curve**2))
        assert scores[index] == pytest.approx(own, rel=1e-6, abs=1e-12)
    # Witness: the probe leaves the restricted means where the risk walk put them, which is
    # not where the targeted fit puts them, while every target of that fit meets the cutoff.
    targeted_scores = np.asarray(targeted.record.standardized_score)
    assert np.all(targeted_scores <= targeted.record.cutoff)
    indices = list(rmtl.values())
    assert not np.allclose(scores[indices], targeted_scores[indices])
    name = functionals.rmtl_name("treated", "cause1")
    assert probed[name].psi != targeted[name].psi


# ------------------------------------------------------------------------- the sizes

PILOT = pilot.read()


def test_the_primary_count_is_the_shared_rules_on_the_committed_pilot() -> None:
    # Before the pilot is committed the record holds the placeholder, which regenerate.py
    # refuses; once it is committed the count must be the rule's, so the placeholder fails.
    if PILOT.empty:
        assert study.PRIMARY_REPLICATES == study.PENDING_REPLICATES
        return
    assert PILOT["replicate"].nunique() == study.PILOT_REPLICATES
    assert set(PILOT["n"]) == {study.PRIMARY_N}
    refused = PILOT.loc[PILOT["refusal"].fillna("").ne(""), "replicate"].nunique()
    assert refused / study.PILOT_REPLICATES <= study.FAILURE_BOUND
    assert pilot.primary_count(PILOT) == study.PRIMARY_REPLICATES


def test_the_calibration_count_meets_the_at_target_rule() -> None:
    declared = properties.CALIBRATION_REPLICATES
    assert declared % sizing.BUDGET_STEP == 0
    assert declared >= pilot.at_target_count() == 2_400
    at_target = float(sizing.calibration_pass(study.MARGINS, declared, np.array([1.0]), 0.95)[0])
    assert at_target >= 0.95
    assert (
        sizing.calibration_control_probability(
            study.MARGINS, declared, properties.SHRUNKEN_SE_FACTOR
        )
        > 0.999
    )
    if not PILOT.empty:
        _, plug_in = pilot.calibration_check(PILOT, declared)
        assert plug_in >= 0.90, "extend the pilot before declaring the calibration cell"


def test_a_declared_run_waits_for_the_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.canonical.continuous_survival_functionals import regenerate

    monkeypatch.setattr(sys, "argv", ["regenerate", "--output", "/tmp/x"])
    if PILOT.empty:
        with pytest.raises(SystemExit, match="run the pilot first"):
            regenerate.refuse_unsized()
    monkeypatch.setattr(sys, "argv", ["regenerate", "--replicates", "2", "--output", "/tmp/x"])
    regenerate.refuse_unsized()


def test_the_pilot_draws_from_its_own_stream() -> None:
    primary = {
        replicate_seed(study.STUDY, study.GATED_CELL, r) for r in range(study.PILOT_REPLICATES)
    }
    seeds = {pilot.pilot_seed(r) for r in range(study.PILOT_REPLICATES)}
    assert len(seeds) == study.PILOT_REPLICATES
    assert not seeds & primary


def test_the_pilot_sizes_from_rows_like_its_own() -> None:
    # Synthetic rows in the pilot's columns, calibrated and unbiased, so the rule runs end to
    # end: the bias clause at the 0.2 margin needs at least 6,000.
    rng = np.random.default_rng(0)
    rows = []
    for replicate in range(study.PILOT_REPLICATES):
        for name in functionals.GATED:
            error = rng.normal(0.0, 0.01)
            rows.append(
                {
                    "replicate": replicate,
                    "estimand": name,
                    "truth": 0.5,
                    "estimate": 0.5 + error,
                    "inference_estimate": 0.5 + error,
                    "inference_scale": "identity",
                    "std_error": 0.01,
                    "covered": int(abs(error) <= 0.0196),
                    "refusal": "",
                }
            )
    frame = pd.DataFrame(rows)
    assert pilot.primary_count(frame) >= 6_000
    at_target, plug_in = pilot.calibration_check(frame, properties.CALIBRATION_REPLICATES)
    assert at_target > 0.99 and plug_in > 0.90
