"""The declaration of ``continuous-survival-functionals`` (X6 plan rev 2, section 5.4)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from cleverly.survival import ContinuousSurvivalTMLE
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_functionals as study
from tests.studies import continuous_survival_functionals_law as functionals
from tests.studies import continuous_survival_functionals_properties as properties
from tests.studies import continuous_survival_law as law

STUDIES = Path(__file__).resolve().parents[1] / "studies"


def test_the_declaration_is_the_plans() -> None:
    assert study.PILOT_REPLICATES == 800
    assert study.PRIMARY_REPLICATES == 3_200 == study.STUDY.replicates
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
