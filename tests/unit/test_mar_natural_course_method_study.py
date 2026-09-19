"""Focused contracts for the ordinary MAR natural-course comparator study.

The registered verdicts are recomputed by ``tests/unit/test_method_evidence.py``.  These checks
cover what that module cannot see: the reference payload, the R adapter's pinned settings, the
scale-workaround probe, and per-replication parity with R ``tmle`` 2.1.1.  The 0.15-SD
similarity margin alone cannot separate a missing targeting step, a wrong outcome scale, or a
different variance rule, so each has its own bound and a deliberate-mutation control here.  The
wrong scale is bounded against the probe's nonzero witness, ``unplanted_point_difference``,
because the targeting ratio admits a shift that small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_mar as mar
from tests.studies import canonical_mar_natural_course as study
from tests.studies.evidence.registry import ROOT

#: The largest ``|cleverly - R| / |targeting move|`` the parity check admits in any
#: replication.  Both implementations target the same supplied predictions, so their initial
#: estimates agree to rounding and their targeted estimates differ only through where each
#: fluctuation solver stops.  cleverly iterates to a relative score of ``1e-10``.  R ``tmle``
#: fits the fluctuation with ``glm`` at its default ``epsilon = 1e-8``, a relative change in
#: deviance.  The deviance is quadratic in the coefficient near its minimum, so that stop
#: resolves the coefficient, and with it the move, to about ``sqrt(1e-8) = 1e-4`` of its scale.
#: The bound allows a hundredfold over that.  A missing targeting step has ratio 1, so the
#: bound sits two orders below it.  The rule and its derivation are the arm-indexed study's
#: (``tests/unit/test_arm_indexed_cvtmle_method_study.py``).
TARGETING_RATIO_BOUND = 1e-2

#: The largest ``|initial_cleverly - initial_R|`` in any replication.  R's plug-in is the mean
#: of the same serialized predictions, so only CSV round-trip and summation order separate them.
INITIAL_ESTIMATE_BOUND = 1e-12

#: The largest ``|SE_cleverly - SE_R| / SE_R`` in any replication.  A non-cross-fitted
#: natural-course fit uses the centered rule ``var(IC, ddof=1) / n``
#: (``src/cleverly/estimators/tmle.py``, ``covariance_rule``), which is R's ``var(IC) / n``.
#: The two curves differ only through the targeted fit, which the targeting bound above
#: already resolves.  The stacked study's uncentered second-moment rule differs from R's by the
#: factor ``sqrt((n - 1) / n)``, a relative gap of about ``2.5e-4`` at ``n = 2000``.  The bound
#: sits two orders below that rule gap.  It was declared before the regeneration it checks.
SE_RELATIVE_BOUND = 1e-6

#: The largest ``|cleverly - R| / unplanted_point_difference`` in any continuous replication.
#: The witness is the point shift of R's own fit when its scale moves from ``c(0, 1)`` to the
#: observed range.  A ``cleverly`` estimate that matched the unplanted scale would sit about one
#: witness from R's published point, a ratio near 1.  The committed gap is solver-stop noise,
#: and its largest measured ratio is ``1.44e-3``, at replication 596.  The bound sits two orders
#: below the wrong-scale ratio and about sevenfold above that maximum.  Unlike the bounds above,
#: it was chosen after the regeneration, from the committed artifacts.  It is also not derived:
#: the witness has no floor for this law, and a law whose observed range reaches ``(0, 1)``
#: would drive it to zero.  The check therefore covers the committed rows and no other draw.
WITNESS_RATIO_BOUND = 1e-2


@pytest.mark.parametrize("scenario", study.SCENARIOS)
def test_the_reference_payload_serializes_the_fitted_predictions(scenario: str) -> None:
    frame, _ = study.draw_scenario(scenario, 200, 0)
    result = study.fit_cleverly(frame, scenario)
    sample = study.reference_sample(frame, result, scenario=scenario, replicate=0)

    # The expectations come from the finite law and the drawn frame, not from the fit.  A
    # payload that read another arm's column, a counterfactual prediction, or the scaled
    # predictions under a data-range scaler would fail here, because both nuisances differ
    # by arm in every covariate level (``tests/discrete_law_mar.py``).
    level = frame["W"].to_numpy(dtype=int)
    arm = frame["A"].to_numpy(dtype=int)
    respondent = frame["Delta"].to_numpy() == 1
    law_mean = mar.Q[level, arm]
    if scenario == study.CONTINUOUS_SCENARIO:
        # The continuous oracle is the least-squares line of the respondents' outcome on the
        # law mean (``tests/conftest.py``, ``OracleOutcomeContinuous``).  The line is affine,
        # so fitting it on the outcome scale gives the unscaled prediction directly.
        slope, intercept = np.polyfit(law_mean[respondent], frame["Y"].to_numpy()[respondent], 1)
        law_mean = intercept + slope * law_mean
    np.testing.assert_allclose(sample["qn"], law_mean, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(sample["pin"], mar.PI[level, arm])
    # Both laws fit on the [0, 1] scale, so the outcome scaler is the identity.  The qn check
    # above therefore also reads the scale R's Qbounds = c(0, 1) assumes.
    assert (result.nuisance.scaler.lower, result.nuisance.scaler.upper) == study.Q_BOUNDS
    np.testing.assert_array_equal(sample["Y"], frame["Y"])
    np.testing.assert_array_equal(sample["Delta"], frame["Delta"])
    assert list(sample.columns) == ["scenario", "replicate", "W", "A", "Y", "Delta", "qn", "pin"]
    assert set(sample["A"]) == {0.0, 1.0}
    assert (sample["pin"] > study.NUISANCE_BOUND).all()
    assert ((sample["qn"] >= 0.0) & (sample["qn"] <= 1.0)).all()
    # R's plug-in is mean(qn); it has to be the subject's published initial estimate.
    (row,) = study.cleverly_rows(frame, {"ey_obs": study.TRUTH}, scenario, 0, result=result)
    assert row["initial_estimate"] == pytest.approx(sample["qn"].mean(), abs=1e-15)


def test_the_r_adapter_pins_the_audited_settings() -> None:
    source = (study.STUDY.artifacts / "run_study.R").read_text(encoding="utf-8")
    for pinned in (
        "q_bounds <- c(0, 1)",
        "gbound <- 0.01",
        "alpha <- 0.9995",
        'binary_mar_natural_course = "binomial"',
        'continuous_mar_natural_course = "gaussian"',
        "A = rep(1, n)",
        "W = data.frame(A_original = frame$A, W = frame$W)",
        "Q = qn",
        "qn <- cbind(frame$qn, frame$qn)",
        "g1W = rep(1, n)",
        "pDelta1 = cbind(frame$pin, frame$pin)",
        "family = family",
        'fluctuation = "logistic"',
        "Qbounds = q_bounds",
        "gbound = gbound",
        "alpha = alpha",
        "cvQinit = FALSE",
        "prescreenW.g = FALSE",
        "target.gwt = FALSE",
        "B = 1",
        "evalATT = FALSE",
        "y <- plant(y, delta == 0)",
        "tmle:::.initStage1(",
        "fit$estimates$EY1$var.psi",
        "fit$estimates$EY1$CI",
        "initial_estimate = mean(frame$qn)",
    ):
        assert pinned in source, pinned
    assert "A = frame$A" not in source
    # An ordinary fit has one fold; the stacked runner's ten-fold guard must not be copied.
    assert "fold" not in source
    assert study.Q_BOUNDS == (0.0, 1.0)
    assert study.NUISANCE_BOUND == 0.01
    probe = study.STUDY.artifacts / "probe_scale_workaround.R"
    assert probe.exists()
    assert not list(study.STUDY.artifacts.glob("run_*probe*.R"))
    probe_source = probe.read_text(encoding="utf-8")
    for pinned in (
        "q_bounds <- c(0, 1)",
        "rebuild_tolerance <- 1e-12",
        "at_unplanted <- fit_at(y, a, w, delta, q, g1, p)",
        "rebuilt <= rebuild_tolerance && unplanted_point > 0",
    ):
        assert pinned in probe_source, pinned


def test_the_scale_probe_failure_refuses_publication() -> None:
    passing = pd.DataFrame(
        {
            "scenario": [study.CONTINUOUS_SCENARIO] * 3,
            "replicate": [0, 1, 2],
            "passed": [True, True, True],
        }
    )
    failures = study.scientific_failures({"scale-probe.csv": passing})
    assert list(failures) == ["scale-workaround probe"]
    assert failures["scale-workaround probe"].empty
    failing = passing.assign(passed=[True, False, True])
    assert len(study.scientific_failures({"scale-probe.csv": failing})["scale-workaround probe"])
    for short in (passing.iloc[:0], passing.iloc[1:], passing.assign(replicate=[0, 0, 1])):
        assert "scale-workaround probe coverage" in study.scientific_failures(
            {"scale-probe.csv": short}
        )
    binary = passing.assign(scenario="binary_mar_natural_course")
    assert "scale-workaround probe coverage" in study.scientific_failures(
        {"scale-probe.csv": binary}
    )


# ------------------------------------------------------------------ committed parity


def _paired_primary_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The committed primary rows of each implementation, aligned by replication."""
    rows = pd.read_csv(study.STUDY.artifact("replicates.csv.gz"))
    key = ["scenario", "replicate", "estimand"]
    subject = rows[rows["implementation"] == study.STUDY.implementation].set_index(key)
    reference = rows[rows["implementation"] == study.STUDY.reference].set_index(key)
    assert len(subject) == len(reference) == len(rows) // 2
    assert len(subject) == study.STUDY.replicates * len(study.SCENARIOS)
    subject = subject.sort_index()
    return subject, reference.loc[subject.index]


def _targeting_ratio(
    estimate: pd.Series, subject: pd.DataFrame, reference: pd.DataFrame
) -> pd.Series:
    move = (subject["estimate"] - subject["initial_estimate"]).abs()
    assert (move > 0).all(), "a replication has no targeting move"
    return (estimate - reference["estimate"]).abs() / move


def _se_relative_difference(standard_error: pd.Series, reference: pd.DataFrame) -> pd.Series:
    return (standard_error - reference["std_error"]).abs() / reference["std_error"]


def test_the_initial_estimates_are_the_same_plug_in() -> None:
    subject, reference = _paired_primary_rows()
    shared = (subject["initial_estimate"] - reference["initial_estimate"]).abs()
    assert shared.max() < INITIAL_ESTIMATE_BOUND, shared.idxmax()


def test_the_r_parity_reproduces_every_targeting_move() -> None:
    """The paired tests cannot see a missing targeting step; this ratio can.

    The oracle nuisances leave the initial plug-in unbiased, so an estimate that skipped
    targeting still passes every paired equivalence test and every truth test.  Each
    replication's difference from R is compared with the size of its own targeting move.
    """
    subject, reference = _paired_primary_rows()
    ratio = _targeting_ratio(subject["estimate"], subject, reference)
    assert ratio.max() < TARGETING_RATIO_BOUND, (ratio.idxmax(), ratio.max())


def test_a_missing_targeting_step_fails_the_parity_ratio() -> None:
    """A deliberate mutation: cleverly's reported estimate replaced by its initial plug-in."""
    subject, reference = _paired_primary_rows()
    ratio = _targeting_ratio(subject["initial_estimate"], subject, reference)
    assert (ratio >= TARGETING_RATIO_BOUND).all()
    assert ratio.min() > 0.9


def test_the_standard_errors_are_r_s_centered_rule() -> None:
    subject, reference = _paired_primary_rows()
    relative = _se_relative_difference(subject["std_error"], reference)
    assert relative.max() < SE_RELATIVE_BOUND, (relative.idxmax(), relative.max())


def test_the_uncentered_variance_rule_fails_the_standard_error_bound() -> None:
    """A deliberate mutation: the stacked study's second-moment rule, ``sqrt((n - 1) / n)``."""
    subject, reference = _paired_primary_rows()
    factor = np.sqrt((subject["n"] - 1) / subject["n"])
    relative = _se_relative_difference(subject["std_error"] * factor, reference)
    assert (relative >= SE_RELATIVE_BOUND).all(), relative.idxmin()


def test_every_continuous_fit_passed_the_scale_probe() -> None:
    """The probe's nonzero witness: R without the workaround reports a different point."""
    probe = pd.read_csv(study.STUDY.artifact("scale-probe.csv"))
    assert list(probe["replicate"]) == list(range(study.STUDY.replicates))
    assert set(probe["scenario"]) == {study.CONTINUOUS_SCENARIO}
    assert probe["passed"].astype(bool).all()
    assert probe["scale_exact"].astype(bool).all()
    assert probe["unplanted_scale_differs"].astype(bool).all()
    assert (probe[["planted_scale_lower", "planted_scale_upper"]] == [0.0, 1.0]).all().all()
    assert (probe["moved_rows_difference"] == 0).all()
    assert (probe["rebuild_difference"] <= probe["rebuild_tolerance"]).all()
    assert (probe["rebuild_tolerance"] == 1e-12).all()
    assert (probe["unplanted_point_difference"] > 0).all()
    assert study.scientific_failures({"scale-probe.csv": probe})["scale-workaround probe"].empty


def _continuous_witness() -> tuple[pd.Series, pd.Series, pd.Series]:
    """cleverly's estimate, R's planted estimate, and the scale witness, per replication."""
    subject, reference = _paired_primary_rows()
    subject = subject.xs(study.CONTINUOUS_SCENARIO, level="scenario").droplevel("estimand")
    reference = reference.xs(study.CONTINUOUS_SCENARIO, level="scenario").droplevel("estimand")
    probe = pd.read_csv(study.STUDY.artifact("scale-probe.csv")).set_index("replicate")
    witness = probe["unplanted_point_difference"]
    assert list(witness.index) == list(subject.index) == list(reference.index)
    assert (witness > 0).all()
    return subject["estimate"], reference["estimate"], witness


def test_the_r_parity_is_finer_than_the_scale_witness() -> None:
    """The targeting ratio admits a missing workaround; this comparison with the witness does not.

    In every continuous replication, cleverly's distance from R's planted fit is compared with
    the distance between R's planted and unplanted fits.
    """
    estimate, reference, witness = _continuous_witness()
    ratio = (estimate - reference).abs() / witness
    assert ratio.max() <= WITNESS_RATIO_BOUND, (ratio.idxmax(), ratio.max())


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["above", "below"])
def test_an_unplanted_scale_estimate_fails_the_witness_check(sign: float) -> None:
    """A deliberate mutation: cleverly's estimate moved by the witness, R's unplanted shift.

    The probe records the shift's size and not its sign, so both signs are substituted.
    """
    estimate, reference, witness = _continuous_witness()
    ratio = (estimate + sign * witness - reference).abs() / witness
    assert (ratio > WITNESS_RATIO_BOUND).all(), (ratio.idxmin(), ratio.min())
    assert ratio.min() > 0.9


def test_the_study_directory_is_the_registered_one() -> None:
    assert study.STUDY.artifacts == ROOT / "tests" / "canonical" / "tmle_mar_natural_course"
    assert study.STUDY.artifacts.joinpath("regenerate.py").exists()
