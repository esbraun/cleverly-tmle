"""The declared design of ``point-survival-curve``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A three-replication run of every fit set shows both.
* The efficiency bounds are exact: X13's pinned bounds, and three new ones recomputed here.
* The both-wrong control and the untargeted plug-in clear their margins.  Their standardized
  biases under the one-step walk, in efficiency-bound standard errors at n = 2,000, were
  measured as a mean over eight draws of 20,000 rows at seeds 9,140,000 to 9,140,007
  (:data:`CONTROL_LIMITS`, ``_x14_h_limits.py`` and ``.log`` beside the plan).
* The power cell is sized on the exact law (:data:`DESIGN_POWER`).
* The comparator's grid, arms and updater settings, transcribed by hand into
  ``tests/canonical/tmle3_survival/run_study.R``, are the Python ones, and the Dockerfile pins
  the declared ``glmnet``.

Seeds: ``SEED`` 20261071 and ``RESAMPLING_SEED`` 2026107101, as the plan declares.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.studies import canonical_point_survival_curve as study
from tests.studies import point_survival_curve_properties as properties
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("point_survival_curve_design")

CANONICAL = ROOT / "tests" / "canonical" / "tmle3_survival"
#: A record, not a check: each control's bias over the efficiency-bound SE, and its Monte
#: Carlo SE.  The verdict reads the bias over the cell's empirical SD instead.
CONTROL_LIMITS = {"both_wrong": (1.379, 0.146), "untargeted": (5.774, 0.0)}
#: The exact-law power of the power cell.
DESIGN_POWER = 0.9999315907366597
FIT_SET_SIZES = {
    ("interval_calibration", "survival_t5"): (2_000, 9_600),
    ("interval_calibration", "competing_t4"): (2_000, 4_800),
    ("interval_calibration", "three_arm"): (2_000, 4_800),
    ("interval_calibration", "weighted_t5"): (2_000, 4_800),
    ("interval_calibration", "weighted_t5_per_horizon"): (2_000, 4_800),
    ("interval_calibration", "isotonic_t2"): (2_000, 4_800),
    ("clustered_inference", "clustered_t5"): (2_000, 4_800),
    ("clustered_inference", "clustered_t5_per_horizon"): (2_000, 4_800),
    ("double_robustness", "both_correct"): (2_000, 1_200),
    ("double_robustness", "outcome_correct"): (2_000, 1_200),
    ("double_robustness", "mechanism_correct"): (2_000, 1_200),
    ("double_robustness", "both_wrong"): (2_000, 1_200),
    ("targeting_necessity", "wrong_hazard"): (2_000, 1_200),
    ("root_n_and_efficiency", "n_1000"): (1_000, 800),
    ("root_n_and_efficiency", "n_2000"): (2_000, 800),
    ("root_n_and_efficiency", "n_8000"): (8_000, 800),
    ("power", "alternative"): (2_000, 800),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_600, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261071, 2026107101)
    assert record.pending_reference == "tmle3"
    assert record.comparator_state == "pending"
    assert set(record.scenarios) == {"survival"}
    assert len(record.scenarios["survival"]) == 15
    assert (study.ALGORITHM, study.STEP_SIZE, study.TOL) == ("one_step", 1e-3, 1e-10)
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert properties.TARGETING_DISPLACEMENT == 0.25
    assert properties.CLUSTER_N // properties.CLUSTER_SIZE == 25


def test_the_new_efficiency_bounds_are_exact() -> None:
    for label in ("rule_t3", "policy_t3", "isotonic_t2", "weighted_t5_per_horizon"):
        assert properties.efficiency_bound(label) == pytest.approx(
            properties.EFFICIENCY_SD[label], rel=1e-12
        )


def test_the_rule_and_the_policy_are_different_parameters() -> None:
    rule = properties.TRUTHS["rule_t3"][properties.ESTIMAND["rule_t3"]]
    policy = properties.TRUTHS["policy_t3"][properties.ESTIMAND["policy_t3"]]
    assert abs(rule - policy) > 1e-3


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # One replicate per fit set suffices: the cells a run publishes are fixed by the fit set,
        # and test_method_evidence.py checks the summary, which needs two, on the committed rows.
        rows = properties.generate_property_rows(n_jobs=1, budget=1)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert int(rows["failed_replicates"].max()) == 0
    for (family, name), cell in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        np.testing.assert_allclose(
            selected["truth"], cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=1e-12
        )
        assert set(selected["n"]) == {cell.n}


def test_no_two_families_share_a_declared_stream() -> None:
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    shared = {key: families for key, families in streams.items() if len(families) > 1}
    # The joint bands read the survival calibration fits on purpose.
    assert set(shared) <= {
        (cell.dgp.name, cell.seed)
        for cell in properties.declared_cells()
        if cell.property == "simultaneous_coverage"
    }


def test_the_study_draws_from_its_own_seeds() -> None:
    mine = {study.STUDY.seed, study.STUDY.resampling_seed}
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not mine & others


def _floor(replicates: int) -> float:
    return Margins().standardized_bias + 5.0 / np.sqrt(replicates)


def test_the_controls_clear_their_margins() -> None:
    for name, replicates in (
        ("both_wrong", properties.DOUBLE_ROBUST_REPLICATES),
        ("untargeted", properties.NECESSITY_REPLICATES),
    ):
        bias, monte_carlo = CONTROL_LIMITS[name]
        assert bias - 3.0 * monte_carlo > _floor(replicates), name
    truth = properties.TRUTHS["survival_t5"][properties.ESTIMAND["survival_t5"]]
    standard_error = properties.EFFICIENCY_SD["survival_t5"] / np.sqrt(properties.NECESSITY_N)
    assert CONTROL_LIMITS["untargeted"][0] == pytest.approx(abs(truth) / standard_error, abs=1e-3)


def test_the_power_cell_reaches_its_floor_on_the_exact_law() -> None:
    contrast = properties.TRUTHS["survival_t5"][properties.ESTIMAND["survival_t5"]]
    power = design_power(contrast, properties.EFFICIENCY_SD["survival_t5"], properties.POWER_N)
    assert power == pytest.approx(DESIGN_POWER, abs=1e-6)
    assert power >= 0.80
    assert power_cell_pass_probability(power, properties.POWER_REPLICATES) >= 0.99


def test_the_runner_transcribes_the_python_design() -> None:
    text = (CANONICAL / "run_study.R").read_text(encoding="utf-8")
    assert f"grid <- 1:{max(study.HORIZONS)}" in text
    assert "arms <- c(arm0 = 0, arm1 = 1)" in text
    assert f'reference <- "{study.REFERENCE}"' in text
    assert f"maxit <- {study.TMLE3_MAXIT}" in text
    assert f"delta_epsilon <- {study.TMLE3_DELTA_EPSILON}" in text
    assert 'convergence_type = "sample_size"' in text
    assert "cvtmle = FALSE" in text
    assert 'fit_method = "l2"' in text
    assert "if (length(fit_lambda) == 1)" in text
    assert "lambda.min.ratio <- 0.8 * lambda_guess / fit_lambda[1]" in text
    assert "lambda_min_ratio <-" not in text
    assert "suppressWarnings" not in text
    dockerfile = (CANONICAL / "Dockerfile").read_text(encoding="utf-8")
    assert f"ARG GLMNET_VERSION={study.GLMNET_VERSION}" in dockerfile
    assert f"ARG GLMNET_SHA256={study.GLMNET_TARBALL_SHA256}" in dockerfile
    assert f"ARG TMLE3_COMMIT={study.REFERENCE_METADATA['tmle3_commit']}" in dockerfile
    assert f"ARG SL3_COMMIT={study.REFERENCE_METADATA['sl3_commit']}" in dockerfile


def test_the_primary_fit_is_the_paired_construction() -> None:
    """Two replications: every reported name, a monotone curve and a nonzero walk."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phase = study.draw_and_fit(replicates=2, n=2_000, n_jobs=1)
    assert phase.failures is None
    samples, truths, rows = phase.samples, phase.truths, phase.rows
    assert samples is not None and truths is not None
    assert set(rows["estimand"]) == set(study.ESTIMANDS["survival"])
    assert set(truths["estimand"]) == set(study.ESTIMANDS["survival"])
    assert np.any(np.abs(rows["estimate"] - rows["initial_estimate"]) > 0.0)
    for replicate in (0, 1):
        for arm in ("arm0", "arm1"):
            selected = rows.loc[
                (rows["replicate"] == replicate)
                & rows["estimand"].str.startswith(f"risk_regimen[{arm} ")
            ]
            curve = selected.set_index("estimand").loc[
                [f"risk_regimen[{arm} @ t={t}]" for t in study.HORIZONS], "estimate"
            ]
            assert np.all(np.diff(curve.to_numpy()) >= 0.0)
    assert list(samples[["replicate"]].drop_duplicates()["replicate"]) == [0, 1]
