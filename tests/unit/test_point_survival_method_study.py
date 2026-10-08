"""The declared design of ``point-treatment-survival``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The pinned efficiency bounds are the exact standard deviations of the efficient influence
  function, recomputed here by enumeration over each law's support.
* The both-wrong control clears its margin.  Its standardized bias, in efficiency-bound
  standard errors at n = 2,000, was measured as a mean over eight draws of 8,000 rows
  (:data:`CONTROL_LIMITS`, ``_x13_A_limits.log`` beside the plan).
* The IID control of the clustered cell understates the standard error by construction.  The
  clusters share ``W``, so the exact IID-to-cluster SE ratio is
  :math:`\\sqrt{V / (V + (m - 1) B)}` with ``V`` the bound's variance and ``B`` the variance of
  :math:`E[D \\mid W]`; at ``m = 80`` it is below the control's 0.80 ceiling
  (:data:`IID_SE_RATIO`).
* The power cell is sized on the exact law (:data:`DESIGN_POWER`).
* The comparator's reported visits, cause labels and arms, transcribed by hand into
  ``tests/canonical/point_survival_runner.R``, are the Python ones.

Failure rule: a replication that raises is never redrawn.  ``failure_probe`` found no failure
on streams 0 to 19 of every fit set (:data:`FAILURE_PROBE`).  The shared harness refuses a cell
that lost a replication, so a failure in the run stops the run.

Seeds: ``SEED`` 20261060 and ``RESAMPLING_SEED`` 2026106001 are new.  The plan named
20261010 to 20261012, which ``canonical-multi-arm-mar-drtmle``,
``full-refit-bootstrap-and-derived-contrasts`` and ``clustered-unequal-cvtmle`` already use.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest

from tests.studies import canonical_point_survival as study
from tests.studies import point_survival_common as common
from tests.studies import point_survival_properties as properties
from tests.studies.evidence.property_verdicts import (
    CLUSTER_ROBUST_CONTROL_SE_CEILING,
    design_power,
    power_cell_pass_probability,
)
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("point_survival_design")

RUNNER = ROOT / "tests" / "canonical" / "point_survival_runner.R"
#: A record, not a check: the failure-only probe on streams 0 to 19 found no failure.
FAILURE_PROBE = dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0)
#: A record, not a check: the both-wrong control's standardized bias and its Monte Carlo SE.
CONTROL_LIMITS = {"both_wrong": (1.121, 0.173)}
#: The exact IID-to-cluster SE ratio of the clustered cell.
IID_SE_RATIO = 0.7328152080303988
#: The exact-law power of the power cell.
DESIGN_POWER = 0.9999999997272832
FIT_SET_SIZES = {
    ("double_robustness", "both_correct"): (2_000, 1_200),
    ("double_robustness", "outcome_correct"): (2_000, 1_200),
    ("double_robustness", "mechanism_correct"): (2_000, 1_200),
    ("double_robustness", "both_wrong"): (2_000, 1_200),
    ("root_n_and_efficiency", "n_1000"): (1_000, 800),
    ("root_n_and_efficiency", "n_2000"): (2_000, 800),
    ("root_n_and_efficiency", "n_8000"): (8_000, 800),
    ("interval_calibration", "survival_t5"): (2_000, 9_600),
    ("interval_calibration", "competing_t4"): (2_000, 1_600),
    ("interval_calibration", "three_arm_t3"): (2_000, 1_600),
    ("interval_calibration", "continuous_t4"): (2_000, 1_600),
    ("interval_calibration", "end_of_study"): (2_000, 1_600),
    ("interval_calibration", "weighted_t5"): (2_000, 1_600),
    ("power", "alternative"): (4_000, 800),
    ("clustered_inference", "clustered_t5"): (2_000, 1_600),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_600, 2_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261060, 2026106001)
    assert record.reference == "survtmle"
    assert set(record.scenarios) == {"survival", "competing"}
    assert study.REPORTED == {"survival": (1, 3, 5), "competing": (2, 4)}
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert (properties.CLUSTER_SIZE, properties.CLUSTER_N) == (80, 2_000)


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties.generate_property_rows(n_jobs=1, budget=2)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert int(rows["failed_replicates"].max()) == 0
    for (family, name), cell in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        np.testing.assert_allclose(
            selected["truth"], cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=1e-12
        )
        assert set(selected["n"]) == {cell.n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = properties.summarize_properties(rows)
    assert {(row.property, row.cell) for row in summary.itertuples()} == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }


def test_no_two_families_share_a_declared_stream() -> None:
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    shared = {key: families for key, families in streams.items() if len(families) > 1}
    # The joint band reads the survival calibration fits on purpose.
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


def test_the_pinned_efficiency_bounds_are_the_exact_ones() -> None:
    for label, value in properties.EFFICIENCY_SD.items():
        assert properties.efficiency_bound(label) == pytest.approx(value, rel=1e-12)


def test_the_primary_truths_are_the_exact_sums() -> None:
    for scenario, names in study.ESTIMANDS.items():
        assert set(study.TRUTH[scenario]) == set(names)
    truth = study.TRUTH["survival"]
    assert truth["ate_regimen[arm1 vs arm0 @ t=5]"] == pytest.approx(
        truth["risk_regimen[arm1 @ t=5]"] - truth["risk_regimen[arm0 @ t=5]"], abs=1e-15
    )
    for name in study.ESTIMANDS["survival"]:
        if name.startswith("ate_regimen"):
            assert abs(truth[name]) > 0.01, name


def test_the_rmst_truth_is_the_sum_of_the_risks() -> None:
    truths = properties.TRUTHS["rmst_5"]
    risks = [truths[common.contrast_name("arm1", "arm0", t)] for t in range(1, 5)]
    assert truths["rmst_regimen[arm1 vs arm0 @ t=5]"] == pytest.approx(-sum(risks), abs=1e-15)


def _floor(replicates: int) -> float:
    return Margins().standardized_bias + 5.0 / np.sqrt(replicates)


def test_the_controls_clear_their_margins() -> None:
    bias, monte_carlo = CONTROL_LIMITS["both_wrong"]
    assert bias - 3.0 * monte_carlo > _floor(properties.DOUBLE_ROBUST_REPLICATES)
    total = common.efficiency_sd(common.SURVIVAL, [common.Target(1.0, "static", 0, 5)])
    between = common.efficiency_sd(
        common.SURVIVAL, [common.Target(1.0, "static", 0, 5)], between=True
    )
    ratio = math.sqrt(total**2 / (total**2 + (properties.CLUSTER_SIZE - 1) * between**2))
    assert ratio == pytest.approx(IID_SE_RATIO, rel=1e-12)
    assert ratio < CLUSTER_ROBUST_CONTROL_SE_CEILING - 0.05


def test_the_power_cell_reaches_its_floor_on_the_exact_law() -> None:
    contrast = properties.TRUTHS["survival_t5"][properties.ESTIMAND["survival_t5"]]
    power = design_power(contrast, properties.EFFICIENCY_SD["survival_t5"], properties.POWER_N)
    assert power == pytest.approx(DESIGN_POWER, abs=1e-6)
    assert power_cell_pass_probability(power, properties.POWER_REPLICATES) >= 0.99


def test_the_runner_transcribes_the_python_design() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "reported <- list(survival = c(1, 3, 5), competing = c(2, 4))" in text
    assert 'competing = c("1" = "relapse", "2" = "death")' in text
    assert "arms <- c(arm0 = 0, arm1 = 1)" in text
    assert f'reference <- "{study.REFERENCE}"' in text
    assert common.CAUSES == {1: "relapse", 2: "death"}
    match = re.search(r'glm\.trt = "([^"]+)", glm\.ftime = "([^"]+)"', text)
    assert match is not None
    assert match.groups() == ("W1 + W2", "trt*(W1 + W2)")


def test_the_primary_fit_is_the_paired_construction() -> None:
    """Two replications: estimands, rows and a nonzero targeting step on both scenarios."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, _, rows = study.draw_and_fit(replicates=2, n=2_000, n_jobs=1)
    assert set(rows["scenario"]) == {"survival", "competing"}
    for scenario, names in study.ESTIMANDS.items():
        selected = rows.loc[rows["scenario"] == scenario]
        assert set(selected["estimand"]) == set(names)
    assert np.all(np.abs(rows["estimate"] - rows["initial_estimate"]) > 0.0)
    order = samples[["replicate", "scenario"]].drop_duplicates()
    assert list(order.itertuples(index=False, name=None)) == [
        (0, "survival"),
        (0, "competing"),
        (1, "survival"),
        (1, "competing"),
    ]


@pytest.mark.parametrize("label", ["survival_t5", "rmst_5", "continuous_t4", "weighted_t5"])
def test_a_pinned_bound_matches_the_spread_of_a_large_fit(label: str) -> None:
    """The enumeration and the estimator's curve agree, which neither could show alone."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = properties.draw(label, 20_000, 23)
        result = properties.fit_label(label, frame)
    spread = float(np.std(properties.estimate_of(result, label).influence_curve))
    assert spread == pytest.approx(properties.EFFICIENCY_SD[label], rel=0.05)
