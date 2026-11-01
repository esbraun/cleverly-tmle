"""The declared design of ``point-survival-hazard``.

Every number here was fixed before any registered run.

* The truth-binding gate of ``tests/unit/test_method_evidence.py`` needs the published
  ``(property, cell)`` set to equal the declared set, and each published truth to be the
  declared law's own.  A two-replication run of every fit set shows both.
* The efficiency bounds are X13's pinned exact bounds of the same parameters.
* The both-wrong control and the untargeted plug-in clear their margins.  Their standardized
  biases, in efficiency-bound standard errors at n = 2,000, were measured as a mean over eight
  draws of 20,000 rows at seeds 9,140,000 to 9,140,007 (:data:`CONTROL_LIMITS`,
  ``_x14_h_limits.py`` and ``.log`` beside the plan).  The plug-in contrast of the wrong hazard
  is zero in every sample, so its bias is the truth's own size.
* The power cell is sized on the exact law (:data:`DESIGN_POWER`).
* The comparator's reported visits, cause labels, arms and ``survtmle`` settings, transcribed
  by hand into ``tests/canonical/point_survival_hazard_runner.R``, are the Python ones.

Seeds: ``SEED`` 20261070 and ``RESAMPLING_SEED`` 2026107001, as the plan declares.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from tests.studies import canonical_point_survival_hazard as study
from tests.studies import point_survival_hazard_properties as properties
from tests.studies.canonical_point_survival import REFERENCE, REPORTED
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("point_survival_hazard_design")

RUNNER = ROOT / "tests" / "canonical" / "point_survival_hazard_runner.R"
#: A record, not a check: each control's bias over the efficiency-bound SE, and its Monte
#: Carlo SE.  The verdict reads the bias over the cell's empirical SD instead.
CONTROL_LIMITS = {"both_wrong": (1.377, 0.146), "untargeted": (5.774, 0.0)}
#: The exact-law power of the power cell.
DESIGN_POWER = 0.9999315907366597
FIT_SET_SIZES = {
    ("interval_calibration", "survival_t5"): (2_000, 9_600),
    ("interval_calibration", "competing_t4"): (2_000, 4_800),
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
    assert (study.SEED, study.RESAMPLING_SEED) == (20261070, 2026107001)
    assert record.reference == "survtmle"
    assert record.comparator_state == "compared"
    assert set(record.scenarios) == {"survival", "competing"}
    assert (study.ALGORITHM, study.MAX_ITER, study.TOL) == ("iterative", 100, 1e-10)
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES
    assert properties.EFFICIENCY_RATIO_BAND == (0.90, 1.10)
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert properties.TARGETING_DISPLACEMENT == 0.25


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
    text = RUNNER.read_text(encoding="utf-8")
    assert "reported <- list(survival = c(1, 3, 5), competing = c(2, 4))" in text
    assert REPORTED == {"survival": (1, 3, 5), "competing": (2, 4)}
    assert 'competing = c("1" = "relapse", "2" = "death")' in text
    assert "arms <- c(arm0 = 0, arm1 = 1)" in text
    assert f'reference <- "{REFERENCE}"' in text
    assert f"max_iter <- {study.SURVTMLE_MAX_ITER}" in text
    assert "gtol <- 1e-3" in text and study.SURVTMLE_GTOL == 1e-3
    assert "tol <- 1 / n" in text
    assert 'method = "hazard"' in text
    assert "list(tol = Inf)" in text
    assert "suppressWarnings" not in text
    match = re.search(r'glm\.trt = "([^"]+)"', text)
    assert match is not None and match.group(1) == "W1 + W2"


def test_the_primary_fit_is_the_paired_construction() -> None:
    """Two replications: estimands, rows and a nonzero targeting step on both scenarios."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phase = study.draw_and_fit(replicates=2, n=2_000, n_jobs=1)
    assert phase.failures is None
    samples, rows = phase.samples, phase.rows
    assert samples is not None
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


def test_a_pinned_bound_matches_the_spread_of_a_large_hazard_fit() -> None:
    """The X13 enumeration and this estimator's curve agree, which neither could show alone."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = properties.draw("survival_t5", 20_000, 23)
        result = properties.fit_label("survival_t5", frame)
    curve = result[properties.ESTIMAND["survival_t5"]].influence_curve
    assert float(np.std(curve)) == pytest.approx(properties.EFFICIENCY_SD["survival_t5"], rel=0.05)


def test_the_correct_one_cause_hazard_learner_leaves_targeting_work_to_do() -> None:
    """A saturated hazard fitted in sample has a zero initial score; the additive one does not.

    The calibration cells are evidence for targeting only if targeting moves their fits.
    """
    frame = properties.draw("survival_t5", 2_000, 99_000_123)
    result = properties.fit_label("survival_t5", frame)
    records = {id(fit.hazard): fit.hazard for fit in result.fits.values()}
    assert all(record.converged for record in records.values())
    assert max(float(np.max(np.abs(r.score_initial))) for r in records.values()) > 1e-4
    saturated = properties.learners("both_correct", causes=2)["outcome_learner"]
    assert type(saturated).__name__ == "CellMeans"
