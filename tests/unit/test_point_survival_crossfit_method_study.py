"""The declared design of ``point-treatment-survival-crossfit``.

Every number here was fixed before any registered run.  The declared cells are the cells a
two-replication run publishes, the both-wrong control clears its margin
(:data:`CONTROL_LIMITS`, eight draws of 8,000 rows at five folds), and the failure-only probe
found no failure on streams 0 to 19 of every fit set (:data:`FAILURE_PROBE`).  A replication
that raises is never redrawn, and the shared harness refuses a cell that lost one.

Seeds: ``SEED`` 20261061 and ``RESAMPLING_SEED`` 2026106101 are new.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.studies import canonical_point_survival_crossfit as study
from tests.studies import point_survival_crossfit_properties as properties
from tests.studies.evidence.registry import Margins, registered

pytestmark = pytest.mark.xdist_group("point_survival_crossfit_design")

FAILURE_PROBE = dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0)
CONTROL_LIMITS = {"both_wrong": (1.476, 0.191)}
FIT_SET_SIZES = {
    ("double_robustness", "both_correct"): (2_000, 1_200),
    ("double_robustness", "outcome_correct"): (2_000, 1_200),
    ("double_robustness", "mechanism_correct"): (2_000, 1_200),
    ("double_robustness", "both_wrong"): (2_000, 1_200),
    ("interval_calibration", "survival_t5"): (2_000, 9_600),
    ("interval_calibration", "three_arm_t3"): (2_000, 1_600),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.reference is None
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N, study.N_FOLDS) == (1_600, 2_000, 5)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261061, 2026106101)
    assert {
        (family, label): (n, replicates)
        for family, label, _, n, replicates, _ in properties.FIT_SETS
    } == FIT_SET_SIZES


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties.generate_property_rows(n_jobs=1, budget=2)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
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


def test_the_study_draws_from_its_own_seeds() -> None:
    mine = {study.STUDY.seed, study.STUDY.resampling_seed}
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not mine & others


def test_the_both_wrong_control_clears_its_margin() -> None:
    bias, monte_carlo = CONTROL_LIMITS["both_wrong"]
    floor = Margins().standardized_bias + 5.0 / np.sqrt(properties.DOUBLE_ROBUST_REPLICATES)
    assert bias - 3.0 * monte_carlo > floor


def test_the_primary_rows_are_cross_fitted() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame, _ = study.draw_scenario("survival", 2_000, 0)
        result = study.fit("survival", frame)
        rows = study.draw_and_fit(replicates=2, n=2_000, n_jobs=1)
    assert result.folds.n_folds == 5
    for scenario, names in study.ESTIMANDS.items():
        assert set(rows.loc[rows["scenario"] == scenario, "estimand"]) == set(names)


#: The page's and the owner row's reading of the three-arm red cell, by ``(n, folds)``.
THREE_ARM_READING = {
    (2_000, 1): 1.0003,
    (2_000, 5): 1.082,
    (2_000, 10): 1.055,
    (8_000, 5): 1.0085,
}


def test_the_three_arm_diagnostic_reproduces_the_quoted_reading() -> None:
    """The red three-arm calibration cell is a finite-sample cost of cross-fitting.

    ``tests/diagnostics/x13_crossfit_three_arm/`` refits the cell's configuration on fresh
    draws.  The mean reported standard error reaches the exact bound in sample, exceeds it at
    five folds, falls at ten folds, and nearly reaches it at n = 8,000.  The rows rebuild the
    numbers the page quotes, at the precision it prints them.
    """
    import pandas as pd

    from tests.diagnostics.x13_crossfit_three_arm import run as diagnostic

    rows = pd.read_csv(diagnostic.HERE / "rows.csv.gz", float_precision="round_trip")
    assert len(rows) == sum(replications for _, _, replications in diagnostic.DESIGNS)
    reading = diagnostic.reading(rows).set_index(["n", "folds"])["reported_over_bound"]
    for design, quoted in THREE_ARM_READING.items():
        digits = len(str(quoted).split(".")[1])
        assert round(float(reading[design]), digits) == quoted, design
    assert reading[(2_000, 5)] > reading[(2_000, 10)] > reading[(8_000, 5)] > reading[(2_000, 1)]
