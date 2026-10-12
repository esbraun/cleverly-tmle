"""The declared design of ``point-survival-curve-crossfit``.

Every number here was fixed before any registered run.  A two-replication run of every fit set
shows that the published ``(property, cell)`` set equals the declared one and that each published
truth is the declared law's own.  The efficiency bounds are X13's pinned exact bounds.

Seeds: ``SEED`` 20261072 and ``RESAMPLING_SEED`` 2026107201, as the plan declares.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.studies import canonical_point_survival_curve_crossfit as study
from tests.studies import point_survival_curve_crossfit_properties as properties
from tests.studies.evidence.registry import Margins, registered

pytestmark = pytest.mark.xdist_group("point_survival_curve_crossfit_design")

FIT_SET_SIZES = {
    ("interval_calibration", "survival_t5"): (2_000, 9_600),
    ("interval_calibration", "survival_t5_per_horizon"): (2_000, 1_600),
    ("interval_calibration", "competing_t4"): (2_000, 1_600),
    ("double_robustness", "both_correct"): (2_000, 1_200),
    ("double_robustness", "outcome_correct"): (2_000, 1_200),
    ("double_robustness", "mechanism_correct"): (2_000, 1_200),
    ("double_robustness", "both_wrong"): (2_000, 1_200),
}


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert record.reference is None
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N, study.N_FOLDS) == (1_600, 2_000, 5)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261072, 2026107201)
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


def test_the_primary_fit_is_cross_fitted() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.fit(study.LAW.draw(400, 1))
    assert result.folds.n_folds == study.N_FOLDS
    assert set(study.ESTIMANDS["survival"]) <= set(result.estimates)
