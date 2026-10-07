"""The declared design of ``point-treatment-survival-policies``.

Every number here was fixed before any registered run.  The declared cells are the cells a
two-replication run publishes; the pinned efficiency bounds are the exact sums and agree with
the influence-curve spread of a large saturated fit; the pairing columns are the law's own
policy and ratios; the comparator's plans and copies are the Python ones; and the failure-only
probe found no failure on streams 0 to 19 of every fit set (:data:`FAILURE_PROBE`).  A
replication that raises is never redrawn, and the shared harness refuses a cell that lost one.

Seeds: ``SEED`` 20261062 and ``RESAMPLING_SEED`` 2026106201 are new.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cleverly.datasets import survival_point as law_module
from tests.studies import canonical_point_survival_policies as study
from tests.studies import point_survival_common as common
from tests.studies import point_survival_policies_properties as properties
from tests.studies.evidence.registry import ROOT, Margins, registered

pytestmark = pytest.mark.xdist_group("point_survival_policies_design")

RUNNER = ROOT / "tests" / "canonical" / "point_survival_policies_runner.R"
FAILURE_PROBE = dict.fromkeys((f"{family}/{label}" for family, label, *_ in properties.FIT_SETS), 0)


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.reference == "lmtp"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N, study.COPIES) == (1_600, 2_000, 4)
    assert (study.SEED, study.RESAMPLING_SEED) == (20261062, 2026106201)
    assert study.REPORTED == (1, 3, 5)
    assert (properties.CALIBRATION_N, properties.CALIBRATION_REPLICATES) == (2_000, 1_600)


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties.generate_property_rows(n_jobs=1, budget=2)
        summary = properties.summarize_properties(rows)
    assert {tuple(key) for key in rows.groupby(["property", "cell"]).groups} == set(declared)
    for (family, name), cell in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        np.testing.assert_allclose(
            selected["truth"], cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=1e-12
        )
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


def test_the_contrasts_are_parameters_the_natural_course_does_not_reach() -> None:
    for scenario, plan in study.PLAN.items():
        name = common.contrast_name(plan, study.REFERENCE, 5)
        assert abs(study.TRUTH[scenario][name]) > 0.02, scenario


def test_the_pinned_bounds_are_exact_and_match_a_large_fit() -> None:
    for label, value in properties.EFFICIENCY_SD.items():
        assert properties.efficiency_bound(label) == pytest.approx(value, rel=1e-12)
        scenario = study.SCENARIOS[properties.SCENARIO_OF[label]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = properties.fit_label(label, scenario.draw(20_000, 17))
        spread = float(np.std(result[properties.ESTIMAND[label]].influence_curve))
        assert spread == pytest.approx(value, rel=0.05), label


def test_the_pairing_columns_are_the_laws_policy_and_ratios() -> None:
    frame, _ = study.draw_scenario("policy", 2_000, 0)
    columns = study.pairing_columns("policy", frame)
    q = common.policy_probability(frame["W1"], frame["W2"])
    copies = np.column_stack([columns[f"shift__policy__{c}"] for c in range(1, study.COPIES + 1)])
    np.testing.assert_allclose(copies.mean(axis=1), q, atol=1e-12)
    g = law_module.treatment_probabilities(frame["W1"], frame["W2"])[:, 1]
    a = frame["A"].to_numpy()
    expected = np.where(a == 1, q / g, (1 - q) / (1 - g))
    np.testing.assert_allclose(columns["ratio__policy__1"], expected, rtol=1e-12)
    assert abs(columns["ratio__policy__1"].mean() - 1.0) < 0.1
    retained = frame["C3"].to_numpy() == 1.0
    assert np.all(columns["ratio__policy__3"][~retained] == 0.0)
    assert np.all(columns["ratio__policy__3"][retained] > 1.0)
    dose, _ = study.draw_scenario("mtp", 2_000, 0)
    moved = study.pairing_columns("mtp", dose)
    np.testing.assert_array_equal(moved["shift__mtp__1"], common.minus_one(dose["D"]))
    assert np.all(moved["ratio__mtp__1"][dose["D"].to_numpy() == 5.0] == 0.0)


def test_the_runner_transcribes_the_python_design() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'plans <- list(policy = c("natural", "policy"), mtp = c("natural", "mtp"))' in text
    assert "copies <- c(natural = 1L, policy = 4L, mtp = 1L)" in text
    assert "reported <- c(1L, 3L, 5L)" in text
    assert f'reference <- "{study.REFERENCE}"' in text
    assert {label for plans in study.REGIMENS.values() for label in plans} == {
        "natural",
        "policy",
        "mtp",
    }
