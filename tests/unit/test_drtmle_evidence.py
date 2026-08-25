"""The registered protocol around the costly DR-TMLE evidence run."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.studies.canonical_drtmle import (
    G_BOUNDS,
    N_FOLDS,
    STUDY,
    _learners,
    _solver_passed,
    draw_scenario,
    fit_cleverly,
    truth,
)
from tests.studies.drtmle_properties import Cell, _property_replicate
from tests.studies.evidence.comparison import comparison_conclusion


def test_the_paper_law_has_an_independent_quadrature_truth() -> None:
    target = truth()
    assert target["ey0"] == pytest.approx(0.5, abs=1e-13)
    assert target["ate"] == pytest.approx(target["ey1"] - target["ey0"], abs=1e-15)
    assert target["ate"] == pytest.approx(0.03802642731065542, abs=1e-14)


def test_the_sample_carries_one_deterministic_treatment_stratified_fold_vector() -> None:
    first, target = draw_scenario("both_correct", 1000, 3)
    second, repeated_target = draw_scenario("both_correct", 1000, 3)
    assert np.array_equal(first["fold"], second["fold"])
    assert target == repeated_target == truth()
    assert set(first["fold"]) == set(range(N_FOLDS))
    by_fold = first.groupby("fold")["A"].agg(["min", "max"])
    assert (by_fold["min"] == 0.0).all()
    assert (by_fold["max"] == 1.0).all()


def test_the_misspecified_glms_drop_only_the_interaction() -> None:
    outcome, treatment = _learners("both_wrong")
    assert outcome.columns == (0, 1, 2)
    assert treatment.columns == (0, 1)
    outcome, treatment = _learners("both_correct")
    assert outcome.columns is None
    assert treatment.columns is None


def test_the_both_wrong_property_control_has_an_independent_seed_stream() -> None:
    row = _property_replicate(
        (Cell("double_robustness", "both_wrong", "both_wrong", 100, 1, 40_000, "control"), 0)
    )
    assert row["cell"] == "both_wrong"
    assert row["role"] == "control"


def test_solver_health_reads_only_explicit_fluctuation_failures() -> None:
    check = SimpleNamespace(
        rows=(
            SimpleNamespace(kind="fluctuation", failure=""),
            SimpleNamespace(kind="identity", failure="max_iter_reached"),
        )
    )
    assert _solver_passed(check)
    failed = SimpleNamespace(
        rows=(SimpleNamespace(kind="fluctuation", failure="max_iter_reached"),)
    )
    assert not _solver_passed(failed)


def test_the_local_smoke_fit_uses_the_declared_default_construction() -> None:
    frame, _ = draw_scenario("both_correct", 200, 1)
    result = fit_cleverly(frame, "both_correct")
    assert result.diagnostics.score_equations().passed
    raw = result.repeats[0].nuisance.propensity.values
    assert np.array_equal(raw, result.repeats[0].nuisance.bounded_propensity(G_BOUNDS))
    assert set(result.estimates) == {"ey0", "ey1", "ate"}


@pytest.mark.parametrize(
    ("similar", "not_inferior", "superior", "expected"),
    [
        (True, True, False, ("equivalent", True)),
        (False, True, True, ("superior", True)),
        (False, False, False, ("inconclusive", False)),
        (False, True, False, ("inconclusive", False)),
    ],
)
def test_the_comparison_conclusion_names_each_predeclared_route(
    similar: bool,
    not_inferior: bool,
    superior: bool,
    expected: tuple[str, bool],
) -> None:
    assert comparison_conclusion(
        similar=similar,
        not_inferior=not_inferior,
        coverage_superior=superior,
    ) == expected


def test_drtmle_evidence_is_a_reporting_study() -> None:
    assert STUDY.publication_policy == "reporting"
    assert STUDY.reference == "drtmle-r"
    assert STUDY.replicates == 800
    assert STUDY.n == 3000
