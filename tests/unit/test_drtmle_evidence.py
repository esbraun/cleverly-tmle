"""The registered protocol around the costly DR-TMLE evidence run."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.studies.canonical_drtmle import (
    G_BOUNDS,
    MAX_OUTER,
    N_FOLDS,
    SCORE_AUDIT_TOLERANCE,
    STUDY,
    _learners,
    _solver_passed,
    draw_scenario,
    extra_artifacts,
    fit_cleverly,
    scientific_failures,
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
    ("similar", "not_inferior", "superior", "resolved", "expected"),
    [
        (True, True, False, True, ("equivalent", True)),
        (False, True, True, True, ("superior", True)),
        (False, False, False, True, ("inconclusive", False)),
        (False, True, False, True, ("inconclusive", False)),
        # An unresolvable cell is named apart from an unsettled one. Both fail, and only one
        # of them is evidence about the subject.
        (False, False, False, False, ("underpowered", False)),
        (True, False, False, False, ("underpowered", False)),
        # Resolution is a precondition on failure, never a route to a pass: a cell that met
        # both claims is equivalent whatever its width was.
        (True, True, False, False, ("equivalent", True)),
    ],
)
def test_the_comparison_conclusion_names_each_predeclared_route(
    similar: bool,
    not_inferior: bool,
    superior: bool,
    resolved: bool,
    expected: tuple[str, bool],
) -> None:
    assert (
        comparison_conclusion(
            similar=similar,
            not_inferior=not_inferior,
            coverage_superior=superior,
            resolved=resolved,
        )
        == expected
    )


def test_the_fit_audit_judges_both_implementations_at_one_bar() -> None:
    """The score bar is the library's, and the solver flag is not published as a comparison.

    Both halves are regression tests for a published claim that was wrong.  The R runner had
    no convergence flag and wrote ``TRUE``, so ``24 failures against 0`` compared a column the
    reference could not fail; and each side applied its own score constant, so one column held
    two quantities.
    """
    n = 3000
    rows = pd.DataFrame(
        [
            {
                "implementation": implementation,
                "scenario": "both_correct",
                "replicate": 0,
                "n": n,
                "estimand": estimand,
                "std_error": std_error,
                "score_max": score_max,
                "solver_reported": implementation == STUDY.implementation,
                "solver_passed": True if implementation == STUDY.implementation else float("nan"),
                "bound_active": False,
            }
            for implementation, score_max in (
                (STUDY.implementation, 1e-11),
                (str(STUDY.reference), 5e-6),
            )
            for estimand, std_error in (("ey0", 0.01), ("ate", 0.02))
        ]
    )
    diagnostics = extra_artifacts(rows)["fit-diagnostics.csv"].set_index("implementation")

    # One threshold rule, formed from each side's own largest reported standard error.
    expected = SCORE_AUDIT_TOLERANCE * 0.02 / np.sqrt(n)
    assert diagnostics.loc[STUDY.implementation, "score_threshold"] == pytest.approx(expected)
    assert diagnostics.loc[str(STUDY.reference), "score_threshold"] == pytest.approx(expected)

    # And it separates the two fits, where the retired 1e-4 bar passed both.
    assert bool(diagnostics.loc[STUDY.implementation, "score_passed"])
    assert not bool(diagnostics.loc[str(STUDY.reference), "score_passed"])
    assert 5e-6 <= 1e-4, "the retired study bar would have passed the reference fit above"

    # The reference reports no solver flag, and none is manufactured for it.
    assert bool(diagnostics.loc[STUDY.implementation, "solver_reported"])
    assert not bool(diagnostics.loc[str(STUDY.reference), "solver_reported"])
    assert pd.isna(diagnostics.loc[str(STUDY.reference), "solver_passed"])

    failures = scientific_failures({"fit-diagnostics.csv": diagnostics.reset_index()})
    audit = next(frame for name, frame in failures.items() if name.startswith("score audit"))
    assert list(audit["implementation"]) == [str(STUDY.reference)]
    solver = next(frame for name, frame in failures.items() if name.startswith("solver health"))
    assert solver.empty, "a reference row must never reach the solver-health frame"


def test_drtmle_evidence_is_a_reporting_study() -> None:
    assert STUDY.publication_policy == "reporting"
    assert STUDY.reference == "drtmle-r"
    assert STUDY.replicates == 800
    assert STUDY.n == 3000
    assert STUDY.margins.calibration_noninferiority == 0.05


def test_the_registered_alternation_cap_reaches_the_solver() -> None:
    """``max_outer`` is the loop's cap and ``max_iter`` is not, and the record says which ran.

    The manifest published ``max_iter: 100`` as the alternation setting while the loop ran at
    50, so the check is that the *recorded* cap is the *declared* one rather than that some
    number reached the solver.
    """
    frame, _ = draw_scenario("both_correct", 200, 1)
    reduction = fit_cleverly(frame, "both_correct").repeats[0].fluctuations["mean"].reduction
    assert reduction.max_outer == MAX_OUTER
    assert reduction.rounds <= MAX_OUTER
    assert (reduction.exit_reason == "cap") == (reduction.rounds == MAX_OUTER)


# Five fits at the registered n, which the fast tier cannot afford. The cheap half of the
# same claim -- that the two rulers are applied together at all -- is a unit test in
# ``tests/unit/test_drtmle_fit.py``; this is the one that names the draws it was found on.
@pytest.mark.slow
@pytest.mark.parametrize(
    ("replicate", "solved"),
    [
        # Ruler artifacts. Each exceeded the *relative* bar the retired verdict read and sat
        # below both the loop's own absolute bar and score_check's threshold.
        (34, True),
        (759, True),
        (626, True),
        # Genuinely unsolved: worst absolute score three to four orders above the threshold.
        # Without these the change reads as "stop reporting failures" rather than "report the
        # ones that are real".
        (266, False),
        (415, False),
    ],
)
def test_the_solver_verdict_agrees_with_the_librarys_own_score_check(
    replicate: int, solved: bool
) -> None:
    """A recorded solver failure means the score check fails, and nothing weaker.

    Both directions are pinned deliberately.  Equations (9) and (10) carry covariates that
    vanish where their own nuisance is right, so a *relative* score cannot reach a tight bar
    however well the equation is solved -- and the verdict used to be read on that scale alone.
    A test that only pinned the artifacts would pass equally well if the verdict had been
    deleted, which is why the two unsolved witnesses are here.
    """
    frame, _ = draw_scenario("outcome_correct", STUDY.n, replicate)
    check = fit_cleverly(frame, "outcome_correct").diagnostics.score_equations()
    rows = [row for row in check.rows if row.kind == "fluctuation"]
    assert _solver_passed(check) is solved
    assert all(row.passed for row in rows) is solved
