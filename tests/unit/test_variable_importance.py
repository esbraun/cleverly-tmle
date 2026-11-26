"""Exposure-wise orchestration and corrected multiplicity inference."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import sklearn.linear_model

from cleverly import Replayability, StudyProtocol, variable_importance, variable_importance_plan
from cleverly._summary import vocabulary
from cleverly.assessment import (
    AssessmentStatus,
    CandidateReports,
    DiagnosticsFacade,
    replayability,
)
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.variable_importance import KEEP_FITS_REFUSAL, _bh_adjust


def test_bh_adjustment_is_monotone_in_rank_and_uses_both_tails() -> None:
    adjusted = _bh_adjust((0.04, 0.01, 0.03))
    np.testing.assert_allclose(adjusted, (0.04, 0.03, 0.04))


def test_each_candidate_gets_its_own_declared_fit() -> None:
    rng = np.random.default_rng(5)
    n = 180
    w = rng.normal(size=n)
    x1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-w)))
    x2 = rng.binomial(1, 0.5, size=n)
    y = 1.2 * x1 + 0.2 * w + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "W": w})
    result = variable_importance(
        frame,
        outcome="Y",
        candidates=("X1", "X2"),
        covariates=("W",),
        estimator=TMLE(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=1,
        ),
    )
    assert set(result.fits) == {"X1", "X2"}
    entries = {entry.candidate: entry for entry in result}
    assert entries["X1"].adjustment_set == ("W", "X2")
    assert entries["X2"].adjustment_set == ("W", "X1")
    assert entries["X1"].estimate.pvalue <= entries["X1"].adjusted_pvalue
    assert entries["X1"].estimate.psi > entries["X2"].estimate.psi


def test_the_summary_preserves_a_polars_callers_backend() -> None:
    rng = np.random.default_rng(9)
    n = 80
    frame = pl.DataFrame(
        {
            "Y": rng.normal(size=n),
            "X1": rng.binomial(1, 0.5, size=n),
            "X2": rng.binomial(1, 0.5, size=n),
            "W": rng.normal(size=n),
        }
    )
    result = variable_importance(
        frame,
        outcome="Y",
        candidates=("X1", "X2"),
        covariates=("W",),
        estimator=TMLE(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=4,
        ),
    )
    assert isinstance(result.to_frame(), pl.DataFrame)


def test_a_candidate_with_missing_values_is_refused_with_its_own_limit() -> None:
    """variable_importance takes no treatment_delta=, so the refusal must not name it."""
    import pytest

    from cleverly.exceptions import DataError

    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "W": rng.normal(size=n),
            "A": rng.integers(0, 2, n).astype(float),
            "B": rng.integers(0, 2, n).astype(float),
        }
    )
    frame["Y"] = frame["A"] + frame["W"] + rng.normal(size=n)
    frame.loc[:5, "A"] = np.nan
    with pytest.raises(DataError, match="variable_importance takes no missing treatment") as info:
        variable_importance(frame, outcome="Y", candidates=["A", "B"], covariates=["W"])
    assert "treatment_delta" not in str(info.value)


# ------------------------------------------------------------------ assessment (X104)


def _assessed_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    w = rng.normal(size=n)
    x1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-w)))
    x2 = rng.binomial(1, 0.5, size=n)
    y = x1 + 0.5 * w + rng.normal(size=n)
    return pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "W": w})


def _assessed(**settings: Any) -> Any:
    options: dict[str, Any] = {"covariates": ("W",), **settings}
    return variable_importance(
        _assessed_frame(),
        outcome="Y",
        candidates=("X1", "X2"),
        estimator=TMLE(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        ),
        **options,
    )


@pytest.fixture(scope="module")
def clean() -> Any:
    """Each candidate adjusts for ``W`` and the other candidate, so every bound runs."""
    return _assessed()


@pytest.fixture(scope="module")
def mixed() -> Any:
    """``X2`` adjusts for ``W`` alone, so its benchmark has no proper subset to drop."""
    return _assessed(covariates=("W", "X2"), adjust_for_other_candidates=False)


def _frames_equal(left: Any, right: Any) -> None:
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


def test_a_composed_row_carries_the_refusing_candidates_record(mixed: Any) -> None:
    """The composition mask: X1's benchmark runs, X2's refuses, so the run's row refuses.

    The row is X2's own row with its condition named after X2, and the call raises it.
    """
    own = mixed.fits["X2"].sensitivity.capability("benchmark")
    assert mixed.fits["X1"].sensitivity.capability("benchmark").available
    assert not own.available
    row = mixed.sensitivity.capability("benchmark")
    assert not row.available and row.status is own.status
    assert row.refusal.reason is own.refusal.reason
    assert row.refusal.condition == f"candidate X2: {own.refusal.condition}"
    with pytest.raises(CapabilityError, match="candidate X2: ") as caught:
        mixed.sensitivity.benchmark()
    assert own.refusal.condition in str(caught.value)


def test_each_candidate_report_is_that_fits_own_call(clean: Any) -> None:
    """On the clean law the rows run and return each fit's own answer, candidate by candidate."""
    support = clean.diagnostics.support()
    assert isinstance(support, CandidateReports) and list(support) == ["X1", "X2"]
    for name, fit in clean.fits.items():
        _frames_equal(support[name].to_frame(), fit.diagnostics.support().to_frame())
    stacked = support.to_frame()
    assert next(iter(stacked.columns)) == "candidate"
    assert stacked["candidate"].tolist().count("X1") == len(
        clean.fits["X1"].diagnostics.support().to_frame()
    )
    bounds = clean.sensitivity.omitted_confounding(cf_y=0.05, cf_d=0.05)
    for name, fit in clean.fits.items():
        own = fit.sensitivity.omitted_confounding(cf_y=0.05, cf_d=0.05)
        assert bounds[name].lower == own.lower and bounds[name].upper == own.upper
    assert clean.sensitivity.capability("benchmark").available


def test_a_refit_runs_on_every_candidate(clean: Any) -> None:
    """No replay gate of the run's own: each candidate's refutation runs on its stored fit."""
    assert clean.diagnostics.capability("refute").available
    refuted = clean.diagnostics.refute(n_replicates=1, random_state=0)
    assert set(refuted) == {"X1", "X2"}
    report = clean.assess(
        include_refits=True, arguments={"refute": {"n_replicates": 1}}, random_state=0
    )
    item = report.diagnostics["refute"]
    assert item.status not in (AssessmentStatus.UNAVAILABLE, AssessmentStatus.DEFERRED)
    assert set(item.report) == {"X1", "X2"}


def test_a_run_without_fits_refuses_every_row_with_the_keep_fits_record() -> None:
    result = _assessed(keep_fits=False)
    rows = (*result.diagnostics.capabilities, *result.sensitivity.capabilities)
    assert rows and all(row.refusal is KEEP_FITS_REFUSAL for row in rows)
    assert all(not row.available for row in rows)
    with pytest.raises(CapabilityError) as caught:
        result.fits  # noqa: B018
    assert caught.value.refusal is KEEP_FITS_REFUSAL
    for call in (
        result.diagnostics.support,
        result.sensitivity.evalue,
        result.diagnostics.influence,
    ):
        with pytest.raises(CapabilityError) as caught:
            call()
        assert caught.value.refusal.condition == KEEP_FITS_REFUSAL.condition
    report = result.assess()
    assert all(item.status is AssessmentStatus.UNAVAILABLE for item in report.diagnostics.items)
    assert result.replayability == Replayability(
        False, False, False, False, False, ("candidate fits",)
    )


def test_the_replay_record_is_every_candidates(clean: Any) -> None:
    each = [replayability(fit) for fit in clean.fits.values()]
    joined = clean.replayability
    assert joined.refit_nuisances is all(slot.refit_nuisances for slot in each)
    assert joined.summarize_existing_artifacts and joined.refit_nuisances


def test_statuses_combine_by_the_worst_and_name_the_candidate() -> None:
    from cleverly.assessment import AssessmentItem, _combined_candidate_item

    items = {
        "X1": AssessmentItem("support", AssessmentStatus.COMPLETED, "fine"),
        "X2": AssessmentItem("support", AssessmentStatus.WARNING, "thin overlap", ("look",)),
    }
    combined = _combined_candidate_item("support", items)
    assert combined.status is AssessmentStatus.WARNING
    assert combined.detail == "X2: thin overlap" and combined.next_steps == ("look",)
    failed = {**items, "X3": AssessmentItem("support", AssessmentStatus.FAILED, "broken")}
    assert _combined_candidate_item("support", failed).status is AssessmentStatus.FAILED
    settled = {name: replace(item, status=AssessmentStatus.PASSED) for name, item in items.items()}
    assert _combined_candidate_item("support", settled).detail == "X1: fine; X2: thin overlap"


def test_a_candidate_that_refuses_inside_a_call_makes_it_refuse(
    clean: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No partial report: a candidate's own refusal raises, named after the candidate."""
    original = DiagnosticsFacade.support
    target = clean.fits["X2"]

    def refusing(self: Any) -> Any:
        if self._result is target:
            raise CapabilityError("no support here", reason=RefusalReason.INVALID_DECLARATION)
        return original(self)

    monkeypatch.setattr(DiagnosticsFacade, "support", refusing)
    with pytest.raises(CapabilityError, match="candidate X2: no support here"):
        replace(clean).diagnostics.support()


def test_influence_redirects_to_the_candidate_fits(clean: Any) -> None:
    with pytest.raises(CapabilityError) as caught:
        clean.diagnostics.influence()
    assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION
    assert "result.fits[candidate].diagnostics.influence()" in caught.value.refusal.alternative


def test_the_validation_rows_combine_the_candidates(clean: Any) -> None:
    report = clean.validate()
    for item in report.items:
        each = [fit.validate()[item.name].status for fit in clean.fits.values()]
        order = (
            AssessmentStatus.FAILED,
            AssessmentStatus.WARNING,
            AssessmentStatus.PASSED,
            AssessmentStatus.COMPLETED,
        )
        assert item.status is next(status for status in order if status in each), item


def test_the_composed_refit_preview_names_itself(
    clean: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X124 B5: a candidate's preview refusal keeps the preview's name in the composed call."""
    original = DiagnosticsFacade.planned_refits
    target = clean.fits["X2"]
    preview = "diagnostic 'planned_refits' previews 'refute', which is unavailable: "

    def refusing(self: Any, **kwargs: Any) -> Any:
        if self._result is target:
            raise CapabilityError(
                f"{preview}no refit here", reason=RefusalReason.INVALID_DECLARATION
            )
        return original(self, **kwargs)

    monkeypatch.setattr(DiagnosticsFacade, "planned_refits", refusing)
    with pytest.raises(CapabilityError) as caught:
        replace(clean).diagnostics.planned_refits(n_replicates=2, random_state=0)
    assert caught.value.refusal.condition == f"{preview}candidate X2: no refit here"


def _ranking_protocol() -> StudyProtocol:
    return StudyProtocol(
        target_population="Adults discharged home",
        eligibility=["Discharged alive"],
        time_zero="Hospital discharge",
        treatment_strategies=["Give the practice", "Withhold the practice"],
        treatment_versions=["As the team gives it", "Usual discharge"],
        outcome="Y",
        horizon="30 days after discharge",
        intercurrent_event_handling=["None"],
        interference_unit="Patient",
        assumption_rationale=["W records every common cause of each practice and Y"],
    )


def test_the_protocol_is_recorded_and_printed_as_the_study_description() -> None:
    protocol = _ranking_protocol()
    ranking = _assessed(protocol=protocol)

    assert ranking.protocol is protocol
    full = str(ranking.summary(detail="full"))
    assert vocabulary.SECTION_STUDY in full
    for line in protocol.summary_lines():
        assert line in full
    # A short summary prints no protocol line, as a single fit's does.
    assert protocol.target_population not in str(ranking.summary())


def test_a_run_without_a_protocol_says_none_was_recorded(clean: Any) -> None:
    assert clean.protocol is None
    full = str(clean.summary(detail="full"))
    assert vocabulary.NO_PROTOCOL in full


def test_the_protocol_leaves_the_estimates_unchanged(clean: Any) -> None:
    ranking = _assessed(protocol=_ranking_protocol())
    _frames_equal(ranking.to_frame(), clean.to_frame())


def test_a_protocol_that_is_not_a_record_is_refused_before_any_fit() -> None:
    with pytest.raises(TypeError, match="StudyProtocol"):
        _assessed(protocol={"target_population": "Adults"})
    with pytest.raises(TypeError, match="StudyProtocol"):
        variable_importance_plan(
            _assessed_frame(),
            outcome="Y",
            candidates=("X1", "X2"),
            covariates=("W",),
            protocol="adults",
        )
    with pytest.raises(TypeError, match="StudyProtocol"):
        replace(_assessed(), protocol="adults")


def test_the_plan_takes_the_protocol_and_counts_the_same_fits() -> None:
    arguments: dict[str, Any] = {
        "outcome": "Y",
        "candidates": ("X1", "X2"),
        "covariates": ("W",),
        "estimator": TMLE(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        ),
    }
    with_record = variable_importance_plan(
        _assessed_frame(), protocol=_ranking_protocol(), **arguments
    )
    without = variable_importance_plan(_assessed_frame(), **arguments)
    assert with_record.total_fits == without.total_fits


def test_each_candidate_fit_stores_the_protocol_fingerprint(clean: Any) -> None:
    protocol = _ranking_protocol()
    ranking = _assessed(protocol=protocol)
    for fit in ranking.fits.values():
        assert fit.provenance.protocol_fingerprint == protocol.fingerprint
    # The nonzero witness: without a protocol the fits store none.
    for fit in clean.fits.values():
        assert fit.provenance.protocol_fingerprint is None
