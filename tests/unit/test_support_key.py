"""The support key, ``classify``, ``combine_support`` and the phase-1 record switch.

The key is read from an estimator and its data before any learner runs. Each test names the
axis it moves and a witness that differs from the baseline, so a key reader that ignores the
axis fails. The record switch changes no output: a fit with the switch on returns the numbers
of a fit with it off.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import _support
from cleverly._refusal import Refusal, RefusalReason
from cleverly._support import (
    RECORD_ENVIRONMENT_VARIABLE,
    Classification,
    Support,
    SupportKey,
    SupportRow,
    classify,
    classify_key,
    combine_support,
    point_key,
    recording_directory,
)
from cleverly.data.causal_data import CausalData
from cleverly.datasets import make_clustered, make_linear_ate, make_missing_outcome_binary
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.longitudinal import LTMLE
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import LONGITUDINAL_N, longitudinal_learners
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

SUPPORTED_A = Support("supported", "A study covers it.", ("study-a",), row="R1", limits=("l1",))
SUPPORTED_B = Support("supported", "A study covers it.", ("study-b",), row="R2")
EXPERIMENTAL_X = Support("experimental", "No study covers X.", ("X1",))
EXPERIMENTAL_Y = Support("experimental", "No study covers Y.", ("X1", "Y2"), limits=("l2",))


def prepared(frame: Any, **roles: Any) -> CausalData:
    """The prepared data a fit on ``frame`` reads."""
    return CausalData.from_frame(frame, outcome="Y", treatment="A", **roles)


def key_of(estimator: TMLE, frame: Any, estimands: tuple[str, ...] = ("ate",), **roles: Any):
    data = prepared(frame, **roles)
    return point_key(estimator, data, estimands)


@pytest.fixture(scope="module")
def frame() -> Any:
    return make_linear_ate(n=200, seed=3)[0]


def clustered_frame(k: int) -> Any:
    return make_clustered(n=4 * k, cluster_size=4, seed=5)[0]


class TestCombineSupport:
    def test_equal_records_stay(self) -> None:
        assert combine_support([SUPPORTED_A, SUPPORTED_A, SUPPORTED_A]) is SUPPORTED_A

    def test_one_record_is_itself(self) -> None:
        assert combine_support([EXPERIMENTAL_X]) is EXPERIMENTAL_X

    def test_an_experimental_input_makes_the_result_experimental(self) -> None:
        """The witness: a supported record beside an experimental one must not stay supported."""
        combined = combine_support([SUPPORTED_A, EXPERIMENTAL_X])
        assert combined.label == "experimental"
        assert combined.reason == EXPERIMENTAL_X.reason
        assert combined.evidence == ("X1",)
        assert combined.row is None

    def test_reasons_and_evidence_join_in_first_seen_order(self) -> None:
        combined = combine_support([EXPERIMENTAL_X, SUPPORTED_A, EXPERIMENTAL_Y, EXPERIMENTAL_X])
        assert combined.reason == f"{EXPERIMENTAL_X.reason} {EXPERIMENTAL_Y.reason}"
        assert combined.evidence == ("X1", "Y2")
        assert combined.limits == ("l1", "l2")

    def test_two_supported_rows_are_experimental(self) -> None:
        combined = combine_support([SUPPORTED_A, SUPPORTED_B])
        assert combined.label == "experimental"
        assert combined.reason == "This estimate combines estimates of two support rows."

    def test_the_same_row_with_other_evidence_is_two_records(self) -> None:
        other = Support("supported", "A study covers it.", ("study-c",), row="R1", limits=("l1",))
        assert combine_support([SUPPORTED_A, other]).label == "experimental"

    def test_an_empty_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            combine_support([])


class TestPointKey:
    def test_a_default_tmle_key(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        assert key.kind == "point"
        assert key.engine == ("tmle",)
        assert key.family == ("arm", "ate")
        assert key.outcome == ("gaussian", "unbounded")
        assert key.missing == ("complete",)
        assert key.treatment == ("binary",)
        assert key.weighting == "none"
        assert key.splitting == ("in_sample",)
        assert key.dependence == "iid"
        assert key.qualifiers == frozenset()
        assert dict(key.options) == {
            "fluctuation": "logistic",
            "targeting": "iterative",
            "target_weights": False,
            "multiplier_kind": "rademacher",
        }
        assert learners["outcome"] is LinearRegression
        assert learners["treatment"] is LogisticRegression
        assert learners["missingness"] is None

    def test_the_estimands_are_sorted_and_named(self, frame: Any) -> None:
        key, _ = key_of(TMLE(**linear_in_sample()), frame, ("ey1", "ate", "ey0"))
        assert key.family == ("arm", "ate", "ey0", "ey1")

    @pytest.mark.parametrize(
        ("overrides", "splitting"),
        [
            ({"cross_fit": False}, ("in_sample",)),
            ({"cross_fit": True}, ("stacked",)),
            ({"cross_fit": True, "cv_evaluation": True}, ("fold_evaluated",)),
            ({"cross_fit": True, "targeting_scheme": "fold"}, ("fold_targeted",)),
            ({"cross_fit": True, "repeats": 2}, ("stacked", "repeated")),
        ],
    )
    def test_splitting(self, frame: Any, overrides: dict[str, Any], splitting: tuple) -> None:
        key, _ = key_of(TMLE(**linear_in_sample(**overrides)), frame)
        assert key.splitting == splitting

    def test_options_move_the_key(self, frame: Any) -> None:
        base, _ = key_of(TMLE(**linear_in_sample()), frame)
        for overrides in (
            {"fluctuation": "linear"},
            {"targeting": "one_step"},
            {"target_weights": True},
            {"multiplier_kind": "mammen"},
        ):
            moved, _ = key_of(TMLE(**linear_in_sample(**overrides)), frame)
            assert moved != base, overrides
            assert len(base.differing_axes(moved)) == 1
            assert base.differing_axes(moved) == ("options",)

    def test_the_outcome_bounds_axis(self, frame: Any) -> None:
        bounded = TMLE(**linear_in_sample(q_bounds=(-100.0, 100.0)))
        key, _ = key_of(bounded, frame)
        assert key.outcome == ("gaussian", "bounded")

    def test_a_missing_outcome_is_on_the_missing_axis(self) -> None:
        data = make_missing_outcome_binary(n=200, seed=1)[0]
        key, _ = key_of(TMLE(**linear_in_sample()), data, delta="Delta")
        assert key.missing == ("outcome",)

    def test_estimated_weights_differ_from_fixed_and_from_none(self, frame: Any) -> None:
        weighted = frame.assign(w=1.0 + 0.37 * (frame["W1"] > 0))
        estimator = TMLE(**linear_in_sample())
        none, _ = key_of(estimator, frame)
        fixed, _ = key_of(estimator, weighted, weights="w")
        estimated, _ = key_of(estimator, weighted, weights="w", weights_estimated=True)
        assert (none.weighting, fixed.weighting, estimated.weighting) == (
            "none",
            "fixed",
            "estimated",
        )

    def test_strata_are_a_qualifier(self, frame: Any) -> None:
        stratified = frame.assign(S=(frame["W1"] > 0).astype(int))
        key, _ = key_of(TMLE(**linear_in_sample()), stratified, strata=["S"])
        assert key.qualifiers == frozenset({"strata"})

    def test_a_named_reference_is_a_qualifier(self, frame: Any) -> None:
        key, _ = key_of(TMLE(**linear_in_sample(reference=0)), frame)
        assert "reference_arm" in key.qualifiers

    def test_an_explicit_learner_class_is_read(self, frame: Any) -> None:
        class MyOutcome(LinearRegression):
            pass

        _, learners = key_of(TMLE(**linear_in_sample(outcome_learner=MyOutcome())), frame)
        assert learners["outcome"] is MyOutcome
        assert learners["outcome"] is not LinearRegression

    def test_drtmle_options_and_engine(self, frame: Any) -> None:
        key, _ = key_of(DRTMLE(**linear_in_sample()), frame)
        assert key.engine == ("drtmle",)
        options = dict(key.options)
        assert {"guard", "reduction", "reduced_crossfit", "update_order", "randomized"} <= set(
            options
        )
        assert options["evaluation"] is False
        plain, _ = key_of(TMLE(**linear_in_sample()), frame)
        assert "guard" not in dict(plain.options)

    def test_ctmle_engine_holds_strategy_and_design(self, frame: Any) -> None:
        key, _ = key_of(CTMLE(**linear_in_sample(strategy="oat", oat_design="per_arm")), frame)
        assert key.engine == ("collaborative_tmle", "oat", "per_arm")

    @pytest.mark.parametrize(
        ("clusters", "tier"),
        [(5, "below_floor"), (9, "below_floor"), (10, "t"), (39, "t"), (40, "clustered")],
    )
    def test_the_dependence_tiers_at_their_boundaries(self, clusters: int, tier: str) -> None:
        """The floors are 10 and 40: each boundary count lands on the named side."""
        estimator = TMLE(**linear_in_sample())
        key, _ = key_of(estimator, clustered_frame(clusters), id="cluster")
        assert key.dependence == tier

    def test_an_iid_fit_is_iid(self, frame: Any) -> None:
        key, _ = key_of(TMLE(**linear_in_sample()), frame)
        assert key.dependence == "iid"

    def test_the_longitudinal_floor_is_twenty(self) -> None:
        assert _support._dependence(19, 20) == "below_floor"
        assert _support._dependence(20, 20) == "t"
        assert _support._dependence(40, 20) == "clustered"
        assert _support._dependence(19, 10) == "t"


class TestKeyAxes:
    def test_differing_axes_name_each_moved_axis_once(self, frame: Any) -> None:
        base, _ = key_of(TMLE(**linear_in_sample()), frame)
        other, _ = key_of(TMLE(**linear_in_sample(cross_fit=True)), frame, ("ate", "ey1"))
        assert base.differing_axes(other) == ("family", "splitting")
        assert base.differing_axes(base) == ()

    def test_to_json_round_trips_through_json(self, frame: Any) -> None:
        key, _ = key_of(TMLE(**linear_in_sample()), frame)
        again = json.loads(json.dumps(key.to_json()))
        assert again["engine"] == ["tmle"]
        assert again["qualifiers"] == []


def make_row(row_id: str, key: SupportKey, **kwargs: Any) -> SupportRow:
    return SupportRow(row_id, key, frozenset({"point", "wald"}), ("study",), **kwargs)


class TestClassify:
    def test_no_rows_is_unmatched_with_no_nearest(self, frame: Any) -> None:
        data = prepared(frame)
        classification = classify(TMLE(**linear_in_sample()), data, ("ate",))
        assert isinstance(classification, Classification)
        assert classification.row is None
        assert classification.nearest is None
        assert classification.label == "experimental"

    def test_a_matching_row_is_supported(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        row = make_row("R1", key)
        classification = classify_key(key, learners, rows=[row])
        assert classification.row is row
        assert classification.label == "supported"

    def test_a_condition_makes_a_matched_fit_experimental(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        condition = Refusal(RefusalReason.IMPLEMENTATION_GAP, "A condition.", "Instead.", "X60")
        classification = classify_key(key, learners, [condition], rows=[make_row("R1", key)])
        assert classification.row is not None
        assert classification.label == "experimental"

    def test_an_unmatched_key_names_the_nearest_row_and_its_axes(self, frame: Any) -> None:
        base, learners = key_of(TMLE(**linear_in_sample()), frame)
        far, _ = key_of(TMLE(**linear_in_sample(cross_fit=True, repeats=2)), frame, ("ey1",))
        near, _ = key_of(TMLE(**linear_in_sample(cross_fit=True)), frame)
        rows = [make_row("FAR", far), make_row("NEAR", near)]
        classification = classify_key(base, learners, rows=rows)
        assert classification.row is None
        assert classification.nearest == ("NEAR", ("splitting",))

    def test_a_learner_the_row_names_must_be_the_fits(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        row = make_row("R1", key, learners={"outcome": LogisticRegression})
        classification = classify_key(key, learners, rows=[row])
        assert classification.row is None
        assert classification.nearest == ("R1", ("learners",))
        exact = make_row("R2", key, learners={"outcome": LinearRegression})
        assert classify_key(key, learners, rows=[exact]).row is exact

    def test_two_matching_rows_are_a_bug(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        with pytest.raises(RuntimeError, match="all match one fit"):
            classify_key(key, learners, rows=[make_row("A", key), make_row("B", key)])


class TestRecordSwitch:
    def test_the_directory_is_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)
        assert recording_directory() is None
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, "")
        assert recording_directory() is None

    def test_with_the_switch_off_the_classifier_never_runs(
        self, monkeypatch: pytest.MonkeyPatch, frame: Any
    ) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)

        def fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("the classifier ran with the switch off")

        monkeypatch.setattr(_support, "classify", fail)
        monkeypatch.setattr(_support, "longitudinal_key", fail)
        TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A")

    def test_a_point_fit_records_and_changes_no_number(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)
        off = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path / "fast"))
        on = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        assert on.estimates["ate"].psi == off.estimates["ate"].psi
        assert on.estimates["ate"].std_error == off.estimates["ate"].std_error

        records = read_records(tmp_path / "fast")
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]
        start, end = records
        assert start["source"] == "TMLE._fit_single"
        assert start["key"]["engine"] == ["tmle"]
        assert start["label"] == "experimental"
        assert start["row"] is None
        assert start["subdirectory"] == "fast"
        assert "test_support_key.py" in start["pytest"]
        assert end["statuses"] == ["influence_curve"]
        assert end["supplies_inference"] == [True]
        assert end["band_built_today"] is False
        assert end["n_estimates"] == len(on.estimates)

    def test_a_bootstrap_replicate_does_not_classify(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        estimator = TMLE(**linear_in_sample(n_bootstrap=3))
        result = estimator.fit(frame, outcome="Y", treatment="A").single()
        assert result.bootstrap is not None
        records = read_records(tmp_path)
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]
        assert records[-1]["bootstrap_attached"] is True

    def test_retarget_classifies_its_estimand_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        result = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        result.estimator.retarget(result.data, result.nuisance, estimands=("ey1",))
        records = read_records(tmp_path)
        assert [r["source"] for r in records] == ["TMLE.retarget", "TMLE.retarget"]
        assert records[0]["key"]["family"] == ["arm", "ey1"]

    def test_a_refusal_site_is_recorded_and_the_refusal_still_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        data = make_missing_outcome_binary(n=300, seed=1)[0]
        estimator = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            fluctuation="linear",
            simultaneous=False,
            n_folds=2,
            estimands=("ate",),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CapabilityError) as raised:
                estimator.fit(data, outcome="Y", treatment="A", delta="Delta")
        assert raised.value.refusal.roadmap == "X60"
        (record,) = read_records(tmp_path)
        assert record["record"] == "refusal_site"
        assert record["site"] == "arm.linear_fluctuation"
        assert record["code"] == "implementation_gap"
        assert record["roadmap"] == "X60"
        assert record["flagged"] is False

    def test_a_flagged_site_is_marked_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        data = make_missing_outcome_binary(n=300, seed=1)[0]
        estimator = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            targeting_scheme="fold",
            simultaneous=False,
            n_folds=2,
            estimands=("ate",),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CapabilityError):
                estimator.fit(data, outcome="Y", treatment="A", delta="Delta")
        (record,) = read_records(tmp_path)
        assert (record["site"], record["flagged"], record["roadmap"]) == (
            "arm.fold_targeting",
            True,
            "X53",
        )

    def test_a_classifier_failure_is_recorded_and_the_fit_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))

        def fail(*args: Any, **kwargs: Any) -> None:
            raise KeyError("boom")

        monkeypatch.setattr(_support, "classify", fail)
        TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A")
        (record,) = read_records(tmp_path)
        assert record["record"] == "classification_error"
        assert "boom" in record["error"]


class TestLongitudinalRecord:
    def fit(self, **overrides: Any) -> Any:
        frame = multivalue_panel(n=LONGITUDINAL_N, seed=43)
        return LTMLE(
            {"never": 0, "always": 1}, reference="never", **longitudinal_learners(**overrides)
        ).fit(frame, **COLUMNS)

    def test_a_fit_records_once_and_changes_no_number(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)
        off = self.fit()
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        on = self.fit()
        for name, estimate in off.estimates.items():
            assert on.estimates[name].psi == estimate.psi
        records = read_records(tmp_path)
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]
        key = records[0]["key"]
        assert key["kind"] == "longitudinal"
        assert key["engine"] == ["tmle"]
        assert key["family"] == ["end_of_study", "regimens", "static"]
        assert key["splitting"] == ["in_sample"]
        assert key["treatment"] == ["3", "2"]
        assert key["missing"] == ["censoring"]

    def test_a_bootstrap_replicate_does_not_classify(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        result = self.fit(n_bootstrap=2)
        assert result.bootstrap is not None
        records = read_records(tmp_path)
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]


def read_records(directory: Path) -> list[dict[str, Any]]:
    """Every record under ``directory``, in file and line order."""
    lines: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    return [json.loads(line) for line in lines]
