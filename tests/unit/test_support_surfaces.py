"""The support label at the surfaces that read an estimate, each with a mutation control.

An experimental fit shows its label in variable importance, the E-value row, the
``nuisance_models`` assessment item, the ``CoverageStudy`` summary, the bootstrap line, the
band header and the positivity sweep. Each test has a control that drops the label at its
source and fails the same assertion.
"""

from __future__ import annotations

import importlib
import pickle
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pytest

from cleverly import _support, variable_importance
from cleverly import load as cleverly_load
from cleverly._summary import vocabulary
from cleverly._summary._estimates import plain_support_reason
from cleverly._support import Support, combine_support
from cleverly.datasets import make_binary_outcome, make_clustered
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.inference import influence as influence_module
from cleverly.inference import results as results_module
from cleverly.learners import ParametricMechanism, ParametricOutcome
from cleverly.sensitivity._derived import _derived_risk_ratio
from cleverly.sensitivity.evalue import evalue
from cleverly.sensitivity.positivity import truncation_curve
from cleverly.validation import CoverageStudy
from tests.conftest import linear_in_sample
from tests.unit._inference_status_support import assert_support_round_trips
from tests.unit.test_outcome_adaptive_per_arm import binary_frame

SUPPORTED = Support(label="supported", reason="", evidence=(), row=None, limits=())


def clustered_fit(estimands: tuple[str, ...] = ("ey1", "ey0", "ate")) -> Any:
    """A clustered binary fit, which no registered row covers."""
    frame = make_clustered(n=160, cluster_size=4, seed=5, family="binomial")[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(**linear_in_sample(estimands=estimands))
            .fit(frame, outcome="Y", treatment="A", id="cluster")
            .single()
        )


@pytest.fixture(scope="module")
def clustered() -> Any:
    return clustered_fit()


tmle_module = importlib.import_module("cleverly.estimators.tmle")


@pytest.fixture
def always_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: every estimate and bootstrap summary reads as supported."""
    for module in (influence_module, results_module):
        monkeypatch.setattr(module, "support_for", lambda *args, **kwargs: SUPPORTED)


def nuisance_item(result: Any) -> Any:
    report = result.assess()
    items = [*report.validation.items, *report.diagnostics.items, *report.sensitivity.items]
    return next(item for item in items if item.name == "nuisance_models")


class TestEValue:
    def check(self, result: Any) -> None:
        row = evalue(result, "ate")
        assert row.support.label == "experimental"
        assert row.to_dict()["support"] == "experimental"
        summary = row.summary()
        assert summary.note("experimental").kind == "note"
        assert summary.note("experimental").text == plain_support_reason(row.support.reason)
        settings = row.summary(detail="full").section("Settings").lines
        assert f"Why experimental: {plain_support_reason(row.support.reason)}" in settings

    def test_an_experimental_fit_marks_its_e_value(self, clustered: Any) -> None:
        self.check(clustered)

    def test_the_derived_ratio_of_a_clustered_fit_is_experimental(self, clustered: Any) -> None:
        derived = _derived_risk_ratio(clustered, "ate")
        assert derived.support.label == "experimental"
        assert evalue(clustered, "ate").source_estimand == "ate"

    def test_a_support_that_reads_supported_fails(self, always_supported: None) -> None:
        with pytest.raises(AssertionError):
            self.check(clustered_fit())


class TestVariableImportance:
    @staticmethod
    def importance() -> Any:
        frame = make_clustered(n=160, cluster_size=4, seed=5, family="binomial")[0]
        frame["B"] = (frame["W1"] > 0).astype(int) if "W1" in frame else frame["A"]
        covariates = [c for c in frame.columns if c.startswith("W")][:2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return variable_importance(
                frame,
                outcome="Y",
                candidates=["A", "B"],
                covariates=covariates,
                estimator=TMLE(**linear_in_sample(estimands=("ate",))),
                id="cluster",
            )

    def test_the_frame_carries_a_support_column(self) -> None:
        frame = self.importance().to_frame()
        assert set(frame["support"]) == {"experimental"}

    def test_a_supported_label_leaves_no_column(self, always_supported: None) -> None:
        with pytest.raises((AssertionError, KeyError)):
            self.test_the_frame_carries_a_support_column()


class TestAssessmentItem:
    def test_the_nuisance_item_names_the_experimental_reason(self, clustered: Any) -> None:
        item = nuisance_item(clustered)
        reason = clustered["ate"].support.reason
        assert f"experimental: {reason}" in item.detail

    def test_a_supported_fit_adds_no_clause(self) -> None:
        frame = make_binary_outcome(n=300, seed=2)[0]
        result = (
            TMLE(**linear_in_sample(estimands=("ey1", "ey0")))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        item = nuisance_item(result)
        assert "experimental" not in item.detail

    def test_a_label_dropped_at_the_estimate_fails(self, clustered: Any) -> None:
        stripped = replace(
            clustered,
            estimates={
                name: replace(estimate, support=SUPPORTED)
                for name, estimate in clustered.estimates.items()
            },
        )
        item = nuisance_item(stripped)
        assert "experimental:" not in item.detail


class TestCoverageStudy:
    @staticmethod
    def study() -> Any:
        def dgp(n: int, seed: int) -> Any:
            frame = make_clustered(n=n, cluster_size=4, seed=seed, family="binomial")[0]
            return frame, {"ate": 0.0}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CoverageStudy(
                dgp=dgp,
                estimator=lambda: TMLE(**linear_in_sample(estimands=("ate",))),
                n=160,
                n_replicates=2,
                fit_kwargs={"outcome": "Y", "treatment": "A", "id": "cluster"},
                seed=1,
            ).run()

    def check(self, study: Any) -> None:
        assert study["ate"].support == "experimental"
        assert study.to_frame()["support"].tolist() == ["experimental"]
        assert (
            "estimate of ate has no validation study" in study.summary().note("experimental").text
        )

    def test_the_summary_and_frame_show_the_label(self) -> None:
        self.check(self.study())

    def test_a_record_that_reads_supported_fails(self, always_supported: None) -> None:
        with pytest.raises(AssertionError):
            self.check(self.study())


class TestBootstrapAndBand:
    @staticmethod
    def weighted() -> Any:
        frame = make_binary_outcome(n=500, seed=3)[0]
        frame["w"] = np.random.default_rng(3).uniform(0.5, 2.0, len(frame))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                DRTMLE(
                    **linear_in_sample(
                        estimands=("ate", "ey0", "ey1"), simultaneous=True, n_bootstrap=3
                    )
                )
                .fit(frame, outcome="Y", treatment="A", weights="w", weights_estimated=True)
                .single()
            )

    def check(self, result: Any) -> None:
        assert result.simultaneous is not None
        assert result.simultaneous.support.label == "experimental"
        summary = result.summary()
        bands = summary.note("simultaneous_bands").text
        assert vocabulary.NOTE_TEXTS["bands_experimental"] in bands
        assert "bootstrap_experimental" in summary.note_codes

    def test_a_band_at_a_shown_status_is_experimental_and_marked(self) -> None:
        self.check(self.weighted())

    def test_a_support_that_reads_supported_fails(self, always_supported: None) -> None:
        with pytest.raises(AssertionError):
            self.check(self.weighted())

    def test_a_band_without_its_support_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tmle_module, "band_support", lambda *a, **k: SUPPORTED)
        with pytest.raises(AssertionError):
            self.check(self.weighted())

    def test_a_band_inherits_its_estimates_support(self) -> None:
        result = self.weighted()
        assert result.simultaneous.support == combine_support(
            [result.simultaneous.support, *(e.support for e in result.estimates.values())]
        )


class TestPositivitySweep:
    def check(self, result: Any) -> None:
        curve = truncation_curve(result)
        assert set(curve["support"]) == {"experimental"}
        assert result["ate"].support.reason in set(curve["support_reason"])

    def test_the_sweep_keeps_the_condition_text(self, clustered: Any) -> None:
        self.check(clustered)

    def test_a_retarget_that_reads_supported_fails(
        self, clustered: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = tmle_module.classify_point_fit

        def supported(*args: Any, **kwargs: Any) -> Any:
            classification = original(*args, **kwargs)
            return replace(classification, support=SUPPORTED) if classification else None

        monkeypatch.setattr(tmle_module, "classify_point_fit", supported)
        with pytest.raises(Exception):  # noqa: B017 - any failure of the sweep check
            self.check(clustered)


class TestCTMLEOutcomeAdaptive:
    @staticmethod
    def fit() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                CTMLE(
                    strategy="oat",
                    outcome_learner=ParametricOutcome(),
                    treatment_learner=ParametricMechanism(),
                    cross_fit=False,
                    simultaneous=False,
                    random_state=0,
                    estimands=("ate",),
                )
                .fit(binary_frame(300, 21), outcome="Y", treatment="A")
                .single()
            )

    def check(self, result: Any) -> None:
        support = result["ate"].support
        assert (support.label, support.row) == ("supported", "C11.binary")
        assert result.to_frame()["support"].tolist() == ["supported"]

    def test_the_point_fit_reads_its_row(self) -> None:
        self.check(self.fit())

    def test_with_no_row_the_label_is_lost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_support, "registered_rows", lambda: ())
        with pytest.raises(AssertionError):
            self.check(self.fit())


class TestPersistence:
    """``support``, ``config.support_key`` and the bootstrap and band fields survive a save."""

    @pytest.fixture(scope="class")
    def point(self) -> Any:
        return TestBootstrapAndBand.weighted()

    @pytest.fixture(scope="class")
    def longitudinal(self) -> Any:
        from tests.unit.test_longitudinal_cluster_status import fit_competing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_competing(40)

    @staticmethod
    def routes(result: Any, tmp_path: Path) -> dict[str, Any]:
        path = result.save(tmp_path / "fit.cleverly")
        return {
            "save/load": cleverly_load(path),
            "dumps/loads": loads(dumps(result)),
            "pickle": pickle.loads(pickle.dumps(result)),
            "cloudpickle": pickle.loads(cloudpickle.dumps(result)),
        }

    def test_an_experimental_point_fit_with_a_bootstrap_and_a_band(
        self, point: Any, tmp_path: Path
    ) -> None:
        assert point.config.support_key is not None
        assert point.simultaneous.support.label == "experimental"
        assert point["ate"].bootstrap.support.label == "experimental"
        for restored in self.routes(point, tmp_path).values():
            assert_support_round_trips(point, restored)
            assert restored.simultaneous.support.label == "experimental"
            assert restored["ate"].bootstrap.support.label == "experimental"

    def test_an_experimental_longitudinal_fit(self, longitudinal: Any, tmp_path: Path) -> None:
        assert longitudinal.config.support_key is not None
        for restored in self.routes(longitudinal, tmp_path).values():
            assert_support_round_trips(longitudinal, restored)
            for estimate in restored.estimates.values():
                assert estimate.support.label == "experimental"

    def test_a_loaded_estimate_that_lost_its_support_fails(
        self, point: Any, tmp_path: Path
    ) -> None:
        """The mutation: the loaded estimate carries the default support instead."""
        restored = loads(dumps(point))
        restored = replace(
            restored,
            estimates={
                name: replace(estimate, support=SUPPORTED)
                for name, estimate in restored.estimates.items()
            },
        )
        with pytest.raises(AssertionError):
            assert_support_round_trips(point, restored)

    def test_a_loaded_config_that_lost_its_key_fails(self, point: Any) -> None:
        restored = loads(dumps(point))
        restored = replace(restored, config=replace(restored.config, support_key=None))
        with pytest.raises(AssertionError):
            assert_support_round_trips(point, restored)
