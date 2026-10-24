"""The support label of ``StructuralNestedMean`` (RM42b).

The rows ``G42`` of :mod:`cleverly._support_rows` name the engine ``snmm``. Each covers one key
that the registered study ``structural-nested-mean`` (X44) fits: cross-fitted, with the point
estimate and the Wald interval. Each class below names a component that can vanish and a
witness that fails when it does.

* ``TestTheStudyRowsCoverTheFit`` pins that a cross-fitted ``GLM`` fit of two binary nodes reads
  ``supported`` on ``G42.two_nodes``, that its band stays ``experimental``, and that the fit
  reads ``experimental`` once the ``G42`` rows are removed.
* ``TestAnUncoveredFitReadsExperimental`` pins the label, the reason and the nearest row of an
  in-sample fit, which no row covers, at every surface a reader meets: the estimate, the frame,
  the summary, the bands and a saved result.
* ``TestTheKeyReadsTheFit`` pins the axes of the support key, so a later row can cover a fit.
* ``TestARowCoversTheKey`` registers a row built from a fit's own key. The same fit then reads
  ``supported``, and a row that differs on one axis leaves it ``experimental``. The control
  that drops the classification reads the unclassified default and fails the first test.
* ``TestTheStatusesKeepTheirIntervals`` pins which statuses show an interval under the label
  and which withhold it.
* ``TestTheModelCheckCarriesTheLabel`` pins the label of the check and its refusal on a
  withheld status.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import _support
from cleverly._summary import vocabulary
from cleverly._summary._estimates import plain_support_reason
from cleverly._support import UNCLASSIFIED, SupportRow, structural_key
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.learners import GLM
from cleverly.longitudinal import StructuralNestedMean, snmm
from cleverly.longitudinal import structural as module

from .. import discrete_law_structural as laws
from ._inference_status_support import assert_shows_interval
from .test_structural_nested_mean import GLMS, REGIMENS, SIM, _fit, _simulated

#: The reason of an estimate that no row covers, up to the nearest-row report.
NO_STUDY = "No registered study covers this configuration."

#: The known support of the simulated outcome, which a cross-fitted check of a continuous outcome needs.
BOUNDS = (-15.0, 15.0)


def fit(frame: Any = None, **settings: Any) -> Any:
    """A cross-fitted GLM fit with two static regimens and its default bands."""
    options = {"regimens": REGIMENS, "n_folds": 2, **GLMS, "simultaneous": True} | settings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return snmm(_simulated() if frame is None else frame, **options, **SIM)


@pytest.fixture(scope="module")
def result() -> Any:
    return fit()


@pytest.fixture(scope="module")
def uncovered() -> Any:
    """An in-sample ``GLM`` fit. The ``G42`` rows are cross-fitted, so no row covers it."""
    return fit(n_folds=1)


class TestTheStudyRowsCoverTheFit:
    def test_every_estimate_is_supported_on_the_two_node_row(self, result: Any) -> None:
        assert result.estimates
        for name, estimate in result.estimates.items():
            assert estimate.support.label == "supported", name
            assert estimate.support.row == "G42.two_nodes", name
            assert estimate.support.evidence == ("structural-nested-mean",), name
        assert set(result.to_frame()["support"]) == {"supported"}

    def test_the_band_is_experimental_because_the_row_licenses_no_band(self, result: Any) -> None:
        assert result.simultaneous is not None
        support = result.simultaneous.support
        assert support.label == "experimental"
        assert "G42.two_nodes licenses no simultaneous band" in support.reason

    def test_removing_the_study_rows_leaves_the_fit_experimental(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = tuple(row for row in _support.registered_rows() if not row.id.startswith("G42"))
        monkeypatch.setattr(_support, "registered_rows", lambda: rows)
        estimate = fit()["ey_regimen[always]"]
        assert estimate.support.label == "experimental"
        assert estimate.support.reason.startswith(NO_STUDY)

    def test_a_booster_regression_learner_leaves_the_fit_experimental(self) -> None:
        """The rows name ``regression`` as ``None``, so any ``regression_learner`` leaves a row."""
        booster = HistGradientBoostingRegressor(max_iter=10)
        estimate = fit(regression_learner=booster)["ey_regimen[always]"]
        assert estimate.support.label == "experimental"
        assert estimate.support.row is None
        assert "Nearest row: G42.two_nodes" in estimate.support.reason
        assert "differing on learners" in estimate.support.reason

    def test_dropping_the_regression_role_from_the_rows_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row that omits the role licenses a booster, so the test above fails on it."""
        rows = tuple(
            replace(
                row,
                learners={k: v for k, v in (row.learners or {}).items() if k != "regression"},
            )
            if row.id.startswith("G42")
            else row
            for row in _support.registered_rows()
        )
        monkeypatch.setattr(_support, "registered_rows", lambda: rows)
        estimate = fit(regression_learner=HistGradientBoostingRegressor(max_iter=10))[
            "ey_regimen[always]"
        ]
        assert estimate.support.label == "supported"


class TestAnUncoveredFitReadsExperimental:
    def test_every_estimate_is_experimental_with_the_nearest_row(self, uncovered: Any) -> None:
        assert uncovered.estimates
        for name, estimate in uncovered.estimates.items():
            support = estimate.support
            assert support.label == "experimental", name
            assert support.row is None
            assert support.reason.startswith(NO_STUDY), name
            assert "Nearest row: G42.two_nodes" in support.reason, name
            assert "differing on splitting" in support.reason, name
            assert support is not UNCLASSIFIED

    def test_the_repr_and_the_frame_carry_the_label(self, uncovered: Any) -> None:
        assert repr(uncovered["ey_regimen[always]"]).count(", experimental") == 1
        frame = uncovered.to_frame()
        assert set(frame["support"]) == {"experimental"}
        assert all(str(reason).startswith(NO_STUDY) for reason in frame["support_reason"])

    def test_the_summary_prints_the_column_and_one_note_per_reason(self, uncovered: Any) -> None:
        summary = uncovered.summary(detail="full")
        assert summary.table("estimates").columns[-1] == vocabulary.SUPPORT
        assert summary.note_codes.count("experimental") == 1
        reasons = [
            line
            for line in summary.section(vocabulary.SECTION_SETTINGS).lines
            if line.startswith("Why the rows are experimental:")
        ]
        assert len(reasons) == 1
        reason = uncovered["ey_regimen[always]"].support.reason
        assert plain_support_reason(reason) in reasons[0]

    def test_the_bands_and_a_contrast_carry_the_label(self, uncovered: Any) -> None:
        assert uncovered.simultaneous is not None
        assert uncovered.simultaneous.support.label == "experimental"
        bands = uncovered.summary().note("simultaneous_bands").text
        assert vocabulary.NOTE_TEXTS["bands_experimental"] in bands
        derived = uncovered.contrast(
            lambda psi: psi[0] - psi[1],
            ["ey_regimen[always]", "ey_regimen[never]"],
        )
        assert derived.support.label == "experimental"

    def test_the_label_and_the_key_survive_save_and_load(
        self, uncovered: Any, tmp_path: Any
    ) -> None:
        path = tmp_path / "snmm.joblib"
        uncovered.save(path)
        loaded = cleverly.load(path)
        assert loaded.config.support_key == uncovered.config.support_key
        assert loaded.config.support_key is not None
        for name, estimate in uncovered.estimates.items():
            assert loaded[name].support == estimate.support
        assert loaded.simultaneous.support == uncovered.simultaneous.support

    def test_dropping_the_classification_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(module, "classify_structural_fit", lambda *a, **k: None)
        mutant = fit()
        estimate = mutant["ey_regimen[always]"]
        assert estimate.support == UNCLASSIFIED
        assert mutant.config.support_key is None
        assert not estimate.support.reason.startswith(NO_STUDY)


class TestTheKeyReadsTheFit:
    def key(self, **settings: Any) -> Any:
        return fit(**settings).config.support_key

    def test_the_engine_family_and_splitting(self) -> None:
        key = self.key()
        assert key.engine == ("snmm",)
        assert key.family == frozenset({"end_of_study", "blips", "regimens", "static"})
        assert key.splitting == ("cross_fitted",)
        assert key.weighting == "none"
        assert key.dependence == "iid"
        assert dict(key.options) == {"covariate_features": False, "multiplier_kind": "rademacher"}

    def test_an_in_sample_fit_a_missing_regimen_and_a_covariate_feature(self) -> None:
        assert self.key(n_folds=1).splitting == ("in_sample",)
        assert self.key(regimens=None).family == frozenset({"end_of_study", "blips"})
        with_covariate = self.key(regimens=None, blip_features={2: ["S2"]})
        assert dict(with_covariate.options)["covariate_features"] is True

    def test_weights_and_clusters(self) -> None:
        frame = _simulated(clusters=8)
        frame["w"] = np.random.default_rng(1).uniform(0.5, 1.5, len(frame))
        assert self.key(frame=frame, weights="w").weighting == "fixed"
        assert self.key(frame=frame, weights="w", weights_estimated=True).weighting == "estimated"
        assert self.key(frame=frame, id="id").dependence == "below_floor"
        assert self.key(frame=_simulated(clusters=30), id="id").dependence == "t"

    def test_a_dose_node_reads_a_continuous_treatment_axis(self) -> None:
        key = _fit(laws.law_c(), None, None).config.support_key
        assert key.treatment[0] == "continuous"

    def test_the_learner_classes_are_in_the_classification_and_not_the_key(self) -> None:
        estimator = StructuralNestedMean(REGIMENS, **GLMS)
        key, learners = structural_key(
            estimator,
            _prepared(),
            cross_fitted=True,
            has_regimens=True,
            covariate_features=False,
        )
        assert learners == {"outcome": GLM, "treatment": GLM, "regression": None}
        assert "GLM" not in repr(key)


def _prepared() -> Any:
    from cleverly.longitudinal.estimator import _SequentialEstimator

    return _SequentialEstimator._prepare(
        _simulated(),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[["S1"], ["S2"]],
        censoring=None,
        id=None,
        weights=None,
        weights_type="probability",
        weights_estimated=False,
        family="auto",
    )


def covering_row(key: Any, *, row_id: str = "SNM-TEST", **replacements: Any) -> SupportRow:
    """A row that names every axis of ``key``, licenses every kind and names the GLM learners."""
    from dataclasses import replace

    return SupportRow(
        id=row_id,
        key=replace(key, **replacements),
        kinds=frozenset({"point", "wald", "wald_t", "band", "derived"}),
        evidence=("structural-nested-mean-test",),
        learners={"outcome": GLM, "treatment": GLM},
    )


class TestARowCoversTheKey:
    def test_a_row_built_from_the_fit_key_makes_the_fit_supported(
        self, result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = covering_row(result.config.support_key)
        monkeypatch.setattr(_support, "registered_rows", lambda: (row,))
        covered = fit()
        for name, estimate in covered.estimates.items():
            assert estimate.support.label == "supported", name
            assert estimate.support.row == "SNM-TEST"
            assert estimate.support.evidence == ("structural-nested-mean-test",)
        assert covered.simultaneous.support.label == "supported"
        assert set(covered.to_frame()["support"]) == {"supported"}
        derived = covered.contrast(
            lambda psi: psi[0] - psi[1], ["ey_regimen[always]", "ey_regimen[never]"]
        )
        assert derived.support.label == "supported"
        assert vocabulary.SUPPORT not in covered.summary().table("estimates").columns

    def test_a_row_that_differs_on_one_axis_leaves_the_fit_experimental(
        self, result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = covering_row(result.config.support_key, weighting="fixed")
        monkeypatch.setattr(_support, "registered_rows", lambda: (row,))
        other = fit()
        estimate = other["ey_regimen[always]"]
        assert estimate.support.label == "experimental"
        assert "Nearest row: SNM-TEST" in estimate.support.reason
        assert "differing on weighting" in estimate.support.reason

    def test_a_row_that_names_another_learner_leaves_the_fit_experimental(
        self, result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        row = replace(
            covering_row(result.config.support_key), learners={"outcome": LinearRegression}
        )
        monkeypatch.setattr(_support, "registered_rows", lambda: (row,))
        assert fit()["ey_regimen[always]"].support.label == "experimental"

    def test_a_row_without_the_band_kind_leaves_the_band_experimental(
        self, result: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        row = replace(
            covering_row(result.config.support_key),
            kinds=frozenset({"point", "wald", "derived"}),
        )
        monkeypatch.setattr(_support, "registered_rows", lambda: (row,))
        partial = fit()
        assert partial["ey_regimen[always]"].support.label == "supported"
        assert partial.simultaneous.support.label == "experimental"


class TestTheStatusesKeepTheirIntervals:
    def test_estimated_weights_show_the_plug_in_interval_under_the_label(self) -> None:
        frame = _simulated()
        frame["w"] = np.random.default_rng(1).uniform(0.5, 1.5, len(frame))
        estimated = fit(frame, weights="w", weights_estimated=True)
        assert_shows_interval(estimated, "estimated_weight_plugin", None)
        reason = estimated["ey_regimen[always]"].support.reason
        assert module.status_record("estimated_weight_plugin").shown_reason in reason
        assert "X50" in estimated["ey_regimen[always]"].support.evidence

    def test_few_clusters_show_a_t_interval_under_the_label(self) -> None:
        few = fit(_simulated(clusters=8), id="id", simultaneous=False)
        assert_shows_interval(few, "few_cluster_plugin", 6)

    def test_two_clusters_withhold_the_interval(self) -> None:
        two = fit(_simulated(clusters=2), id="id", simultaneous=False)
        assert two.inference_status == "no_cluster_variance_plugin"
        estimate = two["ey_regimen[always]"]
        assert estimate.support.label == "experimental"
        assert not estimate.supplies_inference
        with pytest.raises(CapabilityError):
            _ = estimate.ci
        assert two.simultaneous is None
        assert "support" in two.summary()

    def test_an_in_sample_adaptive_fit_withholds_the_interval(self) -> None:
        adaptive = fit(
            n_folds=1,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            simultaneous=False,
        )
        assert adaptive.inference_status == "adaptive_baseline_plugin"
        estimate = adaptive["ey_regimen[always]"]
        assert estimate.support.label == "experimental"
        with pytest.raises(CapabilityError):
            _ = estimate.ci


class TestTheModelCheckCarriesTheLabel:
    def test_the_check_is_experimental_and_prints_the_reason(self, result: Any) -> None:
        """The structural side is supported, and the ``LTMLE`` side of the check has no row."""
        check = result.check_model(q_bounds=BOUNDS)
        assert check.support.label == "experimental"
        assert NO_STUDY in check.support.reason
        text = check.summary().note("experimental").text
        assert plain_support_reason(check.support.reason) in text

    def test_a_check_on_a_few_cluster_fit_runs_under_the_label(self) -> None:
        few = fit(_simulated(clusters=8), id="id", simultaneous=False)
        check = few.check_model(q_bounds=BOUNDS)
        assert check.support.label == "experimental"
        shown = module.status_record("few_cluster_plugin").shown_reason
        assert shown in check.support.reason

    def test_a_withheld_status_still_refuses_the_check(self) -> None:
        two = fit(_simulated(clusters=2), id="id", simultaneous=False)
        with pytest.raises(CapabilityError) as raised:
            two.check_model()
        assert raised.value.refusal.reason == RefusalReason.INFERENCE_GAP
