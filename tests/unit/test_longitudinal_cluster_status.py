"""An in-sample clustered ``LTMLE`` fit withholds its interval below 40 clusters.

Roadmap row RM26 applies the RM20 few-cluster decision to the longitudinal path. A fit
with fewer than :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD` clusters with
positive weight mass stamps ``"few_cluster_plugin"`` on every mean, contrast and MSM
coefficient, and F22 holds the route that reopens it. ``LTMLE`` refuses ``id=`` above one
fold, so the in-sample fit is the whole surface.

Each kind of fit draws 400 rows from its law and passes the labels
``np.arange(400) * k // 400`` as ``id=``. Only the labels differ between the witness at 39
clusters and the control at 40, so the point estimates are the same numbers, and only the
status moves. Each mutation in :class:`TestTheMutationsFailTheWitness` must fail the check
its surface passes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
from cleverly._inference_status import (
    FEW_CLUSTER_THRESHOLD,
    NO_SIMULTANEOUS_BANDS,
    NON_INFERENTIAL,
)
from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
)
from cleverly.exceptions import CapabilityError
from cleverly.inference import cluster as cluster_module
from cleverly.inference.cluster import cluster_inference_status
from cleverly.longitudinal import LTMLE
from cleverly.utils.frames import as_frame
from tests.unit._inference_status_support import (
    DIAGNOSTIC_COLUMNS,
    INFERENTIAL_COLUMNS,
    ROUTES,
    assert_assessment_note,
    assert_keeps_inference,
    assert_no_inferential_name,
    assert_restamped,
    assert_withholds,
    at_or_below,
    legacy_copy,
    longitudinal_estimator,
    restore,
)
from tests.unit._inference_status_support import LONGITUDINAL_N as N
from tests.unit._inference_status_support import LONGITUDINAL_NODES as NODES
from tests.unit._inference_status_support import cluster_labels as labels
from tests.unit._inference_status_support import longitudinal_learners as learners
from tests.unit.test_longitudinal_msm import DOSE
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

pytestmark = pytest.mark.xdist_group("longitudinal_cluster_status")

FEW = "few_cluster_plugin"

COMPETING: dict[str, Any] = {
    "outcome": {"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
    **NODES,
}
SURVIVAL: dict[str, Any] = {"outcome": ["Y1", "Y2"], **NODES}

#: The kinds of fit whose report adds a curve over the horizons.
CURVES = ("survival", "competing risks")


def fit_end_of_study(k: int | None, *, weights: Any = None, **settings: Any) -> Any:
    """The RM26 roadmap law, with ``k`` clusters: two means and one contrast.

    ``k=None`` fits the same rows with no cluster labels.
    """
    frame = multivalue_panel(n=N, seed=43)
    roles: dict[str, Any] = {}
    if k is not None:
        frame = frame.assign(cluster=labels(N, k))
        roles["id"] = "cluster"
    if weights is not None:
        frame = frame.assign(w=weights(frame["cluster"].to_numpy()))
        roles["weights"] = "w"
    return LTMLE({"never": 0, "always": 1}, reference="never", **learners(**settings)).fit(
        frame, **COLUMNS, **roles
    )


def fit_survival(k: int) -> Any:
    """One event at two horizons: four risks and two contrasts."""
    frame, _ = make_longitudinal_survival(n=N, seed=2)
    frame = frame.assign(cluster=labels(N, k))
    return LTMLE({"always": 1, "never": 0}, reference="never", **learners()).fit(
        frame, **SURVIVAL, id="cluster"
    )


def fit_competing(k: int) -> Any:
    """Two causes at two horizons: eight incidences and four contrasts."""
    frame, _ = make_longitudinal_competing(n=N, seed=3)
    frame = frame.assign(cluster=labels(N, k))
    return LTMLE({"always": 1, "never": 0}, reference="never", **learners()).fit(
        frame, **COMPETING, id="cluster"
    )


def fit_msm(k: int) -> Any:
    """A working model over the regimens: an intercept and a duration coefficient."""
    frame, _ = make_longitudinal(n=N, seed=0)
    frame = frame.assign(cluster=labels(N, k))
    return LTMLE({"always": 1, "never": 0}, msm=DOSE, **learners()).fit(
        frame, outcome="Y", **NODES, id="cluster"
    )


FITS: dict[str, Callable[[int], Any]] = {
    "end of study": fit_end_of_study,
    "survival": fit_survival,
    "competing risks": fit_competing,
    "msm": fit_msm,
}

ONE_ZERO_MASS_CLUSTER = {"weights": lambda cluster: (cluster >= 1).astype(float)}


@pytest.fixture(scope="module", params=list(FITS))
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def few_result(kind: str) -> Any:
    return FITS[kind](FEW_CLUSTER_THRESHOLD - 1)


@pytest.fixture(scope="module")
def control_result(kind: str) -> Any:
    return FITS[kind](FEW_CLUSTER_THRESHOLD)


def assert_reports_withhold(result: Any, kind: str) -> None:
    """The report each kind of fit adds to ``to_frame`` renames its spread too."""
    if kind == "survival":
        risk, survival = result.curve("risk"), result.curve("survival")
        for curve in (risk, survival):
            assert set(curve.columns) >= DIAGNOSTIC_COLUMNS
            assert_no_inferential_name(curve.columns)
            assert (curve["inference"] == FEW).all()
        # The survival view maps each plug-in bound by the rule of its scale: a level
        # mirrors about a half, and a contrast is negated.
        for at_risk, surviving in zip(risk.itertuples(), survival.itertuples(), strict=True):
            estimate = result[at_risk.parameter]
            assert surviving.parameter == at_risk.parameter
            assert (at_risk.plugin_interval_lower, at_risk.plugin_interval_upper) == (
                estimate.plugin_interval
            )
            assert surviving.plugin_std_err == at_risk.plugin_std_err == estimate.plugin_std_error
            flip = 1.0 if estimate.scale == "level" else 0.0
            assert surviving.plugin_interval_lower == flip - at_risk.plugin_interval_upper
            assert surviving.plugin_interval_upper == flip - at_risk.plugin_interval_lower
    elif kind == "competing risks":
        curve = result.curve()
        assert set(curve.columns) >= DIAGNOSTIC_COLUMNS
        assert_no_inferential_name(curve.columns)
        for row in curve.itertuples():
            estimate = result[row.parameter]
            assert row.inference == FEW
            assert row.plugin_std_err == estimate.plugin_std_error
            assert (row.plugin_interval_lower, row.plugin_interval_upper) == (
                estimate.plugin_interval
            )
        total = result.incidence_total()
        assert "plugin_std_err" in total.columns
        assert_no_inferential_name(total.columns)
        # The same spread through the joint covariance, a second path to the number.
        for row in total.itertuples():
            names = [
                name
                for name, (regimen, _, horizon) in result.parameter_index.items()
                if name.startswith("cif_regimen[") and (regimen, horizon) == (row.regimen, row.time)
            ]
            assert len(names) == len(result.config.causes)
            expected = float(np.sqrt(result.covariance(names).sum()))
            assert row.plugin_std_err == pytest.approx(expected, rel=1e-12)
    elif kind == "msm":
        columns = result.coefficients().columns
        assert set(columns) >= DIAGNOSTIC_COLUMNS
        assert_no_inferential_name(columns)
    else:
        contrast = result.contrast(
            lambda psi: psi[0] - psi[1], ["ey_regimen[always]", "ey_regimen[never]"]
        )
        assert contrast.inference == FEW
        with pytest.raises(CapabilityError):
            _ = contrast.ci


class TestFewClustersWithholdTheLongitudinalInterval:
    """The row's first witness: 39 clusters withhold every interval, on every kind of fit."""

    def test_the_estimates_withhold_their_inference(self, few_result: Any) -> None:
        assert_withholds(few_result, FEW)

    def test_each_report_renames_its_spread(self, few_result: Any, kind: str) -> None:
        assert_reports_withhold(few_result, kind)

    def test_an_explicit_band_request_is_skipped_and_named(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1, simultaneous=True)
        assert not caught
        assert result.simultaneous is None
        assert NO_SIMULTANEOUS_BANDS in result.summary()

    def test_the_nuisance_item_and_the_assessment_carry_the_note(self) -> None:
        assert_assessment_note(fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1), FEW)

    def test_the_truncation_curve_replays_under_the_status(self) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1)
        curve = result.diagnostics.truncation_curve(bounds=[0.01, 0.05])
        fitted = curve[curve["is_fitted_bound"]]
        assert len(fitted) == len(result.estimates)
        for row in fitted.itertuples():
            assert row.psi == result[row.estimand].psi

    def test_zero_mass_clusters_do_not_count(self) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD, **ONE_ZERO_MASS_CLUSTER)
        # The witness that the labels alone would keep the interval.
        assert result.data.n_clusters == FEW_CLUSTER_THRESHOLD
        assert_withholds(result, FEW)
        assert f"positive weight mass in {FEW_CLUSTER_THRESHOLD - 1}" in result.summary()

    def test_the_study_workflow_withholds(self) -> None:
        frame, _ = make_longitudinal(n=390, seed=0, cluster_size=10)
        result = (
            CausalStudy(
                frame,
                design=LongitudinalTreatment(
                    outcome="Y",
                    treatment=("A1", "A2"),
                    baseline=("W1", "W2"),
                    time_varying=((), ("L2",)),
                    censoring=("C1", "C2"),
                    cluster="id",
                ),
            )
            .identify(RegimeMean({"always": 1, "never": 0}, reference="always"))
            .estimate(
                outcome_learner=LinearRegression(),
                pseudo_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                random_state=0,
                simultaneous=True,
            )
        )
        assert result.data.n_clusters == FEW_CLUSTER_THRESHOLD - 1
        assert result.inference_status == FEW
        assert result.simultaneous is None


class TestFortyClustersKeepTheLongitudinalInterval:
    """The row's second witness: the same rows in 40 clusters keep every interval."""

    def test_the_estimates_keep_their_interval(self, control_result: Any, kind: str) -> None:
        assert_keeps_inference(control_result)
        if kind in CURVES:
            assert set(control_result.curve().columns) >= INFERENTIAL_COLUMNS - {"p_value"}
            assert "inference" not in control_result.curve().columns
        if kind == "competing risks":
            assert "std_err" in control_result.incidence_total().columns
        assert NON_INFERENTIAL[FEW].reason not in control_result.summary()

    def test_the_default_bands_are_built(self) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD, simultaneous=True)
        assert result.simultaneous is not None
        assert NO_SIMULTANEOUS_BANDS not in result.summary()

    def test_the_cluster_count_moves_no_point_estimate(
        self, few_result: Any, control_result: Any
    ) -> None:
        assert list(few_result.estimates) == list(control_result.estimates)
        for name, estimate in few_result.estimates.items():
            assert estimate.psi == control_result[name].psi

    def test_all_positive_weights_keep_the_interval(self) -> None:
        result = fit_end_of_study(
            FEW_CLUSTER_THRESHOLD, weights=lambda cluster: np.ones(cluster.size)
        )
        assert_keeps_inference(result)
        assert "positive weight mass" not in result.summary()


class TestTheMutationsFailTheWitness:
    """Each wrong rule, or a stamp in the wrong place, fails the check its surface passes."""

    def test_restoring_the_influence_curve_status_fails(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row's third witness: the fit stamps what the rule returns."""
        monkeypatch.setattr(
            longitudinal_estimator,
            "_inference_status",
            lambda data, folds, regimens, msm=None: "influence_curve",
        )
        with pytest.raises(AssertionError):
            assert_withholds(FITS[kind](FEW_CLUSTER_THRESHOLD - 1), FEW)

    def test_a_threshold_of_zero_fails_the_witness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The longitudinal path reads RM20's threshold in the rule's own module."""
        monkeypatch.setattr(cluster_module, "FEW_CLUSTER_THRESHOLD", 0)
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1)
        for check in (assert_withholds, assert_assessment_note):
            with pytest.raises(AssertionError):
                check(result, FEW)

    def test_an_at_or_below_comparison_fails_the_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(longitudinal_estimator, "cluster_inference_status", at_or_below)
        # The mutant still passes the witness, so only the control can catch it.
        assert_withholds(fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1), FEW)
        with pytest.raises(AssertionError):
            assert_keeps_inference(fit_end_of_study(FEW_CLUSTER_THRESHOLD))

    def test_a_status_that_ignores_the_weights_fails_the_zero_mass_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            longitudinal_estimator,
            "_inference_status",
            lambda data, folds, regimens, msm=None: cluster_inference_status(
                data.cluster, cross_fit=folds.n_folds > 1
            ),
        )
        with pytest.raises(AssertionError):
            assert_withholds(fit_end_of_study(FEW_CLUSTER_THRESHOLD, **ONE_ZERO_MASS_CLUSTER), FEW)

    def test_a_replay_without_the_status_fails_the_truncation_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stamp belongs in the builders the replay shares, and not in ``fit`` alone."""
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1)
        refit = longitudinal_estimator._refit_bound

        def unstamped(*args: Any, **kwargs: Any) -> Any:
            replay = refit(*args, **kwargs)
            estimates = {
                name: replace(estimate, inference="influence_curve")
                for name, estimate in replay.estimates.items()
            }
            return replace(replay, estimates=estimates)

        monkeypatch.setattr(longitudinal_estimator, "_refit_bound", unstamped)
        with pytest.raises(CapabilityError, match="longitudinal_replay_fitted_bound_mismatch"):
            result.diagnostics.truncation_curve(bounds=[0.05])


class TestAnOlderLongitudinalArtifact:
    """A result saved before the status loads under the status its data determines."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_few_cluster_fit_saved_before_the_status_loads_under_it(self, route: str) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD - 1, simultaneous=True)
        restored = assert_restamped(result, FEW, route)
        # The re-stamp and the replay read the same data and folds, so the replay matches.
        curve = restored.diagnostics.truncation_curve(bounds=[0.05])
        assert len(curve) == len(result.estimates)

    @pytest.mark.parametrize("route", ROUTES)
    @pytest.mark.parametrize(
        ("k", "status"),
        [(None, "influence_curve"), (FEW_CLUSTER_THRESHOLD - 1, FEW)],
        ids=("unclustered", "39 clusters"),
    )
    def test_data_saved_before_its_weights_field_loads(
        self, k: int | None, status: str, route: str
    ) -> None:
        """An artifact from before weights existed retains its reports and replay."""
        legacy = legacy_copy(fit_end_of_study(k))
        if k is None:
            legacy.__dict__["simultaneous"] = None
        legacy.data.__dict__["_template"] = as_frame(legacy.to_frame())
        for field in ("weights", "weights_name", "weight_spec", "backend"):
            del legacy.data.__dict__[field]
        # The nonzero witness: the saved data really lacks the required fields.
        assert (
            not {"weights", "weights_name", "weight_spec", "backend"} & legacy.data.__dict__.keys()
        )
        restored = restore(legacy, route)
        assert restored.inference_status == status
        assert restored.simultaneous is None
        assert np.array_equal(restored.data.weights, np.ones(N))
        assert restored.data.weights_name is None
        assert restored.data.is_weighted is False
        assert restored.data.backend == "pandas"
        assert "_template" not in restored.data.__dict__
        assert "Longitudinal TMLE" in restored.summary()
        assert len(restored.to_frame()) == len(restored.estimates)
        restored.validate()
        restored.assess()
        restored.diagnostics.run_all()
        replay = restored.diagnostics.truncation_curve(bounds=[0.05])
        assert len(replay) == len(restored.estimates)

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_forty_cluster_artifact_loads_as_saved(self, route: str) -> None:
        result = fit_end_of_study(FEW_CLUSTER_THRESHOLD, simultaneous=True)
        legacy = legacy_copy(result)
        restored = restore(legacy, route)
        assert restored.inference_status == "influence_curve"
        assert restored.simultaneous == legacy.simultaneous
        assert restored.assessment_cache == legacy.assessment_cache
        for name, estimate in restored.estimates.items():
            assert estimate.ci == result[name].ci


SAVED_GROUPED = "cross_fitted_longitudinal_plugin"


def fit_saved_grouped_shape(monkeypatch: pytest.MonkeyPatch, shape: str) -> Any:
    """Build a pre-refusal result shape, then let the restore path decide its status."""
    frame, _ = make_longitudinal(n=N, seed=0)
    if shape == "unequal":
        cluster = np.repeat(np.arange(40), np.tile([5, 15], 20))
    else:
        count = 20 if shape == "few" else 40
        cluster = labels(N, count)
    frame = frame.assign(cluster=cluster)
    # Releases 0.1.0 and 0.1.1 accepted this design. The patch restores that entry
    # point only for the witness; current public fits still refuse it before fitting.
    monkeypatch.setattr(LTMLE, "_refuse_cross_fitted_design", lambda self, data: None)
    return LTMLE(
        {"always": 1, "never": 0},
        reference="never",
        **learners(n_folds=5, simultaneous=True),
    ).fit(frame, outcome="Y", **NODES, id="cluster")


class TestASavedCrossFittedClusteredResult:
    """RM29: every saved grouped longitudinal fit withholds an unsupported interval."""

    @pytest.mark.parametrize("route", ROUTES)
    @pytest.mark.parametrize("shape", ["equal", "few", "unequal"])
    def test_restored_artifact_takes_its_own_status(
        self, monkeypatch: pytest.MonkeyPatch, shape: str, route: str
    ) -> None:
        result = fit_saved_grouped_shape(monkeypatch, shape)
        if shape == "unequal":
            assert set(np.bincount(result.data.cluster)) == {5, 15}
        restored = assert_restamped(result, SAVED_GROUPED, route)
        assert_withholds(restored, SAVED_GROUPED)
        assert_assessment_note(restored, SAVED_GROUPED)
        assert NO_SIMULTANEOUS_BANDS in restored.summary()
        replay = restored.diagnostics.truncation_curve(bounds=[0.05])
        assert len(replay) == len(restored.estimates)

    def test_skipping_the_grouped_decision_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = fit_saved_grouped_shape(monkeypatch, "equal")
        monkeypatch.setattr(
            longitudinal_estimator,
            "_inference_status",
            lambda data, folds, regimens, msm=None: cluster_inference_status(
                data.cluster, cross_fit=folds.n_folds > 1
            ),
        )
        with pytest.raises(AssertionError):
            assert_restamped(result, SAVED_GROUPED, "pickle")

    def test_supported_artifacts_keep_their_intervals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        in_sample = fit_end_of_study(FEW_CLUSTER_THRESHOLD, simultaneous=True)
        assert_keeps_inference(restore(in_sample, "pickle"))
        frame, _ = make_longitudinal(n=N, seed=0)
        unclustered = LTMLE(
            {"always": 1, "never": 0},
            reference="never",
            **learners(n_folds=5, simultaneous=True),
        ).fit(frame, outcome="Y", **NODES)
        assert_keeps_inference(restore(unclustered, "pickle"))
