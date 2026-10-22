"""X82 part D: the assessment family and the refute adapter of the structural nested mean model.

The exact-law witness runs the package's own g-estimation on an enumerated two-node law. Each
support point is a row, weighted by its probability, and every nuisance is a saturated lookup
of the weighted cell mean, so the fit is the population g-estimate.
"""

from __future__ import annotations

import importlib
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.assessment import (
    ASSESSMENT_CAPABILITIES,
    AssessmentStatus,
    DiagnosticsFacade,
    assessment_capabilities,
)
from cleverly.datasets import make_longitudinal
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.learners import GLM, SuperLearner
from cleverly.longitudinal import StructuralNestedMean
from cleverly.validation import _interpretation, refute
from cleverly.validation import _structural_refute as structural_refute
from cleverly.validation.refute import refute_refusal

BLIP = "blip[A1:1.0:intercept]"
CONTRAST = "ate_regimen[never vs always]"


# ----------------------------------------------------------------------- fixtures


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    frame, _ = make_longitudinal(n=n, seed=seed, censoring=False)
    return frame


def _fit(frame: pd.DataFrame, **settings: object) -> object:
    options: dict[str, object] = {
        "outcome_learner": GLM(family="binomial"),
        "treatment_learner": GLM(family="binomial"),
        "n_folds": 2,
        "random_state": 0,
        "simultaneous": False,
    }
    options.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuralNestedMean({"always": 1, "never": 0}, **options).fit(  # type: ignore[arg-type]
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
        )


@pytest.fixture(scope="module")
def structural_result() -> object:
    return _fit(_frame())


# ------------------------------------------------------------ exact-law witness


class _Lookup(BaseEstimator):
    """The weighted mean of ``y`` in each distinct row of ``X``: the exact conditional mean."""

    def fit(self, X, y, sample_weight=None):  # type: ignore[no-untyped-def]
        design = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        weight = np.ones(target.size) if sample_weight is None else np.asarray(sample_weight)
        self.table_ = {}
        for key in {tuple(row) for row in design}:
            rows = np.all(design == np.asarray(key), axis=1)
            self.table_[key] = float(np.average(target[rows], weights=weight[rows]))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.array([self.table_[tuple(row)] for row in np.asarray(X, dtype=float)])

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


def _observed_law() -> pd.DataFrame:
    """Two binary nodes; ``L2`` reads ``A1``; ``A2`` reads ``A1`` and ``L2``.

    ``Y = 1 + W + 0.5 A1 + 0.8 L2 + 0.6 A2``, so the node-2 blip is 0.6 and the node-1 blip
    is ``0.5 + 0.8 * 0.3 = 0.74`` (the effect of ``A1`` through ``L2``, with ``A2 = 0``).
    """
    rows = []
    for w, a1, l2, a2 in itertools.product((0, 1), repeat=4):
        p = 0.5
        p_a1 = 0.3 + 0.4 * w
        p *= p_a1 if a1 else 1.0 - p_a1
        p_l2 = 0.2 + 0.3 * w + 0.3 * a1
        p *= p_l2 if l2 else 1.0 - p_l2
        p_a2 = 0.2 + 0.3 * a1 + 0.3 * l2
        p *= p_a2 if a2 else 1.0 - p_a2
        y = 1.0 + w + 0.5 * a1 + 0.8 * l2 + 0.6 * a2
        rows.append({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y, "weight": p})
    return pd.DataFrame(rows)


def _placebo_law(observed: pd.DataFrame, *, stale: bool = False) -> pd.DataFrame:
    """The law ``P(O^-) x K``: every non-treatment column kept, trajectories from ``K``.

    ``K`` is the first-order kernel of the observed treatments. With ``stale``, the drawn
    node-2 treatment reads the observed ``A2``, which ``Y`` reads: a leak of the original
    treatment, which the placebo must not have.
    """
    weight = observed["weight"]
    p1 = float(np.average(observed["A1"], weights=weight))
    p21 = {
        a: float(
            np.average(observed["A2"][observed["A1"] == a], weights=weight[observed["A1"] == a])
        )
        for a in (0, 1)
    }
    rows = []
    for _, unit in observed.iterrows():
        for a1, a2 in itertools.product((0, 1), repeat=2):
            p = p1 if a1 else 1.0 - p1
            p_a2 = 0.2 + 0.6 * unit["A2"] if stale else p21[a1]
            p *= p_a2 if a2 else 1.0 - p_a2
            rows.append(
                {
                    "W": unit["W"],
                    "A1": a1,
                    "L2": unit["L2"],
                    "A2": a2,
                    "Y": unit["Y"],
                    "weight": unit["weight"] * p,
                }
            )
    return pd.DataFrame(rows)


def _population_fit(law: pd.DataFrame, features: object = None) -> object:
    estimator = StructuralNestedMean(
        {"always": 1, "never": 0},
        blip_features=features,  # type: ignore[arg-type]
        outcome_learner=_Lookup(),
        treatment_learner=_Lookup(),
        regression_learner=_Lookup(),
        n_folds=1,
        simultaneous=False,
        min_eigenvalue=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            law,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            weights="weight",
            family="gaussian",
        )


def test_blips_are_exactly_zero_on_the_enumerated_placebo_law() -> None:
    observed = _observed_law()
    # Nonzero witness: the same construction recovers the blips of the observed law.
    fitted = _population_fit(observed)
    assert fitted["blip[A1:1:intercept]"].psi == pytest.approx(0.74, abs=1e-10)
    assert fitted["blip[A2:1:intercept]"].psi == pytest.approx(0.60, abs=1e-10)
    assert fitted["ate_regimen[never vs always]"].psi == pytest.approx(-1.34, abs=1e-10)
    placebo = _population_fit(_placebo_law(observed))
    for name in ("blip[A1:1:intercept]", "blip[A2:1:intercept]", "ate_regimen[never vs always]"):
        assert abs(placebo[name].psi) < 1e-12, name
        assert structural_refute.structural_null(placebo, name) == 0.0
    # Levels are equal to each other but not to any constant: no fixed null.
    assert placebo["ey_regimen[always]"].psi == pytest.approx(placebo["ey_reference"].psi)
    assert structural_refute.structural_null(placebo, "ey_reference") is None


def test_blips_with_history_features_are_exactly_zero_on_the_placebo_law() -> None:
    features = {1: ["W"], 2: ["A1"]}
    placebo = _population_fit(_placebo_law(_observed_law()), features)
    blips = [name for name in placebo if placebo.kinds[name] == "blip"]
    assert len(blips) == 4
    for name in blips:
        assert abs(placebo[name].psi) < 1e-12, name


def test_a_stale_treatment_placebo_moves_the_blip() -> None:
    # Mutation control: a draw that reads the observed treatment leaks its effect.
    leaked = _population_fit(_placebo_law(_observed_law(), stale=True))
    assert abs(leaked["blip[A2:1:intercept]"].psi) > 0.3


# ---------------------------------------------------------------- declarations


def test_every_operation_runs_or_refuses_with_a_code(structural_result) -> None:  # type: ignore[no-untyped-def]
    rows = assessment_capabilities(structural_result)
    assert {row.operation for row in rows} == {
        "support",
        "nuisance_models",
        "score_equations",
        "corrections",
        "truncation_curve",
        "refute",
    }
    assert all(row.methods == ("snmm",) for row in rows)
    facades = (DiagnosticsFacade(structural_result), structural_result.sensitivity)
    for facade in facades:
        for row in facade.capabilities:
            if row.operation == "refute":
                # The bare request names no estimand, so the row defers on it, and a call
                # that names one runs.
                assert row.status is AssessmentStatus.DEFERRED
                assert row.requires_arguments == ("estimand",)
                report = facade.refute(estimand=BLIP, tests=("subset",), n_replicates=2)
                assert report.tests[0].name == "subset"
                continue
            assert not row.available
            assert isinstance(row.refusal.reason, RefusalReason)
            assert row.reason
            if row.refusal.reason is RefusalReason.IMPLEMENTATION_GAP:
                assert row.refusal.roadmap in {"F13", "F16"}
            with pytest.raises(CapabilityError) as raised:
                getattr(facade, row.operation)()
            assert raised.value.refusal.reason is row.refusal.reason


def test_structural_rows_redirect_to_the_fit_diagnostics(structural_result) -> None:  # type: ignore[no-untyped-def]
    rows = {row.operation: row for row in assessment_capabilities(structural_result)}
    for operation in ("support", "nuisance_models"):
        alternative = rows[operation].refusal.alternative
        assert rows[operation].refusal.reason is RefusalReason.API_REDIRECTION
        for attribute in alternative.removeprefix("Read ").rstrip(".").split(" and "):
            assert hasattr(structural_result, "diagnostics")
            name = attribute.removeprefix("result.diagnostics.")
            assert hasattr(structural_result.diagnostics, name), name
    for operation in ("score_equations", "corrections", "truncation_curve"):
        assert rows[operation].status is AssessmentStatus.NOT_APPLICABLE


def test_the_sensitivity_rows_follow_the_matrix(structural_result) -> None:  # type: ignore[no-untyped-def]
    rows = {row.operation: row for row in structural_result.sensitivity.capabilities}
    assert rows["simulated_confounding"].refusal.roadmap == "F13"
    for operation in ("omitted_confounding", "evalue", "benchmark", "contour"):
        assert rows[operation].refusal.roadmap == "F16"
    for operation in ("missingness", "tipping_gamma"):
        assert rows[operation].status is AssessmentStatus.NOT_APPLICABLE
    # An explicit estimand lifts no structural row.
    with pytest.raises(CapabilityError, match="structural nested mean"):
        structural_result.sensitivity.evalue(estimand=CONTRAST)


def test_the_capability_table_declares_one_row_per_operation_for_structural() -> None:
    structural = [row for row in ASSESSMENT_CAPABILITIES if row.result_family == "structural"]
    assert len(structural) == len({row.operation for row in structural}) == 6


def test_replayability_offers_the_rerun_only(structural_result) -> None:  # type: ignore[no-untyped-def]
    replay = structural_result.replayability
    assert replay.rerun_configuration
    assert not replay.refit_nuisances
    assert not replay.retarget_cached_nuisances


# ------------------------------------------------------------------ refit hook


def test_refit_with_the_fit_seed_reproduces_an_unseeded_fit() -> None:
    result = _fit(_frame(), random_state=None)
    assert result.config.fit_seed is not None
    again = result.estimator.refit(
        result.data,
        frozen_folds=structural_refute.structural_frozen_folds(result),
        frozen_seeds=structural_refute.structural_fit_seeds(result),
    )
    for name in result:
        assert again[name].psi == result[name].psi
    assert np.array_equal(again.folds.assignment, result.folds.assignment)


def test_refit_refuses_more_than_one_split(structural_result) -> None:  # type: ignore[no-untyped-def]
    folds = structural_result.folds
    with pytest.raises(ValueError, match="one split"):
        structural_result.estimator.refit(structural_result.data, frozen_folds=(folds, folds))


class _Columns(BaseEstimator):
    """A linear learner on fixed columns, so it ignores an appended noise column.

    A target in {0, 1} takes a logistic model, and any other target a linear one.
    """

    def __init__(self, columns: tuple[int, ...] = (0,)) -> None:
        self.columns = columns

    def _design(self, X):  # type: ignore[no-untyped-def]
        return np.asarray(X, dtype=float)[:, list(self.columns)]

    def fit(self, X, y, sample_weight=None):  # type: ignore[no-untyped-def]
        target = np.asarray(y, dtype=float)
        self.classifier_ = bool(np.all((target == 0.0) | (target == 1.0)))
        model = LogisticRegression() if self.classifier_ else LinearRegression()
        self.model_ = model.fit(self._design(X), target, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        if self.classifier_:
            return self.model_.predict_proba(self._design(X))[:, 1]
        return self.model_.predict(self._design(X))

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


def _seeded_ensemble_fit() -> object:
    """An unseeded fit whose ensembles inherit the drawn fit seed for their inner folds."""
    ensemble = SuperLearner(library=[_Columns((0,)), _Columns((1,))], n_folds=3)
    return _fit(
        _frame(),
        random_state=None,
        outcome_learner=ensemble,
        treatment_learner=ensemble,
        regression_learner=ensemble,
    )


def test_the_paired_shift_is_exactly_zero_when_the_noise_is_unused() -> None:
    result = _seeded_ensemble_fit()
    report = refute(result, estimand=BLIP, tests=("random_common_cause",), n_replicates=3)
    (test,) = report.tests
    assert test.statistic == 0.0
    assert all(value == result[BLIP].psi for value in test.values)


def test_the_paired_shift_sees_seed_noise_without_the_fit_seed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Mutation control: refits that do not reuse the fit seed redraw the ensembles' inner
    # folds, and the shift is no longer exactly 0.
    result = _seeded_ensemble_fit()
    module = importlib.import_module("cleverly.validation.refute")
    monkeypatch.setattr(module, "structural_fit_seeds", lambda _result: None)
    report = refute(
        result, estimand=BLIP, tests=("random_common_cause",), n_replicates=3, random_state=11
    )
    (test,) = report.tests
    assert test.statistic != 0.0


# --------------------------------------------------------------------- refute


def test_refute_runs_every_structural_test_descriptively(structural_result) -> None:  # type: ignore[no-untyped-def]
    report = refute(
        structural_result,
        estimand=CONTRAST,
        tests=("placebo", "random_common_cause", "subset", "bootstrap_stability"),
        n_replicates=3,
        noise_nodes=True,
    )
    assert [test.name for test in report.tests] == [
        "placebo",
        "random_common_cause",
        "subset",
        "bootstrap_stability",
    ]
    assert all(test.passed is None for test in report.tests)
    placebo = report.tests[0]
    assert placebo.expectation == "~ 0"
    assert "every blip and every regimen contrast is exactly 0" in placebo.detail


def test_an_empty_registry_leaves_every_structural_verdict_open(
    structural_result, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(_interpretation, "VERDICT_EVIDENCE", ())
    report = refute(structural_result, estimand=BLIP, tests=("placebo", "subset"), n_replicates=3)
    assert all(test.passed is None for test in report.tests)


def test_the_structural_configuration_reads_the_fit(structural_result) -> None:  # type: ignore[no-untyped-def]
    fit = _interpretation.fit_configuration(structural_result, BLIP)
    assert fit.method == "StructuralNestedMean"
    assert fit.estimand_kind == "blip"
    assert fit.nodes == 2
    assert fit.cross_fitting == "folds=2"
    assert fit.outcome_family == "binomial"
    assert fit.min_event_count is not None
    # The unset regression learner resolves to the outcome learner, not the default library.
    assert fit.learners[2] == fit.learners[0]
    assert "default-library" not in fit.learners


def test_a_level_refuses_the_placebo_before_any_refit(structural_result, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def forbidden(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("no refit may run")

    monkeypatch.setattr(type(structural_result.estimator), "refit", forbidden)
    for level in ("ey_reference", "ey_regimen[always]"):
        with pytest.raises(CapabilityError, match="no-effect value"):
            refute(structural_result, estimand=level, tests=("placebo",))
    # A level still runs the stability tests.
    assert refute_refusal(structural_result, estimand="ey_reference", tests=("subset",)) is None


@pytest.mark.parametrize(
    ("tests", "options"),
    [
        (("dummy_outcome",), {}),
        (("future_treatment",), {"future_treatment_control": np.zeros(300)}),
        (("bootstrap_measurement_error",), {}),
    ],
)
def test_tests_without_a_structural_adapter_refuse(structural_result, tests, options) -> None:  # type: ignore[no-untyped-def]
    refusal = refute_refusal(structural_result, estimand=BLIP, tests=tests, **options)
    assert refusal is not None
    assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert refusal.roadmap == "X82"


def test_a_point_placebo_mode_is_refused(structural_result) -> None:  # type: ignore[no-untyped-def]
    refusal = refute_refusal(
        structural_result, estimand=BLIP, tests=("placebo",), placebo_mode="permute"
    )
    assert refusal is not None
    assert "Markov kernel" in refusal.condition


def test_subset_keeps_each_unit_in_its_fold(structural_result) -> None:  # type: ignore[no-untyped-def]
    index = np.arange(0, structural_result.data.n, 3)
    (folds,) = structural_refute.structural_frozen_folds(structural_result, index)
    original = np.asarray(structural_result.folds.assignment)[index]
    assert np.array_equal(np.asarray(folds.assignment), original)


def test_a_thin_subset_refuses_before_the_first_refit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    result = _fit(_frame(n=80))
    calls = []
    original = type(result.estimator).refit

    def counting(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(result.estimator), "refit", counting)
    with pytest.raises(CapabilityError, match="subset_fraction"):
        refute(result, estimand=BLIP, tests=("subset",), subset_fraction=0.05, n_replicates=3)
    assert not calls


def test_subset_reads_the_plugin_error_of_an_in_sample_adaptive_fit() -> None:
    # An in-sample fit with a learner other than GLM reports no interval. The subset reading
    # still runs on its plug-in standard error, and states no verdict.
    result = _fit(
        _frame(),
        n_folds=1,
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
    )
    assert result[BLIP].inference == "adaptive_baseline_plugin"
    report = refute(result, estimand=BLIP, tests=("subset",), n_replicates=3)
    (test,) = report.tests
    assert test.passed is None
    assert np.isfinite(test.statistic)


# ----------------------------------------------------------- negative control


def _controls(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """A null control and a control that tracks the first treatment beyond its history."""
    rng = np.random.default_rng(5)
    first = frame["A1"].to_numpy(dtype=float)
    null = (rng.random(len(frame)) < 0.4).astype(float)
    confounded = (rng.random(len(frame)) < 0.3 + 0.4 * first).astype(float)
    return null, confounded


def test_the_negative_control_reads_a_control_against_the_blip_null(structural_result) -> None:  # type: ignore[no-untyped-def]
    null, confounded = _controls(_frame())
    readings = {}
    for label, control in (("null", null), ("confounded", confounded)):
        report = refute(
            structural_result,
            estimand=BLIP,
            tests=("negative_control_outcome",),
            negative_control_outcome=control,
        )
        (test,) = report.tests
        assert test.passed is None
        assert test.expectation == "~ 0"
        readings[label] = test.statistic
    # A control that A1 moves beyond its history is the nonzero witness: the refit reads the
    # control's own blip, not the primary outcome's.
    assert abs(readings["null"]) < 2.0
    assert readings["confounded"] > 5.0


def test_the_negative_control_refit_keeps_the_split_and_the_seed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    result = _fit(_frame(), random_state=None)
    seen = []
    original = type(result.estimator).refit

    def recording(self, data, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs)
        return original(self, data, **kwargs)

    monkeypatch.setattr(type(result.estimator), "refit", recording)
    null, _ = _controls(_frame())
    refute(
        result, estimand=BLIP, tests=("negative_control_outcome",), negative_control_outcome=null
    )
    (kwargs,) = seen
    assert kwargs["frozen_seeds"] == (result.config.fit_seed,)
    (folds,) = kwargs["frozen_folds"]
    assert np.array_equal(np.asarray(folds.assignment), np.asarray(result.folds.assignment))


def test_a_pre_treatment_control_runs_and_a_baseline_copy_refuses(structural_result) -> None:  # type: ignore[no-untyped-def]
    frame = _frame()
    null, _ = _controls(frame)
    report = refute(
        structural_result,
        estimand=CONTRAST,
        tests=("negative_control_outcome",),
        negative_control_outcome=null,
        negative_control_kind="pre_treatment",
    )
    (test,) = report.tests
    assert "pre-treatment reading" in test.scope_sentence
    # The fit regresses a continuous control, so the copy reaches the adjustment-set check.
    refusal = refute_refusal(
        _fit(frame, regression_learner=GLM()),
        estimand=BLIP,
        tests=("negative_control_outcome",),
        negative_control_outcome=frame["W1"].to_numpy(dtype=float),
        negative_control_kind="pre_treatment",
    )
    assert refusal is not None
    assert "W1" in refusal.condition


def test_a_continuous_control_needs_a_regression_learner(structural_result) -> None:  # type: ignore[no-untyped-def]
    control = np.random.default_rng(1).normal(size=structural_result.data.n)
    refusal = refute_refusal(
        structural_result,
        estimand=BLIP,
        tests=("negative_control_outcome",),
        negative_control_outcome=control,
    )
    assert refusal is not None
    assert refusal.reason is RefusalReason.INVALID_DECLARATION
    assert "regression_learner" in refusal.condition
    result = _fit(_frame(), regression_learner=GLM())
    report = refute(
        result, estimand=BLIP, tests=("negative_control_outcome",), negative_control_outcome=control
    )
    assert np.isfinite(report.tests[0].statistic)


def test_a_level_refuses_the_negative_control(structural_result) -> None:  # type: ignore[no-untyped-def]
    null, _ = _controls(_frame())
    refusal = refute_refusal(
        structural_result,
        estimand="ey_reference",
        tests=("negative_control_outcome",),
        negative_control_outcome=null,
    )
    assert refusal is not None
    assert refusal.reason is RefusalReason.INVALID_DECLARATION
