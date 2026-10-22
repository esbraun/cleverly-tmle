"""Refutation, simulated confounding and the derived E-value on AdaptiveIPTW (X82 part I).

The placebo witness rests on one identity. On the placebo law the exact treatment mechanism is
the constant arm share. With a constant ``g`` the reduced regression has one regressor value,
so the tilt covariate is constant and the tilted mechanism solves to the realized share. The
targeted Horvitz-Thompson mean is then the arm mean of the permuted data, and over every
assignment of the treated count the unweighted ATE averages to exactly 0.
"""

from __future__ import annotations

import importlib
import itertools
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.assessment import assessment_capabilities
from cleverly.estimators import AdaptiveIPTW
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity._derived import _derived_risk_ratio
from cleverly.sensitivity.simulated_confounding import flipped_mechanism
from cleverly.validation import (
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    GaussianAdjustmentOutcome,
    _interpretation,
    refute,
)
from cleverly.validation._interpretation import (
    VERDICT_EVIDENCE,
    CalibrationScope,
    VerdictEvidence,
    fit_configuration,
    resolve,
)

#: The module, not the function that ``cleverly.validation`` re-exports under its name.
refute_module = importlib.import_module("cleverly.validation.refute")

# --------------------------------------------------------------------------- fixtures


def _frame(n: int, seed: int, *, binary: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1, w2, z = rng.normal(size=(3, n))
    p = 1.0 / (1.0 + np.exp(-(0.4 * w1 - 0.3 * w2)))
    a = rng.binomial(1, p)
    if binary:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.3 + 0.6 * a + 0.4 * w1))))
    else:
        y = 1.0 + 0.5 * a + 0.6 * w1 - 0.4 * w2 + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "Z": z, "p0": 1.0 - p, "p1": p})
    return frame, p


def _estimator(*, cross_fit: bool = True, form: str = "targeted", **kwargs: Any) -> AdaptiveIPTW:
    return AdaptiveIPTW(
        treatment_learner=LogisticRegression(max_iter=1000),
        reduced_outcome_learner=LinearRegression(),
        cross_fit=cross_fit,
        n_folds=3,
        form=form,
        random_state=5,
        **kwargs,
    )


def _direct(
    frame: pd.DataFrame,
    p: np.ndarray,
    *,
    known: bool,
    cross_fit: bool = True,
    form: str = "targeted",
    estimands: tuple[str, ...] = ("ey1", "ey0", "ate"),
) -> Any:
    options: dict[str, Any] = {}
    if known:
        options["treatment_probabilities"] = np.c_[1.0 - p, p]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            _estimator(cross_fit=cross_fit, form=form, estimands=list(estimands))
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], **options)
            .single()
        )


def _study_fit(*, known: bool, n: int = 300, seed: int = 11) -> Any:
    frame, _ = _frame(n, seed)
    design = PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        treatment_probabilities={0: "p0", 1: "p1"} if known else None,
    )
    study = CausalStudy(frame, design=design)
    from cleverly import AdaptiveIPTWMethod, CrossFitting, Inference, ModelSpec, Runtime

    method = AdaptiveIPTWMethod(
        models=ModelSpec(
            treatment_learner=LogisticRegression(max_iter=1000),
            outcome_learner=LinearRegression(),
        ),
        cross_fitting=CrossFitting(enabled=False),
        inference=Inference(simultaneous=False),
        runtime=Runtime(random_state=3),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(ATE()).estimate(method=method)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    return _frame(240, 3)


@pytest.fixture(scope="module")
def estimated(data: tuple[pd.DataFrame, np.ndarray]) -> Any:
    return _direct(*data, known=False)


@pytest.fixture(scope="module")
def known(data: tuple[pd.DataFrame, np.ndarray]) -> Any:
    return _direct(*data, known=True)


def _refute(result: Any, test: str, **options: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(result, estimand="ate", tests=(test,), **options).tests[0]


# ------------------------------------------------------------------------- placebo


#: The enumeration law: n = 10 (the data floor) with 5 treated, so C(10, 5) = 252 assignments.
#: With 3 folds no fold holds 5 rows, so every training complement holds both arms.
_N, _TREATED = 10, 5


def _enumeration_fit(*, cross_fit: bool, form: str, weights: np.ndarray | None = None) -> Any:
    rng = np.random.default_rng(23)
    w = rng.normal(size=_N)
    y = 2.0 * w + rng.normal(size=_N)
    a = np.zeros(_N)
    a[:_TREATED] = 1.0
    share = _TREATED / _N
    frame = pd.DataFrame({"Y": y, "A": a, "W": w})
    options: dict[str, Any] = {"treatment_probabilities": np.tile([1 - share, share], (_N, 1))}
    if weights is not None:
        frame["wt"] = weights
        options["weights"] = "wt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            AdaptiveIPTW(
                treatment_learner=LogisticRegression(),
                reduced_outcome_learner=LinearRegression(),
                cross_fit=cross_fit,
                n_folds=3,
                form=form,
                estimands=["ey1", "ey0", "ate"],
                g_bounds=(0.01, 0.99),
                random_state=1,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], **options)
            .single()
        )


def _assignments(stale: tuple[int, ...] = ()) -> list[np.ndarray]:
    """Every assignment of the treated count; ``stale`` rows keep their observed arm."""
    observed = np.zeros(_N)
    observed[:_TREATED] = 1.0
    rows = []
    for treated in itertools.combinations(range(_N), _TREATED):
        assignment = np.zeros(_N)
        assignment[list(treated)] = 1.0
        assignment[list(stale)] = observed[list(stale)]
        if 0 < assignment.sum() < _N:
            rows.append(assignment)
    return rows


def _placebo_ates(
    result: Any, assignments: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, np.ndarray]:
    """Refit through the package's placebo construction, one assignment at a time.

    The permutation is replaced by the enumerated assignment; the construction then adds the
    placebo law's exact mechanism, the arm shares of the observed column.
    """
    estimates, contrasts = [], []
    data = result.data
    y = np.asarray(data.outcome, dtype=float)
    weights = np.ones(data.n) if data.weights is None else np.asarray(data.weights, dtype=float)
    for assignment in assignments:
        monkeypatch.setattr(
            refute_module, "_permuted_treatment", lambda *_, chosen=assignment: chosen
        )
        placebo = refute_module._placebo_data(data, np.random.default_rng(0), None)
        assert placebo.known_treatment is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refit = result.estimator.refit(placebo, random_state=1)
        assert refit["ate"].inference == "influence_curve"
        estimates.append(refit["ate"].psi)
        treated = assignment == 1.0
        contrasts.append(
            np.sum(weights[treated] * y[treated]) / np.sum(weights[treated])
            - np.sum(weights[~treated] * y[~treated]) / np.sum(weights[~treated])
        )
    return np.asarray(estimates), np.asarray(contrasts)


class TestPlaceboEnumeration:
    @pytest.mark.parametrize("form", ["targeted", "one_step"])
    def test_every_in_sample_refit_is_the_arm_mean_contrast_and_the_mean_is_zero(
        self, form: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _enumeration_fit(cross_fit=False, form=form)
        estimates, contrasts = _placebo_ates(result, _assignments(), monkeypatch)
        assert len(estimates) == 252
        np.testing.assert_allclose(estimates, contrasts, rtol=0, atol=1e-12)
        # Unit weights: each unit is treated in half the assignments, so the mean is 0.
        assert abs(float(np.mean(estimates))) < 1e-12

    @pytest.mark.parametrize("form", ["targeted", "one_step"])
    def test_the_cross_fitted_mean_is_zero_by_the_label_swap(
        self, form: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each fold's reduced regression is its own training arm mean, so the tilt varies by
        # fold and a refit is not the arm-mean contrast. The enumeration is balanced (5 of
        # 10) and the placebo shares are equal, so swapping the arm labels maps each
        # assignment to its complement with the negated ATE, and the mean over all 252 is
        # 0. The test reads the swap on 20 pairs.
        result = _enumeration_fit(cross_fit=True, form=form)
        first = _assignments()[:20]
        estimates, _ = _placebo_ates(result, first, monkeypatch)
        swapped, _ = _placebo_ates(result, [1.0 - item for item in first], monkeypatch)
        assert float(np.std(estimates)) > 0.1
        np.testing.assert_allclose(estimates, -swapped, rtol=0, atol=1e-12)

    def test_a_stale_row_placebo_moves_the_mean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mutation control: two rows keep their observed treatment, so the placebo law no
        # longer separates treatment from the outcome and the enumeration mean is nonzero.
        # In sample only: the stale rows change the arm counts, and some cross-fitted
        # training complements would then hold one arm.
        result = _enumeration_fit(cross_fit=False, form="targeted")
        estimates, contrasts = _placebo_ates(result, _assignments(stale=(0, 1)), monkeypatch)
        np.testing.assert_allclose(estimates, contrasts, rtol=0, atol=1e-12)
        assert abs(float(np.mean(estimates))) > 1e-3

    def test_a_weighted_refit_is_the_weighted_arm_mean_contrast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A weighted arm mean is a ratio, so its permutation mean is not exactly 0. Only the
        # per-refit identity holds.
        weights = np.linspace(0.5, 2.0, _N)
        result = _enumeration_fit(cross_fit=False, form="targeted", weights=weights)
        estimates, contrasts = _placebo_ates(result, _assignments()[:12], monkeypatch)
        np.testing.assert_allclose(estimates, contrasts, rtol=0, atol=1e-12)

    def test_a_refit_that_drops_the_known_mechanism_changes_the_status(self, known: Any) -> None:
        # Mutation control of the known-mechanism placebo: a refit that re-estimates g
        # becomes an estimated-mechanism fit and withholds its interval.
        placebo = known.data.with_treatment(
            refute_module._permuted_treatment(known.data, np.random.default_rng(1))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refit = known.estimator.refit(placebo, random_state=1)
        assert refit["ate"].inference == "estimated_mechanism_plugin"


class TestPlaceboReport:
    def test_an_estimated_mechanism_placebo_runs_with_the_t_rule(self, estimated: Any) -> None:
        test = _refute(estimated, "placebo")
        assert test.passed is None and test.rung == "L0"
        assert test.statistic_name == "t" and np.isfinite(test.statistic)
        labels = [label for label, _ in test.extras]
        assert labels == ["mean plug-in SE over SD of the draws"]

    def test_a_known_mechanism_placebo_names_the_reported_se(self, known: Any) -> None:
        test = _refute(known, "placebo")
        assert [label for label, _ in test.extras] == ["mean reported SE over SD of the draws"]
        assert "placebo law's own arm shares" in test.detail


# ------------------------------------------------------------- random common cause


class TestRandomCommonCause:
    def test_a_known_mechanism_fit_does_not_move(self, known: Any) -> None:
        # Complete data and a declared g: no fitted function reads the noise column.
        test = _refute(known, "random_common_cause")
        assert abs(test.statistic) < 1e-12
        np.testing.assert_allclose(test.values, known["ate"].psi, rtol=0, atol=1e-12)

    def test_an_estimated_mechanism_fit_moves(self, estimated: Any) -> None:
        # The nonzero witness of the pair: the logistic g reads the noise column.
        test = _refute(estimated, "random_common_cause")
        assert test.passed is None and abs(test.statistic) > 1e-6


# -------------------------------------------------------------------------- subset


class TestSubset:
    def test_a_cross_fitted_subset_keeps_each_unit_in_its_fold(self, estimated: Any) -> None:
        index = np.arange(0, estimated.data.n, 2)
        sliced = refute_module._sliced_folds(estimated, index)
        assert sliced is not None
        original = np.asarray(estimated.repeats[0].folds.assignment)[index]
        relabelled = np.asarray(sliced[0].assignment)
        # The same partition of the kept rows, under compact labels.
        for label in np.unique(original):
            assert np.unique(relabelled[original == label]).size == 1

    def test_an_estimated_mechanism_reference_is_named_a_diagnostic(self, estimated: Any) -> None:
        test = _refute(estimated, "subset")
        assert test.passed is None
        assert "the fit reports no interval" in test.detail

    def test_a_known_mechanism_reference_is_not(self, known: Any) -> None:
        test = _refute(known, "subset")
        assert test.passed is None and np.isfinite(test.statistic)
        assert "the fit reports no interval" not in test.detail


# ------------------------------------------------------------------ negative control


class TestNegativeControl:
    @pytest.mark.parametrize("kind", [None, "pre_treatment"])
    def test_the_control_fit_equals_a_direct_refit(
        self, data: tuple[pd.DataFrame, np.ndarray], estimated: Any, kind: str | None
    ) -> None:
        frame, _ = data
        control = np.asarray(frame["Z"], dtype=float)
        test = _refute(
            estimated,
            "negative_control_outcome",
            negative_control_outcome=control,
            negative_control_kind=kind,
        )
        replacement = estimated.data.with_outcome(control, name="negative_control_outcome")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = estimated.estimator.refit(replacement, random_state=5)
        assert test.values == (direct["ate"].psi,)
        assert "bounds its outcome" not in test.detail
        assert "plug-in standard errors" in test.detail

    def test_a_known_mechanism_control_reads_standard_errors(
        self, data: tuple[pd.DataFrame, np.ndarray], known: Any
    ) -> None:
        test = _refute(
            known,
            "negative_control_outcome",
            negative_control_outcome=np.asarray(data[0]["Z"], dtype=float),
        )
        assert "z = " in test.detail and "plug-in" not in test.detail


# ----------------------------------------------------------------- bootstraps


class TestBootstraps:
    def test_bootstrap_stability_runs(self, estimated: Any) -> None:
        test = _refute(estimated, "bootstrap_stability", n_replicates=8)
        assert test.passed is None and len(test.values) == 8

    def test_measurement_error_holds_the_declared_mechanism(self, known: Any) -> None:
        declaration = BootstrapMeasurementError(variables=("W1",))
        drawn = known.data.subset(np.arange(0, known.data.n, 3))
        perturbed = refute_module._perturb_measurement_error(drawn, declaration, seed=4)
        assert perturbed.known_treatment is not None
        np.testing.assert_array_equal(
            perturbed.known_treatment.values, known.data.known_treatment.values[::3]
        )
        test = _refute(
            known,
            "bootstrap_measurement_error",
            bootstrap_measurement_error=declaration,
            measurement_error_rule=EmpiricalInclusionRule(minimum_draws=40),
            n_replicates=40,
        )
        assert test.passed is None and len(test.values) == 40


# ------------------------------------------------------------ generated outcomes


@pytest.fixture(scope="module")
def study_estimated() -> Any:
    return _study_fit(known=False)


@pytest.fixture(scope="module")
def study_known() -> Any:
    return _study_fit(known=True)


class TestGeneratedOutcomes:
    @pytest.mark.parametrize("test_name", ["simulated_outcome", "plasmode"])
    def test_an_estimated_mechanism_reads_bias_only(
        self, study_estimated: Any, test_name: str
    ) -> None:
        test = _refute(study_estimated, test_name, n_replicates=40)
        assert [reading.name for reading in test.readings] == ["bias"]
        assert "the fit reports no interval: estimated_mechanism_plugin" in test.detail

    @pytest.mark.parametrize("test_name", ["simulated_outcome", "plasmode"])
    def test_a_known_mechanism_reads_all_three(self, study_known: Any, test_name: str) -> None:
        # Mutation control of the withholding: the same law with an interval prints all.
        assert study_known["ate"].inference == "influence_curve"
        test = _refute(study_known, test_name, n_replicates=40)
        assert [reading.name for reading in test.readings] == ["bias", "se_ratio", "coverage"]
        assert "reports no interval" not in test.detail

    def test_the_plasmode_keeps_the_known_mechanism_on_every_draw(self, study_known: Any) -> None:
        law = refute_module._PlasmodeLaw.fit(study_known)
        draw = law.draw(study_known.data, GaussianAdjustmentOutcome(), seed=3)
        assert draw.known_treatment is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refit = study_known.estimator.refit(draw, random_state=3)
        assert refit["ate"].inference == "influence_curve"
        assert refit.estimator.form == study_known.estimator.form


# ---------------------------------------------------------------- empty registry


def test_an_empty_registry_leaves_every_reading_descriptive(
    study_estimated: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_interpretation, "VERDICT_EVIDENCE", ())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = refute(
            study_estimated,
            estimand="ate",
            tests=("placebo", "random_common_cause", "subset", "simulated_outcome"),
            n_replicates=12,
        )
    assert all(test.passed is None for test in report.tests)
    assert report.passed is None


# ------------------------------------------------------------- scope of a fit


class TestFitConfiguration:
    def test_the_reduced_learner_is_the_fingerprinted_one(
        self, data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        # The reduced regression takes reduced_outcome_learner, else outcome_learner. A fit
        # that names the same learner either way runs the same regression, so the two
        # fingerprints agree; a different reduced learner differs. The pre-part-I read took
        # outcome_learner only, which reads "default-library" on the first fit.
        frame, _ = data
        learner = LinearRegression(fit_intercept=False)
        options: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
        fits = []
        for names in (
            {"reduced_outcome_learner": learner},
            {"outcome_learner": learner},
            {"reduced_outcome_learner": LinearRegression()},
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits.append(
                    AdaptiveIPTW(
                        treatment_learner=LogisticRegression(max_iter=1000),
                        cross_fit=False,
                        estimands=["ey1", "ey0", "ate"],
                        random_state=5,
                        **names,
                    )
                    .fit(frame, **options)
                    .single()
                )
        reduced, outcome, other = (fit_configuration(fit, "ate").learners for fit in fits)
        assert reduced == outcome
        assert reduced != other
        assert "default-library" not in reduced

    def test_the_form_is_on_the_method_axis(self, data: tuple[pd.DataFrame, np.ndarray]) -> None:
        frame, p = data
        targeted = fit_configuration(_direct(frame, p, known=False), "ate").method
        one_step = fit_configuration(_direct(frame, p, known=False, form="one_step"), "ate")
        assert targeted == "AdaptiveIPTW"
        assert one_step.method == "AdaptiveIPTW[one_step]"

    def test_a_ctmle_strategy_row_does_not_cover_another_strategy(self) -> None:
        from cleverly.estimators import CTMLE

        frame, _ = _frame(200, 9)
        fits = {}
        for strategy in ("greedy", "oat"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits[strategy] = (
                    CTMLE(
                        strategy=strategy,
                        outcome_learner=LinearRegression(),
                        treatment_learner=LogisticRegression(max_iter=1000),
                        cross_fit=False,
                        estimands=["ey1", "ey0", "ate"],
                        random_state=1,
                    )
                    .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                    .single()
                )
        greedy = fit_configuration(fits["greedy"], "ate")
        oat = fit_configuration(fits["oat"], "ate")
        assert greedy.method == "CTMLE[greedy]"
        assert oat.method == "CTMLE[oat:per_arm]"
        row = VerdictEvidence(
            instrument="placebo",
            rule="t",
            rung="L2",
            theory="witness",
            scope=CalibrationScope(methods=frozenset({"CTMLE[greedy]"})),
            sentence="covered",
            study=("refuter-calibration-point", "C1"),
        )
        assert resolve("placebo", "t", replace(greedy, replicates=20), evidence=(row,)).licensed
        assert not resolve("placebo", "t", replace(oat, replicates=20), evidence=(row,)).licensed

    def test_no_shipped_row_scopes_a_variant_class_by_its_bare_name(self) -> None:
        # method_axis never reads a bare "CTMLE", so a row scoped on it would cover no fit.
        # A row must name the variant its cells ran, for example "CTMLE[greedy]".
        bare = {"CTMLE"}
        for row in VERDICT_EVIDENCE:
            assert not (row.scope.methods or frozenset()) & bare, row


# ------------------------------------------------------ simulated confounding


class TestSimulatedConfounding:
    def test_the_zero_cell_is_the_fit_and_a_flip_moves_it(self, study_estimated: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = simulated_confounding(
                study_estimated,
                "ate",
                grid=ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0,)),
                random_state=1,
            )
        cells = {cell.treatment_strength: cell for cell in report.cells}
        assert cells[0.0].estimate == study_estimated["ate"].psi
        assert abs(cells[0.3].estimate - study_estimated["ate"].psi) > 1e-6

    def test_a_known_mechanism_cell_carries_the_flipped_mechanism(
        self, study_known: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strength = 0.3
        captured: list[Any] = []
        original = AdaptiveIPTW.refit

        def spy(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            captured.append(data)
            return original(self, data, *args, **kwargs)

        monkeypatch.setattr(AdaptiveIPTW, "refit", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = simulated_confounding(
                study_known,
                "ate",
                grid=ConfounderStrengthGrid(treatment=(0.0, strength), outcome=(0.0,)),
                random_state=1,
            )
        cell = next(cell for cell in report.cells if cell.treatment_strength == strength)
        (flipped,) = captured
        np.testing.assert_allclose(
            flipped.known_treatment.values,
            flipped_mechanism(study_known.data.known_treatment.values, strength),
            rtol=0,
            atol=1e-15,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = study_known.estimator.refit(flipped, random_state=3)
        assert direct["ate"].inference == "influence_curve"
        assert cell.estimate == pytest.approx(direct["ate"].psi, abs=1e-12)


# ---------------------------------------------------------------- derived E-value


class TestDerivedRiskRatio:
    @pytest.mark.parametrize("known_mechanism", [True, False], ids=["known", "estimated"])
    @pytest.mark.parametrize("cross_fit", [True, False], ids=["cross_fit", "in_sample"])
    def test_the_retarget_equals_a_same_seed_ratio_fit(
        self, known_mechanism: bool, cross_fit: bool
    ) -> None:
        frame, p = _frame(400, 21, binary=True)
        source = _direct(frame, p, known=known_mechanism, cross_fit=cross_fit, estimands=("ate",))
        target = _direct(
            frame, p, known=known_mechanism, cross_fit=cross_fit, estimands=("ate", "rr")
        )
        derived = _derived_risk_ratio(source, "ate")
        assert derived.psi == pytest.approx(target["rr"].psi, rel=1e-10)
        assert derived.plugin_std_error == pytest.approx(target["rr"].plugin_std_error, rel=1e-10)
        assert derived.inference == target["rr"].inference


# --------------------------------------------------------------- capability rows


@pytest.mark.parametrize("operation", ["refute"])
def test_the_capability_row_reads_available(known: Any, operation: str) -> None:
    (row,) = (row for row in assessment_capabilities(known) if row.operation == operation)
    assert row.available and row.refusal is None, row
