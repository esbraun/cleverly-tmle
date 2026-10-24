"""Generated outcomes on the held history of a longitudinal fit (X82 part C3a).

The draw keeps ``(W, L, A, C)`` and replaces the outcome by ``f(W) + effect * sum_s A_s +
epsilon``. The outcome reads no ``L``, so a static regimen has an exact sample-conditional
level, and a policy contrast of the independent process is exactly 0.

The witnesses use a nonzero effect, so a dropped treatment term, a wrong sign, or a wrong
horizon fails. The mean of the refits sits on the truth within Monte Carlo error. A level
reads bias only, and a contrast also reads the standard-error ratio and the coverage.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.longitudinal import LTMLE, DynamicRegimen, Repeated
from cleverly.validation import refute
from cleverly.validation.refute import (
    GaussianAdjustmentOutcome,
    GaussianIndependentOutcome,
    GaussianNoise,
    _longitudinal_generated_target,
    refute_refusal,
)

CONTRAST = "ate_regimen[never vs always]"
EFFECT = 0.7
PROCESS = GaussianAdjustmentOutcome(effect=EFFECT)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _panel(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """A two-node Gaussian panel whose censoring at node 2 reads ``L2``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a1 = rng.binomial(1, _expit(w)).astype(float)
    c1 = rng.binomial(1, 0.95, size=n).astype(float)
    l2 = np.where(c1 == 1, rng.normal(size=n) + w + 0.5 * a1, np.nan)
    a2 = np.where(c1 == 1, rng.binomial(1, _expit(np.nan_to_num(l2))), np.nan)
    c2 = np.where(c1 == 1, rng.binomial(1, _expit(2.0 + 0.5 * np.nan_to_num(l2))), np.nan)
    y1 = np.where(c1 == 1, w + a1 + rng.normal(size=n), np.nan)
    y2 = np.where(c2 == 1, w + np.nan_to_num(l2) + np.nan_to_num(a2) + rng.normal(size=n), np.nan)
    return pd.DataFrame(
        {"W": w, "A1": a1, "C1": c1, "Y1": y1, "L2": l2, "A2": a2, "C2": c2, "Y2": y2}
    )


def _ltmle(plans: Any = None, **settings: Any) -> LTMLE:
    options: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "censoring_learner": LogisticRegression(),
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 3,
    }
    options.update(settings)
    return LTMLE({"always": 1, "never": 0} if plans is None else plans, **options)


def _fit(frame: pd.DataFrame, *, repeated: bool = False, **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ltmle(**settings).fit(
            frame,
            outcome=Repeated(["Y1", "Y2"]) if repeated else "Y2",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]] if repeated else [[], ["Y1", "L2"]],
            censoring=["C1", "C2"],
        )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _panel()


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> Any:
    return _fit(frame)


@pytest.fixture(scope="module")
def repeated(frame: pd.DataFrame) -> Any:
    return _fit(frame, repeated=True)


def _run(result: Any, estimand: str, name: str, draws: int = 10, **request: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = refute(
            result,
            estimand=estimand,
            tests=(name,),
            n_replicates=draws,
            random_state=1,
            simulated_outcome=PROCESS,
            **request,
        )
    return report[name]


def _refused(result: Any, estimand: str, name: str, **request: Any) -> CapabilityError:
    """The row and the call refuse ``request`` with one message."""
    arguments = {"tests": (name,), "n_replicates": 10, **request}
    refusal = refute_refusal(result, estimand=estimand, **arguments)
    assert refusal is not None
    with pytest.raises(CapabilityError) as error:
        refute(result, estimand=estimand, **arguments)
    assert error.value.refusal.message() == refusal.message()
    return error.value


# ------------------------------------------------------------------------- the truth


def _shift(result: Any) -> float:
    """The weighted sample mean of ``f(W)`` on the held baseline."""
    return float(np.average(PROCESS.adjustment(result.data.baseline), weights=result.data.weights))


class TestTheTruth:
    @pytest.mark.parametrize(
        ("estimand", "dose"),
        [("ey_regimen[always]", 2.0), ("ey_regimen[never]", 0.0), (CONTRAST, -2.0)],
    )
    def test_an_end_of_study_static_regimen(self, fitted: Any, estimand: str, dose: float) -> None:
        target = _longitudinal_generated_target(fitted, estimand, PROCESS, "simulated_outcome")
        level = estimand != CONTRAST
        assert target.truth == pytest.approx((_shift(fitted) if level else 0.0) + EFFECT * dose)
        # A level carries f(W), so it reads bias only. A contrast cancels f(W).
        assert target.covariate_part is (not level)

    @pytest.mark.parametrize(
        ("estimand", "dose"),
        [
            ("ey_regimen[always @ t=1]", 1.0),
            ("ey_regimen[always @ t=2]", 2.0),
            ("ate_regimen[never vs always @ t=1]", -1.0),
            ("ate_regimen[never vs always @ t=2]", -2.0),
        ],
    )
    def test_a_repeated_outcome_adds_the_nodes_up_to_its_horizon(
        self, repeated: Any, estimand: str, dose: float
    ) -> None:
        target = _longitudinal_generated_target(repeated, estimand, PROCESS, "simulated_outcome")
        level = estimand.startswith("ey")
        assert target.truth == pytest.approx((_shift(repeated) if level else 0.0) + EFFECT * dose)

    def test_a_held_design_adds_its_one_decision_once(self, frame: pd.DataFrame) -> None:
        # The held treatment repeats A1 at node 2, and the process adds the effect once per
        # decision, so the level of "always" carries the effect once.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            held = _ltmle().fit(
                frame,
                outcome="Y2",
                treatment="A1",
                baseline=["W"],
                time_varying=[[], ["Y1", "L2"]],
                censoring=["C1", "C2"],
            )
        estimand = next(key for key in held.estimates if key.startswith("ate"))
        target = _longitudinal_generated_target(held, estimand, PROCESS, "simulated_outcome")
        assert abs(target.truth) == pytest.approx(EFFECT)

    def test_the_independent_process_has_the_noise_mean(self, fitted: Any) -> None:
        process = GaussianIndependentOutcome(GaussianNoise(mean=0.3))
        level = _longitudinal_generated_target(
            fitted, "ey_regimen[always]", process, "dummy_outcome"
        )
        contrast = _longitudinal_generated_target(fitted, CONTRAST, process, "dummy_outcome")
        assert (level.truth, contrast.truth) == (pytest.approx(0.3), 0.0)
        # f is 0, so a level has no covariate part either.
        assert level.covariate_part and contrast.covariate_part

    def test_the_draw_holds_the_history_and_reads_no_covariate(self, fitted: Any) -> None:
        target = _longitudinal_generated_target(fitted, CONTRAST, PROCESS, "simulated_outcome")
        first = target.draw(fitted.data, seed=5, name="simulated_outcome")
        data = fitted.data
        np.testing.assert_array_equal(first.uncensored, data.uncensored)
        np.testing.assert_array_equal(first.treatment, data.treatment)
        for node in (1, 2):
            np.testing.assert_array_equal(
                first.covariate_history(node), data.covariate_history(node)
            )
        observed = ~np.isnan(np.asarray(data.outcome))
        np.testing.assert_array_equal(~np.isnan(np.asarray(first.outcome)), observed)
        # The residual of f(W) + effect * (A1 + A2) is the noise alone: it carries no L2.
        codes = np.nan_to_num(np.asarray(data.treatment, dtype=float))
        residual = (
            np.asarray(first.outcome)[observed]
            - (PROCESS.adjustment(data.baseline) + EFFECT * codes.sum(axis=1))[observed]
        )
        l2 = np.asarray(data.time_varying[1][:, -1])[observed]
        assert abs(np.corrcoef(residual, l2)[0, 1]) < 0.15


class TestTheRefits:
    def test_the_refits_sit_on_the_truth(self, fitted: Any) -> None:
        test = _run(fitted, CONTRAST, "simulated_outcome", draws=40)
        values = np.asarray(test.values)
        error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        assert test.declared_effect == pytest.approx(-2.0 * EFFECT)
        assert abs(float(np.mean(values)) - test.declared_effect) < 4.0 * error
        assert {reading.name for reading in test.readings} == {"bias", "se_ratio", "coverage"}

    def test_a_level_reads_bias_only(self, fitted: Any) -> None:
        test = _run(fitted, "ey_regimen[always]", "simulated_outcome")
        assert [reading.name for reading in test.readings] == ["bias"]
        assert "withheld" in test.detail
        assert "uncensored share by node 0.9" in test.detail

    def test_the_independent_process_reads_a_dynamic_policy(self, frame: pd.DataFrame) -> None:
        plans = {
            "always": 1,
            "never": 0,
            "if_l2": DynamicRegimen(
                "if_l2", (0, lambda history: history["L2"] > 0), rule_kind="known"
            ),
        }
        result = _fit(frame, plans=plans)
        estimand = next(key for key in result.estimates if "if_l2 vs" in key)
        test = _run(result, estimand, "dummy_outcome")
        assert test.declared_effect == 0.0
        error = _refused(result, estimand, "simulated_outcome")
        assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert "static regimen only" in error.refusal.message()

    def test_a_repeated_fit_runs(self, repeated: Any) -> None:
        test = _run(repeated, "ate_regimen[never vs always @ t=1]", "simulated_outcome")
        assert test.declared_effect == pytest.approx(-EFFECT)
        assert len(test.values) == 10


def test_a_longitudinal_fit_reads_descriptive_at_the_calibrated_budget(fitted: Any) -> None:
    # The registered generated-outcome rows were measured on point fits. A longitudinal fit
    # sits outside their scope at 400 draws, and a point configuration inside it.
    from dataclasses import replace

    from cleverly.validation.refute import _read_configuration, _resolve

    configuration = _read_configuration(fitted, CONTRAST)
    assert configuration is not None and configuration.nodes == 2
    for rule in ("bias_band", "se_ratio_band", "coverage_band"):
        assert not _resolve("generated_outcome", rule, configuration, 400).licensed
        point = replace(configuration, nodes=None)
        assert _resolve("generated_outcome", rule, point, 400).licensed


# ------------------------------------------------------------------------- refusals


class TestRefusals:
    def test_a_binary_outcome_is_refused(self, frame: pd.DataFrame) -> None:
        binary = frame.assign(Y2=np.where(frame.Y2.isna(), np.nan, (frame.Y2 > 0).astype(float)))
        result = _fit(binary, outcome_learner=LogisticRegression())
        error = _refused(result, CONTRAST, "dummy_outcome")
        assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    def test_declared_bounds_are_refused(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, n_folds=2, q_bounds=(-10.0, 10.0))
        error = _refused(result, CONTRAST, "dummy_outcome")
        assert "n_folds=1" in error.refusal.message()

    def test_a_classifier_outcome_learner_is_refused(self, fitted: Any) -> None:
        # A Gaussian panel cannot be fitted with a classifier, so the stand-in swaps the
        # stored template's learner after the fit.
        template = copy.copy(fitted.estimator)
        template.outcome_learner = LogisticRegression()
        result = copy.copy(fitted)
        object.__setattr__(result, "estimator", template)
        error = _refused(result, CONTRAST, "dummy_outcome")
        assert "regression-capable" in error.refusal.message()

    def test_a_survival_fit_refuses_the_gaussian_default(self) -> None:
        from tests.unit.test_longitudinal_refutation import _ltmle as _survival_ltmle
        from tests.unit.test_longitudinal_refutation import _survival_panel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _survival_ltmle(n_folds=1).fit(
                _survival_panel(),
                outcome=["Y1", "Y2"],
                treatment=["A1", "A2"],
                baseline=["W"],
                time_varying=[[], ["L2"]],
                censoring=["C1", "C2"],
            )
        estimand = next(key for key in result.estimates if key.startswith("ate"))
        # A survival fit reads a hazard declaration, not the Gaussian default.
        error = _refused(result, estimand, "dummy_outcome")
        assert error.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "LogisticHazardOutcome" in error.refusal.message()
