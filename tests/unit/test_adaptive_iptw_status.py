r"""Inference status of AdaptiveIPTW: which fits report an interval, and why the others do not.

* W18: the status of each construction, and what the accessors of an estimate do under it.
* W19: the validity identity behind the status text, on a four-point discrete law.

A fit reports a standard error, an interval and a p-value only when the whole mechanism of every
reported key is known (complete data and a declared ``treatment_probabilities=``). Every other
fit takes ``estimated_mechanism_plugin``, and the accessors raise
:class:`~cleverly.exceptions.CapabilityError` with the record's reason. The plug-in spread stays
readable under ``plugin_std_error`` and ``plugin_interval``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._inference_status import NO_SIMULTANEOUS_BANDS, STATUS_RECORDS
from cleverly.estimators import AdaptiveIPTW
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import Rule, Static, Stochastic
from tests.unit import _adaptive_iptw_support as s
from tests.unit import _capability_sweep_support as sweep
from tests.unit import _known_mechanism_support as known_mechanism

PLUGIN = "estimated_mechanism_plugin"
WEIGHT_PLUGIN = "estimated_weight_plugin"
CURVE = "influence_curve"


def _point_fit(
    *,
    known: bool,
    estimands: list[str] | None = None,
    frame: pd.DataFrame | None = None,
    covariates: list[str] | None = None,
    **extra: Any,
) -> Any:
    """A fit on the known-mechanism law (or ``frame``), with a declared or estimated mechanism."""
    data = known_mechanism.point_frame() if frame is None else frame
    settings: dict[str, Any] = {
        "treatment_learner": LogisticRegression(max_iter=1000),
        "reduced_outcome_learner": LinearRegression(),
        "missingness_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    fit_args: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": covariates}
    if estimands is not None:
        settings["estimands"] = estimands
    if known:
        fit_args["treatment_probabilities"] = known_mechanism.mechanism(data)
    for key in ("weights", "weights_estimated", "delta", "treatment_covariates"):
        if key in extra:
            fit_args[key] = extra.pop(key)
    settings.update(extra)
    if covariates is None and frame is None:
        fit_args["covariates"] = ["W1", "W2"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AdaptiveIPTW(**settings).fit(data, **fit_args).single()


def _accessors(estimate: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("std_error", "ci", "pvalue"):
        try:
            out[name] = getattr(estimate, name)
        except CapabilityError as error:
            out[name] = error
    out["plugin_std_error"] = estimate.plugin_std_error
    out["plugin_interval"] = estimate.plugin_interval
    return out


def assert_reports_inference(result: Any, status: str = CURVE) -> None:
    assert result.inference_status == status
    for estimate in result.estimates.values():
        values = _accessors(estimate)
        assert not any(isinstance(value, Exception) for value in values.values())
        assert np.isfinite(values["std_error"]) and values["std_error"] > 0
        assert np.isfinite(values["plugin_std_error"])


def assert_withholds_inference(result: Any) -> None:
    assert result.inference_status == PLUGIN
    record = STATUS_RECORDS[PLUGIN]
    for estimate in result.estimates.values():
        values = _accessors(estimate)
        for name in ("std_error", "ci", "pvalue"):
            error = values[name]
            assert isinstance(error, CapabilityError), name
            assert record.reason in str(error)
            assert error.refusal.reason is RefusalReason.INFERENCE_GAP
            assert error.refusal.roadmap == "X36"
        assert np.isfinite(values["plugin_std_error"]) and values["plugin_std_error"] > 0
        low, high = values["plugin_interval"]
        assert low < high


# ------------------------------------------------------------------------------------- W18


class TestW18StatusRouting:
    def test_a_known_mechanism_on_complete_data_reports_inference_for_an_arm_fit(self) -> None:
        assert_reports_inference(_point_fit(known=True))

    def test_a_known_mechanism_reports_inference_for_a_regime_fit(self) -> None:
        frame = s.law_frame()
        probabilities = np.column_stack([1.0 - s.G_TILDE, s.G_TILDE])[
            frame["W"].to_numpy(dtype=int)
        ]
        regimes = [
            Static(0, name="never"),
            Rule(
                lambda data: np.rint(data["W"].to_numpy()).astype(int) % 2,
                name="rule",
                rule_kind="known",
            ),
            Stochastic(
                lambda data: np.column_stack(
                    [
                        1.0 - s.G[np.rint(data["W"].to_numpy()).astype(int)],
                        s.G[np.rint(data["W"].to_numpy()).astype(int)],
                    ]
                ),
                name="tilt",
                density_kind="known",
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = s.fit_arm(
                frame,
                interventions=regimes,
                estimands=None,
                fit={"treatment_probabilities": probabilities},
            )
        assert_reports_inference(result)

    def test_an_estimated_mechanism_withholds_inference_and_keeps_the_plugin_spread(self) -> None:
        assert_withholds_inference(_point_fit(known=False))

    def test_a_known_mechanism_with_a_missing_outcome_is_not_known(self) -> None:
        frame = sweep._missing_frame()
        known = np.where(frame["W1"].to_numpy() > 0, 0.7, 0.35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                AdaptiveIPTW(
                    treatment_learner=LogisticRegression(max_iter=1000),
                    missingness_learner=LogisticRegression(max_iter=1000),
                    reduced_outcome_learner=LinearRegression(),
                    cross_fit=False,
                    simultaneous=False,
                    random_state=0,
                )
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    delta="Delta",
                    treatment_probabilities=np.column_stack([1.0 - known, known]),
                )
                .single()
            )
        assert_withholds_inference(result)

    def test_the_known_mechanism_with_a_missing_outcome_and_ey_obs_alone_is_not_known(
        self,
    ) -> None:
        frame = sweep._missing_frame()
        known = np.where(frame["W1"].to_numpy() > 0, 0.7, 0.35)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                AdaptiveIPTW(
                    missingness_learner=LogisticRegression(max_iter=1000),
                    reduced_outcome_learner=LinearRegression(),
                    estimands=["ey_obs"],
                    cross_fit=False,
                    simultaneous=False,
                    random_state=0,
                )
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    delta="Delta",
                    treatment_probabilities=np.column_stack([1.0 - known, known]),
                )
                .single()
            )
        assert_withholds_inference(result)

    def test_the_natural_course_on_complete_data_is_the_closed_form(self) -> None:
        frame = sweep._linear_frame()
        stack = _point_fit(
            known=False, estimands=["ey_obs", "ey0", "par"], frame=frame, covariates=None
        )
        assert_withholds_inference(stack)
        assert "ey_obs" in stack.estimates
        alone = _point_fit(known=False, estimands=["ey_obs"], frame=frame, covariates=None)
        assert_reports_inference(alone)

    def test_a_non_inferential_fit_prints_the_band_line_and_builds_no_band(self) -> None:
        result = _point_fit(known=False, simultaneous=True)
        assert result.inference_status == PLUGIN
        assert result.simultaneous is None
        assert NO_SIMULTANEOUS_BANDS in result.summary()
        known = _point_fit(known=True, simultaneous=True)
        assert known.simultaneous is not None
        assert NO_SIMULTANEOUS_BANDS not in known.summary()

    def test_estimated_weights_do_not_rescue_an_estimated_mechanism(self) -> None:
        result = _point_fit(known=False, weights="wt", weights_estimated=True)
        assert result.inference_status == PLUGIN

    def test_a_treatment_adjustment_with_an_estimated_mechanism_is_not_known(self) -> None:
        result = _point_fit(known=False, treatment_covariates=["W1"])
        assert result.inference_status == PLUGIN

    def test_a_known_mechanism_and_a_treatment_adjustment_are_refused_before_the_fit(self) -> None:
        """The plan expected ``estimated_mechanism_plugin``. The data layer refuses the pair."""
        with pytest.raises(DataError, match="treatment_covariates= restricts a fitted"):
            _point_fit(known=True, treatment_covariates=["W1"])

    def test_estimated_weights_with_a_known_mechanism_take_the_weight_plugin(self) -> None:
        result = _point_fit(known=True, weights="wt", weights_estimated=True)
        assert result.inference_status == WEIGHT_PLUGIN
        for estimate in result.estimates.values():
            assert np.isfinite(estimate.std_error)
            assert len(estimate.ci) == 2
        assert "experimental" in result.summary()

    def test_estimated_weights_on_complete_data_leave_a_closed_form_alone(self) -> None:
        result = _point_fit(known=False, estimands=["ey_obs"], weights="wt", weights_estimated=True)
        assert_reports_inference(result)


# ------------------------------------------------------------------------------------- W19


class TestW19ValidityIdentity:
    """WP302's second-order term, on a discrete law, equals the linear term of the verifier.

    ``Qr_{0,n}`` regresses the outcome of the treated on both ``g_0`` and ``g_n``. Because
    ``(g_0 - g_n) / g_0`` is a function of the pair, the term equals
    ``P_0[(Q_0 - Qr_0)(g_0 - g_n)/g_0]`` exactly, and that is linear in ``g_n - g_0``.
    """

    P_W = np.array([0.20, 0.30, 0.30, 0.20])
    G0 = np.array([0.30, 0.30, 0.60, 0.60])
    Q0 = np.array([0.20, 0.60, 0.30, 0.70])

    @staticmethod
    def _regression(mass: np.ndarray, q: np.ndarray, *classes: np.ndarray) -> np.ndarray:
        keys = list(zip(*[np.round(c, 12) for c in classes], strict=True))
        out = np.empty(len(q))
        for key in set(keys):
            rows = np.array([k == key for k in keys])
            out[rows] = np.sum(mass[rows] * q[rows]) / np.sum(mass[rows])
        return out

    def _terms(self, g_n: np.ndarray) -> tuple[float, float]:
        mass = self.P_W * self.G0
        qr_0 = self._regression(mass, self.Q0, self.G0)
        qr_0n = self._regression(mass, self.Q0, self.G0, g_n)
        factor = (self.G0 - g_n) / self.G0
        wp302 = float(np.sum(self.P_W * (qr_0n - qr_0) * factor))
        linear = float(np.sum(self.P_W * (self.Q0 - qr_0) * factor))
        return wp302, linear

    def test_the_law_has_tied_g_and_a_distinct_outcome_regression_in_each_pair(self) -> None:
        mass = self.P_W * self.G0
        qr_0 = self._regression(mass, self.Q0, self.G0)
        assert self.G0[0] == self.G0[1] and self.G0[2] == self.G0[3]
        assert np.min(np.abs(self.Q0 - qr_0)) > 0.1

    def test_a_tie_breaking_mechanism_gives_a_term_equal_to_the_linear_form(self) -> None:
        g_n = np.array([0.25, 0.35, 0.55, 0.70])
        wp302, linear = self._terms(g_n)
        assert wp302 == pytest.approx(linear, abs=1e-14)
        assert abs(wp302) > 1e-3

    def test_a_mechanism_that_is_a_function_of_g0_gives_a_zero_term(self) -> None:
        g_n = 0.1 + 0.5 * self.G0
        wp302, linear = self._terms(g_n)
        assert abs(wp302) < 1e-14
        assert abs(linear) < 1e-14

    def test_the_term_is_linear_in_the_perturbation(self) -> None:
        h = np.array([1.0, -1.0, 0.5, 2.0])
        mass = self.P_W * self.G0
        qr_0 = self._regression(mass, self.Q0, self.G0)
        limit = -float(np.sum(self.P_W * (self.Q0 - qr_0) * (1.0 - self.G0) * h))
        assert abs(limit) > 1e-3
        ratios = []
        for delta in (1e-3, 2e-3):
            g_n = expit(logit(self.G0) + delta * h)
            ratios.append(self._terms(g_n)[0] / delta)
        # A linear term has a ratio that tends to the limit. A quadratic term would give a ratio
        # that tends to zero and doubles with delta.
        assert ratios[0] == pytest.approx(limit, abs=5e-3)
        assert ratios[1] == pytest.approx(limit, abs=1e-2)

    def test_the_status_text_states_the_condition_the_identity_pins(self) -> None:
        reason = STATUS_RECORDS[PLUGIN].reason
        assert "linear" in reason
        assert "function of the true g" in reason
