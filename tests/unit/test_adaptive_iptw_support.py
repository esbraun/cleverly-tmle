r"""Support-key plumbing of AdaptiveIPTW (plan 2.9, RM42b).

The key carries the engine and the ``form`` option, so a row of one form never covers a fit of
the other. The rows themselves wait for the registered study ``adaptive-iptw``, which this change
does not add: a row needs a registered evidence slug
(``test_every_evidence_slug_is_a_registered_study``). Until the rows land, every estimate takes
the ``experimental`` label with the reason that no registered study covers it. This module
asserts the plumbing and the table of configurations that stay experimental by design.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import _support, _support_rows
from cleverly.data import CausalData
from cleverly.estimators import AdaptiveIPTW
from tests.unit import _capability_sweep_support as sweep
from tests.unit import _known_mechanism_support as known


def _estimator(**overrides: Any) -> AdaptiveIPTW:
    settings: dict[str, Any] = {
        "treatment_learner": LogisticRegression(max_iter=1000),
        "reduced_outcome_learner": LinearRegression(),
        "missingness_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return AdaptiveIPTW(**settings)


def _fit(estimator: AdaptiveIPTW, frame: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("outcome", "Y")
    kwargs.setdefault("treatment", "A")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, **kwargs).single()


class TestKeyPlumbing:
    @pytest.fixture(scope="class")
    def data(self) -> CausalData:
        return CausalData.from_frame(
            sweep._linear_frame(), outcome="Y", treatment="A", covariates=["W1", "W2"]
        )

    def test_the_engine_is_the_assessment_method(self, data: CausalData) -> None:
        key, _ = _support.point_key(_estimator(), data, ("ey1", "ey0"))
        assert key.engine == ("adaptive_iptw",)

    def test_the_form_is_an_option_of_the_key(self, data: CausalData) -> None:
        targeted, _ = _support.point_key(_estimator(), data, ("ey1", "ey0"))
        one_step, _ = _support.point_key(_estimator(form="one_step"), data, ("ey1", "ey0"))
        assert dict(targeted.options)["form"] == "targeted"
        assert dict(one_step.options)["form"] == "one_step"
        assert targeted != one_step

    def test_the_default_options_of_the_engine_name_the_targeted_form(self) -> None:
        assert _support_rows.default_options("adaptive_iptw")["form"] == "targeted"
        # The shipped engines carry no form option, so no shipped row changes.
        for engine in ("tmle", "drtmle", "collaborative_tmle"):
            assert "form" not in _support_rows.default_options(engine)

    def test_the_reduced_learner_stays_outside_the_key_and_inside_the_allowlist(
        self, data: CausalData
    ) -> None:
        assert "reduced_outcome_learner" in _support_rows.OPTION_ALLOWLIST
        cited = (
            Path(__file__).resolve().parents[2]
            / _support_rows.OPTION_ALLOWLIST["reduced_outcome_learner"]
        )
        assert cited.is_file()
        one, _ = _support.point_key(_estimator(), data, ("ey1",))
        other, _ = _support.point_key(
            _estimator(reduced_outcome_learner=LogisticRegression()), data, ("ey1",)
        )
        assert one == other

    def test_a_known_mechanism_is_a_qualifier_of_the_key(self) -> None:
        frame = known.point_frame()
        data = CausalData.from_frame(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            treatment_probabilities=known.mechanism(frame),
        )
        key, _ = _support.point_key(_estimator(), data, ("ey1", "ey0"))
        assert "known_mechanism" in key.qualifiers


class TestExperimentalByDesign:
    """Configurations that no row of the plan covers carry the experimental label."""

    def _assert_experimental(self, result: Any) -> None:
        assert result.estimates
        for estimate in result.estimates.values():
            assert estimate.support.label == "experimental"
            assert estimate.support.reason

    def test_a_one_step_form_with_an_estimated_mechanism(self) -> None:
        self._assert_experimental(_fit(_estimator(form="one_step"), sweep._linear_frame()))

    def test_a_known_mechanism_with_a_missing_outcome(self) -> None:
        frame = sweep._missing_frame()
        probability = np.where(frame["W1"].to_numpy() > 0, 0.7, 0.35)
        result = _fit(
            _estimator(),
            frame,
            delta="Delta",
            treatment_probabilities=np.column_stack([1.0 - probability, probability]),
        )
        self._assert_experimental(result)

    def test_a_treatment_adjustment(self) -> None:
        result = _fit(
            _estimator(),
            sweep._linear_frame(),
            covariates=["W1", "W2"],
            treatment_covariates=["W1"],
        )
        self._assert_experimental(result)

    def test_declared_estimated_weights_with_a_known_mechanism(self) -> None:
        frame = known.point_frame()
        result = _fit(
            _estimator(),
            frame,
            covariates=["W1", "W2"],
            weights="wt",
            weights_estimated=True,
            treatment_probabilities=known.mechanism(frame),
        )
        assert result.inference_status == "estimated_weight_plugin"
        self._assert_experimental(result)

    def test_an_in_sample_natural_course_fit(self) -> None:
        self._assert_experimental(sweep.fit_iptw_natural_course())

    def test_a_ratio_target_with_an_estimated_mechanism(self) -> None:
        frame = known.point_frame()
        result = _fit(_estimator(estimands=["rr", "or"]), frame, covariates=["W1", "W2"])
        self._assert_experimental(result)

    def test_the_reason_names_the_nearest_row_and_the_differing_axes(self) -> None:
        result = _fit(_estimator(), sweep._linear_frame(), covariates=["W1", "W2"])
        reason = result.estimates["ey1"].support.reason
        assert "No registered study covers this configuration" in reason
        assert "differing on" in reason
