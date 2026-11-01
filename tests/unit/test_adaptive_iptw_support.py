r"""Support rows of AdaptiveIPTW (plan 2.9, RM42b).

The key carries the engine, the ``form`` option and the learner families, so a row of one form
never covers a fit of the other, and a row of parametric learners never covers a boosted or default
fit. The rows of the registered study ``adaptive-iptw`` license an estimate where a gated
cell passed. A known mechanism licenses ``wald`` on a cross-fitted fit, and an estimated mechanism
licenses ``point`` only, because the estimator withholds the interval there. Any other
configuration stays ``experimental``, and the tables below name each one.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

from cleverly import _support, _support_rows
from cleverly.data import CausalData
from cleverly.datasets.synthetic import make_binary_outcome, make_missing_outcome_binary
from cleverly.estimators import TMLE, AdaptiveIPTW
from cleverly.learners import GLM
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


ARMS = ["ey0", "ey1", "ate"]


def _binary() -> Any:
    frame, _ = make_binary_outcome(n=600, seed=3)
    return frame


def _binary_missing() -> Any:
    frame, _ = make_missing_outcome_binary(n=600, seed=3)
    return frame


def _known_fit(**overrides: Any) -> Any:
    """A cross-fitted fit of the binary point law with the declared mechanism."""
    frame = known.point_frame()
    settings = {"cross_fit": True, "n_folds": 3, "estimands": ARMS} | overrides
    return _fit(
        _estimator(**settings),
        frame,
        covariates=["W1", "W2"],
        treatment_probabilities=known.mechanism(frame),
    )


def _row(row_id: str) -> Any:
    return next(row for row in _support_rows.POINT_ROWS if row.id == row_id)


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

    def test_the_default_options_of_the_engine_name_the_default_fit(self) -> None:
        options = _support_rows.default_options("adaptive_iptw")
        assert (
            options.items()
            >= {
                "form": "targeted",
                "mechanism_learner": "adaptive",
                "reduced_learner": "selecting",
            }.items()
        )
        # The shipped engines carry none of these options, so no shipped row changes.
        for engine in ("tmle", "drtmle", "collaborative_tmle"):
            options = _support_rows.default_options(engine)
            assert not {"form", "mechanism_learner", "reduced_learner"} & set(options)

    def test_another_engine_s_key_reads_no_learner_family(self, data: CausalData) -> None:
        estimator = TMLE(
            outcome_learner=HistGradientBoostingRegressor(),
            treatment_learner=HistGradientBoostingClassifier(),
            cross_fit=False,
        )
        key, _ = _support.point_key(estimator, data, ("ey1",))
        assert not {"mechanism_learner", "reduced_learner"} & set(dict(key.options))

    def test_the_reduced_learner_object_stays_in_the_allowlist_and_its_family_in_the_key(
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
        assert dict(one.options)["reduced_learner"] == "fixed_basis"
        # Mutation control: a boosted reduced learner and the default both select.
        for reduced in (HistGradientBoostingRegressor(), None):
            boosted, _ = _support.point_key(
                _estimator(reduced_outcome_learner=reduced), data, ("ey1",)
            )
            assert dict(boosted.options)["reduced_learner"] == "selecting", reduced

    @pytest.mark.parametrize(
        ("treatment", "family"),
        [
            (LogisticRegression(max_iter=1000), "parametric"),
            (GLM(), "parametric"),
            (HistGradientBoostingClassifier(), "adaptive"),
            (LogisticRegressionCV(), "adaptive"),
            (None, "adaptive"),
        ],
    )
    def test_the_mechanism_learner_family_is_an_option_of_the_key(
        self, data: CausalData, treatment: Any, family: str
    ) -> None:
        key, _ = _support.point_key(_estimator(treatment_learner=treatment), data, ("ey1",))
        assert dict(key.options)["mechanism_learner"] == family

    def test_a_declared_marker_reads_as_parametric(self) -> None:
        class Marked(HistGradientBoostingClassifier):
            cleverly_learner_family = "parametric"

        assert _support.LEARNER_FAMILY_MARKER == "cleverly_learner_family"
        assert _support.learner_family(Marked()) == "parametric"
        assert _support.learner_family(HistGradientBoostingClassifier()) == "adaptive"

    def test_an_adaptive_missingness_learner_makes_the_mechanism_adaptive(self) -> None:
        data = CausalData.from_frame(_binary_missing(), outcome="Y", treatment="A", delta="Delta")
        parametric, _ = _support.point_key(_estimator(), data, ("ey1",))
        boosted, _ = _support.point_key(
            _estimator(missingness_learner=HistGradientBoostingClassifier()), data, ("ey1",)
        )
        assert dict(parametric.options)["mechanism_learner"] == "parametric"
        assert dict(boosted.options)["mechanism_learner"] == "adaptive"

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

    def test_a_known_mechanism_fitted_in_sample(self) -> None:
        """The calibration cells are cross-fitted, so an in-sample known fit has no row."""
        frame = known.point_frame()
        result = _fit(
            _estimator(estimands=ARMS),
            frame,
            covariates=["W1", "W2"],
            treatment_probabilities=known.mechanism(frame),
        )
        self._assert_experimental(result)

    def test_a_boosted_mechanism_matches_no_row(self) -> None:
        """The study reads a bias of 0.7 to 1.2 SD for a boosted mechanism on law P."""
        estimator = _estimator(
            treatment_learner=HistGradientBoostingClassifier(max_iter=20),
            cross_fit=True,
            n_folds=3,
            estimands=ARMS,
        )
        result = _fit(estimator, _binary())
        self._assert_experimental(result)
        assert "options" in result.estimates["ey1"].support.reason

    def test_a_default_learner_fit_matches_no_row(self) -> None:
        """The default Super Learner fills both roles; the key alone decides, without a fit."""
        data = CausalData.from_frame(_binary(), outcome="Y", treatment="A")
        for settings in (
            {"treatment_learner": None, "reduced_outcome_learner": None},
            {"reduced_outcome_learner": None},
        ):
            estimator = _estimator(cross_fit=True, n_folds=3, estimands=ARMS, **settings)
            key, learners = _support.point_key(estimator, data, tuple(ARMS))
            classified = _support.classify_key(key, learners)
            assert classified.label == "experimental", settings
            assert classified.row is None, settings

    def test_a_default_reduced_learner_with_a_known_mechanism_matches_no_wald_row(self) -> None:
        frame = known.point_frame()
        data = CausalData.from_frame(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            treatment_probabilities=known.mechanism(frame),
        )
        estimator = _estimator(
            reduced_outcome_learner=None, cross_fit=True, n_folds=3, estimands=ARMS
        )
        key, learners = _support.point_key(estimator, data, tuple(ARMS))
        assert dict(key.options)["mechanism_learner"] == "known"
        assert _support.classify_key(key, learners).row is None

    def test_the_reason_names_the_nearest_row_and_the_differing_axes(self) -> None:
        result = _fit(_estimator(), sweep._linear_frame(), covariates=["W1", "W2"])
        reason = result.estimates["ey1"].support.reason
        assert "No registered study covers this configuration" in reason
        assert "differing on" in reason


class TestSupportedRows:
    """A configuration that a row of the study covers carries the label ``supported``."""

    def _assert_supported(self, result: Any, row: str) -> None:
        assert result.estimates
        for estimate in result.estimates.values():
            assert estimate.support.label == "supported", estimate.support.reason
            assert estimate.support.row == row

    @pytest.mark.parametrize(
        ("settings", "row"),
        [
            ({}, "AI.points.in_sample"),
            ({"cross_fit": True, "n_folds": 3}, "AI.points.cross_fit"),
        ],
    )
    def test_an_estimated_mechanism_on_a_complete_binary_outcome(
        self, settings: dict[str, Any], row: str
    ) -> None:
        result = _fit(_estimator(estimands=ARMS, **settings), _binary())
        assert result.inference_status == "estimated_mechanism_plugin"
        self._assert_supported(result, row)

    @pytest.mark.parametrize(
        ("settings", "row"),
        [
            ({}, "AI.points_missing_outcome.in_sample"),
            ({"cross_fit": True, "n_folds": 3}, "AI.points_missing_outcome.cross_fit"),
        ],
    )
    def test_an_estimated_mechanism_with_a_missing_outcome(
        self, settings: dict[str, Any], row: str
    ) -> None:
        result = _fit(_estimator(estimands=ARMS, **settings), _binary_missing(), delta="Delta")
        self._assert_supported(result, row)

    def test_a_glm_fit_stays_supported(self) -> None:
        estimator = _estimator(
            treatment_learner=GLM(),
            reduced_outcome_learner=GLM(),
            cross_fit=True,
            n_folds=3,
            estimands=ARMS,
        )
        self._assert_supported(_fit(estimator, _binary()), "AI.points.cross_fit")

    def test_a_cross_fitted_known_mechanism_reads_the_interval_from_the_targeted_row(
        self,
    ) -> None:
        result = _known_fit()
        assert result.inference_status == "influence_curve"
        self._assert_supported(result, "AI.known_arms.targeted")

    def test_a_ratio_target_with_a_known_mechanism_reads_the_arm_row(self) -> None:
        result = _known_fit(estimands=["ey1", "rr"])
        self._assert_supported(result, "AI.known_arms.targeted")

    def test_the_one_step_form_with_a_known_mechanism_reads_its_own_row(self) -> None:
        self._assert_supported(_known_fit(form="one_step"), "AI.known_arms.one_step")


ESTIMATED_ROWS = [row for row in _support_rows.POINT_ROWS if row.id.startswith("AI.points")]
KNOWN_ROWS = [row for row in _support_rows.POINT_ROWS if row.id.startswith("AI.known")]


class TestKindsOfTheRows:
    """Each row licenses the kinds that its gated cells measured."""

    def test_every_row_names_the_study_and_the_engine(self) -> None:
        for row in (*ESTIMATED_ROWS, *KNOWN_ROWS):
            assert row.evidence == ("adaptive-iptw",), row.id
            assert row.key.engine == ("adaptive_iptw",), row.id

    def test_an_estimated_mechanism_row_licenses_the_point_only(self) -> None:
        assert len(ESTIMATED_ROWS) == 14
        for row in ESTIMATED_ROWS:
            assert row.kinds == {"point"}, row.id
            assert "known_mechanism" not in row.key.qualifiers, row.id
            assert any("plug-in" in limit for limit in row.limits), row.id

    def test_a_known_mechanism_row_licenses_the_wald_interval(self) -> None:
        assert len(KNOWN_ROWS) == 7
        for row in KNOWN_ROWS:
            assert {"point", "wald"} <= row.kinds, row.id
            assert "known_mechanism" in row.key.qualifiers, row.id

    def test_only_the_targeted_arm_row_licenses_the_band(self) -> None:
        with_band = {row.id for row in KNOWN_ROWS if "band" in row.kinds}
        assert with_band == {"AI.known_arms.targeted"}

    def test_the_arm_rows_license_the_ratio_targets_and_the_regime_rows_do_not(self) -> None:
        for form in ("targeted", "one_step"):
            row = _row(f"AI.known_arms.{form}")
            assert {"P1", "P2"} <= row.key.family and "derived" in row.kinds
        assert not _row("AI.known_regimes.targeted").key.family & {"P1", "P2"}

    def test_a_form_never_shares_a_row_with_the_other(self) -> None:
        forms = {row.id: dict(row.key.options)["form"] for row in KNOWN_ROWS}
        assert forms["AI.known_arms.one_step"] == "one_step"
        assert forms["AI.known_regimes.one_step"] == "one_step"
        assert {forms[name] for name in forms if not name.endswith("one_step")} == {"targeted"}

    def test_a_row_names_the_pending_comparator_and_the_learners_of_its_cells(self) -> None:
        for row in (*ESTIMATED_ROWS, *KNOWN_ROWS):
            assert any("drtmle comparator runs are pending" in limit for limit in row.limits)
            assert any("super learner" in limit for limit in row.limits), row.id

    def test_every_cross_fitted_row_states_the_open_epsilon_path_condition(self) -> None:
        for row in (*ESTIMATED_ROWS, *KNOWN_ROWS):
            stated = any("epsilon-path condition of S5" in limit for limit in row.limits)
            assert stated == (row.key.splitting == ("stacked",)), row.id
            # The fold-wise reduced regression removes term [B], so no row states it.
            assert not any("term [B]" in limit for limit in row.limits), row.id

    def test_every_row_needs_the_learner_families_of_its_cells(self) -> None:
        for row in ESTIMATED_ROWS:
            options = dict(row.key.options)
            assert options["mechanism_learner"] == "parametric", row.id
            assert options["reduced_learner"] == "fixed_basis", row.id
        for row in KNOWN_ROWS:
            options = dict(row.key.options)
            assert options["mechanism_learner"] == "known", row.id
            assert options["reduced_learner"] == "fixed_basis", row.id

    def test_every_cell_of_a_row_is_a_cell_of_the_study(self) -> None:
        from tests.studies import adaptive_iptw_properties as properties

        declared = {f"{cell.property}/{cell.cell}" for cell in properties.declared_cells()}
        for row in (*ESTIMATED_ROWS, *KNOWN_ROWS):
            assert set(row.cells) <= declared, (row.id, sorted(set(row.cells) - declared))
