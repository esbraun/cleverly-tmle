"""``LTMLE(node_bounds=)`` bounds the logistic fluctuation of the named nodes (X33 part (i)).

At a named node ``t`` with bounds ``(l, u)`` the recursion regresses
``Ytilde = (Y_t - l) / (u - l)``, clips the prediction to ``[0, 1]`` and maps it back to
``l + (u - l) p``.  The node fluctuates along ``logit((Q - l) / (u - l)) + epsilon H`` with
outcome ``Ytilde`` (Gruber and van der Laan 2010).  The score equation has a root only when
the weighted mean of ``Ytilde`` over the followers lies in ``(0, 1)``, so the solver checks
that first.

The exact tests run on the laws of ``tests/discrete_law_longitudinal.py``,
``tests/discrete_law_survival.py`` and ``tests/discrete_law_competing.py``, whose samples
realise the law, so a saturated learner returns the oracle regressions.  The bounds
``(0.3, 0.7)`` hold every follower mean of the end-of-study law but not its cell means
(``0.25`` and ``0.75``), so they bind: the unbounded regression leaves them and the bounded
fit differs.  The bounds ``(0.05, 0.95)`` hold the truth, so the bounded fit returns it.

Three mutation controls patch the implementation and assert that a test above fails: the
inverted map, the bounds of node ``t`` applied at node ``t - 1``, and a skipped root check.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import Targeting, TMLEMethod
from cleverly.datasets import make_longitudinal_survival
from cleverly.exceptions import (
    CapabilityError,
    ConvergenceWarning,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.interventions import Incremental
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import sequential as sequential_module
from cleverly.methods import GComputationMethod, IPSIMethod, IPWMethod, SDRMethod

from .. import discrete_law_competing as competing
from .. import discrete_law_longitudinal as law
from .. import discrete_law_survival as survival
from ..studies.canonical_ltmle import declared_regimens
from . import test_longitudinal_msm as msm_module

NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
SURVIVAL_COLUMNS: dict[str, Any] = {**COLUMNS, "outcome": ["Y1", "Y2"]}
COMPETING_COLUMNS: dict[str, Any] = {**COLUMNS, "outcome": competing.outcome_columns()}

#: Holds every follower mean of the law, and binds the saturated regression.
BINDING = (0.3, 0.7)
#: Holds the truth: every cell mean of the law lies inside.
HOLDING = (0.05, 0.95)

#: The pinned warning, written out again here so the library's constant is not its own oracle.
PINNED = (
    "the declared node bound at {t!r} excludes the data: the weighted mean outcome of the "
    "{arm} followers at that node, {m:.4g}, lies outside ({l:g}, {u:g}), so no fluctuation "
    "inside the bound solves the score equation. Widen node_bounds at {t!r}."
)


def _estimator(module: Any, cls: type = LTMLE, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "reference": module.REGIMEN_REFERENCE,
        "outcome_learner": law.CellMeans(),
        "pseudo_learner": law.CellMeans(),
        "treatment_learner": law.CellMeans(),
        "censoring_learner": law.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    return cls(declared_regimens(module.REGIMEN_SPEC), **settings)


def fit_end_of_study(**overrides: Any) -> Any:
    """An LTMLE fit of the end-of-study law with the saturated learner at every node."""
    return _estimator(law, **overrides).fit(law.frame(), **COLUMNS)


def fit_survival(**overrides: Any) -> Any:
    """An LTMLE fit of the survival law."""
    return _estimator(survival, **overrides).fit(survival.frame(), **SURVIVAL_COLUMNS)


def fit_competing(**overrides: Any) -> Any:
    """An LTMLE fit of the competing-risks law."""
    return _estimator(competing, **overrides).fit(competing.frame(), **COMPETING_COLUMNS)


FINITE_NODES: dict[str, Any] = {
    "outcome": ["Y1", "Y2"],
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
#: Hold every follower mean of the finite-sample survival fixture and bind its regressions.
FINITE_BOUNDS = (0.05, 0.45)


def fit_finite(**overrides: Any) -> Any:
    """A finite-sample survival fit with estimated mechanisms, where the nuisances are not exact."""
    frame, _ = make_longitudinal_survival(n=300, seed=1)
    settings: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000, random_state=0),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000, random_state=0),
        "censoring_learner": LogisticRegression(max_iter=1000, random_state=0),
        "learner_folds": 2,
        "simultaneous": False,
        "random_state": 5,
        "n_folds": 1,
        **overrides,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE({"always": 1, "never": 0}, **settings).fit(frame, **FINITE_NODES)


def node_warnings(record: list[warnings.WarningMessage]) -> list[str]:
    """The texts of the bound warnings in ``record``."""
    return [
        str(item.message)
        for item in record
        if issubclass(item.category, ConvergenceWarning)
        and str(item.message).startswith("the declared node bound")
    ]


def steps_of(result: Any, time: int) -> list[Any]:
    """Every regimen's step at node ``time``."""
    return [step for fit in result.fits.values() for step in fit.steps if step.time == time]


def max_error(result: Any, truth: dict[str, float]) -> float:
    """The largest distance from the reported estimates to their oracle values."""
    return max(abs(result.psi(name) - truth[name]) for name in truth)


@pytest.fixture(scope="module")
def baseline() -> Any:
    """The fit without the keyword."""
    return fit_end_of_study()


@pytest.fixture(scope="module")
def binding() -> Any:
    """The fit whose bounds bind both nodes."""
    return fit_end_of_study(node_bounds={1: BINDING, 2: BINDING})


# ------------------------------------------------------------------------------ exact tests


class TestBoundedNodesAreExact:
    """The map, the range of ``Q*`` and the score of a bounded node."""

    @pytest.mark.parametrize("time", [1, 2])
    def test_targeted_prediction_stays_inside_the_bounds(self, binding: Any, time: int) -> None:
        steps = steps_of(binding, time)
        assert steps
        for step in steps:
            targeted = step.targeted[step.at_risk]
            assert targeted.min() >= BINDING[0] - 1e-12
            assert targeted.max() <= BINDING[1] + 1e-12

    @pytest.mark.parametrize("time", [1, 2])
    def test_score_is_solved_to_the_fit_tolerance(self, binding: Any, time: int) -> None:
        data = binding.data
        for step in steps_of(binding, time):
            followers = np.asarray(step.trained_on, dtype=bool)
            residual = np.where(followers, step.pseudo_outcome - step.targeted, 0.0)
            score = float(np.mean(np.asarray(data.weights) * np.asarray(step.clever) * residual))
            assert abs(score) < 1e-9
            assert step.fluctuation.converged
            assert step.fluctuation.failure is None
            assert float(step.fluctuation.score[0]) == pytest.approx(score, abs=1e-12)

    def test_the_score_moved_from_its_starting_value(self, binding: Any) -> None:
        """A witness: the bounded regression leaves a nonzero score for the node to solve."""
        moved = [
            abs(float(step.fluctuation.score_initial[0]))
            for step in steps_of(binding, 2)
            if step.fluctuation.score_initial is not None
        ]
        assert max(moved) > 1e-4

    def test_bounds_that_hold_the_truth_return_it(self) -> None:
        check_exact_truth()

    def test_survival_law_bounds_that_hold_the_truth_return_it(self) -> None:
        held = fit_survival(node_bounds={1: (0.0, 0.95), 2: (0.0, 0.95)})
        assert max_error(held, survival.TRUTH) < 1e-10

    def test_a_cross_fitted_fit_bounds_and_solves_each_pooled_node(self) -> None:
        crossfit = fit_end_of_study(node_bounds={1: BINDING, 2: BINDING}, n_folds=2)
        for time in (1, 2):
            for step in steps_of(crossfit, time):
                targeted = step.targeted[step.at_risk]
                assert targeted.min() >= BINDING[0] - 1e-12
                assert targeted.max() <= BINDING[1] + 1e-12
                followers = np.asarray(step.trained_on, dtype=bool)
                residual = np.where(followers, step.pseudo_outcome - step.targeted, 0.0)
                score = np.mean(
                    np.asarray(crossfit.data.weights) * np.asarray(step.clever) * residual
                )
                assert abs(float(score)) < 1e-9

    @pytest.mark.parametrize(
        "settings",
        [
            {"stopping": "sample_size"},
            {"step_bound": 5.0},
            {"iterate": "best"},
        ],
    )
    def test_the_x22_settings_reach_the_bounded_node(self, settings: dict[str, Any]) -> None:
        """The bounded node honours ``stopping=``, ``step_bound=`` and ``iterate=``."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            ruled = fit_end_of_study(node_bounds={1: BINDING, 2: BINDING}, **settings)
        for step in steps_of(ruled, 2):
            fluctuation = step.fluctuation
            assert fluctuation.failure != "bound_excludes_data"
            if "stopping" in settings:
                assert fluctuation.stopping == settings["stopping"]
                assert fluctuation.thresholds is not None
            if "iterate" in settings:
                assert fluctuation.best_iterate is not None
            targeted = step.targeted[step.at_risk]
            assert targeted.min() >= BINDING[0] - 1e-12
            assert targeted.max() <= BINDING[1] + 1e-12


# ------------------------------------------------------------------------------- root check


def check_root_check() -> None:
    """The bound ``(0, 0.1)`` at node 2 excludes every follower mean of the law."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        excluded = fit_end_of_study(node_bounds={2: (0.0, 0.1)})
    texts = node_warnings(record)
    fits = list(excluded.fits.values())
    assert len(texts) == len(fits)
    for fit in fits:
        node = fit.steps[1]
        assert node.time == 2
        assert node.fluctuation.failure == "bound_excludes_data"
        assert not node.fluctuation.converged
        assert node.fluctuation.n_iter == 0
        assert not fit.converged
        weights = np.asarray(excluded.data.weights) * np.asarray(node.clever)
        mean = float(np.sum(weights * node.pseudo_outcome) / np.sum(weights))
        expected = PINNED.format(t=2, arm=fit.regimen.label, m=mean, l=0.0, u=0.1)
        assert expected in texts
        assert fit.steps[0].fluctuation.failure is None


class TestRootCheck:
    def test_a_bound_that_excludes_the_data_warns_and_reports_the_failure(self) -> None:
        check_root_check()

    def test_the_pinned_text_is_exact_for_one_regimen(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            fit_end_of_study(node_bounds={2: (0.0, 0.1)})
        assert (
            "the declared node bound at 2 excludes the data: the weighted mean outcome of the "
            "never followers at that node, 0.4687, lies outside (0, 0.1), so no fluctuation "
            "inside the bound solves the score equation. Widen node_bounds at 2."
        ) in node_warnings(record)

    def test_the_failure_kind_has_a_description(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            excluded = fit_end_of_study(node_bounds={2: (0.0, 0.1)})
        description = next(iter(excluded.fits.values())).steps[1].fluctuation.describe_failure()
        assert "widen node_bounds" in description


# --------------------------------------------------------------------------------- witness


class TestBoundsChangeTheFit:
    """With exact mechanisms any regression gives the truth, so the witness is a finite sample."""

    @pytest.fixture(scope="class")
    def plain(self) -> Any:
        return fit_finite()

    @pytest.fixture(scope="class")
    def bounded(self) -> Any:
        return fit_finite(node_bounds={1: FINITE_BOUNDS, 2: FINITE_BOUNDS})

    def test_the_unbounded_regression_leaves_the_bounds(self, plain: Any) -> None:
        initial = np.concatenate(
            [step.initial[step.at_risk] for fit in plain.fits.values() for step in fit.steps]
        )
        assert initial.min() < FINITE_BOUNDS[0]
        assert initial.max() > FINITE_BOUNDS[1]

    def test_bounded_and_unbounded_estimates_differ(self, plain: Any, bounded: Any) -> None:
        gaps = [abs(bounded.psi(name) - plain.psi(name)) for name in plain.estimates]
        assert max(gaps) > 1e-4

    def test_the_bounded_fit_keeps_every_prediction_inside_and_solves_every_score(
        self, bounded: Any
    ) -> None:
        for fit in bounded.fits.values():
            for step in fit.steps:
                targeted = step.targeted[step.at_risk]
                assert targeted.min() >= FINITE_BOUNDS[0] - 1e-12
                assert targeted.max() <= FINITE_BOUNDS[1] + 1e-12
                followers = np.asarray(step.trained_on, dtype=bool)
                residual = np.where(followers, step.pseudo_outcome - step.targeted, 0.0)
                weights = np.asarray(bounded.data.weights) * np.asarray(step.clever)
                assert abs(float(np.mean(weights * residual))) < 1e-9
                assert step.fluctuation.converged

    def test_a_bounded_cross_fitted_fit_stays_inside_the_bounds(self) -> None:
        crossfit = fit_finite(node_bounds={1: FINITE_BOUNDS, 2: FINITE_BOUNDS}, n_folds=2)
        for fit in crossfit.fits.values():
            for step in fit.steps:
                targeted = step.targeted[step.at_risk]
                assert targeted.min() >= FINITE_BOUNDS[0] - 1e-12
                assert targeted.max() <= FINITE_BOUNDS[1] + 1e-12


# ----------------------------------------------------------------------- default identity


class TestDefaultIdentity:
    @pytest.mark.parametrize("declared", [None, {}, {1: (0.0, 1.0), 2: (0.0, 1.0)}])
    def test_no_bound_is_the_shipped_fit_bit_for_bit(self, baseline: Any, declared: Any) -> None:
        same = fit_end_of_study(node_bounds=declared)
        for name in law.NAMES:
            assert same.psi(name) == baseline.psi(name)
            np.testing.assert_array_equal(
                same.influence_curves[name], baseline.influence_curves[name]
            )

    def test_a_node_after_the_bounded_one_is_untouched(self, baseline: Any) -> None:
        """The recursion runs from node 2 down, so a bound at node 1 leaves node 2 alone."""
        bounded = fit_end_of_study(node_bounds={1: BINDING})
        for before, after in zip(steps_of(baseline, 2), steps_of(bounded, 2), strict=True):
            np.testing.assert_array_equal(before.targeted, after.targeted)
            np.testing.assert_array_equal(before.initial, after.initial)
        assert any(
            not np.array_equal(before.targeted, after.targeted)
            for before, after in zip(steps_of(baseline, 1), steps_of(bounded, 1), strict=True)
        )

    def test_the_summary_prints_the_bounds_only_when_declared(
        self, baseline: Any, binding: Any
    ) -> None:
        assert "node_bounds" not in baseline.summary()
        assert "node_bounds: t=1 in [0.3, 0.7]; t=2 in [0.3, 0.7]" in binding.summary()

    def test_the_replay_carries_the_bounds(self, binding: Any) -> None:
        recipe = binding.replay_recipe
        assert recipe.node_bounds is not None
        replayed = estimator_module._refit_bound(binding, recipe, binding.config.g_bounds)
        for name, estimate in replayed.estimates.items():
            assert estimate.psi == pytest.approx(binding.psi(name), abs=1e-12)

    def test_the_bounds_reach_the_bootstrap_replicate(self) -> None:
        estimator = _estimator(law, node_bounds={1: BINDING}, n_bootstrap=2)
        assert estimator._replicate_estimator().node_bounds == {1: BINDING}


# ----------------------------------------------------------------------- mutation controls


def check_exact_truth() -> None:
    """Bounds that hold the truth return it, and the bounded regression is the oracle's."""
    held = fit_end_of_study(node_bounds={1: HOLDING, 2: HOLDING})
    assert max_error(held, law.TRUTH) < 1e-10
    plain = fit_end_of_study()
    for time in (1, 2):
        for before, after in zip(steps_of(plain, time), steps_of(held, time), strict=True):
            np.testing.assert_allclose(
                after.initial[after.at_risk], before.initial[before.at_risk], atol=1e-12
            )


def check_node_placement() -> None:
    """Bounds at node 2 fail node 2 of the fit and no other."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        excluded = fit_end_of_study(node_bounds={2: (0.0, 0.1)})
    assert node_warnings(record)
    for fit in excluded.fits.values():
        assert [step.fluctuation.failure for step in fit.steps] == [None, "bound_excludes_data"]


class TestMutationControls:
    def test_the_unmutated_checks_pass(self) -> None:
        check_exact_truth()
        check_node_placement()
        check_root_check()

    def test_inverting_the_map_fails_the_exact_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def inverted(self: Any, values: Any) -> Any:
            return self.lower + self.span * (1.0 - np.asarray(values))

        monkeypatch.setattr(sequential_module.NodeBound, "unscale", inverted)
        with pytest.raises(AssertionError):
            check_exact_truth()

    def test_applying_the_bounds_one_node_early_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = estimator_module._SequentialEstimator._resolve_node_bounds

        def shifted(self: Any, data: Any, scaler: Any) -> Any:
            resolved, declared = original(self, data, scaler)
            if resolved is None:
                return resolved, declared
            return {
                cause: {time - 1: bound for time, bound in nodes.items() if time > 1}
                for cause, nodes in resolved.items()
            }, declared

        monkeypatch.setattr(estimator_module._SequentialEstimator, "_resolve_node_bounds", shifted)
        with pytest.raises(AssertionError):
            check_node_placement()

    def test_skipping_the_root_check_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sequential_module, "_root_exists", lambda mean: True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(AssertionError):
                check_root_check()


# --------------------------------------------------------------------------------- refusals


class RaisesIfFitted(BaseEstimator):
    """A learner that fails the test when a fit reaches it."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RaisesIfFitted:
        raise AssertionError("a learner was fitted before node_bounds was checked")

    def predict(self, X: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a learner was used")


def refused_fit(bounds: Any, module: Any = law, columns: Any = COLUMNS) -> Any:
    learner = RaisesIfFitted()
    estimator = _estimator(
        module,
        node_bounds=bounds,
        outcome_learner=learner,
        pseudo_learner=learner,
        treatment_learner=learner,
        censoring_learner=learner,
    )
    return estimator.fit(module.frame(), **columns)


class TestRefusals:
    @pytest.mark.parametrize(
        ("bounds", "message"),
        [
            (
                {3: (0.0, 0.5)},
                "node_bounds names grid time 3, which is not a node of this fit; its grid is [1, 2]",
            ),
            (
                {1: (0.5, 0.5)},
                "node_bounds at 1 needs lower < upper; got (0.5, 0.5)",
            ),
            (
                {1: (0.7, 0.2)},
                "node_bounds at 1 needs lower < upper; got (0.7, 0.2)",
            ),
            (
                {2: (-0.1, 0.5)},
                "node_bounds at 2 lies outside the outcome range [0, 1]",
            ),
            (
                {2: (0.1, 1.5)},
                "node_bounds at 2 lies outside the outcome range [0, 1]",
            ),
        ],
    )
    def test_a_bound_the_fit_cannot_hold_is_refused_before_any_learner(
        self, bounds: Any, message: str
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(message)) as caught:
            refused_fit(bounds)
        assert type(caught.value) is ValueError

    def test_an_undeclared_cause_is_refused_before_any_learner(self) -> None:
        message = "node_bounds names cause 'cancer'; this fit declared ['relapse', 'death']"
        with pytest.raises(ValueError, match=re.escape(message)):
            refused_fit({"cancer": {1: (0.0, 0.5)}}, competing, COMPETING_COLUMNS)

    def test_a_bound_outside_a_declared_outcome_range_is_refused(self) -> None:
        learner = RaisesIfFitted()
        data = law.frame()
        data["Y"] = data["Y"] * 4.0
        estimator = _estimator(
            law,
            node_bounds={2: (-1.0, 3.0)},
            q_bounds=(0.0, 4.0),
            outcome_learner=learner,
            pseudo_learner=learner,
            treatment_learner=learner,
            censoring_learner=learner,
        )
        with pytest.raises(ValueError, match=re.escape("lies outside the outcome range [0, 4]")):
            estimator.fit(data, **COLUMNS)

    def test_a_non_mapping_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="node_bounds must map grid times"):
            LTMLE({"always": 1}, node_bounds=[(0.0, 1.0)])

    def test_the_hazard_walk_is_a_gap_with_a_roadmap_item(self) -> None:
        with pytest.raises(CapabilityError) as caught:
            LTMLE({"always": 1}, survival_model="hazard", node_bounds={1: (0.0, 0.5)})
        refusal = caught.value.refusal
        assert refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert refusal.roadmap == "X78"
        assert refusal.alternative == "Fit node_bounds= with survival_model='iterated_mean'."
        assert "bounded hazard walk" in refusal.condition

    def test_a_working_model_is_a_gap_with_a_roadmap_item(self) -> None:
        with pytest.raises(CapabilityError) as caught:
            LTMLE({"always": 1}, msm=msm_module.DOSE, node_bounds={1: (0.0, 0.5)})
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert caught.value.refusal.roadmap == "X78"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: SDR({"always": 1}, node_bounds={1: (0.0, 0.5)}),
            lambda: IPSI(
                {"tilt": Incremental(2.0)},
                node_bounds={1: (0.0, 0.5)},
            ),
            lambda: SequentialGComputation({"always": 1}, node_bounds={1: (0.0, 0.5)}),
            lambda: SequentialIPW({"always": 1}, node_bounds={1: (0.0, 0.5)}),
            lambda: ParametricGFormula({"always": 1}, node_bounds={1: (0.0, 0.5)}),
        ],
    )
    def test_an_estimator_with_no_node_fluctuation_refuses_the_keyword(
        self, build: Callable[[], Any]
    ) -> None:
        with pytest.raises(MethodConfigurationError) as caught:
            build()
        refusal = caught.value.refusal
        assert refusal.reason is RefusalReason.INVALID_DECLARATION
        assert refusal.alternative is not None
        assert "LTMLE(node_bounds=...)" in refusal.alternative

    @pytest.mark.parametrize("method_class", [SDRMethod, IPSIMethod, GComputationMethod, IPWMethod])
    def test_a_method_with_no_node_fluctuation_refuses_the_setting(
        self, method_class: type
    ) -> None:
        method = method_class(targeting=Targeting(node_bounds={1: (0.0, 0.5)}))
        with pytest.raises(MethodConfigurationError) as caught:
            method.estimator_kwargs(longitudinal=True)
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "LTMLE(node_bounds=...)" in str(caught.value.refusal.alternative)

    def test_a_point_treatment_fit_refuses_the_setting(self) -> None:
        method = TMLEMethod(targeting=Targeting(node_bounds={1: (0.0, 0.5)}))
        with pytest.raises(MethodConfigurationError, match="node_bounds="):
            method.estimator_kwargs(longitudinal=False)

    def test_the_tmle_method_translates_the_setting_for_a_longitudinal_fit(self) -> None:
        declared = {1: (0.0, 0.5)}
        method = TMLEMethod(targeting=Targeting(node_bounds=declared))
        assert method.estimator_kwargs(longitudinal=True)["node_bounds"] == declared
        assert TMLEMethod().estimator_kwargs(longitudinal=True)["node_bounds"] is None
        assert TMLEMethod().with_overrides(node_bounds=declared).targeting.node_bounds == declared


# -------------------------------------------------------------------------- competing risks


class TestCompetingRisks:
    def test_a_cause_mapping_applies_to_that_cause_only(self) -> None:
        plain = fit_competing()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            bounded = fit_competing(node_bounds={"relapse": {1: (0.0, 0.2), 2: (0.0, 0.3)}})
        assert node_warnings(record)
        failed_causes = set()
        for key, fit in bounded.fits.items():
            failures = {step.fluctuation.failure for step in fit.steps}
            if "bound_excludes_data" in failures:
                failed_causes.add(fit.cause)
            if fit.cause == "death":
                assert failures == {None}, key
        assert failed_causes == {"relapse"}
        death = [name for name in competing.NAMES if "death" in name]
        relapse = [name for name in competing.NAMES if "relapse" in name]
        assert death
        for name in death:
            assert bounded.psi(name) == plain.psi(name)
            np.testing.assert_array_equal(
                bounded.influence_curves[name], plain.influence_curves[name]
            )
        assert max(abs(bounded.psi(name) - plain.psi(name)) for name in relapse) > 1e-3

    def test_bounds_that_hold_the_truth_return_it_for_every_cause(self) -> None:
        held = fit_competing(
            node_bounds={
                "relapse": {1: (0.0, 0.9), 2: (0.0, 0.9)},
                "death": {1: (0.0, 0.9), 2: (0.0, 0.9)},
            }
        )
        assert max_error(held, competing.TRUTH) < 1e-10
