"""``LTMLE(node_bounds=)`` on a hazard fit bounds the cause-specific hazards (X33 part (i)).

At a bounded node cause ``j`` holds ``theta_j = logit((lambda_j - l_j) / s_j)`` with the fixed
scale ``s_j = u_j - l_j``.  The fluctuation adds ``s_j h_j . epsilon`` to ``theta_j``, and the
node's loss is the quasi-binomial loss of ``(dN_j - l_j) / s_j``.  Its derivative in ``epsilon``
is the hazard score ``h_j (dN_j - lambda_j)``, so both algorithms solve the shipped score
equations (Benkeser, Carone and Gilbert 2018, Section 3, eq. (3)).

The exact checks run on the laws of ``tests/discrete_law_point_survival.py`` with the exact
mechanism.  There every targeted estimate is the truth for any initial hazard, so the bounds
cannot move an estimate.  The witness that they move one is a finite sample with estimated
mechanisms.  The nonzero witness of the step is the derivative of the walk loss along each
direction at a state where the score is not solved: it equals minus ``n`` times the score only
when the step is ``+ s_j h_j``.  Three mutation controls break the scale, the sign and the
initial clip, and each fails a check of this module.
"""

from __future__ import annotations

import functools
import re
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import make_point_survival
from cleverly.exceptions import CapabilityError, ConvergenceWarning, RefusalReason
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal.estimator import _fitted_replay_matches, _refit_bound
from cleverly.longitudinal.hazard import (
    WALK_LOSS_TOLERANCE,
    HazardBounds,
    HazardState,
    direction,
    hazard_bounds,
    hazard_inputs,
    scores,
    solve_per_horizon,
    walk_curve,
    walk_loss,
)
from cleverly.longitudinal.sequential import NodeBound, constant_nodes
from cleverly.utils.bounds import OutcomeScaler
from tests.discrete_law_point_survival import CellMeans, CellProbabilities, PointLaw, static
from tests.unit import test_hazard_full_cells as full_cells
from tests.unit.test_hazard_survival_law import ArmOnly

ONE = PointLaw("survival", 2)
TWO = PointLaw("survival", 2, causes=2)
REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
ALGORITHMS = ("iterative", "one_step")
STEP = 2e-2
#: Holds every hazard of the one-cause law, whose hazards lie in [0.25, 0.75].
HOLDING = (0.1, 0.8)
#: Mixed: node 1 bounds both causes with upper bounds summing to one, node 2 is unbounded.
#: Some death hazards of the two-cause law (0.5) lie above 0.4, so the bound binds.
MIXED = {"relapse": {1: (0.1, 0.6)}, "death": {1: (0.05, 0.4)}}
#: The witness sample of the finite-sample checks and its columns.
WITNESS_COLUMNS = {
    "outcome": [f"Y{k}" for k in range(1, 6)],
    "treatment": "A",
    "baseline": ["W1", "W2"],
    "censoring": [f"C{k}" for k in range(1, 6)],
}
#: Clips 32% of the initial logistic hazards at node 2 of the witness sample.
WITNESS_BOUNDS = {2: (0.02, 0.2)}


class High(CellMeans):
    """A hazard learner that predicts 0.95 everywhere, above every bound of this module."""

    def predict(self, X: Any) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], 0.95)

    def predict_proba(self, X: Any) -> np.ndarray:
        rows = np.asarray(X).shape[0]
        return np.column_stack([np.full(rows, 0.05), np.full(rows, 0.95)])


class _Refuse(CellMeans):
    def fit(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("a learner ran")


def _fit(law: PointLaw, targeting: str, learner: Any = None, **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            REGIMENS,
            n_folds=1,
            survival_model="hazard",
            targeting=targeting,
            step_size=STEP,
            outcome_learner=learner if learner is not None else ArmOnly(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
            **settings,
        ).fit(law.frame(), **law.fit_columns())


def _refused(law: PointLaw, bounds: Any) -> Any:
    learner = _Refuse()
    return LTMLE(
        REGIMENS,
        n_folds=1,
        survival_model="hazard",
        node_bounds=bounds,
        outcome_learner=learner,
        treatment_learner=learner,
        censoring_learner=learner,
    ).fit(law.frame(), **law.fit_columns())


@functools.cache
def _witness(targeting: str, bounded: bool = False) -> Any:
    """The witness sample fitted unbounded, or with :data:`WITNESS_BOUNDS`, once per module."""
    settings: dict[str, Any] = {"node_bounds": WITNESS_BOUNDS} if bounded else {}
    frame, _ = make_point_survival(n=300, seed=3, layout="wide")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"a1": 1, "a0": 0},
            n_folds=1,
            survival_model="hazard",
            targeting=targeting,
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            **settings,
        ).fit(frame, **WITNESS_COLUMNS)


def _bounds(result: Any, declared: dict[Any, Any] | None) -> HazardBounds | None:
    """The hazard bounds a fit of ``result``'s data resolves from ``declared``."""
    data = result.data
    causes = data.cause_labels or (None,)
    scaler = OutcomeScaler(0.0, 1.0)
    resolved: dict[str | None, dict[int, NodeBound]] = {}
    if declared is not None:
        items = declared.items() if data.cause_labels else [(None, declared)]
        for cause, mapping in items:
            for node, (lower, upper) in mapping.items():
                resolved.setdefault(cause, {})[node] = NodeBound(
                    lower, upper, node, (lower, upper), scaler
                )
    return hazard_bounds(resolved or None, data.n_times, causes)


def _internals(result: Any, bounds: HazardBounds | None) -> tuple[Any, HazardState]:
    """The targeting inputs and the initial state of ``result`` under ``bounds``."""
    recipe = result.replay_recipe
    cumulative = [
        result.mechanism.cumulative(result.data, level, result.config.g_bounds)
        for level in recipe.hazard.levels
    ]
    inputs = hazard_inputs(
        result.data, recipe.plans, cumulative, alpha=recipe.alpha, node_bounds=bounds
    )
    start = HazardState.from_hazards(
        result.initial_hazards.hazards, inputs.bounds, inputs.empty, inputs.full, bounds
    )
    return inputs, start


def _solve(result: Any, bounds: HazardBounds | None, targeting: str) -> list[Any]:
    """The targeted states and records of ``result`` under ``bounds``, at the solver level."""
    inputs, start = _internals(result, bounds)
    horizons = result.config.horizons
    record = next(iter(result.fits.values())).hazard
    causes = result.data.cause_labels or (None,)
    plans = result.replay_recipe.plans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if targeting == "one_step":
            targets = tuple((p.label, c, t) for p in plans for c in causes for t in horizons)
            return [walk_curve(start, inputs, horizons, targets, step_size=STEP, tol=record.tol)]
        return [
            solve_per_horizon(
                start,
                inputs,
                t,
                tuple((p.label, c, t) for p in plans for c in causes),
                max_iter=20,
                tol=record.tol,
            )
            for t in horizons
        ]


def _truth(law: PointLaw, fit: Any) -> float:
    code = 1 if fit.cause in (None, "relapse") else 2
    return law.functional(law.probs, static(REGIMENS[fit.regimen.label]), fit.horizon, code)


def _assert_in_bounds(state: HazardState, bounds: HazardBounds) -> None:
    k = bounds.bounded
    lower = bounds.lower[k][None, :, None, :]
    upper = lower + bounds.scale[k][None, :, None, :]
    hazards = state.hazards[:, k]
    assert np.all(hazards >= lower)
    assert np.all(hazards <= upper)


# ------------------------------------------------------------------------- refusals


class TestRefusals:
    def test_r_i2_upper_bounds_summing_above_one_are_refused_before_any_learner(self) -> None:
        message = (
            "node_bounds at 1 has upper bounds summing to 1.1; a hazard fit needs them to sum "
            "to at most 1, so that the bounded hazards leave room for no event. Lower the "
            "upper bounds at 1 so that they sum to at most 1."
        )
        bounds = {"relapse": {1: (0.1, 0.6)}, "death": {1: (0.05, 0.5)}}
        with pytest.raises(CapabilityError, match=re.escape(message)) as caught:
            _refused(TWO, bounds)
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert caught.value.refusal.alternative is not None

    def test_r_i3_a_node_that_bounds_some_causes_is_refused_before_any_learner(self) -> None:
        message = (
            "node_bounds bounds cause(s) ['relapse'] at 1 and not ['death']; a hazard fit "
            "bounds every cause of a node or none. A bounded node holds each cause on a fixed "
            "scale, and an unbounded cause there would share the mass the bounded causes "
            "leave. Bound every cause at 1, or none of them."
        )
        with pytest.raises(CapabilityError, match=re.escape(message)) as caught:
            _refused(TWO, {"relapse": {1: (0.1, 0.6)}})
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_r_i1_reads_the_hazard_range(self) -> None:
        with pytest.raises(ValueError, match=re.escape("lies outside the hazard range [0, 1]")):
            _refused(ONE, {1: (0.2, 1.5)})

    def test_upper_bounds_summing_to_one_are_accepted(self) -> None:
        result = _fit(TWO, "iterative", node_bounds=MIXED)
        assert all(fit.hazard.converged for fit in result.fits.values())

    def test_a_small_width_warns(self) -> None:
        with (
            pytest.warns(ConvergenceWarning, match=re.escape("has a width of 5e-05")),
            pytest.raises(AssertionError, match="a learner ran"),
        ):
            _refused(ONE, {1: (0.3, 0.30005)})


# ---------------------------------------------------------------------- identities


class TestIdentities:
    @pytest.mark.parametrize("targeting", ALGORITHMS)
    def test_a_unit_bound_at_one_cause_is_the_shipped_fit(self, targeting: str) -> None:
        """``(0, 1)`` at every node leaves the fit unbounded, so it is bit-identical."""
        plain = _fit(ONE, targeting)
        unit = _fit(ONE, targeting, node_bounds={1: (0.0, 1.0), 2: (0.0, 1.0)})
        for name in plain.estimates:
            assert plain[name].psi == unit[name].psi
            assert np.array_equal(plain[name].influence_curve, unit[name].influence_curve)

    @pytest.mark.parametrize("targeting", ALGORITHMS)
    def test_a_bounded_unit_node_equals_the_softmax_node(self, targeting: str) -> None:
        """``theta = logit(lambda) = eta`` at one cause and ``(0, 1)``, so the paths agree.

        The two nodes compute a hazard as ``l + s expit(theta)`` and as a softmax, which agree
        to rounding.  The start, its score and its first direction agree to ``1e-12``.  The
        per-horizon Newton step reads its Jacobian by forward differences of length
        ``1e-6``, which scales a rounding difference by about ``1e6``, so its endpoint agrees
        to ``1e-9``.  Both endpoints take the same number of steps and converge alike.
        """
        result = _fit(ONE, targeting)
        unit = HazardBounds(np.ones(2, dtype=bool), np.zeros((2, 1)), np.ones((2, 1)))
        horizons = result.config.horizons
        inputs, start = _internals(result, None)
        unit_inputs, unit_start = _internals(result, unit)
        np.testing.assert_allclose(unit_start.hazards, start.hazards, rtol=1e-12, atol=1e-15)
        score, _ = scores(start, inputs, horizons)
        unit_score, _ = scores(unit_start, unit_inputs, horizons)
        np.testing.assert_allclose(unit_score, score, rtol=1e-12, atol=1e-15)
        ones = np.ones_like(score)
        np.testing.assert_allclose(
            direction(unit_start, unit_inputs, horizons, ones),
            direction(start, inputs, horizons, ones),
            rtol=1e-12,
            atol=1e-15,
        )
        tolerance = 1e-12 if targeting == "one_step" else 1e-9
        plain = _solve(result, None, targeting)
        bounded = _solve(result, unit, targeting)
        for (state, record), (other, other_record) in zip(plain, bounded, strict=True):
            np.testing.assert_allclose(other.hazards, state.hazards, rtol=tolerance, atol=1e-15)
            assert other_record.n_iter == record.n_iter
            assert other_record.converged == record.converged


# ------------------------------------------------------------------------ exactness


class TestExact:
    @pytest.mark.parametrize("targeting", ALGORITHMS)
    @pytest.mark.parametrize("learner", [ArmOnly, High], ids=["arm-only", "high"])
    def test_bounds_that_hold_the_truth_return_it(self, targeting: str, learner: type) -> None:
        bounds = {1: HOLDING, 2: HOLDING}
        result = _fit(ONE, targeting, learner(), node_bounds=bounds)
        for key, fit in result.fits.items():
            assert fit.hazard.converged, key
            assert fit.psi_scaled == pytest.approx(_truth(ONE, fit), abs=1e-8), key
        resolved = _bounds(result, bounds)
        assert resolved is not None
        for state, _ in _solve(result, resolved, targeting):
            _assert_in_bounds(state, resolved)

    @pytest.mark.parametrize("targeting", ALGORITHMS)
    def test_an_initial_hazard_outside_the_bound_is_clipped(self, targeting: str) -> None:
        """The nonzero witness of the clip: every initial hazard of 0.95 lies above 0.8."""
        bounds = {1: HOLDING, 2: HOLDING}
        result = _fit(ONE, targeting, High(), node_bounds=bounds)
        record = next(iter(result.fits.values())).hazard
        assert record.share_clipped == ((1, 0, 1.0), (2, 0, 1.0))
        assert record.bound_excludes_data == ()
        resolved = _bounds(result, bounds)
        assert resolved is not None
        check_initial_clip(result, resolved)

    @pytest.mark.parametrize("targeting", ALGORITHMS)
    def test_mixed_nodes_and_competing_causes_solve_every_score(self, targeting: str) -> None:
        result = _fit(TWO, targeting, node_bounds=MIXED)
        rows = result.diagnostics.score_equations().rows
        assert len(rows) == len(result.fits)
        for row in rows:
            assert row.passed
            assert row.relative_score <= 1e-10
        for key, fit in result.fits.items():
            assert fit.hazard.converged, key
            assert fit.psi_scaled == pytest.approx(_truth(TWO, fit), abs=1e-8), key
        resolved = _bounds(result, MIXED)
        assert resolved is not None
        assert list(resolved.bounded) == [True, False]
        inputs, _ = _internals(result, resolved)
        for state, record in _solve(result, resolved, targeting):
            assert record.converged
            _assert_in_bounds(state, resolved)
            score, _ = scores(state, inputs, sorted({t for _, _, t in record.targets}))
            assert np.max(np.abs(score)) < 1e-9

    def test_the_bounded_walk_loss_never_rises(self) -> None:
        result = _fit(TWO, "one_step", node_bounds=MIXED)
        record = next(iter(result.fits.values())).hazard
        losses = np.asarray(record.walk_loss_trace)
        assert len(losses) == record.n_iter + 1 > 2
        assert np.all(np.diff(losses) <= WALK_LOSS_TOLERANCE)


# -------------------------------------------------------------------- the witness


class TestBoundsChangeTheFit:
    @pytest.mark.parametrize("targeting", ALGORITHMS)
    def test_bounded_and_unbounded_estimates_differ(self, targeting: str) -> None:
        plain = _witness(targeting)
        bounded = _witness(targeting, bounded=True)
        record = next(iter(bounded.fits.values())).hazard
        assert all(fit.hazard.converged for fit in bounded.fits.values())
        node, cause, share = record.share_clipped[0]
        assert (node, cause) == (2, 0) and share > 0.1
        assert max(abs(bounded[n].psi - plain[n].psi) for n in plain.estimates) > 1e-4
        resolved = _bounds(bounded, WITNESS_BOUNDS)
        assert resolved is not None
        for state, _ in _solve(bounded, resolved, targeting):
            _assert_in_bounds(state, resolved)

    def test_the_summary_prints_the_hazard_scale_and_the_clipped_share(self) -> None:
        text = _witness("one_step", bounded=True).summary()
        assert "node_bounds: t=2 in [0.02, 0.2] (bounded cause-specific hazards" in text
        assert "hazard node bounds: largest clipped share of initial hazards" in text

    @pytest.mark.parametrize("targeting", ALGORITHMS)
    def test_the_replay_carries_the_bounds(self, targeting: str) -> None:
        result = _witness(targeting, bounded=True)
        replay = _refit_bound(result, result.replay_recipe, result.config.g_bounds)
        assert _fitted_replay_matches(result, replay)


# ------------------------------------------------------------- held and excluded cells


def _empty_witness(**settings: Any) -> Any:
    frame, _ = make_point_survival(n=400, seed=36, layout="wide", hazard_scale={3: 0.05})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"a1": 1, "a0": 0},
            n_folds=1,
            survival_model="hazard",
            targeting="one_step",
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
            **settings,
        ).fit(frame, **WITNESS_COLUMNS)


@pytest.fixture(scope="module")
def empty_witness() -> Any:
    return _empty_witness()


class TestHeldCells:
    def test_an_empty_cell_under_a_zero_lower_bound_is_held(self, empty_witness: Any) -> None:
        bounds = _bounds(empty_witness, {3: (0.0, 0.5)})
        inputs, start = _internals(empty_witness, bounds)
        arm = int(np.flatnonzero(np.asarray(empty_witness.data.treatment_levels[0]) == 1)[0])
        assert inputs.empty[2, arm, 0]
        assert inputs.excluded is not None and not inputs.excluded.any()
        assert np.all(start.hazards[:, 2, arm, 0] == 0.0)
        assert inputs.constant_nodes(0, 5) == (3,)

    def test_an_empty_cell_under_a_positive_lower_bound_is_excluded(
        self, empty_witness: Any
    ) -> None:
        bounds = _bounds(empty_witness, {3: (0.01, 0.5)})
        inputs, start = _internals(empty_witness, bounds)
        arm = int(np.flatnonzero(np.asarray(empty_witness.data.treatment_levels[0]) == 1)[0])
        assert not inputs.empty[2].any()
        assert inputs.excluded is not None
        expected = np.zeros_like(inputs.excluded)
        expected[2, arm, 0] = True
        assert np.array_equal(inputs.excluded, expected)
        # Not held, so pinned at the clip of the bound rather than at zero.
        assert np.all(start.hazards[:, 2, arm, 0] >= 0.01)
        assert inputs.constant_nodes(0, 5) == ()

    def test_a_fit_with_an_excluded_cell_warns_and_records_it(self) -> None:
        with pytest.warns(ConvergenceWarning, match="excludes the data") as record:
            frame, _ = make_point_survival(n=400, seed=36, layout="wide", hazard_scale={3: 0.05})
            result = LTMLE(
                {"a1": 1, "a0": 0},
                n_folds=1,
                survival_model="hazard",
                node_bounds={3: (0.01, 0.5)},
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                censoring_learner=LogisticRegression(max_iter=1000),
            ).fit(frame, **WITNESS_COLUMNS)
        text = next(str(w.message) for w in record if "excludes the data" in str(w.message))
        assert text.startswith("the declared node bound at node(s) 3 excludes the data: 1 cell")
        arm = int(np.flatnonzero(np.asarray(result.data.treatment_levels[0]) == 1)[0])
        for fit in result.fits.values():
            assert fit.hazard.bound_excludes_data == ((3, arm, 0),)
            assert constant_nodes(fit) == ()
            assert fit.hazard.converged or fit.hazard.failure == "bound_excludes_data"

    def test_a_full_cell_at_a_bounded_node_is_excluded(self) -> None:
        result = full_cells._fit("one_step")
        bounds = _bounds(result, {2: (0.0, 0.9)})
        inputs, _ = _internals(result, bounds)
        arm = int(np.flatnonzero(np.asarray(result.data.treatment_levels[0]) == 1)[0])
        assert not inputs.full[1].any()
        assert inputs.excluded is not None and inputs.excluded[1, arm, 0]


# ------------------------------------------------------------------ mutation controls


def check_loss_derivative() -> None:
    """The walk loss falls along each direction at the rate of the score, at every node kind.

    At an unsolved state of the two-cause law with node 1 bounded (``l > 0``, ``s < 1``) and
    node 2 unbounded, the central difference of the walk loss along ``direction(c)`` equals
    ``-n c . score``.  A wrong scale or sign of the bounded step breaks the equality.
    """
    result = _fit(TWO, "iterative")
    bounds = _bounds(result, MIXED)
    inputs, start = _internals(result, bounds)
    horizons = result.config.horizons
    score, _ = scores(start, inputs, horizons)
    assert np.max(np.abs(score)) > 1e-3
    rng = np.random.default_rng(0)
    for _ in range(3):
        coefficients = rng.normal(size=score.shape)
        move = hazard_module.direction(start, inputs, horizons, coefficients)
        dx = 1e-6
        ahead = walk_loss(start.moved(dx * move, inputs.bounds), inputs)
        behind = walk_loss(start.moved(-dx * move, inputs.bounds), inputs)
        slope = (ahead - behind) / (2 * dx)
        expected = -inputs.n * float(np.sum(coefficients * score))
        assert slope == pytest.approx(expected, rel=1e-5)


def check_initial_clip(result: Any, bounds: HazardBounds) -> None:
    """The initial bounded hazards are the learner's hazards clipped into the bound."""
    inputs, start = _internals(result, bounds)
    alpha = inputs.alpha
    k = bounds.bounded
    lower = bounds.lower[k][None, :, None, :]
    scale = bounds.scale[k][None, :, None, :]
    raw = result.initial_hazards.hazards[:, k]
    clipped = np.clip(raw, lower + (1 - alpha) * scale, lower + alpha * scale)
    np.testing.assert_allclose(start.hazards[:, k], clipped, rtol=1e-12, atol=1e-15)
    assert np.any(raw > lower + scale)


class TestMutationControls:
    def test_the_unmutated_checks_pass(self) -> None:
        check_loss_derivative()

    def test_dropping_the_scale_from_the_step_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unscaled(state: HazardState, inputs: Any, horizons: Any, c: Any) -> Any:
            out = direction(state, inputs, horizons, c)
            if state.node_bounds is None:
                return out
            return out / state.node_bounds.scale[None, :, None, :]

        monkeypatch.setattr(hazard_module, "direction", unscaled)
        with pytest.raises(AssertionError):
            check_loss_derivative()

    def test_a_scale_without_the_lower_bound_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``survtmle`` master drops ``- l_j`` from the scale (the comparator notes)."""

        def upper(state: HazardState, inputs: Any, horizons: Any, c: Any) -> Any:
            bounds = state.node_bounds
            if bounds is None:
                return direction(state, inputs, horizons, c)
            out = direction(state, inputs, horizons, c) / bounds.scale[None, :, None, :]
            ceiling = np.where(bounds.bounded[:, None], bounds.lower + bounds.scale, 1.0)
            return out * ceiling[None, :, None, :]

        monkeypatch.setattr(hazard_module, "direction", upper)
        with pytest.raises(AssertionError):
            check_loss_derivative()

    def test_flipping_the_sign_of_the_step_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def flipped(state: HazardState, inputs: Any, horizons: Any, c: Any) -> Any:
            out = direction(state, inputs, horizons, c)
            if state.node_bounds is None:
                return out
            sign = np.where(state.node_bounds.bounded, -1.0, 1.0)
            return out * sign[None, :, None, None]

        monkeypatch.setattr(hazard_module, "direction", flipped)
        with pytest.raises(AssertionError):
            check_loss_derivative()

    def test_skipping_the_bounded_initial_map_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The initial logits of a bounded node read as softmax logits, unclipped by the bound."""
        result = _fit(ONE, "iterative", High(), node_bounds={1: HOLDING, 2: HOLDING})
        bounds = _bounds(result, {1: HOLDING, 2: HOLDING})
        assert bounds is not None
        original = HazardState.from_hazards.__func__  # type: ignore[attr-defined]

        def softmax_logits(
            cls: Any,
            hazards: Any,
            clip: Any,
            empty: Any = None,
            full: Any = None,
            node_bounds: Any = None,
        ) -> Any:
            plain = original(cls, hazards, clip, empty, full, None)
            return cls(plain.logits, plain.full, node_bounds)

        monkeypatch.setattr(HazardState, "from_hazards", classmethod(softmax_logits))
        with pytest.raises(AssertionError):
            check_initial_clip(result, bounds)
