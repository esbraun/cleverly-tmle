"""Per-row arm assignments and per-arm predictions in the cross-fitted fold recursions.

A learned two-node regimen (X11 part (b), PR K) learns one rule per outer fold on that fold's
training rows, and each row then follows the rule of the fold that held it out.  This module
pins the groundwork it builds on, which changes no shipped fit:

* ``Plan.with_assignment`` and ``stitch_fold_assignments`` build a plan whose arms differ row
  by row, and ``FoldRecursionCell.fold_assignments`` runs each fold under its own assignment;
* ``FoldRecursionCell.arm_nodes`` predicts the horizon of a deterministic node at every arm,
  one stratified regression per arm, on every row of every fold.  Below the horizon it is
  refused: the later node's regression does not condition on the earlier arm.

=====  ======================================================================================
claim  what fails it
=====  ======================================================================================
R1     each row's assignment equal to the regimen's own reproduces every shipped estimate,
       influence curve and node array bit for bit: static, rule and stochastic plans,
       cross-fitted and in sample, end of study and survival
R2     per-arm predictions at the horizon are the single-arm recursions, fold by fold and
       stitched; every node below the horizon is refused, under the observed arm and under a
       static one
R3     each fold's recursion reads its own assignment, each row's masks read its fold's
       rule, and each row's per-arm prediction at its own arm is the matching single-arm
       recursion; ``with_assignment`` that ignores the assignment (mutation) fails it.  A
       selection at the plan's arm instead of the row's is a premise check: the two differ
R3b    under fold rules at node 1, each fold's horizon arm block on its training rows is the
       single-arm recursion under that fold's plan; a single-arm call that resets the
       earlier nodes fails it
R3c    the mechanism evaluated at the stitched plan is, on each fold's held-out rows, the one
       evaluated at that fold's rule; the regimen's own mechanism (mutation) is not
R4     malformed cells are refused before any fold runs: the assignment count, shape and
       codes, a Markov order, and an arm node outside the horizon or at a drawn node
=====  ======================================================================================
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from cleverly.interventions.incremental import Incremental
from cleverly.longitudinal import LongitudinalResult, estimator, sequential
from cleverly.longitudinal.regimen import Plan, resolve_plans, resolve_regimens
from cleverly.longitudinal.sequential import (
    FoldRecursionCell,
    StitchedInitial,
    stitch_fold_assignments,
    untargeted_fold_recursions,
    untargeted_recursion,
)

from .test_pooled_longitudinal_policy_targeting import _fit as fit_policy
from .test_pooled_longitudinal_targeting import fit_categorical, fit_end_of_study, fit_survival

LOGISTIC = LogisticRegression(max_iter=1000)

#: Each configuration of R1, fitted twice: as shipped and with per-row assignments.
CONFIGURATIONS = {
    "static_crossfit": lambda: fit_end_of_study(),
    "static_in_sample": lambda: fit_end_of_study(n_folds=1),
    "survival_crossfit": lambda: fit_survival(),
    "rule_crossfit": lambda: fit_categorical(),
    "rule_in_sample": lambda: fit_categorical(n_folds=1),
    "policy_crossfit": lambda: fit_policy(),
    "policy_in_sample": lambda: fit_policy(n_folds=1),
}


def _own_copy(values: np.ndarray) -> np.ndarray:
    return np.array(values, dtype=float, copy=True)


def _horizon_arm_nodes(data: Any, cell: FoldRecursionCell) -> frozenset[int]:
    """The horizon, when it is a deterministic categorical node: arms that move nothing."""
    deterministic = not (
        cell.plan.is_policy_node(cell.horizon) or cell.plan.is_mtp_node(cell.horizon)
    )
    if deterministic and data.node_kind(cell.horizon) == "categorical":
        return frozenset({cell.horizon})
    return frozenset()


def _with_per_row_assignments(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Route every fit through per-row assignments equal to the regimen's own arms.

    The targeting plan is rebuilt from one copied assignment per fold, stitched by the fold
    that holds each row out; each fold's recursion runs under its own copy; and the horizon
    is also predicted at every arm.  Returns the log of the calls the patches saw.
    """
    seen: list[str] = []
    fit_regimen = estimator.fit_regimen
    recursions = sequential.untargeted_fold_recursions

    def per_row_fit(data: Any, plan: Plan, mechanism: Any, *, folds: Any, **kwargs: Any) -> Any:
        copies = [_own_copy(plan.values) for _ in range(folds.n_folds)]
        stitched = copies[0] if folds.is_single else stitch_fold_assignments(copies, folds)
        seen.append("fit")
        return fit_regimen(data, plan.with_assignment(stitched), mechanism, folds=folds, **kwargs)

    def per_fold(data: Any, cells: Any, *, folds: Any, **kwargs: Any) -> Any:
        cells = [
            replace(
                cell,
                fold_assignments=tuple(_own_copy(cell.plan.values) for _ in range(folds.n_folds)),
                arm_nodes=_horizon_arm_nodes(data, cell),
            )
            for cell in cells
        ]
        seen.append("folds")
        return recursions(data, cells, folds=folds, **kwargs)

    monkeypatch.setattr(estimator, "fit_regimen", per_row_fit)
    monkeypatch.setattr(sequential, "untargeted_fold_recursions", per_fold)
    return seen


def _assert_identical(shipped: LongitudinalResult, per_row: LongitudinalResult) -> None:
    assert shipped.fits.keys() == per_row.fits.keys()
    for key, fit in shipped.fits.items():
        other = per_row.fits[key]
        assert fit.psi_scaled == other.psi_scaled, key
        assert np.array_equal(fit.influence_curve_scaled, other.influence_curve_scaled), key
        assert np.array_equal(fit.assignment, other.assignment, equal_nan=True), key
        assert len(fit.steps) == len(other.steps)
        for step, twin in zip(fit.steps, other.steps, strict=True):
            for name in ("at_risk", "trained_on", "initial", "targeted", "clever"):
                assert np.array_equal(getattr(step, name), getattr(twin, name)), (key, name)
            if step.initial_by_arm is not None:
                assert twin.initial_by_arm is not None
                assert np.array_equal(step.initial_by_arm, twin.initial_by_arm), key
    assert shipped.estimates.keys() == per_row.estimates.keys()
    for label, estimate in shipped.estimates.items():
        assert estimate.psi == per_row.estimates[label].psi, label
        assert estimate.std_error == per_row.estimates[label].std_error, label


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_r1_the_regimens_own_assignment_per_row_reproduces_every_fit(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = CONFIGURATIONS[name]
    shipped = build()
    with monkeypatch.context() as patch:
        seen = _with_per_row_assignments(patch)
        per_row = build()
    assert "fit" in seen
    if shipped.folds.n_folds > 1:
        assert "folds" in seen
    _assert_identical(shipped, per_row)


# --------------------------------------------------------------------------------------
# R2: per-arm predictions are the single-arm calls.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def static() -> LongitudinalResult:
    return fit_end_of_study()


def _plan(result: LongitudinalResult, label: str) -> Plan:
    return next(plan for plan in result.replay_recipe.plans if plan.label == label)


def _recursions(
    result: LongitudinalResult, cells: list[FoldRecursionCell], outcome_learner: Any = None
) -> list[StitchedInitial]:
    return untargeted_fold_recursions(
        result.data,
        cells,
        folds=result.folds,
        scaler=result.scaler,
        cause=None,
        outcome_learner=DummyRegressor() if outcome_learner is None else outcome_learner,
        pseudo_learner=DummyRegressor(),
    )


def _set_node(plan: Plan, time: int, column: np.ndarray | float) -> Plan:
    """``plan`` with node ``time`` reassigned, built without ``Plan.with_assignment``.

    The references here must not read the method a mutation below replaces.
    """
    values = _own_copy(plan.values)
    values[:, time - 1] = column
    return replace(plan, values=values)


def _cell(result: LongitudinalResult, plan: Plan, **kwargs: Any) -> FoldRecursionCell:
    return FoldRecursionCell(plan, plan.masks(result.data), result.data.n_times, **kwargs)


def test_r2_horizon_arm_predictions_are_the_single_arm_recursions(
    static: LongitudinalResult,
) -> None:
    data, horizon = static.data, static.data.n_times
    plan = _plan(static, "always")
    (both,) = _recursions(static, [_cell(static, plan, arm_nodes=frozenset({horizon}))], LOGISTIC)
    singles = _recursions(
        static, [_cell(static, _set_node(plan, horizon, a)) for a in (0, 1)], LOGISTIC
    )
    block = both.arm_predictions[horizon]
    assert block.shape == (data.n, 2)
    for arm, single in enumerate(singles):
        assert np.array_equal(block[:, arm], single.initial[horizon])
    # The two arms are different regressions, so the equality is not one column twice.
    at_risk = plan.masks(data).at_risk(horizon)
    assert np.max(np.abs(block[at_risk, 0] - block[at_risk, 1])) > 1e-3
    # Each fold's block on every row is that fold's own single-arm recursion.
    folds = list(static.folds)
    assert len(both.fold_arm_predictions[horizon]) == len(folds)
    for fold, (train, test) in enumerate(folds):
        rows = np.zeros(data.n, dtype=bool)
        rows[train] = True
        own = both.fold_arm_predictions[horizon][fold]
        assert np.array_equal(own[test], block[test])
        for arm in (0, 1):
            arm_plan = _set_node(plan, horizon, arm)
            (node, *_) = untargeted_recursion(
                data,
                arm_plan,
                masks=arm_plan.masks(data),
                horizon=horizon,
                scaler=static.scaler,
                cause=None,
                outcome_learner=LOGISTIC,
                pseudo_learner=DummyRegressor(),
                fit_rows=rows,
            )
            assert np.array_equal(own[:, arm], node.initial)
    # Asking for both arms moves nothing else.
    (alone,) = _recursions(static, [_cell(static, plan)], LOGISTIC)
    for time in alone.initial:
        assert np.array_equal(alone.initial[time], both.initial[time])
        assert np.array_equal(alone.regression_target[time], both.regression_target[time])
    assert alone.arm_predictions == {}
    assert alone.fold_arm_predictions == {}


@pytest.mark.parametrize("assignment", ("observed", "static"))
def test_r2_every_node_below_the_horizon_is_refused(
    static: LongitudinalResult, assignment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the horizon the later regression pools this node's arms, so no arm is asked for.

    Under the observed node-1 arm the node-2 regression is fitted on both node-1 arms with no
    block for ``A1``, and under the static arm the other arm's rows hold the filler.  Both are
    refused before any fold runs.
    """
    plan = _plan(static, "always")
    if assignment == "observed":
        plan = _set_node(plan, 1, np.nan_to_num(static.data.treatment[:, 0], nan=0.0))
    monkeypatch.setattr(sequential, "map_parallel", _no_fold_runs)
    with pytest.raises(
        ValueError, match=re.escape("below the horizon 2. The later node's regression")
    ):
        _recursions(static, [_cell(static, plan, arm_nodes=frozenset({1}))])


def _no_fold_runs(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a fold ran before the cell was refused")


def test_r2_a_policy_node_refuses_arm_predictions() -> None:
    result = fit_policy()
    plan = _plan(result, "mix")
    assert plan.is_policy_node(2)
    with pytest.raises(ValueError, match="initial_by_arm"):
        _recursions(result, [_cell(result, plan, arm_nodes=frozenset({2}))])


# --------------------------------------------------------------------------------------
# R3: each row reads its own fold's assignment.
# --------------------------------------------------------------------------------------


def _fold_rules(result: LongitudinalResult) -> tuple[np.ndarray, ...]:
    """One node-2 rule per fold: treat when ``L2`` is above a cut that differs by fold."""
    plan = _plan(result, "always")
    # The node-2 history is ``[W1, W2, L2]``, so its last column is ``L2``.
    l2 = result.data.covariate_history(2)[:, -1]
    cuts = np.quantile(l2, [0.3, 0.5, 0.7])
    return tuple(
        _set_node(plan, 2, (l2 > cut).astype(float)).values for cut in cuts[: result.folds.n_folds]
    )


def _r3_gaps(result: LongitudinalResult) -> dict[str, float]:
    """The largest distance between each R3 reading and the per-fold reference."""
    data, folds = result.data, result.folds
    plan = _plan(result, "always")
    rules = _fold_rules(result)
    stitched_rule = stitch_fold_assignments(rules, folds)
    (per_row,) = _recursions(
        result, [_cell(result, plan, fold_assignments=rules, arm_nodes=frozenset({2}))]
    )
    references = _recursions(result, [_cell(result, replace(plan, values=rule)) for rule in rules])
    gaps = {"initial": 0.0, "masks": 0.0, "selected": 0.0}
    for fold, (_, test) in enumerate(folds):
        for time in (1, 2):
            gap = np.abs(per_row.initial[time][test] - references[fold].initial[time][test])
            gaps["initial"] = max(gaps["initial"], float(np.max(gap)))
    # Each row follows the rule of the fold that held it out.
    following = plan.with_assignment(stitched_rule).masks(data).following(2)
    expected = (
        plan.masks(data).at_risk(2)
        & data.uncensored_through(2)
        & (data.treatment[:, 1] == stitched_rule[:, 1])
    )
    gaps["masks"] = float(np.mean(following != expected))
    # The prediction at each row's own arm is the single-arm recursion of that arm.
    singles = _recursions(result, [_cell(result, _set_node(plan, 2, a)) for a in (0, 1)])
    arms = stitched_rule[:, 1].astype(np.int64)
    selected = per_row.arm_predictions[2][np.arange(data.n), arms]
    matching = np.where(arms == 0, singles[0].initial[2], singles[1].initial[2])
    gaps["selected"] = float(np.max(np.abs(selected - matching)))
    return gaps


def test_r3_the_fold_rules_assign_both_arms_and_differ_by_fold(static: LongitudinalResult) -> None:
    """The nonzero premise: rows take both arms, and the folds' rules disagree on rows."""
    rules = _fold_rules(static)
    stitched = stitch_fold_assignments(rules, static.folds)[:, 1]
    assert 0.2 < float(np.mean(stitched)) < 0.8
    assert float(np.mean(rules[0][:, 1] != rules[-1][:, 1])) > 0.2


def test_r3_each_row_reads_its_own_folds_assignment(static: LongitudinalResult) -> None:
    gaps = _r3_gaps(static)
    assert gaps == {"initial": 0.0, "masks": 0.0, "selected": 0.0}


def test_r3_an_ignored_assignment_fails(
    static: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: ``with_assignment`` keeps the plan's own arms, so no row reads its rule."""
    monkeypatch.setattr(Plan, "with_assignment", lambda self, assignment: self)
    gaps = _r3_gaps(static)
    assert gaps["initial"] > 1e-3
    assert gaps["masks"] > 0.05


def test_r3_selecting_at_the_plans_arm_fails(static: LongitudinalResult) -> None:
    """Premise check: the arm prediction read at the plan's arm is not the row's arm's.

    The selection here is the test's own, not production code.  It shows that the per-row
    selection of R3 could be told apart from the plan's arm.
    """
    data = static.data
    plan = _plan(static, "always")
    rules = _fold_rules(static)
    (per_row,) = _recursions(
        static, [_cell(static, plan, fold_assignments=rules, arm_nodes=frozenset({2}))]
    )
    singles = _recursions(static, [_cell(static, _set_node(plan, 2, a)) for a in (0, 1)])
    arms = stitch_fold_assignments(rules, static.folds)[:, 1].astype(np.int64)
    wrong = per_row.arm_predictions[2][np.arange(data.n), plan.values[:, 1].astype(np.int64)]
    matching = np.where(arms == 0, singles[0].initial[2], singles[1].initial[2])
    assert float(np.max(np.abs(wrong - matching))) > 1e-3


def _node_one_rules(result: LongitudinalResult) -> tuple[np.ndarray, ...]:
    """One node-1 rule per fold: treat when ``W1`` is above a cut that differs by fold."""
    plan = _plan(result, "always")
    w1 = result.data.covariate_history(1)[:, 0]
    cuts = np.quantile(w1, [0.3, 0.5, 0.7])
    return tuple(
        _set_node(plan, 1, (w1 > cut).astype(float)).values for cut in cuts[: result.folds.n_folds]
    )


def _earlier_rule_gaps(result: LongitudinalResult) -> tuple[float, float]:
    """R3b: the distance to each fold's own single-arm recursion, and to the plan's.

    The first is the claim and must be zero.  The second is the premise: the fold's node-1
    rule moves the horizon's arm block, so a call that ignores it would be seen.
    """
    data = result.data
    plan = _plan(result, "always")
    rules = _node_one_rules(result)
    (per_row,) = _recursions(
        result,
        [_cell(result, plan, fold_assignments=rules, arm_nodes=frozenset({2}))],
        LOGISTIC,
    )
    claim = premise = 0.0
    for fold, (train, _) in enumerate(result.folds):
        rows = np.zeros(data.n, dtype=bool)
        rows[train] = True
        block = per_row.fold_arm_predictions[2][fold]
        for arm in (0, 1):
            for source, name in ((rules[fold], "claim"), (plan.values, "premise")):
                arm_plan = _set_node(replace(plan, values=source), 2, arm)
                (node, *_) = untargeted_recursion(
                    data,
                    arm_plan,
                    masks=arm_plan.masks(data),
                    horizon=2,
                    scaler=result.scaler,
                    cause=None,
                    outcome_learner=LOGISTIC,
                    pseudo_learner=DummyRegressor(),
                    fit_rows=rows,
                )
                gap = float(np.max(np.abs(block[rows, arm] - node.initial[rows])))
                if name == "claim":
                    claim = max(claim, gap)
                else:
                    premise = max(premise, gap)
    return claim, premise


def test_r3b_an_earlier_fold_rule_reaches_the_horizon_arm_block(
    static: LongitudinalResult,
) -> None:
    rules = _node_one_rules(static)
    assert float(np.mean(rules[0][:, 0] != rules[-1][:, 0])) > 0.2
    claim, premise = _earlier_rule_gaps(static)
    assert claim == 0.0
    assert premise > 1e-3


def _mechanism(result: LongitudinalResult, plan: Plan) -> Any:
    return sequential.fit_mechanism(
        result.data,
        [plan],
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        folds=result.folds,
    )


def test_r3c_the_mechanism_is_evaluated_at_each_rows_fold_rule(static: LongitudinalResult) -> None:
    data, bounds = static.data, (1e-8, 1.0)
    plan = _plan(static, "always")
    rules = _fold_rules(static)
    stitched = plan.with_assignment(stitch_fold_assignments(rules, static.folds))
    cumulative = _mechanism(static, stitched).cumulative(data, stitched, bounds)
    for fold, (_, test) in enumerate(static.folds):
        own = replace(plan, values=rules[fold])
        expected = _mechanism(static, own).cumulative(data, own, bounds)
        np.testing.assert_allclose(cumulative[test], expected[test], rtol=0, atol=1e-12)
    # Mutation: the regimen's own mechanism, read at the stitched plan, keeps its own arms.
    wrong = _mechanism(static, plan).cumulative(data, stitched, bounds)
    assert float(np.max(np.abs(wrong - cumulative))) > 1e-3


def test_r4_a_fold_assignment_count_shape_or_code_is_refused(
    static: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sequential, "map_parallel", _no_fold_runs)
    plan = _plan(static, "always")
    rules = _fold_rules(static)
    with pytest.raises(ValueError, match="2 fold assignments for 3 folds"):
        _recursions(static, [_cell(static, plan, fold_assignments=rules[:2])])
    with pytest.raises(ValueError, match="assigns shape"):
        _recursions(static, [_cell(static, plan, fold_assignments=tuple(r[:, :1] for r in rules))])
    for code in (2.0, 0.5, -1.0):
        bad = _own_copy(rules[1])
        bad[0, 1] = code
        with pytest.raises(ValueError, match="outside the 2 level codes"):
            _recursions(static, [_cell(static, plan, fold_assignments=(rules[0], bad, rules[2]))])


def test_r4_fold_assignments_under_a_markov_order_are_refused(
    static: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sequential, "map_parallel", _no_fold_runs)
    plan = _plan(static, "always")
    windowed = replace(static, data=replace(static.data, markov_order=1))
    with pytest.raises(ValueError, match="under a Markov order"):
        _recursions(windowed, [_cell(windowed, plan, fold_assignments=_fold_rules(static))])


@pytest.mark.parametrize("node", (0, 3))
def test_r4_an_arm_node_outside_the_recursion_is_refused(
    static: LongitudinalResult, node: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sequential, "map_parallel", _no_fold_runs)
    plan = _plan(static, "always")
    with pytest.raises(ValueError, match=re.escape("outside the recursion's nodes 1..2")):
        _recursions(static, [_cell(static, plan, arm_nodes=frozenset({node}))])


def test_r4_an_arm_node_at_a_tilt_is_refused(
    static: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tilt node, before its density is attached, is not a deterministic node."""
    monkeypatch.setattr(sequential, "map_parallel", _no_fold_runs)
    plan = _tilt_plan(static)
    with pytest.raises(ValueError, match="not a deterministic categorical node"):
        _recursions(static, [_cell(static, plan, arm_nodes=frozenset({2}))])


def _tilt_plan(result: LongitudinalResult) -> Plan:
    """A static node 1 and an odds tilt at node 2, as resolved, before ``with_tilts``."""
    (plan,) = resolve_plans(resolve_regimens({"tilt": (1, Incremental(2.0))}, 2), result.data)
    assert plan.is_tilt_node(2) and not plan.is_policy_node(2)
    return plan


# --------------------------------------------------------------------------------------
# The two helpers' own refusals.
# --------------------------------------------------------------------------------------


def test_stitch_takes_each_row_from_the_fold_that_holds_it_out(static: LongitudinalResult) -> None:
    folds = static.folds
    shape = _plan(static, "always").values.shape
    assignments = [np.full(shape, float(fold)) for fold in range(folds.n_folds)]
    stitched = stitch_fold_assignments(assignments, folds)
    assert np.array_equal(stitched[:, 0], np.asarray(folds.assignment, dtype=float))
    with pytest.raises(ValueError, match="one assignment per outer fold"):
        stitch_fold_assignments(assignments[:-1], folds)


def test_with_assignment_refuses_a_wrong_shape_and_a_drawn_node() -> None:
    result = fit_policy()
    plan = _plan(result, "mix")
    with pytest.raises(ValueError, match="must have the shape"):
        plan.with_assignment(plan.values[:, :1])
    moved = _own_copy(plan.values)
    moved[:, 1] = 1.0 - np.nan_to_num(moved[:, 1])
    with pytest.raises(ValueError, match="draws its arm"):
        plan.with_assignment(moved)
    same = plan.with_assignment(plan.values)
    assert np.array_equal(same.values, plan.values, equal_nan=True)
    assert same.values is not plan.values


def test_with_assignment_refuses_a_tilt_node_before_its_density(
    static: LongitudinalResult,
) -> None:
    """Only ``is_tilt_node`` guards a tilt column before ``with_tilts`` makes it a policy."""
    plan = _tilt_plan(static)
    moved = _own_copy(plan.values)
    moved[:, 1] = 1.0 - np.nan_to_num(moved[:, 1])
    with pytest.raises(ValueError, match="node 2 of regimen 'tilt' draws its arm"):
        plan.with_assignment(moved)
    moved = _own_copy(plan.values)
    moved[:, 0] = 0.0
    assert np.array_equal(plan.with_assignment(moved).values[:, 0], np.zeros(static.data.n))
