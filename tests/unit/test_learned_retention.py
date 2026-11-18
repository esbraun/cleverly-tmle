"""A learned-regimen fit carries int8 rule codes and builds its record before the recursion.

X106 part J3d.  Each fold's rule codes are int8 from the fold's return, the record is built
right after learning, and the recursion then carries only the codes and the split, not every
fold's blips.  The record is compared bitwise with a verbatim copy of the function that built it
before, run on float codes, and the fit's record with that copy applied to the rules the fit
learned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import learned
from cleverly.longitudinal.learned import (
    LearnedRegimenNode,
    LearnedRegimenRecord,
    LearnedRules,
    _disagreement,
    _quantiles,
)
from tests.unit.test_longitudinal_learned_regimen import fit, logistic, panel


def _baseline_record(data: Any, plan: Any, rules: LearnedRules, *, seed: Any) -> Any:
    """``learned_regimen_record`` as it stood before X106 J3d (main 70ccd44)."""
    folds = rules.folds
    assignments = np.stack(rules.assignments)
    blips = np.stack(rules.blips)
    uncensored = plan.with_assignment(np.zeros_like(plan.values)).masks(data).uncensored[:, 1]
    first_arm = np.asarray(data.node_codes(1), dtype=float)
    shares: list[list[float]] = [[], []]
    gaps: list[list[tuple[float, ...]]] = [[], []]
    for fold, (_, test) in enumerate(folds):
        held = np.zeros(data.n, dtype=bool)
        held[test] = True
        at_risk = (held, held & uncensored & (first_arm == assignments[fold, :, 0]))
        for column in (0, 1):
            rows = at_risk[column]
            treated = assignments[fold, rows, column]
            shares[column].append(float(np.mean(treated)) if treated.size else float("nan"))
            gaps[column].append(_quantiles(blips[fold, rows, column]))
    nodes = tuple(
        LearnedRegimenNode(
            node=column + 1,
            treatment=str(data.treatment_names[column]),
            arms=tuple(str(level) for level in data.treatment_levels[column]),  # type: ignore[arg-type]
            treated_shares=tuple(shares[column]),
            gap_quantiles=tuple(gaps[column]),
            rule_disagreement=_disagreement(
                assignments[:, :, column],
                np.ones(data.n, dtype=bool) if column == 0 else np.asarray(uncensored),
            ),
        )
        for column in (0, 1)
    )
    return LearnedRegimenRecord(
        label=plan.label,
        seed=seed,
        fold_sizes=tuple(int(np.sum(folds.assignment == fold)) for fold in range(folds.n_folds)),
        nodes=nodes,
        fold_rules=assignments.astype(np.int8),
        fold_blips=blips,
    )


def _as_float(rules: LearnedRules) -> LearnedRules:
    return LearnedRules(
        assignments=tuple(np.asarray(a, dtype=float) for a in rules.assignments),
        blips=rules.blips,
        second_node_blips=rules.second_node_blips,
        folds=rules.folds,
    )


def _same_record(one: LearnedRegimenRecord, two: LearnedRegimenRecord) -> None:
    assert one.fold_rules.dtype == two.fold_rules.dtype == np.int8
    assert one.fold_rules.tobytes() == two.fold_rules.tobytes()
    assert one.fold_blips.tobytes() == two.fold_blips.tobytes()
    assert (one.label, one.seed, one.fold_sizes) == (two.label, two.seed, two.fold_sizes)
    for a, b in zip(one.nodes, two.nodes, strict=True):
        assert (a.node, a.treatment, a.arms) == (b.node, b.treatment, b.arms)
        assert np.asarray(a.treated_shares).tobytes() == np.asarray(b.treated_shares).tobytes()
        assert np.asarray(a.gap_quantiles).tobytes() == np.asarray(b.gap_quantiles).tobytes()
        assert (
            np.float64(a.rule_disagreement).tobytes() == np.float64(b.rule_disagreement).tobytes()
        )


class _Spy:
    """Captures the rules a fit learned and the rules its recursion received."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cleverly.longitudinal import estimator

        self.learned: list[tuple[Any, Any, LearnedRules]] = []
        self.recursed: list[LearnedRules] = []
        self.events: list[str] = []
        learn, target = estimator.learn_fold_rules, estimator.fit_learned_regimen
        record = estimator.learned_regimen_record

        def record_spy(*args: Any, **kwargs: Any) -> Any:
            self.events.append("record")
            return record(*args, **kwargs)

        def learn_spy(data: Any, plan: Any, **kwargs: Any) -> LearnedRules:
            rules = learn(data, plan, **kwargs)
            self.learned.append((data, plan, rules))
            return rules

        def target_spy(data: Any, plan: Any, mechanism: Any, rules: Any, **kwargs: Any) -> Any:
            self.recursed.append(rules)
            self.events.append("recursion")
            return target(data, plan, mechanism, rules, **kwargs)

        monkeypatch.setattr(estimator, "learn_fold_rules", learn_spy)
        monkeypatch.setattr(estimator, "fit_learned_regimen", target_spy)
        monkeypatch.setattr(estimator, "learned_regimen_record", record_spy)


@pytest.mark.parametrize(
    "overrides", [{}, {"n_folds": 5, "random_state": 3}, {"n_jobs": 2}], ids=str
)
def test_the_fit_record_is_the_baseline_record(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]
) -> None:
    spy = _Spy(monkeypatch)
    result = fit(panel(600, seed=4), **overrides)
    ((data, plan, rules),) = spy.learned
    # Codes are int8 0/1 from each fold's return; the blips stay float.
    for codes in rules.assignments:
        assert codes.dtype == np.int8 and set(np.unique(codes)) <= {0, 1}
    record = result.learned_regimen
    assert record is not None
    stitched = plan.with_assignment(rules.stitched())
    _same_record(record, _baseline_record(data, stitched, _as_float(rules), seed=record.seed))
    # The fit's stitched assignment is the float stitch of float codes.
    (fitted,) = result.fits.values()
    float_rules = _as_float(rules)
    assert np.asarray(fitted.assignment).tobytes() == float_rules.stitched().tobytes()
    # The recursion received only the codes and the split.
    ((carried,),) = (spy.recursed,)
    assert carried.blips == () and carried.second_node_blips == ()
    assert carried.assignments is rules.assignments
    # The record was built from the full rules before the recursion started.
    assert spy.events == ["record", "recursion"]


def test_the_record_reaches_censoring_and_both_arms() -> None:
    """Nonzero premises: censoring after the first node (nan blips), shares strictly inside
    (0, 1), and folds that disagree on some rows."""
    result = fit(panel(600, seed=4))
    record = result.learned_regimen
    assert record is not None
    assert np.isnan(record.fold_blips[:, :, 1]).any()
    assert all(0.0 < share < 1.0 for node in record.nodes for share in node.treated_shares)
    assert all(node.rule_disagreement > 0.0 for node in record.nodes)


def test_a_flipped_code_is_seen() -> None:
    """Mutation control: one code changed in one fold moves the record's shares."""
    result = fit(panel(600, seed=4))
    data = result.data
    rules = learned.learn_fold_rules(
        data,
        learned.learned_plan(learned.LearnedRegimen(), data),
        folds=result.folds,
        scaler=result.scaler,
        outcome_learner=logistic(),
        pseudo_learner=LinearRegression(),
    )
    plan = learned.learned_plan(learned.LearnedRegimen(), data).with_assignment(rules.stitched())
    flipped = [a.copy() for a in rules.assignments]
    (_, test) = next(iter(result.folds))
    flipped[0][test[0], 0] = 1 - flipped[0][test[0], 0]
    mutated = LearnedRules(tuple(flipped), rules.blips, rules.second_node_blips, rules.folds)
    one = learned.learned_regimen_record(data, plan, rules, seed=0)
    two = learned.learned_regimen_record(data, plan, mutated, seed=0)
    assert one.nodes[0].treated_shares != two.nodes[0].treated_shares


def _learned(result: Any) -> tuple[Any, Any, LearnedRules]:
    data = result.data
    plan = learned.learned_plan(learned.LearnedRegimen(), data)
    rules = learned.learn_fold_rules(
        data,
        plan,
        folds=result.folds,
        scaler=result.scaler,
        outcome_learner=logistic(),
        pseudo_learner=LinearRegression(),
    )
    return data, plan, rules


@pytest.mark.parametrize("case", ["treats nobody", "nobody at risk"])
def test_degenerate_folds_match_the_baseline(case: str) -> None:
    """A fold whose rule treats nobody (share 0) and a fold with no held row at risk at the
    second node (empty ``treated``, share NaN) give the baseline record."""
    data, plan, rules = _learned(fit(panel(600, seed=4)))
    codes = [a.copy() for a in rules.assignments]
    (_, test) = next(iter(rules.folds))
    if case == "treats nobody":
        codes[0][test, :] = 0
    else:
        # Every held row's first code disagrees with the arm it received: nobody follows.
        first = np.asarray(data.node_codes(1), dtype=float)
        codes[0][test, 0] = (1.0 - first[test]).astype(np.int8)
    rules = LearnedRules(tuple(codes), rules.blips, rules.second_node_blips, rules.folds)
    stitched = plan.with_assignment(rules.stitched())
    ours = learned.learned_regimen_record(data, stitched, rules, seed=0)
    _same_record(ours, _baseline_record(data, stitched, _as_float(rules), seed=0))
    shares = ours.nodes[0 if case == "treats nobody" else 1].treated_shares
    if case == "treats nobody":
        assert shares[0] == 0.0
    else:
        assert np.isnan(shares[0])


def test_float_codes_still_give_an_int8_record() -> None:
    """The study law and two stubs pass float codes; the record's codes stay int8 for them."""
    data, plan, rules = _learned(fit(panel(600, seed=4)))
    floats = _as_float(rules)
    assert floats.assignments[0].dtype == np.float64
    stitched = plan.with_assignment(floats.stitched())
    ours = learned.learned_regimen_record(data, stitched, floats, seed=0)
    _same_record(ours, _baseline_record(data, stitched, floats, seed=0))
