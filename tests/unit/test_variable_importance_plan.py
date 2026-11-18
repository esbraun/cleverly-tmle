"""``variable_importance_plan`` counts the fits ``variable_importance`` makes, and fits nothing.

The witness is part I's: counting learners and a class-wide ``fit`` patch, so the preview must
fit nothing and each priced role must equal the realized count. Each candidate's rows are also
compared with the fits that candidate's own run made, which the per-candidate scopes allow.
"""

from __future__ import annotations

import importlib
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cleverly import variable_importance, variable_importance_plan
from cleverly._planned_fits import PlanBuilder, PlannedFits
from cleverly.datasets import make_linear_ate
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, DataError
from tests.unit import _planned_fits_support as support
from tests.unit.test_ctmle import _W4, CTMLE_KWARGS
from tests.unit.test_planned_fits import PLAIN
from tests.unit.test_variable_importance_retention import REFUSALS, _Counting

module = importlib.import_module("cleverly.variable_importance")
planned_fits_module = importlib.import_module("cleverly._planned_fits")
BOUNDS = (-30.0, 30.0)


def _frame(n: int = 200, seed: int = 4, levels: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    x1 = rng.binomial(1, 1 / (1 + np.exp(-w)))
    x2 = rng.integers(0, levels, n)
    x3 = rng.binomial(1, 0.4, n)
    y = 1.2 * x1 + 0.3 * x2 + 0.2 * w + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {"Y": y, "X1": x1, "X2": x2, "X3": x3, "W": w, "cid": np.arange(n) // 4, "wt": 1.0 + w**2}
    )


def _tmle(**overrides: Any) -> TMLE:
    settings: dict[str, Any] = {
        "outcome_learner": support.linear("outcome_learner"),
        "treatment_learner": support.logistic("treatment_learner"),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return TMLE(**settings)


#: name -> (frame, keyword arguments built fresh per call)
CASES: dict[str, tuple[Callable[[], pd.DataFrame], Callable[[], dict[str, Any]]]] = {
    "in sample": (_frame, lambda: {"estimator": _tmle()}),
    "cross-fitted": (
        _frame,
        lambda: {"estimator": _tmle(cross_fit=True, n_folds=3, q_bounds=BOUNDS)},
    ),
    "a Super Learner outcome": (
        _frame,
        lambda: {
            "estimator": _tmle(
                outcome_learner=support.ensemble(
                    "outcome_learner", base=LinearRegression, n_folds=3, random_state=0
                )
            )
        },
    ),
    "clustered and weighted": (
        _frame,
        lambda: {
            "estimator": _tmle(cross_fit=True, n_folds=3, q_bounds=BOUNDS),
            "id": "cid",
            "weights": "wt",
        },
    ),
    "one at a time": (
        _frame,
        lambda: {"estimator": _tmle(), "adjust_for_other_candidates": False},
    ),
    "a three-level candidate beside binary ones": (
        lambda: _frame(levels=3),
        lambda: {"estimator": _tmle()},
    ),
    "a DR-TMLE template": (
        _frame,
        lambda: {
            "estimator": DRTMLE(
                outcome_learner=support.linear("outcome_learner"),
                treatment_learner=support.logistic("treatment_learner"),
                reduced_outcome_learner=support.linear("reduced_outcome_learner"),
                reduced_mechanism_residual_learner=support.linear(
                    "reduced_mechanism_residual_learner"
                ),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
            )
        },
    ),
}
PLAIN_ROLES = {
    **PLAIN,
    "reduced_outcome_learner": "LinearRegression",
    "reduced_mechanism_residual_learner": "LinearRegression",
}


def _arguments(case: str) -> tuple[pd.DataFrame, Callable[[], dict[str, Any]]]:
    make_frame, settings = CASES[case]
    frame = make_frame()

    def fresh() -> dict[str, Any]:
        return {
            "outcome": "Y",
            "candidates": ("X1", "X2", "X3"),
            "covariates": ("W",),
            **settings(),
        }

    return frame, fresh


def _per_candidate(monkeypatch: pytest.MonkeyPatch) -> dict[str, Counter[str]]:
    """Record the labelled fits each candidate's own fit made, by its treatment name."""
    seen: dict[str, Counter[str]] = {}
    for engine in (TMLE, DRTMLE):
        original = engine.fit

        def fit(self: Any, data: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            before = Counter(support.COUNTS)
            out = _original(self, data, *args, **kwargs)
            delta = Counter(support.COUNTS)
            delta.subtract(before)
            seen[data.treatment_name] = +delta
            return out

        monkeypatch.setattr(engine, "fit", fit)
    return seen


def _candidate_rows(plan: PlannedFits) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for row in plan.rows:
        name = re.fullmatch(r".*, candidate '(\w+)'", row.scope)
        assert name is not None, row.scope
        out.setdefault(name.group(1), Counter())[row.role] += row.fits
    return out


def _run(
    case: str, keep_fits: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[support.Witness, dict[str, Counter[str]]]:
    frame, fresh = _arguments(case)
    seen = _per_candidate(monkeypatch)
    found = support.witness(
        lambda: variable_importance_plan(frame, **fresh()),
        lambda: variable_importance(frame, keep_fits=keep_fits, **fresh()),
    )
    return found, seen


def _check_candidates(plan: PlannedFits, seen: dict[str, Counter[str]]) -> None:
    rows = _candidate_rows(plan)
    assert set(rows) == set(seen)
    for name, priced in rows.items():
        realized = Counter({role: support.realized_fits(seen[name], role) for role in priced})
        if plan.exact:
            assert +realized == +priced, (name, dict(priced), dict(seen[name]))
        else:
            assert all(realized[role] <= priced[role] for role in priced), (name, priced)


@pytest.mark.parametrize("keep_fits", [True, False], ids=["kept fits", "no kept fits"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_preview_counts_every_fit_of_the_run(
    case: str, keep_fits: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    found, seen = _run(case, keep_fits, monkeypatch)
    # DR-TMLE's reduced regressions are priced as an upper bound by its own preview.
    assert found.plan.exact == (case != "a DR-TMLE template")
    support.check(found, plain=PLAIN_ROLES)
    _check_candidates(found.plan, seen)
    assert dict(found.plan.multipliers)["candidate exposures"] == 3


def test_a_multiplier_that_differs_between_candidates_is_named_not_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found, _ = _run("a three-level candidate beside binary ones", True, monkeypatch)
    plan = found.plan
    per_candidate = {
        name: dict(
            variable_importance_plan(
                _frame(levels=3), outcome="Y", candidates=(name,), covariates=("W", *others),
                estimator=_tmle(),
            ).multipliers
        )
        for name, others in (("X1", ("X2", "X3")), ("X2", ("X1", "X3")))
    }  # fmt: skip
    differing = {
        factor
        for factor in set(per_candidate["X1"]) | set(per_candidate["X2"])
        if per_candidate["X1"].get(factor) != per_candidate["X2"].get(factor)
    } - {"candidate exposures"}
    assert differing, "the fixture must make a multiplier differ between candidates"
    for factor in differing:
        assert factor not in dict(plan.multipliers)
        assert any(f"differ in {factor}" in note for note in plan.notes)
    shared = set(per_candidate["X1"]) & set(per_candidate["X2"]) - differing - {
        "candidate exposures"
    }
    for factor in shared:
        assert dict(plan.multipliers)[factor] == per_candidate["X1"][factor]


def test_counting_one_candidate_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberate mutation: the preview prices the first candidate only.
    original = module._checked_candidates

    def first_only(*args: Any, **kwargs: Any) -> Any:
        yield next(original(*args, **kwargs))

    frame, fresh = _arguments("in sample")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_checked_candidates", first_only)
        plan = variable_importance_plan(frame, **fresh())
    found = support.witness(lambda: plan, lambda: variable_importance(frame, **fresh()))
    with pytest.raises(AssertionError):
        support.check(found, plain=PLAIN_ROLES)


def test_scopes_that_collide_across_candidates_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberate mutation: every candidate's rows keep their own scope, so they merge.
    original = PlanBuilder.absorb

    def merged(self: PlanBuilder, plan: PlannedFits, scope: Any, **kwargs: Any) -> None:
        original(self, plan, lambda text: f"{text}, candidate 'X1'", **kwargs)

    frame, fresh = _arguments("in sample")
    seen = _per_candidate(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(PlanBuilder, "absorb", merged)
        plan = variable_importance_plan(frame, **fresh())
    support.reset()
    variable_importance(frame, **fresh())
    with pytest.raises(AssertionError):
        _check_candidates(plan, seen)


def test_multipliers_not_copied_are_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberate mutation of the outer record only: it keeps the candidate count and drops
    # every multiplier the candidates' own previews report. The engines' previews keep the
    # PlanBuilder they imported, so their multipliers are intact.
    class CandidatesOnly(PlanBuilder):
        def multiply(self, name: str, value: int) -> None:
            if name == "candidate exposures":
                super().multiply(name, value)

    frame, fresh = _arguments("cross-fitted")
    plan = variable_importance_plan(frame, **fresh())
    single = TMLE(
        outcome_learner=LinearRegression(), cross_fit=True, n_folds=3, q_bounds=BOUNDS,
        simultaneous=False, random_state=0,
    )._planned_fits(frame, outcome="Y", treatment="X1", covariates=["W", "X2", "X3"])  # fmt: skip
    assert dict(single.multipliers)
    assert set(dict(single.multipliers)) <= set(dict(plan.multipliers))
    with monkeypatch.context() as patch:
        patch.setattr(planned_fits_module, "PlanBuilder", CandidatesOnly)
        mutated = variable_importance_plan(frame, **fresh())
    assert not set(dict(single.multipliers)) & set(dict(mutated.multipliers))


def _ctmle_refusal() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, _ = make_linear_ate(n=300, seed=21)
    frame = frame.assign(B=(frame["W4"] > 0).astype(float))
    estimator = CTMLE(**{**CTMLE_KWARGS, "strategy": "discrete", "candidates": [_W4]})
    return frame, {
        "outcome": "Y",
        "candidates": ["A", "B"],
        "covariates": list(_W4),
        "estimator": estimator,
    }


def _later_check_refusal() -> tuple[pd.DataFrame, dict[str, Any]]:
    # The first candidate's fit would refuse the ratio on a continuous outcome inside its own
    # preflight, and the second candidate has a missing value. The run checks every candidate
    # before its first fit, so it raises the second candidate's refusal; so must the preview.
    frame = _frame()
    frame["X2"] = frame["X2"].astype(float).where(frame.index != 5)
    return frame, {
        "outcome": "Y",
        "candidates": ["X1", "X2"],
        "covariates": ["W"],
        "estimand": "rr",
        "adjust_for_other_candidates": False,
        "estimator": TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=_Counting(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=1,
        ),
    }


@pytest.mark.parametrize(
    "case", [*REFUSALS, "a collaborative inference status", "a later candidate's check"]
)
def test_the_preview_refuses_as_the_run_does_and_fits_nothing(case: str) -> None:
    phrase = ""
    if case in REFUSALS:
        frame, extra, kind, phrase = REFUSALS[case]()
        settings: dict[str, Any] = {
            "outcome": "Y",
            "candidates": ("X1", "X2"),
            "covariates": ("W",),
            "estimator": TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=_Counting(max_iter=1000),
                cross_fit=True,
                simultaneous=False,
                random_state=1,
            ),
            **extra,
        }
        settings["estimator"].treatment_learner = _Counting(max_iter=1000)
    elif case == "a later candidate's check":
        frame, settings = _later_check_refusal()
        kind, phrase = DataError, "candidate 'X2' has missing values"
    else:
        frame, settings = _ctmle_refusal()
        kind = CapabilityError
    messages = []
    for call in (variable_importance_plan, variable_importance):
        _Counting.calls = 0
        support.reset()
        with support.class_wide(), pytest.raises(kind, match=re.escape(phrase)) as raised:
            call(frame, **settings)
        assert _Counting.calls == 0
        assert sum(support.CLASS_WIDE.values()) == 0
        messages.append((type(raised.value), str(raised.value)))
    assert messages[0] == messages[1]
