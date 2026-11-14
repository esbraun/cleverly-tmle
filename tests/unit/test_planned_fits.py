"""The planned-fit preview counts the learner fits ``estimate`` makes, and fits nothing.

Each route fixture previews a fit, runs it with counting learners, and compares the counts
(:func:`tests.unit._planned_fits_support.check`): an exact row equals the realized count per
role and per candidate, an upper row brackets it, the class-wide total equals the labelled
total, and the preview itself fits nothing.  The mutation controls patch one term out of
the preview and show the comparison then fails.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import cleverly
from cleverly import ATE, CausalStudy, PlannedFits, PointTreatment
from cleverly import _planned_fits as planned_module
from cleverly.datasets import (
    make_cde,
    make_clustered,
    make_linear_ate,
    make_missing_outcome,
    make_multi_arm,
    make_shift_dose,
)
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, LearnedRule, Shift, SupportBoundary
from cleverly.msm import MSM
from tests import law_q_point as lawp
from tests.unit import _planned_fits_support as support
from tests.unit._natural_course_support import NeverFit

tmle_module = importlib.import_module("cleverly.estimators.tmle")

COVARIATES = ["W1", "W2", "W3", "W4"]
COLUMNS: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": COVARIATES}
PLAIN: dict[str, str | None] = {
    "outcome_learner": "LinearRegression",
    "treatment_learner": "LogisticRegression",
    "missingness_learner": "LogisticRegression",
    "intermediate_learner": "LogisticRegression",
    "learned_rule": "LinearRegression",
    "tuner": None,
}
ROLES = ("outcome_learner", "treatment_learner", "missingness_learner", "intermediate_learner")
BOUNDS = (-30.0, 30.0)


def _plain(**overrides: Any) -> dict[str, Any]:
    """Counting linear learners on every role, three outer folds, seeded."""
    return {
        "outcome_learner": support.linear("outcome_learner"),
        "treatment_learner": support.logistic("treatment_learner"),
        "missingness_learner": support.logistic("missingness_learner"),
        "intermediate_learner": support.logistic("intermediate_learner"),
        "n_folds": 3,
        "random_state": 0,
        "simultaneous": False,
        "q_bounds": BOUNDS,
        **overrides,
    }


def _linear_frame(n: int = 200) -> pd.DataFrame:
    frame, _ = make_linear_ate(n=n, seed=2)
    return frame


def _engine(
    settings: Callable[[], dict[str, Any]],
    frame: pd.DataFrame,
    columns: dict[str, Any],
    mutate: Callable[[pytest.MonkeyPatch], None] | None = None,
) -> support.Witness:
    """The engine's preview and its fit on the same arguments, with fresh learners each."""

    def preview() -> PlannedFits:
        if mutate is None:
            return TMLE(**settings())._planned_fits(frame, **columns)
        with pytest.MonkeyPatch.context() as patch:
            mutate(patch)
            return TMLE(**settings())._planned_fits(frame, **columns)

    return support.witness(preview, lambda: TMLE(**settings()).fit(frame, **columns))


# ----------------------------------------------------------------- the route matrix


def _dose() -> pd.DataFrame:
    frame, _ = make_shift_dose(n=240, seed=0)
    return frame


def _missing_treatment(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    out = frame.copy()
    recorded = rng.uniform(size=len(out)) > 0.2
    out["DeltaA"] = recorded.astype(int)
    out.loc[~recorded, "A"] = np.nan
    return out


def _known_weighted() -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(4)
    n = 240
    w = rng.normal(size=(n, 2))
    p1 = 1.0 / (1.0 + np.exp(-0.4 * w[:, 0]))
    a = (rng.uniform(size=n) < p1).astype(float)
    y = w[:, 0] + a + rng.normal(size=n)
    frame = pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": a, "Y": y, "p1": p1, "p0": 1 - p1})
    frame["w"] = rng.uniform(0.5, 1.5, size=n) * (1.0 + a)
    columns = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": ["W1", "W2"],
        "weights": "w",
        "treatment_probabilities": {0.0: "p0", 1.0: "p1"},
    }
    return frame, columns


def _rule_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.3 * w1 - 0.2 * w2)))).astype(float)
    y = w1 + a * (0.1 + w1) + rng.normal(size=n)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


def _constant_arm_frame() -> pd.DataFrame:
    """A binary outcome that is 1 on every treated row, so every treated training fold is constant."""
    frame = _linear_frame(240)
    rng = np.random.default_rng(3)
    frame["Y"] = (rng.uniform(size=len(frame)) < 0.5).astype(int)
    frame.loc[frame["A"] == 1, "Y"] = 1
    return frame


def _three_arms(rare: int | None = None) -> pd.DataFrame:
    frame, _ = make_multi_arm(n=300, seed=0)
    if rare is not None:
        arms = np.sort(frame["A"].unique())
        rows = frame.index[frame["A"] == arms[-1]]
        keep = set(rows[:rare])
        frame = frame[(frame["A"] != arms[-1]) | frame.index.isin(keep)].reset_index(drop=True)
    return frame


def _arm_columns(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "outcome": "Y",
        "treatment": "A",
        "covariates": [c for c in frame.columns if c.startswith("W")],
    }


def _tuner_learner(label: str) -> Any:
    return support.counting(GradientBoostingRegressor, label)(
        loss="quantile", alpha=0.5, n_estimators=5, random_state=0
    )


class _NoWeightLinear(Ridge):
    """A boundary learner whose ``fit`` takes no ``sample_weight``; each attempt counts.

    It is not one of the class-wide guard's base classes, so its failed attempts and its
    fits count under ``"tuner"`` alone.
    """

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
        support.COUNTS["tuner"] += 1
        if kwargs:
            raise TypeError(f"fit() got an unexpected keyword argument {next(iter(kwargs))!r}")
        return super().fit(X, y)

    fit.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("X", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )


DOSE_COLUMNS = {"outcome": "Y", "treatment": "A", "treatment_kind": "continuous"}
TUNED_COLUMNS = {
    "outcome": "Y",
    "treatment": "A",
    "covariates": ["W"],
    "treatment_kind": "continuous",
}


def _tuned(learner: Any, **overrides: Any) -> Callable[[], dict[str, Any]]:
    def settings() -> dict[str, Any]:
        cap = SupportBoundary(learner(), 0.5, features=("W",))
        return _plain(
            policies=[
                Shift(1.0, cap=cap, target="fold_average", name="tuned"),
                Shift(0.0, cap=None),
            ],
            density_bins=10,
            q_bounds=(0.0, 1.0),
            **overrides,
        )

    return settings


#: name -> (settings factory, frame factory, columns, plain-role names)
ROUTES: dict[
    str, tuple[Callable[[], dict[str, Any]], Callable[[], pd.DataFrame], dict[str, Any]]
] = {
    "ate_cross_fit": (lambda: _plain(), _linear_frame, COLUMNS),
    "ate_in_sample": (lambda: _plain(cross_fit=False), _linear_frame, COLUMNS),
    "ate_repeats": (lambda: _plain(repeats=2), _linear_frame, COLUMNS),
    "att_rr": (lambda: _plain(estimands=("att", "atc")), _linear_frame, COLUMNS),
    "ensembles": (
        lambda: _plain(
            outcome_learner=support.ensemble("outcome_learner", LinearRegression, n_folds=4),
            treatment_learner=support.ensemble("treatment_learner", n_folds=4),
        ),
        _linear_frame,
        COLUMNS,
    ),
    "three_arms_one_vs_rest": (
        lambda: _plain(
            treatment_learner=support.ensemble(
                "treatment_learner", task="classification", n_folds=4
            )
        ),
        _three_arms,
        {},
    ),
    "rare_arm_caps_a_member": (
        lambda: _plain(
            treatment_learner=support.ensemble(
                "treatment_learner", task="classification", n_folds=10
            )
        ),
        lambda: _three_arms(rare=12),
        {},
    ),
    "rare_class_caps_inner_folds": (
        lambda: _plain(
            n_folds=5, treatment_learner=support.ensemble("treatment_learner", n_folds=10)
        ),
        lambda: support.rare_class_frame(n=240, rare=12),
        {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]},
    ),
    "outcome_by_arm": (lambda: _plain(outcome_by_arm=True), _linear_frame, COLUMNS),
    "constant_binary_arm": (
        lambda: _plain(outcome_by_arm=True, q_bounds=None),
        _constant_arm_frame,
        COLUMNS,
    ),
    "missing_outcome": (
        lambda: _plain(cross_fit=False),
        lambda: make_missing_outcome(n=240, seed=4)[0],
        {"outcome": "Y", "treatment": "A", "delta": "Delta"},
    ),
    "natural_course": (
        lambda: _plain(cross_fit=False, estimands=("ey_obs",)),
        lambda: make_missing_outcome(n=240, seed=4)[0],
        {"outcome": "Y", "treatment": "A", "delta": "Delta"},
    ),
    "missing_treatment": (
        lambda: _plain(cross_fit=False),
        lambda: _missing_treatment(_linear_frame(240)),
        {**COLUMNS, "treatment_delta": "DeltaA"},
    ),
    "known_sample_law": (
        lambda: _plain(cross_fit=False),
        lambda: _known_weighted()[0],
        {**_known_weighted()[1], "mechanism_law": "sample"},
    ),
    "known_target_law": (
        lambda: _plain(cross_fit=False),
        lambda: _known_weighted()[0],
        {**_known_weighted()[1], "mechanism_law": "target"},
    ),
    "strata": (
        lambda: _plain(),
        lambda: _linear_frame().assign(S=lambda f: (f["W2"] > 0).astype(int)),
        {**COLUMNS, "covariates": [*COVARIATES, "S"], "strata": ["S"]},
    ),
    "clustered": (
        lambda: _plain(n_folds=5, cross_fit=False),
        lambda: make_clustered(n=200, cluster_size=5, seed=0)[0],
        {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"], "id": "cluster"},
    ),
    "msm": (lambda: _plain(msm=MSM.linear()), _linear_frame, COLUMNS),
    "ipsi_two_tilts": (
        lambda: _plain(incremental=[Incremental(1.0), Incremental(2.0)]),
        _linear_frame,
        COLUMNS,
    ),
    "mtp_density": (
        lambda: _plain(policies=[Shift(0.5, cap=5.0)], density_bins=8),
        _dose,
        DOSE_COLUMNS,
    ),
    "mtp_classifier_two_branches": (
        lambda: _plain(
            policies=[Shift(0.5, cap=5.0, name="a"), Shift(1.0, cap=5.0, name="b")],
            ratio="classifier",
        ),
        _dose,
        DOSE_COLUMNS,
    ),
    "learned_rule_blip": (
        lambda: _plain(
            learned_rule=LearnedRule(loss="blip", learner=support.linear("learned_rule"))
        ),
        _rule_frame,
        {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]},
    ),
    "tuned_cap_learner": (
        _tuned(lambda: _tuner_learner("tuner")),
        lambda: lawp.sample(240, 11),
        TUNED_COLUMNS,
    ),
    "tuned_cap_weight_retry": (
        _tuned(_NoWeightLinear),
        lambda: lawp.sample(240, 11),
        TUNED_COLUMNS,
    ),
    "bootstrap_no_binding_cap": (lambda: _plain(n_bootstrap=2), _linear_frame, COLUMNS),
    "bootstrap_repeats": (lambda: _plain(n_bootstrap=2, repeats=2), _linear_frame, COLUMNS),
    "cde_bootstrap": (
        lambda: _plain(n_bootstrap=2, cross_fit=False),
        lambda: make_cde(n=200, seed=3)[0],
        {"outcome": "Y", "treatment": "A", "intermediate": "Z"},
    ),
}


def _route(
    name: str, mutate: Callable[[pytest.MonkeyPatch], None] | None = None
) -> support.Witness:
    settings, frame, columns = ROUTES[name]
    data = frame()
    return _engine(settings, data, columns or _arm_columns(data), mutate)


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_the_preview_counts_the_fits_of_each_route(name: str) -> None:
    found = _route(name)
    support.check(found, plain=PLAIN)


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_the_preview_fits_nothing(name: str) -> None:
    settings, frame, columns = ROUTES[name]
    data = frame()

    def never() -> dict[str, Any]:
        values = settings()
        for role in ROLES:
            values[role] = NeverFit()
        return values

    NeverFit.calls = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = TMLE(**never())._planned_fits(data, **(columns or _arm_columns(data)))
    assert NeverFit.calls == 0
    assert plan.total_fits > 0


def test_exact_routes_are_exact_and_bootstrap_routes_are_upper() -> None:
    assert _route("ate_repeats").plan.exact
    found = _route("bootstrap_no_binding_cap")
    assert not found.plan.exact
    assert found.result.single().bootstrap.n_failed == 0


def test_with_no_binding_cap_the_bootstrap_bound_is_reached() -> None:
    found = _route("bootstrap_no_binding_cap")
    for role, fits in support.totals(found.plan).items():
        assert support.realized_fits(found.counts, role) == fits


def test_the_repeat_and_bootstrap_factors() -> None:
    plan = _route("bootstrap_repeats").plan
    per_draw = 3 * 2  # three folds, treatment and outcome
    assert plan.total_fits == per_draw * 2 * (1 + 2)
    assert dict(plan.multipliers)["repeats"] == 2


def test_the_cde_bootstrap_runs_once_per_level() -> None:
    plan = _route("cde_bootstrap").plan
    shared = sum(row.fits for row in plan.rows if "bootstrap" not in row.scope)
    replicates = sum(row.fits for row in plan.rows if "bootstrap" in row.scope)
    assert shared == 3  # treatment, intermediate and outcome, in sample
    assert replicates == 2 * 2 * shared


def test_one_vs_rest_sums_the_classes() -> None:
    plan = _route("three_arms_one_vs_rest").plan
    (row,) = [row for row in plan.rows if row.role == "treatment_learner"]
    assert row.fits == 3
    # Each fold fits three binary ensembles of two candidates, each over four inner folds.
    assert row.candidate_fits == 3 * 3 * 2 * (4 + 1)


def test_a_rare_class_caps_the_inner_folds() -> None:
    plan = _route("rare_class_caps_inner_folds").plan
    (row,) = [row for row in plan.rows if row.role == "treatment_learner"]
    assert row.candidate_fits < 5 * 2 * (10 + 1)


def test_a_constant_fold_fits_no_learner() -> None:
    plan = _route("constant_binary_arm").plan
    (row,) = [row for row in plan.rows if row.role == "outcome_learner"]
    assert row.fits == 3  # the untreated arm in three folds; the treated arm is constant


def test_the_tuner_retry_counts_two_attempts() -> None:
    plan = _route("tuned_cap_weight_retry").plan
    tuner = [row for row in plan.rows if row.role == "tuner"]
    # Once in sample at the preflight, then on every row and in three folds: two attempts each.
    assert sum(row.fits for row in tuner) == 2 * (1 + 1 + 3)
    assert "in_sample_policies" in plan.unpreviewed


def test_the_folds_are_one_draw_without_a_seed() -> None:
    plan = TMLE(**_plain(random_state=None))._planned_fits(_linear_frame(), **COLUMNS)
    assert plan.folds == "one draw"
    assert plan.total_fits == 6


# ----------------------------------------------------------------- the default library


@pytest.fixture
def study() -> CausalStudy:
    frame, _ = make_linear_ate(n=400, seed=1)
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=tuple(COVARIATES))
    )


def test_the_default_library_makes_360_candidate_fits(study: CausalStudy) -> None:
    """A binary ATE on the default library: 2 roles x 10 folds x 3 candidates x (5 + 1)."""
    effect = study.identify(ATE())
    with support.counting_default_library():
        found = support.witness(
            lambda: effect.planned_fits(random_state=0, q_bounds=BOUNDS),
            lambda: effect.estimate(random_state=0, q_bounds=BOUNDS),
        )
    plan = found.plan
    assert plan.exact
    assert plan.total_fits == 20
    assert plan.total_candidate_fits == 360
    assert {name: fits for name, fits, _ in plan.candidates} == {
        "hist_gradient_boosting": 120,
        "random_forest": 120,
        "lasso": 120,
    }
    for task in ("classification", "regression"):
        for name in ("hist_gradient_boosting", "random_forest", "lasso"):
            assert found.counts[f"{task}:{name}"] == 60
    assert sum(found.class_wide.values()) == 360


def test_the_default_lasso_reports_its_own_folds(study: CausalStudy) -> None:
    plan = study.planned_fits(ATE(), random_state=0, q_bounds=BOUNDS)
    assert {name: cv for name, _, cv in plan.candidates}["lasso"] == 5
    assert plan.total_candidate_fits == 360


def test_the_study_shortcut_and_the_effect_agree(study: CausalStudy) -> None:
    settings = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "random_state": 0,
        "q_bounds": BOUNDS,
    }
    assert study.planned_fits(ATE(), **settings) == study.identify(ATE()).planned_fits(**settings)


def test_the_preview_refuses_what_estimate_refuses(study: CausalStudy) -> None:
    effect = study.identify(ATE())
    with pytest.raises(CapabilityError) as previewed:
        effect.planned_fits(random_state=0)
    with pytest.raises(CapabilityError) as estimated:
        effect.estimate(random_state=0)
    assert str(previewed.value) == str(estimated.value)


def test_planned_fits_is_exported() -> None:
    assert cleverly.PlannedFits is PlannedFits


def test_the_record_prints_and_converts(study: CausalStudy) -> None:
    plan = _route("bootstrap_no_binding_cap").plan
    text = plan.summary(detail="full")
    assert "Planned model fits" in text and "at most" in text
    frame = plan.to_frame(backend="pandas")
    assert list(frame["fits"]) == [row.fits for row in plan.rows]
    assert repr(plan).startswith("PlannedFits(")


def test_an_unpriced_engine_says_so() -> None:
    from cleverly.estimators import CTMLE

    with pytest.raises(NotImplementedError, match="phase 2"):
        CTMLE(**_plain())._planned_fits(_linear_frame(), **COLUMNS)


def test_the_preview_draws_no_shared_random_numbers() -> None:
    np.random.seed(7)  # noqa: NPY002
    before = np.random.get_state()[1].copy()  # noqa: NPY002
    TMLE(**_plain(repeats=2))._planned_fits(_linear_frame(), **COLUMNS)
    assert np.array_equal(np.random.get_state()[1], before)  # noqa: NPY002


# ----------------------------------------------------------------- mutation controls


def _skip_after_first(name: str) -> Callable[[pytest.MonkeyPatch], None]:
    """Patch a preview method to run on its first call only."""

    def mutate(patch: pytest.MonkeyPatch) -> None:
        original = getattr(TMLE, name)
        calls = {"n": 0}

        def once(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return original(self, *args, **kwargs) if calls["n"] == 1 else None

        patch.setattr(TMLE, name, once)

    return mutate


def _no_bootstrap(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(TMLE, "_plan_bootstrap", lambda *args, **kwargs: None)


def _uncapped_folds(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(planned_module, "inner_fold_count", lambda n_folds, *args: max(n_folds, 2))


def _no_constant_skip(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(planned_module, "_one_value", lambda *args: False)


def _no_tuner(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(tmle_module, "plan_tuner_fits", lambda *args, **kwargs: planned_module.Count())


def _classes_times_first(patch: pytest.MonkeyPatch) -> None:
    """``K * C * (V + 1)`` in place of the per-class sum: every member counted as the first."""
    original = planned_module._count_super_learner

    def mutated(count: Any, ensemble: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        levels = np.unique(y)
        task = ensemble.task or planned_module.__dict__.get("infer_task", lambda _: "")(y)
        if task == "classification" and levels.size > 2:
            first = planned_module.Count()
            original(
                first,
                planned_module._with_task(ensemble, "classification"),
                (y == levels[0]).astype(float),
                *args,
                **kwargs,
            )
            count.add(first, int(levels.size))
            return
        original(count, ensemble, y, *args, **kwargs)

    patch.setattr(planned_module, "_count_super_learner", mutated)


MUTATIONS: dict[str, tuple[str, Callable[[pytest.MonkeyPatch], None]]] = {
    "the (1 + B) factor": ("bootstrap_no_binding_cap", _no_bootstrap),
    "the repeat factor": ("ate_repeats", _skip_after_first("_plan_draw")),
    "the per-class sum of one-vs-rest": ("rare_arm_caps_a_member", _classes_times_first),
    "the shrunken inner folds": ("rare_class_caps_inner_folds", _uncapped_folds),
    "the constant-fold skip": ("constant_binary_arm", _no_constant_skip),
    "the per-level CDE bootstrap": ("cde_bootstrap", _skip_after_first("_plan_bootstrap")),
    "the tuner row": ("tuned_cap_learner", _no_tuner),
}


@pytest.mark.parametrize("label", sorted(MUTATIONS))
def test_each_mutation_fails_its_fixture(label: str) -> None:
    name, mutate = MUTATIONS[label]
    found = _route(name, mutate)
    with pytest.raises(AssertionError):
        support.check(found, plain=PLAIN)


def test_the_unmutated_mutation_fixtures_pass() -> None:
    for name in {name for name, _ in MUTATIONS.values()}:
        support.check(_route(name), plain=PLAIN)
