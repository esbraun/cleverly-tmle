"""The planned-fit preview counts the learner fits ``estimate`` makes, and fits nothing.

Each route fixture previews a fit, runs it with counting learners, and compares the counts
(:func:`tests.unit._planned_fits_support.check`): an exact row equals the realized count per
role and per candidate, an upper row brackets it, the class-wide total equals the labelled
total, and the preview itself fits nothing.  The mutation controls patch one term out of
the preview and show the comparison then fails.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import cleverly
from cleverly import (
    ATE,
    AdaptiveTMLEMethod,
    CausalStudy,
    ConditionalContrast,
    CrossFitting,
    ModelSpec,
    PlannedFits,
    PointTreatment,
    SuperLearner,
    TrialATE,
    TrialWithExternalData,
)
from cleverly import _planned_fits as planned_module
from cleverly.bases import Modifiers, Sieve
from cleverly.datasets import (
    make_cde,
    make_clustered,
    make_linear_ate,
    make_missing_outcome,
    make_multi_arm,
    make_shift_dose,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE, AdaptiveIPTW
from cleverly.exceptions import CapabilityError
from cleverly.interventions import (
    Incremental,
    JointPolicy,
    LearnedRule,
    Shift,
    SupportBoundary,
)
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
    engine: type[TMLE] = TMLE,
) -> support.Witness:
    """The engine's preview and its fit on the same arguments, with fresh learners each."""

    def preview() -> PlannedFits:
        if mutate is None:
            return engine(**settings())._planned_fits(frame, **columns)
        with pytest.MonkeyPatch.context() as patch:
            mutate(patch)
            return engine(**settings())._planned_fits(frame, **columns)

    return support.witness(preview, lambda: engine(**settings()).fit(frame, **columns))


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


def _vector_frame(n: int = 300) -> pd.DataFrame:
    """A binary flag and a continuous dose, jointly the treatment."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=n)
    flag = (rng.uniform(size=n) < 0.5).astype(float)
    dose = w + flag + rng.normal(size=n)
    return pd.DataFrame({"W": w, "F": flag, "D": dose, "Y": w + 0.3 * dose + rng.normal(size=n)})


VECTOR_COLUMNS = {
    "outcome": "Y",
    "treatment": ["F", "D"],
    "covariates": ["W"],
    "continuous_treatment": ["D"],
}
RULE_COLUMNS = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
RULE_MODIFIERS = Modifiers(["W1", "W2"], domains={"W1": (-1.0, 1.0)}, levels={"W2": (0.0, 1.0)})


def _dose_up() -> list[Any]:
    return [JointPolicy("dose up", components={"D": Shift(0.5, cap=None)}, policy_kind="known")]


def _ep_rule(sieve: Sieve) -> Callable[[], dict[str, Any]]:
    def settings() -> dict[str, Any]:
        rule = LearnedRule(
            loss="ep",
            modifiers=RULE_MODIFIERS,
            sieve=sieve,
            learner=support.linear("learned_rule"),
        )
        return _plain(learned_rule=rule)

    return settings


def _nested_ensemble() -> Any:
    """A Super Learner whose library holds a Super Learner of ten folds."""
    inner = SuperLearner(
        library=[
            (name, support.logistic(f"treatment_learner:inner/{name}")) for name in ("a", "b")
        ],
        n_folds=10,
    )
    outer = support.ensemble("treatment_learner", names=("c",), n_folds=3)
    outer.library = [("inner", inner), *list(outer.library or ())]
    return outer


RARE_COLUMNS = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}

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
    "vector_classifier": (
        lambda: _plain(policies=_dose_up(), ratio="classifier"),
        _vector_frame,
        VECTOR_COLUMNS,
    ),
    "vector_density": (
        lambda: _plain(policies=_dose_up(), density_bins=4),
        _vector_frame,
        VECTOR_COLUMNS,
    ),
    "learned_rule_ep_fixed": (
        _ep_rule(Sieve.default(RULE_MODIFIERS, max_size=3, size=2)),
        lambda: _rule_frame(300),
        RULE_COLUMNS,
    ),
    "learned_rule_ep_cv": (
        _ep_rule(Sieve.default(RULE_MODIFIERS, max_size=2)),
        lambda: _rule_frame(300),
        RULE_COLUMNS,
    ),
    "nested_super_learner": (
        lambda: _plain(treatment_learner=_nested_ensemble()),
        lambda: support.rare_class_frame(n=240, rare=14),
        RARE_COLUMNS,
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


def test_a_nested_super_learner_caps_its_own_folds() -> None:
    plan = _route("nested_super_learner").plan
    (row,) = [row for row in plan.rows if row.role == "treatment_learner"]
    names = dict(row.by_candidate)
    # Uncapped, the inner Super Learner fits each candidate (10 + 1) times per outer fit.
    assert names["inner/a"] < row.fits * (3 + 1) * (10 + 1)


# ----------------------------------------------------------------- the variant engines

ATE_ONLY = ("ate",)
SELECTION = {"estimands": ATE_ONLY, "selection_folds": 3, "selection_inner_folds": 2}
THREE_SETS = [["W1"], ["W1", "W2"], ["W1", "W2", "W3", "W4"]]
REDUCED = {
    **PLAIN,
    "reduced_outcome_learner": "LinearRegression",
    "reduced_treatment_learner": "LogisticRegression",
    "reduced_mechanism_residual_learner": "LinearRegression",
    "preorder": "LogisticRegression",
}


def _ctmle(**overrides: Any) -> Callable[[], dict[str, Any]]:
    selector = overrides.get("strategy", "greedy") not in ("oat", "derivative")
    base = SELECTION if selector else {"estimands": ATE_ONLY}
    return lambda: _plain(**{**base, **overrides})


def _drtmle(**overrides: Any) -> Callable[[], dict[str, Any]]:
    return lambda: _plain(estimands=ATE_ONLY, **overrides)


def _iptw(**overrides: Any) -> Callable[[], dict[str, Any]]:
    def settings() -> dict[str, Any]:
        values = _plain(**{"estimands": ATE_ONLY, **overrides})
        for name in ("q_bounds", "simultaneous", "intermediate_learner"):
            values.pop(name, None)
        return values

    return settings


def _strata_frame() -> pd.DataFrame:
    return _linear_frame().assign(S=lambda f: (f["W2"] > 0).astype(int))


def _missing_frame() -> pd.DataFrame:
    return make_missing_outcome(n=240, seed=4)[0]


MISSING = {"outcome": "Y", "treatment": "A", "delta": "Delta"}
STRATA = {**COLUMNS, "covariates": [*COVARIATES, "S"], "strata": ["S"]}

#: name -> (engine, settings factory, frame factory, columns, check keywords)
VARIANT_ROUTES: dict[
    str,
    tuple[
        type[TMLE],
        Callable[[], dict[str, Any]],
        Callable[[], pd.DataFrame],
        dict[str, Any],
        dict[str, Any],
    ],
] = {
    "ctmle_greedy": (CTMLE, _ctmle(), _linear_frame, COLUMNS, {}),
    "ctmle_greedy_patience": (CTMLE, _ctmle(patience=1), _linear_frame, COLUMNS, {}),
    "ctmle_greedy_patience_at_length": (CTMLE, _ctmle(patience=5), _linear_frame, COLUMNS, {}),
    "ctmle_greedy_by_arm": (CTMLE, _ctmle(outcome_by_arm=True), _linear_frame, COLUMNS, {}),
    "ctmle_greedy_missing": (
        CTMLE,
        _ctmle(cross_fit=False),
        _missing_frame,
        MISSING,
        {},
    ),
    "ctmle_ordered": (
        CTMLE,
        _ctmle(strategy="ordered", preorder="partial_correlation"),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_ordered_patience": (
        CTMLE,
        _ctmle(strategy="ordered", preorder="partial_correlation", patience=1),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_ordered_logistic": (
        CTMLE,
        _ctmle(strategy="ordered"),
        _linear_frame,
        COLUMNS,
        {"engine_role": "preorder"},
    ),
    "ctmle_discrete": (
        CTMLE,
        _ctmle(strategy="discrete", candidates=THREE_SETS),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_discrete_patience": (
        CTMLE,
        _ctmle(strategy="discrete", candidates=THREE_SETS, patience=1),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_lasso": (CTMLE, _ctmle(strategy="lasso"), _linear_frame, COLUMNS, {}),
    "ctmle_learners": (
        CTMLE,
        _ctmle(
            strategy="learners",
            treatment_candidates=[support.logistic("treatment_learner", C=0.1)],
        ),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_learners_patience": (
        CTMLE,
        _ctmle(
            strategy="learners",
            treatment_candidates=[support.logistic("treatment_learner", C=0.1)],
            patience=1,
        ),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_derivative": (
        CTMLE,
        _ctmle(
            strategy="derivative",
            treatment_candidates=[
                support.logistic("treatment_learner"),
                support.logistic("treatment_learner", C=0.1),
            ],
        ),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_derivative_lasso": (
        CTMLE,
        _ctmle(strategy="derivative"),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_oat_per_arm": (CTMLE, _ctmle(strategy="oat"), _linear_frame, COLUMNS, {}),
    "ctmle_oat_shared": (
        CTMLE,
        _ctmle(strategy="oat", oat_design="shared"),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_oat_in_sample": (
        CTMLE,
        _ctmle(strategy="oat", cross_fit=False),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_oat_by_arm": (
        CTMLE,
        _ctmle(strategy="oat", outcome_by_arm=True),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_oat_missing": (CTMLE, _ctmle(strategy="oat"), _missing_frame, MISSING, {}),
    "ctmle_oat_strata": (CTMLE, _ctmle(strategy="oat"), _strata_frame, STRATA, {}),
    "ctmle_selection_folds_cap": (
        CTMLE,
        _ctmle(treatment_learner=support.ensemble("treatment_learner", n_folds=10)),
        lambda: support.rare_class_frame(n=240, rare=14),
        RARE_COLUMNS,
        {},
    ),
    "ctmle_unseeded": (
        CTMLE,
        _ctmle(
            random_state=None,
            treatment_learner=support.ensemble("treatment_learner", n_folds=3),
        ),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "ctmle_bootstrap": (CTMLE, _ctmle(n_bootstrap=2), _linear_frame, COLUMNS, {}),
    "drtmle_foldwise": (DRTMLE, _drtmle(), _linear_frame, COLUMNS, {}),
    "drtmle_pooled": (DRTMLE, _drtmle(reduced_crossfit="pooled"), _linear_frame, COLUMNS, {}),
    "drtmle_cap_1": (DRTMLE, _drtmle(max_outer=1), _linear_frame, COLUMNS, {}),
    "drtmle_cap_2": (DRTMLE, _drtmle(max_outer=2), _linear_frame, COLUMNS, {}),
    "drtmle_cap_3": (DRTMLE, _drtmle(max_outer=3), _linear_frame, COLUMNS, {}),
    "drtmle_reduced_learners": (
        DRTMLE,
        _drtmle(
            max_outer=1,
            reduced_outcome_learner=support.linear("reduced_outcome_learner"),
            reduced_treatment_learner=support.logistic("reduced_treatment_learner"),
            reduced_mechanism_residual_learner=support.linear("reduced_mechanism_residual_learner"),
        ),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "drtmle_bivariate_g": (
        DRTMLE,
        _drtmle(reduction="bivariate", guard=("g",)),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "drtmle_no_guard": (DRTMLE, _drtmle(guard=()), _linear_frame, COLUMNS, {}),
    "drtmle_in_sample": (DRTMLE, _drtmle(cross_fit=False), _linear_frame, COLUMNS, {}),
    "drtmle_composite": (DRTMLE, _drtmle(cross_fit=False), _missing_frame, MISSING, {}),
    "drtmle_randomized_missing": (
        DRTMLE,
        _drtmle(cross_fit=False, randomized=True),
        _missing_frame,
        MISSING,
        {},
    ),
    "drtmle_strata": (DRTMLE, _drtmle(), _strata_frame, STRATA, {}),
    "drtmle_by_arm": (DRTMLE, _drtmle(outcome_by_arm=True), _linear_frame, COLUMNS, {}),
    "drtmle_bootstrap": (
        DRTMLE,
        _drtmle(n_bootstrap=2, max_outer=2),
        _linear_frame,
        COLUMNS,
        {},
    ),
    "iptw_cross_fit": (AdaptiveIPTW, _iptw(), _linear_frame, COLUMNS, {}),
    "iptw_at_max_iter": (AdaptiveIPTW, _iptw(tol=0.0, max_iter=2), _linear_frame, COLUMNS, {}),
    "iptw_in_sample": (AdaptiveIPTW, _iptw(cross_fit=False), _linear_frame, COLUMNS, {}),
    "iptw_reduced_learner": (
        AdaptiveIPTW,
        lambda: {
            **_iptw()(),
            "outcome_learner": None,
            "reduced_outcome_learner": support.linear("reduced_outcome_learner"),
        },
        _linear_frame,
        COLUMNS,
        {},
    ),
    "iptw_missing_outcome": (
        AdaptiveIPTW,
        _iptw(estimands=("ate", "ey_obs")),
        _missing_frame,
        MISSING,
        {},
    ),
    "iptw_strata": (AdaptiveIPTW, _iptw(), _strata_frame, STRATA, {}),
    "iptw_binary_outcome": (AdaptiveIPTW, _iptw(), _constant_arm_frame, COLUMNS, {}),
    "iptw_known_mechanism": (
        AdaptiveIPTW,
        _iptw(),
        lambda: _known_weighted()[0],
        {**_known_weighted()[1], "mechanism_law": "target"},
        {},
    ),
}


def _variant(
    name: str, mutate: Callable[[pytest.MonkeyPatch], None] | None = None
) -> support.Witness:
    engine, settings, frame, columns, _ = VARIANT_ROUTES[name]
    return _engine(settings, frame(), columns, mutate, engine)


@pytest.mark.parametrize("name", sorted(VARIANT_ROUTES))
def test_the_preview_counts_the_fits_of_each_variant_route(name: str) -> None:
    check = VARIANT_ROUTES[name][4]
    support.check(_variant(name), plain=REDUCED, **check)


@pytest.mark.parametrize("name", sorted(VARIANT_ROUTES))
def test_the_variant_preview_fits_nothing(name: str) -> None:
    engine, settings, frame, columns, _ = VARIANT_ROUTES[name]
    data = frame()

    def never() -> dict[str, Any]:
        values = settings()
        for role in (*ROLES, "reduced_outcome_learner", "reduced_treatment_learner"):
            if values.get(role) is not None:
                values[role] = NeverFit()
        if values.get("treatment_candidates"):
            values["treatment_candidates"] = [NeverFit() for _ in values["treatment_candidates"]]
        return values

    NeverFit.calls = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = engine(**never())._planned_fits(data, **columns)
    assert NeverFit.calls == 0
    assert plan.total_fits > 0


def _reached(found: support.Witness) -> None:
    """Every row's upper bound equals the fit's count: the forcing fixture reached it."""
    for role, fits in support.totals(found.plan).items():
        assert support.realized_fits(found.counts, role) == fits, role


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_a_small_reduction_cap_is_reached(cap: int) -> None:
    found = _variant(f"drtmle_cap_{cap}")
    assert not found.plan.exact
    for repeat in found.result.single().repeats:
        assert repeat.fluctuations["mean"].reduction.exit_reason == "cap"
    _reached(found)
    assert {name: steps for name, steps, _, _ in found.plan.searches} == {
        "DR-TMLE reduction rounds": cap
    }


def test_the_reduced_learners_fall_back_to_the_primary_fields() -> None:
    roles = {row.role for row in _variant("drtmle_cap_1").plan.rows}
    assert roles == {"treatment_learner", "outcome_learner"}
    roles = {row.role for row in _variant("drtmle_reduced_learners").plan.rows}
    assert {
        "reduced_outcome_learner",
        "reduced_treatment_learner",
        "reduced_mechanism_residual_learner",
    } <= roles


def test_the_adaptive_iptw_update_cap_is_reached() -> None:
    found = _variant("iptw_at_max_iter")
    assert found.result.single().repeats[0].fluctuations["mean"].failure == "max_iter_reached"
    _reached(found)


def test_a_patience_at_the_path_length_reaches_the_bound() -> None:
    found = _variant("ctmle_greedy_patience_at_length")
    assert not found.plan.exact
    _reached(found)
    assert _variant("ctmle_greedy").plan.exact


def test_a_patience_stop_is_bracketed() -> None:
    found = _variant("ctmle_greedy_patience")
    rows = [row for row in found.plan.rows if row.role == "treatment_learner"]
    assert all(row.bound == "upper" for row in rows)
    assert support.realized_fits(found.counts, "treatment_learner") < sum(r.fits for r in rows)


def test_a_greedy_path_fits_p_times_p_plus_one_over_two_sets_per_branch() -> None:
    plan = _variant("ctmle_greedy").plan
    full = [row for row in plan.rows if "analysis sample" in row.scope]
    folds = [
        row for row in plan.rows if "per selection fold" in row.scope and "propens" in row.scope
    ]
    # Four covariates: 10 sets, cross-fitted over 3 outer folds, and over 2 + 1 nested fits
    # in each of 3 selection folds.
    assert sum(row.fits for row in full) == 10 * 3
    assert sum(row.fits for row in folds) == 10 * 3 * 3


def test_the_selection_folds_cap_the_inner_folds() -> None:
    found = _variant("ctmle_selection_folds_cap")
    rows = [row for row in found.plan.rows if row.role == "treatment_learner"]
    fits = sum(row.fits for row in rows)
    candidates = sum(row.candidate_fits for row in rows)
    assert candidates < fits * 2 * (10 + 1)


# ----------------------------------------------------------------- the study routes


def _ep_frame() -> pd.DataFrame:
    return _rule_frame(300)


def _ep_effect(frame: pd.DataFrame, domain: tuple[float, float] = (-1.0, 1.0)) -> Any:
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    return study.identify(ConditionalContrast(Modifiers(["W1"], domains={"W1": domain})))


def _ep(**overrides: Any) -> Callable[[], dict[str, Any]]:
    def settings() -> dict[str, Any]:
        return {
            "outcome_learner": support.linear("outcome_learner"),
            "treatment_learner": support.logistic("treatment_learner"),
            "n_folds": 3,
            "random_state": 0,
            "q_bounds": BOUNDS,
            **overrides,
        }

    return settings


EP_MODIFIERS = Modifiers(["W1"], domains={"W1": (-1.0, 1.0)})


def _trial_frame(n: int = 240, *, external_treated: bool = True, missing: bool = False) -> Any:
    rng = np.random.default_rng(0)
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    a = rng.integers(0, 2, n)
    if not external_treated:
        a = np.where(s == 0, 0, a)
    y = a + w + 0.5 * (1 - s) + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W": w, "S": s, "R": rng.binomial(1, 0.9, n)})
    if missing:
        frame.loc[frame["R"] == 0, "Y"] = np.nan
    return frame


def _trial_effect(frame: pd.DataFrame, missing: bool = False) -> Any:
    design = TrialWithExternalData(
        outcome="Y",
        treatment="A",
        adjustment=("W",),
        source="S",
        missingness="R" if missing else None,
    )
    return CausalStudy(frame, design=design).identify(TrialATE("trial"))


def _atmle(
    form: str = "cv", *, cross_fit: bool = True, random_state: int | None = 0
) -> Callable[[], dict[str, Any]]:
    def settings() -> dict[str, Any]:
        method = AdaptiveTMLEMethod(
            form=form,  # type: ignore[arg-type]
            models=ModelSpec(
                outcome_learner=support.linear("outcome_learner"),
                treatment_learner=support.logistic("treatment_learner"),
                missingness_learner=support.logistic("missingness_learner"),
            ),
            source_learner=support.logistic("source_learner"),
            cross_fitting=CrossFitting(n_folds=3, learner_folds=3, enabled=cross_fit),
        ).with_overrides(random_state=random_state)
        return {"method": method}

    return settings


STUDY_PLAIN = {**PLAIN, "final_learner": "LinearRegression", "source_learner": "LogisticRegression"}

#: name -> (effect factory, method name or None, settings factory)
STUDY_ROUTES: dict[str, tuple[Callable[[], Any], str | None, Callable[[], dict[str, Any]]]] = {
    "ep_fixed": (
        lambda: _ep_effect(_ep_frame()),
        "ep",
        _ep(
            sieve=Sieve.default(EP_MODIFIERS, max_size=3, size=2),
            final_learner=support.linear("final_learner"),
        ),
    ),
    "ep_cv": (
        lambda: _ep_effect(_ep_frame()),
        "ep",
        _ep(
            sieve=Sieve.default(EP_MODIFIERS, max_size=3),
            final_learner=support.linear("final_learner"),
        ),
    ),
    "ep_cv_least_squares": (
        lambda: _ep_effect(_ep_frame()),
        "ep",
        _ep(sieve=Sieve.default(EP_MODIFIERS, max_size=2)),
    ),
    "ep_cv_inner_cap": (
        lambda: _ep_effect(support.rare_class_frame(n=240, rare=14), (-6.0, 6.0)),
        "ep",
        _ep(
            sieve=Sieve.default(Modifiers(["W1"], domains={"W1": (-6.0, 6.0)}), max_size=2),
            treatment_learner=support.ensemble("treatment_learner", n_folds=10),
        ),
    ),
    "atmle_cv": (lambda: _trial_effect(_trial_frame()), None, _atmle()),
    "atmle_pooled": (lambda: _trial_effect(_trial_frame()), None, _atmle("pooled")),
    "atmle_pooled_in_sample": (
        lambda: _trial_effect(_trial_frame()),
        None,
        _atmle("pooled", cross_fit=False),
    ),
    "atmle_controls_only": (
        lambda: _trial_effect(_trial_frame(external_treated=False)),
        None,
        _atmle(),
    ),
    "atmle_missing_outcome": (
        lambda: _trial_effect(_trial_frame(missing=True), missing=True),
        None,
        _atmle(),
    ),
    "atmle_unseeded": (lambda: _trial_effect(_trial_frame()), None, _atmle(random_state=None)),
}


def _study_route(
    name: str, mutate: Callable[[pytest.MonkeyPatch], None] | None = None
) -> support.Witness:
    effect_factory, method, settings = STUDY_ROUTES[name]
    effect = effect_factory()

    def call(action: Callable[..., Any]) -> Any:
        values = settings()
        named = values.pop("method", method)
        return action(named, **values)

    def preview() -> PlannedFits:
        if mutate is None:
            return cast("PlannedFits", call(effect.planned_fits))
        with pytest.MonkeyPatch.context() as patch:
            mutate(patch)
            return cast("PlannedFits", call(effect.planned_fits))

    return support.witness(preview, lambda: call(effect.estimate))


@pytest.mark.parametrize("name", sorted(STUDY_ROUTES))
def test_the_preview_counts_the_fits_of_each_study_route(name: str) -> None:
    found = _study_route(name)
    assert found.plan.exact
    support.check(found, plain=STUDY_PLAIN)


def test_the_ep_inner_cap_binds() -> None:
    plan = _study_route("ep_cv_inner_cap").plan
    rows = [row for row in plan.rows if row.role == "treatment_learner" and "inner" in row.scope]
    fits = sum(row.fits for row in rows)
    assert sum(row.candidate_fits for row in rows) < fits * 2 * (10 + 1)


def test_the_ep_final_learner_fits_once_per_size_and_fold_then_once() -> None:
    plan = _study_route("ep_cv").plan
    final = sum(row.fits for row in plan.rows if row.role == "final_learner")
    assert final == 3 * 3 + 1
    assert {name: steps for name, steps, _, _ in plan.searches} == {"EP sieve sizes": 3}


def test_the_atmle_lambda_path_fits_no_learner() -> None:
    plan = _study_route("atmle_cv").plan
    ((name, _, early, fits),) = plan.searches
    assert "penalty path" in name and not early and fits == 0
    # Three outer folds and three pairs, five fits per call.
    assert plan.total_fits == (3 + 3) * 5


def test_the_study_previews_fit_nothing() -> None:
    effect = _ep_effect(_ep_frame())
    NeverFit.calls = 0
    plan = effect.planned_fits(
        "ep",
        outcome_learner=NeverFit(),
        treatment_learner=NeverFit(),
        final_learner=NeverFit(),
        sieve=Sieve.default(EP_MODIFIERS, max_size=2),
        n_folds=3,
        random_state=0,
        q_bounds=BOUNDS,
    )
    assert NeverFit.calls == 0 and plan.total_fits > 0
    method = AdaptiveTMLEMethod(
        models=ModelSpec(outcome_learner=NeverFit(), treatment_learner=NeverFit()),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
    ).with_overrides(random_state=0)
    plan = _trial_effect(_trial_frame()).planned_fits(method=method)
    assert NeverFit.calls == 0 and plan.total_fits > 0


def _no_final_learner(patch: pytest.MonkeyPatch) -> None:
    """Preview as if no final learner were declared: the EP final-learner term drops out."""
    from cleverly.estimators import ep as ep_module

    original = ep_module.EPEstimator._planned_fits

    def without(self: Any, data: Any) -> Any:
        copied = copy.copy(self)
        copied.final_learner = None
        return original(copied, data)

    patch.setattr(ep_module.EPEstimator, "_planned_fits", without)


def _inner_on_outer_rows(patch: pytest.MonkeyPatch) -> None:
    """Preview each fold's inner fit on every row instead of the fold's training rows."""
    from cleverly.data.causal_data import CausalData

    patch.setattr(CausalData, "subset", lambda self, rows: self)


STUDY_MUTATIONS: dict[str, tuple[str, Callable[[pytest.MonkeyPatch], None]]] = {
    "the EP final-learner term": ("ep_cv", _no_final_learner),
    "the inner counts on the fold's own rows": ("ep_cv_inner_cap", _inner_on_outer_rows),
}


@pytest.mark.parametrize("label", sorted(STUDY_MUTATIONS))
def test_each_study_mutation_fails_its_fixture(label: str) -> None:
    name, mutate = STUDY_MUTATIONS[label]
    found = _study_route(name, mutate)
    with pytest.raises(AssertionError):
        support.check(found, plain=STUDY_PLAIN)


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
