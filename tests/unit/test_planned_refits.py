"""The refit preview of a refutation (X106 part G2, comparison plan "Witnesses").

- W1 counting: for each adapter and test, the fits ``refute()`` makes equal ``learner_fits`` on
  an exact row and lie in ``[min_fits, learner_fits]`` on an upper row.
- W2 fits nothing: the preview records no fit and leaves the result as it was.
- W3 refusal parity: a request ``refute()`` refuses before its first refit raises the same
  record through ``planned_refits``.
- W4 seed spread, W6 no cache crossing, W7 variable importance, W8 shared preflight, W9 the
  replay copy and the longitudinal plasmode preparation.

W5 (the structural preview) lives in ``test_planned_fits_refit.py`` with the other engines.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_longitudinal
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.structural import StructuralNestedMean
from cleverly.survival import CoxHazard
from cleverly.survival.engine import ContinuousSurvivalTMLE
from cleverly.validation import PlannedRefits, planned_refits, refute
from cleverly.validation import refute as refute_function
from tests.unit import _planned_fits_support as support

refute_module: Any = __import__("cleverly.validation.refute", fromlist=["refute"])

R = 3


def _quiet(call: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _point_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-w[:, 0])))
    y = 0.5 * w[:, 0] - 0.3 * w[:, 1] + 0.5 * a + rng.normal(size=n)
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": a, "Y": y})


@functools.cache
def _point() -> Any:
    """A study fit, so the generated-outcome tests have the identification they read."""
    study = CausalStudy(
        _point_frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    return _quiet(
        lambda: study.identify(ATE()).estimate(
            outcome_learner=support.linear("Q"),
            treatment_learner=support.logistic("g"),
            n_folds=3,
            random_state=0,
            q_bounds=(-30.0, 30.0),
        )
    )


@functools.cache
def _in_sample() -> Any:
    """An in-sample study fit with no outcome bounds, which the generated-outcome tests take."""
    study = CausalStudy(
        _point_frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    return _quiet(
        lambda: study.identify(ATE()).estimate(
            outcome_learner=support.linear("Q"),
            treatment_learner=support.logistic("g"),
            cross_fit=False,
            random_state=0,
        )
    )


@functools.cache
def _rare() -> Any:
    """A treatment arm of 4 units in 240, so a resampled arm can leave a fold with one
    class, which the fit models as a constant."""
    frame = support.rare_class_frame(n=240, rare=4, seed=1)
    estimator = TMLE(
        outcome_learner=support.linear("Q"),
        treatment_learner=support.logistic("g"),
        n_folds=5,
        random_state=0,
        q_bounds=(-30.0, 30.0),
    )
    covariates = [name for name in frame.columns if name not in ("Y", "A")]
    return _quiet(
        lambda: estimator.fit(frame, outcome="Y", treatment="A", covariates=covariates).single()
    )


@functools.cache
def _longitudinal() -> Any:
    frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    estimator = LTMLE(
        {"always": 1, "never": 0},
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=0,
    )
    return _quiet(
        lambda: estimator.fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
        )
    )


@functools.cache
def _gaussian_panel() -> Any:
    """A censored panel with a Gaussian outcome, which the longitudinal plasmode takes."""
    from tests.unit.test_longitudinal_plasmode import _binary_panel, _ltmle

    return _quiet(
        lambda: _ltmle({"always": 1, "never": 0}).fit(
            _binary_panel(n=300),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    )


@functools.cache
def _structural() -> Any:
    frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    estimator = StructuralNestedMean(
        {"always": 1, "never": 0},
        outcome_learner=GLM(family="binomial"),
        treatment_learner=GLM(family="binomial"),
        n_folds=2,
        random_state=0,
        simultaneous=False,
    )
    return _quiet(
        lambda: estimator.fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
        )
    )


@functools.cache
def _survival() -> Any:
    """A continuous-time fit with a bootstrap, which its refits drop."""
    from tests.unit.test_continuous_survival_refute import _frame

    estimator = ContinuousSurvivalTMLE(
        [1.0, 2.0],
        cross_fit=True,
        n_folds=2,
        hazard_library=[CoxHazard("main")],
        censoring_library=[CoxHazard("main")],
        treatment_learner=support.logistic("g"),
        n_bootstrap=2,
        random_state=0,
    )
    return _quiet(
        lambda: estimator.fit(
            _frame(n=300), time="time", event="event", treatment="A", baseline=["W"]
        )
    )


def _realized(call: Callable[[], Any]) -> int:
    """Every learner fit ``call`` makes, counted class-wide on the final estimators."""
    support.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with support.class_wide():
            call()
    return sum(support.CLASS_WIDE.values())


def _preview(result: Any, **kwargs: Any) -> PlannedRefits:
    """The preview, checked to fit nothing (W2)."""
    support.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with support.class_wide():
            plan = planned_refits(result, **kwargs)
    assert sum(support.CLASS_WIDE.values()) == 0, dict(support.CLASS_WIDE)
    return plan


def _within(plan: PlannedRefits, realized: int) -> None:
    (test,) = plan.tests
    if test.bound == "exact":
        assert realized == test.learner_fits, (test, realized)
    else:
        assert test.min_fits <= realized <= test.learner_fits, (test, realized)


ADAPTERS: dict[str, tuple[Callable[[], Any], tuple[str, ...], dict[str, Any]]] = {
    "point": (
        _point,
        (
            "placebo",
            "random_common_cause",
            "subset",
            "negative_control_outcome",
            "bootstrap_stability",
        ),
        {},
    ),
    "point in sample": (_in_sample, ("dummy_outcome", "simulated_outcome", "plasmode"), {}),
    "longitudinal": (
        _longitudinal,
        ("placebo", "random_common_cause", "subset", "negative_control_outcome"),
        {},
    ),
    "longitudinal plasmode": (_gaussian_panel, ("plasmode",), {}),
    "structural": (
        _structural,
        ("placebo", "random_common_cause", "subset", "negative_control_outcome"),
        {"estimand": "blip[A1:1.0:intercept]"},
    ),
    "survival": (_survival, ("placebo", "random_common_cause", "subset"), {}),
}


def _contrast(result: Any) -> str:
    """The first contrast the fit reports, else its first estimand."""
    names = list(result)
    return next((name for name in names if name.startswith("ate")), names[0])


def _arguments(result: Any, name: str, base: dict[str, Any]) -> dict[str, Any]:
    draws = 10 if name in ("dummy_outcome", "simulated_outcome", "plasmode") else R
    arguments = {"estimand": _contrast(result), **base, "tests": [name], "n_replicates": draws}
    if name == "negative_control_outcome":
        rng = np.random.default_rng(5)
        arguments["negative_control_outcome"] = rng.integers(0, 2, int(result.data.n)).astype(float)
    return arguments


CASES = [(adapter, name) for adapter, (_, names, _) in ADAPTERS.items() for name in names]


SPREAD = [(adapter, name) for adapter, name in CASES if name == "random_common_cause"]


@pytest.mark.parametrize(
    ("adapter", "name", "spread"),
    [(*case, False) for case in CASES] + [(*case, True) for case in SPREAD],
)
def test_w1_the_realized_fits_lie_in_the_preview(adapter: str, name: str, spread: bool) -> None:
    build, _, base = ADAPTERS[adapter]
    result = build()
    arguments = _arguments(result, name, base)
    arguments["seed_spread"] = spread
    plan = _preview(result, **arguments)
    _within(plan, _realized(lambda: refute(result, **arguments)))


def test_w1_a_constant_fold_makes_the_nominal_rows_upper() -> None:
    """A resample of a four-unit arm can hold too few treated units for a fold, which the fit
    models as a constant, so the drawn-data row counts more fits than run: an exact label
    would fail."""
    result = _rare()
    arguments: dict[str, Any] = {
        "tests": ["bootstrap_stability"],
        "n_replicates": 10,
        "random_state": 0,
    }
    plan = _preview(result, **arguments)
    realized = _realized(lambda: refute(result, **arguments))
    _within(plan, realized)
    assert plan.tests[0].bound == "upper"
    assert realized < plan.tests[0].learner_fits


def test_w1_a_survival_unit_drops_the_bootstrap() -> None:
    """The noise refits read the fit's seed, so their row is exact; a unit that kept the
    fit's two bootstrap replicates would count three times the fits."""
    result = _survival()
    arguments: dict[str, Any] = {
        "estimand": _contrast(result),
        "tests": ["random_common_cause"],
        "n_replicates": R,
    }
    plan = _preview(result, **arguments)
    assert plan.tests[0].bound == "exact"
    assert _realized(lambda: refute(result, **arguments)) == plan.tests[0].learner_fits


def test_w2_the_preview_leaves_the_result_as_it_was() -> None:
    result = _point()
    estimator = result.estimator
    before = dict(vars(estimator))
    cache = dict(result.assessment_cache)
    _preview(result, tests=["placebo", "random_common_cause", "subset"], n_replicates=R)
    assert result.estimator is estimator
    assert vars(estimator).keys() == before.keys()
    assert all(vars(estimator)[key] is before[key] for key in before)
    assert dict(result.assessment_cache) == cache


# ----------------------------------------------------------------- W3 refusal parity


class _Stop(Exception):
    """Raised where ``refute`` would make its first refit."""


def _refute_refusal(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], Any]
) -> tuple[Any, ...] | None:
    """The refusal ``call`` (a refute call) raises before its first refit, or ``None``."""
    original = refute_module._refute_preflight

    def stop(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise _Stop

    with monkeypatch.context() as patch:
        patch.setattr(refute_module, "_refute_preflight", stop)
        return _raised(call, stop_ok=True)


def _raised(call: Callable[[], Any], *, stop_ok: bool = False) -> tuple[Any, ...] | None:
    try:
        _quiet(call)
    except _Stop:
        assert stop_ok
        return None
    except (CapabilityError, ValueError, TypeError) as error:
        refusal = getattr(error, "refusal", None)
        return (
            type(error),
            getattr(refusal, "reason", None),
            getattr(refusal, "condition", None) or str(error),
        )
    return None


REFUSED: dict[str, tuple[Callable[[], Any], dict[str, Any]]] = {
    "unknown test": (_point, {"tests": ["placeboo"]}),
    "a budget below ten": (_point, {"tests": ["dummy_outcome"], "n_replicates": 5}),
    "a missing control": (_point, {"tests": ["negative_control_outcome"]}),
    "a survival control": (
        _survival,
        {"tests": ["negative_control_outcome"], "negative_control_outcome": np.zeros(300)},
    ),
    "a structural plasmode": (
        _structural,
        {"tests": ["plasmode"], "estimand": "blip[A1:1.0:intercept]"},
    ),
    "a survival generated outcome": (_survival, {"tests": ["dummy_outcome"]}),
    "a longitudinal future treatment without a control": (
        _longitudinal,
        {"tests": ["future_treatment"]},
    ),
    "an unreported estimand": (_point, {"estimand": "att"}),
    "an unregistered measurement error": (_point, {"tests": ["bootstrap_measurement_error"]}),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_w3_a_refused_request_raises_the_same_record(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build, arguments = REFUSED[case]
    result = build()
    expected = _refute_refusal(monkeypatch, lambda: refute(result, **arguments))
    assert expected is not None, case
    assert _raised(lambda: planned_refits(result, **arguments)) == expected


def _kinds() -> list[str]:
    from tests.unit._capability_sweep_support import KINDS

    return list(KINDS)


@pytest.mark.parametrize("kind", _kinds())
def test_w3_the_capability_kinds_refuse_alike(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every capability-sweep kind: the facade's preview and its refute raise the same record
    before any refit, for the default request and for each test alone."""
    from tests.unit._capability_sweep_support import KINDS

    result = KINDS[kind].build()
    facade = result.diagnostics
    names = ("placebo", "random_common_cause", "subset", "plasmode", "bootstrap_stability")
    for tests in (None, *[[name] for name in names]):
        arguments: dict[str, Any] = {"n_replicates": 10}
        if tests is not None:
            arguments["tests"] = tests
        expected = _refute_refusal(monkeypatch, lambda a=arguments: facade.refute(**a))
        got = _raised(lambda a=arguments: facade.planned_refits(**a))
        assert got == expected, (tests, got, expected)


# ----------------------------------------------------------------- W4 seed spread


def test_w4_seed_spread_adds_r_refits_to_the_noise_test_only() -> None:
    result = _point()
    tests = ["placebo", "random_common_cause", "subset"]
    off = _preview(result, tests=tests, n_replicates=R, seed_spread=False)
    on = _preview(result, tests=tests, n_replicates=R, seed_spread=True)
    for first, second in zip(off.tests, on.tests, strict=True):
        if first.name == "random_common_cause":
            assert second.refits == first.refits + R
            assert second.refits_per_draw == 2
        else:
            assert second == first


# ----------------------------------------------------------------- W6, W7 facades


def test_w6_the_preview_and_refute_never_cross_in_the_cache() -> None:
    result = dataclasses.replace(_point())
    facade = result.diagnostics
    arguments: dict[str, Any] = {
        "tests": ["random_common_cause"],
        "n_replicates": R,
        "random_state": 0,
    }
    before = len(result.assessment_cache)
    assert isinstance(facade.planned_refits(**arguments), PlannedRefits)
    assert len(result.assessment_cache) == before
    report = _quiet(lambda: facade.refute(**arguments))
    assert type(report).__name__ == "RefutationResult"
    after = len(result.assessment_cache)
    assert isinstance(facade.planned_refits(**arguments), PlannedRefits)
    assert len(result.assessment_cache) == after


def test_w7_the_variable_importance_frame_totals_its_candidates() -> None:
    from tests.unit._capability_sweep_support import KINDS

    result = KINDS["variable_importance"].build()
    reports = result.diagnostics.planned_refits(n_replicates=R, random_state=0)
    assert list(reports) == list(result.fits)
    frame = reports.to_frame()
    assert set(frame["candidate"]) == set(reports)
    assert int(frame["learner_fits"].sum()) == sum(
        plan.total_learner_fits for plan in reports.values()
    )


# ----------------------------------------------------------------- W8 shared preflight


def test_w8_one_check_in_the_shared_preflight_reaches_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _point()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise CapabilityError("a planted check", reason=RefusalReason.INVALID_DECLARATION)

    monkeypatch.setattr(refute_module, "refuse_augmented_trial", refuse)
    for call in (planned_refits, refute_function):
        with pytest.raises(CapabilityError, match="a planted check"):
            call(result, tests=["random_common_cause"], n_replicates=R)


# ----------------------------------------------------------------- W9 replay and plasmode


def _count_calls(
    monkeypatch: pytest.MonkeyPatch, owner: type, name: str, seen: Counter[str]
) -> None:
    original = getattr(owner, name).__func__

    def counted(cls: Any, *args: Any, **kwargs: Any) -> Any:
        seen[f"{owner.__name__}.{name}"] += 1
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(owner, name, classmethod(counted))


def test_w9_the_longitudinal_plasmode_preview_builds_no_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cleverly.validation import _longitudinal_plasmode as plasmode

    result = _gaussian_panel()
    seen: Counter[str] = Counter()
    _count_calls(monkeypatch, plasmode.LongitudinalPlasmodeLaw, "fit", seen)
    _count_calls(monkeypatch, plasmode._Logistic, "fit", seen)
    _count_calls(monkeypatch, plasmode._Linear, "fit", seen)
    plan = _preview(result, estimand=_contrast(result), tests=["plasmode"], n_replicates=10)
    assert sum(seen.values()) == 0, dict(seen)
    (test,) = plan.tests
    _quiet(lambda: plasmode.LongitudinalPlasmodeLaw.fit(result))
    assert seen["LongitudinalPlasmodeLaw.fit"] == 1
    assert test.preparation == seen["_Logistic.fit"] + seen["_Linear.fit"] > 0


def test_w9_a_held_rule_is_priced_on_its_replay_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The noise refits of a learned rule replay the held rule, which fits no rule learner;
    pricing ``result.estimator`` itself counts the rule's learners and misses the row."""
    from sklearn.linear_model import LinearRegression

    from cleverly.interventions import LearnedRule, learned
    from tests.unit import _learned_rule_support as rules

    result = _quiet(
        lambda: rules.fit(learned_rule=LearnedRule(loss="blip", learner=LinearRegression()))
    )
    arguments: dict[str, Any] = {
        "estimand": _contrast(result),
        "tests": ["random_common_cause"],
        "n_replicates": 2,
        "random_state": 0,
    }
    plan = _preview(result, **arguments)
    (test,) = plan.tests
    assert test.bound == "exact"
    realized = _realized(lambda: refute(result, **arguments))
    assert realized == test.learner_fits
    monkeypatch.setattr(learned, "frozen_rule_replay", lambda estimator, *args: estimator)
    (mutated,) = _preview(result, **arguments).tests
    assert mutated.learner_fits != realized
