"""The planned-fit preview refuses what ``estimate`` refuses, and names what it leaves to the fit.

Three witnesses (X106 part I, comparison plan 6.4):

- Every use of :func:`tests.unit._declaration_support.assert_refused_before_any_call` replays
  its request with each fit entry replaced by the preview.  A static test here checks that no
  use bypasses that helper, and two mutation controls check that the replay has teeth.
- On a tuned cap whose boundary is a learner, the preview stops before the tuner's fit.  A
  ``sys.setprofile`` hook finds the checks ``estimate`` runs after that point and before its
  first nuisance learner that the preview never runs, and the set must equal the record's
  ``unpreviewed`` and the pinned set.  A dummy check added after the stop must fail it.
- When ``estimate`` raises one of those deferred checks, here the tuner's weight refusal, the
  preview raises nothing, and a refusal outside them is not excused.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import sys
import types
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import cleverly
import cleverly.exceptions as exceptions
from cleverly import (
    AdaptiveTMLEMethod,
    CausalStudy,
    CrossFitting,
    LongitudinalTreatment,
    ModelSpec,
    RegimeMean,
    TrialATE,
    TrialWithExternalData,
)
from cleverly.datasets import make_linear_ate
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import Shift, SupportBoundary
from cleverly.longitudinal import DynamicRegimen
from tests import law_q_point as lawp
from tests import law_q_tuned as lawq
from tests.conftest import linear_in_sample
from tests.unit import _planned_fits_support as support
from tests.unit._declaration_support import assert_refused_before_any_call
from tests.unit._natural_course_support import NeverFit, never_fit_learners

pytestmark = pytest.mark.xdist_group("planned_fits_parity")

tmle_module = importlib.import_module("cleverly.estimators.tmle")
drtmle_module = importlib.import_module("cleverly.estimators.drtmle")
SOURCE = Path(cleverly.__file__).parent
TESTS = Path(__file__).resolve().parents[1]
HELPER = "assert_refused_before_any_call"

# ------------------------------------------------------------------ every use replays


def _uses() -> dict[Path, int]:
    """The call sites of the helper per test module, outside the module that defines it."""
    out: dict[Path, int] = {}
    for path in sorted(TESTS.rglob("*.py")):
        if path.name == "_declaration_support.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == HELPER)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == HELPER)
            )
        )
        if calls:
            out[path] = calls
    return out


def test_every_use_of_the_refusal_helper_replays_through_the_preview() -> None:
    """Each module that calls the helper imports it from the declaration support module.

    A module that defined its own helper of that name, or imported one from elsewhere, would
    check its refusals without the preview replay.
    """
    uses = _uses()
    # The count when the replay was added (X106 part I); new uses only add to it.
    assert sum(uses.values()) >= 55, uses
    for path in uses:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "tests.unit._declaration_support"
            and any(alias.name == HELPER and alias.asname is None for alias in node.names)
            for node in ast.walk(tree)
        )
        defined = any(
            isinstance(node, ast.FunctionDef) and node.name == HELPER for node in ast.walk(tree)
        )
        assert imported and not defined, f"{path.relative_to(TESTS)} does not use the helper"
    # The helper itself replays: its body calls the parity check.
    support_tree = ast.parse((TESTS / "unit" / "_declaration_support.py").read_text())
    (helper,) = (
        node
        for node in support_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == HELPER
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_preview_parity"
        for node in ast.walk(helper)
    )


def _strata_frame() -> pd.DataFrame:
    frame, _ = make_linear_ate(n=200, seed=2)
    frame = frame.assign(S=(frame["W1"] > 0).astype(int))
    return frame.assign(A=np.where(frame["S"] == 1, 0, frame["A"]))


def _drtmle_fold_refusal() -> Any:
    """A DR-TMLE stratum with no trainable rows, which a fold check refuses."""
    settings = linear_in_sample(
        **never_fit_learners(),
        reduced_outcome_learner=NeverFit(),
        reduced_treatment_learner=NeverFit(),
        reduced_mechanism_residual_learner=NeverFit(),
    )
    return DRTMLE(estimands=("ate",), **settings).fit(
        _strata_frame(),
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2", "W3", "W4", "S"],
        strata=["S"],
    )


@contextlib.contextmanager
def _during_preview(patch: pytest.MonkeyPatch, owner: Any, name: str, value: Any) -> Iterator[None]:
    """``owner.name`` replaced by ``value`` while a preview runs, and only then."""
    original = getattr(owner, name)

    def wrap(preview: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            setattr(owner, name, value)
            try:
                return preview(*args, **kwargs)
            finally:
                setattr(owner, name, original)

        return run

    patch.setattr(TMLE, "_planned_fits", wrap(TMLE._planned_fits))
    patch.setattr(TMLE, "_planned_fit_single", wrap(TMLE._planned_fit_single))
    yield


def test_mutation_a_preview_that_skips_the_preflight_fails_the_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cleverly._planned_fits import PlannedFits

    monkeypatch.setattr(TMLE, "_planned_fits", lambda self, data, **kwargs: PlannedFits(rows=()))
    with pytest.raises(AssertionError, match="raised nothing"):
        assert_refused_before_any_call(
            _drtmle_fold_refusal, None, "", "no trainable rows", error=ValueError
        )


def test_mutation_a_preview_that_skips_the_fold_checks_fails_the_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_refused_before_any_call(
        _drtmle_fold_refusal, None, "", "no trainable rows", error=ValueError
    )
    with (
        _during_preview(
            monkeypatch, drtmle_module, "refuse_untrainable_stratum_folds", lambda *a, **k: None
        ),
        pytest.raises(AssertionError, match="raised nothing"),
    ):
        assert_refused_before_any_call(
            _drtmle_fold_refusal, None, "", "no trainable rows", error=ValueError
        )


# ------------------------------------------------------------------ the trial route


def _trial_request(g_bounds: Any) -> tuple[Any, Any]:
    """A trial effect and an adaptive TMLE method with the given ``g_bounds``."""
    rng = np.random.default_rng(0)
    n = 120
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    a = rng.integers(0, 2, n)
    frame = pd.DataFrame({"Y": a + w + rng.normal(size=n), "A": a, "W": w, "S": s})
    design = TrialWithExternalData(outcome="Y", treatment="A", adjustment=("W",), source="S")
    effect = CausalStudy(frame, design=design).identify(TrialATE("trial"))
    learners = never_fit_learners()
    method = AdaptiveTMLEMethod(
        models=ModelSpec(
            outcome_learner=learners["outcome_learner"],
            treatment_learner=learners["treatment_learner"],
        ),
        source_learner=NeverFit(),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
    ).with_overrides(random_state=0, g_bounds=g_bounds)
    return effect, method


def _trial_estimate(g_bounds: Any) -> Any:
    """Adaptive TMLE of a trial with external rows, refused for its ``g_bounds``."""
    effect, method = _trial_request(g_bounds)
    return effect.estimate(method)


@pytest.mark.parametrize(
    ("g_bounds", "fragment"),
    [
        ((0.9, 0.1), "g_bounds pair must satisfy 0 < lower < upper < 1"),
        (0.6, "scalar g_bounds must lie in (0, 0.5)"),
        ("nope", "g_bounds must be 'auto', a float, or a pair"),
    ],
)
def test_the_trial_preview_refuses_bad_g_bounds_as_the_fit_does(
    g_bounds: Any, fragment: str
) -> None:
    """The trial route's preview runs the fit's own pre-learner prefix (X106 review B1)."""
    reached: list[str] = []
    raised = assert_refused_before_any_call(
        lambda: _trial_estimate(g_bounds), None, "", fragment, error=ValueError
    )
    with support.previews_in_place_of_fits(reached), pytest.raises(ValueError):
        _trial_estimate(g_bounds)
    assert reached == ["IdentifiedEffect.estimate"], reached
    assert "resolve_g_bounds" in support.raised_in(raised)


# ------------------------------------------------------------------ the replay's entries


def test_the_replay_patches_every_fit_entry_both_ways() -> None:
    """The patched entries are derived, and equal the engines' fit entries and the study's.

    Every engine class that defines ``fit`` in its own body has an entry, and every entry's
    owner is such a class.  Every fit entry an engine resolves through its MRO has an entry
    at the class that defines it.  Every ``_estimate_*`` route of ``IdentifiedEffect``
    calls an engine entry or a study fit function, and each study fit function has a
    ``plan_`` sibling.
    """
    owners = support.engine_entry_owners()
    engines = support.engine_classes() | support.standalone_engines()
    assert {cls for cls, name in owners if name == "fit"} == {
        cls for cls in engines if "fit" in vars(cls)
    }
    assert {cls for cls, _ in owners} <= engines, {cls for cls, _ in owners} - engines
    for cls in engines:
        for name in support.ENGINE_ENTRIES:
            owner = next((base for base in cls.__mro__ if name in vars(base)), None)
            if owner is not None:
                assert (owner, name) in owners, (cls, owner, name)
    functions = support.study_fit_functions()
    assert {(module.__name__, name) for module, name, _ in functions} == {
        ("cleverly.estimators.atmle_result", "fit_trial_effect")
    }
    for module, _, plan in functions:
        assert callable(getattr(module, plan))
    from cleverly.study import IdentifiedEffect

    called_names = {*support.ENGINE_ENTRIES, *(name for _, name, _ in functions)}
    tree = ast.parse(Path(cleverly.study.__file__).read_text(encoding="utf-8"))
    (cls_node,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == IdentifiedEffect.__name__
    )
    routes = [
        node
        for node in cls_node.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_estimate_")
    ]
    assert len(routes) == 6, [node.name for node in routes]
    for route in routes:
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(route)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert calls & called_names, (route.name, sorted(calls))
    labels = [support.entry_label(owner, name) for owner, name, _ in support.preview_entries()]
    assert len(labels) == len(set(labels)), labels


def _trial_direct() -> Any:
    """The trial route's fit function, called directly, refused for its ``g_bounds``."""
    from cleverly.estimators.atmle_result import fit_trial_effect

    effect, method = _trial_request((0.9, 0.1))
    return fit_trial_effect(effect, effect._trial_method(method))


@pytest.mark.parametrize(
    ("drop", "build", "fragment"),
    [
        (frozenset({"fit_trial_effect"}), _trial_direct, "g_bounds pair"),
        (
            frozenset({"TMLE.fit", "TMLE._fit_single"}),
            _drtmle_fold_refusal,
            "no trainable rows",
        ),
    ],
    ids=["trial function", "point engine"],
)
def test_mutation_an_entry_dropped_from_the_derived_list_fails_the_replay(
    drop: frozenset[str], build: Callable[[], Any], fragment: str
) -> None:
    raised = assert_refused_before_any_call(build, None, "", fragment, error=ValueError)
    with pytest.raises(AssertionError, match="an entry the replay does not patch"):
        support.assert_preview_parity(build, raised, drop)


# ------------------------------------------------------------------ the tuner's weight refusal


class _NoWeightRidge(Ridge):
    """A boundary learner whose ``fit`` takes no ``sample_weight``."""

    def fit(self, X: Any, y: Any) -> Any:  # type: ignore[override]
        return super().fit(X, y)


def _tuned_weighted(learner: Any) -> Any:
    frame = lawp.sample(240, 11)
    frame = frame.assign(w=np.linspace(0.5, 1.5, len(frame)))
    cap = SupportBoundary(learner, 0.5, features=("W",))
    return TMLE(
        **never_fit_learners(),
        policies=[Shift(1.0, cap=cap, target="fold_average", name="tuned"), Shift(0.0, cap=None)],
        density_bins=10,
        q_bounds=(0.0, 1.0),
        n_folds=3,
        random_state=0,
        simultaneous=False,
    ).fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W"],
        treatment_kind="continuous",
        weights="w",
    )


def test_the_tuner_weight_refusal_is_deferred_and_the_preview_raises_nothing() -> None:
    raised = assert_refused_before_any_call(
        lambda: _tuned_weighted(_NoWeightRidge()), None, "", "sample_weight"
    )
    assert "fit_tuner" in support.raised_in(raised)
    reached: list[str] = []
    with support.previews_in_place_of_fits(reached), pytest.raises(support.PreviewReturned) as done:
        _tuned_weighted(_NoWeightRidge())
    assert reached == ["TMLE.fit"]
    assert "fit_tuner" in done.value.plan.unpreviewed


def test_a_refusal_outside_the_deferred_checks_is_not_excused() -> None:
    """The preview of an admitted request returns; a refusal it does not list fails parity."""
    with pytest.raises(AssertionError, match="outside the deferred checks"):
        support.assert_preview_parity(
            lambda: _tuned_weighted(LinearRegression()),
            CapabilityError(
                "a refusal no deferred check raised", reason=RefusalReason.INVALID_DECLARATION
            ),
        )


# ------------------------------------------------------------------ the deferred checks

#: Names of every ``CleverlyError`` subclass, and the two names that build a refusal.
_REFUSING = frozenset(
    {
        name
        for name, value in vars(exceptions).items()
        if isinstance(value, type) and issubclass(value, exceptions.CleverlyError)
    }
    | {"Refusal", "from_refusal"}
)


def _refuses(code: types.CodeType) -> bool:
    """Whether ``code`` raises a ``CleverlyError`` subclass or builds a ``Refusal``."""
    return bool(_REFUSING & set(code.co_names))


class _Tuner(GradientBoostingRegressor):
    """The boundary learner, whose own ``fit`` marks where the preview stops."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None, monitor: Any = None) -> Any:
        return super().fit(X, y, sample_weight=sample_weight, monitor=monitor)


class _Outcome(LinearRegression):
    """A nuisance learner, whose own ``fit`` marks the first nuisance fit."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        return super().fit(X, y, sample_weight=sample_weight)


class _Mechanism(LogisticRegression):
    """A nuisance classifier, whose own ``fit`` marks the first nuisance fit."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        return super().fit(X, y, sample_weight=sample_weight)


_STOP = frozenset({_Outcome.fit.__code__, _Mechanism.fit.__code__})


def _in_package(code: types.CodeType) -> bool:
    return code.co_filename.startswith(str(SOURCE))


def _window(run: Callable[[], Any]) -> tuple[list[types.CodeType], list[types.CodeType]]:
    """The package functions ``run`` calls from the first tuner fit to the first nuisance fit.

    Also the package frames active when the window opens, innermost first: the tuner fit's
    callers, which run their remaining code inside the window.
    """
    calls: list[types.CodeType] = []
    state: dict[str, Any] = {"on": False, "done": False, "stack": []}

    def hook(frame: types.FrameType, event: str, arg: Any) -> None:
        del arg
        if event != "call":
            return
        code = frame.f_code
        if code is _Tuner.fit.__code__ and not (state["on"] or state["done"]):
            state["on"] = True
            back = frame.f_back
            while back is not None:
                state["stack"].append(back.f_code)
                back = back.f_back
        elif code in _STOP and state["on"]:
            state["on"], state["done"] = False, True
        elif state["on"] and _in_package(code):
            calls.append(code)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.setprofile(hook)
        try:
            run()
        finally:
            sys.setprofile(None)
    assert state["done"], "the window never closed: no tuner fit, or no nuisance fit after it"
    return calls, state["stack"]


def _called(run: Callable[[], Any]) -> set[types.CodeType]:
    """Every package function ``run`` calls."""
    out: set[types.CodeType] = set()

    def hook(frame: types.FrameType, event: str, arg: Any) -> None:
        del arg
        if event == "call" and _in_package(frame.f_code):
            out.add(frame.f_code)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.setprofile(hook)
        try:
            run()
        finally:
            sys.setprofile(None)
    return out


def deferred(fit: Callable[[], Any], preview: Callable[[], Any]) -> set[str]:
    """The checks ``fit`` runs after the preview's stop that the preview never runs.

    A package function counts when it is called inside the window, or is a caller of the
    first tuner fit below the innermost function the preview also calls, and when its code
    can refuse.  Each is named by its qualified name, as ``unpreviewed`` names it.
    """
    calls, stack = _window(fit)
    previewed = _called(preview)
    callers: list[types.CodeType] = []
    for code in stack:
        if code in previewed:
            break
        if _in_package(code):
            callers.append(code)
    return {
        code.co_qualname for code in {*calls, *callers} if code not in previewed and _refuses(code)
    }


def _point() -> tuple[Callable[[], Any], Callable[[], Any]]:
    frame = lawp.sample(240, 11)
    columns = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": ["W"],
        "treatment_kind": "continuous",
    }

    def engine() -> TMLE:
        cap = SupportBoundary(
            _Tuner(loss="quantile", alpha=0.5, n_estimators=5, random_state=0), 0.5, features=("W",)
        )
        return TMLE(
            outcome_learner=_Outcome(),
            treatment_learner=_Mechanism(max_iter=1000),
            policies=[
                Shift(1.0, cap=cap, target="fold_average", name="tuned"),
                Shift(0.0, cap=None),
            ],
            density_bins=10,
            q_bounds=(0.0, 1.0),
            n_folds=3,
            random_state=0,
            simultaneous=False,
        )

    return (
        lambda: engine().fit(frame, **columns),
        lambda: engine()._planned_fits(frame, **columns),
    )


def _longitudinal(method: str | None) -> tuple[Callable[[], Any], Callable[[], Any]]:
    cap = SupportBoundary(_Tuner(loss="quantile", alpha=0.5, n_estimators=20), 0.5, features=("W",))
    regimen = DynamicRegimen(
        "t", (Shift(1.0, cap=cap, target="fold_average"), Shift(0.0, cap=None))
    )
    study = CausalStudy(
        lawq.sample(600, 11),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["A1", "A2"],
        ),
    )
    effect = study.identify(RegimeMean({"t": regimen, "natural": Shift(0.0, cap=None)}))

    def settings() -> dict[str, Any]:
        return {
            "outcome_learner": _Outcome(),
            "pseudo_learner": _Outcome(),
            "treatment_learner": _Mechanism(max_iter=1000),
            "censoring_learner": _Mechanism(max_iter=1000),
            "learner_folds": 2,
            "n_folds": 3,
            "random_state": 5,
            "simultaneous": False,
            "density_bins": 10,
            "q_bounds": (0, 1),
        }

    return (
        lambda: effect.estimate(method, **settings()),
        lambda: effect.planned_fits(method, **settings()),
    )


#: The tuned policies' own checks, which the point and longitudinal fits share.
_POLICY_CHECKS = frozenset(
    {
        "fit_tuner",
        "Piece.__post_init__",
        "ModifiedPolicy.__post_init__",
        "_check_pieces",
        "_check_partition",
        "_check_monotone",
        "_per_row",
    }
)

#: The deferred set of each tuned fixture, found by the profiler and pinned here.
PINNED: dict[str, frozenset[str]] = {
    "point": _POLICY_CHECKS
    | {
        "_fitted",
        "FunctionDeclaration.check",
        "check_continuous_policy",
        "_refuse_undeclared_atoms",
    },
    **{
        f"longitudinal {method}": _POLICY_CHECKS
        | {
            "_resolve_tuned_plan",
            "_guarded",
            f"{owner}._resolve_tilts",
            "_SequentialEstimator._refuse_fold_average_compositions",
            "_SequentialEstimator._refuse_undeclared_policy_scale",
            "preflight_known_mechanisms",
            "_check_policy_arm_support",
            "_require_regimen_followers",
        }
        for method, owner in (("ltmle", "_SequentialEstimator"), ("sdr", "SDR"))
    },
}

FIXTURES: dict[str, Callable[[], tuple[Callable[[], Any], Callable[[], Any]]]] = {
    "point": _point,
    "longitudinal ltmle": lambda: _longitudinal(None),
    "longitudinal sdr": lambda: _longitudinal("sdr"),
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_deferred_checks_are_found_listed_and_pinned(name: str) -> None:
    fit, preview = FIXTURES[name]()
    found = deferred(fit, preview)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = preview()
    assert found == set(plan.unpreviewed), found ^ set(plan.unpreviewed)
    assert found == PINNED[name]
    previewed = {code.co_qualname for code in _called(preview)}
    assert not previewed & PINNED[name]


def test_a_design_refusal_the_tuned_preview_skips_is_listed() -> None:
    """Each longitudinal engine lists its ``_refuse_design`` exactly when it can refuse.

    The preview of a learner-backed tuned cap resolves no plans, so it skips
    ``_refuse_design``.  No study request reaches a tuned g-computation, IPW or parametric
    g-formula fit past its earlier refusals, so no profiler fixture covers them.  The check
    here uses the profiler's own test of a function that can refuse (:func:`_refuses`): the
    engine lists the method it resolves when that method's code can refuse, and not when it
    is the base class's empty check.
    """
    import inspect

    from cleverly.longitudinal.estimator import _SequentialEstimator

    listed: set[str] = set()
    for cls in support._subclasses(_SequentialEstimator):
        if inspect.isabstract(cls):
            continue
        engine = object.__new__(cls)
        engine.survival_model = None
        engine.hazard_pooling = None
        names = set(cls._unpreviewed_checks(engine, None))
        method = cls._refuse_design
        expected = {method.__qualname__} if _refuses(method.__code__) else set()
        assert {name for name in names if name.endswith("._refuse_design")} == expected, cls
        listed |= expected
    assert listed == {
        "SequentialGComputation._refuse_design",
        "SequentialIPW._refuse_design",
        "ParametricGFormula._refuse_design",
    }


def _dummy_post_stop_check(*args: Any, **kwargs: Any) -> None:
    """A check that can refuse, placed after the stop by the mutation control."""
    if kwargs.get("never"):
        raise CapabilityError(
            "a check the preview does not run", reason=RefusalReason.IMPLEMENTATION_GAP
        )


@pytest.mark.parametrize(
    ("name", "owner", "attribute"),
    [
        ("point", tmle_module, "preflight_point_policies"),
        ("longitudinal ltmle", "_SequentialEstimator", "_refuse_undeclared_policy_scale"),
    ],
)
def test_mutation_a_check_added_after_the_stop_fails(
    monkeypatch: pytest.MonkeyPatch, name: str, owner: Any, attribute: str
) -> None:
    if isinstance(owner, str):
        from cleverly.longitudinal import estimator as longitudinal

        owner = getattr(longitudinal, owner)
    original = getattr(owner, attribute)
    # Compiled as package code, so the profiler sees it as a function of the package.
    dummy = types.FunctionType(
        _dummy_post_stop_check.__code__.replace(
            co_filename=str(SOURCE / "_dummy_check.py"), co_qualname="_dummy_post_stop_check"
        ),
        {"CapabilityError": CapabilityError, "RefusalReason": RefusalReason},
    )

    def checked(*args: Any, **kwargs: Any) -> Any:
        dummy()
        return original(*args, **kwargs)

    checked.__code__ = checked.__code__.replace(co_filename=str(SOURCE / "_dummy_check.py"))
    monkeypatch.setattr(owner, attribute, checked)
    fit, preview = FIXTURES[name]()
    found = deferred(fit, preview)
    assert "_dummy_post_stop_check" in found
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = preview()
    assert found != set(plan.unpreviewed)
