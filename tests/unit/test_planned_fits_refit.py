"""The fits of one refit, the engines the preview covers, and the preflight it shares.

- ``refit_plan`` (X106 part I, comparison plan 6.3): for every kind of :data:`KINDS`, the
  previewed candidate fits of one refit equal the fits the refit makes, counted class-wide on
  the final estimators, and a kind whose refit refuses makes ``refit_plan`` raise the same
  record.
- Coverage (plan 6.5): every engine class the study dispatches to, and every estimator class
  of :data:`KINDS`, defines a preview beside its ``fit``; the exemption set is empty.
- The shared preflight (plan 5.2): the preview and the fit call one preflight function per
  engine, and the fit calls it as often as it did before the preview existed.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sklearn.ensemble
import sklearn.isotonic
import sklearn.linear_model
import sklearn.tree
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly._planned_fits import refit_plan
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.estimators import CTMLE, DRTMLE, TMLE, AdaptiveIPTW, DoseResponseTMLE
from cleverly.estimators.ep import EPEstimator
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
    SurvivalCTMLE,
    SurvivalDRTMLE,
)
from cleverly.longitudinal.estimator import _SequentialEstimator
from cleverly.longitudinal.structural import StructuralNestedMean
from cleverly.survival.engine import ContinuousSurvivalTMLE
from tests.unit import _planned_fits_support as support
from tests.unit._capability_sweep_support import KINDS, fit_continuous_survival

SOURCE = Path(cleverly.__file__).parent

# ------------------------------------------------------------------ the refit of every kind


def _final_estimators() -> tuple[type, ...]:
    """Every class whose ``fit`` is one candidate fit: scikit-learn's and the package's own.

    Wrappers are left out (a pipeline, a screener, the constant-design guard), since their
    fit calls a final estimator counted already, as are the kernel smoothers of
    ``SurvivalDRTMLE``, which the preview does not count (plan 3.1).
    """
    from cleverly.learners._l1_logistic import L1Logistic
    from cleverly.learners.glm import GLM
    from cleverly.learners.parametric import _ParametricLogistic
    from cleverly.survival.cox import CoxHazard, StratifiedNelsonAalen

    found: list[type] = []
    for module in (sklearn.linear_model, sklearn.isotonic, sklearn.ensemble, sklearn.tree):
        found.extend(
            value
            for value in vars(module).values()
            if inspect.isclass(value) and "fit" in vars(value)
        )
    return (*found, GLM, L1Logistic, _ParametricLogistic, CoxHazard, StratifiedNelsonAalen)


@contextlib.contextmanager
def _class_wide(counts: Counter[str]) -> Iterator[None]:
    """Count each outermost ``fit`` of a final estimator in ``counts``, by class."""
    classes = _final_estimators()
    originals = {cls: vars(cls)["fit"] for cls in classes}
    depth = [0]

    def wrap(cls: type, original: Callable[..., Any]) -> Callable[..., Any]:
        def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
            if depth[0] == 0:
                counts[cls.__name__] += 1
            depth[0] += 1
            try:
                return original(self, *args, **kwargs)
            finally:
                depth[0] -= 1

        return fit

    try:
        for cls, original in originals.items():
            cls.fit = wrap(cls, original)  # type: ignore[method-assign]
        yield
    finally:
        for cls, original in originals.items():
            cls.fit = original  # type: ignore[method-assign]


def refit(result: Any) -> Any:
    """One refit of ``result`` on its own data, as the noise-covariate refutation runs it.

    The estimator's ``refit`` with the fit's own seed and the recorded seed of every repeat; a
    continuous-time fit at its recorded seed; an adaptive TMLE fit rerun at the seed its
    provenance records.
    """
    if result.assessment_family == "augmented_trial":
        from cleverly.estimators.atmle_result import fit_trial_effect

        method = result.method.with_overrides(random_state=result.provenance.random_state)
        return fit_trial_effect(result.identified_effect, method)
    estimator = result.estimator
    if result.assessment_family == "continuous_survival":
        return estimator.refit(result.data, random_state=estimator.random_state)
    options: dict[str, Any] = {}
    if hasattr(result, "intermediate_value"):
        options["intermediate_value"] = result.intermediate_value
    seeds = [item.seed for item in getattr(result, "repeats", None) or ()]
    if seeds and all(seed is not None for seed in seeds):
        options.update(random_state=seeds[0], frozen_seeds=seeds)
    return estimator.refit(result.data, **options)


def _refusal(call: Callable[[], Any]) -> tuple[type, Any, str] | None:
    try:
        call()
    except Exception as error:
        return type(error), getattr(getattr(error, "refusal", None), "reason", None), str(error)
    return None


@pytest.mark.parametrize("kind", list(KINDS))
def test_refit_plan_counts_the_fits_of_one_refit(kind: str) -> None:
    result = KINDS[kind].build()
    estimator = getattr(result, "estimator", None)
    # An adaptive TMLE fit keeps no estimator and reruns its method instead.
    reruns = result.assessment_family == "augmented_trial"
    if not reruns and (estimator is None or type(estimator) in UNPREVIEWED):
        # No refit to compare with: an effect curve, a dose-response fit and a
        # variable-importance run keep no estimator, and an unpreviewed engine refits with
        # nothing to price it.  Each refuses as a CapabilityError, never a crash.
        refused = _refusal(lambda: refit_plan(result))
        assert refused is not None and refused[0] is CapabilityError, (kind, refused)
        expected = (
            RefusalReason.INVALID_DECLARATION
            if estimator is None
            else RefusalReason.IMPLEMENTATION_GAP
        )
        assert refused[1] is expected, (kind, refused)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refused = _refusal(lambda: refit_plan(result))
        counts: Counter[str] = Counter()
        with _class_wide(counts):
            refit_refused = _refusal(lambda: refit(result))
    assert refused == refit_refused
    if refused is not None:
        # The reconfigured kinds and a learned regimen, whose refit this version refuses.
        assert not KINDS[kind].refits or kind == "learned_regimen"
        return
    plan = refit_plan(result)
    assert plan.folds == "realized"
    realized = sum(counts.values())
    if plan.exact:
        assert realized == plan.total_candidate_fits, (dict(counts), plan.total_candidate_fits)
    else:
        assert plan.total_fits == plan.total_candidate_fits
        assert plan.min_fits <= realized <= plan.total_fits, (plan.min_fits, realized)


def test_an_unseeded_fit_refits_its_recorded_draws() -> None:
    """``refit_plan`` reads the recorded seeds, so an unseeded fit's folds read as realized."""
    frame, _ = make_linear_ate(n=200, seed=3)
    result = TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        n_folds=4,
        repeats=2,
        simultaneous=False,
        q_bounds=(-30.0, 30.0),
    ).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3", "W4"])
    member = result.single()
    plan = refit_plan(member)
    assert plan.folds == "realized"
    counts: Counter[str] = Counter()
    with _class_wide(counts):
        refit(member)
    assert sum(counts.values()) == plan.total_candidate_fits == 2 * 4 * 2


def test_refit_plan_refuses_a_result_with_no_estimator_as_refute_does() -> None:
    from cleverly.exceptions import CapabilityError, RefusalReason
    from cleverly.validation import refute

    result = TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=False,
        random_state=0,
        estimands=["ate"],
    ).fit(make_linear_ate(n=120, seed=1)[0], **_COLUMNS)
    bare = replace(result.single(), estimator=None)
    with pytest.raises(CapabilityError) as planned:
        refit_plan(bare)
    with pytest.raises(CapabilityError) as refused:
        refute(bare, estimand="ate", tests=("random_common_cause",))
    assert planned.value.refusal.reason == RefusalReason.INVALID_DECLARATION
    assert planned.value.refusal.roadmap is None
    assert "the fitted estimator that produced the result" in str(refused.value)
    assert str(planned.value).replace("refit_plan", "refute") == str(refused.value)


# ------------------------------------------------------------------ coverage

#: The engine classes the study's estimate dispatches to (``study.py``: the point engines,
#: the longitudinal engines, the effect-curve, continuous-time and dose-response engines).
#: The trial route is a function, priced by ``plan_trial_effect``.
DISPATCHED: frozenset[type] = frozenset(
    {
        TMLE,
        CTMLE,
        DRTMLE,
        AdaptiveIPTW,
        EPEstimator,
        LTMLE,
        SDR,
        IPSI,
        SequentialGComputation,
        SequentialIPW,
        ParametricGFormula,
        SurvivalCTMLE,
        SurvivalDRTMLE,
        ContinuousSurvivalTMLE,
        DoseResponseTMLE,
    }
)

#: Engine classes allowed to fit without a preview.  Pinned both ways, and empty.
UNPRICED: frozenset[type] = frozenset()

#: Estimator classes of :data:`KINDS` that the study does not dispatch to and that define no
#: preview yet, each with the roadmap item that owns it.  Pinned both ways, and empty: X106
#: part G2 gave the structural nested mean model, fitted outside CausalStudy.estimate, its
#: preview.
UNPREVIEWED: dict[type, str] = {}


def _priced(cls: type) -> bool:
    """Whether the class that defines ``cls.fit`` defines ``_planned_fits`` beside it, with
    ``fit``'s parameters, and the class that defines ``refit`` defines ``_planned_refit``
    with a ``nominal`` switch, which the refit preview of a refutation reads (X106 part G2).
    ``_planned_fits`` may take ``nominal`` beside ``fit``'s parameters."""
    for entry, preview in (("fit", "_planned_fits"), ("refit", "_planned_refit")):
        owner = next((base for base in cls.__mro__ if entry in vars(base)), None)
        if owner is None:
            continue
        if preview not in vars(owner):
            return False
        if entry == "refit" and "nominal" not in inspect.signature(vars(owner)[preview]).parameters:
            return False
    owner = next(base for base in cls.__mro__ if "fit" in vars(base))
    fit = inspect.signature(vars(owner)["fit"]).parameters
    previewed = inspect.signature(vars(owner)["_planned_fits"]).parameters
    return list(fit) == [name for name in previewed if name != "nominal"]


def _engines() -> set[type]:
    return support.engine_classes()


def test_every_engine_defines_a_preview_beside_its_fit() -> None:
    assert _engines() >= DISPATCHED
    kinds = {
        type(result.estimator)
        for result in (kind.build() for kind in KINDS.values())
        if getattr(result, "estimator", None) is not None
    }
    outside = kinds - _engines()
    assert StructuralNestedMean in outside
    assert {cls for cls in outside if not _priced(cls)} == set(UNPREVIEWED), outside
    assert not any(_priced(cls) for cls in UNPREVIEWED)
    assert UNPREVIEWED == {}
    kinds -= set(UNPREVIEWED)
    unpriced = {cls for cls in _engines() | kinds if not _priced(cls)}
    assert unpriced == UNPRICED
    assert frozenset() == UNPRICED


def test_no_preview_is_left_unfinished() -> None:
    """No marker of an unpriced route remains, and no preview raises ``NotImplementedError``."""
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "TODO(X106" not in text, path
        assert "not priced" not in text, path
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.lstrip("_").startswith(("plan", "refit_plan")):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Raise) and "NotImplementedError" in ast.dump(inner):
                    raise AssertionError(f"{path.name}:{inner.lineno} in {node.name}")


# ------------------------------------------------------------------ the shared preflight


@contextlib.contextmanager
def _calls(targets: list[tuple[Any, str]]) -> Iterator[Counter[str]]:
    """Count the calls of each ``owner.name``."""
    counts: Counter[str] = Counter()
    saved = [(owner, name, getattr(owner, name)) for owner, name in targets]

    def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return counted

    try:
        for owner, name, original in saved:
            setattr(owner, name, wrap(name, original))
        yield counts
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


_POINT = [(TMLE, "_preflight_fit_configuration"), (TMLE, "_repeat_draws")]
_COLUMNS = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2", "W3", "W4"]}


def _point_settings(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "n_folds": 3,
        "random_state": 0,
        "q_bounds": (-20.0, 20.0),
        "simultaneous": False,
        **overrides,
    }


def _cde_frame() -> Any:
    frame, _ = make_linear_ate(n=200, seed=1)
    return frame.assign(Z=np.random.default_rng(0).binomial(1, 0.5, len(frame)))


def _longitudinal(engine: type) -> Callable[[], Any]:
    frame, _ = make_longitudinal(n=300, seed=12)
    settings = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "random_state": 5,
        "simultaneous": False,
        "n_folds": 2,
    }
    columns = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }
    return lambda: engine({"always": 1, "never": 0}, reference="never", **settings).fit(
        frame, **columns
    )


def _cde_refit() -> Any:
    result = TMLE(**_point_settings()).fit(_cde_frame(), intermediate="Z", **_COLUMNS)[0.0]
    return lambda: result.estimator.refit(result.data, intermediate_value=0.0)


#: The calls each fit made on ``origin/main`` before the preview, measured there with
#: these fixtures (``PYTHONPATH=<worktree>/src``); the preview adds none.  CDE fits both
#: levels from one shared preflight and then one per level (``tmle.py``), so three.
BASELINE: dict[str, tuple[Callable[[], Any], list[tuple[Any, str]], dict[str, int]]] = {
    "tmle_repeats": (
        lambda: TMLE(**_point_settings(repeats=2)).fit(
            make_linear_ate(n=200, seed=1)[0], **_COLUMNS
        ),
        _POINT,
        {"_preflight_fit_configuration": 1, "_repeat_draws": 1},
    ),
    "cde_fit": (
        lambda: TMLE(**_point_settings()).fit(_cde_frame(), intermediate="Z", **_COLUMNS),
        _POINT,
        {"_preflight_fit_configuration": 3, "_repeat_draws": 1},
    ),
    "ctmle": (
        lambda: CTMLE(**_point_settings(estimands=["ate"])).fit(
            make_linear_ate(n=200, seed=1)[0], **_COLUMNS
        ),
        _POINT,
        {"_preflight_fit_configuration": 1, "_repeat_draws": 1},
    ),
    "iptw": (
        lambda: AdaptiveIPTW(
            treatment_learner=LogisticRegression(),
            reduced_outcome_learner=LinearRegression(),
            n_folds=3,
            random_state=0,
            estimands=["ate"],
        ).fit(make_linear_ate(n=200, seed=1)[0], **_COLUMNS),
        _POINT,
        {"_preflight_fit_configuration": 1, "_repeat_draws": 1},
    ),
}


def _longitudinal_targets() -> list[tuple[Any, str]]:
    from cleverly.longitudinal import estimator

    return [
        (estimator, "preflight_mechanism_support"),
        (_SequentialEstimator, "_preflight_followers"),
    ]


def _continuous_targets() -> list[tuple[Any, str]]:
    from cleverly.survival import engine

    return [(ContinuousSurvivalTMLE, "_arms"), (engine, "check_memory")]


@pytest.mark.parametrize("name", [*BASELINE, "cde_refit", "ltmle", "sdr", "continuous"])
def test_the_fit_calls_its_preflight_as_often_as_before_the_preview(name: str) -> None:
    if name in BASELINE:
        run, targets, expected = BASELINE[name]
    elif name == "cde_refit":
        run, targets = _cde_refit(), _POINT
        expected = {"_preflight_fit_configuration": 1, "_repeat_draws": 1}
    elif name in ("ltmle", "sdr"):
        run, targets = _longitudinal(LTMLE if name == "ltmle" else SDR), _longitudinal_targets()
        expected = {"preflight_mechanism_support": 1, "_preflight_followers": 1}
    else:
        run, targets = fit_continuous_survival, _continuous_targets()
        expected = {"_arms": 1, "check_memory": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _calls(targets) as counts:
            run()
    assert dict(counts) == expected


def _shared(
    engine: Any, entry: str, preflight: str, fit: Callable[[], Any], preview: Callable[[], Any]
) -> None:
    """``fit`` and ``preview`` each call ``engine.preflight`` once from their entry."""
    del entry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _calls([(engine, preflight)]) as fitted:
            fit()
        with _calls([(engine, preflight)]) as previewed:
            preview()
    assert fitted[preflight] >= 1 and previewed[preflight] == fitted[preflight], (
        dict(fitted),
        dict(previewed),
    )


def test_the_preview_and_the_fit_call_the_same_preflight() -> None:
    frame = make_linear_ate(n=200, seed=1)[0]
    for preflight in ("_single_preflight", "_draw_preflight", "_preflight_fit_configuration"):
        _shared(
            TMLE,
            "fit",
            preflight,
            lambda: TMLE(**_point_settings(repeats=2)).fit(frame, **_COLUMNS),
            lambda: TMLE(**_point_settings(repeats=2))._planned_fits(frame, **_COLUMNS),
        )
    panel, _ = make_longitudinal(n=300, seed=12)
    columns = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }

    def ltmle() -> LTMLE:
        return LTMLE(
            {"always": 1, "never": 0},
            reference="never",
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            random_state=5,
            simultaneous=False,
            n_folds=2,
        )

    _shared(
        _SequentialEstimator,
        "fit",
        "_preflight",
        lambda: ltmle().fit(panel, **columns),
        lambda: ltmle()._planned_fits(panel, **columns),
    )
    result = fit_continuous_survival()
    _shared(
        ContinuousSurvivalTMLE,
        "fit_data",
        "_initial_inputs",
        lambda: result.estimator.refit(result.data, random_state=result.estimator.random_state),
        lambda: refit_plan(result),
    )


def test_the_trial_preview_and_fit_share_their_split() -> None:
    from cleverly.estimators import atmle_result

    result = KINDS["atmle"].build()
    with _calls([(atmle_result, "_outer_folds")]) as counts:
        refit(result)
        refit_plan(result)
    assert counts["_outer_folds"] == 2
