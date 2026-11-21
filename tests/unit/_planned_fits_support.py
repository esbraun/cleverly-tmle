"""Counting learners and the comparison of a planned-fit preview with the fit it previews.

A preview (:meth:`cleverly.IdentifiedEffect.planned_fits`, or an engine's ``_planned_fits``)
counts the fits ``estimate`` would make.  The witness runs the preview, then the fit, with
learners that count their own fits under a label, and compares:

- an exact row equals the realized count per role and per candidate;
- an upper row satisfies ``min_fits <= realized <= fits``;
- every role the preview prices observes at least one fit, and every role observed is
  priced (the observed-fit guard);
- a class-wide patch of ``fit`` on the base classes totals every candidate fit, labelled or
  not, so an unlabelled fit site makes the total disagree (the class-wide guard);
- no Super Learner candidate failed, so an exact row is tested as exact.

The counting learners are subclasses of the scikit-learn class they count, built per
label, so ``clone`` keeps the label and ``isinstance`` holds.  The class keeps the base
class's name, so the preview names a candidate as it names the base class.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import types
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import SuperLearner
from cleverly._planned_fits import PlannedFits
from cleverly.learners import GLM
from cleverly.learners import library as library_module
from cleverly.survival import CoxHazard

#: Fits per label, across every clone.  Reset by :func:`reset`.
COUNTS: Counter[str] = Counter()
#: Fits of each base class, labelled or not.  Filled under :func:`class_wide`.
CLASS_WIDE: Counter[str] = Counter()

_CLASSES: dict[tuple[type, str], type] = {}

#: The base classes whose ``fit`` the class-wide guard patches: the final estimators the
#: fixtures fit.  Never a pipeline or a transformer, whose fit calls a final step that is
#: counted already.  The guard counts the outermost patched ``fit`` only, so a final
#: estimator that fits another one inside counts once.
BASES: tuple[Any, ...] = (LinearRegression, LogisticRegression, GLM, CoxHazard)


def reset() -> None:
    """Zero every counter."""
    COUNTS.clear()
    CLASS_WIDE.clear()


def counting(base: type, label: str) -> type:
    """The subclass of ``base`` that adds each fit to ``COUNTS[label]``.

    Parameters
    ----------
    base : type
        A scikit-learn estimator class.
    label : str
        The label its fits count under.

    Returns
    -------
    type
        A subclass, one per ``(base, label)``.
    """
    key = (base, label)
    if key not in _CLASSES:
        parent: Any = base

        def fit(self: Any, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
            COUNTS[label] += 1
            # Looked up at call time, so the class-wide patch of ``base.fit`` sees it.
            return parent.fit(self, X, y, *args, **kwargs)

        made = type(base.__name__, (base,), {"fit": fit, "label": label})
        made.__module__ = __name__
        _CLASSES[key] = made
    return _CLASSES[key]


def linear(label: str, **params: Any) -> Any:
    """A counting ``LinearRegression``."""
    return counting(LinearRegression, label)(**params)


def logistic(label: str, **params: Any) -> Any:
    """A counting ``LogisticRegression``."""
    return counting(LogisticRegression, label)(**{"max_iter": 1000, **params})


class CountingSuperLearner(SuperLearner):
    """A Super Learner that adds each top-level fit to ``COUNTS[label + "#ensemble"]``.

    ``label`` is a class attribute of a per-label subclass (:func:`ensemble`), so ``clone``
    and the estimator's seeding keep it.  A one-vs-rest member is a plain
    :class:`SuperLearner`, so only the role's own fits count.
    """

    label = ""

    def fit(self, X: Any, y: Any, sample_weight: Any = None, *, groups: Any = None) -> Any:
        COUNTS[f"{type(self).label}#ensemble"] += 1
        return super().fit(X, y, sample_weight, groups=groups)


_ENSEMBLES: dict[str, type] = {}


def ensemble(
    label: str, base: type = LogisticRegression, names: tuple[str, ...] = ("a", "b"), **params: Any
) -> SuperLearner:
    """A counting Super Learner whose candidates count under ``label:name``.

    Parameters
    ----------
    label : str
        The role label.
    base : type
        The candidates' class.
    names : tuple of str
        The candidates' names.
    **params : Any
        Super Learner settings.

    Returns
    -------
    SuperLearner
        The ensemble.
    """
    if label not in _ENSEMBLES:
        _ENSEMBLES[label] = type("SuperLearner", (CountingSuperLearner,), {"label": label})
    extra = {"max_iter": 1000} if base is LogisticRegression else {}
    library = [(name, counting(base, f"{label}:{name}")(**extra)) for name in names]
    return _ENSEMBLES[label](library=library, **params)


@contextlib.contextmanager
def class_wide() -> Iterator[None]:
    """Count every ``fit`` of the base classes, labelled or not, in ``CLASS_WIDE``."""
    originals: dict[Any, Callable[..., Any]] = {base: base.fit for base in BASES}
    depth = [0]

    def wrap(base: type, original: Callable[..., Any]) -> Callable[..., Any]:
        def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
            if depth[0] == 0:
                CLASS_WIDE[base.__name__] += 1
            depth[0] += 1
            try:
                return original(self, *args, **kwargs)
            finally:
                depth[0] -= 1

        return fit

    try:
        for base, original in originals.items():
            base.fit = wrap(base, original)
        yield
    finally:
        for base, original in originals.items():
            base.fit = original


@contextlib.contextmanager
def counting_default_library() -> Iterator[None]:
    """Replace the default library with counting linear candidates, by task.

    Candidates count under ``"{task}:{name}"`` with the default library's names, so the
    preview that resolves the patched factory names them as the fit does.
    """
    original = library_module.default_library

    def patched(task: str, *, random_state: int | None = None) -> list[tuple[str, Any]]:
        base = LogisticRegression if task == "classification" else LinearRegression
        extra = {"max_iter": 1000} if base is LogisticRegression else {}
        return [
            (name, counting(base, f"{task}:{name}")(**extra))
            for name in ("hist_gradient_boosting", "random_forest", "lasso")
        ]

    module: Any = library_module
    module.default_library = patched
    try:
        yield
    finally:
        module.default_library = original


@dataclass
class Witness:
    """A preview, the fit it previews, and what the fit counted.

    Parameters
    ----------
    plan : PlannedFits
        The preview.
    result : Any
        The fit's result.
    counts : Counter
        Fits per label during the fit.
    class_wide : Counter
        Fits per base class during the fit.
    warnings : list of str
        The messages of the fit's warnings.
    """

    plan: PlannedFits
    result: Any
    counts: Counter[str]
    class_wide: Counter[str]
    warnings: list[str]


def witness(preview: Callable[[], PlannedFits], fit: Callable[[], Any]) -> Witness:
    """Run ``preview``, check it fitted nothing, then run ``fit`` under the class-wide guard.

    Parameters
    ----------
    preview : callable
        Returns the preview.
    fit : callable
        Runs the fit and returns its result.

    Returns
    -------
    Witness
        Both, with the fit's counts.
    """
    reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with class_wide():
            plan = preview()
    assert sum(COUNTS.values()) == 0, f"the preview fitted {dict(COUNTS)}"
    assert sum(CLASS_WIDE.values()) == 0, f"the preview fitted {dict(CLASS_WIDE)}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with class_wide():
            result = fit()
    return Witness(
        plan, result, Counter(COUNTS), Counter(CLASS_WIDE), [str(w.message) for w in caught]
    )


def role_rows(plan: PlannedFits) -> dict[str, list[Any]]:
    """The preview's rows by role."""
    out: dict[str, list[Any]] = {}
    for row in plan.rows:
        out.setdefault(row.role, []).append(row)
    return out


def realized_fits(counts: Mapping[str, int], role: str) -> int:
    """The top-level fits a role made: its plain label, or its ensemble label."""
    return int(counts.get(role, 0) + counts.get(f"{role}#ensemble", 0))


def realized_candidates(counts: Mapping[str, int], role: str) -> dict[str, int]:
    """The candidate fits of a role's ensemble candidates, by name."""
    prefix = f"{role}:"
    return {
        label[len(prefix) :]: int(value)
        for label, value in counts.items()
        if label.startswith(prefix)
    }


def check(
    found: Witness,
    *,
    plain: Mapping[str, str | None] | None = None,
    outside: tuple[str, ...] = ("tuner",),
    engine_role: str | None = None,
) -> None:
    """Assert every comparison of the module docstring on one witness.

    Parameters
    ----------
    found : Witness
        The preview and the fit.
    plain : mapping of str to str or None
        For a role fitted by a plain counting learner, the base-class name the preview
        names its candidate by, or None for the one name the preview gives.  A role not
        listed, or one fitted by a counting ensemble, is compared per candidate.
    outside : tuple of str
        Labels whose learners are not among :data:`BASES`, so the class-wide guard does not
        see them; their fits join the class-wide total only for the comparison with the
        preview.
    engine_role : str or None
        A role whose learner the engine builds itself, so no label counts it: its realized
        fits are the class-wide fits no label claims.
    """
    plain = dict(plain or {})
    plan = found.plan
    counts = Counter(found.counts)
    if engine_role is not None:
        claimed = sum(
            value
            for label, value in counts.items()
            if not label.endswith("#ensemble") and label not in outside
        )
        counts[engine_role] = sum(found.class_wide.values()) - claimed
    assert not [m for m in found.warnings if "dropping learner" in m], found.warnings
    _check_diagnostics(found.result)
    priced = role_rows(plan)
    observed_roles = {label.split(":")[0].split("#")[0] for label in counts}
    missing = observed_roles - set(priced)
    assert not missing, f"the fit observed fits of {sorted(missing)}, which the preview omits"
    for role, rows in priced.items():
        fits = sum(row.fits for row in rows)
        low = sum(row.fits if row.min_fits is None else row.min_fits for row in rows)
        exact = all(row.bound == "exact" for row in rows)
        realized = realized_fits(counts, role)
        if fits > 0:
            assert realized > 0, f"{role}: the preview prices {fits} fits and none was observed"
        if exact:
            assert realized == fits, f"{role}: preview {fits}, fit {realized}"
        else:
            assert low <= realized <= fits, f"{role}: {low} <= {realized} <= {fits} fails"
        planned: Counter[str] = Counter()
        for row in rows:
            planned.update(dict(row.by_candidate))
        if role in plain and f"{role}#ensemble" not in counts:
            name = plain[role] or next(iter(planned), "")
            assert set(planned) == {name}, (role, dict(planned))
            observed = {name: realized}
        else:
            observed = realized_candidates(counts, role)
        if exact:
            assert dict(planned) == observed, f"{role}: preview {dict(planned)}, fit {observed}"
        else:
            for name, value in observed.items():
                assert value <= planned[name], (role, name, value, planned[name])
    labelled = sum(
        value
        for label, value in counts.items()
        if not label.endswith("#ensemble") and label not in outside
    )
    total = sum(found.class_wide.values())
    assert total == labelled, f"an unlabelled fit site: class-wide {total}, labelled {labelled}"
    total += sum(counts.get(label, 0) for label in outside)
    if plan.exact:
        assert total == plan.total_candidate_fits, (total, plan.total_candidate_fits)
    else:
        assert total <= plan.total_candidate_fits, (total, plan.total_candidate_fits)


def _check_diagnostics(result: Any) -> None:
    """No Super Learner candidate failed in any draw of the result, or of each level's."""
    members = getattr(result, "results", None)
    if isinstance(members, Mapping):
        for member in members.values():
            _check_diagnostics(member)
        return
    for repeat in getattr(result, "repeats", ()):
        nuisance = getattr(repeat, "nuisance", None)
        if nuisance is not None:
            values: list[Any] = list(nuisance.diagnostics.values())
        else:
            # A longitudinal draw: the mechanism's diagnostics, node by node.
            mechanism = repeat.mechanism
            values = [
                *mechanism.treatment_diagnostics,
                *mechanism.censoring_diagnostics,
                *mechanism.competing_diagnostics,
            ]
        for value in values:
            for item in value if isinstance(value, (list, tuple)) else (value,):
                failed = getattr(item, "failed", ())
                assert failed == (), failed


def totals(plan: PlannedFits) -> dict[str, int]:
    """The preview's fits by role."""
    out: Counter[str] = Counter()
    for row in plan.rows:
        out[row.role] += row.fits
    return dict(out)


def rare_class_frame(n: int = 240, rare: int = 7, seed: int = 0) -> Any:
    """A binary-treatment frame whose treated arm holds ``rare`` rows.

    With five outer folds each training fold holds five or six treated rows, so a
    five-fold inner Super Learner on the treatment caps its folds there.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = np.zeros(n)
    a[rng.choice(n, size=rare, replace=False)] = 1.0
    y = w[:, 0] + 0.5 * a + rng.normal(size=n)
    return pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "A": a, "Y": y})


# ------------------------------------------------------------------ the refusal replay


class PreviewReturned(BaseException):
    """Raised by a patched fit entry once the preview in its place returned a record.

    A ``BaseException``, so that no ``except Exception`` on the way out of the package
    wraps it, and the replay sees that the preview ran to its end and raised nothing.

    Parameters
    ----------
    plan : PlannedFits
        The preview's record.
    """

    def __init__(self, plan: PlannedFits) -> None:
        super().__init__("the preview returned a record")
        self.plan = plan


#: The fit entries an engine class can define, each with the stand-in that calls the preview
#: beside it.  ``fit`` takes the ``_planned_fits`` its owner defines, which has ``fit``'s
#: parameters (``test_every_engine_defines_a_preview_beside_its_fit``).  The inner entries are
#: :meth:`TMLE._fit_single`, which ``refit`` calls, and the container entries of the
#: continuous-time engine.
def _engine_previews() -> dict[str, Callable[[type], Callable[..., Any]]]:
    def single(owner: type) -> Callable[..., Any]:
        del owner

        def run(self: Any, data: Any, *, intermediate_value: Any = None, shared: Any = None) -> Any:
            del shared
            return self._planned_fit_single(data, intermediate_value=intermediate_value)

        return run

    def container(owner: type) -> Callable[..., Any]:
        del owner

        def run(self: Any, data: Any, *, folds: Any = None, retained: bool = False) -> Any:
            del retained
            return self._planned_fit_data(data, folds=folds)

        return run

    return {
        "fit": lambda owner: vars(owner)["_planned_fits"],
        "_fit_single": single,
        "fit_data": lambda owner: lambda self, data: self._planned_fit_data(data),
        "_fit_data": container,
    }


#: Names of the fit entries an engine class can define.
ENGINE_ENTRIES: tuple[str, ...] = ("fit", "_fit_single", "fit_data", "_fit_data")


def _subclasses(base: type) -> set[type]:
    out = {base}
    for child in base.__subclasses__():
        out |= _subclasses(child)
    return out


def engine_classes() -> set[type]:
    """The engines the study dispatches to, and every subclass of one loaded."""
    from cleverly.estimators.ep import EPEstimator
    from cleverly.estimators.tmle import TMLE
    from cleverly.longitudinal.estimator import _SequentialEstimator
    from cleverly.survival.engine import ContinuousSurvivalTMLE

    package_classes()  # every module imported, so every subclass is loaded
    found = _subclasses(TMLE) | _subclasses(_SequentialEstimator)
    return found | {EPEstimator, ContinuousSurvivalTMLE}


def standalone_engines() -> set[type]:
    """Engines with a planned-fit preview that the study layer does not dispatch to.

    ``StructuralNestedMean`` is fitted directly; its preview prices its refits for
    :func:`cleverly.validation.planned_refits`, so its ``fit`` is a previewed entry too.
    """
    from cleverly.longitudinal.structural import StructuralNestedMean

    return {StructuralNestedMean}


@functools.cache
def package_classes() -> tuple[type, ...]:
    """Every class the ``cleverly`` package defines, from every module, in a fixed order."""
    import importlib
    import inspect
    import pkgutil

    import cleverly

    found: dict[str, type] = {}
    for info in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if inspect.isclass(value) and value.__module__ == module.__name__:
                found.setdefault(f"{value.__module__}.{value.__qualname__}", value)
    return tuple(found[name] for name in sorted(found))


def engine_entry_owners() -> tuple[tuple[type, str], ...]:
    """Each ``(class, name)`` where a package engine defines a fit entry in its own body.

    An engine is a class with a planned-fit preview (``_planned_fits``) on its MRO; a
    learner, a result or a law that defines ``fit`` has none.
    """
    return tuple(
        (cls, name)
        for cls in package_classes()
        if any("_planned_fits" in vars(base) for base in cls.__mro__)
        for name in ENGINE_ENTRIES
        if name in vars(cls)
    )


def study_fit_functions() -> tuple[tuple[Any, str, str], ...]:
    """Each module-level fit function :mod:`cleverly.study` imports, with its preview.

    ``(module, "fit_x", "plan_x")``: :meth:`cleverly.IdentifiedEffect.estimate` calls
    ``fit_x`` for a route with no engine class, and :meth:`~cleverly.IdentifiedEffect.planned_fits`
    calls ``plan_x`` from the same module.
    """
    import importlib

    import cleverly.study as study

    tree = ast.parse(Path(study.__file__).read_text(encoding="utf-8"))
    out: list[tuple[Any, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            if not alias.name.startswith("fit_"):
                continue
            module = importlib.import_module("." + node.module, "cleverly" if node.level else None)
            out.append((module, alias.name, "plan_" + alias.name[len("fit_") :]))
    return tuple(sorted(set(out), key=lambda item: (item[0].__name__, item[1])))


def preview_entries(
    drop: frozenset[str] = frozenset(),
) -> tuple[tuple[Any, str, Callable[..., Any]], ...]:
    """Each fit entry of the package, with the preview that stands in for it, derived.

    ``(owner, name, preview)``: calling ``owner.name(...)`` under the replay calls
    ``preview(...)`` instead.  The entries are every fit entry an engine class defines in its
    own body (:func:`engine_entry_owners`), every fit function the study layer imports with
    its ``plan_`` sibling (:func:`study_fit_functions`), and
    :meth:`cleverly.IdentifiedEffect.estimate` with
    :meth:`~cleverly.IdentifiedEffect.planned_fits`.  ``refit`` is not an entry: a refit
    refusal raises as it does and its inner entry previews.

    Parameters
    ----------
    drop : frozenset of str
        Labels (:func:`entry_label`) to leave out: the mutation control's handle.
    """
    from cleverly.study import IdentifiedEffect

    previews = _engine_previews()
    entries: list[tuple[Any, str, Callable[..., Any]]] = [
        (cls, name, previews[name](cls)) for cls, name in engine_entry_owners()
    ]
    for module, name, plan in study_fit_functions():
        entries.append((module, name, getattr(module, plan)))
    entries.append((IdentifiedEffect, "estimate", IdentifiedEffect.planned_fits))
    return tuple(entry for entry in entries if entry_label(entry[0], entry[1]) not in drop)


def entry_label(owner: Any, name: str) -> str:
    """``"TMLE.fit"`` for a class entry, ``"fit_trial_effect"`` for a module function."""
    return name if isinstance(owner, types.ModuleType) else f"{owner.__qualname__}.{name}"


@contextlib.contextmanager
def previews_in_place_of_fits(
    reached: list[str], drop: frozenset[str] = frozenset()
) -> Iterator[None]:
    """Patch every entry of :func:`preview_entries`; each one reached is appended to ``reached``.

    A patched entry runs its preview and raises :class:`PreviewReturned` with the record, so
    the caller's ``build()`` stops there whatever it would do with a fitted result.  ``drop``
    leaves entries unpatched, for the mutation control.
    """

    def stand_in(label: str, preview: Callable[..., Any]) -> Callable[..., Any]:
        def entry(*args: Any, **kwargs: Any) -> Any:
            reached.append(label)
            raise PreviewReturned(preview(*args, **kwargs))

        return entry

    entries = preview_entries(drop)
    saved = [(owner, name, vars(owner)[name]) for owner, name, _ in entries]
    try:
        for owner, name, preview in entries:
            setattr(owner, name, stand_in(entry_label(owner, name), preview))
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


def _record(error: BaseException) -> tuple[type, Any, str]:
    """An exception's class, reason code and message."""
    refusal = getattr(error, "refusal", None)
    return type(error), getattr(refusal, "reason", None), str(error)


def raised_in(error: BaseException) -> set[str]:
    """The qualified names of the functions on ``error``'s traceback and its chained causes.

    Qualified as :attr:`cleverly.PlannedFits.unpreviewed` names them, such as
    ``"Piece.__post_init__"``.
    """
    import traceback

    names: set[str] = set()
    seen: BaseException | None = error
    while seen is not None:
        names.update(frame.f_code.co_qualname for frame, _ in traceback.walk_tb(seen.__traceback__))
        seen = seen.__cause__ or seen.__context__
    return names


#: Names of the package functions on a fit path: the engine entries, any ``fit_`` function,
#: and :meth:`cleverly.IdentifiedEffect.estimate` (by qualified name: the
#: ``CausalStudy.estimate`` shortcut runs ``identify`` first).  A refusal raised under one
#: of them, by a replay that reached no patched entry, went through an entry the replay does
#: not patch.
_FIT_PATH = frozenset(ENGINE_ENTRIES)
_FIT_PATH_QUALIFIED = frozenset({"IdentifiedEffect.estimate"})


def fit_path_frames(error: BaseException) -> list[str]:
    """The package frames on ``error``'s traceback, or its causes', that lie on a fit path."""
    import traceback

    import cleverly

    source = str(Path(cleverly.__file__).parent)
    out: list[str] = []
    seen: BaseException | None = error
    while seen is not None:
        for frame, _ in traceback.walk_tb(seen.__traceback__):
            code = frame.f_code
            if code.co_filename.startswith(source) and (
                code.co_name in _FIT_PATH
                or code.co_name.startswith("fit_")
                or code.co_qualname in _FIT_PATH_QUALIFIED
            ):
                out.append(code.co_qualname)
        seen = seen.__cause__ or seen.__context__
    return out


def assert_preview_parity(
    build: Callable[[], Any], original: BaseException, drop: frozenset[str] = frozenset()
) -> list[str]:
    """Replay ``build()`` with every fit entry previewed, and compare with ``original``.

    The replay either reaches a patched entry, and the preview raises ``original``'s class,
    reason code and message, or refuses upstream of every entry (in ``identify``, a
    constructor or the method's normalization) with the same record, and then ``original``'s
    traceback holds no package frame on a fit path (:func:`fit_path_frames`).  One exception:
    when ``original`` was raised inside a check the preview defers (a name in the record's
    ``unpreviewed``), the preview raises nothing.  Anything else raises ``AssertionError``.

    Parameters
    ----------
    build : callable
        The refused request.
    original : BaseException
        What ``build()`` raised.
    drop : frozenset of str
        Entries left unpatched, for the mutation control.

    Returns
    -------
    list of str
        The entries the replay reached, empty for an upstream refusal.
    """
    reached: list[str] = []
    with previews_in_place_of_fits(reached, drop):
        try:
            build()
        except PreviewReturned as returned:
            deferred = set(returned.plan.unpreviewed)
            assert deferred & raised_in(original), (
                f"the preview at {reached[0]} raised nothing, and estimate raised {original!r} "
                f"outside the deferred checks {sorted(deferred)}"
            )
            return reached
        except Exception as replayed:
            where = (
                f"the preview at {reached[0]}" if reached else "the replay, with no entry reached,"
            )
            assert _record(replayed) == _record(original), (
                f"{where} raised {_record(replayed)}; the fit raised {_record(original)}"
            )
            if not reached:
                on_path = fit_path_frames(original)
                assert not on_path, (
                    f"the replay reached no patched entry, and the fit raised {original!r} "
                    f"inside the fit path {on_path}: an entry the replay does not patch"
                )
            return reached
    raise AssertionError(f"the replay raised nothing; the fit raised {original!r}")
