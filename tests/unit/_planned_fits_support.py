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

import contextlib
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import SuperLearner
from cleverly._planned_fits import PlannedFits
from cleverly.learners import library as library_module

#: Fits per label, across every clone.  Reset by :func:`reset`.
COUNTS: Counter[str] = Counter()
#: Fits of each base class, labelled or not.  Filled under :func:`class_wide`.
CLASS_WIDE: Counter[str] = Counter()

_CLASSES: dict[tuple[type, str], type] = {}

#: The base classes whose ``fit`` the class-wide guard patches.  Never a pipeline or a
#: transformer, whose fit calls a final step that is counted already.
BASES: tuple[Any, ...] = (LinearRegression, LogisticRegression)


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

    def wrap(base: type, original: Callable[..., Any]) -> Callable[..., Any]:
        def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
            CLASS_WIDE[base.__name__] += 1
            return original(self, *args, **kwargs)

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
