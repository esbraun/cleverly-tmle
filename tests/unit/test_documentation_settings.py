"""The settings reference agrees with the configuration objects it documents.

``docs/user-guide/settings.md`` lists every field of the five setting groups (``ModelSpec``,
``CrossFitting``, ``Targeting``, ``Inference``, ``Runtime``) with its keyword, its default and the
fits that use it, and every option of each typed method with its default. Each of those cells is
a fact the code owns, so this module derives it from the code and compares:

* the field names, against :func:`dataclasses.fields`;
* the defaults, against ``repr`` of the field default;
* the keyword, against :data:`cleverly.methods.SHORTCUTS`;
* the "used on" kinds, by translating a method that changes the field
  (:meth:`TMLEMethod.estimator_kwargs`) and from the read sets of the exact-times engine
  (``cleverly.survival.data._READ``) and of the trial method (``ATMLE_FIELDS``);
* the table of changed defaults, against each method's groups and the resolved translations.

A field added to a group, a renamed keyword, a changed default or a newly routed setting fails
here until the page says so. Each check has a deliberate-mutation control that runs it on a
doctored page.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (
    CrossFitting,
    Inference,
    ModelSpec,
    Runtime,
    SplitPlan,
    Targeting,
    TMLEMethod,
)
from cleverly.learners import random_partition
from cleverly.methods import ATMLE_FIELDS, SHORTCUTS
from cleverly.survival.data import _READ, continuous_kwargs

PAGE = Path(__file__).resolve().parents[2] / "docs" / "user-guide" / "settings.md"

#: Section heading on the page to the group it documents, as ``(attribute, class)``.
GROUP_SECTIONS: dict[str, tuple[str, type]] = {
    "Models and learners": ("models", ModelSpec),
    "Folds": ("cross_fitting", CrossFitting),
    "The correction step": ("targeting", Targeting),
    "Intervals and bands": ("inference", Inference),
    "Seeds, run labels and parallel jobs": ("runtime", Runtime),
}

#: ``###`` heading under "Method options" to the method class it documents.
METHOD_SECTIONS: dict[str, str] = {
    "Collaborative TMLE": "CollaborativeTMLEMethod",
    "DR-TMLE": "DRTMLEMethod",
    "Adaptive-propensity IPTW": "AdaptiveIPTWMethod",
    "Adaptive TMLE for a trial with external data": "AdaptiveTMLEMethod",
    "EP learner": "EPMethod",
    "Sequential IPW": "IPWMethod",
    "Parametric g-formula": "ParametricGFormulaMethod",
}

#: Every typed method. Those absent from :data:`METHOD_SECTIONS` must have no own options.
METHODS = (
    "TMLEMethod",
    "CollaborativeTMLEMethod",
    "DRTMLEMethod",
    "AdaptiveIPTWMethod",
    "AdaptiveTMLEMethod",
    "EPMethod",
    "SDRMethod",
    "IPSIMethod",
    "GComputationMethod",
    "IPWMethod",
    "ParametricGFormulaMethod",
)

GROUP_FIELDS = ("models", "cross_fitting", "targeting", "inference", "runtime")

CHANGES_SECTION = "Where a method changes a default"

#: A value other than the default for each group field, used to ask whether a translation
#: reads the field. ``EXTRA`` holds companion settings a value needs to be constructible.
SAMPLE: dict[str, dict[str, Any]] = {
    "models": {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "missingness_learner": LogisticRegression(),
        "intermediate_learner": LogisticRegression(),
        "pseudo_learner": LinearRegression(),
        "censoring_learner": LogisticRegression(),
        "competing_learner": LogisticRegression(),
        "density_bins": 10,
        "ratio": "density",
        "screen_treatment": True,
        "screen_threshold": 0.2,
        "min_retain": 2,
        "hazard_pooling": "pooled",
        "outcome_by_arm": True,
        "survival_model": "hazard",
    },
    "cross_fitting": {
        "enabled": False,
        "n_folds": 5,
        "learner_folds": 3,
        "repeats": 2,
        "stratify_by": "treatment",
        "targeting_scheme": "fold",
        "fold_evaluation": True,
        "split_plan": SplitPlan.from_folds([random_partition(50, 10, seed=1)]),
    },
    "targeting": {
        "fluctuation": "linear",
        "algorithm": "one_step",
        "g_bounds": 0.05,
        "q_bounds": (0.0, 1.0),
        "nuisance_bound": 0.02,
        "submodel_alpha": 0.99,
        "target_weights": True,
        "step_size": 0.01,
        "max_iter": 30,
        "tol": 1e-8,
        "stopping": "sample_size",
        "step_bound": 0.5,
        "step_search": False,
        "iterate": "best",
        "conditional_form": "known_score",
        "isotonic": True,
        "ratio_trim": 0.99,
        "node_bounds": {1: (0.0, 1.0)},
    },
    "inference": {
        "alpha": 0.1,
        "n_bootstrap": 50,
        "bootstrap_resampling": "iid",
        "simultaneous": False,
        "n_multiplier": 500,
        "multiplier_kind": "normal",
        "band_scale": "identity",
        "band_reference": "t",
    },
    "runtime": {"random_state": 3, "run_id": "label", "n_jobs": 2},
}
EXTRA: dict[tuple[str, str], dict[str, Any]] = {
    ("targeting", "step_search"): {"step_bound": 0.5},
    # Only "none" is accepted when folds are drawn; an in-sample fit carries the declaration.
    ("cross_fitting", "stratify_by"): {"enabled": False},
}

#: Group fields whose longitudinal translation a method resolves to its own default.
RESOLVED_KEYS = ("simultaneous", "survival_model", "max_iter")


# --- parsing -------------------------------------------------------------------------------


def _sections(text: str, level: str) -> dict[str, str]:
    """Each heading of ``level`` (``"##"`` or ``"###"``) to the text under it.

    A section ends at the next heading of the same level or a higher one.
    """
    depth = len(level)
    out: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        hashes = len(line) - len(line.lstrip("#"))
        if hashes and line[hashes : hashes + 1] == " " and hashes <= depth:
            if current is not None:
                out[current] = "\n".join(lines)
            current = line[hashes + 1 :].strip() if hashes == depth else None
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        out[current] = "\n".join(lines)
    return out


def _table(text: str) -> list[list[str]]:
    """The body rows of the first pipe table in ``text``, each a list of stripped cells."""
    rows: list[list[str]] = []
    started = False
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            if all(set(cell) <= {"-", " "} for cell in cells):
                started = True
                continue
            if started:
                rows.append(cells)
        elif started:
            break
    return rows


def _code(cell: str) -> str:
    """The text of a cell written as one backticked literal."""
    assert cell.startswith("`") and cell.endswith("`"), cell
    return cell[1:-1]


# --- what the code says -------------------------------------------------------------------


def _default(field: dataclasses.Field) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    raise AssertionError(f"{field.name} has no default")


def _translation(method: TMLEMethod, longitudinal: bool) -> dict[str, Any] | None:
    try:
        return method.estimator_kwargs(longitudinal=longitudinal)
    except cleverly.exceptions.MethodConfigurationError:
        return None


def _reads(group: str, field: str, longitudinal: bool) -> bool:
    """Whether the point or longitudinal translation carries a changed value of the field."""
    base = TMLEMethod()
    extra = EXTRA.get((group, field), {})
    reference = dataclasses.replace(
        base, **{group: dataclasses.replace(getattr(base, group), **extra)}
    )
    changed = dataclasses.replace(
        base,
        **{
            group: dataclasses.replace(
                getattr(base, group), **extra, **{field: SAMPLE[group][field]}
            )
        },
    )
    after = _translation(changed, longitudinal)
    before = _translation(reference, longitudinal)
    if after is None or before is None:
        return False
    return any(
        after.get(key) is not before.get(key) and repr(after.get(key)) != repr(before.get(key))
        for key in set(after) | set(before)
    )


def used_on(group: str, field: str) -> set[str]:
    kinds = set()
    if _reads(group, field, longitudinal=False):
        kinds.add("point")
    if _reads(group, field, longitudinal=True):
        kinds.add("over time")
    if (group, field) in _READ:
        kinds.add("exact times")
    if (group, field) in ATMLE_FIELDS:
        kinds.add("trial")
    return kinds


def _keyword(group: str, field: str) -> str:
    matches = [key for key, name in SHORTCUTS[group].items() if name == field]
    assert len(matches) == 1, (group, field, matches)
    return matches[0]


def expected_changes() -> set[tuple[str, str, str]]:
    """Every ``(setting, where, value)`` at which a method or design replaces a default."""
    out: set[tuple[str, str, str]] = set()
    base = TMLEMethod()
    for name in METHODS:
        method = getattr(cleverly, name)()
        for group in GROUP_FIELDS:
            for field in dataclasses.fields(getattr(method, group)):
                ours = getattr(getattr(method, group), field.name)
                if repr(ours) != repr(getattr(getattr(base, group), field.name)):
                    out.add((field.name, name, repr(ours)))
    base_over_time = base.estimator_kwargs(longitudinal=True)
    for name in ("CollaborativeTMLEMethod", "DRTMLEMethod"):
        over_time = getattr(cleverly, name)().estimator_kwargs(longitudinal=True)
        for key in RESOLVED_KEYS:
            if repr(over_time[key]) != repr(base_over_time[key]):
                out.add((key, f"{name} over time", repr(over_time[key])))
    if repr(base_over_time["g_bounds"]) != repr(base.targeting.g_bounds):
        out.add(("g_bounds", "over time", repr(base_over_time["g_bounds"])))
    exact = continuous_kwargs(base)
    for field in ("max_iter", "step_size", "tol"):
        if repr(exact[field]) != repr(getattr(base.targeting, field)):
            out.add((field, "exact times", repr(exact[field])))
    return out


# --- the checks ----------------------------------------------------------------------------


def group_problems(text: str) -> list[str]:
    problems = []
    sections = _sections(text, "##")
    for heading, (group, cls) in GROUP_SECTIONS.items():
        if heading not in sections:
            problems.append(f"no section {heading!r}")
            continue
        rows = {_code(row[0]): row for row in _table(sections[heading])}
        fields = {field.name: field for field in dataclasses.fields(cls)}
        for name in sorted(fields.keys() - rows.keys()):
            problems.append(f"{cls.__name__}.{name} is not on the page")
        for name in sorted(rows.keys() - fields.keys()):
            problems.append(f"{heading}: {name} is not a field of {cls.__name__}")
        for name in sorted(fields.keys() & rows.keys()):
            _setting, keyword, default, kinds, _what = rows[name]
            if _code(keyword) != _keyword(group, name):
                problems.append(f"{name}: keyword {keyword} is not {_keyword(group, name)!r}")
            if _code(default) != repr(_default(fields[name])):
                problems.append(f"{name}: default {default} is not {_default(fields[name])!r}")
            written = {kind.strip() for kind in kinds.split(",") if kind.strip()}
            derived = used_on(group, name)
            if written != derived:
                problems.append(f"{name}: used on {sorted(written)} is not {sorted(derived)}")
    return problems


def method_problems(text: str) -> list[str]:
    problems = []
    options = _sections(text, "##").get("Method options", "")
    sections = _sections(options, "###")
    for name in METHODS:
        cls = getattr(cleverly, name)
        own = {
            field.name: field
            for field in dataclasses.fields(cls)
            if field.name not in {*GROUP_FIELDS, "name"}
        }
        heading = next((h for h, n in METHOD_SECTIONS.items() if n == name), None)
        if heading is None:
            if own:
                problems.append(f"{name} has options {sorted(own)} and no section")
            elif f"`{name}`" not in options:
                problems.append(f"{name} is not named as a method with no options")
            continue
        if heading not in sections:
            problems.append(f"no section {heading!r}")
            continue
        rows = {_code(row[0]): row for row in _table(sections[heading])}
        for field in sorted(own.keys() - rows.keys()):
            problems.append(f"{name}.{field} is not on the page")
        for field in sorted(rows.keys() - own.keys()):
            problems.append(f"{heading}: {field} is not an option of {name}")
        for field in sorted(own.keys() & rows.keys()):
            if _code(rows[field][1]) != repr(_default(own[field])):
                problems.append(f"{name}.{field}: default {rows[field][1]}")
    return problems


def change_problems(text: str) -> list[str]:
    rows = _table(_sections(text, "##").get(CHANGES_SECTION, ""))
    written = set()
    for setting, where, value in rows:
        place = where.replace("`", "")
        written.add((_code(setting), place, _code(value)))
    derived = expected_changes()
    problems = [
        f"changed default missing from the page: {row}" for row in sorted(derived - written)
    ]
    problems += [
        f"page lists a change the code does not make: {row}" for row in sorted(written - derived)
    ]
    return problems


def test_group_tables_match_the_dataclasses():
    problems = group_problems(PAGE.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)


def test_method_tables_match_the_method_objects():
    problems = method_problems(PAGE.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)


def test_changed_defaults_match_the_translations():
    problems = change_problems(PAGE.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("check", "old", "new"),
    [
        (group_problems, "| `n_folds` | `n_folds` | `10` |", "| `n_folds` | `n_folds` | `9` |"),
        (group_problems, "| `enabled` | `cross_fit` |", "| `enabled` | `enabled` |"),
        (
            group_problems,
            "| `isotonic` | `isotonic` | `False` | over time |",
            "| `isotonic` | `isotonic` | `False` | point, over time |",
        ),
        (group_problems, "| `run_id` |", "| `run_label` |"),
        (method_problems, "| `patience` | `None` |", "| `patience` | `3` |"),
        (method_problems, "| `normalization` |", "| `weights` |"),
        (change_problems, "| `n_folds` | `AdaptiveTMLEMethod` | `5` |\n", ""),
    ],
)
def test_each_check_fails_on_a_doctored_page(check, old, new):
    """Deliberate-mutation controls: the doctored page must produce a finding."""
    text = PAGE.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    assert check(text.replace(old, new))
