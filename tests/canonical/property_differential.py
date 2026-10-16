"""The all-study property differential and its clause binding (RM39b2 rev 3 section 8, R3-2).

Each declared rule restates a verdict that a study's own ``summarize_properties`` computes.
The differential perturbs every committed study's property rows, summarizes them with the
study's own code, and requires the declared rules to give the same ``passed`` and
``property_passed`` on every verdict row.  ``longitudinal_sdr_properties`` runs on both its
committed rows and synthetic rows built from its declared cells: its committed rows make no
perturbation decide the ``P_D`` two-size clause alone, and its synthetic rows do.

| subject | rows |
| --- | --- |
| every registered study with a committed ``property-replicates.csv.gz`` | the committed rows |
| :data:`SYNTHETIC_MODULES` | :func:`synthetic_rows`, from the module's declared cells |

The perturbations are
:data:`~tests.studies.evidence.property_failures.PERTURBATIONS`, the graded kinds of
:func:`~tests.studies.evidence.property_failures.binding_perturbations`, and a scaling and a
shift per extra column and per cell
(:func:`~tests.studies.evidence.property_failures.extra_perturbations`).  Each summary runs with
the study record replaced by a copy whose ``bootstrap_replicates`` is
:data:`DIFFERENTIAL_BOOTSTRAP`.  The rule and the module read the same summary columns, so the
agreement does not depend on the bootstrap count.

The clause binding requires that each clause which applies on a subject decides a verdict alone
under some perturbation of *that* subject.  A rule that dropped such a clause would then
disagree with the study, so the differential binds it.  Many clauses read a per-module
threshold, such as an efficiency band, so a witness in one study says nothing about the
threshold of another.  :data:`BINDINGS` names one perturbation for each subject and each
``(property, clause)`` it uses.  A pair that its subject cannot decide alone has a row in
:data:`REASONS` instead, with a reason in :data:`REASON_CHECKS` that this module verifies.
``--bindings`` searches the subjects, and ``--write`` rewrites the witness table.

The RM39b2 remediation measured the pass at 5,797 cases, about 6,400 CPU-seconds and about 30 minutes on four workers, against a
fast-tier budget of about 120 seconds.  So it runs as the ``property-differential`` job of
``.github/workflows/ci.yml`` on every pull request that touches ``tests/studies/``,
``tests/canonical/``, ``src/cleverly/``, ``pyproject.toml``, ``uv.lock`` or the workflow.  Its file name is not a test file name, so the fast tier does not collect
it.  ``tests/unit/test_method_evidence.py`` checks that :data:`SUBJECTS` lists every committed
study, and that :data:`BINDINGS` names only subjects, perturbations and clauses that exist.

Usage::

    pytest -q -n auto --dist loadgroup tests/canonical/property_differential.py
    python -m tests.canonical.property_differential --bindings DIR --subjects SLUG ...
    python -m tests.canonical.property_differential --write DIR
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import importlib
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.studies.evidence import property_failures
from tests.studies.evidence.registry import StudyRecord, registered

#: The bootstrap count of every differential summary (rev 3 R3-2).  At most the fast tier's
#: ``CONTROL_BOOTSTRAP`` of 2,000.
DIFFERENTIAL_BOOTSTRAP = 200

#: The property modules that run on :func:`synthetic_rows`: ``longitudinal_sdr_properties``,
#: whose committed rows leave the clause ``P_D large-size bias bound under the small`` without a
#: binding witness.
SYNTHETIC_MODULES = ("tests.studies.longitudinal_sdr_properties",)

#: The seed root of :func:`synthetic_rows`, and a control cell's bias in standard errors.
SYNTHETIC_SEED = 20_261_016
SYNTHETIC_CONTROL_BIAS = 5.0

#: The clause-binding witnesses: for each subject and each ``(property, clause)`` it uses, one
#: perturbation of that subject under which the clause alone decides a verdict.
BINDINGS = Path(__file__).with_name("clause-bindings.csv")
BINDING_COLUMNS = ("subject", "property", "clause", "perturbation")

#: The pairs that their subject cannot decide alone, each with a reason of :data:`REASON_CHECKS`.
REASONS = Path(__file__).with_name("clause-binding-reasons.csv")
REASON_COLUMNS = ("subject", "property", "clause", "reason")

_STUDIES = {record.slug: record for record in registered()}

#: Every subject of the differential: each committed study's slug, then each synthetic module.
SUBJECTS = (
    *(
        slug
        for slug, record in _STUDIES.items()
        if record.artifact("property-replicates.csv.gz").exists()
    ),
    *SYNTHETIC_MODULES,
)

#: The perturbations every subject runs, before its extra-column kinds.
KINDS = (*property_failures.PERTURBATIONS, *property_failures.binding_perturbations())


def subject_of(subject: str) -> tuple[ModuleType, StudyRecord]:
    """The property module and the study record of a subject.

    Parameters
    ----------
    subject : str
        A slug of :data:`SUBJECTS`, or a module of :data:`SYNTHETIC_MODULES`.

    Returns
    -------
    tuple
        The module and its record.
    """
    if subject in SYNTHETIC_MODULES:
        module = importlib.import_module(subject)
        return module, module.STUDY
    record = _STUDIES[subject]
    return record.properties(), record


@contextlib.contextmanager
def reduced(record: StudyRecord) -> Iterator[StudyRecord]:
    """Replace the record by a copy with :data:`DIFFERENTIAL_BOOTSTRAP` bootstrap replicates.

    Every attribute of a loaded ``tests.studies`` module that *is* the record is replaced, so a
    shared helper that imported the record reads the copy too.  The seeds do not read the
    margins (:func:`~tests.studies.evidence.seeds.stream_seed`), so every draw keeps its seed.

    Parameters
    ----------
    record : StudyRecord
        The study record.

    Yields
    ------
    StudyRecord
        The reduced copy.
    """
    margins = dataclasses.replace(record.margins, bootstrap_replicates=DIFFERENTIAL_BOOTSTRAP)
    cheap = dataclasses.replace(record, margins=margins)
    patched = [
        (module, name)
        for key, module in list(sys.modules.items())
        if key.startswith("tests.studies") and module is not None
        for name, value in list(vars(module).items())
        if value is record
    ]
    for module, name in patched:
        setattr(module, name, cheap)
    try:
        yield cheap
    finally:
        for module, name in patched:
            setattr(module, name, record)


def synthetic_rows(module: ModuleType) -> pd.DataFrame:
    """Rows built from a module's declared cells, through its own ``generate_property_rows``.

    The module's ``fit_properties`` is replaced by a function that returns, for each key, one
    row per cell the key fills, at the declared budget.  Each row's truth is zero and its
    estimate a seeded normal draw with standard error ``n ** -0.5``, centred on zero in a
    positive cell and on :data:`SYNTHETIC_CONTROL_BIAS` standard errors in a control.  The
    module's own code then derives its calibration controls and orders the rows, so the rows
    have the shape a run publishes.  They are no run's output: the differential reads only
    whether a rule and the module agree on them.

    Parameters
    ----------
    module : module
        A property module that declares ``declared_cells`` and fits through ``fit_properties``.

    Returns
    -------
    pandas.DataFrame
        The rows.
    """
    declared = {(cell.property, cell.cell): cell for cell in module.declared_cells()}
    critical = float(norm.ppf(1.0 - module.STUDY.margins.alpha / 2.0))

    def fit_properties(
        keys: Any, fit_one: Any, *, cells_of: Any, replicate_of: Any, **options: Any
    ) -> list[list[dict[str, Any]]]:
        controls = {name for names in (options.get("derived") or {}).values() for name in names}
        counts: dict[str, int] = {}
        for key in keys:
            for name in cells_of(key):
                counts[name] = counts.get(name, 0) + 1
        out = []
        for key in keys:
            replicate = replicate_of(key)
            rows = []
            for name in cells_of(key):
                if name in controls:
                    continue
                family, cell = name.split("/", 1)
                spec = declared[(family, cell)]
                rng = np.random.default_rng(
                    [SYNTHETIC_SEED, property_failures.cell_key(family, cell), replicate]
                )
                scale = float(spec.n) ** -0.5
                centre = SYNTHETIC_CONTROL_BIAS * scale if spec.role == "control" else 0.0
                estimate = rng.normal(centre, scale)
                std_error = scale * rng.uniform(0.9, 1.1)
                rows.append(
                    {
                        "property": family,
                        "cell": cell,
                        "role": spec.role,
                        "replicate": replicate,
                        "n": spec.n,
                        "requested_replicates": counts[name],
                        "failed_replicates": 0,
                        "truth": 0.0,
                        "estimate": estimate,
                        "std_error": std_error,
                        "covered": int(abs(estimate) <= critical * std_error),
                        "rejected": int(abs(estimate) > critical * std_error),
                    }
                )
            out.append(rows)
        return out

    original = getattr(module, "fit_properties")  # noqa: B009
    setattr(module, "fit_properties", fit_properties)  # noqa: B010
    try:
        rows: pd.DataFrame = module.generate_property_rows(n_jobs=1)
    finally:
        setattr(module, "fit_properties", original)  # noqa: B010
    return rows


@functools.cache
def subject_rows(subject: str) -> pd.DataFrame:
    """A subject's property rows, read or built once per process.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.

    Returns
    -------
    pandas.DataFrame
        The committed rows, or :func:`synthetic_rows`.
    """
    module, record = subject_of(subject)
    if subject in SYNTHETIC_MODULES:
        with reduced(record):
            return synthetic_rows(module)
    return pd.read_csv(record.artifact("property-replicates.csv.gz"))


def kinds(subject: str) -> tuple[str, ...]:
    """Every perturbation a subject runs: :data:`KINDS`, then its extra-column kinds.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.

    Returns
    -------
    tuple of str
        The :func:`~tests.studies.evidence.property_failures.perturbed` kinds.
    """
    return (*KINDS, *property_failures.extra_perturbations(subject_rows(subject)))


@dataclasses.dataclass(frozen=True)
class Judgement:
    """One perturbation of one subject, read by the study and by the rules.

    Parameters
    ----------
    wrong : pandas.DataFrame
        The verdict rows where the rules and the study disagree.
    used : frozenset of tuple
        The ``(property, clause)`` pairs that apply.
    decisive : frozenset of tuple
        The pairs that alone decide a verdict.
    moved : bool
        Whether some verdict is red, so the perturbation reached a threshold.
    """

    wrong: pd.DataFrame
    used: frozenset[tuple[str, str]]
    decisive: frozenset[tuple[str, str]]
    moved: bool


@functools.cache
def judge(subject: str, kind: str) -> Judgement:
    """Summarize one perturbation with the study's code and read it with the rules.

    Cached per process, so a witness or a binding check that a worker already ran as a
    differential case costs nothing more.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.
    kind : str
        A :func:`kinds` entry.

    Returns
    -------
    Judgement
        The disagreements and the clause binding.
    """
    module, record = subject_of(subject)
    rows = subject_rows(subject)
    with reduced(record) as cheap:
        moved = property_failures.perturbed(rows, kind, alpha=cheap.margins.alpha)
        summary = module.summarize_properties(moved)
        ctx = property_failures.module_context(cheap, moved, summary, {}, module)
        wrong = property_failures.differential(summary, ctx, module=module)
        used, decisive = property_failures.clause_binding(summary, ctx, module=module)
    red = ~(summary["passed"].astype(bool) & summary["property_passed"].astype(bool))
    return Judgement(wrong, frozenset(used), frozenset(decisive), bool(red.any()))


def read_bindings(path: Path = BINDINGS) -> pd.DataFrame:
    """The committed clause-binding witnesses.

    Parameters
    ----------
    path : pathlib.Path
        The table.

    Returns
    -------
    pandas.DataFrame
        One row per subject and ``(property, clause)``, with :data:`BINDING_COLUMNS`.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return table.loc[:, list(BINDING_COLUMNS)]


def read_reasons(path: Path = REASONS) -> pd.DataFrame:
    """The pairs that their subject cannot decide alone, and why.

    Parameters
    ----------
    path : pathlib.Path
        The table.

    Returns
    -------
    pandas.DataFrame
        One row per subject and ``(property, clause)``, with :data:`REASON_COLUMNS`.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return table.loc[:, list(REASON_COLUMNS)]


def _pairs(table: pd.DataFrame, subject: str) -> set[tuple[str, str]]:
    chosen = table.loc[table["subject"] == subject, ["property", "clause"]]
    return set(chosen.itertuples(index=False, name=None))


def fixed_at_size(subject: str, family: str, clause_name: str, argument: str = "") -> list[str]:
    """Check that a proportion clause gives one verdict at every count its rows can reach.

    An S2 or S3 clause reads a Clopper-Pearson interval of ``j`` successes in ``R``
    replicates.  No perturbation changes ``R``.  So when the clause gives the same verdict
    for every ``j`` from 0 to ``R`` on every row it applies to, no perturbation can make it
    decide a verdict, and a rule that dropped it agrees with the study on every input of
    that size.  The root-n control's ``coverage resolves`` is such a clause at ``R`` of 800
    or more: the 99 per cent interval is then narrower than the gap between the coverage
    floor and the nominal level, so it lies clear of one of them at every count.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.
    family : str
        The property.
    clause_name : str
        The clause, an S2 or S3 proportion clause.
    argument : str
        Unused.

    Returns
    -------
    list of str
        The rows where the verdict changes with the count, as ``cell: R``.  Empty when the
        reason holds.
    """
    return [
        f"{cell}: {size}"
        for _, cell, size, counts in counts_that_move(subject, family, clause_name)
    ]


def counts_that_move(
    subject: str, family: str, clause_name: str
) -> list[tuple[str, str, int, list[int]]]:
    """Each row of a proportion clause whose verdict depends on its success count.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.
    family : str
        The property.
    clause_name : str
        An S2 or S3 clause.

    Returns
    -------
    list of tuple
        ``(role, cell, R, counts)``, where ``counts`` are the success counts at which the
        clause gives the verdict it does not give on the committed rows.
    """
    module, record = subject_of(subject)
    rows = subject_rows(subject)
    with reduced(record) as cheap:
        summary = module.summarize_properties(rows)
        ctx = property_failures.module_context(cheap, rows, summary, {}, module)
        rule = property_failures.rule_for(family, module)
        assert rule is not None, (subject, family)
        clause = rule.clause(clause_name)
        prefix = {"S2": "coverage", "S3": "rejection"}[clause.statistic]
        level = ctx.margins.confidence_level
        moving: list[tuple[str, str, int, list[int]]] = []
        applying = summary.loc[summary["property"] == family]
        assert not applying.empty, (subject, family)
        for _, row in applying.iterrows():
            if not clause.applies(ctx, row):
                continue
            size = int(row["replicates"])
            committed = clause.read(ctx, row).passed
            counts = []
            for count in range(size + 1):
                interval = property_failures.clopper_pearson(count, size, confidence_level=level)
                moved = row.copy()
                moved[f"{prefix}_ci_lower"] = interval.low
                moved[f"{prefix}_ci_upper"] = interval.high
                if clause.read(ctx, moved).passed != committed:
                    counts.append(count)
            if counts:
                moving.append((str(row["role"]), str(row["cell"]), size, counts))
    return moving


def same_module(subject: str, family: str, clause_name: str, argument: str) -> list[str]:
    """Check that another subject of the same property module binds the pair.

    A clause reads the thresholds of its module and of its study record.  Two subjects that
    run one module with one record share every threshold, and differ only in their rows.  So
    a witness of the other subject binds the thresholds this subject reads.  The committed and
    the synthetic rows of ``longitudinal_sdr_properties`` are such a pair.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.
    family : str
        The property.
    clause_name : str
        The clause.
    argument : str
        The other subject.

    Returns
    -------
    list of str
        The failed conditions.  Empty when the reason holds.
    """
    problems: list[str] = []
    module, record = subject_of(subject)
    other_module, other_record = subject_of(argument)
    if other_module is not module or other_record is not record:
        problems.append(f"{argument} runs another module or record")
    witnesses = read_bindings()
    named = witnesses.loc[
        (witnesses["subject"] == argument)
        & (witnesses["property"] == family)
        & (witnesses["clause"] == clause_name)
    ]
    if named.empty:
        problems.append(f"{argument} has no witness of the pair")
    for kind in named["perturbation"]:
        if (family, clause_name) not in judge(argument, kind).decisive:
            problems.append(f"{argument} {kind} does not decide the pair")
    return problems


#: The mean-error factors that :func:`implied_by` sweeps, from a sign flip to three times.
IMPLIED_SWEEP = tuple(float(value) for value in np.linspace(-3.0, 3.0, 41))


def implied_by(subject: str, family: str, clause_name: str, argument: str) -> list[str]:
    """Check that another clause of the same rows fails wherever this one fails.

    Two bias clauses of one row can read the same centre against two thresholds, such as a
    discrimination margin and a both-wrong floor, or a margin from the truth and an
    equivalence to an alternative truth.  When the other clause's threshold is the stricter
    one in this module, this clause never fails alone, so no perturbation makes it decide.
    The check sweeps each applying role's mean error over :data:`IMPLIED_SWEEP` with
    ``centre/<role>/<c>``, and its spread with ``spread_role/<role>/<g>``.  It reads every
    :data:`KINDS` perturbation as well.  A row where
    this clause fails and the other passes refutes the reason.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.
    family : str
        The property.
    clause_name : str
        The clause.
    argument : str
        The other clause, of the same family.

    Returns
    -------
    list of str
        The refuting rows, as ``kind cell``.  Empty when the reason holds.
    """
    module, record = subject_of(subject)
    rows = subject_rows(subject)
    rule = property_failures.rule_for(family, module)
    assert rule is not None, (subject, family)
    with reduced(record) as cheap:
        summary = module.summarize_properties(rows)
        ctx = property_failures.module_context(cheap, rows, summary, {}, module)
        clause, other = rule.clause(clause_name), rule.clause(argument)
        applying = summary.loc[summary["property"] == family]
        roles = sorted(
            {str(row["role"]) for _, row in applying.iterrows() if clause.applies(ctx, row)}
        )
    problems: list[str] = []
    sweep = [
        *(f"centre/{role}/{c:g}" for role in roles for c in IMPLIED_SWEEP),
        *(f"spread_role/{role}/{g:g}" for role in roles for g in property_failures.ROLE_SPREADS),
    ]
    for kind in (*KINDS, *sweep):
        with reduced(record) as cheap:
            moved = property_failures.perturbed(rows, kind, alpha=cheap.margins.alpha)
            summary = module.summarize_properties(moved)
            ctx = property_failures.module_context(cheap, moved, summary, {}, module)
            for _, row in summary.loc[summary["property"] == family].iterrows():
                if not clause.applies(ctx, row):
                    continue
                if not other.applies(ctx, row):
                    problems.append(f"{kind} {row['cell']}: {argument} does not apply")
                elif not clause.read(ctx, row).passed and other.read(ctx, row).passed:
                    problems.append(f"{kind} {row['cell']}")
    return problems


#: Each reason a pair can carry in :data:`REASONS`, as ``code`` or ``code: argument``, and the
#: function that verifies it.
REASON_CHECKS = {
    "fixed at size": fixed_at_size,
    "same module as": same_module,
    "implied by": implied_by,
}


def candidates_at_no_loss(subject: str) -> tuple[int, list[tuple[Any, ...]]]:
    """Read each clause's candidates on the complete case, where no replicate is lost.

    With ``k = 0`` every candidate set holds the complete case alone, so each candidate must
    give the complete-case value and verdict (F18).  A candidate that reads another scale or
    another set of rows than the summary, such as log-scale SEs under a ratio cell, differs
    here.  The check covers every statistic type of every clause a subject uses.

    Parameters
    ----------
    subject : str
        A :data:`SUBJECTS` entry.

    Returns
    -------
    tuple
        The number of candidates read, and each one that differs as
        ``(property, cell, clause, statistic, label, complete case, candidate)``.
    """
    module, record = subject_of(subject)
    rows = subject_rows(subject)
    wrong: list[tuple[Any, ...]] = []
    read = 0
    with reduced(record) as cheap:
        summary = module.summarize_properties(rows)
        ctx = property_failures.module_context(cheap, rows, summary, {}, module)
        for family in sorted(set(summary["property"].astype(str))):
            rule = property_failures.rule_for(family, module)
            if rule is None:
                continue
            for cell, judged in property_failures._judged(rule, family, summary, ctx).items():
                selected = (summary["property"] == family) & (summary["cell"] == cell)
                row = summary.loc[selected].iloc[0]
                for clause, value in judged:
                    if clause.candidates is None:
                        continue
                    for label, candidate in clause.candidates(ctx, row):
                        read += 1
                        same = candidate.passed == value.passed and math.isclose(
                            candidate.slack, value.slack, rel_tol=1e-9, abs_tol=1e-12
                        )
                        if not same:
                            name = (family, cell, clause.name, clause.statistic, label)
                            wrong.append((*name, value.slack, candidate.slack))
    return read, wrong


# ---- the CI job ------------------------------------------------------------------------------


def _by_subject(subject: str, *values: Any) -> Any:
    """One case, grouped on its subject's worker so that the subject's judgements are reused."""
    return pytest.param(subject, *values, marks=pytest.mark.xdist_group(subject))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("subject", [_by_subject(subject) for subject in SUBJECTS])
def test_the_rules_agree_with_the_study_on_each_perturbation(subject: str, kind: str) -> None:
    """F10: a rule missing a conjunct, or reading a moved threshold, disagrees somewhere."""
    judged = judge(subject, kind)
    assert judged.wrong.empty, judged.wrong.to_string()


@pytest.mark.parametrize("subject", [_by_subject(subject) for subject in SUBJECTS])
def test_the_rules_agree_with_the_study_on_each_extra_column(subject: str) -> None:
    """L10: a scaling and a shift of each extra column in each cell, such as a bootstrap count."""
    extra = property_failures.extra_perturbations(subject_rows(subject))
    wrong = {kind: judge(subject, kind).wrong for kind in extra}
    disagreeing = {kind: frame.to_string() for kind, frame in wrong.items() if not frame.empty}
    assert disagreeing == {}


@pytest.mark.parametrize("subject", [_by_subject(subject) for subject in SUBJECTS])
def test_the_perturbations_reach_a_threshold(subject: str) -> None:
    """A perturbation set that moved no verdict of a subject would test no clause of it."""
    if subject_rows(subject).empty:
        _, record = subject_of(subject)
        published = pd.read_csv(record.artifact("properties.csv"))
        assert published.empty, f"{subject} publishes verdicts without property rows"
        return
    assert judge(subject, "se_low").moved or judge(subject, "bias").moved


@pytest.mark.parametrize("subject", [_by_subject(subject) for subject in SUBJECTS])
def test_each_clause_a_subject_uses_has_its_own_binding_witness(subject: str) -> None:
    """Rev 3 section 8, per subject: a witness of the subject, or a verified reason."""
    if subject_rows(subject).empty:
        return
    covered = _pairs(read_bindings(), subject) | _pairs(read_reasons(), subject)
    used = judge(subject, "none").used
    assert sorted(used - covered) == [], "unbound clauses: no witness of this subject"


@pytest.mark.parametrize("subject", [_by_subject(subject) for subject in SUBJECTS])
def test_each_candidate_at_no_loss_reproduces_the_complete_case(subject: str) -> None:
    """F18 for every statistic type: at ``k = 0`` each candidate is the complete case."""
    read, wrong = candidates_at_no_loss(subject)
    assert wrong == []
    if not subject_rows(subject).empty and judge(subject, "none").used:
        assert read > 0


def _rows(path: Path, columns: tuple[str, ...]) -> list[Any]:
    if not path.exists():
        return [None]
    table = pd.read_csv(path, dtype=str, keep_default_na=False).loc[:, list(columns)]
    return [
        pytest.param(row, marks=pytest.mark.xdist_group(row[0]))
        for row in table.itertuples(index=False, name=None)
    ]


@pytest.mark.parametrize(
    "witness",
    _rows(BINDINGS, BINDING_COLUMNS),
    ids=lambda witness: "missing" if witness is None else "/".join(witness[:3]),
)
def test_each_binding_witness_decides_its_clause(witness: tuple[str, ...] | None) -> None:
    """The witness makes the clause decide a verdict alone, and the study agrees there."""
    assert witness is not None, f"{BINDINGS} does not exist"
    subject, family, clause, kind = witness
    judged = judge(subject, kind)
    assert (family, clause) in judged.decisive
    assert judged.wrong.empty, judged.wrong.to_string()


@pytest.mark.parametrize(
    "reason",
    _rows(REASONS, REASON_COLUMNS),
    ids=lambda reason: "missing" if reason is None else "/".join(reason[:3]),
)
def test_each_stated_reason_holds(reason: tuple[str, ...] | None) -> None:
    """A pair without a witness of its subject carries a reason, and the reason is checked."""
    assert reason is not None, f"{REASONS} does not exist"
    subject, family, clause, why = reason
    code, _, argument = why.partition(": ")
    assert (family, clause) not in judge(subject, "none").decisive
    assert REASON_CHECKS[code](subject, family, clause, argument) == []


# ---- the binding search ------------------------------------------------------------------------

#: The statistic types that the targeted kinds move: an SE ratio (S4, S5), a spread ratio (S9)
#: and an RMSE ratio (S13).  A proportion or a bias reads ``covered``, ``rejected`` or the
#: centre, which the two-point kinds keep.
TARGETED_STATISTICS = frozenset({"S4", "S5", "S9", "S13"})


def _targetable(subject: str, pair: tuple[str, str]) -> bool:
    module, _ = subject_of(subject)
    rule = property_failures.rule_for(pair[0], module)
    return rule is not None and rule.clause(pair[1]).statistic in TARGETED_STATISTICS


def _count_kinds(subject: str, pairs: set[tuple[str, str]]) -> list[str]:
    """``coverage_role`` kinds at the counts where a proportion clause changes its verdict.

    A coverage share of the graded grid can miss a narrow window of counts, such as the
    counts at which a root-n control's interval reaches both the floor and nominal.  This
    reads the window off :func:`counts_that_move` and sets the role's coverage inside it.
    """
    module, _ = subject_of(subject)
    out: list[str] = []
    for family, clause_name in sorted(pairs):
        rule = property_failures.rule_for(family, module)
        if rule is None or rule.clause(clause_name).statistic != "S2":
            continue
        for role, _, size, counts in counts_that_move(subject, family, clause_name):
            middle = counts[len(counts) // 2]
            for count in dict.fromkeys((middle, counts[0], counts[-1])):
                out.append(f"coverage_role/{role}/{count / size:.12g}")
    return out


def _search(subject: str, out: Path) -> None:
    started = time.perf_counter()
    found: dict[str, str] = {}
    used: set[tuple[str, str]] = set()
    # A search that stopped part way resumes from its partial file: each kind it judged keeps
    # its result, so a rerun judges only the kinds after it.
    partial = out / f"{subject}.partial.json"
    done: dict[str, Any] = json.loads(partial.read_text()) if partial.exists() else {}

    def run(kind: str) -> None:
        if kind in done:
            pairs, decisive = done[kind]
        else:
            judged = judge(subject, kind)
            if not judged.wrong.empty:
                raise AssertionError(f"{subject} {kind}: {judged.wrong.to_string()}")
            pairs = sorted(judged.used)
            decisive = sorted(judged.decisive)
            done[kind] = [pairs, decisive]
            partial.write_text(json.dumps(done))
        used.update(tuple(pair) for pair in pairs)
        for family, clause in decisive:
            found.setdefault(f"{family}\x1f{clause}", kind)

    for kind in kinds(subject):
        run(kind)
    reasoned = _pairs(read_reasons(), subject) if REASONS.exists() else set()

    def missing() -> set[tuple[str, str]]:
        return {pair for pair in used - reasoned if f"{pair[0]}\x1f{pair[1]}" not in found}

    targeted = 0
    for kind in property_failures.targeted_perturbations():
        if not any(_targetable(subject, pair) for pair in missing()):
            break
        run(kind)
        targeted += 1
    for kind in property_failures.composed_perturbations():
        if not missing():
            break
        run(kind)
        targeted += 1
    for kind in _count_kinds(subject, missing()):
        if not missing():
            break
        run(kind)
        targeted += 1
    elapsed = time.perf_counter() - started
    record = {
        "subject": subject,
        "seconds": elapsed,
        "targeted": targeted,
        "used": sorted(f"{family}\x1f{clause}" for family, clause in used),
        "found": found,
    }
    (out / f"{subject}.json").write_text(json.dumps(record, indent=1))
    partial.unlink(missing_ok=True)
    print(
        f"{subject}: {elapsed:.1f}s, {len(found)} of {len(used)} clauses bound, "
        f"{targeted} targeted kinds",
        flush=True,
    )


def _write(directory: Path) -> int:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    missing = sorted(set(SUBJECTS) - {record["subject"] for record in records})
    if missing:
        print(f"no search result for {missing}", file=sys.stderr)
        return 1
    reasons = read_reasons() if REASONS.exists() else pd.DataFrame(columns=list(REASON_COLUMNS))
    reasoned = {
        (subject, f"{family}\x1f{clause}")
        for subject, family, clause, _ in reasons.itertuples(index=False, name=None)
    }
    rows: list[list[str]] = []
    unbound: list[str] = []
    for record in sorted(records, key=lambda record: record["subject"]):
        subject = record["subject"]
        for pair in record["used"]:
            if pair in record["found"]:
                rows.append([subject, *pair.split("\x1f"), record["found"][pair]])
            elif (subject, pair) not in reasoned:
                unbound.append(f"{subject}: {pair.replace(chr(31), ' | ')}")
    pd.DataFrame(rows, columns=list(BINDING_COLUMNS)).to_csv(BINDINGS, index=False)
    print(f"{len(rows)} witnesses written to {BINDINGS}")
    if unbound:
        print("unbound clauses:", *unbound, sep="\n  ", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Search the subjects for clause-binding witnesses, or write the table from a search.

    Parameters
    ----------
    argv : list of str or None
        The command-line arguments.

    Returns
    -------
    int
        Zero when every used clause has a witness.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bindings", type=Path, help="search and write one JSON per subject")
    parser.add_argument("--subjects", nargs="*", default=None, help="the subjects to search")
    parser.add_argument("--write", type=Path, help="write clause-bindings.csv from a search")
    arguments = parser.parse_args(argv)
    if arguments.write is not None:
        return _write(arguments.write)
    if arguments.bindings is None:
        parser.error("give --bindings DIR or --write DIR")
    arguments.bindings.mkdir(parents=True, exist_ok=True)
    for subject in arguments.subjects or SUBJECTS:
        _search(subject, arguments.bindings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
