"""The all-study property differential and its clause binding (RM39b2 rev 3 section 8, R3-2).

Each declared rule restates a verdict that a study's own ``summarize_properties`` computes.
The differential perturbs every committed study's property rows, summarizes them with the
study's own code, and requires the declared rules to give the same ``passed`` and
``property_passed`` on every verdict row.  Three modules have no committed rows, so they run
on synthetic rows built from their declared cells.

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
under some perturbation of some subject.  A rule that dropped such a clause would then disagree
with the study, so the differential binds it.  :data:`BINDINGS` names one witness, a subject
and a perturbation, for each clause.  This module checks each witness, and it checks that each
clause a subject uses has a witness.  ``--bindings`` searches the subjects and rewrites the
table.

Commit 7e measured the pass at 4,700 CPU-seconds, about 20 minutes on four workers, against
a fast-tier budget of about 120 seconds.  So it runs as the ``property-differential`` job of
``.github/workflows/ci.yml`` on every pull request that touches ``tests/studies/`` or
``tests/canonical/``.  Its file name is not a test file name, so the fast tier does not collect
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

#: The property modules with no committed rows, which run on :func:`synthetic_rows`.
SYNTHETIC_MODULES = (
    "tests.studies.longitudinal_sdr_properties",
    "tests.studies.longitudinal_incremental_properties",
    "tests.studies.longitudinal_incremental_dropout_properties",
)

#: The seed root of :func:`synthetic_rows`, and a control cell's bias in standard errors.
SYNTHETIC_SEED = 20_261_016
SYNTHETIC_CONTROL_BIAS = 5.0

#: The clause-binding witnesses: for each ``(property, clause)``, one subject and one
#: perturbation under which the clause alone decides a verdict.
BINDINGS = Path(__file__).with_name("clause-bindings.csv")
BINDING_COLUMNS = ("property", "clause", "subject", "perturbation")

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
        ctx = property_failures.Context(
            cheap, moved, summary, {}, property_failures.module_groups(module)
        )
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
        One row per ``(property, clause)``, with :data:`BINDING_COLUMNS`.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return table.loc[:, list(BINDING_COLUMNS)]


# ---- the CI job ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("subject", SUBJECTS)
def test_the_rules_agree_with_the_study_on_each_perturbation(subject: str, kind: str) -> None:
    """F10: a rule missing a conjunct, or reading a moved threshold, disagrees somewhere."""
    judged = judge(subject, kind)
    assert judged.wrong.empty, judged.wrong.to_string()


@pytest.mark.parametrize("subject", SUBJECTS)
def test_the_rules_agree_with_the_study_on_each_extra_column(subject: str) -> None:
    """L10: a scaling and a shift of each extra column in each cell, such as a bootstrap count."""
    extra = property_failures.extra_perturbations(subject_rows(subject))
    wrong = {kind: judge(subject, kind).wrong for kind in extra}
    disagreeing = {kind: frame.to_string() for kind, frame in wrong.items() if not frame.empty}
    assert disagreeing == {}


@pytest.mark.parametrize("subject", SUBJECTS)
def test_the_perturbations_reach_a_threshold(subject: str) -> None:
    """A perturbation set that moved no verdict of a subject would test no clause of it."""
    if subject_rows(subject).empty:
        _, record = subject_of(subject)
        published = pd.read_csv(record.artifact("properties.csv"))
        assert published.empty, f"{subject} publishes verdicts without property rows"
        return
    assert judge(subject, "se_low").moved or judge(subject, "bias").moved


@pytest.mark.parametrize("subject", SUBJECTS)
def test_each_clause_a_subject_uses_has_a_binding_witness(subject: str) -> None:
    """Rev 3 section 8: a clause no perturbation can make decisive is reported as unbound."""
    if subject_rows(subject).empty:
        return
    bound = set(read_bindings().loc[:, ["property", "clause"]].itertuples(index=False, name=None))
    used = judge(subject, "none").used
    assert sorted(used - bound) == [], "unbound clauses: no witness in clause-bindings.csv"


def _witnesses() -> list[tuple[str, str, str, str] | None]:
    if not BINDINGS.exists():
        return [None]
    return list(read_bindings().itertuples(index=False, name=None))


@pytest.mark.parametrize(
    "witness",
    _witnesses(),
    ids=lambda witness: "missing" if witness is None else "/".join(witness[:2]),
)
def test_each_binding_witness_decides_its_clause(
    witness: tuple[str, str, str, str] | None,
) -> None:
    """The witness makes the clause decide a verdict alone, and the study agrees there."""
    assert witness is not None, f"{BINDINGS} does not exist"
    family, clause, subject, kind = witness
    judged = judge(subject, kind)
    assert (family, clause) in judged.decisive
    assert judged.wrong.empty, judged.wrong.to_string()


# ---- the binding search ------------------------------------------------------------------------


def _search(subject: str, out: Path) -> None:
    started = time.perf_counter()
    found: dict[str, str] = {}
    used: set[tuple[str, str]] = set()
    for kind in kinds(subject):
        judged = judge(subject, kind)
        if not judged.wrong.empty:
            raise AssertionError(f"{subject} {kind}: {judged.wrong.to_string()}")
        used |= judged.used
        for family, clause in sorted(judged.decisive):
            found.setdefault(f"{family}\x1f{clause}", kind)
    elapsed = time.perf_counter() - started
    record = {
        "subject": subject,
        "seconds": elapsed,
        "used": sorted(f"{family}\x1f{clause}" for family, clause in used),
        "found": found,
    }
    (out / f"{subject}.json").write_text(json.dumps(record, indent=1))
    print(f"{subject}: {elapsed:.1f}s, {len(found)} of {len(used)} clauses bound", flush=True)


def _write(directory: Path) -> int:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    missing = sorted(set(SUBJECTS) - {record["subject"] for record in records})
    if missing:
        print(f"no search result for {missing}", file=sys.stderr)
        return 1
    per_kind = {kind: index for index, kind in enumerate(KINDS)}
    witnesses: dict[str, tuple[float, int, str, str]] = {}
    used: set[str] = set()
    for record in records:
        used |= set(record["used"])
        for pair, kind in record["found"].items():
            # The cheapest subject first, then the earliest kind: a named perturbation is read
            # before a graded one, and a graded one before an extra column.
            rank = (record["seconds"], per_kind.get(kind, len(KINDS)), record["subject"], kind)
            if pair not in witnesses or rank < witnesses[pair]:
                witnesses[pair] = rank
    unbound = sorted(used - set(witnesses))
    rows = [
        [*pair.split("\x1f"), subject, kind]
        for pair, (_, _, subject, kind) in sorted(witnesses.items())
        if pair in used
    ]
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
