"""The red-cell ledger: every red verdict a registered study publishes, and the ask that owns it.

A ``reporting`` study publishes its red verdicts instead of refusing to publish, which is the
whole point of the policy -- and also why a red cell can sit on a study page for months with no
record of who is meant to close it.  RM18's prose named most of them and missed one outright:
the paired ``ate_shift[+0.25 vs natural course]`` row of ``shift-policies`` was red and no
roadmap row mentioned it.  A hand-kept list drifts the same way every hand-typed number does.

So the ledger is generated.  :func:`red_rows` reads each study's committed verdict tables,
:data:`OWNERS` declares which roadmap ask owns each red row, and :func:`findings` states every
way the two can disagree.  ``tests/unit/test_red_cell_ledger.py`` asserts there are none and that
the published page is what :func:`fill` renders.

Three kinds of row can be red, and each is read the way the regeneration gates it.

* A **truth** row of ``performance-tests.csv`` that fails.  The comparator's own row is left
  out when the study declares an accepted reference failure, as ``regenerate.py`` does.
* A **paired** row of ``equivalence.csv`` that fails.  The verdict is *recomputed* from the
  committed leg endpoints rather than read off ``passed``, and a disagreement is refused.  A
  ledger that trusted the column would inherit any hand edit to it.
* A **property** row of ``properties.csv`` that fails its own rule *or* its family's joint
  clause, which is the rule ``claims.property_cells_passed`` counts by.  A diagnostic row
  states no verdict and is never red.

The fit-health audits two DR-TMLE studies publish are counts over fits rather than verdict
cells, and the ledger does not read them.  ``claims.load`` is never called either: it reads
the multi-megabyte replication archives, and no verdict here needs them.

Run it after ``python -m tests.studies.evidence.document``, which also calls :func:`fill`::

    python -m tests.studies.evidence.red_cells
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tests.documents import pipe_table
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.comparison import comparison_conclusion
from tests.studies.evidence.document import _CLOSE, _OPEN, _table, measured, render
from tests.studies.evidence.manifest import write_lines
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord, registered

#: The published ledger.
PAGE = ROOT / "docs" / "technical-reference" / "method-evidence" / "red-cells.md"

#: The roadmap, and the header of the RM18 table whose ``id`` column names every owner.
ROADMAP = ROOT / "docs" / "roadmap.md"
OWNER_TABLE = ("id", "work", "acceptance")

#: Where a reader finds each owner.  ``F18`` and ``F19`` have rows of their own; every other
#: owner is a row of RM18's "What this row asks for" table, which has no anchor of its own.
_RM18 = "rm18-red-property-cells-after-the-fold-scale-and-law-changes"
_OWNER_ANCHORS = {
    "F18": "f18-selector-path-c-tmle-inference",
    "F19": "f19-outcome-adaptive-c-tmle-generated-design-inference",
}

#: The three kinds of red row, in the order the ledger prints them.
KINDS = ("truth", "paired", "property")

#: The three generated blocks of :data:`PAGE`.
BLOCKS = ("red-cell-summary", "red-cells", "reporting-without-red")

SUMMARY_COLUMNS = ("owner", "red rows", "studies")
RED_COLUMNS = ("study", "kind", "row", "role", "fails by", "measured", "owner")
QUIET_COLUMNS = ("study", "verdicts it publishes", "red rows")


@dataclass(frozen=True, order=True)
class RedKey:
    """Which committed verdict a ledger row is.

    Parameters
    ----------
    slug : str
        The study's registered slug.
    kind : str
        One of :data:`KINDS`.
    key : str
        ``property/cell`` for a property row, ``scenario/estimand`` for a paired row, and
        ``implementation/scenario/estimand`` for a truth row.
    """

    slug: str
    kind: str
    key: str


@dataclass(frozen=True)
class RedRow:
    """One red verdict, with what it failed and the endpoint it failed on.

    Parameters
    ----------
    key : RedKey
        The verdict this row is.
    role : str
        The property role, ``paired``, or the implementation a truth row judges.
    fails_by : str
        Which rule, clause or leg failed.
    measured : str
        The endpoints the failed verdict was read from.
    """

    key: RedKey
    role: str
    fails_by: str
    measured: str


def _keys(slug: str, kind: str, *keys: str) -> tuple[RedKey, ...]:
    return tuple(RedKey(slug, kind, key) for key in keys)


#: The roadmap ask that owns each red row, by the ``id`` RM18 gives it.
#:
#: Written owner-first so a reader sees each ask's cells together, and flattened into
#: :data:`OWNERS` below, which refuses a row claimed twice.  Every entry must be red and every
#: red row must have an entry; :func:`findings` states both directions.
CLAIMS: dict[str, tuple[RedKey, ...]] = {
    "F18": (
        *_keys(
            "canonical-ctmle-selector",
            "property",
            "selector_necessity/collaborative",
            "selector_necessity/empty_control",
            "type_i_error/sharp_null",
        ),
        *_keys(
            "canonical-multi-arm-ctmle-selector",
            "property",
            "selector_necessity/greedy",
            "selector_necessity/ordered",
            "selector_necessity/discrete",
            "selector_necessity/empty_control",
            "type_i_error/sharp_null",
            "interval_calibration/correctly_specified",
            "root_n_and_efficiency/n_500",
        ),
    ),
    "F19": _keys(
        "canonical-multi-arm-ctmle-oat",
        "property",
        "generated_design/estimated",
        "generated_design/oracle_design",
    ),
    "RM18-fixed-weights": (
        *_keys(
            "weighted-ltmle-crossfit",
            "property",
            "interval_calibration/static__correctly_specified",
        ),
        *_keys(
            "weighted-ltmle-crossfit",
            "paired",
            "selected_censored_end_of_study/ey_regimen[never]",
        ),
    ),
    "RM18-boundary": (
        *_keys(
            "canonical-drtmle",
            "property",
            "double_robust_contraction/treatment_correct_n1500",
        ),
        *_keys(
            "canonical-multi-arm-drtmle",
            "property",
            "double_robust_contraction/outcome_correct_n4000",
            "interval_calibration/correctly_specified",
            "root_n_and_efficiency/n_500",
        ),
        *_keys(
            "weighted-ltmle-crossfit",
            "property",
            "double_robustness/static__both_wrong",
        ),
        *_keys(
            "weighted-ltmle-crossfit",
            "paired",
            "selected_censored_end_of_study/ey_regimen[always]",
            "selected_censored_end_of_study/ey_regimen[treat then continue if l2 positive]",
        ),
    ),
    "RM18-one-sided-bias": (
        *_keys(
            "canonical-drtmle",
            "property",
            "double_robustness/outcome_correct",
            "double_robustness/treatment_correct",
        ),
        *_keys(
            "canonical-multi-arm-drtmle",
            "property",
            "double_robustness/treatment_correct",
        ),
    ),
    "RM18-slopes": _keys(
        "canonical-multi-arm-drtmle",
        "property",
        "double_robust_contraction/rate_outcome_correct",
        "double_robust_contraction/rate_treatment_correct",
    ),
    "RM18-ordinary-weighted": _keys(
        "weighted-ltmle",
        "property",
        "interval_calibration/static__correctly_specified",
        "type_i_error/static__sharp_null",
        "targeting_necessity/dynamic__targeted",
        "targeting_necessity/dynamic__untargeted",
        "targeting_necessity/static__targeted",
        "targeting_necessity/static__untargeted",
    ),
    "RM18-comparator-density": _keys(
        "shift-policies",
        "paired",
        "continuous_modified_policy/ate_shift[+0.25 vs natural course]",
    ),
}


def owners(claims: Mapping[str, Sequence[RedKey]]) -> dict[RedKey, str]:
    """Flatten owner-first claims into one owner per red row.

    Parameters
    ----------
    claims : Mapping
        Owner ``id`` to the red rows it claims.

    Returns
    -------
    dict
        Each claimed row, keyed to its one owner.

    Raises
    ------
    ValueError
        If two owners, or one owner twice, claim the same row.
    """
    flat: dict[RedKey, str] = {}
    for owner, keys in claims.items():
        for key in keys:
            if key in flat:
                raise ValueError(f"{key} is claimed by {flat[key]!r} and again by {owner!r}")
            flat[key] = owner
    return flat


OWNERS = owners(CLAIMS)

#: The artefacts :func:`red_rows` reads, by the short names ``claims.ARTIFACTS`` uses.
FRAMES = {
    "performance": "performance-tests.csv",
    "equivalence": "equivalence.csv",
    "properties": "properties.csv",
}


def frames(record: StudyRecord) -> dict[str, pd.DataFrame]:
    """Read the three committed verdict tables of one study.

    Parameters
    ----------
    record : StudyRecord
        The study to read.

    Returns
    -------
    dict
        Each verdict table, keyed by its short name in :data:`FRAMES`.
    """
    return {name: pd.read_csv(record.artifact(filename)) for name, filename in FRAMES.items()}


@dataclass(frozen=True)
class PairedLegs:
    """One paired row's verdict, rebuilt from its committed leg endpoints.

    Parameters
    ----------
    similar : bool
        The paired-difference interval lies inside the similarity margin.
    rmse : bool
        The RMSE-ratio bound clears its non-inferiority margin.
    coverage : bool
        The coverage-difference bound clears its non-inferiority margin.
    calibration : bool
        The calibration-excess bound clears its margin, or the standard errors are not
        comparable.
    resolved : bool
        The calibration leg could have concluded at all.
    superior : bool
        The subject covers better and passes the other legs.
    conclusion : str
        What :func:`~tests.studies.evidence.comparison.comparison_conclusion` concludes.
    passed : bool
        The verdict the legs produce.
    """

    similar: bool
    rmse: bool
    coverage: bool
    calibration: bool
    resolved: bool
    superior: bool
    conclusion: str
    passed: bool

    def failed(self) -> list[str]:
        """Name every leg that failed, in the order the comparison reads them.

        Returns
        -------
        list of str
            One entry per failed leg, or per unresolved design.
        """
        legs = {
            "similarity leg": self.similar,
            "RMSE leg": self.rmse,
            "coverage leg": self.coverage,
            "calibration leg": self.calibration,
        }
        out = [name for name, held in legs.items() if not held]
        if not self.resolved:
            out.append("calibration resolution")
        return out


def paired_legs(row: Any, margins: Margins, subject_valid: bool) -> PairedLegs:
    """Rebuild a paired verdict from its committed endpoints.

    Each leg is the comparison :func:`tests.studies.evidence.comparison.equivalence` makes,
    against the study's declared margins, and the conclusion comes from the same
    :func:`~tests.studies.evidence.comparison.comparison_conclusion`.  Only the endpoints are
    read from the row.  The verdict columns are what this rebuilds, so they are not inputs.

    Parameters
    ----------
    row : Any
        One ``equivalence.csv`` row, as ``itertuples`` yields it.
    margins : Margins
        The study's declared margins.
    subject_valid : bool
        Whether the subject passes its own truth gate on this scenario and estimand.

    Returns
    -------
    PairedLegs
        Every leg, and the verdict they produce.
    """
    comparable = bool(row.se_comparable)
    similar = bool(
        row.paired_ci_lower >= -row.mean_margin and row.paired_ci_upper <= row.mean_margin
    )
    rmse = bool(row.rmse_ratio_upper <= margins.rmse_noninferiority)
    coverage = bool(row.coverage_difference_lower >= margins.coverage_noninferiority)
    calibration = bool(
        not comparable or row.calibration_excess_upper <= margins.calibration_noninferiority
    )
    resolution = float(row.calibration_excess_resolution)
    resolved = bool(
        not comparable
        or (math.isfinite(resolution) and resolution <= margins.calibration_noninferiority)
    )
    superior = bool(row.coverage_difference_lower > 0.0 and subject_valid and rmse and calibration)
    conclusion, passed = comparison_conclusion(
        similar=similar,
        not_inferior=rmse and coverage and calibration,
        coverage_superior=superior,
        resolved=resolved,
    )
    return PairedLegs(similar, rmse, coverage, calibration, resolved, superior, conclusion, passed)


def _paired_measured(row: Any, margins: Margins) -> str:
    text = (
        f"difference {render(float(row.paired_ci_lower))} to "
        f"{render(float(row.paired_ci_upper))} within {render(float(row.mean_margin))}, "
        f"RMSE ratio bound {render(float(row.rmse_ratio_upper))} "
        f"vs {render(margins.rmse_noninferiority)}, "
        f"coverage difference bound {render(float(row.coverage_difference_lower))} "
        f"vs {render(margins.coverage_noninferiority)}"
    )
    if bool(row.se_comparable):
        text += (
            f", calibration excess bound {render(float(row.calibration_excess_upper))} "
            f"vs {render(margins.calibration_noninferiority)}, "
            f"resolution {render(float(row.calibration_excess_resolution))}"
        )
    return text


def _truth_measured(row: Any) -> str:
    return (
        f"bias {render(float(row.bias_ci_lower))} to {render(float(row.bias_ci_upper))}, "
        f"coverage {render(float(row.coverage))}, SE ratio {render(float(row.se_ratio))}"
    )


def red_rows(record: StudyRecord, tables: Mapping[str, pd.DataFrame]) -> list[RedRow]:
    """Every red verdict one study's committed tables publish.

    Parameters
    ----------
    record : StudyRecord
        The study the tables belong to.
    tables : Mapping
        Its verdict tables, keyed as :func:`frames` returns them.

    Returns
    -------
    list of RedRow
        One row per red verdict, truth rows first, then paired, then property.

    Raises
    ------
    ValueError
        If a committed paired verdict disagrees with the verdict its legs produce.
    """
    out: list[RedRow] = []
    performance = tables["performance"]
    for row in performance.itertuples():
        if bool(row.passed):
            continue
        if record.accepted_reference_failure and row.implementation == record.reference:
            continue
        out.append(
            RedRow(
                RedKey(record.slug, "truth", f"{row.implementation}/{row.scenario}/{row.estimand}"),
                str(row.implementation),
                "its truth gate",
                _truth_measured(row),
            )
        )

    valid = performance.set_index(["implementation", "scenario", "estimand"])["passed"]
    for row in tables["equivalence"].itertuples():
        subject_valid = bool(valid.loc[(record.implementation, row.scenario, row.estimand)])
        legs = paired_legs(row, record.margins, subject_valid)
        if (legs.passed, legs.conclusion) != (bool(row.passed), str(row.comparison_conclusion)):
            raise ValueError(
                f"{record.slug} commits {row.comparison_conclusion!r} (passed={row.passed}) for "
                f"{row.scenario}/{row.estimand}, but its legs conclude {legs.conclusion!r} "
                f"(passed={legs.passed})"
            )
        if legs.passed:
            continue
        out.append(
            RedRow(
                RedKey(record.slug, "paired", f"{row.scenario}/{row.estimand}"),
                "paired",
                f"{legs.conclusion}: {', '.join(legs.failed())}",
                _paired_measured(row, record.margins),
            )
        )

    for row in tables["properties"].itertuples():
        if str(row.role) == property_verdicts.DIAGNOSTIC_ROLE:
            continue
        own, joint = bool(row.passed), bool(row.property_passed)
        if own and joint:
            continue
        out.append(
            RedRow(
                RedKey(record.slug, "property", f"{row.property}/{row.cell}"),
                str(row.role),
                "its own rule" if not own else "the family's joint clause",
                measured(row),
            )
        )
    return out


def verdict_count(tables: Mapping[str, pd.DataFrame]) -> int:
    """How many verdicts one study publishes, counted the way :func:`red_rows` reads them.

    Parameters
    ----------
    tables : Mapping
        The study's verdict tables, keyed as :func:`frames` returns them.

    Returns
    -------
    int
        Truth rows, paired rows, and property rows other than diagnostics.
    """
    properties = tables["properties"]
    scored = properties.loc[properties["role"] != property_verdicts.DIAGNOSTIC_ROLE]
    return len(tables["performance"]) + len(tables["equivalence"]) + len(scored)


def owner_ids(roadmap: Path = ROADMAP) -> list[str]:
    """The owner names RM18's "What this row asks for" table declares.

    Parameters
    ----------
    roadmap : Path
        The roadmap to read.

    Returns
    -------
    list of str
        Each ``id`` cell, with its code quotes removed, in table order.
    """
    return [row["id"].strip("`") for row in pipe_table(roadmap, OWNER_TABLE)]


def findings(
    rows: Iterable[RedRow],
    owned: Mapping[RedKey, str],
    policies: Mapping[str, str],
    ids: Iterable[str],
) -> list[str]:
    """Every way the red rows, their owners, the roadmap and the registry disagree.

    Pure, so a test can hand it a deliberately broken input and watch it object.

    Parameters
    ----------
    rows : Iterable of RedRow
        The red rows the committed tables publish.
    owned : Mapping
        The declared owner of each red row.
    policies : Mapping
        Each study slug's registered publication policy.
    ids : Iterable of str
        The owner names the roadmap declares.

    Returns
    -------
    list of str
        One sentence per disagreement.  Empty when the ledger is consistent.
    """
    rows = list(rows)
    known = set(ids)
    red = {row.key for row in rows}
    out = [f"{key} is red and has no owner" for key in sorted(red - set(owned))]
    out += [f"{key} has an owner but is not red" for key in sorted(set(owned) - red)]
    out += [
        f"{key} is owned by {owner!r}, which the roadmap's id column does not list"
        for key, owner in sorted(owned.items())
        if owner not in known
    ]
    out += [
        f"{row.key} is red in a study registered as {policies.get(row.key.slug)!r}"
        for row in rows
        if policies.get(row.key.slug) != "reporting"
    ]
    return out


def _markdown_link(text: str, target: str) -> str:
    # Joined from parts because ``test_documentation_links`` reads every bracket-parenthesis
    # pair in a source file as a link, and a template's placeholder names no file.
    return "".join(("[", text, "]", "(", target, ")"))


def _link(record: StudyRecord) -> str:
    return _markdown_link(record.name, Path(record.document).name)


def _owner(owner: str) -> str:
    anchor = _OWNER_ANCHORS.get(owner, _RM18)
    return _markdown_link(f"`{owner}`", f"../../roadmap.md#{anchor}")


def _ordered(
    rows: Iterable[RedRow], records: Sequence[StudyRecord]
) -> list[tuple[StudyRecord, RedRow]]:
    position = {record.slug: index for index, record in enumerate(records)}
    by_slug = {record.slug: record for record in records}
    return [
        (by_slug[row.key.slug], row)
        for row in sorted(
            rows,
            key=lambda row: (position[row.key.slug], KINDS.index(row.key.kind), row.key.key),
        )
    ]


def render_summary(
    rows: Sequence[RedRow],
    owned: Mapping[RedKey, str],
    records: Sequence[StudyRecord],
    ids: Sequence[str],
) -> list[str]:
    """One row per owner that holds a red row, in roadmap order, and a total.

    Parameters
    ----------
    rows : Sequence of RedRow
        The red rows.
    owned : Mapping
        The owner of each red row.
    records : Sequence of StudyRecord
        The registered studies, in grid order.
    ids : Sequence of str
        The roadmap's owner names, in table order.

    Returns
    -------
    list of str
        The table's lines.
    """
    ordered = _ordered(rows, records)
    table: list[tuple[str, str, str]] = []
    for owner in ids:
        held = [(record, row) for record, row in ordered if owned.get(row.key) == owner]
        if not held:
            continue
        names = dict.fromkeys(record.name for record, _ in held)
        table.append((_owner(owner), str(len(held)), ", ".join(names)))
    slugs = {row.key.slug for row in rows}
    table.append(("total", str(len(rows)), f"{len(slugs)} studies"))
    return _table(SUMMARY_COLUMNS, table)


def render_red_cells(
    rows: Sequence[RedRow], owned: Mapping[RedKey, str], records: Sequence[StudyRecord]
) -> list[str]:
    """One row per red verdict, in grid order.

    Parameters
    ----------
    rows : Sequence of RedRow
        The red rows.
    owned : Mapping
        The owner of each red row.
    records : Sequence of StudyRecord
        The registered studies, in grid order.

    Returns
    -------
    list of str
        The table's lines.
    """
    table = [
        (
            _link(record),
            row.key.kind,
            f"`{row.key.key}`",
            row.role,
            row.fails_by,
            row.measured,
            _owner(owned[row.key]) if row.key in owned else "**none**",
        )
        for record, row in _ordered(rows, records)
    ]
    return _table(RED_COLUMNS, table)


def render_reporting_without_red(
    rows: Sequence[RedRow],
    records: Sequence[StudyRecord],
    counts: Mapping[str, int],
) -> list[str]:
    """Every ``reporting`` study that publishes no red row.

    The policy check reads one way: a red row needs ``reporting``, and ``reporting`` needs no
    red row.  Listing these studies keeps that asymmetry visible instead of implied.

    Parameters
    ----------
    rows : Sequence of RedRow
        The red rows.
    records : Sequence of StudyRecord
        The registered studies, in grid order.
    counts : Mapping
        Each study slug's verdict count, from :func:`verdict_count`.

    Returns
    -------
    list of str
        The table's lines.
    """
    red = {row.key.slug for row in rows}
    table = [
        (_link(record), str(counts[record.slug]), "0")
        for record in records
        if record.publication_policy == "reporting" and record.slug not in red
    ]
    return _table(QUIET_COLUMNS, table)


def rendered(records: Sequence[StudyRecord] | None = None) -> dict[str, list[str]]:
    """Render all three blocks from the committed tables.

    Parameters
    ----------
    records : Sequence of StudyRecord, optional
        The studies to read.  Defaults to every registered study.

    Returns
    -------
    dict
        Each block's lines, keyed by its name in :data:`BLOCKS`.
    """
    records = registered() if records is None else records
    rows: list[RedRow] = []
    counts: dict[str, int] = {}
    for record in records:
        tables = frames(record)
        rows += red_rows(record, tables)
        counts[record.slug] = verdict_count(tables)
    ids = owner_ids()
    return {
        "red-cell-summary": render_summary(rows, OWNERS, records, ids),
        "red-cells": render_red_cells(rows, OWNERS, records),
        "reporting-without-red": render_reporting_without_red(rows, records, counts),
    }


def published(page: Path = PAGE) -> dict[str, list[list[str]]]:
    """Every generated block the page carries, by name, as each occurrence's lines.

    Parameters
    ----------
    page : Path
        The ledger page.

    Returns
    -------
    dict
        Block name to one list of lines per occurrence of that block.

    Raises
    ------
    LookupError
        If a block is opened and never closed.
    """
    lines = page.read_text(encoding="utf-8").splitlines()
    out: dict[str, list[list[str]]] = {}
    for name in BLOCKS:
        opening = _OPEN.format(name=name)
        for head, line in enumerate(lines):
            if line.strip() != opening:
                continue
            tail = next(
                (index for index in range(head + 1, len(lines)) if lines[index].strip() == _CLOSE),
                None,
            )
            if tail is None:
                raise LookupError(f"{opening} is never closed by {_CLOSE}")
            out.setdefault(name, []).append(lines[head + 1 : tail])
    return out


def fill(page: Path = PAGE) -> list[str]:
    """Rewrite the ledger page's generated blocks from the committed tables.

    Parameters
    ----------
    page : Path
        The ledger page.

    Returns
    -------
    list of str
        One line per block whose content changed.

    Raises
    ------
    LookupError
        If the page does not carry each block exactly once.
    """
    blocks = rendered()
    lines = page.read_text(encoding="utf-8").splitlines(keepends=True)
    changed: list[str] = []
    for name, block in blocks.items():
        opening = _OPEN.format(name=name)
        heads = [index for index, line in enumerate(lines) if line.strip() == opening]
        if len(heads) != 1:
            raise LookupError(f"{page.name} carries {opening} {len(heads)} times, not once")
        head = heads[0]
        tail = next(
            (index for index in range(head + 1, len(lines)) if lines[index].strip() == _CLOSE),
            None,
        )
        if tail is None:
            raise LookupError(f"{opening} is never closed by {_CLOSE}")
        replacement = [line + "\n" for line in block]
        if lines[head + 1 : tail] != replacement:
            changed.append(f"{name}: {len(block)} line(s)")
        lines[head + 1 : tail] = replacement
    write_lines(page, "".join(lines))
    return changed


def report() -> None:
    """Rewrite the ledger page and print what changed."""
    changed = fill()
    print(f"red-cell ledger: {len(changed)} block(s) updated in {PAGE.relative_to(ROOT)}")
    for line in changed:
        print(f"  {line}")


def main() -> None:
    """Parse no arguments, so ``--help`` describes the module, and rewrite the page."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()


if __name__ == "__main__":
    main()
