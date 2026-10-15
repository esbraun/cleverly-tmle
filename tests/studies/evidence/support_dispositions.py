"""The disposition of each positive red row of the validation ledger (plan section 6).

A support row licenses a kind of inference only where its study cells are green. A positive red
row is a cell that fails: a property row with the role ``positive``, a truth row of the
``cleverly`` implementation, or a paired row. Each such row needs one decision, because the
row that its study licenses cannot claim the kind that the red cell measures.

This module holds the decision table and builds its draft. ``DISPOSITIONS`` and ``DECISIONS``
stay empty until the owner answers the flagged lines, and then one commit fills them.
``python -m tests.studies.evidence.support_dispositions --draft`` prints the draft.

Examples
--------
    python -m tests.studies.evidence.support_dispositions --draft
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.studies.evidence.red_cells import RedKey, RedRow

__all__ = [
    "DECISIONS",
    "DISPOSITIONS",
    "Disposition",
    "DraftEntry",
    "PositiveRow",
    "draft_entries",
    "kinds_of",
    "main",
    "positive_red_rows",
    "render_draft",
]

Choice = Literal["supported with a stated limit", "experimental", "withheld (status)"]

SUPPORTED_WITH_LIMIT: Choice = "supported with a stated limit"
EXPERIMENTAL: Choice = "experimental"
WITHHELD: Choice = "withheld (status)"

#: The owners whose red rows default to a stated limit: the owner reads a finite-sample effect
#: and no defect, with the measured numbers (plan section 6.2).
LIMIT_OWNERS = frozenset(
    {
        "band-finite-sample",
        "strata-boundary-mean",
        "X13-finite-sample",
        "mtp-point-calibration",
        "mtp-longitudinal-limits",
        "F1-power-design",
        "X4-finite-sample",
        "F4-calibration-draws",
        "composite-high-arm",
        "RM18-ordinary-weighted",
        "RM18-fixed-weights",
        "RM18-boundary",
        "X8-identity-small-stratum",
        "X8-drtmle-small-stratum",
        "X16-plugin-variance",
        "RM18-comparator-density",
    }
)
#: The owners whose red rows default to experimental, and the owner decides (flagged).
FLAGGED_EXPERIMENTAL = {
    "X8-logit-small-stratum": "the logit slope at a small stratum reads an SE ratio of 0.78 to 0.94",
    "X8-drtmle-one-sided-bias": "the one-sided bias of the DR-TMLE cell",
    "RM18-one-sided-bias": "the one-sided bias of the DR-TMLE cell",
    "F28": "the informative-size t cells read a bias of 0.11 to 0.13 and an SE ratio of 0.83",
    "F27": "the boundary truth rows read a coverage of 0.894 and an SE ratio of 0.82",
}
#: The studies whose interval calibration cells use a t reference, so a Wald row there is ``wald_t``.
T_REFERENCE_SLUGS = frozenset({"clustered-few-cluster-tmle", "few-cluster-cross-fitted-ltmle"})

_POINT_FAMILIES = frozenset(
    {
        "double_robustness",
        "double_robust_contraction",
        "sequential_double_robustness",
        "targeting_necessity",
        "selector_necessity",
        "generated_design",
    }
)
_THEORY_LEGS = {"bias_equivalent": "point", "coverage_valid": "wald", "se_calibrated": "wald"}


@dataclass(frozen=True)
class Disposition:
    """The decision about one positive red row.

    Parameters
    ----------
    row : str
        The support row the decision applies to. The study of the red row is in its evidence.
    kinds : tuple of str
        The kinds of inference the red row measures, as :data:`cleverly._support.Kind`.
    disposition : {"supported with a stated limit", "experimental", "withheld (status)"}
        What the row licenses for those kinds.
    limit : str
        The limit printed in the row note. Empty unless the disposition is a stated limit.
    flag : str, default ""
        When non-empty, the owner decides the line (Q6).
    """

    row: str
    kinds: tuple[str, ...]
    disposition: Choice
    limit: str
    flag: str = ""


@dataclass(frozen=True)
class PositiveRow:
    """A positive red row with what the committed tables say about its failed legs.

    Parameters
    ----------
    red : RedRow
        The red verdict.
    owner : str
        The owner of the red row in the red-cell ledger, or an empty text.
    failed_legs : tuple of str, default ()
        For a truth row, the legs of the performance table that failed.
    """

    red: RedRow
    owner: str = ""
    failed_legs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DraftEntry:
    """One line of the draft: the default decision and the rows it could apply to.

    Parameters
    ----------
    key : RedKey
        The red row.
    kinds : tuple of str
        The kinds the red row measures.
    disposition : str
        The default disposition.
    limit : str
        The default limit text.
    flag : str
        The reason the owner decides, or an empty text.
    candidates : tuple of str
        The rows whose evidence names the study and whose kinds include the red row's.
    owner : str
        The owner of the red row.
    """

    key: RedKey
    kinds: tuple[str, ...]
    disposition: str
    limit: str
    flag: str
    candidates: tuple[str, ...]
    owner: str


#: The committed decisions, one per positive red row. They stay empty until the owner answers.
DISPOSITIONS: Mapping[RedKey, Disposition] = {}
#: The owner's answer for each flagged key.
DECISIONS: Mapping[RedKey, str] = {}


def kinds_of(positive: PositiveRow) -> tuple[str, ...]:
    """The kinds of inference one positive red row measures (plan section 6.2).

    Parameters
    ----------
    positive : PositiveRow
        The red row.

    Returns
    -------
    tuple of str
        Kinds in the order point, wald, wald_t, band, bootstrap, test. A property family with
        no rule gives an empty tuple, so the draft shows the gap.
    """
    key = positive.red.key
    if key.kind == "paired":
        return ("wald",)
    if key.kind == "truth":
        legs = {_THEORY_LEGS[leg] for leg in positive.failed_legs if leg in _THEORY_LEGS}
        return tuple(kind for kind in ("point", "wald") if kind in legs) or ("point", "wald")
    family, _, cell = key.key.partition("/")
    if family == "interval_calibration":
        if cell.startswith("boot_"):
            return ("bootstrap",)
        return ("wald_t",) if key.slug in T_REFERENCE_SLUGS else ("wald",)
    if family in ("few_cluster_reference", "clustered_inference"):
        return ("wald_t", "wald")
    if family == "simultaneous_coverage":
        return ("band",)
    if family == "root_n_and_efficiency":
        return ("point", "wald")
    if family in _POINT_FAMILIES:
        return ("point",)
    if family in ("type_i_error", "power"):
        return ("test",)
    return ()


def _default(positive: PositiveRow) -> tuple[str, str, str]:
    """The default disposition, limit and flag of one positive red row."""
    owner, key = positive.owner, positive.red.key
    if owner in ("F18", "F19"):
        return WITHHELD, "", "Q1"
    if owner in LIMIT_OWNERS:
        return SUPPORTED_WITH_LIMIT, f"{owner}: {positive.red.measured}", ""
    if owner == "X20-bootstrap":
        if "clustered" in key.key:
            return EXPERIMENTAL, "", "X20 clustered cell: coverage 0.912 to 0.934"
        return EXPERIMENTAL, "", ""
    if owner in FLAGGED_EXPERIMENTAL:
        return EXPERIMENTAL, "", f"{owner}: {FLAGGED_EXPERIMENTAL[owner]}"
    return EXPERIMENTAL, "", f"no default for owner {owner or 'none'}"


def draft_entries(
    positives: Iterable[PositiveRow], rows: Iterable[tuple[str, frozenset[str], Sequence[str]]]
) -> list[DraftEntry]:
    """The default decision of each positive red row.

    Parameters
    ----------
    positives : iterable of PositiveRow
        The positive red rows.
    rows : iterable of tuple
        Each support row as ``(id, kinds, evidence)``.

    Returns
    -------
    list of DraftEntry
        One entry per red row, in the order given. The candidates are the rows that name the
        study and list every kind the red row measures. A withheld status applies to any row of
        the study, so its candidates ignore the kinds.
    """
    table = list(rows)
    entries = []
    for positive in positives:
        kinds = kinds_of(positive)
        disposition, limit, flag = _default(positive)
        if not kinds:
            flag = (flag + "; " if flag else "") + "no kind rule for this property family"
        slug = positive.red.key.slug
        candidates = tuple(
            row_id
            for row_id, row_kinds, evidence in table
            if slug in evidence and (disposition == WITHHELD or set(kinds) <= row_kinds)
        )
        if not candidates:
            flag = (flag + "; " if flag else "") + "no row of the study lists these kinds"
        entries.append(
            DraftEntry(
                positive.red.key, kinds, disposition, limit, flag, candidates, positive.owner
            )
        )
    return entries


def positive_red_rows() -> list[PositiveRow]:
    """Every positive red row of the committed verdict tables.

    Returns
    -------
    list of PositiveRow
        Property rows with the role ``positive``, truth rows of an implementation that starts
        with ``cleverly``, and paired rows, across the registered studies.
    """
    from tests.studies.evidence.red_cells import OWNERS, frames, red_rows
    from tests.studies.evidence.registry import registered

    found: list[PositiveRow] = []
    for record in registered():
        tables = frames(record)
        performance = tables["performance"].set_index(["implementation", "scenario", "estimand"])
        for red in red_rows(record, tables):
            kind = red.key.kind
            positive = (
                (kind == "property" and red.role == "positive")
                or (kind == "truth" and red.role.startswith("cleverly"))
                or kind == "paired"
            )
            if not positive:
                continue
            legs: tuple[str, ...] = ()
            if kind == "truth":
                row = performance.loc[tuple(red.key.key.split("/"))]
                legs = tuple(leg for leg in _THEORY_LEGS if leg in row and not bool(row[leg]))
            found.append(PositiveRow(red, OWNERS.get(red.key, ""), legs))
    return found


def render_draft(entries: Sequence[DraftEntry]) -> str:
    """The draft as Markdown, with the flagged lines marked.

    Parameters
    ----------
    entries : sequence of DraftEntry
        The draft entries.

    Returns
    -------
    str
        A counts block and one table row per entry. A flagged line starts with ``FLAG``.
    """
    by_disposition = Counter(entry.disposition for entry in entries)
    flagged = [entry for entry in entries if entry.flag]
    lines = [
        "# B4 draft: dispositions of the positive red rows",
        "",
        f"Entries: {len(entries)}. Flagged: {len(flagged)}. "
        f"With no candidate row: {sum(1 for entry in entries if not entry.candidates)}.",
        "",
        *(f"- {name}: {count}" for name, count in sorted(by_disposition.items())),
        "",
        "| flag | study | row key | kinds | owner | disposition | candidate rows | limit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        cells = (
            f"FLAG: {entry.flag}" if entry.flag else "",
            entry.key.slug,
            f"{entry.key.kind}: {entry.key.key}",
            ", ".join(entry.kinds) or "none",
            entry.owner,
            entry.disposition,
            ", ".join(entry.candidates) or "none",
            entry.limit,
        )
        lines.append("| " + " | ".join(re.sub(r"[|\n]", "/", cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser of ``python -m tests.studies.evidence.support_dispositions``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tests.studies.evidence.support_dispositions",
        description=(
            "Print the draft disposition of each positive red row of the validation ledger. "
            "The script reads the committed verdict tables and the support rows. It runs no "
            "study."
        ),
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        required=True,
        help="print the draft entries, with the flagged lines marked",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the draft to this file (default: standard out)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the draft.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The arguments. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The exit status, ``0``.
    """
    args = build_parser().parse_args(argv)
    from cleverly._support import registered_rows

    rows = [(row.id, row.kinds, row.evidence) for row in registered_rows()]
    text = render_draft(draft_entries(positive_red_rows(), rows))
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
