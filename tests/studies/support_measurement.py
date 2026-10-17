"""The phase-1 measurement of the support label: read the record files and write the counts.

Run a process with ``CLEVERLY_SUPPORT_RECORD=<dir>``. Each fit then writes JSON lines to
``<dir>/<pid>.jsonl``, and this script reads every such file under one root. The directory of a
file names its source: ``fast``, ``notebooks/<name>``, ``cases`` or ``studies/<name>``.

The tables follow section 8 of the RM42b plan.

======================= ===========================================================================
table                   content
======================= ===========================================================================
fits by label           per source, the calls and the distinct keys that classify supported or
                        experimental
experimental keys       each distinct experimental key with its nearest row and differing axes
withheld to shown       per status, the fits whose interval moves to shown under the default and
                        under the alternative of Q1
bootstrap and band      per row of plan section 3.4, the fits that keep or change their label
group-A reach           the calls that reach each refusal site, flagged sites apart
studies                 the runner keys of each study that classify experimental or match no case
======================= ===========================================================================

A call is one record. A distinct key counts once per source. The script reads files only. It
runs no fit and no study, and it changes nothing under the record root.

Examples
--------
    python -m tests.studies.support_measurement --records $REC --out $REC/measurement.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Measurement", "canonical_directories", "case_keys", "main", "measure", "read_records"]

#: The statuses whose interval the default of plan section 3.3 shows.
DEFAULT_SHOWN = ("estimated_weight_plugin", "few_cluster_plugin")
#: The statuses the alternative (Q1 answered "show") adds to the default.
ALTERNATIVE_SHOWN = (
    *DEFAULT_SHOWN,
    "working_mechanism_plugin",
    "generated_design_plugin",
    "adaptive_baseline_plugin",
    "constant_node_plugin",
)
#: The longitudinal and point engines whose bootstrap stays diagnostic (plan section 3.4).
DIAGNOSTIC_BOOTSTRAP_ENGINE = "collaborative_tmle"

ROOT = Path(__file__).resolve().parents[2]

#: The record kinds a fit count reads.
FIT_RECORD = "classification"
OUTCOME_RECORD = "fit_outcome"


def _category(group: str) -> str:
    """The first part of a record directory, such as ``fast`` or ``studies``."""
    return group.split("/", 1)[0] if group != "." else "."


def read_records(root: Path) -> Iterator[dict[str, Any]]:
    """Every record under ``root``, with the directory of its file as ``group``.

    Parameters
    ----------
    root : Path
        The record root. It holds one directory per source, and each holds ``<pid>.jsonl``.

    Yields
    ------
    dict
        The parsed record, plus ``group`` (the directory relative to ``root``, ``"."`` for the
        root itself) and ``malformed`` set to ``True`` for a line that is not JSON.
    """
    for path in sorted(root.rglob("*.jsonl")):
        group = path.parent.relative_to(root).as_posix()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    yield {"record": "malformed", "group": group, "path": str(path)}
                    continue
                if isinstance(record, dict):
                    record["group"] = group
                    yield record


def _key_text(key: Mapping[str, Any]) -> str:
    """A stable text of a key, equal for equal keys."""
    return json.dumps(key, sort_keys=True, default=str)


def _summary(key: Mapping[str, Any]) -> str:
    """A short text of a key: its engine, families, outcome, missing, treatment and splitting."""
    parts = []
    for axis in (
        "engine",
        "family",
        "outcome",
        "missing",
        "treatment",
        "weighting",
        "splitting",
        "dependence",
        "qualifiers",
    ):
        value = key.get(axis)
        if value in (None, [], ()) or (axis == "qualifiers" and not value):
            continue
        parts.append(",".join(map(str, value)) if isinstance(value, list) else str(value))
    return " | ".join(parts)


def case_keys() -> dict[str, set[str]]:
    """The key text of every row case, by the slug of its study.

    Returns
    -------
    dict of str to set of str
        For each registered slug a case names, the keys of the cases that name it. The cases
        run each study's own fit up to the classifier and draw small samples, so the call
        takes about a minute.
    """
    from tests.studies.support_cases import CASES, case_key

    keys: dict[str, set[str]] = defaultdict(set)
    for case in CASES:
        keys[case.slug].add(_key_text(case_key(case).to_json()))
    return dict(keys)


def canonical_directories(
    canonical: Path | None = None, runner_modules: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The study slug of each ``tests/canonical/<dir>`` regeneration script.

    A script names its runner module in its imports. The map matches that module to the
    registered study that declares it.

    Parameters
    ----------
    canonical : Path or None, default None
        The ``tests/canonical`` directory. ``None`` reads the repository's own.
    runner_modules : mapping of str to str or None, default None
        Each registered slug with its runner module. ``None`` reads the registry.

    Returns
    -------
    dict of str to str
        The directory name to the slug. A directory whose script names no registered runner
        is absent.
    """
    canonical = canonical or ROOT / "tests" / "canonical"
    if runner_modules is None:
        from tests.studies.evidence.registry import registered

        runner_modules = {record.slug: record.runner_module for record in registered()}
    by_module = {module.rsplit(".", 1)[-1]: slug for slug, module in runner_modules.items()}
    found: dict[str, str] = {}
    for script in sorted(canonical.glob("*/regenerate.py")):
        text = script.read_text(encoding="utf-8")
        names = re.findall(r"\b[a-z][a-z0-9_]*\b", text)
        for name in names:
            if name in by_module:
                found[script.parent.name] = by_module[name]
                break
    return found


@dataclass
class _Fit:
    """What the records say about one distinct (source, key, label)."""

    key: Mapping[str, Any]
    label: str
    row: str | None
    nearest: Any
    conditions: tuple[str, ...]
    calls: int = 0


@dataclass
class Measurement:
    """The tables of one measurement.

    Parameters
    ----------
    totals : dict of str to int
        Record counts by kind, with ``fits`` the number of classification records.
    fits : dict of str to dict
        Per source: ``calls`` and ``keys`` per label.
    experimental : list of dict
        Each distinct experimental key, with its calls and sources.
    status : dict of str to dict
        Per status: the calls and keys that carry it, and the calls that move to shown.
    flips : dict of str to dict
        Per bootstrap and band row: the calls and the distinct keys.
    reach : list of dict
        Per refusal site: the calls by category.
    studies : list of dict
        The runner keys that classify experimental or match no case.
    silent_studies : list of str
        Registered studies with no record.
    errors : list of dict
        The classifier failures.
    """

    totals: dict[str, int] = field(default_factory=dict)
    fits: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    experimental: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, dict[str, int]] = field(default_factory=dict)
    flips: dict[str, dict[str, int]] = field(default_factory=dict)
    reach: list[dict[str, Any]] = field(default_factory=list)
    studies: list[dict[str, Any]] = field(default_factory=list)
    silent_studies: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


def _bootstrap_row(record: Mapping[str, Any], kinds: frozenset[str]) -> str:
    """The row of plan section 3.4 a bootstrap record falls in."""
    engine = (record.get("key", {}).get("engine") or [""])[0]
    supplies = record.get("supplies_inference", [])
    dependence = record.get("key", {}).get("dependence")
    if record["bootstrap_inferential_today"]:
        if "bootstrap" in kinds:
            return "licensed bootstrap: inferential and supported"
        return "inferential today, no bootstrap in the row: becomes experimental"
    if engine == DIAGNOSTIC_BOOTSTRAP_ENGINE:
        return "C-TMLE bootstrap: stays diagnostic"
    if not all(supplies):
        return "bootstrap at a withheld status: stays diagnostic"
    if dependence == "below_floor":
        return "bootstrap below the cluster floor: stays diagnostic"
    return "diagnostic today at a supplying status: inferential and experimental"


def measure(
    records: Iterable[Mapping[str, Any]],
    *,
    rows: Mapping[str, frozenset[str]],
    cases: Mapping[str, set[str]] | None = None,
    directories: Mapping[str, str] | None = None,
    registered_slugs: Iterable[str] = (),
) -> Measurement:
    """Count the records into the tables.

    Parameters
    ----------
    records : iterable of mapping
        The records, each with ``group`` as :func:`read_records` sets it.
    rows : mapping of str to frozenset of str
        The kinds each registered row licenses, by row id.
    cases : mapping of str to set of str or None, default None
        The key text of each study's cases. ``None`` skips the case comparison, and every study
        key then counts as matching.
    directories : mapping of str to str or None, default None
        The slug of each ``studies/<name>`` directory. A name with no entry stands for itself.
    registered_slugs : iterable of str, default ()
        The registered studies, to list those with no record.

    Returns
    -------
    Measurement
        The tables.
    """
    directories = directories or {}
    result = Measurement()
    fits: dict[tuple[str, str, str], _Fit] = {}
    by_group_calls: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    status_calls: dict[str, Counter[str]] = defaultdict(Counter)
    status_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    flip_calls: Counter[str] = Counter()
    flip_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    sites: dict[tuple[str, bool], dict[str, Any]] = {}
    study_seen: set[str] = set()

    for record in records:
        kind = str(record.get("record"))
        group = str(record.get("group", "."))
        totals[kind] += 1
        if kind == "classification_error":
            result.errors.append(
                {"group": group, "source": record.get("source"), "error": record.get("error")}
            )
            continue
        if kind == "refusal_site":
            site = (str(record.get("site")), bool(record.get("flagged")))
            entry = sites.setdefault(
                site,
                {
                    "site": site[0],
                    "flagged": site[1],
                    "code": record.get("code"),
                    "roadmap": record.get("roadmap"),
                    "calls": Counter(),
                },
            )
            entry["calls"][_category(group)] += 1
            continue
        if kind == FIT_RECORD:
            key = record["key"]
            text = _key_text(key)
            label = str(record["label"])
            slot = fits.setdefault(
                (group, text, label),
                _Fit(
                    key,
                    label,
                    record.get("row"),
                    record.get("nearest"),
                    tuple(map(str, record.get("conditions") or ())),
                ),
            )
            slot.calls += 1
            by_group_calls[group][label] += 1
            totals["fits"] += 1
            if _category(group) == "studies":
                study_seen.add(directories.get(group.split("/", 1)[1], group.split("/", 1)[1]))
            continue
        if kind != OUTCOME_RECORD:
            continue
        text = _key_text(record.get("key", {}))
        statuses = [s for s in record.get("statuses", []) if s != "influence_curve"]
        for status in statuses:
            status_calls[status][group] += 1
            status_keys[status].add((group, text))
        if statuses:
            if set(statuses) <= set(DEFAULT_SHOWN):
                flip_calls["default"] += 1
                flip_keys["default"].add((group, text))
            if set(statuses) <= set(ALTERNATIVE_SHOWN):
                flip_calls["alternative"] += 1
                flip_keys["alternative"].add((group, text))
        row_kinds = rows.get(record.get("row") or "", frozenset())
        if record.get("bootstrap_attached"):
            label = _bootstrap_row(record, row_kinds)
            result.flips.setdefault(label, {"calls": 0, "keys": 0})["calls"] += 1
            flip_keys[f"bootstrap:{label}"].add((group, text))
        if record.get("band_built_today"):
            label = (
                "band built, row lists band: supported"
                if "band" in row_kinds
                else "band built, shape not in a row: experimental"
            )
            result.flips.setdefault(label, {"calls": 0, "keys": 0})["calls"] += 1
            flip_keys[f"band:{label}"].add((group, text))
        elif set(statuses) & set(DEFAULT_SHOWN):
            label = "no band today, status shown by default: a band is built when requested"
            result.flips.setdefault(label, {"calls": 0, "keys": 0})["calls"] += 1
            flip_keys[f"band:{label}"].add((group, text))

    for label, slot in result.flips.items():
        for prefix in ("bootstrap:", "band:"):
            keys = flip_keys.get(prefix + label)
            if keys is not None:
                slot["keys"] = len(keys)

    # fits by label
    for (group, _text, label), fit in sorted(fits.items()):
        entry = result.fits.setdefault(
            group,
            {
                "supported": {"calls": 0, "keys": 0},
                "experimental": {"calls": 0, "keys": 0},
            },
        )
        entry[label]["calls"] += fit.calls
        entry[label]["keys"] += 1

    # experimental keys: one row per distinct key and nearest
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for (group, text, label), fit in fits.items():
        if label != "experimental":
            continue
        nearest = json.dumps(fit.nearest, sort_keys=True) if fit.nearest else "none"
        slot = merged.setdefault(
            (text, nearest),
            {
                "key": _summary(fit.key),
                "nearest": fit.nearest,
                "conditions": set(),
                "calls": Counter(),
            },
        )
        slot["conditions"].update(fit.conditions)
        slot["calls"][_category(group)] += fit.calls
    result.experimental = sorted(
        (
            {**entry, "conditions": sorted(entry["conditions"]), "calls": dict(entry["calls"])}
            for entry in merged.values()
        ),
        key=lambda entry: -sum(entry["calls"].values()),
    )

    # withheld to shown
    for status in ALTERNATIVE_SHOWN:
        result.status[status] = {
            "calls": sum(status_calls[status].values()),
            "keys": len(status_keys[status]),
            "default_shows": int(status in DEFAULT_SHOWN),
        }
    result.status["(fits whose every status moves)"] = {
        "calls": flip_calls["default"],
        "keys": len(flip_keys["default"]),
        "default_shows": 1,
        "alternative_calls": flip_calls["alternative"],
        "alternative_keys": len(flip_keys["alternative"]),
    }

    # group-A reach
    result.reach = sorted(
        (
            {**{k: v for k, v in entry.items() if k != "calls"}, "calls": dict(entry["calls"])}
            for entry in sites.values()
        ),
        key=lambda entry: (not entry["flagged"], entry["site"]),
    )

    # studies: runner keys that classify experimental or match no case
    for (group, text, label), fit in sorted(fits.items()):
        if _category(group) != "studies":
            continue
        name = group.split("/", 1)[1].split("/", 1)[0]
        slug = directories.get(name, name)
        matches = cases is None or text in cases.get(slug, set())
        if label == "experimental" or not matches:
            result.studies.append(
                {
                    "study": slug,
                    "directory": name,
                    "label": label,
                    "row": fit.row,
                    "nearest": fit.nearest,
                    "matches_a_case": matches,
                    "calls": fit.calls,
                    "key": _summary(fit.key),
                }
            )
    result.silent_studies = sorted(set(registered_slugs) - study_seen)
    result.totals = dict(sorted(totals.items()))
    return result


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        cells = [str(cell).replace("|", "/").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render(result: Measurement) -> str:
    """The tables as Markdown.

    Parameters
    ----------
    result : Measurement
        The tables of :func:`measure`.

    Returns
    -------
    str
        The Markdown text.
    """
    out: list[str] = ["# Support measurement", ""]
    out += ["## Record counts", ""]
    out += [*_table(["record kind", "count"], result.totals.items()), ""]

    out += ["## Fits by label", ""]
    rows = []
    for group, entry in sorted(result.fits.items()):
        rows.append(
            (
                group,
                entry["supported"]["calls"],
                entry["supported"]["keys"],
                entry["experimental"]["calls"],
                entry["experimental"]["keys"],
            )
        )
    out += _table(
        ["source", "supported calls", "supported keys", "experimental calls", "experimental keys"],
        rows,
    )
    out += [""]

    out += ["## Experimental keys", ""]
    out += _table(
        ["key", "nearest row", "differing axes", "group-A conditions", "calls by source"],
        (
            (
                entry["key"],
                (entry["nearest"] or ["none", []])[0],
                ", ".join((entry["nearest"] or ["none", []])[1]),
                ", ".join(entry["conditions"]),
                ", ".join(f"{k} {v}" for k, v in sorted(entry["calls"].items())),
            )
            for entry in result.experimental
        ),
    )
    out += [""]

    out += ["## Withheld to shown", ""]
    out += [
        "A status moves to shown when every labelled status of a fit is in the shown set "
        "(a withheld status wins over a shown one). The default shows the first two statuses. "
        "The alternative of Q1 shows all six.",
        "",
    ]
    out += _table(
        ["status", "calls", "keys", "shown by default", "calls that move (alternative)"],
        (
            (
                status,
                entry["calls"],
                entry["keys"],
                "yes" if entry["default_shows"] else "no",
                entry.get("alternative_calls", ""),
            )
            for status, entry in result.status.items()
        ),
    )
    out += [""]

    out += ["## Bootstrap and band flips", ""]
    out += _table(
        ["row of plan section 3.4", "calls", "distinct keys"],
        ((label, entry["calls"], entry["keys"]) for label, entry in sorted(result.flips.items())),
    )
    out += [""]

    out += ["## Group-A and flagged reach", ""]
    out += _table(
        ["site", "flagged", "code", "roadmap", "calls by source"],
        (
            (
                entry["site"],
                "yes" if entry["flagged"] else "",
                entry["code"],
                entry["roadmap"],
                ", ".join(f"{k} {v}" for k, v in sorted(entry["calls"].items())),
            )
            for entry in result.reach
        ),
    )
    out += [""]

    out += ["## Studies", ""]
    out += [
        "Each runner key that classifies experimental or equals no case key of its study.",
        "",
    ]
    out += _table(
        ["study", "label", "row", "nearest", "matches a case", "calls", "key"],
        (
            (
                entry["study"],
                entry["label"],
                entry["row"] or "",
                (entry["nearest"] or ["none", []])[0]
                + (": " + ", ".join(entry["nearest"][1]) if entry["nearest"] else ""),
                "yes" if entry["matches_a_case"] else "no",
                entry["calls"],
                entry["key"],
            )
            for entry in result.studies
        ),
    )
    out += [
        "",
        "Unmapped runner configurations: "
        + str(sum(1 for entry in result.studies if not entry["matches_a_case"])),
        "",
    ]
    if result.silent_studies:
        out += ["Registered studies with no record: " + ", ".join(result.silent_studies), ""]
    if result.errors:
        out += ["## Classifier errors", ""]
        out += _table(
            ["source", "group", "error"],
            ((e["source"], e["group"], e["error"]) for e in result.errors),
        )
        out += [""]
    return "\n".join(out) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser of ``python -m tests.studies.support_measurement``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tests.studies.support_measurement",
        description=(
            "Read the CLEVERLY_SUPPORT_RECORD files under a root and write the phase-1 "
            "measurement tables as Markdown. The script runs no fit and no study."
        ),
    )
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="the record root: one directory per source, each holding <pid>.jsonl files",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="the Markdown file to write (default: standard out)"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the tables as JSON to this file"
    )
    parser.add_argument(
        "--skip-cases",
        action="store_true",
        help="do not build the row cases, so every study key counts as matching a case",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    case_source: Callable[[], dict[str, set[str]]] = case_keys,
) -> int:
    """Run the measurement.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The arguments. ``None`` reads ``sys.argv``.
    case_source : callable, default case_keys
        Returns the key text of each study's cases. A test passes a stub.

    Returns
    -------
    int
        The exit status: ``0``, or ``2`` when the record root is not a directory.
    """
    args = build_parser().parse_args(argv)
    if not args.records.is_dir():
        print(f"error: {args.records} is not a directory", file=sys.stderr)
        return 2
    from cleverly._support import registered_rows
    from tests.studies.evidence.registry import registered

    rows = {row.id: row.kinds for row in registered_rows()}
    runners = {record.slug: record.runner_module for record in registered()}
    result = measure(
        read_records(args.records),
        rows=rows,
        cases=None if args.skip_cases else case_source(),
        directories=canonical_directories(runner_modules=runners),
        registered_slugs=runners,
    )
    text = render(result)
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result.__dict__, indent=2, sort_keys=True, default=lambda v: sorted(v)),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
