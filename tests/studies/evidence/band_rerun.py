"""The audited targeted rerun of every band cell the default band moves (band-finite-sample Wave 2).

PR 2 of band-finite-sample moved the default band to the Mammen multiplier, 10,000 draws and the
logit scale.  No estimate, standard error or pointwise interval moves, and the band's multiplier
draws its own generator (``multiplier.py``), so only the rows a band determines change.  This
module rebuilds exactly those rows and keeps every other published byte (plan section 8).

**Audit.**  :func:`build_plan` writes ``tests/canonical/band-rerun-plan.json`` before any fit.
For each registered study with a positive band cell it reads the committed ``properties.csv``
(the band cells, their R, n and ``passed``), scans the study's modules for explicit band options
(:func:`band_pins`), and lists the property keys whose cells include a band cell
(:func:`capture`).  A study counts only when the flip moves its band rows: ``longitudinal-sdr``
builds its band on explicit options, and ``targeting-curves`` reads the curve band, which keeps
its own (:data:`EXCLUDED`).  :func:`validate_plan` is the fast test's check of the plan against
the tree.

**Capture.**  A study's ``generate_property_rows`` runs with
:func:`~tests.studies.evidence.replication.fit_properties` swapped for a recorder.  The recorder
keeps each call's keys, worker, draw and ``cells_of``, selects the keys whose cells include a band
cell, and hands the study back the committed rows of every key, so nothing is fitted and the
study's own code names the keys.

**Rerun.**  :func:`execute` fits the selected keys with their committed seeds and the new defaults,
through the same ``fit_properties`` (so failures, checkpoints and worker processes behave as in
a regeneration), and :func:`replay` holds every non-band row of a refitted key to the committed
row: numeric fields within :data:`REPLAY_TOLERANCE`, absolute or relative, whichever is larger,
and keys, counts, cells, roles and the discrete fields exactly.  A miss stops the study, which
then takes a complete rerun.  :func:`splice` replaces the band rows, the study's own
``summarize_properties`` reads the complete table, and :func:`composite_manifest` records the
baseline, the selection, the replay, the refused bands and the producer.  Primary, reference and
equivalence artifacts stay byte-identical, so no R container runs.

**Blocked mode** (plan section 4.4) writes only ``tests/canonical/<study>/band_rerun/``: the band
rows, the recomputed summary with its verdicts labelled diagnostic, and the record.  Every
committed artifact stays byte-identical, and no support or grid row reads a sidecar.

**Smoke and timing.**  ``--smoke`` fits the first :data:`SMOKE_REPLICATES` band keys of each
band cell on :data:`SMOKE_STREAM`, which salts every study seed function and is never the declared
stream, and reports per cell the smallest Kish count of a banded curve, the share of refused
bands, the out-of-range levels and the seconds per key.  ``--timing`` fits :data:`TIMING_KEYS`
band keys per study the same way and extrapolates the study's CPU hours.  Neither writes into the
repository.  ``--replay-check`` refits :data:`TIMING_KEYS` band keys on their declared seeds and
applies the replay rule to their other rows, without writing anything.

Run one study at a time, into scratch output outside the repository, and copy a verified run in
with ``--install``::

    python -m tests.studies.evidence.band_rerun --study <slug> --jobs N --cache DIR \\
        --output OUT [--blocked] [--install]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tests.studies.evidence import property_failures, replication, seeds, simultaneous
from tests.studies.evidence.manifest import csv_text, provenance, read_rows, write_csv, write_rows
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

MODE = "targeted_band_rerun"
#: The commit whose committed artifacts and ``passed`` columns the plan reads.
BASELINE_COMMIT = "cef2aa65ec023744523a4b239f2517e025516ed4"
PLAN_PATH = ROOT / "tests/canonical/band-rerun-plan.json"
SCHEMA_VERSION = 1
#: Declared before any fit (plan section 8), as ``seed_repair.REPLAY_TOLERANCE``.
REPLAY_TOLERANCE = 1e-9
REPLAY_RULE = (
    "every non-band row of a refitted key: abs(new - committed) <= 1e-9 * max(1, abs(committed)) "
    "on truth, estimate and std_error; property, cell, role, replicate, n, requested_replicates, "
    "failed_replicates, covered and rejected exactly"
)
#: The band the flip moved to.
DEFAULT_BAND = {"band_scale": "logit", "kind": "mammen", "n_replicates": 10_000}
#: The family every band row belongs to, and the cell suffixes whose rows a band determines:
#: the band cell, the IC band a bootstrap-gated cell reports beside it, and the band a plug-in
#: fit's shipped curves give.  Every other row of a refitted key must replay.
BAND_FAMILY = simultaneous.FAMILY
BAND_SUFFIXES = ("__simultaneous_band", "__ic_band", "__shipped_curve_band_control")
#: The studies with a positive band cell that the flip does not move, and why.
EXCLUDED = {
    "longitudinal-sdr": (
        "the six_regimens band and every reporting band are built on explicit BandOptions "
        "(longitudinal_sdr_properties.BAND_OPTIONS and REPORTING), so the default never reaches "
        "them; the band-finite-sample PR 1 run measured them"
    ),
    "targeting-curves": (
        "its band cells read CurveBand from effect_curve(), which keeps Rademacher at 1,000 "
        "draws (plan section 4.3); its rows stay byte-identical"
    ),
}
#: The text that shows a study's band options are explicit, for each excluded study.
EXCLUSION_WITNESS = {
    "longitudinal-sdr": 'BandOptions(band_scale="logit", kind="mammen", n_replicates=10_000',
}
SIDECAR = "band_rerun"
SIDECAR_FILES = ("band-rows.parquet", "diagnostic-properties.csv")
#: The smoke stream.  It salts every study seed function, so no smoke sample is a declared one.
SMOKE_STREAM = "band-finite-sample/wave2-smoke"
SMOKE_REPLICATES = 20
TIMING_KEYS = 5
STRUCTURAL_FIELDS = (
    "property",
    "cell",
    "role",
    "replicate",
    "n",
    "requested_replicates",
    "failed_replicates",
    "covered",
    "rejected",
)
NUMERIC_FIELDS = ("truth", "estimate", "std_error")
RUNTIME_FIELDS = ("python", "numpy", "scipy", "pandas", "scikit_learn", "joblib")
#: An explicit band option in a study module: a keyword, a dict key or a constant.
PIN_PATTERN = re.compile(
    r"""["']?\b(n_multiplier|multiplier_kind|band_scale|N_MULTIPLIER|BandOptions)\b["']?\s*(=|:|\()"""
)

Key = tuple[str, int]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def is_band_cell(cell: str) -> bool:
    """Whether a ``simultaneous_coverage`` cell's rows read a multiplier band."""
    return cell.endswith(BAND_SUFFIXES)


def band_cells(summary: pd.DataFrame) -> pd.DataFrame:
    """The positive band cells of a committed ``properties.csv``."""
    mask = (
        (summary["property"] == BAND_FAMILY)
        & (summary["role"] == "positive")
        & summary["cell"].str.endswith("__simultaneous_band")
    )
    return summary.loc[mask]


def moved_cells(summary: pd.DataFrame) -> list[str]:
    """Every ``property/cell`` whose rows a band determines, positive or not."""
    rows = summary.loc[(summary["property"] == BAND_FAMILY) & summary["cell"].map(is_band_cell)]
    return sorted(f"{BAND_FAMILY}/{cell}" for cell in rows["cell"])


def band_pins(record: StudyRecord) -> list[str]:
    """Each line of a study module that sets a band option, as ``"<module>: <text>"``.

    The study's own modules are read, not the shared evidence framework or the package, so a
    pin is a choice of the study.  No line number is kept: an unrelated edit must not move it.
    """
    out = set()
    for module in record.modules:
        if not module.endswith(".py") or module.startswith(("src/", "tests/studies/evidence/")):
            continue
        path = ROOT / module
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith("#") or not PIN_PATTERN.search(text):
                continue
            out.add(f"{module}: {text}")
    return sorted(out)


def candidate_records() -> list[StudyRecord]:
    """Every registered study whose committed ``properties.csv`` holds a positive band cell."""
    out = []
    for record in registered():
        path = record.artifacts / "properties.csv"
        if path.exists() and not band_cells(pd.read_csv(path)).empty:
            out.append(record)
    return out


def record_for(slug: str) -> StudyRecord:
    return next(record for record in registered() if record.slug == slug)


# ---------------------------------------------------------------------------- capture


@dataclass
class Call:
    """One ``fit_properties`` call of a study, and the keys of it that write a band row.

    Parameters
    ----------
    label : str
        The call's label.
    keys : list
        Every key of the call.
    selected : list of int
        The positions of the keys whose cells include a band cell.
    options : dict
        The call's other arguments, as the study passed them.
    """

    label: str
    keys: list[Any]
    selected: list[int]
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def cells_of(self) -> Callable[[Any], Sequence[str]]:
        return self.options["cells_of"]

    @property
    def replicate_of(self) -> Callable[[Any], int]:
        return self.options["replicate_of"]

    def controls(self) -> set[str]:
        return {name for names in (self.options.get("derived") or {}).values() for name in names}

    def written(self, key: Any) -> set[str]:
        """The ``property/cell`` names a key's worker returns: its cells but its controls."""
        return set(self.cells_of(key)) - self.controls()


Index = dict[tuple[str, int], list[dict[str, Any]]]


def committed_index(rows: pd.DataFrame) -> Index:
    """The committed rows by ``(property/cell, replicate)``."""
    index: Index = {}
    for row in rows.loc[:, list(REPLICATE_COLUMNS)].to_dict("records"):
        name = f"{row['property']}/{row['cell']}"
        index.setdefault((name, int(row["replicate"])), []).append(row)
    return index


@contextlib.contextmanager
def swapped(name: str, original: Any, replacement: Any) -> Iterator[None]:
    """Bind ``replacement`` wherever a loaded test module binds ``original`` as ``name``."""
    touched = []
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("tests") and (
            getattr(module, name, None) is original
        ):
            setattr(module, name, replacement)
            touched.append(module)
    try:
        yield
    finally:
        for module in touched:
            setattr(module, name, original)


def capture(record: StudyRecord, moved: Iterable[str], index: Index) -> list[Call]:
    """Every ``fit_properties`` call of the study's property phase, without a fit.

    The recorder hands each key its committed rows (wrapped as the study's ``rows_at`` asks),
    so the study's own code runs to the end and names its keys.  A study whose code cannot run
    on those values stops early; :func:`check_selection` then finds any band cell left
    uncovered.
    """
    moved = set(moved)
    module = record.properties()
    calls: list[Call] = []

    def recorder(keys: Sequence[Any], fit_one: Any, **options: Any) -> list[Any]:
        options = {"fit_one": fit_one, **options}
        options.setdefault("label", "properties")
        call = Call(str(options["label"]), list(keys), [], options)
        call.selected = [i for i, key in enumerate(call.keys) if moved & set(call.cells_of(key))]
        calls.append(call)
        rows_at = options.get("rows_at")
        values = []
        for key in call.keys:
            replicate = int(call.replicate_of(key))
            rows = [row for name in call.written(key) for row in index.get((name, replicate), [])]
            value: Any = rows
            if rows_at is not None:
                parts: list[Any] = [None, None] if rows_at < 2 else [None] * (rows_at + 1)
                parts[rows_at] = rows if len(rows) != 1 else rows[0]
                value = tuple(parts)
            values.append(value)
        return values

    with (
        swapped("fit_properties", replication.fit_properties, recorder),
        contextlib.suppress(Exception),
        replication.recording(record.implementation),
    ):
        module.generate_property_rows(n_jobs=1)
    return calls


def selection(calls: Sequence[Call], moved: Iterable[str]) -> list[dict[str, Any]]:
    """Each call's band keys and the band cells they write, as the plan records them."""
    moved = set(moved)
    out = []
    for call in calls:
        if not call.selected:
            continue
        cells: dict[str, int] = {}
        for position in call.selected:
            for name in sorted(moved & call.written(call.keys[position])):
                cells[name] = cells.get(name, 0) + 1
        out.append(
            {
                "label": call.label,
                "keys": len(call.keys),
                "band_keys": len(call.selected),
                "cells": dict(sorted(cells.items())),
            }
        )
    return out


def check_selection(
    chosen: Sequence[Mapping[str, Any]], replicates: Mapping[str, int]
) -> list[str]:
    """Each moved cell that the selected keys do not write exactly once per replicate."""
    found: dict[str, int] = {}
    for call in chosen:
        for name, count in call["cells"].items():
            found[name] = found.get(name, 0) + int(count)
    return [
        f"{name}: {found.get(name, 0)} band keys for {count} replicates"
        for name, count in sorted(replicates.items())
        if found.get(name, 0) != count
    ]


# ---------------------------------------------------------------------------- the plan


def audit_study(record: StudyRecord) -> dict[str, Any]:
    """One study's entry of the plan, from its committed artifacts and its code."""
    directory = record.artifacts
    summary = pd.read_csv(directory / "properties.csv")
    manifest_raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    rows = read_rows(directory / "property-replicates.parquet")
    moved = moved_cells(summary)
    calls = capture(record, moved, committed_index(rows))
    chosen = selection(calls, moved)
    replicates = {
        f"{BAND_FAMILY}/{cell}": int(count)
        for cell, count in zip(summary["cell"], summary["replicates"], strict=True)
        if f"{BAND_FAMILY}/{cell}" in moved
    }
    findings = check_selection(chosen, replicates)
    if findings:
        raise ValueError(f"{record.slug}: the capture does not cover every band cell: {findings}")
    pins = band_pins(record)
    cells = [
        {
            "cell": str(row.cell),
            "n": int(row.n),
            "replicates": int(row.replicates),
            "passed": bool(row.passed),
            "coverage": float(row.coverage),
            "coverage_ci_lower": float(row.coverage_ci_lower),
            "coverage_ci_upper": float(row.coverage_ci_upper),
        }
        for row in band_cells(summary).sort_values("cell").itertuples()
    ]
    carried = rows.loc[
        ~(rows["property"] + "/" + rows["cell"]).isin(set(moved)), list(REPLICATE_COLUMNS)
    ]
    return {
        "artifact_directory": directory.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(manifest_raw),
        "artifact_sha256": manifest["sha256"],
        "baseline_subject": {
            field: manifest["generated_with"]["subject"].get(field) for field in RUNTIME_FIELDS
        },
        "reference": record.reference,
        "publication_policy": record.publication_policy,
        "band_cells": cells,
        "previously_red": [cell["cell"] for cell in cells if not cell["passed"]],
        "moved_cells": moved,
        "pins": pins,
        "pinned_draws": [pin for pin in pins if "N_MULTIPLIER" in pin or "n_replicates" in pin],
        "selection": chosen,
        "band_fits": sum(call["band_keys"] for call in chosen),
        "carried_rows_sha256": digest(csv_text(carried.reset_index(drop=True))),
        "moved_rows": int(len(rows) - len(carried)),
    }


def build_plan() -> dict[str, Any]:
    """The audit: every study the default flip moves, before any fit."""
    studies = {}
    excluded = {}
    for record in candidate_records():
        if record.slug in EXCLUDED:
            excluded[record.slug] = {
                "reason": EXCLUDED[record.slug],
                "pins": band_pins(record),
            }
            continue
        studies[record.slug] = audit_study(record)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "baseline_commit": BASELINE_COMMIT,
        "replay_tolerance": REPLAY_TOLERANCE,
        "replay_rule": REPLAY_RULE,
        "default_band": DEFAULT_BAND,
        "band_suffixes": list(BAND_SUFFIXES),
        "excluded": excluded,
        "studies": studies,
        "totals": totals(studies),
    }
    return json.loads(json_bytes(plan))


def totals(studies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "studies": len(studies),
        "band_cells": sum(len(entry["band_cells"]) for entry in studies.values()),
        "band_fits": sum(int(entry["band_fits"]) for entry in studies.values()),
        "previously_red": sorted(
            f"{slug}:{cell}" for slug, entry in studies.items() for cell in entry["previously_red"]
        ),
        "with_reference": sorted(slug for slug, entry in studies.items() if entry["reference"]),
        "pinning_draws": sorted(slug for slug, entry in studies.items() if entry["pinned_draws"]),
    }


def baseline_manifest_raw(record: StudyRecord) -> bytes:
    """The study's baseline manifest: the committed one, or the one a composite embeds."""
    raw = (record.artifacts / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("generation_mode") == MODE:
        return str(manifest["composite"]["baseline"]["manifest_utf8"]).encode()
    return raw


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """What the plan says that the tree does not, without fitting or capturing anything.

    The studies are every registered study with a positive band cell but the excluded ones;
    each baseline manifest is the one the plan hashed; while a study's committed manifest is
    its baseline, its band cells, R, n and ``passed`` are the committed ones; its pins are its
    modules' pins; and its selection writes each moved cell once per replicate.
    """
    out = []
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("mode") != MODE:
        out.append("wrong plan schema or mode")
    if plan.get("baseline_commit") != BASELINE_COMMIT:
        out.append("wrong baseline commit")
    if plan.get("replay_tolerance") != REPLAY_TOLERANCE or plan.get("replay_rule") != REPLAY_RULE:
        out.append("the replay rule changed after its declaration")
    if plan.get("default_band") != DEFAULT_BAND or plan.get("band_suffixes") != list(BAND_SUFFIXES):
        out.append("the declared band or its cells changed")
    candidates = {record.slug: record for record in candidate_records()}
    studies = plan.get("studies", {})
    if set(studies) != set(candidates) - set(EXCLUDED):
        out.append(
            f"studies differ: missing {sorted(set(candidates) - set(EXCLUDED) - set(studies))}, "
            f"extra {sorted(set(studies) - set(candidates))}"
        )
    if set(plan.get("excluded", {})) != set(EXCLUDED) & set(candidates):
        out.append("the excluded studies differ")
    for slug, entry in plan.get("excluded", {}).items():
        record = candidates.get(slug)
        if record is None:
            continue
        if entry != {"reason": EXCLUDED[slug], "pins": band_pins(record)}:
            out.append(f"{slug}: its exclusion or its pins changed")
        witness = EXCLUSION_WITNESS.get(slug)
        if witness is not None and not any(witness in pin for pin in entry["pins"]):
            out.append(f"{slug}: no explicit band options in its modules")
    for slug, entry in studies.items():
        record = candidates.get(slug)
        if record is None:
            continue
        out += [f"{slug}: {text}" for text in validate_entry(record, entry)]
    if plan.get("totals") != totals(studies):
        out.append("the totals differ from the studies")
    return out


def validate_entry(record: StudyRecord, entry: Mapping[str, Any]) -> list[str]:
    out = []
    raw = baseline_manifest_raw(record)
    if digest(raw) != entry["manifest_sha256"]:
        out.append("the baseline manifest differs from the one the plan hashed")
    if json.loads(raw)["sha256"] != entry["artifact_sha256"]:
        out.append("the baseline artifact hashes differ")
    if entry["artifact_directory"] != record.artifacts.relative_to(ROOT).as_posix():
        out.append("the artifact directory moved")
    if entry["reference"] != record.reference:
        out.append("the reference changed")
    if entry["pins"] != band_pins(record):
        out.append("the band options in the study's modules changed")
    committed = (record.artifacts / "manifest.json").read_bytes()
    if digest(committed) == entry["manifest_sha256"]:
        summary = pd.read_csv(record.artifacts / "properties.csv")
        cells = band_cells(summary).set_index("cell")
        if sorted(cells.index) != [cell["cell"] for cell in entry["band_cells"]]:
            out.append("the band cells differ from properties.csv")
        else:
            for cell in entry["band_cells"]:
                row = cells.loc[cell["cell"]]
                if (int(row["n"]), int(row["replicates"]), bool(row["passed"])) != (
                    cell["n"],
                    cell["replicates"],
                    cell["passed"],
                ):
                    out.append(f"{cell['cell']}: n, R or passed differ from properties.csv")
        if moved_cells(summary) != entry["moved_cells"]:
            out.append("the moved cells differ from properties.csv")
    replicates = {
        f"{BAND_FAMILY}/{cell['cell']}": cell["replicates"] for cell in entry["band_cells"]
    }
    replicates = {name: count for name, count in replicates.items() if name in entry["moved_cells"]}
    if [cell["cell"] for cell in entry["band_cells"] if not cell["passed"]] != entry[
        "previously_red"
    ]:
        out.append("the previously red cells differ from the band cells")
    for name in entry["moved_cells"]:
        if not any(name in call["cells"] for call in entry["selection"]):
            out.append(f"{name}: no band key writes it")
    out += check_selection(entry["selection"], replicates)
    if sum(call["band_keys"] for call in entry["selection"]) != entry["band_fits"]:
        out.append("the band fits differ from the selection")
    return out


# ---------------------------------------------------------------------------- the rerun


class ReplayMiss(RuntimeError):
    """A non-band row of a refitted key did not replay: the study takes a complete rerun."""


def rows_of(value: Any, rows_at: int | None) -> list[dict[str, Any]]:
    """A key's rows on :data:`REPLICATE_COLUMNS`, from what its worker returned."""
    frame = replication._rows_of(value, rows_at)
    return [
        {column: row[column] for column in REPLICATE_COLUMNS} for row in frame.to_dict("records")
    ]


def compare(committed: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, float]:
    """The replay rule on one row: the absolute difference of each numeric field.

    Raises
    ------
    ReplayMiss
        If a structural or discrete field differs, or a numeric one exceeds its bound.
    """
    where = f"{committed['property']}/{committed['cell']}/r{committed['replicate']}"
    for name in STRUCTURAL_FIELDS:
        before, after = committed[name], new[name]
        if name in ("property", "cell", "role"):
            same = str(before) == str(after)
        else:
            same = float(before) == float(after) and float(after).is_integer()
        if not same:
            raise ReplayMiss(f"{where}: {name} {before!r} -> {after!r}")
    out = {}
    for name in NUMERIC_FIELDS:
        before, after = float(committed[name]), float(new[name])
        if not (math.isfinite(before) and math.isfinite(after)):
            if before != after:
                raise ReplayMiss(f"{where}: nonfinite {name} {before!r} -> {after!r}")
            out[name] = 0.0
            continue
        difference = abs(after - before)
        if difference > REPLAY_TOLERANCE * max(1.0, abs(before)):
            raise ReplayMiss(f"{where}: {name} moved by {difference!r}")
        out[name] = difference
    return out


@dataclass
class Refit:
    """The rows of every refitted key: its band rows and its replay comparison."""

    band_rows: list[dict[str, Any]]
    compared: int
    exact: int
    maximum: dict[str, float]
    refused: dict[str, int]


def replay(
    calls: Sequence[Call],
    values: Mapping[str, Sequence[Any]],
    index: Index,
    moved: Iterable[str],
) -> Refit:
    """Hold every non-band row of every refitted key to the committed row (the replay rule).

    ``values[label]`` holds the worker values of the call's selected keys, in their order.
    """
    moved = set(moved)
    band_rows: list[dict[str, Any]] = []
    compared = exact = 0
    maximum = dict.fromkeys(NUMERIC_FIELDS, 0.0)
    refused: dict[str, int] = {}
    for call in calls:
        if not call.selected:
            continue
        fitted = values[call.label]
        if len(fitted) != len(call.selected):
            raise ReplayMiss(f"{call.label}: {len(fitted)} values for {len(call.selected)} keys")
        for position, value in zip(call.selected, fitted, strict=True):
            key = call.keys[position]
            replicate = int(call.replicate_of(key))
            new = rows_of(value, call.options.get("rows_at"))
            names = {f"{row['property']}/{row['cell']}" for row in new}
            if names != call.written(key):
                raise ReplayMiss(f"{call.label} r{replicate}: wrote {sorted(names)}")
            refusal = False
            for row in new:
                name = f"{row['property']}/{row['cell']}"
                if int(row["replicate"]) != replicate:
                    raise ReplayMiss(f"{name}: replicate {row['replicate']} for key {replicate}")
                if name in moved:
                    band_rows.append(row)
                    refusal = refusal or is_refused(row)
                    continue
                matches = index.get((name, replicate), [])
                if len(matches) != 1:
                    raise ReplayMiss(f"{name}/r{replicate}: {len(matches)} committed rows")
                differences = compare(matches[0], row)
                compared += 1
                exact += all(value == 0.0 for value in differences.values())
                for field_name, value in differences.items():
                    maximum[field_name] = max(maximum[field_name], value)
            for name in sorted(moved & names):
                refused[name] = refused.get(name, 0) + int(refusal)
    return Refit(band_rows, compared, exact, maximum, refused)


def is_refused(row: Mapping[str, Any]) -> bool:
    """Whether a band row is the sentinel of a refused band (``simultaneous`` module docstring)."""
    return int(row["covered"]) == 0 and float(row["estimate"]) == float(row["std_error"]) + (
        simultaneous.REFUSED_MARGIN
    )


def splice(committed: pd.DataFrame, band_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The committed rows with each band row replaced in place, every other row untouched."""
    table = committed.loc[:, list(REPLICATE_COLUMNS)].copy()
    position = {
        (str(prop), str(cell), int(rep)): i
        for i, (prop, cell, rep) in enumerate(
            zip(table["property"], table["cell"], table["replicate"], strict=True)
        )
    }
    if len(position) != len(table):
        raise ValueError("the committed rows repeat a key")
    seen = set()
    for row in band_rows:
        key = (str(row["property"]), str(row["cell"]), int(row["replicate"]))
        if key in seen or key not in position:
            raise ValueError(f"band row {key} is new or repeated")
        seen.add(key)
        at = position[key]
        for name in STRUCTURAL_FIELDS[:7]:
            if str(table.iloc[at][name]) != str(row[name]) and not (
                name not in ("property", "cell", "role")
                and float(table.iloc[at][name]) == float(row[name])
            ):
                raise ValueError(f"band row {key} changed {name}")
        for name in ("truth", "estimate", "std_error", "covered", "rejected"):
            table.iat[at, table.columns.get_loc(name)] = row[name]
    return table


def band_mask(rows: pd.DataFrame, moved: Iterable[str]) -> pd.Series:
    return (rows["property"] + "/" + rows["cell"]).isin(set(moved))


def analysis(record: StudyRecord, rows: pd.DataFrame, *, jobs: int) -> pd.DataFrame:
    """The study's own summary of the complete table, as the regeneration driver computes it."""
    module = record.properties()
    summary = module.summarize_properties(rows)
    accounting = property_failures.property_accounting(
        rows, None, summary, record=record, module=module, witnesses=None, n_jobs=jobs
    )
    return property_failures.correct_properties(summary, accounting, module=module)


def run_calls(
    record: StudyRecord, calls: Sequence[Call], *, jobs: int, cache: Path | None
) -> dict[str, list[Any]]:
    """Fit every selected key through ``fit_properties``: its checkpoints and its workers.

    Raises
    ------
    ReplayMiss
        If a selected key's fit fails: the committed run fitted every band key, so the
        inherited rows cannot be spliced.
    """
    values = {}
    with replication.recording(record.implementation) as collected:
        for call in calls:
            if not call.selected:
                continue
            options = {
                name: value for name, value in call.options.items() if name not in ("fit_one",)
            }
            options["n_jobs"] = jobs
            options["checkpoint"] = cache
            values[call.label] = replication.fit_properties(
                [call.keys[i] for i in call.selected], call.options["fit_one"], **options
            )
    failures = collected.failures
    if failures is not None and not failures.empty:
        first = failures.iloc[0]
        raise ReplayMiss(
            f"{len(failures)} band key(s) failed, first {first['group_id']}: {first['message']}"
        )
    return values


# ---------------------------------------------------------------------------- provenance


def operation_record() -> dict[str, Any]:
    """The producer: the subject provenance, the source digests and the installed versions."""
    distributions = sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
    )
    sources = {
        path.relative_to(ROOT).as_posix(): digest(path.read_bytes())
        for path in sorted((ROOT / "src/cleverly").rglob("*.py"))
    }
    sources[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__).read_bytes())
    operation: dict[str, Any] = {
        "subject": provenance(),
        "source_sha256": sources,
        "distribution_versions": distributions,
        "distribution_versions_sha256": digest(json_bytes(distributions)),
    }
    operation["producer_sha256"] = digest(json_bytes(operation))
    return json.loads(json_bytes(operation))


def stack_differences(baseline: Mapping[str, Any], subject: Mapping[str, Any]) -> dict[str, Any]:
    """Each runtime version of the replacement fits that differs from the baseline's."""
    return {
        name: {"baseline": baseline.get(name), "ran": subject.get(name)}
        for name in RUNTIME_FIELDS
        if baseline.get(name) != subject.get(name)
    }


def cell_readings(
    summary: pd.DataFrame, entry: Mapping[str, Any], refit: Refit
) -> list[dict[str, Any]]:
    """Each band cell before and after, with its refused bands, for the record."""
    after = band_cells(summary).set_index("cell")
    out = []
    for cell in entry["band_cells"]:
        row = after.loc[cell["cell"]]
        refused = refit.refused.get(f"{BAND_FAMILY}/{cell['cell']}", 0)
        out.append(
            {
                "cell": cell["cell"],
                "replicates": cell["replicates"],
                "baseline_passed": cell["passed"],
                "baseline_coverage": cell["coverage"],
                "passed": bool(row["passed"]),
                "coverage": float(row["coverage"]),
                "coverage_ci_lower": float(row["coverage_ci_lower"]),
                "coverage_ci_upper": float(row["coverage_ci_upper"]),
                "refused": refused,
                "refused_share": refused / cell["replicates"],
            }
        )
    return out


def attribution(baseline_hashes: Mapping[str, str]) -> dict[str, str]:
    return {
        name: (
            "composite_replicates"
            if name == "property-replicates.parquet"
            else "analysis"
            if name == "properties.csv"
            else "baseline"
        )
        for name in baseline_hashes
    }


def composite_manifest(
    baseline_raw: bytes,
    *,
    plan_raw: bytes,
    slug: str,
    operation: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    composite_raw: bytes,
    refit: Refit,
    readings: list[dict[str, Any]],
    command: Sequence[str],
) -> dict[str, Any]:
    """The baseline manifest, with the targeted record that says what moved and why."""
    baseline = json.loads(baseline_raw)
    plan = json.loads(plan_raw)
    entry = plan["studies"][slug]
    manifest = copy.deepcopy(baseline)
    manifest["sha256"] = dict(artifact_hashes)
    manifest["generation_mode"] = MODE
    manifest["composite"] = {
        "baseline": {
            "commit": plan["baseline_commit"],
            "manifest_utf8": baseline_raw.decode(),
            "manifest_sha256": digest(baseline_raw),
        },
        "operation_plan": {
            "path": PLAN_PATH.relative_to(ROOT).as_posix(),
            "sha256": digest(plan_raw),
        },
        "replacement": {
            **operation,
            "reason": "band-finite-sample PR 2: the default band moved to " + str(DEFAULT_BAND),
            "command": list(command),
            "selection": entry["selection"],
            "band_fits": entry["band_fits"],
            "moved_cells": entry["moved_cells"],
            "replay": replay_record(refit),
            "declared_stack": stack_differences(entry["baseline_subject"], operation["subject"]),
        },
        "analysis": {**operation, "cells": readings},
        "artifact_attribution": attribution(baseline["sha256"]),
        "composite_property_csv_sha256": digest(composite_raw),
        "carried_property_rows_sha256": entry["carried_rows_sha256"],
    }
    return manifest


def replay_record(refit: Refit) -> dict[str, Any]:
    return {
        "rule": REPLAY_RULE,
        "tolerance": REPLAY_TOLERANCE,
        "rows_compared": refit.compared,
        "rows_exact": refit.exact,
        "maximum_absolute_differences": refit.maximum,
        "structural_fields_and_decisions_exact": True,
    }


def carried_digest(rows: pd.DataFrame, moved: Iterable[str]) -> str:
    """The CSV digest of every row a band does not determine."""
    carried = rows.loc[~band_mask(rows, moved), list(REPLICATE_COLUMNS)].reset_index(drop=True)
    return digest(csv_text(carried))


def validate_composite_manifest(manifest: Mapping[str, Any], plan_raw: bytes) -> None:
    """The fast tier's gate on a band-rerun composite: attribution, replay and carried bytes."""
    if manifest.get("generation_mode") != MODE:
        raise ValueError("not a band-rerun composite")
    plan = json.loads(plan_raw)
    slug = manifest["slug"]
    entry = plan["studies"][slug]
    section = manifest["composite"]
    baseline_raw = str(section["baseline"]["manifest_utf8"]).encode()
    baseline = json.loads(baseline_raw)
    if digest(baseline_raw) != entry["manifest_sha256"] or section["baseline"] != {
        "commit": plan["baseline_commit"],
        "manifest_utf8": baseline_raw.decode(),
        "manifest_sha256": entry["manifest_sha256"],
    }:
        raise ValueError("the embedded baseline is not the plan's")
    for name in (
        "generated_with",
        "study_module_sha256",
        "reference_sha256",
        "configuration",
        "study",
        "slug",
        "schema_version",
    ):
        if manifest.get(name) != baseline.get(name):
            raise ValueError(f"historical field restamped: {name}")
    if section["operation_plan"] != {
        "path": PLAN_PATH.relative_to(ROOT).as_posix(),
        "sha256": digest(plan_raw),
    }:
        raise ValueError("the operation plan attribution changed")
    owners = section["artifact_attribution"]
    if owners != attribution(baseline["sha256"]) or set(manifest["sha256"]) != set(owners):
        raise ValueError("artifact attribution or coverage differs")
    for name, owner in owners.items():
        if owner == "baseline" and manifest["sha256"][name] != baseline["sha256"][name]:
            raise ValueError(f"a baseline artifact changed: {name}")
    replacement = section["replacement"]
    for name in ("selection", "band_fits", "moved_cells"):
        if replacement[name] != entry[name]:
            raise ValueError(f"the replacement {name} differs from the plan")
    record = replacement["replay"]
    if (
        record["rule"] != REPLAY_RULE
        or record["tolerance"] != REPLAY_TOLERANCE
        or record["structural_fields_and_decisions_exact"] is not True
        or int(record["rows_compared"]) <= 0
    ):
        raise ValueError("the replay record does not hold the declared rule")
    for name, value in record["maximum_absolute_differences"].items():
        if name not in NUMERIC_FIELDS or not float(value) >= 0.0:
            raise ValueError("the replay differences were altered")
    for name in ("replacement", "analysis"):
        metadata = section[name]
        producer = {
            key: metadata[key]
            for key in (
                "subject",
                "source_sha256",
                "distribution_versions",
                "distribution_versions_sha256",
            )
        }
        if metadata.get("producer_sha256") != digest(json_bytes(producer)):
            raise ValueError(f"the {name} producer digest changed")
        subject = metadata["subject"]
        if subject.get("cleverly_worktree_clean") is not True or (
            len(str(subject.get("cleverly_commit", ""))) != 40
        ):
            raise ValueError(f"the {name} producer is not a clean commit")
    if replacement["declared_stack"] != stack_differences(
        entry["baseline_subject"], replacement["subject"]
    ):
        raise ValueError("the stack differences were altered")
    if section["carried_property_rows_sha256"] != entry["carried_rows_sha256"]:
        raise ValueError("the carried rows receipt is not the baseline's")
    cells = section["analysis"]["cells"]
    if [cell["cell"] for cell in cells] != [cell["cell"] for cell in entry["band_cells"]]:
        raise ValueError("the analysis does not read every band cell")


# ---------------------------------------------------------------------------- the sidecar


def sidecar_record_path(directory: Path) -> Path:
    return directory / SIDECAR / "record.json"


def validate_sidecar(directory: Path, plan: Mapping[str, Any]) -> list[str]:
    """What a blocked run's sidecar says that its files and the plan do not."""
    out = []
    path = sidecar_record_path(directory)
    record = json.loads(path.read_bytes())
    slug = record.get("slug")
    entry = plan["studies"].get(slug)
    if record.get("mode") != MODE or record.get("blocked") is not True or entry is None:
        return [f"{path}: not a blocked band-rerun record of a planned study"]
    if (ROOT / entry["artifact_directory"]).resolve() != directory.resolve():
        out.append(f"{slug}: the sidecar sits in another study's directory")
    for name in SIDECAR_FILES:
        raw = (directory / SIDECAR / name).read_bytes()
        if digest(raw) != record["sha256"].get(name):
            out.append(f"{slug}: {name} differs from its recorded hash")
    if set(record["sha256"]) != set(SIDECAR_FILES):
        out.append(f"{slug}: the record hashes other files")
    committed = (directory / "manifest.json").read_bytes()
    if digest(committed) != entry["manifest_sha256"]:
        out.append(f"{slug}: a blocked run changed the committed manifest")
    if record.get("plan_sha256") is None or record.get("baseline_commit") != BASELINE_COMMIT:
        out.append(f"{slug}: the record names no plan or another baseline")
    return out


# ---------------------------------------------------------------------------- execution


def preflight(plan: Mapping[str, Any], slug: str, *, scientific: bool) -> Mapping[str, Any]:
    """Refuse a run the plan does not cover, or a scientific one from an unclean tree."""
    findings = validate_plan(plan)
    if findings:
        raise SystemExit("refused: the band-rerun plan does not hold: " + "; ".join(findings))
    entry = plan["studies"].get(slug)
    if entry is None:
        raise SystemExit(f"refused: {slug} is not in the band-rerun plan")
    record = record_for(slug)
    if digest((record.artifacts / "manifest.json").read_bytes()) != entry["manifest_sha256"]:
        raise SystemExit(f"refused: {slug} is not at its planned baseline")
    if scientific:
        subject = provenance()
        if subject["cleverly_worktree_clean"] is not True:
            raise SystemExit("refused: a band rerun runs from a clean committed tree")
        blocked = [
            name
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if os.environ.get(name) != "1"
        ]
        if blocked:
            raise SystemExit(f"refused: set {', '.join(blocked)} to 1")
    return entry


def execute(
    slug: str,
    output: Path,
    *,
    jobs: int,
    cache: Path | None,
    blocked: bool,
    plan_raw: bytes,
    command: Sequence[str],
    scientific: bool = True,
) -> dict[str, Any]:
    """The targeted rerun of one study, into ``output``; returns the study's band readings."""
    plan = json.loads(plan_raw)
    entry = preflight(plan, slug, scientific=scientific)
    record = record_for(slug)
    directory = record.artifacts
    baseline_raw = (directory / "manifest.json").read_bytes()
    committed = read_rows(directory / "property-replicates.parquet")
    index = committed_index(committed)
    moved = entry["moved_cells"]
    calls = capture(record, moved, index)
    if selection(calls, moved) != entry["selection"]:
        raise SystemExit(f"refused: {slug}'s band keys differ from the plan")
    print(
        f"{slug}: {entry['band_fits']} band keys in {len(entry['selection'])} call(s)", flush=True
    )
    operation = operation_record()
    started = time.monotonic()
    values = run_calls(record, calls, jobs=jobs, cache=cache)
    refit = replay(calls, values, index, moved)
    composite = splice(committed, refit.band_rows)
    if carried_digest(composite, moved) != entry["carried_rows_sha256"]:
        raise ReplayMiss("the splice moved a row a band does not determine")
    summary = analysis(record, composite, jobs=jobs)
    readings = cell_readings(summary, entry, refit)
    fit_seconds = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    if blocked:
        side = output / SIDECAR
        side.mkdir(exist_ok=True)
        write_rows(
            composite.loc[band_mask(composite, moved)].reset_index(drop=True),
            side / SIDECAR_FILES[0],
        )
        diagnostic = summary.rename(columns={"passed": "diagnostic_passed"})
        diagnostic = diagnostic.rename(columns={"property_passed": "diagnostic_property_passed"})
        write_csv(diagnostic, side / SIDECAR_FILES[1])
        side_record = {
            "mode": MODE,
            "blocked": True,
            "slug": slug,
            "baseline_commit": BASELINE_COMMIT,
            "plan_sha256": digest(plan_raw),
            "note": (
                "Diagnostic only: a blocked Wave 2 leaves every committed artifact byte-identical "
                "(plan section 4.4); no support row and no grid row reads these verdicts."
            ),
            "sha256": {name: digest((side / name).read_bytes()) for name in SIDECAR_FILES},
            "replacement": {
                "subject": operation["subject"],
                "command": list(command),
                "selection": entry["selection"],
                "replay": replay_record(refit),
                "declared_stack": stack_differences(
                    entry["baseline_subject"], operation["subject"]
                ),
                "fit_seconds": fit_seconds,
            },
            "cells": readings,
        }
        (side / "record.json").write_bytes(json_bytes(side_record))
        return {"slug": slug, "blocked": True, "cells": readings}
    for name in json.loads(baseline_raw)["sha256"]:
        if name not in ("property-replicates.parquet", "properties.csv"):
            shutil.copyfile(directory / name, output / name)
    write_rows(composite, output / "property-replicates.parquet")
    composite_raw = csv_text(read_rows(output / "property-replicates.parquet"))
    write_csv(summary, output / "properties.csv")
    hashes = {
        name: digest((output / name).read_bytes()) for name in json.loads(baseline_raw)["sha256"]
    }
    manifest = composite_manifest(
        baseline_raw,
        plan_raw=plan_raw,
        slug=slug,
        operation=operation,
        artifact_hashes=hashes,
        composite_raw=composite_raw,
        refit=refit,
        readings=readings,
        command=command,
    )
    validate_composite_manifest(manifest, plan_raw)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    log = {
        "operation": MODE,
        "command": list(command),
        "source_commit": operation["subject"]["cleverly_commit"],
        "plan_sha256": digest(plan_raw),
        "band_fits": entry["band_fits"],
        "fit_seconds": fit_seconds,
        "artifact_sha256": {
            **hashes,
            "manifest.json": digest((output / "manifest.json").read_bytes()),
        },
        "threads": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    (output / "run.log").write_bytes(json_bytes(log))
    return {"slug": slug, "blocked": False, "cells": readings}


def install(slug: str, output: Path, *, blocked: bool) -> None:
    """Copy a verified run's files into the study directory."""
    directory = record_for(slug).artifacts
    if blocked:
        target = directory / SIDECAR
        target.mkdir(exist_ok=True)
        for name in (*SIDECAR_FILES, "record.json"):
            shutil.copyfile(output / SIDECAR / name, target / name)
        return
    manifest = json.loads((output / "manifest.json").read_bytes())
    for name in (*manifest["sha256"], "manifest.json", "run.log"):
        shutil.copyfile(output / name, directory / name)


# ---------------------------------------------------------------------------- smoke and timing


_ORIGINAL_SEEDS: dict[str, Callable[..., int]] = {
    "stream_seed": seeds.stream_seed,
    "replicate_seed": seeds.replicate_seed,
}
#: How many seeds the salted functions handed out in this process.
SALTED_CALLS = [0]
_WORKER_SALT: list[contextlib.AbstractContextManager[None]] = []


def _salted_stream_seed(record: StudyRecord, *labels: str | int) -> int:
    SALTED_CALLS[0] += 1
    return _ORIGINAL_SEEDS["stream_seed"](record, SMOKE_STREAM, *labels)


def _salted_replicate_seed(record: StudyRecord, scenario: str, replicate: int) -> int:
    SALTED_CALLS[0] += 1
    return _ORIGINAL_SEEDS["stream_seed"](record, SMOKE_STREAM, "replicate", scenario, replicate)


def _salted_pilot_seed(stream: str, replicate: int) -> int:
    from tests import long_horizon_design

    SALTED_CALLS[0] += 1
    original = _ORIGINAL_SEEDS.get("pilot", long_horizon_design.stream_seed)
    return original(f"{SMOKE_STREAM}/{stream}", replicate)


@contextlib.contextmanager
def salted() -> Iterator[None]:
    """Every study seed function draws from :data:`SMOKE_STREAM` instead of its own stream."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            swapped("stream_seed", _ORIGINAL_SEEDS["stream_seed"], _salted_stream_seed)
        )
        stack.enter_context(
            swapped("replicate_seed", _ORIGINAL_SEEDS["replicate_seed"], _salted_replicate_seed)
        )
        if "tests.long_horizon_design" in sys.modules:
            from tests import long_horizon_design

            original = long_horizon_design.stream_seed
            _ORIGINAL_SEEDS.setdefault("pilot", original)
            stack.enter_context(swapped("stream_seed", original, _salted_pilot_seed))
        yield


def _salt_worker(slug: str) -> None:
    """A smoke worker's initializer: import the study, then salt its seeds for good."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    record_for(slug).properties()
    # Held for the worker's life: a dropped context manager is collected and unsalts the seeds.
    _WORKER_SALT.append(salted())
    _WORKER_SALT[-1].__enter__()


def _smoke_key(task: tuple[Any, Any, Any]) -> dict[str, Any]:
    """Fit one band key in a salted worker, with the band observer on."""
    fit_one, draw, key = task
    before = SALTED_CALLS[0]
    simultaneous.BAND_OBSERVER = []
    started = time.process_time()
    wall = time.monotonic()
    try:
        drawn = None if draw is None else draw(key)
        value = fit_one(key, drawn)
        error = None
    except Exception as raised:  # a smoke reports every failure instead of stopping
        value, error = None, f"{type(raised).__name__}: {raised}"
    observed = simultaneous.BAND_OBSERVER
    simultaneous.BAND_OBSERVER = None
    return {
        "value": value,
        "error": error,
        "observed": observed,
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
        "salted_calls": SALTED_CALLS[0] - before,
    }


def smoke(
    slug: str,
    *,
    jobs: int,
    replicates: int = SMOKE_REPLICATES,
    keys_per_study: int | None = None,
) -> dict[str, Any]:
    """Fit the first band keys of each band cell on the smoke stream, and read their bands.

    ``replicates`` keys per band cell for ``--smoke``; ``keys_per_study`` keys in all for
    ``--timing``.  Nothing is written into the repository.
    """
    record = record_for(slug)
    plan = json.loads(PLAN_PATH.read_bytes())
    entry = plan["studies"][slug]
    moved = set(entry["moved_cells"])
    committed = read_rows(record.artifacts / "property-replicates.parquet")
    index = committed_index(committed)
    declared = capture(record, moved, index)
    with salted():
        calls = capture(record, moved, index)
    tasks: list[tuple[Call, Any]] = []
    taken: dict[str, int] = {}
    for call in calls:
        for position in call.selected:
            key = call.keys[position]
            names = moved & call.written(key)
            if keys_per_study is not None:
                if len(tasks) >= keys_per_study:
                    break
            elif all(taken.get(name, 0) >= replicates for name in names):
                continue
            for name in names:
                taken[name] = taken.get(name, 0) + 1
            tasks.append((call, key))
    same_keys = sum(
        1
        for (call, key), plain in zip(tasks, _declared_keys(declared, tasks), strict=True)
        if key == plain
    )
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=max(1, jobs), initializer=_salt_worker, initargs=(slug,)
    ) as pool:
        results = list(
            pool.map(
                _smoke_key,
                [(call.options["fit_one"], call.options.get("draw"), key) for call, key in tasks],
            )
        )
    wall = time.monotonic() - started
    cells: dict[str, dict[str, Any]] = {}
    unverified = 0
    for (call, key), plain, result in zip(
        tasks, _declared_keys(declared, tasks), results, strict=True
    ):
        if key == plain and result["salted_calls"] == 0:
            unverified += 1
        cell_names = sorted(moved & call.written(key))
        rows = (
            [] if result["value"] is None else rows_of(result["value"], call.options.get("rows_at"))
        )
        for name in cell_names:
            cell = cells.setdefault(
                name,
                {
                    "keys": 0,
                    "failed": 0,
                    "refused": 0,
                    "out_of_range_levels": 0,
                    "kish_min": math.inf,
                    "cpu_seconds": 0.0,
                    "covered": 0,
                    "options": set(),
                    "errors": [],
                },
            )
            cell["keys"] += 1
            cell["cpu_seconds"] += result["cpu_seconds"]
            if result["error"] is not None:
                cell["failed"] += 1
                cell["errors"].append(result["error"][:300])
                continue
            band = [row for row in rows if f"{row['property']}/{row['cell']}" == name]
            cell["refused"] += int(any(is_refused(row) for row in band))
            cell["covered"] += int(all(int(row["covered"]) == 1 for row in band))
            label = name.split("/", 1)[1].rsplit("__", 1)[0]
            for item in result["observed"]:
                if item["label"] != label:
                    continue
                cell["out_of_range_levels"] += int(item["out_of_range"])
                cell["kish_min"] = min(cell["kish_min"], float(item["kish_min"]))
                cell["options"].add((item["band_scale"], item["kind"], item["n_replicates"]))
    report = {}
    for name, cell in sorted(cells.items()):
        keys = cell["keys"]
        report[name] = {
            "keys": keys,
            "failed": cell["failed"],
            "refused": cell["refused"],
            "refused_share": cell["refused"] / keys if keys else math.nan,
            "out_of_range_levels": cell["out_of_range_levels"],
            "kish_min": None if math.isinf(cell["kish_min"]) else round(cell["kish_min"], 1),
            "seconds_per_key": cell["cpu_seconds"] / keys if keys else math.nan,
            "covered_share": cell["covered"] / max(1, keys - cell["failed"]),
            "band_options": sorted(map(list, cell["options"])),
            "errors": cell["errors"][:3],
        }
    cpu = sum(result["cpu_seconds"] for result in results)
    per_key = cpu / max(1, len(results))
    return {
        "slug": slug,
        "stream": SMOKE_STREAM,
        "keys": len(results),
        "keys_identical_to_declared": same_keys,
        "unverified_salt": unverified,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "cpu_seconds_per_key": per_key,
        "band_fits": entry["band_fits"],
        "estimated_cpu_hours": per_key * entry["band_fits"] / 3600.0,
        "cells": report,
    }


def replay_check(slug: str, *, jobs: int, keys: int = TIMING_KEYS) -> dict[str, Any]:
    """Refit the first band keys on their declared seeds and hold their other rows to the rule.

    A dry run of the replay half of a rerun: nothing is spliced or written.
    """
    record = record_for(slug)
    entry = json.loads(PLAN_PATH.read_bytes())["studies"][slug]
    index = committed_index(read_rows(record.artifacts / "property-replicates.parquet"))
    calls = capture(record, entry["moved_cells"], index)
    left = keys
    for call in calls:
        call.selected = call.selected[: max(0, left)]
        left -= len(call.selected)
    started = time.process_time()
    values = run_calls(record, calls, jobs=jobs, cache=None)
    refit = replay(calls, values, index, entry["moved_cells"])
    return {
        "slug": slug,
        "keys": keys - max(0, left),
        "cpu_seconds_parent": time.process_time() - started,
        **replay_record(refit),
        "band_rows": len(refit.band_rows),
        "refused": refit.refused,
    }


def _declared_keys(declared: Sequence[Call], tasks: Sequence[tuple[Call, Any]]) -> list[Any]:
    """The unsalted key at the same position as each smoke task's key."""
    by_label = {call.label: call for call in declared}
    out = []
    for call, key in tasks:
        plain = by_label[call.label]
        out.append(plain.keys[call.keys.index(key)] if key in call.keys else None)
    return out


# ---------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", help="one study slug from the plan")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cache", type=Path, help="checkpoint directory; a rerun resumes from it")
    parser.add_argument("--output", type=Path, help="scratch output outside the repository")
    parser.add_argument("--blocked", action="store_true", help="write the band_rerun/ sidecar only")
    parser.add_argument("--install", action="store_true", help="copy the verified output in")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--replay-check", action="store_true", help="refit five declared band keys and replay"
    )
    parser.add_argument("--report", type=Path, help="where --smoke or --timing writes its JSON")
    parser.add_argument("--list", action="store_true", help="print each study's band keys")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--check-plan", action="store_true")
    args = parser.parse_args(argv)
    if args.write_plan:
        PLAN_PATH.write_bytes(json_bytes(build_plan()))
        print(f"wrote {PLAN_PATH.relative_to(ROOT)}")
        return 0
    plan_raw = PLAN_PATH.read_bytes()
    plan = json.loads(plan_raw)
    if args.check_plan:
        findings = validate_plan(plan)
        print("\n".join(findings) or "the band-rerun plan holds")
        return 1 if findings else 0
    if args.list:
        for slug, entry in plan["studies"].items():
            if args.study and slug != args.study:
                continue
            for call in entry["selection"]:
                print(
                    f"{slug}\t{call['label']}\t{call['band_keys']}/{call['keys']}\t{call['cells']}"
                )
        return 0
    if args.study is None or args.jobs < 1:
        parser.error("--study and a positive --jobs are required")
    if args.replay_check:
        try:
            print(json.dumps(replay_check(args.study, jobs=args.jobs), indent=2, sort_keys=True))
        except ReplayMiss as miss:
            print(f"replay miss: {miss}", file=sys.stderr)
            return 2
        return 0
    if args.smoke or args.timing:
        report = smoke(
            args.study,
            jobs=args.jobs,
            keys_per_study=TIMING_KEYS if args.timing else None,
        )
        text = json.dumps(report, indent=2, sort_keys=True, default=str)
        if args.report is not None:
            if args.report.resolve().is_relative_to(ROOT.resolve()):
                parser.error("--report writes outside the repository")
            args.report.write_text(text + "\n")
        print(text)
        return 0
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        parser.error("--output must lie outside the repository")
    started = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        result = execute(
            args.study,
            output,
            jobs=args.jobs,
            cache=args.cache,
            blocked=args.blocked,
            plan_raw=plan_raw,
            command=[Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
        )
    except ReplayMiss as miss:
        output.mkdir(parents=True, exist_ok=True)
        (output / "replay-miss.json").write_bytes(
            json_bytes({"study": args.study, "started": started, "miss": str(miss)})
        )
        print(f"replay miss: {miss}; {args.study} takes a complete rerun", file=sys.stderr)
        return 2
    if args.install:
        install(args.study, output, blocked=args.blocked)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
