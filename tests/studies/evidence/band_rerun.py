"""The audited targeted rerun of every band cell the default band moves (band-finite-sample Wave 2).

PR 2 of band-finite-sample moved the default band to the Mammen multiplier, 10,000 draws and the
logit scale.  No estimate, standard error or pointwise interval moves, and the band's multiplier
draws its own generator (``multiplier.py``), so only the rows a band determines change.  This
module rebuilds exactly those rows and keeps every other published byte (plan section 8).

**Audit.**  :func:`build_plan` writes ``tests/canonical/band-rerun-plan.json`` before any fit.
For each registered study with a positive band cell it reads the committed ``properties.csv``
(every moved cell with its role, R, n and ``passed``), scans the study's modules for explicit
band options (:func:`band_pins`), and lists the property keys whose cells include a moved cell
(:func:`capture`).  A study counts only when the flip moves its band rows: ``longitudinal-sdr``
builds its band on explicit options (:data:`EXCLUDED`), and a cell that reads no multiplier band
stays out of the moved cells (:data:`NOT_MOVED`).  :func:`validate_plan` is the fast test's check
of the plan against the tree.

**Capture.**  A study's ``generate_property_rows`` runs with
:func:`~tests.studies.evidence.replication.fit_properties` swapped for a recorder.  The recorder
keeps each call's keys, worker, draw and ``cells_of``, selects the keys whose cells include a
moved cell, and hands the study back the committed rows of every key, every column included, so
nothing is fitted and the study's own code names the keys.

**Fit.**  :func:`execute` fits the selected keys with their committed seeds and the new
defaults through :func:`~tests.studies.evidence.replication.fit_replicates`, into a checkpoint
cache nested by study, commit and plan (:func:`cache_directory`).  A key that fails is
classified from its traceback (:func:`classify`): a failure with a band entry point on its stack
is a Wave 2 reading, its band rows not covered, which enters its cell's interval test; any other
failure, and any failure the harness records (a draw included), is drift.  :func:`replay` holds
every non-band row of a refitted key, and every column of it, to the committed row: numeric
fields within :data:`REPLAY_TOLERANCE`, absolute or relative, whichever is larger, and the
discrete fields exactly.  Drift stops the study (:class:`ReplayMiss`), which then takes a
complete rerun.  :func:`splice` replaces only the value fields of the band rows,
:func:`post_fit` re-summarizes with the study's own code and holds every ``properties.csv`` row
outside the moved cells to the baseline within :data:`SUMMARY_TOLERANCE`, and
:func:`composite_manifest` records the baseline, the selection, the replay, the refused rows
and the producer.  Primary, reference and equivalence artifacts stay byte-identical, so no R
container runs.

**The wave is one unit.**  Every planned study fits into one wave directory from one commit.
``--decide`` reads them all, applies the section 8 rule that the plan binds before the wave
(``decision_rule``: ``strict`` or ``paired``; :func:`decide`), and writes ``decision.json``.
While the plan's rule is ``null`` no wave fit and no decision runs.  ``--install-wave`` re-validates every output
and installs either every composite or, when the flip is blocked, every diagnostic sidecar
(``tests/canonical/<study>/band_rerun/``), which no support or grid row reads.

**A replay miss takes a complete rerun.**  A study whose targeted run missed holds only
``WAVE/<slug>/replay-miss.json``.  It reruns whole with its own regeneration driver at the
wave's commit, and that output is copied as is to ``WAVE/<slug>/complete/``
(:func:`validate_complete`).  ``--decide`` reads it as route ``complete_rerun`` beside the
targeted studies (route ``targeted``).  Every moved band cell is read from the rows against
the baseline rows as for a targeted study, paired over the union of replicates (a replicate one
side lost is not covered there); the runtime versions that differ from the baseline's are
recorded, and every other verdict that changed is listed under ``complete_rerun_changes``
without blocking.  On a flip ``--install-wave`` installs the complete
artifacts as the regeneration wrote them, with ``band_rerun/complete-rerun.json`` keeping the
baseline manifest the plan hashed; on a block they join the study's sidecar, labelled
diagnostic.

**Smoke and timing.**  ``--smoke`` fits the first :data:`SMOKE_REPLICATES` band keys of each
moved cell on :data:`SMOKE_STREAM`, which salts every study seed function and is never the
declared stream, and reports per cell the smallest Kish count of a banded curve, the share of
refused band rows, the out-of-range levels and the seconds per key.  ``--timing`` fits
:data:`TIMING_KEYS_PER_STRATUM` keys of each stratum of keys that write the same moved cells and
weights each stratum's seconds per key by its band keys.  ``--replay-check`` refits
:data:`REPLAY_CHECK_KEYS` band keys on their declared seeds and applies the replay rule.  None of
them writes into the repository.

Every fitting command refuses to start unless each of :data:`THREAD_VARIABLES` is ``1`` in its
environment, so the variables precede every numpy import in the driver and in its workers::

    python -m tests.studies.evidence.band_rerun --study <slug> --jobs N --cache DIR --output WAVE
    python -m tests.studies.evidence.band_rerun --decide WAVE
    python -m tests.studies.evidence.band_rerun --install-wave WAVE
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binom

from tests.canonical.regenerate import ARTIFACT_NAMES as REGENERATION_ARTIFACTS
from tests.canonical.regenerate import PROPERTY_FAILURE_FILES
from tests.studies.evidence import failures, property_failures, replication, seeds, simultaneous
from tests.studies.evidence.manifest import csv_text, provenance, read_rows, write_csv, write_rows
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

MODE = "targeted_band_rerun"
#: The commit whose committed artifacts and ``passed`` columns the plan reads.
BASELINE_COMMIT = "cef2aa65ec023744523a4b239f2517e025516ed4"
#: The producer of every wave output descends from the commit that stated the new default.
FLIP_COMMIT = "62c966e982757cb8f2cb18a60727781d0a10c19d"
PLAN_PATH = ROOT / "tests/canonical/band-rerun-plan.json"
SCHEMA_VERSION = 2
#: Declared before any fit (plan section 8), as ``seed_repair.REPLAY_TOLERANCE``.
REPLAY_TOLERANCE = 1e-9
REPLAY_RULE = (
    "every non-band row of a refitted key: abs(new - committed) <= 1e-9 * max(1, abs(committed)) "
    "on truth, estimate, std_error and every other numeric column; property, cell, role, "
    "replicate, n, requested_replicates, failed_replicates, covered, rejected and every "
    "non-numeric column exactly.  A band row keeps every field but truth, estimate, std_error, "
    "covered and rejected"
)
#: How far a re-summarized ``properties.csv`` row outside the moved cells may move.
SUMMARY_TOLERANCE = 1e-12
SUMMARY_EXACT = ("passed", "property_passed")
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
}
#: The text that shows a study's band options are explicit, for each excluded study.
EXCLUSION_WITNESS = {
    "longitudinal-sdr": 'BandOptions(band_scale="logit", kind="mammen", n_replicates=10_000',
}
#: Band-suffixed cells that read no multiplier band, by study.  Their rows must replay.
NOT_MOVED = {
    "long-horizon-categorical-ltmle": {
        f"{BAND_FAMILY}/k5__simultaneous_band": (
            "the gated cell reads the studentized bootstrap sup-t band: its statistic is the "
            "identity-scale max-t statistic of the pointwise control and its critical value the "
            "bootstrap's (long_horizon_properties.band_rows), so no multiplier option reaches it"
        ),
    },
}
SIDECAR = "band_rerun"
SIDECAR_FILES = (
    "band-rows.parquet",
    "diagnostic-properties.csv",
    "band-rerun-decision.json",
    "band-rerun-output-manifest.json",
)
DECISION = "decision.json"
RECORD = "band-rerun-record.json"
RUN_LOG = "run.log"
MISS = "replay-miss.json"
#: A study that missed: its complete rerun's output directory, beside its ``replay-miss.json``.
COMPLETE = "complete"
TARGETED_ROUTE = "targeted"
COMPLETE_ROUTE = "complete_rerun"
#: The record a flip installs beside a complete rerun's artifacts: the baseline it replaced.
COMPLETE_RECORD = "complete-rerun.json"
#: The prefix of a complete rerun's artifacts in a blocked study's sidecar.
COMPLETE_PREFIX = "complete-"
VERDICT_LABELS = {
    "passed": "diagnostic_passed",
    "property_passed": "diagnostic_property_passed",
    "gate_passed": "diagnostic_gate_passed",
}
#: The runner's record of a complete rerun's exit, ``{"returncode": 0}``, beside its output.
#: A study that publishes verdicts needs it: its driver refuses after writing on a failed gate.
EXIT_RECORD = "exit.json"
#: The configuration key every property run writes and a baseline written before it lacks;
#: its presence shows the property phase ran (``regenerate._property_failure_bound``).
PROPERTY_BOUND_KEY = property_failures.PROPERTY_BOUND_KEY
PRIMARY_BOUND_KEY = failures.BOUND_KEY
#: The rule each band cell answers to (``property_verdicts.simultaneous_coverage_verdicts``):
#: a positive cell's 99% Clopper-Pearson interval inside :data:`COVERAGE_BAND`, a control's
#: upper limit below ``1 - CONTROL_ALPHA``, a diagnostic cell no verdict.
COVERAGE_CONFIDENCE = 0.99
CONTROL_ALPHA = 0.05
#: The section 8 rule: the closed coverage band of a positive cell at 99% Clopper-Pearson.
COVERAGE_BAND = (0.92, 0.98)
DECISION_ALPHA = 0.01
RULES = ("strict", "paired")
#: The smoke stream.  It salts every study seed function, so no smoke sample is a declared one.
SMOKE_STREAM = "band-finite-sample/wave2-smoke"
SMOKE_REPLICATES = 20
TIMING_KEYS_PER_STRATUM = 2
REPLAY_CHECK_KEYS = 5
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
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
#: The fields a band row may change.
VALUE_FIELDS = ("truth", "estimate", "std_error", "covered", "rejected")
NUMERIC_FIELDS = ("truth", "estimate", "std_error")
RUNTIME_FIELDS = ("python", "numpy", "scipy", "pandas", "scikit_learn", "joblib")
#: An explicit band option in a study module: a keyword, a dict key or a constant.
PIN_PATTERN = re.compile(
    r"""["']?\b(n_multiplier|multiplier_kind|band_scale|N_MULTIPLIER|BandOptions)\b["']?\s*(=|:|\()"""
)
#: A pin that fixes the number of draws to a literal.
DRAWS_PIN = re.compile(
    r"""["']?\b(N_MULTIPLIER|n_multiplier|n_replicates)\b["']?\s*[=:]\s*[0-9_]+"""
)
#: The frames of a traceback that belong to the band.
#: Every frame of this module is the band.
BAND_MODULE = "cleverly.inference.multiplier"
#: The band entry points outside it, as ``(module, function)``: the harness's band readers and
#: the estimators' own band builders.  ``bootstrap_band`` and a study's fit wrappers such as
#: ``dose_response_properties.band_fit`` are not band entry points.
BAND_ENTRY_POINTS = frozenset(
    {
        (BAND_MODULE, "simultaneous_bands"),
        (BAND_MODULE, "median_bands"),
        (BAND_MODULE, "multiplier_critical_value"),
        (BAND_MODULE, "_band_endpoints"),
        (BAND_MODULE, "_refuse_out_of_range"),
        ("tests.studies.evidence.simultaneous", "joint_coverage_rows"),
        ("tests.studies.evidence.simultaneous", "scale_coverage_rows"),
        ("tests.studies.evidence.simultaneous", "refused_critical_value"),
        ("cleverly.longitudinal.estimator", "_bands"),
        ("cleverly.longitudinal.estimator", "_project_bands"),
        ("cleverly.longitudinal.structural", "_bands"),
        ("cleverly.study", "_narrow_bands"),
    }
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def thread_findings() -> list[str]:
    """Each thread variable that is not ``1``."""
    return [name for name in THREAD_VARIABLES if os.environ.get(name) != "1"]


def require_threads() -> None:
    blocked = thread_findings()
    if blocked:
        raise SystemExit(f"refused: export {', '.join(f'{name}=1' for name in blocked)} first")


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


def moved_cells(summary: pd.DataFrame, slug: str) -> list[str]:
    """Every ``property/cell`` whose rows a multiplier band determines, of any role."""
    rows = summary.loc[(summary["property"] == BAND_FAMILY) & summary["cell"].map(is_band_cell)]
    names = {f"{BAND_FAMILY}/{cell}" for cell in rows["cell"]}
    return sorted(names - set(NOT_MOVED.get(slug, {})))


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


def failure_files(record: StudyRecord) -> list[str]:
    """The property failure files a study commits; a planned study commits none."""
    names = (
        property_failures.PROPERTY_FAILURES,
        property_failures.PROPERTY_ACCOUNTING,
        property_failures.PROPERTY_WITNESSES,
    )
    return [name for name in names if (record.artifacts / name).exists()]


# ---------------------------------------------------------------------------- capture


@dataclass
class Call:
    """One ``fit_properties`` call of a study, and the keys of it that write a moved cell.

    Parameters
    ----------
    label : str
        The call's label.
    keys : list
        Every key of the call.
    selected : list of int
        The positions of the keys whose cells include a moved cell.
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
    """The committed rows, every column, by ``(property/cell, replicate)``."""
    index: Index = {}
    for row in rows.to_dict("records"):
        name = f"{row['property']}/{row['cell']}"
        index.setdefault((name, int(row["replicate"])), []).append(row)
    return index


def committed_value(call: Call, key: Any, index: Index) -> Any:
    """What the key's worker returned in the committed run, rebuilt from its rows."""
    replicate = int(call.replicate_of(key))
    rows = [row for name in call.written(key) for row in index.get((name, replicate), [])]
    rows_at = call.options.get("rows_at")
    if rows_at is None:
        return rows
    parts: list[Any] = [None, None] if rows_at < 2 else [None] * (rows_at + 1)
    parts[rows_at] = rows if len(rows) != 1 else rows[0]
    return tuple(parts)


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
        # A module imported inside the block bound ``replacement`` through a swapped module.
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("tests") and (
                getattr(module, name, None) is replacement
            ):
                touched.append(module)
        for module in touched:
            setattr(module, name, original)


def capture(record: StudyRecord, moved: Iterable[str], index: Index) -> list[Call]:
    """Every ``fit_properties`` call of the study's property phase, without a fit.

    The recorder hands each key its committed rows (wrapped as the study's ``rows_at`` asks),
    so the study's own code runs to the end and names its keys.  A study whose code cannot run
    on those values stops early; :func:`check_selection` then finds any moved cell left
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
        return [committed_value(call, key, index) for key in call.keys]

    with (
        swapped("fit_properties", replication.fit_properties, recorder),
        contextlib.suppress(Exception),
        replication.recording(record.implementation),
    ):
        module.generate_property_rows(n_jobs=1)
    return calls


def selection(calls: Sequence[Call], moved: Iterable[str]) -> list[dict[str, Any]]:
    """Each call's band keys and the moved cells they write, as the plan records them."""
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


def cell_reading(row: Any) -> dict[str, Any]:
    """One ``properties.csv`` row of a moved cell, as the plan and the decision read it."""

    def number(value: Any) -> float | None:
        return None if pd.isna(value) else float(value)

    passed = None if pd.isna(row["passed"]) else bool(row["passed"])
    return {
        "cell": str(row["cell"]),
        "role": str(row["role"]),
        "n": int(row["n"]),
        "replicates": int(row["replicates"]),
        "passed": passed,
        "coverage": number(row["coverage"]),
        "coverage_ci_lower": number(row["coverage_ci_lower"]),
        "coverage_ci_upper": number(row["coverage_ci_upper"]),
    }


def moved_readings(summary: pd.DataFrame, moved: Iterable[str]) -> list[dict[str, Any]]:
    names = set(moved)
    rows = summary.loc[(summary["property"] + "/" + summary["cell"]).isin(names)]
    return [cell_reading(row) for _, row in rows.sort_values("cell").iterrows()]


def audit_study(record: StudyRecord) -> dict[str, Any]:
    """One study's entry of the plan, from its committed artifacts and its code."""
    directory = record.artifacts
    summary = pd.read_csv(directory / "properties.csv")
    manifest_raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    rows = read_rows(directory / "property-replicates.parquet")
    moved = moved_cells(summary, record.slug)
    calls = capture(record, moved, committed_index(rows))
    chosen = selection(calls, moved)
    readings = moved_readings(summary, moved)
    replicates = {f"{BAND_FAMILY}/{cell['cell']}": cell["replicates"] for cell in readings}
    findings = check_selection(chosen, replicates)
    if findings:
        raise ValueError(f"{record.slug}: the capture does not cover every moved cell: {findings}")
    pins = band_pins(record)
    return {
        "artifact_directory": directory.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(manifest_raw),
        "artifact_sha256": manifest["sha256"],
        "baseline_subject": {
            field: manifest["generated_with"]["subject"].get(field) for field in RUNTIME_FIELDS
        },
        "reference": record.reference,
        "publication_policy": record.publication_policy,
        "property_artifacts": list(record.property_artifacts),
        "failure_files": failure_files(record),
        "cells": readings,
        "previously_red": [
            cell["cell"] for cell in readings if cell["role"] == "positive" and not cell["passed"]
        ],
        "moved_cells": moved,
        "not_moved": NOT_MOVED.get(record.slug, {}),
        "pins": pins,
        "pinned_draws": [pin for pin in pins if DRAWS_PIN.search(pin)],
        "selection": chosen,
        "band_fits": sum(call["band_keys"] for call in chosen),
        "carried_rows_sha256": carried_digest(rows, moved),
        "moved_rows": int(band_mask(rows, moved).sum()),
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
        "flip_commit": FLIP_COMMIT,
        "replay_tolerance": REPLAY_TOLERANCE,
        "replay_rule": REPLAY_RULE,
        "summary_tolerance": SUMMARY_TOLERANCE,
        "default_band": DEFAULT_BAND,
        "band_suffixes": list(BAND_SUFFIXES),
        "decision_rule": bound_rule(),
        "excluded": excluded,
        "studies": studies,
        "totals": totals(studies),
    }
    return json.loads(json_bytes(plan))


def bound_rule() -> str | None:
    """The committed plan's ``decision_rule``, which a rebuilt plan keeps: the owner sets it."""
    if not PLAN_PATH.exists():
        return None
    rule = json.loads(PLAN_PATH.read_bytes()).get("decision_rule")
    return rule if rule in RULES else None


def totals(studies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "studies": len(studies),
        "moved_cells": sum(len(entry["moved_cells"]) for entry in studies.values()),
        "positive_band_cells": sum(
            1
            for entry in studies.values()
            for cell in entry["cells"]
            if cell["role"] == "positive" and cell["cell"].endswith("__simultaneous_band")
        ),
        "band_fits": sum(int(entry["band_fits"]) for entry in studies.values()),
        "previously_red": sorted(
            f"{slug}:{cell}" for slug, entry in studies.items() for cell in entry["previously_red"]
        ),
        "with_reference": sorted(slug for slug, entry in studies.items() if entry["reference"]),
        "pinning_draws": sorted(slug for slug, entry in studies.items() if entry["pinned_draws"]),
    }


def baseline_manifest_raw(record: StudyRecord) -> bytes:
    """The study's baseline manifest: the committed one, or the one a composite embeds, or the
    one an installed complete rerun's record keeps (:func:`installed_complete`)."""
    raw = (record.artifacts / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("generation_mode") == MODE:
        return str(manifest["composite"]["baseline"]["manifest_utf8"]).encode()
    complete = installed_complete(record.artifacts)
    if complete is not None:
        return str(complete["baseline"]["manifest_utf8"]).encode()
    return raw


def installed_complete(directory: Path) -> dict[str, Any] | None:
    """The record of a complete rerun a flip installed in ``directory``, while it holds.

    The record holds while the committed manifest is the one it installed; ``None`` when no
    record exists.

    Raises
    ------
    ValueError
        If a record exists but the committed manifest is another one.
    """
    path = directory / SIDECAR / COMPLETE_RECORD
    if not path.exists():
        return None
    record = json.loads(path.read_bytes())
    if digest((directory / "manifest.json").read_bytes()) != record.get("manifest_sha256"):
        raise ValueError(f"{path}: the committed manifest is not the complete rerun it installed")
    return dict(record)


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """What the plan says that the tree does not, without fitting or capturing anything.

    The studies are every registered study with a positive band cell but the excluded ones;
    each baseline manifest is the one the plan hashed; while a study's committed manifest is
    its baseline, its moved cells, R, n and ``passed`` are the committed ones; its pins are its
    modules' pins; and its selection writes each moved cell once per replicate.
    """
    out = []
    expected = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "baseline_commit": BASELINE_COMMIT,
        "flip_commit": FLIP_COMMIT,
        "replay_tolerance": REPLAY_TOLERANCE,
        "replay_rule": REPLAY_RULE,
        "summary_tolerance": SUMMARY_TOLERANCE,
        "default_band": DEFAULT_BAND,
        "band_suffixes": list(BAND_SUFFIXES),
    }
    for name, value in expected.items():
        if plan.get(name) != value:
            out.append(f"the plan's {name} differs from its declaration")
    if "decision_rule" not in plan or plan["decision_rule"] not in (None, *RULES):
        out.append(f"the plan's decision_rule must be null or one of {RULES}")
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
    if entry["property_artifacts"] or list(record.property_artifacts):
        out.append("a study with property artifacts takes a complete rerun")
    if entry["pins"] != band_pins(record):
        out.append("the band options in the study's modules changed")
    if entry["pinned_draws"] != [pin for pin in entry["pins"] if DRAWS_PIN.search(pin)]:
        out.append("the draw pins differ from the pins")
    if entry["not_moved"] != NOT_MOVED.get(record.slug, {}):
        out.append("the cells kept out of the moved cells changed")
    committed = (record.artifacts / "manifest.json").read_bytes()
    if digest(committed) == entry["manifest_sha256"]:
        if entry["failure_files"] or failure_files(record):
            out.append("a study with property failure files takes a complete rerun")
        summary = pd.read_csv(record.artifacts / "properties.csv")
        if moved_cells(summary, record.slug) != entry["moved_cells"]:
            out.append("the moved cells differ from properties.csv")
        elif moved_readings(summary, entry["moved_cells"]) != entry["cells"]:
            out.append("the moved cells' role, n, R, coverage or passed differ from properties.csv")
    if [cell["cell"] for cell in entry["cells"]] != sorted(
        name.split("/", 1)[1] for name in entry["moved_cells"]
    ):
        out.append("the readings do not cover the moved cells")
    red = [c["cell"] for c in entry["cells"] if c["role"] == "positive" and not c["passed"]]
    if red != entry["previously_red"]:
        out.append("the previously red cells differ from the readings")
    replicates = {f"{BAND_FAMILY}/{cell['cell']}": cell["replicates"] for cell in entry["cells"]}
    out += check_selection(entry["selection"], replicates)
    if sum(call["band_keys"] for call in entry["selection"]) != entry["band_fits"]:
        out.append("the band fits differ from the selection")
    return out


# ---------------------------------------------------------------------------- the rows


class ReplayMiss(RuntimeError):
    """Drift outside the band: the study takes a complete rerun."""


@dataclass(frozen=True)
class KeyFailure:
    """A band key whose refit raised, with the side of the band its traceback shows.

    Parameters
    ----------
    inside_band : bool
        Whether a frame of the traceback belongs to the band (:func:`classify`).
    message : str
        The error's type and text.
    frames : tuple of str
        ``module:function`` of every frame, outermost first.
    """

    inside_band: bool
    message: str
    frames: tuple[str, ...]


def classify(error: BaseException) -> KeyFailure:
    """Whether a refit's failure arose inside the band.

    The failure is inside the band when a band entry point (:data:`BAND_ENTRY_POINTS`, or any
    frame of :data:`BAND_MODULE`) is on the stack of the raise: a frame of the traceback, which
    runs from the worker down to the raise site.
    """
    frames = []
    inside = False
    for frame, _ in traceback.walk_tb(error.__traceback__):
        module = str(frame.f_globals.get("__name__", ""))
        name = frame.f_code.co_name
        frames.append(f"{module}:{name}")
        if module == BAND_MODULE or (module, name) in BAND_ENTRY_POINTS:
            inside = True
    return KeyFailure(inside, f"{type(error).__name__}: {error}", tuple(frames))


@dataclass(frozen=True)
class Guarded:
    """A study worker that returns a :class:`KeyFailure` in place of raising.

    Parameters
    ----------
    fit_one : Callable
        The study's ``fit_one(key, drawn)``.
    """

    fit_one: Callable[[Any, Any], Any]

    def __call__(self, key: Any, drawn: Any) -> Any:
        try:
            return self.fit_one(key, drawn)
        except Exception as error:  # every failure is classified, none stops the wave
            return classify(error)


def rows_of(value: Any, rows_at: int | None) -> list[dict[str, Any]]:
    """A key's rows, every column, from what its worker returned (as ``replication._rows_of``)."""
    rows = value if rows_at is None else value[rows_at]
    if isinstance(rows, pd.DataFrame):
        return list(rows.to_dict("records"))
    if isinstance(rows, Mapping):
        return [dict(rows)]
    return [dict(row) for row in rows]


def _call_rows_valid(call: Call, fitted: Sequence[Any]) -> bool:
    """Whether every row the call's keys returned passes the row rules, checked at once."""
    rows = [
        row
        for value in fitted
        if not isinstance(value, KeyFailure)
        for row in rows_of(value, call.options.get("rows_at"))
    ]
    if not rows:
        return True
    extra = call.options.get("extra")
    return replication.invalid_rows(rows, coverage=False) is None and (
        extra is None or extra(pd.DataFrame(rows)) is None
    )


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _numeric(value: Any) -> bool:
    return (isinstance(value, int | float) and not isinstance(value, bool)) or (
        hasattr(value, "dtype") and getattr(value.dtype, "kind", "") in "iuf"
    )


def field_difference(name: str, before: Any, after: Any, *, tolerance: float) -> float | None:
    """The absolute difference of one field, or ``None`` when the rule refuses it."""
    if _missing(before) or _missing(after):
        return 0.0 if _missing(before) and _missing(after) else None
    if _numeric(before) and _numeric(after):
        b, a = float(before), float(after)
        if not (math.isfinite(b) and math.isfinite(a)):
            return 0.0 if b == a else None
        difference = abs(a - b)
        return difference if difference <= tolerance * max(1.0, abs(b)) else None
    return 0.0 if str(before) == str(after) else None


def compare(
    committed: Mapping[str, Any], new: Mapping[str, Any], *, free: Iterable[str] = ()
) -> dict[str, float]:
    """The replay rule on one row: the absolute difference of each numeric field.

    ``free`` names the fields the rule does not hold (a band row's value fields).

    Raises
    ------
    ReplayMiss
        If a structural or discrete field differs, or a numeric one exceeds its bound.
    """
    where = f"{committed['property']}/{committed['cell']}/r{committed['replicate']}"
    free = set(free)
    out: dict[str, float] = {}
    for name in STRUCTURAL_FIELDS:
        if name in free:
            continue
        before, after = committed[name], new.get(name)
        if name in ("property", "cell", "role"):
            same = str(before) == str(after)
        else:
            same = after is not None and float(before) == float(after) == int(after)
        if not same:
            raise ReplayMiss(f"{where}: {name} {before!r} -> {after!r}")
    for name in [*NUMERIC_FIELDS, *(c for c in committed if c not in REPLICATE_COLUMNS)]:
        if name in free:
            continue
        before, after = committed[name], new.get(name)
        difference = field_difference(name, before, after, tolerance=REPLAY_TOLERANCE)
        if difference is None:
            raise ReplayMiss(f"{where}: {name} {before!r} -> {after!r}")
        key = name if name in NUMERIC_FIELDS else "other_columns"
        out[key] = max(out.get(key, 0.0), difference)
    extra = set(new) - set(committed)
    if extra:
        raise ReplayMiss(f"{where}: the refit wrote new columns {sorted(extra)}")
    return out


@dataclass
class Refit:
    """The rows of every refitted key: its band rows and its replay comparison."""

    band_rows: list[dict[str, Any]]
    compared: int
    exact: int
    maximum: dict[str, float]
    refused: dict[str, int]
    band_failures: dict[str, int] = field(default_factory=dict)
    failure_messages: list[str] = field(default_factory=list)
    baseline_lost: int = 0


def is_refused(row: Mapping[str, Any]) -> bool:
    """Whether a band row is the sentinel of a refused band (``simultaneous`` module docstring)."""
    return int(row["covered"]) == 0 and float(row["estimate"]) == float(row["std_error"]) + (
        simultaneous.REFUSED_MARGIN
    )


def _failed_band_row(committed: Mapping[str, Any]) -> dict[str, Any]:
    """The band row of a key whose band failed: the committed row, not covered."""
    return {**committed, "covered": 0, "rejected": 1}


def replay(
    calls: Sequence[Call],
    values: Mapping[str, Sequence[Any]],
    index: Index,
    moved: Iterable[str],
) -> Refit:
    """Hold every non-band row of every refitted key to the committed row (the replay rule).

    ``values[label]`` holds the worker values of the call's selected keys, in their order.  A
    :class:`KeyFailure` inside the band gives the key's band rows not covered, each counted in
    ``band_failures``; one outside the band is drift.  A key the baseline lost must fail again.
    """
    moved = set(moved)
    out = Refit([], 0, 0, dict.fromkeys((*NUMERIC_FIELDS, "other_columns"), 0.0), {})
    for call in calls:
        if not call.selected:
            continue
        fitted = values[call.label]
        if len(fitted) != len(call.selected):
            raise ReplayMiss(f"{call.label}: {len(fitted)} values for {len(call.selected)} keys")
        extra = call.options.get("extra")
        # The rules run once over the call; a key is checked alone only when the call breaks one.
        valid = _call_rows_valid(call, fitted)
        for position, value in zip(call.selected, fitted, strict=True):
            key = call.keys[position]
            replicate = int(call.replicate_of(key))
            written = call.written(key)
            committed_rows = {name: index.get((name, replicate), []) for name in sorted(written)}
            lost = all(not rows for rows in committed_rows.values())
            if lost:
                if not isinstance(value, KeyFailure):
                    raise ReplayMiss(f"{call.label} r{replicate}: a key the baseline lost fits")
                out.baseline_lost += 1
                continue
            if any(len(rows) != 1 for rows in committed_rows.values()):
                raise ReplayMiss(f"{call.label} r{replicate}: not one committed row per cell")
            if isinstance(value, KeyFailure):
                if not value.inside_band:
                    raise ReplayMiss(
                        f"{call.label} r{replicate} failed outside the band: {value.message}"
                    )
                out.failure_messages.append(f"{call.label} r{replicate}: {value.message}")
                for name in sorted(moved & written):
                    out.band_rows.append(_failed_band_row(committed_rows[name][0]))
                    out.band_failures[name] = out.band_failures.get(name, 0) + 1
                continue
            new = rows_of(value, call.options.get("rows_at"))
            names = {f"{row['property']}/{row['cell']}" for row in new}
            if names != written or len(new) != len(written):
                raise ReplayMiss(f"{call.label} r{replicate}: wrote {sorted(names)}")
            broken = None if valid else replication.invalid_rows(new, coverage=False)
            if not valid and broken is None and extra is not None:
                broken = extra(pd.DataFrame(new))
            if broken is not None:
                plain = [row for row in new if f"{row['property']}/{row['cell']}" not in moved]
                if plain and replication.invalid_rows(plain, coverage=False) is not None:
                    raise ReplayMiss(f"{call.label} r{replicate}: {broken}")
                out.failure_messages.append(f"{call.label} r{replicate}: invalid band row {broken}")
                for name in sorted(moved & written):
                    out.band_rows.append(_failed_band_row(committed_rows[name][0]))
                    out.band_failures[name] = out.band_failures.get(name, 0) + 1
                continue
            for row in new:
                name = f"{row['property']}/{row['cell']}"
                if int(row["replicate"]) != replicate:
                    raise ReplayMiss(f"{name}: replicate {row['replicate']} for key {replicate}")
                committed = committed_rows[name][0]
                if name in moved:
                    compare(committed, row, free=VALUE_FIELDS)
                    out.band_rows.append(row)
                    out.refused[name] = out.refused.get(name, 0) + int(is_refused(row))
                    continue
                differences = compare(committed, row)
                out.compared += 1
                out.exact += all(value == 0.0 for value in differences.values())
                for field_name, difference in differences.items():
                    out.maximum[field_name] = max(out.maximum[field_name], difference)
    return out


def splice(committed: pd.DataFrame, band_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The committed rows, every column, with only the value fields of each band row replaced."""
    table = committed.copy()
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
        for name in VALUE_FIELDS:
            column = table.columns.get_loc(name)
            before = table.iat[at, column]
            after = row[name]
            if name in ("covered", "rejected"):
                after = type(before)(int(after)) if not isinstance(before, float) else float(after)
            table.iat[at, column] = after
    return table


def band_mask(rows: pd.DataFrame, moved: Iterable[str]) -> pd.Series:
    return (rows["property"] + "/" + rows["cell"]).isin(set(moved))


def carried_digest(rows: pd.DataFrame, moved: Iterable[str]) -> str:
    """The CSV digest of every row a band does not determine, every column."""
    return digest(csv_text(rows.loc[~band_mask(rows, moved)].reset_index(drop=True)))


# ---------------------------------------------------------------------------- the summary


class SummaryMiss(ReplayMiss):
    """A ``properties.csv`` row outside the moved cells moved."""


def summary_text(summary: pd.DataFrame) -> str:
    buffer = io.StringIO()
    write_csv(summary, buffer)  # type: ignore[arg-type]
    return buffer.getvalue()


def _parse(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), low_memory=False)


def compose_summary(
    baseline_text: str, new_text: str, moved: Iterable[str]
) -> tuple[str, dict[str, Any]]:
    """The published ``properties.csv``: each row that did not move keeps its baseline bytes.

    Every row outside the moved cells must agree with the baseline: numeric columns within
    :data:`SUMMARY_TOLERANCE`, absolute or relative, ``passed`` and ``property_passed`` and
    every other column exactly.  A moved cell's row that agrees too keeps its baseline line, so
    a rerun whose bands did not move publishes the baseline bytes.

    Raises
    ------
    SummaryMiss
        If a row outside the moved cells moved, or the tables differ in shape.
    """
    moved = set(moved)
    newline = replication_newline(baseline_text)
    base_lines = baseline_text.split(newline)
    new_lines = new_text.split(newline)
    before, after = _parse(baseline_text), _parse(new_text)
    if list(before.columns) != list(after.columns) or base_lines[0] != new_lines[0]:
        raise SummaryMiss("the re-summarized properties.csv has other columns")
    keys = list(zip(before["property"], before["cell"], strict=True))
    if keys != list(zip(after["property"], after["cell"], strict=True)):
        raise SummaryMiss("the re-summarized properties.csv has other rows")
    if len(base_lines) != len(before) + 2 or len(new_lines) != len(after) + 2:
        raise SummaryMiss("a properties.csv field holds a line break")
    largest = 0.0
    lines = [base_lines[0]]
    changed = []
    for i, (prop, cell) in enumerate(keys):
        name = f"{prop}/{cell}"
        same = True
        for column in before.columns:
            exact = column in SUMMARY_EXACT
            difference = field_difference(
                column,
                before.iat[i, before.columns.get_loc(column)],
                after.iat[i, after.columns.get_loc(column)],
                tolerance=0.0 if exact else SUMMARY_TOLERANCE,
            )
            if difference is None:
                if name not in moved:
                    raise SummaryMiss(f"properties.csv {name}: {column} moved")
                same = False
            elif name not in moved:
                largest = max(largest, difference)
        lines.append(base_lines[i + 1] if same else new_lines[i + 1])
        if not same:
            changed.append(name)
    lines.append("")
    return newline.join(lines), {
        "summary_tolerance": SUMMARY_TOLERANCE,
        "maximum_carried_difference": largest,
        "changed_rows": changed,
    }


def replication_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


# ---------------------------------------------------------------------------- post fit


@dataclass
class Composite:
    """What a study's fits publish: the rows, the summary and their records."""

    rows: pd.DataFrame
    summary: str
    refit: Refit
    readings: list[dict[str, Any]]
    summary_record: dict[str, Any]
    #: The digest of the published rows' CSV text, set when they are written.
    csv_sha256: str = ""


def analysis(record: StudyRecord, rows: pd.DataFrame, *, jobs: int) -> pd.DataFrame:
    """The study's own summary of the complete table, as the regeneration driver computes it."""
    module = record.properties()
    summary = module.summarize_properties(rows)
    accounting = property_failures.property_accounting(
        rows, None, summary, record=record, module=module, witnesses=None, n_jobs=jobs
    )
    return property_failures.correct_properties(summary, accounting, module=module)


def post_fit(
    record: StudyRecord,
    entry: Mapping[str, Any],
    committed: pd.DataFrame,
    calls: Sequence[Call],
    values: Mapping[str, Sequence[Any]],
    *,
    jobs: int,
    summarize: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    index: Index | None = None,
) -> Composite:
    """Replay, splice and re-summarize: every step after the fits.

    ``summarize`` replaces :func:`analysis` for a test that has shown the spliced rows are the
    committed rows; a wave always runs the study's own summary.
    """
    moved = entry["moved_cells"]
    index = committed_index(committed) if index is None else index
    refit = replay(calls, values, index, moved)
    rows = splice(committed, refit.band_rows)
    if carried_digest(rows, moved) != entry["carried_rows_sha256"]:
        raise ReplayMiss("the splice moved a row a band does not determine")
    with (record.artifacts / "properties.csv").open(encoding="utf-8", newline="") as handle:
        baseline_text = handle.read()
    text, summary_record = compose_summary(
        baseline_text,
        summary_text(analysis(record, rows, jobs=jobs) if summarize is None else summarize(rows)),
        moved,
    )
    readings = moved_readings(_parse(text), moved)
    for reading in readings:
        name = f"{BAND_FAMILY}/{reading['cell']}"
        reading["refused_rows"] = refit.refused.get(name, 0)
        reading["band_failures"] = refit.band_failures.get(name, 0)
        reading["refused_share"] = reading["refused_rows"] / reading["replicates"]
    return Composite(rows, text, refit, readings, summary_record)


def write_outputs(composite: Composite, record: StudyRecord, output: Path) -> dict[str, str]:
    """Write the composite artifacts, every other baseline artifact copied; their hashes."""
    names = json.loads(baseline_manifest_raw(record))["sha256"]
    for name in names:
        if name not in ("property-replicates.parquet", "properties.csv"):
            shutil.copyfile(record.artifacts / name, output / name)
    source = record.artifacts / "property-replicates.parquet"
    text = csv_text(composite.rows)
    if text == csv_text(read_rows(source)):
        # Rows that did not move keep their bytes: a Parquet file also records its writer.
        shutil.copyfile(source, output / "property-replicates.parquet")
        composite.csv_sha256 = digest(text)
    else:
        write_rows(composite.rows, output / "property-replicates.parquet")
        composite.csv_sha256 = digest(csv_text(read_rows(output / "property-replicates.parquet")))
    (output / "properties.csv").write_bytes(composite.summary.encode("utf-8"))
    return {name: digest((output / name).read_bytes()) for name in names}


# ---------------------------------------------------------------------------- fitting


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def descends_from_flip(commit: str) -> bool:
    return git("merge-base", "--is-ancestor", FLIP_COMMIT, commit).returncode == 0


def cache_directory(cache: Path, slug: str, commit: str, plan_sha: str) -> Path:
    """``<cache>/<slug>/<commit>/<plan_sha>``, stamped; a cache with another stamp is refused."""
    directory = cache / slug / commit / plan_sha
    stamp = {
        "mode": MODE,
        "slug": slug,
        "commit": commit,
        "plan_sha256": plan_sha,
        "default_band": DEFAULT_BAND,
    }
    path = directory / "stamp.json"
    if path.exists():
        if json.loads(path.read_bytes()) != stamp:
            raise SystemExit(f"refused: the cache {directory} was stamped for another run")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json_bytes(stamp))
    return directory


def run_calls(
    record: StudyRecord, calls: Sequence[Call], *, jobs: int, cache: Path | None
) -> dict[str, list[Any]]:
    """Fit every selected key, each failure returned as a :class:`KeyFailure`."""
    values = {}
    for call in calls:
        if not call.selected:
            continue
        outcomes = replication.fit_replicates(
            [call.keys[i] for i in call.selected],
            Guarded(call.options["fit_one"]),
            n_jobs=jobs,
            label=call.label,
            implementation=record.implementation,
            cell_of=call.options["cell_of"],
            replicate_of=call.replicate_of,
            draw=call.options.get("draw"),
            seed_of=call.options.get("seed_of"),
            checkpoint=cache,
        )
        if not outcomes.failures.empty:
            first = outcomes.failures.iloc[0]
            raise ReplayMiss(
                f"{call.label}: {len(outcomes.failures)} recorded failure(s), first "
                f"{first['group_id']}: {first['message']}"
            )
        if any(value is None for value in outcomes.fitted):
            raise ReplayMiss(f"{call.label}: a key returned no value")
        values[call.label] = list(outcomes.fitted)
    return values


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


def replay_record(refit: Refit) -> dict[str, Any]:
    return {
        "rule": REPLAY_RULE,
        "tolerance": REPLAY_TOLERANCE,
        "rows_compared": refit.compared,
        "rows_exact": refit.exact,
        "maximum_absolute_differences": refit.maximum,
        "structural_fields_and_decisions_exact": True,
        "band_failures": refit.band_failures,
        "failure_messages": refit.failure_messages[:20],
        "baseline_lost_keys": refit.baseline_lost,
    }


def composite_manifest(
    baseline_raw: bytes,
    *,
    plan_raw: bytes,
    slug: str,
    operation: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    composite: Composite,
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
            "replay": replay_record(composite.refit),
            "declared_stack": stack_differences(entry["baseline_subject"], operation["subject"]),
        },
        "analysis": {**operation, "cells": composite.readings, **composite.summary_record},
        "artifact_attribution": attribution(baseline["sha256"]),
        "composite_property_csv_sha256": composite.csv_sha256 or digest(csv_text(composite.rows)),
        "carried_property_rows_sha256": entry["carried_rows_sha256"],
    }
    return manifest


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
        if name not in (*NUMERIC_FIELDS, "other_columns") or not float(value) >= 0.0:
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
    if section["replacement"]["subject"] != section["analysis"]["subject"]:
        raise ValueError("the replacement and the analysis name different producers")
    if replacement["declared_stack"] != stack_differences(
        entry["baseline_subject"], replacement["subject"]
    ):
        raise ValueError("the stack differences were altered")
    if section["carried_property_rows_sha256"] != entry["carried_rows_sha256"]:
        raise ValueError("the carried rows receipt is not the baseline's")
    analysis_record = section["analysis"]
    if analysis_record.get("summary_tolerance") != SUMMARY_TOLERANCE or not (
        float(analysis_record.get("maximum_carried_difference", math.inf)) <= SUMMARY_TOLERANCE
    ):
        raise ValueError("the summary record does not hold the declared tolerance")
    cells = analysis_record["cells"]
    if [cell["cell"] for cell in cells] != [cell["cell"] for cell in entry["cells"]]:
        raise ValueError("the analysis does not read every moved cell")


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
        if plan.get("decision_rule") is None:
            raise SystemExit("refused: the plan binds no decision_rule yet; no wave fit runs")
        require_threads()
        subject = provenance()
        if subject["cleverly_worktree_clean"] is not True:
            raise SystemExit("refused: a band rerun runs from a clean committed tree")
        if not descends_from_flip(str(subject["cleverly_commit"])):
            raise SystemExit(f"refused: the producer commit does not descend from {FLIP_COMMIT}")
    return entry


def execute(
    slug: str,
    output: Path,
    *,
    jobs: int,
    cache: Path | None,
    plan_raw: bytes,
    command: Sequence[str],
) -> dict[str, Any]:
    """The targeted rerun of one study, into ``output``; returns the study's band readings.

    Nothing in the repository changes: ``--decide`` and ``--install-wave`` act on the wave.
    """
    plan = json.loads(plan_raw)
    entry = preflight(plan, slug, scientific=True)
    record = record_for(slug)
    output.mkdir(parents=True, exist_ok=True)
    (output / MISS).unlink(missing_ok=True)
    baseline_raw = (record.artifacts / "manifest.json").read_bytes()
    committed = read_rows(record.artifacts / "property-replicates.parquet")
    moved = entry["moved_cells"]
    calls = capture(record, moved, committed_index(committed))
    if selection(calls, moved) != entry["selection"]:
        raise SystemExit(f"refused: {slug}'s band keys differ from the plan")
    print(
        f"{slug}: {entry['band_fits']} band keys in {len(entry['selection'])} call(s)", flush=True
    )
    operation = operation_record()
    commit = str(operation["subject"]["cleverly_commit"])
    directory = None
    if cache is not None:
        directory = cache_directory(cache.resolve(), slug, commit, digest(plan_raw))
    started = time.monotonic()
    values = run_calls(record, calls, jobs=jobs, cache=directory)
    fit_seconds = time.monotonic() - started
    composite = post_fit(record, entry, committed, calls, values, jobs=jobs)
    hashes = write_outputs(composite, record, output)
    manifest = composite_manifest(
        baseline_raw,
        plan_raw=plan_raw,
        slug=slug,
        operation=operation,
        artifact_hashes=hashes,
        composite=composite,
        command=command,
    )
    validate_composite_manifest(manifest, plan_raw)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    log = {
        "operation": MODE,
        "slug": slug,
        "command": list(command),
        "source_commit": commit,
        "plan_sha256": digest(plan_raw),
        "band_fits": entry["band_fits"],
        "fit_seconds": fit_seconds,
        "cache": None if directory is None else str(directory),
        "artifact_sha256": {
            **hashes,
            "manifest.json": digest((output / "manifest.json").read_bytes()),
        },
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }
    (output / RUN_LOG).write_bytes(json_bytes(log))
    return {"slug": slug, "cells": composite.readings, "summary": composite.summary_record}


# ---------------------------------------------------------------------------- the wave


def mcnemar_less(b: int, c: int) -> float:
    """The exact one-sided McNemar p-value that the new band covers less.

    ``b`` pairs were covered before and not now, ``c`` the reverse.  Under the null of equal
    coverage ``b`` is binomial on ``b + c`` discordant pairs with probability one half, and the
    p-value is ``P(B >= b)``.
    """
    if b + c == 0:
        return 1.0
    return float(binom.sf(b - 1, b + c, 0.5))


def _covered_by_replicate(rows: pd.DataFrame, name: str) -> pd.Series:
    prop, cell = name.split("/", 1)
    rows = rows.loc[(rows["property"] == prop) & (rows["cell"] == cell)]
    return rows.set_index("replicate")["covered"].astype(int)


def discordant(before: pd.DataFrame, after: pd.DataFrame, name: str) -> tuple[int, int, int]:
    """``(b, c, pairs)`` of one cell, by replicate; the replicate sets must agree."""
    old, new = _covered_by_replicate(before, name), _covered_by_replicate(after, name)
    if sorted(old.index) != sorted(new.index):
        raise ValueError(f"{name}: the replicates differ")
    new = new.loc[old.index]
    b = int(((old == 1) & (new == 0)).sum())
    c = int(((old == 0) & (new == 1)).sum())
    return b, c, len(old)


def union_reading(
    before: pd.DataFrame, after: pd.DataFrame, name: str, role: str
) -> dict[str, Any]:
    """A complete rerun's reading of one moved cell from its rows, as a targeted reading counts.

    The trials are the union of both sides' replicates (:func:`discordant_union`), and a
    replicate the rerun lacks is not covered, as a band key that failed inside the band is in a
    targeted reading.  The coverage, its 99% Clopper-Pearson interval and ``passed`` follow
    ``property_verdicts.simultaneous_coverage_verdicts``.

    Raises
    ------
    ValueError
        For a role whose verdict reads more than the cell's coverage (a scale control).
    """
    from tests.studies.evidence.inference import clopper_pearson

    new = _covered_by_replicate(after, name)
    union = sorted(set(_covered_by_replicate(before, name).index) | set(new.index))
    hits = int(new.reindex(union, fill_value=0).sum())
    interval = clopper_pearson(hits, len(union), confidence_level=COVERAGE_CONFIDENCE)
    if role == "positive":
        passed = COVERAGE_BAND[0] <= interval.low and interval.high <= COVERAGE_BAND[1]
    elif role == "control":
        passed = interval.high < 1.0 - CONTROL_ALPHA
    elif role == "diagnostic":
        passed = True
    else:
        raise ValueError(f"{name}: a {role} cell's verdict is not read from its coverage alone")
    return {
        "coverage": hits / len(union),
        "coverage_ci_lower": float(interval.low),
        "coverage_ci_upper": float(interval.high),
        "passed": bool(passed),
    }


def discordant_union(
    before: pd.DataFrame, after: pd.DataFrame, name: str
) -> tuple[int, int, int, int, int]:
    """``(b, c, pairs, lost_now, lost_before)`` of one cell of a complete rerun.

    The pairs run over the union of both sides' replicates.  A replicate absent on one side is
    not covered on that side, so a covered baseline replicate the rerun lost adds to ``b``.
    ``lost_now`` counts the replicates the rerun lacks, ``lost_before`` those the baseline lacks.
    """
    old, new = _covered_by_replicate(before, name), _covered_by_replicate(after, name)
    if old.index.has_duplicates or new.index.has_duplicates:
        raise ValueError(f"{name}: a replicate repeats")
    union = sorted(set(old.index) | set(new.index))
    lost_now = len(set(old.index) - set(new.index))
    lost_before = len(set(new.index) - set(old.index))
    old = old.reindex(union, fill_value=0)
    new = new.reindex(union, fill_value=0)
    b = int(((old == 1) & (new == 0)).sum())
    c = int(((old == 0) & (new == 1)).sum())
    return b, c, len(union), lost_now, lost_before


def interval_red(reading: Mapping[str, Any]) -> bool | None:
    """The closed-interval test of a positive band cell: lower < 0.92 or upper > 0.98."""
    lower, upper = reading["coverage_ci_lower"], reading["coverage_ci_upper"]
    if lower is None or upper is None:
        return None
    return bool(lower < COVERAGE_BAND[0] or upper > COVERAGE_BAND[1])


def validate_output(slug: str, output: Path, plan_raw: bytes) -> dict[str, Any]:
    """Re-validate one study's wave output; returns its run log."""
    if (output / COMPLETE).exists():
        raise ValueError(f"{slug}: the output holds both a composite and a complete rerun")
    findings = []
    manifest = json.loads((output / "manifest.json").read_bytes())
    log = json.loads((output / RUN_LOG).read_bytes())
    validate_composite_manifest(manifest, plan_raw)
    if manifest["slug"] != slug or log["slug"] != slug:
        findings.append("the output names another study")
    for name, expected in manifest["sha256"].items():
        if digest((output / name).read_bytes()) != expected:
            findings.append(f"{name} differs from its manifest hash")
    hashes = {name: digest((output / name).read_bytes()) for name in log["artifact_sha256"]}
    if hashes != log["artifact_sha256"]:
        findings.append("the run log's hashes differ from the files")
    commit = manifest["composite"]["replacement"]["subject"]["cleverly_commit"]
    if log["source_commit"] != commit:
        findings.append("the run log and the manifest name different commits")
    if log["plan_sha256"] != digest(plan_raw):
        findings.append("the output ran under another plan")
    if (output / MISS).exists():
        findings.append("the output holds a replay miss")
    if findings:
        raise ValueError(f"{slug}: " + "; ".join(findings))
    return log


def validate_complete(slug: str, output: Path, plan_raw: bytes) -> dict[str, Any]:
    """Re-validate a study's complete rerun, the declared response to its replay miss.

    ``output`` holds the targeted run's ``replay-miss.json`` and, under ``complete/``, the
    output of the study's own regeneration driver, copied as is.  The study is planned, the
    miss names it, the directory holds no composite, and ``complete/manifest.json`` is a normal
    regeneration manifest of the study from a clean committed tree whose every artifact matches
    its hash.  It is a normal run of the property phase: its configuration is the baseline's
    but for :data:`PROPERTY_BOUND_KEY`, which every property run writes and so must be present;
    its ``property-replicates.parquet`` and ``properties.csv`` are not the baseline's bytes (a
    ``--skip-properties`` run copies them); and each moved cell requests the baseline's
    replicates.  Its exit is clean: an ``exit.json`` beside the output, read whenever present,
    holds ``returncode`` 0, and a study that publishes verdicts must have one.  Returns the
    route, the producer commit and the digests the decision records.

    The miss file is the targeted run's record, written at the wave's commit; it is hashed into
    the decision and not otherwise interpreted.
    """
    if slug not in json.loads(plan_raw)["studies"]:
        raise ValueError(f"{slug}: not planned")
    if (output / "manifest.json").exists() or (output / RUN_LOG).exists():
        raise ValueError(f"{slug}: the output holds both a composite and a complete rerun")
    if not (output / MISS).exists():
        raise ValueError(f"{slug}: a complete rerun answers a replay miss, and none is recorded")
    findings = []
    miss_raw = (output / MISS).read_bytes()
    if json.loads(miss_raw).get("study") != slug:
        findings.append("the replay miss names another study")
    complete = output / COMPLETE
    manifest_raw = (complete / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if "generation_mode" in manifest or "composite" in manifest:
        findings.append("the complete manifest is not a normal regeneration manifest")
    if manifest.get("slug") != slug:
        findings.append("the complete rerun names another study")
    hashes = manifest.get("sha256") or {}
    absent = sorted(set(REGENERATION_ARTIFACTS) - set(hashes))
    if absent:
        findings.append(f"the complete manifest hashes no {absent}")
    for name, expected in sorted(hashes.items()):
        path = complete / name
        if Path(name).name != name or not path.is_file():
            findings.append(f"{name} is not a file of the complete rerun")
        elif digest(path.read_bytes()) != expected:
            findings.append(f"{name} differs from its manifest hash")
    subject = (manifest.get("generated_with") or {}).get("subject") or {}
    commit = str(subject.get("cleverly_commit", ""))
    if len(commit) != 40:
        findings.append("the complete rerun names no commit")
    if subject.get("cleverly_worktree_clean") is not True:
        findings.append("the complete rerun ran from an unclean tree")
    if findings:
        raise ValueError(f"{slug}: " + "; ".join(findings))
    entry = json.loads(plan_raw)["studies"][slug]
    artifacts = record_for(slug).artifacts
    baseline_raw = (artifacts / "manifest.json").read_bytes()
    if digest(baseline_raw) != entry["manifest_sha256"]:
        raise ValueError(f"{slug}: the committed artifacts are not the baseline")
    findings += normal_run_findings(
        json.loads(baseline_raw), manifest, dict(record_for(slug).failure_bound)
    )
    exit_path = complete / EXIT_RECORD
    if exit_path.exists():
        if json.loads(exit_path.read_bytes()).get("returncode") != 0:
            findings.append("the complete rerun's driver exited with an error")
    elif (manifest.get("configuration") or {}).get("publication_policy") != "reporting":
        findings.append(f"a study that publishes verdicts needs {EXIT_RECORD} with returncode 0")
    if not findings:
        before = read_rows(artifacts / "property-replicates.parquet")
        after = read_rows(complete / "property-replicates.parquet")
        for name in entry["moved_cells"]:
            if requested(before, name) != requested(after, name):
                findings.append(f"{name} requests other replicates than the baseline")
    if findings:
        raise ValueError(f"{slug}: " + "; ".join(findings))
    return {
        "route": COMPLETE_ROUTE,
        "source_commit": commit,
        "manifest_sha256": digest(manifest_raw),
        "miss_sha256": digest(miss_raw),
    }


def normal_run_findings(
    baseline: Mapping[str, Any],
    manifest: Mapping[str, Any],
    declared_failure_bound: Mapping[str, float] | None = None,
) -> list[str]:
    """What shows a complete rerun is not a normal run of the baseline's declaration.

    Two failure-accounting keys may appear in the rerun's configuration although the baseline
    predates them: ``property_failure_bound``, which every property phase now writes, and the
    primary ``failure_bound``, which must then equal the bound the study's record declares at
    the run commit (``declared_failure_bound``).  Neither changes a fit or a row; each bounds
    how many failed replicates a verdict tolerates.
    """
    out = []
    old = dict(baseline.get("configuration") or {})
    new = dict(manifest.get("configuration") or {})
    if PRIMARY_BOUND_KEY in new and PRIMARY_BOUND_KEY not in old:
        if dict(new[PRIMARY_BOUND_KEY]) != dict(declared_failure_bound or {}):
            out.append(f"the complete rerun's {PRIMARY_BOUND_KEY} is not the declared one")
        new.pop(PRIMARY_BOUND_KEY)
    if PROPERTY_BOUND_KEY not in new:
        out.append(f"the complete configuration records no {PROPERTY_BOUND_KEY}")
    elif PROPERTY_BOUND_KEY in old and old[PROPERTY_BOUND_KEY] != new[PROPERTY_BOUND_KEY]:
        out.append(f"the complete rerun changed {PROPERTY_BOUND_KEY}")
    old.pop(PROPERTY_BOUND_KEY, None)
    new.pop(PROPERTY_BOUND_KEY, None)
    if old != new:
        changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        out.append(f"the complete configuration differs from the baseline's: {changed}")
    for name in ("property-replicates.parquet", "properties.csv"):
        if manifest["sha256"].get(name) == baseline["sha256"].get(name):
            out.append(f"the complete {name} is the baseline's file: no property phase ran")
    return out


def requested(rows: pd.DataFrame, name: str) -> list[int]:
    """The ``requested_replicates`` values one cell's rows carry."""
    prop, cell = name.split("/", 1)
    values = rows.loc[(rows["property"] == prop) & (rows["cell"] == cell), "requested_replicates"]
    return sorted({int(value) for value in values.dropna()})


def read_output(slug: str, output: Path, plan_raw: bytes) -> dict[str, Any]:
    """Validate one study's output by its route; returns the route and its producer commit."""
    if (output / COMPLETE).exists():
        return validate_complete(slug, output, plan_raw)
    if not (output / "manifest.json").exists():
        state = "a replay miss and no complete rerun" if (output / MISS).exists() else "no output"
        raise ValueError(f"the wave holds {state} for {slug}")
    log = validate_output(slug, output, plan_raw)
    return {"route": TARGETED_ROUTE, "source_commit": log["source_commit"]}


def verdict(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def verdict_changes(
    before: pd.DataFrame, after: pd.DataFrame, moved: Iterable[str]
) -> dict[str, list[dict[str, Any]]]:
    """Every verdict of a complete rerun that changed outside the moved cells.

    ``cells`` lists each ``property/cell`` whose ``passed`` changed (a row present on one side
    only reads ``None`` on the other), ``properties`` each property whose ``property_passed``
    values changed.  Both are ordinary regeneration results: they are reported, never blocking.
    """
    moved = set(moved)

    def passed(summary: pd.DataFrame) -> dict[str, str | None]:
        names = summary["property"].astype(str) + "/" + summary["cell"].astype(str)
        return {name: verdict(value) for name, value in zip(names, summary["passed"], strict=True)}

    def property_passed(summary: pd.DataFrame) -> dict[str, list[str]]:
        return {
            str(prop): sorted({str(v) for v in group["property_passed"].dropna()})
            for prop, group in summary.groupby("property", sort=True)
        }

    old, new = passed(before), passed(after)
    cells = [
        {"cell": name, "baseline_passed": old.get(name), "passed": new.get(name)}
        for name in sorted((set(old) | set(new)) - moved)
        if old.get(name) != new.get(name)
    ]
    old_props, new_props = property_passed(before), property_passed(after)
    properties = [
        {
            "property": prop,
            "baseline_property_passed": old_props.get(prop, []),
            "property_passed": new_props.get(prop, []),
        }
        for prop in sorted(set(old_props) | set(new_props))
        if old_props.get(prop, []) != new_props.get(prop, [])
    ]
    return {"cells": cells, "properties": properties}


def breach_direction(role: str, reading: Mapping[str, Any], baseline_coverage: float) -> str:
    """The direction a cell that lost its verdict moved: ``"less"`` or ``"more"`` coverage.

    A positive cell whose lower limit is below 0.92 covers less, and one whose upper limit is
    above 0.98 covers more.  A control loses its side in the direction its coverage moved.
    """
    if role == "positive":
        lower, upper = reading["coverage_ci_lower"], reading["coverage_ci_upper"]
        if (
            upper is not None
            and upper > COVERAGE_BAND[1]
            and not (lower is not None and lower < COVERAGE_BAND[0])
        ):
            return "more"
        return "less"
    return "more" if float(reading["coverage"]) > float(baseline_coverage) else "less"


def blocks(rule: str, lost: bool, p_value: float) -> bool:
    """Whether a cell blocks the flip: under ``strict`` any loss, under ``paired`` a loss whose
    exact one-sided McNemar p-value in the direction of the breach is at most 0.01."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    return lost and (rule == "strict" or p_value <= DECISION_ALPHA)


def decide(
    wave: Path, *, studies: Sequence[str] | None = None, plan_raw: bytes | None = None
) -> dict[str, Any]:
    """The section 8 decision over one wave: flip, or blocked with every reason.

    The rule is the plan's ``decision_rule``, bound before the wave.  Every planned study must
    have an output in ``wave/<slug>`` from one commit that descends from :data:`FLIP_COMMIT`,
    under the current plan.  ``studies`` names a subset for a rehearsal; the decision is then
    ``partial`` and no install reads it.  A band key that failed inside the band is a reading:
    its rows are not covered and enter the cell's interval test.

    A study that missed the replay rule is read from its complete rerun (:func:`read_output`,
    route ``complete_rerun``) under the same commit as every other study.  Its moved cells are
    read as a targeted study's, against the committed baseline rows, but paired over the union
    of replicates (:func:`discordant_union`): a replicate one side lost is not covered there.
    The coverage, its interval, ``passed``, the loss and the breach come from those rows
    (:func:`union_reading`), not from the regeneration's complete-case summary, whose verdict
    is kept as ``regeneration_passed``.  ``band_failures`` is then the replicates the rerun
    lost, each counted not covered, and the property failure files it wrote are listed.  Every other ``passed`` or ``property_passed`` that
    changed is listed under ``complete_rerun_changes`` and does not block, and the runtime
    versions that differ from the baseline's are recorded under ``stack_differences``.
    """
    plan_raw = PLAN_PATH.read_bytes() if plan_raw is None else plan_raw
    plan = json.loads(plan_raw)
    rule = plan.get("decision_rule")
    if rule not in RULES:
        raise ValueError("the plan binds no decision_rule: no decision runs")
    planned = list(plan["studies"])
    chosen = planned if studies is None else list(studies)
    if set(chosen) - set(planned):
        raise ValueError(f"not planned: {sorted(set(chosen) - set(planned))}")
    missing = [slug for slug in chosen if not (wave / slug).is_dir()]
    if missing:
        raise ValueError(f"the wave has no output for {missing}")
    sources = {slug: read_output(slug, wave / slug, plan_raw) for slug in chosen}
    commits = {source["source_commit"] for source in sources.values()}
    if len(commits) != 1:
        raise ValueError(f"the wave ran from {len(commits)} commits: {sorted(commits)}")
    (commit,) = commits
    if not descends_from_flip(commit):
        raise ValueError(f"{commit} does not descend from {FLIP_COMMIT}")
    blocking: list[str] = []
    owner_rows: list[str] = []
    readings: dict[str, Any] = {}
    flips: list[str] = []
    for slug in chosen:
        entry = plan["studies"][slug]
        record = record_for(slug)
        if digest((record.artifacts / "manifest.json").read_bytes()) != entry["manifest_sha256"]:
            raise ValueError(f"{slug}: the committed artifacts are not the baseline")
        source = sources[slug]
        complete = source["route"] == COMPLETE_ROUTE
        margins = getattr(record, "margins", None)
        if complete and margins is not None:
            declared = (
                margins.confidence_level,
                tuple(margins.calibration_coverage),
                margins.alpha,
            )
            if declared != (COVERAGE_CONFIDENCE, COVERAGE_BAND, CONTROL_ALPHA):
                raise ValueError(f"{slug}: its margins are not the rule the decision reads")
        output = wave / slug / COMPLETE if complete else wave / slug
        before_rows = read_rows(record.artifacts / "property-replicates.parquet")
        after_rows = read_rows(output / "property-replicates.parquet")
        before = pd.read_csv(record.artifacts / "properties.csv")
        after = pd.read_csv(output / "properties.csv")
        manifest = json.loads((output / "manifest.json").read_bytes())
        failures: dict[str, int] = {}
        if not complete:
            failures = manifest["composite"]["replacement"]["replay"]["band_failures"]
        after_cells = {cell["cell"]: cell for cell in moved_readings(after, entry["moved_cells"])}
        study = []
        for baseline in entry["cells"]:
            name = f"{BAND_FAMILY}/{baseline['cell']}"
            now = after_cells[baseline["cell"]]
            lost_counts: dict[str, Any] = {}
            if complete:
                b, c, pairs, lost_now, lost_before = discordant_union(before_rows, after_rows, name)
                lost_counts = {
                    "replicates_lost_now": lost_now,
                    "replicates_lost_before": lost_before,
                    "regeneration_passed": now["passed"],
                }
                failures[name] = lost_now
                now = {
                    "cell": baseline["cell"],
                    **union_reading(before_rows, after_rows, name, baseline["role"]),
                }
            else:
                b, c, pairs = discordant(before_rows, after_rows, name)
            reading = {
                **lost_counts,
                "cell": baseline["cell"],
                "role": baseline["role"],
                "replicates": pairs,
                "baseline_passed": baseline["passed"],
                "passed": now["passed"],
                "baseline_coverage": baseline["coverage"],
                "coverage": now["coverage"],
                "coverage_ci_lower": now["coverage_ci_lower"],
                "coverage_ci_upper": now["coverage_ci_upper"],
                "interval_red": interval_red(now) if baseline["role"] == "positive" else None,
                "b_covered_before_only": b,
                "c_covered_now_only": c,
                "mcnemar_p_covers_less": mcnemar_less(b, c),
                "mcnemar_p_covers_more": mcnemar_less(c, b),
                "band_failures": int(failures.get(name, 0)),
            }
            lost = baseline["passed"] is True and now["passed"] is not True
            direction = breach_direction(baseline["role"], now, baseline["coverage"])
            p = reading[f"mcnemar_p_covers_{direction}"]
            reading["breach"] = direction if lost else None
            where = f"{slug}:{baseline['cell']}"
            if lost:
                if blocks(rule, lost, p):
                    blocking.append(f"{where}: previously green, now red (paired p {p:.3g})")
                else:
                    owner_rows.append(f"{where}: now red (paired p {p:.3g}); add an owner row")
            reading["lost"] = lost
            study.append(reading)
        for prop in sorted(set(before["property"])):
            old = before.loc[before["property"] == prop, "property_passed"].dropna()
            new = after.loc[after["property"] == prop, "property_passed"].dropna()
            if old.astype(str).tolist() != new.astype(str).tolist():
                flips.append(f"{slug}:{prop}: property_passed changed")
        if complete:
            readings[slug] = {
                "route": COMPLETE_ROUTE,
                "cells": study,
                "complete_rerun_changes": verdict_changes(before, after, entry["moved_cells"]),
                "failure_files": sorted(set(manifest["sha256"]) & set(PROPERTY_FAILURE_FILES)),
                "manifest_sha256": source["manifest_sha256"],
                "miss_sha256": source["miss_sha256"],
                "stack_differences": stack_differences(
                    json.loads((record.artifacts / "manifest.json").read_bytes())["generated_with"][
                        "subject"
                    ],
                    manifest["generated_with"]["subject"],
                ),
            }
            continue
        readings[slug] = {
            "route": TARGETED_ROUTE,
            "cells": study,
            "summary": {
                key: manifest["composite"]["analysis"][key]
                for key in ("maximum_carried_difference", "changed_rows")
            },
            "manifest_sha256": digest((wave / slug / "manifest.json").read_bytes()),
        }
    partial = studies is not None and set(chosen) != set(planned)
    outcome = "partial" if partial else ("blocked" if blocking else "flip")
    return {
        "mode": MODE,
        "rule": rule,
        "alpha": DECISION_ALPHA,
        "coverage_band": list(COVERAGE_BAND),
        "commit": commit,
        "plan_sha256": digest(plan_raw),
        "studies": chosen,
        "decision": outcome,
        "would_block": blocking,
        "owner_rows": owner_rows,
        "study_verdict_flips": flips,
        "readings": readings,
    }


def sidecar_record_path(directory: Path) -> Path:
    return directory / SIDECAR / "record.json"


def sidecar_files(route: str, output_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """The files of a blocked study's sidecar.

    A complete rerun's sidecar adds every other artifact of the rerun, as
    ``complete-<name>``; its ``properties.csv`` is the diagnostic one.
    """
    if route == TARGETED_ROUTE:
        return SIDECAR_FILES
    if route != COMPLETE_ROUTE:
        raise ValueError(f"unknown route {route!r}")
    extra = sorted(name for name in output_manifest["sha256"] if name != "properties.csv")
    return (*SIDECAR_FILES, *(COMPLETE_PREFIX + name for name in extra))


def is_table(name: str) -> bool:
    return name.endswith((".csv", ".csv.gz"))


def diagnostic_table(source: Path, target: Path) -> bool:
    """Write ``source`` with its verdict columns labelled diagnostic (:data:`VERDICT_LABELS`).

    The values are read round trip.  Returns whether a column was relabelled; a table with no
    verdict column is copied byte for byte.
    """
    table = pd.read_csv(source, low_memory=False, float_precision="round_trip")
    if not set(VERDICT_LABELS) & set(table.columns):
        shutil.copyfile(source, target)
        return False
    compression = {"method": "gzip", "mtime": 0} if target.name.endswith(".gz") else None
    write_csv(table.rename(columns=VERDICT_LABELS), target, compression=compression)
    return True


def write_sidecar(
    slug: str, output: Path, target: Path, decision_raw: bytes, *, route: str = TARGETED_ROUTE
) -> None:
    """A blocked wave's diagnostic files for one study, from its unblocked output.

    ``output`` is the directory that holds the study's ``manifest.json``: the composite's, or a
    complete rerun's ``complete/``.  Every verdict column the sidecar carries is labelled
    diagnostic.
    """
    plan_raw = PLAN_PATH.read_bytes()
    entry = json.loads(plan_raw)["studies"][slug]
    side = target / SIDECAR
    side.mkdir(parents=True, exist_ok=True)
    rows = read_rows(output / "property-replicates.parquet")
    write_rows(
        rows.loc[band_mask(rows, entry["moved_cells"])].reset_index(drop=True),
        side / SIDECAR_FILES[0],
    )
    diagnostic_table(output / "properties.csv", side / SIDECAR_FILES[1])
    (side / SIDECAR_FILES[2]).write_bytes(decision_raw)
    shutil.copyfile(output / "manifest.json", side / SIDECAR_FILES[3])
    manifest = json.loads((output / "manifest.json").read_bytes())
    names = sidecar_files(route, manifest)
    relabelled = []
    for name in names[len(SIDECAR_FILES) :]:
        source = output / name.removeprefix(COMPLETE_PREFIX)
        if is_table(name) and diagnostic_table(source, side / name):
            relabelled.append(name)
        elif not is_table(name):
            shutil.copyfile(source, side / name)
    record = {
        "mode": MODE,
        "route": route,
        "blocked": True,
        "slug": slug,
        "baseline_commit": BASELINE_COMMIT,
        "plan_sha256": digest(plan_raw),
        "decision_sha256": digest(decision_raw),
        "note": (
            "Diagnostic only: a blocked Wave 2 leaves every committed artifact byte-identical "
            "(plan section 4.4); no support row and no grid row reads these verdicts."
        ),
        "sha256": {name: digest((side / name).read_bytes()) for name in names},
        "output_manifest_sha256": digest((output / "manifest.json").read_bytes()),
    }
    if route == COMPLETE_ROUTE:
        record["relabelled"] = relabelled
    (side / "record.json").write_bytes(json_bytes(record))


def validate_sidecar(directory: Path, plan: Mapping[str, Any]) -> list[str]:
    """What a blocked wave's sidecar says that its files and the plan do not."""
    out = []
    path = sidecar_record_path(directory)
    record = json.loads(path.read_bytes())
    slug = record.get("slug")
    entry = plan["studies"].get(slug)
    if record.get("mode") != MODE or record.get("blocked") is not True or entry is None:
        return [f"{path}: not a blocked band-rerun record of a planned study"]
    if (ROOT / entry["artifact_directory"]).resolve() != directory.resolve():
        out.append(f"{slug}: the sidecar sits in another study's directory")
    route = record.get("route", TARGETED_ROUTE)
    if route not in (TARGETED_ROUTE, COMPLETE_ROUTE):
        return [*out, f"{slug}: the record names no route"]
    expected: tuple[str, ...] = SIDECAR_FILES
    if route == COMPLETE_ROUTE:
        output_manifest = json.loads((directory / SIDECAR / SIDECAR_FILES[3]).read_bytes())
        if output_manifest.get("slug") != slug or "generation_mode" in output_manifest:
            out.append(f"{slug}: the sidecar's output manifest is not the study's complete rerun")
        expected = sidecar_files(route, output_manifest)
        relabelled = record.get("relabelled")
        if not isinstance(relabelled, list) or not set(relabelled) <= set(expected):
            out.append(f"{slug}: the record names no relabelled files")
            relabelled = []
        for name in expected[len(SIDECAR_FILES) :]:
            copied = directory / SIDECAR / name
            original = output_manifest["sha256"][name.removeprefix(COMPLETE_PREFIX)]
            if name in relabelled:
                if not is_table(name):
                    out.append(f"{slug}: {name} is relabelled but is no table")
            elif not copied.is_file() or digest(copied.read_bytes()) != original:
                out.append(f"{slug}: {name} is not the complete rerun's file")
    for name in expected:
        path = directory / SIDECAR / name
        if not path.is_file() or digest(path.read_bytes()) != record["sha256"].get(name):
            out.append(f"{slug}: {name} differs from its recorded hash")
    if set(record["sha256"]) != set(expected):
        out.append(f"{slug}: the record hashes other files")
    committed = (directory / "manifest.json").read_bytes()
    if digest(committed) != entry["manifest_sha256"]:
        out.append(f"{slug}: a blocked wave changed the committed manifest")
    if record.get("plan_sha256") is None or record.get("baseline_commit") != BASELINE_COMMIT:
        out.append(f"{slug}: the record names no plan or another baseline")
    return out


def installed_state(plan: Mapping[str, Any]) -> dict[str, str]:
    """Each planned study's state in the tree: ``baseline``, ``composite`` or ``sidecar``.

    A complete rerun a flip installed (:func:`installed_complete`) reads ``composite``: the
    flip installed it in place of the study's composite.
    """
    out = {}
    for slug, entry in plan["studies"].items():
        directory = ROOT / entry["artifact_directory"]
        manifest = json.loads((directory / "manifest.json").read_bytes())
        sidecar = sidecar_record_path(directory).exists()
        if manifest.get("generation_mode") == MODE or installed_complete(directory) is not None:
            out[slug] = "composite+sidecar" if sidecar else "composite"
        else:
            out[slug] = "sidecar" if sidecar else "baseline"
    return out


def install_complete(
    slug: str, output: Path, target: Path, decision_raw: bytes, plan_raw: bytes
) -> None:
    """A flip's install of a complete rerun: its artifacts as the regeneration wrote them.

    ``output`` is the study's wave directory.  Every file the complete manifest hashes and the
    manifest itself replace the committed ones, and a baseline artifact the rerun did not write
    is removed, as a regeneration removes it; the study's own sources stay.  The record
    ``band_rerun/complete-rerun.json`` keeps the baseline manifest the plan hashed, so the plan
    still validates (:func:`baseline_manifest_raw`).
    """
    complete = output / COMPLETE
    manifest_raw = (complete / "manifest.json").read_bytes()
    names = json.loads(manifest_raw)["sha256"]
    baseline_raw = (target / "manifest.json").read_bytes()
    for name in json.loads(baseline_raw)["sha256"]:
        if name not in names:
            (target / name).unlink(missing_ok=True)
    for name in (*names, "manifest.json"):
        shutil.copyfile(complete / name, target / name)
    record = {
        "mode": MODE,
        "route": COMPLETE_ROUTE,
        "slug": slug,
        "baseline": {
            "commit": BASELINE_COMMIT,
            "manifest_utf8": baseline_raw.decode(),
            "manifest_sha256": digest(baseline_raw),
        },
        "plan_sha256": digest(plan_raw),
        "decision_sha256": digest(decision_raw),
        "manifest_sha256": digest(manifest_raw),
        "replay_miss": json.loads((output / MISS).read_bytes()),
    }
    (target / SIDECAR).mkdir(exist_ok=True)
    (target / SIDECAR / COMPLETE_RECORD).write_bytes(json_bytes(record))


def install_wave(wave: Path) -> str:
    """Install a decided wave: every composite, or every sidecar, after every check passes.

    A complete rerun installs its own artifacts on a flip (:func:`install_complete`) and joins
    the sidecars when the flip is blocked.
    """
    decision_raw = (wave / DECISION).read_bytes()
    recorded = json.loads(decision_raw)
    if recorded["decision"] not in ("flip", "blocked"):
        raise SystemExit(f"refused: a {recorded['decision']} decision installs nothing")
    plan_raw = PLAN_PATH.read_bytes()
    again = decide(wave, plan_raw=plan_raw)
    if json_bytes(again) != json_bytes(recorded):
        raise SystemExit("refused: the wave no longer gives its recorded decision")
    if set(installed_state(json.loads(plan_raw)).values()) != {"baseline"}:
        raise SystemExit("refused: the tree is not at the baseline of every planned study")
    plan = json.loads(plan_raw)
    for slug in recorded["studies"]:
        target = ROOT / plan["studies"][slug]["artifact_directory"]
        output = wave / slug
        route = recorded["readings"][slug]["route"]
        if recorded["decision"] == "flip" and route == COMPLETE_ROUTE:
            install_complete(slug, output, target, decision_raw, plan_raw)
        elif recorded["decision"] == "flip":
            manifest = json.loads((output / "manifest.json").read_bytes())
            for name in (*manifest["sha256"], "manifest.json"):
                shutil.copyfile(output / name, target / name)
        else:
            source = output / COMPLETE if route == COMPLETE_ROUTE else output
            write_sidecar(slug, source, target, decision_raw, route=route)
    return str(recorded["decision"])


# ---------------------------------------------------------------------------- smoke and timing


_ORIGINAL_SEEDS: dict[str, Callable[..., int]] = {
    "stream_seed": seeds.stream_seed,
    "replicate_seed": seeds.replicate_seed,
}
#: How many seeds the salted functions handed out in this process, and their values.
SALTED_CALLS = [0]
SALTED_SEEDS: set[int] = set()
#: The seeds the declared stream would have handed out for the same calls.
DECLARED_SEEDS: set[int] = set()
#: A seed below this is not read off a key: a replicate index or a size could equal it.
SEED_FLOOR = 2**20
_WORKER_SALT: list[contextlib.AbstractContextManager[None]] = []


def _salt(seed: int, declared: int) -> int:
    SALTED_CALLS[0] += 1
    SALTED_SEEDS.add(int(seed))
    DECLARED_SEEDS.add(int(declared))
    return seed


def _salted_stream_seed(record: StudyRecord, *labels: str | int) -> int:
    original = _ORIGINAL_SEEDS["stream_seed"]
    return _salt(original(record, SMOKE_STREAM, *labels), original(record, *labels))


def _salted_replicate_seed(record: StudyRecord, scenario: str, replicate: int) -> int:
    return _salt(
        _ORIGINAL_SEEDS["stream_seed"](record, SMOKE_STREAM, "replicate", scenario, replicate),
        _ORIGINAL_SEEDS["replicate_seed"](record, scenario, replicate),
    )


def _salted_pilot_seed(stream: str, replicate: int) -> int:
    from tests import long_horizon_design

    original = _ORIGINAL_SEEDS.get("pilot", long_horizon_design.stream_seed)
    return _salt(original(f"{SMOKE_STREAM}/{stream}", replicate), original(stream, replicate))


def key_seeds(key: Any) -> set[int]:
    """Every integer a key carries that could be a seed (at least :data:`SEED_FLOOR`)."""
    out: set[int] = set()
    stack = [key]
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            if abs(item) >= SEED_FLOOR:
                out.add(int(item))
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, tuple | list | frozenset | set):
            stack.extend(item)
        elif hasattr(item, "__dataclass_fields__"):
            stack.extend(getattr(item, name) for name in item.__dataclass_fields__)
    return out


def salt_findings(keys: Sequence[Any], worker_salted: Sequence[int]) -> list[str]:
    """Each smoke key not shown to run on the smoke stream.

    A key runs on it when its worker drew a salted seed, or when it carries a seed the salted
    functions handed out while the keys were built.  A key that carries a seed of the declared
    stream (one the salted functions computed beside each salted seed) is refused whatever else
    it carries.
    """
    declared = DECLARED_SEEDS
    out = []
    for key, drawn in zip(keys, worker_salted, strict=True):
        carried = key_seeds(key)
        if carried & (declared - SALTED_SEEDS):
            out.append(f"{key!r} carries a declared seed")
        elif drawn == 0 and not carried & SALTED_SEEDS:
            out.append(f"{key!r} drew no salted seed")
    return out


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


def salt_check(slug: str) -> list[str]:
    """Whether the first band key of each call of a planned study runs on the smoke stream.

    The keys are captured with every seed function salted.  A key passes when it carries a
    salted seed, or when its draw takes one; a key that carries a declared seed fails.  The
    check runs no fit, so a study whose seed is taken only inside ``fit_one`` fails it and
    needs its seed path added here before its smoke can be trusted.
    """
    record = record_for(slug)
    entry = json.loads(PLAN_PATH.read_bytes())["studies"][slug]
    index = committed_index(read_rows(record.artifacts / "property-replicates.parquet"))
    out = []
    with salted():
        calls = capture(record, entry["moved_cells"], index)
        for call in calls:
            if not call.selected:
                continue
            key = call.keys[call.selected[0]]
            drawn = 0
            draw = call.options.get("draw")
            if not key_seeds(key) & SALTED_SEEDS and draw is not None:
                before = SALTED_CALLS[0]
                draw(key)
                drawn = SALTED_CALLS[0] - before
            out += [f"{call.label}: {text}" for text in salt_findings([key], [drawn])]
    return out


def _salt_worker(slug: str) -> None:
    """A smoke worker's initializer: import the study, then salt its seeds for good."""
    require_threads()
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


def smoke(slug: str, *, jobs: int, timing: bool = False) -> dict[str, Any]:
    """Fit band keys on the smoke stream, and read their bands.

    ``--smoke`` takes the first :data:`SMOKE_REPLICATES` keys of each moved cell.  ``--timing``
    takes :data:`TIMING_KEYS_PER_STRATUM` keys of each stratum (the keys that write one set of
    moved cells) and weights each stratum's seconds per key by its band keys.  Nothing is
    written into the repository.

    Raises
    ------
    SystemExit
        When a fitted key drew no salted seed: its sample may be a declared one.
    """
    require_threads()
    record = record_for(slug)
    plan = json.loads(PLAN_PATH.read_bytes())
    entry = plan["studies"][slug]
    moved = set(entry["moved_cells"])
    index = committed_index(read_rows(record.artifacts / "property-replicates.parquet"))
    with salted():
        calls = capture(record, moved, index)
    strata: dict[frozenset[str], list[tuple[Call, Any]]] = {}
    for call in calls:
        for position in call.selected:
            key = call.keys[position]
            strata.setdefault(frozenset(moved & call.written(key)), []).append((call, key))
    tasks: list[tuple[Call, Any]] = []
    if timing:
        for members in strata.values():
            tasks += members[:TIMING_KEYS_PER_STRATUM]
    else:
        taken: dict[str, int] = {}
        for members in strata.values():
            for call, key in members:
                names = moved & call.written(key)
                if all(taken.get(name, 0) >= SMOKE_REPLICATES for name in names):
                    continue
                for name in names:
                    taken[name] = taken.get(name, 0) + 1
                tasks.append((call, key))
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
    unverified = salt_findings(
        [key for _, key in tasks], [result["salted_calls"] for result in results]
    )
    if unverified:
        raise SystemExit(
            f"refused: {len(unverified)} smoke key(s) not on the smoke stream, first "
            f"{unverified[0]}"
        )
    cells: dict[str, dict[str, Any]] = {}
    for (call, key), result in zip(tasks, results, strict=True):
        rows = (
            [] if result["value"] is None else rows_of(result["value"], call.options.get("rows_at"))
        )
        for name in sorted(moved & call.written(key)):
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
            cell["refused"] += sum(is_refused(row) for row in band)
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
            "refused_rows": cell["refused"],
            "refused_share": cell["refused"] / keys if keys else math.nan,
            "out_of_range_levels": cell["out_of_range_levels"],
            "kish_min": None if math.isinf(cell["kish_min"]) else round(cell["kish_min"], 1),
            "seconds_per_key": cell["cpu_seconds"] / keys if keys else math.nan,
            "covered_share": cell["covered"] / max(1, keys - cell["failed"]),
            "band_options": sorted(map(list, cell["options"])),
            "errors": cell["errors"][:3],
        }
    seconds: dict[frozenset[str], list[float]] = {}
    for (call, key), result in zip(tasks, results, strict=True):
        seconds.setdefault(frozenset(moved & call.written(key)), []).append(result["cpu_seconds"])
    weighted = sum(
        sum(values) / len(values) * len(strata[stratum]) for stratum, values in seconds.items()
    )
    cpu = sum(result["cpu_seconds"] for result in results)
    return {
        "slug": slug,
        "stream": SMOKE_STREAM,
        "keys": len(results),
        "strata": len(strata),
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "band_fits": entry["band_fits"],
        "estimated_cpu_hours": weighted / 3600.0,
        "cells": report,
    }


def replay_check(slug: str, *, jobs: int, keys: int = REPLAY_CHECK_KEYS) -> dict[str, Any]:
    """Refit the first band keys on their declared seeds and hold their other rows to the rule.

    A dry run of the replay half of a rerun: nothing is spliced or written.
    """
    require_threads()
    record = record_for(slug)
    entry = json.loads(PLAN_PATH.read_bytes())["studies"][slug]
    index = committed_index(read_rows(record.artifacts / "property-replicates.parquet"))
    calls = capture(record, entry["moved_cells"], index)
    left = keys
    for call in calls:
        call.selected = call.selected[: max(0, left)]
        left -= len(call.selected)
    values = run_calls(record, calls, jobs=jobs, cache=None)
    refit = replay(calls, values, index, entry["moved_cells"])
    return {
        "slug": slug,
        "keys": keys - max(0, left),
        **replay_record(refit),
        "band_rows": len(refit.band_rows),
        "refused_rows": refit.refused,
    }


# ---------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", help="one study slug from the plan")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cache", type=Path, help="checkpoint root; a rerun resumes from it")
    parser.add_argument("--output", type=Path, help="the wave directory, outside the repository")
    parser.add_argument("--decide", type=Path, metavar="WAVE")
    parser.add_argument("--studies", nargs="+", help="rehearse --decide on these studies")
    parser.add_argument("--install-wave", type=Path, metavar="WAVE")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--replay-check", action="store_true")
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
    if args.decide is not None:
        wave = args.decide.resolve()
        if plan.get("decision_rule") not in RULES:
            parser.error(
                "the plan binds no decision_rule yet; --decide runs once the owner sets it"
            )
        decision = decide(wave, studies=args.studies)
        path = wave / DECISION
        if path.exists():
            earlier = json.loads(path.read_bytes())
            if (earlier.get("rule"), earlier.get("plan_sha256")) != (
                decision["rule"],
                decision["plan_sha256"],
            ):
                parser.error(f"{path} holds a decision under another rule or plan; move it first")
        path.write_bytes(json_bytes(decision))
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    if args.install_wave is not None:
        print(f"installed: {install_wave(args.install_wave.resolve())}")
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
        report = smoke(args.study, jobs=args.jobs, timing=args.timing)
        text = json.dumps(report, indent=2, sort_keys=True, default=str)
        if args.report is not None:
            if args.report.resolve().is_relative_to(ROOT.resolve()):
                parser.error("--report writes outside the repository")
            args.report.write_text(text + "\n")
        print(text)
        return 0
    if args.output is None:
        parser.error("--output is required")
    wave = args.output.resolve()
    if wave.is_relative_to(ROOT.resolve()):
        parser.error("--output must lie outside the repository")
    output = wave / args.study
    started = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        result = execute(
            args.study,
            output,
            jobs=args.jobs,
            cache=args.cache,
            plan_raw=plan_raw,
            command=[Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
        )
    except ReplayMiss as miss:
        output.mkdir(parents=True, exist_ok=True)
        (output / MISS).write_bytes(
            json_bytes({"study": args.study, "started": started, "miss": str(miss)})
        )
        print(f"replay miss: {miss}; {args.study} takes a complete rerun", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
