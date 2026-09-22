"""Read the four scratch arms of each study under the rule RM18 declared before they ran.

Each arm is one code state crossed with one runtime.  ``docs/roadmap.md`` declares the arms,
the two preconditions that validate the harness, the statistics read and the reading rule,
under RM18 in "The runtime isolation, declared before it runs".  This module applies that
declaration and nothing else.  It changes no verdict, policy, budget, margin or law, and it
writes only to this directory.

The arms sit in one scratch root, laid out as ``<root>/<study>/<arm>/output`` for what
``--output`` received and ``<root>/<study>/<arm>/cache`` for what ``--cache`` received.  The
cache keeps ``samples.csv.gz``, which the driver otherwise writes to a temporary directory.

It reads git history, so it runs in a full clone only.  ``--output`` and
``--publish-manifests`` are required, so a bare run cannot overwrite the committed record.

    python -m tests.diagnostics.rm18_runtime.compare --arms <scratch-root>         --output <scratch>/isolation.csv --publish-manifests <scratch>/arms
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.diagnostics.rm18_runtime.row_drift import (
    AFTER,
    PRIMARY_KEYS,
    PROPERTY_KEYS,
    committed,
    drift,
    label,
)
from tests.studies.evidence.manifest import write_csv, write_lines

HERE = Path(__file__).resolve().parent

#: Code F fits the fold-local update.  Code P carries the declaration, and its ``src/`` is
#: identical to :data:`~tests.diagnostics.rm18_runtime.row_drift.AFTER`.
CODE = {
    "F": "7d5485abf42545843c7dc801dfdac57ec68fd0f0",
    "P": "56100ce9ddb48e3faba4130db2aa8a8ed07c5f6a",
}
RUNTIMES = ("R11", "R13")

#: The Python and SciPy versions each runtime pins, as the manifest records them.
RUNTIME_VERSIONS = {
    "R11": {"python": "3.11.13", "scipy": "1.17.1"},
    "R13": {"python": "3.13.7", "scipy": "1.18.0"},
}

#: Libraries the declaration holds identical in all four arms of a study.
SHARED_LIBRARIES = ("numpy", "pandas", "scikit_learn")
ARMS = tuple(f"{code}-{runtime}" for code in CODE for runtime in RUNTIMES)

#: Each precondition arm and the commit whose rows it must reproduce.
PRECONDITIONS = {"F-R11": CODE["F"], "P-R13": AFTER}

#: The declared precondition tolerance on every estimate and every standard error.
TOLERANCE = 1e-9

#: The four comparisons the reading rule makes, as (axis, held level, first arm, second arm).
COMPARISONS = (
    ("code", "R11", "F-R11", "P-R11"),
    ("code", "R13", "F-R13", "P-R13"),
    ("runtime", "F", "F-R11", "F-R13"),
    ("runtime", "P", "P-R11", "P-R13"),
)

NOT_VALIDATED = "harness not validated, no attribution"


@dataclass(frozen=True)
class Cell:
    """One declared cell: where its verdict lives and which statistics are read beside it."""

    name: str
    table: str
    where: dict[str, str]
    statistics: tuple[str, ...]


@dataclass(frozen=True)
class Study:
    """One study of the diagnostic, by its committed directory."""

    directory: str
    name: str
    slug: str
    implementation: str
    reference_implementation: str
    constructions: dict[str, tuple[str, str | None]]
    row_tables: dict[str, Sequence[str]]
    cells: tuple[Cell, ...]
    #: Cells whose declared statistic is also the count of rows that differ from the
    #: committed rows.
    control_cells: tuple[str, ...] = ()


#: Verdict columns of each summary table, and the keys that identify a row of it.
VERDICTS = {
    "properties.csv": (("property", "cell", "role"), ("passed", "property_passed")),
    "equivalence.csv": (("scenario", "estimand"), ("passed", "comparison_conclusion")),
    "performance-tests.csv": (("implementation", "scenario", "estimand"), ("passed",)),
}

STUDIES = (
    Study(
        "weighted_lmtp_ltmle",
        "cross-fitted weighted end-of-study longitudinal TMLE",
        "weighted-ltmle-crossfit",
        "cleverly-cross-fitted-weighted-ltmle",
        "lmtp-weighted",
        {
            "F": ("fold_specific_cross_fit_weighted", None),
            "P": ("pooled_cross_fit", "training_fold_fluctuation"),
        },
        {
            "replicates.csv.gz": PRIMARY_KEYS,
            "property-replicates.csv.gz": PROPERTY_KEYS,
            "reference-inference.csv.gz": (*PRIMARY_KEYS, "inference_method"),
        },
        (
            Cell(
                "interval_calibration/static__correctly_specified",
                "properties.csv",
                {"property": "interval_calibration", "cell": "static__correctly_specified"},
                (
                    "efficiency_empirical_ratio",
                    "efficiency_empirical_ci_lower",
                    "efficiency_empirical_ci_upper",
                ),
            ),
            Cell(
                "paired ey_regimen[never]",
                "equivalence.csv",
                {"estimand": "ey_regimen[never]"},
                (
                    "coverage_difference",
                    "coverage_difference_lower",
                    "coverage_noninferiority_margin",
                    "comparison_conclusion",
                ),
            ),
        ),
    ),
    Study(
        "lmtp_ltmle",
        "cross-fitted end-of-study longitudinal TMLE",
        "canonical-ltmle-crossfit",
        "cleverly-cross-fitted-ltmle",
        "lmtp",
        {
            "F": ("fold_specific_cross_fit", None),
            "P": ("pooled_cross_fit", "training_fold_fluctuation"),
        },
        {
            "replicates.csv.gz": PRIMARY_KEYS,
            "property-replicates.csv.gz": PROPERTY_KEYS,
        },
        (
            Cell(
                "crossfit_overfitting/cross_fitted_ltmle",
                "properties.csv",
                {"property": "crossfit_overfitting", "cell": "cross_fitted_ltmle"},
                ("se_ratio", "se_ratio_ci_lower", "se_ratio_ci_upper", "coverage"),
            ),
            Cell(
                "crossfit_overfitting/in_sample_control",
                "properties.csv",
                {"property": "crossfit_overfitting", "cell": "in_sample_control"},
                ("se_ratio", "coverage"),
            ),
        ),
        control_cells=("in_sample_control",),
    ),
)


def reading(changes: dict[tuple[str, str], bool]) -> str:
    """The declared reading of one cell from whether each comparison changed its verdict."""
    expected = {(axis, level) for axis, level, _, _ in COMPARISONS}
    if set(changes) != expected:
        raise AssertionError(f"the reading needs exactly {sorted(expected)}, not {sorted(changes)}")
    code = [changed for (axis, _), changed in changes.items() if axis == "code"]
    runtime = [changed for (axis, _), changed in changes.items() if axis == "runtime"]
    if all(code) and not any(runtime):
        return "code"
    if all(runtime) and not any(code):
        return "runtime"
    if any(code) and any(runtime):
        return "both"
    if not any(code) and not any(runtime):
        return "neither"
    # A verdict is pass or fail, so a change occurs in zero, two or four comparisons, and a
    # change on one axis alone is always on both of its comparisons.
    raise AssertionError(f"a verdict pattern the declared rule does not cover: {changes}")


class Arms:
    """The scratch arms of one study, read lazily."""

    def __init__(self, root: Path, study: Study) -> None:
        self.root = root / study.directory
        self.study = study

    def output(self, arm: str, name: str) -> Path:
        return self.root / arm / "output" / name

    def table(self, arm: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.output(arm, name))

    def manifest(self, arm: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(
            self.output(arm, "manifest.json").read_text(encoding="utf-8")
        )
        return loaded

    def samples_digest(self, arm: str) -> str:
        path = self.root / arm / "cache" / "samples.csv.gz"
        return hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()


def _verdict_differences(left: pd.DataFrame, right: pd.DataFrame, name: str) -> int:
    keys, columns = VERDICTS[name]
    merged = left.merge(
        right,
        on=list(keys),
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = int((merged["_merge"] != "both").sum())
    shared = merged.loc[merged["_merge"] == "both"]
    changed = pd.Series(False, index=shared.index)
    for column in columns:
        changed |= shared[f"{column}_left"].astype(str) != shared[f"{column}_right"].astype(str)
    return unmatched + int(changed.sum())


def _record(study: str, section: str, subject: str, statistic: str, value: Any) -> dict[str, Any]:
    return {
        "study": study,
        "section": section,
        "subject": subject,
        "statistic": statistic,
        "value": value,
    }


def _provenance(arms: Arms, arm: str) -> list[dict[str, Any]]:
    manifest = arms.manifest(arm)
    generated = manifest.get("generated_with")
    generated = generated if isinstance(generated, dict) else {}
    subject = generated.get("subject")
    subject = subject if isinstance(subject, dict) else {}
    return [
        _record(arms.study.directory, "provenance", arm, key, subject.get(key))
        for key in (
            "cleverly_commit",
            "cleverly_worktree_clean",
            "python",
            "scipy",
            "numpy",
            "pandas",
            "scikit_learn",
        )
    ]


def _same_rows(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str]) -> bool:
    """Whether two tables contain the same columns and exact values under their row keys."""
    if list(left.columns) != list(right.columns):
        return False
    ordered = list(keys)
    if any(key not in left.columns for key in ordered):
        return False
    if left.duplicated(ordered).any() or right.duplicated(ordered).any():
        return False
    return (
        left.sort_values(ordered)
        .reset_index(drop=True)
        .equals(right.sort_values(ordered).reset_index(drop=True))
    )


def _configuration_without_construction(configuration: dict[str, Any]) -> dict[str, Any]:
    """The conditions RM18 holds fixed rather than varies on the code axis."""
    return {
        key: value
        for key, value in configuration.items()
        if key not in {"construction", "reference_construction"}
    }


def _manifest_failures(arms: Arms, arm: str, manifest: dict[str, Any]) -> list[str]:
    """Identity, provenance, configuration, and consumed-artifact failures for one arm."""
    failures = []
    code, runtime = arm.split("-")
    generated = manifest.get("generated_with")
    if not isinstance(generated, dict):
        failures.append(f"{arm}: generated_with is not an object")
        generated = {}
    subject = generated.get("subject")
    if not isinstance(subject, dict):
        failures.append(f"{arm}: generated_with.subject is not an object")
        subject = {}
    reference = generated.get("reference")
    configuration = manifest.get("configuration")
    digests = manifest.get("sha256")

    expected_identity = {
        "schema_version": 2,
        "study": arms.study.name,
        "slug": arms.study.slug,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            failures.append(f"{arm}: manifest {key} {manifest.get(key)!r} is not {expected!r}")
    if subject.get("implementation") != arms.study.implementation:
        failures.append(
            f"{arm}: implementation {subject.get('implementation')!r} is not "
            f"{arms.study.implementation!r}"
        )
    if (
        not isinstance(reference, dict)
        or reference.get("implementation") != arms.study.reference_implementation
    ):
        failures.append(
            f"{arm}: reference implementation is not {arms.study.reference_implementation!r}"
        )

    commit = str(subject.get("cleverly_commit"))
    if commit != CODE[code]:
        failures.append(f"{arm}: cleverly_commit {commit} is not {CODE[code]}")
    if subject.get("cleverly_worktree_clean") is not True:
        failures.append(
            f"{arm}: cleverly_worktree_clean is {subject.get('cleverly_worktree_clean')!r}"
        )
    for key, pinned in RUNTIME_VERSIONS[runtime].items():
        if str(subject.get(key)) != pinned:
            failures.append(f"{arm}: {key} {subject.get(key)} is not {pinned}")

    if not isinstance(configuration, dict):
        failures.append(f"{arm}: configuration is not an object")
    else:
        construction, reference_construction = arms.study.constructions[code]
        if configuration.get("construction") != construction:
            failures.append(
                f"{arm}: construction {configuration.get('construction')!r} is not {construction!r}"
            )
        if configuration.get("reference_construction") != reference_construction:
            failures.append(
                f"{arm}: reference_construction "
                f"{configuration.get('reference_construction')!r} is not "
                f"{reference_construction!r}"
            )

    required = set(arms.study.row_tables) | set(VERDICTS)
    if not isinstance(digests, dict):
        failures.append(f"{arm}: sha256 is not an object")
    else:
        for name in sorted(required):
            expected = digests.get(name)
            path = arms.output(arm, name)
            if not isinstance(expected, str):
                failures.append(f"{arm}: manifest has no sha256 for consumed artifact {name}")
            elif not path.is_file():
                failures.append(f"{arm}: consumed artifact {name} is missing")
            else:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    failures.append(
                        f"{arm}: consumed artifact {name} hashes to {actual}, not {expected}"
                    )
    return failures


def _reference_failures(arms: Arms) -> list[str]:
    """Failures of the declaration that every arm reads identical ``lmtp`` rows."""
    failures = []
    names = ["replicates.csv.gz"]
    if "reference-inference.csv.gz" in arms.study.row_tables:
        names.append("reference-inference.csv.gz")
    if any(not arms.output(arm, name).is_file() for arm in ARMS for name in names):
        # _manifest_failures owns the missing-artifact message.  Do not try to read a file
        # after that check has already made the harness invalid.
        return failures

    baseline = arms.table("F-R11", "replicates.csv.gz")
    baseline = baseline.loc[baseline["implementation"] == arms.study.reference_implementation]
    if baseline.empty:
        failures.append(
            f"F-R11: replicates.csv.gz has no {arms.study.reference_implementation!r} rows"
        )
        return failures
    for arm in ARMS[1:]:
        candidate = arms.table(arm, "replicates.csv.gz")
        candidate = candidate.loc[
            candidate["implementation"] == arms.study.reference_implementation
        ]
        if not _same_rows(baseline, candidate, PRIMARY_KEYS):
            failures.append(f"{arm}: lmtp comparator rows differ from F-R11")

    if "reference-inference.csv.gz" in arms.study.row_tables:
        baseline = arms.table("F-R11", "reference-inference.csv.gz")
        keys = arms.study.row_tables["reference-inference.csv.gz"]
        for arm in ARMS[1:]:
            candidate = arms.table(arm, "reference-inference.csv.gz")
            if not _same_rows(baseline, candidate, keys):
                failures.append(f"{arm}: reference-inference rows differ from F-R11")
    return failures


def fixed_condition_failures(arms: Arms) -> list[str]:
    """Every declared fixed condition that an arm's manifest breaks, as readable messages.

    Each arm must run its pinned code from a clean worktree under its pinned runtime.  The four
    arms must share the declared configuration, comparator, and fixed libraries.  Each consumed
    artifact must match the manifest that identifies its run.  An arm that breaks one of these
    is not the arm the declaration names, so the harness is not validated.
    """
    failures = []
    manifests = {arm: arms.manifest(arm) for arm in ARMS}
    for arm, manifest in manifests.items():
        failures += _manifest_failures(arms, arm, manifest)

    generated = {
        arm: value if isinstance(value := manifest.get("generated_with"), dict) else {}
        for arm, manifest in manifests.items()
    }
    subjects = {
        arm: value if isinstance(value := source.get("subject"), dict) else {}
        for arm, source in generated.items()
    }
    for key in SHARED_LIBRARIES:
        versions = {arm: str(subject.get(key)) for arm, subject in subjects.items()}
        if len(set(versions.values())) != 1:
            failures.append(f"{key} differs across arms: {versions}")
    references = {arm: source.get("reference") for arm, source in generated.items()}
    if len({json.dumps(value, sort_keys=True) for value in references.values()}) != 1:
        failures.append(f"reference provenance differs across arms: {references}")

    configurations = {arm: manifest.get("configuration") for arm, manifest in manifests.items()}
    valid_configurations = {
        arm: value for arm, value in configurations.items() if isinstance(value, dict)
    }
    if len(valid_configurations) == len(ARMS):
        fixed = {
            arm: _configuration_without_construction(value)
            for arm, value in valid_configurations.items()
        }
        if len({json.dumps(value, sort_keys=True) for value in fixed.values()}) != 1:
            failures.append(f"fixed configuration differs across arms: {fixed}")
    failures += _reference_failures(arms)
    return failures


def _precondition(arms: Arms, arm: str, revision: str) -> tuple[bool, list[dict[str, Any]]]:
    study = arms.study
    subject = f"{arm} against {label(revision)}"
    records = []
    held = True
    for name, keys in study.row_tables.items():
        expected = committed(revision, f"tests/canonical/{study.directory}/{name}")
        result = drift(expected, arms.table(arm, name), keys, tolerance=TOLERANCE)
        held &= (
            result["rows_outside_tolerance"] == 0
            and result["rows_only_older"] == 0
            and result["rows_only_newer"] == 0
        )
        records += [
            _record(study.directory, "precondition", subject, f"{name}: {key}", value)
            for key, value in result.items()
        ]
    for name in VERDICTS:
        expected = committed(revision, f"tests/canonical/{study.directory}/{name}")
        changed = _verdict_differences(expected, arms.table(arm, name), name)
        held &= changed == 0
        records.append(
            _record(
                study.directory, "precondition", subject, f"{name}: verdicts differing", changed
            )
        )
    records.append(_record(study.directory, "precondition", subject, "held", held))
    return held, records


def _cell_row(arms: Arms, arm: str, cell: Cell) -> pd.Series:
    frame = arms.table(arm, cell.table)
    selected = frame
    for column, value in cell.where.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"{cell.name} matched {len(selected)} rows in {arm}")
    return selected.iloc[0]


def _verdict(row: pd.Series) -> bool:
    """The cell's own verdict.

    Not ``property_passed``, which is the verdict of the whole family.  The in-sample control
    passes its own cell at ``7d5485a`` while its family fails with the positive cell, and the
    declared rule reads each cell's verdict alone.

    The column must hold a boolean.  ``bool()`` of a string such as ``"False"`` is true, so a
    column that pandas did not parse as boolean would silently read every verdict as a pass.
    """
    value = row["passed"]
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"passed must be boolean, not {type(value).__name__}: {value!r}")
    return bool(value)


def _control_drift(arms: Arms, arm: str, cell: str) -> list[dict[str, Any]]:
    rows = arms.table(arm, "property-replicates.csv.gz")
    rows = rows.loc[rows["cell"] == cell]
    records = []
    for revision in (CODE["F"], AFTER):
        expected = committed(
            revision, f"tests/canonical/{arms.study.directory}/property-replicates.csv.gz"
        )
        expected = expected.loc[expected["cell"] == cell]
        result = drift(expected, rows, PROPERTY_KEYS)
        records += [
            _record(
                arms.study.directory,
                "control_rows",
                f"{arm} against {label(revision)}, {cell}",
                key,
                value,
            )
            for key, value in result.items()
        ]
    return records


def compare(root: Path, *, publish: Path | None = None) -> pd.DataFrame:
    """Every declared record for both studies, one statistic per row."""
    records: list[dict[str, Any]] = []
    for study in STUDIES:
        arms = Arms(root, study)
        for arm in ARMS:
            records += _provenance(arms, arm)
            if publish is not None:
                target = publish / f"{study.directory}-{arm}-manifest.json"
                write_lines(target, arms.output(arm, "manifest.json").read_text(encoding="utf-8"))

        digests = {arm: arms.samples_digest(arm) for arm in ARMS}
        identical = len(set(digests.values())) == 1
        records += [
            _record(study.directory, "samples", arm, "decompressed_sha256", digest)
            for arm, digest in digests.items()
        ]
        records.append(_record(study.directory, "samples", "all arms", "identical", identical))

        failures = fixed_condition_failures(arms)
        for failure in failures:
            print(f"{study.directory}: fixed condition broken: {failure}")
        validated = identical and not failures
        for arm, revision in PRECONDITIONS.items():
            held, found = _precondition(arms, arm, revision)
            validated &= held
            records += found
        records.append(_record(study.directory, "harness", "all arms", "validated", validated))

        for axis, level, first, second in COMPARISONS:
            subject = f"{axis} axis at {level}: {first} against {second}"
            for name, keys in study.row_tables.items():
                result = drift(
                    arms.table(first, name), arms.table(second, name), keys, tolerance=TOLERANCE
                )
                records += [
                    _record(study.directory, "axis_rows", subject, f"{name}: {key}", value)
                    for key, value in result.items()
                ]

        for cell in study.cells:
            verdicts = {}
            for arm in ARMS:
                row = _cell_row(arms, arm, cell)
                verdicts[arm] = _verdict(row)
                records += [
                    _record(study.directory, "statistic", f"{cell.name}, {arm}", key, row[key])
                    for key in dict.fromkeys((*cell.statistics, *VERDICTS[cell.table][1]))
                ]
                records.append(
                    _record(
                        study.directory, "verdict", f"{cell.name}, {arm}", "passed", verdicts[arm]
                    )
                )
            changes = {
                (axis, level): verdicts[first] != verdicts[second]
                for axis, level, first, second in COMPARISONS
            }
            records += [
                _record(
                    study.directory,
                    "verdict_change",
                    f"{cell.name}, {axis} axis at {level}",
                    "changed",
                    changed,
                )
                for (axis, level), changed in changes.items()
            ]
            records.append(
                _record(
                    study.directory,
                    "reading",
                    cell.name,
                    "reading",
                    reading(changes) if validated else NOT_VALIDATED,
                )
            )

        for cell_name in study.control_cells:
            for arm in ARMS:
                records += _control_drift(arms, arm, cell_name)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", type=Path, required=True, help="the scratch root of the arms")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"where to write the table; the committed record is {HERE / 'isolation.csv'}",
    )
    parser.add_argument(
        "--publish-manifests",
        type=Path,
        required=True,
        help=(
            "copy each arm's manifest.json here, with LF line endings; the committed copies "
            f"are in {HERE / 'arms'}"
        ),
    )
    arguments = parser.parse_args()
    arguments.publish_manifests.mkdir(parents=True, exist_ok=True)
    frame = compare(arguments.arms, publish=arguments.publish_manifests)
    write_csv(frame, arguments.output)
    shown = frame.loc[frame["section"].isin(["harness", "samples", "verdict_change", "reading"])]
    with pd.option_context("display.width", 250, "display.max_rows", None):
        print(shown.to_string(index=False))
        print(frame.loc[frame["section"] == "statistic"].to_string(index=False))
    print(f"wrote {len(frame)} rows to {arguments.output}")


if __name__ == "__main__":
    main()
