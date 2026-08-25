"""The regeneration driver every registered study shares.

A study's own ``regenerate.py`` says which study it is and how to reach its reference
implementation, and nothing else.  The sequence -- draw, fit, run the reference on the same
rows, validate the replicate contract, write six artefacts and a manifest, then refuse the run
if any gate failed -- is identical for every one of them, and it was written out seven times
before this file existed.  The costly consequence was not the duplication itself: it was that a
fix landed in the copy the author happened to be in.  The survival study's copy grew a
zero-row property fallback for smoke runs and an empty-frame guard on the failure query, and
the end-of-study copy it was cloned from still lacked both.

Three shapes have to fit through one driver, so they are declared rather than branched on:

* **A mounted runner.**  The ``ltmle`` studies bind their reference sources at ``/fixture`` and
  pass the script as an argument, so the image carries only the packages.  Related studies can
  share a Docker context while retaining separate runners and artifact directories.
* **A baked runner.**  The ``tmle3`` and ``ctmle`` studies ``COPY`` the script into the image
  and name it in the ``ENTRYPOINT``, so the container takes only the three data paths.
* **No reference at all.**  ``cvtmle_fold`` compares against nothing, because no maintained
  package ships its construction.  It writes a valid *empty* equivalence artefact rather than a
  surrogate comparison, and its ``draw_and_fit`` returns rows directly instead of the
  ``(samples, truths, rows)`` triple a paired study has to hand to R.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pandas as pd

from tests.parallel import available_cores
from tests.studies.evidence.comparison import empty_equivalence, equivalence
from tests.studies.evidence.manifest import write_csv, write_manifest
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.properties import REPLICATE_COLUMNS as PROPERTY_COLUMNS
from tests.studies.evidence.schema import REPLICATE_COLUMNS, validate_replicates

#: Every published artefact, in the order the manifest hashes them.
ARTIFACT_NAMES = (
    "replicates.csv.gz",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.csv.gz",
    "properties.csv",
)

#: Columns the reference writes as text and the schema requires as numbers.
_TEXT_COLUMNS = frozenset({"implementation", "scenario", "estimand", "inference_scale"})


@dataclass(frozen=True)
class Reference:
    """How to run one study's canonical comparator, and what to hash as its provenance.

    ``mount_runner`` picks between the two container conventions.  ``False`` is the baked form:
    the image's ``ENTRYPOINT`` already names the script, so the container is handed only the
    three data paths.  ``True`` binds ``runner_root`` read-only at ``/fixture`` and passes the
    script as the first argument.  The root defaults to the study directory.

    ``extra_files`` are further reference sources whose bytes belong in the manifest -- a
    sourced adapter or a second script a maintainer runs by hand, for instance.
    """

    image: str
    runner: str
    mount_runner: bool = False
    extra_files: tuple[str, ...] = ()
    #: Optional shared Docker context.  The cross-fitted longitudinal studies use one
    #: digest-pinned ``lmtp`` image while keeping separate runners and artefact directories.
    build_context: Path | None = None
    #: Optional root mounted at ``/fixture``.  ``runner`` and ``extra_files`` are relative
    #: to this root when supplied; existing studies continue to resolve them from ``here``.
    runner_root: Path | None = None

    def files(self, here: Path) -> list[Path]:
        context = self.build_context or here
        root = self.runner_root or here
        return [
            context / "Dockerfile",
            root / self.runner,
            *(root / name for name in self.extra_files),
        ]

    def run(self, here: Path, samples: Path, truths: Path, output: Path, *, cores: int) -> None:
        context = self.build_context or here
        root = self.runner_root or here
        subprocess.run(["docker", "build", "-t", self.image, str(context)], check=True)
        mounts = ["-v", f"{samples.parent.resolve()}:/work"]
        arguments: list[str] = []
        if self.mount_runner:
            mounts += ["-v", f"{root.resolve()}:/fixture:ro"]
            arguments.append(f"/fixture/{self.runner}")
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                f"CLEVERLY_R_CORES={cores}",
                *mounts,
                self.image,
                *arguments,
                f"/work/{samples.name}",
                f"/work/{truths.name}",
                f"/work/{output.name}",
            ],
            check=True,
        )


@dataclass
class _Phase:
    """The Python side's output, whichever shape the study's ``draw_and_fit`` returns."""

    rows: pd.DataFrame
    samples: pd.DataFrame | None = None
    truths: pd.DataFrame | None = None
    cached: bool = False
    paths: dict[str, Path] = field(default_factory=dict)


def _arguments(study: ModuleType, here: Path, reference: Reference | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Regenerate {study.STUDY.name} artefacts.")
    parser.add_argument("--replicates", type=int, default=study.PRIMARY_REPLICATES)
    parser.add_argument("--n", type=int, default=study.PRIMARY_N)
    parser.add_argument("--skip-properties", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="write disposable primary diagnostics without properties or a manifest",
    )
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument(
        "--cache", type=Path, help="reuse a previous Python phase from this directory"
    )
    parser.add_argument(
        "--refresh-python",
        action="store_true",
        help="refit the Python phase while retaining a compatible cached reference result",
    )
    if reference is not None:
        parser.add_argument(
            "--skip-r", action="store_true", help="reuse the committed reference rows"
        )
        parser.add_argument(
            "--r-jobs",
            type=int,
            help="reference-process concurrency (defaults to --jobs)",
        )
    arguments = parser.parse_args()
    if arguments.replicates < 2 or arguments.n < 50:
        parser.error("replicates must be >= 2 and n must be >= 50")
    if getattr(arguments, "r_jobs", None) is not None and arguments.r_jobs < 1:
        parser.error("--r-jobs must be >= 1")
    return arguments


def _python_phase(study: ModuleType, arguments: argparse.Namespace, scratch: Path) -> _Phase:
    """Draw the samples and fit the subject, reusing a cached phase when one is declared."""
    paths = {
        name: scratch / name
        for name in ("samples.csv.gz", "truth.csv", "python-rows.csv.gz", "r-results.csv")
    }
    cache = getattr(arguments, "cache", None)
    reusable = ("samples.csv.gz", "truth.csv", "python-rows.csv.gz")
    if cache and not arguments.refresh_python and all(paths[name].exists() for name in reusable):
        rows = pd.read_csv(paths["python-rows.csv.gz"])
        if rows["replicate"].nunique() != arguments.replicates:
            raise RuntimeError("the cached Python phase has the wrong replication count")
        print(f"reusing the cached Python phase in {scratch}", flush=True)
        return _Phase(rows=rows, cached=True, paths=paths)

    drawn = study.draw_and_fit(
        replicates=arguments.replicates, n=arguments.n, n_jobs=arguments.jobs
    )
    if not isinstance(drawn, tuple):
        # A study with no comparator has no reason to keep the realized rows around: nothing
        # else ever reads them, so it returns the estimate table straight out.
        return _Phase(rows=drawn, paths=paths)
    samples, truths, rows = drawn
    write_csv(samples, paths["samples.csv.gz"], compression="gzip")
    write_csv(truths, paths["truth.csv"])
    write_csv(rows, paths["python-rows.csv.gz"], compression="gzip")
    return _Phase(rows=rows, samples=samples, truths=truths, paths=paths)


def _reference_rows(
    study: ModuleType,
    reference: Reference,
    arguments: argparse.Namespace,
    here: Path,
    phase: _Phase,
) -> pd.DataFrame:
    if getattr(arguments, "skip_r", False):
        committed = pd.read_csv(here / "replicates.csv.gz")
        rows = committed.loc[
            (committed["implementation"] == study.STUDY.reference)
            & (committed["replicate"] < arguments.replicates)
            & (committed["n"] == arguments.n)
        ]
        if rows.empty:
            raise RuntimeError("no compatible committed reference rows for --skip-r")
        return rows
    cached_reference = phase.paths["r-results.csv"]
    if (phase.cached or getattr(arguments, "cache", None)) and cached_reference.exists():
        rows = pd.read_csv(cached_reference)
        expected = set(range(arguments.replicates))
        observed = set(rows["replicate"].unique())
        if observed != expected or set(rows["n"].unique()) != {arguments.n}:
            raise RuntimeError("the cached reference phase has incompatible replications")
        print(f"reusing the cached reference phase in {cached_reference.parent}", flush=True)
        return rows
    reference.run(
        here,
        phase.paths["samples.csv.gz"],
        phase.paths["truth.csv"],
        phase.paths["r-results.csv"],
        cores=arguments.r_jobs or arguments.jobs,
    )
    return pd.read_csv(cached_reference)


def _property_artifacts(
    properties: ModuleType,
    arguments: argparse.Namespace,
    here: Path,
    paths: dict[str, Path],
) -> pd.DataFrame | None:
    """The property study, or the committed rows, or a schema-bearing placeholder.

    Returns the summary when it was computed, and ``None`` when the artefacts were reused or
    stubbed -- so the failure gate below can tell "every cell passed" from "no cell ran".
    """
    if not arguments.skip_properties:
        rows = properties.generate_property_rows(n_jobs=arguments.jobs)
        write_csv(
            rows, paths["property-replicates.csv.gz"], compression={"method": "gzip", "mtime": 0}
        )
        summary = properties.summarize_properties(rows)
        write_csv(summary, paths["properties.csv"])
        return summary

    names = ("property-replicates.csv.gz", "properties.csv")
    if all((here / name).exists() for name in names):
        for name in names:
            if paths[name].resolve() != (here / name).resolve():
                paths[name].write_bytes((here / name).read_bytes())
        return None
    if paths["properties.csv"].parent.resolve() == here.resolve():
        raise RuntimeError("skipping the property study needs committed property artefacts")
    # A disposable smoke run has nothing to reuse.  Empty, schema-bearing files keep its
    # manifest complete without pretending the statistical study ran; publication to the
    # committed directory still refuses this state above.
    write_csv(
        pd.DataFrame(columns=list(PROPERTY_COLUMNS)),
        paths["property-replicates.csv.gz"],
        compression={"method": "gzip", "mtime": 0},
    )
    write_csv(pd.DataFrame(columns=["property", "cell", "passed"]), paths["properties.csv"])
    return None


def main(
    study: ModuleType,
    properties: ModuleType,
    *,
    here: Path,
    reference: Reference | None = None,
) -> None:
    """Regenerate one study's committed artefacts, and refuse the run if a gate failed."""
    arguments = _arguments(study, here, reference)
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)
    record = dataclasses.replace(
        study.STUDY, replicates=arguments.replicates, n=arguments.n, artifacts=out
    )
    print(f"regenerating {record.name}: {record.replicates} x n={record.n}", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"cleverly-{record.slug}-") as raw:
        cache = getattr(arguments, "cache", None)
        scratch = Path(cache) if cache else Path(raw)
        scratch.mkdir(parents=True, exist_ok=True)
        phase = _python_phase(study, arguments, scratch)
        rows = phase.rows
        if reference is not None:
            rows = pd.concat(
                [phase.rows, _reference_rows(study, reference, arguments, here, phase)],
                ignore_index=True,
            )

    extra_frames = study.extra_artifacts(rows) if hasattr(study, "extra_artifacts") else {}
    if set(extra_frames) != set(record.extra_artifacts):
        raise RuntimeError(
            f"{record.slug} produced extra artifacts {sorted(extra_frames)}, "
            f"expected {sorted(record.extra_artifacts)}"
        )
    for column in REPLICATE_COLUMNS:
        if column not in _TEXT_COLUMNS:
            rows[column] = pd.to_numeric(rows[column], errors="raise")
    rows = rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["scenario", "replicate", "estimand", "implementation"], ignore_index=True
    )
    validate_replicates(rows, record=record)

    artifact_names = (*ARTIFACT_NAMES, *record.extra_artifacts)
    paths = {name: out / name for name in artifact_names}
    write_csv(rows, paths["replicates.csv.gz"], compression={"method": "gzip", "mtime": 0})
    summaries = summarize(rows)
    write_csv(summaries, paths["summary.csv"])
    performance = independent_performance_tests(rows, record=record, n_jobs=arguments.jobs)
    write_csv(performance, paths["performance-tests.csv"])
    paired = (
        empty_equivalence()
        if reference is None
        else equivalence(rows, summaries, performance, record=record, n_jobs=arguments.jobs)
    )
    write_csv(paired, paths["equivalence.csv"])
    for name, frame in extra_frames.items():
        write_csv(frame, paths[name])

    if arguments.primary_only:
        print(performance.to_string(index=False))
        print(paired.to_string(index=False))
        print(
            "primary-only probe: no property artefacts or publishable manifest were written",
            flush=True,
        )
        return

    property_summary = _property_artifacts(properties, arguments, here, paths)

    write_manifest(
        out / "manifest.json",
        record,
        [paths[name] for name in artifact_names],
        reference_files=() if reference is None else reference.files(here),
        reference_metadata=getattr(study, "REFERENCE_METADATA", None),
        configuration=study.CONFIGURATION,
    )

    failures = {
        "independent performance": performance.loc[~performance["passed"]],
        # Both columns, not just ``passed``: a family whose claim spans its cells records that
        # verdict in ``property_passed`` alone, so gating on the per-row column would let a
        # failed joint claim through while every row read green.
        "statistical property": None
        if property_summary is None
        else property_summary.loc[
            ~property_summary["passed"] | ~property_summary["property_passed"]
        ],
    }
    if reference is not None:
        failures["paired similarity and non-inferiority"] = paired.loc[~paired["passed"]]
        # The subject's own verdict is gated unconditionally above.  This one is about the
        # *comparator*, and a study may declare in advance that its comparator fails its own
        # truth gates while remaining a usable similarity and non-inferiority reference --
        # see ``StudyRecord.accepted_reference_failure``.  Without the declaration the run
        # still refuses, so an unannounced reference regression cannot pass silently.
        if not record.accepted_reference_failure:
            failures["reference validity"] = paired.loc[~paired["reference_valid"]]
        elif paired["reference_valid"].all():
            raise RuntimeError(
                f"{record.slug} declares an accepted reference failure "
                f"({record.accepted_reference_failure!r}) but every reference row is valid; "
                f"remove the declaration rather than carrying a stale exception"
            )
    if hasattr(study, "scientific_failures"):
        failures.update(study.scientific_failures(extra_frames))
    reported = {
        name: frame for name, frame in failures.items() if frame is not None and not frame.empty
    }
    if reported and record.publication_policy == "gated" and not arguments.allow_failures:
        raise RuntimeError(
            "\n\n".join(
                f"{name} gates failed:\n{frame.to_string(index=False)}"
                for name, frame in reported.items()
            )
        )
    if reported and record.publication_policy == "reporting":
        print(
            "reporting policy: published scientific failures:\n"
            + "\n\n".join(
                f"{name}:\n{frame.to_string(index=False)}" for name, frame in reported.items()
            ),
            flush=True,
        )
    print(f"wrote {len(artifact_names)} artefacts and a manifest to {out}", flush=True)
