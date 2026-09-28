"""The run form of the two RM30 learned-rule studies: rules L6 to L9.

Both ``regenerate.py`` drivers call :func:`run`.  The regeneration itself is the shared
:func:`tests.canonical.regenerate.main`.  This module adds the run form in front of it.

A **declared run** passes no ``--replicates``.  It takes ``--output`` and ``--jobs`` and refuses
every other flag.  ``--output`` names an empty scratch directory outside the repository.  The run
refuses to start unless RM18 rule R6 holds (through
:func:`tests.diagnostics.rm18_shared.refusals`: a clean tree, ``HEAD`` equal to its upstream,
and ``cleverly`` imported from this tree's ``src``), the runtime is the one rule L7 declares,
and every thread variable is 1.  The run writes its artefacts, its manifest and a ``run.log``
into the scratch directory, so the tree is still clean when the manifest records its
provenance.  It then copies them into the study directory.  The copy happens whenever the driver
wrote a manifest, so a gated study whose verdict fails still publishes the run, and the run
then stops with the driver's error.

A **smoke run** passes a ``--replicates`` other than the declared budget.  It runs on the
throwaway record of rule L9, whose seeds come from the labels ``("rm30", "smoke")`` and
``("rm30", "smoke-resampling")``.  It refuses an ``--output`` inside the repository, and it
refuses a sample seed that equals a declared sample seed of either study.  The shared driver
then skips the property study, and a reading says ``smoke run, not the declared budget``.

``--property-harness-check CAP`` is a smoke-only check of the property path.  It fits at most
``CAP`` draws of each property cell of the throwaway record and prints the rule agreement, the
solver warnings and the wall time.  It prints no bias, coverage or SE ratio.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from tests.canonical.regenerate import ARTIFACT_NAMES
from tests.canonical.regenerate import main as regenerate
from tests.diagnostics.rm18_shared import THREAD_VARIABLES, refusals, run_log
from tests.parallel import available_cores
from tests.studies import _learned_rule_law as law
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The interpreter version of rules L6 and L7.
PYTHON = "3.13.7"

#: The package versions of rule L7, by import name.
PACKAGES = {
    "numpy": "2.4.6",
    "scipy": "1.18.0",
    "pandas": "3.0.5",
    "sklearn": "1.9.0",
    "joblib": "1.5.3",
}

#: What a declared run writes beside the study's extra artefacts.
RECORD_FILES = ("manifest.json", "run.log")


def declared_sample_seeds() -> set[int]:
    """Every sample seed a declared run of either study draws."""
    from tests.studies import learned_rule_cvtmle as gated
    from tests.studies import learned_rule_cvtmle_boundary as boundary
    from tests.studies import learned_rule_cvtmle_properties as properties

    return law.declared_sample_seeds(
        gated.STUDY,
        gated.PRIMARY_REPLICATES,
        properties.PROPERTY_DRAWS,
        properties.suffixes(gated.STUDY),
    ) | law.declared_sample_seeds(boundary.STUDY, boundary.PRIMARY_REPLICATES, (), {})


def runtime_refusals() -> list[str]:
    """Why the runtime is not the one rules L6 and L7 declare, with one thread per library."""
    import importlib

    out = []
    if platform.python_version() != PYTHON:
        out.append(f"Python is {platform.python_version()}, not {PYTHON} (rules L6 and L7)")
    for name, version in PACKAGES.items():
        found = importlib.import_module(name).__version__
        if found != version:
            out.append(f"{name} is {found}, not {version} (rule L7)")
    out.extend(
        f"{name} is {os.environ.get(name, 'unset')}, not 1 (rule R6)"
        for name in THREAD_VARIABLES
        if os.environ.get(name) != "1"
    )
    return out


def _outside(output: Path, kind: str) -> Path:
    """``output`` resolved, or a refusal when it lies inside the repository."""
    resolved = output.resolve()
    if resolved.is_relative_to(ROOT.resolve()):
        raise SystemExit(
            f"refused: a {kind} run writes to a scratch --output outside the repository, "
            f"not {resolved}"
        )
    return resolved


def publish_run(
    study: Any,
    properties: ModuleType,
    *,
    here: Path,
    output: Path,
    arguments: Sequence[str],
    title: str,
) -> None:
    """Run the shared driver into the scratch ``output`` and copy the record into ``here``.

    The driver writes its artefacts and manifest into ``output``, outside the repository, so the
    provenance the manifest records sees the tree as the run found it.  ``run_log`` then closes
    its block, and every artefact, the manifest and the log are copied into ``here``.  A driver
    error after the manifest exists is a failed verdict of a gated study: the record is copied
    and the error is raised again.  A driver error before the manifest exists copies nothing.
    """
    names = (*ARTIFACT_NAMES, *study.STUDY.extra_artifacts, *RECORD_FILES)
    sys.argv = ["regenerate", *arguments, "--output", str(output)]
    failure: BaseException | None = None
    try:
        with run_log(output, title):
            regenerate(study, properties, here=here)
    except RuntimeError as error:
        if not (output / "manifest.json").exists():
            raise
        failure = error
    missing = [name for name in names if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"the run wrote no {missing} to {output}; nothing is copied")
    here.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(output / name, here / name)
    print(f"copied {len(names)} files from {output} to {here}", flush=True)
    if failure is not None:
        raise failure


def _declared(study: ModuleType, properties: ModuleType, here: Path, rest: Sequence[str]) -> None:
    """A declared run: the scratch output, the guard, the run log and the copy."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=available_cores())
    known, unknown = parser.parse_known_args(list(rest))
    if unknown:
        raise SystemExit(
            f"refused: a declared run takes --output and --jobs only, not {unknown} (rule L11)"
        )
    if known.output is None:
        raise SystemExit("refused: a declared run needs a scratch --output outside the repository")
    output = _outside(known.output, "declared")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refused: the scratch --output {output} is not empty")
    refused = refusals(False) + runtime_refusals()
    if refused:
        raise SystemExit("refused:\n  " + "\n  ".join(refused))
    output.mkdir(parents=True, exist_ok=True)
    publish_run(
        study,
        properties,
        here=here,
        output=output,
        arguments=("--jobs", str(known.jobs)),
        title=f"{study.STUDY.slug} declared run",
    )


def _smoke_study(study: ModuleType, record: Any) -> SimpleNamespace:
    """``study`` with its record and its sampler bound to the throwaway record."""
    names = {name: getattr(study, name) for name in dir(study) if not name.startswith("__")}
    proxy = SimpleNamespace(**names)
    proxy.STUDY = record
    proxy.draw_and_fit = partial(study.draw_and_fit, record=record)
    return proxy


def _property_harness_check(
    properties: ModuleType, record: Any, cap: int, output: Path, jobs: int
) -> None:
    """Fit ``cap`` draws of each property cell of ``record`` and print the harness readings."""
    if not hasattr(properties, "PROPERTY_DRAWS"):
        print("the study declares no property cell, so there is nothing to check", flush=True)
        return
    suffixes = properties.suffixes(record)
    clash = {
        stream_seed(
            record,
            "property_sample",
            draw.family,
            draw.label,
            r,
            *suffixes[draw.family, draw.label],
        )
        for draw in properties.PROPERTY_DRAWS
        for r in range(min(cap, draw.replicates))
    } & declared_sample_seeds()
    if clash:
        raise SystemExit(f"refused: the property check draws declared sample seeds {sorted(clash)}")
    start = time.perf_counter()
    rows = properties.generate_property_rows(n_jobs=jobs, record=record, cap=cap)
    wall = time.perf_counter() - start
    write_csv(
        rows, output / "property-harness-check.csv.gz", compression={"method": "gzip", "mtime": 0}
    )
    fitted = rows.loc[rows["rule_rows_checked"] > 0]
    agreement = (fitted["rule_rows_checked"] == fitted["n"]).groupby(fitted["cell"]).all()
    warnings = rows.groupby("cell")["solver_warnings"].sum()
    print(f"property harness check, {len(rows)} rows, wall {wall:.1f} s", flush=True)
    print(f"rule agreement 1 on every fitted row, by cell:\n{agreement.to_string()}", flush=True)
    print(f"solver warnings, by cell:\n{warnings.to_string()}", flush=True)


def run(study: ModuleType, properties: ModuleType, *, here: Path) -> None:
    """Regenerate one learned-rule study under the RM30 run form."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--property-harness-check", type=int, default=None)
    known, rest = parser.parse_known_args(sys.argv[1:])
    if known.replicates is None:
        if known.property_harness_check is not None:
            raise SystemExit("refused: --property-harness-check is a smoke-only check")
        _declared(study, properties, here, rest)
        return
    if known.replicates == study.PRIMARY_REPLICATES:
        raise SystemExit(
            "refused: a declared run passes no --replicates, and a smoke run passes a count "
            f"other than the declared {study.PRIMARY_REPLICATES}"
        )

    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--output", type=Path, default=here)
    options.add_argument("--jobs", type=int, default=available_cores())
    smoke, _ = options.parse_known_args(list(rest))
    output = _outside(smoke.output, "smoke")
    record = law.smoke_record(study.STUDY)
    seeds = {
        replicate_seed(record, scenario, r)
        for scenario in record.scenarios
        for r in range(known.replicates)
    }
    clash = seeds & declared_sample_seeds()
    if clash:
        raise SystemExit(f"refused: the smoke draws declared sample seeds {sorted(clash)}")
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = ["regenerate", "--replicates", str(known.replicates), *rest]
    with run_log(output, f"{study.STUDY.slug} smoke, {known.replicates} draws"):
        regenerate(_smoke_study(study, record), properties, here=here)
        if known.property_harness_check is not None:
            _property_harness_check(
                properties, record, known.property_harness_check, output, smoke.jobs
            )
