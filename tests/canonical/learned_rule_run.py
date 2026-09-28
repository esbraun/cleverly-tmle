"""The run form of the two RM30 learned-rule studies: rules L6, L8 and L9.

Both ``regenerate.py`` drivers call :func:`run`.  The regeneration itself is the shared
:func:`tests.canonical.regenerate.main`.  This module adds three things in front of it.

A **declared run** is one whose ``--replicates`` is the declared budget, the default.  It
applies RM18 rule R6 through :func:`tests.diagnostics.rm18_shared.refusals`: the tree must be
clean, ``HEAD`` must equal its upstream, and ``cleverly`` must import from this tree's ``src``.
Rule L6 adds Python 3.13.7, which the manifest then records.

A **smoke run** is any other ``--replicates``.  It runs on the throwaway record of rule L9,
whose seeds come from the labels ``("rm30", "smoke")`` and ``("rm30", "smoke-resampling")``.
It refuses an ``--output`` inside the repository, and it refuses a sample seed that equals a
declared sample seed of either study.  The shared driver then skips the property study, and a
reading says ``smoke run, not the declared budget``.

``--property-harness-check CAP`` is a smoke-only check of the property path.  It fits at most
``CAP`` draws of each property cell of the throwaway record and prints the rule agreement, the
solver warnings and the wall time.  It prints no bias, coverage or SE ratio.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from tests.canonical.regenerate import main as regenerate
from tests.diagnostics.rm18_shared import refusals, run_log
from tests.parallel import available_cores
from tests.studies import _learned_rule_law as law
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The interpreter version of rule L6 and L7.
PYTHON = "3.13.7"


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


def _arguments(study: ModuleType, here: Path) -> argparse.Namespace:
    """The arguments this wrapper reads.

    ``--property-harness-check`` is this wrapper's own, so it leaves ``sys.argv`` here.  The
    shared driver then parses the rest of the command line itself.
    """
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--property-harness-check", type=int, default=None)
    check, rest = own.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--replicates", type=int, default=study.PRIMARY_REPLICATES)
    shared.add_argument("--output", type=Path, default=here)
    shared.add_argument("--jobs", type=int, default=available_cores())
    known, _ = shared.parse_known_args()
    known.property_harness_check = check.property_harness_check
    return known


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
    arguments = _arguments(study, here)
    if arguments.replicates == study.PRIMARY_REPLICATES:
        if arguments.property_harness_check is not None:
            raise SystemExit("refused: --property-harness-check is a smoke-only check")
        refused = refusals(False)
        if platform.python_version() != PYTHON:
            refused.append(f"Python is {platform.python_version()}, not {PYTHON} (rule L6)")
        if refused:
            raise SystemExit("refused:\n  " + "\n  ".join(refused))
        regenerate(study, properties, here=here)
        return

    output = arguments.output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise SystemExit(
            f"refused: a smoke run writes to a scratch --output outside the repository, "
            f"not {output}"
        )
    record = law.smoke_record(study.STUDY)
    seeds = {
        replicate_seed(record, scenario, r)
        for scenario in record.scenarios
        for r in range(arguments.replicates)
    }
    clash = seeds & declared_sample_seeds()
    if clash:
        raise SystemExit(f"refused: the smoke draws declared sample seeds {sorted(clash)}")
    output.mkdir(parents=True, exist_ok=True)
    with run_log(output, f"{study.STUDY.slug} smoke, {arguments.replicates} draws"):
        regenerate(_smoke_study(study, record), properties, here=here)
        if arguments.property_harness_check is not None:
            _property_harness_check(
                properties, record, arguments.property_harness_check, output, arguments.jobs
            )
