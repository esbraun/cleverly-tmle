"""The declared run form of a registered regeneration: a guard, a scratch output, a log, a copy.

:func:`tests.canonical.regenerate.main` writes its artifacts before :func:`write_manifest`
records ``git status``.  A run into the study directory therefore records
``cleverly_worktree_clean: false`` even from a clean pushed commit.  The declared run form
closes that gap.  It writes the artifacts, the manifest and a ``run.log`` into a scratch
``--output`` outside the repository, and it copies them into the study directory afterwards.

A **declared run** passes no ``--replicates``.  It takes ``--output``, ``--jobs`` and ``--fresh``
and refuses every other flag.  It refuses to start unless RM18 rule R6 holds (through
:func:`tests.diagnostics.rm18_shared.refusals`: a clean tree, ``HEAD`` equal to its upstream,
and ``cleverly`` imported from this tree's ``src``) and every thread variable is 1.  A study
can add refusals of its own.  The copy happens whenever the driver wrote a manifest, so a gated
study whose verdict fails still publishes the run, and the run then stops with the driver's
error.

A declared run keeps its scratch directory and its checkpoints until its PR merges, for audit.
The driver prints both paths, and the implementer records them in the row's progress file.  A
crashed declared run resumes from them: run it again with a new empty ``--output``, and
``run.log`` records what it reused.  ``--fresh`` discards them first.

A **smoke run** passes a ``--replicates`` other than the declared primary budget.  It goes to
the shared driver unchanged, which then skips the property study.  It refuses an ``--output``
inside the repository, so it cannot overwrite a committed artifact.

The two RM30 learned-rule studies and ``canonical-multi-arm-drtmle`` (RM18 Design SL) use this
form.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from tests.canonical.regenerate import ARTIFACT_NAMES, RUN_NOTES, Reference
from tests.canonical.regenerate import main as regenerate
from tests.diagnostics.rm18_shared import THREAD_VARIABLES, note, refusals, run_log
from tests.parallel import available_cores
from tests.studies.evidence.registry import ROOT

#: What a declared run writes beside the study's artifacts.
RECORD_FILES = ("manifest.json", "run.log")


def thread_refusals() -> list[str]:
    """Each thread variable that is not 1, as RM18 rule R6 requires of a declared run."""
    return [
        f"{name} is {os.environ.get(name, 'unset')}, not 1 (rule R6)"
        for name in THREAD_VARIABLES
        if os.environ.get(name) != "1"
    ]


def outside(output: Path, kind: str) -> Path:
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
    reference: Reference | None = None,
) -> None:
    """Run the shared driver into the scratch ``output`` and copy the record into ``here``.

    The driver writes its artifacts and manifest into ``output``, outside the repository, so the
    provenance the manifest records sees the tree as the run found it.  ``run_log`` then closes
    its block, and every artifact, the manifest and the log are copied into ``here``.  A driver
    error after the manifest exists is a failed verdict of a gated study: the record is copied
    and the error is raised again.  A driver error before the manifest exists copies nothing.
    """
    names = (*ARTIFACT_NAMES, *study.STUDY.extra_artifacts, *RECORD_FILES)
    invoked = list(sys.argv)
    sys.argv = ["regenerate", *arguments, "--output", str(output)]
    failure: BaseException | None = None
    RUN_NOTES.clear()
    try:
        with run_log(output, title, argv=invoked):
            try:
                regenerate(study, properties, here=here, reference=reference)
            finally:
                for line in RUN_NOTES:
                    note(line)
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


def declared(
    study: ModuleType,
    properties: ModuleType,
    *,
    here: Path,
    arguments: Sequence[str],
    reference: Reference | None = None,
    study_refusals: Callable[[], list[str]] = list,
    flag_rule: str | None = None,
) -> None:
    """A declared run: the guard, the scratch output, the run log and the copy.

    ``flag_rule`` names the study's own declared rule in the refusal of an extra flag, as RM30
    names its rule L11.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument("--fresh", action="store_true")
    known, unknown = parser.parse_known_args(list(arguments))
    if unknown:
        cited = "" if flag_rule is None else f" ({flag_rule})"
        raise SystemExit(
            f"refused: a declared run takes --output, --jobs and --fresh only, not {unknown}{cited}"
        )
    if known.output is None:
        raise SystemExit("refused: a declared run needs a scratch --output outside the repository")
    output = outside(known.output, "declared")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refused: the scratch --output {output} is not empty")
    refused = refusals(False) + thread_refusals() + study_refusals()
    if refused:
        raise SystemExit("refused:\n  " + "\n  ".join(refused))
    output.mkdir(parents=True, exist_ok=True)
    publish_run(
        study,
        properties,
        here=here,
        output=output,
        arguments=(
            "--jobs",
            str(known.jobs),
            "--keep-cache",
            *(("--fresh",) if known.fresh else ()),
        ),
        title=f"{study.STUDY.slug} declared run",
        reference=reference,
    )


def run(
    study: ModuleType,
    properties: ModuleType,
    *,
    here: Path,
    reference: Reference | None = None,
) -> None:
    """Regenerate one registered study under the declared run form, or smoke it into scratch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    known, rest = parser.parse_known_args(sys.argv[1:])
    if known.replicates is None:
        forwarded = [] if known.output is None else ["--output", str(known.output)]
        declared(study, properties, here=here, arguments=[*rest, *forwarded], reference=reference)
        return
    if known.replicates == study.PRIMARY_REPLICATES:
        raise SystemExit(
            "refused: a declared run passes no --replicates, and a smoke run passes a count "
            f"other than the declared {study.PRIMARY_REPLICATES}"
        )
    if known.output is None:
        raise SystemExit("refused: a smoke run needs a scratch --output outside the repository")
    output = outside(known.output, "smoke")
    sys.argv = ["regenerate", "--replicates", str(known.replicates), "--output", str(output), *rest]
    regenerate(study, properties, here=here, reference=reference)
