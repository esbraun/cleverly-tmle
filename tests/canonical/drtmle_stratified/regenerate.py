"""Regenerate the stratified DR-TMLE evidence against R ``drtmle`` under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs``, and nothing else.  :mod:`tests.canonical.declared_run` describes the
guard, the run log and the copy into this directory.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.canonical.regenerate import Reference
from tests.studies import canonical_stratified_drtmle, stratified_drtmle_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-drtmle-reference:538a3a2",
    runner="drtmle_stratified/run_drtmle_stratified.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "drtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    run(
        canonical_stratified_drtmle,
        stratified_drtmle_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
