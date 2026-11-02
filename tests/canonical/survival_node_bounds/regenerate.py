"""Regenerate node-bound evidence on survival fits (``survival-node-bounds``).

The ``survtmle`` comparator (P1 with ``method="mean"``, P2 with ``method="hazard"``) runs in the
image :data:`REFERENCE` builds, through ``run_study.R`` beside this file, and the manifest
records the bytes of its Dockerfile, runner and harness.  The declared comparator run reuses the
committed property phase: ``regenerate --skip-properties``.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_survival_node_bounds, survival_node_bounds_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-survtmle:1.1.1",
    runner="survival_node_bounds/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "survtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_survival_node_bounds,
        survival_node_bounds_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
