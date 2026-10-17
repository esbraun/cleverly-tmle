"""Regenerate node-bound evidence on survival fits (``survival-node-bounds``).

The ``survtmle`` comparator (P1 with ``method="mean"``, P2 with ``method="hazard"``) is pending
(``STUDY.pending_reference``).  The driver runs no reference process.  :data:`REFERENCE` names
the declared sources, so the manifest records their bytes at the run.  The runner script is not
written yet, and the driver hashes it into the manifest, so write it before a declared run.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_survival_node_bounds, survival_node_bounds_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-survtmle:1.1.1",
    runner="survival_node_bounds_runner.R",
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
