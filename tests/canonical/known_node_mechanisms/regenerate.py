"""Regenerate the known-node-mechanism LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_known_node_mechanisms, known_node_mechanisms_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-ltmle-reference:1.3-0",
    runner="known_node_mechanisms/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "ltmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_known_node_mechanisms,
        known_node_mechanisms_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
