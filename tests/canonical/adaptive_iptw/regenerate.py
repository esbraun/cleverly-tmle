"""Regenerate the AdaptiveIPTW gate evidence (``adaptive-iptw``).

The R ``drtmle`` comparator is pending (``STUDY.pending_reference``).  The driver runs no
reference process.  :data:`REFERENCE` names the pinned image and its runner, so the manifest
records their bytes at the run.  The R runners of the pairing study (``adaptive-iptw-pairing``)
come with the change that declares it.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import adaptive_iptw_properties, canonical_adaptive_iptw
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-drtmle-reference:538a3a2",
    runner="drtmle/run_drtmle.R",
    mount_runner=True,
    build_context=ROOT / "tests" / "canonical" / "drtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_adaptive_iptw,
        adaptive_iptw_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
