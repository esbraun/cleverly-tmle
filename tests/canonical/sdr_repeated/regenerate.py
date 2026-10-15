"""Regenerate repeated-outcome longitudinal SDR evidence (``longitudinal-repeated-sdr``).

The study declares the lmtp comparator as pending, so this driver runs no reference process.
The pending mechanism needs a ``Reference``, because the manifest records the declared source
bytes and the type requires a runner path.  The runner is ``longitudinal_sdr_runner.R``, with
the adapter ``lmtp_sdr_adapter.R`` and the ``lmtp_crossfit`` Dockerfile, which plan X32b section
K12 reuses unchanged.  An edit to that runner for ``longitudinal-sdr`` is result-neutral for this
pending study until PR C replaces it (``tests/canonical/provenance-revisions.md``).  The change
that registers the lmtp rows moves the study's ``pending_reference`` to ``reference``.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_sdr_repeated, sdr_repeated_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="longitudinal_sdr_runner.R",
    mount_runner=True,
    extra_files=("lmtp_sdr_adapter.R",),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_sdr_repeated,
        sdr_repeated_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
