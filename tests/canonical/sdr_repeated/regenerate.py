"""Regenerate repeated-outcome longitudinal SDR evidence (``longitudinal-repeated-sdr``).

The study declares the lmtp comparator as pending, so this driver runs no reference process.
The pending mechanism needs a ``Reference``, because the manifest records the declared source
bytes.  The runner is the stub ``tests/canonical/longitudinal_repeated_sdr_runner.R``, which
stops with a message.  It belongs to this study alone, and no other study's runner is
referenced.  The change that registers the lmtp rows replaces the stub with the per-horizon
runner (plan K12) and moves the study's ``pending_reference`` to ``reference``.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_sdr_repeated, sdr_repeated_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="longitudinal_repeated_sdr_runner.R",
    mount_runner=True,
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
