"""Regenerate repeated-outcome longitudinal SDR evidence (``longitudinal-repeated-sdr``).

The study declares the lmtp comparator as pending, so this driver runs no reference process.
The ``Reference`` below names the runner that the manifest records as the later comparator
change's source.  The files are the ``longitudinal-sdr`` runner files, which exist today.  The
change that registers the lmtp rows replaces this value with the runner of the repeated-outcome
pairing and moves the study's ``pending_reference`` to ``reference``.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_sdr_repeated, sdr_repeated_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="longitudinal_sdr_runner.R",
    mount_runner=True,
    extra_files=(
        "lmtp_sdr_adapter.R",
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
    ),
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
