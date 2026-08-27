"""Regenerate known-stochastic-regime evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_stochastic_regimes, stochastic_regime_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="stochastic_regimes/run_study.R",
    mount_runner=True,
    extra_files=("lmtp_point_adapter.R", "study_harness.R"),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_stochastic_regimes,
        stochastic_regime_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
