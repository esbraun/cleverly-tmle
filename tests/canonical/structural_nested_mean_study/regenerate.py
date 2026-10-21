"""Regenerate the structural nested mean evidence (``structural-nested-mean``).

The EconML ``DynamicDML`` comparator is pending (``STUDY.pending_reference``).  The driver runs
no reference process.  :data:`REFERENCE` names the declared sources, the Dockerfile, the
requirements lock and the fixture generator of ``tests/canonical/econml_dynamic_dml``, so the
manifest records their bytes at the run.  A study runner for the comparator phase comes with
the later change that registers it.

The caller sets ``CLEVERLY_SUPPORT_RECORD=<root>/studies/structural_nested_mean_study`` on the
run, as the RM42b measurement does for every study, so each fit records its support key.  The
driver does not set it.  ``SUPPORT_KEYS`` of the study module declares the keys the run records.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_structural_nested_mean, structural_nested_mean_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-econml-parity",
    runner="econml_dynamic_dml/generate_parity_fixture.py",
    mount_runner=True,
    interpreter="python",
    extra_files=("econml_dynamic_dml/requirements.txt",),
    build_context=ROOT / "tests" / "canonical" / "econml_dynamic_dml",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_structural_nested_mean,
        structural_nested_mean_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
