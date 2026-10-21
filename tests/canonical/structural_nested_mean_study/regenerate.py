"""Regenerate the structural nested mean evidence (``structural-nested-mean``).

The comparator is EconML 0.17.0 ``DynamicDML``.  :data:`REFERENCE` runs
``tests/canonical/econml_dynamic_dml/run_study.py`` in the ``cleverly-econml-parity`` image on the
samples and the task table of the primary phase.  The manifest records the bytes of the
Dockerfile, the runner, the requirements lock and the parity fixture generator.  While
``STUDY.pending_reference`` is set, the driver runs no reference process and records the sources
only.  The registered record is ``COMPARATOR_STUDY`` of the study module.

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
    runner="econml_dynamic_dml/run_study.py",
    mount_runner=True,
    interpreter="python",
    extra_files=(
        "econml_dynamic_dml/requirements.txt",
        "econml_dynamic_dml/generate_parity_fixture.py",
    ),
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
