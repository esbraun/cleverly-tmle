"""Regenerate censored price-plan evidence (``mtp-pricing-censored``) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard (a clean tree, ``HEAD`` equal to its upstream, every thread variable 1), the
run log and the copy into this directory.  A stopped run resumes from its checkpoints when the
same command runs again at the same commit with a new empty ``--output``.

The R ``lmtp`` comparator is pending (``STUDY.pending_reference``): the driver starts no
reference process.  :data:`REFERENCE` names the pinned ``lmtp`` image and the pricing runner,
which the manifest records as declared sources, a placeholder until the comparator change
adapts the runner to the censoring columns.

**Red-cell reading** (declared before any run): the module docstring of
:mod:`tests.studies.canonical_mtp_pricing_censored` holds it.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.canonical.regenerate import Reference
from tests.studies import canonical_mtp_pricing_censored, mtp_pricing_censored_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="mtp_pricing_runner.R",
    mount_runner=True,
    extra_files=(
        "lmtp_mtp_adapter.R",
        "lmtp_policy_adapter.R",
        "lmtp_sdr_adapter.R",
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    run(
        canonical_mtp_pricing_censored,
        mtp_pricing_censored_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
