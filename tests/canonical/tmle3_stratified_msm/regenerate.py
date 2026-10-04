"""Regenerate the identity-link stratified MSM evidence against R ``tmle3`` under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs``, and nothing else.  :mod:`tests.canonical.declared_run` describes the
guard, the run log and the copy into this directory.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.canonical.regenerate import Reference
from tests.studies import canonical_stratified_msm_identity, stratified_msm_identity_properties
from tests.studies.evidence.registry import ROOT

if __name__ == "__main__":
    run(
        canonical_stratified_msm_identity,
        stratified_msm_identity_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-tmle3-reference:ed72f8a",
            runner="tmle3_stratified_msm/run_study.R",
            mount_runner=True,
            extra_files=("study_harness.R",),
            build_context=ROOT / "tests" / "canonical" / "tmle3",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )
