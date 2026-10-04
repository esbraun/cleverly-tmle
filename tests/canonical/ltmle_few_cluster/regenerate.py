"""Regenerate the few-cluster t-reference evidence under the declared run form.

The primary rows pair with R ``ltmle`` 1.3-0. A declared run passes ``--output`` (an empty
scratch directory outside the repository) and optionally ``--jobs``, and nothing else.
:mod:`tests.canonical.declared_run` describes the guard, the run log and the copy into this
directory.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.canonical.regenerate import Reference
from tests.studies import clustered_few_cluster_properties, clustered_few_cluster_tmle

ROOT = Path(__file__).parents[3]

REFERENCE = Reference(
    image="cleverly-ltmle-reference:1.3-0",
    runner="ltmle_few_cluster/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "ltmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    run(
        clustered_few_cluster_tmle,
        clustered_few_cluster_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )
