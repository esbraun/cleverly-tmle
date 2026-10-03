"""Regenerate the few-cluster t-reference evidence, paired with R ltmle 1.3-0."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import clustered_few_cluster_properties, clustered_few_cluster_tmle

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        clustered_few_cluster_tmle,
        clustered_few_cluster_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-ltmle-reference:1.3-0",
            runner="ltmle_few_cluster/run_study.R",
            mount_runner=True,
            extra_files=("study_harness.R",),
            build_context=ROOT / "tests" / "canonical" / "ltmle",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )
