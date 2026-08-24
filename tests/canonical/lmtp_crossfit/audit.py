"""Run a retained pinned-``lmtp`` audit outside the registered evidence gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.canonical.regenerate import Reference
from tests.parallel import available_cores
from tests.studies.evidence.registry import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("end", "survival"))
    parser.add_argument("samples", type=Path)
    parser.add_argument("truths", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=available_cores())
    arguments = parser.parse_args()
    parents = {
        path.resolve().parent for path in (arguments.samples, arguments.truths, arguments.output)
    }
    if len(parents) != 1:
        parser.error("samples, truths, and output must share one directory")
    runners = {
        "end": "lmtp_ltmle/run_study.R",
        "survival": "lmtp_ltmle_survival/run_study.R",
    }
    reference = Reference(
        image="cleverly-lmtp-crossfit:1.5.4",
        runner=runners[arguments.study],
        mount_runner=True,
        extra_files=("lmtp_crossfit_adapter.R",),
        build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
        runner_root=ROOT / "tests" / "canonical",
    )
    reference.run(
        ROOT / "tests" / "canonical" / "lmtp_crossfit",
        arguments.samples,
        arguments.truths,
        arguments.output,
        cores=arguments.jobs,
    )


if __name__ == "__main__":
    main()
