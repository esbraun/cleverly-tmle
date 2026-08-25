"""Run a retained pinned-``lmtp`` audit, and summarize what it found.

The audit is what a registered row rests on when it declines to publish a comparator.  It
therefore has to be reproducible in the same sense the row is: ``run`` fits the pinned R
implementation on panels this repository drew, and ``summarize`` turns those rows into the
committed table using the same verdict machinery every registered study uses.  Neither step
writes a registered artifact.  A hand-recorded table would say the same numbers and could not
be checked against anything.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
from tests.canonical.regenerate import Reference
from tests.parallel import available_cores
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.registry import ROOT, StudyRecord

#: Which study each subcommand audits, and the runner that fits the comparator for it.
STUDIES = {
    "end": ("tests.studies.canonical_ltmle_crossfit", "lmtp_ltmle/run_study.R"),
    "survival": (
        "tests.studies.canonical_ltmle_survival_crossfit",
        "lmtp_ltmle_survival/run_study.R",
    ),
}

#: The comparator implementation label the R runners write into their rows.
REFERENCE = "lmtp"

#: What the committed audit table publishes, in order.  Coverage and the SE ratio are the
#: comparator's own -- the gates it failed -- and ``rmse_ratio_upper`` is the paired bound,
#: which is a statement about the pair and is the second reason the row declined to publish.
AUDIT_COLUMNS = (
    "estimand",
    "replicates",
    "n",
    "coverage",
    "coverage_ci_lower",
    "coverage_ci_upper",
    "se_ratio",
    "se_ratio_ci_lower",
    "se_ratio_ci_upper",
    "rmse_ratio_upper",
)


def _record(study: str) -> StudyRecord:
    """The study's record with the comparator restored, which is what the audit asks about.

    The registered record declares no reference, so the paired machinery refuses it.  The
    audit's whole question is what that machinery *would* say, so it is asked here against a
    copy rather than by changing what the study publishes.
    """
    from importlib import import_module

    return dataclasses.replace(import_module(STUDIES[study][0]).STUDY, reference=REFERENCE)


def _run(arguments: argparse.Namespace) -> None:
    parents = {
        path.resolve().parent for path in (arguments.samples, arguments.truths, arguments.output)
    }
    if len(parents) != 1:
        raise SystemExit("samples, truths, and output must share one directory")
    reference = Reference(
        image="cleverly-lmtp-crossfit:1.5.4",
        runner=STUDIES[arguments.study][1],
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


def audit_table(rows: pd.DataFrame, record: StudyRecord, *, n_jobs: int) -> pd.DataFrame:
    """The comparator's own verdicts, plus the paired RMSE bound, one row per estimand."""
    summaries = summarize(rows)
    performance = independent_performance_tests(rows, record=record, n_jobs=n_jobs)
    paired = equivalence(rows, summaries, performance, record=record, n_jobs=n_jobs)

    comparator = performance.loc[performance["implementation"] == REFERENCE].set_index("estimand")
    counts = (
        rows.loc[rows["implementation"] == REFERENCE].groupby("estimand")["replicate"].nunique()
    )
    sizes = rows.groupby("estimand")["n"].first()
    table = paired.set_index("estimand")[["rmse_ratio_upper"]].join(
        comparator[
            [
                "coverage",
                "coverage_ci_lower",
                "coverage_ci_upper",
                "se_ratio",
                "se_ratio_ci_lower",
                "se_ratio_ci_upper",
            ]
        ]
    )
    table["replicates"] = counts
    table["n"] = sizes
    return (
        table.reset_index().loc[:, list(AUDIT_COLUMNS)].sort_values("estimand", ignore_index=True)
    )


def _summarize(arguments: argparse.Namespace) -> None:
    record = _record(arguments.study)
    rows = pd.concat(
        [pd.read_csv(arguments.python_rows), pd.read_csv(arguments.reference_rows)],
        ignore_index=True,
    )
    unexpected = set(rows["implementation"]) - {record.implementation, REFERENCE}
    if unexpected:
        raise SystemExit(f"unexpected implementations in the audit rows: {sorted(unexpected)}")
    table = audit_table(rows, record, n_jobs=arguments.jobs)
    write_csv(table, arguments.output)
    print(table.to_string(index=False))


def main(argv: Any = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fit the pinned comparator on drawn panels")
    run.add_argument("study", choices=tuple(STUDIES))
    run.add_argument("samples", type=Path)
    run.add_argument("truths", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--jobs", type=int, default=available_cores())
    run.set_defaults(handler=_run)

    table = sub.add_parser("summarize", help="build the committed audit table from those rows")
    table.add_argument("study", choices=tuple(STUDIES))
    table.add_argument("python_rows", type=Path)
    table.add_argument("reference_rows", type=Path)
    table.add_argument("output", type=Path)
    table.add_argument("--jobs", type=int, default=available_cores())
    table.set_defaults(handler=_summarize)

    arguments = parser.parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
