"""The sizing pilot of the ``learned-rule-ep`` study (P1 plan 5.5): SE ratios for the R rule.

A pilot, not a result.  Each cell fits the declared estimator of one calibration cell (or the
primary phase, or the fold-locality control) at its declared size and records the estimate,
the reported standard error, the harness truth and the CPU seconds.  :func:`summarize` writes
``learned_rule_ep_pilot.csv``: per cell, the replicates, the SE ratio (the mean reported SE over
the SD of the error ``estimate - truth``), the coverage of the truth, the oracle-SE ratio, the
mean CPU seconds of a fit and the share of folds whose cross-validated sieve size is each size.
``tests/unit/test_learned_rule_ep_design.py`` reads the CSV and applies the plan's sizing rule
and its assurance.

The truth is rule L4's value of the fit's fold rules, but the pilot does not refit them: it
records the ``FoldRule`` each fold fitted (:func:`captured_rules`) and integrates those, and the
harness still requires each recorded rule on every validation row.  The declared run refits.

Every draw comes from ``stream_seed(PILOT_RECORD, "se_ratio_pilot", cell, r)`` and its fold seed
from ``stream_seed(PILOT_RECORD, "se_ratio_pilot_folds", cell, r)``, where :data:`PILOT_RECORD`
is the study's record at the pilot seed :data:`PILOT_SEED`, labels no declared stream reads.
The design test checks the pilot's seeds against every declared learned-rule seed::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.learned_rule_ep_pilot --raw <scratch dir> --jobs 4 [--summarize]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.interventions import _rule_learners
from tests.studies import _learned_rule_ep_law as law
from tests.studies import _learned_rule_law as base
from tests.studies import learned_rule_ep as study
from tests.studies.evidence.seeds import stream_seed

#: The seed the pilot ran under, before the record's own seed was chosen to avoid it.
PILOT_SEED = 20261131
PILOT_RECORD = replace(study.STUDY, seed=PILOT_SEED, resampling_seed=PILOT_SEED + 1)

STREAM = "se_ratio_pilot"
FOLD_STREAM = "se_ratio_pilot_folds"

#: ``cell -> (kind, law, n, replicates)``.  The plan's pilot is 2,000 draws of the
#: ``non_exceptional`` calibration cell; the other gated cells get 600 each so that every
#: calibration cell's assurance reads a pilot of its own, and the primary and control cells a
#: few hundred to check their margins.
CELLS: dict[str, tuple[str, str, int, int]] = {
    "non_exceptional__declared": ("declared", "non_exceptional", 4_000, 2_000),
    "misspecified_limit__declared": ("declared", "misspecified_limit", 4_000, 600),
    "optimal_known_g__declared": ("optimal", "non_exceptional", 4_000, 600),
    "non_exceptional__primary": ("declared", "non_exceptional", 2_000, 300),
    "non_exceptional__validation_rule_control": ("control", "non_exceptional", 2_000, 60),
}

PILOT_CSV = Path(__file__).with_name("learned_rule_ep_pilot.csv")

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@contextlib.contextmanager
def captured_rules() -> Iterator[list[Any]]:
    """Record each ``FoldRule`` the fit learns, and replay it in the harness's refit.

    ``rule_learner_treat`` fits the folds in order, so the ``v``-th recorded rule is fold
    ``v``'s.  The harness's training-row refit returns it instead of refitting; a validation-row
    refit (the control) runs as declared.
    """
    rules: list[Any] = []
    fit_fold_rule = _rule_learners.fit_fold_rule
    fold_rule = base.fold_rule

    def recording(*args: Any, **kwargs: Any) -> Any:
        fitted = fit_fold_rule(*args, **kwargs)
        rules.append(fitted)
        return fitted

    def replay(result: Any, learner: Any, fold: int, *, source: Any = "training") -> Any:
        if source == "training":
            return rules[fold].treat
        return fold_rule(result, learner, fold, source=source)

    _rule_learners.fit_fold_rule = recording
    base.fold_rule = replay
    try:
        yield rules
    finally:
        _rule_learners.fit_fold_rule = fit_fold_rule
        base.fold_rule = fold_rule


def one(task: tuple[str, int]) -> dict[str, Any]:
    cell, replicate = task
    kind, name, n, _ = CELLS[cell]
    frame = base.draw(name, n, stream_seed(PILOT_RECORD, STREAM, cell, replicate))
    fold_seed = stream_seed(PILOT_RECORD, FOLD_STREAM, cell, replicate)
    start = time.process_time()
    with captured_rules() as rules:
        if kind == "optimal":
            fitted = law.optimal_measured(frame, fold_seed)
            estimate = fitted.result.estimates[law.OPTIMAL_ESTIMAND]
            truth, oracle = fitted.truth, fitted.refit.oracle_se
            warnings = fitted.warnings
        elif kind == "control":
            result, value, warnings = law.control(frame, name, fold_seed)
            estimate = result.estimates[law.ESTIMAND]
            truth, oracle = value.truth, value.oracle_se
        else:
            result, value, warnings = base.measured(
                frame, base.outcome_learner(), name, fold_seed, rule=law.rule()
            )
            estimate = result.estimates[law.ESTIMAND]
            truth, oracle = value.truth, value.oracle_se
    if kind != "control" and len(rules) != base.N_FOLDS:
        raise RuntimeError(f"{cell} {replicate}: recorded {len(rules)} fold rules")
    sizes = [int(rule.ep.chosen) for rule in rules if rule.ep is not None]
    return {
        "cell": cell,
        "replicate": replicate,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
        "truth": float(truth),
        "oracle_se": float(oracle),
        "solver_warnings": int(warnings),
        "sizes": sizes,
        "seconds": time.process_time() - start,
    }


def summarize(raw: Path) -> pd.DataFrame:
    """One row per pilot cell, from the raw ``<cell>.jsonl`` records."""
    out = []
    for cell in CELLS:
        path = raw / f"{cell}.jsonl"
        if not path.exists():
            continue
        frame = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])
        error = frame["estimate"] - frame["truth"]
        low = frame["estimate"] - 1.959963984540054 * frame["std_error"]
        high = frame["estimate"] + 1.959963984540054 * frame["std_error"]
        sizes = np.concatenate(
            [np.asarray(s, dtype=int) for s in frame["sizes"]] or [np.empty(0, dtype=int)]
        )
        row = {
            "cell": cell,
            "n": CELLS[cell][2],
            "replicates": len(frame),
            "se_ratio": float(frame["std_error"].mean() / error.std(ddof=1)),
            "oracle_ratio": float(frame["oracle_se"].mean() / error.std(ddof=1)),
            "coverage": float(np.mean((low <= frame["truth"]) & (frame["truth"] <= high))),
            "standardized_bias": float(error.mean() / error.std(ddof=1)),
            "solver_warnings": int(frame["solver_warnings"].sum()),
            "seconds": float(frame["seconds"].mean()),
        }
        for size in range(1, 7):
            row[f"size_{size}"] = float(np.mean(sizes == size)) if sizes.size else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--summarize", action="store_true")
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    if arguments.summarize:
        summarize(arguments.raw).to_csv(PILOT_CSV, index=False, float_format="%.6g")
        return
    unset = [name for name in THREAD_VARIABLES if os.environ.get(name) != "1"]
    if unset:
        raise SystemExit(f"set {', '.join(unset)} to 1: the pilot times one thread per fit")
    tasks = []
    for cell, (_, _, _, replicates) in CELLS.items():
        path = arguments.raw / f"{cell}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["replicate"] for line in path.read_text().splitlines()}
        count = replicates if arguments.cap is None else min(arguments.cap, replicates)
        tasks += [(cell, r) for r in range(count) if r not in done]
    with Pool(arguments.jobs, maxtasksperchild=20) as pool:
        for record in pool.imap_unordered(one, tasks, chunksize=1):
            with (arguments.raw / f"{record['cell']}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(record["cell"], record["replicate"], round(record["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "CELLS",
    "FOLD_STREAM",
    "PILOT_CSV",
    "PILOT_RECORD",
    "PILOT_SEED",
    "STREAM",
    "captured_rules",
    "summarize",
]
