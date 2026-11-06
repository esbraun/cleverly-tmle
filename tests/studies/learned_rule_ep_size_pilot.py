"""A sizing pilot of the EP rule learner at larger n: does the rule-learning excess shrink?

A pilot, not a result, and not part of the declared ``learned-rule-ep`` study.  The declared
pilot (:mod:`tests.studies.learned_rule_ep_pilot`) read an SE ratio of 0.970 at n = 4,000, below
the plan's sizing bound.  This pilot fits the declared estimator of the ``non_exceptional``
calibration cell at n = 8,000 and 16,000 to size a redeclaration at a larger n.

Each replicate also fits the paired *limit-rule* fit: the same estimator, sample and outer folds
with every row's rule fixed at the EP limit rule ``1{theta_1(W) > 0}``
(:func:`tests.studies._learned_rule_ep_law.limit_rule`), whose truth is fixed.  Its error is
highly correlated with the learned fit's, so the paired variance ratio
``Var(error) / Var(limit error)`` resolves the excess variance that rule learning adds far more
tightly than the SE ratio alone.

Every draw comes from ``stream_seed(PILOT_RECORD, "se_ratio_size_pilot_1", cell, r)`` and its fold
seed from ``stream_seed(PILOT_RECORD, "se_ratio_size_pilot_1_folds", cell, r)``
(the unsuffixed names met two declared sample seeds of another study).  The run resumes: a
restart with the same ``--raw`` fits only the replicates no complete record holds::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.learned_rule_ep_size_pilot --raw <scratch dir> --jobs 4 [--summarize]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.interventions.base import RegimeSet
from tests.studies import _learned_rule_ep_law as law
from tests.studies import _learned_rule_law as base
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_ep_pilot import PILOT_RECORD, THREAD_VARIABLES, captured_rules

STREAM = "se_ratio_size_pilot_1"
FOLD_STREAM = "se_ratio_size_pilot_1_folds"

#: ``cell -> (kind, law, n, replicates)``.  The two ``non_exceptional`` cells resolve the excess
#: across n; the other two gated cells at n = 16,000 give each one a pilot of its own.
CELLS: dict[str, tuple[str, str, int, int]] = {
    "non_exceptional__n8000": ("declared", "non_exceptional", 8_000, 600),
    "non_exceptional__n16000": ("declared", "non_exceptional", 16_000, 1_400),
    "misspecified_limit__n16000": ("declared", "misspecified_limit", 16_000, 1_400),
    "optimal_known_g__n16000": ("optimal", "non_exceptional", 16_000, 1_400),
}

PILOT_CSV = Path(__file__).with_name("learned_rule_ep_size_pilot.csv")


def limit_treat(covariates: np.ndarray, name: str) -> np.ndarray:
    """The EP limit rule of law ``name`` at covariate rows ``(W1, W2)``."""
    w1, w2 = covariates[:, 0], covariates[:, 1]
    out = np.zeros(len(w1), dtype=bool)
    for level, score in law.limit_scores(name).items():
        rows = w2 == level
        out[rows] = np.interp(w1[rows], base.GRID, score) > 0.0
    return out


class LimitRule(TMLE):
    """Study-only: every row's rule is the EP limit rule; nuisances and targeting unchanged.

    The law is read from :attr:`LAW`, which :func:`limit_estimator` sets on a subclass.
    """

    LAW = "non_exceptional"

    def _regimes(self, data: Any, estimates: Any, *, seed: int | None = None) -> RegimeSet | None:
        assert self.learned_rule is not None
        treat = limit_treat(np.asarray(data.covariates, dtype=float), self.LAW)
        values = np.zeros((data.n, 2, 1), dtype=float)
        values[:, 1, 0] = treat
        values[:, 0, 0] = ~treat
        return RegimeSet((self.learned_rule.name,), values, 0.0)


def limit_estimator(name: str, fold_seed: int, *, optimal: bool) -> TMLE:
    """The cell's estimator with its rule fixed at the EP limit rule of law ``name``."""
    kind = type(f"LimitRule_{name}", (LimitRule,), {"LAW": name})
    template = (
        law.optimal_estimator(fold_seed)
        if optimal
        else base.estimator(base.outcome_learner(), fold_seed, rule=law.rule())
    )
    extra = {"estimands": [law.optimal_law.TARGET]} if optimal else {}
    return kind(
        learned_rule=template.learned_rule,
        n_folds=base.N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=base.outcome_learner(),
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
        **extra,
    )


def one(task: tuple[str, int]) -> dict[str, Any]:
    cell, replicate = task
    kind, name, n, _ = CELLS[cell]
    frame = base.draw(name, n, stream_seed(PILOT_RECORD, STREAM, cell, replicate))
    fold_seed = stream_seed(PILOT_RECORD, FOLD_STREAM, cell, replicate)
    start = time.process_time()
    with captured_rules() as rules:
        if kind == "optimal":
            fitted = law.optimal_measured(frame, fold_seed)
            result, warnings = fitted.result, fitted.warnings
            estimate = result.estimates[law.OPTIMAL_ESTIMAND]
            truth, oracle = fitted.truth, fitted.refit.oracle_se
        else:
            result, value, warnings = base.measured(
                frame, base.outcome_learner(), name, fold_seed, rule=law.rule()
            )
            estimate = result.estimates[law.ESTIMAND]
            truth, oracle = value.truth, value.oracle_se
    seconds = time.process_time() - start
    if len(rules) != base.N_FOLDS:
        raise RuntimeError(f"{cell} {replicate}: recorded {len(rules)} fold rules")
    optimal = kind == "optimal"
    fit_arguments: dict[str, Any] = {"covariates": list(base.COVARIATES)}
    if optimal:
        fit_arguments["treatment_probabilities"] = law.optimal_law.known_probabilities(frame)
    fixed = (
        limit_estimator(name, fold_seed, optimal=optimal)
        .fit(frame, outcome="Y", treatment="A", **fit_arguments)
        .single()
    )
    if not law._same_folds(result, fixed):
        raise RuntimeError("the limit-rule fit drew other outer folds")
    limit = fixed.estimates[law.OPTIMAL_ESTIMAND if optimal else law.ESTIMAND]
    return {
        "cell": cell,
        "replicate": replicate,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
        "truth": float(truth),
        "oracle_se": float(oracle),
        "limit_estimate": float(limit.psi),
        "limit_std_error": float(limit.std_error),
        "limit_truth": float(law.limit_value(name)[0]),
        "solver_warnings": int(warnings),
        "sizes": [int(rule.ep.chosen) for rule in rules if rule.ep is not None],
        "seconds": seconds,
        "limit_seconds": time.process_time() - start - seconds,
    }


def records(path: Path) -> list[dict[str, Any]]:
    """The complete records of one cell's file; a line torn by a killed run is dropped."""
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


BOOTSTRAP = 2_000


def _paired_bootstrap(
    frame: pd.DataFrame, error: np.ndarray, limit_error: np.ndarray, cell: str
) -> np.ndarray:
    """Bootstrap draws of the paired variance ratio and the paired SE ratio.

    The paired SE ratio is the SE ratio read against the limit-rule fit: the mean reported SE
    over the limit fit's, over the square root of the paired variance ratio.  It equals the SE
    ratio when the limit fit's own SE ratio is 1.
    """
    rng = np.random.default_rng(stream_seed(PILOT_RECORD, "size_pilot_bootstrap", cell))
    se = frame["std_error"].to_numpy()
    limit_se = frame["limit_std_error"].to_numpy()
    out = np.empty((BOOTSTRAP, 2))
    for b in range(BOOTSTRAP):
        rows = rng.integers(0, len(error), len(error))
        ratio = np.var(error[rows], ddof=1) / np.var(limit_error[rows], ddof=1)
        out[b] = ratio, se[rows].mean() / limit_se[rows].mean() / np.sqrt(ratio)
    return out


def summarize(raw: Path) -> pd.DataFrame:
    """One row per cell: the SE, oracle and paired ratios, coverage, bias and cost."""
    z = 1.959963984540054
    out = []
    for cell, (_, _, n, _) in CELLS.items():
        path = raw / f"{cell}.jsonl"
        if not path.exists():
            continue
        frame = pd.DataFrame(records(path)).drop_duplicates("replicate")
        error = frame["estimate"] - frame["truth"]
        limit_error = frame["limit_estimate"] - frame["limit_truth"]
        sd = error.std(ddof=1)
        covered = (frame["estimate"] - z * frame["std_error"] <= frame["truth"]) & (
            frame["truth"] <= frame["estimate"] + z * frame["std_error"]
        )
        sizes = np.concatenate([np.asarray(s, dtype=int) for s in frame["sizes"]])
        paired = _paired_bootstrap(frame, error.to_numpy(), limit_error.to_numpy(), cell)
        row = {
            "cell": cell,
            "n": n,
            "replicates": len(frame),
            "se_ratio": float(frame["std_error"].mean() / sd),
            "oracle_ratio": float(frame["oracle_se"].mean() / sd),
            "coverage": float(covered.mean()),
            "standardized_bias": float(error.mean() / sd),
            "limit_se_ratio": float(frame["limit_std_error"].mean() / limit_error.std(ddof=1)),
            "variance_ratio": float(error.var(ddof=1) / limit_error.var(ddof=1)),
            "variance_ratio_sd": float(np.std(paired[:, 0])),
            "paired_se_ratio": float(
                frame["std_error"].mean()
                / frame["limit_std_error"].mean()
                * np.sqrt(limit_error.var(ddof=1) / error.var(ddof=1))
            ),
            "paired_se_ratio_sd": float(np.std(paired[:, 1])),
            "error_correlation": float(np.corrcoef(error, limit_error)[0, 1]),
            "se_over_limit_se": float(frame["std_error"].mean() / frame["limit_std_error"].mean()),
            "solver_warnings": int(frame["solver_warnings"].sum()),
            "seconds": float(frame["seconds"].mean()),
            "limit_seconds": float(frame["limit_seconds"].mean()),
        }
        for size in range(1, 7):
            row[f"size_{size}"] = float(np.mean(sizes == size))
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
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
            done = {record["replicate"] for record in records(path)}
        tasks += [(cell, r) for r in range(replicates) if r not in done]
    # Interleave the cells so a stopped run leaves both partly done.
    tasks.sort(key=lambda task: (task[1] / CELLS[task[0]][3], task[0]))
    with Pool(arguments.jobs, maxtasksperchild=20) as pool:
        for record in pool.imap_unordered(one, tasks, chunksize=1):
            with (arguments.raw / f"{record['cell']}.jsonl").open("a") as handle:
                handle.write("\n" + json.dumps(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(record["cell"], record["replicate"], round(record["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
