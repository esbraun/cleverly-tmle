"""The sizing pilot of the ``survival-ctmle`` study, on its own seed streams.

It draws no declared sample: draw ``r`` of a label takes ``stream_seed(STUDY, "pilot", label,
r)``, a stream no declared cell reads.  One label per scenario of
:data:`~tests.studies.survival_ctmle.SCENARIOS`, each :data:`PILOT_REPLICATES` draws at the
scenario's declared size, and every draw fits all three implementations.  It writes one row per
fit and estimand to ``survival_ctmle_pilot.csv.gz`` beside this module.  A row carries the
estimate, the plug-in standard error and coverage, the targeting's convergence (``converged``,
``failure``), the seconds of the fit, the ``SurvivalCTMLE`` selection readings of
:data:`~tests.studies.survival_ctmle.EXTRA_COLUMNS` and, when the fit raised, its error class in
``refusal``.

Run it as ``python -m tests.studies.survival_ctmle_pilot --jobs N``.  It checkpoints after each
label, writing the CSV atomically, and skips a label whose rows are already complete, so a
restarted runner resumes where it stopped.  ``--labels`` runs some labels; ``--draws`` caps each
label for a smoke run, which writes nothing.

:func:`primary_replicates` reads the finished pilot by the shared rule of
:func:`~tests.studies.repeated_sizing.primary_replicates`; the design test compares it with
:data:`~tests.studies.survival_ctmle.PRIMARY_REPLICATES` once the CSV exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import repeated_sizing as sizing
from tests.studies import survival_ctmle as study
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("survival_ctmle_pilot.csv.gz")
PILOT_REPLICATES = study.PILOT_REPLICATES

#: ``label -> (scenario, n, draws)``: one label per scenario, at its declared size.
PILOTS: dict[str, tuple[str, int, int]] = {
    name: (name, study.SCENARIOS[name][3], PILOT_REPLICATES) for name in study.SCENARIOS
}

COLUMNS = (
    "label",
    "replicate",
    "implementation",
    "n",
    "estimand",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "converged",
    "failure",
    "refusal",
    *study.EXTRA_COLUMNS,
)


def convergence(result: Any) -> tuple[bool, str]:
    """``(converged, failure)`` over the targeting records of a hazard fit."""
    records = [fit.hazard for fit in result.fits.values() if getattr(fit, "hazard", None)]
    failures = [str(record.failure) for record in records if not record.converged]
    return not failures, ",".join(failures)


def _one(label: str, replicate: int) -> list[dict[str, Any]]:
    scenario, n, _ = PILOTS[label]
    odds, source, _, _, settings = study.SCENARIOS[scenario]
    seed = stream_seed(study.STUDY, "pilot", label, replicate)
    frame = study.sample(
        n, seed, odds=odds, source=source, binarized=bool(settings.get("binarized"))
    )
    rows: list[dict[str, Any]] = []
    for implementation in study.IMPLEMENTATIONS:
        base = {"label": label, "replicate": replicate, "implementation": implementation, "n": n}
        start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = study.fit(frame, scenario, implementation)
            seconds = time.perf_counter() - start
            converged, failure = convergence(result)
            readings = dict.fromkeys(study.EXTRA_COLUMNS[1:], np.nan)
            if implementation == study.CTMLE_ARM:
                readings.update(study.selection_readings(result))
            for name in study.ESTIMAND_NAMES:
                estimate = result[name]
                std_error, low, high = reported_inference(estimate)
                truth = study.TRUTH[name]
                rows.append(
                    {
                        **base,
                        "estimand": name,
                        "truth": truth,
                        "estimate": float(estimate.psi),
                        "std_error": float(std_error),
                        "covered": int(low <= truth <= high),
                        "converged": int(converged),
                        "failure": failure,
                        "refusal": "",
                        "seconds": seconds,
                        **readings,
                    }
                )
        except Exception as error:  # a failed fit is a pilot finding, recorded with its class
            rows.append(
                {**base, "refusal": type(error).__name__, "seconds": time.perf_counter() - start}
            )
    return rows


def _done(kept: pd.DataFrame, label: str) -> bool:
    """Whether ``kept`` holds every draw of ``label`` for every implementation."""
    if kept.empty:
        return False
    rows = kept.loc[kept["label"] == label]
    pairs = set(zip(rows["replicate"], rows["implementation"], strict=True))
    wanted = {(r, i) for r in range(PILOTS[label][2]) for i in study.IMPLEMENTATIONS}
    return wanted <= pairs


def _write(frame: pd.DataFrame) -> None:
    """Write the CSV atomically, so a restart never reads half a file."""
    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    frame.reindex(columns=list(COLUMNS)).to_csv(partial, index=False, compression="gzip")
    os.replace(partial, OUTPUT)


def run(n_jobs: int, labels: list[str] | None = None, draws: int | None = None) -> pd.DataFrame:
    """Run ``labels`` (every label by default), checkpointing the CSV after each label.

    A label already complete in the CSV is skipped.  ``draws`` caps each label for a smoke
    run, which reads and writes nothing.
    """
    chosen = labels if labels else list(PILOTS)
    unknown = sorted(set(chosen) - set(PILOTS))
    if unknown:
        raise ValueError(f"unknown pilot labels {unknown}; the labels are {sorted(PILOTS)}")
    if draws is not None:
        jobs = [(label, r) for label in chosen for r in range(draws)]
        outcomes = map_parallel(_one, jobs, n_jobs=n_jobs)
        return pd.DataFrame([row for rows in outcomes for row in rows]).reindex(
            columns=list(COLUMNS)
        )
    kept = pd.read_csv(OUTPUT, low_memory=False) if OUTPUT.exists() else pd.DataFrame()
    for label in chosen:
        if _done(kept, label):
            print(f"{label}: complete, skipped", flush=True)
            continue
        start = time.perf_counter()
        jobs = [(label, r) for r in range(PILOTS[label][2])]
        outcomes = map_parallel(_one, jobs, n_jobs=n_jobs)
        fresh = pd.DataFrame([row for rows in outcomes for row in rows])
        others = kept.loc[kept["label"] != label] if not kept.empty else kept
        kept = pd.concat([others, fresh], ignore_index=True)
        _write(kept)
        print(f"{label}: {len(jobs)} draws in {time.perf_counter() - start:.0f} s", flush=True)
    return kept


def primary_replicates(pilot: pd.DataFrame) -> int:
    """The primary replicate count the shared rule reads from a finished pilot.

    :func:`~tests.studies.repeated_sizing.primary_replicates` with one implementation, the
    subject, over every gated cell and estimand of
    :data:`~tests.studies.survival_ctmle.GATED_ESTIMANDS`, both shares at
    :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`.  Every gated cell is point-only
    (F18), so the rule reads a coverage floor of zero and the bias clauses size it; one bias
    group per gated scenario.
    """
    subject = pilot.loc[pilot["implementation"] == study.CTMLE_ARM]
    covered: dict[str, int] = {}
    groups = []
    draws = set()
    for scenario in study.GATED_SCENARIOS:
        names = list(study.GATED_ESTIMANDS[scenario])
        rows = subject.loc[subject["label"] == scenario]
        hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
        covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
        error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
            index="replicate", columns="estimand", values="error"
        )[names]
        standardized = error / error.std(ddof=1)
        draws.add(len(hits))
        groups.append(
            sizing.PilotGroup(
                standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
            )
        )
    (pilot_draws,) = draws
    share = sizing.PRIMARY_CLAUSE_SHARE
    # The cells are point-only, so the rule reads no coverage floor and no SE screen: a floor
    # of zero passes the coverage clauses at every budget, the rule the declared count was
    # sized under.
    return sizing.primary_replicates(
        dataclasses.replace(study.STUDY.margins, coverage_floor=0.0, se_ratio_sanity=(0.0, 1e6)),
        covered,
        pilot_draws,
        implementations=1,
        coverage_pass=share,
        bias_groups=groups,
        bias_pass=share,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--draws", type=int, default=None)
    arguments = parser.parse_args()
    frame = run(arguments.jobs, arguments.labels, arguments.draws)
    if arguments.draws is not None:
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(frame)
