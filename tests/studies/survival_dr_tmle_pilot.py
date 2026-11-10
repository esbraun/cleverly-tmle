"""The sizing pilot of the ``survival-dr-tmle`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``survival_dr_tmle_pilot.csv.gz`` beside this module, with the fit's error class when it raised.
Run it as ``python -m tests.studies.survival_dr_tmle_pilot --jobs N``.

======================  ======================================================================
label                   what it sizes
======================  ======================================================================
``<cell>``              each primary cell of the record at its size, 1,000 draws: the primary
                        replicate count by the shared rule
``dr_<configuration>``  each double-robustness configuration at 16,000, 300 draws: the
                        positives' bias and the control's gap, stated before the run
``calibration_both``    the calibration cell's risk difference at 16,000, 8,000 draws
                        (:data:`~tests.studies.repeated_sizing.CALIBRATION_PILOT_DRAWS`)
``slow_<kind>_16k``     each slow reported cell at 16,000, 300 draws: whether the declared
                        offset moves the arm risks by at least two standard deviations
======================  ======================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import survival_dr_tmle as study
from tests.studies import survival_dr_tmle_law as law
from tests.studies import survival_dr_tmle_properties as properties
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("survival_dr_tmle_pilot.csv.gz")

#: ``label -> (kind, cell or configuration, n, draws)``; ``kind`` is ``"primary"`` for a record
#: cell and ``"configured"`` for a property or reported configuration.
PILOTS: dict[str, tuple[str, str, int, int]] = {
    **{
        cell: ("primary", cell, n, study.PILOT_REPLICATES)
        for cell, (_, n) in study.SCENARIOS.items()
    },
    **{
        f"dr_{configuration}": ("configured", configuration, properties.ROBUSTNESS_N, 300)
        for configuration, _ in properties.ROBUSTNESS_CELLS.values()
    },
    "calibration_both": ("primary", "both_16k", properties.CALIBRATION_N, 8_000),
    "slow_censoring_16k": ("configured", "slow_censoring", 16_000, 300),
    "slow_hazard_16k": ("configured", "slow_hazard", 16_000, 300),
}
#: The calibration label writes only the estimand its cell reads.
ROWS_OF: dict[str, tuple[str, ...]] = {"calibration_both": (properties.RD,)}


def _one(label: str, replicate: int) -> list[dict[str, Any]]:
    kind, name, n, _ = PILOTS[label]
    seed = stream_seed(study.STUDY, "pilot", label, replicate)
    if kind == "primary":
        configuration, _ = study.SCENARIOS[name]
        trial = configuration == "known"
    else:
        trial, _ = properties.configuration_settings(name, n)
    frame = law.sample(n, seed, trial=trial)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "primary":
                result = study.fit(frame, name)
            else:
                result = properties.configured_fit(frame, name, n)
            rows = []
            for estimand in ROWS_OF.get(label, law.ESTIMANDS):
                estimate = result[estimand]
                low, high = estimate.ci
                truth = law.TRUTH[estimand]
                rows.append(
                    {
                        "label": label,
                        "replicate": replicate,
                        "n": n,
                        "estimand": estimand,
                        "truth": truth,
                        "estimate": float(estimate.psi),
                        "std_error": float(estimate.std_error),
                        "covered": int(low <= truth <= high),
                        "refusal": "",
                    }
                )
            return rows
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{"label": label, "replicate": replicate, "n": n, "refusal": type(error).__name__}]


def run(n_jobs: int, labels: list[str] | None = None) -> pd.DataFrame:
    """Run ``labels`` (every label by default) and replace their rows in the CSV."""
    chosen = labels if labels else list(PILOTS)
    jobs = [(label, r) for label in chosen for r in range(PILOTS[label][3])]
    outcomes = map_parallel(_one, jobs, n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows])
    if OUTPUT.exists():
        kept = pd.read_csv(OUTPUT)
        frame = pd.concat([kept.loc[~kept["label"].isin(chosen)], frame], ignore_index=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.labels)
