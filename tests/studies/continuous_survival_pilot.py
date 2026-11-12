"""The sizing pilot of the ``continuous-time-survival`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``continuous_survival_pilot.csv.gz`` beside this module, with the fit's error class when it
raised.  Run it as ``python -m tests.studies.continuous_survival_pilot --jobs N``.

======================  ======================================================================
label                   what it sizes
======================  ======================================================================
``<cell>``              each primary cell of the record at its size, 1,000 draws (300 at
                        4b232921, extended on the same streams): the primary replicate count
                        by the shared rule
``dr_<configuration>``  each double-robustness configuration at 4,000, 300 draws: the
                        positives' bias and the control's gap, stated before the run
``calibration_both``    the calibration cell's risk difference at 4,000, 2,000 draws (fewer
                        than the shared 8,000, at about a minute per fit)
``positivity_4k``       the reported positivity cell, 100 draws
``dr_<config>_1k``      each double-robustness configuration at 1,000, 300 draws
``calibration_1k``      the calibration cell's risk difference at 1,000, the shared 8,000 draws
======================  ======================================================================

It also reads the design's pilot gates: the initial plug-in of the treated arm's cause-1 risk in
``dr_mechanism_correct`` sits at least 2 SE from the truth (the wrong hazard model is wrong
where it matters), and the share of ``(unit, arm)`` pairs whose :math:`\\pi S_c(t_K^-)` is below
the floor on the dependent-censoring law; and the walk's steps, its convergence and the time
per fit.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import continuous_survival as study
from tests.studies import continuous_survival_law as law
from tests.studies import continuous_survival_properties as properties
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("continuous_survival_pilot.csv.gz")


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
    "calibration_both": ("primary", "both_cf_4k", properties.CALIBRATION_N, 2_000),
    "positivity_4k": ("configured", "positivity", 4_000, 100),
    # The n = 1,000 property pilots, for a record gated at that size.
    **{
        f"dr_{configuration}_1k": ("configured", configuration, 1_000, 300)
        for configuration, _ in properties.ROBUSTNESS_CELLS.values()
    },
    "calibration_1k": ("primary", "both_cf_1k", 1_000, 8_000),
}
#: The calibration labels write only the estimand their cell reads.
ROWS_OF: dict[str, tuple[str, ...]] = {
    "calibration_both": (properties.RD,),
    "calibration_1k": (properties.RD,),
}


def _one(label: str, replicate: int) -> list[dict[str, Any]]:
    kind, name, n, _ = PILOTS[label]
    seed = stream_seed(study.STUDY, "pilot", label, replicate)
    law_name = "base" if kind == "primary" else properties.CONFIGURATIONS[name][0]
    frame = study.sample(n, seed, law_name=law_name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            if kind == "primary":
                result = study.fit(frame, name)
            else:
                result = properties.configured_fit(frame, name)
            seconds = time.perf_counter() - start
        extra = {"seconds": seconds, **study.readings(result)}
        initial = result.initial_estimates
        rows = []
        for estimand in ROWS_OF.get(label, tuple(law.TRUTH)):
            truth = law.TRUTH[estimand]
            estimate = result[estimand]
            low, high = estimate.ci
            rows.append(
                {
                    "label": label,
                    "replicate": replicate,
                    "n": n,
                    "estimand": estimand,
                    "truth": truth,
                    "estimate": float(estimate.psi),
                    "std_error": float(estimate.std_error),
                    "initial_estimate": float(initial[estimand]),
                    "covered": int(low <= truth <= high),
                    "refusal": "",
                    **extra,
                }
            )
        return rows
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{"label": label, "replicate": replicate, "n": n, "refusal": type(error).__name__}]


def run(
    n_jobs: int, labels: list[str] | None = None, draws: int | None = None, first: int = 0
) -> pd.DataFrame:
    """Run draws ``first`` onwards of ``labels`` (every cell by default) and replace their rows.

    Rows of the chosen labels below ``first`` are kept, so a label extends on its own streams.
    """
    chosen = labels if labels else list(PILOTS)
    jobs = [
        (label, r)
        for label in chosen
        for r in range(first, PILOTS[label][3] if draws is None else min(draws, PILOTS[label][3]))
    ]
    outcomes = map_parallel(_one, jobs, n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows])
    if OUTPUT.exists():
        kept = pd.read_csv(OUTPUT, low_memory=False)
        keep = ~kept["label"].isin(chosen) | (kept["replicate"] < first)
        frame = pd.concat([kept.loc[keep], frame], ignore_index=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--first", type=int, default=0)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.labels, arguments.draws, arguments.first)
