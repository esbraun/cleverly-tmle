"""The sizing pilot of the ``eliminated-competing-events`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``eliminated_competing_events_pilot.csv`` beside this module, with the fit's refusal when it
had one.  Run it as ``python -m tests.studies.eliminated_competing_events_pilot --jobs N``.

=====================  ======================================================================
label                  what it sizes
=====================  ======================================================================
``hazard_16k``         the hazard primary scenario and the ``hazard_a1_t3`` calibration cell,
                       n = 16,000, 1,000 draws
``ltmle_16k``          the one-fold time-varying primary scenario and the ``ltmle_always_t2``
                       calibration cell, n = 16,000, 1,000 draws
``crossfit_16k``       the cross-fitted primary scenario, n = 16,000, 1,000 draws
``dr_*_16k``           the two positive robustness cells, n = 16,000, 300 draws each
``*_4k``               the first declaration's time-varying scenarios at n = 4,000, 400 draws:
                       coverage 0.915 to 0.93 and SE ratios 0.91 to 0.93 at ``t = 2`` (one
                       fold), and a standardized bias up to 0.23 with SE ratios up to 1.21
                       (two folds), which moved every scenario to 16,000
=====================  ======================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import eliminated_competing_events as study
from tests.studies import eliminated_competing_events_properties as properties
from tests.studies import eliminated_competing_law as law
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("eliminated_competing_events_pilot.csv")

#: ``label -> (scenario, n, draws, robustness cell or "")``.  The 4,000 labels measured the
#: first declaration's size and are kept as the record of why the sizes moved to 16,000.
PILOTS: dict[str, tuple[str, int, int, str]] = {
    "hazard_16k": (study.HAZARD_HB, 16_000, 1_000, ""),
    "crossfit_4k": (study.CROSSFIT_TV, 4_000, 400, ""),
    "ltmle_4k": (study.LTMLE_TV, 4_000, 400, ""),
    "ltmle_16k": (study.LTMLE_TV, 16_000, 1_000, ""),
    "crossfit_16k": (study.CROSSFIT_TV, 16_000, 1_000, ""),
    "dr_tv_rule_16k": (study.LTMLE_TV, 16_000, 300, "tv_rule_t2__mechanism_correct"),
    "dr_hb_a1_16k": (study.HAZARD_HB, 16_000, 300, "hb_a1_t2__mechanism_correct"),
}


def _one(job: tuple[str, int]) -> list[dict[str, Any]]:
    label, replicate = job
    scenario, n, _, cell = PILOTS[label]
    seed = stream_seed(study.STUDY, "pilot", label, replicate)
    key, _, _ = study.SCENARIOS[scenario]
    frame = law.tv_sample(n, seed) if key == "time_varying" else law.hb_sample(n, seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = properties.robustness_fit(frame, cell) if cell else study.fit(frame, scenario)
            rows = []
            for name in study.ESTIMANDS[scenario]:
                estimate = result[name]
                low, high = estimate.ci
                truth = study.TRUTH[scenario][name]
                rows.append(
                    {
                        "label": label,
                        "replicate": replicate,
                        "n": n,
                        "estimand": name,
                        "truth": truth,
                        "estimate": float(estimate.psi),
                        "std_error": float(estimate.std_error),
                        "covered": int(low <= truth <= high),
                        "refusal": "",
                    }
                )
            return rows
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        return [{"label": label, "replicate": replicate, "n": n, "refusal": type(error).__name__}]


def run(n_jobs: int, labels: list[str] | None = None) -> pd.DataFrame:
    """Run ``labels`` (every label by default) and replace their rows in the CSV."""
    chosen = labels if labels else list(PILOTS)
    jobs = [(label, r) for label in chosen for r in range(PILOTS[label][2])]
    outcomes = map_parallel(_one, [(job,) for job in jobs], n_jobs=n_jobs)
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
