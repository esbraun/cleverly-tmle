"""The sizing pilot of the ``separable-effects`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``separable_effects_pilot.csv.gz`` beside this module, with the fit's refusal when it had one.  Every row also carries the targeting's convergence
(``converged``, ``failure``, ``relative_score``), which the study counts as a failure when it
is 0, so the pilot measures the failure rate as well as the spread.
Run it as ``python -m tests.studies.separable_effects_pilot --jobs N``.

=====================  ======================================================================
label                  what it sizes
=====================  ======================================================================
``in_sample_16k``      the in-sample primary scenario, n = 16,000, 1,000 draws
``cross_fitted_16k``   the cross-fitted primary scenario, n = 16,000, 1,000 draws
``dr_*_16k``           the three positive robustness cells, n = 16,000, 300 draws each
``calibration_*_16k``  the two calibration cells' estimands alone, n = 16,000, 8,000 draws
                       each (:data:`~tests.studies.repeated_sizing.CALIBRATION_PILOT_DRAWS`)
=====================  ======================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import separable_effects as study
from tests.studies import separable_effects_law as law
from tests.studies import separable_effects_properties as properties
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("separable_effects_pilot.csv.gz")

#: ``label -> (scenario, n, draws, robustness cell or "")``.
PILOTS: dict[str, tuple[str, int, int, str]] = {
    "in_sample_16k": (study.IN_SAMPLE, 16_000, 1_000, ""),
    "cross_fitted_16k": (study.CROSS_FITTED, 16_000, 1_000, ""),
    "dr_event_16k": (study.IN_SAMPLE, 16_000, 300, "sep10_relapse_t3__event_hazard_wrong"),
    "dr_competing_16k": (
        study.IN_SAMPLE,
        16_000,
        300,
        "sep10_relapse_t3__competing_hazard_wrong",
    ),
    "dr_mechanism_16k": (study.CROSS_FITTED, 16_000, 300, "sep10_relapse_t3__mechanism_wrong"),
    "calibration_in_sample_16k": (study.IN_SAMPLE, 16_000, 8_000, ""),
    "calibration_cross_fitted_16k": (study.CROSS_FITTED, 16_000, 8_000, ""),
}
#: The cross-fitted calibration and robustness labels write only the estimand their cell reads.
#: ``calibration_in_sample_16k`` writes every estimand, which costs nothing more per fit.
ROWS_OF: dict[str, tuple[str, ...]] = {
    "calibration_cross_fitted_16k": (
        properties.CALIBRATION_CELLS["cross_fitted_sep01_death_t3"][1],
    ),
    **{
        label: (properties.ROBUSTNESS_ESTIMAND,)
        for label, (_, _, _, cell) in PILOTS.items()
        if cell
    },
}


def _one(job: tuple[str, int]) -> list[dict[str, Any]]:
    label, replicate = job
    scenario, n, _, cell = PILOTS[label]
    seed = stream_seed(study.STUDY, "pilot", label, replicate)
    frame = law.sample(n, seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = properties.robustness_fit(frame, cell) if cell else study.fit(frame, scenario)
            converged, failure, score = study.convergence(result)
            rows = []
            for name in ROWS_OF.get(label, study.ESTIMANDS[scenario]):
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
                        "converged": int(converged),
                        "failure": failure,
                        "relative_score": score,
                        "refusal": "",
                    }
                )
            return rows
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        return [{"label": label, "replicate": replicate, "n": n, "refusal": type(error).__name__}]


def run(n_jobs: int, labels: list[str] | None = None, draws: int | None = None) -> pd.DataFrame:
    """Run ``labels`` (every label by default) and replace their rows in the CSV.

    ``draws`` caps each label for a smoke run, which writes nothing.
    """
    chosen = labels if labels else list(PILOTS)
    count = {label: PILOTS[label][2] if draws is None else draws for label in chosen}
    jobs = [(label, r) for label in chosen for r in range(count[label])]
    outcomes = map_parallel(_one, [(job,) for job in jobs], n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows])
    if draws is not None:
        return frame
    if OUTPUT.exists():
        kept = pd.read_csv(OUTPUT)
        frame = pd.concat([kept.loc[~kept["label"].isin(chosen)], frame], ignore_index=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--draws", type=int, default=None)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.labels, arguments.draws)
