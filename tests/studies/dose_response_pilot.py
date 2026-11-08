"""The sizing pilot of the ``dose-response`` study (X93 S1), on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``dose_response_pilot.csv.gz`` beside this module, with the fit's refusal when it had one.  Run
it as ``python -m tests.studies.dose_response_pilot --jobs N``.

====================  ==========================================================================
label                 what it sizes
====================  ==========================================================================
``binary_4k``         the ``binary`` primary scenario and the band cell (each row carries the
                      band's endpoints), n = 4,000, 1,000 draws
``strata_4k``         the ``strata`` primary scenario at n = 4,000, 1,000 draws: it read coverage
                      0.908 in one stratum, so the scenario moved to n = 8,000
``strata_8k``         the ``strata`` primary scenario, n = 8,000, 1,000 draws
``gaussian_4k``       the ``gaussian`` primary scenario, n = 4,000, 1,000 draws
``tail_1k``           the ``tail`` primary scenario, n = 1,000, 1,000 draws
``calibration_4k``    the calibration pair fitted alone, n = 4,000, 1,000 draws
``dr_positive_4k``    the positive robustness cell, n = 4,000, 300 draws
``dr_control_4k``     the robustness control, n = 4,000, 300 draws
``calibration_4k_b``  the calibration pair fitted alone, n = 4,000, 6,000 more draws, so the
                      shared calibration rule reads a pooled pilot of 8,000 fits of the pair
``dr_targeting_4k``   the outcome-wrong, declared-density robustness fit with its plug-in,
                      n = 4,000, 400 draws: it sizes the targeting gain
``agreement_4k``      20 draws, each fitted with the 15-pair declaration and with the calibration
                      pair alone; each row carries both estimates of the pair, which the design
                      test requires to agree
====================  ==========================================================================

Every row carries the pair's untargeted plug-in in ``initial_estimate``; the labels run before
that column existed leave it empty.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import dose_response as study
from tests.studies import dose_response_law as law
from tests.studies import dose_response_properties as properties
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("dose_response_pilot.csv.gz")

#: ``label -> (scenario, n, draws, kind)``; ``kind`` is ``"primary"``, ``"calibration"`` or a
#: robustness cell.
PILOTS: dict[str, tuple[str, int, int, str]] = {
    "binary_4k": (study.BINARY, 4_000, 1_000, "primary"),
    "strata_4k": (study.STRATA, 4_000, 1_000, "primary"),
    "strata_8k": (study.STRATA, 8_000, 1_000, "primary"),
    "gaussian_4k": (study.GAUSSIAN, 4_000, 1_000, "primary"),
    "tail_1k": (study.TAIL, 1_000, 1_000, "primary"),
    "calibration_4k": (study.BINARY, 4_000, 1_000, "calibration"),
    "dr_positive_4k": (study.BINARY, 4_000, 300, "h1_a3_5__outcome_correct_density_wrong"),
    "dr_control_4k": (study.BINARY, 4_000, 300, "h1_a3_5__both_wrong"),
    "calibration_4k_b": (study.BINARY, 4_000, 6_000, "calibration"),
    "dr_targeting_4k": (study.BINARY, 4_000, 400, "h1_a3_5__outcome_wrong_density_declared"),
    "agreement_4k": (study.BINARY, 4_000, 20, "agreement"),
}


def _fit(label: str, frame: pd.DataFrame) -> tuple[Any, dict[str, float], dict[str, float]]:
    scenario, _, _, kind = PILOTS[label]
    if kind == "primary":
        result, initial = study.fit_with_initials(frame, scenario)
        return result, study.TRUTH[scenario], initial
    if kind == "agreement":
        result, initial = study.fit_with_initials(frame, study.BINARY)
        return result, study.TRUTH[study.BINARY], initial
    if kind == "calibration":
        result, initial = study.fit_with_initials(
            frame, study.BINARY, curve=properties.single_pair(*properties.CALIBRATION_PAIR)
        )
        return result, properties.CALIBRATION_TRUTH_BY_NAME, initial
    result, plug_in = properties.robustness_fit(frame, kind)
    name = properties.ROBUSTNESS_NAME
    return result, {name: properties.ROBUSTNESS_TRUTH}, {name: plug_in}


def _one(job: tuple[str, int]) -> list[dict[str, Any]]:
    label, replicate = job
    scenario, n, _, _ = PILOTS[label]
    seed = stream_seed(study.STUDY, "pilot", label, replicate)
    frame = law.sample(n, seed, gaussian=study.SCENARIOS[scenario].gaussian)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, truth, initial = _fit(label, frame)
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        return [{"label": label, "replicate": replicate, "n": n, "refusal": type(error).__name__}]
    seconds = time.perf_counter() - start
    single: dict[str, float] = {}
    if PILOTS[label][3] == "agreement":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alone = properties.calibration_fit(frame)[properties.CALIBRATION_NAME]
        single = {
            "single_pair_estimate": float(alone.psi),
            "single_pair_std_error": float(alone.std_error),
        }
    rows = []
    for support in result.pair_support:
        name = support.name
        if single and name != properties.CALIBRATION_NAME:
            continue
        estimate = result[name]
        low, high = estimate.ci
        band_low, band_high = result.band.bands[name]
        reference = float(truth[name])
        rows.append(
            {
                "label": label,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "initial_estimate": float(initial[name]),
                "std_error": float(estimate.std_error),
                "covered": int(low <= reference <= high),
                "band_covered": int(band_low <= reference <= band_high),
                "critical": float(result.band.critical_value),
                "kish": float(support.kish),
                "below_floor": int(support.below_floor),
                "quadrature_shift": float(result.quadrature_shift),
                "seconds": seconds,
                "refusal": "",
                **single,
            }
        )
    return rows


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
