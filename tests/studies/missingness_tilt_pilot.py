"""The sizing pilot of the ``missingness-tilt-inference`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``missingness_tilt_pilot.csv.gz`` beside this module, with the fit's refusal when it had one.
Run it as ``python -m tests.studies.missingness_tilt_pilot --jobs N [--labels ...]``.

====================  ===========================================================================
label                 what it sizes or records
====================  ===========================================================================
``<scenario>``        each primary scenario at its declared n, :data:`PRIMARY_DRAWS` draws
``calibration``       the calibration fit (tilted ``ate`` at gamma 2 with the A1-era plug-in and
                      MAR SE), :data:`CALIBRATION_DRAWS` draws
``necessity``         the main-terms outcome fit with its plug-in, :data:`NECESSITY_DRAWS` draws
``necessity_limit``   the same fit at n = :data:`LIMIT_N`, :data:`LIMIT_DRAWS` draws: the large-n
                      limits of both necessity arms (:func:`necessity_limits`)
====================  ===========================================================================

The sizes are read by :func:`sizes`, and the design test compares them with the declared
counts.  Every label must be present: there is no smaller pilot to fall back on.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests.studies import missingness_tilt_inference as study
from tests.studies import missingness_tilt_law as law
from tests.studies import missingness_tilt_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("missingness_tilt_pilot.csv.gz")

PRIMARY_DRAWS = sizing.PILOT_DRAWS
CALIBRATION_DRAWS = sizing.CALIBRATION_PILOT_DRAWS
NECESSITY_DRAWS = 1_000
LIMIT_N = 3_000_000
LIMIT_DRAWS = 4

CALIBRATION = "calibration"
NECESSITY = "necessity"
LIMIT = "necessity_limit"

#: ``label -> (scenario or family, n, draws, kind)``; ``kind`` is ``"primary"``,
#: ``"calibration"`` or ``"necessity"``.
PILOTS: dict[str, tuple[str, int, int, str]] = {
    **{
        scenario: (scenario, study.STUDY.sizes(study.PRIMARY_N)[scenario], PRIMARY_DRAWS, "primary")
        for scenario in study.SCENARIOS
    },
    CALIBRATION: (CALIBRATION, properties.PROPERTY_N, CALIBRATION_DRAWS, CALIBRATION),
    NECESSITY: (NECESSITY, properties.PROPERTY_N, NECESSITY_DRAWS, NECESSITY),
    LIMIT: (NECESSITY, LIMIT_N, LIMIT_DRAWS, NECESSITY),
}

#: The calibration positive's name, as :func:`~tests.studies.repeated_sizing.calibration_size`
#: reads it (``label__arm``).
CALIBRATION_ARM = properties.POSITIVE_CELL


def _rows(label: str, frame: pd.DataFrame, replicate: int) -> list[dict[str, Any]]:
    scenario, n, _, kind = PILOTS[label]
    if kind == "primary":
        truth = study.TRUTH[scenario]
        rows = study.replicate_rows(frame, truth, scenario, replicate, n)
        return [{**row, "mar_std_error": float("nan")} for row in rows]
    family = properties.CALIBRATION if kind == CALIBRATION else properties.NECESSITY
    result = properties.property_fit(frame, family)
    estimate, std_error, plug_in = properties.tilted(result)
    half = properties.CRITICAL * std_error
    return [
        {
            "scenario": kind,
            "replicate": replicate,
            "n": n,
            "estimand": properties.NAME,
            "truth": properties.TRUTH,
            "estimate": estimate,
            "inference_estimate": estimate,
            "inference_scale": "identity",
            "std_error": std_error,
            "covered": int(abs(estimate - properties.TRUTH) <= half),
            "initial_estimate": plug_in,
            "mar_std_error": float(result[properties.TARGET].std_error),
        }
    ]


def _one(job: tuple[str, int]) -> list[dict[str, Any]]:
    label, replicate = job
    scenario, n, _, _ = PILOTS[label]
    weighted = scenario in study.SCENARIOS and study.SCENARIOS[scenario].weighted
    frame = law.sample(n, stream_seed(study.STUDY, "pilot", label, replicate), weighted=weighted)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = _rows(label, frame, replicate)
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        return [{"label": label, "replicate": replicate, "n": n, "refusal": type(error).__name__}]
    seconds = time.perf_counter() - start
    return [{"label": label, **row, "seconds": seconds, "refusal": ""} for row in rows]


def run(n_jobs: int, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Run ``labels`` (every label by default) and replace their rows in the CSV."""
    chosen = list(labels) if labels else list(PILOTS)
    jobs = [(label, r) for label in chosen for r in range(PILOTS[label][2])]
    outcomes = map_parallel(_one, [(job,) for job in jobs], n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows])
    if OUTPUT.exists():
        kept = pd.read_csv(OUTPUT, low_memory=False)
        frame = pd.concat([kept.loc[~kept["label"].isin(chosen)], frame], ignore_index=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


# ---------------------------------------------------------------------------- sizing


def _kept(pilot: pd.DataFrame, label: str) -> pd.DataFrame:
    """The label's rows without refusals; refuses a label the pilot does not hold."""
    rows = pilot.loc[pilot["label"] == label]
    if rows.empty:
        raise ValueError(f"the pilot holds no {label!r} rows")
    return rows.loc[rows["refusal"].isna() | (rows["refusal"] == "")]


def _inference_error(rows: pd.DataFrame) -> pd.Series:
    """Each row's error on its inference scale: the log scale for a ratio."""
    log = rows["inference_scale"] == "log"
    truth = np.where(log, np.log(rows["truth"].to_numpy(float)), rows["truth"].to_numpy(float))
    return rows["inference_estimate"] - truth


def primary_size(pilot: pd.DataFrame) -> int:
    """:func:`~tests.studies.repeated_sizing.primary_replicates` on every primary scenario.

    One implementation, one bias group per scenario from the standardized errors on the
    inference scale, both shares at :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`,
    as ``survival_ctmle_pilot`` reads its pilot, with this study's coverage floor.
    """
    covered: dict[str, int] = {}
    groups = []
    draws = set()
    for scenario in study.SCENARIOS:
        names = list(study.ESTIMANDS[scenario])
        rows = _kept(pilot, scenario)
        hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
        covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
        error = rows.assign(error=_inference_error(rows)).pivot(
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
    return sizing.primary_replicates(
        study.STUDY.margins,
        covered,
        pilot_draws,
        implementations=1,
        coverage_pass=share,
        bias_groups=groups,
        bias_pass=share,
    )


def _one_label(labels: Sequence[str]) -> np.ndarray:
    return np.ones((len(labels), len(labels)))


def calibration_reading(pilot: pd.DataFrame) -> dict[str, float]:
    """The calibration pilot's SE ratio, standardized bias, draws and A1-era control ratio."""
    rows = _kept(pilot, CALIBRATION)
    errors = rows["estimate"].to_numpy(float) - rows["truth"].to_numpy(float)
    spread = float(np.std(errors, ddof=1))
    plug = rows["initial_estimate"].to_numpy(float) - rows["truth"].to_numpy(float)
    return {
        "ratio": float(rows["std_error"].mean()) / spread,
        "bias": float(np.mean(errors)) / spread,
        "draws": float(len(rows)),
        "coverage": float(rows["covered"].mean()),
        "mar_ratio": float(rows["mar_std_error"].mean()) / float(np.std(plug, ddof=1)),
        "mar_coverage": float(
            np.mean(np.abs(plug) <= properties.CRITICAL * rows["mar_std_error"].to_numpy(float))
        ),
    }


def _ratio_upper(ratio: float, draws: int) -> float:
    """The one-sided 80% upper bound of an SE ratio read on ``draws`` normal estimates."""
    return ratio + float(norm.ppf(sizing.PILOT_LEVEL)) * ratio / np.sqrt(2.0 * (draws - 1))


def calibration_size(pilot: pd.DataFrame) -> int:
    """The calibration family's size.

    :func:`~tests.studies.repeated_sizing.calibration_size` on the positive (one cell, so its
    correlation is one) with its pilot bias, raised if needed to the smallest multiple of 400 at
    which each SE control leaves the band with probability at least
    ``1 - DISCRIMINATION_FAIL``: the shrunken control and the noise control (ratio ``1/sqrt(2)``)
    at the upper bound of the positive's pilot ratio, and the A1-era control at the upper bound
    of its own.
    """
    reading = calibration_reading(pilot)
    draws = int(reading["draws"])
    margins = study.STUDY.margins
    chosen = sizing.calibration_size(
        margins,
        {CALIBRATION_ARM: reading["ratio"]},
        draws,
        _one_label,
        biases={CALIBRATION_ARM: reading["bias"]},
    )
    if not chosen.budget_met:
        raise ValueError("no calibration size up to the cap meets the budget")
    high = _ratio_upper(reading["ratio"], draws)
    controls = (
        properties.SHRUNKEN_SE_FACTOR * high,
        high / np.sqrt(2.0),
        _ratio_upper(reading["mar_ratio"], draws),
    )
    replicates = chosen.replicates
    while (
        min(
            sizing.calibration_control_probability(margins, replicates, value) for value in controls
        )
        < 1.0 - sizing.DISCRIMINATION_FAIL
    ):
        replicates += sizing.BUDGET_STEP
    return replicates


def necessity_size(pilot: pd.DataFrame) -> int:
    """The smallest multiple of 200 at which the necessity family passes its share.

    The positive's bias equivalence at the one-sided 80% upper bound of its pilot bias, times
    the control's discrimination at the lower bound of its pilot bias (read in the positive's
    spread, as the verdict reads it), must reach
    :data:`~tests.studies.repeated_sizing.SIZED_SHARE`.
    """
    rows = _kept(pilot, NECESSITY)
    truth = rows["truth"].to_numpy(float)
    targeted = rows["estimate"].to_numpy(float) - truth
    plug = rows["initial_estimate"].to_numpy(float) - truth
    sd = float(np.std(targeted, ddof=1))
    shift = float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(rows))
    positive_bias = abs(float(np.mean(targeted))) / sd + shift
    control_gap = max(abs(float(np.mean(plug))) / sd - shift, 0.0)
    margins = study.STUDY.margins
    step = sizing.PROPERTY_STEP
    for replicates in range(step, sizing.MAXIMUM_REPLICATES + 1, step):
        joint = float(sizing.bias_equivalent_probability(margins, positive_bias, replicates)) * (
            sizing.bias_discriminated_probability(margins, control_gap, replicates)
        )
        if joint >= sizing.SIZED_SHARE:
            return replicates
    raise ValueError("no necessity size up to the cap meets the share")


def necessity_limits(pilot: pd.DataFrame) -> dict[str, float]:
    """The large-n limits of both necessity arms, in the n = 1,000 positive arm's SD.

    The mean error of the :data:`LIMIT_DRAWS` fits at n = :data:`LIMIT_N`, and its standard
    error over the fits, each divided by the empirical SD of the targeted arm in the
    ``necessity`` pilot at n = 1,000.
    """
    unit = float(
        np.std(
            _kept(pilot, NECESSITY)["estimate"].to_numpy(float) - properties.TRUTH,
            ddof=1,
        )
    )
    rows = _kept(pilot, LIMIT)
    targeted = (rows["estimate"].to_numpy(float) - properties.TRUTH) / unit
    untargeted = (rows["initial_estimate"].to_numpy(float) - properties.TRUTH) / unit
    return {
        "targeted": float(np.mean(targeted)),
        "targeted_se": float(np.std(targeted, ddof=1) / np.sqrt(len(targeted))),
        "untargeted": float(np.mean(untargeted)),
        "untargeted_se": float(np.std(untargeted, ddof=1) / np.sqrt(len(untargeted))),
    }


def sizes(pilot: pd.DataFrame | None = None) -> dict[str, int]:
    """Every declared size the committed pilot implies."""
    frame = pd.read_csv(OUTPUT, low_memory=False) if pilot is None else pilot
    return {
        "primary": primary_size(frame),
        "calibration": calibration_size(frame),
        "necessity": necessity_size(frame),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.labels)
