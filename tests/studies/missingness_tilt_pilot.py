"""The sizing pilot of the ``missingness-tilt-inference`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``missingness_tilt_pilot.csv.gz`` beside this module, with the fit's refusal when it had one.
Run it as ``python -m tests.studies.missingness_tilt_pilot --jobs N [--labels ...]``.

Two sets of labels.  The ``_local`` labels are the small pilot run on a workstation before the
design review: three primary scenarios and both property families at a few hundred draws.  The
others are the full pilot the runner executes before the declared run; once their rows exist,
:func:`sizes` reads them in place of the local ones.

======================  =========================================================================
label                   what it sizes
======================  =========================================================================
``<scenario>``          each primary scenario at its declared n, :data:`PRIMARY_DRAWS` draws
``calibration``         the calibration fit (tilted ``ate`` at gamma 2 with the A1-era plug-in
                        and MAR SE), :data:`CALIBRATION_DRAWS` draws
``necessity``           the main-terms outcome fit with its plug-in, :data:`NECESSITY_DRAWS`
``<scenario>_local``    ``in_sample``, ``cross_fit`` and ``flexible_q``, 200 draws each
``calibration_local``   the calibration fit, 1,000 draws
``necessity_local``     the necessity fit, 400 draws
======================  =========================================================================
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
from scipy.stats import beta, norm

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
LOCAL_PRIMARY = (study.IN_SAMPLE, study.CROSS_FIT, study.FLEXIBLE_Q)

CALIBRATION = "calibration"
NECESSITY = "necessity"

#: ``label -> (scenario or family, n, draws, kind)``; ``kind`` is ``"primary"``,
#: ``"calibration"`` or ``"necessity"``.
PILOTS: dict[str, tuple[str, int, int, str]] = {
    **{
        scenario: (scenario, study.STUDY.sizes(study.PRIMARY_N)[scenario], PRIMARY_DRAWS, "primary")
        for scenario in study.SCENARIOS
    },
    CALIBRATION: (CALIBRATION, properties.PROPERTY_N, CALIBRATION_DRAWS, CALIBRATION),
    NECESSITY: (NECESSITY, properties.PROPERTY_N, NECESSITY_DRAWS, NECESSITY),
    **{
        f"{scenario}_local": (
            scenario,
            study.STUDY.sizes(study.PRIMARY_N)[scenario],
            200,
            "primary",
        )
        for scenario in LOCAL_PRIMARY
    },
    f"{CALIBRATION}_local": (CALIBRATION, properties.PROPERTY_N, 1_000, CALIBRATION),
    f"{NECESSITY}_local": (NECESSITY, properties.PROPERTY_N, 400, NECESSITY),
}
LOCAL_LABELS = tuple(label for label in PILOTS if label.endswith("_local"))


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
    """Run ``labels`` (the local labels by default) and replace their rows in the CSV."""
    chosen = list(labels) if labels else list(LOCAL_LABELS)
    jobs = [(label, r) for label in chosen for r in range(PILOTS[label][2])]
    outcomes = map_parallel(_one, [(job,) for job in jobs], n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows])
    if OUTPUT.exists():
        kept = pd.read_csv(OUTPUT)
        frame = pd.concat([kept.loc[~kept["label"].isin(chosen)], frame], ignore_index=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


# ---------------------------------------------------------------------------- sizing


def _label(pilot: pd.DataFrame, base: str) -> str | None:
    """The full label when its rows exist, else the local one, else ``None``."""
    for label in (base, f"{base}_local"):
        if (pilot["label"] == label).any():
            return label
    return None


def _ratio_bounds(ratio: float, draws: int) -> tuple[float, float]:
    """The one-sided 80% bounds of an SE ratio read on ``draws`` normal estimates."""
    spread = ratio / np.sqrt(2.0 * (draws - 1))
    z = float(norm.ppf(sizing.PILOT_LEVEL))
    return ratio - z * spread, ratio + z * spread


def _bias_bound(errors: np.ndarray, *, upper: bool) -> float:
    """The one-sided 80% bound of a standardized bias, away from (upper) or toward zero."""
    sd = float(np.std(errors, ddof=1))
    bias = float(np.mean(errors)) / sd
    shift = float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(errors))
    return abs(bias) + shift if upper else max(abs(bias) - shift, 0.0)


def primary_size(pilot: pd.DataFrame) -> int:
    """:func:`~tests.studies.repeated_sizing.primary_replicates` on every piloted scenario.

    One implementation, the default family-wise pass 0.95, read on each primary cell's
    covered count; every label must hold the same number of draws.
    """
    covered: dict[str, int] = {}
    draws: set[int] = set()
    for scenario in study.SCENARIOS:
        label = _label(pilot, scenario)
        if label is None:
            continue
        rows = pilot.loc[
            (pilot["label"] == label) & (pilot["refusal"].isna() | (pilot["refusal"] == ""))
        ]
        draws.add(int(rows["replicate"].nunique()))
        for estimand, group in rows.groupby("estimand"):
            covered[f"{scenario}/{estimand}"] = int(group["covered"].sum())
    if len(draws) != 1:
        raise ValueError(f"the piloted scenarios hold different draw counts: {sorted(draws)}")
    return sizing.primary_replicates(study.STUDY.margins, covered, draws.pop(), implementations=1)


def calibration_size(pilot: pd.DataFrame) -> int:
    """The smallest multiple of 400 at which the calibration family passes its share.

    The positive's pass probability (:func:`~tests.studies.repeated_sizing.calibration_pass`)
    at the worse of the pilot SE ratio's one-sided 80% bounds must reach
    :data:`~tests.studies.repeated_sizing.SIZED_SHARE`, and each SE control (shrunken, noise
    with ratio ``1/sqrt(2)``, and the A1-era control at the upper bound of its pilot ratio) must
    leave the band with probability at least ``1 - DISCRIMINATION_FAIL``.
    """
    label = _label(pilot, CALIBRATION)
    if label is None:
        raise ValueError("no calibration pilot rows")
    rows = pilot.loc[
        (pilot["label"] == label) & (pilot["refusal"].isna() | (pilot["refusal"] == ""))
    ]
    draws = len(rows)
    errors = rows["estimate"].to_numpy(float) - rows["truth"].to_numpy(float)
    spread = float(np.std(errors, ddof=1))
    ratio = float(rows["std_error"].mean()) / spread
    plug_errors = rows["initial_estimate"].to_numpy(float) - rows["truth"].to_numpy(float)
    mar_ratio = float(rows["mar_std_error"].mean()) / float(np.std(plug_errors, ddof=1))
    margins = study.STUDY.margins
    low, high = _ratio_bounds(ratio, draws)
    mar_high = _ratio_bounds(mar_ratio, draws)[1]
    controls = (properties.SHRUNKEN_SE_FACTOR * high, 1.0 / np.sqrt(2.0) * high, mar_high)
    for replicates in range(sizing.BUDGET_STEP, sizing.MAXIMUM_REPLICATES + 1, sizing.BUDGET_STEP):
        positive = float(
            np.min(sizing.calibration_pass(margins, replicates, np.array([low, high])))
        )
        discriminated = min(
            sizing.calibration_control_probability(margins, replicates, value) for value in controls
        )
        if positive >= sizing.SIZED_SHARE and discriminated >= 1.0 - sizing.DISCRIMINATION_FAIL:
            return replicates
    return sizing.MAXIMUM_REPLICATES


def necessity_size(pilot: pd.DataFrame) -> int:
    """The smallest multiple of 200 at which the necessity family passes its share."""
    label = _label(pilot, NECESSITY)
    if label is None:
        raise ValueError("no necessity pilot rows")
    rows = pilot.loc[
        (pilot["label"] == label) & (pilot["refusal"].isna() | (pilot["refusal"] == ""))
    ]
    truth = rows["truth"].to_numpy(float)
    targeted = rows["estimate"].to_numpy(float) - truth
    plug = rows["initial_estimate"].to_numpy(float) - truth
    sd = float(np.std(targeted, ddof=1))
    positive_bias = _bias_bound(targeted, upper=True)
    # The control's bias is read in the positive arm's spread, as the verdict reads it.
    control_gap = max(
        abs(float(np.mean(plug))) / sd - float(norm.ppf(sizing.PILOT_LEVEL)) / np.sqrt(len(plug)),
        0.0,
    )
    margins = study.STUDY.margins
    step = sizing.PROPERTY_STEP
    for replicates in range(step, sizing.MAXIMUM_REPLICATES + 1, step):
        joint = float(sizing.bias_equivalent_probability(margins, positive_bias, replicates)) * (
            sizing.bias_discriminated_probability(margins, control_gap, replicates)
        )
        if joint >= sizing.SIZED_SHARE:
            return replicates
    return sizing.MAXIMUM_REPLICATES


def coverage_lower_bound(covered: int, draws: int) -> float:
    """The one-sided 80% lower Clopper-Pearson bound of a pilot coverage."""
    return float(beta.ppf(1.0 - sizing.PILOT_LEVEL, covered, draws - covered + 1))


def sizes(pilot: pd.DataFrame | None = None) -> dict[str, int]:
    """Every declared size the committed pilot implies."""
    frame = pd.read_csv(OUTPUT) if pilot is None else pilot
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
