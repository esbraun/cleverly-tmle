"""The pilot of the ``hazard-msm`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``tests/canonical/hazard_msm/pilot.csv.gz``, with the fit's refusal when it had one.  Run it as
``python -m tests.studies.hazard_msm_pilot --jobs N [--labels ...]``; a label that runs replaces
its rows and keeps the others.

=================================  ==============================================================
label                              what it measures
=================================  ==============================================================
``fits_differ``                    the ``iterated_mean_difference`` family, on the ``overlap``
                                   law at n = 2,000, 200 draws.  Each draw fits the declared
                                   hazard fit, the iterated-mean fit of the same working model,
                                   and an iterated-mean fit with no working model (its risk
                                   curve)
``fits_differ_mechanism_correct``  the ``survival`` law's ``mechanism_correct`` configuration at
                                   n = 2,000, 200 draws: the hazard fit with
                                   :class:`~tests.studies.hazard_msm_law.NoW2Logistic` and the
                                   iterated-mean fit of the same working model with its
                                   regressions on ``W1`` only, both with the correct mechanism
``wrong_learner``                  the wrong hazard learner on the gated ``survival`` law at
                                   n = 2,000, 200 draws: the hazard fit with
                                   :class:`~tests.studies.hazard_msm_law.NoW2Logistic` and the
                                   correct mechanism
``calibration_<fit>``              the calibration cells of one fit label of
                                   :data:`~tests.studies.hazard_msm_properties.FITS`, n = 2,000,
                                   8,000 draws each
                                   (:data:`~tests.studies.repeated_sizing.CALIBRATION_PILOT_DRAWS`).
                                   They size the calibration family, and ``survival_logit`` and
                                   ``competing_logit`` size the primary rows, whose fits they are
=================================  ==============================================================

A ``fits_differ`` or ``fits_differ_mechanism_correct`` coefficient row holds the hazard fit's ``estimate``, ``std_error`` and
``covered``, and the iterated-mean fit's (``iterated_estimate``, ``iterated_std_error``,
``iterated_covered``).  A ``wrong_learner`` row holds the wrong learner's plug-in and targeted
coefficients (``wrong_plug_in``, ``wrong_targeted``).  Its
``largest_decrease`` row holds the largest decrease of the iterated-mean risk curve between
visits as ``estimate`` and that difference's standard error as ``std_error``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_hazard_msm as study
from tests.studies import hazard_msm_properties as properties
from tests.studies.evidence.seeds import stream_seed

OUTPUT = study.STUDY.artifacts / "pilot.csv.gz"
PILOT_N = 2_000
FITS_DIFFER_DRAWS = 200
MECHANISM_DIFFER_DRAWS = 200
WRONG_LEARNER_DRAWS = 200
CALIBRATION_DRAWS = 8_000

#: ``label -> (fit label, draws)``.
PILOTS: dict[str, tuple[str, int]] = {
    "fits_differ": (properties.DIFFERENCE_FIT, FITS_DIFFER_DRAWS),
    "fits_differ_mechanism_correct": ("survival_logit", MECHANISM_DIFFER_DRAWS),
    "wrong_learner": ("survival_logit", WRONG_LEARNER_DRAWS),
    **{f"calibration_{fit}": (fit, CALIBRATION_DRAWS) for fit in properties.CALIBRATION_FITS},
}
COLUMNS = (
    "label",
    "replicate",
    "n",
    "estimand",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "iterated_estimate",
    "iterated_std_error",
    "iterated_covered",
    "wrong_plug_in",
    "wrong_targeted",
    "refusal",
)


def _row(label: str, replicate: int, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS, np.nan)
    row.update(label=label, replicate=replicate, n=PILOT_N, refusal="", **values)
    return row


def _coefficient_rows(label: str, replicate: int, fit_label: str, result: Any) -> list[dict]:
    rows = []
    for cell_label, name in properties.COEFFICIENTS[fit_label]:
        estimate = result[name]
        low, high = estimate.ci
        truth = properties.TRUTHS[cell_label]
        rows.append(
            _row(
                label,
                replicate,
                estimand=name,
                truth=truth,
                estimate=float(estimate.psi),
                std_error=float(estimate.std_error),
                covered=int(low <= truth <= high),
            )
        )
    return rows


def _paired_rows(
    label: str, replicate: int, fit: str, hazard: Any, iterated: Any
) -> list[dict[str, Any]]:
    rows = _coefficient_rows(label, replicate, fit, hazard)
    for row in rows:
        other = iterated[str(row["estimand"])]
        low, high = other.ci
        row["iterated_estimate"] = float(other.psi)
        row["iterated_std_error"] = float(other.std_error)
        row["iterated_covered"] = int(low <= float(row["truth"]) <= high)
    return rows


def _mechanism_differ(label: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    hazard = properties.fit_label("survival_logit", frame, configuration="mechanism_correct")
    iterated = properties.iterated_fit(frame, scenario="survival", regression_correct=False)
    return _paired_rows(label, replicate, "survival_logit", hazard, iterated)


def _fits_differ(label: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    fit = properties.DIFFERENCE_FIT
    scenario, _ = properties.FITS[fit]
    hazard = properties.fit_label(fit, frame)
    iterated = properties.iterated_fit(frame, scenario=scenario)
    curve = properties.iterated_fit(frame, working_model=False, scenario=scenario)
    rows = _paired_rows(label, replicate, fit, hazard, iterated)
    drop, error = properties.largest_decrease(curve)
    rows.append(
        _row(
            label, replicate, estimand="largest_decrease", truth=0.0, estimate=drop, std_error=error
        )
    )
    return rows


def _wrong_learner(label: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    wrong = properties.fit_label("survival_logit", frame, configuration="mechanism_correct")
    rows = _coefficient_rows(label, replicate, "survival_logit", wrong)
    for row in rows:
        name = str(row["estimand"])
        row["wrong_plug_in"] = properties.plug_in(wrong, name)
        row["wrong_targeted"] = float(wrong[name].psi)
    return rows


def _one(job: tuple[str, int]) -> list[dict[str, Any]]:
    label, replicate = job
    fit_label, _ = PILOTS[label]
    frame = properties.draw(fit_label, PILOT_N, stream_seed(study.STUDY, "pilot", label, replicate))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if label == "fits_differ":
                return _fits_differ(label, replicate, frame)
            if label == "fits_differ_mechanism_correct":
                return _mechanism_differ(label, replicate, frame)
            if label == "wrong_learner":
                return _wrong_learner(label, replicate, frame)
            result = properties.fit_label(fit_label, frame)
            return _coefficient_rows(label, replicate, fit_label, result)
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        return [_row(label, replicate, refusal=type(error).__name__)]


def run(
    n_jobs: int,
    labels: list[str] | None = None,
    *,
    draws: int | None = None,
    output: Path = OUTPUT,
) -> pd.DataFrame:
    """Run ``labels`` (every label by default) and replace their rows in ``output``.

    ``draws`` caps each label for a smoke run, which must then write outside the committed
    file.
    """
    if draws is not None and output == OUTPUT:
        raise SystemExit("refused: a capped pilot writes to --output, not the committed pilot")
    chosen = labels if labels else list(PILOTS)
    jobs = [
        (label, r) for label in chosen for r in range(PILOTS[label][1] if draws is None else draws)
    ]
    outcomes = map_parallel(_one, [(job,) for job in jobs], n_jobs=n_jobs)
    frame = pd.DataFrame([row for rows in outcomes for row in rows], columns=list(COLUMNS))
    if output.exists():
        kept = pd.read_csv(output)
        frame = pd.concat([kept.loc[~kept["label"].isin(chosen)], frame], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.labels, draws=arguments.draws, output=arguments.output)
