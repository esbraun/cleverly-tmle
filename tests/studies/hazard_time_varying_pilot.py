"""The pilot of the ``hazard-time-varying`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``tests/canonical/hazard_time_varying/pilot.csv.gz``, with the fit's refusal when it had one.
Run it as ``python -m tests.studies.hazard_time_varying_pilot --jobs N --labels <label>``, one
label per run: the pilot writes only when the label finishes, so a restart loses at most one
label.  A label that runs replaces its rows and keeps the others.

=================================  ==============================================================
label                              what it measures
=================================  ==============================================================
``calibration_<fit>``              the calibration fits of one fit label of
                                   :data:`~tests.studies.canonical_hazard_time_varying.FITS`,
                                   n = 2,000, 8,000 draws each
                                   (:data:`~tests.studies.repeated_sizing.CALIBRATION_PILOT_DRAWS`).
                                   They size the calibration family, and the primary fit labels'
                                   pilots size the primary rows, whose fits they are
``wrong_learner_one_cause``        the hazard and continuation without ``L`` with the correct
                                   mechanism on ``one_cause`` at n = 2,000, 800 draws (plan X70
                                   rev 2.7): the untargeted backward plug-in and the targeted
                                   estimate of each draw.  It sizes the wrong-learner arms
``fits_differ_<fit>``              the ``iterated_mean_difference`` family on ``one_cause`` and
                                   ``competing`` at n = 2,000, 200 draws each: the hazard fit,
                                   the iterated-mean fit and the isotonic iterated-mean fit
``fits_differ_mechanism_correct``  the ``one_cause`` ``mechanism_correct`` configuration at
                                   n = 2,000, 200 draws: the hazard fit and the iterated-mean
                                   fit, both with regressions that omit ``L``; its hazard fit is
                                   the wrong-learner fit on its own seeds, and the sizing rule
                                   pools it (plan X70 rev 2.7)
``ic_gap``                         each fit label's influence-curve gap against the exact
                                   efficient influence function (5 draws at n = 2,000 and one at
                                   n = 20,000), and one fit at n = 40,000 per fit label whose
                                   influence-curve SD is read against the bound
=================================  ==============================================================

Order of the runner work (plan X70 rev 2.4): the calibration pilots and the wrong-learner pilot,
one label per run and committed after each; then the row thread sets the replicate counts from
them; then the declaration commit with no skipped design test; then the declared run.

A ``wrong_learner`` row holds the plug-in and targeted estimates (``wrong_plug_in``,
``wrong_targeted``).  A ``fits_differ`` row holds the hazard fit's ``estimate``, ``std_error``
and ``covered``, the iterated-mean fit's (``iterated_estimate``, ``iterated_std_error``,
``iterated_covered``) and the isotonic fit's ``isotonic_estimate``; its ``largest_decrease`` row
holds the largest decrease of an iterated-mean curve between nodes, and on ``competing`` its
``largest_excess`` rows hold each fit's largest sum of cumulative incidences minus one.  Every
hazard fit's row holds node 1's initial plug-in (``initial_estimate``), so
``estimate - initial_estimate`` is node 1's walk move (plan X70 section 5 asks the pilot for the
mean walk move), and the fit's accepted steps over every node walk of every regimen
(``walk_steps``).  Every row records its key's fits: ``converged``, ``final_score`` and
``unexpected_warnings``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_hazard_time_varying as study
from tests.studies import hazard_time_varying_properties as properties
from tests.studies.evidence.seeds import stream_seed

OUTPUT = study.STUDY.artifacts / "pilot.csv.gz"
PILOT_N = 2_000
FITS_DIFFER_DRAWS = 200
MECHANISM_DIFFER_DRAWS = 200
WRONG_LEARNER_DRAWS = 800
CALIBRATION_DRAWS = 8_000
#: The n of the large fits whose influence-curve SD is read against the efficiency bound.
BOUND_N = 40_000
#: Every fit of the ``ic_gap`` label: ``(fit label, n, draw)``.
IC_GAP_JOBS: tuple[tuple[str, int, int], ...] = (
    *(
        (fit, n, r)
        for fit in properties.CALIBRATION_FITS
        for n, draws in zip(properties.IC_GAP_SIZES, (properties.IC_GAP_DRAWS, 1), strict=True)
        for r in range(draws)
    ),
    *((fit, BOUND_N, 0) for fit in properties.CALIBRATION_FITS),
)

#: ``label -> (fit label, draws)``; ``ic_gap`` runs :data:`IC_GAP_JOBS`, one fit label each.
PILOTS: dict[str, tuple[str, int]] = {
    **{f"calibration_{fit}": (fit, CALIBRATION_DRAWS) for fit in properties.CALIBRATION_FITS},
    "wrong_learner_one_cause": ("one_cause", WRONG_LEARNER_DRAWS),
    **{f"fits_differ_{fit}": (fit, FITS_DIFFER_DRAWS) for fit in study.DIFFERENCE_FITS},
    "fits_differ_mechanism_correct": ("one_cause", MECHANISM_DIFFER_DRAWS),
    "ic_gap": ("", len(IC_GAP_JOBS)),
}
COLUMNS = (
    "label",
    "replicate",
    "fit_label",
    "n",
    "estimand",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "iterated_estimate",
    "iterated_std_error",
    "iterated_covered",
    "isotonic_estimate",
    "initial_estimate",
    "walk_steps",
    "wrong_plug_in",
    "wrong_targeted",
    "converged",
    "final_score",
    "unexpected_warnings",
    "refusal",
)


def _row(label: str, replicate: int, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS, np.nan)
    row.update(label=label, replicate=replicate, n=PILOT_N, refusal="")
    row.update(values)
    return row


def _estimate_rows(label: str, replicate: int, fit_label: str, result: Any) -> list[dict]:
    """One row per estimate of the fit, against its exact truth."""
    law_name = study.FITS[fit_label][0]
    truth = study.truths(law_name)
    initials = study.initial_estimates(result, law_name)
    steps = sum(int(regimen_fit.hazard.n_iter) for regimen_fit in result.fits.values())
    rows = []
    for name, reference in truth.items():
        estimate = result[name]
        low, high = estimate.ci
        rows.append(
            _row(
                label,
                replicate,
                fit_label=fit_label,
                estimand=name,
                truth=reference,
                estimate=float(estimate.psi),
                std_error=float(estimate.std_error),
                covered=int(low <= reference <= high),
                initial_estimate=initials[name],
                walk_steps=steps,
            )
        )
    return rows


def _paired_rows(
    label: str, replicate: int, fit: str, hazard: Any, iterated: Any, isotonic: Any = None
) -> list[dict[str, Any]]:
    rows = _estimate_rows(label, replicate, fit, hazard)
    for row in rows:
        other = iterated[str(row["estimand"])]
        low, high = other.ci
        row["iterated_estimate"] = float(other.psi)
        row["iterated_std_error"] = float(other.std_error)
        row["iterated_covered"] = int(low <= float(row["truth"]) <= high)
        if isotonic is not None:
            row["isotonic_estimate"] = float(isotonic[str(row["estimand"])].psi)
    return rows


def _fits_differ(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    fit, _ = PILOTS[label]
    law_name = study.FITS[fit][0]
    hazard = records.hazard(fit, frame)
    iterated = records.iterated(fit, frame)
    isotonic = records.iterated(fit, frame, isotonic=True)
    rows = _paired_rows(label, replicate, fit, hazard, iterated, isotonic)
    drop, error = properties.largest_decrease(iterated, law_name)
    rows.append(
        _row(
            label,
            replicate,
            fit_label=fit,
            estimand="largest_decrease",
            truth=0.0,
            estimate=drop,
            std_error=error,
        )
    )
    if fit in study.COHERENCE_FITS:
        for arm, result in zip(study.COHERENCE_ARMS, (hazard, iterated, isotonic), strict=True):
            rows.append(
                _row(
                    label,
                    replicate,
                    fit_label=fit,
                    estimand=f"largest_excess_{arm}",
                    truth=0.0,
                    estimate=properties.largest_excess(result, law_name)[0],
                )
            )
    return rows


def _mechanism_differ(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    hazard = records.hazard("one_cause", frame, configuration="mechanism_correct")
    iterated = records.iterated("one_cause", frame, regression_correct=False)
    return _paired_rows(label, replicate, "one_cause", hazard, iterated)


def _wrong_learner(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    fit, _ = PILOTS[label]
    wrong = records.hazard(fit, frame, configuration="mechanism_correct")
    plug_in = records.untargeted(fit, frame, configuration="mechanism_correct")
    rows = _estimate_rows(label, replicate, fit, wrong)
    for row in rows:
        name = str(row["estimand"])
        row["wrong_plug_in"] = float(plug_in[name].psi)
        row["wrong_targeted"] = float(wrong[name].psi)
    return rows


def _ic_gap(replicate: int) -> list[dict[str, Any]]:
    """One fit of :data:`IC_GAP_JOBS`: each arm's influence-curve reading.

    At a size of :data:`~tests.studies.hazard_time_varying_properties.IC_GAP_SIZES` the
    ``estimate`` is the gap SD(IC - EIF) / bound against the exact per-unit efficient influence
    function; at :data:`BOUND_N` it is SD(IC) / bound.
    """
    fit_label, n, draw = IC_GAP_JOBS[replicate]
    current = study.law_of(fit_label)
    frame = current.draw(n, stream_seed(study.STUDY, "pilot", "ic_gap", fit_label, n, draw))
    records = properties.FitRecords()
    result = records.hazard(fit_label, frame)
    rows = []
    for label, name in properties.COEFFICIENTS[fit_label]:
        curve = np.asarray(result[name].influence_curve)
        if n != BOUND_N:
            curve = curve - current.frame_curves(properties.target(label), frame)
        rows.append(
            _row(
                "ic_gap",
                replicate,
                fit_label=fit_label,
                n=n,
                estimand=label,
                estimate=float(np.std(curve) / properties.EFFICIENCY_SD[label]),
                std_error=float(result[name].std_error),
                **records.columns(),
            )
        )
    return rows


def _one(job: tuple[str, int]) -> list[dict[str, Any]]:
    label, replicate = job
    if label == "ic_gap":
        try:
            return _ic_gap(replicate)
        except Exception as error:  # a refusal is a pilot finding, recorded with its class
            return [_row(label, replicate, refusal=type(error).__name__)]
    fit_label, _ = PILOTS[label]
    frame = properties.draw(fit_label, PILOT_N, stream_seed(study.STUDY, "pilot", label, replicate))
    records = properties.FitRecords()
    try:
        if label == "fits_differ_mechanism_correct":
            rows = _mechanism_differ(label, replicate, frame, records)
        elif label.startswith("fits_differ_"):
            rows = _fits_differ(label, replicate, frame, records)
        elif label.startswith("wrong_learner_"):
            rows = _wrong_learner(label, replicate, frame, records)
        else:
            rows = _estimate_rows(label, replicate, fit_label, records.hazard(fit_label, frame))
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        return [_row(label, replicate, fit_label=fit_label, refusal=type(error).__name__)]
    columns = records.columns()
    return [{**row, **columns} for row in rows]


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
