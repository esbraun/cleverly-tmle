"""The pilot of the ``hazard-mtp`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``tests/canonical/hazard_mtp/pilot.csv.gz``, with the fit's refusal when it had one.  Run it as
``python -m tests.studies.hazard_mtp_pilot --jobs N --labels <label>``, one label per run: the
pilot has no checkpoints and writes only when the label finishes, so a restart loses at most one
label.  A label that runs replaces its rows and keeps the others.

=================================  ==============================================================
label                              what it measures
=================================  ==============================================================
``fits_differ_<fit>``              the ``iterated_mean_difference`` family on ``levels`` and
                                   ``gaussian_density`` at n = 2,000, 200 draws each.  Each draw
                                   fits the declared hazard fit, the iterated-mean fit of the
                                   same plans, and the iterated-mean fit with ``isotonic=True``
``fits_differ_mechanism_correct``  the ``levels`` ``mechanism_correct`` configuration at
                                   n = 2,000, 200 draws: the hazard fit with
                                   :class:`~tests.studies.hazard_mtp_law.NoWLogistic` and the
                                   iterated-mean fit with regressions that omit ``W``, both with
                                   the correct mechanism
``wrong_learner_<fit>``            the wrong hazard learner and the correct mechanism on
                                   ``levels``, ``gaussian_density``, ``gaussian_classifier``,
                                   ``gaussian_crossfit`` and ``vector`` at n = 2,000, 200 draws
                                   each: the fit's plug-in and targeted estimates
``ic_gap``                         each fit label's influence-curve gap against the exact
                                   efficient influence function (5 draws at n = 2,000 and one
                                   at n = 20,000, the reported ``untruncated_density`` too), and
                                   one fit at n = 40,000 per gated fit label whose
                                   influence-curve SD is read against the bound
``calibration_<fit>``              the calibration fits of one fit label of
                                   :data:`~tests.studies.canonical_hazard_mtp.FITS`, n = 2,000,
                                   8,000 draws each
                                   (:data:`~tests.studies.repeated_sizing.CALIBRATION_PILOT_DRAWS`).
                                   They size the calibration family, and the primary fit labels'
                                   pilots size the primary rows, whose fits they are
=================================  ==============================================================

Order of the runner work (plan X70 rev 2.4, design review d): the calibration pilots, one label
per run and committed after each; then the row thread sets the replicate counts from them; then
the declaration commit with no skipped design test; then the declared run.

A ``fits_differ`` row holds the hazard fit's ``estimate``, ``std_error`` and ``covered``, the
iterated-mean fit's (``iterated_estimate``, ``iterated_std_error``, ``iterated_covered``) and
the isotonic fit's ``isotonic_estimate``; its ``largest_decrease`` row holds the largest decrease
of the iterated-mean risk curve between visits as ``estimate`` and that difference's standard
error as ``std_error``.  A ``wrong_learner`` row holds the plug-in and targeted estimates
(``wrong_plug_in``, ``wrong_targeted``).  A calibration row reads every estimate of the fit.
Every row records its key's fits: ``converged`` (1 when every hazard walk met its tolerance)
and ``unexpected_warnings`` (the count of warnings outside the expected classes of
:data:`~tests.studies.canonical_hazard_mtp.EXPECTED_WARNINGS`); no warning is suppressed
wholesale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_hazard_mtp as study
from tests.studies import hazard_mtp_properties as properties
from tests.studies.evidence.seeds import stream_seed

OUTPUT = study.STUDY.artifacts / "pilot.csv.gz"
PILOT_N = 2_000
FITS_DIFFER_DRAWS = 200
MECHANISM_DIFFER_DRAWS = 200
WRONG_LEARNER_DRAWS = 200
CALIBRATION_DRAWS = 8_000
#: Every fit label with a positive arm that fits the wrong hazard learner.
WRONG_LEARNER_FITS = (
    "levels",
    "gaussian_density",
    "gaussian_classifier",
    "gaussian_crossfit",
    "vector",
)

#: The n of the large fits whose influence-curve SD is read against the efficiency bound.
BOUND_N = 40_000
#: Every fit of the ``ic_gap`` label: ``(fit label, n, draw)``.  Each gated fit label and the
#: reported one take :data:`~tests.studies.hazard_mtp_properties.IC_GAP_DRAWS` draws at the
#: small size and one at the large size; each gated fit label takes one draw at
#: :data:`BOUND_N`.
IC_GAP_JOBS: tuple[tuple[str, int, int], ...] = (
    *(
        (fit, n, r)
        for fit in (*properties.CALIBRATION_FITS, *properties.REPORTED_FITS)
        for n, draws in zip(properties.IC_GAP_SIZES, (properties.IC_GAP_DRAWS, 1), strict=True)
        for r in range(draws)
    ),
    *((fit, BOUND_N, 0) for fit in properties.CALIBRATION_FITS),
)

#: ``label -> (fit label, draws)``; ``ic_gap`` runs :data:`IC_GAP_JOBS`, one fit label each.
PILOTS: dict[str, tuple[str, int]] = {
    **{f"fits_differ_{fit}": (fit, FITS_DIFFER_DRAWS) for fit in study.DIFFERENCE_FITS},
    "fits_differ_mechanism_correct": ("levels", MECHANISM_DIFFER_DRAWS),
    **{f"wrong_learner_{fit}": (fit, WRONG_LEARNER_DRAWS) for fit in WRONG_LEARNER_FITS},
    **{f"calibration_{fit}": (fit, CALIBRATION_DRAWS) for fit in properties.CALIBRATION_FITS},
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
    "wrong_plug_in",
    "wrong_targeted",
    "converged",
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
    truth = study.truths(study.FITS[fit_label][0])
    rows = []
    for name, reference in truth.items():
        estimate = result[name]
        low, high = estimate.ci
        rows.append(
            _row(
                label,
                replicate,
                estimand=name,
                truth=reference,
                estimate=float(estimate.psi),
                std_error=float(estimate.std_error),
                covered=int(low <= reference <= high),
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
    hazard = records.hazard(fit, frame)
    iterated = records.iterated(fit, frame)
    isotonic = records.iterated(fit, frame, isotonic=True)
    rows = _paired_rows(label, replicate, fit, hazard, iterated, isotonic)
    drop, error = properties.largest_decrease(iterated, study.law_of(fit).n_times)
    rows.append(
        _row(
            label, replicate, estimand="largest_decrease", truth=0.0, estimate=drop, std_error=error
        )
    )
    return rows


def _mechanism_differ(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    hazard = records.hazard("levels", frame, configuration="mechanism_correct")
    iterated = records.iterated("levels", frame, regression_correct=False)
    return _paired_rows(label, replicate, "levels", hazard, iterated)


def _wrong_learner(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    fit, _ = PILOTS[label]
    wrong = records.hazard(fit, frame, configuration="mechanism_correct")
    rows = _estimate_rows(label, replicate, fit, wrong)
    for row in rows:
        name = str(row["estimand"])
        row["wrong_plug_in"] = properties.plug_in(wrong, name)
        row["wrong_targeted"] = float(wrong[name].psi)
    return rows


def _ic_gap(replicate: int) -> list[dict[str, Any]]:
    """One fit of :data:`IC_GAP_JOBS`: each arm's influence-curve reading.

    At a size of :data:`~tests.studies.hazard_mtp_properties.IC_GAP_SIZES` the ``estimate`` is
    the gap SD(IC - EIF) / bound against the exact per-unit efficient influence function of
    :meth:`~tests.studies.hazard_mtp_law.Law.frame_curves`; at :data:`BOUND_N` it is
    SD(IC) / bound.
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
        return [_row(label, replicate, refusal=type(error).__name__)]
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
