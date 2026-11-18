"""The pilot of the ``hazard-time-varying-multinode`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``tests/canonical/hazard_time_varying_multinode/pilot.csv.gz``.  A draw whose fit held a cell
(:class:`~tests.studies.canonical_hazard_time_varying.HeldCellError`) writes one row with
``held`` 1, as the declared run records a lost replicate; any other exception writes one row
with its class in ``refusal``.  Run it as
``python -m tests.studies.hazard_time_varying_multinode_pilot --jobs N --labels <label>``, one
label per run: a label that runs replaces its rows and keeps the others.

=====================================  ==========================================================
label                                  what it measures
=====================================  ==========================================================
``calibration_<fit>``                  the calibration fits of ``multinode`` (banded, adding the
                                       rows ``joint_band`` and ``joint_pointwise``) and
                                       ``crossfit``, n = 2,000, 8,000 draws each.  They size the
                                       calibration family, and ``multinode``'s sizes the
                                       primary rows
``wrong_learner_<configuration>``      the four configurations with a wrong learner
                                       (``outcome_correct``, ``mechanism_correct``,
                                       ``continuation_wrong`` and ``both_wrong``) at n = 2,000,
                                       800 draws each (plan X70 rev 2.7): the untargeted backward
                                       plug-in and the targeted estimate of each draw.  Each
                                       sizes its configuration's robustness fits
``untargeted_outcome_correct_n8000``   the untargeted backward plug-in of ``outcome_correct``
                                       at n = 8,000, 800 draws (plan X70 rev 3.8): it sizes the
                                       untargeted ``outcome_correct`` arms, which run at that n
``fits_differ_multinode``              the ``iterated_mean_difference`` family at n = 2,000, 200
                                       draws: the hazard fit, the iterated-mean fit and the
                                       isotonic iterated-mean fit
``ic_gap``                             each fit label's influence-curve gap against the exact
                                       efficient influence function (5 draws at n = 2,000 and
                                       one at n = 20,000), and one fit at n = 40,000 per fit
                                       label whose influence-curve SD is read against the bound
=====================================  ==========================================================

Order of the runner work: the calibration and wrong-learner pilots, one label per run and
committed after each; then the row thread sets the replicate counts from them; then the
declaration commit with no skipped design test; then the declared run.  Rev 3.8 added
``untargeted_outcome_correct_n8000`` after the first eight labels were committed: it runs alone,
appending its rows, and the untargeted arms' count waits on it.

Every hazard fit's row holds node 1's initial plug-in (``initial_estimate``) and the fit's
accepted steps over every node walk (``walk_steps``), and every row records its key's fits:
``converged``, ``final_score`` and ``unexpected_warnings``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_hazard_time_varying_multinode as study
from tests.studies import hazard_time_varying_multinode_properties as properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows

OUTPUT = study.STUDY.artifacts / "pilot.csv.gz"
PILOT_N = 2_000
FITS_DIFFER_DRAWS = 200
WRONG_LEARNER_DRAWS = 800
CALIBRATION_DRAWS = 8_000
UNTARGETED_DRAWS = 800
BOUND_N = 40_000
IC_GAP_JOBS: tuple[tuple[str, int, int], ...] = (
    *(
        (fit, n, r)
        for fit in study.FITS
        for n, draws in zip(properties.IC_GAP_SIZES, (properties.IC_GAP_DRAWS, 1), strict=True)
        for r in range(draws)
    ),
    *((fit, BOUND_N, 0) for fit in study.FITS),
)

#: Each wrong-learner label: ``label -> configuration``.
WRONG_LEARNER_LABELS: dict[str, str] = {
    label: configuration for configuration, label in properties.WRONG_LEARNER_ARMS.items()
}
JOINT_ROWS = ("joint_band", "joint_pointwise")
#: The untargeted ``outcome_correct`` pilot at the arms' own n (plan X70 rev 3.8).
UNTARGETED_LABEL = "untargeted_outcome_correct_n8000"

#: ``label -> (fit label, draws)``; ``ic_gap`` runs :data:`IC_GAP_JOBS`.
PILOTS: dict[str, tuple[str, int]] = {
    **{f"calibration_{fit}": (fit, CALIBRATION_DRAWS) for fit in study.FITS},
    **dict.fromkeys(WRONG_LEARNER_LABELS, ("multinode", WRONG_LEARNER_DRAWS)),
    UNTARGETED_LABEL: ("multinode", UNTARGETED_DRAWS),
    **{f"fits_differ_{fit}": (fit, FITS_DIFFER_DRAWS) for fit in study.DIFFERENCE_FITS},
    "ic_gap": ("", len(IC_GAP_JOBS)),
}
#: The labels drawn at another n than :data:`PILOT_N`.
PILOT_SIZES: dict[str, int] = {UNTARGETED_LABEL: properties.UNTARGETED_N}


def pilot_n(label: str) -> int:
    """The n a pilot label draws (``ic_gap`` draws per job, :data:`IC_GAP_JOBS`)."""
    return PILOT_SIZES.get(label, PILOT_N)


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
    *properties.INFLUENCE_COLUMNS,
    "held",
    "refusal",
)


def _row(label: str, replicate: int, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS, np.nan)
    row.update(label=label, replicate=replicate, n=pilot_n(label), held=0, refusal="")
    row.update(values)
    return row


def _estimate_rows(label: str, replicate: int, fit_label: str, result: Any) -> list[dict]:
    """One row per estimate of the fit, against its exact truth."""
    initials = study.initial_estimates(result)
    steps = study.walk_steps(result)
    rows = []
    for name, reference in study.TRUTH["multinode"].items():
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


def _fits_differ(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    fit, _ = PILOTS[label]
    hazard = records.hazard(fit, frame)
    iterated = records.iterated(fit, frame)
    isotonic = records.iterated(fit, frame, isotonic=True)
    rows = _estimate_rows(label, replicate, fit, hazard)
    for row in rows:
        other = iterated[str(row["estimand"])]
        low, high = other.ci
        row["iterated_estimate"] = float(other.psi)
        row["iterated_std_error"] = float(other.std_error)
        row["iterated_covered"] = int(low <= float(row["truth"]) <= high)
        row["isotonic_estimate"] = float(isotonic[str(row["estimand"])].psi)
    drop, error = properties.largest_decrease(iterated)
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
    return rows


def _calibration(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    """Every estimate of a calibration fit, and the joint band's rows where it has one."""
    fit, draws = PILOTS[label]
    joint = f"{fit}_curve"
    banded = joint in properties.JOINT_LABELS
    result = records.hazard(fit, frame, simultaneous=banded)
    rows = _estimate_rows(label, replicate, fit, result)
    if banded:
        band, pointwise = joint_coverage_rows(
            result,
            study.TRUTH["multinode"],
            tuple(result.estimates),
            label=joint,
            replicate=replicate,
            n=PILOT_N,
            requested=draws,
            pointwise_critical=properties.CRITICAL,
        )
        for name, reading in zip(JOINT_ROWS, (band, pointwise), strict=True):
            rows.append(
                _row(label, replicate, fit_label=fit, estimand=name, covered=reading["covered"])
            )
    return rows


def _wrong_learner(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    configuration = WRONG_LEARNER_LABELS[label]
    wrong = records.hazard("multinode", frame, configuration=configuration)
    plug_in = records.untargeted("multinode", frame, configuration=configuration)
    rows = _estimate_rows(label, replicate, "multinode", wrong)
    for row in rows:
        name = str(row["estimand"])
        row["wrong_plug_in"] = float(plug_in[name].psi)
        row["wrong_targeted"] = float(wrong[name].psi)
    return rows


def _untargeted(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    """Every estimate of the untargeted ``outcome_correct`` backward pass, against its truth."""
    configuration = "outcome_correct"
    plug_in = records.untargeted("multinode", frame, configuration=configuration)
    rows = []
    for name, reference in study.TRUTH["multinode"].items():
        estimate = plug_in[name]
        low, high = estimate.ci
        rows.append(
            _row(
                label,
                replicate,
                fit_label="multinode",
                estimand=name,
                truth=reference,
                estimate=float(estimate.psi),
                std_error=float(estimate.std_error),
                covered=int(low <= reference <= high),
                wrong_plug_in=float(estimate.psi),
            )
        )
    return rows


def _ic_gap(replicate: int) -> list[dict[str, Any]]:
    """One fit of :data:`IC_GAP_JOBS`: each arm's influence-curve reading."""
    fit_label, n, draw = IC_GAP_JOBS[replicate]
    current = study.law.LAW
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
        except study.HeldCellError:  # the declared run's lost replicate, recorded as one
            return [_row(label, replicate, held=1)]
        except Exception as error:  # a refusal is a pilot finding, recorded with its class
            return [_row(label, replicate, refusal=type(error).__name__)]
    fit_label, _ = PILOTS[label]
    seed = stream_seed(study.STUDY, "pilot", label, replicate)
    frame = properties.draw(fit_label, pilot_n(label), seed)
    records = properties.FitRecords()
    try:
        if label.startswith("fits_differ_"):
            rows = _fits_differ(label, replicate, frame, records)
        elif label in WRONG_LEARNER_LABELS:
            rows = _wrong_learner(label, replicate, frame, records)
        elif label == UNTARGETED_LABEL:
            rows = _untargeted(label, replicate, frame, records)
        else:
            rows = _calibration(label, replicate, frame, records)
    except study.HeldCellError:  # the declared run's lost replicate, recorded as one
        return [_row(label, replicate, fit_label=fit_label, held=1)]
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
