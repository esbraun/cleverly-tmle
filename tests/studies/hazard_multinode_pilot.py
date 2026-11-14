"""The pilot of the ``hazard-multinode`` study, on its own seed streams.

It draws no declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  It writes one row per fit and estimand to
``tests/canonical/hazard_multinode/pilot.csv.gz``.  A draw whose fit held a prefix cell
(:class:`~tests.studies.canonical_hazard_multinode.HeldCellError`) writes one row with
``held`` 1, as the declared run records a lost replicate; any other exception writes one row
with its class in ``refusal``, which every reading test rejects.  Run
it as ``python -m tests.studies.hazard_multinode_pilot --jobs N --labels <label>``, one label
per run: the pilot has no checkpoints and writes only when the label finishes, so a restart
loses at most one label.  A label that runs replaces its rows and keeps the others.

=================================  ==============================================================
label                              what it measures
=================================  ==============================================================
``calibration_<fit>``              the calibration fits of one primary scenario of
                                   :data:`~tests.studies.canonical_hazard_multinode.SCENARIOS`,
                                   n = 2,000, 8,000 draws each
                                   (:data:`~tests.studies.repeated_sizing.CALIBRATION_PILOT_DRAWS`).
                                   They size the calibration family, and, being the primary
                                   fits of every scenario, the primary rows.  ``base`` and
                                   ``four`` fit with ``simultaneous=True``, as the declared
                                   calibration fits do, and add the joint band's two rows
``wrong_learner_<fit>``            the wrong hazard learner and the correct mechanism on
                                   ``base``, ``four`` and ``msm`` at n = 2,000, 800 draws each:
                                   the fit's plug-in and targeted estimates.  They size every
                                   positive arm that fits the wrong hazard learner by the rule
                                   of plan X70 rev 2.7
``wrong_learner_node``             the node-only hazard learner and the correct mechanism on
                                   ``four``, 800 draws, sized by the same rule
``both_wrong_base``                the ``both_wrong`` configuration on ``base``, 20 draws: its
                                   convergence and time, which no other label measures
``fits_differ_<fit>``              the ``iterated_mean_difference`` family on ``base`` and
                                   ``four`` at n = 2,000, 200 draws each.  Each draw fits the
                                   declared hazard fit, the iterated-mean fit of the same
                                   regimens, and the iterated-mean fit with ``isotonic=True``
``fits_differ_mechanism_correct``  the ``base`` ``mechanism_correct`` configuration at
                                   n = 2,000, 200 draws: the hazard fit with
                                   :class:`~tests.studies.hazard_multinode_law.LastArmLogistic`
                                   and the iterated-mean fit with regressions that omit ``W``,
                                   both with the correct mechanism
=================================  ==============================================================

Order of the runner work (plan X70 rev 2.4, design review d): the calibration pilots and the
wrong-learner pilots, one label per run and committed after each; then the row thread sets the
replicate counts from them; then the declaration commit with no skipped design test; then the
declared run.

A ``fits_differ`` row holds the hazard fit's ``estimate``, ``std_error`` and ``covered``, the
iterated-mean fit's (``iterated_estimate``, ``iterated_std_error``, ``iterated_covered``) and
the isotonic fit's ``isotonic_estimate``; its ``largest_decrease`` row holds the largest decrease
of an iterated-mean risk curve between nodes as ``estimate`` and that difference's standard
error as ``std_error``.  A ``wrong_learner`` row holds the plug-in and targeted estimates
(``wrong_plug_in``, ``wrong_targeted``).  A calibration row reads every estimate of the fit;
the joint rows ``joint_band`` and ``joint_pointwise`` hold in ``covered`` whether the band, and
every pointwise interval, covered the whole curve.  Every row records its key's fits:
``converged`` (1 when every hazard walk met its tolerance), ``final_score`` (the largest final
relative score of a walk), ``n_iter`` (the largest iteration count of a walk),
``unexpected_warnings`` (the count of warnings outside the expected classes of
:data:`~tests.studies.canonical_hazard_multinode.EXPECTED_WARNINGS`; no warning is suppressed
wholesale), ``smallest_cumulative`` (the smallest estimated cumulative treatment and retention
probability, before ``g_bounds``) and ``seconds`` (the draw's wall time on one core).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_hazard_multinode as study
from tests.studies import hazard_multinode_properties as properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows

OUTPUT = study.STUDY.artifacts / "pilot.csv.gz"
PILOT_N = 2_000
FITS_DIFFER_DRAWS = 200
MECHANISM_DIFFER_DRAWS = 200
WRONG_LEARNER_DRAWS = 800
CALIBRATION_DRAWS = 8_000
BOTH_WRONG_DRAWS = 20
#: Every fit label with a positive arm that fits the wrong hazard learner.
WRONG_LEARNER_FITS = properties.CALIBRATION_FITS
#: The pilot label of the node-only hazard learner.
NODE_PILOT = "wrong_learner_node"

#: ``label -> (fit label, draws)``.
PILOTS: dict[str, tuple[str, int]] = {
    **{f"calibration_{fit}": (fit, CALIBRATION_DRAWS) for fit in properties.CALIBRATION_FITS},
    **{f"wrong_learner_{fit}": (fit, WRONG_LEARNER_DRAWS) for fit in WRONG_LEARNER_FITS},
    NODE_PILOT: (properties.FIT_OF[properties.NODE_LABEL], WRONG_LEARNER_DRAWS),
    "both_wrong_base": ("base", BOTH_WRONG_DRAWS),
    **{f"fits_differ_{fit}": (fit, FITS_DIFFER_DRAWS) for fit in study.DIFFERENCE_FITS},
    "fits_differ_mechanism_correct": ("base", MECHANISM_DIFFER_DRAWS),
}
#: The estimand names of the joint band's two rows on a calibration label.
JOINT_ROWS = ("joint_band", "joint_pointwise")
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
    "final_score",
    "n_iter",
    "unexpected_warnings",
    "smallest_cumulative",
    "seconds",
    "held",
    "refusal",
)


def _row(label: str, replicate: int, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS, np.nan)
    row.update(label=label, replicate=replicate, n=PILOT_N, held=0, refusal="")
    row.update(values)
    return row


def _estimate_rows(label: str, replicate: int, fit_label: str, result: Any) -> list[dict]:
    """One row per estimate of the fit, against its exact truth."""
    rows = []
    for name, reference in study.TRUTH[fit_label].items():
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
    drop, error = properties.largest_decrease(iterated, fit)
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


def _mechanism_differ(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    hazard = records.hazard("base", frame, configuration="mechanism_correct")
    iterated = records.iterated("base", frame, regression_correct=False)
    return _paired_rows(label, replicate, "base", hazard, iterated)


def _calibration(
    label: str, replicate: int, frame: pd.DataFrame, records: Any
) -> list[dict[str, Any]]:
    """Every estimate of a calibration fit, and the joint band's rows where it has one."""
    fit, _ = PILOTS[label]
    joint = f"{fit}_curve"
    banded = joint in properties.JOINT_LABELS
    result = records.hazard(fit, frame, simultaneous=banded)
    rows = _estimate_rows(label, replicate, fit, result)
    if banded:
        band, pointwise = joint_coverage_rows(
            result,
            study.truths(fit),
            tuple(result.estimates),
            label=joint,
            replicate=replicate,
            n=PILOT_N,
            requested=PILOTS[label][1],
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
    fit, _ = PILOTS[label]
    configuration = {NODE_PILOT: properties.NODE_CONFIGURATION, "both_wrong_base": "both_wrong"}
    wrong = records.hazard(fit, frame, configuration=configuration.get(label, "mechanism_correct"))
    rows = _estimate_rows(label, replicate, fit, wrong)
    for row in rows:
        name = str(row["estimand"])
        row["wrong_plug_in"] = properties.plug_in(wrong, name)
        row["wrong_targeted"] = float(wrong[name].psi)
    return rows


def _one(job: tuple[str, int]) -> list[dict[str, Any]]:
    label, replicate = job
    fit_label, _ = PILOTS[label]
    frame = properties.draw(fit_label, PILOT_N, stream_seed(study.STUDY, "pilot", label, replicate))
    records = properties.FitRecords()
    start = time.perf_counter()
    try:
        if label == "fits_differ_mechanism_correct":
            rows = _mechanism_differ(label, replicate, frame, records)
        elif label.startswith("fits_differ_"):
            rows = _fits_differ(label, replicate, frame, records)
        elif label.startswith("wrong_learner_") or label == "both_wrong_base":
            rows = _wrong_learner(label, replicate, frame, records)
        else:
            rows = _calibration(label, replicate, frame, records)
    except study.HeldCellError:  # the declared run's lost replicate, recorded as one
        return [_row(label, replicate, fit_label=fit_label, held=1)]
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        return [_row(label, replicate, fit_label=fit_label, refusal=type(error).__name__)]
    columns = {
        **records.columns(),
        "smallest_cumulative": records.smallest_cumulative(),
        "seconds": time.perf_counter() - start,
    }
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
