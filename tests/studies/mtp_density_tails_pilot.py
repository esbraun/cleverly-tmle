"""The pilots and reported readings of the ``mtp-density-tails`` study, checkpointed per draw.

No label draws a declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  Each finished draw appends its rows to
``tests/canonical/mtp_density_tails/pilot-checkpoints/<label>.jsonl`` at once, so a stopped run
loses at most the draws in flight, and a restart skips every replicate already there.
``--collect`` folds every checkpoint into ``pilot.csv.gz``.

Run one label, or one replicate range of it on a cloud part, as::

    python -m tests.studies.mtp_density_tails_pilot --jobs 4 --labels <label> [--replicates A:B]

A capped smoke (``--draws N``) writes to ``--output``, a directory outside the committed one.

=================================  ==============================================================
label                              what it reads
=================================  ==============================================================
``calibration_density_<law>``      the density route with the oracle hazard and the correct
                                   regression, G, T and C, n = 2,000, 400 draws: sizes the
                                   gated density calibration arms
``calibration_dr_<law>``           the same with the wrong regression, G and T, n = 8,000, 400
``calibration_classifier_<law>``   the classifier route (default stack, five folds, the steps),
                                   G, T and S, n = 2,000, 400 draws; an extension adds 400
``calibration_<LTMLE|SDR>``        the longitudinal default route, n = 2,000, 400 draws
``ic_gap_<law>_n<size>``           the oracle-hazard gap on G, T, C, S, B and U, 80 draws at
                                   n = 2,000 and 30 at n = 20,000, and the doses' cluster
                                   readings (shipped, gap-disabled, eager) on the same draws
``boosting_<law>_n<size>``         the default library's booster read by the shipped reading and
                                   the pre-fix one, with the misfit signature on each draw: all
                                   six laws at n = 2,000 (20 draws), G, T, C and S at 20,000 (10)
``stacked_<law>``                  the default stacked density, the same two readings, G, T, C
                                   and S at n = 2,000, 5 draws
``logistic_<law>_n<size>``         the pooled and per-bin logistic hazards, G, T, C and S at
                                   n = 2,000 (20 draws) and 20,000 (10)
``learned_<B|U>``                  boosting and the pooled logistic hazard on B and U with the
                                   declared support and with it ignored, n = 2,000, 20 draws
``classifier_<case>_n2000``        the default stack's ratio with and without the steps on G, T,
                                   C, S, Gcap and Bp2, n = 2,000, 20 draws
``classifier_<law>_n20000``        the same at n = 20,000, 10 draws: without the steps on G, T,
                                   C and S, with them on T and G
=================================  ==============================================================

Every row records the draw's per-fit seconds (``seconds``), so the design test rebudgets each
cell from its pilot.  A fit that raises is a pilot finding, recorded with its class in
``refusal``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests import mtp_density_tails_law as law
from tests.studies import canonical_mtp_density_tails as study
from tests.studies import mtp_density_tails_fits as fits
from tests.studies.evidence.seeds import stream_seed

ARTIFACTS = study.STUDY.artifacts
OUTPUT = ARTIFACTS / "pilot.csv.gz"
CHECKPOINTS = ARTIFACTS / "pilot-checkpoints"
CALIBRATION_DRAWS = 400
#: The draws of each ``ic_gap`` size.
GAP_DRAWS = {2_000: 80, 20_000: 30}
LEARNED_LAWS = ("G", "T", "C", "S")
CLASSIFIER_CASES = ("G", "T", "C", "S", "Gcap", "Bp2")
#: The laws whose 20,000 classifier cells also run with the steps.
STEPS_AT_LARGE = ("T", "G")

#: ``label -> (kind, case, n, draws)``.
PILOTS: dict[str, tuple[str, str, int, int]] = {
    **{
        f"calibration_density_{name}": ("density", name, 2_000, CALIBRATION_DRAWS)
        for name in study.DENSITY_GATED
    },
    **{
        f"calibration_dr_{name}": ("dr", name, 8_000, CALIBRATION_DRAWS)
        for name in study.ROBUST_GATED
    },
    **{
        f"calibration_classifier_{name}": ("classifier", name, 2_000, CALIBRATION_DRAWS)
        for name in study.CLASSIFIER_GATED
    },
    **{
        f"calibration_{estimator}": ("longitudinal", estimator, 2_000, CALIBRATION_DRAWS)
        for estimator in study.LONGITUDINAL
    },
    **{
        f"ic_gap_{name}_n{n}": ("ic_gap", name, n, draws)
        for name in law.LAWS
        for n, draws in GAP_DRAWS.items()
    },
    **{f"boosting_{name}_n2000": ("boosting", name, 2_000, 20) for name in law.LAWS},
    **{f"boosting_{name}_n20000": ("boosting", name, 20_000, 10) for name in LEARNED_LAWS},
    **{f"stacked_{name}": ("stacked", name, 2_000, 5) for name in LEARNED_LAWS},
    **{f"logistic_{name}_n2000": ("logistic", name, 2_000, 20) for name in LEARNED_LAWS},
    **{f"logistic_{name}_n20000": ("logistic", name, 20_000, 10) for name in LEARNED_LAWS},
    **{f"learned_{name}": ("learned", name, 2_000, 20) for name in law.SUPPORTED},
    **{
        f"classifier_{case}_n2000": ("classifier_ratio", case, 2_000, 20)
        for case in CLASSIFIER_CASES
    },
    **{
        f"classifier_{name}_n20000": ("classifier_ratio", name, 20_000, 10) for name in LEARNED_LAWS
    },
}

COLUMNS = (
    "label",
    "replicate",
    "case",
    "n",
    "reading",
    "estimate",
    "std_error",
    "covered",
    "ratio_error",
    "exact_error",
    "worst_share",
    "log_gap",
    "seconds",
    "refusal",
)


def seed(label: str, replicate: int) -> int:
    """The draw seed of a pilot replicate."""
    return stream_seed(study.STUDY, "pilot", label, replicate)


def _row(label: str, replicate: int, case: str, n: int, reading: str, **values: Any) -> dict:
    row: dict[str, Any] = dict.fromkeys(COLUMNS, None)
    row.update(label=label, replicate=replicate, case=case, n=n, reading=reading, refusal="")
    row.update(values)
    return row


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start


def _estimate_row(
    label: str, replicate: int, case: str, n: int, reading: str, fitted: Any, truth: float, s: float
) -> dict[str, Any]:
    low, high = fitted.ci
    return _row(
        label,
        replicate,
        case,
        n,
        reading,
        estimate=fitted.estimate,
        std_error=fitted.std_error,
        covered=int(low <= truth <= high),
        seconds=s,
    )


def _learned_rows(
    label: str, replicate: int, case: str, frame: pd.DataFrame, learner: str, **options: Any
) -> list[dict[str, Any]]:
    name = options.pop("reading_name", learner)
    rows = []
    for kind in ("shipped", "pre_fix"):
        fitted, s = _timed(
            lambda kind=kind: fits.learned_density(
                case, frame, learner, reading_kind=kind, **options
            )
        )
        marks = fits.signature(case, frame, fitted) if kind == "shipped" else None
        rows.append(
            _row(
                label,
                replicate,
                case,
                len(frame),
                name if kind == "shipped" else f"{name}_pre_fix",
                ratio_error=fitted.error,
                exact_error=None if marks is None else marks.exact_error,
                worst_share=None if marks is None else marks.worst_share,
                log_gap=None if marks is None else marks.log_gap,
                seconds=s,
            )
        )
    return rows


def _one(label: str, replicate: int) -> list[dict[str, Any]]:
    kind, case, n, _ = PILOTS[label]
    if kind == "longitudinal":
        frame = fits.draw_longitudinal(n, seed(label, replicate))
        result, s = _timed(lambda: fits.fit_longitudinal(case, frame))
        rows = []
        for estimand in fits.LONGITUDINAL_ESTIMANDS:
            estimate = result[estimand]
            low, high = estimate.ci
            truth = fits.LONGITUDINAL_TRUTH[estimand]
            rows.append(
                _row(
                    label,
                    replicate,
                    case,
                    n,
                    estimand,
                    estimate=float(estimate.psi),
                    std_error=float(estimate.std_error),
                    covered=int(low <= truth <= high),
                    seconds=s,
                )
            )
        return rows
    frame = law.draw(fits.law_of(case), n, seed(label, replicate))
    truth = fits.truth(case)
    if kind == "density":
        fitted, s = _timed(lambda: fits.fit_density(case, frame))
        return [_estimate_row(label, replicate, case, n, "density", fitted, truth, s)]
    if kind == "dr":
        fitted, s = _timed(lambda: fits.fit_density(case, frame, outcome="wrong"))
        return [_estimate_row(label, replicate, case, n, "wrong_outcome", fitted, truth, s)]
    if kind == "classifier":
        fitted, s = _timed(lambda: fits.fit_classifier(case, frame))
        row = _estimate_row(label, replicate, case, n, "classifier", fitted, truth, s)
        row["ratio_error"] = fits.ratio_error(case, fitted.ratio, frame)
        return [row]
    if kind == "ic_gap":
        fitted, s = _timed(lambda: fits.fit_density(case, frame))
        row = _estimate_row(label, replicate, case, n, "ic_gap", fitted, truth, s)
        row["ratio_error"] = fits.ic_gap(case, frame, fitted)
        doses = frame["A"].to_numpy(dtype=float)
        clusters = [
            _row(label, replicate, case, n, reading, estimate=float(flag))
            for reading, flag in (
                ("cluster", fits.warns(doses)),
                ("cluster_gap_disabled", fits.warns_gap_disabled(doses)),
                ("cluster_eager", fits.warns_eager(doses)),
            )
        ]
        return [row, *clusters]
    if kind in {"boosting", "stacked"}:
        return _learned_rows(label, replicate, case, frame, kind)
    if kind == "logistic":
        return [
            *_learned_rows(label, replicate, case, frame, "pooled_logistic"),
            *_learned_rows(label, replicate, case, frame, "per_bin_logistic"),
        ]
    if kind == "learned":
        rows = []
        for learner in ("boosting", "pooled_logistic"):
            for with_support in (True, False):
                suffix = "" if with_support else "_support_ignored"
                rows += _learned_rows(
                    label,
                    replicate,
                    case,
                    frame,
                    learner,
                    with_support=with_support,
                    reading_name=f"{learner}{suffix}",
                )
        return rows
    # classifier_ratio
    rows = []
    with_steps = (True, False) if n == 2_000 or case in STEPS_AT_LARGE else (False,)
    for enabled in with_steps:
        ratio, s = _timed(
            lambda enabled=enabled: fits.classifier_ratio(case, frame, with_steps=enabled)
        )
        rows.append(
            _row(
                label,
                replicate,
                case,
                n,
                "with_steps" if enabled else "without_steps",
                ratio_error=fits.ratio_error(case, ratio, frame),
                seconds=s,
            )
        )
    return rows


def _guarded(label: str, replicate: int) -> list[dict[str, Any]]:
    try:
        return _one(label, replicate)
    except Exception as error:  # a refusal is a pilot finding, recorded with its class
        _, case, n, _ = PILOTS[label]
        return [_row(label, replicate, case, n, "", refusal=type(error).__name__)]


def done(path: Path) -> set[int]:
    """The replicates a checkpoint already holds."""
    if not path.exists():
        return set()
    with path.open() as handle:
        return {int(json.loads(line)["replicate"]) for line in handle if line.strip()}


def _clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(
    n_jobs: int,
    labels: list[str],
    *,
    replicates: range | None = None,
    draws: int | None = None,
    checkpoints: Path = CHECKPOINTS,
) -> None:
    """Fit every missing replicate of ``labels`` and append each draw to its checkpoint.

    ``replicates`` restricts a label to a range (a cloud part); ``draws`` caps each label for a
    smoke, which must then write outside the committed checkpoints.
    """
    if draws is not None and checkpoints == CHECKPOINTS:
        raise SystemExit("refused: a capped pilot writes to --output, not the committed pilot")
    checkpoints.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, int]] = []
    for label in labels:
        total = PILOTS[label][3] if draws is None else draws
        span = range(total) if replicates is None else replicates
        held = done(checkpoints / f"{label}.jsonl")
        jobs += [(label, r) for r in span if r not in held]
    with ProcessPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        futures = {pool.submit(_guarded, label, r): label for label, r in jobs}
        for future in as_completed(futures):
            label = futures[future]
            with (checkpoints / f"{label}.jsonl").open("a") as handle:
                for row in future.result():
                    handle.write(json.dumps({k: _clean(v) for k, v in row.items()}) + "\n")


def collect(checkpoints: Path = CHECKPOINTS, output: Path = OUTPUT) -> pd.DataFrame:
    """Fold every checkpoint into one frame, sorted, and write it to ``output``."""
    frames = [pd.read_json(path, lines=True) for path in sorted(checkpoints.glob("*.jsonl"))]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    frame = frame.reindex(columns=list(COLUMNS)).sort_values(
        ["label", "replicate", "reading"], ignore_index=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def _range(text: str | None) -> range | None:
    if text is None:
        return None
    start, stop = (int(part) for part in text.split(":"))
    return range(start, stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--replicates", default=None, help="START:STOP of one cloud part")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--output", type=Path, default=CHECKPOINTS)
    parser.add_argument("--collect", action="store_true")
    arguments = parser.parse_args()
    if arguments.labels:
        run(
            arguments.jobs,
            arguments.labels,
            replicates=_range(arguments.replicates),
            draws=arguments.draws,
            checkpoints=arguments.output,
        )
    if arguments.collect:
        collect(
            arguments.output,
            arguments.output.parent / "pilot.csv.gz" if arguments.output != CHECKPOINTS else OUTPUT,
        )
