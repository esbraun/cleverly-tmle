"""The pilots and reported readings of the ``mtp-density-tails`` study, checkpointed per draw.

No label draws a declared sample: every draw takes ``stream_seed(STUDY, "pilot", label, r)``, a
stream no declared cell reads.  Each finished draw appends its rows to
``tests/canonical/mtp_density_tails/pilot-checkpoints/<label>.jsonl`` at once, so a stopped run
loses at most the draws in flight, and a restart skips every replicate already there.
``--collect`` folds every checkpoint into ``pilot.csv.gz``; a later pilot set's draw supersedes
an earlier one's, which stays marked (:data:`PILOT_SETS`).

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
``ic_gap_tail_<law>_n20000``       the oracle-hazard gap at n = 20,000 on each gated law, 3,000
                                   draws: the exceedance count the 20,000 clause's Jeffreys
                                   beta-binomial pass probability reads
``cluster_<law>_n<size>``          the three cluster readings on doses alone, 3,000 draws: C at
                                   20,000, G and T at 2,000 and 20,000
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

A calibration pilot whose plug-in pass probability is below 0.90 is extended by
:data:`EXTENSION_DRAWS` replicates on its own stream (``--replicates 400:800`` and so on), at most
:data:`MAX_EXTENSIONS` times; the design test applies the rule block by block and ignores every
draw past the block where it stops, so blocks may run ahead of the reading.

Every row records the draw's per-fit seconds (``seconds``), so the design test rebudgets each
cell from its pilot.  Each shipped learned-hazard row also records the misfit signature's three
marks (``mark_exact``, ``mark_top5``, ``mark_logp``), so :func:`reported_readings` routes each
breach without a cached fit.  A fit that raises is a pilot finding, recorded with its class in
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
#: The committed reported readings, recomputed by a fast test from ``pilot.csv.gz`` and the
#: declared run's ``properties.csv``.
READINGS = ARTIFACTS / "readings.json"
PROPERTIES = ARTIFACTS / "properties.csv"
CHECKPOINTS = ARTIFACTS / "pilot-checkpoints"
#: The committed pilot sets, in the order they ran.  A later set's draw of a label supersedes
#: every row of the same label and replicate in an earlier set: those rows stay in
#: ``pilot.csv.gz`` with ``superseded`` set, and the readings read only the current rows
#: (:func:`load`).  Set C re-pilots the labels that the thin-stretch count's bound and the
#: one-sided observed bound of the classifier steps move.
PILOT_SETS: tuple[str, ...] = ("pilot-checkpoints", "pilot-checkpoints-c")
#: The shipped interval limits of the fast calibration pilots, refitted on the merged code: the
#: pilot rows predate the shipped skew-widened interval, and the gates' coverage leg reads it.
LIMITS = ARTIFACTS / "pilot-limits.csv.gz"
CALIBRATION_DRAWS = 400
#: An extension adds this many draws to a calibration pilot, at most this many times.
EXTENSION_DRAWS = 400
MAX_EXTENSIONS = 3
#: The 20,000 tail pilot and the cluster dose pilots: draws per label.
TAIL_DRAWS = 3_000
DOSE_DRAWS = 3_000
#: The cluster dose pilots: ``(law, n)``.
DOSE_SOURCES: tuple[tuple[str, int], ...] = (
    ("C", 20_000),
    ("G", 2_000),
    ("G", 20_000),
    ("T", 2_000),
    ("T", 20_000),
)
#: The booster's ceilings: the largest count of 20 draws at 2,000 above a ratio error of 1.0
#: before the grid choice is reopened for that law, whatever the signature says.  Each is the
#: 0.95 predictive quantile of the Jeffreys posterior of the probe's count (2 of 60 on G, 2 of
#: 20 on T, 0 of 20 on C, 0 of 30 on S, 1 of 10 on B, 0 of 10 on U), frozen before any booster
#: pilot draw.
BOOSTER_CEILINGS: dict[str, int] = {"G": 3, "T": 6, "C": 2, "S": 2, "B": 8, "U": 4}
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
        for name in study.CLASSIFIER_LAWS
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
    **{
        f"ic_gap_tail_{name}_n20000": ("ic_gap_tail", name, 20_000, TAIL_DRAWS)
        for name in study.GAP_GATED
    },
    **{f"cluster_{name}_n{n}": ("cluster", name, n, DOSE_DRAWS) for name, n in DOSE_SOURCES},
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
    "mark_exact",
    "mark_top5",
    "mark_logp",
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
        flags = (None, None, None) if marks is None else tuple(int(m) for m in marks.marks)
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
                mark_exact=flags[0],
                mark_top5=flags[1],
                mark_logp=flags[2],
                seconds=s,
            )
        )
    return rows


def _cluster_rows(
    label: str, replicate: int, case: str, n: int, frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """The shipped, gap-disabled and eager cluster readings of one draw's doses."""
    doses = frame["A"].to_numpy(dtype=float)
    return [
        _row(label, replicate, case, n, reading, estimate=float(flag))
        for reading, flag in (
            ("cluster", fits.warns(doses)),
            ("cluster_gap_disabled", fits.warns_gap_disabled(doses)),
            ("cluster_eager", fits.warns_eager(doses)),
        )
    ]


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
    if kind == "ic_gap_tail":
        fitted, s = _timed(lambda: fits.fit_density(case, frame))
        row = _row(label, replicate, case, n, "ic_gap", seconds=s)
        row["ratio_error"] = fits.ic_gap(case, frame, fitted)
        return [row]
    if kind == "cluster":
        return _cluster_rows(label, replicate, case, n, frame)
    if kind == "ic_gap":
        fitted, s = _timed(lambda: fits.fit_density(case, frame))
        row = _estimate_row(label, replicate, case, n, "ic_gap", fitted, truth, s)
        row["ratio_error"] = fits.ic_gap(case, frame, fitted)
        return [row, *_cluster_rows(label, replicate, case, n, frame)]
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


def _limits(label: str, replicate: int) -> dict[str, Any]:
    """One fast calibration draw refitted: its estimate, SE and shipped interval limits."""
    kind, case, n, _ = PILOTS[label]
    frame = law.draw(fits.law_of(case), n, seed(label, replicate))
    fitted = fits.fit_density(case, frame, outcome={"density": "correct", "dr": "wrong"}[kind])
    low, high = fitted.ci
    return {
        "label": label,
        "replicate": replicate,
        "estimate": float(fitted.estimate),
        "std_error": float(fitted.std_error),
        "ci_lower": float(low),
        "ci_upper": float(high),
    }


def limit_labels() -> tuple[str, ...]:
    """The calibration pilots whose shipped limits are stored: the density and DR arms."""
    return tuple(label for label, spec in PILOTS.items() if spec[0] in ("density", "dr"))


def write_limits(n_jobs: int, pilot_rows: pd.DataFrame, output: Path = LIMITS) -> pd.DataFrame:
    """Refit every current replicate of :func:`limit_labels` and write its shipped limits."""
    jobs = [
        (label, int(r))
        for label in limit_labels()
        for r in sorted(set(pilot_rows.loc[pilot_rows["label"] == label, "replicate"]))
    ]
    with ProcessPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        rows = list(pool.map(_limits, *zip(*jobs, strict=True), chunksize=20))
    frame = pd.DataFrame(rows).sort_values(["label", "replicate"], ignore_index=True)
    frame.to_csv(output, index=False, float_format="%.17g")
    return frame


def load_limits(path: Path = LIMITS) -> pd.DataFrame:
    """The stored shipped limits, read back exactly."""
    return pd.read_csv(path, float_precision="round_trip")


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
    """Fold every checkpoint into one frame, sorted, and write it to ``output``.

    The committed checkpoints read every set of :data:`PILOT_SETS` that exists, in order;
    another directory is one set.  ``pilot_set`` is a row's position in that order, and
    ``superseded`` marks a row whose label and replicate a later set drew again.
    """
    sets = [checkpoints]
    if checkpoints == CHECKPOINTS:
        sets += [ARTIFACTS / name for name in PILOT_SETS[1:]]
    frames = []
    for rank, directory in enumerate(sets):
        for path in sorted(directory.glob("*.jsonl")):
            part = pd.read_json(path, lines=True)
            part["pilot_set"] = rank
            frames.append(part)
    columns = [*COLUMNS, "pilot_set"]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    frame = frame.reindex(columns=columns)
    _refuse_mixed_sets(frame)
    latest = frame.groupby(["label", "replicate"])["pilot_set"].transform("max")
    frame["superseded"] = frame["pilot_set"] < latest
    frame = frame.sort_values(["label", "replicate", "pilot_set", "reading"], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def _refuse_mixed_sets(frame: pd.DataFrame) -> None:
    """Refuse a fold that would put two code versions in one reading.

    A label a later set reruns must cover every replicate an earlier set holds, or the rows it
    leaves out would stay current beside the rerun ones; and one set holds each
    ``(label, replicate, reading)`` once.
    """
    duplicated = frame.duplicated(["pilot_set", "label", "replicate", "reading"], keep=False)
    if duplicated.any():
        first = frame.loc[duplicated].iloc[0]
        raise ValueError(
            f"pilot set {first['pilot_set']} holds {first['label']} replicate "
            f"{first['replicate']} reading {first['reading']!r} more than once"
        )
    for (rank, label), rows in frame.groupby(["pilot_set", "label"]):
        earlier = frame.loc[(frame["label"] == label) & (frame["pilot_set"] < rank), "replicate"]
        missing = sorted(set(earlier) - set(rows["replicate"]))
        if missing:
            raise ValueError(
                f"pilot set {rank} reruns {label} without replicates {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''} that an earlier set holds"
            )


def load(path: Path = OUTPUT) -> pd.DataFrame:
    """The current rows of a folded pilot: every row :func:`collect` marked superseded is out."""
    frame = pd.read_csv(path, low_memory=False)
    if "superseded" in frame:
        frame = frame.loc[~frame["superseded"].astype(bool)].reset_index(drop=True)
    return frame


# ------------------------------------------------------------------ reported readings

#: The booster's tail reading: at 2,000 at most 10% of draws above 1.0 (2 of 20; none of the 5
#: stacked draws) and none above 3.0; at 20,000 at most 10% above 0.5 (1 of 10).  A draw past
#: the first threshold of its size is a breach.
TAIL_SHARE = 0.10
TAIL_SMALL, TAIL_SMALL_NONE = 1.0, 3.0
TAIL_LARGE = 0.5
#: The laws whose booster median ratio error falls by at least a third from 2,000 to 20,000.
FALL_LAWS = ("G", "C")
FALL_FACTOR = 2.0 / 3.0
#: With the declared support, the median ratio error is at most this times the
#: support-ignored reading's.
SUPPORT_SLACK = 1.10
#: The classifier steps readings: T's 20,000 median over its 2,000 median, with and without
#: the steps; the cut on Gcap and Bp2; the change allowed on G, C and S.
STEPS_FALL, STALL = 0.8, 0.9
STEPS_CUT = 0.75
STEPS_NEUTRAL = 0.10
STEPS_CUT_CASES = ("Gcap", "Bp2")
STEPS_NEUTRAL_CASES = ("G", "C", "S")
#: The classifier DR reading on G: the mean remainder proxy within this many SEs of zero.
DR_REMAINDER_G = 0.2
#: A T classifier DR coverage below this is reported with its remainder.
DR_COVERAGE_T = 0.90


def _number(value: Any) -> float | None:
    value = float(value)
    return None if np.isnan(value) else float(f"{value:.10g}")


def _shipped(rows: pd.DataFrame, label: str, reading: str) -> np.ndarray:
    selected = rows.loc[(rows["label"] == label) & (rows["reading"] == reading), "ratio_error"]
    return selected.to_numpy(dtype=float)


def _median(values: np.ndarray) -> float | None:
    return _number(np.median(values)) if values.size else None


def _tail_small(errors: np.ndarray) -> dict[str, Any]:
    above = int(np.sum(errors > TAIL_SMALL))
    extreme = int(np.sum(errors > TAIL_SMALL_NONE))
    return {
        "draws": int(errors.size),
        "above_1": above,
        "above_3": extreme,
        "holds": bool(above <= int(TAIL_SHARE * errors.size) and extreme == 0),
    }


def _ceiling(errors: np.ndarray, name: str) -> dict[str, Any]:
    above = int(np.sum(errors > TAIL_SMALL))
    return {
        "ceiling": BOOSTER_CEILINGS[name],
        "above_1": above,
        "reopened": bool(above > BOOSTER_CEILINGS[name]),
    }


def breach_routes(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Every breaching learned-hazard draw with its route from the three marks.

    A shipped booster, stacked or learned-support booster draw breaches above
    :data:`TAIL_SMALL` at 2,000 or :data:`TAIL_LARGE` at 20,000.  It is a learner limitation
    when all three marks hold, and a defect suspect otherwise, a missing mark included.
    """
    learned = rows["label"].astype(str).str.match(r"^(boosting|stacked|learned)_")
    selected = rows.loc[learned & rows["reading"].isin(["boosting", "stacked"])]
    out = []
    for _, row in selected.sort_values(["label", "replicate"]).iterrows():
        error = float(row["ratio_error"])
        threshold = TAIL_SMALL if int(row["n"]) == 2_000 else TAIL_LARGE
        if not error > threshold:
            continue
        marks = [row.get(column) for column in ("mark_exact", "mark_top5", "mark_logp")]
        held = all(mark is not None and not pd.isna(mark) and int(mark) == 1 for mark in marks)
        out.append(
            {
                "label": str(row["label"]),
                "replicate": int(row["replicate"]),
                "ratio_error": _number(error),
                "route": "learner_limitation" if held else "defect_suspect",
            }
        )
    return out


def _booster_readings(rows: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in law.LAWS:
        small = _shipped(rows, f"boosting_{name}_n2000", "boosting")
        large = _shipped(rows, f"boosting_{name}_n20000", "boosting")
        reading: dict[str, Any] = {
            "median_2000": _median(small),
            "tail_2000": _tail_small(small),
            **_ceiling(small, name),
        }
        if name in LEARNED_LAWS:
            above = int(np.sum(large > TAIL_LARGE))
            reading["median_20000"] = _median(large)
            reading["tail_20000"] = {
                "draws": int(large.size),
                "above_05": above,
                "holds": bool(above <= int(TAIL_SHARE * large.size)),
            }
        if name in FALL_LAWS and small.size and large.size:
            fall = float(np.median(large) / np.median(small))
            reading["fall"] = {"ratio": _number(fall), "holds": bool(fall <= FALL_FACTOR)}
        out[name] = reading
    return out


def _learned_readings(rows: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in law.SUPPORTED:
        label = f"learned_{name}"
        booster = _shipped(rows, label, "boosting")
        support: dict[str, Any] = {}
        for learner in ("boosting", "pooled_logistic"):
            declared = _shipped(rows, label, learner)
            pre_fix = _shipped(rows, label, f"{learner}_pre_fix")
            ignored = _shipped(rows, label, f"{learner}_support_ignored")
            holds = bool(
                declared.size
                and pre_fix.size
                and ignored.size
                and np.median(declared) < np.median(pre_fix)
                and np.median(declared) <= SUPPORT_SLACK * np.median(ignored)
            )
            support[learner] = {
                "median": _median(declared),
                "pre_fix_median": _median(pre_fix),
                "support_ignored_median": _median(ignored),
                "holds": holds,
            }
        out[name] = {"tail_2000": _tail_small(booster), **_ceiling(booster, name), **support}
    return out


def _steps_readings(rows: pd.DataFrame) -> dict[str, Any]:
    def median(case: str, n: int, reading: str) -> float:
        values = _shipped(rows, f"classifier_{case}_n{n}", reading)
        return float(np.median(values)) if values.size else float("nan")

    out: dict[str, Any] = {}
    with_fall = median("T", 20_000, "with_steps") / median("T", 2_000, "with_steps")
    without_fall = median("T", 20_000, "without_steps") / median("T", 2_000, "without_steps")
    out["T"] = {
        "with_steps_fall": _number(with_fall),
        "without_steps_fall": _number(without_fall),
        "holds": bool(with_fall <= STEPS_FALL and without_fall >= STALL),
    }
    for case in (*STEPS_CUT_CASES, *STEPS_NEUTRAL_CASES):
        ratio = median(case, 2_000, "with_steps") / median(case, 2_000, "without_steps")
        holds = ratio <= STEPS_CUT if case in STEPS_CUT_CASES else abs(ratio - 1) <= STEPS_NEUTRAL
        out.setdefault(case, {}).update(steps_ratio_2000=_number(ratio), holds=bool(holds))
    return out


def _summary_row(summary: pd.DataFrame, cell: str) -> pd.Series | None:
    selected = summary.loc[
        (summary["property"] == "interval_calibration") & (summary["cell"] == cell)
    ]
    return None if selected.empty else selected.iloc[0]


def _classifier_dr_readings(summary: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cell in ("G_classifier_dr__reported", "T_classifier_dr__reported"):
        row = _summary_row(summary, cell)
        if row is None:
            continue
        remainder = float(row["remainder_proxy"])
        coverage = float(row["coverage"])
        reading: dict[str, Any] = {
            "bias_in_se": _number(float(row["bias"]) / float(row["empirical_se"])),
            "remainder": _number(remainder),
            "coverage": _number(coverage),
        }
        if cell.startswith("G_"):
            reading["holds"] = bool(abs(remainder) <= DR_REMAINDER_G)
        else:
            reading["holds"] = bool(remainder < 0.0)
            reading["coverage_below_090"] = bool(coverage < DR_COVERAGE_T)
        out[cell] = reading
    row = _summary_row(summary, "G_classifier_dr_both_wrong__reported")
    if row is not None:
        out["G_classifier_dr_both_wrong__reported"] = {
            "bias_in_se": _number(float(row["bias"]) / float(row["empirical_se"])),
            "remainder": _number(float(row["remainder_proxy"])),
            "coverage": _number(float(row["coverage"])),
        }
    return out


def _reported_calibration(summary: pd.DataFrame) -> dict[str, Any]:
    low, high = study.STUDY.margins.calibration_se_ratio
    c_low, c_high = study.STUDY.margins.calibration_coverage
    selected = summary.loc[
        (summary["property"] == "interval_calibration")
        & summary["cell"].astype(str).str.endswith("__reported")
        & ~summary["cell"].astype(str).str.contains("_classifier_dr")
    ]
    out: dict[str, Any] = {}
    for _, row in selected.sort_values("cell").iterrows():
        inside = bool(
            low <= float(row["se_ratio_ci_lower"])
            and float(row["se_ratio_ci_upper"]) <= high
            and c_low <= float(row["coverage_ci_lower"])
            and float(row["coverage_ci_upper"]) <= c_high
        )
        out[str(row["cell"])] = {
            "se_ratio": _number(row["se_ratio"]),
            "coverage": _number(row["coverage"]),
            "inside_band": inside,
        }
    return out


def reported_readings(pilot_rows: pd.DataFrame, property_rows: pd.DataFrame) -> dict[str, Any]:
    """Every declared reported reading, from the pilot rows and the declared run's summary.

    ``pilot_rows`` is ``pilot.csv.gz``; ``property_rows`` is the declared run's
    ``properties.csv``.  The booster's median falls, its tail readings and each law's count
    against :data:`BOOSTER_CEILINGS` (also on the boosters of ``learned_B`` and ``learned_U``);
    the stacked tail reading; the B and U support readings; the classifier steps readings; the
    classifier DR bias, remainder and coverage; the reported calibration rows; and the route of
    every breach (:func:`breach_routes`).  The stacked cells read no ceiling.
    """
    stacked = {
        name: _tail_small(_shipped(pilot_rows, f"stacked_{name}", "stacked"))
        for name in LEARNED_LAWS
    }
    return {
        "boosting": _booster_readings(pilot_rows),
        "learned": _learned_readings(pilot_rows),
        "stacked": stacked,
        "classifier_steps": _steps_readings(pilot_rows),
        "classifier_dr": _classifier_dr_readings(property_rows),
        "reported_calibration": _reported_calibration(property_rows),
        "breaches": breach_routes(pilot_rows),
    }


def write_readings(
    pilot_path: Path = OUTPUT, properties_path: Path = PROPERTIES, output: Path = READINGS
) -> dict[str, Any]:
    """Recompute :func:`reported_readings` from the committed artifacts and write it."""
    readings = reported_readings(load(pilot_path), pd.read_csv(properties_path))
    output.write_text(json.dumps(readings, indent=2, sort_keys=True) + "\n")
    return readings


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
    parser.add_argument("--readings", action="store_true", help="write readings.json")
    parser.add_argument(
        "--limits", action="store_true", help="refit the fast calibration pilots' limits"
    )
    arguments = parser.parse_args()
    if arguments.limits:
        write_limits(arguments.jobs, load())
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
    if arguments.readings:
        write_readings()
