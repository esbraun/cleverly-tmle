"""RM19: localize the one-sided DR-TMLE increment of ``cleverly`` over R ``drtmle``.

RM19 of ``docs/roadmap.md`` declares the design in "The localization design, declared before it
runs", as amended.  Each part runs on its own, in this order:

* A refits the registered primary draws of the three scenarios, validates the harness (V1 to
  V3 and V5) and attributes the committed signal;
* AV rebuilds the payloads of the committed Part A draws, counts R's rounds on them and fits the
  twin of ``T(0, 0, 0, 0)`` (rules 1 and 2), then validates Part A again from the committed rows.
  A declared Part A run takes this step itself;
* B fits 2,000 fresh ``treatment_correct`` draws at n = 3,000 with ``C`` and the 16 arms of the
  transcription;
* V4 runs the pinned R container on the first 200 Part B draws;
* C fits 2,000 fresh draws at n = 1,500 and 6,000 with ``C`` and ``T(0, 0, 0, 0)``.

    python -m tests.diagnostics.rm19_one_sided_increment.run --part A --output <scratch>

``--read-only`` rebuilds the part's reading from what ``--output`` holds.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from tests.canonical.drtmle.regenerate import HERE as REFERENCE_HERE
from tests.canonical.drtmle.regenerate import REFERENCE
from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm19_one_sided_increment.rtrans import (
    K_PARTS,
    TOL_IC,
    Solver,
    Switches,
    transcribe,
)
from tests.studies import canonical_drtmle as study
from tests.studies.canonical_drtmle import STUDY as BINARY
from tests.studies.evidence.inference import Interval, student_interval
from tests.studies.evidence.manifest import ROOT, write_csv
from tests.studies.evidence.seeds import replicate_seed

HERE = Path(__file__).resolve().parent
DESIGN = "one-sided-increment"
PARTS = ("A", "AV", "B", "V4", "C")
SCENARIO = "treatment_correct"
CONTROLS = ("outcome_correct", "both_correct")
ESTIMANDS = study.ESTIMANDS

#: The declared budget of each fresh part and its sizes, in the seed assignment order.
REPLICATES = 2_000
SIZES = {"B": (study.PRIMARY_N,), "C": (1_500, 6_000)}
#: V4: the first 200 Part B draws, at most seven R workers (``tests/canonical/drtmle/README``).
#: The counting runs of V4 and of step AV use the same seven, below the cap of eight (W5).
V4_DRAWS = 200
R_WORKERS = 7
#: Rule 1: the counting wrapper, in the pinned image of the canonical runner.  It mounts
#: ``tests/`` so that it can source ``canonical/drtmle/run_drtmle.R`` unchanged.
COUNTING = dataclasses.replace(
    REFERENCE,
    runner="diagnostics/rm19_one_sided_increment/count_rounds.R",
    runner_root=ROOT / "tests",
)

#: The budget rule's inputs: the registered similarity margin of the ``ate`` row, the committed
#: paired SD and the 99% normal quantile.  The resolution is one fifth of the margin.
MARGIN = 0.002990
COMMITTED_SD = 0.010261
Z = 2.575829
RESOLUTION = 0.000598
#: The bracket: ``T(1, 1, 1, 1) - C`` must lie inside this, one tenth of the committed increment.
BRACKET = 1e-4
#: V2 and V4 on a draw where R reaches ``maxIter``: the amended tolerance.  A draw that exits at
#: ``tolIC`` keeps the R4 tolerance of ``rm18_shared``.
MAX_ITER_TOLERANCE = 1e-4
TOL_IC_EXIT, MAX_ITER_EXIT = "tolIC", "maxIter"
#: The rule by round and conditioning (W2).  Rule 1: the counting rerun reproduces the committed
#: R rows to this.  Rule 2: the twin marks a draw rounding-sensitive when it moves an estimate
#: or a standard error by more than this.  Rule 4: a sensitive draw passes within this, when
#: the twin moves by at least this share of the draw's difference.  Rule 5 reuses the
#: ``maxIter`` tolerance.
RERUN_TOLERANCE = 1e-15
SENSITIVE_SHIFT = 1e-12
SENSITIVE_TOLERANCE = 1e-7
TWIN_SHARE = 0.1
#: The classes of rules 3 to 5, in the order of the validation rows.
SAME_TOL_IC, SAME_MAX_ITER, SENSITIVE, DIFFERS = (
    "same round, tolIC draws",
    "same round, maxIter draws",
    "rounding-sensitive tolIC draws",
    "round and exit status",
)
#: P-ctrl: the committed paired SD of each control scenario.
CONTROL_SD = {"outcome_correct": 0.002223, "both_correct": 0.000869}
#: The seed that ``random_partition`` cannot take as ``seed + 1``.
LAST_SEED = 2**32 - 1

#: The factorial arms, in ``(J, P, S, K)`` order, and the other named arms.
FACTORS = {"J": "joint_tilt", "P": "prime", "S": "cleverly_exit", "K": "cleverly_numerics"}
FACTORIAL = tuple(
    Switches(joint_tilt=bool(j), prime=bool(p), cleverly_exit=bool(s), cleverly_numerics=bool(k))
    for j, p, s, k in itertools.product((0, 1), repeat=len(FACTORS))
)
INTERACTIONS = tuple(itertools.combinations(FACTORS, 2))
REFERENCE_ARM = Switches()
FULL = Switches(True, True, True, True)
WITHOUT_K = Switches(True, True, True)
NO_GUARD = Switches(no_guard=True)
CONTROL_ARMS = (REFERENCE_ARM, Switches(joint_tilt=True), FULL)
#: K5, supplementary: each part of K alone, added to ``T(1, 1, 1, 0)``, on Part A.
DECOMPOSITION = tuple(Switches(True, True, True, part=part) for part in K_PARTS)
C = "C"
T0000 = REFERENCE_ARM.label
T1000 = Switches(joint_tilt=True).label
T1110 = WITHOUT_K.label
T1111 = FULL.label
T0G = NO_GUARD.label
#: Rule 2: ``T(0, 0, 0, 0)`` with a QR least-squares solve.
TWIN = f"{T0000}[qr]"

INCREMENT = "increment confirmed"
REVERSE = "reverse increment"
NO_INCREMENT = "no increment at the declared resolution"
UNRESOLVED = "unresolved"
UNEXPLAINED = "unexplained"
NOTHING = "nothing to localize"
INTERACTION_ONLY = "interaction only"
NO_LABEL = "no localization label"
COMMITTED = "committed-draw attribution"
SUPPLEMENTARY = "supplementary"
HOLDS, FAILS = shared.HOLDS, shared.FAILS

ROW_COLUMNS = (
    "part",
    "scenario",
    "n",
    "replicate",
    "seed",
    "arm",
    "truth",
    *ESTIMANDS,
    *(f"se_{name}" for name in ESTIMANDS),
    *(f"initial_{name}" for name in ESTIMANDS),
    "iterations",
    "exit",
    "closing",
    "guard_events",
    "at_bound",
    "score_max",
)


# ------------------------------------------------------------------------------- seeds


def labels(part: str) -> list[tuple[Any, ...]]:
    """The declared seed labels ``("rm19", DESIGN, n, k)`` of a fresh part, in order."""
    return [("rm19", DESIGN, n, k) for n in SIZES[part] for k in range(REPLICATES)]


def taken_seeds() -> set[int]:
    """What RM19's first label may not take: the registered seeds, BD-1's, and ``2^32 - 1``."""
    return rm18_seeds.binary_registered() | set(rm18_seeds.binary_seeds()) | {LAST_SEED}


@functools.cache
def fresh_seeds() -> Mapping[str, tuple[int, ...]]:
    """The declared seeds of Parts B and C, assigned in the declared order after BD-1.

    A read-only view, because the cache hands every caller the same object.
    """
    taken = taken_seeds()
    return MappingProxyType(
        {part: tuple(shared.fresh_seeds(BINARY, labels(part), taken)) for part in ("B", "C")}
    )


def draws(part: str, cap: int | None = None) -> list[tuple[str, int, int, int]]:
    """``(scenario, n, replicate, seed)`` of each draw a part fits, the first ``cap`` per size."""
    if part == "A":
        count = shared.budget(BINARY.replicates, cap)
        return [
            (scenario, study.PRIMARY_N, k, replicate_seed(BINARY, scenario, k))
            for scenario in study.SCENARIOS
            for k in range(count)
        ]
    count = shared.budget(REPLICATES, cap)
    return [
        (SCENARIO, n, k, seed)
        for (_, _, n, k), seed in zip(labels(part), fresh_seeds()[part], strict=True)
        if k < count
    ]


def arms(part: str, scenario: str) -> tuple[Switches, ...]:
    """The transcription arms of one part and scenario: the reference arm first, ``T0+G`` after
    the declared arms, and the K decomposition last on Part A ``treatment_correct``."""
    if part == "C":
        return (REFERENCE_ARM,)
    if scenario != SCENARIO:
        return (*CONTROL_ARMS, NO_GUARD)
    return (*FACTORIAL, NO_GUARD, *(DECOMPOSITION if part == "A" else ()))


# -------------------------------------------------------------------------------- fits


def payload_frame(frame: pd.DataFrame, result: Any) -> pd.DataFrame:
    """The shared initial arrays beside the draw, as ``canonical_drtmle._replicate`` builds them."""
    nuisance = result.repeats[0].nuisance
    out = frame.copy()
    out["qn0"] = nuisance.outcome.arms[0.0]
    out["qn1"] = nuisance.outcome.arms[1.0]
    out["gn1"] = nuisance.propensity.arm(1.0)
    return out


def _base(part: str, scenario: str, n: int, k: int, seed: int, truth: float) -> dict[str, Any]:
    return {
        "part": part,
        "scenario": scenario,
        "n": n,
        "replicate": k,
        "seed": seed,
        "truth": truth,
    }


def _initial(payload: pd.DataFrame) -> dict[str, float]:
    ey0, ey1 = float(payload["qn0"].mean()), float(payload["qn1"].mean())
    return {
        "initial_ey0": ey0,
        "initial_ey1": ey1,
        "initial_ate": float((payload["qn1"] - payload["qn0"]).mean()),
    }


def cleverly_row(result: Any) -> dict[str, Any]:
    """The ``C`` arm's estimates and instruments."""
    fluctuation = result.repeats[0].fluctuations["mean"]
    reduction = fluctuation.reduction
    upper = np.asarray(fluctuation.mechanism.propensity, dtype=float)
    lower, high = study.G_BOUNDS
    return {
        "arm": C,
        **{name: float(result.estimates[name].psi) for name in ESTIMANDS},
        **{f"se_{name}": float(result.estimates[name].std_error) for name in ESTIMANDS},
        "iterations": int(reduction.rounds),
        "exit": str(reduction.exit_reason),
        "closing": int(reduction.closing),
        "guard_events": 0,
        "at_bound": int(np.sum((upper <= lower) | (upper >= high))),
        "score_max": max(
            abs(float(row.score)) for row in result.diagnostics.score_equations().rows
        ),
    }


def transcribed_row(
    payload: pd.DataFrame, switches: Switches, *, solver: Solver = "gelsd"
) -> dict[str, Any]:
    out = transcribe(payload, switches, solver=solver)
    return {
        "arm": switches.label if solver == "gelsd" else TWIN,
        **out.estimates,
        **{f"se_{name}": value for name, value in out.std_errors.items()},
        "iterations": out.iterations,
        "exit": out.exit,
        "closing": out.closing,
        "guard_events": out.guard_events,
        "at_bound": out.at_bound,
        "score_max": out.score_max,
    }


def draw_payload(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, float, Any]:
    """One draw, its registered ``C`` fit, and the payload every arm reads."""
    frame, truth = study.draw_from_seed(scenario, n, seed)
    result = study.fit_cleverly(frame, scenario)
    return payload_frame(frame, result), float(truth["ate"]), result


def fit_draw(call: tuple[str, str, int, int, int]) -> list[dict[str, Any]]:
    """Every arm of one draw.  ``T0+G`` is a copy of ``T(0, 0, 0, 0)`` on a draw with no guard
    event, because the guards then change no step."""
    part, scenario, n, k, seed = call
    payload, truth, result = draw_payload(scenario, n, seed)
    common = {**_base(part, scenario, n, k, seed, truth), **_initial(payload)}
    rows = [{**common, **cleverly_row(result)}]
    reference: dict[str, Any] | None = None
    for switches in arms(part, scenario):
        if switches == NO_GUARD and reference is not None and reference["guard_events"] == 0:
            rows.append({**reference, "arm": NO_GUARD.label})
            continue
        row = {**common, **transcribed_row(payload, switches)}
        if switches == REFERENCE_ARM:
            reference = row
        rows.append(row)
    return rows


def fit_part(part: str, cap: int | None, jobs: int) -> pd.DataFrame:
    calls = [(part, scenario, n, k, seed) for scenario, n, k, seed in draws(part, cap)]
    fitted = shared.pool(fit_draw, calls, jobs)
    rows = pd.DataFrame([row for rows in fitted for row in rows], columns=list(ROW_COLUMNS))
    columns = [*ESTIMANDS, *(f"se_{name}" for name in ESTIMANDS)]
    return shared.require_finite(rows, columns)


# -------------------------------------------------------------------------- validation

#: The six values of a draw that V1, V2 and V4 compare: three estimates, three standard errors.
VALUES = (*ESTIMANDS, *(f"se_{name}" for name in ESTIMANDS))
DRAW = ["scenario", "replicate"]


def _long(rows: pd.DataFrame, arm: str) -> pd.DataFrame:
    """One arm's rows as ``(scenario, replicate, estimand, estimate, std_error)``."""
    selected = rows.loc[rows["arm"] == arm]
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "scenario": selected["scenario"].to_numpy(),
                    "replicate": selected["replicate"].to_numpy(),
                    "estimand": name,
                    "estimate": selected[name].to_numpy(),
                    "std_error": selected[f"se_{name}"].to_numpy(),
                }
            )
            for name in ESTIMANDS
        ],
        ignore_index=True,
    )


def _wide_reference(reference: pd.DataFrame) -> pd.DataFrame:
    """R's long rows as one row per draw, in the columns of :data:`VALUES`."""
    wide = reference.pivot(index=DRAW, columns="estimand", values=["estimate", "std_error"])
    wide.columns = [name if kind == "estimate" else f"se_{name}" for kind, name in wide.columns]
    return wide.reset_index()


def exit_status(transcribed: pd.Series, score_max: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The exit status of each draw, for ``T(0, 0, 0, 0)`` and for R.

    R's is ``tolIC`` when its largest absolute score mean is at most ``tolIC``, and ``maxIter``
    otherwise; the transcription's is its own recorded exit.
    """
    own = np.where(transcribed.to_numpy() == TOL_IC_EXIT, TOL_IC_EXIT, MAX_ITER_EXIT)
    reference = np.where(score_max.to_numpy(dtype=float) <= TOL_IC, TOL_IC_EXIT, MAX_ITER_EXIT)
    return own, reference


def _statuses(transcribed: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    status = transcribed[[*DRAW, "exit"]].merge(
        scores.drop_duplicates(DRAW), on=DRAW, how="outer", validate="1:1"
    )
    if status[["exit", "score_max"]].isna().any().any():
        raise RuntimeError("a draw appears on one side only")
    status["own"], status["reference"] = exit_status(status["exit"], status["score_max"])
    return status


def max_iter_drift(
    transcribed: pd.DataFrame, reference: pd.DataFrame, scores: pd.DataFrame
) -> tuple[int, float]:
    """F8: the count and the signed mean ``ate`` difference, ``T(0, 0, 0, 0)`` minus R, over the
    draws where both sides reach ``maxIter``.  Reported, and read by no rule."""
    status = _statuses(transcribed, scores)
    selected = status.loc[
        (status["own"] == MAX_ITER_EXIT) & (status["reference"] == MAX_ITER_EXIT), DRAW
    ]
    if selected.empty:
        return 0, float("nan")
    keys = [*DRAW, "estimand"]
    merged = (
        _long(transcribed, T0000)
        .merge(selected, on=DRAW)
        .merge(reference[[*keys, "estimate"]], on=keys, suffixes=("", "_r"))
    )
    ate = merged.loc[merged["estimand"] == "ate"]
    return len(ate), float(np.mean(ate["estimate"] - ate["estimate_r"]))


def conditions(scores: pd.DataFrame, r_rows: pd.DataFrame, twins: pd.DataFrame) -> pd.DataFrame:
    """One row per draw: R's ``score_max``, R's round count (rule 1) and the twin (rule 2).

    ``scores`` gives R's ``score_max`` of each draw, ``r_rows`` the rows of the counting run
    with their ``rounds``, and ``twins`` the twin rows.  A draw that lacks either raises.
    """
    rounds = r_rows[[*DRAW, "rounds"]].drop_duplicates()
    if rounds.duplicated(DRAW).any():
        raise RuntimeError("a draw carries two R round counts")
    twin = twins[[*DRAW, *VALUES, "iterations", "exit"]].rename(
        columns={column: f"twin_{column}" for column in (*VALUES, "iterations", "exit")}
    )
    out = (
        scores[[*DRAW, "score_max"]]
        .drop_duplicates(DRAW)
        .merge(rounds, on=DRAW, how="left", validate="1:1")
        .merge(twin, on=DRAW, how="left", validate="1:1")
    )
    if out[["rounds", "twin_iterations"]].isna().any().any():
        raise RuntimeError("a draw has no R round count or no twin")
    return out


def _largest(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """Per draw, the largest scaled difference of ``left`` from ``right`` over :data:`VALUES`."""
    return np.max(
        np.column_stack(
            [shared.scaled_difference(left[column], right[column]) for column in VALUES]
        ),
        axis=1,
    )


def classify(
    transcribed: pd.DataFrame, reference: pd.DataFrame, conditioned: pd.DataFrame
) -> pd.DataFrame:
    """Rules 2 to 5 on each draw: its class, ``abs(T - R)``, the twin's shift, and the verdict.

    ``transcribed`` holds the ``T(0, 0, 0, 0)`` rows, ``reference`` R's long rows and
    ``conditioned`` the :func:`conditions` of the same draws.  A draw is in the same round when
    its round count and exit status equal R's.  Such a draw at ``maxIter`` on both sides keeps
    the declared ``1e-4``, sensitive or not.  A sensitive ``tolIC`` draw passes within rule 3's
    ``1e-9`` or under rule 4; a draw in another round passes only under rule 5.  A draw that
    rule 3 fails and rule 4 or 5 passes is excused (rule 6).
    """
    status = _statuses(transcribed, conditioned)
    ours = transcribed[[*DRAW, *VALUES, "iterations"]]
    merged = status.merge(ours, on=DRAW, validate="1:1").merge(
        _wide_reference(reference), on=DRAW, suffixes=("", "_r"), validate="1:1"
    )
    gap = _largest(
        merged[list(VALUES)],
        merged[[f"{column}_r" for column in VALUES]].set_axis(list(VALUES), axis=1),
    )
    twin = merged[[f"twin_{column}" for column in VALUES]].set_axis(list(VALUES), axis=1)
    shift = _largest(twin, merged[list(VALUES)])
    moves = (merged["twin_iterations"] != merged["iterations"]) | (
        merged["twin_exit"] != merged["exit"]
    )
    sensitive = moves | (shift > SENSITIVE_SHIFT)
    same = (merged["iterations"] == merged["rounds"]) & (merged["own"] == merged["reference"])
    at_cap = merged["reference"] == MAX_ITER_EXIT
    kind = np.select([~same, at_cap, ~sensitive], [DIFFERS, SAME_MAX_ITER, SAME_TOL_IC], SENSITIVE)
    rule_three = same & (gap <= np.where(at_cap, MAX_ITER_TOLERANCE, shared.TOLERANCE))
    passes = np.select(
        [kind == DIFFERS, kind == SENSITIVE],
        [
            moves & (gap <= MAX_ITER_TOLERANCE),
            rule_three | ((gap <= SENSITIVE_TOLERANCE) & (shift >= TWIN_SHARE * gap)),
        ],
        rule_three,
    )
    return pd.DataFrame(
        {
            **{column: merged[column].to_numpy() for column in DRAW},
            "class": kind,
            "gap": gap,
            "shift": shift,
            "sensitive": sensitive.to_numpy(),
            "passes": passes.astype(bool),
            "excused": (passes & ~rule_three).astype(bool),
            "ate_difference": (merged["ate"] - merged["ate_r"]).to_numpy(),
        }
    )


def excused(
    transcribed: pd.DataFrame, reference: pd.DataFrame, conditioned: pd.DataFrame
) -> tuple[int, float]:
    """Rule 6: the count of excused draws and their signed mean ``ate`` difference, T minus R."""
    classes = classify(transcribed, reference, conditioned)
    selected = classes.loc[classes["excused"], "ate_difference"]
    return len(selected), float(selected.mean()) if len(selected) else float("nan")


def compare_by_exit(
    part: str,
    check: str,
    transcribed: pd.DataFrame,
    reference: pd.DataFrame,
    conditioned: pd.DataFrame,
) -> list[dict[str, Any]]:
    """V2 and V4: ``T(0, 0, 0, 0)`` against R by round and conditioning (rules 2 to 6).

    ``transcribed`` holds the ``T(0, 0, 0, 0)`` rows, ``reference`` R's long rows by
    ``(scenario, replicate, estimand)``, and ``conditioned`` the :func:`conditions` of the
    draws.  One validation row per class of :func:`classify`, each holding when every draw of
    its class passes; ``compared`` counts the class's draws and ``largest_difference`` is their
    largest ``abs(T - R)``.  The ``maxIter`` count, largest difference and signed mean ``ate``
    difference, and the count and signed mean of the excused draws, go to ``run.log``.
    """
    try:
        classes = classify(transcribed, reference, conditioned)
    except RuntimeError as error:
        raise RuntimeError(f"{check}: {error}") from None
    out = []
    for kind in (SAME_TOL_IC, SAME_MAX_ITER, SENSITIVE, DIFFERS):
        selected = classes.loc[classes["class"] == kind]
        largest = float(selected["gap"].max()) if len(selected) else 0.0
        held = bool(selected["passes"].all())
        out.append(shared.validation_row(part, f"{check} {kind}", (held, largest, len(selected))))
        if kind == SAME_MAX_ITER:
            count, drift = max_iter_drift(transcribed, reference, conditioned)
            shared.note(
                f"{check}: {len(selected)} maxIter draws, largest scaled difference {largest:.3g}, "
                f"signed mean ate difference {drift:.3g} over {count}"
            )
    chosen = classes.loc[classes["excused"], "ate_difference"]
    mean = float(chosen.mean()) if len(chosen) else float("nan")
    shared.note(f"{check}: {len(chosen)} excused draws, signed mean ate difference {mean:.3g}")
    return out


def _committed_reference(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The committed ``cleverly`` and R rows and R's ``score_max`` of the draws ``rows`` holds."""
    committed = shared.read_rows(BINARY.artifact("replicates.csv.gz"))
    fitted = rows[DRAW].drop_duplicates()
    committed = committed.merge(fitted, on=DRAW)
    diagnostics = shared.read_rows(BINARY.artifact("fit-diagnostics.csv"))
    diagnostics = diagnostics.loc[diagnostics["implementation"] == BINARY.reference].merge(
        fitted, on=DRAW
    )
    return (
        committed.loc[committed["implementation"] == BINARY.implementation],
        committed.loc[committed["implementation"] == BINARY.reference],
        diagnostics[[*DRAW, "score_max"]],
    )


def validate_committed(
    rows: pd.DataFrame, r_rows: pd.DataFrame, twins: pd.DataFrame
) -> pd.DataFrame:
    """V1, V2, V3 and V5 on the Part A rows, against the committed ``replicates.csv.gz``.

    V2 reads the counting rerun ``r_rows`` and the ``twins`` of step AV: rule 1 checks the
    rerun against the committed R rows, and rules 2 to 5 read its round counts and the twins.
    """
    ours, theirs, scores = _committed_reference(rows)
    keys = [*DRAW, "estimand"]
    out = [
        shared.validation_row(
            "A", "V1 cleverly rows", shared.compare_rows(_long(rows, C), ours, keys)
        )
    ]
    _, largest, compared = shared.compare_rows(r_rows, theirs, keys)
    out.append(
        shared.validation_row(
            "A", "V2 drtmle-r rerun", (largest <= RERUN_TOLERANCE, largest, compared)
        )
    )
    out += compare_by_exit(
        "A",
        "V2 drtmle-r",
        rows.loc[rows["arm"] == T0000],
        theirs,
        conditions(scores, r_rows, twins),
    )
    initial = _long(rows, C).drop(columns=["estimate", "std_error"])
    initial["initial"] = np.concatenate(
        [rows.loc[rows["arm"] == C, f"initial_{name}"].to_numpy() for name in ESTIMANDS]
    )
    merged = initial.merge(theirs[[*keys, "initial_estimate"]], on=keys, validate="1:1")
    largest = float(np.max(shared.scaled_difference(merged["initial"], merged["initial_estimate"])))
    out.append(
        shared.validation_row("A", "V3 initial estimates", (largest <= 1e-12, largest, len(merged)))
    )
    out.append(
        shared.validation_row(
            "A",
            "V5 T1111 against C",
            shared.compare_rows(_long(rows, T1111), _long(rows, C), keys, columns=("estimate",)),
        )
    )
    return shared.validation_frame(out)


def _record(name: str, output: Path, smoke: bool) -> Path:
    return shared.record_path(HERE, name, output, smoke)


def _v4_exit_status(path: Path, part: str) -> pd.DataFrame:
    """The V4 row of rule 5 in the V4 record at ``path``, relabelled for ``part``, if any.

    Its name ends in ``exit status``: a draw in another round or exit, without a twin that
    moves, stops the design.
    """
    checks = shared.optional_rows(path)
    if checks.empty:
        return shared.validation_frame([])
    checks = checks.loc[checks["check"].astype(str).str.endswith("exit status")]
    return checks.assign(part=part)


def prior_checks(part: str, output: Path, smoke: bool) -> pd.DataFrame:
    """The earlier checks a later part carries into its record; a missing record carries none.

    B and C carry every Part A check.  C also carries the V4 check of rule 5, because a miss
    there stops the design.
    """
    frames = [shared.validation_frame([])]
    if part in ("B", "C"):
        checks = shared.optional_rows(
            _record(shared.part_paths(output, "A")[1].name, output, smoke)
        )
        if not checks.empty:
            frames.append(checks.assign(part=part, check="Part A " + checks["check"].astype(str)))
    if part == "C":
        frames.append(
            _v4_exit_status(_record(shared.part_paths(output, "V4")[1].name, output, smoke), part)
        )
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------- rules 1 and 2: AV and V4


def twin_path(output: Path, part: str) -> Path:
    """The twin rows of step AV or of V4."""
    return output / f"{part.lower()}-twin-rows.csv.gz"


def condition_draw(
    call: tuple[str, str, int, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """One draw rebuilt: the payload for R, its truth row, its ``C`` row and its twin row."""
    part, scenario, n, k, seed = call
    payload, truth, result = draw_payload(scenario, n, seed)
    sample = payload.copy()
    sample.insert(0, "replicate", k)
    sample.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": k,
        **{f"truth_{name}": value for name, value in study.truth().items()},
    }
    twin = {
        **_base(part, scenario, n, k, seed, truth),
        **_initial(payload),
        **transcribed_row(payload, REFERENCE_ARM, solver="qr"),
    }
    return sample, truth_row, {"scenario": scenario, "replicate": k, **cleverly_row(result)}, twin


def count_rounds(
    built: list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]],
    output: Path,
) -> pd.DataFrame:
    """Rule 1: the counting wrapper on the rebuilt payloads, in a scratch directory.

    Returns the runner's rows with each draw's ``rounds``.  The wrapper stops on a lost worker
    or a draw without one round count, and ``Reference.run`` then raises.
    """
    samples = pd.concat([sample for sample, *_ in built], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, *_ in built])
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        directory = Path(scratch)
        write_csv(samples, directory / "samples.csv.gz", compression="gzip")
        write_csv(truths, directory / "truth.csv")
        result = directory / "reference-results.csv"
        COUNTING.run(
            REFERENCE_HERE,
            directory / "samples.csv.gz",
            directory / "truth.csv",
            result,
            cores=R_WORKERS,
        )
        rows = shared.read_rows(result)
        rounds = shared.read_rows(directory / "reference-results-rounds.csv")
    merged = rows.merge(rounds, on=DRAW, how="left", validate="m:1")
    if merged["rounds"].isna().any() or len(rounds) != len(rows[DRAW].drop_duplicates()):
        raise RuntimeError("the counting run returned rows and round counts for different draws")
    return merged


def r_phase(
    calls: list[tuple[str, str, int, int, int]], output: Path, jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rules 1 and 2 on the draws ``calls``: R's counted rows, the ``C`` refits, the twins.

    One scenario at a time, so that one R input holds at most one scenario's draws.
    """
    r_frames, refits, twins = [], [], []
    for scenario in dict.fromkeys(call[1] for call in calls):
        built = shared.pool(condition_draw, [call for call in calls if call[1] == scenario], jobs)
        r_frames.append(count_rounds(built, output))
        refits += [refit for _, _, refit, _ in built]
        twins += [twin for *_, twin in built]
    twin_rows = shared.require_finite(pd.DataFrame(twins, columns=list(ROW_COLUMNS)), VALUES)
    return pd.concat(r_frames, ignore_index=True), pd.DataFrame(refits), twin_rows


def _payload_check(part: str, refit: pd.DataFrame, rows: pd.DataFrame) -> dict[str, Any]:
    """The ``C`` refit that rebuilds each payload must equal its recorded ``C`` row (V1)."""
    fitted = rows.loc[rows["replicate"].isin(refit["replicate"])]
    return shared.validation_row(
        part,
        f"{part} payload C refit",
        shared.compare_rows(_long(refit, C), _long(fitted, C), [*DRAW, "estimand"]),
    )


def condition_part_a(rows: pd.DataFrame, output: Path, jobs: int) -> None:
    """Step AV: rules 1 and 2 on the draws of the Part A rows, then Part A's validation.

    Writes ``av-rows.csv.gz`` (the counting rerun), ``av-twin-rows.csv.gz``,
    ``av-validation.csv`` (the payload check) and ``a-validation.csv``.
    """
    rows_path, validation_path, reading_path = shared.part_paths(output, "AV")
    selected = rows.loc[rows["arm"] == C]
    calls = [
        ("AV", str(row.scenario), int(row.n), int(row.replicate), int(row.seed))
        for row in selected.itertuples()
    ]
    r_rows, refit, twins = r_phase(calls, output, jobs)
    shared.write_table(r_rows, rows_path)
    shared.write_table(twins, twin_path(output, "AV"))
    shared.write_table(
        shared.validation_frame([_payload_check("AV", refit, rows)]), validation_path
    )
    shared.write_table(validate_committed(rows, r_rows, twins), shared.part_paths(output, "A")[1])
    shared.write_table(table("AV", output, jobs), reading_path)


def image_id() -> str:
    """The local ID of the pinned R image, for ``run.log``."""
    done = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", REFERENCE.image],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() or f"unknown ({done.stderr.strip()})"


def _b_rows(output: Path, smoke: bool) -> pd.DataFrame:
    """The Part B rows a later part reads: the pushed record for a declared run (F5)."""
    path = _record(shared.part_paths(output, "B")[0].name, output, smoke)
    if not path.exists():
        raise RuntimeError(f"{path} does not exist; Part B must run first")
    return shared.read_rows(path)


def run_v4(output: Path, cap: int | None, jobs: int) -> None:
    rows_path, validation_path, _ = shared.part_paths(output, "V4")
    b_rows = _b_rows(output, cap is not None)
    count = shared.budget(V4_DRAWS, cap)
    selected = b_rows.loc[(b_rows["arm"] == C) & (b_rows["replicate"] < count)]
    calls = [
        ("V4", SCENARIO, int(row.n), int(row.replicate), int(row.seed))
        for row in selected.itertuples()
    ]
    r_rows, refit, twins = r_phase(calls, output, jobs)
    shared.note(f"V4 image {REFERENCE.image}: {image_id()}")
    shared.write_table(r_rows, rows_path)
    shared.write_table(twins, twin_path(output, "V4"))
    fresh = b_rows.loc[b_rows["replicate"].isin(refit["replicate"])]
    validation = shared.validation_frame(
        [
            _payload_check("V4", refit, fresh),
            *compare_by_exit(
                "V4",
                "V4 R",
                fresh.loc[fresh["arm"] == T0000],
                r_rows,
                conditions(r_rows, r_rows, twins),
            ),
        ]
    )
    shared.write_table(validation, validation_path)


def _conditioned(output: Path, part: str, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """The :func:`conditions` that step AV or V4 recorded in ``output``."""
    r_rows = shared.read_rows(shared.part_paths(output, part)[0])
    twins = shared.read_rows(twin_path(output, part))
    return conditions(r_rows if scores is None else scores, r_rows, twins)


# ------------------------------------------------------------------------------ readings


def budget_rule() -> int:
    """``ceil((z * SD / resolution)^2)``: 1,954, which the declaration rounds up to 2,000."""
    return int(np.ceil((Z * COMMITTED_SD / RESOLUTION) ** 2))


def primary_label(interval: Interval) -> str:
    """The primary reading of the interval of the mean of ``C - T(0, 0, 0, 0)``."""
    if interval.low > 0.0:
        return INCREMENT
    if interval.high < 0.0:
        return REVERSE
    if interval.width / 2.0 <= RESOLUTION:
        return NO_INCREMENT
    return UNRESOLVED


def side(interval: Interval, point: float) -> float:
    """The side of the primary difference: its interval's, or its point estimate's sign."""
    if interval.low > 0.0:
        return 1.0
    if interval.high < 0.0:
        return -1.0
    return float(np.sign(point))


def excludes_on(interval: Interval, direction: float) -> bool:
    """Whether a main effect lies wholly on the side of the primary difference."""
    if direction > 0:
        return interval.low > 0.0
    if direction < 0:
        return interval.high < 0.0
    return False


def bracket_holds(interval: Interval) -> bool:
    return interval.within(-BRACKET, BRACKET)


def localization_label(
    primary: str,
    primary_interval: Interval,
    primary_point: float,
    bracket: Interval,
    effects: dict[str, Interval],
) -> str:
    """The localization reading, read from the top of the declared table."""
    if not bracket_holds(bracket):
        return UNEXPLAINED
    if primary == NO_INCREMENT:
        return NOTHING
    direction = side(primary_interval, primary_point)
    named = [name for name, interval in effects.items() if excludes_on(interval, direction)]
    if len(named) == 1:
        return f"attributed to {named[0]}"
    if len(named) > 1:
        return "shared by " + (", ".join(named[:-1]) + f" and {named[-1]}")
    if primary == INCREMENT:
        return INTERACTION_ONLY
    return NO_LABEL


def _wide(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.pivot(index="replicate", columns="arm", values="ate")


def _level_arms(attribute: str, level: bool) -> list[str]:
    return [arm.label for arm in FACTORIAL if getattr(arm, attribute) == level]


def main_effects(wide: pd.DataFrame, guard: bool) -> dict[str, np.ndarray]:
    """The per-draw main effect of each factor, and of G when a guard event occurred."""
    out = {
        name: (
            wide[_level_arms(attribute, True)].mean(axis=1)
            - wide[_level_arms(attribute, False)].mean(axis=1)
        ).to_numpy()
        for name, attribute in FACTORS.items()
    }
    if guard:
        out["G"] = (wide[T0G] - wide[T0000]).to_numpy()
    return out


def interaction(wide: pd.DataFrame, pair: tuple[str, str]) -> np.ndarray:
    """The 2^4 two-way contrast: the arms whose two levels agree minus those whose differ."""
    first, second = (FACTORS[name] for name in pair)
    agree = [arm.label for arm in FACTORIAL if getattr(arm, first) == getattr(arm, second)]
    differ = [arm.label for arm in FACTORIAL if getattr(arm, first) != getattr(arm, second)]
    return (wide[agree].mean(axis=1) - wide[differ].mean(axis=1)).to_numpy()


def _statistic(
    part: str, scope: str, name: str, values: np.ndarray, level: float, result: str = ""
) -> dict[str, Any]:
    return shared.reading(
        part,
        scope,
        name,
        value=float(np.mean(values)),
        interval=student_interval(np.asarray(values, dtype=float), confidence_level=level),
        result=result,
    )


def guard_fired(rows: pd.DataFrame) -> bool:
    """Whether ``T(0, 0, 0, 0)`` records a guard event on any draw of ``rows`` (the G family)."""
    return bool(rows.loc[rows["arm"] == T0000, "guard_events"].sum() > 0)


def factorial_table(
    part: str, rows: pd.DataFrame, declared: int, *, guard: bool, prefix: str = ""
) -> list[dict[str, Any]]:
    """The primary, the family, the bracket, the interactions and the localization of one part.

    ``guard`` says whether G joins the family; the caller reads it off every row of the part.
    """
    scope = f"{SCENARIO}, n = {int(rows['n'].iloc[0])}"
    smoke = bool(rows.groupby("arm").size().ne(declared).any())
    wide = _wide(rows)

    def labelled(text: str) -> str:
        return shared.label(prefix + text, smoke)

    primary_values = (wide[C] - wide[T0000]).to_numpy()
    primary_interval = student_interval(primary_values, confidence_level=shared.CONFIDENCE)
    primary = primary_label(primary_interval)
    effects = main_effects(wide, guard)
    level = 1.0 - (1.0 - shared.CONFIDENCE) / len(effects)
    intervals = {
        name: student_interval(values, confidence_level=level) for name, values in effects.items()
    }
    bracket_values = (wide[T1111] - wide[C]).to_numpy()
    bracket = student_interval(bracket_values, confidence_level=shared.CONFIDENCE)
    out = [
        shared.reading(
            part,
            scope,
            f"C - {T0000}",
            value=float(np.mean(primary_values)),
            interval=primary_interval,
            result=labelled(primary),
        ),
        shared.reading(part, scope, f"C - {T0000} half-width", value=primary_interval.width / 2.0),
        shared.reading(
            part,
            scope,
            f"guard events of {T0000}",
            value=float(rows.loc[rows["arm"] == T0000, "guard_events"].sum()),
            result="G in the family" if guard else "inert on these draws",
        ),
    ]
    out += [
        _statistic(part, scope, f"main effect {name}", values, level, f"level {level:.6g}")
        for name, values in effects.items()
    ]
    out.append(
        shared.reading(
            part,
            scope,
            f"{T1111} - C",
            value=float(np.mean(bracket_values)),
            interval=bracket,
            result=labelled("bracket " + (HOLDS if bracket_holds(bracket) else FAILS)),
        )
    )
    out += [
        _statistic(
            part,
            scope,
            f"interaction {a}{b}",
            interaction(wide, (a, b)),
            shared.CONFIDENCE,
            SUPPLEMENTARY,
        )
        for a, b in INTERACTIONS
    ]
    label = localization_label(
        primary, primary_interval, float(np.mean(primary_values)), bracket, intervals
    )
    out.append(shared.reading(part, scope, "localization", result=labelled(label)))
    return out


def decomposition_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """K5, supplementary: each part of K alone added to ``T(1, 1, 1, 0)``, and K whole."""
    wide = _wide(rows)
    scope = f"{SCENARIO}, n = {int(rows['n'].iloc[0])}"
    arms_by_name = {arm.part: arm.label for arm in DECOMPOSITION}
    out = [
        _statistic(
            "A",
            scope,
            f"{arms_by_name[part]} - {T1110}",
            (wide[arms_by_name[part]] - wide[T1110]).to_numpy(),
            shared.CONFIDENCE,
            SUPPLEMENTARY,
        )
        for part in K_PARTS
    ]
    out.append(
        _statistic(
            "A",
            scope,
            f"{T1111} - {T1110}",
            (wide[T1111] - wide[T1110]).to_numpy(),
            shared.CONFIDENCE,
            SUPPLEMENTARY,
        )
    )
    return out


def arm_summaries(part: str, rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Each arm's bias against the truth, and its instruments: read by no rule."""
    out = []
    for (scenario, n, arm), group in rows.groupby(["scenario", "n", "arm"], sort=False):
        scope = f"{scenario}, n = {n}, {arm}"
        error = (group["ate"] - group["truth"]).to_numpy()
        exits = ", ".join(
            f"{name} {count}" for name, count in group["exit"].value_counts().sort_index().items()
        )
        out += [
            _statistic(part, scope, "bias", error, shared.CONFIDENCE, SUPPLEMENTARY),
            shared.reading(
                part,
                scope,
                "mean iterations",
                value=float(group["iterations"].mean()),
                result=exits,
            ),
            shared.reading(part, scope, "mean closing steps", value=float(group["closing"].mean())),
            shared.reading(part, scope, "guard events", value=float(group["guard_events"].sum())),
            shared.reading(
                part, scope, "rows at a mechanism bound", value=float(group["at_bound"].sum())
            ),
            shared.reading(part, scope, "largest score", value=float(group["score_max"].max())),
        ]
    return out


def control_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """P-ctrl on each control scenario, and its bracket, which no rule reads."""
    out = []
    for scenario in CONTROLS:
        group = rows.loc[rows["scenario"] == scenario]
        wide = _wide(group)
        smoke = bool(group.groupby("arm").size().ne(BINARY.replicates).any())
        difference = (wide[T1000] - wide[T0000]).to_numpy()
        interval = student_interval(difference, confidence_level=shared.CONFIDENCE)
        spread = float(np.std(difference, ddof=1))
        held = interval.contains(0.0) and spread <= CONTROL_SD[scenario]
        out += [
            shared.reading(
                "A",
                scenario,
                f"{T1000} - {T0000}",
                value=float(np.mean(difference)),
                interval=interval,
                result=shared.label("P-ctrl " + (HOLDS if held else FAILS), smoke),
            ),
            shared.reading(
                "A",
                scenario,
                f"{T1000} - {T0000} SD",
                value=spread,
                result=f"bound {CONTROL_SD[scenario]}",
            ),
            _statistic(
                "A",
                scenario,
                f"{T1111} - C",
                (wide[T1111] - wide[C]).to_numpy(),
                shared.CONFIDENCE,
                SUPPLEMENTARY,
            ),
        ]
    return out


def scaling_table(rows: pd.DataFrame, b_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """``sqrt(n)`` times the mean of ``C - T(0, 0, 0, 0)`` at each size, Bonferroni over three."""
    level = 1.0 - (1.0 - shared.CONFIDENCE) / 3.0
    combined = pd.concat([b_rows, rows], ignore_index=True)
    out = []
    for n, group in combined.groupby("n"):
        wide = _wide(group.loc[group["arm"].isin([C, T0000])])
        values = np.sqrt(float(n)) * (wide[C] - wide[T0000]).to_numpy()
        interval = student_interval(values, confidence_level=level)
        label = "increment" if interval.low > 0 else REVERSE if interval.high < 0 else ""
        out.append(
            shared.reading(
                "C",
                f"{SCENARIO}, n = {n}",
                f"sqrt(n) (C - {T0000})",
                value=float(np.mean(values)),
                interval=interval,
                result=shared.label(label, len(wide) != REPLICATES),
            )
        )
    return out


def context_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """The supplementary Part C rows: BD-1's bias at 1,500 and the committed n = 6,000 rung's.

    The ``C`` bias beside each comes from :func:`arm_summaries`.
    """
    bd1 = shared.read_rows(ROOT / "tests/diagnostics/rm18_boundary/bd-1-reading.csv")
    bd1_bias = bd1.loc[
        bd1["scope"].str.startswith(BINARY.slug) & (bd1["statistic"] == "bias")
    ].iloc[0]
    rung = shared.read_rows(BINARY.artifact("properties.csv")).set_index(["property", "cell"])
    rung_row = rung.loc[("double_robust_contraction", "treatment_correct_n6000")]
    references = {
        1_500: ("BD-1 bias", bd1_bias["value"], bd1_bias["ci_lower"], bd1_bias["ci_upper"]),
        6_000: (
            "committed rung bias",
            rung_row["bias"],
            rung_row["bias_ci_lower"],
            rung_row["bias_ci_upper"],
        ),
    }
    return [
        shared.reading(
            "C",
            f"{SCENARIO}, n = {n}",
            name,
            value=float(value),
            interval=Interval(float(low), float(high)),
            result=SUPPLEMENTARY,
        )
        for n, (name, value, low, high) in references.items()
        if (rows["n"] == n).any()
    ]


def excused_reading(part: str, count: int, mean: float) -> dict[str, Any]:
    """Rule 6: the signed mean ``ate`` difference over the excused draws, reported."""
    return shared.reading(
        part,
        f"{count} excused draws",
        f"{T0000} - R, signed mean ate difference",
        value=mean,
        result=SUPPLEMENTARY,
    )


def drift_reading(part: str, count: int, drift: float) -> dict[str, Any]:
    """F8: the signed mean ``ate`` difference over the ``maxIter`` draws, reported."""
    return shared.reading(
        part,
        f"{count} maxIter draws",
        f"{T0000} - R, signed mean ate difference",
        value=drift,
        result=SUPPLEMENTARY,
    )


# ---------------------------------------------------------------------------- the parts


def run_part(part: str, output: Path, cap: int | None, jobs: int) -> None:
    rows_path, validation_path, _ = shared.part_paths(output, part)
    smoke = cap is not None
    if part == "V4":
        run_v4(output, cap, jobs)
        return
    if part == "A":
        rows = fit_part(part, cap, jobs)
        shared.write_table(rows, rows_path)
        condition_part_a(rows, output, jobs)
        return
    if part == "AV":
        # W3: the committed Part A rows, not a refit of any arm.
        rows = shared.read_rows(_record(shared.part_paths(output, "A")[0].name, output, smoke))
        if smoke:
            rows = rows.loc[rows["replicate"] < shared.budget(BINARY.replicates, cap)]
        condition_part_a(rows, output, jobs)
        return
    validation = prior_checks(part, output, smoke)
    if part == "C":
        # The scaling reads n = 3,000 from the Part B rows, so they must exist (F5).
        _b_rows(output, smoke)
    shared.write_table(validation, validation_path)
    if shared.validated(validation, part):
        shared.write_table(fit_part(part, cap, jobs), rows_path)


def _smoke_rows(rows: pd.DataFrame, declared: int) -> bool:
    return bool(rows.groupby(["n", "arm"]).size().ne(declared).any())


def table(part: str, output: Path, jobs: int) -> pd.DataFrame:
    del jobs
    rows_path, validation_path, _ = shared.part_paths(output, part)
    validation = shared.read_rows(validation_path)
    v4_path = shared.part_paths(output, "V4")[1]
    v4_ran = True
    if part == "B":
        # V4 is a precondition of the Part B reading: a V4 that has not run leaves none.
        v4 = shared.optional_rows(v4_path)
        v4_ran = not v4.empty
        validation = pd.concat([validation, v4.assign(part="B")], ignore_index=True)
    elif part == "A":
        # F3: a V4 exit-status mismatch stops the design, so it stops the Part A reading too.
        validation = pd.concat([validation, _v4_exit_status(v4_path, "A")], ignore_index=True)
    out = shared.validation_readings(validation, part)
    if part == "AV":
        return shared.reading_frame(out)
    if part == "V4":
        r_rows = shared.read_rows(rows_path)
        b_rows = _b_rows(output, r_rows["replicate"].nunique() != V4_DRAWS)
        fresh = b_rows.loc[(b_rows["arm"] == T0000) & b_rows["replicate"].isin(r_rows["replicate"])]
        conditioned = _conditioned(output, part)
        out.append(drift_reading(part, *max_iter_drift(fresh, r_rows, conditioned)))
        out.append(excused_reading(part, *excused(fresh, r_rows, conditioned)))
        return shared.reading_frame(out)
    if not (v4_ran and shared.validated(validation, part)):
        out.append(shared.reading(part, "", "reading", result=shared.NOT_VALIDATED))
        return shared.reading_frame(out)
    rows = shared.read_rows(rows_path)
    if part == "A":
        _, theirs, scores = _committed_reference(rows)
        t0000 = rows.loc[rows["arm"] == T0000]
        out.append(drift_reading(part, *max_iter_drift(t0000, theirs, scores)))
        out.append(
            excused_reading(part, *excused(t0000, theirs, _conditioned(output, "AV", scores)))
        )
        focus = rows.loc[rows["scenario"] == SCENARIO]
        out += factorial_table(
            part, focus, BINARY.replicates, guard=guard_fired(rows), prefix=f"{COMMITTED}: "
        )
        out += decomposition_table(focus)
        out += control_table(rows)
    elif part == "B":
        out += factorial_table(part, rows, REPLICATES, guard=guard_fired(rows))
    else:
        out += scaling_table(rows, _b_rows(output, _smoke_rows(rows, REPLICATES)))
        out += context_table(rows)
    out += arm_summaries(part, rows)
    return shared.reading_frame(out)


def pushed(part: str) -> tuple[str, ...]:
    """The records a declared part reads from the pushed upstream."""
    directory = HERE.relative_to(ROOT).as_posix()
    if part == "AV":
        return (f"{directory}/a-rows.csv.gz",)
    if part == "B":
        return (f"{directory}/a-validation.csv",)
    if part == "C":
        return (
            f"{directory}/a-validation.csv",
            f"{directory}/b-rows.csv.gz",
            f"{directory}/v4-validation.csv",
        )
    if part == "V4":
        return (f"{directory}/b-rows.csv.gz",)
    return ()


if __name__ == "__main__":
    shared.main(__doc__.splitlines()[0], PARTS, HERE, run_part, table, pushed)
