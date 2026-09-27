"""RM19: localize the one-sided DR-TMLE increment of ``cleverly`` over R ``drtmle``.

RM19 of ``docs/roadmap.md`` declares the design in "The localization design, declared before it
runs".  Each part runs on its own, in this order:

* A refits the registered primary draws of the three scenarios, validates the harness (V1 to
  V3) and attributes the committed signal;
* B fits 2,000 fresh ``treatment_correct`` draws at n = 3,000 with ``C`` and every arm of the
  transcription;
* V4 runs the pinned R container on the first 200 Part B draws;
* C fits 2,000 fresh draws at n = 1,500 and 6,000 with ``C`` and ``T(0, 0, 0)``.

    python -m tests.diagnostics.rm19_one_sided_increment.run --part A --output <scratch>

``--read-only`` rebuilds the part's reading from what ``--output`` holds.
"""

from __future__ import annotations

import functools
import itertools
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm19_one_sided_increment.rtrans import TOL_IC, Switches, transcribe
from tests.studies import canonical_drtmle as study
from tests.studies.canonical_drtmle import STUDY as BINARY
from tests.studies.evidence.inference import Interval, student_interval
from tests.studies.evidence.manifest import ROOT, write_csv
from tests.studies.evidence.seeds import replicate_seed

HERE = Path(__file__).resolve().parent
DESIGN = "one-sided-increment"
PARTS = ("A", "B", "V4", "C")
SCENARIO = "treatment_correct"
CONTROLS = ("outcome_correct", "both_correct")
ESTIMANDS = study.ESTIMANDS

#: The declared budget of each fresh part and its sizes, in the seed assignment order.
REPLICATES = 2_000
SIZES = {"B": (study.PRIMARY_N,), "C": (1_500, 6_000)}
#: V4: the first 200 Part B draws, at most seven R workers (``tests/canonical/drtmle/README``).
V4_DRAWS = 200
R_WORKERS = 7

#: The budget rule's inputs: the registered similarity margin of the ``ate`` row, the committed
#: paired SD and the 99% normal quantile.  The resolution is one fifth of the margin.
MARGIN = 0.002990
COMMITTED_SD = 0.010261
Z = 2.575829
RESOLUTION = 0.000598
#: The bracket: ``T(1, 1, 1) - C`` must lie inside this, one tenth of the committed increment.
BRACKET = 1e-4
#: V2 and V4 on a draw where R reaches ``maxIter``: the amended tolerance.  A draw that exits at
#: ``tolIC`` keeps the R4 tolerance of ``rm18_shared``.
MAX_ITER_TOLERANCE = 1e-4
TOL_IC_EXIT, MAX_ITER_EXIT = "tolIC", "maxIter"
#: P-ctrl: the committed paired SD of each control scenario.
CONTROL_SD = {"outcome_correct": 0.002223, "both_correct": 0.000869}
#: The seed that ``random_partition`` cannot take as ``seed + 1``.
LAST_SEED = 2**32 - 1

#: The factorial arms, in ``(J, P, S)`` order, and the guard arm.
FACTORIAL = tuple(
    Switches(joint_tilt=bool(j), prime=bool(p), cleverly_exit=bool(s))
    for j, p, s in itertools.product((0, 1), repeat=3)
)
REFERENCE_ARM = Switches()
NO_GUARD = Switches(no_guard=True)
CONTROL_ARMS = (REFERENCE_ARM, Switches(joint_tilt=True), Switches(True, True, True))
FACTORS = {"J": "joint_tilt", "P": "prime", "S": "cleverly_exit"}
INTERACTIONS = (("J", "P"), ("J", "S"), ("P", "S"))
C = "C"
T000 = REFERENCE_ARM.label
T111 = Switches(True, True, True).label
T0G = NO_GUARD.label

INCREMENT = "increment confirmed"
REVERSE = "reverse increment"
NO_INCREMENT = "no increment at the declared resolution"
UNRESOLVED = "unresolved"
UNEXPLAINED = "unexplained"
NOTHING = "nothing to localize"
INTERACTION_ONLY = "interaction only"
NO_LABEL = "no localization label"
COMMITTED = "committed-draw attribution"
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
def fresh_seeds() -> dict[str, tuple[int, ...]]:
    """The declared seeds of Parts B and C, assigned in the declared order after BD-1."""
    taken = taken_seeds()
    return {part: tuple(shared.fresh_seeds(BINARY, labels(part), taken)) for part in ("B", "C")}


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
    """The transcription arms of one part and scenario, ``T0+G`` last."""
    if part == "C":
        return (REFERENCE_ARM,)
    if scenario == SCENARIO:
        return (*FACTORIAL, NO_GUARD)
    return (*CONTROL_ARMS, NO_GUARD)


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


def transcribed_row(payload: pd.DataFrame, switches: Switches) -> dict[str, Any]:
    out = transcribe(payload, switches)
    return {
        "arm": switches.label,
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
    """Every arm of one draw.  ``T0+G`` is a copy of ``T(0, 0, 0)`` on a draw with no guard event,
    because the guards then change no step."""
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


def exit_status(transcribed: pd.Series, score_max: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The exit status of each draw, for ``T(0, 0, 0)`` and for R.

    R's is ``tolIC`` when its largest absolute score mean is at most ``tolIC``, and ``maxIter``
    otherwise; the transcription's is its own recorded exit.
    """
    own = np.where(transcribed.to_numpy() == TOL_IC_EXIT, TOL_IC_EXIT, MAX_ITER_EXIT)
    reference = np.where(score_max.to_numpy(dtype=float) <= TOL_IC, TOL_IC_EXIT, MAX_ITER_EXIT)
    return own, reference


def compare_by_exit(
    part: str,
    check: str,
    transcribed: pd.DataFrame,
    reference: pd.DataFrame,
    scores: pd.DataFrame,
) -> list[dict[str, Any]]:
    """V2 and V4: ``T(0, 0, 0)`` against R, with the tolerance of each draw's exit status.

    ``transcribed`` holds the ``T(0, 0, 0)`` rows, ``reference`` R's long rows by
    ``(scenario, replicate, estimand)``, and ``scores`` R's ``score_max`` by draw.  Every draw
    must have the same exit status on both sides.  A ``tolIC`` draw then meets the R4 tolerance,
    and a ``maxIter`` draw :data:`MAX_ITER_TOLERANCE`.  The ``maxIter`` count and largest
    difference also go to ``run.log``.
    """
    draw = ["scenario", "replicate"]
    status = transcribed[[*draw, "exit"]].merge(
        scores.drop_duplicates(draw), on=draw, how="outer", validate="1:1"
    )
    if status[["exit", "score_max"]].isna().any().any():
        raise RuntimeError(f"{check}: a draw appears on one side only")
    status["own"], status["reference"] = exit_status(status["exit"], status["score_max"])
    matched = status["own"] == status["reference"]
    out = [
        shared.validation_row(
            part,
            f"{check} exit status",
            (bool(matched.all()), float(np.sum(~matched)), len(status)),
        )
    ]
    keys = [*draw, "estimand"]
    for exit_name, tolerance in (
        (TOL_IC_EXIT, shared.TOLERANCE),
        (MAX_ITER_EXIT, MAX_ITER_TOLERANCE),
    ):
        selected = status.loc[matched & (status["reference"] == exit_name), draw]
        largest = 0.0
        if len(selected):
            _, largest, _ = shared.compare_rows(
                _long(transcribed, T000).merge(selected, on=draw),
                reference.merge(selected, on=draw),
                keys,
            )
        out.append(
            shared.validation_row(
                part, f"{check} {exit_name} draws", (largest <= tolerance, largest, len(selected))
            )
        )
        if exit_name == MAX_ITER_EXIT:
            shared.note(
                f"{check}: {len(selected)} maxIter draws, largest scaled difference {largest:.3g}"
            )
    return out


def validate_committed(rows: pd.DataFrame) -> pd.DataFrame:
    """V1, V2 and V3 on the Part A rows, against the committed ``replicates.csv.gz``."""
    committed = shared.read_rows(BINARY.artifact("replicates.csv.gz"))
    keys = ["scenario", "replicate", "estimand"]
    fitted = rows[["scenario", "replicate"]].drop_duplicates()
    committed = committed.merge(fitted, on=["scenario", "replicate"])
    selection = committed.loc[committed["implementation"] == BINARY.implementation]
    out = [
        shared.validation_row(
            "A", "V1 cleverly rows", shared.compare_rows(_long(rows, C), selection, keys)
        )
    ]
    diagnostics = shared.read_rows(BINARY.artifact("fit-diagnostics.csv"))
    diagnostics = diagnostics.loc[diagnostics["implementation"] == BINARY.reference].merge(
        fitted, on=["scenario", "replicate"]
    )
    out += compare_by_exit(
        "A",
        "V2 drtmle-r",
        rows.loc[rows["arm"] == T000],
        committed.loc[committed["implementation"] == BINARY.reference],
        diagnostics[["scenario", "replicate", "score_max"]],
    )
    initial = _long(rows, C).drop(columns=["estimate", "std_error"])
    initial["initial"] = np.concatenate(
        [rows.loc[rows["arm"] == C, f"initial_{name}"].to_numpy() for name in ESTIMANDS]
    )
    reference = committed.loc[committed["implementation"] == BINARY.reference]
    merged = initial.merge(reference[[*keys, "initial_estimate"]], on=keys, validate="1:1")
    largest = float(np.max(shared.scaled_difference(merged["initial"], merged["initial_estimate"])))
    out.append(
        shared.validation_row("A", "V3 initial estimates", (largest <= 1e-12, largest, len(merged)))
    )
    return shared.validation_frame(out)


def record_path(name: str, output: Path, smoke: bool) -> Path:
    """A precondition file: the pushed record for a declared run, ``output`` for a smoke run."""
    return (output if smoke else HERE) / name


def prior_checks(part: str, output: Path, smoke: bool) -> pd.DataFrame:
    """The earlier checks a later part carries into its record; a missing record carries none.

    B and C carry every Part A check.  C also carries the V4 exit-status check, because an
    exit-status mismatch in V4 stops the design.
    """
    frames = []
    for source, keep in (("A", None), ("V4", "exit status")):
        if source == "V4" and part != "C":
            continue
        path = record_path(shared.part_paths(output, source)[1].name, output, smoke)
        checks = shared.optional_rows(path)
        if checks.empty:
            continue
        if keep is not None:
            checks = checks.loc[checks["check"].astype(str).str.endswith(keep)]
        prefix = "Part A " if source == "A" else ""
        frames.append(checks.assign(part=part, check=prefix + checks["check"].astype(str)))
    if not frames:
        return shared.validation_frame([])
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------------- V4


def v4_payloads(b_rows: pd.DataFrame, cap: int | None, jobs: int) -> tuple[pd.DataFrame, ...]:
    """Rebuild the payloads of the first Part B draws, with each draw's ``C`` row."""
    count = shared.budget(V4_DRAWS, cap)
    selected = b_rows.loc[(b_rows["arm"] == C) & (b_rows["replicate"] < count)]
    calls = [(int(row.n), int(row.replicate), int(row.seed)) for row in selected.itertuples()]
    return tuple(shared.pool(_v4_draw, calls, jobs))


def _v4_draw(call: tuple[int, int, int]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    n, k, seed = call
    payload, _, result = draw_payload(SCENARIO, n, seed)
    sample = payload.copy()
    sample.insert(0, "replicate", k)
    sample.insert(0, "scenario", SCENARIO)
    truth_row = {
        "scenario": SCENARIO,
        "replicate": k,
        **{f"truth_{name}": value for name, value in study.truth().items()},
    }
    return sample, truth_row, {"scenario": SCENARIO, "replicate": k, **cleverly_row(result)}


def run_v4(output: Path, cap: int | None, jobs: int) -> None:
    from tests.canonical.drtmle.regenerate import HERE as REFERENCE_HERE
    from tests.canonical.drtmle.regenerate import REFERENCE

    rows_path, validation_path, _ = shared.part_paths(output, "V4")
    smoke = cap is not None
    b_rows = shared.read_rows(record_path(shared.part_paths(output, "B")[0].name, output, smoke))
    built = v4_payloads(b_rows, cap, jobs)
    samples = pd.concat([sample for sample, _, _ in built], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in built])
    refit = pd.DataFrame([row for _, _, row in built])
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        directory = Path(scratch)
        write_csv(samples, directory / "samples.csv.gz", compression="gzip")
        write_csv(truths, directory / "truth.csv")
        result = directory / "reference-results.csv"
        REFERENCE.run(
            REFERENCE_HERE,
            directory / "samples.csv.gz",
            directory / "truth.csv",
            result,
            cores=min(jobs, R_WORKERS),
        )
        r_rows = shared.read_rows(result)
    shared.write_table(r_rows, rows_path)
    keys = ["scenario", "replicate", "estimand"]
    fresh = b_rows.loc[b_rows["replicate"].isin(refit["replicate"])]
    validation = shared.validation_frame(
        [
            shared.validation_row(
                "V4",
                "V4 payload C refit",
                shared.compare_rows(_long(refit, C), _long(fresh, C), keys),
            ),
            *compare_by_exit(
                "V4",
                "V4 R",
                fresh.loc[fresh["arm"] == T000],
                r_rows,
                r_rows[["scenario", "replicate", "score_max"]],
            ),
        ]
    )
    shared.write_table(validation, validation_path)


# ------------------------------------------------------------------------------ readings


def budget_rule() -> int:
    """``ceil((z * SD / resolution)^2)``: 1,954, which the declaration rounds up to 2,000."""
    return int(np.ceil((Z * COMMITTED_SD / RESOLUTION) ** 2))


def primary_label(interval: Interval) -> str:
    """The primary reading of the interval of the mean of ``C - T(0, 0, 0)``."""
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


def main_effects(wide: pd.DataFrame, guard: bool) -> dict[str, np.ndarray]:
    """The per-draw main effect of each factor, and of G when a guard event occurred."""
    out = {}
    for name, attribute in FACTORS.items():
        high = [arm.label for arm in FACTORIAL if getattr(arm, attribute)]
        low = [arm.label for arm in FACTORIAL if not getattr(arm, attribute)]
        out[name] = (wide[high].mean(axis=1) - wide[low].mean(axis=1)).to_numpy()
    if guard:
        out["G"] = (wide[T0G] - wide[T000]).to_numpy()
    return out


def interaction(wide: pd.DataFrame, pair: tuple[str, str]) -> np.ndarray:
    """The 2^3 two-way contrast: the arms whose two levels agree minus those whose differ."""
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


def factorial_table(
    part: str, rows: pd.DataFrame, declared: int, prefix: str = ""
) -> list[dict[str, Any]]:
    """The primary, the family, the bracket, the interactions and the localization of one part."""
    scope = f"{SCENARIO}, n = {int(rows['n'].iloc[0])}"
    smoke = rows.groupby("arm").size().ne(declared).any()
    wide = _wide(rows)
    guard = bool(rows.loc[rows["arm"] == T000, "guard_events"].sum() > 0)

    def labelled(text: str) -> str:
        return shared.label(prefix + text, bool(smoke))

    primary_values = (wide[C] - wide[T000]).to_numpy()
    primary_interval = student_interval(primary_values, confidence_level=shared.CONFIDENCE)
    primary = primary_label(primary_interval)
    effects = main_effects(wide, guard)
    level = 1.0 - (1.0 - shared.CONFIDENCE) / len(effects)
    intervals = {
        name: student_interval(values, confidence_level=level) for name, values in effects.items()
    }
    bracket_values = (wide[T111] - wide[C]).to_numpy()
    bracket = student_interval(bracket_values, confidence_level=shared.CONFIDENCE)
    out = [
        shared.reading(
            part,
            scope,
            "C - T000",
            value=float(np.mean(primary_values)),
            interval=primary_interval,
            result=labelled(primary),
        ),
        shared.reading(part, scope, "C - T000 half-width", value=primary_interval.width / 2.0),
        shared.reading(
            part,
            scope,
            "guard events of T000",
            value=float(rows.loc[rows["arm"] == T000, "guard_events"].sum()),
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
            "T111 - C",
            value=float(np.mean(bracket_values)),
            interval=bracket,
            result=labelled("P-bracket " + (HOLDS if bracket_holds(bracket) else FAILS)),
        )
    )
    out += [
        _statistic(
            part,
            scope,
            f"interaction {a}{b}",
            interaction(wide, (a, b)),
            shared.CONFIDENCE,
            "supplementary",
        )
        for a, b in INTERACTIONS
    ]
    label = localization_label(
        primary, primary_interval, float(np.mean(primary_values)), bracket, intervals
    )
    out.append(shared.reading(part, scope, "localization", result=labelled(label)))
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
            _statistic(part, scope, "bias", error, shared.CONFIDENCE, "supplementary"),
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
        difference = (wide[Switches(joint_tilt=True).label] - wide[T000]).to_numpy()
        interval = student_interval(difference, confidence_level=shared.CONFIDENCE)
        spread = float(np.std(difference, ddof=1))
        held = interval.contains(0.0) and spread <= CONTROL_SD[scenario]
        out += [
            shared.reading(
                "A",
                scenario,
                "T100 - T000",
                value=float(np.mean(difference)),
                interval=interval,
                result=shared.label("P-ctrl " + (HOLDS if held else FAILS), smoke),
            ),
            shared.reading(
                "A",
                scenario,
                "T100 - T000 SD",
                value=spread,
                result=f"bound {CONTROL_SD[scenario]}",
            ),
            _statistic(
                "A",
                scenario,
                "T111 - C",
                (wide[T111] - wide[C]).to_numpy(),
                shared.CONFIDENCE,
                "supplementary",
            ),
        ]
    return out


def scaling_table(rows: pd.DataFrame, b_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """``sqrt(n)`` times the mean of ``C - T(0, 0, 0)`` at each size, Bonferroni over three."""
    level = 1.0 - (1.0 - shared.CONFIDENCE) / 3.0
    combined = pd.concat([b_rows, rows], ignore_index=True) if not b_rows.empty else rows
    out = []
    for n, group in combined.groupby("n"):
        wide = _wide(group.loc[group["arm"].isin([C, T000])])
        values = np.sqrt(float(n)) * (wide[C] - wide[T000]).to_numpy()
        interval = student_interval(values, confidence_level=level)
        label = "increment" if interval.low > 0 else REVERSE if interval.high < 0 else ""
        smoke = len(wide) != REPLICATES
        out.append(
            shared.reading(
                "C",
                f"{SCENARIO}, n = {n}",
                "sqrt(n) (C - T000)",
                value=float(np.mean(values)),
                interval=interval,
                result=shared.label(label, smoke) if label else "",
            )
        )
    return out


def context_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """The supplementary Part C row: the ``C`` bias beside BD-1 and the committed n = 6,000 rung."""
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
    out = []
    for n, (name, value, low, high) in references.items():
        group = rows.loc[(rows["arm"] == C) & (rows["n"] == n)]
        if group.empty:
            continue
        out += [
            _statistic(
                "C",
                f"{SCENARIO}, n = {n}, C",
                "bias",
                (group["ate"] - group["truth"]).to_numpy(),
                shared.CONFIDENCE,
                "supplementary",
            ),
            shared.reading(
                "C",
                f"{SCENARIO}, n = {n}",
                name,
                value=float(value),
                interval=Interval(float(low), float(high)),
                result="supplementary",
            ),
        ]
    return out


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
        shared.write_table(validate_committed(rows), validation_path)
        return
    validation = prior_checks(part, output, smoke)
    shared.write_table(validation, validation_path)
    if shared.validated(validation, part):
        shared.write_table(fit_part(part, cap, jobs), rows_path)


def table(part: str, output: Path, jobs: int) -> pd.DataFrame:
    del jobs
    rows_path, validation_path, _ = shared.part_paths(output, part)
    validation = shared.read_rows(validation_path)
    v4_ran = True
    if part == "B":
        # V4 is a precondition of the Part B reading: a V4 that has not run leaves none.
        v4 = shared.optional_rows(shared.part_paths(output, "V4")[1])
        v4_ran = not v4.empty
        validation = pd.concat([validation, v4.assign(part="B")], ignore_index=True)
    out = shared.validation_readings(validation, part)
    if part == "V4":
        return shared.reading_frame(out)
    if not (v4_ran and shared.validated(validation, part)):
        out.append(shared.reading(part, "", "reading", result=shared.NOT_VALIDATED))
        return shared.reading_frame(out)
    rows = shared.read_rows(rows_path)
    if part == "A":
        focus = rows.loc[rows["scenario"] == SCENARIO]
        out += factorial_table(part, focus, BINARY.replicates, prefix=f"{COMMITTED}: ")
        out += control_table(rows)
    elif part == "B":
        out += factorial_table(part, rows, REPLICATES)
    else:
        b_rows = shared.optional_rows(shared.part_paths(output, "B")[0])
        out += scaling_table(rows, b_rows)
        out += context_table(rows)
    out += arm_summaries(part, rows)
    return shared.reading_frame(out)


def pushed(part: str) -> tuple[str, ...]:
    """The records a declared part reads from the pushed upstream."""
    directory = HERE.relative_to(ROOT).as_posix()
    if part == "B":
        return (f"{directory}/a-validation.csv",)
    if part == "C":
        return (f"{directory}/a-validation.csv", f"{directory}/v4-validation.csv")
    if part == "V4":
        return (f"{directory}/b-rows.csv.gz",)
    return ()


if __name__ == "__main__":
    shared.main(__doc__.splitlines()[0], PARTS, HERE, run_part, table, pushed)
