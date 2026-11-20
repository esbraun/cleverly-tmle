"""Probe: why some in-sample bootstrap refits of ``continuous-survival-bootstrap`` raise.

Exploratory, not a study.  The committed pilot (:mod:`.continuous_survival_bootstrap_pilot`) lost
2 of 120 in-sample fits because more than 1% of their refits raised ``DataError: An initial hazard
fit puts a combined jump of one or more at some time``.  This module rebuilds such refits and
reads the hazard fit that raises, and screens original samples for the same failure.  Its readings
are written in ``tests/canonical/continuous_survival_bootstrap/pilot-diagnosis.md``.

Each fit is stopped right after the engine's checks of the initial hazards (the walk never runs),
so a screened fit costs its nuisance fits only.  ``python -m
tests.studies.continuous_survival_bootstrap_ties --jobs N`` prints every reading.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import Any

import numpy as np
import pandas as pd

from cleverly.survival import _chunked
from cleverly.survival import engine as engine_module
from cleverly.utils.parallel import map_parallel
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_bootstrap as study
from tests.studies import continuous_survival_bootstrap_pilot as pilot
from tests.studies.evidence.seeds import stream_seed

#: The pilot's in-sample draws whose bootstrap was withheld.
WITHHELD_DRAWS = (38, 45)


class _Stop(Exception):
    """Raised in place of the walk: the checks of the initial hazards have run."""


_STATE: dict[str, Any] = {}


@contextmanager
def _instrumented(*, stop_main: bool) -> Iterator[dict[str, Any]]:
    """Record every hazard selection and the largest combined cause jump of each fit.

    ``stop_main`` stops every fit after its checks; otherwise only bootstrap replicates stop,
    so the fit itself runs whole and spawns its replicates.
    """
    original_jumps = _chunked.InitialHazards.jumps
    original_select = engine_module.select_hazard
    original_walk = engine_module.walk_grid
    original_replicate = engine_module._replicate

    def jumps(self: Any, lo: int, hi: int) -> np.ndarray:
        out = original_jumps(self, lo, hi)
        total = out.sum(axis=1)  # (2, units, M): every cause at each arm, unit and time
        fit = _STATE["fit"]
        fit["max_jump"] = max(fit["max_jump"], float(np.nanmax(total)))
        if np.any(total >= 1.0) and fit.get("offending") is None:
            a, unit, m = np.unravel_index(int(np.nanargmax(total)), total.shape)
            fit["offending"] = {
                "arm": int(a),
                "unit": int(lo + unit),
                "m": int(m),
                "cause_jumps": [float(v) for v in out[a, :, unit, m]],
                "sources": self.causes,
            }
        return out

    def select(*args: Any, **kwargs: Any) -> Any:
        chosen = original_select(*args, **kwargs)
        _STATE["fit"]["selections"].append(chosen)
        return chosen

    def walk(*args: Any, **kwargs: Any) -> Any:
        if stop_main or _STATE.get("in_replicate"):
            raise _Stop
        return original_walk(*args, **kwargs)

    def replicate(estimator: Any, data: Any, index: np.ndarray, seed: int) -> Any:
        _STATE["in_replicate"] = True
        _STATE["fit"] = _new_fit()
        try:
            values, reason = original_replicate(estimator, data, index, seed)
        finally:
            _STATE["in_replicate"] = False
        record = _STATE["fit"]
        record["reason"] = reason
        offending = record.get("offending")
        if offending is not None:
            record["reading"] = _reading(engine_module.resample(data, index), index, record)
        record.pop("offending", None)
        _STATE["replicates"].append(record)
        return values, reason

    _chunked.InitialHazards.jumps = jumps  # type: ignore[method-assign]
    engine_module.select_hazard = select
    engine_module.walk_grid = walk
    engine_module._replicate = replicate
    _STATE.update(fit=_new_fit(), replicates=[], in_replicate=False)
    try:
        yield _STATE
    finally:
        _chunked.InitialHazards.jumps = original_jumps  # type: ignore[method-assign]
        engine_module.select_hazard = original_select
        engine_module.walk_grid = original_walk
        engine_module._replicate = original_replicate


def _new_fit() -> dict[str, Any]:
    return {"max_jump": 0.0, "selections": [], "offending": None}


def _reading(data: Any, ids: np.ndarray, record: dict[str, Any]) -> dict[str, Any]:
    """The offending jump of one fit, read against its rows: who is at risk, who ties, why."""
    offending = record["offending"]
    a, unit, m = offending["arm"], offending["unit"], offending["m"]
    T = np.asarray(data.time, dtype=float)
    D = np.asarray(data.event)
    # The engine codes the reference arm 0 and the other arm 1.
    A = (np.asarray(data.treatment) == study.law.ARMS["treated"]).astype(int)
    ids = np.asarray(ids)
    source = offending["sources"][int(np.argmax(offending["cause_jumps"]))]
    grid = source._grid
    t = float(grid[m])
    at_risk = t <= T
    events = (t == T) & (D > 0)
    cause = int(np.argmax(offending["cause_jumps"]))
    code = sorted(study.law.CAUSES)[cause]
    scale = source._scale
    entry = next(iter(source._entries.values()))
    base_jump = float(entry.base[m] - (entry.base[m - 1] if m else 0.0))
    own = scale[A, np.arange(T.size)] if scale is not None else np.full(T.size, np.nan)
    tied_ids, tied_counts = np.unique(ids[events & (code == D)], return_counts=True)
    _, risk_counts = np.unique(ids[at_risk], return_counts=True)
    names = [s.chosen_name for s in record["selections"][: len(offending["sources"])]]
    return {
        "time": t,
        "time_rank_from_end": int(np.sum(np.unique(T[D > 0]) > t)),
        "last_event_time": float(T[D > 0].max()),
        "last_time": float(T.max()),
        "at_risk": int(at_risk.sum()),
        "distinct_at_risk": int(np.unique(ids[at_risk]).size),
        "largest_copies_at_risk": int(risk_counts.max()),
        "cause": study.law.CAUSES[code],
        "events_at_time": int((events & (code == D)).sum()),
        "event_units": int(tied_ids.size),
        "event_copies": [int(c) for c in tied_counts],
        "selected": names,
        "arm": ("control", "treated")[a],
        "unit_id": int(ids[unit]),
        "unit_copies": int(np.sum(ids == ids[unit])),
        "event_ids": [int(i) for i in tied_ids],
        "unit_received_arm": int(A[unit]),
        "unit_risk_score": float(scale[a, unit]) if scale is not None else float("nan"),
        "risk_set_scores_max": float(own[at_risk].max()),
        "risk_set_scores_sum": float(own[at_risk].sum()),
        "risk_set_scores_median": float(np.median(own[at_risk])),
        "baseline_jump": base_jump,
        "breslow_check": base_jump * float(own[at_risk].sum()),
        "cause_jumps": offending["cause_jumps"],
        "combined_jump": float(sum(offending["cause_jumps"])),
    }


# ----------------------------------------------------------------------- the readings


def bootstrap_readings(replicate: int, n_bootstrap: int = study.PILOT_BOOTSTRAP) -> dict[str, Any]:
    """Rebuild one pilot in-sample draw's bootstrap: each refit's largest jump and failure."""
    seed = pilot.pilot_seed("in_sample", replicate)
    frame = registered.sample(study.PRIMARY_N, seed)
    with warnings.catch_warnings(), _instrumented(stop_main=False) as state:
        warnings.simplefilter("ignore")
        # A withheld bootstrap raises in the study; the point is its records, which are kept.
        with suppress(Exception):
            registered.fit_with(
                frame, study.engine(cross_fit=False, n_bootstrap=n_bootstrap, seed=seed)
            )
        records = list(state["replicates"])
    return {
        "draw": replicate,
        "max_jumps": [r["max_jump"] for r in records],
        "failures": [
            {"replicate": b, "reason": r["reason"], **r.get("reading", {})}
            for b, r in enumerate(records)
            if r["reason"] is not None and not r["reason"].startswith("_Stop")
        ],
    }


def _screen(n: int, cross_fit: bool, r: int) -> dict[str, Any]:
    seed = stream_seed(study.STUDY, "ties", f"{n}/{int(cross_fit)}", r)
    frame = registered.sample(n, seed)
    with warnings.catch_warnings(), _instrumented(stop_main=True) as state:
        warnings.simplefilter("ignore")
        try:
            registered.fit_with(frame, study.engine(cross_fit=cross_fit, n_bootstrap=0, seed=seed))
            outcome = "walked"
        except _Stop:
            outcome = "checked"
        except Exception as error:  # the readings are the outcomes
            outcome = f"{type(error).__name__}: {str(error).split('. ')[0][:80]}"
        fit = state["fit"]
    return {
        "n": n,
        "cross_fit": cross_fit,
        "replicate": r,
        "max_jump": fit["max_jump"],
        "outcome": outcome,
    }


def original_screen(
    sizes: Sequence[int], draws: int, n_jobs: int, cross_fits: Sequence[bool] = (False, True)
) -> pd.DataFrame:
    """Original (non-bootstrap) samples: each fit's largest combined jump and its outcome."""
    keys = [(n, cf, r) for n in sizes for cf in cross_fits for r in range(draws)]
    return pd.DataFrame(map_parallel(_screen, keys, n_jobs=n_jobs))


def main(argv: Sequence[str] | None = None) -> None:
    """Print the probe's readings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--bootstrap", type=int, nargs="*", default=list(WITHHELD_DRAWS))
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)
    readings = map_parallel(bootstrap_readings, list(args.bootstrap), n_jobs=args.jobs)
    for reading in readings:
        jumps = np.asarray(reading["max_jumps"])
        print(
            f"draw {reading['draw']}: {len(reading['failures'])} of {jumps.size} refits failed; "
            f"largest combined jump quantiles 50/90/99/max "
            f"{np.round(np.quantile(jumps, [0.5, 0.9, 0.99, 1.0]), 3).tolist()}"
        )
        for failure in reading["failures"]:
            print("  ", failure)
    screen = original_screen((study.PRIMARY_N, 200), args.draws, args.jobs)
    summary = screen.groupby(["n", "cross_fit"]).agg(
        draws=("max_jump", "size"),
        at_least_one=("max_jump", lambda v: int((v >= 1.0).sum())),
        above_half=("max_jump", lambda v: int((v >= 0.5).sum())),
        median=("max_jump", "median"),
        q99=("max_jump", lambda v: float(np.quantile(v, 0.99))),
        largest=("max_jump", "max"),
    )
    print(summary.to_string())
    print(screen["outcome"].str.split(":").str[0].value_counts().to_string())
    others = screen.loc[~screen["outcome"].isin(["checked", "walked"]), "outcome"]
    print(others.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
