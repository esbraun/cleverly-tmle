"""The memory guard of ``LTMLE``, ``SDR`` and ``IPSI`` (X106 part J2d, ``_memory_guard.py``).

Before any learner the fit predicts the peak of the whole call from counts it knows once the
plans, horizons and draws are resolved, and refuses when the prediction exceeds
``max_memory_bytes`` or the memory available.  These witnesses check:

- the calibration literals against the records they were generated from;
- the bound ``peak <= predicted <= FACTOR * peak`` on traced fits of every family at n 2,000
  (one untraced warm-up fit first, the panel built outside the traced window), and mutation
  controls that drop one term set from the installed constants, after which the bound fails;
- the worker term's arithmetic on fixed counts, against the formula written out here, with
  mutation controls on each of its constants;
- the in-range predicate on one fit of each kind, inside and outside;
- the count sources (width, separable and mixed-node families, the settings each count reads)
  and the cost of reading them;
- the refusal at its edges, the default budget, the budget kept by a refit, the replicate skip,
  the preview, the facade's spelled call and the construction checks.

The worker term spans processes, so its bytes have no traced witness here; the calibration's cold
worker runs are its evidence (``docs/technical-reference/longitudinal-tmle.md``, "Memory").
"""

from __future__ import annotations

import math
import re
import tracemalloc
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal, make_point_survival
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental
from cleverly.learners import SuperLearner
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    DynamicRegimen,
    LongitudinalData,
    SurvivalCTMLE,
    SurvivalDRTMLE,
    _memory_guard,
)
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal._memory_guard import FitCounts, LongitudinalMemoryError
from cleverly.msm import MSM
from cleverly.utils import memory as memory_module
from tests.diagnostics.x106_j2d_memory import generate
from tests.unit import _longitudinal_memory_guard_support as support

N = 2_000
MIB = 2**20
REFERENCE = Path(__file__).parents[2] / "docs" / "technical-reference" / "longitudinal-tmle.md"


def _bound_holds(peak: int, predicted: float) -> bool:
    return peak <= predicted <= _memory_guard.FACTOR * peak


# ------------------------------------------------------------------ calibration literals


def test_the_installed_constants_are_the_calibration_records() -> None:
    worker, helper, share = generate.workers()
    fitted, boxes = generate.predicate()
    assert generate.constants() == _memory_guard.CONSTANTS
    assert (worker, helper) == (_memory_guard.WORKER_BYTES, _memory_guard.HELPER_BYTES)
    assert (worker, helper) == (240 * MIB, 64 * MIB)
    assert share == _memory_guard.WORKER_SHARE
    assert fitted == _memory_guard.FITTED
    assert boxes == _memory_guard.EXTENSIONS
    assert (_memory_guard.MARGIN, _memory_guard.FACTOR, _memory_guard.MAX_JOBS) == (1.35, 3.5, 4)
    assert set(_memory_guard.CONSTANTS) == set(generate.FAMILIES)
    for constants in _memory_guard.CONSTANTS.values():
        assert set(constants) <= set(_memory_guard.TERMS) | {"c0"}
        assert min(constants.values()) >= 0.0


def test_the_reference_carries_the_generated_tables() -> None:
    text = REFERENCE.read_text()
    assert generate.constants_table() in text
    assert generate.constants_table(generate.ROUTE_FAMILIES) in text
    assert generate.ranges_table() in text
    assert generate.extensions_table() in text


def test_the_reference_quotes_the_copied_calibration_outputs() -> None:
    # Every range the accuracy section quotes is read from an output copied beside generate.py.
    text = " ".join(REFERENCE.read_text().split())
    missing = [phrase for phrase in generate.accuracy_phrases() if phrase not in text]
    assert not missing
    low, high = generate.accuracy_readings()["checked"]
    assert (low["family"], low["censor"], low["n"]) == ("sdr", "True", "20000")
    assert high["est"] == "learned"
    assert "plans repository" not in text


def test_the_predicate_reads_extbox_out_with_one_documented_correction() -> None:
    # Every fitted bound but the time-to-event families' censoring is the record's own.
    text = (Path(generate.__file__).parent / "extbox.out").read_text()
    for family, bounds in _memory_guard.FITTED.items():
        line = next(row for row in text.splitlines() if row.startswith(f"{family} {{"))
        for field, (low, high) in bounds.items():
            if family in generate.POINT_SURVIVAL and field == "censored":
                assert (low, high) == (0, 1)
            else:
                assert f"'{field}': ({low}, {high})" in line
    assert "hazard_paths" not in _memory_guard.EXTENSIONS  # the R 8 path group is excluded


# ------------------------------------------------------------------ traced bounds and controls

#: Per fixture: the calibration case, the family the guard must pick, and the term set the
#: mutation control drops (with the expected margined ratio after it, from the traced peaks of
#: ``witness11.jsonl``; the two continuous fixtures' ratios are read against the rev 11c
#: constants).
TRACED: dict[str, tuple[dict[str, Any], str, tuple[str, ...]]] = {
    # LTMLE survival T 4, R 4, H 4: 1.97, then 0.63 without R S J and R H J.
    "iterated_mean": (
        {"est": "ltmle_surv", "T": 4, "R": 4, "folds": 1, "H": 4},
        "ltmle",
        ("RSJ", "RHJ"),
    ),
    # LTMLE end of study T 8, R 4, least squares: 1.37, then 0.70.
    "end_of_study": (
        {"est": "ltmle_eos", "T": 8, "R": 4, "folds": 1, "H": 1, "outcome": "linear"},
        "ltmle",
        ("RT", "RSJ", "FRT"),
    ),
    # LTMLE T 4, R 4, 64 baseline columns: 1.47, then 0.38.
    "width": (
        {"est": "ltmle_eos", "T": 4, "R": 4, "folds": 1, "H": 1, "P": 64, "outcome": "linear"},
        "ltmle",
        ("p", "Rp", "Tp"),
    ),
    # SDR survival T 4, R 4, H 4: 1.61, then 0.31.
    "sdr": ({"est": "sdr_surv", "T": 4, "R": 4, "folds": 1, "H": 4}, "sdr", ("RSJ",)),
    # IPSI survival T 4, R 4, H 4: 1.37, then 0.30.
    "ipsi": ({"est": "ipsi_surv", "T": 4, "R": 4, "folds": 1, "H": 4}, "ipsi", ("RSJ",)),
    # LTMLE shift policies T 2, R 2, 40 bins: 1.37, then 0.07.
    "mtp": (
        {"est": "mtp", "T": 2, "R": 2, "folds": 1, "H": 1, "B": 40},
        "ltmle_continuous",
        ("BB", "TB"),
    ),
    # SDR shift policies T 2, R 2, 40 bins: 1.38, then 0.13.
    "sdr_mtp": (
        {"est": "sdr_mtp", "T": 2, "R": 2, "folds": 1, "H": 1, "B": 40},
        "sdr_continuous",
        ("BB", "TB"),
    ),
    # Hazard path fit T 8, R 4, no covariate after baseline: 1.35, then 0.15.
    "paths": (
        {"est": "hz_path", "T": 8, "R": 4, "folds": 1, "H": 8},
        "hazard_paths",
        ("RT", "RTJ"),
    ),
    # Hazard path fit T 4, R 2 and a coin policy at 4 nodes (18 slots): 1.52, then 0.45 without
    # the slot terms and the path_peak floor.
    "policy_paths": (
        {"est": "hz_policy", "T": 4, "R": 2, "folds": 1, "H": 4, "k": 4},
        "hazard_paths",
        ("TU", "TUJ"),
    ),
    # Hazard, one node, T 8, R 2, two causes: 1.35, then 0.37.
    "one_node": (
        {"est": "hz_point", "T": 8, "R": 2, "folds": 1, "H": 8, "J": 2},
        "hazard_one_node",
        ("TJ", "T"),
    ),
    # Hazard backward pass T 4, R 4: 1.65, then 0.51.
    "backward": (
        {"est": "hz_back", "T": 4, "R": 4, "folds": 1, "H": 4},
        "hazard_backward",
        ("RTJ", "TJ", "RSJ", "FRT"),
    ),
    # Hazard with two static and two separable arms, T 8: 1.72, then 0.53.
    "separable": (
        {"est": "hz_separable", "T": 8, "R": 4, "folds": 1, "H": 8, "J": 2},
        "hazard_separable",
        ("RSJ", "RT"),
    ),
}


@pytest.mark.parametrize("name", sorted(TRACED))
def test_the_prediction_bounds_the_traced_peak_and_a_dropped_term_breaks_it(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, family, dropped = TRACED[name]
    peak, counts = support.traced(case, N)
    assert counts.family == family
    predicted = _memory_guard.predicted_bytes(counts)
    assert _bound_holds(peak, predicted), predicted / peak
    # Mutation control: drop the term set from the installed constants.
    kept = {k: v for k, v in _memory_guard.CONSTANTS[family].items() if k not in dropped}
    monkeypatch.setitem(_memory_guard.CONSTANTS, family, kept)
    if "TU" in dropped:
        monkeypatch.setattr(hazard_module, "path_peak", lambda *args: 0.0)
    mutated = _memory_guard.predicted_bytes(counts)
    assert mutated < predicted
    assert not _bound_holds(peak, mutated), mutated / peak


def test_the_refined_grid_is_counted_and_a_nominal_count_breaks_the_mtp_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The density counts the bins of the refined grid, not ``density_bins``.

    ``bin_edges`` splits the wide outer core bins, so the ``mtp`` case's 40 core bins fit 49.
    Counted at 40, its prediction fell below the traced peak until X106 D1a built the hazard
    design once instead of twice (2.24 and 1.55 times the peak since), so the control is the
    size of the drop: the refined count adds at least a third to the prediction.
    """
    case, family, _ = TRACED["mtp"]
    peak, counts = support.traced(case, N)
    assert counts.family == family
    assert counts.bins > case["B"]
    predicted = _memory_guard.predicted_bytes(counts)
    assert _bound_holds(peak, predicted)
    monkeypatch.setattr(_memory_guard, "refined_bins", lambda data, nominal: nominal)
    _, nominal = support.traced(case, N)
    assert nominal.bins == case["B"]
    assert _memory_guard.predicted_bytes(nominal) < 0.75 * predicted


def test_the_two_cause_survival_bound() -> None:
    peak, counts = support.traced(
        {"est": "ltmle_surv", "T": 4, "R": 2, "folds": 1, "H": 4, "J": 2}, N
    )
    assert counts.causes == 2
    assert _bound_holds(peak, _memory_guard.predicted_bytes(counts))


def test_the_repeats_factor_is_witnessed() -> None:
    # Two draws kept: 1.91 times the peak.  One draw counted would read 0.96, too close to
    # the bound to be a control a library release cannot flip, so the exact-value witness
    # ``test_two_draws_count_twice_at_one_worker_and_at_four`` is the control.
    peak, counts = support.traced(
        {"est": "ltmle_eos", "T": 4, "R": 2, "folds": 3, "H": 1, "repeats": 2}, N
    )
    assert counts.repeats == 2
    assert _bound_holds(peak, _memory_guard.predicted_bytes(counts))


def test_the_bootstrap_term_is_witnessed() -> None:
    # One replicate beside the kept result at one worker: 1.81 times the peak.  Without the
    # replicate the prediction reads 0.90, so the exact-value witness
    # ``test_the_bootstrap_and_the_fold_fan_out_take_the_larger`` is the control.
    peak, counts = support.traced(
        {"est": "ltmle_eos", "T": 4, "R": 2, "folds": 1, "H": 1, "boot": 2}, N
    )
    assert counts.bootstrap == 2
    assert _bound_holds(peak, _memory_guard.predicted_bytes(counts))


# ------------------------------------------------------------------ the worker term


def _fixed(**changes: Any) -> FitCounts:
    base = FitCounts(
        estimator="SDR",
        family="sdr",
        n=20_000,
        n_times=8,
        plans=2,
        horizons=(8,),
        causes=1,
        folds=10,
        width=17,
        bins=0,
        msm=False,
        repeats=1,
        bootstrap=0,
        jobs=4,
        panel_bytes=30 * MIB,
    )
    return replace(base, **changes)


#: ``(family, folds) -> (w, m)``, read off ``worker-share.json`` and worked out by hand:
#: linear between measured folds, the last segment's slope above the last, never below the
#: largest measured value, and flat for a family measured at one fold count.
SHARES = {
    ("sdr", 3): (0.68, 2.10),
    ("sdr", 5): (0.85, 1.71),
    ("sdr", 7): (0.85 + 0.67 * 2 / 5, 1.71 + 0.51 * 2 / 5),
    ("sdr", 10): (1.52, 2.22),
    ("sdr", 20): (1.81, 2.67),
    ("sdr", 30): (1.81 + 0.29, 2.67 + 0.45),
    ("ipsi", 30): (1.39, 2.46 + 0.50),
    ("ltmle", 3): (0.39, 1.54),
    ("ltmle", 10): (0.39, 1.54),
}


def _written_out(counts: FitCounts, w: float, m: float) -> int:
    """The whole-call formula of plan rev 11b, with 240 and 64 MiB per worker and helpers."""
    draw = _memory_guard.draw_bytes(counts)
    j = counts.jobs
    started = j * 240 * MIB + 64 * MIB
    workers = 0.0
    if j > 1 and counts.folds > 1:
        workers = (m - 1) * draw + min(j, counts.folds) * w * draw + started
    bootstrap = 0.0
    if counts.bootstrap:
        bootstrap = (
            draw if j == 1 else min(j, counts.bootstrap) * (draw + counts.panel_bytes) + started
        )
    return math.ceil(1.35 * (counts.repeats * draw + max(workers, bootstrap)))


def _witness(counts: FitCounts, w: float, m: float) -> bool:
    return abs(_memory_guard.predicted_bytes(counts) - _written_out(counts, w, m)) <= 1


@pytest.mark.parametrize(("family", "folds"), sorted(SHARES))
def test_the_worker_term_follows_the_measured_shares(family: str, folds: int) -> None:
    w, m = SHARES[(family, folds)]
    assert _memory_guard.share(family, folds) == pytest.approx((w, m), abs=1e-12)
    counts = _fixed(family=family, folds=folds)
    assert _witness(counts, w, m)


def test_two_draws_count_twice_at_one_worker_and_at_four() -> None:
    for changes in ({"jobs": 1}, {"jobs": 4, "folds": 10}):
        one = _fixed(**changes)
        two = replace(one, repeats=2)
        w, m = SHARES[("sdr", one.folds)] if one.jobs > 1 else (0.0, 0.0)
        assert _witness(two, w, m)
        draw = _memory_guard.draw_bytes(one)
        assert _memory_guard.predicted_bytes(two) - _memory_guard.predicted_bytes(one) == (
            pytest.approx(1.35 * draw, abs=2)
        )


def test_one_worker_or_one_fold_adds_no_worker_term() -> None:
    for counts in (_fixed(jobs=1), _fixed(folds=1)):
        assert _memory_guard.predict(counts).workers == 0.0
        draw = _memory_guard.draw_bytes(counts)
        assert _memory_guard.predicted_bytes(counts) == math.ceil(1.35 * draw)
        assert not _memory_guard.predict(counts).parallel


def test_more_workers_than_folds_counts_every_started_worker() -> None:
    # loky starts all j workers on the first submit: 4 baselines at F 2, not 2.
    counts = _fixed(family="ltmle", folds=2, jobs=4)
    draw = _memory_guard.draw_bytes(counts)
    expected = math.ceil(1.35 * (draw + 0.54 * draw + 2 * 0.39 * draw + 4 * 240 * MIB + 64 * MIB))
    assert _memory_guard.predicted_bytes(counts) == expected


def test_the_bootstrap_and_the_fold_fan_out_take_the_larger() -> None:
    replicates = _fixed(folds=3, bootstrap=4, panel_bytes=400 * MIB)
    folds = _fixed(folds=10, bootstrap=1, panel_bytes=MIB)
    for counts in (replicates, folds):
        w, m = SHARES[("sdr", counts.folds)]
        assert _witness(counts, w, m)
    assert _memory_guard.predict(replicates).bootstrap > _memory_guard.predict(replicates).workers
    assert _memory_guard.predict(folds).workers > _memory_guard.predict(folds).bootstrap
    one = _fixed(jobs=1, bootstrap=2)
    draw = _memory_guard.draw_bytes(one)
    assert _memory_guard.predicted_bytes(one) == math.ceil(1.35 * 2 * draw)


@pytest.mark.parametrize(
    ("changes", "parallel"),
    [
        ({"jobs": 4, "folds": 10}, True),
        ({"jobs": 4, "folds": 1}, False),
        ({"jobs": 4, "folds": 1, "bootstrap": 2}, True),
        ({"jobs": 1, "folds": 10, "bootstrap": 2}, False),
    ],
)
def test_the_n_jobs_remedy_appears_exactly_when_the_worker_term_is_not_zero(
    changes: dict[str, int], parallel: bool
) -> None:
    prediction = _memory_guard.predict(_fixed(**changes))
    assert prediction.parallel is parallel
    error = _memory_guard.refusal(prediction, 1, explicit=True)
    assert ("n_jobs=1" in str(error.refusal.alternative)) is parallel
    assert ("parallel workers" in str(error)) is parallel


def _mutated_share(monkeypatch: pytest.MonkeyPatch, rule: Callable[[float, float], Any]) -> None:
    points = _memory_guard.WORKER_SHARE["sdr"]
    monkeypatch.setitem(
        _memory_guard.WORKER_SHARE, "sdr", tuple((f, *rule(w, m)) for f, w, m in points)
    )


def test_mutation_a_parent_multiple_of_one_fails_the_f10_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mutated_share(monkeypatch, lambda w, m: (w, 1.0))
    assert not _witness(_fixed(folds=10), *SHARES[("sdr", 10)])


def test_mutation_a_share_frozen_at_three_folds_fails_the_f10_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mutated_share(monkeypatch, lambda w, m: (0.68, m))
    assert not _witness(_fixed(folds=10), *SHARES[("sdr", 10)])


@pytest.mark.parametrize("constant", ["WORKER_BYTES", "HELPER_BYTES"])
def test_mutation_a_dropped_process_constant_fails_the_four_worker_witness(
    constant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = _fixed(jobs=4, folds=10)
    assert _witness(counts, *SHARES[("sdr", 10)])
    monkeypatch.setattr(_memory_guard, constant, 0)
    assert not _witness(counts, *SHARES[("sdr", 10)])


# ------------------------------------------------------------------ the in-range predicate


def _ltmle(**changes: Any) -> FitCounts:
    return _fixed(estimator="LTMLE", family="ltmle", **{"folds": 3, "jobs": 1, **changes})


@pytest.mark.parametrize(
    ("counts", "named"),
    [
        (
            _ltmle(n_times=12, plans=4, horizons=tuple(range(1, 13))),
            "12 nodes, 12 horizons with 4 regimens",
        ),
        (
            _fixed(family="hazard_paths", estimator="LTMLE", plans=8, causes=2, folds=3, jobs=1),
            "8 regimens",
        ),
        (_ltmle(censored=True, causes=2, width=9), "censoring with 8 nodes"),
        (_ltmle(markov=True), "a Markov order"),
        (_fixed(family="ipsi", estimator="IPSI", levels=3, jobs=1), "3 treatment levels"),
        (_ltmle(n=2_000), "2,000 units"),
        (_ltmle(jobs=8), "n_jobs 8"),
        (_ltmle(n_times=12, plans=3, horizons=(12,), width=25), "12 nodes with 3 regimens"),
    ],
)
def test_the_clause_names_each_kind_of_fit_outside_the_range(counts: FitCounts, named: str) -> None:
    assert _memory_guard.outside_range(counts) == named
    assert not _memory_guard.in_range(counts)
    prediction = _memory_guard.predict(counts)
    clause = f"outside the calibrated range of the memory model ({named})"
    assert clause in str(_memory_guard.refusal(prediction, 1, explicit=False))
    assert clause in _memory_guard.preview_note(prediction)


@pytest.mark.parametrize(
    "counts",
    [
        _fixed(),  # SDR T 8, R 2, F 10 at four workers
        _ltmle(plans=4, causes=2, horizons=tuple(range(1, 9)), folds=10, jobs=4),
        # The checked extension LTMLE T 12 at two regimens, three folds and one cause.
        _ltmle(n_times=12, plans=2, horizons=(12,), width=25),
        _fixed(family="hazard_one_node", estimator="LTMLE", censored=True, causes=2, plans=2),
    ],
)
def test_no_clause_inside_the_range(counts: FitCounts) -> None:
    assert _memory_guard.in_range(counts)
    prediction = _memory_guard.predict(counts)
    assert "calibrated range" not in str(_memory_guard.refusal(prediction, 1, explicit=False))
    assert "calibrated range" not in _memory_guard.preview_note(prediction)


def _registered(family: str, **changes: Any) -> FitCounts:
    """The continuous fits of the registered ``longitudinal_mtp`` and ``longitudinal_sdr`` studies."""
    base = FitCounts(
        estimator="SDR" if family == "sdr_continuous" else "LTMLE",
        family=family,
        n=2_000,
        n_times=2,
        plans=4,
        horizons=(2,),
        causes=1,
        folds=1,
        width=4,
        bins=320,
        msm=False,
        repeats=1,
        bootstrap=0,
        jobs=1,
        panel_bytes=0,
    )
    return replace(base, **changes)


@pytest.mark.parametrize("family", ["ltmle_continuous", "sdr_continuous"])
def test_the_registered_continuous_studies_lie_in_the_fitted_range(family: str) -> None:
    # Rev 11c fitted the continuous families at 2,000 units, width 4 and up to 320 bins.
    assert _memory_guard.in_range(_registered(family))


@pytest.mark.parametrize("family", ["ltmle_continuous", "sdr_continuous"])
@pytest.mark.parametrize(
    ("changes", "named"), [({"bins": 321}, "321 density bins"), ({"n": 1_999}, "1,999 units")]
)
def test_one_step_past_the_continuous_fitted_edge_is_out(
    family: str, changes: dict[str, int], named: str
) -> None:
    assert _memory_guard.outside_range(_registered(family, **changes)) == named


#: The traced peak of ``sdr_mtp`` at n 5,000, T 4, R 1, F 5 and B 40: the ``peak`` of that row
#: of ``bins-5k.jsonl`` (``rows/X106/J2/evidence/2026-11-19-j2d/rev9/`` in the plans repository).
SDR_F5_B40_PEAK = 107_567_399


def test_the_sdr_continuous_f5_b40_cell_is_covered_by_the_margin() -> None:
    # The rev 11b constants read 1.20 here; the rev 11c refit fitted this cell, so the margined
    # draw covers at least 1.35 times its peak (1.46 under the installed constants).
    counts = FitCounts(
        estimator="SDR",
        family="sdr_continuous",
        n=5_000,
        n_times=4,
        plans=1,
        horizons=(4,),
        causes=1,
        folds=5,
        width=9,
        bins=40,
        msm=False,
        repeats=1,
        bootstrap=0,
        jobs=1,
        panel_bytes=460_000,
    )
    # One draw at one worker: the prediction is the margined draw, rounded up to whole bytes.
    assert _memory_guard.predicted_bytes(counts) >= 1.35 * SDR_F5_B40_PEAK


def _continuous(
    family: str,
    n: int,
    n_times: int,
    plans: int,
    folds: int,
    *,
    width: int,
    bins: int,
    panel_bytes: int,
) -> FitCounts:
    return FitCounts(
        estimator="SDR" if family == "sdr_continuous" else "LTMLE",
        family=family,
        n=n,
        n_times=n_times,
        plans=plans,
        horizons=(n_times,),
        causes=1,
        folds=folds,
        msm=False,
        repeats=1,
        bootstrap=0,
        jobs=1,
        width=width,
        bins=bins,
        panel_bytes=panel_bytes,
    )


#: The held-out continuous cells (the ``bins_check`` rows, checked and not fitted, of the X106 part D
#: refit, ``rows/X106/D/evidence/2026-11-25-j2d-recal/grid-*.jsonl`` in the plans repository): each
#: row's guard counts and its traced peak.
HELD_OUT: dict[str, tuple[FitCounts, int]] = {
    "mtp T 8, B 320": (
        _continuous("ltmle_continuous", 2_000, 8, 1, 1, width=17, bins=320, panel_bytes=320_000),
        1_000_158_283,
    ),
    "mtp F 10, B 320": (
        _continuous("ltmle_continuous", 2_000, 2, 4, 10, width=5, bins=320, panel_bytes=116_000),
        794_556_292,
    ),
    "mtp n 5,000, B 20": (
        _continuous("ltmle_continuous", 5_000, 2, 4, 3, width=5, bins=35, panel_bytes=290_000),
        46_532_843,
    ),
    "sdr_mtp T 8, B 320": (
        _continuous("sdr_continuous", 2_000, 8, 1, 1, width=17, bins=320, panel_bytes=320_000),
        1_000_160_483,
    ),
    "sdr_mtp F 10, B 320": (
        _continuous("sdr_continuous", 2_000, 2, 4, 10, width=5, bins=320, panel_bytes=116_000),
        1_111_518_731,
    ),
    "sdr_mtp n 5,000, B 20": (
        _continuous("sdr_continuous", 5_000, 2, 4, 3, width=5, bins=35, panel_bytes=290_000),
        75_347_076,
    ),
}


#: The held-out cells of the two density routes: n 5,000, T 4, R 2, F 5, with 160 bins on the index
#: route.  The index rows are the part D refit's ``dens_check`` rows; the classifier rows are J2d's
#: ``dens-check.jsonl`` (``rows/X106/J2/evidence/2026-11-19-j2d/rev9/``), whose family was not
#: refitted.
HELD_OUT.update(
    {
        f"{family} F 5": (
            replace(
                _continuous(
                    family.replace("_index", "").replace("_classifier", ""),
                    5_000,
                    4,
                    2,
                    5,
                    width=9,
                    bins=0 if family.endswith("_classifier") else 160,
                    panel_bytes=460_000,
                ),
                family=family,
            ),
            peak,
        )
        for family, peak in (
            ("ltmle_continuous_index", 195_805_546),
            ("ltmle_continuous_classifier", 9_629_809),
            ("sdr_continuous_index", 599_921_986),
            ("sdr_continuous_classifier", 16_309_789),
        )
    }
)


@pytest.mark.parametrize("name", sorted(HELD_OUT))
def test_a_held_out_continuous_cell_in_range_reads_at_least_the_floor(name: str) -> None:
    # The floor of 1.32 covers the resident set over the traced peak (``packet10.out``).
    counts, peak = HELD_OUT[name]
    assert _memory_guard.in_range(counts)
    assert _memory_guard.predicted_bytes(counts) >= 1.32 * peak


# ------------------------------------------------------------------ the density route


class _IndexLogistic(LogisticRegression):
    """A logistic hazard that reads only the bin index, as ``IndexLogistic`` in ``cases.py``."""

    bin_design = "index"


def _route_counts(est: str, n: int = 300, **case: Any) -> FitCounts:
    """The counts of one continuous fit, which a one-byte budget refuses before any learner."""
    overrides = case.pop("overrides", {})
    spec = {"est": est, "T": 2, "R": 2, "folds": 1, "H": 1, "B": 12, **case}
    _, call = support.build(spec, n, max_memory_bytes=1, **overrides)
    with support.captured_counts() as seen, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(LongitudinalMemoryError):
            call()
    return seen[0]


@pytest.mark.parametrize(("est", "prefix"), [("mtp", "ltmle"), ("sdr_mtp", "sdr")])
@pytest.mark.parametrize(
    ("overrides", "route", "bins"),
    [
        ({}, "", 12),
        ({"treatment_learner": _IndexLogistic(max_iter=1000)}, "_index", 12),
        ({"ratio": "classifier"}, "_classifier", 0),
    ],
)
def test_each_density_route_is_its_own_family(
    est: str, prefix: str, overrides: dict[str, Any], route: str, bins: int
) -> None:
    # The classifier route fits no binned density and refuses a bin count, so it runs without
    # one; a binned route's count is the refined grid of its 12 core bins.
    counts = _route_counts(est, overrides=overrides, B=bins)
    assert counts.family == f"{prefix}_continuous{route}"
    assert counts.bins >= bins if bins else counts.bins == 0


@pytest.mark.parametrize(
    "learner",
    [None, SuperLearner([_IndexLogistic(max_iter=1000)], task="classification")],
    ids=["default", "super_learner"],
)
def test_a_default_or_stacked_treatment_learner_counts_as_binned(learner: Any) -> None:
    # Neither carries ``bin_design``, so the density step builds the indicator block whatever
    # a library member sets.
    counts = _route_counts("mtp", overrides={"treatment_learner": learner})
    assert (counts.family, counts.bins) == ("ltmle_continuous", 12)


def test_a_bin_design_lost_on_cloning_keeps_the_binned_family() -> None:
    # The density step may receive a clone of the learner, which drops an instance attribute;
    # the guard then counts the binned route, which over-counts.
    learner = LogisticRegression(max_iter=1000)
    learner.bin_design = "index"
    assert _memory_guard.density_route(LTMLE({"natural": 0}, treatment_learner=learner)) == ""
    assert _memory_guard.density_route(LTMLE({"natural": 0}, ratio="classifier")) == "_classifier"


def test_the_registered_mtp_primary_on_the_index_route_predicts_tens_of_mib() -> None:
    # The ``longitudinal_mtp`` primary: n 2,000, T 2, four shift plans, 320 bins and an
    # index-only hazard.  The binned family predicted 2.2 GiB for it.
    counts = _route_counts(
        "mtp", 2_000, R=4, B=320, overrides={"treatment_learner": _IndexLogistic(max_iter=1000)}
    )
    assert counts.family == "ltmle_continuous_index"
    assert _memory_guard.in_range(counts)  # width 5 here
    assert _memory_guard.predicted_bytes(counts) < 200 * MIB
    # With the study's one baseline column the width is 4, below the fitted 5.
    primary = replace(counts, width=4)
    assert _memory_guard.outside_range(primary) == "4 design columns"
    assert _memory_guard.predicted_bytes(primary) < 200 * MIB


def test_mutation_ignoring_the_density_route_fails_the_route_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_memory_guard, "density_route", lambda estimator: "")
    index = {"treatment_learner": _IndexLogistic(max_iter=1000)}
    assert _route_counts("mtp", overrides=index).family == "ltmle_continuous"
    assert _route_counts("sdr_mtp", overrides={"ratio": "classifier"}, B=0).bins > 0
    counts = _route_counts("mtp", 2_000, R=4, B=320, overrides=index)
    assert _memory_guard.predicted_bytes(counts) > 2**30


# ------------------------------------------------------------------ the simultaneous band

#: The band-on cell that reads lowest in ``band_check.out``: ``mtp`` on the classifier route at
#: n 2,000, T 4, R 4, F 1 with ``simultaneous=True``; its traced peak is the ``peak`` of that
#: row of ``band-2k.jsonl`` (``rows/X106/J2/evidence/2026-11-19-j2d/rev9/`` in the plans
#: repository).
BAND_CELL_PEAK = 13_214_633


def _band_cell(simultaneous: bool) -> FitCounts:
    return FitCounts(
        estimator="LTMLE",
        family="ltmle_continuous_classifier",
        n=2_000,
        n_times=4,
        plans=4,
        horizons=(4,),
        causes=1,
        folds=1,
        width=9,
        bins=0,
        msm=False,
        repeats=1,
        bootstrap=0,
        jobs=1,
        panel_bytes=184_000,
        simultaneous=simultaneous,
    )


def test_the_band_term_covers_the_lowest_band_on_cell() -> None:
    # 1.54 with the band term; 0.70 without it.
    assert _memory_guard.predicted_bytes(_band_cell(True)) >= 1.35 * BAND_CELL_PEAK


def test_mutation_a_dropped_band_term_fails_the_band_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_memory_guard, "band_bytes", lambda counts: 0.0)
    assert _memory_guard.predicted_bytes(_band_cell(True)) < BAND_CELL_PEAK


def test_the_band_term_is_twice_the_draw_buffer_and_absent_without_the_band() -> None:
    from cleverly.inference.multiplier import _block_size

    on = _memory_guard.predict(_band_cell(True))
    off = _memory_guard.predict(_band_cell(False))
    assert on.band == 2 * _block_size(2_000, 10_000) * 8 * 2_000 == 2 * 256 * 8 * 2_000
    assert off.band == 0.0
    assert off.predicted == math.ceil(_memory_guard.MARGIN * off.draw)
    assert on.predicted == math.ceil(_memory_guard.MARGIN * (on.draw + on.band))
    # A small multiplier caps the block; the term is added once, not per worker.
    few = _memory_guard.predict(replace(_band_cell(True), n_multiplier=8))
    assert few.band == 2 * 8 * 8 * 2_000
    parallel = _memory_guard.predict(replace(_band_cell(True), folds=4, jobs=4))
    assert parallel.band == on.band
    assert "the simultaneous band" in _memory_guard.preview_note(on)
    assert "the simultaneous band" not in _memory_guard.preview_note(off)


@pytest.mark.parametrize(("repeats", "jobs", "folds"), [(2, 1, 1), (1, 4, 4), (2, 4, 4)])
def test_the_band_counts_once_per_call_in_the_prediction(
    repeats: int, jobs: int, folds: int
) -> None:
    # A repeated fit builds its draws' bands one at a time, and the band runs in the parent
    # after the workers return, so neither the repeats nor the workers multiply it.
    counts = replace(_band_cell(True), repeats=repeats, jobs=jobs, folds=folds)
    prediction = _memory_guard.predict(counts)
    assert prediction.band == 2 * 256 * 8 * 2_000
    whole = repeats * prediction.draw + max(prediction.workers, prediction.bootstrap)
    assert prediction.predicted == math.ceil(_memory_guard.MARGIN * (whole + prediction.band))


@pytest.mark.parametrize("simultaneous", [True, False])
def test_the_counts_read_the_band_setting_of_the_estimator(simultaneous: bool) -> None:
    counts = _route_counts("mtp", overrides={"simultaneous": simultaneous, "n_multiplier": 1_000})
    assert (counts.simultaneous, counts.n_multiplier) == (simultaneous, 1_000)
    assert (_memory_guard.predict(counts).band > 0) is simultaneous


#: One fitted cell of each route family, n 2,000, T 4, R 4, F 3, B 320 on the index route: the
#: index rows from the part D refit (``rows/X106/D/evidence/2026-11-25-j2d-recal/grid-2000.jsonl``
#: in the plans repository), the classifier rows from J2d's ``dens.jsonl`` (``rows/X106/J2/
#: evidence/2026-11-19-j2d/rev9/``); each row's guard counts, with the route's family and bins, and
#: its traced peak.
ROUTE_CELLS: dict[str, tuple[int, int]] = {
    "ltmle_continuous_index": (320, 169_586_005),
    "sdr_continuous_index": (320, 362_604_524),
    "ltmle_continuous_classifier": (0, 6_934_545),
    "sdr_continuous_classifier": (0, 10_641_843),
}


@pytest.mark.parametrize("family", sorted(ROUTE_CELLS))
def test_a_fitted_route_cell_lies_between_the_margin_and_the_stated_bound(family: str) -> None:
    bins, peak = ROUTE_CELLS[family]
    counts = FitCounts(
        estimator="SDR" if family.startswith("sdr") else "LTMLE",
        family=family,
        n=2_000,
        n_times=4,
        plans=4,
        horizons=(4,),
        causes=1,
        folds=3,
        width=9,
        bins=bins,
        msm=False,
        repeats=1,
        bootstrap=0,
        jobs=1,
        panel_bytes=184_000,
    )
    assert _memory_guard.in_range(counts)
    predicted = _memory_guard.predicted_bytes(counts)
    assert 1.35 * peak <= predicted <= _memory_guard.FACTOR * peak


# ------------------------------------------------------------------ count sources

ROLES: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _learners(**changes: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "n_folds": 1,
        "random_state": 3,
        "simultaneous": False,
        **changes,
    }


def _previewed(engine: Any, data: Any, **roles: Any) -> FitCounts:
    """The counts the preview's preflight reads (it fits nothing)."""
    with support.captured_counts() as seen, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine._planned_fits(data, **roles)
    assert len(seen) == 1
    return seen[0]


def _two_node(n: int = 400) -> pd.DataFrame:
    return make_longitudinal(n=n, seed=4)[0]


def test_each_setting_moves_its_own_count() -> None:
    frame = _two_node()
    regimens = {"always": 1, "never": 0}
    base = _previewed(LTMLE(regimens, **_learners()), frame, **ROLES)
    assert (base.family, base.n, base.n_times, base.plans, base.horizons) == (
        "ltmle",
        400,
        2,
        2,
        (2,),
    )
    assert (base.folds, base.repeats, base.bootstrap, base.jobs, base.censored) == (
        1,
        1,
        0,
        1,
        True,
    )
    assert base.width == 2 + 1 + 2  # W1, W2, L2 and both treatments
    three = _previewed(LTMLE({**regimens, "first": (1, 0)}, **_learners()), frame, **ROLES)
    assert three.plans == 3
    assert _previewed(LTMLE(regimens, **_learners(n_folds=3)), frame, **ROLES).folds == 3
    assert (
        _previewed(LTMLE(regimens, **_learners(n_folds=3, repeats=2)), frame, **ROLES).repeats == 2
    )
    assert _previewed(LTMLE(regimens, **_learners(n_bootstrap=3)), frame, **ROLES).bootstrap == 3
    assert _previewed(LTMLE(regimens, **_learners(n_jobs=2)), frame, **ROLES).jobs == 2
    frame["X1"] = np.random.default_rng(0).normal(size=len(frame))
    wider = _previewed(
        LTMLE(regimens, **_learners()), frame, **{**ROLES, "baseline": ["W1", "W2", "X1"]}
    )
    assert wider.width == base.width + 1
    assert _previewed(SDR(regimens, **_learners()), frame, **ROLES).family == "sdr"


def test_horizons_causes_and_bins_reach_the_counts() -> None:
    engine, _ = support.build(
        {"est": "ltmle_surv", "T": 4, "R": 2, "folds": 1, "H": 2, "J": 2}, 500
    )
    frame, blocks = support.panel(500, 4, survival=True, causes=2)
    nodes = range(1, 5)
    survival = _previewed(
        engine,
        frame,
        outcome={"event": [f"Y{t}" for t in nodes], "other": [f"D{t}" for t in nodes]},
        treatment=[f"A{t}" for t in nodes],
        baseline=["W1", "W2"],
        time_varying=blocks,
    )
    assert (survival.horizons, survival.steps, survival.causes) == ((3, 4), 7, 2)
    with support.captured_counts() as seen, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        support.build({"est": "mtp", "T": 2, "R": 2, "folds": 1, "H": 1, "B": 12}, 500)[1]()
    assert seen[0].family == "ltmle_continuous"
    assert seen[0].bins >= 12


def test_the_draw_is_the_written_out_sum_of_its_terms() -> None:
    counts = _ltmle(msm=True, horizons=(2, 4, 8), causes=2)
    c = _memory_guard.CONSTANTS["ltmle"]
    n, t, r, s, h, j, p, f = 20_000, 8, 2, 14, 3, 2, 17, 3
    expected = c["c0"] + 8 * n * (
        c["RSJ"] * r * s * j
        + c["RT"] * r * t
        + c["RHJ"] * r * h * j
        + c["p"] * p
        + c["Rp"] * r * p
        + c["FRT"] * f * r * t
        + c["RS_msm"] * r * s
    )
    assert _memory_guard.draw_bytes(counts) == pytest.approx(expected, rel=1e-12)


def test_a_three_level_node_counts_three_levels() -> None:
    frame = _two_node()
    observed = frame["A1"].notna()
    frame.loc[observed, "A1"] = np.random.default_rng(1).integers(0, 3, int(observed.sum()))
    ltmle = _previewed(LTMLE({"two": (2, 1), "never": 0}, **_learners()), frame, **ROLES)
    assert ltmle.levels == 3
    ipsi = _previewed(IPSI([Incremental({0.0: 2.0})], **_learners()), frame, **ROLES)
    assert ipsi.levels == 3
    assert "3 treatment levels" in str(_memory_guard.outside_range(ipsi))
    assert _previewed(LTMLE({"always": 1}, **_learners()), _two_node(), **ROLES).levels == 2


def _arm_design(label: Any, horizon: int, rows: Any) -> np.ndarray:
    del horizon
    return np.column_stack([np.ones(len(rows)), np.full(len(rows), float(label == "always"))])


def test_a_working_model_adds_exactly_its_term() -> None:
    frame = _two_node()
    regimens = {"always": 1, "never": 0}
    model = MSM(design=cast(Any, _arm_design), terms=("(intercept)", "always"), design_kind="known")
    with_msm = _previewed(LTMLE(regimens, msm=model, **_learners()), frame, **ROLES)
    without = _previewed(LTMLE(regimens, **_learners()), frame, **ROLES)
    assert (with_msm.msm, without.msm) == (True, False)
    term = 8 * with_msm.n * _memory_guard.CONSTANTS["ltmle"]["RS_msm"]
    assert _memory_guard.draw_bytes(with_msm) - _memory_guard.draw_bytes(without) == (
        pytest.approx(term * with_msm.plans * with_msm.steps, rel=1e-12)
    )


def test_a_separable_fit_is_its_own_family() -> None:
    separable = {"est": "hz_separable", "T": 3, "R": 4, "folds": 1, "H": 3, "J": 2}
    one_node = {"est": "hz_point", "T": 3, "R": 2, "folds": 1, "H": 3, "J": 2}
    for case, family, plans in (
        (separable, "hazard_separable", 4),
        (one_node, "hazard_one_node", 2),
    ):
        engine, _ = support.build(case, 400)
        data = support._point(400, 3, 2)
        counts = _previewed(engine, data)
        assert (counts.family, counts.plans) == (family, plans)


def test_a_mixed_panel_counts_as_continuous_without_raising() -> None:
    from tests import discrete_law_longitudinal_mtp as law
    from tests import longitudinal_mtp as mtp

    data = LongitudinalData.from_frame(law.mixed_frame(), **mtp.MIXED_COLUMNS)
    counts = _memory_guard.fit_counts(
        LTMLE({"natural": 0}, density_bins=8),
        data,
        plans=1,
        horizons=(data.n_times,),
        folds=1,
        repeats=1,
        paths=None,
        multinode=True,
        later=True,
    )
    assert (counts.family, counts.bins, counts.levels) == ("ltmle_continuous", 8, 2)
    # No mixed cell was calibrated, so the fit is out of range; the prediction still applies.
    assert counts.mixed
    outside = _memory_guard.outside_range(counts)
    assert outside is not None and "a mixed treatment node" in outside
    assert _memory_guard.outside_range(replace(counts, mixed=False)) != outside


def _width_panels() -> dict[str, LongitudinalData]:
    from tests import discrete_law_longitudinal_mtp as law
    from tests import longitudinal_mtp as mtp

    frame, blocks = support.panel(300, 3, survival=False)
    roles: dict[str, Any] = {
        "outcome": "Y",
        "treatment": ["A1", "A2", "A3"],
        "baseline": ["W1", "W2"],
        "time_varying": blocks,
    }
    binary = LongitudinalData.from_frame(frame, **roles)
    levels = frame.copy()
    rng = np.random.default_rng(7)
    for t in (1, 2, 3):
        levels[f"A{t}"] = rng.integers(0, 3, size=len(levels)).astype(float)
    dosed = frame.copy()
    for t in (1, 2, 3):
        dose = rng.normal(size=len(dosed))
        dose[rng.random(len(dosed)) < 0.3] = 0.0
        dosed[f"A{t}"] = dose
    continuous = LongitudinalData.from_frame(
        dosed,
        **roles,
        continuous_treatment=["A1", "A2", "A3"],
        dose_atoms={"A1": [0.0], "A2": [0.0], "A3": [0.0]},
    )
    held_frame, _ = make_point_survival(n=300, seed=3, n_times=3, layout="wide", time_varying=True)
    held = LongitudinalTreatment(
        outcome=["Y1", "Y2", "Y3"],
        treatment="A",
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"], ["L3"]],
        censoring=["C1", "C2", "C3"],
    ).prepare(held_frame)
    return {
        "binary": binary,
        "categorical": LongitudinalData.from_frame(levels, **roles),
        "continuous": continuous,
        "mixed": LongitudinalData.from_frame(law.mixed_frame(), **mtp.MIXED_COLUMNS),
        "markov": binary.with_markov_order(1),
        "held": held,
    }


def test_the_counted_width_is_the_design_width_on_every_kind_of_panel() -> None:
    panels = _width_panels()
    assert any(panels["continuous"].node_atoms(t) for t in (1, 2, 3))
    assert any(panels["held"].is_held_node(t) for t in (1, 2, 3))
    assert panels["mixed"].node_kind(1) == "mixed"
    for name, data in panels.items():
        for t in range(1, data.n_times + 1):
            for current in (False, True):
                design = data.history_design(t, include_current=current)
                assert data.history_width(t, include_current=current) == design.shape[1], (
                    name,
                    t,
                    current,
                )


def _markov_panel() -> LongitudinalData:
    """Twenty covariates at node 1, one at each later node, a four-level node 1, order 1."""
    rng = np.random.default_rng(0)
    n, n_times = 300, 4
    d: dict[str, np.ndarray] = {"W1": rng.normal(size=n)}
    order, blocks = ["W1"], []
    for t in range(1, n_times + 1):
        names = [f"L{t}_{k}" for k in range(20 if t == 1 else 1)]
        for name in names:
            d[name] = rng.normal(size=n)
        levels = 4 if t == 1 else 2
        d[f"A{t}"] = rng.integers(0, levels, size=n).astype(float)
        order += [*names, f"A{t}"]
        blocks.append(names)
    d["Y"] = rng.binomial(1, 0.4, size=n).astype(float)
    order.append("Y")
    frame = pd.DataFrame({name: d[name] for name in order})
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=[f"A{t}" for t in range(1, n_times + 1)],
        baseline=["W1"],
        time_varying=blocks,
    )
    return data.with_markov_order(1)


def test_a_markov_order_counts_the_widest_node_not_the_last() -> None:
    data = _markov_panel()
    widths = [data.history_width(t, include_current=True) for t in range(1, 5)]
    assert widths == [24, 26, 8, 6]
    counts = _memory_guard.fit_counts(
        LTMLE({"never": 0}),
        data,
        plans=1,
        horizons=(4,),
        folds=1,
        repeats=1,
        paths=None,
        multinode=True,
        later=True,
    )
    assert counts.width == 26
    assert counts.markov
    assert _memory_guard.outside_range(counts) is not None


def _counting_cost(n: int) -> int:
    frame, blocks = support.panel(n, 4, survival=False)
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=[f"A{t}" for t in range(1, 5)],
        baseline=["W1", "W2"],
        time_varying=blocks,
    )
    engine = LTMLE({"always": 1, "never": 0})
    settings: dict[str, Any] = {
        "plans": 2,
        "horizons": (4,),
        "folds": 3,
        "repeats": 1,
        "paths": None,
        "multinode": True,
        "later": True,
    }
    _memory_guard.fit_counts(engine, data, **settings)
    tracemalloc.start()
    try:
        _memory_guard.fit_counts(engine, data, **settings)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_reading_the_counts_costs_the_same_at_any_n() -> None:
    small, large = _counting_cost(2_000), _counting_cost(20_000)
    assert large < 64 * 1024
    assert abs(large - small) <= 4 * 1024


# ------------------------------------------------------------------ the refusal


class _Touched(Exception):
    """A learner was fitted."""


class _UntouchedLinear(LinearRegression):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise _Touched


class _UntouchedLogistic(LogisticRegression):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise _Touched


def _untouched() -> dict[str, Any]:
    return {
        "outcome_learner": _UntouchedLinear(),
        "treatment_learner": _UntouchedLogistic(),
        "censoring_learner": _UntouchedLogistic(),
        "pseudo_learner": _UntouchedLinear(),
    }


def _hazard_paths(**changes: Any) -> tuple[Callable[..., Any], pd.DataFrame, dict[str, Any]]:
    frame, _ = support.panel(400, 3, survival=True)
    roles: dict[str, Any] = {
        "outcome": ["Y1", "Y2", "Y3"],
        "treatment": ["A1", "A2", "A3"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], [], []],
    }

    def make(**settings: Any) -> Any:
        settings = {**_learners(survival_model="hazard", targeting="one_step"), **settings}
        settings.pop("pseudo_learner", None)
        return LTMLE({"always": 1, "never": 0}, **settings)

    return make, frame, roles


def _engines() -> dict[str, tuple[Callable[..., Any], Any, dict[str, Any]]]:
    frame = _two_node()
    return {
        "LTMLE": (lambda **kw: LTMLE({"always": 1, "never": 0}, **kw), frame, ROLES),
        "SDR": (lambda **kw: SDR({"always": 1, "never": 0}, **kw), frame, ROLES),
        "IPSI": (
            lambda **kw: IPSI([Incremental(1.0), Incremental(2.0)], **kw),
            frame,
            ROLES,
        ),
        "path fit": _hazard_paths(),
    }


def _predicted(make: Callable[..., Any], data: Any, roles: dict[str, Any], **kw: Any) -> int:
    """The prediction of one call, read from its preflight (a learner then stops it)."""
    with support.captured_counts() as seen, pytest.raises(_Touched):
        make(**{**_learners(**kw), **_untouched()}, max_memory_bytes=2**62).fit(data, **roles)
    return _memory_guard.predicted_bytes(seen[0])


@pytest.mark.parametrize("name", ["LTMLE", "SDR", "IPSI", "path fit"])
def test_a_budget_one_byte_short_refuses_before_any_learner(name: str) -> None:
    make, data, roles = _engines()[name]
    predicted = _predicted(make, data, roles)
    with pytest.raises(LongitudinalMemoryError) as caught, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make(**{**_learners(), **_untouched()}, max_memory_bytes=predicted - 1).fit(data, **roles)
    error = caught.value
    assert (error.predicted, error.budget) == (predicted, predicted - 1)
    text = str(error)
    cls = "LTMLE" if name == "path fit" else name
    assert f"{predicted:,} bytes" in text
    assert f"{predicted - 1:,} bytes" in text
    assert "which max_memory_bytes sets" in text
    assert f"(an argument of {cls}, and a keyword of estimate()" in str(error.refusal.alternative)
    assert "No learner was fitted" in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = make(**_learners(), max_memory_bytes=predicted).fit(data, **roles)
    assert result.estimates


def test_the_path_cap_refuses_first_and_the_memory_guard_at_its_prediction() -> None:
    # One budget feeds both guards.  One byte under the path cap's own prediction, the path
    # cap (H17) refuses before the memory guard reads any count; at that prediction the path
    # cap admits and the memory guard, which margins it, refuses.
    n_times = 6
    nodes = range(1, n_times + 1)
    frame, _ = support.panel(N, n_times, survival=True)
    data = LongitudinalData.from_frame(
        frame,
        outcome=[f"Y{t}" for t in nodes],
        treatment=[f"A{t}" for t in nodes],
        baseline=["W1", "W2"],
        time_varying=[[] for _ in nodes],
    )
    coin = {"coin": DynamicRegimen("coin", tuple(support._coin(t, True) for t in nodes))}
    peak = hazard_module.path_peak(N, n_times, 2**n_times, 1, hazard_module._hazard_columns(data))
    budget = math.ceil(peak)

    def fit(max_memory_bytes: int) -> None:
        settings = _learners(survival_model="hazard", **_untouched())
        settings.pop("pseudo_learner")
        LTMLE(coin, max_memory_bytes=max_memory_bytes, **settings).fit(data)

    with support.captured_counts() as seen, pytest.raises(CapabilityError, match="64 treatment"):
        fit(budget - 1)
    assert seen == []
    with support.captured_counts() as seen, pytest.raises(LongitudinalMemoryError) as caught:
        fit(budget)
    assert seen[0].paths == 2**n_times
    predicted = caught.value.predicted
    assert predicted is not None and predicted >= 1.35 * peak


def test_the_default_budget_is_the_memory_available_now(monkeypatch: pytest.MonkeyPatch) -> None:
    make, data, roles = _engines()["LTMLE"]
    predicted = _predicted(make, data, roles)
    monkeypatch.setattr(memory_module, "available_memory", lambda: predicted - 1)
    with pytest.raises(LongitudinalMemoryError, match="the memory available now"):
        make(**{**_learners(), **_untouched()}).fit(data, **roles)
    monkeypatch.setattr(memory_module, "available_memory", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert make(**_learners()).fit(data, **roles).estimates


def test_the_bootstrap_replicates_are_counted_before_the_first_learner() -> None:
    make, data, roles = _engines()["LTMLE"]
    one = _predicted(make, data, roles)
    two = _predicted(make, data, roles, n_bootstrap=2)
    assert two > one
    with pytest.raises(LongitudinalMemoryError):
        make(**{**_learners(n_bootstrap=2), **_untouched()}, max_memory_bytes=one).fit(
            data, **roles
        )


def test_a_replicate_skips_the_guard_its_call_already_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make, data, roles = _engines()["LTMLE"]
    calls = iter([2**62])
    # The first read admits the call; every later one would refuse it.
    monkeypatch.setattr(memory_module, "available_memory", lambda: next(calls, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = make(**_learners(n_bootstrap=2)).fit(data, **roles)
    assert result.bootstrap is not None
    # Mutation control: a replicate that runs the guard again is refused.
    real = estimator_module._SequentialEstimator._replicate_estimator

    def checking(self: Any) -> Any:
        replicate = real(self)
        replicate._check_memory = True
        return replicate

    monkeypatch.setattr(estimator_module._SequentialEstimator, "_replicate_estimator", checking)
    refused = []
    real_refusal = _memory_guard.refusal

    def recording(*args: Any, **kwargs: Any) -> LongitudinalMemoryError:
        refused.append(args)
        return real_refusal(*args, **kwargs)

    monkeypatch.setattr(_memory_guard, "refusal", recording)
    calls = iter([2**62])
    # run_bootstrap drops a replicate that raises, and raises once every replicate has.
    with (
        pytest.raises(RuntimeError, match="all 2 bootstrap replicates failed"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        make(**_learners(n_bootstrap=2)).fit(data, **roles)
    assert len(refused) == 2


def _refit_on_twice_the_units(result: Any) -> Any:
    index = np.concatenate([np.arange(result.data.n), np.arange(result.data.n)])
    return result.estimator.refit(result.data.subset(index))


def test_a_refit_reads_the_stored_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    make, data, roles = _engines()["LTMLE"]
    predicted = _predicted(make, data, roles)
    monkeypatch.setattr(memory_module, "available_memory", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = make(**_learners(), max_memory_bytes=predicted).fit(data, **roles)
    with pytest.raises(LongitudinalMemoryError) as caught:
        _refit_on_twice_the_units(result)
    assert caught.value.budget == predicted
    # Mutation control: a refit template without the budget admits the larger refit.
    real = estimator_module._SequentialEstimator._refit_template

    def forgetting(self: Any) -> Any:
        template = real(self)
        template.max_memory_bytes = None
        return template

    monkeypatch.setattr(estimator_module._SequentialEstimator, "_refit_template", forgetting)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = make(**_learners(), max_memory_bytes=predicted).fit(data, **roles)
        assert _refit_on_twice_the_units(result).estimates


# ------------------------------------------------------------------ preview and facade


def test_the_preview_refuses_as_the_fit_does_and_notes_the_same_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make, data, roles = _engines()["SDR"]
    predicted = _predicted(make, data, roles)
    plan = make(**_learners())._planned_fits(data, **roles)
    note = next(text for text in plan.notes if "memory guard" in text)
    assert f"({predicted:,} bytes)" in note
    assert "calibrated with GLM, logistic and linear learners" in note
    assert "worker count" in note
    with pytest.raises(LongitudinalMemoryError) as previewed:
        make(**_learners(), max_memory_bytes=predicted - 1)._planned_fits(data, **roles)
    with pytest.raises(LongitudinalMemoryError) as fitted, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make(**{**_learners(), **_untouched()}, max_memory_bytes=predicted - 1).fit(data, **roles)
    assert str(previewed.value) == str(fitted.value)


def _study() -> tuple[CausalStudy, Any]:
    study = CausalStudy(_two_node(), design=LongitudinalTreatment(**ROLES))
    effect = study.identify(RegimeContrast(regimens={"always": 1, "never": 0}, reference="never"))
    return study, effect


def test_the_facade_takes_the_budget_and_names_it_in_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A study reaches the budget as ``max_memory_bytes=``, and its refusal says so."""
    study, effect = _study()
    settings = _learners(n_folds=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        facade = effect.estimate("tmle", **settings)
        budgeted = effect.estimate("tmle", **settings, max_memory_bytes=2**62)
    for name, estimate in facade.estimates.items():
        assert estimate.psi == budgeted.estimates[name].psi
        assert estimate.std_error == budgeted.estimates[name].std_error
    for call in (effect.estimate, effect.planned_fits):
        with pytest.raises(LongitudinalMemoryError) as caught:
            call("tmle", **settings, max_memory_bytes=1)
        assert "which max_memory_bytes sets" in str(caught.value)
        alternative = str(caught.value.refusal.alternative)
        assert "keyword of estimate() or a field of Runtime on a study" in alternative
        assert caught.value.budget == 1
    monkeypatch.setattr(memory_module, "available_memory", lambda: 1)
    with pytest.raises(LongitudinalMemoryError) as caught:
        effect.estimate("tmle", **settings)
    assert "the memory available now" in str(caught.value)
    assert study.data is not None


# ------------------------------------------------------------------ construction and scope


def test_sdr_and_ipsi_check_the_budget_as_ltmle_does() -> None:
    cases: tuple[tuple[type, Any], ...] = (
        (LTMLE, 0),
        (SDR, 0),
        (IPSI, "1"),
        (SDR, True),
        (IPSI, -5),
    )
    for build, value in cases:
        with pytest.raises(DataError, match="max_memory_bytes must be a positive integer"):
            build({"always": 1}, max_memory_bytes=value)
    assert SDR({"always": 1}, max_memory_bytes=10**9).max_memory_bytes == 10**9
    assert IPSI([Incremental(2.0)], max_memory_bytes=10**9).max_memory_bytes == 10**9


def test_only_ltmle_sdr_and_ipsi_run_the_guard() -> None:
    assert LTMLE._retention_guard and SDR._retention_guard and IPSI._retention_guard
    assert not SurvivalCTMLE._retention_guard
    assert not SurvivalDRTMLE._retention_guard
    assert not estimator_module._SequentialEstimator._retention_guard


def test_the_message_omits_counts_at_one_and_names_bins() -> None:
    counts = _fixed(family="ltmle_continuous", estimator="LTMLE", jobs=1, folds=3, bins=20)
    error = _memory_guard.refusal(_memory_guard.predict(counts), 2**30, explicit=True)
    text = str(error)
    assert "repeat" not in text.split(" is predicted")[0]
    assert "n_jobs" not in text.split(" is predicted")[0]
    assert "20 density bins" in text
    assert "fewer density_bins=" in str(error.refusal.alternative)
    assert 'ratio="classifier" (no binned density)' in str(error.refusal.alternative)
    assert (
        "At one worker it can exceed the true peak by up to 3.5 times, more on fits under about "
        "20 MiB with the simultaneous band on, and more with n_jobs > 1 or a liblinear treatment learner."
        in text
    )
    # The classifier route builds no binned density, so neither bins remedy appears.
    classifier = replace(counts, family="ltmle_continuous_classifier", bins=0)
    other = _memory_guard.refusal(_memory_guard.predict(classifier), 2**20, explicit=True)
    assert "density_bins" not in str(other.refusal.alternative)
    assert "ratio=" not in str(other.refusal.alternative)
    assert re.search(r"the budget is 1\.0 GiB \(1,073,741,824 bytes\)", text)
