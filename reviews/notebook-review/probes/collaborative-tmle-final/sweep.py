"""Seed sweep of the fits that docs/examples/collaborative-tmle.ipynb shows.

Each seed ``s`` draws ``make_instrument(n=2000, seed=s)`` and runs the notebook's Step 6 fit
(greedy C-TMLE, linear Q, in sample, three selection folds and two inner folds), its Step 8 fit
(plain TMLE with the same learners), its Step 9 constant-Q pair (``DummyRegressor`` outcome
model, plain and C-TMLE), and its Step 11 regression with the HC0 standard error.  Every
``random_state`` is set to ``s``.  The code of each fit is the notebook's, with ``44`` replaced
by ``s``.  One fit is the probe's own: plain TMLE on the design that leaves out
``queue_lottery_draw`` (``p_ni_*``), which isolates what the instrument in g does to the plain
interval.  The true ATE of this law is 1 (``truth.py``).

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 5000 600 12``.  Outputs: ``sweep.csv`` (one row per seed) and ``sweep.log``,
which ``summarize.py`` reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``,
``n_jobs=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ["baseline_readiness", "queue_lottery_draw", "social_support"]
RENAME = {
    "Y": "transition_score",
    "A": "transition_navigation",
    "W1": "baseline_readiness",
    "W2": "queue_lottery_draw",
    "W3": "social_support",
}


def one_seed(seed: int) -> dict:
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        ATE,
        CausalStudy,
        CollaborativeTMLEMethod,
        CrossFitting,
        ModelSpec,
        PointTreatment,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_instrument

    start = time.perf_counter()
    frame, truth = make_instrument(n=2_000, seed=seed)
    frame = frame.rename(columns=RENAME)
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=tuple(COVARIATES),
        ),
    ).identify(ATE(reference=0))
    models = ModelSpec(
        outcome_learner=LinearRegression(n_jobs=1),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
    )
    folds = CrossFitting(enabled=False)
    runtime = Runtime(random_state=seed, n_jobs=1)
    collaborative_method = CollaborativeTMLEMethod(
        models=models,
        cross_fitting=folds,
        runtime=runtime,
        strategy="greedy",
        selection_folds=3,
        selection_inner_folds=2,
    )
    plain_method = TMLEMethod(models=models, cross_fitting=folds, runtime=runtime)
    collaborative = effect.estimate(method=collaborative_method)
    plain = effect.estimate(method=plain_method)
    weak_models = ModelSpec(
        outcome_learner=DummyRegressor(),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
    )
    weak_plain = effect.estimate(method=replace(plain_method, models=weak_models))
    weak_collaborative = effect.estimate(method=replace(collaborative_method, models=weak_models))
    no_instrument = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=("baseline_readiness", "social_support"),
        ),
    ).identify(ATE(reference=0))
    no_instrument_point = no_instrument.estimate(method=plain_method)["ate"]
    ni_low, ni_high = no_instrument_point.ci
    selection = collaborative.diagnostics.nuisance_models().selection
    weak_selection = weak_collaborative.diagnostics.nuisance_models().selection

    # The notebook's Step 11 regression: the offer coefficient and its HC0 standard error.
    offer = frame["transition_navigation"].to_numpy()
    score = frame["transition_score"].to_numpy()
    design = np.column_stack([np.ones(len(frame)), frame[COVARIATES].to_numpy()])
    full = np.column_stack([design, offer])
    residual = score - full @ np.linalg.lstsq(full, score, rcond=None)[0]
    offer_part = offer - design @ np.linalg.lstsq(design, offer, rcond=None)[0]
    coefficient = (offer_part * score).sum() / (offer_part**2).sum()
    hc0 = np.sqrt((offer_part**2 * residual**2).sum()) / (offer_part**2).sum()

    point, plain_point = collaborative["ate"], plain["ate"]
    weak_point, weak_plain_point = weak_collaborative["ate"], weak_plain["ate"]
    plugin_low, plugin_high = point.plugin_interval
    weak_low, weak_high = weak_point.plugin_interval
    plain_low, plain_high = plain_point.ci
    weak_plain_low, weak_plain_high = weak_plain_point.ci
    support = plain.diagnostics.support()
    return {
        "seed": seed,
        "truth": truth["ate"],
        "c_psi": point.psi,
        "c_pse": point.plugin_std_error,
        "c_plugin_covers": plugin_low <= truth["ate"] <= plugin_high,
        "c_selected": "|".join(selection.selected_covariates),
        "c_gap10": selection.cv_risk[1] - selection.cv_risk[0],
        "c_first_step": selection.path[1][0] if len(selection.path) > 1 else "",
        "c_risk_rises": any(b > a for a, b in zip(selection.train_risk, selection.train_risk[1:])),
        "p_psi": plain_point.psi,
        "p_se": plain_point.std_error,
        "p_covers": plain_low <= truth["ate"] <= plain_high,
        "p_below": support.tail_mass[0.1]["below"],
        "p_above": support.tail_mass[0.1]["above"],
        "p_ni_psi": no_instrument_point.psi,
        "p_ni_se": no_instrument_point.std_error,
        "p_ni_covers": ni_low <= truth["ate"] <= ni_high,
        "wp_psi": weak_plain_point.psi,
        "wp_se": weak_plain_point.std_error,
        "wp_covers": weak_plain_low <= truth["ate"] <= weak_plain_high,
        "wc_psi": weak_point.psi,
        "wc_pse": weak_point.plugin_std_error,
        "wc_plugin_covers": weak_low <= truth["ate"] <= weak_high,
        "wc_selected": "|".join(weak_selection.selected_covariates),
        "ols": coefficient,
        "hc0": hc0,
        "seconds": time.perf_counter() - start,
    }


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    import cleverly

    log = [f"cleverly {cleverly.__version__} from {cleverly.__file__}"]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, range(first, first + count)))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "sweep.csv", index=False, lineterminator="\n")
    log.append(f"seeds {first} to {first + count - 1}, {workers} workers")
    log.append(f"wall time {time.perf_counter() - start:.0f} s")
    (HERE / "sweep.log").write_text("\n".join(log) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
