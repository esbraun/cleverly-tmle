"""Seed sweep of the plan-means fit that docs/examples/weekly-outcomes.ipynb shows.

Each seed ``s`` draws ``make_weekly_orders(12_000, seed=s)`` and runs the page's in-sample LTMLE
of ``RegimeMean`` over the three plans (``never``, ``always``, ``win_back``): nine weekly order
shares. It records each share, its standard error, whether its 95% interval holds the truth from
``weekly_orders_truth()``, and its support label. It also records, for each seed, whether the
``never`` week-3 interval and the ``win_back`` against ``never`` week-3 contrast interval of the
page's ``RegimeContrast`` fit miss together.

Usage: ``python means_sweep.py <first seed> <count> <workers>``. The page cites
``python means_sweep.py 3000 400 2``. Seed 3000 is the page's draw, so its fit is the first row of
``means_sweep.csv``. Outputs: ``means_sweep.csv`` (one row per seed) and ``means_sweep.log`` (the
summary). The companion ``sweep.py`` covers the page's other fits; this file is separate so that
``sweep.csv`` and ``sweep.log`` stay as they ran.
Every worker is single-threaded (``OMP_NUM_THREADS=1``, ``n_jobs=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 12_000
WEEKS = (1, 2, 3)
PLANS = ("never", "always", "win_back")


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        CausalStudy,
        CrossFitting,
        LongitudinalTreatment,
        ModelSpec,
        RegimeContrast,
        RegimeMean,
        Repeated,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_weekly_orders, offer_after_no_order
    from cleverly.longitudinal import DynamicRegimen

    frame, truth = make_weekly_orders(N, seed=seed)
    design = LongitudinalTreatment(
        outcome=Repeated(["order_1", "order_2", "order_3"]),
        treatment=("offer_1", "offer_2", "offer_3"),
        baseline=("tenure", "order_0"),
        time_varying=(("engaged_1",), ("engaged_2",), ("engaged_3",)),
        censoring=("tracked_1", "tracked_2", "tracked_3"),
    )
    study = CausalStudy(frame, design=design)
    win_back = DynamicRegimen(
        "win_back",
        tuple(
            (lambda history, column=f"order_{week - 1}": offer_after_no_order(history[column]))
            for week in WEEKS
        ),
        rule_kind="known",
    )
    plans = {"never": (0, 0, 0), "always": (1, 1, 1), "win_back": win_back}
    logistic = LogisticRegression(max_iter=1000, random_state=42)
    models = ModelSpec(
        outcome_learner=logistic,
        pseudo_learner=LinearRegression(),
        treatment_learner=logistic,
        censoring_learner=logistic,
    )
    method = TMLEMethod(
        models=models,
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=42, n_jobs=1),
    )
    means = study.identify(RegimeMean(plans)).estimate(method=method)
    row: dict = {"seed": seed}
    for plan in PLANS:
        for week in WEEKS:
            key = f"ey_regimen[{plan} @ t={week}]"
            estimate = means[key]
            low, high = estimate.ci
            stem = f"{plan}_{week}"
            row[f"{stem}_psi"] = estimate.psi
            row[f"{stem}_se"] = estimate.std_error
            row[f"{stem}_covers"] = bool(low <= truth[key] <= high)
            row[f"{stem}_label"] = estimate.support.label
    contrast = study.identify(RegimeContrast(plans, reference="never")).estimate(method=method)
    key = "ate_regimen[win_back vs never @ t=3]"
    low, high = contrast[key].ci
    row["win_back_contrast_3_covers"] = bool(low <= truth[key] <= high)
    return row


def summarize(frame, truth) -> str:
    import numpy as np

    lines = [f"draws: {len(frame)} (seeds {frame['seed'].min()} to {frame['seed'].max()})"]
    for plan in PLANS:
        for week in WEEKS:
            stem = f"{plan}_{week}"
            key = f"ey_regimen[{plan} @ t={week}]"
            bias = frame[f"{stem}_psi"] - truth[key]
            lines.append(
                f"mean_{stem}: coverage {frame[f'{stem}_covers'].mean():.3f}, "
                f"held on {int(frame[f'{stem}_covers'].sum())} of {len(frame)}, "
                f"mean bias {bias.mean():+.4f} (MC se {bias.std() / np.sqrt(len(bias)):.4f}), "
                f"SE ratio {frame[f'{stem}_se'].mean() / frame[f'{stem}_psi'].std():.3f}, "
                f"labels {sorted(frame[f'{stem}_label'].unique())}"
            )
    never_miss = ~frame["never_3_covers"]
    contrast_miss = ~frame["win_back_contrast_3_covers"]
    lines.append(
        f"week 3: never mean misses on {int(never_miss.sum())}, win_back contrast misses on "
        f"{int(contrast_miss.sum())}, both on {int((never_miss & contrast_miss).sum())} "
        f"of {len(frame)}"
    )
    return "\n".join(lines)


def main() -> None:
    import pandas as pd

    from cleverly.datasets import weekly_orders_truth

    first, count, workers = (int(value) for value in sys.argv[1:4])
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, range(first, first + count)))
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(HERE / "means_sweep.csv", index=False)
    text = summarize(frame, weekly_orders_truth())
    text += f"\nwall time {time.time() - start:.0f} s with {workers} workers"
    (HERE / "means_sweep.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
