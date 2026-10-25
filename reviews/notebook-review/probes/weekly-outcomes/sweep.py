"""Seed sweep of the fits that docs/examples/weekly-outcomes.ipynb shows.

Each seed ``s`` draws ``make_weekly_orders(12_000, seed=s)`` and runs the page's fits: the
in-sample LTMLE of the weekly curve of ``always`` and ``win_back`` against ``never``; the same
curve by cross-fitted SDR; the in-sample IPSI curve of doubling the odds of an offer against the
natural course; the separate end-of-study fits per week that leave the earlier orders out of the
history; the total over the three weeks, with its paired standard error from ``contrast`` and the
unpaired one that adds the weekly variances; and the crude comparison of followers. Truths come
from ``weekly_orders_truth()``.

Usage: ``python sweep.py <first seed> <count> <workers>``. The page cites
``python sweep.py 3000 400 3``. The notebook draws seed 3000, the first of that range, by a rule
fixed before the page ran with it, so its fits are the first row of ``sweep.csv``. Outputs:
``sweep.csv`` (one row per seed) and ``sweep.log`` (the summary).
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
PLANS = ("always", "win_back")


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        CausalStudy,
        CrossFitting,
        IncrementalEffect,
        IPSIMethod,
        LongitudinalTreatment,
        ModelSpec,
        RegimeContrast,
        Repeated,
        Runtime,
        SDRMethod,
        TMLEMethod,
    )
    from cleverly.datasets import make_weekly_orders, offer_after_no_order
    from cleverly.interventions import Incremental
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
    logistic = LogisticRegression(max_iter=1000)
    models = ModelSpec(
        outcome_learner=logistic,
        pseudo_learner=LinearRegression(),
        treatment_learner=logistic,
        censoring_learner=logistic,
    )
    runtime = Runtime(random_state=42, n_jobs=1)
    in_sample = CrossFitting(enabled=False)
    effect = study.identify(RegimeContrast(plans, reference="never"))
    row: dict = {"seed": seed}

    fits = {
        "ltmle": effect.estimate(
            method=TMLEMethod(models=models, cross_fitting=in_sample, runtime=runtime)
        ),
        "sdr": effect.estimate(
            method=SDRMethod(models=models, cross_fitting=CrossFitting(n_folds=5), runtime=runtime)
        ),
    }
    for label, fit in fits.items():
        for plan in PLANS:
            for week in WEEKS:
                key = f"ate_regimen[{plan} vs never @ t={week}]"
                estimate = fit[key]
                low, high = estimate.ci
                stem = f"{label}_{plan}_{week}"
                row[f"{stem}_psi"] = estimate.psi
                row[f"{stem}_se"] = estimate.std_error
                row[f"{stem}_covers"] = bool(low <= truth[key] <= high)
                row[f"{stem}_label"] = estimate.support.label

    tilt = study.identify(
        IncrementalEffect([Incremental(1.0), Incremental(2.0)], reference="natural course")
    ).estimate(method=IPSIMethod(models=models, cross_fitting=in_sample, runtime=runtime))
    for week in WEEKS:
        key = f"ate_regimen[odds x2 vs natural course @ t={week}]"
        estimate = tilt[key]
        low, high = estimate.ci
        row[f"ipsi_{week}_psi"] = estimate.psi
        row[f"ipsi_{week}_se"] = estimate.std_error
        row[f"ipsi_{week}_covers"] = bool(low <= truth[key] <= high)
        row[f"ipsi_{week}_label"] = estimate.support.label

    # Separate end-of-study fits per week, with the earlier orders left out of the history.
    for week in WEEKS:
        separate = LongitudinalTreatment(
            outcome=f"order_{week}",
            treatment=tuple(f"offer_{t}" for t in range(1, week + 1)),
            baseline=("tenure", "order_0"),
            time_varying=tuple((f"engaged_{t}",) for t in range(1, week + 1)),
            censoring=tuple(f"tracked_{t}" for t in range(1, week + 1)),
        )
        fit = (
            CausalStudy(frame, design=separate)
            .identify(
                RegimeContrast({"never": (0,) * week, "always": (1,) * week}, reference="never")
            )
            .estimate(method=TMLEMethod(models=models, cross_fitting=in_sample, runtime=runtime))
        )
        estimate = fit["ate_regimen[always vs never]"]
        low, high = estimate.ci
        key = f"ate_regimen[always vs never @ t={week}]"
        row[f"separate_{week}_psi"] = estimate.psi
        row[f"separate_{week}_covers"] = bool(low <= truth[key] <= high)

    # The total over the three weeks: paired from the joint fit, and unpaired.
    for plan in PLANS:
        keys = [f"ate_regimen[{plan} vs never @ t={week}]" for week in WEEKS]
        total = fits["ltmle"].contrast(lambda values: sum(values), keys, name=f"{plan} total")
        unpaired = float(np.sqrt(sum(fits["ltmle"][key].std_error ** 2 for key in keys)))
        true_total = sum(truth[key] for key in keys)
        low, high = total.ci
        row[f"total_{plan}_psi"] = total.psi
        row[f"total_{plan}_se"] = total.std_error
        row[f"total_{plan}_unpaired_se"] = unpaired
        row[f"total_{plan}_covers"] = bool(low <= true_total <= high)
        row[f"total_{plan}_unpaired_covers"] = bool(
            total.psi - 1.959964 * unpaired <= true_total <= total.psi + 1.959964 * unpaired
        )

    # The crude comparison: customers tracked through each week who followed a plan so far.
    for plan in PLANS:
        for week in WEEKS:
            kept = frame[frame[f"tracked_{week}"] == 1]
            follow = {name: np.ones(len(kept), dtype=bool) for name in ("never", plan)}
            for t in range(1, week + 1):
                offer = kept[f"offer_{t}"].to_numpy()
                previous = kept[f"order_{t - 1}"].to_numpy()
                follow["never"] &= offer == 0
                wanted = 1.0 if plan == "always" else offer_after_no_order(previous)
                follow[plan] &= offer == wanted
            outcome = kept[f"order_{week}"].to_numpy()
            row[f"crude_{plan}_{week}"] = (
                outcome[follow[plan]].mean() - outcome[follow["never"]].mean()
            )
    return row


def summarize(frame, truth) -> str:
    import numpy as np

    lines = [f"draws: {len(frame)} (seeds {frame['seed'].min()} to {frame['seed'].max()})"]
    for label in ("ltmle", "sdr"):
        for plan in PLANS:
            for week in WEEKS:
                stem = f"{label}_{plan}_{week}"
                key = f"ate_regimen[{plan} vs never @ t={week}]"
                bias = frame[f"{stem}_psi"] - truth[key]
                lines.append(
                    f"{stem}: coverage {frame[f'{stem}_covers'].mean():.3f}, "
                    f"mean bias {bias.mean():+.4f} (MC se {bias.std() / np.sqrt(len(bias)):.4f}), "
                    f"SE ratio {frame[f'{stem}_se'].mean() / frame[f'{stem}_psi'].std():.3f}, "
                    f"labels {sorted(frame[f'{stem}_label'].unique())}"
                )
    for week in WEEKS:
        key = f"ate_regimen[odds x2 vs natural course @ t={week}]"
        bias = frame[f"ipsi_{week}_psi"] - truth[key]
        lines.append(
            f"ipsi_{week}: coverage {frame[f'ipsi_{week}_covers'].mean():.3f}, "
            f"mean bias {bias.mean():+.5f} (MC se {bias.std() / np.sqrt(len(bias)):.5f}), "
            f"SE ratio {frame[f'ipsi_{week}_se'].mean() / frame[f'ipsi_{week}_psi'].std():.3f}, "
            f"labels {sorted(frame[f'ipsi_{week}_label'].unique())}"
        )
    for week in WEEKS:
        key = f"ate_regimen[always vs never @ t={week}]"
        bias = frame[f"separate_{week}_psi"] - truth[key]
        lines.append(
            f"separate_{week}: mean bias {bias.mean():+.4f} "
            f"(MC se {bias.std() / np.sqrt(len(bias)):.4f}), "
            f"coverage {frame[f'separate_{week}_covers'].mean():.3f}, "
            f"below truth on {(bias < 0).sum()} of {len(bias)}"
        )
    for plan in PLANS:
        lines.append(
            f"total_{plan}: paired coverage {frame[f'total_{plan}_covers'].mean():.3f}, "
            f"unpaired coverage {frame[f'total_{plan}_unpaired_covers'].mean():.3f}, "
            f"paired SE mean {frame[f'total_{plan}_se'].mean():.4f}, "
            f"unpaired SE mean {frame[f'total_{plan}_unpaired_se'].mean():.4f}, "
            f"empirical SD {frame[f'total_{plan}_psi'].std():.4f}, "
            f"paired above unpaired on "
            f"{(frame[f'total_{plan}_se'] > frame[f'total_{plan}_unpaired_se']).sum()} "
            f"of {len(frame)}"
        )
    for plan in PLANS:
        for week in WEEKS:
            key = f"ate_regimen[{plan} vs never @ t={week}]"
            bias = frame[f"crude_{plan}_{week}"] - truth[key]
            lines.append(
                f"crude_{plan}_{week}: mean bias {bias.mean():+.4f} "
                f"(MC se {bias.std() / np.sqrt(len(bias)):.4f})"
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
    frame.to_csv(HERE / "sweep.csv", index=False)
    text = summarize(frame, weekly_orders_truth())
    text += f"\nwall time {time.time() - start:.0f} s with {workers} workers"
    (HERE / "sweep.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
