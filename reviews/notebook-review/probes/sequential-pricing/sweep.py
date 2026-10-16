"""Seed sweep of the fits that docs/examples/sequential-pricing.ipynb shows.

Each seed ``s`` draws ``make_sequential_pricing(10_000, seed=s)`` and runs the notebook's code: the
in-sample LTMLE of the renewal at the third price and of the revenue, with the price rise and the
price cut against the engine's own prices; the logistic regression of the renewal on usage,
engagement, and the three prices, with and without engagement, whose predictions use the plan's
prices; the factor ladder (a 6%, a 12%, and a 25% rise); and the node-3 support of the renewal
fit. Truths come from ``pricing_truth()``.

Usage: ``python sweep.py <first seed> <count> <workers>``. The page cites
``python sweep.py 1000 200 4``. The notebook draws seed 1000, the first of that range, by a rule
fixed before the page ran with it, so its fits are the first row of ``sweep.csv``. Outputs:
``sweep.csv`` (one row per seed) and ``sweep.log``. Every worker is single-threaded
(``OMP_NUM_THREADS=1``, ``n_jobs=1``).
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
N = 10_000
PLANS = ("price_up", "price_down")
LADDER = ("up_6", "price_up", "up_25")


def one_seed(seed: int) -> dict:
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    from cleverly import (
        CausalStudy,
        CrossFitting,
        LongitudinalTreatment,
        ModelSpec,
        RegimeContrast,
        Runtime,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import (
        PRICE_BANDS,
        PRICE_PLANS,
        make_sequential_pricing,
        price_rise,
        pricing_protocol,
        pricing_truth,
    )
    from cleverly.learners import thread_limit

    truth = pricing_truth()
    frame, _ = make_sequential_pricing(N, seed=seed)
    protocol = pricing_protocol()
    reference = "natural_prices"

    def design(outcome):
        return LongitudinalTreatment(
            outcome=outcome,
            treatment=[f"price_{k}" for k in (1, 2, 3)],
            baseline=["usage"],
            time_varying=[[f"engaged_{k}"] for k in (1, 2, 3)],
            continuous_treatment=PRICE_BANDS,
        )

    logistic = LogisticRegression(max_iter=1000, random_state=42)
    ratio_model = make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=1e6, tol=1e-8, max_iter=100_000, random_state=42),
    )

    def method(*, outcome_learner, q_bounds=None):
        return TMLEMethod(
            models=ModelSpec(
                outcome_learner=outcome_learner,
                pseudo_learner=LinearRegression(),
                treatment_learner=ratio_model,
                ratio="classifier",
            ),
            targeting=Targeting(q_bounds=q_bounds) if q_bounds else Targeting(),
            cross_fitting=CrossFitting(enabled=False),
            runtime=Runtime(random_state=1000, n_jobs=1),
        )

    def fit(outcome, plans, **kwargs):
        study = CausalStudy(frame, design=design(outcome), protocol=protocol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = study.identify(RegimeContrast(plans, reference=reference)).estimate(
                method=method(**kwargs)
            )
        return result, sorted({w.category.__name__ for w in caught})

    started = time.time()
    row: dict = {"seed": seed}
    renewal, renewal_warnings = fit("renewed", PRICE_PLANS, outcome_learner=logistic)
    revenue, revenue_warnings = fit(
        "revenue", PRICE_PLANS, outcome_learner=LinearRegression(), q_bounds=(0.0, 10.0)
    )
    row["renewal_warnings"] = ";".join(renewal_warnings)
    row["revenue_warnings"] = ";".join(revenue_warnings)
    for result, prefix, kind in ((renewal, "y", "ate_regimen"), (revenue, "r", "ate_revenue")):
        for plan in PLANS:
            key = f"ate_regimen[{plan} vs {reference}]"
            estimate = result[key]
            tkey = f"{kind}[{plan} vs {reference}]"
            low, high = estimate.ci
            row[f"{prefix}_{plan}_psi"] = estimate.psi
            row[f"{prefix}_{plan}_se"] = estimate.std_error
            row[f"{prefix}_{plan}_lo"], row[f"{prefix}_{plan}_hi"] = low, high
            row[f"{prefix}_{plan}_truth"] = truth[tkey]
            row[f"{prefix}_{plan}_label"] = estimate.support.label
    support = renewal.diagnostics.support().to_frame()
    for plan in PLANS:
        last = support[(support["regimen"] == plan) & (support["time"] == 3)].iloc[0]
        row[f"maxw_{plan}"] = float(last["max_weight"])
        row[f"essr_{plan}"] = float(last["effective_n"] / last["n_followed"])
        row[f"trunc_{plan}"] = float(last["share_truncated"])

    # The failure mode: one logistic regression of the renewal on the recorded variables, with the
    # prices replaced by the plan's prices.
    def planned(plan, node):
        price = frame[f"price_{node}"].to_numpy()
        engaged = frame[f"engaged_{node}"].to_numpy()
        if plan == "price_up":
            return np.where(price < (8 + 2 * engaged) / 1.12, 1.12 * price, price)
        return np.where(price < (4 + 2 * engaged) / 0.85, price, 0.85 * price)

    prices = [f"price_{k}" for k in (1, 2, 3)]
    for label, columns in (
        ("adjust", ["usage", "engaged_1", "engaged_2", "engaged_3", *prices]),
        ("omit", ["usage", *prices]),
    ):
        with thread_limit():
            model = LogisticRegression(max_iter=2000, C=1e6).fit(frame[columns], frame["renewed"])
            for plan in PLANS:
                moved = frame[columns].copy()
                for node in (1, 2, 3):
                    moved[f"price_{node}"] = planned(plan, node)
                row[f"{label}_{plan}"] = float(
                    np.mean(
                        model.predict_proba(moved)[:, 1] - model.predict_proba(frame[columns])[:, 1]
                    )
                )

    # The support step: a 6%, a 12%, and a 25% rise against the engine's own prices.
    rises = {
        reference: PRICE_PLANS[reference],
        "up_6": price_rise(1.06),
        "price_up": PRICE_PLANS["price_up"],
        "up_25": price_rise(1.25),
    }
    ladder, ladder_warnings = fit("renewed", rises, outcome_learner=logistic)
    row["ladder_warnings"] = ";".join(ladder_warnings)
    ladder_support = ladder.diagnostics.support().to_frame()
    for plan in LADDER:
        key = f"ate_regimen[{plan} vs {reference}]"
        estimate = ladder[key]
        low, high = estimate.ci
        row[f"ladder_{plan}_psi"] = estimate.psi
        row[f"ladder_{plan}_se"] = estimate.std_error
        row[f"ladder_{plan}_lo"], row[f"ladder_{plan}_hi"] = low, high
        row[f"ladder_{plan}_truth"] = truth[key]
        row[f"ladder_{plan}_label"] = estimate.support.label
        last = ladder_support[
            (ladder_support["regimen"] == plan) & (ladder_support["time"] == 3)
        ].iloc[0]
        row[f"ladder_{plan}_maxw"] = float(last["max_weight"])
        row[f"ladder_{plan}_essr"] = float(last["effective_n"] / last["n_followed"])
        row[f"ladder_{plan}_trunc"] = float(last["share_truncated"])
    row["seconds"] = time.time() - started
    return row


def main() -> None:
    import pandas as pd

    first, count, workers = (int(value) for value in sys.argv[1:4])
    seeds = list(range(first, first + count))
    rows = []
    start = time.time()
    log = (HERE / "sweep.log").open("w")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(one_seed, seeds), start=1):
            rows.append(row)
            log.write(f"{done}/{count} seed {row['seed']} at {time.time() - start:.0f}s\n")
            log.flush()
            pd.DataFrame(rows).to_csv(HERE / "sweep.csv", index=False, float_format="%.10g")
    log.close()


if __name__ == "__main__":
    main()
