"""Bias of the notebook's LTMLE fit on fresh draws, per contrast and per level.

Each seed ``s`` draws ``make_sequential_offers(20_000, seed=s)`` and fits the declared-mechanism
in-sample LTMLE of ``docs/examples/sequential-offers.ipynb`` twice: once as
``RegimeMean`` (the five levels) and once as ``RegimeContrast`` against ``none`` (the four
contrasts). Truths come from ``offers_truth()``. The seeds are disjoint from the 1000 to 1099
range of ``sweep.py``.

Usage: ``python bias_check.py <first seed> <count> <workers>``. The page cites
``python bias_check.py 2000 400 4``. Outputs: ``bias_check.csv`` (one row per seed) and
``bias_check.log`` (progress). ``summarize_bias.py`` prints the table the page quotes.
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
CONTRASTS = ("discount", "shipping", "win_back", "win_back_ship")


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        CausalStudy,
        CrossFitting,
        LongPanel,
        ModelSpec,
        RegimeContrast,
        RegimeMean,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import (
        OFFER_WEEKS,
        make_sequential_offers,
        offers_protocol,
        win_back,
        win_back_then_ship,
    )
    from cleverly.longitudinal import DynamicRegimen, each_node

    panel, _ = make_sequential_offers(20_000, seed=seed)
    design = LongPanel(
        unit="customer",
        time="week",
        treatment="offer",
        baseline=["tenure"],
        time_varying=["engaged"],
        censoring="consent",
        outcome="full_price",
        treatment_probabilities={"none": "p_none", "discount": "p_discount", "shipping": "p_shipping"},
    )

    def rule(label, offer_for):
        return DynamicRegimen(
            label,
            each_node(lambda h, week: offer_for(h[f"engaged@{week}"]), OFFER_WEEKS),
            rule_kind="known",
        )

    plans = {
        "none": ("none",) * OFFER_WEEKS,
        "discount": ("discount",) * OFFER_WEEKS,
        "shipping": ("shipping",) * OFFER_WEEKS,
        "win_back": rule("win_back", win_back),
        "win_back_ship": rule("win_back_ship", win_back_then_ship),
    }
    logistic = LogisticRegression(max_iter=1000, random_state=42)
    runtime = Runtime(random_state=42, n_jobs=1)
    methods = {
        "ltmle": TMLEMethod(
            models=ModelSpec(
                outcome_learner=logistic,
                pseudo_learner=LinearRegression(),
                censoring_learner=logistic,
            ),
            cross_fitting=CrossFitting(enabled=False),
            runtime=runtime,
        ),
    }
    study = CausalStudy(panel, design=design, protocol=offers_protocol())
    levels = study.identify(RegimeMean(plans))
    contrasts = study.identify(RegimeContrast(plans, reference="none"))
    row: dict = {"seed": seed}
    for name, method in methods.items():
        level_fit = levels.estimate(method=method)
        contrast_fit = contrasts.estimate(method=method)
        for plan in plans:
            estimate = level_fit[f"ey_regimen[{plan}]"]
            row[f"{name}_level_{plan}_psi"] = estimate.psi
            row[f"{name}_level_{plan}_se"] = estimate.std_error
        for plan in CONTRASTS:
            estimate = contrast_fit[f"ate_regimen[{plan} vs none]"]
            row[f"{name}_contrast_{plan}_psi"] = estimate.psi
            row[f"{name}_contrast_{plan}_se"] = estimate.std_error
    return row


def main() -> None:
    import pandas as pd

    first, count, workers = (int(value) for value in sys.argv[1:4])
    rows = []
    start = time.time()
    log = (HERE / "bias_check.log").open("w")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(one_seed, range(first, first + count)), start=1):
            rows.append(row)
            log.write(f"{done}/{count} seed {row['seed']} at {time.time() - start:.0f}s\n")
            log.flush()
            pd.DataFrame(rows).to_csv(HERE / "bias_check.csv", index=False, float_format="%.10g")
    log.close()


if __name__ == "__main__":
    main()
