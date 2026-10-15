"""Seed sweep of the fits that docs/examples/sequential-offers.ipynb shows.

Each seed ``s`` draws ``make_sequential_offers(20_000, seed=s)`` and runs the notebook's code: the
in-sample LTMLE of the four contrasts with the declared offer mechanism; the crude follower
contrasts of the association step; the blind-outcome comparison (LTMLE and g-computation with
outcome regressions that read no history); the three paired comparisons; the synthetic deployment
check with a pilot drawn at seed ``s + 100_000``; the stagewise support; and the benchmark that
drops engagement from the history of the static plans. Truths come from ``offers_truth()``.

Usage: ``python sweep.py <first seed> <count> <workers>``. The page cites
``python sweep.py 1000 100 3``. The notebook draws seed 1000, the first of that range, by a
rule fixed before the page ran with it, so its fits are the first row of ``sweep.csv``. Outputs:
``sweep.csv`` (one row per seed) and ``sweep.log``.
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
PAIRS = (("win_back", "discount"), ("win_back", "shipping"), ("win_back_ship", "win_back"))
STATIC = ("none", "discount", "shipping")


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        CausalStudy,
        CrossFitting,
        GComputationMethod,
        LongPanel,
        ModelSpec,
        RegimeContrast,
        RegimeMean,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import (
        OFFER_PLANS,
        OFFER_WEEKS,
        make_sequential_offers,
        offers_protocol,
        offers_truth,
        win_back,
        win_back_then_ship,
    )
    from cleverly.longitudinal import DynamicRegimen, each_node

    truth = offers_truth()
    panel, _ = make_sequential_offers(20_000, seed=seed)
    protocol = offers_protocol()
    logged = {"none": "p_none", "discount": "p_discount", "shipping": "p_shipping"}

    def design(*, time_varying=("engaged",)):
        return LongPanel(
            unit="customer",
            time="week",
            treatment="offer",
            baseline=["tenure"],
            time_varying=list(time_varying),
            censoring="consent",
            outcome="full_price",
            treatment_probabilities=logged,
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
    ltmle = TMLEMethod(
        models=ModelSpec(
            outcome_learner=logistic,
            pseudo_learner=LinearRegression(),
            censoring_learner=logistic,
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=42, n_jobs=1),
    )
    study = CausalStudy(panel, design=design(), protocol=protocol)
    effect = study.identify(RegimeContrast(plans, reference="none"))
    result = effect.estimate(method=ltmle)
    row: dict = {"seed": seed}
    for contrast in CONTRASTS:
        estimate = result[f"ate_regimen[{contrast} vs none]"]
        row[f"ltmle_{contrast}_psi"] = estimate.psi
        row[f"ltmle_{contrast}_se"] = estimate.std_error
        row[f"ltmle_{contrast}_lo"], row[f"ltmle_{contrast}_hi"] = estimate.ci

    # The association step: the crude contrasts of the customers who followed each plan in all
    # three weeks and still consented after week 3.
    wide = panel.pivot(index="customer", columns="week")
    kept = wide[wide[("consent", OFFER_WEEKS)] == 1]
    engaged_first = kept[("engaged", 1)].astype(int).to_numpy()
    row["kept_engaged_week1_share"] = float(engaged_first.mean())
    crude = {}
    for name, plan_rule in OFFER_PLANS.items():
        followed = np.ones(len(kept), dtype=bool)
        for week in range(1, OFFER_WEEKS + 1):
            engaged = kept[("engaged", week)].astype(int).to_numpy()
            followed &= kept[("offer", week)].to_numpy() == np.asarray(
                plan_rule(engaged), dtype=object
            )
        crude[name] = float(kept[("full_price", 1)][followed].mean())
        row[f"crude_{name}_followers"] = int(followed.sum())
        row[f"crude_{name}_engaged_week1_share"] = float(engaged_first[followed].mean())
    for contrast in CONTRASTS:
        row[f"crude_{contrast}_psi"] = crude[contrast] - crude["none"]

    # The blind-outcome comparison: outcome regressions that read no history.
    blind = ModelSpec(
        outcome_learner=DummyClassifier(strategy="prior"),
        pseudo_learner=DummyRegressor(),
        censoring_learner=logistic,
    )
    blind_ltmle = effect.estimate(
        method=TMLEMethod(models=blind, cross_fitting=ltmle.cross_fitting, runtime=ltmle.runtime)
    )
    blind_gcomp = effect.estimate(method=GComputationMethod(models=blind, runtime=ltmle.runtime))
    for contrast in CONTRASTS:
        key = f"ate_regimen[{contrast} vs none]"
        row[f"blind_{contrast}_psi"] = blind_ltmle[key].psi
        row[f"blind_{contrast}_se"] = blind_ltmle[key].std_error
        row[f"blind_{contrast}_lo"], row[f"blind_{contrast}_hi"] = blind_ltmle[key].ci
        row[f"blindg_{contrast}_psi"] = blind_gcomp[key].psi

    for better, other in PAIRS:
        keys = [f"ate_regimen[{better} vs none]", f"ate_regimen[{other} vs none]"]
        paired = result.contrast(lambda values: values[0] - values[1], keys)
        label = f"{better}_minus_{other}"
        row[f"{label}_psi"] = paired.psi
        row[f"{label}_se"] = paired.std_error
        row[f"{label}_lo"], row[f"{label}_hi"] = paired.ci
        row[f"{label}_unpaired_se"] = float(
            np.hypot(result[keys[0]].std_error, result[keys[1]].std_error)
        )

    pilot, _ = make_sequential_offers(10_000, seed=seed + 100_000, policy="win_back")
    outcome = pilot.loc[pilot["week"] == OFFER_WEEKS, "full_price"]
    level = study.identify(RegimeMean(plans)).estimate(method=ltmle)["ey_regimen[win_back]"]
    row["level_psi"], row["level_se"] = level.psi, level.std_error
    row["pilot_mean"] = float(outcome.mean())
    row["pilot_se"] = float(outcome.std() / np.sqrt(len(outcome)))

    support = result.assess().report("support").to_frame()
    row["min_kish_ratio"] = float((support["effective_n"] / support["n_followed"]).min())
    row["min_followed"] = int(support["n_followed"].min())
    row["max_weight"] = float(support["max_weight"].max())
    row["max_truncated"] = float(support["share_truncated"].max())
    followed_at_end = support[support["time"] == OFFER_WEEKS].set_index("regimen")["n_followed"]
    for name in plans:
        row[f"followed_{name}"] = int(followed_at_end[name])

    static = {name: plans[name] for name in STATIC}
    full = study.identify(RegimeContrast(static, reference="none")).estimate(method=ltmle)
    reduced = (
        CausalStudy(panel, design=design(time_varying=()), protocol=protocol)
        .identify(RegimeContrast(static, reference="none"))
        .estimate(method=ltmle)
    )
    for contrast in ("discount", "shipping"):
        key = f"ate_regimen[{contrast} vs none]"
        row[f"drop_{contrast}_move"] = reduced[key].psi - full[key].psi
        row[f"drop_{contrast}_se"] = full[key].std_error
    row["truth_check"] = truth["ate_regimen[win_back vs none]"]
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
