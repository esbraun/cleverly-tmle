"""Seed sweep of the two rankings that docs/examples/ranking-exposures.ipynb shows.

Each seed ``s`` draws ``make_discharge_practices(n=4000, seed=s)`` and runs the notebook's two
``variable_importance`` calls (every other practice adjusted for, then
``adjust_for_other_candidates=False`` with the navigation offer added to the covariates), with the
notebook's estimator and ``random_state=s``.  It also records each practice's simple difference in
the share staying home.  The page's own seed, 0, is not among the swept seeds.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 1000 400 4``.  Outputs: ``sweep.csv`` (one row per seed and practice) and
``sweep.log`` (each rate the page reads, with a 95% Wilson interval, and the mean false discovery
proportion with its standard error).  Every worker is single-threaded.

This checks readings of one page on one simulated law.  It is not a registered study, and it does
not validate the false discovery claim in general (roadmap item X125 plans that study).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ["discharge_risk", "age", "prior_utilization"]
NULLS = ("teach_back", "home_health_referral")
NAVIGATION = "transition_navigation"
LEVEL = 0.05


def one_seed(seed: int) -> list[dict]:
    from sklearn.linear_model import LogisticRegression

    from cleverly import variable_importance
    from cleverly.datasets import PRACTICES, make_discharge_practices
    from cleverly.estimators import TMLE

    frame, truth = make_discharge_practices(n=4_000, seed=seed)
    estimator = TMLE(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=5,
        random_state=seed,
    )
    shared = dict(outcome="stayed_home", candidates=list(PRACTICES), estimator=estimator)
    ranking = variable_importance(frame, covariates=COVARIATES, **shared).to_frame()
    chain = variable_importance(
        frame,
        covariates=[*COVARIATES, NAVIGATION],
        adjust_for_other_candidates=False,
        **shared,
    ).to_frame()
    rows = []
    for name in PRACTICES:
        first = ranking.set_index("candidate").loc[name]
        second = chain.set_index("candidate").loc[name]
        by_level = frame.groupby(name)["stayed_home"].mean()
        rows.append(
            {
                "seed": seed,
                "practice": name,
                "adjusted_truth": truth[f"adjusted[{name}]"],
                "total_truth": truth[f"total[{name}]"],
                "psi": first["psi"],
                "std_err": first["std_err"],
                "ci_lower": first["ci_lower"],
                "ci_upper": first["ci_upper"],
                "p_value": first["p_value"],
                "p_value_adjusted": first["p_value_adjusted"],
                "chain_psi": second["psi"],
                "chain_std_err": second["std_err"],
                "chain_ci_lower": second["ci_lower"],
                "chain_ci_upper": second["ci_upper"],
                "naive": by_level.loc[1.0] - by_level.loc[0.0],
            }
        )
    return rows


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959964
    centre = (hits + z * z / 2) / (n + z * z)
    half = z * math.sqrt(hits * (n - hits) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def summarize(table: pd.DataFrame) -> list[str]:
    seeds = table["seed"].nunique()
    lines = [f"seeds: {seeds} (from {table['seed'].min()} to {table['seed'].max()})"]

    def rate(label: str, flags: pd.Series) -> None:
        hits, n = int(flags.sum()), int(flags.size)
        low, high = wilson(hits, n)
        lines.append(f"{label}: {hits}/{n} = {hits / n:.3f} (95% Wilson {low:.3f} to {high:.3f})")

    for name, rows in table.groupby("practice", sort=False):
        covered = rows["ci_lower"].le(rows["adjusted_truth"]) & rows["ci_upper"].ge(
            rows["adjusted_truth"]
        )
        rate(f"default interval covers adjusted truth, {name}", covered)
        chain_covered = rows["chain_ci_lower"].le(rows["total_truth"]) & rows[
            "chain_ci_upper"
        ].ge(rows["total_truth"])
        rate(f"second-fit interval covers total truth, {name}", chain_covered)
        wrong_side = (rows["naive"] - rows["total_truth"]).abs() > rows["std_err"]
        rate(f"simple difference more than one SE from the truth, {name}", wrong_side)
    navigation = table[table["practice"] == NAVIGATION]
    excludes = navigation["ci_upper"].lt(navigation["total_truth"]) | navigation["ci_lower"].gt(
        navigation["total_truth"]
    )
    rate("default navigation interval excludes the total truth", excludes)
    booking = table[table["practice"] == "followup_booked"]
    close = (booking["psi"] - booking["chain_psi"]).abs() < booking["std_err"]
    rate("booking's two estimates differ by less than one SE", close)
    nulls = table[table["practice"].isin(NULLS)].groupby("seed")
    rate("both no-effect practices at adjusted p above 0.05", nulls["p_value_adjusted"].min() > LEVEL)
    rate("some no-effect practice at raw p below 0.05", nulls["p_value"].min() < LEVEL)
    rate("some no-effect practice at adjusted p below 0.05", nulls["p_value_adjusted"].min() < LEVEL)
    proportions = []
    for _, rows in table.groupby("seed"):
        called = rows[rows["p_value_adjusted"] < LEVEL]
        false = called["practice"].isin(NULLS).sum()
        proportions.append(false / len(called) if len(called) else 0.0)
    series = pd.Series(proportions)
    lines.append(
        f"mean false discovery proportion: {series.mean():.4f} "
        f"(standard error {series.std(ddof=1) / math.sqrt(len(series)):.4f}); "
        f"Benjamini-Hochberg bound at 0.05 with 2 of 5 nulls: {LEVEL * 2 / 5:.4f}"
    )
    return lines


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(one_seed, range(first, first + count)))
    table = pd.DataFrame([row for rows in results for row in rows])
    table.to_csv(HERE / "sweep.csv", index=False, float_format="%.6g")
    lines = summarize(table)
    (HERE / "sweep.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
