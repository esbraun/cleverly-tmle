"""Seed sweep of the fits that docs/examples/conditional-effects.ipynb shows.

Each seed ``s`` draws the notebook's coupon data, ``simulate_coupons(n=8000, seed=s)``, and runs
the notebook's EP curve at the declared schedule, its TMLE projection on ``Cosine(["age"], 3)``
with the band over ages 20 to 80, and its held-out rule score, with every ``random_state`` set
to ``s``.  The code of each fit is the notebook's, with ``seed=1`` and ``random_state=0``
replaced by ``s``.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 1000 200 4``.  Outputs: ``sweep.csv`` (one row per seed), which
``summarize.py`` reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
AGES = np.linspace(20, 80, 7)
COST = 0.08


def true_uplift(age):
    """The coupon's true effect on the chance of a purchase, by age (the notebook's)."""
    return 0.25 / (1.0 + np.exp((age - 45.0) / 8.0))


def simulate_coupons(n, seed):
    """The notebook's generator, unchanged."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    spend = rng.uniform(0, 1, n)
    visits = rng.poisson(3, n).clip(0, 10)
    chance = 1.0 / (1.0 + np.exp(-(-1.0 + 2.5 * spend + 0.02 * (age - 50))))
    coupon = rng.binomial(1, chance)
    buys = 0.15 + 0.4 * spend + 0.02 * visits + coupon * true_uplift(age)
    bought = rng.binomial(1, buys)
    return pd.DataFrame(
        {"age": age, "spend": spend, "visits": visits, "coupon": coupon, "bought": bought}
    )


def waves_for(n_rows):
    return max(1, 3 + int(np.floor(np.log2(n_rows / 1000))))


def shape_truth(shape, by_age):
    grid = pd.DataFrame({"age": np.linspace(18, 80, 100_001)})
    terms = shape.evaluate(grid, by_age)
    coef = np.linalg.lstsq(terms, true_uplift(grid["age"]), rcond=None)[0]
    return coef, shape.evaluate(pd.DataFrame({"age": AGES}), by_age) @ coef


def one(seed):
    warnings.simplefilter("ignore")
    from cleverly import CausalStudy, ConditionalContrast, PointTreatment, RegimeContrast
    from cleverly.bases import Cosine, Modifiers, Sieve
    from cleverly.interventions import Rule, Static

    learners = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
    }
    frame = simulate_coupons(8_000, seed)
    design = PointTreatment(outcome="bought", treatment="coupon", adjustment=("age", "spend", "visits"))
    by_age = Modifiers(["age"], domains={"age": (18, 80)})
    study = CausalStudy(frame, design=design)
    row = {"seed": seed}

    curve = study.identify(ConditionalContrast(by_age)).estimate(
        "ep", sieve=Sieve.default(by_age, size=waves_for), outcome_by_arm=True, **learners,
        random_state=seed,
    )
    grid = np.arange(18, 81)
    row["ep_rmse"] = float(np.sqrt(np.mean((curve.predict({"age": grid}) - true_uplift(grid)) ** 2)))
    row["ep_status"] = curve.status

    shape = Cosine(["age"], 3)
    coef, at_ages = shape_truth(shape, by_age)
    projection = study.identify(ConditionalContrast(by_age, model=shape)).estimate(
        outcome_by_arm=True, **learners, random_state=seed
    )
    table = projection.to_frame()
    for j, name in enumerate(["intercept", "cos1", "cos2", "cos3"]):
        row[f"{name}_covers"] = bool(table["ci_lower"].iloc[j] <= coef[j] <= table["ci_upper"].iloc[j])
    shape_curve = projection.effect_curve(grid={"age": AGES})
    point = shape_curve.predict({"age": AGES}, interval=True)
    band = shape_curve.band
    for k, age in enumerate(AGES):
        row[f"point_covers_{int(age)}"] = bool(point["ci_lower"].iloc[k] <= at_ages[k] <= point["ci_upper"].iloc[k])
    row["band_covers"] = bool(np.all((band.lower <= at_ages) & (at_ages <= band.upper)))
    row["heterogeneity_p"] = float(projection.heterogeneity_test().p_value)

    half = frame.sample(frac=0.5, random_state=seed).index
    first, second = frame.loc[half], frame.drop(index=half)
    first_curve = CausalStudy(first, design=design).identify(ConditionalContrast(by_age)).estimate(
        "ep", sieve=Sieve.default(by_age, size=waves_for), outcome_by_arm=True, **learners,
        random_state=seed,
    )
    below = grid[first_curve.predict({"age": grid}) <= COST]
    cutoff = int(below.min()) - 1 if below.size else int(grid.max())
    above = grid[first_curve.predict({"age": grid}) > COST]
    row["old_cutoff"] = int(above.max()) if above.size else np.nan
    row["cutoff"] = cutoff
    rule = Rule(lambda rows: (rows["age"] <= cutoff).astype(float), name="rule", rule_kind="known")
    scored = CausalStudy(second, design=design).estimate(
        RegimeContrast((rule, Static(0, name="no coupon")), reference="no coupon"),
        **learners, random_state=seed,
    )
    rise = scored.to_frame().iloc[0]
    coupon_cost = COST * (second["age"] <= cutoff).mean()
    true_net = float(np.mean((true_uplift(second["age"]) - COST) * (second["age"] <= cutoff)))
    row["net"] = float(rise["psi"] - coupon_cost)
    row["true_net"] = true_net
    row["net_covers"] = bool(rise["ci_lower"] - coupon_cost <= true_net <= rise["ci_upper"] - coupon_cost)
    return row


def main():
    first, count, workers = (int(value) for value in sys.argv[1:4])
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(one, range(first, first + count)))
    pd.DataFrame(rows).to_csv(HERE / "sweep.csv", index=False)


if __name__ == "__main__":
    main()
