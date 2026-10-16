"""Which part of the revenue fit carries its bias? A probe of the outcome learner.

The sweep shows a bias of the revenue contrast of the price rise with the linear outcome learner
that the page uses. This probe refits the revenue contrast on the first 60 seeds of the sweep with
a quadratic outcome learner (degree-2 terms, scaled, then least squares) and nothing else changed.
If the bias goes away, the misspecified linear regression carries it.

Usage: ``python revenue_learner.py <workers> > revenue_learner.log``.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

SEEDS = range(1000, 1060)
PLANS = ("price_up", "price_down")


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
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
        pricing_protocol,
        pricing_truth,
    )

    truth = pricing_truth()
    frame, _ = make_sequential_pricing(10_000, seed=seed)
    design = LongitudinalTreatment(
        outcome="revenue",
        treatment=[f"price_{k}" for k in (1, 2, 3)],
        baseline=["usage"],
        time_varying=[[f"engaged_{k}"] for k in (1, 2, 3)],
        continuous_treatment=PRICE_BANDS,
    )
    effect = CausalStudy(frame, design=design, protocol=pricing_protocol()).identify(
        RegimeContrast(PRICE_PLANS, reference="natural_prices")
    )
    ratio_model = make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=1e6, max_iter=2000, random_state=42),
    )
    quadratic = make_pipeline(
        PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression()
    )
    row = {"seed": seed}
    for name, learner in (("linear", LinearRegression()), ("quadratic", quadratic)):
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=learner,
                pseudo_learner=LinearRegression(),
                treatment_learner=ratio_model,
                ratio="classifier",
            ),
            targeting=Targeting(q_bounds=(0.0, 10.0)),
            cross_fitting=CrossFitting(enabled=False),
            runtime=Runtime(random_state=1000, n_jobs=1),
        )
        result = effect.estimate(method=method)
        for plan in PLANS:
            estimate = result[f"ate_regimen[{plan} vs natural_prices]"]
            exact = truth[f"ate_revenue[{plan} vs natural_prices]"]
            row[f"{name}_{plan}_psi"] = estimate.psi
            row[f"{name}_{plan}_cover"] = int(estimate.ci[0] <= exact <= estimate.ci[1])
            row[f"{name}_{plan}_truth"] = exact
    return row


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    with ProcessPoolExecutor(int(sys.argv[1])) as pool:
        frame = pd.DataFrame(list(pool.map(one_seed, SEEDS)))
    n = len(frame)
    print(f"draws {n}, seeds {frame['seed'].min()} to {frame['seed'].max()}, n = 10000")
    for plan in PLANS:
        for name in ("linear", "quadratic"):
            psi = frame[f"{name}_{plan}_psi"]
            exact = frame[f"{name}_{plan}_truth"].iloc[0]
            print(
                f"  revenue {plan:11s} {name:9s} outcome learner: "
                f"bias {psi.mean() - exact:+.4f} (MC SE {psi.std(ddof=1) / np.sqrt(n):.4f}); "
                f"coverage {int(frame[f'{name}_{plan}_cover'].sum())} of {n}"
            )
