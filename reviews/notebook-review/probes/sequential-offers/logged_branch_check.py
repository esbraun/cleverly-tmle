"""Check the generator against ``offers_truth()`` with no fitted nuisance.

Two checks, both with known probabilities and no model.

1. ``policy=`` draws. For each plan, ``DRAWS`` cohorts of ``N`` customers are drawn with the
   plan deployed, and the mean full-price order rate is compared with the truth.
2. The logged branch (``policy=None``), which is the estimation sample of the notebook. A
   Horvitz-Thompson estimate weights each customer who followed the plan and stayed consenting by
   one over the product of the logged offer probabilities and the true consent probabilities
   ``_consent``. The weights use the logged ``p_*`` columns, so a mismatch between the logged law
   and the truth shows as a gap, in standard errors of the mean over cohorts.

Usage: ``python logged_branch_check.py > logged_branch_check.log``. The seeds are fixed:
``policy=`` cohort ``j`` of plan ``k`` uses seed ``700_000 + 100 * k + j`` and logged cohort
``j`` uses seed ``500_000 + j``. One process, single-threaded.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time

import numpy as np

from cleverly.datasets import OFFER_LEVELS, OFFER_PLANS, make_sequential_offers, offers_truth
from cleverly.datasets.offers import _consent

N, DRAWS, LOGGED_DRAWS, WEEKS = 50_000, 60, 300, 3
truth = offers_truth()
np.seterr(invalid="ignore", divide="ignore")  # 0 / 0 on unobserved rows, masked by np.where
start = time.time()

for index, plan in enumerate(OFFER_PLANS):
    outcomes = []
    for draw in range(DRAWS):
        panel, _ = make_sequential_offers(N, seed=700_000 + 100 * index + draw, policy=plan)
        last = panel[panel["week"] == WEEKS]
        assert len(last) == N and last["full_price"].notna().all()
        outcomes.append(last["full_price"].to_numpy())
    y = np.concatenate(outcomes)
    mean, se = y.mean(), y.std() / np.sqrt(len(y))
    value = truth[f"ey_regimen[{plan}]"]
    print(
        f"policy {plan:14s} MC {mean:.5f} se {se:.5f} truth {value:.5f} z {(mean - value) / se:+.2f}"
    )
print(f"elapsed {time.time() - start:.0f}s")

estimates = {plan: [] for plan in OFFER_PLANS}
for draw in range(LOGGED_DRAWS):
    panel, _ = make_sequential_offers(N, seed=500_000 + draw)
    wide = panel.sort_values(["customer", "week"]).pivot(index="customer", columns="week")
    tenure = wide["tenure"][1].to_numpy()
    engaged = np.full((N, WEEKS), -1)
    offer = np.full((N, WEEKS), None, dtype=object)
    consent = np.zeros((N, WEEKS))
    logged = np.zeros((N, WEEKS, len(OFFER_LEVELS)))
    for week in range(1, WEEKS + 1):
        seen = wide["engaged"][week].notna().to_numpy()
        engaged[seen, week - 1] = wide["engaged"][week][seen].astype(int)
        offer[seen, week - 1] = wide["offer"][week][seen]
        consent[seen, week - 1] = wide["consent"][week][seen]
        for column, level in enumerate(OFFER_LEVELS):
            logged[seen, week - 1, column] = wide[f"p_{level}"][week][seen]
    outcome = wide["full_price"][WEEKS].to_numpy()
    outcome = np.where(np.isnan(outcome), 0.0, outcome)
    for plan, rule in OFFER_PLANS.items():
        weight = np.ones(N)
        for week in range(WEEKS):
            alive = (weight > 0) & (engaged[:, week] >= 0)
            engaged_now = np.where(alive, engaged[:, week], 0)
            planned = np.asarray(rule(engaged_now), dtype=object)
            followed = alive & (offer[:, week] == planned)
            column = np.array([OFFER_LEVELS.index(level) for level in planned])
            offer_probability = logged[np.arange(N), week, column]
            consent_probability = _consent(engaged_now, tenure)
            weight = np.where(
                followed & (consent[:, week] == 1),
                weight / offer_probability / consent_probability,
                0.0,
            )
        estimates[plan].append(np.mean(weight * outcome))
print(f"elapsed {time.time() - start:.0f}s")

estimates = {plan: np.asarray(values) for plan, values in estimates.items()}
for plan in OFFER_PLANS:
    mean = estimates[plan].mean()
    se = estimates[plan].std(ddof=1) / np.sqrt(LOGGED_DRAWS)
    value = truth[f"ey_regimen[{plan}]"]
    print(
        f"logged HT-IPW level    {plan:14s} {mean:.5f} se {se:.5f} truth {value:.5f} "
        f"z {(mean - value) / se:+.2f}"
    )
for plan in ("discount", "shipping", "win_back", "win_back_ship"):
    gap = estimates[plan] - estimates["none"]
    mean, se = gap.mean(), gap.std(ddof=1) / np.sqrt(LOGGED_DRAWS)
    value = truth[f"ate_regimen[{plan} vs none]"]
    print(
        f"logged HT-IPW contrast {plan:14s} {mean:.5f} se {se:.5f} truth {value:.5f} "
        f"z {(mean - value) / se:+.2f}"
    )
