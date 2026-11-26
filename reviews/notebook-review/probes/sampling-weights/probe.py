"""Seed sweep of the fits of docs/examples/sampling-weights.ipynb.

The fits are those of "Estimate the effect", "What an unweighted analysis answers", "Effects for
high- and low-risk patients", "When some offer records are missing" and "The same weights over two
decisions".

Run from the repository root: ``python reviews/notebook-review/probes/sampling-weights/probe.py``.
It writes ``sweep.csv`` and ``summary.log`` beside this file.  The fits copy the notebook's
learners, folds, sizes and record rule; only the seeds change.
"""

from __future__ import annotations

import csv
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    LongitudinalTreatment,
    ModelSpec,
    PointTreatment,
    RegimeMean,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import make_biased_sample, make_longitudinal_weighted, sampling_probability

HERE = Path(__file__).resolve().parent
SEEDS = range(1000, 1500)
COVARIATES = ("W1", "W2")
#: The notebook's grid over W1, standard normal over every discharge; the effect is 1 + 1.2 W1.
RISK = np.linspace(-8.0, 8.0, 160_001)
DENSITY = np.exp(-(RISK**2) / 2)
INTERVIEWED = DENSITY * sampling_probability(RISK)


def true_effect(mass: np.ndarray, rows: object = True) -> float:
    mass = mass * rows
    return float(np.sum((1 + 1.2 * RISK) * mass) / np.sum(mass))


#: Each fit's own targets: every discharge (weighted) or patients like those interviewed.
SELECTED = true_effect(INTERVIEWED)
GROUP_TRUTH = {1.0: true_effect(DENSITY, RISK > 0), 0.0: true_effect(DENSITY, RISK <= 0)}
SELECTED_GROUP = {1.0: true_effect(INTERVIEWED, RISK > 0), 0.0: true_effect(INTERVIEWED, RISK <= 0)}


def point_method(seed: int, *, missingness: bool = False, linear: bool = True) -> TMLEMethod:
    extra = {"missingness_learner": LogisticRegression(max_iter=1000, random_state=seed)}
    return TMLEMethod(
        models=ModelSpec(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
            **(extra if missingness else {}),
        ),
        cross_fitting=CrossFitting(enabled=False),
        targeting=Targeting(fluctuation="linear") if linear else Targeting(),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )


def covers(point, value: float) -> bool:
    low, high = point.ci
    return bool(low <= value <= high)


def one(seed: int) -> dict[str, object]:
    warnings.simplefilter("ignore")
    frame, truth = make_biased_sample(n_population=8_000, seed=seed)
    frame["high_risk"] = (frame["W1"] > 0).astype(float)
    method = point_method(seed)

    def fit(data, method=method, **design):
        study = CausalStudy(data, design=PointTreatment(outcome="Y", treatment="A", **design))
        return study.identify(ATE(reference=0)).estimate(method=method)

    row: dict[str, object] = {"seed": seed, "ate_selected": truth["ate_selected"]}
    weighted = fit(frame, adjustment=COVARIATES, weights="sampling_weight")["ate"]
    plain = point_method(seed, linear=False)
    unweighted = fit(frame, method=plain, adjustment=COVARIATES)["ate"]
    row.update(
        w_psi=weighted.psi,
        w_se=weighted.std_error,
        w_covers=covers(weighted, truth["ate"]),
        u_psi=unweighted.psi,
        u_covers_all=covers(unweighted, truth["ate"]),
        u_covers_sample=covers(unweighted, SELECTED),
    )
    for label, weights, targets, group_method in (
        ("g", "sampling_weight", GROUP_TRUTH, method),
        ("gu", None, SELECTED_GROUP, plain),
    ):
        grouped = fit(
            frame,
            method=group_method,
            adjustment=(*COVARIATES, "high_risk"),
            strata=("high_risk",),
            weights=weights,
        )
        for group, name in ((1.0, "hi"), (0.0, "lo")):
            point = grouped[f"ate[high_risk={group}]"]
            row[f"{label}_{name}_psi"] = point.psi
            row[f"{label}_{name}_covers"] = covers(point, targets[group])

    # The notebook's record rule, with the record draw seeded one above the data seed.
    records = np.random.default_rng(seed + 1)
    kept = 1 / (1 + np.exp(-(1.5 + 0.8 * frame["W2"] - 0.7 * frame["A"])))
    patchy = frame.assign(offer_recorded=(records.random(len(frame)) < kept).astype(float))
    patchy.loc[patchy["offer_recorded"] == 0, "A"] = np.nan
    missing = fit(
        patchy,
        method=point_method(seed, missingness=True),
        adjustment=COVARIATES,
        weights="sampling_weight",
        treatment_missingness="offer_recorded",
    )["ate"]
    row.update(m_psi=missing.psi, m_covers=covers(missing, truth["ate"]))

    plans_frame, plans_truth = make_longitudinal_weighted(n=8_000, seed=seed)
    plans_method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(n_folds=5),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )
    for label, weights in (("lw", "w"), ("lu", None)):
        design = LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
            weights=weights,
        )
        result = (
            CausalStudy(plans_frame, design=design)
            .identify(RegimeMean(regimens={"always": 1, "never": 0}))
            .estimate(method=plans_method)
        )
        for plan in ("always", "never"):
            key = f"ey_regimen[{plan}]"
            row[f"{label}_{plan}_psi"] = result[key].psi
            row[f"{label}_{plan}_covers"] = covers(result[key], plans_truth[key])
    return row


def rate(rows: list[dict[str, object]], column: str) -> str:
    hits = sum(bool(row[column]) for row in rows)
    share = hits / len(rows)
    half = 1.96 * np.sqrt(share * (1 - share) / len(rows))
    return f"{hits} of {len(rows)} ({share:.3f}; Wald 95% {share - half:.3f} to {share + half:.3f})"


def main() -> None:
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, SEEDS))
    with (HERE / "sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    w_bias = np.array([float(row["w_psi"]) - 1.0 for row in rows])
    w_se = np.array([float(row["w_se"]) for row in rows])
    u_minus = np.array([float(row["u_psi"]) - SELECTED for row in rows])
    lines = [
        f"cleverly {cleverly.__version__}",
        f"draws: {len(rows)}, seeds {SEEDS.start} to {SEEDS.stop - 1}",
        f"truth: ATE 1.0, groups high {GROUP_TRUTH[1.0]:.6f}, low {GROUP_TRUTH[0.0]:.6f}; "
        f"interviewed-population ATE {SELECTED:.6f}, groups high {SELECTED_GROUP[1.0]:.6f}, "
        f"low {SELECTED_GROUP[0.0]:.6f}",
        f"weighted: covers ATE {rate(rows, 'w_covers')}; bias {w_bias.mean():+.5f} "
        f"(MC SE {w_bias.std(ddof=1) / np.sqrt(len(rows)):.5f}); "
        f"SD/SE {w_bias.std(ddof=1) / w_se.mean():.3f}",
        f"unweighted: covers ATE {rate(rows, 'u_covers_all')}; covers interviewed ATE "
        f"{rate(rows, 'u_covers_sample')}; mean minus interviewed ATE {u_minus.mean():+.5f}",
        f"weighted groups: high {rate(rows, 'g_hi_covers')}; low {rate(rows, 'g_lo_covers')}",
        f"unweighted groups: high {rate(rows, 'gu_hi_covers')}; low {rate(rows, 'gu_lo_covers')}",
        f"missing offer records, weighted: covers ATE {rate(rows, 'm_covers')}",
        f"two decisions, weighted: always {rate(rows, 'lw_always_covers')}; "
        f"never {rate(rows, 'lw_never_covers')}",
        f"two decisions, unweighted: always {rate(rows, 'lu_always_covers')}; "
        f"never {rate(rows, 'lu_never_covers')}",
    ]
    (HERE / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
