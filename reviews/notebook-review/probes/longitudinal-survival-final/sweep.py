"""Seed sweep of the fits that docs/examples/longitudinal-survival.ipynb shows.

Each seed ``s`` runs the notebook's code with the retention seed 52 replaced by ``s`` and the
competing seed 53 replaced by ``s + 1``:

- the follower comparison (Step 3) and the clustered retention fit (Steps 6 and 7);
- the competing-risk fit with disenrollment as censoring (Steps 8 and 9), and event-free
  survival as one minus the incidence total;
- the death-as-censoring recoding with ``K_k = C_k (1 - D_k)`` (Step 10);
- the support reports (Step 11) and the three covariate-drop refits (Step 12).

Learners are fixed and deterministic, as on the page. Truths come from ``truth.py`` (the
quadrature values, copied below).

Usage: ``python sweep.py <first seed> <count> <workers> <output csv>``. The page cites
``python sweep.py 1000 400 12 sweep.csv`` and ``python sweep.py 5000 400 12 sweep-confirm.csv``.
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
TRUTH = {  # truth.py, 60-point quadrature
    "exit always t=1": 0.149736,
    "exit always t=2": 0.317166,
    "exit never t=1": 0.257943,
    "exit never t=2": 0.455195,
    "readmission always t=1": 0.118721,
    "readmission always t=2": 0.246694,
    "readmission never t=1": 0.145230,
    "readmission never t=2": 0.244344,
    "death always t=1": 0.031015,
    "death always t=2": 0.070472,
    "death never t=1": 0.112712,
    "death never t=2": 0.210852,
    "death-first always t=2": 0.264627,
    "death-first never t=2": 0.304906,
}
EXIT_RENAME = {
    "W1": "age",
    "W2": "baseline_readiness",
    "A1": "navigation_p1",
    "C1": "tracked_p1",
    "Y1": "plan_exit_p1",
    "L2": "identified_needs",
    "A2": "navigation_p2",
    "C2": "tracked_p2",
    "Y2": "plan_exit_p2",
    "id": "navigator_team",
}
EVENT_RENAME = {
    "W1": "age",
    "W2": "baseline_readiness",
    "A1": "navigation_p1",
    "C1": "enrolled_p1",
    "L2": "identified_needs",
    "A2": "navigation_p2",
    "C2": "enrolled_p2",
    "D1": "death_p1",
    "D2": "death_p2",
    "R1": "readmission_p1",
    "R2": "readmission_p2",
}
DROPPED = ("age", "baseline_readiness", "identified_needs")


def covers(estimate, value):
    low, high = estimate.ci
    return bool(low <= value <= high)


def one_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore")
    from dataclasses import replace

    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        CausalStudy,
        CrossFitting,
        LongitudinalTreatment,
        ModelSpec,
        RegimeMean,
        Runtime,
        TMLEMethod,
    )
    from cleverly.datasets import make_longitudinal_competing, make_longitudinal_survival

    start = time.time()
    row: dict = {"seed": seed}
    sequential = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000, random_state=41),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=41),
            censoring_learner=LogisticRegression(max_iter=1000, random_state=41),
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=41, n_jobs=1),
    )
    nodes = {
        "treatment": ("navigation_p1", "navigation_p2"),
        "baseline": ("age", "baseline_readiness"),
        "time_varying": ((), ("identified_needs",)),
    }
    plans = {"always": 1, "never": 0}
    levels_by_plan = RegimeMean(plans, reference="never", horizons=(1, 2))

    def difference(result, first, second):
        return result.contrast(lambda psi: psi[0] - psi[1], [first, second], name="d")

    # ---- retention (Steps 2, 3, 6, 7, 11, 12)
    frame, _ = make_longitudinal_survival(n=4_000, seed=seed, cluster_size=20)
    frame = frame.rename(columns=EXIT_RENAME)
    first = {"always": frame["navigation_p1"].eq(1), "never": frame["navigation_p1"].eq(0)}
    second = {"always": ~frame["navigation_p2"].eq(0), "never": ~frame["navigation_p2"].eq(1)}
    followers = {1: first, 2: {p: first[p] & second[p] for p in first}}
    for h in (1, 2):
        outcome = frame[f"plan_exit_p{h}"]
        shares = {p: float(outcome[r & outcome.notna()].mean()) for p, r in followers[h].items()}
        row[f"naive_t{h}"] = shares["always"] - shares["never"]
    design = LongitudinalTreatment(
        outcome=("plan_exit_p1", "plan_exit_p2"),
        censoring=("tracked_p1", "tracked_p2"),
        cluster="navigator_team",
        **nodes,
    )
    exit_result = CausalStudy(frame, design=design).identify(levels_by_plan).estimate(
        method=sequential
    )
    for p in plans:
        for h in (1, 2):
            e = exit_result[f"risk_regimen[{p} @ t={h}]"]
            row[f"exit_{p}_t{h}"] = e.psi
            row[f"exit_{p}_t{h}_se"] = e.std_error
            row[f"exit_{p}_t{h}_cov"] = covers(e, TRUTH[f"exit {p} t={h}"])
    exit_diff = {}
    for h in (1, 2):
        d = difference(exit_result, f"risk_regimen[always @ t={h}]", f"risk_regimen[never @ t={h}]")
        exit_diff[h] = d
        row[f"exit_diff_t{h}"] = d.psi
        row[f"exit_diff_t{h}_se"] = d.std_error
        row[f"exit_diff_t{h}_cov"] = covers(
            d, TRUTH[f"exit always t={h}"] - TRUTH[f"exit never t={h}"]
        )
    bands = exit_result.simultaneous
    row["bands_all_cover"] = all(
        low <= TRUTH[f"exit {name.split('[')[1].split(' ')[0]} t={name[-2]}"] <= high
        for name, (low, high) in bands.bands.items()
    )
    support = exit_result.assess().report("support").to_frame()
    ratios = support["effective_n"] / support["n_followed"]
    row["exit_min_ess_ratio"] = float(ratios.min())
    weakest = support.loc[ratios.idxmin()]
    row["exit_weakest_row"] = f"{weakest['regimen']}/{weakest['time']}"
    row["exit_max_truncated"] = float(support["share_truncated"].max())
    row["exit_all_converged"] = bool(support["converged"].all())
    for dropped in DROPPED:
        reduced = replace(
            design,
            baseline=tuple(n for n in design.baseline if n != dropped),
            time_varying=tuple(tuple(n for n in node if n != dropped) for node in design.time_varying),
        )
        refit = CausalStudy(frame, design=reduced).identify(levels_by_plan).estimate(method=sequential)
        d = difference(refit, "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]")
        row[f"move_{dropped}"] = (d.psi - exit_diff[2].psi) / exit_diff[2].std_error

    # ---- competing risks with disenrollment as censoring (Steps 8, 9)
    events, _ = make_longitudinal_competing(n=4_000, seed=seed + 1, censoring=True)
    events = events.rename(columns=EVENT_RENAME)
    row["disenrolled_p1"] = int(events["enrolled_p1"].eq(0).sum())
    row["disenrolled_p2"] = int(events["enrolled_p2"].eq(0).sum())
    event_levels = (
        CausalStudy(
            events,
            design=LongitudinalTreatment(
                outcome={
                    "readmission": ("readmission_p1", "readmission_p2"),
                    "death": ("death_p1", "death_p2"),
                },
                censoring=("enrolled_p1", "enrolled_p2"),
                **nodes,
            ),
        )
        .identify(levels_by_plan)
        .estimate(method=sequential)
    )
    zs = []
    misses = []
    for p in plans:
        for c in ("readmission", "death"):
            for h in (1, 2):
                e = event_levels[f"cif_regimen[{p}, {c} @ t={h}]"]
                value = TRUTH[f"{c} {p} t={h}"]
                row[f"cif_{p}_{c}_t{h}"] = e.psi
                row[f"cif_{p}_{c}_t{h}_se"] = e.std_error
                row[f"cif_{p}_{c}_t{h}_cov"] = covers(e, value)
                zs.append((e.psi - value) / e.std_error)
                if not covers(e, value):
                    misses.append(f"{p}/{c}/{h}")
    event_diff = {}
    for c in ("readmission", "death"):
        for h in (1, 2):
            d = difference(
                event_levels, f"cif_regimen[always, {c} @ t={h}]", f"cif_regimen[never, {c} @ t={h}]"
            )
            event_diff[c, h] = d
            value = TRUTH[f"{c} always t={h}"] - TRUTH[f"{c} never t={h}"]
            row[f"cifd_{c}_t{h}"] = d.psi
            row[f"cifd_{c}_t{h}_se"] = d.std_error
            row[f"cifd_{c}_t{h}_cov"] = covers(d, value)
            zs.append((d.psi - value) / d.std_error)
            if not covers(d, value):
                misses.append(f"diff/{c}/{h}")
    row["misses"] = len(misses)
    row["miss_names"] = ";".join(misses)
    row["max_abs_z"] = max(abs(z) for z in zs)
    totals = event_levels.incidence_total()
    row["excess_max"] = float(totals["excess"].max())
    for r in totals.itertuples(index=False):
        value = 1 - TRUTH[f"readmission {r.regimen} t={r.time}"] - TRUTH[f"death {r.regimen} t={r.time}"]
        free = 1 - r.total
        row[f"free_{r.regimen}_t{r.time}"] = free
        row[f"free_{r.regimen}_t{r.time}_cov"] = bool(
            free - 1.959964 * r.std_err <= value <= free + 1.959964 * r.std_err
        )
    event_support = event_levels.assess().report("support").to_frame()
    i = event_support["epsilon"].abs().idxmax()
    row["eps_max_row"] = "/".join(
        str(event_support.loc[i, k]) for k in ("regimen", "cause", "horizon", "time")
    )
    row["event_max_truncated"] = float(event_support["share_truncated"].max())
    row["event_all_converged"] = bool(event_support["converged"].all())

    # ---- death as censoring, K_k = C_k (1 - D_k) (Step 10)
    kept_p1 = events["enrolled_p1"] * (1.0 - events["death_p1"].fillna(0.0))
    through_p1 = kept_p1.eq(1) & events["readmission_p1"].eq(0)
    kept_p2 = (events["enrolled_p2"] * (1.0 - events["death_p2"].fillna(0.0))).where(through_p1)
    recoded = events.assign(
        enrolled_alive_p1=kept_p1,
        enrolled_alive_p2=kept_p2,
        readmission_p1=events["readmission_p1"].where(kept_p1.eq(1)),
        readmission_p2=events["readmission_p2"].where(
            kept_p2.eq(1) | events["readmission_p1"].eq(1)
        ),
    )
    eliminated = (
        CausalStudy(
            recoded,
            design=LongitudinalTreatment(
                outcome=("readmission_p1", "readmission_p2"),
                censoring=("enrolled_alive_p1", "enrolled_alive_p2"),
                **nodes,
            ),
        )
        .identify(levels_by_plan)
        .estimate(method=sequential)
    )
    for p in plans:
        e = eliminated[f"risk_regimen[{p} @ t=2]"]
        row[f"el_{p}_t2"] = e.psi
        row[f"el_{p}_t2_cov"] = covers(e, TRUTH[f"death-first {p} t=2"])
    d = difference(eliminated, "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]")
    row["el_diff_t2"] = d.psi
    row["el_diff_t2_se"] = d.std_error
    row["el_diff_t2_cov_functional"] = covers(
        d, TRUTH["death-first always t=2"] - TRUTH["death-first never t=2"]
    )
    row["el_diff_t2_cov_total"] = covers(
        d, TRUTH["readmission always t=2"] - TRUTH["readmission never t=2"]
    )
    row["raised_never"] = row["el_never_t2"] - row["cif_never_readmission_t2"]
    row["raised_always"] = row["el_always_t2"] - row["cif_always_readmission_t2"]
    row["secs"] = time.time() - start
    return row


def main() -> None:
    import cleverly
    import pandas as pd

    first, count, workers = (int(v) for v in sys.argv[1:4])
    output = HERE / sys.argv[4]
    assert "bridge-cse" in cleverly.__file__, cleverly.__file__
    print("cleverly", cleverly.__file__, flush=True)
    seeds = range(first, first + count)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, seeds))
    pd.DataFrame(rows).sort_values("seed").to_csv(output, index=False, lineterminator="\n")
    print(f"wrote {len(rows)} rows to {output.name}", flush=True)


if __name__ == "__main__":
    main()
