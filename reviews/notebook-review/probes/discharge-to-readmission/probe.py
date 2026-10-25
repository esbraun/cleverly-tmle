"""Review probe of docs/examples/discharge-to-readmission.ipynb: the page's two fits over 400 seeds.

The page reads one simulated sample of each program. This probe redraws each one 400 times at the
page's sizes and refits it with the page's settings, so the readings can say what holds across
samples rather than on one draw.

- The time-to-event part: the simple day-90 difference (the share with a recorded readmission,
  offered minus not offered), the fitted difference, its interval coverage, and its bias.
- The team part: the fitted difference with ``cluster="team"`` and the same fit without it. Their
  standard errors against the spread of the estimates, and their interval coverage.

Usage: ``python probe.py > probe.log``. The page quotes ``probe.log``. A review probe is not a
registered study.
"""

import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    LongitudinalTreatment,
    ModelSpec,
    RegimeContrast,
    Runtime,
    TimeToEvent,
    TMLEMethod,
)
from cleverly.datasets import make_point_survival, make_team_readmission

DRAWS = 400
SURVIVAL_SEEDS = range(10_000, 10_000 + DRAWS)
TEAM_SEEDS = range(20_000, 20_000 + DRAWS)

METHOD = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=41),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=41),
        censoring_learner=LogisticRegression(max_iter=1000, random_state=41),
    ),
    cross_fitting=CrossFitting(enabled=False),
    runtime=Runtime(random_state=41, n_jobs=1),
)
TEAM_METHOD = replace(METHOD, cross_fitting=CrossFitting(n_folds=5))
SURVIVAL_TARGET = "ate_regimen[offered vs not offered @ t=4]"
TEAM_TARGET = "ate_regimen[offer and call vs neither]"


def survival_draw(seed: int) -> tuple[float, float, float, float, float]:
    raw, truth = make_point_survival(n=2_000, seed=seed, continuous_event_time=True, n_times=4)
    frame = raw.rename(
        columns={"W1": "lives_alone", "W2": "prior_admissions", "A": "offer", "event": "readmitted"}
    )
    frame["day"] = 30.0 * frame.pop("time")
    by_day = frame["readmitted"].eq(1) & frame["day"].le(90.0)
    shares = by_day.groupby(frame["offer"]).mean()
    design = TimeToEvent(
        time="day",
        event="readmitted",
        treatment="offer",
        baseline=("lives_alone", "prior_admissions"),
        grid=(15.0, 30.0, 60.0, 90.0),
    )
    plans = {"offered": 1, "not offered": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = (
            CausalStudy(frame, design=design)
            .identify(RegimeContrast(plans, reference="not offered", horizons=(90.0,)))
            .estimate(method=METHOD)[SURVIVAL_TARGET]
        )
    low, high = fit.ci
    return shares[1] - shares[0], fit.psi, fit.std_error, low, high


def team_draw(seed: int) -> tuple[float, ...]:
    teams, _ = make_team_readmission(n_teams=100, team_size=40, seed=seed)
    design = LongitudinalTreatment(
        treatment=("offer", "call"),
        outcome="readmitted_90",
        censoring=("followed_30", "followed_90"),
        baseline=("lives_alone", "discharge_risk"),
        time_varying=((), ("readmitted_30",)),
        cluster="team",
    )
    schedules = {"offer and call": 1, "neither": 0}
    out: list[float] = []
    for cluster in ("team", None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = (
                CausalStudy(teams, design=replace(design, cluster=cluster))
                .identify(RegimeContrast(schedules, reference="neither"))
                .estimate(method=TEAM_METHOD)[TEAM_TARGET]
            )
        low, high = fit.ci
        out += [fit.psi, fit.std_error, low, high]
    return tuple(out)


def main() -> None:
    _, survival_truth = make_point_survival(n=10, seed=0, continuous_event_time=True, n_times=4)
    _, team_truth = make_team_readmission(n_teams=1, team_size=1, seed=0)
    truth_90 = survival_truth["ate_regimen[arm1 vs arm0 @ t=4]"]
    truth_team = team_truth["ate_regimen[always vs never]"]

    started = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        survival = np.array(list(pool.map(survival_draw, SURVIVAL_SEEDS, chunksize=10)))
        teams = np.array(list(pool.map(team_draw, TEAM_SEEDS, chunksize=10)))
    print(f"{DRAWS} draws of each part, seeds {SURVIVAL_SEEDS[0]} and {TEAM_SEEDS[0]} on, "
          f"{time.time() - started:.0f} s")

    naive, psi, se, low, high = survival.T
    covered = np.mean((low <= truth_90) & (truth_90 <= high))
    print("time-to-event part, day-90 difference, offered minus not offered")
    print(f"  true value {truth_90:.4f}")
    print(f"  simple difference: mean {naive.mean():.4f}, SD {naive.std(ddof=1):.4f}, "
          f"bias {naive.mean() - truth_90:.4f}")
    print(f"  fitted difference: mean {psi.mean():.4f}, SD {psi.std(ddof=1):.4f}, "
          f"bias {psi.mean() - truth_90:.4f}, mean SE {se.mean():.4f}, coverage {covered:.4f}")

    print("team part, offer and call against neither, by day 90")
    print(f"  true value {truth_team:.4f}")
    spread = None
    for index, label in enumerate(("teams declared", "patients as independent")):
        psi, se, low, high = teams[:, 4 * index : 4 * index + 4].T
        spread = psi.std(ddof=1) if spread is None else spread
        covered = np.mean((low <= truth_team) & (truth_team <= high))
        print(f"  {label}: mean {psi.mean():.4f}, SD {psi.std(ddof=1):.4f}, "
              f"mean SE {se.mean():.4f}, SE over SD {se.mean() / spread:.4f}, "
              f"coverage {covered:.4f}")


if __name__ == "__main__":
    main()
