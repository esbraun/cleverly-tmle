r"""Registered evidence for the risk under eliminated competing events (X91 part A).

The subject is ``eliminate=["death"]`` on ``LTMLE``: the risk of relapse had death been
prevented, Young et al. (2020) g-formula (23), with the order censoring, death, relapse within a
node.  Retention has two factors, censoring and then the prevented cause, and the second is
fitted by ``competing_learner``.  The laws are those of
:mod:`tests.studies.eliminated_competing_law`, and every truth is the closed-form g-formula
written as quotients of cell masses.

===============================  ============================================================
scenario                         law, engine and estimands
===============================  ============================================================
``ltmle_time_varying``           time-varying law, n = 16,000; iterated-mean ``LTMLE``, one
                                 fold, ``competing_learner`` passed; the risk of every plan at
                                 ``t = 1`` and ``t = 2`` and the contrasts against ``never``
``ltmle_time_varying_crossfit``  the same law, size and estimands; two outer folds and no
                                 ``competing_learner``, so the censoring learner's clone fits
                                 the death factor out of fold, the default path
``hazard_held_baseline``         held-baseline law, the same size;
                                 ``LTMLE(survival_model="hazard")``; the risk of every arm at
                                 ``t = 1, 2, 3`` and the contrasts against ``a0``
===============================  ============================================================

Every nuisance is a saturated cell mean (``CellMeans``, ``CellProbabilities``), the
nonparametric model on these supports, so the one-fold intervals rest on a finite-dimensional
class.  The bounds never bind on these laws (``test_the_bounds_never_bind``).

**Scope** (declared before any run).  The record measures iterated-mean and hazard ``LTMLE``
with saturated learners, one fold and two folds.  ``SDR`` is not measured: with saturated
nuisances it returns the iterated ``LTMLE`` estimate and standard error on every sample
(``test_sdr_reduces_to_ltmle_under_saturated_nuisances``), so its rows would repeat these.  The
other engines rest on the exact-law checks of ``tests/unit/test_eliminated_competing_law.py``.

**Size** (declared before any run, from the pilot in
``tests/studies/eliminated_competing_events_pilot.py``).  Every scenario runs at 16,000.  The
first declaration ran the time-varying scenarios at 4,000; the pilot's 400 draws there read
coverage 0.915 to 0.93 and SE ratios 0.91 to 0.93 at ``t = 2`` with one fold, and a
standardized bias up to 0.23 with SE ratios up to 1.21 with two folds.  At 4,000 the hazard
scenario's ``t = 3`` cells rest on Kish sizes of 29 to 60, its coverage there read 0.89 to 0.92
in 400 draws, and 0.75% of its fits met a node regression with a constant target, against a 1%
failure bound.  At 16,000 the pilot's 1,000 draws per scenario read coverage 0.935 to 0.963 and
lost no fit.  :data:`PRIMARY_REPLICATES` is the smallest multiple of 400, from 1,000 up, whose
family-wise pass probability reaches 0.95 under the pooled reading of
:mod:`tests.studies.repeated_sizing` (one group per scenario, noise read from the pilot draws):
0.89 at 1,000 and 0.975 at 1,200.
``tests/unit/test_eliminated_competing_events_design.py`` recomputes it.

**Properties.** :mod:`tests.studies.eliminated_competing_events_properties` declares the
double-robustness family, which witnesses the prevented-cause factor and the targeting step,
and the calibration family.

**Comparator.** None in this record.  R ``lmtp`` with the composite censoring column (censoring
or death) is the matching implementation of the time-varying scenarios; its runner and its runs
come in a later change on the local queue, which declares it (roadmap X91).  The method merges
on its Python evidence.

The red-cell readings are declared in ``tests/canonical/eliminated_competing_events/regenerate.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.longitudinal import LTMLE, SequentialGComputation
from tests.discrete_law_point_survival import CellMeans, CellProbabilities
from tests.parallel import STUDY_JOBS
from tests.studies import eliminated_competing_law as law
from tests.studies.canonical_point_survival_hazard import initial_estimates
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 1_200
PRIMARY_N = 16_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 20261191
RESAMPLING_SEED = 2026119101
#: Cumulative bounds that never bind on these laws (see the module docstring).
G_BOUNDS = (1e-3, 1.0)
#: The prevented cause, as each fit names it.
ELIMINATE = ("death",)

LTMLE_TV = "ltmle_time_varying"
CROSSFIT_TV = "ltmle_time_varying_crossfit"
HAZARD_HB = "hazard_held_baseline"

#: ``scenario -> (law key, estimands, outer folds)``.
SCENARIOS: dict[str, tuple[str, tuple[str, ...], int]] = {
    LTMLE_TV: ("time_varying", law.TV_ESTIMANDS, 1),
    CROSSFIT_TV: ("time_varying", law.TV_ESTIMANDS, 2),
    HAZARD_HB: ("held_baseline", law.HB_ESTIMANDS, 1),
}
ESTIMANDS = {scenario: names for scenario, (_, names, _) in SCENARIOS.items()}
TRUTH: dict[str, dict[str, float]] = {
    scenario: dict(law.TV_TRUTH if key == "time_varying" else law.HB_TRUTH)
    for scenario, (key, _, _) in SCENARIOS.items()
}
EFFICIENCY_SD: dict[str, float] = {
    **{name: law.efficiency_sd(law.tv_eif(name), law.competing.PROBS) for name in law.TV_ESTIMANDS},
    **{name: law.efficiency_sd(law.hb_eif(name), law.HB_LAW.probs) for name in law.HB_ESTIMANDS},
}

#: Every published property cell, by family (see the properties module).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": (
        "tv_rule_t2__mechanism_correct",
        "tv_never_t2__outcome_wrong_competing_wrong",
        "hb_a1_t2__mechanism_correct",
        "hb_a2_t1__outcome_wrong_competing_wrong",
    ),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in ("ltmle_always_t2", "hazard_a1_t3")
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
}

STUDY = StudyRecord(
    name="eliminated competing events longitudinal TMLE",
    slug="eliminated-competing-events",
    artifacts=ROOT / "tests" / "canonical" / "eliminated_competing_events",
    document="docs/technical-reference/method-evidence/eliminated-competing-events.md",
    anchor="eliminated-competing-events-longitudinal-tmle",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-eliminated-ltmle",
    reference=None,
    failure_bound={"cleverly-eliminated-ltmle": FAILURE_BOUND},
    efficiency_bounds=EFFICIENCY_SD,
    modules=(
        "tests/studies/eliminated_competing_events.py",
        "tests/studies/eliminated_competing_events_properties.py",
        "tests/studies/eliminated_competing_law.py",
        "tests/discrete_law_competing.py",
        "tests/discrete_law_point_survival.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_point_survival_hazard.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.eliminated_competing_events",
    properties_module="tests.studies.eliminated_competing_events_properties",
    property_cells=PROPERTY_CELLS,
)

CONFIGURATION = {
    "construction": "eliminated competing events (Young 2020, eq. 23)",
    "eliminate": list(ELIMINATE),
    "within_node_order": ["censoring", "death", "relapse"],
    "engines": {
        LTMLE_TV: "LTMLE iterated mean, one fold",
        CROSSFIT_TV: "LTMLE iterated mean, two outer folds",
        HAZARD_HB: "LTMLE survival_model='hazard', one fold",
    },
    "sizes": {scenario: STUDY.n_of(scenario) for scenario in SCENARIOS},
    "learners": "CellMeans (outcome, pseudo, censoring, competing), CellProbabilities (treatment)",
    "competing_learner": {
        LTMLE_TV: "CellMeans, passed",
        CROSSFIT_TV: "not passed: the censoring learner's clone",
        HAZARD_HB: "CellMeans, passed",
    },
    "initial_estimate": (
        "the untargeted plug-in: SequentialGComputation with the same learners and folds on "
        "the iterated scenarios, the initial hazards' plug-in on the hazard scenario"
    ),
    "g_bounds": list(G_BOUNDS),
    "simultaneous_intervals": False,
    "variance_method": "ic",
}


def _learners(scenario: str) -> dict[str, Any]:
    chosen: dict[str, Any] = {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
    }
    if scenario != CROSSFIT_TV:
        chosen["competing_learner"] = CellMeans()
    return chosen


def fit(frame: pd.DataFrame, scenario: str, **learners: Any) -> Any:
    """The one fit of each primary scenario.  ``learners`` replace the declared ones."""
    key, _, folds = SCENARIOS[scenario]
    settings: dict[str, Any] = {
        "n_folds": folds,
        "g_bounds": G_BOUNDS,
        "simultaneous": False,
        "random_state": 0,
        **_learners(scenario),
        **learners,
    }
    if key == "time_varying":
        return LTMLE(law.TV_REGIMENS, reference=law.TV_REFERENCE, **settings).fit(
            frame, eliminate=list(ELIMINATE), **law.TV_FIT_COLUMNS
        )
    return LTMLE(
        law.HB_REGIMENS, reference=law.HB_REFERENCE, survival_model="hazard", **settings
    ).fit(frame, eliminate=list(ELIMINATE), **law.hb_fit_columns())


def untargeted(frame: pd.DataFrame, scenario: str, **learners: Any) -> Any:
    """The iterated scenarios' untargeted plug-in: the same learners, no fluctuation.

    It fits in sample in both scenarios, since ``SequentialGComputation`` refuses outer folds.
    """
    return SequentialGComputation(
        law.TV_REGIMENS,
        reference=law.TV_REFERENCE,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        random_state=0,
        **{**_learners(scenario), **learners},
    ).fit(frame, eliminate=list(ELIMINATE), **law.TV_FIT_COLUMNS)


def initials(result: Any, frame: pd.DataFrame, scenario: str) -> dict[str, float]:
    """The untargeted plug-in behind every reported parameter."""
    names = ESTIMANDS[scenario]
    if scenario == HAZARD_HB:
        return initial_estimates(result, names)
    plug_in = untargeted(frame, scenario)
    return {name: float(plug_in[name].psi) for name in names}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    key, _, _ = SCENARIOS[scenario]
    frame = law.tv_sample(n, seed) if key == "time_varying" else law.hb_sample(n, seed)
    return frame, dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    plug_in: Mapping[str, float],
) -> list[dict[str, Any]]:
    """One row per estimand on the identity scale."""
    rows = []
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(plug_in[name]),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit(frame, scenario)
    return result_rows(result, truth, scenario, replicate, initials(result, frame, scenario))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every primary row, each scenario at its declared size.  No sample is kept."""
    sizes = STUDY.sizes(n)
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in SCENARIOS
            for replicate in range(replicates)
        ],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])
