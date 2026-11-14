r"""Registered evidence for the restricted means and risk ratios of the continuous-time TMLE (X6 (c)).

The subject is :class:`~cleverly.survival.ContinuousSurvivalTMLE` fitted with
``restricted_horizons=(2.32,)``: the one-step TMLE of Rytgaard and van der Laan (2024) whose walk
targets each arm's restricted mean time lost (RMTL) to each cause beside the risks.  The law is
the registered study's base law (:mod:`tests.studies.continuous_survival_law`) and the fit is
that study's gated configuration (``both_cf_1k``: the default libraries, cross-fitted with 10
outer folds and selection inside 5 inner folds, :math:`n = 1{,}000`, target times at the three
event-time quartiles) with the restricted horizon added (X6 plan rev 2, section 5.4).

**Targets** (:data:`~tests.studies.continuous_survival_functionals_law.GATED`, 25 rows).  At the
median and upper quartile: each arm's risk of each cause and each risk difference (the RMTL
targets enlarge the walk, so these estimates differ from the registered study's and need their own
rows), and each cause's risk ratio on the log scale.  At :math:`\tau = 2.32`: the RMTL of each arm
and cause, the RMST of each arm, and their differences.

**Interval.** The one the library ships when the study runs: today the influence-curve variance,
the range-logit interval of :math:`(0, 1)` for a risk and of :math:`(0, \tau)` for an RMTL or RMST
(Fix-3 PR 2, #637, and X6 (c)), the log interval for a ratio, and the identity interval for a
difference until Fix-3 PR 3.  The standard error of every level is on its reported scale.

**Gates.** Every row takes the primary rule of :data:`MARGINS`: the coverage interval above
0.90, the SE ratio's interval inside (0.8, 1.2) and the bias equivalent to zero within 0.2 SD.
The RMST difference also has a calibration cell with its two derived controls, the shrunk-SE
control at 0.85 among them, in :mod:`tests.studies.continuous_survival_functionals_properties`,
which also holds the targeting-necessity reading and the declared second-phase band row.

**Size and seeds.** The pilot (:mod:`tests.studies.continuous_survival_functionals_pilot`,
:data:`PILOT_REPLICATES` draws on its own stream) sets :data:`PRIMARY_REPLICATES` by the shared
primary rule, :func:`~tests.studies.repeated_sizing.primary_replicates`.  Until the pilot is
committed the count is the placeholder 0, and ``regenerate.py`` refuses a declared run; the
design test recomputes the count from the committed pilot and fails on any other value.
:data:`SEED` and :data:`RESAMPLING_SEED` are used by no other study; replicate ``r`` of a cell
draws from ``replicate_seed(STUDY, cell, r)``.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.survival import ContinuousSurvivalTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_functionals_law as functionals
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: The plan's pilot (X6 plan rev 2, 5.4), on its own stream.
PILOT_REPLICATES = 3_200
#: The placeholder until the pilot is committed; then the shared primary rule's count from it
#: (:func:`tests.studies.continuous_survival_functionals_pilot.primary_count`).
PENDING_REPLICATES = 0
PRIMARY_REPLICATES = 14_800
PRIMARY_N = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026611361
RESAMPLING_SEED = 2026611362
N_FOLDS = registered.N_FOLDS
LEARNER_FOLDS = registered.LEARNER_FOLDS
TAU = functionals.TAU

SUBJECT = "cleverly-continuous-survival-functionals"

GATED_CELL = "functionals_cf_1k"
SCENARIOS: dict[str, int] = {GATED_CELL: PRIMARY_N}
ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(SCENARIOS, functionals.GATED)
MARGINS = Margins(standardized_bias=0.2)

READINGS_FILE = "targeting-readings.csv"

STUDY = StudyRecord(
    name="continuous-time survival restricted means and risk ratios",
    slug="continuous-survival-functionals",
    artifacts=ROOT / "tests" / "canonical" / "continuous_survival_functionals",
    document="docs/technical-reference/method-evidence/continuous-survival-functionals.md",
    anchor="continuous-time-survival-restricted-means-and-risk-ratios",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation=SUBJECT,
    reference=None,
    failure_bound={SUBJECT: FAILURE_BOUND},
    nuisance_count=3,
    extra_artifacts=(READINGS_FILE,),
    modules=(
        "tests/studies/continuous_survival_functionals.py",
        "tests/studies/continuous_survival_functionals_law.py",
        "tests/studies/continuous_survival_functionals_properties.py",
        "tests/studies/continuous_survival.py",
        "tests/studies/continuous_survival_law.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        # Imported through the registered study's module; none of them is called by these fits.
        "tests/studies/stratified_incremental_msm_properties.py",
        "tests/studies/canonical_stratified_incremental_msm.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law.py",
    ),
    runner_module="tests.studies.continuous_survival_functionals",
    properties_module="tests.studies.continuous_survival_functionals_properties",
    property_cells={
        "interval_calibration": tuple(
            f"rmst_rd__{cell}"
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
    },
    property_artifacts=("necessity-cells.csv", "necessity-replicates.csv"),
)

CONFIGURATION = {
    "construction": "Rytgaard and van der Laan (2024) one-step TMLE, both arms tilted, "
    "restricted means targeted",
    "law": "the continuous-time-survival base law (tests/studies/continuous_survival_law.py)",
    "times": list(law.TIMES),
    "restricted_horizons": [TAU],
    "n_folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "learners": "cox.default_library() for each hazard; logistic + lasso C in (1, 0.1, 0.03)",
    "sizes": dict(SCENARIOS),
    "variance_method": "influence curve",
}


# -------------------------------------------------------------------------- the draws


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return registered.sample(n, seed), dict(functionals.TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# --------------------------------------------------------------------------- the fits


def engine(*, restricted: bool = True) -> ContinuousSurvivalTMLE:
    """The registered study's gated configuration, with the restricted horizon by default."""
    return ContinuousSurvivalTMLE(
        law.TIMES,
        restricted_horizons=(TAU,) if restricted else (),
        arms=law.ARMS,
        reference=law.REFERENCE,
        causes=law.CAUSES,
        cross_fit=True,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        simultaneous=False,
        random_state=0,
    )


def fit(frame: pd.DataFrame, scenario: str = GATED_CELL) -> Any:
    """The subject's fit of one replicate of the primary cell."""
    return registered.fit_with(frame, engine())


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per gated target: a ratio on the log scale, every other target as reported."""
    rows = []
    initial = result.initial_estimates
    for name in functionals.GATED:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        ratio = estimate.scale == "ratio"
        rows.append(
            {
                "implementation": SUBJECT,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.log_psi) if ratio else float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                # The scale of ``std_error`` and ``inference_estimate``: the log for a ratio,
                # the reported scale otherwise.  A risk, an RMTL and an RMST form their interval
                # on the logit of their place in (0, 1) or (0, tau) and map it back, as #637's
                # risks do; ``interval_scale`` records that, and ``ci_lower``, ``ci_upper`` and
                # ``covered`` record that interval.
                "inference_scale": "log" if ratio else "identity",
                "interval_scale": estimate.interval_scale,
                "covered": int(low <= reference <= high),
                "initial_estimate": float(initial.get(name, float("nan"))),
            }
        )
    return rows


#: The columns a primary row carries beside the published schema.
EXTRA_COLUMNS = (
    "interval_scale",
    "seconds",
    "steps",
    "halvings",
    "converged",
    "bounded_share",
    "last_time_floor_share",
    "frozen_share",
    "off_grid_share",
)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result = fit(frame, scenario)
        seconds = time.perf_counter() - start
    extra = {"seconds": seconds, **registered.readings(result)}
    return [{**row, **extra} for row in result_rows(result, truth, scenario, replicate)]


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    scenarios: tuple[str, ...] | None = None,
) -> replication.PhaseResult:
    """Every cell of ``scenarios`` (all by default), each at its declared size."""
    sizes = STUDY.sizes(n)
    chosen = tuple(STUDY.scenarios) if scenarios is None else scenarios
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in chosen
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
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]])


# --------------------------------------------------------------- reported artifacts


def account_failures(
    failures: pd.DataFrame | None, artifacts: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """No scientific gate reads the extra table, so a lost fit counts only against the bound."""
    return evidence_failures.audits({}, missing={})


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The targeting record per cell; not a verdict."""
    return {READINGS_FILE: registered.targeting_readings(rows)}
