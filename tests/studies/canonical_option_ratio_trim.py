r"""Declared evidence for ``ratio_trim=`` on LTMLE and SDR (``option-ratio-trim``).

The subject is ``LTMLE(ratio_trim=q)`` and ``SDR(ratio_trim=q)``: every per-node density ratio
above the type-7 quantile of all ``n x T`` entries is replaced by that quantile (X76 part (e),
Fix-8 plan sections 2.5 and 5.6).  The estimand does not change, but the remainder keeps a term
first order in the outcome-regression error on the trimmed share, so the trimmed fit is
consistent when every outcome regression is, and the package withholds its interval
(``trimmed_ratio_plugin``).  The row claims the point only, and only with the study's linear
outcome regressions.  Laws, learners and limits are :mod:`tests.studies._option_trim_law`; the
property families are :mod:`tests.studies.option_ratio_trim_properties`:

=====================  =================================================================
family                 cells
=====================  =================================================================
``trim_point``         LTMLE and SDR trimmed at 0.95 with correct outcome regressions, in
                       sample and cross-fitted: 4 gated bias positives
``trim_necessity``     LTMLE and SDR in sample with the W-blind outcome regression: the
                       trimmed fit (gated control, bias beyond the margin) and the
                       untrimmed fit (gated positive, double robustness)
``trim_reported``      the fits trimmed at 0.99, the untrimmed correct fits, and the
                       trimmed share per draw
=====================  =================================================================

**Primary.**  The in-sample LTMLE fit trimmed at 0.95 with correct outcome regressions, on
``LE``.  Its scenario is point-only: the truth row passes on the bias leg alone, and its
coverage and SE ratio, read from the plug-in diagnostic, are published.

**No comparator.**  ``lmtp``'s ``.trim`` is the rule the package transcribes, and its parity
fixtures are in ``tests/canonical/option_fixtures/``; a trimmed ``lmtp`` fit carries the same
remainder, so it would not be a reference for the truth.  The study reads every cell against
its exact truth.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/option_ratio_trim/regenerate.py`` before any run.

**Registered.**  The declared run is committed at e5b19a2bfb, and the record is in
``registered()``.  The ``G56`` support rows and the evidence page landed with the registration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _option_trim_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "option-ratio-trim"
SEED = 20267641
RESAMPLING_SEED = 20267642
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the in-sample LTMLE fit trimmed at 0.95 on ``LE`` at n = 2,000.
PRIMARY_SCENARIO = law.LAW
PRIMARY_SETTING = "ltmle_in_sample_right_trim95"
PRIMARY_N = 2_000
PRIMARY_REPLICATES = 1_600
PRIMARY_ESTIMANDS = (law.ESTIMAND,)

#: Filled by :mod:`tests.studies.option_ratio_trim_properties`, which this module cannot
#: import; the design test pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "trim_point": ("ltmle_in_sample", "ltmle_stacked", "sdr_in_sample", "sdr_stacked"),
    "trim_necessity": ("ltmle_trimmed", "ltmle_untrimmed", "sdr_trimmed", "sdr_untrimmed"),
    "trim_reported": (
        "ltmle_right_trim99",
        "sdr_right_trim99",
        "ltmle_wrong_trim99",
        "sdr_wrong_trim99",
        "ltmle_right_untrimmed",
        "sdr_right_untrimmed",
        "trimmed_share",
    ),
}

STUDY = StudyRecord(
    name="Ratio trimming on LTMLE and SDR",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "option_ratio_trim",
    document="docs/technical-reference/method-evidence/option-ratio-trim.md",
    anchor="ratio-trimming-on-ltmle-and-sdr",
    scenarios={PRIMARY_SCENARIO: PRIMARY_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    point_only_scenarios=frozenset({PRIMARY_SCENARIO}),
    modules=(
        "tests/studies/canonical_option_ratio_trim.py",
        "tests/studies/option_ratio_trim_properties.py",
        "tests/studies/_option_trim_law.py",
        "src/cleverly/longitudinal/ratio_trim.py",
        "src/cleverly/longitudinal/estimator.py",
        "src/cleverly/longitudinal/sequential.py",
        "src/cleverly/longitudinal/sdr.py",
        "src/cleverly/inference/influence.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_option_ratio_trim",
    properties_module="tests.studies.option_ratio_trim_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "in-sample and 5-fold cross-fitted LTMLE and SDR with ratio_trim=",
    "law": "W ~ U(0, 1); A1 ~ Bernoulli(expit(-3 + 4 W)); C1, C2 ~ Bernoulli(expit(2.5 + 0.5 W)); "
    "L2 ~ Bernoulli(0.2 + 0.3 W + 0.2 A1); A2 ~ Bernoulli(expit(-2.5 + 3 W + 0.5 L2 + 0.5 A1)); "
    "Y ~ Bernoulli(0.05 + 0.6 W + 0.15 L2 + 0.05 A1 + 0.1 A2)",
    "regimen": {law.REGIMEN: 1},
    "truth": law.TRUTH,
    "outcome_learner": "LinearRegression (and pseudo_learner); wrong: LinearRegression without W",
    "mechanism_learners": "unpenalized logistic regression by Newton (treatment, censoring)",
    "ratio_trim": {"gated": law.TRIM, "reported": law.REPORTED_TRIM},
    "g_bounds": list(law.G_BOUNDS),
    "max_iter": law.MAX_ITER,
    "tol": law.TOL,
    "n_folds": {"in_sample": 1, "stacked": law.N_FOLDS},
    "simultaneous_intervals": False,
    "discarded_trim": "ratio_trim=0.99 for the gated cells: its wrong-outcome controls read "
    "0.43 SD (LTMLE) and 0.51 SD (SDR) on the rev 2 pilot, under the 0.45 SD floor",
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(n, seed), {law.ESTIMAND: law.TRUTH}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Fit one primary sample and convert it to the shared replication schema.

    The trimmed fit withholds its interval, so the row reads the plug-in diagnostic
    (:func:`~tests.studies.evidence.schema.reported_inference`).
    """
    result = law.fit(frame, PRIMARY_SETTING)
    estimate = result.estimates[law.ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    reference = float(truth[law.ESTIMAND])
    initial = float(np.mean(result.fits[law.REGIMEN].steps[0].initial))
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": len(frame),
            "estimand": law.ESTIMAND,
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": initial,
        }
    ]


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(PRIMARY_SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The exact truth of every primary estimand, by scenario."""
    return {PRIMARY_SCENARIO: {law.ESTIMAND: law.TRUTH}}


__all__ = [
    "PROPERTY_CELLS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
