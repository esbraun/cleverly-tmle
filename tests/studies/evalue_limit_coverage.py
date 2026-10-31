"""Registered evidence for the confidence limits of the part G E-values (X82 SG).

An E-value limit is a monotone map of the risk-ratio limit it reads, so the coverage of that
risk-ratio interval is the claim a registry row needs (``X82-SG-plan.md`` in the plans
repository).  This study holds the two cells no registered study already covers:

====  ===================  =============================================  ====================
cell  ``evalue_limit``     law                                            parameters
====  ===================  =============================================  ====================
E3    ``stratum``          :mod:`tests.studies.stratified_law`            ``rr[V=0]``, ``rr[V=1]``, ``rr[V=2]``
E4    ``time_to_event``    the ``survival`` law of ``point-treatment-survival``  the ratio of the two risks at ``t = 5``
====  ===================  =============================================  ====================

E3 reads the fit's own stratum risk ratio.  E4 reads ``result.ratio(numerator, denominator,
kind="rr")``, the interval :func:`cleverly.sensitivity.evalue` maps.  Both fits are the
registered studies' own correctly specified GLM fits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import pandas as pd

from cleverly.validation._interpretation import fit_configuration
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_point_survival_hazard as survival
from tests.studies import canonical_stratified_tmle as stratified
from tests.studies import stratified_law
from tests.studies.canonical_point_survival import TRUTH as SURVIVAL_TRUTH
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20262311
RESAMPLING_SEED = 20262312

#: The survival scenario and horizon of E4, and its two risks.
SURVIVAL_SCENARIO = "survival"
HORIZON = 5
NUMERATOR = f"risk_regimen[arm1 @ t={HORIZON}]"
DENOMINATOR = f"risk_regimen[arm0 @ t={HORIZON}]"

STRATUM_RATIOS = tuple(stratified_law.stratum_name("rr", s) for s in stratified_law.STRATA)

SCENARIOS = {"E3": STRATUM_RATIOS, "E4": (f"rr_t{HORIZON}",)}

#: The file that records the fit configuration of each cell.
CELLS_FILE = "cells.csv"

PROPERTY_CELLS = {
    "interval_calibration": (
        *(f"{cell}_{row}__correctly_specified" for cell, rows in SCENARIOS.items() for row in rows),
        *(f"{cell}_{row}__inflated_se_control" for cell, rows in SCENARIOS.items() for row in rows),
    ),
}

STUDY = StudyRecord(
    name="E-value limit coverage",
    slug="evalue-limit-coverage",
    artifacts=ROOT / "tests" / "canonical" / "evalue_limit_coverage",
    document="docs/technical-reference/method-evidence/evalue-limit-coverage.md",
    anchor="e-value-limit-coverage",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-evalue-limit",
    reference=None,
    failure_bound={"cleverly-evalue-limit": 0.01},
    modules=(
        "tests/studies/evalue_limit_coverage.py",
        "tests/studies/evalue_limit_coverage_properties.py",
        "tests/studies/stratified_law.py",
        "tests/studies/canonical_stratified_tmle.py",
        "tests/studies/canonical_point_survival.py",
        "tests/studies/canonical_point_survival_hazard.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.evalue_limit_coverage",
    properties_module="tests.studies.evalue_limit_coverage_properties",
    property_artifacts=(CELLS_FILE,),
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "E3": "stratified_law (V in {0, 1, 2}, binary W-levels, binary Y); in-sample TMLE with "
    "logistic outcome and treatment GLMs, strata=['V'], estimands ('rr',), as "
    "canonical-stratified-tmle fits",
    "E4": f"the {SURVIVAL_SCENARIO!r} law of point-treatment-survival; LTMLE on the "
    "TimeToEvent container with unpenalized logistic hazards, as "
    f"point-treatment-survival-hazard fits; the ratio {NUMERATOR} / {DENOMINATOR}",
    "primary_row": "the reported risk-ratio interval, on the log scale",
}


def truths(scenario: str) -> dict[str, float]:
    """The true risk ratio of each row of ``scenario``."""
    if scenario == "E3":
        return {
            name: stratified_law.truth(stratified_law.stratum_name("ey[1]", s))
            / stratified_law.truth(stratified_law.stratum_name("ey[0]", s))
            for name, s in zip(STRATUM_RATIOS, stratified_law.STRATA, strict=True)
        }
    if scenario == "E4":
        risks = SURVIVAL_TRUTH[SURVIVAL_SCENARIO]
        return {SCENARIOS["E4"][0]: risks[NUMERATOR] / risks[DENOMINATOR]}
    raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit."""
    if scenario == "E3":
        return stratified_law.sample(n, seed), truths(scenario)
    if scenario == "E4":
        frame, _ = survival.draw_from_seed(SURVIVAL_SCENARIO, n, seed)
        return frame, truths(scenario)
    raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit(scenario: str, frame: pd.DataFrame) -> Any:
    """The declared fit of ``scenario``."""
    if scenario == "E3":
        return stratified.fit_cleverly(frame, estimands=("rr",))
    if scenario == "E4":
        return survival.fit(SURVIVAL_SCENARIO, frame)
    raise KeyError(scenario)


def estimates(scenario: str, frame: pd.DataFrame) -> dict[str, Any]:
    """The declared fit's estimate object of each row of ``scenario``."""
    result = fit(scenario, frame)
    if scenario == "E3":
        return {name: result[name] for name in STRATUM_RATIOS}
    return {SCENARIOS["E4"][0]: result.ratio(NUMERATOR, DENOMINATOR, kind="rr")}


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema, one row per risk ratio."""
    rows: list[dict[str, Any]] = []
    for name, estimate in estimates(scenario, frame).items():
        reference = float(truth[name])
        std_error, low, high = reported_inference(estimate)
        if estimate.log_psi is None:  # pragma: no cover - estimator contract guard
            raise AssertionError(f"{scenario} {name}: a risk ratio needs a log-scale estimate")
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.log_psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "log",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


def cell_configurations() -> pd.DataFrame:
    """The fit configuration of each row, read off one fit per cell.

    The draw comes from a stream of its own.  Every learner is fixed and fits in sample, so
    one draw records the configuration that every replication of the cell fits.  E4 records
    the configuration of its numerator risk, which the ratio shares with its denominator.
    """
    records = []
    for scenario, rows in SCENARIOS.items():
        frame, _ = draw_from_seed(
            scenario, PRIMARY_N, stream_seed(STUDY, "configuration", scenario)
        )
        result = fit(scenario, frame)
        for row in rows:
            name = row if scenario == "E3" else NUMERATOR
            configuration = asdict(fit_configuration(result, name))
            configuration["learners"] = " | ".join(configuration["learners"])
            records.append({"scenario": scenario, "parameter": row, **configuration})
    return pd.DataFrame(records)
