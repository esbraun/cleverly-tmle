"""Repeated-sampling properties for ordinary multi-arm TMLE.

The ``simultaneous_coverage`` family measures the default band of a multi-arm fit.  Declared
before any run:

* each replication fits ``TMLE`` in sample with the calibration cell's learners (the oracle
  outcome regression and a main-terms multinomial logistic treatment model),
  ``estimands=("ey", "ate", "rr", "or")``, reference arm ``high``, the study's ``g_bounds``,
  and ``simultaneous=True`` with its defaults: 1000 rademacher draws seeded by
  ``random_state=0``.  The band covers nine parameters: three arm means, two differences, two
  risk ratios and two odds ratios.  The ratio bands are exponentiated from the log scale;
* the law is ``Sampler(effect=0.6)``, n = 2,000 and R = 2,400 on a dedicated stream
  ``stream_seed(STUDY, "property_sample", "simultaneous_coverage", "arms", r)``;
* the band cell passes when its 99% joint-coverage interval lies inside ``[0.92, 0.98]``, and
  its pointwise control passes when the 99% upper endpoint of joint coverage is below 0.95;
* a red cell is not repaired with a budget, a margin, a law, a learner, a size, a seed or a
  multiplier setting.  It is diagnosed against the oracle band first: a defect in the band is fixed and the
  study regenerated, and a finite-sample shortfall is published under ``reporting`` with an
  owner, or turns the default off for this shape.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common, multi_arm_properties
from tests.studies.canonical_multi_arm_tmle import STUDY
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyBatch,
    PropertyCell,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import (
    FAMILY,
    joint_coverage_rows,
    joint_property_cells,
)

#: The joint cell's label, budget and size.
JOINT_LABEL = "arms"
JOINT_REPLICATES = 2_400
JOINT_N = 2_000
JOINT_ESTIMANDS = ("ey", "ate", "rr", "or")
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def cells() -> tuple[PropertyCell, ...]:
    return (
        *multi_arm_properties.robustness_cells(seed=21_100),
        *multi_arm_properties.asymptotic_cells(seed=21_100),
    )


def _estimator(cell: PropertyCell):  # type: ignore[no-untyped-def]
    return lambda: TMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=False,
        estimands="ate",
        reference=multi_arm_common.REFERENCE,
        simultaneous=False,
        g_bounds=multi_arm_common.G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def sampling_batches() -> tuple[PropertyBatch, ...]:
    """The actual sampling calls, shared by complete and targeted regeneration."""
    return (PropertyBatch("properties", cells(), _estimator),)


def _joint_seed(replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", FAMILY, JOINT_LABEL, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """The sampled cells, then the joint pair on its own label-keyed stream."""
    return (
        *cells(),
        *joint_property_cells(
            JOINT_LABEL, n=JOINT_N, replicates=JOINT_REPLICATES, seed=_joint_seed(0)
        ),
    )


def fit_joint(frame: pd.DataFrame) -> Any:
    """The calibration cell's fit over the whole arm vector, with the default band."""
    return (
        TMLE(
            outcome_learner=multi_arm_properties.oracle_outcome()(),
            treatment_learner=multi_arm_properties.correct_treatment()(),
            cross_fit=False,
            estimands=JOINT_ESTIMANDS,
            reference=multi_arm_common.REFERENCE,
            simultaneous=True,
            g_bounds=multi_arm_common.G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


def _joint_replication(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, seed = payload
    frame, truth = multi_arm_properties.Sampler()(JOINT_N, seed)
    result = fit_joint(frame)
    return joint_coverage_rows(
        result,
        truth,
        tuple(result.estimates),
        label=JOINT_LABEL,
        replicate=replicate,
        n=JOINT_N,
        requested=JOINT_REPLICATES,
        pointwise_critical=CRITICAL,
    )


def joint_rows(*, n_jobs: int = STUDY_JOBS, replicates: int = JOINT_REPLICATES) -> pd.DataFrame:
    """The joint cell's rows, from its dedicated batch."""
    payloads = [((replicate, _joint_seed(replicate)),) for replicate in range(replicates)]
    outcomes = map_parallel(_joint_replication, payloads, n_jobs=n_jobs)
    return pd.DataFrame([row for rows in outcomes for row in rows]).loc[:, list(REPLICATE_COLUMNS)]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return pd.concat(
        [sampling_batches()[0].run(n_jobs=n_jobs), joint_rows(n_jobs=n_jobs)],
        ignore_index=True,
    )


def generate_smoke_property_rows(*, n_jobs: int = 1, replicates: int = 1) -> pd.DataFrame:
    """The first ``replicates`` declared joint replications, unsummarized."""
    return joint_rows(n_jobs=n_jobs, replicates=replicates)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)
