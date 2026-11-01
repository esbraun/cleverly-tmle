"""The two small-n parity records of X59 (``X59-plan.md`` section 5), fitted fresh on both sides.

``long-horizon-ltmle-parity`` pairs ``LTMLE`` with R ``ltmle`` 1.3-0 on ``L_R`` at K = 5, m = 2
and n = 4,000, over the four deterministic plans and their contrasts.
``long-horizon-lmtp-parity`` pairs ``mix`` on ``L_R`` at K = 5, m = 3 and n = 8,000 through
``lmtp_tmle`` at one fold and ``lmtp_sdr`` at one and five folds, against ``LTMLE`` and ``SDR``.
Small n reaches the sparse strata and near-separation where two implementations of one
estimator part, and the declared mutations run here cheaply.

Both records are parity-only.  They are not registered (``registry.registered`` omits them),
they make no truth claim, and every margins pair is descriptive: the only readings are the
exact gates of ``parity.csv`` (G-P, G-C and G1, section 4.2).  The rows carry the oracle truth
because the shared schema needs one; no page reads it.  Every fit is Wald-only, with no
bootstrap.  Each record draws from its own seed stream
(``tests/unit/test_x59_parity_records.py`` checks it against every registered study).

The subject phase runs first (``regenerate.py``); the comparator then runs reference-only on
the same seeds (``regenerate.py --reference-only``), as for the three primary studies.  A
refusal on either side is a recorded failure under ``FAILURE_BOUND``, never a skip.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from tests.studies import long_horizon_primary as primary
from tests.studies.evidence import replication
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

#: Every parity record runs 500 replicates (section 5).
PARITY_REPLICATES = 500


def scenario_estimands(scenarios: Sequence[primary.Scenario]) -> dict[str, tuple[str, ...]]:
    """The record's ``scenarios``: Wald rows only, because no parity fit bootstraps."""
    return {scenario.name: primary.wald_estimands(scenario) for scenario in scenarios}


def _fit_one(
    record: StudyRecord,
    scenarios: Mapping[str, primary.Scenario],
    estimators: Mapping[str, str],
    key: Any,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    name, replicate, n = key
    scenario = scenarios[name]
    result = primary.fit(frame, scenario, estimators[name], bootstrap=False)
    return primary.rows(record, result, scenario, replicate, n)


def run_parity(
    record: StudyRecord,
    scenarios: Sequence[primary.Scenario],
    estimators: Mapping[str, str],
    *,
    replicates: int,
    n: int,
    n_jobs: int,
) -> replication.PhaseResult:
    """The subject phase of a parity record: every scenario Wald-only with its own estimator."""
    table = {scenario.name: scenario for scenario in scenarios}
    sizes = record.sizes(n)
    keys = [
        (name, replicate, sizes[name])
        for replicate in range(replicates)
        for name in record.scenarios
    ]

    def draw(key: Any) -> pd.DataFrame:
        from tests import discrete_law_longitudinal_horizon as law_

        name, replicate, size = key
        scenario = table[name]
        seed = replicate_seed(record, name, replicate)
        return law_.draw(scenario.K, scenario.m, size, seed, scenario.law)

    outcomes = replication.fit_primary(
        keys,
        functools.partial(_fit_one, record, table, estimators),
        draw=draw,
        n_jobs=n_jobs,
        implementation=record.implementation,
        rows_at=None,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    frame = pd.DataFrame([row for kept in outcomes.kept for row in kept])
    if frame.empty:
        frame = pd.DataFrame(columns=list(REPLICATE_COLUMNS))
    return replication.result(outcomes, frame.loc[:, list(REPLICATE_COLUMNS)])
