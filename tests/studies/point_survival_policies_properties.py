"""Property families of ``point-treatment-survival-policies``.

==========================  ===============================================================
family                      cells, replications and size
==========================  ===============================================================
``interval_calibration``    ``policy_t5`` (the known policy against the natural course at
                            visit 5) and ``mtp_t5`` (the modified treatment policy against
                            the natural course at visit 5); 1,600 each at n = 2,000, with the
                            two derived controls of each
==========================  ===============================================================

The positive cells fit the saturated cell means for every nuisance, which are correct on these
finite laws, so the mechanism and the ratio numerator are estimated.  Each efficiency bound is
the exact standard deviation of the efficient influence function, summed over the law's
support: for the natural course and the MTP the treatment mechanism is part of the parameter,
and the bound carries its term.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_point_survival_policies as study
from tests.studies import point_survival_common as common
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = 1_600
CALIBRATION_LABELS = ("policy_t5", "mtp_t5")
SCENARIO_OF = {"policy_t5": "policy", "mtp_t5": "mtp"}
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
ESTIMAND = {
    label: common.contrast_name(study.PLAN[scenario], study.REFERENCE, 5)
    for label, scenario in SCENARIO_OF.items()
}


def _targets(label: str) -> list[common.Target]:
    scenario = SCENARIO_OF[label]
    natural = common.Target(-1.0, "mtp", lambda a: a, 5)
    if scenario == "policy":
        return [common.Target(1.0, "policy", common.known_policy, 5), natural]
    return [common.Target(1.0, "mtp", common.minus_one, 5), natural]


def efficiency_bound(label: str) -> float:
    """The exact efficiency bound of a label's contrast, as a standard deviation."""
    return common.efficiency_sd(study.SCENARIOS[SCENARIO_OF[label]], _targets(label))


#: Pinned from :func:`efficiency_bound`; the design test recomputes them.
EFFICIENCY_SD: dict[str, float] = {"policy_t5": 0.30868872551941806, "mtp_t5": 0.4088910040631876}


def fit_label(label: str, frame: pd.DataFrame) -> Any:
    scenario = SCENARIO_OF[label]
    return LTMLE(
        study.REGIMENS[scenario],
        reference=study.REFERENCE,
        n_folds=1,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        outcome_learner=common.CellMeans(),
        pseudo_learner=common.CellMeans(),
        treatment_learner=common.CellProbabilities(),
        censoring_learner=common.CellMeans(),
    ).fit(study.SCENARIOS[scenario].container(frame))


FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = tuple(
    (
        "interval_calibration",
        label,
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        label,
    )
    for label in CALIBRATION_LABELS
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


class DeclaredLaw:
    """The law a declared cell reads, with its exact truths.

    Parameters
    ----------
    label : str
        A calibration label.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = f"point_survival_policies__{label}"

    def truth(self) -> dict[str, float]:
        return dict(study.TRUTH[SCENARIO_OF[self.label]])


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    return tuple(
        PropertyCell(
            property="interval_calibration",
            cell=f"{label}__{cell}",
            dgp=DeclaredLaw(label),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES,
            seed=_seed("interval_calibration", label, 0),
            role=role,
            estimand=ESTIMAND[label],
        )
        for label in CALIBRATION_LABELS
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        )
    )


def _fit_set_rows(payload: tuple[str, str, str, int, int, int, str]) -> list[dict[str, Any]]:
    family, stream, _, replicate, n, requested, label = payload
    scenario = SCENARIO_OF[label]
    frame = study.SCENARIOS[scenario].draw(n, _seed(family, stream, replicate))
    result = fit_label(label, frame)
    return [
        replicate_row(
            property_name=family,
            cell=f"{label}__correctly_specified",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=study.TRUTH[scenario][ESTIMAND[label]],
            estimate=result[ESTIMAND[label]],
            alpha=STUDY.margins.alpha,
        )
    ]


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, stream, configuration, n, replicates, label in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(
            ((family, stream, configuration, r, n, requested, label),) for r in range(requested)
        )
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = map_parallel(_fit_set_rows, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CALIBRATION_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def failure_probe(draws: int, *, n_jobs: int = 1, start: int = 0) -> dict[str, int]:
    """Failed fits per fit set over streams ``start`` to ``start + draws - 1``."""
    payloads = [
        ((family, stream, configuration, r, n, draws, label),)
        for family, stream, configuration, n, _, label in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{family}/{stream}": 0 for family, stream, *_ in FIT_SETS}
    for (payload,), failed in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, str, int, int, int, str]) -> bool:
    try:
        rows = _fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    return finish(summary, rates)
