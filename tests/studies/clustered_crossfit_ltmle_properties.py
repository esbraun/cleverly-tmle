"""Property families of ``clustered-cross-fitted-ltmle``.

``clustered_inference`` (gated)
    Four pairs of a cluster-robust arm and an IID control. Each pair reads one fit per draw,
    and both arms read the same influence curve: the cluster-sum variance against the row
    variance.

    ==========================  ===========================================================
    pair                        draw and target
    ==========================  ===========================================================
    ``cluster_robust``          ``equal40``, 100 clusters; ``ate_regimen[always vs never]``
    ``cluster_robust_dynamic``  the same fits; the dynamic rule against ``never``
    ``cluster_robust_unequal``  ``unequal40``, 100 clusters; the static contrast
    ``cluster_robust_survival`` the survival law, ``equal40``, 100 clusters; the risk
                                contrast at ``t=2``, through the
                                ``canonical-ltmle-survival-crossfit`` subject
    ==========================  ===========================================================

    The rules are
    :func:`~tests.studies.evidence.property_verdicts.clustered_inference_verdicts`. Each pair
    has 6,000 replications: at the probed coverage 0.935, the positive rule passes with
    probability 0.59 at 2,400 and 0.97 at 6,000 (``tests/unit/test_clustered_crossfit_ltmle_design.py``).
    The unequal draws hold a random number of rows, mean 4,000; every row publishes the
    nominal ``n = 4,000``, as the schema requires.
``simultaneous_coverage`` (gated)
    The band over the five reported names of the primary fit, with cluster multipliers, and
    its pointwise joint control. 2,400 replications: the design correlation of ten fits gives
    ``p0 = 0.859`` and a control power of 1.0 at that budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.inference import influence_variance
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ltmle_survival_crossfit as survival_study
from tests.studies import clustered_longitudinal_laws as law
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism, QuasiBinomialGLM
from tests.studies.canonical_ltmle_crossfit import CONTRAST_NAMES, ESTIMANDS
from tests.studies.clustered_crossfit_ltmle import (
    BAND_LABEL,
    CLUSTER_SIZE,
    CLUSTERS,
    ID,
    NODES,
    PRIMARY_N,
    STUDY,
    fit_cleverly,
)
from tests.studies.evidence.properties import PropertyCell, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    clustered_inference_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import (
    FAMILY as JOINT,
)
from tests.studies.evidence.simultaneous import (
    joint_coverage_rows,
    joint_property_cells,
)

CLUSTERED = "clustered_inference"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The replications of each ``clustered_inference`` pair.
CLUSTERED_REPLICATES = 6_000
#: The replications of the band cell, fixed by the control power before any run.
BAND_REPLICATES = 2_400
SURVIVAL_TARGET = "ate_regimen[always vs never @ t=2]"

#: Each pair: (positive cell, control cell, draw kind, target).
PAIRS = (
    ("cluster_robust", "iid_control", "end_of_study", CONTRAST_NAMES[0]),
    ("cluster_robust_dynamic", "iid_control_dynamic", "end_of_study", CONTRAST_NAMES[1]),
    ("cluster_robust_unequal", "iid_control_unequal", "unequal", CONTRAST_NAMES[0]),
    ("cluster_robust_survival", "iid_control_survival", "survival", SURVIVAL_TARGET),
)
#: The draw kinds, each with its own sample stream.
KINDS = ("end_of_study", "unequal", "survival")


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every reported name.

    Parameters
    ----------
    name : str
        One of :data:`KINDS`.
    """

    name: str

    def truth(self) -> dict[str, float]:
        """The ``make_longitudinal`` or ``make_longitudinal_survival`` truths."""
        if self.name == "survival":
            _, truth = law.draw_survival(2, "equal1", 0)
        else:
            _, truth = law.draw_end_of_study(2, "equal1", 0)
        return {name: float(value) for name, value in truth.items()}


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with the law it reads, its estimand and its stream root."""
    cells: list[PropertyCell] = []
    for positive, control, kind, target in PAIRS:
        for cell, role in ((positive, "positive"), (control, "control")):
            cells.append(
                PropertyCell(
                    property=CLUSTERED,
                    cell=cell,
                    dgp=DeclaredLaw(kind),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=PRIMARY_N,
                    replicates=CLUSTERED_REPLICATES,
                    seed=_seed(CLUSTERED, kind, 0),
                    role=role,
                    estimand=target,
                )
            )
    cells.extend(
        joint_property_cells(
            BAND_LABEL,
            n=PRIMARY_N,
            replicates=BAND_REPLICATES,
            seed=_seed(JOINT, BAND_LABEL, 0),
        )
    )
    return tuple(cells)


def fit_survival(frame: pd.DataFrame) -> Any:
    """The ``canonical-ltmle-survival-crossfit`` subject with ``id=``."""
    return LTMLE(
        survival_study.REGIMENS,
        reference=survival_study.REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=5,
        learner_folds=2,
        g_bounds=survival_study.G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, outcome=["Y1", "Y2"], id=ID, **NODES)


def fit_kind(kind: str, seed: int) -> tuple[Any, dict[str, float]]:
    """One draw of ``kind`` and its cross-fitted clustered fit."""
    if kind == "survival":
        frame, truth = law.draw_survival(CLUSTERS, f"equal{CLUSTER_SIZE}", seed)
        return fit_survival(frame), truth
    sizes = "unequal40" if kind == "unequal" else f"equal{CLUSTER_SIZE}"
    frame, truth = law.draw_end_of_study(CLUSTERS, sizes, seed)
    return fit_cleverly(frame), truth


def _iid(estimate: Any) -> float:
    return float(np.sqrt(influence_variance(estimate.influence_curve)))


def _pair_rows(payload: tuple[int, int, str]) -> list[dict[str, Any]]:
    """Both arms of every pair that reads this draw kind, from one fit."""
    replicate, requested, kind = payload
    result, truth = fit_kind(kind, _seed(CLUSTERED, kind, replicate))
    rows: list[dict[str, Any]] = []
    for positive, control, pair_kind, target in PAIRS:
        if pair_kind != kind:
            continue
        estimate = result[target]
        common = {
            "property_name": CLUSTERED,
            "replicate": replicate,
            "n": PRIMARY_N,
            "requested": requested,
            "truth": float(truth[target]),
        }
        rows.append(
            replicate_row(
                cell=positive,
                role="positive",
                estimate=estimate,
                alpha=STUDY.margins.alpha,
                **common,
            )
        )
        rows.append(
            control_row(
                cell=control,
                estimate=float(estimate.psi),
                standard_error=_iid(estimate),
                critical=CRITICAL,
                **common,
            )
        )
    return rows


def _band_rows(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, requested = payload
    frame, truth = law.draw_end_of_study(
        CLUSTERS, f"equal{CLUSTER_SIZE}", _seed(JOINT, BAND_LABEL, replicate)
    )
    result = fit_cleverly(frame, simultaneous=True)
    return joint_coverage_rows(
        result,
        {name: float(truth[name]) for name in ESTIMANDS},
        ESTIMANDS,
        label=BAND_LABEL,
        replicate=replicate,
        n=PRIMARY_N,
        requested=requested,
        pointwise_critical=CRITICAL,
    )


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    clustered = CLUSTERED_REPLICATES if budget is None else budget
    band = BAND_REPLICATES if budget is None else budget
    out: list[tuple[Any, ...]] = [
        ((_pair_rows, (r, clustered, kind)),) for kind in KINDS for r in range(clustered)
    ]
    out.extend(((_band_rows, (r, band)),) for r in range(band))
    return out


def _run(job: tuple[Any, tuple[Any, ...]]) -> list[dict[str, Any]]:
    function, payload = job
    return list(function(payload))


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pre-run check."""
    outcomes = map_parallel(_run, _payloads(budget), n_jobs=n_jobs)
    return pd.DataFrame([row for rows in outcomes for row in rows])


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the declared clustered and joint-coverage rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        rate_labels=(),
    )
    for positive, control, _, _ in PAIRS:
        clustered_inference_verdicts(
            summary, rows, STUDY, positive_cell=positive, control_cell=control
        )
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)
