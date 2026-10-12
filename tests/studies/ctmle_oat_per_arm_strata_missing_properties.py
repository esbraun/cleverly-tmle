"""Repeated-sampling properties of the per-arm outcome-adaptive C-TMLE with strata and missing data.

Every cell reads the laws of :mod:`tests.studies.oat_per_arm_extension_laws` and fits the
per-arm design with the study's learners: the correctly specified logistic GLM with arm and
stratum interactions for the outcome, and the cubic in ``logit q`` for every factor.  The
study claims consistency of the point estimate, so every verdict is a bias verdict.

==================================================  ==============================================
cell                                                fit
==================================================  ==============================================
``robustness_contract`` ``natural_course_insample``  ``missing_outcome``, in sample, ``n = 2,250``:
                                                    ``ey_obs``, ``par``, ``paf``
``robustness_contract`` ``missing_outcome_crossfit`` ``missing_outcome``, five folds, ``n = 2,000``:
                                                    the arm means, ``ate``, ``rr``, ``or``,
                                                    ``ey_obs``, ``par``, ``paf``
``robustness_contract`` ``strata_missing``           ``strata_missing``, five folds, ``strata=S``,
                                                    ``n = 3,000``: marginal and stratum ``ate``,
                                                    ``ey_obs``, ``par``, ``paf``
``robustness_contract`` ``weighted_strata_missing``  the same under the declared known weight
                                                    ``w(W) = 0.5 + 1{W1 > 0}``: marginal and
                                                    stratum ``ate``, truth under the tilted law
``robustness_contract``                             the control: ``strata_missing`` with X17's
``strata_missing_outcome_wrong``                    outcome model without ``W1``: marginal
                                                    ``ate``
``root_n_and_efficiency`` ``n_500`` to ``n_8000``    ``strata_missing``, five folds, ``strata=S``:
                                                    marginal ``ate``.  The ``root_n_rate`` rows
                                                    read the ladder
==================================================  ==============================================

A configuration reports several estimands from one fit, and a
:class:`~tests.studies.evidence.properties.PropertyCell` measures one, so
:func:`configuration_rows` fits once per replication and writes one row per estimand, in the
X17 ``joint_rows`` pattern.  Each row's cell is ``<estimand slug>__<configuration>``.  The
draws come from ``stream_seed(STUDY, "property_sample", family, configuration, replicate)``.
A replication that raises is not redrawn: its rows are absent, every row of the configuration
records the failed count, and the shared summary refuses the cell.

Positive cells pass on ``bias_equivalent`` with an SE ratio inside ``UNION_MODEL_SE_BAND``
(:func:`~tests.studies.evidence.property_verdicts.robustness_verdicts`).  The control passes on
``bias_discriminated``.  The ladder's rungs answer to the shared ``root_n_and_efficiency`` rule,
whose coverage and SE-ratio readings fall under the study's reading rule
(:data:`~tests.studies.canonical_ctmle_oat_per_arm_strata_missing.READING_RULE`).

**A budget the plan moved.**  The plan named ``R = 500`` per rung.  Condition (r) of the same
plan requires a pass probability of at least 0.90 for a cell whose true standardized bias is
0.10, and at ``R = 500`` that probability is 0.777.  Each rung takes ``R = 800`` (0.950), as
every other cell does.  The move was made before any fit of the estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import oat_per_arm_extension_laws as laws
from tests.studies.canonical_ctmle_oat_per_arm_strata_missing import (
    LADDER,
    PROPERTY_ESTIMANDS,
    STUDY,
    estimator,
    multi_cells,
    roles,
)
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    finish,
    robustness_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

#: Every property cell's budget, by condition (r): a cell whose true standardized bias is
#: 0.10 passes the bias rule with probability 0.950.
PROPERTY_REPLICATES = 800
LADDER_REPLICATES = 800
#: The control's population bias on marginal ``ate`` and its standard deviation at its own
#: size ``n = 3,000``, from two cross-fitted control fits at ``n = 200,000`` (seeds 1 and 2;
#: bias 0.0638 and 0.0736, curve standard deviation 1.147 and 1.143), as
#: ``_x29_law_design.py`` records.  Condition (s): the standardized bias is 3.29, and the
#: control clears the margin at ``R = 800`` with probability 1.000.
CONTROL_BIAS = 0.0687
CONTROL_SD = 0.0209


@dataclass(frozen=True)
class Configuration:
    """One fit configuration, and the estimands its rows report.

    Parameters
    ----------
    name : str
        The configuration's name, the suffix of each of its cells.
    family : str
        The property family.
    law : ExtensionLaw
        The law it samples.
    n : int
        The sample size.
    replicates : int
        The declared budget.
    estimands : tuple of str
        The estimands, as the fit names them.
    cross_fit : bool
        Whether the fit cross-fits with five folds.
    outcome : str
        ``"correct"`` or ``"wrong"``.
    role : str
        ``"positive"`` or ``"control"``.
    cells : tuple of str
        The published cell of each estimand.
    """

    name: str
    family: str
    law: laws.ExtensionLaw
    n: int
    replicates: int
    estimands: tuple[str, ...]
    cross_fit: bool = True
    outcome: str = "correct"
    role: str = "positive"
    cells: tuple[str, ...] = ()

    def requests(self) -> tuple[str, ...]:
        """The estimand requests: the marginal names, which with strata add each stratum."""
        return tuple(dict.fromkeys(name.partition("[")[0] for name in self.estimands))

    def seed(self, replicate: int) -> int:
        """The sample seed of one replication."""
        return stream_seed(STUDY, "property_sample", self.family, self.name, replicate)


def _robustness(
    name: str,
    law: laws.ExtensionLaw,
    n: int,
    *,
    cross_fit: bool = True,
    outcome: str = "correct",
    role: str = "positive",
) -> Configuration:
    estimands = PROPERTY_ESTIMANDS[name]
    return Configuration(
        name=name,
        family="robustness_contract",
        law=law,
        n=n,
        replicates=PROPERTY_REPLICATES,
        estimands=estimands,
        cross_fit=cross_fit,
        outcome=outcome,
        role=role,
        cells=multi_cells(name, estimands),
    )


CONFIGURATIONS: tuple[Configuration, ...] = (
    _robustness("natural_course_insample", laws.MISSING_OUTCOME, 2_250, cross_fit=False),
    _robustness("missing_outcome_crossfit", laws.MISSING_OUTCOME, 2_000),
    _robustness("strata_missing", laws.STRATA_MISSING, 3_000),
    _robustness("weighted_strata_missing", laws.STRATA_MISSING_WEIGHTED, 3_000),
    _robustness(
        "strata_missing_outcome_wrong",
        laws.STRATA_MISSING,
        3_000,
        outcome="wrong",
        role="control",
    ),
    *(
        Configuration(
            name=f"n_{n}",
            family="root_n_and_efficiency",
            law=laws.STRATA_MISSING,
            n=n,
            replicates=LADDER_REPLICATES,
            estimands=("ate",),
            cells=(f"n_{n}",),
        )
        for n in LADDER
    ),
)
CONFIGURATION_BY_NAME: Mapping[str, Configuration] = {
    configuration.name: configuration for configuration in CONFIGURATIONS
}


def fit(configuration: Configuration, frame: pd.DataFrame) -> Any:
    """One configuration's fit on one sample."""
    return (
        estimator(
            configuration.law,
            configuration.requests(),
            cross_fit=configuration.cross_fit,
            outcome=configuration.outcome,
        )
        .fit(frame, **roles(configuration.law))
        .single()
    )


def _replication(payload: tuple[str, int, int]) -> list[dict[str, Any]] | None:
    """One replication's rows, or ``None`` when the fit raised."""
    name, replicate, requested = payload
    configuration = CONFIGURATION_BY_NAME[name]
    frame, truth = configuration.law.sample(configuration.n, configuration.seed(replicate))
    try:
        result = fit(configuration, frame)
    except Exception:  # a raising replication is counted, never redrawn
        return None
    alpha = STUDY.margins.alpha
    return [
        replicate_row(
            property_name=configuration.family,
            cell=cell,
            role=configuration.role,
            replicate=replicate,
            n=configuration.n,
            requested=requested,
            truth=float(truth[estimand]),
            estimate=result[estimand],
            alpha=alpha,
        )
        for estimand, cell in zip(configuration.estimands, configuration.cells, strict=True)
    ]


def configuration_rows(
    configuration: Configuration, *, n_jobs: int, replicates: int | None = None
) -> pd.DataFrame:
    """Every replication of one configuration, one row per estimand.

    Parameters
    ----------
    configuration : Configuration
        The configuration.
    n_jobs : int
        Worker processes.
    replicates : int or None, optional
        A smaller budget for a disposable smoke; the declared one by default.

    Returns
    -------
    DataFrame
        :data:`~tests.studies.evidence.properties.REPLICATE_COLUMNS` rows, with the failed
        count of the configuration on every row.
    """
    budget = configuration.replicates if replicates is None else replicates
    outcomes = map_parallel(
        _replication,
        [((configuration.name, replicate, budget),) for replicate in range(budget)],
        n_jobs=n_jobs,
    )
    failed = sum(outcome is None for outcome in outcomes)
    rows = [row for outcome in outcomes if outcome is not None for row in outcome]
    frame = pd.DataFrame(rows, columns=list(REPLICATE_COLUMNS))
    frame["failed_replicates"] = failed
    return frame


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    frames = [configuration_rows(item, n_jobs=n_jobs) for item in CONFIGURATIONS]
    return pd.concat(frames, ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]


def generate_smoke_property_rows(*, n_jobs: int = 1, replicates: int = 20) -> pd.DataFrame:
    """The first ``replicates`` replications of every configuration, for a disposable smoke."""
    frames = [
        configuration_rows(item, n_jobs=n_jobs, replicates=replicates) for item in CONFIGURATIONS
    ]
    return pd.concat(frames, ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]


def _unused(*_: Any) -> None:  # pragma: no cover - the declared cells fit nothing themselves
    return None


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every cell this study publishes, so each committed truth is read back against its law.

    No cell is passed to :func:`~tests.studies.evidence.properties.run_cells`:
    :func:`configuration_rows` fits each configuration once per replication.  Each cell's
    ``seed`` names the root of its configuration's stream.

    Returns
    -------
    tuple of PropertyCell
        One cell per configuration and estimand.
    """
    learner: Callable[[], Any] = _unused
    return tuple(
        PropertyCell(
            property=configuration.family,
            cell=cell,
            dgp=configuration.law,
            outcome_learner=learner,
            treatment_learner=learner,
            n=configuration.n,
            replicates=configuration.replicates,
            seed=stream_seed(STUDY, "property_sample", configuration.family, configuration.name),
            role=configuration.role,
            estimand=estimand,
        )
        for configuration in CONFIGURATIONS
        for estimand, cell in zip(configuration.estimands, configuration.cells, strict=True)
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY)
    robustness_verdicts(summary, family="robustness_contract")
    return finish(summary, rates)


def ladder_se_ratio_upper(rows: pd.DataFrame) -> dict[str, float]:
    """The upper end of each ladder rung's 99% bootstrap SE-ratio interval.

    The reading rule routes a rung red only on coverage or its SE ratio to F19 when this is
    below :data:`~tests.studies.canonical_ctmle_oat_per_arm_strata_missing.ROUTED_SE_RATIO_CEILING`.
    """
    from tests.studies.evidence.properties import se_ratio_interval

    out: dict[str, float] = {}
    ladder = rows.loc[rows["property"] == "root_n_and_efficiency"]
    for cell, group in ladder.groupby("cell", sort=True):
        interval = se_ratio_interval(
            group.sort_values("replicate"),
            replicates=STUDY.margins.bootstrap_replicates,
            confidence_level=STUDY.margins.confidence_level,
            seed=stream_seed(STUDY, "reading_rule", str(cell)),
        )
        out[str(cell)] = float(interval.high)
    return out


def configuration_truths() -> dict[str, dict[str, float]]:
    """Each configuration's truths, by estimand, for the design test."""
    return {
        configuration.name: {
            estimand: float(configuration.law.truth()[estimand])
            for estimand in configuration.estimands
        }
        for configuration in CONFIGURATIONS
    }


def control_standardized_bias() -> float:
    """Condition (s): the control's population bias over its standard deviation."""
    return float(np.abs(CONTROL_BIAS) / CONTROL_SD)
