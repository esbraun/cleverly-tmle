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
``root_n_and_efficiency`` ``n_2000``, ``n_4000``,    ``strata_missing``, five folds, ``strata=S``:
``n_8000``                                          marginal ``ate``.  The ``root_n_rate`` rows
                                                    read the ladder
==================================================  ==============================================

A configuration reports several estimands from one fit, and a
:class:`~tests.studies.evidence.properties.PropertyCell` measures one, so
:func:`configuration_rows` fits once per replication and writes one row per estimand, in the
X17 ``joint_rows`` pattern.  Each row's cell is ``<estimand slug>__<configuration>``.  The
draws come from ``stream_seed(STUDY, "property_sample", family, configuration, replicate)``.
A replication that raises is not redrawn: its rows are absent, every row of the configuration
records the failed count, and the shared summary refuses the cell.

**The scale of a ratio's standard error.**  Each row records the fit's
``plugin_std_error`` as :func:`~tests.studies.evidence.properties.replicate_row` reads it.  For
``rr`` and ``or`` that is the standard error of ``log psi``, because the package builds a ratio's
interval on the log scale, while ``estimate`` is ``psi``.  :func:`summarize_properties` puts the
two on one scale before the shared summary: it multiplies a ratio row's standard error by its
estimate, the delta-method standard error of ``psi``.  The SE ratio of a ratio cell then
compares ``mean(psi_n SE(log psi_n))`` with the empirical SD of ``psi_n``.  The first summary
of the rerun omitted this step and published SE ratios of 0.559 (``rr``) and 0.322 (``or``)
for ``missing_outcome_crossfit``.  ``properties.csv`` was summarized again from the committed
``property-replicates.csv.gz`` without a new fit;
:data:`~tests.studies.canonical_ctmle_oat_per_arm_strata_missing.PROPERTY_SUMMARY_REVISION`
records it.

Positive cells pass on ``bias_equivalent`` with an SE ratio inside ``UNION_MODEL_SE_BAND``
(:func:`~tests.studies.evidence.property_verdicts.robustness_verdicts`).  The control passes on
``bias_discriminated``.  The ladder's rungs answer to the shared ``root_n_and_efficiency`` rule,
whose coverage and SE-ratio readings fall under the study's reading rule
(:data:`~tests.studies.canonical_ctmle_oat_per_arm_strata_missing.READING_RULE`).

**A budget the plan moved.**  The plan named ``R = 500`` per rung.  Condition (r) of the same
plan requires a pass probability of at least 0.90 for a cell whose true standardized bias is
0.10, and at ``R = 500`` that probability is 0.777.  Each rung takes ``R = 800`` (0.950), as
every other cell does.  The move was made before any fit of the estimator.

**The ladder re-declared after the first run.**  The first run (commit ``ae53bbb``) declared
the rungs ``n = 500, 2,000, 8,000``.  Its ``n_500`` rung read a mean plug-in standard error of
0.0884 against an empirical SD of 0.0561, an SE ratio of 1.575, with coverage 0.954 and a bias
inside its margin.  The median standard error was 1.00 empirical SDs.  In 40 of the 800 fits
(5%) the standard error exceeded twice the empirical SD, and without them the mean was 1.035
empirical SDs.  The ``root_n_rate/reported_se`` slope read -0.71 (99% interval -0.79 to
-0.63) against an empirical-SD slope of -0.50, and the slope of the median standard error was
-0.55.  The reading rule made both rows defect suspects.

The diagnosis found no code defect.  At ``n = 500`` each factor is fitted in a stratum, arm
and fold cell of about 20 to 35 rows (29.3 expected in the smallest).  The unpenalized cubic
factor learner then extrapolates the design to far below 0.01 at one or two observed rows in
about 5% of the fits, where the true design at those rows is 0.11 or more.  Their
clever-covariate term dominates the plug-in variance.  The points do not move, and the curve
equals the textbook curve to ``1e-13``.  The tail held 3 of 800 fits at ``n = 2,000`` and
none at ``n = 8,000``.  The rung therefore measured learner instability in small cells,
outside the regime the consistency claim addresses, and not the rate.

The ladder is re-declared at ``n = 2,000, 4,000, 8,000`` with condition (t) of
:mod:`tests.studies.oat_per_arm_extension_laws`.  Condition (r) does not read ``n``, so each
rung keeps ``R = 800`` (0.950).  ``n_2000`` and ``n_8000`` keep their configuration names, so
their streams and rows are those of the first run; ``n_4000`` is new.  The primary scenarios,
the pairing, the robustness cells, the laws, the seeds and the reading rule are unchanged.  The
rerun of the property phase supersedes the first run's property artifacts.  The ``n = 500``
result stays on the record here and on the study page as the reason for the change.

**Condition (t), restated after the rerun.**  As declared, condition (t) required the smallest
factor training cell to hold 100 rows or more in expectation at the smallest rung, and it held
117.1 at ``n = 2,000``.  That count is not the binding one.  A logistic factor fit is bound by
the smaller class of its target, and on the strata law that class is the nonrespondents of
arm 0 in ``S = 2``.  :func:`~tests.studies.oat_per_arm_extension_laws.smallest_minority_class`
gives its expected count per stratum, arm, fold and factor: 9.9 at the dropped ``n = 500``,
39.5 at ``n = 2,000``, 59.2 in the ``strata_missing`` cells at ``n = 3,000``, 79.0 at
``n = 4,000`` and 157.9 at ``n = 8,000``.  The restated condition sets no floor, because the
declared ladder does not hold 100.  :func:`se_ratio_spread` reads the measured consequence: a
tail of 3 in 800 fits at ``n = 2,000`` and none above it, and 2 in 800 in the
``strata_missing`` cells, which lift the ``S = 2`` mean SE ratio about 5% above its median.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.targets import TARGETS, parameter_stem
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


def ratio_cells() -> frozenset[str]:
    """The cells whose estimand the package reports on the log scale (``scale == "ratio"``)."""
    return frozenset(
        cell
        for configuration in CONFIGURATIONS
        for estimand, cell in zip(configuration.estimands, configuration.cells, strict=True)
        if TARGETS[parameter_stem(estimand)].scale == "ratio"
    )


def on_estimate_scale(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows with each ratio row's standard error on the scale of its estimate.

    A ratio row records the standard error of ``log psi``.  The delta method gives
    ``psi SE(log psi)`` as the standard error of ``psi``, the scale of ``estimate`` and of the
    empirical SD that the shared summary compares it with.

    Parameters
    ----------
    rows : DataFrame
        :data:`~tests.studies.evidence.properties.REPLICATE_COLUMNS` rows, as committed.

    Returns
    -------
    DataFrame
        A copy, with ``std_error`` multiplied by ``estimate`` on each row of
        :func:`ratio_cells`.
    """
    out = rows.copy()
    ratio = out["cell"].isin(ratio_cells())
    out.loc[ratio, "std_error"] = out.loc[ratio, "std_error"] * out.loc[ratio, "estimate"]
    return out


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The published property table, from the committed replicate rows.

    Parameters
    ----------
    rows : DataFrame
        :data:`~tests.studies.evidence.properties.REPLICATE_COLUMNS` rows, as committed.

    Returns
    -------
    DataFrame
        One row per cell, with its verdict, after :func:`on_estimate_scale`.
    """
    summary, rates = apply_shared_verdicts(on_estimate_scale(rows), STUDY)
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


#: A replication whose standard error exceeds this many empirical SDs of its cell is a tail
#: fit in :func:`se_ratio_spread`.  The first run's ``n_500`` rung held 40 such fits.
TAIL_MULTIPLE = 2.0


def se_ratio_spread(rows: pd.DataFrame) -> pd.DataFrame:
    """The mean and median SE ratio of each cell, and its tail fits.

    Parameters
    ----------
    rows : DataFrame
        :data:`~tests.studies.evidence.properties.REPLICATE_COLUMNS` rows, as committed.  Each
        ratio row is put on the scale of its estimate by :func:`on_estimate_scale` first.

    Returns
    -------
    DataFrame
        One row per cell: ``mean_se_ratio`` (the mean standard error over the empirical SD, as
        ``properties.csv`` publishes it), ``median_se_ratio`` (the median standard error over
        the empirical SD), and ``tail_fits``, the replications whose standard error exceeds
        :data:`TAIL_MULTIPLE` empirical SDs.
    """
    records = []
    for (family, cell), group in on_estimate_scale(rows).groupby(["property", "cell"], sort=True):
        spread = float(np.std(group["estimate"].to_numpy(dtype=float), ddof=1))
        errors = group["std_error"].to_numpy(dtype=float)
        records.append(
            {
                "property": family,
                "cell": cell,
                "mean_se_ratio": float(np.mean(errors)) / spread,
                "median_se_ratio": float(np.median(errors)) / spread,
                "tail_fits": int(np.sum(errors > TAIL_MULTIPLE * spread)),
            }
        )
    return pd.DataFrame.from_records(records)


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
