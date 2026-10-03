"""Joint-coverage cells for the default simultaneous band of each shipped fit shape.

One family, ``simultaneous_coverage``.  Each shape of :data:`tests.studies.default_bands.SHAPES`
publishes ``<label>__simultaneous_band`` and, unless the design computation below shows it
cannot discriminate, ``<label>__pointwise_joint_control``.  Both rows read one fit:
:func:`tests.studies.evidence.simultaneous.joint_coverage_rows` builds them from the fit's band
and its pointwise intervals, over every estimate the fit reports.

The design computation fixes each cell's budget before any run.  It averages the covariance
of the reported influence curves over :data:`DESIGN_FITS` independent fits at the cell's own
size, which estimates the asymptotic correlation of the family.  Then it reads three numbers
off :data:`DESIGN_DRAWS` normal draws at :data:`DESIGN_SEED`: the pointwise joint coverage
``p0``, the oracle band critical value, and the probability that the pointwise control fails
its rule at the declared budget.  A clustered fit's curves are summed to the cluster first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, norm

from cleverly.utils.parallel import map_parallel
from tests.discrete_law_competing import TRUTH as COMPETING_TRUTH
from tests.parallel import STUDY_JOBS
from tests.studies import (
    canonical_categorical_ltmle,
    canonical_categorical_ltmle_crossfit,
    canonical_cde_tmle,
    canonical_clustered_tmle,
    canonical_deterministic_regimes,
    canonical_drtmle,
    canonical_incremental_interventions,
    canonical_learned_weighted_tmle,
    canonical_longitudinal_msm,
    canonical_ltmle_competing_crossfit,
    canonical_ltmle_crossfit,
    canonical_ltmle_survival_crossfit,
    canonical_mar_drtmle,
    canonical_mar_tmle,
    canonical_multi_arm_drtmle,
    canonical_point_msm,
    canonical_shift_policies,
    canonical_stochastic_regimes,
    canonical_tmle,
    canonical_weighted_ltmle,
    canonical_weighted_ltmle_crossfit,
    canonical_weighted_tmle,
    default_bands,
    fold_evaluated_cvtmle,
)
from tests.studies.default_bands import SHAPES, STUDY, Shape
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import (
    CONTROL,
    FAMILY,
    joint_coverage_rows,
    joint_property_cells,
)

CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: Independent fits whose influence-curve covariances the design computation averages,
#: unless a shape declares fewer.
DESIGN_FITS = 10
#: The normal draws behind each design number, and their seed.
DESIGN_DRAWS = 2_000_000
DESIGN_SEED = 20261002
#: The smallest control power the design accepts at a cell's declared budget.
MINIMUM_CONTROL_POWER = 0.99

Sample = tuple[pd.DataFrame, dict[str, float]]


@dataclass(frozen=True)
class Source:
    """How one shape draws a sample and fits it.

    Parameters
    ----------
    draw : callable
        ``(n, seed) -> (frame, truth)`` through the source study's own sampler.
    fit : callable
        ``frame -> result`` through the source study's own subject fit, with the band on.
    twins : dict
        Reported names whose truth is an exact identity with another name's.
    """

    draw: Callable[[int, int], Sample]
    fit: Callable[[pd.DataFrame], Any]
    twins: dict[str, str]


def _first_scenario(module: Any) -> str:
    return str(next(iter(module.STUDY.scenarios)))


def _plain(module: Any, scenario: str | None = None) -> Source:
    """A source whose ``fit_cleverly`` takes the frame alone."""
    name = _first_scenario(module) if scenario is None else scenario
    return Source(
        draw=lambda n, seed: module.draw_from_seed(name, n, seed),
        fit=lambda frame: module.fit_cleverly(frame, simultaneous=True),
        twins={},
    )


def _scenario_fit(module: Any, scenario: str) -> Source:
    """A source whose ``fit_cleverly`` also takes the scenario."""
    return Source(
        draw=lambda n, seed: module.draw_from_seed(scenario, n, seed),
        fit=lambda frame: module.fit_cleverly(frame, scenario, simultaneous=True),
        twins={},
    )


def _multi_arm_drtmle() -> Source:
    """The multi-arm DR-TMLE subject fit, without the source study's bound audit.

    The source study raises when its propensity bound activates, because it publishes
    ``bound_active`` as ``False`` for every replication.  The band does not depend on that
    claim, and a truncated replication is still the shipped fit.  The first declared run met
    one such sample on this study's own seeds and stopped before any joint verdict, so the
    audit is switched off here and the replication is kept as drawn.
    """
    module = canonical_multi_arm_drtmle
    scenario = _first_scenario(module)
    return Source(
        draw=lambda n, seed: module.draw_from_seed(scenario, n, seed),
        fit=lambda frame: module.fit_cleverly(
            frame, scenario, simultaneous=True, require_unbounded=False
        ),
        twins={},
    )


def _cde(level: int) -> Source:
    scenario = canonical_cde_tmle.SCENARIOS[level]
    return Source(
        draw=lambda n, seed: canonical_cde_tmle.draw_from_seed(scenario, n, seed),
        fit=lambda frame: canonical_cde_tmle.fit_cleverly(frame, simultaneous=True)[float(level)],
        twins={},
    )


_RULE = "treat then continue if l2 positive"

#: The survival plan that follows the rule assigns treatment at node one, as ``always`` does,
#: so its first-horizon risk and contrast are the ``always`` ones.
_SURVIVAL_TWINS = {
    f"risk_regimen[{_RULE} @ t=1]": "risk_regimen[always @ t=1]",
    f"ate_regimen[{_RULE} vs never @ t=1]": "ate_regimen[always vs never @ t=1]",
}


def _competing() -> Source:
    module = canonical_ltmle_competing_crossfit
    scenario = _first_scenario(module)

    def draw(n: int, seed: int) -> Sample:
        frame, _ = module.draw_from_seed(scenario, n, seed)
        return frame, {name: float(value) for name, value in COMPETING_TRUTH.items()}

    return Source(
        draw=draw, fit=lambda frame: module.fit_cleverly(frame, simultaneous=True), twins={}
    )


def _survival() -> Source:
    source = _plain(canonical_ltmle_survival_crossfit)
    return Source(draw=source.draw, fit=source.fit, twins=dict(_SURVIVAL_TWINS))


def _default_fit() -> Source:
    return Source(
        draw=lambda n, seed: default_bands.draw_from_seed(default_bands.SCENARIO, n, seed),
        fit=default_bands.fit_cleverly,
        twins={},
    )


SOURCES: dict[str, Source] = {
    "default_fit": _default_fit(),
    "ordinary_binary": _scenario_fit(canonical_tmle, "binary"),
    "fold_evaluated": _scenario_fit(fold_evaluated_cvtmle, "binary"),
    "weighted": _plain(canonical_weighted_tmle),
    "learned_weighted": _plain(canonical_learned_weighted_tmle),
    "missing_outcome": _plain(canonical_mar_tmle),
    "missing_outcome_drtmle": _plain(canonical_mar_drtmle),
    "drtmle": _scenario_fit(canonical_drtmle, "both_correct"),
    "multi_arm_drtmle": _multi_arm_drtmle(),
    "cde_z0": _cde(0),
    "cde_z1": _cde(1),
    "point_msm": _plain(canonical_point_msm),
    "clustered": _plain(canonical_clustered_tmle),
    "shift_grid": _plain(canonical_shift_policies),
    "incremental_grid": _plain(canonical_incremental_interventions),
    "stochastic_regimes": _plain(canonical_stochastic_regimes),
    "deterministic_regimes": _plain(canonical_deterministic_regimes),
    "ltmle_crossfit": _plain(canonical_ltmle_crossfit),
    "survival_crossfit": _survival(),
    "competing_crossfit": _competing(),
    "weighted_ltmle": _plain(canonical_weighted_ltmle),
    "weighted_ltmle_crossfit": _plain(canonical_weighted_ltmle_crossfit),
    "categorical_ltmle": _plain(canonical_categorical_ltmle),
    "categorical_ltmle_crossfit": _plain(canonical_categorical_ltmle_crossfit),
    "longitudinal_msm": _plain(canonical_longitudinal_msm),
}

SHAPE_BY_LABEL: dict[str, Shape] = {shape.label: shape for shape in SHAPES}


def _seed(label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", FAMILY, label, replicate)


def full_truth(label: str, result: Any, truth: dict[str, float]) -> dict[str, float]:
    """The truth of every name the fit reports, filling each exact twin from its pair."""
    twins = SOURCES[label].twins
    out: dict[str, float] = {}
    for name in result.estimates:
        key = twins.get(name, name)
        if key not in truth:
            raise KeyError(f"{label} has no truth for {name!r}")
        out[name] = float(truth[key])
    return out


def fit_shape(label: str, n: int, seed: int) -> tuple[Any, dict[str, float]]:
    """One replication's fit of ``label`` and the truth of every name it reports."""
    source = SOURCES[label]
    frame, truth = source.draw(n, seed)
    result = source.fit(frame)
    return result, full_truth(label, result, truth)


def _fit_replication(payload: tuple[str, int, int, int, int]) -> list[dict[str, Any]]:
    label, replicate, n, requested, seed = payload
    result, truth = fit_shape(label, n, seed)
    rows = joint_coverage_rows(
        result,
        truth,
        tuple(result.estimates),
        label=label,
        replicate=replicate,
        n=n,
        requested=requested,
        pointwise_critical=CRITICAL,
    )
    if not SHAPE_BY_LABEL[label].pointwise_control:
        rows = [row for row in rows if not str(row["cell"]).endswith(CONTROL)]
    return rows


def _payloads(replicates: int | None = None) -> list[tuple[tuple[str, int, int, int, int]]]:
    out: list[tuple[tuple[str, int, int, int, int]]] = []
    for shape in SHAPES:
        count = shape.replicates if replicates is None else replicates
        out += [
            ((shape.label, replicate, shape.n, count, _seed(shape.label, replicate)),)
            for replicate in range(count)
        ]
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Fit every declared replication of every joint cell."""
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def generate_smoke_property_rows(*, n_jobs: int = 1, replicates: int = 1) -> pd.DataFrame:
    """Fit the first ``replicates`` declared replications of every cell, unsummarized."""
    outcomes = map_parallel(_fit_replication, _payloads(replicates), n_jobs=n_jobs)
    return pd.DataFrame([row for result in outcomes for row in result])


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, read by the joint-coverage rule."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every joint cell, each declaring the zero-truth law its rows are read against."""
    return tuple(
        cell
        for shape in SHAPES
        for cell in joint_property_cells(
            shape.label,
            n=shape.n,
            replicates=shape.replicates,
            seed=_seed(shape.label, 0),
            control=shape.pointwise_control,
        )
    )


# ------------------------------------------------------------------------------ design


def _curves(result: Any) -> np.ndarray:
    """The reported curves, summed to the cluster when the fit is clustered."""
    curves = np.column_stack([result[name].influence_curve for name in result.estimates])
    cluster = getattr(getattr(result, "data", None), "cluster", None)
    if cluster is None:
        return curves
    codes = np.asarray(cluster)
    _, inverse = np.unique(codes, return_inverse=True)
    summed = np.zeros((int(inverse.max()) + 1, curves.shape[1]))
    np.add.at(summed, inverse, curves)
    return summed


def design_correlation(label: str) -> np.ndarray:
    """The family's correlation, from the curve covariance averaged over the shape's fits."""
    shape = SHAPE_BY_LABEL[label]
    total: np.ndarray | None = None
    for index in range(shape.design_fits):
        result, _ = fit_shape(label, shape.n, stream_seed(STUDY, "design", label, index))
        curves = _curves(result)
        centred = curves - curves.mean(axis=0)
        covariance = centred.T @ centred / len(centred)
        total = covariance if total is None else total + covariance
    assert total is not None
    sd = np.sqrt(np.diag(total))
    return np.asarray(total / np.outer(sd, sd), dtype=float)


@dataclass(frozen=True)
class Design:
    """The design numbers of one joint cell.

    Parameters
    ----------
    p0 : float
        The asymptotic joint coverage of the pointwise intervals.
    critical : float
        The oracle band critical value.
    control_power : float
        The probability that the pointwise control fails its rule at the budget.
    """

    p0: float
    critical: float
    control_power: float


def control_power(p0: float, replicates: int, *, confidence_level: float = 0.99) -> float:
    """The probability that the control's exact upper endpoint falls below the nominal rate."""
    successes = np.arange(replicates + 1)
    tail = (1.0 - confidence_level) / 2.0
    upper = np.where(
        successes < replicates,
        beta.ppf(1.0 - tail, successes + 1, np.maximum(replicates - successes, 1)),
        1.0,
    )
    passing = successes[upper < 1.0 - STUDY.margins.alpha]
    if passing.size == 0:
        return 0.0
    return float(binom.cdf(int(passing.max()), replicates, p0))


def design(correlation: np.ndarray, replicates: int) -> Design:
    """``p0``, the oracle critical value and the control power for one family."""
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    rng = np.random.default_rng(DESIGN_SEED)
    maxima = np.empty(DESIGN_DRAWS)
    block = 200_000
    for start in range(0, DESIGN_DRAWS, block):
        draws = rng.standard_normal((block, correlation.shape[0])) @ factor.T
        maxima[start : start + block] = np.abs(draws).max(axis=1)
    p0 = float(np.mean(maxima <= CRITICAL))
    critical = float(np.quantile(maxima, 1.0 - STUDY.margins.alpha))
    return Design(p0=p0, critical=critical, control_power=control_power(p0, replicates))
