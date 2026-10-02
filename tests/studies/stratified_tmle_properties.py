"""Repeated-sampling properties of ordinary point-treatment TMLE with baseline strata.

Five families, each against the exact truths of :mod:`tests.studies.stratified_law`.  Declared
before any run:

=================================  ==================================================  =====  =====
family                             cells                                               n      R
=================================  ==================================================  =====  =====
``interval_calibration``           ``v<s>_<key>__correctly_specified`` for the six      2,000  2,400
                                   stratum parameters of each stratum, from in-sample
                                   fits with ``("ey", "ate", "att", "atc", "par")``,
                                   with shrunken-SE and noise controls
``simultaneous_coverage``          ``strata__*``: the default band over all 24          2,000  2,400
                                   reported parameters of the same fits
``interval_calibration``           ``v<s>_ate_crossfit__correctly_specified``, from      2,000  2,400
(cross-fitted)                     the default cross-fitted fits (``n_folds=10``,
                                   pooled) with ``("ey", "ate")``, with controls
``simultaneous_coverage``          ``crossfit_strata__*``: the band over the twelve      2,000  2,400
(cross-fitted)                     parameters of the same fits
``double_robustness``              ``v<s>_ate__{both_correct, outcome_correct,          2,000  1,200
                                   treatment_correct, both_wrong}``; a wrong model is
                                   intercept-only
``stratum_targeting_necessity``    ``v<s>_ate__stratified`` against                     2,000  1,200
                                   ``v<s>_ate__marginal_fluctuation`` on the same
                                   draws, with ``Q`` fitted on ``(A, W)`` only, in
                                   the two outer strata
=================================  ==================================================  =====  =====

Every positive fit uses the main-terms logistic learners of the primary scenario, which are
correct for this law.  Every band uses the shipped defaults: 1000 rademacher draws seeded by
``random_state=0``.  Each family reads the shared rule: the calibration rule with efficiency
ratios inside :data:`EFFICIENCY_RATIO_BAND`, the joint-coverage rule, the double-robustness
rule, and the necessity rule with displacement at least :data:`TARGETING_DISPLACEMENT`
empirical SD.

The marginal-fluctuation control is written longhand, as ``ltmle_properties.untargeted`` is.
It fits the same ``Q`` on ``(A, W)``, solves one logistic fluctuation per arm with clever
covariate ``I(A = a) / g_a``, and averages the targeted predictions inside each stratum.  The
fluctuation solves the marginal score only, so a ``Q`` that omits ``V`` stays biased inside a
stratum.  The stratified fit solves one score block per stratum and is consistent there,
because ``g`` is correct.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_law as law
from tests.studies.canonical_stratified_tmle import (
    ATE_LABELS,
    CALIBRATION_LABELS,
    CROSSFIT_LABELS,
    DOUBLE_ROBUST_CONFIGURATIONS,
    G_BOUNDS,
    NECESSITY_ARMS,
    NECESSITY_LABELS,
    NECESSITY_STRATA,
    STUDY,
    fit_cleverly,
    label_name,
    logistic,
)
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells

CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = 2_400
CROSSFIT_REPLICATES = 2_400
DOUBLE_ROBUST_N = 2_000
DOUBLE_ROBUST_REPLICATES = 1_200
NECESSITY_N = 2_000
NECESSITY_REPLICATES = 1_200
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGETING_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The estimands of the calibration fits, and of the cross-fitted fits.
CALIBRATION_ESTIMANDS = ("ey", "ate", "att", "atc", "par")
CROSSFIT_ESTIMANDS = ("ey", "ate")

#: The efficiency-bound SD of every calibration label.
EFFICIENCY_SD: dict[str, float] = {
    label: law.EFFICIENCY_SD[label_name(label)] for label in (*CALIBRATION_LABELS, *CROSSFIT_LABELS)
}

#: Each sampled batch: its family, its seed key, its size, its budget and its configuration.
BATCHES: tuple[tuple[str, str, int, int, str], ...] = (
    ("interval_calibration", "in_sample", CALIBRATION_N, CALIBRATION_REPLICATES, "in_sample"),
    ("interval_calibration", "crossfit", CALIBRATION_N, CROSSFIT_REPLICATES, "crossfit"),
    *(
        (
            "double_robustness",
            configuration,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
        )
        for configuration in DOUBLE_ROBUST_CONFIGURATIONS
    ),
    ("stratum_targeting_necessity", "paired", NECESSITY_N, NECESSITY_REPLICATES, "necessity"),
)


class ColumnSubset(BaseEstimator, ClassifierMixin):
    """A classifier fitted on the first ``keep`` design columns only.

    The outcome design is ``[A, W, V]``, so ``keep=2`` fits ``Q`` on ``(A, W)`` and omits the
    stratum.

    Parameters
    ----------
    base : estimator
        The classifier to fit on the kept columns.
    keep : int
        How many leading design columns it reads.
    """

    def __init__(self, base: Any = None, keep: int = 2) -> None:
        self.base = base
        self.keep = keep

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ColumnSubset:
        values = np.asarray(design, dtype=float)[:, : self.keep]
        self.model_ = clone(self.base).fit(values, target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)[:, : self.keep]
        return np.asarray(self.model_.predict_proba(values), dtype=float)


class StratifiedSampler:
    """The stratified law as a coverage-study sampler with its own exact truths."""

    name = "stratified_law"

    def truth(self) -> dict[str, float]:
        """Every marginal and stratum truth of the law."""
        return dict(law.TRUTH)

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        return law.sample(n, seed), dict(law.TRUTH)


def _seed(family: str, key: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, key, replicate)


def _arm_fluctuation(
    y: np.ndarray, arm: np.ndarray, initial: np.ndarray, covariate: np.ndarray
) -> float:
    """The logistic fluctuation of one arm's prediction, solving its marginal score."""
    offset = logit(np.clip(initial, 1e-9, 1.0 - 1e-9))
    epsilon = 0.0
    for _ in range(100):
        fitted = expit(offset + epsilon * covariate)
        score = float(np.sum(arm * covariate * (y - fitted)))
        slope = float(np.sum(arm * covariate**2 * fitted * (1.0 - fitted)))
        step = score / slope
        epsilon += step
        if abs(step) < 1e-13:
            break
    return epsilon


def marginal_fluctuation(frame: pd.DataFrame) -> dict[int, float]:
    """The stratum ATEs after one marginal fluctuation per arm, written longhand.

    The same ``Q`` the positive arm fits, on ``(A, W)``, and the same main-terms ``g`` on
    ``(W, V)`` inside the same bounds.  Each arm solves its own marginal score with clever
    covariate ``I(A = a) / g_a``; nothing solves a stratum's score.
    """
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    design = frame[["A", "W", "V"]].to_numpy(dtype=float)
    outcome = ColumnSubset(logistic(), keep=2).fit(design, y)
    treated = design.copy()
    treated[:, 0] = 1.0
    control = design.copy()
    control[:, 0] = 0.0
    q1 = outcome.predict_proba(treated)[:, 1]
    q0 = outcome.predict_proba(control)[:, 1]
    covariates = frame[["W", "V"]].to_numpy(dtype=float)
    g = np.clip(logistic().fit(covariates, a).predict_proba(covariates)[:, 1], *G_BOUNDS)
    epsilon1 = _arm_fluctuation(y, a, q1, 1.0 / g)
    epsilon0 = _arm_fluctuation(y, 1.0 - a, q0, 1.0 / (1.0 - g))
    star1 = expit(logit(np.clip(q1, 1e-9, 1.0 - 1e-9)) + epsilon1 / g)
    star0 = expit(logit(np.clip(q0, 1e-9, 1.0 - 1e-9)) + epsilon0 / (1.0 - g))
    v = frame["V"].to_numpy()
    return {stratum: float(np.mean((star1 - star0)[v == stratum])) for stratum in law.STRATA}


def _learners(configuration: str) -> dict[str, Any]:
    """The outcome and treatment learners of one robustness configuration."""
    outcome_right = configuration in {"both_correct", "outcome_correct"}
    treatment_right = configuration in {"both_correct", "treatment_correct"}
    return {
        "outcome_learner": logistic() if outcome_right else DummyClassifier(strategy="prior"),
        "treatment_learner": logistic() if treatment_right else DummyClassifier(strategy="prior"),
    }


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    name: str,
    result: Any,
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=float(law.TRUTH[name]),
        estimate=result[name],
        alpha=STUDY.margins.alpha,
    )


def _joint(result: Any, label: str, replicate: int, n: int, requested: int) -> list[dict[str, Any]]:
    return joint_coverage_rows(
        result,
        {name: float(law.TRUTH[name]) for name in result.estimates},
        tuple(result.estimates),
        label=label,
        replicate=replicate,
        n=n,
        requested=requested,
        pointwise_critical=CRITICAL,
    )


def _calibration(
    frame: pd.DataFrame, family: str, replicate: int, n: int, requested: int, crossfit: bool
) -> list[dict[str, Any]]:
    result = fit_cleverly(
        frame,
        estimands=CROSSFIT_ESTIMANDS if crossfit else CALIBRATION_ESTIMANDS,
        cross_fit=crossfit,
        simultaneous=True,
    )
    rows = [
        _row(
            family,
            f"{label}__correctly_specified",
            "positive",
            replicate,
            n,
            requested,
            label_name(label),
            result,
        )
        for label in (CROSSFIT_LABELS if crossfit else CALIBRATION_LABELS)
    ]
    label = "crossfit_strata" if crossfit else "strata"
    return rows + _joint(result, label, replicate, n, requested)


def _necessity(
    frame: pd.DataFrame, family: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    result = fit_cleverly(
        frame, estimands=("ate",), outcome_learner=ColumnSubset(logistic(), keep=2)
    )
    longhand = marginal_fluctuation(frame)
    rows: list[dict[str, Any]] = []
    for stratum, label in zip(NECESSITY_STRATA, NECESSITY_LABELS, strict=True):
        name = label_name(label)
        positive = _row(
            family, f"{label}__stratified", "positive", replicate, n, requested, name, result
        )
        rows.append(positive)
        rows.append(
            control_row(
                property_name=family,
                cell=f"{label}__marginal_fluctuation",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=positive["truth"],
                estimate=longhand[stratum],
                standard_error=float(result[name].std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    family, _, replicate, n, requested, seed, configuration = payload
    frame = law.sample(n, seed)
    if configuration in {"in_sample", "crossfit"}:
        return _calibration(frame, family, replicate, n, requested, configuration == "crossfit")
    if family == "double_robustness":
        result = fit_cleverly(frame, estimands=("ate",), **_learners(configuration))
        role = "control" if configuration == "both_wrong" else "positive"
        return [
            _row(
                family,
                f"{label}__{configuration}",
                role,
                replicate,
                n,
                requested,
                label_name(label),
                result,
            )
            for label in ATE_LABELS
        ]
    if family == "stratum_targeting_necessity":
        return _necessity(frame, family, replicate, n, requested)
    raise KeyError(family)  # pragma: no cover - declaration guard


def _payloads(
    replicates: int | None = None,
) -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    out: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for family, key, n, budget, configuration in BATCHES:
        count = budget if replicates is None else replicates
        out += [
            ((family, key, replicate, n, budget, _seed(family, key, replicate), configuration),)
            for replicate in range(count)
        ]
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Fit every declared replication, then derive the calibration controls."""
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=(*CALIBRATION_LABELS, *CROSSFIT_LABELS),
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


def generate_smoke_property_rows(*, n_jobs: int = 1, replicates: int = 1) -> pd.DataFrame:
    """Fit the first ``replicates`` declared replications of every batch, unsummarized."""
    outcomes = map_parallel(_fit_replication, _payloads(replicates), n_jobs=n_jobs)
    return pd.DataFrame([row for result in outcomes for row in result])


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared rules, the efficiency band, the joint rule and the necessity rule."""
    margins = STUDY.margins
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement",),
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    simultaneous_coverage_verdicts(summary, margins=margins)
    necessity_verdicts(
        summary,
        rows,
        family="stratum_targeting_necessity",
        labels=NECESSITY_LABELS,
        arms=NECESSITY_ARMS,
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    return finish(summary, rates)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with the law, the estimand and the stream root it reads."""
    sampler = StratifiedSampler()
    cells: list[PropertyCell] = []

    def add(family: str, key: str, cell: str, role: str, name: str, n: int, budget: int) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=sampler,
                outcome_learner=logistic,
                treatment_learner=logistic,
                n=n,
                replicates=budget,
                seed=_seed(family, key, 0),
                role=role,
                estimand=name,
            )
        )

    for labels, key, budget in (
        (CALIBRATION_LABELS, "in_sample", CALIBRATION_REPLICATES),
        (CROSSFIT_LABELS, "crossfit", CROSSFIT_REPLICATES),
    ):
        for label in labels:
            for kind, role in (
                ("correctly_specified", "positive"),
                ("shrunken_se_control", "control"),
                ("noise_control", "control"),
            ):
                add(
                    "interval_calibration",
                    key,
                    f"{label}__{kind}",
                    role,
                    label_name(label),
                    CALIBRATION_N,
                    budget,
                )
    for label, key, budget in (
        ("strata", "in_sample", CALIBRATION_REPLICATES),
        ("crossfit_strata", "crossfit", CROSSFIT_REPLICATES),
    ):
        cells.extend(
            joint_property_cells(
                label,
                n=CALIBRATION_N,
                replicates=budget,
                seed=_seed("interval_calibration", key, 0),
            )
        )
    for configuration in DOUBLE_ROBUST_CONFIGURATIONS:
        role = "control" if configuration == "both_wrong" else "positive"
        for label in ATE_LABELS:
            add(
                "double_robustness",
                configuration,
                f"{label}__{configuration}",
                role,
                label_name(label),
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
            )
    for label in NECESSITY_LABELS:
        for arm, role in zip(NECESSITY_ARMS, ("positive", "control"), strict=True):
            add(
                "stratum_targeting_necessity",
                "paired",
                f"{label}__{arm}",
                role,
                label_name(label),
                NECESSITY_N,
                NECESSITY_REPLICATES,
            )
    return tuple(cells)
