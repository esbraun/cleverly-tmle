"""Repeated-sampling properties of stratified incremental and MSM targeting.

Every family is measured against the exact truths of
:mod:`tests.studies.stratified_alternating_law`.  Declared before any run:

==================================  ============================================================  =====  =====
family                              cells                                                         n      R
==================================  ============================================================  =====  =====
``interval_calibration``            ``v<s>_<key>``: the fifteen stratum incremental parameters    2,000  2,000
                                    of in-sample fits with correct main-terms logistic learners,
                                    with shrunken-SE and noise controls
``simultaneous_coverage``           ``ipsi_strata__*``: the band over all twenty parameters of    2,000  2,000
                                    the same fits
``stratum_targeting_necessity``     ``v<s>_ey_x2__stratified`` against                           2,000  1,200
                                    ``__marginal_fluctuation`` in the two outer strata, with
                                    ``Q`` fitted on ``(A, W)`` only
``interval_calibration`` (MSM)      ``logit_<scope>_<term>``: the logit-link MSM ``(1, a, W)``,   2,000  2,000
                                    marginal and stratum, with controls
``double_robustness`` (MSM)         ``v<s>_a__{both_correct, outcome_correct,                     2,000  1,200
                                    treatment_correct}`` and ``v2_a__both_wrong``; a wrong
                                    model is intercept-only
``targeting_necessity`` (MSM)       ``v<s>_a__targeted`` against ``v<s>_a__untargeted`` in        2,000  1,200
                                    strata 0 and 1, the stratum projection of the initial ``Q``
                                    fitted on ``(A, W)``
``interval_calibration`` (dose)     ``continuous_<link>_<scope>_<term>``: the identity and logit  2,000  2,000
                                    MSMs ``(1, a)`` of L2 over :data:`GRID`, with controls
``simultaneous_coverage`` (dose)    ``continuous_strata__*``: the band over the eight             2,000  2,000
                                    identity-link coefficients of the same fits
==================================  ============================================================  =====  =====

One configuration shares one fit per replication, seeded by
``stream_seed(STUDY, "property_sample", family, configuration, r)``: the fifteen incremental
calibration cells and both band cells read one fit, and so do the twelve logit calibration
cells, and each dose link's eight cells.  Correct learners are main-terms logistic
regressions (``C=1e6``) on ``(A, W, V)`` and ``(W, V)``, and for L2 a linear (identity) or
logistic (logit) outcome regression with ``a:V`` and the package's binned density on six bins.

The incremental marginal-fluctuation control is the package's own unstratified fit on the
same draw, its targeted mixture averaged inside the stratum.  It solves no stratum's score, so
a ``Q`` that omits ``V`` stays biased inside a stratum.

Three controls of the plan are not declared, because
``tests/unit/test_stratified_alternating_design.py`` shows on L1 that they cannot fail.  A
mechanism that omits ``W`` is repaired by the stratum mechanism tilt, whose three covariates
saturate the three ``W`` cells (at most 0.09 SD off).  A both-wrong MSM fit is 0.0 and 0.22 SD
off in strata 0 and 1, so only stratum 2 (0.47 SD) carries that control.  A marginal MSM
fluctuation is at most 0.54 SD off in every stratum, so the MSM family pairs the stratified fit
with the untargeted projection instead.  The exact-law tests carry
the mechanism block and the stratum MSM blocks instead
(``tests/unit/test_stratified_incremental_exact.py``,
``tests/unit/test_stratified_msm_exact.py``).

The study claims calibration, not efficiency: the binned density of the dose fits is not the
law's density, so no efficiency ratio is published.  The efficiency bounds size the noise
controls only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.msm import MSM, solve_projection
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_alternating_law as law
from tests.studies.canonical_stratified_incremental_msm import (
    BOTH_WRONG_LABELS,
    CALIBRATION_KINDS,
    CONTINUOUS_LABELS,
    DOUBLE_ROBUST_CONFIGURATIONS,
    DOUBLE_ROBUST_LABELS,
    IPSI_LABELS,
    MSM_LABELS,
    NECESSITY_ARMS,
    NECESSITY_LABELS,
    NECESSITY_STRATA,
    STUDY,
    TARGETING_ARMS,
    TARGETING_LABELS,
    fit_cleverly,
    label_name,
    msm_label_name,
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
CALIBRATION_REPLICATES = 2_000
NECESSITY_N = 2_000
NECESSITY_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
DOUBLE_ROBUST_REPLICATES = 1_200
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.25
#: The study publishes no efficiency ratio; see the module docstring.
EFFICIENCY_RATIO_BAND = None
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
DENSITY_BINS = 6

#: The MSM the L1 families fit, and the dose MSM of L2.
L1_MSM = MSM.linear(modifiers=("W",), interaction=False, link="logit")


def l2_msm(link: str) -> MSM:
    return MSM.linear(doses=law.GRID, link=link)  # type: ignore[arg-type]


def _efficiency_bounds() -> dict[str, float]:
    bounds = {label: law.efficiency_sd(label_name(label)) for label in IPSI_LABELS}
    for label in MSM_LABELS:
        bounds[label] = law.efficiency_sd(msm_label_name(label), "logit")
    for label in CONTINUOUS_LABELS:
        link = label.split("_")[1]
        scope = label.split("_")[2]
        stratum = None if scope == "marginal" else int(scope[1:])
        term = 0 if label.endswith("intercept") else 1
        bounds[label] = float(law.l2_efficiency_sd(link, stratum)[term])
    return bounds


#: The SD each noise control is sized by.  For the dose labels it is the SD of the curve the
#: package reports, at the law's own density.
EFFICIENCY_SD: dict[str, float] = _efficiency_bounds()

#: Each sampled batch: its family, its configuration key, its size and its budget.
BATCHES: tuple[tuple[str, str, int, int], ...] = (
    ("interval_calibration", "ipsi", CALIBRATION_N, CALIBRATION_REPLICATES),
    ("interval_calibration", "msm_logit", CALIBRATION_N, CALIBRATION_REPLICATES),
    ("interval_calibration", "continuous_identity", CALIBRATION_N, CALIBRATION_REPLICATES),
    ("interval_calibration", "continuous_logit", CALIBRATION_N, CALIBRATION_REPLICATES),
    ("stratum_targeting_necessity", "paired", NECESSITY_N, NECESSITY_REPLICATES),
    *(
        ("double_robustness", configuration, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES)
        for configuration in (*DOUBLE_ROBUST_CONFIGURATIONS, "both_wrong")
    ),
    ("targeting_necessity", "paired", NECESSITY_N, NECESSITY_REPLICATES),
)


def logistic() -> LogisticRegression:
    """The unpenalized main-terms logistic regression the correct L1 fits use."""
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


class Columns(BaseEstimator, ClassifierMixin):
    """A classifier fitted on the declared design columns only.

    The outcome design is ``[A, W, V]`` and the treatment design ``[W, V]``.

    Parameters
    ----------
    base : estimator
        The classifier fitted on the kept columns.
    keep : tuple of int
        The design columns it reads.
    """

    def __init__(self, base: Any = None, keep: tuple[int, ...] = (0, 1)) -> None:
        self.base = base
        self.keep = keep

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> Columns:
        values = np.asarray(design, dtype=float)[:, list(self.keep)]
        self.model_ = clone(self.base).fit(values, target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)[:, list(self.keep)]
        return np.asarray(self.model_.predict_proba(values), dtype=float)


class DoseInteraction(BaseEstimator, RegressorMixin, ClassifierMixin):
    """The correct L2 outcome regression: ``(A, W, V, A V)``, linear or logistic.

    Parameters
    ----------
    link : {"identity", "logit"}
        Linear least squares for the identity outcome, logistic for the binary one.
    """

    def __init__(self, link: str = "identity") -> None:
        self.link = link

    @staticmethod
    def _design(design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        return np.column_stack([values, values[:, 0] * values[:, 2]])

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DoseInteraction:
        model = LinearRegression() if self.link == "identity" else logistic()
        self.model_ = model.fit(self._design(design), target, sample_weight=sample_weight)
        if self.link != "identity":
            self.classes_ = self.model_.classes_
        return self

    def predict(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(design)), dtype=float)

    def predict_proba(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._design(design)), dtype=float)


class L1Sampler:
    """L1 as a coverage-study sampler: incremental truths, or one link's MSM truths."""

    def __init__(self, link: str | None = None) -> None:
        self.link = link
        self.name = "stratified_l1" if link is None else f"stratified_l1_msm_{link}"

    def truth(self) -> dict[str, float]:
        return dict(law.truths(self.link))

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        return law.sample(n, seed), self.truth()


class L2Sampler:
    """L2 as a coverage-study sampler, with one link's truths."""

    def __init__(self, link: str) -> None:
        self.link = link
        self.name = f"stratified_l2_{link}"

    def truth(self) -> dict[str, float]:
        return law.l2_truths(self.link)

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        return law.l2_sample(n, seed), self.truth()


def _seed(family: str, key: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, key, replicate)


def fit_msm(frame: pd.DataFrame, *, strata: bool = True, **learners: Any) -> Any:
    """The in-sample logit-link MSM fit of L1."""
    settings = {"outcome_learner": logistic(), "treatment_learner": logistic(), **learners}
    return (
        TMLE(
            msm=L1_MSM,
            cross_fit=False,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            **settings,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W", "V"],
            **({"strata": ["V"]} if strata else {}),
        )
        .single()
    )


def fit_dose(frame: pd.DataFrame, link: str, *, simultaneous: bool = False) -> Any:
    """The in-sample stratified dose MSM fit of L2."""
    return (
        TMLE(
            msm=l2_msm(link),
            outcome_learner=DoseInteraction(link),
            treatment_learner=logistic(),
            density_bins=DENSITY_BINS,
            cross_fit=False,
            simultaneous=simultaneous,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Yc" if link == "identity" else "Yd",
            treatment="D",
            treatment_kind="continuous",
            covariates=["W", "V"],
            strata=["V"],
        )
        .single()
    )


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: Any,
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=float(truth),
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def _calibration(
    frame: pd.DataFrame, key: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    family = "interval_calibration"
    if key == "ipsi":
        result = fit_cleverly(
            frame, outcome_learner=logistic(), treatment_learner=logistic(), simultaneous=True
        )
        truth = law.truths()
        rows = [
            _row(
                family,
                f"{label}__correctly_specified",
                "positive",
                replicate,
                n,
                requested,
                truth[label_name(label)],
                result[label_name(label)],
            )
            for label in IPSI_LABELS
        ]
        return rows + joint_coverage_rows(
            result,
            truth,
            tuple(result.estimates),
            label="ipsi_strata",
            replicate=replicate,
            n=n,
            requested=requested,
            pointwise_critical=CRITICAL,
        )
    if key == "msm_logit":
        result = fit_msm(frame)
        truth = law.truths("logit")
        return [
            _row(
                family,
                f"{label}__correctly_specified",
                "positive",
                replicate,
                n,
                requested,
                truth[msm_label_name(label)],
                result[msm_label_name(label)],
            )
            for label in MSM_LABELS
        ]
    link = key.split("_", 1)[1]
    result = fit_dose(frame, link, simultaneous=link == "identity")
    truth = law.l2_truths(link)
    labels = [label for label in CONTINUOUS_LABELS if label.split("_")[1] == link]
    rows = [
        _row(
            family,
            f"{label}__correctly_specified",
            "positive",
            replicate,
            n,
            requested,
            truth[msm_label_name(label)],
            result[msm_label_name(label)],
        )
        for label in labels
    ]
    if link == "identity":
        rows += joint_coverage_rows(
            result,
            truth,
            tuple(result.estimates),
            label="continuous_strata",
            replicate=replicate,
            n=n,
            requested=requested,
            pointwise_critical=CRITICAL,
        )
    return rows


def marginal_ipsi(result: Any, frame: pd.DataFrame, name: str, stratum: int) -> float:
    """The stratum average of an unstratified incremental fit's targeted mixture."""
    fluctuation = result.fluctuations["ipsi"]
    tilts = result.nuisance.incremental.at(fluctuation.mechanism.propensity)
    arms = np.column_stack(
        [result.nuisance.scaler.unscale_levels(fluctuation.targeted.arms[a]) for a in (0.0, 1.0)]
    )
    mixtures = np.einsum("ikr,ik->ir", tilts.values, arms)
    code = float(tilts.names.index(name))
    inside = frame["V"].to_numpy() == stratum
    return float(np.mean(mixtures[inside, int(code)]))


def _necessity(frame: pd.DataFrame, replicate: int, n: int, requested: int) -> list[dict[str, Any]]:
    family = "stratum_targeting_necessity"
    outcome = Columns(logistic(), keep=(0, 1))
    stratified = fit_cleverly(frame, outcome_learner=outcome, treatment_learner=logistic())
    marginal = fit_cleverly(
        frame,
        outcome_learner=Columns(logistic(), keep=(0, 1)),
        treatment_learner=logistic(),
        strata=False,
    )
    truth = law.truths()
    rows: list[dict[str, Any]] = []
    for stratum, label in zip(NECESSITY_STRATA, NECESSITY_LABELS, strict=True):
        name = label_name(label)
        positive = _row(
            family,
            f"{label}__stratified",
            "positive",
            replicate,
            n,
            requested,
            truth[name],
            stratified[name],
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
                estimate=marginal_ipsi(marginal, frame, "odds x2", stratum),
                standard_error=float(stratified[name].std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _double_robustness(
    frame: pd.DataFrame, configuration: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    outcome_right = configuration in {"both_correct", "outcome_correct"}
    treatment_right = configuration in {"both_correct", "treatment_correct"}
    result = fit_msm(
        frame,
        outcome_learner=logistic() if outcome_right else DummyClassifier(strategy="prior"),
        treatment_learner=logistic() if treatment_right else DummyClassifier(strategy="prior"),
    )
    truth = law.truths("logit")
    role = "control" if configuration == "both_wrong" else "positive"
    labels = BOTH_WRONG_LABELS if configuration == "both_wrong" else DOUBLE_ROBUST_LABELS
    return [
        _row(
            "double_robustness",
            f"{label}__{configuration}",
            role,
            replicate,
            n,
            requested,
            truth[f"msm[a][V={label[1]}]"],
            result[f"msm[a][V={label[1]}]"],
        )
        for label in labels
    ]


def initial_projection(result: Any, frame: pd.DataFrame, stratum: int, term: int) -> float:
    """The stratum projection of a stratified MSM fit's initial regression, untargeted."""
    msm = result.nuisance.msm
    raw = np.column_stack(
        [result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[a]) for a in msm.arms]
    )
    inside = np.flatnonzero(frame["V"].to_numpy() == stratum)
    beta = solve_projection(
        msm.design[inside], msm.weights[inside], raw[inside], np.ones(inside.size), msm.link
    ).beta
    return float(beta[term])


def _targeting(frame: pd.DataFrame, replicate: int, n: int, requested: int) -> list[dict[str, Any]]:
    family = "targeting_necessity"
    result = fit_msm(frame, outcome_learner=Columns(logistic(), keep=(0, 1)))
    truth = law.truths("logit")
    rows: list[dict[str, Any]] = []
    for label in TARGETING_LABELS:
        stratum = int(label[1])
        name = f"msm[a][V={stratum}]"
        positive = _row(
            family,
            f"{label}__targeted",
            "positive",
            replicate,
            n,
            requested,
            truth[name],
            result[name],
        )
        rows.append(positive)
        rows.append(
            control_row(
                property_name=family,
                cell=f"{label}__untargeted",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=positive["truth"],
                estimate=initial_projection(result, frame, stratum, 1),
                standard_error=float(result[name].std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _fit_replication(payload: tuple[str, str, int, int, int, int]) -> list[dict[str, Any]]:
    family, key, replicate, n, requested, seed = payload
    frame = law.l2_sample(n, seed) if key.startswith("continuous") else law.sample(n, seed)
    if family == "interval_calibration":
        return _calibration(frame, key, replicate, n, requested)
    if family == "stratum_targeting_necessity":
        return _necessity(frame, replicate, n, requested)
    if family == "double_robustness":
        return _double_robustness(frame, key, replicate, n, requested)
    if family == "targeting_necessity":
        return _targeting(frame, replicate, n, requested)
    raise KeyError(family)  # pragma: no cover - declaration guard


def _payloads(budget: int | None = None) -> list[tuple[tuple[str, str, int, int, int, int]]]:
    out: list[tuple[tuple[str, str, int, int, int, int]]] = []
    for family, key, n, declared in BATCHES:
        count = declared if budget is None else budget
        out += [
            ((family, key, replicate, n, declared, _seed(family, key, replicate)),)
            for replicate in range(count)
        ]
    return out


def _calibration_labels() -> tuple[str, ...]:
    return (*IPSI_LABELS, *MSM_LABELS, *CONTINUOUS_LABELS)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every declared replication, or the first ``budget`` of each batch, then the controls."""
    outcomes = map_parallel(_fit_replication, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=_calibration_labels(),
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
    """The shared rules, the joint rule, the mechanism rule and both necessity rules."""
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
    for family, labels, arms in (
        ("stratum_targeting_necessity", NECESSITY_LABELS, NECESSITY_ARMS),
        ("targeting_necessity", TARGETING_LABELS, TARGETING_ARMS),
    ):
        necessity_verdicts(
            summary,
            rows,
            family=family,
            labels=labels,
            arms=arms,
            column="targeting_displacement",
            threshold=TARGETING_DISPLACEMENT,
        )
    return finish(summary, rates)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with the law, the estimand and the stream root it reads."""
    cells: list[PropertyCell] = []

    def add(
        family: str, key: str, cell: str, role: str, name: str, sampler: Any, n: int, budget: int
    ) -> None:
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

    roles = dict(zip(CALIBRATION_KINDS, ("positive", "control", "control"), strict=True))
    for label in IPSI_LABELS:
        for kind, role in roles.items():
            add(
                "interval_calibration",
                "ipsi",
                f"{label}__{kind}",
                role,
                label_name(label),
                L1Sampler(),
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
            )
    for label in MSM_LABELS:
        for kind, role in roles.items():
            add(
                "interval_calibration",
                "msm_logit",
                f"{label}__{kind}",
                role,
                msm_label_name(label),
                L1Sampler("logit"),
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
            )
    for label in CONTINUOUS_LABELS:
        link = label.split("_")[1]
        for kind, role in roles.items():
            add(
                "interval_calibration",
                f"continuous_{link}",
                f"{label}__{kind}",
                role,
                msm_label_name(label),
                L2Sampler(link),
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
            )
    for label, key in (("ipsi_strata", "ipsi"), ("continuous_strata", "continuous_identity")):
        cells.extend(
            joint_property_cells(
                label,
                n=CALIBRATION_N,
                replicates=CALIBRATION_REPLICATES,
                seed=_seed("interval_calibration", key, 0),
            )
        )
    for label in NECESSITY_LABELS:
        for arm, role in zip(NECESSITY_ARMS, ("positive", "control"), strict=True):
            add(
                "stratum_targeting_necessity",
                "paired",
                f"{label}__{arm}",
                role,
                label_name(label),
                L1Sampler(),
                NECESSITY_N,
                NECESSITY_REPLICATES,
            )
    for configuration in (*DOUBLE_ROBUST_CONFIGURATIONS, "both_wrong"):
        role = "control" if configuration == "both_wrong" else "positive"
        labels = BOTH_WRONG_LABELS if configuration == "both_wrong" else DOUBLE_ROBUST_LABELS
        for label in labels:
            add(
                "double_robustness",
                configuration,
                f"{label}__{configuration}",
                role,
                f"msm[a][V={label[1]}]",
                L1Sampler("logit"),
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
            )
    for label in TARGETING_LABELS:
        for arm, role in zip(TARGETING_ARMS, ("positive", "control"), strict=True):
            add(
                "targeting_necessity",
                "paired",
                f"{label}__{arm}",
                role,
                f"msm[a][V={label[1]}]",
                L1Sampler("logit"),
                NECESSITY_N,
                NECESSITY_REPLICATES,
            )
    return tuple(cells)
