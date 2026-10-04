"""Property families of ``cross-fitted-longitudinal-msm``.

Every family but two reads the finite-support law of :mod:`tests.discrete_law_longitudinal`,
the four regimens and the declared projection of ``longitudinal-msm``, and five outer folds.
The truth is the fixed projection of the law's regimen means.  The two exceptions read
``make_longitudinal``: the overfitting pair, whose trees need a continuous history, and the
band cell, which reuses the primary subject.

=========================  ==================================================================
family                     cells, replications and size
=========================  ==================================================================
``double_robustness``      both terms in four nuisance configurations; 1,000 at n = 2,000
``root_n_and_efficiency``  both terms at n = 2,000 and 8,000; 700 each.  The n = 500 rung of
                           ``longitudinal-msm`` was dropped before the run: see below
``root_n_rate``            the two rate rows of each term, from the ladder
``interval_calibration``   both terms and the ``duration`` coefficient of a logit working
                           model (``duration_logit``); 4,000 at n = 2,000, with the two
                           derived controls of each
``type_i_error``           ``duration`` under a sharp null of duration; 800 at n = 4,000
``power``                  ``duration`` under the law; 800 at n = 4,000
``targeting_necessity``    ``duration``, targeted and the untargeted projection of the same
                           fit's stitched initial fit, with a correct mechanism only; 1,000
``projection_necessity``   ``duration`` under the declared and uniform weights; 1,000
``crossfit_overfitting``   ``duration`` with fully grown trees on ``make_longitudinal``:
                           five folds (positive) and one fold (control), paired draws;
                           8,000 at n = 1,000
``in_sample_agreement``    a diagnostic: ``|beta_cf - beta_in| / SE_cf`` on the n = 8,000
                           ladder draws, whose mean the summary publishes as
                           ``mean_abs_difference_over_se``.  It states no verdict
``simultaneous_coverage``  the band over both coefficients of the primary subject and its
                           pointwise control; :data:`BAND_REPLICATES` at n = 2,500
=========================  ==================================================================

The shared constants (``EFFICIENCY_RATIO_BAND``, ``SHRUNKEN_SE_FACTOR``, both displacements)
are the ones ``longitudinal-msm`` declares.  The mechanism is the law's own
(``KnownDiscreteMechanism``) where a configuration declares it correct, because a saturated
cell fit can meet an empty cell in a training fold.  Streams are
``stream_seed(STUDY, "property_sample", family, label, replicate)``: the label is the
configuration, the size or the cell, ``paired`` for the overfitting pair, and the derived
controls and ``in_sample_agreement`` read their positive cell's fits.

A replication that raises is not redrawn.  :func:`failure_probe` measures each fit set on its
streams 0 to 1,999 before the declared run, and a fit set with any failure is dropped before the
run, with its gap stated as a page limit.  The shared harness refuses a cell that lost a
replication, so a raise in the declared run stops it.  The failing cell then drops to its
red-cell owner (the study module names the row), with its failure count published, and the run
repeats without it with no other change.

The declared probe on streams 0 to 1,999 found 5 failures in ``root_n_and_efficiency/n_500``
and none elsewhere.  Each failure is the cross-fit refusal "every unit following regimen
'always' through time 2 in outer training fold k has the same outcome", on streams 393, 602,
945, 1,323 and 1,813: at n = 500 a training fold can hold only events among the followers of
``always``.  The rule dropped that fit set before the run.  The ladder therefore has no control
rung, and each rate is fitted from two sizes.  The page states the gap as a limit.

Measured budget, from one single-process draw of each fit set before the declaration.  The
first row includes the process warm-up.

=================================================  =====  ========  ==========  =========
fit set                                            draws  fits per  seconds     CPU
                                                          draw      per draw    seconds
=================================================  =====  ========  ==========  =========
``double_robustness/both_correct``                 1,000  1         0.89        890
``double_robustness/outcome_correct``              1,000  1         0.20        200
``double_robustness/mechanism_correct``            1,000  1         0.12        120
``double_robustness/both_wrong``                   1,000  1         0.13        130
``root_n_and_efficiency/n_2000``                   700    1         0.12        84
``root_n_and_efficiency/n_8000`` (and agreement)   700    2         0.50        350
``interval_calibration/correctly_specified``       4,000  1         0.13        520
``interval_calibration/duration_logit``            4,000  1         0.21        840
``type_i_error/sharp_null``                        800    1         0.19        152
``power/alternative``                              800    1         0.24        192
``targeting_necessity/targeted``                   1,000  1         0.14        140
``projection_necessity/declared_weights``          1,000  2         0.25        250
``crossfit_overfitting/paired``                    8,000  2         0.36        2,880
``simultaneous_coverage`` band                     4,000  1         0.21        840
=================================================  =====  ========  ==========  =========

That is 29,000 draws, 38,700 fits and about 7,600 CPU seconds.  The primary adds 800 fits at
about 0.2 s and 3,200 R ``lmtp`` regimen fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as law
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_msm_properties as in_sample
from tests.studies.canonical_crossfit_longitudinal_msm import (
    BAND_LABEL,
    LEARNER_FOLDS,
    N_FOLDS,
    PRIMARY_N,
    RANDOM_STATE,
    SCENARIO,
    STUDY,
    draw_from_seed,
    fit_cleverly,
)
from tests.studies.canonical_longitudinal_msm import (
    COLUMNS as PRIMARY_COLUMNS,
)
from tests.studies.canonical_longitudinal_msm import (
    ESTIMANDS as PRIMARY_ESTIMANDS,
)
from tests.studies.canonical_longitudinal_msm import (
    G_BOUNDS,
    TERMS,
    declared_msm,
    initial_beta,
    project_means,
)
from tests.studies.canonical_longitudinal_msm import (
    REGIMENS as PRIMARY_REGIMENS,
)
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells
from tests.studies.ltmle_crossfit_properties import KnownDiscreteMechanism

DOUBLE_ROBUST_REPLICATES = 1_000
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 700
RATE_SIZES = (2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
TARGETING_REPLICATES = DOUBLE_ROBUST_REPLICATES
TARGETING_N = DOUBLE_ROBUST_N
PROJECTION_REPLICATES = DOUBLE_ROBUST_REPLICATES
PROJECTION_N = DOUBLE_ROBUST_N
OVERFIT_REPLICATES = 8_000
OVERFIT_N = 1_000
#: The replications of the band cell: the RM36 rule needs a control power of at least 0.99
#: at ``p0 + 0.005``, and ``tests/unit/test_crossfit_longitudinal_msm_design.py`` pins the
#: design the count was read from.
BAND_REPLICATES = 4_000

EFFICIENCY_RATIO_BAND = in_sample.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = in_sample.SHRUNKEN_SE_FACTOR
TARGETING_DISPLACEMENT = in_sample.TARGETING_DISPLACEMENT
PROJECTION_DISPLACEMENT = in_sample.PROJECTION_DISPLACEMENT
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

LABELS = in_sample.LABELS
REGIMENS = in_sample.REGIMENS
DURATION = in_sample.DURATION
PROJECTION_WEIGHT = in_sample.PROJECTION_WEIGHT
COLUMNS = in_sample.COLUMNS
NAMES = in_sample.NAMES
LOGIT = "duration_logit"
#: The estimand of an ``in_sample_agreement`` row: ``|beta_cf - beta_in| / SE_cf``.
AGREEMENT = "abs_difference_over_se"
#: The summary column that publishes the declared statistic, the mean of :data:`AGREEMENT`.
AGREEMENT_COLUMN = "mean_abs_difference_over_se"
CALIBRATION_LABELS = (*TERMS, LOGIT)

NULL_PROBS = in_sample.NULL_PROBS
#: The overfitting pair reads ``duration`` on the primary law.
OVERFIT_TARGET = "msm_regimen[duration]"


# ---------------------------------------------------------------------------- truths


def regimen_means(probs: np.ndarray) -> np.ndarray:
    """The law's regimen means, in :data:`LABELS` order."""
    return np.array([law.functional(probs, f"ey_regimen[{label}]") for label in LABELS])


def _logit_parts(beta: np.ndarray, means: np.ndarray) -> tuple[np.ndarray, ...]:
    design = np.column_stack([np.ones(len(LABELS)), [DURATION[label] for label in LABELS]])
    weights = np.array([PROJECTION_WEIGHT[label] for label in LABELS])
    mean = expit(design @ beta)
    slope = mean * (1.0 - mean)
    bend = slope * (1.0 - 2.0 * mean)
    score = design.T @ (weights * slope * (means - mean))
    jacobian = design.T @ ((weights * (slope**2 - (means - mean) * bend))[:, None] * design)
    return design, weights, slope, score, jacobian


def logit_coefficients(probs: np.ndarray) -> np.ndarray:
    """The weighted logistic projection of the regimen means, by a longhand Newton solve.

    Not :func:`cleverly.msm.solve_projection`: the truth of a cell must not come from the code
    the cell checks.
    """
    means = regimen_means(probs)
    beta = np.zeros(2)
    for _ in range(100):
        _, _, _, score, jacobian = _logit_parts(beta, means)
        step = np.linalg.solve(jacobian, score)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-15:
            break
    return beta


def logit_influence_curves() -> np.ndarray:
    """``M^{-1} sum_c h dm/deta phi EIF_c`` over the support, for the logit coefficients."""
    means = regimen_means(law.PROBS)
    design, weights, slope, _, jacobian = _logit_parts(logit_coefficients(law.PROBS), means)
    curves = np.column_stack([law.eif(f"ey_regimen[{label}]") for label in LABELS])
    contribution = curves @ (design * (weights * slope)[:, None])
    return np.asarray(contribution @ np.linalg.inv(jacobian).T)


TRUTH = in_sample.TRUTH
NULL_TRUTH = in_sample.NULL_TRUTH
LOGIT_TRUTH = float(logit_coefficients(law.PROBS)[TERMS.index("duration")])
EFFICIENCY_SD = {
    **in_sample.EFFICIENCY_SD,
    LOGIT: float(
        np.sqrt(np.sum(law.PROBS * logit_influence_curves()[:, TERMS.index("duration")] ** 2))
    ),
}


# ---------------------------------------------------------------------------- fits


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return (
        law.CellMeans() if q_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if q_correct else DummyRegressor(strategy="mean"),
        KnownDiscreteMechanism("treatment") if g_correct else DummyClassifier(strategy="prior"),
        KnownDiscreteMechanism("censoring") if g_correct else DummyClassifier(strategy="prior"),
    )


def working_model(*, uniform: bool = False, link: str = "identity") -> MSM:
    """The declared working model of the property law, with uniform weights or a link."""
    weights = dict.fromkeys(LABELS, 1.0) if uniform else PROJECTION_WEIGHT
    declared = declared_msm(DURATION, weights)
    if link == "identity":
        return declared
    return MSM(
        design=declared.design,
        terms=declared.terms,
        weights=declared.weights,
        weights_kind="known",
        design_kind="known",
        link=link,
    )


def fit(
    frame: pd.DataFrame,
    configuration: str = "both_correct",
    *,
    uniform: bool = False,
    link: str = "identity",
    n_folds: int = N_FOLDS,
) -> Any:
    outcome, pseudo, treatment, censoring = _learners(configuration)
    return LTMLE(
        REGIMENS,
        msm=working_model(uniform=uniform, link=link),
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=n_folds,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=RANDOM_STATE,
    ).fit(frame, **COLUMNS)


def overfit(frame: pd.DataFrame, *, cross_fit: bool) -> Any:
    """Fully grown trees on the primary law, at five folds or in sample."""
    return LTMLE(
        PRIMARY_REGIMENS,
        msm=declared_msm(),
        outcome_learner=DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
        pseudo_learner=DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=N_FOLDS if cross_fit else 1,
        learner_folds=5,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=RANDOM_STATE,
    ).fit(frame, **PRIMARY_COLUMNS)


def overfit_draw(seed: int, n: int = OVERFIT_N) -> tuple[pd.DataFrame, float]:
    """One ``make_longitudinal`` panel and the projection truth of ``duration``."""
    frame, truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
    means = {label: float(truth[f"ey_regimen[{label}]"]) for label in PRIMARY_REGIMENS}
    return frame, float(project_means(means)[TERMS.index("duration")])


# ---------------------------------------------------------------------------- the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        ``"discrete"``, ``"discrete_null"``, ``"discrete_logit"``, ``"discrete_agreement"``
        or ``"make_longitudinal"``.  ``"discrete_agreement"`` is the law of the difference
        ``beta_cf - beta_in`` on the ladder's draws, whose truth is zero.  It has its own name
        because its rows read the ladder's stream: the stream is shared by declaration, and
        the collision gate keys a stream by law and seed.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name == "discrete":
            return dict(TRUTH)
        if self.name == "discrete_agreement":
            return {AGREEMENT: 0.0}
        if self.name == "discrete_null":
            return {NAMES["duration"]: NULL_TRUTH}
        if self.name == "discrete_logit":
            return {NAMES["duration"]: LOGIT_TRUTH}
        _, truth = overfit_draw(0, n=10)
        return {OVERFIT_TARGET: truth}


#: Every fit set: ``(family, stream label, configuration, n, replicates, law)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        (
            "double_robustness",
            configuration,
            configuration,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            "discrete",
        )
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "discrete")
        for size in RATE_SIZES
    ),
    (
        "interval_calibration",
        "correctly_specified",
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "discrete",
    ),
    (
        "interval_calibration",
        LOGIT,
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "discrete_logit",
    ),
    ("type_i_error", "sharp_null", "both_correct", NULL_N, NULL_REPLICATES, "discrete_null"),
    ("power", "alternative", "both_correct", NULL_N, NULL_REPLICATES, "discrete"),
    (
        "targeting_necessity",
        "targeted",
        "mechanism_correct",
        TARGETING_N,
        TARGETING_REPLICATES,
        "discrete",
    ),
    (
        "projection_necessity",
        "declared_weights",
        "both_correct",
        PROJECTION_N,
        PROJECTION_REPLICATES,
        "discrete",
    ),
    (
        "crossfit_overfitting",
        "paired",
        "overfit",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "make_longitudinal",
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream.

    The ``root_n_rate`` rows are fitted from the ladder's rows rather than sampled, so they
    are published in the summary and declared in ``STUDY.property_cells`` but not here.
    """
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        law_name: str,
        n: int,
        replicates: int,
        label: str,
        estimand: str,
        stream_family: str | None = None,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(law_name),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=replicates,
                seed=_seed(stream_family or family, label, 0),
                role=role,
                estimand=estimand,
            )
        )

    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
        role = "control" if configuration == "both_wrong" else "positive"
        for term in TERMS:
            add(
                "double_robustness",
                f"{term}__{configuration}",
                role,
                "discrete",
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                configuration,
                NAMES[term],
            )
    for size in RATE_SIZES:
        role = "positive"
        for term in TERMS:
            add(
                "root_n_and_efficiency",
                f"{term}__n_{size}",
                role,
                "discrete",
                size,
                RATE_REPLICATES,
                f"n_{size}",
                NAMES[term],
            )
    for label in CALIBRATION_LABELS:
        stream = LOGIT if label == LOGIT else "correctly_specified"
        law_name = "discrete_logit" if label == LOGIT else "discrete"
        estimand = NAMES["duration"] if label == LOGIT else NAMES[label]
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                law_name,
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                stream,
                estimand,
            )
    add(
        "type_i_error",
        "duration__sharp_null",
        "positive",
        "discrete_null",
        NULL_N,
        NULL_REPLICATES,
        "sharp_null",
        NAMES["duration"],
    )
    add(
        "power",
        "duration__alternative",
        "positive",
        "discrete",
        NULL_N,
        NULL_REPLICATES,
        "alternative",
        NAMES["duration"],
    )
    add(
        "targeting_necessity",
        "duration__targeted",
        "positive",
        "discrete",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        NAMES["duration"],
    )
    add(
        "targeting_necessity",
        "duration__untargeted",
        "control",
        "discrete",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        NAMES["duration"],
    )
    add(
        "projection_necessity",
        "duration__declared_weights",
        "positive",
        "discrete",
        PROJECTION_N,
        PROJECTION_REPLICATES,
        "declared_weights",
        NAMES["duration"],
    )
    add(
        "projection_necessity",
        "duration__uniform_weights",
        "control",
        "discrete",
        PROJECTION_N,
        PROJECTION_REPLICATES,
        "declared_weights",
        NAMES["duration"],
    )
    add(
        "crossfit_overfitting",
        "cross_fitted_msm",
        "positive",
        "make_longitudinal",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "paired",
        OVERFIT_TARGET,
    )
    add(
        "crossfit_overfitting",
        "in_sample_control",
        "control",
        "make_longitudinal",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "paired",
        OVERFIT_TARGET,
    )
    for term in TERMS:
        add(
            "in_sample_agreement",
            f"{term}__in_sample_agreement",
            DIAGNOSTIC_ROLE,
            "discrete_agreement",
            max(RATE_SIZES),
            RATE_REPLICATES,
            f"n_{max(RATE_SIZES)}",
            AGREEMENT,
            stream_family="root_n_and_efficiency",
        )
    cells.extend(
        joint_property_cells(
            BAND_LABEL, n=PRIMARY_N, replicates=BAND_REPLICATES, seed=_seed(JOINT, BAND_LABEL, 0)
        )
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _role(family: str, label: str, n: int) -> str:
    if label == "both_wrong":
        return "control"
    return "positive"


def _fit_set_rows(payload: tuple[str, str, str, int, int, int, str]) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes."""
    family, label, configuration, replicate, n, requested, law_name = payload
    seed = _seed(family, label, replicate)
    alpha = STUDY.margins.alpha
    common = {"replicate": replicate, "n": n, "requested": requested}
    if family == "crossfit_overfitting":
        frame, truth = overfit_draw(seed, n)
        rows = []
        for cell, cross_fit in (("cross_fitted_msm", True), ("in_sample_control", False)):
            result = overfit(frame, cross_fit=cross_fit)
            rows.append(
                replicate_row(
                    property_name=family,
                    cell=cell,
                    role="positive" if cross_fit else "control",
                    truth=truth,
                    estimate=result[OVERFIT_TARGET],
                    alpha=alpha,
                    **common,
                )
            )
        return rows
    probs = NULL_PROBS if law_name == "discrete_null" else law.PROBS
    frame = in_sample.sample(probs, n, seed)
    if law_name == "discrete_logit":
        result = fit(frame, configuration, link="logit")
        return [
            replicate_row(
                property_name=family,
                cell=f"{LOGIT}__correctly_specified",
                role="positive",
                truth=LOGIT_TRUTH,
                estimate=result[NAMES["duration"]],
                alpha=alpha,
                **common,
            )
        ]
    result = fit(frame, configuration)
    single = family in {"type_i_error", "power", "targeting_necessity", "projection_necessity"}
    terms = ("duration",) if single else TERMS
    rows: list[dict[str, Any]] = []
    for term in terms:
        name = NAMES[term]
        truth = NULL_TRUTH if law_name == "discrete_null" else float(TRUTH[name])
        cell = f"{term}__{label}"
        rows.append(
            replicate_row(
                property_name=family,
                cell=cell,
                role=_role(family, label, n),
                truth=truth,
                estimate=result[name],
                alpha=alpha,
                **common,
            )
        )
        if family == "targeting_necessity":
            # The untargeted arm is the projection of the same fit's stitched initial fit:
            # the cross-fitted plug-in the pooled update starts from.
            plug_in = initial_beta(result, LABELS)
            rows.append(
                control_row(
                    property_name=family,
                    cell=f"{term}__untargeted",
                    truth=truth,
                    estimate=float(plug_in[TERMS.index(term)]),
                    standard_error=float(result[name].std_error),
                    critical=CRITICAL,
                    **common,
                )
            )
        if family == "projection_necessity":
            wrong = fit(frame, configuration, uniform=True)
            rows.append(
                control_row(
                    property_name=family,
                    cell=f"{term}__uniform_weights",
                    truth=truth,
                    estimate=float(wrong[name].psi),
                    standard_error=float(wrong[name].std_error),
                    critical=CRITICAL,
                    **common,
                )
            )
        if family == "root_n_and_efficiency" and n == max(RATE_SIZES):
            if term == TERMS[0]:
                inside = fit(frame, configuration, n_folds=1)
            rows.append(
                control_row(
                    property_name="in_sample_agreement",
                    cell=f"{term}__in_sample_agreement",
                    truth=0.0,
                    # The declared statistic, |beta_cf - beta_in| / SE_cf, per draw.  The
                    # unit standard error makes the row's mean that statistic, and its
                    # coverage the share of draws whose in-sample coefficient lies inside
                    # the cross-fitted Wald interval.
                    estimate=float(
                        abs(result[name].psi - inside[name].psi) / result[name].std_error
                    ),
                    standard_error=1.0,
                    critical=CRITICAL,
                    role=DIAGNOSTIC_ROLE,
                    **common,
                )
            )
    return rows


def _band_rows(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, requested = payload
    frame, truth = draw_from_seed(SCENARIO, PRIMARY_N, _seed(JOINT, BAND_LABEL, replicate))
    result = fit_cleverly(frame, simultaneous=True)
    return joint_coverage_rows(
        result,
        truth,
        PRIMARY_ESTIMANDS,
        label=BAND_LABEL,
        replicate=replicate,
        n=PRIMARY_N,
        requested=requested,
        pointwise_critical=CRITICAL,
    )


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, label, configuration, n, replicates, law_name in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(
            ((_fit_set_rows, (family, label, configuration, r, n, requested, law_name)),)
            for r in range(requested)
        )
    band = BAND_REPLICATES if budget is None else budget
    out.extend(((_band_rows, (r, band)),) for r in range(band))
    return out


def _run(job: tuple[Any, tuple[Any, ...]]) -> list[dict[str, Any]]:
    function, payload = job
    return list(function(payload))


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = map_parallel(_run, _payloads(budget), n_jobs=n_jobs)
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
    """Failed fits per fit set over streams ``start`` to ``start + draws - 1``.

    It records no estimate.  A fit set with any failure is dropped before the declared run.
    """
    payloads = [
        ((family, label, configuration, r, n, draws, law_name),)
        for family, label, configuration, n, _, law_name in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{family}/{label}": 0 for family, label, *_ in FIT_SETS}
    for (payload,), failed in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, str, int, int, int, str]) -> bool:
    try:
        rows = _fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


# ---------------------------------------------------------------------------- verdicts


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the study's own declared rules, and the reported diagnostic."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "projection_displacement",
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
            AGREEMENT_COLUMN,
        ),
        rate_labels=TERMS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("duration",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="projection_necessity",
        labels=("duration",),
        arms=("declared_weights", "uniform_weights"),
        column="projection_displacement",
        threshold=PROJECTION_DISPLACEMENT,
    )
    crossfit_overfitting_verdicts(summary, rows, STUDY, positive_cell="cross_fitted_msm")
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    # A reported family: its rows publish the declared statistic and state no verdict.  The
    # bias and SE-ratio columns describe an estimator against a truth, and these rows have
    # neither, so they are blanked rather than published as though they had failed.
    agreement = summary["property"] == "in_sample_agreement"
    summary.loc[agreement, AGREEMENT_COLUMN] = summary.loc[agreement, "mean_estimate"]
    for column in (
        "bias_margin",
        "standardized_bias",
        "root_n_bias",
        "se_ratio",
        "rejection_rate",
        "rejection_ci_lower",
        "rejection_ci_upper",
    ):
        if column in summary:
            summary.loc[agreement, column] = np.nan
    for column in ("bias_equivalent", "bias_discriminated"):
        if column in summary:
            summary[column] = summary[column].astype(object)
            summary.loc[agreement, column] = None
    summary.loc[agreement, "passed"] = True
    summary.loc[agreement, "property_passed"] = True
    return finish(summary, rates)
