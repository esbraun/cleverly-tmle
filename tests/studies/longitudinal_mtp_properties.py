"""Property families of ``longitudinal-mtp``.

The estimand of every single-parameter family is the continuous contrast
``ate_regimen[up vs natural]`` (label ``up``) unless the table says otherwise.

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``double_robustness``           ``up`` in four nuisance configurations; 1,000 at n = 2,000
``root_n_and_efficiency``       ``up`` at n = 500, 2,000 and 8,000; 600 each.  n = 500 is
                                the ladder's control rung
``root_n_rate``                 the two rate rows of ``up``, from the ladder
``interval_calibration``        ``up``; ``up`` on the classifier ratio route with the
                                correctly specified ``common.OracleLogOdds``
                                (``classifier_route``); the categorical law's ``minus one``
                                contrast (``categorical_mtp``); a vector node of two binary
                                components (``vector_node``); a randomized node-2 policy
                                (``randomized_mtp``); the cumulative risk at t = 2 of a
                                survival plan (``survival_mtp_h2``); the ``dose`` coefficient
                                of a working model over three continuous cells (``msm_mtp``).
                                2,000 at n = 2,000 each, with the two derived controls of each
``type_i_error``                ``up`` under a law where no dose moves the outcome; 600 at
                                n = 4,000
``power``                       ``up`` under the law; 600 at n = 4,000
``targeting_necessity``         ``up``, targeted and the longhand untargeted plug-in, with a
                                correct mechanism only; 1,000 at n = 2,000
``inverse_necessity``           ``up then history`` with the declared inverse, and the same
                                fit with the inverse dropped from Equation (3); the outcome is
                                misspecified so the ratio carries the estimate; 1,000 at
                                n = 2,000
``crossfit_overfitting``        ``up`` with fully grown trees and a noise column: five folds
                                (positive) and one fold (control), paired draws; 10,000 at
                                n = 1,000
``simultaneous_coverage``       the band over the seven continuous primary estimands and its
                                pointwise control; 2,000 at n = 2,000
==============================  ===========================================================

The plan's ``ratio_route``, ``categorical_mtp``, ``vector_node``, ``randomized_mtp``,
``survival_calibration`` and ``msm_mtp__interval_calibration`` cells are labels of
``interval_calibration``, the route ``stochastic-categorical-ltmle`` and
``cross-fitted-longitudinal-msm`` took for their survival and working-model cells, so they
answer to that family's coverage band and SE-ratio band and carry its derived controls.  The
efficiency bounds the SE-ratio and rate rows read (:data:`EFFICIENCY_SD`) are computed from the
closed-form nuisances, never from a fit
(:func:`~tests.studies.longitudinal_mtp_common.continuous_contrast_sd`).

A replication that raises is not redrawn.  :func:`failure_probe` counts failed fits per fit
set and records no estimate.  A fit set with any failure is dropped before the declared run,
with its gap stated as a page limit.  The shared harness refuses a cell that lost a
replication, so a raise in the declared run stops it.  The failing cell then drops to its
red-cell owner with its failure count published, and the run repeats without it with no other
change.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from cleverly.interventions import ModifiedPolicy, Piece, Randomizer
from cleverly.interventions import policy as policy_module
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.msm import MSM
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as binary_law
from tests import discrete_law_longitudinal_mtp as vector_law
from tests import discrete_law_longitudinal_multivalue as multivalue
from tests import discrete_law_survival as survival
from tests import longitudinal_mtp as vector_plans
from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import longitudinal_mtp_common as common
from tests.studies.canonical_longitudinal_mtp import (
    BAND_LABEL,
    CALIBRATION_LABELS,
    CATEGORICAL,
    CONTINUOUS,
    DENSITY_BINS,
    ESTIMANDS,
    G_BOUNDS,
    PRIMARY_N,
    STUDY,
    TRUTH,
    density_bins,
    edges_of,
    fit,
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
    crossfit_overfitting_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells

DOUBLE_ROBUST_REPLICATES = 1_000
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 600
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 600
NULL_N = 4_000
TARGETING_REPLICATES = 1_000
TARGETING_N = 2_000
INVERSE_REPLICATES = 1_000
INVERSE_N = 2_000
OVERFIT_REPLICATES = 10_000
OVERFIT_N = 1_000
#: The band cell's replications; ``tests/unit/test_longitudinal_mtp_design.py`` pins that they
#: meet the RM36 control-power rule at the measured ``p0``.
BAND_REPLICATES = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.10
INVERSE_DISPLACEMENT = 0.10
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

UP = "ate_regimen[up vs natural]"
HISTORY = "ate_regimen[up then history vs natural]"
CATEGORICAL_NAME = "ate_regimen[minus one vs natural]"
VECTOR_NAME = "ate_regimen[vector vs natural]"
RANDOM_NAME = "ate_regimen[random at 2 vs natural]"
SURVIVAL_NAME = "risk_regimen[mtp @ t=2]"
MSM_NAME = "msm_regimen[dose]"
MSM_CELLS = ("natural", "up at 2", "up")
DOSE = {"natural": 0.0, "up at 2": 0.5, "up": 1.0}
PROJECTION_WEIGHT = {"natural": 1.0, "up at 2": 2.0, "up": 1.5}

#: The efficiency bound of each calibration label, as a standard deviation, computed once by
#: ``continuous_contrast_sd`` (400,000 draws of the closed-form influence function) and the
#: complex step on the finite laws.  ``tests/unit/test_longitudinal_mtp_design.py`` recomputes
#: the finite ones exactly and the continuous ones to Monte Carlo accuracy.
EFFICIENCY_SD: dict[str, float] = {
    "up": 0.39050650977914014,
    "classifier_route": 0.39050650977914014,
    "categorical_mtp": 0.41811204797453305,
    "vector_node": 0.6244173513005088,
    "randomized_mtp": 0.1317617396210603,
    "survival_mtp_h2": 0.6773556088479237,
    "msm_mtp": 0.396481773437949,
}


# ---------------------------------------------------------------------------- truths


def _projection_map() -> np.ndarray:
    design = np.array([[1.0, DOSE[label]] for label in MSM_CELLS])
    weights = np.array([PROJECTION_WEIGHT[label] for label in MSM_CELLS])
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T * weights)


PROJECTION = _projection_map()
MSM_TRUTH = float(PROJECTION[1] @ np.array([common.continuous_truth(label) for label in MSM_CELLS]))
RANDOM_TRUTH = 0.5 * (common.continuous_truth("up at 2") - common.continuous_truth("natural"))
VECTOR_TRUTH = vector_law.truth("ey_regimen[vector]") - vector_law.truth("ey_regimen[natural]")
SURVIVAL_TRUTH = float(vector_law.functional_mtp_survival(survival.PROBS, 2))
NULL_TRUTH = 0.0


def null_sample(n: int, seed: int) -> pd.DataFrame:
    """The continuous law with no dose effect: ``L2`` and ``Y`` ignore both doses."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    a1 = common.truncated_draw(rng, common.mean1(w))
    l2 = rng.binomial(1, common.p_l2(w, np.full(n, 2.8))).astype(float)
    a2 = common.truncated_draw(rng, common.mean2(w, a1, l2))
    y = rng.binomial(1, common.outcome_mean(w, np.full(n, 2.8), l2, np.full(n, 2.6))).astype(float)
    return pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y})


# ---------------------------------------------------------------------------- fits


def _random_at_two() -> ModifiedPolicy:
    up = (
        Piece(
            -np.inf,
            common.CAP - common.STEP,
            lambda a, h: a + common.STEP,
            lambda b, h: b - common.STEP,
            lambda b, h: 1.0,
        ),
        Piece(common.CAP - common.STEP, np.inf),
    )
    return ModifiedPolicy(
        "random",
        pieces={"up": up, "stay": (Piece(-np.inf, np.inf),)},
        randomizer=Randomizer(("up", "stay"), (0.5, 0.5)),
        policy_kind="known",
    )


def _up_plans() -> dict[str, Any]:
    return {"natural": common.NATURAL, "up": common.UP}


def continuous_fit(frame: pd.DataFrame, configuration: str, **settings: Any) -> Any:
    return fit(frame, CONTINUOUS, configuration=configuration, plans=_up_plans(), **settings)


def classifier_fit(frame: pd.DataFrame) -> Any:
    return fit(
        frame,
        CONTINUOUS,
        configuration="both_correct",
        plans=_up_plans(),
        ratio="classifier",
        treatment_learner=common.OracleLogOdds("up"),
    )


def random_fit(frame: pd.DataFrame) -> Any:
    plans = {
        "natural": common.NATURAL,
        "random at 2": DynamicRegimen("random at 2", (common.NATURAL, _random_at_two())),
    }
    return fit(frame, CONTINUOUS, configuration="both_correct", plans=plans)


def history_fit(frame: pd.DataFrame, *, drop_inverse: bool = False) -> Any:
    plans = {
        "natural": common.NATURAL,
        "up then history": common.continuous_regimens()["up then history"],
    }
    with _inverse_dropped(drop_inverse):
        return fit(frame, CONTINUOUS, configuration="mechanism_correct", plans=plans)


@contextmanager
def _inverse_dropped(active: bool) -> Iterator[None]:
    """Equation (3) read at the dose itself instead of at the declared inverse, if ``active``.

    The control of ``inverse_necessity``: the ratio of each declared moving piece becomes
    ``g(b) |b'| / g(b)``, which is one on a shift.  Only the ratio is replaced, so the policy's
    own checks still read the declared inverse.  Restored on exit, so a worker's next fit
    reads the declared inverse again.
    """
    if not active:
        yield
        return
    original = policy_module._PieceBranch.ratio

    def no_inverse(self: Any, values: Any, frame: Any, density: Any) -> Any:
        b = np.asarray(values, dtype=float).reshape(-1)
        denominator = density(b)
        numerator = np.zeros(b.size)
        for piece in self.pieces:
            if piece.map is None:
                continue
            slope = policy_module._per_row(
                policy_module._call(piece.derivative, b, frame()), b.size
            )
            inside = policy_module._member(b, *self.bounds(piece, frame, b.size), piece.closed)
            numerator = numerator + np.where(inside, density(b) * np.abs(slope), 0.0)
        safe = np.where(denominator > 0.0, denominator, 1.0)
        covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
        for piece in self.pieces:
            if piece.map is None:
                covariate = covariate + policy_module._member(
                    b, *self.bounds(piece, frame, b.size), piece.closed
                )
        return covariate

    policy_module._PieceBranch.ratio = no_inverse  # type: ignore[method-assign]
    try:
        yield
    finally:
        policy_module._PieceBranch.ratio = original  # type: ignore[method-assign]


def categorical_fit(frame: pd.DataFrame) -> Any:
    return fit(frame, CATEGORICAL)


def vector_fit(frame: pd.DataFrame) -> Any:
    return LTMLE(
        vector_plans.regimens(("natural", "vector")),
        reference="natural",
        outcome_learner=binary_law.CellMeans(),
        pseudo_learner=binary_law.CellMeans(),
        treatment_learner=multivalue.CellProbabilities(),
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=[["A1a", "A1b"], ["A2a", "A2b"]],
        baseline=["W"],
        time_varying=[[], ["L2"]],
    )


def vector_sample(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    full = vector_law.frame(vector=True)
    cells = rng.choice(len(vector_law.SUPPORT), size=n, p=vector_law.PROBS)
    return full.iloc[vector_law.first_row_of()[cells]].reset_index(drop=True)


def survival_sample(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(survival.SUPPORT), size=n, p=survival.PROBS)
    return survival.frame().iloc[survival.first_row_of()[cells]].reset_index(drop=True)


def survival_fit(frame: pd.DataFrame) -> Any:
    return LTMLE(
        {"mtp": vector_plans.SURVIVAL_PLAN},
        outcome_learner=survival.CellMeans(),
        pseudo_learner=survival.CellMeans(),
        treatment_learner=survival.CellMeans(),
        censoring_learner=survival.CellMeans(),
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        censoring=["C1", "C2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
    )


def working_model() -> MSM:
    return MSM(
        design=lambda label, horizon, w: np.column_stack(
            [np.ones(len(w)), np.full(len(w), DOSE[label])]
        ),
        terms=("intercept", "dose"),
        design_kind="known",
        weights=lambda label, horizon, w: np.full(len(w), PROJECTION_WEIGHT[label]),
        weights_kind="known",
    )


def msm_fit(frame: pd.DataFrame) -> Any:
    plans = {
        "natural": common.NATURAL,
        "up at 2": DynamicRegimen("up at 2", (common.NATURAL, common.UP)),
        "up": common.UP,
    }
    outcome, pseudo, treatment = common.continuous_learners("both_correct", edges_of(frame))
    return LTMLE(
        plans,
        msm=working_model(),
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        n_folds=1,
        g_bounds=G_BOUNDS,
        density_bins=DENSITY_BINS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        continuous_treatment=["A1", "A2"],
    )


def untargeted(frame: pd.DataFrame, configuration: str) -> float:
    r"""The in-sample sequential plug-in of ``up`` minus ``natural``, with no fluctuation.

    Longhand, for the reason ``stochastic-categorical-ltmle`` gives: in a one-fold recursion
    node 1 regresses node 2's targeted value, so the fit's own first-node initial already
    carries node 2's update.  Node 2 is regressed on ``[W, L2, A1, A2]`` and predicted at the
    policy dose, node 1 regresses that on ``[W, A1]`` and is predicted at the policy dose.
    """
    outcome, pseudo, _ = common.continuous_learners(
        configuration, edges_of(frame, density_bins(configuration))
    )
    w = frame["W"].to_numpy(dtype=float)
    a1 = frame["A1"].to_numpy(dtype=float)
    l2 = frame["L2"].to_numpy(dtype=float)
    a2 = frame["A2"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)

    def mean(d1: Any, d2: Any) -> float:
        later = clone(outcome).fit(np.column_stack([w, l2, a1, a2]), y)
        carried = np.clip(_predict(later, np.column_stack([w, l2, a1, d2(a2)]), True), 0.0, 1.0)
        earlier = clone(pseudo).fit(np.column_stack([w, a1]), carried)
        return float(
            np.mean(np.clip(_predict(earlier, np.column_stack([w, d1(a1)]), False), 0.0, 1.0))
        )

    identity = common.CONTINUOUS_MAPS["natural"][0]
    up = common.CONTINUOUS_MAPS["up"][0]
    return mean(up, up) - mean(identity, identity)


def _predict(model: Any, design: np.ndarray, classification: bool) -> np.ndarray:
    if classification and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(design)[:, -1], dtype=float)
    return np.asarray(model.predict(design), dtype=float)


# ---------------------------------------------------------------------------- the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        ``"continuous"``, ``"continuous_null"``, ``"continuous_noise"``,
        ``"continuous_classifier"``, ``"continuous_random"``, ``"continuous_history"``,
        ``"continuous_msm"``, ``"categorical"``, ``"vector"`` or ``"survival"``.  The names
        differ where two families read one law on different streams, so the collision gate
        can key a stream by law and seed.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name == "continuous_null":
            return {UP: NULL_TRUTH}
        if self.name == "continuous_random":
            return {RANDOM_NAME: RANDOM_TRUTH}
        if self.name == "continuous_history":
            return {HISTORY: TRUTH[CONTINUOUS][HISTORY]}
        if self.name == "continuous_msm":
            return {MSM_NAME: MSM_TRUTH}
        if self.name == "categorical":
            return {CATEGORICAL_NAME: TRUTH[CATEGORICAL][CATEGORICAL_NAME]}
        if self.name == "vector":
            return {VECTOR_NAME: VECTOR_TRUTH}
        if self.name == "survival":
            return {SURVIVAL_NAME: SURVIVAL_TRUTH}
        return dict(TRUTH[CONTINUOUS])


#: Every fit set: ``(family, stream label, configuration, n, replicates, law)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        (
            "double_robustness",
            configuration,
            configuration,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            "continuous",
        )
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "continuous")
        for size in RATE_SIZES
    ),
    (
        "interval_calibration",
        "correctly_specified",
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "continuous",
    ),
    *(
        ("interval_calibration", label, "both_correct", CALIBRATION_N, CALIBRATION_REPLICATES, law)
        for label, law in (
            ("classifier_route", "continuous_classifier"),
            ("categorical_mtp", "categorical"),
            ("vector_node", "vector"),
            ("randomized_mtp", "continuous_random"),
            ("survival_mtp_h2", "survival"),
            ("msm_mtp", "continuous_msm"),
        )
    ),
    ("type_i_error", "sharp_null", "both_correct", NULL_N, NULL_REPLICATES, "continuous_null"),
    ("power", "alternative", "both_correct", NULL_N, NULL_REPLICATES, "continuous"),
    (
        "targeting_necessity",
        "targeted",
        "mechanism_correct",
        TARGETING_N,
        TARGETING_REPLICATES,
        "continuous",
    ),
    (
        "inverse_necessity",
        "paired",
        "mechanism_correct",
        INVERSE_N,
        INVERSE_REPLICATES,
        "continuous_history",
    ),
    (
        "crossfit_overfitting",
        "paired",
        "overfit",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "continuous_noise",
    ),
)

#: Each calibration label's estimand.
CALIBRATION_ESTIMANDS = {
    "up": UP,
    "classifier_route": UP,
    "categorical_mtp": CATEGORICAL_NAME,
    "vector_node": VECTOR_NAME,
    "randomized_mtp": RANDOM_NAME,
    "survival_mtp_h2": SURVIVAL_NAME,
    "msm_mtp": MSM_NAME,
}


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream.

    The ``root_n_rate`` rows are fitted from the ladder's rows rather than sampled, so they
    are published in the summary and declared in ``PROPERTY_CELLS`` but not here.
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
                seed=_seed(family, label, 0),
                role=role,
                estimand=estimand,
            )
        )

    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
        role = "control" if configuration == "both_wrong" else "positive"
        add(
            "double_robustness",
            f"up__{configuration}",
            role,
            "continuous",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
            UP,
        )
    for size in RATE_SIZES:
        role = "control" if size == min(RATE_SIZES) else "positive"
        add(
            "root_n_and_efficiency",
            f"up__n_{size}",
            role,
            "continuous",
            size,
            RATE_REPLICATES,
            f"n_{size}",
            UP,
        )
    for label, law_name, stream in (
        ("up", "continuous", "correctly_specified"),
        ("classifier_route", "continuous_classifier", "classifier_route"),
        ("categorical_mtp", "categorical", "categorical_mtp"),
        ("vector_node", "vector", "vector_node"),
        ("randomized_mtp", "continuous_random", "randomized_mtp"),
        ("survival_mtp_h2", "survival", "survival_mtp_h2"),
        ("msm_mtp", "continuous_msm", "msm_mtp"),
    ):
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
                CALIBRATION_ESTIMANDS[label],
            )
    add(
        "type_i_error",
        "up__sharp_null",
        "positive",
        "continuous_null",
        NULL_N,
        NULL_REPLICATES,
        "sharp_null",
        UP,
    )
    add(
        "power",
        "up__alternative",
        "positive",
        "continuous",
        NULL_N,
        NULL_REPLICATES,
        "alternative",
        UP,
    )
    add(
        "targeting_necessity",
        "up__targeted",
        "positive",
        "continuous",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        UP,
    )
    add(
        "targeting_necessity",
        "up__untargeted",
        "control",
        "continuous",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        UP,
    )
    add(
        "inverse_necessity",
        "history__declared_inverse",
        "positive",
        "continuous_history",
        INVERSE_N,
        INVERSE_REPLICATES,
        "paired",
        HISTORY,
    )
    add(
        "inverse_necessity",
        "history__inverse_dropped_control",
        "control",
        "continuous_history",
        INVERSE_N,
        INVERSE_REPLICATES,
        "paired",
        HISTORY,
    )
    add(
        "crossfit_overfitting",
        "cross_fitted_mtp_ltmle",
        "positive",
        "continuous_noise",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "paired",
        UP,
    )
    add(
        "crossfit_overfitting",
        "in_sample_control",
        "control",
        "continuous_noise",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "paired",
        UP,
    )
    cells.extend(
        joint_property_cells(
            BAND_LABEL, n=PRIMARY_N, replicates=BAND_REPLICATES, seed=_seed(JOINT, BAND_LABEL, 0)
        )
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _role(family: str, label: str, n: int) -> str:
    if label == "both_wrong" or (family == "root_n_and_efficiency" and n == min(RATE_SIZES)):
        return "control"
    return "positive"


def _fit_set_rows(payload: tuple[str, str, str, int, int, int, str]) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes."""
    family, label, configuration, replicate, n, requested, law_name = payload
    seed = _seed(family, label, replicate)
    alpha = STUDY.margins.alpha
    common_row = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell: str, role: str, truth: float, estimate: Any) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=truth,
            estimate=estimate,
            alpha=alpha,
            **common_row,
        )

    calibration = f"{label}__correctly_specified"
    if law_name == "continuous_classifier":
        result = classifier_fit(common.sample_continuous(n, seed))
        return [row(calibration, "positive", float(TRUTH[CONTINUOUS][UP]), result[UP])]
    if law_name == "categorical":
        result = categorical_fit(common.CATEGORICAL_LAW.sample(n, seed))
        truth = float(TRUTH[CATEGORICAL][CATEGORICAL_NAME])
        return [row(calibration, "positive", truth, result[CATEGORICAL_NAME])]
    if law_name == "vector":
        result = vector_fit(vector_sample(n, seed))
        return [row(calibration, "positive", VECTOR_TRUTH, result[VECTOR_NAME])]
    if law_name == "continuous_random":
        result = random_fit(common.sample_continuous(n, seed))
        return [row(calibration, "positive", RANDOM_TRUTH, result[RANDOM_NAME])]
    if law_name == "survival":
        result = survival_fit(survival_sample(n, seed))
        return [row(calibration, "positive", SURVIVAL_TRUTH, result[SURVIVAL_NAME])]
    if law_name == "continuous_msm":
        result = msm_fit(common.sample_continuous(n, seed))
        return [row(calibration, "positive", MSM_TRUTH, result[MSM_NAME])]
    if family == "crossfit_overfitting":
        frame = common.sample_continuous(n, seed, noise=True)
        truth = float(TRUTH[CONTINUOUS][UP])
        return [
            row(
                "cross_fitted_mtp_ltmle",
                "positive",
                truth,
                continuous_fit(frame, "overfit_crossfit", n_folds=5)[UP],
            ),
            row(
                "in_sample_control",
                "control",
                truth,
                continuous_fit(frame, "overfit_control", n_folds=1)[UP],
            ),
        ]
    if family == "inverse_necessity":
        frame = common.sample_continuous(n, seed)
        truth = float(TRUTH[CONTINUOUS][HISTORY])
        declared = history_fit(frame)
        dropped = history_fit(frame, drop_inverse=True)
        return [
            row("history__declared_inverse", "positive", truth, declared[HISTORY]),
            control_row(
                property_name=family,
                cell="history__inverse_dropped_control",
                truth=truth,
                estimate=float(dropped[HISTORY].psi),
                standard_error=float(dropped[HISTORY].std_error),
                critical=CRITICAL,
                **common_row,
            ),
        ]
    frame = (
        null_sample(n, seed) if law_name == "continuous_null" else common.sample_continuous(n, seed)
    )
    result = continuous_fit(frame, configuration)
    truth = NULL_TRUTH if law_name == "continuous_null" else float(TRUTH[CONTINUOUS][UP])
    stream = "correctly_specified" if family == "interval_calibration" else label
    rows = [row(f"up__{stream}", _role(family, label, n), truth, result[UP])]
    if family == "targeting_necessity":
        rows.append(
            control_row(
                property_name=family,
                cell="up__untargeted",
                truth=truth,
                estimate=float(untargeted(frame, configuration)),
                standard_error=float(result[UP].std_error),
                critical=CRITICAL,
                **common_row,
            )
        )
    return rows


def _band_rows(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, requested = payload
    frame = common.sample_continuous(PRIMARY_N, _seed(JOINT, BAND_LABEL, replicate))
    result = fit(frame, CONTINUOUS, simultaneous=True)
    return joint_coverage_rows(
        result,
        TRUTH[CONTINUOUS],
        ESTIMANDS[CONTINUOUS],
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


#: The measured peak of one ``mechanism_correct`` fit at n = 2,000 and 320 bins, with its
#: margin: 1.57 GiB of traced numpy peak and about 2.5 GB resident.  These fit sets run in a
#: pool capped by :func:`tests.parallel.memory_capped_workers`; at 16 workers they would need
#: 25 to 40 GB.
MECHANISM_ONLY_FIT_BYTES = 3 * 1024**3


def _heavy(job: tuple[Any, tuple[Any, ...]]) -> bool:
    """Whether a payload fits at :data:`MECHANISM_ONLY_BINS`."""
    function, payload = job
    return function is _fit_set_rows and density_bins(payload[2]) > DENSITY_BINS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check.

    The fit sets at :data:`MECHANISM_ONLY_BINS` run in their own pool, capped by the free
    memory; the worker count does not change any row.
    """
    jobs = _payloads(budget)
    heavy = [job for job in jobs if _heavy(job[0])]
    light = [job for job in jobs if not _heavy(job[0])]
    outcomes = map_parallel(_run, light, n_jobs=n_jobs)
    # Read the free memory only once the light pool is done.  Read at the start, it saw the
    # memory the reference phase's container still held, and capped the pool at one worker.
    capped = memory_capped_workers(n_jobs, MECHANISM_ONLY_FIT_BYTES)
    outcomes += map_parallel(_run, heavy, n_jobs=capped)
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
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "inverse_displacement",
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
        ),
        rate_labels=("up",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("up",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="inverse_necessity",
        labels=("history",),
        arms=("declared_inverse", "inverse_dropped_control"),
        column="inverse_displacement",
        threshold=INVERSE_DISPLACEMENT,
    )
    crossfit_overfitting_verdicts(summary, rows, STUDY, positive_cell="cross_fitted_mtp_ltmle")
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)


__all__ = [
    "FIT_SETS",
    "declared_cells",
    "failure_probe",
    "generate_property_rows",
    "summarize_properties",
]
