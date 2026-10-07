"""Property families of ``stochastic-categorical-ltmle``.

Every end-of-study family reads the two-node three-level law of
:mod:`tests.discrete_law_longitudinal_multivalue` with the declared policies of
:mod:`tests.discrete_law_longitudinal_policy`.  The policy contrast ``ate_regimen[mix vs low]``
is the estimand of every single-parameter family and is labelled ``mix``.

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``double_robustness``           ``mix`` in four nuisance configurations; 1,200 at n = 2,000
``root_n_and_efficiency``       ``mix`` at n = 500, 2,000 and 8,000; 800 each.  n = 500 is
                                the ladder's control rung
``root_n_rate``                 the two rate rows of ``mix``, from the ladder
``interval_calibration``        ``mix``; the cumulative risk at t = 2 of a survival policy
                                (``policy_risk_h2``); the ``dose`` coefficient of a working
                                model over ``low``, ``mix`` and ``taper``
                                (``msm_policy``).  4,000 at n = 2,000 each, with the two
                                derived controls of each
``type_i_error``                ``mix`` under a law where no arm moves the outcome; 800 at
                                n = 4,000
``power``                       ``mix`` under the law; 800 at n = 4,000
``targeting_necessity``         ``mix``, targeted and the same fit's untargeted plug-in, with
                                a correct mechanism only; 1,200 at n = 2,000
``policy_necessity``            ``mix`` under the declared policies and under uniform ones,
                                scored against the declared truth; 1,200 at n = 2,000
``randomizer_projection``       ``ey_regimen[mix]``: the integrated estimator and Theorem 3 as
                                stated on the same sample with a recorded randomizer; 1,200
                                at n = 2,000
``crossfit_overfitting``        ``mix`` with fully grown trees and a noise column: five folds
                                (positive) and one fold (control), paired draws; 40,000 at
                                n = 1,000
``simultaneous_coverage``       the band over the five primary estimands and its pointwise
                                control; 2,000 at n = 2,000
==============================  ===========================================================

The replication counts and sizes are the ones ``canonical-categorical-ltmle`` and
``canonical-categorical-ltmle-crossfit`` declare, copied before any run.  The survival and
working-model calibration cells are labels of ``interval_calibration``, so they answer to
that family's coverage band and SE-ratio band and carry its derived controls.  The plan
named them as families of their own; filing them as labels is the route
``cross-fitted-longitudinal-msm`` took for its logit calibration cell.

``randomizer_projection`` has its own rule (:func:`randomizer_projection_verdicts`): each arm
must establish its bias inside the equivalence margin, and the one-sided 99% bootstrap upper
bound of ``sd(integrated) / sd(recorded)`` must lie below one.  The recorded arm's outcome
learner groups each history by the arms the rule assigns there, which is the saturated fit of
the sufficient history (the randomizer moves no outcome).

A replication that raises is not redrawn.  :func:`failure_probe` counts failed fits per fit
set and records no estimate.  A fit set with any failure is dropped before the declared run,
with its gap stated as a page limit.  The shared harness refuses a cell that lost a
replication, so a raise in the declared run stops it.  The failing cell then drops to its
red-cell owner with its failure count published, and the run repeats without it with no other
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone

from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as binary_law
from tests import discrete_law_longitudinal_multivalue as law
from tests import discrete_law_longitudinal_policy as policy
from tests import discrete_law_survival as survival
from tests import longitudinal_policies as policies
from tests.parallel import STUDY_JOBS
from tests.studies import categorical_longitudinal_common as categorical
from tests.studies.canonical_stochastic_categorical_ltmle import (
    BAND_LABEL,
    CALIBRATION_LABELS,
    ESTIMANDS,
    G_BOUNDS,
    LABELS,
    PRIMARY_N,
    STUDY,
    TRUTH,
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

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 4_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
TARGETING_REPLICATES = 1_200
TARGETING_N = 2_000
POLICY_REPLICATES = 1_200
POLICY_N = 2_000
RANDOMIZER_REPLICATES = 1_200
RANDOMIZER_N = 2_000
OVERFIT_REPLICATES = 40_000
OVERFIT_N = 1_000
#: The band cell's replications, the plan's 2,000.  The RM36 rule needs a control power of at
#: least 0.99 at ``p0 + 0.005``; ten fits of the primary subject give ``p0 = 0.8560``, where
#: 2,000 replications give power 1.0.  ``tests/unit/test_stochastic_categorical_ltmle_design.py``
#: pins that design.
BAND_REPLICATES = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.10
POLICY_DISPLACEMENT = 0.10
#: The one-sided level of the projection-gain bound, ``sd(integrated) / sd(recorded)``.
PROJECTION_LEVEL = 0.99
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

MIX = "ate_regimen[mix vs low]"
MIX_MEAN = "ey_regimen[mix]"
RECORDED_MEAN = "ey_regimen[mix (recorded)]"
SURVIVAL_NAME = "risk_regimen[draw @ t=2]"
DOSE = {"low": 0.0, "mix": 1.0, "taper": 2.5}
PROJECTION_WEIGHT = {"low": 1.0, "mix": 2.0, "taper": 3.5}
MSM_NAME = "msm_regimen[dose]"


# ---------------------------------------------------------------------------- truths


def _null_outcome() -> np.ndarray:
    """The categorical study's sharp null: no arm at either node moves the outcome mean."""
    outcome = np.empty_like(law.Q)
    for w in range(2):
        for a1 in range(3):
            for l2 in range(2):
                outcome[w, a1, l2, :] = 0.5 + 0.2 * (l2 - law.P_L2[w, a1])
    return outcome


NULL_PROBS = law.probabilities(_null_outcome())
NULL_TRUTH = float(policy.functional(NULL_PROBS, MIX))


def _projection_map() -> np.ndarray:
    design = np.array([[1.0, DOSE[label]] for label in LABELS])
    weights = np.array([PROJECTION_WEIGHT[label] for label in LABELS])
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T * weights)


PROJECTION = _projection_map()
MSM_TRUTH = float(PROJECTION[1] @ np.array([TRUTH[f"ey_regimen[{label}]"] for label in LABELS]))


def _survival_function(probs: Any) -> Any:
    return policy.functional_policy_survival(probs, 2)


SURVIVAL_TRUTH = float(_survival_function(survival.PROBS))


def _sd(probs: np.ndarray, curve: np.ndarray) -> float:
    return float(np.sqrt(np.sum(probs * curve**2)))


EFFICIENCY_SD = {
    "mix": _sd(policy.PROBS, policy.eif_policy(policy.PROBS, MIX)),
    "policy_risk_h2": _sd(survival.PROBS, policy.eif(_survival_function, survival.PROBS)),
    "msm_policy": _sd(
        policy.PROBS,
        PROJECTION[1]
        @ np.array([policy.eif_policy(policy.PROBS, f"ey_regimen[{label}]") for label in LABELS]),
    ),
}


# ---------------------------------------------------------------------------- fits


class RecordedMechanism(BaseEstimator):
    """The law's mechanism on a history that also carries the recorded randomizer.

    The randomizer is independent of every node, so the mechanism ignores it.  This drops the
    ``E1`` and ``E2`` columns from the design the estimator hands over and reads the rest with
    :class:`~tests.studies.categorical_longitudinal_common.KnownCategoricalMechanism`.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RecordedMechanism:
        self.base_ = categorical.KnownCategoricalMechanism().fit(X, y)
        self.classes_ = self.base_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] == 2:
            kept = matrix[:, [0]]
        elif matrix.shape[1] == 6:
            kept = matrix[:, [0, 2, 4, 5]]
        else:
            raise ValueError(f"unexpected recorded mechanism design {matrix.shape}")
        return np.asarray(self.base_.predict_proba(kept), dtype=float)


class RecordedCellMeans(BaseEstimator):
    """Cell means over the arms the recorded rule assigns, the sufficient history.

    The outcome regression of the recorded rule reads ``[W, E1]`` at the first node and
    ``[W, E1, L2, E2]`` at the second.  On the rows that follow the rule the arms are
    ``d_1(W, E1)`` and ``d_2(W, E1, L2, E2)``, and the outcome depends on the history through
    them alone.  So this groups each row by ``(W, a1)`` or ``(W, a1, L2, a2)`` and fits a mean
    per group, which is the saturated fit of the sufficient history and needs no randomizer
    cell to be populated.
    """

    def _keys(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w = matrix[:, 0].astype(int)
        e1 = matrix[:, 1].astype(int)
        a1 = np.array([policy._draw(policy.POLICY1[wi], ei) for wi, ei in zip(w, e1, strict=True)])
        if matrix.shape[1] == 2:
            return np.column_stack([w, a1])
        l2 = matrix[:, 2].astype(int)
        e2 = matrix[:, 3].astype(int)
        a2 = np.array(
            [
                policy._draw(policy.POLICY2[wi, ai, li], ei)
                for wi, ai, li, ei in zip(w, a1, l2, e2, strict=True)
            ]
        )
        return np.column_stack([w, a1, l2, a2])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RecordedCellMeans:
        keys = self._keys(X)
        target = np.asarray(y, dtype=float)
        self.groups_, inverse = np.unique(keys, axis=0, return_inverse=True)
        self.means_ = np.array([target[inverse == g].mean() for g in range(len(self.groups_))])
        self.default_ = float(target.mean())
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        keys = self._keys(X)
        out = np.full(len(keys), self.default_)
        for group, mean in zip(self.groups_, self.means_, strict=True):
            out[np.all(keys == group, axis=1)] = mean
        return out

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = self.predict(X)
        return np.column_stack([1.0 - probability, probability])


def recorded_fit(frame: pd.DataFrame, seed: int) -> Any:
    """Theorem 3 as stated: the rule that reads a fresh recorded randomizer per node."""
    rng = np.random.default_rng([seed, 1])
    augmented = frame.assign(
        E1=rng.integers(0, policy.RANDOMIZER_LEVELS, len(frame)).astype(float),
        E2=rng.integers(0, policy.RANDOMIZER_LEVELS, len(frame)).astype(float),
    )
    return LTMLE(
        {"mix (recorded)": policies.recorded_regimen()},
        outcome_learner=RecordedCellMeans(),
        pseudo_learner=RecordedCellMeans(),
        treatment_learner=RecordedMechanism(),
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        augmented,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[["E1"], ["L2", "E2"]],
    )


def survival_sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of the binary survival law, drawn cell by cell."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(survival.SUPPORT), size=n, p=survival.PROBS)
    full = survival.frame()
    first = survival.first_row_of()
    return full.iloc[first[cells]].reset_index(drop=True)


def survival_fit(frame: pd.DataFrame) -> Any:
    return LTMLE(
        {"draw": policies.survival_regimen()},
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
    return LTMLE(
        policies.regimens(LABELS),
        msm=working_model(),
        outcome_learner=binary_law.CellMeans(),
        pseudo_learner=binary_law.CellMeans(),
        treatment_learner=categorical.KnownCategoricalMechanism(),
        n_folds=1,
        learner_folds=2,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, outcome="Y", treatment=["A1", "A2"], baseline=["W"], time_varying=[[], ["L2"]])


def _blocks(labels: np.ndarray) -> np.ndarray:
    """Drop-first indicators over the estimator's sorted level order."""
    return np.column_stack([(labels == level).astype(float) for level in categorical.LEVELS[1:]])


def _policy_rows(table: np.ndarray, *index: np.ndarray) -> np.ndarray:
    """Each row's policy over the sorted levels, from a table in the law's raw arm order."""
    order = [law.ARM_LABELS.index(level) for level in categorical.LEVELS]
    return np.asarray(table[index], dtype=float)[:, order]


def untargeted(frame: pd.DataFrame, label: str, configuration: str) -> float:
    r"""The in-sample sequential plug-in of one plan, with every fluctuation removed.

    Written longhand, as :func:`tests.studies.categorical_longitudinal_common.untargeted` is,
    rather than read off the fit: in a one-fold recursion node 1 regresses node 2's *targeted*
    value, so the fit's own first-node initial prediction already carries node 2's update.
    Here node 2 is regressed on its rows, predicted at each level, and carried as the policy
    mean :math:`\sum_j q_2(j)\, Q_2(j, H_2)`; node 1 regresses that and the plug-in averages
    :math:`\sum_j q_1(j)\, Q_1(j, H_1)`.  ``low`` is the label at both nodes, fitted on its
    followers with the covariate history alone.
    """
    outcome, pseudo, _ = categorical._learners(configuration)
    w = frame["W"].to_numpy(dtype=float)
    l2 = frame["L2"].to_numpy(dtype=float)
    a1 = frame["A1"].to_numpy()
    a2 = frame["A2"].to_numpy()
    y = frame["Y"].to_numpy(dtype=float)
    if label == "low":
        first = a1 == "low"
        second = first & (a2 == "low")
        history = np.column_stack([w, l2])
        later = clone(outcome).fit(history[second], y[second])
        carried = np.clip(categorical._predict(later, history, classification=True), 0.0, 1.0)
        earlier = clone(pseudo).fit(w[first, None], carried[first])
        return float(
            np.mean(
                np.clip(categorical._predict(earlier, w[:, None], classification=False), 0.0, 1.0)
            )
        )
    if label != "mix":
        raise ValueError(f"no longhand plug-in for plan {label!r}")
    raw_a1 = np.array([law.ARM_LABELS.index(item) for item in a1], dtype=int)
    q1 = _policy_rows(policy.POLICY1, w.astype(int))
    q2 = _policy_rows(policy.POLICY2, w.astype(int), raw_a1, l2.astype(int))

    def second_design(arm: np.ndarray) -> np.ndarray:
        return np.column_stack([w, l2, _blocks(a1), _blocks(arm)])

    def first_design(arm: np.ndarray) -> np.ndarray:
        return np.column_stack([w, _blocks(arm)])

    levels = np.asarray(categorical.LEVELS, dtype=object)
    later = clone(outcome).fit(second_design(a2), y)
    by_arm = np.column_stack(
        [
            np.clip(
                categorical._predict(
                    later, second_design(np.full(len(y), level)), classification=True
                ),
                0.0,
                1.0,
            )
            for level in levels
        ]
    )
    carried = np.sum(q2 * by_arm, axis=1)
    earlier = clone(pseudo).fit(first_design(a1), carried)
    by_arm = np.column_stack(
        [
            np.clip(
                categorical._predict(
                    earlier, first_design(np.full(len(y), level)), classification=False
                ),
                0.0,
                1.0,
            )
            for level in levels
        ]
    )
    return float(np.mean(np.sum(q1 * by_arm, axis=1)))


# ---------------------------------------------------------------------------- the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        ``"policy"``, ``"policy_null"``, ``"policy_noise"``, ``"policy_randomizer"``,
        ``"survival_policy"`` or ``"policy_msm"``.  The names differ where two families read
        the same law on different streams, so the collision gate can key a stream by law and
        seed.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name == "policy_null":
            return {MIX: NULL_TRUTH}
        if self.name == "survival_policy":
            return {SURVIVAL_NAME: SURVIVAL_TRUTH}
        if self.name == "policy_msm":
            return {MSM_NAME: MSM_TRUTH}
        if self.name == "policy_randomizer":
            return {MIX_MEAN: TRUTH[MIX_MEAN]}
        return dict(TRUTH)


#: Every fit set: ``(family, stream label, configuration, n, replicates, law)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        (
            "double_robustness",
            configuration,
            configuration,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            "policy",
        )
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "policy")
        for size in RATE_SIZES
    ),
    (
        "interval_calibration",
        "correctly_specified",
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "policy",
    ),
    (
        "interval_calibration",
        "policy_risk_h2",
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "survival_policy",
    ),
    (
        "interval_calibration",
        "msm_policy",
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "policy_msm",
    ),
    ("type_i_error", "sharp_null", "both_correct", NULL_N, NULL_REPLICATES, "policy_null"),
    ("power", "alternative", "both_correct", NULL_N, NULL_REPLICATES, "policy"),
    (
        "targeting_necessity",
        "targeted",
        "mechanism_correct",
        TARGETING_N,
        TARGETING_REPLICATES,
        "policy",
    ),
    ("policy_necessity", "paired", "both_correct", POLICY_N, POLICY_REPLICATES, "policy"),
    (
        "randomizer_projection",
        "paired",
        "both_correct",
        RANDOMIZER_N,
        RANDOMIZER_REPLICATES,
        "policy_randomizer",
    ),
    ("crossfit_overfitting", "paired", "overfit", OVERFIT_N, OVERFIT_REPLICATES, "policy_noise"),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream.

    The ``root_n_rate`` rows are fitted from the ladder's rows rather than sampled, so they
    are published in the summary and declared in :data:`PROPERTY_CELLS` but not here.
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
            f"mix__{configuration}",
            role,
            "policy",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
            MIX,
        )
    for size in RATE_SIZES:
        role = "control" if size == min(RATE_SIZES) else "positive"
        add(
            "root_n_and_efficiency",
            f"mix__n_{size}",
            role,
            "policy",
            size,
            RATE_REPLICATES,
            f"n_{size}",
            MIX,
        )
    for label, law_name, stream, estimand in (
        ("mix", "policy", "correctly_specified", MIX),
        ("policy_risk_h2", "survival_policy", "policy_risk_h2", SURVIVAL_NAME),
        ("msm_policy", "policy_msm", "msm_policy", MSM_NAME),
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
                estimand,
            )
    add(
        "type_i_error",
        "mix__sharp_null",
        "positive",
        "policy_null",
        NULL_N,
        NULL_REPLICATES,
        "sharp_null",
        MIX,
    )
    add(
        "power",
        "mix__alternative",
        "positive",
        "policy",
        NULL_N,
        NULL_REPLICATES,
        "alternative",
        MIX,
    )
    add(
        "targeting_necessity",
        "mix__targeted",
        "positive",
        "policy",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        MIX,
    )
    add(
        "targeting_necessity",
        "mix__untargeted",
        "control",
        "policy",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        MIX,
    )
    add(
        "policy_necessity",
        "mix__declared_policy",
        "positive",
        "policy",
        POLICY_N,
        POLICY_REPLICATES,
        "paired",
        MIX,
    )
    add(
        "policy_necessity",
        "mix__uniform_control",
        "control",
        "policy",
        POLICY_N,
        POLICY_REPLICATES,
        "paired",
        MIX,
    )
    add(
        "randomizer_projection",
        "mix__integrated",
        "positive",
        "policy_randomizer",
        RANDOMIZER_N,
        RANDOMIZER_REPLICATES,
        "paired",
        MIX_MEAN,
    )
    add(
        "randomizer_projection",
        "mix__recorded_randomizer",
        "positive",
        "policy_randomizer",
        RANDOMIZER_N,
        RANDOMIZER_REPLICATES,
        "paired",
        MIX_MEAN,
    )
    add(
        "crossfit_overfitting",
        "cross_fitted_policy_ltmle",
        "positive",
        "policy_noise",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "paired",
        MIX,
    )
    add(
        "crossfit_overfitting",
        "in_sample_control",
        "control",
        "policy_noise",
        OVERFIT_N,
        OVERFIT_REPLICATES,
        "paired",
        MIX,
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
    common = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell: str, role: str, truth: float, estimate: Any) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=truth,
            estimate=estimate,
            alpha=alpha,
            **common,
        )

    if law_name == "survival_policy":
        result = survival_fit(survival_sample(n, seed))
        return [
            row(f"{label}__correctly_specified", "positive", SURVIVAL_TRUTH, result[SURVIVAL_NAME])
        ]
    if law_name == "policy_msm":
        result = msm_fit(law.sample(law.PROBS, n, seed))
        return [row(f"{label}__correctly_specified", "positive", MSM_TRUTH, result[MSM_NAME])]
    if family == "crossfit_overfitting":
        frame = law.sample(law.PROBS, n, seed, noise=True)
        truth = float(TRUTH[MIX])
        return [
            row(
                "cross_fitted_policy_ltmle",
                "positive",
                truth,
                fit(frame, configuration="overfit_crossfit", n_folds=categorical.N_FOLDS)[MIX],
            ),
            row(
                "in_sample_control",
                "control",
                truth,
                fit(frame, configuration="overfit_control", n_folds=1)[MIX],
            ),
        ]
    probs = NULL_PROBS if law_name == "policy_null" else law.PROBS
    frame = law.sample(probs, n, seed)
    if family == "randomizer_projection":
        truth = float(TRUTH[MIX_MEAN])
        integrated = fit(frame, configuration=configuration)
        recorded = recorded_fit(frame, seed)
        return [
            row("mix__integrated", "positive", truth, integrated[MIX_MEAN]),
            row("mix__recorded_randomizer", "positive", truth, recorded[RECORDED_MEAN]),
        ]
    if family == "policy_necessity":
        truth = float(TRUTH[MIX])
        declared = fit(frame, configuration=configuration)
        uniform = fit(frame, configuration=configuration, uniform=True)
        return [
            row("mix__declared_policy", "positive", truth, declared[MIX]),
            control_row(
                property_name=family,
                cell="mix__uniform_control",
                truth=truth,
                estimate=float(uniform[MIX].psi),
                standard_error=float(uniform[MIX].std_error),
                critical=CRITICAL,
                **common,
            ),
        ]
    result = fit(frame, configuration=configuration)
    truth = NULL_TRUTH if law_name == "policy_null" else float(TRUTH[MIX])
    rows = [row(f"mix__{label}", _role(family, label, n), truth, result[MIX])]
    if family == "targeting_necessity":
        plug_in = untargeted(frame, "mix", configuration) - untargeted(frame, "low", configuration)
        rows.append(
            control_row(
                property_name=family,
                cell="mix__untargeted",
                truth=truth,
                estimate=float(plug_in),
                standard_error=float(result[MIX].std_error),
                critical=CRITICAL,
                **common,
            )
        )
    return rows


def _band_rows(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, requested = payload
    frame = law.sample(law.PROBS, PRIMARY_N, _seed(JOINT, BAND_LABEL, replicate))
    result = fit(frame, simultaneous=True)
    return joint_coverage_rows(
        result,
        TRUTH,
        ESTIMANDS,
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


def projection_sd_ratio(
    rows: pd.DataFrame, *, replicates: int, level: float, seed: int
) -> tuple[float, float]:
    """``sd(integrated) / sd(recorded)`` and its one-sided bootstrap upper bound.

    The two arms are paired on ``replicate``, so a bootstrap draw resamples replicates and
    keeps each pair together.
    """
    family = rows.loc[rows["property"] == "randomizer_projection"]
    integrated = family.loc[family["cell"] == "mix__integrated"].set_index("replicate")["estimate"]
    recorded = family.loc[family["cell"] == "mix__recorded_randomizer"].set_index("replicate")[
        "estimate"
    ]
    paired = pd.concat([integrated, recorded], axis=1, keys=["integrated", "recorded"]).dropna()
    values = paired.to_numpy(dtype=float)
    point = float(np.std(values[:, 0], ddof=1) / np.std(values[:, 1], ddof=1))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    sampled = values[draws]
    ratios = np.std(sampled[:, :, 0], axis=1, ddof=1) / np.std(sampled[:, :, 1], axis=1, ddof=1)
    return point, float(np.quantile(ratios, level))


def randomizer_projection_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Both arms inside the bias margin, and the integrated spread established below the recorded.

    Each arm's own verdict is its bias endpoint.  The joint claim adds the one-sided
    :data:`PROJECTION_LEVEL` bootstrap upper bound of ``sd(integrated) / sd(recorded)``, which
    must lie below one: the integrated curve is the projection of the recorded one, so its
    variance is no larger, and the cell establishes that the gain is real on this law.
    """
    mask = summary["property"] == "randomizer_projection"
    if not mask.any():
        return
    summary.loc[mask, "passed"] = summary.loc[mask, "bias_equivalent"]
    point, upper = projection_sd_ratio(
        rows,
        replicates=STUDY.margins.bootstrap_replicates,
        level=PROJECTION_LEVEL,
        seed=stream_seed(STUDY, "randomizer_projection", "sd_ratio"),
    )
    summary.loc[mask, "projection_sd_ratio"] = point
    summary.loc[mask, "projection_sd_ratio_ci_upper"] = upper
    summary.loc[mask, "property_passed"] = bool(summary.loc[mask, "passed"].all() and upper < 1.0)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "policy_displacement",
            "projection_sd_ratio",
            "projection_sd_ratio_ci_upper",
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
        ),
        rate_labels=("mix",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("mix",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="policy_necessity",
        labels=("mix",),
        arms=("declared_policy", "uniform_control"),
        column="policy_displacement",
        threshold=POLICY_DISPLACEMENT,
    )
    randomizer_projection_verdicts(summary, rows)
    crossfit_overfitting_verdicts(summary, rows, STUDY, positive_cell="cross_fitted_policy_ltmle")
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)
