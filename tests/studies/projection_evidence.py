r"""Declaration and driver of the ``projection-evidence`` study (P1 PR 2 plan rev 3, section 5A).

The study measures what P1 PR 2 makes reachable and what PR 1 left experimental on the
projection TMLE of a :class:`cleverly.ConditionalContrast`:

- the projection of a controlled direct effect at both levels of a binary intermediate, on the
  difference and the log-ratio scale (``cde_4k``, ``cde_crr_4k``), with the plug-in-stratum,
  selection, intermediate-density and total-effect controls that fail where a term that
  vanishes at the truth is dropped (``cde_dr_4k``, ``cde_total_4k``);
- one pooled outcome regression with an ``A x spline(W)`` interaction (``pooled_4k``,
  ``pooled_crr_4k``);
- the simultaneous band of ``effect_curve(grid=)`` on the ``T49``, ``T50``, ``T49.clustered``
  and ``T49.repeats`` keys, each with its pointwise joint control.

Everything a run reads is fixed here before it starts: the laws (:func:`sample`, with the
``cde_obs`` law of :mod:`tests.studies.projection_evidence_law`), the exact truths by quadrature,
the fit groups (:data:`GROUPS`) and the cells their fits fill (:func:`group_cells`), the learners,
the primary phase and the study record (:data:`STUDY`).

The calibration ``R`` comes from the 3,200-draw pilot (:mod:`tests.studies.projection_evidence_pilot`,
with its ``pilot_2`` extension where an arm needs it): :data:`BUDGETED_R` is the plan's budget, and
:data:`DECLARED_R` is the pilot's ``R`` where larger, never lower.  If no ``R`` up to
50,000 meets X32a's rule after ``pilot_2``, the declaration reads each calibration row at its
pooled pilot ratio and takes the smallest multiple of 400 at which the product of the rows' pass
probabilities is at least :data:`POINT_READING_TARGET` (0.95, plan 5.0 rule 4, chosen on
2026-11-20), under the two-sided precondition of :mod:`tests.studies.point_reading`.  The study is
registered (:func:`tests.studies.evidence.registry.registered`) from its declared run at
``f60abdf7``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Linear, Modifiers
from cleverly.effect_modification import EffectModification
from cleverly.estimators import TMLE
from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as cp
from tests.studies import projection_evidence_law as cde
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SLUG = "projection-evidence"

#: The record seeds.  The plan named 20261501, which ``calibration-slope-warning`` already holds
#: (with 20261502 as its resampling seed), so the study takes the next free pair.
SEED = 20261511
RESAMPLING_SEED = 20261512

#: The ``cde_obs`` modifier and basis.
CDE_MODIFIERS = Modifiers(["W1"])
CDE_BASIS = Linear(["W1"])
CDE_TERMS = cde.TERMS
LEVELS = cde.LEVELS

#: The budgeted calibration ``R`` (plan 5.4): the pilot's ``R`` replaces it when larger.
BUDGETED_R = 12_000
#: The family target of the point-reading fallback (plan 5.0 rule 4; chosen 2026-11-20).
POINT_READING_TARGET = 0.95
#: Replications of each band group (plan 5.4): a joint coverage of 0.945 passes with 0.990.
BAND_R = 2_400
#: A band group whose pilot joint coverage reads below this is rebuilt at :data:`BAND_REBUILT_R`.
BAND_REBUILD_BELOW = 0.94
BAND_REBUILT_R = 4_000
#: Replications of the wrong-``Qbar`` group and of the total-effect control (plan 5.2).
DR_R = 1_000
TOTAL_R = 500
#: Pilot draws per gated cell, and the ``pilot_2`` extension (plan 5.0 rule 3).
PILOT_DRAWS = 3_200
PILOT_2_DRAWS = 3_200
#: An arm whose plug-in pass reads below this after the pilot runs ``pilot_2``.
PILOT_2_BELOW = 0.90
#: The shrunken control's factor.
SHRUNKEN_FACTOR = 0.70
#: The multiplier draws of every band, the plan's band key (plan 3.3).
BAND_DRAWS = 1_000
#: Outer folds of every ``cde_obs`` fit (plan 5.2).
CDE_FOLDS = 5

CRITICAL = float(stats.norm.ppf(0.975))


# ------------------------------------------------------------------------------ the laws


#: Every law of the study: ``cde_obs`` and two ``conditional-projection`` laws.
LAWS: tuple[str, ...] = ("cde_obs", "observational", "clustered")


def sample(law: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` rows of ``law``."""
    if law == "cde_obs":
        return cde.sample(n, rng)
    if law in ("observational", "clustered"):
        return cp.sample(law, n, rng)
    raise KeyError(law)


# ------------------------------------------------------------------------- the fit groups


CALIBRATION = "interval_calibration"
BIAS = "bias"
SIMULTANEOUS = "simultaneous_coverage"
BAND_KIND = "simultaneous_band"
JOINT_CONTROL_KIND = "pointwise_joint_control"


@dataclass(frozen=True)
class Group:
    """One declared fit group: a law, a size, its replications, and what its fits fill.

    Parameters
    ----------
    label : str
        The group key, which the seed stream and the cell names read.
    law : str
        A key of :data:`LAWS`.
    n : int
        Rows per replication.
    replicates : int
        Replications in the declared run.
    kind : {"cde", "cde_dr", "cde_total", "pooled", "band"}
        What the group fits: the CDE projection at both levels from one engine fit, the same
        at the declared wrong ``Qbar`` with the true mechanisms, the total-effect projection on
        ``cde_obs`` without the intermediate, the pooled projection, or the projection with a
        band on the ``proj_obs`` grid.
    scale : str
        The contrast's scale.
    arms : tuple of str
        The calibration arms the group's rows join; a CDE arm ends in its level (``_z0``).
    bias_rows : bool
        Whether the group's TMLE coefficients at both levels are positive ``bias`` rows.
    controls : Mapping of str to tuple of str
        Each control the group's fits compute, and the terms it publishes rows for.
    band : {"", "gated", "diagnostic"}
        Whether the group publishes a band: gated with its pointwise joint control, or a free
        reading under the diagnostic role.
    repeats : int
        Cross-fitting repeats.
    seconds : float
        Single-thread seconds per replication, measured by the 20-replicate smoke on the
        declaration's container (:data:`SMOKE_TIMINGS`).
    """

    label: str
    law: str
    n: int
    replicates: int
    kind: str
    scale: str = "difference"
    arms: tuple[str, ...] = ()
    bias_rows: bool = False
    controls: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    band: str = ""
    repeats: int = 1
    seconds: float = 0.5

    @property
    def stem(self) -> str:
        """The reported name stem of the group's coefficients."""
        return "cate_projection" if self.scale == "difference" else "log_crr_projection"

    @property
    def terms(self) -> tuple[str, ...]:
        """The basis terms the group's fit reports."""
        return CDE_TERMS if self.law == "cde_obs" else cp.TERMS

    @property
    def levels(self) -> tuple[int | None, ...]:
        """The intermediate levels the group reports: both on ``cde_obs``, else none."""
        return LEVELS if self.law == "cde_obs" else (None,)

    def name(self, term: str, level: int | None) -> str:
        """The reported name of ``term`` at ``level``: ``z0:cate_projection[W1]`` on a CDE fit."""
        base = f"{self.stem}[{term}]"
        return base if level is None else f"z{level}:{base}"

    def names(self, level: int | None = None) -> tuple[str, ...]:
        """Every reported name at ``level``."""
        return tuple(self.name(term, level) for term in self.terms)


def arm_level(arm: str) -> int | None:
    """The level a CDE arm reads, from its ``_z0``/``_z1`` suffix; ``None`` otherwise."""
    for level in LEVELS:
        if arm.endswith(f"_z{level}"):
            return level
    return None


#: The declared calibration ``R``: X32a's rule on the 3,200-draw pilot (committed at 8edc4541;
#: no arm needed ``pilot_2``) gives 12,000, the budget.
DECLARED_R = 12_000
R = DECLARED_R
GROUPS: tuple[Group, ...] = (
    Group(
        "cde_4k",
        "cde_obs",
        4_000,
        R,
        "cde",
        arms=("__cde_z0", "__cde_z1"),
        controls={
            "plugin_stratum": ("(intercept)",),
            "shrunken_se": ("(intercept)",),
        },
        band="diagnostic",
        seconds=0.41,
    ),
    Group(
        "cde_crr_4k",
        "cde_obs",
        4_000,
        R,
        "cde",
        scale="log_ratio",
        arms=("__cde_crr_z0", "__cde_crr_z1"),
        seconds=0.43,
    ),
    Group(
        "cde_dr_4k",
        "cde_obs",
        4_000,
        DR_R,
        "cde_dr",
        bias_rows=True,
        controls={
            "no_selection": ("(intercept)",),
            "no_intermediate_weight": ("(intercept)",),
            "plugin_wrong": ("W1",),
        },
        seconds=0.29,
    ),
    Group(
        "cde_total_4k",
        "cde_obs",
        4_000,
        TOTAL_R,
        "cde_total",
        controls={"total_effect": ("(intercept)",)},
        seconds=0.23,
    ),
    Group(
        "pooled_4k",
        "observational",
        4_000,
        R,
        "pooled",
        arms=("__pooled",),
        band="diagnostic",
        seconds=0.37,
    ),
    Group(
        "pooled_crr_4k",
        "observational",
        4_000,
        R,
        "pooled",
        scale="log_ratio",
        arms=("__pooled_crr",),
        seconds=0.28,
    ),
    Group("band_obs_4k", "observational", 4_000, BAND_R, "band", band="gated", seconds=0.26),
    Group(
        "band_crr_4k",
        "observational",
        4_000,
        BAND_R,
        "band",
        scale="log_ratio",
        band="gated",
        seconds=0.29,
    ),
    Group("band_clustered_4k", "clustered", 4_000, BAND_R, "band", band="gated", seconds=0.18),
    Group(
        "band_rep3_4k",
        "observational",
        4_000,
        BAND_R,
        "band",
        band="gated",
        repeats=3,
        seconds=0.76,
    ),
)

GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}

#: What each control computes from its group's fits, and the rule it must fail (plan 5.3).
CONTROL_CLAIMS: Mapping[str, str] = {
    "plugin_stratum": "the targeted contrast averaged over the Z = z rows, M_z^-1 P_n[1{Z = z} "
    "phi (Q*_t - Q*_r)]: its bias must lie wholly outside +-0.25 of its own SD (-0.71 and "
    "+0.65 SD on the intercepts)",
    "shrunken_se": "the reported SE times 0.70: its SE-ratio interval must fall below the band",
    "no_selection": "the TMLE with 1{Z = z} dropped from the observed covariate, at the declared "
    "wrong Qbar: its bias must lie wholly outside the margin (-3.70 and +2.99 SD)",
    "no_intermediate_weight": "the TMLE with q_z omitted from the covariate, at the declared "
    "wrong Qbar: its bias must lie wholly outside the margin (+3.15 and -3.06 SD)",
    "plugin_wrong": "the plug-in at the declared wrong Qbar: its bias must lie wholly outside "
    "the margin (-1.06 SD on both slopes), the witness that the wrong Qbar is wrong",
    "total_effect": "the total-effect projection, fitted without the intermediate, read against "
    "the CDE truth: its bias must lie wholly outside the margin (-4.34 and +4.36 SD)",
    "pointwise_joint": "the pointwise 95% intervals over the band grid read jointly: the upper "
    "end of their joint coverage must lie below 0.95",
}

#: The calibration controls, by the shared kind their cell names end in.
CALIBRATION_CONTROL_KINDS: Mapping[str, str] = {"shrunken_se": "shrunken_se_control"}


def term_label(term: str) -> str:
    """A cell-name fragment of ``term``."""
    return "intercept" if term == "(intercept)" else term


def level_label(level: int | None) -> str:
    """A cell-name fragment of ``level``: ``_z0``, or nothing."""
    return "" if level is None else f"_z{level}"


def calibration_cell(group: Group, arm: str, term: str) -> str:
    """The ``interval_calibration`` cell of ``term`` in ``arm`` (``label__arm``)."""
    return f"{group.label}{level_label(arm_level(arm))}_{term_label(term)}{arm}"


def bias_cell(group: Group, term: str, level: int | None) -> str:
    """The positive ``bias`` cell of ``term`` at ``level`` on ``group``."""
    return f"{group.label}{level_label(level)}_{term_label(term)}"


def control_cell(control: str, group: Group, term: str, level: int | None) -> str:
    """The cell of ``control``'s row for ``term`` at ``level``.

    A calibration control is ``<group>_<level>_<term>__<kind>``; a bias control is
    ``<control>_<size><level>_<term>``.
    """
    if control in CALIBRATION_CONTROL_KINDS:
        return (
            f"{group.label}{level_label(level)}_{term_label(term)}"
            f"__{CALIBRATION_CONTROL_KINDS[control]}"
        )
    size = group.label.rsplit("_", 1)[-1]
    return f"{control}_{size}{level_label(level)}_{term_label(term)}"


def control_levels(control: str, group: Group) -> tuple[int | None, ...]:
    """The levels a control publishes rows for: the shrunken control reads ``z = 0`` only."""
    return (0,) if control == "shrunken_se" else group.levels


def band_labels(group: Group) -> tuple[str, ...]:
    """The joint labels of ``group``'s bands: one per level on a CDE fit."""
    if not group.band:
        return ()
    return tuple(f"{group.label}{level_label(level)}" for level in group.levels)


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells ``group``'s fits fill, by family."""
    cells: dict[str, list[str]] = {CALIBRATION: [], BIAS: [], SIMULTANEOUS: []}
    for arm in group.arms:
        cells[CALIBRATION] += [calibration_cell(group, arm, term) for term in group.terms]
    if group.bias_rows:
        cells[BIAS] += [
            bias_cell(group, term, level) for level in group.levels for term in group.terms
        ]
    for control, terms in group.controls.items():
        family = CALIBRATION if control in CALIBRATION_CONTROL_KINDS else BIAS
        cells[family] += [
            control_cell(control, group, term, level)
            for level in control_levels(control, group)
            for term in terms
        ]
    for label in band_labels(group):
        cells[SIMULTANEOUS].append(f"{label}__{BAND_KIND}")
        if group.band == "gated":
            cells[SIMULTANEOUS].append(f"{label}__{JOINT_CONTROL_KIND}")
    return {family: tuple(names) for family, names in cells.items() if names}


def property_cells() -> dict[str, tuple[str, ...]]:
    """Every declared property cell, by family, in group order."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for family, names in group_cells(group).items():
            out.setdefault(family, []).extend(names)
    return {family: tuple(names) for family, names in out.items()}


PROPERTY_CELLS = property_cells()


def calibration_arms() -> dict[str, tuple[str, ...]]:
    """The gated calibration arms and their rows: one parameter kind of one fit and level each."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for arm in group.arms:
            out.setdefault(arm, []).extend(calibration_cell(group, arm, t) for t in group.terms)
    return {arm: tuple(rows) for arm, rows in out.items()}


CALIBRATION_ARMS = calibration_arms()


# ---------------------------------------------------------------------------- the learners


class SplineIntermediate(LogisticRegression):
    """The per-arm CDE outcome learner: cubic splines of ``W1, W2, W3``, then the raw ``Z``.

    A per-arm fit (``outcome_by_arm=True``) reads ``[W1, W2, W3, Z]``.  The law's arm-``a``
    logit is additive in ``W`` plus ``z (0.5 - 0.9 a)``, so the main-terms GLM on these
    features holds it up to the spline's approximation of ``sin(pi W3)``.
    """

    def __init__(self, max_iter: int = 2_000) -> None:
        super().__init__(max_iter=max_iter)

    def _features(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "splines_"):
            self.splines_ = SplineTransformer(n_knots=5, degree=3).fit(X[:, :3])
        return np.column_stack([self.splines_.transform(X[:, :3]), X[:, 3:4]])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SplineIntermediate:
        if hasattr(self, "splines_"):
            del self.splines_
        return super().fit(self._features(X), y, sample_weight=sample_weight)

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(self._features(X))


class PooledSpline(LogisticRegression):
    """The pooled outcome learner: ``spline(W)``, ``A`` and ``A x spline(W)``, one GLM.

    A pooled fit (``outcome_by_arm=False``) reads ``[A, W1, W2, W3]``.
    """

    def __init__(self, max_iter: int = 4_000) -> None:
        super().__init__(max_iter=max_iter)

    def _features(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "splines_"):
            self.splines_ = SplineTransformer(n_knots=5, degree=3).fit(X[:, 1:4])
        a = X[:, :1]
        basis = self.splines_.transform(X[:, 1:4])
        return np.column_stack([basis, a, a * basis])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> PooledSpline:
        if hasattr(self, "splines_"):
            del self.splines_
        return super().fit(self._features(X), y, sample_weight=sample_weight)

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(self._features(X))


class DeclaredDirectOutcome(ClassifierMixin, BaseEstimator):
    """The declared wrong ``Qbar`` of ``cde_dr_4k``: the true logit plus the declared error.

    A pooled fit reads ``[A, W1, W2, W3, Z]`` and learns nothing, so the error is the declared
    one in every fold and the TMLE limits of :mod:`tests.studies.projection_evidence_law` are
    exact.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> DeclaredDirectOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        a, w, z = X[:, 0], X[:, 1:4], X[:, 4]
        p = expit(cde.outcome_logit(w, a, z) + cde.declared_error(w, a))
        return np.column_stack([1.0 - p, p])


class TruePropensity(ClassifierMixin, BaseEstimator):
    """The true ``g(1 | W)`` of ``cde_obs``; reads ``[W1, W2, W3]``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TruePropensity:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        p = cp.propensity(np.asarray(X, dtype=float))
        return np.column_stack([1.0 - p, p])


class TrueIntermediate(ClassifierMixin, BaseEstimator):
    """The true ``q_1(a, W)`` of ``cde_obs``; reads ``[A, W1, W2, W3]``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TrueIntermediate:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p = cde.intermediate_probability(X[:, 1:4], X[:, 0])
        return np.column_stack([1.0 - p, p])


def learners(group: Group) -> dict[str, Any]:
    """The nuisance learners and fit settings of ``group``."""
    glm = LogisticRegression(max_iter=2_000)
    if group.kind == "cde":
        settings: dict[str, Any] = {
            "outcome_learner": SplineIntermediate(),
            "treatment_learner": glm,
            "intermediate_learner": LogisticRegression(max_iter=2_000),
            "outcome_by_arm": True,
            "n_folds": CDE_FOLDS,
        }
    elif group.kind == "cde_dr":
        settings = {
            "outcome_learner": DeclaredDirectOutcome(),
            "treatment_learner": TruePropensity(),
            "intermediate_learner": TrueIntermediate(),
            "outcome_by_arm": False,
            "n_folds": CDE_FOLDS,
        }
    elif group.kind == "cde_total":
        settings = {
            "outcome_learner": cp.spline_glm(),
            "treatment_learner": glm,
            "outcome_by_arm": True,
            "n_folds": CDE_FOLDS,
        }
    elif group.kind == "pooled":
        settings = {"outcome_learner": PooledSpline(), "treatment_learner": glm}
    else:
        settings = {
            "outcome_learner": cp.spline_glm(),
            "treatment_learner": glm,
            "outcome_by_arm": True,
        }
    if group.repeats > 1:
        settings["repeats"] = group.repeats
    return settings


# -------------------------------------------------------------------------------- the fits


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def fit_cde(group: Group, frame: pd.DataFrame, random_state: int) -> Any:
    """One engine fit of the CDE projection at both levels, which share every nuisance."""
    estimator = TMLE(
        effect_modification=EffectModification(CDE_BASIS, CDE_MODIFIERS, group.scale),  # type: ignore[arg-type]
        random_state=random_state,
        **learners(group),
    )
    return _quiet(
        estimator.fit,
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2", "W3"],
        intermediate="Z",
    )


def design(group: Group) -> PointTreatment:
    """The study design of a ``proj_obs`` or ``cde_total`` group."""
    roles: dict[str, Any] = {}
    if group.law == "clustered":
        roles["cluster"] = "cluster"
    return PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), **roles)


def estimand(group: Group) -> ConditionalContrast:
    """The estimand of a ``proj_obs`` or ``cde_total`` group."""
    if group.law == "cde_obs":
        return ConditionalContrast(CDE_MODIFIERS, scale=group.scale, model=CDE_BASIS)  # type: ignore[arg-type]
    return ConditionalContrast(cp.MODIFIERS, scale=group.scale, model=cp.BASIS)  # type: ignore[arg-type]


def fit_projection(group: Group, frame: pd.DataFrame, random_state: int) -> Any:
    """The projection of a ``proj_obs`` group, or the total-effect one without the intermediate."""
    if group.kind == "cde_total":
        frame = frame.drop(columns=["Z"])
    study = CausalStudy(frame, design=design(group))
    effect = study.identify(estimand(group))
    return _quiet(effect.estimate, **learners(group), random_state=random_state)


def cde_phi(fit: Any) -> np.ndarray:
    """The basis ``(1, W1)`` at the fit's rows."""
    w1 = np.asarray(fit.data.covariates, dtype=float)[:, fit.data.covariate_names.index("W1")]
    return np.column_stack([np.ones(len(w1)), w1])


def _ols(phi: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.linalg.solve(phi.T @ phi, phi.T @ values)


def plugin_stratum(fit: Any, level: int) -> np.ndarray:
    """The plug-in-stratum control: the targeted contrast projected over the ``Z = level`` rows."""
    targeted = fit.repeats[0].fluctuations["effect_modification"].targeted
    keep = np.asarray(fit.data.intermediate, dtype=float) == float(level)
    phi = cde_phi(fit)[keep]
    return _ols(phi, (targeted.arms[1.0] - targeted.arms[0.0])[keep])


def mutant_tmle(fit: Any, level: int, covariate: str) -> np.ndarray:
    r"""The difference-scale TMLE at ``level`` with a mutant clever covariate, on ``fit``'s nuisances.

    The fit's stacked out-of-fold initial ``Qbar``, ``g`` and ``q_z`` (at the fit's own bounds)
    are fluctuated once, logistically, along ``s_a 1{A = a} 1{Z = z} phi / (g_a q_z(a))`` with
    ``"no_selection"`` dropping ``1{Z = z}`` and ``"no_density"`` dropping ``q_z``; the
    counterfactual covariate at ``(a, z)`` is ``s_a phi / (g_a q_z(a))`` (without ``q_z`` for
    ``"no_density"``).  ``eps`` solves the empirical score by Newton's method and the
    coefficients are the projection of ``Q*_eps(1, z) - Q*_eps(0, z)``.  With ``"full"`` this
    is the shipped pooled TMLE's covariate, which the design test checks against the fit.
    """
    nuisance = fit.repeats[0].nuisance
    data = fit.data
    a = np.asarray(data.treatment, dtype=float)
    y = np.asarray(data.outcome, dtype=float)
    z = np.asarray(data.intermediate, dtype=float)
    phi = cde_phi(fit)
    g = np.asarray(nuisance.bounded_propensity(tuple(fit.config.g_bounds)), dtype=float)
    column = {arm: nuisance.propensity.column_for(arm) for arm in (0.0, 1.0)}
    density = np.asarray(
        nuisance.intermediate_density(float(level), fit.config.missingness_bound), dtype=float
    )
    if covariate == "no_density":
        density = np.ones_like(density)
    elif covariate not in ("full", "no_selection"):
        raise KeyError(covariate)
    counterfactual = {
        arm: (1.0 if arm == 1.0 else -1.0)
        * phi
        / (g[:, column[arm]] * density[:, column[arm]])[:, None]
        for arm in (0.0, 1.0)
    }
    keep = np.ones(len(a)) if covariate == "no_selection" else (z == float(level)).astype(float)
    observed = keep[:, None] * np.where(a[:, None] == 1.0, counterfactual[1.0], counterfactual[0.0])
    start = logit(np.clip(nuisance.outcome.observed, 1e-9, 1 - 1e-9))
    epsilon = np.zeros(phi.shape[1])
    for _ in range(50):
        q = expit(start + observed @ epsilon)
        score = observed.T @ (y - q)
        jacobian = -(observed.T @ ((q * (1.0 - q))[:, None] * observed))
        step = np.linalg.solve(jacobian, score)
        epsilon = epsilon - step
        if np.max(np.abs(step)) < 1e-10:
            break
    targeted = {
        arm: expit(
            logit(np.clip(nuisance.outcome.arms[arm], 1e-9, 1 - 1e-9))
            + counterfactual[arm] @ epsilon
        )
        for arm in (0.0, 1.0)
    }
    return _ols(phi, targeted[1.0] - targeted[0.0])


def plugin_wrong(fit: Any) -> np.ndarray:
    """The plug-in projection of the initial (declared wrong) ``Qbar``."""
    arms = fit.repeats[0].nuisance.outcome.arms
    return _ols(cde_phi(fit), arms[1.0] - arms[0.0])


def band_grid(group: Group) -> pd.DataFrame:
    """The band grid: ``W1`` in ``linspace(-0.8, 0.8, 5)`` on ``cde_obs``, else ``{-0.8, 0, 0.8}^2``."""
    if group.law == "cde_obs":
        return pd.DataFrame({"W1": cde.BAND_GRID_W1})
    return cde.proj_band_grid()


def band_truth(group: Group, truths: Mapping[str, float], level: int | None) -> np.ndarray:
    """The true curve at the band grid's points."""
    grid = band_grid(group)
    beta = np.array([truths[group.name(term, level)] for term in group.terms])
    phi = np.column_stack([np.ones(len(grid)), *(grid[c].to_numpy() for c in grid.columns)])
    return phi @ beta


def band_reading(result: Any, group: Group, random_state: int) -> dict[str, Any]:
    """The curve, its SE and band at the grid, and the band's critical value."""
    grid = band_grid(group)
    curve = result.effect_curve(grid=grid, n_replicates=BAND_DRAWS, random_state=random_state)
    band = curve.band
    return {
        "estimate": np.asarray(band.estimate, dtype=float),
        "std_error": np.asarray(curve.model.std_error(grid), dtype=float),
        "lower": np.asarray(band.lower, dtype=float),
        "upper": np.asarray(band.upper, dtype=float),
        "critical": float(band.critical_value),
    }


def fit_group(group: Group, frame: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    """Fit one replication of ``group`` and return what each of its cells reads.

    Returns
    -------
    dict
        ``"estimates"``: the reported estimate objects by name (``z0:``-prefixed on a CDE
        fit); ``"controls"``: each control's values by ``(term, level)``; ``"bands"``: each
        band label's reading (:func:`band_reading`).
    """
    out: dict[str, Any] = {"estimates": {}, "controls": {}, "bands": {}}
    if group.kind in ("cde", "cde_dr"):
        fits = fit_cde(group, frame, random_state)
        results: dict[int | None, Any] = {level: fits[float(level)] for level in LEVELS}
        for level, result in results.items():
            for term in group.terms:
                out["estimates"][group.name(term, level)] = result.estimates[
                    f"{group.stem}[{term}]"
                ]
    else:
        result = fit_projection(group, frame, random_state)
        results = {None: result}
        for term in group.terms:
            out["estimates"][group.name(term, None)] = result.estimates[f"{group.stem}[{term}]"]
    for control, terms in group.controls.items():
        values: dict[tuple[str, int | None], Any] = {}
        for level in control_levels(control, group):
            if control == "plugin_stratum":
                beta = plugin_stratum(results[level], level)  # type: ignore[arg-type]
            elif control == "no_selection":
                beta = mutant_tmle(results[level], level, "no_selection")  # type: ignore[arg-type]
            elif control == "no_intermediate_weight":
                beta = mutant_tmle(results[level], level, "no_density")  # type: ignore[arg-type]
            elif control == "plugin_wrong":
                beta = plugin_wrong(results[level])
            elif control == "total_effect":
                result = results[None]
                beta = np.array([result.estimates[f"{group.stem}[{t}]"].psi for t in group.terms])
            elif control == "shrunken_se":
                beta = np.array(
                    [
                        SHRUNKEN_FACTOR * float(out["estimates"][group.name(t, level)].std_error)
                        for t in group.terms
                    ]
                )
            else:
                raise KeyError(control)
            for term, value in zip(group.terms, beta, strict=True):
                if term in terms:
                    values[term, level] = float(value)
        out["controls"][control] = values
    for level in group.levels if group.band else ():
        label = f"{group.label}{level_label(level)}"
        out["bands"][label] = band_reading(results[level], group, random_state)
    return out


def control_truth(
    group: Group, control: str, term: str, level: int | None, truths: Mapping[str, float]
) -> float:
    """The truth a control row reads: the CDE truth at its level on the difference scale."""
    if group.kind == "cde_total":
        return float(truths[f"z{level}:cate_projection[{term}]"])
    return float(truths[group.name(term, level)])


# ----------------------------------------------------------------------- the truths and draws


def truth(group: Group) -> dict[str, float]:
    """The exact coefficients of a group, keyed by :meth:`Group.name`."""
    if group.law == "cde_obs":
        scale = "difference" if group.kind == "cde_total" else group.scale
        stem = "cate_projection" if scale == "difference" else "log_crr_projection"
        return {
            f"z{level}:{stem}[{term}]": value
            for level in LEVELS
            for term, value in zip(CDE_TERMS, cde.truth(level, scale), strict=True)
        }
    return cp.truth(group.law, group.scale)


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed, with the group's truths."""
    group = GROUP_BY_LABEL[group_label]
    return sample(group.law, n, np.random.default_rng(seed)), truth(group)


# ----------------------------------------------------------------------- the primary phase

#: The primary phase (plan 5.2): the ``cde_4k`` intercepts at both levels on the primary stream.
PRIMARY_GROUP = "cde_4k"
PRIMARY_REPLICATES = 2_400
PRIMARY_N = GROUP_BY_LABEL[PRIMARY_GROUP].n
#: The coverage floor the primary reads (plan 5.2).
PRIMARY_COVERAGE_FLOOR = 0.90

SCENARIOS = {
    PRIMARY_GROUP: tuple(
        GROUP_BY_LABEL[PRIMARY_GROUP].name("(intercept)", level) for level in LEVELS
    )
}

STUDY = StudyRecord(
    name="Projection evidence (CDE, pooled regression, band)",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "projection_evidence",
    document="docs/technical-reference/method-evidence/projection-evidence.md",
    anchor="projection-evidence",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-projection-tmle",
    reference=None,
    failure_bound={"cleverly-projection-tmle": 0.01},
    modules=(
        "tests/studies/projection_evidence.py",
        "tests/studies/projection_evidence_law.py",
        "tests/studies/projection_evidence_pilot.py",
        "tests/studies/projection_evidence_properties.py",
        "tests/studies/conditional_projection.py",
        "tests/studies/point_reading.py",
        "tests/studies/repeated_sizing.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "src/cleverly/effect_curves.py",
        "src/cleverly/effect_modification.py",
        "src/cleverly/estimators/direct_effect.py",
        "src/cleverly/estimators/targeting.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/estimators/base.py",
        "src/cleverly/fluctuation/submodel.py",
        "src/cleverly/inference/influence.py",
        "src/cleverly/inference/multiplier.py",
        "src/cleverly/inference/results.py",
        "src/cleverly/_support_rows.py",
    ),
    runner_module="tests.studies.projection_evidence",
    properties_module="tests.studies.projection_evidence_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset({BIAS}),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    group.label: (
        f"{group.law} law, n={group.n}, R={group.replicates}, {group.kind}, scale {group.scale}"
        f"{'' if group.repeats == 1 else f', repeats={group.repeats}'}"
    )
    for group in GROUPS
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One primary row per declared intercept of the ``cde_4k`` fit."""
    group = GROUP_BY_LABEL[scenario]
    fold_seed = stream_seed(STUDY, "fold_partition", "primary", scenario, replicate)
    fitted = fit_group(group, frame, random_state=fold_seed)["estimates"]
    rows = []
    for name in SCENARIOS[scenario]:
        estimate = fitted[name]
        reference = float(truths[name])
        std_error, low, high = reported_inference(estimate)
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float("nan"),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truths = drawn
    return primary_rows(frame, truths, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


# ---------------------------------------------------------------------- the design numbers


def design_numbers() -> dict[str, Any]:
    """Every design number that needs no pilot, computed rather than copied (plan 5.1 to 5.4)."""
    table = cde.design_table()
    cde_band = cde.cde_band_reading(0)
    proj_band = cde.proj_band_reading()
    return {
        "cde": table,
        "proj_beta": tuple(cp.truth("observational").values()),
        "proj_sd": tuple(np.sqrt(np.diag(cp.efficient_covariance("observational", 4_000)))),
        "smallest_density_product": cde.smallest_density_product(),
        "cde_band": cde_band,
        "proj_band": proj_band,
        "calibration_arms": {arm: len(rows) for arm, rows in CALIBRATION_ARMS.items()},
        "cpu_hours": run_cpu_hours(),
        "pilot_cpu_hours": pilot_cpu_hours(),
    }


# ------------------------------------------------------------------------- the run's cost


def run_cpu_hours() -> float:
    """Single-thread CPU-hours of the declared property run and its primary phase."""
    primary = GROUP_BY_LABEL[PRIMARY_GROUP].seconds * PRIMARY_REPLICATES
    return (sum(g.seconds * g.replicates for g in GROUPS) + primary) / 3_600.0


def pilot_groups() -> tuple[str, ...]:
    """The groups the pilot fits: every calibration group and every gated band group."""
    return tuple(g.label for g in GROUPS if g.arms or g.band == "gated")


def pilot_cpu_hours(draws: int = PILOT_DRAWS) -> float:
    """Single-thread CPU-hours of the pilot: ``draws`` per pilot group."""
    return sum(GROUP_BY_LABEL[label].seconds * draws for label in pilot_groups()) / 3_600.0


def smoke_cpu_hours(replicates: int = 20) -> float:
    """Single-thread CPU-hours of a ``replicates``-draw smoke of every group."""
    return sum(g.seconds * replicates for g in GROUPS) / 3_600.0


#: Plan 5.0's cost rule: a group whose smoke cost exceeds this multiple of its declared
#: ``seconds`` runs as a second declared run.
COST_GATE = 1.5

#: The smoke timings (``group``, ``fits``, ``seconds_per_fit``), written by
#: ``tests/canonical/projection_evidence/regenerate.py --property-smoke 20 --write-timings``.
SMOKE_TIMINGS = ROOT / "tests" / "studies" / "projection_evidence_smoke.csv"

#: Every file the declared run writes into :attr:`StudyRecord.artifacts`.
RUN_FILES = (
    "replicates.parquet",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.parquet",
    "properties.csv",
    "manifest.json",
    "run.log",
)
CONDITIONAL_RUN_FILES = (
    "property-failures.csv",
    "property-failure-accounting.csv",
    "property-failure-witnesses.parquet",
)
