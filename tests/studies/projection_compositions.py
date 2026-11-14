r"""Declaration and driver of the ``projection-compositions`` study (P1 plan rev 2.1, section 5.1).

The study measures the compositions of the projection TMLE of a
:class:`cleverly.ConditionalContrast` that P1 PR 1 opens: the in-sample fit with its Wald and
full-refit bootstrap intervals, a missing treatment through the composite indicator
``Delta_A * Delta * 1{A = a}``, baseline ``strata=`` on both scales (the log ratio reads its
marginal names from a second, unstratified fluctuation), and ``repeats=3`` by the median rule
and the matrix rule.  Everything a run reads is fixed here before it starts:

- the laws, as samplers (:func:`sample`): the ``conditional-projection`` observational and
  ``hetero_trial`` laws, and the new ``mtreat``, ``mtreat_both``, ``mtreat_dr`` and ``strata``;
- the exact truths, by Gauss--Legendre quadrature over ``U(-1, 1)^3`` (:func:`truth`);
- the fit groups (:data:`GROUPS`) and the cells their fits fill (:func:`group_cells`), with the
  controls that share their fits;
- the design numbers that need no pilot (:func:`design_numbers`): the ``mtreat`` complete-case
  and ``pi_A``-dropped displacements at the factor GLMs' population limits, the smallest
  ``g_c,a``, the escalation of ``c`` the plan declares, the stratum-minus-marginal gaps, the
  strata learner's population limit, and the heterogeneity power;
- the study record (:data:`STUDY`).

The calibration ``R`` is not here: the plan computes it from 3,200-draw pilots per arm
(``calibration_family_pass_probability`` over the arms of :data:`CALIBRATION_ARMS`), which have
not run.  :data:`BUDGETED_R` is the plan's budget; the declaration commit replaces it by the
computed ``R`` when that is larger, and never lowers it.  The study is not registered: it joins
:func:`tests.studies.evidence.registry.registered` with the artifacts of its declared run.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Linear, Modifiers
from cleverly.effect_curves import _pseudo_outcomes
from cleverly.estimators.composite import targeting_inputs
from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as cp
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "projection-compositions"

#: The record seeds, next to ``conditional-projection``'s and on no other study's stream.
SEED = 20261401
RESAMPLING_SEED = 20261402

#: The ``mtreat`` modifiers and basis: ``V = (W1)``, so ``W3`` lies outside ``span(phi)``.
MTREAT_MODIFIERS = Modifiers(["W1"])
MTREAT_BASIS = Linear(["W1"])
MTREAT_TERMS = ("(intercept)", "W1")

#: The declared coefficient ``c`` of ``W3`` in the recorded-treatment law, and its escalation.
TREATMENT_RECORDING_C = 2.0
ESCALATED_C = 2.5
#: The complete-case displacement of the intercept the ``complete_case`` control needs, in SD.
COMPLETE_CASE_FLOOR = 0.6

#: ``P(S = 1)`` of the ``strata`` law; ``S`` is independent of ``W``.
STRATUM_PROBABILITY = 0.4

#: The bootstrap replicates of each bootstrap group, as the shipped bootstrap study.
N_BOOTSTRAP = 200
#: The shrink factor of the ``boot_shrunk`` control, about the median of the replicates.
SHRUNKEN_FACTOR = 0.70

#: The plan's budgeted calibration ``R`` (5.1); the pilots' computed ``R`` replaces it if larger.
BUDGETED_R = 6_000
#: Pilot draws per calibration arm (5.1).
PILOT_DRAWS = 3_200

CRITICAL = float(stats.norm.ppf(0.975))


# ------------------------------------------------------------------------------ the laws


def treatment_recording(w: np.ndarray, a: Any, c: float = TREATMENT_RECORDING_C) -> np.ndarray:
    """``P(Delta_A = 1 | A, W) = expit(1.0 + c W3 - 0.5 A)``: MAR given ``(A, W)``."""
    return expit(1.0 + c * w[..., 2] - 0.5 * np.asarray(a, dtype=float))


def strata_blip(w: np.ndarray, s: Any) -> np.ndarray:
    """The ``strata`` law's arm-1 logit increment: the shipped blip ``+ 0.5 S - 0.4 S W1``."""
    s = np.asarray(s, dtype=float)
    return cp.blip(w) + 0.5 * s - 0.4 * s * w[..., 0]


def strata_outcome_logit(w: np.ndarray, a: Any, s: Any) -> np.ndarray:
    """``logit mu(a, S, W)`` of the ``strata`` law."""
    return -0.3 + 0.6 * w[..., 0] - 0.4 * w[..., 1] + np.asarray(a) * strata_blip(w, s)


def sample(
    law: str, n: int, rng: np.random.Generator, *, slope: float = 0.0, c: float | None = None
) -> pd.DataFrame:
    """Draw ``n`` rows of ``law``.

    Parameters
    ----------
    law : str
        A key of :data:`LAWS`.
    n : int
        Rows.
    rng : numpy.random.Generator
        The replication's generator.
    slope : float
        The ``hetero_trial`` slope.
    c : float or None
        The recorded-treatment coefficient; ``None`` reads :data:`TREATMENT_RECORDING_C`.

    Returns
    -------
    pandas.DataFrame
        ``Y``, ``A``, ``W1``, ``W2``, ``W3`` and the law's own columns: ``DeltaA`` (and
        ``Delta``) on the ``mtreat`` laws, ``S`` on ``strata``.
    """
    if law in ("observational", "hetero_trial"):
        return cp.sample(law, n, rng, slope=slope)
    w = rng.uniform(-1.0, 1.0, (n, 3))
    a = rng.binomial(1, cp.propensity(w))
    if law == "strata":
        s = rng.binomial(1, STRATUM_PROBABILITY, n)
        y = rng.binomial(1, expit(strata_outcome_logit(w, a, s))).astype(float)
        return cp._frame(w, a, y, S=s)
    if law not in ("mtreat", "mtreat_both", "mtreat_dr"):
        raise KeyError(law)
    y = rng.binomial(1, expit(cp.outcome_logit(w, a))).astype(float)
    recorded = rng.binomial(1, treatment_recording(w, a, TREATMENT_RECORDING_C if c is None else c))
    extra: dict[str, Any] = {"DeltaA": recorded.astype(float)}
    if law == "mtreat_both":
        observed = rng.binomial(1, cp.observation(w, a)).astype(float)
        y = np.where(observed == 1.0, y, np.nan)
        extra["Delta"] = observed
    frame = cp._frame(w, a.astype(float), y, **extra)
    frame.loc[recorded == 0, "A"] = np.nan
    return frame


#: Every law of the study.
LAWS: tuple[str, ...] = (
    "observational",
    "mtreat",
    "mtreat_both",
    "mtreat_dr",
    "strata",
    "hetero_trial",
)


# ---------------------------------------------------------------------------- the truths


def mtreat_design(w: np.ndarray) -> np.ndarray:
    """The ``mtreat`` basis ``(1, W1)`` at the rows of ``w``."""
    return np.column_stack([np.ones(len(w)), w[:, 0]])


def _names(stem: str, terms: tuple[str, ...], values: np.ndarray, label: str | None = None):
    suffix = "" if label is None else f"[{label}]"
    return {f"{stem}[{t}]{suffix}": float(v) for t, v in zip(terms, values, strict=True)}


def stratum_label(s: int) -> str:
    """The reported label of stratum ``s`` of the ``strata`` law."""
    return f"S={s}"


def truth(
    law: str, scale: str = "difference", *, slope: float = 0.0, nodes: int | None = None
) -> dict[str, float]:
    """The exact coefficients of a cell, keyed by the name the fit reports.

    The ``mtreat`` laws project onto ``(1, W1)``; a missing treatment value does not move the
    target, which is the full-data projection.  ``strata`` reports the marginal projection over
    ``(S, W)`` and each stratum's projection over ``W`` given ``S = s``.

    Parameters
    ----------
    law : str
        A key of :data:`LAWS`.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    slope : float
        The ``hetero_trial`` slope.
    nodes : int or None
        Quadrature nodes per axis; ``None`` reads the shipped study's count.

    Returns
    -------
    dict of str to float
        The truth of each reported parameter.
    """
    stem = "cate_projection" if scale == "difference" else "log_crr_projection"
    if law in ("observational", "hetero_trial"):
        return cp.truth(law, scale, slope=slope, nodes=nodes)
    w, mass = cp._grid(nodes)
    if law in ("mtreat", "mtreat_both", "mtreat_dr"):
        q_t, q_r = cp.arm_means("observational", w)
        beta = cp.projection(mtreat_design(w), mass, q_t, q_r, scale)
        return _names(stem, MTREAT_TERMS, beta)
    if law != "strata":
        raise KeyError(law)
    phi = cp.design_matrix(w)
    out: dict[str, float] = {}
    stacked_phi, stacked_mass, stacked_t, stacked_r = [], [], [], []
    for s, p_s in ((0, 1.0 - STRATUM_PROBABILITY), (1, STRATUM_PROBABILITY)):
        q_t = expit(strata_outcome_logit(w, 1, s))
        q_r = expit(strata_outcome_logit(w, 0, s))
        out.update(_names(stem, cp.TERMS, cp.projection(phi, mass, q_t, q_r, scale), f"S={s}"))
        stacked_phi.append(phi)
        stacked_mass.append(p_s * mass)
        stacked_t.append(q_t)
        stacked_r.append(q_r)
    marginal = cp.projection(
        np.vstack(stacked_phi),
        np.concatenate(stacked_mass),
        np.concatenate(stacked_t),
        np.concatenate(stacked_r),
        scale,
    )
    out.update(_names(stem, cp.TERMS, marginal))
    return out


# ------------------------------------------------------------------------- the fit groups


CALIBRATION = "interval_calibration"
BIAS = "bias"
TYPE_I = "type_i_error"
POWER = "power"
POSITIVE = "correctly_specified"


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
    learner : str
        A key of :func:`learners`.
    scale : str
        The contrast's scale.
    splitting : {"in_sample", "stacked"}
        In sample (``cross_fit=False``) or stacked cross-fitting.
    repeats : int
        Cross-fitting repeats.
    bootstrap : bool
        Whether the fit runs the full-refit bootstrap (:data:`N_BOOTSTRAP` replicates).
    arms : tuple of str
        The calibration arms (``"__wald_diff"`` and so on) the group's rows join; empty for a
        group that claims no calibration.
    bias_rows : tuple of str
        Terms whose TMLE estimate is a positive ``bias`` row.
    controls : Mapping of str to tuple of str
        Each control the group's fits compute, and the terms it publishes rows for.
    reporting : bool
        Whether the group publishes without a verdict.
    slope : float
        The ``hetero_trial`` slope.
    folds : int or None
        Outer folds; ``None`` keeps the estimator's default.
    seconds : float
        Declared single-thread seconds per replication (the plan's estimates; the smoke
        replaces them).
    """

    label: str
    law: str
    n: int
    replicates: int
    learner: str = "spline"
    scale: str = "difference"
    splitting: str = "stacked"
    repeats: int = 1
    bootstrap: bool = False
    arms: tuple[str, ...] = ()
    bias_rows: tuple[str, ...] = ()
    controls: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    reporting: bool = False
    slope: float = 0.0
    folds: int | None = None
    seconds: float = 0.3

    @property
    def terms(self) -> tuple[str, ...]:
        """The basis terms the group's fit reports."""
        return MTREAT_TERMS if self.law.startswith("mtreat") else cp.TERMS

    @property
    def stem(self) -> str:
        """The reported name stem of the group's coefficients."""
        return "cate_projection" if self.scale == "difference" else "log_crr_projection"

    def names(self) -> tuple[str, ...]:
        """Every reported name the group's cells read: marginal, then stratum names."""
        labels: tuple[str | None, ...] = (None,)
        if self.law == "strata":
            labels = (None, stratum_label(0), stratum_label(1))
        return tuple(
            f"{self.stem}[{term}]" + ("" if label is None else f"[{label}]")
            for label in labels
            for term in self.terms
        )


R = BUDGETED_R
GROUPS: tuple[Group, ...] = (
    Group(
        "insample_2k",
        "observational",
        2_000,
        R,
        splitting="in_sample",
        bootstrap=True,
        arms=("__wald_diff", "__boot_diff"),
        controls={"boot_shrunk": cp.TERMS},
        seconds=12.0,
    ),
    Group(
        "insample_crr_2k",
        "observational",
        2_000,
        R,
        scale="log_ratio",
        splitting="in_sample",
        bootstrap=True,
        arms=("__wald_crr", "__boot_crr"),
        seconds=20.0,
    ),
    Group(
        "boot_stacked_2k",
        "observational",
        2_000,
        1_000,
        bootstrap=True,
        arms=("__boot_stacked",),
        reporting=True,
        seconds=44.0,
    ),
    Group(
        "mtreat_4k",
        "mtreat",
        4_000,
        R,
        splitting="in_sample",
        arms=("__mtreat",),
        controls={"complete_case": ("(intercept)",)},
        seconds=0.7,
    ),
    Group(
        "mtreat_both_4k",
        "mtreat_both",
        4_000,
        R,
        splitting="in_sample",
        arms=("__mtreat_both",),
        seconds=0.7,
    ),
    Group(
        "mtreat_crr_4k",
        "mtreat",
        4_000,
        R,
        scale="log_ratio",
        splitting="in_sample",
        arms=("__mtreat_crr",),
        seconds=0.7,
    ),
    Group(
        "mtreat_dr_4k",
        "mtreat_dr",
        4_000,
        1_000,
        learner="declared_missing",
        splitting="in_sample",
        bias_rows=MTREAT_TERMS,
        controls={"no_treatment_observation": ("(intercept)",)},
        seconds=0.7,
    ),
    Group(
        "strata_4k",
        "strata",
        4_000,
        R,
        learner="strata_spline",
        arms=("__strata",),
        controls={"marginal_as_stratum": ("(intercept)", "W1")},
        seconds=0.65,
    ),
    Group(
        "strata_crr_4k",
        "strata",
        4_000,
        R,
        learner="strata_spline",
        scale="log_ratio",
        arms=("__strata_crr",),
        seconds=0.7,
    ),
    Group(
        "rep3_obs_4k",
        "observational",
        4_000,
        R,
        repeats=3,
        arms=("__rep3",),
        controls={"se_over_root_r": cp.TERMS},
        seconds=1.4,
    ),
    Group(
        "rep3_crr_4k",
        "observational",
        4_000,
        R,
        scale="log_ratio",
        repeats=3,
        arms=("__rep3_crr",),
        seconds=1.4,
    ),
    Group(
        "rep3_sl_2k",
        "observational",
        2_000,
        R,
        learner="ensemble",
        repeats=3,
        folds=5,
        arms=("__rep3_sl",),
        seconds=111.0,
    ),
    Group(
        "rep3_null_2k",
        "hetero_trial",
        2_000,
        2_400,
        learner="risk_glm",
        repeats=3,
        seconds=0.6,
    ),
    Group(
        "rep3_power_4k",
        "hetero_trial",
        4_000,
        1_000,
        learner="risk_glm",
        repeats=3,
        slope=0.10,
        seconds=1.0,
    ),
)

GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}

#: What each control computes from its group's fit, and the rule it must fail.
CONTROL_CLAIMS: Mapping[str, str] = {
    "boot_shrunk": "the percentile interval shrunk by 0.70 about the replicates' median: its "
    "SE-ratio interval must fall below the band",
    "complete_case": "the projection fitted on the rows with a recorded treatment, with the "
    "same learners: its bias must lie wholly outside +-0.25 of its own SD",
    "no_treatment_observation": "the one-step with pi_A replaced by 1 in g_c,a, on the declared "
    "wrong mu: its bias must lie wholly outside the margin",
    "marginal_as_stratum": "the marginal coefficients read against the stratum-1 truth: their "
    "bias must lie wholly outside the margin",
    "se_over_root_r": "the reported SE divided by sqrt(3), the error of reading three draws as "
    "independent replicates: its SE-ratio interval must fall below the band",
}

#: The controls read by ``interval_calibration`` (the rest are ``bias`` controls).
CALIBRATION_CONTROLS = frozenset({"boot_shrunk", "se_over_root_r"})


def term_label(term: str) -> str:
    """A cell-name fragment of ``term``."""
    return "intercept" if term == "(intercept)" else term


def name_label(name: str) -> str:
    """A cell-name fragment of a reported name: its term and, for a stratum name, its label."""
    inner = name.split("[", 1)[1]
    term, _, rest = inner.partition("]")
    stratum = rest.strip("[]").replace("=", "")
    return term_label(term) + (f"_{stratum}" if stratum else "")


def calibration_cell(group: Group, arm: str, name: str) -> str:
    """The ``interval_calibration`` cell of one name of one arm of a group (``label__arm``)."""
    return f"{group.label}_{name_label(name)}{arm}"


def control_label(control: str, group: Group) -> str:
    """The cell prefix of ``control`` on ``group``: the control name and the size."""
    return f"{control}_{group.label.rsplit('_', 1)[-1]}"


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells ``group``'s fits fill, by family."""
    cells: dict[str, list[str]] = {CALIBRATION: [], BIAS: [], TYPE_I: [], POWER: []}
    for arm in group.arms:
        cells[CALIBRATION] += [calibration_cell(group, arm, name) for name in group.names()]
    cells[BIAS] += [f"{group.label}_{term_label(term)}" for term in group.bias_rows]
    for control, terms in group.controls.items():
        family = CALIBRATION if control in CALIBRATION_CONTROLS else BIAS
        cells[family] += [f"{control_label(control, group)}_{term_label(term)}" for term in terms]
    if group.label == "rep3_null_2k":
        cells[TYPE_I].append("heterogeneity__rep3_sharp_null")
    if group.label == "rep3_power_4k":
        cells[POWER].append("heterogeneity__rep3_slope_0.10")
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
    """The gated calibration arms and their rows: one interval kind, one law and scale each."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        if group.reporting:
            continue
        for arm in group.arms:
            out.setdefault(arm, []).extend(calibration_cell(group, arm, n) for n in group.names())
    return {arm: tuple(rows) for arm, rows in out.items()}


CALIBRATION_ARMS = calibration_arms()


# ---------------------------------------------------------------------------- the learners


class StrataSpline(LogisticRegression):
    """The ``strata`` outcome learner: spline features of ``W1, W2, W3``, then ``S`` and ``S W1``.

    It reads ``[W1, W2, W3, S]`` (the adjustment order of :func:`design`) and fits a main-terms
    logistic regression on the cubic-spline features of the three ``W`` columns with the raw
    ``S`` and ``S W1`` appended, so the arm-1 regression can hold ``0.5 S - 0.4 S W1``.
    """

    def __init__(self, max_iter: int = 2_000) -> None:
        super().__init__(max_iter=max_iter)

    def _features(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "splines_"):
            self.splines_ = SplineTransformer(n_knots=5, degree=3).fit(X[:, :3])
        s = X[:, 3:4]
        return np.column_stack([self.splines_.transform(X[:, :3]), s, s * X[:, :1]])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> StrataSpline:
        if hasattr(self, "splines_"):
            del self.splines_
        return super().fit(self._features(X), y, sample_weight=sample_weight)

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(self._features(X))


def learners(group: Group) -> dict[str, Any]:
    """The nuisance learners and fit settings of ``group``."""
    glm = LogisticRegression(max_iter=2_000)
    if group.learner == "ensemble":
        settings: dict[str, Any] = {
            "outcome_learner": cp.ensemble((0.005, 0.995)),
            "treatment_learner": cp.ensemble((0.01, 0.99)),
            "outcome_by_arm": True,
        }
    elif group.learner == "risk_glm":
        settings = {"outcome_learner": cp.RiskGLM(), "outcome_by_arm": True}
    elif group.learner == "declared_missing":
        settings = {"outcome_learner": cp.DeclaredOutcome("missing"), "treatment_learner": glm}
    elif group.learner == "strata_spline":
        settings = {
            "outcome_learner": StrataSpline(),
            "treatment_learner": glm,
            "outcome_by_arm": True,
        }
    else:
        settings = {
            "outcome_learner": make_pipeline(
                SplineTransformer(n_knots=5, degree=3), LogisticRegression(max_iter=2_000)
            ),
            "treatment_learner": glm,
            "outcome_by_arm": True,
        }
    if group.law.startswith("mtreat"):
        settings["missingness_learner"] = LogisticRegression(max_iter=2_000)
    if group.splitting == "in_sample":
        settings["cross_fit"] = False
    if group.repeats > 1:
        settings["repeats"] = group.repeats
    if group.folds is not None:
        settings["n_folds"] = group.folds
    if group.bootstrap:
        settings["n_bootstrap"] = N_BOOTSTRAP
        settings["n_jobs"] = 1
    return settings


def design(group: Group) -> PointTreatment:
    """The study design of ``group``'s law."""
    roles: dict[str, Any] = {}
    adjustment: tuple[str, ...] = ("W1", "W2", "W3")
    if group.law == "hetero_trial":
        roles["treatment_probabilities"] = {0: "p0", 1: "p1"}
    if group.law.startswith("mtreat"):
        roles["treatment_missingness"] = "DeltaA"
    if group.law == "mtreat_both":
        roles["missingness"] = "Delta"
    if group.law == "strata":
        adjustment = ("W1", "W2", "W3", "S")
        roles["strata"] = ("S",)
    return PointTreatment(outcome="Y", treatment="A", adjustment=adjustment, **roles)


def estimand(group: Group) -> ConditionalContrast:
    """The estimand of ``group`` on its scale and basis."""
    if group.law.startswith("mtreat"):
        return ConditionalContrast(MTREAT_MODIFIERS, scale=group.scale, model=MTREAT_BASIS)  # type: ignore[arg-type]
    return ConditionalContrast(cp.MODIFIERS, scale=group.scale, model=cp.BASIS)  # type: ignore[arg-type]


# -------------------------------------------------------------------------------- the fits


def _fit(group: Group, frame: pd.DataFrame, random_state: int) -> Any:
    study = CausalStudy(frame, design=design(group))
    effect = study.identify(estimand(group))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(**learners(group), random_state=random_state)


def complete_case(group: Group, frame: pd.DataFrame, random_state: int) -> dict[str, float]:
    """The ``complete_case`` control: the same fit on the rows with a recorded treatment."""
    recorded = frame.loc[frame["DeltaA"] == 1.0].drop(columns=["DeltaA"]).reset_index(drop=True)
    plain = Group(group.label, "observational", len(recorded), group.replicates)
    study = CausalStudy(recorded, design=design(plain))
    effect = study.identify(estimand(group))
    settings = {k: v for k, v in learners(group).items() if k != "missingness_learner"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = effect.estimate(**settings, random_state=random_state)
    return {term: float(fit.estimates[f"{group.stem}[{term}]"].psi) for term in group.terms}


def no_treatment_observation(fit: Any, group: Group) -> dict[str, float]:
    """The one-step coefficients with ``pi_A`` replaced by 1 in ``g_c,a``, from ``fit``'s nuisances.

    On the composite view the one-step pseudo-outcome is ``Qbar_1 - Qbar_0 + C_1 / g_c,1 (Y -
    Qbar_1) - C_0 / g_c,0 (Y - Qbar_0)``; the control divides by ``g(a | Delta_A = 1, W)`` alone.
    Its ratio to the shipped pseudo-outcome's correction is ``pi_A(W)``, which is first order
    only through a wrong outcome regression, as on ``mtreat_dr``.
    """
    view, _ = targeting_inputs(fit)
    factor = replace(fit.repeats[0].nuisance, missingness=None, treatment_observation=None)
    frame = pd.DataFrame(np.asarray(view.covariates, dtype=float), columns=view.covariate_names)
    phi = np.asarray(MTREAT_BASIS.evaluate(frame, MTREAT_MODIFIERS), dtype=float)
    _, _, d_t, d_r = _pseudo_outcomes(factor, view, (1.0, 0.0), g_bounds=tuple(fit.config.g_bounds))
    values = d_t - d_r
    beta = cp._projection_of(phi, values, np.ones(len(phi)))
    return dict(zip(group.terms, beta, strict=True))


def fit_group(group: Group, frame: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    """Fit one replication of ``group`` and return what each of its cells reads.

    Parameters
    ----------
    group : Group
        The group.
    frame : pandas.DataFrame
        One replication drawn by :func:`sample`.
    random_state : int
        The fold seed.

    Returns
    -------
    dict
        ``"estimates"``: the reported estimate objects by name; ``"bootstrap_failed"``: the
        failed bootstrap replicates (0 without the bootstrap); ``"controls"``: each control's
        values by term; for a heterogeneity group ``"statistic"`` and ``"p_value"``; for a
        repeated group ``"se_without_displacement"`` by name.
    """
    fit = _fit(group, frame, random_state)
    estimates = {name: fit.estimates[name] for name in group.names()}
    out: dict[str, Any] = {"estimates": estimates, "controls": {}, "bootstrap_failed": 0}
    if group.bootstrap:
        out["bootstrap_failed"] = int(fit.bootstrap.n_failed)
    if group.repeats > 1:
        out["se_without_displacement"] = {
            name: float(fit.se_without_displacement[name]) for name in group.names()
        }
    for control in group.controls:
        if control == "complete_case":
            out["controls"][control] = complete_case(group, frame, random_state)
        elif control == "no_treatment_observation":
            out["controls"][control] = no_treatment_observation(fit, group)
        elif control == "marginal_as_stratum":
            out["controls"][control] = {
                term: float(estimates[f"{group.stem}[{term}]"].psi) for term in group.terms
            }
        elif control == "boot_shrunk":
            shrunk = {}
            for term in group.terms:
                summary = estimates[f"{group.stem}[{term}]"].bootstrap
                median = float(np.median(summary.draws))
                low, high = summary.ci
                shrunk[term] = (
                    median + SHRUNKEN_FACTOR * (low - median),
                    median + SHRUNKEN_FACTOR * (high - median),
                    float(summary.std_error) * SHRUNKEN_FACTOR,
                )
            out["controls"][control] = shrunk
        elif control == "se_over_root_r":
            out["controls"][control] = {
                term: float(estimates[f"{group.stem}[{term}]"].std_error) / math.sqrt(3.0)
                for term in group.terms
            }
        else:
            raise KeyError(control)
    if group.law == "hetero_trial":
        test = fit.heterogeneity_test()
        out["statistic"], out["p_value"] = float(test.statistic), float(test.p_value)
    return out


def control_truth(group: Group, control: str, term: str, truths: Mapping[str, float]) -> float:
    """The truth a control row reads: the stratum-1 truth for ``marginal_as_stratum``."""
    if control == "marginal_as_stratum":
        return float(truths[f"{group.stem}[{term}][{stratum_label(1)}]"])
    return float(truths[f"{group.stem}[{term}]"])


# ----------------------------------------------------------------------- the primary phase

#: The primary phase is ``mtreat_4k`` (plan 5.1): 1,000 replications, the framework's legs.
PRIMARY_GROUP = "mtreat_4k"
PRIMARY_REPLICATES = 1_000
PRIMARY_N = GROUP_BY_LABEL[PRIMARY_GROUP].n

SCENARIOS = {PRIMARY_GROUP: tuple(f"cate_projection[{term}]" for term in MTREAT_TERMS)}

STUDY = StudyRecord(
    name="Projection compositions",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "projection_compositions",
    document="docs/technical-reference/method-evidence/projection-compositions.md",
    anchor="projection-compositions",
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
        "tests/studies/projection_compositions.py",
        "tests/studies/projection_compositions_properties.py",
        "tests/studies/conditional_projection.py",
        "src/cleverly/effect_modification.py",
        "src/cleverly/effect_curves.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/targeting.py",
        "src/cleverly/estimators/composite.py",
        "src/cleverly/estimators/_strata.py",
        "src/cleverly/estimators/base.py",
        "src/cleverly/inference/bootstrap.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.projection_compositions",
    properties_module="tests.studies.projection_compositions_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset({BIAS}),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    group.label: (
        f"{group.law} law, n={group.n}, R={group.replicates}, learner {group.learner}, "
        f"{group.splitting}"
        f"{'' if group.repeats == 1 else f', repeats={group.repeats}'}"
        f"{'' if group.folds is None else f', {group.folds} outer folds'}"
        f"{f', n_bootstrap={N_BOOTSTRAP}' if group.bootstrap else ''}, scale {group.scale}"
    )
    for group in GROUPS
}


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed, with the group's truths."""
    group = GROUP_BY_LABEL[group_label]
    frame = sample(group.law, n, np.random.default_rng(seed), slope=group.slope)
    return frame, truth(group.law, group.scale, slope=group.slope)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One primary row per coefficient of the ``mtreat_4k`` fit."""
    group = GROUP_BY_LABEL[scenario]
    fitted = fit_group(group, frame, random_state=replicate)["estimates"]
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
                "initial_estimate": math.nan,
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


def _glm_limit(features: np.ndarray, p: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """The population limit of a main-terms logistic GLM of a binary ``Y`` with ``P(Y=1) = p``.

    Fitted on the quadrature nodes twice, once with ``Y = 1`` at weight ``mass p`` and once
    with ``Y = 0`` at weight ``mass (1 - p)``, so the weighted likelihood is the population one.
    """
    model = LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-12)
    x = np.vstack([features, features])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
    weight = np.concatenate([mass * p, mass * (1.0 - p)])
    model.fit(x, y, sample_weight=weight / weight.sum() * len(weight))
    return model.predict_proba(features)[:, 1]


@dataclass(frozen=True)
class MtreatFactors:
    """The ``mtreat`` treatment factors on the quadrature nodes, true and at the GLM limits.

    Parameters
    ----------
    w : numpy.ndarray
        The nodes.
    mass : numpy.ndarray
        Their probability weights.
    pi_a : numpy.ndarray
        ``P(Delta_A = 1 | W)``, true.
    g_c : numpy.ndarray
        ``P(Delta_A = 1, A = a | W)`` by arm ``(m, 2)``, true.
    pi_a_glm : numpy.ndarray
        The main-terms logistic GLM's limit for ``pi_A``.
    g_recorded_glm : numpy.ndarray
        The GLM's limit for ``g(1 | Delta_A = 1, W)``.
    """

    w: np.ndarray
    mass: np.ndarray
    pi_a: np.ndarray
    g_c: np.ndarray
    pi_a_glm: np.ndarray
    g_recorded_glm: np.ndarray

    @property
    def g_c_glm(self) -> np.ndarray:
        """``g_c,a`` at the GLM limits, by arm."""
        g1 = self.g_recorded_glm
        return np.column_stack([self.pi_a_glm * (1.0 - g1), self.pi_a_glm * g1])


def mtreat_factors(c: float = TREATMENT_RECORDING_C, *, nodes: int = 32) -> MtreatFactors:
    """The ``mtreat`` factors at recorded-treatment coefficient ``c``."""
    w, mass = cp._grid(nodes)
    g1 = cp.propensity(w)
    recorded = np.column_stack([treatment_recording(w, 0, c), treatment_recording(w, 1, c)])
    g_c = np.column_stack([(1.0 - g1) * recorded[:, 0], g1 * recorded[:, 1]])
    pi_a = g_c.sum(axis=1)
    pi_a_glm = _glm_limit(w, pi_a, mass)
    g_recorded_glm = _glm_limit(w, g_c[:, 1] / pi_a, mass * pi_a)
    return MtreatFactors(w, mass, pi_a, g_c, pi_a_glm, g_recorded_glm)


def mtreat_efficient_sd(factors: MtreatFactors, n: int) -> np.ndarray:
    """The efficient SD at ``n`` of each ``mtreat`` coefficient on the difference scale.

    The influence curve is ``M^{-1} phi(V) (tau(W) - phi' beta + sum_a s_a C_a / g_c,a (Y -
    Qbar_a))`` with ``s_1 = 1`` and ``s_0 = -1``; its variance integrates ``(tau - phi' beta)^2
    + sum_a Qbar_a (1 - Qbar_a) / g_c,a`` against ``phi phi'``.
    """
    w, mass = factors.w, factors.mass
    phi = mtreat_design(w)
    q1, q0 = cp.arm_means("observational", w)
    gram = phi.T @ (mass[:, None] * phi)
    beta = np.linalg.solve(gram, phi.T @ (mass * (q1 - q0)))
    residual = (q1 - q0 - phi @ beta) ** 2
    noise = q1 * (1.0 - q1) / factors.g_c[:, 1] + q0 * (1.0 - q0) / factors.g_c[:, 0]
    middle = phi.T @ ((mass * (residual + noise))[:, None] * phi)
    inverse = np.linalg.inv(gram)
    return np.sqrt(np.diag(inverse @ middle @ inverse) / n)


def complete_case_displacement(factors: MtreatFactors, n: int = 4_000) -> np.ndarray:
    """The ``complete_case`` control's displacement from the truth, in SD at ``n``, by term.

    The complete-case fit with correct learners projects ``tau`` under the law of ``W`` given
    ``Delta_A = 1``, whose density is proportional to ``pi_A(W)``.
    """
    w, mass = factors.w, factors.mass
    q1, q0 = cp.arm_means("observational", w)
    phi = mtreat_design(w)
    full = cp.projection(phi, mass, q1, q0, "difference")
    recorded = cp.projection(phi, mass * factors.pi_a / (mass @ factors.pi_a), q1, q0, "difference")
    return (recorded - full) / mtreat_efficient_sd(factors, n)


def dropped_factor_displacement(factors: MtreatFactors, n: int = 4_000) -> np.ndarray:
    """The ``no_treatment_observation`` control's displacement, in SD at ``n``, by term.

    The one-step's conditional mean is ``tau_w + sum_a s_a (g_c,a / g_used,a)(Qbar_a -
    Qbar_w,a)`` with ``Qbar_w`` the declared wrong regression; ``g_used,a`` is the GLM limit
    of ``g(a | Delta_A = 1, W)`` alone, against the truth's ``g_c,a``.
    """
    w, mass = factors.w, factors.mass
    q1, q0 = cp.arm_means("observational", w)
    w1 = expit(cp.outcome_logit(w, 1) + cp.declared_error("missing", w, 1))
    w0 = expit(cp.outcome_logit(w, 0) + cp.declared_error("missing", w, 0))
    used = np.column_stack([1.0 - factors.g_recorded_glm, factors.g_recorded_glm])
    pseudo = (
        w1
        - w0
        + factors.g_c[:, 1] / used[:, 1] * (q1 - w1)
        - factors.g_c[:, 0] / used[:, 0] * (q0 - w0)
    )
    phi = mtreat_design(w)
    gram = phi.T @ (mass[:, None] * phi)
    shifted = np.linalg.solve(gram, phi.T @ (mass * pseudo))
    truth_beta = np.linalg.solve(gram, phi.T @ (mass * (q1 - q0)))
    return (shifted - truth_beta) / mtreat_efficient_sd(factors, n)


def targeted_dr_displacement(factors: MtreatFactors, n: int = 4_000) -> np.ndarray:
    """The ``mtreat_dr_4k`` targeted point's displacement at the GLM factor limits, in SD.

    The one-step's conditional mean with ``g_c,a`` at the GLM limits (``pi_A`` and ``g(a |
    Delta_A = 1, W)`` both main-terms GLMs), against the declared wrong ``Qbar``.
    """
    w, mass = factors.w, factors.mass
    q1, q0 = cp.arm_means("observational", w)
    w1 = expit(cp.outcome_logit(w, 1) + cp.declared_error("missing", w, 1))
    w0 = expit(cp.outcome_logit(w, 0) + cp.declared_error("missing", w, 0))
    used = factors.g_c_glm
    pseudo = (
        w1
        - w0
        + factors.g_c[:, 1] / used[:, 1] * (q1 - w1)
        - factors.g_c[:, 0] / used[:, 0] * (q0 - w0)
    )
    phi = mtreat_design(w)
    gram = phi.T @ (mass[:, None] * phi)
    shifted = np.linalg.solve(gram, phi.T @ (mass * pseudo))
    truth_beta = np.linalg.solve(gram, phi.T @ (mass * (q1 - q0)))
    return (shifted - truth_beta) / mtreat_efficient_sd(factors, n)


def declared_c() -> float:
    """The declared recorded-treatment coefficient, after the plan's escalation rule.

    ``c = 2.0`` unless its complete-case displacement of the intercept is below
    :data:`COMPLETE_CASE_FLOOR`; then ``2.5``.  Past 2.5 the law returns to plan review.
    """
    for c in (TREATMENT_RECORDING_C, ESCALATED_C):
        if complete_case_displacement(mtreat_factors(c))[0] >= COMPLETE_CASE_FLOOR:
            return c
    raise RuntimeError("the complete-case floor is not reached at c = 2.5: back to plan review")


def strata_efficient_sd(n: int = 4_000, *, nodes: int = 32) -> np.ndarray:
    """The efficient SD at ``n`` of the stratum-1 coefficients on the difference scale.

    A stratum coefficient's curve is the stratum's own projection curve scaled by ``n / n_s``,
    so its variance is the within-stratum variance divided by ``n P(S = 1)``.
    """
    w, mass = cp._grid(nodes)
    phi = cp.design_matrix(w)
    q1 = expit(strata_outcome_logit(w, 1, 1))
    q0 = expit(strata_outcome_logit(w, 0, 1))
    g1 = cp.propensity(w)
    gram = phi.T @ (mass[:, None] * phi)
    beta = np.linalg.solve(gram, phi.T @ (mass * (q1 - q0)))
    integrand = (q1 - q0 - phi @ beta) ** 2 + q1 * (1 - q1) / g1 + q0 * (1 - q0) / (1 - g1)
    inverse = np.linalg.inv(gram)
    middle = phi.T @ ((mass * integrand)[:, None] * phi)
    return np.sqrt(np.diag(inverse @ middle @ inverse) / (n * STRATUM_PROBABILITY))


def stratum_gap(n: int = 4_000) -> np.ndarray:
    """Stratum-1 minus marginal truth on the difference scale, in stratum-1 SD at ``n``."""
    truths = truth("strata", "difference", nodes=32)
    marginal = np.array([truths[f"cate_projection[{t}]"] for t in cp.TERMS])
    stratum = np.array([truths[f"cate_projection[{t}][S=1]"] for t in cp.TERMS])
    return (stratum - marginal) / strata_efficient_sd(n)


def strata_learner_error(*, nodes: int = 24) -> tuple[float, float]:
    """The strata learner's and the shipped spline GLM's population-limit errors.

    Returns the root-mean-square error, under the law of the arm-1 rows, of the arm-1 regression
    of :class:`StrataSpline` on the ``strata`` law and of the shipped spline GLM on the
    observational law: the strata learner must be within the shipped learner's own
    approximation error (plan 5.1).  The mean square, not the largest error, is the reading:
    the largest error of either learner sits at a corner of the cube where neither law puts
    material mass, and the TMLE reads the regression only through ``P_0``-weighted moments.
    """
    w, mass = cp._grid(nodes)

    def limit(learner: Any, features: np.ndarray, p: np.ndarray, weight: np.ndarray):
        x = np.vstack([features, features])
        y = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
        sw = np.concatenate([weight * p, weight * (1.0 - p)])
        last = learner.steps[-1][0] if hasattr(learner, "steps") else None
        fit_kw = {f"{last}__sample_weight" if last else "sample_weight": sw / sw.sum() * len(sw)}
        learner.fit(x, y, **fit_kw)
        return learner.predict_proba(features)[:, 1]

    g1 = cp.propensity(w)
    errors = []
    for s, p_s in ((0, 1.0 - STRATUM_PROBABILITY), (1, STRATUM_PROBABILITY)):
        features = np.column_stack([w, np.full(len(w), float(s))])
        truth_p = expit(strata_outcome_logit(w, 1, s))
        errors.append((features, truth_p, mass * g1 * p_s))
    features = np.vstack([e[0] for e in errors])
    target = np.concatenate([e[1] for e in errors])
    weight = np.concatenate([e[2] for e in errors])
    strata_fit = limit(StrataSpline(max_iter=10_000), features, target, weight)
    shipped_target = expit(cp.outcome_logit(w, 1))
    shipped = limit(cp.spline_glm(), w, shipped_target, mass * g1)
    return (
        float(np.sqrt(np.sum(weight * (strata_fit - target) ** 2) / np.sum(weight))),
        float(np.sqrt(np.sum(mass * g1 * (shipped - shipped_target) ** 2) / np.sum(mass * g1))),
    )


def design_numbers() -> dict[str, Any]:
    """Every design number that needs no pilot, computed rather than copied.

    The calibration ``R`` and the family-wise pass probability need the 3,200-draw pilots per
    arm and are computed by the declaration commit; the rest is here.
    """
    c = declared_c()
    factors = mtreat_factors(c)
    complete = complete_case_displacement(factors)
    dropped = dropped_factor_displacement(factors)
    targeted = targeted_dr_displacement(factors)
    relative = float(np.max(np.abs(factors.g_c_glm - factors.g_c) / factors.g_c))
    power, noncentrality = cp.heterogeneity_power(4_000, 0.10)
    learner_error, shipped_error = strata_learner_error()
    return {
        "c": c,
        "mtreat_sd": tuple(mtreat_efficient_sd(factors, 4_000)),
        "complete_case_sd": tuple(complete),
        "no_treatment_observation_sd": tuple(dropped),
        "mtreat_dr_bias_sd": tuple(targeted),
        "smallest_g_c": float(factors.g_c.min()),
        "factor_relative_error": relative,
        "stratum_gap_sd": tuple(stratum_gap()),
        "strata_learner_error": learner_error,
        "shipped_learner_error": shipped_error,
        "power_noncentrality": noncentrality,
        "power": power,
        "calibration_arms": {arm: len(rows) for arm, rows in CALIBRATION_ARMS.items()},
        "cpu_hours": sum(g.seconds * g.replicates for g in GROUPS) / 3_600.0,
    }
