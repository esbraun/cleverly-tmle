r"""Declaration and driver of the ``score-assessment`` study (X97 plan, section 5.1).

The study measures the TMLE score assessment of
:meth:`cleverly.study.IdentifiedEffect.assess_score`: interval calibration of one external
score's risk difference and calibration coefficients, and of the paired risk difference of two
scores (``compare=``), on the difference and the log-ratio scale, with an estimated and a known
mechanism, on a binary and a continuous law; the size and power of the paired Wald test of 0;
the bias of the TMLE under a declared wrong outcome regression; and controls that must fail.
Everything a run reads is fixed here before it starts:

- the laws, as samplers (:func:`sample`): the ``conditional-projection`` binary laws (the
  ``observational`` law its ``assess`` and ``crr`` cells read, ``trial`` with ``g = 0.5``
  declared known, and ``noweight_law``) and the ``targeting-curves`` continuous law ``H``;
- the scores of each group (:func:`scores`), fixed functions of ``W1`` and ``W3``;
- the exact truths (:func:`truth`), by Gauss--Legendre quadrature over ``U(-1, 1)^3`` with 48
  nodes per axis.  On law ``H`` every integrand is a polynomial of degree at most 4, which the
  rule integrates exactly; on the binary laws :func:`quadrature_error` bounds the error;
- the fit groups (:data:`GROUPS`), each a law, ``n``, ``R``, the scores and the cells its fits
  fill, with the controls that share its fits;
- the design numbers (:func:`design_numbers`): the calibration replication count by the
  pooled predictive rule at the pilot's SE ratios, with the log-ratio risk-difference rows in a
  pool of their own; each positive bias row's second-order bias (:func:`theory_bias`) and the
  bias replication count it sets; each control's SE ratio or standardized displacement and its
  detection probability; the paired Wald test's exact-law power and size pass probabilities;
  and the cost of one run, computed from the influence curves rather than copied;
- the pre-run n rule (:func:`n_rule`), which compared the log-ratio groups' pilot at 2,000 rows
  with the oracle reference (:mod:`tests.studies.score_assessment_oracle`) and moved them to
  4,000;
- the study record (:data:`STUDY`) with its margins, failure bound, publication policy and
  per-group seed streams keyed by label.

The run is ``python -m tests.canonical.score_assessment.regenerate`` (the declared form of
:mod:`tests.canonical.declared_run`); its docstring holds the red-cell reading and the changes
since the first declaration.  The pilot (:mod:`tests.studies.score_assessment_pilot`) ran before
this declaration on its own stream, 2,400 draws of every calibration group.  Its first run
found 12 of 1,600 log-ratio fits whose alternation did not settle, all with a calibration slope
near a point where the log-ratio numerators are collinear; the estimator's reduced-span retry
(``cleverly.effect_curves._alternate_log_ratio``), which also requires every original score
equation to meet ``scaled_var``, was added before the committed pilot, which has no failure.

The single-score log-ratio risk differences carry a second-order term from estimating the best
constant (plan section 2.6): an upward bias of about ``Var(R_n'(c_0)) / (2 R''(c_0))`` and an
SE ratio above 1.  The design prices it (plan rev 2.2): the bias rows at their
:func:`priced_bias`, the calibration rows in their own pool against the oracle reference.

What the controls read, and where they differ from the plan's wording:

- ``unpaired_se`` is the paired row's estimate with the SE ``sqrt(SE_1^2 + SE_2^2)`` of the two
  reported single-score risk differences, which ignores their covariance.  Its cell kind is the
  shared ``inflated_se_control``: the SE-ratio interval must lie above the band.
- ``no_residual_se`` is the paired row's estimate with the SE of the paired curve's plug-in part
  ``(s_1^2 - s_2^2) - 2 (s_1 - s_2) tau_0`` at the fit's initial ``tau_0``, without the residual
  term.  Its cell kind is the shared ``term_dropped_control``: the interval must lie below.
- ``untargeted`` is the plug-in at the fit's initial outcome regression, the declared wrong one.
- the one-step at the initial fit, which the plan publishes ungated, is in the run's
  ``comparison.csv`` beside the TMLE mean of every parameter of every group, with each
  parameter's TMLE coverage, so the DR groups' coverage is published ungated too.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Linear, Modifiers
from cleverly.effect_curves import _one_step_estimates, _pseudo_outcomes
from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as cp
from tests.studies import repeated_sizing as sizing
from tests.studies import targeting_curves as tcs
from tests.studies.evidence import replication
from tests.studies.evidence.property_verdicts import power_cell_pass_probability
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "score-assessment"

#: The declared run's seed and its resampling seed (plan 5.1; no other file holds them).
SEED = 20261297
RESAMPLING_SEED = 20261298

#: The modifiers and the basis of every estimand.  ``assess_score`` reads only the modifiers.
MODIFIERS = Modifiers(["W1", "W3"])
BASIS = Linear(["W1", "W3"])

#: Gauss--Legendre nodes per axis of the truths.
QUADRATURE_NODES = 48

#: The two-sided 95% critical value of every interval of the study.
CRITICAL = float(stats.norm.ppf(0.975))

#: The ``power_cont_2k`` score's ``W3`` coefficient: ``S = tau + c W3``.
POWER_C = 0.13

#: The noise SD of law ``H``.
H_SIGMA = tcs.LAWS["H"].sigma

#: The columns of the two scores in every frame.
FIRST, SECOND = "S", "SB"


# ----------------------------------------------------------------- the laws and the scores


def sample(law: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` rows of ``law``, without the scores.

    Parameters
    ----------
    law : {"observational", "trial", "noweight_law", "H"}
        The law.
    n : int
        Rows.
    rng : numpy.random.Generator
        The replication's generator.

    Returns
    -------
    pandas.DataFrame
        ``Y``, ``A``, ``W1``, ``W2``, ``W3``, and ``p0``, ``p1`` on ``trial``.
    """
    if law == "H":
        frame = tcs.sample(tcs.LAWS["H"], n, rng)
        return frame.loc[:, ["Y", "A", "W1", "W2", "W3"]]
    w = rng.uniform(-1.0, 1.0, (n, 3))
    if law == "trial":
        a = rng.binomial(1, 0.5, n)
    elif law == "noweight_law":
        a = rng.binomial(1, cp.noweight_propensity(w))
    elif law == "observational":
        a = rng.binomial(1, cp.propensity(w))
    else:
        raise KeyError(law)
    y = rng.binomial(1, expit(cp.outcome_logit(w, a))).astype(float)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2]})
    if law == "trial":
        frame["p0"], frame["p1"] = 0.5, 0.5
    return frame


@dataclass(frozen=True)
class ScorePair:
    """The scores of a group: the first, and the second it is compared with, if any.

    Parameters
    ----------
    first : Callable
        ``s_1(W)`` over an ``(m, 3)`` array.
    second : Callable or None
        ``s_2(W)``, or ``None`` for a single-score group.
    first_inputs : tuple of str
        The columns ``s_1`` is computed from.
    second_inputs : tuple of str
        The columns ``s_2`` is computed from.
    """

    first: Callable[[np.ndarray], np.ndarray]
    second: Callable[[np.ndarray], np.ndarray] | None
    first_inputs: tuple[str, ...] = ("W1",)
    second_inputs: tuple[str, ...] = ("W1", "W3")


def _w1(w: np.ndarray) -> np.ndarray:
    return w[..., 0]


def _w3(w: np.ndarray) -> np.ndarray:
    return w[..., 2]


#: The scores of each kind (plan 5.1, table "Groups and cells").
SCORES: Mapping[str, ScorePair] = {
    "binary_paired": ScorePair(
        lambda w: 0.2 + 0.3 * _w1(w),
        lambda w: 0.2 + 0.3 * _w1(w) + 0.15 * np.sin(np.pi * _w3(w)),
    ),
    "cont_paired": ScorePair(
        lambda w: 1.0 + 0.8 * _w1(w), lambda w: 1.0 + 0.8 * _w1(w) + 0.3 * _w3(w)
    ),
    # S - tau = 0.5 W1 and SB - tau = 0.5 W3: equal risks, both curve terms nonzero.
    "cont_equal": ScorePair(lambda w: 1.0 + 1.5 * _w1(w), lambda w: 1.0 + _w1(w) + 0.5 * _w3(w)),
    "cont_power": ScorePair(
        lambda w: 1.0 + _w1(w) + POWER_C * _w3(w),
        lambda w: 1.0 + _w1(w),
        first_inputs=("W1", "W3"),
        second_inputs=("W1",),
    ),
    "crr_single": ScorePair(lambda w: 0.15 + 0.25 * _w1(w), None),
    "crr_paired": ScorePair(
        lambda w: 0.15 + 0.25 * _w1(w),
        lambda w: 0.15 + 0.25 * _w1(w) + 0.2 * np.sin(np.pi * _w3(w)),
    ),
    # The second score loads on W1^2, the shape of the declared outcome error (plan 5.1).
    "dr": ScorePair(
        lambda w: 0.2 + 0.3 * _w1(w),
        lambda w: 0.1 + 0.1 * _w1(w) + 0.4 * _w1(w) ** 2,
        second_inputs=("W1",),
    ),
    "crr_dr": ScorePair(
        lambda w: 0.15 + 0.25 * _w1(w),
        lambda w: 0.05 + 0.1 * _w1(w) + 0.4 * _w1(w) ** 2,
        second_inputs=("W1",),
    ),
}


def h_tau(w: np.ndarray) -> np.ndarray:
    """Law ``H``'s effect ``tau(W) = 1 + W1``."""
    return 1.0 + w[..., 0]


def arm_means(law: str, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(Qbar(1, W), Qbar(0, W))`` of ``law``."""
    if law == "H":
        base = 0.5 * w[..., 0] - 0.5 * w[..., 1] + w[..., 2]
        return base + h_tau(w), base
    return expit(cp.outcome_logit(w, 1)), expit(cp.outcome_logit(w, 0))


def mechanism(law: str, w: np.ndarray) -> np.ndarray:
    """``g(1 | W)`` of ``law``."""
    if law == "trial":
        return np.full(w.shape[:-1], 0.5)
    if law == "noweight_law":
        return cp.noweight_propensity(w)
    if law == "H":
        return tcs.propensity(w.reshape(-1, 3)).reshape(w.shape[:-1])
    return cp.propensity(w)


def outcome_variances(law: str, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(Var(Y | A = 1, W), Var(Y | A = 0, W))`` of ``law``."""
    if law == "H":
        level = np.full(w.shape[:-1], H_SIGMA**2)
        return level, level
    q_t, q_r = arm_means(law, w)
    return q_t * (1.0 - q_t), q_r * (1.0 - q_r)


# --------------------------------------------------------------- the truths


def grid(nodes: int = QUADRATURE_NODES) -> tuple[np.ndarray, np.ndarray]:
    """``U(-1, 1)^3`` quadrature nodes ``(m, 3)`` and their probability weights ``(m,)``."""
    points, weights = leggauss(nodes)
    mesh = np.stack(np.meshgrid(points, points, points, indexing="ij"), axis=-1).reshape(-1, 3)
    mass = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1) / 8.0
    return mesh, mass


def softplus_losses(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""``(\ell_t, \ell_r) = (\log(1 + e^{-\theta}), \log(1 + e^\theta))``."""
    return np.logaddexp(0.0, -theta), np.logaddexp(0.0, theta)


def parameter_labels(paired: bool) -> tuple[str, ...]:
    """The cell label of every parameter a group reports, in report order."""
    single = ("risk_difference", "calibration_intercept", "calibration_slope")
    labels = [f"{FIRST}_{name}" for name in single]
    if paired:
        labels += [f"{SECOND}_{name}" for name in single] + ["paired"]
    return tuple(labels)


def _risks(
    scale: str, q_t: np.ndarray, q_r: np.ndarray, s: np.ndarray, mass: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """``(risk R(s), risk difference against the best constant, calibration (b0, b1))``."""
    phi = np.column_stack([np.ones(len(s)), s])
    if scale == "difference":
        tau = q_t - q_r
        risk = float(mass @ (s**2 - 2.0 * s * tau))
        ate = float(mass @ tau)
        return risk, risk + ate**2, cp.projection(phi, mass, q_t, q_r, "difference")
    lt, lr = softplus_losses(s)
    risk = float(mass @ (q_t * lt + q_r * lr))
    c = math.log(float(mass @ q_t) / float(mass @ q_r))
    ct, cr = softplus_losses(np.full(len(s), c))
    risk_c = float(mass @ (q_t * ct + q_r * cr))
    return risk, risk - risk_c, cp.projection(phi, mass, q_t, q_r, "log_ratio")


def _parameters(
    scale: str, q_t: np.ndarray, q_r: np.ndarray, pair: ScorePair, w: np.ndarray, mass: np.ndarray
) -> dict[str, float]:
    """Every parameter of the scores ``pair`` at the arm regressions ``(q_t, q_r)``."""
    out: dict[str, float] = {}
    risks = []
    for column, function in ((FIRST, pair.first), (SECOND, pair.second)):
        if function is None:
            continue
        risk, difference, beta = _risks(scale, q_t, q_r, function(w), mass)
        risks.append(risk)
        out[f"{column}_risk_difference"] = difference
        out[f"{column}_calibration_intercept"] = float(beta[0])
        out[f"{column}_calibration_slope"] = float(beta[1])
    if len(risks) == 2:
        out["paired"] = risks[0] - risks[1]
    return out


def truth(law: str, scale: str, kind: str, *, nodes: int = QUADRATURE_NODES) -> dict[str, float]:
    """The exact value of every parameter of a group, keyed by its cell label.

    Parameters
    ----------
    law : str
        The law.
    scale : {"difference", "log_ratio"}
        The scale the scores predict.
    kind : str
        A key of :data:`SCORES`.
    nodes : int
        Quadrature nodes per axis.

    Returns
    -------
    dict of str to float
        The truth of each label of :func:`parameter_labels`.
    """
    w, mass = grid(nodes)
    q_t, q_r = arm_means(law, w)
    return _parameters(scale, q_t, q_r, SCORES[kind], w, mass)


def quadrature_error(law: str, scale: str, kind: str) -> float:
    """The largest change in a truth when the nodes per axis drop from 48 to 32."""
    fine = truth(law, scale, kind)
    coarse = truth(law, scale, kind, nodes=32)
    return max(abs(fine[name] - coarse[name]) for name in fine)


# ---------------------------------------------------------- the fit groups


CALIBRATION = "interval_calibration"
BIAS = "bias"
TYPE_I = "type_i_error"
POWER = "power"

#: The positive arm of a calibration cell, and the shared kinds of its two SE controls.
POSITIVE = "correctly_specified"
UNPAIRED_SE = "inflated_se_control"
NO_RESIDUAL_SE = "term_dropped_control"

#: The cells of the paired Wald test of 0.
SIZE_CELL = "paired_wald__equal_risks"
POWER_CELL = "paired_wald__c_0.13"


@dataclass(frozen=True)
class Group:
    """One declared fit group: a law, a size, its replications, and the cells its fits fill.

    Every cell of a group reads the same draws and the same fit, so its controls share their
    replicates with its positive cells.  A group's draws come from the stream
    :func:`tests.studies.evidence.seeds.stream_seed` keys by its label.

    Parameters
    ----------
    label : str
        The group key, which the seed stream and the cell names read.
    law : str
        The law :func:`sample` draws.
    n : int
        Rows per replication.
    replicates : int
        Replications in the declared run.
    scale : str
        The scale the scores predict.
    kind : str
        A key of :data:`SCORES`.
    learner : str
        ``"spline"`` (binary laws), ``"linear_spline"`` (law ``H``) or ``"declared"`` (the
        declared wrong outcome regression of ``noweight_law``).
    calibration : tuple of str
        The labels whose estimates are ``interval_calibration`` positives.
    bias : tuple of str
        The labels whose TMLE estimate is a positive ``bias`` row, and whose plug-in at the
        initial fit is an ``untargeted`` control row.
    controls : tuple of str
        The SE controls on the paired row: ``unpaired_se`` and ``no_residual_se``.
    wald : str or None
        ``"size"`` or ``"power"`` when the paired Wald test of 0 fills a cell.
    seconds : float
        Single-thread seconds per replication, from the pilot (:data:`PILOT_CSV`) where the
        group is piloted, for the cost.
    """

    label: str
    law: str
    n: int
    replicates: int
    scale: str
    kind: str
    learner: str
    calibration: tuple[str, ...] = ()
    bias: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    wald: str | None = None
    seconds: float = 0.4

    @property
    def paired(self) -> bool:
        """Whether the group compares two scores."""
        return SCORES[self.kind].second is not None


#: The calibration replication count: the pooled predictive rule at the pilot's SE ratios
#: (:func:`calibration_replicates`), which ``tests/unit/test_score_assessment_design.py``
#: recomputes.  The log-ratio risk-difference rows pool apart (plan rev 2.2, B2); at their
#: 4,000-row pilot (mean SE ratio 1.032) 15,200 reads a family-wise pass probability of 0.952
#: and 0.978 for each of those rows.
R_CAL = 15_200

#: Replications of the power group (plan 5.1).
R_POWER = 1_000

#: Replications of the two bias groups: the smallest multiple of 400 at which every positive
#: bias row passes with probability at least 0.99 at the larger of its theory bias and its
#: oracle bias (:func:`bias_replicates`, plan rev 2.2, B1, and the re-review's nit R2).  The
#: log-ratio rows' second-order bias is about 0.12 SD at ``n = 4,000`` (oracle 0.128 and
#: 0.121), where the plan's 1,000 reads 0.94 and 0.95 and 1,600 reads 0.9895 at the oracle.
R_BIAS = 2_000

PAIRED = parameter_labels(True)
SINGLE = parameter_labels(False)
DR_ROWS = ("paired", f"{FIRST}_risk_difference", f"{SECOND}_risk_difference")

#: The groups of plan 5.1, in run order.  ``seconds`` are single-thread seconds per fit: the
#: pilot's mean where the group is piloted (:data:`PILOT_CSV`), and for the power and bias
#: groups the mean of 35 timed fits on the pilot stream before the declaration.
GROUPS: tuple[Group, ...] = (
    Group(
        "paired_2k",
        "observational",
        2_000,
        R_CAL,
        "difference",
        "binary_paired",
        "spline",
        calibration=PAIRED,
        controls=("unpaired_se",),
        seconds=0.143,
    ),
    Group(
        "paired_trial_2k",
        "trial",
        2_000,
        R_CAL,
        "difference",
        "binary_paired",
        "spline",
        calibration=PAIRED,
        seconds=0.125,
    ),
    Group(
        "paired_cont_2k",
        "H",
        2_000,
        R_CAL,
        "difference",
        "cont_paired",
        "linear_spline",
        calibration=PAIRED,
        controls=("unpaired_se",),
        seconds=0.103,
    ),
    Group(
        "equal_cont_2k",
        "H",
        2_000,
        R_CAL,
        "difference",
        "cont_equal",
        "linear_spline",
        calibration=("paired",),
        controls=("no_residual_se",),
        wald="size",
        seconds=0.109,
    ),
    Group(
        "power_cont_2k",
        "H",
        2_000,
        R_POWER,
        "difference",
        "cont_power",
        "linear_spline",
        wald="power",
        seconds=0.105,
    ),
    Group(
        "single_crr_4k",
        "observational",
        4_000,
        R_CAL,
        "log_ratio",
        "crr_single",
        "spline",
        calibration=SINGLE,
        seconds=0.201,
    ),
    Group(
        "paired_crr_4k",
        "observational",
        4_000,
        R_CAL,
        "log_ratio",
        "crr_paired",
        "spline",
        calibration=PAIRED,
        seconds=0.241,
    ),
    Group(
        "paired_dr_4k",
        "noweight_law",
        4_000,
        R_BIAS,
        "difference",
        "dr",
        "declared",
        bias=DR_ROWS,
        seconds=0.06,
    ),
    Group(
        "paired_crr_dr_4k",
        "noweight_law",
        4_000,
        R_BIAS,
        "log_ratio",
        "crr_dr",
        "declared",
        bias=DR_ROWS,
        seconds=0.099,
    ),
)

GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}

#: The two log-ratio calibration groups as first declared, at ``n = 2,000``.  The pre-run n rule
#: of plan rev 2.2 (:func:`n_rule`) read their 2,400-draw pilot against the oracle reference and
#: moved both to ``n = 4,000`` (``single_crr_4k``, ``paired_crr_4k``).  They are not run; their
#: pilot and oracle rows stay committed as the record of that decision.
N_RULE_GROUPS: tuple[Group, ...] = tuple(
    replace(GROUP_BY_LABEL[label.replace("_2k", "_4k")], label=label, n=2_000)
    for label in ("single_crr_2k", "paired_crr_2k")
)

#: Every group a pilot or the oracle reference draws, the run's and the n rule's.
PILOT_GROUP_BY_LABEL: Mapping[str, Group] = {
    group.label: group for group in (*GROUPS, *N_RULE_GROUPS)
}

#: The groups the pilot fits: every group with calibration rows, then the n rule's.
PILOT_GROUPS: tuple[str, ...] = tuple(
    group.label for group in (*GROUPS, *N_RULE_GROUPS) if group.calibration
)

#: What each control computes from its group's fit, and the rule it must fail.
CONTROL_CLAIMS: Mapping[str, str] = {
    "unpaired_se": "the paired estimate with the SE sqrt(SE_1^2 + SE_2^2) of the two single "
    "risk differences, ignoring their covariance: its SE-ratio interval must lie above the band",
    "no_residual_se": "the paired estimate with the SE of its curve's plug-in part at the "
    "initial tau, without the residual term: its SE-ratio interval must lie below the band",
    "untargeted": "the plug-in at the declared wrong outcome regression: its bias must lie "
    "wholly outside +-0.25 of its own SD",
}


def calibration_cell(group: Group, label: str) -> str:
    """The ``interval_calibration`` positive cell of one parameter of one group."""
    return f"{group.label}_{label}__{POSITIVE}"


def control_cell(group: Group, control: str) -> str:
    """The calibration cell of an SE control on ``group``'s paired row."""
    kind = UNPAIRED_SE if control == "unpaired_se" else NO_RESIDUAL_SE
    return f"{group.label}_paired__{kind}"


def bias_cell(group: Group, label: str, *, control: bool = False) -> str:
    """A ``bias`` cell: the TMLE row, or the ``untargeted`` control's, of one parameter."""
    prefix = group.label.replace("paired_", "untargeted_", 1) if control else group.label
    return f"{prefix}_{label}"


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells ``group``'s fits fill, by family."""
    cells: dict[str, list[str]] = {CALIBRATION: [], BIAS: [], TYPE_I: [], POWER: []}
    cells[CALIBRATION] += [calibration_cell(group, label) for label in group.calibration]
    cells[CALIBRATION] += [control_cell(group, control) for control in group.controls]
    cells[BIAS] += [bias_cell(group, label) for label in group.bias]
    cells[BIAS] += [bias_cell(group, label, control=True) for label in group.bias]
    if group.wald == "size":
        cells[TYPE_I].append(SIZE_CELL)
    if group.wald == "power":
        cells[POWER].append(POWER_CELL)
    return {family: tuple(names) for family, names in cells.items() if names}


def property_cells() -> dict[str, tuple[str, ...]]:
    """Every declared property cell, by family, in group order."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for family, names in group_cells(group).items():
            out.setdefault(family, []).extend(names)
    return {family: tuple(names) for family, names in out.items()}


PROPERTY_CELLS = property_cells()


# -------------------------------------------------------------- the fits


def learners(group: Group) -> dict[str, Any]:
    """The nuisance learners and fit settings of ``group``."""
    glm = LogisticRegression(max_iter=2_000)
    if group.learner == "declared":
        return {"outcome_learner": cp.DeclaredOutcome("noweight"), "treatment_learner": glm}
    if group.learner == "linear_spline":
        return {
            "outcome_learner": tcs.spline(False),
            "treatment_learner": glm,
            "outcome_by_arm": True,
            "q_bounds": tcs.Q_BOUNDS,
        }
    return {"outcome_learner": cp.spline_glm(), "treatment_learner": glm, "outcome_by_arm": True}


def design(group: Group) -> PointTreatment:
    """The study design of ``group``'s law: ``trial`` declares its mechanism known."""
    roles: dict[str, Any] = {}
    if group.law == "trial":
        roles["treatment_probabilities"] = {0: "p0", 1: "p1"}
    return PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), **roles)


def with_scores(group: Group, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the group's score columns."""
    w = frame.loc[:, ["W1", "W2", "W3"]].to_numpy(dtype=float)
    pair = SCORES[group.kind]
    out = frame.copy()
    out[FIRST] = pair.first(w)
    if pair.second is not None:
        out[SECOND] = pair.second(w)
    return out


def _label(score: str, name: str) -> str:
    if " - " in score:
        return "paired"
    if name == "risk_difference":
        return f"{score}_risk_difference"
    if name == "calibration[(intercept)]":
        return f"{score}_calibration_intercept"
    return f"{score}_calibration_slope"


def _labelled(values: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """The one-step helper's nested output, keyed by cell label."""
    out = {}
    for score, block in values.items():
        for name, value in block.items():
            out["paired" if name.startswith("risk_difference[") else _label(score, name)] = value
    return out


def fit_group(group: Group, frame: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    """Fit one replication of ``group`` and return what each of its cells reads.

    The fit is :meth:`~cleverly.study.IdentifiedEffect.assess_score`; its private twin returns
    the ATE fit it targeted as well, so the one-step and the plug-in at the initial fit read
    that fit's own nuisances without a second fit.

    Parameters
    ----------
    group : Group
        The group.
    frame : pandas.DataFrame
        One replication drawn by :func:`sample`, without the scores.
    random_state : int
        The fold seed.

    Returns
    -------
    dict
        ``"estimates"``: the reported estimate of each label; ``"one_step"`` and
        ``"plug_in"``: each label's one-step and plug-in at the initial fit;
        ``"unpaired_se"`` and ``"no_residual_se"``: the controls' standard errors of the
        paired row, or ``None`` on a single-score group; ``"reduced"``: the directions the
        log-ratio fluctuation dropped (the span record's ``reduced``), 0 on a full span.
    """
    frame = with_scores(group, frame)
    pair = SCORES[group.kind]
    study = CausalStudy(frame, design=design(group))
    effect = study.identify(ConditionalContrast(MODIFIERS, scale=group.scale, model=BASIS))  # type: ignore[arg-type]
    settings = {**learners(group), "random_state": random_state}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assessment, fit, data = effect._assess_score_with_fit(
            FIRST,
            computed_from=list(pair.first_inputs),
            score_kind="external",
            compare=SECOND if group.paired else None,
            compare_computed_from=list(pair.second_inputs) if group.paired else None,
            scale=None,
            alpha=0.05,
            **settings,
        )
    estimates = {_label(score, name): estimate for score, name, estimate in assessment.rows()}
    g_bounds = (float(fit.config.g_bounds[0]), float(fit.config.g_bounds[1]))
    q_t, q_r, d_t, d_r = _pseudo_outcomes(
        fit.nuisance,
        data,
        (1.0, 0.0),
        g_bounds=g_bounds,
        missingness_bound=float(fit.config.missingness_bound),
    )
    arrays = {FIRST: frame[FIRST].to_numpy(dtype=float)}
    if group.paired:
        arrays[SECOND] = frame[SECOND].to_numpy(dtype=float)
    weights = np.ones(len(frame))
    one_step = _labelled(
        _one_step_estimates(q_t, q_r, d_t, d_r, arrays, scale=group.scale, weights=weights)
    )
    plug_in = _labelled(
        _one_step_estimates(q_t, q_r, q_t, q_r, arrays, scale=group.scale, weights=weights)
    )
    out: dict[str, Any] = {
        "estimates": estimates,
        "one_step": one_step,
        "plug_in": plug_in,
        "unpaired_se": None,
        "no_residual_se": None,
        "reduced": int(assessment.span["reduced"]),
    }
    if group.paired:
        first = estimates[f"{FIRST}_risk_difference"]
        second = estimates[f"{SECOND}_risk_difference"]
        out["unpaired_se"] = math.hypot(reported_inference(first)[0], reported_inference(second)[0])
        s1, s2 = arrays[FIRST], arrays[SECOND]
        if group.scale == "difference":
            part = (s1**2 - s2**2) - 2.0 * (s1 - s2) * (q_t - q_r)
        else:
            t1, r1 = softplus_losses(s1)
            t2, r2 = softplus_losses(s2)
            part = q_t * (t1 - t2) + q_r * (r1 - r2)
        out["no_residual_se"] = float(np.std(part, ddof=1) / math.sqrt(len(part)))
    return out


def wald_rejects(estimate: Any) -> int:
    """Whether the reported 95% interval of ``estimate`` excludes 0."""
    _, low, high = reported_inference(estimate)
    return int(not low <= 0.0 <= high)


# ------------------------------------------------------------ the primary phase

#: The primary phase is ``paired_2k``'s paired row: the paired comparison with an estimated
#: mechanism on the binary law (plan 5.1).
PRIMARY_GROUP = "paired_2k"
PRIMARY_REPLICATES = 2_000
PRIMARY_N = GROUP_BY_LABEL[PRIMARY_GROUP].n
PRIMARY_NAME = f"risk_difference[{FIRST} - {SECOND}]"

SCENARIOS = {PRIMARY_GROUP: (PRIMARY_NAME,)}

STUDY = StudyRecord(
    name="Score assessment by TMLE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "score_assessment",
    document="docs/technical-reference/method-evidence/score-assessment.md",
    anchor="score-assessment-by-tmle",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-score-assessment-tmle",
    reference=None,
    failure_bound={"cleverly-score-assessment-tmle": 0.01},
    modules=(
        "tests/studies/score_assessment.py",
        "tests/studies/score_assessment_properties.py",
        "tests/studies/conditional_projection.py",
        "tests/studies/targeting_curves.py",
        "tests/studies/repeated_sizing.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.score_assessment",
    properties_module="tests.studies.score_assessment_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset({BIAS}),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    property_artifacts=("comparison.csv",),
)


#: Each group's fit configuration, as the manifest records it.
CONFIGURATION = {
    group.label: (
        f"{group.law} law, n={group.n}, R={group.replicates}, learner {group.learner}, "
        f"scale {group.scale}, scores {group.kind}"
    )
    for group in GROUPS
}


@functools.cache
def _group_truth(law: str, scale: str, kind: str) -> tuple[tuple[str, float], ...]:
    """:func:`truth`, computed once per process: every replicate of a group shares it."""
    return tuple(truth(law, scale, kind).items())


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed, with the group's truths."""
    group = PILOT_GROUP_BY_LABEL[group_label]
    frame = sample(group.law, n, np.random.default_rng(seed))
    return frame, dict(_group_truth(group.law, group.scale, group.kind))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary row of the paired estimate, with the one-step at the initial fit."""
    group = GROUP_BY_LABEL[scenario]
    fitted = fit_group(group, frame, random_state=replicate)
    estimate = fitted["estimates"]["paired"]
    reference = float(truths["paired"])
    std_error, low, high = reported_inference(estimate)
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": len(frame),
            "estimand": PRIMARY_NAME,
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": float(fitted["one_step"]["paired"]),
        }
    ]


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


# ---------------------------------------------------------- the design numbers


def _curves(
    group: Group,
    w: np.ndarray,
    mass: np.ndarray,
    arms: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """The influence curves' two parts at the law's arm regressions, or at ``arms``.

    Each curve is ``a(W) + b_t(W) e_t + b_r(W) e_r`` with ``e_a`` the arm-``a`` residual term
    ``1{A = a} (Y - Qbar_a) / g(a | W)``, whose conditional mean is 0 and conditional variance
    is ``Var(Y | A = a, W) / g(a | W)``.  Returns ``"<label>:a"``, ``"<label>:t"`` and
    ``"<label>:r"`` for every risk label and the paired row.
    """
    q_t, q_r = arm_means(group.law, w) if arms is None else arms
    pair = SCORES[group.kind]
    s1 = pair.first(w)
    s2 = pair.second(w) if pair.second is not None else None
    out: dict[str, np.ndarray] = {}
    if group.scale == "difference":
        tau = q_t - q_r
        ate = float(mass @ tau)
        for column, s in ((FIRST, s1), (SECOND, s2)):
            if s is None:
                continue
            a = s**2 - 2.0 * s * tau + 2.0 * ate * tau
            out[f"{column}_risk_difference:a"] = a - float(mass @ a)
            out[f"{column}_risk_difference:t"] = 2.0 * (ate - s)
            out[f"{column}_risk_difference:r"] = -2.0 * (ate - s)
        if s2 is not None:
            a = (s1**2 - s2**2) - 2.0 * (s1 - s2) * tau
            out["paired:a"] = a - float(mass @ a)
            out["paired:t"] = -2.0 * (s1 - s2)
            out["paired:r"] = 2.0 * (s1 - s2)
        return out
    c = math.log(float(mass @ q_t) / float(mass @ q_r))
    ct, cr = softplus_losses(np.full(len(s1), c))
    for column, s in ((FIRST, s1), (SECOND, s2)):
        if s is None:
            continue
        lt, lr = softplus_losses(s)
        a = q_t * (lt - ct) + q_r * (lr - cr)
        out[f"{column}_risk_difference:a"] = a - float(mass @ a)
        out[f"{column}_risk_difference:t"] = lt - ct
        out[f"{column}_risk_difference:r"] = lr - cr
    if s2 is not None:
        t1, r1 = softplus_losses(s1)
        t2, r2 = softplus_losses(s2)
        a = q_t * (t1 - t2) + q_r * (r1 - r2)
        out["paired:a"] = a - float(mass @ a)
        out["paired:t"] = t1 - t2
        out["paired:r"] = r1 - r2
    return out


def efficient_sd(group: Group, label: str, *, nodes: int = 32) -> float:
    """The efficient SD at ``group.n`` rows of the TMLE of ``label``'s risk parameter."""
    w, mass = grid(nodes)
    curves = _curves(group, w, mass)
    g = mechanism(group.law, w)
    v_t, v_r = outcome_variances(group.law, w)
    a, b_t, b_r = curves[f"{label}:a"], curves[f"{label}:t"], curves[f"{label}:r"]
    variance = float(mass @ (a**2 + b_t**2 * v_t / g + b_r**2 * v_r / (1.0 - g)))
    return math.sqrt(variance / group.n)


def se_control_ratio(group: Group, control: str, *, nodes: int = 32) -> float:
    """The limit of an SE control's ratio of reported SE to the paired row's true SD."""
    sd = efficient_sd(group, "paired", nodes=nodes)
    if control == "unpaired_se":
        return (
            math.hypot(
                efficient_sd(group, f"{FIRST}_risk_difference", nodes=nodes),
                efficient_sd(group, f"{SECOND}_risk_difference", nodes=nodes),
            )
            / sd
        )
    w, mass = grid(nodes)
    plug_in = _curves(group, w, mass)["paired:a"]
    return math.sqrt(float(mass @ plug_in**2) / group.n) / sd


def se_control_detection(ratio: float, replicates: int) -> float:
    """P(an SE control's 99% SE-ratio interval leaves the band on its side) at ``ratio``.

    Below the band this is :func:`tests.studies.repeated_sizing.calibration_control_probability`;
    above it is the mirror computation with the same normal approximation.
    """
    margins = Margins()
    if ratio < 1.0:
        return sizing.calibration_control_probability(margins, replicates, ratio)
    spread = ratio / math.sqrt(2.0 * (replicates - 1))
    z = float(stats.norm.ppf(1.0 - (1.0 - margins.confidence_level) / 2.0))
    return float(stats.norm.sf((margins.calibration_se_ratio[1] + z * spread - ratio) / spread))


def untargeted_moments(group: Group, label: str, *, nodes: int = 32) -> tuple[float, float]:
    """``(displacement, SD)`` of the plug-in at the declared wrong outcome regression.

    The declared regression learns nothing, so the plug-in is a sample mean of a fixed function
    of ``W``; the risk difference also holds the squared mean, or the constant, of the arm
    regressions, whose delta-method term is in the curve.  Its limit minus the truth is the
    displacement, and the curve's SD at ``group.n`` rows is the plug-in's own SD.
    """
    w, mass = grid(nodes)
    m_t = expit(cp.outcome_logit(w, 1) + cp.declared_error("noweight", w, 1))
    m_r = expit(cp.outcome_logit(w, 0) + cp.declared_error("noweight", w, 0))
    limit = _parameters(group.scale, m_t, m_r, SCORES[group.kind], w, mass)[label]
    true = truth(group.law, group.scale, group.kind, nodes=nodes)[label]
    curve = _curves(group, w, mass, arms=(m_t, m_r))[f"{label}:a"]
    return limit - true, math.sqrt(float(mass @ curve**2) / group.n)


def design_limits(group: Group, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The limits of ``group``'s fitted arm regressions: the declared wrong ones, or the law's."""
    if group.learner == "declared":
        return (
            expit(cp.outcome_logit(w, 1) + cp.declared_error("noweight", w, 1)),
            expit(cp.outcome_logit(w, 0) + cp.declared_error("noweight", w, 0)),
        )
    return arm_means(group.law, w)


def limit_variance(
    group: Group,
    a: np.ndarray,
    b_t: np.ndarray | float,
    b_r: np.ndarray | float,
    w: np.ndarray,
    mass: np.ndarray,
) -> float:
    r"""The variance of the curve ``a + b_t (D_t - m_t) + b_r (D_r - m_r)`` at the design's limits.

    ``D_a`` is the AIPW pseudo-outcome at the limit ``m_a`` of the fitted arm regression
    (:func:`design_limits`) and the true mechanism.  Its conditional mean is ``q_a`` and, with
    ``e_a = q_a - m_a``, ``Var(D_t | W) = v_t / g + e_t^2 (1 - g) / g``, ``Var(D_r | W) =
    v_r / (1 - g) + e_r^2 g / (1 - g)`` and ``Cov(D_t, D_r | W) = -e_t e_r``.  Where the
    mechanism is fitted (every law but ``trial``), the logistic regression on ``(1, W)`` is
    correct, so the curve loses its projection on that model's score, whose covariance with
    the curve is ``E[x ((1 - g) b_t e_t - g b_r e_r)]``.  With ``m = q`` the projection is 0 and
    this is the efficient variance.
    """
    m_t, m_r = design_limits(group, w)
    q_t, q_r = arm_means(group.law, w)
    g = mechanism(group.law, w)
    v_t, v_r = outcome_variances(group.law, w)
    e_t, e_r = q_t - m_t, q_r - m_r
    mean = a + b_t * e_t + b_r * e_r
    mean = mean - float(mass @ mean)
    conditional = (
        b_t**2 * (v_t / g + e_t**2 * (1.0 - g) / g)
        + b_r**2 * (v_r / (1.0 - g) + e_r**2 * g / (1.0 - g))
        - 2.0 * b_t * b_r * e_t * e_r
    )
    variance = float(mass @ (mean**2 + conditional))
    if group.law != "trial":
        x = np.column_stack([np.ones(len(mass)), w])
        information = (x * (mass * g * (1.0 - g))[:, None]).T @ x
        covariance = x.T @ (mass * ((1.0 - g) * b_t * e_t - g * b_r * e_r))
        variance -= float(covariance @ np.linalg.solve(information, covariance))
    return variance


def limit_sd(group: Group, label: str, *, nodes: int = 32) -> float:
    """The SD at ``group.n`` rows of ``label``'s estimate at the design's nuisance limits."""
    w, mass = grid(nodes)
    curves = _curves(group, w, mass, arms=design_limits(group, w))
    a, b_t, b_r = curves[f"{label}:a"], curves[f"{label}:t"], curves[f"{label}:r"]
    return math.sqrt(limit_variance(group, a, b_t, b_r, w, mass) / group.n)


def theory_bias(group: Group, label: str, *, nodes: int = 32) -> float:
    r"""The second-order bias of ``label``'s estimate, in SDs (plan rev 2.2, section 2.6).

    A single-score risk difference is :math:`R(s) - R(c)` at the estimated best constant
    :math:`c_n`, which minimises the estimated risk, so it is biased upward by about
    :math:`\operatorname{Var}(R_n'(c_0)) / (2 R''(c_0))`.  On the log-ratio scale
    :math:`R_n'(c) = -\operatorname{expit}(-c) P_n D_t + \operatorname{expit}(c) P_n D_r` and
    :math:`R''(c) = (E Y_t + E Y_r)\operatorname{expit}(c)\operatorname{expit}(-c)`; on the
    difference scale the term is :math:`(\widehat{\mathrm{ATE}} - \mathrm{ATE})^2`, whose mean
    is :math:`\operatorname{Var}(\widehat{\mathrm{ATE}})`.  Both variances are read at the
    design's nuisance limits (:func:`limit_variance`).  The paired row has no such term: the
    constant cancels.  The SD is :func:`limit_sd`.
    """
    if label == "paired":
        return 0.0
    w, mass = grid(nodes)
    m_t, m_r = design_limits(group, w)
    q_t, q_r = arm_means(group.law, w)
    if group.scale == "log_ratio":
        mean_t, mean_r = float(mass @ q_t), float(mass @ q_r)
        c = math.log(mean_t / mean_r)
        b_t, b_r = -float(expit(-c)), float(expit(c))
        curvature = (mean_t + mean_r) * float(expit(c) * expit(-c))
        variance = limit_variance(group, b_t * m_t + b_r * m_r, b_t, b_r, w, mass) / group.n
        bias = variance / (2.0 * curvature)
    else:
        bias = limit_variance(group, m_t - m_r, 1.0, -1.0, w, mass) / group.n
    return bias / limit_sd(group, label, nodes=nodes)


def bias_pass_probability(standardized: float, replicates: int, *, control: bool) -> float:
    """P(the 99% interval of a standardized bias is inside, or wholly outside, +-0.25)."""
    return cp.bias_pass_probability(standardized, replicates, control=control)


def wald_power(group: Group, *, alpha: float = 0.05, nodes: int = 32) -> float:
    """The exact-law power of the two-sided level-``alpha`` Wald test of a zero paired risk."""
    psi = truth(group.law, group.scale, group.kind)["paired"]
    sd = efficient_sd(group, "paired", nodes=nodes)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(stats.norm.cdf(psi / sd - z) + stats.norm.cdf(-psi / sd - z))


def type_i_pass_probability(size: float, replicates: int) -> float:
    """P(the shipped type-I rule passes): the 99% CP upper end at most ``alpha + 0.05``."""
    return cp.type_i_pass_probability(size, replicates)


#: The SE-ratio pilot's committed summary (:mod:`tests.studies.score_assessment_pilot`): a
#: pilot on its own seed stream, read by :func:`design_numbers`, not a result of the run.
PILOT_CSV = ROOT / "tests" / "studies" / "score_assessment_pilot.csv"

#: The oracle reference (:mod:`tests.studies.score_assessment_oracle`): the one-step of the
#: single-score risk differences at the law's own nuisances, and of the DR rows at the declared
#: nuisance limits, on its own seed stream.  The declared finite-sample prediction of rev 2.2.
ORACLE_CSV = ROOT / "tests" / "studies" / "score_assessment_oracle.csv"

#: The pilot's correlation of every two estimates of a group, which the sizing rule reads as
#: the correlation of their SE ratios' noise (squared).
PILOT_CORRELATIONS_CSV = ROOT / "tests" / "studies" / "score_assessment_pilot_correlations.csv"


def pilot_rows() -> dict[tuple[str, str], dict[str, Any]]:
    """The pilot's summary row of each ``(group, label)``."""
    frame = pd.read_csv(PILOT_CSV)
    return {(str(row["group"]), str(row["label"])): row for row in frame.to_dict("records")}


def pilot_correlations() -> dict[tuple[str, str], float]:
    """The pilot correlation of every two calibration cells' estimates, by cell label."""
    frame = pd.read_csv(PILOT_CORRELATIONS_CSV)
    out: dict[tuple[str, str], float] = {}
    for row in frame.to_dict("records"):
        a = f"{row['group']}_{row['a']}"
        b = f"{row['group']}_{row['b']}"
        out[(a, b)] = out[(b, a)] = float(row["correlation"])
    return out


def label_correlations(labels: Sequence[str]) -> np.ndarray:
    """The correlation of the estimates of the cells ``labels``: the pilot's within a group, 0
    across groups (their draws are independent)."""
    known = pilot_correlations()
    matrix = np.eye(len(labels))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                matrix[i, j] = known.get((a, b), 0.0)
    return matrix


def label_correlations_at(labels: Mapping[str, str]) -> Callable[[Sequence[str]], np.ndarray]:
    """:func:`label_correlations` with each group of ``labels`` read from its pilot source."""

    def correlations(names: Sequence[str]) -> np.ndarray:
        renamed = []
        for name in names:
            for target, source in labels.items():
                if name.startswith(f"{target}_"):
                    name = source + name[len(target) :]
            renamed.append(name)
        return label_correlations(renamed)

    return correlations


def calibration_parameters() -> list[tuple[Group, str, str]]:
    """Each positive calibration cell with its group and its label."""
    return [
        (group, calibration_cell(group, label), label)
        for group in GROUPS
        for label in group.calibration
    ]


def pilot_ratios() -> dict[str, float]:
    """The pilot SE ratio of each positive calibration cell, by cell name."""
    pilot = pilot_rows()
    return {
        cell: float(pilot[(group.label, label)]["se_ratio"])
        for group, cell, label in calibration_parameters()
    }


#: The single-score log-ratio risk-difference rows, which carry the best-constant term (plan
#: rev 2.2, B2): their sizing pool is their own.
LOG_RATIO_RD: tuple[tuple[str, str], ...] = (
    ("single_crr_4k", f"{FIRST}_risk_difference"),
    ("paired_crr_4k", f"{FIRST}_risk_difference"),
    ("paired_crr_4k", f"{SECOND}_risk_difference"),
)

#: The sizing pool of the rows in :data:`LOG_RATIO_RD`.
LOG_RATIO_POOL = "log_ratio_risk_difference"

#: The least pass probability of each row in :data:`LOG_RATIO_RD` that the calibration rule
#: requires beside the family-wise 0.95 (plan rev 2.2, B2).
LOG_RATIO_ROW_PASS = 0.90


def sizing_ratios() -> dict[str, float]:
    """:func:`pilot_ratios` by sizing name: ``<group>_<label>__<pool>``.

    The rows of :data:`LOG_RATIO_RD` sit in the pool :data:`LOG_RATIO_POOL` and every other
    positive calibration cell in the pool of its kind, so the rule pools each set apart.
    """
    return _sizing_ratios_at({})


def pilot_replicates(labels: Mapping[str, str] | None = None) -> int:
    """The smallest count of good pilot fits behind any calibration ratio.

    ``labels`` maps a group to the group whose pilot rows stand for it, as :func:`n_rule` reads
    the log-ratio groups' 2,000-row pilot.
    """
    pilot = pilot_rows()
    source = labels or {}
    return min(
        int(pilot[(source.get(g.label, g.label), label)]["replicates"])
        for g, _, label in calibration_parameters()
    )


def _sizing_groups() -> tuple[list[sizing.PilotGroup], float, list[str]]:
    ratios = sizing_ratios()
    groups = sizing.calibration_groups(ratios, pilot_replicates(), label_correlations)
    names = [name for pool in sizing.calibration_arms(ratios).values() for name in pool]
    return groups, sizing.heterogeneity_bound(groups), names


def _row_pass(
    groups: list[sizing.PilotGroup], tau: float, names: list[str], replicates: int
) -> dict[str, float]:
    cell = sizing._calibration_cell(Margins(), replicates)

    def passes(values: np.ndarray) -> np.ndarray:
        return np.ones_like(values)

    out = {}
    for group_label, label in LOG_RATIO_RD:
        target = f"{group_label}_{label}__{LOG_RATIO_POOL}"
        cells = [cell if name == target else passes for name in names]
        out[target.split("__")[0]] = sizing.pooled_pass_probability(groups, tau, cells)
    return out


def log_ratio_row_pass(replicates: int) -> dict[str, float]:
    """Each :data:`LOG_RATIO_RD` row's pass probability at ``replicates`` under the pooled reading.

    The row's true SE ratio is its pool's pilot mean, with that mean's noise, plus the
    heterogeneity spread, as :func:`tests.studies.repeated_sizing.pooled_pass_probability`
    reads every cell; the other cells count as passing.
    """
    groups, tau, names = _sizing_groups()
    return _row_pass(groups, tau, names, replicates)


def calibration_replicates() -> int:
    """The declared rule for :data:`R_CAL`: the pooled predictive rule at the pilot.

    The smallest multiple of 400 at which every gated calibration cell passes with family-wise
    probability at least 0.95 (:func:`tests.studies.repeated_sizing.calibration_replicates`)
    and each row of :data:`LOG_RATIO_RD` passes with probability at least
    :data:`LOG_RATIO_ROW_PASS` (:func:`log_ratio_row_pass`).  Each cell's true SE ratio is
    drawn about its pool's pilot mean, with the pilot's own noise and the heterogeneity bound
    between cells.  The log-ratio risk-difference rows form one pool and every other positive
    cell the other (plan rev 2.2, B2).
    """
    replicates = sizing.calibration_replicates(
        Margins(), sizing_ratios(), pilot_replicates(), label_correlations
    )
    while min(log_ratio_row_pass(replicates).values()) < LOG_RATIO_ROW_PASS:
        replicates += sizing.BUDGET_STEP
        if replicates > sizing.MAXIMUM_REPLICATES:
            raise ValueError("no budget up to 50,000 replications meets the log-ratio row rule")
    return replicates


def priced_bias(group: Group, label: str) -> float:
    """The standardized bias a bias row is priced at: the larger of its :func:`theory_bias`
    and the oracle reference's bias (:data:`ORACLE_CSV`), where the oracle has the row."""
    oracle = oracle_rows().get((group.label, label))
    theory = theory_bias(group, label)
    return theory if oracle is None else max(theory, float(oracle["standardized_bias"]))


def bias_replicates() -> int:
    """The declared rule for :data:`R_BIAS`: the smallest multiple of 400 at which every
    positive bias row passes with probability at least 0.99 at its :func:`priced_bias`."""
    biases = [priced_bias(group, label) for group in GROUPS for label in group.bias]
    replicates = sizing.BUDGET_STEP
    while min(bias_pass_probability(b, replicates, control=False) for b in biases) < 0.99:
        replicates += sizing.BUDGET_STEP
    return replicates


def oracle_rows() -> dict[tuple[str, str], dict[str, Any]]:
    """The oracle reference's row of each ``(group, label)`` (:data:`ORACLE_CSV`)."""
    frame = pd.read_csv(ORACLE_CSV)
    return {(str(row["group"]), str(row["label"])): row for row in frame.to_dict("records")}


#: The confidence of the n rule's comparison of the pilot with the oracle reference.
N_RULE_Z = float(stats.norm.ppf(0.995))


def oracle_agreement(single: str, paired: str) -> dict[str, float]:
    """The pilot's pooled ``S`` risk-difference SE ratio against the oracle reference's.

    The pilot rows of the groups ``single`` and ``paired`` are pooled by inverse variance and
    compared with the oracle row of ``paired`` (the same law, score and ``n``).  ``excess`` is
    the pilot minus the reference and ``interval`` the 99% half-width of their difference.
    """
    pilot = pilot_rows()
    oracle = oracle_rows()[(paired, f"{FIRST}_risk_difference")]
    rows = [pilot[(label, f"{FIRST}_risk_difference")] for label in (single, paired)]
    precision = [1.0 / float(row["se_ratio_se"]) ** 2 for row in rows]
    pooled = sum(p * float(row["se_ratio"]) for p, row in zip(precision, rows, strict=True))
    pooled /= sum(precision)
    pooled_se = 1.0 / math.sqrt(sum(precision))
    return {
        "pilot": pooled,
        "pilot_se": pooled_se,
        "oracle": float(oracle["se_ratio"]),
        "oracle_se": float(oracle["se_ratio_se"]),
        "excess": pooled - float(oracle["se_ratio"]),
        "interval": N_RULE_Z * math.hypot(pooled_se, float(oracle["se_ratio_se"])),
    }


def _sizing_ratios_at(labels: Mapping[str, str]) -> dict[str, float]:
    """:func:`sizing_ratios` with the log-ratio groups read from the pilot rows of ``labels``."""
    pilot = pilot_rows()
    out = {}
    for group, _, label in calibration_parameters():
        source = labels.get(group.label, group.label)
        pool = LOG_RATIO_POOL if (group.label, label) in LOG_RATIO_RD else POSITIVE
        out[f"{group.label}_{label}__{pool}"] = float(pilot[(source, label)]["se_ratio"])
    return out


def n_rule() -> dict[str, Any]:
    """The pre-run n rule of plan rev 2.2 (B2), read at the first declaration's ``n = 2,000``.

    - ``"stop"``: the pilot's pooled ``S`` risk-difference SE ratio exceeds the oracle
      reference by more than the 99% interval of their difference
      (:func:`oracle_agreement`).  The excess is not the theory term, so the work stops for a
      diagnosis; n does not move to absorb it.
    - ``"move"``: the pilot agrees, but no ``R`` up to the sizing cap of 50,000 meets the
      calibration rule (:func:`calibration_replicates`: family-wise 0.95 and each
      :data:`LOG_RATIO_RD` row :data:`LOG_RATIO_ROW_PASS`) with the log-ratio groups at their
      2,000-row pilot: both groups move to ``n = 4,000``.
    - ``"stay"``: otherwise both stay at 2,000.

    Returns the agreement, the family-wise and row pass probabilities at the cap, and the
    decision.
    """
    agreement = oracle_agreement("single_crr_2k", "paired_crr_2k")
    labels = {group.label.replace("_2k", "_4k"): group.label for group in N_RULE_GROUPS}
    ratios = _sizing_ratios_at(labels)
    groups = sizing.calibration_groups(
        ratios, pilot_replicates(labels), label_correlations_at(labels)
    )
    tau = sizing.heterogeneity_bound(groups)
    cap = sizing.MAXIMUM_REPLICATES
    family = sizing.calibration_family_pass_probability(Margins(), cap, groups, tau)
    names = [name for pool in sizing.calibration_arms(ratios).values() for name in pool]
    rows = _row_pass(groups, tau, names, cap)
    if agreement["excess"] > agreement["interval"]:
        decision = "stop"
    elif family < sizing.FAMILY_WISE_PASS or min(rows.values()) < LOG_RATIO_ROW_PASS:
        decision = "move"
    else:
        decision = "stay"
    return {**agreement, "family_at_cap": family, "rows_at_cap": rows, "decision": decision}


def design_numbers() -> dict[str, Any]:
    """Every computed number the declaration fixes before the run.

    Returns
    -------
    dict
        ``calibration_replicates`` (the rule's count) and ``calibration_family`` (its
        family-wise pass probability at :data:`R_CAL`); ``per_row_pass`` at ratio 1;
        ``controls``: each SE control's ratio and detection probability, and each
        ``untargeted`` row's standardized displacement and detection probability;
        ``log_ratio_row_pass``: each log-ratio risk-difference row's pass probability at
        :data:`R_CAL`; ``bias_replicates`` (the rule's count), ``theory_bias``: each positive
        bias row's second-order bias in SDs, and ``bias_pass``: its pass probability at the larger
        of that and the oracle bias (:func:`priced_bias`);
        ``power``,
        ``power_pass``, ``size_pass``; ``truths``: the paired truths; ``cost``.
    """
    margins = Margins()
    groups, tau, _ = _sizing_groups()
    family = sizing.calibration_family_pass_probability(margins, R_CAL, groups, tau)
    controls: dict[str, dict[str, float]] = {}
    bias_pass: dict[str, float] = {}
    biases: dict[str, float] = {}
    for group in GROUPS:
        for control in group.controls:
            ratio = se_control_ratio(group, control)
            controls[control_cell(group, control)] = {
                "ratio": ratio,
                "detection": se_control_detection(ratio, group.replicates),
            }
        for label in group.bias:
            shift, sd = untargeted_moments(group, label)
            controls[bias_cell(group, label, control=True)] = {
                "standardized": shift / sd,
                "detection": bias_pass_probability(shift / sd, group.replicates, control=True),
            }
            biases[bias_cell(group, label)] = theory_bias(group, label)
            bias_pass[bias_cell(group, label)] = bias_pass_probability(
                priced_bias(group, label), group.replicates, control=False
            )
    power_group = GROUP_BY_LABEL["power_cont_2k"]
    power = wald_power(power_group)
    cpu = sum(group.replicates * group.seconds for group in GROUPS)
    cpu += PRIMARY_REPLICATES * GROUP_BY_LABEL[PRIMARY_GROUP].seconds
    return {
        "calibration_replicates": calibration_replicates(),
        "calibration_family": family,
        "log_ratio_row_pass": log_ratio_row_pass(R_CAL),
        "heterogeneity_bound": tau,
        "bias_replicates": bias_replicates(),
        "theory_bias": biases,
        "per_row_pass": float(sizing.calibration_pass(margins, R_CAL, np.array(1.0))),
        "controls": controls,
        "bias_pass": bias_pass,
        "power": power,
        "power_pass": power_cell_pass_probability(power, power_group.replicates),
        "size_pass": type_i_pass_probability(0.07, GROUP_BY_LABEL["equal_cont_2k"].replicates),
        "truths": {
            group.label: truth(group.law, group.scale, group.kind)["paired"]
            for group in GROUPS
            if group.paired
        },
        "cost": {"cpu_hours": cpu / 3600.0, "wall_hours_4": cpu / 3600.0 / 4.0},
    }
