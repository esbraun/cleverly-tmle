r"""Declaration and driver of the ``ep-crr`` study (P1 plan, section 5.3).

The study measures the EP learner of the conditional log relative risk
(:class:`cleverly.EPMethod` on ``ConditionalContrast(..., scale="log_ratio")``, ``model=None``)
against the oracle learner :math:`\theta_{n0}` of EPL Theorems 3 to 5: the minimiser over the
same sieve of the one-step log-CRR risk at the true nuisances,

.. math::

   \frac{1}{n}\sum_i \{D_{1,0} + D_{0,0}\}\log(1 + e^{\theta(W_i)}) - D_{1,0}\,\theta(W_i),
   \qquad D_{a,0} = \mu_0(a, W) + \frac{1\{A = a\}}{\pi_0(a \mid W)}\{Y - \mu_0(a, W)\},

solved by Newton with step halving from the plug-in fit at the true means.  The DR loss is not
convex in general (EPL Sec 3.2), so each replicate records whether the Hessian at the solution
is positive definite (:func:`oracle_fit`); every learner's oracle is the same fit, so it is
checked once per replicate.  The oracle-only pilot (:mod:`tests.studies.ep_crr_hessian_pilot`)
found no non-PD Hessian in 24,000 checks (:func:`hessian_rate`).  A replicate whose Hessian is not is excluded from
every oracle comparison of its group and counts against :data:`FAILURE_BOUND`; the count is
the ``oracle_hessian`` diagnostic row.  Every gate reads the paired mean of
:math:`\log \mathrm{ISE}(\text{estimator}) - \log \mathrm{ISE}(\theta_{n0})` within a replicate,
the ISE of the log CRR by Monte Carlo on :data:`ISE_POINTS` fixed draws of ``W``.

**Laws.**  hte3 at ``eac64673``, ``generate_data_RR.R`` ``sim.RR(n, hard, positivity)``, read
for this declaration.  It settles the summation scope the plan's App B.2 reading left open
(P1-sources section 6), and it differs from the plan's declared reading; the code is used:

- ``logit mu0(0, w) = -1 + 0.3 (sum w_k + sum sin(4 w_k))`` (the plan declared
  ``-1 + 0.3 sum w_k + sum sin(4 w_k) / 3``);
- the simple log CRR is ``0.1 (-1 + sum w_k)`` and the complex one
  ``0.1 (-1 + sum w_k + sum sin(4 w_k))``, under both overlaps;
- ``positivity=TRUE`` is the limited overlap, ``logit pi = sum w``, and ``positivity=FALSE``
  the moderate one, ``logit pi = sum w / 3``, as in the CATE generator.

hte3 clips ``mu(1, w) = mu0 exp(theta)`` at 1; the declared domain check
(:func:`max_treated_mean`) shows it stays below 0.8, so nothing is clipped.

**Learners.**  The classification ensemble of plan 5.0 for ``mu`` per arm (``clip`` 0.005 to
0.995) and for ``pi`` (0.01 to 0.99; the EP bounds ``g`` at the resolved ``g_bounds``, 0.0229
at 1k, so the ensemble's own wider clip changes nothing).  On ``d3_complex`` with limited
overlap, two declared wrong outcome learners beside it, each with the same ensemble ``pi``:
the plan's per-arm ``DummyClassifier(strategy="prior")``, which ignores ``W``, and
:class:`OffsetOutcome`, the true regression with the treated arm's logit off by
``2 {(sum W)^2 - 1}``, whose cross terms lie outside the additive sieve.  5 outer folds.

**What the rescue cells see** (design review B1 of PR-b).  The log-ratio debiasing features
span a per-arm logistic model on the sieve, and this law's ``logit mu0`` is almost additive.
With the intercept-only learner, a debiasing that drops the ``1 / pi`` weight is a consistent
per-arm sieve regression, so its rescue cell shows that the debiasing step corrects an
intercept-only ``mu`` at this law, and does not see the ``1 / pi`` weight (a 6-replicate probe
read the mutated EP at -0.16 against the margin 0.405).  The offset learner's error lies
outside the sieve, so only the ``1 / pi`` weight removes it: the offset rescue cell is the
study-level witness of that weight, and its pilot (:mod:`tests.studies.ep_crr_offset_pilot`)
reads the unit-weight mutation far above the margin (:func:`mutation_detected`): 31 replicates
at each end read EP at 0.25 (SD 0.34) at 4,000, its T-learner at 1.35 (0.35) and the mutation
at 1.72 (0.34), against the margin 0.405.  The unit
witness of the weight is ``test_the_inverse_propensity_weight[log_ratio]`` in
``tests/unit/test_ep_exact.py``.

**Estimators** per replicate: EP at the schedule :func:`tests.studies.ep_cate.schedule`
(3 and 5 cosines per modifier at 1k and 4k); the T-learner on each learner (the plug-in risk on
the same basis); the oracle.  No DR-learner (EPL Sec 3.2), no IPW E-learner.  One reporting
group fits EP with the size chosen by cross-validation.

**Gates** (plan 5.3, one-sided 99% Student bounds):

- Thm 5 oracle comparison, every law, ensemble, at n = 4,000: upper bound below ``log 1.5``;
- no growth from 1k to 4k, ensemble: upper bound below ``log 1.25``;
- DR rescue, intercept learner and offset learner, ``d3_complex_limited``, at 4,000: upper
  bound below ``log 1.5``, and for the intercept learner its growth bound below ``log 1.25``
  (the offset learner's growth row is reported, :data:`REPORTED_GROWTH`);
- control that must fail, per rescue learner: its T-learner, lower bound at 4,000 above
  ``log 1.5``.

The run is ``python -m tests.canonical.ep_crr.regenerate``; its docstring holds the red-cell
reading.  ``R`` per law is fixed by :func:`design_numbers` from the committed pilot
(:mod:`tests.studies.ep_crr_pilot`).  The primary phase is the projection TMLE of
``conditional-projection``'s ``obs_4k`` on this study's own seed, as in ``ep-cate``: the EP
curve carries no interval for the harness's primary rows.  The comparator, hte3
``Lrnr_crr_EP``, is declared pending (:data:`COMPARATOR`).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Modifiers, Sieve
from cleverly.effect_curves import _pseudo_outcomes
from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as projection
from tests.studies import ep_cate
from tests.studies.ep_cate import (
    CONFIDENCE,
    DIAGNOSTIC,
    GROWTH_MARGIN,
    NUISANCE,
    ORACLE,
    OUTER_FOLDS,
    SIEVE_MAX,
    TARGET_PASS,
    THM5_MARGIN,
    Comparator,
    gate_pass,
    growth_pass,
    ise,
    schedule,
)
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate, stream_seed

SLUG = "ep-crr"

#: The primary seed and the resampling seed (distinct from every other study's).
SEED = 20261123
RESAMPLING_SEED = 20261124

ISE_POINTS = 20_000
DESCRIPTIVE_REPLICATES = 100

#: The learners of the rescue law's DR-rescue cells: the plan's intercept-only learner, and
#: the out-of-sieve offset of :class:`OffsetOutcome` (design review B1 of PR-b).
RESCUE_LEARNERS = ("misspecified", "offset")

#: The rescue learners whose 1k-to-4k growth row is reported, not gated (plan amendment rev 2.5):
#: the offset pilot reads the EP ratio rising, 0.16 at 1k to 0.25 at 4k, and with ``mu`` wrong
#: outside the sieve the remainder is first order in the error of ``pi``, so no theorem bounds
#: the ratio's growth; the growth gate's assurance is 0.78 at 600 replicates.
REPORTED_GROWTH = ("offset",)

#: The rescue margin of plan 5.3: EP stays at the oracle's level with the intercept learner.
RESCUE_MARGIN = math.log(1.5)

R_CANDIDATES = (300, 450, 600)

#: The share of a cell's requested replications the run may lose, oracle Hessians included.
FAILURE_BOUND = 0.01

#: The family of the oracle's per-replicate Hessian check (diagnostic).
HESSIAN = "oracle_hessian"


# ---------------------------------------------------------------- the laws


@dataclass(frozen=True)
class Law:
    """One log-CRR law.

    Parameters
    ----------
    label : str
        The law's name in every cell.
    crr : {"simple", "complex"}
        The log CRR.
    overlap : {"moderate", "limited"}
        The propensity.
    sizes : tuple of int
        The sample sizes.
    rescue : bool
        Whether the law carries the intercept-learner cells.
    """

    label: str
    crr: str
    overlap: str
    sizes: tuple[int, ...] = (1_000, 4_000)
    rescue: bool = False

    columns: tuple[str, ...] = ("W1", "W2", "W3")


LAWS: tuple[Law, ...] = (
    Law("d3_simple_moderate", "simple", "moderate"),
    Law("d3_simple_limited", "simple", "limited"),
    Law("d3_complex_moderate", "complex", "moderate"),
    Law("d3_complex_limited", "complex", "limited", rescue=True),
)
LAW_BY_LABEL: Mapping[str, Law] = {law.label: law for law in LAWS}


def covariates(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, (n, 3))


def propensity(law: Law, w: np.ndarray) -> np.ndarray:
    total = w.sum(axis=1)
    return expit(total if law.overlap == "limited" else total / 3.0)


def baseline(w: np.ndarray) -> np.ndarray:
    """``P(Y = 1 | A = 0, W)``, hte3's code."""
    return expit(-1.0 + 0.3 * (w.sum(axis=1) + np.sin(4 * w).sum(axis=1)))


def log_crr(law: Law, w: np.ndarray) -> np.ndarray:
    """The log CRR ``log mu(1, W) - log mu(0, W)``."""
    if law.crr == "complex":
        return 0.1 * (-1.0 + w.sum(axis=1) + np.sin(4 * w).sum(axis=1))
    return 0.1 * (-1.0 + w.sum(axis=1))


def treated_mean(law: Law, w: np.ndarray) -> np.ndarray:
    return baseline(w) * np.exp(log_crr(law, w))


def max_treated_mean(law: Law, grid: int = 81) -> float:
    """The largest ``mu(1, w)`` on a grid over ``[-1, 1]^3`` (the declared no-clip check)."""
    axis = np.linspace(-1.0, 1.0, grid)
    w = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return float(np.max(treated_mean(law, w)))


def sample(law: Law, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """One replication of ``law`` at ``n`` rows."""
    w = covariates(n, rng)
    a = rng.binomial(1, propensity(law, w))
    mean = np.where(a == 1, treated_mean(law, w), baseline(w))
    frame = pd.DataFrame(w, columns=list(law.columns))
    frame["A"] = a
    frame["Y"] = rng.binomial(1, mean)
    return frame


def ise_points(law: Law) -> np.ndarray:
    return covariates(
        ISE_POINTS, np.random.default_rng(stream_seed(STUDY, "ise_points", law.label))
    )


# ---------------------------------------------------------------- the learners and fits


#: The out-of-sieve outcome error of the ``offset`` learner, on the treated arm's logit.
OFFSET_SCALE = 2.0


def offset_error(w: np.ndarray) -> np.ndarray:
    """``2 {(sum_k w_k)^2 - 1}``: centred under ``U(-1, 1)^3``, mostly outside any additive model.

    Its cross terms ``4 sum_{j<k} w_j w_k`` carry most of its variance and lie outside the
    additive sieve; its square terms lie inside.
    """
    return OFFSET_SCALE * (w.sum(axis=1) ** 2 - 1.0)


class OffsetOutcome(BaseEstimator, ClassifierMixin):
    """A declared outcome regression that learns nothing: the truth, wrong on the treated arm.

    It predicts ``expit(logit mu(a, W) + 2 a {(sum W)^2 - 1})``.  The error is a function of
    ``W`` whose cross terms the additive sieve cannot absorb, so the debiasing must weight by
    ``1 / pi`` to remove it (design review B1 of PR-b: the intercept-only learner's error is
    almost in the sieve's span, so its rescue cell does not see that weight).  The pooled fit
    hands it ``[A, W1, W2, W3]``.

    Parameters
    ----------
    law : str
        The law's label.
    """

    def __init__(self, law: str = "d3_complex_limited") -> None:
        self.law = law

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OffsetOutcome:
        design = np.asarray(X, dtype=float)
        if design.shape[1] != 4 or not np.isin(design[:, 0], (0.0, 1.0)).all():
            raise ValueError("OffsetOutcome reads [A, W1, W2, W3] with a binary A")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        a, w = design[:, 0], design[:, 1:]
        law = LAW_BY_LABEL[self.law]
        treated = logit(treated_mean(law, w)) + offset_error(w)
        p = expit(np.where(a == 1.0, treated, logit(baseline(w))))
        return np.column_stack([1.0 - p, p])


def learner_settings(learner: str, random_state: int) -> dict[str, Any]:
    by_arm = True
    if learner == "ensemble":
        outcome: Any = projection.ensemble((0.005, 0.995))
    elif learner == "misspecified":
        outcome = DummyClassifier(strategy="prior")
    else:
        outcome, by_arm = OffsetOutcome(), False
    return {
        "outcome_learner": outcome,
        "treatment_learner": projection.ensemble((0.01, 0.99)),
        "outcome_by_arm": by_arm,
        "n_folds": OUTER_FOLDS,
        "random_state": random_state,
    }


def modifiers(law: Law) -> Modifiers:
    return Modifiers(list(law.columns), domains=dict.fromkeys(law.columns, (-1.0, 1.0)))


def fit_ep(law: Law, frame: pd.DataFrame, learner: str, random_state: int, *, size: Any) -> Any:
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=law.columns)
    )
    effect = study.identify(ConditionalContrast(modifiers(law), scale="log_ratio"))
    sieve = Sieve.default(modifiers(law), max_size=SIEVE_MAX, size=size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate("ep", sieve=sieve, **learner_settings(learner, random_state))


def _risk(design: np.ndarray, d_t: np.ndarray, d_r: np.ndarray, beta: np.ndarray) -> float:
    eta = design @ beta
    return float(np.mean((d_t + d_r) * np.logaddexp(0.0, eta) - d_t * eta))


def oracle_fit(
    design: np.ndarray, d_t: np.ndarray, d_r: np.ndarray, start: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Minimise the one-step log-CRR risk from ``start``; whether the Hessian is PD there.

    Newton steps with step halving, at most 100, stopping at a stationary point.
    """
    beta = np.asarray(start, dtype=float).copy()
    current = _risk(design, d_t, d_r, beta)
    total = d_t + d_r
    for _ in range(100):
        p = expit(design @ beta)
        gradient = design.T @ (total * p - d_t) / len(p)
        hessian = design.T @ ((total * p * (1.0 - p))[:, None] * design) / len(p)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = gradient
        size = 1.0
        while size > 1e-8:
            candidate = beta - size * step
            value = _risk(design, d_t, d_r, candidate)
            if value <= current:
                break
            size /= 2.0
        else:
            break
        beta, change = candidate, current - value
        current = value
        if np.max(np.abs(size * step)) < 1e-12 or change < 1e-15:
            break
    p = expit(design @ beta)
    hessian = design.T @ ((total * p * (1.0 - p))[:, None] * design) / len(p)
    return beta, bool(np.min(np.linalg.eigvalsh(hessian)) > 0.0)


def oracle_pseudo(law: Law, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(D_{1,0}, D_{0,0})`` at the true nuisances."""
    w = frame.loc[:, list(law.columns)].to_numpy()
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    pi = propensity(law, w)
    mu1, mu0 = treated_mean(law, w), baseline(w)
    return mu1 + a / pi * (y - mu1), mu0 + (1.0 - a) / (1.0 - pi) * (y - mu0)


def curves(law: Law, frame: pd.DataFrame, fit: Any) -> dict[str, Any]:
    """Every estimator's ISE at one EP fit's nuisances and basis, and the nuisance errors."""
    points = ise_points(law)
    truth = log_crr(law, points)
    point_frame = pd.DataFrame(points, columns=list(law.columns))
    rows = frame.loc[:, list(law.columns)]
    design = fit.model.design(rows)
    at_points = fit.model.design(point_frame)
    nuisance_fit = fit.fit.nuisance_fit
    nuisance = nuisance_fit.nuisance
    q_t, q_r, _, _ = _pseudo_outcomes(
        nuisance,
        fit.fit.data,
        tuple(nuisance.effect_modification.pair),
        g_bounds=tuple(nuisance_fit.config.g_bounds),
        missingness_bound=float(nuisance_fit.config.missingness_bound),
    )
    w = rows.to_numpy()
    out: dict[str, Any] = {}
    out["ep"], out["ep_mc_se"] = ise(fit.predict(point_frame), truth)
    t_beta, _ = oracle_fit(design, q_t, q_r, np.zeros(design.shape[1]))
    out["t_learner"], _ = ise(at_points @ t_beta, truth)
    plug_in, _ = oracle_fit(design, treated_mean(law, w), baseline(w), np.zeros(design.shape[1]))
    d_t, d_r = oracle_pseudo(law, frame)
    beta, positive = oracle_fit(design, d_t, d_r, plug_in)
    out["oracle"], out["oracle_mc_se"] = ise(at_points @ beta, truth)
    out["oracle_pd"] = float(positive)
    errors = np.concatenate([q_t - treated_mean(law, w), q_r - baseline(w)])
    out["mu_l2"] = float(np.sqrt(np.mean(errors**2)))
    g = np.asarray(nuisance.propensity.values, dtype=float)[:, nuisance.propensity.column_for(1.0)]
    out["pi_l2"] = float(np.sqrt(np.mean((g - propensity(law, w)) ** 2)))
    out["size"] = int(fit.sieve.chosen)
    return out


def fit_replicate(law: Law, n: int, frame: pd.DataFrame, replicate: int) -> dict[str, Any]:
    out = {"ensemble": curves(law, frame, fit_ep(law, frame, "ensemble", replicate, size=schedule))}
    if law.rescue:
        for learner in RESCUE_LEARNERS:
            wrong = fit_ep(law, frame, learner, replicate, size=schedule)
            out[learner] = curves(law, frame, wrong)
    return out


def fit_cv_replicate(law: Law, frame: pd.DataFrame, replicate: int) -> dict[str, Any]:
    return {"ensemble": curves(law, frame, fit_ep(law, frame, "ensemble", replicate, size="cv"))}


# ---------------------------------------------------------------- groups and cells


@dataclass(frozen=True)
class Group:
    label: str
    law: str
    n: int
    replicates: int
    kind: str = "gated"  # "gated" or "cv"


#: ``R`` per law from :func:`design_numbers` (pinned by the design test).
REPLICATES: Mapping[str, int] = {
    "d3_simple_moderate": 300,
    "d3_simple_limited": 300,
    "d3_complex_moderate": 300,
    "d3_complex_limited": 300,
}

CV_GROUP = "d3_complex_moderate_1k_cv"

GROUPS: tuple[Group, ...] = (
    *(
        Group(ep_cate.group_label(law.label, n), law.label, n, REPLICATES[law.label])
        for law in LAWS
        for n in law.sizes
    ),
    Group(CV_GROUP, "d3_complex_moderate", 1_000, DESCRIPTIVE_REPLICATES, "cv"),
)
GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}


group_label = ep_cate.group_label


def gated_n(law: Law) -> tuple[int, int]:
    return law.sizes[0], law.sizes[-1]


def oracle_cells(group: Group) -> dict[str, str]:
    """The ``oracle_comparison`` cells of a group, each with its role."""
    if group.kind == "cv":
        return {f"{group.label}__ep": DIAGNOSTIC}
    law = LAW_BY_LABEL[group.law]
    largest = group.n == law.sizes[-1]
    cells = {
        f"{group.label}__ep": "positive" if largest else DIAGNOSTIC,
        f"{group.label}__t_learner": DIAGNOSTIC,
    }
    if law.rescue:
        for learner in RESCUE_LEARNERS:
            cells[f"{group.label}__ep_{learner}"] = "positive" if largest else DIAGNOSTIC
            cells[f"{group.label}__t_{learner}"] = "control" if largest else DIAGNOSTIC
    return cells


def learners_of(group: Group) -> tuple[str, ...]:
    rescue = group.kind == "gated" and LAW_BY_LABEL[group.law].rescue
    return ("ensemble", *RESCUE_LEARNERS) if rescue else ("ensemble",)


def hessian_cells(group: Group) -> tuple[str, ...]:
    """One check per group: every learner's oracle is the same fit on the same design."""
    return (f"{group.label}__oracle_pd",)


def nuisance_cells(group: Group) -> tuple[str, ...]:
    if group.kind != "gated":
        return ()
    return tuple(
        f"{group.label}__{learner}_{name}"
        for learner in learners_of(group)
        for name in ("mu_l2", "pi_l2", "product")
    )


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    cells = {ORACLE: tuple(oracle_cells(group)), HESSIAN: hessian_cells(group)}
    if nuisance_cells(group):
        cells[NUISANCE] = nuisance_cells(group)
    return cells


def property_cells() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for family, cells in group_cells(group).items():
            out.setdefault(family, []).extend(cells)
    return {family: tuple(cells) for family, cells in out.items()}


PROPERTY_CELLS = property_cells()


# ---------------------------------------------------------------- the study record

PRIMARY_GROUP = ep_cate.PRIMARY_GROUP
PRIMARY_REPLICATES = ep_cate.PRIMARY_REPLICATES
PRIMARY_N = ep_cate.PRIMARY_N
SCENARIOS = ep_cate.SCENARIOS
IMPLEMENTATION = "cleverly-ep-crr"

#: hte3's EP learner of the log CRR on the same draws (pending, as in ``ep-cate``).
COMPARATOR = Comparator(
    reference="hte3",
    commit=ep_cate.COMPARATOR.commit,
    learner=(
        "Lrnr_crr_EP(screen_basis_with_lasso = FALSE), additive cosine sieve of interaction "
        "order 1 with basisN = k(n) per modifier"
    ),
    kind="different estimator",
    runner="hte3_crr_runner.R",
    status="pending",
    reason=(
        "hte3 runs on the owner's machine (P1 plan 5.3); this session has no R. The runner "
        "reads DRAWS.csv and POINTS.csv, which no code writes yet: the draws regenerate from "
        "the run's stream seeds, so the declared run saves nothing extra, and the comparator "
        "run needs an exporter first"
    ),
)

STUDY = StudyRecord(
    name="EP learner of the log CRR",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "ep_crr",
    document="docs/technical-reference/method-evidence/ep-crr.md",
    anchor="ep-learner-of-the-log-crr",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/studies/ep_crr.py",
        "tests/studies/ep_crr_properties.py",
        "tests/studies/ep_cate.py",
        "tests/studies/ep_cate_properties.py",
        "tests/studies/conditional_projection.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.ep_crr",
    properties_module="tests.studies.ep_crr_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset(),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    group.label: (
        f"{group.law} log-CRR law, n={group.n}, R={group.replicates}, {group.kind}, "
        f"{OUTER_FOLDS} outer folds"
    )
    for group in GROUPS
}


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if group_label == PRIMARY_GROUP:
        return ep_cate.draw_from_seed(group_label, n, seed)
    group = GROUP_BY_LABEL[group_label]
    return sample(LAW_BY_LABEL[group.law], n, np.random.default_rng(seed)), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw under this study's seed, as :func:`draw_and_fit` draws it."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """The shared primary phase under this study's seed (:func:`ep_cate.primary_phase`)."""
    return ep_cate.primary_phase(__name__, replicates=replicates, n=n, n_jobs=n_jobs)


# ---------------------------------------------------------------- the design numbers

PILOT_CSV = ROOT / "tests" / "studies" / "ep_crr_pilot.csv"


#: The second pilot, ``d3_simple_limited`` at 100 replicates per end on its own stream
#: (design review B2 of PR-b).
GROWTH_CSV = ROOT / "tests" / "studies" / "ep_crr_growth_pilot.csv"


def _rows(path: Any) -> dict[tuple[str, str], dict[str, float]]:
    frame = pd.read_csv(path)
    return {
        (str(row["group"]), str(row["statistic"])): {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "replicates": int(row["replicates"]),
        }
        for _, row in frame.iterrows()
    }


def pooled(first: Mapping[str, float], second: Mapping[str, float]) -> dict[str, float]:
    """Two pilots of one statistic as one sample: the pooled mean and SD."""
    m, k = int(first["replicates"]), int(second["replicates"])
    mean = (m * first["mean"] + k * second["mean"]) / (m + k)
    squares = (
        (m - 1) * first["sd"] ** 2
        + (k - 1) * second["sd"] ** 2
        + m * (first["mean"] - mean) ** 2
        + k * (second["mean"] - mean) ** 2
    )
    return {"mean": mean, "sd": math.sqrt(squares / (m + k - 1)), "replicates": m + k}


#: The offset learner's pilot (:mod:`tests.studies.ep_crr_offset_pilot`, design review B1).
OFFSET_CSV = ROOT / "tests" / "studies" / "ep_crr_offset_pilot.csv"


def pilot_rows() -> dict[tuple[str, str], dict[str, float]]:
    """Every pilot's statistics, pooled where two pilots measured the same one."""
    rows = _rows(PILOT_CSV)
    for path in (GROWTH_CSV, OFFSET_CSV):
        for key, row in _rows(path).items():
            rows[key] = pooled(rows[key], row) if key in rows else row
    return rows


def mutation_detected(
    pilot: Mapping[tuple[str, str], Mapping[str, float]], replicates: int
) -> float:
    """P(the offset rescue gate goes red under the unit-weight mutation) at ``R``.

    The assurance of the mutated EP's rescue gate failing: the cell sees the ``1 / pi``
    weight only if this is near 1 (design review B1 of PR-b).
    """
    row = pilot[(group_label("d3_complex_limited", 4_000), "ep_offset_unit_weight")]
    seed = stream_seed(STUDY, "assurance", "d3_complex_limited", "unit_weight")
    return 1.0 - ep_cate.gate_assurance(row, replicates, RESCUE_MARGIN, below=True, seed=seed)


def law_gates(
    law: Law, pilot: Mapping[tuple[str, str], Mapping[str, float]], replicates: int
) -> dict[str, float]:
    """Every gate's and control's pass probability of ``law`` at ``R`` replicates."""
    small, large = (ep_cate.group_label(law.label, n) for n in gated_n(law))

    def at(group: str, statistic: str) -> tuple[float, float]:
        row = pilot[(group, statistic)]
        return row["mean"], row["sd"]

    out = {
        "thm5": gate_pass(*at(large, "ep"), replicates, THM5_MARGIN, below=True),
        "growth": growth_pass(at(small, "ep"), at(large, "ep"), replicates, GROWTH_MARGIN),
    }
    if law.rescue:
        for learner in RESCUE_LEARNERS:
            suffix = ep_cate.rescue_suffix(learner)
            out[f"rescue{suffix}"] = gate_pass(
                *at(large, f"ep_{learner}"), replicates, RESCUE_MARGIN, below=True
            )
            if learner not in REPORTED_GROWTH:
                out[f"rescue{suffix}_growth"] = growth_pass(
                    at(small, f"ep_{learner}"),
                    at(large, f"ep_{learner}"),
                    replicates,
                    GROWTH_MARGIN,
                )
            out[f"control{suffix}"] = gate_pass(
                *at(large, f"t_{learner}"), replicates, RESCUE_MARGIN, below=False
            )
    return out


def assured(
    law: Law, pilot: Mapping[tuple[str, str], Mapping[str, float]], replicates: int
) -> dict[str, float]:
    """:func:`tests.studies.ep_cate.law_assurance` at this study's rescue margin."""
    return ep_cate.law_assurance(
        law,
        pilot,
        replicates,
        rescue_margin=RESCUE_MARGIN,
        record=STUDY,
        rescue_learners=RESCUE_LEARNERS,
        reported_growth=REPORTED_GROWTH,
    )


def sized_replicates(
    law: Law, pilot: Mapping[tuple[str, str], Mapping[str, float]]
) -> tuple[int, bool]:
    """The N2 rule on the gates' assurance, as in :func:`tests.studies.ep_cate.sized_replicates`."""
    for replicates in R_CANDIDATES:
        if ep_cate.min_gate(assured(law, pilot, replicates)) >= TARGET_PASS:
            return replicates, True
    return R_CANDIDATES[-1], False


HESSIAN_CSV = ROOT / "tests" / "studies" / "ep_crr_hessian_pilot.csv"


def hessian_rate() -> dict[str, float]:
    """The oracle-Hessian pilot's non-PD count, pooled over every law and ``n``.

    The pilot (:mod:`tests.studies.ep_crr_hessian_pilot`) checks the oracle alone, which needs
    no nuisance fit, so it runs 3,000 checks per law and ``n``.  Returns the failures, the
    checks, the Jeffreys point estimate ``(k + 1/2) / (m + 1)`` of the per-replicate rate and
    its one-sided 95% upper limit.
    """
    from scipy import stats

    frame = pd.read_csv(HESSIAN_CSV)
    k = int(frame["non_pd"].sum())
    checks = int(frame["checks"].sum())
    upper = float(stats.beta.ppf(0.95, k + 0.5, checks - k + 0.5))
    return {
        "failures": float(k),
        "checks": float(checks),
        "rate": (k + 0.5) / (checks + 1.0),
        "upper": upper,
    }


def hessian_failure_pass(rate: float, group: Group) -> float:
    """P(a group's non-PD oracle count stays within the failure bound) at ``rate``."""
    from scipy import stats

    allowed = math.floor(FAILURE_BOUND * group.replicates + 1e-9)
    return float(stats.binom.cdf(allowed, group.replicates, rate))


def hessian_family_pass(rate: float) -> float:
    """P(every gated group stays within the failure bound) at ``rate``, groups independent."""
    return float(
        np.prod([hessian_failure_pass(rate, group) for group in GROUPS if group.kind == "gated"])
    )


#: Measured by the design review at 181 and 183 s per CV replicate at 1k.
CV_SECONDS = 182.0


def cost_hours(pilot: Mapping[tuple[str, str], Mapping[str, float]]) -> dict[str, float]:
    total = 0.0
    for group in GROUPS:
        key = (group.label, "seconds")
        if key in pilot:
            total += pilot[key]["mean"] * group.replicates
        elif group.kind == "cv":
            total += CV_SECONDS * group.replicates
        # The offset pilot timed two EP fits per replicate; the run fits one.
        offset = pilot.get((group.label, "offset_seconds"))
        if offset is not None:
            total += offset["mean"] / 2.0 * group.replicates
    total += PRIMARY_REPLICATES * 0.31
    hours = total / 3600.0
    return {"cpu_hours": hours, "wall_hours_4": hours / 4.0, "wall_hours_16": hours / 16.0}


def design_numbers() -> dict[str, Any]:
    """Every computed number the declaration fixes before the run."""
    pilot = pilot_rows()
    laws = {}
    for law in LAWS:
        replicates, powered = sized_replicates(law, pilot)
        laws[law.label] = {
            "replicates": replicates,
            "powered": powered,
            "gates": law_gates(law, pilot, replicates),
            "assurance": assured(law, pilot, replicates),
            "max_treated_mean": max_treated_mean(law),
        }
    nuisance = {}
    for law in LAWS:
        sizes = np.array(law.sizes, dtype=float)
        products = [
            pilot[(ep_cate.group_label(law.label, int(n)), "product")]["mean"] for n in law.sizes
        ]
        slope = float(np.polyfit(np.log(sizes), np.log(products), 1)[0])
        nuisance[law.label] = {"product_slope": slope, "faster_than_root_n": slope < -0.5}
    rate = hessian_rate()
    hessian = {
        "pilot": rate,
        "pass": {
            group.label: hessian_failure_pass(rate["rate"], group)
            for group in GROUPS
            if group.kind == "gated"
        },
        "family_pass": hessian_family_pass(rate["rate"]),
        "family_pass_upper": hessian_family_pass(rate["upper"]),
    }
    rescue = LAW_BY_LABEL["d3_complex_limited"]
    return {
        "laws": laws,
        "family": ep_cate.family_assurance(laws),
        "mutation_detected": mutation_detected(pilot, laws[rescue.label]["replicates"]),
        "nuisance": nuisance,
        "hessian": hessian,
        "cost": cost_hours(pilot),
        "confidence": CONFIDENCE,
    }
