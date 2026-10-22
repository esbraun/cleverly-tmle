r"""Registered evidence for ``AdaptiveIPTW`` (``adaptive-iptw``, X21, ``X21-plan.md`` 3.5 and 3.6).

The subject is :class:`~cleverly.estimators.AdaptiveIPTW`: the Horvitz-Thompson mean whose
propensity is targeted along :math:`H = Q_r / g` (van der Laan 2012, WP302, Theorem 1).  The
estimator reports an interval only when the whole mechanism is declared known.  An estimated
mechanism gives a point estimate with the status ``estimated_mechanism_plugin``, so the study
reads **bias** for every estimated-mechanism cell, and **calibration** for the known-mechanism
cells only.

**The gate study.**  This record is the Python gate (``X21-plan.md`` 3.6).  The ``drtmle``
comparator is pending: the record declares ``pending_reference="drtmle"``, its paired cells
read 0/0, and the separate study ``adaptive-iptw-pairing`` runs the pairing on its own
scenarios and seeds.  The study draws its folds through the package's own random partition
(``random_state`` from the replication seed), not from R.

**Laws and cells.**  :mod:`tests.studies.adaptive_iptw_laws` holds the laws P, PD, VL1 and VL0
and the learners.  Law M, the composite laws C and MA, the weighted law WT, the clustered law
CL and the stratified law ST are the shared law modules.  The tables :data:`READERS` and
:data:`SCENES` below are the one list of what the study fits and what it reads.  The design test
(``tests/unit/test_adaptive_iptw_study_design.py``) computes every size, every role and every
necessity row, and it checks them against this module.

**Failure bound.**  ``failure_bound`` is ``{"adaptive_iptw": 0.01}`` (:data:`FAILURE_BOUND`).  A
cell may lose ``floor(0.01 R)`` replicates, and the accounting fills each lost value at its
adverse support bound (``docs/development/method-benchmarking.md``).

**Publication policy** is ``reporting``.  Only ``bias/flexible`` and ``spread/parametric_q_ne_qr``
are routed to reporting before any run (:data:`PRE_ROUTED_GROUPS`).  Every other red cell gets an owner
row after the run.  No re-run uses a new budget, margin, law, learner or size.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import AdaptiveIPTW
from cleverly.interventions import Rule, Static, Stochastic
from tests import discrete_law_composite as composite_law
from tests.parallel import STUDY_JOBS
from tests.studies import adaptive_iptw_laws as laws
from tests.studies import clustered_unequal_laws as clustered
from tests.studies import mar_arm_indexed_laws as mar
from tests.studies import multi_arm_common as multi
from tests.studies import stratified_law as stratified
from tests.studies import weighted_point_common as weighted
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

IMPLEMENTATION = "adaptive_iptw"
PENDING_REFERENCE = "drtmle"
PENDING_REASON = (
    "the drtmle pairing runs in the study adaptive-iptw-pairing (X21-plan.md 3.6): its R "
    "runners draw_folds.R, run_study.R and print_fixtures.R are written with that study and "
    "have not run, and the nuisance gate of the 1e-8 rule runs there before its declaration. "
    "This gate study runs no R process. The declared reference sources in the manifest are the "
    "pinned drtmle image and its runner, a placeholder until adaptive-iptw-pairing lands; that "
    "change clears pending_reference and comparison"
)
SEED = 20262101
RESAMPLING_SEED = 2026210101
FAILURE_BOUND = 0.01

#: The settings of every fit: the cross-fitting folds (5 on law M), the mechanism bounds, and
#: the targeting loop's relative score tolerance and round cap (``drtmle``'s ``tolIC`` 1e-8 and
#: ``maxIter`` 100).
N_FOLDS = 10
M_FOLDS = 5
G_BOUNDS = (0.01, 0.99)
TOL = 1e-8
MAX_ITER = 100

# ----------------------------------------------------------------------- design-test outcomes
#: The replication counts the design test sets (items 1 and 4).  ``R_CAL`` is the smallest
#: multiple of 200 at which a perfectly calibrated cell passes with probability at least 0.90.
R_CAL = 2_000
R_BIAS = 400
R_LADDER = 1_000
R_VL = 1_200
#: The scenes whose readers need more than ``R_BIAS`` (items 1 and 4).  The design test
#: computes each from the pass probabilities of the rows the scene hosts.
R_RAISED: dict[str, int] = {"p_rate": 600, "rm": 800}

#: The declared sample sizes.  A clustered size is the number of clusters.
N_PD = 2_000
N_P = 3_000
LADDER_SIZES = (500, 1_500, 4_500)
N_M = 2_000
N_C = 2_000
N_WT = 2_000
N_CL = 200
N_ST = 3_000
N_CN = 2_000
N_RM = 3_000
N_MA = 2_000
N_MA_CF = 7_000
N_VL = 2_000

#: Degrees of the reduced regression's polynomial basis in ``logit g`` (capped at the number of
#: distinct training values minus one).  The cap is the saturated regression in sample and
#: for a known mechanism.  A cross-fitted estimated mechanism gives each fold up to
#: ``(N_FOLDS - 1) m`` leave-two-out values near the ``m`` cell values, so the cap does not bind
#: and the fit is not saturated.
DEGREE = {
    "p": 1,
    "st": 1,
    "m": 3,
    "c": 3,
    "wt": 3,
    "cl": 3,
    "pd": 3,
    "cn": 5,
    "rm": 2,
    "ma": 2,
    "vl": 3,
}


# ------------------------------------------------------------------------------------- labels

#: Result key -> the regex-safe label a cell name carries.
KEY_LABELS: dict[str, str] = {
    "ey0": "ey0",
    "ey1": "ey1",
    "ate": "ate",
    "rr": "rr",
    "or": "or",
    "ey_obs": "ey_obs",
    "par": "par",
    "paf": "paf",
    "ey[high]": "ey_high",
    "ey[low]": "ey_low",
    "ey[medium]": "ey_medium",
    "ey[mid]": "ey_mid",
    "ate[mid vs high]": "ate_mid",
    "ate[low vs high]": "ate_low",
    "ate[medium vs high]": "ate_medium",
    "ey_regime[never]": "ey_never",
    "ey_regime[d]": "ey_d",
    "ey_regime[s]": "ey_s",
    "ate_regime[d vs never]": "ate_d",
    "ate_regime[s vs never]": "ate_s",
    "ate_regime[s vs d]": "ate_s_vs_d",
    "ate[V=0]": "ate_v0",
    "ate[V=1]": "ate_v1",
    "ate[V=2]": "ate_v2",
}

P_KEYS = ("ey0", "ey1", "ate")
PD_ARM_KEYS = ("ey0", "ey1", "ate", "rr", "or")
PD_REGIME_KEYS = (
    "ey_regime[never]",
    "ey_regime[d]",
    "ey_regime[s]",
    "ate_regime[d vs never]",
    "ate_regime[s vs never]",
)
M_KEYS = tuple(multi.MEAN_ESTIMANDS)
C_KEYS = {
    "l1": ("ey0", "ey1", "ate"),
    "l3": tuple(name for name in mar.ESTIMANDS["l3"] if not mar.is_ratio(name)),
}
ST_KEYS = ("ate", "ate[V=0]", "ate[V=1]", "ate[V=2]")
CN_KEYS = ("ey_obs", "ey0", "par", "paf")
RM_KEYS = ("ey_regime[d]", "ey_regime[s]", "ate_regime[s vs d]")
MA_ARM_KEYS = ("ey0", "ey1", "ate")
MA_REGIME_KEYS = ("ey_regime[d]",)
RATIO_KEYS = frozenset({"rr", "or"})


# ------------------------------------------------------------------------------------- truths


@cache
def _truth_c(key: str) -> dict[str, float]:
    return dict(mar.truths(mar.LAWS[key]))


def _regime_truths(law: Any, rule: np.ndarray, tilt: np.ndarray) -> dict[str, float]:
    """``E Y(d)`` and ``E Y(s)`` of a two-arm table law, for a rule and a stochastic regime."""
    p_w, mu = np.asarray(law.p_w), np.asarray(law.mu)
    on_rule = float(np.sum(p_w * mu[np.arange(len(p_w)), rule]))
    stochastic = float(np.sum(p_w * (tilt * mu[:, 1] + (1.0 - tilt) * mu[:, 0])))
    return {"d": on_rule, "s": stochastic}


#: The rule ``d(W) = 1{W != 0}`` and the stochastic regime ``s(1 | W) = (0.2, 0.8, 0.8)`` of the
#: composite-law regime cells.
RM_RULE = np.array([0, 1, 1])
RM_TILT = np.array([0.2, 0.8, 0.8])


@cache
def truths(law: str) -> dict[str, float]:
    """The exact value of every key a law reports, by its result name."""
    if law == "p":
        return laws.truth_p()
    if law == "pd":
        return laws.truth_pd()
    if law in laws.VL_LAWS:
        return laws.truth_vl(law)
    if law == "m":
        return dict(multi.truth_for(multi.law()))
    if law in ("c_l1", "c_l3"):
        return _truth_c(law[2:])
    if law == "wt":
        return dict(weighted.truth_for(weighted.Q, weighted.P_W))
    if law == "cl":
        return {k: v for k, v in clustered.informative_truth().items() if "_cluster" not in k}
    if law == "st":
        out = {"ate": stratified.truth("ate")}
        out.update(
            {f"ate[V={s}]": stratified.truth(stratified.stratum_name("ate", s)) for s in (0, 1, 2)}
        )
        return out
    if law == "cn":
        base = mar.LAWS["l1"]
        ey_obs = float(np.sum(base.p_w[:, None] * base.g * base.mu))
        ey0 = _truth_c("l1")["ey0"]
        return {"ey_obs": ey_obs, "ey0": ey0, "par": ey_obs - ey0, "paf": (ey_obs - ey0) / ey_obs}
    if law == "rm":
        regimes = _regime_truths(mar.LAWS["l1"], RM_RULE, RM_TILT)
        return {
            "ey_regime[d]": regimes["d"],
            "ey_regime[s]": regimes["s"],
            "ate_regime[s vs d]": regimes["s"] - regimes["d"],
        }
    if law == "ma":
        two = composite_law.TWO
        regimes = _regime_truths(two.law(), RM_RULE, RM_TILT)
        out = {name: float(two.truth(name)) for name in MA_ARM_KEYS}
        out["ey_regime[d]"] = regimes["d"]
        return out
    raise KeyError(law)


# ------------------------------------------------------------------------------------ samplers


def draw_frame(law: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``law``.  A clustered ``n`` is the number of clusters."""
    if law == "p":
        return laws.sample_p(n, seed)
    if law == "pd":
        return laws.sample_p(n, seed, effect_modified=True)
    if law in laws.VL_LAWS:
        return laws.sample_vl(law, n, seed)
    if law == "m":
        frame, _ = multi.law().sample(n, seed=seed, backend="pandas")
        return frame
    if law in ("c_l1", "c_l3"):
        return mar.sample(mar.LAWS[law[2:]], n, seed)
    if law == "wt":
        return weighted.sample_selected(weighted.Q, n, seed)
    if law == "cl":
        return clustered.draw_informative(n, np.random.default_rng(seed))
    if law == "st":
        return stratified.sample(n, seed)
    if law in ("cn", "rm"):
        return mar.sample(mar.LAWS["l1"], n, seed)
    if law == "ma":
        return mar.sample(composite_law.TWO.law(), n, seed)
    raise KeyError(law)


def known_mechanism(law: str, frame: pd.DataFrame) -> np.ndarray:
    """The declared known mechanism of a known-``g`` law, the ``(n, 2)`` matrix ``[1 - g, g]``."""
    if law in ("p", "pd"):
        return laws.known_matrix(frame)
    if law == "wt":
        g = weighted.G[frame["W"].to_numpy(dtype=int)]
    elif law == "cl":
        g = clustered.informative_propensity(frame["W1"].to_numpy(), frame["W2"].to_numpy())
    elif law == "st":
        g = np.asarray(stratified.propensity(frame["W"].to_numpy(), frame["V"].to_numpy()))
    else:
        raise KeyError(law)
    return np.column_stack([1.0 - g, g])


# ------------------------------------------------------------------------------------- regimes


def pd_regimes() -> list[Any]:
    """The regimes of law PD: never treat, the rule ``1{W1 > 0}`` and the stochastic ``expit(W1)``."""
    return [
        Static(0, name=laws.NEVER_NAME),
        Rule(
            lambda frame: laws.rule_arm(frame),
            name=laws.RULE_NAME,
            rule_kind="known",
        ),
        Stochastic(
            lambda frame: np.column_stack(
                [1.0 - laws.stochastic_one(frame), laws.stochastic_one(frame)]
            ),
            name=laws.STOCHASTIC_NAME,
            density_kind="known",
        ),
    ]


def rm_regimes() -> list[Any]:
    """The regimes of the composite-law cells: the rule ``1{W != 0}`` and the tilt ``(0.2, 0.8, 0.8)``."""
    return [
        Rule(
            lambda frame: (np.asarray(frame["W"]) != 0).astype(int),
            name="d",
            rule_kind="known",
        ),
        Stochastic(
            lambda frame: np.column_stack(
                [
                    1.0 - RM_TILT[np.asarray(frame["W"]).astype(int)],
                    RM_TILT[np.asarray(frame["W"]).astype(int)],
                ]
            ),
            name="s",
            density_kind="known",
        ),
    ]


def ma_regimes() -> list[Any]:
    """The one regime of the missing-treatment cell: the rule ``1{W != 0}``."""
    return rm_regimes()[:1]


# --------------------------------------------------------------------------------------- fits


def _estimator(
    degree: int,
    *,
    treatment: Any = None,
    missingness: Any = None,
    folds: int | None = N_FOLDS,
    form: str = "targeted",
    seed: int = 0,
    simultaneous: bool = False,
    **settings: Any,
) -> AdaptiveIPTW:
    """The shared construction: the reduced basis, the bounds, the tolerance and the folds."""
    learners: dict[str, Any] = {}
    if treatment is not None:
        learners["treatment_learner"] = treatment
    if missingness is not None:
        learners["missingness_learner"] = missingness
    cross = {"cross_fit": True, "n_folds": folds} if folds else {"cross_fit": False}
    return AdaptiveIPTW(
        form=form,
        reduced_outcome_learner=laws.DesignBasisGLM(degree),
        simultaneous=simultaneous,
        g_bounds=G_BOUNDS,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=seed,
        **cross,
        **learners,
        **settings,
    )


def fit_p(
    frame: pd.DataFrame, n: int, seed: int, *, kind: str = "correct", insample: bool = False
) -> Any:
    """Law P with a mechanism of one of the kinds ``correct``, ``rate``, ``wrong``, ``flexible``."""
    return (
        _estimator(
            DEGREE["p"],
            treatment=laws.p_mechanism(kind, n),
            folds=None if insample else N_FOLDS,
            seed=seed,
            estimands=list(P_KEYS),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=list(laws.P_COLUMNS))
        .single()
    )


def fit_pd_arms(
    frame: pd.DataFrame, n: int, seed: int, *, form: str = "targeted", simultaneous: bool = False
) -> Any:
    """Law PD with the declared known mechanism, the arm axis."""
    return (
        _estimator(
            DEGREE["pd"],
            form=form,
            seed=seed,
            simultaneous=simultaneous,
            estimands=list(PD_ARM_KEYS),
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(laws.P_COLUMNS),
            treatment_probabilities=known_mechanism("pd", frame),
        )
        .single()
    )


def fit_pd_regimes(
    frame: pd.DataFrame, n: int, seed: int, *, form: str = "targeted", known: bool = True
) -> Any:
    """Law PD on the regime axis: the declared known mechanism, or ``rate_scaled`` estimated."""
    estimator = _estimator(
        DEGREE["pd"],
        form=form,
        seed=seed,
        treatment=None if known else laws.p_mechanism("rate_scaled", n),
        interventions=pd_regimes(),
    )
    arguments: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": list(laws.P_COLUMNS),
    }
    if known:
        arguments["treatment_probabilities"] = known_mechanism("pd", frame)
    return estimator.fit(frame, **arguments).single()


def fit_m(frame: pd.DataFrame, n: int, seed: int, *, rate: bool = False) -> Any:
    """Law M, three arms, five folds: the correct multinomial fit, or the fit shrunk to uniform.

    The shrinkage ``2 n^(-1/3)`` maps each arm probability by a function of the true one, so it
    keeps the level sets of ``g_a`` (law M is not H1, so a shift along a covariate would not).
    """
    return (
        _estimator(
            DEGREE["m"],
            treatment=laws.ShrunkMultinomial(2.0 if rate else 0.0, n),
            folds=M_FOLDS,
            seed=seed,
            estimands=["ey", "ate"],
            reference=multi.REFERENCE,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


def fit_c(frame: pd.DataFrame, law: str, seed: int, *, insample: bool = False) -> Any:
    """A composite law with a missing outcome, the oracle mechanisms declared as estimated."""
    table = mar.LAWS[law]
    settings: dict[str, Any] = {
        "estimands": ["ey0", "ey1", "ate"] if law == "l1" else ["ey", "ate"]
    }
    if law == "l3":
        settings["reference"] = mar.REFERENCE
    return (
        _estimator(
            DEGREE["c"],
            treatment=mar.LawTreatment(table),
            missingness=mar.LawResponse(table),
            folds=None if insample else N_FOLDS,
            seed=seed,
            **settings,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def fit_wt(frame: pd.DataFrame, n: int, seed: int, *, known: bool) -> Any:
    """Law WT with known design weights: the known mechanism, or the saturated fit."""
    estimator = _estimator(
        DEGREE["wt"],
        treatment=None if known else laws.CellFrequency(),
        seed=seed,
        estimands=list(P_KEYS),
    )
    arguments: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": ["W"],
        "weights": "obs_weight",
    }
    if known:
        arguments["treatment_probabilities"] = known_mechanism("wt", frame)
    return estimator.fit(frame, **arguments).single()


def fit_cl(frame: pd.DataFrame, n: int, seed: int, *, known: bool) -> Any:
    """Law CL with grouped folds: the known mechanism, or the correct logistic fit."""
    estimator = _estimator(
        DEGREE["cl"],
        treatment=None if known else LogisticRegression(C=np.inf, solver="newton-cholesky"),
        seed=seed,
        estimands=list(P_KEYS),
    )
    arguments: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": ["W1", "W2"],
        "id": "cluster",
    }
    if known:
        arguments["treatment_probabilities"] = known_mechanism("cl", frame)
    return estimator.fit(frame, **arguments).single()


def fit_st(frame: pd.DataFrame, n: int, seed: int, *, known: bool) -> Any:
    """Law ST with its declared strata: the known mechanism, or the correct logistic fit."""
    estimator = _estimator(
        DEGREE["st"],
        treatment=None if known else LogisticRegression(C=np.inf, solver="newton-cholesky"),
        seed=seed,
        estimands=list(P_KEYS),
    )
    arguments: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": ["W", "V"],
        "strata": ["V"],
    }
    if known:
        arguments["treatment_probabilities"] = known_mechanism("st", frame)
    return estimator.fit(frame, **arguments).single()


def _first_column_shift(size: float) -> Callable[[np.ndarray], np.ndarray]:
    """The multiple ``1 + A`` of the response perturbation, read from design column 0."""
    del size
    return lambda design: 1.0 + design[:, 0]


def fit_cn(frame: pd.DataFrame, n: int, seed: int) -> Any:
    """The natural course on law C l1: ``rate_response`` and a saturated treatment fit."""
    return (
        _estimator(
            DEGREE["cn"],
            treatment=laws.CellFrequency(),
            missingness=laws.CellFrequency(shift=_first_column_shift(2.0), size=2.0, n=n),
            seed=seed,
            estimands=["ey_obs", "ey0", "par", "paf"],
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def _rule_shift(design: np.ndarray) -> np.ndarray:
    """``2 d(W) - 1`` with ``d(W) = 1{W != 0}``, read from design column 0."""
    return 2.0 * (design[:, 0] != 0).astype(float) - 1.0


def fit_rm(frame: pd.DataFrame, n: int, seed: int) -> Any:
    """The regimes ``d`` and ``s`` on law C l1 in sample, with the rule-aligned perturbation."""
    return (
        _estimator(
            DEGREE["rm"],
            treatment=laws.CellFrequency(shift=_rule_shift, size=2.0, n=n),
            missingness=laws.CellFrequency(),
            folds=None,
            seed=seed,
            interventions=rm_regimes(),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def fit_ma(frame: pd.DataFrame, n: int, seed: int, *, mode: str) -> Any:
    """Law MA with a missing treatment and a missing outcome.

    ``mode`` is ``"arms"`` and ``"regime"`` for the in-sample fits with the oracle factor
    learners, and ``"crossfit"`` for the cross-fitted arm fit with saturated factor learners.
    """
    table = composite_law.TWO.law()
    cross = mode == "crossfit"
    settings: dict[str, Any] = (
        {"interventions": ma_regimes()} if mode == "regime" else {"estimands": list(MA_ARM_KEYS)}
    )
    return (
        _estimator(
            DEGREE["ma"],
            treatment=laws.CellFrequency() if cross else mar.LawTreatment(table),
            missingness=laws.CellFrequency() if cross else mar.LawResponse(table),
            folds=N_FOLDS if cross else None,
            seed=seed,
            **settings,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_delta="DeltaA",
        )
        .single()
    )


def fit_vl(frame: pd.DataFrame, n: int, seed: int, *, kind: str) -> Any:
    """A validity law with the logistic mechanism (``glm``) or the level-set-breaking one (``lsb``)."""
    return (
        _estimator(
            DEGREE["vl"],
            treatment=laws.ValidityLogistic(kind, n),
            seed=seed,
            estimands=["ey1"],
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        .single()
    )


#: Fit name -> ``fit(frame, n, seed)``.  Every scene names the fits it runs on one draw.
FITS: dict[str, Callable[[pd.DataFrame, int, int], Any]] = {
    "p_correct": lambda f, n, s: fit_p(f, n, s, kind="correct"),
    "p_rate": lambda f, n, s: fit_p(f, n, s, kind="rate"),
    "p_wrong": lambda f, n, s: fit_p(f, n, s, kind="wrong"),
    "p_flexible": lambda f, n, s: fit_p(f, n, s, kind="flexible"),
    "p_insample": lambda f, n, s: fit_p(f, n, s, kind="correct", insample=True),
    "pd_arms": lambda f, n, s: fit_pd_arms(f, n, s, simultaneous=True),
    "pd_arms_one_step": lambda f, n, s: fit_pd_arms(f, n, s, form="one_step"),
    "pd_regimes": lambda f, n, s: fit_pd_regimes(f, n, s),
    "pd_regimes_one_step": lambda f, n, s: fit_pd_regimes(f, n, s, form="one_step"),
    "pd_regimes_estimated": lambda f, n, s: fit_pd_regimes(f, n, s, known=False),
    "m_correct": lambda f, n, s: fit_m(f, n, s),
    "m_rate": lambda f, n, s: fit_m(f, n, s, rate=True),
    "c_l1": lambda f, n, s: fit_c(f, "l1", s),
    "c_l3": lambda f, n, s: fit_c(f, "l3", s),
    "c_l1_insample": lambda f, n, s: fit_c(f, "l1", s, insample=True),
    "c_l3_insample": lambda f, n, s: fit_c(f, "l3", s, insample=True),
    "wt_known": lambda f, n, s: fit_wt(f, n, s, known=True),
    "wt_estimated": lambda f, n, s: fit_wt(f, n, s, known=False),
    "cl_known": lambda f, n, s: fit_cl(f, n, s, known=True),
    "cl_estimated": lambda f, n, s: fit_cl(f, n, s, known=False),
    "st_known": lambda f, n, s: fit_st(f, n, s, known=True),
    "st_estimated": lambda f, n, s: fit_st(f, n, s, known=False),
    "cn": fit_cn,
    "rm": fit_rm,
    "ma_arms": lambda f, n, s: fit_ma(f, n, s, mode="arms"),
    "ma_regime": lambda f, n, s: fit_ma(f, n, s, mode="regime"),
    "ma_crossfit": lambda f, n, s: fit_ma(f, n, s, mode="crossfit"),
    "vl1_glm": lambda f, n, s: fit_vl(f, n, s, kind="glm"),
    "vl1_lsb": lambda f, n, s: fit_vl(f, n, s, kind="lsb"),
    "vl0_lsb": lambda f, n, s: fit_vl(f, n, s, kind="lsb"),
}


# ------------------------------------------------------------------------------------- scenes


@dataclass(frozen=True)
class Scene:
    """One stream of draws and the fits it runs.

    Parameters
    ----------
    name : str
        The stream name; the seed of replicate ``r`` is ``stream_seed(STUDY, "property_sample",
        name, r)``.
    law : str
        The law the stream samples (:func:`draw_frame`).
    n : int
        The declared size; the number of clusters for law CL.
    replicates : int
        The replication count of the stream.
    fits : tuple of str
        The fits (:data:`FITS`) run on every draw.
    extra_fits : tuple of str
        Fits run on the first :attr:`extra_replicates` draws only.
    extra_replicates : int
        How many leading draws also run :attr:`extra_fits`.
    """

    name: str
    law: str
    n: int
    replicates: int
    fits: tuple[str, ...]
    extra_fits: tuple[str, ...] = ()
    extra_replicates: int = 0


def _scenes() -> tuple[Scene, ...]:
    out = [
        Scene(
            "pd_known",
            "pd",
            N_PD,
            R_CAL,
            ("pd_arms", "pd_arms_one_step", "pd_regimes", "pd_regimes_one_step"),
        ),
        Scene("wt_known", "wt", N_WT, R_CAL, ("wt_known",), ("wt_estimated",), R_BIAS),
        Scene("cl_known", "cl", N_CL, R_CAL, ("cl_known",), ("cl_estimated",), R_BIAS),
        Scene("st_known", "st", N_ST, R_CAL, ("st_known",), ("st_estimated",), R_BIAS),
        Scene("p_correct", "p", N_P, R_RAISED.get("p_correct", R_BIAS), ("p_correct",)),
        Scene("p_rate", "p", N_P, R_RAISED.get("p_rate", R_BIAS), ("p_rate",)),
        Scene("p_insample", "p", N_P, R_RAISED.get("p_insample", R_BIAS), ("p_insample",)),
        Scene("p_wrong", "p", N_P, R_RAISED.get("p_wrong", R_BIAS), ("p_wrong",)),
        Scene("p_flexible", "p", N_P, R_RAISED.get("p_flexible", R_BIAS), ("p_flexible",)),
        *(Scene(f"p_ladder_{n}", "p", n, R_LADDER, ("p_rate",)) for n in LADDER_SIZES),
        Scene("m_correct", "m", N_M, R_RAISED.get("m_correct", R_BIAS), ("m_correct",)),
        Scene("m_rate", "m", N_M, R_RAISED.get("m_rate", R_BIAS), ("m_rate",)),
        Scene("c_l1", "c_l1", N_C, R_RAISED.get("c_l1", R_BIAS), ("c_l1",)),
        Scene("c_l3", "c_l3", N_C, R_RAISED.get("c_l3", R_BIAS), ("c_l3",)),
        Scene(
            "c_l1_insample", "c_l1", N_C, R_RAISED.get("c_l1_insample", R_BIAS), ("c_l1_insample",)
        ),
        Scene(
            "c_l3_insample", "c_l3", N_C, R_RAISED.get("c_l3_insample", R_BIAS), ("c_l3_insample",)
        ),
        Scene("pd_rate", "pd", N_PD, R_RAISED.get("pd_rate", R_BIAS), ("pd_regimes_estimated",)),
        Scene("cn", "cn", N_CN, R_RAISED.get("cn", R_BIAS), ("cn",)),
        Scene("rm", "rm", N_RM, R_RAISED.get("rm", R_BIAS), ("rm",)),
        Scene(
            "ma_insample", "ma", N_MA, R_RAISED.get("ma_insample", R_BIAS), ("ma_arms", "ma_regime")
        ),
        Scene("ma_crossfit", "ma", N_MA_CF, R_RAISED.get("ma_crossfit", R_BIAS), ("ma_crossfit",)),
        Scene("vl1", "vl1", N_VL, R_VL, ("vl1_glm", "vl1_lsb")),
        Scene("vl0", "vl0", N_VL, R_RAISED.get("vl0", R_BIAS), ("vl0_lsb",)),
    ]
    return tuple(out)


SCENES: dict[str, Scene] = {scene.name: scene for scene in _scenes()}


# ------------------------------------------------------------------------------------ readers


@dataclass(frozen=True)
class Reader:
    """One row family of a cell: which key of which fit of which scene a cell reads.

    Parameters
    ----------
    family : str
        The property family.
    cell : str
        The cell name.
    role : str
        ``positive``, ``control`` or ``diagnostic``.
    scene : str
        The stream (:data:`SCENES`).
    fit : str
        The fit of the stream (:data:`FITS`).
    key : str
        The result key the row reads.
    kind : str
        ``estimate`` reads the reported estimate, ``untargeted`` the Horvitz-Thompson mean at
        the initial mechanism, ``band`` the simultaneous band of the fit, and ``spread`` the
        estimate with its plug-in diagnostic (a diagnostic row).
    """

    family: str
    cell: str
    role: str
    scene: str
    fit: str
    key: str
    kind: str = "estimate"


def label_of(key: str) -> str:
    """The cell-name label of a result key."""
    return KEY_LABELS[key]


def _calibration(scene: str, fit: str, cell: str, keys: Sequence[str]) -> list[Reader]:
    return [
        Reader(
            "interval_calibration",
            f"{label_of(key)}__{cell}__correctly_specified",
            "positive",
            scene,
            fit,
            key,
        )
        for key in keys
    ]


def _bias(
    cell: str,
    scene: str,
    fit: str,
    keys: Sequence[str],
    role: str = "positive",
    *,
    overrides: Mapping[str, str] | None = None,
) -> list[Reader]:
    return [
        Reader(
            "bias",
            f"{label_of(key)}__{cell}",
            (overrides or {}).get(key, role),
            scene,
            fit,
            key,
        )
        for key in keys
    ]


#: The necessity families' labels: one per row of the plan's necessity list.  Each pairs the
#: targeted estimate with the untargeted Horvitz-Thompson mean of the same fit.
NECESSITY_LABELS: dict[str, tuple[str, str, str, str]] = {
    # label -> (scene, fit, key, law)
    "p_ey0": ("p_rate", "p_rate", "ey0", "p"),
    "p_ey1": ("p_rate", "p_rate", "ey1", "p"),
    "p_ate": ("p_rate", "p_rate", "ate", "p"),
    "pd_d": ("pd_rate", "pd_regimes_estimated", "ey_regime[d]", "pd"),
    "pd_s": ("pd_rate", "pd_regimes_estimated", "ey_regime[s]", "pd"),
    "cn_ey_obs": ("cn", "cn", "ey_obs", "cn"),
    "rm_d": ("rm", "rm", "ey_regime[d]", "rm"),
}
NECESSITY_THRESHOLD = 1.0


def _necessity() -> list[Reader]:
    out = []
    for label, (scene, fit, key, _) in NECESSITY_LABELS.items():
        out.append(Reader("targeting_necessity", f"{label}__targeted", "positive", scene, fit, key))
        out.append(
            Reader(
                "targeting_necessity",
                f"{label}__untargeted",
                "control",
                scene,
                fit,
                key,
                kind="untargeted",
            )
        )
    return out


#: The role of a row the design test reads off the predicted bias ``b`` (item 2), by
#: ``family/cell``.  A row absent here keeps the role its reader declares.
ROLE_OVERRIDES: dict[str, str] = {}


def _readers() -> tuple[Reader, ...]:
    cal: list[Reader] = []
    cal += _calibration("pd_known", "pd_arms", "known_arms", PD_ARM_KEYS)
    cal += _calibration("pd_known", "pd_arms_one_step", "one_step_known_arms", PD_ARM_KEYS)
    cal += _calibration("pd_known", "pd_regimes", "known_regimes", PD_REGIME_KEYS)
    cal += _calibration("pd_known", "pd_regimes_one_step", "one_step_known_regimes", PD_REGIME_KEYS)
    cal += _calibration("wt_known", "wt_known", "known_weighted", P_KEYS)
    cal += _calibration("cl_known", "cl_known", "known_clustered", P_KEYS)
    cal += _calibration("st_known", "st_known", "known_stratified", ST_KEYS)

    out = [
        *cal,
        Reader(
            "simultaneous_coverage",
            "known_band__simultaneous_band",
            "positive",
            "pd_known",
            "pd_arms",
            "max_t",
            kind="band",
        ),
        *_bias("correct", "p_correct", "p_correct", P_KEYS),
        *_bias("rate", "p_rate", "p_rate", P_KEYS),
        *_bias("insample_correct", "p_insample", "p_insample", P_KEYS),
        *_bias("wrong_mechanism", "p_wrong", "p_wrong", P_KEYS, "control"),
        *_bias("flexible", "p_flexible", "p_flexible", P_KEYS),
        *_bias("three_arm", "m_correct", "m_correct", M_KEYS),
        *_bias("rate_reduced", "m_rate", "m_rate", M_KEYS),
        *_bias("composite_l1", "c_l1", "c_l1", C_KEYS["l1"]),
        *_bias("composite_l3", "c_l3", "c_l3", C_KEYS["l3"]),
        *_bias("insample_composite_l1", "c_l1_insample", "c_l1_insample", C_KEYS["l1"]),
        *_bias("insample_composite_l3", "c_l3_insample", "c_l3_insample", C_KEYS["l3"]),
        *_bias("weighted", "wt_known", "wt_estimated", P_KEYS),
        *_bias("clustered", "cl_known", "cl_estimated", P_KEYS),
        *_bias("stratified", "st_known", "st_estimated", ST_KEYS),
        *_bias("regime", "pd_rate", "pd_regimes_estimated", PD_REGIME_KEYS),
        *_bias("natural_course", "cn", "cn", CN_KEYS),
        *_bias("insample_regime_missing", "rm", "rm", RM_KEYS),
        *_bias("insample_missing_treatment", "ma_insample", "ma_arms", MA_ARM_KEYS),
        *_bias("insample_regime_missing_treatment", "ma_insample", "ma_regime", MA_REGIME_KEYS),
        *_bias("missing_treatment_crossfit", "ma_crossfit", "ma_crossfit", MA_ARM_KEYS),
        *_bias("level_set_breaking", "vl1", "vl1_lsb", ("ey1",), "control"),
        *_bias("level_set_breaking_h1", "vl0", "vl0_lsb", ("ey1",)),
        *_bias("parametric_q_ne_qr", "vl1", "vl1_glm", ("ey1",)),
        *(
            Reader("root_n_points", f"n_{n}", "positive", f"p_ladder_{n}", "p_rate", "ate")
            for n in LADDER_SIZES
        ),
        *_necessity(),
        Reader(
            "spread",
            "parametric_q_ne_qr",
            "diagnostic",
            "vl1",
            "vl1_glm",
            "ey1",
            kind="spread",
        ),
    ]
    return tuple(
        Reader(
            r.family,
            r.cell,
            ROLE_OVERRIDES.get(f"{r.family}/{r.cell}", r.role),
            r.scene,
            r.fit,
            r.key,
            r.kind,
        )
        for r in out
    )


READERS: tuple[Reader, ...] = _readers()

#: Cell groups the plan routes to reporting before any run (``X21-plan.md`` 3.5, the
#: pre-declared routing).  A red verdict in one of these is read as the declared reading, not as
#: a defect.  A group names every cell that ends with its name.
PRE_ROUTED_GROUPS: dict[str, str] = {
    "bias/flexible": (
        "law P is H1, so the validity term vanishes for this learner; the gradient-boosting "
        "learner's rate is not shown, so a red cell reads as the first or third C3 term"
    ),
    "spread/parametric_q_ne_qr": (
        "the Theorem 1 curve is conservative under a fitted parametric mechanism when "
        "Q_0 != Qr_0 (the projection of 1.2); this is why AdaptiveIPTW withholds the interval"
    ),
}

#: The middle-band cells the design test (item 2, rev 3.9) routes to reporting, because no
#: replication count of at most 1,600 passes their predicted standardized bias ``b`` with
#: probability 0.90 or because ``|b|`` does not fall from ``n`` to ``4 n``.  Each reading holds
#: the cell's ``b``.  The set is empty: the law P ``rate`` rows are second order and fall with
#: ``n``, and the law M ``rate`` learner keeps the level sets of each arm mechanism.
DESIGN_ROUTED: dict[str, str] = {}


def _routed() -> dict[str, str]:
    cells = {
        reader.family + "/" + reader.cell: reader.family + "/" + reader.cell for reader in READERS
    }
    out = {}
    for group, reading in PRE_ROUTED_GROUPS.items():
        family, _, suffix = group.partition("/")
        for name in cells:
            if name.startswith(family + "/") and (name == group or name.endswith("__" + suffix)):
                out[name] = reading
    return {**out, **DESIGN_ROUTED}


PRE_ROUTED: dict[str, str] = _routed()

#: The reading of a red row, declared before any run (``X21-plan.md`` 3.5, rev 3.9, and
#: ``AGENT-BRIEF.md``, "Reading a red calibration row").  The routing names above enter the
#: manifest through :data:`CONFIGURATION`, and no verdict code reads them: a pre-routed row
#: publishes its verdict, and this reading says how to take a red there.
RED_READING: dict[str, str] = {
    "primary": "a red primary row blocks the merge and gets a diagnosis",
    "calibration": (
        "a red calibration row is a finite-sample red only when the 99% interval of its SE ratio "
        "contains 1, its coverage interval contains 0.95 or lies inside [0.92, 0.98], and the "
        "diagnosis finds no defect; it then goes to reporting with an owner"
    ),
    "bias": (
        "a red positive bias row is a finite-sample red only when the 99% interval of its "
        "standardized bias contains the design-predicted b, the other rows of its scene are "
        "consistent with their own b, and the diagnosis finds no defect; it then goes to "
        "reporting with an owner"
    ),
    "ladder": (
        "a red root_n_points rung or root_n_rate slope reads as a bias row does: finite-sample "
        "only when its interval contains the predicted b of the rung"
    ),
    "necessity": (
        "a targeted positive row reads as a bias row; a red displacement, or an untargeted "
        "control that does not fail, is a defect suspect and never a finite-sample red"
    ),
    "control": (
        "a control that does not fail is a defect suspect in the instrument, never a "
        "finite-sample red"
    ),
}

#: The red rows a correct estimator is expected to show by chance, from the design pass
#: probabilities (the design test computes both numbers).  A run with about this many reds,
#: read by :data:`RED_READING`, is the expected outcome and no defect.
EXPECTED_CHANCE_REDS: dict[str, float] = {"calibration": 2.0, "bias_ladder_necessity": 1.5}
EXPECTED_CHANCE_REDS_TOTAL = 3.4


def property_cells() -> dict[str, tuple[str, ...]]:
    """The cells the committed property summary must contain, by family."""
    cells: dict[str, list[str]] = {}
    for reader in READERS:
        names = cells.setdefault(reader.family, [])
        if reader.cell not in names:
            names.append(reader.cell)
    cells["root_n_rate"] = ["empirical_sd"]
    return {family: tuple(names) for family, names in cells.items()}


PROPERTY_CELLS = property_cells()

# --------------------------------------------------------------------------------------- record

#: The primary scenario: the cheapest meaningful one, the known-mechanism arm fit of law PD.
PRIMARY_SCENARIO = "known_arms"
PRIMARY_KEYS = P_KEYS
PRIMARY_N = N_PD
PRIMARY_REPLICATES = R_CAL

STUDY = StudyRecord(
    name="adaptive-propensity IPTW",
    slug="adaptive-iptw",
    artifacts=ROOT / "tests" / "canonical" / "adaptive_iptw",
    document="docs/technical-reference/method-evidence/adaptive-iptw.md",
    anchor="adaptive-iptw",
    scenarios={PRIMARY_SCENARIO: PRIMARY_KEYS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    nuisance_count=1,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    pending_reference=PENDING_REFERENCE,
    pending_reason=PENDING_REASON,
    comparison="matching implementation",
    calibration_efficiency_ratio=False,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/studies/canonical_adaptive_iptw.py",
        "tests/studies/adaptive_iptw_properties.py",
        "tests/studies/adaptive_iptw_laws.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/mar_arm_indexed_laws.py",
        "tests/studies/weighted_point_common.py",
        "tests/studies/clustered_unequal_laws.py",
        "tests/studies/stratified_law.py",
        "tests/discrete_law_composite.py",
        "tests/discrete_law_mar.py",
        "tests/discrete_law_multi.py",
        "src/cleverly/estimators/iptw.py",
        "src/cleverly/estimators/reduced.py",
        "src/cleverly/fluctuation/mechanism.py",
        "src/cleverly/fluctuation/_score.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_adaptive_iptw",
    properties_module="tests.studies.adaptive_iptw_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "estimator": "AdaptiveIPTW",
    "form": "targeted (the one-step form is fitted beside it on the known-mechanism cells)",
    "reduced_outcome_learner": "unpenalized logistic (least squares for a stochastic-regime key) "
    "on a polynomial of logit(g), degree per law, capped at the number of distinct values "
    "minus one",
    "degree": dict(DEGREE),
    "g_bounds": list(G_BOUNDS),
    "tolerance": TOL,
    "max_iter": MAX_ITER,
    "n_folds": {"default": N_FOLDS, "m": M_FOLDS},
    "folds": "the package's random partition, random_state from the replication seed",
    "sizes": {
        "pd": N_PD,
        "p": N_P,
        "ladder": list(LADDER_SIZES),
        "m": N_M,
        "c": N_C,
        "wt": N_WT,
        "cl_clusters": N_CL,
        "st": N_ST,
        "cn": N_CN,
        "rm": N_RM,
        "ma": N_MA,
        "ma_crossfit": N_MA_CF,
        "vl": N_VL,
    },
    "replicates": {"cal": R_CAL, "bias": R_BIAS, "ladder": R_LADDER, "vl": R_VL},
    "pre_routed": dict(PRE_ROUTED),
    "raised_replicates": dict(R_RAISED),
    "red_reading": dict(RED_READING),
    "expected_chance_reds": {**EXPECTED_CHANCE_REDS, "total": EXPECTED_CHANCE_REDS_TOTAL},
    "reference": "none yet; the drtmle 538a3a2 pairing runs in adaptive-iptw-pairing",
}


# ----------------------------------------------------------------------------- primary phase


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of the primary scenario with its exact truths."""
    del scenario
    return draw_frame("pd", n, seed), dict(truths("pd"))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def untargeted_value(result: Any, key: str) -> float:
    """The Horvitz-Thompson mean at the initial mechanism for a reported key."""
    record = result.extra["adaptive_iptw"]
    if key in record.untargeted:
        return float(record.untargeted[key])
    stem, _, argument = key.partition("[")
    parts = argument.rstrip("]").split(" vs ")
    if key == "ate" and {"0", "1"} <= set(record.untargeted):
        return float(record.untargeted["1"] - record.untargeted["0"])
    if key == "ey0":
        return float(record.untargeted["0"])
    if key == "ey1":
        return float(record.untargeted["1"])
    if stem in ("ey_regime", "ate_regime") and parts:
        values = [float(record.untargeted[name]) for name in parts]
        return values[0] if stem == "ey_regime" else values[0] - values[1]
    if key in ("par", "paf") and {"ey_obs", "0"} <= set(record.untargeted):
        par = float(record.untargeted["ey_obs"] - record.untargeted["0"])
        return par if key == "par" else par / float(record.untargeted["ey_obs"])
    raise KeyError(f"no untargeted value for {key!r}")


def primary_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """The primary rows of one replication of the known-mechanism arm fit."""
    rows = []
    for name in STUDY.scenarios[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": IMPLEMENTATION,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": untargeted_value(result, name),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    frame, truth = drawn
    seed = replicate_seed(STUDY, scenario, replicate)
    result = fit_pd_arms(frame, n, seed)
    return primary_rows(result, truth, scenario, replicate, n)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of the primary scenario: the known-mechanism arm fit of law PD."""
    outcomes = replication.fit_primary(
        [(PRIMARY_SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=IMPLEMENTATION,
        rows_at=None,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    estimates = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)])


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand, by scenario."""
    return {PRIMARY_SCENARIO: {name: truths("pd")[name] for name in PRIMARY_KEYS}}


__all__ = [
    "FITS",
    "PROPERTY_CELLS",
    "READERS",
    "SCENES",
    "STUDY",
    "draw_and_fit",
    "truths",
]
