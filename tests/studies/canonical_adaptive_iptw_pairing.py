r"""Registered pairing of ``AdaptiveIPTW`` with R ``drtmle::adaptive_iptw`` (``adaptive-iptw-pairing``).

The subject is :class:`~cleverly.estimators.AdaptiveIPTW` with the construction of the gate
study :mod:`tests.studies.canonical_adaptive_iptw`: the same laws, learners, reduced basis,
bounds, tolerance and round cap.  The comparator is R ``drtmle`` 1.1.2 at ``538a3a2``,
``adaptive_iptw``, through ``tests/canonical/adaptive_iptw_pairing/run_study.R``.  The gate study
keeps its own seeds and verdicts; this study only pairs (``X21-plan.md`` 3.6, revised by the
pairing plan of 2026-11-01).

**What pairs.**  ``drtmle`` fits no mechanism here: the runner passes this package's mechanism
at each key as ``gn``, and both sides fit the reduced regression ``Qr`` on the same basis of
``logit gn`` and target the propensity along ``Qr / g``.

- *Known mechanism, cross-fitted* (laws PD and ST, ten folds): the initial ``Qr`` and the
  one-step form are the same computation.  The targeted form differs at order ``1/n`` after the
  first round, because the package moves fold ``k``'s design along fold ``k``'s own covariate
  (declared difference D9).  Points and standard errors pair.
- *Estimated mechanism, in sample* (laws P, M, C, PD, CN, RM, MA): the same algorithm on both
  sides.  Points pair.  Both sides' ``std_error`` is the Theorem 1 curve, which the package
  withholds as inference under ``estimated_mechanism_plugin``; its rows carry the plug-in
  diagnostic (:func:`~tests.studies.evidence.schema.reported_inference`).
- *Estimated mechanism, cross-fitted*: not paired.  ``drtmle`` trains fold ``k``'s ``Qr`` on a
  supplied out-of-fold ``gn`` that saw fold ``k``; the package trains it on leave-two-folds-out
  designs.  They are different estimators (D9; ``X21-red-cell-diagnosis.md``).

**Folds.**  ``adaptive_iptw`` draws its folds inline as ``split(sample(seq_len(n)), rep(1:K,
length = n))``.  For each cross-fitted draw this module draws a uniform permutation and deals
row ``perm[i]`` to fold ``i mod K`` (:func:`deal`), which is that partition, fits the package on
it (:class:`PairingAdaptiveIPTW`), and writes each row's 1-based position in the permutation.
The runner hands ``drtmle`` that permutation in place of ``sample``.  Law ST deals each stratum
separately, because the runner calls ``drtmle`` once per stratum.

**Recodes.**  The runner builds every regime, composite and natural-course recode from the raw
columns and the law's rule and density, written in R.  This module supplies only the folds and,
for each key, the package's initial mechanism ``gn`` and its initial ``Qr`` for the nuisance
gate.

Publication policy is ``reporting``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from cleverly.estimators import AdaptiveIPTW
from cleverly.learners.crossfit import Folds, check_integrity
from tests import discrete_law_composite as composite_law
from tests.parallel import STUDY_JOBS
from tests.studies import adaptive_iptw_laws as laws
from tests.studies import canonical_adaptive_iptw as gate
from tests.studies import mar_arm_indexed_laws as mar
from tests.studies import multi_arm_common as multi
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed
from tests.studies.point_study_helpers import primary_rows

IMPLEMENTATION = "adaptive_iptw"
REFERENCE = "drtmle"
DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
SEED = 20262102
RESAMPLING_SEED = 2026210201
FAILURE_BOUND = 0.01
PRIMARY_REPLICATES = 1_000
PRIMARY_N = gate.N_PD
N_FOLDS = gate.N_FOLDS

#: The largest ``|Qr_python - Qr_drtmle|`` over a key's rows that the runner accepts on a
#: replicate (the 1e-8 rule of ``X21-plan.md`` 3.5).  The runner carries its own copy.
NUISANCE_GATE = 1e-8


# ----------------------------------------------------------------------------------- draws


@dataclass(frozen=True)
class Draw:
    """One sample stream and how it is cut into folds.

    Parameters
    ----------
    law : str
        The law (:func:`tests.studies.canonical_adaptive_iptw.draw_frame`).
    n : int
        The declared size.
    folds : str
        ``"none"`` in sample, ``"whole"`` one permutation of the sample, ``"strata"`` one
        permutation per stratum of ``V``.
    """

    law: str
    n: int
    folds: str


DRAWS: dict[str, Draw] = {
    "pd_known": Draw("pd", gate.N_PD, "whole"),
    "st_known": Draw("st", gate.N_ST, "strata"),
    "p_insample": Draw("p", gate.N_P, "none"),
    "m_insample": Draw("m", gate.N_M, "none"),
    "c_l1_insample": Draw("c_l1", gate.N_C, "none"),
    "c_l3_insample": Draw("c_l3", gate.N_C, "none"),
    "pd_insample": Draw("pd", gate.N_PD, "none"),
    "cn_insample": Draw("cn", gate.N_CN, "none"),
    "rm": Draw("rm", gate.N_RM, "none"),
    "ma_insample": Draw("ma", gate.N_MA, "none"),
    "c_l1_crossfit": Draw("c_l1", gate.N_C, "whole"),
    "c_l3_crossfit": Draw("c_l3", gate.N_C, "whole"),
    "cn_crossfit": Draw("cn", gate.N_CN, "whole"),
    "ma_crossfit": Draw("ma", gate.N_MA, "whole"),
}


@dataclass(frozen=True)
class Scenario:
    """One paired cell family: a fit on a draw, and the keys it reports.

    Parameters
    ----------
    draw : str
        The draw (:data:`DRAWS`).
    fit : str
        The fit (:data:`FITS`).
    keys : tuple of str
        The reported keys, in publication order.
    """

    draw: str
    fit: str
    keys: tuple[str, ...]


ST_KEYS = gate.ST_KEYS

SCENARIOS: dict[str, Scenario] = {
    "pd_known_arms": Scenario("pd_known", "pd_known_arms", gate.PD_ARM_KEYS),
    "pd_known_arms_one_step": Scenario("pd_known", "pd_known_arms_one_step", gate.PD_ARM_KEYS),
    "pd_known_regimes": Scenario("pd_known", "pd_known_regimes", gate.PD_REGIME_KEYS),
    "pd_known_regimes_one_step": Scenario(
        "pd_known", "pd_known_regimes_one_step", gate.PD_REGIME_KEYS
    ),
    "st_known": Scenario("st_known", "st_known", ST_KEYS),
    "p_insample": Scenario("p_insample", "p_insample", gate.P_KEYS),
    "p_rate_insample": Scenario("p_insample", "p_rate_insample", gate.P_KEYS),
    "m_insample": Scenario("m_insample", "m_insample", gate.M_KEYS),
    "m_rate_insample": Scenario("m_insample", "m_rate_insample", gate.M_KEYS),
    "c_l1_insample": Scenario("c_l1_insample", "c_l1_insample", gate.C_KEYS["l1"]),
    "c_l3_insample": Scenario("c_l3_insample", "c_l3_insample", gate.C_KEYS["l3"]),
    "pd_regimes_insample": Scenario("pd_insample", "pd_regimes_insample", gate.PD_REGIME_KEYS),
    "cn_insample": Scenario("cn_insample", "cn_insample", gate.CN_KEYS),
    "rm": Scenario("rm", "rm", gate.RM_KEYS),
    "ma_arms": Scenario("ma_insample", "ma_arms", gate.MA_ARM_KEYS),
    "ma_regime": Scenario("ma_insample", "ma_regime", gate.MA_REGIME_KEYS),
    "c_l1_crossfit": Scenario("c_l1_crossfit", "c_l1_crossfit", gate.C_KEYS["l1"]),
    "c_l1_crossfit_one_step": Scenario(
        "c_l1_crossfit", "c_l1_crossfit_one_step", gate.C_KEYS["l1"]
    ),
    "c_l3_crossfit": Scenario("c_l3_crossfit", "c_l3_crossfit", gate.C_KEYS["l3"]),
    "c_l3_crossfit_one_step": Scenario(
        "c_l3_crossfit", "c_l3_crossfit_one_step", gate.C_KEYS["l3"]
    ),
    "cn_crossfit": Scenario("cn_crossfit", "cn_crossfit", gate.CN_KEYS),
    "cn_crossfit_one_step": Scenario("cn_crossfit", "cn_crossfit_one_step", gate.CN_KEYS),
    "ma_crossfit": Scenario("ma_crossfit", "ma_crossfit", gate.MA_ARM_KEYS),
    "ma_crossfit_one_step": Scenario("ma_crossfit", "ma_crossfit_one_step", gate.MA_ARM_KEYS),
}

#: The per-replicate agreement bound of each scenario, in units of the package's standard
#: error: the 99th percentile over replicates of ``|cleverly - drtmle| / SE`` must stay below
#: it for every key.  In sample, and for the one-step form, the two run the same arithmetic and
#: differ by the stop rules (D3) and R's default ``glm`` convergence: about 1e-6.  A targeted
#: cross-fitted fit differs at order ``1/n`` after its first round (D9): at most 0.007 SE in the
#: planning probes and 0.006 SE in the two-replicate R test.
AGREEMENT_EXACT = 1e-4
AGREEMENT_CROSSFIT = 0.05
#: The one-step scenarios.  The runner carries the same list (``ONE_STEP`` in ``run_study.R``).
ONE_STEP = frozenset(
    {
        "pd_known_arms_one_step",
        "pd_known_regimes_one_step",
        "c_l1_crossfit_one_step",
        "c_l3_crossfit_one_step",
        "cn_crossfit_one_step",
        "ma_crossfit_one_step",
    }
)
#: The cross-fitted targeted fits on a composite that takes a few values.  drtmle's loop and the
#: package's reach different solutions of the same score: the 30-replicate pilot measured
#: 99th-percentile gaps of 0.015 (C l1) to 0.22 SE (C l3 ``ey[low]``), heavy-tailed, against
#: 0.008 to 0.013 SE on law PD.  Their agreement is reported with no verdict; the one-step form
#: of each, which reads the same folds and the same initial ``Qr``, carries the verdict at
#: :data:`AGREEMENT_EXACT`.
DESCRIPTIVE = frozenset({"c_l1_crossfit", "c_l3_crossfit", "cn_crossfit", "ma_crossfit"})
#: The gate's reduced-basis degree key of each law (:data:`tests.studies.canonical_adaptive_iptw.DEGREE`).
#: The runner carries each scenario's degree (``DEGREE`` in ``run_study.R``).
DEGREE_KEY: dict[str, str] = {
    "pd": "pd",
    "st": "st",
    "p": "p",
    "m": "m",
    "c_l1": "c",
    "c_l3": "c",
    "cn": "cn",
    "rm": "rm",
    "ma": "ma",
}
#: The quantile of the per-replicate gaps the agreement bound applies to.
AGREEMENT_QUANTILE = 0.99
#: The largest ``|initial_cleverly - initial_drtmle|`` accepted: both sides read one
#: Horvitz-Thompson mean at the same mechanism.
INITIAL_AGREEMENT = 1e-10
#: A key whose mean ``|estimate - initial_estimate| / SE`` stays below this multiple of its
#: agreement bound is published as "targeting not witnessed": the pair could not have told a
#: targeting step from none.  No verdict reads it.
WITNESS_MULTIPLE = 10.0

#: Scenario -> the scenario whose stream it draws: the first scenario of each draw.
SEED_OWNERS: dict[str, str] = {}
_first: dict[str, str] = {}
for _name, _scenario in SCENARIOS.items():
    _owner = _first.setdefault(_scenario.draw, _name)
    if _owner != _name:
        SEED_OWNERS[_name] = _owner

#: The arm labels of a string-coded treatment, in the package's arm-code order.  The runner
#: carries the same table (``ARM_LABELS`` in ``run_study.R``); a fit whose codes differ stops.
ARM_LABELS: dict[str, tuple[str, ...]] = {
    "m_insample": ("high", "low", "medium"),
    "m_rate_insample": ("high", "low", "medium"),
    "c_l3_insample": ("high", "low", "mid"),
    "c_l3_crossfit": ("high", "low", "mid"),
    "c_l3_crossfit_one_step": ("high", "low", "mid"),
}

#: The covariate and treatment columns the samples file carries, by law.
LAW_COLUMNS: dict[str, tuple[str, ...]] = {
    "pd": ("W1", "W2", "W12", "A", "Y"),
    "st": ("V", "W", "A", "Y"),
    "p": ("W1", "W2", "W12", "A", "Y"),
    "m": ("W1", "W2", "W3", "A", "Y"),
    "c_l1": ("W", "A", "Y", "Delta"),
    "c_l3": ("W", "A", "Y", "Delta"),
    "cn": ("W", "A", "Y", "Delta"),
    "rm": ("W", "A", "Y", "Delta"),
    "ma": ("W", "A", "Y", "Delta", "DeltaA"),
}
SAMPLE_BASE = ("W", "W1", "W2", "W3", "W12", "V", "A", "Y", "Delta", "DeltaA")
#: The call labels: arm codes, the never-treat and rule keys, the two arms of the stochastic
#: regime, and the natural course.
CALL_LABELS = ("a0", "a1", "a2", "never", "d", "s0", "s1", "obs")
ARM_COLUMNS = ("a0", "a1", "a2")
#: The raw factors R composes each key's mechanism from: the treatment factor ``tr``, the
#: outcome observation factor ``pi`` per arm, the treatment observation factor ``pA``, the
#: treatment factor's truncation rule (``simplex``, ``g_lo``, ``g_hi``) and the observation
#: factors' floor.
FACTOR_COLUMNS = (
    *(f"tr_{arm}" for arm in ARM_COLUMNS),
    *(f"pi_{arm}" for arm in ARM_COLUMNS),
    "pA",
    "simplex",
    "g_lo",
    "g_hi",
    "floor",
)
SAMPLE_COLUMNS = (
    "scenario",
    "replicate",
    *SAMPLE_BASE,
    "position",
    *FACTOR_COLUMNS,
    *(f"{kind}_{label}" for label in CALL_LABELS for kind in ("gn", "qr", "lb", "ub")),
)


# ------------------------------------------------------------------------------- the record

STUDY = StudyRecord(
    name="Adaptive-propensity IPTW against drtmle",
    slug="adaptive-iptw-pairing",
    artifacts=ROOT / "tests" / "canonical" / "adaptive_iptw_pairing",
    document="docs/technical-reference/method-evidence/adaptive-iptw-pairing.md",
    anchor="adaptive-propensity-iptw-against-drtmle",
    scenarios={name: scenario.keys for name, scenario in SCENARIOS.items()},
    replicates=PRIMARY_REPLICATES,
    n=gate.N_PD,
    scenario_n={name: DRAWS[scenario.draw].n for name, scenario in SCENARIOS.items()},
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners=SEED_OWNERS,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND, REFERENCE: FAILURE_BOUND},
    comparison="matching implementation",
    modules=(
        "tests/studies/canonical_adaptive_iptw_pairing.py",
        "tests/studies/adaptive_iptw_pairing_properties.py",
        "tests/studies/canonical_adaptive_iptw.py",
        "tests/studies/adaptive_iptw_laws.py",
        "tests/studies/clustered_unequal_laws.py",
        "tests/studies/mar_arm_indexed_laws.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/stratified_law.py",
        "tests/studies/weighted_point_common.py",
        "tests/discrete_law_composite.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/drtmle/Dockerfile",
        "tests/canonical/adaptive_iptw_pairing/run_study.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_adaptive_iptw_pairing",
    properties_module="tests.studies.adaptive_iptw_pairing_properties",
    property_cells={},
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "drtmle::adaptive_iptw(W, A, Y, a_0, family, gn = this package's initial mechanism at "
        "each key, glm_Qr = the key's polynomial in qlogis(gn), tolIC = 1e-8, maxIter = 100, "
        "tolg = the key's lower mechanism bound, cvFolds = 10 on the supplied permutation or 1 "
        "in sample); regime, composite and natural-course recodes built in R from the raw "
        "columns; contrasts, ratios, par and paf and their standard errors computed by the "
        "runner from drtmle's estimates and the curves it rebuilds from drtmle's nuisances"
    ),
    "folds": (
        "one uniform permutation per cross-fitted draw (per stratum on law ST), dealt "
        "round-robin; drtmle receives it in place of sample(seq_len(n))"
    ),
    "nuisance_gate": NUISANCE_GATE,
    "agreement": {
        "exact": AGREEMENT_EXACT,
        "cross_fitted_targeted": AGREEMENT_CROSSFIT,
        "quantile": AGREEMENT_QUANTILE,
        "initial": INITIAL_AGREEMENT,
        "witness_multiple": WITNESS_MULTIPLE,
    },
    "not_paired": (
        "cross-fitted estimated mechanisms (declared difference D9: drtmle's fold-k reduced "
        "regression trains on a supplied out-of-fold mechanism that saw fold k); weights and "
        "clusters (adaptive_iptw takes neither); the validity laws VL0 and VL1"
    ),
}

CONFIGURATION = {
    "construction": "the gate study's (tests.studies.canonical_adaptive_iptw._estimator)",
    "n_folds": N_FOLDS,
    "g_bounds": list(gate.G_BOUNDS),
    "tol": gate.TOL,
    "max_iter": gate.MAX_ITER,
    "degree": dict(gate.DEGREE),
    "draws": {name: [draw.law, draw.n, draw.folds] for name, draw in DRAWS.items()},
    "scenarios": {name: [scenario.draw, scenario.fit] for name, scenario in SCENARIOS.items()},
}


# ------------------------------------------------------------------------------------ folds


def deal(permutation: np.ndarray, n_folds: int) -> np.ndarray:
    """The fold of each row when row ``permutation[i]`` goes to fold ``i mod n_folds``.

    This is the partition ``split(perm, rep(1:K, length = n))`` gives in R, with folds counted
    from 0.

    Parameters
    ----------
    permutation : ndarray
        A permutation of ``0..n-1``.
    n_folds : int
        The number of folds.

    Returns
    -------
    ndarray
        ``(n,)`` fold codes.
    """
    permutation = np.asarray(permutation, dtype=np.int64)
    folds = np.empty(permutation.size, dtype=np.int64)
    folds[permutation] = np.arange(permutation.size) % n_folds
    return folds


def fold_plan(draw: str, frame: pd.DataFrame, replicate: int) -> tuple[np.ndarray, np.ndarray]:
    """The fold of each row and its 1-based position in its unit's permutation.

    A ``"strata"`` draw permutes and deals each stratum of ``V`` on its own, so each stratum's
    rows are exactly the partition that stratum's ``drtmle`` call draws.
    """
    spec = DRAWS[draw]
    n = len(frame)
    rng = np.random.default_rng(stream_seed(STUDY, "pairing_folds", draw, replicate))
    folds = np.empty(n, dtype=np.int64)
    position = np.empty(n, dtype=np.int64)
    units = (
        [np.arange(n)]
        if spec.folds == "whole"
        else [
            np.flatnonzero(frame["V"].to_numpy() == level)
            for level in np.unique(frame["V"].to_numpy())
        ]
    )
    for rows in units:
        permutation = rng.permutation(rows.size)
        folds[rows] = deal(permutation, N_FOLDS)
        position[rows[permutation]] = np.arange(1, rows.size + 1)
    return folds, position


# ------------------------------------------------------------------------------ the subject


class PairingAdaptiveIPTW(AdaptiveIPTW):
    """Study-only ``AdaptiveIPTW`` with a supplied fold vector that records each key's start.

    ``assignment`` replaces the package's random partition (the seam of
    :class:`tests.studies.canonical_drtmle.FixedFoldDRTMLE`).  Every targeting problem's
    initial mechanism, initial reduced regression and lower mechanism bound are kept in
    :attr:`captured`, for the runner's ``gn``, its nuisance gate and its ``tolg``.  Nothing
    else changes.
    """

    def __init__(self, assignment: np.ndarray | None = None, **kwargs: Any) -> None:
        self._assignment = None if assignment is None else np.asarray(assignment, np.int64)
        self.captured: list[tuple[Any, np.ndarray, np.ndarray, tuple[float, float]]] = []
        super().__init__(**kwargs)

    def _folds(self, data: Any, seed: int | None = None) -> Folds:
        if self._assignment is None:
            return super()._folds(data, seed)
        if data.n != self._assignment.size:
            raise ValueError(
                f"fold assignment has {self._assignment.size} rows for a {data.n}-row fit"
            )
        folds = Folds(self._assignment.copy(), int(self._assignment.max()) + 1)
        check_integrity(folds, cluster=data.cluster)
        return folds

    def _solve_key(self, *args: Any, **kwargs: Any) -> Any:
        solved = super()._solve_key(*args, **kwargs)
        self.captured.append(
            (
                solved.spec,
                np.asarray(solved.mechanism_initial, dtype=float).copy(),
                np.asarray(solved.regression_initial, dtype=float).copy(),
                (float(solved.bounds[0]), float(solved.bounds[1])),
            )
        )
        return solved


def _estimator(
    degree: int,
    assignment: np.ndarray | None,
    *,
    treatment: Any = None,
    missingness: Any = None,
    form: str = "targeted",
    **settings: Any,
) -> PairingAdaptiveIPTW:
    """The gate's construction (:func:`tests.studies.canonical_adaptive_iptw._estimator`)."""
    learners: dict[str, Any] = {}
    if treatment is not None:
        learners["treatment_learner"] = treatment
    if missingness is not None:
        learners["missingness_learner"] = missingness
    cross = (
        {"cross_fit": True, "n_folds": N_FOLDS} if assignment is not None else {"cross_fit": False}
    )
    return PairingAdaptiveIPTW(
        assignment=assignment,
        form=form,
        reduced_outcome_learner=laws.DesignBasisGLM(degree),
        simultaneous=False,
        g_bounds=gate.G_BOUNDS,
        max_iter=gate.MAX_ITER,
        tol=gate.TOL,
        random_state=0,
        **cross,
        **learners,
        **settings,
    )


Fit = Callable[[pd.DataFrame, int, "np.ndarray | None"], tuple[Any, PairingAdaptiveIPTW]]


def _pd_known(form: str, regimes: bool) -> Fit:
    def fit(
        frame: pd.DataFrame, n: int, assignment: np.ndarray | None
    ) -> tuple[Any, PairingAdaptiveIPTW]:
        del n
        settings: dict[str, Any] = (
            {"interventions": gate.pd_regimes()}
            if regimes
            else {"estimands": list(gate.PD_ARM_KEYS)}
        )
        estimator = _estimator(gate.DEGREE["pd"], assignment, form=form, **settings)
        result = estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(laws.P_COLUMNS),
            treatment_probabilities=gate.known_mechanism("pd", frame),
        )
        return result.single(), estimator

    return fit


def _st_known(
    frame: pd.DataFrame, n: int, assignment: np.ndarray | None
) -> tuple[Any, PairingAdaptiveIPTW]:
    del n
    estimator = _estimator(gate.DEGREE["st"], assignment, estimands=list(gate.P_KEYS))
    result = estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W", "V"],
        strata=["V"],
        treatment_probabilities=gate.known_mechanism("st", frame),
    )
    return result.single(), estimator


def _p_insample(
    frame: pd.DataFrame, n: int, assignment: np.ndarray | None
) -> tuple[Any, PairingAdaptiveIPTW]:
    estimator = _estimator(
        gate.DEGREE["p"],
        assignment,
        treatment=laws.p_mechanism("correct", n),
        estimands=list(gate.P_KEYS),
    )
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=list(laws.P_COLUMNS))
    return result.single(), estimator


def _p_rate_insample(
    frame: pd.DataFrame, n: int, assignment: np.ndarray | None
) -> tuple[Any, PairingAdaptiveIPTW]:
    estimator = _estimator(
        gate.DEGREE["p"],
        assignment,
        treatment=laws.p_mechanism("rate", n),
        estimands=list(gate.P_KEYS),
    )
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=list(laws.P_COLUMNS))
    return result.single(), estimator


def _m_insample(
    frame: pd.DataFrame, n: int, assignment: np.ndarray | None, *, shrink: float = 0.0
) -> tuple[Any, PairingAdaptiveIPTW]:
    estimator = _estimator(
        gate.DEGREE["m"],
        assignment,
        treatment=laws.ShrunkMultinomial(shrink, n),
        estimands=["ey", "ate"],
        reference=multi.REFERENCE,
    )
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
    return result.single(), estimator


def _c_insample(law: str, form: str = "targeted") -> Fit:
    def fit(
        frame: pd.DataFrame, n: int, assignment: np.ndarray | None
    ) -> tuple[Any, PairingAdaptiveIPTW]:
        del n
        table = mar.LAWS[law]
        settings: dict[str, Any] = {
            "estimands": ["ey0", "ey1", "ate"] if law == "l1" else ["ey", "ate"]
        }
        if law == "l3":
            settings["reference"] = mar.REFERENCE
        estimator = _estimator(
            gate.DEGREE["c"],
            assignment,
            treatment=mar.LawTreatment(table),
            missingness=mar.LawResponse(table),
            form=form,
            **settings,
        )
        result = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        return result.single(), estimator

    return fit


def _pd_regimes_insample(
    frame: pd.DataFrame, n: int, assignment: np.ndarray | None
) -> tuple[Any, PairingAdaptiveIPTW]:
    estimator = _estimator(
        gate.DEGREE["pd"],
        assignment,
        treatment=laws.p_mechanism("rate_scaled", n),
        interventions=gate.pd_regimes(),
    )
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=list(laws.P_COLUMNS))
    return result.single(), estimator


def perturbed_response(n: int) -> np.ndarray:
    """Law C l1's response table moved by ``2 n^(-1/3) (1 + a)`` on the logit.

    A fixed table, so the cross-fitted natural-course fit trains no response model on the data
    (every fold's mechanism is the same function, as with a known mechanism), and targeting
    has a perturbation to correct.
    """
    table = mar.LAWS["l1"]
    pi = np.asarray(table.pi, dtype=float).copy()
    shift = 2.0 * n ** (-1.0 / 3.0)
    for code in range(table.arms):
        column = table.column(code)
        pi[:, column] = expit(logit(pi[:, column]) + shift * (1.0 + code))
    return pi


def _cn_crossfit(form: str) -> Fit:
    def fit(
        frame: pd.DataFrame, n: int, assignment: np.ndarray | None
    ) -> tuple[Any, PairingAdaptiveIPTW]:
        table = mar.LAWS["l1"]
        estimator = _estimator(
            gate.DEGREE["cn"],
            assignment,
            treatment=mar.LawTreatment(table),
            missingness=mar.LawResponse(table, pi=perturbed_response(n)),
            form=form,
            estimands=list(gate.CN_KEYS),
        )
        result = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        return result.single(), estimator

    return fit


def _cn_insample(
    frame: pd.DataFrame, n: int, assignment: np.ndarray | None
) -> tuple[Any, PairingAdaptiveIPTW]:
    estimator = _estimator(
        gate.DEGREE["cn"],
        assignment,
        treatment=laws.CellFrequency(),
        missingness=laws.CellFrequency(shift=gate._first_column_shift(2.0), size=2.0, n=n),
        estimands=list(gate.CN_KEYS),
    )
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
    return result.single(), estimator


def _rm(
    frame: pd.DataFrame, n: int, assignment: np.ndarray | None
) -> tuple[Any, PairingAdaptiveIPTW]:
    estimator = _estimator(
        gate.DEGREE["rm"],
        assignment,
        treatment=laws.CellFrequency(shift=gate._rule_shift, size=2.0, n=n),
        missingness=laws.CellFrequency(),
        interventions=gate.rm_regimes(),
    )
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
    return result.single(), estimator


def _ma(mode: str, form: str = "targeted") -> Fit:
    def fit(
        frame: pd.DataFrame, n: int, assignment: np.ndarray | None
    ) -> tuple[Any, PairingAdaptiveIPTW]:
        del n
        table = composite_law.TWO.law()
        settings: dict[str, Any] = (
            {"interventions": gate.ma_regimes()}
            if mode == "regime"
            else {"estimands": list(gate.MA_ARM_KEYS)}
        )
        estimator = _estimator(
            gate.DEGREE["ma"],
            assignment,
            treatment=mar.LawTreatment(table),
            missingness=mar.LawResponse(table),
            form=form,
            **settings,
        )
        result = estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_delta="DeltaA",
        )
        return result.single(), estimator

    return fit


FITS: dict[str, Fit] = {
    "pd_known_arms": _pd_known("targeted", regimes=False),
    "pd_known_arms_one_step": _pd_known("one_step", regimes=False),
    "pd_known_regimes": _pd_known("targeted", regimes=True),
    "pd_known_regimes_one_step": _pd_known("one_step", regimes=True),
    "st_known": _st_known,
    "p_insample": _p_insample,
    "p_rate_insample": _p_rate_insample,
    "m_insample": _m_insample,
    "m_rate_insample": lambda frame, n, assignment: _m_insample(frame, n, assignment, shrink=2.0),
    "c_l1_insample": _c_insample("l1"),
    "c_l3_insample": _c_insample("l3"),
    "pd_regimes_insample": _pd_regimes_insample,
    "cn_insample": _cn_insample,
    "rm": _rm,
    "ma_arms": _ma("arms"),
    "ma_regime": _ma("regime"),
    "c_l1_crossfit": _c_insample("l1"),
    "c_l1_crossfit_one_step": _c_insample("l1", "one_step"),
    "c_l3_crossfit": _c_insample("l3"),
    "c_l3_crossfit_one_step": _c_insample("l3", "one_step"),
    "cn_crossfit": _cn_crossfit("targeted"),
    "cn_crossfit_one_step": _cn_crossfit("one_step"),
    "ma_crossfit": _ma("arms"),
    "ma_crossfit_one_step": _ma("arms", "one_step"),
}


# ------------------------------------------------------------------------------------- rows


def call_label(spec: Any) -> str:
    """The runner's label of one targeting problem: an arm code, a regime or its arm, or
    the natural course."""
    if spec.kind == "arm":
        return f"a{int(spec.owner)}"
    if spec.kind == "regime":
        return str(spec.name)
    if spec.kind == "regime_arm":
        owner, _, arm = str(spec.name).partition("/")
        return f"{owner}{arm}"
    if spec.kind == "natural_course":
        return "obs"
    raise ValueError(f"unknown targeting problem kind {spec.kind!r}")


def _untargeted(result: Any) -> Mapping[str, float]:
    return {
        str(key): float(value) for key, value in result.extra["adaptive_iptw"].untargeted.items()
    }


def stratum_means(estimator: PairingAdaptiveIPTW, strata: np.ndarray) -> dict[str, float]:
    """The Horvitz-Thompson mean at the initial mechanism of each arm within each stratum.

    The fit reports these means pooled over strata only, so they are recomputed from each
    arm's indicator, target and initial mechanism, keyed as the fit keys its strata
    (``"1 | V=0"``).
    """
    out: dict[str, float] = {}
    for spec, mechanism, _, _ in estimator.captured:
        terms = np.asarray(spec.indicator) * np.asarray(spec.target) / mechanism
        for level in np.unique(strata):
            rows = strata == level
            out[f"{spec.owner_name} | V={int(level)}"] = float(np.mean(terms[rows]))
    return out


def initial_value(result: Any, key: str, extra: Mapping[str, float] | None = None) -> float:
    """The Horvitz-Thompson mean at the initial mechanism of a reported key.

    ``initial_estimate`` of a row, on the estimate's scale.  An arm key reads its code's mean
    (``ey0`` reads arm ``0``), a stratum key ``[V=s]`` the stratum's arm means, and a contrast,
    ratio or attributable fraction the same function of the means as its estimate.
    """
    means = {**_untargeted(result), **(extra or {})}
    if key in means:
        return means[key]
    stratum = ""
    stem, _, argument = key.partition("[")
    argument = argument.rstrip("]")
    if argument.startswith("V="):
        stratum, argument = f" | {argument}", ""

    def mean(label: str) -> float:
        return means[f"{label}{stratum}"]

    if stem in ("ey0", "ey1"):
        return mean(stem[-1])
    if stem in ("ate", "rr", "or") and not argument:
        one, zero = mean("1"), mean("0")
        if stem == "ate":
            return one - zero
        if stem == "rr":
            return one / zero
        return (one / (1.0 - one)) / (zero / (1.0 - zero))
    if stem in ("ey", "ey_regime"):
        return mean(argument)
    if stem in ("ate", "ate_regime"):
        left, right = argument.split(" vs ")
        return mean(left) - mean(right)
    if stem in ("par", "paf"):
        observed, unexposed = mean("ey_obs"), mean("0")
        return observed - unexposed if stem == "par" else (observed - unexposed) / observed
    raise KeyError(f"no initial value for {key!r}")


def _treatment_codes(scenario: str, frame: pd.DataFrame, estimator: PairingAdaptiveIPTW) -> Any:
    """The treatment as the package's arm codes; a string-coded arm maps through its labels."""
    treatment = frame["A"]
    labels = ARM_LABELS.get(scenario)
    if labels is None:
        return treatment.to_numpy(dtype=float)
    seen = {
        str(spec.owner_name): int(spec.owner)
        for spec, _, _, _ in estimator.captured
        if spec.kind == "arm"
    }
    if seen != {label: code for code, label in enumerate(labels)}:
        raise RuntimeError(f"{scenario}: the fit's arm codes {seen} are not ARM_LABELS {labels}")
    return treatment.map({label: float(code) for code, label in enumerate(labels)}).to_numpy(
        dtype=float
    )


def sample_rows(
    scenario: str,
    replicate: int,
    frame: pd.DataFrame,
    position: np.ndarray | None,
    estimator: PairingAdaptiveIPTW,
    result: Any,
) -> pd.DataFrame:
    """The runner's input for one fit: the raw columns, the folds, the raw mechanism factors,
    and each key's start (its mechanism for the composition check, its initial ``Qr`` for the
    nuisance gate, and its bounds)."""
    law = DRAWS[SCENARIOS[scenario].draw].law
    out = pd.DataFrame(index=range(len(frame)), columns=list(SAMPLE_COLUMNS), dtype=float)
    for column in LAW_COLUMNS[law]:
        if column != "A":
            out[column] = frame[column].to_numpy(dtype=float)
    out["A"] = _treatment_codes(scenario, frame, estimator)
    if position is not None:
        out["position"] = position.astype(float)
    nuisance = result.nuisance
    treatment = np.asarray(nuisance.propensity.values, dtype=float)
    for code in range(treatment.shape[1]):
        out[f"tr_a{code}"] = treatment[:, code]
    if nuisance.missingness is not None:
        response = np.asarray(nuisance.missingness, dtype=float)
        for code in range(response.shape[1]):
            out[f"pi_a{code}"] = response[:, code]
    if nuisance.treatment_observation is not None:
        out["pA"] = np.asarray(nuisance.treatment_observation, dtype=float)
    out["simplex"] = float(bool(nuisance.propensity.simplex))
    out["g_lo"], out["g_hi"] = (float(value) for value in result.config.g_bounds)
    out["floor"] = float(result.config.missingness_bound)
    labels: set[str] = set()
    for spec, mechanism, regression, (lower, upper) in estimator.captured:
        label = call_label(spec)
        if label in labels:
            raise RuntimeError(f"{scenario}: two targeting problems share the label {label}")
        labels.add(label)
        out[f"gn_{label}"] = mechanism
        out[f"qr_{label}"] = regression
        out[f"lb_{label}"] = lower
        out[f"ub_{label}"] = upper
    out["scenario"] = scenario
    out["replicate"] = replicate
    return out


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of the scenario's law and the exact truth of each key it reports."""
    law = DRAWS[SCENARIOS[scenario].draw].law
    every = gate.truths(law)
    return gate.draw_frame(law, n, seed), {name: every[name] for name in STUDY.scenarios[scenario]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_scenario(
    scenario: str, frame: pd.DataFrame, n: int, replicate: int
) -> tuple[Any, PairingAdaptiveIPTW, np.ndarray | None]:
    """The subject's fit of one scenario on one draw, on the draw's folds."""
    spec = SCENARIOS[scenario]
    draw = DRAWS[spec.draw]
    position: np.ndarray | None = None
    assignment: np.ndarray | None = None
    if draw.folds != "none":
        owner = SEED_OWNERS.get(scenario, scenario)
        assignment, position = fold_plan(SCENARIOS[owner].draw, frame, replicate)
    result, estimator = FITS[spec.fit](frame, n, assignment)
    return result, estimator, position


def cleverly_rows(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    extra: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    keys = STUDY.scenarios[scenario]
    return primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=keys,
        initials={key: initial_value(result, key, extra) for key in keys},
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = key
    frame, truth = drawn
    result, estimator, position = fit_scenario(scenario, frame, n, replicate)
    active = {
        key: count for key, count in result.extra["adaptive_iptw"].bound_active.items() if count
    }
    if active:
        # D2: drtmle trims a mechanism differently from the package's clip, so a fit whose
        # bounds bind is not the same computation on both sides.
        raise RuntimeError(f"{scenario}: the mechanism bounds bind at {active}")
    sample = sample_rows(scenario, replicate, frame, position, estimator, result)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    extra = (
        stratum_means(estimator, frame["V"].to_numpy())
        if DRAWS[SCENARIOS[scenario].draw].folds == "strata"
        else None
    )
    return sample, truth_rows, cleverly_rows(result, truth, scenario, replicate, extra)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication of every scenario, with the samples and truths the runner reads.

    Keys run replicate by replicate, so the samples file groups by replicate.
    """
    sizes = STUDY.sizes(n)
    keys = [
        (scenario, replicate, sizes[scenario])
        for replicate in range(replicates)
        for scenario in SCENARIOS
    ]
    # Dependent: the reference reads the subject's folds and each key's initial mechanism.
    outcomes = replication.fit_primary(
        keys,
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
        spill_samples_at=0,
    )
    samples = replication.assembled_samples(outcomes)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )


def agreement_bound(scenario: str) -> float:
    """The scenario's agreement bound: a cross-fitted targeted fit differs at order ``1/n``."""
    cross = DRAWS[SCENARIOS[scenario].draw].folds != "none"
    return AGREEMENT_CROSSFIT if cross and scenario not in ONE_STEP else AGREEMENT_EXACT


def agreement(rows: pd.DataFrame) -> pd.DataFrame:
    """The per-replicate agreement of the two implementations, one row per scenario and key.

    ``gap`` is ``|cleverly - drtmle| / SE`` on the inference scale (the log for a ratio), with
    the package's standard error; ``se_gap`` is the relative difference of the standard errors.
    ``passed`` needs the :data:`AGREEMENT_QUANTILE` quantile of both under the scenario's bound,
    a pair for every package replicate, and initial estimates within
    :data:`INITIAL_AGREEMENT`.  A :data:`DESCRIPTIVE` scenario (``verdict`` false) reports its
    gaps and needs only the pairs and the initial estimates.  ``witnessed`` is false when the mean targeting movement,
    ``|estimate - initial_estimate| / SE``, stays under :data:`WITNESS_MULTIPLE` times the
    bound: the pair then could not tell a targeting step from none.  No verdict reads it.
    """
    key = ["scenario", "replicate", "estimand"]
    subject = rows.loc[rows["implementation"] == IMPLEMENTATION].set_index(key)
    reference = rows.loc[rows["implementation"] == REFERENCE].set_index(key)
    paired = subject.join(reference, how="left", lsuffix="_c", rsuffix="_d")
    out = []
    for (scenario, estimand), group in paired.groupby(level=["scenario", "estimand"], sort=False):
        bound = agreement_bound(str(scenario))
        descriptive = scenario in DESCRIPTIVE
        se = group["std_error_c"]
        gap = (group["inference_estimate_c"] - group["inference_estimate_d"]).abs() / se
        se_gap = (group["std_error_c"] / group["std_error_d"] - 1.0).abs()
        initial = (group["initial_estimate_c"] - group["initial_estimate_d"]).abs()
        initial_scale = group["initial_estimate_c"]
        if estimand.split("[")[0] in ("rr", "or"):
            moved = (group["inference_estimate_c"] - np.log(initial_scale)).abs() / se
        else:
            moved = (group["estimate_c"] - initial_scale).abs() / se
        complete = bool(group["estimate_d"].notna().all())
        gap_q = float(gap.quantile(AGREEMENT_QUANTILE)) if complete else float("nan")
        se_q = float(se_gap.quantile(AGREEMENT_QUANTILE)) if complete else float("nan")
        initial_max = float(initial.max()) if complete else float("nan")
        out.append(
            {
                "scenario": scenario,
                "estimand": estimand,
                "replicates": len(group),
                "paired": int(group["estimate_d"].notna().sum()),
                "bound": bound,
                "gap_q99": gap_q,
                "gap_max": float(gap.max()),
                "se_gap_q99": se_q,
                "initial_gap_max": initial_max,
                "movement_mean": float(moved.mean()),
                "verdict": not descriptive,
                "passed": complete
                and initial_max <= INITIAL_AGREEMENT
                and (descriptive or (gap_q <= bound and se_q <= bound)),
                "witnessed": float(moved.mean()) >= WITNESS_MULTIPLE * bound,
            }
        )
    return pd.DataFrame(out)


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every key, by scenario."""
    out: dict[str, dict[str, float]] = {}
    for name, scenario in SCENARIOS.items():
        every = gate.truths(DRAWS[scenario.draw].law)
        out[name] = {key: every[key] for key in scenario.keys}
    return out


__all__ = [
    "DRAWS",
    "FITS",
    "SCENARIOS",
    "STUDY",
    "deal",
    "draw_and_fit",
    "fold_plan",
    "initial_value",
]
