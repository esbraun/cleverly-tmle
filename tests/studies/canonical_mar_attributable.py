r"""Registered evidence for PAR and PAF with outcomes missing at random.

The subject is ordinary TMLE's joint natural-course fit: the shipped natural-course
fluctuation and the shipped arm-mean fluctuation, solved separately from one initial fit
and reported from one stack (``docs/technical-reference/point-treatment-tmle.md``,
"Population interventions").  The comparator is R ``tmle`` 2.1.1 composed in the same way:
one population-mean fit for the natural course and one per-arm population-mean fit for each
arm mean, all with the supplied predictions of this fit.  The runner forms PAR, PAF and
their curves in R from R's own influence curves, ``fit$estimates$IC$IC.EY1``.

Seven primary scenarios, all at :data:`PRIMARY_N`:

=======================================  ====  ===========================================
scenario                                  law   fit
=======================================  ====  ===========================================
``binary_mar_attributable``               L1    in sample, the law's own nuisances
``continuous_mar_attributable``           L1    in sample, Beta outcome, ``q_bounds=(0, 1)``
``three_arm_mar_attributable``            L3    in sample, ``reference="low"``
``binary_mar_attributable_cvtmle``        L1    stacked, ten folds, depth-5 trees
``three_arm_mar_attributable_cvtmle``     L3    stacked, ten folds, depth-5 trees
``binary_mar_attributable_weighted``      L1    in sample, fixed weights of ``W``
``binary_mar_attributable_clustered``     L1    in sample, 100 clusters of 20 rows, ``id=``
=======================================  ====  ===========================================

L1 and L3 are the laws of :mod:`tests.studies.mar_arm_indexed_laws`.  The clustered sampler
draws ``W`` once per cluster from ``P(W)`` and every other column per row, so each row has the
law L1 and the rows of one cluster share ``W``.  :func:`stack_covariance` computes the exact
efficient influence covariance of every reported name on both laws.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.tree import DecisionTreeClassifier

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcomeContinuous, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.missing_outcome_study_helpers import NaturalCourseLaw
from tests.studies.point_study_helpers import primary_rows

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20262021
RESAMPLING_SEED = 20262022
NUISANCE_BOUND = 0.01
Q_BOUNDS = (0.0, 1.0)
BETA_CONCENTRATION = 24.0
N_FOLDS = 10
#: The clustered sampler: this many clusters of :data:`CLUSTER_SIZE` rows, at least 40, so
#: the fit keeps its ``influence_curve`` status.
CLUSTERS = 100
CLUSTER_SIZE = PRIMARY_N // CLUSTERS
#: The fixed analysis weight of each level of ``W``.
WEIGHT_BY_W = np.array([0.6, 1.0, 1.8])

L1 = laws.LAWS["l1"]
L3 = laws.LAWS["l3"]
#: The declared reference arm of the three-arm law.  Not the default reference ("high").
THREE_ARM_REFERENCE = "low"

BINARY = "binary_mar_attributable"
CONTINUOUS = "continuous_mar_attributable"
THREE_ARM = "three_arm_mar_attributable"
BINARY_CV = "binary_mar_attributable_cvtmle"
THREE_ARM_CV = "three_arm_mar_attributable_cvtmle"
WEIGHTED = "binary_mar_attributable_weighted"
CLUSTERED = "binary_mar_attributable_clustered"
SCENARIOS = (BINARY, CONTINUOUS, THREE_ARM, BINARY_CV, THREE_ARM_CV, WEIGHTED, CLUSTERED)
CROSS_FITTED = frozenset({BINARY_CV, THREE_ARM_CV})
THREE_ARM_SCENARIOS = frozenset({THREE_ARM, THREE_ARM_CV})

BINARY_NAMES = ("ey_obs", "ey0", "par", "paf")
CONTINUOUS_NAMES = ("ey_obs", "ey0", "par")
THREE_ARM_NAMES = (
    "ey_obs",
    *(f"ey[{label}]" for label in laws.THREE_ARM_CODES),
    f"par[{THREE_ARM_REFERENCE}]",
    f"paf[{THREE_ARM_REFERENCE}]",
)
#: What each scenario's fit requests, in the engine's vocabulary.
REQUESTS: dict[str, tuple[str, ...]] = {
    BINARY: BINARY_NAMES,
    CONTINUOUS: CONTINUOUS_NAMES,
    THREE_ARM: ("ey_obs", "ey", "par", "paf"),
    BINARY_CV: BINARY_NAMES,
    THREE_ARM_CV: ("ey_obs", "ey", "par", "paf"),
    WEIGHTED: BINARY_NAMES,
    CLUSTERED: BINARY_NAMES,
}
NAMES: dict[str, tuple[str, ...]] = {
    scenario: THREE_ARM_NAMES
    if scenario in THREE_ARM_SCENARIOS
    else CONTINUOUS_NAMES
    if scenario == CONTINUOUS
    else BINARY_NAMES
    for scenario in SCENARIOS
}


# ------------------------------------------------------------------------------ exact law


def reference_column(law: laws.Law) -> int:
    """The table column of the declared reference arm: arm ``0`` on L1, ``low`` on L3."""
    return law.labels.index(THREE_ARM_REFERENCE) if law.arms == 3 else 0


def natural_course_mean(law: laws.Law, mu: np.ndarray | None = None) -> float:
    r""":math:`\psi_{\mathrm{obs}}=\sum_w P(w)\sum_a g(a\mid w)\,m(a,w)` on the outcome scale."""
    return float(law.p_w @ (law.g * law.mean(mu)).sum(axis=1))


def _names(law: laws.Law) -> tuple[str, ...]:
    return THREE_ARM_NAMES if law.arms == 3 else BINARY_NAMES


def _arm_name(law: laws.Law, column: int) -> str:
    if law.arms == 2:
        return f"ey{column}"
    return f"ey[{law.labels[column]}]"


def values(law: laws.Law, psi_obs: float, means: np.ndarray) -> dict[str, float]:
    """Every reported name from the natural-course mean and the arm means."""
    reference = float(means[reference_column(law)])
    out = {"ey_obs": psi_obs}
    for column in range(law.arms):
        out[_arm_name(law, column)] = float(means[column])
    suffix = f"[{THREE_ARM_REFERENCE}]" if law.arms == 3 else ""
    out[f"par{suffix}"] = psi_obs - reference
    out[f"paf{suffix}"] = 1.0 - reference / psi_obs
    return {name: out[name] for name in _names(law)}


def truths(law: laws.Law) -> dict[str, float]:
    """The exact value of each reported name."""
    return values(law, natural_course_mean(law), laws.arm_means(law))


def _parent_covariance(law: laws.Law) -> np.ndarray:
    r"""Covariance of the curves of ``(psi_obs, psi_a for each table column)``.

    Every curve is a residual term plus a plug-in term.  Within a ``(w, a)`` cell the
    residual has mean zero given the cell, so

    .. math::

        E[D_iD_j] = \sum_{w,a} P(w)g(a\mid w)\bigl[\pi\,\sigma^2 c_ic_j + P_iP_j\bigr](a,w),

    with :math:`c_{\mathrm{obs}}=1/\pi`, :math:`c_b=1\{a=b\}/(g_b\pi_b)`,
    :math:`P_{\mathrm{obs}}=m(a,w)-\psi_{\mathrm{obs}}` and :math:`P_b=m_b(w)-\psi_b`.
    """
    mean = law.mean()
    variance = law.variance()
    mass = law.p_w[:, None] * law.g
    psi_obs = natural_course_mean(law)
    psi = laws.arm_means(law)
    k = law.arms
    residual = [1.0 / law.pi] + [
        np.where(np.arange(k)[None, :] == b, 1.0 / (law.g[:, [b]] * law.pi[:, [b]]), 0.0)
        for b in range(k)
    ]
    plug_in = [mean - psi_obs] + [
        np.broadcast_to(mean[:, [b]] - psi[b], mean.shape) for b in range(k)
    ]
    size = k + 1
    out = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            out[i, j] = float(
                np.sum(
                    mass * (law.pi * variance * residual[i] * residual[j] + plug_in[i] * plug_in[j])
                )
            )
    return out


def _jacobian(law: laws.Law) -> np.ndarray:
    """Rows: each reported name's gradient in ``(psi_obs, psi_a ...)``."""
    psi_obs = natural_course_mean(law)
    reference = reference_column(law)
    psi_ref = float(laws.arm_means(law)[reference])
    rows = []
    for name in _names(law):
        row = np.zeros(law.arms + 1)
        if name == "ey_obs":
            row[0] = 1.0
        elif name.startswith("par"):
            row[0], row[1 + reference] = 1.0, -1.0
        elif name.startswith("paf"):
            row[0] = psi_ref / psi_obs**2
            row[1 + reference] = -1.0 / psi_obs
        else:
            column = next(index for index in range(law.arms) if _arm_name(law, index) == name)
            row[1 + column] = 1.0
        rows.append(row)
    return np.vstack(rows)


def stack_covariance(law: laws.Law) -> np.ndarray:
    """The exact efficient influence covariance of every reported name, in report order."""
    jacobian = _jacobian(law)
    return np.asarray(jacobian @ _parent_covariance(law) @ jacobian.T, dtype=float)


def efficiency_sd(law: laws.Law, name: str) -> float:
    """The standard deviation of one reported name's efficient influence curve."""
    index = _names(law).index(name)
    return float(np.sqrt(stack_covariance(law)[index, index]))


def natural_course_limit(
    law: laws.Law, *, mu: np.ndarray | None = None, pi: np.ndarray | None = None
) -> float:
    r"""The large-sample limit of the natural-course fluctuation under working nuisances.

    The logistic fluctuation uses every respondent and the clever covariate
    :math:`1/\tilde\pi`.  Its limit solves
    :math:`\sum_{w,a}P(w)g\pi\,\tilde H\{\mu-\operatorname{expit}(\operatorname{logit}
    \tilde\mu+\epsilon\tilde H)\}=0`, and the mean is
    :math:`\sum_{w,a}P(w)g\operatorname{expit}(\ldots)`.
    """
    working_mu = law.mu if mu is None else mu
    clever = 1.0 / (law.pi if pi is None else pi)
    mass = law.p_w[:, None] * law.g
    offset = logit(working_mu)

    def score(epsilon: float) -> float:
        fitted = expit(offset + epsilon * clever)
        return float(np.sum(mass * law.pi * clever * (law.mu - fitted)))

    epsilon = brentq(score, -50.0, 50.0, xtol=1e-15, rtol=1e-15)
    scaled = float(np.sum(mass * expit(offset + epsilon * clever)))
    return law.offset + law.span * scaled


def stack_limit(
    law: laws.Law,
    *,
    mu: np.ndarray | None = None,
    g: np.ndarray | None = None,
    pi: np.ndarray | None = None,
) -> dict[str, float]:
    """The large-sample limit of every reported name under fixed working nuisances."""
    arms = laws.targeted_limit(law, mu=mu, g=g, pi=pi)
    means = np.array(
        [
            arms[f"ey{column}" if law.arms == 2 else f"ey[{law.labels[column]}]"]
            for column in range(law.arms)
        ]
    )
    return values(law, natural_course_limit(law, mu=mu, pi=pi), means)


WEIGHT_CELLS = mar.cell_weights(lambda w, a, k: WEIGHT_BY_W[w])
TRUTHS: dict[str, dict[str, float]] = {
    BINARY: truths(L1),
    CONTINUOUS: {name: truths(L1)[name] for name in CONTINUOUS_NAMES},
    THREE_ARM: truths(L3),
    BINARY_CV: truths(L1),
    THREE_ARM_CV: truths(L3),
    WEIGHTED: {
        name: float(mar.weighted_functional(mar.PROBS, name, WEIGHT_CELLS)) for name in BINARY_NAMES
    },
    CLUSTERED: truths(L1),
}
EFFICIENCY_SD = {name: efficiency_sd(L1, name) for name in ("par", "paf")}


# ------------------------------------------------------------------------------ the record

STUDY = StudyRecord(
    name="missing-outcome attributable-effect TMLE",
    slug="mar-attributable-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_mar_attributable",
    document="docs/technical-reference/method-evidence/missing-outcome-attributable-effects-tmle.md",
    anchor="missing-outcome-attributable-effect-tmle",
    scenarios=NAMES,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    nuisance_count=3,
    # The stacked scenarios refit the in-sample scenario's samples, so the in-sample versus
    # stacked comparison is paired.  The weighted scenario's weights are a function of W,
    # so it reads the same L1 samples too.
    scenario_seed_owners={
        BINARY_CV: BINARY,
        THREE_ARM_CV: THREE_ARM,
        WEIGHTED: BINARY,
    },
    margins=Margins(),
    implementation="cleverly-mar-attributable-tmle",
    reference="tmle-r-composed-attributable",
    modules=(
        "tests/studies/canonical_mar_attributable.py",
        "tests/studies/mar_attributable_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/studies/mar_arm_indexed_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law_mar.py",
        "tests/discrete_law_multi.py",
        "tests/conftest.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_mar_attributable",
    properties_module="tests.studies.mar_attributable_properties",
    property_cells={
        "mar_robustness": (
            "both_correct",
            "outcome_correct",
            "mechanisms_correct",
            "treatment_wrong",
            "observation_wrong",
            "product_only",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "par__correctly_specified",
            "paf__correctly_specified",
            "par__shrunken_se_control",
            "par__noise_control",
            "par__inflated_se_control",
            "paf__inflated_se_control",
        ),
        "targeting_necessity": ("par__targeted", "par__untargeted"),
        "missingness_necessity": (
            "par__declared",
            "par__complete_case_control",
            "paf__complete_case_control",
        ),
        "simultaneous_coverage": (
            "attributable_binary__simultaneous_band",
            "attributable_binary__pointwise_joint_control",
            "attributable_three_arm__simultaneous_band",
            "attributable_three_arm__pointwise_joint_control",
            "attributable_binary_cvtmle__simultaneous_band",
            "attributable_binary_cvtmle__pointwise_joint_control",
        ),
    },
    efficiency_bounds=EFFICIENCY_SD,
    extra_artifacts=("scale-probe.csv",),
)

TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_inference": (
        "R tmle population-mean fits composed by the delta method from R's own influence "
        "curves; centered n-1 sample variance over rows, or over cluster means with id="
    ),
}

CONFIGURATION = {
    "construction": (
        "joint natural-course TMLE: the shipped natural-course fluctuation and the shipped "
        "arm-mean fluctuation, solved separately from one initial fit"
    ),
    "source": (
        "Diaz, Carone, and van der Laan (2016), Section 2, equations 1-5; Diaz and van der "
        "Laan (2017), Section 2.1 and equation 1; Hubbard and van der Laan (2008), equation 2"
    ),
    "fluctuation": "logistic",
    "targeting": "iterative",
    "target_weights": False,
    "simultaneous_intervals": False,
    "missingness_bound": NUISANCE_BOUND,
    "in_sample_nuisances": "the law's own outcome, treatment and response tables",
    "stacked_nuisances": (
        "ten package-generated folds; separate depth-five classification trees with minimum "
        "leaf size 25 for the outcome regression, the treatment mechanism and the response "
        "mechanism"
    ),
    "three_arm_reference": THREE_ARM_REFERENCE,
    "continuous_outcome": f"beta with concentration {BETA_CONCENTRATION:g}, q_bounds=(0, 1)",
    "weights": f"fixed weights {WEIGHT_BY_W.tolist()} by level of W",
    "clusters": f"{CLUSTERS} clusters of {CLUSTER_SIZE} rows that share W",
    "reference_arguments": (
        "natural course: A=rep(1, n), W=data.frame(A_original, W), Q=cbind(qn, qn), "
        "g1W=rep(1, n), pDelta1=cbind(pin, pin); each arm a: A=rep(1, n), "
        "Delta=1{A = a} Delta, Q=cbind(q_a, q_a), g1W=g_a, pDelta1=cbind(pi_a, pi_a); "
        f"fluctuation=logistic, Qbounds=c(0, 1), gbound={NUISANCE_BOUND}, alpha=0.9995, "
        "cvQinit=FALSE, prescreenW.g=FALSE, target.gwt=FALSE, B=1, with obsWeights= and "
        "id= on the weighted and clustered scenarios"
    ),
    "reference_scale_workaround": (
        "R tmle takes the continuous outcome range from every non-NA Y, so each continuous "
        "fit sets Y to the two q_bounds on two rows that the fit's response indicator "
        "excludes; scale-probe.csv records the exact-equality probe of that workaround on "
        "both R paths of every continuous replication"
    ),
    "comparator_search": (
        "R tmle 2.1.1 reports no PAR or PAF. Used its population-mean path once for the "
        "natural course and once per arm with the response indicator 1{A = a} Delta, which "
        "is the arm-mean clever covariate 1{A = a} Delta / (g_a pi_a). Both fits use this "
        "fit's supplied predictions, and the runner forms PAR and PAF from R's own "
        "influence curves. The comparison conditions on those predictions. Rejected tmle3 "
        "0.2.0 at ed72f8a, whose tmle_PAR uses Y - psi with no response score; rejected "
        "zEpid 0.9.1, which targets inside each fold."
    ),
}


# ------------------------------------------------------------------------------ sampling


def _sample_continuous(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.choice(len(mar.P_W), size=n, p=mar.P_W)
    a = rng.binomial(1, mar.G[w])
    delta = rng.binomial(1, mar.PI[w, a])
    mean = mar.Q[w, a]
    y = rng.beta(mean * BETA_CONCENTRATION, (1.0 - mean) * BETA_CONCENTRATION)
    return pd.DataFrame(
        {
            "W": w.astype(float),
            "A": a.astype(float),
            "Y": np.where(delta == 1, y, np.nan),
            "Delta": delta.astype(float),
        }
    )


def _sample_clustered(n: int, seed: int) -> pd.DataFrame:
    """L1 rows in clusters that share ``W``; every other column is drawn per row."""
    if n % CLUSTER_SIZE:
        raise ValueError(f"n = {n} is not a multiple of the cluster size {CLUSTER_SIZE}")
    rng = np.random.default_rng(seed)
    clusters = n // CLUSTER_SIZE
    level = rng.choice(len(mar.P_W), size=clusters, p=mar.P_W)
    w = np.repeat(level, CLUSTER_SIZE)
    a = rng.binomial(1, mar.G[w])
    delta = rng.binomial(1, mar.PI[w, a])
    y = rng.binomial(1, mar.Q[w, a]).astype(float)
    return pd.DataFrame(
        {
            "W": w.astype(float),
            "A": a.astype(float),
            "Y": np.where(delta == 1, y, np.nan),
            "Delta": delta.astype(float),
            "cluster": np.repeat(np.arange(clusters), CLUSTER_SIZE),
        }
    )


def law_of(scenario: str) -> laws.Law:
    return L3 if scenario in THREE_ARM_SCENARIOS else L1


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIOS:
        raise KeyError(scenario)
    if scenario == CONTINUOUS:
        frame = _sample_continuous(n, seed)
    elif scenario == CLUSTERED:
        frame = _sample_clustered(n, seed)
    else:
        frame = laws.sample(law_of(scenario), n, seed)
    if scenario == WEIGHTED:
        frame = frame.assign(weight=WEIGHT_BY_W[frame["W"].to_numpy(dtype=int)])
    return frame, dict(TRUTHS[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ------------------------------------------------------------------------------ fitting


def flexible_learner(seed: int) -> DecisionTreeClassifier:
    """The predeclared data-adaptive learner of every stacked nuisance."""
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=seed)


def oracle_learners(
    scenario: str,
    *,
    mu: np.ndarray | None = None,
    g: np.ndarray | None = None,
    pi: np.ndarray | None = None,
) -> dict[str, Any]:
    """The law-table learners of an in-sample scenario; a table argument replaces the law's."""
    if scenario == CONTINUOUS:
        law = NaturalCourseLaw()
        return {
            "outcome_learner": OracleOutcomeContinuous(law),
            "treatment_learner": OracleTreatment(law),
            "missingness_learner": OracleMissingness(law),
        }
    law = law_of(scenario)
    return {
        "outcome_learner": laws.LawOutcome(law, mu),
        "treatment_learner": laws.LawTreatment(law, g),
        "missingness_learner": laws.LawResponse(law, pi),
    }


def fit_cleverly(
    frame: pd.DataFrame,
    scenario: str,
    *,
    simultaneous: bool = False,
    learners: Mapping[str, Any] | None = None,
    request: tuple[str, ...] | None = None,
    delta: bool = True,
) -> Any:
    """The subject fit of ``scenario``, or a property fit with replaced learners."""
    cross_fit = scenario in CROSS_FITTED
    if learners is None:
        learners = (
            {
                "outcome_learner": flexible_learner(0),
                "treatment_learner": flexible_learner(2),
                "missingness_learner": flexible_learner(1),
            }
            if cross_fit
            else oracle_learners(scenario)
        )
    roles: dict[str, Any] = {}
    if delta:
        roles["delta"] = "Delta"
    if "weight" in frame.columns:
        roles["weights"] = "weight"
    if "cluster" in frame.columns:
        roles["id"] = "cluster"
    settings: dict[str, Any] = {
        "estimands": REQUESTS[scenario] if request is None else request,
        "cross_fit": cross_fit,
        "fluctuation": "logistic",
        "targeting": "iterative",
        "target_weights": False,
        "simultaneous": simultaneous,
        "nuisance_bound": NUISANCE_BOUND,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
        **dict(learners),
    }
    if not delta:
        settings.pop("missingness_learner")
    if cross_fit:
        settings.update(n_folds=N_FOLDS, repeats=1, stratify_folds="none")
    if scenario == CONTINUOUS:
        settings["q_bounds"] = Q_BOUNDS
    if scenario in THREE_ARM_SCENARIOS:
        settings["reference"] = THREE_ARM_REFERENCE
    return (
        TMLE(**settings).fit(frame, outcome="Y", treatment="A", covariates=["W"], **roles).single()
    )


def initial_estimates(result: Any, scenario: str) -> dict[str, float]:
    """The untargeted plug-in of every reported name, from the initial outcome fit."""
    weights = np.asarray(result.data.weights, dtype=float)
    scaler = result.nuisance.scaler
    outcome = result.nuisance.outcome
    psi_obs = float(
        np.average(np.asarray(scaler.unscale_levels(outcome.observed)), weights=weights)
    )
    law = law_of(scenario)
    means = np.empty(law.arms)
    for code, arm in enumerate(result.data.arm_codes):
        column = law.labels.index(law.codes[code]) if law.arms == 3 else int(arm)
        means[column] = float(
            np.average(np.asarray(scaler.unscale_levels(outcome.arms[arm])), weights=weights)
        )
    reported = values(law, psi_obs, means)
    return {name: reported[name] for name in NAMES[scenario]}


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    result: Any | None = None,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario) if result is None else result
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=NAMES[scenario],
        initials=initial_estimates(result, scenario),
    )


def reference_sample(
    frame: pd.DataFrame, result: Any, *, scenario: str, replicate: int
) -> pd.DataFrame:
    """The supplied predictions both R paths target, one row per sample row.

    ``qn`` and ``pin`` are the initial outcome and response predictions at each row's own
    arm.  ``q<k>``, ``g<k>`` and ``pi<k>`` are the outcome, treatment and response predictions
    at arm code ``k``; ``A`` carries the arm code and ``ref`` the reference code.  The payload
    refuses a fit whose response or product bound binds, because R would then target a
    different nuisance.
    """
    data = result.data
    nuisance = result.nuisance
    response = np.asarray(nuisance.missingness_at_realised_arm(data.treatment), dtype=float)
    if np.any(response <= NUISANCE_BOUND):
        raise AssertionError(f"{scenario}/{replicate}: the response bound binds")
    scaler = nuisance.scaler
    sample = pd.DataFrame(
        {
            "scenario": scenario,
            "replicate": replicate,
            "W": frame["W"].to_numpy(dtype=float),
            "A": np.asarray(data.treatment, dtype=float),
            "Y": frame["Y"].to_numpy(dtype=float),
            "Delta": frame["Delta"].to_numpy(dtype=float),
            "weight": np.asarray(data.weights, dtype=float),
            "cluster": (
                frame["cluster"].to_numpy(dtype=int)
                if "cluster" in frame.columns
                else np.arange(len(frame))
            ),
            "ref": float(result.config.reference_arm),
            "qn": np.asarray(scaler.unscale_levels(nuisance.outcome.observed), dtype=float),
            "pin": response,
        }
    )
    missingness = np.asarray(nuisance.missingness, dtype=float)
    for index, arm in enumerate(data.arm_codes):
        code = int(arm)
        product = nuisance.propensity.arm(arm) * missingness[:, index]
        if np.any(product <= NUISANCE_BOUND):
            raise AssertionError(f"{scenario}/{replicate}: the product bound binds at {arm}")
        sample[f"q{code}"] = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[arm]))
        sample[f"g{code}"] = np.asarray(nuisance.propensity.arm(arm), dtype=float)
        sample[f"pi{code}"] = missingness[:, index]
    return sample


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    sample = reference_sample(frame, result, scenario=scenario, replicate=replicate)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, cleverly_rows(frame, truth, scenario, replicate, result=result)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((scenario, replicate, n),) for scenario in SCENARIOS for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]


# ------------------------------------------------------------------ scale-workaround probe

#: The probe runner.  Not named ``run_*.R``, because
#: ``tests/unit/test_canonical_runner_parity.py`` holds every such file to the published-row
#: contract, and the probe publishes a check table instead.
PROBE_RUNNER = "tmle_mar_attributable/probe_scale_workaround.R"
PROBE_ARTIFACT = "scale-probe.csv"
PROBE_PATHS = ("natural_course", "reference_arm")


def reference_artifacts(
    *,
    reference: Any,
    here: Path,
    samples: Path,
    truths_path: Path,
    reference_results: Path,
    output: Path,
    cores: int,
) -> dict[str, pd.DataFrame]:
    """Run the exact-equality probe of the planted-row workaround on every continuous fit."""
    path = output / PROBE_ARTIFACT
    reference.run(here, samples, truths_path, path, cores=cores, runner=PROBE_RUNNER)
    probe = pd.read_csv(path).sort_values(["replicate", "path"], ignore_index=True)
    reference_rows = pd.read_csv(reference_results, usecols=["scenario", "replicate"])
    expected = sorted(
        set(reference_rows.loc[reference_rows["scenario"] == CONTINUOUS, "replicate"])
    )
    for name in PROBE_PATHS:
        probed = sorted(probe.loc[probe["path"] == name, "replicate"])
        if probed != expected:
            raise RuntimeError(f"the {name} probe does not match the reference replications")
    return {PROBE_ARTIFACT: probe}


def scientific_failures(extra_frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Refuse publication when the continuous comparison's scale workaround is not exact."""
    probe = extra_frames[PROBE_ARTIFACT]
    if probe.empty:
        return {"scale-workaround probe coverage": pd.DataFrame([{"error": "no probe rows"}])}
    count = len(probe) // len(PROBE_PATHS)
    for name in PROBE_PATHS:
        replicates = sorted(int(value) for value in probe.loc[probe["path"] == name, "replicate"])
        if set(probe["scenario"]) != {CONTINUOUS} or replicates != list(range(count)):
            return {"scale-workaround probe coverage": probe}
    return {"scale-workaround probe": probe.loc[~probe["passed"].fillna(False).eq(True)]}
