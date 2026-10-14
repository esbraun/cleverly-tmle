r"""Registered evidence for repeated cross-fitting of longitudinal fits (X46).

A longitudinal fit with ``repeats=R`` draws ``R`` outer splits, fits every draw completely, and
reports the coordinatewise median of the draw estimates with the split-adjusted median
variance of Chernozhukov et al. (2018), Definition 3.3 and equation (3.14).  This study
measures that report on five finite-support laws with exact truths.  Every nuisance is a
saturated cell mean (:class:`~tests.discrete_law_longitudinal.CellMeans`), refitted in every
fold of every draw, with five outer folds, two learner folds, ``g_bounds = (0.01, 1)`` and
three draws.

====  ====================================================  =========  =======================
law   source                                                fit        estimands
====  ====================================================  =========  =======================
D     :mod:`tests.discrete_law_longitudinal`, two censored  ``LTMLE``  three regimen means, two
      decisions, sampled by                                 and        contrasts against
      :func:`tests.studies.ltmle_crossfit_properties.       ``SDR``    ``never`` and the
      sample`                                                          derived risk ratio
I     law ``A`` of                                          ``IPSI``   odds multipliers 0.5,
      :mod:`tests.discrete_law_longitudinal_incremental`                1 and 2 and two
                                                                       contrasts
S     law ``D`` of the incremental oracle, survival over    ``LTMLE``  risks and risk
      three nodes                                                      differences at 1-3
C     law ``Ck`` of the incremental oracle, clusters of 10  ``LTMLE``  static means and their
                                                                       contrast
M     law D with the working model of                       ``LTMLE``  the two coefficients
      :mod:`tests.studies.crossfit_longitudinal_msm_                   of the centred model
      properties`, duration centred at :data:`M_STAR`
====  ====================================================  =========  =======================

The primary study holds laws D, S, C and M at :data:`PRIMARY_N`.  Law I needs n = 40,000 for
its second-order bias (``X46-plan.md`` 5.4), and a study row has one size, so its primary
rows are the ``ipsi_primary`` property family of
:mod:`tests.studies.longitudinal_repeated_crossfit_properties`, read by the same three rules.

**Law M's centred duration.**  Under the declared projection weights the two coefficient
estimators of law M correlate at -0.941, so a pointwise pair covers jointly about 0.934 and a
band control has no power.  The duration term is centred at
:math:`m^* = -\operatorname{cov}(D_0, D_1) / \operatorname{var}(D_1)`, from the exact influence
curves of the two coefficients.  That sets the correlation of the intercept and slope
estimators to zero.  The slope does not move under a shift of the covariate.  The intercept
becomes :math:`b_0 + m^* b_1`, and :func:`msm_truth` evaluates it through this module's own
projection on the shifted durations.

No comparator is run.  The pinned ``lmtp`` and ``npcausal`` adapters accept a supplied fold
list, so per-draw parity is the evidence of the single-draw rows; running them once per draw
and applying this package's median would compare the aggregation with itself.  The study
writes the zero-row equivalence artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import IPSI, LTMLE, SDR
from tests import discrete_law_longitudinal as dlaw
from tests import discrete_law_longitudinal_incremental as ilaw
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_incremental_common as common
from tests.studies import longitudinal_msm_properties as msm_law
from tests.studies.canonical_longitudinal_msm import declared_msm
from tests.studies.evidence import replication
from tests.studies.evidence.constructions import POOLED_LONGITUDINAL_CROSS_FIT
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.ltmle_crossfit_properties import REGIMENS as D_REGIMENS
from tests.studies.ltmle_crossfit_properties import sample as sample_law_d

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 8_000
SEED = 20261446
RESAMPLING_SEED = 2026144601
N_FOLDS = 5
LEARNER_FOLDS = 2
REPEATS = 3
#: The shipped default.  No true node probability of law D is below 0.25 and no cumulative
#: product below 0.0156, so the lower bound never binds at the truth; the design test checks
#: this on laws D, S, C and I.  A lower bound of 1e-8 turned an empty training cell into a
#: weight of 1e8 once in 3,000 pilot draws (``X46-plan.md`` 5.2).
G_BOUNDS = (0.01, 1.0)
#: The size of law I, the incremental law, for its primary and calibration cells.
IPSI_N = 40_000
#: The cluster size of law C: n / 10 clusters, far above the few-cluster threshold of 40.
CLUSTER_SIZE = 10
#: The convergence settings every ``LTMLE`` fit uses.  ``SDR`` and ``IPSI`` take neither.
LTMLE_ITERATION: dict[str, Any] = {"max_iter": 100, "tol": 1e-10}

# ------------------------------------------------------------------------------ law D

D_REFERENCE = "never"
D_MEANS = tuple(f"ey_regimen[{label}]" for label in D_REGIMENS)
D_CONTRASTS = {
    "static": "ate_regimen[always vs never]",
    "dynamic": "ate_regimen[treat_if_l2 vs never]",
}
D_RATIO = "rr_regimen[always vs never]"
D_NAMES = (*D_MEANS, *D_CONTRASTS.values(), D_RATIO)
D_TRUTH = {
    **{
        name: float(dlaw.functional(dlaw.PROBS, name)) for name in (*D_MEANS, *D_CONTRASTS.values())
    },
    D_RATIO: float(
        dlaw.functional(dlaw.PROBS, "ey_regimen[always]")
        / dlaw.functional(dlaw.PROBS, "ey_regimen[never]")
    ),
}
D_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}

# ------------------------------------------------------------------------------ law I

I_PLANS = common.constant_plans(ilaw.DELTAS)
I_REFERENCE = ilaw.REFERENCE
I_NAMES = common.names(I_PLANS, I_REFERENCE)
I_TRUTH = common.truths(ilaw.LAW_A, I_PLANS, I_NAMES)
I_CONTRASTS = {
    "contrast_x2": f"ate_regimen[odds x2 vs {I_REFERENCE}]",
    "contrast_x0_5": f"ate_regimen[odds x0.5 vs {I_REFERENCE}]",
}

# -------------------------------------------------------------------------- laws S and C

ARMS: dict[str, common.Plan] = {
    "always": tuple(ilaw.arm(1) for _ in range(3)),
    "never": tuple(ilaw.arm(0) for _ in range(3)),
}
STATIC_REFERENCE = "never"
HORIZONS = (1, 2, 3)
S_NAMES = common.names(ARMS, STATIC_REFERENCE, horizons=HORIZONS)
S_TRUTH = common.truths(ilaw.LAW_D, ARMS, S_NAMES)
C_NAMES = common.names(ARMS, STATIC_REFERENCE)
C_TRUTH = common.truths(ilaw.LAW_CK.marginal, ARMS, C_NAMES)


def s_contrast(horizon: int) -> str:
    """The risk difference of always against never at ``horizon`` on law S."""
    return f"ate_regimen[always vs never @ t={horizon}]"


# ------------------------------------------------------------------------------ law M

M_TERMS = ("(intercept)", "duration")
M_NAMES = tuple(f"msm_regimen[{term}]" for term in M_TERMS)


def _centring() -> float:
    """:math:`m^*`, which makes the exact intercept and slope curves uncorrelated."""
    curves = msm_law.influence_curves()
    covariance = float(np.sum(dlaw.PROBS * curves[:, 0] * curves[:, 1]))
    variance = float(np.sum(dlaw.PROBS * curves[:, 1] ** 2))
    return -covariance / variance


#: The centre of the duration term (1.2304 on this law).
M_STAR = _centring()
M_DURATION = {label: float(value) - M_STAR for label, value in msm_law.DURATION.items()}


def msm_truth() -> dict[str, float]:
    """Both coefficients of the centred working model, by this module's own projection."""
    labels = tuple(msm_law.LABELS)
    design = np.column_stack([np.ones(len(labels)), [M_DURATION[label] for label in labels]])
    mass = np.array([msm_law.PROJECTION_WEIGHT[label] for label in labels])
    means = np.array([dlaw.functional(dlaw.PROBS, f"ey_regimen[{label}]") for label in labels])
    coefficients = np.linalg.solve(design.T @ (mass[:, None] * design), design.T @ (mass * means))
    return dict(zip(M_NAMES, (float(value) for value in coefficients), strict=True))


M_TRUTH = msm_truth()

# --------------------------------------------------------------------------- the study

SCENARIOS: dict[str, tuple[str, ...]] = {
    "law_d_ltmle": D_NAMES,
    "law_d_sdr": D_NAMES,
    "law_s": S_NAMES,
    "law_c": C_NAMES,
    "law_m": M_NAMES,
}
TRUTHS: dict[str, dict[str, float]] = {
    "law_d_ltmle": D_TRUTH,
    "law_d_sdr": D_TRUTH,
    "law_s": S_TRUTH,
    "law_c": C_TRUTH,
    "law_m": M_TRUTH,
}

#: The calibration labels of the properties module, by fit set.
CALIBRATION_LABELS = (
    "d_ltmle_static",
    "d_ltmle_dynamic",
    "d_sdr_static",
    "d_sdr_dynamic",
    "i_contrast_x2",
    "i_contrast_x0_5",
    "s_contrast_h2",
    "c_static",
    "m_duration",
)
#: The joint labels of the ``simultaneous_coverage`` family.
BAND_LABELS = ("d_ltmle", "d_sdr", "i_ipsi", "s_ltmle", "c_ltmle", "m_ltmle")
#: The labels of the ``repeat_stability`` family.
STABILITY_LABELS = (
    "d_ltmle_static",
    "d_ltmle_dynamic",
    "d_sdr_static",
    "d_sdr_dynamic",
    "i_contrast_x2",
)
#: The labels of the ``ipsi_primary`` family: law I's primary rows at :data:`IPSI_N`.
IPSI_PRIMARY_LABELS = {
    "mean_x0_5": "ey_regimen[odds x0.5]",
    "mean_x1": f"ey_regimen[{I_REFERENCE}]",
    "mean_x2": "ey_regimen[odds x2]",
    "contrast_x0_5": I_CONTRASTS["contrast_x0_5"],
    "contrast_x2": I_CONTRASTS["contrast_x2"],
}
RATE_SIZES = (2_000, 8_000, 32_000)
RATE_LABELS = ("ltmle_static", "sdr_static")

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control")
    ),
    "simultaneous_coverage": tuple(
        f"{label}__{kind}"
        for label in BAND_LABELS
        for kind in ("simultaneous_band", "pointwise_joint_control")
    ),
    "repeat_stability": tuple(
        f"{label}__{kind}"
        for label in STABILITY_LABELS
        for kind in ("three_repeats", "one_repeat_control")
    ),
    "repeat_increment": (
        *(f"h{h}__{kind}" for h in HORIZONS for kind in ("repeat_median", "single_draw")),
        "h3__inflated_se_control",
    ),
    "single_draw_reading": tuple(f"{label}__draw_0" for label in CALIBRATION_LABELS),
    "split_displacement": tuple(f"{label}__share" for label in CALIBRATION_LABELS),
    "root_n_and_efficiency": tuple(
        f"{label}__n_{size}" for label in RATE_LABELS for size in RATE_SIZES
    ),
    "root_n_rate": tuple(
        f"{label}__{statistic}"
        for label in RATE_LABELS
        for statistic in ("empirical_sd", "reported_se")
    ),
    "ipsi_primary": tuple(f"{label}__repeated" for label in IPSI_PRIMARY_LABELS),
}

STUDY = StudyRecord(
    name="repeated cross-fitted longitudinal TMLE, SDR and IPSI",
    slug="longitudinal-repeated-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "longitudinal_repeated_crossfit",
    document="docs/technical-reference/method-evidence/repeated-cross-fitted-longitudinal.md",
    anchor="repeated-cross-fitted-longitudinal-tmle-sdr-and-ipsi",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={"law_d_sdr": "law_d_ltmle"},
    margins=Margins(),
    implementation="cleverly-repeated-longitudinal",
    reference=None,
    modules=(
        "tests/studies/longitudinal_repeated_crossfit.py",
        "tests/studies/longitudinal_repeated_crossfit_properties.py",
        "tests/studies/longitudinal_incremental_common.py",
        "tests/studies/longitudinal_msm_properties.py",
        "tests/studies/canonical_longitudinal_msm.py",
        "tests/studies/ltmle_crossfit_properties.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/ltmle_properties.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_incremental.py",
        "tests/longitudinal_ipsi.py",
    ),
    runner_module="tests.studies.longitudinal_repeated_crossfit",
    properties_module="tests.studies.longitudinal_repeated_crossfit_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": POOLED_LONGITUDINAL_CROSS_FIT,
    "cross_fit": True,
    "outer_folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "repeats": REPEATS,
    "g_bounds": list(G_BOUNDS),
    "learners": "saturated cell means in every slot, refitted in every fold of every draw",
    "repeat_aggregation": (
        "coordinatewise median point; median of within-draw variance plus squared split "
        "displacement (Chernozhukov et al. 2018, Definition 3.3, equation 3.14); ratios on the "
        "log scale; bands by the median of the per-draw critical values"
    ),
    "random_state": "0 in every primary and property fit except repeat stability",
    "laws": {
        "law_d": "tests/discrete_law_longitudinal.py, two censored decisions",
        "law_i": f"incremental law A at n = {IPSI_N} (ipsi_primary and calibration)",
        "law_s": "incremental survival law D, horizons 1 to 3",
        "law_c": f"incremental clustered law Ck, clusters of {CLUSTER_SIZE}, id=",
        "law_m": f"law D, identity MSM with duration centred at {M_STAR:.6f}",
    },
}


# ------------------------------------------------------------------------- the fits


def _cell_means() -> dict[str, Any]:
    return {
        "outcome_learner": dlaw.CellMeans(),
        "pseudo_learner": dlaw.CellMeans(),
        "treatment_learner": dlaw.CellMeans(),
    }


def fit_law_d(
    frame: pd.DataFrame,
    estimator: str,
    *,
    repeats: int = REPEATS,
    random_state: int = 0,
    simultaneous: bool = False,
) -> Any:
    """``LTMLE`` or ``SDR`` on law D: three regimens, reference ``never``."""
    cls = {"ltmle": LTMLE, "sdr": SDR}[estimator]
    extra = LTMLE_ITERATION if cls is LTMLE else {}
    return cls(
        D_REGIMENS,
        reference=D_REFERENCE,
        **_cell_means(),
        censoring_learner=dlaw.CellMeans(),
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        simultaneous=simultaneous,
        random_state=random_state,
        repeats=repeats,
        **extra,
    ).fit(frame, **D_COLUMNS)


def fit_law_i(
    frame: pd.DataFrame,
    *,
    repeats: int = REPEATS,
    random_state: int = 0,
    simultaneous: bool = False,
) -> Any:
    """``IPSI`` on law A: three constant odds multipliers, reference the natural course."""
    return IPSI(
        common.regimens(I_PLANS),
        reference=I_REFERENCE,
        **_cell_means(),
        censoring_learner=None,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        q_bounds=common.Q_BOUNDS,
        simultaneous=simultaneous,
        random_state=random_state,
        repeats=repeats,
    ).fit(frame, **ilaw.columns(ilaw.LAW_A))


def fit_static(
    frame: pd.DataFrame,
    target: Any,
    *,
    horizons: tuple[int, ...] | None = None,
    repeats: int = REPEATS,
    random_state: int = 0,
    simultaneous: bool = False,
    **declared: Any,
) -> Any:
    """``LTMLE`` with the static arms always and never on an incremental-oracle law."""
    return LTMLE(
        common.regimens(ARMS),
        reference=STATIC_REFERENCE,
        horizons=None if horizons is None else list(horizons),
        **_cell_means(),
        censoring_learner=dlaw.CellMeans(),
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        q_bounds=None if target.survival else common.Q_BOUNDS,
        simultaneous=simultaneous,
        random_state=random_state,
        repeats=repeats,
        **LTMLE_ITERATION,
    ).fit(frame, **ilaw.columns(target), **declared)


def fit_law_m(
    frame: pd.DataFrame,
    *,
    repeats: int = REPEATS,
    random_state: int = 0,
    simultaneous: bool = False,
) -> Any:
    """``LTMLE`` with the centred identity working model on law D."""
    return LTMLE(
        msm_law.REGIMENS,
        msm=declared_msm(M_DURATION, msm_law.PROJECTION_WEIGHT),
        **_cell_means(),
        censoring_learner=dlaw.CellMeans(),
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        simultaneous=simultaneous,
        random_state=random_state,
        repeats=repeats,
        **LTMLE_ITERATION,
    ).fit(frame, **msm_law.COLUMNS)


def sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """One draw of ``law`` (``"d"``, ``"i"``, ``"s"``, ``"c"`` or ``"m"``)."""
    if law in {"d", "m"}:
        return sample_law_d(dlaw.PROBS, n, seed)
    if law == "i":
        return ilaw.sample(ilaw.LAW_A, n, seed)
    if law == "s":
        return ilaw.sample(ilaw.LAW_D, n, seed)
    if law == "c":
        return ilaw.sample_clustered(n, CLUSTER_SIZE, seed)
    raise KeyError(law)


def fit_scenario(
    scenario: str,
    frame: pd.DataFrame,
    *,
    repeats: int = REPEATS,
    random_state: int = 0,
    simultaneous: bool = False,
) -> Any:
    """The declared fit of one primary scenario, or of law I under ``"law_i"``."""
    options: dict[str, Any] = {
        "repeats": repeats,
        "random_state": random_state,
        "simultaneous": simultaneous,
    }
    if scenario in {"law_d_ltmle", "law_d_sdr"}:
        return fit_law_d(frame, scenario.rsplit("_", 1)[1], **options)
    if scenario == "law_i":
        return fit_law_i(frame, **options)
    if scenario == "law_s":
        return fit_static(frame, ilaw.LAW_D, horizons=HORIZONS, **options)
    if scenario == "law_c":
        return fit_static(frame, ilaw.LAW_CK.marginal, id="id", **options)
    if scenario == "law_m":
        return fit_law_m(frame, **options)
    raise KeyError(scenario)


#: The law each scenario samples.
SCENARIO_LAW = {
    "law_d_ltmle": "d",
    "law_d_sdr": "d",
    "law_i": "i",
    "law_s": "s",
    "law_c": "c",
    "law_m": "m",
}


def estimates(scenario: str, result: Any) -> dict[str, Any]:
    """Every estimate a scenario publishes, the derived law-D ratio included."""
    names = I_NAMES if scenario == "law_i" else SCENARIOS[scenario]
    out = {name: result[name] for name in names if name != D_RATIO}
    if D_RATIO in names:
        out[D_RATIO] = result.ratio("ey_regimen[always]", "ey_regimen[never]", name=D_RATIO)
    return out


# ------------------------------------------------------------------------- primary rows


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample of ``scenario`` from an explicit seed, with its exact truths."""
    return sample(SCENARIO_LAW[scenario], n, seed), dict(TRUTHS[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """The replication rows of one fit.  A median of draws has no single plug-in."""
    rows: list[dict[str, Any]] = []
    for name, estimate in estimates(scenario, result).items():
        low, high = estimate.ci
        reference = float(truth[name])
        ratio = estimate.scale == "ratio"
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.log_psi if ratio else estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if ratio else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float("nan"),
            }
        )
    return rows


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Fit one primary replication and convert it to the shared schema."""
    result = fit_scenario(scenario, frame)
    if result.n_repeats != REPEATS:
        raise RuntimeError(f"{scenario}: the fit drew {result.n_repeats} splits, not {REPEATS}")
    return rows_from_result(result, truth, scenario, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication, one checkpoint per replicate."""
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for scenario in SCENARIOS for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        rows_at=None,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for value in outcomes.values for row in value])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])
