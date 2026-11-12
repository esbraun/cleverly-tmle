r"""Registered evidence for the small-sample Wald rule (Fix-3, plan rev 4, section 5).

The subject is the interval rule of Fix-3, computed here from the public functions the shipped
default will read, because PR 1 leaves the default unchanged (plan 5.1):

* a binary-outcome mean or difference, an estimate with ``bounds``, takes the Wald interval on
  the range-logit scale, ``Transform.range_logit(*bounds)`` with
  :func:`~cleverly.inference.wald_ci` and the normal reference (rule (a));
* every other binary-outcome target keeps the normal reference on its own scale (rule (b));
* a continuous-outcome target takes ``wald_ci(..., df=max(small_sample_df, 5))`` (rule (c)).

:func:`rule_interval` is that rule, one function for the fits and for every summary.  The
outcome family of each fit group is declared here (:attr:`Group.family`), because no PR 1
builder records it on an estimate.  An interval is withheld on one of two triggers, read off
the data and the estimate (:func:`withheld_trigger`): a treatment arm the estimate reads holds
one observed outcome value in its stratum (``constant_cell``, by
:func:`~cleverly.inference.constant_outcome_cells`), or a bounded estimate lies on or outside
its bounds (``out_of_range``).  The constant cell wins when both fire.

**Cells.**  :data:`GROUPS` lists every fit group: its law, size, replicate count, outcome
family, fit and the cells its rows fill.  The 14 gated two-sided coverage cells are the eight
owner and diagnosis cells of plan 5.2.1 and the six held-out cells of 5.2.2; the paired-gain
clauses read ``boundary_mean`` and ``student_ey1``; the non-H1 family reads law L of
``drtmle-one-inconsistent-nuisance``; every other cell is reported.  The laws are in
:mod:`tests.studies.small_sample_laws`; the owner laws use the owner study's sampler and fit
function on this study's seeds.  :mod:`tests.studies.small_sample_wald_properties` holds the
rows and the verdicts.

**Primary.**  :data:`PRIMARY_REPLICATES` fits of the owner law of ``strata-boundary-mean``
(``canonical-stratified-tmle``) at :data:`PRIMARY_N`, its ``ey[1][V=2]`` and ``ate[V=2]``.
The primary rows record the interval the package ships, the baseline the rule is read against;
the rule's interval and the small-sample columns of the same fits go to ``primary-rule.csv``
(:func:`extra_artifacts`).

**Design numbers.**  :func:`two_leg_pass`, :func:`lower_leg_pass`, :func:`paired_gain_power`
and :func:`family_pass` are the exact binomial sizing of plan 5.4 (``rev4_sizing.py``), and
``tests/unit/test_small_sample_wald_design.py`` recomputes every number the planning scripts
printed from them and from the law module.

**Seeds.**  :data:`SEED` and :data:`RESAMPLING_SEED` are used by no other study.  Property
replicate ``r`` of group ``label`` draws from ``stream_seed(STUDY, "property_sample", label,
r)``, disjoint from every owner study's stream and from the planning simulations' seeds.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

from cleverly.estimators import TMLE
from cleverly.inference import Transform, constant_outcome_cells, wald_ci
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_composite_drtmle as composite
from tests.studies import canonical_multi_arm_mar_drtmle as multi_arm_mar
from tests.studies import canonical_stratified_drtmle as stratified_drtmle
from tests.studies import canonical_stratified_msm_identity as identity_msm
from tests.studies import canonical_stratified_tmle as stratified_tmle
from tests.studies import drtmle_one_wrong as one_wrong
from tests.studies import small_sample_laws as laws
from tests.studies import stratified_alternating_law as alternating
from tests.studies import stratified_incremental_msm_properties as msm_properties
from tests.studies import stratified_law as l1
from tests.studies.evidence import replication
from tests.studies.evidence.properties import SMALL_SAMPLE_COLUMNS, small_sample_columns
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "small-sample-wald"
SEED = 2026111201
RESAMPLING_SEED = 2026111202
MARGINS = Margins()
ALPHA = MARGINS.alpha
#: The cap of the Student t reference on continuous-outcome targets (plan 2.5).
DF_CAP = 5.0

PRIMARY_SCENARIO = "boundary_v2"
PRIMARY_ESTIMANDS = ("ey[1][V=2]", "ate[V=2]")
PRIMARY_N = 2_000
PRIMARY_REPLICATES = 1_000

#: The gated coverage families, their reported family, and the others of plan 5.2.1.
CORRECTED = "corrected_coverage"
DIAGNOSIS_BOUNDED = "diagnosis_bounded"
DIAGNOSIS_CONTINUOUS = "diagnosis_continuous"
HELDOUT = "heldout_coverage"
REPORTED = "reported"
NON_H1 = "non_h1"
WITNESS = "witness"
CONTROL = "control"
COVERAGE_FAMILIES = (CORRECTED, DIAGNOSIS_BOUNDED, DIAGNOSIS_CONTINUOUS, HELDOUT)
#: The role of a reported, witness or control row: published with no gate of its own.
DIAGNOSTIC = "diagnostic"

#: The two cells whose verdict also needs the paired gain over the identity normal interval.
PAIRED_GAIN_CELLS = ("boundary_mean", "student_ey1")
#: The one-sided level of the paired gain's exact conditional lower bound (plan 5.1).
PAIRED_GAIN_LEVEL = 0.995
#: The two-sided floor of a non-H1 positive cell and the ceiling its control must fall below.
NON_H1_FLOOR = 0.92
NON_H1_CONTROL_CEILING = 0.90
#: Among the constant-cell fits, the plug-in interval's coverage must have its 99% upper
#: bound below this (the witness of plan 5.2.1).
WITNESS_CEILING = 0.50
#: The cell whose constant-cell fits the witness reads.
WITNESS_SOURCE = "boundary_mean_n500_ey1"
#: The coverage levels reported beside the 0.95 interval.
REPORTED_LEVELS = (0.80, 0.90, 0.99)


# ------------------------------------------------------------------------------ the rule


def bounded(estimate: Any) -> tuple[float, float] | None:
    """The bounds rule (a) reads: an unclustered level or difference with bounds, no transform."""
    if estimate.bounds is None or estimate.transform is not None:
        return None
    if estimate.scale not in ("level", "difference") or estimate.n_clusters < estimate.n:
        return None
    return (float(estimate.bounds[0]), float(estimate.bounds[1]))


def rule_interval(
    psi: float,
    std_error: float,
    *,
    bounds: tuple[float, float] | None,
    outcome_family: str,
    nu: float | None,
    alpha: float = ALPHA,
) -> tuple[float, float, str]:
    """The Fix-3 interval of one estimate, and the scale it is formed on.

    Parameters
    ----------
    psi : float
        The estimate.
    std_error : float
        Its plug-in standard error.
    bounds : tuple of float or None
        The open range rule (a) reads (:func:`bounded`), or ``None``.
    outcome_family : str
        ``"binomial"`` or ``"gaussian"``.
    nu : float or None
        The estimate's ``small_sample_df``.
    alpha : float
        The two-sided level.

    Returns
    -------
    tuple
        The lower and upper limits, ``nan`` when the estimate is on or outside its bounds, and
        ``"range_logit"`` or ``"identity"``.
    """
    if bounds is not None:
        low, high = bounds
        if not low < psi < high:
            return math.nan, math.nan, "range_logit"
        scale = Transform.range_logit(low, high)
        lower, upper = wald_ci(
            float(scale.forward(psi)), std_error * scale.slope(psi), alpha, df=None
        )
        return float(scale.inverse(lower)), float(scale.inverse(upper)), "range_logit"
    df = None if outcome_family != "gaussian" or nu is None else max(float(nu), DF_CAP)
    lower, upper = wald_ci(psi, std_error, alpha, df=df)
    return float(lower), float(upper), "identity"


def vector_interval(
    psi: np.ndarray,
    std_error: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
    df: np.ndarray,
    logit: bool,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """A Wald interval over many rows: on the range-logit scale when ``logit``, else identity.

    ``df`` holds each row's degrees of freedom, ``inf`` for the normal reference.  On the
    range-logit scale a row with no finite bounds, or outside them, gets ``nan``.  The summary
    reads every candidate interval through this; :func:`rule_interval` is its scalar twin, and
    the design test holds the two equal.
    """
    psi = np.asarray(psi, dtype=float)
    se = np.asarray(std_error, dtype=float)
    df = np.asarray(df, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        quantile = np.where(
            np.isinf(df),
            stats.norm.ppf(1.0 - alpha / 2.0),
            stats.t.ppf(1.0 - alpha / 2.0, np.where(np.isinf(df), 1.0, df)),
        )
        if not logit:
            return psi - quantile * se, psi + quantile * se
        a, b = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        inside = np.isfinite(a) & np.isfinite(b) & (a < psi) & (psi < b)
        centre = np.log((psi - a) / (b - psi))
        spread = se * (b - a) / ((psi - a) * (b - psi))
        lower = a + (b - a) * expit(centre - quantile * spread)
        upper = a + (b - a) * expit(centre + quantile * spread)
    return np.where(inside, lower, np.nan), np.where(inside, upper, np.nan)


# ------------------------------------------------------------------------- the triggers


def arms_read(estimand: str) -> tuple[Any, ...]:
    """The treatment arms an estimand reads, by the package's name."""
    if estimand.startswith(("ey[1]", "ey1")):
        return (1,)
    if estimand.startswith(("ey[0]", "ey0")):
        return (0,)
    if estimand.startswith("ey["):
        return (estimand[3:].split("]", 1)[0],)
    if estimand.startswith("ate[") and " vs " in estimand:
        left, right = estimand[4:].split("]", 1)[0].split(" vs ")
        return (left, right)
    return (0, 1)


def stratum_rows(frame: pd.DataFrame, estimand: str) -> np.ndarray:
    """The rows an estimand reads: its stratum's, or every row."""
    if "[V=" in estimand:
        stratum = int(estimand.split("[V=", 1)[1].split("]", 1)[0])
        return np.asarray(frame["V"].to_numpy() == stratum)
    return np.ones(len(frame), dtype=bool)


def observed_rows(frame: pd.DataFrame) -> np.ndarray | None:
    """Rows with an observed outcome and treatment, or ``None`` when every row has both."""
    if "Delta" not in frame:
        return None
    observed = frame["Delta"].to_numpy(dtype=float) == 1.0
    if "DeltaA" in frame:
        observed &= frame["DeltaA"].to_numpy(dtype=float) == 1.0
    return np.asarray(observed)


def withheld_trigger(frame: pd.DataFrame, estimand: str, estimate: Any) -> str:
    """``constant_cell``, ``out_of_range`` or ``""``, in that order of precedence."""
    constant = constant_outcome_cells(
        frame["Y"].to_numpy(),
        frame["A"].to_numpy(),
        stratum_rows(frame, estimand),
        observed=observed_rows(frame),
    )
    if any(arm in constant for arm in arms_read(estimand)):
        return "constant_cell"
    limits = bounded(estimate)
    if limits is not None and not limits[0] < float(estimate.psi) < limits[1]:
        return "out_of_range"
    return ""


# ------------------------------------------------------------------------------ the groups


@dataclass(frozen=True)
class Cell:
    """One property cell a group's rows fill.

    Parameters
    ----------
    name : str
        The cell.
    family : str
        Its property family.
    role : str
        ``positive``, ``control`` or :data:`DIAGNOSTIC`.
    estimand : str
        The package's name of the estimand it reads.
    fit : str
        Which of the group's fits it reads, ``"main"`` unless the group fits two.
    """

    name: str
    family: str
    role: str
    estimand: str
    fit: str = "main"


@dataclass(frozen=True)
class Group:
    """One set of fits: a law, a size, a replicate count, an outcome family and cells.

    Parameters
    ----------
    label : str
        The group, and the seed label of its draws.
    law : str
        The law it samples.
    n : int
        Rows per draw.
    replicates : int
        Declared replicates.
    family : str
        The declared outcome family, ``"binomial"`` or ``"gaussian"``.
    cells : tuple of Cell
        The cells its rows fill.
    fits : tuple of str
        The fits made on each draw.
    """

    label: str
    law: str
    n: int
    replicates: int
    family: str
    cells: tuple[Cell, ...]
    fits: tuple[str, ...] = ("main",)


def _cells(family: str, role: str, *pairs: tuple[str, str], fit: str = "main") -> tuple[Cell, ...]:
    return tuple(Cell(name, family, role, estimand, fit) for name, estimand in pairs)


EY1, EY0, ATE = laws.STRATUM_NAMES

GROUPS: tuple[Group, ...] = (
    Group("boundary_2000", "boundary", 2_000, 2_000, "binomial",
          _cells(CORRECTED, "positive", ("boundary_mean", EY1))),
    Group("boundary_8000", "boundary", 8_000, 2_000, "binomial",
          _cells(CORRECTED, "positive", ("boundary_mean_n8000", EY1))),
    Group("boundary_1000", "boundary", 1_000, 2_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("boundary_mean_n1000_ey1", EY1),
                 ("boundary_mean_n1000_ate", ATE))),
    Group("boundary_500", "boundary", 500, 2_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("boundary_mean_n500_ey1", EY1),
                 ("boundary_mean_n500_ate", ATE))),
    Group("logit_msm", "l1_logit", 2_000, 2_000, "binomial",
          _cells(CORRECTED, "positive", ("logit_msm_v2_a", "msm[a][V=2]"))),
    Group("identity_msm", "l1_beta", 2_000, 2_000, "gaussian",
          _cells(CORRECTED, "positive", ("identity_msm_v2_w", "msm[W][V=2]"))
          + _cells(REPORTED, DIAGNOSTIC, ("identity_msm_v2_a", "msm[a][V=2]"))),
    Group("rare", "rare", 2_000, 2_000, "binomial",
          _cells(DIAGNOSIS_BOUNDED, "positive", ("rare_ey0", EY0))
          + _cells(REPORTED, DIAGNOSTIC, ("rare_ey1", EY1), ("rare_ate", ATE))),
    Group("student", "student", 2_000, 2_000, "gaussian",
          _cells(DIAGNOSIS_CONTINUOUS, "positive", ("student_ey1", EY1), ("student_ate", ATE))),
    Group("lognormal06", "lognormal06", 2_000, 2_000, "gaussian",
          _cells(DIAGNOSIS_CONTINUOUS, "positive", ("lognormal06_ey1", EY1))),
    Group("lognormal", "lognormal", 2_000, 2_000, "gaussian",
          _cells(REPORTED, DIAGNOSTIC, ("lognormal_ey1", EY1))),
    Group("leverage", "boundary", 2_000, 1_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("leverage_crossfit_ey1", EY1),
                 ("leverage_crossfit_ate", ATE), fit="crossfit")
          + _cells(REPORTED, DIAGNOSTIC, ("leverage_in_sample_ey1", EY1),
                   ("leverage_in_sample_ate", ATE), fit="in_sample"),
          fits=("crossfit", "in_sample")),
    Group("composite", "composite", 2_000, 2_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("composite_high", "ey[high]"))),
    Group("mar_multi_arm", "mar_multi_arm", 2_000, 2_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("mar_multi_arm_ate", "ate[low vs high]"))),
    Group("drtmle_v2", "drtmle_stratified", 2_000, 2_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("drtmle_v2_ey1", EY1), ("drtmle_v2_ate", ATE),
                 fit="pooled")
          + _cells(REPORTED, DIAGNOSTIC, ("drtmle_v2_ey1_foldwise", EY1), fit="foldwise"),
          fits=("pooled", "foldwise")),
    Group("nonh1_glm_g_cf", "law_l_glm_cf", 4_000, 2_000, "binomial",
          _cells(NON_H1, "positive", ("nonh1_glm_g_cf_ey1", "ey1"),
                 ("nonh1_glm_g_cf_ey0", "ey0"))),
    Group("nonh1_slow_g", "law_l_slow_g", 4_000, 1_000, "binomial",
          _cells(NON_H1, "control", ("nonh1_slow_g_control", "ate"))),
    Group("ho_sparse_treat", "h1", 2_000, 2_000, "binomial",
          _cells(HELDOUT, "positive", ("ho_sparse_treat_ey1", "ey[1]"),
                 ("ho_sparse_treat_ate", "ate"))
          + _cells(REPORTED, DIAGNOSTIC, ("ho_sparse_treat_ey0", "ey[0]"))),
    Group("ho_hurdle", "h2", 2_000, 2_000, "gaussian",
          _cells(HELDOUT, "positive", ("ho_hurdle_ey1", EY1), ("ho_hurdle_ate", ATE))
          + _cells(REPORTED, DIAGNOSTIC, ("ho_hurdle_ey0", EY0))),
    Group("ho_logit_msm", "h3", 2_000, 2_000, "binomial",
          _cells(HELDOUT, "positive", ("ho_logit_msm_a", "msm[a][V=2]"),
                 ("ho_logit_msm_w", "msm[W][V=2]"))
          + _cells(REPORTED, DIAGNOSTIC, ("ho_logit_msm_intercept", "msm[(intercept)][V=2]"))),
)  # fmt: skip

GROUP_BY_LABEL = {group.label: group for group in GROUPS}
CELL_BY_NAME = {cell.name: cell for group in GROUPS for cell in group.cells}
GROUP_OF_CELL = {cell.name: group for group in GROUPS for cell in group.cells}
#: The laws no estimator has been fitted on before the declared run (plan 5.2.2).
HELDOUT_LAWS = ("h1", "h2", "h3")
GATED_CELLS = tuple(
    cell.name for group in GROUPS for cell in group.cells if cell.family in COVERAGE_FAMILIES
)

#: The control candidates each coverage cell's rows are re-read under (plan 5.2.1, 5.3).
CONTROLS = (
    "normal_reference_control",
    "kish_control",
    "logit_t_control",
    "capped_t_binary_control",
)


def controls_of(group: Group, cell: Cell) -> tuple[str, ...]:
    """The controls a coverage cell is re-read under: the range-logit one on a bounded target,
    the binary t on a binary-outcome target, the two identity ones everywhere."""
    out = ["normal_reference_control", "kish_control"]
    if group.family == "binomial" and not cell.estimand.startswith("msm["):
        out.append("logit_t_control")
    if group.family == "binomial":
        out.append("capped_t_binary_control")
    return tuple(out)


# ------------------------------------------------------------------------------ draws, fits


def sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """One draw of a law."""
    if law in ("boundary", "l1_logit"):
        return l1.sample(n, seed)
    if law == "l1_beta":
        return alternating.beta_sample(n, seed)
    if law in laws.DIAGNOSIS_LAWS:
        return laws.diagnosis_sample(law, n, seed)
    if law == "composite":
        return composite.draw_from_seed(composite.THREE_ARM, n, seed)[0]
    if law == "mar_multi_arm":
        return multi_arm_mar.draw_from_seed(multi_arm_mar.SCENARIO, n, seed)[0]
    if law == "drtmle_stratified":
        return stratified_drtmle.draw_from_seed(stratified_drtmle.SCENARIO, n, seed)[0]
    if law in ("law_l_glm_cf", "law_l_slow_g"):
        return one_wrong.draw_law_l(n, seed)
    if law == "h1":
        return laws.h1_sample(n, seed)
    if law == "h2":
        return laws.h2_sample(n, seed)
    if law == "h3":
        return laws.h3_sample(n, seed)
    raise KeyError(law)


def truths(law: str) -> Mapping[str, float]:
    """Every truth of a law, by estimand name."""
    if law in ("boundary",):
        return l1.TRUTH
    if law == "l1_logit":
        return alternating.truths("logit")
    if law == "l1_beta":
        return alternating.truths("identity")
    if law == "composite":
        return composite.draw_from_seed(composite.THREE_ARM, 10, 0)[1]
    if law == "mar_multi_arm":
        return multi_arm_mar.draw_from_seed(multi_arm_mar.SCENARIO, 10, 0)[1]
    if law == "drtmle_stratified":
        return stratified_drtmle.draw_from_seed(stratified_drtmle.SCENARIO, 10, 0)[1]
    if law in ("law_l_glm_cf", "law_l_slow_g"):
        return one_wrong.truth()
    return laws.truths(law)


def _forest_fit(frame: pd.DataFrame, *, cross_fit: bool) -> Any:
    """The leverage pair's fit: a random-forest outcome model, a correct logistic mechanism."""
    return (
        TMLE(
            outcome_learner=RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=1
            ),
            treatment_learner=laws.logistic(),
            cross_fit=cross_fit,
            n_folds=10,
            estimands=("ey", "ate"),
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
        .single()
    )


_FITS: dict[str, Callable[[pd.DataFrame], Any]] = {
    "boundary:main": stratified_tmle.fit_cleverly,
    "l1_logit:main": msm_properties.fit_msm,
    "l1_beta:main": identity_msm.fit_cleverly,
    "boundary:crossfit": lambda frame: _forest_fit(frame, cross_fit=True),
    "boundary:in_sample": lambda frame: _forest_fit(frame, cross_fit=False),
    "composite:main": lambda frame: composite.fit_cleverly(
        frame, composite.LAWS[composite.THREE_ARM], simultaneous=False
    ),
    "mar_multi_arm:main": lambda frame: multi_arm_mar.fit_cleverly(frame, simultaneous=False),
    "drtmle_stratified:pooled": stratified_drtmle.fit_cleverly,
    "drtmle_stratified:foldwise": lambda frame: stratified_drtmle.fit_cleverly(
        frame, reduced_crossfit="foldwise"
    ),
    "law_l_glm_cf:main": lambda frame: one_wrong.fit(frame, "q_wrong_g_glm_cf"),
    "law_l_slow_g:main": lambda frame: one_wrong.fit(frame, "q_slow_g_wrong"),
    "h1:main": laws.h1_fit,
    "h2:main": laws.h2_fit,
    "h3:main": laws.h3_fit,
}


def fit_group(group: Group, frame: pd.DataFrame) -> dict[str, Any]:
    """Every fit of one draw of a group, by fit name."""
    out = {}
    for name in group.fits:
        key = f"{group.law}:{name}"
        fit: Callable[[pd.DataFrame], Any] = (
            (lambda f: laws.diagnosis_fit(f, group.law))
            if group.law in laws.DIAGNOSIS_LAWS
            else _FITS[key]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = fit(frame)
    return out


# ------------------------------------------------------------------------------ the rows


#: The columns a property row of this study carries beside the shared ones: the raw moment df,
#: the Kish count over the carrying rows, the bounds rule (a) reads, and the rule's interval of
#: the plug-in (``nan`` on or outside the bounds), so every candidate is recomputed from rows.
RULE_COLUMNS = ("nu", "kish", "bound_low", "bound_high", "rule_lower", "rule_upper")
EXTRA_PROPERTY_COLUMNS = (*SMALL_SAMPLE_COLUMNS, *RULE_COLUMNS)


def rule_columns(frame: pd.DataFrame, estimand: str, estimate: Any, family: str) -> dict[str, Any]:
    """The study's columns of one estimate, and whether its rule interval covers ``truth``.

    Returns every :data:`EXTRA_PROPERTY_COLUMNS` entry; the caller sets ``covered``.
    """
    from cleverly.inference import kish_count

    trigger = withheld_trigger(frame, estimand, estimate)
    limits = bounded(estimate)
    nu = estimate.small_sample_df
    low, high, scale = rule_interval(
        float(estimate.psi),
        float(estimate.plugin_std_error),
        bounds=limits,
        outcome_family=family,
        nu=nu,
    )
    curve = np.asarray(estimate.influence_curve, dtype=float)
    return {
        **small_sample_columns(
            estimate, outcome_family=family, interval_scale=scale, withheld=trigger
        ),
        "nu": math.nan if nu is None else float(nu),
        "kish": kish_count(curve, rows=curve != 0.0),
        "bound_low": math.nan if limits is None else limits[0],
        "bound_high": math.nan if limits is None else limits[1],
        "rule_lower": low,
        "rule_upper": high,
    }


# ------------------------------------------------------------------------------ the record

#: The primary rows' rule columns, which the replicate file drops (:func:`extra_artifacts`).
PRIMARY_RULE_FILE = "primary-rule.csv"

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {}
for _group in GROUPS:
    for _cell in _group.cells:
        PROPERTY_CELLS.setdefault(_cell.family, ())
        PROPERTY_CELLS[_cell.family] += (_cell.name,)
PROPERTY_CELLS[WITNESS] = ("constant_cell_witness",)
PROPERTY_CELLS[CONTROL] = tuple(
    f"{cell.name}__{control}"
    for group in GROUPS
    for cell in group.cells
    if cell.family in COVERAGE_FAMILIES
    for control in controls_of(group, cell)
)

STUDY = StudyRecord(
    name="Small-sample Wald intervals",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "small_sample_wald",
    document="docs/technical-reference/method-evidence/small-sample-wald-intervals.md",
    anchor="small-sample-wald-intervals",
    scenarios={PRIMARY_SCENARIO: PRIMARY_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation="cleverly-small-sample-wald",
    reference=None,
    failure_bound={"cleverly-small-sample-wald": 0.01},
    modules=(
        "tests/studies/small_sample_wald.py",
        "tests/studies/small_sample_wald_properties.py",
        "tests/studies/small_sample_laws.py",
        "tests/studies/canonical_composite_drtmle.py",
        "tests/studies/canonical_multi_arm_mar_drtmle.py",
        "tests/studies/canonical_stratified_drtmle.py",
        "tests/studies/canonical_stratified_msm_identity.py",
        "tests/studies/canonical_stratified_tmle.py",
        "tests/studies/drtmle_one_wrong.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_incremental_msm_properties.py",
        "tests/studies/stratified_law.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/canonical_stratified_incremental_msm.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/mar_arm_indexed_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law.py",
        "tests/discrete_law_mar.py",
        "tests/discrete_law_multi.py",
    ),
    runner_module="tests.studies.small_sample_wald",
    properties_module="tests.studies.small_sample_wald_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
    property_artifacts=("falsifiers.csv",),
    extra_artifacts=(PRIMARY_RULE_FILE,),
)

CONFIGURATION = {
    "rule": (
        "bounded binary mean or difference: range-logit Wald, normal reference; other "
        "binary-outcome targets: normal reference on their scale; continuous-outcome targets: "
        "Student t at max(satterthwaite_df over the carrying rows, 5)"
    ),
    "withheld": "constant_cell (constant_outcome_cells), then out_of_range (bounds check)",
    "groups": {
        group.label: f"{group.law}, n={group.n}, R={group.replicates}, family {group.family}"
        for group in GROUPS
    },
    "primary": (
        f"{PRIMARY_SCENARIO}: canonical_stratified_tmle.fit_cleverly at n={PRIMARY_N}, the "
        "shipped interval in the shared columns"
    ),
}


# ------------------------------------------------------------------------------ the primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of the primary scenario from an explicit seed, with its truths."""
    if scenario != PRIMARY_SCENARIO:
        raise KeyError(f"{SLUG} has no scenario {scenario!r}")
    return l1.sample(n, seed), dict(l1.TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Primary replicate ``replicate`` from this study's own seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


#: The columns a primary row carries beside the published schema.
PRIMARY_EXTRA_COLUMNS = (*EXTRA_PROPERTY_COLUMNS, "rule_covered")


def primary_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: the shipped interval, and the rule beside it."""
    result = stratified_tmle.fit_cleverly(frame)
    rows = []
    for name in PRIMARY_ESTIMANDS:
        estimate = result[name]
        reference = float(truth[name])
        std_error, low, high = reported_inference(estimate)
        extra = rule_columns(frame, name, estimate, "binomial")
        shown = extra["withheld"] == ""
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
                **extra,
                "rule_covered": int(
                    shown and extra["rule_lower"] <= reference <= extra["rule_upper"]
                ),
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
    frame, truth = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return primary_rows(frame, truth, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every primary replicate."""
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
    return replication.result(outcomes, built.loc[:, [*REPLICATE_COLUMNS, *PRIMARY_EXTRA_COLUMNS]])


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The rule's reading of every primary row, beside the shipped interval.

    The replicate file keeps the published schema only, so :data:`PRIMARY_RULE_FILE` is where
    the primary phase publishes the rule's interval, its withheld trigger and the small-sample
    columns of each draw.  No verdict reads it.
    """
    keys = ["scenario", "replicate", "estimand", "truth", "estimate", "std_error"]
    frame = rows.loc[:, [*keys, *PRIMARY_EXTRA_COLUMNS]]
    order = ["scenario", "replicate", "estimand"]
    return {PRIMARY_RULE_FILE: frame.sort_values(order, ignore_index=True)}


# ------------------------------------------------------------------------------ the design


#: Each gated two-sided cell's withheld share at perfect calibration (plan 5.4): the exact
#: constant-cell probability of every arm it reads, and H2's empty-design-cell probability.
DECLARED_WITHHELD = {
    "boundary_mean": 1.2e-4,
    "boundary_mean_n8000": 0.0,
    "logit_msm_v2_a": 1.2e-4,
    "identity_msm_v2_w": 0.0,
    "rare_ey0": 1.8e-4,
    "student_ey1": 0.0,
    "student_ate": 0.0,
    "lognormal06_ey1": 0.0,
    "ho_sparse_treat_ey1": 1.9e-4,
    "ho_sparse_treat_ate": 1.9e-4,
    "ho_hurdle_ey1": 1.6e-3,
    "ho_hurdle_ate": 1.6e-3,
    "ho_logit_msm_a": 0.0,
    "ho_logit_msm_w": 0.0,
}
#: The exploratory readings of the cells whose laws chose the rule (plan 5.4); the others
#: enter the designed chance of a red row at perfect calibration.
PREDICTED_COVERAGE = {
    "boundary_mean": 0.9547,
    "logit_msm_v2_a": 0.9533,
    "identity_msm_v2_w": 0.9540,
    "rare_ey0": 0.9560,
    "student_ey1": 0.9535,
    "student_ate": 0.9480,
    "lognormal06_ey1": 0.9435,
}
#: The non-H1 cells' committed readings in ``drtmle-one-inconsistent-nuisance``.
NON_H1_COMMITTED = (0.964, 0.968)
#: The discordant shares of ``boundary_mean`` in rev 1's 3,000 fits: corrected covers and
#: identity normal misses (159), and the reverse (69).
GAIN_DISCORDANCE = (159 / 3_000, 69 / 3_000)
#: The power of ``student_ey1``'s paired gain, ``P(Poisson(21) >= 8)`` (plan 5.3).
STUDENT_GAIN_POWER = 0.9996


def _lower_limits(replicates: int, level: float = MARGINS.confidence_level) -> np.ndarray:
    counts = np.arange(replicates + 1)
    tail = (1.0 - level) / 2.0
    return np.where(
        counts > 0, stats.beta.ppf(tail, np.maximum(counts, 1), replicates - counts + 1), 0.0
    )


def two_leg_pass(coverage: float, replicates: int = 2_000, withheld: float = 0.0) -> float:
    """P(a gated cell passes both legs) at shown coverage ``coverage`` and a withheld share.

    The lower leg reads the withheld fits as misses over all ``replicates``; the upper leg
    reads the shown fits only.  Exact: the binomial count of shown fits, then the binomial
    count of covers among them (``rev4_sizing.py``'s ``two``).
    """
    low, high = MARGINS.calibration_coverage
    tail = (1.0 - MARGINS.confidence_level) / 2.0
    lower = _lower_limits(replicates)
    floor = int(np.arange(replicates + 1)[lower >= low].min())
    shown = np.arange(replicates + 1)
    weights = stats.binom.pmf(shown, replicates, 1.0 - withheld)
    total = 0.0
    for count, weight in zip(shown[weights > 1e-13], weights[weights > 1e-13], strict=True):
        covers = np.arange(count + 1)
        upper = np.where(
            covers < count,
            stats.beta.ppf(1.0 - tail, covers + 1, np.maximum(count - covers, 1)),
            1.0,
        )
        mass = stats.binom.pmf(covers, count, coverage)
        total += weight * float(mass[(covers >= floor) & (upper <= high)].sum())
    return total


def lower_leg_pass(coverage: float, replicates: int = 2_000, floor: float = NON_H1_FLOOR) -> float:
    """P(the exact 99% interval's lower end clears ``floor``) at ``coverage``."""
    counts = np.arange(replicates + 1)
    mass = stats.binom.pmf(counts, replicates, coverage)
    return float(mass[_lower_limits(replicates) >= floor].sum())


def gain_passes(b: int, c: int, level: float = PAIRED_GAIN_LEVEL) -> bool:
    """Whether the exact conditional one-sided lower bound of ``b / (b + c)`` is above 1/2."""
    return bool(b > 0 and stats.beta.ppf(1.0 - level, b, c + 1) > 0.5)


def paired_gain_power(
    rates: tuple[float, float] = GAIN_DISCORDANCE, replicates: int = 2_000
) -> float:
    """Exact multinomial power of the paired gain at discordant shares ``rates``."""
    gain, loss = rates
    power = 0.0
    for b in range(replicates + 1):
        weight = stats.binom.pmf(b, replicates, gain)
        if weight < 1e-14:
            continue
        c = np.arange(replicates - b + 1)
        mass = stats.binom.pmf(c, replicates - b, loss / (1.0 - gain))
        ok = np.array([gain_passes(b, int(k)) for k in c[mass > 1e-16]])
        power += weight * float(mass[mass > 1e-16][ok].sum())
    return power


def coverage_gain(nu: float, alpha: float = ALPHA) -> float:
    """The coverage gain of a ``t_nu`` interval over the normal one for a normal statistic."""
    return float(2.0 * stats.norm.cdf(stats.t.ppf(1.0 - alpha / 2.0, nu)) - 1.0 - (1.0 - alpha))


def unseen_share(refits: int) -> float:
    """The 99% upper limit of the share of fits below the smallest of ``refits`` dfs."""
    return float(1.0 - 0.005 ** (1.0 / refits))


def family_pass(
    *,
    predicted: bool,
    gain_power: float = 0.987,
    withheld: Mapping[str, float] | None = None,
) -> float:
    """The family pass probability: the 14 cells, both paired gains and the non-H1 pair.

    At perfect calibration with ``predicted=False``; at the exploratory readings and the
    committed non-H1 readings with ``predicted=True`` (the designed chance of no red row).
    """
    shares = DECLARED_WITHHELD if withheld is None else withheld
    cells = 1.0
    for cell, share in shares.items():
        coverage = PREDICTED_COVERAGE.get(cell, 0.95) if predicted else 0.95
        cells *= two_leg_pass(coverage, withheld=share)
    if predicted:
        non_h1 = lower_leg_pass(NON_H1_COMMITTED[0]) * lower_leg_pass(NON_H1_COMMITTED[1])
    else:
        non_h1 = lower_leg_pass(0.95) ** 2
    return cells * gain_power * STUDENT_GAIN_POWER * non_h1
