r"""Registered evidence for Fix-3 PR 3's asymmetric intervals (plan rev 6, sections 9.2 to 9.7).

The subject is the interval rule PR 3 will ship, computed here from the public functions the
shipped default will read, because the default does not change before this study's run (plan
section 7, items 7 and 8):

* a bounded difference of two bounded means of one fit takes the correlated MOVER interval
  (:func:`~cleverly.inference.mover_ci`) on the arms' shipped range-logit intervals, with the
  arms' correlation implied by the three variances
  (:func:`~cleverly.inference.implied_correlation`; plan 9.2);
* a bounded mean keeps the shipped PR 2 rule, the range-logit normal interval (its
  ``plugin_interval``);
* a continuous-outcome target takes the skew-widened t
  (:func:`~cleverly.inference.skew_widened_ci`) at ``max(small_sample_df, 5)`` with the
  estimate's :attr:`~cleverly.inference.ParameterEstimate.acceleration` (plan 9.3).

:func:`rule_interval` is that rule for one estimate, and
:func:`tests.studies.asymmetric_wald_properties.candidate_interval` its vector twin over rows;
the design test holds the two equal.  An interval is withheld when the shipped estimate carries
one of PR 2's statuses (``constant_arm_cell_plugin`` or ``out_of_range_plugin``), which a
difference inherits from its arms.

**Cells.**  :data:`GROUPS` lists every fit group.  The 12 gated cells are the nine held-out
cells of plan 9.5 (``heldout_coverage``, laws D1, D2, C1, C2, C3 of
:mod:`tests.studies.asymmetric_wald_laws`) and the three owner cells (``owner_coverage``);
every other cell is reported.  Each group draws :data:`REPLICATES` = 4,000 replicates.

**The gated reading** (plan 9.5, rev 6): the two coverage legs of study 1's rule read the first
:data:`LEG_REPLICATES` = 2,000 replicates by index (fixed before the run), the tail clause
(:data:`TAIL_THRESHOLD`), the witnesses, the width clause and the paired gains read all 4,000.

**Primary.**  :data:`PRIMARY_REPLICATES` fits of ``strata-boundary-mean``'s owner law at
:data:`PRIMARY_N`, its ``ate[V=2]`` under MOVER in the shared columns (a withheld interval as the
point, a miss); the shipped identity interval in ``primary-shipped.csv``.  The plan names no
primary; this one is a diagnosis law, read by the reporting policy (``pr3-notes.md``).

**Design numbers.**  :func:`legs_pass`, :func:`tails_pass`, :func:`joint_single`,
:func:`gain_power`, :func:`tail_fire` and :func:`family_pass` are ``rev6_sizing.py``'s exact
sums, and ``tests/unit/test_asymmetric_wald_design.py`` recomputes every number it printed.

**Seeds.**  :data:`SEED` and :data:`RESAMPLING_SEED` are used by no other study.  Property
replicate ``r`` of group ``label`` draws from ``stream_seed(STUDY, "property_sample", label,
r)``, disjoint from study 1's stream and from the planning simulations' ``2**40 + ...`` and
``2**41 + ...`` seeds.
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

from cleverly.inference import implied_correlation, mover_ci, skew_widened_ci
from tests.parallel import STUDY_JOBS
from tests.studies import asymmetric_wald_laws as heldout
from tests.studies import canonical_stratified_msm_identity as identity_msm
from tests.studies import canonical_stratified_tmle as stratified_tmle
from tests.studies import small_sample_laws as laws
from tests.studies import stratified_alternating_law as alternating
from tests.studies import stratified_law as l1
from tests.studies.evidence import replication
from tests.studies.evidence.properties import SMALL_SAMPLE_COLUMNS, small_sample_columns
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "asymmetric-wald"
SEED = 2026111301
RESAMPLING_SEED = 2026111302
MARGINS = Margins()
ALPHA = MARGINS.alpha
#: The cap of the Student t reference on continuous-outcome targets (plan 2.5).
DF_CAP = 5.0

#: Every group's replicates, and the first replicates the coverage legs read (plan 9.5).
REPLICATES = 4_000
LEG_REPLICATES = 2_000
#: The tail clause: each one-sided miss rate's 99% CP lower bound must be at most this.
TAIL_THRESHOLD = 0.040
#: The C3 width clause: the 99% bootstrap upper bound of the mean width ratio must not exceed it.
WIDTH_THRESHOLD = 1.08
#: The cells whose width clause is a falsifier (plan 9.6).
WIDTH_CELLS = ("c3_ey1", "c3_ate")
#: The one-sided level of the paired gains' exact conditional lower bound (plan 9.6).
PAIRED_GAIN_LEVEL = 0.995

PRIMARY_SCENARIO = "boundary_v2"
PRIMARY_ESTIMANDS = ("ate[V=2]",)
PRIMARY_N = 2_000
PRIMARY_REPLICATES = 1_000

HELDOUT = "heldout_coverage"
OWNER = "owner_coverage"
REPORTED = "reported"
WITNESS = "witness"
CONTROL = "control"
COVERAGE_FAMILIES = (HELDOUT, OWNER)
#: The role of a reported, witness or control row: published with no gate of its own.
DIAGNOSTIC = "diagnostic"

#: The owner cells' paired gains: the cell and the candidate its rule is compared with.
PAIRED_GAINS = {
    "ho_sparse_treat_ate": "identity_normal",
    "ho_hurdle_ey1": "capped_t",
    "lognormal06_ey1": "capped_t",
}
#: The deliberate-mutation witnesses: name, source cell, the mutated candidate (plan 9.6).
WITNESSES = (
    ("sign_flip_witness", "ho_hurdle_ey1", "sign_flip"),
    ("mover_swap_witness", "ho_sparse_treat_ate", "mover_swap"),
)
#: The coverage levels reported beside the 0.95 interval.
REPORTED_LEVELS = (0.80, 0.90, 0.99)

#: The shipped statuses that withhold an interval, and the trigger a row records.
WITHHELD_STATUS = {
    "constant_arm_cell_plugin": "constant_cell",
    "out_of_range_plugin": "out_of_range",
}


# ------------------------------------------------------------------------------ the rule


def is_bounded_difference(estimate: Any) -> bool:
    """An unclustered difference with bounds ``(-1, 1)`` and no transform."""
    return bool(
        estimate.bounds == (-1.0, 1.0)
        and estimate.scale == "difference"
        and estimate.transform is None
        and estimate.n_clusters >= estimate.n
    )


def is_bounded_mean(estimate: Any) -> bool:
    """An unclustered mean with bounds ``(0, 1)``, the PR 2 rule's scope."""
    return estimate.interval_scale == "range_logit"


def capped_df(nu: float | None) -> float | None:
    """``max(nu, 5)``, ``None`` (the normal) when ``nu`` is missing."""
    return None if nu is None or not math.isfinite(nu) else max(float(nu), DF_CAP)


def rule_interval(
    estimate: Any,
    *,
    family: str,
    arms: tuple[Any, Any] | None,
    alpha: float = ALPHA,
) -> tuple[float, float, str]:
    """PR 3's interval of one shown estimate, and the construction's name.

    Parameters
    ----------
    estimate : ParameterEstimate
        The estimate.
    family : str
        The group's declared outcome family.
    arms : tuple of ParameterEstimate or None
        The two arm means ``(minuend, subtrahend)`` of a bounded difference of one fit.
    alpha : float
        The two-sided level.

    Returns
    -------
    tuple
        The lower and upper limits, and ``"mover"``, ``"range_logit"``, ``"skew_t"`` or
        ``"identity"``.
    """
    psi, se = float(estimate.psi), float(estimate.plugin_std_error)
    if family == "binomial" and is_bounded_difference(estimate):
        if arms is None:
            raise ValueError(f"{estimate.name}: a bounded difference needs both arm estimates")
        one, zero = arms
        low_1, high_1 = _range_logit(one, alpha)
        low_0, high_0 = _range_logit(zero, alpha)
        r = implied_correlation(float(estimate.variance), float(one.variance), float(zero.variance))
        low, high = mover_ci(
            float(one.psi), (low_1, high_1), float(zero.psi), (low_0, high_0), correlation=r
        )
        return low, high, "mover"
    if family == "binomial" and is_bounded_mean(estimate):
        low, high = _range_logit(estimate, alpha)
        return low, high, "range_logit"
    if family == "gaussian":
        if estimate.acceleration is None:
            raise ValueError(f"{estimate.name}: a continuous target needs its acceleration")
        low, high = skew_widened_ci(
            psi,
            se,
            alpha,
            df=capped_df(estimate.small_sample_df),
            acceleration=float(estimate.acceleration),
        )
        return low, high, "skew_t"
    raise ValueError(f"{estimate.name}: no PR 3 rule for a {family} target of this kind")


def _range_logit(estimate: Any, alpha: float) -> tuple[float, float]:
    """The shipped PR 2 interval of a bounded mean at level ``1 - alpha``."""
    from cleverly.inference import Transform, wald_ci

    psi = float(estimate.psi)
    if not 0.0 < psi < 1.0:
        return math.nan, math.nan
    scale = Transform.range_logit(0.0, 1.0)
    low, high = wald_ci(
        float(scale.forward(psi)), float(estimate.plugin_std_error) * scale.slope(psi), alpha
    )
    return float(scale.inverse(low)), float(scale.inverse(high))


def withheld_trigger(estimate: Any, arms: tuple[Any, Any] | None) -> str:
    """The shipped estimate's withheld trigger, ``""`` when its interval is shown.

    A difference inherits its arms' statuses under PR 2; the arms are read too, so a difference
    is never shown with a withheld arm.
    """
    for source in (estimate, *(arms or ())):
        trigger = WITHHELD_STATUS.get(str(source.inference))
        if trigger is not None:
            return trigger
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
        ``positive`` or :data:`DIAGNOSTIC`.
    estimand : str
        The package's name of the estimand it reads.
    """

    name: str
    family: str
    role: str
    estimand: str


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
    family : str
        The declared outcome family, ``"binomial"`` or ``"gaussian"``.
    cells : tuple of Cell
        The cells its rows fill.
    replicates : int
        Declared replicates.
    """

    label: str
    law: str
    n: int
    family: str
    cells: tuple[Cell, ...]
    replicates: int = REPLICATES


def _cells(family: str, role: str, *pairs: tuple[str, str]) -> tuple[Cell, ...]:
    return tuple(Cell(name, family, role, estimand) for name, estimand in pairs)


EY1, EY0, ATE = laws.STRATUM_NAMES

GROUPS: tuple[Group, ...] = (
    Group("d1", "d1", heldout.LAWS["d1"].n, "binomial",
          _cells(HELDOUT, "positive", ("d1_ey1", EY1), ("d1_ate", ATE))
          + _cells(REPORTED, DIAGNOSTIC, ("d1_ey0", EY0), ("d1_ate_v0", "ate[V=0]"))),
    Group("d2", "d2", heldout.LAWS["d2"].n, "binomial",
          _cells(HELDOUT, "positive", ("d2_ey0", "ey[0]"), ("d2_ate", "ate"))
          + _cells(REPORTED, DIAGNOSTIC, ("d2_ey1", "ey[1]"))),
    Group("c1", "c1", heldout.LAWS["c1"].n, "gaussian",
          _cells(HELDOUT, "positive", ("c1_ey1", EY1), ("c1_ate", ATE))
          + _cells(REPORTED, DIAGNOSTIC, ("c1_ey0", EY0))),
    Group("c2", "c2", heldout.LAWS["c2"].n, "gaussian",
          _cells(HELDOUT, "positive", ("c2_ey1", EY1))
          + _cells(REPORTED, DIAGNOSTIC, ("c2_ate", ATE), ("c2_ey0", EY0))),
    Group("c3", "c3", heldout.LAWS["c3"].n, "gaussian",
          _cells(HELDOUT, "positive", ("c3_ey1", EY1), ("c3_ate", ATE))
          + _cells(REPORTED, DIAGNOSTIC, ("c3_ey0", EY0))),
    Group("ho_sparse_treat", "h1", 2_000, "binomial",
          _cells(OWNER, "positive", ("ho_sparse_treat_ate", "ate"))),
    Group("ho_hurdle", "h2", 2_000, "gaussian",
          _cells(OWNER, "positive", ("ho_hurdle_ey1", EY1))
          + _cells(REPORTED, DIAGNOSTIC, ("ho_hurdle_ate", ATE), ("ho_hurdle_ey0", EY0))),
    Group("lognormal06", "lognormal06", 2_000, "gaussian",
          _cells(OWNER, "positive", ("lognormal06_ey1", EY1))),
    Group("student", "student", 2_000, "gaussian",
          _cells(REPORTED, DIAGNOSTIC, ("student_ey1", EY1), ("student_ate", ATE))),
    Group("lognormal", "lognormal", 2_000, "gaussian",
          _cells(REPORTED, DIAGNOSTIC, ("lognormal_ey1", EY1))),
    Group("boundary_2000", "boundary", 2_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("boundary_2000_ate", ATE))),
    Group("boundary_1000", "boundary", 1_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("boundary_1000_ate", ATE))),
    Group("rare", "rare", 2_000, "binomial",
          _cells(REPORTED, DIAGNOSTIC, ("rare_ate", ATE))),
    Group("identity_msm", "l1_beta", 2_000, "gaussian",
          _cells(REPORTED, DIAGNOSTIC, ("identity_msm_v2_w", "msm[W][V=2]"))),
)  # fmt: skip

GROUP_BY_LABEL = {group.label: group for group in GROUPS}
CELL_BY_NAME = {cell.name: cell for group in GROUPS for cell in group.cells}
GROUP_OF_CELL = {cell.name: group for group in GROUPS for cell in group.cells}
#: The laws no estimator has been fitted on before the declared run (plan 9.5).
HELDOUT_LAWS = heldout.HELDOUT_LAWS
#: The groups a pre-run smoke may fit: every group whose law is not held out.
SMOKE_GROUPS = tuple(group.label for group in GROUPS if group.law not in HELDOUT_LAWS)
GATED_CELLS = tuple(
    cell.name for group in GROUPS for cell in group.cells if cell.family in COVERAGE_FAMILIES
)


def kind_of(group: Group, cell: Cell) -> str:
    """``difference``, ``mean`` or ``continuous``: which construction the cell's rule takes."""
    if group.family == "gaussian":
        return "continuous"
    return "difference" if cell.estimand.startswith("ate") else "mean"


def controls_of(group: Group, cell: Cell) -> tuple[str, ...]:
    """The controls a gated cell's fits are re-read under (plan 9.6, ``pr3-notes.md``)."""
    kind = kind_of(group, cell)
    if kind == "difference":
        return (
            "normal_reference_control",
            "range_logit_control",
            "logit_t_control",
            "cf_shift_control",
        )
    if kind == "mean":
        return ("normal_reference_control", "logit_t_control")
    return ("normal_reference_control", "capped_t_control", "cf_shift_control")


def arm_names(estimand: str) -> tuple[str, str] | None:
    """The two arm means a difference reads, by the package's names, or ``None``."""
    if not estimand.startswith("ate"):
        return None
    tag = estimand[3:]
    return (f"ey[1]{tag}", f"ey[0]{tag}")


# ------------------------------------------------------------------------------ draws, fits


def sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """One draw of a law."""
    if law in HELDOUT_LAWS:
        return heldout.sample(law, n, seed)
    if law == "boundary":
        return l1.sample(n, seed)
    if law == "l1_beta":
        return alternating.beta_sample(n, seed)
    if law in laws.DIAGNOSIS_LAWS:
        return laws.diagnosis_sample(law, n, seed)
    if law == "h1":
        return laws.h1_sample(n, seed)
    if law == "h2":
        return laws.h2_sample(n, seed)
    raise KeyError(law)


def truths(law: str) -> Mapping[str, float]:
    """Every truth of a law, by estimand name."""
    if law in HELDOUT_LAWS:
        return heldout.truths(law)
    if law == "boundary":
        return l1.TRUTH
    if law == "l1_beta":
        return alternating.truths("identity")
    return laws.truths(law)


_FITS: dict[str, Callable[[pd.DataFrame], Any]] = {
    "boundary": stratified_tmle.fit_cleverly,
    "l1_beta": identity_msm.fit_cleverly,
    "h1": laws.h1_fit,
    "h2": laws.h2_fit,
}


def fit_group(group: Group, frame: pd.DataFrame) -> Any:
    """The fit of one draw of a group."""
    if group.law in HELDOUT_LAWS:
        fit: Callable[[pd.DataFrame], Any] = lambda f: heldout.fit(group.law, f)  # noqa: E731
    elif group.law in laws.DIAGNOSIS_LAWS:
        fit = lambda f: laws.diagnosis_fit(f, group.law)  # noqa: E731
    else:
        fit = _FITS[group.law]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(frame)


# ------------------------------------------------------------------------------ the rows


#: The columns a property row carries beside the shared ones: the raw moment df, the curve's
#: acceleration, the bounds, the arm estimates and SEs of a difference with their implied
#: correlation, the rule's interval (``nan`` when withheld) and the construction, so every
#: candidate interval is recomputed from rows.
RULE_COLUMNS = (
    "nu",
    "acceleration",
    "bound_low",
    "bound_high",
    "arm1_estimate",
    "arm1_std_error",
    "arm0_estimate",
    "arm0_std_error",
    "arm_correlation",
    "rule_lower",
    "rule_upper",
    "construction",
)
EXTRA_PROPERTY_COLUMNS = (*SMALL_SAMPLE_COLUMNS, *RULE_COLUMNS)


def rule_columns(result: Any, estimand: str, family: str) -> dict[str, Any]:
    """The study's columns of one estimand of one fit.

    Returns every :data:`EXTRA_PROPERTY_COLUMNS` entry; the caller sets ``covered``.
    """
    estimate = result[estimand]
    names = arm_names(estimand)
    arms = None
    if family == "binomial" and names is not None:
        arms = (result[names[0]], result[names[1]])
    if (family == "binomial") != (estimate.bounds is not None) and not estimand.startswith("msm["):
        raise ValueError(
            f"{estimand}: a {family} group's target {'has no' if estimate.bounds is None else 'has'}"
            " bounds; the group's family is misdeclared"
        )
    trigger = withheld_trigger(estimate, arms)
    if trigger:
        low, high = math.nan, math.nan
        construction = (
            "mover" if arms is not None else "range_logit" if family == "binomial" else "skew_t"
        )
    else:
        low, high, construction = rule_interval(estimate, family=family, arms=arms)
    nu = estimate.small_sample_df
    accel = estimate.acceleration
    row: dict[str, Any] = {
        **small_sample_columns(
            estimate, outcome_family=family, interval_scale=construction, withheld=trigger
        ),
        "nu": math.nan if nu is None else float(nu),
        "acceleration": math.nan if accel is None else float(accel),
        "bound_low": math.nan if estimate.bounds is None else float(estimate.bounds[0]),
        "bound_high": math.nan if estimate.bounds is None else float(estimate.bounds[1]),
        "arm1_estimate": math.nan,
        "arm1_std_error": math.nan,
        "arm0_estimate": math.nan,
        "arm0_std_error": math.nan,
        "arm_correlation": math.nan,
        "rule_lower": low,
        "rule_upper": high,
        "construction": construction,
    }
    if arms is not None:
        one, zero = arms
        row.update(
            {
                "arm1_estimate": float(one.psi),
                "arm1_std_error": float(one.plugin_std_error),
                "arm0_estimate": float(zero.psi),
                "arm0_std_error": float(zero.plugin_std_error),
                "arm_correlation": implied_correlation(
                    float(estimate.variance), float(one.variance), float(zero.variance)
                ),
            }
        )
    return row


# ------------------------------------------------------------------------------ the record

#: The shipped interval and the study's columns of every primary row (:func:`extra_artifacts`).
PRIMARY_SHIPPED_FILE = "primary-shipped.csv"

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {}
for _group in GROUPS:
    for _cell in _group.cells:
        PROPERTY_CELLS.setdefault(_cell.family, ())
        PROPERTY_CELLS[_cell.family] += (_cell.name,)
PROPERTY_CELLS[WITNESS] = tuple(name for name, _, _ in WITNESSES)
PROPERTY_CELLS[CONTROL] = tuple(
    f"{cell.name}__{control}"
    for group in GROUPS
    for cell in group.cells
    if cell.family in COVERAGE_FAMILIES
    for control in controls_of(group, cell)
)

STUDY = StudyRecord(
    name="Asymmetric Wald intervals",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "asymmetric_wald",
    document="docs/technical-reference/method-evidence/asymmetric-wald-intervals.md",
    anchor="asymmetric-wald-intervals",
    scenarios={PRIMARY_SCENARIO: PRIMARY_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation="cleverly-asymmetric-wald",
    reference=None,
    failure_bound={"cleverly-asymmetric-wald": 0.01},
    modules=(
        "tests/studies/asymmetric_wald.py",
        "tests/studies/asymmetric_wald_properties.py",
        "tests/studies/asymmetric_wald_laws.py",
        "tests/studies/small_sample_laws.py",
        "tests/studies/canonical_stratified_msm_identity.py",
        "tests/studies/canonical_stratified_tmle.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_incremental_msm_properties.py",
        "tests/studies/stratified_law.py",
        "tests/studies/canonical_stratified_incremental_msm.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law.py",
    ),
    runner_module="tests.studies.asymmetric_wald",
    properties_module="tests.studies.asymmetric_wald_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
    property_artifacts=("falsifiers.csv",),
    extra_artifacts=(PRIMARY_SHIPPED_FILE,),
)

CONFIGURATION = {
    "rule": (
        "bounded difference of two bounded means of one fit: correlated MOVER on the arms' "
        "range-logit normal intervals, correlation implied by the three variances; bounded "
        "mean: range-logit normal (PR 2); continuous-outcome target: skew-widened t, the hull "
        "of the t interval at max(satterthwaite_df, 5) and its Cornish-Fisher correction "
        "with the curve's acceleration"
    ),
    "withheld": "the shipped PR 2 status of the estimate or of either arm it reads",
    "reading": (
        f"coverage legs on the first {LEG_REPLICATES} replicates; tail clause (99% CP lower "
        f"bound of each one-sided miss rate at most {TAIL_THRESHOLD}), width clause "
        f"(threshold {WIDTH_THRESHOLD}), witnesses and paired gains on all {REPLICATES}"
    ),
    "width_bootstrap": (
        "fits resampled with replacement, paired widths together; B = Margins."
        "bootstrap_replicates; seed stream_seed(STUDY, 'width_ratio', cell); percentile, "
        "upper bound the 0.995 quantile (R6-1)"
    ),
    "groups": {
        group.label: f"{group.law}, n={group.n}, R={group.replicates}, family {group.family}"
        for group in GROUPS
    },
    "primary": (
        f"{PRIMARY_SCENARIO}: canonical_stratified_tmle.fit_cleverly at n={PRIMARY_N}, "
        "MOVER's interval of ate[V=2] in the shared columns (a withheld one as the point), the "
        "shipped interval in primary-shipped.csv"
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


#: The shipped interval's columns a primary row carries beside the published schema.
SHIPPED_COLUMNS = ("shipped_std_error", "shipped_ci_lower", "shipped_ci_upper", "shipped_covered")
#: The columns a primary row carries beside the published schema.
PRIMARY_EXTRA_COLUMNS = (*EXTRA_PROPERTY_COLUMNS, *SHIPPED_COLUMNS)


def primary_limits(estimate: float, extra: Mapping[str, Any]) -> tuple[float, float]:
    """The rule's limits a primary row publishes: the point itself when withheld."""
    if extra["withheld"] != "":
        return estimate, estimate
    return float(extra["rule_lower"]), float(extra["rule_upper"])


def primary_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: MOVER's interval, and the shipped one beside it."""
    result = stratified_tmle.fit_cleverly(frame)
    rows = []
    for name in PRIMARY_ESTIMANDS:
        estimate = result[name]
        reference = float(truth[name])
        shipped_se, shipped_low, shipped_high = reported_inference(estimate)
        extra = rule_columns(result, name, "binomial")
        psi = float(estimate.psi)
        low, high = primary_limits(psi, extra)
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": psi,
                "inference_estimate": psi,
                "std_error": float(estimate.plugin_std_error),
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
                **extra,
                "shipped_std_error": shipped_se,
                "shipped_ci_lower": shipped_low,
                "shipped_ci_upper": shipped_high,
                "shipped_covered": int(shipped_low <= reference <= shipped_high),
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
    """The shipped interval of every primary row, beside MOVER's reading of it."""
    keys = ["scenario", "replicate", "estimand", "truth", "estimate", "std_error"]
    frame = rows.loc[:, [*keys, *PRIMARY_EXTRA_COLUMNS]]
    order = ["scenario", "replicate", "estimand"]
    return {PRIMARY_SHIPPED_FILE: frame.sort_values(order, ignore_index=True)}


# ------------------------------------------------------------------------------ the design

#: Each gated cell's withheld share as ``rev6_sizing.py`` priced it: the exact constant-cell
#: probability of every arm read (D1, D2 by :func:`~tests.studies.asymmetric_wald_laws.
#: constant_cell`), and on H2 its empty-design-cell probability plus its constant cells.
DECLARED_WITHHELD = {
    "d1_ey1": 1.12e-3,
    "d1_ate": 1.12e-3,
    "d2_ey0": 6.07e-5,
    "d2_ate": 6.07e-5,
    "c1_ey1": 0.0,
    "c1_ate": 0.0,
    "c2_ey1": 0.0,
    "c3_ey1": 0.0,
    "c3_ate": 0.0,
    "ho_sparse_treat_ate": 1.9e-4,
    "ho_hurdle_ey1": 1.6e-3,
    "lognormal06_ey1": 0.0,
}
#: The owner cells' exploratory readings on diagnosis laws (coverage, below, above), which
#: the designed chance of a red row reads (``witness_inputs.out``).
PREDICTED = {
    "ho_sparse_treat_ate": (0.9567, 0.0317, 0.0117),
    "ho_hurdle_ey1": (0.9443, 0.0390, 0.0167),
    "lognormal06_ey1": (0.9503, 0.0377, 0.0120),
}
#: The paired gains' discordant rates per fit (b, c) on the diagnosis fits.
GAIN_RATES = {
    "ho_sparse_treat_ate": (0.0483, 0.0213),
    "ho_hurdle_ey1": (0.0190, 0.0),
    "lognormal06_ey1": (0.0133, 0.0),
}
#: The mutation witnesses' heavy-side rates and withheld shares on the diagnosis fits.
WITNESS_RATES = {"sign_flip_witness": (0.0581, 1.6e-3), "mover_swap_witness": (0.1123, 1.9e-4)}
#: The width clause's predicted ratios on C3 (``width_predict.py``) and the per-fit SD of
#: the ratio from the same normal approximation to the law of ``a_hat``.
WIDTH_PREDICTED = {"c3_ey1": (1.049, 0.037), "c3_ate": (1.039, 0.030)}
_TAIL = (1.0 - MARGINS.confidence_level) / 2.0


def cp_low(k: Any, n: int) -> np.ndarray:
    """The 99% Clopper-Pearson lower ends of ``k`` successes in ``n``."""
    k = np.asarray(k)
    return np.where(k > 0, stats.beta.ppf(_TAIL, np.maximum(k, 1), np.maximum(n - k + 1, 1)), 0.0)


def cp_high(k: Any, n: int) -> np.ndarray:
    """The 99% Clopper-Pearson upper ends of ``k`` successes in ``n``."""
    k = np.asarray(k)
    return np.where(k < n, stats.beta.ppf(1.0 - _TAIL, k + 1, np.maximum(n - k, 1)), 1.0)


def max_tail(n: int, threshold: float = TAIL_THRESHOLD) -> int:
    """The largest one-sided miss count of ``n`` shown fits that passes the tail clause."""
    counts = np.arange(n + 1)
    return int(counts[cp_low(counts, n) <= threshold].max())


def legs_pass(coverage: float, withheld: float, replicates: int = LEG_REPLICATES) -> float:
    """P(the lower leg, withheld as misses, and the shown-only upper leg pass) over ``replicates``."""
    low, high = MARGINS.calibration_coverage
    counts = np.arange(replicates + 1)
    floor = int(counts[cp_low(counts, replicates) >= low].min())
    weights = stats.binom.pmf(counts, replicates, 1.0 - withheld)
    keep = weights > 1e-12
    total = 0.0
    for shown, weight in zip(counts[keep], weights[keep], strict=True):
        covers = np.arange(floor, shown + 1)
        covers = covers[cp_high(covers, int(shown)) <= high]
        total += float(weight) * float(stats.binom.pmf(covers, shown, coverage).sum())
    return total


def tails_pass(below: float, above: float, withheld: float, replicates: int = REPLICATES) -> float:
    """P(both one-sided miss counts of the shown fits pass the tail clause), exact."""
    counts = np.arange(replicates + 1)
    weights = stats.binom.pmf(counts, replicates, 1.0 - withheld)
    keep = weights > 1e-12
    total = 0.0
    for shown, weight in zip(counts[keep], weights[keep], strict=True):
        top = max_tail(int(shown))
        b = np.arange(top + 1)
        total += float(weight) * float(
            np.sum(
                stats.binom.pmf(b, shown, below)
                * stats.binom.cdf(top, shown - b, above / (1.0 - below))
            )
        )
    return total


def joint_single(
    coverage: float, below: float, above: float, withheld: float, replicates: int
) -> float:
    """Exact P(legs and tails pass) when both read the same ``replicates`` (the single-R design)."""
    counts = np.arange(replicates + 1)
    floor = int(counts[cp_low(counts, replicates) >= MARGINS.calibration_coverage[0]].min())
    weights = stats.binom.pmf(counts, replicates, 1.0 - withheld)
    keep = weights > 1e-12
    share = below / (below + above)
    total = 0.0
    for shown, weight in zip(counts[keep], weights[keep], strict=True):
        covers = np.arange(floor, shown + 1)
        covers = covers[cp_high(covers, int(shown)) <= MARGINS.calibration_coverage[1]]
        top = max_tail(int(shown))
        misses = shown - covers
        lo, hi = np.maximum(misses - top, 0), np.minimum(misses, top)
        inside = np.where(
            hi >= lo,
            stats.binom.cdf(hi, misses, share) - stats.binom.cdf(lo - 1, misses, share),
            0.0,
        )
        total += float(weight) * float(np.sum(stats.binom.pmf(covers, shown, coverage) * inside))
    return total


def split_pass(coverage: float, below: float, above: float, withheld: float) -> float:
    """The declared design's cell pass probability: legs on 2,000 times tails on 4,000.

    A lower bound: both events fall with the misses and the 2,000 are a subset of the 4,000,
    so they are positively associated (plan 9.7).
    """
    return legs_pass(coverage, withheld) * tails_pass(below, above, withheld)


def gain_passes(b: int, c: int, level: float = PAIRED_GAIN_LEVEL) -> bool:
    """Whether the exact conditional one-sided lower bound of ``b / (b + c)`` is above 1/2."""
    return bool(b > 0 and stats.beta.ppf(1.0 - level, b, c + 1) > 0.5)


def gain_power(b_rate: float, c_rate: float, replicates: int = REPLICATES) -> float:
    """The paired gain's power with Poisson discordant counts at the given per-fit rates."""
    lam_b, lam_c = b_rate * replicates, c_rate * replicates
    bs = np.arange(int(lam_b + 10 * np.sqrt(lam_b) + 20))
    cs = np.arange(int(lam_c + 10 * np.sqrt(lam_c) + 20))
    total = 0.0
    for c in cs:
        weight = stats.poisson.pmf(c, lam_c)
        if weight < 1e-14:
            continue
        lower = np.where(
            bs > 0, stats.beta.ppf(1.0 - PAIRED_GAIN_LEVEL, np.maximum(bs, 1), c + 1), 0.0
        )
        total += float(weight) * float(stats.poisson.pmf(bs, lam_b)[lower > 0.5].sum())
    return total


def tail_fire(rate: float, replicates: int = REPLICATES, withheld: float = 0.0) -> float:
    """P(the tail clause fires on one side) at a true one-sided rate, at the rounded shown count."""
    shown = round(replicates * (1.0 - withheld))
    return float(stats.binom.sf(max_tail(shown), shown, rate))


def width_fire(ratio: float, per_fit_sd: float, replicates: int = REPLICATES) -> float:
    """P(the width clause fires) at a true mean ratio, normal approximation to the mean ratio."""
    se = per_fit_sd / math.sqrt(replicates)
    z = float(stats.norm.ppf(1.0 - _TAIL))
    return float(stats.norm.sf((WIDTH_THRESHOLD - z * se - ratio) / se))


def width_midpoint(per_fit_sd: float, replicates: int = REPLICATES) -> float:
    """The true mean ratio at which the width clause fires with probability 1/2."""
    return WIDTH_THRESHOLD - float(stats.norm.ppf(1.0 - _TAIL)) * per_fit_sd / math.sqrt(replicates)


def family_pass(*, predicted: bool) -> float:
    """The family pass probability: the 12 gated cells and the three paired gains.

    At perfect calibration (coverage 0.95, tails 0.025) with ``predicted=False``; with the owner
    cells at their exploratory readings with ``predicted=True`` (the designed chance of no red
    row).  A lower bound (:func:`split_pass`).
    """
    total = 1.0
    for cell, withheld in DECLARED_WITHHELD.items():
        if predicted and cell in PREDICTED:
            coverage, below, above = PREDICTED[cell]
            mass = coverage + below + above
            total *= split_pass(coverage / mass, below / mass, above / mass, withheld)
        else:
            total *= split_pass(0.95, 0.025, 0.025, withheld)
    for b_rate, c_rate in GAIN_RATES.values():
        total *= gain_power(b_rate, c_rate)
    return total
