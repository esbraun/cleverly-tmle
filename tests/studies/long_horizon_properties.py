r"""The property phase shared by the three long-horizon studies, and its part driver.

``long_horizon_{ltmle,categorical_ltmle,sdr}_properties`` declare their replicate budgets and
call this module.  Each family is one property loop under the label ``properties/<family>``
(:func:`tests.studies.evidence.replication.fit_properties`), and each loop is a list of *fit
sets*.  A fit set is one independent replicate stream: one law, one n, one seed stream, and the
arms that fit each draw.  Two arms of one fit set read the same draw:

* the cross-fitted arm and its in-sample control (``crossfit_overfitting``);
* the default-bound and the unbounded arm of one ``support_loss`` pair;
* the calibration positive of a paired policy contrast and its ``inflated_se_control``, which
  reads the same fit.

Every other positive and control is its own fit set, also where the two n agree.  The cells, laws, n and estimator configurations are the design's
(:mod:`tests.long_horizon_design`): the calibration cells, the bias-family positives and controls,
the overfitting positives and :data:`~tests.long_horizon_design.POWER_N`.

**Rules.**  ``interval_calibration`` (with the shrunken, noise and inflated controls),
``double_robustness``, ``root_n_and_efficiency``, ``root_n_rate``, ``type_i_error``, ``power``,
``simultaneous_coverage`` and ``crossfit_overfitting`` read the shared verdicts and their shared
rules.  The necessity families and ``sequential_double_robustness`` read
:func:`bias_family_verdicts`: a positive passes when its 99 percent standardized-bias interval
lies inside the margin, and a control when it lies wholly outside.  :func:`bias_rule` restates
it.  A positive and its control are independent streams at different n, so no displacement is
computed between them.  ``support_loss``, ``learned_mechanism`` and the 4,000-unit
``sequential_double_robustness`` rows are reporting rows (role ``diagnostic``): they publish the
summary statistics and state no verdict.

**The band** (plan rev 4.3).  The ``simultaneous_coverage`` fit runs the refit bootstrap
(``n_bootstrap=B_GATE``) and writes three cells from one fit: ``k5__simultaneous_band`` (the
gated studentized bootstrap sup-t band), ``k5__ic_band`` (today's Wald-curve Rademacher band, a
reporting row with no verdict) and ``k5__pointwise_joint_control`` (must fail).  The SDR study
has no band cell: its bootstrap band stays experimental.

**Parts.**  :mod:`tests.long_horizon_property_parts` splits the property phase into parts of
replicate blocks, so that two machines can share it; the regeneration then refits nothing that a
part finished.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from tests import discrete_law_longitudinal_horizon as law_
from tests import long_horizon_design as D
from tests.discrete_law_longitudinal_horizon import Law
from tests.studies import long_horizon_common as common
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT
from tests.studies.evidence.simultaneous import joint_cells, joint_coverage_rows

#: The cell suffix of the reporting band: the IC-curve Rademacher band (plan rev 4.3).
IC_BAND = "ic_band"


def ic_band_cell(label: str) -> str:
    """The reporting cell of the Wald-curve band of one joint label."""
    return f"{label}__{IC_BAND}"


#: The calibration positive's suffix (``property_verdicts.calibration_controls``).
POSITIVE = "known_mechanism"
#: ``shrunken_se_control`` multiplies the positive's standard error by this.
SHRUNKEN_SE_FACTOR = 0.8
#: The root-n ladder; the smallest rung is the control rung, below ``n_kappa``.
ROOT_N_SIZES = D.ROOT_N_SIZES
#: The n of every ``support_loss`` pair, and the extra unbounded sizes at three arms.
SUPPORT_LOSS_N = 4_000
SUPPORT_LOSS_EXTRA = (16_000, 64_000)
#: The ``LTMLE`` default bound, which the ``__default_bounds`` arms apply.
DEFAULT_BOUNDS = (0.01, 1.0)
#: The reporting size of the two S3 patterns, beside their gated size.
PATTERN_REPORTING_N = 4_000
#: The ``learned_mechanism`` n: the ``respond_vs_ref`` K = 5 calibration n of each arm count.
LEARNED_N = {2: 40_000, 3: 48_000}
#: The band's law and n, and its plans, per arm count.
BAND = D.BAND
#: The overfitting cell names, positive then control.
CROSS_FITTED = D.CROSS_FITTED
IN_SAMPLE = D.IN_SAMPLE

#: The families whose verdict is :func:`bias_family_verdicts`.
BIAS_FAMILIES = (
    "targeting_necessity",
    "censoring_necessity",
    "categorical_probability_necessity",
    "rule_necessity",
    "policy_necessity",
    "sequential_double_robustness",
)
#: The reporting families: no verdict, no accounting row.
REPORTING_FAMILIES = ("support_loss", "learned_mechanism")
#: Every family, in the order the property phase fits them.
FAMILIES = (
    "interval_calibration",
    "root_n_and_efficiency",
    "double_robustness",
    *BIAS_FAMILIES,
    "type_i_error",
    "power",
    JOINT,
    "crossfit_overfitting",
    *REPORTING_FAMILIES,
)
CRITICAL = float(norm.ppf(0.975))
#: The binding share of each node, recorded on the rows of the ``support_loss`` and
#: ``learned_mechanism`` fits (review minor 5).  Every other row leaves them empty.
BINDING_COLUMNS = tuple(f"binding_share_{t}" for t in range(1, 6))
#: The families whose rows record the binding share.
BINDING_FAMILIES = ("support_loss", "learned_mechanism")
#: The two fitted slope rows of the root-n ladder, by cell suffix.
RATE_KINDS = D.RATE_KINDS


# ----------------------------------------------------------------------------- declarations
@dataclass(frozen=True)
class Arm:
    """One cell that a fit set writes from each draw.

    Parameters
    ----------
    cell : str
        The cell name.
    role : str
        ``positive``, ``control`` or ``diagnostic``.
    config : Config
        The estimator.
    read : str
        ``estimate`` (one fit), ``plugin`` (an untargeted number with a unit placeholder
        standard error) or ``band`` (the simultaneous band over ``plans``).
    bounds : tuple of float or None
        ``g_bounds`` through the learner route for treatment (``support_loss``); ``None`` keeps
        the law's columns and the gated bound.
    learned : bool
        Main-terms logistic treatment and censoring learners instead of the law's columns.
    inflated : str
        The ``inflated_se_control`` cell this fit also writes, or empty.
    plans : tuple of str
        The band's plans.
    """

    cell: str
    role: str
    config: D.Config
    read: str = "estimate"
    bounds: tuple[float, float] | None = None
    learned: bool = False
    inflated: str = ""
    plans: tuple[str, ...] = ()


@dataclass(frozen=True)
class FitSet:
    """One independent replicate stream of one family.

    Parameters
    ----------
    family : str
        The property family.
    name : str
        The stream's name, unique in the study; it names the seed stream and the checkpoint.
    arms : tuple of Arm
        The cells written from each draw.
    n : int
        The declared n.
    """

    family: str
    name: str
    arms: tuple[Arm, ...]
    n: int

    @property
    def law(self) -> Law:
        return self.arms[0].config.law

    @property
    def K(self) -> int:
        return self.arms[0].config.K

    @property
    def m(self) -> int:
        return self.arms[0].config.m

    def cells(self) -> list[str]:
        """Every ``property/cell`` the stream's rows fill, derived controls excluded."""
        out = []
        for arm in self.arms:
            if arm.read == "band":
                out += [
                    f"{JOINT}/{cell}" for cell in (*joint_cells(arm.cell), ic_band_cell(arm.cell))
                ]
                continue
            out.append(f"{self.family}/{arm.cell}")
            if arm.inflated:
                out.append(f"{self.family}/{arm.inflated}")
        return out

    def cost(self) -> float:
        """CPU seconds of one replicate, by the design's cost model."""
        total = 0.0
        for arm in self.arms:
            if arm.read == "band":
                study = "S1" if self.m == 2 else "S2"
                total += D.fit_cost(self.K, self.m, arm.plans, self.n)
                total += D.B_GATE * D.resample_cost(f"band:{study}")
                continue
            try:
                total += D.config_cost(arm.config, self.n)
            except (KeyError, ValueError):
                total += D.fit_cost(self.K, self.m, (arm.config.first,), self.n)
        return total * (1.5 if any(arm.learned for arm in self.arms) else 1.0)


def estimator(tag: str) -> str:
    return D.study_estimator(tag)


def arm_counts(tag: str) -> tuple[int, ...]:
    return {"S1": (2,), "S2": (3,), "S3": (2, 3)}[tag]


def prefix(tag: str, m: int) -> str:
    """The cell prefix of a study that runs both arm counts (S3), empty elsewhere."""
    return D.cell_prefix(tag, m)


def _top(m: int) -> str:
    return "always" if m == 2 else "high"


def _policy(m: int) -> str:
    return "coin" if m == 2 else "mix"


def _bias_cell(tag: str, family: str, cell: str, m: int) -> str:
    return D.bias_cell(tag, family, cell, m)


def _calibration(tag: str) -> list[FitSet]:
    out = []
    for cell in D.calibration_cells():
        if cell.study != tag:
            continue
        inflated = (
            f"{cell.name}__inflated_se_control" if "policy_vs_policylast" in cell.name else ""
        )
        arm = Arm(f"{cell.name}__{POSITIVE}", "positive", cell.config(), inflated=inflated)
        out.append(FitSet("interval_calibration", f"calibration:{cell.name}", (arm,), cell.n))
    return out


def _root_n(tag: str) -> list[FitSet]:
    label = D.root_n_label(tag)
    config = D.root_n_config(tag)
    out = []
    for n in ROOT_N_SIZES:
        role = "control" if n == min(ROOT_N_SIZES) else "positive"
        arm = Arm(f"{label}__n_{n}", role, config)
        out.append(FitSet("root_n_and_efficiency", f"root_n:{label}:{n}", (arm,), n))
    return out


def rate_label(tag: str) -> str:
    """The root-n ladder's label."""
    return D.root_n_label(tag)


def _bias_families(tag: str) -> list[FitSet]:
    out = []
    for positive in D.positives():
        if positive.study != tag:
            continue
        cell = _bias_cell(tag, positive.family, positive.cell, positive.config.m)
        arm = Arm(cell, "positive", positive.config)
        out.append(FitSet(positive.family, f"{positive.family}:{cell}", (arm,), positive.n))
    for control in D.controls():
        if control.study != tag:
            continue
        cell = _bias_cell(tag, control.family, control.cell, control.config.m)
        read = "plugin" if control.config.estimator == "plugin" else "estimate"
        arm = Arm(cell, "control", control.config, read=read)
        out.append(FitSet(control.family, f"{control.family}:{cell}", (arm,), control.n))
    if tag == "S3":
        for name, odd in (("P_odd", True), ("P_even", False)):
            config = D.pattern("SDR", "respond", odd)
            cell = f"{name}__n{PATTERN_REPORTING_N}"
            arm = Arm(cell, DIAGNOSTIC_ROLE, config)
            out.append(
                FitSet(
                    "sequential_double_robustness",
                    f"sequential_double_robustness:{cell}",
                    (arm,),
                    PATTERN_REPORTING_N,
                )
            )
    order = {family: index for index, family in enumerate(FAMILIES)}
    return sorted(out, key=lambda fit_set: order[fit_set.family])


def _testing(tag: str) -> list[FitSet]:
    if tag != "S1":
        return []
    null = Law(*law_.L_R.key, null_outcome=True)
    out = []
    for family, cell, law in (
        ("type_i_error", "policy_vs_policylast__k5__sharp_null", null),
        ("power", "policy_vs_policylast__k5__alternative", law_.L_R),
    ):
        config = D.Config("LTMLE", 5, 2, law, "coin", "coinlast")
        out.append(FitSet(family, f"{family}:{cell}", (Arm(cell, "positive", config),), D.POWER_N))
    return out


def _band(tag: str) -> list[FitSet]:
    if tag == "S3":
        return []
    config, plans, n = D.band_config(tag)
    arm = Arm("k5", "positive", config, read="band", plans=plans)
    return [FitSet(JOINT, f"{JOINT}:k5", (arm,), n)]


def _overfitting(tag: str) -> list[FitSet]:
    out = []
    for positive in D.overfitting_positives():
        if positive.study != tag:
            continue
        control = dataclasses.replace(positive.config, n_folds=1)
        arms = (
            Arm(CROSS_FITTED, "positive", positive.config),
            Arm(IN_SAMPLE, "control", control),
        )
        out.append(FitSet("crossfit_overfitting", "crossfit_overfitting:k5", arms, positive.n))
    return out


def _support_loss(tag: str) -> list[FitSet]:
    out = []
    for m in arm_counts(tag):
        head = prefix(tag, m)
        for kind, plan in (("static", _top(m)), ("policy", _policy(m))):
            config = D.Config(estimator(tag), 5, m, law_.L_R, plan)
            stem = f"{head}{kind}__k5__n{SUPPORT_LOSS_N}"
            arms = (
                Arm(f"{stem}__default_bounds", DIAGNOSTIC_ROLE, config, bounds=DEFAULT_BOUNDS),
                Arm(f"{stem}__no_bounds", DIAGNOSTIC_ROLE, config, bounds=common.G_BOUNDS),
            )
            out.append(FitSet("support_loss", f"support_loss:{stem}", arms, SUPPORT_LOSS_N))
        if m == 3:
            config = D.Config(estimator(tag), 5, m, law_.L_R, _top(m))
            for n in SUPPORT_LOSS_EXTRA:
                cell = f"{head}static__k5__n{n}__no_bounds"
                arm = Arm(cell, DIAGNOSTIC_ROLE, config, bounds=common.G_BOUNDS)
                out.append(FitSet("support_loss", f"support_loss:{cell}", (arm,), n))
    return out


def _learned(tag: str) -> list[FitSet]:
    out = []
    for m in arm_counts(tag):
        first, second = ("respond", "never") if m == 2 else ("respond", "low")
        config = D.Config(estimator(tag), 5, m, law_.L_R, first, second)
        cell = f"{prefix(tag, m)}respond_vs_ref__k5__learned"
        arm = Arm(cell, DIAGNOSTIC_ROLE, config, learned=True)
        out.append(FitSet("learned_mechanism", f"learned_mechanism:{cell}", (arm,), LEARNED_N[m]))
    return out


@functools.cache
def fit_sets(tag: str) -> tuple[FitSet, ...]:
    """Every fit set of a study, in the order the property phase fits them."""
    out = [
        *_calibration(tag),
        *_root_n(tag),
        *_bias_families(tag),
        *_testing(tag),
        *_band(tag),
        *_overfitting(tag),
        *_support_loss(tag),
        *_learned(tag),
    ]
    names = [fit_set.name for fit_set in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{tag}: two fit sets share a name")
    cells = [cell for fit_set in out for cell in fit_set.cells()]
    if len(set(cells)) != len(cells):
        raise ValueError(f"{tag}: two fit sets write one cell")
    return tuple(out)


@functools.cache
def fit_sets_by_name(tag: str) -> dict[str, FitSet]:
    """Every fit set of a study, by name."""
    return {fit_set.name: fit_set for fit_set in fit_sets(tag)}


def _fit_set(tag: str, name: str) -> FitSet:
    return fit_sets_by_name(tag)[name]


def calibration_labels(tag: str) -> tuple[str, ...]:
    """The calibration labels, each the cell name without its arm suffix."""
    return tuple(
        fit_set.name.split(":", 1)[1]
        for fit_set in fit_sets(tag)
        if fit_set.family == "interval_calibration"
    )


def derived_cells(tag: str) -> dict[str, tuple[str, ...]]:
    """Each calibration positive to its derived controls (``DERIVED_CELLS``)."""
    return calibration_derived(calibration_labels(tag), positive_suffix=POSITIVE)


@functools.cache
def efficiency_sd(tag: str) -> dict[str, float]:
    """The exact SD of ``D*`` of each calibration label, the noise control's unit."""
    return {
        cell.name: D.calibration_sigmas(cell)[0]
        for cell in D.calibration_cells()
        if cell.study == tag
    }


def _record(tag: str) -> StudyRecord:
    return D.STUDIES[tag].STUDY


@dataclass(frozen=True)
class _Declared:
    """The law and truths a declared cell reads (``PropertyCell.dgp``)."""

    name: str
    truths: Mapping[str, float] = field(default_factory=dict)

    def truth(self) -> dict[str, float]:
        return dict(self.truths)


def declared_cells(tag: str, replicates: Mapping[str, int]) -> tuple[PropertyCell, ...]:
    """Every sampled cell of a study, with its role, n and replicate count.

    The calibration controls derived from each positive's rows are not listed.
    """
    record = _record(tag)
    out = []
    for fit_set in fit_sets(tag):
        for arm in fit_set.arms:
            if arm.read == "band":
                band, control = joint_cells(arm.cell)
                pairs = [
                    (JOINT, band, "positive"),
                    (JOINT, control, "control"),
                    (JOINT, ic_band_cell(arm.cell), DIAGNOSTIC_ROLE),
                ]
            else:
                pairs = [(fit_set.family, arm.cell, arm.role)]
                if arm.inflated:
                    pairs.append((fit_set.family, arm.inflated, "control"))
            for family, cell, role in pairs:
                out.append(
                    PropertyCell(
                        property=family,
                        cell=cell,
                        dgp=_Declared(D.law_name(fit_set.law)),
                        outcome_learner=lambda: None,
                        treatment_learner=lambda: None,
                        n=fit_set.n,
                        replicates=replicates[fit_set.family],
                        seed=stream_seed(record, "property_sample", fit_set.family, fit_set.name),
                        role=role,
                        estimand=arm.config.ident(),
                    )
                )
    return tuple(out)


# ----------------------------------------------------------------------------- keys and fits
#: A key: ``(study tag, fit set name, replicate, requested replicates, n)``.
Key = tuple[str, str, int, int, int]


def keys(tag: str, family: str, replicates: Mapping[str, int]) -> list[Key]:
    """Every key of one family's loop, fit set by fit set."""
    return [
        (tag, fit_set.name, replicate, replicates[family], fit_set.n)
        for fit_set in fit_sets(tag)
        if fit_set.family == family
        for replicate in range(replicates[family])
    ]


def seed_of(key: Key) -> int:
    tag, name, replicate, _, _ = key
    family = _fit_set(tag, name).family
    return stream_seed(_record(tag), "property_sample", family, name, replicate)


def cell_of(key: Key) -> str:
    """The checkpoint cell: the fit set and its n."""
    return f"{key[1]}|n{key[4]}"


def replicate_of(key: Key) -> int:
    return int(key[2])


def cells_of(key: Key) -> list[str]:
    tag, name, *_ = key
    names = _fit_set(tag, name).cells()
    derived = derived_cells(tag)
    return [*names, *(control for cell in names for control in derived.get(cell, ()))]


def draw(key: Key) -> pd.DataFrame:
    """The sample of one key, a function of the key alone."""
    tag, name, _, _, n = key
    fit_set = _fit_set(tag, name)
    return law_.draw(fit_set.K, fit_set.m, n, seed_of(key), fit_set.law)


def _learners(m: int) -> dict[str, Any]:
    def logistic() -> LogisticRegression:
        return LogisticRegression(penalty=None, max_iter=2_000)

    return {
        "probabilities": False,
        "treatment_learner": logistic(),
        "censoring_learner": logistic(),
    }


def _route(arm: Arm) -> dict[str, Any]:
    config = arm.config
    route: dict[str, Any] = dict(config._route())
    if arm.learned:
        route.update(_learners(config.m))
    if arm.bounds is not None:
        route.update(
            {
                "g_bounds": arm.bounds,
                "treatment_learner": common.KnownHorizonMechanism(config.K, config.m, config.law),
                "probabilities": "censoring",
            }
        )
    return route


def _call(arm: Arm, data: pd.DataFrame, plans: tuple[str, ...], **extra: Any) -> Any:
    config = arm.config
    outcome, pseudo = config.learners_for(plans[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return common.fit(
            data,
            config.K,
            config.m,
            plans,
            estimator=config.estimator,
            outcome_learner=outcome,
            pseudo_learner=pseudo,
            n_folds=config.n_folds,
            **{**_route(arm), **extra},
        )


@dataclass(frozen=True)
class Fitted:
    """One fit's estimate and Wald interval."""

    psi: float
    std_error: float

    @property
    def ci(self) -> tuple[float, float]:
        half = CRITICAL * self.std_error
        return self.psi - half, self.psi + half


def _of(estimate: Any) -> Fitted:
    return Fitted(float(estimate.psi), float(estimate.std_error))


def estimate(arm: Arm, sample: pd.DataFrame) -> tuple[Fitted, Any]:
    """One arm's estimate on a drawn sample, and the fit result that holds it."""
    config = arm.config
    if config.msm or config.estimator == "plugin":
        values = config.fit_sample(sample)
        return Fitted(values["estimate"], values["std_error"]), None
    data = config.mechanism_frame(sample, population=True)
    if config._separate():
        # One plan per call; the contrast's curve is the difference of the two curves.
        first = _call(arm, data, (config.first,))[f"ey_regimen[{config.first}]"]
        second = _call(arm, data, (config.second,))[f"ey_regimen[{config.second}]"]
        curve = np.asarray(first.influence_curve) - np.asarray(second.influence_curve)
        se = float(np.std(curve, ddof=1) / math.sqrt(len(curve)))
        return Fitted(float(first.psi - second.psi), se), None
    if config.second:
        result = _call(arm, data, (config.second, config.first))
        return _of(result[f"ate_regimen[{config.first} vs {config.second}]"]), result
    result = _call(arm, data, (config.first,))
    return _of(result[f"ey_regimen[{config.first}]"]), result


def binding_shares(result: Any, plans: Sequence[str], K: int) -> dict[str, float]:
    """The share of followers at each node whose cumulative probability ``g_bounds`` moved.

    Read from ``result.diagnostics.support()`` (``share_truncated``).  A contrast fits two plans;
    the node's share is the larger of the two, so that it answers whether the bound binds at the
    node for either arm.  ``nan`` when the fit has no result or no row at the node.
    """
    out = dict.fromkeys(BINDING_COLUMNS[:K], math.nan)
    if result is None:
        return out
    rows = result.diagnostics.support().rows
    for t in range(1, K + 1):
        shares = [row.share_truncated for row in rows if row.regimen in plans and row.time == t]
        if shares:
            out[f"binding_share_{t}"] = float(max(shares))
    return out


def _row(
    family: str, cell: str, role: str, key: Key, truth: float, fitted: Fitted
) -> dict[str, Any]:
    low, high = fitted.ci
    _, _, replicate, requested, n = key
    z = abs(fitted.psi / fitted.std_error) if fitted.std_error > 0 else math.inf
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": fitted.psi,
        "std_error": fitted.std_error,
        "covered": int(low <= truth <= high),
        "rejected": int(z > CRITICAL),
    }


def band_rows(
    result: Any,
    config: D.Config,
    names: Sequence[str],
    *,
    label: str,
    replicate: int,
    n: int,
    requested: int,
) -> list[dict[str, Any]]:
    """The three joint rows of one fit: the bootstrap band, the IC band and the pointwise control.

    ``joint_coverage_rows`` reads the fit's Wald-curve band (``result.simultaneous``) and its
    pointwise intervals.  The gated cell replaces its critical value and its coverage by the
    studentized bootstrap sup-t band's (``result.bootstrap_band``), and the Wald-curve band
    becomes the reporting cell :func:`ic_band_cell`.  All three share the max-t statistic.
    """
    truth = _band_truths(config.K, config.m, config.law, names)
    ic, control = joint_coverage_rows(
        result,
        truth,
        names,
        label=label,
        replicate=replicate,
        n=n,
        requested=requested,
        pointwise_critical=CRITICAL,
    )
    boot = result.bootstrap_band
    if boot is None:
        raise AssertionError(f"the {label} fit reported no bootstrap band")
    if set(boot.bands) != set(names):
        raise AssertionError(f"the {label} bootstrap band covers {sorted(boot.bands)}")
    covered = all(boot.bands[name][0] <= truth[name] <= boot.bands[name][1] for name in names)
    band_cell, _ = joint_cells(label)
    gated = {
        **ic,
        "cell": band_cell,
        "role": "positive",
        "std_error": float(boot.critical_value),
        "covered": int(covered),
        "rejected": int(not covered),
    }
    reporting = {**ic, "cell": ic_band_cell(label), "role": DIAGNOSTIC_ROLE}
    return [gated, reporting, control]


def _band_truths(K: int, m: int, law: Law, names: Sequence[str]) -> dict[str, float]:
    out = {}
    for name in names:
        inner = name[name.index("[") + 1 : -1]
        if name.startswith("ey_regimen["):
            out[name] = float(law_.truth(K, m, inner, law))
        else:
            first, second = inner.split(" vs ")
            out[name] = float(law_.truth(K, m, first, law) - law_.truth(K, m, second, law))
    return out


def fit_key(key: Key, sample: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row one key writes, fitted on its drawn sample."""
    tag, name, *_ = key
    fit_set = _fit_set(tag, name)
    rows: list[dict[str, Any]] = []
    for arm in fit_set.arms:
        config = arm.config
        if arm.read == "band":
            result = _call(
                arm, sample, arm.plans, simultaneous=True, reference=None, n_bootstrap=D.B_GATE
            )
            names = sorted(result.simultaneous.bands)
            rows += band_rows(
                result,
                config,
                names,
                label=arm.cell,
                replicate=key[2],
                n=key[4],
                requested=key[3],
            )
            continue
        truth = config.truth
        if arm.read == "plugin":
            value = config.fit_sample(sample)["estimate"]
            rows.append(
                control_row(
                    property_name=fit_set.family,
                    cell=arm.cell,
                    replicate=key[2],
                    n=key[4],
                    requested=key[3],
                    truth=truth,
                    estimate=float(value),
                    standard_error=1.0,
                    critical=CRITICAL,
                )
            )
            continue
        fitted, result = estimate(arm, sample)
        row = _row(fit_set.family, arm.cell, arm.role, key, truth, fitted)
        if fit_set.family in BINDING_FAMILIES:
            row.update(binding_shares(result, config.plans, config.K))
        rows.append(row)
        if arm.inflated:
            # The contrast's standard error from the two plan variances alone.
            first = result[f"ey_regimen[{config.first}]"].std_error
            second = result[f"ey_regimen[{config.second}]"].std_error
            wide = Fitted(fitted.psi, float(math.hypot(first, second)))
            rows.append(_row(fit_set.family, arm.inflated, "control", key, truth, wide))
    return rows


# ----------------------------------------------------------------------------- the phase
@dataclass(frozen=True)
class Loop:
    """One family's ``fit_properties`` call: its keys and options."""

    family: str
    keys: list[Key]
    label: str
    derived: dict[str, tuple[str, ...]]

    def options(self, n_jobs: int) -> dict[str, Any]:
        return {
            "cells_of": cells_of,
            "cell_of": cell_of,
            "replicate_of": replicate_of,
            "n_jobs": n_jobs,
            "label": self.label,
            "draw": draw,
            "derived": self.derived,
            "seed_of": seed_of,
        }


def loops(tag: str, replicates: Mapping[str, int]) -> list[Loop]:
    """One loop per family that the study declares, in :data:`FAMILIES` order."""
    present = {fit_set.family for fit_set in fit_sets(tag)}
    derived = derived_cells(tag)
    return [
        Loop(family, keys(tag, family, replicates), f"properties/{family}", derived)
        for family in FAMILIES
        if family in present
    ]


def run_loops(
    tag: str,
    replicates: Mapping[str, int],
    fit: Callable[..., list[Any]],
    n_jobs: int,
) -> pd.DataFrame:
    """Fit every loop through ``fit`` (the module's ``fit_properties``) and add the controls."""
    values = [
        value
        for loop in loops(tag, replicates)
        for value in fit(loop.keys, fit_key, **loop.options(n_jobs))
    ]
    rows = pd.DataFrame([row for value in values for row in value])
    return with_controls(tag, rows)


def with_controls(tag: str, rows: pd.DataFrame) -> pd.DataFrame:
    """The rows with the shrunken and noise controls of each calibration label, sorted."""
    record = _record(tag)
    sizes = {
        fit_set.name.split(":", 1)[1]: fit_set.n
        for fit_set in fit_sets(tag)
        if fit_set.family == "interval_calibration"
    }
    sd = efficiency_sd(tag)
    controls = [
        calibration_controls(
            rows,
            record,
            labels=(label,),
            efficiency_bounds={label: sd[label]},
            calibration_n=sizes[label],
            shrunken_se_factor=SHRUNKEN_SE_FACTOR,
            critical=float(norm.ppf(1.0 - record.margins.alpha / 2.0)),
            positive_suffix=POSITIVE,
        )
        for label in calibration_labels(tag)
    ]
    rows = pd.concat([rows, *controls], ignore_index=True)
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *BINDING_COLUMNS]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def bias_family_verdicts(summary: pd.DataFrame) -> None:
    """Each bias-family row against the standardized-bias margin; reporting rows pass.

    A positive passes when its 99 percent standardized-bias interval lies inside the margin,
    and a control when the interval lies wholly outside it.  A
    reporting row states no verdict.
    """
    family = summary["property"].isin(BIAS_FAMILIES)
    positive = family & (summary["role"] == "positive")
    control = family & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"].astype(bool)
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"].astype(bool)
    reported = summary["role"] == DIAGNOSTIC_ROLE
    summary.loc[reported, "passed"] = True
    summary.loc[reported, "property_passed"] = True


def bias_rule() -> pf.FamilyRule:
    """The rule :func:`bias_family_verdicts` states: S1 on each gated row."""
    return pf.FamilyRule(
        families=frozenset(BIAS_FAMILIES),
        shared="bias_family_verdicts",
        clauses=(
            pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
            pf.bias_clause("bias discriminated", kind="beyond", applies=pf.roles("control")),
        ),
    )


def binding_summary(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Each cell's mean binding share per node, from its rows, in the summary's columns.

    The page prints these beside the exact-law shares of plan section 5.6.  A cell with no
    recorded share keeps ``nan``.
    """
    present = [column for column in BINDING_COLUMNS if column in rows]
    if not present:
        return
    means = rows.groupby(["property", "cell"])[present].mean()
    for column in present:
        summary[column] = [
            means[column].get((family, cell), math.nan)
            for family, cell in zip(summary["property"], summary["cell"], strict=True)
        ]


def summarize(tag: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Every verdict of a study's property rows."""
    record = _record(tag)
    present = set(rows["property"])
    summary, rates = apply_shared_verdicts(
        rows,
        record,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper", *BINDING_COLUMNS),
        rate_labels=(rate_label(tag),) if "root_n_and_efficiency" in present else (),
    )
    binding_summary(summary, rows)
    calibration_verdicts(summary, margins=record.margins, positive_suffix=POSITIVE)
    simultaneous_coverage_verdicts(summary, margins=record.margins)
    if "crossfit_overfitting" in present:
        crossfit_overfitting_verdicts(
            summary, rows, record, positive_cell=CROSS_FITTED, control_cell=IN_SAMPLE
        )
    bias_family_verdicts(summary)
    return finish(summary, rates)
