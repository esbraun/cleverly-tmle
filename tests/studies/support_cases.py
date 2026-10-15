"""The configuration each support row declares, built at the size of the study cell it stands for.

A row of :mod:`cleverly._support_rows` names a configuration that a registered study covers.
A :class:`RowCase` builds one concrete fit of that configuration without fitting anything: an
estimator, the prepared data, and the resolved estimands. The test in
``tests/unit/test_support_rows.py`` classifies each case and requires it to land on its own
row, and the phase-1 run records the key of each study's real runner so that a case that drifts
from its runner shows up as a mismatch.

The data come from the generator below, not from the law of the study. The key reads the
structure of a fit (the roles, the family, the arm count, the weights, the cluster count, the
missing columns) and never reads a value, so a small draw with the study's structure
classifies as the study's fit does. The cluster counts follow the cell: a cell at 10 to 39
clusters uses its own count, so it lands in the ``t`` tier. Each case names in ``source`` the
file and line of the runner or property function whose estimator settings it copies.

Longitudinal cases join in a later change.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.data.causal_data import CausalData
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.interventions import Incremental, Rule, Scale, Shift, Static, Stochastic
from cleverly.interventions.learned import LearnedRule
from cleverly.msm import MSM
from cleverly.targets import resolve_estimands

__all__ = ["CASES", "RowCase", "frame"]

Built = tuple[Any, CausalData, tuple[str, ...]]


@dataclass(frozen=True)
class RowCase:
    """One concrete fit that stands for a support row.

    Parameters
    ----------
    row : str
        The support row id.
    slug : str
        The registered study.
    cell : str
        The runner scenario or property cell the case stands for.
    source : str
        ``file:line`` of the runner or property function whose settings the case copies.
    build : callable
        Builds the unfitted estimator, the prepared data and the resolved estimands.
    """

    row: str
    slug: str
    cell: str
    source: str
    build: Callable[[], Built]


# ----------------------------------------------------------------------------- data


def frame(
    n: int = 240,
    *,
    y: str = "binary",
    arms: int = 2,
    dose: bool = False,
    delta: bool = False,
    delta_a: bool = False,
    strata: bool = False,
    weights: bool = False,
    clusters: int | None = None,
    known: bool = False,
    intermediate: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """A small sample with the structure of a study cell.

    Parameters
    ----------
    n : int, default 240
        Rows. With ``clusters`` the count rounds down to a multiple of it.
    y : {"binary", "continuous", "bounded"}, default "binary"
        The outcome law.
    arms : int, default 2
        The number of treatment levels, ignored when ``dose`` is set.
    dose : bool, default False
        A continuous treatment.
    delta, delta_a : bool, default False
        Missing outcomes and a missing treatment, as columns ``Delta`` and ``DeltaA``.
    strata : bool, default False
        A baseline stratum column ``V`` with three levels.
    weights : bool, default False
        A column ``w`` of positive weights.
    clusters : int or None, default None
        The cluster count. Each cluster holds the same number of rows.
    known : bool, default False
        Probability columns ``p0``, ``p1`` (and ``p2``) of a declared mechanism.
    intermediate : bool, default False
        A binary intermediate column ``Z``.
    seed : int, default 0
        The seed.

    Returns
    -------
    DataFrame
        The columns ``Y``, ``A``, ``W1``, ``W2`` and those the flags add.
    """
    rng = np.random.default_rng(seed)
    if clusters is not None:
        n = (n // clusters) * clusters
    w1, w2 = rng.standard_normal(n), rng.standard_normal(n)
    if dose:
        a = np.round(2.0 + 0.5 * w1 + rng.standard_normal(n), 3)
    else:
        a = rng.integers(0, arms, n).astype(float)
        a[:arms] = np.arange(arms, dtype=float)
    mean = 0.3 * a + 0.4 * w1 - 0.2 * w2
    if y == "binary":
        outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-mean))).astype(float)
    elif y == "bounded":
        outcome = rng.beta(2.0 + np.exp(mean) / 4.0, 3.0)
    else:
        outcome = mean + rng.standard_normal(n)
    data = pd.DataFrame({"Y": outcome, "A": a, "W1": w1, "W2": w2})
    if delta:
        observed = rng.random(n) < 0.8
        observed[:4] = True
        data["Delta"] = observed.astype(float)
        data.loc[~observed, "Y"] = np.nan
    if delta_a:
        recorded = rng.random(n) < 0.85
        recorded[:4] = True
        data["DeltaA"] = recorded.astype(float)
        data.loc[~recorded, "A"] = np.nan
    if strata:
        data["V"] = np.arange(n) % 3
    if weights:
        data["w"] = 0.5 + rng.random(n)
    if clusters is not None:
        data["cluster"] = np.arange(n) * clusters // n
    if intermediate:
        data["Z"] = rng.binomial(1, 0.5, n).astype(float)
    if known:
        levels = 1 if dose else arms
        for index in range(levels):
            data[f"p{index}"] = 1.0 / levels
    return data


def _roles(data: pd.DataFrame, **flags: bool) -> dict[str, Any]:
    """The ``from_frame`` keywords for the columns ``data`` holds."""
    roles: dict[str, Any] = {}
    if "Delta" in data:
        roles["delta"] = "Delta"
    if "DeltaA" in data:
        roles["treatment_delta"] = "DeltaA"
    if "V" in data:
        roles["strata"] = ["V"]
    if "w" in data:
        roles["weights"] = "w"
    if "cluster" in data:
        roles["id"] = "cluster"
    if "Z" in data:
        roles["intermediate"] = "Z"
    if "p0" in data:
        levels = [column for column in data if column.startswith("p")]
        roles["treatment_probabilities"] = {
            float(index): column for index, column in enumerate(levels)
        }
    if flags.get("dose"):
        roles["treatment_kind"] = "continuous"
    return roles


def _build(
    estimator: Any, data: pd.DataFrame, estimands: Sequence[str] | None = None, **flags: bool
) -> Built:
    """Prepare the data and resolve the estimands of ``estimator``, as the fit does."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = CausalData.from_frame(
            data,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", *(["V"] if "V" in data else [])],
            **_roles(data, **flags),
        )
    names = resolve_estimands(
        estimator.estimands if estimands is None else estimands,
        prepared.family,
        prepared.n_arms,
        axis=estimator._axis,
    )
    return estimator, prepared, names


def _case(
    row: str,
    slug: str,
    cell: str,
    source: str,
    make: Callable[[], Any],
    data: Mapping[str, Any],
    estimands: Sequence[str] | None = None,
    **flags: bool,
) -> RowCase:
    """A case whose estimator is ``make()`` and whose data is ``frame(**data)``."""
    return RowCase(
        row,
        slug,
        cell,
        source,
        lambda: _build(make(), frame(**data), estimands, **flags),
    )


# ----------------------------------------------------------------------------- settings

ARM_ALL = ("ey1", "ey0", "ate", "att", "atc", "ey_obs", "par", "paf", "rr", "or")
ARM_CONTINUOUS = ("ey1", "ey0", "ate", "att", "atc", "ey_obs", "par")
THREE = ("ey", "ate", "rr", "or")
ATE = ("ey0", "ey1", "ate")

_ST = "tests/studies/"


def _tmle(**kwargs: Any) -> Callable[[], TMLE]:
    return lambda: TMLE(simultaneous=False, **kwargs)


def _drtmle(**kwargs: Any) -> Callable[[], DRTMLE]:
    return lambda: DRTMLE(simultaneous=False, **kwargs)


def _ctmle(**kwargs: Any) -> Callable[[], CTMLE]:
    return lambda: CTMLE(simultaneous=False, **kwargs)


def _regimes() -> Any:
    return TMLE(
        cross_fit=False,
        interventions=(
            Static(0.0, name="never"),
            Rule(lambda data: np.zeros(len(data)), name="rule", rule_kind="known"),
        ),
    )


def _stochastic() -> Any:
    return TMLE(
        cross_fit=False,
        interventions=(
            Static(0.0, name="never"),
            Stochastic(lambda data: np.full(len(data), 0.5), name="tilt", density_kind="known"),
        ),
    )


def _msm(**kwargs: Any) -> Callable[[], TMLE]:
    def design(arm: Any, data: Any) -> np.ndarray:
        w = np.asarray(data["W1"], dtype=float)
        return np.column_stack([np.ones(len(w)), np.full(len(w), float(arm)), w])

    declared = MSM(
        design=design, terms=("(intercept)", "a", "W1"), design_kind="known", weights_kind="known"
    )
    return lambda: TMLE(simultaneous=False, msm=declared, **kwargs)


def _t_cases() -> tuple[RowCase, ...]:
    cases: list[RowCase] = []
    add = cases.append
    for cell, outcome, estimands in (
        ("binary", "binary", ARM_ALL),
        ("continuous", "continuous", ARM_CONTINUOUS),
    ):
        add(
            _case(
                f"T01.{cell}",
                "canonical-tmle",
                cell,
                _ST + "canonical_tmle.py:249",
                _tmle(cross_fit=False),
                {"y": outcome},
                estimands,
            )
        )
    add(
        _case(
            "T01.binary",
            "full-refit-bootstrap-and-derived-contrasts",
            "boot_ate_point_tmle",
            _ST + "full_refit_bootstrap_properties.py:283",
            _tmle(cross_fit=False, n_bootstrap=5),
            {"y": "binary"},
            ("ate",),
        )
    )
    for cell, outcome, estimands in (
        ("binary", "binary", ARM_ALL),
        ("continuous", "bounded", ARM_CONTINUOUS),
    ):
        q = {} if outcome == "binary" else {"q_bounds": (0.0, 1.0)}
        add(
            _case(
                f"T02.{cell}",
                "canonical-cvtmle",
                cell,
                _ST + "canonical_cvtmle.py:209",
                _tmle(cross_fit=True, n_folds=10, **q),
                {"y": outcome},
                estimands,
            )
        )
        add(
            _case(
                f"T03.{cell}",
                "fold-evaluated-cvtmle",
                cell,
                _ST + "fold_evaluated_cvtmle.py:128",
                _tmle(cross_fit=True, n_folds=10, cv_evaluation=True, **q),
                {"y": outcome},
                ("ey1", "ey0", "ate", "att", "atc"),
            )
        )
        add(
            _case(
                f"T05.{cell}",
                "repeated-crossfit-tmle",
                cell,
                _ST + "repeated_crossfit.py:134",
                _tmle(cross_fit=True, n_folds=5, repeats=3, **q),
                {"y": outcome},
                estimands,
            )
        )
    add(
        _case(
            "T03.binary",
            "fold-targeted-cvtmle",
            "fold_evaluated_leg",
            _ST + "fold_targeted_cvtmle.py:181",
            _tmle(cross_fit=True, n_folds=2, cv_evaluation=True),
            {"y": "binary"},
            ("ate",),
        )
    )
    add(
        _case(
            "T04",
            "fold-targeted-cvtmle",
            "binary",
            _ST + "fold_targeted_cvtmle.py:181",
            _tmle(cross_fit=True, n_folds=2, cv_evaluation=True, targeting_scheme="fold"),
            {"y": "binary"},
            ("ate",),
        )
    )
    add(
        _case(
            "T06",
            "weighted-tmle",
            "binary_biased_sample",
            _ST + "canonical_weighted_tmle.py:188",
            _tmle(cross_fit=False),
            {"weights": True},
            ("ey0", "ey1", "ate", "rr", "or"),
        )
    )
    add(
        _case(
            "T07",
            "learned-weighted-tmle",
            "continuous_selected_weighted_nuisances",
            _ST + "canonical_learned_weighted_tmle.py:155",
            _tmle(cross_fit=False, fluctuation="linear"),
            {"y": "continuous", "weights": True},
            ("ey0", "ey1", "ate"),
        )
    )
    add(
        _case(
            "T08",
            "canonical-multi-arm-tmle",
            "multi_arm_binary",
            _ST + "canonical_multi_arm_tmle.py:109",
            _tmle(cross_fit=False, reference=0.0),
            {"arms": 3},
            THREE,
        )
    )
    add(
        _case(
            "T10.unweighted",
            "clustered-tmle",
            "clustered_binary",
            _ST + "canonical_clustered_tmle.py:215",
            _tmle(cross_fit=True, n_folds=5),
            {"clusters": 50},
            ATE,
        )
    )
    add(
        _case(
            "T10.unweighted",
            "clustered-unequal-cvtmle",
            "unequal_informative",
            _ST + "clustered_unequal_cvtmle.py:228",
            _tmle(cross_fit=True, n_folds=5),
            {"clusters": 200, "n": 1200},
            ATE,
        )
    )
    add(
        _case(
            "T10.cluster_average",
            "clustered-unequal-cvtmle",
            "unequal_informative_cluster_average",
            _ST + "clustered_unequal_cvtmle.py:228",
            _tmle(cross_fit=True, n_folds=5),
            {"clusters": 200, "n": 1200, "weights": True},
            ATE,
        )
    )
    add(
        _case(
            "T10.fold_evaluated",
            "clustered-unequal-cvtmle",
            "fold_evaluated_property",
            _ST + "clustered_unequal_properties.py:168",
            _tmle(cross_fit=True, n_folds=5, cv_evaluation=True),
            {"clusters": 200, "n": 1200},
            ("ate",),
        )
    )
    for suffix, settings in (
        ("in_sample", {"cross_fit": False}),
        ("stacked", {"cross_fit": True, "n_folds": 5}),
        ("fold_evaluated", {"cross_fit": True, "n_folds": 5, "cv_evaluation": True}),
    ):
        add(
            _case(
                f"T11.{suffix}",
                "clustered-few-cluster-tmle",
                "few_cluster_reference",
                _ST + "clustered_few_cluster_properties.py:196",
                _tmle(**settings),
                {"clusters": 20, "n": 400},
                ("ate",),
            )
        )
    add(
        _case(
            "T13",
            "canonical-stratified-tmle",
            "stratified_binary",
            _ST + "canonical_stratified_tmle.py:230",
            _tmle(cross_fit=False),
            {"strata": True},
            ("ey1", "ey0", "ate", "att", "par"),
        )
    )
    add(
        _case(
            "T14",
            "canonical-stratified-tmle",
            "stratified_binary",
            _ST + "canonical_stratified_tmle.py:230",
            _tmle(cross_fit=True, n_folds=5),
            {"strata": True},
            ("ey1", "ey0", "ate"),
        )
    )
    add(
        _case(
            "T15",
            "canonical-stratified-incremental-msm",
            "stratified_incremental",
            _ST + "canonical_stratified_incremental_msm.py:326",
            _tmle(cross_fit=False, incremental=[Incremental(2.0), Incremental(0.5)]),
            {"strata": True},
        )
    )
    add(
        _case(
            "T16.logit_binary",
            "canonical-stratified-incremental-msm",
            "stratified_l1_logit",
            _ST + "stratified_incremental_msm_properties.py:314",
            _msm(cross_fit=False),
            {"strata": True},
        )
    )
    add(
        _case(
            "T16.dose_logit",
            "canonical-stratified-incremental-msm",
            "stratified_l2_logit",
            _ST + "stratified_incremental_msm_properties.py:364",
            _msm(cross_fit=False, density_bins=5),
            {"strata": True, "dose": True},
            dose=True,
        )
    )
    add(
        _case(
            "T16.dose_identity",
            "canonical-stratified-incremental-msm",
            "stratified_l2_identity",
            _ST + "stratified_incremental_msm_properties.py:364",
            _msm(cross_fit=False, density_bins=5),
            {"strata": True, "dose": True, "y": "continuous"},
            dose=True,
        )
    )
    add(
        _case(
            "T16.natural_course",
            "canonical-stratified-incremental-msm",
            "stratified_natural_course",
            _ST + "stratified_incremental_msm_properties.py:337",
            _tmle(cross_fit=True, n_folds=5, estimands=("ey_obs",)),
            {"strata": True, "delta": True},
        )
    )
    add(
        _case(
            "T17",
            "canonical-stratified-msm-identity",
            "stratified_msm_identity",
            _ST + "canonical_stratified_msm_identity.py:190",
            _msm(cross_fit=False),
            {"strata": True, "y": "continuous"},
        )
    )
    add(
        _case(
            "T18",
            "point-msm",
            "bounded_continuous_projection",
            _ST + "canonical_point_msm.py:186",
            _msm(cross_fit=False),
            {"y": "continuous"},
        )
    )
    add(
        RowCase(
            "T19",
            "deterministic-regimes",
            "binary_dynamic_rule",
            _ST + "canonical_deterministic_regimes.py:136",
            lambda: _build(_regimes(), frame()),
        )
    )
    add(
        RowCase(
            "T20",
            "stochastic-regimes",
            "binary_known_stochastic",
            _ST + "canonical_stochastic_regimes.py:139",
            lambda: _build(_stochastic(), frame()),
        )
    )
    add(
        _case(
            "T21",
            "shift-policies",
            "continuous_modified_policy",
            _ST + "canonical_shift_policies.py:215",
            _tmle(
                cross_fit=False,
                policies=(Shift(0.0, cap=None), Shift(0.25, cap=None)),
                density_bins=5,
            ),
            {"y": "continuous", "dose": True},
            dose=True,
        )
    )
    add(
        _case(
            "T22",
            "policy-point-mtp",
            "policy_point",
            _ST + "canonical_policy_point_mtp.py:429",
            _tmle(
                cross_fit=False,
                policies=(Shift(0.0, cap=None), Scale(1.25, cap=5.0, name="x1.25")),
                density_bins=5,
            ),
            {"dose": True},
            dose=True,
        )
    )
    add(
        _case(
            "T23",
            "incremental-interventions",
            "binary_incremental_odds",
            _ST + "canonical_incremental_interventions.py:126",
            _tmle(cross_fit=False, incremental=[Incremental(2.0), Incremental(0.5)]),
            {},
        )
    )
    for slug, cell, source in (
        ("learned-rule-cvtmle", "non_exceptional", _ST + "_learned_rule_law.py:164"),
        ("learned-rule-cvtmle", "misspecified_limit", _ST + "_learned_rule_law.py:164"),
    ):
        add(
            _case(
                "T24",
                slug,
                cell,
                source,
                _tmle(cross_fit=True, n_folds=10, cv_evaluation=True, learned_rule=LearnedRule()),
                {},
            )
        )
    add(
        _case(
            "T25",
            "mar-tmle",
            "binary_mar_observational",
            _ST + "canonical_mar_tmle.py:118",
            _tmle(cross_fit=False),
            {"delta": True},
            ATE,
        )
    )
    add(
        _case(
            "T25",
            "mar-natural-course-tmle",
            "binary_mar_natural_course",
            _ST + "canonical_mar_natural_course.py:198",
            _tmle(cross_fit=False),
            {"delta": True},
            ("ey_obs",),
        )
    )
    add(
        _case(
            "T25",
            "mar-attributable-tmle",
            "binary_mar_attributable",
            _ST + "canonical_mar_attributable.py:583",
            _tmle(cross_fit=False),
            {"delta": True},
            ("ey_obs", "ey0", "par", "paf"),
        )
    )
    for suffix, outcome, arms, estimands in (
        ("binary", "binary", 2, ("ey0", "ey1", "ate", "rr", "or")),
        ("continuous", "bounded", 2, ("ey0", "ey1", "ate")),
        ("three_arm", "binary", 3, THREE),
        ("three_arm_continuous", "bounded", 3, ("ey", "ate")),
    ):
        q = {} if outcome == "binary" else {"q_bounds": (0.0, 1.0)}
        add(
            _case(
                f"T26.{suffix}",
                "stacked-mar-arm-indexed-cvtmle",
                f"{suffix}_stacked",
                _ST + "canonical_mar_arm_indexed_cvtmle.py:337",
                _tmle(cross_fit=True, n_folds=10, reference=0.0 if arms == 3 else None, **q),
                {"y": outcome, "arms": arms, "delta": True},
                estimands,
            )
        )
    add(
        _case(
            "T26.binary",
            "stacked-mar-natural-course-cvtmle",
            "binary_mar_natural_course_flexible",
            _ST + "canonical_mar_natural_course_cvtmle.py:211",
            _tmle(cross_fit=True, n_folds=10),
            {"delta": True},
            ("ey_obs",),
        )
    )
    add(
        _case(
            "T26.binary",
            "mar-attributable-tmle",
            "binary_mar_attributable_cvtmle",
            _ST + "canonical_mar_attributable.py:583",
            _tmle(cross_fit=True, n_folds=10),
            {"delta": True},
            ("ey_obs", "ey0", "par", "paf"),
        )
    )
    add(
        _case(
            "T26.three_arm",
            "mar-attributable-tmle",
            "three_arm_mar_attributable_cvtmle",
            _ST + "canonical_mar_attributable.py:583",
            _tmle(cross_fit=True, n_folds=10, reference=0.0),
            {"delta": True, "arms": 3},
            ("ey_obs", "ey", "par"),
        )
    )
    add(
        _case(
            "T27.continuous",
            "mar-natural-course-tmle",
            "continuous_mar_natural_course",
            _ST + "canonical_mar_natural_course.py:198",
            _tmle(cross_fit=False, q_bounds=(0.0, 1.0)),
            {"y": "bounded", "delta": True},
            ("ey_obs",),
        )
    )
    add(
        _case(
            "T27.continuous",
            "mar-attributable-tmle",
            "continuous_mar_attributable",
            _ST + "canonical_mar_attributable.py:583",
            _tmle(cross_fit=False, q_bounds=(0.0, 1.0)),
            {"y": "bounded", "delta": True},
            ("ey_obs", "ey0", "par"),
        )
    )
    for suffix, slug_cell, kwargs, data, estimands in (
        (
            "three_arm",
            "three_arm_mar_attributable",
            {"reference": 0.0},
            {"delta": True, "arms": 3},
            ("ey_obs", "ey", "par"),
        ),
        (
            "weighted",
            "binary_mar_attributable_weighted",
            {},
            {"delta": True, "weights": True},
            ("ey_obs", "ey0", "par", "paf"),
        ),
        (
            "clustered",
            "binary_mar_attributable_clustered",
            {},
            {"delta": True, "clusters": 100, "n": 1000},
            ("ey_obs", "ey0", "par", "paf"),
        ),
    ):
        add(
            _case(
                f"T29.{suffix}",
                "mar-attributable-tmle",
                slug_cell,
                _ST + "canonical_mar_attributable.py:583",
                _tmle(cross_fit=False, **kwargs),
                data,
                estimands,
            )
        )
    add(
        _case(
            "T30",
            "cde-tmle",
            "binary_cde_z0_mar",
            _ST + "canonical_cde_tmle.py:145",
            _tmle(cross_fit=False),
            {"delta": True, "intermediate": True},
            ("ey0", "ey1", "ate", "rr", "or"),
        )
    )
    add(
        _case(
            "T31",
            "known-treatment-mechanism",
            "binary_q_correct",
            _ST + "canonical_known_mechanism.py:303",
            _tmle(cross_fit=False),
            {"known": True},
            ("ey0", "ey1", "ate", "rr", "or", "att", "atc"),
        )
    )
    add(
        _case(
            "T31",
            "known-treatment-mechanism",
            "ate__known_g__q_wrong__percentile",
            _ST + "canonical_known_mechanism.py:303",
            _tmle(cross_fit=False, n_bootstrap=5),
            {"known": True},
            ("ate",),
        )
    )
    add(
        _case(
            "T31.incremental",
            "known-treatment-mechanism",
            "known_g__incremental",
            _ST + "canonical_known_mechanism.py:303",
            _tmle(cross_fit=False, incremental=[Incremental(2.0)]),
            {"known": True},
        )
    )
    add(
        _case(
            "T32",
            "known-treatment-mechanism",
            "known_g__cvtmle",
            _ST + "canonical_known_mechanism.py:303",
            _tmle(cross_fit=True, n_folds=5),
            {"known": True},
            ("ate",),
        )
    )
    return tuple(cases)


def _d_cases() -> tuple[RowCase, ...]:
    cases: list[RowCase] = []
    add = cases.append
    stacked = {"cross_fit": True, "n_folds": 5}
    add(
        _case(
            "D01",
            "canonical-drtmle",
            "both_correct",
            _ST + "canonical_drtmle.py:362",
            _drtmle(**stacked),
            {},
            ATE,
        )
    )
    add(
        _case(
            "D01",
            "drtmle-one-wrong",
            "q_wrong_g_glm_cf",
            _ST + "drtmle_one_wrong.py:452",
            _drtmle(**stacked),
            {},
            ATE,
        )
    )
    add(
        _case(
            "D02",
            "canonical-multi-arm-drtmle",
            "multi_arm_binary_drtmle",
            _ST + "canonical_multi_arm_drtmle.py:249",
            _drtmle(reference=0.0, **stacked),
            {"arms": 3},
            THREE,
        )
    )
    add(
        _case(
            "D03",
            "canonical-stratified-drtmle",
            "stratified_both_correct",
            _ST + "canonical_stratified_drtmle.py:282",
            _drtmle(**stacked),
            {"strata": True},
            ("ey", "ate"),
        )
    )
    add(
        _case(
            "D04",
            "clustered-unequal-cvtmle",
            "unequal_informative",
            _ST + "clustered_unequal_properties.py:203",
            _drtmle(**stacked),
            {"clusters": 200, "n": 1200},
            ("ate",),
        )
    )
    add(
        _case(
            "D05",
            "clustered-few-cluster-tmle",
            "few_cluster_reference",
            _ST + "clustered_few_cluster_properties.py:214",
            _drtmle(**stacked),
            {"clusters": 20, "n": 400},
            ("ate",),
        )
    )
    for suffix, guard, arms in (
        ("none", (), 2),
        ("q", ("Q",), 2),
        ("g", ("g",), 2),
        ("qg", ("Q", "g"), 2),
        ("none_three_arm", (), 3),
    ):
        add(
            _case(
                f"D06.{suffix}",
                "known-treatment-mechanism-drtmle",
                f"guard_{suffix}",
                _ST + "canonical_known_mechanism_drtmle.py:179",
                _drtmle(cross_fit=False, guard=guard),
                {"known": True, "arms": arms},
                ATE if arms == 2 else ("ey", "ate"),
            )
        )
    add(
        _case(
            "D06.qg",
            "drtmle-one-wrong",
            "q_wrong_g_known",
            _ST + "drtmle_one_wrong.py:452",
            _drtmle(cross_fit=False),
            {"known": True},
            ATE,
        )
    )
    add(
        _case(
            "D07.estimated_g",
            "drtmle-one-wrong",
            "q_wrong_g_glm",
            _ST + "drtmle_one_wrong.py:452",
            _drtmle(cross_fit=False),
            {},
            ATE,
        )
    )
    add(
        _case(
            "D08",
            "mar-drtmle",
            "binary_mar_randomized",
            _ST + "canonical_mar_drtmle.py:134",
            _drtmle(cross_fit=False, randomized=True),
            {"delta": True, "known": True},
            ATE,
        )
    )
    add(
        _case(
            "D08",
            "drtmle-one-wrong",
            "mar_q_wrong_pi_slow",
            _ST + "drtmle_one_wrong.py:440",
            _drtmle(cross_fit=False, randomized=True),
            {"delta": True, "known": True},
            ATE,
        )
    )
    add(
        _case(
            "D09",
            "multi-arm-mar-drtmle",
            "three_arm_mar_randomized",
            _ST + "canonical_multi_arm_mar_drtmle.py:185",
            _drtmle(cross_fit=False, randomized=True, reference=0.0),
            {"delta": True, "known": True, "arms": 3},
            THREE,
        )
    )
    for suffix, arms, flags, tmle in (
        ("outcome", 2, {"delta": True}, False),
        ("outcome_treatment", 2, {"delta": True, "delta_a": True}, False),
        ("three_arm", 3, {"delta": True, "delta_a": True, "arms": 3}, False),
        ("tmle_outcome_treatment", 2, {"delta": True, "delta_a": True}, True),
        ("tmle_three_arm", 3, {"delta": True, "delta_a": True, "arms": 3}, True),
    ):
        make = (
            _tmle(cross_fit=False, reference=0.0 if arms == 3 else None)
            if tmle
            else _drtmle(cross_fit=False, reference=0.0 if arms == 3 else None)
        )
        add(
            _case(
                f"D10.{suffix}",
                "composite-missing-drtmle",
                {
                    "outcome": "binary_observational_mar",
                    "outcome_treatment": "binary_mar_outcome_and_treatment",
                    "three_arm": "three_arm_mar_outcome_and_treatment",
                    "tmle_outcome_treatment": "binary_mar_outcome_and_treatment",
                    "tmle_three_arm": "three_arm_mar_outcome_and_treatment",
                }[suffix],
                _ST + "canonical_composite_drtmle.py:261",
                make,
                flags,
                ("ey0", "ey1", "ate", "rr", "or") if arms == 2 else THREE,
            )
        )
    return tuple(cases)


def _c_cases() -> tuple[RowCase, ...]:
    cases: list[RowCase] = []
    add = cases.append
    for strategy in ("greedy", "ordered", "discrete"):
        extra: dict[str, Any] = (
            {"candidates": [["W1"], ["W1", "W2"]]} if strategy == "discrete" else {}
        )
        if strategy == "ordered":
            extra = {"ordering": ["W1", "W2"]}
        add(
            _case(
                f"C01.{strategy}",
                "canonical-ctmle-selector",
                f"binary_{strategy}",
                _ST + "canonical_ctmle_selector.py:170",
                _ctmle(strategy=strategy, cross_fit=False, **extra),
                {},
                ("ate",),
            )
        )
        add(
            _case(
                f"C02.{strategy}",
                "canonical-multi-arm-ctmle-selector",
                f"multi_arm_selector_{strategy}",
                _ST + "canonical_multi_arm_ctmle_selector.py:132",
                _ctmle(strategy=strategy, cross_fit=False, reference=0.0, **extra),
                {"arms": 3},
                THREE,
            )
        )
    shared = {"strategy": "oat", "oat_design": "shared"}
    per_arm = {"strategy": "oat", "oat_design": "per_arm"}
    add(
        _case(
            "C03",
            "canonical-ctmle-oat",
            "binary",
            _ST + "canonical_ctmle_oat.py:140",
            _ctmle(cross_fit=False, **shared),
            {},
            ("ey0", "ey1", "ate", "rr", "or"),
        )
    )
    add(
        _case(
            "C03.stacked",
            "canonical-ctmle-oat",
            "crossfit_overfitting",
            _ST + "ctmle_oat_properties.py:185",
            _ctmle(cross_fit=True, n_folds=5, q_bounds=(0.0, 1.0), **shared),
            {"y": "bounded"},
            ("ate",),
        )
    )
    add(
        _case(
            "C04",
            "canonical-multi-arm-ctmle-oat",
            "multi_arm_binary_oat",
            _ST + "canonical_multi_arm_ctmle_oat.py:121",
            _ctmle(cross_fit=False, reference=0.0, **shared),
            {"arms": 3},
            THREE,
        )
    )
    add(
        _case(
            "C05.binary",
            "ctmle-oat-per-arm",
            "binary_active",
            _ST + "canonical_ctmle_oat_per_arm.py:357",
            _ctmle(cross_fit=False, **per_arm),
            {},
            ("ey0", "ey1", "ate", "rr", "or", "par", "paf"),
        )
    )
    add(
        _case(
            "C05.three_arm",
            "ctmle-oat-per-arm",
            "three_arm_active",
            _ST + "canonical_ctmle_oat_per_arm.py:357",
            _ctmle(cross_fit=False, reference=0.0, **per_arm),
            {"arms": 3},
            THREE,
        )
    )
    add(
        _case(
            "C05.binary_stacked",
            "ctmle-oat-per-arm",
            "interval_calibration",
            _ST + "ctmle_oat_per_arm_properties.py:367",
            _ctmle(cross_fit=True, n_folds=5, **per_arm),
            {},
            ("ate",),
        )
    )
    add(
        _case(
            "C05.three_arm_stacked",
            "ctmle-oat-per-arm",
            "simultaneous_coverage",
            _ST + "ctmle_oat_per_arm_properties.py:410",
            _ctmle(cross_fit=True, n_folds=5, reference=0.0, **per_arm),
            {"arms": 3},
            ("ey", "ate"),
        )
    )
    add(
        _case(
            "C05.weighted_stacked",
            "ctmle-oat-per-arm",
            "binary_weighted__correctly_specified",
            _ST + "ctmle_oat_per_arm_properties.py:367",
            _ctmle(cross_fit=True, n_folds=5, **per_arm),
            {"weights": True},
            ("ate",),
        )
    )
    add(
        _case(
            "C05.repeated",
            "ctmle-oat-per-arm",
            "binary_repeats__correctly_specified",
            _ST + "ctmle_oat_per_arm_properties.py:367",
            _ctmle(cross_fit=True, n_folds=5, repeats=3, **per_arm),
            {},
            ("ate",),
        )
    )
    strata_missing = _ST + "canonical_ctmle_oat_per_arm_strata_missing.py:403"
    add(
        _case(
            "C06.strata",
            "ctmle-oat-per-arm-strata-missing",
            "strata",
            strata_missing,
            _ctmle(cross_fit=False, **per_arm),
            {"strata": True},
            ATE,
        )
    )
    add(
        _case(
            "C06.missing_outcome",
            "ctmle-oat-per-arm-strata-missing",
            "missing_outcome",
            strata_missing,
            _ctmle(cross_fit=False, **per_arm),
            {"delta": True},
            ("ey0", "ey1", "ate", "rr", "or"),
        )
    )
    add(
        _case(
            "C06.three_arm_missing",
            "ctmle-oat-per-arm-strata-missing",
            "three_arm_missing",
            strata_missing,
            _ctmle(cross_fit=False, reference=0.0, **per_arm),
            {"delta": True, "arms": 3},
            THREE,
        )
    )
    add(
        _case(
            "C06.missing_treatment",
            "ctmle-oat-per-arm-strata-missing",
            "missing_treatment",
            strata_missing,
            _ctmle(cross_fit=False, **per_arm),
            {"delta": True, "delta_a": True},
            ("ey0", "ey1", "ate", "rr", "or"),
        )
    )
    return tuple(cases)


#: The point-treatment cases: ordinary TMLE, DR-TMLE and C-TMLE.
CASES: tuple[RowCase, ...] = (*_t_cases(), *_d_cases(), *_c_cases())
