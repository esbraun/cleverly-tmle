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

A point case builds its estimator and data and classifies them. A longitudinal case, and a
point case that copies a study's own fit function, runs that function with the record switch's
entry points replaced: the fit stops where the classifier reads its key, before any learner
runs. The case then holds the key and the learner classes the study's real fit produces, with
no second copy of the study's settings.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from cleverly._support import (
    Classification,
    SupportKey,
    classify,
    classify_key,
    longitudinal_key,
    point_key,
)
from cleverly.data.causal_data import CausalData
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.interventions import Incremental, Rule, Scale, Shift, Static, Stochastic
from cleverly.interventions.learned import LearnedRule
from cleverly.msm import MSM
from cleverly.targets import resolve_estimands

__all__ = ["CASES", "Observed", "RowCase", "capture", "case_key", "classification", "frame"]

Built = tuple[Any, CausalData, tuple[str, ...]]


@dataclass(frozen=True)
class Observed:
    """The key and learner classes of a study's own fit, read where the classifier reads them.

    Parameters
    ----------
    key : SupportKey
        The key of the fit.
    learners : mapping of str to type or None
        The learner class per nuisance role.
    """

    key: SupportKey
    learners: Mapping[str, type | None]


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
        Builds the unfitted estimator, the prepared data and the resolved estimands, or returns
        the :class:`Observed` key of a study's own fit.
    """

    row: str
    slug: str
    cell: str
    source: str
    build: Callable[[], Built | Observed]


def classification(case: RowCase) -> Classification:
    """Classify a case against the registered rows.

    Parameters
    ----------
    case : RowCase
        The case.

    Returns
    -------
    Classification
        The row the case lands on, or the nearest row.
    """
    built = case.build()
    if isinstance(built, Observed):
        return classify_key(built.key, built.learners)
    estimator, data, estimands = built
    return classify(estimator, data, estimands)


def case_key(case: RowCase) -> SupportKey:
    """The key a case's fit produces.

    Parameters
    ----------
    case : RowCase
        The case.

    Returns
    -------
    SupportKey
        The key the classifier reads for the case.
    """
    built = case.build()
    if isinstance(built, Observed):
        return built.key
    estimator, data, estimands = built
    return point_key(estimator, data, estimands)[0]


class _Reached(BaseException):
    """Raised by the replaced entry points, so no fit goes past its classifier."""

    def __init__(self, observed: Observed) -> None:
        super().__init__()
        self.observed = observed


def capture(run: Callable[[], Any]) -> Observed:
    """Run a study's fit until the support classifier reads its key.

    Parameters
    ----------
    run : callable
        Calls the study's own fit function on a drawn sample.

    Returns
    -------
    Observed
        The key and the learner classes of the fit.

    Raises
    ------
    RuntimeError
        When the call returns without reaching a classifier.
    """
    longitudinal = importlib.import_module("cleverly.longitudinal.estimator")
    point = importlib.import_module("cleverly.estimators.tmle")

    def at_longitudinal(estimator: Any, data: Any, **settings: Any) -> None:
        settings.pop("source", None)
        raise _Reached(Observed(*longitudinal_key(estimator, data, **settings)))

    def at_point(
        estimator: Any, data: Any, estimands: Any, *, source: str = "", conditions: Any = ()
    ) -> None:
        raise _Reached(Observed(*point_key(estimator, data, tuple(estimands))))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with (
            mock.patch.object(longitudinal, "classify_longitudinal_fit", at_longitudinal),
            mock.patch.object(point, "classify_point_fit", at_point),
        ):
            try:
                run()
            except _Reached as reached:
                return reached.observed
    raise RuntimeError("the study's fit returned without reaching the support classifier")


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
    derivative = "canonical_ctmle_derivative"

    def derived(**options: Any) -> Callable[[], Any]:
        return lambda: _study(derivative).estimator(lambdas=None, **options)

    where = _ST + "canonical_ctmle_derivative.py:253"
    for row, cell, options, data, names in (
        ("C10.in_sample", "in_sample", {}, {}, ATE),
        ("C10.stacked", "cross_fitted", {"cross_fit": True}, {}, ("ate",)),
        ("C10.multi_arm", "multi_arm", {"three_arm": True}, {"arms": 3}, ("ate",)),
        ("C10.missing_outcome", "missing_outcome", {}, {"delta": True}, ("ate",)),
        (
            "C10.clustered",
            "cluster_robust",
            {"cross_fit": True},
            {"n": 2000, "clusters": 50},
            ("ate",),
        ),
    ):
        add(_case(row, "ctmle-derivative", cell, where, derived(**options), data, names))
    parametric = "canonical_ctmle_oat_per_arm_parametric"
    where = _ST + "canonical_ctmle_oat_per_arm_parametric.py:681"
    for row, cell, data in (
        ("C11.binary", "binary_active__ate__corrected", {}),
        ("C11.bounded", "bounded_outcome__ate__corrected", {"y": "bounded"}),
    ):
        options = {} if not data else {"q_bounds": (0.0, 1.0)}
        add(
            _case(
                row,
                "ctmle-oat-per-arm-parametric",
                cell,
                where,
                lambda options=options: _study(parametric).estimator(
                    _study(parametric).Fit("main", **options)
                ),
                data,
                ("ate",),
            )
        )
    crossfit = "canonical_ctmle_oat_per_arm_parametric_crossfit"
    where = _ST + "canonical_ctmle_oat_per_arm_parametric_crossfit.py:625"
    for row, cell, repeats, names in (
        ("C11.stacked", "crossfit_1500__ate__corrected", 1, ("ey0", "ey1", "ate", "rr", "or")),
        ("C11.repeated", "repeats__ate__corrected", 3, ATE),
    ):
        add(
            _case(
                row,
                "ctmle-oat-per-arm-parametric-crossfit",
                cell,
                where,
                lambda repeats=repeats, names=names: _study(crossfit).estimator(
                    _study(crossfit).Fit("main", estimands=names, repeats=repeats)
                ),
                {},
                names,
            )
        )
    extensions = "canonical_ctmle_oat_per_arm_parametric_extensions"
    where = _ST + "canonical_ctmle_oat_per_arm_parametric_extensions.py:876"
    # The configurations of X58b2. A case needs no reference arm: the key reads structure only,
    # and the generator labels the arms by number.
    for row, cell, fit, data, names in (
        (
            "C11.binary",
            "binary_population__par__corrected",
            {"estimands": ("ey0", "ey1", "ate", "ey_obs", "par", "paf"), "simultaneous": True},
            {},
            ("ey0", "ey1", "ate", "ey_obs", "par", "paf"),
        ),
        (
            "C11.k_arms",
            "three_arm__ey_0__corrected",
            {"estimands": ("ey", "ate", "rr", "or", "ey_obs", "par", "paf"), "k": 3},
            {"arms": 3},
            ("ey", "ate", "rr", "or", "ey_obs", "par", "paf"),
        ),
        (
            "C11.weighted",
            "weighted__ate__corrected",
            {
                "estimands": ("ey0", "ey1", "ate", "rr", "or", "ey_obs", "par", "paf"),
                "weighted": True,
            },
            {"weights": True},
            ("ey0", "ey1", "ate", "rr", "or", "ey_obs", "par", "paf"),
        ),
        (
            "C11.weighted_stacked",
            "weighted_crossfit__ate__corrected",
            {"estimands": ATE, "cross_fit": True, "weighted": True},
            {"weights": True},
            ATE,
        ),
        (
            "C11.stacked_k_arms",
            "three_arm_crossfit__ate_1_vs_0__corrected",
            {"estimands": ("ey", "ate"), "k": 3, "cross_fit": True, "simultaneous": True},
            {"arms": 3},
            ("ey", "ate"),
        ),
        (
            "C11.stacked_k_arms_weighted",
            "composition__ate_0_vs_1__corrected",
            {"estimands": ("ey", "ate"), "k": 3, "cross_fit": True, "weighted": True},
            {"arms": 3, "weights": True},
            ("ey", "ate"),
        ),
    ):
        add(
            _case(
                row,
                "ctmle-oat-per-arm-parametric-extensions",
                cell,
                where,
                lambda fit=fit: _study(extensions).estimator(_study(extensions).Fit("main", **fit)),
                data,
                names,
            )
        )
    return tuple(cases)


# ----------------------------------------------------------------- studies' own fits


def _study(name: str) -> Any:
    return importlib.import_module(f"tests.studies.{name}")


def _observed(
    row: str,
    slug: str,
    cell: str,
    where: str,
    module: str,
    draw: Callable[[Any], Any],
    fit: Callable[[Any, Any], Any],
) -> RowCase:
    """A case that runs the study's own ``fit`` on ``draw(study)`` up to the classifier.

    ``where`` is ``<file>:<line>`` of the fit function, relative to ``tests/studies``.
    """

    def build() -> Observed:
        study = _study(module)
        frame = draw(study)
        return capture(lambda: fit(study, frame))

    return RowCase(row, slug, cell, _ST + where, build)


def _seeded(
    row: str,
    slug: str,
    cell: str,
    where: str,
    module: str,
    scenario: str,
    n: int,
    fit: Callable[[Any, Any], Any] | None = None,
) -> RowCase:
    """A case that draws ``n`` rows of ``scenario`` and runs the study's ``fit_cleverly``."""
    return _observed(
        row,
        slug,
        cell,
        where,
        module,
        lambda study: study.draw_from_seed(scenario, n, 0)[0],
        fit or (lambda study, frame: study.fit_cleverly(frame)),
    )


def _crossfit_properties(name: str) -> Callable[[Any], Any]:
    """The calibration draw of an SDR property law, at the size its fits read."""
    return lambda study: study._calibration_frame(name, 2000, 0)[0]


def _long_cases() -> tuple[RowCase, ...]:
    """The longitudinal cases: each one runs the study's own fit to its classifier."""
    cases: list[RowCase] = []

    def seeded(
        row: str,
        slug: str,
        cell: str,
        where: str,
        scenario: str,
        n: int,
        fit: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        module = where.partition(".py")[0]
        cases.append(_seeded(row, slug, cell, where, module, scenario, n, fit))

    def observed(
        row: str,
        slug: str,
        cell: str,
        where: str,
        draw: Callable[[Any], Any],
        fit: Callable[[Any, Any], Any],
    ) -> None:
        cases.append(_observed(row, slug, cell, where, where.partition(".py")[0], draw, fit))

    # Study primary fits, one sample of the runner's scenario at a size that keeps its structure.
    end = "censored_end_of_study"
    seeded("G01", "canonical-ltmle", end, "canonical_ltmle.py:238", end, 400)
    seeded("G02", "canonical-ltmle-crossfit", end, "canonical_ltmle_crossfit.py:194", end, 400)
    selected = "selected_censored_end_of_study"
    seeded("G03", "weighted-ltmle", selected, "canonical_weighted_ltmle.py:99", selected, 400)
    seeded(
        "G04",
        "weighted-ltmle-crossfit",
        selected,
        "canonical_weighted_ltmle_crossfit.py:121",
        selected,
        400,
    )
    clustered = "clustered_end_of_study"
    seeded(
        "G05.cross_fitted",
        "clustered-cross-fitted-ltmle",
        clustered,
        "clustered_crossfit_ltmle.py:224",
        clustered,
        4000,
    )
    few = "few_unequal_informative_j20"
    seeded("G06", "clustered-few-cluster-tmle", few, "clustered_few_cluster_tmle.py:250", few, 250)
    few = "few_equal_j20"
    seeded(
        "G07", "few-cluster-cross-fitted-ltmle", few, "few_cluster_crossfit_ltmle.py:192", few, 800
    )
    cat = "categorical_end_of_study"
    seeded(
        "G08.in_sample",
        "canonical-categorical-ltmle",
        cat,
        "canonical_categorical_ltmle.py:97",
        cat,
        400,
    )
    seeded(
        "G08.cross_fitted",
        "canonical-categorical-ltmle-crossfit",
        cat,
        "canonical_categorical_ltmle_crossfit.py:107",
        cat,
        400,
    )
    policy = "categorical_policy_end_of_study"
    stochastic = "canonical_stochastic_categorical_ltmle.py:238"
    seeded("G08.in_sample", "stochastic-categorical-ltmle", policy, stochastic, policy, 400)
    seeded(
        "G08.cross_fitted",
        "stochastic-categorical-ltmle",
        "crossfit",
        stochastic,
        policy,
        400,
        lambda study, frame: study.fit(frame, n_folds=5),
    )
    for suffix, scenario in (
        ("continuous", "mtp_continuous"),
        ("continuous_cross_fitted", "mtp_continuous_crossfit"),
        ("categorical", "mtp_categorical"),
        ("rr_tilt", "rr_tilt"),
    ):
        seeded(
            f"G10.{suffix}",
            "longitudinal-mtp",
            scenario,
            "canonical_longitudinal_mtp.py:382",
            scenario,
            400,
            lambda study, frame, scenario=scenario: study.fit(frame, scenario),
        )
    survival = "censored_survival_curve"
    seeded(
        "G11",
        "canonical-ltmle-survival",
        survival,
        "canonical_ltmle_survival.py:246",
        survival,
        400,
    )
    seeded(
        "G12",
        "canonical-ltmle-survival-crossfit",
        survival,
        "canonical_ltmle_survival_crossfit.py:217",
        survival,
        400,
    )
    competing = "censored_competing_risk_curve"
    seeded(
        "G13.in_sample",
        "canonical-ltmle-competing",
        competing,
        "canonical_ltmle_competing.py:266",
        competing,
        400,
    )
    seeded(
        "G13.cross_fitted",
        "canonical-ltmle-competing-crossfit",
        competing,
        "canonical_ltmle_competing_crossfit.py:87",
        competing,
        400,
    )

    # Point-treatment survival: a held baseline treatment.
    def held(file: str, line: int, scenario: str) -> tuple[str, str, Callable[[Any, Any], Any]]:
        return (
            f"{file}.py:{line}",
            scenario,
            lambda study, frame: study.fit(scenario, frame),
        )

    for row, slug, file, line, scenario in (
        ("G14.survival", "point-treatment-survival", "canonical_point_survival", 240, "survival"),
        ("G14.competing", "point-treatment-survival", "canonical_point_survival", 240, "competing"),
        (
            "G17.survival",
            "point-treatment-survival-crossfit",
            "canonical_point_survival_crossfit",
            150,
            "survival",
        ),
        (
            "G17.three_arm",
            "point-treatment-survival-crossfit",
            "canonical_point_survival_crossfit",
            150,
            "three_arm",
        ),
        (
            "G18.stochastic",
            "point-treatment-survival-policies",
            "canonical_point_survival_policies",
            238,
            "policy",
        ),
        (
            "G18.policy",
            "point-treatment-survival-policies",
            "canonical_point_survival_policies",
            238,
            "mtp",
        ),
    ):
        where, name, fit = held(file, line, scenario)
        observed(
            row,
            slug,
            name,
            where,
            lambda study, name=name: study.draw_from_seed(name, 400, 0)[0],
            fit,
        )
    held_properties = "point_survival_properties.py:262"
    for row, label in (
        ("G14.three_arm", "three_arm_t3"),
        ("G14.continuous_time", "continuous_t4"),
        ("G14.end_of_study", "end_of_study"),
        ("G15", "weighted_t5"),
        ("G16", "clustered_t5"),
    ):
        observed(
            row,
            "point-treatment-survival",
            label,
            held_properties,
            lambda study, label=label: study.draw(label, 2000, 0),
            lambda study, frame, label=label: study.fit_label(label, frame),
        )

    # Hazard targeting of a held baseline treatment.
    hazard_module = "canonical_point_survival_hazard"
    for row, scenario in (("G30.survival", "survival"), ("G30.competing", "competing")):
        observed(
            row,
            "point-survival-hazard",
            scenario,
            f"{hazard_module}.py:230",
            lambda study, scenario=scenario: study.draw_from_seed(scenario, 400, 0)[0],
            lambda study, frame, scenario=scenario: study.fit(scenario, frame),
        )
    curve_properties = "point_survival_curve_properties.py:262"
    for row, slug, label, folds in (
        ("G30.survival", "point-survival-curve", "survival_t5_per_horizon", 1),
        ("G31.survival", "point-survival-curve", "survival_t5", 1),
        ("G31.competing", "point-survival-curve", "competing_t4", 1),
        ("G31.three_arm", "point-survival-curve", "rule_t3", 1),
        ("G31.weighted", "point-survival-curve", "weighted_t5", 1),
        ("G31.weighted_per_horizon", "point-survival-curve", "weighted_t5_per_horizon", 1),
        ("G31.clustered", "point-survival-curve", "clustered_t5", 1),
        ("G31.clustered_per_horizon", "point-survival-curve", "clustered_t5_per_horizon", 1),
        ("G31.isotonic", "point-survival-curve", "isotonic_t2", 1),
        ("G32.survival", "point-survival-curve-crossfit", "survival_t5", 5),
        ("G32.survival_per_horizon", "point-survival-curve-crossfit", "survival_t5_per_horizon", 5),
        ("G32.competing", "point-survival-curve-crossfit", "competing_t4", 5),
    ):
        observed(
            row,
            slug,
            label,
            curve_properties,
            lambda study, label=label: study.draw(label, 2000, 0),
            lambda study, frame, label=label, folds=folds: study.fit_label(
                label, frame, n_folds=folds
            ),
        )
    seeded(
        "G19",
        "longitudinal-msm",
        "censored_regimen_projection",
        "canonical_longitudinal_msm.py:184",
        "censored_regimen_projection",
        400,
    )
    seeded(
        "G20",
        "cross-fitted-longitudinal-msm",
        "cross_fitted_regimen_projection",
        "canonical_crossfit_longitudinal_msm.py:207",
        "cross_fitted_regimen_projection",
        400,
    )
    seeded(
        "G21",
        "known-node-mechanisms",
        "smart_q_correct",
        "canonical_known_node_mechanisms.py:194",
        "smart_q_correct",
        400,
    )
    seeded(
        "G27",
        "longitudinal-repeated-outcome",
        "censored_repeated_outcome",
        "canonical_ltmle_repeated.py:460",
        "censored_repeated_outcome",
        400,
    )

    # The IS and cross-fitted fits of ``SDR``: the runner's scenarios and the property cells.
    sdr = "canonical_longitudinal_sdr.py:362"
    sdr_slug = "longitudinal-sdr"
    for row, scenario in (
        ("G23.in_sample", "sdr_end_of_study"),
        ("G23.cross_fitted", "sdr_end_of_study_crossfit"),
        ("G24.survival.in_sample", "sdr_survival"),
        ("G24.mtp.in_sample", "sdr_mtp"),
        ("G24.policy.in_sample", "sdr_policy"),
    ):
        seeded(
            row,
            sdr_slug,
            scenario,
            sdr,
            scenario,
            400,
            lambda study, frame, scenario=scenario: study.fit(frame, scenario),
        )
    properties = "longitudinal_sdr_properties.py"
    for row, law, fit_name, line, n in (
        ("G24.survival.cross_fitted", "survival", "survival_fit", 406, 2000),
        ("G24.competing", "competing", "competing_fit", 425, 2000),
        ("G24.policy.cross_fitted", "policy", "policy_fit", 445, 2000),
        ("G24.mtp.cross_fitted", "mtp", "mtp_fit", 456, 2000),
        ("G24.msm", "msm", "msm_fit", 475, 2000),
        ("G25", "weighted", "weighted_fit", 487, 2000),
        ("G26.clusters_200", "clusters_200", "clustered_fit", 505, 2000),
        ("G26.clusters_20", "clusters_20", "clustered_fit", 505, 2000),
    ):
        observed(
            row,
            sdr_slug,
            law,
            f"{properties}:{line}",
            lambda study, law=law, n=n: study._calibration_frame(law, n, 0)[0],
            lambda study, frame, fit_name=fit_name: getattr(study, fit_name)(frame),
        )

    # ``IPSI`` (X19): each property fit, on its own stream's draw at a small size.
    s1, s1_slug = "longitudinal_incremental_properties.py", "longitudinal-incremental"
    s1_cases: tuple[tuple[str, str, str, int, Callable[[Any, Any], Any]], ...] = (
        (
            "G33.cross_fitted",
            "contrast_x2",
            "law_a",
            505,
            lambda m, f: m._fit(f, m.law.LAW_A, m.PLANS),
        ),
        (
            "G33.in_sample",
            "contrast_x2_in_sample",
            "law_a",
            506,
            lambda m, f: m._fit(f, m.law.LAW_A, m.PLANS, n_folds=1),
        ),
        (
            "G33.known",
            "known_x2",
            "law_a",
            507,
            lambda m, f: m._fit(
                f,
                m.law.LAW_A,
                m.PLANS,
                treatment_probabilities=m.common.known_treatment(f, m.law.LAW_A),
            ),
        ),
        (
            "G33.two_nodes",
            "no_positivity",
            "law_ax",
            485,
            lambda m, f: m._fit(f, m.law.LAW_AX, m.AX_PLANS),
        ),
    )
    for row, cell, stream, line, fit in s1_cases:
        observed(
            row,
            s1_slug,
            cell,
            f"{s1}:{line}",
            lambda m, stream=stream: m._draw_fit_set(
                ("interval_calibration", stream, "calibration", 0, 2000, 1, stream)
            ),
            fit,
        )
    s2, s2_slug = (
        "longitudinal_incremental_dropout_properties.py",
        "longitudinal-incremental-dropout",
    )
    s2_cases: tuple[tuple[str, str, str, int, Callable[[Any, Any], Any]], ...] = (
        (
            "G34.cross_fitted",
            "dropout_contrast_x2",
            "law_c",
            478,
            lambda m, f: m._fit(f, m.law.LAW_C),
        ),
        (
            "G34.in_sample",
            "dropout_contrast_x2_in_sample",
            "law_c",
            481,
            lambda m, f: m._fit(f, m.law.LAW_C, n_folds=1),
        ),
        (
            "G34.msm",
            "msm_log_delta",
            "law_c",
            479,
            lambda m, f: m._fit(f, m.law.LAW_C, reference=None, msm=m.MSM_LOG_DELTA),
        ),
        (
            "G34.weighted",
            "weighted_contrast_x0_5",
            "law_c",
            480,
            lambda m, f: m._fit(f.assign(weight=m.weight(f)), m.law.LAW_C, weights="weight"),
        ),
        (
            "G34.competing",
            "competing_contrast_x2_h3",
            "law_dc",
            442,
            lambda m, f: m._fit(f, m.law.LAW_DC, horizons=[3]),
        ),
        (
            "G34.clustered",
            "mean_x2__cluster_robust",
            "law_ck",
            454,
            lambda m, f: m._fit(f, m.law.LAW_CK.marginal, id="id"),
        ),
    )
    for row, cell, stream, line, fit in s2_cases:
        observed(
            row,
            s2_slug,
            cell,
            f"{s2}:{line}",
            lambda m, stream=stream: m._draw_fit_set(("interval_calibration", stream, 0, 2000, 1)),
            fit,
        )
    for row, folds, line in (
        ("G34.survival.cross_fitted", None, 525),
        ("G34.survival.in_sample", 1, 526),
    ):
        observed(
            row,
            s2_slug,
            "survival_grid" if folds is None else "survival_contrast_x2_h3_in_sample",
            f"{s2}:{line}",
            lambda m: m._draw_band((0, 1)),
            lambda m, f, folds=folds: m._fit(
                f, m.law.LAW_D, horizons=[3], **({} if folds is None else {"n_folds": folds})
            ),
        )

    # The sequential baselines: g-computation and inverse-probability weighting.
    baselines = "canonical_sequential_baselines.py:275"
    for scenario in ("d2_ice", "d2_hajek", "d2_ht", "g_ice", "g_hajek", "g_ht"):
        law, _, engine = scenario.partition("_")
        row = {"ice": "gcomp", "hajek": "ipw.hajek", "ht": "ipw.horvitz_thompson"}[engine]
        observed(
            f"G28.{law}.{row}",
            "sequential-baselines",
            scenario,
            baselines,
            lambda study, scenario=scenario: study.draw_from_seed(scenario, 600, 0)[0],
            lambda study, frame, law=law, engine=engine: study.fit(engine, law.upper(), frame),
        )

    # The parametric g-formula: the study's own fit of each law, at the size that keeps its
    # structure. ``G2`` and ``G5`` share one row.
    gformula = "canonical_parametric_gformula.py:285"
    for row, law in (
        ("G29.g1", "G1"),
        ("G29.g2_g5", "G2"),
        ("G29.g2_g5", "G5"),
        ("G29.g3", "G3"),
        ("G29.g4", "G4"),
    ):
        observed(
            row,
            "parametric-gformula",
            f"nice_{law.lower()}",
            gformula,
            lambda study, law=law: study.law_of(law).sample(400, 0),
            lambda study, frame, law=law: study.fit("nice", law, frame),
        )

    # The full-refit bootstrap study fits one sample per group, with its own folds and bootstrap.
    boot = "full_refit_bootstrap_properties.py:277"
    for row, label in (
        ("G01", "ratio_end_of_study"),
        ("G02", "boot_crossfit"),
        ("G05.in_sample", "boot_clustered"),
        ("G11.nodes4", "derived_survival"),
        ("G12.nodes4", "derived_crossfit"),
    ):
        observed(
            row,
            "full-refit-bootstrap-and-derived-contrasts",
            label,
            boot,
            lambda study, label=label: study._sample(label, study.GROUPS[label][0], 0),
            lambda study, frame, label=label: study._fit(label, study.GROUPS[label][0], 0, frame),
        )
    return tuple(cases)


def _gap_cases() -> tuple[RowCase, ...]:
    """Point cases that run a study's own fit: the C-TMLE sequences, C06 cells and default bands."""
    cases: list[RowCase] = []
    slug = "ctmle-oat-per-arm-strata-missing"
    where = "ctmle_oat_per_arm_strata_missing_properties.py:249"
    for row, name in (
        ("C06.missing_outcome", "natural_course_insample"),
        ("C06.missing_outcome_stacked", "missing_outcome_crossfit"),
        ("C06.strata_missing_stacked", "strata_missing"),
        ("C06.weighted_strata_missing_stacked", "weighted_strata_missing"),
    ):
        cases.append(
            _observed(
                row,
                slug,
                name,
                where,
                "ctmle_oat_per_arm_strata_missing_properties",
                lambda study, name=name: study.CONFIGURATION_BY_NAME[name].law.sample(600, 0)[0],
                lambda study, frame, name=name: study.fit(study.CONFIGURATION_BY_NAME[name], frame),
            )
        )
    sequences = "canonical_ctmle_sequences.py:903"
    for scenario, size in (("binary_lasso", 400), ("binary_learners", 2000)):
        cases.append(
            _seeded(
                f"C09.{scenario.removeprefix('binary_')}",
                "canonical-ctmle-sequences",
                scenario,
                sequences,
                "canonical_ctmle_sequences",
                scenario,
                size,
                lambda study, frame, scenario=scenario: study.fit_cleverly(frame, scenario, 1),
            )
        )
    cases.append(
        _seeded(
            "T02.binary",
            "default-simultaneous-bands",
            "default_fit",
            "default_bands.py:343",
            "default_bands",
            "default_fit",
            400,
        )
    )
    return tuple(cases)


def _vector_cases() -> tuple[RowCase, ...]:
    """The ``vector-mtp`` cases: a primary scenario fit, or a calibration cell's fit (X31)."""
    slug = "vector-mtp"
    primary = "canonical_vector_mtp.py:639"
    calibration = "vector_mtp_properties.py:317"
    cases: list[RowCase] = []
    for row, scenario in (
        ("T33.density", "point_vector_mixed"),
        ("T34", "point_vector_categorical"),
        ("G35.density", "longitudinal_vector_mixed"),
        ("G36.classifier", "longitudinal_vector_ordered_classifier"),
    ):
        cases.append(
            _seeded(
                row,
                slug,
                scenario,
                primary,
                "canonical_vector_mtp",
                scenario,
                400,
                lambda study, frame, scenario=scenario: study.fit(frame, scenario),
            )
        )
    for row, label, scenario in (
        ("T33.classifier", "point_vector_classifier", "point_vector_mixed"),
        ("G35.classifier", "longitudinal_vector_classifier", "longitudinal_vector_mixed"),
        (
            "G36.density",
            "longitudinal_vector_ordered_density",
            "longitudinal_vector_ordered_classifier",
        ),
    ):
        cases.append(
            _observed(
                row,
                slug,
                f"{label}__correctly_specified",
                calibration,
                "vector_mtp_properties",
                lambda module, scenario=scenario: module.study.draw_from_seed(scenario, 400, 0)[0],
                lambda module, frame, label=label: module.calibration_fit(label, frame),
            )
        )
    return tuple(cases)


#: The point-treatment cases: ordinary TMLE, DR-TMLE and C-TMLE.
CASES: tuple[RowCase, ...] = (
    *_t_cases(),
    *_d_cases(),
    *_c_cases(),
    *_gap_cases(),
    *_long_cases(),
    *_vector_cases(),
)
