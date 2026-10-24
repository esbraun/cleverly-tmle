"""A broad set of identified questions, one per kind of identification record (X89 T5).

``tests/unit/test_setup_summaries.py`` pins which plain-language condition each statement of
each question reaches, so a statement that a new prefix captures by mistake fails there.
Nothing here fits a model.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from cleverly import (
    ATC,
    ATE,
    ATT,
    CausalStudy,
    ControlledDirectEffect,
    CounterfactualMean,
    IncrementalEffect,
    IncrementalMean,
    LongitudinalTreatment,
    ModifiedTreatmentPolicy,
    NaturalCourseMean,
    OddsRatio,
    PointTreatment,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
    RegimeContrast,
    RegimeMean,
    RiskRatio,
)
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_longitudinal,
    make_multi_arm,
)
from cleverly.interventions import Incremental, Shift
from tests import discrete_law_cde, discrete_law_mar
from tests import law_q_point as lawp
from tests import law_q_tuned as lawq
from tests import longitudinal_mtp as mtp

__all__ = ["QUESTIONS"]


@functools.cache
def _binary_treatment() -> CausalStudy:
    frame = make_linear_ate(n=200, seed=1)[0]
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )


@functools.cache
def _binary_outcome() -> CausalStudy:
    frame = make_binary_outcome(n=200, seed=2)[0]
    covariates = tuple(name for name in frame.columns if name not in ("Y", "A"))
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=covariates)
    )


@functools.cache
def _multi_arm() -> CausalStudy:
    frame = make_multi_arm(n=180, seed=17)[0]
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
    )


@functools.cache
def _dose() -> CausalStudy:
    return CausalStudy(
        lawp.sample(200, 1),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
    )


@functools.cache
def _missing_outcome() -> CausalStudy:
    return CausalStudy(
        discrete_law_mar.frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"),
    )


@functools.cache
def _intermediate() -> CausalStudy:
    return CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), missingness="Delta", intermediate="Z"
        ),
    )


@functools.cache
def _randomized() -> CausalStudy:
    frame = make_linear_ate(n=200, seed=4)[0]
    return CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", randomized=True))


@functools.cache
def _uncensored_panel() -> CausalStudy:
    rng = np.random.default_rng(6)
    n = 200
    frame = pd.DataFrame({"W": rng.normal(size=n), "A1": rng.integers(0, 2, size=n).astype(float)})
    frame["bp"] = rng.normal(size=n)
    frame["A2"] = rng.integers(0, 2, size=n).astype(float)
    frame["Y"] = rng.normal(size=n)
    return CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y", treatment=("A1", "A2"), baseline=("W",), time_varying=((), ("bp",))
        ),
    )


@functools.cache
def _panel() -> CausalStudy:
    frame = make_longitudinal(n=200, seed=33)[0].rename(columns={"L2": "bp"})
    return CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("bp",)),
            censoring=("C1", "C2"),
        ),
    )


@functools.cache
def _dose_panel() -> CausalStudy:
    return CausalStudy(
        lawq.sample(200, 1).rename(columns={"L2": "bp"}),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W",),
            time_varying=((), ("bp",)),
            continuous_treatment=("A1", "A2"),
        ),
    )


def _quiet(build: Callable[[], Any]) -> Callable[[], Any]:
    @functools.cache
    def wrapped() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return build()

    return wrapped


SHIFTS = (Shift(0.0, cap=None), Shift(0.5, cap=None))

QUESTIONS: dict[str, Callable[[], Any]] = {
    name: _quiet(build)
    for name, build in {
        "ate": lambda: _binary_treatment().identify(ATE()),
        "att": lambda: _binary_treatment().identify(ATT()),
        "atc": lambda: _binary_treatment().identify(ATC()),
        "mean": lambda: _binary_treatment().identify(CounterfactualMean(treatment=1)),
        "natural_course": lambda: _binary_treatment().identify(NaturalCourseMean()),
        "incremental": lambda: _binary_treatment().identify(
            IncrementalMean(interventions=[Incremental(2.0)])
        ),
        "risk_ratio": lambda: _binary_outcome().identify(RiskRatio()),
        "odds_ratio": lambda: _binary_outcome().identify(OddsRatio()),
        "par": lambda: _binary_outcome().identify(PopulationAttributableRisk()),
        "paf": lambda: _binary_outcome().identify(PopulationAttributableFraction()),
        "multi_arm_ate": lambda: _multi_arm().identify(ATE(reference="low")),
        "multi_arm_att": lambda: _multi_arm().identify(ATT(reference="low")),
        "multi_arm_atc": lambda: _multi_arm().identify(ATC(reference="low")),
        "dose_policy": lambda: _dose().identify(ModifiedTreatmentPolicy(policies=SHIFTS)),
        "missing_ate": lambda: _missing_outcome().identify(ATE()),
        "missing_natural_course": lambda: _missing_outcome().identify(NaturalCourseMean()),
        "direct_effect": lambda: _intermediate().identify(ControlledDirectEffect(intermediate=0.0)),
        "randomized_ate": lambda: _randomized().identify(ATE()),
        "uncensored_regime": lambda: _uncensored_panel().identify(
            RegimeMean({"always": 1, "never": 0}, reference="never")
        ),
        "regime": lambda: _panel().identify(
            RegimeMean({"always": 1, "never": 0}, reference="never")
        ),
        "regime_incremental": lambda: _panel().identify(
            IncrementalEffect([Incremental(1.0), Incremental(2.0)])
        ),
        "regime_policy": lambda: _dose_panel().identify(
            RegimeContrast(mtp.regimens(("natural", "up")), reference="natural")
        ),
    }.items()
}
