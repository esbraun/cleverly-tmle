"""Study-setup objects whose summaries the contract test checks (X89 T5).

One fixture per kind of header, table and note a study, a question, a methods list, a
protocol or a simulation summary can print. Nothing here fits a model, so every fixture
builds in well under a second. The readers in ``_setup_readers`` parse each printed number
back and compare it with the object.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any

import numpy as np
import pandas as pd

from cleverly import (
    ATE,
    ATT,
    CausalStudy,
    ControlledDirectEffect,
    LongitudinalTreatment,
    ModifiedTreatmentPolicy,
    PointTreatment,
    RegimeMean,
    StudyProtocol,
)
from cleverly.datasets import make_clustered, make_linear_ate, make_longitudinal
from cleverly.interventions import Shift
from cleverly.utils.phases import PhaseProfile
from cleverly.validation import EstimandSummary, ReplicationFailure, StudyResult
from tests import discrete_law_cde
from tests import law_q_point as lawp

from . import Case
from ._setup_readers import (
    read_methods,
    read_phases,
    read_protocol,
    read_question,
    read_simulation,
    read_study,
)

__all__ = ["FIXTURES", "protocol", "simulation_record"]

POINT_ADJUSTMENT = ("W1", "W2", "W3", "W4")


def protocol(**overrides: Any) -> StudyProtocol:
    """A complete protocol in the shape of ``tests/unit/test_study_protocol.py``."""
    values: dict[str, Any] = {
        "target_population": "Adults eligible for transition navigation",
        "eligibility": ["Discharged alive", "Lives in the service area"],
        "time_zero": "Hospital discharge",
        "treatment_strategies": ["Offer navigation", "Usual care"],
        "treatment_versions": ["Calls at discharge and day seven", "No navigation call"],
        "outcome": "Top-box transition score",
        "horizon": "30 days after discharge",
        "intercurrent_event_handling": ["Use the score regardless of readmission"],
        "interference_unit": "Patient",
        "assumption_rationale": [
            "Recorded baseline variables cover the common causes used for adjustment"
        ],
    }
    values.update(overrides)
    return StudyProtocol(**values)


# ----------------------------------------------------------------- studies


@functools.cache
def point_study() -> CausalStudy:
    """A binary treatment adjusted for four covariates, no protocol."""
    frame = make_linear_ate(n=300, seed=1)[0]
    return CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=POINT_ADJUSTMENT),
    )


@functools.cache
def study_with_protocol() -> CausalStudy:
    """The same study with a protocol, so the header says it was recorded."""
    frame = make_linear_ate(n=300, seed=1)[0]
    return CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=POINT_ADJUSTMENT),
        protocol=protocol(),
    )


@functools.cache
def study_dropped_column() -> CausalStudy:
    """A constant adjustment column, which the data drops with a warning the summary quotes."""
    frame = make_linear_ate(n=300, seed=2)[0].assign(K=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "K")),
        )


@functools.cache
def clustered_study() -> CausalStudy:
    """Clusters, so the header counts them."""
    frame = make_clustered(n=120, cluster_size=10, seed=5)[0]
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2"), cluster="cluster"
        ),
    )


@functools.cache
def randomized_study() -> CausalStudy:
    """A randomized design that adjusts for nothing."""
    frame = make_linear_ate(n=200, seed=4)[0]
    return CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", randomized=True))


@functools.cache
def continuous_study() -> CausalStudy:
    """A continuous dose, so the header gives its mean and range."""
    return CausalStudy(
        lawp.sample(200, 1),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
    )


def _panel() -> pd.DataFrame:
    # "L2" reads like a roadmap id, so the time-varying column takes a plain name.
    frame = make_longitudinal(n=200, seed=33)[0]
    return frame.rename(columns={"L2": "blood_pressure", "W1": "age", "W2": "sex"})


@functools.cache
def longitudinal_study() -> CausalStudy:
    """Two treatment times with censoring and a time-varying covariate."""
    return CausalStudy(
        _panel(),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("age", "sex"),
            time_varying=((), ("blood_pressure",)),
            censoring=("C1", "C2"),
        ),
    )


@functools.cache
def markup_study() -> CausalStudy:
    """Column names with markup, which the HTML must escape."""
    frame = make_linear_ate(n=200, seed=3)[0].rename(
        columns={"Y": "<script>Y</script>", "W1": "W&1"}
    )
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="<script>Y</script>", treatment="A", adjustment=("W&1", "W2")
        ),
    )


# ----------------------------------------------------------------- questions


@functools.cache
def ate_effect() -> Any:
    """The average effect of a binary treatment."""
    return point_study().identify(ATE())


@functools.cache
def att_effect() -> Any:
    """The effect among the treated."""
    return point_study().identify(ATT())


@functools.cache
def restricted_effect() -> Any:
    """A treatment model restricted to two of the covariates, which adds a restriction."""
    frame = make_linear_ate(n=300, seed=1)[0]
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=POINT_ADJUSTMENT,
            treatment_adjustment=("W1", "W2"),
        ),
    )
    return study.identify(ATE())


@functools.cache
def mtp_effect() -> Any:
    """A modified treatment policy, which has two causal readings."""
    shifts = (Shift(0.0, cap=None), Shift(0.5, cap=None))
    return continuous_study().identify(ModifiedTreatmentPolicy(policies=shifts))


@functools.cache
def direct_effect() -> Any:
    """A controlled direct effect with missing outcomes, so every condition prints."""
    study = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    )
    return study.identify(ControlledDirectEffect(intermediate=0.0))


@functools.cache
def regime_effect() -> Any:
    """Two treatment regimes over two times."""
    return longitudinal_study().identify(RegimeMean({"always": 1, "never": 0}, reference="never"))


@functools.cache
def markov_effect() -> Any:
    """Three treatment times with a one-step history, which adds a model restriction."""
    rng = np.random.default_rng(4)
    n = 200
    columns: dict[str, Any] = {"age": rng.normal(size=n)}
    for t in (1, 2, 3):
        columns[f"bp_{t}"] = rng.normal(size=n)
        columns[f"A{t}"] = rng.integers(0, 2, size=n).astype(float)
    columns["Y"] = rng.normal(size=n)
    study = CausalStudy(
        pd.DataFrame(columns),
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2", "A3"),
            baseline=("age",),
            time_varying=(("bp_1",), ("bp_2",), ("bp_3",)),
            markov_order=1,
        ),
    )
    return study.identify(RegimeMean({"always": 1, "never": 0}, reference="never"))


@functools.cache
def unbound_effect() -> Any:
    """A question detached from its study, as after a round trip without the data."""
    return dataclasses.replace(ate_effect(), _study=None)


# ----------------------------------------------------------------- methods


@functools.cache
def ate_methods() -> Any:
    """The methods for an average effect."""
    return ate_effect().available_methods()


@functools.cache
def regime_methods() -> Any:
    """The methods for a regime contrast."""
    return regime_effect().available_methods()


# ----------------------------------------------------------------- simulation


def simulation_record(**overrides: Any) -> EstimandSummary:
    """One estimand's coverage record, in the shape of ``test_simulation_summary.py``."""
    values: dict[str, Any] = {
        "estimand": "ate",
        "truth": 0.5,
        "n": 500,
        "n_replicates": 4,
        "estimates": np.array([0.40, 0.50, 0.60, 0.55]),
        "std_errors": np.array([0.08, 0.09, 0.07, 0.08]),
        "covered": np.ones(4),
        "rejected": np.array([1.0, 1.0, 0.0, 1.0]),
    }
    values.update(overrides)
    return EstimandSummary(**values)


def _study_result(summaries: dict[str, EstimandSummary], **overrides: Any) -> StudyResult:
    values: dict[str, Any] = {
        "summaries": summaries,
        "replications": (),
        "failures": (),
        "n": 500,
        "n_replicates": 4,
        "alpha": 0.05,
        "label": "coverage check",
    }
    values.update(overrides)
    return StudyResult(**values)


@functools.cache
def simulation_passing() -> StudyResult:
    """Two estimands that cover, one a ratio whose spread is on the log scale."""
    ratio = np.array([1.40, 1.50, 1.60, 1.55])
    return _study_result(
        {
            "ate": simulation_record(),
            "rr": simulation_record(
                estimand="rr",
                truth=1.5,
                estimates=ratio,
                inference_estimates=np.log(ratio),
                std_errors=np.array([0.05, 0.06, 0.05, 0.04]),
            ),
        }
    )


@functools.cache
def simulation_undercovering() -> StudyResult:
    """An estimand whose spread is ten times too small, and one failed replicate."""
    failure = ReplicationFailure(
        replicate=4, seed=1234, error_type="LinAlgError", message="singular matrix"
    )
    return _study_result(
        {
            "ate": simulation_record(
                std_errors=np.full(4, 0.005), covered=np.zeros(4), rejected=np.ones(4)
            )
        },
        failures=(failure,),
        n_replicates=5,
    )


@functools.cache
def simulation_diagnostic() -> StudyResult:
    """Replicates that supply no interval, read as their plug-in diagnostic, with a mix."""
    return _study_result(
        {
            "ate": simulation_record(
                inference="no_cluster_variance_plugin",
                status_counts=(("no_cluster_variance_plugin", 3), ("influence_curve", 1)),
            )
        }
    )


# ----------------------------------------------------------------- phase timings


@functools.cache
def phase_profile() -> PhaseProfile:
    """Two nested phases and a parallel worker, with round times a reader can check."""
    workers = PhaseProfile(
        inclusive={"fit": 3.0}, exclusive={"fit": 3.0}, counts={"fit": 4}, total_seconds=3.0
    )
    return PhaseProfile(
        inclusive={"outer": 2.0, "inner": 1.5},
        exclusive={"outer": 0.5, "inner": 1.5},
        counts={"outer": 1, "inner": 3},
        total_seconds=2.5,
        workers=workers,
    )


def _quoted_warnings(obj: Any) -> tuple[str, ...]:
    study = getattr(obj, "_study", obj)
    data = getattr(study, "data", None)
    return tuple(str(record.message) for record in getattr(data, "warnings", ()) or ())


FIXTURES = {
    "point_study": Case(point_study, fidelity=read_study),
    "study_with_protocol": Case(study_with_protocol, fidelity=read_study),
    "study_dropped_column": Case(
        study_dropped_column, fidelity=read_study, quoted=_quoted_warnings
    ),
    "clustered_study": Case(clustered_study, fidelity=read_study),
    "randomized_study": Case(randomized_study, fidelity=read_study),
    "continuous_study": Case(continuous_study, fidelity=read_study),
    "longitudinal_study": Case(
        longitudinal_study,
        names=("Y", "A1", "A2", "C1", "C2", "age", "sex", "blood_pressure"),
        fidelity=read_study,
    ),
    "markup_study": Case(markup_study, fidelity=read_study),
    "ate_effect": Case(ate_effect, names=("Y", "A", *POINT_ADJUSTMENT), fidelity=read_question),
    "att_effect": Case(att_effect, names=("Y", "A", *POINT_ADJUSTMENT), fidelity=read_question),
    "restricted_effect": Case(
        restricted_effect, names=("Y", "A", *POINT_ADJUSTMENT), fidelity=read_question
    ),
    "mtp_effect": Case(mtp_effect, names=("Y", "A", "W"), fidelity=read_question),
    "direct_effect": Case(
        direct_effect, names=("Y", "A", "W", "Z", "Delta"), fidelity=read_question
    ),
    "regime_effect": Case(
        regime_effect,
        names=("Y", "A1", "A2", "C1", "C2", "age", "sex", "blood_pressure", "always", "never"),
        fidelity=read_question,
    ),
    "markov_effect": Case(
        markov_effect,
        names=("Y", "A1", "A2", "A3", "age", "bp_1", "bp_2", "bp_3", "always", "never"),
        fidelity=read_question,
    ),
    "unbound_effect": Case(
        unbound_effect, names=("Y", "A", *POINT_ADJUSTMENT), fidelity=read_question
    ),
    "ate_methods": Case(ate_methods, names=("Y", "A"), fidelity=read_methods),
    "regime_methods": Case(regime_methods, names=("Y", "always", "never"), fidelity=read_methods),
    "protocol": Case(protocol, fidelity=read_protocol),
    "simulation_passing": Case(simulation_passing, names=("ate", "rr"), fidelity=read_simulation),
    "simulation_undercovering": Case(
        simulation_undercovering,
        names=("ate",),
        fidelity=read_simulation,
        quoted=("singular matrix",),
    ),
    "simulation_diagnostic": Case(simulation_diagnostic, names=("ate",), fidelity=read_simulation),
    "phase_profile": Case(phase_profile, names=("outer", "inner", "fit"), fidelity=read_phases),
}
