"""Fixtures that fire the note codes no other module reaches (X89 T6, contract item 8).

Contract item 8 requires every note code of the vocabulary to be produced by a fixture, so the
note's text passes the house-style and uniqueness checks. A seam thread's fixtures cover the
notes of its own objects. These cover the notes that need a second configuration of an object
a seam already builds: a fit that recorded a warning of every category, a natural-course
question, a question no method fits, and so on. Each fixture names the code it is there to fire.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

from cleverly import NaturalCourseMean
from cleverly._summary import Summary
from cleverly._warnings import FitWarning
from cleverly.methods import MethodAvailabilities

from . import Case, point_tmle
from . import _record_readers as read
from . import _sensitivity_readers as sensitivity_read
from . import setup as setup_fixtures
from ._check_fidelity import header_values, read_refutation
from ._setup_readers import read_methods, read_question
from .sensitivity import NAMES as sensitivity_names

__all__ = ["FIXTURES"]

#: One recorded warning for each category the summary names a code for.
_WARNING_CATEGORIES = (
    "PositivityWarning",
    "WeightingWarning",
    "MonteCarloWarning",
    "ParameterSpaceWarning",
    "DataWarning",
)


@functools.cache
def warned_every_category() -> Any:
    """A fit that recorded one warning of each named category."""
    records = tuple(FitWarning(name, f"Recorded {name} text.") for name in _WARNING_CATEGORIES)
    return dataclasses.replace(point_tmle.single_ate(), fit_warnings=records)


@functools.cache
def natural_course_complete() -> Any:
    """The natural-course mean with every outcome recorded: as treated, needing no model."""
    return setup_fixtures.point_study().identify(NaturalCourseMean())


@functools.cache
def natural_course_missing() -> Any:
    """The natural-course mean with unrecorded outcomes, which a model must fill in."""
    from cleverly import CausalStudy, PointTreatment
    from cleverly.datasets import make_missing_outcome

    frame = make_missing_outcome(n=300, seed=4)[0]
    design = PointTreatment(
        outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), missingness="Delta"
    )
    return CausalStudy(frame, design=design).identify(NaturalCourseMean())


@functools.cache
def no_method_fits() -> Any:
    """A methods list in which no method fits: the shipped questions all have one."""
    records = setup_fixtures.ate_methods()
    refusal = next(record.refusal for record in records if not record.available)
    return MethodAvailabilities(
        (dataclasses.replace(record, available=False, refusal=refusal) for record in records),
        question=records.question,
    )


class FacadeDisplay:
    """What a notebook shows for ``result.diagnostics`` or ``result.sensitivity``.

    A facade has no ``summary()`` of its own: its ``_repr_html_`` renders the summary this
    class returns, so the contract can run its items over the same text. The two subclasses
    carry the facades' class names because the repr check reads them.
    """

    surface = ""

    def __init__(self, facade: Any) -> None:
        self.facade = facade

    def summary(self, detail: str = "short") -> Summary:
        from cleverly._assessment_summary import facade_summary
        from cleverly._summary import check_detail

        check_detail(detail)
        return facade_summary(self.surface, self.facade.capabilities)

    def _repr_html_(self) -> str:
        return str(self.facade._repr_html_())

    def __repr__(self) -> str:
        return repr(self.facade)


class DiagnosticsFacade(FacadeDisplay):
    surface = "diagnostics"


class SensitivityFacade(FacadeDisplay):
    surface = "sensitivity"


def read_facade(display: FacadeDisplay, text: str) -> None:
    """The listed and available counts equal the facade's own capability rows."""
    rows = display.facade.capabilities
    facts = header_values(text)
    assert facts["Checks"] == str(len(rows))
    assert facts["Available"] == str(sum(row.available for row in rows))
    for row in rows:
        assert row.operation in text


@functools.cache
def diagnostics_facade() -> FacadeDisplay:
    """The checks of a fit, some of which cannot run on it."""
    return DiagnosticsFacade(point_tmle.single_ate().diagnostics)


@functools.cache
def sensitivity_facade() -> FacadeDisplay:
    """The sensitivity analyses of a fit with no missing outcomes: three cannot run."""
    return SensitivityFacade(point_tmle.single_ate().sensitivity)


@functools.cache
def refute_future_treatment() -> Any:
    """A two-node panel refuted with a control the later treatment cannot move."""
    import warnings

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly.longitudinal import LTMLE
    from cleverly.validation import refute

    rng = np.random.default_rng(11)
    n = 400
    w, u = rng.normal(size=n), rng.normal(size=n)
    a1 = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w))).astype(float)
    l2 = rng.normal(size=n) + 0.5 * w + 0.5 * a1
    a2 = rng.binomial(1, 1 / (1 + np.exp(-0.8 * l2 - 1.0 * u))).astype(float)
    y1 = rng.binomial(1, 1 / (1 + np.exp(0.5 - 0.4 * a1 - 0.3 * w))).astype(float)
    y2 = rng.binomial(1, 1 / (1 + np.exp(0.5 - 0.4 * a2 - 0.3 * l2 - 0.3 * y1))).astype(float)
    panel = pd.DataFrame({"W": w, "U": u, "A1": a1, "Y1": y1, "L2": l2, "A2": a2, "Y2": y2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = LTMLE(
            {"always": 1, "never": 0},
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            simultaneous=False,
            random_state=3,
        ).fit(
            panel,
            outcome="Y2",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["Y1", "L2"]],
        )
        return refute(
            fit,
            estimand="ate_regimen[never vs always]",
            tests=("future_treatment",),
            future_treatment_control=panel["U"].to_numpy(),
            random_state=1,
        )


def _quiet(build: Any) -> Any:
    """Build without the warnings a compact fit raises."""

    @functools.wraps(build)
    def quiet() -> Any:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return build()

    return functools.cache(quiet)


@_quiet
def simulation_flagged() -> Any:
    """A g-formula report with two covariates at one time point and a large simulation share."""
    from . import longitudinal_records as records

    report = records.simulation()
    return dataclasses.replace(
        report,
        covariate_types=(*report.covariate_types, (2, "L3", "normal")),
        monte_carlo_share={**report.monte_carlo_share, "ey_regimen[always]": 0.25},
    )


@_quiet
def survival_projected() -> Any:
    """Event risks that the monotone projection moved at one level."""
    from . import longitudinal

    result = longitudinal.survival()
    moved = dataclasses.replace(
        result.repeats[0], isotonic_moves=(("always", None, 1, 0.31, 0.29),)
    )
    return dataclasses.replace(
        result,
        repeats=(moved, *result.repeats[1:]),
        config=dataclasses.replace(result.config, isotonic=True),
    )


@_quiet
def regime_report_unsupported() -> Any:
    """The overlap of three regimes, the first with three rows no history supports."""
    from . import longitudinal_records as records

    report = records.regime_report()
    first, *rest = report.regimes.items()
    regimes = {first[0]: dataclasses.replace(first[1], unsupported=3), **dict(rest)}
    return dataclasses.replace(report, regimes=regimes)


@_quiet
def policy_row_unsupported() -> Any:
    """The overlap of a dose-shifting policy with two rows whose shifted dose has no density."""
    from . import longitudinal_records as records

    return dataclasses.replace(records.policy_row(), unsupported=2)


# ----------------------------------------------------------------- sensitivity


def _sensitivity(name: str) -> Any:
    from . import sensitivity

    return sensitivity.FIXTURES[name]()


@_quiet
def surface_incremental() -> Any:
    """A simulated-confounder surface for an incremental intervention."""
    surface = _sensitivity("surface")
    return dataclasses.replace(surface, estimand="ey_ipsi[delta=2]")


@_quiet
def surface_learned_rule() -> Any:
    """A simulated-confounder surface for the value of a learned rule."""
    surface = _sensitivity("surface")
    return dataclasses.replace(surface, estimand="ey_learned_rule[tree]")


@_quiet
def surface_reads_observation() -> Any:
    """A surface whose weight also reads the response indicator, with bounded rows and
    folds drawn again for each cell."""
    surface = _sensitivity("surface")
    return dataclasses.replace(
        surface, observation_reads=("Delta",), bounded_share=0.125, refit_folds="redrawn"
    )


@_quiet
def surface_failed_cell() -> Any:
    """A surface in which one cell failed and one calibration has no finite strength."""
    from cleverly.sensitivity.simulated_confounding import SEPARATION_METHOD
    from cleverly.validation.simulation import ReplicationFailure

    surface = _sensitivity("surface")
    failure = ReplicationFailure(0, 0, "ValueError", "the refit failed")
    cells = (dataclasses.replace(surface.cells[0], failure=failure), *surface.cells[1:])
    calibration = dataclasses.replace(
        surface.calibrations[0], strength=float("nan"), method=SEPARATION_METHOD
    )
    return dataclasses.replace(
        surface, cells=cells, calibrations=(calibration, *surface.calibrations[1:])
    )


@_quiet
def evalue_truncated() -> Any:
    """An E-value whose converted lower bound fell outside the risk-ratio scale."""
    return dataclasses.replace(_sensitivity("evalue_converted"), truncated_bound=-0.2)


@_quiet
def evalue_conditions() -> Any:
    """An E-value that states the conditions of a censored and a per-arm ratio."""
    import importlib

    module = importlib.import_module("cleverly.sensitivity.evalue")

    value = _sensitivity("evalue_ratio")
    parts = (
        ("censoring", module._TIME_TO_EVENT_CONDITIONS),
        ("per_arm", module._PER_ARM_CONDITIONS),
    )
    return dataclasses.replace(value, note_parts=parts)


@_quiet
def bounds_diagnostic() -> Any:
    """A bound whose fit claims no interval, so its limits are a diagnostic."""
    return dataclasses.replace(_sensitivity("bounds"), inference="no_cluster_variance_plugin")


@_quiet
def benchmark_clipped() -> Any:
    """A benchmark of two covariates whose strengths exceed what the weights admit."""
    from . import sensitivity

    return sensitivity._linear().sensitivity.benchmark(covariates=("W1", "W2"))


FIXTURES = {
    "warned_every_category": warned_every_category,
    "natural_course_complete": Case(
        natural_course_complete, names=("Y", "A", "W1", "W2", "W3", "W4"), fidelity=read_question
    ),
    "natural_course_missing": Case(
        natural_course_missing, names=("Y", "A", "W1", "W2", "W3", "Delta"), fidelity=read_question
    ),
    "no_method_fits": Case(no_method_fits, names=("Y", "A"), fidelity=read_methods),
    # The check names are the method names the user types, which the style checks read as names.
    "simulation_flagged": Case(
        simulation_flagged, names=("L2", "L3", "Y"), fidelity=read.read_gformula_report
    ),
    "survival_projected": survival_projected,
    "regime_report_unsupported": Case(
        regime_report_unsupported, names=("screen",), fidelity=read.read_support_report
    ),
    "policy_row_unsupported": Case(
        policy_row_unsupported, names=("up",), fidelity=read.read_policy_support
    ),
    "surface_incremental": Case(
        surface_incremental, names=sensitivity_names, fidelity=sensitivity_read.surface
    ),
    "surface_learned_rule": Case(
        surface_learned_rule, names=sensitivity_names, fidelity=sensitivity_read.surface
    ),
    "surface_reads_observation": Case(
        surface_reads_observation, names=sensitivity_names, fidelity=sensitivity_read.surface
    ),
    "surface_failed_cell": Case(
        surface_failed_cell, names=sensitivity_names, fidelity=sensitivity_read.surface
    ),
    "evalue_truncated": Case(
        evalue_truncated, names=sensitivity_names, fidelity=sensitivity_read.evalue
    ),
    "evalue_conditions": Case(
        evalue_conditions, names=sensitivity_names, fidelity=sensitivity_read.evalue
    ),
    "bounds_diagnostic": Case(
        bounds_diagnostic, names=sensitivity_names, fidelity=sensitivity_read.bounds
    ),
    "benchmark_clipped": Case(
        benchmark_clipped, names=sensitivity_names, fidelity=sensitivity_read.benchmark
    ),
    "refute_future_treatment": Case(
        refute_future_treatment,
        names=("W", "U", "A1", "A2", "L2", "Y1", "Y2"),
        fidelity=read_refutation,
    ),
    "diagnostics_facade": Case(
        diagnostics_facade,
        names=("support", "nuisance_models", "score_equations", "truncation_curve", "refute"),
        fidelity=read_facade,
        sectioned=False,
    ),
    "sensitivity_facade": Case(
        sensitivity_facade,
        names=(
            "omitted_confounding",
            "robustness_value",
            "elements",
            "benchmark",
            "simulated_confounding",
            "contour",
            "evalue",
            "missingness",
            "tipping_gamma",
            "marginal_bounds",
        ),
        fidelity=read_facade,
        sectioned=False,
    ),
}
