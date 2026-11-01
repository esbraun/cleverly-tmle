"""Combined reports whose summaries the contract test checks (X89 T3).

``result.validate()``, ``result.diagnostics.run_all()``, ``result.sensitivity.run_all()`` and
``result.assess()`` on a compact point fit (300 rows, linear and logistic learners) and on a
longitudinal fit, plus reports built by hand so that every status prints: a failed check, a
warning, a deferral, an unavailable and a not-applicable row, and a name two surfaces share.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly._assessment_summary import quoted_reasons
from cleverly.assessment import (
    AssessmentItem,
    AssessmentReport,
    AssessmentStatus,
    DiagnosticReport,
    ValidationReport,
)
from cleverly.datasets import make_linear_ate

from . import Case, point_methods
from . import longitudinal as longitudinal_fits
from ._battery_reader import read_battery

__all__ = ["FIXTURES"]

_COVARIATES = ("W1", "W2", "W3", "W4")


@functools.cache
def _fit() -> Any:
    frame = make_linear_ate(n=300, seed=2)[0]
    design = PointTreatment(outcome="Y", treatment="A", adjustment=_COVARIATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CausalStudy(frame, design=design)
            .identify(ATE())
            .estimate(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                random_state=0,
            )
        )


def _quiet(build: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def validation() -> ValidationReport:
    """The quick checks of a point fit."""
    return _quiet(_fit().validate)


def diagnostics() -> DiagnosticReport:
    """The default diagnostics of a point fit."""
    return _quiet(_fit().diagnostics.run_all)


def sensitivity() -> DiagnosticReport:
    """The default sensitivity analyses of a point fit."""
    return _quiet(_fit().sensitivity.run_all)


def assessment() -> AssessmentReport:
    """Every row of a point fit."""
    return _quiet(_fit().assess)


def longitudinal() -> AssessmentReport:
    """Every row of a longitudinal fit."""
    return _quiet(longitudinal_fits.means().assess)


def collaborative() -> AssessmentReport:
    """Every row of a collaborative fit, whose interval carries a label."""
    return _quiet(point_methods.greedy().assess)


def drtmle() -> AssessmentReport:
    """Every row of a DR-TMLE fit, with its corrections check."""
    return _quiet(point_methods.drtmle().assess)


def _item(name: str, status: AssessmentStatus, detail: str) -> AssessmentItem:
    return AssessmentItem(name, status, detail, (f"call result.diagnostics.{name}()",))


def every_status() -> AssessmentReport:
    """Witness: one row of each status, and a name the validation and sensitivity share."""
    return AssessmentReport(
        validation=ValidationReport(
            (
                _item("score_equations", AssessmentStatus.FAILED, "2 of 3 equations missed."),
                _item("support", AssessmentStatus.PASSED, "No unit was capped."),
            )
        ),
        diagnostics=DiagnosticReport(
            (
                _item(
                    "nuisance_models", AssessmentStatus.WARNING, "The treatment model separates."
                ),
                _item("refute", AssessmentStatus.DEFERRED, "not run by default; pass a flag"),
                _item("corrections", AssessmentStatus.UNAVAILABLE, "this fit has none"),
            )
        ),
        sensitivity=DiagnosticReport(
            (
                _item("missingness", AssessmentStatus.NOT_APPLICABLE, "no outcome is missing"),
                _item("evalue", AssessmentStatus.COMPLETED, "An E-value of 2.10."),
                _item("score_equations", AssessmentStatus.UNAVAILABLE, "not for this estimand"),
            ),
            surface="sensitivity",
        ),
    )


def failed_validation() -> ValidationReport:
    """Witness: validation with a failed check and a deferred one."""
    return ValidationReport(
        (
            _item("score_equations", AssessmentStatus.FAILED, "2 of 3 equations missed."),
            _item("nuisance_models", AssessmentStatus.DEFERRED, "needs the stored models"),
        )
    )


def _quoted(report: Any) -> tuple[str, ...]:
    items = (
        [item for _, item in report._presented()]
        if isinstance(report, AssessmentReport)
        else list(report.items)
    )
    return quoted_reasons(items)


def _case(build: Any) -> Case:
    return Case(build, names=("Y", "A", *_COVARIATES), fidelity=read_battery, quoted=_quoted)


FIXTURES: dict[str, Case] = {
    "battery_validation": _case(validation),
    "battery_diagnostics": _case(diagnostics),
    "battery_sensitivity": _case(sensitivity),
    "battery_assessment": _case(assessment),
    "battery_longitudinal": _case(longitudinal),
    "battery_collaborative": _case(collaborative),
    "battery_drtmle": _case(drtmle),
    "battery_every_status": _case(every_status),
    "battery_failed_validation": _case(failed_validation),
}
