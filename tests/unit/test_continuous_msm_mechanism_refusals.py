"""A continuous-dose MSM refuses a missing outcome and an intermediate variable, by name.

The clever covariate that this package builds for a continuous-dose MSM divides by the treatment
density at the observed dose and at each dose of the integration grid.  With a missing outcome or
an intermediate variable, that construction must also divide by the second mechanism at each of
those doses, and no targeting step for it is written.  Before RM32 in ``docs/roadmap.md``, the in-sample fit raised
``ValueError`` from the nuisance fit after two learner fits, and the cross-fitted fit met the
F21 refusal, whose remedy is that in-sample fit.  X10 in ``docs/roadmap.md`` tracks the
construction.  This module pins these things:

* the fit refuses with ``CapabilityError`` before any learner, in sample and cross-fitted, with
  ``delta=``, ``intermediate=`` and both, and names each missing mechanism;
* ``refit`` on new data with a missing outcome refuses the same way;
* the cross-fitted fit is not sent to the in-sample fit;
* ``CausalStudy.identify`` refuses ``MSMProjection`` on a continuous design with a missing
  outcome, and ``estimate`` fits no learner;
* the rule keys on a missing outcome: a declared indicator with every outcome observed fits,
  and reports the number of the fit without it;
* a mutation that removes the estimator check fails every fit witness, a mutation that removes
  the identification check moves the study refusal to ``estimate``, and a mutation that
  removes both lets a learner fit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cleverly.study as study_module
from cleverly import CausalStudy, PointTreatment
from cleverly.data import CausalData
from cleverly.datasets import make_missing_outcome
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.msm import MSM
from cleverly.study import MSMProjection
from tests.conftest import linear_in_sample
from tests.unit._declaration_support import (
    assert_every_witness_fails,
    assert_refused,
    assert_refused_before_any_call,
    tmle_module,
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners

COVARIATES = ["W1", "W2", "W3"]
DOSES = (-1.0, 0.0, 0.5, 1.0, 2.0)
X10 = "docs/roadmap.md X10"
GRID = "at each dose of the integration grid"
RESPONSE = "P(Delta = 1 | dose, W)"
INTERMEDIATE = "P(Z = z | dose, W)"
SUBJECT = "An MSM (msm=) on a continuous dose"


def dose_frame(n: int = 200) -> pd.DataFrame:
    """``make_missing_outcome(n, seed=4)``, a dose ``A + N(0, 1)``, and a binary ``Z``."""
    frame, _ = make_missing_outcome(n=n, seed=4)
    frame["dose"] = frame["A"] + np.random.default_rng(0).normal(size=n)
    frame["Z"] = (np.random.default_rng(1).uniform(size=n) < 0.5).astype(float)
    return frame


def complete(frame: pd.DataFrame) -> pd.DataFrame:
    """Every outcome observed, with ``Delta`` kept as a declared column of ones."""
    frame = frame.copy()
    frame.loc[frame["Delta"] == 0, "Y"] = 0.0
    frame["Delta"] = 1
    return frame


def estimator(learners: dict[str, Any], **extra: Any) -> TMLE:
    settings: dict[str, Any] = {
        "msm": MSM.linear(doses=DOSES),
        "cross_fit": False,
        "density_bins": 6,
        "simultaneous": False,
        "random_state": 0,
        "estimands": None,
        **learners,
        **extra,
    }
    return TMLE(**settings)


def fit(est: TMLE, frame: pd.DataFrame, **columns: Any) -> Any:
    return est.fit(
        frame,
        outcome="Y",
        treatment="dose",
        covariates=COVARIATES,
        treatment_kind="continuous",
        **columns,
    )


def study(frame: pd.DataFrame) -> CausalStudy:
    design = PointTreatment(
        outcome="Y",
        treatment="dose",
        adjustment=tuple(COVARIATES),
        missingness="Delta",
        treatment_kind="continuous",
    )
    return CausalStudy(frame, design=design)


def projection() -> MSMProjection:
    return MSMProjection(MSM.linear(doses=DOSES))


#: A refused fit: the frame, the fit columns, the estimator settings, and the fragments.
Row = tuple[Callable[[], pd.DataFrame], dict[str, str], dict[str, Any], tuple[str, ...]]

DIRECT: dict[str, Row] = {
    "missing outcomes, in sample": (
        dose_frame,
        {"delta": "Delta"},
        {},
        (SUBJECT, "missing outcomes (delta=)", RESPONSE, GRID, X10),
    ),
    "missing outcomes, cross-fitted": (
        dose_frame,
        {"delta": "Delta"},
        {"cross_fit": True, "n_folds": 5},
        (SUBJECT, "missing outcomes (delta=)", RESPONSE, GRID, X10),
    ),
    "intermediate": (
        lambda: complete(dose_frame()).drop(columns="Delta"),
        {"intermediate": "Z"},
        {},
        (SUBJECT, "an intermediate variable (intermediate=)", INTERMEDIATE, GRID, X10),
    ),
    "intermediate, cross-fitted": (
        lambda: complete(dose_frame()).drop(columns="Delta"),
        {"intermediate": "Z"},
        {"cross_fit": True, "n_folds": 5},
        (SUBJECT, "an intermediate variable (intermediate=)", INTERMEDIATE, GRID, X10),
    ),
    "both": (
        dose_frame,
        {"delta": "Delta", "intermediate": "Z"},
        {},
        (
            SUBJECT,
            "missing outcomes (delta=) and an intermediate variable (intermediate=)",
            f"{RESPONSE} and the intermediate mechanism {INTERMEDIATE}",
            X10,
        ),
    ),
}

CROSS_FITTED = "missing outcomes, cross-fitted"


def fit_witness(name: str) -> None:
    frame, columns, extra, fragments = DIRECT[name]
    assert_refused_before_any_call(
        lambda: fit(estimator(never_fit_learners(), **extra), frame(), **columns),
        None,
        "",
        *fragments,
    )


def refit_witness() -> None:
    """``refit`` on new data with a missing outcome, which the replay and the refutations use."""
    data = CausalData.from_frame(
        dose_frame(),
        outcome="Y",
        treatment="dose",
        covariates=COVARIATES,
        delta="Delta",
        treatment_kind="continuous",
    )
    assert_refused_before_any_call(
        lambda: estimator(never_fit_learners()).refit(data),
        None,
        "",
        SUBJECT,
        "missing outcomes (delta=)",
        RESPONSE,
        X10,
    )


def identify_witness() -> None:
    assert_refused(
        lambda: study(dose_frame()).identify(projection()),
        CapabilityError,
        "MSMProjection on a continuous dose",
        "missing outcomes (PointTreatment(missingness=...))",
        RESPONSE,
        GRID,
        X10,
    )


def estimate_witness() -> None:
    """The study refuses before any learner, whichever check raises."""
    assert_refused_before_any_call(
        lambda: study(dose_frame()).estimate(
            projection(),
            **never_fit_learners(),
            cross_fit=False,
            density_bins=6,
            simultaneous=False,
        ),
        None,
        "",
        RESPONSE,
        X10,
    )


class TestTheFitRefusesBeforeAnyLearner:
    @pytest.mark.parametrize("name", list(DIRECT))
    def test_the_fit_names_the_missing_mechanism(self, name: str) -> None:
        fit_witness(name)

    def test_a_cross_fitted_fit_is_not_sent_in_sample(self) -> None:
        """The F21 remedy is the in-sample fit, which this composition cannot run either."""
        frame, columns, extra, _ = DIRECT[CROSS_FITTED]
        with pytest.raises(CapabilityError) as raised:
            fit(estimator(never_fit_learners(), **extra), frame(), **columns)
        message = str(raised.value)
        assert "F21" not in message
        assert "cross_fit=False" not in message
        assert "in sample or cross-fitted" in message

    def test_a_refit_on_new_data_names_the_missing_mechanism(self) -> None:
        refit_witness()


class TestTheStudyRefusesAtIdentify:
    def test_identify_names_the_missing_mechanism(self) -> None:
        identify_witness()

    def test_estimate_fits_no_learner(self) -> None:
        estimate_witness()


class TestTheRuleKeysOnAMissingOutcome:
    def test_a_declared_indicator_with_every_outcome_observed_fits(self) -> None:
        """A rule keyed on ``delta_name is not None`` would refuse the first fit."""
        frame = complete(dose_frame())
        declared = fit(estimator(linear_in_sample()), frame, delta="Delta").single()
        undeclared = fit(estimator(linear_in_sample()), frame.drop(columns="Delta")).single()
        assert "msm[a]" in declared.estimates
        for key, estimate in undeclared.estimates.items():
            assert declared.estimates[key].psi == pytest.approx(estimate.psi, abs=1e-12)

    def test_identify_admits_a_declared_indicator_with_every_outcome_observed(self) -> None:
        effect = study(complete(dose_frame())).identify(projection())
        assert isinstance(effect.estimand, MSMProjection)


class TestTheWitnessesHaveTeeth:
    @staticmethod
    def remove(monkeypatch: pytest.MonkeyPatch, *modules: Any) -> None:
        for module in modules:
            monkeypatch.setattr(module, "refuse_continuous_msm_mechanisms", lambda *a, **k: None)

    def test_removing_the_fit_check_fails_every_fit_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The in-sample rows reach ``NeverFit.fit``, and the cross-fitted rows meet another
        refusal.  The refit witness runs last and in sample, so a learner has then fitted."""
        self.remove(monkeypatch, tmle_module)
        witnesses = [lambda name=name: fit_witness(name) for name in DIRECT]
        assert_every_witness_fails([*witnesses, refit_witness])
        assert NeverFit.calls > 0
        identify_witness()

    def test_removing_the_identify_check_moves_the_refusal_to_estimate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The estimator check still refuses before any learner."""
        self.remove(monkeypatch, study_module)
        assert_every_witness_fails([identify_witness])
        estimate_witness()

    def test_removing_both_checks_lets_a_learner_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.remove(monkeypatch, study_module, tmle_module)
        assert_every_witness_fails([estimate_witness])
        assert NeverFit.calls > 0
