"""Shared repeated-sampling cells for the multi-arm point-treatment family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from cleverly.data import CausalData
from cleverly.datasets import MultiArmDGP
from cleverly.estimators import CTMLE, DRTMLE
from cleverly.utils.bounds import expit
from tests.studies import multi_arm_common
from tests.studies.evidence.properties import PropertyCell
from tests.studies.evidence.property_verdicts import FOLD_POLICY_FAMILY as FOLD_POLICY_FAMILY
from tests.studies.evidence.property_verdicts import (
    FOLD_POLICY_REFERENCE_CELL,
    fold_policy_strata,
    make_fold_policy_cells,
)
from tests.studies.evidence.property_verdicts import fold_policy_seed as fold_policy_seed
from tests.studies.evidence.registry import StudyRecord

# At 600 draws the 99% bias interval spends about 0.105 empirical SD on Monte
# Carlo error, leaving more than half of the fixed 0.25-SD equivalence margin for a
# real departure.  Four hundred size-ladder draws resolve a calibrated 95% coverage
# rate above the 0.90 floor; calibration gets the larger budget because its two-sided
# 7% SE-ratio band is the tighter claim.
ROBUST_REPLICATES = 600
RATE_REPLICATES = 400
CALIBRATION_REPLICATES = 1_600
NULL_REPLICATES = 400
GENERATED_REPLICATES = 800
RATE_SIZES = (500, 2000, 8000)
CALIBRATION_N = 2000
ESTIMAND = "ate[medium vs high]"


@dataclass(frozen=True)
class Sampler:
    """Callable wrapper that makes a :class:`MultiArmDGP` a coverage-study law."""

    effect: float = 0.6

    @property
    def name(self) -> str:
        """What the wrapped law calls itself, so a finding can name it."""
        return str(multi_arm_common.law(effect=self.effect).name)

    def truth(self) -> dict[str, float]:
        """The wrapped law's own integral, for a cell that declares this sampler.

        A cell carries a sampler rather than a ``DGP`` here, and the truth-binding test
        recomputes every committed truth from the law a cell names.  Exposing it is what
        lets a multi-arm study declare its cells: without it the test would have no law to
        integrate, and a row that swapped one could publish a truth belonging to the law it
        replaced.

        Returns
        -------
        dict
            Arm means, reference-arm differences, risk ratios and odds ratios.
        """
        return multi_arm_common.truth_for(multi_arm_common.law(effect=self.effect))

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        process = multi_arm_common.law(effect=self.effect)
        frame, _ = process.sample(n, seed=seed, backend="pandas")
        return frame, multi_arm_common.truth_for(process)


@dataclass(frozen=True)
class SelectorSampler:
    """A multi-arm confounder/instrument/predictor law that makes selection load-bearing.

    The instrument is strong enough to make a selector's choice change the answer.  That is
    also strong enough to push a material share of units past the estimator's declared
    truncation bounds, which is why :meth:`process` is public: the selector row states that
    condition, and ``tests/unit/test_multi_arm_studies.py`` asserts it off this law rather
    than off a second copy of these coefficients.
    """

    def process(self) -> MultiArmDGP:
        """The law itself, so a control can measure it without restating its coefficients."""
        base = multi_arm_common.law()

        def arm_logits(w: np.ndarray) -> np.ndarray:
            return np.column_stack(
                (
                    np.zeros(len(w)),
                    0.9 * w[:, 0] + 1.5 * w[:, 1],
                    -0.7 * w[:, 0] - 1.3 * w[:, 1],
                )
            )

        def outcome_mean(w: np.ndarray, arm: int) -> np.ndarray:
            code = multi_arm_common.LABELS.index(base.labels[arm])
            return expit(-0.5 + 0.55 * code + 1.2 * w[:, 0] + 0.8 * w[:, 2])

        return replace(
            base,
            name="multi_arm_selector_instrument",
            arm_logits=arm_logits,
            outcome_mean=outcome_mean,
        )

    @property
    def name(self) -> str:
        """What the instrument law calls itself, so a finding can name it."""
        return str(self.process().name)

    def truth(self) -> dict[str, float]:
        """The instrument law's own integral, for a cell that declares this sampler.

        Returns
        -------
        dict
            Arm means, reference-arm differences, risk ratios and odds ratios.
        """
        return multi_arm_common.truth_for(self.process())

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        process = self.process()
        frame, _ = process.sample(n, seed=seed, backend="pandas")
        return frame, multi_arm_common.truth_for(process)


def correct_outcome():  # type: ignore[no-untyped-def]
    return lambda: LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


def correct_treatment():  # type: ignore[no-untyped-def]
    return lambda: LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


def oracle_outcome(effect: float = 0.6):  # type: ignore[no-untyped-def]
    process = multi_arm_common.law(effect=effect)
    return lambda: multi_arm_common.OracleMultiOutcome(process)


def wrong_outcome():  # type: ignore[no-untyped-def]
    return lambda: DummyClassifier(strategy="prior")


def wrong_treatment():  # type: ignore[no-untyped-def]
    return lambda: DummyClassifier(strategy="prior")


def robustness_cells(
    *,
    seed: int,
    misspecified_outcome: Callable[[], Any] | None = None,
    misspecified_treatment: Callable[[], Any] | None = None,
) -> tuple[PropertyCell, ...]:
    """The four nuisance regimes, over the shared law and the shared seed stream.

    A method may supply its own misspecified nuisances.  The default pair returns the sample
    prior, which is the plainest thing a wrong model can be and the right control for a method
    that fits nothing else off it.  DR-TMLE is not such a method: its guard regressions take
    the *other* nuisance as their single regressor, so a constant one hands them a constant
    design and the reported standard error stops meaning anything.  That method passes a
    covariate-dependent misspecification instead, and its cells then measure the union model
    rather than a degenerate reduced regression.
    """
    law = Sampler()
    outcome_wrong = misspecified_outcome or wrong_outcome()
    treatment_wrong = misspecified_treatment or wrong_treatment()
    configurations = (
        ("both_correct", correct_outcome(), correct_treatment(), "positive"),
        ("outcome_correct", correct_outcome(), treatment_wrong, "positive"),
        ("treatment_correct", outcome_wrong, correct_treatment(), "positive"),
        ("both_wrong", outcome_wrong, treatment_wrong, "control"),
    )
    sizes = {"treatment_correct": 2000}
    return tuple(
        PropertyCell(
            "double_robustness",
            cell,
            law,
            outcome,
            treatment,
            sizes.get(cell, 1000),
            ROBUST_REPLICATES,
            seed + index,
            role=role,
            estimand=ESTIMAND,
        )
        for index, (cell, outcome, treatment, role) in enumerate(configurations)
    )


def asymptotic_cells(*, seed: int, include_null_power: bool = True) -> tuple[PropertyCell, ...]:
    cells: list[PropertyCell] = []
    for index, size in enumerate(RATE_SIZES):
        cells.append(
            PropertyCell(
                "root_n_and_efficiency",
                f"n_{size}",
                Sampler(),
                correct_outcome(),
                correct_treatment(),
                size,
                RATE_REPLICATES,
                seed + 100 + index,
                estimand=ESTIMAND,
            )
        )
    cells.append(
        PropertyCell(
            "interval_calibration",
            "correctly_specified",
            Sampler(),
            oracle_outcome(),
            correct_treatment(),
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            seed + 200,
            estimand=ESTIMAND,
        )
    )
    if include_null_power:
        cells.extend(
            (
                PropertyCell(
                    "type_i_error",
                    "sharp_null",
                    Sampler(effect=0.0),
                    correct_outcome(),
                    correct_treatment(),
                    1200,
                    NULL_REPLICATES,
                    seed + 300,
                    estimand=ESTIMAND,
                ),
                PropertyCell(
                    "power",
                    "alternative",
                    Sampler(effect=0.6),
                    correct_outcome(),
                    correct_treatment(),
                    1200,
                    NULL_REPLICATES,
                    seed + 301,
                    estimand=ESTIMAND,
                ),
            )
        )
    return tuple(cells)


def oat_robustness_cells(*, seed: int) -> tuple[PropertyCell, ...]:
    return (
        PropertyCell(
            "robustness_contract",
            "outcome_correct",
            Sampler(),
            oracle_outcome(),
            wrong_treatment(),
            3000,
            ROBUST_REPLICATES,
            seed,
            estimand=ESTIMAND,
        ),
        PropertyCell(
            "robustness_contract",
            "outcome_wrong",
            Sampler(),
            wrong_outcome(),
            wrong_treatment(),
            1000,
            ROBUST_REPLICATES,
            seed + 1,
            role="control",
            estimand=ESTIMAND,
        ),
    )


def generated_design_cells(*, seed: int) -> tuple[PropertyCell, ...]:
    """Paired known and estimated outcome-adaptive designs."""
    return (
        PropertyCell(
            "generated_design",
            "oracle_design",
            Sampler(),
            oracle_outcome(),
            correct_treatment(),
            1000,
            GENERATED_REPLICATES,
            seed,
            estimand=ESTIMAND,
        ),
        PropertyCell(
            "generated_design",
            "estimated",
            Sampler(),
            correct_outcome(),
            correct_treatment(),
            1000,
            GENERATED_REPLICATES,
            seed,
            estimand=ESTIMAND,
        ),
    )


#: The selector paths the necessity family scores, and the forced path it scores them against.
#: All four run the same law, the same learners and the same draws, so the only thing that
#: differs between a positive cell and the control is which mechanism paths the selector may
#: reach.  Every declared strategy gets its own cell: a family that scored one strategy would
#: report a verdict about the selector while exercising a third of it, and the discrete ladder
#: below is exactly the path that behaves differently from the other two on this law.
SELECTOR_PATHS = ("greedy", "ordered", "discrete")
SELECTOR_CONTROL = "empty_control"


def selector_cells(*, seed: int) -> tuple[PropertyCell, ...]:
    """Each declared selector path, and the forced empty path, on identical draws."""
    return tuple(
        PropertyCell(
            "selector_necessity",
            cell,
            SelectorSampler(),
            wrong_outcome(),
            correct_treatment(),
            1500,
            RATE_REPLICATES,
            seed,
            role="control" if cell == SELECTOR_CONTROL else "positive",
            estimand=ESTIMAND,
        )
        for cell in (*SELECTOR_PATHS, SELECTOR_CONTROL)
    )


# ------------------------------------------------------------- fold-policy seam

#: The two split policies the multi-arm diagnostic runs, and how many draws each gets.
#:
#: Two policies rather than the three :mod:`tests.studies.bounded_cv_laws` reports.  The
#: third reads the outcome, and a C-TMLE selection split that reads the outcome is scored
#: on the candidate loss it then chooses by, which is a second defect on top of the one
#: this family measures.  The reference arm and the role come from the framework, which
#: owns the reporting rule; only the policies, the size and the budget belong here.
#:
#: 8,000 paired draws, because the quantity has to be resolved rather than merely reported.
#: ``docs/roadmap.md`` records the pilot sizing: the two policies disagreed about coverage
#: on 0.0117 of the selector study's draws and 0.0250 of the DR-TMLE study's, and the
#: gate-relevant difference is 0.005.  At 400 draws the projected 99% interval was three to
#: four times wider than that; the 8,000-draw run produced half-widths of 0.0038 and 0.0051.
#: The selector instrument met the projected resolution target and the DR-TMLE instrument
#: narrowly missed it, while both point estimates supported the registered boundary reading.
#: The gated
#: ``root_n_and_efficiency/n_500`` cell keeps its own law, learners, size and 400-draw
#: budget.  This family declares no margin and states no verdict, so its budget buys a
#: narrower interval on a paired difference rather than a step toward a fixed endpoint.
FOLD_POLICIES = (FOLD_POLICY_REFERENCE_CELL, "treatment_stratified")
FOLD_POLICY_N = 500
FOLD_POLICY_REPLICATES = 8_000


class FoldPolicyMixin:
    """Study-only seam that draws every split of a fit under a named policy.

    **Why the seam replaces a method rather than configures the estimator.**
    :func:`~cleverly.learners.crossfit.fold_strata_refusal` rejects any
    ``stratify_folds`` other than ``"none"`` at construction whenever the fit draws a
    split, which is whenever ``cross_fit`` is set and at every selector-based
    collaborative setting.  :meth:`~cleverly.estimators.TMLE._fold_strata` records the
    consequence: its remaining branches are reachable only by replacing the method.  A
    diagnostic restricted to the policy the package still permits could report nothing
    about the one it refuses, which is the one thing this family exists to report.  The
    estimator is still constructed with ``stratify_folds="none"``, so nothing is refused
    and no arm is fitted through a path the package rejects.

    **Why :meth:`~cleverly.estimators.TMLE._fold_strata` rather than**
    ``_folds``.  :class:`~tests.studies.bounded_cv_laws.FoldPolicyTMLE` overrides
    ``_folds``, which reaches one split.  A selector-based C-TMLE fit draws three, and all
    three read this method: the outer nuisance folds, the selection folds the candidate
    path is scored over, and the nested folds each selection fold's training predictions
    are made out of.  A seam at ``_folds`` would vary the outer layer alone and leave the
    other two drawing the reference arm's split, so the published difference would name a
    change the fit made on a strict sub-part of its splits.
    ``tests/unit/test_fold_policy_rules.py::TestTheFoldPolicySeamReachesEverySplitLayer``
    records every partition a fit draws and is what holds all three to the policy.
    Replacing the strata instead
    leaves each layer to keep drawing the fold count its own configuration declares.  No
    fold count is pinned here for the same reason: these cells run five outer folds, three
    selection folds and two nested ones, and the only thing that may differ between the
    two arms is the policy.

    **What the recorded plan says, and does not.**
    :meth:`~cleverly.estimators.TMLE.crossfit_plan` reads ``self.stratify_folds``
    directly rather than this method, so
    :attr:`~cleverly.learners.crossfit.CrossFitPlan.stratify_by` stays empty on the
    ``treatment_stratified`` arm and misdescribes the split that arm actually realized.
    Nothing on the property path reads it: the published rows carry the estimate, its
    interval and the coverage flag, and the paired difference is computed from those.  The
    mismatch is recorded here because a reader who opened a restored result would see a
    plan that names no strata beside a fit that used them.
    """

    def __init__(self, policy: str, **kwargs: Any) -> None:
        if policy not in FOLD_POLICIES:
            raise ValueError(f"policy must be one of {FOLD_POLICIES}; got {policy!r}")
        self._policy = policy
        super().__init__(**kwargs)

    def _fold_strata(self, data: CausalData) -> np.ndarray | None:
        return fold_policy_strata(data, self._policy)


class FoldPolicyCTMLE(FoldPolicyMixin, CTMLE):
    """Selector-based multi-arm C-TMLE whose three splits are drawn under one policy."""


class FoldPolicyDRTMLE(FoldPolicyMixin, DRTMLE):
    """Multi-arm DR-TMLE whose one outer split is drawn under one policy."""


def fold_policy_cells(record: StudyRecord) -> tuple[PropertyCell, ...]:
    """The two fold-policy arms, on the ``n = 500`` law and one set of draws.

    The law, the learners and the size are the gated ``root_n_and_efficiency/n_500``
    cell's, because the reading this family owes is about that cell's regime.  Only the
    seed and the budget differ, and both differ on purpose.

    Parameters
    ----------
    record : StudyRecord
        The study registering the family, which supplies the seed stream.

    Returns
    -------
    tuple of PropertyCell
        One cell per policy in :data:`FOLD_POLICIES`, both with role ``"diagnostic"`` and
        both on :func:`fold_policy_seed`.  The shared seed is what makes the reported
        coverage difference a *paired* one: the two cells see identical samples and differ
        only in how their splits were drawn.
    """
    return make_fold_policy_cells(
        record,
        policies=FOLD_POLICIES,
        dgp=Sampler(),
        outcome_learner=correct_outcome(),
        treatment_learner=correct_treatment(),
        n=FOLD_POLICY_N,
        replicates=FOLD_POLICY_REPLICATES,
        estimand=ESTIMAND,
    )
