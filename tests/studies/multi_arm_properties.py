"""Shared repeated-sampling cells for the multi-arm point-treatment family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import MultiArmDGP
from cleverly.utils.bounds import expit
from tests.studies import multi_arm_common
from tests.studies.evidence.properties import PropertyCell

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
            role="control",
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
