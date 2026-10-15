"""Sequential IPW and g-computation with a declared three-level offer mechanism.

The offers tutorial (``docs/examples/sequential-offers.ipynb``) declares the logged offer
probabilities of a three-level treatment and estimates the censoring model.  This module is the
fast-tier coverage of that composition for the two baselines, which
``tests/unit/test_known_node_mechanisms.py`` gives to ``LTMLE`` and ``SDR``.

The law in :mod:`tests.unit._declared_offers_support` is exactly realised, so a baseline with the
true offer mechanism and a correctly specified censoring GLM equals the g-formula written out
from the tables.  Each equality below has a mutation control in the same module: a declaration
of the same shape with two level columns exchanged must move the estimate it should move, or the
equality could hold for a declaration the fit never reads.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.learners import GLM
from cleverly.longitudinal import DynamicRegimen, SequentialGComputation, SequentialIPW

from . import _declared_offers_support as law

#: A GLM that saturates the law: products of every subset of the binary columns.
SATURATED = GLM(interactions=8, tol=1e-13)
#: Bounds that never bind: the law's offer probabilities lie in [0.25, 0.5].
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)

REGIMENS: dict[str, Any] = {
    "none": ("none", "none"),
    "discount": ("discount", "discount"),
    "ship_then_none": ("shipping", "none"),
    "respond": DynamicRegimen(
        "respond",
        ("discount", lambda history: np.where(history["L1"] == 1, "shipping", "none")),
        rule_kind="known",
    ),
}
TRUTH = {
    "none": law.truth("none", "none"),
    "discount": law.truth("discount", "discount"),
    "ship_then_none": law.truth("shipping", "none"),
    "respond": law.truth("discount", lambda l1: "shipping" if l1 == 1 else "none"),
}
VARIANTS = ("gcomp", "hajek", "horvitz_thompson")


def _estimator(variant: str, **settings: Any) -> Any:
    settings = {"g_bounds": NO_TRUNCATION, "simultaneous": False, **settings}
    if variant == "gcomp":
        return SequentialGComputation(
            REGIMENS,
            outcome_learner=settings.pop("outcome_learner", SATURATED),
            treatment_learner=settings.pop("treatment_learner", SATURATED),
            censoring_learner=settings.pop("censoring_learner", SATURATED),
            **settings,
        )
    return SequentialIPW(
        REGIMENS,
        normalization=variant,
        treatment_learner=settings.pop("treatment_learner", SATURATED),
        censoring_learner=settings.pop("censoring_learner", SATURATED),
        **settings,
    )


def _fit(variant: str, frame: pd.DataFrame, *, declare: Any = None, **settings: Any) -> Any:
    estimator = _estimator(variant, **settings)
    extra = {} if declare is None else {"treatment_probabilities": declare}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, **law.COLUMNS, **extra)


def _resample(n: int = 1500, seed: int = 4) -> pd.DataFrame:
    frame = law.frame()
    rows = np.random.default_rng(seed).choice(len(frame), size=n, replace=True)
    return frame.iloc[rows].reset_index(drop=True)


@pytest.fixture(scope="module")
def exact_frame() -> pd.DataFrame:
    return law.frame()


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_declared_three_level_mechanism_with_estimated_censoring_is_the_g_formula(
    exact_frame: pd.DataFrame, variant: str
) -> None:
    """Both baselines equal the g-formula, with the offer factors declared and C1 fitted.

    The censoring GLM is the only mechanism a learner fits, so a declaration read by position
    rather than label, or a dropped cumulative factor, moves an estimate here.
    """
    result = _fit(variant, exact_frame, declare=law.declared())
    assert result.config.known_factors == ("A1", "A2")
    assert set(result.estimates) >= {f"ey_regimen[{name}]" for name in TRUTH}
    for name, value in TRUTH.items():
        assert result.psi(f"ey_regimen[{name}]") == pytest.approx(value, abs=1e-10)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_wrong_declared_column_moves_the_estimate_it_should_move(
    exact_frame: pd.DataFrame, variant: str
) -> None:
    """Mutation control: exchanging two level columns moves IPW and leaves g-computation.

    The wrong declaration has the right shape and the right row sums.  IPW divides by it, so each
    plan that sends a discount or a free-shipping code moves by more than 1e-3, against the exact
    tolerance of 1e-10.  The self-normalized Hajek estimate moves least, by 0.004.
    The g-computation estimate reads the declaration in its diagnostics only, so it keeps the
    truth.  The no-offer plan reads the unexchanged column and stays at the truth for both.
    """
    right = _fit(variant, exact_frame, declare=law.declared())
    wrong = _fit(variant, exact_frame, declare=law.declared(swap=True))
    moved = {
        name: abs(wrong.psi(f"ey_regimen[{name}]") - right.psi(f"ey_regimen[{name}]"))
        for name in TRUTH
    }
    assert moved["none"] == pytest.approx(0.0, abs=1e-10)
    if variant == "gcomp":
        assert max(moved.values()) < 1e-10
    else:
        assert min(moved[name] for name in ("discount", "ship_then_none", "respond")) > 1e-3


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_declaration_equals_a_learner_that_returns_it(variant: str) -> None:
    """The declaration and a treatment learner returning it give the same estimate.

    The frame is a bootstrap resample, so the censoring GLM and the estimate are not at the
    population values, and the equality is not the exactness of the previous tests.
    """
    frame = _resample()
    main_terms: dict[str, Any] = {"censoring_learner": GLM()}
    if variant == "gcomp":
        main_terms["outcome_learner"] = GLM()
    declared = _fit(variant, frame, declare=law.declared(), **main_terms)
    fitted = _fit(variant, frame, treatment_learner=law.DeclaredOffers(), **main_terms)
    assert declared.config.known_factors == ("A1", "A2")
    assert fitted.config.known_factors == ()
    for name in declared.estimates:
        np.testing.assert_allclose(declared[name].psi, fitted[name].psi, rtol=1e-12)
    if variant == "gcomp":
        for name in declared.estimates:
            np.testing.assert_allclose(
                declared[name].influence_curve, fitted[name].influence_curve, rtol=1e-9, atol=1e-12
            )
    else:
        # A declaration fits no parameter, so the stacked curve has no treatment projection and
        # stays valid with a fitted censoring GLM. A treatment learner that is not a GLM has no
        # stacked curve, so its fit takes the plug-in status and a different curve.
        assert declared.inference_status == "influence_curve"
        assert fitted.inference_status == "adaptive_baseline_plugin"


def test_the_declared_factor_of_g_computation_enters_only_the_support_report(
    exact_frame: pd.DataFrame,
) -> None:
    """The support report divides by the declared factor, and the estimate does not.

    The correct declaration reproduces the report of a fitted treatment model.  The exchanged
    declaration leaves the estimates alone and changes the weights of the plans that send the
    exchanged offers.  The test reads the report rows, because ``to_frame()`` of a baseline's
    support report raises ``IndexError``.
    """
    right = _fit("gcomp", exact_frame, declare=law.declared())
    fitted = _fit("gcomp", exact_frame, treatment_learner=law.DeclaredOffers())
    wrong = _fit("gcomp", exact_frame, declare=law.declared(swap=True))
    support = {
        name: [
            (row.regimen, row.time, row.n_followed, row.max_weight, row.effective_n)
            for row in result.diagnostics.support().rows
        ]
        for name, result in (("right", right), ("fitted", fitted), ("wrong", wrong))
    }
    assert len(support["right"]) == len(TRUTH) * 2
    for kept, reference in zip(support["right"], support["fitted"], strict=True):
        assert kept[:3] == reference[:3]
        np.testing.assert_allclose(kept[3:], reference[3:], rtol=1e-9)
    changed = {
        kept[0]
        for kept, other in zip(support["right"], support["wrong"], strict=True)
        if abs(kept[3] - other[3]) > 1e-6
    }
    assert changed
    assert changed <= {"discount", "ship_then_none", "respond"}
    for name in TRUTH:
        assert right.psi(f"ey_regimen[{name}]") == wrong.psi(f"ey_regimen[{name}]")
