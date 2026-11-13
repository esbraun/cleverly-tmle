"""Exact-law witnesses of ``option-treatment-adjustment`` (X76 part (d), Fix-8 plan section 4).

The study sizes its control from :func:`tests.studies._option_adjustment_laws.targeted_limit`,
the population targeting step on a weighted finite set of points.  These tests run that
function on an exact finite law, whose empirical law is the law, and check it against the
package fit with the study's own learners at ``1e-10``:

* the constructed-error outcome learner on a law whose instrument ``Z`` is left out of the
  mechanism reproduces the remainder ``P_0[(1 - g_0/gbar)(Q* - Q_0)]`` at the targeted ``Q*``
  under the default logistic fluctuation, and ``P_0[(1 - g_0/gbar)(Q_n - Q_0)]`` at the initial
  ``Q_n`` under ``fluctuation="linear"``, where the targeting update is a function of ``W_g``
  that the factor annihilates (``docs/technical-reference/point-treatment-tmle.md``);
* the term is nonzero (above ``1e-3``), and it vanishes when the mechanism is fitted on the
  whole set, which is the mutation control;
* a confounder left out of ``adjustment=`` biases the fit while an instrument left out does
  not, and a confounder left out of the mechanism alone does no harm when the outcome
  regression is correct.

The exact law draws ``W1``, ``W2`` and ``Z`` as independent Bernoulli(1/2) variables,
``g_0 = 0.25 + 0.25 W1 + 0.25 Z`` and ``Q_0(a, x) = expit(-log 3 + log 3 (a + x))`` in
quarters, with ``x = W1`` (``Z`` an instrument) or ``x = Z`` (``Z`` a confounder).  Every cell
probability is a multiple of 1/64 and each ``(W1, W2, Z)`` cell holds ``8 m`` rows, so ``P_n``
equals ``P_0``.  The logistic GLM on ``(W1, W2)`` is saturated for ``gbar = 0.375 + 0.25 W1``
and the outcome GLM is correct, so both return the law's values to Newton's tolerance.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from cleverly.data import CausalData
from cleverly.estimators import TMLE
from tests.studies import _option_adjustment_laws as laws

#: Rows per ``(W1, W2, Z)`` cell are ``8 M``.
M = 8
#: The constructed error's coefficient on this law, a fixed number rather than ``k n^(-1/4)``.
DELTA = 0.7
#: ``E[Z | W1, W2]`` on this law.
CENTRE = 0.5
#: Nonzero floor of a witness, and the tolerance of an exact identity.
FLOOR = 1e-3
EXACT = 1e-10

Role = Literal["instrument", "confounder"]


def g_true(w1: Any, z: Any) -> np.ndarray:
    return np.asarray(0.25 + 0.25 * np.asarray(w1) + 0.25 * np.asarray(z), dtype=float)


def q_true(a: Any, x: Any) -> np.ndarray:
    return np.asarray(0.25 + 0.25 * np.asarray(a) + 0.25 * np.asarray(x), dtype=float)


def exact_frame(role: Role = "instrument") -> pd.DataFrame:
    """The exact law: ``Q_0`` reads ``W1`` when ``Z`` is an instrument and ``Z`` otherwise."""
    rows: list[tuple[float, ...]] = []
    for w1 in (0.0, 1.0):
        for w2 in (0.0, 1.0):
            for z in (0.0, 1.0):
                total = 8 * M
                treated = round(total * float(g_true(w1, z)))
                x = w1 if role == "instrument" else z
                for a, count in ((1.0, treated), (0.0, total - treated)):
                    events = round(count * float(q_true(a, x)))
                    rows += [(1.0, a, w1, w2, z)] * events + [(0.0, a, w1, w2, z)] * (
                        count - events
                    )
    return pd.DataFrame(rows, columns=["Y", "A", "W1", "W2", "Z"])


def points() -> laws.Points:
    cells = np.array([(w1, w2, z) for w1 in (0, 1) for w2 in (0, 1) for z in (0, 1)], float)
    return laws.Points(cells[:, 0], cells[:, 1], cells[:, 2], np.full(8, 1.0 / 8.0))


def truth(role: Role = "instrument") -> dict[str, float]:
    at = points()
    x = at.w1 if role == "instrument" else at.z
    one, zero = at.mean(q_true(1.0, x)), at.mean(q_true(0.0, x))
    return {"ey1": one, "ey0": zero, "ate": one - zero}


def fit(
    frame: pd.DataFrame,
    *,
    delta: float = DELTA,
    treatment_covariates: list[str] | None = None,
    covariates: tuple[str, ...] = laws.COVARIATES,
    fluctuation: Literal["logistic", "linear"] = "logistic",
) -> Any:
    """A fit with the study's learners: the constructed-error outcome GLM, the mechanism GLM."""
    estimator = TMLE(
        outcome_learner=laws.PerturbedLogistic(
            offset=delta,
            offset_column=1 + covariates.index("Z") if "Z" in covariates else 1,
            centre=CENTRE,
        ),
        treatment_learner=laws.PerturbedLogistic(),
        cross_fit=False,
        estimands=("ey1", "ey0", "ate"),
        simultaneous=False,
        g_bounds=laws.G_BOUNDS,
        max_iter=laws.MAX_ITER,
        tol=laws.TOL,
        random_state=0,
        fluctuation=fluctuation,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(covariates),
            treatment_covariates=treatment_covariates,
        ).single()


def error(result: Any, role: Role = "instrument", name: str = "ate") -> float:
    return float(result[name].psi) - truth(role)[name]


def limit(
    *, role: Role = "instrument", subset: bool = True, fluctuation: str = "logistic"
) -> laws.TargetedLimit:
    """:func:`~tests.studies._option_adjustment_laws.targeted_limit` on the exact law."""
    at = points()
    g = g_true(at.w1, at.z)
    g_fit = 0.375 + 0.25 * at.w1 if subset else g
    x = at.w1 if role == "instrument" else at.z
    return laws.targeted_limit(
        at,
        g_true=g,
        g_fit=g_fit,
        q0=lambda arm: q_true(arm, x),
        initial=lambda arm: np.asarray(expit(logit(q_true(arm, x)) + DELTA * (at.z - CENTRE))),
        fluctuation=fluctuation,
    )


def rows_formula(frame: pd.DataFrame, result: Any, *, targeted: bool) -> float:
    """``P_0[(1 - pi_a/gbar_a)(Q_a - Q_0a)]`` over the rows, arm 1 minus arm 0.

    ``targeted`` reads the fit's targeted regressions, else its initial ones.
    """
    w1, z = frame["W1"].to_numpy(), frame["Z"].to_numpy()
    g, gbar = g_true(w1, z), 0.375 + 0.25 * w1
    source = (result.fluctuations["mean"].targeted if targeted else result.nuisance.outcome).arms
    total = 0.0
    for arm, sign in ((1.0, 1.0), (0.0, -1.0)):
        pi, fitted = (g, gbar) if arm == 1.0 else (1.0 - g, 1.0 - gbar)
        q = np.asarray(source[arm], dtype=float)
        total += sign * float(np.mean((1.0 - pi / fitted) * (q - q_true(arm, w1))))
    return total


def check_logistic_witness() -> None:
    frame = exact_frame()
    result = fit(frame, treatment_covariates=list(laws.SUBSET))
    expected = limit().ate_remainder
    assert abs(error(result) - expected) < EXACT
    assert abs(error(result) - rows_formula(frame, result, targeted=True)) < EXACT
    assert abs(error(result)) > FLOOR


def check_linear_witness() -> None:
    frame = exact_frame()
    result = fit(frame, treatment_covariates=list(laws.SUBSET), fluctuation="linear")
    assert abs(error(result) - limit(fluctuation="linear").ate_remainder) < EXACT
    assert abs(error(result) - rows_formula(frame, result, targeted=False)) < EXACT
    assert abs(error(result)) > FLOOR


class TestTheExactLaw:
    def test_the_empirical_law_is_the_law(self) -> None:
        frame = exact_frame()
        assert len(frame) == 64 * M
        cells = frame.groupby(["W1", "Z"])["A"].mean()
        for (w1, z), value in cells.items():
            assert value == pytest.approx(float(g_true(w1, z)), abs=1e-15)
        assert frame.groupby(["A", "W1"])["Y"].mean().to_numpy() == pytest.approx(
            [0.25, 0.5, 0.5, 0.75], abs=1e-15
        )

    def test_the_study_glms_return_the_law_s_values(self) -> None:
        frame = exact_frame()
        outcome = laws.PerturbedLogistic().fit(frame[["A", "W1", "W2", "Z"]], frame["Y"])
        assert outcome.beta_ == pytest.approx([-np.log(3), np.log(3), np.log(3), 0, 0], abs=1e-9)
        mechanism = laws.PerturbedLogistic().fit(frame[list(laws.SUBSET)], frame["A"])
        fitted = mechanism.predict_proba(frame[list(laws.SUBSET)])[:, 1]
        assert fitted == pytest.approx(0.375 + 0.25 * frame["W1"].to_numpy(), abs=1e-12)

    def test_the_constructed_error_reaches_the_prediction(self) -> None:
        frame = exact_frame()
        design = frame[["A", "W1", "W2", "Z"]]
        plain = laws.PerturbedLogistic().fit(design, frame["Y"])
        perturbed = laws.PerturbedLogistic(offset=DELTA, centre=CENTRE).fit(design, frame["Y"])
        moved = perturbed.decision_function(design) - plain.decision_function(design)
        assert moved == pytest.approx(DELTA * (frame["Z"].to_numpy() - CENTRE), abs=1e-12)
        assert perturbed.get_params()["offset"] == DELTA


class TestTheRemainderWitness:
    def test_logistic_fluctuation_reproduces_the_targeted_remainder(self) -> None:
        check_logistic_witness()

    def test_linear_fluctuation_reproduces_the_initial_remainder(self) -> None:
        check_linear_witness()

    def test_each_arm_matches_the_limit(self) -> None:
        frame = exact_frame()
        result = fit(frame, treatment_covariates=list(laws.SUBSET))
        expected = limit()
        assert abs(error(result, name="ey1") - expected.remainder[1.0]) < EXACT
        assert abs(error(result, name="ey0") - expected.remainder[0.0]) < EXACT

    def test_the_initial_formula_holds_only_for_the_linear_fluctuation(self) -> None:
        """Under the logistic update the error differs from the initial-``Q`` formula."""
        frame = exact_frame()
        result = fit(frame, treatment_covariates=list(laws.SUBSET))
        assert abs(error(result) - rows_formula(frame, result, targeted=False)) > 1e-4

    def test_mutation_fitting_the_mechanism_on_the_full_set_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        check_logistic_witness()
        check_linear_witness()
        monkeypatch.setattr(
            CausalData, "mechanism_covariates", property(lambda self: self.covariates)
        )
        # The mechanism GLM now returns g_0, the term vanishes, and the nonzero floor fails.
        with pytest.raises(AssertionError):
            check_logistic_witness()
        with pytest.raises(AssertionError):
            check_linear_witness()

    def test_without_the_declaration_the_term_vanishes(self) -> None:
        frame = exact_frame()
        result = fit(frame, treatment_covariates=None)
        assert abs(error(result)) < EXACT
        assert abs(limit(subset=False).ate_remainder) < EXACT


class TestDroppedConfounderAgainstDroppedInstrument:
    def test_an_instrument_left_out_of_adjustment_does_no_harm(self) -> None:
        result = fit(exact_frame("instrument"), delta=0.0, covariates=("W1", "W2"))
        assert abs(error(result, "instrument")) < EXACT

    def test_a_confounder_left_out_of_adjustment_biases_the_fit(self) -> None:
        result = fit(exact_frame("confounder"), delta=0.0, covariates=("W1", "W2"))
        assert abs(error(result, "confounder")) > FLOOR

    def test_a_confounder_left_out_of_the_mechanism_alone_needs_a_correct_outcome_fit(
        self,
    ) -> None:
        frame = exact_frame("confounder")
        correct = fit(frame, delta=0.0, treatment_covariates=list(laws.SUBSET))
        assert abs(error(correct, "confounder")) < EXACT
        perturbed = fit(frame, treatment_covariates=list(laws.SUBSET))
        expected = limit(role="confounder").ate_remainder
        assert abs(error(perturbed, "confounder") - expected) < EXACT
        assert abs(expected) > FLOOR


class TestTheStudyFits:
    def test_the_declared_fits_hand_the_mechanism_the_subset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        widths: list[int] = []
        original = laws.PerturbedLogistic.fit

        def spy(self: Any, design: Any, target: Any, sample_weight: Any = None) -> Any:
            widths.append(np.asarray(design).shape[1])
            return original(self, design, target, sample_weight)

        monkeypatch.setattr(laws.PerturbedLogistic, "fit", spy)
        frame = laws.draw("LDB", 400, 0)
        laws.fit(frame, "ldb_slow")
        assert sorted(widths) == [2, 4]
        widths.clear()
        laws.fit(frame, "ldb_slow_full")
        assert sorted(widths) == [3, 4]

    def test_the_constructed_error_shrinks_like_n_to_the_minus_quarter(self) -> None:
        learner = laws.outcome_learner(laws.SETTINGS["ldb_slow"], 16_000)
        assert learner.offset == pytest.approx(laws.SLOW_OFFSET / 16_000**0.25, rel=1e-15)
        assert laws.outcome_learner(laws.SETTINGS["ldb_parametric"], 16_000).offset == 0.0

    def test_a_declared_fit_is_keyed_off_the_default_row(self) -> None:
        """The restriction reaches the support key: the declared fit leaves ``T01``.

        At registration this asserts the ``T54`` row instead.  Dropping the option from the key
        would put the declared fit on ``T01.binary``, which this test refuses.
        """
        frame = laws.draw("LDA", 400, 0)
        declared = laws.fit(frame, "lda_in_sample")["ate"].support
        assert declared.label == "experimental"
        assert declared.row is None
        assert "treatment_adjustment" in declared.reason
        free = (
            laws.estimator(laws.SETTINGS["lda_in_sample"], 400)
            .fit(frame, outcome="Y", treatment="A", covariates=list(laws.COVARIATES))
            .single()["ate"]
            .support
        )
        assert free.row == "T01.binary"
