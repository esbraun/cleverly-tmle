r"""Does a weighted *longitudinal* fit estimate the parameter the package says it does?

:mod:`cleverly.data.weighting` makes one claim and makes it for the whole library: with
observation weights :math:`w`, the estimand is the causal parameter of the tilted law
:math:`dP_w = w\,dP/E[w]`, and its efficient influence function is

.. math::

    D^*_{\Psi_w}(o) = \frac{w(o)}{E[w]}\, D^*_{P_w}(o).

``tests/unit/test_weighted_estimand.py`` checks that at one time point.  This module checks
it for the sequential regression, where there is much more to get wrong: the tilt moves
*every* node's mechanism, every node's censoring factor, the distribution of the
time-varying confounder and every intermediate regression in the backward recursion, and
an implementation that reweighted only the final average -- or only the outcome model --
would still solve a score equation and still report a plausible number.

The check is made from the definition, on the finite-support law of
:mod:`tests.discrete_law_longitudinal`: :math:`\Psi(P_w)` is the same longhand
g-formula applied to the tilted cell probabilities, differentiated along a contamination of
:math:`P` by complex step.  Nothing in that derivation touches a cumulative product, a
clever covariate or a weighted score equation, so agreement is a check rather than a
restatement.

Two weight functions, and the second is the interesting one:

* ``w = 1 + 3W/5`` -- a function of the baseline covariate.  The survey case: the tilt
  moves the covariate distribution and leaves every conditional alone.
* ``w = 1 + A_1/2 + 2(1 - C_1)/5 + 3L_2/10 + 4Y/5`` -- a function of a treatment, a
  *censoring* indicator, the time-varying confounder and the outcome.  Now the tilt moves
  :math:`g_1`, :math:`c_1`, :math:`P(L_2 \mid \cdot)`, :math:`g_2`, :math:`c_2` and
  :math:`\bar Q` alike, and a fit whose nuisances stayed at :math:`P_0` estimates something
  else entirely.  This is the case that pins the statement down.

Because the weights are a function of the observed row only, the saturated learner fitted
by weighted loss *is* the tilted law's conditional -- exactly, on a sample that realises
the law -- so every score is already solved, ``epsilon`` comes back zero, and the reported
curve is the EIF itself rather than an estimate of it.  That is what makes the assertions
below exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE

from .. import discrete_law_longitudinal as law

#: Truncation wide enough never to bind: the law's conditionals all lie in [0.25, 0.75].
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)

#: Three of the law's six regimens: two constants and the rule that reads ``L2``.  The
#: remaining three exist to separate a rule from the constant it generalises, which
#: ``tests/unit/test_influence_gateaux_longitudinal.py`` already does unweighted and which
#: the weighting cannot break on its own -- and the fast tier pays per regimen per weight
#: function.
LABELS = ("never", "always", "treat_if_l2")
SPEC = {label: law.REGIMEN_SPEC[label] for label in LABELS}
REFERENCE = "never"

NAMES: tuple[str, ...] = tuple(f"ey_regimen[{label}]" for label in LABELS) + tuple(
    f"ate_regimen[{label} vs {REFERENCE}]" for label in LABELS if label != REFERENCE
)

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _node(value: Any) -> float:
    """A node a censored unit never reached, read as zero rather than ``None``.

    The weight of such a unit is a statement about the row that *was* observed, so the
    unreached nodes have to contribute nothing -- and they must do so identically on both
    sides of the comparison, which is why this is one function used by the weight
    definitions rather than a convention repeated in each.
    """
    return 0.0 if value is None else float(value)


#: ``(label, weight function of the whole support point)``.  See the module docstring.
WEIGHT_FUNCTIONS = {
    "baseline": lambda w, a1, c1, l2, a2, c2, y: 1.0 + 0.6 * w,
    "whole_history": lambda w, a1, c1, l2, a2, c2, y: (
        1.0 + 0.5 * a1 + 0.4 * (1 - c1) + 0.3 * _node(l2) + 0.8 * _node(y)
    ),
}


def _fit(weights: np.ndarray, **overrides: Any) -> Any:
    """An oracle-nuisance longitudinal fit under a given per-row weight vector."""
    settings: dict[str, Any] = {
        "reference": REFERENCE,
        "outcome_learner": law.CellMeans(),
        "pseudo_learner": law.CellMeans(),
        "treatment_learner": law.CellMeans(),
        "censoring_learner": law.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        # Nothing here reads the bands, and they would be a multiplier bootstrap over a
        # matrix nobody looks at.
        "simultaneous": False,
    }
    settings.update(overrides)
    return LTMLE(SPEC, **settings).fit(law.frame().assign(w=weights), weights="w", **COLUMNS)


@pytest.fixture(scope="module", params=sorted(WEIGHT_FUNCTIONS))
def weighted_fit(request: Any) -> tuple[Any, np.ndarray]:
    cells = law.cell_weights(WEIGHT_FUNCTIONS[request.param])
    return _fit(law.row_weights(cells)), cells


class TestTheDerivationItself:
    """Properties the numerical derivative must have before it can referee anything."""

    @pytest.mark.parametrize("label", sorted(WEIGHT_FUNCTIONS))
    def test_the_weighted_eif_has_mean_zero_under_the_sampling_law(self, label: str) -> None:
        # Mean zero under P, not under P_w: the rows are drawn from P, and an influence
        # function is centred at the law that generates them. This is the property that
        # separates the right weighted EIF from the plausible wrong ones.
        cells = law.cell_weights(WEIGHT_FUNCTIONS[label])
        for name in NAMES:
            centred = float((law.PROBS * law.weighted_eif(name, cells)).sum())
            assert centred == pytest.approx(0.0, abs=1e-11)

    def test_constant_weights_reproduce_the_unweighted_derivation(self) -> None:
        ones = np.ones(len(law.SUPPORT))
        for name in NAMES:
            np.testing.assert_allclose(law.weighted_eif(name, ones), law.eif(name), atol=1e-12)

    @pytest.mark.parametrize("label", sorted(WEIGHT_FUNCTIONS))
    def test_the_tilt_moves_the_estimand(self, label: str) -> None:
        # If Psi(P_w) equalled Psi(P) the comparisons below would hold for a fit that
        # ignored the weights, so the weighting has to be doing something first.
        cells = law.cell_weights(WEIGHT_FUNCTIONS[label])
        name = "ate_regimen[always vs never]"
        tilted = float(law.weighted_functional(law.PROBS, name, cells))
        assert abs(tilted - law.TRUTH[name]) > 1e-3

    def test_the_history_weight_moves_the_nuisances_and_not_only_the_marginal(self) -> None:
        """The premise that makes the second weight function more than a repeat of the first.

        A weight that is a function of ``W`` alone leaves every conditional exactly where
        it was and reweights only the covariate marginal.  A weight reading the treatment,
        the censoring indicator, the confounder and the outcome moves the mechanism at both
        nodes, the censoring factor at both nodes and the outcome regression -- so a fit
        that reweighted the plug-in and left the nuisances at ``P_0`` cannot pass the
        comparisons below by accident.
        """
        baseline = law.tilt(law.PROBS, law.cell_weights(WEIGHT_FUNCTIONS["baseline"]))
        history = law.tilt(law.PROBS, law.cell_weights(WEIGHT_FUNCTIONS["whole_history"]))

        # P(A1 = 1 | W = 0) under each tilt, from the cell masses alone.
        def g1_given(probs: Any) -> float:
            treated = float(law._mass(probs, w=0, a1=1))
            return treated / float(law._mass(probs, w=0))

        # P(C1 = 1 | W = 0, A1 = 1), the factor a censoring-dependent weight moves.
        def c1_given(probs: Any) -> float:
            stayed = float(law._mass(probs, w=0, a1=1, c1=1))
            return stayed / float(law._mass(probs, w=0, a1=1))

        assert g1_given(baseline) == pytest.approx(law.G1[0], abs=1e-12)
        assert c1_given(baseline) == pytest.approx(law.C1[0, 1], abs=1e-12)
        assert abs(g1_given(history) - law.G1[0]) > 1e-3
        assert abs(c1_given(history) - law.C1[0, 1]) > 1e-3


class TestTheWeightedFitIsTheWeightedParameter:
    def test_targeting_has_nothing_left_to_do(self, weighted_fit: tuple[Any, np.ndarray]) -> None:
        """Zero at every node, which is what makes the two comparisons below exact.

        It is also the assertion that catches a learner fitted *unweighted* while the
        estimating equation is weighted: the initial fit would then be the wrong
        conditional, the score would not already be solved, and ``epsilon`` would move off
        zero at the node where the disagreement is.
        """
        result, _ = weighted_fit
        for regimen_fit in result.fits.values():
            for step in regimen_fit.steps:
                assert step.fluctuation.converged
                assert abs(float(step.fluctuation.epsilon[0])) < 1e-9

    @pytest.mark.parametrize("name", NAMES)
    def test_the_point_estimate_is_the_tilted_g_formula(
        self, weighted_fit: tuple[Any, np.ndarray], name: str
    ) -> None:
        result, cells = weighted_fit
        assert result.psi(name) == pytest.approx(
            float(law.weighted_functional(law.PROBS, name, cells)), abs=1e-12
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_influence_curve_is_the_weighted_eif(
        self, weighted_fit: tuple[Any, np.ndarray], name: str
    ) -> None:
        result, cells = weighted_fit
        reported = result.influence_curves[name][law.first_row_of()]
        # ``rtol=0``, as in every sibling module: these curves reach order 20 before the
        # weights multiply them, so numpy's default relative tolerance would loosen this
        # to ~1e-6 while still reading as exact.
        np.testing.assert_allclose(reported, law.weighted_eif(name, cells), atol=1e-11, rtol=0)

    def test_the_curve_is_a_function_of_the_support_point_alone(
        self, weighted_fit: tuple[Any, np.ndarray]
    ) -> None:
        """Two rows with the same observed history carry the same curve -- and weight.

        The weight is a function of the row here, so this says the tilt was applied per
        support point and not, say, per fold or per position in the frame.
        """
        result, _ = weighted_fit
        curve = result.influence_curves["ey_regimen[always]"]
        for position, start in enumerate(law.first_row_of()):
            stop = start + law.COUNTS[position]
            np.testing.assert_allclose(curve[start:stop], curve[start], atol=1e-12, rtol=0)

    def test_dropping_the_normalisation_term_would_be_caught(
        self, weighted_fit: tuple[Any, np.ndarray]
    ) -> None:
        """The negative control: the check distinguishes the Hajek form from the naive one.

        The estimator is a ratio ``sum(w f) / sum(w)``, so its influence curve carries the
        centring ``w (f - psi)`` that linearises the random denominator.  Reporting
        ``w f - psi`` instead -- multiplying only the residual terms, or subtracting the
        estimate outside the weight -- is the classic error, and it has to fail here by far
        more than the tolerance above.
        """
        result, cells = weighted_fit
        name = "ey_regimen[always]"
        truth = law.weighted_eif(name, cells)
        rows = law.first_row_of()
        weights = np.asarray(result.data.weights)[rows]
        naive = truth + (weights - 1.0) * result.psi(name)
        assert np.max(np.abs(naive - truth)) > 1e-2


class TestConventions:
    def test_the_fit_is_invariant_to_the_scale_of_the_weights(
        self, weighted_fit: tuple[Any, np.ndarray]
    ) -> None:
        result, cells = weighted_fit
        rescaled = _fit(17.5 * law.row_weights(cells))
        for name in NAMES:
            assert rescaled.psi(name) == pytest.approx(result.psi(name), rel=0, abs=1e-14)
            assert rescaled[name].std_error == pytest.approx(
                result[name].std_error, rel=0, abs=1e-14
            )

    def test_a_constant_weight_column_is_the_unweighted_fit_bit_for_bit(self) -> None:
        """The regression surface: weighting is a generalisation, not a second estimator.

        Not ``approx``.  Every array the weighted path touches -- the nuisance fits, the
        Newton steps, the plug-in, the curve -- must come back the identical float, or some
        expression is being evaluated in a different association than it was and the claim
        that unweighted fits are untouched is only approximately true.
        """
        weighted = _fit(np.full(law.N, 3.0))
        unweighted = LTMLE(
            SPEC,
            reference=REFERENCE,
            outcome_learner=law.CellMeans(),
            pseudo_learner=law.CellMeans(),
            treatment_learner=law.CellMeans(),
            censoring_learner=law.CellMeans(),
            n_folds=1,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
        ).fit(law.frame(), **COLUMNS)
        assert not weighted.data.is_weighted
        for name in NAMES:
            assert weighted.psi(name) == unweighted.psi(name)
            np.testing.assert_array_equal(
                weighted.influence_curves[name], unweighted.influence_curves[name]
            )
        for label, regimen_fit in weighted.fits.items():
            for left, right in zip(regimen_fit.steps, unweighted.fits[label].steps, strict=True):
                np.testing.assert_array_equal(left.fluctuation.epsilon, right.fluctuation.epsilon)
                np.testing.assert_array_equal(left.targeted, right.targeted)

    def test_zero_weighting_a_stratum_and_deleting_it_agree(self) -> None:
        """Both halves of the documented convention, on the estimate and on the interval.

        The stratum is the units censored at the first node, which is the one a
        longitudinal fit can drop without leaving the law with nothing to adjust for.  Two
        things have to hold and the second looks alarming until the arithmetic is done: the
        two fits agree, and the weighted one still counts ``n`` rows -- because the
        normalisation has already scaled the surviving influence-curve values up by exactly
        the factor the larger ``n`` divides out.  A zero weight excludes a row from the
        *target population*; it does not pretend the row was never sampled.
        """
        frame = law.frame()
        keep = np.asarray(frame["C1"] == 1)
        learners: dict[str, Any] = {
            "reference": REFERENCE,
            "outcome_learner": law.CellMeans(),
            "pseudo_learner": law.CellMeans(),
            "treatment_learner": law.CellMeans(),
            "censoring_learner": law.CellMeans(),
            "n_folds": 1,
            "simultaneous": False,
        }
        # ``g_bounds`` left at "auto" deliberately: it is the one setting the row count
        # would resolve differently from the effective n, so this checks the truncation as
        # well as the estimate.
        weighted = LTMLE(SPEC, **learners).fit(
            frame.assign(w=keep.astype(float)), weights="w", **COLUMNS
        )
        dropped = LTMLE(SPEC, **learners).fit(frame.loc[keep].reset_index(drop=True), **COLUMNS)
        name = "ate_regimen[always vs never]"
        assert weighted.psi(name) == pytest.approx(dropped.psi(name), abs=1e-12)
        assert weighted[name].std_error == pytest.approx(dropped[name].std_error, rel=1e-3)
        assert weighted.config.g_bounds == pytest.approx(dropped.config.g_bounds, rel=1e-12)
        assert weighted.n == law.N
        assert dropped.n == int(keep.sum())

    def test_that_stratum_is_load_bearing(self) -> None:
        """The negative control for the test above: the tilt has to move the answer.

        Censoring at the first node depends on ``W`` and ``A1``, so conditioning the
        population on it is a different parameter -- if it were not, the agreement above
        would hold for a fit that ignored the weight column outright.
        """
        frame = law.frame()
        keep = np.asarray(frame["C1"] == 1)
        cells = np.array(
            [0.0 if point[law._NODES.index("c1")] == 0 else 1.0 for point in law.SUPPORT]
        )
        name = "ate_regimen[always vs never]"
        tilted = float(law.weighted_functional(law.PROBS, name, cells))
        assert keep.sum() < law.N
        assert abs(tilted - law.TRUTH[name]) > 1e-3
