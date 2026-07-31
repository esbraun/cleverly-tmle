"""The working model over regimens, against the oracle law.

This is the evidence the coefficients and their influence curves are right. On a law the
sample realises exactly, handed a saturated learner, the reported ``beta`` must be the
population projection to the last bit and the reported curve must equal the complex-step
Gateaux derivative of an independently written functional -- to ``1e-14`` **absolute**,
with ``rtol=0`` passed explicitly, because these curves reach order 20 and numpy's default
relative tolerance would loosen the check by six orders of magnitude while still reading
as exact.

The oracle's working model is deliberately **not saturated** (three coefficients against
twelve cells) and its weights deliberately **not uniform**, for the reasons
``tests/discrete_law.py`` gives at one time point: a saturated design agrees with the
per-regimen means whatever the projection code does, and a uniform ``h`` can leave the
design orthogonal enough that a coefficient collapses into something the per-regimen
report already gives. Both choices are asserted to be load-bearing rather than assumed to
be.

What this law **cannot** see: its outcome is binary, so ``OutcomeScaler`` is the identity
and every mutation to the raw-scale handling is silent here.
``tests/e2e/test_ltmle_msm.py`` pins that on a continuous outcome instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM

from .. import discrete_law_longitudinal as law

#: Wide enough that no factor is truncated: the law's conditionals all sit in
#: ``[0.25, 0.75]``, so a bound this loose cannot bind and the estimator runs on the
#: unmodified mechanism.
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}

#: Which oracle family uses which link.  ``"saturated"`` is a *design*, not a link.
LINK_OF = {"identity": "identity", "log": "log", "logit": "logit", "saturated": "identity"}


def declared(family: str) -> MSM:
    """The estimator side of an oracle family: a design over a dataframe.

    Stated in the estimator's own representation -- a callable reading ``W`` off the
    baseline frame -- rather than handed the oracle's array, so that the two sides are
    two statements of one model and a slip in either is a wrong number.
    """
    design, weights, terms = law._MSM_FAMILIES[family]
    labels = tuple(law.REGIMEN_ARMS)

    def build(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        w = np.asarray(frame["W"], dtype=int)
        return design[w, labels.index(label), :]

    def weight(label: Any, horizon: int, frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=int)
        del horizon
        return weights[w, labels.index(label)]

    return MSM(design=build, terms=terms, weights=weight, link=LINK_OF[family])


def oracle_fit(family: str) -> Any:
    return LTMLE(
        law.REGIMEN_SPEC,
        msm=declared(family),
        outcome_learner=law.CellMeans(),
        pseudo_learner=law.CellMeans(),
        treatment_learner=law.CellMeans(),
        censoring_learner=law.CellMeans(),
        n_folds=1,
        g_bounds=NO_TRUNCATION,
        simultaneous=False,
    ).fit(law.frame(), **COLUMNS)


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    return {family: oracle_fit(family) for family in law.MSM_NAMES}


FAMILIES = list(law.MSM_NAMES)

#: Absolute windows, passed with ``rtol=0`` as every sibling module does.
#:
#: Under the **identity** link and on the **saturated** design both sides evaluate a
#: closed form and agree to the last bit or two: the measured gaps are ``2e-14`` and
#: ``2e-15`` on curves reaching magnitude 18, so ``1e-12`` is already two orders of slack.
#:
#: Under a **link** the two sides run *independent* iterative solves -- the oracle a fixed
#: forty Newton steps, the estimator an alternation that exits on a relative shift in
#: ``beta`` of ``1e-10`` -- so ``beta`` itself is only settled to about ``1e-11``, and a
#: curve of magnitude 18 inherits that. The measured gaps are ``7e-12`` on the point
#: estimate and ``1.8e-10`` on the curve. The windows below leave two orders above the
#: measurement and, per ``TestTheComparisonHasTeeth``, seven below the smallest wrong
#: answer they have to reject. This is the same trade
#: ``tests/unit/test_influence_gateaux_msm.py`` records at one time point, one order
#: looser because there is an alternation here where there is a single solve there.
POINT_TOLERANCE = {"identity": 1e-12, "saturated": 1e-12, "log": 1e-10, "logit": 1e-10}
CURVE_TOLERANCE = {"identity": 1e-12, "saturated": 1e-12, "log": 1e-9, "logit": 1e-9}


class TestTheLawItself:
    """Checks on the oracle before any estimator is compared to it."""

    def test_a_saturated_working_model_is_the_per_regimen_report(self) -> None:
        """One indicator per regimen, uniform weights: beta *is* the vector of means.

        A property of the projection, not of any implementation -- and the reason the
        oracle's real design is not saturated.
        """
        for label in law.REGIMEN_ARMS:
            assert law.MSM_TRUTH[f"msm_regimen_saturated[{label}]"] == pytest.approx(
                law.TRUTH[f"ey_regimen[{label}]"], abs=1e-14
            )

    def test_the_weights_are_load_bearing(self) -> None:
        """With ``h == 1`` a coefficient could collapse into something already reported."""
        weighted = law.msm_coefficients(law.PROBS)
        uniform = law.msm_coefficients(law.PROBS, weights=np.ones_like(law.MSM_REGIMEN_WEIGHTS))
        assert np.max(np.abs(weighted - uniform)) > 1e-3

    def test_the_design_is_not_saturated(self) -> None:
        """Three coefficients against twelve cells, so beta really is a projection."""
        design = law.MSM_REGIMEN_DESIGN
        assert design.shape[2] < design.shape[0] * design.shape[1]
        beta = law.msm_coefficients(law.PROBS)
        residual = np.array(
            [
                law._conditional_mean(law.PROBS, label, w) - design[w, index] @ beta
                for w in range(2)
                for index, label in enumerate(law.REGIMEN_ARMS)
            ]
        )
        assert np.max(np.abs(residual)) > 1e-3

    @pytest.mark.parametrize("family", ["log", "logit"])
    def test_the_newton_solve_has_converged(
        self, family: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run without a convergence test, so the step count is asserted to be enough.

        Newton converges quadratically in the value *and* in the derivative, so past the
        point where the real part stops moving the imaginary part is exact too. Doubling
        the count must move neither -- and the *curve* is the half that could still be
        moving after the value has settled, so it is the one worth checking.
        """
        before = {name: law.eif(name) for name in law.MSM_NAMES[family]}
        beta = law.msm_coefficients(law.PROBS, link=family)
        monkeypatch.setattr(law, "MSM_NEWTON_STEPS", 2 * law.MSM_NEWTON_STEPS)
        np.testing.assert_allclose(
            law.msm_coefficients(law.PROBS, link=family), beta, atol=1e-14, rtol=0
        )
        for name in law.MSM_NAMES[family]:
            np.testing.assert_allclose(law.eif(name), before[name], atol=1e-12, rtol=0)


class TestTheFitAgreesWithTheOracle:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_reported_coefficient_has_an_oracle_and_no_more(
        self, fits: dict[str, Any], family: str
    ) -> None:
        """Both directions.

        ``test_registry.py``'s gate walks the *target registry*, and neither a regimen nor
        a working model over regimens is a ``Target``, so these parameters sit outside it
        and need their own. A coefficient with no longhand functional has no evidence
        behind it; a functional no fit reports is dead code.
        """
        fit = fits[family]
        reported = set(fit)
        # The fit reports every family under one head; the oracle names them apart.
        expected = {
            name.replace(f"msm_regimen_{family}[", "msm_regimen[") for name in law.MSM_NAMES[family]
        }
        assert reported == expected
        assert set(law.MSM_NAMES[family]) <= set(law.MSM_TRUTH)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_point_estimate_is_the_projection(self, fits: dict[str, Any], family: str) -> None:
        fit = fits[family]
        for name in law.MSM_NAMES[family]:
            reported = name.replace(f"msm_regimen_{family}[", "msm_regimen[")
            assert fit[reported].psi == pytest.approx(
                law.MSM_TRUTH[name], abs=POINT_TOLERANCE[family]
            )

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_influence_curve_is_the_gateaux_derivative(
        self, fits: dict[str, Any], family: str
    ) -> None:
        """``rtol=0`` on purpose: these curves reach magnitude 18."""
        fit = fits[family]
        rows = law.first_row_of()
        for name in law.MSM_NAMES[family]:
            reported = name.replace(f"msm_regimen_{family}[", "msm_regimen[")
            np.testing.assert_allclose(
                fit.influence_curves[reported][rows],
                law.eif(name),
                atol=CURVE_TOLERANCE[family],
                rtol=0,
            )

    @pytest.mark.parametrize("family", FAMILIES)
    def test_the_curve_is_a_function_of_the_support_point_alone(
        self, fits: dict[str, Any], family: str
    ) -> None:
        fit = fits[family]
        counts = law.COUNTS
        starts = law.first_row_of()
        for name in law.MSM_NAMES[family]:
            reported = name.replace(f"msm_regimen_{family}[", "msm_regimen[")
            curve = fit.influence_curves[reported]
            for start, count in zip(starts, counts, strict=True):
                block = curve[start : start + count]
                if block.size:
                    np.testing.assert_allclose(block, block[0], atol=1e-12, rtol=0)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_targeting_had_nothing_to_do(self, fits: dict[str, Any], family: str) -> None:
        """The saturated learner is exact in the sample, so every pooled score is already
        zero -- for *every* cell, since the fluctuation is shared across them."""
        fit = fits[family]
        for regimen_fit in fit.fits.values():
            for step in regimen_fit.steps:
                assert step.fluctuation.converged
                assert np.max(np.abs(step.fluctuation.epsilon)) < 1e-8
                np.testing.assert_allclose(step.targeted, step.initial, atol=1e-9, rtol=0)

    def test_a_saturated_working_model_reports_the_per_regimen_means(
        self, fits: dict[str, Any]
    ) -> None:
        """On this law the reduction is exact, because no Newton step is taken at all.

        Off it the pooled solve's convergence test and line search are taken over all
        ``C * n`` stacked rows, so the agreement is to ``1e-11`` rather than bit for bit;
        ``tests/e2e/test_ltmle_msm.py`` is where that is asserted.
        """
        fit = fits["saturated"]
        for label in law.REGIMEN_ARMS:
            assert fit[f"msm_regimen[{label}]"].psi == pytest.approx(
                law.TRUTH[f"ey_regimen[{label}]"], abs=1e-12
            )


class TestTheAlternation:
    @pytest.mark.parametrize("family", ["log", "logit"])
    def test_it_converged_and_the_covariate_was_built_at_the_reported_beta(
        self, fits: dict[str, Any], family: str
    ) -> None:
        """The fixed point is stated over the whole pass: at exit every node's ``Qbar*``
        is the fluctuation along the covariate at ``beta-hat``, and ``beta-hat`` is the
        projection of the first node's."""
        alternation = fits[family].msm_fits[0].alternation
        assert alternation.converged and alternation.failure is None
        assert alternation.trace[-1][2] <= 1e-10

    def test_the_identity_link_does_not_alternate(self, fits: dict[str, Any]) -> None:
        """``dm/deta`` is one, so the covariate never reads a beta and one pass suffices."""
        assert fits["identity"].msm_fits[0].alternation.trace == ()


class TestTheComparisonHasTeeth:
    """The windows above are set from a measurement, so they need a floor to sit under.

    Each of these is a fit that is *wrong* in one identifiable way, compared against the
    oracle the correct fit is compared against. The gap has to be orders larger than
    ``CURVE_TOLERANCE``, or a passing test would say nothing.
    """

    def test_a_uniformly_weighted_model_is_a_different_parameter(
        self, fits: dict[str, Any]
    ) -> None:
        """``h`` is not decoration: drop it and the projection lands somewhere else."""
        design, _, terms = law._MSM_FAMILIES["identity"]
        labels = tuple(law.REGIMEN_ARMS)
        uniform = MSM(
            design=lambda label, horizon, frame: design[
                np.asarray(frame["W"], dtype=int), labels.index(label), :
            ],
            terms=terms,
        )
        wrong = LTMLE(
            law.REGIMEN_SPEC,
            msm=uniform,
            outcome_learner=law.CellMeans(),
            pseudo_learner=law.CellMeans(),
            treatment_learner=law.CellMeans(),
            censoring_learner=law.CellMeans(),
            n_folds=1,
            g_bounds=NO_TRUNCATION,
            simultaneous=False,
        ).fit(law.frame(), **COLUMNS)
        rows = law.first_row_of()
        gaps = [
            np.max(np.abs(wrong.influence_curves[name][rows] - law.eif(name)))
            for name in law.MSM_NAMES["identity"]
        ]
        assert min(gaps) > 1e-2

    def test_a_curve_from_the_wrong_link_is_rejected(self, fits: dict[str, Any]) -> None:
        """The loosest window here is ``1e-9``; the smallest thing it must reject is not
        close to it."""
        rows = law.first_row_of()
        for wrong, right in (("log", "logit"), ("logit", "log")):
            gaps = [
                np.max(
                    np.abs(
                        fits[wrong].influence_curves[
                            name.replace(f"msm_regimen_{right}[", "msm_regimen[")
                        ][rows]
                        - law.eif(name)
                    )
                )
                for name in law.MSM_NAMES[right]
            ]
            assert min(gaps) > 1e-2
