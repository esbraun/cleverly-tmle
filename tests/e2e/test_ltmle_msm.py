"""A working model over regimens, through ``LTMLE``.

Two things here need saying about what is and is not asserted.

The **saturated** check is not bit for bit, and cannot be. The pooled fluctuation's
Newton solve normalises its convergence test by the total weight and runs its line search
on the pooled quasi-log-likelihood, both taken over all ``C * n`` stacked rows, so a
saturated model's blocks -- which are exactly the plain fit's covariates, and which
``test_longitudinal_msm_submodel.py`` asserts *are* bit for bit -- can still stop on a
different iterate. What is asserted is agreement to ``1e-11``, which is the footing
``tests/e2e/test_msm.py`` already puts the point-treatment saturated check on.

The **scale** check is here rather than on the exact law because the exact law cannot see
it: its outcome is binary, so ``OutcomeScaler`` is the identity and ``range`` is one, and
every mutation to the raw-scale handling is silent there. A continuous outcome under an
affine relabelling is what makes it fail.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sklearn.linear_model

from cleverly.assessment import _longitudinal_msm_scores
from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM

#: How long each regimen treats for -- the summary the working model is a dose-response
#: in. A table rather than anything parsed out of the label, which is what ``MSM.linear``
#: is refused for.
DURATION: dict[str, float] = {"always": 2.0, "never": 0.0, "early": 1.0, "late": 1.0}
SPEC: dict[str, Any] = {"always": 1, "never": 0, "early": (1, 0), "late": (0, 1)}

FAST: dict[str, Any] = {
    "outcome_learner": sklearn.linear_model.LinearRegression(),
    "pseudo_learner": sklearn.linear_model.LinearRegression(),
    "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
    "n_folds": 1,
    "learner_folds": 3,
    "random_state": 0,
    "simultaneous": False,
}

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}

TERMS = ("(intercept)", "duration")


def dose(label: Any, horizon: int, frame: Any) -> np.ndarray:
    del horizon
    n = len(frame)
    return np.column_stack([np.ones(n), np.full(n, DURATION[label])])


def saturated(labels: tuple[str, ...]) -> MSM:
    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        return np.eye(len(labels))[labels.index(label)] * np.ones((len(frame), 1))

    return MSM(design=design, terms=labels)


def projection(means: dict[str, float], labels: tuple[str, ...]) -> np.ndarray:
    """The population beta, written out independently: least squares of the truths."""
    design = np.column_stack([np.ones(len(labels)), [DURATION[label] for label in labels]])
    target = np.array([means[label] for label in labels])
    return np.linalg.lstsq(design, target, rcond=None)[0]


@pytest.fixture(scope="module")
def fitted() -> Any:
    frame, truth = make_longitudinal(n=3000, seed=0)
    result = LTMLE(SPEC, msm=MSM(design=dose, terms=TERMS), **FAST).fit(frame, **COLUMNS)
    return result, truth


class TestItReportsTheProjection:
    def test_the_parameters_are_the_declared_terms(self, fitted: Any) -> None:
        result, _ = fitted
        assert list(result) == ["msm_regimen[(intercept)]", "msm_regimen[duration]"]

    def test_each_coefficient_recovers_the_population_projection(self, fitted: Any) -> None:
        """Not the truth of any regimen: beta is the least-squares summary of all of them,
        and is what it is whether or not the working model is right."""
        result, truth = fitted
        labels = tuple(SPEC)
        means = {label: truth[f"ey_regimen[{label}]"] for label in labels}
        target = projection(means, labels)
        for column, term in enumerate(TERMS):
            estimate = result[f"msm_regimen[{term}]"]
            assert abs(estimate.psi - target[column]) < 2.5 * estimate.std_error

    def test_the_working_model_is_wrong_and_the_interval_is_still_honest(self, fitted: Any) -> None:
        """A projection is well defined under misspecification, which is the whole claim.

        If the four means happened to lie on a line the recovery above would say nothing
        about the projection machinery, so the misspecification is asserted rather than
        hoped for.
        """
        _, truth = fitted
        labels = tuple(SPEC)
        means = np.array([truth[f"ey_regimen[{label}]"] for label in labels])
        design = np.column_stack([np.ones(len(labels)), [DURATION[label] for label in labels]])
        fitted_beta = projection(dict(zip(labels, means, strict=True)), labels)
        residual = means - design @ fitted_beta
        assert np.max(np.abs(residual)) > 0.02

    def test_targeting_moved_the_fit(self, fitted: Any) -> None:
        result, _ = fitted
        epsilons = [step.fluctuation.epsilon for fit in result.fits.values() for step in fit.steps]
        assert max(float(np.max(np.abs(eps))) for eps in epsilons) > 1e-8

    @staticmethod
    def _stitched_score(result: Any, time: int) -> Any:
        """``sum_i sum_c w_i h(c,V_i) phi(c,V_i) h^c_t(i) (Z - Qbar*)`` at one node.

        One equation per term, shared across the cells, and summed over the units that
        *followed*. Written out here rather than read off the fluctuation, because what it
        is used for below is checking the fluctuation's own ``score`` against an
        independent computation of the same quantity.
        """
        model = result.msm
        covariate = model.weighted_design_at(result.msm_fits[0].beta)
        weights = result.data.weights
        score = np.zeros(model.n_terms)
        for index, cell in enumerate(model.cells):
            step = next(s for s in result.fits[cell.label].steps if s.time == time)
            residual = step.pseudo_outcome - step.targeted
            score += np.einsum("i,ip,i->p", weights * step.clever, covariate[:, index, :], residual)
        return score

    def test_the_pooled_score_is_solved_at_every_node(self) -> None:
        """The estimating equation the pooled fluctuation exists to zero.

        Checked on ordinary data rather than on the exact law, and deliberately: there the
        initial fit is exact, so the score is zero before targeting and this would pass
        however the fluctuation were written. It is what catches an ``at_risk`` mask where
        ``trained_on`` belongs, which the exact law cannot see.

        At ``n_folds=1``, because that is the evidenced longitudinal MSM construction and
        the equation posed to its solver.
        """
        frame, _ = make_longitudinal(n=3000, seed=0)
        result = LTMLE(SPEC, msm=MSM(design=dose, terms=TERMS), **{**FAST, "n_folds": 1}).fit(
            frame, **COLUMNS
        )
        model = result.msm
        covariate = model.weighted_design_at(result.msm_fits[0].beta)
        scale = float(np.sum(result.data.weights * np.abs(covariate).max(axis=(1, 2))))
        for time in (1, 2):
            np.testing.assert_allclose(
                self._stitched_score(result, time) / scale, np.zeros(model.n_terms), atol=1e-9
            )

    def test_the_reported_score_is_the_score_of_the_reported_fit(self, fitted: Any) -> None:
        """What the aggregated node fluctuation carries is the *stitched* score.

        Independent of the fluctuation: the left side is built from the step arrays and the
        model's own weighted design, the right side is what the aggregation computed. They
        agree only if the aggregation stacked the cells, the loss weights and the mask the
        way the pooled solve did.

        The identity link is what makes them comparable at all -- ``weighted_design_at``
        reads the *reported* beta, and above one fold each fold fluctuated at its own, so
        under a link the two sides are covariates built at different coefficients.
        """
        result, _ = fitted
        assert result.msm.link == "identity"
        # `Fluctuation.score` is a *mean* over the stacked rows, and every cell is live at
        # both nodes of an end-of-study fit, so the divisor is `C * n`.
        stacked = result.msm.n_cells * result.n
        for time in (1, 2):
            step = next(s for s in result.fits["always"].steps if s.time == time)
            np.testing.assert_allclose(
                self._stitched_score(result, time) / stacked,
                np.asarray(step.fluctuation.score),
                rtol=1e-10,
                atol=1e-16,
            )

    def test_the_report_carries_no_contrast(self, fitted: Any) -> None:
        """A working model reports coefficients; a difference of two is ``contrast()``."""
        result, _ = fitted
        assert not any(name.startswith("ate_regimen") for name in result)
        difference = result.contrast(
            lambda p: p[0] - p[1], ["msm_regimen[duration]", "msm_regimen[(intercept)]"]
        )
        # The docstring's claim is that contrast() is where a difference of two
        # coefficients comes from, so check it *is* that difference. `isfinite` alone was
        # true of any number contrast() might have returned, including the wrong one.
        assert difference.psi == pytest.approx(
            result["msm_regimen[duration]"].psi - result["msm_regimen[(intercept)]"].psi,
            rel=1e-12,
        )


#: The evidenced fold count for a longitudinal working model.
FOLD_COUNTS = [1]


class TestASaturatedModelIsThePerRegimenReport:
    @pytest.mark.parametrize("n_folds", FOLD_COUNTS)
    def test_every_coefficient_is_its_regimen_s_mean(self, n_folds: int) -> None:
        frame, _ = make_longitudinal(n=1500, seed=1)
        labels = ("always", "never", "early")
        spec = {label: SPEC[label] for label in labels}
        settings = {**FAST, "n_folds": n_folds}
        plain = LTMLE(spec, reference="never", **settings).fit(frame, **COLUMNS)
        model = LTMLE(spec, msm=saturated(labels), **settings).fit(frame, **COLUMNS)
        for label in labels:
            expected = plain[f"ey_regimen[{label}]"]
            got = model[f"msm_regimen[{label}]"]
            assert got.psi == pytest.approx(expected.psi, rel=1e-11)
            np.testing.assert_allclose(
                model.influence_curves[f"msm_regimen[{label}]"],
                plain.influence_curves[f"ey_regimen[{label}]"],
                rtol=1e-10,
                atol=1e-12,
            )

    @pytest.mark.parametrize("n_folds", FOLD_COUNTS)
    def test_it_holds_through_a_link(self, n_folds: int) -> None:
        """Which is what says a link is a reparameterisation rather than a second
        estimator: ``expit(beta_r)`` is the regimen's mean.

        Above one fold each outer fold alternates to its *own* ``beta`` on its training
        rows, so this also says the reported coefficient is the projection of the stitched
        fit rather than any fold's -- if it were a fold's, or an average of them, the
        identity would miss by the spread across folds.
        """
        frame, _ = make_longitudinal(n=1500, seed=1)
        labels = ("always", "never")
        spec = {label: SPEC[label] for label in labels}
        settings = {**FAST, "n_folds": n_folds}
        plain = LTMLE(spec, reference="never", **settings).fit(frame, **COLUMNS)
        design = saturated(labels)
        linked = LTMLE(
            spec, msm=MSM(design=design.design, terms=labels, link="logit"), **settings
        ).fit(frame, **COLUMNS)
        for label in labels:
            beta = linked[f"msm_regimen[{label}]"].psi
            assert 1.0 / (1.0 + np.exp(-beta)) == pytest.approx(
                plain[f"ey_regimen[{label}]"].psi, rel=1e-9
            )


class TestALink:
    @pytest.mark.parametrize("link", ["log", "logit"])
    def test_the_alternation_converges(self, link: str) -> None:
        """Under a link a round is a whole backward pass, so it had better be few of them."""
        frame, _ = make_longitudinal(n=1500, seed=2)
        result = LTMLE(SPEC, msm=MSM(design=dose, terms=TERMS, link=link), **FAST).fit(
            frame, **COLUMNS
        )
        alternation = result.msm_fits[0].alternation
        assert alternation.converged
        assert alternation.n_outer <= 10
        # beta reaches the covariate only through the smooth dm/deta, so the shift falls
        # fast; a loop that merely stopped would show a flat trace.
        shifts = [row[2] for row in alternation.trace]
        assert shifts[-1] < 1e-9 < shifts[0]

    def test_the_exponentiated_view_names_what_each_row_is(self) -> None:
        frame, _ = make_longitudinal(n=1000, seed=3)
        result = LTMLE(SPEC, msm=MSM(design=dose, terms=TERMS, link="logit"), **FAST).fit(
            frame, **COLUMNS
        )
        frame_out = result.coefficients(scale="ratio")
        assert list(frame_out["scale"]) == ["baseline", "odds ratio"]
        np.testing.assert_allclose(
            np.asarray(frame_out["psi"]),
            np.exp([result[f"msm_regimen[{term}]"].psi for term in TERMS]),
        )

    def test_the_exponentiated_view_is_refused_on_an_identity_fit(self, fitted: Any) -> None:
        result, _ = fitted
        with pytest.raises(ValueError, match="not a quantity anybody reports"):
            result.coefficients(scale="ratio")


class TestTheProjectionIsSolvedOnTheOutcomeScale:
    def test_an_affine_relabelling_of_the_outcome_moves_beta_affinely(self) -> None:
        """The only check that sees ``scaler.range``.

        ``Y' = c + d Y`` is the same experiment reported in different units, so the
        projection of the counterfactual means onto the same design must satisfy
        ``beta'_0 = c + d beta_0`` on an intercept column and ``beta'_j = d beta_j`` on
        every other. Dropping ``scaler.range`` from the residual half, or unscaling twice,
        breaks this -- and neither is visible on a binary outcome, where the scaler is the
        identity.
        """
        frame, _ = make_longitudinal(n=1500, seed=4)
        shift, factor = 3.0, 5.0
        relabelled = frame.copy()
        relabelled["Y"] = shift + factor * frame["Y"]

        settings = {
            **FAST,
            "outcome_learner": sklearn.linear_model.LinearRegression(),
            "pseudo_learner": sklearn.linear_model.LinearRegression(),
        }
        base = LTMLE(SPEC, msm=MSM(design=dose, terms=TERMS), **settings).fit(
            frame, family="gaussian", **COLUMNS
        )
        moved = LTMLE(SPEC, msm=MSM(design=dose, terms=TERMS), **settings).fit(
            relabelled, family="gaussian", **COLUMNS
        )
        intercept = base["msm_regimen[(intercept)]"].psi
        slope = base["msm_regimen[duration]"].psi
        assert moved["msm_regimen[(intercept)]"].psi == pytest.approx(
            shift + factor * intercept, rel=1e-8
        )
        assert moved["msm_regimen[duration]"].psi == pytest.approx(factor * slope, rel=1e-8)
        # And the standard error scales the same way, since the curve does.
        assert moved["msm_regimen[duration]"].std_error == pytest.approx(
            factor * base["msm_regimen[duration]"].std_error, rel=1e-8
        )


class TestTheSurroundingMachineryStillWorks:
    def test_cross_fitted_diagnostic_pools_cells_and_uses_the_stitched_design(self) -> None:
        """A private artifact probe for the engine retained behind the public refusal."""
        n = 4
        cells = (SimpleNamespace(horizon=1), SimpleNamespace(horizon=1))
        model = SimpleNamespace(
            cells=cells,
            n_terms=2,
            terms=TERMS,
            weights=np.column_stack([np.ones(n), np.full(n, 2.0)]),
        )
        records = tuple(
            SimpleNamespace(score=np.array([1e-8, -1e-8]), score_scale=np.ones(2)) for _ in range(2)
        )
        fluctuation = SimpleNamespace(
            folds=records,
            converged=True,
            n_iter=2,
            failure=None,
            score_scale=np.ones(2),
        )
        residuals = (np.array([1.0, -0.5, 0.25, -0.25]), np.array([-0.2, 0.3, -0.1, 0.4]))
        fits = tuple(
            SimpleNamespace(
                steps=(
                    SimpleNamespace(
                        time=1,
                        clever=np.ones(n),
                        pseudo_outcome=residual,
                        targeted=np.zeros(n),
                        fluctuation=fluctuation,
                    ),
                )
            )
            for residual in residuals
        )
        design = np.array(
            [
                [[1.0, 0.0], [1.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.5]],
                [[1.0, 0.0], [1.0, 2.0]],
                [[1.0, 0.0], [1.0, 2.5]],
            ]
        )
        msm_fit = SimpleNamespace(
            model=model,
            fits=fits,
            cause=None,
            fluctuation_design=design,
        )
        result = SimpleNamespace(
            msm=object(),
            msm_fits=(msm_fit,),
            data=SimpleNamespace(
                n=n,
                weights=np.ones(n),
                cluster=None,
                backend=None,
            ),
            scaler=SimpleNamespace(range=1.0),
        )

        diagnostics = _longitudinal_msm_scores(result, tolerance=1e-3)
        assert len(diagnostics.rows) == 4
        stitching = [row for row in diagnostics.rows if row.kind == "stitching"]
        assert [row.component for row in stitching] == list(TERMS)
        assert all(row.regimen is None for row in stitching)
        expected = np.mean(
            design[:, 0, :] * residuals[0][:, None] + 2.0 * design[:, 1, :] * residuals[1][:, None],
            axis=0,
        )
        np.testing.assert_allclose([row.score for row in stitching], expected)

    def test_score_diagnostics_report_each_pooled_component_once_per_node(
        self, fitted: Any
    ) -> None:
        result, _ = fitted
        rows = result.diagnostics.score_equations().rows
        assert len(rows) == 2 * len(TERMS)
        assert {row.component for row in rows} == set(TERMS)
        assert {row.regimen for row in rows} == {None}
        assert {row.kind for row in rows} == {"solver"}

    def test_diagnostics_reports_one_epsilon_column_per_term(self, fitted: Any) -> None:
        """The pooled epsilon is shared across the regimens, so a single column would
        print the first coefficient on every row and read as though each had its own."""
        result, _ = fitted
        frame = result.diagnostics.stagewise().to_frame()
        assert "epsilon" not in frame.columns
        assert list(TERMS) == [
            name[len("epsilon[") : -1] for name in frame.columns if name.startswith("epsilon[")
        ]
        # Identical down the regimens at a given time, by construction.
        for time in (1, 2):
            rows = frame[frame["time"] == time]
            assert rows["epsilon[duration]"].nunique() == 1

    def test_the_covariance_is_over_the_coefficients(self, fitted: Any) -> None:
        result, _ = fitted
        covariance = result.covariance()
        assert covariance.shape == (2, 2)
        np.testing.assert_allclose(covariance, covariance.T)

    def test_the_curve_is_refused(self, fitted: Any) -> None:
        result, _ = fitted
        with pytest.raises(ValueError, match="no horizon to index a curve by"):
            result.curve()

    def test_the_settings_report_names_the_working_model(self, fitted: Any) -> None:
        result, _ = fitted
        lines = result.config.describe()
        assert any("working model: 2 term(s) (intercept), duration" in line for line in lines)
        assert not any(line.startswith("reference:") for line in lines)

    def test_the_provenance_digests_the_evaluated_design(self, fitted: Any) -> None:
        """A design is a closure with no stable fingerprint; the arrays are what was used."""
        result, _ = fitted
        assert result.config.msm_fingerprint is not None
        assert result.config.msm_link == "identity"


class TestItRefusesByName:
    def test_cross_fitted_working_model_inference_is_refused_pending_evidence(self) -> None:
        with pytest.raises(ValueError, match="unsaturated projection property"):
            LTMLE(SPEC, msm=MSM(design=dose, terms=TERMS), n_folds=2)

    def test_a_reference_regimen_and_a_working_model_cannot_be_combined(self) -> None:
        with pytest.raises(ValueError, match="reference= names the regimen"):
            LTMLE(SPEC, reference="never", msm=MSM(design=dose, terms=TERMS))

    def test_a_working_model_must_be_an_msm(self) -> None:
        with pytest.raises(TypeError, match=r"must be a cleverly\.msm\.MSM"):
            LTMLE(SPEC, msm={"design": dose})

    def test_a_fit_without_one_refuses_the_coefficients_view(self) -> None:
        frame, _ = make_longitudinal(n=400, seed=5)
        result = LTMLE({"always": 1, "never": 0}, reference="never", **FAST).fit(frame, **COLUMNS)
        with pytest.raises(ValueError, match="has no working model"):
            result.coefficients()
