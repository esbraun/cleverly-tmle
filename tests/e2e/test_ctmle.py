"""C-TMLE end to end: a real fit, and everything downstream of one.

The design commitment behind :class:`~cleverly.CTMLE` is that collaborative selection
returns the complete selected ``(g_k, Qbar*_k)`` pair and then uses the ordinary pooled
targeting, influence-curve, sensitivity and validation layers.  This file checks that
commitment by taking a C-TMLE result through the whole public surface.

The statistical payoff -- smaller variance when an instrument is present -- needs
replications and lives in ``test_coverage_slow.py``.

:class:`TestSelectionIsForcedWhenTheOutcomeModelCannotHelp` is the exception, and the one
class here that is about the *selection* rather than about the plumbing around it.  It
exists because every other C-TMLE claim in the suite happens to be satisfiable by a
selector that always returns the empty propensity model; it takes that escape route away.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pytest
import sklearn.linear_model
from scipy.special import expit

from cleverly import CapabilityError, SuperLearner, load
from cleverly._inference_status import NON_INFERENTIAL
from cleverly._typing import FloatArray
from cleverly.datasets import (
    instrument_dgp,
    make_instrument,
    make_missing_outcome,
    make_multi_arm,
)
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators.targeting import build_submodel
from cleverly.exceptions import (
    WORKING_MECHANISM_ASSESSMENT_NOTE,
    WORKING_MECHANISM_NOT_INFERENTIAL,
)
from cleverly.inference.influence import counterfactual_means
from cleverly.validation.nuisance import NUISANCE_SELECTION_MISSING
from tests.conftest import FAST_KWARGS, SELECTOR_CONFIGS, linear_ctmle, linear_in_sample
from tests.unit._natural_course_support import NeverFit, never_fit_learners

TMLE_SETTINGS = {**FAST_KWARGS, "estimands": ("ate", "ey1", "ey0")}

#: Three selection folds rather than the default five keeps this file in the fast tier;
#: nothing asserted here turns on the fold count.
SETTINGS = {**TMLE_SETTINGS, "selection_folds": 3}


def _binary_instrument_dgp():
    """``instrument_dgp`` mapped onto a probability, for a fit that keeps cross-fitting.

    ``instrument_dgp``'s outcome is Gaussian and unbounded, so a cross-fitted fit of it
    needs a declared ``q_bounds`` it cannot state truthfully. Most of this module fits in
    sample instead, but the diagnostics tests below read the *propensity* model's
    cross-validated calibration, which is a property of holding folds out and does not
    survive turning cross-fitting off. This process keeps the same confounder, instrument
    and predictor roles on the probability scale, so the propensity fold structure is
    unchanged and the fit no longer needs a bound it cannot state.
    """
    base = instrument_dgp()

    def outcome_mean(w: FloatArray, a: Any, z: Any) -> FloatArray:
        del z
        return expit(-0.2 + 0.6 * a + 0.9 * w[:, 0] + 0.5 * w[:, 2])

    return replace(base, family="binomial", outcome_mean=outcome_mean)


def _make_binary_instrument(
    n: int, *, seed: int | None = None, backend: str | None = None
) -> tuple[Any, dict[str, float]]:
    return _binary_instrument_dgp().sample(n, seed=seed, backend=backend)


@pytest.fixture(scope="module")
def frame_and_truth() -> tuple[object, dict[str, float]]:
    return _make_binary_instrument(n=900, seed=5)


@pytest.fixture(scope="module")
def fit(frame_and_truth) -> object:
    frame, _ = frame_and_truth
    return CTMLE(**SETTINGS).fit(frame, outcome="Y", treatment="A").single()


class TestTheFit:
    def test_it_recovers_the_truth(self, fit, frame_and_truth) -> None:
        _, truth = frame_and_truth
        estimate = fit["ate"]
        # The diagnostic supplies a *scale* for this sanity check and not a coverage
        # claim: the package reports no interval for a selector path, and "within three
        # plug-in spreads" is a statement about this one fit.
        assert abs(estimate.psi - truth["ate"]) < 3.0 * estimate.plugin_std_error

    def test_it_solves_the_score_equation(self, fit) -> None:
        check = fit.diagnostics.score_equations()
        assert check.passed, check.summary()

    def test_the_ate_influence_curve_is_the_difference_of_the_means(self, fit) -> None:
        # Exact identity, and it holds only if the selected propensity reached every
        # estimand through the same fluctuation.
        assert np.allclose(
            fit["ate"].influence_curve,
            fit["ey1"].influence_curve - fit["ey0"].influence_curve,
            atol=1e-12,
        )

    def test_the_selection_is_reported(self, fit) -> None:
        selection = fit.extra["ctmle"]
        assert selection.strategy == "greedy"
        assert selection.estimand == "ate"
        assert 0 <= selection.selected < len(selection.path)
        assert set(selection.selected_covariates) <= set(fit.data.covariate_names)

    def test_the_working_mechanism_plugin_error_is_below_a_plain_fits_standard_error(
        self, fit, frame_and_truth
    ) -> None:
        """The plug-in spread is the smaller number, and that is not a precision claim.

        The old name for this test asserted the reading ``docs/roadmap.md`` RM12
        forbids: "a user reads the smaller standard error as a precision gain, and the
        interval undercovers". The package now refuses to call the collaborative number
        a standard error at all. What remains true, and is what this test pins, is the
        mechanism: a narrower propensity model means a smaller ``1/g`` and a smaller
        influence curve. One sample, so it is a statement about this fit.
        """
        frame, _ = frame_and_truth
        plain = TMLE(**TMLE_SETTINGS).fit(frame, outcome="Y", treatment="A").single()
        assert fit["ate"].plugin_std_error < plain["ate"].std_error


class TestDownstreamMachineryStillWorks:
    def test_sensitivity_analyses_run_against_the_selected_model(self, fit) -> None:
        report = fit.diagnostics.support()
        assert report is not None
        curve = fit.diagnostics.truncation_curve()
        assert len(curve) > 0

    def test_the_nuisance_diagnostics_describe_the_selected_model(self, fit) -> None:
        diagnostics = fit.diagnostics.nuisance_models()
        assert "propensity" in {model.name for model in diagnostics.models}
        assert diagnostics.summary()
        # Computed from the selected mechanism itself, not from a shared g(W) fit.
        report = diagnostics["propensity"]
        assert report.metrics["mean_predicted"] == pytest.approx(
            float(np.mean(fit.nuisance.propensity.arm(1.0)))
        )

    def test_the_nuisance_report_retains_and_scopes_the_selection(self, fit) -> None:
        diagnostics = fit.diagnostics.nuisance_models()

        assert diagnostics.selection is fit.extra["ctmle"]
        assert diagnostics.treatment_role == "collaborative_working_model"
        assert diagnostics.reported_repeat == 1
        assert diagnostics.selection.path == fit.extra["ctmle"].path
        assert "C-TMLE greedy selected candidate" in diagnostics.summary()
        assert diagnostics.selection.describe() in diagnostics.summary()
        assert "complete adjustment set" in diagnostics.summary()

        combined = fit.diagnostics.run_all()
        retained = combined.report("nuisance_models")
        assert retained.selection.path == diagnostics.selection.path
        assert "C-TMLE greedy selected candidate" in combined["nuisance_models"].detail

    def test_the_role_suppresses_two_claims_and_no_others(self, fit) -> None:
        """Exactly two claims are about the treatment *law*, and only those two go quiet.

        Both suppressed claims read a metric of a model nobody fitted: the selected
        working mechanism is an intercept-only candidate here, so its AUC sits at chance
        and its calibration slope is far from one.  Reporting "overlap is excellent" or
        "poorly calibrated" from those describes assignment given the complete adjustment
        set, which this fit never estimated.

        The high-AUC positivity note is the control that keeps the suppression narrow.
        ``CTMLE._nuisances`` puts the selected mechanism on ``nuisance.propensity``, so it
        *is* the denominator of the clever covariate, and an AUC near one there is a
        statement about the estimator's own weights rather than about the treatment law.
        Suppressing it by role would hide a positivity problem the fit really has.
        """
        diagnostics = fit.diagnostics.nuisance_models()
        assert diagnostics["propensity"].metrics["auc"] < 0.55
        assert diagnostics["propensity"].metrics["calibration_slope"] < 0.7

        # Suppressed: both claims would otherwise fire on these very metrics.
        assert "confounding by these covariates is limited" not in diagnostics.verdict()
        assert not any("propensity is poorly calibrated" in item for item in diagnostics.findings)
        wrong_role = replace(diagnostics, treatment_role="estimated_treatment_law")
        assert "confounding by these covariates is limited" in wrong_role.verdict()
        assert any("propensity is poorly calibrated" in item for item in wrong_role.findings)

        # Not suppressed: the same metric read as a property of the weights, not the law.
        propensity = diagnostics["propensity"]
        high_auc = replace(propensity, metrics={**propensity.metrics, "auc": 0.95})
        high_auc_models = tuple(
            high_auc if model.name == "propensity" else model for model in diagnostics.models
        )
        collaborative_high_auc = replace(diagnostics, models=high_auc_models)
        ordinary_high_auc = replace(
            collaborative_high_auc, treatment_role="estimated_treatment_law"
        )
        for report in (collaborative_high_auc, ordinary_high_auc):
            assert any("positivity problem" in item for item in report.findings)
        assert "positivity problem" in collaborative_high_auc.verdict()

    def test_a_mean_only_learner_library_is_reported_for_a_working_model_too(self, fit) -> None:
        """A fact about a learner library, and not an interpretation of a treatment law.

        "no candidate beat predicting the average" says a super learner found nothing in
        its library worth weighting, whatever the fitted object is for. It was suppressed
        for a collaborative fit along with the two claims that do read the treatment law,
        and the suppression had no test at all, so a mean-only ``oat`` mechanism reported
        nothing about a library that had failed.

        The selected mechanism here comes off the candidate path and has no library, so
        the weights are substituted. That is the state an ``oat`` fit reaches honestly.
        """
        diagnostics = fit.diagnostics.nuisance_models()
        propensity = diagnostics["propensity"]
        assert propensity.learner_weights == {}
        mean_only = replace(propensity, learner_weights={"mean": 0.95, "glm": 0.05})
        report = replace(
            diagnostics,
            models=tuple(
                mean_only if model.name == "propensity" else model for model in diagnostics.models
            ),
        )

        assert report.treatment_role == "collaborative_working_model"
        notes = [note for note in report.findings if "weight on the marginal mean" in note]
        assert len(notes) == 1
        assert notes[0].startswith("propensity put 95%")
        assert notes[0] in report.verdict()
        # The two treatment-law claims are still quiet on the very same report.
        assert not any("poorly calibrated" in note for note in report.findings)
        assert "confounding by these covariates is limited" not in report.verdict()
        # And a weight under the threshold reports nothing, so the rule still has a gate.
        below = replace(propensity, learner_weights={"mean": 0.8, "glm": 0.2})
        quiet = replace(
            report,
            models=tuple(below if model.name == "propensity" else model for model in report.models),
        )
        assert not any("weight on the marginal mean" in note for note in quiet.findings)

    def test_the_role_reaches_a_multi_arm_report_named_for_its_arm(self) -> None:
        """The one input on which the two propensity predicates disagree.

        Role suppression tests ``propensity[<label>]`` by prefix, while the two AUC rules
        test ``"propensity"`` exactly.  Every other test here fits a binary treatment,
        where the single report is named ``"propensity"`` and both predicates agree, so
        nothing distinguishes them and the widening is a term that vanishes at the truth.

        A K-armed collaborative fit is where they part.  Its one selected mechanism is
        reported once per arm, and the calibration slope of an intercept-only selection is
        far from one on every arm.  Narrowing the suppression to the exact name would make
        this report state, three times, that a model nobody fitted is poorly calibrated.
        """
        # A binary outcome, not the Gaussian default: a cross-fitted fit of a continuous
        # outcome now needs a declared q_bounds (the fold and scale rules), and
        # make_multi_arm's Gaussian outcome has none to declare truthfully. The switch
        # leaves the propensity fold structure -- and so the calibration slopes this test
        # reads -- unchanged.
        frame, _ = make_multi_arm(n=600, seed=5, family="binomial")
        result = (
            CTMLE(
                outcome_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
                n_folds=3,
                learner_folds=2,
                random_state=5,
                n_jobs=1,
                strategy="greedy",
                selection_folds=3,
                selection_inner_folds=2,
                estimands=("ate",),
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

        report = result.diagnostics.nuisance_models()
        assert report.treatment_role == "collaborative_working_model"
        arms = [model for model in report.models if model.name.startswith("propensity[")]
        # The witness: every arm's slope is outside the gate the rule applies.
        assert len(arms) >= 2
        assert all(not 0.7 <= model.metrics["calibration_slope"] <= 1.4 for model in arms)

        assert not any("poorly calibrated" in note for note in report.findings)
        ordinary = replace(report, treatment_role="estimated_treatment_law")
        fired = [note for note in ordinary.findings if "poorly calibrated" in note]
        assert len(fired) == len(arms)
        assert all(any(model.name in note for note in fired) for model in arms)

    def test_the_working_model_verdict_replaces_the_ordinary_reassurance(self, fit) -> None:
        """With no finding to report, the two roles must not print the same sentence.

        "nuisance fits look reasonable" is a claim about fitted models, and a selected
        working mechanism is not one of those. The collaborative branch says so and sends
        the reader to the artifacts that do carry the evidence.
        """
        diagnostics = fit.diagnostics.nuisance_models()
        assert diagnostics.findings == ()
        assert diagnostics.verdict() == (
            "VERDICT: C-TMLE working-model metrics are descriptive, and this fit reports "
            "no confidence interval and no p-value; inspect the selection and support "
            "reports."
        )
        assert diagnostics.verdict() in diagnostics.summary()

        # Flipping the *role* alone does not retract the RM12 sentence, because that
        # sentence is keyed on the declared inference status. The outcome-adaptive path
        # carries this same role and does publish an interval, so a line keyed on the role
        # would say the wrong thing about it. This assertion is what pins the two apart.
        role_only = replace(diagnostics, treatment_role="estimated_treatment_law")
        assert "no confidence interval or p-value is available" in role_only.summary()

        ordinary = replace(
            role_only,
            inference="influence_curve",
        )
        assert ordinary.findings != ()  # the calibration claim comes back
        no_metrics = replace(
            ordinary,
            models=tuple(replace(model, metrics={}) for model in diagnostics.models),
        )
        assert no_metrics.verdict() == "VERDICT: nuisance fits look reasonable."

    def test_a_missing_selection_stays_a_machine_readable_omission(self, fit) -> None:
        """A lost artifact is an absent artifact, and never an absent method.

        ``treatment_role`` is derived from ``fitted_method``, not from ``extra["ctmle"]``.
        That separation is the point of this PR: tying the role to the artifact would let
        a detached collaborative result print the treatment-law claims about a working
        mechanism, which is the misinterpretation the role exists to prevent. Every other
        assertion in this test passes under that mistake, so the role is asserted here.
        """
        detached = replace(fit, extra={})
        diagnostics = detached.diagnostics.nuisance_models()

        assert diagnostics.selection is None
        assert diagnostics.selection_omission == NUISANCE_SELECTION_MISSING
        assert (
            f"C-TMLE selection unavailable: {NUISANCE_SELECTION_MISSING}" in diagnostics.summary()
        )

        assert detached.fitted_method == "collaborative_tmle"
        assert diagnostics.treatment_role == "collaborative_working_model"
        assert "confounding by these covariates is limited" not in diagnostics.verdict()
        assert not any("propensity is poorly calibrated" in item for item in diagnostics.findings)
        assert "The propensity metrics describe the selected C-TMLE working mechanism" in (
            diagnostics.summary()
        )

        detail = detached.diagnostics.run_all()["nuisance_models"].detail
        assert f"C-TMLE selection unavailable: {NUISANCE_SELECTION_MISSING}" in detail
        assert "selected candidate" not in detail

    def test_an_artifact_of_the_wrong_type_is_no_artifact(self, fit) -> None:
        """``ctmle_selection`` narrows the ``extra`` slot to the two classes that describe.

        ``extra`` is a plain mapping, so anything can occupy the ``ctmle`` key: an older
        artifact, or a value a caller wrote. The typed accessor reports it as absent
        rather than calling ``describe`` on it, which keeps a foreign value out of the
        report as an omission instead of as a traceback inside a diagnostic.
        """
        assert fit.ctmle_selection is fit.extra["ctmle"]

        foreign = replace(fit, extra={"ctmle": "a value from somewhere else"})
        assert foreign.ctmle_selection is None
        report = foreign.diagnostics.nuisance_models()
        assert report.selection is None
        assert report.selection_omission == NUISANCE_SELECTION_MISSING
        assert NUISANCE_SELECTION_MISSING in report.summary()

    def test_the_warmed_method_report_survives_persistence(self, fit, tmp_path) -> None:
        before = fit.diagnostics.run_all()
        restored = load(fit.save(tmp_path / "ctmle-method-report.joblib"))
        after = restored.diagnostics.run_all()

        assert after["nuisance_models"].detail == before["nuisance_models"].detail
        retained = after.report("nuisance_models")
        assert retained.treatment_role == "collaborative_working_model"
        assert retained.selection.path == fit.extra["ctmle"].path

    def test_a_selected_candidate_mutation_moves_the_combined_row(self, fit) -> None:
        selection = fit.extra["ctmle"]
        assert len(selection.path) > 1
        changed = replace(selection, selected=(selection.selected + 1) % len(selection.path))
        mutated = replace(fit, extra={"ctmle": changed})

        before = fit.diagnostics.run_all()["nuisance_models"].detail
        after = mutated.diagnostics.run_all()["nuisance_models"].detail
        assert after != before
        assert f"candidate {changed.selected + 1} of {len(changed.path)}" in after

    def test_and_report_no_learner_table_for_a_selected_mechanism(self, fit) -> None:
        """Empty on purpose, and the one thing an accepted regression would look like.

        A selector's ``g`` comes off the candidate path -- often the intercept-only
        candidate, which has no learner behind it at all -- so there is no super-learner
        weighting to report.  Before the shared pass went outcome-first this key held the
        ordinary ``g(W)`` table, describing a model the estimate never used.  ``oat`` does
        have one shared fit, and the assertion below keeps the two paths distinguishable.
        """
        report = fit.diagnostics.nuisance_models()["propensity"]
        assert report.learner_weights == {}
        assert report.learner_risks == {}

    def test_but_oat_reports_the_table_from_its_one_shared_fit(self, frame_and_truth) -> None:
        frame, _ = frame_and_truth
        oat = (
            CTMLE(
                **{
                    **TMLE_SETTINGS,
                    "treatment_learner": SuperLearner(
                        [sklearn.linear_model.LogisticRegression(max_iter=1000)],
                        n_folds=3,
                    ),
                },
                strategy="oat",
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        report = oat.diagnostics.nuisance_models()["propensity"]
        assert report.learner_weights and report.learner_risks
        diagnostics = oat.diagnostics.nuisance_models()
        assert diagnostics.selection is oat.extra["ctmle"]
        assert diagnostics.treatment_role == "collaborative_working_model"
        features = len(oat.extra["ctmle"].treatment_features)
        assert features > 0
        assert (
            f"C-TMLE outcome-adaptive fit used {features} Qbar feature(s)" in diagnostics.summary()
        )
        assert diagnostics.selection.describe() in diagnostics.summary()
        assert (
            f"C-TMLE outcome-adaptive fit used {features} Qbar feature(s)"
            in oat.diagnostics.run_all()["nuisance_models"].detail
        )

    def test_refutation_runs(self, fit) -> None:
        # A placebo refit goes back through CTMLE._nuisances, so the selection is
        # redone on the permuted data rather than reused -- which is the point.
        refutation = fit.diagnostics.refute(tests=("placebo",), n_replicates=2, random_state=0)
        assert "placebo" in {test.name for test in refutation.tests}

    def test_the_summary_prints(self, fit) -> None:
        assert "Targeted maximum likelihood estimation" in fit.summary()

    def test_to_frame_returns_the_callers_backend(self, fit) -> None:
        frame = fit.to_frame()
        assert len(frame) == len(fit.estimates)

    @pytest.mark.parametrize("targeting", ["iterative", "one_step"])
    def test_every_truncation_retarget_solves_the_score(self, targeting: str) -> None:
        from sklearn.dummy import DummyRegressor

        frame, _ = instrument_dgp().sample(400, seed=22)
        result = (
            CTMLE(
                **{
                    **SETTINGS,
                    "outcome_learner": DummyRegressor(strategy="mean"),
                    "n_folds": 3,
                    "selection_folds": 2,
                    "targeting": targeting,
                    # This is about the retarget solving the score after truncation, not
                    # about cross-fitting; the Gaussian outcome has no q_bounds to declare.
                    "cross_fit": False,
                }
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        for lower in (0.05, 0.2, 0.4):
            estimates, fluctuations = result.estimator.retarget(
                result.data,
                result.nuisance,
                estimands=("ate",),
                g_bounds=(lower, 1.0 - lower),
                g_bounds_conditional=(lower, 1.0 - lower),
            )
            assert fluctuations["mean"].score_norm < 1e-8
            assert abs(float(np.mean(estimates["ate"].influence_curve))) < 1e-8


class TestBackendParity:
    def test_pandas_and_polars_agree_bit_for_bit(self) -> None:
        pandas_frame, _ = make_instrument(n=500, seed=6, backend="pandas")
        polars_frame, _ = make_instrument(n=500, seed=6, backend="polars")
        columns = {"outcome": "Y", "treatment": "A"}
        # Backend parity, not cross-fitting: fit in sample so the Gaussian outcome needs
        # no declared q_bounds.
        settings = {**SETTINGS, "cross_fit": False}
        from_pandas = CTMLE(**settings).fit(pandas_frame, **columns).single()
        from_polars = CTMLE(**settings).fit(polars_frame, **columns).single()

        assert from_pandas.psi("ate") == from_polars.psi("ate")
        assert (
            from_pandas.extra["ctmle"].selected_covariates
            == from_polars.extra["ctmle"].selected_covariates
        )
        pandas_report = from_pandas.diagnostics.nuisance_models()
        polars_report = from_polars.diagnostics.nuisance_models()
        assert pandas_report.treatment_role == polars_report.treatment_role
        assert pandas_report.selection.path == polars_report.selection.path
        assert pandas_report.summary() == polars_report.summary()
        assert (
            from_pandas.diagnostics.run_all()["nuisance_models"].detail
            == from_polars.diagnostics.run_all()["nuisance_models"].detail
        )


def test_the_documented_seed_reports_chance_auc_without_a_finding() -> None:
    """The two values a suppressed working-model report would otherwise misread.

    At this seed the search cuts at the intercept-only candidate, so the working
    mechanism is a constant: AUC lands on chance and the calibration slope near ``-2``.
    Those are the two values that would each raise a finding under the ordinary role, and
    pinning them together with a ``completed`` status is the claim this test makes. The
    mutated-role report below is the control that shows the numbers are extreme enough
    for the suppression to be doing the work.

    ``docs/examples/collaborative-tmle.ipynb`` makes the *same* claim on a different fit.
    Its search stops at ``social_support`` rather than at the intercept, so its selected
    mechanism reports an AUC of ``0.516`` and a calibration slope near ``1``, and the
    suppression is what keeps those two numbers from raising a finding there. The extreme
    end of that behaviour is pinned here and not there.

    The fit here is on a binary outcome rather than the notebook's continuous one: a
    cross-fitted fit of a continuous outcome now needs a declared ``q_bounds``
    (the fold and scale rules), which ``instrument_dgp``'s unbounded outcome cannot state
    truthfully. This test keeps cross-fitting, because its subject is a cross-validated
    calibration slope, which an in-sample fit does not have. The notebook needs a
    continuous outcome for its own closing step, so it fits in sample instead. Each is the
    honest choice for its own subject, and the two therefore describe different fits.
    """
    frame, _ = _make_binary_instrument(n=2_000, seed=44)
    result = (
        CTMLE(
            outcome_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            n_folds=3,
            learner_folds=2,
            random_state=44,
            n_jobs=1,
            strategy="greedy",
            selection_folds=3,
            selection_inner_folds=2,
            estimands=("ate",),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2", "W3"))
        .single()
    )

    report = result.diagnostics.nuisance_models()
    metrics = report["propensity"].metrics
    assert metrics["auc"] == pytest.approx(0.490, abs=5e-3)
    assert metrics["calibration_slope"] == pytest.approx(-2.00, abs=5e-2)
    assert report.findings == ()

    item = result.diagnostics.run_all()["nuisance_models"]
    assert item.status.value == "completed"
    assert "C-TMLE greedy selected candidate" in item.detail

    ordinary = replace(report, treatment_role="estimated_treatment_law")
    assert any("poorly calibrated" in note for note in ordinary.findings)
    assert "confounding by these covariates is limited" in ordinary.verdict()


#: The option each test in :class:`TestCombinedWithOtherOptions` combines the selection
#: with.  Fitted once in the ``variants`` fixture.
COMBINATIONS: dict[str, dict[str, object]] = {
    "one_step": {"targeting": "one_step"},
    "weighted_form": {"target_weights": True},
    "linear": {"fluctuation": "linear"},
    "ordered": {"strategy": "ordered"},
}


class TestCombinedWithOtherOptions:
    @pytest.fixture(scope="class")
    def variants(self, frame_and_truth) -> dict[str, object]:
        frame, _ = frame_and_truth
        return {
            name: CTMLE(**{**SETTINGS, **overrides}).fit(frame, outcome="Y", treatment="A").single()
            for name, overrides in COMBINATIONS.items()
        }

    @pytest.mark.parametrize("variant", list(COMBINATIONS))
    def test_the_selection_composes_with_the_targeting_options(
        self, variants, variant: str
    ) -> None:
        result = variants[variant]
        assert result.diagnostics.score_equations().passed
        assert "ctmle" in result.extra

    def test_fold_targeted_composition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="published pooled collaborative estimator"):
            CTMLE(**{**SETTINGS, "targeting_scheme": "fold"})

    def test_missingness_is_refit_inside_selection_folds(self) -> None:
        """In sample, because C-TMLE with cross-fitted missing outcomes is refused below.

        The selection folds are C-TMLE's own, so the missingness refit inside them does
        not depend on outer cross-fitting.
        """
        frame, _ = make_missing_outcome(n=300, seed=17)
        result = (
            CTMLE(
                **{
                    **SETTINGS,
                    "missingness_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
                    "selection_folds": 2,
                    "cross_fit": False,
                }
            )
            .fit(frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        assert result.nuisance.missingness is not None
        assert result.diagnostics.score_equations().passed

    @pytest.mark.parametrize("stratify_folds", ["none", "treatment"])
    def test_cross_fitted_missingness_is_refused_before_selection(
        self, stratify_folds: str
    ) -> None:
        """No audited selection and inference result covers this composition (arm-indexed contract).

        ``stratify_folds='treatment'`` is refused earlier still: C-TMLE now refuses every
        stratified fold policy at construction, before the missingness-specific refusal
        this test is otherwise about ever runs. Both branches still refuse before any
        nuisance is fitted, which is what ``NeverFit.calls == 0`` checks either way.
        """
        frame, _ = make_missing_outcome(n=300, seed=17)
        settings = {
            **SETTINGS,
            **never_fit_learners(),
            "selection_folds": 2,
            "n_folds": 3,
            "stratify_folds": stratify_folds,
        }
        if stratify_folds == "treatment":
            with pytest.raises(ValueError, match="balances the selection and nested folds"):
                CTMLE(**settings)
            assert NeverFit.calls == 0
            return
        estimator = CTMLE(**settings)
        with pytest.raises(CapabilityError, match=r"C-TMLE \(CTMLE, or CollaborativeTMLEMethod\)"):
            estimator.fit(frame, outcome="Y", treatment="A", delta="Delta")
        assert NeverFit.calls == 0

    def test_the_bootstrap_repeats_the_selection(self, frame_and_truth) -> None:
        # The influence-curve standard error treats the selected propensity model as
        # given, so it cannot see the variability the selection itself contributes.
        # The bootstrap can, because each replicate re-runs the search -- which is why
        # _bootstrap_point_estimates goes through the selection hook.
        frame, _ = make_instrument(n=400, seed=7)
        # About the bootstrap re-running the selection, not about cross-fitting; the
        # Gaussian outcome has no q_bounds to declare.
        result = (
            CTMLE(**{**SETTINGS, "n_bootstrap": 4, "cross_fit": False})
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert result.bootstrap is not None
        assert result["ate"].bootstrap is not None
        assert result["ate"].bootstrap.std_error > 0.0


class TestSelectionIsForcedWhenTheOutcomeModelCannotHelp:
    """The test that a do-nothing selector fails, and the reason the others do not.

    Every C-TMLE claim elsewhere in the suite is made on a process whose outcome model is
    *correctly specified* -- :func:`~cleverly.datasets.instrument_dgp` has outcome mean
    ``1 + a + 1.5 W1 + 0.8 W3``, which a GLM fits exactly.  Under collaborative double
    robustness the confounding is then already handled before ``g`` is asked for anything,
    so an **empty** propensity model is genuinely the mean-squared-error-minimising choice,
    and C-TMLE duly selects one: the ordered search selects nothing on all five fixed
    ``n = 700`` seeds in the unit evidence tier.

    That is correct behaviour, and it is also why the variance and RMSE comparisons against
    plain TMLE prove less than they appear to.  A hypothetical selector hard-wired to
    return the empty model would pass all of them, because on such a process adjusting for
    nothing really is unbiased and minimum-variance.  Those comparisons establish that a
    propensity model containing an instrument costs variance -- a fact about plain TMLE --
    not that the collaborative search discriminates between covariates.

    This class removes the escape route.  The outcome learner is reduced to a constant, so
    every bit of confounding adjustment has to come through ``g``, and the empty model goes
    from optimal to badly biased.  A working search must now *include* the confounder, and
    the measured gap is not subtle: mean absolute error 0.036 for the collaborative fit
    against 0.695 for a selector restricted to the empty candidate, a factor of nineteen.
    The fits below are in sample (``cross_fit=False``): this class is about the selection's
    bias under confounding, not about cross-fitting, and ``instrument_dgp``'s Gaussian
    outcome has no ``q_bounds`` a cross-fitted fit could declare (the fold and scale rules).
    """

    SEEDS = (0, 1, 2)
    N = 1500

    #: Settings whose only unusual feature is an outcome model that cannot fit anything.
    #: About the selection's bias under confounding, not about cross-fitting: fit in
    #: sample, since the Gaussian outcome has no q_bounds to declare.
    FORCED: ClassVar[dict[str, object]] = {
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "n_folds": 5,
        "learner_folds": 3,
        "cross_fit": False,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": 0,
        "selection_folds": 3,
    }

    @pytest.fixture(scope="class")
    def fits(self) -> list[tuple[object, object, float]]:
        from sklearn.dummy import DummyRegressor

        dgp = instrument_dgp()
        truth = float(dgp.truth()["ate"])
        settings = {**self.FORCED, "outcome_learner": DummyRegressor(strategy="mean")}
        out = []
        for seed in self.SEEDS:
            frame, _ = dgp.sample(self.N, seed=seed)
            collaborative = CTMLE(**settings).fit(frame, outcome="Y", treatment="A").single()
            nothing = (
                CTMLE(**{**settings, "strategy": "discrete", "candidates": [()]})
                .fit(frame, outcome="Y", treatment="A")
                .single()
            )
            out.append((collaborative, nothing, truth))
        return out

    def test_the_outcome_model_really_is_useless(self, fits) -> None:
        """The premise, and worth asserting precisely rather than trusting the learner.

        If the outcome regression retained any predictive power the empty propensity model
        would stay defensible and this class would prove nothing.  Two things make it
        useless here: it ignores the treatment, so the two counterfactual predictions are
        *identical*; and it ignores the covariates, so the only variation left is the
        fold-to-fold difference between five training means -- about 0.002 on the ``[0, 1]``
        scale, against an outcome that spans it.
        """
        for collaborative, _, _ in fits:
            initial = collaborative.nuisance.outcome
            np.testing.assert_allclose(initial.arms[1.0], initial.arms[0.0], atol=1e-12, rtol=0)
            spread = float(np.std(initial.arms[1.0]))
            assert spread < 0.01, spread
            # And negligible beside the outcome's own spread, which is what "carries no
            # information about this row" means.
            scaled = collaborative.nuisance.scaler.scale(collaborative.data.outcome)
            assert spread < 0.05 * float(np.std(scaled))

    def test_it_selects_the_confounder(self, fits) -> None:
        # The claim no degenerate selector can satisfy: W1 drives both treatment and
        # outcome, so with a constant Qbar it is the covariate g cannot do without.
        for collaborative, _, _ in fits:
            selected = collaborative.extra["ctmle"].selected_covariates
            assert "W1" in selected, selected

    def test_it_never_selects_nothing(self, fits) -> None:
        for collaborative, _, _ in fits:
            assert collaborative.extra["ctmle"].selected_covariates != ()

    def test_a_multistep_selection_reports_the_selected_targeted_state(self, fits) -> None:
        collaborative = fits[0][0]
        selection = collaborative.extra["ctmle"]
        assert selection.n_steps[selection.selected] > 1
        targeted = collaborative.nuisance.targeting_outcome
        assert targeted is not None
        submodel = build_submodel(
            collaborative.data,
            collaborative.nuisance,
            "mean",
            bounds=collaborative.config.g_bounds,
            nuisance_bound=collaborative.config.missingness_bound,
            intermediate_value=None,
        )
        scaled = collaborative.nuisance.scaler.scale(collaborative.data.outcome)
        means = counterfactual_means(
            scaled,
            targeted,
            submodel,
            collaborative.data.weights,
            collaborative.data.observed,
        )
        expected = collaborative.nuisance.scaler.unscale_difference(means[1.0].psi - means[0.0].psi)
        assert collaborative.psi("ate") == pytest.approx(expected, abs=1e-10)

    def test_it_still_leaves_the_instrument_out(self, fits) -> None:
        # And this is now a real exclusion rather than a consequence of selecting nothing:
        # the selected sets here are non-empty by the test above.
        included = sum("W2" in c.extra["ctmle"].selected_covariates for c, _, _ in fits)
        assert included <= 1, "the instrument should be the covariate of last resort"

    def test_a_do_nothing_selector_would_be_badly_biased(self, fits) -> None:
        # The comparison that gives the three assertions above their force. Restricted to
        # the empty candidate the estimator is essentially unadjusted, and on a confounded
        # process that is a first-order error rather than a variance penalty.
        collaborative = np.array([abs(c.psi("ate") - t) for c, _, t in fits])
        nothing = np.array([abs(n.psi("ate") - t) for _, n, t in fits])
        assert nothing.mean() > 0.4, nothing
        assert collaborative.mean() < 0.15, collaborative
        assert nothing.mean() > 5.0 * collaborative.mean()

    def test_the_empty_fit_is_biased_beyond_its_own_plugin_spread(self, fits) -> None:
        # Not a coverage study -- three fits -- but the failure here is systematic, not a
        # matter of luck: the empty model's interval is narrow and centred in the wrong
        # place, which is the signature of a bias a standard error cannot see.
        #
        # ``plugin_interval`` and not ``ci``: neither of these fits publishes an interval,
        # because both take a selector path. The plug-in interval is a diagnostic, and
        # this test reads it as one -- the claim is about where the point estimate sits
        # relative to the spread of its own curve, not about coverage.
        for collaborative, nothing, truth in fits:
            low, high = collaborative["ate"].plugin_interval
            assert low <= truth <= high, (low, high, truth)
            low, high = nothing["ate"].plugin_interval
            assert not (low <= truth <= high), (low, high, truth)
            # The file's most interval-dependent assertion is also its witness that the
            # package publishes no interval here.
            with pytest.raises(CapabilityError) as raised:
                _ = collaborative["ate"].ci
            assert WORKING_MECHANISM_NOT_INFERENTIAL in str(raised.value)


#: The most of the exact variance the reported working-mechanism diagnostic may account
#: for.  **Declared before the run that measured it**, from roadmap row RM12's own
#: published numbers: reported standard error 0.0441 against an HC0 standard error of
#: 0.0545 on ``make_instrument(n=2000, seed=44)``, a variance ratio of 0.65.  The bound is
#: 0.81, which is 0.90 on the standard-error scale, so the witness has real headroom and
#: still fails long before the two agree.
WORKING_MECHANISM_VARIANCE_RATIO = 0.81


def _hc0_treatment_coefficient(frame, covariates) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    """The least-squares coefficient on ``A`` given every covariate, and its HC0 error.

    Written out here rather than taken from a library, so the number the witness compares
    against is one this test computes from the sandwich formula and not one the package
    could also have got wrong.
    """
    columns = frame[["A", *covariates]].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(columns)), columns])
    response = frame["Y"].to_numpy(dtype=float)
    gram_inverse = np.linalg.inv(design.T @ design)
    beta = gram_inverse @ design.T @ response
    residual = response - design @ beta
    meat = design.T @ (design * residual[:, None] ** 2)
    covariance = gram_inverse @ meat @ gram_inverse
    # Column 0 is the intercept, so column 1 is A.
    return float(beta[1]), float(np.sqrt(covariance[1, 1]))


class TestTheWorkingMechanismDiagnosticIsNotTheEstimatorsVariance:
    """Corroboration of RM12's first witness on a continuous law.

    The exact witness is
    ``tests/unit/test_ctmle.py::TestTheWorkingMechanismDiagnosticMissesTheExactVariance``,
    on a finite-support law whose variances are closed-form.  This class checks the same
    gap where the exact variance is not available and the sandwich stands in for it.

    A ``discrete`` fit whose only candidate is the intercept-only one, with a correct
    linear outcome regression, **is** the least-squares coefficient on the treatment given
    every covariate. The test computes that coefficient's HC0 sandwich variance. The
    reported plug-in curve variance is the intercept-only representer's,
    :math:`\\sigma^2 \\nu^2 / n`, which ignores the association between the treatment and
    the covariates and is therefore too small.

    Like the exact witness, it is a **nonzero witness** as the scientific-change rule demands. A
    mutation that made the retained diagnostic the *right* variance -- which would make
    the whole refusal pointless, because then there would be nothing wrong with reporting
    it -- drives the ratio to one and fails this test. A test that only asserted "the
    accessor raises" would pass against that mutation.
    """

    @pytest.fixture(scope="class")
    def fitted(self):  # type: ignore[no-untyped-def]
        # The law and the seed roadmap row RM12 published its measurement on.
        frame, _ = make_instrument(n=2000, seed=44)
        covariates = [name for name in frame.columns if name.startswith("W")]
        result = (
            linear_ctmle(
                "discrete",
                candidates=((),),
                selection_folds=3,
                selection_inner_folds=2,
                estimands=("ate",),
                ctmle_estimand="ate",
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()
        )
        return result, frame, covariates

    def test_the_fit_is_the_least_squares_coefficient(self, fitted) -> None:
        """What makes the comparison below one of two variances of *one* estimator."""
        result, frame, covariates = fitted
        coefficient, _ = _hc0_treatment_coefficient(frame, covariates)
        assert result["ate"].psi == pytest.approx(coefficient, abs=1e-9)

    def test_the_reported_diagnostic_is_smaller_than_the_exact_variance(self, fitted) -> None:
        result, frame, covariates = fitted
        _, exact_error = _hc0_treatment_coefficient(frame, covariates)
        reported = result["ate"].plugin_std_error

        ratio = (reported / exact_error) ** 2
        assert ratio <= WORKING_MECHANISM_VARIANCE_RATIO, (reported, exact_error, ratio)

    def test_the_package_reports_no_interval_for_it(self, fitted) -> None:
        """The refusal and the witness belong in one place: the gap is why it refuses."""
        result, _, _ = fitted
        with pytest.raises(CapabilityError) as raised:
            _ = result["ate"].ci
        assert WORKING_MECHANISM_NOT_INFERENTIAL in str(raised.value)


class TestTheSelectorPathsPublishNoInference:
    """Every selector strategy refuses all three accessors; an ordinary TMLE keeps them.

    The ordinary control is not optional. Without it a refusal broadened to every fit on
    this law passes every other test in this class. ``oat`` was that control until RM20
    gave it a status of its own, so the tests here also pin that it refuses with its own
    reason and not with the selector reason.
    """

    #: The key :meth:`_fit` reads as the ordinary TMLE on the same law and learners.
    ORDINARY = "tmle"

    @classmethod
    def _fit(cls, strategy: str):  # type: ignore[no-untyped-def]
        """A fresh fit, for a test that calls a facade and so writes the result's cache."""
        frame, _ = make_instrument(n=400, seed=5)
        covariates = [name for name in frame.columns if name.startswith("W")]
        estimator = (
            TMLE(**linear_in_sample(estimands=("ate",)))
            if strategy == cls.ORDINARY
            else linear_ctmle(strategy, estimands=("ate",), **SELECTOR_CONFIGS.get(strategy, {}))
        )
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=covariates).single()

    @pytest.fixture(scope="class")
    def shared(self) -> Callable[[str], Any]:
        """One fit per strategy, for the tests that only read estimates and frames.

        Those reads write nothing on the result. A test that calls ``summary``,
        ``assess``, a diagnostics facade or a capability row fills the assessment cache,
        so it takes a fresh fit from :meth:`_fit` instead.
        """
        fits: dict[str, Any] = {}

        def fit(strategy: str) -> Any:
            if strategy not in fits:
                fits[strategy] = self._fit(strategy)
            return fits[strategy]

        return fit

    @pytest.mark.parametrize("strategy", sorted(SELECTOR_CONFIGS))
    @pytest.mark.parametrize("accessor", ["ci", "pvalue", "std_error"])
    def test_a_selector_path_refuses_and_names_its_cause(
        self, shared: Callable[[str], Any], strategy: str, accessor: str
    ) -> None:
        estimate = shared(strategy)["ate"]
        assert estimate.inference == "working_mechanism_plugin"
        with pytest.raises(CapabilityError) as raised:
            getattr(estimate, accessor)
        assert WORKING_MECHANISM_NOT_INFERENTIAL in str(raised.value)

    @pytest.mark.parametrize("accessor", ["ci", "pvalue", "std_error"])
    def test_an_ordinary_tmle_still_answers(
        self, shared: Callable[[str], Any], accessor: str
    ) -> None:
        estimate = shared(self.ORDINARY)["ate"]
        assert estimate.inference == "influence_curve"
        assert getattr(estimate, accessor) is not None

    @pytest.mark.parametrize("accessor", ["ci", "pvalue", "std_error"])
    def test_the_outcome_adaptive_path_refuses_by_its_own_reason(
        self, shared: Callable[[str], Any], accessor: str
    ) -> None:
        estimate = shared("oat")["ate"]
        assert estimate.inference == "generated_design_plugin"
        with pytest.raises(CapabilityError) as raised:
            getattr(estimate, accessor)
        assert NON_INFERENTIAL["generated_design_plugin"].reason in str(raised.value)
        assert WORKING_MECHANISM_NOT_INFERENTIAL not in str(raised.value)

    def test_the_retained_diagnostic_is_the_refused_number(
        self, shared: Callable[[str], Any]
    ) -> None:
        """Reframed, not recomputed: an ordinary fit answers both names with one value."""
        estimate = shared(self.ORDINARY)["ate"]
        assert estimate.plugin_std_error == estimate.std_error
        assert estimate.plugin_interval == estimate.ci

    def test_the_frame_swaps_the_inference_columns(self, shared: Callable[[str], Any]) -> None:
        refused = shared("discrete").to_frame()
        ordinary = shared(self.ORDINARY).to_frame()

        assert "inference" in refused.columns
        assert set(refused.columns) & {"std_err", "ci_lower", "ci_upper", "p_value"} == set()
        assert {
            "plugin_std_err",
            "plugin_interval_lower",
            "plugin_interval_upper",
        } <= set(refused.columns)
        assert list(refused["inference"]) == ["working_mechanism_plugin"]

        # The ordinary frame is unchanged, which is what keeps every other fit in the
        # package byte-identical.
        assert "inference" not in ordinary.columns
        assert {"std_err", "ci_lower", "ci_upper", "p_value"} <= set(ordinary.columns)

    def test_the_summary_prints_the_refusal_and_no_interval_heading(self) -> None:
        result = self._fit("greedy")
        summary = result.summary()

        # The refusal's own sentence, with only its first letter raised, so the summary
        # cannot paraphrase the raise into a second claim.
        assert WORKING_MECHANISM_NOT_INFERENTIAL[1:] in summary
        assert "working-mechanism se" in summary
        # No `95% CI` heading with a "-" under it: the whole column is refused, and a dash
        # beneath that heading would still tell a reader an interval belongs there.
        assert "95% CI" not in summary
        assert "p_value" not in summary
        # The selector is named in the facts block, so a reader does not have to go to the
        # nuisance report to learn which mechanism the refusal is about.
        assert result.extra["ctmle"].describe() in summary

    def test_the_assessment_carries_the_same_sentence(self) -> None:
        result = self._fit("greedy")
        # The detail string of the nuisance-model row, which is where the label goes:
        # the capability table is keyed by method and cannot tell a selector path from
        # the outcome-adaptive one, so it has no row to mark unavailable here.
        detail = result.diagnostics.run_all()["nuisance_models"].detail
        assert WORKING_MECHANISM_ASSESSMENT_NOTE in detail
        # And it reaches the assessment a reader actually prints.
        assert WORKING_MECHANISM_ASSESSMENT_NOTE in result.assess().summary()

    def test_the_evalue_capability_goes_unavailable_rather_than_raising(self) -> None:
        """The row must say so, not raise from inside the computation.

        Only an ``_EValueRefusal`` raised in the selection reaches the capability row. A
        refusal left to fall out of ``estimate.ci`` deep in the derivation would leave the
        row advertising ``available=True`` beside a call that raises, which is the failure
        this test exists for. RM11 keeps a ratio E-value "because that formula reads only
        the estimate and its interval"; on this path there is no interval to read.
        """
        result = self._fit("greedy")
        capability = result.sensitivity.capability("evalue")
        assert capability.available is False
        assert WORKING_MECHANISM_NOT_INFERENTIAL in (capability.reason or "")
        assert result.sensitivity.capability("evalue").available is False

        # The ordinary fit keeps it, and the outcome-adaptive path refuses by its own reason.
        assert self._fit(self.ORDINARY).sensitivity.capability("evalue").available
        adaptive = self._fit("oat").sensitivity.capability("evalue")
        assert adaptive.available is False
        assert NON_INFERENTIAL["generated_design_plugin"].reason in (adaptive.reason or "")

    def test_the_truncation_curve_reports_the_diagnostic_rather_than_raising(self) -> None:
        """Reachable on a C-TMLE fit, and it builds an inference-shaped frame per bound."""
        curve = self._fit("greedy").diagnostics.truncation_curve()
        assert "plugin_std_err" in curve.columns
        assert "ci_lower" not in curve.columns
