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

from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
import sklearn.linear_model

from cleverly import SuperLearner, load
from cleverly.datasets import (
    instrument_dgp,
    make_instrument,
    make_missing_outcome,
    make_multi_arm,
)
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators.targeting import build_submodel
from cleverly.inference.influence import counterfactual_means
from cleverly.validation.nuisance import NUISANCE_SELECTION_MISSING
from tests.conftest import FAST_KWARGS

TMLE_SETTINGS = {**FAST_KWARGS, "estimands": ("ate", "ey1", "ey0")}

#: Three selection folds rather than the default five keeps this file in the fast tier;
#: nothing asserted here turns on the fold count.
SETTINGS = {**TMLE_SETTINGS, "selection_folds": 3}


@pytest.fixture(scope="module")
def frame_and_truth() -> tuple[object, dict[str, float]]:
    return make_instrument(n=900, seed=5)


@pytest.fixture(scope="module")
def fit(frame_and_truth) -> object:
    frame, _ = frame_and_truth
    return CTMLE(**SETTINGS).fit(frame, outcome="Y", treatment="A").single()


class TestTheFit:
    def test_it_recovers_the_truth(self, fit, frame_and_truth) -> None:
        _, truth = frame_and_truth
        estimate = fit["ate"]
        assert abs(estimate.psi - truth["ate"]) < 3.0 * estimate.std_error

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

    def test_it_buys_a_smaller_standard_error_than_a_plain_fit(self, fit, frame_and_truth) -> None:
        # One sample, so this is a statement about this fit rather than about the
        # estimator -- but the mechanism is deterministic given the data: a narrower
        # propensity model means a smaller 1/g and a smaller influence curve.
        frame, _ = frame_and_truth
        plain = TMLE(**TMLE_SETTINGS).fit(frame, outcome="Y", treatment="A").single()
        assert fit["ate"].std_error < plain["ate"].std_error


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
        frame, _ = make_multi_arm(n=600, seed=5)
        result = (
            CTMLE(
                outcome_learner=sklearn.linear_model.LinearRegression(),
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
            "VERDICT: C-TMLE working-model metrics are descriptive; inspect the selection "
            "and support reports."
        )
        assert diagnostics.verdict() in diagnostics.summary()

        ordinary = replace(diagnostics, treatment_role="estimated_treatment_law")
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
        from_pandas = CTMLE(**SETTINGS).fit(pandas_frame, **columns).single()
        from_polars = CTMLE(**SETTINGS).fit(polars_frame, **columns).single()

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
    """The exact numbers ``docs/examples/collaborative-tmle.md`` asks a reader to read.

    At this seed the search cuts at the intercept-only candidate, so the working
    mechanism is a constant: AUC lands on chance and the calibration slope on ``-2``.
    Those are the two values the page tells the reader not to interpret as a treatment
    law, and they are also the two values that would each raise a finding under the
    ordinary role.  Pinning them together with a ``completed`` status is the claim the
    page makes; the mutated-role report below is the control that shows the numbers are
    extreme enough for the suppression to be doing the work.
    """
    frame, _ = make_instrument(n=2_000, seed=44)
    result = (
        CTMLE(
            outcome_learner=sklearn.linear_model.LinearRegression(),
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
    assert metrics["auc"] == pytest.approx(0.499, abs=5e-3)
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
        frame, _ = make_missing_outcome(n=300, seed=17)
        result = (
            CTMLE(
                **{
                    **SETTINGS,
                    "missingness_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
                    "selection_folds": 2,
                    "n_folds": 3,
                }
            )
            .fit(frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        assert result.nuisance.missingness is not None
        assert result.diagnostics.score_equations().passed

    def test_the_bootstrap_repeats_the_selection(self, frame_and_truth) -> None:
        # The influence-curve standard error treats the selected propensity model as
        # given, so it cannot see the variability the selection itself contributes.
        # The bootstrap can, because each replicate re-runs the search -- which is why
        # _bootstrap_point_estimates goes through the selection hook.
        frame, _ = make_instrument(n=400, seed=7)
        result = (
            CTMLE(**{**SETTINGS, "n_bootstrap": 4}).fit(frame, outcome="Y", treatment="A").single()
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
    the measured gap is not subtle: mean absolute error 0.017 for the collaborative fit
    against 0.696 for a selector restricted to the empty candidate, a factor of forty-one.
    """

    SEEDS = (0, 1, 2)
    N = 1500

    #: Settings whose only unusual feature is an outcome model that cannot fit anything.
    FORCED: ClassVar[dict[str, object]] = {
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "n_folds": 5,
        "learner_folds": 3,
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

    def test_the_collaborative_fit_covers_the_truth_where_the_empty_one_cannot(self, fits) -> None:
        # Not a coverage study -- three fits -- but the failure here is systematic, not a
        # matter of luck: the empty model's interval is narrow and centred in the wrong
        # place, which is the signature of a bias a standard error cannot see.
        for collaborative, nothing, truth in fits:
            low, high = collaborative["ate"].ci
            assert low <= truth <= high, (low, high, truth)
            low, high = nothing["ate"].ci
            assert not (low <= truth <= high), (low, high, truth)
