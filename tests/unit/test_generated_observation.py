"""Point generated outcomes on observation and intermediate structure (X82 part E2).

Missing outcomes, missing treatments, controlled direct effects, the natural-course targets
(``ey_obs``, ``par``, ``paf``), baseline strata and the clustered plasmode. Each truth is
checked against a closed form computed here, and each mask has a mutation control that fails
when the mask is dropped.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    ControlledDirectEffect,
    CounterfactualMean,
    DRTMLEMethod,
    NaturalCourseMean,
    PointTreatment,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
)
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.validation import (
    BernoulliAdjustmentOutcome,
    GaussianAdjustmentOutcome,
    refute,
)
from cleverly.validation._point_generated import (
    PLASMODE_G_BOUNDS,
    PlasmodeLaw,
    PointTarget,
    arm_means,
    held_mixture,
    held_weights,
)

REFUTE = sys.modules["cleverly.validation.refute"]


def _expit(values: Any) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float))))


def _frame(n: int = 300, *, seed: int = 3, cluster_size: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = (rng.random(n) < _expit(0.5 * w[:, 0])).astype(int)
    delta = (rng.random(n) < _expit(1.2 + 0.6 * w[:, 1] + 0.4 * a)).astype(int)
    recorded = (rng.random(n) < _expit(1.5 - 0.5 * w[:, 0])).astype(int)
    z = (rng.random(n) < _expit(-0.2 + 0.8 * a + 0.5 * w[:, 0])).astype(int)
    return pd.DataFrame(
        {
            "W1": w[:, 0],
            "W2": w[:, 1],
            "S": (w[:, 1] > 0).astype(int),
            "A": a,
            "Ag": np.where(recorded == 1, a, np.nan),
            "Yg": rng.normal(size=n),
            "Yb": (rng.random(n) < 0.4).astype(float),
            "D": delta,
            "R": recorded,
            "Z": z,
            "C": np.arange(n) // cluster_size,
        }
    )


def _fit(
    frame: pd.DataFrame,
    estimand: Any,
    *,
    outcome: str = "Yg",
    treatment: str = "A",
    adjustment: tuple[str, ...] = ("W1", "W2"),
    method: Any = "tmle",
    **design: Any,
) -> Any:
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome=outcome, treatment=treatment, adjustment=adjustment, **design
        ),
    )
    learner = LogisticRegression() if outcome == "Yb" else LinearRegression()
    return study.identify(estimand).estimate(
        method=method,
        outcome_learner=learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=1,
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


#: A logit risk whose arm contrast varies with W, so a mask that is dropped moves the truth.
LOGIT = BernoulliAdjustmentOutcome(intercept=-0.3, effect=1.0, adjustment_scale=1.5)


class TestTruths:
    """Each truth against a closed form, with the mutation that breaks it."""

    covariates = np.array([[-1.0, 0.5], [0.0, -0.5], [1.0, 0.0], [2.0, 1.0]])

    def _data(self, treatment: Any, **fields: Any) -> Any:
        return SimpleNamespace(
            arm_codes=(0.0, 1.0),
            treatment_levels=(0, 1),
            treatment=np.asarray(treatment, dtype=float),
            weights=np.ones(4),
            covariates=self.covariates,
            **fields,
        )

    def _closed(self, process: BernoulliAdjustmentOutcome, a: float, z: float = 0.0) -> Any:
        x = self.covariates
        s = ((x - x.mean(axis=0)) / x.std(axis=0)).sum(axis=1) / np.sqrt(2.0)
        linear = process.intercept + process.adjustment_scale * s + process.effect * a
        return _expit(linear + process.intermediate_effect * z)

    def test_the_natural_course_targets(self) -> None:
        data = self._data([1, 0, 1, 0])
        m1, m0 = self._closed(LOGIT, 1.0), self._closed(LOGIT, 0.0)
        observed = np.mean(np.where(data.treatment == 1.0, m1, m0))
        expected = {
            "natural": observed,
            "par": observed - m0.mean(),
            "paf": (observed - m0.mean()) / observed,
        }
        for kind, value in expected.items():
            target = PointTarget(
                kind,  # type: ignore[arg-type]
                "all",
                None if kind == "natural" else 0.0,
                None,
                arms=(0.0, 1.0),
            )
            means = arm_means(LOGIT, self.covariates, target.codes, data)
            mixture = held_mixture(target, data)
            truth = target.truth(means, held_weights(target, data), mixture)
            assert truth == pytest.approx(value, rel=1e-12)
        # Mutation: every row at the reference arm has no attributable part.
        par = PointTarget("par", "all", 0.0, None, arms=(0.0, 1.0))
        flat = {0.0: np.ones(4), 1.0: np.zeros(4)}
        means = arm_means(LOGIT, self.covariates, par.codes, data)
        assert par.truth(means, np.ones(4), flat) == 0.0
        assert abs(expected["par"]) > 0.05

    def test_a_natural_course_target_needs_its_mixture(self) -> None:
        target = PointTarget("natural", "all", None, None, arms=(0.0, 1.0))
        means = arm_means(LOGIT, self.covariates, target.codes, self._data([1, 0, 1, 0]))
        with pytest.raises(ValueError, match="arm probabilities"):
            target.truth(means, np.ones(4))

    def test_the_controlled_direct_effect_fixes_the_intermediate(self) -> None:
        process = replace(LOGIT, intermediate_effect=1.2)
        data = self._data([1, 0, 1, 0])
        target = PointTarget("difference", "all", 1.0, 0.0, intermediate=1.0)
        means = arm_means(process, self.covariates, target.codes, data, intermediate=1.0)
        expected = np.mean(self._closed(process, 1.0, 1.0) - self._closed(process, 0.0, 1.0))
        assert target.truth(means, np.ones(4)) == pytest.approx(expected, rel=1e-12)
        # Mutation: the contrast at z = 0 is another number on a logit link.
        at_zero = arm_means(process, self.covariates, target.codes, data, intermediate=0.0)
        assert abs(target.truth(at_zero, np.ones(4)) - expected) > 1e-4

    def test_a_stratum_target_sums_its_own_rows(self) -> None:
        data = self._data([1, 0, 1, 0], strata=np.array([0, 0, 1, 1]))
        target = PointTarget("difference", "all", 1.0, 0.0, stratum=1)
        weights = held_weights(target, data)
        np.testing.assert_array_equal(weights, [0.0, 0.0, 1.0, 1.0])
        means = arm_means(LOGIT, self.covariates, target.codes, data)
        contrast = self._closed(LOGIT, 1.0) - self._closed(LOGIT, 0.0)
        assert target.truth(means, weights) == pytest.approx(contrast[2:].mean(), rel=1e-12)
        # Mutation: without the stratum mask the truth is the whole-sample contrast.
        assert abs(contrast.mean() - contrast[2:].mean()) > 0.01

    def test_the_covariate_part_of_a_natural_course_target(self) -> None:
        data = self._data([1, 0, 1, 0])
        weights = np.ones(4)
        flat = BernoulliAdjustmentOutcome(intercept=0.0, effect=0.0, adjustment_scale=0.0)
        moved = BernoulliAdjustmentOutcome(intercept=0.0, effect=0.6, adjustment_scale=0.0)
        for kind in ("natural", "par", "paf"):
            target = PointTarget(
                kind,  # type: ignore[arg-type]
                "all",
                None if kind == "natural" else 0.0,
                None,
                arms=(0.0, 1.0),
            )
            mixture = held_mixture(target, data)
            constant = arm_means(flat, self.covariates, target.codes, data)
            assert target.covariate_part_vanishes(constant, weights, mixture)
            # An arm effect makes m(A_i) differ between the rows of two arms.
            varying = arm_means(moved, self.covariates, target.codes, data)
            assert not target.covariate_part_vanishes(varying, weights, mixture)


class TestPlasmodeObservation:
    def test_each_observation_model_is_frozen_and_clipped(self, frame: pd.DataFrame) -> None:
        cases = (
            (_fit(frame, ATE(), missingness="D"), "response", "outcome response"),
            (
                _fit(frame, ATE(), treatment="Ag", treatment_missingness="R"),
                "recording",
                "treatment recording",
            ),
            (
                _fit(frame, ControlledDirectEffect(intermediate=1.0), intermediate="Z"),
                "mediator",
                "intermediate",
            ),
        )
        for result, field, label in cases:
            law = PlasmodeLaw.fit(result)
            model = np.asarray(getattr(law, field))
            assert law.observation_models == (label,)
            assert model.min() >= PLASMODE_G_BOUNDS[0] and model.max() <= PLASMODE_G_BOUNDS[1]
            assert np.ptp(model) > 0.1

    def test_a_missing_treatment_draw_keeps_every_arm_on_its_recorded_rows(
        self, frame: pd.DataFrame
    ) -> None:
        result = _fit(frame, ATE(), treatment="Ag", treatment_missingness="R")
        law = PlasmodeLaw.fit(result)
        target = PointTarget("difference", "all", 1.0, 0.0)
        process = GaussianAdjustmentOutcome()
        for seed in range(5):
            sample = law.draw(result.data, process, seed=seed, target=target)
            recorded = sample.treatment_recorded
            assert 0 < recorded.sum() < sample.n
            assert set(np.unique(sample.treatment[recorded])) == {0.0, 1.0}
            assert np.all(np.isnan(sample.treatment[~recorded]))

    def test_the_arm_retry_is_rare(self, frame: pd.DataFrame) -> None:
        """Review nit: a retry conditions the draw, so it must almost never happen."""
        result = _fit(frame, ATE(), treatment="Ag", treatment_missingness="R")
        law = PlasmodeLaw.fit(result)
        recorded = law.recording
        assert recorded is not None
        # Each drawn row is a uniform support point, recorded and of arm a with probability
        # p = mean_i r_i g(a | W_i), so a draw holds no such row with probability (1 - p)^n.
        n = result.data.n
        for index in range(len(law.arm_codes)):
            p = float(np.mean(recorded * law.probabilities[:, index]))
            assert n * np.log1p(-p) < np.log(1e-12)

    def test_a_clustered_draw_resamples_and_relabels_whole_clusters(
        self, frame: pd.DataFrame
    ) -> None:
        result = _fit(frame, ATE(), cluster="C")
        law = PlasmodeLaw.fit(result)
        rows, ids = law._rows(np.random.default_rng(0))
        assert ids is not None and law.clusters is not None
        original = law.clusters[rows]
        sizes = np.bincount(law.clusters)
        assert np.unique(ids).size == np.unique(law.clusters).size
        for copy in np.unique(ids):
            members = original[ids == copy]
            # One copy is one whole original cluster.
            assert np.unique(members).size == 1 and members.size == sizes[members[0]]
        # Mutation: the original ids would merge two copies of a cluster into one.
        assert np.unique(original).size < np.unique(ids).size

    def test_the_natural_course_truth_is_the_law_itself(self, frame: pd.DataFrame) -> None:
        result = _fit(frame, NaturalCourseMean(), outcome="Yb")
        law = PlasmodeLaw.fit(result)
        target = PointTarget("natural", "all", None, None, arms=(0.0, 1.0))
        means, weights, mixture = law.means(LOGIT, result.data, target)
        exact = target.truth(means, weights, mixture)
        rng = np.random.default_rng(11)
        rows = rng.integers(0, result.data.n, size=400_000)
        treated = rng.random(rows.size) < law.treated[rows]
        risk = np.where(treated, means[1.0][rows], means[0.0][rows])
        spread = float(np.std(risk) / np.sqrt(rows.size))
        assert abs(float(np.mean(risk)) - exact) < 4 * spread
        # Mutation: the held arms in place of the frozen model describe another law.
        held = held_mixture(target, result.data)
        assert abs(target.truth(means, weights, held) - exact) > 4 * spread


class TestEndToEnd:
    """Real fits with GLM learners at R = 400, the calibrated minimum."""

    @pytest.mark.parametrize(
        ("case", "seed"),
        [
            ("missing_outcome", 4),
            ("missing_treatment", 4),
            ("controlled_direct_effect", 4),
            ("strata", 4),
        ],
    )
    def test_three_band_designs(self, frame: pd.DataFrame, case: str, seed: int) -> None:
        process: Any = GaussianAdjustmentOutcome()
        alias = "ate"
        if case == "missing_outcome":
            result = _fit(frame, ATE(), missingness="D")
        elif case == "missing_treatment":
            result = _fit(frame, ATE(), treatment="Ag", treatment_missingness="R")
        elif case == "controlled_direct_effect":
            result = _fit(frame, ControlledDirectEffect(intermediate=1.0), intermediate="Z")
            process = GaussianAdjustmentOutcome(intermediate_effect=0.7)
        else:
            result = _fit(frame, ATE(), adjustment=("W1", "W2", "S"), strata=("S",))
            alias = "ate[S=1]"
        report = refute(
            result,
            estimand=alias,
            tests=("simulated_outcome", "plasmode"),
            simulated_outcome=process,
            n_replicates=400,
            random_state=seed,
        )
        for name in ("simulated_outcome", "plasmode"):
            assert report[name].declared_effect == pytest.approx(1.0)
            assert report[name].rung == "L2" and report[name].passed is True
            assert len(report[name].readings) == 3

    def test_a_logit_controlled_direct_effect(self, frame: pd.DataFrame) -> None:
        process = replace(LOGIT, intermediate_effect=1.2)
        report = refute(
            _fit(frame, ControlledDirectEffect(intermediate=1.0), outcome="Yb", intermediate="Z"),
            tests=("simulated_outcome", "plasmode"),
            simulated_outcome=process,
            n_replicates=400,
            random_state=4,
        )
        assert [r.name for r in report["simulated_outcome"].readings] == ["bias"]
        assert report["simulated_outcome"].passed is True
        assert report["plasmode"].passed is True and len(report["plasmode"].readings) == 3
        assert "a frozen intermediate model" in report["plasmode"].scope_sentence

    @pytest.mark.parametrize(
        ("estimand", "alias", "outcome", "process", "seed"),
        [
            (NaturalCourseMean(), "ey_obs", "Yg", GaussianAdjustmentOutcome(), 5),
            (PopulationAttributableRisk(), "par", "Yb", BernoulliAdjustmentOutcome(), 4),
            (PopulationAttributableFraction(), "paf", "Yb", BernoulliAdjustmentOutcome(), 4),
        ],
    )
    def test_the_natural_course_targets(
        self,
        frame: pd.DataFrame,
        estimand: Any,
        alias: str,
        outcome: str,
        process: Any,
        seed: int,
    ) -> None:
        report = refute(
            _fit(frame, estimand, outcome=outcome),
            estimand=alias,
            tests=("simulated_outcome", "plasmode"),
            simulated_outcome=process,
            n_replicates=400,
            random_state=seed,
        )
        held = report["simulated_outcome"]
        # The covariate part m(A, W) remains on the held rows, so the held design reads bias.
        assert [reading.name for reading in held.readings] == ["bias"]
        assert held.passed is True
        assert report["plasmode"].passed is True and len(report["plasmode"].readings) == 3

    def test_a_clustered_plasmode_with_a_cluster_intercept(self) -> None:
        clustered = _frame(cluster_size=4)
        report = refute(
            _fit(clustered, CounterfactualMean(treatment=1), cluster="C"),
            estimand="ey1",
            tests=("plasmode",),
            simulated_outcome=GaussianAdjustmentOutcome(cluster_sd=1.0),
            n_replicates=400,
            random_state=4,
        )
        test = report["plasmode"]
        assert test.rung == "L2" and test.passed is True
        assert "whole clusters with a cluster intercept of SD 1" in test.scope_sentence


class TestMutations:
    """Each mask, dropped, moves a reading out of its band."""

    def test_dropping_the_stratum_mask_biases_the_held_design(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _fit(frame, ATE(), outcome="Yb", adjustment=("W1", "W2", "S"), strata=("S",))

        def run() -> Any:
            return refute(
                result,
                estimand="ate[S=1]",
                tests=("simulated_outcome",),
                simulated_outcome=LOGIT,
                n_replicates=400,
                random_state=4,
            )["simulated_outcome"]

        assert run().passed is True
        monkeypatch.setattr(REFUTE, "_stratum_code", lambda *_: None)
        assert run().passed is False

    def test_a_truth_at_the_wrong_intermediate_biases_the_held_design(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _fit(
            frame, ControlledDirectEffect(intermediate=1.0), outcome="Yb", intermediate="Z"
        )
        process = BernoulliAdjustmentOutcome(intercept=-0.5, effect=0.8, intermediate_effect=1.2)
        original = REFUTE.arm_means
        monkeypatch.setattr(
            REFUTE,
            "arm_means",
            lambda *args, **kwargs: original(*args, **{**kwargs, "intermediate": 0.0}),
        )
        test = refute(
            result,
            tests=("simulated_outcome",),
            simulated_outcome=process,
            n_replicates=400,
            random_state=4,
        )["simulated_outcome"]
        assert test.passed is False and not test.readings[0].inside

    def test_a_refit_blind_to_the_clusters_undercovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _fit(_frame(cluster_size=4), CounterfactualMean(treatment=1), cluster="C")
        original = PlasmodeLaw.draw
        monkeypatch.setattr(
            PlasmodeLaw,
            "draw",
            lambda self, *args, **kwargs: replace(original(self, *args, **kwargs), cluster=None),
        )
        test = refute(
            result,
            estimand="ey1",
            tests=("plasmode",),
            simulated_outcome=GaussianAdjustmentOutcome(cluster_sd=1.0),
            n_replicates=400,
            random_state=4,
        )["plasmode"]
        assert test.passed is False
        assert not test.readings[1].inside and not test.readings[2].inside


class TestUnrecordedRows:
    """Review R2: no admitted method reads the outcome of a row with no recorded treatment."""

    @pytest.mark.parametrize("method", ["tmle", "drtmle", "adaptive_iptw"])
    def test_an_unrecorded_outcome_inside_the_range_is_not_read(
        self, frame: pd.DataFrame, method: str
    ) -> None:
        chosen: Any = method
        if method == "drtmle":
            chosen = DRTMLEMethod(
                reduced_outcome_learner=LinearRegression(),
                reduced_treatment_learner=LogisticRegression(max_iter=1000),
            )
        result = _fit(frame, ATE(), treatment="Ag", treatment_missingness="R", method=chosen)
        data = result.data
        recorded = data.treatment_recorded
        y = np.asarray(data.outcome, dtype=float)
        garbage = y.copy()
        garbage[~recorded] = np.random.default_rng(9).uniform(
            y[recorded].min(), y[recorded].max(), size=int((~recorded).sum())
        )
        # The held draw copies a recorded value onto these rows, as the baseline does here.
        copied = np.where(recorded, y, y[np.flatnonzero(recorded)[0]])
        before = result.estimator.refit(
            data.with_outcome(copied, family="gaussian"), random_state=1
        )["ate"].psi
        after = result.estimator.refit(
            data.with_outcome(garbage, family="gaussian"), random_state=1
        )["ate"].psi
        assert after == before

    def test_an_outcome_outside_the_range_is_read_through_the_bounds(
        self, frame: pd.DataFrame
    ) -> None:
        """Mutation control: why the held draw copies a recorded value onto these rows."""
        result = _fit(frame, ATE(), treatment="Ag", treatment_missingness="R")
        data = result.data
        far = np.asarray(data.outcome, dtype=float).copy()
        far[~data.treatment_recorded] = 1e6
        before = result.estimator.refit(data, random_state=1)["ate"].psi
        after = result.estimator.refit(data.with_outcome(far, family="gaussian"), random_state=1)[
            "ate"
        ].psi
        assert after != before

    def test_the_held_draw_copies_a_recorded_value(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _fit(frame, ATE(), treatment="Ag", treatment_missingness="R")
        seen: list[Any] = []
        original = REFUTE._run_empirical_refits

        def capture(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs["replacement"](0, 123))
            return original(*args, **kwargs)

        monkeypatch.setattr(REFUTE, "_run_empirical_refits", capture)
        refute(result, tests=("simulated_outcome",), n_replicates=10, random_state=1)
        drawn = np.asarray(seen[0].outcome, dtype=float)
        recorded = result.data.treatment_recorded
        first = int(np.flatnonzero(recorded)[0])
        assert np.all(drawn[~recorded] == drawn[first])


class TestRefusals:
    """Each request E2 keeps refused, before any refit."""

    def test_a_weighted_plasmode(self, frame: pd.DataFrame) -> None:
        result = _fit(frame.assign(w=1.0 + frame["W1"].abs()), ATE(), weights="w")
        with pytest.raises(CapabilityError, match="weights") as caught:
            refute(result, tests=("plasmode",), n_replicates=10)
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    def test_two_observation_models_in_one_plasmode(self, frame: pd.DataFrame) -> None:
        result = _fit(
            frame, ControlledDirectEffect(intermediate=1.0), intermediate="Z", missingness="D"
        )
        with pytest.raises(CapabilityError, match="a missing outcome and an intermediate"):
            refute(result, tests=("plasmode",), n_replicates=10)
        # The held design keeps both indicators, so it runs.
        test = refute(result, tests=("simulated_outcome",), n_replicates=10, random_state=1)
        assert test["simulated_outcome"].rung == "L1"

    def test_a_cluster_intercept_outside_the_clustered_plasmode(self, frame: pd.DataFrame) -> None:
        process = GaussianAdjustmentOutcome(cluster_sd=0.5)
        clustered = _fit(frame, ATE(), cluster="C")
        with pytest.raises(CapabilityError, match="cluster_sd"):
            refute(
                clustered,
                tests=("simulated_outcome",),
                simulated_outcome=process,
                n_replicates=10,
            )
        with pytest.raises(CapabilityError, match="cluster_sd"):
            refute(_fit(frame, ATE()), tests=("plasmode",), simulated_outcome=process)

    def test_an_intermediate_effect_without_an_intermediate(self, frame: pd.DataFrame) -> None:
        with pytest.raises(CapabilityError, match="no intermediate"):
            refute(
                _fit(frame, ATE()),
                tests=("simulated_outcome",),
                simulated_outcome=GaussianAdjustmentOutcome(intermediate_effect=0.5),
                n_replicates=10,
            )

    def test_point_terms_on_a_longitudinal_panel(self) -> None:
        from tests.unit.test_longitudinal_generated_outcomes import CONTRAST, _panel
        from tests.unit.test_longitudinal_generated_outcomes import _fit as _panel_fit

        with pytest.raises(CapabilityError, match="intermediate_effect and cluster_sd"):
            refute(
                _panel_fit(_panel()),
                estimand=CONTRAST,
                tests=("simulated_outcome",),
                simulated_outcome=GaussianAdjustmentOutcome(cluster_sd=0.5),
                n_replicates=10,
            )

    def test_invalid_declarations(self) -> None:
        with pytest.raises(ValueError, match="cluster_sd"):
            GaussianAdjustmentOutcome(cluster_sd=-1.0)
        with pytest.raises(ValueError, match="intermediate_effect"):
            BernoulliAdjustmentOutcome(intermediate_effect=float("nan"))
        assert not GaussianAdjustmentOutcome(
            effect=0.0, adjustment_scale=0.0, intermediate_effect=0.3
        ).independent
