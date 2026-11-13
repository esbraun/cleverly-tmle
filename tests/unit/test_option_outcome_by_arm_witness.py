"""Exact-law witnesses of ``option-outcome-by-arm`` (X76 part (b), Fix-8 plan section 4).

``outcome_by_arm=True`` fits one outcome regression per arm on the arm's rows and the
covariates alone.  On an exact finite law with an arm-by-covariate interaction, a per-arm
logistic GLM is correct and a pooled main-terms GLM is not.  With an intercept-only mechanism
only the outcome regression can remove the confounding, so

* the per-arm fit returns the truth to ``1e-10``;
* the pooled fit misses it by more than ``1e-3``, by exactly the amount that
  :func:`tests.studies._option_by_arm_law.targeted_limit` predicts at the pooled GLM's limit;
* with the correct mechanism both fits return the truth, which is double robustness, and the
  reason the witness needs the wrong mechanism.

The mutation controls ignore the option inside the nuisance fit, and read each row's prediction
from the other arm's model.  Each makes the per-arm witness fail.

The exact law draws ``W1`` and ``W2`` as independent Bernoulli(1/2) variables, with
``logit g_0 = log 3 W1``, ``logit Q_0(1, W) = log 3 (W1 + W2 - 1)`` and
``logit Q_0(0, W) = log 3 (W2 - W1)``: every probability is a quarter, a half or three
quarters, and each ``(W1, W2)`` cell holds ``16 M`` rows, so ``P_n`` equals ``P_0``.  The
mechanism is not symmetric in ``W1`` and ``W2``: on the symmetric law ``logit g_0 = log 3 (W1 -
W2)`` the pooled fit's two arm errors cancel, and the witness would read zero.

The support tests read the rows: each study setting lands on its ``T53`` row and reads
supported, the same fit without the option lands on ``T01``, and without ``T53`` in the table
the per-arm fit matches no row.  On the effect-modification axis the reversal holds, so the
measured projection fit is the per-arm one (``T49``) and a pooled projection matches no row.
No preflight condition remains for the option; the key decides.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cleverly.estimators import TMLE
from tests.studies import _option_by_arm_law as law

nuisance_module = importlib.import_module("cleverly.estimators._nuisance")
tmle_module = importlib.import_module("cleverly.estimators.tmle")
support_module = importlib.import_module("cleverly._support")

#: Rows per ``(W1, W2)`` cell are ``16 M``.
M = 2
FLOOR = 1e-3
EXACT = 1e-10
LOG3 = float(np.log(3.0))


def g_true(w1: Any, w2: Any) -> np.ndarray:
    """``P(A = 1 | W)``: it reads ``W1`` alone, and ``w2`` only sets the shape."""
    return np.asarray(
        expit(LOG3 * np.asarray(w1, dtype=float)) * np.ones_like(np.asarray(w2, dtype=float))
    )


def q_true(a: float, w1: Any, w2: Any) -> np.ndarray:
    w1, w2 = np.asarray(w1, dtype=float), np.asarray(w2, dtype=float)
    eta = LOG3 * (w1 + w2 - 1.0) if a == 1.0 else LOG3 * (w2 - w1)
    return np.asarray(expit(eta), dtype=float)


def exact_frame() -> pd.DataFrame:
    """The exact law, ``16 M`` rows per covariate cell."""
    rows: list[tuple[float, ...]] = []
    for w1 in (0.0, 1.0):
        for w2 in (0.0, 1.0):
            total = 16 * M
            treated = round(total * float(g_true(w1, w2)))
            for a, count in ((1.0, treated), (0.0, total - treated)):
                events = round(count * float(q_true(a, w1, w2)))
                rows += [(1.0, a, w1, w2)] * events + [(0.0, a, w1, w2)] * (count - events)
    return pd.DataFrame(rows, columns=["Y", "A", "W1", "W2"])


def points() -> law.Points:
    cells = np.array([(w1, w2) for w1 in (0, 1) for w2 in (0, 1)], float)
    return law.Points(cells[:, 0], cells[:, 1], np.full(4, 0.25))


def truth() -> dict[str, float]:
    at = points()
    one, zero = at.mean(q_true(1.0, at.w1, at.w2)), at.mean(q_true(0.0, at.w1, at.w2))
    return {"ey1": one, "ey0": zero, "ate": one - zero}


def fit(frame: pd.DataFrame, *, by_arm: bool, mechanism: str = "intercept") -> Any:
    """A fit with the study's own Newton logistic learners."""
    estimator = TMLE(
        outcome_learner=law.Logistic(),
        treatment_learner=law.Logistic(columns=() if mechanism == "intercept" else None),
        outcome_by_arm=by_arm,
        cross_fit=False,
        estimands=("ey1", "ey0", "ate"),
        simultaneous=False,
        g_bounds=law.G_BOUNDS,
        max_iter=law.MAX_ITER,
        tol=law.TOL,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"]).single()


def error(result: Any, name: str = "ate") -> float:
    return float(result[name].psi) - truth()[name]


def pooled_limit() -> law.TargetedLimit:
    """The targeting limit of the pooled main-terms GLM with the intercept-only mechanism."""
    at = points()
    g = g_true(at.w1, at.w2)
    design = np.vstack([np.column_stack([np.full(4, a), at.w1, at.w2]) for a in (1.0, 0.0)])
    target = np.concatenate([q_true(1.0, at.w1, at.w2), q_true(0.0, at.w1, at.w2)])
    weight = np.concatenate([at.weight * g, at.weight * (1.0 - g)])
    beta = law.newton_logistic(design, target, weight)

    def initial(a: float) -> np.ndarray:
        return np.asarray(expit(beta[0] + beta[1] * a + beta[2] * at.w1 + beta[3] * at.w2))

    return law.targeted_limit(
        at,
        g_true=g,
        g_fit=np.full(4, at.mean(g)),
        q0=lambda a: q_true(a, at.w1, at.w2),
        initial=initial,
    )


def check_by_arm_witness() -> None:
    result = fit(exact_frame(), by_arm=True)
    for name in ("ey1", "ey0", "ate"):
        assert abs(error(result, name)) < EXACT


class TestTheExactLaw:
    def test_the_empirical_law_is_the_law(self) -> None:
        frame = exact_frame()
        assert len(frame) == 64 * M
        for (w1, w2), value in frame.groupby(["W1", "W2"])["A"].mean().items():
            assert value == pytest.approx(float(g_true(w1, w2)), abs=1e-15)
        for (a, w1, w2), value in frame.groupby(["A", "W1", "W2"])["Y"].mean().items():
            assert value == pytest.approx(float(q_true(a, w1, w2)), abs=1e-15)

    def test_the_per_arm_glm_is_correct_and_the_pooled_glm_is_not(self) -> None:
        frame = exact_frame()
        for a, expected in ((1.0, [-LOG3, LOG3, LOG3]), (0.0, [0.0, -LOG3, LOG3])):
            rows = frame["A"] == a
            model = law.Logistic().fit(frame.loc[rows, ["W1", "W2"]], frame.loc[rows, "Y"])
            assert model.beta_ == pytest.approx(expected, abs=1e-9)
        pooled = law.Logistic().fit(frame[["A", "W1", "W2"]], frame["Y"])
        fitted = pooled.predict_proba(frame[["A", "W1", "W2"]])[:, 1]
        exact = np.where(
            frame["A"] == 1.0,
            q_true(1.0, frame["W1"], frame["W2"]),
            q_true(0.0, frame["W1"], frame["W2"]),
        )
        assert np.max(np.abs(fitted - exact)) > 0.05

    def test_the_intercept_mechanism_is_wrong(self) -> None:
        at = points()
        g = g_true(at.w1, at.w2)
        assert np.max(np.abs(g - at.mean(g))) == pytest.approx(0.125)


class TestTheOutcomeWitness:
    def test_the_per_arm_fit_returns_the_truth_under_a_wrong_mechanism(self) -> None:
        check_by_arm_witness()

    def test_the_pooled_fit_misses_by_the_predicted_remainder(self) -> None:
        result = fit(exact_frame(), by_arm=False)
        expected = pooled_limit()
        reference = truth()
        assert abs(error(result) - (expected.ate - reference["ate"])) < EXACT
        assert abs(error(result, "ey1") - (expected.means[1.0] - reference["ey1"])) < EXACT
        assert abs(error(result, "ey0") - (expected.means[0.0] - reference["ey0"])) < EXACT
        assert abs(error(result)) > FLOOR

    def test_the_correct_mechanism_rescues_the_pooled_fit(self) -> None:
        frame = exact_frame()
        for by_arm in (True, False):
            assert abs(error(fit(frame, by_arm=by_arm, mechanism="glm"))) < EXACT

    def test_the_per_arm_regressions_are_the_targeting_start(self) -> None:
        frame = exact_frame()
        result = fit(frame, by_arm=True)
        for a in (1.0, 0.0):
            initial = np.asarray(result.nuisance.outcome.arms[a], dtype=float)
            assert initial == pytest.approx(q_true(a, frame["W1"], frame["W2"]), abs=1e-12)

    def test_mutation_ignoring_the_option_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = tmle_module.fit_nuisances

        def pooled(*args: Any, **kwargs: Any) -> Any:
            return original(*args, **{**kwargs, "outcome_by_arm": False})

        monkeypatch.setattr(tmle_module, "fit_nuisances", pooled)
        with pytest.raises(AssertionError):
            check_by_arm_witness()

    def test_mutation_reading_the_other_arm_s_model_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = nuisance_module._gather_by_arm

        def swapped(data: Any, arm: Any, table: Any) -> Any:
            return original(data, 1.0 - np.asarray(arm, dtype=float), table)

        monkeypatch.setattr(nuisance_module, "_gather_by_arm", swapped)
        with pytest.raises(AssertionError):
            check_by_arm_witness()


class TestTheStudyFits:
    def test_each_arm_learner_reads_the_arm_s_rows_and_the_covariates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, int, float]] = []
        original = law.Logistic.fit

        def spy(self: Any, design: Any, target: Any, sample_weight: Any = None) -> Any:
            values = np.asarray(design)
            calls.append((values.shape[0], values.shape[1], float(np.mean(target))))
            return original(self, design, target, sample_weight)

        monkeypatch.setattr(law.Logistic, "fit", spy)
        frame = law.draw("LB", 400, 0)
        law.fit(frame, "binary_in_sample")
        treated = int(frame["A"].sum())
        outcome_calls = sorted(c for c in calls if c[1] == 2 and c[0] < len(frame))
        assert outcome_calls == sorted(
            [
                (treated, 2, float(frame.loc[frame["A"] == 1, "Y"].mean())),
                (len(frame) - treated, 2, float(frame.loc[frame["A"] == 0, "Y"].mean())),
            ]
        )
        calls.clear()
        law.fit(frame, "pooled_binary_in_sample")
        assert sorted(c[:2] for c in calls) == [(len(frame), 2), (len(frame), 3)]

    def test_the_bounded_cell_declares_its_support(self) -> None:
        frame = law.draw("LB-b", 400, 0)
        assert frame["Y"].between(0.0, 1.0, inclusive="neither").all()
        assert law.estimator(law.SETTINGS["bounded_stacked"]).q_bounds == law.Q_BOUNDS
        assert law.estimator(law.SETTINGS["continuous_in_sample"]).q_bounds is None


def _without_rows(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """Drop every registered row whose id starts with ``prefix`` from the support table."""
    rows = support_module.registered_rows()
    monkeypatch.setattr(
        support_module,
        "registered_rows",
        lambda: tuple(row for row in rows if not row.id.startswith(prefix)),
    )


class TestTheSupportRow:
    @pytest.mark.parametrize(
        ("setting", "row"),
        [
            ("binary_in_sample", "T53.in_sample.binary"),
            ("continuous_in_sample", "T53.in_sample.continuous"),
            ("binary_stacked", "T53.stacked.binary"),
            ("bounded_stacked", "T53.stacked.bounded"),
        ],
    )
    def test_a_per_arm_fit_reads_its_t53_row(self, setting: str, row: str) -> None:
        frame = law.draw(law.SETTINGS[setting].law, 400, 0)
        for estimate in law.fit(frame, setting).estimates.values():
            assert (estimate.support.label, estimate.support.row) == ("supported", row)
            assert any("option-outcome-by-arm" in limit for limit in estimate.support.limits)

    def test_the_same_fit_without_the_option_reads_t01(self) -> None:
        frame = law.draw("LB", 400, 0)
        support = law.fit(frame, "pooled_binary_in_sample")["ate"].support
        assert (support.label, support.row) == ("supported", "T01.binary")

    def test_without_t53_a_per_arm_fit_matches_no_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key, not a fit condition, routes the fit: without ``T53`` it is experimental."""
        _without_rows(monkeypatch, "T53")
        support = law.fit(law.draw("LB", 400, 0), "binary_in_sample")["ate"].support
        assert (support.label, support.row) == ("experimental", None)
        assert "outcome_by_arm=True" in support.reason

    def test_the_effect_modification_axis_measures_the_per_arm_fit(self) -> None:
        """The reversal: ``T49`` is the per-arm projection, and a pooled one matches no row."""
        from cleverly import CausalStudy
        from tests.studies import conditional_projection as projection

        group = projection.GROUP_BY_LABEL["obs_4k"]
        frame = projection.draw_from_seed("obs_4k", 800, 0)[0]
        effect = CausalStudy(frame, design=projection.design(group)).identify(
            projection.estimand(group)
        )
        settings = {**projection.learners(group), "random_state": 0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            by_arm = effect.estimate(**settings)
            pooled = effect.estimate(**{**settings, "outcome_by_arm": False})
        assert settings["outcome_by_arm"] is True
        assert {e.support.row for e in by_arm.estimates.values()} == {"T49"}
        for estimate in pooled.estimates.values():
            assert estimate.support.label == "experimental"
            assert estimate.support.row is None
            assert "No registered study covers this configuration" in estimate.support.reason
            assert "pools the arms" not in estimate.support.reason
