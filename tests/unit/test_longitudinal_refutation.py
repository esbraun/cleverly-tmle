"""The longitudinal refutation adapter (X82 part C1).

The placebo draws each treatment trajectory from a covariate-free first-order Markov kernel
and masks it by the observed at-risk pattern. The exact witness enumerates the placebo law
of a discrete two-node survival panel and finds every policy contrast 0 to rounding. Its
mutation control fits the kernel within each follow-up pattern, which makes the placebo
depend on the event process, and the contrast moves. The other tests pin the refit seam:
the stored template, the frozen folds and seeds, the noise columns, and the refusals.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal
from cleverly.exceptions import CapabilityError, DataError
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.estimator import ParametricGFormula
from cleverly.msm import MSM
from cleverly.validation import refute
from cleverly.validation._markov_placebo import MarkovKernel, placebo_known_mechanisms
from cleverly.validation.refute import (
    _longitudinal_msm_reference,
    _refuse_placebo_kernel,
    _RefuteRequest,
    refute_refusal,
)

CONTRAST = "ate_regimen[never vs always]"
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _ltmle(**settings: Any) -> LTMLE:
    options: dict[str, Any] = {
        "n_folds": 2,
        "random_state": 3,
        "simultaneous": False,
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
    }
    options.update(settings)
    return LTMLE({"always": 1, "never": 0}, **options)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_longitudinal(n=240, seed=4)[0]


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> Any:
    return _ltmle().fit(frame, **COLUMNS)


# ------------------------------------------------------------------- the exact witness


def _survival_panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A discrete two-node survival panel whose treatment is confounded by ``W`` and ``L2``."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    a1 = rng.binomial(1, 0.3 + 0.4 * w).astype(float)
    c1 = rng.binomial(1, 0.9 - 0.1 * a1).astype(float)
    y1 = np.where(c1 == 1, rng.binomial(1, 0.1 + 0.2 * w + 0.15 * a1), np.nan)
    risk = (c1 == 1) & (y1 == 0)
    l2 = np.where(risk, rng.binomial(1, 0.3 + 0.3 * a1), np.nan)
    a2 = np.where(risk, rng.binomial(1, 0.2 + 0.5 * np.nan_to_num(l2)), np.nan)
    c2 = np.where(risk, rng.binomial(1, 0.9), np.nan)
    stays = risk & (c2 == 1)
    y2 = np.where(
        stays, rng.binomial(1, 0.1 + 0.2 * np.nan_to_num(l2) + 0.2 * np.nan_to_num(a2)), np.nan
    )
    y2 = np.where(y1 == 1, 1.0, y2)
    return pd.DataFrame(
        {"W": w, "A1": a1, "C1": c1, "Y1": y1, "L2": l2, "A2": a2, "C2": c2, "Y2": y2}
    )


def _survival_data(frame: pd.DataFrame) -> LongitudinalData:
    return LongitudinalData.from_frame(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _risk_under(
    frame: pd.DataFrame, laws: list[dict[tuple[int, ...], float]], regimen: tuple[int, int]
) -> float:
    """The two-node risk of ``regimen`` under the enumerated law of ``frame``.

    Each row keeps its non-treatment variables and carries every unmasked trajectory with
    the probability its law gives. The iterated conditional expectations then run on exact
    weighted cell means, with no learner, so the value is the law's own g-formula.
    """
    rows = []
    for index, unit in enumerate(frame.itertuples(index=False)):
        for codes, probability in laws[index].items():
            if probability == 0.0:
                continue
            a2 = float(codes[1]) if not np.isnan(unit.A2) else np.nan
            rows.append(
                (
                    unit.W,
                    float(codes[0]),
                    unit.C1,
                    unit.Y1,
                    unit.L2,
                    a2,
                    unit.C2,
                    unit.Y2,
                    probability,
                )
            )
    data = pd.DataFrame(rows, columns=["W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2", "p"])

    def mean(values: pd.Series, weights: pd.Series) -> float:
        return float(np.sum(values * weights) / np.sum(weights))

    stays = data[(data.C1 == 1) & (data.Y1 == 0) & (data.C2 == 1)]
    second: dict[tuple[float, float, float, float], float] = {}
    for key, cell in stays.groupby(["W", "A1", "L2", "A2"]):
        second[key] = mean(cell.Y2, cell.p)
    observed1 = data[data.C1 == 1]
    first: dict[float, float] = {}
    for w_value, cell in observed1[regimen[0] == observed1.A1].groupby("W"):
        pseudo = np.where(
            cell.Y1 == 1,
            1.0,
            [
                second.get((w_value, float(regimen[0]), l2, float(regimen[1])), np.nan)
                if y1 == 0
                else 0.0
                for l2, y1 in zip(cell.L2, cell.Y1, strict=True)
            ],
        )
        # A unit at risk at node 2 that was censored there reads the regression too.
        first[w_value] = mean(pd.Series(pseudo, index=cell.index), cell.p)
    marginal = data.groupby("W").p.sum() / data.p.sum()
    return float(sum(marginal[w] * first[w] for w in marginal.index))


def _laws(kernel: MarkovKernel, n: int) -> list[dict[tuple[int, ...], float]]:
    law = dict(kernel.trajectory_law())
    return [law] * n


class TestPlaceboLaw:
    def test_every_policy_contrast_is_exactly_zero(self) -> None:
        frame = _survival_panel()
        kernel = MarkovKernel.fit(_survival_data(frame))
        laws = _laws(kernel, len(frame))
        always = _risk_under(frame, laws, (1, 1))
        never = _risk_under(frame, laws, (0, 0))
        mixed = _risk_under(frame, laws, (1, 0))
        assert abs(always - never) < 1e-12
        assert abs(always - mixed) < 1e-12

    def test_a_kernel_fitted_within_follow_up_patterns_moves_the_contrast(self) -> None:
        # Mutation control: the law of the drawn treatment now depends on whether the unit
        # stayed at risk, a function of the event process, and the null is lost.
        frame = _survival_panel()
        data = _survival_data(frame)
        at_risk = ~np.isnan(np.asarray(data.treatment)[:, 1])
        laws: list[dict[tuple[int, ...], float]] = [{} for _ in range(len(frame))]
        for pattern in (True, False):
            rows = np.flatnonzero(at_risk == pattern)
            kernel = MarkovKernel.fit(data.subset(rows))
            if pattern:
                law = dict(kernel.trajectory_law())
            else:
                # No unit of this pattern holds node 2, so the second code is drawn from the
                # transition of the full panel. It is masked for every row here anyway.
                first = kernel.node_law(1, np.zeros(1))[0]
                second = MarkovKernel.fit(data).node_law(2, np.asarray([0.0, 1.0]))
                law = {(a, b): first[a] * second[a, b] for a in (0, 1) for b in (0, 1)}
            for row in rows:
                laws[row] = law
        always = _risk_under(frame, laws, (1, 1))
        never = _risk_under(frame, laws, (0, 0))
        assert abs(always - never) > 1e-3

    def test_the_draws_follow_the_trajectory_law(self) -> None:
        kernel = MarkovKernel.fit(_survival_data(_survival_panel()))
        drawn = kernel.draw(np.random.default_rng(1), 200_000)
        for codes, probability in kernel.trajectory_law():
            share = float(np.mean(np.all(drawn == np.asarray(codes, dtype=float), axis=1)))
            assert share == pytest.approx(probability, abs=5e-3)

    def test_the_transition_is_the_empirical_one(self) -> None:
        data = _survival_data(_survival_panel())
        kernel = MarkovKernel.fit(data)
        treatment = np.asarray(data.treatment, dtype=float)
        both = ~np.isnan(treatment).any(axis=1)
        for previous in (0.0, 1.0):
            donors = both & (treatment[:, 0] == previous)
            expected = np.mean(treatment[donors, 1] == 1.0)
            assert kernel.node_law(2, np.asarray([previous]))[0, 1] == pytest.approx(expected)

    def test_the_mask_is_the_observed_at_risk_pattern(self) -> None:
        data = _survival_data(_survival_panel())
        placebo = MarkovKernel.fit(data).placebo(data, np.random.default_rng(2))
        assert np.array_equal(np.isnan(placebo), np.isnan(np.asarray(data.treatment)))
        replaced = data.with_treatment(placebo)
        assert np.array_equal(
            np.asarray(replaced.outcome), np.asarray(data.outcome), equal_nan=True
        )

    def test_a_permuted_trajectory_is_refused(self) -> None:
        # A whole-trajectory permutation gives a unit at risk a donor's missing treatment.
        data = _survival_data(_survival_panel())
        treatment = np.asarray(data.treatment, dtype=float)
        permuted = treatment[np.random.default_rng(0).permutation(data.n)]
        assert not np.array_equal(np.isnan(permuted), np.isnan(treatment))
        with pytest.raises(DataError, match="not at risk"):
            data.with_treatment(permuted)

    def test_a_dose_is_drawn_from_the_observed_doses(self) -> None:
        rng = np.random.default_rng(5)
        n = 200
        frame = pd.DataFrame(
            {
                "W": rng.normal(size=n),
                "A1": rng.choice([0.5, 1.0, 2.0], size=n),
                "A2": rng.choice([0.0, 1.5, 3.0], size=n),
                "Y": rng.normal(size=n),
            }
        )
        data = LongitudinalData.from_frame(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], []],
            continuous_treatment=["A1", "A2"],
        )
        drawn = MarkovKernel.fit(data).draw(np.random.default_rng(0), 500)
        assert set(np.unique(drawn[:, 0])) <= {0.5, 1.0, 2.0}
        assert set(np.unique(drawn[:, 1])) <= {0.0, 1.5, 3.0}

    def test_a_held_design_repeats_the_first_decision(self) -> None:
        frame = make_longitudinal(n=120, seed=1)[0]
        data = LongitudinalData.from_frame(
            frame, outcome="Y", treatment="A1", baseline=["W1", "W2"], censoring=["C1", "C2"]
        )
        drawn = MarkovKernel.fit(data).draw(np.random.default_rng(0), 300)
        assert np.array_equal(drawn[:, 0], drawn[:, 1])

    def test_a_cluster_constant_treatment_draws_one_trajectory_per_cluster(self) -> None:
        rng = np.random.default_rng(3)
        clusters = np.repeat(np.arange(30), 4)
        a1 = rng.integers(0, 2, 30)[clusters].astype(float)
        a2 = rng.integers(0, 2, 30)[clusters].astype(float)
        frame = pd.DataFrame(
            {
                "id": clusters,
                "W": rng.normal(size=clusters.size),
                "A1": a1,
                "A2": a2,
                "Y": rng.normal(size=clusters.size),
            }
        )
        data = LongitudinalData.from_frame(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], []],
            id="id",
        )
        kernel = MarkovKernel.fit(data)
        assert kernel.cluster_codes is not None
        placebo = kernel.placebo(data, np.random.default_rng(0))
        for cluster in range(30):
            rows = placebo[clusters == cluster]
            assert np.all(rows == rows[0])

    def test_a_known_node_takes_the_kernel_law(self) -> None:
        frame = make_longitudinal(n=200, seed=2)[0]
        probabilities = np.full((len(frame), 2), 0.5)
        data = LongitudinalData.from_frame(frame, **COLUMNS, treatment_probabilities=probabilities)
        kernel = MarkovKernel.fit(data)
        placebo = kernel.placebo(data, np.random.default_rng(0))
        known = placebo_known_mechanisms(data, kernel, placebo)
        assert known is not None
        node2 = np.asarray(known.treatment_at(2))
        risk = ~np.isnan(placebo[:, 1])
        expected = kernel.node_law(2, placebo[risk, 0])
        assert np.allclose(node2[risk], expected)
        assert np.all(np.isnan(node2[~risk]))


# --------------------------------------------------------------------- the refit seam


class _Spy:
    """Record every refit's data and options, and run the real one."""

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def refit(self, data: Any, **options: Any) -> Any:
        self.calls.append((data, options))
        return self._estimator.refit(data, **options)


def _spied(result: Any) -> tuple[Any, _Spy]:
    from dataclasses import replace

    spy = _Spy(result.estimator)
    return replace(result, estimator=spy), spy


class TestRefit:
    def test_the_stored_template_reproduces_the_fit(self, fitted: Any) -> None:
        seeds = [repeat.seed for repeat in fitted.repeats]
        again = fitted.estimator.refit(fitted.data, frozen_seeds=seeds)
        for name in fitted:
            assert again[name].psi == fitted[name].psi

    def test_frozen_folds_reproduce_the_fit(self, fitted: Any) -> None:
        folds = [repeat.folds for repeat in fitted.repeats]
        seeds = [repeat.seed for repeat in fitted.repeats]
        again = fitted.estimator.refit(fitted.data, frozen_folds=folds, frozen_seeds=seeds)
        assert again[CONTRAST].psi == fitted[CONTRAST].psi

    def test_a_refit_drops_the_bootstrap_and_the_band(self, frame: pd.DataFrame) -> None:
        result = _ltmle(n_bootstrap=20, simultaneous=True).fit(frame, **COLUMNS)
        assert result.estimator.n_bootstrap == 0
        assert result.estimator.simultaneous is False

    def test_a_saved_result_keeps_its_template(self, fitted: Any) -> None:
        from cleverly.estimators.serialize import dumps, loads

        restored = loads(dumps(fitted))
        seeds = [repeat.seed for repeat in fitted.repeats]
        again = restored.estimator.refit(restored.data, frozen_seeds=seeds)
        assert again[CONTRAST].psi == fitted[CONTRAST].psi


class TestAdapter:
    def test_every_test_runs_and_reads_descriptive(self, fitted: Any) -> None:
        report = refute(
            fitted,
            estimand=CONTRAST,
            tests=("placebo", "random_common_cause", "subset", "bootstrap_stability"),
            n_replicates=3,
            random_state=0,
        )
        for name in ("placebo", "random_common_cause", "subset", "bootstrap_stability"):
            assert report[name].passed is None
            assert len(report[name].values) == 3

    def test_the_placebo_refits_on_masked_kernel_draws(self, fitted: Any) -> None:
        result, spy = _spied(fitted)
        refute(result, estimand=CONTRAST, tests=("placebo",), n_replicates=2, random_state=0)
        observed = np.isnan(np.asarray(fitted.data.treatment))
        for data, _ in spy.calls:
            assert np.array_equal(np.isnan(np.asarray(data.treatment)), observed)
            assert not np.array_equal(
                np.nan_to_num(np.asarray(data.treatment)),
                np.nan_to_num(np.asarray(fitted.data.treatment)),
            )

    def test_noise_refits_reuse_the_original_seeds_and_folds(self, fitted: Any) -> None:
        result, spy = _spied(fitted)
        refute(
            result,
            estimand=CONTRAST,
            tests=("random_common_cause",),
            n_replicates=2,
            random_state=0,
            noise_nodes=True,
        )
        seeds = [repeat.seed for repeat in fitted.repeats]
        for data, options in spy.calls:
            assert list(options["frozen_seeds"]) == seeds
            assert data.baseline_names[-1].startswith("_noise_")
            for time, names in enumerate(data.time_varying_names, start=1):
                assert names[-1].endswith(f"_node_{time}")

    def test_subset_refits_keep_the_original_fold_labels(self, fitted: Any) -> None:
        result, spy = _spied(fitted)
        refute(result, estimand=CONTRAST, tests=("subset",), n_replicates=2, random_state=0)
        for data, options in spy.calls:
            (folds,) = options["frozen_folds"]
            assert folds.n == data.n

    def test_a_regimen_mean_refuses_the_placebo(self, fitted: Any) -> None:
        refusal = refute_refusal(fitted, estimand="ey_regimen[always]", tests=("placebo",))
        assert refusal is not None
        assert "fixed no-effect value" in refusal.message()

    def test_a_pending_test_is_refused_with_its_roadmap_item(self, fitted: Any) -> None:
        with pytest.raises(CapabilityError) as error:
            refute(fitted, estimand=CONTRAST, tests=("dummy_outcome",))
        assert error.value.refusal.roadmap == "X82"

    def test_a_point_draw_is_refused_on_a_panel(self, fitted: Any) -> None:
        refusal = refute_refusal(
            fitted, estimand=CONTRAST, tests=("placebo",), placebo_mode="permute"
        )
        assert refusal is not None
        assert "Markov kernel" in refusal.message()
        assert (
            refute_refusal(fitted, estimand=CONTRAST, tests=("placebo",), placebo_mode="markov")
            is None
        )

    def test_a_mixed_node_refuses_the_placebo(self) -> None:
        stand_in = SimpleNamespace(
            assessment_family="longitudinal", data=SimpleNamespace(has_mixed_node=True)
        )
        request = _RefuteRequest(estimand=CONTRAST, tests=("placebo",))
        refusal = _refuse_placebo_kernel(stand_in, request)  # type: ignore[arg-type]
        assert refusal is not None
        assert refusal.roadmap == "X82"
        nothing = SimpleNamespace(
            assessment_family="longitudinal", data=SimpleNamespace(has_mixed_node=False)
        )
        assert _refuse_placebo_kernel(nothing, request) is None  # type: ignore[arg-type]

    def test_noise_nodes_is_refused_on_the_parametric_g_formula(self, frame: pd.DataFrame) -> None:
        result: Any = ParametricGFormula(
            {"always": 1, "never": 0},
            covariate_types={"L2": "normal"},
            random_state=0,
            simultaneous=False,
        ).fit(frame, **COLUMNS)
        refusal = refute_refusal(
            result, estimand=CONTRAST, tests=("random_common_cause",), noise_nodes=True
        )
        assert refusal is not None
        assert "forward simulation" in refusal.message()

    def test_the_facade_row_runs(self, fitted: Any) -> None:
        row = fitted.diagnostics._capability_for_arguments("refute", {"estimand": CONTRAST})
        assert row.available


class TestWorkingModelReference:
    @staticmethod
    def _fit(frame: pd.DataFrame) -> Any:
        def design(label: Any, horizon: int, rows: Any) -> np.ndarray:
            del horizon
            n = len(rows)
            return np.column_stack([np.ones(n), np.full(n, 1.0 if label == "always" else 0.0)])

        known: Any = design
        model = MSM(design=known, terms=("(intercept)", "always"), design_kind="known")
        return _ltmle(msm=model, n_folds=1).fit(frame, **COLUMNS)

    def test_a_contrast_coefficient_has_the_exact_null(self, frame: pd.DataFrame) -> None:
        result = self._fit(frame)
        reference = _longitudinal_msm_reference(result, "msm_regimen[always]")
        assert reference is not None
        assert reference.exact
        assert reference.value == 0.0

    def test_the_intercept_has_no_exact_null(self, frame: pd.DataFrame) -> None:
        # The witness that the operator check reads the coefficient's row: the intercept's
        # row is the regimen mean and never vanishes.
        result = self._fit(frame)
        reference = _longitudinal_msm_reference(result, "msm_regimen[(intercept)]")
        assert reference is not None
        assert not reference.exact
        assert np.isnan(reference.value)


def test_the_kernel_reads_no_covariate() -> None:
    """Two panels that differ only in their covariates give one kernel."""
    frame = _survival_panel()
    other = frame.assign(W=1.0 - frame.W, L2=1.0 - frame.L2)
    first = MarkovKernel.fit(_survival_data(frame))
    second = MarkovKernel.fit(_survival_data(other))
    assert np.array_equal(first.trajectories, second.trajectories, equal_nan=True)
    counts: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for codes, probability in first.trajectory_law():
        counts[codes] += probability
    assert sum(counts.values()) == pytest.approx(1.0)


class TestPlaceboPositivity:
    @staticmethod
    def _stand_in(blocked: bool) -> tuple[Any, MarkovKernel]:
        from cleverly.longitudinal.regimen import Regimen

        rng = np.random.default_rng(7)
        n = 200
        a1 = rng.integers(0, 2, n).astype(float)
        a2 = rng.integers(0, 2, n).astype(float)
        if blocked:
            # No unit treated at node 1 stays treated at node 2.
            a2 = np.where(a1 == 1.0, 0.0, a2)
        frame = pd.DataFrame({"W": rng.normal(size=n), "A1": a1, "A2": a2, "Y": rng.normal(size=n)})
        data = LongitudinalData.from_frame(
            frame, outcome="Y", treatment=["A1", "A2"], baseline=["W"], time_varying=[[], []]
        )
        regimens = (Regimen("always", (1, 1)), Regimen("never", (0, 0)))
        stand_in = SimpleNamespace(data=data, config=SimpleNamespace(regimens=regimens))
        return stand_in, MarkovKernel.fit(data)

    def test_a_regimen_the_kernel_never_draws_is_named(self) -> None:
        from cleverly.validation.refute import _placebo_positivity_gap

        stand_in, kernel = self._stand_in(blocked=True)
        assert kernel.trajectory_probability((1, 1)) == 0.0
        gap = _placebo_positivity_gap(stand_in, kernel)
        assert gap is not None
        assert "'always'" in gap

    def test_a_panel_with_every_transition_has_no_gap(self) -> None:
        from cleverly.validation.refute import _placebo_positivity_gap

        stand_in, kernel = self._stand_in(blocked=False)
        assert _placebo_positivity_gap(stand_in, kernel) is None

    def test_the_trajectory_probability_is_the_law(self) -> None:
        _, kernel = self._stand_in(blocked=False)
        for codes, probability in kernel.trajectory_law():
            assert kernel.trajectory_probability(codes) == probability


def test_each_noise_column_joins_its_own_node_block(fitted: Any) -> None:
    """A node's noise column sits in that node's block, measured before its treatment."""
    data = fitted.data
    values = np.arange(data.n * data.n_times, dtype=float).reshape(data.n, data.n_times)
    names = [f"_noise_node_{time}" for time in range(1, data.n_times + 1)]
    noisy = data.with_node_covariates(values, names)
    for time, block in enumerate(noisy.time_varying_names, start=1):
        assert block[-1] == names[time - 1]
        assert block[:-1] == data.time_varying_names[time - 1]
