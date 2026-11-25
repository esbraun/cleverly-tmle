"""The outcome screen of ``screen_treatment=True`` (Fix-20).

The screen keeps the treatment-model covariates that predict the outcome with the
treatment held fixed.  Each witness below has a deliberate-mutation control: the same
check run on a screen with one component removed (partialling on the treatment, the
interaction columns, the spline basis, the floor), which must fail.  The laws were chosen
by a 200-seed sweep (Fix-20 row files): every witness held on all 200 seeds and every
mutation failed on all 200, so the three seeds used here are not hand-picked survivors.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.learners.screeners as screeners
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import outcome_screen
from cleverly.interventions import Shift
from cleverly.learners import screen_by_outcome
from cleverly.learners.screeners import OutcomeScreen, ScreenedLearner, default_min_retain

SEEDS = (0, 1, 2)
C, Z, P, N = 0, 1, 2, 3


def _order(
    w: np.ndarray, t: np.ndarray, y: np.ndarray, *, discrete: bool, mutation: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The lasso-min support and the entry order, with one component optionally removed."""
    rows = np.arange(y.size)
    partial, linear = screeners._TreatmentBasis(t[:, None], (discrete,)).build(rows)
    target = y
    if mutation == "marginal":
        partial, linear = np.ones((y.size, 1)), np.zeros((y.size, 0))
    elif mutation == "on_treatment":
        partial, linear, target = np.ones((y.size, 1)), np.zeros((y.size, 0)), t
    elif mutation == "additive":
        linear = np.zeros((y.size, 0))
    elif mutation == "linear_dose":
        partial = np.column_stack([np.ones(y.size), t - t.mean()])
    keep, entry, strength = screeners._lasso_screen(
        w, partial, linear, target, np.ones(y.size), rows, None
    )
    return keep, screeners._entry_order(entry, strength)


def instrument_law(seed: int, n: int = 2000) -> tuple[np.ndarray, ...]:
    """Confounder C, strong instrument Z, weak outcome-only P, noise N; large effect."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 4))
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * w[:, C] - 2.5 * w[:, Z]))).astype(float)
    y = 4.0 * a + w[:, C] + 0.3 * w[:, P] + rng.normal(size=n)
    return w, a, y


def interaction_law(seed: int, n: int = 2000) -> tuple[np.ndarray, ...]:
    """C confounds, but reaches the outcome only through ``(2A - 1) C``."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 3))
    a = (rng.uniform(size=n) < 1 / (1 + np.exp(-w[:, 0]))).astype(float)
    y = (2 * a - 1) * w[:, 0] + 0.5 * w[:, 1] + rng.normal(size=n)
    return w, a, y


def dose_law(seed: int, n: int = 2000) -> tuple[np.ndarray, ...]:
    """A continuous dose with a quadratic response; Z is an instrument of the dose."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 4))
    a = 0.5 * w[:, C] + 1.5 * w[:, Z] + rng.normal(size=n)
    y = a**2 + w[:, C] + 0.5 * w[:, P] + rng.normal(size=n)
    return w, a, y


class TestTheScreenDropsInstruments:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_outcome_predictors_enter_before_the_instrument_and_noise(self, seed: int) -> None:
        w, a, y = instrument_law(seed)
        _, order = _order(w, a, y, discrete=True)
        assert set(order[:2].tolist()) == {C, P}

    @pytest.mark.parametrize("mutation", ["marginal", "on_treatment"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_without_holding_the_treatment_fixed_the_instrument_leads(
        self, seed: int, mutation: str
    ) -> None:
        """Control: a marginal outcome screen (R's), or a screen on the treatment."""
        w, a, y = instrument_law(seed)
        _, order = _order(w, a, y, discrete=True, mutation=mutation)
        assert set(order[:2].tolist()) != {C, P}

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_confounder_acting_through_an_interaction_enters_first(self, seed: int) -> None:
        w, a, y = interaction_law(seed)
        assert _order(w, a, y, discrete=True)[1][0] == 0
        # Control: without the interaction columns the confounder does not lead.
        assert _order(w, a, y, discrete=True, mutation="additive")[1][0] != 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_nonlinear_dose_response_is_held_fixed(self, seed: int) -> None:
        w, a, y = dose_law(seed)
        assert set(_order(w, a, y, discrete=False)[1][:2].tolist()) == {C, P}
        # Control: partialling the dose linearly leaves a**2, which the instrument predicts.
        assert set(_order(w, a, y, discrete=False, mutation="linear_dose")[1][:2].tolist()) != {
            C,
            P,
        }

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_categorical_treatment_is_partialled_by_arm(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = 3000
        w = rng.normal(size=(n, 3))  # instrument, outcome-only, noise
        logits = np.column_stack([np.zeros(n), 2.0 * w[:, 0], -2.0 * w[:, 0]])
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        a = (rng.uniform(size=n)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
        y = 3.0 * (a == 2) + 0.5 * w[:, 1] + rng.normal(size=n)
        assert _order(w, a.astype(float), y, discrete=True)[1][0] == 1
        # Control: a marginal outcome screen ranks the instrument of arm 2 first.
        assert _order(w, a.astype(float), y, discrete=True, mutation="marginal")[1][0] != 1

    def test_a_covariate_orthogonal_to_the_outcome_in_each_arm_is_dropped(self) -> None:
        """An exact zero, not a ranking: its lasso coefficients vanish at every penalty."""
        w, a, y = instrument_law(4)
        span = np.column_stack([np.ones(y.size), w, y])
        z = np.random.default_rng(9).normal(size=y.size)
        for arm in (0.0, 1.0):
            rows = a == arm
            z[rows] -= span[rows] @ np.linalg.lstsq(span[rows], z[rows], rcond=None)[0]
        keep = screen_by_outcome(np.column_stack([w, z]), a, y, min_retain=1)
        assert keep[C] and keep[P] and not keep[4]


class TestTheFloor:
    def test_the_default_floor_is_r_default(self) -> None:
        assert default_min_retain(4, 2000, [1000, 1000]) == 4
        assert default_min_retain(500, 2000, [1000, 1000]) == 100
        # A rare arm: n_eff = 5 * 40 = 200, so ceil(200 / 20) = 10.
        assert default_min_retain(500, 2000, [40, 1960]) == 10
        # A dose: n_eff = n.
        assert default_min_retain(500, 2000, None) == 100
        assert default_min_retain(3, 10, [5, 5]) == 1

    def test_the_default_keeps_every_covariate_when_p_is_small(self) -> None:
        w, a, y = instrument_law(0)
        assert screen_by_outcome(w, a, y).all()

    def test_the_default_screens_once_p_exceeds_n_eff_over_20(self) -> None:
        rng = np.random.default_rng(6)
        n, p = 400, 40  # floor ceil(400 / 20) = 20 < 40
        w = rng.normal(size=(n, p))
        a = (rng.uniform(size=n) < 0.5).astype(float)
        y = a + w[:, 0] + w[:, 1] + rng.normal(size=n)
        keep = screen_by_outcome(w, a, y)
        assert keep[0] and keep[1]
        assert 20 <= keep.sum() < p

    def test_the_entry_order_follows_the_outcome_coefficients(self) -> None:
        """Orthogonal covariates with coefficients 3, 2, 1, 0 enter in that order."""
        rng = np.random.default_rng(8)
        n = 1000
        a = (rng.uniform(size=n) < 0.5).astype(float)
        raw = rng.normal(size=(n, 4))
        basis = np.column_stack([np.ones(n), a])
        w = np.linalg.qr(screeners._residualise(raw, basis))[0] * np.sqrt(n)
        y = a + w @ np.array([3.0, 2.0, 1.0, 0.0]) + 0.5 * rng.normal(size=n)
        screen = OutcomeScreen(w, a, (True,), y, np.ones(n, bool), np.ones(n), None, 4)
        assert screen.entry_order().tolist() == [0, 1, 2, 3]

    def test_the_floor_fills_in_entry_order(self) -> None:
        w, a, y = instrument_law(1)
        screen = OutcomeScreen(w, a, (True,), y, np.ones(y.size, bool), np.ones(y.size), None, 1)
        order = screen.entry_order()
        at_min = screen.support()
        for floor in range(1, 5):
            screen.min_retain = floor
            screen._cache.clear()
            keep = screen.support()
            expected = at_min.copy()
            for index in order:
                if expected.sum() >= floor:
                    break
                expected[index] = True
            np.testing.assert_array_equal(keep, expected)
            assert keep.sum() >= floor


class TestTheRowsTheScreenReads:
    def _screen(self, y: np.ndarray, eligible: np.ndarray, weights: np.ndarray) -> OutcomeScreen:
        w, a, _ = instrument_law(2, n=600)
        return OutcomeScreen(w, a, (True,), y, eligible, weights, None, 1)

    def test_a_training_set_reads_only_its_own_outcomes(self) -> None:
        _, _, y = instrument_law(2, n=600)
        rows = np.arange(400)
        moved = y.copy()
        moved[400:] = np.random.default_rng(0).normal(scale=50.0, size=200)
        ones = np.ones(600)
        np.testing.assert_array_equal(
            self._screen(y, ones > 0, ones).support(rows),
            self._screen(moved, ones > 0, ones).support(rows),
        )

    def test_unobserved_outcomes_and_zero_weights_are_not_read(self) -> None:
        _, _, y = instrument_law(2, n=600)
        hidden = np.zeros(600, dtype=bool)
        hidden[::7] = True
        moved = np.where(hidden, 1e6, y)
        ones = np.ones(600)
        reference = self._screen(y, ~hidden, ones).support()
        np.testing.assert_array_equal(self._screen(moved, ~hidden, ones).support(), reference)
        np.testing.assert_array_equal(
            self._screen(moved, ones > 0, np.where(hidden, 0.0, 1.0)).support(), reference
        )

    def test_weights_enter_the_residual_correlations(self) -> None:
        """Integer weights equal duplicated rows in the partialled, standardised design."""
        w, a, y = instrument_law(5, n=300)
        weights = np.random.default_rng(1).integers(1, 4, size=300).astype(float)
        rows = np.arange(300)
        repeat = np.repeat(rows, weights.astype(int))

        def strength(wt: np.ndarray, index: np.ndarray) -> np.ndarray:
            partial, linear = screeners._TreatmentBasis(a[:, None], (True,)).build(index)
            return screeners._lasso_screen(w[index], partial, linear, y[index], wt, index, None)[2]

        np.testing.assert_allclose(
            strength(weights, rows), strength(np.ones(repeat.size), repeat), atol=1e-10
        )

    def test_the_screen_is_deterministic(self) -> None:
        w, a, y = instrument_law(7)
        first = screen_by_outcome(w, a, y, min_retain=1)
        np.testing.assert_array_equal(first, screen_by_outcome(w, a, y, min_retain=1))

    def test_a_rank_deficient_basis_projects_out_its_span_only(self) -> None:
        rng = np.random.default_rng(2)
        basis = rng.normal(size=(200, 3))
        matrix = rng.normal(size=(200, 5))
        np.testing.assert_allclose(
            screeners._residualise(matrix, np.column_stack([basis, basis[:, :1], basis[:, :2]])),
            screeners._residualise(matrix, basis),
            atol=1e-10,
        )

    def test_a_dose_with_a_point_mass_gets_a_spline_without_empty_columns(self) -> None:
        dose = np.maximum(0.0, np.random.default_rng(4).normal(size=500) - 0.5)
        assert np.mean(dose == 0.0) > 0.5
        partial, _ = screeners._TreatmentBasis(dose[:, None], (False,)).build(np.arange(500))
        assert np.all(np.abs(partial).sum(axis=0) > 0.0)
        assert np.linalg.matrix_rank(partial) == partial.shape[1]

    def test_inner_folds_keep_each_cluster_whole(self) -> None:
        rows = np.arange(100)
        groups = np.repeat(np.arange(20), 5)
        for train, test in screeners._inner_splits(rows, groups):
            assert set(groups[train]).isdisjoint(groups[test])


class _NoStepsLogit(LogisticRegression):
    """A logistic regression that takes the classifier design without policy steps."""

    policy_steps = "omit"


def _point_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    w, a, y = instrument_law(seed, n=n)
    return pd.DataFrame(w, columns=["C", "Z", "P", "N"]).assign(A=a, Y=y)


class TestEstimators:
    def test_the_reported_covariates_are_the_whole_sample_support(self) -> None:
        frame = _point_frame()
        result = (
            TMLE(
                estimands=("ate",),
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
                screen_treatment=True,
                min_retain=1,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["C", "Z", "P", "N"])
            .single()
        )
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["C", "Z", "P", "N"]
        )
        keep = outcome_screen(data, 1).support()
        expected = tuple(name for name, flag in zip("CZPN", keep, strict=True) if flag)
        assert result.repeats[0].nuisance.treatment_covariates == expected
        assert "C" in expected and "P" in expected

    def test_a_screened_learner_refuses_a_fit_without_its_rows(self) -> None:
        w, a, y = instrument_law(0, n=100)
        screen = OutcomeScreen(w, a, (True,), y, np.ones(100, bool), np.ones(100), None, 1)
        with pytest.raises(TypeError, match="learner_for_rows"):
            ScreenedLearner(LogisticRegression(), screen).fit(w, a)


def _dose_frame(n: int = 500, seed: int = 0, atom: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 4))
    a = 0.5 * w[:, 0] + 0.8 * w[:, 1] + rng.normal(size=n)
    if atom:
        # Recorded mass at the top dose 1.0, which the capped shift holds in place.
        a = np.minimum(a, 1.0)
    logit = 0.5 * a + w[:, 0] + 0.5 * w[:, 2]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    return pd.DataFrame(w, columns=["C", "Z", "P", "N"]).assign(A=a, Y=y)


def _dose_fit(frame: pd.DataFrame, screen: bool, floor: int | None, **options: Any) -> Any:
    fit_options = options.pop("fit_options", {})
    return (
        TMLE(
            policies=[Shift(0.3, cap=1.0)],
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            simultaneous=False,
            random_state=0,
            screen_treatment=screen,
            min_retain=floor,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["C", "Z", "P", "N"], **fit_options)
        .single()
    )


ROUTES: dict[str, tuple[dict[str, Any], bool]] = {
    "density": ({"density_bins": 4}, False),
    "classifier": ({"ratio": "classifier"}, False),
    "atoms": ({"density_bins": 4, "fit_options": {"dose_atoms": (1.0,)}}, True),
}


class TestEveryDoseRouteScreensItsTrainingSets:
    """Each route fits through ``learner_for_rows``; a forgotten one raises TypeError."""

    @pytest.mark.parametrize("route", list(ROUTES))
    def test_a_full_floor_screen_equals_the_unscreened_fit(self, route: str) -> None:
        options, atom = ROUTES[route]
        frame = _dose_frame(atom=atom)
        plain = _dose_fit(frame, False, None, **dict(options))
        screened = _dose_fit(frame, True, None, **dict(options))
        for name, estimate in plain.estimates.items():
            assert abs(screened.estimates[name].psi - estimate.psi) <= 1e-10

    @pytest.mark.parametrize("route", list(ROUTES))
    def test_a_screen_that_drops_columns_moves_the_fit(self, route: str) -> None:
        options, atom = ROUTES[route]
        frame = _dose_frame(atom=atom)
        plain = _dose_fit(frame, False, None, **dict(options))
        screened = _dose_fit(frame, True, 1, **dict(options))
        assert screened.repeats[0].nuisance.treatment_covariates != ("C", "Z", "P", "N")
        moved = max(
            abs(screened.estimates[name].psi - estimate.psi)
            for name, estimate in plain.estimates.items()
        )
        assert moved > 1e-8


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_a_vector_treatment_screens_its_training_sets(
    ratio: Literal["density", "classifier"],
) -> None:
    """Every factor of the chain rule, or the stacked classifier, fits through the screen."""
    import warnings

    from cleverly.interventions import JointPolicy

    rng = np.random.default_rng(3)
    n = 500
    w, noise = rng.normal(size=n), rng.normal(size=n)
    flag = (rng.uniform(size=n) < 1 / (1 + np.exp(-w))).astype(float)
    dose = 0.5 * w + 0.5 * flag + rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.4 * dose + w)))).astype(float)
    frame = pd.DataFrame({"W": w, "N": noise, "F": flag, "D": dose, "Y": y})
    policies = [
        JointPolicy("natural", components={}, policy_kind="known"),
        JointPolicy("dose up", components={"D": Shift(0.3, cap=None)}, policy_kind="known"),
    ]
    data = CausalData.from_frame(
        frame,
        outcome="Y",
        treatment=["D", "F"],
        covariates=["W", "N"],
        continuous_treatment=["D"],
    )

    def fit(**screen: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                TMLE(
                    policies=policies,
                    outcome_learner=LogisticRegression(),
                    treatment_learner=_NoStepsLogit(),
                    ratio=ratio,
                    n_folds=3,
                    random_state=0,
                    simultaneous=False,
                    **screen,
                )
                .fit(data)
                .single()
            )

    plain = fit()
    for name, estimate in plain.estimates.items():
        assert abs(fit(screen_treatment=True).estimates[name].psi - estimate.psi) <= 1e-10
    dropped = fit(screen_treatment=True, min_retain=1)
    assert all(np.isfinite(estimate.psi) for estimate in dropped.estimates.values())
    assert "W" in dropped.repeats[0].nuisance.treatment_covariates


class TestEachRouteMasksTheCovariatesOfItsTrainingSet:
    """The mask sits on the covariate block, and each support reads one training set."""

    @pytest.mark.parametrize("route", list(ROUTES))
    def test_the_mask_and_its_rows(self, route: str, monkeypatch: pytest.MonkeyPatch) -> None:
        options, atom = ROUTES[route]
        masks: list[np.ndarray] = []
        supports: list[tuple[np.ndarray | None, np.ndarray]] = []
        fit_mask = screeners.ScreenMask.fit
        support = OutcomeScreen.support

        def spy_fit(self: Any, X: Any, y: Any = None) -> Any:
            masks.append(np.asarray(self.support, dtype=bool))
            return fit_mask(self, X, y)

        def spy_support(self: Any, rows: Any = None) -> np.ndarray:
            kept = support(self, rows)
            supports.append((None if rows is None else np.sort(np.asarray(rows)), kept))
            return kept

        monkeypatch.setattr(screeners.ScreenMask, "fit", spy_fit)
        monkeypatch.setattr(OutcomeScreen, "support", spy_support)
        result = _dose_fit(_dose_frame(atom=atom), True, 1, **dict(options))
        p = 4
        trains = [np.sort(train) for train, _ in result.repeats[0].nuisance.folds]
        fold_supports = [kept for rows, kept in supports if rows is not None]
        # The rows a fit trains on: a whole training set, or its non-atom rows on the
        # continuous part of a mixed density.
        assert all(
            any(np.isin(rows, train).all() for train in trains)
            for rows, _ in supports
            if rows is not None
        ), "a support read rows outside one training set"
        assert any(
            any(np.array_equal(rows, train) for train in trains)
            for rows, _ in supports
            if rows is not None
        )
        assert masks
        last = route == "classifier"
        for mask in masks:
            block, rest = (mask[-p:], mask[:-p]) if last else (mask[:p], mask[p:])
            assert rest.all(), "the mask dropped a column that is not a covariate"
            assert any(np.array_equal(block, kept) for kept in fold_supports)
        assert any(not mask.all() for mask in masks), "no fold dropped a covariate: vacuous"


def test_a_screened_learner_shows_the_learner_opt_in_flags() -> None:
    """The classifier route reads ``policy_steps`` off the learner before fitting."""
    w, a, y = instrument_law(0, n=100)
    screen = OutcomeScreen(w, a, (True,), y, np.ones(100, bool), np.ones(100), None, 1)
    screened = ScreenedLearner(_NoStepsLogit(), screen)
    assert screened.policy_steps == "omit"
    assert getattr(ScreenedLearner(LogisticRegression(), screen), "bin_design", None) is None
