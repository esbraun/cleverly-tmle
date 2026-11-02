"""The truth harness of ``learned-regimen-ltmle``, checked without a study run.

The harness is the study's instrument: rule L4 (a scikit-learn refit of each fold's Q-learning
must give the fit's stored rule pairs on every row), the quadrature value of each fold's pair, and
the exactness gate (the variance rebuilt from the stored node steps). Each check here has a
deliberate mutation that the harness must catch, because a check that passes at the truth can be
blind to a wrong component.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.studies import _learned_regimen_law as law

pytestmark = pytest.mark.xdist_group("learned_regimen_ltmle_harness")

N = 2_000
SAMPLE_SEED = 911_001
FOLD_SEED = 911_002


@pytest.fixture(scope="module")
def frame():
    return law.draw(N, SAMPLE_SEED)


@pytest.fixture(scope="module")
def declared(frame):
    return law.measured(frame, "declared", FOLD_SEED)


@pytest.fixture(scope="module")
def mutated(frame):
    return law.measured(frame, "declared", FOLD_SEED, mutation=True)


def _monte_carlo(pair_rule, draws: int = 200_000, seed: int = 5) -> tuple[float, float]:
    """The value of a rule pair by Monte Carlo of the intervened law, independent of the
    quadrature: ``W``, then ``L2`` under the node-1 arm, then the node-2 arm, and the mean of
    ``Qbar0``. Returns the mean and its SE."""
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, draws)
    w2 = rng.binomial(1, 0.5, draws).astype(float)
    a1, second = pair_rule(w1, w2)
    l2 = law.l2_mean(a1, w1, w2) + rng.standard_normal(draws)
    a2 = second(a1, l2, w1, w2)
    q = law.qbar0(a1, a2, l2, w1, w2)
    return float(np.mean(q)), float(np.std(q) / np.sqrt(draws))


def test_the_quadrature_value_of_the_optimal_pair_matches_monte_carlo() -> None:
    def optimal(w1, w2):
        first = np.zeros_like(w1)
        for level in law.W2_LEVELS:
            rows = w2 == level
            first[rows] = np.interp(w1[rows], law.GRID, law.optimal_pair().first[level]) > 0.5
        return first, lambda a1, l2, w1, w2: (law.blip2(a1, l2) > 0.0).astype(float)

    value, se = _monte_carlo(optimal)
    assert law.law_limit()["truth"] == pytest.approx(value, abs=4 * se)


@pytest.mark.parametrize(("a1", "a2"), [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
def test_the_quadrature_value_of_each_static_regimen_matches_monte_carlo(
    a1: float, a2: float
) -> None:
    def static(w1, w2):
        return np.full_like(w1, a1), lambda *_: a2

    value, se = _monte_carlo(static)
    assert law.static_values()[a1, a2] == pytest.approx(value, abs=4 * se)


def test_the_grid_is_fine_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Against 16,001 points of ``W1`` and 64 Gauss-Legendre nodes, the declared grid moves the
    optimal pair's value by about 1e-5, 0.05% of an SE at n = 2,000."""
    coarse = law.rule_value(law.optimal_pair())[0]
    # Read before the patch, so the cached limit stays the declared grid's.
    se = law.law_limit()["bound"] / np.sqrt(N)
    grid = np.linspace(-1.0, 1.0, 16_001)
    mass = np.full(grid.size, grid[1] - grid[0])
    mass[[0, -1]] /= 2.0
    monkeypatch.setattr(law, "GRID", grid)
    monkeypatch.setattr(law, "MASS", mass * 0.25)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    monkeypatch.setattr(law, "_NODES", nodes)
    monkeypatch.setattr(law, "_WEIGHTS", weights)
    fine = law.rule_value(law.optimal_pair.__wrapped__())[0]
    assert abs(fine - coarse) < 3e-5
    assert abs(fine - coarse) < 0.002 * se


def test_the_harness_checks_every_row(declared) -> None:
    assert declared.truth.rows_checked == N
    assert len(declared.truth.fold_values) == law.N_FOLDS
    assert declared.solver_warnings == 0
    # The fold values lie at or below the optimal value, up to the quadrature error.
    assert max(declared.truth.fold_values) <= law.law_limit()["truth"] + 1e-5
    assert abs(declared.truth.truth - law.law_limit()["truth"]) < 0.05
    se = float(declared.estimate.std_error)
    assert 0.7 < declared.truth.oracle_se / se < 1.4


def test_rule_l4_catches_a_wrong_learner(declared, frame) -> None:
    """The deliberate mutation of rule L4: refitting with another node-1 learner gives another
    rule pair on some rows, and the harness refuses the fit."""
    with pytest.raises(law.HarnessMismatch, match="refit rule pair differs"):
        law.harness(declared.result, frame, "drop_w1")


def test_the_exactness_gate_matches_and_catches_a_moved_variance(
    declared, frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    reported = float(declared.estimate.std_error) ** 2
    assert law.curve_variance(declared.result) == pytest.approx(reported, rel=1e-10)
    original = law.curve_variance
    monkeypatch.setattr(law, "curve_variance", lambda result: original(result) * (1 + 1e-6))
    with pytest.raises(law.HarnessMismatch, match="rebuilt variance"):
        law.harness(declared.result, frame, "declared")


def test_the_validation_row_control_learns_other_rules(declared, mutated, frame) -> None:
    """The ``fold_locality`` control: on one sample and fold seed, rules learned on each fold's
    validation rows differ from the fold-local ones, the harness reproduces them under
    ``mutation=True`` and refuses them without it, and the control reads optimistic."""
    local = np.asarray(declared.result.learned_regimen.fold_rules)
    learned_on_validation = np.asarray(mutated.result.learned_regimen.fold_rules)
    assert np.any(local != learned_on_validation)
    assert mutated.truth.rows_checked == N
    with pytest.raises(law.HarnessMismatch):
        law.harness(mutated.result, frame, "declared")
    optimism = float(mutated.estimate.psi) - mutated.truth.truth
    assert optimism > 2 * float(mutated.estimate.std_error)


def test_the_swap_is_restored() -> None:
    from cleverly.longitudinal import estimator as module

    original = module.learn_fold_rules
    with pytest.raises(RuntimeError), law.validation_rows():
        assert module.learn_fold_rules is law._validation_fold_rules
        raise RuntimeError
    assert module.learn_fold_rules is original


def test_the_untargeted_plug_in_is_the_initial_mean(declared) -> None:
    (regimen_fit,) = declared.result.fits.values()
    initial = np.mean(np.asarray(regimen_fit.steps[0].initial, dtype=float))
    scaler = declared.result.scaler
    expected = scaler.lower + (scaler.upper - scaler.lower) * initial
    assert law.untargeted_estimate(declared.result) == pytest.approx(expected, rel=1e-12)
    assert law.untargeted_estimate(declared.result) != pytest.approx(
        float(declared.estimate.psi), abs=1e-6
    )


def test_the_fit_keys_the_learned_regimen(declared) -> None:
    key = declared.result.config.support_key
    assert "learned" in key.family
    assert key.splitting == ("cross_fitted",)
    assert key.missing == ("censoring",)


@pytest.mark.parametrize("node1", ["drop_w1", "squared_w1"])
def test_the_harness_reproduces_each_misspecified_learner(frame, node1: str) -> None:
    fitted = law.measured(frame, node1, FOLD_SEED)
    assert fitted.truth.rows_checked == N
    # The other learner's refit gives other rules: rule L4 tells the learners apart.
    other = "squared_w1" if node1 == "drop_w1" else "drop_w1"
    with pytest.raises(law.HarnessMismatch):
        law.harness(fitted.result, frame, other)
