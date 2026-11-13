"""A seeded fit is a function of its data and seed (Fix-11).

``SuperLearner(random_state=s)`` gives every library candidate with no seed of its own the
seed ``s``, and ``resolve_learner`` does the same for any learner an estimator passes a seed
to. A stochastic candidate left at ``random_state=None`` would otherwise draw from numpy's
global generator, which every worker process seeds from OS entropy.

Each determinism check fits under two different global seeds. The probe of the first test
shows that the global seed does move an unseeded SAGA fit, so equality elsewhere is not
vacuous; the A-TMLE checks carry their own mutation controls.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import cleverly.estimators._working_basis as working_basis_module
import cleverly.learners.super_learner as super_learner_module
from cleverly import (
    AdaptiveTMLEMethod,
    AdaptiveTMLEResult,
    CausalStudy,
    CrossFitting,
    ModelSpec,
    SuperLearner,
    TrialATE,
    TrialWithExternalData,
)
from cleverly.exceptions import MethodConfigurationError
from cleverly.learners._fitting import seed_unset
from cleverly.learners.super_learner import resolve_learner

#: Two global seeds under which an unseeded stochastic fit differs.
GLOBAL_SEEDS = (1, 2)


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """Seeding numpy's global generator here must not leak into other tests.

    The legacy global generator is the subject of this file (an unseeded learner draws from
    it), so the calls NPY002 flags are the point here, not a habit to migrate.
    """
    state = np.random.get_state()  # noqa: NPY002
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # scikit-learn's notices about LogisticRegressionCV's future defaults.
        warnings.simplefilter("ignore", FutureWarning)
        yield
    np.random.set_state(state)  # noqa: NPY002


def _under(seed: int, fit: Callable[[], Any]) -> Any:
    np.random.seed(seed)  # noqa: NPY002
    return fit()


def _draw(n: int = 400, *, classes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    score = x @ np.array([1.0, -1.0, 0.5])
    if classes == 2:
        return x, (rng.random(n) < 1.0 / (1.0 + np.exp(-score))).astype(float)
    return x, np.digitize(score + rng.normal(size=n), [-0.7, 0.7]).astype(float)


def _saga(random_state: Any = None) -> LogisticRegressionCV:
    """A candidate whose fit draws its sample order from ``random_state``."""
    return LogisticRegressionCV(solver="saga", penalty="l1", max_iter=50, random_state=random_state)


def _probabilities(learner: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(learner.fit(x, y).predict_proba(x), dtype=float)


def test_the_global_seed_moves_an_unseeded_saga_fit() -> None:
    """Mutation control: the probe can see the defect.

    If a future scikit-learn makes SAGA deterministic this fails and says so, rather than
    letting the determinism checks below pass vacuously.
    """
    x, y = _draw()
    first, second = (_under(seed, lambda: _probabilities(_saga(), x, y)) for seed in GLOBAL_SEEDS)
    assert np.abs(first - second).max() > 1e-8


def test_a_seeded_ensemble_is_a_function_of_its_data_and_seed() -> None:
    x, y = _draw()

    def ensemble() -> SuperLearner:
        return SuperLearner(
            [("saga", _saga()), ("glm", LogisticRegression())],
            task="classification",
            n_folds=3,
            random_state=0,
        )

    first, second = (
        _under(seed, lambda: _probabilities(ensemble(), x, y)) for seed in GLOBAL_SEEDS
    )
    assert np.array_equal(first, second)


def test_an_unseeded_ensemble_stays_unseeded() -> None:
    """``random_state=None`` is the caller asking for no seed, for the candidates too.

    One candidate carries weight one, so predictions come from its full-data refit alone and
    the fold split cannot be what differs.
    """
    x, y = _draw()
    fitted = []
    for seed in GLOBAL_SEEDS:
        ensemble = SuperLearner(
            [("saga", _saga())], task="classification", n_folds=3, random_state=None
        )
        fitted.append(_under(seed, lambda ensemble=ensemble: ensemble.fit(x, y)))
    first, second = (np.asarray(model.predict_proba(x)) for model in fitted)
    assert np.abs(first - second).max() > 1e-8
    assert all(model.fitted_learners_[0].random_state is None for model in fitted)


def test_one_vs_rest_members_inherit_the_seed() -> None:
    x, y = _draw(classes=3)

    def ensemble() -> SuperLearner:
        return SuperLearner([("saga", _saga())], task="classification", n_folds=3, random_state=0)

    fitted = [_under(seed, lambda: ensemble().fit(x, y)) for seed in GLOBAL_SEEDS]
    assert fitted[0].is_multiclass
    assert np.array_equal(fitted[0].predict_proba(x), fitted[1].predict_proba(x))
    for member in fitted[0].one_vs_rest_:
        assert member.fitted_learners_[0].random_state == 0


@pytest.mark.parametrize("explicit", [7, np.random.RandomState(7)], ids=["int", "RandomState"])
def test_the_caller_is_not_mutated_and_explicit_seeds_are_kept(explicit: Any) -> None:
    x, y = _draw()
    unseeded, seeded = _saga(), _saga(explicit)
    ensemble = SuperLearner(
        [("unseeded", unseeded), ("seeded", seeded)],
        task="classification",
        n_folds=3,
        random_state=0,
    ).fit(x, y)

    assert unseeded.random_state is None
    assert seeded.random_state is explicit
    fitted = dict(zip(ensemble.learner_names_, ensemble.fitted_learners_, strict=True))
    assert fitted["unseeded"].random_state == 0
    if isinstance(explicit, int):
        assert fitted["seeded"].random_state == explicit
    else:
        # A RandomState is cloned with its state, not replaced by the ensemble's seed.
        assert isinstance(fitted["seeded"].random_state, np.random.RandomState)


def test_a_nested_random_state_is_seeded() -> None:
    x, y = _draw()
    candidate = Pipeline([("scale", StandardScaler()), ("model", _saga())])
    ensemble = SuperLearner(
        [("pipeline", candidate)], task="classification", n_folds=3, random_state=4
    ).fit(x, y)

    assert ensemble.fitted_learners_[0].get_params()["model__random_state"] == 4
    assert candidate.get_params()["model__random_state"] is None
    seeded = seed_unset(candidate, 4)
    assert seeded is not candidate
    assert seeded.get_params()["model__random_state"] == 4


def test_nothing_to_seed_returns_the_learner_itself() -> None:
    learner = LinearRegression()
    assert seed_unset(learner, 0) is learner
    already = _saga(3)
    assert seed_unset(already, 0) is already
    assert seed_unset(_saga(), None).random_state is None
    assert seed_unset(object(), 0).__class__ is object


class _NoSetParams:
    """A learner with ``get_params`` and no ``set_params``, which ``clone`` accepts."""

    def __init__(self, random_state: Any = None, nested: Any = None) -> None:
        self.random_state = random_state
        self.nested = nested

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {"random_state": self.random_state, "nested": self.nested}
        if deep and self.nested is not None:
            params.update(
                {f"nested__{key}": value for key, value in self.nested.get_params().items()}
            )
        return params

    def fit(self, X: Any, y: Any) -> _NoSetParams:
        return self

    def predict(self, X: Any) -> Any:
        return np.zeros(len(X))


def test_a_learner_without_set_params_is_rebuilt_with_the_seed() -> None:
    learner = _NoSetParams()
    seeded = seed_unset(learner, 5)
    assert isinstance(seeded, _NoSetParams)
    assert seeded is not learner
    assert seeded.random_state == 5
    assert learner.random_state is None


def test_a_nested_unset_seed_without_set_params_is_refused_by_name() -> None:
    learner = _NoSetParams(random_state=1, nested=_saga())
    with pytest.raises(MethodConfigurationError, match="_NoSetParams has no set_params"):
        seed_unset(learner, 5)


def test_resolve_learner_seeds_a_bare_learner_on_a_clone() -> None:
    bare = _saga()
    resolved = resolve_learner(bare, task="classification", random_state=3)
    assert resolved is not bare
    assert resolved.random_state == 3
    assert bare.random_state is None
    assert resolve_learner(bare, task="classification", random_state=None) is bare
    assert resolve_learner(_saga(9), task="classification", random_state=3).random_state == 9


def test_resolve_learner_seeds_an_unseeded_super_learner_on_a_clone() -> None:
    ensemble = SuperLearner([("saga", _saga())], task="classification", random_state=None)
    resolved = resolve_learner(ensemble, task="classification", random_state=3)
    assert resolved.random_state == 3
    assert ensemble.random_state is None
    # The library is a list, which get_params does not expand: the candidates are seeded
    # by the ensemble's own fit.
    assert resolved.library[0][1].random_state is None


# ----------------------------------------------------------------------------- A-TMLE


def _trial_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    a = np.where(s == 1, rng.binomial(1, 0.5, n), rng.binomial(1, 0.3, n))
    y = 1.0 + a + w + 0.5 * (1 - s) * (1 + a) + rng.normal(size=n)
    return pd.DataFrame({"Y": y, "A": a, "W": w, "S": s})


def _atmle(**declared: Any) -> tuple[float, float]:
    design = TrialWithExternalData(outcome="Y", treatment="A", adjustment=("W",), source="S")
    effect = CausalStudy(_trial_frame(), design=design).identify(TrialATE(population="trial"))
    method = AdaptiveTMLEMethod(
        form="cv",
        models=ModelSpec(
            outcome_learner=LinearRegression(), treatment_learner=LogisticRegression()
        ),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
        **declared,
    ).with_overrides(random_state=7)
    result = effect.estimate(method=method)
    assert isinstance(result, AdaptiveTMLEResult)
    return float(result.estimate.psi), float(result.estimate.std_error)


def _identity(learner: Any, seed: Any) -> Any:
    return learner


def test_an_atmle_fit_with_an_unseeded_candidate_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trial-membership ensemble's SAGA candidate takes the fit's seed.

    The ensemble itself carries no seed, so ``resolve_learner`` seeds it and its fit seeds the
    candidate. With seeding switched off the two global seeds give two estimates.
    """

    def fit() -> tuple[float, float]:
        source = SuperLearner(
            [("saga", _saga()), ("glm", LogisticRegression())],
            task="classification",
            n_folds=3,
        )
        return _atmle(source_learner=source)

    first, second = (_under(seed, fit) for seed in GLOBAL_SEEDS)
    assert first == second

    monkeypatch.setattr(super_learner_module, "seed_unset", _identity)
    first, second = (_under(seed, fit) for seed in GLOBAL_SEEDS)
    assert first != second


def test_an_atmle_working_basis_without_a_seed_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random features with no seed of their own take the fit's seed.

    Four components rather than ten: at ten the working-model lasso on these collinear
    features runs for minutes at this ``n``, and four already moves the estimate.
    """

    def fit() -> tuple[float, float]:
        basis = make_pipeline(StandardScaler(), RBFSampler(n_components=4))
        return _atmle(working_basis=basis)

    first, second = (_under(seed, fit) for seed in GLOBAL_SEEDS)
    assert first == second

    monkeypatch.setattr(working_basis_module, "seed_unset", _identity)
    first, second = (_under(seed, fit) for seed in GLOBAL_SEEDS)
    assert first != second
