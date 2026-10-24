"""The sensitivity analyses whose summaries the contract test checks (X89 T4).

Each fixture is a :class:`Case` with a fidelity reader from :mod:`._sensitivity_readers`,
because no sensitivity object carries an estimates frame. The fits are compact, use linear
learners and skip cross-fitting where the analysis allows it, so every fixture runs in the
fast tier. Each fixture names the one note code it is there to fire.
"""

from __future__ import annotations

import functools
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.unit.summary_fixtures import Case
from tests.unit.summary_fixtures import _sensitivity_readers as read

#: The covariate names the fits print.
NAMES = ("Y", "A", "W1", "W2", "W3", "W4", "Delta")


def _learners(**changes: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "random_state": 0,
        **changes,
    }


@functools.cache
def _linear() -> Any:
    from cleverly.datasets import make_linear_ate
    from cleverly.estimators import TMLE

    frame = make_linear_ate(n=300, seed=1)[0]
    return TMLE(**_learners()).fit(frame, outcome="Y", treatment="A").single()


@functools.cache
def _forest() -> Any:
    """A fit whose treatment learner the plug-in limits do not admit."""
    from sklearn.ensemble import RandomForestClassifier

    from cleverly.datasets import make_linear_ate
    from cleverly.estimators import TMLE

    frame = make_linear_ate(n=300, seed=1)[0]
    learner = RandomForestClassifier(n_estimators=10, random_state=0)
    return (
        TMLE(**_learners(treatment_learner=learner)).fit(frame, outcome="Y", treatment="A").single()
    )


@functools.cache
def _collaborative() -> Any:
    from cleverly.datasets import make_linear_ate
    from cleverly.estimators import CTMLE

    frame = make_linear_ate(n=300, seed=1)[0]
    return (
        CTMLE(**_learners(), estimands=("ate",), selection_folds=3)
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def _missing() -> Any:
    from cleverly.datasets import make_missing_outcome
    from cleverly.estimators import TMLE

    frame = make_missing_outcome(n=300, seed=4)[0]
    return (
        TMLE(**_learners(), estimands=("ate", "ey1", "ey0"))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


@functools.cache
def _missing_collaborative() -> Any:
    """A selector-path collaborative fit with missing outcomes, which claims no interval."""
    from cleverly.datasets import make_missing_outcome
    from cleverly.estimators import CTMLE

    frame = make_missing_outcome(n=300, seed=4)[0]
    return (
        CTMLE(**_learners(), estimands=("ate",), selection_folds=3)
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


@functools.cache
def _binary() -> Any:
    from cleverly.datasets import make_binary_outcome
    from cleverly.estimators import TMLE

    frame = make_binary_outcome(n=300, seed=2)[0]
    learners = _learners(outcome_learner=LogisticRegression(max_iter=1000))
    return TMLE(**learners, estimands=("ate", "rr")).fit(frame, outcome="Y", treatment="A").single()


@functools.cache
def _studied(target: str) -> Any:
    """A study fit, which carries what the simulated-confounding refit needs."""
    from cleverly import ATE, ATT, CausalStudy, PointTreatment
    from cleverly.datasets import make_linear_ate

    frame = make_linear_ate(n=300, seed=1)[0]
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    estimand = ATE() if target == "ate" else ATT()
    return CausalStudy(frame, design=design).identify(estimand).estimate(**_learners())


def _quantile(tau: float) -> Any:
    from sklearn.linear_model import QuantileRegressor

    return QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")


@functools.cache
def _missing_binary() -> Any:
    from cleverly.datasets import make_missing_outcome_binary
    from cleverly.estimators import TMLE

    frame = make_missing_outcome_binary(n=300, seed=4)[0]
    learners = _learners(outcome_learner=LogisticRegression(max_iter=1000))
    return (
        TMLE(**learners, estimands=("ate", "rr"))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def _grid() -> Any:
    from cleverly.sensitivity import ConfounderStrengthGrid

    return ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.5))


FIXTURES = {
    # sensitivity_assumption, rv_limit: the doubly robust bound with its limits.
    "bounds": Case(
        lambda: _linear().sensitivity.omitted_confounding(cf_y=0.05, cf_d=0.05),
        names=NAMES,
        fidelity=read.bounds,
    ),
    # alignment_stated: an alignment below the worst case.
    "bounds_aligned": Case(
        lambda: _linear().sensitivity.omitted_confounding(cf_y=0.05, cf_d=0.05, rho=0.5),
        names=NAMES,
        fidelity=read.bounds,
    ),
    # limits_withheld: a treatment learner the plug-in standard error does not admit.
    "bounds_plugin": Case(
        lambda: _forest().sensitivity.omitted_confounding(
            cf_y=0.05, cf_d=0.05, nu2_estimator="plugin"
        ),
        names=NAMES,
        fidelity=read.bounds,
    ),
    # consistent_mechanism: a collaborative point bound.
    "bounds_consistent": Case(
        lambda: _collaborative().sensitivity.omitted_confounding(
            cf_y=0.05, cf_d=0.05, assume_consistent_mechanism=True
        ),
        names=NAMES,
        fidelity=read.bounds,
    ),
    # joint_strength, limits_descriptive: a weight that also reads the response model.
    "bounds_missing": Case(
        lambda: _missing().sensitivity.omitted_confounding(cf_y=0.05, cf_d=0.05),
        names=NAMES,
        fidelity=read.bounds,
    ),
    "robustness": Case(
        lambda: _linear().sensitivity.robustness_value(),
        names=NAMES,
        fidelity=read.robustness,
    ),
    # The unreached verdict: at zero alignment no strength moves the estimate.
    "robustness_unreachable": Case(
        lambda: _linear().sensitivity.robustness_value(rho=0.0),
        names=NAMES,
        fidelity=read.robustness,
    ),
    "elements": Case(
        lambda: _linear().sensitivity.elements(estimand="ate"),
        names=NAMES,
        fidelity=read.elements,
    ),
    "benchmark": Case(
        lambda: _linear().sensitivity.benchmark(covariates=("W1",)),
        names=NAMES,
        fidelity=read.benchmark,
    ),
    # grid_thinned: a 20 by 20 grid prints 6 rows and 5 columns.
    "contour": Case(
        lambda: _linear().sensitivity.contour(grid_size=20),
        names=NAMES,
        fidelity=read.contour,
    ),
    # approximate_scale: a mean difference converted to a risk ratio.
    "evalue_converted": Case(
        lambda: _linear().sensitivity.evalue(),
        names=NAMES,
        fidelity=read.evalue,
    ),
    # An exact risk ratio, with no conversion note.
    "evalue_ratio": Case(
        lambda: _binary().sensitivity.evalue(estimand="rr"),
        names=NAMES,
        fidelity=read.evalue,
    ),
    "marginal": Case(
        lambda: _linear().sensitivity.marginal_bounds(
            gamma=(1.0, 1.5, 2.0),
            quantile_learner=_quantile,
            pseudo_outcome_learner=LinearRegression(),
        ),
        names=NAMES,
        fidelity=read.marginal,
    ),
    "tilt": Case(
        lambda: _missing().sensitivity.missingness(gamma=(-1.0, 0.0, 1.0)),
        names=NAMES,
        fidelity=read.tilt,
    ),
    # no_inference, tilt_diagnostic: a fit that claims no interval.
    "tilt_withheld": Case(
        lambda: _missing_collaborative().sensitivity.missingness(gamma=(0.0, 1.0)),
        names=NAMES,
        fidelity=read.tilt,
    ),
    # tilt_diagnostic: off a gamma of zero a ratio reports a diagnostic spread.
    "tilt_ratio": Case(
        lambda: _missing_binary().sensitivity.missingness(gamma=(0.0, 1.0)),
        names=NAMES,
        fidelity=read.tilt,
    ),
    "surface": Case(
        lambda: _studied("ate").sensitivity.simulated_confounding(
            grid=_grid(), benchmark_covariates=("W1",), random_state=0
        ),
        names=NAMES,
        fidelity=read.surface,
    ),
    # population_moves: an ATT cell rebuilds its own population.
    "surface_att": Case(
        lambda: _studied("att").sensitivity.simulated_confounding(grid=_grid(), random_state=0),
        names=NAMES,
        fidelity=read.surface,
    ),
}
