r"""The refute adapter of the structural nested mean model (X82 part D).

:func:`cleverly.validation.refute` runs four tests on a
:class:`~cleverly.longitudinal.StructuralNestedMeanResult`. This module holds what differs from
the longitudinal adapter: the null of each reported parameter, the split and seed every refit
reuses, the request refusals, and the scope axes the registry reads.

The placebo law
---------------

The placebo draws each trajectory from the first-order Markov kernel of
:mod:`cleverly.validation._markov_placebo`. The estimator refuses censoring, events, held and
known nodes, so no mask applies. Given the data, the drawn trajectories are independent of
every non-treatment variable :math:`O^-`, so no :math:`L_t` and not :math:`Y` reads the drawn
treatment. Then :math:`E[Y \mid H_t, A_t]` is free of :math:`A_t`, every true blip is 0, and
:math:`\psi = 0` solves the population g-estimating equation of every node. The diagonal
Jacobian blocks stay above the floor, so it is the unique root.

==============================  ===================================================
parameter                       placebo reference
==============================  ===================================================
``blip[...]``                   0
``ate_regimen[...]``            0: a contrast is linear in the blips
``ey_reference``                none: a level has no constant null
``ey_regimen[...]``             none
==============================  ===================================================

The model holds on the placebo law with every blip 0, so a regimen contrast needs no regimen
positivity: it is read from the blips, not from the units that follow the regimen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import Refusal, RefusalReason

if TYPE_CHECKING:
    from ..learners.crossfit import Folds
    from ._interpretation import FitConfiguration

__all__ = [
    "STRUCTURAL_TESTS",
    "is_structural",
    "structural_configuration",
    "structural_fit_seeds",
    "structural_frozen_folds",
    "structural_null",
    "structural_subset_refusal",
    "structural_tests_refusal",
]

#: The tests the structural adapter runs (X82 part D).
STRUCTURAL_TESTS: tuple[str, ...] = (
    "placebo",
    "random_common_cause",
    "subset",
    "bootstrap_stability",
    "negative_control_outcome",
)

#: The parameter kinds whose placebo value is exactly 0.
_NULL_KINDS = ("blip", "contrast")


def is_structural(result: Any) -> bool:
    """Whether ``result`` is a structural nested mean fit.

    Parameters
    ----------
    result : Any
        A fitted result.

    Returns
    -------
    bool
        ``True`` when the result declares the ``"structural"`` assessment family.
    """
    return getattr(result, "assessment_family", None) == "structural"


def structural_tests_refusal(tests: Any) -> Refusal | None:
    """Refuse a test the structural adapter does not run.

    Parameters
    ----------
    tests : sequence of str
        The requested test names.

    Returns
    -------
    Refusal or None
        The refusal, or ``None`` when every requested test runs.
    """
    pending = [name for name in tests if name not in STRUCTURAL_TESTS]
    if not pending:
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"refutation test(s) {pending} have no structural nested mean adapter yet. The "
        "generated outcomes, the future-treatment control and the measurement-error bootstrap "
        "of a structural fit are X82 work.",
        alternative=(
            "The placebo, random_common_cause, subset, bootstrap_stability and "
            "negative_control_outcome tests run on this fit."
        ),
        roadmap="X82",
    )


def structural_null(result: Any, estimand: str) -> float | None:
    """Return the placebo and no-effect value of one structural parameter.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fit.
    estimand : str
        A reported parameter name.

    Returns
    -------
    float or None
        ``0.0`` for a blip coefficient or a regimen contrast, and ``None`` for a level.
    """
    kind = getattr(result, "kinds", {}).get(estimand)
    return 0.0 if kind in _NULL_KINDS else None


def structural_frozen_folds(result: Any, index: np.ndarray | None = None) -> tuple[Folds, ...]:
    """The original split of the rows a refit fits, as the one-entry ``frozen_folds``.

    A subset or a resample keeps each drawn unit's original fold label, so no refit redraws
    the split. Labels that a draw leaves empty are compacted.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fit.
    index : ndarray or None
        The original row of each refit row. ``None`` keeps every row in order.

    Returns
    -------
    tuple of Folds
        One split.
    """
    from ..learners.crossfit import Folds

    folds = result.folds
    if index is None:
        return (folds,)
    if folds.is_single:
        return (Folds.single(int(np.asarray(index).size)),)
    labels = np.asarray(folds.assignment)[np.asarray(index)]
    _, compact = np.unique(labels, return_inverse=True)
    return (Folds(compact.astype(np.int64), int(compact.max()) + 1),)


def structural_fit_seeds(result: Any) -> tuple[int, ...] | None:
    """The seed the fit drew its split and seeded its learners with, as ``frozen_seeds``.

    A refit given it reproduces the original learner seeds as well as the split, so a paired
    shift carries no seed noise on an unseeded fit.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fit.

    Returns
    -------
    tuple of int or None
        One seed, or ``None`` on a result that records none.
    """
    seed = getattr(getattr(result, "config", None), "fit_seed", None)
    return None if seed is None else (int(seed),)


def structural_subset_refusal(result: Any, draws: Any) -> Refusal | None:
    """Refuse subset draws that leave a fold with too little treatment variation.

    Each subset row keeps its original fold, so a small ``subset_fraction`` can leave a fold
    whose rows hold fewer than two units of some level of a categorical node. The refit would
    then stop at the Jacobian floor part way through the test. The check reads every draw
    before the first subset refit.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fit.
    draws : sequence of ndarray
        The original rows of each subset.

    Returns
    -------
    Refusal or None
        The refusal, or ``None`` when every fold of every draw holds two units of each level.
    """
    data = result.data
    treatment = np.asarray(data.treatment, dtype=float)
    categorical = [
        time for time in range(1, data.n_times + 1) if data.node_kind(time) == "categorical"
    ]
    if not categorical:
        return None
    for index in draws:
        (folds,) = structural_frozen_folds(result, index)
        labels = np.asarray(folds.assignment) if not folds.is_single else np.zeros(index.size)
        for label in np.unique(labels):
            rows = np.asarray(index)[labels == label]
            for time in categorical:
                levels = len(data.treatment_levels[time - 1])
                counts = np.bincount(treatment[rows, time - 1].astype(np.int64), minlength=levels)
                if counts.min() < 2:
                    return Refusal(
                        RefusalReason.INVALID_DECLARATION,
                        f"a subset draw leaves a fold with {int(counts.min())} unit(s) of a "
                        f"level of node {time} ({data.treatment_names[time - 1]}). Each "
                        "subset row keeps its original fold, so the refit would have no "
                        "treatment variation to solve the node's blip from. Raise "
                        "subset_fraction, or drop subset from tests=.",
                    )
    return None


def structural_configuration(result: Any, estimand: str) -> FitConfiguration:
    """Read the scope axes of a structural fit for one estimand.

    The estimand kind is the parameter stem, such as ``"blip"`` or ``"ate_regimen"``,
    because the bracketed part names the user's columns and regimens.

    Parameters
    ----------
    result : StructuralNestedMeanResult
        The fit.
    estimand : str
        A reported parameter name.

    Returns
    -------
    FitConfiguration
        The scope axes as the fitted result records them.
    """
    from ..targets.base import parameter_stem
    from ._interpretation import FitConfiguration, learner_fingerprint

    data = result.data
    estimator = result.estimator
    estimate = result[estimand]
    n = int(data.n)
    learners = tuple(
        learner_fingerprint(getattr(estimator, name, None), n)
        for name in ("outcome_learner", "treatment_learner", "regression_learner")
    )
    folds = int(result.config.n_folds)
    if data.has_continuous_node:
        treatment_kind = "continuous"
    elif any(len(columns) > 1 for columns in data.treatment_components):
        treatment_kind = "vector"
    else:
        treatment_kind = "discrete"
    width = data.baseline.shape[1] + sum(block.shape[1] for block in data.time_varying)
    arms = max((len(levels) for levels in data.treatment_levels), default=1)
    events: int | None = None
    if str(data.family) == "binomial":
        count = int(np.count_nonzero(np.asarray(data.outcome, dtype=float) > 0.5))
        events = min(count, n - count)
    return FitConfiguration(
        method=type(estimator).__name__,
        estimand_kind=parameter_stem(estimand),
        outcome_family=str(data.family),
        treatment_kind=treatment_kind,
        learners=learners,
        cross_fitting=f"folds={folds}" if folds > 1 else "off",
        inference=str(estimate.inference),
        known_mechanism=False,
        n=n,
        covariates_per_row=float(width) / max(1, n),
        min_event_count=events,
        clusters=None if data.cluster is None else int(np.unique(data.cluster).size),
        arms=max(1, arms),
        nodes=int(data.n_times),
    )
