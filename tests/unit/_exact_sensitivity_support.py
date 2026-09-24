"""The oracle fits on the exact finite laws, which the Gateaux and sensitivity tests share.

Each builder fits TMLE on :mod:`tests.discrete_law` or :mod:`tests.discrete_law_multi` with
that law's own nuisances, in sample.  The sample realises the law exactly, so the targeting
step moves nothing and every curve the fit reports is the curve at the law.  A weighted fit
hands the learners the nuisances of the tilted law, which is what a weighted learner
converges to.

Four test modules built these fits by hand before RM22, with the same arguments in each.
They live here once.  The three-arm law gets its contamination derivative and its tilt here
too, from the generic :func:`tests.discrete_law.contamination_eif` and
:func:`tests.discrete_law.tilt_on`.  :mod:`tests.discrete_law_multi` does not import them
itself, because a registered study imports that module and a new import would open a gap
in that study's recorded module list.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from cleverly.estimators import TMLE
from tests import discrete_law as law
from tests import discrete_law_multi as multi
from tests.conftest import OracleOutcome, OracleTreatment

#: The estimands of the two-arm law that the omitted-variable bound applies to.
BINARY_ESTIMANDS: tuple[str, ...] = ("ey1", "ey0", "ate", "att", "atc")

#: Every parameter group the three-arm fits report.
MULTI_ESTIMANDS: tuple[str, ...] = ("ey", "ate", "att", "atc")

#: The reference of the three-arm fits, as a label.  ``"low"`` has arm code 1, because the
#: levels sort to ``("high", "low", "mid")``, so a test whose reference were code 0 could
#: not tell an implementation that reads the reference from one that assumes it.
REFERENCE = "low"

#: The observation weight of each value of ``W`` on the three-arm law, before normalisation.
#:
#: A function of ``W`` alone, and that is not a convenience.  ``g(a | W)`` and ``Qbar(a, W)``
#: are conditional on ``W``, so such a weight leaves both where they were, the oracle
#: nuisances stay exact, and the weighted score is zero cell by cell at ``epsilon = 0``.
RAW_WEIGHT = np.array([0.5, 1.0, 2.0])


def binary_oracle_fit(
    weights: str | None = None, *, estimands: Any = BINARY_ESTIMANDS, cluster: Any = None
) -> tuple[Any, np.ndarray | None]:
    """TMLE on the two-arm law with its oracle nuisances.

    Parameters
    ----------
    weights : str or None
        A key of :data:`tests.discrete_law.WEIGHT_FUNCTIONS`, or ``None`` for no weights.
    estimands : Any
        The estimands to request, passed to :class:`~cleverly.estimators.TMLE`.
    cluster : Any
        One cluster label per row of :func:`tests.discrete_law.frame`, or ``None``.

    Returns
    -------
    tuple
        The single fit, and the cell weights of the tilt or ``None``.
    """
    frame = law.frame()
    kwargs: dict[str, Any] = {}
    cells = None
    dgp = law.DiscreteLaw()
    if weights is not None:
        cells = law.cell_weights(law.WEIGHT_FUNCTIONS[weights])
        dgp = law.DiscreteLaw(law.tilt(law.PROBS, cells))
        frame = frame.assign(w=law.row_weights(cells))
        kwargs["weights"] = "w"
    if cluster is not None:
        frame = frame.assign(cluster=cluster)
        kwargs["id"] = "cluster"
    result = (
        TMLE(
            outcome_learner=OracleOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            cross_fit=False,
            estimands=estimands,
            simultaneous=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], **kwargs)
        .single()
    )
    return result, cells


def multi_oracle_fit(weighted: bool = False, *, estimands: Any = MULTI_ESTIMANDS) -> Any:
    """TMLE on the three-arm law with its oracle nuisances, against :data:`REFERENCE`.

    Parameters
    ----------
    weighted : bool
        Whether the rows carry :data:`RAW_WEIGHT` of their ``W``.
    estimands : Any
        The parameter groups to request.

    Returns
    -------
    Any
        The single fit.
    """
    frame = multi.frame()
    kwargs: dict[str, Any] = {}
    if weighted:
        frame = frame.assign(obs_weight=RAW_WEIGHT[frame["W"].to_numpy().astype(int)])
        kwargs["weights"] = "obs_weight"
    return (
        TMLE(
            outcome_learner=multi.OracleMultiOutcome(),
            treatment_learner=multi.OracleMultiTreatment(),
            cross_fit=False,
            estimands=estimands,
            reference=REFERENCE,
            simultaneous=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], **kwargs)
        .single()
    )


def multi_cell_weights() -> np.ndarray:
    """:data:`RAW_WEIGHT` of each support point of the three-arm law, in support order."""
    return np.array([RAW_WEIGHT[w] for w, _, _ in multi.SUPPORT], dtype=float)


def multi_tilt(probs: Any) -> Any:
    """The three-arm law tilted by :data:`RAW_WEIGHT`, analytic in ``probs``."""
    return law.tilt_on(probs, multi_cell_weights(), multi.SUPPORT)


def multi_gateaux_eif(functional: Callable[[Any], Any]) -> np.ndarray:
    """The Gateaux derivative of ``functional`` at every support point of the three-arm law."""
    return law.contamination_eif(functional, multi.PROBS, multi.SUPPORT)


def multi_first_row_of() -> np.ndarray:
    """Index of the first row of each support point of :func:`tests.discrete_law_multi.frame`."""
    counts = np.array([multi.COUNTS[w, a, y] for w, a, y in multi.SUPPORT])
    return np.concatenate([[0], np.cumsum(counts)[:-1]])
