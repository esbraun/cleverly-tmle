"""The weights a longitudinal fit divided by, per unit and node.

:func:`node_weights` reads what one fitted regimen stored: the weight each follower carried
at each node (the data's weight times the clever covariate), and the cumulative probability
of following the regimen up to that node before and after the fit's ``g_bounds``. It
computes nothing the fit did not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import BoolArray, FloatArray
from ..data.weighting import effective_sample_size
from ..exceptions import CapabilityError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sequential import RegimenFit


@dataclass(frozen=True)
class NodeWeights:
    """The weights one fitted regimen used at each node, one row per unit.

    Parameters
    ----------
    regimen : str
        The regimen's label.
    cause : str or None
        The cause the fit is for under competing events, ``None`` otherwise.
    horizon : int
        The last node the fit reads: the horizon of a cumulative-risk fit, the final node
        of an end-of-study outcome.
    repeat : int
        The cross-fitting repeat the weights come from, counted from one.
    n_repeats : int
        How many repeats the result holds.
    times : tuple of int
        The nodes, from one.
    node_kinds : tuple of str
        What decided the arm at each node: ``"label"``, ``"rule"``, ``"policy"``,
        ``"tilt"``, ``"mtp"``, or ``"held"``. ``"held"`` marks a later node of a held design
        whose regimen declares a one-node rule, policy or modified treatment policy; a
        static arm reports ``"label"`` there.
    used : ndarray of shape (n, T)
        The weight the fit used: the data's weight times the clever covariate, ``nan`` for
        a unit that is not a follower at that node.
    followed : ndarray of bool, shape (n, T)
        Which units were followers, the rows the fit's fluctuation read.
    cumulative_unbounded : ndarray of shape (n, T)
        The cumulative probability of following the regimen to the node, before the
        bounds.
    cumulative : ndarray of shape (n, T)
        The same after the bounds, which is what the fit divided by.
    bounds : tuple of float
        The ``g_bounds`` limits on the cumulative probability.
    unit_weights : bool
        Whether every observation weight was one, so that ``1 / bounds[0]`` is the largest
        weight a node can carry at a static or rule node.
    cross_fitted : bool
        Whether the probabilities are out-of-fold predictions; in-sample otherwise.
    reports_curve : bool
        Whether the fit reports a curve over horizons, so that :attr:`horizon` is one of
        several; otherwise it is the final time point of an end-of-study outcome.
    """

    regimen: str
    cause: str | None
    horizon: int
    repeat: int
    n_repeats: int
    times: tuple[int, ...]
    node_kinds: tuple[str, ...]
    used: FloatArray
    followed: BoolArray
    cumulative_unbounded: FloatArray
    cumulative: FloatArray
    bounds: tuple[float, float]
    unit_weights: bool
    cross_fitted: bool = False
    reports_curve: bool = False

    @property
    def effective_n(self) -> FloatArray:
        """Kish effective sample size of the used weights at each node."""
        return np.asarray(
            [
                effective_sample_size(self.used[self.followed[:, index], index], on_degenerate=0.0)
                for index in range(len(self.times))
            ],
            dtype=float,
        )

    @property
    def share_beyond_bounds(self) -> FloatArray:
        """The share of followers at each node whose cumulative probability the bounds moved."""
        shares = []
        for index in range(len(self.times)):
            mask = self.followed[:, index]
            moved = self.cumulative_unbounded[mask, index] != self.cumulative[mask, index]
            shares.append(float(np.mean(moved)) if moved.size else float("nan"))
        return np.asarray(shares, dtype=float)

    def to_frame(self, backend: str | None = None) -> Any:
        """Return one row per unit and node.

        Parameters
        ----------
        backend : {"pandas", "polars", "pyarrow"} or None, default=None
            Dataframe backend to return. ``None`` uses pandas when installed, then the first
            available backend.

        Returns
        -------
        dataframe
            The columns ``row``, ``time``, ``node_kind``, ``followed``, ``weight_used``,
            ``cumulative_unbounded`` and ``cumulative``, with the regimen, cause, horizon
            and repeat on every row.
        """
        from ..utils.frames import frame_from_dict

        n, nodes = self.used.shape
        columns: dict[str, list[Any]] = {
            "regimen": [self.regimen] * (n * nodes),
            "cause": [self.cause] * (n * nodes),
            "horizon": [self.horizon] * (n * nodes),
            "repeat": [self.repeat] * (n * nodes),
            "row": [row for _ in range(nodes) for row in range(n)],
            "time": [self.times[index] for index in range(nodes) for _ in range(n)],
            "node_kind": [self.node_kinds[index] for index in range(nodes) for _ in range(n)],
            "followed": self.followed.T.reshape(-1).tolist(),
            "weight_used": self.used.T.reshape(-1).tolist(),
            "cumulative_unbounded": self.cumulative_unbounded.T.reshape(-1).tolist(),
            "cumulative": self.cumulative.T.reshape(-1).tolist(),
        }
        return frame_from_dict(columns, backend=backend)


def _select(
    fits: dict[str, RegimenFit],
    regimen: str | None,
    horizon: int | None,
    cause: str | None,
) -> RegimenFit:
    labels = list(dict.fromkeys(fit.regimen.label for fit in fits.values()))
    if regimen is None:
        if len(labels) > 1:
            raise DataError(f"the result holds several regimens {labels}; name one with regimen=")
        regimen = labels[0]
    pool = [fit for fit in fits.values() if fit.regimen.label == regimen]
    if not pool:
        raise DataError(f"the result holds no regimen {regimen!r}; it holds {labels}")
    causes = list(dict.fromkeys(fit.cause for fit in pool))
    if cause is None:
        cause = causes[0]
    pool = [fit for fit in pool if fit.cause == cause]
    if not pool:
        raise DataError(f"the regimen {regimen!r} has no cause {cause!r}; it has {causes}")
    horizons = sorted({fit.horizon for fit in pool})
    chosen = horizons[-1] if horizon is None else horizon
    match = [fit for fit in pool if fit.horizon == chosen]
    if not match:
        raise DataError(f"the regimen {regimen!r} has no horizon {horizon!r}; it has {horizons}")
    return match[0]


def node_weights(
    result: Any,
    *,
    regimen: str | None = None,
    horizon: int | None = None,
    cause: str | None = None,
    repeat: int = 1,
) -> NodeWeights:
    """Read the weights a fitted regimen used at each node, one row per unit.

    Use it to see which units carried the weight at each decision point, and to see where
    the fit's ``g_bounds`` replaced a cumulative probability. The weight is the one the
    fit's fluctuation used, the data's weight times the clever covariate, so it equals
    ``1 / cumulative`` on a static plan with unit weights. Nothing is refit.

    Parameters
    ----------
    result : LongitudinalResult
        A fitted inverse-weighting or targeted longitudinal result.
    regimen : str or None, default=None
        The regimen's label. May be omitted when the result holds one.
    horizon : int or None, default=None
        The horizon of a curve fit; the largest when omitted.
    cause : str or None, default=None
        The cause, under competing events; the first when omitted.
    repeat : int, default=1
        The cross-fitting repeat, counted from one.

    Returns
    -------
    NodeWeights
        The weights and cumulative probabilities, per unit and node.

    Raises
    ------
    CapabilityError
        When the result is a g-computation or parametric g-formula fit, which divides by no
        treatment mechanism.
    DataError
        When the regimen, cause, horizon or repeat is not in the result.

    See Also
    --------
    cleverly.plotting.weights : The same weights as a figure.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly.datasets import make_longitudinal
    >>> from cleverly.longitudinal import LTMLE, node_weights
    >>> frame, _ = make_longitudinal(n=200, seed=0)
    >>> result = LTMLE(
    ...     {"always": 1},
    ...     n_folds=1,
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     censoring_learner=LogisticRegression(max_iter=1000),
    ... ).fit(
    ...     frame,
    ...     outcome="Y",
    ...     treatment=["A1", "A2"],
    ...     baseline=["W1", "W2"],
    ...     time_varying=[[], ["L2"]],
    ...     censoring=["C1", "C2"],
    ... )
    >>> weights = node_weights(result)
    >>> weights.times
    (1, 2)
    >>> bool(np.all(weights.used[weights.followed] > 0))
    True
    """
    estimator = result.config.estimator
    if estimator in ("gcomp", "parametric_gformula"):
        from .._refusal import RefusalReason

        raise CapabilityError(
            "a g-computation or parametric g-formula fit divides by no treatment mechanism, "
            "so it has no weights to show.",
            reason=RefusalReason.API_REDIRECTION,
            alternative="Fit SequentialIPW or LTMLE for weights.",
        )
    repeats = result.repeats
    if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= len(repeats):
        raise DataError(f"repeat must be an integer from 1 to {len(repeats)}; got {repeat!r}")
    fit = _select(repeats[repeat - 1].fits, regimen, horizon, cause)
    from ..validation.longitudinal import _node_kind

    nodes = fit.scored_nodes
    n = fit.obs_weights.shape[0]
    used = np.full((n, len(nodes)), np.nan)
    followed = np.zeros((n, len(nodes)), dtype=bool)
    raw = np.empty((n, len(nodes)))
    bounded = np.empty((n, len(nodes)))
    for index, node in enumerate(nodes):
        weight = fit.obs_weights * node.clever
        used[node.trained_on, index] = weight[node.trained_on]
        followed[:, index] = node.trained_on
        raw[:, index] = fit.cumulative_unbounded[:, node.time - 1]
        bounded[:, index] = fit.cumulative[:, node.time - 1]
    return NodeWeights(
        regimen=fit.regimen.label,
        cause=fit.cause,
        horizon=fit.horizon,
        repeat=repeat,
        n_repeats=len(repeats),
        times=tuple(node.time for node in nodes),
        node_kinds=tuple(_node_kind(fit, node.time) for node in nodes),
        used=used,
        followed=followed,
        cumulative_unbounded=raw,
        cumulative=bounded,
        bounds=(float(result.config.g_bounds[0]), float(result.config.g_bounds[1])),
        unit_weights=bool(np.all(fit.obs_weights == 1.0)),
        cross_fitted=int(getattr(result.config, "n_folds", 1)) > 1,
        reports_curve=bool(result.data.reports_curve),
    )


__all__ = ["NodeWeights", "node_weights"]
