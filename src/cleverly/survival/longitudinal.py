r"""The continuous-time longitudinal TMLE: time-varying treatment and covariates (X6 part (e)).

:class:`ContinuousLongitudinalTMLE` estimates

.. math::

    \psi_{r,j,\tau} = P^{G_r}(T \le \tau,\ \text{cause } j),

the risk of each cause by each target time had every unit followed regime :math:`r` at
baseline and at every decision visit, with censoring removed and the visit process left as it
would run under the regime (Rytgaard, Gerds and van der Laan 2022, display (8), with
:math:`Y = N^{d_j}(\tau)`).  A regime is a static level or a deterministic rule read at the
decision visits.  The law of the covariates is never modelled.

A node is a visit.  Node :math:`k`'s rows are the units that reached visit :math:`k` following
the regime through visit :math:`k - 1`; between two visits a unit's history is fixed, and the
death, visit and censoring intensities run on the pooled grid of observed times.  Each regime
runs one backward pass: node :math:`K^*` (the visit cap, :mod:`._longitudinal_nuisance`) is
walked first, its targeted curve gives node :math:`K^* - 1`'s continuation outcome, and so on
to the baseline node, each node walked once after every later node is final
(:mod:`._longitudinal_targeting`).  Node :math:`k`'s scores read only node :math:`k`'s
intensities and continuation, the mechanism and the later nodes' final values, so after the
baseline node every term of :math:`P_n D^*` is solved.

Each estimate reads the influence-curve variance :math:`P_n \hat D^2` (display (27)); the
switched variance of the baseline engine does not carry over, because the covariate term's
predictable variation needs the law of :math:`L(s)` the method avoids.  Every estimate reads
experimental until the registered study ``continuous-time-longitudinal`` passes (X6 part (e),
E2).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final, Literal

import numpy as np

from .._refusal import NO_ALTERNATIVE, RefusalReason
from .._typing import FloatArray, IntArray
from .._warnings import warn as _warn
from ..exceptions import CapabilityError, DataError, MethodConfigurationError, PositivityWarning
from ..learners._fitting import seed_unset
from ..utils.memory import check_max_memory_bytes
from ..utils.random import note_fresh_entropy
from . import _chunked
from . import _longitudinal_nuisance as nuisance
from . import _longitudinal_targeting as targeting
from ._longitudinal_data import ContinuousLongitudinalData
from ._longitudinal_nuisance import (
    HISTORY_RESTRICTION,
    ContinuationModel,
    Segments,
)
from ._longitudinal_targeting import NodeProblem, NodeState, NodeWalk
from ._memory import check_longitudinal_memory, longitudinal_peak_bytes
from ._targeting import default_cutoff
from .cox import CoxHazard, HazardSelection, StratifiedNelsonAalen, default_library
from .engine import PropensitySelection
from .longitudinal_result import FEW_FOLLOWERS, PAST_CAP_SHARE, build_result

__all__ = ["ContinuousLongitudinalRecord", "ContinuousLongitudinalTMLE"]


def _gap(text: str, roadmap: str | None = None) -> CapabilityError:
    return CapabilityError(
        text,
        reason=RefusalReason.IMPLEMENTATION_GAP,
        alternative=NO_ALTERNATIVE,
        roadmap=roadmap,
    )


def _t(time: float) -> str:
    return f"{time:g}"


def policy_refusal(kind: str) -> CapabilityError:
    """Refusal C7: a policy other than a static regime or a deterministic rule.

    Parameters
    ----------
    kind : str
        What was requested, in words.

    Returns
    -------
    CapabilityError
        The refusal.
    """
    return _gap(
        "ContinuousLongitudinalTMLE estimates static regimes and deterministic rules at every "
        f"decision visit; {kind} is not built in continuous time. The discrete-time hazard "
        "LTMLE fits known stochastic policies; their continuous-time submodel is not written."
    )


def levels_refusal(count: int) -> CapabilityError:
    """Refusal C14: a treatment with more than two levels.

    Parameters
    ----------
    count : int
        The levels.

    Returns
    -------
    CapabilityError
        The refusal.
    """
    return _gap(
        "ContinuousLongitudinalTMLE fits a binary treatment: the treatment model at each visit "
        "is a binary logistic library (_fit_propensity, engine.py:372-405). A treatment with "
        f"{count} levels is not built."
    )


IN_SAMPLE_LEARNER_TEXT: Final = (
    "an in-sample fit takes the default finite libraries; Lemma S8 of Rytgaard et al. (2022) "
    "needs nuisances of bounded variation, which a user learner fitted in sample does not "
    "guarantee. Use cross_fit=True."
)


def _user_learners_in_sample(estimator: ContinuousLongitudinalTMLE) -> None:
    """Refusal C13: a user learner for any nuisance with ``cross_fit=False``."""
    if estimator.cross_fit or estimator._default_nuisances_only():
        return
    raise MethodConfigurationError(IN_SAMPLE_LEARNER_TEXT)


@dataclass(frozen=True)
class ContinuousLongitudinalRecord:
    """What the continuous-time longitudinal fit chose and how its targeting ended.

    Parameters
    ----------
    hazards : tuple of HazardSelection
        The selection of each cause's hazard (the first fold's when cross-fitted, at the first
        regime's cap).
    visit : HazardSelection or None
        The selection of the visit intensity, ``None`` when no visit is modelled.
    censoring : HazardSelection
        The selection of the censoring hazard.
    propensity : PropensitySelection or None
        The selection of the treatment model at the visits, ``None`` for a user learner.
    regimes : tuple of str
        The regime labels, the reference first.
    caps : tuple of int
        Each regime's visit cap :math:`K^*`: visits after it are not modelled.
    past_cap_share : tuple of tuple of float
        Per regime and target time, the estimated share of the regime's followers with more
        than :math:`K^*` visits before the target time, inverse weighted to the regime: an
        estimate of :math:`P^{G_r}(N^v(\\tau) > K^*)`, which bounds the gap between the
        reported risk and the risk under the regime at every visit.
    followers_at_last : tuple of int
        Per regime, the units that followed it at every decision visit and are at risk at the
        last target time.
    steps : tuple of tuple of int
        Per regime, each node's accepted steps, the baseline node first.
    converged : bool
        Whether every node walk reached its cutoff.
    cutoff : float
        :math:`s_n`, the bound on each target's :math:`|P_n D^*| / \\hat\\sigma`; each node
        walk stops at :math:`s_n / N`, with :math:`N` the regime's nodes.
    node_scores : tuple of ndarray
        Per regime, ``(N, P)`` each node's final :math:`|P_n D^p_k| / \\hat\\sigma_p`.
    standardized_score : tuple of float
        Each target's final :math:`|P_n D^*| / \\hat\\sigma`, regime-major.
    initial_psi : tuple of float
        Each target's plug-in at the initial fits, regime-major.
    bounded_share : float
        The share of followers' weighted (row, time) cells at risk whose
        :math:`\\pi S_c` product was raised to the floor.
    weight_floor : float
        The floor on the product, as resolved.
    jump_bound_share : float
        The share of node rows with a death cell whose combined jump the engine bounded below
        one.
    visit_bound_share : float
        The share of node rows with a visit jump of one or more that the engine bounded at
        ``1 - JUMP_BOUND``, at a time where some unit at risk did not visit.
    history_restriction : str
        The model restriction of the history window.
    cross_fitted : bool
        Whether the nuisances were fitted out of fold.
    n_folds : int
        The cross-fitting folds, ``1`` in sample.
    grid_size : int
        The grid times up to the last target time.
    dropped_visits : int
        The visits read as following a death at the same time and dropped.
    """

    hazards: tuple[HazardSelection, ...]
    visit: HazardSelection | None
    censoring: HazardSelection
    propensity: PropensitySelection | None
    regimes: tuple[str, ...]
    caps: tuple[int, ...]
    past_cap_share: tuple[tuple[float, ...], ...]
    followers_at_last: tuple[int, ...]
    steps: tuple[tuple[int, ...], ...]
    converged: bool
    cutoff: float
    node_scores: tuple[FloatArray, ...]
    standardized_score: tuple[float, ...]
    initial_psi: tuple[float, ...]
    bounded_share: float
    weight_floor: float
    jump_bound_share: float
    visit_bound_share: float
    history_restriction: str
    cross_fitted: bool
    n_folds: int
    grid_size: int
    dropped_visits: int


# ------------------------------------------------------------------ the order rule


def visit_exit(deaths: FloatArray, visit: FloatArray) -> FloatArray:
    """The visit exit :math:`dE_v(u) = (1 - \\sum_l d\\Lambda^{d_l}(u))\\, d\\Lambda^v(u)`.

    Within one time a death comes before the visit (the order rule), so the visit at :math:`u`
    needs survival of every death at :math:`u`.

    Parameters
    ----------
    deaths : ndarray
        ``(J, c, M)`` the death jumps.
    visit : ndarray
        ``(c, M)`` the visit jumps.

    Returns
    -------
    ndarray
        ``(c, M)`` the visit exit.
    """
    return (1.0 - deaths.sum(axis=0)) * visit


def censoring_before(retention: FloatArray) -> FloatArray:
    """:math:`S_c(u^-)` from :math:`S_c(u)`: censoring at :math:`u` comes after the exits.

    Parameters
    ----------
    retention : ndarray
        ``(c, M)`` the censoring survival through each grid time.

    Returns
    -------
    ndarray
        ``(c, M)`` the survival just before each grid time.
    """
    return np.concatenate([np.ones((retention.shape[0], 1)), retention[:, :-1]], axis=1)


def node_cutoff(cutoff: float, nodes: int) -> float:
    """Each node walk's cutoff :math:`s_n / N`, so the nodes' scores add up to at most :math:`s_n`.

    Parameters
    ----------
    cutoff : float
        :math:`s_n`.
    nodes : int
        The regime's nodes :math:`N`.

    Returns
    -------
    float
        The node cutoff.
    """
    return cutoff / nodes


# ------------------------------------------------------------------ regimes


@dataclass(frozen=True)
class _Regime:
    label: str
    level: Any = None
    rule: Callable[[Any], Any] | None = None


def _read_regimes(regimes: Any) -> list[_Regime]:
    """The regimes as static levels or rules; C7 for any other policy."""
    from ..interventions import (
        Incremental,
        Intervention,
        ModifiedPolicy,
        RiskRatioTilt,
        Shift,
        Static,
        Stochastic,
    )

    if not isinstance(regimes, Mapping) or len(regimes) < 2:
        raise ValueError(
            "regimes is a mapping of at least two labels to a static level or a rule, such as "
            "{'always': 1, 'never': 0}"
        )
    out = []
    for label, value in regimes.items():
        if isinstance(value, Static):
            out.append(_Regime(str(label), level=value.level))
        elif isinstance(value, Stochastic):
            raise policy_refusal("a known stochastic policy")
        elif isinstance(value, (Shift, ModifiedPolicy)):
            raise policy_refusal("a modified treatment policy")
        elif isinstance(value, (Incremental, RiskRatioTilt)):
            raise policy_refusal("an odds tilt")
        elif isinstance(value, Intervention):
            raise policy_refusal(f"a {type(value).__name__} intervention")
        elif callable(value):
            out.append(_Regime(str(label), rule=value))
        elif isinstance(value, (Mapping, list, tuple, np.ndarray)):
            raise policy_refusal("an intervention on the visit intensity or a policy object")
        else:
            out.append(_Regime(str(label), level=value))
    return out


def rule_frame(data: ContinuousLongitudinalData, visits: IntArray) -> Any:
    """The frame a rule reads at the given visits.

    Parameters
    ----------
    data : ContinuousLongitudinalData
        The container.
    visits : ndarray
        Visit positions.

    Returns
    -------
    pandas.DataFrame
        ``time``, ``visit``, the baseline and time-varying covariates by name, and
        ``previous_treatment`` (the treatment at the unit's previous visit, the regime's
        value when the unit followed it there; at baseline, ``None``).
    """
    import pandas as pd

    columns: dict[str, Any] = {
        "time": data.visit_time[visits],
        "visit": data.visit_index[visits],
    }
    for j, name in enumerate(data.baseline_names):
        columns[name] = data.baseline[data.unit[visits], j]
    for j, name in enumerate(data.time_varying_names):
        columns[name] = data.covariates[visits, j]
    previous = data.previous_treatment[visits].astype(object)
    previous[data.visit_index[visits] == 0] = None
    columns["previous_treatment"] = previous
    return pd.DataFrame(columns)


# ------------------------------------------------------------------ the estimator


class ContinuousLongitudinalTMLE:
    r"""TMLE of cause-specific risks under treatment regimes with a time-varying treatment.

    Experimental (X6 part (e)): no registered study measures this engine yet.

    Parameters
    ----------
    horizons : sequence of float
        The target times :math:`\tau_1 < \dots < \tau_K`.
    regimes : mapping of str to level or callable
        Each regime: a treatment level held at every decision visit, or a rule that takes a
        frame of the decision visits (``time``, ``visit``, the covariates by name and
        ``previous_treatment``) and returns a level per row.
    reference : str or None
        The comparison regime of the risk differences; ``None`` takes the first.
    causes : mapping of int to str, or None
        The label of each nonzero event code.
    cross_fit : bool
        Fit every nuisance out of fold, with folds drawn per unit.
    n_folds : int
        Cross-fitting folds.
    learner_folds : int
        Folds of each library's cross-validation, drawn per unit.
    hazard_library : sequence or None
        Learners for every cause's hazard; ``None`` takes three Cox models.  Each fits on the
        segment rows with ``entry=``.
    visit_library : sequence or None
        Learners for the visit intensity; ``None`` takes the same three.
    censoring_library : sequence or None
        Learners for the censoring hazard; ``None`` takes the same three.
    treatment_learner : classifier or None
        The treatment model at the decision visits; ``None`` selects among a main-terms and L1
        logistic models by cross-validated log loss.
    continuation_learner : regressor or None
        The regression of each bin's soft event fraction on the window, the visit time and the
        bin, with ``fit(X, y, sample_weight)`` and ``predict(X)``; ``None`` selects between a
        main-terms and an L1 logistic model on the soft labels.
    norm : {"covariance", "variance", "euclidean"}
        The metric of each node walk, held at the initial fit.
    g_bounds : "auto" or float
        The floor on :math:`\prod \pi \cdot S_c(u^-)`; ``"auto"`` is
        :math:`5 / (\sqrt n \log n)`.
    max_iter : int
        The cap on each node walk's accepted steps.
    step_size : float
        The first step of each node walk.
    tol : float or None
        :math:`s_n`, the bound on each target's :math:`|P_n D^*| / \hat\sigma`; ``None`` takes
        :math:`1 / (\sqrt n \log n)`.
    alpha : float
        Significance level.
    random_state : int or None
        Seed of the folds, the selections and the lasso members; ``None`` draws one.
    simultaneous : bool
        Add a sup-t band over every risk and risk difference.
    max_memory_bytes : int or None
        The memory budget; the fit refuses before any learner when its predicted peak exceeds
        it.
    n_jobs : int or None
        Threads each node walk runs its row blocks on; results do not depend on it.

    See Also
    --------
    cleverly.survival.ContinuousSurvivalTMLE : The risks under a treatment fixed at baseline.
    cleverly.survival.ContinuousLongitudinalResult : Each regime's risks and the record.

    Examples
    --------
    One row per interval between visits: a unit's first row starts at 0, a later row at a
    visit, where ``l`` is measured and ``a`` decided.

        >>> import numpy as np
        >>> import pandas as pd
        >>> from cleverly.survival import ContinuousLongitudinalTMLE, CoxHazard
        >>> rng = np.random.default_rng(0)
        >>> rows = []
        >>> for unit in range(300):
        ...     w, l, a = rng.integers(2), rng.integers(2), rng.integers(2)
        ...     start, visit = 0.0, rng.exponential(3.0)
        ...     end = rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
        ...     if visit < min(end, 8.0):
        ...         rows.append((unit, 0.0, visit, 0, w, l, a))
        ...         l = rng.integers(2)
        ...         a = int(rng.random() < 0.3 + 0.4 * l)
        ...         start = visit
        ...         end = visit + rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
        ...     rows.append((unit, start, min(end, 8.0), int(end < 8.0), w, l, a))
        >>> frame = pd.DataFrame(rows, columns=["id", "start", "stop", "event", "w", "l", "a"])
        >>> cox = [CoxHazard("main")]
        >>> fit = ContinuousLongitudinalTMLE(
        ...     [2.0, 6.0],
        ...     regimes={"always": 1, "never": 0},
        ...     reference="never",
        ...     cross_fit=False,
        ...     hazard_library=cox,
        ...     visit_library=cox,
        ...     censoring_library=cox,
        ...     random_state=0,
        ... ).fit(
        ...     frame,
        ...     id="id",
        ...     start="start",
        ...     stop="stop",
        ...     event="event",
        ...     treatment="a",
        ...     baseline=["w"],
        ...     time_varying=["l"],
        ... )
        >>> fit.arms
        ('never', 'always')
        >>> list(fit)[:2]
        ['risk_regimen[never @ t=2]', 'risk_regimen[never @ t=6]']
        >>> fit["risk_regimen[always @ t=2]"].support.label
        'experimental'
    """

    def __init__(
        self,
        horizons: Sequence[float],
        *,
        regimes: Mapping[str, Any],
        reference: str | None = None,
        causes: Mapping[int, str] | None = None,
        cross_fit: bool = True,
        n_folds: int = 10,
        learner_folds: int = 5,
        hazard_library: Sequence[Any] | None = None,
        visit_library: Sequence[Any] | None = None,
        censoring_library: Sequence[Any] | None = None,
        treatment_learner: Any = None,
        continuation_learner: Any = None,
        norm: Literal["covariance", "variance", "euclidean"] = "covariance",
        g_bounds: Literal["auto"] | float = "auto",
        max_iter: int = 500,
        step_size: float = 0.1,
        tol: float | None = None,
        alpha: float = 0.05,
        random_state: int | None = None,
        simultaneous: bool = False,
        max_memory_bytes: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        times = tuple(float(t) for t in horizons)
        if not times or any(not np.isfinite(t) or t <= 0 for t in times):
            raise ValueError("horizons are positive finite target times, at least one")
        if any(b <= a for a, b in pairwise(times)):
            raise ValueError("horizons are strictly increasing")
        if norm not in ("covariance", "variance", "euclidean"):
            raise ValueError("norm is 'covariance', 'variance' or 'euclidean'")
        self.horizons = times
        self._regimes = _read_regimes(regimes)
        self.regimes = dict(regimes)
        labels = [r.label for r in self._regimes]
        self.reference = labels[0] if reference is None else str(reference)
        if self.reference not in labels:
            raise ValueError(f"reference {self.reference!r} is not one of the regimes {labels}")
        self.causes = None if causes is None else {int(k): str(v) for k, v in causes.items()}
        self.cross_fit = bool(cross_fit)
        self.n_folds = int(n_folds)
        self.learner_folds = int(learner_folds)
        self.hazard_library = tuple(hazard_library) if hazard_library else default_library()
        self.visit_library = tuple(visit_library) if visit_library else default_library()
        self.censoring_library = (
            tuple(censoring_library) if censoring_library else default_library()
        )
        self.treatment_learner = treatment_learner
        self.continuation_learner = continuation_learner
        self._defaults = (
            hazard_library is None
            and visit_library is None
            and censoring_library is None
            and treatment_learner is None
            and continuation_learner is None
        )
        self.norm = norm
        self.g_bounds = g_bounds
        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.tol = tol
        self.alpha = float(alpha)
        self.random_state = random_state
        self.simultaneous = bool(simultaneous)
        self.max_memory_bytes = check_max_memory_bytes(max_memory_bytes)
        self.n_jobs = 1 if n_jobs is None else int(n_jobs)
        _user_learners_in_sample(self)

    def _default_nuisances_only(self) -> bool:
        """Whether every nuisance is a shipped finite learner (no user learner)."""
        shipped = (CoxHazard, StratifiedNelsonAalen)
        libraries = (*self.hazard_library, *self.visit_library, *self.censoring_library)
        return (
            all(isinstance(member, shipped) for member in libraries)
            and self.treatment_learner is None
            and self.continuation_learner is None
        )

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        data: Any,
        *,
        id: str,
        start: str,
        stop: str,
        event: str,
        treatment: str,
        baseline: Sequence[str],
        time_varying: Sequence[str],
        decision: str | None = None,
    ) -> Any:
        """Fit the nuisances and target every regime's risks.

        Parameters
        ----------
        data : DataFrame
            The counting-process frame, one row per interval between visits
            (:meth:`ContinuousLongitudinalData.from_frame`).
        id : str
            The unit identifier.
        start : str
            The visit time that starts each row.
        stop : str
            The row's end.
        event : str
            The event code on a unit's last row, ``0`` for censored.
        treatment : str
            The treatment decided at each visit.
        baseline : sequence of str
            The baseline covariates.
        time_varying : sequence of str
            The covariates measured at each visit.
        decision : str or None
            A boolean column of the visits where the treatment may change.

        Returns
        -------
        ContinuousLongitudinalResult
            The targeted risks, their differences and the record.
        """
        container = ContinuousLongitudinalData.from_frame(
            data,
            id=id,
            start=start,
            stop=stop,
            event=event,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            causes=self.causes,
            decision=decision,
        )
        return self.fit_data(container)

    def fit_data(self, data: ContinuousLongitudinalData) -> Any:
        """Fit a :class:`ContinuousLongitudinalData` container.

        Parameters
        ----------
        data : ContinuousLongitudinalData
            The container.

        Returns
        -------
        ContinuousLongitudinalResult
            The targeted risks, their differences and the record.
        """
        return _Fit(self, data).run()

    def _resolved_seed(self) -> int:
        if self.random_state is not None:
            return int(self.random_state)
        note_fresh_entropy()
        return int(np.random.SeedSequence().generate_state(1)[0])

    def _copy(self, seed: int) -> ContinuousLongitudinalTMLE:
        estimator = copy.copy(self)
        estimator.random_state = seed
        return estimator


# ------------------------------------------------------------------ one fit


@dataclass
class _CapFits:
    """The nuisance fits of one visit cap: per outer fold, the models."""

    cap: int
    segments: Segments
    segment_of_visit: IntArray
    hazards: list[list[Any]]
    visit: list[Any]
    censoring: list[Any]
    p1: list[FloatArray]
    labels: IntArray
    records: dict[str, Any]


@dataclass
class _NodeBase:
    """A node's inputs that no pass changes."""

    k: int
    visits: IntArray
    units: IntArray
    segment: IntArray
    slot: IntArray
    exits: FloatArray
    problem: NodeProblem
    death_bounded: np.ndarray
    visit_bounded: np.ndarray


@dataclass
class _PassResult:
    walks: list[NodeWalk]
    bases: list[_NodeBase]
    continuation: list[Any]
    start: FloatArray


class _Fit:
    """One fit of :class:`ContinuousLongitudinalTMLE` to one container."""

    def __init__(self, estimator: ContinuousLongitudinalTMLE, data: ContinuousLongitudinalData):
        self.est = estimator
        self.data = data
        self.seed = estimator._resolved_seed()
        self.rng = np.random.default_rng(self.seed)

    # .......................................................... inputs

    def _levels(self) -> None:
        data = self.data
        levels, kind = nuisance.check_levels(np.asarray(data.treatment))
        if kind == "continuous":
            raise policy_refusal("a continuous treatment")
        if kind == "levels":
            raise levels_refusal(len(levels))
        self.levels = levels
        index = {level: code for code, level in enumerate(levels)}
        self.coded = np.array([index[v] for v in np.asarray(data.treatment).tolist()], dtype=int)
        self.level_index = index

    def _causes(self) -> None:
        data = self.data
        codes = sorted(int(c) for c in np.unique(data.event) if c != 0)
        declared = self.est.causes if self.est.causes is not None else data.causes
        if declared is not None:
            missing = set(codes) - set(declared)
            if missing:
                raise DataError(f"event codes {sorted(missing)} are not named by causes")
            absent = sorted(int(k) for k in declared if int(k) not in codes)
            if absent:
                raise _gap(
                    f"The declared causes {[declared[k] for k in absent]} have no event in the "
                    "data, so their risks are zero by construction and have no interval."
                )
            self.causes = {int(k): str(v) for k, v in sorted(declared.items())}
        else:
            self.causes = {1: "1"} if codes == [1] else {c: str(c) for c in codes}
        self.codes = list(self.causes)

    def _regime_values(self) -> None:
        """Each regime's coded level at every visit, and who follows it."""
        data = self.data
        V = data.visit_time.size
        decision = np.asarray(data.decision, dtype=bool)
        self.values: list[IntArray] = []
        self.followed: list[np.ndarray] = []
        heads = data.visit_index == 0
        unit_start = np.flatnonzero(heads)
        for regime in self.est._regimes:
            out = np.full(V, -1, dtype=int)
            if regime.rule is None:
                if regime.level not in self.level_index:
                    raise ValueError(
                        f"regime {regime.label!r} sets the treatment to {regime.level!r}, which "
                        f"is not one of its levels {self.levels}"
                    )
                out[:] = self.level_index[regime.level]
            else:
                at = np.flatnonzero(decision)
                given = np.asarray(regime.rule(rule_frame(data, at)), dtype=object).reshape(-1)
                if given.size != at.size:
                    raise ValueError(
                        f"rule {regime.label!r} returned {given.size} values for {at.size} visits"
                    )
                bad = [
                    value
                    for value in given.tolist()
                    if value is None
                    or (isinstance(value, float) and np.isnan(value))
                    or value not in self.level_index
                ]
                if bad:
                    raise DataError(
                        f"rule {regime.label!r} returned {bad[0]!r} at {len(bad)} visits; it "
                        f"must return one of the treatment levels {self.levels}."
                    )
                out[at] = [self.level_index[v] for v in given.tolist()]
                # Between decision visits the regime's value is its value at the last one.
                position = np.where(decision, np.arange(V), 0)
                position = np.maximum.accumulate(position)
                out = out[position]
            follow = self.coded == out
            # Followed through each visit: no departure at this or an earlier visit.
            departed = np.cumsum(~follow)
            base = np.repeat(
                departed[unit_start] - (~follow[unit_start]).astype(int),
                np.diff(np.r_[unit_start, V]),
            )
            self.values.append(out)
            self.followed.append(departed - base == 0)

    def _checks(self) -> None:
        """C11 and C12 on the units following each regime at baseline."""
        data = self.data
        first = data.first_visit
        tau0, last = self.est.horizons[0], self.est.horizons[-1]
        for r, regime in enumerate(self.est._regimes):
            follow = self.followed[r][first]
            T, D = data.time[follow], data.event[follow]
            for code, label in self.causes.items():
                times = T[code == D]
                if times.size == 0 or times.min() > tau0:
                    raise DataError(
                        f"The units following regime {regime.label!r} at baseline have no event "
                        f"of cause {label!r} at or before the first target time {_t(tau0)}, so "
                        "that risk's influence curve is zero and its interval undefined. Choose "
                        "a later first target time."
                    )
            followed = float(T.max()) if T.size else 0.0
            if followed < last:
                raise DataError(
                    f"The units following regime {regime.label!r} at baseline are followed only "
                    f"to {_t(followed)}, before the last target time {_t(last)}, so its risk "
                    "there would be an extrapolation of the hazard model with no data behind "
                    "it. Choose target times within every regime's follow-up."
                )

    def _caps(self) -> None:
        data = self.data
        last = self.est.horizons[-1]
        self.caps: list[int] = []
        for r, regime in enumerate(self.est._regimes):
            cap, fold, count = nuisance.visit_cap(
                data, self.followed[r], self.folds, last, self.est.cross_fit
            )
            visits_any = np.any((data.visit_index > 0) & (data.visit_time < last))
            if cap < 1 and visits_any and count >= 0 and fold >= 0:
                raise DataError(
                    f"regime {regime.label!r}: training fold {fold} holds {count} units that "
                    f"follow it at baseline and visit before {_t(last)}, fewer than "
                    f"{nuisance.VISIT_CAP_MIN_FOLLOWERS}, so no visit after baseline can be "
                    "modelled. Use fewer folds, more units or an earlier last target time."
                )
            self.caps.append(int(cap))

    # .......................................................... nuisances

    def _inner(self, units: np.ndarray) -> IntArray:
        """Inner selection labels per unit for the training units ``units``, drawn now."""
        labels = np.full(self.data.n, -1, dtype=int)
        labels[units] = nuisance.unit_folds(
            self.data.event[units], self.est.learner_folds, self.rng
        )
        return labels

    def _fit_cap(self, cap: int) -> _CapFits:
        est, data = self.est, self.data
        last = est.horizons[-1]
        seg = nuisance.segments(data, cap, last)
        X = nuisance.design(data, self.coded, seg.visit)
        delta = nuisance.shift(data)
        seed = self.seed
        hazard_library = tuple(seed_unset(m, seed) for m in est.hazard_library)
        visit_library = tuple(seed_unset(m, seed) for m in est.visit_library)
        censoring_library = tuple(seed_unset(m, seed) for m in est.censoring_library)
        segment_of_visit = np.full(data.visit_time.size, -1, dtype=int)
        segment_of_visit[seg.visit] = np.arange(seg.size)
        # Treatment rows: the decision visits of the segments.
        decide = np.zeros(data.visit_time.size, dtype=bool)
        decide[seg.visit] = True
        decide &= np.asarray(data.decision, dtype=bool)
        TX = nuisance.treatment_design(data, self.coded, np.arange(data.visit_time.size))
        weight = data._weight
        row_weight = None if weight is None else weight[seg.unit]
        visit_weight = None if weight is None else weight[data.unit]
        has_visit = cap > 0
        visit_rows = seg.k < cap
        labels = np.unique(self.folds) if est.cross_fit else np.array([0])
        hazards, visits, censors, p1s = [], [], [], []
        records: dict[str, Any] = {}
        for position, fold in enumerate(labels):
            train_units = self.folds != fold if est.cross_fit else np.ones(data.n, dtype=bool)
            inner = self._inner(np.flatnonzero(train_units))
            train = train_units[seg.unit]
            inner_rows = inner[seg.unit]
            fold_hazards = []
            for code in self.codes:
                selection = nuisance.fit_hazard(
                    hazard_library,
                    X[train],
                    seg.stop[train],
                    (seg.exit[train] == code).astype(float),
                    seg.entry[train],
                    inner_rows[train],
                    None if row_weight is None else row_weight[train],
                )
                fold_hazards.append(selection.model)
                if position == 0:
                    records.setdefault("hazards", []).append(selection)
            hazards.append(fold_hazards)
            if has_visit:
                rows = train & visit_rows
                stop_v = np.where(seg.exit > 0, seg.stop - delta, seg.stop)
                selection = nuisance.fit_hazard(
                    visit_library,
                    X[rows],
                    stop_v[rows],
                    (seg.exit[rows] == -1).astype(float),
                    seg.entry[rows],
                    inner_rows[rows],
                    None if row_weight is None else row_weight[rows],
                )
                visits.append(selection.model)
                if position == 0:
                    records["visit"] = selection
            else:
                visits.append(None)
            stop_c = np.where(seg.exit != 0, seg.stop - delta, seg.stop)
            selection = nuisance.fit_hazard(
                censoring_library,
                X[train],
                stop_c[train],
                (seg.exit[train] == 0).astype(float),
                seg.entry[train] - delta,
                inner_rows[train],
                None if row_weight is None else row_weight[train],
            )
            censors.append(selection.model)
            if position == 0:
                records["censoring"] = selection
            train_visits = decide & train_units[data.unit]
            predict = np.zeros(data.visit_time.size, dtype=bool)
            predict[seg.visit] = True
            p1 = np.full(data.visit_time.size, np.nan)
            values, chosen = nuisance.fit_treatment(
                est.treatment_learner,
                TX,
                self.coded,
                train_visits,
                predict,
                inner[data.unit][train_visits],
                seed,
                visit_weight,
            )
            p1[predict] = values
            p1s.append(p1)
            if position == 0:
                records["propensity"] = chosen
        return _CapFits(cap, seg, segment_of_visit, hazards, visits, censors, p1s, labels, records)

    # .......................................................... nodes

    def _model_index(self, fits: _CapFits, units: IntArray, fixed: int | None) -> IntArray:
        if fixed is not None:
            return np.full(units.size, fixed, dtype=int)
        if not self.est.cross_fit:
            return np.zeros(units.size, dtype=int)
        return np.searchsorted(fits.labels, self.folds[units])

    def _cumulative(self, models: list[Any], index: IntArray, X: FloatArray) -> FloatArray:
        """``(c, M)`` each row's cumulative hazard on the grid, from its fold's model."""
        out = np.zeros((X.shape[0], self.grid.size))
        for m in np.unique(index):
            rows = np.flatnonzero(index == m)
            out[rows] = models[m].cumulative_hazard(X[rows], self.grid)
        return out

    def _jumps(self, cumulative: FloatArray, entry: FloatArray, closed: bool) -> FloatArray:
        jumps = np.diff(np.concatenate([np.zeros((cumulative.shape[0], 1)), cumulative], 1), axis=1)
        if closed:
            jumps[self.grid[None, :] < entry[:, None]] = 0.0
        else:
            jumps[self.grid[None, :] <= entry[:, None]] = 0.0
        return jumps

    def _prefix(
        self, fits: _CapFits, r: int, units_mask: np.ndarray, fixed: int | None
    ) -> FloatArray:
        """``(V,)`` :math:`\\prod_{m<k} \\pi_m S_{c,m}` at each visit, on the observed history."""
        data, seg = self.data, fits.segments
        V = data.visit_time.size
        prefix = np.full(V, np.nan)
        rows = np.flatnonzero(units_mask[seg.unit])
        units = seg.unit[rows]
        index = self._model_index(fits, units, fixed)
        X = nuisance.design(data, self.coded, seg.visit[rows])
        cum = self._cumulative(fits.censoring, index, X)
        dC = self._jumps(cum, seg.entry[rows], closed=True)
        end = np.where(seg.next_visit[rows] >= 0, seg.stop[rows], np.inf)
        inside = self.grid[None, :] < end[:, None]
        retained = np.prod(np.where(inside, 1.0 - np.clip(dC, 0.0, 1.0), 1.0), axis=1)
        visits = seg.visit[rows]
        decision = np.asarray(data.decision, dtype=bool)[visits]
        p1 = np.array([fits.p1[i][v] for i, v in zip(index, visits, strict=True)])
        observed = self.coded[visits]
        pi = np.where(decision, np.where(observed == 1, p1, 1.0 - p1), 1.0)
        factor = pi * retained
        # Visits of a unit are contiguous, so the prefix of visit k is the product before it.
        order = np.argsort(visits, kind="stable")
        visits, factor, units = visits[order], factor[order], units[order]
        running = 1.0
        for i in range(visits.size):
            if data.visit_index[visits[i]] == 0:
                running = 1.0
            prefix[visits[i]] = running
            running = running * factor[i]
        return prefix

    def _node_base(
        self,
        fits: _CapFits,
        r: int,
        k: int,
        units_mask: np.ndarray,
        fixed: int | None,
        prefix: FloatArray,
    ) -> _NodeBase | None:
        est, data, seg = self.est, self.data, fits.segments
        cap = fits.cap
        J = len(self.codes)
        candidates = np.flatnonzero(fits.segment_of_visit >= 0)
        at_k = candidates[data.visit_index[candidates] == k]
        at_k = at_k[units_mask[data.unit[at_k]]]
        if k > 0:
            at_k = at_k[self.followed[r][at_k - 1]]
        if at_k.size == 0:
            return None
        units = data.unit[at_k]
        rows = fits.segment_of_visit[at_k]
        slot = self.values[r][at_k]
        index = self._model_index(fits, units, fixed)
        X_slot = nuisance.design(data, self.coded, at_k, slot=slot)
        entry = seg.entry[rows]
        stop = seg.stop[rows]
        exit_code = seg.exit[rows]
        c, M = at_k.size, self.grid.size
        has_visit = k < cap
        E = J + int(has_visit)
        deaths = np.empty((J, c, M))
        for j in range(J):
            cum = self._cumulative([h[j] for h in fits.hazards], index, X_slot)
            deaths[j] = self._jumps(cum, entry, closed=False)
        bounded = _chunked.bound_jumps(deaths[None])
        death_bounded = np.zeros(c, dtype=bool) if bounded is None else bounded[0].any(axis=-1)
        exits = np.empty((E, c, M))
        exits[:J] = deaths
        at_risk = (self.grid[None, :] > entry[:, None]) & (self.grid[None, :] <= stop[:, None])
        events = np.zeros((E, c, M), dtype=bool)
        position = np.searchsorted(self.grid, stop)
        inside = (position < M) & (stop <= self.grid[-1])
        for j, code in enumerate(self.codes):
            hit = np.flatnonzero(inside & (exit_code == code))
            events[j, hit, position[hit]] = True
        follow = self.coded[at_k] == slot
        visit_bounded = np.zeros(c, dtype=bool)
        visit_at = np.full(c, -1, dtype=int)
        if has_visit:
            cum = self._cumulative(fits.visit, index, X_slot)
            dV = self._jumps(cum, entry, closed=False)
            factual = dV.copy()
            others = np.flatnonzero(~follow)
            if others.size:
                X_obs = nuisance.design(data, self.coded, at_k[others])
                cum = self._cumulative(fits.visit, index[others], X_obs)
                factual[others] = self._jumps(cum, entry[others], closed=False)
            exits_here = events[:J].any(axis=0)
            hit = np.flatnonzero(inside & (exit_code == -1))
            visit_seen = np.zeros((c, M), dtype=bool)
            visit_seen[hit, position[hit]] = True
            exits_here = exits_here | visit_seen
            # A visit jump of one is kept only where every row at risk whose factual jump
            # reaches one exits there (a scheduled visit); elsewhere it is bounded.
            reach = (factual >= 1.0) & at_risk
            scheduled = reach.any(axis=0) & ~(reach & ~exits_here).any(axis=0)
            over = dV >= 1.0
            keep = over & scheduled[None, :]
            clip = over & ~scheduled[None, :]
            dV = np.where(keep, 1.0, np.where(clip, 1.0 - _chunked.JUMP_BOUND, dV))
            visit_bounded = clip.any(axis=1)
            exits[J] = visit_exit(deaths, dV)
            events[J] = visit_seen
            visit_at = np.searchsorted(self.grid[self.visit_grid], stop)
            visit_at = np.where((exit_code == -1) & (visit_at < self.visit_grid.size), visit_at, -1)
        # The slot weight: prod_{m<k} pi_m S_{c,m}, times pi(d | s_k), times S_c(u-) at the slot.
        cum = self._cumulative(fits.censoring, index, X_slot)
        dC = self._jumps(cum, entry, closed=True)
        retention = np.cumprod(1.0 - np.clip(dC, 0.0, 1.0), axis=1)
        before = censoring_before(retention)
        decision = np.asarray(data.decision, dtype=bool)[at_k]
        p1 = np.array([fits.p1[i][v] for i, v in zip(index, at_k, strict=True)])
        pi = np.where(decision, np.where(slot == 1, p1, 1.0 - p1), 1.0)
        product = (prefix[at_k] * pi)[:, None] * before
        floor = self.floor
        weight = 1.0 / np.maximum(product, floor)
        weight[self.grid[None, :] <= entry[:, None]] = 0.0
        relevant = at_risk & follow[:, None]
        self.floor_cells[0] += int(np.count_nonzero(relevant))
        self.floor_cells[1] += int(np.count_nonzero(relevant & (product < floor)))
        unit_weight = np.ones(c) if data._weight is None else data._weight[units]
        P = len(self.targets)
        problem = NodeProblem(
            rows=units,
            unit_weight=unit_weight,
            total_weight=self.total_weight,
            at_risk=at_risk,
            events=events,
            follow=follow,
            slot_weight=weight,
            cause=np.array([j for j, _ in self.targets], dtype=int),
            tau=np.array([self.tau_pos[q] for _, q in self.targets], dtype=int),
            tau_bin=np.array([q for _, q in self.targets], dtype=int),
            bin_of=self.bin_of,
            n_bins=len(est.horizons),
            visit_grid=self.visit_grid if has_visit else np.empty(0, dtype=int),
            visit_at=visit_at,
            next_value=np.zeros((c, P)),
        )
        return _NodeBase(k, at_k, units, rows, slot, exits, problem, death_bounded, visit_bounded)

    def _continuation_input(
        self, fits: _CapFits, r: int, base: _NodeBase, model: Any
    ) -> FloatArray:
        """``(J, c, U, B)`` the bin hazards of a node's rows at every visit-grid time."""
        data = self.data
        c = base.visits.size
        U = self.visit_grid.size
        J = len(self.codes)
        B = len(self.est.horizons)
        X = nuisance.design(data, self.coded, base.visits, slot=base.slot)
        u = self.grid[self.visit_grid]
        out = np.zeros((J, c, U, B))
        entry = data.visit_time[base.visits]
        models = model if isinstance(model, list) else None
        index = (
            self._model_index(fits, base.units, None)
            if models is not None
            else np.zeros(c, dtype=int)
        )
        for m in np.unique(index):
            rows = np.flatnonzero(index == m)
            source = models[m] if models is not None else model
            Xr = np.repeat(X[rows], U, axis=0)
            ur = np.tile(u, rows.size)
            lam = source.hazards(Xr, ur).reshape(J, rows.size, U, B)
            out[:, rows] = lam
        out[:, (u[None, :] <= entry[:, None])] = 0.0
        return out

    def _fit_continuation(
        self, base_next: _NodeBase, labels: FloatArray, base: _NodeBase
    ) -> ContinuationModel:
        """Fit node ``k``'s continuation on node ``k + 1``'s rows and soft outcomes."""
        data = self.data
        previous = base_next.visits - 1
        X = nuisance.design(data, self.coded, previous)
        u = data.visit_time[base_next.visits]
        weight = base_next.problem.unit_weight
        return ContinuationModel.fit(
            self.est.continuation_learner,
            X,
            u,
            labels,
            weight,
            np.asarray(self.est.horizons),
            self.continuation_folds[base_next.units],
            self.seed,
        )

    def _pass(
        self,
        fits: _CapFits,
        r: int,
        units_mask: np.ndarray,
        fixed: int | None,
        continuation: Literal["fit"] | list[list[Any]],
        walk: bool,
        sigma: FloatArray | None = None,
        covariance: FloatArray | None = None,
        cutoff: float = 0.0,
        bases: list[_NodeBase | None] | None = None,
    ) -> _PassResult:
        """One backward pass of regime ``r`` over the units ``units_mask``."""
        est = self.est
        cap = fits.cap
        if bases is None:
            prefix = self._prefix(fits, r, units_mask, fixed)
            bases = [self._node_base(fits, r, k, units_mask, fixed, prefix) for k in range(cap + 1)]
        P = len(self.targets)
        walks: list[NodeWalk | None] = [None] * (cap + 1)
        models: list[Any] = [None] * (cap + 1)
        following: _NodeBase | None = None
        soft: FloatArray | None = None
        following_start: FloatArray | None = None
        for k in range(cap, -1, -1):
            base = bases[k]
            if base is None:
                following, soft, following_start = None, None, None
                continue
            problem = base.problem
            lam = None
            if problem.has_visit:
                if following is None or soft is None:
                    # No row reached the next visit: the visit exit has no continuation.
                    lam = np.zeros(
                        (len(self.codes), base.visits.size, self.visit_grid.size, len(est.horizons))
                    )
                else:
                    if continuation == "fit":
                        model = self._fit_continuation(following, soft, base)
                        models[k] = model
                        lam = self._continuation_input(fits, r, base, model)
                    else:
                        lam = self._continuation_input(fits, r, base, continuation[k])
                # Z_{k+1} at each follower's observed next visit.
                next_value = np.zeros((base.visits.size, P))
                if following is not None and following_start is not None:
                    where = {int(v): i for i, v in enumerate(following.visits)}
                    nexts = fits.segments.next_visit[base.segment]
                    for i, v in enumerate(nexts):
                        j = where.get(int(v))
                        if j is not None and problem.follow[i]:
                            next_value[i] = following_start[j]
                problem = _replace_next(problem, next_value)
            state = NodeState(base.exits.copy(), lam)
            record = targeting.walk_node(
                problem,
                state,
                sigma=np.ones(P) if sigma is None else sigma,
                covariance=np.eye(P) if covariance is None else covariance,
                cutoff=cutoff,
                norm=est.norm,
                step=est.step_size,
                max_steps=est.max_iter,
                walk=walk,
                n_jobs=est.n_jobs,
            )
            walks[k] = record
            soft = targeting.soft_outcome(problem, record.state)
            following, following_start = base, record.start
        start = np.zeros((self.data.n, P))
        base0 = bases[0]
        walk0 = walks[0]
        assert base0 is not None and walk0 is not None
        start[base0.units] = walk0.start
        return _PassResult(
            [w for w in walks if w is not None],
            [b for b in bases if b is not None],
            models,
            start,
        )

    # .......................................................... run

    def _influence(self, result: _PassResult) -> tuple[FloatArray, FloatArray, list[FloatArray]]:
        n = self.data.n
        weight = self.unit_weights
        psi = (weight @ result.start) / self.total_weight
        curve = result.start - psi
        scores = []
        for base, record in zip(result.bases, result.walks, strict=True):
            np.add.at(curve, base.units, record.parts)
            scores.append(targeting.score_of(base.problem, record.parts))
        del n
        return psi, curve, scores

    def run(self) -> Any:
        est, data = self.est, self.data
        self._levels()
        self._causes()
        n = data.n
        self.unit_weights = np.ones(n) if data._weight is None else data._weight
        self.total_weight = float(self.unit_weights.sum())
        last = est.horizons[-1]
        if est.cross_fit:
            self.folds = nuisance.unit_folds(data.event, est.n_folds, self.rng)
        else:
            self.folds = np.zeros(n, dtype=int)
        self._regime_values()
        self._checks()
        self._caps()
        visits_kept = nuisance.kept_visits(data, last)
        times = np.concatenate(
            [
                data.time[data.time <= last],
                data.visit_time[visits_kept & (data.visit_index > 0)],
                np.asarray(est.horizons),
            ]
        )
        self.grid = np.unique(times)
        self.tau_pos = np.searchsorted(self.grid, est.horizons)
        self.bin_of = np.searchsorted(np.asarray(est.horizons), self.grid, side="left")
        later = visits_kept & (data.visit_index > 0)
        self.visit_grid = np.searchsorted(self.grid, np.unique(data.visit_time[later]))
        self.targets = [(j, q) for j in range(len(self.codes)) for q in range(len(est.horizons))]
        self.floor = (
            float(5.0 / (np.sqrt(n) * np.log(n))) if est.g_bounds == "auto" else float(est.g_bounds)
        )
        self.floor_cells = [0, 0]
        # Before any learner: the memory check.
        rows = max(int(np.count_nonzero(data.visit_index == k)) for k in range(max(self.caps) + 1))
        predicted = longitudinal_peak_bytes(
            rows=rows,
            times=self.grid.size,
            causes=len(self.codes),
            visit_times=self.visit_grid.size,
            targets=len(self.targets),
            bins=len(est.horizons),
            units=n * len(est._regimes),
            covariates=nuisance.design(data, self.coded, np.arange(1)).shape[1],
            block_cells=targeting.NODE_CELLS,
        )
        check_longitudinal_memory(predicted, est.max_memory_bytes)
        fits = {cap: self._fit_cap(cap) for cap in sorted(set(self.caps))}
        # The continuation's selection folds, by unit, drawn after the nuisance fits so the
        # draws before them follow the baseline engine's order.
        self.continuation_folds = nuisance.unit_folds(
            data.event, max(2, est.learner_folds), self.rng
        )
        cutoff = default_cutoff(n) if est.tol is None else float(est.tol)
        everyone = np.ones(n, dtype=bool)
        psis, curves, initial_psi, node_scores, steps = [], [], [], [], []
        converged = True
        totals: list[FloatArray] = []
        death_rows = visit_rows = total_rows = 0
        for r, _regime in enumerate(est._regimes):
            fit = fits[self.caps[r]]
            if est.cross_fit:
                stitched: list[list[Any]] = [[None] * fit.labels.size for _ in range(fit.cap + 1)]
                for position, fold in enumerate(fit.labels):
                    train = self.folds != fold
                    trained = self._pass(fit, r, train, position, "fit", walk=False)
                    for k, model in enumerate(trained.continuation):
                        stitched[k][position] = model
                source: Literal["fit"] | list[list[Any]] = [
                    [m if m is not None else _empty_model(self) for m in per] for per in stitched
                ]
            else:
                source = "fit"
            prefix = self._prefix(fit, r, everyone, None)
            bases = [self._node_base(fit, r, k, everyone, None, prefix) for k in range(fit.cap + 1)]
            initial = self._pass(fit, r, everyone, None, source, walk=False, bases=bases)
            psi0, curve0, _ = self._influence(initial)
            weight = self.unit_weights / self.total_weight
            centred = curve0
            covariance = (centred * weight[:, None]).T @ centred
            sigma = np.sqrt(np.diag(covariance))
            N = len(initial.walks)
            targeted = self._pass(
                fit,
                r,
                everyone,
                None,
                source,
                walk=True,
                sigma=sigma,
                covariance=covariance,
                cutoff=node_cutoff(cutoff, N),
                bases=bases,
            )
            psi, curve, scores = self._influence(targeted)
            total = np.abs(np.sum(scores, axis=0)) / np.where(sigma > 0, sigma, 1.0)
            psis.append(psi)
            curves.append(curve)
            initial_psi.append(psi0)
            node_scores.append(np.array([w.standardized for w in targeted.walks]))
            steps.append(tuple(int(w.steps) for w in targeted.walks))
            converged = converged and all(w.converged for w in targeted.walks)
            for base in targeted.bases:
                death_rows += int(base.death_bounded.sum())
                visit_rows += int(base.visit_bounded.sum())
                total_rows += base.visits.size
            totals.append(total)
        if death_rows or visit_rows:
            _warn(
                f"The initial hazard fits put a combined jump of one or more at {death_rows} "
                f"node rows' death cells and a visit jump of one or more at {visit_rows} node "
                "rows where some unit at risk did not visit; the engine bounded each at "
                f"1 - {_chunked.JUMP_BOUND:g} and the fit's estimates read experimental.",
                PositivityWarning,
            )
        if not converged:
            from ..exceptions import ConvergenceWarning

            _warn(
                "A node walk stopped on its step cap without reaching its cutoff; the estimate "
                "is kept and the record lists every node's standardized score.",
                ConvergenceWarning,
            )
        shares, at_last = self._past_cap(fits), self._followers_at_last()
        first = fits[self.caps[0]]
        record = ContinuousLongitudinalRecord(
            hazards=tuple(first.records["hazards"]),
            visit=first.records.get("visit"),
            censoring=first.records["censoring"],
            propensity=first.records.get("propensity"),
            regimes=tuple(r.label for r in est._regimes),
            caps=tuple(self.caps),
            past_cap_share=shares,
            followers_at_last=at_last,
            steps=tuple(steps),
            converged=converged,
            cutoff=float(cutoff),
            node_scores=tuple(node_scores),
            standardized_score=tuple(float(v) for t in totals for v in t),
            initial_psi=tuple(float(v) for p in initial_psi for v in p),
            bounded_share=self.floor_cells[1] / self.floor_cells[0] if self.floor_cells[0] else 0.0,
            weight_floor=self.floor,
            jump_bound_share=death_rows / total_rows if total_rows else 0.0,
            visit_bound_share=visit_rows / total_rows if total_rows else 0.0,
            history_restriction=HISTORY_RESTRICTION,
            cross_fitted=est.cross_fit,
            n_folds=int(np.unique(self.folds).size) if est.cross_fit else 1,
            grid_size=int(self.grid.size),
            dropped_visits=int(data.dropped_visits),
        )
        return build_result(
            self,
            psis,
            curves,
            record,
            jump_bounded=bool(death_rows or visit_rows),
        )

    def _followers_at_last(self) -> tuple[int, ...]:
        data, est = self.data, self.est
        last = est.horizons[-1]
        out = []
        kept = nuisance.kept_visits(data, last)
        for r, regime in enumerate(est._regimes):
            ok = np.ones(data.n, dtype=bool)
            np.logical_and.at(ok, data.unit[kept], self.followed[r][kept])
            count = int(np.count_nonzero(ok & (data.time >= last)))
            out.append(count)
            if count < FEW_FOLLOWERS:
                _warn(
                    f"{count} units follow regime {regime.label!r} at every decision visit and "
                    f"are at risk at the last target time {_t(last)}, fewer than {FEW_FOLLOWERS}: "
                    "their inverse weights are large, and the regime's estimates read "
                    "experimental.",
                    UserWarning,
                )
        return tuple(out)

    def _past_cap(self, fits: dict[int, _CapFits]) -> tuple[tuple[float, ...], ...]:
        r"""Per regime and target time, :math:`\hat P^{G_r}(N^v(\tau) > K^*)` (W2)."""
        data, est = self.data, self.est
        out = []
        for r, regime in enumerate(est._regimes):
            fit = fits[self.caps[r]]
            K = fit.cap
            prefix = self._prefix(fit, r, np.ones(data.n, dtype=bool), None)
            shares = []
            # Visit K + 1 of a unit that followed r through visit K: a unit's visits are
            # contiguous, so the row before it is the unit's visit K.
            at = np.flatnonzero(data.visit_index == K + 1)
            at = at[self.followed[r][at - 1]]
            for tau in est.horizons:
                if K == 0 and not np.any((data.visit_index > 0) & (data.visit_time < tau)):
                    shares.append(0.0)
                    continue
                chosen = at[data.visit_time[at] < tau]
                if chosen.size == 0:
                    shares.append(0.0)
                    continue
                previous = chosen - 1
                decision = np.asarray(data.decision, dtype=bool)[previous]
                index = self._model_index(fit, data.unit[previous], None)
                p1 = np.array([fit.p1[i][v] for i, v in zip(index, previous, strict=True)])
                slot = self.values[r][previous]
                pi = np.where(decision, np.where(slot == 1, p1, 1.0 - p1), 1.0)
                X = nuisance.design(data, self.coded, previous, slot=slot)
                cum = self._cumulative(fit.censoring, index, X)
                dC = self._jumps(cum, data.visit_time[previous], closed=True)
                u = data.visit_time[chosen]
                inside = self.grid[None, :] < u[:, None]
                retained = np.prod(np.where(inside, 1.0 - np.clip(dC, 0.0, 1.0), 1.0), axis=1)
                product = prefix[previous] * pi * retained
                weight = self.unit_weights[data.unit[chosen]]
                share = float(np.sum(weight / np.maximum(product, self.floor)) / self.total_weight)
                shares.append(share)
            out.append(tuple(shares))
            worst = int(np.argmax(shares))
            if shares[worst] > PAST_CAP_SHARE:
                _warn(
                    f"An estimated {shares[worst]:.3f} of regime {regime.label!r}'s followers "
                    f"visit more than K* = {K} times before {_t(est.horizons[worst])}. Visits "
                    "after the cap are not modelled, and that share bounds the gap between the "
                    "reported risk and the risk under the regime at every visit.",
                    UserWarning,
                )
        return tuple(out)


def _replace_next(problem: NodeProblem, next_value: FloatArray) -> NodeProblem:
    from dataclasses import replace

    return replace(problem, next_value=next_value)


@dataclass
class _ZeroContinuation:
    """A continuation with no event in any bin, for a fold whose node had no follower."""

    J: int
    B: int

    def hazards(self, x: FloatArray, u: FloatArray) -> FloatArray:
        return np.zeros((self.J, x.shape[0], self.B))


def _empty_model(fit: _Fit) -> _ZeroContinuation:
    return _ZeroContinuation(len(fit.codes), len(fit.est.horizons))
