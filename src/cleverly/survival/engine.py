r"""Continuous-time survival and competing-risks TMLE for a binary baseline treatment.

:class:`ContinuousSurvivalTMLE` estimates :math:`E\,F_j(t_k \mid a, L)`, the absolute risk of
each cause at each target time had every unit received arm :math:`a`, from one row per unit with
a follow-up time that is not binned.  It fits the cause-specific hazards, the censoring hazard and
the propensity, then runs the one-step TMLE of Rytgaard and van der Laan (2024) on every target at
once (:mod:`cleverly.survival._targeting`).

Two departures are stated in the design (X91 part G, rev 2):

- each arm's hazard is tilted for every unit, as the source's submodel tilts the hazard function;
  ``concrete`` 1.0.5 tilts only units whose received arm is the target's arm, which is not doubly
  robust;
- an event and a censoring at one time are ordered event first, and the censoring hazard leaves
  that event out of its risk set.  The engine shifts each event time below the censoring times it
  ties with before it fits the censoring hazard.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from .._refusal import NO_ALTERNATIVE, RefusalReason
from .._support import Support
from .._typing import FloatArray, IntArray
from .._warnings import suppressed as _suppressed
from .._warnings import warn as _warn
from ..exceptions import (
    CapabilityError,
    CleverlyError,
    ConvergenceWarning,
    DataError,
    MethodConfigurationError,
)
from ..inference.multiplier import DEFAULT_BAND_SCALE, DEFAULT_KIND, DEFAULT_N_REPLICATES
from ..learners._fitting import seed_unset
from ..utils.memory import available_memory, check_max_memory_bytes
from ..utils.parallel import available_cpus, map_parallel
from ..utils.random import note_fresh_entropy
from . import _chunked
from ._chunked import HazardSource, InitialHazards, blocks, chunk_size
from ._memory import (
    BOOTSTRAP_HELD,
    bootstrap_held,
    check_memory,
    design_width,
    estimate_count,
    overflow_rows,
    peak_bytes,
    resident_count,
    train_size,
)
from ._targeting import (
    _EMPTY,
    Evaluation,
    Norm,
    PredictableVariation,
    Problem,
    curves,
    default_cutoff,
    evaluate_rows,
    point_weights,
    predictable_rows,
    restricted_weights,
)
from .cox import HazardSelection, StratifiedNelsonAalen, default_library, select_hazard
from .result import (
    BAND_SUPPORT,
    MEASURED_DEFAULTS,
    ContinuousSurvivalResult,
    attach_bootstrap,
    build_estimates,
    continuous_survival_support,
    risk_names,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._planned_fits import PlannedFits
    from ..inference.influence import ParameterEstimate

__all__ = ["ContinuousSurvivalRecord", "ContinuousSurvivalTMLE"]


def _gap(text: str) -> CapabilityError:
    """A refusal of a composition the continuous-time engine has no source or study for."""
    return CapabilityError(
        text, reason=RefusalReason.IMPLEMENTATION_GAP, alternative=NO_ALTERNATIVE, roadmap="X6"
    )


ROADMAP: Final = "roadmap X6"
#: An initial survival below this before the last time a target reads is refused: the clever
#: covariate's quotient is a conditional risk computed on the survival's own scale, which
#: double precision holds only above its smallest normal numbers.
SURVIVAL_UNDERFLOW: Final = 1e-250
#: Off-grid times drawn for the diagnostic of item 10 Table 4.
OFF_GRID_TIMES: Final = 100


def _time_text(t: float) -> str:
    return f"{t:g}"


@dataclass(frozen=True)
class PropensitySelection:
    """The cross-validated choice of a treatment model.

    Parameters
    ----------
    names : tuple of str
        The candidate models.
    losses : tuple of float
        Each candidate's cross-validated log loss.
    chosen : int
        The index of the selected candidate.
    """

    names: tuple[str, ...]
    losses: tuple[float, ...]
    chosen: int


@dataclass(frozen=True)
class ContinuousSurvivalRecord:
    """What the continuous-time fit chose and how its targeting ended.

    Parameters
    ----------
    hazards : tuple of HazardSelection
        The selection for each cause's hazard, in cause order (the full-sample fit, or the
        first fold's when cross-fitted).
    censoring : HazardSelection
        The selection for the censoring hazard.
    propensity : PropensitySelection or None
        The selection for the treatment model, or ``None`` for a user learner.
    steps : int
        Accepted targeting steps.
    halvings : int
        Rejected steps.
    converged : bool
        Whether every standardized score reached the cutoff.
    cutoff : float
        The bound on the largest :math:`|P_n D^*| / \\hat\\sigma`.
    standardized_score : tuple of float
        Each target's final :math:`|P_n D^*| / \\hat\\sigma`.
    initial_psi : tuple of float
        Each target's plug-in at the initial hazards, before any step, in target order.
    bounded_share : float
        The share of (unit, time) cells of each unit's received arm, while it is followed,
        whose :math:`\\pi S_c(u^-)` was raised to the positivity floor.
    last_time_floor_share : float
        The share of (unit, arm) pairs, over both arms whatever the unit received, whose
        :math:`\\pi S_c(u_M^-)` at the grid's end is below the floor: the counterfactual
        reach of the positivity bound.  The grid ends at the last target time, or at the last
        restricted horizon when that is later.
    frozen_share : float
        The share of (unit, arm) pairs, over both arms, that the walk held for at least one
        trial step because that step would have taken their survival at the grid's end below
        its floor.  A pair is held for one step only.
    off_grid_share : float
        The share of random times in the window, among those where a score is not identically
        zero, at which the score equation holds to the cutoff.  The window runs to the grid's
        end, so on a fit with a restricted horizon past the last target time the share is not
        comparable with the registered study's reading over the target times.
    cross_fitted : bool
        Whether the nuisances were fitted out of fold.
    n_folds : int
        The number of cross-fitting folds, ``1`` for a full-sample fit.
    grid_size : int
        The number of distinct observed times up to the last target time or restricted
        horizon.
    arm : ndarray
        ``(n,)`` each row's received arm in row order: ``0`` for the reference arm and ``1``
        for the treated arm, the codes of ``arms``.
    p1 : ndarray
        ``(n,)`` each row's estimated probability of the treated arm (arm code 1, the
        non-reference arm) given its baseline covariates, :math:`\\hat\\pi(1 \\mid L_i)`,
        in row order.  It is predicted out of fold when the fit is cross-fitted and in sample
        otherwise.  The values are as fitted, before any bound: the engine never bounds
        :math:`\\hat\\pi` alone.  It bounds the product :math:`\\hat\\pi(a \\mid L)
        \\hat S_c(u^- \\mid a, L)` below by :attr:`weight_floor`, for both arms, and
        :attr:`bounded_share` counts the cells of the received arm that bound raised.
    retention_before : ndarray
        ``(2, n, n_target_times)`` each row's estimated probability of remaining uncensored
        just before each target time had it received each arm,
        :math:`\\hat S_c(t_k^- \\mid a, L_i) = \\prod_{u < t_k}\\{1 - d\\hat\\Lambda_c(u
        \\mid a, L_i)\\}` over the observed times :math:`u`.  Axis 0 is the arm code, in the
        order of ``arms`` (reference, then treated), and column ``k`` is ``horizons[k]``.
        Predicted as :attr:`p1` is, before any bound.
    weight_floor : float
        The lower bound on :math:`\\hat\\pi(a \\mid L)\\hat S_c(u^- \\mid a, L)` that the
        targeting weights applied, exactly as resolved: ``g_bounds`` as given, or
        :math:`5 / (\\sqrt n \\log n)` for ``"auto"``.
    retention_at_time : ndarray
        ``(n,)`` each row's estimated probability of remaining uncensored just before its own
        observed time, at the arm it received, :math:`\\hat S_c(\\min(\\tilde T_i, u_M)^-
        \\mid A_i, L_i)`, with :math:`u_M` the grid's last time. Predicted as :attr:`p1` is,
        before any bound. No estimate reads it; the covariate balance of a unit whose event
        was observed before a target time does
        (:func:`~cleverly.sensitivity.covariate_balance`).
    """

    hazards: tuple[HazardSelection, ...]
    censoring: HazardSelection
    propensity: PropensitySelection | None
    steps: int
    halvings: int
    converged: bool
    cutoff: float
    standardized_score: tuple[float, ...]
    initial_psi: tuple[float, ...]
    bounded_share: float
    last_time_floor_share: float
    frozen_share: float
    off_grid_share: float
    cross_fitted: bool
    n_folds: int
    grid_size: int
    arm: IntArray
    p1: FloatArray
    retention_before: FloatArray
    weight_floor: float
    retention_at_time: FloatArray


#: The inverse L1 penalties of the lasso members of the treatment-model library.
LASSO_C = (1.0, 0.1, 0.03)


def _logistic_candidates(random_state: int | None) -> list[tuple[str, Any]]:
    """The treatment-model library: a main-terms logistic and three lasso penalties.

    One discrete cross-validated selection over the four, so no member nests its own
    cross-validation inside the selection's folds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    members: list[tuple[str, Any]] = [
        (
            "logistic",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=np.inf, max_iter=2000)),
                ]
            ),
        )
    ]
    for c in LASSO_C:
        model = LogisticRegression(
            C=c, l1_ratio=1.0, solver="saga", max_iter=5000, random_state=random_state
        )
        members.append(
            (f"lasso C={c:g}", Pipeline([("scale", StandardScaler()), ("model", model)]))
        )
    return members


def _fit_propensity(
    learner: Any,
    L: FloatArray,
    A: IntArray,
    train: Any,
    predict: Any,
    folds: IntArray,
    random_state: int | None,
) -> tuple[FloatArray, PropensitySelection | None]:
    """``P(A = 1 | L)`` on the prediction rows, from a model fitted on the training rows."""
    from sklearn.base import clone

    Lt, At = L[train], A[train]
    if learner is not None:
        # A user learner left unseeded takes the fit's seed, so a seeded fit is a function of
        # its data and seed (docs/architecture-invariants.md).
        model = clone(seed_unset(learner, random_state)).fit(Lt, At)
        return model.predict_proba(L[predict])[:, 1], None
    candidates = _logistic_candidates(random_state)
    losses = []
    for _, candidate in candidates:
        loss = 0.0
        for fold in np.unique(folds):
            fit_rows = folds != fold
            model = clone(candidate).fit(Lt[fit_rows], At[fit_rows])
            p = np.clip(model.predict_proba(Lt[~fit_rows])[:, 1], 1e-12, 1 - 1e-12)
            y = At[~fit_rows]
            loss -= float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
        losses.append(loss)
    chosen = int(np.argmin(losses))
    model = clone(candidates[chosen][1]).fit(Lt, At)
    selection = PropensitySelection(tuple(name for name, _ in candidates), tuple(losses), chosen)
    return model.predict_proba(L[predict])[:, 1], selection


def _stratified_folds(event: IntArray, n_folds: int, rng: np.random.Generator) -> IntArray:
    """Seeded, shuffled fold labels stratified by event code."""
    folds = np.empty(event.size, dtype=int)
    for code in np.unique(event):
        rows = rng.permutation(np.flatnonzero(event == code))
        folds[rows] = np.arange(rows.size) % n_folds
    return folds


class ContinuousSurvivalTMLE:
    r"""One-step TMLE of cause-specific absolute risks in continuous time.

    Parameters
    ----------
    horizons : sequence of float
        The target times :math:`t_1 < \dots < t_K`.
    restricted_horizons : sequence of float
        The restriction times :math:`\tau_1 < \dots < \tau_R` of the restricted means.  Each
        :math:`\tau` adds the restricted mean time lost to each cause,
        :math:`E \int_0^\tau F_j(u \mid a, L)\, du`, of both arms to the targets the walk
        solves, so :meth:`ContinuousSurvivalResult.rmtl` and
        :meth:`~ContinuousSurvivalResult.rmst` can read them.  The extra targets change the walk,
        so the risks of such a fit differ slightly from those of a fit without them.
    arms : mapping of str to level, or None
        The label of each treatment level, two levels.  ``None`` labels the two observed
        levels by ``str(level)``.
    reference : str or None
        The label of the comparison arm of the contrasts.  ``None`` takes the first label.
    causes : mapping of int to str, or None
        The label of each nonzero event code.  ``None`` reads one code ``1`` as one event,
        and several codes as competing causes labelled ``str(code)``.
    cross_fit : bool
        Fit every nuisance out of fold on ``n_folds`` folds, selecting each model inside the
        training folds.  ``False`` fits each on every row, the sources' setting.
    n_folds : int
        Cross-fitting folds.
    learner_folds : int
        Folds of the cross-validation that selects each nuisance model.
    hazard_library : sequence or None
        Hazard learners for every cause.  ``None`` takes three unstratified Cox models.
    censoring_library : sequence or None
        Hazard learners for the censoring hazard.  ``None`` takes the same three.
    treatment_learner : classifier or None
        The treatment model.  ``None`` selects between a main-terms logistic model and an L1
        logistic model by cross-validated log loss.  A user learner needs ``cross_fit=True``.
    norm : {"covariance", "variance", "euclidean"}
        The inner product of the universal submodel, with the covariance held at the initial
        fit.
    g_bounds : "auto" or float
        The lower bound on :math:`\pi(a \mid L)\, S_c(u^- \mid a, L)`.  ``"auto"`` is
        :math:`5 / (\sqrt n \log n)`.
    max_iter : int
        The cap on accepted targeting steps.
    step_size : float
        The first step size.
    tol : float or None
        Stop when the largest :math:`|P_n D^*| / \hat\sigma` is at or below ``tol``.
        ``None`` uses :math:`1 / (\sqrt n \log n)` (Rytgaard and van der Laan 2024, Sec 4.3).
    alpha : float
        Significance level of the intervals.
    random_state : int or None
        Seed of the folds, the inner selection folds, the lasso members, any unseeded user
        learner, the off-grid diagnostic and the band.  ``None`` draws one seed per fit,
        which the result's stored estimator records.
    tilt : {"all", "factual"}
        ``"factual"`` reproduces the ``concrete`` 1.0.5 update, for the mutation control only.
    simultaneous : bool
        Add a sup-t band over every risk and risk difference.
    n_multiplier : int
        Multiplier draws of the band.
    multiplier_kind : str
        The multiplier distribution, as :func:`~cleverly.inference.simultaneous_bands` takes.
    band_scale : {"logit", "identity"}
        The scale of the band of a bounded estimate.
    variance : {"switched", "influence_curve"}
        The variance every estimate reports.  ``"switched"`` reads the influence-curve
        variance, raised to the predictable variation of the influence curve's martingale part
        where the curve is sparse: where the relative standard deviation of the
        influence-curve variance exceeds
        :data:`~cleverly.inference.sparse.SPARSE_INFLUENCE_THRESHOLD` (0.125), the variance is
        the larger of the two (:mod:`cleverly.inference.switched`).  It is never below the
        influence-curve variance and equals it with probability tending to one.
        ``"influence_curve"`` reads the influence-curve variance alone.  A band takes its
        critical value from the influence curves and its half-widths from the reported
        standard errors.
    max_memory_bytes : int or None
        The memory budget of a fit, in bytes.  The walk works through blocks of units; before
        fitting any nuisance, the fit predicts the largest number of bytes it holds at once with
        no block kept between passes, which grows with ``n`` times the number of targets, and
        refuses with :class:`~cleverly.exceptions.MethodConfigurationError` when the prediction
        exceeds the budget.  The walk then keeps as many blocks as the budget allows and
        rebuilds the others on each pass, so the budget sets how long a fit runs and never what
        it returns.  ``None`` reads the memory available at that moment, keeps blocks up to
        three quarters of it, and checks nothing (keeping every block) on a platform that does
        not report it.  A value overrides the reading either way.  Every refit of a result, and
        so every refutation and truncation curve, uses the value of the fit.
    n_jobs : int or None
        The threads the targeting walk runs its blocks of units on.  ``-1`` takes the CPUs the
        process may use (its affinity, capped by a control group quota) and ``None`` one.  The
        result is bit for bit the same at any value; each thread holds one block's working
        arrays, which the memory budget counts.  With ``n_bootstrap``, the bootstrap refits run
        on this many worker processes, each walking on one thread.
    n_bootstrap : int
        Full-refit bootstrap replicates, ``0`` for none and otherwise at least 2.  Each replicate
        resamples units with replacement and reruns the whole fit: the folds, every nuisance
        selection and the walk.  Copies of a unit share every fold, the cross-fitting folds and
        the selections' inner folds, so no copy is validated on its own twin.  Each estimate
        then carries a :class:`~cleverly.inference.BootstrapSummary` beside its Wald interval,
        and with ``simultaneous=True`` the result adds a studentized bootstrap band over every
        risk and risk difference.  When more than 1% of the refits fail, every bootstrap
        interval is withheld with a :class:`~cleverly.exceptions.ConvergenceWarning`, and the
        Wald intervals stand.  No bootstrap interval of this engine reads supported.
    bootstrap_interval : {"percentile", "bca"}
        The bootstrap interval each estimate carries: the percentile interval of the replicate
        estimates, or the bias-corrected and accelerated interval, with the acceleration read
        from the estimate's own influence curve.
    """

    def __init__(
        self,
        horizons: Sequence[float],
        *,
        restricted_horizons: Sequence[float] = (),
        arms: Mapping[str, Any] | None = None,
        reference: str | None = None,
        causes: Mapping[int, str] | None = None,
        cross_fit: bool = True,
        n_folds: int = 10,
        learner_folds: int = 5,
        hazard_library: Sequence[Any] | None = None,
        censoring_library: Sequence[Any] | None = None,
        treatment_learner: Any = None,
        norm: Norm = "covariance",
        g_bounds: Literal["auto"] | float = "auto",
        max_iter: int = 500,
        step_size: float = 0.1,
        tol: float | None = None,
        alpha: float = 0.05,
        random_state: int | None = None,
        tilt: Literal["all", "factual"] = "all",
        simultaneous: bool = False,
        n_multiplier: int = DEFAULT_N_REPLICATES,
        multiplier_kind: str = DEFAULT_KIND,
        band_scale: str = DEFAULT_BAND_SCALE,
        variance: Literal["switched", "influence_curve"] = "switched",
        max_memory_bytes: int | None = None,
        n_jobs: int | None = 1,
        n_bootstrap: int = 0,
        bootstrap_interval: Literal["percentile", "bca"] = "percentile",
    ) -> None:
        budget = check_max_memory_bytes(max_memory_bytes)
        if (
            isinstance(n_bootstrap, bool)
            or not isinstance(n_bootstrap, (int, np.integer))
            or n_bootstrap < 0
            or n_bootstrap == 1
        ):
            raise ValueError(f"n_bootstrap is 0 or an integer of at least 2; got {n_bootstrap!r}")
        if bootstrap_interval not in ("percentile", "bca"):
            raise ValueError(
                f"bootstrap_interval is 'percentile' or 'bca'; got {bootstrap_interval!r}"
            )
        if bootstrap_interval != "percentile" and not n_bootstrap:
            raise ValueError(
                f"bootstrap_interval={bootstrap_interval!r} names an interval of the bootstrap, "
                "and n_bootstrap=0 runs none; set n_bootstrap"
            )
        if n_jobs is None:
            n_jobs = 1
        if (
            isinstance(n_jobs, bool)
            or not isinstance(n_jobs, (int, np.integer))
            or not (n_jobs >= 1 or n_jobs == -1)
        ):
            raise ValueError(f"n_jobs is a positive integer, -1 or None; got {n_jobs!r}")
        times = tuple(float(t) for t in horizons)
        if not times or any(not np.isfinite(t) or t <= 0 for t in times):
            raise ValueError("horizons are positive finite target times, at least one")
        if any(b <= a for a, b in pairwise(times)):
            raise ValueError("horizons are strictly increasing")
        restricted = tuple(float(t) for t in restricted_horizons)
        if any(not np.isfinite(t) or t <= 0 for t in restricted):
            raise ValueError("restricted_horizons are positive finite restriction times")
        if any(b <= a for a, b in pairwise(restricted)):
            raise ValueError("restricted_horizons are strictly increasing")
        if norm not in ("covariance", "variance", "euclidean"):
            raise ValueError("norm is 'covariance', 'variance' or 'euclidean'")
        if variance not in ("switched", "influence_curve"):
            raise ValueError("variance is 'switched' or 'influence_curve'")
        if treatment_learner is not None and not cross_fit:
            raise _gap(
                "A user treatment learner needs cross_fit=True: without cross-fitting every "
                "nuisance is fitted and predicted on the same rows, and the interval assumes a "
                "finite parametric model there (a Donsker class), which the default "
                "main-terms and L1 logistic selection is. A flexible learner fitted in sample "
                "pushes the propensities to 0 and 1."
            )
        self.horizons = times
        self.restricted_horizons = restricted
        self.arms = None if arms is None else dict(arms)
        self.reference = reference
        self.causes = None if causes is None else {int(k): str(v) for k, v in causes.items()}
        self.cross_fit = bool(cross_fit)
        self.n_folds = int(n_folds)
        self.learner_folds = int(learner_folds)
        self.hazard_library = tuple(hazard_library) if hazard_library else default_library()
        self.censoring_library = (
            tuple(censoring_library) if censoring_library else default_library()
        )
        self.treatment_learner = treatment_learner
        self._default_nuisances = (
            hazard_library is None and censoring_library is None and treatment_learner is None
        )
        self.norm: Norm = norm
        self.g_bounds = g_bounds
        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.tol = tol
        self.alpha = float(alpha)
        self.random_state = random_state
        self.tilt: Literal["all", "factual"] = tilt
        self.simultaneous = bool(simultaneous)
        self.n_multiplier = int(n_multiplier)
        self.multiplier_kind = multiplier_kind
        self.band_scale = band_scale
        self.variance: Literal["switched", "influence_curve"] = variance
        self.max_memory_bytes = budget
        self.n_jobs = int(n_jobs)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_interval: Literal["percentile", "bca"] = bootstrap_interval
        #: The outer cross-fitting label of each row, recorded on the estimator a result
        #: stores; a subset refit reads them so its refits keep the fit's split.
        self._outer_folds: IntArray | None = None
        #: The frozen labels a refit was given in place of a draw, so a replay passes them
        #: again; ``None`` when the folds were drawn from the seed.
        self._fold_override: IntArray | None = None
        #: The whole split the fit used: its outer labels, each outer fold's inner selection
        #: labels and the generator state after the nuisance phase.  A simulated-confounding
        #: replay passes it back, so a refit of changed event codes keeps every fold.
        self._fold_plan: FoldPlan | None = None
        #: The plan a refit was given in place of its draws, so a replay passes it again.
        self._plan_override: FoldPlan | None = None

    # ------------------------------------------------------------------ inputs

    def _arms(self, treatment: Any) -> tuple[tuple[str, str], IntArray]:
        levels = list(dict.fromkeys(np.unique(treatment).tolist()))
        if self.arms is None:
            if len(levels) != 2:
                raise _gap(
                    f"The continuous-time engine takes a binary baseline treatment; the "
                    f"column holds {len(levels)} levels. Every source of this estimator "
                    f"has two arms ({ROADMAP})."
                )
            labels = {str(level): level for level in levels}
        else:
            labels = self.arms
            if len(labels) != 2:
                raise _gap(
                    f"The continuous-time engine contrasts two static arms; {len(labels)} "
                    f"were declared ({ROADMAP})."
                )
            if set(np.unique(treatment).tolist()) - set(labels.values()):
                raise _gap(
                    "The treatment column holds a level no arm names; the continuous-time "
                    f"engine takes a binary baseline treatment ({ROADMAP})."
                )
        names = list(labels)
        reference = self.reference if self.reference is not None else names[0]
        if reference not in labels:
            raise ValueError(f"reference {reference!r} is not one of the arms {names}")
        ordered = (reference, next(name for name in names if name != reference))
        coded = np.where(np.asarray(treatment) == labels[ordered[1]], 1, 0).astype(int)
        return ordered, coded

    def _causes(self, event: IntArray, declared: Mapping[int, str] | None = None) -> dict[int, str]:
        codes = sorted(int(c) for c in np.unique(event) if c != 0)
        declared = self.causes if declared is None else declared
        if declared is not None:
            missing = set(codes) - set(declared)
            if missing:
                raise DataError(f"event codes {sorted(missing)} are not named by causes")
            absent = sorted(int(k) for k in declared if int(k) not in codes)
            if absent:
                raise _gap(
                    f"The declared causes {[declared[k] for k in absent]} have no event in the "
                    "data, so their risks are zero by construction and have no interval. Drop "
                    "them from causes=, or check the event column."
                )
            return {int(k): str(v) for k, v in sorted(declared.items())}
        if codes == [1]:
            return {1: "1"}
        return {code: str(code) for code in codes}

    # --------------------------------------------------------------------- fit

    def fit(
        self,
        data: Any,
        *,
        time: str,
        event: str,
        treatment: str,
        baseline: Sequence[str],
    ) -> ContinuousSurvivalResult:
        """Fit the nuisances and target every risk.

        Parameters
        ----------
        data : DataFrame
            A pandas or polars frame with one row per unit.
        time : str
            The follow-up time column: positive finite numbers.
        event : str
            The event-code column: ``0`` censored, a positive integer per cause.
        treatment : str
            The binary baseline treatment column.
        baseline : sequence of str
            The baseline covariates.

        Returns
        -------
        ContinuousSurvivalResult
            The targeted risks, their contrasts and the record.
        """
        return self._fit_data(
            self._container(data, time=time, event=event, treatment=treatment, baseline=baseline)
        )

    def _planned_fits(
        self,
        data: Any,
        *,
        time: str,
        event: str,
        treatment: str,
        baseline: Sequence[str],
    ) -> PlannedFits:
        """The model fits :meth:`fit` would make on the same arguments, counted unfitted.

        Takes :meth:`fit`'s arguments, runs its checks of the frame, and previews the
        container as :meth:`_planned_fit_data` does.
        """
        return self._planned_fit_data(
            self._container(data, time=time, event=event, treatment=treatment, baseline=baseline)
        )

    @staticmethod
    def _container(
        data: Any, *, time: str, event: str, treatment: str, baseline: Sequence[str]
    ) -> Any:
        """The validated container :meth:`fit` builds from a frame."""
        from ..utils.frames import as_frame, backend_of
        from .data import ContinuousSurvivalData

        frame = as_frame(data)
        T = frame[time].to_numpy().astype(float)
        D_raw = frame[event].to_numpy().astype(float)
        if not np.all(np.isfinite(T)) or np.any(T <= 0):
            raise DataError("every follow-up time is a positive finite number")
        if not np.all(np.isfinite(D_raw)) or np.any(D_raw < 0) or np.any(D_raw % 1):
            raise DataError("every event code is a non-negative integer, 0 for censored")
        L = np.column_stack([frame[name].to_numpy().astype(float) for name in baseline])
        if not np.all(np.isfinite(L)):
            raise DataError("baseline covariates are finite numbers")
        container = ContinuousSurvivalData(
            T,
            D_raw.astype(int),
            frame[treatment].to_numpy(),
            L,
            (str(time), str(event), str(treatment), *(str(name) for name in baseline)),
            None,
            backend_of(frame),
        )
        return container

    def fit_data(self, data: Any) -> ContinuousSurvivalResult:
        """Fit a :class:`~cleverly.survival.data.ContinuousSurvivalData` container.

        Parameters
        ----------
        data : ContinuousSurvivalData
            The validated container.

        Returns
        -------
        ContinuousSurvivalResult
            The targeted risks, their contrasts and the record.
        """
        return self._fit_data(data)

    def refit(
        self,
        data: Any,
        *,
        random_state: int | None,
        _frozen_folds: Any = None,
        _frozen_plan: FoldPlan | None = None,
    ) -> ContinuousSurvivalResult:
        """Fit a copy of this estimator, at the seed ``random_state``, to ``data``.

        ``result.estimator.refit(result.data, random_state=result.estimator.random_state)``
        reproduces the fit bit for bit in one process: the stored estimator carries the
        fit's resolved seed, and every learner, the folds, the off-grid draws and the band
        read that seed.

        Parameters
        ----------
        data : ContinuousSurvivalData
            The rows to fit.
        random_state : int or None
            The seed of the copy.  ``None`` draws one, which the new result's estimator
            records.
        _frozen_folds : array-like or None
            Private: one outer cross-fitting label per row of ``data``, used instead of a
            fresh draw.  A subset refutation passes each drawn unit's original label, so the
            refits keep the fit's split.  When omitted, a fit that was itself produced with
            frozen labels reuses them on data of the same size.
        _frozen_plan : FoldPlan or None
            Private: the fold plan of a fit of data of the same size, used instead of every
            fold draw.  The outer labels, each outer fold's inner selection labels and the
            generator state after the nuisance phase are the plan's, so a refit of the same
            rows at the same seed reproduces that fit bit for bit, and a refit of changed
            event codes keeps every fold.  A simulated-confounding replay passes it.  When
            omitted, a fit that was itself produced under a plan reuses it on data of the same
            size.

        Returns
        -------
        ContinuousSurvivalResult
            The fit of the copy.

        Raises
        ------
        ValueError
            If both ``_frozen_folds`` and ``_frozen_plan`` are passed, or the plan does not
            fit the data.
        """
        folds, plan = self._refit_split(data, _frozen_folds, _frozen_plan)
        estimator = self._copy(random_state)
        return estimator._fit_data(data, folds=folds, plan=plan)

    def _planned_refit(
        self,
        data: Any,
        *,
        random_state: int | None,
        _frozen_folds: Any = None,
        _frozen_plan: FoldPlan | None = None,
        nominal: bool = False,
    ) -> PlannedFits:
        """The model fits :meth:`refit` would make on the same arguments, counted unfitted.

        :func:`cleverly._planned_fits.refit_plan` reads it.  ``nominal`` counts at the
        libraries' nominal inner folds, as :meth:`_planned_fit_data` does.
        """
        folds, plan = self._refit_split(data, _frozen_folds, _frozen_plan)
        return self._copy(random_state)._planned_fit_data(
            data, folds=folds, plan=plan, nominal=nominal
        )

    def _refit_split(
        self, data: Any, frozen: Any, frozen_plan: FoldPlan | None
    ) -> tuple[Any, FoldPlan | None]:
        """The outer labels or the fold plan a refit on ``data`` reuses, or neither.

        A plan holds its own outer labels, so both together are refused.  Without either, a
        fit made under a plan reuses it on data of the same size, and otherwise a fit made on
        frozen labels reuses those (:meth:`_refit_folds`).
        """
        if frozen_plan is not None and frozen is not None:
            raise ValueError(
                "pass a frozen fold plan or frozen outer labels, not both: the plan holds its "
                "own outer labels"
            )
        plan = frozen_plan
        if plan is not None and plan.outer.size != len(data.time):
            raise ValueError(
                f"a frozen fold plan holds {plan.outer.size} rows; the data hold {len(data.time)}"
            )
        prior = self._plan_override
        if (
            frozen is None
            and plan is None
            and prior is not None
            and prior.outer.size == len(data.time)
        ):
            plan = prior
        if plan is not None:
            return None, plan
        return self._refit_folds(data, frozen), None

    def _refit_folds(self, data: Any, frozen: Any) -> Any:
        """The outer labels a refit on ``data`` reuses, or ``None`` for a fresh draw."""
        override = self._fold_override
        if frozen is None and override is not None and override.size == len(data.time):
            return override
        return frozen

    def _copy(self, random_state: int | None) -> ContinuousSurvivalTMLE:
        """A copy at ``random_state`` that shares no mapping with this estimator."""
        estimator = copy.copy(self)
        estimator.random_state = random_state
        estimator.arms = None if self.arms is None else dict(self.arms)
        estimator.causes = None if self.causes is None else dict(self.causes)
        return estimator

    def _resolved_seed(self) -> int:
        """The seed every random draw of a fit reads: the declared one, or one drawn now."""
        if self.random_state is not None:
            return int(self.random_state)
        note_fresh_entropy()
        return int(np.random.SeedSequence().generate_state(1)[0])

    def _threads(self) -> int:
        """The walk's threads: ``n_jobs``, with ``-1`` read as the CPUs available now."""
        return available_cpus() if self.n_jobs == -1 else self.n_jobs

    def _floor(self, n: int) -> float:
        """The floor on :math:`\\pi(a \\mid L) S_c(u^- \\mid a, L)` that ``g_bounds`` sets."""
        if self.g_bounds == "auto":
            return float(5.0 / (np.sqrt(n) * np.log(n)))
        return float(self.g_bounds)

    def _fit_data(
        self, data: Any, *, folds: Any = None, plan: FoldPlan | None = None
    ) -> ContinuousSurvivalResult:
        seed = self._resolved_seed()
        initial = self._initial(data, seed, folds=folds, plan=plan)
        settings = {key: getattr(self, key) for key in MEASURED_DEFAULTS}
        estimates, record, covariance = self._target(
            initial, self._floor(initial.n), settings=settings
        )
        bands = None
        if self.simultaneous:
            from ..inference.multiplier import simultaneous_bands

            # The band covers the risks and their differences, as the study's band row reads.
            risks = risk_names(initial.arm_labels, tuple(initial.causes.values()), self.horizons)
            banded = {k: estimates[k] for k in risks}
            bands = simultaneous_bands(
                banded,
                alpha=self.alpha,
                n_replicates=self.n_multiplier,
                kind=self.multiplier_kind,  # type: ignore[arg-type]
                random_state=seed,
                band_scale=self.band_scale,  # type: ignore[arg-type]
            )
            bands = replace(bands, support=BAND_SUPPORT)
        stored = self._copy(seed)
        stored._outer_folds = initial.folds.copy()
        stored._fold_override = None if folds is None else initial.folds.copy()
        stored._fold_plan = initial.plan
        stored._plan_override = plan
        result = ContinuousSurvivalResult(
            estimates=estimates,
            simultaneous=bands,
            record=record,
            arms=initial.arm_labels,
            causes=tuple(initial.causes.values()),
            horizons=self.horizons,
            n=initial.n,
            backend=data.backend,
            restricted_horizons=self.restricted_horizons,
            predictable=covariance,
            estimator=stored,
            # Copy the arrays: the caller's frame or container may be edited after the fit,
            # and a refutation must refit the rows this fit read.
            data=replace(
                data,
                time=np.array(data.time, copy=True),
                event=np.array(data.event, copy=True),
                treatment=np.array(data.treatment, copy=True),
                baseline=np.array(data.baseline, copy=True),
            ),
        )
        if not self.n_bootstrap:
            return result
        return self._bootstrap(result, initial, seed)

    def _for_estimate_refits(self) -> ContinuousSurvivalTMLE:
        """This estimator, or a copy without the bootstrap, for refits that read only estimates.

        A refutation or a simulated-confounding cell refits the stored estimator many times and
        reads each refit's estimates alone; the bootstrap would multiply each refit's cost by
        ``n_bootstrap + 1`` and change none of them.  An estimator with no bootstrap is returned
        itself, so a caller that keys its refits on the estimator sees one estimator.
        """
        return self._without_bootstrap() if self.n_bootstrap else self

    def _without_bootstrap(self) -> ContinuousSurvivalTMLE:
        """A copy of this estimator that runs no bootstrap."""
        estimator = self._copy(self.random_state)
        estimator.n_bootstrap = 0
        estimator.bootstrap_interval = "percentile"
        return estimator

    def _replica(self, causes: Mapping[int, str], budget: int | None) -> ContinuousSurvivalTMLE:
        """The estimator each bootstrap replicate refits: this one, with no bootstrap or band.

        Its causes are the fit's, so a resample that loses a cause is refused rather than
        reporting fewer risks, and its walk runs on one thread, since the replicates run on
        :attr:`n_jobs` workers.  ``budget`` is the memory each replicate may use.
        """
        replica = self._without_bootstrap()
        replica.causes = dict(causes)
        replica.simultaneous = False
        replica.n_jobs = 1
        replica.max_memory_bytes = budget
        replica._outer_folds = replica._fold_override = None
        replica._fold_plan = replica._plan_override = None
        return replica

    def _bootstrap_workers(self) -> int:
        """The worker processes the bootstrap refits run on: ``n_jobs``, at most the replicates."""
        return max(1, min(self._threads(), self.n_bootstrap))

    def _replicate_budget(self, shape: Mapping[str, Any]) -> int | None:
        """The memory budget of one replicate: its worker's share of what the result leaves.

        With :attr:`max_memory_bytes` the budget is that, less the result and the draws
        (:func:`._memory.bootstrap_held`), per worker.  The fit already refused, before any
        learner, a budget that does not hold the result beside one replicate per worker at the
        replicate's predicted peak, so a replicate never refuses its own.  Under the default
        the budget is the memory available now, after the fit, which already leaves out the
        resident result, so nothing more is subtracted; memory taken by another process since
        the fit's check can still make a replicate refuse, and the bootstrap counts it as a
        failed refit.  ``None`` where the platform reports no available memory.
        """
        if self.max_memory_bytes is not None:
            held = bootstrap_held(**{key: shape[key] for key in BOOTSTRAP_HELD})
            budget = self.max_memory_bytes - held
        else:
            available = available_memory()
            if available is None:
                return None
            budget = available
        return max(1, int(budget // self._bootstrap_workers()))

    def _bootstrap(
        self, result: ContinuousSurvivalResult, initial: _Initial, seed: int
    ) -> ContinuousSurvivalResult:
        """Run the full-refit bootstrap and attach its intervals and band to ``result``.

        The row index of replicate ``b`` is the one
        :func:`~cleverly.inference.bootstrap.run_index_bootstrap` draws for the fit's seed, and
        the replicate refits at the seed spawned for it beside that index.  Every fold of a
        replicate is drawn on its distinct units (:func:`replicate_plan`).  A replicate that
        raises a counted refusal (:func:`_replicate`) is counted with its reason; more than
        :data:`BOOTSTRAP_FAILURE_SHARE` of them withholds every bootstrap interval.
        """
        from ..inference.bootstrap import _bootstrap_design

        assert result.data is not None
        data = result.data
        n = initial.n
        design = _bootstrap_design(
            _Units(n), n_replicates=self.n_bootstrap, resampling="iid", random_state=seed
        )
        replica = self._replica(initial.causes, self._replicate_budget(initial.memory))
        # Each worker draws its replicate's row index from the replicate's seed sequence, so
        # the B indices are never held at once.
        payloads = [(replica, data, draw.sequence, draw.seed) for draw in design.draws]
        with _suppressed():
            # A resample routinely trips convergence and positivity warnings; one per replicate
            # would bury the fit's own, and none of them belongs to the fit.
            outcomes = map_parallel(_drawn_replicate, payloads, n_jobs=self._bootstrap_workers())
        return attach_bootstrap(
            result,
            outcomes,
            interval=self.bootstrap_interval,
            simultaneous=self.simultaneous,
            alpha=self.alpha,
            cross_fitted=self.cross_fit,
        )

    def _memory_shape(
        self,
        n: int,
        causes: int,
        times: int,
        target_weights: FloatArray,
        covariates: int,
        folds: IntArray,
    ) -> dict[str, Any]:
        """The keyword arguments of :func:`._memory.peak_bytes` for a fit of these sizes."""
        weights = np.asarray(target_weights)
        return {
            "n": n,
            "causes": causes,
            "times": times,
            "rows": int(weights.shape[0]),
            "covariates": covariates,
            "train_size": train_size(folds, n, self.cross_fit),
            "folds": int(np.unique(np.asarray(folds)).size),
            "design_width": design_width(
                (*self.hazard_library, *self.censoring_library), covariates
            ),
            "simultaneous": self.simultaneous,
            "n_multiplier": self.n_multiplier,
            "factual_tilt": self.tilt == "factual",
            "stratified_learner": any(
                isinstance(member, StratifiedNelsonAalen)
                for member in (*self.hazard_library, *self.censoring_library)
            ),
            "restricted": not _point_rows(weights),
            "switched": self.variance == "switched",
            "n_jobs": self._threads(),
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_workers": self._bootstrap_workers() if self.n_bootstrap else 1,
            "estimates": estimate_count(causes, len(self.horizons), len(self.restricted_horizons)),
        }

    def _planned_fit_data(
        self,
        data: Any,
        *,
        folds: Any = None,
        plan: FoldPlan | None = None,
        nominal: bool = False,
    ) -> PlannedFits:
        """The model fits :meth:`fit_data` would make, counted without fitting.

        Runs the same checks before the first fit (:meth:`_initial_inputs`).  Per outer fold
        each cause's hazard and the censoring hazard are chosen from their library by
        cross-validation over the inner folds, ``M * V`` fits and one refit of the winner (one
        fit for a library of one), and the treatment model is the user's learner, one fit,
        or the selection over the four default logistic models, ``4 * V + 1`` fits.  ``V`` is
        the inner fold count the stratified labels reach on the fold's training rows.  With
        :attr:`n_bootstrap` every replicate's refits are added, per bootstrap replicate
        (:meth:`_plan_bootstrap`).

        Parameters
        ----------
        data : ContinuousSurvivalData
            The container.
        folds : array-like or None
            One outer label per row in place of a fresh draw, as :meth:`_fit_data` takes it.
        plan : FoldPlan or None
            A recorded fold plan in place of every fold draw, as :meth:`_fit_data` takes it.
        nominal : bool
            Count each selection at :attr:`learner_folds` inner folds, whatever the fold's
            event counts allow, and a user treatment learner at its nominal counts, so the
            record bounds a fit of any data drawn like ``data`` (X106 part G2).  Each
            bootstrap replicate is then counted as one such fit.

        Returns
        -------
        PlannedFits
            The record.
        """
        from .._planned_fits import Count, PlanBuilder, count_learner
        from .._summary._learners import learner_name

        seed = self._resolved_seed()
        inputs = self._initial_inputs(data, seed, folds=folds, plan=plan)
        A, D, n, codes, folds = inputs[1], inputs[4], inputs[6], inputs[7], inputs[14]
        hazard_library = tuple(seed_unset(member, seed) for member in self.hazard_library)
        censoring_library = tuple(seed_unset(member, seed) for member in self.censoring_library)

        def selection(library: tuple[Any, ...], inner: int) -> Count:
            count = Count(fits=1)
            if len(library) == 1:
                count.candidates[type(library[0]).__name__] += 1
                return count
            for member in library:
                count.candidates[type(member).__name__] += inner
            # The refit of the winner, whose name is known only after the selection.
            count.candidates[type(library[0]).__name__] += 1
            return count

        builder = PlanBuilder()
        why = (
            "drawn-data refits are counted at the nominal inner folds, whatever a fold's "
            "event counts allow"
            if nominal
            else None
        )
        names = ", ".join(learner_name(member) for member in hazard_library)
        for fold in np.unique(folds):
            train = folds != fold if self.cross_fit else np.ones(n, dtype=bool)
            observed = np.unique(D[train], return_counts=True)[1]
            inner = (
                int(self.learner_folds) if nominal else min(self.learner_folds, int(observed.max()))
            )
            for _ in codes:
                builder.add(
                    "hazard_library",
                    "cause-specific hazard selection, per cause and fold",
                    names,
                    selection(hazard_library, inner),
                    upper=why,
                )
            builder.add(
                "censoring_library",
                "censoring hazard selection, per fold",
                ", ".join(learner_name(member) for member in censoring_library),
                selection(censoring_library, inner),
                upper=why,
            )
            if self.treatment_learner is not None:
                count = count_learner(
                    seed_unset(self.treatment_learner, seed), A[train], nominal=nominal
                )
                builder.add(
                    "treatment_learner",
                    "treatment model, per fold",
                    learner_name(self.treatment_learner),
                    count,
                    upper=why,
                )
                continue
            candidates = _logistic_candidates(seed)
            count = Count(fits=1)
            for _, candidate in candidates:
                count.candidates.update(count_learner(candidate, None).scaled(inner).candidates)
            count.candidates.update(count_learner(candidates[0][1], None).candidates)
            builder.add(
                "treatment_learner",
                "treatment model selection over the default logistic models, per fold",
                "default logistic selection",
                count,
                upper=why,
            )
        if self.n_bootstrap and nominal:
            unit = self._copy(seed)
            unit.n_bootstrap = 0
            counted = unit._planned_fit_data(data, folds=folds, nominal=True)
            for _ in range(int(self.n_bootstrap)):
                builder.absorb(counted, lambda scope: f"{scope}, per bootstrap replicate")
            builder.multiply("bootstrap replicates", self.n_bootstrap)
        elif self.n_bootstrap:
            self._plan_bootstrap(builder, data, seed, inputs[2])
        if len(hazard_library) > 1 or len(censoring_library) > 1:
            builder.note(
                "A hazard selection's refit is filed under the first member's class; the "
                "winner is known only after the cross-validation."
            )
        return builder.build(
            "realized"
            if self.random_state is not None
            else "one draw"
            if folds is None
            else "outer realized"
        )

    def _plan_bootstrap(
        self, builder: Any, data: Any, seed: int, causes: Mapping[int, str]
    ) -> None:
        """Add every bootstrap replicate's planned fits to ``builder``, counted unfitted.

        Each replicate's row index and fold plan follow from the fit's seed alone, as
        :meth:`_bootstrap` and :func:`_replicate` draw them, so the replicate's fits are
        counted on its resample with :func:`replicate_plan`'s folds, one replicate at a time:
        no learner runs and one resample is held at once.  A replicate the checks before the
        first fit refuse (a lost cause, say) makes no fit and adds none.
        """
        from ..inference.bootstrap import _bootstrap_design

        n = int(np.asarray(data.time).shape[0])
        design = _bootstrap_design(
            _Units(n), n_replicates=self.n_bootstrap, resampling="iid", random_state=seed
        )
        replica = self._replica(causes, None)
        refused = 0
        for draw in design.draws:
            index = _replicate_index(data, draw.sequence)
            sample = resample(data, index)
            plan = replicate_plan(
                index,
                sample.event,
                cross_fit=replica.cross_fit,
                n_folds=replica.n_folds,
                learner_folds=replica.learner_folds,
                seed=draw.seed,
            )
            try:
                counted = replica._copy(draw.seed)._planned_fit_data(sample, plan=plan)
            except (CleverlyError, np.linalg.LinAlgError, FloatingPointError):
                refused += 1
                continue
            builder.absorb(counted, lambda scope: f"{scope}, per bootstrap replicate")
        builder.multiply("bootstrap replicates", self.n_bootstrap)
        builder.note(
            f"The bootstrap refits every model on each of {self.n_bootstrap} resamples, with "
            "folds drawn on that resample's distinct units; the rows marked per bootstrap "
            "replicate count them, summed over the replicates. A replicate whose refit fails "
            "stops at the failure, so it makes at most the fits counted."
        )
        if refused:
            builder.note(
                f"{refused} of the {self.n_bootstrap} resamples fail the checks before the "
                "first fit, so their replicates make no fit and none is counted."
            )

    def _initial_inputs(
        self,
        data: Any,
        seed: int,
        *,
        folds: Any = None,
        plan: FoldPlan | None = None,
    ) -> tuple[Any, ...]:
        """Every check :meth:`_initial` runs before its first fit, and what they resolve.

        :meth:`_initial` and :meth:`_planned_fits` both call it: the arms and causes, the
        follow-up refusals, the outer split drawn from ``seed`` and the memory check.

        Parameters
        ----------
        data : ContinuousSurvivalData
            The container.
        seed : int
            The fit's resolved seed.
        folds : array-like or None
            As :meth:`_initial`.
        plan : FoldPlan or None
            As :meth:`_initial`; its outer labels replace the split.

        Returns
        -------
        tuple
            The arm labels, the arm codes, the causes, the times, event codes and covariates,
            ``n``, the cause codes and their count, the grid, the target index and weights, the
            design, the generator after the split, the fold labels and their count.
        """
        arm_labels, A = self._arms(data.treatment)
        causes = self._causes(data.event, self.causes if self.causes is not None else data.causes)
        T = np.asarray(data.time, dtype=float)
        D = np.asarray(data.event, dtype=int)
        L = np.asarray(data.baseline, dtype=float)
        n = T.size
        codes = list(causes)
        J = len(codes)
        restricted = self.restricted_horizons
        # The first and last time some target reads, with what the refusals call it.
        first, first_kind = self.horizons[0], "target time"
        if restricted and restricted[0] < first:
            first, first_kind = restricted[0], "restricted horizon"
        ends = [(self.horizons[-1], "target time", "risk")]
        if restricted:
            ends.append((restricted[-1], "restricted horizon", "restricted mean"))
        # Both checks below read this draw's events and follow-up, not the configuration, so a
        # draw that trips one is a data failure (a study records it as a lost replicate), not an
        # unsupported composition.
        for a in (0, 1):
            for code, label in causes.items():
                first_event = T[(a == A) & (code == D)]
                if first_event.size == 0 or first_event.min() > first:
                    what = "risk" if first_kind == "target time" else "restricted mean"
                    raise DataError(
                        f"Arm {arm_labels[a]!r} has no event of cause {label!r} at or before "
                        f"the first {first_kind} {_time_text(first)}, so that {what}'s "
                        "influence curve is zero and its interval undefined. Choose a later "
                        f"first {first_kind}."
                    )
        for a in (0, 1):
            followed = float(T[a == A].max())
            for end, kind, what in ends:
                if followed < end:
                    raise DataError(
                        f"Arm {arm_labels[a]!r} is followed only to {_time_text(followed)}, "
                        f"before the last {kind} {_time_text(end)}, so its {what} there would "
                        "be an extrapolation of the hazard model with no data behind it. "
                        f"Choose {kind}s within every arm's follow-up."
                    )
        last = max(end for end, _, _ in ends)
        grid = np.unique(T[last >= T])
        target_index = np.searchsorted(grid, self.horizons, side="right") - 1
        target_weights = np.vstack(
            [point_weights(target_index, grid.size)]
            # Each restricted row is scaled by 1 / tau: the walk targets RMTL / tau, a share of
            # [0, 1] like a risk, so its metric's floor and ridge do not depend on the unit of
            # time.  build_estimates reports RMTL in time units.
            + [restricted_weights(grid, tau)[None, :] / tau for tau in restricted]
        )
        X = np.column_stack([A.astype(float), L])
        rng = np.random.default_rng(seed)
        if plan is not None:
            if plan.outer.shape != (n,):
                raise ValueError(
                    f"a frozen fold plan holds {plan.outer.size} rows; the data hold {n}"
                )
            folds = plan.outer.copy()
            if not self.cross_fit and np.unique(folds).size != 1:
                raise ValueError("an in-sample fit takes one frozen fold label")
            n_folds = int(np.unique(folds).size) if self.cross_fit else 1
        elif folds is not None:
            folds = np.asarray(folds, dtype=int).copy()
            if folds.shape != (n,):
                raise ValueError(f"frozen fold labels hold {folds.size} rows; the data hold {n}")
            if not self.cross_fit and np.unique(folds).size != 1:
                raise ValueError("an in-sample fit takes one frozen fold label")
            n_folds = int(np.unique(folds).size) if self.cross_fit else 1
        elif self.cross_fit:
            folds = _stratified_folds(D, self.n_folds, rng)
            n_folds = self.n_folds
        else:
            folds = np.zeros(n, dtype=int)
            n_folds = 1
        # Before any nuisance fit, so a refused request costs no learner time.
        shape = self._memory_shape(n, J, grid.size, target_weights, L.shape[1], folds)
        check_memory(peak_bytes(**shape), self.max_memory_bytes)
        return (
            arm_labels,
            A,
            causes,
            T,
            D,
            L,
            n,
            codes,
            J,
            grid,
            target_index,
            target_weights,
            X,
            rng,
            folds,
            n_folds,
        )

    def _initial(
        self,
        data: Any,
        seed: int,
        *,
        folds: Any = None,
        plan: FoldPlan | None = None,
    ) -> _Initial:
        """Fit every nuisance and build what each walk reads, at the seed ``seed``.

        ``folds`` replaces the outer cross-fitting draw with one label per row.  ``plan``
        replaces every fold draw, outer and inner, and the generator state after them, with a
        recorded :class:`FoldPlan`.  The
        generator state after the nuisance phase is kept, so each walk's off-grid draws
        start from the same state.
        """
        (
            arm_labels,
            A,
            causes,
            T,
            D,
            L,
            n,
            codes,
            J,
            grid,
            target_index,
            target_weights,
            X,
            rng,
            folds,
            n_folds,
        ) = self._initial_inputs(data, seed, folds=folds, plan=plan)
        shape = self._memory_shape(n, J, grid.size, target_weights, L.shape[1], folds)
        gap = np.diff(np.unique(T))
        shift = 0.5 * float(gap.min()) if gap.size else 0.5
        censor_time = np.where(D > 0, T - shift, T)
        hazard_library = tuple(seed_unset(member, seed) for member in self.hazard_library)
        censoring_library = tuple(seed_unset(member, seed) for member in self.censoring_library)

        block = chunk_size(n, J, grid.size)
        sources = tuple(HazardSource(X, folds, grid, block) for _ in codes)
        censoring_source = HazardSource(X, folds, grid, block)
        p1 = np.zeros(n)
        hazard_records: list[HazardSelection] = []
        censoring_record: HazardSelection | None = None
        propensity_record: PropensitySelection | None = None
        inner_labels: list[IntArray] = []
        for position, fold in enumerate(np.unique(folds)):
            predict = folds == fold
            train = ~predict if self.cross_fit else np.ones(n, dtype=bool)
            if plan is None:
                inner = _stratified_folds(D[train], self.learner_folds, rng)
            else:
                inner = plan.inner[position].copy()
                if inner.shape != (int(train.sum()),):
                    raise ValueError(
                        "a frozen fold plan's inner labels do not fit its outer labels"
                    )
            inner_labels.append(inner.copy())
            for j, code in enumerate(codes):
                selection = select_hazard(
                    hazard_library, X[train], T[train], (D[train] == code).astype(float), inner
                )
                sources[j].add(fold, predict, selection.model)
                if fold == folds.min():
                    hazard_records.append(selection)
            selection = select_hazard(
                censoring_library,
                X[train],
                censor_time[train],
                (D[train] == 0).astype(float),
                inner,
            )
            censoring_source.add(fold, predict, selection.model)
            if fold == folds.min():
                censoring_record = selection
            p1[predict], chosen = _fit_propensity(
                self.treatment_learner, L, A, train, predict, inner, seed
            )
            if fold == folds.min():
                propensity_record = chosen
        assert censoring_record is not None
        hazards = InitialHazards(sources, censoring_source)

        # One pass over the blocks: the checks on the initial fit, the times where some hazard
        # jumps, and the censoring readers the record stores.  No estimate reads the latter.
        pi = np.stack([1.0 - p1, p1])
        # S_c(t_k-) multiplies over the observed times strictly before t_k.
        before = np.searchsorted(grid, self.horizons, side="left") - 1
        # S_c just before each row's own time (strictly before, as the targets read), at the
        # arm it received; a time past the grid's end reads the grid's last time.
        own = np.minimum(np.searchsorted(grid, T, side="left"), grid.size - 1)
        retention_at_targets = np.empty((2, n, len(self.horizons)))
        retention_at_time = np.empty(n)
        lowest = np.empty(n)
        jumping = np.zeros(grid.size, dtype=bool)
        combined = low_survival = False
        for k, (lo, hi) in enumerate(blocks(n, block)):
            dlam = hazards.jumps(lo, hi)
            if (
                k == 0
                and hazards.calls_model
                and not (
                    np.array_equal(dlam, hazards.jumps(lo, hi), equal_nan=True)
                    and np.array_equal(
                        hazards.censoring_jumps(lo, hi),
                        hazards.censoring_jumps(lo, hi),
                        equal_nan=True,
                    )
                )
            ):
                raise MethodConfigurationError(
                    "A hazard learner whose predictions change between calls cannot be "
                    "replayed: the continuous-time engine rebuilds each block of units' initial "
                    "hazards from the fitted model whenever the walk needs them, so "
                    "cumulative_hazard must return the same values for the same rows on every "
                    "call, each row predicted from that row alone."
                )
            total = dlam.sum(axis=1)
            # Both checks read the fitted hazards on these data, so a draw that trips one is a
            # data failure (recorded by a study), not an unsupported composition.  A Cox jump is
            # s_i d / sum of the risk set's scores; with one event at a time it is below one for
            # any unit in the risk set at its received arm, so the combined-jump check trips
            # only on an extrapolated cell (a counterfactual arm, or a time after the unit left
            # follow-up), or on tied events, as a bootstrap resample's copies make.
            combined = combined or bool(np.any(total >= 1.0))
            survival = np.cumprod(1.0 - total, axis=-1)
            low_survival = low_survival or bool(np.any(survival[..., -1] < SURVIVAL_UNDERFLOW))
            # A NaN jump is not zero, so its time stays on the walk's grid.
            jumping |= (dlam != 0).any(axis=(0, 1, 2))
            del dlam, total, survival
            retention = np.cumprod(
                1.0 - np.clip(hazards.censoring_jumps(lo, hi), 0.0, 1.0), axis=-1
            )
            retention_before = np.concatenate(
                [np.ones((2, hi - lo, 1)), retention[..., :-1]], axis=-1
            )
            product = pi[:, lo:hi, None] * retention_before
            lowest[lo:hi] = product.min(axis=(0, 2))
            at_targets = retention[..., np.maximum(before, 0)]
            at_targets[..., before < 0] = 1.0
            retention_at_targets[:, lo:hi] = at_targets
            retention_at_time[lo:hi] = retention_before[A[lo:hi], np.arange(hi - lo), own[lo:hi]]
            del retention, retention_before, product, at_targets
        if combined:
            raise DataError(
                "An initial hazard fit puts a combined jump of one or more at some time, so the "
                "product-limit survival is not a probability there. A unit's jump is its risk "
                "score times the baseline jump, so it reaches one only where the hazard model "
                "extrapolates: usually at the unit's counterfactual arm, for covariates outside "
                "the range that arm's units cover, where the unit's score exceeds the summed "
                "scores of everyone at risk. Tied event times, as in a bootstrap resample, "
                "multiply the baseline jump. Check the arms' overlap (the fitted treatment "
                "probabilities), or choose earlier target times or restricted horizons, or a "
                "smaller hazard library."
            )
        if low_survival:
            raise DataError(
                f"The initial survival falls below {SURVIVAL_UNDERFLOW:.0e} before the last time "
                "a target reads (the last target time or restricted horizon) for some unit and "
                "arm, beyond the range in which double precision holds the clever covariate's "
                "conditional risks. Choose an earlier last target time or restricted horizon, "
                "or a hazard library that does not extrapolate so far."
            )
        observed = T[D > 0]
        kept = walk_grid(grid, jumping, observed, target_index)
        initial = _Initial(
            n=n,
            T=T,
            A=A,
            arm_labels=arm_labels,
            causes=causes,
            grid=grid,
            target_index=target_index,
            target_weights=target_weights,
            folds=folds,
            n_folds=n_folds,
            p1=p1,
            retention_at_targets=retention_at_targets,
            retention_at_time=retention_at_time,
            hazard_records=tuple(hazard_records),
            censoring_record=censoring_record,
            propensity_record=propensity_record,
            D=D,
            memory=shape,
            hazards=hazards,
            kept=kept,
            block=block,
            lowest=lowest,
            # Risks first, then restricted means, in the order of the weight rows.
            shares=tuple(
                float(np.mean(observed <= time))
                for time in (*self.horizons, *self.restricted_horizons)
            ),
            rng_state=rng.bit_generator.state if plan is None else copy.deepcopy(plan.rng_state),
            plan=FoldPlan(
                outer=folds.copy(),
                inner=tuple(inner_labels),
                rng_state=(
                    copy.deepcopy(rng.bit_generator.state)
                    if plan is None
                    else copy.deepcopy(plan.rng_state)
                ),
            ),
        )
        return initial

    def _target(
        self,
        initial: _Initial,
        floor: float,
        *,
        settings: Mapping[str, Any],
    ) -> tuple[dict[str, ParameterEstimate], ContinuousSurvivalRecord, Any]:
        """Run the walk from ``initial`` with the weights bounded below by ``floor``.

        ``settings`` holds the values of the keys of :data:`MEASURED_DEFAULTS` the support of
        each estimate reads.  The walk runs over blocks of units (:mod:`._chunked`); the
        blocks it keeps between passes follow the memory budget, which changes how long the
        walk runs and never what it returns.
        """
        n, A, grid = initial.n, initial.A, initial.grid
        plan = _FitPlan(initial, floor)
        cutoff = default_cutoff(n) if self.tol is None else float(self.tol)
        rng = np.random.default_rng()
        rng.bit_generator.state = initial.rng_state
        off_grid = _OffGrid(plan, rng)
        # The switched variance's predictable variation reads the targeted jumps on the full
        # grid, a buffer of units at a time, in the final pass.
        variation = _Predictable(plan) if self.variance == "switched" else None
        readers = [off_grid.add] if off_grid.active else []
        if variation is not None:
            readers.append(variation.take)

        def final(k: int, lo: int, hi: int, state: Any) -> Callable[[], None]:
            # Runs on a walk thread; what each reader returns is called in block order on the
            # calling thread, so the sums never depend on n_jobs.
            thens = [reader(k, lo, hi, state) for reader in readers]

            def commit() -> None:
                for then in thens:
                    then()

            return commit

        run = _chunked.run_walk(
            plan,
            cutoff=cutoff,
            norm=self.norm,
            step=self.step_size,
            max_steps=self.max_iter,
            tilt=self.tilt,
            resident=resident_blocks(plan, self.max_memory_bytes),
            final=final if readers else None,
            n_jobs=plan.shape["n_jobs"],
        )
        result = run.result
        # Only the means and influence curves are read from here on; a walk that kept its one
        # block returns its curves on the walk's grid.
        result.dlam = _EMPTY
        result.evaluation = replace(result.evaluation, survival=_EMPTY, incidence=_EMPTY)
        off_grid_share = off_grid.share(cutoff)
        predictable = None if variation is None else variation.matrix()
        del variation
        if not result.converged:
            worst = np.argsort(-result.standardized_score)[:3]
            _warn(
                f"The targeting walk stopped after {result.steps} steps without reaching the "
                f"cutoff {cutoff:.3g}; the largest standardized scores are "
                + ", ".join(f"{result.standardized_score[p] / cutoff:.3g}x" for p in worst)
                + " the cutoff. The estimate is kept and the record lists every target.",
                ConvergenceWarning,
            )
        record = ContinuousSurvivalRecord(
            hazards=initial.hazard_records,
            censoring=initial.censoring_record,
            propensity=initial.propensity_record,
            steps=result.steps,
            halvings=result.halvings,
            converged=result.converged,
            cutoff=cutoff,
            standardized_score=tuple(float(v) for v in result.standardized_score),
            initial_psi=tuple(float(v) for v in result.initial.psi),
            bounded_share=plan.bounded_share,
            last_time_floor_share=plan.last_time_floor_share,
            frozen_share=result.frozen_share,
            off_grid_share=off_grid_share,
            cross_fitted=self.cross_fit,
            n_folds=initial.n_folds,
            grid_size=int(grid.size),
            arm=A.copy(),
            p1=initial.p1.copy(),
            retention_before=initial.retention_at_targets.copy(),
            weight_floor=float(floor),
            retention_at_time=initial.retention_at_time.copy(),
        )
        J = len(initial.causes)
        restricted = self.restricted_horizons

        def support(scale: str, q: int, relative_sd: float) -> Support:
            return continuous_survival_support(
                scale=scale,
                event_share=initial.shares[q],
                n=n,
                causes=J,
                cross_fitted=self.cross_fit,
                default_nuisances=self._default_nuisances,
                settings=settings,
                restricted=q >= len(self.horizons),
                restricted_horizons=len(restricted),
                variance_rel_sd=relative_sd,
                past_last_time=bool(restricted) and max(restricted) > max(self.horizons),
            )

        estimates, covariance = build_estimates(
            result,
            initial.arm_labels,
            initial.causes,
            self.horizons,
            n=n,
            alpha=self.alpha,
            support=support,
            restricted=restricted,
            predictable=predictable,
        )
        return estimates, record, covariance


@dataclass(frozen=True)
class FoldPlan:
    """Every fold draw of one fit, so a refit can reuse them instead of drawing.

    Parameters
    ----------
    outer : ndarray
        One outer cross-fitting label per row; one label on an in-sample fit.
    inner : tuple of ndarray
        Per outer fold, in label order, the selection fold of each training row.
    rng_state : mapping
        The generator state after the nuisance phase, from which the off-grid draws start.
        No nuisance fit draws from the generator, so it is the state after the last fold draw.
    """

    outer: IntArray
    inner: tuple[IntArray, ...]
    rng_state: Mapping[str, Any]


@dataclass
class _Initial:
    """What every walk of one fit reads: the nuisance fits and the fixed per-unit arrays.

    ``hazards`` rebuilds the initial hazard jumps of any block of units from the fitted
    models.  ``kept`` holds the grid positions the walk runs on (the event-support grid), and
    ``block`` the units per block.  ``lowest`` is each unit's smallest
    :math:`\\hat\\pi(a \\mid L)\\hat S_c(u^- \\mid a, L)` over both arms and the grid, which a
    truncation curve reads.  ``rng_state`` is the generator state after the nuisance phase,
    from which every walk's off-grid draws start.
    """

    n: int
    T: FloatArray
    A: IntArray
    D: IntArray
    arm_labels: tuple[str, str]
    causes: dict[int, str]
    grid: FloatArray
    target_index: IntArray
    target_weights: FloatArray
    folds: IntArray
    n_folds: int
    p1: FloatArray
    retention_at_targets: FloatArray
    retention_at_time: FloatArray
    hazard_records: tuple[HazardSelection, ...]
    censoring_record: HazardSelection
    propensity_record: PropensitySelection | None
    memory: dict[str, Any]
    hazards: InitialHazards
    kept: IntArray
    block: int
    lowest: FloatArray
    shares: tuple[float, ...]
    rng_state: Mapping[str, Any]
    plan: FoldPlan


class _FitPlan:
    """The blocks of one walk of a fit: each block's inputs on the walk's grid at one floor."""

    def __init__(self, initial: _Initial, floor: float) -> None:
        self.fit = initial
        self.floor = float(floor)
        self.n = initial.n
        J = len(initial.causes)
        Q = initial.target_weights.shape[0]
        self.n_targets = 2 * J * Q
        self.bounds = blocks(initial.n, initial.block)
        grid, kept = initial.grid, initial.kept
        self.reduced = kept.size < grid.size
        self.times = grid[kept]
        self.dropped = np.ones(grid.size, dtype=bool)
        self.dropped[kept] = False
        # A kept time preceded by a dropped one; a trailing run of dropped times carries no
        # target weight after it, so its direction is zero.
        previous = np.concatenate([[-1], kept[:-1]])
        self._gap_ends = np.flatnonzero(kept - previous > 1)
        weights = initial.target_weights
        points = _point_rows(weights)
        self.expand: IntArray | None = None
        self.kept: IntArray | None = None
        if not self.reduced:
            self.weights = weights
        elif points:
            # A risk's weight row is one at a kept time, so the reduced row picks it exactly.
            self.weights = np.take(weights, kept, axis=-1)
        else:
            # A restricted mean's weights sit on dropped times too: its sums run on the full
            # grid, reading the reduced curves through ``expand``.
            self.weights = weights
            self.expand = np.searchsorted(kept, np.arange(grid.size), side="right") - 1
            self.kept = kept
        self.pi = np.stack([1.0 - initial.p1, initial.p1])
        #: The memory model's arguments for this walk: its grid, blocks and target weights.
        self.shape = {
            **initial.memory,
            "kept": int(kept.size),
            "block": int(initial.block),
            "restricted": self.expand is not None,
        }
        self._largest_dropped = np.zeros((2, initial.n))
        self._counted = np.zeros(len(self.bounds), dtype=bool)
        self._bounded = self._relevant = self._last = 0
        # Blocks may be built on several threads; the counts are sums of integers, so only
        # their updates need the lock.
        self._count = threading.Lock()

    def _weight(self, k: int) -> FloatArray:
        """``(2, c, M)`` the bounded inverse weight of block ``k`` on the full grid."""
        lo, hi = self.bounds[k]
        product = self._product(lo, hi)
        if not self._counted[k]:
            initial = self.fit
            self._counted[k] = True
            factual = initial.A[None, lo:hi, None] == np.arange(2)[:, None, None]
            reach = initial.T[None, lo:hi, None] >= initial.grid[None, None, :]
            relevant = factual & reach
            counts = (
                int(np.count_nonzero(relevant)),
                int(np.count_nonzero(product[relevant] < self.floor)),
                int(np.count_nonzero(product[..., -1] < self.floor)),
            )
            del factual, reach, relevant
            with self._count:
                self._relevant += counts[0]
                self._bounded += counts[1]
                self._last += counts[2]
        return self._inverse(lo, hi, product)

    def _inverse(self, lo: int, hi: int, product: FloatArray) -> FloatArray:
        """``(2, hi - lo, M)`` the inverse weights of units ``lo:hi`` from their product."""
        return 1.0 / np.maximum(product, self.floor)

    def _product(self, lo: int, hi: int) -> FloatArray:
        """``(2, hi - lo, M)`` the propensity times the censoring retention just before."""
        retention = np.cumprod(
            1.0 - np.clip(self.fit.hazards.censoring_jumps(lo, hi), 0.0, 1.0), axis=-1
        )
        retention_before = np.concatenate([np.ones((2, hi - lo, 1)), retention[..., :-1]], axis=-1)
        del retention
        return self.pi[:, lo:hi, None] * retention_before

    @property
    def bounded_share(self) -> float:
        """Received-arm cells while followed whose product the floor raised, over all blocks."""
        assert self._counted.all()
        return self._bounded / self._relevant if self._relevant else 0.0

    @property
    def last_time_floor_share(self) -> float:
        """Pairs whose product at the grid's end is below the floor, over all blocks."""
        assert self._counted.all()
        return self._last / (2 * self.n)

    def problem(self, k: int) -> Problem:
        """The fixed inputs of block ``k`` on the walk's grid.

        Parameters
        ----------
        k : int
            The block.

        Returns
        -------
        Problem
            The block's inputs.
        """
        lo, hi = self.bounds[k]
        initial = self.fit
        weight = self._weight(k)
        if self.reduced:
            self._largest_dropped[:, lo:hi] = weight[..., self.dropped].max(axis=-1)
            weight = np.take(weight, initial.kept, axis=-1)
        T, D = initial.T[lo:hi], initial.D[lo:hi]
        at_risk = T[:, None] >= self.times[None, :]
        events = np.stack(
            [(T[:, None] == self.times[None, :]) & (D[:, None] == code) for code in initial.causes]
        )
        return Problem(
            initial.A[lo:hi], at_risk, events, weight, self.weights, self.expand, self.kept
        )

    def initial(self, k: int) -> FloatArray:
        """``(2, J, c, M)`` the initial jumps of block ``k`` on the walk's grid.

        Parameters
        ----------
        k : int
            The block.

        Returns
        -------
        ndarray
            The jumps.
        """
        lo, hi = self.bounds[k]
        jumps = self.fit.hazards.jumps(lo, hi)
        # np.take keeps the block C-ordered, as the dense arrays are.
        return np.take(jumps, self.fit.kept, axis=-1) if self.reduced else jumps

    def dropped_weight(self, k: int) -> FloatArray | None:
        """``(2, c)`` each pair's largest weight at a dropped time, ``None`` with none dropped.

        Parameters
        ----------
        k : int
            The block.

        Returns
        -------
        ndarray or None
            The largest weights.
        """
        if not self.reduced:
            return None
        lo, hi = self.bounds[k]
        return self._largest_dropped[:, lo:hi]

    def gap_ends(self) -> IntArray:
        """The walk-grid times that a run of dropped times precedes.

        Returns
        -------
        ndarray
            Positions on the walk's grid.
        """
        return self._gap_ends

    def overflow(
        self, k: int, units: IntArray, state: Any, v: FloatArray, step: float
    ) -> np.ndarray | None:
        """``(2, len(units))`` the pairs whose dense increment overflows at a dropped time.

        The direction of the units is computed on the full grid with the dense operations,
        from the block's curves read at every full-grid time.

        Parameters
        ----------
        k : int
            The block.
        units : ndarray
            Positions within the block.
        state : _State
            The block's current hazards and curves.
        v : ndarray
            ``(P,)`` the direction in target space.
        step : float
            The step size.

        Returns
        -------
        ndarray or None
            The overflowing pairs.
        """
        lo, hi = self.bounds[k]
        initial = self.fit
        expand = np.searchsorted(initial.kept, np.arange(initial.grid.size), side="right") - 1
        weights = state.problem.target_weights
        if state.problem.expand is None:
            # The walk's own weight rows, read on the full grid (zero at the dropped times).
            weights = np.zeros((weights.shape[0], initial.grid.size))
            weights[:, initial.kept] = state.problem.target_weights
        out = np.zeros((2, units.size), dtype=bool)
        # The units run in runs of rows short enough that the full-grid arrays of one run stay
        # within a block's arrays on the walk's grid (memory model, ``overflow_rows``).
        rows = overflow_rows(hi - lo, initial.kept.size, initial.grid.size)
        for start in range(0, hi - lo, rows):
            inside = (units >= start) & (units < start + rows)
            if not inside.any():
                continue
            chosen = units[inside]
            survival = state.survival[:, chosen]
            padded = np.concatenate([np.ones((*survival.shape[:-1], 1)), survival], axis=-1)
            survival = np.take(padded, expand + 1, axis=-1)
            del padded
            # The jumps on the full grid: zero at every dropped time, as the dense walk's are.
            jumps = np.zeros((*state.jumps.shape[:2], chosen.size, initial.grid.size))
            jumps[..., initial.kept] = state.jumps[:, :, chosen]
            first, last = int(chosen.min()), int(chosen.max())
            span = (lo + first, lo + last + 1)
            weight = self._inverse(*span, self._product(*span))[:, chosen - first]
            at = lo + chosen
            at_risk = initial.T[at, None] >= initial.grid[None, :]
            events = np.zeros((len(initial.causes), *at_risk.shape), dtype=bool)
            full = Problem(initial.A[at], at_risk, events, weight, weights)
            light = _chunked._light(survival, _chunked.T._EMPTY)
            direction = _chunked.T._direction(full, light, v, jumps)
            del survival, jumps, light, weight, at_risk, events, full
            with np.errstate(over="ignore"):
                grows = np.isinf(np.exp(direction[..., self.dropped] * step))
            out[:, np.flatnonzero(inside)] = grows.any(axis=(1, 3))
            del direction, grows
        return out


class _Predictable:
    r"""The switched variance's predictable variation, accumulated in the final pass.

    The final pass hands over each block of the walk's targeted jumps on the walk's grid
    (:meth:`take`, on the block's thread), and :meth:`add` takes them in row order on the
    calling thread, so the result is the same bits for any ``n_jobs``.  They are copied onto
    the full grid (zero at the dropped times, which no step moves) into a buffer of
    :func:`~cleverly.survival._targeting.predictable_rows` units, cut where
    :func:`~cleverly.survival._targeting.predictable_covariance` cuts its blocks.  When the
    buffer is full its units' curves, bounded weight and floored ratio
    :math:`\min(1, \hat\pi \hat S_c / g_{\min})` are built and added to a
    :class:`~cleverly.survival._targeting.PredictableVariation`, so the pass holds one buffer
    and every unit's plug-ins, whatever the walk's blocks, and its martingale term is the
    dense matrix's bit for bit.
    """

    def __init__(self, plan: _FitPlan) -> None:
        initial = plan.fit
        self.plan = plan
        self.J = len(initial.causes)
        self.M = initial.grid.size
        Q = initial.target_weights.shape[0]
        self.size = predictable_rows(self.J, Q, self.M)
        self.variation = PredictableVariation(initial.n, self.J, Q)
        self.start = 0
        self.buffer: FloatArray | None = None

    def take(self, k: int, lo: int, hi: int, state: Any) -> Callable[[], None]:
        """Hold block ``k``'s targeted jumps for :meth:`add`, which the walk calls in block order.

        Called on a walk thread; nothing is written here, so the buffer and the variation are
        only ever touched on the calling thread, in row order, whatever ``n_jobs``.

        Parameters
        ----------
        k : int
            The block.
        lo : int
            Its first unit.
        hi : int
            One past its last unit.
        state : _State
            Its targeted hazards.

        Returns
        -------
        callable
            Adds the block's jumps (:meth:`add`).
        """
        jumps = state.jumps

        def carry() -> None:
            self.add(k, lo, hi, jumps)

        return carry

    def add(self, k: int, lo: int, hi: int, jumps: FloatArray) -> None:
        """Copy block ``k``'s targeted jumps into the buffer, adding each buffer as it fills.

        Parameters
        ----------
        k : int
            The block.
        lo : int
            Its first unit.
        hi : int
            One past its last unit.
        jumps : ndarray
            ``(2, J, hi - lo, G)`` its targeted jumps on the walk's grid.
        """
        initial = self.plan.fit
        at = lo
        while at < hi:
            stop = min(self.start + self.size, initial.n)
            if self.buffer is None:
                self.buffer = np.zeros((2, self.J, stop - self.start, self.M))
            end = min(hi, stop)
            rows = slice(at - self.start, end - self.start)
            source = jumps[:, :, at - lo : end - lo]
            if self.plan.reduced:
                self.buffer[:, :, rows][..., initial.kept] = source
            else:
                self.buffer[:, :, rows] = source
            at = end
            if end == stop:
                self._flush(stop)

    def _flush(self, stop: int) -> None:
        """Add the buffer's units ``self.start:stop`` and release it."""
        plan, initial = self.plan, self.plan.fit
        start, dlam = self.start, self.buffer
        assert dlam is not None
        self.buffer = None
        b = stop - start
        product = plan._product(start, stop)
        weight = plan._inverse(start, stop, product)
        # pi S_c w^2 = w min(1, pi S_c / floor): the predictable variation reads the floored
        # weight squared, as P_n D^2 does.
        floored = product / plan.floor
        np.minimum(floored, 1.0, out=floored)
        del product
        # PredictableVariation reads the indicators' shapes only.
        problem = Problem(
            initial.A[start:stop],
            np.broadcast_to(np.zeros(1, dtype=bool), (b, self.M)),
            np.broadcast_to(np.zeros(1, dtype=bool), (self.J, b, self.M)),
            weight,
            initial.target_weights,
        )
        # The curves are passed without a name here, so the incidences are released once the
        # plug-ins are read.
        self.variation.add(
            problem, Evaluation(_EMPTY, _EMPTY, _EMPTY, *curves(dlam)), dlam, floored
        )
        self.start = stop

    def matrix(self) -> FloatArray:
        """``(P, P)`` the second moments, once every unit has been added.

        Returns
        -------
        ndarray
            The matrix of :func:`~cleverly.survival._targeting.predictable_covariance` at the
            targeted jumps on the full grid.
        """
        return self.variation.matrix()


def walk_grid(
    grid: FloatArray, jumping: np.ndarray, observed: FloatArray, target_index: IntArray
) -> IntArray:
    """The positions on the full grid that the walk keeps: the event-support grid.

    Every time where some initial hazard jumps (a NaN jump counts), some event is observed or
    a risk's target sits.  Elsewhere every jump is zero and stays zero along the walk, so the
    curves there repeat the last kept time's.

    Parameters
    ----------
    grid : ndarray
        ``(M,)`` the full grid.
    jumping : ndarray
        ``(M,)`` whether some initial hazard jump is not zero there.
    observed : ndarray
        The observed event times.
    target_index : ndarray
        The risks' positions on the grid.

    Returns
    -------
    ndarray
        The kept positions, increasing; every position with the grid reduction off.
    """
    if not _chunked.REDUCE_GRID:
        return np.arange(grid.size)
    support = jumping | np.isin(grid, observed)
    support[target_index] = True
    return np.flatnonzero(support)


def _point_rows(weights: FloatArray) -> bool:
    """Whether every weight row is a unit vector, a risk's."""
    return bool(np.all((weights == 0) | (weights == 1)) and np.all(weights.sum(axis=1) == 1))


def resident_blocks(plan: _FitPlan, max_memory_bytes: int | None) -> int:
    """The blocks a walk keeps between passes, the first ones in row order.

    Every block when the budget is unknown; otherwise as many as the memory model lets the
    budget hold beside the walk's fixed and working arrays.  It never refuses, and it changes
    only how often a pass rebuilds a block, never a result.

    Parameters
    ----------
    plan : _FitPlan
        The walk's blocks.
    max_memory_bytes : int or None
        The fit's budget.

    Returns
    -------
    int
        The kept blocks.
    """
    return resident_count(plan.shape, plan.bounds, max_memory_bytes)


class _OffGrid:
    """Item 10 Table 4: the share of random times at which the score equation holds.

    The random times are drawn before the walk from the generator state after the nuisance
    phase, and each block's rows are added in the final pass.  The means over units are carried
    across blocks in row order, the order of NumPy's mean over the first axis, and the
    influence curves are kept in one ``(n, 2J x times)`` array for their spread.
    """

    def __init__(self, plan: _FitPlan, rng: np.random.Generator) -> None:
        initial = plan.fit
        grid = initial.grid
        draws = np.sort(rng.uniform(0.0, float(grid[-1]), OFF_GRID_TIMES))
        index = np.searchsorted(grid, draws, side="right") - 1
        keep = index >= 0
        self.active = bool(keep.any())
        if not self.active:
            return
        if plan.reduced:
            # Each time reads the last kept grid time at or before it; a time before the first
            # kept one scores zero, as on the full grid.
            position = np.searchsorted(initial.kept, index[keep], side="right") - 1
            self.weights = np.zeros((position.size, initial.kept.size))
            inside = position >= 0
            self.weights[np.flatnonzero(inside), position[inside]] = 1.0
        else:
            self.weights = point_weights(index[keep], grid.size)
        self.n = initial.n
        self.width = 2 * len(initial.causes) * self.weights.shape[0]
        # Allocated in the final pass, so the walk does not hold it.
        self.spread: FloatArray | None = None
        self.parts: FloatArray | None = None
        self.plugins: FloatArray | None = None
        self._allocate = threading.Lock()

    def add(self, k: int, lo: int, hi: int, state: Any) -> Callable[[], None]:
        """Compute block ``k``'s rows at its targeted hazards; return what adds them.

        The block's influence rows are written in place; its sums are carried into the means
        by the returned callable, which the walk calls in block order.

        Parameters
        ----------
        k : int
            The block.
        lo : int
            Its first unit.
        hi : int
            One past its last unit.
        state : _State
            Its targeted hazards and curves.

        Returns
        -------
        callable
            Adds the block's parts and plug-ins to the sums carried over blocks.
        """
        shifted = replace(state.problem, target_weights=self.weights, expand=None, kept=None)
        parts, plugins = evaluate_rows(shifted, state.jumps, state.survival, state.incidence)
        with self._allocate:
            if self.spread is None:
                self.spread = np.empty((self.n, self.width))
        np.add(parts, plugins, out=self.spread[lo:hi])

        def carry() -> None:
            self.parts = _carried(self.parts, parts)
            self.plugins = _carried(self.plugins, plugins)

        return carry

    def share(self, cutoff: float) -> float:
        """The share of informative random times whose score is within the cutoff.

        Parameters
        ----------
        cutoff : float
            The walk's cutoff.

        Returns
        -------
        float
            The share, ``1.0`` when no time is informative.
        """
        if not self.active:
            return 1.0
        assert self.parts is not None and self.plugins is not None and self.spread is not None
        score = self.parts / self.n
        psi = self.plugins / self.n
        influence = self.spread
        influence -= psi
        np.square(influence, out=influence)
        sigma = np.sqrt(np.mean(influence, axis=0))
        self.spread = None
        #: The vectors the share read, kept for the tests that compare them to the dense walk's.
        self.vectors = (score, psi, sigma)
        informative = sigma > 0
        if not informative.any():
            return 1.0
        ok = np.abs(score[informative]) <= cutoff * sigma[informative]
        return float(np.mean(ok))


def _carried(total: FloatArray | None, rows: FloatArray) -> FloatArray:
    """The sum over the first axis of every row so far, added in row order."""
    if total is None:
        return rows.sum(axis=0)
    return np.concatenate([total[None, :], rows]).sum(axis=0)


@dataclass(frozen=True)
class _Units:
    """The row count the bootstrap draws from; a continuous-time fit has no clusters."""

    n: int
    cluster: None = None

    def subset(self, index: Any) -> Any:  # pragma: no cover - the draws never subset it
        raise NotImplementedError


def replicate_plan(
    index: IntArray,
    event: IntArray,
    *,
    cross_fit: bool,
    n_folds: int,
    learner_folds: int,
    seed: int,
) -> FoldPlan:
    """The fold plan of a bootstrap replicate: every fold drawn on its distinct units.

    The units of ``index`` are drawn into the outer folds stratified by their event code, as a
    fit draws its rows (:func:`_stratified_folds`), and within each outer fold's training units
    into the selections' inner folds, so every copy of a unit has its unit's outer and inner
    labels.  Drawn on rows instead, a copy could train a selection that validates its twin,
    and the cross-validated loss would favour the larger models on bootstrap samples (Cai and
    van der Laan 2020, Section 4).

    Parameters
    ----------
    index : ndarray
        ``(n,)`` the original row of each resampled row, repeats kept.
    event : ndarray
        ``(n,)`` the event code of each resampled row.
    cross_fit : bool
        Whether the replicate is cross-fitted; otherwise every row takes outer label ``0``.
    n_folds : int
        The outer folds of a cross-fitted replicate.
    learner_folds : int
        The selections' inner folds.
    seed : int
        The replicate's seed, from which the draws start.

    Returns
    -------
    FoldPlan
        The outer label of each resampled row, each outer fold's inner labels of its training
        rows, and the generator state after the draws.
    """
    units, inverse = np.unique(np.asarray(index), return_inverse=True)
    unit_event = np.empty(units.size, dtype=int)
    unit_event[inverse] = np.asarray(event, dtype=int)
    rng = np.random.default_rng(seed)
    if cross_fit:
        unit_outer = _stratified_folds(unit_event, n_folds, rng)
    else:
        unit_outer = np.zeros(units.size, dtype=int)
    outer = unit_outer[inverse]
    inner = []
    for fold in np.unique(outer):
        training = unit_outer != fold if cross_fit else np.ones(units.size, dtype=bool)
        unit_inner = np.full(units.size, -1, dtype=int)
        unit_inner[training] = _stratified_folds(unit_event[training], learner_folds, rng)
        rows = outer != fold if cross_fit else np.ones(outer.size, dtype=bool)
        inner.append(unit_inner[inverse][rows])
    return FoldPlan(
        outer=outer, inner=tuple(inner), rng_state=copy.deepcopy(rng.bit_generator.state)
    )


def resample(data: Any, index: IntArray) -> Any:
    """The container of the rows ``index`` of ``data``, repeats kept, in that order.

    Parameters
    ----------
    data : ContinuousSurvivalData
        The fit's rows.
    index : ndarray
        The rows to take.

    Returns
    -------
    ContinuousSurvivalData
        The resampled container.
    """
    return replace(
        data,
        time=np.asarray(data.time)[index],
        event=np.asarray(data.event)[index],
        treatment=np.asarray(data.treatment)[index],
        baseline=np.asarray(data.baseline)[index],
    )


def _drawn_replicate(
    estimator: ContinuousSurvivalTMLE, data: Any, sequence: Any, seed: int
) -> tuple[dict[str, tuple[float, float]] | None, str | None]:
    """:func:`_replicate` on the row index that ``sequence`` draws, as the shared bootstrap
    draws it (:func:`~cleverly.inference.bootstrap._bootstrap_draw`)."""
    return _replicate(estimator, data, _replicate_index(data, sequence), seed)


def _replicate_index(data: Any, sequence: Any) -> IntArray:
    """The row index of the replicate whose seed sequence is ``sequence``, as the shared
    bootstrap draws it (:func:`~cleverly.inference.bootstrap._bootstrap_draw`)."""
    from ..inference.bootstrap import _bootstrap_draw

    n = int(np.asarray(data.time).shape[0])
    return _bootstrap_draw(n, None, np.random.default_rng(sequence))[0]


def _replicate(
    estimator: ContinuousSurvivalTMLE, data: Any, index: IntArray, seed: int
) -> tuple[dict[str, tuple[float, float]] | None, str | None]:
    """One bootstrap replicate: each estimate and standard error, or why the refit failed.

    The failures a resampled fit can legitimately raise are caught and named, so the fit can
    count them and give their reasons; none is redrawn.  Any other exception is a
    defect and propagates.
    """
    try:
        sample = resample(data, index)
        plan = replicate_plan(
            index,
            sample.event,
            cross_fit=estimator.cross_fit,
            n_folds=estimator.n_folds,
            learner_folds=estimator.learner_folds,
            seed=seed,
        )
        fit = estimator._copy(seed)._fit_data(sample, plan=plan)
    # The engine's refusals on the drawn data (a lost cause or event, a combined jump, an
    # underflow, a memory refusal) are CleverlyErrors; a singular system in a learner raises
    # LinAlgError, and floating-point signals that a learner turns into errors raise
    # FloatingPointError.  A TypeError, IndexError or any other exception is a defect, and
    # counting it as a failed refit would hide it below the 1% share.
    except (CleverlyError, np.linalg.LinAlgError, FloatingPointError) as error:
        message = str(error).split(". ")[0].strip()
        return None, f"{type(error).__name__}: {message}" if message else type(error).__name__
    values = {
        name: (float(estimate.psi), float(estimate.std_error))
        for name, estimate in fit.estimates.items()
    }
    return values, None
