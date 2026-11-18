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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from .._refusal import NO_ALTERNATIVE, RefusalReason
from .._support import Support
from .._typing import FloatArray, IntArray
from .._warnings import warn as _warn
from ..exceptions import CapabilityError, ConvergenceWarning, DataError
from ..inference.multiplier import DEFAULT_BAND_SCALE, DEFAULT_KIND, DEFAULT_N_REPLICATES
from ..learners._fitting import seed_unset
from ..utils.memory import check_max_memory_bytes
from ..utils.random import note_fresh_entropy
from ._memory import check_memory, largest_fold, peak_bytes
from ._targeting import (
    Norm,
    Problem,
    WalkResult,
    default_cutoff,
    point_weights,
    restricted_weights,
    walk,
)
from .cox import HazardSelection, StratifiedNelsonAalen, default_library, select_hazard
from .result import (
    BAND_SUPPORT,
    MEASURED_DEFAULTS,
    ContinuousSurvivalResult,
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
#: An initial survival at or below this before the last time a target reads is refused.
SURVIVAL_FLOOR: Final = 1e-12
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


def _jumps(cumulative: FloatArray) -> FloatArray:
    return np.diff(np.concatenate([np.zeros((cumulative.shape[0], 1)), cumulative], axis=1), axis=1)


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
    max_memory_bytes : int or None
        The memory budget of a fit, in bytes.  Before fitting any nuisance, the fit predicts the
        largest number of bytes its arrays hold at once, which grows as ``n`` times the number of
        distinct observed times up to the last target time or restricted horizon, and refuses
        with :class:`~cleverly.exceptions.MethodConfigurationError` when the prediction exceeds
        the budget.  ``None`` reads the memory available at that moment and checks nothing on
        a platform that does not report it.  A value overrides the reading either way.  Every
        refit of a result, and so every refutation and truncation curve, uses the value of
        the fit.
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
        max_memory_bytes: int | None = None,
    ) -> None:
        budget = check_max_memory_bytes(max_memory_bytes)
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
        self.max_memory_bytes = budget
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
    ) -> PlannedFits:
        """The model fits :meth:`refit` would make on the same arguments, counted unfitted.

        :func:`cleverly._planned_fits.refit_plan` reads it.
        """
        folds, plan = self._refit_split(data, _frozen_folds, _frozen_plan)
        return self._copy(random_state)._planned_fit_data(data, folds=folds, plan=plan)

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
        estimates, record = self._target(
            initial, self._floor(initial.n), settings=settings, consume=True
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
        return ContinuousSurvivalResult(
            estimates=estimates,
            simultaneous=bands,
            record=record,
            arms=initial.arm_labels,
            causes=tuple(initial.causes.values()),
            horizons=self.horizons,
            n=initial.n,
            backend=data.backend,
            restricted_horizons=self.restricted_horizons,
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

    def _planned_fit_data(
        self, data: Any, *, folds: Any = None, plan: FoldPlan | None = None
    ) -> PlannedFits:
        """The model fits :meth:`fit_data` would make, counted without fitting.

        Runs the same checks before the first fit (:meth:`_initial_inputs`).  Per outer fold
        each cause's hazard and the censoring hazard are chosen from their library by
        cross-validation over the inner folds, ``M * V`` fits and one refit of the winner (one
        fit for a library of one), and the treatment model is the user's learner, one fit,
        or the selection over the four default logistic models, ``4 * V + 1`` fits.  ``V`` is
        the inner fold count the stratified labels reach on the fold's training rows.

        Parameters
        ----------
        data : ContinuousSurvivalData
            The container.
        folds : array-like or None
            One outer label per row in place of a fresh draw, as :meth:`_fit_data` takes it.
        plan : FoldPlan or None
            A recorded fold plan in place of every fold draw, as :meth:`_fit_data` takes it.

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
        names = ", ".join(learner_name(member) for member in hazard_library)
        for fold in np.unique(folds):
            train = folds != fold if self.cross_fit else np.ones(n, dtype=bool)
            observed = np.unique(D[train], return_counts=True)[1]
            inner = min(self.learner_folds, int(observed.max()))
            for _ in codes:
                builder.add(
                    "hazard_library",
                    "cause-specific hazard selection, per cause and fold",
                    names,
                    selection(hazard_library, inner),
                )
            builder.add(
                "censoring_library",
                "censoring hazard selection, per fold",
                ", ".join(learner_name(member) for member in censoring_library),
                selection(censoring_library, inner),
            )
            if self.treatment_learner is not None:
                count = count_learner(seed_unset(self.treatment_learner, seed), A[train])
                builder.add(
                    "treatment_learner",
                    "treatment model, per fold",
                    learner_name(self.treatment_learner),
                    count,
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
            )
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

    def _initial_inputs(
        self,
        data: Any,
        seed: int,
        *,
        folds: Any = None,
        retained: bool = False,
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
        retained : bool
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
        for a in (0, 1):
            for code, label in causes.items():
                first_event = T[(a == A) & (code == D)]
                if first_event.size == 0 or first_event.min() > first:
                    what = "risk" if first_kind == "target time" else "restricted mean"
                    raise _gap(
                        f"Arm {arm_labels[a]!r} has no event of cause {label!r} at or before "
                        f"the first {first_kind} {_time_text(first)}, so that {what}'s "
                        "influence curve is zero and its interval undefined. Choose a later "
                        f"first {first_kind}."
                    )
        for a in (0, 1):
            followed = float(T[a == A].max())
            for end, kind, what in ends:
                if followed < end:
                    raise _gap(
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
        check_memory(
            peak_bytes(
                n,
                J,
                grid.size,
                target_weights.shape[0],
                covariates=L.shape[1],
                fold_size=largest_fold(folds, n),
                simultaneous=self.simultaneous,
                n_multiplier=self.n_multiplier,
                retained=retained,
                factual_tilt=self.tilt == "factual",
                stratified_learner=any(
                    isinstance(member, StratifiedNelsonAalen)
                    for member in (*self.hazard_library, *self.censoring_library)
                ),
            ),
            self.max_memory_bytes,
        )
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
        retained: bool = False,
        plan: FoldPlan | None = None,
    ) -> _Initial:
        """Fit every nuisance and build what each walk reads, at the seed ``seed``.

        ``folds`` replaces the outer cross-fitting draw with one label per row.  ``plan``
        replaces every fold draw, outer and inner, and the generator state after them, with a
        recorded :class:`FoldPlan`.  ``retained``
        tells the memory check that the caller keeps the initial jumps and weight product
        through every walk, as a truncation curve does.  The
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
        ) = self._initial_inputs(data, seed, folds=folds, retained=retained, plan=plan)
        gap = np.diff(np.unique(T))
        shift = 0.5 * float(gap.min()) if gap.size else 0.5
        censor_time = np.where(D > 0, T - shift, T)
        hazard_library = tuple(seed_unset(member, seed) for member in self.hazard_library)
        censoring_library = tuple(seed_unset(member, seed) for member in self.censoring_library)

        dlam = np.zeros((2, J, n, grid.size))
        dgam = np.zeros((2, n, grid.size))
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
                for a in (0, 1):
                    Xa = X[predict].copy()
                    Xa[:, 0] = a
                    dlam[a, j][predict] = _jumps(selection.model.cumulative_hazard(Xa, grid))
                if fold == folds.min():
                    hazard_records.append(selection)
            selection = select_hazard(
                censoring_library,
                X[train],
                censor_time[train],
                (D[train] == 0).astype(float),
                inner,
            )
            for a in (0, 1):
                Xa = X[predict].copy()
                Xa[:, 0] = a
                dgam[a][predict] = _jumps(selection.model.cumulative_hazard(Xa, grid))
            if fold == folds.min():
                censoring_record = selection
            p1[predict], chosen = _fit_propensity(
                self.treatment_learner, L, A, train, predict, inner, seed
            )
            if fold == folds.min():
                propensity_record = chosen
        assert censoring_record is not None

        # Both checks read the fitted hazards on these data, so a draw that trips one is a
        # data failure (recorded by a study), not an unsupported composition.
        if np.any(dlam.sum(axis=1) >= 1.0):
            raise DataError(
                "An initial hazard fit puts a combined jump of one or more at some time, so the "
                "product-limit survival is not a probability there. A Breslow jump can exceed "
                "one where few units are at risk; choose earlier target times or restricted "
                "horizons, or a smaller library."
            )
        survival = np.cumprod(1.0 - dlam.sum(axis=1), axis=-1)
        if np.any(survival[..., -1] <= SURVIVAL_FLOOR):
            raise DataError(
                "The initial survival reaches zero before the last time a target reads (the "
                "last target time or restricted horizon) for some unit and arm, so the clever "
                "covariate divides by zero. The estimator needs the survival bounded away from "
                "zero there; choose an earlier last target time or restricted horizon."
            )
        retention = np.cumprod(1.0 - np.clip(dgam, 0.0, 1.0), axis=-1)
        retention_before = np.concatenate([np.ones((2, n, 1)), retention[..., :-1]], axis=-1)
        pi = np.stack([1.0 - p1, p1])
        product = pi[:, :, None] * retention_before
        # Stored for the overlap readers; read before the arrays below are released, and no
        # estimate reads them.
        # S_c(t_k-) multiplies over the observed times strictly before t_k.
        before = np.searchsorted(grid, self.horizons, side="left") - 1
        retention_at_targets = retention[..., np.maximum(before, 0)]
        retention_at_targets[..., before < 0] = 1.0
        # S_c just before each row's own time (strictly before, as the targets read), at the
        # arm it received; a time past the grid's end reads the grid's last time.
        own = np.minimum(np.searchsorted(grid, T, side="left"), grid.size - 1)
        retention_at_time = retention_before[A, np.arange(n), own]
        # Each of these is (2, n, M); at n = 4,000 one is ~200 MB, so none outlives its use.
        del survival, retention, retention_before, dgam

        # Indicators stay boolean: arithmetic with the float hazards promotes them on use.
        at_risk = T[:, None] >= grid[None, :]
        events = np.stack([(T[:, None] == grid[None, :]) & (D[:, None] == code) for code in codes])
        observed = T[D > 0]
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
            at_risk=at_risk,
            events=events,
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
            product=product,
            jumps=[dlam],
        )
        # Handed over through the container's list so no name here keeps the initial jumps
        # alive once a consuming walk has replaced them (~400 MB at n = 4,000).
        del dlam, product
        return initial

    def _target(
        self,
        initial: _Initial,
        floor: float,
        *,
        settings: Mapping[str, Any],
        consume: bool,
    ) -> tuple[dict[str, ParameterEstimate], ContinuousSurvivalRecord]:
        """Run the walk from ``initial`` with the weights bounded below by ``floor``.

        ``settings`` holds the values of the keys of :data:`MEASURED_DEFAULTS` the support of
        each estimate reads.  ``consume`` releases the initial jumps and the bounded product
        into the walk, as a fit does; a truncation curve keeps both for its next floor.
        """
        n, A, T, grid = initial.n, initial.A, initial.T, initial.grid
        if consume:
            assert initial.product is not None
            product = initial.product
            initial.product = None
        else:
            assert initial.product is not None
            product = initial.product
        weight = 1.0 / np.maximum(product, floor)
        factual = A[None, :, None] == np.arange(2)[:, None, None]
        reach = T[None, :, None] >= grid[None, None, :]
        relevant = factual & reach
        bounded_share = float(np.mean(product[relevant] < floor)) if relevant.any() else 0.0
        last_time_floor_share = float(np.mean(product[..., -1] < floor))
        del product, factual, reach, relevant

        problem = Problem(A, initial.at_risk, initial.events, weight, initial.target_weights)
        cutoff = default_cutoff(n) if self.tol is None else float(self.tol)
        # A consuming walk takes the initial jumps with no name here keeping them, so they are
        # freed at its first accepted step rather than held beside every later trial.
        result = walk(
            problem,
            initial.jumps.pop() if consume else initial.jumps[0],
            cutoff=cutoff,
            norm=self.norm,
            step=self.step_size,
            max_steps=self.max_iter,
            tilt=self.tilt,
        )
        if not result.converged:
            worst = np.argsort(-result.standardized_score)[:3]
            _warn(
                f"The targeting walk stopped after {result.steps} steps without reaching the "
                f"cutoff {cutoff:.3g}; the largest standardized scores are "
                + ", ".join(f"{result.standardized_score[p] / cutoff:.3g}x" for p in worst)
                + " the cutoff. The estimate is kept and the record lists every target.",
                ConvergenceWarning,
            )
        rng = np.random.default_rng()
        rng.bit_generator.state = initial.rng_state
        off_grid = _off_grid_share(problem, result, grid, rng, cutoff)
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
            bounded_share=bounded_share,
            last_time_floor_share=last_time_floor_share,
            frozen_share=result.frozen_share,
            off_grid_share=off_grid,
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

        def support(scale: str, q: int) -> Support:
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
                past_last_time=bool(restricted) and max(restricted) > max(self.horizons),
            )

        estimates = build_estimates(
            result,
            initial.arm_labels,
            initial.causes,
            self.horizons,
            n=n,
            alpha=self.alpha,
            support=support,
            restricted=restricted,
        )
        return estimates, record


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
    """What every walk of one fit reads: the nuisance fits and the fixed arrays.

    ``product`` is :math:`\\hat\\pi(a \\mid L)\\hat S_c(u^- \\mid a, L)` on the grid, shape
    ``(2, n, M)``, and ``jumps`` holds the initial hazard jumps ``(2, J, n, M)`` in a
    one-element list, so a consuming walk can take them without a second name keeping them
    alive.  ``rng_state`` is the generator state after the nuisance phase, from which every
    walk's off-grid draws start.
    """

    n: int
    T: FloatArray
    A: IntArray
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
    at_risk: Any
    events: Any
    shares: tuple[float, ...]
    rng_state: Mapping[str, Any]
    plan: FoldPlan
    product: FloatArray | None
    jumps: list[FloatArray]


def _off_grid_share(
    problem: Problem,
    result: WalkResult,
    grid: FloatArray,
    rng: np.random.Generator,
    cutoff: float,
) -> float:
    """Item 10 Table 4: the share of random times at which the score equation holds."""
    from ._targeting import evaluate

    draws = np.sort(rng.uniform(0.0, float(grid[-1]), OFF_GRID_TIMES))
    index = np.searchsorted(grid, draws, side="right") - 1
    keep = index >= 0
    if not keep.any():
        return 1.0
    shifted = Problem(
        problem.arm,
        problem.at_risk,
        problem.events,
        problem.weight,
        point_weights(index[keep], grid.size),
    )
    evaluation = evaluate(shifted, result.dlam)
    sigma = evaluation.sigma
    informative = sigma > 0
    if not informative.any():
        return 1.0
    ok = np.abs(evaluation.score[informative]) <= cutoff * sigma[informative]
    return float(np.mean(ok))
