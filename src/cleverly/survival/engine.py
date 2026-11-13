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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final, Literal

import numpy as np

from .._refusal import NO_ALTERNATIVE, RefusalReason
from .._typing import FloatArray, IntArray
from .._warnings import warn as _warn
from ..exceptions import CapabilityError, ConvergenceWarning, DataError
from ._targeting import Norm, Problem, WalkResult, default_cutoff, walk
from .cox import HazardSelection, default_library, select_hazard
from .result import ContinuousSurvivalResult, build_estimates

__all__ = ["ContinuousSurvivalRecord", "ContinuousSurvivalTMLE"]


def _gap(text: str) -> CapabilityError:
    """A refusal of a composition the continuous-time engine has no source or study for."""
    return CapabilityError(
        text, reason=RefusalReason.IMPLEMENTATION_GAP, alternative=NO_ALTERNATIVE, roadmap="X6"
    )


ROADMAP: Final = "roadmap X6"
#: An initial survival at or below this before the last target time is refused.
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
        :math:`\\pi S_c(t_K^-)` at the last target time is below the floor: the
        counterfactual reach of the positivity bound.
    frozen_share : float
        The share of (unit, arm) pairs, over both arms, that the walk held for at least one
        trial step because that step would have taken their survival at the last target time
        below its floor.  A pair is held for one step only.
    off_grid_share : float
        The share of random times in the window, among those where a score is not identically
        zero, at which the score equation holds to the cutoff.
    cross_fitted : bool
        Whether the nuisances were fitted out of fold.
    n_folds : int
        The number of cross-fitting folds, ``1`` for a full-sample fit.
    grid_size : int
        The number of distinct observed times up to the last target.
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
        model = clone(learner).fit(Lt, At)
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
        Seed of the folds and the off-grid diagnostic.
    tilt : {"all", "factual"}
        ``"factual"`` reproduces the ``concrete`` 1.0.5 update, for the mutation control only.
    simultaneous : bool
        Add a sup-t band over every risk and risk difference.
    n_multiplier : int
        Multiplier draws of the band.
    multiplier_kind : str
        The multiplier distribution, as :func:`~cleverly.inference.simultaneous_bands` takes.
    band_scale : {"identity", "logit"}
        The scale of the band of a bounded estimate.
    """

    def __init__(
        self,
        horizons: Sequence[float],
        *,
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
        n_multiplier: int = 1000,
        multiplier_kind: str = "rademacher",
        band_scale: str = "identity",
    ) -> None:
        times = tuple(float(t) for t in horizons)
        if not times or any(not np.isfinite(t) or t <= 0 for t in times):
            raise ValueError("horizons are positive finite target times, at least one")
        if any(b <= a for a, b in pairwise(times)):
            raise ValueError("horizons are strictly increasing")
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
        from ..utils.frames import as_frame, backend_of

        frame = as_frame(data)
        T = frame[time].to_numpy().astype(float)
        D_raw = frame[event].to_numpy().astype(float)
        if not np.all(np.isfinite(T)) or np.any(T <= 0):
            raise DataError("every follow-up time is a positive finite number")
        if not np.all(np.isfinite(D_raw)) or np.any(D_raw < 0) or np.any(D_raw % 1):
            raise DataError("every event code is a non-negative integer, 0 for censored")
        D = D_raw.astype(int)
        L = np.column_stack([frame[name].to_numpy().astype(float) for name in baseline])
        if not np.all(np.isfinite(L)):
            raise DataError("baseline covariates are finite numbers")
        arm_labels, A = self._arms(frame[treatment].to_numpy())
        causes = self._causes(D)
        return self._fit_arrays(T, D, A, L, arm_labels, causes, backend=backend_of(frame))

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
        arm_labels, A = self._arms(data.treatment)
        causes = self._causes(data.event, self.causes if self.causes is not None else data.causes)
        return self._fit_arrays(
            data.time, data.event, A, data.baseline, arm_labels, causes, backend=data.backend
        )

    def _fit_arrays(
        self,
        T: FloatArray,
        D: IntArray,
        A: IntArray,
        L: FloatArray,
        arm_labels: tuple[str, str],
        causes: dict[int, str],
        *,
        backend: str | None = None,
    ) -> ContinuousSurvivalResult:
        n = T.size
        codes = list(causes)
        J = len(codes)
        last = self.horizons[-1]
        for a in (0, 1):
            for code, label in causes.items():
                first_event = T[(a == A) & (code == D)]
                if first_event.size == 0 or first_event.min() > self.horizons[0]:
                    raise _gap(
                        f"Arm {arm_labels[a]!r} has no event of cause {label!r} at or before "
                        f"the first target time {_time_text(self.horizons[0])}, so that risk's "
                        "influence curve is zero and its interval undefined. Choose a later "
                        "first target time."
                    )
        for a in (0, 1):
            followed = float(T[a == A].max())
            if followed < last:
                raise _gap(
                    f"Arm {arm_labels[a]!r} is followed only to {_time_text(followed)}, before "
                    f"the last target time {_time_text(last)}, so its risk there would be an "
                    "extrapolation of the hazard model with no data behind it. Choose target "
                    "times within every arm's follow-up."
                )
        grid = np.unique(T[last >= T])
        target_index = np.searchsorted(grid, self.horizons, side="right") - 1
        X = np.column_stack([A.astype(float), L])
        rng = np.random.default_rng(self.random_state)
        folds = (
            _stratified_folds(D, self.n_folds, rng) if self.cross_fit else np.zeros(n, dtype=int)
        )
        gap = np.diff(np.unique(T))
        shift = 0.5 * float(gap.min()) if gap.size else 0.5
        censor_time = np.where(D > 0, T - shift, T)

        dlam = np.zeros((2, J, n, grid.size))
        dgam = np.zeros((2, n, grid.size))
        p1 = np.zeros(n)
        hazard_records: list[HazardSelection] = []
        censoring_record: HazardSelection | None = None
        propensity_record: PropensitySelection | None = None
        for fold in np.unique(folds):
            predict = folds == fold
            train = ~predict if self.cross_fit else np.ones(n, dtype=bool)
            inner = _stratified_folds(D[train], self.learner_folds, rng)
            for j, code in enumerate(codes):
                selection = select_hazard(
                    self.hazard_library, X[train], T[train], (D[train] == code).astype(float), inner
                )
                for a in (0, 1):
                    Xa = X[predict].copy()
                    Xa[:, 0] = a
                    dlam[a, j][predict] = _jumps(selection.model.cumulative_hazard(Xa, grid))
                if fold == folds.min():
                    hazard_records.append(selection)
            selection = select_hazard(
                self.censoring_library,
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
                self.treatment_learner, L, A, train, predict, inner, self.random_state
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
                "one where few units are at risk; choose earlier target times or a smaller "
                "library."
            )
        survival = np.cumprod(1.0 - dlam.sum(axis=1), axis=-1)
        if np.any(survival[..., -1] <= SURVIVAL_FLOOR):
            raise DataError(
                "The initial survival reaches zero before the last target time for some unit "
                "and arm, so the clever covariate divides by zero. The estimator needs "
                "S(t_K | a, L) bounded away from zero; choose an earlier last target time."
            )
        retention = np.cumprod(1.0 - np.clip(dgam, 0.0, 1.0), axis=-1)
        retention_before = np.concatenate([np.ones((2, n, 1)), retention[..., :-1]], axis=-1)
        pi = np.stack([1.0 - p1, p1])
        floor = 5.0 / (np.sqrt(n) * np.log(n)) if self.g_bounds == "auto" else float(self.g_bounds)
        product = pi[:, :, None] * retention_before
        weight = 1.0 / np.maximum(product, floor)
        factual = A[None, :, None] == np.arange(2)[:, None, None]
        reach = T[None, :, None] >= grid[None, None, :]
        relevant = factual & reach
        bounded_share = float(np.mean(product[relevant] < floor)) if relevant.any() else 0.0
        last_time_floor_share = float(np.mean(product[..., -1] < floor))
        # Each of these is (2, n, M); at n = 4,000 one is ~200 MB, so none outlives its use.
        del survival, retention, retention_before, product, factual, reach, relevant, dgam

        # Indicators stay boolean: arithmetic with the float hazards promotes them on use.
        at_risk = T[:, None] >= grid[None, :]
        events = np.stack([(T[:, None] == grid[None, :]) & (D[:, None] == code) for code in codes])
        problem = Problem(A, at_risk, events, weight, target_index)
        cutoff = default_cutoff(n) if self.tol is None else float(self.tol)
        # Handed over through a list so no name here keeps the initial jumps alive once the
        # walk has replaced them (~400 MB at n = 4,000).
        jumps = [dlam]
        del dlam
        result = walk(
            problem,
            jumps.pop(),
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
        off_grid = _off_grid_share(problem, result, grid, rng, cutoff)
        record = ContinuousSurvivalRecord(
            hazards=tuple(hazard_records),
            censoring=censoring_record,
            propensity=propensity_record,
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
            n_folds=self.n_folds if self.cross_fit else 1,
            grid_size=int(grid.size),
        )
        estimates = build_estimates(
            result, arm_labels, causes, self.horizons, n=n, alpha=self.alpha
        )
        bands = None
        if self.simultaneous:
            from ..inference.multiplier import simultaneous_bands

            banded = {k: v for k, v in estimates.items() if not k.startswith("rr_regimen")}
            bands = simultaneous_bands(
                banded,
                alpha=self.alpha,
                n_replicates=self.n_multiplier,
                kind=self.multiplier_kind,  # type: ignore[arg-type]
                random_state=self.random_state,
                band_scale=self.band_scale,  # type: ignore[arg-type]
            )
        return ContinuousSurvivalResult(
            estimates=estimates,
            simultaneous=bands,
            record=record,
            arms=arm_labels,
            causes=tuple(causes.values()),
            horizons=self.horizons,
            n=n,
            backend=backend,
        )


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
    shifted = Problem(problem.arm, problem.at_risk, problem.events, problem.weight, index[keep])
    evaluation = evaluate(shifted, result.dlam)
    sigma = evaluation.sigma
    informative = sigma > 0
    if not informative.any():
        return 1.0
    ok = np.abs(evaluation.score[informative]) <= cutoff * sigma[informative]
    return float(np.mean(ok))
