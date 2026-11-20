r"""The simulated common-cause replay of a continuous-time survival fit (X101 part (b)).

The shipped latent :math:`U_i` flips the binary baseline treatment in its upper tail, as on a
point fit.  The outcome strength :math:`\gamma` shifts the cause-specific log-hazards of each
unit by :math:`\gamma U_i`: every cause by default, or one named cause.  Event and censoring
times are then redrawn by an inverse-transform coupling on the replay's own Cox fits.

**Replay models.**  One main-terms Cox model with the Breslow baseline per cause, and one for
censoring (event units' times moved back by the engine's half gap, so an event precedes a
censoring at the same time), fitted once per surface on ``[A, standardized L]``.  Each model is
read as a grouped proportional-hazards law: at an atom :math:`u` of the baseline, the
conditional probability of the event is :math:`1 - \exp\{-h(u)\}` with
:math:`h(u) = e^{\eta(x)}\Delta\Lambda_0(u)`, which stays a probability under every shift.

**Latent times.**  Each cause and the censoring process get a latent level.  A cause's latent
time is the first atom at which its shifted cumulative baseline hazard exceeds the level; the
first latent time wins, and a tie between causes is broken in proportion to the shifted jumps.
Each level is drawn from its posterior given the unit's observed outcome, so at zero strength
the replayed data are the observed data array for array.  At an atom shared by several causes
the posterior over which other causes tied there is drawn exactly.

**Treatments.**  The event levels are read at the observed treatment (a flipped unit keeps its
outcome process), and the censoring level is conditioned at the observed treatment and inverted
at the perturbed one, so censoring stays at random given the replayed past when the censoring
model is right.

The design note is ``rows/X101/pr3-design.md`` (rev 2) in the planning repository.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

import numpy as np

from .._refusal import Refusal, RefusalReason

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .simulated_confounding import ConfounderStrengthGrid, SimulatedConfoundingResult

#: The refusal of a result built by hand, without the estimator or the data that produced it.
MISSING_REFUSAL: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "simulated_confounding needs the fitted estimator and the data that produced the result",
)

#: The calibration rule of an outcome row of a continuous-time surface.
REPLAY_COX_METHOD: Final = (
    "coefficient of the standardized covariate in the replay's Cox model of the cause "
    "(log-hazard ratio per standard deviation)"
)

#: The calibration rule of an outcome row whose Cox fit did not converge.
COX_NONCONVERGENCE_METHOD: Final = (
    "the replay's Cox fit did not converge (a monotone partial likelihood); strength not defined"
)

#: The calibration rule of an outcome row whose replay model failed.
COX_FAILED_METHOD: Final = "the replay's Cox fit failed; strength not defined"

#: Uniform rows of the coupling array beyond one per cause: censoring, tied set, cause tie.
_EXTRA_STREAMS: Final = 3


def continuous_survival_simulated_refusal(result: Any) -> Refusal | None:
    """The fit-wide refusal of a continuous-time surface, or ``None``.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fitted result.

    Returns
    -------
    Refusal or None
        :data:`MISSING_REFUSAL` when the result holds no estimator or no data.
    """
    if getattr(result, "estimator", None) is None or getattr(result, "data", None) is None:
        return MISSING_REFUSAL
    return None


# ------------------------------------------------------------------------ the model seam


class HazardModel(Protocol):
    """A proportional-hazards model as the replay reads it.

    ``eta`` is the unit's linear predictor, ``cumulative`` and ``left`` the baseline
    cumulative hazard at and just before a time, ``jump`` its jump at a time (zero off an
    atom), and ``inverse`` the first time at which the baseline exceeds a level (``inf`` past
    the last atom).
    """

    atomic: bool

    def eta(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...

    def cumulative(self, t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...

    def left(self, t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...

    def jump(self, t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...

    def inverse(self, level: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...


@dataclass(frozen=True)
class BreslowHazard:
    """A fitted :class:`~cleverly.survival.CoxHazard` read as a step baseline.

    Parameters
    ----------
    name : str
        What the model fits.
    model : CoxHazard
        The fitted model.
    converged : bool
        Whether the Newton iterations converged.
    """

    name: str
    model: Any
    converged: bool
    atomic: bool = True

    @property
    def atoms(self) -> np.ndarray[Any, Any]:
        """The baseline's jump times, ascending."""
        return np.asarray(self.model.baseline_times_, dtype=float)

    @property
    def jumps(self) -> np.ndarray[Any, Any]:
        """The baseline's jumps at :attr:`atoms`."""
        return np.asarray(self.model.baseline_jumps_, dtype=float)

    @property
    def _cum(self) -> np.ndarray[Any, Any]:
        # The same cumulative sum CoxHazard.cumulative_hazard reads. Not cached: a memo would
        # ride into every saved artifact that holds the model.
        return np.cumsum(self.jumps)

    def eta(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """The linear predictor on the scale the baseline reads."""
        return np.asarray(self.model.linear_predictor(X), dtype=float)

    def cumulative(self, t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """:math:`\\Lambda_0(t)`."""
        index = np.searchsorted(self.atoms, t, side="right") - 1
        return np.where(index >= 0, self._cum[np.maximum(index, 0)], 0.0)

    def left(self, t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """:math:`\\Lambda_0(t^-)`."""
        index = np.searchsorted(self.atoms, t, side="left") - 1
        return np.where(index >= 0, self._cum[np.maximum(index, 0)], 0.0)

    def jump(self, t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """:math:`\\Delta\\Lambda_0(t)`, zero off an atom."""
        atoms = self.atoms
        if atoms.size == 0:
            return np.zeros(np.shape(t))
        index = np.minimum(np.searchsorted(atoms, t, side="left"), atoms.size - 1)
        return np.where(atoms[index] == t, self.jumps[index], 0.0)

    def inverse(self, level: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """The first atom at which :math:`\\Lambda_0` exceeds ``level``, or ``inf``."""
        cum = self._cum
        index = np.searchsorted(cum, level, side="right")
        out = np.full(np.shape(level), np.inf)
        inside = index < cum.size
        out[inside] = self.atoms[index[inside]]
        return out

    def describe(self) -> str:
        """One line naming the model and whether it converged."""
        return (
            f"{self.name}: Cox, main terms in the treatment and the standardized baseline "
            "covariates, Breslow baseline, unpenalized, read as a grouped proportional-hazards "
            "law" + ("" if self.converged else "; did not converge (monotone partial likelihood)")
        )


# --------------------------------------------------------------------------- replay models


@dataclass(frozen=True)
class ReplayModels:
    """The replay's fitted hazards and the observed rows they were fitted on.

    Parameters
    ----------
    causes : tuple of HazardModel
        One model per cause, in cause-code order.
    censoring : HazardModel or None
        The censoring model, or ``None`` when no unit was censored.
    design : ndarray
        ``(n, 1 + p)`` observed design ``[A, standardized L]``.
    baseline : ndarray
        ``(n, p)`` standardized baseline block.
    failure : str or None
        Why the models cannot be read (a non-finite predictor, or a zero or non-finite jump
        at an observed unit's own time), or ``None``.
    """

    causes: tuple[Any, ...]
    censoring: Any
    design: np.ndarray[Any, Any]
    baseline: np.ndarray[Any, Any]
    failure: str | None = None

    def describe(self) -> tuple[str, ...]:
        """One line per fitted model, and the failure when there is one."""
        lines = [
            model.describe()
            for model in (*self.causes, self.censoring)
            if model is not None and hasattr(model, "describe")
        ]
        if self.failure is not None:
            lines.append(f"the replay models cannot be read: {self.failure}")
        return tuple(lines)


def standardized_baseline(baseline: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The baseline block standardized at the unit level (no weights: the container has none).

    Parameters
    ----------
    baseline : ndarray
        ``(n, p)`` baseline covariates.

    Returns
    -------
    ndarray
        Each column centred and scaled to unit (population) standard deviation; a constant
        column is centred only.
    """
    from sklearn.preprocessing import StandardScaler

    return np.asarray(StandardScaler().fit_transform(np.asarray(baseline, dtype=float)))


def censoring_shift(time: np.ndarray[Any, Any]) -> float:
    """Half the smallest gap between distinct times, as the engine moves event units back."""
    gap = np.diff(np.unique(time))
    return 0.5 * float(gap.min()) if gap.size else 0.5


def _fit_cox(name: str, X: Any, time: Any, event: Any) -> BreslowHazard:
    from ..exceptions import ConvergenceWarning
    from ..survival.cox import CoxHazard

    with warnings.catch_warnings():
        # The replay records non-convergence in replay_models and in the calibration.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = CoxHazard("main").fit(X, time, event)
    return BreslowHazard(name, model, bool(model.converged_))


def fit_replay_models(
    time: np.ndarray[Any, Any],
    code: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    baseline: np.ndarray[Any, Any],
    codes: Sequence[int],
    labels: Sequence[str],
) -> ReplayModels:
    """Fit the replay's Cox models once, on the observed rows.

    Parameters
    ----------
    time : ndarray
        ``(n,)`` observed times.
    code : ndarray
        ``(n,)`` observed event codes, 0 for censored.
    treatment : ndarray
        ``(n,)`` treatment coded 0/1.
    baseline : ndarray
        ``(n, p)`` baseline covariates.
    codes : sequence of int
        The cause codes, ascending.
    labels : sequence of str
        The label of each cause code.

    Returns
    -------
    ReplayModels
        The fits, with :func:`check_models` applied.
    """
    standardized = standardized_baseline(baseline)
    X = np.column_stack([np.asarray(treatment, dtype=float), standardized])
    causes = tuple(
        _fit_cox(f"hazard of cause {label!r}", X, time, (code == c).astype(float))
        for c, label in zip(codes, labels, strict=True)
    )
    censoring = None
    if np.any(code == 0):
        moved = np.where(code > 0, time - censoring_shift(time), time)
        censoring = _fit_cox("censoring hazard", X, moved, (code == 0).astype(float))
    models = ReplayModels(causes, censoring, X, standardized)
    return replace(models, failure=check_models(models, time, code, codes))


def check_models(
    models: ReplayModels,
    time: np.ndarray[Any, Any],
    code: np.ndarray[Any, Any],
    codes: Sequence[int],
) -> str | None:
    """Why the models cannot be read on these rows, or ``None``.

    Every linear predictor must be finite, and every observed unit's jump at its own time
    (its cause's for an event, the censoring model's for a censored unit) positive and
    finite; a Cox fit on a monotone-likelihood boundary can overflow to break either.

    Parameters
    ----------
    models : ReplayModels
        The fits.
    time : ndarray
        ``(n,)`` observed times.
    code : ndarray
        ``(n,)`` observed codes.
    codes : sequence of int
        The cause codes.

    Returns
    -------
    str or None
        The reason, or ``None``.
    """
    X = models.design
    with np.errstate(over="ignore", invalid="ignore"):
        for position, c in enumerate(codes):
            model = models.causes[position]
            eta = model.eta(X)
            if not np.all(np.isfinite(eta)):
                return f"the linear predictor of cause {c} is not finite for some unit"
            rows = code == c
            h = np.exp(eta[rows]) * model.jump(time[rows]) if model.atomic else None
            if h is not None and not np.all(np.isfinite(h) & (h > 0.0)):
                return f"the hazard of cause {c} has a zero or non-finite jump at an observed time"
        if models.censoring is not None:
            eta = models.censoring.eta(X)
            if not np.all(np.isfinite(eta)):
                return "the linear predictor of censoring is not finite for some unit"
            rows = code == 0
            if models.censoring.atomic:
                h = np.exp(eta[rows]) * models.censoring.jump(time[rows])
                if not np.all(np.isfinite(h) & (h > 0.0)):
                    return "the censoring hazard has a zero or non-finite jump at a censoring time"
    return None


# ------------------------------------------------------------------- the observed levels


@dataclass(frozen=True)
class Levels:
    """Every latent level of every unit, drawn once per surface from the coupling uniforms.

    Parameters
    ----------
    cause : ndarray
        ``(J, n)`` cause levels on each cause's baseline cumulative-hazard scale, at the
        observed treatment.
    censoring : ndarray or None
        ``(n,)`` censoring levels on the censoring baseline scale at the observed treatment.
    tie : ndarray
        ``(n,)`` the cause-tie uniform, conditioned on the observed cause at a shared atom.
    eta : ndarray
        ``(J, n)`` each cause's linear predictor at the observed treatment.
    eta_censoring : ndarray or None
        ``(n,)`` the censoring model's linear predictor at the observed treatment.
    """

    cause: np.ndarray[Any, Any]
    censoring: np.ndarray[Any, Any] | None
    tie: np.ndarray[Any, Any]
    eta: np.ndarray[Any, Any]
    eta_censoring: np.ndarray[Any, Any] | None


def _on_jump(
    model: Any,
    t: np.ndarray[Any, Any],
    eta: np.ndarray[Any, Any],
    raw: np.ndarray[Any, Any],
    *,
    end: float,
) -> np.ndarray[Any, Any]:
    r"""Levels on the jump at ``t``: :math:`V` uniform on :math:`[S(t), S(t^-))`.

    Computed additively in hazard units, :math:`\Lambda_0(t^-) + \delta / e^\eta`, so the sum
    never falls below its first term; one ``nextafter`` pulls a level that rounding put on
    :math:`\Lambda_0(t)` back inside, at an atom only.  On a continuous baseline the level is
    :math:`\Lambda_0(t)` exactly.  A grouped baseline with no atom at ``t`` cannot have
    produced the time there.  At ``end``, the last observed time, the unit was censored by the
    end of follow-up, so its latent time is past ``t`` and the level is drawn past
    :math:`\Lambda_0(t)`, as :func:`_after` draws it.  Before ``end`` no level reproduces the
    data, and the draw raises.
    """
    if not model.atomic:
        return np.asarray(model.cumulative(t), dtype=float)
    left = model.left(t)
    jump = model.jump(t)
    scale = np.exp(eta)
    h = scale * jump
    delta = -np.log1p(raw * np.expm1(-h))
    level = left + delta / scale
    top = model.cumulative(t)
    over = (jump > 0.0) & (level >= top)
    level[over] = np.nextafter(top[over], -np.inf)
    off = jump <= 0.0
    if np.any(off & (t != end)):
        raise ValueError(
            "a grouped replay model has no atom at an observed time before the end of follow-up, "
            "so no latent level reproduces that unit"
        )
    if off.any():
        level[off] = _after(model, t[off], eta[off], raw[off])
    return np.asarray(level, dtype=float)


def _after(
    model: Any,
    t: np.ndarray[Any, Any],
    eta: np.ndarray[Any, Any],
    raw: np.ndarray[Any, Any],
    *,
    left: bool = False,
) -> np.ndarray[Any, Any]:
    r"""Levels past ``t``, or from ``t^-`` with ``left=True``: :math:`V` uniform on :math:`(0, S)`.

    The level is :math:`\Lambda_0 + (-\log(1 - R)) / e^\eta`, at least :math:`\Lambda_0`.
    """
    start = model.left(t) if left else model.cumulative(t)
    return np.asarray(start - np.log1p(-raw) / np.exp(eta), dtype=float)


def _tie_choice(
    weights: np.ndarray[Any, Any], position: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """The cause index the tie rule picks: the first with cumulative weight above the position.

    ``weights`` is ``(J, m)``, zero outside each unit's tied set; ``position`` is ``(m,)`` in
    ``[0, 1)``.
    """
    cumulative = np.cumsum(weights, axis=0)
    target = position * cumulative[-1]
    index = (cumulative <= target[None, :]).sum(axis=0)
    return np.minimum(index, weights.shape[0] - 1)


def _tied_sets(
    h: np.ndarray[Any, Any], k: int, position: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    r"""Draw which other causes also jumped at an event unit's own atom.

    Given that cause ``k`` jumped at the unit's time ``u`` and was the cause recorded, the
    set :math:`S` of other causes that jumped at ``u`` has posterior probability
    proportional to :math:`\prod_{i \in S} p_i \prod_{i \notin S} (1 - p_i) \,
    h_k / (h_k + \sum_{i \in S} h_i)`, with :math:`p_i = 1 - e^{-h_i}`: the prior that
    ``i`` jumps times the tie rule's chance of picking ``k``.  A cause with no atom at
    ``u`` has :math:`h_i = 0` and is never in the set.

    Parameters
    ----------
    h : ndarray
        ``(J, m)`` each cause's unit-level jump :math:`e^{\eta_i} \Delta\Lambda_{0i}(u)`.
    k : int
        The observed cause.
    position : ndarray
        ``(m,)`` the set-draw uniforms.

    Returns
    -------
    ndarray
        ``(J, m)`` boolean, true for ``k`` and for each cause drawn into the set.
    """
    J, m = h.shape
    others = [i for i in range(J) if i != k]
    p = {i: -np.expm1(-h[i]) for i in others}
    masks = range(1 << len(others))
    weights = np.zeros((len(masks), m))
    for mask in masks:
        members = [others[b] for b in range(len(others)) if mask >> b & 1]
        weight = np.ones(m)
        for i in others:
            weight = weight * (p[i] if i in members else 1.0 - p[i])
        total = h[k] + sum((h[i] for i in members), np.zeros(m))
        # Off an atom every jump is zero and the set is {k}: its share is one.
        share = np.divide(h[k], total, out=np.ones(m), where=total > 0.0)
        weights[mask] = weight * share
    cumulative = np.cumsum(weights, axis=0)
    chosen = np.minimum(
        (cumulative <= (position * cumulative[-1])[None, :]).sum(axis=0), len(masks) - 1
    )
    tied = np.zeros((J, m), dtype=bool)
    for b, i in enumerate(others):
        tied[i] = (chosen >> b & 1).astype(bool)
    tied[k] = True
    return tied


def observed_levels(
    models: ReplayModels,
    time: np.ndarray[Any, Any],
    cause_index: np.ndarray[Any, Any],
    uniforms: np.ndarray[Any, Any],
) -> Levels:
    """Draw every latent level from its posterior given the observed outcome.

    Parameters
    ----------
    models : ReplayModels
        The replay's models.
    time : ndarray
        ``(n,)`` observed times.
    cause_index : ndarray
        ``(n,)`` the observed cause's position in the cause list, ``-1`` for censored.
    uniforms : ndarray
        ``(J + 3, n)`` coupling uniforms: one row per cause, then censoring, the tied set
        and the cause tie.

    Returns
    -------
    Levels
        The levels.
    """
    J = len(models.causes)
    n = time.size
    X = models.design
    r_cause = uniforms[:J]
    r_censoring, r_set, r_tie = uniforms[J], uniforms[J + 1], uniforms[J + 2].copy()
    eta = np.stack([model.eta(X) for model in models.causes]) if J else np.zeros((0, n))
    event = cause_index >= 0
    end = float(np.max(time)) if n else 0.0
    levels = np.zeros((J, n))
    with np.errstate(over="ignore", invalid="ignore"):
        # Unit-level jumps of every cause at each unit's own time.
        h = (
            np.stack([np.exp(eta[k]) * models.causes[k].jump(time) for k in range(J)])
            if J
            else np.zeros((0, n))
        )
    # Censored units: every cause after C.
    censored = ~event
    for k in range(J):
        levels[k, censored] = _after(
            models.causes[k], time[censored], eta[k, censored], r_cause[k, censored]
        )
    # Event units: the tied set, then the levels, then the tie uniform.
    tied = np.zeros((J, n), dtype=bool)
    for k in range(J):
        rows = np.flatnonzero(cause_index == k)
        if rows.size == 0:
            continue
        tied[:, rows] = _tied_sets(h[:, rows], k, r_set[rows])
        for i in range(J):
            model = models.causes[i]
            t = time[rows]
            on = tied[i, rows]
            out = np.empty(rows.size)
            if on.any():
                out[on] = _on_jump(model, t[on], eta[i, rows][on], r_cause[i, rows][on], end=end)
            if (~on).any():
                out[~on] = _after(model, t[~on], eta[i, rows][~on], r_cause[i, rows][~on])
            levels[i, rows] = out
        # The tie uniform, conditioned on the sub-interval that assigns k among the tied set.
        set_weights = np.where(tied[:, rows], h[:, rows], 0.0)
        many = tied[:, rows].sum(axis=0) > 1
        if many.any():
            sub = set_weights[:, many]
            total = sub.sum(axis=0)
            low = sub[:k].sum(axis=0) / total
            high = low + sub[k] / total
            position = low + (high - low) * r_tie[rows][many]
            # Rounding can put the position on a boundary; move it inside until the rule
            # read with the same arithmetic picks k.
            for _ in range(8):
                wrong = _tie_choice(sub, position) != k
                if not wrong.any():
                    break
                pick = _tie_choice(sub, position)
                position[wrong] = np.where(
                    pick[wrong] > k,
                    np.nextafter(position[wrong], -np.inf),
                    np.nextafter(position[wrong], np.inf),
                )
            values = r_tie[rows]
            values[many] = position
            r_tie[rows] = values
    censoring = eta_c = None
    if models.censoring is not None:
        model = models.censoring
        eta_c = model.eta(X)
        censoring = np.empty(n)
        censoring[event] = _after(model, time[event], eta_c[event], r_censoring[event], left=True)
        censoring[~event] = _on_jump(
            model, time[~event], eta_c[~event], r_censoring[~event], end=end
        )
    return Levels(levels, censoring, r_tie, eta, eta_c)


# -------------------------------------------------------------------------- one cell


@dataclass(frozen=True)
class Replay:
    """One cell's replayed outcome and its shares.

    Parameters
    ----------
    time : ndarray
        ``(n,)`` replayed times.
    cause_index : ndarray
        ``(n,)`` replayed cause positions, ``-1`` for censored.
    censoring_redrawn_share : float or None
        Units whose censoring time was drawn at a changed treatment, over all units.
    extended_share : float
        Units whose replayed time exceeds the observed one, over all units.
    new_event_share : float or None
        Replayed events whose time or cause differs from the observed event, over replayed
        events.
    """

    time: np.ndarray[Any, Any]
    cause_index: np.ndarray[Any, Any]
    censoring_redrawn_share: float | None
    extended_share: float
    new_event_share: float | None


def latent_draws(
    models: ReplayModels,
    levels: Levels,
    treatment: np.ndarray[Any, Any],
    perturbed: np.ndarray[Any, Any],
    latent: np.ndarray[Any, Any],
    shifts: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    r"""Invert the stored levels into one cell's latent cause and censoring times.

    Cause ``k``'s level is scaled by :math:`e^{-\gamma_k U}` and inverted on its baseline, so
    its hazard is multiplied by :math:`e^{\gamma_k U}` at the observed treatment.  The
    censoring level is scaled by :math:`e^{\eta_c(A) - \eta_c(A')}` where the cell's
    treatment differs from the observed one, and is left untouched elsewhere.

    Parameters
    ----------
    models : ReplayModels
        The replay's models.
    levels : Levels
        The observed levels.
    treatment : ndarray
        ``(n,)`` observed treatment, coded 0/1.
    perturbed : ndarray
        ``(n,)`` the cell's treatment, coded 0/1.
    latent : ndarray
        ``(n,)`` the latent.
    shifts : ndarray
        ``(J,)`` each cause's log-hazard shift per unit of latent.

    Returns
    -------
    latent_times : ndarray
        ``(J, n)`` each cause's latent event time, ``inf`` past its last atom.
    censored_at : ndarray
        ``(n,)`` the censoring time, ``inf`` without a censoring model.
    """
    J = len(models.causes)
    n = latent.size
    latent_times = np.full((J, n), np.inf)
    for k in range(J):
        factor = np.exp(-shifts[k] * latent)
        latent_times[k] = models.causes[k].inverse(levels.cause[k] * factor)
    changed = perturbed != treatment
    if models.censoring is not None and levels.censoring is not None:
        level = levels.censoring.copy()
        if changed.any() and levels.eta_censoring is not None:
            moved = models.censoring.eta(_design_at(models, perturbed[changed], changed))
            level[changed] = level[changed] * np.exp(levels.eta_censoring[changed] - moved)
        censored_at = np.asarray(models.censoring.inverse(level), dtype=float)
    else:
        censored_at = np.full(n, np.inf)
    return latent_times, censored_at


def replay_cell(
    models: ReplayModels,
    levels: Levels,
    time: np.ndarray[Any, Any],
    cause_index: np.ndarray[Any, Any],
    treatment: np.ndarray[Any, Any],
    perturbed: np.ndarray[Any, Any],
    latent: np.ndarray[Any, Any],
    shifts: np.ndarray[Any, Any],
    *,
    tau: float | None = None,
) -> Replay:
    r"""Replay one cell from the stored levels.

    Parameters
    ----------
    models : ReplayModels
        The replay's models.
    levels : Levels
        The observed levels.
    time : ndarray
        ``(n,)`` observed times.
    cause_index : ndarray
        ``(n,)`` observed cause positions, ``-1`` for censored.
    treatment : ndarray
        ``(n,)`` observed treatment, coded 0/1.
    perturbed : ndarray
        ``(n,)`` the cell's treatment, coded 0/1.
    latent : ndarray
        ``(n,)`` the latent.
    shifts : ndarray
        ``(J,)`` the log-hazard shift per unit of latent of each cause: the outcome strength
        for a shifted cause, zero otherwise.
    tau : float or None, default None
        The administrative censoring time; the largest observed time when None.

    Returns
    -------
    Replay
        The replayed outcome and the shares.
    """
    J = len(models.causes)
    n = time.size
    end_of_study = float(np.max(time)) if tau is None else float(tau)
    latent_times, censored_at = latent_draws(models, levels, treatment, perturbed, latent, shifts)
    first = latent_times.min(axis=0) if J else np.full(n, np.inf)
    # The cause: the first latent time, or the tie rule among those on the first atom.
    finite = np.isfinite(first)
    on_first = (latent_times == first[None, :]) & finite[None, :]
    winner = np.argmax(on_first, axis=0)
    many = on_first.sum(axis=0) > 1
    if many.any():
        with np.errstate(over="ignore"):
            weights = np.stack(
                [
                    np.where(
                        on_first[k, many],
                        np.exp(levels.eta[k, many] + shifts[k] * latent[many])
                        * models.causes[k].jump(first[many]),
                        0.0,
                    )
                    for k in range(J)
                ]
            )
        winner[many] = _tie_choice(weights, levels.tie[many])
    changed = perturbed != treatment
    end = np.minimum(censored_at, end_of_study)
    event = first <= end
    replayed_time = np.where(event, first, end)
    replayed_cause = np.where(event, winner, -1)
    extended = float(np.mean(replayed_time > time))
    observed_event = cause_index >= 0
    new = event & (~observed_event | (replayed_time != time) | (replayed_cause != cause_index))
    return Replay(
        time=replayed_time,
        cause_index=replayed_cause,
        censoring_redrawn_share=(float(np.mean(changed)) if models.censoring is not None else None),
        extended_share=extended,
        new_event_share=float(new.sum() / event.sum()) if event.any() else None,
    )


def _design_at(models: ReplayModels, treatment: Any, rows: Any) -> np.ndarray[Any, Any]:
    """The design of ``rows`` with their treatment replaced."""
    design = models.design[rows].copy()
    design[:, 0] = treatment
    return design


# ---------------------------------------------------------------------- the request


@dataclass(frozen=True)
class ContinuousRequest:
    """A validated continuous-time request.

    Parameters
    ----------
    points : tuple of float
        The treatment flip shares.
    calibration_names : tuple of str
        The benchmark covariates.
    movement_scale : {"estimate_difference", "log_ratio"}
        The displacement scale of the estimand.
    outcome_cause : str or None
        The named cause, or ``None`` to shift every cause.
    """

    points: tuple[float, ...]
    calibration_names: tuple[str, ...]
    movement_scale: Literal["estimate_difference", "log_ratio"]
    outcome_cause: str | None


def validate_continuous_request(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: Any,
    *,
    arm_pair: Any = None,
    latent_level: Any = None,
    latent_rho: Any = None,
    observation_reads: Any = None,
    bounded_outcome_law: Any = None,
    outcome_cause: Any = None,
) -> ContinuousRequest:
    """Check a continuous-time request before any fit, draw, or refit.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fitted result.
    estimand : str
        The reported parameter.
    grid : ConfounderStrengthGrid
        The declared strengths.
    benchmark_covariates : str or sequence of str
        The benchmark covariates.
    arm_pair : Any
        Must be ``None``: the treatment is binary.
    latent_level : Any
        ``None`` or ``"row"``: the container has no cluster.
    latent_rho : Any
        Must be ``None``.
    observation_reads : Any
        Must be ``None``: the container records no missing observation.
    bounded_outcome_law : Any
        Must be ``None``: the outcome is a time to event.
    outcome_cause : Any
        A cause label of a fit with two or more causes, or ``None``.

    Returns
    -------
    ContinuousRequest
        The validated request.

    Raises
    ------
    CapabilityError
        If the fit-wide rule refuses the fit.
    TypeError
        If ``benchmark_covariates`` holds a name that is not a string.
    ValueError
        If the estimand, the grid, or an argument does not apply to this fit.
    """
    from ..exceptions import CapabilityError

    refusal = continuous_survival_simulated_refusal(result)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    if estimand not in result.estimates:
        raise ValueError(
            f"estimand {estimand!r} was not reported by this fit; it reports "
            f"{list(result.estimates)!r}"
        )
    if arm_pair is not None:
        raise ValueError(
            "arm_pair names the pairwise swap of a multi-arm point treatment; a continuous-time "
            "survival fit has a binary treatment"
        )
    if latent_level not in (None, "row") or latent_rho is not None:
        raise ValueError(
            "latent_level and latent_rho declare the latent level of a clustered fit; a "
            "continuous-time survival fit declares no cluster, so the latent is one value per unit"
        )
    if observation_reads is not None:
        raise ValueError(
            "observation_reads declares the observation mechanisms of a fit with missing "
            "observations; a continuous-time survival fit observes every treatment and covariate, "
            "and its censoring is replayed"
        )
    if bounded_outcome_law is not None:
        raise ValueError(
            "bounded_outcome_law= reads the q_bounds of a Gaussian fit; a continuous-time "
            "survival fit's outcome is a time to event, shifted on the log-hazard scale"
        )
    if any(value != 0.0 for value in grid.intermediate):
        raise ValueError(
            "ConfounderStrengthGrid(intermediate=...) moves the intermediate of a controlled "
            "direct effect, and this fit declares no intermediate"
        )
    if grid.components:
        raise ValueError(
            "ConfounderStrengthGrid(treatment={...}) keys strengths by treatment component; a "
            "continuous-time survival fit has one binary treatment"
        )
    points = tuple(float(value) for value in grid.treatment)  # type: ignore[arg-type]
    if any(value < 0.0 or value > 0.5 for value in points):
        raise ValueError("binary treatment strengths must be between 0 and 0.5")
    # Outcome strengths are finite: ConfounderStrengthGrid refuses any other value.
    causes = tuple(result.causes)
    if outcome_cause is not None:
        if len(causes) < 2:
            raise ValueError(
                "outcome_cause picks one of several competing causes, and this fit declares "
                "one cause, whose hazard the default law already shifts; leave outcome_cause unset"
            )
        if outcome_cause not in causes:
            raise ValueError(
                f"outcome_cause {outcome_cause!r} is not a cause of this fit; its causes are "
                f"{list(causes)!r}"
            )
    names = tuple(
        [benchmark_covariates] if isinstance(benchmark_covariates, str) else benchmark_covariates
    )
    if len(set(names)) != len(names):
        raise ValueError("benchmark_covariates contains duplicates")
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"benchmark_covariates holds {name!r}, which is not a column name")
        if name not in result.data.baseline_names:
            raise ValueError(
                f"benchmark covariate {name!r} is unavailable; a benchmark names a baseline "
                f"covariate of this fit, one of {list(result.data.baseline_names)!r}"
            )
    scale = getattr(result[estimand], "scale", None)
    return ContinuousRequest(
        points=points,
        calibration_names=names,
        movement_scale="log_ratio" if scale == "ratio" else "estimate_difference",
        outcome_cause=None if outcome_cause is None else str(outcome_cause),
    )


# ------------------------------------------------------------------------- calibration


def _calibrate(
    result: Any,
    request: ContinuousRequest,
    models: ReplayModels | None,
    coded: np.ndarray[Any, Any],
    labels: Sequence[str],
) -> tuple[Any, ...]:
    """The treatment row and the outcome rows of each benchmark covariate."""
    from .simulated_confounding import ObservedConfounderCalibration, _binary_calibration

    if not request.calibration_names:
        return ()
    data = result.data
    standardized = standardized_baseline(data.baseline)
    weights = np.ones(data.n)
    rows = []
    for name in request.calibration_names:
        index = list(data.baseline_names).index(name)
        strength, method = _binary_calibration(standardized, coded.astype(float), [index], weights)
        rows.append(
            ObservedConfounderCalibration(
                covariate=name,
                role="treatment",
                family="binomial",
                strength=strength,
                method=method,
            )
        )
        for position, label in enumerate(labels):
            if request.outcome_cause is not None and label != request.outcome_cause:
                continue
            if models is None or models.failure is not None:
                value, text = float("nan"), COX_FAILED_METHOD
            else:
                model = models.causes[position]
                if not model.converged:
                    value, text = float("nan"), COX_NONCONVERGENCE_METHOD
                else:
                    value, text = float(model.model.coef_[1 + index]), REPLAY_COX_METHOD
            rows.append(
                ObservedConfounderCalibration(
                    covariate=name,
                    role="outcome",
                    family="hazard",
                    strength=value,
                    method=text,
                    component=label if len(labels) > 1 else None,
                )
            )
    return tuple(rows)


# --------------------------------------------------------------------------- the surface


@dataclass(frozen=True)
class _State:
    """What every cell of one surface shares."""

    models: ReplayModels | None
    levels: Levels | None
    error: Exception | None


def _observed_arrays(result: Any) -> tuple[Any, ...]:
    """The coded treatment and causes of the stored data, with the arm levels."""
    data = result.data
    estimator = result.estimator
    _, coded = estimator._arms(data.treatment)
    causes = estimator._causes(
        data.event, estimator.causes if estimator.causes is not None else data.causes
    )
    codes = list(causes)
    labels = [causes[c] for c in codes]
    treatment = np.asarray(data.treatment)
    level0 = treatment[coded == 0][0]
    level1 = treatment[coded == 1][0]
    event = np.asarray(data.event, dtype=int)
    position = {c: i for i, c in enumerate(codes)}
    cause_index = np.array([position.get(int(c), -1) if c > 0 else -1 for c in event])
    return coded, codes, labels, (level0, level1), cause_index


def _state(
    result: Any,
    coded: Any,
    codes: Any,
    labels: Any,
    cause_index: Any,
    seed: int,
) -> _State:
    data = result.data
    time = np.asarray(data.time, dtype=float)
    event = np.asarray(data.event, dtype=int)
    try:
        models = fit_replay_models(time, event, coded, data.baseline, codes, labels)
    except Exception as error:  # every nonzero cell then records it as its failure
        return _State(None, None, error)
    if models.failure is not None:
        return _State(
            models, None, RuntimeError(f"the replay models cannot be read: {models.failure}")
        )
    return _drawn_state(models, time, cause_index, coded, len(codes), seed, preflight=True)


def _drawn_state(
    models: ReplayModels,
    time: np.ndarray[Any, Any],
    cause_index: np.ndarray[Any, Any],
    coded: np.ndarray[Any, Any],
    n_causes: int,
    seed: int,
    *,
    preflight: bool,
) -> _State:
    """Draw the levels and run the preflight; an error in either fails every moved cell.

    Supplied continuous models skip the preflight: their inversion returns the observed time
    only to rounding, which the known-hazard checks bound instead.
    """
    n = time.size
    try:
        uniforms = np.random.default_rng(seed).uniform(size=(n_causes + _EXTRA_STREAMS, n))
        levels = observed_levels(models, time, cause_index, uniforms)
        if not preflight:
            return _State(models, levels, None)
        # Preflight: the general path at zero strength must return the observed data.
        zero = replay_cell(
            models, levels, time, cause_index, coded, coded, np.zeros(n), np.zeros(n_causes)
        )
    except Exception as error:  # every nonzero cell then records it as its failure
        return _State(models, None, error)
    if not (np.array_equal(zero.time, time) and np.array_equal(zero.cause_index, cause_index)):
        return _State(
            models,
            levels,
            RuntimeError(
                "the replay at zero strength does not reproduce the observed data; this is a "
                "package defect, report it"
            ),
        )
    return _State(models, levels, None)


def _outcome_law(request: ContinuousRequest, labels: Sequence[str]) -> str:
    if request.outcome_cause is not None:
        which = (
            f"the log-hazard of cause {request.outcome_cause!r} is shifted by the outcome "
            "strength times the unit's latent value; every other cause-specific hazard is "
            "unchanged"
        )
    elif len(labels) > 1:
        which = (
            "every cause-specific log-hazard is shifted by the outcome strength times the "
            "unit's latent value, so the cause shares at each time are unchanged"
        )
    else:
        which = "the log-hazard is shifted by the outcome strength times the unit's latent value"
    return (
        f"Each unit's {which}. Event and censoring times are redrawn from the replay's Cox "
        "fits, read as grouped proportional-hazards laws, through latent levels drawn given "
        "the observed outcome, so a zero strength keeps the observed data; a tie between causes "
        "at one time goes to each in proportion to its shifted jump. The event hazards are read "
        "at the observed treatment, censoring is redrawn at the perturbed treatment where it "
        "changed, and a unit followed past the data's last time is censored there."
    )


def continuous_surface(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    request: ContinuousRequest,
    random_state: int | None,
    *,
    _replay_models: ReplayModels | None = None,
) -> SimulatedConfoundingResult:
    """Evaluate every declared cell of a continuous-time surface.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fitted result.
    estimand : str
        The reported parameter.
    grid : ConfounderStrengthGrid
        The declared strengths.
    request : ContinuousRequest
        The validated request.
    random_state : int or None
        The root seed.
    _replay_models : ReplayModels or None
        Private: models used instead of the fitted ones, for the known-hazard checks.

    Returns
    -------
    SimulatedConfoundingResult
        One cell per treatment strength and outcome strength.
    """
    from ..data.weighting import WeightSpec, describe_weights
    from ..utils.random import resolve_assessment_seed
    from ..validation.simulation import ReplicationFailure
    from ._simulated_confounding_survival import coupling_seed
    from .simulated_confounding import (
        SimulatedConfoundingCell,
        SimulatedConfoundingResult,
        _draw_latent,
        _flip_mask,
        _latent_child_seed,
        _weighted_correlation,
    )

    data = result.data
    estimator = result.estimator
    root_seed = resolve_assessment_seed(result, random_state)
    coded, codes, labels, (level0, level1), cause_index = _observed_arrays(result)
    seed = coupling_seed(root_seed)
    if _replay_models is None:
        state = _state(result, coded, codes, labels, cause_index, seed)
    else:
        state = _known_state(result, _replay_models, coded, cause_index, len(codes), seed)
    calibrations = _calibrate(result, request, state.models, coded, labels)
    latent_seed = _latent_child_seed(root_seed)
    latent = _draw_latent(data.n, latent_seed, None, "row", None)
    own = estimator.random_state
    frozen = own is not None and int(own) == root_seed and estimator._fold_plan is not None
    original_parameter = result[estimand]
    original = float(original_parameter.psi)
    original_inference = original_parameter.inference_value
    weights = np.ones(data.n)
    time = np.asarray(data.time, dtype=float)
    treatment = np.asarray(data.treatment)
    shifted = [request.outcome_cause is None or label == request.outcome_cause for label in labels]
    cells: list[Any] = []
    for cell_index, (point, outcome_strength) in enumerate(product(request.points, grid.outcome)):
        if outcome_strength == 0.0 and point == 0.0:
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=point,
                    outcome_strength=outcome_strength,
                    estimate=original,
                    displacement=0.0,
                    induced_treatment_association=_weighted_correlation(
                        latent, coded.astype(float), weights
                    ),
                    target_population_fraction=1.0,
                )
            )
            continue
        association: Any = None
        try:
            mask = _flip_mask(latent, point)
            perturbed = np.where(mask, 1 - coded, coded)
            association = _weighted_correlation(latent, perturbed.astype(float), weights)
            if state.error is not None:
                raise state.error
            assert state.models is not None and state.levels is not None
            shifts = np.where(shifted, float(outcome_strength), 0.0)
            replay = replay_cell(
                state.models, state.levels, time, cause_index, coded, perturbed, latent, shifts
            )
            labelled = treatment.copy()
            labelled[mask & (coded == 0)] = level1
            labelled[mask & (coded == 1)] = level0
            codes_array = np.asarray(codes, dtype=int)
            event = np.where(
                replay.cause_index >= 0, codes_array[np.maximum(replay.cause_index, 0)], 0
            )
            replacement = replace(data, time=replay.time, event=event, treatment=labelled)
            refitted = estimator._for_estimate_refits().refit(
                replacement,
                random_state=root_seed,
                _frozen_plan=estimator._fold_plan if frozen else None,
            )
            refitted_parameter = refitted[estimand]
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=point,
                    outcome_strength=outcome_strength,
                    estimate=float(refitted_parameter.psi),
                    displacement=refitted_parameter.inference_value - original_inference,
                    induced_treatment_association=association,
                    target_population_fraction=1.0,
                    censoring_redrawn_share=replay.censoring_redrawn_share,
                    extended_share=replay.extended_share,
                    new_event_share=replay.new_event_share,
                )
            )
        except Exception as error:
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=point,
                    outcome_strength=outcome_strength,
                    estimate=None,
                    displacement=None,
                    induced_treatment_association=association,
                    target_population_fraction=1.0,
                    failure=ReplicationFailure(
                        replicate=cell_index,
                        seed=root_seed,
                        error_type=type(error).__name__,
                        message=str(error),
                    ),
                )
            )
    replay_lines: tuple[str, ...] = ()
    if state.models is not None:
        replay_lines = state.models.describe()
    elif state.error is not None:
        replay_lines = (f"the replay models could not be fitted: {state.error}",)
    return SimulatedConfoundingResult(
        estimand=estimand,
        original_estimate=original,
        movement_scale=request.movement_scale,
        scale=str(original_parameter.scale),
        grid=grid,
        cells=tuple(cells),
        calibrations=calibrations,
        root_seed=root_seed,
        latent_seed=latent_seed,
        refit_seed=root_seed,
        n_repeats=1,
        repeat_aggregation="coordinatewise_median",
        treatment_family="binary",
        outcome_family="competing_risks" if len(labels) > 1 else "survival",
        treatment_law="Treatment is flipped in the declared upper latent-normal tail.",
        outcome_law=_outcome_law(request, labels),
        weight_report=describe_weights(np.ones(data.n), WeightSpec()),
        backend=data.backend,
        refit_folds="original" if frozen else "redrawn",
        replay_models=replay_lines,
        coupling_seed=seed,
        outcome_cause=request.outcome_cause,
    )


def _known_state(
    result: Any, models: ReplayModels, coded: Any, cause_index: Any, n_causes: int, seed: int
) -> _State:
    """The state of a surface run on supplied models (the known-hazard checks)."""
    time = np.asarray(result.data.time, dtype=float)
    return _drawn_state(
        models, time, cause_index, np.asarray(coded), n_causes, seed, preflight=False
    )
