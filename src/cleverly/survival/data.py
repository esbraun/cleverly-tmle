"""The container of a continuous-time survival design, and its link to the study API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Final

import numpy as np

from .._refusal import Refusal, RefusalReason
from ..exceptions import CapabilityError, DataError, MethodConfigurationError

__all__ = ["ContinuousSurvivalData", "continuous_kwargs", "static_arms"]

ROADMAP: Final = "X6"
_ALTERNATIVE: Final = (
    "Declare grid= on TimeToEvent for the binned survival estimators, or estimate with "
    "TMLEMethod on the continuous design."
)
#: Why every method other than the one-step TMLE refuses a continuous follow-up time.
CONTINUOUS_METHOD_REFUSAL: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "on TimeToEvent(grid='continuous') only the one-step TMLE has a continuous-time form.",
    alternative=_ALTERNATIVE,
    roadmap=ROADMAP,
)


@dataclass(frozen=True)
class ContinuousSurvivalData:
    """One row per unit with a follow-up time that is not binned.

    Parameters
    ----------
    time : ndarray
        ``(n,)`` positive finite follow-up times.
    event : ndarray
        ``(n,)`` event codes, ``0`` for censored.
    treatment : ndarray
        ``(n,)`` the baseline treatment.
    baseline : ndarray
        ``(n, p)`` baseline covariates.
    names : tuple of str
        ``(time, event, treatment, *baseline)`` column names.
    causes : dict of int to str or None
        The label of each nonzero event code, as declared.
    backend : str or None
        The dataframe library the frame arrived in, so results come back in it.
    """

    time: np.ndarray
    event: np.ndarray
    treatment: np.ndarray
    baseline: np.ndarray
    names: tuple[str, ...]
    causes: dict[int, str] | None
    backend: str | None = None

    @property
    def n(self) -> int:
        """The number of units."""
        return int(self.time.size)

    @property
    def baseline_names(self) -> tuple[str, ...]:
        """The baseline covariate names."""
        return self.names[3:]

    @classmethod
    def from_frame(
        cls,
        data: Any,
        *,
        time: str,
        event: str,
        treatment: str,
        baseline: Sequence[str],
        causes: Mapping[int, str] | None = None,
    ) -> ContinuousSurvivalData:
        """Validate a frame and hold its columns.

        Parameters
        ----------
        data : DataFrame
            A pandas or polars frame with one row per unit.
        time : str
            The follow-up time column.
        event : str
            The event-code column.
        treatment : str
            The baseline treatment column.
        baseline : sequence of str
            The baseline covariates, at least one.
        causes : mapping of int to str, or None
            The label of each nonzero event code.

        Returns
        -------
        ContinuousSurvivalData
            The container.
        """
        from ..utils.frames import as_frame, backend_of, is_dataframe

        if not is_dataframe(data):
            raise DataError("TimeToEvent(grid='continuous') expects a pandas or polars DataFrame")
        frame = as_frame(data)
        names = (str(time), str(event), str(treatment), *(str(b) for b in baseline))
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise DataError(f"columns {missing} are not in the data")
        if len(names) < 4:
            raise DataError("TimeToEvent names at least one baseline covariate")
        T = frame[names[0]].to_numpy().astype(float)
        D = frame[names[1]].to_numpy().astype(float)
        if not np.all(np.isfinite(T)) or np.any(T <= 0):
            raise DataError("every follow-up time is a positive finite number")
        if not np.all(np.isfinite(D)) or np.any(D < 0) or np.any(D % 1):
            raise DataError("every event code is a non-negative integer, 0 for censored")
        codes = sorted({int(c) for c in D if c != 0})
        if not codes:
            raise DataError("the event column holds no event")
        if causes is not None and set(codes) - set(causes):
            raise DataError(
                f"event codes {sorted(set(codes) - set(causes))} are not named by causes"
            )
        L = np.column_stack([frame[name].to_numpy().astype(float) for name in names[3:]])
        if not np.all(np.isfinite(L)):
            raise DataError("baseline covariates are finite numbers")
        A = frame[names[2]].to_numpy()
        return cls(
            T,
            D.astype(int),
            A,
            L,
            names,
            None if causes is None else {int(k): str(v) for k, v in causes.items()},
            backend_of(frame),
        )


def static_arms(regimens: Any, levels: Sequence[Any]) -> dict[str, Any]:
    """Read two static arms from a regimen declaration, or refuse.

    Parameters
    ----------
    regimens : mapping or sequence
        ``{label: level}``, ``{label: Static(level)}`` or a sequence of levels.
    levels : sequence
        The observed treatment levels.

    Returns
    -------
    dict of str to level
        Two labels, each mapped to a distinct observed level.
    """
    from ..interventions.base import Static

    if isinstance(regimens, Mapping):
        items = list(regimens.items())
    elif isinstance(regimens, Sequence) and not isinstance(regimens, (str, bytes)):
        items = [(str(getattr(v, "level", v)), v) for v in regimens]
    else:
        items = []
    arms: dict[str, Any] = {}
    for label, value in items:
        level = value.level if isinstance(value, Static) else value
        if (
            isinstance(level, (Mapping, Sequence, np.ndarray)) and not isinstance(level, str)
        ) or callable(level):
            arms = {}
            break
        arms[str(label)] = level
    observed = set(np.asarray(levels).tolist())
    if len(arms) != 2 or len(set(arms.values())) != 2 or set(arms.values()) != observed:
        raise CapabilityError(
            "TimeToEvent(grid='continuous') estimates two static arms that set the binary "
            "baseline treatment to each of its two observed levels, such as "
            "RegimeContrast({'treated': 1, 'control': 0}, reference='control'). A dynamic "
            "rule, a stochastic policy or more than two levels has no continuous-time study "
            f"yet (roadmap {ROADMAP}).",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_ALTERNATIVE,
            roadmap=ROADMAP,
        )
    return arms


#: The method settings the continuous-time engine reads, as ``(group, field)``.
_READ: Final = frozenset(
    {
        ("cross_fitting", "enabled"),
        ("cross_fitting", "n_folds"),
        ("cross_fitting", "learner_folds"),
        ("models", "treatment_learner"),
        ("targeting", "g_bounds"),
        ("targeting", "max_iter"),
        ("targeting", "step_size"),
        ("targeting", "tol"),
        ("inference", "alpha"),
        ("inference", "simultaneous"),
        ("inference", "n_multiplier"),
        ("inference", "multiplier_kind"),
        ("inference", "band_scale"),
        ("runtime", "random_state"),
        ("runtime", "run_id"),
        ("runtime", "n_jobs"),
    }
)
_GROUPS: Final = ("models", "cross_fitting", "targeting", "inference", "runtime")


def continuous_kwargs(method: Any) -> dict[str, Any]:
    r"""Translate a :class:`~cleverly.TMLEMethod` for the continuous-time engine.

    Parameters
    ----------
    method : TMLEMethod
        The normalized method.

    Returns
    -------
    dict of str to Any
        :class:`~cleverly.survival.ContinuousSurvivalTMLE` keyword arguments and the band
        settings.

    Notes
    -----
    ``max_iter``, ``step_size`` and ``tol`` at :class:`~cleverly.TMLEMethod`'s defaults, which
    belong to the binned walk, read as the one-step walk's own defaults: 500 steps from a
    first step of 0.1, stopped by item 10's rule :math:`1 / (\sqrt n \log n)`.  A value equal
    to the binned default cannot be told apart from it, so ``tol=1e-10`` given explicitly
    also reads as item 10's rule; pass any other value to set the cutoff.
    """
    from ..methods import SHORTCUTS, TMLEMethod, _differs_from_default

    if type(method) is not TMLEMethod:
        raise CapabilityError(
            f"{type(method).__name__} has no continuous-time form; on "
            "TimeToEvent(grid='continuous') only TMLEMethod (the one-step TMLE of Rytgaard and "
            f"van der Laan 2024) runs (roadmap {ROADMAP}).",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_ALTERNATIVE,
            roadmap=ROADMAP,
        )
    defaults = TMLEMethod()
    refused = []
    for group in _GROUPS:
        public = {attribute: shortcut for shortcut, attribute in SHORTCUTS.get(group, {}).items()}
        for item in fields(getattr(method, group)):
            if (group, item.name) in _READ:
                continue
            value = getattr(getattr(method, group), item.name)
            default = getattr(getattr(defaults, group), item.name)
            if _differs_from_default(value, default):
                refused.append(public.get(item.name, item.name))
    if refused:
        raise MethodConfigurationError(
            f"TimeToEvent(grid='continuous') cannot apply {sorted(refused)}: the one-step "
            "continuous-time TMLE reads cross_fit, n_folds, learner_folds, treatment_learner, "
            "g_bounds, max_iter, step_size, tol, alpha, the band settings and the runtime "
            f"settings only (roadmap {ROADMAP})."
        )
    targeting = method.targeting
    g_bounds = targeting.g_bounds
    if g_bounds != "auto":
        if isinstance(g_bounds, Sequence):
            low, high = (float(v) for v in g_bounds)
            if high < 1.0:
                raise MethodConfigurationError(
                    f"g_bounds={tuple(g_bounds)!r} caps the treatment probability at {high:g}; "
                    "the continuous-time TMLE bounds only pi * S_c(u-) from below, so pass a "
                    "lower bound alone or a pair whose upper bound is 1."
                )
            g_bounds = low
        else:
            g_bounds = float(g_bounds)
    inference = method.inference
    return {
        "cross_fit": method.cross_fitting.enabled,
        "n_folds": method.cross_fitting.n_folds,
        "learner_folds": method.cross_fitting.learner_folds,
        "treatment_learner": method.models.treatment_learner,
        "g_bounds": g_bounds,
        # The defaults belong to the binned walk; the one-step walk's own are 500 steps
        # from a first step of 0.1 (`concrete`'s MaxUpdateIter and OneStepEps).
        "max_iter": 500
        if targeting.max_iter == defaults.targeting.max_iter
        else targeting.max_iter,
        "step_size": (
            0.1 if targeting.step_size == defaults.targeting.step_size else targeting.step_size
        ),
        "tol": None if targeting.tol == defaults.targeting.tol else targeting.tol,
        "alpha": inference.alpha,
        # None is the package default: a band whenever a fit reports several parameters,
        # which this one always does.
        "simultaneous": True if inference.simultaneous is None else bool(inference.simultaneous),
        "n_multiplier": int(inference.n_multiplier),
        "multiplier_kind": inference.multiplier_kind,
        "band_scale": inference.band_scale,
        "random_state": method.runtime.random_state,
    }
