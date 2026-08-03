r"""Modified treatment policies: shifting a continuous treatment.

A :mod:`regime <cleverly.interventions.base>` sets the treatment to an arm, or spreads
mass over the arms, as a function of :math:`W` alone.  A **modified treatment policy**
reads the treatment a unit actually received and moves it:

.. math::

    d_\delta(a, w) = \begin{cases}
        a + \delta & a + \delta \le u(w) \\
        a          & \text{otherwise,}
    \end{cases}
    \qquad
    \Psi_\delta = E\bigl[\bar Q\bigl(d_\delta(A, W), W\bigr)\bigr].

"Everyone's exposure rises by :math:`\delta`, except where that would take them past what
is achievable."  The cap is what makes the parameter well defined without a positivity
assumption nobody can check: :math:`\bar Q` is never evaluated outside the range the data
covers (Diaz & van der Laan 2018; Haneuse & Rotnitzky 2013).

**The clever covariate.**  For the shift :math:`r`, evaluated at an arbitrary treatment
value :math:`a`,

.. math::

    h_r(a, W) = \frac{g(a - \delta_r \mid W)}{g(a \mid W)}\,\mathbb 1\{a \le u_r\}
                + \mathbb 1\{a > u_r - \delta_r\}

It is the ratio :math:`g^d / g` written out, where
:math:`g^d(b \mid w) = \sum_{a : d(a,w) = b} g(a \mid w)` is the density the policy
induces.  Both indicators come from that preimage and neither is decoration: a unit lands
at :math:`b` either by *being shifted there* from :math:`b - \delta` (possible only when
:math:`b \le u`, since otherwise the shift from :math:`b - \delta` was itself held back)
or by *staying put* (when :math:`b + \delta > u`).

The first indicator is invisible whenever the cap sits at or above the largest treatment
value, which is the common case -- and that is exactly why
``tests/discrete_law_shift.py`` declares **two** caps.  Dropping it passed every check
under a loose cap and failed the Gateaux derivative under a tight one.

Three checks it must pass, and does: at :math:`\delta = 0` it is identically one and the
influence curve collapses to :math:`Y - \Psi`, which is the influence curve of
:math:`E[Y]`; at a :math:`\delta` so large that nobody can move it is identically one
again; and on doses :math:`\{0,1,2,3\}` with :math:`\delta = 1` it gives
:math:`0,\ g(0)/g(1),\ g(1)/g(2),\ g(2)/g(3) + 1` when :math:`u = 3` and
:math:`0,\ g(0)/g(1),\ (g(1)+g(2))/g(2),\ 1` when :math:`u = 2`.

**Why an MTP is not the stochastic regime that induces it.**  Write
:math:`g^d(b \mid w) = \sum_{a : d(a,w) = b} g(a \mid w)` for the density the policy
induces.  The two parameters agree --

.. math::

    E[\bar Q(d(A,W), W)]
      = E_W\Bigl[\sum_a g(a \mid W)\, \bar Q(d(a,W), W)\Bigr]
      = E_W\Bigl[\sum_b g^d(b \mid W)\, \bar Q(b, W)\Bigr]

-- and the clever covariates agree entry for entry.  The *influence curves do not*: a
regime's plug-in term is :math:`\sum_b g^d(b \mid W) \bar Q(b, W)`, a function of
:math:`W`, while an MTP's is :math:`\bar Q(d(A,W), W)`, which reads the treatment the unit
actually received.  They agree only in conditional expectation given :math:`W`.  See
:func:`~cleverly.inference.influence.shift_means` for the variance identity this implies
and for why the two paths must not share an implementation.

**Why this needs no second fluctuation and an incremental intervention does.**  Both have
a :math:`g^\star` involving :math:`g`, which is exactly why the resemblance is dangerous,
and the distinction got *more* useful once both were implemented rather than less.  An
:class:`~cleverly.interventions.Incremental` intervention *defines its intervention
through* :math:`g`: its :math:`q_\delta` is a functional of :math:`P`, so
:math:`\Psi(\delta)` mentions the mechanism, the efficient influence function carries a
further term for the pathwise derivative through it (Kennedy 2019), and the estimator has
to fluctuate :math:`g` as well as :math:`\bar Q`.  A shift's :math:`d(a, w) = a + \delta`
is a known function and :math:`\Psi_\delta` above mentions no mechanism at all; the
induced density moving with :math:`P` is precisely what the plug-in-at-the-observed-row
term already accounts for.  So: a shift's :math:`g` is a nuisance, an incremental
intervention's is half of the estimand.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray
from ..data.causal_data import CausalData
from ..data.weighting import effective_sample_size
from ..exceptions import DataError, PositivityWarning
from ..learners.density import ConditionalDensity, warn_if_unresolved

__all__ = ["Shift", "ShiftSet", "ShiftSupport", "check_shift_support"]

_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


@dataclass(frozen=True)
class Shift:
    """Add ``delta`` to everyone's treatment, up to ``cap``.

    Attributes
    ----------
    delta:
        How far to move the treatment.  ``0.0`` is the *natural course* -- the policy that
        changes nothing -- whose mean is :math:`E[Y]` and which is the usual reference.
    cap:
        The largest treatment value the policy will assign; a unit whose shifted dose would
        exceed it keeps its own.  **Required, with no default**, and it is a declaration
        rather than something estimated.

        Estimating :math:`u(w)` from the data would make the *parameter* data-dependent:
        the reported standard error would condition on a fitted support boundary, and every
        bootstrap replicate would target a slightly different policy.  Defaulting it to
        ``max(A)`` is worse, pinning the estimand to an extreme order statistic.  So the
        analyst says what dose is achievable, which is a question about the world.

        ``cap=None`` means no cap.  It is allowed, it is the same arithmetic with the
        indicator zeroed, and it warns with the share of rows whose shifted dose leaves the
        observed range -- because there :math:`\\bar Q` is being extrapolated and
        identification needs :math:`A + \\delta` to be supported.
    name:
        What this policy is called in reported parameter names.  Defaults to ``"+0.5"`` /
        ``"-1"`` style, with ``"natural course"`` for ``delta=0``.
    """

    delta: float
    cap: float | None
    name: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta):
            raise DataError(f"a shift's delta must be finite; got {self.delta!r}")
        if self.cap is not None and not np.isfinite(self.cap):
            raise DataError(f"a shift's cap must be finite or None; got {self.cap!r}")
        if not self.name:
            default = "natural course" if self.delta == 0.0 else f"{self.delta:+g}"
            object.__setattr__(self, "name", default)

    def apply(self, treatment: FloatArray) -> tuple[FloatArray, BoolArray]:
        """``(shifted, capped)`` -- the assigned dose, and which rows the cap held back."""
        a = np.asarray(treatment, dtype=float).reshape(-1)
        moved = np.asarray(a + self.delta, dtype=float)
        if self.cap is None:
            return moved, np.zeros(a.size, dtype=bool)
        held = np.asarray(moved > float(self.cap), dtype=bool)
        return np.asarray(np.where(held, a, moved), dtype=float), held


@dataclass(frozen=True)
class ShiftSet:
    """Every declared shift, evaluated on the data and on the estimated density.

    Holds arrays and no callables, for the reason
    :class:`~cleverly.interventions.RegimeSet` does: a fit reached through
    :meth:`~cleverly.estimators.TMLE.retarget` -- a truncation sweep, the bootstrap, a
    result loaded from disk -- targets the same declared policies without the density
    being refit or the caller's objects being reachable.

    Attributes
    ----------
    names:
        One per shift, in the order they were declared.  Report labels.
    deltas:
        The shift sizes, in declaration order.  Note these are *not* the keys of the
        per-parameter arrays: those are the **codes** ``0.0 .. S-1.0``, exactly as
        :class:`~cleverly.interventions.RegimeSet` keys by regime code rather than by the
        regime itself.  Two shifts could share a delta only by being different policies
        with different caps, and a float key derived from a user-supplied number is a
        worse dictionary key than an ordinal.
    shifted:
        ``(n, S)``, :math:`d_r(A_i, W_i)`.
    ratio:
        ``(n, S)``, :math:`h_r(A_i, W_i)` -- the clever covariate at the *observed*
        treatment.  **Untruncated**, exactly as :attr:`Propensity.values
        <cleverly.estimators._nuisance.Propensity.values>` is: the bound belongs to
        targeting time so a truncation curve can sweep it without refitting.
    ratio_at:
        ``(n, S, S)``, :math:`h_r(d_s(A_i, W_i), W_i)` at ``[i, s, r]``.  The fluctuation
        updates :math:`\\bar Q` as a function of :math:`(a, W)`, so obtaining
        :math:`\\bar Q^*(d_s(A,W), W)` needs the covariate evaluated *at the shifted dose*
        -- hence a matrix per row rather than a vector.
    capped:
        ``(n, S)`` boolean, whether the cap held that row back.
    """

    names: tuple[str, ...]
    deltas: tuple[float, ...]
    shifted: FloatArray
    ratio: FloatArray
    ratio_at: FloatArray
    capped: BoolArray
    reference: float = 0.0

    def __post_init__(self) -> None:
        s = len(self.names)
        if len(self.deltas) != s:
            raise ValueError(f"{s} shift names but {len(self.deltas)} deltas")
        if len(set(self.names)) != s:
            raise ValueError(f"shift names must be distinct; got {list(self.names)}")
        n = self.shifted.shape[0]
        for name, array, shape in (
            ("shifted", self.shifted, (n, s)),
            ("ratio", self.ratio, (n, s)),
            ("capped", self.capped, (n, s)),
            ("ratio_at", self.ratio_at, (n, s, s)),
        ):
            if np.asarray(array).shape != shape:
                raise ValueError(
                    f"ShiftSet.{name} has shape {np.asarray(array).shape}, expected {shape}"
                )
        if self.reference not in self.codes:
            raise ValueError(
                f"reference={self.reference} is not one of the shift codes {list(self.codes)}"
            )

    # ------------------------------------------------------------------ build

    @classmethod
    def evaluate(
        cls,
        shifts: tuple[Shift, ...],
        data: CausalData,
        density: ConditionalDensity,
        *,
        reference: str | None = None,
    ) -> ShiftSet:
        """Evaluate every shift against the data and the estimated density.

        Every entry is a lookup into ``density``'s stored bin probabilities, so
        :math:`g(A \\mid W)` and :math:`g(A - \\delta \\mid W)` for one row necessarily come
        from the same out-of-fold model -- there is no second model to get wrong.
        """
        if not shifts:
            raise DataError("at least one shift is required")
        a = np.asarray(data.treatment, dtype=float).reshape(-1)
        names = tuple(shift.name for shift in shifts)
        deltas = tuple(float(shift.delta) for shift in shifts)
        shifted = np.column_stack([shift.apply(a)[0] for shift in shifts])
        capped = np.column_stack([shift.apply(a)[1] for shift in shifts])

        ratio = np.column_stack([_ratio(density, a, shift) for shift in shifts])
        ratio_at = np.stack(
            [
                np.column_stack([_ratio(density, shifted[:, s], shift) for shift in shifts])
                for s in range(len(shifts))
            ],
            axis=1,
        )

        for index, shift in enumerate(shifts):
            # The natural course is *meant* to move nobody -- it is the reference the
            # other shifts are contrasted against, and its mean is E[Y]. Warning that a
            # zero shift crosses no bin edge, or leaves no dose outside the support,
            # would fire on the recommended way to declare a fit and teach the reader to
            # ignore the warning that matters.
            if shift.delta == 0.0:
                continue
            warn_if_unresolved(density, shifted[:, index], a)
            _warn_outside_support(shift, shifted[:, index], a)

        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(f"reference={reference!r} is not one of the shifts {list(names)}")
            code = float(names.index(reference))
        return cls(names, deltas, shifted, ratio, ratio_at, capped, code)

    # ----------------------------------------------------------------- access

    @property
    def n(self) -> int:
        return int(self.shifted.shape[0])

    @property
    def n_shifts(self) -> int:
        return len(self.names)

    @property
    def codes(self) -> tuple[float, ...]:
        """The keys the per-parameter arrays use, ``(0.0, ..., S-1.0)``."""
        return tuple(float(index) for index in range(self.n_shifts))

    @property
    def labels(self) -> dict[float, str]:
        """Code to reported label, which is what ``parameter_name`` is given."""
        return {float(index): name for index, name in enumerate(self.names)}

    def label(self, code: float) -> str:
        return self.labels[float(code)]

    @property
    def design(self) -> FloatArray:
        """``(n, S + 1, S)`` -- the covariate at the observed dose, then at each shifted one.

        One array rather than two, because :func:`~cleverly.fluctuation.submodel_for`
        dispatches on the group name alone and every builder takes the same keyword-only
        signature; ``shifts=`` is that one keyword.  Row block ``0`` is the covariate at
        the observed treatment and block ``s + 1`` the covariate at :math:`d_s(A, W)`.
        """
        return np.concatenate([self.ratio[:, None, :], self.ratio_at], axis=1)

    def subset(self, index: Any) -> ShiftSet:
        """The same policies on a row subset -- a fold, a bootstrap resample.

        Sliced rather than re-evaluated, for the reason
        :meth:`~cleverly.interventions.RegimeSet.subset` gives: a policy is the same
        policy on a subsample, and re-deriving it would let the resample redefine the
        estimand.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(
            self,
            shifted=self.shifted[idx],
            ratio=self.ratio[idx],
            ratio_at=self.ratio_at[idx],
            capped=self.capped[idx],
        )


def _ratio(density: ConditionalDensity, values: FloatArray, shift: Shift) -> FloatArray:
    """:math:`h_r(a, W)` at the given treatment values -- the module docstring's formula."""
    a = np.asarray(values, dtype=float).reshape(-1)
    numerator = density.density_at(a - shift.delta)
    denominator = density.density_at(a)
    # A row whose *observed* dose has zero estimated density cannot be reweighted at all;
    # it is a support failure rather than a large weight, and reporting it as an infinite
    # covariate would put a NaN through the Newton solve. Zero is the honest value: the
    # row contributes nothing to the score, and check_shift_support counts it.
    safe = np.where(denominator > 0.0, denominator, 1.0)
    covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
    if shift.cap is not None:
        cap = float(shift.cap)
        # Reachable-from-below: a unit can only have been *shifted* to `a` if the shift
        # from `a - delta` was not itself held back, which needs `a <= cap`. Above the cap
        # the only way to be at `a` is to have stayed there, so the ratio term drops out
        # entirely and the indicator below is the whole covariate.
        covariate = covariate * (a <= cap).astype(float)
        covariate = covariate + (a > cap - shift.delta).astype(float)
    return np.asarray(covariate, dtype=float)


def _warn_outside_support(shift: Shift, shifted: FloatArray, observed: FloatArray) -> None:
    """An uncapped shift extrapolates; say how much of the sample it does that for."""
    if shift.cap is not None:
        return
    beyond = float(np.mean(shifted > float(np.max(observed))))
    if beyond > 0.0:
        warnings.warn(
            f"shift {shift.name!r} has cap=None, and {beyond:.1%} of rows are assigned a "
            f"dose above the largest one observed ({float(np.max(observed)):.3g}). The "
            "outcome regression is extrapolating there, and identification needs the "
            "shifted dose to be supported. Declare a cap= if you know what dose is "
            "achievable.",
            PositivityWarning,
            stacklevel=4,
        )


# ------------------------------------------------------------------ diagnostics


@dataclass(frozen=True)
class ShiftSupport:
    """Overlap for one shift: how hard the density ratio is working, and where it fails."""

    name: str
    delta: float
    cap: float | None
    min_density: float
    ratio_quantiles: dict[float, float]
    max_ratio: float
    effective_sample_size: float
    ess_ratio: float
    capped_fraction: float
    unsupported: int
    #: Smallest :math:`\pi(A, W)\,q_z(A, W)` among the mechanisms that divide the
    #: covariate alongside the ratio, or ``None`` when the fit declared neither.  The
    #: quantiles and ESS above are of the *whole* weight when this is not ``None``.
    min_mechanism: float | None = None

    def summary(self) -> str:
        quantiles = ", ".join(f"{q:.0%}: {v:.3g}" for q, v in sorted(self.ratio_quantiles.items()))
        mechanism = (
            "" if self.min_mechanism is None else f", min mechanism={self.min_mechanism:.3g}"
        )
        label = "ratio" if self.min_mechanism is None else "weight"
        return (
            f"{self.name}: min g(A|W)={self.min_density:.3g}, max {label}={self.max_ratio:.3g}"
            f"{mechanism}, "
            f"ESS={self.effective_sample_size:.0f} ({self.ess_ratio:.1%} of n), "
            f"capped={self.capped_fraction:.1%}, unsupported={self.unsupported}\n"
            f"    {label} quantiles -- {quantiles}"
        )


def check_shift_support(
    shifts: ShiftSet,
    density: ConditionalDensity,
    treatment: FloatArray,
    *,
    mechanisms: Sequence[FloatArray] = (),
) -> dict[str, ShiftSupport]:
    """Per-shift overlap, in the vocabulary :mod:`cleverly.interventions.support` uses.

    The quantity a shift's positivity rests on is not a propensity but the *ratio*
    :math:`g(a - \\delta \\mid w) / g(a \\mid w)`: it is what multiplies each residual, so
    its tail is where one row starts to dominate the estimating equation.

    ``mechanisms`` are the further ``(n, S + 1)`` denominators a fit declared -- the
    missingness mechanism under ``delta=``, the intermediate density under
    ``intermediate=`` -- of which only column ``0``, the value at the row's own dose, is
    read here.  The weight the estimating equation forms is
    :math:`h_r(A, W) / \\{\\pi(A, W) q_z(A, W)\\}`, so the two reweightings *multiply* and
    an effective sample size taken of the ratio alone understates the strain --
    :mod:`cleverly.sensitivity.positivity` makes the same argument about ``1 / g`` and a
    population weight.  Passing nothing reports the ratio by itself, which is what a fit
    with no such mechanism means.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    observed_density = density.density_at(a)
    at_observed = [np.asarray(m, dtype=float)[:, 0] for m in mechanisms]
    denominator = np.ones(a.size)
    for values in at_observed:
        denominator = denominator * values
    out: dict[str, ShiftSupport] = {}
    for index, name in enumerate(shifts.names):
        weight = shifts.ratio[:, index] / denominator
        finite = weight[np.isfinite(weight)]
        ess = effective_sample_size(finite, on_degenerate=0.0)
        out[name] = ShiftSupport(
            name=name,
            delta=shifts.deltas[index],
            cap=None,
            min_density=float(observed_density.min()),
            ratio_quantiles={q: float(np.quantile(finite, q)) for q in _QUANTILES},
            max_ratio=float(finite.max()) if finite.size else 0.0,
            effective_sample_size=ess,
            ess_ratio=ess / a.size if a.size else 0.0,
            capped_fraction=float(np.mean(shifts.capped[:, index])),
            unsupported=int(np.sum(observed_density <= 0.0)),
            min_mechanism=float(denominator.min()) if at_observed else None,
        )
    return out
