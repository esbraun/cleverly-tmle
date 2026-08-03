"""Bounding, link functions and the bounded-continuous-outcome transformation.

TMLE for a continuous outcome is carried out on a ``[0, 1]``-scaled version of the
outcome, using a logistic fluctuation, and the resulting estimate is mapped back
to the original scale (Gruber & van der Laan, 2010).  Doing so keeps the targeted
estimate inside the observed range of ``Y`` and makes the fluctuation step a
weighted logistic regression regardless of the outcome type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from .._typing import FloatArray, GBounds

__all__ = [
    "OutcomeScaler",
    "bound",
    "expit",
    "logit",
    "resolve_g_bounds",
    "shrink_probabilities",
]

# Largest magnitude passed to ``exp`` before ``expit`` saturates anyway.
_EXP_CLIP = 709.0


def bound(x: FloatArray, lower: float, upper: float) -> FloatArray:
    """Clip ``x`` into ``[lower, upper]``.

    A thin wrapper over :func:`numpy.clip` that exists so the intent reads
    clearly at call sites, where bounding is a statistical decision rather than
    a numerical convenience.
    """
    if upper < lower:
        raise ValueError(f"upper bound {upper} is below lower bound {lower}")
    return np.clip(np.asarray(x, dtype=float), lower, upper)


def expit(x: FloatArray) -> FloatArray:
    """Numerically stable inverse logit."""
    z = np.clip(np.asarray(x, dtype=float), -_EXP_CLIP, _EXP_CLIP)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def logit(p: FloatArray, eps: float = 1e-12) -> FloatArray:
    """Logit with the argument pulled inside ``(0, 1)`` first."""
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.asarray(np.log(q / (1.0 - q)), dtype=float)


def shrink_probabilities(p: FloatArray, alpha: float) -> FloatArray:
    """Keep predicted probabilities away from the boundary of ``(0, 1)``.

    ``alpha`` follows the convention of R's ``tmle`` package: values are bounded
    into ``[1 - alpha, alpha]``, so the default ``alpha=0.9995`` maps to
    ``[0.0005, 0.9995]``.  Boundary values would make ``logit`` infinite and the
    fluctuation step degenerate.
    """
    if not 0.5 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0.5, 1); got {alpha}")
    return bound(p, 1.0 - alpha, alpha)


#: Target groups whose clever covariate reweights one arm by the propensity **odds**
#: ``g1 / g0``, and so needs the tighter ``for_att=True`` truncation.  Every other group
#: divides by ``g`` once, like the mean family, and takes the ordinary bound.
#:
#: A named set rather than the ``group == "mean"`` test the three call sites used to
#: repeat.  That test was written when ``mean``, ``att`` and ``atc`` were the only groups,
#: where "not mean" *was* "an odds"; as groups were added they silently inherited the ATT
#: bound instead -- and a group registered through
#: :func:`~cleverly.fluctuation.register_submodel` inherited it too, with nothing saying
#: so.  ``TMLEConfig.describe`` has always reported this bound as the "ATT/ATC" one, which
#: is what it now is.
CONDITIONAL_GROUPS: frozenset[str] = frozenset({"att", "atc"})


def g_bounds_for(
    group: str, mean_bounds: tuple[float, float], conditional_bounds: tuple[float, float]
) -> tuple[float, float]:
    """Which of a fit's two truncation bounds applies to ``group``."""
    return conditional_bounds if group in CONDITIONAL_GROUPS else mean_bounds


def resolve_g_bounds(spec: GBounds, n: float, *, for_att: bool = False) -> tuple[float, float]:
    """Turn a user-facing ``g_bounds`` specification into an explicit pair.

    ``"auto"`` reproduces the sample-size dependent defaults of R's ``tmle``:
    ``5 / (sqrt(n) * log(n))`` for the mean/ATE family, and a fixed ``0.025``
    for the ATT and ATC, whose influence curves are more sensitive to small
    ``1 - g(W)`` in the control arm.

    ``n`` is a float rather than a count because a weighted fit passes its *effective*
    sample size.  The rule is a bias-variance compromise -- truncate hard enough that the
    variance of ``1/g`` stays controlled, loosely enough that the truncation bias vanishes
    -- and the sample size governing both sides is the one the estimator's variance is
    really working from.  Under a design effect of 4 the row count would set a bound
    nearly three times too loose.  This is a deliberate divergence from R's ``tmle``,
    which resolves the rule at the row count whatever the weights say; it takes effect
    only for weighted fits, and is reported in the summary when it does.
    """
    if isinstance(spec, str):
        if spec != "auto":
            raise ValueError(f"g_bounds must be 'auto', a float, or a pair; got {spec!r}")
        if for_att:
            lower = 0.025
        else:
            if n < 3:
                raise ValueError("g_bounds='auto' needs at least 3 observations")
            lower = 5.0 / (np.sqrt(n) * np.log(n))
        lower = float(min(lower, 0.5 - 1e-9))
        return lower, 1.0 - lower

    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        lower = float(spec)
        if not 0.0 < lower < 0.5:
            raise ValueError(f"scalar g_bounds must lie in (0, 0.5); got {lower}")
        return lower, 1.0 - lower

    lower, upper = (float(v) for v in cast("tuple[float, float]", spec))
    if not 0.0 < lower < upper < 1.0:
        raise ValueError(f"g_bounds pair must satisfy 0 < lower < upper < 1; got {(lower, upper)}")
    return lower, upper


@dataclass(frozen=True)
class OutcomeScaler:
    """Maps a bounded continuous outcome onto ``[0, 1]`` and back.

    Attributes
    ----------
    lower, upper:
        The assumed support of the outcome.  Estimates are mapped back with
        ``range = upper - lower``; a *level* (a mean such as ``EY1``) maps back as
        ``lower + range * value`` while a *difference* (ATE, ATT, ATC) and every
        influence curve maps back as ``range * value`` — the location shift
        cancels in a contrast and does not affect a centred influence curve.
    """

    lower: float
    upper: float

    @property
    def range(self) -> float:
        return self.upper - self.lower

    @classmethod
    def identity(cls) -> OutcomeScaler:
        """Scaler for outcomes already on ``[0, 1]`` (binary or proportions)."""
        return cls(0.0, 1.0)

    @classmethod
    def from_outcome(
        cls,
        y: FloatArray,
        bounds: tuple[float, float] | None = None,
        *,
        pad: float = 0.1,
    ) -> OutcomeScaler:
        """Derive a scaler from the observed outcome.

        With ``bounds=None`` the observed range is widened by ``pad`` (10% by
        default) on each side, so the targeted estimate is not pinned to the
        sample extremes.  Pass ``bounds`` explicitly whenever the outcome has a
        known support -- it is the analogue of ``Qbounds`` in R's ``tmle``.
        """
        finite = np.asarray(y, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("cannot derive outcome bounds: no finite outcome values")

        if bounds is not None:
            lower, upper = (float(v) for v in bounds)
            if upper <= lower:
                raise ValueError(f"q_bounds must satisfy lower < upper; got {(lower, upper)}")
            if finite.min() < lower or finite.max() > upper:
                raise ValueError(
                    "observed outcomes fall outside q_bounds "
                    f"{(lower, upper)}: observed range "
                    f"{(float(finite.min()), float(finite.max()))}"
                )
            return cls(lower, upper)

        lo, hi = float(finite.min()), float(finite.max())
        if hi == lo:
            # Degenerate outcome: widen by a unit so the scaler stays invertible.
            return cls(lo - 0.5, hi + 0.5)
        width = hi - lo
        return cls(lo - pad * width, hi + pad * width)

    def scale(self, y: FloatArray) -> FloatArray:
        """Map outcomes onto ``[0, 1]``."""
        return np.asarray((np.asarray(y, dtype=float) - self.lower) / self.range, dtype=float)

    def unscale_level(self, value: float) -> float:
        """Map a mean back onto the original outcome scale."""
        return self.lower + self.range * value

    def unscale_difference(self, value: float) -> float:
        """Map a contrast of means back onto the original outcome scale."""
        return self.range * value

    def unscale_levels(self, values: FloatArray) -> FloatArray:
        """:meth:`unscale_level` applied elementwise to an array of predictions.

        A matrix of counterfactual predictions -- one column per arm, regime or cell --
        mapped back onto the outcome's own scale.  It is the same ``lower + range * v``
        the scalar method is, and it exists because three call sites had written that
        expression out rather than call a method annotated for one mean.

        The identity scaler returns the array rather than rebuilding it.  That is an
        allocation saved and not a difference: ``0.0 + 1.0 * v`` is ``v`` for every value
        a prediction can take.
        """
        array = np.asarray(values, dtype=float)
        if self.is_identity:
            return array
        return np.asarray(self.lower + self.range * array, dtype=float)

    def unscale_influence(self, ic: FloatArray) -> FloatArray:
        """Map an influence curve back onto the original outcome scale."""
        return np.asarray(self.range * np.asarray(ic, dtype=float), dtype=float)

    @property
    def is_identity(self) -> bool:
        return self.lower == 0.0 and self.upper == 1.0
