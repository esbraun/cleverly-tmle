r"""Simultaneous confidence bands via the multiplier bootstrap.

Reporting seven estimands with seven 95% intervals does not give 95% confidence
that *all seven* contain their truth -- with correlated estimates the joint
coverage is somewhere below that, and the usual fixes (Bonferroni) are needlessly
wide precisely because the estimates are correlated.

The multiplier bootstrap uses the influence curves, which are already available,
to get the joint distribution right.  Draw i.i.d. mean-zero, unit-variance
multipliers :math:`\xi_i` and form

.. math::

    R^*_{b,j} = \frac{1}{n}\sum_i \xi_{b,i}
                \bigl(\mathrm{IC}_{i,j} - \overline{\mathrm{IC}}_j\bigr),
    \qquad
    t^*_b = \max_j \frac{|R^*_{b,j}|}{\widehat{se}_j}.

The :math:`1 - \alpha` quantile :math:`\hat c` of :math:`t^*` gives bands
:math:`\hat\psi_j \pm \hat c \cdot \widehat{se}_j` with asymptotic *joint* coverage
:math:`1 - \alpha`.  Because :math:`\hat c` adapts to the observed correlation, it
sits between the pointwise critical value (1.96) and the Bonferroni one, and
collapses to 1.96 when only one estimand is requested.

No resampling of the data is involved, so this costs one matrix product per chunk
of replicates -- cheap enough to run by default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._typing import FloatArray, IntArray
from .cluster import cluster_sums
from .influence import ParameterEstimate

__all__ = ["SimultaneousBands", "multiplier_critical_value", "simultaneous_bands"]

MultiplierKind = Literal["rademacher", "mammen", "normal"]

# Replicates are generated in chunks so the multiplier matrix stays small even for
# large n and many replicates.
_CHUNK = 256


@dataclass(frozen=True)
class SimultaneousBands:
    """Joint confidence bands across several estimands."""

    critical_value: float
    bands: dict[str, tuple[float, float]]
    alpha: float
    n_replicates: int
    kind: str

    @property
    def pointwise_critical_value(self) -> float:
        """The critical value a single-estimand interval would have used."""
        from scipy import stats

        return float(stats.norm.ppf(1.0 - self.alpha / 2.0))


def _multipliers(
    rng: np.random.Generator, shape: tuple[int, int], kind: MultiplierKind
) -> FloatArray:
    """Mean-zero, unit-variance multipliers."""
    if kind == "rademacher":
        return np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    if kind == "normal":
        return rng.standard_normal(shape)
    if kind == "mammen":
        # Mammen's two-point distribution: mean 0, variance 1, third moment 1, which
        # matches one more moment of the sampling distribution than Rademacher does.
        root5 = np.sqrt(5.0)
        prob = (root5 + 1.0) / (2.0 * root5)
        low = -(root5 - 1.0) / 2.0
        high = (root5 + 1.0) / 2.0
        return np.where(rng.random(shape) < prob, low, high)
    raise ValueError(f"kind must be 'rademacher', 'mammen' or 'normal'; got {kind!r}")


def multiplier_critical_value(
    influence_curves: FloatArray,
    std_errors: FloatArray,
    *,
    n: int,
    cluster: IntArray | None = None,
    alpha: float = 0.05,
    n_replicates: int = 1000,
    kind: MultiplierKind = "rademacher",
    random_state: int | None = None,
) -> float:
    """The ``1 - alpha`` quantile of the max-t statistic."""
    ic = np.asarray(influence_curves, dtype=float)
    if ic.ndim != 2:
        raise ValueError(f"expected an (n, m) influence-curve matrix; got shape {ic.shape}")
    se = np.asarray(std_errors, dtype=float).reshape(-1)
    if se.shape[0] != ic.shape[1]:
        raise ValueError(
            f"got {ic.shape[1]} influence curve(s) but {se.shape[0]} standard error(s)"
        )
    if ic.shape[1] == 1:
        from scipy import stats

        return float(stats.norm.ppf(1.0 - alpha / 2.0))

    scores = cluster_sums(ic, cluster) if cluster is not None else ic
    centred = scores - scores.mean(axis=0, keepdims=True)
    usable = np.isfinite(se) & (se > 0)
    if not usable.any():
        return float("nan")

    rng = np.random.default_rng(random_state)
    statistics = np.empty(n_replicates, dtype=float)
    done = 0
    while done < n_replicates:
        size = min(_CHUNK, n_replicates - done)
        xi = _multipliers(rng, (size, centred.shape[0]), kind)
        draws = (xi @ centred) / n
        standardised = np.abs(draws[:, usable]) / se[usable]
        statistics[done : done + size] = standardised.max(axis=1)
        done += size
    return float(np.quantile(statistics, 1.0 - alpha))


def simultaneous_bands(
    estimates: Mapping[str, ParameterEstimate] | Sequence[ParameterEstimate],
    *,
    alpha: float = 0.05,
    n_replicates: int = 1000,
    kind: MultiplierKind = "rademacher",
    random_state: int | None = None,
    cluster: IntArray | None = None,
) -> SimultaneousBands:
    """Joint confidence bands over a set of estimates.

    Bands are reported on each estimand's own scale, so a ratio's band is
    exponentiated from the log scale exactly as its pointwise interval is.
    """
    items = (
        list(estimates.items())
        if isinstance(estimates, Mapping)
        else [(estimate.name, estimate) for estimate in estimates]
    )
    if not items:
        raise ValueError("no estimates supplied")
    lengths = {estimate.influence_curve.shape[0] for _, estimate in items}
    if len(lengths) != 1:
        raise ValueError(f"influence curves have inconsistent lengths: {lengths}")

    n = items[0][1].n
    ic = np.column_stack([estimate.influence_curve for _, estimate in items])
    se = np.array([estimate.std_error for _, estimate in items])
    critical = multiplier_critical_value(
        ic,
        se,
        n=n,
        cluster=cluster,
        alpha=alpha,
        n_replicates=n_replicates,
        kind=kind,
        random_state=random_state,
    )

    bands: dict[str, tuple[float, float]] = {}
    for name, estimate in items:
        half_width = critical * estimate.std_error
        if estimate.scale == "ratio":
            assert estimate.log_psi is not None
            bands[name] = (
                float(np.exp(estimate.log_psi - half_width)),
                float(np.exp(estimate.log_psi + half_width)),
            )
        else:
            bands[name] = (estimate.psi - half_width, estimate.psi + half_width)
    return SimultaneousBands(
        critical_value=critical,
        bands=bands,
        alpha=alpha,
        n_replicates=n_replicates,
        kind=kind,
    )
