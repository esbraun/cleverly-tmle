r"""Clustered longitudinal laws whose regimen truths are the unclustered ones, exactly.

Two registered studies use these laws: ``clustered-cross-fitted-ltmle`` and
``few-cluster-cross-fitted-ltmle``. The unit tests of the cross-fitted clustered ``LTMLE`` use
them too.

**Why a study-side law.** ``make_longitudinal(cluster_size=m)`` shares part of the ``L2`` noise
within a cluster. On the contrast its IID SE over the empirical SD is 0.98 at :math:`m=10` and
0.84 at :math:`m=25` (single-process probes of 400 draws), above the 0.80 ceiling that a ``clustered_inference``
control must fall below. The shipped generator and the notebooks that use it are unchanged.

**The end-of-study law.** Draw ``make_longitudinal(n, seed=seed, censoring=True)`` without
``cluster_size``. Every column but ``Y`` is kept byte for byte, and an ``id`` column is added.
The observed outcome is redrawn as

.. math::

    Y_i \sim \mathrm{Bernoulli}\{p_i + \delta\, u_{c(i), A_{1i}}\min(p_i, 1-p_i)\},

with :math:`p_i` the ``make_longitudinal`` outcome probability at the row's own history and
two independent latents per cluster, one per :math:`A_1` arm, so that a regimen contrast
carries the clustering. :math:`\delta = 0.95`.

**The survival law.** Draw ``make_longitudinal_survival(n, seed=seed, censoring=True)``. The
same component goes on the last event node only. :math:`Y_2` is redrawn only where
:math:`Y_1 = 0` and the unit is observed at node 2, from
:math:`h_2 + \delta u \min(h_2, 1-h_2)` at the row's history. The carried-forward 1 stays where
:math:`Y_1 = 1`, and every ``NaN`` stays. A latent on :math:`Y_1` would make survival to node 2
informative about :math:`u`, so none is placed there.

**Truth.** :math:`E_u[p + \delta u \min(p, 1-p)] = p` exactly for every latent with
:math:`E[u] = 0`. The latent enters no node but the last outcome, so no earlier node carries
information about it, the conditional law of the last outcome given the history is the
unclustered law, and the regimen truths are the ``make_longitudinal`` quadrature truths. For
survival, :math:`F(2) = E[h_1 + (1 - h_1) h_2]` is linear in :math:`h_2`, and ``L2`` does not
read ``Y1`` (``survival_truth``), so :math:`E_u` returns ``survival_truth`` exactly. The latent
enters no treatment or censoring node, so ``KnownLongitudinalMechanism`` stays exact.

**Two declared latents.** ``"rademacher"``: :math:`u` uniform on :math:`\{-1, +1\}`, for
``clustered-cross-fitted-ltmle``. ``"scaled"``: :math:`u = s v`, with :math:`s` Rademacher and
:math:`v` uniform on :math:`[0.5, 1]`, for ``few-cluster-cross-fitted-ltmle``. Both have
:math:`E[u] = 0`. With Rademacher latents every training cluster of a few-cluster fit can share
one sign, so the followers carry one outcome value and the fit fails (about
:math:`K\,2^{-\text{training clusters}}`). The registered harness refuses a study with a failed
replication. With a continuous :math:`v`, a degenerate training fold needs every training
cluster near :math:`v = 1`. The scaled latent has :math:`E[u^2] = 7/12`, which weakens the
clustering, and the survival control of the gated study cannot afford that.

**Sizes.** ``"equal<m>"`` gives every cluster ``m`` rows. ``"unequal40"`` draws each size
uniformly from ``{10, ..., 70}``, with mean 40 and finite support, and independently of every
row variable, so the size is not informative. The size is redrawn per replication.

**Streams.** The sizes, the latents and the outcome uniforms come from three child streams of the
replication seed, ``[seed, 7, 0]``, ``[seed, 7, 1]`` and ``[seed, 7, 2]``. An ``equal`` and an
``unequal40`` draw at one seed therefore share their latents. The row variables come from
``seed`` itself, as ``make_longitudinal`` draws them.

**Measured (single-process probes, the ``canonical-ltmle-crossfit`` subject with ``id=``, five
whole-cluster folds).** On ``equal40`` at 100 clusters the IID SE over the empirical SD of the
contrast is 0.55 and the cluster SE over SD is 1.01 (400 draws). On ``unequal40`` at 100
clusters the values are 0.52 and 1.03. On the survival law the risk contrast at ``t=2`` reads
0.66 and 0.99 (600 draws). :data:`INTRACLUSTER_WITNESS` records the within-cluster correlation
that ``tests/unit/test_clustered_longitudinal_laws.py`` pins.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.datasets.longitudinal import _hazard_two, _outcome_probability

#: The scale of the cluster component of the last outcome node.
DELTA: Final[float] = 0.95
#: The declared latents: ``"rademacher"`` for the gated study, ``"scaled"`` for the few-cluster
#: study.
LATENTS: Final[tuple[str, ...]] = ("rademacher", "scaled")
#: The support of the scaled latent's magnitude.
SCALED_MAGNITUDE: Final[tuple[float, float]] = (0.5, 1.0)
#: The inclusive support of the ``"unequal40"`` cluster size.
UNEQUAL_SIZES: Final[tuple[int, int]] = (10, 70)
#: The second element of every child stream key, which keeps the streams apart from any other
#: stream the replication seed feeds.
STREAM_TAG: Final[int] = 7
#: The child stream index of the sizes, the latents and the outcome uniforms.
SIZE_STREAM: Final[int] = 0
LATENT_STREAM: Final[int] = 1
OUTCOME_STREAM: Final[int] = 2
#: The intracluster correlation of ``Y - p`` among observed ``A1 = 1`` rows on the
#: end-of-study law, ``equal40``, 100 clusters, seed 20261101 (one-way ANOVA estimator): measured 0.2741 (Rademacher)
#: and 0.1677 (scaled). The unit test pins a floor below each.
INTRACLUSTER_WITNESS: Final[dict[str, float]] = {"rademacher": 0.25, "scaled": 0.15}


def child_stream(seed: int, index: int) -> np.random.Generator:
    """One of the three child streams of a replication seed.

    Parameters
    ----------
    seed : int
        The replication seed.
    index : int
        :data:`SIZE_STREAM`, :data:`LATENT_STREAM` or :data:`OUTCOME_STREAM`.

    Returns
    -------
    numpy.random.Generator
        The stream ``[seed, STREAM_TAG, index]``.
    """
    return np.random.default_rng([int(seed), STREAM_TAG, int(index)])


def cluster_sizes(sizes: str, clusters: int, seed: int) -> np.ndarray:
    """The row count of each cluster.

    Parameters
    ----------
    sizes : str
        ``"equal<m>"`` or ``"unequal40"``.
    clusters : int
        The number of clusters.
    seed : int
        The replication seed. Only ``"unequal40"`` reads its size stream.

    Returns
    -------
    numpy.ndarray
        One positive integer per cluster.
    """
    if sizes.startswith("equal"):
        return np.full(clusters, int(sizes.removeprefix("equal")), dtype=int)
    if sizes == "unequal40":
        low, high = UNEQUAL_SIZES
        return child_stream(seed, SIZE_STREAM).integers(low, high + 1, size=clusters)
    raise ValueError(f"unknown sizes {sizes!r}; expected 'equal<m>' or 'unequal40'")


def cluster_latents(latent: str, clusters: int, seed: int) -> np.ndarray:
    """Two independent mean-zero latents per cluster, one per first-node arm.

    Parameters
    ----------
    latent : str
        One of :data:`LATENTS`.
    clusters : int
        The number of clusters.
    seed : int
        The replication seed.

    Returns
    -------
    numpy.ndarray
        Shape ``(clusters, 2)``. Column ``a`` serves the rows with ``A1 = a``.
    """
    rng = child_stream(seed, LATENT_STREAM)
    sign = rng.choice([-1.0, 1.0], size=(clusters, 2))
    if latent == "rademacher":
        return sign
    if latent == "scaled":
        low, high = SCALED_MAGNITUDE
        return sign * rng.uniform(low, high, size=(clusters, 2))
    raise ValueError(f"unknown latent {latent!r}; expected one of {LATENTS}")


def perturbed_probability(p: Any, u: Any, delta: float = DELTA) -> np.ndarray:
    """``p + delta * u * min(p, 1 - p)``, which stays in ``[0, 1]`` for ``|delta u| <= 1``.

    Parameters
    ----------
    p : array_like
        The unclustered outcome probability.
    u : array_like
        The latent of the row's cluster and first-node arm.
    delta : float
        The scale of the cluster component.

    Returns
    -------
    numpy.ndarray
        The clustered outcome probability.
    """
    p = np.asarray(p, dtype=float)
    return p + delta * np.asarray(u, dtype=float) * np.minimum(p, 1.0 - p)


def _layout(sizes: str, clusters: int, seed: int) -> tuple[np.ndarray, int]:
    counts = cluster_sizes(sizes, clusters, seed)
    return np.repeat(np.arange(clusters), counts), int(counts.sum())


def draw_end_of_study(
    clusters: int,
    sizes: str,
    seed: int,
    *,
    latent: str = "rademacher",
    delta: float = DELTA,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One replication of the clustered end-of-study law.

    Parameters
    ----------
    clusters : int
        The number of clusters.
    sizes : str
        ``"equal<m>"`` or ``"unequal40"``.
    seed : int
        The replication seed.
    latent : str
        One of :data:`LATENTS`.
    delta : float
        The scale of the cluster component.

    Returns
    -------
    frame : pandas.DataFrame
        The ``make_longitudinal`` columns with the redrawn ``Y`` and a float ``id`` column.
    truth : dict of str to float
        ``make_longitudinal``'s truth dictionary, unchanged.
    """
    ids, n = _layout(sizes, clusters, seed)
    frame, truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
    a1 = frame["A1"].to_numpy().astype(int)
    p = _outcome_probability(
        frame["W1"].to_numpy(),
        frame["W2"].to_numpy(),
        np.nan_to_num(frame["L2"].to_numpy()),
        frame["A1"].to_numpy(),
        np.nan_to_num(frame["A2"].to_numpy()),
    )
    u = cluster_latents(latent, clusters, seed)[ids, a1]
    uniforms = child_stream(seed, OUTCOME_STREAM).random(n)
    y = (uniforms < perturbed_probability(p, u, delta)).astype(float)
    frame["Y"] = np.where(frame["Y"].notna().to_numpy(), y, np.nan)
    frame["id"] = ids.astype(float)
    return frame, truth


def draw_survival(
    clusters: int,
    sizes: str,
    seed: int,
    *,
    latent: str = "rademacher",
    delta: float = DELTA,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One replication of the clustered survival law.

    Parameters
    ----------
    clusters : int
        The number of clusters.
    sizes : str
        ``"equal<m>"`` or ``"unequal40"``.
    seed : int
        The replication seed.
    latent : str
        One of :data:`LATENTS`.
    delta : float
        The scale of the cluster component.

    Returns
    -------
    frame : pandas.DataFrame
        The ``make_longitudinal_survival`` columns with the redrawn ``Y2`` and a float ``id``
        column.
    truth : dict of str to float
        ``make_longitudinal_survival``'s truth dictionary, unchanged.
    """
    ids, n = _layout(sizes, clusters, seed)
    frame, truth = make_longitudinal_survival(n=n, seed=seed, censoring=True, backend="pandas")
    a1 = frame["A1"].to_numpy().astype(int)
    hazard = _hazard_two(
        frame["W1"].to_numpy(),
        frame["W2"].to_numpy(),
        np.nan_to_num(frame["L2"].to_numpy()),
        frame["A1"].to_numpy(),
        np.nan_to_num(frame["A2"].to_numpy()),
    )
    u = cluster_latents(latent, clusters, seed)[ids, a1]
    uniforms = child_stream(seed, OUTCOME_STREAM).random(n)
    y2 = (uniforms < perturbed_probability(hazard, u, delta)).astype(float)
    at_risk = (frame["Y1"].to_numpy() == 0.0) & frame["Y2"].notna().to_numpy()
    frame["Y2"] = np.where(at_risk, y2, frame["Y2"].to_numpy())
    frame["id"] = ids.astype(float)
    return frame, truth
