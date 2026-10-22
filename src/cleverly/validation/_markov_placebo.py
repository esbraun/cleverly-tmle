r"""The longitudinal placebo law: treatment trajectories drawn from a Markov kernel.

A whole-trajectory permutation gives a unit that is still at risk the ``nan`` treatment of a
donor who left earlier, and a permutation stratified by the follow-up pattern makes the
placebo depend on the event process. Neither keeps the exact null. This module draws each
unit's full trajectory from a covariate-free first-order kernel ``K`` fitted to the observed
treatments, independently of the panel and of the other units, and then masks it by the
observed at-risk pattern (X82 plan 5.1.2).

Given the data, the drawn trajectories are independent of every non-treatment variable
``O^-``, and the mask is a function of ``O^-``. So the placebo law factorizes as
``P(O^-) x K``. Each sequential regression is then free of the treatment history, every
policy has the value ``E[Q_1(W)]``, and every policy contrast is exactly 0.

The kernel
----------

========================  ========================================================
node                      draw
========================  ========================================================
1                         i.i.d. from the node-1 treatments of the units at risk
``t >= 2``, categorical   the node-``t`` treatment of a donor whose node-``t - 1``
previous node             treatment equals the drawn one: the empirical transition
                          frequency
``t >= 2``, continuous    the node-``t`` treatment of a donor whose node-``t - 1``
previous node             dose is nearest to the drawn one, ties drawn uniformly
held node                 the decision drawn at node 1
========================  ========================================================

A donor is a unit with a recorded treatment at both nodes. A categorical value that no donor
holds draws from the node-``t`` marginal of the units at risk. The donor is chosen without
reading the receiving unit's non-treatment variables, which is what the proof needs. A
treatment that is constant within every cluster draws one trajectory per cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray

__all__ = ["MarkovKernel", "placebo_known_mechanisms"]


def _cluster_constant(data: Any) -> bool:
    """Whether every node's recorded treatment is constant within each cluster."""
    if data.cluster is None:
        return False
    codes = np.asarray(data.cluster)
    for time in range(1, data.n_times + 1):
        column = np.asarray(data.treatment[:, time - 1], dtype=float)
        present = ~np.isnan(column)
        if not present.any():
            continue
        clusters = codes[present]
        values = column[present]
        order = np.argsort(clusters, kind="stable")
        clusters, values = clusters[order], values[order]
        starts = np.r_[True, clusters[1:] != clusters[:-1]]
        first = np.maximum.accumulate(np.where(starts, np.arange(values.size), 0))
        if not np.array_equal(values, values[first]):
            return False
    return True


@dataclass(frozen=True)
class MarkovKernel:
    """The covariate-free first-order kernel of one longitudinal treatment panel.

    Parameters
    ----------
    trajectories : ndarray
        ``(m, T)`` observed treatment of each sampling unit, ``nan`` where it is not at risk.
        A sampling unit is a row, or a cluster when the treatment is constant within clusters.
    categorical : tuple of bool
        Whether each node holds level codes rather than a dose.
    held : bool
        Whether the design holds one baseline decision over every node.
    cluster_codes : ndarray or None
        The sampling unit of each row when units are clusters, else ``None``.
    levels : tuple of int
        The number of level codes of each node, ``0`` at a continuous node.
    """

    trajectories: FloatArray
    categorical: tuple[bool, ...]
    held: bool
    cluster_codes: IntArray | None
    levels: tuple[int, ...]

    @classmethod
    def fit(cls, data: Any) -> MarkovKernel:
        """Fit the kernel to the observed treatments of ``data``.

        Parameters
        ----------
        data : LongitudinalData
            The panel. A node with a continuous component in a vector is not accepted; the
            caller refuses it first.

        Returns
        -------
        MarkovKernel
            The kernel.
        """
        treatment = np.asarray(data.treatment, dtype=float)
        categorical = tuple(
            data.node_kind(time) == "categorical" for time in range(1, data.n_times + 1)
        )
        levels = tuple(
            len(data.treatment_levels[time - 1]) if categorical[time - 1] else 0
            for time in range(1, data.n_times + 1)
        )
        if _cluster_constant(data):
            codes = np.asarray(data.cluster, dtype=np.int64)
            units = int(codes.max()) + 1
            trajectories = np.full((units, data.n_times), np.nan)
            for time in range(data.n_times):
                present = ~np.isnan(treatment[:, time])
                trajectories[codes[present], time] = treatment[present, time]
            return cls(trajectories, categorical, bool(data.treatment_held), codes, levels)
        return cls(treatment.copy(), categorical, bool(data.treatment_held), None, levels)

    @property
    def n_times(self) -> int:
        """The number of nodes."""
        return int(self.trajectories.shape[1])

    def _donors(self, time: int) -> tuple[FloatArray, FloatArray]:
        """The previous and the node-``time`` treatment of every donor, ``time >= 2``."""
        previous = self.trajectories[:, time - 2]
        current = self.trajectories[:, time - 1]
        both = ~np.isnan(previous) & ~np.isnan(current)
        return previous[both], current[both]

    def _marginal(self, time: int) -> FloatArray:
        column = self.trajectories[:, time - 1]
        return column[~np.isnan(column)]

    def draw(self, rng: np.random.Generator, size: int) -> FloatArray:
        """Draw ``size`` independent unmasked trajectories from the kernel.

        The draw reads only the kernel and ``rng``. It reads no unit's non-treatment
        variables, which is what makes the placebo law factorize.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator.
        size : int
            The number of trajectories.

        Returns
        -------
        ndarray
            ``(size, T)`` drawn treatments. A node with no unit at risk holds ``nan``.
        """
        drawn = np.full((size, self.n_times), np.nan)
        first = self._marginal(1)
        if first.size:
            drawn[:, 0] = rng.choice(first, size=size, replace=True)
        for time in range(2, self.n_times + 1):
            if self.held:
                drawn[:, time - 1] = drawn[:, 0]
                continue
            previous, current = self._donors(time)
            marginal = self._marginal(time)
            if not marginal.size:
                continue
            if not previous.size:
                drawn[:, time - 1] = rng.choice(marginal, size=size, replace=True)
                continue
            drawn[:, time - 1] = self._donor_draw(
                rng, drawn[:, time - 2], previous, current, marginal, self.categorical[time - 2]
            )
        return drawn

    @staticmethod
    def _donor_draw(
        rng: np.random.Generator,
        given: FloatArray,
        previous: FloatArray,
        current: FloatArray,
        marginal: FloatArray,
        categorical: bool,
    ) -> FloatArray:
        """Draw the next treatment of each value in ``given`` from a donor."""
        order = np.argsort(previous, kind="stable")
        sorted_current = current[order]
        values, starts, counts = np.unique(previous[order], return_index=True, return_counts=True)
        out = rng.choice(marginal, size=given.size, replace=True)
        known = ~np.isnan(given)
        position = np.searchsorted(values, np.where(known, given, values[0]))
        if categorical:
            inside = position < values.size
            clipped = np.minimum(position, values.size - 1)
            matched = known & inside & (values[clipped] == given)
            group = clipped
        else:
            lower = np.clip(position - 1, 0, values.size - 1)
            upper = np.clip(position, 0, values.size - 1)
            # The nearer of the two neighbours, and the lower one on a tie.
            closer_upper = np.abs(values[upper] - given) < np.abs(values[lower] - given)
            group = np.where(closer_upper, upper, lower)
            matched = known
        offsets = np.floor(rng.random(given.size) * counts[group]).astype(np.int64)
        picks = starts[group] + np.minimum(offsets, counts[group] - 1)
        out[matched] = sorted_current[picks[matched]]
        return out

    def node_law(self, time: int, previous: FloatArray) -> FloatArray:
        """The law of a categorical node ``time`` given each drawn previous treatment.

        Parameters
        ----------
        time : int
            A categorical node, counted from one.
        previous : ndarray
            One drawn node-``time - 1`` treatment per row. Ignored at node 1.

        Returns
        -------
        ndarray
            ``(len(previous), K_t)`` probabilities in level-code order.
        """
        if not self.categorical[time - 1]:
            raise ValueError(f"node {time} holds a dose and has no probability per level")
        levels = self.levels[time - 1]
        rows = np.asarray(previous, dtype=float).reshape(-1)

        def frequencies(values: FloatArray) -> FloatArray:
            counts = np.bincount(values.astype(np.int64), minlength=levels)[:levels]
            total = counts.sum()
            return counts / total if total else np.full(levels, np.nan)

        marginal = frequencies(self._marginal(time))
        if time == 1:
            return np.tile(marginal, (rows.size, 1))
        if self.held:
            out = np.zeros((rows.size, levels))
            valid = ~np.isnan(rows)
            out[valid, rows[valid].astype(np.int64)] = 1.0
            out[~valid] = np.nan
            return out
        donor_previous, donor_current = self._donors(time)
        if not donor_previous.size:
            return np.tile(marginal, (rows.size, 1))
        out = np.empty((rows.size, levels))
        cache: dict[float, FloatArray] = {}
        values = np.unique(donor_previous)
        for index, value in enumerate(rows):
            if np.isnan(value):
                out[index] = marginal
                continue
            if value not in cache:
                if self.categorical[time - 2]:
                    group = donor_previous == value
                    cache[value] = frequencies(donor_current[group]) if group.any() else marginal
                else:
                    nearest = values[np.argmin(np.abs(values - value))]
                    cache[value] = frequencies(donor_current[donor_previous == nearest])
            out[index] = cache[value]
        return out

    def trajectory_law(self) -> tuple[tuple[tuple[int, ...], float], ...]:
        """Every unmasked trajectory and its probability, on an all-categorical panel.

        Returns
        -------
        tuple
            ``(codes, probability)`` pairs over the product of the level sets.
        """
        if not all(self.categorical):
            raise ValueError("a dose node has no finite trajectory law")
        out: list[tuple[tuple[int, ...], float]] = []
        for codes in product(*(range(levels) for levels in self.levels)):
            probability = float(self.node_law(1, np.zeros(1))[0, codes[0]])
            for time in range(2, self.n_times + 1):
                law = self.node_law(time, np.asarray([float(codes[time - 2])]))
                probability *= float(law[0, codes[time - 1]])
            out.append((tuple(codes), probability))
        return tuple(out)

    def placebo(self, data: Any, rng: np.random.Generator) -> FloatArray:
        """Draw the masked placebo treatment matrix of ``data``.

        Parameters
        ----------
        data : LongitudinalData
            The panel the kernel was fitted to.
        rng : numpy.random.Generator
            The generator.

        Returns
        -------
        ndarray
            ``(n, T)``: the drawn trajectory of each row's sampling unit, ``nan`` exactly
            where the observed treatment is.
        """
        units = self.trajectories.shape[0]
        drawn = self.draw(rng, units)
        rows = drawn if self.cluster_codes is None else drawn[self.cluster_codes]
        observed = np.asarray(data.treatment, dtype=float)
        return np.where(np.isnan(observed), np.nan, rows)


def placebo_known_mechanisms(data: Any, kernel: MarkovKernel, placebo: FloatArray) -> Any:
    """The exact mechanism of the placebo law at every node ``data`` declares known.

    A fit that declared a node's treatment mechanism keeps that node declared on the placebo
    law, with the kernel's probability of each level given the drawn previous treatment. A
    refit that re-estimated it would be another configuration of the estimator. Censoring
    declarations are unchanged, because the placebo does not move censoring.

    Parameters
    ----------
    data : LongitudinalData
        The panel.
    kernel : MarkovKernel
        The kernel the placebo was drawn from.
    placebo : ndarray
        The masked placebo treatment.

    Returns
    -------
    KnownNodeMechanisms or None
        The declaration of the placebo law, or ``None`` when ``data`` declares none.
    """
    from dataclasses import replace

    known = data.known_mechanisms
    if known is None or not known.declares_treatment:
        return known
    nodes: list[FloatArray | None] = []
    for time in range(1, data.n_times + 1):
        declared = known.treatment_at(time)
        if declared is None:
            nodes.append(None)
            continue
        previous = placebo[:, time - 2] if time >= 2 else np.zeros(data.n)
        law = kernel.node_law(time, previous)
        law[np.isnan(placebo[:, time - 1])] = np.nan
        nodes.append(law)
    return replace(known, treatment=tuple(nodes), source="array", treatment_columns=())
