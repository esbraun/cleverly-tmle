"""The longitudinal plasmode of X82 part E4 (plan 5.6).

The law resamples the baseline rows of a fit and draws the rest of each history from models
frozen on the fitted panel. At node ``t`` each time-varying column ``L_t`` comes from a GLM on
``(W, L_1, ..., L_{t-1}, the earlier columns of L_t, A_1, ..., A_{t-1})``: logistic for a
column whose values are 0 and 1, and Gaussian linear for any other column. The treatment
``A_t`` comes from a logistic model on ``(W, L_1, ..., L_t, A_1, ..., A_{t-1})``, clipped to
the fit's ``g`` bounds, and the censoring after node ``t`` from a logistic model on
``(W, L_1, ..., L_t, A_1, ..., A_t)``. Each model is fitted on the units present at the node.
The outcome is a declared Gaussian process that reads ``W``, the treatments and, through
``history_effect``, the time-varying columns. On a survival or competing-risk fit, a declared
hazard draws the event at each node after the censoring, and reads the history through its own
``history_effect``.

The truth is the sequential g-formula of that law under the declared regimen, with no
censoring. When every time-varying column is binary and the branches number at most
``MAX_BRANCHES`` per support point, it enumerates every branch of every history, so the truth
is exact. Otherwise it averages ``N`` simulated histories per support point and reports the
Monte Carlo standard error. A survival truth integrates the event out: each history carries its
survival probability, so only ``L`` and ``A`` branch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from .outcome_processes import GaussianAdjustmentOutcome, standardized_sum

#: The largest number of histories per support point that the truth enumerates. A law with
#: more possible branches gets a Monte Carlo truth.
MAX_BRANCHES = 4096

#: The Monte Carlo standard error the truth may carry, as a share of the fit's standard error.
MONTE_CARLO_SHARE = 0.02

#: The histories per support point of the pilot that sizes a Monte Carlo truth.
PILOT_DRAWS = 64

#: The largest number of simulated histories per support point.
MAX_DRAWS = 2048


@dataclass(frozen=True)
class _Logistic:
    """A frozen logistic model; ``constant`` holds the rate of a column with one class."""

    coef: np.ndarray
    intercept: float
    constant: float | None = None

    @classmethod
    def fit(cls, design: np.ndarray, labels: np.ndarray) -> _Logistic:
        values = np.asarray(labels, dtype=float)
        rate = float(np.mean(values)) if values.size else 0.0
        if values.size == 0 or rate in (0.0, 1.0):
            return cls(np.zeros(design.shape[1]), 0.0, constant=rate)
        model = LogisticRegression(C=1e4, max_iter=5000).fit(design, values.astype(int))
        return cls(np.asarray(model.coef_[0], dtype=float), float(model.intercept_[0]))

    def probability(self, design: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(design.shape[0], self.constant)
        return np.asarray(1.0 / (1.0 + np.exp(-(design @ self.coef + self.intercept))))


@dataclass(frozen=True)
class _Linear:
    """A frozen Gaussian linear model with its residual standard deviation."""

    coef: np.ndarray
    intercept: float
    sd: float

    @classmethod
    def fit(cls, design: np.ndarray, values: np.ndarray) -> _Linear:
        model = LinearRegression().fit(design, values)
        residual = values - model.predict(design)
        dof = max(values.size - design.shape[1] - 1, 1)
        sd = float(np.sqrt(np.sum(residual**2) / dof))
        return cls(np.asarray(model.coef_, dtype=float), float(model.intercept_), sd)

    def mean(self, design: np.ndarray) -> np.ndarray:
        return np.asarray(design @ self.coef + self.intercept, dtype=float)


def _binary(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size) and bool(np.all((finite == 0.0) | (finite == 1.0)))


@dataclass(frozen=True, eq=False)
class LongitudinalPlasmodeLaw:
    """The frozen law of a longitudinal plasmode.

    Parameters
    ----------
    support : LongitudinalData
        The fitted panel. Its baseline rows are the support points.
    covariates : tuple of tuple
        Per node, one frozen model per time-varying column, in column order.
    treatment : tuple of _Logistic
        Per node, the frozen model of ``P(A_t = 1 | H_t)``.
    censoring : tuple of _Logistic or None
        Per node, the frozen model of ``P(uncensored after t | H_t, A_t)``, or ``None``
        when the fit declares no censoring.
    lower : float
        The fit's lower ``g`` bound. A drawn propensity lies in ``[lower, 1 - lower]``.
    centre : ndarray
        The column centres of ``W`` on the support.
    scale : ndarray
        The column scales of ``W`` on the support.
    history_centre : ndarray
        The centre of each time-varying column, over the rows that hold it.
    history_scale : ndarray
        The scale of each time-varying column, over the rows that hold it.
    """

    support: Any
    covariates: tuple[tuple[_Logistic | _Linear, ...], ...]
    treatment: tuple[_Logistic, ...]
    censoring: tuple[_Logistic | None, ...]
    lower: float
    centre: np.ndarray
    scale: np.ndarray
    history_centre: np.ndarray
    history_scale: np.ndarray

    @classmethod
    def fit(cls, result: Any) -> LongitudinalPlasmodeLaw:
        """Freeze the law on the fitted panel of ``result``.

        Parameters
        ----------
        result : LongitudinalResult
            The fit.

        Returns
        -------
        LongitudinalPlasmodeLaw
            The frozen law.
        """
        data = result.data
        nodes = int(data.n_times)
        baseline = np.asarray(data.baseline, dtype=float)
        blocks = [np.asarray(block, dtype=float) for block in data.time_varying]
        treatment = np.asarray(data.treatment, dtype=float)
        uncensored = np.asarray(data.uncensored, dtype=bool)
        covariates: list[tuple[_Logistic | _Linear, ...]] = []
        mechanisms: list[_Logistic] = []
        censoring: list[_Logistic | None] = []
        for node in range(nodes):
            # The units present at the node: uncensored, and with no event, before it.
            rows = ~np.isnan(treatment[:, node])
            earlier = [baseline[rows], *(block[rows] for block in blocks[:node])]
            earlier.append(treatment[rows, :node])
            models: list[_Logistic | _Linear] = []
            for column in range(blocks[node].shape[1]):
                design = np.column_stack([*earlier, blocks[node][rows, :column]])
                values = blocks[node][rows, column]
                if _binary(values):
                    models.append(_Logistic.fit(design, values))
                else:
                    models.append(_Linear.fit(design, values))
            covariates.append(tuple(models))
            history = np.column_stack([*earlier, blocks[node][rows]])
            mechanisms.append(_Logistic.fit(history, treatment[rows, node]))
            if data.censoring_names:
                acted = np.column_stack([history, treatment[rows, node]])
                censoring.append(_Logistic.fit(acted, uncensored[rows, node].astype(float)))
            else:
                censoring.append(None)
        flat = np.column_stack(blocks) if any(b.shape[1] for b in blocks) else np.zeros((data.n, 0))
        bounds = getattr(result.estimator, "g_bounds", None)
        lower = 0.0 if bounds is None else float(bounds[0])
        return cls(
            support=data,
            covariates=tuple(covariates),
            treatment=tuple(mechanisms),
            censoring=tuple(censoring),
            lower=lower,
            centre=np.mean(baseline, axis=0),
            scale=np.std(baseline, axis=0),
            history_centre=np.nanmean(flat, axis=0) if flat.shape[1] else np.zeros(0),
            history_scale=np.nanstd(flat, axis=0) if flat.shape[1] else np.zeros(0),
        )

    @property
    def binary(self) -> bool:
        """Return whether every time-varying column is binary."""
        return all(isinstance(model, _Logistic) for node in self.covariates for model in node)

    @property
    def enumerable(self) -> bool:
        """Return whether the truth can enumerate every branch within :data:`MAX_BRANCHES`.

        The count is the bound before any zero-weight branch is pruned: two levels for each
        time-varying column and each treatment node.
        """
        columns = sum(len(node) for node in self.covariates) + len(self.treatment)
        return self.binary and 2**columns <= MAX_BRANCHES

    # ---------------------------------------------------------------- the outcome

    def outcome_mean(
        self, process: Any, rows: np.ndarray, blocks: list[np.ndarray], treatment: np.ndarray
    ) -> np.ndarray:
        """Return the declared outcome mean of each history.

        Parameters
        ----------
        process : GaussianIndependentOutcome or GaussianAdjustmentOutcome
            The declared process.
        rows : ndarray
            The support row of each history.
        blocks : list of ndarray
            The time-varying columns of each history, one block per node.
        treatment : ndarray
            ``(m, T)`` treatment codes of each history.

        Returns
        -------
        ndarray
            ``mu + f(W) + effect * sum_t A_t + history_effect * h(L)`` per history.
        """
        mean = np.full(rows.size, float(process.noise.mean))
        if not isinstance(process, GaussianAdjustmentOutcome):
            return mean
        baseline = np.asarray(self.support.baseline, dtype=float)[rows]
        mean += process.adjustment(baseline, centre=self.centre, scale=self.scale)
        mean += float(process.effect) * np.sum(treatment, axis=1)
        history_effect = float(getattr(process, "history_effect", 0.0))
        if history_effect and self.history_centre.size:
            flat = np.column_stack(blocks)
            mean += standardized_sum(
                flat, history_effect, centre=self.history_centre, scale=self.history_scale
            )
        return mean

    def hazard_shift(self, process: Any, rows: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
        """Return ``f(W) + history_effect * h_t`` of a hazard at the node after ``blocks``.

        Parameters
        ----------
        process : LogisticHazardOutcome or MultinomialHazardOutcome
            The declared hazard.
        rows : ndarray
            The support row of each history.
        blocks : list of ndarray
            The time-varying columns of nodes 1 to ``t``.

        Returns
        -------
        ndarray
            The shift of the log-odds of each history at node ``t``.
        """
        baseline = np.asarray(self.support.baseline, dtype=float)[rows]
        shift = standardized_sum(
            baseline, process.adjustment_scale, centre=self.centre, scale=self.scale
        )
        history_effect = float(process.history_effect)
        columns = sum(block.shape[1] for block in blocks)
        if history_effect and columns:
            shift = shift + standardized_sum(
                np.column_stack(blocks),
                history_effect,
                centre=self.history_centre[:columns],
                scale=self.history_scale[:columns],
            )
        return shift

    # ---------------------------------------------------------------- the draw

    def draw(self, process: Any, *, seed: int, name: str) -> Any:
        """Draw one panel of the natural course.

        Parameters
        ----------
        process : object
            The declared outcome process, or the declared hazard of a survival fit.
        seed : int
            The child seed of this draw.
        name : str
            The test name. The draw names nothing.

        Returns
        -------
        LongitudinalData
            A panel of ``n`` resampled units with drawn histories and outcomes.
        """
        rng = np.random.default_rng(seed)
        data = self.support
        rows = np.asarray(rng.integers(0, data.n, size=data.n))
        nodes = len(self.treatment)
        blocks: list[np.ndarray] = []
        treatment = np.zeros((rows.size, nodes))
        stays = np.ones((rows.size, nodes), dtype=bool)
        hits = np.zeros((rows.size, nodes), dtype=int)
        survival = bool(data.is_survival)
        causes = int(process.causes) if survival else 0
        baseline = np.asarray(data.baseline, dtype=float)[rows]
        # Every unit draws a whole history; the exits mask it afterwards.
        for node in range(nodes):
            block = self._draw_block(node, baseline, blocks, treatment[:, :node], rng)
            blocks.append(block)
            history = np.column_stack([baseline, *blocks, treatment[:, :node]])
            propensity = self._propensity(node, history)
            treatment[:, node] = (rng.random(rows.size) < propensity).astype(float)
            model = self.censoring[node]
            if model is not None:
                acted = np.column_stack([history, treatment[:, node]])
                stays[:, node] = rng.random(rows.size) < model.probability(acted)
            if survival:
                dose = np.sum(treatment[:, : node + 1], axis=1)
                probabilities = process.probabilities(
                    self.hazard_shift(process, rows, blocks), dose
                )
                edges = np.cumsum(probabilities, axis=1)
                hits[:, node] = np.sum(rng.random(rows.size)[:, None] >= edges, axis=1)
        base = replace(data.subset(rows), time_varying=tuple(blocks), treatment=treatment)
        kept = np.logical_and.accumulate(stays, axis=1)
        if survival:
            return _with_drawn_events(base, kept, hits, causes)
        mean = self.outcome_mean(process, rows, blocks, treatment)
        outcome = mean + process.noise.draw(rng, rows.size) - float(process.noise.mean)
        # A unit censored before a node holds no covariate or treatment there.
        reached = np.column_stack([np.ones(rows.size, dtype=bool), kept[:, :-1]])
        held = [np.where(reached[:, [node]], block, np.nan) for node, block in enumerate(blocks)]
        return replace(
            base,
            time_varying=tuple(held),
            treatment=np.where(reached, treatment, np.nan),
            uncensored=kept,
            outcome=np.where(kept[:, -1], outcome, np.nan),
        )

    def _draw_block(
        self,
        node: int,
        baseline: np.ndarray,
        blocks: list[np.ndarray],
        treatment: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        columns: list[np.ndarray] = []
        for model in self.covariates[node]:
            design = np.column_stack([baseline, *blocks, treatment, *columns])
            if isinstance(model, _Logistic):
                columns.append((rng.random(baseline.shape[0]) < model.probability(design)) * 1.0)
            else:
                noise = rng.normal(size=baseline.shape[0])
                columns.append(model.mean(design) + model.sd * noise)
        return np.column_stack(columns) if columns else np.zeros((baseline.shape[0], 0))

    def _propensity(self, node: int, history: np.ndarray) -> np.ndarray:
        probability = self.treatment[node].probability(history)
        return np.clip(probability, self.lower, 1.0 - self.lower)

    # ---------------------------------------------------------------- the truth

    def truth(
        self,
        process: Any,
        regimen: Any,
        *,
        draws: int | None,
        seed: int = 0,
        horizon: int | None = None,
        cause: int = 0,
    ) -> tuple[float, float]:
        """Return the g-formula value of ``regimen`` and its Monte Carlo standard error.

        Parameters
        ----------
        process : object
            The declared outcome process, or the declared hazard of a survival fit.
        regimen : Regimen or DynamicRegimen
            The regimen, evaluated on each history by the package's plan resolver.
        draws : int or None
            Histories per support point, or ``None`` to enumerate every branch.
        seed : int
            The seed of a Monte Carlo truth.
        horizon : int or None
            The last node of a survival risk. ``None`` reads every node.
        cause : int
            The column of the cause whose risk a competing-risk truth reads.

        Returns
        -------
        tuple of float
            The value, and its Monte Carlo standard error (0 when enumerated).
        """
        data = self.support
        if draws is None and not self.enumerable:
            raise ValueError("this law has more branches than the truth enumerates")
        rng = np.random.default_rng(seed)
        survival = bool(data.is_survival)
        count = 1 if draws is None else int(draws)
        nodes = len(self.treatment) if horizon is None else int(horizon)
        rows = np.repeat(np.arange(data.n), count)
        weight = np.full(rows.size, 1.0 / count)
        # The event of a survival law is integrated out: each history carries its survival
        # probability and its risk of the cause so far.
        alive = np.ones(rows.size)
        risk = np.zeros(rows.size)
        baseline = np.asarray(data.baseline, dtype=float)
        blocks: list[np.ndarray] = []
        treatment = np.zeros((rows.size, 0))
        exact = draws is None
        for node in range(nodes):
            block_columns: list[np.ndarray] = []
            for model in self.covariates[node]:
                design = np.column_stack([baseline[rows], *blocks, treatment, *block_columns])
                if isinstance(model, _Logistic) and exact:
                    p = model.probability(design)
                    index = np.concatenate([np.arange(rows.size)] * 2)
                    level = np.concatenate([np.ones(rows.size), np.zeros(rows.size)])
                    weight = np.concatenate([weight * p, weight * (1.0 - p)])
                    rows, blocks, treatment, block_columns = _expand(
                        index, rows, blocks, treatment, [*block_columns, alive, risk]
                    )
                    *block_columns, alive, risk = block_columns
                    block_columns.append(level)
                elif isinstance(model, _Logistic):
                    block_columns.append((rng.random(rows.size) < model.probability(design)) * 1.0)
                else:
                    noise = rng.normal(size=rows.size)
                    block_columns.append(model.mean(design) + model.sd * noise)
            blocks.append(
                np.column_stack(block_columns) if block_columns else np.zeros((rows.size, 0))
            )
            density = _policy_density(self, regimen, node, rows, blocks, treatment)
            if exact:
                index = np.concatenate([np.arange(rows.size)] * 2)
                code = np.concatenate([np.ones(rows.size), np.zeros(rows.size)])
                weight = np.concatenate([weight * density, weight * (1.0 - density)])
                rows, blocks, treatment, (alive, risk) = _expand(
                    index, rows, blocks, treatment, [alive, risk]
                )
            else:
                code = (rng.random(rows.size) < density) * 1.0
            treatment = np.column_stack([treatment, code])
            if survival:
                shift = self.hazard_shift(process, rows, blocks)
                probabilities = process.probabilities(shift, np.sum(treatment, axis=1))
                risk = risk + alive * probabilities[:, cause]
                alive = alive * (1.0 - probabilities.sum(axis=1))
            keep = weight > 0.0
            rows, weight, treatment = rows[keep], weight[keep], treatment[keep]
            blocks = [block[keep] for block in blocks]
            alive, risk = alive[keep], risk[keep]
        value = risk if survival else self.outcome_mean(process, rows, blocks, treatment)
        per_point = np.bincount(rows, weights=weight * value, minlength=data.n)
        mean = float(np.mean(per_point))
        if exact:
            return mean, 0.0
        # Each support point's mean has variance s_i^2 / N; the truth averages n of them.
        square = np.bincount(rows, weights=weight * value**2, minlength=data.n)
        within = np.maximum(square - per_point**2, 0.0) * count / max(count - 1, 1)
        return mean, float(np.sqrt(np.sum(within) / count) / data.n)


def _with_drawn_events(base: Any, kept: np.ndarray, hits: np.ndarray, causes: int) -> Any:
    """The drawn panel with each unit's first observed event, and its exits masked.

    ``hits[i, t]`` is the drawn cause index at node ``t``, or ``causes`` for no event. An event
    counts at the first node at which the unit is still at risk and stays uncensored.
    """
    n, nodes = kept.shape
    event = np.zeros((n, nodes), dtype=bool)
    cause_event = np.zeros((n, nodes, max(causes, 1)), dtype=bool)
    at_risk = np.ones(n, dtype=bool)
    for node in range(nodes):
        hit = at_risk & kept[:, node] & (hits[:, node] < causes)
        for index in range(causes):
            cause_event[hit & (hits[:, node] == index), node:, index] = True
        event[hit, node:] = True
        at_risk &= kept[:, node] & ~hit
    competing = base.cause_event is not None
    return base.with_events(event, kept, cause_event if competing else None)


def _expand(
    index: np.ndarray,
    rows: np.ndarray,
    blocks: list[np.ndarray],
    treatment: np.ndarray,
    columns: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    return (
        rows[index],
        [block[index] for block in blocks],
        treatment[index],
        [column[index] for column in columns],
    )


def _policy_density(
    law: LongitudinalPlasmodeLaw,
    regimen: Any,
    node: int,
    rows: np.ndarray,
    blocks: list[np.ndarray],
    treatment: np.ndarray,
) -> np.ndarray:
    """Return ``P(A_t = 1)`` under the regimen at each simulated history.

    A label, a rule and a stochastic node read the plan that the package resolves on the
    simulated histories.
    """
    from ..longitudinal.regimen import resolve_plans

    time = node + 1
    (plan,) = resolve_plans((regimen,), _histories(law, rows, blocks, treatment))
    if plan.is_policy_node(time):
        return np.asarray(plan.policy_at(time)[:, 1], dtype=float)
    return np.asarray(plan.values[:, node] == 1.0, dtype=float)


def _histories(
    law: LongitudinalPlasmodeLaw,
    rows: np.ndarray,
    blocks: list[np.ndarray],
    treatment: np.ndarray,
) -> Any:
    """A panel of the simulated histories, which the plan resolver reads node by node.

    The nodes after the current one hold zeros. A rule reads its own node's history only,
    and the truth reads the current node's column of the plan alone.
    """
    data = law.support
    nodes = int(data.n_times)
    done = len(blocks)
    filled = list(blocks)
    for node in range(done, nodes):
        filled.append(np.zeros((rows.size, data.time_varying[node].shape[1])))
    codes = np.zeros((rows.size, nodes))
    codes[:, : treatment.shape[1]] = treatment
    base = data.subset(rows)
    # No simulated history has had an event: the truth integrates the event out.
    events = {}
    if base.event is not None:
        events["event"] = np.zeros_like(base.event)
    if base.cause_event is not None:
        events["cause_event"] = np.zeros_like(base.cause_event)
    return replace(
        base,
        time_varying=tuple(filled),
        treatment=codes,
        uncensored=np.ones((rows.size, nodes), dtype=bool),
        outcome=np.zeros(rows.size),
        **events,
    )
