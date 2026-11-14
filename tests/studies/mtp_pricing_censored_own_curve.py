r"""The own influence curve of the censored pricing fits (law P-C), in numpy.

The censored reading of :mod:`tests.studies.mtp_pricing_own_curve`.  Each node's joint ratio is
the policy's price ratio at the observed dose times ``1{C_t = 1}`` over the retention at the
observed history and dose (Díaz, Williams, Hoffman and Schenck 2023, arXiv v4, Section 3):

.. math::

    r_t = \frac{g^d_t(A_t \mid H_t)}{g_t(A_t \mid H_t)}
          \frac{\mathbf 1\{C_t = 1\}}{P(C_t = 1 \mid A_t, H_t)},
    \qquad \omega_t = \prod_{s \le t} r_s .

Node ``t``'s regression is fitted on the units still observed after ``C_t`` and predicted on the
units at risk at ``t`` (still observed after ``C_{t-1}``), as
``cleverly.longitudinal.sequential`` trains on ``masks.following(time)``.  ``LTMLE``'s curve is
:math:`D = Z_1 - \psi + \sum_t \omega_t (Z_{t+1} - Q^*_t(A_t, H_t))` and ``SDR``'s is
:math:`\varphi_1 - \psi` with :math:`\varphi_t = Q_t(d_t(A_t)) + r_t (\varphi_{t+1} - Q_t(A_t))`,
each term zero once a unit has left.

The censoring learner of every gated cell is a main-terms logistic regression, correctly
specified for law P-C (:func:`~tests.studies.mtp_pricing_censored_common.censoring_learner`), so
its limit is the true retention, which :class:`OwnCurve` reads by default.  The unit check passes
the fitted retention of a ``cleverly`` fit instead, so the two recursions meet to rounding
(``tests/unit/test_mtp_pricing_censored_design.py``).  This module imports nothing from
``cleverly``.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tests.studies import mtp_pricing_censored_common as law
from tests.studies import mtp_pricing_common as pricing_law
from tests.studies import mtp_pricing_own_curve as own
from tests.studies import mtp_tail_size_common as tail

Columns = Mapping[str, np.ndarray]
Retention = Callable[[Columns], list[np.ndarray]]


def columns(frame: pd.DataFrame, tau: int = law.TAU) -> dict[str, np.ndarray]:
    """The arrays the recursions read, each missing entry filled with a value on the support.

    A missing engagement reads 0 and a missing price the node's centre, so every map and ratio
    is finite; the masks :func:`alive` and :func:`at_risk` keep those rows out of every fit
    and every curve term.
    """
    out: dict[str, np.ndarray] = {"W1": np.asarray(frame["W1"], dtype=float)}
    for t in range(1, tau + 1):
        l = np.nan_to_num(np.asarray(frame[f"L{t}"], dtype=float))
        a = np.asarray(frame[f"A{t}"], dtype=float)
        out[f"L{t}"] = l
        out[f"A{t}"] = np.where(np.isnan(a), pricing_law.centre(l), a)
        out[f"C{t}"] = np.nan_to_num(np.asarray(frame[f"C{t}"], dtype=float))
    out["Y"] = np.nan_to_num(np.asarray(frame["Y"], dtype=float))
    out["R"] = np.nan_to_num(np.asarray(frame["R"], dtype=float))
    return out


def alive(cols: Columns, t: int) -> np.ndarray:
    """Still observed after ``C_t``; ``t = 0`` is everybody."""
    out = np.ones(cols["W1"].shape[0], dtype=bool)
    for s in range(1, t + 1):
        out &= cols[f"C{s}"] == 1.0
    return out


def at_risk(cols: Columns, t: int) -> np.ndarray:
    """At risk at node ``t``: still observed after ``C_{t-1}``."""
    return alive(cols, t - 1)


def true_retention(cols: Columns, tau: int = law.TAU) -> list[np.ndarray]:
    """Each node's ``P(C_t = 1 | A_t, H_t)`` at the observed dose, the logistic's limit."""
    return [
        np.asarray(
            law.retention(
                cols["W1"],
                cols[f"L{t}"],
                cols[f"A{t}"] - pricing_law.centre(cols[f"L{t}"]),
            ),
            dtype=float,
        )
        for t in range(1, tau + 1)
    ]


def joint_ratios(
    cols: Columns, label: str, tau: int, retention: list[np.ndarray]
) -> list[np.ndarray]:
    """Each node's joint ratio, zero once the unit has left."""
    policy = own.ratios(cols, label, tau)
    return [
        np.where(alive(cols, t), policy[t - 1] / retention[t - 1], 0.0) for t in range(1, tau + 1)
    ]


def _subset(cols: Columns, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[rows] for name, values in cols.items()}


@dataclass
class OwnCurve(own.OwnCurve):
    """The own curve of one estimator with one set of learners on law P-C.

    Parameters
    ----------
    tau : int
        Nodes.
    estimator : {"ltmle", "sdr"}
        The recursion.
    learners : {"primary", "property"}
        The declared learners.
    labels : tuple of str
        The plans; ``natural_prices`` first.
    retention : callable
        Maps the columns to each node's retention at the observed dose; the true one by
        default.
    """

    retention: Retention = field(default=true_retention)  # type: ignore[assignment]

    def _retention(self, cols: Columns) -> list[np.ndarray]:
        return list(self.retention(cols))

    def _fit_plan(self, cols: Columns, label: str, chunk: int) -> own.PlanFit:
        tau = self.tau
        r = joint_ratios(cols, label, tau, self._retention(cols))
        omega = np.cumprod(np.column_stack(r), axis=1)
        models: list[own.NodeModel] = [None] * tau  # type: ignore[list-item]
        flucts: list[own.Fluctuation | None] = [None] * tau
        z = np.where(alive(cols, tau), self.outcome(cols), 0.0)
        for t in range(tau, 0, -1):
            trained = alive(cols, t)
            dose = own.policy_dose(cols, label, t)

            def observed(rows: slice, t: int = t) -> np.ndarray:
                return own.outcome_design(_subset(cols, rows), t)  # type: ignore[arg-type]

            model = own.fit_node(
                own.node_learner(self.learners, t, tau, _subset(cols, trained)),
                observed,
                z,
                weight=trained.astype(float),
                chunk=chunk,
            )
            models[t - 1] = model
            q_obs = self._predict(model, cols, t, None, chunk)
            q_pol = self._predict(model, cols, t, dose, chunk)
            reached = at_risk(cols, t)
            if self.estimator == "ltmle":
                fluctuation = own.Fluctuation.fit(z, q_obs, omega[:, t - 1], self.alpha)
                flucts[t - 1] = fluctuation
                z = np.where(reached, fluctuation.apply(q_pol), 0.0)
            else:
                z = np.where(reached, q_pol + np.where(trained, r[t - 1] * (z - q_obs), 0.0), 0.0)
        plan = own.PlanFit(label, models, flucts)
        plan.psi = float(np.mean(z))
        return plan

    def curves(
        self, cols: Columns, known: Mapping[str, list[np.ndarray]] | None = None
    ) -> dict[str, np.ndarray]:
        """Each plan's curve ``D`` at the rows of ``cols``; ``known`` holds the joint ratios."""
        retention = None if known is not None else self._retention(cols)
        out = {}
        for label, plan in self.plans.items():
            r = (
                known[label]
                if known is not None
                else joint_ratios(cols, label, self.tau, retention)  # type: ignore[arg-type]
            )
            out[label] = self._curve(cols, plan, r)
        return out

    def _curve(
        self, cols: Columns, plan: own.PlanFit, known: list[np.ndarray] | None = None
    ) -> np.ndarray:
        tau = self.tau
        r = joint_ratios(cols, plan.label, tau, self._retention(cols)) if known is None else known
        z = np.where(alive(cols, tau), self.outcome(cols), 0.0)
        clip = self.estimator == "ltmle"

        def predict(model: own.NodeModel, t: int, dose: np.ndarray | None) -> np.ndarray:
            values = model.predict(own.outcome_design(cols, t, dose))
            return np.clip(values, 0.0, 1.0) if clip else values

        if self.estimator == "sdr":
            for t in range(tau, 0, -1):
                model = plan.models[t - 1]
                q_obs = predict(model, t, None)
                q_pol = predict(model, t, own.policy_dose(cols, plan.label, t))
                trained = alive(cols, t)
                z = np.where(
                    at_risk(cols, t),
                    q_pol + np.where(trained, r[t - 1] * (z - q_obs), 0.0),
                    0.0,
                )
            return z - plan.psi
        omega = np.cumprod(np.column_stack(r), axis=1)
        curve = np.zeros_like(z)
        for t in range(tau, 0, -1):
            fluctuation = plan.fluctuations[t - 1]
            assert fluctuation is not None
            model = plan.models[t - 1]
            q_obs = fluctuation.apply(predict(model, t, None))
            q_pol = fluctuation.apply(predict(model, t, own.policy_dose(cols, plan.label, t)))
            curve = curve + np.where(alive(cols, t), omega[:, t - 1] * (z - q_obs), 0.0)
            z = np.where(at_risk(cols, t), q_pol, 0.0)
        return z - plan.psi + curve


# ------------------------------------------------------------------ the tail size on law P-C

#: The fits whose own curves :func:`law_pc_moments` reads: ``(learners, estimator)``.  The
#: property learners' ``LTMLE`` curve with the exact ratio stands in for the efficient
#: influence function, as on law P; the ``SDR`` one is the proxy of the ``SDR`` calibration
#: cell.
LAW_PC_FITS: tuple[tuple[str, str], ...] = (
    ("primary", "ltmle"),
    ("primary", "sdr"),
    ("property", "ltmle"),
    ("property", "sdr"),
)
#: The plans whose curves the study reads.
LAW_PC_LABELS = ("natural_prices", "price_up")
#: The rows of the limit fit, the draws the moments read, and the rows of one chunk.
LAW_PC_FIT_ROWS = tail.LAW_P_FIT_ROWS
LAW_PC_DRAWS = tail.LAW_P_DRAWS
LAW_PC_CHUNK = tail.LAW_P_CHUNK
#: This study's own seed streams: the limit fit, the draws (plus the chunk), the bootstrap.
LAW_PC_FIT_SEED = 2026771300
LAW_PC_DRAW_SEED = 2026771400
LAW_PC_BOOTSTRAP_SEED = 2026771500


def law_pc_fit(learners: str, estimator: str, rows: int = LAW_PC_FIT_ROWS) -> OwnCurve:
    """The own curve's limit nuisances: one fit on ``rows`` draws of law P-C."""
    cols = columns(law.sample(rows, LAW_PC_FIT_SEED))
    return OwnCurve(law.TAU, estimator, learners, LAW_PC_LABELS).fit(cols)  # type: ignore[arg-type]


def law_pc_curves(
    *, rows: int = LAW_PC_FIT_ROWS, checkpoint: Path | None = None
) -> dict[tuple[str, str], OwnCurve]:
    """Every fit of :data:`LAW_PC_FITS`, kept in ``checkpoint`` when one is given."""
    curves: dict[tuple[str, str], OwnCurve] = {}
    for learners, estimator in LAW_PC_FITS:
        stored = None if checkpoint is None else checkpoint / f"pc_{learners}_{estimator}.pkl"
        if stored is not None and stored.exists():
            curves[learners, estimator] = pickle.loads(stored.read_bytes())
            continue
        curves[learners, estimator] = law_pc_fit(learners, estimator, rows)
        if stored is not None:
            stored.write_bytes(pickle.dumps(curves[learners, estimator]))
    return curves


def chunk_sums(
    curves: Mapping[tuple[str, str], OwnCurve],
    k: int,
    chunk: int = LAW_PC_CHUNK,
    checkpoint: Path | None = None,
) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    """Chunk ``k``'s power sums of every curve, read from or kept in ``checkpoint``."""
    stored = None if checkpoint is None else checkpoint / f"pc_chunk{k:04d}.npz"
    if stored is not None and stored.exists():
        saved = np.load(stored, allow_pickle=False)
        return [tuple(name.split("|")) for name in saved["names"]], saved["sums"]  # type: ignore[misc]
    cols = columns(law.sample(chunk, LAW_PC_DRAW_SEED + k))
    retention = true_retention(cols)
    known = {label: joint_ratios(cols, label, law.TAU, retention) for label in LAW_PC_LABELS}
    names, rows_of = [], []
    for (learners, estimator), curve in curves.items():
        for quantity, values in curve.quantities(cols, known).items():
            names.append((learners, estimator, quantity))
            rows_of.append(tail._power_sums(values))
    sums = np.array(rows_of)
    if stored is not None:
        np.savez(stored, names=np.array(["|".join(n) for n in names]), sums=sums)
    return names, sums


def law_pc_moments(
    *,
    draws: int = LAW_PC_DRAWS,
    rows: int = LAW_PC_FIT_ROWS,
    chunk: int = LAW_PC_CHUNK,
    checkpoint: Path | None = None,
) -> dict[tuple[str, str, str], tail.MonteCarloTail]:
    """The moments of each own curve of :data:`LAW_PC_FITS`, by Monte Carlo.

    :func:`tests.studies.mtp_tail_size_common.law_p_moments` on law P-C: the limit nuisances
    from one fit on ``rows`` draws, the curve on ``draws`` fresh draws in chunks, and
    ``kappa``'s 99% upper bound from a bootstrap of the chunks.

    Returns
    -------
    dict
        ``(learners, estimator, quantity) -> MonteCarloTail``.
    """
    curves = law_pc_curves(rows=rows, checkpoint=checkpoint)
    names: list[tuple[str, str, str]] = []
    parts: list[np.ndarray] = []
    for k in range(draws // chunk):
        names, sums = chunk_sums(curves, k, chunk, checkpoint)
        parts.append(sums)
    stacked = np.stack(parts)
    rng = np.random.default_rng(LAW_PC_BOOTSTRAP_SEED)
    counts = rng.multinomial(
        stacked.shape[0],
        np.full(stacked.shape[0], 1.0 / stacked.shape[0]),
        size=tail.LAW_P_BOOTSTRAP,
    )
    out = {}
    for j, name in enumerate(names):
        pooled = stacked[:, j, :].sum(axis=0)
        sd, skewness, kappa = tail._central(pooled)
        kappas = np.array([tail._central(row)[2] for row in counts @ stacked[:, j, :]])
        out[name] = tail.MonteCarloTail(
            draws=round(pooled[0]),
            sd=sd,
            skewness=skewness,
            kappa=kappa,
            kappa_lower=float(np.quantile(kappas, 1.0 - tail.LAW_P_LEVEL)),
            kappa_upper=float(np.quantile(kappas, tail.LAW_P_LEVEL)),
        )
    return out
