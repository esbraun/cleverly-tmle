"""Law P-C: the pricing law of ``mtp-pricing`` with dropout after each price node.

**Law P-C.**  Law P at ``tau = 3`` (:mod:`tests.studies.mtp_pricing_common`) and, after each
price ``A_t``, a retention indicator

.. code-block:: text

    C_t ~ Bernoulli(expit(ALPHA + 0.5 W1 + 0.8 L_t - 0.6 (A_t - c_t))),   c_t = 6 + 2 L_t

with ``C_t = 1`` while the customer stays.  Dropout is monotone: after ``C_t = 0`` every later
``L``, ``A``, ``C``, ``Y`` and ``R`` is missing.  Retention rises with engagement and falls with
the price gap, so the customers who stay are the likelier buyers: the dropout is informative
given the history and the current price, and at random given them (Díaz, Williams, Hoffman and
Schenck 2023, arXiv v4, Section 3, the paragraph after Theorem 1, with the censoring component of
``A_t`` set to one).

The censoring draw changes no structural equation of ``W1``, ``L``, ``A`` or ``Y``: :func:`sample`
draws law P's record from ``seed`` exactly as :func:`~tests.studies.mtp_pricing_common.sample`
does and then the retention from its own stream, ``(seed, CENSORING_STREAM)``.  So the parameter
of every plan with nobody censored is law P's
(:data:`~tests.studies.mtp_pricing_common.TRUTHS`), and the design test checks both facts.

Every per-node retention is at least ``expit(ALPHA - 1.25 - 1.2)`` on the support (``W1`` in
``[-2.5, 2.5]``, ``L_t`` in ``{0, 1}``, the gap in ``[-2, 2]``).  The retention's logit is linear in
``W1``, ``L_t`` and ``A_t``, so a main-terms logistic regression on the node's history design,
which holds those columns, is correctly specified (:func:`censoring_learner`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from tests.studies import mtp_pricing_common as common

TAU = 3
#: The retention intercept.  A 60-replication pilot at n = 2,000 read 13.6% censored by node 3.
ALPHA = 2.8
BETA_W = 0.5
BETA_L = 0.8
BETA_GAP = -0.6
#: The entropy word of the censoring stream, appended to a sample's seed.
CENSORING_STREAM = 77_0003


def retention(w1: Any, l: Any, gap: Any) -> Any:
    """``P(C_t = 1)`` given ``W1``, the node's engagement and its price gap."""
    return expit(
        ALPHA
        + BETA_W * np.asarray(w1, dtype=float)
        + BETA_L * np.asarray(l, dtype=float)
        + BETA_GAP * np.asarray(gap, dtype=float)
    )


def minimum_retention() -> float:
    """The smallest per-node retention on the support."""
    return float(retention(-common.W_BOUND, 0.0, 2.0))


def censoring_columns(tau: int = TAU) -> list[str]:
    return [f"C{t}" for t in range(1, tau + 1)]


def censor(frame: pd.DataFrame, seed: int, tau: int = TAU) -> pd.DataFrame:
    """Law P's ``frame`` with each node's retention drawn and the later columns removed."""
    rng = np.random.default_rng([seed, CENSORING_STREAM])
    out = frame.copy()
    n = len(out)
    alive = np.ones(n, dtype=bool)
    w1 = out["W1"].to_numpy(dtype=float)
    for t in range(1, tau + 1):
        l = out[f"L{t}"].to_numpy(dtype=float)
        gap = out[f"A{t}"].to_numpy(dtype=float) - common.centre(l)
        stays = rng.random(n) < retention(w1, np.nan_to_num(l), np.nan_to_num(gap))
        out[f"C{t}"] = np.where(alive, stays.astype(float), np.nan)
        alive &= stays
        later = [
            *(f"{v}{s}" for s in range(t + 1, tau + 1) for v in ("L", "A")),
            "Y",
            "R",
        ]
        out.loc[~alive, later] = np.nan
    ordered = ["W1"]
    for t in range(1, tau + 1):
        ordered += [f"L{t}", f"A{t}", f"C{t}"]
    return out.loc[:, [*ordered, "Y", "R"]]


def sample(n: int, seed: int, tau: int = TAU) -> pd.DataFrame:
    """``n`` rows of law P-C: law P's record from ``seed``, then the dropout."""
    return censor(common.sample(n, seed, tau), seed, tau)


def roles(tau: int = TAU) -> dict[str, Any]:
    """The node roles of law P-C."""
    return {
        "treatment": [f"A{t}" for t in range(1, tau + 1)],
        "baseline": ["W1"],
        "time_varying": [[f"L{t}"] for t in range(1, tau + 1)],
        "censoring": censoring_columns(tau),
        "continuous_treatment": {f"A{t}": common.support(t) for t in range(1, tau + 1)},
    }


def censoring_learner() -> LogisticRegression:
    """The declared censoring learner: main-terms logistic, unpenalized in effect."""
    return LogisticRegression(C=1e6, max_iter=2_000)


def ipcw_natural_mean(frame: pd.DataFrame, tau: int = TAU) -> float:
    """The mean of ``Y`` under the natural course, weighted by the true inverse retention.

    The design test's check that the dropout leaves law P's parameter.
    """
    weight = np.ones(len(frame))
    stayed = np.ones(len(frame), dtype=bool)
    w1 = frame["W1"].to_numpy(dtype=float)
    for t in range(1, tau + 1):
        c = frame[f"C{t}"].to_numpy(dtype=float)
        stayed &= c == 1.0
        l = np.nan_to_num(frame[f"L{t}"].to_numpy(dtype=float))
        gap = np.nan_to_num(frame[f"A{t}"].to_numpy(dtype=float)) - common.centre(l)
        weight = np.where(stayed, weight / retention(w1, l, gap), 0.0)
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    return float(np.mean(weight * y))
