"""Censored continuous nodes of a longitudinal modified treatment policy (X77).

The oracle is the censored reading of the unit law in :mod:`tests.discrete_law_longitudinal_mtp`
(``censored_frame``): a censoring indicator after each of the two continuous nodes, whose
retention reads the node's own dose.  The parameter is Theorem 1 of Díaz, Williams, Hoffman and
Schenck (2023) with each censoring component set to one, and the joint ratio of a node is the
policy's dose ratio times ``1{C_t = 1} / P(C_t = 1 | A_t, H_t)`` at the **observed** dose.  A
saturated learner on rows that realise the law exactly makes every comparison exact, so each
engine's estimate and influence curve must equal the oracle to rounding.

Each mutation control breaks one line through ``monkeypatch`` and asks the exact witness to
move by more than :data:`CAUGHT`.  The retention reads the current dose, so every censoring
component the controls guard is nonzero at the truth: the observed and the shifted dose give
different retention, and the dose law among the units that stayed after a node differs from the
one among the units at risk.  The measured deviations, largest over the estimates and the curves, are in each docstring.
"""

from __future__ import annotations

import contextlib
import importlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.longitudinal import LTMLE, SDR, sequential
from cleverly.longitudinal import data as data_module
from cleverly.longitudinal.data import LongitudinalData

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_mtp as law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import longitudinal_mtp as mtp
from ._source_mutation import mutate

sdr_module = importlib.import_module("cleverly.longitudinal.sdr")

# The grid law records a point mass at every dose, which R-b4 refuses; the fixture turns
# that refusal off and leaves the ratio code as shipped.
pytestmark = [
    pytest.mark.xdist_group("longitudinal_mtp_censored"),
    pytest.mark.usefixtures("exact_grid_law"),
]

#: Plans that move the dose at node 1, at node 2, both, by the intervened history, and at random.
LABELS = ("natural", "up", "history", "up at 2", "random")
EXACT = 1e-9
CAUGHT = 1e-4
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
    "continuous_treatment": ["A1", "A2"],
}
ENGINES = {"ltmle": LTMLE, "sdr": SDR}


def _settings(ratio: str, n_folds: int) -> dict[str, Any]:
    return {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": multivalue.CellProbabilities(),
        "censoring_learner": binary_law.CellMeans(),
        "n_folds": n_folds,
        "g_bounds": (1e-8, 1.0),
        "ratio": ratio,
    }


def fit(
    engine: str = "ltmle",
    ratio: str = "classifier",
    *,
    copies: int = 1,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> Any:
    """One engine on ``copies`` copies of the censored law, the copies as folds."""
    from cleverly.learners import Folds

    frame = pd.concat([law.censored_frame()] * copies, ignore_index=True)
    cls = ENGINES[engine]
    if copies > 1:
        assert monkeypatch is not None
        assignment = np.repeat(np.arange(copies), len(frame) // copies)
        monkeypatch.setattr(cls, "_folds", lambda self, data, seed=None: Folds(assignment, copies))
    settings = _settings(ratio, copies)
    if engine == "ltmle":
        settings["simultaneous"] = False
    exact = mtp.exact_bins() if ratio == "density" else contextlib.nullcontext()
    with warnings.catch_warnings(), exact:
        warnings.simplefilter("ignore")
        return cls(mtp.regimens(LABELS), reference=LABELS[0], **settings).fit(frame, **COLUMNS)


def deviation(result: Any, copies: int = 1) -> float:
    """The largest error of any mean, contrast or curve against the censored oracle."""
    size = len(law.censored_frame())
    rows = law.censored_first_row_of()
    names = [f"ey_regimen[{label}]" for label in LABELS] + [
        f"ate_regimen[{label} vs {LABELS[0]}]" for label in LABELS[1:]
    ]
    worst = 0.0
    for name in names:
        worst = max(worst, abs(result.psi(name) - law.censored_truth(name)))
        eif = law.censored_eif(name)
        for copy in range(copies):
            curve = result.influence_curves[name][copy * size + rows]
            worst = max(worst, float(np.max(np.abs(curve - eif))))
    return worst


# ------------------------------------------------------------------ the law


def test_the_censored_law_keeps_the_complete_law_parameter_and_censors_at_both_nodes() -> None:
    """Censoring at random leaves Theorem 1's parameter; both nodes censor on the dose."""
    for label in LABELS:
        name = f"ey_regimen[{label}]"
        assert law.censored_truth(name) == pytest.approx(law.truth(name), abs=1e-14)
    frame = law.censored_frame()
    assert (frame["C1"] == 0).sum() > 0
    assert (frame["C2"] == 0).sum() > 0
    # The retention moves with the dose at each node, so the censoring factor at the observed
    # dose and at the shifted dose differ (the witness of M-a and M-b).
    assert np.ptp(law.R1, axis=1).min() > 0
    assert len({law._r2(0, 0, 0, a2) for a2 in range(4)}) > 1
    # The plans move the dose at each node.
    for label in ("up", "up at 2"):
        assert law.censored_truth(f"ey_regimen[{label}]") != pytest.approx(
            law.censored_truth("ey_regimen[natural]"), abs=1e-3
        )


# ------------------------------------------------------------------ exactness


@pytest.mark.parametrize("ratio", ["classifier", "density"])
@pytest.mark.parametrize("engine", ["ltmle", "sdr"])
def test_a_censored_continuous_fit_equals_the_g_formula_and_its_gateaux_derivative(
    engine: str, ratio: str
) -> None:
    """In sample, both engines and both ratio routes."""
    result = fit(engine, ratio)
    assert [result.data.node_kind(time) for time in (1, 2)] == ["continuous", "continuous"]
    assert deviation(result) < EXACT


@pytest.mark.parametrize("engine", ["ltmle", "sdr"])
def test_a_cross_fitted_censored_fit_is_exact_when_each_fold_is_the_law(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two folds, each one exact copy: the out-of-fold censoring factor and ratio are exact."""
    result = fit(engine, copies=2, monkeypatch=monkeypatch)
    assert deviation(result, copies=2) < EXACT


# ------------------------------------------------------------------ mutation controls


def _caught(engine: str) -> float:
    gap = deviation(fit(engine))
    assert gap > CAUGHT
    return gap


def _shifted_values(plan: Any, data: Any) -> Any:
    """The plan's codes with each continuous policy node's observed dose replaced by its first
    branch's assigned dose: the retention a mutation reads at the shifted dose."""
    values = np.array(plan.values, dtype=float, copy=True)
    for node, targets in enumerate(plan.policy_targets or ()):
        if targets is not None:
            values[:, node] = np.asarray(targets[0], dtype=float)
    return values


@pytest.mark.parametrize("engine", ["ltmle", "sdr"])
def test_ma_retention_at_the_shifted_dose_is_caught(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-a: the censoring factor reads the dose the plan assigns, not the observed one.

    Measured deviation: 30.9 on both engines.
    """
    monkeypatch.setattr(sequential, "_x77_shifted_values", _shifted_values, raising=False)
    edit = (
        "data.history_design(time, treatment=plan.values, include_current=True)",
        "data.history_design(time, treatment=_x77_shifted_values(plan, data), "
        "include_current=True)",
    )
    mutate(monkeypatch, sequential, sequential, "_retention_factor", [edit])
    _caught(engine)


@pytest.mark.parametrize("engine", ["ltmle", "sdr"])
def test_mb_a_retention_design_without_the_current_dose_is_caught(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-b: the censoring model drops the node's own dose.

    Measured deviation: 36.2 on both engines.
    """
    edit = (
        "columns.append(self.dose_block(t, codes))",
        "columns.append(self.dose_block(t, codes) if t < time else np.zeros((self.n, 0)))",
    )
    mutate(monkeypatch, data_module, LongitudinalData, "history_design", [edit])
    _caught(engine)


def test_mc_sdr_weights_without_the_censoring_factor_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M-c: ``SDR``'s per-node factors drop the retention.  Measured deviation: 46.2."""
    edit = (
        "factor = factor * np.asarray(mechanism.censoring[time - 1][plan.label], dtype=float)",
        "factor = factor",
    )
    mutate(monkeypatch, sdr_module, sdr_module, "node_factors", [edit])
    _caught("sdr")


@pytest.mark.parametrize("engine", ["ltmle", "sdr"])
def test_md_a_classifier_trained_on_the_units_that_stayed_is_caught(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-d: the dose classifier trains on the rows uncensored through ``t``, not ``t - 1``.

    Among the units that stayed after node ``t`` the dose law is tilted by the retention, so
    the ratio becomes ``g^d / g`` given ``C_t = 1``.  Measured deviation: 166.7 on both
    engines.
    """
    original = sequential._continuous_node_ratios

    def stayed(data: Any, plans: Any, time: int, at_risk: Any, numerators: Any, **kw: Any) -> Any:
        kw["fit_at"] = kw["fit_at"] & data.uncensored_through(time)
        return original(data, plans, time, at_risk, numerators, **kw)

    monkeypatch.setattr(sequential, "_continuous_node_ratios", stayed)
    _caught(engine)
