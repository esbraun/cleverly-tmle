"""Compositions of a data-tuned cap with weights, clusters, censoring, events and storage.

``X40-plan.md`` section 4.3, parts (a), (b) and (e).  Each composition either works and
has a witness here, or is refused with a test (declared atoms, in ``test_mtp_tuned_caps.py``).
The witness for a working composition is the tuner's term ``sum_t dPhi_n' IF_t`` written
longhand (T-e4 (ii)), with the weights, the at-risk rows and the at-risk share written
out and not read from the library; a mutation control removes the piece that composition
adds and must move the reported term by more than ``CAUGHT``.

* weights: the weighted quantile, the weighted bandwidth, and ``w`` in the term;
* censoring: the tuner reads the rows at risk, and its influence function divides by the
  at-risk share ``pi``;
* clusters: grouped folds train each fold's tuner on whole clusters, and the cluster-robust
  variance reads the curve with the term;
* survival and competing risks: a horizon before a tuned node carries only the nodes it
  reaches (the ``step is None`` branch), and a tuner reads the rows still event free;
* truncation replay and ``dumps``/``loads``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor

from cleverly.estimators import serialize
from cleverly.inference.cluster import cluster_sums
from cleverly.interventions import (
    DoseQuantile,
    LinearQuantileRegression,
    Shift,
    Support,
    SupportBoundary,
)
from cleverly.interventions import tuning as tuning_module
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen
from cleverly.longitudinal import regimen as regimen_module
from cleverly.longitudinal.estimator import longitudinal_truncation_curve

from .. import law_q_tuned as lawq

CAUGHT = 1e-4
expit = lawq.expit
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}
EVENT_COLUMNS: dict[str, Any] = {
    "outcome": ["Y1", "Y2"],
    "treatment": ["A1", "A2"],
    "censoring": ["C1", "C2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}
COMPETING_COLUMNS: dict[str, Any] = {
    **EVENT_COLUMNS,
    "outcome": {"R": ["R1", "R2"], "D": ["D1", "D2"]},
}


def _regimen(kind: str, label: str = "tuned") -> DynamicRegimen:
    if kind == "quantile":
        caps: tuple[Any, Any] = (DoseQuantile(0.5), DoseQuantile(0.5))
    else:
        caps = (
            SupportBoundary(LinearQuantileRegression(["W"]), 0.5),
            SupportBoundary(LinearQuantileRegression(["W", "L2", "A1"]), 0.5),
        )
    return DynamicRegimen(label, tuple(Shift(1.0, cap=cap, target="population") for cap in caps))


def _settings(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=500),
        "censoring_learner": LogisticRegression(max_iter=500),
        "density_bins": 20,
        "random_state": 1,
        "simultaneous": False,
        **overrides,
    }


def _fit(
    estimator: type, regimens: Any, frame: pd.DataFrame, columns: dict[str, Any], **kwargs: Any
) -> Any:
    if columns["outcome"] == "Y":
        kwargs.setdefault("q_bounds", (0.0, 1.0))
    fit_keywords = {
        key: kwargs.pop(key) for key in ("weights", "id", "dose_atoms") if key in kwargs
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(regimens, **_settings(**kwargs)).fit(frame, **columns, **fit_keywords)


# ------------------------------------------------------------------ the data


def _censored(n: int, seed: int) -> pd.DataFrame:
    """Law Q with a unit lost after node 1 at a rate that depends on ``W``.

    A lost unit has no later treatment, covariate or outcome, so the node-2 tuner reads the
    followers only, about 70% of the rows.
    """
    base = lawq.sample(n, seed)
    rng = np.random.default_rng(seed + 1000)
    stays = rng.uniform(size=n) < expit(1.0 - 0.5 * base["W"].to_numpy())
    out = base.copy()
    out["C1"] = stays.astype(float)
    out["C2"] = np.where(stays, 1.0, np.nan)
    for column in ("L2", "A2", "Y"):
        out.loc[~stays, column] = np.nan
    return out[["W", "A1", "C1", "L2", "A2", "C2", "Y"]]


def _events(n: int, seed: int, *, competing: bool = False) -> pd.DataFrame:
    """Law Q with loss to follow-up and an absorbing event (two causes when ``competing``)."""
    base = lawq.sample(n, seed)
    rng = np.random.default_rng(seed + 2000)
    w, a1 = base["W"].to_numpy(), base["A1"].to_numpy()
    stays1 = rng.uniform(size=n) < expit(1.2 - 0.5 * w)
    cause1 = rng.uniform(size=n) < expit(-1.0 + 0.9 * a1 + 0.3 * w)
    cause1b = rng.uniform(size=n) < expit(-2.5 + 0.4 * a1)
    alive = stays1 & ~cause1 & ~(cause1b if competing else False)
    a2, l2 = base["A2"].to_numpy(), base["L2"].to_numpy()
    stays2 = rng.uniform(size=n) < expit(1.5 - 0.3 * a2)
    cause2 = rng.uniform(size=n) < expit(-0.5 + 1.2 * a2 + 0.3 * l2)
    cause2b = rng.uniform(size=n) < expit(-2.0 + 0.3 * a2)
    nan = np.nan
    out = pd.DataFrame(
        {
            "W": w,
            "A1": a1,
            "C1": 1.0,
            "L2": np.where(alive, l2, nan),
            "A2": np.where(alive, a2, nan),
            "C2": np.where(alive, np.where(stays2, 1.0, 0.0), nan),
        }
    )
    out["C1"] = np.where(stays1, 1.0, 0.0)
    node2 = alive & stays2
    if competing:
        first = np.where(stays1, cause1.astype(float), nan)
        second = np.where(stays1, (cause1b & ~cause1).astype(float), nan)
        out["R1"], out["D1"] = first, second
        out["R2"] = np.where(node2, cause2.astype(float), nan)
        out["D2"] = np.where(node2, (cause2b & ~cause2).astype(float), nan)
    else:
        out["Y1"] = np.where(stays1, cause1.astype(float), nan)
        out["Y2"] = np.where(node2, cause2.astype(float), nan)
    return out


# ------------------------------------------------------------------ the longhand


def _wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The smallest value whose weighted distribution function reaches ``q``."""
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    return float(values[order][np.searchsorted(cdf, q - 1e-12)])


def _wscale(values: np.ndarray, weights: np.ndarray) -> float:
    total = np.sum(weights)
    mean = np.sum(weights * values) / total
    sd = np.sqrt(np.sum(weights * (values - mean) ** 2) / total)
    iqr = _wquantile(values, weights, 0.75) - _wquantile(values, weights, 0.25)
    return float(0.9 * (min(sd, iqr / 1.34) if iqr > 0.0 else sd))


def _node_term(
    frame: pd.DataFrame,
    node: int,
    kind: str,
    weights: np.ndarray,
    risk: np.ndarray,
    prefix: np.ndarray,
    probes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``w * IF_t' dPhi_t`` at one node, from the tuner's definition, over the rows at risk.

    Also returns the rows whose regression residual is zero to rounding: a linear program
    interpolates ``p`` rows, and the sign of ``1{residual <= 0}`` there is the solver's noise.
    """
    n = len(frame)
    dose = np.where(risk, frame[f"A{node}"].to_numpy(), 0.0)
    w, wr, ar = weights, weights[risk], dose[risk]
    share = np.sum(wr) / np.sum(w)
    ties = np.zeros(n, dtype=bool)
    if kind == "quantile":
        theta = _wquantile(ar, wr, 0.5)
        h = _wscale(ar, wr) * ar.size**-0.2
        density = np.sum(wr * stats.norm.pdf((theta - ar) / h)) / (h * np.sum(wr))
        influence = np.where(risk, -((dose <= theta) - 0.5) / (share * density), 0.0)[:, None]
        boundary = np.full(n, theta)
        gradient = np.ones((n, 1))
    else:
        names = ["W"] if node == 1 else ["W", "L2", "A1"]
        features = np.where(risk[:, None], frame[names].to_numpy(), 0.0)
        design = np.column_stack([np.ones(n), features])
        model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
        model.fit(features[risk], ar, sample_weight=wr)
        theta = np.concatenate([[model.intercept_], model.coef_])
        residual = dose - design @ theta
        ties = risk & (np.abs(residual) < 1e-9)
        h = _wscale(residual[risk], wr) * ar.size ** (-1.0 / 3.0)
        inside = (np.abs(residual) <= h) & risk
        jacobian = (design * (w * inside)[:, None]).T @ design / (2.0 * h * np.sum(wr))
        score = design * (0.5 - (residual <= 0.0))[:, None]
        influence = np.where(risk[:, None], np.linalg.solve(jacobian, score.T).T / share, 0.0)
        boundary, gradient = design @ theta, design
    gap = dose - (boundary - 1.0)
    h = _wscale(gap[risk], wr) * ar.size**-0.2
    kernel = stats.norm.pdf(gap / h) / h
    row = np.where(risk, w * prefix * kernel * (probes[:, 0] - probes[:, 1]), 0.0)
    derivative = (row @ gradient) / n
    return w * (influence @ derivative), ties


def _term(
    frame: pd.DataFrame,
    fit: Any,
    kind: str,
    *,
    weights: np.ndarray | None = None,
    nodes: tuple[int, ...] = (1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """``sum_t w dPhi_t' IF_t``: the library's probes and prefix, everything else written out."""
    n = len(frame)
    w = np.ones(n) if weights is None else weights * (n / np.sum(weights))
    by_time = {step.time: step for step in fit.steps}
    term = np.zeros(n)
    ties = np.zeros(n, dtype=bool)
    for node in nodes:
        step = by_time[node]
        prefix = np.ones(n) if node == 1 else by_time[node - 1].clever
        risk = frame[f"A{node}"].notna().to_numpy()
        np.testing.assert_array_equal(risk, step.at_risk)
        part, tied = _node_term(frame, node, kind, w, risk, prefix, step.probes)
        term, ties = term + part, ties | tied
    return term, ties


def _same(actual: Any, longhand: tuple[np.ndarray, np.ndarray]) -> None:
    """The reported term is the longhand term, off the rows a linear program ties."""
    term, ties = longhand
    assert ties.sum() <= 12
    keep = ~ties
    np.testing.assert_allclose(np.asarray(actual)[keep], term[keep], rtol=1e-8, atol=1e-10)


def _gap(actual: Any, longhand: tuple[np.ndarray, np.ndarray]) -> float:
    term, ties = longhand
    return float(np.max(np.abs(np.asarray(actual)[~ties] - term[~ties])))


# ------------------------------------------------------------------ weights


@pytest.fixture(scope="module")
def weighted() -> pd.DataFrame:
    frame = lawq.sample(1500, 11)
    rng = np.random.default_rng(3)
    # Weights that depend on the dose, so a weighted quantile differs from the plain one.
    frame["w"] = 0.4 + 1.2 * expit(frame["A1"].to_numpy() + rng.normal(size=len(frame)))
    return frame


def _dropped_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: the tuner and its influence function ignore the observation weights."""
    fit_tuner = regimen_module.fit_tuner
    influence = regimen_module.tuner_influence
    monkeypatch.setattr(
        regimen_module, "fit_tuner", lambda t, d, f, r, w=None: fit_tuner(t, d, f, r, None)
    )
    monkeypatch.setattr(
        regimen_module, "tuner_influence", lambda f, d, fr, r, w=None: influence(f, d, fr, r, None)
    )


@pytest.mark.parametrize(
    ("estimator", "folds", "kind"),
    [
        (LTMLE, 1, "quantile"),
        (LTMLE, 3, "regression"),
        (SDR, 1, "regression"),
        (SDR, 3, "quantile"),
    ],
)
def test_weights_a_weighted_tuned_fit_reports_the_weighted_term_longhand(
    estimator: type, folds: int, kind: str, weighted: pd.DataFrame
) -> None:
    """The weighted quantile and bandwidth, the at-risk share and ``w`` in the term."""
    result = _fit(estimator, [_regimen(kind)], weighted, COLUMNS, n_folds=folds, weights="w")
    fit = result.fits["tuned"]
    longhand = _term(weighted, fit, kind, weights=weighted["w"].to_numpy())
    _same(fit.tuning_term, longhand)
    assert np.std(longhand[0]) > 0.1 * np.std(fit.influence_curve_scaled)
    # The weights moved the tuner: the unweighted quantile is a different cap.
    unweighted = _fit(estimator, [_regimen(kind)], weighted, COLUMNS, n_folds=folds)
    assert abs(unweighted.psi("ey_regimen[tuned]") - result.psi("ey_regimen[tuned]")) > CAUGHT


@pytest.mark.parametrize("kind", ["quantile", "regression"])
def test_weights_a_tuner_that_ignores_the_weights_fails(
    kind: str, weighted: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dropped_weights(monkeypatch)
    result = _fit(LTMLE, [_regimen(kind)], weighted, COLUMNS, n_folds=1, weights="w")
    fit = result.fits["tuned"]
    longhand = _term(weighted, fit, kind, weights=weighted["w"].to_numpy())
    assert _gap(fit.tuning_term, longhand) > CAUGHT


# ------------------------------------------------------------------ censoring


@pytest.fixture(scope="module")
def censored() -> pd.DataFrame:
    return _censored(1500, 11)


@pytest.mark.parametrize(
    ("estimator", "folds", "kind"),
    [
        (LTMLE, 1, "quantile"),
        (LTMLE, 3, "regression"),
        (SDR, 1, "quantile"),
        (SDR, 3, "regression"),
    ],
)
def test_censoring_a_censored_tuned_fit_reports_the_term_longhand(
    estimator: type, folds: int, kind: str, censored: pd.DataFrame
) -> None:
    """T-e4 (ii) with ``pi < 1`` at node 2: the tuner reads the followers and divides by ``pi``."""
    result = _fit(
        estimator, [_regimen(kind)], censored, {**COLUMNS, "censoring": ["C1", "C2"]}, n_folds=folds
    )
    fit = result.fits["tuned"]
    share = float(censored["A2"].notna().mean())
    assert share < 0.8
    longhand = _term(censored, fit, kind)
    _same(fit.tuning_term, longhand)
    assert np.std(longhand[0]) > 0.1 * np.std(fit.influence_curve_scaled)


@pytest.mark.parametrize("kind", ["quantile", "regression"])
def test_censoring_m_share_dropping_the_at_risk_share_fails(
    kind: str, censored: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's mutation ``share``: the tuner curve loses ``1 / pi`` and the term shrinks."""
    original = tuning_module.tuner_influence

    def without_share(
        fitted: Any, doses: Any, frame: Any, at_risk: Any, weights: Any = None
    ) -> Any:
        out = original(fitted, doses, frame, at_risk, weights)
        return out * float(np.mean(np.asarray(at_risk, dtype=bool)))

    monkeypatch.setattr(regimen_module, "tuner_influence", without_share)
    result = _fit(
        LTMLE, [_regimen(kind)], censored, {**COLUMNS, "censoring": ["C1", "C2"]}, n_folds=1
    )
    fit = result.fits["tuned"]
    assert _gap(fit.tuning_term, _term(censored, fit, kind)) > CAUGHT


# ------------------------------------------------------------------ clusters


@pytest.fixture(scope="module")
def clustered() -> pd.DataFrame:
    frame = lawq.sample(1500, 11)
    frame["cluster"] = np.arange(len(frame)) // 3
    return frame


def test_clusters_the_cluster_robust_variance_reads_the_curve_with_the_term(
    clustered: pd.DataFrame,
) -> None:
    plain = _fit(LTMLE, [_regimen("quantile")], clustered, COLUMNS, n_folds=1)
    grouped = _fit(LTMLE, [_regimen("quantile")], clustered, COLUMNS, n_folds=1, id="cluster")
    fit = grouped.fits["tuned"]
    curve = np.asarray(fit.influence_curve_scaled)
    np.testing.assert_array_equal(curve, plain.fits["tuned"].influence_curve_scaled)
    n = len(clustered)
    sums = cluster_sums(curve, clustered["cluster"].to_numpy())
    expected = float(np.sqrt(sums.size * np.var(sums, ddof=1) / n**2))
    assert grouped["ey_regimen[tuned]"].std_error == pytest.approx(expected, rel=1e-10)
    # Without the term, the same clusters give a different variance (the term is not small).
    sums = cluster_sums(curve - np.asarray(fit.tuning_term), clustered["cluster"].to_numpy())
    without = float(np.sqrt(sums.size * np.var(sums, ddof=1) / n**2))
    assert abs(without / expected - 1.0) > 0.05


@pytest.mark.parametrize("kind", ["quantile", "regression"])
def test_clusters_each_fold_trains_its_tuner_on_whole_clusters(
    kind: str, clustered: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[np.ndarray] = []
    fit_tuner = regimen_module.fit_tuner

    def recorded(tuner: Any, doses: Any, frame: Any, rows: Any, weights: Any = None) -> Any:
        seen.append(np.array(rows, dtype=bool))
        return fit_tuner(tuner, doses, frame, rows, weights)

    monkeypatch.setattr(regimen_module, "fit_tuner", recorded)
    result = _fit(LTMLE, [_regimen(kind)], clustered, COLUMNS, n_folds=3, id="cluster")
    cluster = clustered["cluster"].to_numpy()
    held_out = [set(cluster[test]) for _, test in result.folds]
    assert not set.intersection(*held_out)
    # Four fits per node: all rows at risk, then each fold's training rows.
    training = [rows for rows in seen if not rows.all()]
    assert len(training) == 2 * 3
    for rows, out in zip(training[:3], held_out, strict=True):
        assert not set(cluster[rows]) & out
        assert set(cluster[~rows]) == out


# ------------------------------------------------------------------ survival and competing risks


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return _events(3000, 3)


def test_survival_a_horizon_before_the_tuned_node_carries_the_nodes_it_reaches(
    events: pd.DataFrame,
) -> None:
    """At ``t = 1`` the node-2 tuner has no step: its derivative and term are dropped."""
    result = _fit(LTMLE, [_regimen("quantile")], events, EVENT_COLUMNS, n_folds=1)
    early, late = result.fits["tuned @ t=1"], result.fits["tuned @ t=2"]
    assert len(early.tuning_derivatives) == 1
    assert len(late.tuning_derivatives) == 2
    _same(early.tuning_term, _term(events, early, "quantile", nodes=(1,)))
    _same(late.tuning_term, _term(events, late, "quantile"))
    # The tuner at node 2 reads the units still alive and observed there.
    assert 0.3 < float(events["A2"].notna().mean()) < 0.7
    assert np.std(late.tuning_term) > 0.1 * np.std(late.influence_curve_scaled)


def test_survival_m_a_term_past_the_horizon_would_change_the_early_curve(
    events: pd.DataFrame,
) -> None:
    """The control: adding the node-2 term to the horizon-1 curve moves it by more than CAUGHT."""
    result = _fit(LTMLE, [_regimen("quantile")], events, EVENT_COLUMNS, n_folds=1)
    early, late = result.fits["tuned @ t=1"], result.fits["tuned @ t=2"]
    both = _term(events, late, "quantile")
    assert _gap(early.tuning_term, both) > CAUGHT


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_competing_risks_a_cause_specific_curve_carries_the_term(
    estimator: type,
) -> None:
    competing = _events(3000, 3, competing=True)
    result = _fit(estimator, [_regimen("regression")], competing, COMPETING_COLUMNS, n_folds=1)
    fits = {key: fit for key, fit in result.fits.items() if fit.tuning_term is not None}
    assert fits
    for fit in fits.values():
        nodes = tuple(range(1, len(fit.tuning_derivatives) + 1))
        _same(fit.tuning_term, _term(competing, fit, "regression", nodes=nodes))
    assert any(len(fit.tuning_derivatives) == 2 for fit in fits.values())


# ------------------------------------------------------------------ truncation replay, dumps/loads


def _atom_node_two_fit(**kwargs: Any) -> Any:
    """A tuned cap at node 1 with a declared support, and an atom at node 2 (not tuned)."""
    frame = lawq.sample(1500, 11)
    held = np.random.default_rng(5).uniform(size=len(frame)) < 0.3
    frame.loc[held, "A2"] = 3.0
    regimen = DynamicRegimen(
        "tuned", (Shift(1.0, cap=DoseQuantile(0.5), target="population"), Shift(0.0, cap=None))
    )
    return _fit(
        LTMLE,
        [regimen],
        frame,
        {**COLUMNS, "continuous_treatment": {"A1": Support(-4.0, 4.0), "A2": None}},
        n_folds=1,
        dose_atoms={"A2": (3.0,)},
        **kwargs,
    )


@pytest.mark.parametrize("atoms", [False, True])
def test_truncation_replay_reproduces_a_tuned_fit(atoms: bool) -> None:
    """The replay re-resolves the tuned map, with or without an atom node beside it."""
    if atoms:
        result = _atom_node_two_fit(g_bounds=(0.01, 1.0))
    else:
        result = _fit(
            LTMLE,
            [_regimen("quantile")],
            lawq.sample(1500, 11),
            COLUMNS,
            n_folds=1,
            g_bounds=(0.01, 1.0),
        )
    curve = longitudinal_truncation_curve(
        result, [(0.01, 1.0), (0.2, 1.0)], estimands=["ey_regimen[tuned]"]
    ).set_index("is_fitted_bound")
    psi = result.psi("ey_regimen[tuned]")
    assert curve.loc[True, "psi"] == pytest.approx(psi, abs=1e-12)
    # A continuous node carries no bound, so the swept bound moves nothing, and the replay
    # re-resolves the tuned map to do it: an untuned plan could not give the estimate.
    assert curve.loc[False, "psi"] == pytest.approx(psi, abs=1e-12)


def test_dumps_and_loads_keep_a_tuned_result_with_a_support_and_atoms() -> None:
    result = _atom_node_two_fit()
    loaded = serialize.loads(serialize.dumps(result))
    name = "ey_regimen[tuned]"
    assert loaded.psi(name) == result.psi(name)
    np.testing.assert_array_equal(loaded.influence_curves[name], result.influence_curves[name])
    before, after = result.fits["tuned"], loaded.fits["tuned"]
    np.testing.assert_array_equal(after.tuning_term, before.tuning_term)
    assert len(after.tuning_derivatives) == len(before.tuning_derivatives) == 1
    assert float(np.std(before.tuning_term)) > 0.0
    assert result.data.node_atoms(2) == loaded.data.node_atoms(2) == (3.0,)
    assert result.data.node_support(1) == loaded.data.node_support(1)
