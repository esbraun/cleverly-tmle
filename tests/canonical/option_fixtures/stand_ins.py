"""Stand-in payloads for the option fixtures: the R schema, written by Python.

The R scripts cannot run in the cloud session, so the parity comparisons of
:mod:`tests.canonical.option_fixtures.parity` are proven on payloads that Python writes in the
same schema.  ``tests/unit/test_option_fixtures.py`` runs each comparison on its stand-in, which
must pass, and then on a copy with one value changed, which must fail.

The ``tmle3`` and ``drtmle`` stand-ins do not call the code they check.  The first transcribes
``tmle3_Update`` (``update_step``, ``fit_submodel``, ``check_convergence``, ``update_best``) with
``numpy`` and ``scipy``.  The second fits the nuisance models with scikit-learn and solves the
fluctuation score by a root finder.  The ``ctmle`` stand-ins take the selected index, the scores and
the estimate from a ``CTMLE`` fit, because a second implementation of the collaborative search is
what the R run is for.  They prove that the comparison reads every field it claims to read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from tests.canonical.option_fixtures.parity import ctmle_fit, outcome_columns

__all__ = [
    "ctmle_patience_stand_in",
    "ctmle_wg_stand_in",
    "drtmle_stratify_stand_in",
    "tmle3_updater_stand_in",
]

#: The bound ``tmle3`` applies to the current prediction at every step.
BOUND = 0.005
#: R ``optim(method = "Brent")`` stops at ``reltol = sqrt(.Machine$double.eps)``.
BRENT_TOLERANCE = 1.4901161193847656e-08
PROVENANCE = {
    "script": "stand-in",
    "script_sha256": None,
    "image_id": "stand-in",
    "r_version": None,
    "packages": {},
}


def _glm(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """An unpenalized logistic regression to the tolerance of R's ``glm``."""
    return LogisticRegression(C=1e12, tol=1e-12, max_iter=10_000).fit(x, y)


def _binary_sample(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = (rng.uniform(size=n) < expit(0.4 * w1 - 0.3 * w2)).astype(float)
    y = (rng.uniform(size=n) < expit(-0.3 + 0.8 * a + 0.5 * w1 - 0.4 * w2)).astype(float)
    return {"Y": y, "A": a, "W1": w1, "W2": w2}


def _listed(values: Any) -> Any:
    """Arrays as lists, so the payload is JSON."""
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, dict):
        return {key: _listed(value) for key, value in values.items()}
    if isinstance(values, (list, tuple)):
        return [_listed(value) for value in values]
    if isinstance(values, np.generic):
        return values.item()
    return values


# ------------------------------------------------------------------------------- tmle3


class _Tmle3State:
    """The arrays of one ``tmle3`` run: the logits of the current fit and the clever covariates."""

    def __init__(self, sample: dict[str, np.ndarray]) -> None:
        y, a = sample["Y"], sample["A"]
        x = np.column_stack([sample["W1"], sample["W2"]])
        design = np.column_stack([a, x])
        g_model = _glm(x, a)
        p_one = g_model.predict_proba(x)[:, 1]
        q_model = _glm(design, y)
        self.y = y
        self.n = y.size
        self.p_observed = np.where(a == 1.0, p_one, 1.0 - p_one)
        self.q_observed = q_model.predict_proba(design)[:, 1]
        self.q_arms = [
            q_model.predict_proba(np.column_stack([np.full(self.n, level), x]))[:, 1]
            for level in (0.0, 1.0)
        ]
        # H for the parameter of arm j, at the observed treatment and with the treatment set to a.
        g_arm = [1.0 - p_one, p_one]
        self.h_observed = np.column_stack([(a == level) / self.p_observed for level in (0.0, 1.0)])
        self.h_counterfactual = [
            np.column_stack([(1.0 / g_arm[level]) * (level == column) for column in (0, 1)])
            for level in (0, 1)
        ]
        self.eta_observed = logit(self.q_observed)
        self.eta_arms = [logit(q) for q in self.q_arms]

    def snapshot(self) -> tuple[np.ndarray, list[np.ndarray]]:
        return self.eta_observed.copy(), [eta.copy() for eta in self.eta_arms]

    def restore(self, snapshot: tuple[np.ndarray, list[np.ndarray]]) -> None:
        self.eta_observed = snapshot[0].copy()
        self.eta_arms = [eta.copy() for eta in snapshot[1]]

    def estimates(self) -> tuple[np.ndarray, np.ndarray]:
        """``psi`` and the influence curve of each arm mean, as ``Param_TSM$estimates`` forms them."""
        q_observed = expit(self.eta_observed)
        arms = [expit(eta) for eta in self.eta_arms]
        psi = np.array([q.mean() for q in arms])
        curves = np.column_stack(
            [self.h_observed[:, j] * (self.y - q_observed) + arms[j] - psi[j] for j in (0, 1)]
        )
        return psi, curves

    def move(self, direction: np.ndarray) -> None:
        """Add ``H @ direction`` to every logit; ``direction`` is the coefficient of ``H``."""
        self.eta_observed = self.eta_observed + self.h_observed @ direction
        self.eta_arms = [
            eta + self.h_counterfactual[level] @ direction
            for level, eta in enumerate(self.eta_arms)
        ]


def _mle_step(state: _Tmle3State) -> np.ndarray:
    """The logistic MLE of the two coefficients of ``H`` with the current logits as the offset."""
    epsilon = np.zeros(2)
    for _ in range(100):
        p = expit(state.eta_observed + state.h_observed @ epsilon)
        score = state.h_observed.T @ (state.y - p)
        information = (state.h_observed * (p * (1.0 - p))[:, None]).T @ state.h_observed
        step = np.linalg.solve(information, score)
        epsilon = epsilon + step
        if np.max(np.abs(step)) < 1e-13:
            break
    return epsilon


def _tmle3_run(
    sample: dict[str, np.ndarray],
    *,
    rule: str,
    constrained: bool,
    optimise: bool,
    use_best: bool,
    delta: float,
    maxit: int,
) -> dict[str, Any]:
    """One ``tmle3_Update`` run, transcribed, in the schema of the R script."""
    state = _Tmle3State(sample)
    _, curves0 = state.estimates()
    ed0 = curves0.mean(axis=0)
    epsilons: list[np.ndarray] = []
    eds: list[np.ndarray] = []
    best_norm, best_snapshot, best_step = np.inf, None, None
    previous = ed0
    for step in range(1, maxit + 1):
        if constrained:
            direction = previous / np.linalg.norm(previous)
            h = state.h_observed @ direction
            eta = state.eta_observed

            def risk(t: float, h: np.ndarray = h, eta: np.ndarray = eta) -> float:
                p = expit(eta + t * h)
                return float(
                    np.mean(-(sample["Y"] * np.log(p) + (1.0 - sample["Y"]) * np.log(1.0 - p)))
                )

            if optimise:
                length = float(
                    minimize_scalar(
                        risk,
                        bounds=(0.0, delta),
                        method="bounded",
                        options={"xatol": BRENT_TOLERANCE},
                    ).x
                )
            else:
                length = delta
            epsilons.append(np.array([length]))
            state.move(length * direction)
        else:
            coefficient = _mle_step(state)
            epsilons.append(coefficient)
            state.move(coefficient)
        _, curves = state.estimates()
        ed = curves.mean(axis=0)
        eds.append(ed)
        n = state.n
        if rule == "scaled_var":
            thresholds = np.sqrt(curves.var(axis=0, ddof=1) / n) / min(np.log(n), 10.0)
        else:
            thresholds = np.full(2, 1.0 / n)
        converged = bool(np.all(np.abs(ed) <= thresholds))
        if converged:
            break
        if use_best and np.linalg.norm(ed) < best_norm:
            best_norm, best_snapshot, best_step = float(np.linalg.norm(ed)), state.snapshot(), step
        previous = ed
    steps = len(epsilons)
    if use_best:
        if np.linalg.norm(eds[-1]) < best_norm:
            best_snapshot, best_step = state.snapshot(), steps
        assert best_snapshot is not None
        state.restore(best_snapshot)
    psi, curves = state.estimates()
    n = state.n
    if rule == "scaled_var":
        final_thresholds = np.sqrt(curves.var(axis=0, ddof=1) / n) / min(np.log(n), 10.0)
    else:
        final_thresholds = np.full(2, 1.0 / n)
    reported = best_step if use_best else steps
    walk = _replay(state, epsilons, eds, ed0, constrained)
    q_final = expit(state.eta_observed)
    return {
        "options": {
            "convergence_type": rule,
            "constrain_step": constrained,
            "one_dimensional": constrained,
            "delta_epsilon": delta if constrained else None,
            "optim_delta_epsilon": optimise if constrained else None,
            "use_best": use_best,
            "maxit": maxit,
        },
        "steps": steps,
        "converged": steps < maxit,
        "best_step": best_step if use_best else None,
        "epsilons": epsilons,
        "eds": eds,
        "final": {
            "psi": psi,
            "ed": curves.mean(axis=0),
            "thresholds": final_thresholds,
            "q_observed": q_final,
            "q_arm0": expit(state.eta_arms[0]),
            "q_arm1": expit(state.eta_arms[1]),
        },
        "replay": {
            "max_abs_difference": float(np.max(np.abs(walk[0][reported - 1] - q_final))),
            "bound_binds": walk[1],
        },
    }


def _replay(
    state: _Tmle3State,
    epsilons: list[np.ndarray],
    eds: list[np.ndarray],
    ed0: np.ndarray,
    constrained: bool,
) -> tuple[list[np.ndarray], bool]:
    """The observed prediction after each step, rebuilt from the epsilons."""
    q = state.q_observed
    previous = ed0
    binds = False
    trace = []
    for epsilon, ed in zip(epsilons, eds, strict=True):
        binds = binds or bool(np.any((q < BOUND) | (q > 1.0 - BOUND)))
        if constrained:
            shift = (state.h_observed @ (previous / np.linalg.norm(previous))) * epsilon[0]
        else:
            shift = state.h_observed @ epsilon
        q = expit(logit(np.clip(q, BOUND, 1.0 - BOUND)) + shift)
        trace.append(q)
        previous = ed
    binds = binds or bool(np.any((q < BOUND) | (q > 1.0 - BOUND)))
    return trace, binds


#: The updaters of the R script: label, rule, constrained, optimise.
UPDATERS = (
    ("sample_size", "sample_size", False, None),
    ("scaled_var", "scaled_var", False, None),
    ("constrained_optim", "sample_size", True, True),
    ("constrained_fixed", "sample_size", True, False),
    ("constrained_optim_scaled_var", "scaled_var", True, True),
)


def tmle3_updater_stand_in(n: int = 200, seed: int = 11, delta: float = 0.02) -> dict[str, Any]:
    """The ``tmle3_updater.json`` schema, written by a transcription of ``tmle3_Update``.

    Parameters
    ----------
    n : int
        The sample size.
    seed : int
        The seed of the sample.
    delta : float
        ``delta_epsilon`` of the constrained updaters.

    Returns
    -------
    dict
        A JSON-ready payload.
    """
    sample = _binary_sample(n, seed)
    state = _Tmle3State(sample)
    _, curves = state.estimates()
    updaters = []
    for label, rule, constrained, optimise in UPDATERS:
        for use_best in (False, True):
            run = _tmle3_run(
                sample,
                rule=rule,
                constrained=constrained,
                optimise=bool(optimise),
                use_best=use_best,
                delta=delta,
                maxit=200 if constrained else 100,
            )
            updaters.append({"name": f"{label}_{'best' if use_best else 'last'}", **run})
    payload = {
        "fixture": "tmle3_updater",
        "schema": 1,
        "provenance": PROVENANCE,
        "seed": seed,
        "n": n,
        "bound": BOUND,
        "arm_levels": [0.0, 1.0],
        "data": sample,
        "initial": {
            "propensity_observed": state.p_observed,
            "q_observed": state.q_observed,
            "q_arm0": state.q_arms[0],
            "q_arm1": state.q_arms[1],
            "ed": curves.mean(axis=0),
        },
        "updaters": updaters,
    }
    return _listed(payload)  # type: ignore[no-any-return]


# ----------------------------------------------------------------------------- drtmle


def _arm_tmle(
    y: np.ndarray, a: np.ndarray, q_arm: np.ndarray, g_arm: np.ndarray, level: float
) -> float:
    """The TMLE of one arm mean: solve ``sum 1{A=a}/g (Y - expit(logit Q + eps / g)) = 0``."""
    indicator = a == level

    def score(epsilon: float) -> float:
        p = expit(logit(q_arm[indicator]) + epsilon / g_arm[indicator])
        return float(np.sum((y[indicator] - p) / g_arm[indicator]))

    epsilon = brentq(score, -20.0, 20.0, xtol=1e-14, rtol=1e-14)
    return float(np.mean(expit(logit(q_arm) + epsilon / g_arm)))


def drtmle_stratify_stand_in(n: int = 300, seed: int = 3) -> dict[str, Any]:
    """The ``drtmle_stratify.json`` schema, written with scikit-learn fits.

    Parameters
    ----------
    n : int
        The sample size.
    seed : int
        The seed of the sample.

    Returns
    -------
    dict
        A JSON-ready payload.
    """
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = (rng.uniform(size=n) < expit(0.3 * w1 - 0.2 * w2)).astype(float)
    y = (rng.uniform(size=n) < expit(-0.2 + 0.4 * w1 - 0.3 * w2 + a * (0.6 + 1.2 * w2))).astype(
        float
    )
    x = np.column_stack([w1, w2])
    g_one = _glm(x, a).predict_proba(x)[:, 1]
    g_arms = [1.0 - g_one, g_one]
    settings = []
    for stratify in (True, False):
        if stratify:
            q_arms = [
                _glm(x[a == level], y[a == level]).predict_proba(x)[:, 1] for level in (0.0, 1.0)
            ]
        else:
            pooled = _glm(np.column_stack([a, x]), y)
            q_arms = [
                pooled.predict_proba(np.column_stack([np.full(n, level), x]))[:, 1]
                for level in (0.0, 1.0)
            ]
        tmle = [_arm_tmle(y, a, q_arms[i], g_arms[i], float(i)) for i in (0, 1)]
        settings.append(
            {
                "stratify": stratify,
                "formula_q": "W1 + W2" if stratify else "A + W1 + W2",
                "qn_arm0": q_arms[0],
                "qn_arm1": q_arms[1],
                "gn_arm0": g_arms[0],
                "gn_arm1": g_arms[1],
                "gcomp": [q.mean() for q in q_arms],
                "tmle": tmle,
                "drtmle": tmle,
            }
        )
    payload = {
        "fixture": "drtmle_stratify",
        "schema": 1,
        "provenance": PROVENANCE,
        "seed": seed,
        "n": n,
        "tolg": 0.01,
        "arm_levels": [0.0, 1.0],
        "data": {"Y": y, "A": a, "W1": w1, "W2": w2},
        "settings": settings,
    }
    return _listed(payload)  # type: ignore[no-any-return]


# ------------------------------------------------------------------------------- ctmle


def _ctmle_sample(
    n: int,
    seed: int,
    p: int,
    strong: tuple[float, ...],
    outcome: tuple[float, ...],
    q_covariates: list[str],
) -> dict[str, Any]:
    """A binary sample with ``p`` covariates, selection folds and an initial outcome regression."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, p))
    eta_a = w @ np.array([*strong, *([0.0] * (p - len(strong)))])
    a = (rng.uniform(size=n) < expit(eta_a)).astype(float)
    eta_y = -0.2 + 0.5 * a + w @ np.array([*outcome, *([0.0] * (p - len(outcome)))])
    y = (rng.uniform(size=n) < expit(eta_y)).astype(float)
    names = [f"W{j + 1}" for j in range(p)]
    data: dict[str, Any] = {"Y": y, "A": a, **{name: w[:, j] for j, name in enumerate(names)}}
    fold = rng.permutation(np.tile(np.arange(5), n // 5 + 1)[:n])
    return {
        "n": n,
        "data": data,
        "covariates": names,
        "q_covariates": q_covariates,
        "fold": fold,
    }


def _initial_outcome(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The arm predictions of a binomial GLM of the outcome on the treatment and ``q_covariates``."""
    data, names = sample["data"], sample["covariates"]
    x = np.column_stack([data["A"], *(data[name] for name in names)])
    columns = outcome_columns(sample)
    model = _glm(x[:, columns], data["Y"])
    arms = []
    for level in (0.0, 1.0):
        arm = x.copy()
        arm[:, 0] = level
        arms.append(model.predict_proba(arm[:, columns])[:, 1])
    return arms[0], arms[1]


def _run_record(
    sample: dict[str, Any],
    *,
    ordering: list[str],
    patience: int | None,
    pool: list[str] | None,
    candidates: int,
) -> dict[str, Any]:
    """One ``ctmleDiscrete`` run as the R script writes it, from a ``CTMLE`` fit."""
    result = ctmle_fit(sample, ordering=ordering, patience=patience, pool=pool)
    selection = result.extra["ctmle"]
    risk = [float(value) for value in selection.cv_risk]
    best_k = selection.selected + 1
    record: dict[str, Any] = {
        "best_k": best_k,
        "est": float(result["ate"].psi),
        "cv_score": risk + [None] * (candidates - len(risk)),
        "terms": ordering[: max(best_k - 1, 1)],
    }
    if patience is not None:
        record["patience"] = patience
        record["scored"] = len(risk)
        record["candidates"] = candidates
        record["stopped_early"] = selection.stopped_at is not None
    return record


def ctmle_patience_stand_in(n: int = 600, seed: int = 13) -> dict[str, Any]:
    """The ``ctmle_patience.json`` schema, from ``CTMLE`` fits.

    The sample has one confounder (``W1``) and five noise covariates, so the cross-validated risk
    rises after the first covariate and ``patience=2`` stops the path before its end.

    Parameters
    ----------
    n : int
        The sample size.
    seed : int
        The seed of the sample.

    Returns
    -------
    dict
        A JSON-ready payload.
    """
    sample = _ctmle_sample(n, seed, 6, (0.9,), (0.8,), ["W2", "W3", "W4", "W5", "W6"])
    q0, q1 = _initial_outcome(sample)
    names = sample["covariates"]
    payload = {
        "fixture": "ctmle_patience",
        "schema": 1,
        "provenance": PROVENANCE,
        "seed": seed,
        "n": n,
        "patience": 2,
        "data": sample["data"],
        "covariates": names,
        "q_covariates": sample["q_covariates"],
        "fold": sample["fold"],
        "q_arm0": q0,
        "q_arm1": q1,
        "runs": {
            "whole_path": {
                **_run_record(sample, ordering=names, patience=None, pool=None, candidates=7),
                "patience": None,
                "scored": 7,
                "candidates": 7,
                "stopped_early": False,
            },
            "patient": _run_record(sample, ordering=names, patience=2, pool=None, candidates=7),
        },
    }
    return _listed(payload)  # type: ignore[no-any-return]


def ctmle_wg_stand_in(n: int = 400, seed: int = 2) -> dict[str, Any]:
    """The ``ctmle_wg.json`` schema, from ``CTMLE`` fits.

    ``W2`` drives the treatment and lies outside the restricted pool ``{W1, W3}``.

    Parameters
    ----------
    n : int
        The sample size.
    seed : int
        The seed of the sample.

    Returns
    -------
    dict
        A JSON-ready payload.
    """
    sample = _ctmle_sample(n, seed, 4, (0.8, 1.1, 0.3), (0.8, 0.6, 0.4), ["W1", "W3", "W4"])
    q0, q1 = _initial_outcome(sample)
    names = sample["covariates"]
    pool = ["W1", "W3"]
    payload = {
        "fixture": "ctmle_wg",
        "schema": 1,
        "provenance": PROVENANCE,
        "seed": seed,
        "n": n,
        "data": sample["data"],
        "covariates": names,
        "q_covariates": sample["q_covariates"],
        "wg_columns": [1, 3],
        "fold": sample["fold"],
        "q_arm0": q0,
        "q_arm1": q1,
        "runs": {
            "restricted": {
                "pool": pool,
                "order": [1, 2],
                **_run_record(sample, ordering=pool, patience=None, pool=pool, candidates=3),
            },
            "whole_pool": {
                "pool": names,
                "order": [1, 2, 3, 4],
                **_run_record(sample, ordering=names, patience=None, pool=None, candidates=5),
            },
        },
    }
    return _listed(payload)  # type: ignore[no-any-return]
