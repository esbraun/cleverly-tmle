r"""The design probes of the heavy-ratio diagnosis, rebuilt from the shipped code.

Each probe writes one row per draw (``probe*_rows.csv.gz``) and :func:`summaries` reads the
tables the diagnosis cites from those rows (``probe*.csv``).  Every probe checkpoints each batch
of draws, so a stopped run resumes.

``probe1``
    The linearized bootstrap on the 1,000 committed draws of ``pricing_{ltmle,sdr}_tau5``: the
    percentile, BCa, equal-tailed and symmetric bootstrap-t intervals of
    :math:`\psi + \bar D^* - \bar D` (B = 1,999), beside the Wald interval.  Each fit is the
    committed one, so its estimate equals the committed row.
``probe2``
    The full-refit BCa (B = 199) of ``SDR`` on the committed ``pricing_sdr_tau5`` draws whose
    Wald interval of ``ate_regimen[price_down vs natural_prices]`` missed in ``probe1``, and on
    30 draws it covered.
``probe4``
    240 paired draws of law B at n = 4,000: ``SDR`` under ``p4`` with the correctly specified
    outcome, the rich first-node basis projected onto ``[0, 1]`` or not.
``probe5``
    ``variance="policy_weighted"`` on the five configurations of the variance probe: law P
    ``tau5`` with the exact ratio (variance learner d3 on 1,000 committed draws, d5 on the first
    500), law P ``tau5`` with the learned classifier (d3, 300), and law B ``p4`` with the correct
    outcome at n = 4,000 (d3, 800) and with the primary learners at n = 5,000 (d3, 300).  The
    rows hold both standard errors, so the summary reads the default Wald interval, the
    policy-weighted variance alone, and their maximum, which ships.
``probe6``
    The held-out screen preview: law P ``price_down`` at three nodes and ``price_up`` at five
    (300 draws), law B ``p2`` and ``p3`` (150 draws), each draw fitted under d3 and under d5.
``screen``
    The variance-learner screen on the first 24 committed ``pricing_ltmle_tau5`` draws: the
    change-of-measure diagonal's last-node regression under depth 3, 5 and 8, with and without
    the law's gap features, and with the outcome model's :math:`Q_5 (1 - Q_5)` in place of the
    last node's regression, against :math:`n \operatorname{Var}(\text{error})` of the committed
    draws.

d3 is a boosted regression of depth 3 with 60 iterations and leaves of at least 50; d5, the
shipped learner (:data:`cleverly.longitudinal.policy_variance.VARIANCE_LEARNER`), depth 5 with
200 iterations and leaves of at least 20.  The law-B fits read the oracle binned hazard, as the
study does, and the shipped variance reads that density at the policy dose; the probe read the
exact ratio there (median relative difference 0.001).
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta, norm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import LTMLE, SDR
from cleverly.longitudinal import policy_variance as pv
from tests.studies import canonical_mtp_boundaries as boundaries
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_boundaries_common as law_b
from tests.studies import mtp_pricing_common as law_p
from tests.studies.fractional_glm import QuasiBinomialGLM

Z = float(norm.ppf(0.975))
#: The variance learners of the probes.
LEARNERS: dict[str, dict[str, Any]] = {
    "d3": {"max_depth": 3, "max_iter": 60, "min_samples_leaf": 50},
    "d5": {"max_depth": 5, "max_iter": 200, "min_samples_leaf": 20},
    "d8": {"max_depth": 8, "max_iter": 400, "min_samples_leaf": 10},
}
DOWN = "ey_regimen[price_down]"
DOWN_ATE = "ate_regimen[price_down vs natural_prices]"
PRICING_NAMES = (DOWN, DOWN_ATE)
BATCH = 50


def _quiet() -> contextlib.AbstractContextManager[None]:
    manager = warnings.catch_warnings()

    @contextlib.contextmanager
    def quiet() -> Iterator[None]:
        with manager:
            warnings.simplefilter("ignore")
            yield

    return quiet()


def resumable(
    name: str,
    keys: Sequence[Any],
    job: Callable[[Any], list[dict[str, Any]]],
    checkpoint: Path,
    batch: int = BATCH,
) -> pd.DataFrame:
    """Run ``job`` on every key, saving each batch of keys, and return all rows."""
    checkpoint.mkdir(parents=True, exist_ok=True)
    frames = []
    for start in range(0, len(keys), batch):
        path = checkpoint / f"{name}_{start // batch:04d}.parquet"
        if not path.exists():
            rows = [row for key in keys[start : start + batch] for row in job(key)]
            pd.DataFrame(rows).to_parquet(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


@contextlib.contextmanager
def variance_learner(name: str) -> Iterator[None]:
    """The shipped variance regressions with one probe learner's depth, iterations and leaves."""
    shipped = pv.VARIANCE_LEARNER
    pv.VARIANCE_LEARNER = {**shipped, **LEARNERS[name]}  # type: ignore[misc]
    try:
        yield
    finally:
        pv.VARIANCE_LEARNER = shipped  # type: ignore[misc]


# ----------------------------------------------------------------------------- fits


def pricing_model(
    tau: int,
    labels: tuple[str, ...],
    estimator: str,
    *,
    treatment_learner: Any = None,
    variance: str = "influence_curve",
    **extra: Any,
) -> LTMLE | SDR:
    """``canonical_mtp_pricing.fit``'s primary model, with the variance option."""
    settings: dict[str, Any] = {
        "reference": pricing.REFERENCE,
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": clone(
            law_p.OracleRatio() if treatment_learner is None else treatment_learner
        ),
        "n_folds": 1,
        "g_bounds": pricing.G_BOUNDS,
        "ratio": "classifier",
        "simultaneous": False,
        "random_state": 0,
        "variance": variance,
        **extra,
    }
    regimens = law_p.regimens(labels, tau)
    if estimator == "ltmle":
        return LTMLE(regimens, max_iter=100, tol=1e-10, alpha=pricing.PRIMARY_ALPHA, **settings)
    return SDR(regimens, **settings)


def boundaries_fit(frame: pd.DataFrame, label: str, estimator: str, outcome: str) -> Any:
    """``canonical_mtp_boundaries.fit`` on law B with ``variance="policy_weighted"``."""
    q, first, g = boundaries.learners(frame, outcome)  # type: ignore[arg-type]
    if outcome == "correct" and estimator == "sdr":
        first = law_b.UnitRangePseudo()
    settings: dict[str, Any] = {
        "reference": boundaries.REFERENCE,
        "outcome_learner": clone(q),
        "pseudo_learner": clone(first),
        "treatment_learner": clone(g),
        "n_folds": 1,
        "g_bounds": boundaries.G_BOUNDS,
        "density_bins": boundaries.oracle_bins(len(frame)),
        "ratio": "density",
        "simultaneous": False,
        "random_state": 0,
        "variance": "policy_weighted",
    }
    regimens = law_b.regimens((boundaries.REFERENCE, label))
    model: Any = (
        LTMLE(regimens, max_iter=100, tol=1e-10, **settings)
        if estimator == "ltmle"
        else SDR(regimens, **settings)
    )
    return model.fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        continuous_treatment={"A1": law_b.SUPPORT, "A2": law_b.SUPPORT},
    )


def policy_row(estimate: Any) -> dict[str, float]:
    """The estimate and both standard errors of an option estimate."""
    record = estimate.policy
    se_pw = float(np.sqrt(record.policy_variance)) if record.usable else float("nan")
    return {
        "psi": float(estimate.psi),
        "se_ic": float(np.sqrt(max(record.influence_variance, 0.0))),
        "se_pw": se_pw,
        "se": float(estimate.std_error),
    }


# ------------------------------------------------------------------------- probe 1


def bootstrap_intervals(
    psi: float, curve: np.ndarray, rng: np.random.Generator, draws: int = 1_999
) -> dict[str, Any]:
    """The linearized bootstrap intervals of ``psi + mean(D*) - mean(D)``."""
    n = curve.size
    d = curve - curve.mean()
    s = float(np.sqrt(np.mean(d**2) / n))
    index = rng.integers(0, n, (draws, n))
    resampled = d[index]
    m = resampled.mean(1)
    sd = np.sqrt(((resampled - m[:, None]) ** 2).mean(1) / n)
    values = psi + m
    out: dict[str, Any] = {"se_ic": s}
    out["pct"] = tuple(np.quantile(values, [0.025, 0.975]))
    z0 = float(norm.ppf((np.sum(values < psi) + 0.5 * np.sum(values == psi)) / draws))
    a = float(np.sum(d**3) / (6 * np.sum(d**2) ** 1.5))

    def adjusted(level: float) -> float:
        z = z0 + norm.ppf(level)
        return float(norm.cdf(z0 + z / (1 - a * z)))

    out["bca"] = tuple(np.quantile(values, [adjusted(0.025), adjusted(0.975)]))
    t = m / sd
    lo, hi = np.quantile(t, [0.025, 0.975])
    out["bt_eq"] = (psi - hi * s, psi - lo * s)
    c = np.quantile(np.abs(t), 0.95)
    out["bt_sym"] = (psi - c * s, psi + c * s)
    out["kappa"] = float(np.mean(d**4) / np.mean(d**2) ** 2)
    out["skew"] = float(np.mean(d**3) / np.mean(d**2) ** 1.5)
    return out


def probe1_job(key: tuple[str, int]) -> list[dict[str, Any]]:
    estimator, rep = key
    rng = np.random.default_rng(10_000 + rep)
    frame, truth = pricing.draw_scenario(f"pricing_{estimator}_tau5", 5_000, rep)
    with _quiet():
        result = pricing.fit(
            frame,
            5,
            estimator=estimator,
            alpha=pricing.PRIMARY_ALPHA if estimator == "ltmle" else None,
        )
    rows = []
    for name in PRICING_NAMES:
        estimate = result[name]
        out = bootstrap_intervals(
            float(estimate.psi), np.asarray(estimate.influence_curve, float), rng
        )
        low, high = estimate.ci
        row = {
            "est": estimator,
            "rep": rep,
            "estimand": name,
            "truth": truth[name],
            "psi": float(estimate.psi),
            "se": float(estimate.std_error),
            "kappa": out["kappa"],
            "skew": out["skew"],
            "wald_lo": float(low),
            "wald_hi": float(high),
        }
        for kind in ("pct", "bca", "bt_eq", "bt_sym"):
            row[f"{kind}_lo"], row[f"{kind}_hi"] = (float(v) for v in out[kind])
        rows.append(row)
    return rows


def probe1(checkpoint: Path) -> pd.DataFrame:
    keys = [(est, rep) for rep in range(1_000) for est in ("ltmle", "sdr")]
    return resumable("probe1", keys, probe1_job, checkpoint)


# ------------------------------------------------------------------------- probe 2


def probe2_reps(probe1_rows: pd.DataFrame) -> list[int]:
    """The Wald misses of ``SDR``'s contrast and 30 covered draws (``random_state=1``)."""
    g = probe1_rows[(probe1_rows.est == "sdr") & (probe1_rows.estimand == DOWN_ATE)]
    g = g.sort_values("rep")
    missed = (g.wald_lo > g.truth) | (g.wald_hi < g.truth)
    covered = g[~missed].rep.sample(30, random_state=1).tolist()
    return sorted(g[missed].rep.tolist()) + sorted(covered)


def probe2_job(rep: int) -> list[dict[str, Any]]:
    frame, truth = pricing.draw_scenario("pricing_sdr_tau5", 5_000, rep)
    model = pricing_model(
        5, pricing.LABELS, "sdr", n_bootstrap=199, bootstrap_interval="bca", n_jobs=1
    )
    model.random_state = rep
    with _quiet():
        result = model.fit(frame, outcome="Y", **pricing.roles(5))
    rows = []
    for name in PRICING_NAMES:
        estimate = result[name]
        boot = estimate.bootstrap
        assert boot is not None
        rows.append(
            {
                "rep": rep,
                "estimand": name,
                "truth": truth[name],
                "psi": float(estimate.psi),
                "se": float(estimate.std_error),
                "lo": float(boot.ci[0]),
                "hi": float(boot.ci[1]),
                "failed": int(boot.n_failed),
            }
        )
    return rows


def probe2(checkpoint: Path, probe1_rows: pd.DataFrame) -> pd.DataFrame:
    return resumable("probe2", probe2_reps(probe1_rows), probe2_job, checkpoint, batch=10)


# ------------------------------------------------------------------------- probe 4


def probe4_job(rep: int) -> list[dict[str, Any]]:
    frame = law_b.SAMPLERS["B"](4_000, 5_000_000 + rep)
    rows = []
    for kind, pseudo in (
        ("projected", law_b.UnitRangePseudo()),
        ("unprojected", law_b.RichPseudo()),
    ):
        with _quiet():
            result = boundaries.fit(
                frame,
                "B",
                (boundaries.REFERENCE, "p4"),
                estimator="sdr",
                outcome="correct",
                pseudo_learner=pseudo,
            )
        estimate = result["ate_regimen[p4 vs natural]"]
        rows.append(
            {"rep": rep, "kind": kind, "psi": float(estimate.psi), "se": float(estimate.std_error)}
        )
    return rows


def probe4(checkpoint: Path) -> pd.DataFrame:
    return resumable("probe4", list(range(240)), probe4_job, checkpoint, batch=20)


# ------------------------------------------------------------------------- probe 5

#: Each configuration: ``(label, draws, variance learner)``.
PROBE5: dict[str, tuple[int, str]] = {
    "pricing_exact_d3": (1_000, "d3"),
    "pricing_exact_d5": (500, "d5"),
    "pricing_learned_d3": (300, "d3"),
    "lawb_correct_d3": (800, "d3"),
    "lawb_primary_d3": (300, "d3"),
}


def probe5_job(key: tuple[str, str, int]) -> list[dict[str, Any]]:
    config, estimator, rep = key
    learner = PROBE5[config][1]
    with _quiet(), variance_learner(learner):
        if config.startswith("pricing"):
            frame, truth = pricing.draw_scenario(f"pricing_{estimator}_tau5", 5_000, rep)
            classifier = law_p.QuadraticLogistic() if "learned" in config else None
            model = pricing_model(
                5,
                (pricing.REFERENCE, "price_down"),
                estimator,
                treatment_learner=classifier,
                variance="policy_weighted",
            )
            result = model.fit(frame, outcome="Y", **pricing.roles(5))
            names = PRICING_NAMES
        else:
            outcome = "correct" if "correct" in config else "primary"
            n = 4_000 if outcome == "correct" else 5_000
            frame = law_b.SAMPLERS["B"](
                n, 9_100_000 + rep + (0 if estimator == "ltmle" else 500_000)
            )
            result = boundaries_fit(frame, "p4", estimator, outcome)
            names = ("ey_regimen[p4]", "ate_regimen[p4 vs natural]")
            truth = {names[0]: law_b.truth("p4", "B"), names[1]: law_b.contrast("p4")}
    return [
        {
            "config": config,
            "est": estimator,
            "rep": rep,
            "estimand": name,
            "truth": truth[name],
            **policy_row(result[name]),
        }
        for name in names
    ]


def probe5(checkpoint: Path) -> pd.DataFrame:
    keys = [
        (config, est, rep)
        for config, (draws, _) in PROBE5.items()
        for rep in range(draws)
        for est in ("ltmle", "sdr")
    ]
    return resumable("probe5", keys, probe5_job, checkpoint, batch=40)


# ------------------------------------------------------------------------- probe 6

#: The held-out configurations: ``(law, plan, nodes, draws)``.
PROBE6: dict[str, tuple[str, str, int, int]] = {
    "lawP:price_down:3": ("P", "price_down", 3, 300),
    "lawP:price_up:5": ("P", "price_up", 5, 300),
    "lawB:p2": ("B", "p2", 2, 150),
    "lawB:p3": ("B", "p3", 2, 150),
}


def probe6_job(key: tuple[str, str, int]) -> list[dict[str, Any]]:
    config, estimator, rep = key
    kind, label, tau, _ = PROBE6[config]
    rows: dict[str, dict[str, Any]] = {}
    for learner in ("d3", "d5"):
        with _quiet(), variance_learner(learner):
            if kind == "P":
                frame = law_p.sample(5_000, 8_000_000 + 1_000 * tau + rep, tau)
                truth = pricing.truth_of(tau)
                model = pricing_model(
                    tau, (pricing.REFERENCE, label), estimator, variance="policy_weighted"
                )
                result = model.fit(frame, outcome="Y", **pricing.roles(tau))
                names = (f"ey_regimen[{label}]", f"ate_regimen[{label} vs {pricing.REFERENCE}]")
            else:
                frame = law_b.sample_b(4_000, 9_300_000 + 10_000 * int(label[1]) + rep)
                result = boundaries_fit(frame, label, estimator, "correct")
                names = (f"ey_regimen[{label}]", f"ate_regimen[{label} vs natural]")
                truth = {names[0]: law_b.truth(label, "B"), names[1]: law_b.contrast(label)}
        for name in names:
            got = policy_row(result[name])
            row = rows.setdefault(
                name,
                {
                    "config": config,
                    "est": estimator,
                    "rep": rep,
                    "estimand": name,
                    "truth": truth[name],
                    "psi": got["psi"],
                    "se_ic": got["se_ic"],
                },
            )
            row[f"se_pw_{learner}"] = got["se_pw"]
            row[f"psi_{learner}"] = got["psi"]
    return list(rows.values())


def probe6(checkpoint: Path) -> pd.DataFrame:
    keys = [
        (config, est, rep)
        for config, (*_, draws) in PROBE6.items()
        for rep in range(draws)
        for est in ("ltmle", "sdr")
    ]
    return resumable("probe6", keys, probe6_job, checkpoint, batch=20)


# ------------------------------------------------------------- the learner screen


def _recursion(
    designs: list[np.ndarray],
    moved: list[np.ndarray],
    targets: list[np.ndarray],
    ratios: list[np.ndarray],
    learner: str,
) -> float:
    """The diagonal form's backward recursion of Poisson regressions; the first node's mean."""
    carry: np.ndarray | None = None
    for t in range(len(designs) - 1, -1, -1):
        target = targets[t] if carry is None else targets[t] + carry
        model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.1,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
            **LEARNERS[learner],
        ).fit(designs[t], np.maximum(target, 0.0))
        carry = ratios[t] * model.predict(moved[t])
    assert carry is not None
    return float(np.mean(carry))


def screen_job(rep: int) -> list[dict[str, Any]]:
    frame, truth = pricing.draw_scenario("pricing_ltmle_tau5", 5_000, rep)
    with _quiet():
        result = pricing.fit(
            frame,
            5,
            (pricing.REFERENCE, "price_down"),
            estimator="ltmle",
            alpha=pricing.PRIMARY_ALPHA,
        )
    fit = result.fits["price_down"]
    w1 = frame.W1.to_numpy()
    designs, moved, ratios, gaps, gaps_moved = [], [], [], [], []
    for t in range(1, 6):
        a, level = frame[f"A{t}"].to_numpy(), frame[f"L{t}"].to_numpy()
        history = np.column_stack(
            [
                w1,
                *(frame[f"L{s}"].to_numpy() for s in range(1, t + 1)),
                *(frame[f"A{s}"].to_numpy() for s in range(1, t)),
            ]
        )
        gap = (
            np.zeros_like(a)
            if t == 1
            else frame[f"A{t - 1}"].to_numpy() - law_p.centre(frame[f"L{t - 1}"].to_numpy())
        )
        dose = law_p.down_map(a, level)
        designs.append(np.column_stack([a, history]))
        moved.append(np.column_stack([dose, history]))
        ratios.append(law_p.ratio("price_down", dose, law_p.price_mean(w1, level, gap), level))
        before = [
            frame[f"A{s}"].to_numpy() - law_p.centre(frame[f"L{s}"].to_numpy()) for s in range(1, t)
        ]
        gaps.append(np.column_stack([designs[-1], *before, a - law_p.centre(level)]))
        gaps_moved.append(np.column_stack([moved[-1], *before, dose - law_p.centre(level)]))
    residuals = [step.pseudo_outcome - step.targeted for step in fit.steps]
    squared = [e**2 for e in residuals]
    row: dict[str, Any] = {"rep": rep, "err": float(fit.psi_scaled - truth[DOWN])}
    for learner in ("d3", "d5", "d8"):
        row[learner] = _recursion(designs, moved, squared, ratios, learner)
        row[f"{learner}_gap"] = _recursion(gaps, gaps_moved, squared, ratios, learner)
    last = fit.steps[4]
    initial = [*squared[:4], last.initial * (1 - last.initial)]
    targeted = [*squared[:4], last.targeted * (1 - last.targeted)]
    row["q1q_initial_d5"] = _recursion(designs, moved, initial, ratios, "d5")
    row["q1q_targeted_d5"] = _recursion(designs, moved, targeted, ratios, "d5")
    return [row]


def screen(checkpoint: Path) -> pd.DataFrame:
    return resumable("screen", list(range(24)), screen_job, checkpoint, batch=6)


# ----------------------------------------------------------------------------- summaries


def clopper_pearson(covered: int, total: int, alpha: float = 0.01) -> tuple[float, float]:
    low = float(beta.ppf(alpha / 2, covered, total - covered + 1)) if covered > 0 else 0.0
    high = float(beta.ppf(1 - alpha / 2, covered + 1, total - covered)) if covered < total else 1.0
    return low, high


def _quartiles(se: np.ndarray, covered: np.ndarray) -> list[float]:
    quartile = pd.qcut(se, 4, labels=False, duplicates="drop")
    return [float(np.mean(covered[quartile == k])) for k in range(4)]


def interval_reading(err: np.ndarray, se: np.ndarray) -> dict[str, float]:
    """Coverage, the lowest SE quartile's coverage, corr(error, SE), SE ratio and CV of a Wald SE."""
    covered = np.abs(err) <= Z * se
    sd = float(np.std(err, ddof=1))
    return {
        "coverage": float(np.mean(covered)),
        "coverage_q1": _quartiles(se, covered)[0],
        "corr": float(np.corrcoef(err, se)[0, 1]),
        "se_ratio": float(np.mean(se)) / sd,
        "rms_ratio": float(np.sqrt(np.mean(se**2))) / sd,
        "cv": float(np.std(se, ddof=1) / np.mean(se)),
    }


def probe1_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Each estimator, estimand and interval kind: coverage with its 99% interval, the misses on
    each side, the width against Wald and the coverage by Wald SE quartile."""
    out = []
    for (est, name), g in rows.groupby(["est", "estimand"], sort=True):
        truth, psi, se = g.truth.to_numpy(), g.psi.to_numpy(), g.se.to_numpy()
        bounds = {
            kind: (g[f"{kind}_lo"].to_numpy(), g[f"{kind}_hi"].to_numpy())
            for kind in ("wald", "pct", "bca", "bt_eq", "bt_sym")
        }
        if name.startswith("ey"):
            centre, half = logit(psi), Z * se / (psi * (1 - psi))
            bounds["logit_wald"] = (expit(centre - half), expit(centre + half))
        else:
            centre, half = np.arctanh(psi), Z * se / (1 - psi**2)
            bounds["atanh_wald"] = (np.tanh(centre - half), np.tanh(centre + half))
        for kind, (low, high) in bounds.items():
            covered = (low <= truth) & (truth <= high)
            lo, hi = clopper_pearson(int(covered.sum()), covered.size)
            quart = _quartiles(se, covered)
            out.append(
                {
                    "est": est,
                    "estimand": name,
                    "interval": kind,
                    "replicates": covered.size,
                    "coverage": float(covered.mean()),
                    "coverage_lo": lo,
                    "coverage_hi": hi,
                    "miss_below": float(np.mean(low > truth)),
                    "miss_above": float(np.mean(high < truth)),
                    "width_ratio": float(np.mean(high - low) / (2 * Z * np.mean(se))),
                    **{f"coverage_q{k + 1}": quart[k] for k in range(4)},
                }
            )
    return pd.DataFrame(out)


def probe2_summary(rows: pd.DataFrame, probe1_rows: pd.DataFrame) -> pd.DataFrame:
    """The full-refit BCa on the Wald misses and the covered draws, and the coverage it implies."""
    out = []
    for name, g in rows.groupby("estimand", sort=True):
        first = probe1_rows[(probe1_rows.est == "sdr") & (probe1_rows.estimand == name)].set_index(
            "rep"
        )
        wald_covered = (first.wald_lo <= first.truth) & (first.truth <= first.wald_hi)
        g = g.set_index("rep")
        covered = (g.lo <= g.truth) & (g.truth <= g.hi)
        among_missed = covered[~wald_covered.loc[g.index]]
        among_covered = covered[wald_covered.loc[g.index]]
        wald = float(wald_covered.mean())
        out.append(
            {
                "estimand": name,
                "wald_coverage": wald,
                "missed": int(among_missed.size),
                "missed_covered": int(among_missed.sum()),
                "covered": int(among_covered.size),
                "covered_missed": int((~among_covered).sum()),
                "implied_coverage": wald * float(among_covered.mean())
                + (1 - wald) * float(among_missed.mean()),
                "failed_refits": int(g.failed.sum()),
            }
        )
    return pd.DataFrame(out)


def probe4_summary(rows: pd.DataFrame) -> pd.DataFrame:
    truth = law_b.contrast("p4")
    out = []
    for kind, g in rows.groupby("kind", sort=True):
        err = g.psi.to_numpy() - truth
        reading = interval_reading(err, g.se.to_numpy())
        out.append(
            {
                "kind": kind,
                "replicates": len(g),
                "bias_sd": float(err.mean() / err.std(ddof=1)),
                **reading,
            }
        )
    return pd.DataFrame(out)


def probe5_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Each configuration, estimator and estimand under the Wald SE, ``SE_pw`` alone and the max."""
    out = []
    for (config, est, name), g in rows.groupby(["config", "est", "estimand"], sort=True):
        err = (g.psi - g.truth).to_numpy()
        se_ic, se_pw = g.se_ic.to_numpy(), g.se_pw.to_numpy()
        usable = np.isfinite(se_pw)
        se_max = np.where(usable, np.maximum(se_ic, np.nan_to_num(se_pw)), se_ic)
        row: dict[str, Any] = {
            "config": config,
            "est": est,
            "estimand": name,
            "replicates": len(g),
            "nonpositive": int((~usable).sum()),
            "share_pw": float(np.mean(usable & (se_pw > se_ic))),
            "max_equals_reported": float(np.max(np.abs(se_max - g.se.to_numpy()))),
        }
        for source, se in (
            ("wald", se_ic),
            ("pw", np.where(usable, se_pw, se_ic)),
            ("max", se_max),
        ):
            row.update({f"{source}_{k}": v for k, v in interval_reading(err, se).items()})
        out.append(row)
    return pd.DataFrame(out)


def probe6_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (config, est, name), g in rows.groupby(["config", "est", "estimand"], sort=True):
        err = (g.psi - g.truth).to_numpy()
        se_ic = g.se_ic.to_numpy()
        row: dict[str, Any] = {"config": config, "est": est, "estimand": name, "replicates": len(g)}
        row.update({f"wald_{k}": v for k, v in interval_reading(err, se_ic).items()})
        for learner in ("d3", "d5"):
            se_pw = g[f"se_pw_{learner}"].to_numpy()
            usable = np.isfinite(se_pw)
            se_max = np.where(usable, np.maximum(se_ic, np.nan_to_num(se_pw)), se_ic)
            reading = interval_reading(err, se_max)
            row.update({f"{learner}_max_{k}": v for k, v in reading.items()})
            row[f"{learner}_pw_se_ratio"] = float(np.mean(se_pw[usable]) / np.std(err, ddof=1))
            row[f"{learner}_nonpositive"] = int((~usable).sum())
        out.append(row)
    return pd.DataFrame(out)


def screen_choice(summary: pd.DataFrame) -> str:
    """Section 6.2's rule on the preview: drop a learner with a cell outside ``[0.93, 1.25]``,
    take the smaller worst SE ratio, and give a tie within 0.01 to d3."""
    worst = {}
    for learner in ("d3", "d5"):
        ratios = summary[f"{learner}_max_se_ratio"]
        if ratios.min() < 0.93 or ratios.max() > 1.25:
            continue
        worst[learner] = float(ratios.max())
    if not worst:
        return "none"
    if len(worst) == 2 and worst["d5"] > worst["d3"] - 0.01:
        return "d3"
    return min(worst, key=worst.__getitem__)


def committed_error_variance() -> float:
    """:math:`n \\operatorname{Var}(\\text{error})` of the committed ``pricing_ltmle_tau5`` ``ey[price_down]`` rows."""
    rows = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "canonical" / "mtp_pricing" / "replicates.parquet"
    )
    rows = rows[
        (rows.implementation == "cleverly-mtp-pricing")
        & (rows.scenario == "pricing_ltmle_tau5")
        & (rows.estimand == DOWN)
    ]
    return float(np.var(rows.estimate - rows.truth, ddof=1) * 5_000)


def screen_summary(rows: pd.DataFrame) -> pd.DataFrame:
    truth = committed_error_variance()
    out = []
    for column in [c for c in rows.columns if c not in ("rep", "err")]:
        out.append(
            {
                "variant": column,
                "mean": float(rows[column].mean()),
                "corr_error": float(np.corrcoef(rows[column], rows.err)[0, 1]),
                "truth": truth,
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- driver

#: Each probe's rows file, its summary file and the number of fits per key.
PROBES = ("probe1", "probe2", "probe4", "probe5", "probe6", "screen")


def rows_path(name: str) -> Path:
    return Path(__file__).resolve().parent / f"{name}_rows.csv.gz"


def summary_of(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    """The summary of a probe's rows; ``probe2`` reads ``probe1``'s committed rows as well."""
    if name == "probe1":
        return probe1_summary(rows)
    if name == "probe2":
        return probe2_summary(rows, pd.read_csv(rows_path("probe1")))
    if name == "probe4":
        return probe4_summary(rows)
    if name == "probe5":
        return probe5_summary(rows)
    if name == "probe6":
        return probe6_summary(rows)
    return screen_summary(rows)


def run(name: str, checkpoint: Path) -> None:
    """Fit one probe (resuming from ``checkpoint``) and write its rows and its summary."""
    if name == "probe2":
        rows = probe2(checkpoint, pd.read_csv(rows_path("probe1")))
    else:
        rows = {
            "probe1": probe1,
            "probe4": probe4,
            "probe5": probe5,
            "probe6": probe6,
            "screen": screen,
        }[name](checkpoint)
    rows.to_csv(rows_path(name), index=False, float_format="%.12g")
    summary_of(name, rows).to_csv(
        rows_path(name).with_name(f"{name}.csv"), index=False, float_format="%.6g"
    )
