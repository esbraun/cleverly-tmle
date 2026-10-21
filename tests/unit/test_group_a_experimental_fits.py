"""The group-A refusal sites now return labelled results (RM42b section 4).

Each site below used to raise a ``CapabilityError`` for want of a study: a missing-outcome
cross-fit setting that no audit read (X53, X60), a bootstrap that no study measured (F2, X55),
and a weight-tilted missing-outcome DR-TMLE (X62). The code computes the requested quantity at
each of them, so the fit runs and every estimate carries the experimental label with the old
condition as its reason and the roadmap id as its evidence.

Per site the module checks four things:

* the preflight collects the condition before any learner, which :class:`NeverFit` enforces;
* the fit runs, and each estimate, the result and the bootstrap carry the label with the
  condition text and the roadmap id;
* a nonzero witness: the missingness depends on the treatment and a covariate, so the
  complete-case answer is several standard errors from the truth, while the point estimate sits
  within three standard errors of it;
* a deliberate mutation. Dropping the ``delta`` factor from the clever covariate moves the point
  away from the truth. The weight sites also drop the observation weights, the ATT and ATC site
  uses the ATE clever covariate, and the cluster sites use the iid standard error.

The probe of the plan (200 replicates at n = 1,000 per site) is a scratch study and does not
land here. This witness is a single large draw, so it states a bound in standard errors rather
than a coverage rate.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.estimators.targeting as targeting
from cleverly import CapabilityError, SplitPlan
from cleverly.data import CausalData
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import RefusalReason
from cleverly.learners import random_partition
from tests.unit._natural_course_support import NeverFit, never_fit_learners

ARM = ("ate", "ey")
NATURAL_COURSE = ("ey_obs", "par", "paf", "ate")
FOLDS = 5
SEED = 3
#: Rows of the witness draw, and of the cheaper draw that checks the label alone.
N_WITNESS = 16000
N_LABEL = 400
#: Rows of the truth draw. The truth carries a Monte Carlo error of about 1e-3, which is a
#: small fraction of the standard error of the witness fit.
N_TRUTH = 400_000


@dataclass(frozen=True)
class Law:
    """The options of the data-generating law."""

    clusters: bool = False
    continuous: bool = False
    randomized: bool = False
    #: A steeper treatment mechanism and a strong ``A * W1`` effect modification, so that the
    #: ATT and the ATE clever covariates give estimates several standard errors apart.
    steep: bool = False

    def key(self) -> tuple[bool, ...]:
        return (self.clusters, self.continuous, self.randomized, self.steep)


def draw(n: int, seed: int, law: Law) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Draw ``n`` rows: informative missingness, effect modification by ``W1``.

    The outcome logit holds ``A * W1`` and ``W1 ** 2`` terms that a main-effects outcome
    learner misses. The outcome is missing more often under treatment and for a high ``W1``, so
    the complete-case contrast is far from the truth. The treatment and missingness mechanisms
    are logistic and correctly specified.

    Returns
    -------
    tuple
        The observed frame, and the full-data potential outcomes with the weights.
    """
    rng = np.random.default_rng(seed)
    size = 10
    if law.clusters:
        count = n // size
        cluster = np.repeat(np.arange(count), size)
        shift_y, shift_a, shift_d = (
            np.repeat(rng.normal(0, s, count), size) for s in (1.1, 1.2, 0.5)
        )
    else:
        cluster = np.arange(n)
        shift_y = shift_a = shift_d = np.zeros(n)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    propensity = (
        np.full(n, 0.5)
        if law.randomized
        else expit((1.0 if law.steep else 0.6) * w1 - 0.4 * w2 + shift_a)
    )
    arm = (rng.random(n) < propensity).astype(float)

    def logit(a: float) -> np.ndarray:
        square, modify = (0.0, 2.0) if law.steep else (0.8, 1.5)
        return -0.5 + 0.8 * a + 0.7 * w1 + square * w1**2 - 0.4 * w2 + modify * a * w1 + shift_y

    if law.continuous:
        noise = rng.normal(0, 0.8, n)
        y0, y1 = expit(logit(0.0) + noise), expit(logit(1.0) + noise)
    else:
        u = rng.random(n)
        y0, y1 = (u < expit(logit(0.0))).astype(float), (u < expit(logit(1.0))).astype(float)
    outcome = np.where(arm == 1, y1, y0)
    seen = (rng.random(n) < expit(0.4 - 1.5 * arm - 1.4 * w1 + 0.5 * w2 + shift_d)).astype(float)
    weight = np.exp(-0.7 * w1)
    stratum = (w1 > 0).astype(float)
    frame = pd.DataFrame(
        {
            "Y": np.where(seen == 1, outcome, np.nan),
            "A": arm,
            "W1": w1,
            "W2": w2,
            "D": seen,
            "w": weight,
            "id": cluster.astype(float),
            "S": stratum,
        }
    )
    return frame, {"y0": y0, "y1": y1, "y": outcome, "a": arm, "w": weight, "s": stratum}


@cache
def truth(key: tuple[bool, ...]) -> Mapping[str, float]:
    """The target values, from a large full-data draw of the same law."""
    law = Law(*key)
    _, full = draw(N_TRUTH, 987_654_321, law)
    y0, y1, y, a, w, s = (full[name] for name in ("y0", "y1", "y", "a", "w", "s"))
    out = {
        "ate": float(np.mean(y1 - y0)),
        "ey[1]": float(np.mean(y1)),
        "att": float(np.mean((y1 - y0)[a == 1])),
        "atc": float(np.mean((y1 - y0)[a == 0])),
        "ey_obs": float(np.mean(y)),
        "par": float(np.mean(y) - np.mean(y0)),
    }
    out["paf"] = out["par"] / out["ey_obs"]
    out["w_ate"] = float(np.sum(w * (y1 - y0)) / np.sum(w))
    out["w_ey[1]"] = float(np.sum(w * y1) / np.sum(w))
    out["w_ey_obs"] = float(np.sum(w * y) / np.sum(w))
    out["w_par"] = out["w_ey_obs"] - float(np.sum(w * y0) / np.sum(w))
    out["w_paf"] = out["w_par"] / out["w_ey_obs"]
    for level in (0.0, 1.0):
        out[f"ate[S={level}]"] = float(np.mean((y1 - y0)[s == level]))
    return out


def complete_case(frame: pd.DataFrame, name: str, *, weighted: bool) -> float:
    """The naive answer: the contrast of the observed means, from the respondents alone."""
    seen = frame[frame["D"] == 1]

    def mean(rows: pd.DataFrame) -> float:
        return float(np.average(rows["Y"], weights=rows["w"] if weighted else None))

    treated, control = seen[seen["A"] == 1], seen[seen["A"] == 0]
    if name.endswith("ey[1]"):
        return mean(treated)
    if name.endswith("ey_obs"):
        return mean(seen)
    if name.endswith("paf"):
        return (mean(seen) - mean(control)) / mean(seen)
    if name.endswith("par"):
        return mean(seen) - mean(control)
    if "[S=" in name:
        level = float(name.split("S=")[1].rstrip("]"))
        stratum = seen[seen["S"] == level]
        return float(
            stratum[stratum["A"] == 1]["Y"].mean() - stratum[stratum["A"] == 0]["Y"].mean()
        )
    return mean(treated) - mean(control)


@dataclass(frozen=True)
class Site:
    """One converted refusal site.

    Parameters
    ----------
    id : str
        The site, as ``contract.setting``: ``arm`` is the arm-indexed contract, ``nc`` the
        natural-course contract.
    roadmap : str
        The roadmap id the old refusal named.
    fragment : str
        A fragment of the old condition. It is the reason of the label.
    settings : mapping
        Estimator keywords that break the rule. A callable value receives the row count.
    estimands : tuple of str
        The requested estimands.
    targets : tuple of str
        The estimates the witness checks. A name that starts ``w_`` is read from the estimate
        after the prefix, against the weighted truth.
    law : Law
        The data-generating options.
    weights, strata : bool
        Whether the fit declares ``weights=`` or ``strata=``.
    cluster : bool
        Whether the fit declares ``id=``.
    mutations : tuple of str
        The extra mutations the witness applies beside dropping ``delta``: ``weights``,
        ``iid`` or ``att``.
    n_bootstrap : int
        The bootstrap draws of the label check.
    kind : {"tmle", "drtmle"}
        The estimator class.
    n_witness : int
        The rows of the witness draw. A mutation whose effect is small needs more.
    """

    id: str
    roadmap: str
    fragment: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    estimands: tuple[str, ...] = ARM
    targets: tuple[str, ...] = ("ate", "ey[1]")
    law: Law = Law()
    weights: bool = False
    strata: bool = False
    cluster: bool = False
    mutations: tuple[str, ...] = ()
    n_bootstrap: int = 0
    kind: str = "tmle"
    n_witness: int = N_WITNESS


def _plan(n: int) -> SplitPlan:
    return SplitPlan.from_folds([random_partition(n, FOLDS, seed=SEED)])


SITES: tuple[Site, ...] = (
    Site(
        "arm.linear_fluctuation",
        "X60",
        "No audited result covers the linear fluctuation.",
        {"fluctuation": "linear"},
    ),
    Site(
        "arm.one_step",
        "X60",
        "No audited result covers the one-step update.",
        {"targeting": "one_step"},
    ),
    Site(
        "arm.weighted_fluctuation",
        "X60",
        "No audited result covers the weighted fluctuation.",
        {"target_weights": True},
    ),
    Site(
        "arm.fold_evaluated",
        "X53",
        "The fold-evaluated construction has no review of its fold plug-in and variance law.",
        {"cv_evaluation": True},
    ),
    Site(
        "arm.att_atc",
        "X60",
        "No audited result covers ['att', 'atc'].",
        {},
        ("att", "atc"),
        ("att", "atc"),
        law=Law(steep=True),
        mutations=("att",),
        n_witness=48_000,
    ),
    Site(
        "arm.split_plan",
        "X53",
        "No audit covers the balance and weighting of a supplied split plan.",
        {"split_plan": _plan},
    ),
    Site(
        "arm.repeats",
        "X53",
        "No audited result covers the median over repeated splits after CV-TMLE targeting.",
        {"repeats": 3},
    ),
    Site(
        "arm.weights",
        "X60",
        "The contract covers unweighted iid rows.",
        {},
        targets=("w_ate", "w_ey[1]"),
        weights=True,
        mutations=("weights",),
    ),
    Site(
        "arm.clusters",
        "X60",
        "The contract covers unweighted iid rows.",
        {},
        targets=("ey[1]", "ate"),
        law=Law(clusters=True),
        cluster=True,
        mutations=("iid",),
    ),
    Site(
        "arm.strata",
        "X53",
        "No audited result covers baseline strata.",
        {},
        targets=("ate", "ey[1]", "ate[S=1.0]"),
        strata=True,
    ),
    Site(
        "arm.bootstrap",
        "F2",
        "No audited result covers bootstrap inference.",
        {},
        n_bootstrap=30,
    ),
    Site(
        "nc.repeats",
        "X53",
        "No audited result covers the median over repeated splits for this fit.",
        {"repeats": 3},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
    ),
    Site(
        "nc.split_plan",
        "X53",
        "The cross-fitted estimator requires package-generated folds.",
        {"split_plan": _plan},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
    ),
    Site(
        "nc.linear_fluctuation",
        "X60",
        "No audited result covers a fluctuation other than the logistic one here.",
        {"fluctuation": "linear"},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
    ),
    Site(
        "nc.one_step",
        "X60",
        "No audited result covers a targeting algorithm other than the iterative one here.",
        {"targeting": "one_step"},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
    ),
    Site(
        "nc.weighted_fluctuation",
        "X60",
        "No audited result covers the weighted fluctuation here.",
        {"target_weights": True},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
    ),
    Site(
        "nc.bootstrap_crossfit",
        "F2",
        "No audited bootstrap result covers this fit.",
        {},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
        n_bootstrap=30,
    ),
    Site(
        "nc.bootstrap_in_sample",
        "X55",
        "No audited bootstrap result covers this fit.",
        {"cross_fit": False},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
        n_bootstrap=30,
    ),
    Site(
        "nc.weights",
        "X60",
        "The stacked contract covers unweighted iid rows.",
        {},
        NATURAL_COURSE,
        ("w_ey_obs", "w_par", "w_paf"),
        weights=True,
        mutations=("weights",),
    ),
    Site(
        "nc.clusters",
        "X60",
        "The stacked contract covers unweighted iid rows.",
        {},
        NATURAL_COURSE,
        ("ey_obs", "par", "paf"),
        law=Law(clusters=True),
        cluster=True,
        mutations=("iid",),
    ),
    Site(
        "nc.continuous_outcome",
        "X53",
        "The cross-fitted estimator requires a binary outcome.",
        {"q_bounds": (0.0, 1.0)},
        ("ey_obs", "par"),
        ("ey_obs", "par"),
        law=Law(continuous=True),
    ),
    Site(
        "drtmle.weights",
        "X62",
        "Missing-outcome DRTMLE is not certified for a weight-tilted target law.",
        {"randomized": True, "cross_fit": False},
        ("ate",),
        ("w_ate",),
        law=Law(randomized=True),
        weights=True,
        mutations=("weights",),
        kind="drtmle",
    ),
)
BY_ID = {site.id: site for site in SITES}


def learners(site: Site) -> dict[str, Any]:
    """The correctly specified mechanisms and a main-effects outcome learner."""
    outcome = LinearRegression() if site.law.continuous else LogisticRegression(max_iter=1000)
    roles: dict[str, Any] = {
        "outcome_learner": outcome,
        "treatment_learner": LogisticRegression(max_iter=1000),
        "missingness_learner": LogisticRegression(max_iter=1000),
    }
    if site.kind == "drtmle":
        roles["reduced_outcome_learner"] = LinearRegression()
        roles["reduced_treatment_learner"] = LinearRegression()
    return roles


def estimator(site: Site, n: int, roles: Mapping[str, Any], **extra: Any) -> Any:
    """The estimator that breaks the site's rule."""
    settings = {
        "cross_fit": True,
        "n_folds": FOLDS,
        "random_state": SEED,
        "simultaneous": False,
        **{name: value(n) if callable(value) else value for name, value in site.settings.items()},
        **extra,
    }
    if not settings["cross_fit"]:
        settings.pop("n_folds")
    cls = DRTMLE if site.kind == "drtmle" else TMLE
    return cls(**{**roles, "estimands": site.estimands, **settings})


def roles_of(site: Site, *, weights: bool = True) -> dict[str, Any]:
    """The keywords of ``fit`` and of ``CausalData.from_frame`` that the site declares."""
    covariates = ("W1", "W2", "S") if site.strata else ("W1", "W2")
    roles: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": covariates,
        "delta": "D",
    }
    if site.weights and weights:
        roles["weights"] = "w"
    if site.cluster:
        roles["id"] = "id"
    if site.strata:
        roles["strata"] = ["S"]
    return roles


@contextmanager
def mutated(kind: str | None) -> Iterator[None]:
    """Patch the clever covariate: drop the ``delta`` factor, or use the ATE covariate."""
    original = targeting.submodel_for

    def drop_delta(group: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("missingness") is not None:
            kwargs["missingness"] = np.ones_like(np.asarray(kwargs["missingness"], dtype=float))
        return original(group, *args, **kwargs)

    def ate_covariate(group: Any, treatment: Any, propensity: Any, **kwargs: Any) -> Any:
        submodel = original(group, treatment, propensity, **kwargs)
        if group not in ("att", "atc"):
            return submodel
        mean = original("mean", treatment, propensity, **kwargs)
        observed = mean.observed[:, [1]] - mean.observed[:, [0]]
        arms = {level: value[:, [1]] - value[:, [0]] for level, value in mean.arms.items()}
        return replace(submodel, observed=observed, arms=arms)

    patch: Callable[..., Any] | None = {"delta": drop_delta, "att": ate_covariate}.get(kind or "")
    if patch is None:
        yield
        return
    targeting.submodel_for = patch  # type: ignore[assignment]
    try:
        yield
    finally:
        targeting.submodel_for = original  # type: ignore[assignment]


def fit(site: Site, frame: pd.DataFrame, *, drop_weights: bool = False, **extra: Any) -> Any:
    n = len(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            estimator(site, n, learners(site), **extra)
            .fit(frame, **roles_of(site, weights=not drop_weights))
            .single()
        )


def read(estimates: Mapping[str, Any], name: str) -> Any:
    return estimates[name[2:] if name.startswith("w_") else name]


# --------------------------------------------------------------------------- the checks

IDS = [site.id for site in SITES]


@pytest.mark.parametrize("site", SITES, ids=IDS)
def test_the_preflight_collects_the_condition_before_any_learner(site: Site) -> None:
    frame, _ = draw(N_LABEL, 11, site.law)
    roles = roles_of(site)
    data = CausalData.from_frame(frame, **roles)
    est = estimator(site, N_LABEL, never_fit_learners(), **({"n_bootstrap": site.n_bootstrap}))
    conditions = est._preflight_fit_configuration(data).conditions
    matched = [c for c in conditions if site.fragment in c.condition and c.roadmap == site.roadmap]
    assert len(matched) == 1, [c.condition for c in conditions]
    assert matched[0].reason in (RefusalReason.IMPLEMENTATION_GAP, RefusalReason.INFERENCE_GAP)
    assert NeverFit.calls == 0


@pytest.mark.parametrize("site", SITES, ids=IDS)
def test_the_fit_runs_and_every_estimate_carries_the_label(site: Site) -> None:
    frame, _ = draw(N_LABEL, 12, site.law)
    result = fit(site, frame, **({"n_bootstrap": 4} if site.n_bootstrap else {}))
    assert result.estimates
    for name, estimate in result.estimates.items():
        support = estimate.support
        assert support.label == "experimental", name
        assert site.fragment in support.reason, (name, support.reason)
        assert site.roadmap in support.evidence, (name, support.evidence)
    summary = result.summary()
    assert "experimental" in summary
    assert site.fragment in summary
    if site.n_bootstrap:
        boots = [e.bootstrap for e in result.estimates.values() if e.bootstrap is not None]
        assert boots
        assert all(b.support.label == "experimental" for b in boots)
        assert all(site.fragment in b.support.reason for b in boots)


@pytest.mark.parametrize("site", SITES, ids=IDS)
def test_the_point_matches_the_truth_where_the_naive_answer_and_the_mutations_do_not(
    site: Site,
) -> None:
    """The nonzero witness: the factor each mutation drops is large on this law."""
    frame, _ = draw(site.n_witness, 21, site.law)
    expected = truth(site.law.key())
    extra = {"n_bootstrap": 0}
    result = fit(site, frame, **extra)
    errors = {}
    for name in site.targets:
        estimate = read(result.estimates, name)
        errors[name] = (estimate.psi - expected[name]) / estimate.std_error
        assert abs(errors[name]) <= 3.0, (name, errors[name])
    primary = site.targets[0]
    se = read(result.estimates, primary).std_error
    naive = complete_case(frame, primary, weighted=site.weights)
    assert abs(naive - expected[primary]) >= 3.0 * se, (naive, expected[primary], se)

    def departure(estimates: Mapping[str, Any]) -> float:
        return max(
            abs(read(estimates, name).psi - expected[name]) / se_of[name] for name in site.targets
        )

    se_of = {name: read(result.estimates, name).std_error for name in site.targets}
    with mutated("delta"):
        assert departure(fit(site, frame, **extra).estimates) >= 3.0
    if "att" in site.mutations:
        with mutated("att"):
            assert departure(fit(site, frame, **extra).estimates) >= 3.0
    if "weights" in site.mutations:
        unweighted = fit(site, frame, drop_weights=True, **extra)
        assert departure(unweighted.estimates) >= 3.0
    if "iid" in site.mutations:
        curve = np.asarray(read(result.estimates, primary).influence_curve, dtype=float)
        iid = float(np.std(curve, ddof=1) / np.sqrt(curve.size))
        assert iid < 0.95 * se, (primary, iid, se)


# --------------------------------------------------------------------------- the controls


def _refused(site_id: str, **extra: Any) -> CapabilityError:
    """Fit ``site_id``'s frame with never-fit learners and a setting that stays refused."""
    site = BY_ID[site_id]
    frame, _ = draw(N_LABEL, 13, site.law)
    with pytest.raises(CapabilityError) as caught:
        estimator(site, N_LABEL, never_fit_learners(), **extra).fit(frame, **roles_of(site))
    assert NeverFit.calls == 0
    return caught.value


@pytest.mark.parametrize(
    ("site_id", "extra", "fragment", "site_code"),
    [
        (
            "arm.linear_fluctuation",
            {"fluctuation": "logistic", "targeting_scheme": "fold"},
            "fold-specific targeting",
            "X53",
        ),
        (
            "nc.repeats",
            {"repeats": 1, "targeting_scheme": "fold"},
            "targeting_scheme='pooled'",
            "X53",
        ),
        ("nc.repeats", {"repeats": 1, "cv_evaluation": True}, "cv_evaluation=False", "X53"),
    ],
    ids=("arm.fold_targeting", "nc.fold_targeting", "nc.fold_evaluated"),
)
def test_a_site_with_a_misleading_interval_or_no_code_path_stays_a_refusal(
    site_id: str, extra: dict[str, Any], fragment: str, site_code: str
) -> None:
    error = _refused(site_id, **extra)
    assert fragment in str(error)
    assert error.refusal.roadmap == site_code


def test_an_experimental_condition_does_not_hide_a_later_refusal() -> None:
    """The preflight collects a condition and goes on, so the refusal that follows still fires."""
    error = _refused("arm.repeats", targeting_scheme="fold")
    assert "fold-specific targeting" in str(error)
    assert "median over repeated splits" not in str(error)


def test_a_result_saved_and_loaded_keeps_its_label_and_reason() -> None:
    from cleverly.estimators.serialize import dumps, loads

    site = BY_ID["arm.linear_fluctuation"]
    frame, _ = draw(N_LABEL, 14, site.law)
    result = fit(site, frame)
    loaded = loads(dumps(result))
    for name, estimate in loaded.estimates.items():
        assert estimate.support == result.estimates[name].support
        assert site.fragment in estimate.support.reason


def test_dropping_the_conditions_from_the_preflight_drops_the_reason() -> None:
    """The mutation control of the label test: without the conditions the reason is gone."""
    import importlib

    tmle_module = importlib.import_module("cleverly.estimators.tmle")

    site = BY_ID["arm.linear_fluctuation"]
    frame, _ = draw(N_LABEL, 15, site.law)
    original = tmle_module.classify_point_fit

    def without_conditions(*args: Any, **kwargs: Any) -> Any:
        kwargs["conditions"] = ()
        return original(*args, **kwargs)

    tmle_module.classify_point_fit = without_conditions  # type: ignore[assignment]
    try:
        result = fit(site, frame)
    finally:
        tmle_module.classify_point_fit = original  # type: ignore[assignment]
    assert all(site.fragment not in e.support.reason for e in result.estimates.values())


def test_a_clustered_collaborative_fit_with_missing_outcomes_stays_refused() -> None:
    from cleverly.estimators import CTMLE

    site = BY_ID["arm.clusters"]
    frame, _ = draw(N_LABEL, 16, site.law)
    settings = {
        "cross_fit": False,
        "estimands": ("ey",),
        "random_state": SEED,
        **never_fit_learners(),
    }
    with pytest.raises(CapabilityError, match="Drop id= from fit"):
        CTMLE(**settings).fit(frame, **roles_of(site))
    assert NeverFit.calls == 0
