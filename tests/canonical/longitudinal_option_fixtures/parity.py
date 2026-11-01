"""The parity comparisons of the X33 fixtures: the R side's JSON against ``cleverly``.

Each ``compare_*`` function reads one fixture payload (the schema of the R script that writes
it), repeats the R side's computation with ``cleverly`` on the exported data, and returns the
findings.  An empty list is parity.

Tolerances.  The trim is a pmin against one quantile, so both sides compute the same floats
and 1e-15 allows only the last bit of the quantile's interpolation.  The pooled retention is
two logistic fits of the same rows, by R's IRLS and by ``cleverly``'s solver, so they agree to
the solvers' tolerance: the stand-in's gaps are 2.7e-15 per node and 3.3e-9 pooled, and 1e-6
keeps a 300-fold margin while a 1e-5 shift of one hazard fails.

===================  =====================================================  =============
fixture              quantity                                               tolerance
===================  =====================================================  =============
lmtp_history         mechanism and outcome-regression column sets           equal
lmtp_trim            threshold and trimmed matrix (fixed and lmtp's own)    1e-15
lmtp_trim            zero pattern of the per-node ratios                    exact
survtmle_pooled      retention at every at-risk row, per node and pooled    1e-6
===================  =====================================================  =============

Three conventions are compared as declared, not as equal:

* ``lmtp`` fits each outcome regression on every at-risk row, so its design holds the
  treatments of the window.  ``LTMLE`` fits the followers.  A static node's treatment is
  constant among them and is left out, and a rule node's assigned arm enters.
  :data:`RULE_ARM_LAG` states the one further difference: ``LTMLE`` starts the outcome's arm
  blocks one node before ``lmtp``, at ``t - k - 1``, as both start the mechanism's.
* ``survtmle`` sets ``G_dC = 1`` at the first visit and fits its censoring rows from visit 2.
  The comparison declares node 1's retention known, and a declared-known node stays out of
  ``LTMLE``'s pooled fit.  Without that declaration the pooled fit would also read node 1's
  rows, where nobody is censored, and retention would differ by up to 0.05.
* ``survtmle`` keeps an event row at ``t0`` in its censoring fit.  The fixture's law has no
  event at ``t0``, so the fitted rows are the at-risk rows of the pooled fit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from cleverly import LongitudinalTreatment
from cleverly.learners import GLM
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import DynamicRegimen, LongitudinalData
from cleverly.longitudinal import ratio_trim as ratio_trim_module
from cleverly.longitudinal.regimen import resolve_plans
from cleverly.longitudinal.sequential import fit_mechanism, outcome_design

__all__ = [
    "COMPARATORS",
    "HISTORY_ORDERS",
    "POOLED_RETENTION",
    "RULE_ARM_LAG",
    "TRIM_TOLERANCE",
    "compare_lmtp_history",
    "compare_lmtp_trim",
    "compare_survtmle_pooled",
]

Payload = Mapping[str, Any]

TRIM_TOLERANCE = 1e-15
POOLED_RETENTION = 1e-6
#: The orders the history fixture prints; ``"Inf"`` is ``markov_order=None``.
HISTORY_ORDERS = ("0", "1", "2", "Inf")
#: How many nodes before ``lmtp``'s first outcome arm block ``LTMLE``'s rule blocks start.
RULE_ARM_LAG = 1


def _gap(found: np.ndarray, expected: np.ndarray) -> float:
    found = np.asarray(found, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if found.shape != expected.shape:
        return float("inf")
    return float(np.max(np.abs(found - expected), initial=0.0))


def _matrix(columns: Sequence[Sequence[float | None]]) -> np.ndarray:
    return np.column_stack([np.asarray(column, dtype=float) for column in columns])


# ------------------------------------------------------------------------------ lmtp_history


def _history_panel(tau: int, n: int = 64, seed: int = 20263305) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns: dict[str, Any] = {"W": rng.normal(size=n)}
    for t in range(1, tau + 1):
        columns[f"L{t}"] = rng.normal(size=n)
        columns[f"A{t}"] = rng.integers(0, 2, size=n).astype(float)
    columns["Y"] = rng.normal(size=n)
    return pd.DataFrame(columns)


def _names(design: np.ndarray, sources: Mapping[str, np.ndarray]) -> list[str]:
    """The source column each design column equals, in design order."""
    named = []
    for column in np.asarray(design, dtype=float).T:
        matches = [name for name, values in sources.items() if np.array_equal(column, values)]
        named.append(matches[0] if len(matches) == 1 else f"<unmatched: {len(matches)} matches>")
    return named


def _sign_of(name: str) -> Callable[[Any], np.ndarray]:
    """A rule that treats where the node's covariate is positive."""

    def rule(history: Any) -> np.ndarray:
        return (np.asarray(history[name], dtype=float) > 0).astype(float)

    return rule


def _arm(name: str) -> bool:
    return name.startswith("A")


def compare_lmtp_history(payload: Payload) -> list[str]:
    """Compare ``LmtpVars$history`` with the windows of ``markov_order=``.

    The mechanism design at node ``t`` must hold exactly ``history("A", t)``, and the censoring
    design that plus ``A_t``.  The outcome design of a static plan must hold the covariates of
    ``history("L", t + 1)`` and no arm; that of a plan whose every node is a rule must hold
    those covariates, the assigned arms of ``lmtp``'s arm blocks, and the arm of node
    ``t - k - 1`` (:data:`RULE_ARM_LAG`) where it exists.

    Parameters
    ----------
    payload : mapping
        The parsed ``lmtp_history.json``.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    findings: list[str] = []
    tau = int(payload["tau"])
    orders = {str(item["order"]): item for item in payload["orders"]}
    if tuple(orders) != HISTORY_ORDERS:
        return [f"the payload prints orders {tuple(orders)}, not {HISTORY_ORDERS}"]
    panel = _history_panel(tau)
    rule = DynamicRegimen(
        "rule", tuple(_sign_of(f"L{t}") for t in range(1, tau + 1)), rule_kind="known"
    )
    static = DynamicRegimen("always", (1,) * tau)
    for label, item in orders.items():
        order = None if label == "Inf" else int(label)
        data = LongitudinalTreatment(
            outcome="Y",
            treatment=[f"A{t}" for t in range(1, tau + 1)],
            baseline=["W"],
            time_varying=[[f"L{t}"] for t in range(1, tau + 1)],
            markov_order=order,
        ).prepare(panel)
        sources = {name: panel[name].to_numpy(dtype=float) for name in panel.columns}
        (rule_plan,) = resolve_plans((rule,), data)
        (static_plan,) = resolve_plans((static,), data)
        assigned = {
            f"A{t}": (panel[f"L{t}"].to_numpy(dtype=float) > 0).astype(float)
            for t in range(1, tau + 1)
        }
        for node in item["nodes"]:
            t = int(node["node"])
            where = f"order {label}, node {t}"
            lmtp_mechanism = list(node["mechanism"])
            lmtp_outcome = list(node["outcome"])
            mechanism = _names(data.history_design(t), sources)
            if sorted(mechanism) != sorted(lmtp_mechanism):
                findings.append(f"{where}: mechanism columns {mechanism}, lmtp {lmtp_mechanism}")
            censoring = _names(data.history_design(t, include_current=True), sources)
            if sorted(censoring) != sorted([*lmtp_mechanism, f"A{t}"]):
                findings.append(
                    f"{where}: censoring columns {censoring}, lmtp {lmtp_mechanism} + A{t}"
                )
            covariates = sorted(name for name in lmtp_outcome if not _arm(name))
            found = _names(outcome_design(data, static_plan, t), sources)
            if sorted(found) != covariates:
                findings.append(f"{where}: static outcome columns {found}, expected {covariates}")
            rule_columns = _names(outcome_design(data, rule_plan, t), {**sources, **assigned})
            lmtp_arms = sorted(name for name in lmtp_outcome if _arm(name))
            # Without an order the window's covariates are the full history, which determines
            # every rule arm, so no arm block enters.
            expected_arms = set(lmtp_arms) if order is not None else set()
            if order is not None and t - order - RULE_ARM_LAG >= 1:
                expected_arms.add(f"A{t - order - RULE_ARM_LAG}")
            # A name in both sources is a covariate; an assigned arm matched as "A<s>".
            found_covariates = sorted(name for name in rule_columns if not _arm(name))
            found_arms = sorted(name for name in rule_columns if _arm(name))
            if found_covariates != covariates or found_arms != sorted(expected_arms):
                findings.append(
                    f"{where}: rule outcome columns {rule_columns}, expected covariates "
                    f"{covariates} and arms {sorted(expected_arms)} (lmtp arms {lmtp_arms})"
                )
    return findings


# ------------------------------------------------------------------------------ lmtp_trim


class _Plan:
    """A plan stand-in: :func:`~cleverly.longitudinal.ratio_trim.trim_ratios` reads its label."""

    label = "fixture"


def _trim(entries: np.ndarray, quantile: float) -> ratio_trim_module.RatioTrim:
    """``trim_ratios`` on a given per-node ratio matrix, the shipped code end to end."""
    with mock.patch.object(ratio_trim_module, "node_ratio_entries", return_value=entries):
        trims = ratio_trim_module.trim_ratios(None, (_Plan(),), None, quantile)  # type: ignore[arg-type]
    return trims[_Plan.label]


def _trim_findings(
    where: str, entries: np.ndarray, quantile: float, threshold: float | None, trimmed: np.ndarray
) -> list[str]:
    result = _trim(entries, quantile)
    findings = []
    if threshold is not None and abs(result.threshold - threshold) > TRIM_TOLERANCE:
        findings.append(f"{where}: threshold {result.threshold!r}, lmtp {threshold!r}")
    kept = np.where(result.trimmed, result.threshold, result.entries)
    gap = _gap(kept, trimmed)
    if gap > TRIM_TOLERANCE:
        findings.append(f"{where}: the trimmed matrix differs from lmtp's by {gap:.3g}")
    if not np.allclose(np.cumprod(kept, axis=1), result.cumulative, rtol=0, atol=1e-12):
        findings.append(f"{where}: the cumulative trimmed ratio is not the running product")
    return findings


def _trim_frame(fit: Payload) -> pd.DataFrame:
    frame = pd.DataFrame(
        {name: np.asarray(values, dtype=float) for name, values in fit["data"].items()}
    )
    return frame


def compare_lmtp_trim(payload: Payload) -> list[str]:
    """Compare ``lmtp``'s trim with ``trim_ratios``, and the per-node ratios' zero pattern.

    The fixed cases trim a matrix with zeros and ties at four quantiles.  The fit case trims
    the ratio matrix of an ``lmtp_tmle`` fit at ``.trim = 1`` and must give lmtp's matrix at
    ``.trim = q``.  The zero pattern of that untrimmed matrix must be the zero pattern of
    ``node_ratio_entries`` on the same sample, for the static plan that always treats.

    Parameters
    ----------
    payload : mapping
        The parsed ``lmtp_trim.json``.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    findings: list[str] = []
    fixed = payload["fixed"]
    matrix = _matrix(fixed["columns"])
    for case in fixed["cases"]:
        q = float(case["quantile"])
        expected = np.asarray(case["trimmed"], dtype=float).reshape(matrix.shape, order="F")
        findings += _trim_findings(f"fixed q={q}", matrix, q, float(case["threshold"]), expected)
    fit = payload["fit"]
    untrimmed = _matrix(fit["untrimmed"])
    trimmed = _matrix(fit["trimmed"])
    q = float(fit["trim"])
    findings += _trim_findings(f"lmtp fit q={q}", untrimmed, q, None, trimmed)
    if not np.any(trimmed < untrimmed):
        findings.append("the lmtp fit's trim moves no ratio; the case checks nothing")
    tau = int(fit["tau"])
    frame = _trim_frame(fit)
    data = LongitudinalTreatment(
        outcome="Y",
        treatment=[f"A{t}" for t in range(1, tau + 1)],
        baseline=["W"],
        time_varying=[[f"L{t}"] for t in range(1, tau + 1)],
        censoring=[f"C{t}" for t in range(1, tau + 1)],
    ).prepare(frame)
    plans = resolve_plans((DynamicRegimen("always", (1,) * tau),), data)
    mechanism = fit_mechanism(
        data,
        plans,
        treatment_learner=GLM(family="binomial"),
        censoring_learner=GLM(family="binomial"),
        folds=Folds.single(data.n),
    )
    entries = ratio_trim_module.node_ratio_entries(data, plans[0], mechanism)
    # With mtp = FALSE, lmtp multiplies each ratio by the rule indicator, so an off-plan or a
    # censored unit has an exact zero, and so does cleverly's entry.
    lmtp_zero = untrimmed == 0.0
    disagree = int(np.sum((entries == 0.0) != lmtp_zero))
    if disagree:
        findings.append(
            f"the per-node ratios' zero pattern differs from lmtp's in {disagree} of "
            f"{entries.size} entries"
        )
    if not lmtp_zero.any() or lmtp_zero.all():
        findings.append(
            "lmtp's ratio matrix has no zero or no positive entry; the pattern checks nothing"
        )
    return findings


# ------------------------------------------------------------------------------ survtmle_pooled


def _survival_data(payload: Payload) -> LongitudinalData:
    columns = payload["data"]
    frame = pd.DataFrame(
        {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    )
    visits = int(payload["visits"])
    data = LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=[float(t) for t in range(1, visits + 1)],
    )
    # No unit is censored before the first visit: survtmle's G_dC = 1 at t = 1.
    return data.with_known_mechanisms(
        censoring_probabilities={data.censoring_names[0]: np.ones(len(frame))}
    )


def compare_survtmle_pooled(payload: Payload) -> list[str]:
    """Compare survtmle's censoring fits with ``hazard_pooling="node"`` and ``"pooled"``.

    For each setting, the retention ``LTMLE`` fits at node ``t`` (2 to K) must be one minus
    survtmle's censoring hazard at visit ``t - 1``, at every row at risk of censoring there.
    The two R settings must differ, so that the law separates a pooled fit from a per-node one.

    Parameters
    ----------
    payload : mapping
        The parsed ``survtmle_pooled.json``.

    Returns
    -------
    list of str
        One sentence per finding.
    """
    findings: list[str] = []
    settings = {str(item["hazard_pooling"]): item for item in payload["settings"]}
    if set(settings) != {"node", "pooled"}:
        return ["the payload needs one 'node' and one 'pooled' setting"]
    times = np.asarray(payload["data"]["time"], dtype=float)
    events = np.asarray(payload["data"]["event"], dtype=float)
    visits = int(payload["visits"])
    if np.any((times == visits) & (events == 1)):
        findings.append("the sample has an event at t0, which survtmle's censoring rows keep")
    data = _survival_data(payload)
    plans = resolve_plans((DynamicRegimen("arm1", (1,)),), data)
    for pooling, setting in settings.items():
        if int(setting["rank"]) != len(setting["coefficients"]):
            findings.append(f"survtmle's {pooling!r} censoring fit is rank deficient")
        mechanism = fit_mechanism(
            data,
            plans,
            treatment_learner=GLM(family="binomial"),
            censoring_learner=GLM(family="binomial"),
            folds=Folds.single(data.n),
            hazard_pooling=pooling,
        )
        for t in range(2, visits + 1):
            at_risk = data.event_free_through(t - 1) & data.uncensored_through(t - 1)
            expected = 1.0 - np.asarray(setting["censoring_hazard"][t - 2], dtype=float)
            found = np.asarray(mechanism.censoring_observed[t - 1], dtype=float)
            gap = _gap(found[at_risk], expected[at_risk])
            if gap > POOLED_RETENTION:
                findings.append(
                    f"hazard_pooling={pooling!r}: retention at node {t} differs from "
                    f"survtmle's by {gap:.3g}"
                )
    spread = max(
        _gap(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        for a, b in zip(
            settings["node"]["censoring_hazard"],
            settings["pooled"]["censoring_hazard"],
            strict=True,
        )
    )
    if not spread > 1e-3:
        findings.append(
            f"the per-visit and pooled R fits differ by {spread:.3g}; the law does not separate them"
        )
    return findings


COMPARATORS: dict[str, Callable[[Payload], list[str]]] = {
    "lmtp_history": compare_lmtp_history,
    "lmtp_trim": compare_lmtp_trim,
    "survtmle_pooled": compare_survtmle_pooled,
}
