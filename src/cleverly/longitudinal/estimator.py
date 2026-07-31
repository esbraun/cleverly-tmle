r"""Longitudinal TMLE: the mean outcome under a treatment regimen.

:class:`LTMLE` estimates :math:`E[Y_{\bar a}]` -- what the mean outcome would have been
had every unit followed the plan :math:`\bar a = (a_1, \ldots, a_T)` and stayed under
observation -- for each regimen the caller declares, plus the contrast of each against a
reference.  The estimator is the sequential-regression TMLE of van der Laan & Gruber
(2012); :mod:`cleverly.longitudinal.sequential` states the recursion, the submodel and
the influence function it solves.

What this buys over the point-treatment estimator is the whole reason the module exists:
a covariate measured between the two treatment decisions is a *confounder of the second
and a consequence of the first*, and no single adjustment set handles both roles.
Conditioning on it blocks the effect of :math:`A_1` that runs through it; leaving it out
confounds :math:`A_2`.  The sequential regression conditions on it at the node where it
is a confounder and integrates it out at the node where it is a mediator, which is what
the recursion is for.

The identification assumptions are the sequential versions of the familiar three, and
they are strictly stronger than at one time point:

* **sequential exchangeability** -- at every node, treatment is independent of the
  counterfactual outcome given the recorded history to that point.  Censoring is
  assumed non-informative on the same terms;
* **sequential positivity** -- every unit has a positive probability of the regimen's
  arm at every node given its history, and of remaining under observation.  The clever
  covariate divides by a *product* of :math:`2T` probabilities, so this is the
  assumption that bites first: :meth:`LongitudinalResult.diagnostics` reports the
  cumulative weight it produced;
* **consistency**, and no interference.

What is refused rather than approximated is listed in the README; the short version is
that this estimator answers for a regimen -- static or dynamic -- over a binary treatment
at every node, with a single end-of-study outcome and monotone censoring.  A survival
outcome with a time-varying event indicator, competing risks and a marginal structural
model over time each need their own derivation, and each is refused by name -- the
keyword is accepted and rejected with what the derivation would need, rather than
arriving as an ``unexpected keyword argument``.  :data:`_REFUSED` is that table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import FloatArray, GBounds, Learner
from ..estimators.base import _format_pvalue, format_table
from ..inference.cluster import influence_covariance
from ..inference.delta import delta_method
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..inference.multiplier import SimultaneousBands, simultaneous_bands
from ..learners.crossfit import Folds, make_folds, resolve_n_folds
from ..learners.super_learner import resolve_learner
from ..provenance import Provenance, fingerprint_array
from ..provenance import build as provenance_build
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from .data import LongitudinalData
from .regimen import DynamicRegimen, RegimenSpec, describe_plan, resolve_plans, resolve_regimens
from .sequential import Mechanism, RegimenFit, fit_mechanism, fit_regimen

__all__ = ["LTMLE", "LongitudinalConfig", "LongitudinalResult", "ltmle"]


#: What each point-treatment keyword would need before this estimator could take it.
#: The module docstring and the README both say these are refused *by name*; without
#: this table they were refused by absence, which is a ``TypeError`` naming no reason.
_REFUSED: dict[str, str] = {
    "weights": (
        "observation weights put a further per-unit factor in the clever covariate's "
        "denominator at every node, and the weighted efficient influence function has "
        "to be derived rather than re-indexed"
    ),
    "intermediate": (
        "a controlled direct effect fixes a mediator at one time point; over a sequence "
        "of nodes that is a different parameter with a different identification, not a "
        "further column"
    ),
    "msm": (
        "a working model over regimens summarises 2^T plans rather than K arms, so it "
        "needs its own weight function h(a-bar, V) and its own projection"
    ),
    "interventions": (
        "a regime is a density over the arms at one node, and a regimen is a plan "
        "across nodes -- so the longitudinal analogue of a rule d(W) is not a further "
        "parameter axis but a regimen whose nodes are rules. Declare it in regimens=, "
        "for example regimens={'treat once L2 rises': (0, lambda h: h['L2'] > 0)}"
    ),
    "shifts": (
        "a shift moves a continuous dose, and a longitudinal fit takes a binary "
        "treatment at every node"
    ),
    "incremental": (
        "a tilt of the mechanism is built out of g, so over time it needs the product "
        "of tilted mechanisms and a mechanism submodel at every node"
    ),
    "delta": (
        "an outcome missing for a reason other than censoring is a further node in the "
        "likelihood: encode it as a final censoring column, so that its probability is "
        "estimated and enters the cumulative product rather than being assumed one"
    ),
    # Kept as a key rather than deleted now that a survival outcome is supported:
    # dropping it would fall through to "unexpected keyword argument", which reads as a
    # misspelling rather than as a pointer to the keyword that does this.
    "event": (
        "a survival outcome is declared by the outcome columns themselves, not beside "
        "them. Pass one event indicator per time point -- outcome=['Y1', 'Y2', ...], in "
        "the same order as treatment= -- and the fit reports a cumulative risk at every "
        "horizon rather than one number"
    ),
    "competing": (
        "competing risks make the outcome a node at every time point with more than one "
        "absorbing state, and the parameter a set of cumulative incidences. A single "
        "absorbing event is supported: pass one event indicator per time point as "
        "outcome=[...]"
    ),
    "n_bootstrap": (
        "the targeted bootstrap resamples rows and refits, which needs a subset() on the "
        "longitudinal container and a re-run of the whole backward recursion per replicate"
    ),
    "cross_fit": (
        "pass n_folds=1 for in-sample nuisance fits; the fold count is the only control "
        "here, and the config says which of the two a fit used"
    ),
}


#: What separates a regimen from the horizon it is reported at, inside the brackets of a
#: parameter name and in the key of :attr:`LongitudinalResult.fits`.  Both are *composed*
#: from ``(label, horizon)`` rather than parsed back out of each other, so the two can
#: never drift; this constant exists so that a regimen label containing it can be refused
#: rather than producing a name nobody can read either way.
HORIZON_INFIX = " @ t="


def _index(label: str, horizon: int, survival: bool) -> str:
    """What goes inside the brackets of a parameter name."""
    return f"{label}{HORIZON_INFIX}{horizon}" if survival else label


def _fit_key(label: str, horizon: int, survival: bool) -> str:
    """The key one regimen's fit at one horizon is filed under.

    The same string :func:`_index` builds, and deliberately so: a terminal fit is keyed
    by its regimen label exactly as it always was, and a survival fit by the pair that
    indexes its parameter.
    """
    return _index(label, horizon, survival)


def refuse_unsupported(passed: Mapping[str, Any], *, where: str = "LTMLE") -> None:
    """Refuse a point-treatment keyword by name, saying what it would take to support it."""
    for name in passed:
        reason = _REFUSED.get(name)
        if reason is None:
            raise TypeError(
                f"{where} got an unexpected keyword argument {name!r}; see the "
                "longitudinal section of the README for what a longitudinal fit supports"
            )
        raise TypeError(f"{name}= is not supported by a longitudinal fit: {reason}")


@dataclass(frozen=True)
class LongitudinalConfig:
    """A snapshot of the settings a longitudinal fit actually used."""

    family: str
    n_times: int
    #: The outcome column(s): one name for an end-of-study outcome, one per node for a
    #: survival one.  Which it is decides what the fit reports, so it belongs in the
    #: record of what the fit did rather than being recoverable only from the names.
    outcome_names: tuple[str, ...]
    #: The horizons reported, ``(T,)`` on an end-of-study fit.
    horizons: tuple[int, ...]
    regimens: tuple[RegimenSpec, ...]
    reference: str
    n_folds: int
    g_bounds: tuple[float, float]
    q_bounds: tuple[float, float] | None
    alpha_sig: float
    random_state: int | None = None
    #: ``(label, digest)`` per regimen, digesting the ``(n, T)`` arms it assigned *this*
    #: sample.  A static plan is already stated in full by its ``1/0``; a rule is not, and
    #: two different rules would otherwise report identically and carry an identical
    #: provenance record -- ``1{L2 > 0}`` and ``1{L2 > 5}`` are different parameters.
    #: Digesting the resolved matrix rather than the callable is what makes this possible
    #: at all: a closure has no stable fingerprint, and the arms are what the fit used.
    plan_fingerprints: tuple[tuple[str, str], ...] = ()

    def describe(self) -> list[str]:
        plans = ", ".join(
            f"{regimen.label}=({describe_plan(regimen)})" for regimen in self.regimens
        )
        lines = [
            f"time points: {self.n_times}",
            f"outcome family: {self.family}",
            f"regimens: {plans}",
        ]
        if len(self.outcome_names) > 1:
            lines.insert(
                1,
                f"outcome: survival, event indicator at {', '.join(self.outcome_names)}",
            )
            reported = ", ".join(str(horizon) for horizon in self.horizons)
            lines.insert(2, f"horizons reported: t = {reported}")
        dynamic = {
            regimen.label for regimen in self.regimens if isinstance(regimen, DynamicRegimen)
        }
        if dynamic:
            # Worth a line rather than leaving "d" to be guessed at, because it changes
            # how the follower counts below should be read: a static regimen's followers
            # are whoever happened to receive that sequence, and a rule's are whoever the
            # rule would have treated -- a set this sample determines.
            lines.append(
                "  a 'd' is a rule d_t(H_t) read off [W, L_1, ..., L_t]; its followers "
                "are a covariate-dependent set, so the counts below describe this sample"
            )
            # Only for the regimens that have a rule in them, and only when one does:
            # printing a digest of a plan the line above already spells out in full would
            # add noise to every static report to say nothing.
            for label, digest in self.plan_fingerprints:
                if label in dynamic:
                    lines.append(f"  assigned arms, {label}: {digest}")
        lines += [
            f"reference: {self.reference}",
            # A single fold is not cross-fitting, and printing "1 fold(s)" reads as
            # though it were: the nuisances are then fitted on the rows they predict
            # for, which the reported variance does not account for.
            "cross-fitting: none -- nuisances fitted in sample"
            if self.n_folds <= 1
            else f"cross-fitting: {self.n_folds} fold(s)",
            # Both factors of every node's mechanism go through this bound, not just
            # the treatment one, and the difference matters wherever censoring is heavy.
            f"g_bounds: [{self.g_bounds[0]:.4g}, {self.g_bounds[1]:.4g}] on the treatment "
            "and censoring mechanism at every node",
        ]
        if self.q_bounds is not None:
            lines.append(f"q_bounds: [{self.q_bounds[0]:.4g}, {self.q_bounds[1]:.4g}]")
        lines.append(f"confidence level: {(1 - self.alpha_sig) * 100:g}%")
        if self.random_state is not None:
            lines.append(f"random_state: {self.random_state}")
        return lines


@dataclass(frozen=True)
class LongitudinalResult(Mapping[str, ParameterEstimate]):
    """Estimates under each regimen, with everything needed to do inference on them.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`, so ``result["ey_regimen[always]"]``
    and ``for name in result`` both work.
    """

    estimates: dict[str, ParameterEstimate]
    fits: dict[str, RegimenFit]
    data: LongitudinalData
    config: LongitudinalConfig
    scaler: OutcomeScaler
    mechanism: Mechanism
    provenance: Provenance
    simultaneous: SimultaneousBands | None = None

    # ------------------------------------------------------------- mapping API

    def __getitem__(self, name: str) -> ParameterEstimate:
        try:
            return self.estimates[name]
        except KeyError:
            raise KeyError(
                f"{name!r} was not estimated; this fit reports {list(self.estimates)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    # -------------------------------------------------------------- inference

    @property
    def n(self) -> int:
        return self.data.n

    def psi(self, name: str) -> float:
        return self[name].psi

    @property
    def influence_curves(self) -> dict[str, FloatArray]:
        return {name: estimate.influence_curve for name, estimate in self.estimates.items()}

    def covariance(self, names: Sequence[str] | None = None) -> FloatArray:
        """Joint covariance of the requested estimates, at the right independent unit."""
        chosen = self._names(names)
        curves = np.column_stack([self[name].influence_curve for name in chosen])
        return influence_covariance(curves, cluster=self.data.cluster)

    def contrast(
        self,
        function: Callable[[FloatArray], float],
        names: Sequence[str],
        *,
        name: str | None = None,
        scale: Scale = "difference",
        gradient: Callable[[FloatArray], FloatArray] | None = None,
    ) -> ParameterEstimate:
        """A smooth function of several regimens, with the delta method on the joint curve."""
        chosen = self._names(names)
        value, curve = delta_method(
            function,
            [self[key].psi for key in chosen],
            [self[key].influence_curve for key in chosen],
            gradient=gradient,
        )
        return make_estimate(
            name or f"contrast({', '.join(chosen)})",
            value,
            curve,
            n=self.n,
            cluster=self.data.cluster,
            scale=scale,
            alpha=self.config.alpha_sig,
        )

    def _names(self, names: Sequence[str] | None) -> tuple[str, ...]:
        chosen = tuple(self.estimates) if names is None else tuple(names)
        if not chosen:
            raise ValueError(f"no parameters selected; this fit reports {list(self)}")
        missing = [key for key in chosen if key not in self.estimates]
        if missing:
            raise KeyError(f"unknown parameter(s) {missing}; this fit reports {list(self)}")
        return chosen

    # ------------------------------------------------------------ diagnostics

    @property
    def converged(self) -> bool:
        """Whether every node's targeting step reached its tolerance."""
        return all(fit.converged for fit in self.fits.values())

    def diagnostics(self) -> Any:
        """A row per regimen and node: how much data it had, and how hard it leaned on it.

        ``n_followed`` is the number of units that followed the regimen and stayed under
        observation through the node -- the sample the regression there was fitted on.
        ``max_weight`` and ``effective_n`` describe the cumulative clever covariate, which
        is where sequential positivity shows up: they are properties of the *product* of
        the node-by-node mechanisms and can be alarming while every node looks fine.

        ``share_assigned_1`` is the fraction of the units at risk at that node whom the
        regimen would treat.  For a static regimen it is exactly ``0`` or ``1``, so the
        column doubles as a check on the plan the fit actually ran; for a dynamic rule it
        is the number a reader needs, since what a rule assigns is a property of the data
        rather than of the declaration and appears nowhere in the settings report.
        """
        survival = self.data.is_survival
        rows: dict[str, list[Any]] = {
            "regimen": [],
            **({"horizon": []} if survival else {}),
            "time": [],
            "n_followed": [],
            "share_assigned_1": [],
            "max_weight": [],
            "effective_n": [],
            "epsilon": [],
            "converged": [],
        }
        # Read off the fit's own fields rather than the key it is filed under: on a
        # survival fit that key is the regimen *and* the horizon, and a ``regimen``
        # column carrying both would be the one column here nobody could group by.
        for fit in self.fits.values():
            for step in fit.steps:
                weights = step.clever[step.trained_on]
                total = float(np.sum(weights))
                assigned = fit.assignment[step.at_risk, step.time - 1]
                rows["regimen"].append(fit.regimen.label)
                if survival:
                    rows["horizon"].append(fit.horizon)
                rows["time"].append(step.time)
                rows["n_followed"].append(step.n_trained)
                rows["share_assigned_1"].append(
                    float(np.mean(assigned == 1.0)) if assigned.size else float("nan")
                )
                rows["max_weight"].append(float(np.max(weights)) if weights.size else float("nan"))
                rows["effective_n"].append(
                    float(total**2 / np.sum(weights**2)) if total > 0 else 0.0
                )
                rows["epsilon"].append(float(step.fluctuation.epsilon[0]))
                rows["converged"].append(bool(step.fluctuation.converged))
        return self.data.frame_like(rows)

    # ------------------------------------------------- what this fit cannot do

    @property
    def sensitivity(self) -> Any:
        raise NotImplementedError(
            "the sensitivity suite is not available on a longitudinal fit. Every analysis "
            "in it re-targets against cached nuisance fits, and here that is not enough: "
            "g_bounds enters the pseudo-outcome of every earlier node through the "
            "recursion, so changing it changes what the earlier regressions were fitted "
            "to and the whole backward pass has to run again. Use result.diagnostics(), "
            "which reports the cumulative weight and effective n per regimen per node -- "
            "the leverage a longitudinal fit's positivity assumption actually produces"
        )

    @property
    def validation(self) -> Any:
        raise NotImplementedError(
            "the validation suite is not available on a longitudinal fit: it reads "
            "result.repeats and result.estimator, which a longitudinal result does not "
            "carry. The score equations it would check are already reported per node -- "
            "result.diagnostics() gives each node's epsilon and whether it converged, and "
            "result.fits[label].steps[i].fluctuation carries the score norms themselves"
        )

    def save(self, path: Any) -> None:
        raise NotImplementedError(
            "a longitudinal result cannot be serialised yet: cleverly.load rebuilds a "
            "TMLEResult from a CausalData, and the longitudinal container holds a node "
            "ordering that format has no place for. result.to_frame() and "
            "result.diagnostics() are the reportable pieces"
        )

    # ---------------------------------------------------------------- reports

    def curve(self, scale: str = "risk") -> Any:
        """The survival report as a curve: a row per parameter per horizon.

        ``scale="risk"`` reports what the fit estimated, the cumulative risk
        :math:`F(t)` and the risk difference between regimens at each horizon.
        ``scale="survival"`` reports :math:`S(t) = 1 - F(t)` instead.

        The map from one to the other is **not** one rule.  For a level,
        :math:`S = 1 - F`: the estimate is mirrored about a half and so is its interval.
        For a *contrast* it is :math:`S_a - S_b = -(F_a - F_b)`: the estimate is negated
        and the interval negated and swapped.  Applying ``1 - x`` to a risk difference
        would report ``1 - RD``, which is not a quantity, with an interval that would
        look perfectly reasonable -- so this branches on
        :attr:`~cleverly.inference.ParameterEstimate.scale` rather than on the caller
        getting it right.  The standard error is the same either way, both maps being
        linear with slope of modulus one.

        The ``time`` column lives here rather than on :meth:`to_frame`, which keeps the
        column names a point-treatment fit reports.
        """
        if scale not in ("risk", "survival"):
            raise ValueError(f"scale must be 'risk' or 'survival'; got {scale!r}")
        if not self.data.is_survival:
            raise ValueError(
                f"this fit has one end-of-study outcome ({self.data.outcome_name!r}), so "
                "its report is a number and not a curve. Pass one outcome column per "
                "time point -- outcome=[...] -- to estimate a cumulative risk at every "
                "horizon; result.to_frame() is the report for this fit"
            )
        rows: dict[str, list[Any]] = {
            "estimand": [],
            "regimen": [],
            "time": [],
            "psi": [],
            "std_err": [],
            "ci_lower": [],
            "ci_upper": [],
            "scale": [],
        }
        for name, estimate in self.estimates.items():
            index = name[name.index("[") + 1 : -1]
            label, _, horizon = index.rpartition(HORIZON_INFIX)
            low, high = estimate.ci
            if scale == "risk":
                psi, low, high = estimate.psi, low, high
            elif estimate.scale == "level":
                psi, low, high = 1.0 - estimate.psi, 1.0 - high, 1.0 - low
            else:
                psi, low, high = -estimate.psi, -high, -low
            rows["estimand"].append(name)
            rows["regimen"].append(label)
            rows["time"].append(int(horizon))
            rows["psi"].append(float(psi))
            rows["std_err"].append(estimate.std_error)
            rows["ci_lower"].append(float(low))
            rows["ci_upper"].append(float(high))
            rows["scale"].append(estimate.scale)
        return self.data.frame_like(rows)

    def to_frame(self) -> Any:
        """One row per reported parameter, in the backend the data came from.

        Built from :meth:`~cleverly.inference.ParameterEstimate.to_dict`, so the columns
        are the ones a point-treatment fit reports under the same names.  Two result
        objects in one library disagreeing on the name of every column is a worse cost
        than the one this saved.
        """
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        payload: dict[str, list[Any]] = {key: [row[key] for row in rows] for key in rows[0]}
        return self.data.frame_like(payload)

    def summary(self) -> str:
        """A printable report: the estimates, then the settings, then the leverage."""
        level = f"{(1 - self.config.alpha_sig) * 100:g}%"
        rows = []
        for name, estimate in self.estimates.items():
            low, high = estimate.ci
            rows.append(
                [
                    name,
                    f"{estimate.psi:.4f}",
                    f"{estimate.std_error:.4f}",
                    f"[{low:.4f}, {high:.4f}]",
                    _format_pvalue(estimate.pvalue),
                ]
            )
        table = format_table(
            ["parameter", "estimate", "std. error", f"{level} CI", "p-value"], rows
        )
        facts = list(self.config.describe())
        if self.data.cluster is not None:
            facts.append(
                f"clusters = {self.data.n_clusters} ({self.data.cluster_name}, "
                "cluster-robust variance)"
            )
        facts.extend(self.provenance.describe())
        lines = [
            f"Longitudinal TMLE ({self.data.n_times} time points, n = {self.n})",
            "",
            table,
            "",
            *(f"  {line}" for line in facts),
        ]
        # One line per regimen, from the fit that runs to the last node: "throughout"
        # means through the study, and a horizon-k fit stops at k, where the same
        # sentence would be false.  Per-horizon leverage is what diagnostics() is for.
        deepest = max(fit.horizon for fit in self.fits.values())
        for fit in self.fits.values():
            if fit.horizon != deepest:
                continue
            lines.append(
                f"  {fit.regimen.label}: {fit.steps[-1].n_trained} of {self.n} units "
                f"followed it throughout; max weight {fit.max_weight:.1f}, "
                f"effective n {fit.effective_n:.0f}"
            )
        if self.simultaneous is not None:
            lines.append("")
            lines.append(
                f"  simultaneous {level} bands (multiplier bootstrap, critical value "
                f"{self.simultaneous.critical_value:.3f} vs "
                f"{self.simultaneous.pointwise_critical_value:.3f} pointwise):"
            )
            for name, (low, high) in self.simultaneous.bands.items():
                lines.append(f"    {name}  [{low:.4f}, {high:.4f}]")
        if not self.converged:
            lines.append("  warning: at least one node's targeting step did not converge")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LongitudinalResult({', '.join(self.estimates)})"


class LTMLE:
    """Sequential-regression TMLE for a treatment regimen over time.

    Parameters
    ----------
    regimens:
        Mapping from label to plan: a sequence of ``T`` arms, or a single arm meaning
        that arm at every node.  ``{"always": 1, "never": 0}`` is the usual pair.
    reference:
        Which regimen contrasts are taken against; the first declared by default.
        Part of the estimand rather than a display setting -- ``ate_regimen[a vs b]``
        and ``ate_regimen[a vs c]`` are different parameters.
    outcome_learner, pseudo_learner, treatment_learner, censoring_learner:
        Learner specifications, as for :class:`~cleverly.TMLE`.  ``pseudo_learner``
        fits the intermediate regressions, whose outcome is a ``[0, 1]``-valued
        prediction rather than the outcome itself, and defaults to ``outcome_learner``'s
        library read as a regression.
    n_folds:
        Outer cross-fitting folds; one split serves every node and every regimen, so a
        unit is out of fold in all of them at once.
    g_bounds:
        Truncation applied to each mechanism factor *before* the cumulative product.
    alpha:
        Predicted probabilities are bounded into ``[1 - alpha, alpha]`` before the logit
        is taken, as for :class:`~cleverly.TMLE`.
    horizons:
        Which time points a **survival** fit reports the cumulative risk at.  ``None``
        reports all of them, which is the curve.  Each horizon is its own backward pass
        -- the pseudo-outcome carried back differs at every node, so nothing is shared
        between them but the mechanism -- and the cost is therefore ``T(T+1)/2``
        regressions per regimen rather than ``T``.  At two or three nodes that is not
        worth a keyword; over a monthly panel it is the difference between a fit and an
        afternoon, so name the horizons you will report.  Refused on a fit with one
        end-of-study outcome, where the only horizon is the end of the study.
    alpha_sig:
        Significance level for confidence intervals, as for :class:`~cleverly.TMLE`.
        The two ``alpha``\\ s mean what they mean there and not the other way round: this
        pair used to be spelled ``alpha`` / ``alpha_shrink`` here, which made
        ``LTMLE(alpha=0.9995)`` a silent 0.05 %-level interval.
    simultaneous, n_multiplier, multiplier_kind:
        Simultaneous confidence bands across the reported parameters, via the multiplier
        bootstrap.  A fit with several regimens reports several correlated parameters,
        which is what the bands are for; see :mod:`cleverly.inference.multiplier`.
    run_id:
        An identifier of your own, recorded on :attr:`LongitudinalResult.provenance`.
    """

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        outcome_learner: Learner | str | Sequence[Any] | None = None,
        pseudo_learner: Learner | str | Sequence[Any] | None = None,
        treatment_learner: Learner | str | Sequence[Any] | None = None,
        censoring_learner: Learner | str | Sequence[Any] | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        g_bounds: GBounds = "auto",
        q_bounds: tuple[float, float] | None = None,
        alpha: float = 0.9995,
        alpha_sig: float = 0.05,
        simultaneous: bool = True,
        n_multiplier: int = 2000,
        multiplier_kind: str = "rademacher",
        max_iter: int = 20,
        tol: float = 1e-10,
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        **refused: Any,
    ) -> None:
        self.regimens = regimens
        self.reference = reference
        self.horizons = horizons
        self.outcome_learner = outcome_learner
        self.pseudo_learner = pseudo_learner
        self.treatment_learner = treatment_learner
        self.censoring_learner = censoring_learner
        self.n_folds = n_folds
        self.learner_folds = learner_folds
        self.g_bounds = g_bounds
        self.q_bounds = q_bounds
        self.alpha = alpha
        self.alpha_sig = alpha_sig
        self.simultaneous = simultaneous
        self.n_multiplier = n_multiplier
        self.multiplier_kind = multiplier_kind
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.run_id = run_id
        self.n_jobs = n_jobs
        refuse_unsupported(refused)
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Refuse settings that would otherwise produce a report meaning something else.

        ``alpha_sig`` is the one worth spelling out: it and ``alpha`` are a hair apart in
        name and nowhere near in meaning, so a value that belongs to one and is passed as
        the other has to be refused rather than believed.
        """
        # Stricter than ``TMLE``, which takes the whole of (0, 1), and deliberately: the
        # half this refuses is empty of anything a caller means and is exactly where the
        # shrink lives, so refusing it turns the one swap that has no other symptom into
        # a message.  An interval covering less than half the time is not one either.
        if not 0.0 < self.alpha_sig < 0.5:
            raise ValueError(
                f"alpha_sig must lie in (0, 0.5); got {self.alpha_sig}. It is the "
                "significance level of the reported intervals -- alpha= is the "
                "probability shrink applied before the logit, and defaults to 0.9995"
            )
        if not 0.5 < self.alpha < 1.0:
            raise ValueError(
                f"alpha must lie in (0.5, 1); got {self.alpha}. Predicted probabilities "
                "are bounded into [1 - alpha, alpha] -- alpha_sig= is the significance "
                "level, and defaults to 0.05"
            )
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be at least 1; got {self.n_folds}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1; got {self.max_iter}")

    def fit(
        self,
        data: Any,
        *,
        outcome: str | Sequence[str] | None = None,
        treatment: Sequence[str] | None = None,
        baseline: Sequence[str] | None = None,
        time_varying: Sequence[Sequence[str]] | None = None,
        censoring: Sequence[str] | None = None,
        id: str | None = None,
        family: str = "auto",
        **refused: Any,
    ) -> LongitudinalResult:
        """Fit on a wide dataframe, or on an already-built :class:`LongitudinalData`."""
        refuse_unsupported(refused, where="LTMLE.fit")
        prepared = self._prepare(
            data,
            outcome=outcome,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            censoring=censoring,
            id=id,
            family=family,
        )
        regimens = resolve_regimens(self.regimens, prepared.n_times)
        if prepared.is_survival:
            clashing = sorted(r.label for r in regimens if HORIZON_INFIX in r.label)
            if clashing:
                raise ValueError(
                    f"regimen label(s) {clashing} contain {HORIZON_INFIX!r}, which is what "
                    "separates a regimen from the horizon it is reported at on a survival "
                    "fit. Two parameters would then share a name; rename the regimen"
                )
        reference = self._reference(regimens)
        # Every rule is called here and nowhere else, so a mask and the design the
        # mechanism was evaluated at cannot disagree about what the regimen assigned.
        plans = resolve_plans(regimens, prepared)

        folds = self._folds(prepared)
        scaler = self._scaler(prepared)
        bounds = resolve_g_bounds(self.g_bounds, float(prepared.n))

        mechanism = fit_mechanism(
            prepared,
            plans,
            treatment_learner=resolve_learner(
                self.treatment_learner,
                task="classification",
                n_folds=self.learner_folds,
                random_state=self.random_state,
            ),
            censoring_learner=resolve_learner(
                self.censoring_learner,
                task="classification",
                n_folds=self.learner_folds,
                random_state=self.random_state,
                fallback=self.treatment_learner,
            ),
            folds=folds,
            n_jobs=self.n_jobs,
        )

        outcome_task = "classification" if prepared.family == "binomial" else "regression"
        horizons = self._horizons(prepared)
        fits = {
            _fit_key(plan.label, horizon, prepared.is_survival): fit_regimen(
                prepared,
                plan,
                mechanism,
                horizon=horizon,
                outcome_learner=resolve_learner(
                    self.outcome_learner,
                    task=outcome_task,  # type: ignore[arg-type]
                    n_folds=self.learner_folds,
                    random_state=self.random_state,
                ),
                pseudo_learner=resolve_learner(
                    self.pseudo_learner,
                    task="regression",
                    n_folds=self.learner_folds,
                    random_state=self.random_state,
                    fallback=self.outcome_learner,
                ),
                folds=folds,
                scaler=scaler,
                g_bounds=bounds,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                n_jobs=self.n_jobs,
            )
            # Regimen-outer, horizon-inner, so the report reads down a regimen's curve
            # rather than across the regimens at each time.
            for plan in plans
            for horizon in horizons
        }

        config = LongitudinalConfig(
            family=prepared.family,
            n_times=prepared.n_times,
            outcome_names=(prepared.event_names or (prepared.outcome_name,)),
            horizons=horizons,
            regimens=regimens,
            reference=reference.label,
            n_folds=folds.n_folds,
            g_bounds=bounds,
            q_bounds=self.q_bounds,
            alpha_sig=self.alpha_sig,
            random_state=self.random_state,
            # From the matrix ``resolve_plans`` already built, so this is the assignment
            # the fit ran on rather than a second evaluation of the rules.
            plan_fingerprints=tuple((plan.label, fingerprint_array(plan.values)) for plan in plans),
        )
        estimates = self._estimates(prepared, fits, scaler, reference)
        return LongitudinalResult(
            estimates=estimates,
            fits=fits,
            data=prepared,
            config=config,
            scaler=scaler,
            mechanism=mechanism,
            provenance=self._provenance(prepared, folds),
            simultaneous=self._bands(estimates, prepared),
        )

    # ------------------------------------------------------------- internals

    def _prepare(
        self,
        data: Any,
        *,
        outcome: str | Sequence[str] | None,
        treatment: Sequence[str] | None,
        baseline: Sequence[str] | None,
        time_varying: Sequence[Sequence[str]] | None,
        censoring: Sequence[str] | None,
        id: str | None,
        family: str,
    ) -> LongitudinalData:
        if isinstance(data, LongitudinalData):
            declared = {
                "outcome": outcome,
                "treatment": treatment,
                "baseline": baseline,
                "time_varying": time_varying,
                "censoring": censoring,
                "id": id,
            }
            named = sorted(key for key, value in declared.items() if value is not None)
            if family != "auto":
                named.append("family")
            if named:
                raise ValueError(
                    f"{named} cannot be combined with a LongitudinalData input; the node "
                    "ordering and the outcome family are already fixed by the container, "
                    "and passing them again cannot change them. Pass them to "
                    "LongitudinalData.from_frame, which is where the columns are read"
                )
            return data
        if outcome is None or treatment is None or baseline is None:
            raise TypeError(
                "fitting from a dataframe needs outcome=, treatment= and baseline=; "
                "pass a LongitudinalData to supply them once"
            )
        return LongitudinalData.from_frame(
            data,
            outcome=outcome,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            censoring=censoring,
            id=id,
            family=family,
        )

    def _horizons(self, data: LongitudinalData) -> tuple[int, ...]:
        """Which nodes the fit reports a parameter at."""
        if not data.is_survival:
            if self.horizons is not None:
                raise ValueError(
                    "horizons= applies to a survival outcome, and this fit has one "
                    f"end-of-study outcome ({data.outcome_name!r}), whose only horizon is "
                    "the end of the study. Pass one outcome column per time point -- "
                    "outcome=[...] -- to report a curve"
                )
            return (data.n_times,)
        if self.horizons is None:
            return tuple(range(1, data.n_times + 1))
        wanted = tuple(int(horizon) for horizon in self.horizons)
        if not wanted:
            raise ValueError("horizons= is empty; a fit reporting no parameter is not one")
        outside = sorted({h for h in wanted if not 1 <= h <= data.n_times})
        if outside:
            raise ValueError(
                f"horizons= names {outside}, which is outside 1..{data.n_times}; a "
                "horizon is one of the fit's own time points"
            )
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"horizons= repeats a time point: {list(wanted)}")
        return tuple(sorted(wanted))

    def _bands(
        self, estimates: Mapping[str, ParameterEstimate], data: LongitudinalData
    ) -> SimultaneousBands | None:
        """Joint bands over the reported parameters, when there is more than one.

        A fit declaring ``R`` regimens reports ``R`` means and ``R - 1`` contrasts, all
        correlated through the same influence curves -- which is the situation the bands
        exist for, and the reason they are on by default here as they are on a
        point-treatment fit.  One regimen reports one parameter, and a band over one
        estimand is its pointwise interval.
        """
        if not self.simultaneous or len(estimates) < 2:
            return None
        return simultaneous_bands(
            estimates,
            alpha=self.alpha_sig,
            n_replicates=self.n_multiplier,
            kind=self.multiplier_kind,  # type: ignore[arg-type]
            random_state=self.random_state,
            cluster=data.cluster,
        )

    def _scaler(self, data: LongitudinalData) -> OutcomeScaler:
        """The outcome transformation, refusing a bound that cannot apply.

        The same rule as :meth:`cleverly.TMLE._scaler`: a binary outcome is already on
        the ``[0, 1]`` scale the recursion works in, so ``q_bounds=`` would have nothing
        to do -- and a setting with nothing to do is refused rather than dropped, since
        the caller who passed it believes it took effect.  ``family`` may have been
        inferred, so this can only be checked once the data is built.
        """
        if data.family == "binomial":
            if self.q_bounds is not None:
                raise ValueError("q_bounds does not apply to a binary outcome")
            return OutcomeScaler.identity()
        observed = data.outcome[data.uncensored_through(data.n_times)]
        return OutcomeScaler.from_outcome(observed, self.q_bounds)

    def _provenance(self, data: LongitudinalData, folds: Folds) -> Provenance:
        """The same record a point-treatment fit carries, over the longitudinal nodes.

        The data fingerprint covers every node rather than three arrays, so two fits
        that differ only in a covariate measured at the second time point have
        different fingerprints -- which is the property the record exists for.  That is
        the whole of the difference from :func:`cleverly.provenance.record`; the
        environment half is built by :func:`cleverly.provenance.build` so that the
        package versions and ``run_id`` this record is *for* cannot go missing here.

        What the *regimens* were is not in here but on
        :attr:`LongitudinalConfig.plan_fingerprints`, deliberately:
        :func:`cleverly.provenance.build` is shared with the point-treatment path, and a
        field only one estimator can fill would make the shared record answer a question
        half its callers have no answer to.
        """
        return provenance_build(
            n=data.n,
            n_covariates=len(data.baseline_names)
            + sum(len(names) for names in data.time_varying_names),
            n_clusters=None if data.cluster is None else data.n_clusters,
            data_fingerprint=fingerprint_array(
                data.outcome,
                data.treatment,
                data.uncensored.astype(float),
                data.baseline,
                *data.time_varying,
            ),
            fold_fingerprint=fingerprint_array(folds.assignment),
            random_state=self.random_state,
            run_id=self.run_id,
        )

    def _reference(self, regimens: Sequence[RegimenSpec]) -> RegimenSpec:
        if self.reference is None:
            return regimens[0]
        for regimen in regimens:
            if regimen.label == self.reference:
                return regimen
        raise KeyError(
            f"reference={self.reference!r} is not one of the declared regimens "
            f"{[regimen.label for regimen in regimens]}"
        )

    def _folds(self, data: LongitudinalData) -> Folds:
        if self.n_folds <= 1:
            return Folds.single(data.n)
        # Stratified on the first treatment node: it is the one every unit is at risk
        # for, so it is the only stratum a fold can be checked to carry.  The later
        # nodes are stratified only as far as the first one carries them.
        resolved = resolve_n_folds(self.n_folds, data.n, np.nan_to_num(data.treatment[:, 0]))
        return make_folds(
            data.n,
            resolved,
            stratify=np.nan_to_num(data.treatment[:, 0]),
            cluster=data.cluster,
            random_state=self.random_state,
        )

    def _estimates(
        self,
        data: LongitudinalData,
        fits: Mapping[str, RegimenFit],
        scaler: OutcomeScaler,
        reference: RegimenSpec,
    ) -> dict[str, ParameterEstimate]:
        survival = data.is_survival
        # ``ey_regimen`` is a mean of the outcome and ``risk_regimen`` a probability of
        # an event by a horizon.  E[Y_k] *is* that probability, so a single name would
        # not be wrong -- but the two come from different derivations, and a saved frame
        # or a coverage study's truth dict keyed by name is where that would stop being
        # a distinction without a difference.
        head = "risk_regimen" if survival else "ey_regimen"
        estimates: dict[str, ParameterEstimate] = {}
        for fit in fits.values():
            name = f"{head}[{_index(fit.regimen.label, fit.horizon, survival)}]"
            estimates[name] = make_estimate(
                name,
                scaler.unscale_level(fit.psi_scaled),
                scaler.unscale_influence(fit.influence_curve_scaled),
                n=data.n,
                cluster=data.cluster,
                scale="level",
                alpha=self.alpha_sig,
            )
        for fit in fits.values():
            if fit.regimen.label == reference.label:
                continue
            # The contrast is between the same two regimens at the *same* horizon; a
            # difference of risks at different horizons is not a treatment effect.
            base = fits[_fit_key(reference.label, fit.horizon, survival)]
            contrast = f"{fit.regimen.label} vs {reference.label}"
            name = f"ate_regimen[{_index(contrast, fit.horizon, survival)}]"
            estimates[name] = make_estimate(
                name,
                scaler.unscale_difference(fit.psi_scaled - base.psi_scaled),
                scaler.unscale_influence(fit.influence_curve_scaled - base.influence_curve_scaled),
                n=data.n,
                cluster=data.cluster,
                scale="difference",
                alpha=self.alpha_sig,
            )
        return estimates


def ltmle(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str],
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    family: str = "auto",
    **kwargs: Any,
) -> LongitudinalResult:
    """One-call longitudinal TMLE, mirroring :func:`cleverly.tmle`."""
    return LTMLE(regimens, **kwargs).fit(
        data,
        outcome=outcome,
        treatment=treatment,
        baseline=baseline,
        time_varying=time_varying,
        censoring=censoring,
        id=id,
        family=family,
    )
