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
that this estimator answers for a static regimen, a binary treatment at every node, a
single end-of-study outcome and monotone censoring.  A dynamic rule, a survival outcome
with a time-varying event indicator, competing risks and a marginal structural model
over time each need their own derivation, and each is refused by name -- the keyword is
accepted and rejected with what the derivation would need, rather than arriving as an
``unexpected keyword argument``.  :data:`_REFUSED` is that table.
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
from .regimen import Regimen, resolve_regimens
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
        "a regime assigns an arm from W at one node; the longitudinal analogue is a "
        "dynamic rule d_t(H_t), whose followers are a covariate-dependent set at every "
        "node -- the next step, and it needs an oracle law of its own"
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
    "event": (
        "a survival outcome makes Y a node at every time point rather than one at the "
        "end, and the parameter a curve rather than a number"
    ),
    "competing": (
        "competing risks make the outcome a node at every time point with more than one "
        "absorbing state, and the parameter a set of cumulative incidences"
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
    regimens: tuple[Regimen, ...]
    reference: str
    n_folds: int
    g_bounds: tuple[float, float]
    q_bounds: tuple[float, float] | None
    alpha_sig: float
    random_state: int | None = None

    def describe(self) -> list[str]:
        plans = ", ".join(
            f"{regimen.label}=({'/'.join(str(int(v)) for v in regimen.values)})"
            for regimen in self.regimens
        )
        lines = [
            f"time points: {self.n_times}",
            f"outcome family: {self.family}",
            f"regimens: {plans}",
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
        """
        rows: dict[str, list[Any]] = {
            "regimen": [],
            "time": [],
            "n_followed": [],
            "max_weight": [],
            "effective_n": [],
            "epsilon": [],
            "converged": [],
        }
        for label, fit in self.fits.items():
            for step in fit.steps:
                weights = step.clever[step.trained_on]
                total = float(np.sum(weights))
                rows["regimen"].append(label)
                rows["time"].append(step.time)
                rows["n_followed"].append(step.n_trained)
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
        for label, fit in self.fits.items():
            lines.append(
                f"  {label}: {fit.steps[-1].n_trained} of {self.n} units followed it "
                f"throughout; max weight {fit.max_weight:.1f}, "
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
        outcome: str | None = None,
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
        reference = self._reference(regimens)

        folds = self._folds(prepared)
        scaler = self._scaler(prepared)
        bounds = resolve_g_bounds(self.g_bounds, float(prepared.n))

        mechanism = fit_mechanism(
            prepared,
            regimens,
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
        fits = {
            regimen.label: fit_regimen(
                prepared,
                regimen,
                mechanism,
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
            for regimen in regimens
        }

        config = LongitudinalConfig(
            family=prepared.family,
            n_times=prepared.n_times,
            regimens=regimens,
            reference=reference.label,
            n_folds=folds.n_folds,
            g_bounds=bounds,
            q_bounds=self.q_bounds,
            alpha_sig=self.alpha_sig,
            random_state=self.random_state,
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
        outcome: str | None,
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

    def _reference(self, regimens: Sequence[Regimen]) -> Regimen:
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
        reference: Regimen,
    ) -> dict[str, ParameterEstimate]:
        estimates: dict[str, ParameterEstimate] = {}
        for label, fit in fits.items():
            estimates[f"ey_regimen[{label}]"] = make_estimate(
                f"ey_regimen[{label}]",
                scaler.unscale_level(fit.psi_scaled),
                scaler.unscale_influence(fit.influence_curve_scaled),
                n=data.n,
                cluster=data.cluster,
                scale="level",
                alpha=self.alpha_sig,
            )
        base = fits[reference.label]
        for label, fit in fits.items():
            if label == reference.label:
                continue
            name = f"ate_regimen[{label} vs {reference.label}]"
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
    outcome: str,
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
