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
over time each need their own derivation, and each is refused by name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import FloatArray, GBounds, Learner
from ..estimators.base import format_table
from ..inference.cluster import influence_covariance
from ..inference.delta import delta_method
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..learners.crossfit import Folds, make_folds, resolve_n_folds
from ..learners.super_learner import resolve_learner
from ..provenance import Provenance, fingerprint_array
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from .data import LongitudinalData
from .regimen import Regimen, resolve_regimens
from .sequential import Mechanism, RegimenFit, fit_mechanism, fit_regimen

__all__ = ["LTMLE", "LongitudinalConfig", "LongitudinalResult", "ltmle"]


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
        return [
            f"time points: {self.n_times}",
            f"regimens: {plans}",
            f"reference: {self.reference}",
            f"cross-fitting: {self.n_folds} fold(s)",
            f"g_bounds: [{self.g_bounds[0]:.4g}, {self.g_bounds[1]:.4g}] at every node",
        ]


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

    # ---------------------------------------------------------------- reports

    def to_frame(self) -> Any:
        """One row per reported parameter, in the backend the data came from."""
        payload: dict[str, list[Any]] = {
            "parameter": [],
            "estimate": [],
            "std_error": [],
            "ci_low": [],
            "ci_high": [],
            "pvalue": [],
        }
        for name, estimate in self.estimates.items():
            low, high = estimate.ci
            payload["parameter"].append(name)
            payload["estimate"].append(estimate.psi)
            payload["std_error"].append(estimate.std_error)
            payload["ci_low"].append(low)
            payload["ci_high"].append(high)
            payload["pvalue"].append(estimate.pvalue)
        return self.data.frame_like(payload)

    def summary(self) -> str:
        """A printable report: the estimates, then the settings, then the leverage."""
        rows = []
        for name, estimate in self.estimates.items():
            low, high = estimate.ci
            rows.append(
                [
                    name,
                    f"{estimate.psi:.4f}",
                    f"{estimate.std_error:.4f}",
                    f"[{low:.4f}, {high:.4f}]",
                    f"{estimate.pvalue:.4f}",
                ]
            )
        table = format_table(["parameter", "estimate", "std. error", "95% CI", "p-value"], rows)
        lines = [
            f"Longitudinal TMLE ({self.data.n_times} time points, n = {self.n})",
            "",
            table,
            "",
            *(f"  {line}" for line in self.config.describe()),
        ]
        for label, fit in self.fits.items():
            lines.append(
                f"  {label}: {fit.steps[-1].n_trained} of {self.n} units followed it "
                f"throughout; max weight {fit.max_weight:.1f}, "
                f"effective n {fit.effective_n:.0f}"
            )
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
        alpha: float = 0.05,
        alpha_shrink: float = 0.9995,
        max_iter: int = 20,
        tol: float = 1e-10,
        random_state: int | None = None,
        n_jobs: int = 1,
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
        self.alpha_shrink = alpha_shrink
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs

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
    ) -> LongitudinalResult:
        """Fit on a wide dataframe, or on an already-built :class:`LongitudinalData`."""
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
        scaler = (
            OutcomeScaler.identity()
            if prepared.family == "binomial"
            else OutcomeScaler.from_outcome(
                prepared.outcome[prepared.uncensored_through(prepared.n_times)], self.q_bounds
            )
        )
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
                alpha=self.alpha_shrink,
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
            alpha_sig=self.alpha,
            random_state=self.random_state,
        )
        return LongitudinalResult(
            estimates=self._estimates(prepared, fits, scaler, reference),
            fits=fits,
            data=prepared,
            config=config,
            scaler=scaler,
            mechanism=mechanism,
            provenance=self._provenance(prepared, folds),
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

    def _provenance(self, data: LongitudinalData, folds: Folds) -> Provenance:
        """The same record a point-treatment fit carries, over the longitudinal nodes.

        The data fingerprint covers every node rather than three arrays, so two fits
        that differ only in a covariate measured at the second time point have
        different fingerprints -- which is the property the record exists for.
        """
        import platform
        import sys
        from datetime import datetime, timezone

        from .._version import __version__

        return Provenance(
            cleverly_version=__version__,
            python_version=sys.version.split()[0],
            platform=platform.platform(terse=True),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
                alpha=self.alpha,
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
                alpha=self.alpha,
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
