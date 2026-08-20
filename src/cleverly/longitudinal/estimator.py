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

With a **survival** outcome -- one absorbing event indicator per node, declared by passing
``outcome=[...]`` rather than a single name -- the parameter is instead the cumulative
risk curve :math:`F_{\bar a}(t) = P(\text{event by } t)` at every horizon, reported with
the joint influence-curve matrix that makes a simultaneous band over the curve the natural
object.  It is the same recursion and the same clever covariate; what moves is which rows
each node's regression is fitted on.

``weights=`` names a column of observation weights and means what it means on a
point-treatment fit: the parameter is the declared one in the tilted population
:math:`dP_w = w\,dP/E[w]`, every node's nuisance is fitted by weighted loss, every node's
score equation is weighted, and the reported curve is :math:`(w/E[w])\,D^*(P_w)`.  See
:mod:`cleverly.data.weighting` for the statement and its limits, and
:mod:`cleverly.longitudinal.sequential` for what a weight is *not* -- a factor in the
clever covariate.

What is refused rather than approximated is listed in ``docs/methodology.md``, under
*Treatment given over time: the sequential regression*; the short version is
that this estimator answers for a regimen -- static or dynamic -- over a binary treatment
at every node, for one end-of-study outcome or one absorbing event per cause, with
monotone censoring.  Every point-treatment keyword it does not take is accepted and
rejected with what the derivation would need, rather than arriving as an ``unexpected
keyword argument``; :data:`_REFUSED` is that table.

``msm=`` declares a **working model over the regimens** and makes the report its
coefficients rather than a mean per plan -- the answer to the :math:`2^T` problem, and
what the applied literature reports for a grid of dynamic rules.  It is a projection and
not an assumption; :mod:`cleverly.longitudinal.msm` states it, and says why its
fluctuation is pooled across the regimens where the point-treatment one is not.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .._typing import CumulativeGBounds, FloatArray, Learner
from ..exceptions import PositivityWarning
from ..inference.cluster import influence_covariance
from ..inference.influence import ParameterEstimate, Scale, make_estimate
from ..inference.multiplier import SimultaneousBands, simultaneous_bands
from ..inference.results import (
    estimate_covariance,
    estimate_curves,
    select_estimates,
    smooth_contrast,
    sole_estimate,
)
from ..learners.crossfit import Folds, make_folds, resolve_n_folds
from ..learners.library import _validate_learner
from ..learners.super_learner import resolve_learner
from ..msm import MSM
from ..provenance import Provenance, fingerprint_array
from ..provenance import build as provenance_build
from ..utils.bounds import (
    DEFAULT_LTMLE_G_BOUNDS,
    OutcomeScaler,
    resolve_cumulative_g_bounds,
)
from ..utils.phases import PhaseProfile, phase, profile_phases
from ..utils.text import format_pvalue, format_table
from .data import LongitudinalData
from .msm import MSMRegimenFit, RegimenMSM, evaluate_regimen_msm, fit_regimens_msm
from .regimen import DynamicRegimen, RegimenSpec, describe_plan, resolve_plans, resolve_regimens
from .sequential import Mechanism, RegimenFit, fit_mechanism, fit_regimen

__all__ = ["LTMLE", "LongitudinalConfig", "LongitudinalResult", "ltmle"]

#: Match the point-treatment estimator's warning threshold.  The exact share remains
#: available in ``diagnostics()`` below this value; the warning is only the interruption.
_TRUNCATION_WARN_FRACTION = 0.05


#: What each point-treatment keyword would need before this estimator could take it.
#: The module docstring and ``docs/methodology.md`` both say these are refused *by name*;
#: without
#: this table they were refused by absence, which is a ``TypeError`` naming no reason.
_REFUSED: dict[str, str] = {
    # Kept as a key rather than deleted now that observation weights are supported, for the
    # reason ``event`` and ``competing`` are: a weight is a *column of the data*, so it is
    # declared where the columns are read, and falling through to "unexpected keyword
    # argument" here would read as a misspelling rather than as a pointer to the call that
    # takes it.  The refusal this replaced said weights put "a further per-unit factor in
    # the clever covariate's denominator at every node", which was simply wrong: a weight
    # tilts the population, and the denominator is the 2T mechanism factors and nothing
    # else.
    "weights": (
        "observation weights are a column of the data rather than a setting on the "
        "estimator, so they are declared where the columns are read: "
        "LTMLE(...).fit(frame, weights='w', ...), or "
        "LongitudinalData.from_frame(frame, weights='w', ...)"
    ),
    "intermediate": (
        "a controlled direct effect fixes a mediator at one time point; over a sequence "
        "of nodes that is a different parameter with a different identification, not a "
        "further column"
    ),
    # Kept as a key rather than deleted now that a working model over regimens is
    # supported, for the reason ``weights``, ``event`` and ``competing`` are: it is
    # declared on the *estimator* and not where the columns are read, so falling through
    # to "unexpected keyword argument" at ``.fit()`` would read as a misspelling rather
    # than as a pointer to the call that takes it.
    "msm": (
        "a working model is a setting on the estimator rather than a column of the data, "
        "so it is declared where the regimens are: LTMLE(regimens, msm=MSM(...)). Its "
        "design is handed (regimen_label, horizon, baseline_frame)"
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
    # Kept as a key now that competing risks are supported, for the reason ``event`` is:
    # dropping it would fall through to "unexpected keyword argument", which reads as a
    # misspelling rather than as a pointer to the keyword that does this.
    "competing": (
        "competing risks are declared by the outcome columns themselves, not beside them. "
        "Pass a mapping of cause to its indicator column per time point -- "
        "outcome={'relapse': ['R1', 'R2'], 'death': ['D1', 'D2']} -- and the fit reports "
        "a cumulative incidence per cause at every horizon"
    ),
    "eliminate": (
        "eliminating the competing events is a different estimand, not a setting on this "
        "one. What is reported here is the cause-specific cumulative incidence with the "
        "competing causes left alone, so a competing event is part of the history and "
        "enters the clever covariate's indicator. Removing it would make it an intervened "
        "node: a further factor per node in the denominator, and its own no-unmeasured-"
        "confounding and positivity assumptions to state"
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

#: What separates a regimen from the cause whose incidence is reported, on a fit that
#: declared competing risks.  Beside :data:`HORIZON_INFIX` and used the same way: composed
#: into a name, never parsed back out of one, and refused inside a label so that two
#: parameters cannot end up sharing a name.
CAUSE_INFIX = ", "


def _index(label: str, cause: str | None, horizon: int, survival: bool) -> str:
    """What goes inside the brackets of a parameter name."""
    if not survival:
        return label
    stem = label if cause is None else f"{label}{CAUSE_INFIX}{cause}"
    return f"{stem}{HORIZON_INFIX}{horizon}"


def _fit_key(label: str, cause: str | None, horizon: int, survival: bool) -> str:
    """The key one regimen's fit at one cause and horizon is filed under.

    The same string :func:`_index` builds, and deliberately so: a terminal fit is keyed
    by its regimen label exactly as it always was, and a survival fit by the tuple that
    indexes its parameter.
    """
    return _index(label, cause, horizon, survival)


def refuse_unsupported(passed: Mapping[str, Any], *, where: str = "LTMLE") -> None:
    """Refuse a point-treatment keyword by name, saying what it would take to support it."""
    for name in passed:
        reason = _REFUSED.get(name)
        if reason is None:
            raise TypeError(
                f"{where} got an unexpected keyword argument {name!r}; see the "
                "'Treatment given over time' section of docs/user-guide.md for what a "
                "longitudinal fit supports"
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
    #: The absorbing causes reported, empty unless the fit declared competing risks.
    #: Beside :attr:`outcome_names` for the same reason: which of the three parameters a
    #: fit answers is a statement it made, and belongs in the record of what it did.
    causes: tuple[str, ...]
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
    #: The working model's terms and link, ``None`` on a fit that declared none.
    msm_terms: tuple[str, ...] | None = None
    msm_link: str | None = None
    #: A digest of the *evaluated* design and weights.  A design is a closure and has no
    #: stable fingerprint; the arrays are what the fit was handed, which is the same
    #: reasoning :attr:`plan_fingerprints` rests on.
    msm_fingerprint: str | None = None

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
            if self.causes:
                lines[1] = (
                    "outcome: competing risks, absorbing causes "
                    f"{', '.join(self.causes)} at {', '.join(self.outcome_names)}"
                )
                lines.insert(3, f"causes reported: {', '.join(self.causes)}")
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
            # A working model has no reference regimen -- what the intercept is against is
            # whatever the design makes it -- so the working model stands where that line
            # would, rather than beside a field that names nothing.
            f"working model: {len(self.msm_terms)} term(s) "
            f"{', '.join(self.msm_terms)}, link={self.msm_link}"
            if self.msm_terms is not None
            else f"reference: {self.reference}",
            # A single fold is not cross-fitting, and printing "1 fold(s)" reads as
            # though it were: the nuisances are then fitted on the rows they predict
            # for, which the reported variance does not account for.
            "cross-fitting: none -- nuisances fitted in sample"
            if self.n_folds <= 1
            else f"cross-fitting: {self.n_folds} fold(s)",
            # Both factors enter the product before its prefix is bounded, as in ltmle's
            # CalcCumG.  Saying "mechanism at every node" used to conceal the order.
            f"g_bounds: fixed [{self.g_bounds[0]:.4g}, {self.g_bounds[1]:.4g}] on each "
            "cumulative treatment-and-censoring probability"
            + (
                " (package default; R ltmle-compatible heuristic -- inspect truncation diagnostics)"
                if self.g_bounds == DEFAULT_LTMLE_G_BOUNDS
                else ""
            ),
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
    #: ``name -> (regimen, cause, horizon)`` for every reported parameter, composed when
    #: the name was built.  Empty on an end-of-study fit, which has neither index.
    #: ``None`` on a working-model fit, whose parameters are indexed by *term*.
    parameter_index: dict[str, tuple[str, str | None, int]] | None = None
    #: The working model this fit projected onto, ``None`` if it declared none.
    msm: RegimenMSM | None = None
    #: One per cause, in the order the causes are reported.
    msm_fits: tuple[MSMRegimenFit, ...] = ()
    #: Causal-workflow metadata, present on every result created through CausalStudy.
    identified_effect: Any = None
    method: Any = None
    parameter_keys: dict[str, Any] = field(default_factory=dict)
    #: Assessment results keyed by operation and normalized arguments. Structured
    #: persistence writes this alongside the sequential artifacts it was computed from.
    assessment_cache: dict[str, Any] = field(default_factory=dict)

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
    def estimate(self) -> ParameterEstimate:
        """The sole estimate, refusing ambiguity on a multi-parameter result."""
        return sole_estimate(self.estimates)

    @property
    def n(self) -> int:
        return self.data.n

    def psi(self, name: str | None = None) -> float:
        return self.estimate.psi if name is None else self[name].psi

    @property
    def influence_curves(self) -> dict[str, FloatArray]:
        return estimate_curves(self.estimates)

    def covariance(self, names: Sequence[str] | None = None) -> FloatArray:
        """Joint covariance of the requested estimates, at the right independent unit."""
        return estimate_covariance(self.estimates, names, cluster=self.data.cluster)

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
        return smooth_contrast(
            self.estimates,
            function,
            names,
            n=self.n,
            cluster=self.data.cluster,
            alpha=self.config.alpha_sig,
            name=name,
            scale=scale,
            gradient=gradient,
        )

    def _names(self, names: Sequence[str] | None) -> tuple[str, ...]:
        return select_estimates(self.estimates, names)

    # ------------------------------------------------------------ diagnostics

    @property
    def converged(self) -> bool:
        """Whether every node's targeting step reached its tolerance."""
        return all(fit.converged for fit in self.fits.values())

    @property
    def diagnostics(self) -> Any:
        """Unified stagewise, support, nuisance, score, and refutation diagnostics."""
        from ..assessment import DiagnosticsFacade

        return DiagnosticsFacade(self)

    def validate(self) -> Any:
        """Run the inexpensive stagewise default battery without refitting."""
        from ..assessment import validate_result

        return validate_result(self)

    @property
    def replayability(self) -> Any:
        """Which assessment operations survive from the stored sequential artifacts."""
        from ..assessment import replayability

        return replayability(self)

    def incidence_total(self) -> Any:
        """Per regimen and horizon, the incidences summed over the causes.

        A unit leaves the risk set through exactly one cause or not at all, so the
        cumulative incidences and the event-free probability exhaust the mass:
        :math:`\\sum_j F_j(k) + S(k) = 1`.  That identity holds of the *parameters*.  It
        does **not** hold of these estimates, and the difference is not a bug to fix.
        Each cause is a separate backward pass -- its own regressions, its own
        fluctuation -- so nothing constrains the sum, and a total above one is possible.

        It is reported rather than removed, on the same reasoning as
        :attr:`~cleverly.sensitivity.positivity.PositivityReport.simplex_deviation`, where
        a multi-arm mechanism's row is deliberately not rescaled back onto the simplex.
        Renormalising would buy a coherent-looking table by moving every cause's estimate
        away from the one its own score equation solved, and would hide the thing worth
        seeing: a total far from one says the causes disagree about how much risk there
        was, which is a statement about the nuisance fits and not about the parameter.

        The ``total`` column carries a standard error because the sum is itself a
        parameter with an influence curve -- the sum of the causes' curves -- so
        ``excess`` can be read against it rather than eyeballed.
        """
        if not self.data.is_competing:
            raise ValueError(
                "this fit reports one parameter per regimen per horizon, so there is "
                "nothing to sum over. Declare competing risks -- outcome={cause: [...]} "
                "-- to estimate a cumulative incidence per cause"
            )
        rows: dict[str, list[Any]] = {
            "regimen": [],
            "time": [],
            "total": [],
            "std_err": [],
            "excess": [],
        }
        for regimen in self.config.regimens:
            for horizon in self.config.horizons:
                names = [
                    f"cif_regimen[{_index(regimen.label, cause, horizon, True)}]"
                    for cause in self.config.causes
                ]
                total = float(sum(self[name].psi for name in names))
                curve = np.sum(
                    np.column_stack([self[name].influence_curve for name in names]), axis=1
                )
                variance = influence_covariance(curve.reshape(-1, 1), cluster=self.data.cluster)
                rows["regimen"].append(regimen.label)
                rows["time"].append(int(horizon))
                rows["total"].append(total)
                rows["std_err"].append(float(np.sqrt(variance[0, 0] / self.data.n)))
                rows["excess"].append(max(0.0, total - 1.0))
        return self.data.frame_like(rows)

    # ------------------------------------------------- what this fit cannot do

    @property
    def sensitivity(self) -> Any:
        from ..assessment import SensitivityFacade

        return SensitivityFacade(self)

    def save(self, path: Any) -> Any:
        """Persist the complete fitted result to a trusted joblib artifact."""
        from ..estimators.serialize import save as _save

        return _save(self, path)

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
        if self.msm is not None:
            raise ValueError(
                "this fit reports the coefficients of a working model, and a coefficient "
                "has no horizon to index a curve by: the horizon is inside the design, "
                "which is what lets a coefficient be a trend across horizons rather than "
                "one number per horizon. S = 1 - F is not a map on a coefficient either. "
                "result.to_frame() is the report for this fit, and "
                "result.coefficients() the view that names what each one is"
            )
        if not self.data.is_survival:
            raise ValueError(
                f"this fit has one end-of-study outcome ({self.data.outcome_name!r}), so "
                "its report is a number and not a curve. Pass one outcome column per "
                "time point -- outcome=[...] -- to estimate a cumulative risk at every "
                "horizon; result.to_frame() is the report for this fit"
            )
        competing = self.data.is_competing
        rows: dict[str, list[Any]] = {
            "estimand": [],
            "regimen": [],
            **({"cause": []} if competing else {}),
            "time": [],
            "psi": [],
            "std_err": [],
            "ci_lower": [],
            "ci_upper": [],
            "scale": [],
        }
        # Read from the index composed when the name was built, never split back out of
        # it.  With a cause beside the horizon inside one pair of brackets there is no
        # split that is right in general: a regimen legitimately called ``"a, b"`` would
        # send ``rpartition`` to the wrong comma, and the row would be filed under a
        # regimen that does not exist rather than failing.
        index_of = self.parameter_index or {}
        for name, estimate in self.estimates.items():
            label, cause, horizon = index_of[name]
            low, high = estimate.ci
            if scale == "risk":
                psi, low, high = estimate.psi, low, high
            elif estimate.scale == "level":
                psi, low, high = 1.0 - estimate.psi, 1.0 - high, 1.0 - low
            else:
                psi, low, high = -estimate.psi, -high, -low
            rows["estimand"].append(name)
            rows["regimen"].append(label)
            if competing:
                rows["cause"].append(cause)
            rows["time"].append(int(horizon))
            rows["psi"].append(float(psi))
            rows["std_err"].append(estimate.std_error)
            rows["ci_lower"].append(float(low))
            rows["ci_upper"].append(float(high))
            rows["scale"].append(estimate.scale)
        return self.data.frame_like(rows)

    def coefficients(self, scale: str = "link") -> Any:
        """A working model's coefficients, on the link scale or exponentiated.

        The same view :meth:`cleverly.estimators.base.TMLEResult.coefficients` is at one
        time point, and it means the same things.  ``scale="link"`` reports
        :math:`\\hat\\beta` with a Wald interval on the scale the model is linear on;
        ``scale="ratio"`` reports :math:`e^{\\hat\\beta}` with the interval exponentiated
        from that scale.  **What the exponential is depends on the link** -- a risk ratio
        under ``"log"`` and an *odds* ratio under ``"logit"`` -- and the intercept is a
        third thing again, a baseline mean or a baseline odds rather than a ratio of
        anything, so the ``scale`` column names which each row is.  An intercept is found
        by its column being constant one, not by its name.

        Refused on an identity-link fit, where :math:`e^\\beta` of a risk difference is
        not a quantity, and on a fit with no working model.  Nothing is re-estimated.
        """
        if scale not in ("link", "ratio"):
            raise ValueError(f"scale must be 'link' or 'ratio'; got {scale!r}")
        if self.msm is None:
            raise ValueError(
                "this fit has no working model, so it reports a mean per regimen rather "
                "than coefficients; result.to_frame() is its report. Declare msm= to "
                "project those means onto a model whose coefficients are the parameters."
            )
        if scale == "ratio" and self.msm.link == "identity":
            raise ValueError(
                "an identity-link working model's coefficients are risk differences, and "
                "exp() of a difference is not a quantity anybody reports. Declare "
                "link='log' for coefficients that are log risk ratios, or link='logit' "
                "for log odds ratios, and this view exponentiates those."
            )
        ratio = "risk ratio" if self.msm.link == "log" else "odds ratio"
        rows: list[dict[str, Any]] = []
        for msm_fit in self.msm_fits:
            for column, term in enumerate(self.msm.terms):
                inside = term if msm_fit.cause is None else f"{term}{CAUSE_INFIX}{msm_fit.cause}"
                estimate = self.estimates[f"msm_regimen[{inside}]"]
                if scale == "link":
                    rows.append(estimate.to_dict())
                    continue
                exponentiated = replace(
                    estimate, psi=float(np.exp(estimate.psi)), log_psi=estimate.psi, scale="ratio"
                )
                row = exponentiated.to_dict()
                row["scale"] = (
                    "baseline" if bool(np.all(self.msm.design[:, :, column] == 1.0)) else ratio
                )
                rows.append(row)
        return self.data.frame_like({key: [row[key] for row in rows] for key in rows[0]})

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
                    format_pvalue(estimate.pvalue),
                ]
            )
        table = format_table(
            ["parameter", "estimate", "std. error", f"{level} CI", "p-value"], rows
        )
        facts = list(self.config.describe())
        if self.identified_effect is not None:
            facts.extend(self.identified_effect.summary_lines())
        if self.data.cluster is not None:
            facts.append(
                f"clusters = {self.data.n_clusters} ({self.data.cluster_name}, "
                "cluster-robust variance)"
            )
        if self.data.is_weighted:
            report = self.data.weight_report()
            facts.append(
                f"observation weights ({report.name or 'weights'}, "
                + ("estimated" if report.estimated else "fixed")
                + f"): effective n = {report.effective_n:.1f}, "
                f"design effect = {report.design_effect:.2f}"
            )
            facts.append(
                "estimand: the parameter in the weight-tilted population dP_w = w dP / E[w]; "
                "see result.data.weight_report()"
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
            truncated, truncated_time = self._max_truncated(fit)
            lines.append(
                f"  {fit.regimen.label}: {fit.steps[-1].n_trained} of {self.n} units "
                f"followed it throughout; max weight {fit.max_weight:.1f}, "
                f"effective n {fit.effective_n:.0f}; max truncated share "
                f"{truncated:.1%} at t={truncated_time}"
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

    @staticmethod
    def _max_truncated(fit: RegimenFit) -> tuple[float, int]:
        """Largest on-score truncation share and its earliest node."""
        shares = []
        for step in fit.steps:
            raw = fit.cumulative_unbounded[:, step.time - 1][step.trained_on]
            bounded = fit.cumulative[:, step.time - 1][step.trained_on]
            shares.append((float(np.mean(raw != bounded)) if raw.size else 0.0, step.time))
        return max(shares, key=lambda item: (item[0], -item[1]))


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
        Fixed truncation applied to each cumulative treatment-and-censoring probability,
        after multiplying the raw node factors.  The default is the explicit pair
        ``(0.01, 1.0)``, R ``ltmle``'s heuristic convention.  It is not an automatic,
        sample-size-dependent, or follow-up-depth-dependent selection procedure.
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
        msm: MSM | None = None,
        outcome_learner: Learner | None = None,
        pseudo_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        g_bounds: CumulativeGBounds = DEFAULT_LTMLE_G_BOUNDS,
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
        _validate_learner(outcome_learner, "outcome_learner")
        _validate_learner(pseudo_learner, "pseudo_learner")
        _validate_learner(treatment_learner, "treatment_learner")
        _validate_learner(censoring_learner, "censoring_learner")
        self.regimens = regimens
        self.reference = reference
        self.horizons = horizons
        self.msm = msm
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
        if self.msm is not None and not isinstance(self.msm, MSM):
            raise TypeError(
                f"msm= must be a cleverly.msm.MSM; got {type(self.msm).__name__}. A "
                "working model over regimens is declared with the same class a working "
                "model over arms is, and its design is handed "
                "(regimen_label, horizon, baseline_frame)"
            )
        if self.msm is not None and self.reference is not None:
            raise ValueError(
                "msm= and reference= cannot be combined. reference= names the regimen "
                "every contrast is taken against, and a working model reports "
                "coefficients rather than contrasts -- which regimen is the baseline is "
                "decided by the design you gave msm=, and an intercept is whatever that "
                "design makes it. A difference of two coefficients comes from "
                "result.contrast()."
            )

    @staticmethod
    def profile_phases() -> AbstractContextManager[PhaseProfile]:
        """Collect wall-clock timings of a fit's phases, for a benchmark or a profile.

        .. code-block:: python

            with LTMLE.profile_phases() as profile:
                estimator.fit(data)
            print(profile.summary())

        A diagnostic, not part of a result: the shares depend on the box, the learner
        library and the node count, so recording one on a fitted object would invite it
        being compared across runs that are not comparable.  The phases are
        ``mechanism_fit``, ``mask_construction``, ``pseudo_outcome``,
        ``outcome_learner_fit``, ``clever_covariate``, ``fluctuation``,
        ``influence_curve`` and ``inference``.

        Off unless this is entered, and cheap when off -- see
        :mod:`cleverly.utils.phases`.  It is a static method because it installs a
        thread-global collector rather than instrumenting one estimator.
        """
        return profile_phases()

    def fit(
        self,
        data: Any,
        *,
        outcome: str | Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        treatment: Sequence[str] | None = None,
        baseline: Sequence[str] | None = None,
        time_varying: Sequence[Sequence[str]] | None = None,
        censoring: Sequence[str] | None = None,
        id: str | None = None,
        weights: str | None = None,
        weights_type: str = "probability",
        weights_estimated: bool = False,
        family: str = "auto",
        **refused: Any,
    ) -> LongitudinalResult:
        """Fit on a wide dataframe, or on an already-built :class:`LongitudinalData`.

        ``weights=`` names a column of observation weights, read exactly as
        :meth:`cleverly.TMLE.fit` reads them: the estimand becomes the declared parameter
        in the tilted population ``dP_w = w dP / E[w]``, every node's nuisance is fitted by
        weighted loss, every node's score equation is weighted, and the reported curve is
        ``(w / E[w]) D*(P_w)``.  See :mod:`cleverly.data.weighting`.
        """
        refuse_unsupported(refused, where="LTMLE.fit")
        prepared = self._prepare(
            data,
            outcome=outcome,
            treatment=treatment,
            baseline=baseline,
            time_varying=time_varying,
            censoring=censoring,
            id=id,
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            family=family,
        )
        regimens = resolve_regimens(self.regimens, prepared.n_times)
        if prepared.is_survival:
            infixes = (HORIZON_INFIX,) + ((CAUSE_INFIX,) if prepared.is_competing else ())
            for infix in infixes:
                clashing = sorted(r.label for r in regimens if infix in r.label)
                if clashing:
                    raise ValueError(
                        f"regimen label(s) {clashing} contain {infix!r}, which is what "
                        "separates a regimen from the cause and horizon it is reported at. "
                        "Two parameters would then share a name; rename the regimen"
                    )
                bad_causes = sorted(c for c in prepared.cause_labels if infix in c)
                if bad_causes:
                    raise ValueError(
                        f"cause label(s) {bad_causes} contain {infix!r}, which is what "
                        "separates a cause from the regimen and horizon it is reported at. "
                        "Two parameters would then share a name; rename the cause"
                    )
        reference = self._reference(regimens)
        # Every rule is called here and nowhere else, so a mask and the design the
        # mechanism was evaluated at cannot disagree about what the regimen assigned.
        plans = resolve_plans(regimens, prepared)

        folds = self._folds(prepared)
        scaler = self._scaler(prepared)
        # A cumulative path probability is not a point-treatment propensity.  There is no
        # automatic rule here: the constructor default is the visible fixed R ``ltmle``
        # heuristic, and every other accepted value is an explicit scalar or pair.
        bounds = resolve_cumulative_g_bounds(self.g_bounds)

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
        # ``None`` on a single-event or end-of-study fit, so the comprehension below is
        # one loop deeper for every fit and a second code path for none.
        causes: tuple[str | None, ...] = prepared.cause_labels or (None,)
        recursion = {
            "outcome_learner": resolve_learner(
                self.outcome_learner,
                task=outcome_task,  # type: ignore[arg-type]
                n_folds=self.learner_folds,
                random_state=self.random_state,
            ),
            "pseudo_learner": resolve_learner(
                self.pseudo_learner,
                task="regression",
                n_folds=self.learner_folds,
                random_state=self.random_state,
                fallback=self.outcome_learner,
            ),
            "folds": folds,
            "scaler": scaler,
            "g_bounds": bounds,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_jobs": self.n_jobs,
        }
        model: RegimenMSM | None = None
        msm_fits: tuple[MSMRegimenFit, ...] = ()
        if self.msm is None:
            fits = {
                _fit_key(plan.label, cause, horizon, prepared.is_survival): fit_regimen(
                    prepared, plan, mechanism, horizon=horizon, cause=cause, **recursion
                )
                # Regimen-outer, then cause, then horizon, so the report reads down one
                # curve at a time rather than across the regimens at each time.
                for plan in plans
                for cause in causes
                for horizon in horizons
            }
        else:
            # One projection per cause -- a cause is a different estimand, not a further
            # column of the design -- over a grid whose cells cross the regimens with the
            # horizons.  The per-cell fits come back so that ``diagnostics()`` can report
            # the leverage and risk sets, which are questions about a regimen and are the
            # same questions whether or not a working model summarises it.
            model = evaluate_regimen_msm(self.msm, prepared, plans, horizons)
            msm_fits = tuple(
                fit_regimens_msm(prepared, plans, mechanism, model, cause=cause, **recursion)
                for cause in causes
            )
            fits = {
                _fit_key(fit.regimen.label, fit.cause, fit.horizon, prepared.is_survival): fit
                for msm_fit in msm_fits
                for fit in msm_fit.fits
            }

        self._warn_on_truncation(fits)

        config = LongitudinalConfig(
            family=prepared.family,
            n_times=prepared.n_times,
            outcome_names=(prepared.event_names or (prepared.outcome_name,)),
            horizons=horizons,
            causes=prepared.cause_labels,
            regimens=regimens,
            reference=reference.label,
            n_folds=folds.n_folds,
            g_bounds=bounds,
            q_bounds=self.q_bounds,
            alpha_sig=self.alpha_sig,
            random_state=self.random_state,
            # From the matrix ``resolve_plans`` already built, so this is the assignment
            # the fit ran on rather than a second evaluation of the rules.  The dense
            # codes and not the raw labels: the codes are what every fit actually divides
            # and regresses on, and the levels they index are folded into
            # ``data_fingerprint`` -- so two plans that differ in the arm they assign
            # differ here, and two datasets whose labels differ differ there, without
            # ``repr()``-ing one string per unit per node per regimen to say so.  That
            # also keeps the digest stable across numpy versions, which disagree on
            # ``repr(numpy.str_(...))``.
            plan_fingerprints=tuple((plan.label, fingerprint_array(plan.values)) for plan in plans),
            msm_terms=None if model is None else model.terms,
            msm_link=None if model is None else str(model.link),
            # The evaluated arrays rather than the design, for the reason the plans are
            # fingerprinted from their resolved matrix: a closure has no stable digest,
            # and what the fit used is what it was handed.
            msm_fingerprint=(
                None if model is None else fingerprint_array(model.design, model.weights)
            ),
        )
        with phase("influence_curve"):
            if model is None:
                estimates, parameter_index = self._estimates(prepared, fits, scaler, reference)
            else:
                estimates, parameter_index = self._msm_estimates(prepared, msm_fits), None
        with phase("inference"):
            bands = self._bands(estimates, prepared)
        return LongitudinalResult(
            estimates=estimates,
            fits=fits,
            data=prepared,
            config=config,
            scaler=scaler,
            mechanism=mechanism,
            provenance=self._provenance(prepared, folds),
            simultaneous=bands,
            parameter_index=parameter_index,
            msm=model,
            msm_fits=msm_fits,
        )

    # ------------------------------------------------------------- internals

    def _prepare(
        self,
        data: Any,
        *,
        outcome: str | Sequence[str] | Mapping[str, Sequence[str]] | None,
        treatment: Sequence[str] | None,
        baseline: Sequence[str] | None,
        time_varying: Sequence[Sequence[str]] | None,
        censoring: Sequence[str] | None,
        id: str | None,
        weights: str | None,
        weights_type: str,
        weights_estimated: bool,
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
                "weights": weights,
            }
            named = sorted(key for key, value in declared.items() if value is not None)
            if family != "auto":
                named.append("family")
            # The two that are not column names, and so cannot be caught by the ``is not
            # None`` sweep above: they say how a weight column is to be *read*, and a
            # container has already read it.
            if weights_type != "probability":
                named.append("weights_type")
            if weights_estimated:
                named.append("weights_estimated")
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
            weights=weights,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
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

    @staticmethod
    def _warn_on_truncation(fits: Mapping[str, RegimenFit]) -> None:
        """Warn once with every regimen/node whose scored rows are materially clipped."""
        found: dict[tuple[str, int], tuple[float, bool]] = {}
        for fit in fits.values():
            for step in fit.steps:
                raw = fit.cumulative_unbounded[:, step.time - 1][step.trained_on]
                bounded = fit.cumulative[:, step.time - 1][step.trained_on]
                if not raw.size:
                    continue
                share = float(np.mean(raw != bounded))
                constant = bool(share == 1.0 and np.unique(bounded).size == 1)
                key = (fit.regimen.label, step.time)
                previous = found.get(key, (0.0, False))
                found[key] = max(previous[0], share), previous[1] or constant

        affected = [
            (label, time, share, constant)
            for (label, time), (share, constant) in sorted(found.items())
            if share > _TRUNCATION_WARN_FRACTION
        ]
        if not affected:
            return
        details = "; ".join(
            f"{label} at t={time}: {share:.1%}"
            + (
                " (the bounded cumulative probability, and hence the clever covariate, "
                "is constant on scored rows)"
                if constant
                else ""
            )
            for label, time, share, constant in affected
        )
        warnings.warn(
            "Cumulative mechanism truncation exceeded 5% of scored rows: "
            f"{details}. The fit solves the score built from the truncated weights; "
            "material truncation can trade reduced weight extremes for truncation bias and "
            "can make plug-in "
            "influence-curve inference unreliable. Inspect "
            "res.diagnostics.stagewise().to_frame(), report "
            "the configured bounds, and refit the full backward recursion under "
            "substantively justified alternatives.",
            PositivityWarning,
            stacklevel=3,
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
                data.weights,
                *data.time_varying,
                np.asarray(
                    [repr(level) for levels in data.treatment_levels for level in levels],
                    dtype=str,
                ),
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
    ) -> tuple[dict[str, ParameterEstimate], dict[str, tuple[str, str | None, int]]]:
        survival = data.is_survival
        # ``ey_regimen`` is a mean of the outcome, ``risk_regimen`` a probability of an
        # event by a horizon, and ``cif_regimen`` the probability of leaving through one
        # particular cause by then.  E[Y_k] *is* the second, and with a single cause the
        # second *is* the third -- so one name would not be wrong anywhere.  But the three
        # come from different derivations, and a saved frame or a coverage study's truth
        # dict keyed by name is where that would stop being a distinction without a
        # difference.
        head = (
            ("cif_regimen" if data.is_competing else "risk_regimen") if survival else "ey_regimen"
        )
        estimates: dict[str, ParameterEstimate] = {}
        # ``name -> (regimen, cause, horizon)``, composed forward here and never parsed
        # back out of the name.  ``curve()`` reads it: with a cause beside the horizon
        # inside one pair of brackets, recovering either by splitting the string would be
        # guessing where a label ends, and a regimen called "a, b" would decide it wrongly.
        index: dict[str, tuple[str, str | None, int]] = {}
        for fit in fits.values():
            name = f"{head}[{_index(fit.regimen.label, fit.cause, fit.horizon, survival)}]"
            index[name] = (fit.regimen.label, fit.cause, fit.horizon)
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
            # The contrast is between the same two regimens at the same cause *and* the
            # same horizon; a difference of incidences across either is not a treatment
            # effect, and with two indexes there are now two ways to pair the wrong ones.
            base = fits[_fit_key(reference.label, fit.cause, fit.horizon, survival)]
            contrast = f"{fit.regimen.label} vs {reference.label}"
            name = f"ate_regimen[{_index(contrast, fit.cause, fit.horizon, survival)}]"
            index[name] = (contrast, fit.cause, fit.horizon)
            estimates[name] = make_estimate(
                name,
                scaler.unscale_difference(fit.psi_scaled - base.psi_scaled),
                scaler.unscale_influence(fit.influence_curve_scaled - base.influence_curve_scaled),
                n=data.n,
                cluster=data.cluster,
                scale="difference",
                alpha=self.alpha_sig,
            )
        return estimates, index

    def _msm_estimates(
        self, data: LongitudinalData, msm_fits: Sequence[MSMRegimenFit]
    ) -> dict[str, ParameterEstimate]:
        """One estimate per working-model term per cause, and no contrasts.

        **Not** unscaled.  ``beta`` and its curve come off the projection already on the
        outcome's own scale, because a coefficient vector has no single
        :class:`~cleverly.inference.Scale` to map back with -- the reason
        :meth:`cleverly.targets.TargetContext.finish_unscaled` exists at one time point.
        Putting them through :meth:`OutcomeScaler.unscale_level` here would apply the map
        twice, and on a binary outcome, where the scaler is the identity, would look
        perfectly fine while doing it.

        No contrasts: a working model reports coefficients, and a difference of two of
        them is ``result.contrast()`` rather than a row nobody asked for.
        """
        estimates: dict[str, ParameterEstimate] = {}
        for msm_fit in msm_fits:
            for column, term in enumerate(msm_fit.model.terms):
                # Composed forward, as every other name here is: a term may contain a
                # bracket, and a cause may contain the separator.
                inside = term if msm_fit.cause is None else f"{term}{CAUSE_INFIX}{msm_fit.cause}"
                name = f"msm_regimen[{inside}]"
                estimates[name] = make_estimate(
                    name,
                    float(msm_fit.beta[column]),
                    msm_fit.influence_curves[:, column],
                    n=data.n,
                    cluster=data.cluster,
                    scale="level",
                    alpha=self.alpha_sig,
                )
        return estimates


def ltmle(
    data: Any,
    *,
    regimens: Any,
    outcome: str | Sequence[str] | Mapping[str, Sequence[str]],
    treatment: Sequence[str],
    baseline: Sequence[str],
    time_varying: Sequence[Sequence[str]] | None = None,
    censoring: Sequence[str] | None = None,
    id: str | None = None,
    weights: str | None = None,
    weights_type: str = "probability",
    weights_estimated: bool = False,
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
        weights=weights,
        weights_type=weights_type,
        weights_estimated=weights_estimated,
        family=family,
    )
