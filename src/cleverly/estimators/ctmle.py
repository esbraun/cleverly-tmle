r"""Collaborative TMLE: choosing the treatment model against the *target* parameter.

A plain TMLE fits ``g_a(W) = P(A = a | W)`` to predict treatment as well as possible,
and that is the wrong objective.  Consider three covariates:

``W1``
    predicts both treatment and outcome -- a confounder.  Adjusting for it removes
    bias; omitting it leaves bias behind.
``W2``
    predicts treatment strongly and the outcome not at all -- an *instrument*.
    Adjusting for it removes no bias, because there was none to remove, while
    pushing propensity scores towards 0 and 1.  The clever covariate is
    :math:`1/g(W)`, so its variance -- and hence the variance of the estimate --
    inflates.  This is the *instrument-inflation* problem.
``W3``
    predicts only the outcome.  It belongs in ``Qbar``, not in ``g``.

A learner scored on treatment prediction takes ``W2`` eagerly: it is the single best
predictor available.  So the loss used to build ``g`` has to be a loss for the
*outcome* model that ``g`` is going to target -- which is exactly what collaborative
TMLE does.  A sequence of increasingly rich propensity models is built, each is used
to target the initial outcome regression, and the sequence is cut by cross-validation
on the (penalized) loss of the resulting targeted ``Qbar``.  The propensity model is
thereby chosen *in collaboration with* the outcome model rather than on its own
terms.

This buys **collaborative double robustness** (van der Laan & Gruber, 2010): the
fitted ``g`` only has to adjust for whatever confounding the initial ``Qbar`` has not
already handled, rather than for all of it.  Neither nuisance model needs to be
right on its own.

Three ways of building the sequence are available, mirroring the entry points of R's
``ctmle`` package:

``strategy="greedy"`` (default)
    Forward stepwise selection, van der Laan & Gruber (2010).  At each stage every
    covariate not yet in the model is tried, and the one whose resulting *targeted*
    outcome model has the smallest penalized loss is added.  When no addition improves
    on the current candidate the algorithm **increments the TMLE step** -- it takes the
    current targeted fit as the new starting point and searches again -- then accepts the
    best addition from that step.  The unpenalized likelihood fluctuation cannot worsen
    its own loss, but the recorded penalized selection risk can rise.  Costs ``O(V p^2)``
    propensity fits for ``p`` covariates and ``V`` selection folds.

    The penalty belongs in this step and not only in the cross-validation that follows.
    Ranked on the bare loss, an instrument wins: extreme clever-covariate values let the
    fluctuation move the in-sample fit further than a confounder can.  It is the
    variance term that reverses that, and without it the forward search reaches for the
    instrument first.

``strategy="ordered"``
    Scalable C-TMLE (Ju et al., 2019).  The sequence is fixed in advance by an
    ordering, so only ``O(V p)`` fits are needed.  This is what makes C-TMLE usable
    when ``p`` is large.  ``preorder="logistic"`` implements Algorithm 2 of Ju et al.:
    it ranks one-variable propensity models by the empirical loss of the Qbar each
    targets.  ``preorder="partial_correlation"`` implements Algorithm 3, ranking the
    absolute partial correlation of ``Y - Qbar0(A,W)`` and each covariate conditional
    on treatment.  Pass ``ordering=`` to supply a fixed order instead.  A refit that adds
    a covariate, such as ``random_common_cause``, places it after the declared ordering.

``strategy="discrete"``
    Cross-validated selection among an explicit list of candidate covariate sets --
    the analogue of ``ctmleDiscrete`` / ``ctmleGlmnet``.

The three strategies above build a candidate path and cut it at a data-chosen stopping
index.  **The package supplies no inference for any of them.**  ``ci``, ``pvalue`` and
``std_error`` raise :class:`~cleverly.exceptions.CapabilityError`: no result shows the
reported curve is this estimator's influence curve when the selected working mechanism is
not consistent for the treatment law.  The point estimate, the selection path and the
curve remain, and ``plugin_std_error`` and ``plugin_interval`` report the retained
diagnostic under names that make no coverage claim.  F18 in ``docs/roadmap.md`` is the
condition that reopens this.  One ``"discrete"`` fit is the exception.  When the declared
list holds one candidate and that candidate is the full adjustment set, nothing is
selected, the estimator is the ordinary TMLE, and the fit takes the ordinary TMLE status.

``strategy="oat"``
    The outcome-adaptive treatment mechanism of Benkeser, Cai and van der Laan (2020).
    This is not a fourth candidate sequence: it fits treatment on the arm-specific
    outcome predictions and then uses the ordinary all-arm mean fluctuation.
    Consequently it has no candidate path, no parameter-specific selector loss and no
    stopping index.  ``oat_design`` selects one of two designs.  ``"per_arm"``, the
    default, regresses ``1{A = a}`` on the one column ``Qbar(a, W)`` for each arm.
    ``"shared"`` is ``ctmle3::LF_oat``: one categorical mechanism on the vector
    ``[Qbar(a, W): a in arms]``.

    **It also trades away one leg of double robustness, and that is the reason to reach
    for it deliberately rather than as a default.**  The three selector strategies choose
    a *subset of the analyst's covariates*, so the full model is always in the candidate
    path and a search that keeps going recovers :math:`g_0`; collaborative double
    robustness is retained.  Here :math:`W` never enters :math:`g` at all.  The fitted
    mechanism is the projection of :math:`A` on :math:`\sigma(\hat{\bar Q})`, so it is
    consistent for :math:`g_0` only when :math:`g_0` is a function of the outcome
    regression -- and if :math:`\hat{\bar Q}` is inconsistent then in general so is
    :math:`\hat g`, and the estimator has *neither* leg.  What it buys in exchange is the
    collaborative one: an :math:`\hat g` that carries only the confounding the outcome
    regression left behind, which is what keeps an instrument out of the denominator.

    **Its design is a generated regressor.**  In a cross-fitted fit, fold :math:`v`'s
    outcome model is trained outside :math:`v`, evaluated on both its training and
    validation rows, and the adaptive mechanism is then trained on those training-row
    predictions only.  This is the honest nesting in Benkeser, Cai and van der Laan
    (2020): no validation outcome can reach its own mechanism through another row's
    generated feature.  Their Theorem 1 proves the influence curve for one
    treatment-specific mean with one scalar design, and the per-arm design is that
    construction for each arm.  The curve is not the efficient influence function: the
    estimator is superefficient.  A per-arm fit on complete data without baseline strata
    therefore reports ``ci``, ``pvalue`` and ``std_error``, while
    :data:`OAT_PER_ARM_INFERENTIAL` is set.  No result covers the shared design, so every
    ``oat_design="shared"`` fit takes the ``"generated_design_plugin"`` status, as does a
    per-arm fit with ``delta=`` or ``strata=``: ``ci``, ``pvalue`` and ``std_error`` raise
    :class:`~cleverly.exceptions.CapabilityError`, and ``plugin_std_error`` and
    ``plugin_interval`` report the retained diagnostic.  F19 and X29 in
    ``docs/roadmap.md`` hold those parts.  ``ctmle3`` does not cross-fit this fit at all
    -- ``LF_oat`` pins ``cv_fold = -1`` -- so non-cross-fitted parity and a cross-fitted
    result would have different sources.

The loss
--------

Everything happens on the ``[0, 1]`` scale -- the outcome scaling of Gruber & van der
Laan (2010) that the rest of this library already applies.  Two losses for ``Qbar``
are available, and ``loss="auto"`` picks by outcome type as R's ``ctmle`` does:

``"loglik"`` (binary outcome)
    .. math::

        L(\bar Q^*) = -\sum_i w_i \left[ Y_i \log \bar Q^*_i
                                       + (1 - Y_i) \log (1 - \bar Q^*_i) \right],

    the quasi-binomial log-likelihood -- the same loss the targeting step maximises.

``"squared"`` (continuous outcome)
    .. math:: L(\bar Q^*) = \sum_i w_i (Y_i - \bar Q^*_i)^2.

    Preferred for a continuous outcome not only by convention but because it makes
    the criterion *scale-free*: rescaling the outcome multiplies the squared-error
    loss and the penalty below by the same factor, so their balance does not depend
    on the outcome's units.  The log-likelihood has no such property, and on a
    wide-ranging continuous outcome -- where the scaled residuals are squeezed into a
    narrow band near the middle of ``[0, 1]`` -- it becomes an erratic guide.

With ``penalty=True`` (the default) a variance/bias term is added, following the
penalized loss proposed by Gruber & van der Laan (2010) for parameters that are only
borderline identifiable:

.. math::

    L_{\text{pen}} = L(\bar Q^*)
                   + \operatorname{tr}\{\widehat{\operatorname{Cov}}(D^*)\}
                   + n\,\|\bar D^*\|_2^2,

where :math:`D^*` is the candidate's estimated vector efficient influence curve on the
rows being scored and :math:`\bar D^*` its mean -- the part of the score the targeting
step has *not* solved away out of sample.  The two terms are :math:`O(1)` against a
loss that is :math:`O(n)`, so the penalty is negligible except when the influence
curve's variance blows up, which is precisely the near-positivity case it exists to
guard.  It is what makes the instrument in the example above expensive rather than
merely useless.

In the cross-validated selector the two terms are computed from the *pooled* influence
curve across validation folds -- each row's contribution coming from the fold that
held it out -- rather than fold by fold.  This is the ``cvVar + n * cvBias^2`` of the
published criterion, and pooling is not cosmetic: a variance estimated inside a single
validation fold is noisy enough to swamp the difference between the two candidates it
is meant to separate.

.. note::

   This penalty follows the formula as published; it is not a transcription of the
   ``ctmle`` R package's source, and no claim of numerical parity with it is made.
   Set ``penalty=False`` for the plain cross-validated loss selector.

What the final estimate is
--------------------------

Selection chooses the complete candidate pair ``(g_k, Qbar*_k)``.  The reported
estimator continues the pooled targeting step from that selected Qbar rather than
discarding it and restarting from ``Qbar0``.  The continuation is normally numerical
only -- its epsilon is approximately zero -- but keeping it on the ordinary retargeting
path makes the estimate, influence curve, score check and sensitivity analyses agree.
The initial Qbar is retained separately for nuisance diagnostics.

On the selector strategies the package reports the spread of the plug-in curve as a
diagnostic and not as inference. The one exception is the ``"discrete"`` fit whose one
declared candidate is the full adjustment set, which is the ordinary TMLE. The selector
calculation treats the selected candidate as fixed but makes no conditional-on-selection
coverage claim. The outcome-adaptive calculation uses the fold-local nuisance construction of
Benkeser, Cai and van der Laan (2020). Their Theorem 1 proves the adaptive-propensity curve for
one treatment-specific mean under six stated regularity conditions, and the Remark and
Appendix D outline the ATE and cross-validated constructions. The per-arm design applies the
theorem to each arm and stacks the curves; the shared design is not the paper's construction.
See ``docs/roadmap.md F18`` for the selector path and ``docs/roadmap.md F19`` for the shared
design. ``n_bootstrap=`` reruns the adaptive construction, but no reviewed theorem validates
that bootstrap for a selector path or for either outcome-adaptive design.

Fixed probability weights replace the empirical law by its normalized weighted version.
The same row mass reaches nuisance fits, targeting, selector loss, influence-curve penalty,
cross-validated risk and the plug-in. The outcome-adaptive route uses it in both nuisance
fits, targeting and the plug-in. The simulated common-cause surface keeps each weight with
its observed row and reruns this complete fit. That surface refuses a clustered fit and
refuses estimated weights. Canonical ``ctmle`` and archived ``ctmle3`` provide no weighted
comparator, so no numerical parity is claimed for that composition.

State of the evidence
---------------------

Worth knowing before reading a favourable simulation as a verdict on this
implementation.

On a process whose outcome model is correctly specified, the *empty* propensity model is
a legitimate mean-squared-error-minimising choice -- collaborative double robustness says
the confounding is already handled, and adjusting for nothing carries the least variance.
So C-TMLE selects one, often. Measured on the instrument process at ``n = 700`` after
nested selection cross-fitting, the ordered search selects no covariates on all five fixed
unit-test seeds. That is correct behaviour, not a defect.

It does mean a comparison against plain TMLE on such a process is weaker evidence than it
looks.  A selector hard-wired to return the empty model would win it, so winning it does
not show that the search discriminates among covariates; and losing it would not show the
search is broken either, since dominance is contingent on how much variance the discarded
covariates were costing.  Verified by making that substitution: a degenerate
``selected = 0`` left every C-TMLE test in the suite passing except the five that were
added to catch exactly this.

The claim that the search selects what it needs is therefore made where selecting nothing
is *wrong*.  Reduce the outcome learner to a constant, so every bit of adjustment has to
come through ``g``.  On the same instrument process, over the test's three seeds at
``n = 1,500``, the greedy search includes the confounder ``W1`` and never selects the
empty model; the test allows the instrument on at most one seed.  A selector restricted
to the empty candidate has mean absolute error 0.695 there, against the collaborative
fit's 0.036. See
``tests/e2e/test_ctmle.py::TestSelectionIsForcedWhenTheOutcomeModelCannotHelp``.

The ``tmle3`` source is used as a reference for the shared construction -- out-of-fold
nuisance predictions followed by a pooled fluctuation.  It does not implement
collaborative selection, so the search itself is checked against the paper equations,
training-row audit tests and mutation controls rather than presented as cross-package
parity.

References
----------
- van der Laan & Gruber (2010), *Collaborative Double Robust Targeted Maximum
  Likelihood Estimation*.
- Gruber & van der Laan (2010), *An Application of Collaborative Targeted Maximum
  Likelihood Estimation in Causal Inference and Genomics*.
- Ju, Gruber, Lendle, Chambaz, Franklin, Wyss, Schneeweiss & van der Laan (2019),
  *Scalable collaborative targeted learning for high-dimensional data*.

Examples
--------
>>> from cleverly.estimators import CTMLE
>>> from cleverly.datasets import make_instrument
>>> from sklearn.linear_model import LinearRegression, LogisticRegression
>>> frame, truth = make_instrument(n=1000, seed=0)
>>> res = CTMLE(
...     outcome_learner=LinearRegression(),
...     treatment_learner=LogisticRegression(max_iter=1000),
...     cross_fit=False,
...     estimands=["ate"],
... ).fit(
...     frame, outcome="Y", treatment="A"
... ).single()
>>> selected = res.extra["ctmle"].selected_covariates
>>> "W2" in selected
False

``W2`` is the instrument this process is built around, and leaving it out of ``g`` is the
selection the method exists to make.  What the search *does* take says little here:
``LinearRegression`` is correctly specified for ``Qbar`` on this process, so no
confounding is left in the residual for ``g`` to adjust for and the remaining candidates
are close to worthless.  Inspect ``res.extra["ctmle"].cv_risk`` alongside ``.path`` to see
what each one was worth, and read the section above before concluding anything from a
comparison against a plain fit.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from .._inference_status import InferenceStatus
from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..exceptions import CapabilityError
from ..fluctuation.iterative import InitialFit, apply_logistic, check_matching_arms
from ..fluctuation.submodel import Submodel, restrict, weighted_form
from ..inference.cluster import influence_variance
from ..inference.delta import log_odds_ratio_influence, log_ratio_influence, wald_ci
from ..inference.influence import counterfactual_means
from ..interventions.learned import learned_rule_configuration_refusal
from ..learners._fitting import Task, predict_mean, predict_probabilities
from ..learners.crossfit import _POST_DRAW_REMEDY, Folds, check_integrity, make_folds
from ..learners.super_learner import resolve_learner
from ..targets.base import parameter_name
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from ..utils.parallel import map_parallel
from ..utils.text import format_table
from ._nuisance import NuisanceEstimates, Propensity, cross_fit_predictions, fit_on_rows
from .base import MEAN_GROUP_ESTIMANDS, TMLEConfig, resolve_estimands
from .targeting import build_submodel, solve_submodel
from .tmle import TMLE

if TYPE_CHECKING:
    from .base import TMLEResult

__all__ = [
    "CTMLE",
    "CTMLE_SELECTOR_STRATEGIES",
    "OAT_PER_ARM_INFERENTIAL",
    "CTMLELoss",
    "CTMLEOatDesign",
    "CTMLEOutcomeAdaptiveFit",
    "CTMLEPreorder",
    "CTMLESelection",
    "CTMLEStrategy",
    "LogisticPlugin",
    "is_selector_strategy",
    "logistic_plugin",
    "per_arm_design_admits",
]

CTMLEStrategy = Literal["greedy", "ordered", "discrete", "oat"]
CTMLELoss = Literal["auto", "loglik", "squared"]
CTMLEPreorder = Literal["logistic", "partial_correlation"]
CTMLEOatDesign = Literal["per_arm", "shared"]

#: Whether an admitted per-arm outcome-adaptive fit publishes inference. Declared ``True``
#: with the registered study ``ctmle-oat-per-arm``. A red positive coverage or calibration
#: cell of that study, with no defect found, sets this to ``False``. Every per-arm fit then
#: takes the ``"generated_design_plugin"`` status. :func:`per_arm_design_admits` reads it.
OAT_PER_ARM_INFERENTIAL: Final[bool] = True

#: The strategies that build a candidate path and cut it at a data-chosen stopping index.
#: ``"oat"`` is not one: it fits its mechanism on the arm-specific outcome predictions and
#: has no path, no selector loss and no stopping index (see the module docstring).
#: F19 owns the inference of its shared design; F18 owns these three.
CTMLE_SELECTOR_STRATEGIES: frozenset[str] = frozenset({"greedy", "ordered", "discrete"})


def is_selector_strategy(strategy: str | None) -> bool:
    """Whether a strategy selects a working mechanism off a candidate path.

    Parameters
    ----------
    strategy : str or None
        A :data:`CTMLEStrategy` value, or ``None`` for an estimator that declares no
        strategy at all.

    Returns
    -------
    bool
        ``True`` for ``"greedy"``, ``"ordered"`` and ``"discrete"``. ``False`` for
        ``"oat"``, for ``None`` and for any unrecognised value, so a caller that
        broadens the Literal does not silently acquire a refusal it never declared.
    """
    return strategy in CTMLE_SELECTOR_STRATEGIES


def declares_full_adjustment_only(
    strategy: str | None,
    candidates: Sequence[Sequence[str]] | None,
    covariate_names: Sequence[str],
) -> bool:
    """Whether a ``"discrete"`` fit declares one candidate, and that candidate is every covariate.

    Such a fit has nothing to select. Its treatment mechanism is fitted on the full
    adjustment set, so the estimator is the ordinary TMLE. The key reads the declared list
    and the prepared covariate names only, and never the fitted path, so a caller can know
    the answer before the fit.

    Parameters
    ----------
    strategy : str or None
        The collaborative strategy.
    candidates : sequence of sequence of str or None
        The declared ``candidates=`` list.
    covariate_names : sequence of str
        The prepared covariate names, :attr:`~cleverly.data.CausalData.covariate_names`.

    Returns
    -------
    bool
        ``True`` when the strategy is ``"discrete"``, the list has exactly one entry, and
        that entry holds each covariate name exactly once, in any order. A repeated
        candidate counts as two candidates, and a repeated name is not the full set.
    """
    if strategy != "discrete" or candidates is None or len(candidates) != 1:
        return False
    return len(covariate_names) > 0 and sorted(candidates[0]) == sorted(covariate_names)


def per_arm_design_admits(estimator: Any, data: CausalData) -> bool:
    """Whether an outcome-adaptive fit is inside the per-arm result and reports inference.

    The per-arm design regresses ``1{A = a}`` on the one column ``Qbar_n(a, W)`` for each arm.
    Benkeser, Cai and van der Laan (2020), Theorem 1, give the influence curve of that
    construction for one treatment-specific mean. An indicator reduction per arm and a
    fixed-dimension stack give the joint curve of every arm.
    ``docs/technical-reference/collaborative-tmle.md`` states the contract. The key reads
    the estimator configuration, the prepared data and :data:`OAT_PER_ARM_INFERENTIAL`. It
    never reads a fitted array, so a caller knows the answer before the fit.

    Parameters
    ----------
    estimator : CTMLE
        The collaborative estimator. Only ``strategy`` and ``oat_design`` are read.
    data : CausalData
        The prepared data. Missing outcomes, a missing treatment and baseline strata are
        read.

    Returns
    -------
    bool
        ``True`` when the strategy is ``"oat"``, the design is ``"per_arm"``, every outcome
        and treatment is observed, the fit declares no baseline strata, and
        :data:`OAT_PER_ARM_INFERENTIAL` is ``True``.
    """
    return bool(
        OAT_PER_ARM_INFERENTIAL
        and getattr(estimator, "strategy", None) == "oat"
        and getattr(estimator, "oat_design", None) == "per_arm"
        and not data.has_missing_outcome
        and not data.has_missing_treatment
        and not data.has_strata
    )


#: Floor applied to targeted predictions before taking a logarithm in the loss.
_LOSS_EPS = 1e-12

#: The way out of a selection or nested split that cannot fit its nuisances.  The search
#: draws those splits even in sample, so fitting in sample removes neither of them.
#: ``strategy="oat"`` refuses a selector setting away from its default, and the ordinary
#: TMLE cross-fits by default, so the text names the settings under which each runs.
_SELECTION_SPLIT_REMEDY = (
    "fit strategy='oat' with every selector setting (selection_folds, selection_inner_folds, "
    "loss, penalty, ctmle_estimand) at its default, or the ordinary TMLE in sample, "
    "TMLE(cross_fit=False), neither of which draws a selection split"
)

#: Why CTMLE refuses a continuous dose, which every strategy does.
_DISCRETE_TREATMENT_REFUSAL = (
    "CTMLE strategies require a discrete treatment. strategy='oat' fits its mechanism on "
    "one Qbar prediction per arm, and a continuous dose has no finite arm vector."
)

#: The selection split's refusal remedy in the words the nuisance fold loop closes with.
_SELECTION_SPLIT_BACKSTOP = _POST_DRAW_REMEDY.format(remedy=_SELECTION_SPLIT_REMEDY)


@dataclass(frozen=True)
class CTMLESelection:
    """The candidate sequence a C-TMLE fit searched, and where it was cut.

    Attributes
    ----------
    path:
        Covariate sets of the candidates, in the order the search built them.  Each
        is the covariate set used for ``g``; the outcome model always sees every
        covariate.
    n_steps:
        Number of fluctuation steps the search had applied when it reached each
        candidate.  A value above one means the greedy search had to increment the
        TMLE step to keep making progress.
    train_risk:
        The selection criterion for each candidate, evaluated in sample.
        A greedy or ordered search retries from one further targeted fit when no
        addition improves the criterion, then accepts the best addition.  The recorded
        penalized risk may therefore increase.
    cv_risk:
        The same criterion, cross-validated.  This is the quantity actually
        minimised, and unlike ``train_risk`` it is free to turn back up -- which is
        what stops the search from simply taking every covariate.
    selected:
        Index into ``path`` of the chosen candidate.
    folds:
        The realized selection split ``cv_risk`` was scored over.  This is the
        partition the fit actually used, not a rule for rebuilding one, so a paired
        study can hand another implementation the same fold assignment instead of
        recomputing its own and hoping the two agree.

        Its :attr:`~cleverly.learners.Folds.origin` records the generator, the scheme,
        the requested fold count and the seed, at **every** cross-fitting setting.  A
        collaborative fit draws this split whether or not the outer nuisances are
        cross-fitted, so ``cross_fit=False`` -- which records no outer split on
        :class:`~cleverly.learners.CrossFitPlan`, because there is none -- would
        otherwise leave the one split the fit did draw unrecorded.
    """

    strategy: CTMLEStrategy
    preorder: str | None
    estimand: str
    target_names: tuple[str, ...]
    loss: str
    penalized: bool
    path: tuple[tuple[str, ...], ...]
    n_steps: tuple[int, ...]
    train_risk: FloatArray
    train_loss: FloatArray
    penalty: FloatArray
    treatment_risk: FloatArray
    cv_risk: FloatArray
    selected: int
    covariates: tuple[str, ...]
    folds: Folds

    @property
    def selected_covariates(self) -> tuple[str, ...]:
        """The covariate set chosen for the treatment model."""
        return self.path[self.selected]

    @property
    def dropped(self) -> tuple[str, ...]:
        """Covariates the treatment model left out, in their original order."""
        chosen = set(self.selected_covariates)
        return tuple(name for name in self.covariates if name not in chosen)

    def to_frame(self, data: CausalData | None = None) -> Any:
        """One row per candidate: its covariate set, loss and cross-validated risk."""
        payload: dict[str, Any] = {
            "candidate": list(range(len(self.path))),
            "covariates": [", ".join(names) if names else "(intercept)" for names in self.path],
            "n_covariates": [len(names) for names in self.path],
            "n_steps": list(self.n_steps),
            "train_risk": self.train_risk.tolist(),
            "train_loss": self.train_loss.tolist(),
            "penalty": self.penalty.tolist(),
            "treatment_risk": self.treatment_risk.tolist(),
            "cv_risk": self.cv_risk.tolist(),
            "selected": [index == self.selected for index in range(len(self.path))],
        }
        if data is None:
            return payload
        return data.frame_like(payload)

    def summary(self) -> str:
        """A printable report of the selection."""
        rows = []
        for index, names in enumerate(self.path):
            rows.append(
                [
                    str(index),
                    ", ".join(names) if names else "(intercept)",
                    str(self.n_steps[index]),
                    f"{self.train_risk[index]:.6g}",
                    f"{self.cv_risk[index]:.6g}",
                    "<--" if index == self.selected else "",
                ]
            )
        table = format_table(
            ["k", "covariates in g", "steps", "risk", "cv risk", ""],
            rows,
        )
        criterion = "penalized " if self.penalized else ""
        loss_name = "log-likelihood" if self.loss == "loglik" else "squared-error"
        header = [
            "Collaborative TMLE selection",
            "=" * 28,
            f"strategy = {self.strategy}; preorder = {self.preorder or 'n/a'}; "
            f"target = {self.estimand} ({', '.join(self.target_names)}); "
            f"criterion = cross-validated {criterion}{loss_name} loss",
            "",
        ]
        chosen = self.selected_covariates
        footer = [
            "",
            "selected g: " + (", ".join(chosen) if chosen else "(intercept only)"),
        ]
        if self.dropped:
            footer.append(
                "left out: "
                + ", ".join(self.dropped)
                + f". The selected candidate minimized targeted cross-validated {criterion}"
                f"{loss_name} loss; this criterion does not determine why a covariate "
                "was left out"
            )
        return "\n".join([*header, table, *footer])

    def describe(self) -> str:
        """One sentence naming the strategy, the candidate the search cut at, and its target.

        The sentence every report of a collaborative fit prints, in the house pattern of
        :meth:`~cleverly.estimators.TMLEConfig.describe`.  Its counterpart is
        :meth:`CTMLEOutcomeAdaptiveFit.describe`, so a caller renders a retained artifact
        without first discriminating which of the two it holds.  Two call sites
        discriminated by hand and had already drifted to ``candidate=2/5`` against
        ``candidate 2 of 5``.

        Returns
        -------
        str
            A complete sentence, ready to embed in a report line.
        """
        return (
            f"C-TMLE {self.strategy} selected candidate {self.selected + 1} of "
            f"{len(self.path)} for {self.estimand}"
        )

    @property
    def treatment_features(self) -> tuple[str, ...]:
        """The covariates entering the selected treatment model."""
        return self.selected_covariates

    @property
    def treatment_risk_selected(self) -> float:
        """Treatment negative log likelihood at the selected path position."""
        return float(self.treatment_risk[self.selected])


@dataclass(frozen=True)
class CTMLEOutcomeAdaptiveFit:
    """Diagnostics for the outcome-adaptive treatment mechanism of ``strategy="oat"``.

    Parameters
    ----------
    strategy : str
        Always ``"oat"``.
    treatment_features : tuple of str
        The ``Qbar`` features that reached the treatment mechanism, one per arm.
    treatment_risk : float
        The weighted treatment negative log likelihood. Under the per-arm design it is the
        sum over arms of each binary mechanism's negative log likelihood.
    design : {"per_arm", "shared"}
        ``"per_arm"`` fits one binary mechanism per arm on that arm's own ``Qbar``
        feature. ``"shared"`` fits one categorical mechanism on every feature.
    treatment_risk_by_arm : dict of str to float or None
        Each arm's binary negative log likelihood under the per-arm design, keyed by the
        arm label. ``None`` under the shared design.
    """

    strategy: CTMLEStrategy
    treatment_features: tuple[str, ...]
    treatment_risk: float
    design: CTMLEOatDesign = "shared"
    treatment_risk_by_arm: dict[str, float] | None = None

    @property
    def treatment_risk_selected(self) -> float:
        """Treatment negative log likelihood, under the shared C-TMLE diagnostic API."""
        return self.treatment_risk

    def describe(self) -> str:
        """One sentence naming this fit and the treatment design it built.

        The counterpart of :meth:`CTMLESelection.describe`.  There is no candidate path to
        cut, so what a reader wants instead is how the ``Qbar`` features reached ``g``.

        Returns
        -------
        str
            A complete sentence, ready to embed in a report line.
        """
        count = len(self.treatment_features)
        if self.design == "per_arm":
            return (
                f"C-TMLE outcome-adaptive fit used one Qbar feature per arm in {count} "
                "binary mechanisms"
            )
        return (
            f"C-TMLE outcome-adaptive fit used {count} Qbar features in one categorical mechanism"
        )

    def summary(self) -> str:
        """A printable report of the outcome-adaptive treatment design.

        Returns
        -------
        str
            The design, the features and the treatment negative log likelihood.
        """
        features = ", ".join(self.treatment_features)
        if self.design == "per_arm":
            model = "per_arm; treatment model = 1{A = a} ~ Qbar(a, W), one per arm"
        else:
            model = "shared; treatment model = A ~ [Qbar(a, W)]"
        lines = [
            "Collaborative TMLE outcome-adaptive fit",
            "=" * 39,
            f"strategy = oat; design = {model}",
            f"features: {features}",
            f"treatment negative log likelihood: {self.treatment_risk:.6g}",
        ]
        if self.treatment_risk_by_arm is not None:
            lines.extend(
                f"  arm {label}: {value:.6g}" for label, value in self.treatment_risk_by_arm.items()
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class _Candidate:
    """One element of the candidate sequence: a propensity model and its targeted fit."""

    covariates: tuple[str, ...]
    propensity: Propensity
    submodel: Submodel
    targeted: InitialFit
    epsilon: FloatArray
    n_steps: int
    loss: float
    penalty: float
    treatment_risk: float
    risk: float


class CTMLE(TMLE):
    """Collaborative TMLE for a discrete point treatment.

    Selects the covariates entering ``g(W)`` by cross-validating the loss of the
    *targeted* outcome model, rather than the loss of ``g`` itself.  See the module
    docstring for the algorithm and the loss.

    Shared :class:`~cleverly.TMLE` nuisance and targeting controls behave identically
    within the supported pooled point-treatment scope.  The result is an ordinary
    :class:`~cleverly.estimators.TMLEResult` with the selection recorded under
    ``result.extra["ctmle"]``.

    The selector strategies refuse ``ci``, ``pvalue`` and ``std_error``, and report
    ``plugin_std_error`` and ``plugin_interval`` instead.  A contrast of them, a
    simultaneous band and every E-value branch refuse for the same reason, which F18 in the
    roadmap holds.  The exception is a ``"discrete"`` fit whose one declared candidate is
    the full adjustment set: it selects nothing, equals the ordinary TMLE, and reports its
    interval.  ``strategy="oat"`` with the default per-arm design reports an interval on
    complete data without baseline strata.  Benkeser, Cai and van der Laan (2020),
    Theorem 1, give its influence curve.  Every ``oat_design="shared"`` fit, and a per-arm
    fit with missing outcomes or strata, withholds the interval for the reason F19 and X29
    hold.  See the module docstring.

    Parameters
    ----------
    strategy:
        ``"greedy"`` (default), ``"ordered"``, ``"discrete"`` or ``"oat"``.  The
        selector strategies fit one shared categorical propensity path and jointly score
        all components of ``ctmle_estimand``; ``"oat"`` fits treatment on the arm-specific
        outcome predictions without a candidate path (see ``oat_design``).
        ``"oat"`` excludes ``W`` from ``g`` entirely and so gives up
        consistency-when-only-``g``-is-right; see the module docstring.
    ordering:
        Explicit covariate order for ``strategy="ordered"``.  When omitted, ``preorder``
        determines the published data-adaptive ordering.  A refit that adds a covariate,
        such as ``random_common_cause``, places it after the declared ordering.
    preorder:
        ``"logistic"`` (default) or ``"partial_correlation"`` for Algorithms 2 and 3
        of Ju et al. (2019).  Ignored when an explicit ``ordering=`` is supplied.
    candidates:
        Explicit candidate covariate sets for ``strategy="discrete"``.
    selection_folds:
        Folds used to cross-validate the candidate sequence.  Separate from
        ``n_folds``, which cross-fits the nuisance models.
    selection_inner_folds:
        Inner folds used to make selection-training predictions out of fold. Two is the
        default because every selection fold also needs a full-training fit for its
        validation rows; increasing it improves the inner cross-fit at a linear fit-cost.
    loss:
        ``"auto"`` (default) uses the squared-error loss for a continuous outcome and
        the quasi-binomial log-likelihood for a binary one, as R's ``ctmle`` does.
    penalty:
        Add the variance/bias penalty to the selection loss.  Leave it on unless you
        specifically want the unpenalized log-likelihood selector.
    ctmle_estimand:
        Which estimand the selection is *for*.  A collaborative selection is
        parameter-specific -- the loss involves that estimand's influence curve --
        so unlike a plain TMLE, one fit cannot serve every estimand equally. At ``K``
        arms, ``ate``, ``rr`` and ``or`` jointly optimize the ``K - 1`` contrasts against
        ``reference=`` and ``ey`` jointly optimizes all ``K`` means. Must be requested.
    oat_design:
        The treatment design of ``strategy="oat"``.  ``"per_arm"`` (the default under
        ``"oat"``) fits one binary mechanism per arm, ``P(A = a | Qbar(a, W))``, on that
        arm's own outcome prediction.  ``"shared"`` fits one categorical mechanism on the
        vector ``[Qbar(a, W): a in arms]``, as ``ctmle3::LF_oat`` does.  Only the per-arm
        design reports inference; see the module docstring.  ``None`` resolves to
        ``"per_arm"`` under ``"oat"``, and any other value raises under a selector strategy.

    Notes
    -----
    ``att`` and ``atc`` are not supported: their clever covariate conditions on a
    random event, so the candidate sequence would have to be rebuilt per estimand
    and the "one selected ``g``" story breaks down.  Request them from a plain
    :class:`~cleverly.TMLE`.

    ``screen_treatment`` is redundant here and does not apply to the candidate
    propensity models.  It is a *static* marginal pre-screen for exactly the problem
    this class solves collaboratively, and running both would mean the covariate a
    correlation filter rejected never reaches the search that might have wanted it.
    """

    _assessment_method = "collaborative_tmle"

    def _inference_status(self, data: CausalData) -> InferenceStatus:
        """Name the status of each strategy, and admit the one fit that selects nothing.

        Keyed on the strategy. The selector paths take the status that F18 keys on, with one
        exception. A ``"discrete"`` fit that declares one candidate, equal to the full adjustment
        set, selects nothing and is the ordinary TMLE, so it takes the status of
        :class:`~cleverly.TMLE`. :func:`declares_full_adjustment_only` is the key. It reads the
        declared list and the prepared covariate names, and never the fitted path.
        ``docs/technical-reference/collaborative-tmle.md`` states the contract.

        An ``"oat"`` fit with the per-arm design takes the :class:`~cleverly.TMLE` status
        when :func:`per_arm_design_admits` admits it: complete outcomes and treatment, no
        baseline strata, and :data:`OAT_PER_ARM_INFERENTIAL` set. Benkeser, Cai and van der
        Laan (2020), Theorem 1, give its curve. Every other ``"oat"`` fit takes
        ``"generated_design_plugin"``. That is every ``oat_design="shared"`` fit, which F19
        holds, and a per-arm fit with missing outcomes or strata, which X29 holds. The
        status table of ``docs/technical-reference/inference.md`` states both.

        Parameters
        ----------
        data : CausalData
            The prepared data. Its covariate names, missing outcomes, missing treatment and
            strata are read here. An admitted fit passes ``data`` to the
            :class:`~cleverly.TMLE` status.

        Returns
        -------
        str
            One of :data:`~cleverly.inference.influence.InferenceStatus`: the
            :class:`~cleverly.TMLE` status for the admitted ``"discrete"`` fit and the
            admitted per-arm ``"oat"`` fit, ``"working_mechanism_plugin"`` for every other
            ``"greedy"``, ``"ordered"`` and ``"discrete"`` fit, and
            ``"generated_design_plugin"`` for every other ``"oat"`` fit.
        """
        if declares_full_adjustment_only(self.strategy, self.candidates, data.covariate_names):
            return super()._inference_status(data)
        if is_selector_strategy(self.strategy):
            return "working_mechanism_plugin"
        if per_arm_design_admits(self, data):
            return super()._inference_status(data)
        return "generated_design_plugin"

    def _bootstrap_inferential(self, data: CausalData) -> bool:
        """Whether the full-refit bootstrap interval is published as inference.

        ``False`` on every ``"oat"`` fit. Theorem 1 of Benkeser, Cai and van der Laan (2020)
        gives the influence curve of a superefficient estimator, and it does not cover the
        bootstrap of that estimator. Every other strategy takes the :class:`~cleverly.TMLE`
        answer.

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Returns
        -------
        bool
            ``False`` for ``strategy="oat"``, otherwise the :class:`~cleverly.TMLE` answer.
        """
        if self.strategy == "oat":
            return False
        return super()._bootstrap_inferential(data)

    def __init__(
        self,
        *,
        strategy: CTMLEStrategy = "greedy",
        preorder: CTMLEPreorder | None = None,
        ordering: Sequence[str] | None = None,
        candidates: Sequence[Sequence[str]] | None = None,
        selection_folds: int = 5,
        selection_inner_folds: int = 2,
        loss: CTMLELoss = "auto",
        penalty: bool = True,
        ctmle_estimand: str = "ate",
        oat_design: CTMLEOatDesign | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("learned_rule") is not None:
            # Row 1 of the learned-rule refusal table comes before the fold-evaluation refusal
            # below: its remedy, cv_evaluation=False, would meet this refusal next.
            raise CapabilityError(learned_rule_configuration_refusal(self) or "")
        if kwargs.get("cv_evaluation", False):
            raise CapabilityError(
                "CTMLE does not support cv_evaluation=True: canonical CV-TMLE selection "
                "requires a separate fold-specific collaborative derivation."
            )
        # The base constructor validates the fold policy. OAT has no selector folds,
        # so it must know the strategy before that validation runs.
        self.strategy = strategy
        super().__init__(**kwargs)
        self.preorder = preorder
        self.ordering = ordering
        self.candidates = candidates
        self.selection_folds = selection_folds
        self.selection_inner_folds = selection_inner_folds
        self.loss = loss
        self.penalty = penalty
        self.ctmle_estimand = ctmle_estimand
        self.oat_design = oat_design
        self._validate_ctmle_settings()
        if self.strategy == "ordered" and self.ordering is None and self.preorder is None:
            self.preorder = "logistic"
        if self.strategy == "oat" and self.oat_design is None:
            self.oat_design = "per_arm"

    def _validate_ctmle_settings(self) -> None:
        if self.loss not in ("auto", "loglik", "squared"):
            raise ValueError(f"loss must be 'auto', 'loglik' or 'squared'; got {self.loss!r}")
        if self.strategy not in ("greedy", "ordered", "discrete", "oat"):
            raise ValueError(
                f"strategy must be 'greedy', 'ordered', 'discrete' or 'oat'; got {self.strategy!r}"
            )
        if self.preorder not in (None, "logistic", "partial_correlation"):
            raise ValueError(
                f"preorder must be 'logistic' or 'partial_correlation'; got {self.preorder!r}"
            )
        if self.strategy == "discrete" and not self.candidates:
            raise ValueError("strategy='discrete' needs an explicit candidates= list")
        if self.strategy != "discrete" and self.candidates is not None:
            raise ValueError(
                f"candidates= only applies to strategy='discrete', not {self.strategy!r}"
            )
        if self.strategy != "ordered" and self.ordering is not None:
            raise ValueError(f"ordering= only applies to strategy='ordered', not {self.strategy!r}")
        if self.strategy != "ordered" and self.preorder is not None:
            raise ValueError(f"preorder= only applies to strategy='ordered', not {self.strategy!r}")
        if self.ordering is not None and self.preorder is not None:
            raise ValueError("preorder= cannot be combined with an explicit ordering=")
        if self.oat_design not in (None, "per_arm", "shared"):
            raise ValueError(f"oat_design must be 'per_arm' or 'shared'; got {self.oat_design!r}")
        if self.strategy != "oat" and self.oat_design is not None:
            raise ValueError(f"oat_design= only applies to strategy='oat', not {self.strategy!r}")
        if self.selection_folds < 2:
            raise ValueError(f"selection_folds must be at least 2; got {self.selection_folds}")
        if self.selection_inner_folds < 2:
            raise ValueError(
                f"selection_inner_folds must be at least 2; got {self.selection_inner_folds}"
            )
        if self.ctmle_estimand not in MEAN_GROUP_ESTIMANDS:
            raise ValueError(
                f"ctmle_estimand must be one of {sorted(MEAN_GROUP_ESTIMANDS)}; "
                f"got {self.ctmle_estimand!r}"
            )
        if self.targeting_scheme != "pooled":
            raise CapabilityError(
                "CTMLE implements the published pooled collaborative estimator only; "
                "targeting_scheme='fold' composes it with a different CV-TMLE estimator "
                "that has not been derived. Use targeting_scheme='pooled'."
            )

        if not is_selector_strategy(self.strategy) and self.ctmle_estimand != "ate":
            raise ValueError(
                "ctmle_estimand= does not apply to strategy='oat': ctmle3's "
                "outcome-adaptive construction targets all treatment-specific means together"
            )
        overridden = [] if is_selector_strategy(self.strategy) else self._selector_only_overrides()
        if overridden:
            raise ValueError(
                f"{', '.join(f'{name}=' for name in overridden)} configure selector "
                "strategies and do not apply to strategy='oat', which fits its mechanism on "
                "the outcome predictions and selects nothing"
            )

    def _selector_only_overrides(self) -> list[str]:
        """Which selector-only settings sit away from their constructor defaults.

        Read off :meth:`__init__`'s own signature rather than compared against literals,
        so the check cannot invert when a default changes -- a hard-coded ``!= 5`` would
        silently start refusing *every* ``strategy="oat"`` fit the day
        ``selection_folds``'s default moved.

        A value equal to the default passes whether or not it was written out, which is
        deliberate rather than a gap: whole-result persistence
        records every constructor setting by name, so a reloaded ``oat`` fit arrives with
        all four of these supplied explicitly and has to rebuild rather than raise.
        """
        defaults = inspect.signature(CTMLE.__init__).parameters
        return [
            name
            for name in ("selection_folds", "selection_inner_folds", "loss", "penalty")
            if getattr(self, name) != defaults[name].default
        ]

    # --------------------------------------------------------------- the hook

    def _nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        config: TMLEConfig,
        intermediate_value: float | None,
        seed: int | None = None,
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """Fit the outcome model once, then *select* the propensity model against it.

        The draw's ``seed`` reaches the selection folds as well as the nuisance fits, so
        a repeat redraws the split the *selection* was scored against too.  Holding that
        one fixed would leave every draw choosing its stopping point against the same
        partition, which is the noise ``repeats=`` exists to reduce.
        """
        if is_selector_strategy(self.strategy):
            self._preflight_selection_folds(data, seed)
        # Every collaborative strategy replaces the ordinary treatment mechanism: the
        # selectors fit their candidate path and OAT fits A on the Qbar vector.  Fit only
        # the shared outcome/missingness nuisances here so g(W) is not paid for and thrown
        # away first.
        base = self._fit_nuisances(
            data,
            folds,
            scaler,
            intermediate_value,
            seed=seed,
            fit_treatment=False,
        )
        if not is_selector_strategy(self.strategy):
            return self._outcome_adaptive_nuisances(data, base, seed=seed)
        selector = _Selector(self, data, base, config.g_bounds, intermediate_value, seed=seed)

        path = selector.build_path(train=None, tag="full")
        cv_risk, selection_folds = selector.cross_validate(path)
        selected = int(np.argmin(cv_risk))
        chosen = path[selected]

        # `base.diagnostics` carries no "propensity" entry, and deliberately so -- unlike
        # the `oat` branch below, which has one shared fit to report.  A selector's
        # mechanism comes off the candidate path, where `_Selector._fit_propensity_with`
        # returns predictions only and the intercept-only candidate C-TMLE often selects
        # involves no learner at all.  What used to sit under that key was the ordinary
        # g(W) super-learner table, describing a model the estimate does not use.
        # `nuisance_diagnostics` still reports the selected mechanism's calibration and
        # discrimination, which it computes from the array below.
        nuisance = replace(
            base,
            propensity=chosen.propensity,
            targeting_outcome=chosen.targeted,
            treatment_covariates=chosen.covariates,
        )
        selection = CTMLESelection(
            strategy=self.strategy,
            preorder=("custom" if self.ordering is not None else self.preorder)
            if self.strategy == "ordered"
            else None,
            estimand=self.ctmle_estimand,
            target_names=selector.target_names,
            loss=selector.loss_kind,
            penalized=self.penalty,
            path=tuple(candidate.covariates for candidate in path),
            n_steps=tuple(candidate.n_steps for candidate in path),
            train_risk=np.array([candidate.risk for candidate in path], dtype=float),
            train_loss=np.array([candidate.loss for candidate in path], dtype=float),
            penalty=np.array([candidate.penalty for candidate in path], dtype=float),
            treatment_risk=np.array([candidate.treatment_risk for candidate in path], dtype=float),
            cv_risk=cv_risk,
            selected=selected,
            folds=selection_folds,
            covariates=data.covariate_names,
        )
        return nuisance, {"ctmle": selection}

    def _selection_partition(self, data: CausalData, seed: int | None) -> Folds:
        """The split the cross-validated selection risk is scored over.

        One source for the partition, so the preflight below and
        :meth:`_Selector.cross_validate` cannot disagree about which split the search
        used. The same stratum the outer folds use, which under every accepted policy is
        none of them: the draw reads ``n``, the cluster labels and the seed.

        Parameters
        ----------
        data : CausalData
            The rows to split.
        seed : int or None
            The draw's seed. ``None`` means this estimator's own ``random_state``.

        Returns
        -------
        Folds
            The realized selection split, carrying its generator record.
        """
        return make_folds(
            data.n,
            self.selection_folds,
            stratify=self._fold_strata(data),
            cluster=data.cluster,
            random_state=self.random_state if seed is None else seed,
        )

    def _nested_partition(self, train_data: CausalData, seed: int | None) -> Folds:
        """The inner split that makes one selection fold's training predictions out of fold.

        Parameters
        ----------
        train_data : CausalData
            One selection fold's training rows.
        seed : int or None
            The draw's seed. ``None`` means this estimator's own ``random_state``.

        Returns
        -------
        Folds
            The realized inner split over those rows.
        """
        return make_folds(
            train_data.n,
            self.selection_inner_folds,
            stratify=self._fold_strata(train_data),
            cluster=train_data.cluster,
            random_state=self.random_state if seed is None else seed,
        )

    def _preflight_selection_folds(self, data: CausalData, seed: int | None) -> None:
        """Check the search's own splits before the first nuisance fit.

        The outer preflight in :meth:`~cleverly.estimators.TMLE._repeat_draws` covers the
        cross-fitting split, and a selector-based collaborative fit draws two more:
        the selection split the candidate risk is scored over, and the inner split each
        selection fold's training predictions are made out of. Both are drawn here rather
        than deep inside the search, so a partition that cannot carry the search is
        refused before any learner runs rather than after a candidate path has been built.

        The nested split is asked of each selection fold's training rows, which is the set
        it actually partitions.

        Parameters
        ----------
        data : CausalData
            The prepared data.
        seed : int or None
            The draw's seed.
        """
        selection = self._selection_partition(data, seed)
        self._check_training_support(
            data,
            (selection,),
            subject="C-TMLE selection",
            where="selection fold {fold}",
            remedy=_SELECTION_SPLIT_REMEDY,
        )
        for fold, (train, _) in enumerate(selection):
            train_data = data.subset(train)
            nested = self._nested_partition(train_data, seed)
            self._check_training_support(
                train_data,
                (nested,),
                subject=f"the nested split of C-TMLE selection fold {fold}",
                where="inner fold {fold}",
                remedy=_SELECTION_SPLIT_REMEDY,
            )

    def _outcome_adaptive_nuisances(
        self, data: CausalData, base: NuisanceEstimates, *, seed: int | None
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """Fit the outcome-adaptive treatment mechanism on the arm-specific Qbar predictions.

        ``oat_design="per_arm"`` regresses ``1{A = a}`` on the one column ``Qbar(a, W)`` for
        each arm. Column ``a`` of the returned mechanism holds ``P(A = a | Qbar(a, W))``, so
        the rows are not a distribution over the arms and the mechanism is marked off the
        simplex. Every arm's learner is resolved from the one seed of the draw.
        ``oat_design="shared"`` is the construction in ``ctmle3::LF_oat``: one categorical
        mechanism on the vector ``[Qbar(a, W): a in arms]``.

        Neither design has a candidate path or a parameter-specific risk. The ordinary
        K-column mean fluctuation targets the returned nuisances later in the shared TMLE
        pipeline. Its columns ``1{A = a} / g_a`` have disjoint supports, so under the
        per-arm design each arm keeps its own coefficient. With more than one outer fold,
        each mechanism is trained on predictions from an outcome model fitted wholly inside
        the same training fold. That nesting keeps an evaluation row's outcome out of its
        own generated mechanism design.
        """
        arms = data.arm_codes
        per_arm = self.oat_design == "per_arm"
        learner = self._resolve_learner(self.treatment_learner, task="classification", seed=seed)
        if base.folds.is_single and per_arm:
            columns: list[FloatArray] = []
            diagnostics: list[Any] = []
            for arm in arms:
                design = base.outcome.arms[arm][:, None]
                predictions, arm_diagnostics = cross_fit_predictions(
                    learner,
                    design,
                    (data.treatment == arm).astype(float),
                    data.weights,
                    base.folds,
                    task="classification",
                    predict_designs={"g": design},
                    groups=data.cluster,
                    clip=(0.0, 1.0),
                    n_jobs=self.n_jobs,
                )
                columns.append(predictions["g"])
                diagnostics.append(arm_diagnostics)
            propensity_values = np.column_stack(columns)
        elif base.folds.is_single:
            design = np.column_stack([base.outcome.arms[arm] for arm in arms])
            predictions, diagnostics = cross_fit_predictions(
                learner,
                design,
                data.treatment,
                data.weights,
                base.folds,
                task="classification",
                predict_designs={"g": design},
                groups=data.cluster,
                clip=(0.0, 1.0),
                classes=arms,
                n_jobs=self.n_jobs,
            )
            propensity_values = predictions["g"]
        else:
            base, propensity_values, diagnostics = self._cross_fit_outcome_adaptive(
                data, base, learner, seed=seed
            )
        risk_by_arm: dict[str, float] | None
        if per_arm:
            propensity = Propensity(propensity_values, arms, simplex=False)
            risk_by_arm = {
                data.arm_label(arm): _binary_risk(
                    propensity.arm(arm), (data.treatment == arm).astype(float), data.weights
                )
                for arm in arms
            }
            risk = float(sum(risk_by_arm.values()))
        else:
            propensity = Propensity(propensity_values, arms)
            observed_columns = np.array(
                [propensity.column_for(float(arm)) for arm in data.treatment], dtype=int
            )
            observed_probability = propensity.values[np.arange(data.n), observed_columns]
            risk = float(
                -np.sum(data.weights * np.log(np.clip(observed_probability, _LOSS_EPS, 1.0)))
            )
            risk_by_arm = None
        features = tuple(f"Qbar[{data.arm_label(arm)}]" for arm in arms)
        nuisance_diagnostics = dict(base.diagnostics)
        nuisance_diagnostics.pop("propensity", None)
        if (per_arm and any(diagnostics)) or (not per_arm and diagnostics):
            nuisance_diagnostics["propensity"] = diagnostics
        nuisance = replace(
            base,
            propensity=propensity,
            treatment_covariates=features,
            diagnostics=nuisance_diagnostics,
        )
        return nuisance, {
            "ctmle": CTMLEOutcomeAdaptiveFit(
                strategy="oat",
                treatment_features=features,
                treatment_risk=risk,
                design="per_arm" if per_arm else "shared",
                treatment_risk_by_arm=risk_by_arm,
            )
        }

    def _cross_fit_outcome_adaptive(
        self,
        data: CausalData,
        base: NuisanceEstimates,
        treatment_learner: Learner,
        *,
        seed: int | None,
    ) -> tuple[NuisanceEstimates, FloatArray, list[Any]]:
        """Cross-fit ``Qbar -> g`` as one fold-local nuisance algorithm.

        The first implementation stitched global out-of-fold ``Qbar`` predictions and
        cross-fit ``g`` on that matrix.  For a propensity fold ``v``, predictions on its
        training rows then came from other outcome folds whose fits included rows in
        ``v``.  Changing a validation outcome could therefore change that row's own
        propensity.  Here one outcome model fitted on ``train_v`` creates *both* sides
        of fold ``v``'s generated design before ``g_v`` is fitted on ``train_v``.

        Under ``oat_design="per_arm"`` fold ``v`` fits ``K`` binary mechanisms on
        ``train_v``, each regressing ``1{A = a}`` on the column ``Qbar_v(a, W)`` alone.  The
        nesting is the same.  This is the cross-validated construction of Benkeser, Cai and
        van der Laan (2020), Appendix D: fold-trained ``Qbar_v`` and ``g_v(. | Qbar_v)``, and
        one coefficient per arm pooled over the validation rows.

        The outcome predictions returned in ``base`` are replaced by the validation
        pieces from these very models.  That matters for stochastic learners: the
        outcome regression used to build a clever covariate cannot be a second refit
        that merely has the same learner settings.
        """
        arms = data.arm_codes
        per_arm = self.oat_design == "per_arm"
        folds = base.folds
        outcome_task = base.outcome_task
        outcome_learner = self._resolve_learner(self.outcome_learner, task=outcome_task, seed=seed)
        outcome_design = data.treatment_design()
        counterfactual = {arm: data.counterfactual_design(arm) for arm in arms}
        scaled = base.scaler.scale(data.outcome)
        indicators = {arm: (data.treatment == arm).astype(float) for arm in arms}
        jobs = [(fold, train, test) for fold, (train, test) in enumerate(folds)]

        def predict_outcome(model: Learner, design: FloatArray) -> FloatArray:
            return np.clip(predict_mean(model, design, outcome_task), 0.0, 1.0)

        def run_fold(
            fold: int, train: IntArray, test: IntArray
        ) -> tuple[
            int,
            IntArray,
            FloatArray,
            dict[float, FloatArray],
            FloatArray,
            Any,
            Any,
        ]:
            outcome_rows = train[data.observed[train]]
            if outcome_rows.size == 0:
                raise ValueError(
                    "a cross-fitting fold has no observed training outcomes. The split "
                    "is drawn from the seed alone and reads no treatment or outcome, so "
                    "trying fold counts or seeds until one fits would choose the "
                    "partition by the values it must not read. Fit in sample instead "
                    "(cross_fit=False on the engine, CrossFitting(enabled=False)), or "
                    "collect more observed outcomes"
                )
            outcome_model = fit_on_rows(
                outcome_learner,
                outcome_design,
                scaled,
                data.weights,
                outcome_rows,
                outcome_task,
                data.cluster,
            )
            observed = predict_outcome(outcome_model, outcome_design)
            by_arm = {
                arm: predict_outcome(outcome_model, design)
                for arm, design in counterfactual.items()
            }
            generated = np.column_stack([by_arm[arm] for arm in arms])
            treatment_diagnostic: Any
            if per_arm:
                columns = []
                arm_diagnostics = []
                for j, arm in enumerate(arms):
                    arm_model = fit_on_rows(
                        treatment_learner,
                        generated[:, [j]],
                        indicators[arm],
                        data.weights,
                        train,
                        "classification",
                        data.cluster,
                    )
                    columns.append(
                        np.clip(
                            predict_mean(arm_model, generated[test][:, [j]], "classification"),
                            0.0,
                            1.0,
                        )
                    )
                    arm_diagnostics.append(getattr(arm_model, "diagnostics_", None))
                propensity = np.column_stack(columns)
                treatment_diagnostic = (
                    arm_diagnostics if any(item is not None for item in arm_diagnostics) else None
                )
            else:
                treatment_model = fit_on_rows(
                    treatment_learner,
                    generated,
                    data.treatment,
                    data.weights,
                    train,
                    "classification",
                    data.cluster,
                )
                propensity = np.clip(
                    predict_probabilities(treatment_model, generated[test], arms), 0.0, 1.0
                )
                treatment_diagnostic = getattr(treatment_model, "diagnostics_", None)
            return (
                fold,
                test,
                observed[test],
                {arm: values[test] for arm, values in by_arm.items()},
                propensity,
                getattr(outcome_model, "diagnostics_", None),
                treatment_diagnostic,
            )

        observed = np.empty(data.n, dtype=float)
        by_arm = {arm: np.empty(data.n, dtype=float) for arm in arms}
        propensity = np.empty((data.n, len(arms)), dtype=float)
        outcome_diagnostics: list[Any] = []
        treatment_diagnostics: list[Any] = []
        for _, test, fold_observed, fold_arms, fold_g, q_diagnostic, g_diagnostic in map_parallel(
            run_fold, jobs, n_jobs=self.n_jobs
        ):
            observed[test] = fold_observed
            propensity[test] = fold_g
            for arm in arms:
                by_arm[arm][test] = fold_arms[arm]
            if q_diagnostic is not None:
                outcome_diagnostics.append(q_diagnostic)
            if g_diagnostic is not None:
                treatment_diagnostics.append(g_diagnostic)

        if per_arm and treatment_diagnostics:
            # One list per arm, each holding that arm's fold diagnostics, so the per-arm
            # calibration report reads the arm's own learner table.
            treatment_diagnostics = [
                [fold_entry[j] for fold_entry in treatment_diagnostics] for j in range(len(arms))
            ]
        diagnostics = dict(base.diagnostics)
        diagnostics.pop("outcome", None)
        diagnostics.pop("propensity", None)
        if outcome_diagnostics:
            diagnostics["outcome"] = outcome_diagnostics
        if treatment_diagnostics:
            diagnostics["propensity"] = treatment_diagnostics
        honest_base = replace(
            base,
            outcome=InitialFit(observed, by_arm),
            diagnostics=diagnostics,
        )
        return honest_base, propensity, treatment_diagnostics

    def _retarget_detailed(
        self, data: CausalData, nuisance: NuisanceEstimates, **kwargs: Any
    ) -> Any:
        """Continue targeting from the collaboratively selected ``Qbar*``.

        Keep that state separate from ``nuisance.outcome``: the latter is the initial
        outcome learner and is what calibration and risk diagnostics are about.  The
        replacement is local, so the result continues to expose both states faithfully.
        """
        initial = nuisance.targeting_outcome
        if initial is None:
            return super()._retarget_detailed(data, nuisance, **kwargs)
        working = replace(nuisance, outcome=initial, targeting_outcome=None)
        return super()._retarget_detailed(data, working, **kwargs)

    def _configured_for_refit(self, data: CausalData) -> CTMLE:
        """The estimator a refit on ``data`` runs, with an added covariate ordered last.

        An explicit ``ordering=`` must cover every covariate, so a refit that adds one,
        such as the ``random_common_cause`` refutation, would otherwise be refused.  The
        declared ordering ranks covariates by their prior relevance.  An added column of
        independent noise has none, so the copy places each added covariate after the
        declared ordering, in the order ``data`` holds them.
        :meth:`~cleverly.data.CausalData.with_extra_covariate` appends the column last in
        the same way.  A refit that drops a covariate keeps the refusal of an ordering
        that names an unknown covariate.

        A ``"discrete"`` fit takes the same rule for each candidate. The copy appends each
        covariate that ``data`` records as added
        (:attr:`~cleverly.data.CausalData.added_covariates`) to each candidate that does
        not already name it, in the order ``data`` holds them. A covariate the fit already
        had keeps its place, named or not. So a refit of a fit whose one candidate is the
        full adjustment set keeps one full candidate, and keeps the status of the fit it
        refits.

        Parameters
        ----------
        data : CausalData
            The prepared data the refit fits.

        Returns
        -------
        CTMLE
            This estimator when it declares no ordering and no candidates, or when they
            cover every covariate of ``data``. Otherwise a copy with the extended ordering
            or the extended candidates.
        """
        if self.strategy == "discrete" and self.candidates is not None:
            recorded = set(data.added_covariates)
            added = tuple(name for name in data.covariate_names if name in recorded)
            extended = [
                (*candidate, *(name for name in added if name not in candidate))
                for candidate in self.candidates
            ]
            if all(
                len(new) == len(old) for new, old in zip(extended, self.candidates, strict=True)
            ):
                return self
            configured = copy.copy(self)
            configured.candidates = extended
            return configured
        if self.ordering is None:
            return self
        declared = tuple(self.ordering)
        added = tuple(name for name in data.covariate_names if name not in declared)
        if not added:
            return self
        configured = copy.copy(self)
        configured.ordering = (*declared, *added)
        return configured

    def _resolve_estimands_for_data(self, data: CausalData) -> tuple[str, ...]:
        """Resolve targets, and refuse the designs a collaborative search has no result for.

        Runs before any split is drawn, so a refused design pays for no fold generation
        and no learner.
        """
        if self.incremental:
            # Asked before every other collaborative refusal, and before the estimand
            # resolution below, because ``_check_estimands`` reads the arm-indexed
            # estimand names and an incremental fit reports none of them. It would
            # otherwise refuse the tilt for having the wrong estimand list rather than
            # for the reason the tilt is refused.
            raise CapabilityError(
                "CTMLE and incremental= are not combined. C-TMLE cross-validates the "
                "*choice* of g against a loss for the targeted Qbar, and under an "
                "incremental intervention each candidate g defines a different "
                "parameter: Psi(delta) is built out of g. The search would be selecting "
                "between estimands rather than between estimators of one, and the risk "
                "it minimises would have no fixed target. Use a plain TMLE."
            )
        if data.cluster is not None:
            reason = (
                "No reviewed result covers clustered inference for the outcome-adaptive mechanism"
                if not is_selector_strategy(self.strategy)
                else (
                    "The search draws selection folds and, inside each of them, nested folds, "
                    "and no reviewed result covers a grouped draw of those splits or the "
                    "cluster-robust variance of the candidate the search then stops at"
                )
            )
            raise CapabilityError(
                f"C-TMLE has no clustered result. {reason}; docs/roadmap.md F22 tracks this "
                "stop. Drop id= from fit "
                "(PointTreatment(cluster=None)), or use the ordinary TMLE (TMLE, or "
                "TMLEMethod), which has a clustered result."
            )
        self._check_estimands(data)
        return super()._resolve_estimands_for_data(data)

    def _check_policies(self, data: CausalData) -> None:
        """Refuse a continuous dose before the shift check can suggest a shift.

        ``TMLE._check_policies`` runs first in the preflight, and it answers a dose with no
        ``policies=`` by suggesting one, which CTMLE does not fit.
        """
        if data.is_continuous_treatment:
            raise CapabilityError(_DISCRETE_TREATMENT_REFUSAL)
        super()._check_policies(data)

    def _check_estimands(self, data: CausalData) -> None:
        if data.is_continuous_treatment:
            raise CapabilityError(_DISCRETE_TREATMENT_REFUSAL)
        if data.has_intermediate:
            raise CapabilityError(
                "CTMLE does not compose either collaborative strategy with an intermediate "
                "outcome. "
                "Fit each controlled direct effect with TMLE instead."
            )
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms)
        conditional = [name for name in estimands if name not in MEAN_GROUP_ESTIMANDS]
        if conditional:
            raise CapabilityError(
                f"CTMLE does not support estimand(s) {conditional}: the ATT and ATC clever "
                "covariates condition on a random event, so a single collaboratively "
                "selected treatment model cannot serve them alongside the ATE. Request them "
                f"from a plain TMLE, or set estimands={sorted(MEAN_GROUP_ESTIMANDS)!r}."
            )
        if is_selector_strategy(self.strategy) and self.ctmle_estimand not in estimands:
            raise ValueError(
                f"ctmle_estimand={self.ctmle_estimand!r} is not among the requested estimands "
                f"{list(estimands)}; the selection has to be made for an estimand you are "
                "actually reporting."
            )
        supported = {"ate", "ey", "rr", "or"}
        if data.is_binary_treatment:
            supported.update(("ey1", "ey0"))
        if is_selector_strategy(self.strategy) and self.ctmle_estimand not in supported:
            raise CapabilityError(
                f"ctmle_estimand={self.ctmle_estimand!r} has no selector criterion; choose "
                f"from {sorted(supported)}. ey_obs, par and paf involve the observed law "
                "and require a separate collaborative derivation."
            )


class _Selector:
    """The candidate search and its cross-validated selector, for one fit.

    Holds the state a search needs -- the fixed outcome regression, the propensity
    cache, the row weights -- so that :class:`CTMLE` itself stays a plain,
    reusable settings object.
    """

    def __init__(
        self,
        estimator: CTMLE,
        data: CausalData,
        base: NuisanceEstimates,
        bounds: tuple[float, float],
        intermediate_value: float | None,
        seed: int | None = None,
        train_folds: Folds | None = None,
        train_mask: BoolArray | None = None,
    ) -> None:
        self.est = estimator
        self.data = data
        self.base = base
        self.bounds = bounds
        self.intermediate_value = intermediate_value
        if (train_folds is None) != (train_mask is None):
            raise ValueError("train_folds and train_mask must be supplied together")
        self.train_folds = train_folds
        self.train_mask = train_mask
        #: The cross-fitting draw this selector belongs to, under the same convention
        #: ``TMLE._folds`` uses: ``None`` means the estimator's own ``random_state``.
        #: Every split made below is drawn from it, so a repeat redraws the selection.
        self.seed = estimator.random_state if seed is None else seed
        self.scaled = base.scaler.scale(data.outcome)
        self.all_rows: IntArray = np.arange(data.n)
        self.loss_kind = (
            estimator.loss
            if estimator.loss != "auto"
            else ("loglik" if data.family == "binomial" else "squared")
        )
        self.learner: Learner = resolve_learner(
            estimator.treatment_learner,
            task="classification",
            n_folds=estimator.learner_folds,
            random_state=self.seed,
        )
        self.spec = estimator.targeting_spec()
        self._cache: dict[tuple[Any, ...], FloatArray] = {}
        self.reference = estimator._reference_arm(data)
        self.target_names = self._target_names()

    # ------------------------------------------------------------ propensities

    def propensity(
        self, covariates: tuple[str, ...], train: IntArray | None, tag: str
    ) -> Propensity:
        """``g(W_S)`` for one candidate covariate set, cached per search branch."""
        key = (tag, covariates)
        cached = self._cache.get(key)
        if cached is not None:
            return Propensity(cached, self.data.arm_codes)
        values = self._fit_propensity(covariates, train)
        self._cache[key] = values.values
        return values

    def _fit_propensity(self, covariates: tuple[str, ...], train: IntArray | None) -> Propensity:
        return self._fit_propensity_with(self.learner, covariates, train)

    def _fit_propensity_with(
        self, learner: Learner, covariates: tuple[str, ...], train: IntArray | None
    ) -> Propensity:
        data = self.data
        if not covariates:
            return self._intercept_propensity(train)

        columns = [data.covariate_names.index(name) for name in covariates]
        design = np.ascontiguousarray(data.covariates[:, columns])
        if self.train_folds is not None:
            assert self.train_mask is not None
            predictions, _ = cross_fit_predictions(
                learner,
                design,
                data.treatment,
                data.weights,
                self.train_folds,
                task="classification",
                predict_designs={"g1": design},
                fit_mask=self.train_mask,
                groups=data.cluster,
                clip=(0.0, 1.0),
                classes=data.arm_codes,
                n_jobs=self.est.n_jobs,
                remedy=_SELECTION_SPLIT_BACKSTOP,
            )
            return Propensity(predictions["g1"], data.arm_codes)
        if train is None:
            predictions, _ = cross_fit_predictions(
                learner,
                design,
                data.treatment,
                data.weights,
                self.base.folds,
                task="classification",
                predict_designs={"g1": design},
                groups=data.cluster,
                clip=(0.0, 1.0),
                classes=data.arm_codes,
                n_jobs=self.est.n_jobs,
            )
            return Propensity(predictions["g1"], data.arm_codes)

        model = fit_on_rows(
            learner,
            design,
            data.treatment,
            data.weights,
            train,
            "classification",
            data.cluster,
        )
        values = predict_probabilities(model, design, data.arm_codes)
        return Propensity(np.clip(values, 0.0, 1.0), data.arm_codes)

    def _intercept_propensity(self, train: IntArray | None) -> Propensity:
        """Weighted arm probabilities with no covariates -- every path's first candidate.

        Fit by hand rather than by handing a zero-column design to a learner, which
        scikit-learn rejects.  Cross-fitted like any other candidate so it is scored
        on the same footing.
        """
        data = self.data
        values = np.empty((data.n, data.n_arms))

        def proportions(rows: IntArray) -> FloatArray:
            total = float(np.sum(data.weights[rows]))
            return np.array(
                [
                    np.sum(data.weights[rows] * (data.treatment[rows] == arm)) / total
                    for arm in data.arm_codes
                ],
                dtype=float,
            )

        if self.train_folds is not None:
            assert self.train_mask is not None
            for fit_rows, test in self.train_folds:
                eligible = fit_rows[self.train_mask[fit_rows]]
                values[test] = proportions(eligible)
            return Propensity(values, data.arm_codes)
        if train is not None:
            values[:] = proportions(train)
            return Propensity(values, data.arm_codes)
        if self.base.folds.is_single:
            values[:] = proportions(self.all_rows)
            return Propensity(values, data.arm_codes)
        for fit_rows, test in self.base.folds:
            values[test] = proportions(fit_rows)
        return Propensity(values, data.arm_codes)

    # ---------------------------------------------------------------- targeting

    def submodel(self, propensity: Propensity) -> Submodel:
        """The ``mean`` clever covariate at a candidate propensity."""
        nuisance = replace(self.base, propensity=propensity)
        return build_submodel(
            self.data,
            nuisance,
            "mean",
            bounds=self.bounds,
            nuisance_bound=self.est.nuisance_bound,
            intermediate_value=self.intermediate_value,
        )

    def target(
        self, initial: InitialFit, submodel: Submodel, rows: IntArray
    ) -> tuple[InitialFit, FloatArray]:
        """Solve the fluctuation on ``rows``, then apply it to the whole sample.

        Fitting and applying are separated because the cross-validated selector needs
        an ``epsilon`` fit on training rows and evaluated on held-out ones.
        """
        fluctuation = solve_submodel(
            self.scaled[rows],
            _restrict_fit(initial, rows),
            restrict(submodel, rows),
            self.data.weights[rows],
            self.data.observed[rows],
            self.spec,
            warn=False,
        )
        return self.apply(initial, submodel, fluctuation.epsilon), fluctuation.epsilon

    def apply(self, initial: InitialFit, submodel: Submodel, epsilon: FloatArray) -> InitialFit:
        """Move the predictions along the submodel by a fitted ``epsilon``."""
        est = self.est
        moved = weighted_form(submodel, self.data.weights)[0] if est.target_weights else submodel
        if est.fluctuation == "linear":
            check_matching_arms(initial, moved)
            return InitialFit(
                initial.observed + moved.observed @ epsilon,
                {
                    level: values + moved.arms[level] @ epsilon
                    for level, values in initial.arms.items()
                },
            )
        return apply_logistic(initial.shrunk(est.alpha), moved, epsilon, est.alpha)

    # --------------------------------------------------------------------- loss

    def loss(self, targeted: InitialFit, rows: IntArray) -> float:
        """The weighted loss of a targeted fit, summed over the observed ``rows``."""
        observed = self.data.observed[rows]
        index = rows[observed]
        if index.size == 0:
            return 0.0
        y = self.scaled[index]
        w = self.data.weights[index]
        if self.loss_kind == "squared":
            return float(np.sum(w * (y - targeted.observed[index]) ** 2))
        q = np.clip(targeted.observed[index], _LOSS_EPS, 1.0 - _LOSS_EPS)
        return float(-np.sum(w * (y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))

    def penalty(self, targeted: InitialFit, submodel: Submodel, rows: IntArray) -> float:
        """Trace variance plus squared vector bias on ``rows``.

        On the scaled outcome, so it is commensurate with :meth:`loss`.
        """
        return _penalty_of(self.influence(targeted, submodel, rows))

    def influence(self, targeted: InitialFit, submodel: Submodel, rows: IntArray) -> FloatArray:
        """The joint target's efficient influence curve, on the scaled outcome."""
        means = counterfactual_means(
            self.scaled[rows],
            _restrict_fit(targeted, rows),
            restrict(submodel, rows),
            self.data.weights[rows],
            self.data.observed[rows],
        )
        estimand = self.est.ctmle_estimand
        if estimand == "ey1":
            return np.asarray(means[1.0].influence_curve, dtype=float)
        if estimand == "ey0":
            return np.asarray(means[0.0].influence_curve, dtype=float)
        if estimand == "ey":
            return np.column_stack([means[arm].influence_curve for arm in self.data.arm_codes])

        reference = means[self.reference]
        curves = []
        for arm in self.data.arm_codes:
            if arm == self.reference:
                continue
            mean = means[arm]
            if estimand == "ate":
                curve = mean.influence_curve - reference.influence_curve
            elif estimand == "rr":
                _, curve = log_ratio_influence(
                    mean.psi, mean.influence_curve, reference.psi, reference.influence_curve
                )
            else:
                _, curve = log_odds_ratio_influence(
                    mean.psi, mean.influence_curve, reference.psi, reference.influence_curve
                )
            curves.append(np.asarray(curve, dtype=float))
        matrix = np.column_stack(curves)
        return matrix[:, 0] if matrix.shape[1] == 1 else matrix

    def _target_names(self) -> tuple[str, ...]:
        """Registry-compatible labels for the jointly optimized components."""
        stem = self.est.ctmle_estimand
        if stem in {"ey1", "ey0"}:
            return (stem,)
        if stem == "ey":
            return tuple(
                parameter_name("ey", arm=self.data.arm_label(arm)) for arm in self.data.arm_codes
            )
        contrasts = tuple(arm for arm in self.data.arm_codes if arm != self.reference)
        if self.data.is_binary_treatment:
            return (stem,)
        return tuple(
            parameter_name(
                stem,
                arm=self.data.arm_label(arm),
                versus=self.data.arm_label(self.reference),
            )
            for arm in contrasts
        )

    def score(self, candidate: _Candidate, rows: IntArray) -> float:
        """The selection criterion for a candidate, evaluated on a set of rows.

        The same quantity :attr:`_Candidate.risk` holds for the rows the candidate was
        fit on; the point of computing it here is to evaluate it on held-out rows.
        """
        value = self.loss(candidate.targeted, rows)
        if self.est.penalty:
            value += self.penalty(candidate.targeted, candidate.submodel, rows)
        return value

    # ------------------------------------------------------------- path search

    def build_path(self, train: IntArray | None, tag: str) -> list[_Candidate]:
        """The candidate sequence, fit on ``train`` (or cross-fitted when ``None``)."""
        rows = self.all_rows if train is None else train
        if self.est.strategy == "discrete":
            return self._discrete_path(rows, train, tag)
        order = self._ordering(rows, train) if self.est.strategy == "ordered" else None
        return self._forward_path(rows, train, tag, order)

    def _discrete_path(self, rows: IntArray, train: IntArray | None, tag: str) -> list[_Candidate]:
        assert self.est.candidates is not None
        path = []
        for names in self.est.candidates:
            covariates = tuple(names)
            path.append(self._candidate(covariates, self.base.outcome, rows, train, tag, 1))
        return path

    def _forward_path(
        self,
        rows: IntArray,
        train: IntArray | None,
        tag: str,
        order: tuple[str, ...] | None,
    ) -> list[_Candidate]:
        """Build a nested sequence, greedily or in a fixed order.

        The two searches differ only in how the next covariate is picked; both retry from
        one further targeted fit before forcing the best available addition.
        """
        pool = list(order) if order is not None else list(self.data.covariate_names)
        first = self._candidate((), self.base.outcome, rows, train, tag, 1)
        path = [first]

        base_fit = self.base.outcome
        current = first
        n_steps = 1
        stepped = False

        while pool:
            trials = pool[:1] if order is not None else pool
            scored = [
                self._candidate((*current.covariates, name), base_fit, rows, train, tag, n_steps)
                for name in trials
            ]
            best = min(scored, key=lambda candidate: candidate.risk)
            if best.risk > current.risk and not stepped:
                # No addition helps from here.  Take a further fluctuation step and
                # search again from the targeted fit.  The next pass accepts its best
                # addition even when the recorded penalized risk rises, matching the
                # forced-addition construction of van der Laan & Gruber (2010).
                base_fit = current.targeted
                n_steps += 1
                stepped = True
                continue
            stepped = False
            pool.remove(best.covariates[-1])
            path.append(best)
            current = best
        return path

    def _candidate(
        self,
        covariates: tuple[str, ...],
        initial: InitialFit,
        rows: IntArray,
        train: IntArray | None,
        tag: str,
        n_steps: int,
    ) -> _Candidate:
        propensity = self.propensity(covariates, train, tag)
        submodel = self.submodel(propensity)
        targeted, epsilon = self.target(initial, submodel, rows)
        loss = self.loss(targeted, rows)
        penalty = self.penalty(targeted, submodel, rows) if self.est.penalty else 0.0
        g = np.clip(propensity.values[rows], _LOSS_EPS, 1.0 - _LOSS_EPS)
        columns = np.array(
            [propensity.column_for(float(arm)) for arm in self.data.treatment[rows]], dtype=int
        )
        w = self.data.weights[rows]
        treatment_risk = float(-np.sum(w * np.log(g[np.arange(rows.size), columns])))
        return _Candidate(
            covariates=covariates,
            propensity=propensity,
            submodel=submodel,
            targeted=targeted,
            epsilon=epsilon,
            n_steps=n_steps,
            loss=loss,
            penalty=penalty,
            treatment_risk=treatment_risk,
            risk=loss + penalty,
        )

    def _ordering(
        self,
        rows: IntArray,
        train: IntArray | None,
    ) -> tuple[str, ...]:
        """Published logistic or partial-correlation order for the scalable search."""
        if self.est.ordering is not None:
            names = tuple(self.est.ordering)
            unknown = [name for name in names if name not in self.data.covariate_names]
            if unknown:
                raise ValueError(
                    f"ordering names unknown covariate(s) {unknown}; "
                    f"available: {list(self.data.covariate_names)}"
                )
            missing = [name for name in self.data.covariate_names if name not in set(names)]
            if missing:
                raise ValueError(
                    f"ordering must cover every covariate; missing {missing}. Use "
                    "strategy='discrete' to search a restricted set of models."
                )
            return names

        if self.est.preorder == "logistic":
            score_values = []
            logistic = LogisticRegression(C=1e6, max_iter=1000, random_state=self.seed)
            for name in self.data.covariate_names:
                propensity = self._fit_propensity_with(logistic, (name,), train)
                submodel = self.submodel(propensity)
                targeted, _ = self.target(self.base.outcome, submodel, rows)
                score_values.append(self.loss(targeted, rows))
            scores = np.asarray(score_values, dtype=float)
            order = np.argsort(scores, kind="stable")
        else:
            usable = rows[self.data.observed[rows]]
            residual = self.scaled[usable] - self.base.outcome.observed[usable]
            treatment = self.data.treatment[usable]
            conditional = np.column_stack(
                [(treatment == arm).astype(float) for arm in self.data.arm_codes[1:]]
            )
            weights = self.data.weights[usable]
            scores = np.array(
                [
                    abs(
                        _weighted_partial_correlation(
                            residual,
                            self.data.covariates[usable, column],
                            conditional,
                            weights,
                        )
                    )
                    for column in range(self.data.covariates.shape[1])
                ]
            )
            order = np.argsort(-scores, kind="stable")
        return tuple(self.data.covariate_names[j] for j in order)

    # -------------------------------------------------------------- selection

    def cross_validate(self, path: Sequence[_Candidate]) -> tuple[FloatArray, Folds]:
        """Cross-validated risk of each position in the candidate sequence, and its split.

        The sequence is rebuilt inside every training fold -- which covariate lands at
        position ``k`` may differ from fold to fold, and that is the point: what is
        being selected is *how far along the sequence to stop*, not a fixed covariate
        set.  Scoring a fixed set instead would leak the full-sample search into the
        validation folds.

        The loss accumulates fold by fold, but the penalty is computed once from the
        *pooled* cross-validated influence curve -- every row's contribution coming
        from the fold that held it out.  That is the ``cvVar + n * cvBias^2`` of the
        published criterion, and pooling matters in practice: a variance estimated
        inside a single validation fold is noisy enough to swamp the difference
        between two candidates it is supposed to be telling apart.

        The partition is returned rather than left as a local because it is the one
        thing about a C-TMLE fit that an outside party has to be able to *reproduce*
        rather than infer: a paired comparison hands the same rows and the same split
        to another implementation, and a split recomputed from the same rule is not
        evidence that it is the split this fit used.
        """
        data = self.data
        folds = self.est._selection_partition(data, self.seed)
        loss = np.zeros(len(path))
        dimension = len(self.target_names)
        influence = np.zeros((len(path), data.n, dimension))
        for fold, (train, test) in enumerate(folds):
            train_folds, train_mask = self._nested_folds(train)
            fold_base = self._selection_base(train_folds, train_mask)
            fold_data = data.subset(train)
            fold_bounds = resolve_g_bounds(
                self.est.g_bounds, self.est._bounds_n(fold_data), for_att=False
            )
            fold_selector = _Selector(
                self.est,
                data,
                fold_base,
                fold_bounds,
                self.intermediate_value,
                seed=self.seed,
                train_folds=train_folds,
                train_mask=train_mask,
            )
            fold_path = fold_selector.build_path(train=train, tag=f"cv{fold}")
            if len(fold_path) < len(path):
                raise RuntimeError(
                    f"selection fold {fold} produced {len(fold_path)} candidates but the "
                    f"full-sample search produced {len(path)}; the candidate sequence must "
                    "have the same length in every fold"
                )
            for index in range(len(path)):
                candidate = fold_path[index]
                loss[index] += fold_selector.loss(candidate.targeted, test)
                curve = np.asarray(
                    fold_selector.influence(candidate.targeted, candidate.submodel, test),
                    dtype=float,
                )
                influence[index, test] = curve.reshape(test.size, dimension)
        if not self.est.penalty:
            return loss, folds
        return loss + np.array([_penalty_of(row) for row in influence]), folds

    def _nested_folds(self, train: IntArray) -> tuple[Folds, BoolArray]:
        """Inner cross-fit on ``train`` plus one full-training fit for validation rows."""
        data = self.data
        train_data = data.subset(train)
        inner = self.est._nested_partition(train_data, self.seed)
        assignment = np.full(data.n, inner.n_folds, dtype=np.int64)
        assignment[train] = inner.assignment
        mask = np.zeros(data.n, dtype=bool)
        mask[train] = True
        nested = Folds(assignment, inner.n_folds + 1)
        check_integrity(nested, cluster=data.cluster)
        return nested, mask

    def _selection_base(self, train_folds: Folds, train_mask: BoolArray) -> NuisanceEstimates:
        """Cross-fitted nuisances trained only on one selection fold's training rows."""
        data = self.data
        scaled = self.base.scaler.scale(data.outcome)
        outcome_task: Task = "classification" if data.family == "binomial" else "regression"
        learner = self.est._resolve_learner(
            self.est.outcome_learner, task=outcome_task, seed=self.seed
        )
        design = data.treatment_design()
        outcome_out, _ = cross_fit_predictions(
            learner,
            design,
            scaled,
            data.weights,
            train_folds,
            task=outcome_task,
            predict_designs={
                "observed": design,
                **{f"arm@{arm}": data.counterfactual_design(arm) for arm in data.arm_codes},
            },
            fit_mask=train_mask & data.observed,
            groups=data.cluster,
            clip=(0.0, 1.0),
            n_jobs=self.est.n_jobs,
            remedy=_SELECTION_SPLIT_BACKSTOP,
        )
        outcome = InitialFit(
            outcome_out["observed"],
            {arm: outcome_out[f"arm@{arm}"] for arm in data.arm_codes},
        )

        missingness = self.base.missingness
        if data.has_missing_outcome:
            missing_learner = self.est._resolve_learner(
                self.est.missingness_learner,
                task="classification",
                fallback=self.est.treatment_learner,
                seed=self.seed,
            )
            missing_design = data.missingness_design()
            missing_out, _ = cross_fit_predictions(
                missing_learner,
                missing_design,
                data.observed.astype(float),
                data.weights,
                train_folds,
                task="classification",
                predict_designs={
                    f"arm@{arm}": data.counterfactual_design(arm) for arm in data.arm_codes
                },
                fit_mask=train_mask,
                groups=data.cluster,
                clip=(0.0, 1.0),
                n_jobs=self.est.n_jobs,
                remedy=_SELECTION_SPLIT_BACKSTOP,
            )
            missingness = np.column_stack([missing_out[f"arm@{arm}"] for arm in data.arm_codes])
        return replace(
            self.base,
            outcome=outcome,
            targeting_outcome=None,
            scaler=self.base.scaler,
            folds=train_folds,
            missingness=missingness,
            diagnostics={},
        )


def _binary_risk(predicted: FloatArray, indicator: FloatArray, weights: FloatArray) -> float:
    """The weighted negative log likelihood of one binary mechanism.

    Parameters
    ----------
    predicted : ndarray of float
        The fitted probability ``P(A = a | Qbar(a, W))`` of each row.
    indicator : ndarray of float
        ``1{A = a}`` for each row.
    weights : ndarray of float
        The row weights.

    Returns
    -------
    float
        ``-sum_i w_i [I_i log g_i + (1 - I_i) log(1 - g_i)]``, with ``g`` floored away from
        0 and 1 at the loss floor.
    """
    g = np.clip(np.asarray(predicted, dtype=float), _LOSS_EPS, 1.0 - _LOSS_EPS)
    return float(-np.sum(weights * (indicator * np.log(g) + (1.0 - indicator) * np.log1p(-g))))


def _penalty_of(influence_curve: FloatArray) -> float:
    """``tr(Cov(D*)) + n ||mean(D*)||^2`` for a scalar or vector target."""
    curve = np.asarray(influence_curve, dtype=float)
    if curve.ndim == 1:
        curve = curve[:, None]
    if curve.shape[0] < 2:
        return 0.0
    variance = np.sum(np.var(curve, axis=0, ddof=1))
    squared_bias = curve.shape[0] * np.sum(np.mean(curve, axis=0) ** 2)
    return float(variance + squared_bias)


def _weighted_partial_correlation(
    left: FloatArray, right: FloatArray, conditional: FloatArray, weights: FloatArray
) -> float:
    """Weighted correlation of residuals after projecting both variables on ``A``."""
    condition = np.asarray(conditional, dtype=float)
    if condition.ndim == 1:
        condition = condition[:, None]
    design = np.column_stack([np.ones(left.size), condition])
    root = np.sqrt(weights)
    weighted_design = design * root[:, None]

    def residual(values: FloatArray) -> FloatArray:
        coefficient = np.linalg.lstsq(weighted_design, values * root, rcond=None)[0]
        return values - design @ coefficient

    left_residual = residual(np.asarray(left, dtype=float))
    right_residual = residual(np.asarray(right, dtype=float))
    left_centered = left_residual - np.average(left_residual, weights=weights)
    right_centered = right_residual - np.average(right_residual, weights=weights)
    numerator = float(np.sum(weights * left_centered * right_centered))
    denominator = float(
        np.sqrt(np.sum(weights * left_centered**2) * np.sum(weights * right_centered**2))
    )
    return 0.0 if denominator <= np.finfo(float).eps else numerator / denominator


def _restrict_fit(fit: InitialFit, index: IntArray) -> InitialFit:
    """Row-subset an initial fit, the counterpart of :func:`.submodel.restrict`."""
    return fit.map_arms(lambda values: values[index])


#: The start of every ``logistic_plugin`` refusal: what R ``ctmle`` defines the term for.
_LOGISTIC_PLUGIN_SCOPE = (
    "the logistic-estimation term of R ctmle's calc_varIC is defined for a binary "
    "treatment whose propensity is fitted once on all rows without weights, missing "
    "outcomes or an intermediate variable. This fit "
)


@dataclass(frozen=True)
class LogisticPlugin:
    """The plug-in variance R ``ctmle`` reports for a selector C-TMLE fit.

    A diagnostic.  The fit's status does not change: a selecting fit keeps
    ``"working_mechanism_plugin"``, and a ``"discrete"`` fit with one full candidate keeps
    the TMLE status.  No registered study measures the coverage of this interval.  The
    ``plugin_logistic_`` prefix makes no coverage claim.

    Parameters
    ----------
    name : str
        The estimand: ``"ate"``, ``"ey1"`` or ``"ey0"``.
    psi : float
        The reported estimate.
    influence_curve : ndarray
        ``D* + term1 I^{-1} S`` on the outcome scale, with ``S = (A - g) W~`` the
        logistic score of the selected candidate's covariates and an intercept.
    plugin_logistic_std_error : float
        The plug-in standard error of :attr:`influence_curve`, at the observation or
        cluster unit.
    plugin_logistic_interval : tuple of float
        The Wald interval at the fit's level, from that standard error.
    correction_applied : bool
        ``False`` when the selected candidate has no covariates, or when the information
        matrix is singular.  R ``ctmle`` then reports the ``D*`` variance, and so does this.
    """

    name: str
    psi: float
    influence_curve: FloatArray
    plugin_logistic_std_error: float
    plugin_logistic_interval: tuple[float, float]
    correction_applied: bool


def _logistic_plugin_refusal(result: TMLEResult) -> str | None:
    """Why ``result`` is outside the term's definition, or ``None``."""
    data = result.data
    selection = result.extra.get("ctmle")
    if not isinstance(selection, CTMLESelection):
        if isinstance(selection, CTMLEOutcomeAdaptiveFit):
            return "used strategy='oat', which fits no logistic propensity on covariates"
        return "is not a selector C-TMLE fit (strategy 'greedy', 'ordered' or 'discrete')"
    if not data.is_binary_treatment:
        return f"has {data.n_arms} treatment arms"
    if result.config.cross_fit:
        return "cross-fits its propensity"
    if data.is_weighted:
        return "carries observation weights"
    if data.has_missing_outcome:
        return "has missing outcomes"
    if result.intermediate_value is not None:
        return "targets a level of an intermediate variable"
    return None


def logistic_plugin(result: TMLEResult) -> dict[str, LogisticPlugin]:
    r"""The ``calc_varIC(ICg = TRUE)`` variance R ``ctmle`` reports, as a diagnostic.

    R ``ctmle`` 0.1.2 at commit ``18de559`` reports ``var.psi`` and ``CI`` from
    ``calc_varIC(..., ICg = TRUE)`` at the selected candidate
    (``R/functions_discrete.R``, lines 200 and 201, and ``R/ctmle_discrete.R``, lines
    181 to 196).  That variance adds to :math:`D^*` the term for the estimation of the
    candidate's logistic propensity (``R/functions.R``, lines 39 to 60):

    .. math::

        IC_i = D^*_i + \text{term1}\, I^{-1} (A_i - g_i) \tilde W_i,

    with :math:`\tilde W = [1, W_S]` the selected covariates and an intercept,
    :math:`I = P_n[g(1-g)\tilde W \tilde W^\top]`, and :math:`\text{term1}` the
    derivative of :math:`P_n D^*` in the logistic coefficients.  For ``ey1`` it is
    :math:`P_n[-(Y - Q)\tilde W A(1-g)/g]`, for ``ey0``
    :math:`P_n[(Y - Q)\tilde W (1-A)g/(1-g)]`, and ``ate`` is their difference.

    :attr:`~cleverly.inference.ParameterEstimate.plugin_interval` is the ``D*`` interval,
    the first value ``calc_varIC`` returns.  This function returns the second.  The term
    is a parametric-estimation correction only when the treatment learner is an
    unpenalised main-terms logistic regression.  For any other learner it is R's formula
    evaluated at that learner's prediction.  Neither interval carries a coverage claim,
    and the fit's status does not change.

    Parameters
    ----------
    result : TMLEResult
        A selector C-TMLE fit of a binary treatment, fitted in sample.

    Returns
    -------
    dict of str to LogisticPlugin
        One entry for each of ``ate``, ``ey1`` and ``ey0`` that the fit reports.

    Raises
    ------
    CapabilityError
        When the fit is not a selector C-TMLE fit, or has more than two arms, a
        cross-fitted propensity, observation weights, missing outcomes or an
        intermediate variable.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from cleverly.datasets import make_binary_outcome
    >>> from cleverly.estimators import CTMLE, logistic_plugin
    >>> frame, _ = make_binary_outcome(n=200, seed=0)
    >>> result = CTMLE(
    ...     outcome_learner=LogisticRegression(max_iter=1000),
    ...     treatment_learner=LogisticRegression(C=1e6, max_iter=1000),
    ...     strategy="greedy",
    ...     cross_fit=False,
    ...     estimands=("ate",),
    ... ).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"]).single()
    >>> diagnostic = logistic_plugin(result)["ate"]
    >>> diagnostic.plugin_logistic_std_error > 0
    True
    """
    reason = _logistic_plugin_refusal(result)
    if reason is not None:
        raise CapabilityError(_LOGISTIC_PLUGIN_SCOPE + reason)
    data = result.data
    selection = result.extra["ctmle"]
    nuisance = result.nuisance
    g1 = nuisance.propensity.bounded(result.config.g_bounds)[:, 1]
    targeted = result.fluctuations["mean"].targeted
    a = np.asarray(data.treatment, dtype=float)
    scale = nuisance.scaler.range
    # The residual on the outcome scale, where the reported curve lives.
    residual = scale * (nuisance.scaler.scale(data.outcome) - targeted.observed)
    chosen = [data.covariate_names.index(name) for name in selection.selected_covariates]
    design = np.column_stack([np.ones(data.n), data.covariates[:, chosen]])
    corrected = bool(chosen)
    projection = None
    if corrected:
        information = (design * (g1 * (1.0 - g1))[:, None]).T @ design / data.n
        score = design * (a - g1)[:, None]
        try:
            projection = np.linalg.solve(information, score.T).T
        except np.linalg.LinAlgError:
            corrected = False
    pieces = {
        "ey1": -(residual * a * (1.0 - g1) / g1),
        "ey0": residual * (1.0 - a) * g1 / (1.0 - g1),
    }
    pieces["ate"] = pieces["ey1"] - pieces["ey0"]
    out: dict[str, LogisticPlugin] = {}
    for name in ("ate", "ey1", "ey0"):
        if name not in result.estimates:
            continue
        estimate = result.estimates[name]
        curve = np.asarray(estimate.influence_curve, dtype=float)
        if corrected and projection is not None:
            term1 = (design * pieces[name][:, None]).mean(axis=0)
            curve = curve + projection @ term1
        variance = influence_variance(curve, data.cluster)
        std_error = float(np.sqrt(variance)) if np.isfinite(variance) else float("nan")
        out[name] = LogisticPlugin(
            name=name,
            psi=estimate.psi,
            influence_curve=curve,
            plugin_logistic_std_error=std_error,
            plugin_logistic_interval=wald_ci(estimate.psi, std_error, result.config.alpha_sig),
            correction_applied=corrected and projection is not None,
        )
    return out
