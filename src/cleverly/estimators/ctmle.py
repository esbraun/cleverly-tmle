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

Five ways of building the sequence are available, mirroring the entry points of R's
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
    Cross-validated selection among an explicit list of candidate covariate sets.  Each
    candidate fluctuates the initial fit once.

``strategy="lasso"``
    The C-TMLE1 path of R ``ctmleGlmnet`` (Ju et al., 2019, C-LASSO Section 4.2).  The
    candidates are the L1-penalized logistic mechanisms on a strictly decreasing penalty
    grid, on the ``glmnet`` objective.  ``lambdas=None`` uses ``glmnet``'s default grid,
    with the penalty 0 appended.  The sequence is built by the indexed rule below, and
    after the cross-validation a final step fluctuates the selected candidate's targeted
    fit with each smaller penalty and reports the one with the smallest unpenalized loss.

``strategy="learners"``
    The declared sequence of R ``build_gn_seq`` and ``ctmleGeneral``.  The candidates are
    the entries of ``treatment_candidates=``, ordered from the least to the most adaptive,
    then ``treatment_learner`` on every covariate.  An entry is an unfitted learner, or a
    :class:`PropensityArrays` pair of full-sample and out-of-fold predictions under a
    declared ``selection_split=``.

The two indexed strategies share one construction rule, the template of van der Laan and
Gruber (2010), Section 2.4.  From the current initial fit, every remaining candidate is
tried and the best is taken.  The candidates it skipped are recorded as fluctuations of
the current initial fit, its targeted fit becomes the next initial fit, and the search
stops when the best try no longer improves on the best criterion so far.

The five strategies above build a candidate path and cut it at a data-chosen stopping
index.  **The package supplies no inference for any of them.**  ``ci``, ``pvalue`` and
``std_error`` raise :class:`~cleverly.exceptions.CapabilityError`: no result shows the
reported curve is this estimator's influence curve when the selected working mechanism is
not consistent for the treatment law.  The point estimate, the selection path and the
curve remain, and ``plugin_std_error`` and ``plugin_interval`` report the retained
diagnostic under names that make no coverage claim.  F18 in ``docs/roadmap.md`` is the
condition that reopens this.  Two declarations are the exception.  When a ``"discrete"``
list holds one candidate and that candidate is the full adjustment set, or a
``"learners"`` list is empty, nothing is selected, the estimator is the ordinary TMLE, and
the fit takes the ordinary TMLE status.

``strategy="oat"``
    The outcome-adaptive treatment mechanism of Benkeser, Cai and van der Laan (2020).
    This is not a fourth candidate sequence: it fits treatment on the arm-specific
    outcome predictions and then uses the ordinary all-arm mean fluctuation.
    Consequently it has no candidate path, no parameter-specific selector loss and no
    stopping index.  ``oat_design`` selects one of two designs.  ``"per_arm"``, the
    default, regresses ``1{A = a}`` on the one column ``Qbar(a, W)`` for each arm.
    ``"shared"`` is ``ctmle3::LF_oat``: one categorical mechanism on the vector
    ``[Qbar(a, W): a in arms]``.

    The per-arm design also covers baseline strata and missing data, for point estimates.
    With missing outcomes it fits ``P(Delta = 1 | A = a, Qbar(a, W))`` on the same column,
    and with a missing treatment (in sample) ``P(Delta_A = 1 | Qbar(a, W))`` as well, so
    the product of the factors is ``P(C_a = 1 | Qbar(a, W))`` for the composite indicator
    ``C_a = Delta_A Delta 1{A = a}``.  With baseline strata every factor is fitted inside
    each stratum.  The natural-course mean reads the observation factor at the realised
    arm.  The missingness tilt keeps the full-covariate model ``P(Delta = 1 | A, W)``, held
    in ``tilt_missingness``.

    **It also trades away one leg of double robustness, and that is the reason to reach
    for it deliberately rather than as a default.**  The selector strategies end their
    candidate path at a mechanism on every covariate, so a search that keeps going recovers
    :math:`g_0` when that last candidate is consistent; collaborative double robustness is
    retained.  Here :math:`W` never enters :math:`g` at all.  The fitted
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
    generated feature.  Their Theorem 1 states the influence curve for one
    treatment-specific mean with one scalar design, and the per-arm design is that
    construction for each arm.  The theorem needs its condition (v) (preprint Appendix F,
    the remainder ``R24``), and that condition fails for an estimated outcome regression:
    the reported curve then omits a first-order generated-design term.  The registered
    study ``ctmle-oat-per-arm`` measured standard errors 8 to 24 percent too small, flat in
    ``n``, while its oracle-design cells were calibrated.  So every ``oat`` fit, of either
    design, takes the ``"generated_design_plugin"`` status (:data:`OAT_PER_ARM_INFERENTIAL`
    is ``False``): ``ci``, ``pvalue`` and ``std_error`` raise
    :class:`~cleverly.exceptions.CapabilityError`, and ``plugin_std_error`` and
    ``plugin_interval`` report the retained diagnostic.  F19 in ``docs/roadmap.md`` holds
    the generated-design term.  ``ctmle3`` does not cross-fit this fit at all
    -- ``LF_oat`` pins ``cv_fold = -1`` -- so non-cross-fitted parity and a cross-fitted
    result would have different sources.

``strategy="derivative"``
    C-TMLE0 of Ju, Wyss et al. (2019), C-LASSO Section 4.2, equation (6), and R ``ctmle``
    type 3.  It selects nothing.  One TMLE update adds, for each arm, the column
    ``h_a (g_a - g+_a) / g_a`` to the clever covariate ``h_a``, where ``g+`` is a partner
    mechanism and both mechanisms are bounded at ``g_bounds``.  The ``2K`` columns contain
    the ``K`` clever covariates, so the update solves the efficient influence curve
    equation, and the fit takes the ordinary TMLE status.  The pair comes from the lasso
    (``lambdas=None`` chooses the mechanism penalty by ``learner_folds``-fold weighted
    deviance cross-validation and takes its nearest grid neighbour as the partner, or
    ``lambdas=(mechanism, partner)``), or from ``treatment_candidates=(mechanism,
    partner)`` as learners or full-sample arrays.  The interval does not claim the
    collaborative reading of Ju, Chambaz and van der Laan (2018), Section 5.3, for an
    inconsistent outcome regression; F18 holds it.

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
Benkeser, Cai and van der Laan (2020). Their Theorem 1 states the adaptive-propensity curve
for one treatment-specific mean under six stated regularity conditions, and the Remark and
Appendix D outline the ATE and cross-validated constructions. The per-arm design applies the
theorem to each arm and stacks the curves; the shared design is not the paper's construction.
Neither design reports inference, because condition (v) of the theorem fails for an
estimated outcome regression. See ``docs/roadmap.md F18`` for the selector path and
``docs/roadmap.md F19`` for both outcome-adaptive designs. ``n_bootstrap=`` reruns the
adaptive construction, but no reviewed theorem validates that bootstrap for a selector path or
for either outcome-adaptive design, so every C-TMLE fit prints the bootstrap as a diagnostic.

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
- Ju, Wyss, Franklin, Schneeweiss, Häggström & van der Laan (2019), *Collaborative-controlled
  LASSO for constructing propensity score-based estimators in high-dimensional data*.

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
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from .._inference_status import InferenceStatus, precedent_status
from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..data.treatment_covariates import pool_line
from ..exceptions import CapabilityError, Refusal, RefusalReason
from ..fluctuation.iterative import InitialFit, apply_logistic, check_matching_arms
from ..fluctuation.submodel import Submodel, restrict, weighted_form
from ..inference.cluster import influence_variance
from ..inference.delta import log_odds_ratio_influence, log_ratio_influence, wald_ci
from ..inference.influence import counterfactual_means
from ..interventions.learned import learned_rule_configuration_refusal
from ..learners._fitting import Task, predict_mean, predict_probabilities
from ..learners._l1_logistic import (
    check_penalty_grid,
    cv_penalty,
    default_penalty_grid,
    nearest_neighbour,
    penalty_path,
)
from ..learners.crossfit import _POST_DRAW_REMEDY, Folds, check_integrity, make_folds
from ..learners.super_learner import resolve_learner
from ..targets.base import parameter_name
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from ..utils.parallel import map_parallel
from ..utils.text import format_table
from ._ctmle_derivative import (
    DERIVATIVE_ONE_STEP_REFUSAL,
    DERIVATIVE_TARGET_WEIGHTS_REFUSAL,
    fitted_update,
)
from ._nuisance import (
    _SIMPLEX_TOLERANCE,
    ArmQuery,
    NuisanceEstimates,
    Propensity,
    _arm_codes_by_row,
    _fit_arm_models,
    _fit_outcome_by_arm,
    _predict_by_arm,
    cross_fit_predictions,
    fit_on_rows,
)
from .base import MEAN_GROUP_ESTIMANDS, TMLEConfig, resolve_estimands
from .targeting import build_submodel, solve_submodel
from .tmle import TMLE

if TYPE_CHECKING:
    from .base import TMLEResult

__all__ = [
    "CTMLE",
    "CTMLE_SELECTOR_STRATEGIES",
    "OAT_PER_ARM_INFERENTIAL",
    "CTMLEDerivativeFit",
    "CTMLELoss",
    "CTMLEOatDesign",
    "CTMLEOutcomeAdaptiveFit",
    "CTMLEPreorder",
    "CTMLESelection",
    "CTMLEStrategy",
    "LogisticPlugin",
    "PropensityArrays",
    "declares_no_selection",
    "is_per_arm_oat",
    "is_selector_strategy",
    "logistic_plugin",
    "per_arm_design_admits",
]

CTMLEStrategy = Literal["greedy", "ordered", "discrete", "lasso", "learners", "derivative", "oat"]
CTMLELoss = Literal["auto", "loglik", "squared"]
CTMLEPreorder = Literal["logistic", "partial_correlation"]
CTMLEOatDesign = Literal["per_arm", "shared"]

#: Whether an admitted per-arm outcome-adaptive fit publishes inference. ``False``: the
#: declared revert of the registered study ``ctmle-oat-per-arm``. Its calibration, primary
#: coverage, estimated-design and band cells measured standard errors 8 to 24 percent too
#: small, flat in ``n``, while the oracle-design cells were calibrated. The cause is condition
#: (v) of Benkeser, Cai and van der Laan (2020), Theorem 1 (preprint Appendix F, the remainder
#: ``R24``), which fails for an estimated outcome regression, so the reported curve omits a
#: first-order generated-design term. Every per-arm fit takes ``"generated_design_plugin"``.
#: :func:`per_arm_design_admits` reads it; F19 in ``docs/roadmap.md`` holds the term.
OAT_PER_ARM_INFERENTIAL: Final[bool] = False

#: The strategies that build a candidate path and cut it at a data-chosen stopping index.
#: ``"oat"`` is not one: it fits its mechanism on the arm-specific outcome predictions and
#: has no path, no selector loss and no stopping index (see the module docstring).
#: F19 owns the inference of its shared design; F18 owns these five.
CTMLE_SELECTOR_STRATEGIES: frozenset[str] = frozenset(
    {"greedy", "ordered", "discrete", "lasso", "learners"}
)

#: The strategies that build an indexed candidate sequence (van der Laan and Gruber 2010,
#: Section 2.4) rather than a sequence of covariate sets.
_INDEXED_STRATEGIES: frozenset[str] = frozenset({"lasso", "learners"})


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
        ``True`` for ``"greedy"``, ``"ordered"``, ``"discrete"``, ``"lasso"`` and
        ``"learners"``. ``False`` for
        ``"oat"``, for ``None`` and for any unrecognised value, so a caller that
        broadens the Literal does not silently acquire a refusal it never declared.
    """
    return strategy in CTMLE_SELECTOR_STRATEGIES


def declares_no_selection(
    strategy: str | None,
    candidates: Sequence[Sequence[str]] | None,
    treatment_candidates: Sequence[Any] | None,
    covariate_names: Sequence[str],
) -> bool:
    """Whether a selector fit declares a candidate list with nothing to select.

    Two declarations qualify. A ``"discrete"`` fit whose one candidate is every covariate
    fits its treatment mechanism on the full adjustment set. A ``"learners"`` fit with an
    empty ``treatment_candidates=`` list has one candidate, ``treatment_learner`` on every
    covariate. Each fit selects nothing, and the estimator is the ordinary TMLE. The key
    reads the declaration and the prepared covariate names only, and never the fitted path,
    so a caller can know the answer before the fit.

    Parameters
    ----------
    strategy : str or None
        The collaborative strategy.
    candidates : sequence of sequence of str or None
        The declared ``candidates=`` list.
    treatment_candidates : sequence or None
        The declared ``treatment_candidates=`` list.
    covariate_names : sequence of str
        The covariates the treatment mechanism reads,
        :attr:`~cleverly.data.CausalData.mechanism_covariate_names`: the prepared covariate
        names, or the subset a design declares with ``treatment_adjustment=``.

    Returns
    -------
    bool
        ``True`` for a ``"discrete"`` fit whose list has exactly one entry that holds each
        covariate name exactly once, in any order, and for a ``"learners"`` fit whose list
        is empty. A repeated candidate counts as two candidates, and a repeated name is not
        the full set.
    """
    if strategy == "learners":
        return treatment_candidates is not None and len(treatment_candidates) == 0
    if strategy != "discrete" or candidates is None or len(candidates) != 1:
        return False
    return len(covariate_names) > 0 and sorted(candidates[0]) == sorted(covariate_names)


def _refuse_names_outside_pool(data: CausalData, argument: str, names: Sequence[str]) -> None:
    """Refuse a candidate or an ordering name that lies outside the treatment-mechanism covariates.

    The default design reads every covariate, so nothing lies outside it and this refuses
    nothing.

    Parameters
    ----------
    data : CausalData
        The prepared data.
    argument : str
        The estimator argument that holds the names, for the message.
    names : sequence of str
        The names the argument holds.

    Raises
    ------
    ValueError
        If a name is a covariate of the design that the treatment mechanism does not read.
    """
    if data.treatment_covariate_names is None:
        return
    pool = set(data.mechanism_covariate_names)
    outside = list(dict.fromkeys(name for name in names if name not in pool))
    if outside:
        raise ValueError(f"{argument} names {outside} outside treatment_adjustment=")


@dataclass(frozen=True, eq=False)
class PropensityArrays:
    """A declared candidate mechanism, given as predictions: the ``ctmleGeneral`` pair.

    R ``ctmle``'s ``ctmleGeneral`` takes each candidate as two columns:
    ``gn_candidates``, fitted on the whole sample, and ``gn_candidates_cv``, fitted out of
    fold on the selection split. A ``strategy="learners"`` fit takes the pair in the same
    roles. The full-sample path reads :attr:`full`. Inside each selection fold, the training
    rows and the validation rows both read :attr:`cv`, as ``ctmleGeneral`` does.

    The pair needs ``selection_split=``, the fold labels :attr:`cv` was fitted under. The
    package cannot check that :attr:`cv` respects that split, so the caller must. Each
    training-row prediction in :attr:`cv` comes from a fit that used the validation fold's
    treatment and covariates, though never its outcome, as in R. That is a declared
    exception to the selector rule that a selection-validation row reaches no nuisance fit
    that scores it.

    Parameters
    ----------
    full : array-like
        ``n`` rows of probabilities over the arms, in arm order, fitted on every row. Each
        row sums to 1 within ``1e-9``, the tolerance of every
        propensity on the simplex. At two arms a vector of ``P(A = arm 1)`` is accepted.
    cv : array-like
        The same shape, each row predicted by a fit that held out that row's selection
        fold.
    """

    full: Any
    cv: Any

    def resolved(
        self, index: int, n: int, arms: Sequence[float], labels: Sequence[Any]
    ) -> tuple[FloatArray, FloatArray]:
        """Both arrays as ``(n, K)`` probabilities, or ``ValueError`` on a wrong shape.

        Parameters
        ----------
        index : int
            The entry's position in ``treatment_candidates=``, for the message.
        n : int
            The number of analysis rows.
        arms : sequence of float
            The arm codes, in arm order.
        labels : sequence
            The arm labels, in arm order, for the message.

        Returns
        -------
        tuple of ndarray
            ``(full, cv)``, each of shape ``(n, K)``.
        """
        k = len(arms)
        out = []
        shapes = []
        for value in (self.full, self.cv):
            array = np.asarray(value, dtype=float)
            shapes.append(array.shape)
            if k == 2 and array.ndim == 1 and array.shape[0] == n:
                array = np.column_stack([1.0 - array, array])
            out.append(array)
        valid = all(
            array.shape == (n, k)
            and bool(np.all(np.isfinite(array)))
            and bool(np.all((array >= 0.0) & (array <= 1.0)))
            for array in out
        )
        if not valid:
            raise ValueError(
                f"PropensityArrays entry {index}: full and cv must each hold {n} rows of "
                f"probabilities over the {k} arms in arm order {list(labels)}; a vector of "
                f"P(A = {labels[1]}) is accepted at two arms; got shapes {shapes[0]} and "
                f"{shapes[1]}"
            )
        for name, array in zip(("full", "cv"), out, strict=True):
            gap = float(np.max(np.abs(np.sum(array, axis=1) - 1.0)))
            if gap > _SIMPLEX_TOLERANCE:
                raise ValueError(
                    f"PropensityArrays entry {index}: each row of {name} must sum to 1 over "
                    f"the {k} arms, and the largest gap from 1 is {gap:.3g}, above "
                    f"{_SIMPLEX_TOLERANCE:g}. A row that does not sum to 1 is not a "
                    "probability distribution, so the clever covariate would weight the arms "
                    f"wrongly. Pass predict_proba output in arm order {list(labels)}, or "
                    "divide each row by its sum"
                )
        return out[0], out[1]


def is_per_arm_oat(estimator: Any) -> bool:
    """Whether an estimator is the per-arm outcome-adaptive :class:`CTMLE`.

    One predicate for every reader that admits the per-arm design where it refuses the
    other collaborative strategies: :func:`per_arm_design_admits`, the missing-treatment
    refusal and the missing-outcome contracts of :class:`~cleverly.estimators.TMLE`.

    Parameters
    ----------
    estimator : Any
        Any estimator. Only ``strategy`` and ``oat_design`` are read.

    Returns
    -------
    bool
        ``True`` when the strategy is ``"oat"`` and the design is ``"per_arm"``.
    """
    return (
        getattr(estimator, "strategy", None) == "oat"
        and getattr(estimator, "oat_design", None) == "per_arm"
    )


def per_arm_design_admits(estimator: Any, data: CausalData) -> bool:
    """Whether an outcome-adaptive fit is inside the per-arm result and reports inference.

    The per-arm design regresses ``1{A = a}`` on the one column ``Qbar_n(a, W)`` for each arm.
    Benkeser, Cai and van der Laan (2020), Theorem 1, give the influence curve of that
    construction for one treatment-specific mean. An indicator reduction per arm and a
    fixed-dimension stack give the joint curve of every arm. That theorem needs its
    condition (v), which fails for an estimated outcome regression, so
    :data:`OAT_PER_ARM_INFERENTIAL` is ``False`` and the key admits no fit.
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
        :data:`OAT_PER_ARM_INFERENTIAL` is ``True``.  The per-arm design with strata or
        missing data is excluded even with the flag set, because no study measured its
        curve.
    """
    return bool(
        OAT_PER_ARM_INFERENTIAL
        and is_per_arm_oat(estimator)
        and not data.has_missing_outcome
        and not data.has_missing_treatment
        and not data.has_strata
    )


#: Floor applied to targeted predictions before taking a logarithm in the loss.
_LOSS_EPS = 1e-12

#: The tie tolerance of the declared-order check of ``strategy="learners"``: 1%.  A candidate
#: whose treatment negative log likelihood exceeds the previous candidate's by at most this
#: fraction of that loss is a tie.  The reason: a nested candidate with an uninformative extra
#: covariate can rise by sampling noise under the propensity bounds.  On the instrument law of
#: ``tests/studies/ctmle_sequence_properties.py`` the nested terminal rises by about 0.2%, and
#: the unordered ``W1`` candidate of the study's first run by about 19%.
_ORDER_TIE_TOLERANCE: Final[float] = 0.01

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
_DISCRETE_TREATMENT_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "CTMLE strategies require a discrete treatment. strategy='oat' fits its mechanism on "
    "one Qbar prediction per arm, and a continuous dose has no finite arm vector.",
    alternative="Fit TMLE on the continuous dose.",
    roadmap="F5",
)

#: Why CTMLE refuses ``policies=``.  Checked before the continuous-dose refusal, because a
#: shift declares its treatment a dose even when the column holds two arms, and the dose
#: refusal would then give the wrong reason on a binary treatment.
_POLICIES_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "CTMLE and policies= are not combined. A modified treatment policy's mean averages "
    "Qbar at the dose the policy assigns over the treatment law, so it reads g, and each "
    "candidate g of the collaborative search defines a different parameter, as under "
    "incremental=. docs/roadmap.md F5 tracks this stop.",
    alternative="Use a plain TMLE, which fits policies=.",
    roadmap="F5",
)

#: Why CTMLE refuses ``interventions=``.  A named refusal rather than the ATT and ATC or
#: estimand-axis message that a regime fit met before it.
_REGIME_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "CTMLE and interventions= are not combined. A regime mean reads Qbar and the covariate "
    "law only, so the collaborative template carries its point estimate through the "
    "indicator 1{A = d(W)}, but no selection criterion for a regime target is implemented. "
    "docs/roadmap.md X51 tracks the point estimate.",
    alternative="Use a plain TMLE, which fits interventions= with an interval.",
    roadmap="X51",
)

#: Why CTMLE refuses ``msm=`` over discrete arms, for the reason of :data:`_REGIME_REFUSAL`.
_MSM_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "CTMLE and msm= are not combined. A working model over discrete arms is a fixed "
    "full-rank projection of the arm means, so the collaborative template carries its point "
    "estimate, but no selection criterion for a vector projection is implemented. "
    "docs/roadmap.md X51 tracks the point estimate.",
    alternative="Use a plain TMLE, which fits msm=.",
    roadmap="X51",
)

#: The selection split's refusal remedy in the words the nuisance fold loop closes with.
_SELECTION_SPLIT_BACKSTOP = _POST_DRAW_REMEDY.format(remedy=_SELECTION_SPLIT_REMEDY)


def _unordered_candidates(treatment_risk: Sequence[float] | FloatArray) -> tuple[int, ...]:
    """The indices whose treatment loss rises above the previous candidate's by more than a tie.

    ``ctmleGeneral`` requires a declared sequence ordered from the least to the most adaptive
    candidate, with a decreasing treatment loss.  A rise of at most 1% of the previous loss
    (``_ORDER_TIE_TOLERANCE``) counts as a tie, because a nested candidate with an
    uninformative covariate can rise by sampling noise under the propensity bounds.

    Parameters
    ----------
    treatment_risk : sequence of float
        The treatment negative log likelihood of each candidate, in path order.

    Returns
    -------
    tuple of int
        Each index ``k`` whose loss exceeds the loss at ``k - 1`` by more than the tie.
    """
    risk = np.asarray(treatment_risk, dtype=float)
    tolerance = _ORDER_TIE_TOLERANCE * np.abs(risk[:-1])
    return tuple(int(k) + 1 for k in np.flatnonzero(np.diff(risk) > tolerance))


def _warn_on_unordered_candidates(path: Sequence[_Candidate]) -> None:
    """Warn when the declared candidates of ``strategy="learners"`` break the declared order.

    The construction tries the candidates in their declared order, so a candidate that fits
    the treatment worse than its predecessor can stop the search early.  The fit proceeds,
    because a rise can also be a tie of sampling noise.
    """
    risk = [candidate.treatment_risk for candidate in path]
    offending = _unordered_candidates(risk)
    if not offending:
        return
    named = ", ".join(f"{index} ({path[index].label})" for index in offending)
    warnings.warn(
        "strategy='learners' needs treatment_candidates= ordered from the least to the most "
        "adaptive, with a decreasing treatment loss. The treatment negative log likelihood "
        f"rises by more than 1% at candidate {named}. The fit proceeds, but the search can stop "
        "before a candidate that carries a confounder. Declare nested candidates, each one "
        "adding covariates to the one before it.",
        UserWarning,
        stacklevel=2,
    )


@dataclass(frozen=True)
class CTMLESelection:
    """The candidate sequence a C-TMLE fit searched, and where it was cut.

    Parameters
    ----------
    strategy : str
        The selector strategy.
    preorder : str or None
        The preordering rule of ``strategy="ordered"``, ``"custom"`` for a declared
        ordering, and ``None`` for the other strategies.
    estimand : str
        The estimand the selection was made for.
    target_names : tuple of str
        The jointly scored components of :attr:`estimand`.
    loss : str
        ``"loglik"`` or ``"squared"``.
    penalized : bool
        Whether the selection criterion adds the variance/bias penalty.
    path : tuple of tuple of str
        The covariates of each candidate's treatment mechanism, in path order. For a
        covariate-set strategy this is the candidate's covariate set. For ``"lasso"`` it
        holds the covariates with a nonzero coefficient in at least one fit that produced
        the candidate on the analysis sample. For ``"learners"`` it holds every covariate.
        The outcome model always sees every covariate.
    n_steps : tuple of int
        The number of fluctuation steps the search had applied when it reached each
        candidate. A value above one means the search had moved its initial fit.
    train_risk : ndarray
        The selection criterion for each candidate, evaluated in sample. A greedy or
        ordered search retries from one further targeted fit when no addition improves the
        criterion, then accepts the best addition. The recorded penalized risk may
        therefore increase.
    train_loss : ndarray
        The unpenalized loss part of :attr:`train_risk`.
    penalty : ndarray
        The penalty part of :attr:`train_risk`, zero when :attr:`penalized` is ``False``.
    treatment_risk : ndarray
        The weighted treatment negative log likelihood of each candidate.
    cv_risk : ndarray
        The same criterion, cross-validated. This is the quantity minimised, and unlike
        :attr:`train_risk` it is free to turn back up, which stops the search from taking
        every candidate.
    selected : int
        The index into :attr:`path` that the cross-validated risk chose.
    covariates : tuple of str
        The covariates the search drew from, in data order: every covariate, or the subset a
        design declares with ``treatment_adjustment=``.
    folds : Folds
        The realized selection split :attr:`cv_risk` was scored over. This is the partition
        the fit used, so a paired study can hand another implementation the same fold
        assignment. Its :attr:`~cleverly.learners.Folds.origin` records the generator, the
        scheme, the requested fold count and the seed of a drawn split. A split declared
        with ``selection_split=`` has no origin (``None``).
    labels : tuple of str
        One label per candidate: ``"W1, W2"`` style covariate lists, ``"lambda=0.0123"``
        for a lasso candidate, a learner's class name with its position, or
        ``"arrays[1]"`` for a :class:`PropensityArrays` entry.
    final : int or None
        The candidate the ``"lasso"`` final step reported, or ``None`` when no final step
        ran.
    restricted_pool : tuple of str or None
        The subset of covariates the design declares with ``treatment_adjustment=``, or
        ``None`` for a design that declares none. :meth:`summary` names it.
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
    labels: tuple[str, ...] = ()
    final: int | None = None
    restricted_pool: tuple[str, ...] | None = None

    @property
    def reported(self) -> int:
        """The candidate whose mechanism the estimate uses: :attr:`final`, else :attr:`selected`."""
        return self.selected if self.final is None else self.final

    @property
    def selected_covariates(self) -> tuple[str, ...]:
        """The covariates of the reported treatment model."""
        return self.path[self.reported]

    @property
    def dropped(self) -> tuple[str, ...]:
        """Covariates the reported treatment model left out, in their original order."""
        chosen = set(self.selected_covariates)
        return tuple(name for name in self.covariates if name not in chosen)

    def _label(self, index: int) -> str:
        if self.labels:
            return self.labels[index]
        names = self.path[index]
        return ", ".join(names) if names else "(intercept)"

    def to_frame(self, data: CausalData | None = None) -> Any:
        """One row per candidate: its covariate set, loss and cross-validated risk.

        Parameters
        ----------
        data : CausalData or None
            When given, the rows come back in that data's frame type.

        Returns
        -------
        dict or frame
            The columns ``candidate``, ``label``, ``covariates``, ``n_covariates``,
            ``n_steps``, ``train_risk``, ``train_loss``, ``penalty``, ``treatment_risk``,
            ``cv_risk``, ``selected`` and ``final``.
        """
        payload: dict[str, Any] = {
            "candidate": list(range(len(self.path))),
            "label": [self._label(index) for index in range(len(self.path))],
            "covariates": [", ".join(names) if names else "(intercept)" for names in self.path],
            "n_covariates": [len(names) for names in self.path],
            "n_steps": list(self.n_steps),
            "train_risk": self.train_risk.tolist(),
            "train_loss": self.train_loss.tolist(),
            "penalty": self.penalty.tolist(),
            "treatment_risk": self.treatment_risk.tolist(),
            "cv_risk": self.cv_risk.tolist(),
            "selected": [index == self.selected for index in range(len(self.path))],
            "final": [index == self.final for index in range(len(self.path))],
        }
        if data is None:
            return payload
        return data.frame_like(payload)

    @property
    def treatment_risk_decreasing(self) -> bool:
        """Whether :attr:`treatment_risk` does not increase along the path.

        ``ctmleGeneral`` requires a declared sequence ordered from the least to the most
        adaptive candidate, with decreasing empirical loss. The fit does not refuse a
        sequence that breaks the order. :meth:`summary` reports it.
        """
        return bool(np.all(np.diff(self.treatment_risk) <= 0.0))

    @property
    def treatment_risk_increases(self) -> tuple[int, ...]:
        """The candidates whose treatment loss rises above the previous candidate's.

        A rise of at most 1% of the previous candidate's loss is a tie and is not listed.
        A fit of ``strategy="learners"`` warns when this tuple is not empty.

        Returns
        -------
        tuple of int
            The indices into :attr:`path`, in path order.
        """
        return _unordered_candidates(self.treatment_risk)

    def summary(self) -> str:
        """A printable report of the selection.

        Returns
        -------
        str
            The candidate table, the criterion and the reported mechanism.
        """
        indexed = self.strategy in _INDEXED_STRATEGIES
        rows = []
        for index, names in enumerate(self.path):
            marker = "<--" if index == self.selected else ""
            if self.final is not None and index == self.final:
                marker = "<-- final"
            rows.append(
                [
                    str(index),
                    self._label(index) if indexed else (", ".join(names) or "(intercept)"),
                    str(self.n_steps[index]),
                    f"{self.train_risk[index]:.6g}",
                    f"{self.cv_risk[index]:.6g}",
                    marker,
                ]
            )
        table = format_table(
            ["k", "candidate g" if indexed else "covariates in g", "steps", "risk", "cv risk", ""],
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
        if self.restricted_pool is not None:
            header.insert(-1, pool_line(self.restricted_pool))
        chosen = self.selected_covariates
        footer = [""]
        if indexed:
            footer.append(f"selected g: {self._label(self.reported)}")
            footer.append(
                "covariates in selected g: " + (", ".join(chosen) if chosen else "(none)")
            )
            if self.strategy == "learners":
                footer.append(
                    "treatment risk decreases along the declared order: "
                    + ("yes" if self.treatment_risk_decreasing else "no")
                )
            return "\n".join([*header, table, *footer])
        footer.append("selected g: " + (", ".join(chosen) if chosen else "(intercept only)"))
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
        sentence = (
            f"C-TMLE {self.strategy} selected candidate {self.selected + 1} of "
            f"{len(self.path)} for {self.estimand}"
        )
        if self.final is not None:
            sentence += f"; the final step used candidate {self.final + 1}"
        return sentence

    @property
    def treatment_features(self) -> tuple[str, ...]:
        """The covariates entering the reported treatment model."""
        return self.selected_covariates

    @property
    def treatment_risk_selected(self) -> float:
        """Treatment negative log likelihood at the reported path position."""
        return float(self.treatment_risk[self.reported])


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
    strata : tuple of str
        The labels of the baseline strata each per-arm factor was fitted inside. Empty
        without baseline strata and under the shared design.
    factors : tuple of str
        The factors of the per-arm design, for a generic arm ``a``: the treatment factor,
        then the outcome observation factor with missing outcomes, then the treatment
        observation factor with a missing treatment. Empty under the shared design.
    risk_by_factor : dict of str to float or None
        Each factor's binary negative log likelihood, summed over the arms and computed on
        the rows the factor is fitted on, keyed by the entry of ``factors``. ``None`` under
        the shared design.
    """

    strategy: CTMLEStrategy
    treatment_features: tuple[str, ...]
    treatment_risk: float
    design: CTMLEOatDesign = "shared"
    treatment_risk_by_arm: dict[str, float] | None = None
    strata: tuple[str, ...] = ()
    factors: tuple[str, ...] = ()
    risk_by_factor: dict[str, float] | None = None

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
            sentence = (
                f"C-TMLE outcome-adaptive fit used one Qbar feature per arm in {count} "
                "binary mechanisms"
            )
            if len(self.factors) > 1:
                sentence += f", with {len(self.factors)} factors per arm"
            if self.strata:
                sentence += f", inside each of {len(self.strata)} baseline strata"
            return sentence
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
        if self.design == "per_arm" and (len(self.factors) > 1 or self.strata):
            inside = ", inside each baseline stratum" if self.strata else ""
            model = f"per_arm; each factor below ~ Qbar(a, W){inside}, one set per arm"
        elif self.design == "per_arm":
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
        if (len(self.factors) > 1 or self.strata) and self.risk_by_factor is not None:
            lines.append("factors, negative log likelihood summed over arms:")
            lines.extend(
                f"  {factor}: {self.risk_by_factor[factor]:.6g}" for factor in self.factors
            )
        if self.strata:
            lines.append(f"fitted inside each baseline stratum: {'; '.join(self.strata)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CTMLEDerivativeFit:
    """Diagnostics for the C-TMLE0 fit of ``strategy="derivative"``.

    Parameters
    ----------
    strategy : str
        Always ``"derivative"``.
    source : {"lasso", "learners", "arrays"}
        Where the mechanism and its partner came from: the L1-penalized logistic
        mechanism, the two learners of ``treatment_candidates=``, or two declared arrays.
        A pair of one learner and one array reads ``"learners"``.
    penalties : tuple of tuple of float
        One ``(mechanism, partner)`` penalty pair for each training set, in sample one pair.
        Empty off the lasso source.
    cv_risk : tuple of ndarray
        The cross-validated weighted deviance over the grid for each training set. Empty
        unless the lasso source chose its penalty by cross-validation.
    epsilon : ndarray
        The ``2K`` fluctuation coefficients on R's scale, in :attr:`names` order.
    names : tuple of str
        ``h0``, ``h1``, ..., then ``dh0``, ``dh1``, ...
    derivative_score_before : ndarray
        Per arm, the derivative score at the initial outcome regression, on the normalized
        scale.
    derivative_score_after : ndarray
        The same score at the targeted outcome regression.
    treatment_risk : float
        The mean cross-validated deviance at the chosen penalty when the lasso source chose
        it, otherwise the weighted negative log likelihood of the mechanism's predictions.
    """

    strategy: CTMLEStrategy
    source: Literal["lasso", "learners", "arrays"]
    penalties: tuple[tuple[float, float], ...]
    cv_risk: tuple[FloatArray, ...]
    epsilon: FloatArray
    names: tuple[str, ...]
    derivative_score_before: FloatArray
    derivative_score_after: FloatArray
    treatment_risk: float

    @property
    def treatment_risk_selected(self) -> float:
        """The treatment risk, under the shared C-TMLE diagnostic API."""
        return self.treatment_risk

    @property
    def treatment_features(self) -> tuple[str, ...]:
        """No covariate list: the mechanism and its partner read every covariate."""
        return ()

    def describe(self) -> str:
        """One sentence naming this fit and where its two mechanisms came from.

        Returns
        -------
        str
            A complete sentence, ready to embed in a report line.
        """
        if self.source == "lasso" and len(set(self.penalties)) == 1:
            mechanism, partner = self.penalties[0]
            sets = (
                f" in each of {len(self.penalties)} training sets"
                if len(self.penalties) > 1
                else ""
            )
            return (
                f"C-TMLE derivative fit used the lasso at penalty {mechanism:.3g} with "
                f"partner {partner:.3g}{sets}"
            )
        if self.source == "lasso":
            chosen = [pair[0] for pair in self.penalties]
            return (
                f"C-TMLE derivative fit used the lasso at penalties {min(chosen):.3g} to "
                f"{max(chosen):.3g} over {len(self.penalties)} training sets"
            )
        if self.source == "learners":
            return "C-TMLE derivative fit used two declared learners as mechanism and partner"
        return "C-TMLE derivative fit used two declared arrays as mechanism and partner"

    def summary(self) -> str:
        """A printable report of the derivative fit.

        Returns
        -------
        str
            The source, the penalties, the coefficients and the derivative scores.
        """
        lines = [
            "Collaborative TMLE derivative fit (C-TMLE0)",
            "=" * 43,
            f"strategy = derivative; source = {self.source}",
            self.describe(),
            "",
            format_table(
                ["column", "epsilon"],
                [
                    [name, f"{value:.6g}"]
                    for name, value in zip(self.names, self.epsilon, strict=True)
                ],
            ),
            "",
            "derivative score before / after targeting (normalized): "
            + "; ".join(
                f"{before:.3g} / {after:.3g}"
                for before, after in zip(
                    self.derivative_score_before, self.derivative_score_after, strict=True
                )
            ),
            f"treatment risk: {self.treatment_risk:.6g}",
        ]
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
    label: str = ""


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
    roadmap holds.  The exceptions are a ``"discrete"`` fit whose one declared candidate is
    the full adjustment set and a ``"learners"`` fit with an empty declared list: each
    selects nothing, equals the ordinary TMLE, and reports its interval.
    ``strategy="oat"`` withholds the interval on either design, for the reason F19 in the
    roadmap holds: condition (v) of Benkeser, Cai and van der Laan (2020), Theorem 1,
    fails for an estimated outcome regression.  ``strategy="derivative"`` selects nothing
    and reports the ordinary TMLE interval.  See the module docstring.

    Parameters
    ----------
    strategy : {"greedy", "ordered", "discrete", "lasso", "learners", "derivative", "oat"}, \
default="greedy"
        ``"greedy"`` (default), ``"ordered"``, ``"discrete"``, ``"lasso"``, ``"learners"``,
        ``"derivative"`` or ``"oat"``.  ``"derivative"`` is C-TMLE0: one TMLE update with a
        derivative column for each arm (see ``lambdas`` and ``treatment_candidates``).
        The selector strategies fit one shared categorical propensity path and jointly
        score all components of ``ctmle_estimand``; ``"oat"`` fits treatment on the arm-specific
        outcome predictions without a candidate path (see ``oat_design``).
        ``"oat"`` excludes ``W`` from ``g`` entirely and so gives up
        consistency-when-only-``g``-is-right; see the module docstring.
    ordering : sequence of str or None, default=None
        Explicit covariate order for ``strategy="ordered"``.  When omitted, ``preorder``
        determines the published data-adaptive ordering.  A refit that adds a covariate,
        such as ``random_common_cause``, places it after the declared ordering.
    preorder : {"logistic", "partial_correlation"} or None, default=None
        ``"logistic"`` (default) or ``"partial_correlation"`` for Algorithms 2 and 3
        of Ju et al. (2019).  Ignored when an explicit ``ordering=`` is supplied.
    candidates : sequence of sequence of str or None, default=None
        Explicit candidate covariate sets for ``strategy="discrete"``.
    selection_folds : int, default=5
        Folds used to cross-validate the candidate sequence.  Separate from
        ``n_folds``, which cross-fits the nuisance models.  With ``selection_split``, it
        must equal the split's fold count.
    selection_inner_folds : int, default=2
        Inner folds used to make selection-training predictions out of fold. Two is the
        default because every selection fold also needs a full-training fit for its
        validation rows; increasing it improves the inner cross-fit at a linear fit-cost.
    loss : {"auto", "loglik", "squared"}, default="auto"
        ``"auto"`` (default) uses the squared-error loss for a continuous outcome and
        the quasi-binomial log-likelihood for a binary one, as R's ``ctmle`` does.
    penalty : bool, default=True
        Add the variance/bias penalty to the selection loss.  Leave it on unless you
        specifically want the unpenalized log-likelihood selector.
    ctmle_estimand : str, default="ate"
        Which estimand the selection is *for*.  A collaborative selection is
        parameter-specific -- the loss involves that estimand's influence curve --
        so unlike a plain TMLE, one fit cannot serve every estimand equally. At ``K``
        arms, ``ate``, ``rr`` and ``or`` jointly optimize the ``K - 1`` contrasts against
        ``reference=`` and ``ey`` jointly optimizes all ``K`` means. Must be requested.
    oat_design : {"per_arm", "shared"} or None, default=None
        The treatment design of ``strategy="oat"``.  ``"per_arm"`` (the default under
        ``"oat"``) fits one binary mechanism per arm, ``P(A = a | Qbar(a, W))``, on that
        arm's own outcome prediction.  ``"shared"`` fits one categorical mechanism on the
        vector ``[Qbar(a, W): a in arms]``, as ``ctmle3::LF_oat`` does.  Neither design
        reports inference; see the module docstring.  ``None`` resolves to
        ``"per_arm"`` under ``"oat"``, and any other value raises under a selector strategy.
    lambdas : sequence of float or None, default=None
        The penalty grid of ``strategy="lasso"``: two or more finite, non-negative,
        strictly decreasing penalties, with 0 allowed only as the last value.  A user grid
        gets no automatic 0.  ``None`` uses ``glmnet``'s default grid, 100 log-spaced values
        from ``lambda_max`` down, then 0.  The grid reads treatment and covariates only, is fixed
        on the analysis sample, and is reused in every fold.  ``treatment_learner`` is not
        used by the lasso candidates; it stays the fallback of the missingness learner.
        Under ``strategy="derivative"`` it is the pair ``(mechanism, partner)`` of two
        different non-negative penalties, or ``None`` to cross-validate the mechanism penalty
        over the default grid and take its nearest grid neighbour as the partner.
    treatment_candidates : sequence or None, default=None
        The declared candidates of ``strategy="learners"``, from the least to the most
        adaptive.  An entry is an unfitted learner, fitted on every covariate on exactly the
        rows a covariate-set candidate uses, or a :class:`PropensityArrays` pair.
        ``treatment_learner`` on every covariate is appended as the last candidate.  An
        empty list selects nothing.  A scikit-learn ``FrozenEstimator`` passes as a learner
        and is not refit, because ``clone`` returns it unchanged.  Under
        ``strategy="derivative"`` it is the pair ``(mechanism, partner)``: two unfitted
        learners, or two arrays of full-sample treatment probabilities.
    selection_split : sequence of int or None, default=None
        Fold labels ``0`` to ``V - 1``, one for each row, used in place of the random
        selection split, on every selector strategy.  ``selection_folds`` must equal ``V``.
        It refuses ``repeats > 1`` and ``n_bootstrap > 0``.
    **kwargs : Any
        The keyword arguments of :class:`~cleverly.TMLE`: the learners, the fold policy,
        the bounds and the targeting controls.

    Notes
    -----
    ``att`` and ``atc`` are not supported: their clever covariate conditions on a
    random event, so the candidate sequence would have to be rebuilt per estimand
    and the "one selected ``g``" story breaks down.  Request them from a plain
    :class:`~cleverly.TMLE`.

    ``screen_treatment`` is redundant here and does not apply to the candidate
    propensity models, and ``strategy="derivative"`` refuses it.  It is a *static* marginal
    pre-screen for exactly the problem this class solves collaboratively, and running both
    would mean the covariate a correlation filter rejected never reaches the search that
    might have wanted it.
    """

    _assessment_method = "collaborative_tmle"

    def _inference_status(self, data: CausalData) -> InferenceStatus:
        """Name the status of each strategy, and admit the one fit that selects nothing.

        Keyed on the strategy. The selector paths take the status that F18 keys on, with two
        exceptions. A ``"discrete"`` fit that declares one candidate, equal to the full adjustment
        set, and a ``"learners"`` fit that declares an empty list select nothing and are the
        ordinary TMLE, so they take the status of :class:`~cleverly.TMLE`.
        :func:`declares_no_selection` is the key. It reads the declaration and the prepared
        covariate names, and never the fitted path.
        ``docs/technical-reference/collaborative-tmle.md`` states the contract.

        Every ``"oat"`` fit takes ``"generated_design_plugin"``, which F19 holds. The key
        :func:`per_arm_design_admits` would admit a per-arm fit on complete outcomes and
        treatment without baseline strata, but it also reads :data:`OAT_PER_ARM_INFERENTIAL`,
        which is ``False``: the registered study found the per-arm standard errors too small,
        because condition (v) of Benkeser, Cai and van der Laan (2020), Theorem 1, fails for
        an estimated outcome regression. Were the flag set, an admitted fit would take the
        :class:`~cleverly.TMLE` status, or ``"estimated_weight_plugin"`` on weights declared
        estimated, since the conditioning argument concerns the efficient curve. The status
        table of ``docs/technical-reference/inference.md`` states the shipped behaviour.

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
            :class:`~cleverly.TMLE` status for the admitted ``"discrete"`` and ``"learners"``
            fits and a per-arm ``"oat"`` fit that :func:`per_arm_design_admits` admits (none
            while :data:`OAT_PER_ARM_INFERENTIAL` is ``False``), ``"working_mechanism_plugin"``
            for every other selector fit, and
            ``"generated_design_plugin"`` for every other ``"oat"`` fit.
        """
        if declares_no_selection(
            self.strategy,
            self.candidates,
            self.treatment_candidates,
            data.mechanism_covariate_names,
        ):
            return super()._inference_status(data)
        if self.strategy == "derivative":
            # C-TMLE0 selects nothing collaboratively. It is an ordinary TMLE with a larger
            # fluctuation that still holds every clever covariate.
            return super()._inference_status(data)
        if is_selector_strategy(self.strategy):
            return "working_mechanism_plugin"
        if per_arm_design_admits(self, data):
            # The interval conditions on the weights only for the efficient curve. The per-arm
            # curve is Theorem 1's, so estimated weights withhold it, as a guarded DRTMLE does.
            return precedent_status(
                [
                    "estimated_weight_plugin"
                    if data.declares_estimated_weights
                    else "influence_curve",
                    super()._inference_status(data),
                ]
            )
        return "generated_design_plugin"

    def _bootstrap_inferential(self, data: CausalData) -> bool:
        """Whether the full-refit bootstrap interval is published as inference.

        ``False`` on every fit, for every strategy. On an ``"oat"`` fit, Theorem 1 of Benkeser,
        Cai and van der Laan (2020) gives the influence curve of a superefficient estimator,
        and it does not cover the bootstrap of that estimator. On a selector strategy, no
        reviewed theorem covers the bootstrap of a selected path (roadmap F18). The registered
        study ``full-refit-bootstrap-and-derived-contrasts`` measured the ordinary
        point-treatment :class:`~cleverly.TMLE` only. The percentile range prints as a
        diagnostic.

        Parameters
        ----------
        data : CausalData
            The prepared data. It is not read.

        Returns
        -------
        bool
            ``False``.
        """
        del data
        return False

    def _admits_natural_course_with_missing(self) -> bool:
        """Whether this fit may target the natural course with missing outcomes.

        The per-arm outcome-adaptive design does, for point estimates: its natural-course
        observation factor is ``P(Delta = 1 | A, Qbar(A, W))``, read at the realised arm.
        The selector strategies and the shared design have no such result.

        Returns
        -------
        bool
            ``True`` for ``strategy="oat"`` with ``oat_design="per_arm"``.
        """
        return is_per_arm_oat(self)

    def _design_factor_labels(self, data: CausalData) -> tuple[str, str] | None:
        """The per-arm design's observation factor labels, which condition on ``Qbar(a, W)``.

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Returns
        -------
        tuple of str or None
            The outcome and treatment observation factors of a per-arm outcome-adaptive
            fit, otherwise the :class:`~cleverly.TMLE` answer.
        """
        if not is_per_arm_oat(self):
            return super()._design_factor_labels(data)
        labels = {spec.name: spec.label for spec in _per_arm_factor_specs(data)}
        return (
            labels.get("outcome_observation", "P(Delta=1|A=a,Qbar(a,W))"),
            labels.get("treatment_observation", "P(Delta_A=1|Qbar(a,W))"),
        )

    def _on_arm_indexed_stacked_surface(self, data: CausalData, estimands: tuple[str, ...]) -> bool:
        """Whether the arm-indexed stacked MAR contract governs this collaborative fit.

        That contract protects the stacked CV-TMLE interval. A per-arm outcome-adaptive fit
        reports a point estimate only, so it leaves the contract and keeps the general
        cross-fitting rules and its own design preflight. The selector strategies and the
        shared design stay on the contract, which refuses them.

        Parameters
        ----------
        data : CausalData
            The prepared data.
        estimands : tuple of str
            The resolved estimands.

        Returns
        -------
        bool
            ``False`` for a per-arm outcome-adaptive fit, otherwise the
            :class:`~cleverly.TMLE` answer.
        """
        if is_per_arm_oat(self):
            return False
        return super()._on_arm_indexed_stacked_surface(data, estimands)

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
        lambdas: Sequence[float] | None = None,
        treatment_candidates: Sequence[Any] | None = None,
        selection_split: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("learned_rule") is not None:
            # Row 1 of the learned-rule refusal table comes before the fold-evaluation refusal
            # below: its remedy, cv_evaluation=False, would meet this refusal next.
            refusal = learned_rule_configuration_refusal(self)
            assert refusal is not None
            raise CapabilityError.from_refusal(refusal)
        if kwargs.get("cv_evaluation", False):
            # F18 keeps a fold-specific selection, and F19 the outcome-adaptive fold average.
            fold_item = "F18" if is_selector_strategy(strategy) else "F19"
            raise CapabilityError(
                "CTMLE does not support cv_evaluation=True: canonical CV-TMLE selection "
                "requires a separate fold-specific collaborative derivation.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Use cv_evaluation=False.",
                roadmap=fold_item,
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
        self.lambdas = lambdas
        self.treatment_candidates = treatment_candidates
        self.selection_split = selection_split
        self._validate_ctmle_settings()
        if self.strategy == "ordered" and self.ordering is None and self.preorder is None:
            self.preorder = "logistic"
        if self.strategy == "oat" and self.oat_design is None:
            self.oat_design = "per_arm"

    def _validate_ctmle_settings(self) -> None:
        if self.loss not in ("auto", "loglik", "squared"):
            raise ValueError(f"loss must be 'auto', 'loglik' or 'squared'; got {self.loss!r}")
        if self.strategy not in (
            "greedy",
            "ordered",
            "discrete",
            "lasso",
            "learners",
            "derivative",
            "oat",
        ):
            raise ValueError(
                "strategy must be 'greedy', 'ordered', 'discrete', 'lasso', 'learners', "
                f"'derivative' or 'oat'; got {self.strategy!r}"
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
        if self.strategy not in ("lasso", "derivative") and self.lambdas is not None:
            raise ValueError(
                f"lambdas= only applies to strategy='lasso' or 'derivative', not {self.strategy!r}"
            )
        if self.lambdas is not None:
            if self.strategy == "derivative":
                _check_penalty_pair(self.lambdas)
            else:
                grid = check_penalty_grid(self.lambdas)
                if grid.size < 2:
                    raise ValueError(
                        "strategy='lasso' needs lambdas= with at least two penalties; got "
                        f"{grid.tolist()!r}. One penalty gives one candidate, so the fit selects "
                        "nothing. Pass a strictly decreasing sequence of two or more penalties, "
                        "or lambdas=None for the default grid"
                    )
        if (
            self.strategy not in ("learners", "derivative")
            and self.treatment_candidates is not None
        ):
            raise ValueError(
                "treatment_candidates= only applies to strategy='learners' or 'derivative', "
                f"not {self.strategy!r}"
            )
        if (
            self.strategy == "derivative"
            and self.lambdas is not None
            and self.treatment_candidates is not None
        ):
            raise ValueError(
                "strategy='derivative' takes its two mechanisms from lambdas= or from "
                "treatment_candidates=, not from both"
            )
        if self.strategy == "learners" and self.treatment_candidates is None:
            raise ValueError(
                "strategy='learners' needs an explicit treatment_candidates= list, ordered "
                "from the least to the most adaptive mechanism; treatment_learner on every "
                "covariate is appended as the last candidate. Pass an empty list to fit "
                "treatment_learner alone"
            )
        self._validate_treatment_candidates()
        self._validate_selection_split()
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
            fold_item = "F18" if is_selector_strategy(self.strategy) else "F19"
            raise CapabilityError(
                "CTMLE implements the published pooled collaborative estimator only; "
                "targeting_scheme='fold' composes it with a different CV-TMLE estimator "
                "that has not been derived.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Use targeting_scheme='pooled'.",
                roadmap=fold_item,
            )

        if self.strategy == "derivative" and self.ctmle_estimand != "ate":
            raise ValueError(
                "ctmle_estimand= does not apply to strategy='derivative': the derivative "
                "update targets every treatment-specific mean together"
            )
        if self.strategy == "oat" and self.ctmle_estimand != "ate":
            raise ValueError(
                "ctmle_estimand= does not apply to strategy='oat': ctmle3's "
                "outcome-adaptive construction targets all treatment-specific means together"
            )
        overridden = [] if is_selector_strategy(self.strategy) else self._selector_only_overrides()
        if overridden and self.strategy == "derivative":
            raise ValueError(
                f"{', '.join(f'{name}=' for name in overridden)} configure selector "
                "strategies and do not apply to strategy='derivative', which fits one "
                "mechanism and its partner and selects nothing"
            )
        if overridden:
            raise ValueError(
                f"{', '.join(f'{name}=' for name in overridden)} configure selector "
                "strategies and do not apply to strategy='oat', which fits its mechanism on "
                "the outcome predictions and selects nothing"
            )
        if self.strategy == "derivative" and self.targeting == "one_step":
            raise CapabilityError(refusal=DERIVATIVE_ONE_STEP_REFUSAL)
        if self.strategy == "derivative" and self.target_weights:
            raise CapabilityError(refusal=DERIVATIVE_TARGET_WEIGHTS_REFUSAL)
        if self.strategy == "derivative" and self.screen_treatment:
            raise CapabilityError(
                "screen_treatment=True does not apply to strategy='derivative': the lasso "
                "pair, the declared learners and the arrays each fit the mechanism and its "
                "partner on every covariate, and a screen would apply to neither.",
                reason=RefusalReason.API_REDIRECTION,
                alternative=(
                    "Drop screen_treatment, or fit TMLE (TMLE, or TMLEMethod) with a screened "
                    "mechanism."
                ),
            )

    def _validate_treatment_candidates(self) -> None:
        """Refuse a lone array entry, and an arrays entry outside its declared regime."""
        if self.treatment_candidates is None:
            return
        if self.strategy == "derivative":
            self._validate_derivative_entries()
            return
        if isinstance(self.treatment_candidates, (str, bytes)) or not isinstance(
            self.treatment_candidates, Sequence
        ):
            raise ValueError(
                "treatment_candidates= must be a list of learners or PropensityArrays pairs"
            )
        has_arrays = False
        for index, entry in enumerate(self.treatment_candidates):
            if isinstance(entry, PropensityArrays):
                has_arrays = True
                continue
            if entry is None:
                raise ValueError(
                    f"treatment_candidates= entry {index} is None; pass a learner or a "
                    "PropensityArrays pair"
                )
            if _is_array_like(entry):
                raise CapabilityError(
                    f"treatment_candidates= entry {index} is a single propensity array. Pass "
                    "the ctmleGeneral pair instead: PropensityArrays(full=..., cv=...), with "
                    "the full-sample predictions and the out-of-fold predictions, and pass "
                    "the split those out-of-fold predictions used as selection_split=. A "
                    "single array fitted on every row would score a mechanism that saw the "
                    "validation rows.",
                    reason=RefusalReason.API_REDIRECTION,
                    alternative=(
                        "Pass PropensityArrays(full=..., cv=...) with selection_split=, or "
                        "pass the learner that produced the array."
                    ),
                )
            resolve_learner(entry, task="classification")
        if not has_arrays:
            return
        if self.selection_split is None:
            raise CapabilityError(
                "PropensityArrays candidates need selection_split=, the fold labels their "
                "cv= predictions were fitted under. selection_folds= draws a new split, which "
                "those predictions do not respect.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Pass the fold labels of the cv= predictions as selection_split=.",
            )
        if self.cross_fit:
            raise CapabilityError(
                "PropensityArrays candidates are fitted on the analysis sample, and a "
                "cross-fitted fit needs one fit for each outer fold.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative=(
                    "Fit with cross_fit=False, or pass the learner that produced the arrays."
                ),
            )

    def _validate_derivative_entries(self) -> None:
        """Refuse a ``strategy="derivative"`` pair that is not two usable entries."""
        entries = self.treatment_candidates
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise ValueError(
                "Under strategy='derivative', treatment_candidates= is the pair (mechanism, "
                "partner): exactly two entries, each an unfitted learner or a full-sample "
                f"propensity array. Got {type(entries).__name__}"
            )
        if len(entries) != 2:
            raise ValueError(
                "Under strategy='derivative', treatment_candidates= is the pair (mechanism, "
                "partner): exactly two entries, each an unfitted learner or a full-sample "
                f"propensity array. Got {len(entries)} entries"
            )
        has_arrays = False
        for index, entry in enumerate(entries):
            if isinstance(entry, PropensityArrays):
                raise CapabilityError(
                    f"treatment_candidates= entry {index} is a PropensityArrays pair. "
                    "strategy='derivative' has no selection split, so it reads full-sample "
                    "predictions only.",
                    reason=RefusalReason.INVALID_DECLARATION,
                    alternative=(
                        "Pass the full array itself, as R ctmleGeneral(ctmletype=3) takes g1W "
                        "and g1WPrev."
                    ),
                )
            if entry is None:
                raise ValueError(
                    f"treatment_candidates= entry {index} is None; pass a learner or a "
                    "full-sample propensity array"
                )
            if _is_array_like(entry):
                has_arrays = True
                continue
            resolve_learner(entry, task="classification")
        if has_arrays and self.cross_fit:
            raise CapabilityError(
                "A propensity array entry was fitted on the analysis sample, and a "
                "cross-fitted fit needs one fit for each outer fold.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative=(
                    "Fit with cross_fit=False, or pass the learner that produced the array."
                ),
            )
        if has_arrays and self.n_bootstrap:
            raise CapabilityError(
                "A propensity array entry holds one value for each analysis row, and a "
                "bootstrap replicate refits the mechanism on resampled rows.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Use n_bootstrap=0, or pass the learner that produced the array.",
            )

    def _validate_selection_split(self) -> None:
        """Refuse a declared selection split beside a setting that redraws or resamples."""
        if self.selection_split is None or not is_selector_strategy(self.strategy):
            return
        if self.repeats > 1:
            raise CapabilityError(
                "selection_split= fixes the selection folds, and repeats= redraws them for "
                "each repeat.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Use repeats=1 with a declared split.",
            )
        if self.n_bootstrap:
            raise CapabilityError(
                "selection_split= labels the analysis rows, and a bootstrap resample changes "
                "the rows.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Use n_bootstrap=0 with a declared split.",
            )
        declared = _declared_fold_count(self.selection_split)
        if declared is not None and declared != self.selection_folds:
            raise ValueError(
                f"selection_split= declares {declared} folds, and selection_folds= is "
                f"{self.selection_folds}. The two must agree, so that the fit records the "
                f"number of folds it used. Pass selection_folds={declared} with this split"
            )

    def _has_arrays(self) -> bool:
        return any(
            isinstance(entry, PropensityArrays) for entry in (self.treatment_candidates or ())
        )

    def _has_derivative_arrays(self) -> bool:
        return self.strategy == "derivative" and any(
            _is_array_like(entry) for entry in (self.treatment_candidates or ())
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
        names = ("selection_folds", "selection_inner_folds", "loss", "penalty")
        overridden = [name for name in names if getattr(self, name) != defaults[name].default]
        if self.selection_split is not None:
            overridden.append("selection_split")
        return overridden

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
        per_arm = is_per_arm_oat(self)
        # Every collaborative strategy replaces the ordinary treatment mechanism: the
        # selectors fit their candidate path and OAT fits A on the Qbar vector.  Fit only
        # the shared outcome/missingness nuisances here so g(W) is not paid for and thrown
        # away first.  The per-arm design fits its own treatment observation factor, and no
        # reader takes the full-W one, so that learner is skipped as well.
        base = self._fit_nuisances(
            data,
            folds,
            scaler,
            intermediate_value,
            seed=seed,
            fit_treatment=False,
            fit_treatment_observation=not per_arm,
        )
        if self.strategy == "derivative":
            return self._derivative_nuisances(data, base, config, intermediate_value, seed=seed)
        if not is_selector_strategy(self.strategy):
            return self._outcome_adaptive_nuisances(data, base, seed=seed)
        selector = _Selector(self, data, base, config.g_bounds, intermediate_value, seed=seed)

        path = selector.build_path(train=None, tag="full")
        cv_risk, selection_folds = selector.cross_validate(path)
        selected = int(np.argmin(cv_risk))
        chosen = path[selected]
        if self.strategy == "learners":
            _warn_on_unordered_candidates(path)
        final: int | None = None
        if self.strategy == "lasso":
            step = selector.final_step(path, selected)
            if step is not None:
                final, chosen = step

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
        indexed = self.strategy in _INDEXED_STRATEGIES
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
            covariates=data.mechanism_covariate_names,
            labels=tuple(candidate.label for candidate in path) if indexed else (),
            final=final,
            restricted_pool=(
                None if data.treatment_covariate_names is None else data.mechanism_covariate_names
            ),
        )
        return nuisance, {"ctmle": selection}

    def _repeat_draws(
        self, data: CausalData, estimands: tuple[str, ...]
    ) -> tuple[tuple[Folds, int], ...]:
        """Realize every outer draw, then check each one against the per-arm design.

        :meth:`~cleverly.TMLE._repeat_draws` checks what every draw owes the ordinary
        nuisances. A per-arm outcome-adaptive fit with baseline strata or missing data also
        checks each draw with :meth:`_preflight_per_arm_design` here. The check therefore
        runs on every draw before the first learner of any draw, and a later draw cannot
        refuse after an earlier draw has fitted its learners.

        Parameters
        ----------
        data : CausalData
            The prepared data.
        estimands : tuple of str
            The resolved estimands.

        Returns
        -------
        tuple of tuple
            One ``(folds, seed)`` pair per draw.
        """
        draws = super()._repeat_draws(data, estimands)
        if is_per_arm_oat(self) and (
            data.has_strata or data.has_missing_outcome or data.has_missing_treatment
        ):
            for folds, seed in draws:
                self._preflight_per_arm_design(data, folds, seed)
        return draws

    def _preflight_per_arm_design(self, data: CausalData, folds: Folds, seed: int | None) -> None:
        """Refuse a per-arm design that cannot fit some factor, before any learner.

        The per-arm design fits each factor of :func:`_per_arm_factor_specs` for each arm,
        inside each baseline stratum, on each training complement of ``folds``. The
        treatment factor needs a recorded row in arm ``a`` and one outside it. The outcome
        observation factor needs a recorded row of arm ``a`` with an observed outcome. The
        treatment observation factor needs a row with a recorded treatment, which the
        treatment factor's requirement already implies, so it has no check of its own. A
        target with one value on its rows fits no learner. The full-``W`` response model
        that the missingness tilt reads needs a nonrespondent among the recorded rows of
        each complement. A package classification Super Learner needs two rows of each class it
        sees. The general preflight asks none of these questions for each stratum and arm,
        and it does not run for a request of ``ey_obs`` alone.

        Parameters
        ----------
        data : CausalData
            The prepared data.
        folds : Folds
            The draw's realized split.
        seed : int or None
            The draw's seed, to resolve the treatment learner as the fit does.

        Raises
        ------
        ValueError
            If some stratum, arm, factor and training complement cannot be fitted.
        """
        from ..learners.super_learner import SuperLearner
        from ._strata import refuse_untrainable_stratum_folds
        from .tmle import _SUPER_LEARNER_INNER_SPLIT_RULE

        treatment = np.asarray(data.treatment, dtype=float)
        recorded = np.asarray(data.treatment_recorded, dtype=bool)
        observed = np.asarray(data.observed, dtype=bool)

        def treatment_rows(arm: float) -> list[tuple[BoolArray, str]]:
            label = repr(data.arm_label(arm))
            return [
                (recorded & (treatment == arm), f"no recorded row is in arm {label}"),
                (recorded & (treatment != arm), f"every recorded row is in arm {label}"),
            ]

        def observation_rows(arm: float) -> list[tuple[BoolArray, str]]:
            # The treatment observation factor fits on every row and takes a constant
            # target, so it needs only some row in the region. The treatment design above
            # already requires a recorded row there, so it has no requirement of its own.
            if not data.has_missing_outcome:
                return []
            label = repr(data.arm_label(arm))
            return [
                (
                    recorded & (treatment == arm) & observed,
                    f"no recorded row of arm {label} has an observed outcome",
                )
            ]

        refuse_untrainable_stratum_folds(
            data, folds, family="treatment design", requirements=treatment_rows
        )
        refuse_untrainable_stratum_folds(
            data, folds, family="observation design", requirements=observation_rows
        )
        complements = [np.arange(data.n)] if folds.is_single else [train for train, _ in folds]
        if data.has_missing_outcome:
            nonrespondent = recorded & ~observed
            if any(not np.any(nonrespondent[train]) for train in complements):
                message = (
                    "no recorded row has a missing outcome, so the missingness model "
                    "P(Delta = 1 | A, W) that the missingness tilt reads cannot be fitted."
                )
                if folds.is_single:
                    raise ValueError(f"inside the sample: {message}")
                raise ValueError(
                    "inside a cross-fitting fold's training complement: "
                    + message
                    + " "
                    + _POST_DRAW_REMEDY.format(remedy="fit with cross_fit=False")
                )
        learner = self._resolve_learner(self.treatment_learner, task="classification", seed=seed)
        if not isinstance(learner, SuperLearner):
            return
        if data.strata is None:
            regions = [("the sample", np.ones(data.n, dtype=bool))]
        else:
            regions = [
                (f"baseline stratum {data.stratum_label(code)}", data.strata == code)
                for code in range(data.n_strata)
            ]
        rule = _SUPER_LEARNER_INNER_SPLIT_RULE.format(role="treatment")
        for spec in _per_arm_factor_specs(data):
            for arm in data.arm_codes:
                target = spec.target(arm)
                eligible = spec.rows(arm)
                for where, inside in regions:
                    for train in complements:
                        kept = np.zeros(data.n, dtype=bool)
                        kept[train] = True
                        values = target[kept & inside & eligible]
                        classes, counts = np.unique(values, return_counts=True)
                        if classes.size < 2 or counts.min() >= 2:
                            continue
                        raise ValueError(
                            f"inside {where}: the outcome-adaptive factor {spec.label} of arm "
                            f"{data.arm_label(arm)!r} has {int(counts.min())} row(s) of one "
                            f"class in a training complement. {rule} Replace the treatment "
                            "learner with one that is not a package classification "
                            "SuperLearner."
                        )

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
            The realized selection split, carrying its generator record. A split declared
            with ``selection_split=`` is returned as declared, with no generator record.
        """
        if self.selection_split is not None:
            return _declared_split(self.selection_split, data.n)
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

    def _derivative_nuisances(
        self,
        data: CausalData,
        base: NuisanceEstimates,
        config: TMLEConfig,
        intermediate_value: float | None,
        *,
        seed: int | None,
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """Fit the mechanism and its partner, then the C-TMLE0 update.

        Both mechanisms are fitted on exactly the rows the ordinary treatment learner uses:
        every row in sample, and each outer fold's training rows when cross-fitted, with
        predictions on that fold. The lasso source with ``lambdas=None`` chooses the
        mechanism penalty by ``learner_folds``-fold weighted deviance cross-validation over
        ``glmnet``'s default grid inside each training set. The grid is fixed once on the
        analysis sample. The partner is the grid neighbour of :func:`nearest_neighbour`.
        The update is one solve of the ``2K`` columns on every row, pooled over the folds.
        Its targeted fit is the state the ordinary continuation starts from.
        """
        arms = data.arm_codes
        seed_value = self.random_state if seed is None else seed
        everything: IntArray = np.arange(data.n)
        if base.folds.is_single:
            sets = [(everything, everything, "the analysis sample")]
        else:
            sets = [
                (train, test, f"outer fold {fold}") for fold, (train, test) in enumerate(base.folds)
            ]
        penalties: list[tuple[float, float]] = []
        cv_risks: list[FloatArray] = []
        chosen_risks: list[float] = []
        entries = self.treatment_candidates
        if entries is None:
            source: Literal["lasso", "learners", "arrays"] = "lasso"
            mechanism = np.empty((data.n, len(arms)))
            partner = np.empty((data.n, len(arms)))
            grid = (
                default_penalty_grid(data.covariates, data.treatment, data.weights)
                if self.lambdas is None
                else None
            )
            for train, test, layer in sets:
                x, a, w = data.covariates[train], data.treatment[train], data.weights[train]
                if grid is None:
                    pair = _check_penalty_pair(self.lambdas)
                else:
                    inner = make_folds(
                        train.size,
                        self.learner_folds,
                        cluster=None if data.cluster is None else data.cluster[train],
                        random_state=seed_value,
                    )
                    index, risk = cv_penalty(
                        x, a, w, grid, inner.assignment, classes=arms, layer=layer
                    )
                    pair = (float(grid[index]), float(grid[nearest_neighbour(grid, index)]))
                    cv_risks.append(risk)
                    chosen_risks.append(float(risk[index]))
                fits = penalty_path(x, a, w, sorted(pair, reverse=True), classes=arms, layer=layer)
                by_penalty = {fit.penalty: fit for fit in fits}
                mechanism[test] = by_penalty[pair[0]].predict_proba(data.covariates[test])
                partner[test] = by_penalty[pair[1]].predict_proba(data.covariates[test])
                penalties.append(pair)
            values = [mechanism, partner]
        else:
            values = []
            kinds = []
            for index, entry in enumerate(entries):
                if _is_array_like(entry):
                    values.append(_derivative_array(index, entry, data))
                    kinds.append("arrays")
                    continue
                learner = resolve_learner(
                    entry,
                    task="classification",
                    n_folds=self.learner_folds,
                    random_state=seed_value,
                )
                predictions, _ = cross_fit_predictions(
                    learner,
                    data.covariates,
                    data.treatment,
                    data.weights,
                    base.folds,
                    task="classification",
                    predict_designs={"g": data.covariates},
                    groups=data.cluster,
                    clip=(0.0, 1.0),
                    classes=arms,
                    n_jobs=self.n_jobs,
                )
                values.append(predictions["g"])
                kinds.append("learners")
            source = "arrays" if kinds == ["arrays", "arrays"] else "learners"
        propensity = Propensity(np.clip(values[0], 0.0, 1.0), arms)
        nuisance = replace(
            base,
            propensity=propensity,
            treatment_partner=Propensity(np.clip(values[1], 0.0, 1.0), arms),
            treatment_covariates=data.covariate_names,
        )
        update = fitted_update(
            data,
            nuisance,
            bounds=config.g_bounds,
            nuisance_bound=self.nuisance_bound,
            intermediate_value=intermediate_value,
            spec=self.targeting_spec(),
        )
        if chosen_risks:
            treatment_risk = float(np.mean(chosen_risks))
        else:
            columns = np.array([propensity.column_for(float(arm)) for arm in data.treatment])
            observed = propensity.values[everything, columns]
            treatment_risk = float(
                -np.sum(data.weights * np.log(np.clip(observed, _LOSS_EPS, 1.0)))
                / np.sum(data.weights)
            )
        fit = CTMLEDerivativeFit(
            strategy="derivative",
            source=source,
            penalties=tuple(penalties),
            cv_risk=tuple(cv_risks),
            epsilon=update.epsilon,
            names=update.names,
            derivative_score_before=update.score_before,
            derivative_score_after=update.score_after,
            treatment_risk=treatment_risk,
        )
        # The fit record needs this solve.  ``targeting_outcome`` stays empty: the retarget
        # rebuilds the update at its own bounds, so a stored ``Q*`` would never be read.
        return nuisance, {"ctmle": fit}

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

        The per-arm design also fits the observation factors on the same column (see
        :func:`_per_arm_factor_specs`), and fits every factor inside each baseline stratum.
        Their product is ``P(C_a = 1 | Qbar(a, W))``, or ``P(C_a = 1 | Qbar(a, W), S)``,
        for the composite indicator ``C_a = Delta_A Delta 1{A = a}``. The base fit's
        full-``W`` response model moves to ``tilt_missingness``, which only the
        missingness tilt reads.

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
        factor_diagnostics: dict[str, Any]
        if base.folds.is_single and per_arm:
            generated = np.column_stack([base.outcome.arms[arm] for arm in arms])
            every = np.arange(data.n)
            factors, factor_diagnostics = _fit_per_arm_factors(
                data, generated, learner, every, every
            )
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
            factors = {"treatment": predictions["g"]}
            factor_diagnostics = {"treatment": diagnostics}
        else:
            base, factors, factor_diagnostics = self._cross_fit_outcome_adaptive(
                data, base, learner, seed=seed
            )
        risk_by_arm: dict[str, float] | None
        risk_by_factor: dict[str, float] | None = None
        specs = _per_arm_factor_specs(data) if per_arm else ()
        labels = {spec.name: spec.label for spec in specs}
        if per_arm:
            propensity = Propensity(factors["treatment"], arms, simplex=False)
            risks = {
                spec.name: {
                    data.arm_label(arm): _binary_risk(
                        factors[spec.name][rows, j],
                        spec.target(arm)[rows],
                        data.weights[rows],
                    )
                    for j, arm in enumerate(arms)
                    for rows in (np.flatnonzero(spec.rows(arm)),)
                }
                for spec in specs
            }
            risk_by_arm = risks["treatment"]
            risk = float(sum(risk_by_arm.values()))
            risk_by_factor = {
                labels[name]: float(sum(values.values())) for name, values in risks.items()
            }
        else:
            propensity = Propensity(factors["treatment"], arms)
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
        treatment_diagnostics = factor_diagnostics["treatment"]
        if (per_arm and any(treatment_diagnostics)) or (not per_arm and treatment_diagnostics):
            nuisance_diagnostics["propensity"] = treatment_diagnostics
        observation: dict[str, Any] = {}
        if per_arm and data.has_missing_outcome:
            # The targeting reads the design factor. The full-W model stays for the tilt.
            observation["missingness"] = factors["outcome_observation"]
            observation["tilt_missingness"] = base.missingness
            _move_diagnostics(
                nuisance_diagnostics,
                "missingness",
                factor_diagnostics["outcome_observation"],
            )
        if per_arm and data.has_missing_treatment:
            observation["treatment_observation"] = factors["treatment_observation"]
            _move_diagnostics(
                nuisance_diagnostics,
                "treatment_observation",
                factor_diagnostics["treatment_observation"],
            )
        nuisance = replace(
            base,
            propensity=propensity,
            treatment_covariates=features,
            diagnostics=nuisance_diagnostics,
            **observation,
        )
        strata = (
            tuple(data.stratum_label(code) for code in range(data.n_strata))
            if per_arm and data.has_strata
            else ()
        )
        return nuisance, {
            "ctmle": CTMLEOutcomeAdaptiveFit(
                strategy="oat",
                treatment_features=features,
                treatment_risk=risk,
                design="per_arm" if per_arm else "shared",
                treatment_risk_by_arm=risk_by_arm,
                strata=strata,
                factors=tuple(spec.label for spec in specs),
                risk_by_factor=risk_by_factor,
            )
        }

    def _cross_fit_outcome_adaptive(
        self,
        data: CausalData,
        base: NuisanceEstimates,
        treatment_learner: Learner,
        *,
        seed: int | None,
    ) -> tuple[NuisanceEstimates, dict[str, FloatArray], dict[str, Any]]:
        """Cross-fit ``Qbar -> g`` as one fold-local nuisance algorithm.

        The first implementation stitched global out-of-fold ``Qbar`` predictions and
        cross-fit ``g`` on that matrix.  For a propensity fold ``v``, predictions on its
        training rows then came from other outcome folds whose fits included rows in
        ``v``.  Changing a validation outcome could therefore change that row's own
        propensity.  Here one outcome model fitted on ``train_v`` creates *both* sides
        of fold ``v``'s generated design before ``g_v`` is fitted on ``train_v``.

        Under ``oat_design="per_arm"`` fold ``v`` fits ``K`` binary mechanisms on
        ``train_v``, each regressing ``1{A = a}`` on the column ``Qbar_v(a, W)`` alone, and
        each observation factor on the same column (see :func:`_per_arm_factor_specs`),
        inside each baseline stratum.  The nesting is the same.  This is the
        cross-validated construction of Benkeser, Cai and van der Laan (2020), Appendix D:
        fold-trained ``Qbar_v`` and ``g_v(. | Qbar_v)``, and one coefficient per arm pooled
        over the validation rows.

        The outcome predictions returned in ``base`` are replaced by the validation
        pieces from these very models.  That matters for stochastic learners: the
        outcome regression used to build a clever covariate cannot be a second refit
        that merely has the same learner settings.

        Returns
        -------
        tuple
            The base nuisances with the fold-local outcome predictions, each factor's
            stitched out-of-fold predictions keyed by factor name (``"treatment"`` alone
            under the shared design), and each factor's learner diagnostics: per arm under
            the per-arm design, per fold under the shared design.
        """
        arms = data.arm_codes
        per_arm = self.oat_design == "per_arm"
        folds = base.folds
        outcome_task = base.outcome_task
        outcome_learner = self._resolve_learner(self.outcome_learner, task=outcome_task, seed=seed)
        outcome_design = data.treatment_design()
        counterfactual = {arm: data.counterfactual_design(arm) for arm in arms}
        scaled = base.scaler.scale(data.outcome)
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
            dict[str, FloatArray],
            Any,
            dict[str, Any],
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
            outcome_diagnostic: Any
            if self.outcome_by_arm:
                # One model per arm, each on the arm's rows of this fold's training set.
                arm_models = _fit_arm_models(
                    outcome_learner,
                    data,
                    data.covariates,
                    scaled,
                    outcome_rows,
                    task=outcome_task,
                    where=f"training fold {fold + 1}",
                )
                observed = _predict_by_arm(
                    arm_models,
                    data.covariates,
                    _arm_codes_by_row(data, data.treatment),
                    task=outcome_task,
                    clip=(0.0, 1.0),
                )
                by_arm = {
                    arm: _predict_by_arm(
                        arm_models, data.covariates, arm, task=outcome_task, clip=(0.0, 1.0)
                    )
                    for arm in arms
                }
                outcome_diagnostic = [
                    getattr(model, "diagnostics_", None) for model in arm_models.values()
                ]
            else:
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
                outcome_diagnostic = getattr(outcome_model, "diagnostics_", None)
            generated = np.column_stack([by_arm[arm] for arm in arms])
            fold_factors: dict[str, FloatArray]
            fold_diagnostics: dict[str, Any]
            if per_arm:
                fold_factors, fold_diagnostics = _fit_per_arm_factors(
                    data, generated, treatment_learner, train, test
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
                fold_factors = {
                    "treatment": np.clip(
                        predict_probabilities(treatment_model, generated[test], arms), 0.0, 1.0
                    )
                }
                fold_diagnostics = {"treatment": getattr(treatment_model, "diagnostics_", None)}
            return (
                fold,
                test,
                observed[test],
                {arm: values[test] for arm, values in by_arm.items()},
                fold_factors,
                outcome_diagnostic,
                fold_diagnostics,
            )

        observed = np.empty(data.n, dtype=float)
        by_arm = {arm: np.empty(data.n, dtype=float) for arm in arms}
        factors: dict[str, FloatArray] = {}
        outcome_diagnostics: list[Any] = []
        factor_diagnostics: dict[str, Any] = {}
        for (
            _,
            test,
            fold_observed,
            fold_arms,
            fold_factors,
            q_diagnostic,
            fold_diagnostics,
        ) in map_parallel(run_fold, jobs, n_jobs=self.n_jobs):
            observed[test] = fold_observed
            for name, values in fold_factors.items():
                if name not in factors:
                    factors[name] = np.empty((data.n, len(arms)), dtype=float)
                factors[name][test] = values
            for arm in arms:
                by_arm[arm][test] = fold_arms[arm]
            if self.outcome_by_arm:
                outcome_diagnostics.extend(item for item in q_diagnostic if item is not None)
            elif q_diagnostic is not None:
                outcome_diagnostics.append(q_diagnostic)
            for name, entry in fold_diagnostics.items():
                if per_arm:
                    # One list per arm, each holding that arm's fold diagnostics, so the
                    # per-arm calibration report reads the arm's own learner table.
                    stored = factor_diagnostics.setdefault(name, [[] for _ in arms])
                    for j, arm_entries in enumerate(entry):
                        stored[j].extend(arm_entries)
                else:
                    stored = factor_diagnostics.setdefault(name, [])
                    if entry is not None:
                        stored.append(entry)

        diagnostics = dict(base.diagnostics)
        diagnostics.pop("outcome", None)
        if outcome_diagnostics:
            diagnostics["outcome"] = outcome_diagnostics
        honest_base = replace(
            base,
            outcome=InitialFit(observed, by_arm),
            diagnostics=diagnostics,
        )
        return honest_base, factors, factor_diagnostics

    def _retarget_detailed(
        self, data: CausalData, nuisance: NuisanceEstimates, **kwargs: Any
    ) -> Any:
        """Continue targeting from the collaboratively selected ``Qbar*``.

        Keep that state separate from ``nuisance.outcome``: the latter is the initial
        outcome learner and is what calibration and risk diagnostics are about.  The
        replacement is local, so the result continues to expose both states faithfully.
        """
        if self.strategy == "derivative" and nuisance.treatment_partner is not None:
            # C-TMLE0 at the requested bounds: the derivative columns are rebuilt from the
            # initial fit, the mechanism and the partner, so a truncation sweep moves them.
            bounds = kwargs.get("g_bounds") or resolve_g_bounds(
                self.g_bounds, self._bounds_n(data), for_att=False
            )
            bound = kwargs.get("nuisance_bound")
            update = fitted_update(
                data,
                nuisance,
                bounds=bounds,
                nuisance_bound=self.nuisance_bound if bound is None else bound,
                intermediate_value=kwargs.get("intermediate_value"),
                spec=self.targeting_spec(),
                missingness=kwargs.get("missingness"),
            )
            working = replace(nuisance, outcome=update.targeted, targeting_outcome=None)
            return super()._retarget_detailed(data, working, **kwargs)
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
        if self._has_arrays():
            raise CapabilityError(
                "a refit adds, replaces or perturbs a column, and the PropensityArrays "
                "candidates were fitted on the original columns.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Pass learners as treatment_candidates= to refit.",
            )
        if self._has_derivative_arrays():
            raise CapabilityError(
                "The refit changes the data (the covariates, the treatment or the rows), and a "
                "propensity array entry was fitted without that change.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Pass learners to refit.",
            )
        if self.strategy == "discrete" and self.candidates is not None:
            recorded = set(data.added_covariates)
            added = tuple(name for name in data.mechanism_covariate_names if name in recorded)
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
        added = tuple(name for name in data.mechanism_covariate_names if name not in declared)
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
                "it minimises would have no fixed target. docs/roadmap.md F5 tracks this stop.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Use a plain TMLE.",
                roadmap="F5",
            )
        if data.has_strata and self.strategy == "derivative":
            raise CapabilityError(
                "strategy='derivative' has no stratified update. Each stratum estimate would "
                "need its own derivative columns, which neither R ctmle type 3 nor the "
                "C-TMLE0 sources define.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    "Drop strata=, or fit the ordinary TMLE, which reports stratum estimates."
                ),
                roadmap="X62",
            )
        if data.cluster is not None and self.strategy != "derivative":
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
                f"C-TMLE has no clustered result. {reason}. docs/roadmap.md X51 tracks the "
                "point estimate with a cluster as the unit, and docs/roadmap.md F22 the "
                "clustered interval.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    "Drop id= from fit (PointTreatment(cluster=None)), or use the ordinary "
                    "TMLE (TMLE, or TMLEMethod), which has a clustered result."
                ),
                roadmap="F22",
            )
        self._check_treatment_pool(data)
        self._check_estimands(data)
        return super()._resolve_estimands_for_data(data)

    def _check_treatment_pool(self, data: CausalData) -> None:
        """Refuse a pool that a design's treatment-mechanism covariates leave empty of its names.

        Runs before any split is drawn and before any learner.  ``strategy="oat"`` fits its
        mechanism on the outcome predictions and no covariate, so there is nothing for
        ``treatment_adjustment=`` to restrict (R-h2).  An explicit ``candidates=`` or
        ``ordering=`` names the covariates the search draws from, so a name outside the
        design's subset is refused (R-h5).

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Raises
        ------
        CapabilityError
            For ``strategy="oat"`` beside a declared subset.
        ValueError
            For a candidate or an ordering that names a covariate outside the subset.
        """
        if data.treatment_covariate_names is None:
            return
        if self.strategy == "oat":
            raise CapabilityError(
                "strategy='oat' fits the treatment mechanism on the outcome predictions, so "
                "treatment_adjustment= has no covariates to restrict.",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Use another strategy, or omit treatment_adjustment=.",
            )
        if self.candidates is not None:
            _refuse_names_outside_pool(
                data, "candidates", [name for names in self.candidates for name in names]
            )
        if self.ordering is not None:
            _refuse_names_outside_pool(data, "ordering", self.ordering)

    def _check_policies(self, data: CausalData) -> None:
        """Refuse each intervention keyword, then a continuous dose, before the shift check.

        ``TMLE._check_policies`` runs first in the preflight, and it answers a dose with no
        ``policies=`` by suggesting one, which CTMLE does not fit.  ``policies=`` is asked
        first, because a shift declares a dose even on a binary column.  ``interventions=``
        and ``msm=`` over discrete arms are asked next, so neither meets the ATT and ATC
        refusal of :meth:`_check_estimands` for a reason that is not its own.
        """
        if self.policies:
            raise CapabilityError.from_refusal(_POLICIES_REFUSAL)
        if data.is_continuous_treatment:
            raise CapabilityError.from_refusal(_DISCRETE_TREATMENT_REFUSAL)
        if self.interventions:
            raise CapabilityError.from_refusal(_REGIME_REFUSAL)
        if self.msm is not None:
            raise CapabilityError.from_refusal(_MSM_REFUSAL)
        super()._check_policies(data)

    def _check_estimands(self, data: CausalData) -> None:
        if data.is_continuous_treatment:
            raise CapabilityError.from_refusal(_DISCRETE_TREATMENT_REFUSAL)
        if data.has_intermediate:
            raise CapabilityError(
                "CTMLE does not compose either collaborative strategy with an intermediate "
                "outcome.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Fit each controlled direct effect with TMLE instead.",
                roadmap="F5",
            )
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms)
        conditional = [name for name in estimands if name not in MEAN_GROUP_ESTIMANDS]
        if conditional:
            raise CapabilityError(
                f"CTMLE does not support estimand(s) {conditional}: the ATT and ATC clever "
                "covariates condition on a random event, so a single collaboratively "
                "selected treatment model cannot serve them alongside the ATE.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    "Request them from a plain TMLE, or set "
                    f"estimands={sorted(MEAN_GROUP_ESTIMANDS)!r}."
                ),
                roadmap="F5",
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
                "and require a separate collaborative derivation.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative=(
                    f"Choose a ctmle_estimand from {sorted(supported)}, or fit a plain TMLE."
                ),
                roadmap="F5",
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
        #: The units and outcome range a declared stopping rule needs, over all the rows;
        #: `target` hands each candidate solve `stop.restrict(rows)`. `None` under `relative`.
        self.stop = estimator._stop_context(data, base)
        self._cache: dict[tuple[Any, ...], FloatArray] = {}
        self._active: dict[tuple[Any, ...], tuple[str, ...]] = {}
        self.reference = estimator._reference_arm(data)
        self.target_names = self._target_names()
        #: The penalty grid of ``strategy="lasso"``. It reads treatment and covariates only
        #: and is fixed on the analysis sample: every selection fold and inner fold reuses
        #: it, as ``ctmleGlmnet`` reuses its ``lambdas``.
        self.lambdas: FloatArray | None = None
        if estimator.strategy == "lasso":
            self.lambdas = (
                check_penalty_grid(estimator.lambdas)
                if estimator.lambdas is not None
                else default_penalty_grid(data.mechanism_covariates, data.treatment, data.weights)
            )

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
            stop=None if self.stop is None else self.stop.restrict(rows),
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
        if self.est.strategy in _INDEXED_STRATEGIES:
            return self._indexed_candidates_path(rows, train, tag)
        if self.est.strategy == "discrete":
            return self._discrete_path(rows, train, tag)
        order = self._ordering(rows, train) if self.est.strategy == "ordered" else None
        return self._forward_path(rows, train, tag, order)

    # ------------------------------------------------------- indexed sequences

    def indexed_propensities(
        self, train: IntArray | None, tag: str
    ) -> tuple[list[Propensity], list[tuple[str, ...]], list[str]]:
        """The candidates of ``"lasso"`` or ``"learners"`` on one search branch.

        A lasso candidate is the L1-penalized mechanism at each penalty of the grid. A
        learners candidate is each declared entry in the declared order, then
        ``treatment_learner`` on every covariate. Every learner is fitted on exactly the
        rows a covariate-set candidate uses. A :class:`PropensityArrays` entry reads
        :attr:`~PropensityArrays.full` on the analysis sample and
        :attr:`~PropensityArrays.cv` inside each selection fold.

        Parameters
        ----------
        train : ndarray of int or None
            The selection-fold training rows, or ``None`` on the analysis sample.
        tag : str
            The search branch, ``"full"`` or ``"cv<fold>"``.

        Returns
        -------
        tuple
            The propensities, the covariates of each candidate and the label of each.
        """
        names = tuple(self.data.mechanism_covariate_names)
        if self.est.strategy == "lasso":
            assert self.lambdas is not None
            propensities = self._lasso_propensities(train, tag)
            covariates = [self._active[(tag, ("lasso", k))] for k in range(len(propensities))]
            labels = [_penalty_label(value) for value in self.lambdas]
            return propensities, covariates, labels
        entries = list(self.est.treatment_candidates or ())
        propensities = []
        labels = []
        for index, entry in enumerate(entries):
            if isinstance(entry, PropensityArrays):
                propensities.append(self._array_propensity(index, entry))
                labels.append(f"arrays[{index}]")
                continue
            key = (tag, ("learner", index))
            cached = self._cache.get(key)
            if cached is None:
                learner = resolve_learner(
                    entry,
                    task="classification",
                    n_folds=self.est.learner_folds,
                    random_state=self.seed,
                )
                cached = self._fit_propensity_with(learner, names, train).values
                self._cache[key] = cached
            propensities.append(Propensity(cached, self.data.arm_codes))
            labels.append(f"{_learner_name(entry)}[{index}]")
        propensities.append(self.propensity(names, train, tag))
        labels.append(f"{_learner_name(self.est.treatment_learner)}[terminal]")
        return propensities, [names] * len(propensities), labels

    def _array_propensity(self, index: int, entry: PropensityArrays) -> Propensity:
        """``full`` on the analysis sample, ``cv`` on every row inside a selection fold."""
        labels = [self.data.arm_label(arm) for arm in self.data.arm_codes]
        full, cv = entry.resolved(index, self.data.n, self.data.arm_codes, labels)
        values = full if self.train_folds is None else cv
        return Propensity(np.clip(values, 0.0, 1.0), self.data.arm_codes)

    def _fit_rows(self, train: IntArray | None, tag: str) -> list[tuple[IntArray, IntArray, str]]:
        """The ``(fit rows, predicted rows, layer)`` of each fit a candidate needs.

        The same rows :meth:`_fit_propensity_with` gives a covariate-set candidate: the
        nested folds inside a selection fold, the selection-fold training rows, the
        analysis sample in sample, or each outer fold when cross-fitted.
        """
        where = "the analysis sample" if tag == "full" else f"selection fold {tag[2:]}"
        if self.train_folds is not None:
            assert self.train_mask is not None
            out = []
            last = self.train_folds.n_folds - 1
            for fold, (fit_rows, test) in enumerate(self.train_folds):
                eligible = fit_rows[self.train_mask[fit_rows]]
                if eligible.size == 0:
                    raise ValueError(
                        "a cross-fitting fold has no trainable rows for a nuisance model. "
                        f"{_SELECTION_SPLIT_BACKSTOP}"
                    )
                layer = (
                    f"{where}, full-training fit" if fold == last else f"{where}, inner fold {fold}"
                )
                out.append((eligible, test, layer))
            return out
        if train is None:
            if self.base.folds.is_single:
                return [(self.all_rows, self.all_rows, where)]
            return [
                (fit_rows, test, f"outer fold {fold}")
                for fold, (fit_rows, test) in enumerate(self.base.folds)
            ]
        return [(train, self.all_rows, where)]

    def _lasso_propensities(self, train: IntArray | None, tag: str) -> list[Propensity]:
        """One warm-started penalty path for each training set, cached per branch."""
        assert self.lambdas is not None
        data = self.data
        design = data.mechanism_covariates
        count = self.lambdas.size
        keys = [(tag, ("lasso", k)) for k in range(count)]
        if all(key in self._cache for key in keys):
            return [Propensity(self._cache[key], data.arm_codes) for key in keys]
        values = np.empty((count, data.n, data.n_arms))
        active: list[set[int]] = [set() for _ in range(count)]
        for fit_rows, predict_rows, layer in self._fit_rows(train, tag):
            fits = penalty_path(
                design[fit_rows],
                data.treatment[fit_rows],
                data.weights[fit_rows],
                self.lambdas,
                classes=data.arm_codes,
                layer=layer,
            )
            for k, fit in enumerate(fits):
                values[k, predict_rows] = fit.predict_proba(design[predict_rows])
                active[k].update(fit.active)
        out = []
        for k, key in enumerate(keys):
            self._cache[key] = np.clip(values[k], 0.0, 1.0)
            self._active[key] = tuple(
                name for j, name in enumerate(data.mechanism_covariate_names) if j in active[k]
            )
            out.append(Propensity(self._cache[key], data.arm_codes))
        return out

    def _indexed_candidates_path(
        self, rows: IntArray, train: IntArray | None, tag: str
    ) -> list[_Candidate]:
        propensities, covariates, labels = self.indexed_propensities(train, tag)
        self._indexed = (propensities, covariates, labels)

        def fluctuate(index: int, initial: InitialFit, n_steps: int) -> _Candidate:
            return self._fluctuate(
                covariates[index], propensities[index], initial, rows, n_steps, labels[index]
            )

        return _indexed_path(len(propensities), self.base.outcome, fluctuate)

    def final_step(
        self, path: Sequence[_Candidate], selected: int
    ) -> tuple[int, _Candidate] | None:
        """The C-TMLE1 final step of ``ctmleGlmnet`` on the analysis sample.

        Fluctuate the selected candidate's targeted fit with each mechanism of a smaller
        penalty, and report the one with the smallest unpenalized loss. The paper
        fluctuates the selected candidate's initial fit (Ju et al. 2019, C-LASSO
        Appendix A, Algorithm 1, steps 14 and 15); R ``ctmleGlmnet`` and this package
        fluctuate its targeted fit (``R/ctmle_glmnet.R``, line 291).

        Parameters
        ----------
        path : sequence of _Candidate
            The analysis-sample path.
        selected : int
            The cross-validated index.

        Returns
        -------
        tuple of (int, _Candidate) or None
            The final index and its fluctuation, or ``None`` when no smaller penalty
            exists.
        """
        if selected >= len(path) - 1:
            return None
        propensities, covariates, labels = self._indexed
        start = path[selected]
        options = [
            self._fluctuate(
                covariates[index],
                propensities[index],
                start.targeted,
                self.all_rows,
                start.n_steps + 1,
                labels[index],
            )
            for index in range(selected + 1, len(path))
        ]
        losses = np.array([option.loss for option in options], dtype=float)
        losses[~np.isfinite(losses)] = np.inf
        best = int(np.argmin(losses))
        return selected + 1 + best, options[best]

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
        pool = list(order) if order is not None else list(self.data.mechanism_covariate_names)
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
        return self._fluctuate(covariates, propensity, initial, rows, n_steps)

    def _fluctuate(
        self,
        covariates: tuple[str, ...],
        propensity: Propensity,
        initial: InitialFit,
        rows: IntArray,
        n_steps: int,
        label: str = "",
    ) -> _Candidate:
        """Fluctuate ``initial`` along one candidate mechanism and score it on ``rows``."""
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
            label=label,
        )

    def _ordering(
        self,
        rows: IntArray,
        train: IntArray | None,
    ) -> tuple[str, ...]:
        """Published logistic or partial-correlation order for the scalable search."""
        pool = self.data.mechanism_covariate_names
        if self.est.ordering is not None:
            names = tuple(self.est.ordering)
            _refuse_names_outside_pool(self.data, "ordering", names)
            unknown = [name for name in names if name not in self.data.covariate_names]
            if unknown:
                raise ValueError(
                    f"ordering names unknown covariate(s) {unknown}; "
                    f"available: {list(self.data.covariate_names)}"
                )
            missing = [name for name in pool if name not in set(names)]
            if missing:
                raise ValueError(
                    f"ordering must cover every covariate; missing {missing}. Use "
                    "strategy='discrete' to search a restricted set of models."
                )
            return names

        if self.est.preorder == "logistic":
            score_values = []
            logistic = LogisticRegression(C=1e6, max_iter=1000, random_state=self.seed)
            for name in pool:
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
            columns = [self.data.covariate_names.index(name) for name in pool]
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
                    for column in columns
                ]
            )
            order = np.argsort(-scores, kind="stable")
        return tuple(pool[j] for j in order)

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
        if self.est.outcome_by_arm:
            outcome_out, _, _ = _fit_outcome_by_arm(
                learner,
                data,
                scaled,
                train_folds,
                task=outcome_task,
                queries={
                    "observed": ArmQuery(_arm_codes_by_row(data, data.treatment)),
                    **{f"arm@{arm}": ArmQuery(arm) for arm in data.arm_codes},
                },
                fit_mask=train_mask & data.observed,
                n_jobs=self.est.n_jobs,
                remedy=_SELECTION_SPLIT_BACKSTOP,
            )
        else:
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


def _indexed_path(
    count: int,
    initial: InitialFit,
    fluctuate: Callable[[int, InitialFit, int], _Candidate],
) -> list[_Candidate]:
    """The indexed construction rule of van der Laan and Gruber (2010), Section 2.4.

    From the current initial fit, fluctuate with every remaining candidate and take the
    first smallest criterion. When it does not improve on the best criterion so far, stop.
    Otherwise record each candidate from the previous accepted index up to this one as a
    fluctuation of the current initial fit, make the accepted candidate's targeted fit the
    next initial fit, and drop every candidate up to and including it. After a stop, the
    remaining candidates are fluctuations of the last initial fit. Candidate ``k`` is always
    position ``k``, so every branch's path has length ``count``. This is C-LASSO Appendix A,
    Algorithm 1, steps 5 to 12, and ``construct_candidates_general`` in R ``ctmle``
    (``R/functions_general.R``, argmin line 81, jump line 124, stop line 129).

    A pure function of the candidates and the scoring callable, so a test can feed it R's
    own propensity arrays.

    Parameters
    ----------
    count : int
        The number of candidates.
    initial : InitialFit
        The initial outcome regression.
    fluctuate : callable
        ``fluctuate(k, initial, n_steps)`` returns candidate ``k`` fluctuated from
        ``initial``, with its criterion in ``risk`` and its targeted fit in ``targeted``.

    Returns
    -------
    list of _Candidate
        The ``count`` candidates in index order.
    """
    path: list[_Candidate | None] = [None] * count
    current = initial
    best = np.inf
    start = 0
    accepted = 0
    tries: list[_Candidate] = []
    while start < count:
        tries = [fluctuate(index, current, accepted + 1) for index in range(start, count)]
        scores = np.array([candidate.risk for candidate in tries], dtype=float)
        scores[~np.isfinite(scores)] = np.inf
        choice = int(np.argmin(scores))
        if scores[choice] > best:
            break
        best = float(scores[choice])
        for offset in range(choice + 1):
            path[start + offset] = tries[offset]
        current = tries[choice].targeted
        start += choice + 1
        accepted += 1
        tries = []
    for offset, candidate in enumerate(tries):
        path[start + offset] = candidate
    assert all(candidate is not None for candidate in path)
    return [candidate for candidate in path if candidate is not None]


def _derivative_array(index: int, entry: Any, data: CausalData) -> FloatArray:
    """A ``strategy="derivative"`` array entry as ``(n, K)`` probabilities, or ``ValueError``."""
    n, arms = data.n, data.arm_codes
    k = len(arms)
    array = np.asarray(entry, dtype=float)
    shape = array.shape
    if k == 2 and array.ndim == 1 and array.shape[0] == n:
        array = np.column_stack([1.0 - array, array])
    valid = (
        array.shape == (n, k)
        and bool(np.all(np.isfinite(array)))
        and bool(np.all((array >= 0.0) & (array <= 1.0)))
    )
    if not valid:
        labels = [data.arm_label(arm) for arm in arms]
        raise ValueError(
            f"treatment_candidates= entry {index}: the array must hold {n} finite "
            f"probabilities in [0, 1] for each of the {k} arms in arm order {labels}; a "
            f"vector of P(A = {labels[1]}) is accepted at two arms; got shape {shape}"
        )
    return array


def _check_penalty_pair(lambdas: Any) -> tuple[float, float]:
    """The ``strategy="derivative"`` pair as two floats, or ``ValueError``."""
    try:
        values = np.asarray(lambdas, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        values = np.array([np.nan])
    valid = (
        values.size == 2
        and bool(np.all(np.isfinite(values)))
        and bool(np.all(values >= 0.0))
        and values[0] != values[1]
    )
    if not valid:
        raise ValueError(
            "Under strategy='derivative', lambdas= is the pair (mechanism_penalty, "
            "partner_penalty): two finite, non-negative and different penalties in any "
            "order. Omit it to choose the mechanism penalty by cross-validation. Got "
            f"{lambdas!r}"
        )
    return float(values[0]), float(values[1])


def _penalty_label(value: float) -> str:
    return "lambda=0" if value == 0.0 else f"lambda={value:.4g}"


def _learner_name(learner: Any) -> str:
    if learner is None:
        return "SuperLearner"
    return type(learner).__name__


@dataclass(frozen=True)
class _FactorSpec:
    """One factor of the per-arm outcome-adaptive design, fitted on ``Qbar(a, W)``.

    Parameters
    ----------
    name : str
        The key of the factor: ``"treatment"``, ``"outcome_observation"`` or
        ``"treatment_observation"``.
    label : str
        The conditional probability the factor estimates, for a generic arm ``a``.
    target : callable
        Maps an arm code to the binary target on every row.
    rows : callable
        Maps an arm code to the rows the factor is fitted on.
    constant : bool
        Whether a target with one value on the fitting rows predicts that value and fits
        no learner. Only the observation factors take it: a stratum where every row of an
        arm responds has the observation probability one there.
    """

    name: str
    label: str
    target: Callable[[float], FloatArray]
    rows: Callable[[float], BoolArray]
    constant: bool


def _per_arm_factor_specs(data: CausalData) -> tuple[_FactorSpec, ...]:
    """The factors of the per-arm design on ``data``, in the order the product takes them.

    For arm ``a`` with ``q_a = Qbar(a, W)``: the treatment factor
    ``P(A = a | Delta_A = 1, q_a)`` on the rows with a recorded treatment; with missing
    outcomes, the outcome observation factor ``P(Delta = 1 | A = a, Delta_A = 1, q_a)`` on
    the recorded rows of arm ``a``; and with a missing treatment, the treatment observation
    factor ``P(Delta_A = 1 | q_a)`` on every row. With baseline strata each conditional also
    holds ``S``. By the chain rule their product is ``P(C_a = 1 | q_a)`` for the composite
    indicator ``C_a = Delta_A Delta 1{A = a}``, the construction of Benkeser, Cai and van der
    Laan (2020), Theorem 1, applied to ``C_a``.

    Parameters
    ----------
    data : CausalData
        The prepared data.

    Returns
    -------
    tuple of _FactorSpec
        The treatment factor first, then each observation factor the data needs.
    """
    treatment = np.asarray(data.treatment, dtype=float)
    recorded = np.asarray(data.treatment_recorded, dtype=bool)
    observed = np.asarray(data.observed, dtype=bool)
    every = np.ones(data.n, dtype=bool)
    given = "Delta_A=1," if data.has_missing_treatment else ""
    stratum = ",S" if data.has_strata else ""
    specs = [
        _FactorSpec(
            "treatment",
            f"P(A=a|{given}Qbar(a,W){stratum})",
            lambda arm: (treatment == arm).astype(float),
            lambda arm: recorded,
            False,
        )
    ]
    if data.has_missing_outcome:
        specs.append(
            _FactorSpec(
                "outcome_observation",
                f"P(Delta=1|A=a,{given}Qbar(a,W){stratum})",
                lambda arm: observed.astype(float),
                lambda arm: recorded & (treatment == arm),
                True,
            )
        )
    if data.has_missing_treatment:
        specs.append(
            _FactorSpec(
                "treatment_observation",
                f"P(Delta_A=1|Qbar(a,W){stratum})",
                lambda arm: recorded.astype(float),
                lambda arm: every,
                True,
            )
        )
    return tuple(specs)


def _fit_per_arm_factors(
    data: CausalData,
    generated: FloatArray,
    learner: Learner,
    train: IntArray,
    test: IntArray,
) -> tuple[dict[str, FloatArray], dict[str, list[list[Any]]]]:
    """Fit each per-arm factor on ``train`` and predict it on ``test``, inside each stratum.

    Each factor of arm ``a`` regresses its target on the one column ``generated[:, a]``,
    on the rows of ``train`` that the factor is fitted on and that lie in one stratum, and
    predicts that stratum's rows of ``test``. Without baseline strata the whole sample is
    one region, and the treatment factor is the shipped per-arm fit bit for bit. A
    region with no ``test`` row fits nothing. An observation factor whose target takes one
    value on its positive-weight fitting rows predicts that value and fits no learner.

    Parameters
    ----------
    data : CausalData
        The prepared data.
    generated : ndarray
        ``(n, K)``, the outcome model's prediction at each arm on every row.
    learner : Learner
        The resolved treatment learner. Every factor, arm and stratum fits a clone.
    train : ndarray of int
        The rows the factors may train on: every row in sample, or a fold's training
        complement.
    test : ndarray of int
        The rows to predict.

    Returns
    -------
    tuple of dict
        Each factor's ``(len(test), K)`` predictions, and each factor's learner diagnostics
        as one list per arm.

    Raises
    ------
    ValueError
        If a stratum and arm hold no row to fit a factor on. The preflight
        :meth:`CTMLE._preflight_per_arm_design` refuses that case before any learner.
    """
    arms = data.arm_codes
    weights = np.asarray(data.weights, dtype=float)
    if data.strata is None:
        regions = [np.ones(data.n, dtype=bool)]
    else:
        regions = [data.strata == code for code in range(data.n_strata)]
    factors: dict[str, FloatArray] = {}
    diagnostics: dict[str, list[list[Any]]] = {}
    for spec in _per_arm_factor_specs(data):
        out = np.empty((test.size, len(arms)), dtype=float)
        entries: list[list[Any]] = [[] for _ in arms]
        for j, arm in enumerate(arms):
            target = spec.target(arm)
            eligible = spec.rows(arm)
            column = generated[:, [j]]
            for inside in regions:
                at = inside[test]
                if not at.any():
                    continue
                rows = train[eligible[train] & inside[train]]
                if rows.size == 0:
                    raise ValueError(
                        f"no trainable row for the outcome-adaptive factor {spec.label} of arm "
                        f"{data.arm_label(arm)!r}"
                    )
                if spec.constant:
                    values = np.unique(target[rows[weights[rows] > 0.0]])
                    if values.size == 1:
                        out[at, j] = float(values[0])
                        continue
                model = fit_on_rows(
                    learner, column, target, weights, rows, "classification", data.cluster
                )
                out[at, j] = np.clip(
                    predict_mean(model, column[test[at]], "classification"), 0.0, 1.0
                )
                entry = getattr(model, "diagnostics_", None)
                if entry is not None:
                    entries[j].append(entry)
        factors[spec.name] = out
        diagnostics[spec.name] = entries
    return factors, diagnostics


def _move_diagnostics(diagnostics: dict[str, Any], key: str, entries: list[list[Any]]) -> None:
    """Report a per-arm design factor under ``key`` and keep the full-``W`` model's entry.

    The base fit's entry under ``key`` describes the full-``W`` model, which only the
    missingness tilt reads after a per-arm fit. It moves to ``"tilt_" + key``. The outcome
    observation factor has one calibration report, read at the realised arm, so its
    entries are pooled over the arms. The treatment observation factor has one report per
    arm, so it keeps one list per arm.

    Parameters
    ----------
    diagnostics : dict
        The nuisance diagnostics, edited in place.
    key : str
        ``"missingness"`` or ``"treatment_observation"``.
    entries : list of list
        The design factor's learner diagnostics, one list per arm.
    """
    full = diagnostics.pop(key, None)
    if full is not None:
        diagnostics[f"tilt_{key}"] = full
    if not any(entries):
        return
    if key == "treatment_observation":
        diagnostics[key] = entries
    else:
        diagnostics[key] = [entry for arm_entries in entries for entry in arm_entries]


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


def _declared_fold_count(labels: Sequence[int]) -> int | None:
    """The fold count ``V`` that integer labels declare, or ``None`` when they are malformed.

    A malformed split is refused at fit time by :func:`_declared_split`, which also knows ``n``.
    """
    try:
        values = np.asarray(labels, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not bool(np.all(np.isfinite(values) & (values % 1 == 0))):
        return None
    count = int(values.max()) + 1
    return count if count >= 2 else None


def _declared_split(labels: Sequence[int], n: int) -> Folds:
    """The ``selection_split=`` labels as a split, or ``ValueError`` when they are malformed."""
    try:
        values = np.asarray(labels).reshape(-1)
        integral = values.size > 0 and bool(np.all(np.asarray(values, dtype=float) % 1 == 0))
    except (TypeError, ValueError):
        values = np.asarray([], dtype=float)
        integral = False
    codes = values.astype(np.int64) if integral else np.asarray([], dtype=np.int64)
    n_folds = int(codes.max()) + 1 if codes.size else 0
    valid = (
        integral
        and codes.size == n
        and int(codes.min()) == 0
        and n_folds >= 2
        and bool(np.all(np.bincount(codes, minlength=n_folds) > 0))
    )
    if not valid:
        found = (
            f"{values.size} labels with values {sorted(set(values.tolist()))[:6]}"
            if values.size
            else "no labels"
        )
        raise ValueError(
            f"selection_split must hold one fold label for each of the {n} rows, with labels "
            f"0 to V-1 and V >= 2; got {found}"
        )
    return Folds(codes, n_folds)


def _is_array_like(value: Any) -> bool:
    """Whether a ``treatment_candidates=`` entry is data rather than a learner."""
    if hasattr(value, "fit"):
        return False
    return isinstance(value, (np.ndarray, list, tuple)) or hasattr(value, "__array__")


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
        if isinstance(selection, CTMLEDerivativeFit):
            return (
                "used strategy='derivative', which adds derivative columns to the "
                "fluctuation. R ctmleGlmnet(ctmletype=3) reports the plain D* variance, which "
                "std_error gives up to n/(n-1)."
            )
        return (
            "is not a selector C-TMLE fit (strategy 'greedy', 'ordered', 'discrete', 'lasso' "
            "or 'learners')"
        )
    if selection.strategy == "lasso":
        return (
            "used strategy='lasso', which fits an L1-penalized logistic path rather than one "
            "logistic model on a covariate set. R ctmleGlmnet type 1 reports the plain D* "
            "variance, var(D*) / n. plugin_std_error squared is the same variance."
        )
    if selection.strategy == "learners":
        return (
            "used strategy='learners', which fits declared candidates rather than one "
            "logistic model on a covariate set."
        )
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
        raise CapabilityError(
            f"{_LOGISTIC_PLUGIN_SCOPE}{reason}.",
            reason=RefusalReason.DIFFERENT_ESTIMAND,
            alternative="Read estimate.plugin_interval, the D* interval, for this fit.",
        )
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
