r"""Collaborative TMLE: choosing the treatment model against the *target* parameter.

A plain TMLE fits ``g(W) = P(A = 1 | W)`` to predict treatment as well as possible,
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
    current targeted fit as the new starting point and searches again -- which is what
    keeps the risk along the sequence monotone.  Costs ``O(V p^2)`` propensity fits for
    ``p`` covariates and ``V`` selection folds.

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
    on treatment.  Pass ``ordering=`` to supply a fixed order instead.

``strategy="discrete"``
    Cross-validated selection among an explicit list of candidate covariate sets --
    the analogue of ``ctmleDiscrete`` / ``ctmleGlmnet``.

``strategy="oat"``
    The outcome-adaptive treatment mechanism from ``ctmle3::LF_oat``.  This is not a
    fourth candidate sequence: it fits categorical treatment on the complete vector
    ``[Qbar(a, W): a in arms]`` and then uses the ordinary all-arm mean fluctuation.
    Consequently it supports multi-valued treatment and has no selector loss or stopping
    index.

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
                   + \widehat{\operatorname{Var}}(D^*)
                   + n\,\bar D^{*2},

where :math:`D^*` is the candidate's estimated efficient influence curve on the rows
being scored and :math:`\bar D^*` its mean -- the part of the score the targeting
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

The influence-curve standard error conditions on the selected propensity model.  It
does not include the variability the *selection* contributes, and so runs mildly
anti-conservative -- in simulation on the instrument process below, a reported
standard error about 12% under the true spread of the estimates.  That is a smaller
error than the variance C-TMLE saves, so the interval is still narrower and its
coverage no worse than a plain TMLE's; but where the selection matters and honest
inference is the point, pass ``n_bootstrap=``.  Each replicate re-runs the search, so
the bootstrap standard error does see it.

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
come through ``g``, and on the same instrument process the greedy search includes the
confounder ``W1`` in every seed, never selects the empty model, and still leaves the
instrument out -- while a selector restricted to the empty candidate has mean absolute
error 0.696 against the collaborative fit's 0.017. See
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

Example
-------
>>> from cleverly import CTMLE
>>> from cleverly.datasets import make_instrument
>>> frame, truth = make_instrument(n=1000, seed=0)
>>> res = CTMLE(outcome_learner="glm", treatment_learner="glm").fit(
...     frame, outcome="Y", treatment="A"
... ).single()
>>> res.extra["ctmle"].selected_covariates            # doctest: +SKIP
()

On this process that empty result is the right answer rather than a failure to select:
``Qbar`` is correctly specified, so ``g`` has nothing left to adjust for.  Inspect
``res.extra["ctmle"].cv_risk`` alongside ``.path`` to see what each candidate was worth,
and read the section above before concluding anything from a comparison against a plain
fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..fluctuation.iterative import InitialFit, apply_logistic, check_matching_arms
from ..fluctuation.submodel import Submodel, restrict, weighted_form
from ..inference.influence import counterfactual_means, ratio_estimates
from ..learners._fitting import Task, predict_mean
from ..learners.crossfit import Folds, check_integrity, make_folds
from ..learners.super_learner import resolve_learner
from ..utils.bounds import OutcomeScaler, resolve_g_bounds
from ..utils.text import format_table
from ._nuisance import NuisanceEstimates, Propensity, cross_fit_predictions, fit_on_rows
from .base import MEAN_GROUP_ESTIMANDS, TMLEConfig, resolve_estimands
from .targeting import build_submodel, solve_submodel
from .tmle import TMLE

__all__ = [
    "CTMLE",
    "CTMLELoss",
    "CTMLEOutcomeAdaptiveFit",
    "CTMLEPreorder",
    "CTMLESelection",
    "CTMLEStrategy",
]

CTMLEStrategy = Literal["greedy", "ordered", "discrete", "oat"]
CTMLELoss = Literal["auto", "loglik", "squared"]
CTMLEPreorder = Literal["logistic", "partial_correlation"]

#: Floor applied to targeted predictions before taking a logarithm in the loss.
_LOSS_EPS = 1e-12


def _binary_propensity(values: FloatArray) -> Propensity:
    """Wrap a candidate ``P(A = 1 | W)`` as the two-arm treatment mechanism.

    The selection works with the single propensity margin from end to end -- both
    searches order candidate covariates by how much each moves *that* margin -- so the
    per-arm matrix form is built here, at the boundary where a
    :class:`~cleverly.estimators._nuisance.NuisanceEstimates` is assembled, rather than
    threaded through the search.  :meth:`CTMLE._check_estimands` has already refused a
    treatment with more than two arms, which is what makes the complement well defined.
    """
    one = np.asarray(values, dtype=float).reshape(-1)
    return Propensity(np.column_stack([1.0 - one, one]), (0.0, 1.0))


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
        Non-increasing by construction for the greedy and ordered searches -- an
        increase means the search hit its numerical guard rather than finding a
        genuine improvement.
    cv_risk:
        The same criterion, cross-validated.  This is the quantity actually
        minimised, and unlike ``train_risk`` it is free to turn back up -- which is
        what stops the search from simply taking every covariate.
    selected:
        Index into ``path`` of the chosen candidate.
    """

    strategy: CTMLEStrategy
    preorder: str | None
    estimand: str
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
            f"target = {self.estimand}; "
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
                + " -- adjusting for these would have cost more variance than the bias "
                "they remove"
            )
        return "\n".join([*header, table, *footer])

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
    """Diagnostics for the ctmle3-style outcome-adaptive treatment model."""

    strategy: CTMLEStrategy
    treatment_features: tuple[str, ...]
    treatment_risk: float

    @property
    def treatment_risk_selected(self) -> float:
        """Treatment negative log likelihood, under the shared C-TMLE diagnostic API."""
        return self.treatment_risk

    def summary(self) -> str:
        features = ", ".join(self.treatment_features)
        return "\n".join(
            [
                "Collaborative TMLE outcome-adaptive fit",
                "=" * 39,
                "strategy = oat; treatment model = A ~ [Qbar(a, W)]",
                f"features: {features}",
                f"treatment negative log likelihood: {self.treatment_risk:.6g}",
            ]
        )


@dataclass(frozen=True)
class _Candidate:
    """One element of the candidate sequence: a propensity model and its targeted fit."""

    covariates: tuple[str, ...]
    propensity: FloatArray
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

    Parameters
    ----------
    strategy:
        ``"greedy"`` (default), ``"ordered"``, ``"discrete"`` or ``"oat"``.  The
        selector strategies are binary; ``"oat"`` fits treatment on the vector of all
        arm-specific outcome predictions and supports any number of discrete arms.
    ordering:
        Explicit covariate order for ``strategy="ordered"``.  When omitted, ``preorder``
        determines the published data-adaptive ordering.
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
        so unlike a plain TMLE, one fit cannot serve every estimand equally.  Must be
        one of the requested estimands.

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
        **kwargs: Any,
    ) -> None:
        if "search" in kwargs:
            raise TypeError("search= was replaced by strategy=; use CTMLE(strategy=...)")
        if kwargs.get("cv_evaluation", False):
            raise ValueError(
                "CTMLE does not support cv_evaluation=True: canonical CV-TMLE selection "
                "requires a separate fold-specific collaborative derivation."
            )
        super().__init__(**kwargs)
        self.strategy = strategy
        self.preorder = preorder
        self.ordering = ordering
        self.candidates = candidates
        self.selection_folds = selection_folds
        self.selection_inner_folds = selection_inner_folds
        self.loss = loss
        self.penalty = penalty
        self.ctmle_estimand = ctmle_estimand
        self._validate_ctmle_settings()
        if self.strategy == "ordered" and self.ordering is None and self.preorder is None:
            self.preorder = "logistic"

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
        if self.cv_evaluation:
            raise ValueError(
                "CTMLE does not support cv_evaluation=True: canonical CV-TMLE selection "
                "requires a separate fold-specific collaborative derivation."
            )
        if self.targeting_scheme != "pooled":
            raise ValueError(
                "CTMLE implements the published pooled collaborative estimator only; "
                "targeting_scheme='fold' composes it with a different CV-TMLE estimator "
                "that has not been derived. Use targeting_scheme='pooled'."
            )

        if self.strategy == "oat" and self.ctmle_estimand != "ate":
            raise ValueError(
                "ctmle_estimand= does not apply to strategy='oat': ctmle3's "
                "outcome-adaptive construction targets all treatment-specific means together"
            )
        if self.strategy == "oat" and (
            self.selection_folds != 5
            or self.selection_inner_folds != 2
            or self.loss != "auto"
            or not self.penalty
        ):
            raise ValueError(
                "selection_folds=, selection_inner_folds=, loss= and penalty= configure "
                "selector strategies and do not apply to strategy='oat'"
            )

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
        partition, which is the noise ``repeats=`` exists to average away.
        """
        if self.incremental:
            raise ValueError(
                "CTMLE and incremental= are not combined. C-TMLE cross-validates the "
                "*choice* of g against a loss for the targeted Qbar, and under an "
                "incremental intervention each candidate g defines a different "
                "parameter: Psi(delta) is built out of g. The search would be selecting "
                "between estimands rather than between estimators of one, and the risk "
                "it minimises would have no fixed target. Use a plain TMLE."
            )
        self._check_estimands(data)
        base = self._fit_nuisances(data, folds, scaler, intermediate_value, seed=seed)
        if self.strategy == "oat":
            return self._outcome_adaptive_nuisances(data, base, seed=seed)
        selector = _Selector(self, data, base, config.g_bounds, intermediate_value, seed=seed)

        path = selector.build_path(train=None, tag="full")
        cv_risk = selector.cross_validate(path)
        selected = int(np.argmin(cv_risk))
        chosen = path[selected]

        nuisance = replace(
            base,
            propensity=_binary_propensity(chosen.propensity),
            targeting_outcome=chosen.targeted,
            treatment_covariates=chosen.covariates,
        )
        selection = CTMLESelection(
            strategy=self.strategy,
            preorder=("custom" if self.ordering is not None else self.preorder)
            if self.strategy == "ordered"
            else None,
            estimand=self.ctmle_estimand,
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
            covariates=data.covariate_names,
        )
        return nuisance, {"ctmle": selection}

    def _outcome_adaptive_nuisances(
        self, data: CausalData, base: NuisanceEstimates, *, seed: int | None
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """Fit categorical ``A`` on the vector of arm-specific Qbar predictions.

        This is the construction in ``ctmle3::LF_oat``: no candidate path or
        parameter-specific risk is involved.  The ordinary K-column mean fluctuation
        targets the returned nuisances later in the shared TMLE pipeline.
        """
        arms = data.arm_codes
        design = np.column_stack([base.outcome.arms[arm] for arm in arms])
        learner = self._resolve_learner(self.treatment_learner, task="classification", seed=seed)
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
        propensity = Propensity(predictions["g"], arms)
        observed_columns = np.array(
            [propensity.column_for(float(arm)) for arm in data.treatment], dtype=int
        )
        observed_probability = propensity.values[np.arange(data.n), observed_columns]
        risk = float(-np.sum(data.weights * np.log(np.clip(observed_probability, _LOSS_EPS, 1.0))))
        features = tuple(f"Qbar[{data.arm_label(arm)}]" for arm in arms)
        nuisance_diagnostics = dict(base.diagnostics)
        nuisance_diagnostics.pop("propensity", None)
        if diagnostics:
            nuisance_diagnostics["propensity"] = diagnostics
        nuisance = replace(
            base,
            propensity=propensity,
            treatment_covariates=features,
            diagnostics=nuisance_diagnostics,
        )
        return nuisance, {
            "ctmle": CTMLEOutcomeAdaptiveFit(
                strategy="oat", treatment_features=features, treatment_risk=risk
            )
        }

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

    def _check_estimands(self, data: CausalData) -> None:
        if data.is_continuous_treatment:
            raise ValueError(
                "CTMLE strategies require a discrete treatment. strategy='oat' fits a "
                "categorical mechanism on one Qbar prediction per arm, and a continuous "
                "dose has no finite arm vector."
            )
        if data.has_intermediate:
            raise ValueError(
                "CTMLE does not compose either collaborative strategy with an intermediate "
                "outcome. "
                "Fit each controlled direct effect with TMLE instead."
            )
        if not data.is_binary_treatment and self.strategy != "oat":
            raise ValueError(
                f"CTMLE strategy={self.strategy!r} supports a binary treatment only; "
                f"{data.treatment_name} has "
                f"{data.n_arms} levels {list(data.treatment_levels)}. Both searches order "
                "candidates by how much a covariate moves a single propensity margin, and "
                "with more than two arms there is no one margin to order them by -- the "
                "selection would have to choose a model for each arm and score them jointly, "
                "which is a different algorithm rather than a wider loop. Use a plain TMLE."
            )
        estimands = resolve_estimands(self.estimands, data.family, data.n_arms)
        conditional = [name for name in estimands if name not in MEAN_GROUP_ESTIMANDS]
        if conditional:
            raise ValueError(
                f"CTMLE does not support estimand(s) {conditional}: the ATT and ATC clever "
                "covariates condition on a random event, so a single collaboratively "
                "selected treatment model cannot serve them alongside the ATE. Request them "
                f"from a plain TMLE, or set estimands={sorted(MEAN_GROUP_ESTIMANDS)!r}."
            )
        if self.strategy != "oat" and self.ctmle_estimand not in estimands:
            raise ValueError(
                f"ctmle_estimand={self.ctmle_estimand!r} is not among the requested estimands "
                f"{list(estimands)}; the selection has to be made for an estimand you are "
                "actually reporting."
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

    # ------------------------------------------------------------ propensities

    def propensity(
        self, covariates: tuple[str, ...], train: IntArray | None, tag: str
    ) -> FloatArray:
        """``g(W_S)`` for one candidate covariate set, cached per search branch."""
        key = (tag, covariates)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        values = self._fit_propensity(covariates, train)
        self._cache[key] = values
        return values

    def _fit_propensity(self, covariates: tuple[str, ...], train: IntArray | None) -> FloatArray:
        return self._fit_propensity_with(self.learner, covariates, train)

    def _fit_propensity_with(
        self, learner: Learner, covariates: tuple[str, ...], train: IntArray | None
    ) -> FloatArray:
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
                n_jobs=self.est.n_jobs,
            )
            return predictions["g1"]
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
                n_jobs=self.est.n_jobs,
            )
            return predictions["g1"]

        model = fit_on_rows(
            learner,
            design,
            data.treatment,
            data.weights,
            train,
            "classification",
            data.cluster,
        )
        return np.clip(predict_mean(model, design, "classification"), 0.0, 1.0)

    def _intercept_propensity(self, train: IntArray | None) -> FloatArray:
        """``P(A = 1)`` with no covariates -- the first candidate on every path.

        Fit by hand rather than by handing a zero-column design to a learner, which
        scikit-learn rejects.  Cross-fitted like any other candidate so it is scored
        on the same footing.
        """
        data = self.data
        values = np.empty(data.n)
        if self.train_folds is not None:
            assert self.train_mask is not None
            for fit_rows, test in self.train_folds:
                eligible = fit_rows[self.train_mask[fit_rows]]
                values[test] = np.average(data.treatment[eligible], weights=data.weights[eligible])
            return values
        if train is not None:
            values[:] = np.average(data.treatment[train], weights=data.weights[train])
            return values
        if self.base.folds.is_single:
            values[:] = np.average(data.treatment, weights=data.weights)
            return values
        for fit_rows, test in self.base.folds:
            values[test] = np.average(data.treatment[fit_rows], weights=data.weights[fit_rows])
        return values

    # ---------------------------------------------------------------- targeting

    def submodel(self, propensity: FloatArray) -> Submodel:
        """The ``mean`` clever covariate at a candidate propensity."""
        nuisance = replace(self.base, propensity=_binary_propensity(propensity))
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
        """The variance/bias term: ``Var(D*) + n * mean(D*)^2`` on ``rows``.

        On the scaled outcome, so it is commensurate with :meth:`loss`.
        """
        return _penalty_of(self.influence(targeted, submodel, rows))

    def influence(self, targeted: InitialFit, submodel: Submodel, rows: IntArray) -> FloatArray:
        """The target estimand's efficient influence curve, on the scaled outcome."""
        # Two arms throughout -- CTMLE._check_estimands has refused anything else.
        means = counterfactual_means(
            self.scaled[rows],
            _restrict_fit(targeted, rows),
            restrict(submodel, rows),
            self.data.weights[rows],
            self.data.observed[rows],
        )
        one, zero = means[1.0], means[0.0]
        estimand = self.est.ctmle_estimand
        if estimand == "ate":
            return np.asarray(one.influence_curve - zero.influence_curve, dtype=float)
        if estimand == "ey1":
            return np.asarray(one.influence_curve, dtype=float)
        if estimand == "ey0":
            return np.asarray(zero.influence_curve, dtype=float)
        ratios = ratio_estimates(
            one.psi,
            one.influence_curve,
            zero.psi,
            zero.influence_curve,
            n=rows.size,
            which=(estimand,),
        )
        return ratios[estimand].influence_curve

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

        The two searches differ only in how the next covariate is picked; the TMLE-step
        incrementing that keeps the loss monotone is shared.
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
                # search again from the targeted fit -- this is what makes the risk
                # along the sequence monotone (van der Laan & Gruber, 2010).
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
        g = np.clip(propensity[rows], _LOSS_EPS, 1.0 - _LOSS_EPS)
        a = self.data.treatment[rows]
        w = self.data.weights[rows]
        treatment_risk = float(-np.sum(w * (a * np.log(g) + (1.0 - a) * np.log(1.0 - g))))
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
            logistic = resolve_learner(
                "glm",
                task="classification",
                n_folds=self.est.learner_folds,
                random_state=self.seed,
            )
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
            weights = self.data.weights[usable]
            scores = np.array(
                [
                    abs(
                        _weighted_partial_correlation(
                            residual,
                            self.data.covariates[usable, column],
                            treatment,
                            weights,
                        )
                    )
                    for column in range(self.data.covariates.shape[1])
                ]
            )
            order = np.argsort(-scores, kind="stable")
        return tuple(self.data.covariate_names[j] for j in order)

    # -------------------------------------------------------------- selection

    def cross_validate(self, path: Sequence[_Candidate]) -> FloatArray:
        """Cross-validated risk of each position in the candidate sequence.

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
        """
        data = self.data
        folds = make_folds(
            data.n,
            self.est.selection_folds,
            # The same stratum the outer folds use, rather than data.treatment again: a
            # selection fold with no events makes both the loss and the per-fold
            # influence curve below degenerate, which is the failure stratify_folds
            # exists to prevent, and it would be odd for the option to protect one split
            # and not the other.
            stratify=self.est._fold_strata(data),
            cluster=data.cluster,
            random_state=self.seed,
        )
        loss = np.zeros(len(path))
        influence = np.zeros((len(path), data.n))
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
                influence[index, test] = fold_selector.influence(
                    candidate.targeted, candidate.submodel, test
                )
        if not self.est.penalty:
            return loss
        return loss + np.array([_penalty_of(row) for row in influence])

    def _nested_folds(self, train: IntArray) -> tuple[Folds, BoolArray]:
        """Inner cross-fit on ``train`` plus one full-training fit for validation rows."""
        data = self.data
        train_data = data.subset(train)
        inner = make_folds(
            train.size,
            self.est.selection_inner_folds,
            stratify=self.est._fold_strata(train_data),
            cluster=train_data.cluster,
            random_state=self.seed,
        )
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


def _penalty_of(influence_curve: FloatArray) -> float:
    """``Var(D*) + n * mean(D*)^2`` -- the variance/bias penalty of a candidate."""
    if influence_curve.size < 2:
        return 0.0
    return float(
        np.var(influence_curve, ddof=1) + influence_curve.size * np.mean(influence_curve) ** 2
    )


def _weighted_partial_correlation(
    left: FloatArray, right: FloatArray, conditional: FloatArray, weights: FloatArray
) -> float:
    """Weighted correlation of residuals after projecting both variables on ``A``."""
    design = np.column_stack([np.ones(left.size), conditional])
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
