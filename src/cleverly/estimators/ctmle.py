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

``search="greedy"`` (default)
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

``search="ordered"``
    Scalable C-TMLE (Ju et al., 2019).  The sequence is fixed in advance by an
    ordering, so only ``O(V p)`` fits are needed.  This is what makes C-TMLE usable
    when ``p`` is large.  The default ordering is by decreasing marginal association
    with the *outcome*, which puts confounders ahead of instruments; pass
    ``ordering=`` to supply your own.

``search="discrete"``
    Cross-validated selection among an explicit list of candidate covariate sets --
    the analogue of ``ctmleDiscrete`` / ``ctmleGlmnet``.

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

Selection chooses a propensity model; everything after that is an ordinary TMLE
against it, run through :meth:`~cleverly.TMLE.retarget`.  So the reported estimate,
its influence curve, and every sensitivity and validation diagnostic are the same
code paths a plain fit uses, and they are all consistent with each other.

Two consequences are worth stating plainly.

The TMLE-step incrementing inside the greedy search shapes the candidate sequence and
its risks, but the final reported estimator is the TMLE at the selected ``g`` rather
than the multi-step targeted fit the search happened to end on.  The two are
asymptotically equivalent -- and the iterative targeting step already fluctuates
repeatedly until the score is solved -- but they are not identical in finite samples.
``result.extra["ctmle"].n_steps`` records how many steps the search used.

The influence-curve standard error conditions on the selected propensity model.  It
does not include the variability the *selection* contributes, and so runs mildly
anti-conservative -- in simulation on the instrument process below, a reported
standard error about 12% under the true spread of the estimates.  That is a smaller
error than the variance C-TMLE saves, so the interval is still narrower and its
coverage no worse than a plain TMLE's; but where the selection matters and honest
inference is the point, pass ``n_bootstrap=``.  Each replicate re-runs the search, so
the bootstrap standard error does see it.

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
... )
>>> res.extra["ctmle"].selected_covariates            # doctest: +SKIP
('W1',)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .._typing import FloatArray, IntArray, Learner
from ..data.causal_data import CausalData
from ..fluctuation.iterative import InitialFit, _apply_logistic
from ..fluctuation.submodel import Submodel, restrict, weighted_form
from ..inference.influence import counterfactual_means, ratio_estimates
from ..learners._fitting import predict_mean
from ..learners.crossfit import Folds, make_folds
from ..learners.screeners import correlation_strength
from ..utils.bounds import OutcomeScaler
from ._nuisance import NuisanceEstimates, _fit_with_groups, cross_fit_predictions
from .base import MEAN_GROUP_ESTIMANDS, TMLEConfig, format_table, resolve_estimands
from .tmle import TMLE

__all__ = ["CTMLE", "CTMLELoss", "CTMLESearch", "CTMLESelection"]

CTMLESearch = Literal["greedy", "ordered", "discrete"]
CTMLELoss = Literal["auto", "loglik", "squared"]

#: Floor applied to targeted predictions before taking a logarithm in the loss.
_LOSS_EPS = 1e-12


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

    search: CTMLESearch
    estimand: str
    loss: str
    penalized: bool
    path: tuple[tuple[str, ...], ...]
    n_steps: tuple[int, ...]
    train_risk: FloatArray
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
            f"search = {self.search}; target = {self.estimand}; "
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
    risk: float


class CTMLE(TMLE):
    """Collaborative TMLE for a binary point treatment.

    Selects the covariates entering ``g(W)`` by cross-validating the loss of the
    *targeted* outcome model, rather than the loss of ``g`` itself.  See the module
    docstring for the algorithm and the loss.

    Every :class:`~cleverly.TMLE` keyword is accepted and behaves identically; the
    selection happens in place of the ordinary propensity fit.  The result is an
    ordinary :class:`~cleverly.estimators.TMLEResult` with the selection recorded
    under ``result.extra["ctmle"]``.

    Parameters
    ----------
    search:
        ``"greedy"`` (default), ``"ordered"`` or ``"discrete"``; see the module
        docstring.
    ordering:
        Covariate order for ``search="ordered"``.  Defaults to decreasing marginal
        association with the outcome, which is the ordering that puts confounders
        ahead of instruments.
    candidates:
        Explicit candidate covariate sets for ``search="discrete"``.
    selection_folds:
        Folds used to cross-validate the candidate sequence.  Separate from
        ``n_folds``, which cross-fits the nuisance models.
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
        search: CTMLESearch = "greedy",
        ordering: Sequence[str] | None = None,
        candidates: Sequence[Sequence[str]] | None = None,
        selection_folds: int = 5,
        loss: CTMLELoss = "auto",
        penalty: bool = True,
        ctmle_estimand: str = "ate",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.search = search
        self.ordering = ordering
        self.candidates = candidates
        self.selection_folds = selection_folds
        self.loss = loss
        self.penalty = penalty
        self.ctmle_estimand = ctmle_estimand
        self._validate_ctmle_settings()

    def _validate_ctmle_settings(self) -> None:
        if self.loss not in ("auto", "loglik", "squared"):
            raise ValueError(f"loss must be 'auto', 'loglik' or 'squared'; got {self.loss!r}")
        if self.search not in ("greedy", "ordered", "discrete"):
            raise ValueError(
                f"search must be 'greedy', 'ordered' or 'discrete'; got {self.search!r}"
            )
        if self.search == "discrete" and not self.candidates:
            raise ValueError("search='discrete' needs an explicit candidates= list")
        if self.search != "discrete" and self.candidates is not None:
            raise ValueError(f"candidates= only applies to search='discrete', not {self.search!r}")
        if self.search != "ordered" and self.ordering is not None:
            raise ValueError(f"ordering= only applies to search='ordered', not {self.search!r}")
        if self.selection_folds < 2:
            raise ValueError(f"selection_folds must be at least 2; got {self.selection_folds}")
        if self.ctmle_estimand not in MEAN_GROUP_ESTIMANDS:
            raise ValueError(
                f"ctmle_estimand must be one of {sorted(MEAN_GROUP_ESTIMANDS)}; "
                f"got {self.ctmle_estimand!r}"
            )

    # --------------------------------------------------------------- the hook

    def _nuisances(
        self,
        data: CausalData,
        folds: Folds,
        scaler: OutcomeScaler,
        config: TMLEConfig,
        intermediate_value: float | None,
    ) -> tuple[NuisanceEstimates, dict[str, Any]]:
        """Fit the outcome model once, then *select* the propensity model against it."""
        self._check_estimands(data)
        base = self._fit_nuisances(data, folds, scaler, intermediate_value)
        selector = _Selector(self, data, base, config.g_bounds, intermediate_value)

        path = selector.build_path(train=None, tag="full")
        cv_risk = selector.cross_validate(path)
        selected = int(np.argmin(cv_risk))
        chosen = path[selected]

        nuisance = replace(
            base,
            propensity=chosen.propensity,
            treatment_covariates=chosen.covariates,
        )
        selection = CTMLESelection(
            search=self.search,
            estimand=self.ctmle_estimand,
            loss=selector.loss_kind,
            penalized=self.penalty,
            path=tuple(candidate.covariates for candidate in path),
            n_steps=tuple(candidate.n_steps for candidate in path),
            train_risk=np.array([candidate.risk for candidate in path], dtype=float),
            cv_risk=cv_risk,
            selected=selected,
            covariates=data.covariate_names,
        )
        return nuisance, {"ctmle": selection}

    def _check_estimands(self, data: CausalData) -> None:
        estimands = resolve_estimands(self.estimands, data.family)
        conditional = [name for name in estimands if name not in MEAN_GROUP_ESTIMANDS]
        if conditional:
            raise ValueError(
                f"CTMLE does not support estimand(s) {conditional}: the ATT and ATC clever "
                "covariates condition on a random event, so a single collaboratively "
                "selected treatment model cannot serve them alongside the ATE. Request them "
                f"from a plain TMLE, or set estimands={sorted(MEAN_GROUP_ESTIMANDS)!r}."
            )
        if self.ctmle_estimand not in estimands:
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
    ) -> None:
        self.est = estimator
        self.data = data
        self.base = base
        self.bounds = bounds
        self.intermediate_value = intermediate_value
        self.scaled = base.scaler.scale(data.outcome)
        self.all_rows: IntArray = np.arange(data.n)
        self.loss_kind = (
            estimator.loss
            if estimator.loss != "auto"
            else ("loglik" if data.family == "binomial" else "squared")
        )
        self.learner: Learner = estimator._resolve_learner(
            estimator.treatment_learner, task="classification"
        )
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
        data = self.data
        if not covariates:
            return self._intercept_propensity(train)

        columns = [data.covariate_names.index(name) for name in covariates]
        design = np.ascontiguousarray(data.covariates[:, columns])
        if train is None:
            predictions, _ = cross_fit_predictions(
                self.learner,
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

        model = _fit_with_groups(
            self.learner,
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
        nuisance = replace(self.base, propensity=propensity)
        return self.est._submodel(
            self.data, nuisance, "mean", self.bounds, self.intermediate_value, None
        )

    def target(
        self, initial: InitialFit, submodel: Submodel, rows: IntArray
    ) -> tuple[InitialFit, FloatArray]:
        """Solve the fluctuation on ``rows``, then apply it to the whole sample.

        Fitting and applying are separated because the cross-validated selector needs
        an ``epsilon`` fit on training rows and evaluated on held-out ones.
        """
        fluctuation = self.est._solve_rows(
            self.scaled[rows],
            _restrict_fit(initial, rows),
            restrict(submodel, rows),
            self.data.weights[rows],
            self.data.observed[rows],
            warn=False,
        )
        return self.apply(initial, submodel, fluctuation.epsilon), fluctuation.epsilon

    def apply(self, initial: InitialFit, submodel: Submodel, epsilon: FloatArray) -> InitialFit:
        """Move the predictions along the submodel by a fitted ``epsilon``."""
        est = self.est
        moved = weighted_form(submodel, self.data.weights)[0] if est.target_weights else submodel
        if est.fluctuation == "linear":
            return InitialFit(
                initial.observed + moved.observed @ epsilon,
                initial.at_one + moved.at_one @ epsilon,
                initial.at_zero + moved.at_zero @ epsilon,
            )
        return _apply_logistic(initial.shrunk(est.alpha), moved, epsilon, est.alpha)

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
        psi_one, ic_one, psi_zero, ic_zero = counterfactual_means(
            self.scaled[rows],
            _restrict_fit(targeted, rows),
            restrict(submodel, rows),
            self.data.weights[rows],
            self.data.observed[rows],
        )
        estimand = self.est.ctmle_estimand
        if estimand == "ate":
            return np.asarray(ic_one - ic_zero, dtype=float)
        if estimand == "ey1":
            return np.asarray(ic_one, dtype=float)
        if estimand == "ey0":
            return np.asarray(ic_zero, dtype=float)
        ratios = ratio_estimates(psi_one, ic_one, psi_zero, ic_zero, n=rows.size, which=(estimand,))
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
        if self.est.search == "discrete":
            return self._discrete_path(rows, train, tag)
        order = self._ordering() if self.est.search == "ordered" else None
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
        return _Candidate(
            covariates=covariates,
            propensity=propensity,
            submodel=submodel,
            targeted=targeted,
            epsilon=epsilon,
            n_steps=n_steps,
            loss=loss,
            risk=loss + penalty,
        )

    def _ordering(self) -> tuple[str, ...]:
        """Covariate order for the scalable search.

        The default ranks by marginal association with the outcome.  A confounder is
        associated with the outcome; an instrument is not, so it sinks to the end of
        the queue where the cross-validated risk has already turned back up.
        """
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
                    "search='discrete' to search a restricted set of models."
                )
            return names

        observed = self.data.observed
        strength = np.abs(
            correlation_strength(
                self.data.covariates[observed],
                self.data.outcome[observed],
                sample_weight=self.data.weights[observed],
            )
        )
        order = np.argsort(-strength, kind="stable")
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
            stratify=data.treatment,
            cluster=data.cluster,
            random_state=self.est.random_state,
        )
        loss = np.zeros(len(path))
        influence = np.zeros((len(path), data.n))
        for fold, (train, test) in enumerate(folds):
            fold_path = self.build_path(train=train, tag=f"cv{fold}")
            if len(fold_path) < len(path):
                raise RuntimeError(
                    f"selection fold {fold} produced {len(fold_path)} candidates but the "
                    f"full-sample search produced {len(path)}; the candidate sequence must "
                    "have the same length in every fold"
                )
            for index in range(len(path)):
                candidate = fold_path[index]
                loss[index] += self.loss(candidate.targeted, test)
                influence[index, test] = self.influence(
                    candidate.targeted, candidate.submodel, test
                )
        if not self.est.penalty:
            return loss
        return loss + np.array([_penalty_of(row) for row in influence])


def _penalty_of(influence_curve: FloatArray) -> float:
    """``Var(D*) + n * mean(D*)^2`` -- the variance/bias penalty of a candidate."""
    if influence_curve.size < 2:
        return 0.0
    return float(
        np.var(influence_curve, ddof=1) + influence_curve.size * np.mean(influence_curve) ** 2
    )


def _restrict_fit(fit: InitialFit, index: IntArray) -> InitialFit:
    """Row-subset an initial fit, the counterpart of :func:`.submodel.restrict`."""
    return InitialFit(fit.observed[index], fit.at_one[index], fit.at_zero[index])
