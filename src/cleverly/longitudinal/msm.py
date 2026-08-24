r"""A working model over regimens, and the projection that defines its coefficients.

A fit declaring many regimens reports a mean per plan, which is a table rather than an
answer -- the same complaint :mod:`cleverly.msm` exists to answer at one time point, and
worse here because the plans number :math:`2^T`.  ``msm=`` declares a working model
:math:`m(\bar a, V; \beta)` summarising them, and makes the fit's parameters its
coefficients.

:math:`\beta` is a **projection**, not a modelling assumption.  It minimises

.. math::

    E\Big[ \sum_c h(c, V)\,\big( E[Y^{c} \mid V] - m(c, V; \beta) \big)^2 \Big]

over a *known* weight function :math:`h`, so it is well defined whether or not the
working model is right: "the best :math:`m`-shaped summary of the counterfactual means,
in the :math:`h`-weighted least-squares sense" (Neugebauer & van der Laan 2007, and
Orellana, Rotnitzky & Robins 2010 for the regimen-indexed case).  Where the model happens
to be right, :math:`\beta` is the truth; where it is wrong, the interval is still an
honest interval for the projection, which is the thing that was estimated.

**What a cell is.**  :math:`c` ranges over the *cells* of the parameter grid: a declared
regimen on an end-of-study fit, and a ``(regimen, horizon)`` pair on a survival one.  The
horizon is inside the design rather than beside it, so a coefficient can be a trend across
horizons and the whole grid shares one covariance.  A design saturated in the horizon
reproduces the per-horizon coefficients exactly, so this contains the per-horizon report
rather than replacing it.  A *cause*, by contrast, is a different estimand and not a
further column: each cause gets its own projection, sharing every nuisance fit exactly as
the per-regimen recursion already shares them.

**Three things differ from the working model at one time point**, and each is a place the
obvious generalisation is wrong.

*The node fluctuation is pooled across the cells.*  At one time point the design's
:math:`p` columns get their rank from summing over the arms *within one row*: a unit
contributes :math:`\varphi(a, V)` at the arm it received.  Here there is nothing to sum
over within a row -- a regimen is a plan, not a value some unit took -- and whenever the
working model has no effect modifier :math:`\varphi(c, V)` is *constant down the rows*.
A cell's :math:`p` score equations therefore collapse to one.  Each node instead solves
a single fluctuation over the cells stacked, :math:`C \cdot n` rows and one shared
:math:`\epsilon`.  The submodel design is :math:`(dm/d\eta)\varphi(c,V)` and the loss
weight is :math:`h(c,V)h_t^c`, where :math:`h_t^c` is the cumulative inverse treatment
and censoring probability.  Their product is the EIF numerator.  Under a saturated
model the stacked design is exactly block-diagonal and its loss weights are those of the
per-regimen recursion, which is why the report reduces to that one.

*The recursion is therefore lockstep* -- outer loop over the nodes, inner loop over the
cells for the regressions, then one pooled fluctuation, then every cell carried forward
together.  The regressions themselves are unchanged; only where the update happens moved.

*Under a link, one alternation round is a whole backward pass.*  :math:`\beta` enters the
covariate through :math:`dm/d\eta`, so :math:`\bar Q^*_t` moves with it -- and
:math:`\bar Q^*_t` is node :math:`t-1`'s *regression target*, so every earlier node's
learner has to be refit.  There is no fixed :math:`\bar Q^0` here to restart from; the
fixed point is stated over the whole pass instead.  The mechanism is :math:`\beta`-free
and is not refit.

**The projection is solved on the raw outcome scale**, as it is at one time point and for
the same reason: a coefficient vector has no single scale to map back with.  So the curve
carries ``scaler.range`` on its residual half and ``lower + range * Q̄*`` on its plug-in
half, and the estimates never go through the unscaling the per-regimen report does.

**:math:`h(\bar a, V)` and the observation weights are different objects.**  The first
says how the cells are traded off against each other; the second tilts the population
the projection is taken over.  Both multiply the node's loss weight, while only the
observation weight multiplies the finished curve row-wise.  The distinction remains in
the estimand even though multiplication puts them in the same numerical array.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import FloatArray, Learner
from ..estimators.targeting import ProjectionFluctuation
from ..exceptions import DataError
from ..fluctuation.iterative import Fluctuation, InitialFit, solve_fluctuation
from ..fluctuation.submodel import Submodel
from ..learners.crossfit import Folds
from ..msm import MSM, Link, ProjectionFit, check_projection_rank, link_for, solve_projection
from ..utils.bounds import OutcomeScaler
from .data import LongitudinalData
from .regimen import Plan
from .sequential import (
    _FILLER,
    _REGIMEN_ARM,
    Mechanism,
    RegimenFit,
    SequentialStep,
    prepare_node,
    seed_carried,
)

__all__ = [
    "Cell",
    "MSMRegimenFit",
    "RegimenMSM",
    "evaluate_regimen_msm",
    "fit_regimens_msm",
]


#: How much slower a round of the alternation may fall than the last before it is called
#: stalled.  The same threshold as
#: :data:`cleverly.estimators.targeting._STALL_FACTOR`, and read the same way: a
#: contraction that has stopped contracting is not going to start.
_STALL_FACTOR = 0.95

#: A relative shift in ``beta`` above this at exit is reported as a failure rather than as
#: convergence.  Beside :data:`_STALL_FACTOR`.
_UNSOLVED = 1e-6


@dataclass(frozen=True)
class Cell:
    """One column of the parameter grid: a regimen, and the horizon it is reported at.

    On an end-of-study fit the horizon is ``T`` for every cell and the grid is the
    regimens.  A cell is *not* a cause: causes are separate estimands with separate
    projections, and they share every nuisance fit rather than a coefficient vector.
    """

    label: str
    horizon: int


@dataclass(frozen=True)
class RegimenMSM:
    r"""A working model over regimens, evaluated on the data.

    Arrays only, no callables -- the rule :class:`cleverly.msm.MSMSet` follows, and for
    the first of its two reasons: the design is evaluated once, where the fit begins, so
    that every round of an alternation and every quantity read off the result agree on
    what the working model is by construction rather than by re-running the user's
    function and hoping it is deterministic.

    Deliberately **not** :class:`cleverly.msm.MSMSet`.  That class's second axis is the
    treatment arms and says so in its field name, its docstring and its accessors; this
    one's is the cells of a regimen grid, keyed by label and horizon rather than by a
    float code, and its constructor reads a :class:`~cleverly.data.CausalData` throughout.
    What the two share is the rank rule, which lives in
    :func:`cleverly.msm.check_projection_rank` so that it cannot drift.

    Attributes
    ----------
    terms:
        One name per coefficient, in column order.
    design:
        ``(n, C, p)``: ``design[i, k, :]`` is :math:`\varphi(c_k, V_i)`.
    weights:
        ``(n, C)``: ``weights[i, k]`` is :math:`h(c_k, V_i)` -- the *working model's*
        weight, never the observation weights.
    cells:
        What the second axis is keyed by, in its order.
    link:
        The declared link, by name.  A string rather than a :class:`cleverly.msm.Link`,
        keeping this object arrays-and-scalars as its point-treatment sibling is.
    """

    terms: tuple[str, ...]
    design: FloatArray
    weights: FloatArray
    cells: tuple[Cell, ...]
    link: str = "identity"

    def __post_init__(self) -> None:
        link_for(str(self.link))
        design = np.asarray(self.design, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if design.ndim != 3 or design.shape[1] != len(self.cells):
            raise DataError(
                f"the working model's design must have shape (n, {len(self.cells)}, p) -- "
                f"rows, regimen-horizon cells, terms -- got {design.shape}"
            )
        if design.shape[2] != len(self.terms):
            raise DataError(
                f"the design has {design.shape[2]} column(s) but {len(self.terms)} term "
                f"name(s) {list(self.terms)}"
            )
        if weights.shape != design.shape[:2]:
            raise DataError(
                f"the working model's weights must have shape {design.shape[:2]}; "
                f"got {weights.shape}"
            )
        if not np.all(np.isfinite(design)):
            raise DataError("the working model's design contains a non-finite value")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise DataError(
                "the working model's weights must be finite and non-negative; h(a-bar, V) "
                "is a weight in a least-squares projection, not a signed contrast"
            )
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "weights", weights)
        check_projection_rank(self.gram, self.terms, axis="regimens")

    # ------------------------------------------------------------------ access

    @property
    def n(self) -> int:
        return int(self.design.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.design.shape[1])

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def labels(self) -> tuple[str, ...]:
        """The regimen label of each cell, in column order."""
        return tuple(cell.label for cell in self.cells)

    @property
    def gram(self) -> FloatArray:
        r"""``(p, p)`` weighted Gram matrix :math:`P_n \sum_c h\,\varphi\varphi^\top`.

        Here for the rank check only, exactly as :attr:`cleverly.msm.MSMSet.gram` is: the
        matrix the estimate inverts is built inside :func:`cleverly.msm.solve_projection`
        and carries the observation weights and, under a link, a curvature term.
        """
        return np.einsum(
            "ikp,ikq,ik->pq", self.design, self.design, self.weights, optimize=True
        ) / max(self.n, 1)

    def weighted_design_at(self, beta: FloatArray | None) -> FloatArray:
        r""":math:`h\,(dm/d\eta)\,\varphi`, the ``(n, C, p)`` numerator of the covariate.

        Under the identity link ``dm/deta`` is one, ``beta`` is ignored and this is the
        array it was before links existed -- which is what keeps an identity-link fit a
        single pass rather than an alternation.
        """
        spec = link_for(str(self.link))
        if spec.is_identity:
            return self.design * self.weights[:, :, None]
        if beta is None:
            raise ValueError(
                f"link={self.link!r} makes the clever covariate a function of beta, so it "
                "cannot be built before one is available"
            )
        slope = np.asarray(spec.slope(self.fitted(beta)), dtype=float)
        return self.design * (self.weights * slope)[:, :, None]

    def fluctuation_design_at(self, beta: FloatArray | None) -> FloatArray:
        r""":math:`(dm/d\eta)\,\varphi`, the loss-weighted fluctuation's design.

        The projection weight :math:`h` and cumulative inverse probability live in the
        loss weight, following ``ltmle::UpdateQ``.  :meth:`weighted_design_at` retains
        them in the EIF numerator; keeping the two arrays named separately prevents the
        algebraically equivalent score from being mistaken for the same submodel path.
        """
        spec = link_for(str(self.link))
        if spec.is_identity:
            return self.design
        if beta is None:
            raise ValueError(
                f"link={self.link!r} makes the fluctuation design a function of beta, "
                "so it cannot be built before one is available"
            )
        slope = np.asarray(spec.slope(self.fitted(beta)), dtype=float)
        return self.design * slope[:, :, None]

    def fitted(self, beta: FloatArray) -> FloatArray:
        """``(n, C)`` fitted means :math:`m(c, V; \\beta)`."""
        spec = link_for(str(self.link))
        return np.asarray(spec.inverse(np.einsum("ikp,p->ik", self.design, beta)), dtype=float)


# ------------------------------------------------------------------ evaluation


def evaluate_regimen_msm(
    msm: MSM,
    data: LongitudinalData,
    plans: Sequence[Plan],
    horizons: Sequence[int],
) -> RegimenMSM:
    """Evaluate ``msm``'s design and weights at every ``(regimen, horizon)`` cell.

    The design is handed the regimen's **label** -- the caller's own key, not an internal
    code -- the horizon, and the baseline covariates in the backend the data arrived in.
    That the frame stops at the baseline is the estimand's own statement rather than a
    convenience; :meth:`~cleverly.longitudinal.LongitudinalData.baseline_frame` says why.
    """
    if msm.from_linear:
        raise DataError(
            "MSM.linear reads the label it is handed as a dose to interpolate between, "
            "which a treatment arm can be and a regimen cannot: a plan is a sequence of "
            "decisions, and no arithmetic on its name is a summary of it. Pass design= "
            "and code the grid yourself -- design=lambda label, horizon, w: "
            "np.column_stack([np.ones(len(w)), np.full(len(w), duration[label])]) -- so "
            "that what the coefficient is per unit of is written down."
        )
    _check_support(link_for(str(msm.link)), data)
    frame = data.baseline_frame()
    cells = tuple(Cell(plan.label, int(horizon)) for horizon in horizons for plan in plans)
    designs = [_cell_design(msm, cell, frame, data.n) for cell in cells]
    weights = [_cell_weights(msm, cell, frame, data.n) for cell in cells]
    return RegimenMSM(
        msm.terms,
        np.stack(designs, axis=1),
        np.column_stack(weights),
        cells,
        msm.link,
    )


def _call(function: Any, cell: Cell, frame: Any, *, what: str) -> Any:
    """Call a user design or weight function, naming the signature if it has the old one."""
    try:
        return function(cell.label, cell.horizon, frame)
    except TypeError as error:
        shape = "(n, p)" if what == "design" else "(n,)"
        raise DataError(
            f"a working model's {what} over regimens takes three arguments -- "
            f"(regimen_label, horizon, baseline_frame) -> {shape}"
            " -- rather than the (arm_label, covariate_frame) pair a working model over "
            "arms takes. The horizon is inside the design so that a coefficient can be a "
            "trend across horizons; on an end-of-study fit it is the last node and may be "
            f"ignored. Called for regimen {cell.label!r} at horizon {cell.horizon}."
        ) from error


def _cell_design(msm: MSM, cell: Cell, frame: Any, n: int) -> FloatArray:
    values = np.asarray(_as_float(_call(msm.design, cell, frame, what="design")), dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] != n:
        raise DataError(
            f"the working model's design returned {values.shape} for regimen "
            f"{cell.label!r} at horizon {cell.horizon}; it must return (n, p) = "
            f"({n}, {len(msm.terms)})"
        )
    if values.shape[1] != len(msm.terms):
        raise DataError(
            f"the working model's design returned {values.shape[1]} column(s) for regimen "
            f"{cell.label!r} at horizon {cell.horizon} but names {len(msm.terms)} term(s) "
            f"{list(msm.terms)}"
        )
    return values


def _cell_weights(msm: MSM, cell: Cell, frame: Any, n: int) -> FloatArray:
    if msm.weights is None:
        return np.ones(n)
    values = np.asarray(
        _as_float(_call(msm.weights, cell, frame, what="weights")), dtype=float
    ).reshape(-1)
    if values.shape[0] != n:
        raise DataError(
            f"the working model's weights returned {values.shape[0]} value(s) for regimen "
            f"{cell.label!r} at horizon {cell.horizon}; it must return one per row ({n})"
        )
    return values


def _as_float(values: Any) -> Any:
    """Whatever backend's column or frame a user function returned, as an array."""
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _check_support(link: Link, data: LongitudinalData) -> None:
    """Refuse a link whose mean function cannot reach the outcome being modelled.

    Checked against the observed outcome rather than against the predictions, and against
    the *event* indicators on a survival fit, where the parameter is a cumulative
    incidence and so lives in ``[0, 1]`` whatever the link.  The reasoning is
    :func:`cleverly.msm._check_support`'s.
    """
    if link.support is None or data.is_survival:
        return
    observed = np.asarray(data.outcome, dtype=float)[
        np.asarray(data.uncensored_through(data.n_times), dtype=bool)
    ]
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:  # pragma: no cover - an all-missing outcome fails much earlier
        return
    low, high = float(np.min(observed)), float(np.max(observed))
    if link.support == "nonnegative" and low < 0.0:
        raise DataError(
            f"link={link.name!r} models the counterfactual mean as exp(beta' phi), which "
            f"is positive, and {data.outcome_name} takes the value {low:g}. A log-link "
            "working model needs a non-negative outcome -- a risk, a count or a rate. Use "
            "link='identity' for an outcome that can be negative."
        )
    if link.support == "unit" and (low < 0.0 or high > 1.0):
        raise DataError(
            f"link='logit' models the counterfactual mean as a probability, and "
            f"{data.outcome_name} ranges over [{low:g}, {high:g}]. A logit working model "
            "needs an outcome in [0, 1] -- a binary outcome or a proportion. Note that "
            "the projection is solved on the outcome's own scale, so that its "
            "coefficients are reported in the units they were declared in."
        )


# ------------------------------------------------------------------ the fit


@dataclass(frozen=True)
class MSMRegimenFit:
    """One cause's working-model coefficients, and everything they were built from.

    ``beta`` and ``influence_curves`` are on the **raw** outcome scale already, unlike
    :attr:`~cleverly.longitudinal.RegimenFit.psi_scaled`: a coefficient vector has no
    single scale to map back with, so the projection is solved unscaled and the estimate
    layer above must not unscale it a second time.
    """

    model: RegimenMSM
    cause: str | None
    beta: FloatArray
    influence_curves: FloatArray
    projection: ProjectionFit
    alternation: ProjectionFluctuation
    #: Per cell, keyed as :attr:`RegimenMSM.cells` is, so that ``diagnostics()`` can
    #: report the leverage and the risk set a regimen actually had.
    fits: tuple[RegimenFit, ...]
    #: One per node, deepest first, each shared by every cell live at that node.
    nodes: tuple[Fluctuation, ...]

    @property
    def converged(self) -> bool:
        return all(node.converged for node in self.nodes) and self.alternation.converged


def fit_regimens_msm(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    model: RegimenMSM,
    *,
    cause: str | None = None,
    outcome_learner: Learner,
    pseudo_learner: Learner,
    folds: Folds,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    alpha: float = 0.9995,
    max_iter: int = 20,
    tol: float = 1e-10,
    max_outer: int = 50,
    n_jobs: int = 1,
    warn: bool = True,
) -> MSMRegimenFit:
    r"""Target every declared cell against one working model, and project.

    The recursion is **lockstep**: outer loop over the nodes, inner loop over the cells
    live at that node, then a single fluctuation over the cells stacked, then every cell
    carried forward together.  Pooling is not an optimisation -- it is what gives the
    design its rank.  With no effect modifier a cell's :math:`p` columns are constant
    down the rows and the :math:`p` score equations collapse to one; stacking cells with
    distinct :math:`\varphi` is what separates them again.  As in
    ``ltmle::UpdateQ``, cumulative inverse probability and projection weights multiply
    the loss while :math:`(dm/d\eta)\varphi` defines the submodel direction.

    Under the identity link this runs once.  Under a link the covariate reads
    :math:`\beta`, so the pass is repeated at each new one until the coefficients settle
    -- and because the recursion carries :math:`\bar Q^*` forward into the *next*
    regression's target, a round is the whole pass rather than a re-solved fluctuation.
    The mechanism is free of :math:`\beta` and is fitted once, outside.
    """
    plan_of = {plan.label: plan for plan in plans}
    missing = sorted({cell.label for cell in model.cells} - set(plan_of))
    if missing:  # pragma: no cover - LTMLE.fit builds the cells from these very plans
        raise DataError(f"the working model names regimens the fit did not declare: {missing}")
    cumulative_pairs = {
        plan.label: mechanism.cumulative_with_unbounded(data, plan, g_bounds) for plan in plans
    }
    cumulative_unbounded = {label: pair[0] for label, pair in cumulative_pairs.items()}
    cumulative = {label: pair[1] for label, pair in cumulative_pairs.items()}
    # Once per regimen for the whole alternation, not once per node per round: a link
    # costs a further backward pass per round, so rebuilding the masks inside `one_pass`
    # would multiply the quadratic term by the round count as well.
    masks = {plan.label: data.regimen_masks(plan.values) for plan in plans}

    def one_pass(beta: FloatArray | None) -> tuple[list[list[SequentialStep]], list[Fluctuation]]:
        fluctuation_design = model.fluctuation_design_at(beta)
        carried = [seed_carried(data, scaler) for _ in model.cells]
        steps: list[list[SequentialStep]] = [[] for _ in model.cells]
        nodes: list[Fluctuation] = []
        for time in range(max(cell.horizon for cell in model.cells), 0, -1):
            live = [k for k, cell in enumerate(model.cells) if cell.horizon >= time]
            prepared = {
                k: prepare_node(
                    data,
                    plan_of[model.cells[k].label],
                    cumulative[model.cells[k].label],
                    carried[k],
                    time,
                    model.cells[k].horizon,
                    outcome_learner=outcome_learner,
                    pseudo_learner=pseudo_learner,
                    folds=folds,
                    cause=cause,
                    masks=masks[model.cells[k].label],
                    n_jobs=n_jobs,
                )
                for k in live
            }
            initial = np.concatenate([prepared[k].initial for k in live])
            fluctuation = solve_fluctuation(
                np.concatenate([prepared[k].pseudo_outcome for k in live]),
                InitialFit(initial, {_REGIMEN_ARM: initial}),
                Submodel(
                    np.concatenate([fluctuation_design[:, k, :] for k in live]),
                    {_REGIMEN_ARM: np.concatenate([fluctuation_design[:, k, :] for k in live])},
                    tuple(f"epsilon[{term}, t={time}]" for term in model.terms),
                    "sequential",
                ),
                np.concatenate(
                    [data.weights * model.weights[:, k] * prepared[k].counterfactual for k in live]
                ),
                # ``trained_on``: the set the score is taken over, and the same mask
                # ``fit_regimen`` passes.  Note what this is *not* -- a guard on which
                # rows reach the estimating equation.  The loss weight above is nonzero
                # on every at-risk row, so the same mask is material: only followers enter
                # the score, exactly as in the per-regimen update and ltmle::UpdateQ.
                np.concatenate([prepared[k].fitted_on for k in live]),
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                warn=warn,
            )
            targeted = np.split(fluctuation.targeted.arms[_REGIMEN_ARM], len(live))
            for position, k in enumerate(live):
                node = prepared[k]
                steps[k].append(
                    SequentialStep(
                        time=time,
                        trained_on=node.trained_on,
                        at_risk=node.at_risk,
                        pseudo_outcome=node.pseudo_outcome,
                        initial=node.initial,
                        targeted=targeted[position],
                        clever=node.clever,
                        fluctuation=fluctuation,
                    )
                )
                carried[k] = np.where(node.at_risk, targeted[position], _FILLER)
            nodes.append(fluctuation)
        for cell_steps in steps:
            cell_steps.reverse()
        return steps, nodes

    spec = link_for(str(model.link))
    trace: list[tuple[int, float, float]] = []
    failure: str | None = None
    if spec.is_identity:
        # ``dm/deta`` is one, so the covariate never read a beta and there is nothing to
        # alternate with. Bit for bit the single pass it would have been before links.
        steps, node_fits = one_pass(None)
        projection = _project(data, model, steps, scaler)
        beta = projection.beta
    else:
        # The starting point is arbitrary because the fixed point is not: at exit every
        # node's Qbar* is the fluctuation along the covariate at beta-hat of the
        # regression of the *next* node's Qbar*, and beta-hat is the projection of the
        # first node's. Zero is as good a start as the untargeted pass would be, and
        # costs one pass less.
        beta = np.zeros(model.n_terms)
        steps, node_fits = one_pass(beta)
        projection = _project(data, model, steps, scaler)
        previous: float | None = None
        for outer in range(1, max_outer + 1):
            shift = float(np.max(np.abs(projection.beta - beta))) / (
                1.0 + float(np.max(np.abs(beta)))
            )
            worst = max((node.trace[-1] for node in node_fits if node.trace), default=0.0)
            trace.append((outer, float(worst), shift))
            beta = projection.beta
            if shift <= tol:
                break
            if previous is not None and shift > _STALL_FACTOR * previous:
                failure = "stalled"
                break
            previous = shift
            steps, node_fits = one_pass(beta)
            projection = _project(data, model, steps, scaler)
        else:
            failure = "max_iter_reached"
        beta = projection.beta

    last = trace[-1][2] if trace else 0.0
    if failure is None and last > _UNSOLVED:
        failure = "max_iter_reached"
    alternation = ProjectionFluctuation(
        beta=beta,
        trace=tuple(trace),
        converged=bool(last <= tol),
        failure=failure,
    )
    influence = _influence(data, model, steps, scaler, projection, beta)
    return MSMRegimenFit(
        model=model,
        cause=cause,
        beta=beta,
        influence_curves=influence,
        projection=projection,
        alternation=alternation,
        fits=tuple(
            _cell_fit(
                data,
                model,
                steps,
                cause,
                cumulative_unbounded,
                cumulative,
                plan_of,
                k,
            )
            for k in range(model.n_cells)
        ),
        nodes=tuple(node_fits),
    )


def _raw_first(steps: Sequence[Sequence[SequentialStep]], scaler: OutcomeScaler) -> FloatArray:
    r"""``(n, C)`` targeted :math:`\bar Q^*_1`, on the outcome's own scale.

    The projection is solved unscaled for the reason
    :func:`cleverly.inference.influence.msm_coefficients` gives: a coefficient vector has
    no single :class:`~cleverly.inference.Scale` to map back with, and under a link
    :math:`m` is a probability rather than a scaled outcome besides.
    """
    return scaler.unscale_levels(np.column_stack([cell[0].targeted for cell in steps]))


def _project(
    data: LongitudinalData,
    model: RegimenMSM,
    steps: Sequence[Sequence[SequentialStep]],
    scaler: OutcomeScaler,
) -> ProjectionFit:
    """The working model's coefficients, from :func:`cleverly.msm.solve_projection`.

    The very solver the arm-indexed working model uses, with its arm axis read as the
    cell axis: nothing in it is about arms, and keeping one implementation is what makes
    the oracle that checks the point-treatment projection evidence about this one too.
    """
    return solve_projection(
        model.design,
        model.weights,
        _raw_first(steps, scaler),
        data.weights,
        str(model.link),
    )


def _influence(
    data: LongitudinalData,
    model: RegimenMSM,
    steps: Sequence[Sequence[SequentialStep]],
    scaler: OutcomeScaler,
    projection: ProjectionFit,
    beta: FloatArray,
) -> FloatArray:
    r"""``(n, p)`` efficient influence curve of :math:`\hat\beta`.

    .. math::

        D^*_\beta = M^{-1} \sum_c h\,(dm/d\eta)\,\varphi
            \Big[ \sum_t h^c_t (Z^c_t - \bar Q^{*c}_t) + \bar Q^{*c}_1 - m(c, V; \beta) \Big]

    The bracket is the per-cell curve *uncentred*: :math:`m` stands where the plug-in
    mean would stand in :func:`fit_regimen`'s, because what this is a curve *for* is the
    coefficient and not the cell.  :math:`M` is the matrix
    :func:`~cleverly.msm.solve_projection` already returned -- under a link it carries a
    curvature term that vanishes only where the working model fits, which is exactly what
    a projection does not promise -- so it is read from there rather than recomputed.

    The observation weights multiply the whole bracket row-wise, after :math:`m` has been
    subtracted, exactly as :func:`fit_regimen` multiplies after subtracting ``psi``.  The
    working model's own :math:`h` is part of the EIF numerator and, at the targeting
    step, part of the loss weight.
    """
    raw = _raw_first(steps, scaler)
    residual = np.column_stack(
        [
            scaler.unscale_influence(
                sum(
                    (step.clever * (step.pseudo_outcome - step.targeted) for step in cell),
                    start=np.zeros(data.n),
                )
            )
            for cell in steps
        ]
    )
    bracket = residual + raw - model.fitted(beta)
    covariate = model.weighted_design_at(beta)
    contribution = data.weights[:, None] * np.einsum("ikp,ik->ip", covariate, bracket)
    return np.asarray(np.linalg.solve(projection.jacobian, contribution.T).T, dtype=float)


def _cell_fit(
    data: LongitudinalData,
    model: RegimenMSM,
    steps: Sequence[Sequence[SequentialStep]],
    cause: str | None,
    cumulative_unbounded: dict[str, FloatArray],
    cumulative: dict[str, FloatArray],
    plan_of: dict[str, Plan],
    index: int,
) -> RegimenFit:
    """One cell's per-regimen fit, for the leverage and risk sets ``diagnostics()`` reports.

    Its ``psi_scaled`` is the plug-in mean the cell's own targeted predictions give, which
    is *not* what an MSM fit reports -- the report is the projection.  It is kept because
    the positivity diagnostics are about the regimen and are the same question whether or
    not a working model summarises it.
    """
    cell = model.cells[index]
    cell_steps = steps[index]
    psi = float(np.average(cell_steps[0].targeted, weights=data.weights))
    influence = cell_steps[0].targeted - psi
    for step in cell_steps:
        influence = influence + step.clever * (step.pseudo_outcome - step.targeted)
    return RegimenFit(
        regimen=plan_of[cell.label].regimen,
        psi_scaled=psi,
        influence_curve_scaled=data.weights * influence,
        horizon=cell.horizon,
        cause=cause,
        steps=tuple(cell_steps),
        cumulative_unbounded=cumulative_unbounded[cell.label],
        cumulative=cumulative[cell.label],
        assignment=np.asarray(plan_of[cell.label].values),
        obs_weights=np.asarray(data.weights, dtype=float),
    )
