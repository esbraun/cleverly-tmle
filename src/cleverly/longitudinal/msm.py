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

*The node fluctuation is pooled across the cells.*  At one time point the covariate's
:math:`p` columns get their rank from summing over the arms *within one row*: a unit
contributes :math:`\varphi(a, V)` at the arm it received.  Here there is nothing to sum
over within a row -- a regimen is a plan, not a value some unit took -- so a per-cell
covariate is :math:`\varphi(c, V)` times the scalar :math:`h_t^c`, and whenever the
working model has no effect modifier :math:`\varphi(c, V)` is *constant down the rows*
and the :math:`p` score equations collapse to one.  So each node solves a single
fluctuation over the cells stacked, :math:`C \cdot n` rows and one shared
:math:`\epsilon`.  Under a saturated model the stacked covariate is exactly
block-diagonal and each cell's block is the covariate the per-regimen recursion would
have used, which is why the report reduces to that one.

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
says how the cells are traded off against each other and sits inside the covariate; the
second tilts the population the projection is taken over, tiles into the pooled
fluctuation and multiplies the finished curve row-wise.  Merging them would divide the
estimating equation by the very tilt it applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import FloatArray
from ..exceptions import DataError
from ..msm import MSM, Link, check_projection_rank, link_for
from .data import LongitudinalData
from .regimen import Plan

__all__ = [
    "Cell",
    "RegimenMSM",
    "evaluate_regimen_msm",
]


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
        return np.einsum("ikp,ikq,ik->pq", self.design, self.design, self.weights) / max(self.n, 1)

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
