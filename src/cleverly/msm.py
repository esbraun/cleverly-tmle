r"""Marginal structural models: a *working model* the counterfactual means are projected onto.

Every estimand so far reports one number per arm, per regime or per shift.  A study with
five dose levels and two effect modifiers therefore reports ten means, which is a table
rather than an answer.  A **marginal structural model** summarises them: a low-dimensional
model :math:`m(a, V; \beta)` for :math:`E[Y(a) \mid V]`, with :math:`V` a subset of the
baseline covariates, and the parameter is the coefficient vector :math:`\beta`.

**The model does not have to be correct.**  :math:`\beta` is defined as a *projection*,
not as the truth of an assumed regression:

.. math::

    \beta_0 = \arg\min_\beta\;
        E\Big[\textstyle\sum_a h(a, V)\,
              \big(E[Y(a) \mid V] - m(a, V; \beta)\big)^2 \Big]

with :math:`h(a, V)` a **known** weight function.  The estimand exists whatever the true
dose-response looks like: it is "the best :math:`m`-shaped summary of the counterfactual
means, in the :math:`h`-weighted least-squares sense".  That is the formulation of
Neugebauer and van der Laan (2007), and it is the reason this ships as an estimand rather
than as a modelling assumption.  Where the working model happens to be right,
:math:`\beta_0` is the truth; where it is wrong, :math:`\beta_0` is still a well-defined
functional with a valid confidence interval, and the interval is *not* secretly an
interval for a misspecified regression's coefficient.

For a model linear in :math:`\beta`, :math:`m(a, V; \beta) = \beta^\top \varphi(a, V)`,
the projection has a closed form,

.. math::

    \beta_0 = M^{-1}\, E\Big[\textstyle\sum_a h(a, V)\,\varphi(a, V)\, \bar Q(a, W)\Big],
    \qquad
    M = E\Big[\textstyle\sum_a h(a, V)\, \varphi(a, V)\varphi(a, V)^\top\Big],

which is what makes the clever covariate free of :math:`\beta` and the whole thing a
single fluctuation.  See :func:`~cleverly.fluctuation.submodel.msm_submodel` for the
covariate and :func:`~cleverly.inference.influence.msm_coefficients` for the influence
curve.

**A link puts the linear predictor inside a mean function**,
:math:`m = \operatorname{link}^{-1}(\beta^\top\varphi)`, which is how the applied
literature reports a binary outcome: an identity-link MSM is a linear-risk model, while
``link="log"`` makes a coefficient a log risk ratio and ``link="logit"`` a log odds ratio.
Three things follow, and each is a place to go wrong.  The clever covariate gains a factor
:math:`dm/d\eta` and so *depends on* :math:`\beta`; the matrix :math:`M` gains a curvature
term that vanishes only where the model fits, which no saturated check can catch
(:func:`solve_projection`); and the two are solved by alternating between them
(:func:`~cleverly.estimators.targeting.solve_with_projection`) rather than in one pass.
:class:`Link` states a link and :func:`register_link` adds one.  The identity path is
untouched by all of it -- its covariate is the same array and its projection the same
``np.linalg.solve``.

**Why this is a fourth parameter axis.**  The counterfactuals are still the arms -- the
fluctuation updates :math:`\bar Q(a, W)` at every one of them, exactly as the ``mean``
fluctuation does.  What changes is what the *parameters* are indexed by: coefficients of a
working model rather than arms.  There are :math:`p` score equations, one per term, not
:math:`K` one per arm, so it is a separate group as well as a separate axis --
:attr:`cleverly.Target.parameter_axis` says why the axes partition rather than accumulate.

**What is deliberately refused**, because of the derivation and not for want of effort --
see :func:`refuse_unsupported`:

- **weights derived from the estimated mechanism** (the "stabilised" MSM).  Then
  :math:`h` is a functional of :math:`P` and the efficient influence function carries a
  further term for the pathwise derivative through :math:`\hat g` -- the same argument
  that makes an incremental intervention need a second fluctuation and an axis of its
  own (:mod:`cleverly.interventions.incremental`).  A stabilised working model would
  need the same treatment, and until someone derives it the weights must stay known.

References
----------
Neugebauer, R. and van der Laan, M. J. (2007). Nonparametric causal effects based on
marginal structural models. *Journal of Statistical Planning and Inference* 137, 419-434.

van der Laan, M. J. and Rose, S. (2011). *Targeted Learning*, chapter 12.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.special import expit

from ._typing import FloatArray
from .data.causal_data import CausalData
from .exceptions import DataError
from .interventions.base import _as_array, _covariate_frame

__all__ = [
    "LINKS",
    "MSM",
    "Link",
    "MSMLink",
    "MSMSet",
    "ProjectionFit",
    "link_for",
    "refuse_unsupported",
    "register_link",
    "solve_projection",
]

#: The links a working model may declare.  ``"identity"`` alone keeps ``dm/dbeta`` free of
#: :math:`\beta`; the other two make the clever covariate depend on it, which is why a fit
#: declaring them alternates between the projection and the fluctuation
#: (:func:`~cleverly.estimators.targeting.solve_with_projection`).
MSMLink = Literal["identity", "log", "logit"]

#: How badly conditioned the weighted Gram matrix may be before the design is refused.
#: A reciprocal condition number below this means two terms are collinear at every arm,
#: and the projection they define is not one vector -- so it is refused where the design
#: is built rather than left to surface as an implausible coefficient.
_MIN_RCOND = 1e-10


# ---------------------------------------------------------------------- the links


@dataclass(frozen=True)
class Link:
    r"""How a working model's linear predictor becomes a counterfactual mean.

    :math:`m(a, V; \beta) = \operatorname{inverse}(\beta^\top \varphi(a, V))`, and the
    projection needs two derivatives of it.  **Both are given as functions of the mean
    rather than of the linear predictor**, because for every link here they are
    polynomials in :math:`m` -- :math:`m(1-m)` for the logit, :math:`m` for the log -- and
    the mean has already been computed wherever they are wanted.  Writing them in
    :math:`\eta` would recompute an exponential each time to arrive at the same number.

    Attributes
    ----------
    inverse:
        :math:`m(\eta)`.
    slope:
        :math:`dm/d\eta`, as a function of :math:`m`.  It is what turns the clever
        covariate :math:`h\varphi/g` into :math:`h\,(dm/d\eta)\,\varphi/g`, since
        :math:`\partial m/\partial\beta = (dm/d\eta)\,\varphi`.
    curvature:
        :math:`d^2m/d\eta^2`, as a function of :math:`m`.  It appears only in the matrix
        :func:`solve_projection` inverts, and only multiplied by the residual
        :math:`\bar Q - m` -- so it vanishes for a working model that happens to fit, and
        for the identity link at every fit.  See :func:`solve_projection` for why dropping
        it is nevertheless wrong.
    support:
        What the outcome must satisfy for the model to name a quantity: ``"unit"`` for the
        logit, ``"nonnegative"`` for the log, ``None`` for the identity.  Checked in
        :meth:`MSMSet.evaluate`, where the model meets the data.
    """

    name: str
    inverse: Callable[[FloatArray], FloatArray]
    slope: Callable[[FloatArray], FloatArray]
    curvature: Callable[[FloatArray], FloatArray]
    support: Literal["unit", "nonnegative"] | None = None

    @property
    def is_identity(self) -> bool:
        """Whether the covariate is free of ``beta``, and so needs no alternation."""
        return self.name == "identity"


#: The known links, keyed by name.  A registry rather than a branch, on the same terms as
#: :data:`~cleverly.fluctuation.submodel.SUBMODEL_BUILDERS`: a fourth link is a
#: registration and an oracle branch, not an edit to the projection.
LINKS: dict[str, Link] = {}


def register_link(link: Link) -> Link:
    """Declare a link a working model may use."""
    if link.name in LINKS:
        raise ValueError(f"a link named {link.name!r} is already registered")
    LINKS[link.name] = link
    return link


def link_for(name: str) -> Link:
    """The declared link, or a refusal naming the ones that exist."""
    try:
        return LINKS[name]
    except KeyError:
        refuse_unsupported("link", str(name))
        raise  # pragma: no cover - refuse_unsupported always raises


register_link(
    Link(
        "identity",
        inverse=lambda eta: eta,
        slope=np.ones_like,
        curvature=np.zeros_like,
    )
)
register_link(
    Link(
        "log",
        inverse=np.exp,
        # m' = m'' = m, which is the whole of why a log-link coefficient is a log ratio:
        # a unit of eta multiplies the mean rather than adding to it.
        slope=lambda m: m,
        curvature=lambda m: m,
        support="nonnegative",
    )
)
register_link(
    Link(
        "logit",
        inverse=expit,
        slope=lambda m: m * (1.0 - m),
        curvature=lambda m: m * (1.0 - m) * (1.0 - 2.0 * m),
        support="unit",
    )
)


# ------------------------------------------------------------------- the refusals


def refuse_unsupported(kind: str, detail: str = "") -> None:
    """Raise for a working model this package will not fake.

    Kept beside the constructors a user would reach for, and worded to say what the
    estimator would *need*: "not implemented" invites the reader to assume the gap is
    effort rather than a missing derivation.
    """
    if kind == "link":
        raise NotImplementedError(
            f"link={detail!r} is not a link this package knows; the registered ones are "
            f"{sorted(LINKS)}. A link is declared by its mean function and two "
            "derivatives of it -- see cleverly.msm.Link and register_link -- and needs a "
            "branch in tests/discrete_law.py before it can be trusted, since the "
            "projection it defines is solved by Newton rather than in closed form."
        )
    if kind == "estimated_weights":
        raise NotImplementedError(
            "MSM weights derived from the estimated mechanism (a 'stabilised' MSM) are not "
            "implemented. h(a, V) would then be a functional of P, so the efficient "
            "influence function carries a further term for the pathwise derivative through "
            "g-hat that the curve reported here does not have. Pass weights= as a known "
            "function of the arm and the covariates, or leave it None for uniform weights."
        )
    raise ValueError(f"unknown refusal {kind!r}")


# ------------------------------------------------------------------ the declaration


@dataclass(frozen=True)
class MSM:
    r"""A working model, declared as a design function and the names of its terms.

    ``design`` is handed one treatment *level* -- the user's own label, not an internal
    code -- and a dataframe of the covariates in the backend the data arrived in, and
    returns the ``(n, p)`` design :math:`\varphi(a, V)` for that arm.  It is called once
    per arm.

    ``` python
    MSM(
        design=lambda a, w: np.column_stack([np.ones(len(w)), np.full(len(w), a)]),
        terms=("(intercept)", "a"),
    )
    ```

    Reading the covariate frame rather than the whole dataset is the same restriction
    :class:`cleverly.interventions.Rule` works under and for the same reason: a working
    model whose design read ``Y`` would not be a model for a counterfactual mean, and one
    that read the observed ``A`` would be summarising the arms nobody was assigned to
    differently from the ones they were.  The columns are the *encoded* covariates, so a
    categorical column appears as the indicators the data layer expanded it into;
    ``data.covariate_names`` is the list to write a design against.

    Attributes
    ----------
    design:
        ``(arm_label, covariate_frame) -> (n, p)``.
    terms:
        One name per design column, used verbatim in the reported parameter names:
        ``msm[a:W1]``.
    weights:
        ``h(a, V)``, as ``(arm_label, covariate_frame) -> (n,)``.  ``None`` means uniform,
        which weights every arm and every unit equally.  It must be a **known** function;
        see :func:`refuse_unsupported`.
    link:
        ``"identity"``, ``"log"`` or ``"logit"``.  The identity is the only one whose
        :math:`\partial m/\partial\beta` is free of :math:`\beta`, and so the only one
        whose fit is a single fluctuation; the others alternate.  What the coefficients
        *mean* changes with it -- under ``"log"`` a coefficient is a log risk ratio and
        under ``"logit"`` a log odds ratio, which is what
        :meth:`~cleverly.estimators.base.TMLEResult.coefficients` exponentiates.
    """

    design: Callable[[Any, Any], Any]
    terms: tuple[str, ...]
    weights: Callable[[Any, Any], Any] | None = None
    link: MSMLink = "identity"

    def __post_init__(self) -> None:
        link_for(str(self.link))
        terms = tuple(str(term) for term in self.terms)
        if not terms:
            raise DataError("a working model needs at least one term")
        if len(set(terms)) != len(terms):
            raise DataError(f"working-model terms must be distinct; got {list(terms)}")
        if not callable(self.design):
            raise DataError("design= must be callable: (arm_label, covariate_frame) -> (n, p)")
        if self.weights is not None and not callable(self.weights):
            refuse_unsupported("estimated_weights")
        object.__setattr__(self, "terms", terms)

    # ---------------------------------------------------------------- shorthand

    @classmethod
    def linear(
        cls,
        *,
        modifiers: Sequence[str] = (),
        interaction: bool = True,
        weights: Callable[[Any, Any], Any] | None = None,
        link: MSMLink = "identity",
    ) -> MSM:
        r"""The usual dose-response model, without writing the design out by hand.

        ``MSM.linear(modifiers=("W1",))`` is
        :math:`m(a, V) = \beta_0 + \beta_1 a + \beta_2 W_1 + \beta_3 a W_1`, and
        ``interaction=False`` drops the last term.  With no modifiers it is a plain
        dose-response line, whose slope is the average change in counterfactual mean per
        unit of treatment under the :math:`h`-weighted projection.  ``link=`` puts that
        linear predictor inside a link, so ``link="log"`` makes :math:`\beta_1` a log risk
        ratio per unit of treatment rather than a risk difference.

        The arm enters as a *number*, so this requires numeric treatment levels: it reads
        the arms as a dose to be extrapolated between.  A string-labelled treatment is
        refused rather than coded silently, because ``{"low", "medium", "high"}`` sorts
        alphabetically and the resulting slope would be per-step in an order nobody chose.
        Pass ``design=`` and code the arms yourself where that is what you want.
        """
        names = tuple(str(m) for m in modifiers)
        terms = ("(intercept)", "a", *names)
        if interaction:
            terms = (*terms, *(f"a:{name}" for name in names))

        def build(level: Any, frame: Any) -> FloatArray:
            dose = _numeric_level(level)
            columns = [np.ones(_frame_len(frame)), np.full(_frame_len(frame), dose)]
            modifier_values = [_column(frame, name) for name in names]
            columns.extend(modifier_values)
            if interaction:
                columns.extend(dose * values for values in modifier_values)
            return np.column_stack(columns)

        return cls(design=build, terms=terms, weights=weights, link=link)


def _frame_len(frame: Any) -> int:
    """Row count of whatever backend's frame a design was handed."""
    try:
        return len(frame)
    except TypeError:  # pragma: no cover - every supported backend defines __len__
        return int(frame.shape[0])


def _column(frame: Any, name: str) -> FloatArray:
    """One covariate column as a float array, in whichever backend the frame is."""
    try:
        values = frame[name]
    except (KeyError, IndexError) as error:
        raise DataError(
            f"the working model names a modifier {name!r} that is not one of the encoded "
            "covariates. Note that a categorical covariate appears under the indicator "
            "names the data layer expanded it into; data.covariate_names is the list to "
            "write against."
        ) from error
    return np.asarray(_as_array(values), dtype=float).reshape(-1)


def _numeric_level(level: Any) -> float:
    """A treatment level read as a dose, or a refusal naming the alternative."""
    try:
        return float(level)
    except (TypeError, ValueError):
        raise DataError(
            f"MSM.linear reads the treatment level {level!r} as a number, and it is not "
            "one. A working model linear in the arm treats it as a dose to interpolate "
            "between, which a label has no ordering for -- and the sort order a coding "
            "would fall back on is not one anybody chose. Pass design= and code the arms "
            "explicitly."
        ) from None


# ------------------------------------------------------------------- the evaluation


@dataclass(frozen=True)
class MSMSet:
    r"""A working model evaluated on the data: the design and weights at every arm.

    Arrays only, no callables -- the same rule
    :class:`cleverly.interventions.RegimeSet` and
    :class:`cleverly.interventions.ShiftSet` follow, and for the same two reasons.
    Everything reached through :meth:`cleverly.TMLE.retarget` -- a truncation curve, the
    score check, a sensitivity sweep -- must target the *declared* model without the
    user's design function having to be callable again, and a loaded result carries no
    callables at all.

    Attributes
    ----------
    terms:
        One name per coefficient, in column order.
    design:
        ``(n, K, p)``: ``design[i, j, :]`` is :math:`\varphi(\text{arms}[j], V_i)`.
    weights:
        ``(n, K)``: ``weights[i, j]`` is :math:`h(\text{arms}[j], V_i)`.
    arms:
        The arm codes the second axis is keyed by, ascending -- the treatment arms the
        projection sums over, which stay arms even though the parameters do not.
    link:
        The declared link, by name.  A *string* rather than a :class:`Link`, so that this
        object stays arrays-and-scalars: it is what a saved fit writes and what
        :meth:`cleverly.TMLE.retarget` reads, and a callable in either place would make a
        loaded result a different thing from a fresh one.
    """

    terms: tuple[str, ...]
    design: FloatArray
    weights: FloatArray
    arms: tuple[float, ...]
    link: MSMLink = "identity"

    def __post_init__(self) -> None:
        link_for(str(self.link))
        design = np.asarray(self.design, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if design.ndim != 3 or design.shape[1] != len(self.arms):
            raise DataError(
                f"the working model's design must have shape (n, {len(self.arms)}, p) -- "
                f"rows, arms, terms -- got {design.shape}"
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
                "the working model's weights must be finite and non-negative; h(a, V) is "
                "a weight in a least-squares projection, not a signed contrast"
            )
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "weights", weights)
        self._check_rank()

    def _check_rank(self) -> None:
        """Refuse a design whose projection is not one vector.

        Checked here rather than left to the solve, because a rank-deficient design does
        not fail loudly downstream: ``lstsq`` would return the minimum-norm solution and
        the fit would report coefficients for a parameter that is not identified.
        """
        gram = self.gram
        if gram.shape[0] == 0:
            return
        rcond = float(1.0 / np.linalg.cond(gram)) if np.all(np.isfinite(gram)) else 0.0
        if not np.isfinite(rcond) or rcond < _MIN_RCOND:
            raise DataError(
                f"the working model's terms {list(self.terms)} are collinear across the "
                f"arms (reciprocal condition number {rcond:.3g}), so the projection they "
                "define is not a single coefficient vector. Drop a term, or check that a "
                "modifier is not constant and that an interaction is not a copy of its "
                "main effect."
            )

    # ------------------------------------------------------------------ build

    @classmethod
    def evaluate(cls, msm: MSM, data: CausalData) -> MSMSet:
        """Evaluate ``msm``'s design and weights at every arm of ``data``.

        Called once, where the nuisances are fitted, so that a fit and everything
        retargeted from it agree on what the working model is by construction rather than
        by re-running the user's function and hoping it is deterministic.
        """
        if data.is_continuous_treatment:
            raise DataError(
                "a working model is a projection of the counterfactual means over the "
                f"arms, and {data.treatment_name} was declared continuous, which has none. "
                "Declare shifts= for a continuous dose, or drop treatment_kind="
                "'continuous' to read the levels as arms."
            )
        _check_support(link_for(str(msm.link)), data)
        frame = _covariate_frame(data)
        designs: list[FloatArray] = []
        weights: list[FloatArray] = []
        for code in data.arm_codes:
            label = data.arm_label(code)
            designs.append(_arm_design(msm, label, frame, data.n))
            weights.append(_arm_weights(msm, label, frame, data.n))
        return cls(
            msm.terms,
            np.stack(designs, axis=1),
            np.column_stack(weights),
            tuple(float(code) for code in data.arm_codes),
            msm.link,
        )

    # ------------------------------------------------------------------ access

    @property
    def n(self) -> int:
        return int(self.design.shape[0])

    @property
    def n_arms(self) -> int:
        return int(self.design.shape[1])

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def codes(self) -> tuple[float, ...]:
        """Coefficient codes, ``(0.0, ..., p-1.0)`` -- what the parameters are keyed by.

        The arms are keyed separately, by :attr:`arms`.  Both are float codes with labels
        carried alongside, which is the convention every axis in this package follows; the
        difference is only which of the two a *parameter* belongs to.
        """
        return tuple(float(j) for j in range(self.n_terms))

    @property
    def labels(self) -> dict[float, str]:
        """Coefficient code to term name, which is what ``parameter_name`` is given."""
        return {float(j): term for j, term in enumerate(self.terms)}

    @property
    def gram(self) -> FloatArray:
        r"""``M``, the ``(p, p)`` weighted Gram matrix :math:`P_n \sum_a h\,\varphi\varphi^\top`.

        Free of every nuisance estimate: :math:`h` and :math:`\varphi` are known
        functions, so :math:`M` depends on the data only through the empirical
        distribution of :math:`V`.  That is what keeps the influence curve to the two
        terms in :func:`~cleverly.inference.influence.msm_coefficients` rather than
        carrying a third for the estimation of :math:`M`.

        Here for :meth:`_check_rank` only: the *observation*-weighted form the estimate
        uses is built inside :func:`solve_projection`, and this deliberately does not try
        to be it.  Two implementations of one projection would be two things to keep in
        step, and it is the estimate's that is checked against an oracle.

        Under a non-identity link this is **not** the matrix the estimate inverts -- that
        one carries :math:`(dm/d\eta)^2` and a curvature term, and depends on
        :math:`\bar Q` -- but it is still exactly the right thing for the rank check,
        which is a statement about the *design* being one vector's worth of directions and
        not about where the projection lands.
        """
        return np.einsum("ijp,ijq,ij->pq", self.design, self.design, self.weights) / max(self.n, 1)

    @property
    def weighted_design(self) -> FloatArray:
        r"""``(n, K, p)`` array of :math:`h(a, V)\,\varphi(a, V)`.

        The only thing an identity-link clever covariate needs of a working model -- it
        divides this by the mechanism and nothing else -- so it is what
        :func:`~cleverly.fluctuation.submodel.msm_submodel` is handed.  Passing the
        product rather than this object keeps every submodel builder taking plain arrays,
        which is what lets the registry dispatch on the group name alone.

        Under another link the covariate carries a further factor :math:`dm/d\eta`, which
        depends on :math:`\beta`: :meth:`weighted_design_at` is that array, and this is its
        :math:`\beta`-free special case.
        """
        return np.asarray(self.design * self.weights[:, :, None], dtype=float)

    def fitted(self, beta: FloatArray) -> FloatArray:
        r"""``(n, K)`` working-model means :math:`m(a, V; \beta)`, arm by arm."""
        eta = np.einsum("ijp,p->ij", self.design, np.asarray(beta, dtype=float))
        return np.asarray(link_for(str(self.link)).inverse(eta), dtype=float)

    def weighted_design_at(self, beta: FloatArray | None) -> FloatArray:
        r"""``(n, K, p)`` array :math:`h\,(dm/d\eta)\,\varphi`: the clever covariate's numerator.

        With the identity link this *is* :attr:`weighted_design` and ``beta`` is not read
        at all, which is what keeps a fit declaring no link on the code path it was on
        before links existed.  ``beta=None`` is accepted only there: under any other link
        the covariate is not defined without one, and
        :func:`~cleverly.estimators.targeting.solve_with_projection` is what supplies it.
        """
        link = link_for(str(self.link))
        if link.is_identity:
            return self.weighted_design
        if beta is None:
            raise DataError(
                f"a working model with link={self.link!r} has a clever covariate that "
                "depends on beta, so the coefficients have to be solved for before it can "
                "be built. That alternation is solve_with_projection; a caller rebuilding "
                "the covariate on its own has to say which beta it means."
            )
        slope = np.asarray(link.slope(self.fitted(beta)), dtype=float)
        return np.asarray(self.design * (self.weights * slope)[:, :, None], dtype=float)

    def arm_column(self, code: float) -> FloatArray:
        """One arm's ``(n, p)`` design."""
        return np.asarray(self.design[:, self.arms.index(float(code)), :], dtype=float)

    def subset(self, index: Any) -> MSMSet:
        """The same working model on a row subset -- a bootstrap resample, a fold.

        Sliced rather than re-evaluated, exactly as
        :meth:`cleverly.interventions.RegimeSet.subset` is: the design is a function of
        :math:`V` alone, so the two agree, and slicing is what keeps a loaded result --
        which carries the evaluated arrays but not the callables that made them -- usable
        everywhere a freshly fitted one is.

        Note that ``M`` is *not* sliced but recomputed, since it is an empirical mean over
        the rows that remain.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(self, design=self.design[idx], weights=self.weights[idx])


# ------------------------------------------------------------------- the projection


#: A Newton step shorter than this has stopped buying anything; the line search gives up
#: and the caller is told the projection did not converge rather than being handed the
#: iterate it happened to stop on without comment.
_MIN_STEP = 1e-8


@dataclass(frozen=True)
class ProjectionFit:
    r"""The solved coefficients, and how the solve went.

    Attributes
    ----------
    beta:
        ``(p,)`` coefficients of the working model.
    jacobian:
        ``(p, p)`` matrix :math:`M = -\partial U/\partial\beta` at ``beta``.  This is what
        the influence curve is premultiplied by the inverse of, and under the identity
        link it is the weighted Gram matrix.
    converged, n_iter, score:
        Whether the estimating equation was solved, in how many Newton steps, and the
        relative size of what was left.  Reported rather than raised, on the terms every
        other solve here reports: a sensitivity sweep that pushes a fit into bad territory
        must still return a number and say it is bad.
    """

    beta: FloatArray
    jacobian: FloatArray
    converged: bool = True
    n_iter: int = 0
    score: float = 0.0


def solve_projection(
    design: FloatArray,
    model_weights: FloatArray,
    predictions: FloatArray,
    weights: FloatArray,
    link: str = "identity",
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> ProjectionFit:
    r"""Project ``predictions`` onto the working model: the one solver, for every link.

    :math:`\hat\beta` solves the weighted least-squares normal equations

    .. math::

        U(\beta) = P_n\Big[\textstyle\sum_a h(a, V)\,\frac{dm}{d\eta}\,\varphi(a, V)\,
                   \big(\bar Q(a, W) - m(a, V; \beta)\big)\Big] = 0 ,

    and the matrix returned beside it is

    .. math::

        M = -\frac{\partial U}{\partial\beta}
          = P_n\Big[\textstyle\sum_a h\,\Big(\big(\tfrac{dm}{d\eta}\big)^2
            - \big(\bar Q - m\big)\tfrac{d^2m}{d\eta^2}\Big)\,\varphi\varphi^\top\Big].

    **The curvature term is not optional.**  It vanishes for the identity link, where
    :math:`d^2m/d\eta^2 = 0`, and for any link where the working model happens to fit
    exactly -- which is precisely what a *projection* does not promise.  Dropping it gives
    a matrix that is right on a saturated model and wrong on every other, so no saturated
    check can catch it; ``tests/unit/test_influence_gateaux_msm.py`` carries the mutation
    against the deliberately non-saturated oracle.

    With the identity link :math:`U` is linear in :math:`\beta` and this is one
    ``np.linalg.solve`` of the arithmetic that has always been here -- no iteration, and
    bit for bit what the closed form gave.  Otherwise it is damped Newton, which is where
    the outer :math:`(\beta, \epsilon)` alternation comes from: the clever covariate reads
    :math:`dm/d\eta`, so it moves when :math:`\beta` does.

    Parameters
    ----------
    design:
        ``(n, K, p)`` array :math:`\varphi(a, V)`.
    model_weights:
        ``(n, K)`` array :math:`h(a, V)`, the *working model's* weights.
    predictions:
        ``(n, K)`` counterfactual means to project, on the scale the coefficients are to
        be reported in -- the outcome's own, for the reason
        :func:`~cleverly.inference.influence.msm_coefficients` gives.
    weights:
        ``(n,)`` *observation* weights, which tilt the population the projection is taken
        over rather than trading the arms off within it.
    """
    phi = np.asarray(design, dtype=float)
    h = np.asarray(model_weights, dtype=float)
    q = np.asarray(predictions, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mass = float(w.sum())
    spec = link_for(str(link))

    if spec.is_identity:
        # The closed form, written exactly as it was before links existed so that an
        # identity-link fit is bit for bit the fit it was.
        weighted_design = phi * h[:, :, None]
        gram = np.einsum("ijp,ijq,ij,i->pq", phi, phi, h, w) / mass
        moment = np.einsum("ijp,ij,i->p", weighted_design, q, w) / mass
        beta = np.asarray(np.linalg.solve(gram, moment), dtype=float)
        _, _, relative = _projection_state(phi, h, q, w, mass, spec, beta)
        return ProjectionFit(beta, gram, converged=True, n_iter=0, score=relative)

    beta = np.zeros(phi.shape[2])
    u, jacobian, relative = _projection_state(phi, h, q, w, mass, spec, beta)
    n_iter = 0
    while n_iter < max_iter and relative > tol:
        step = _newton_step(jacobian, u)
        length = 1.0
        while True:
            candidate = beta + length * step
            trial = _projection_state(phi, h, q, w, mass, spec, candidate)
            if trial[2] <= relative or length <= _MIN_STEP:
                break
            length *= 0.5
        if trial[2] > relative:
            break  # the line search could not improve on where it stands
        beta = candidate
        u, jacobian, relative = trial
        n_iter += 1
    return ProjectionFit(
        beta, jacobian, converged=bool(relative <= tol), n_iter=n_iter, score=float(relative)
    )


def _projection_state(
    phi: FloatArray,
    h: FloatArray,
    q: FloatArray,
    w: FloatArray,
    mass: float,
    link: Link,
    beta: FloatArray,
) -> tuple[FloatArray, FloatArray, float]:
    """``(U, M, relative score)`` at one ``beta``.

    The score is divided by the largest magnitude it could have had at a unit residual --
    the same dimensionless form :func:`cleverly.fluctuation._score.relative_score` puts
    the fluctuation's score in, so that one tolerance means the same thing on problems
    whose designs differ by orders of magnitude.
    """
    m = np.asarray(link.inverse(np.einsum("ijp,p->ij", phi, beta)), dtype=float)
    residual = q - m
    slope = np.asarray(link.slope(m), dtype=float)
    curvature = np.asarray(link.curvature(m), dtype=float)
    u = np.einsum("ijp,ij,i->p", phi, h * slope * residual, w) / mass
    jacobian = (
        np.einsum("ijp,ijq,ij,i->pq", phi, phi, h * (slope**2 - residual * curvature), w) / mass
    )
    scale = np.einsum("ijp,ij,i->p", np.abs(phi), h * np.abs(slope), w) / mass
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(scale > 0.0, np.abs(u) / scale, 0.0)
    return u, jacobian, float(np.max(relative)) if relative.size else 0.0


def _newton_step(jacobian: FloatArray, u: FloatArray) -> FloatArray:
    """``M^-1 U``, falling back to least squares where ``M`` is singular.

    A singular Jacobian is not the same failure as a collinear design, which
    :meth:`MSMSet._check_rank` has already refused: it means the link's slope has
    collapsed on this iterate -- every fitted value pinned at 0 or 1 under a logit, say --
    so the minimum-norm step is the one that gets out of it, and the line search below
    decides whether it helped.
    """
    try:
        return np.asarray(np.linalg.solve(jacobian, u), dtype=float)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(jacobian, u, rcond=None)[0], dtype=float)


def _arm_design(msm: MSM, label: Any, frame: Any, n: int) -> FloatArray:
    values = np.asarray(_as_array(msm.design(label, frame)), dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape != (n, len(msm.terms)):
        raise DataError(
            f"the working model's design returned shape {values.shape} for arm {label!r}; "
            f"expected ({n}, {len(msm.terms)}) -- one row per unit and one column per term "
            f"{list(msm.terms)}"
        )
    return values


def _arm_weights(msm: MSM, label: Any, frame: Any, n: int) -> FloatArray:
    if msm.weights is None:
        return np.ones(n)
    values = np.asarray(_as_array(msm.weights(label, frame)), dtype=float).reshape(-1)
    if values.shape != (n,):
        raise DataError(
            f"the working model's weights returned shape {values.shape} for arm {label!r}; "
            f"expected ({n},) -- one weight per unit"
        )
    return values


def _check_support(link: Link, data: CausalData) -> None:
    """Refuse a link whose mean function cannot reach the outcome being modelled.

    Checked against the *observed outcome* rather than against the predictions, and that
    is the stronger check rather than a convenient one: the logistic fluctuation keeps
    every targeted prediction inside the outcome's own range, so an outcome in ``[0, 1]``
    guarantees a :math:`\\bar Q^*` a logit model can match and a non-negative one
    guarantees a :math:`\\bar Q^*` a log model can.  The projection would still *run*
    otherwise -- it would just be a least-squares fit of a curve that cannot reach its
    target, reported under a name that says it did.
    """
    if link.support is None:
        return
    observed = np.asarray(data.outcome, dtype=float)[np.asarray(data.observed, dtype=bool)]
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
            "q_bounds rescales the outcome for the *fit* but not for the projection, "
            "which is solved on the outcome's own scale so that its coefficients are "
            "reported in the units they were declared in."
        )
