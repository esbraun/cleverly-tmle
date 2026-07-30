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

**Why this is a fourth parameter axis.**  The counterfactuals are still the arms -- the
fluctuation updates :math:`\bar Q(a, W)` at every one of them, exactly as the ``mean``
fluctuation does.  What changes is what the *parameters* are indexed by: coefficients of a
working model rather than arms.  There are :math:`p` score equations, one per term, not
:math:`K` one per arm, so it is a separate group as well as a separate axis --
:attr:`cleverly.Target.parameter_axis` says why the axes partition rather than accumulate.

**What is deliberately refused**, each because of the derivation and not for want of
effort -- see :func:`refuse_unsupported`:

- a **non-identity link**.  With :math:`m = \operatorname{expit}(\beta^\top \varphi)` the
  derivative :math:`\partial m/\partial\beta` depends on :math:`\beta`, so the clever
  covariate does too and solving the score needs an outer :math:`(\beta, \epsilon)`
  iteration this fluctuation does not run.  Reporting the one-shot version would attach a
  standard error to an equation that was not solved.
- **weights derived from the estimated mechanism** (the "stabilised" MSM).  Then
  :math:`h` is a functional of :math:`P` and the efficient influence function carries a
  further term for the pathwise derivative through :math:`\hat g` -- the same argument
  that refuses incremental propensity-score interventions in
  :func:`cleverly.interventions.refuse_unsupported`.

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

from ._typing import FloatArray
from .data.causal_data import CausalData
from .exceptions import DataError
from .interventions.base import _as_array, _covariate_frame

__all__ = ["MSM", "MSMSet", "refuse_unsupported"]

#: Links whose ``dm/dbeta`` is free of ``beta``, and so admit a one-fluctuation TMLE.
MSMLink = Literal["identity"]

#: How badly conditioned the weighted Gram matrix may be before the design is refused.
#: A reciprocal condition number below this means two terms are collinear at every arm,
#: and the projection they define is not one vector -- so it is refused where the design
#: is built rather than left to surface as an implausible coefficient.
_MIN_RCOND = 1e-10


# ------------------------------------------------------------------- the refusals


def refuse_unsupported(kind: str, detail: str = "") -> None:
    """Raise for a working model this package will not fake.

    Kept beside the constructors a user would reach for, and worded to say what the
    estimator would *need*: "not implemented" invites the reader to assume the gap is
    effort rather than a missing derivation.
    """
    if kind == "link":
        raise NotImplementedError(
            f"link={detail!r} is not implemented; only 'identity' is. A non-identity link "
            "makes dm/dbeta depend on beta, so the clever covariate does too and solving "
            "the score equation needs an outer (beta, epsilon) iteration that this "
            "fluctuation does not run -- a one-shot version would report a standard error "
            "for an equation that was not solved. For a binary outcome an identity-link "
            "MSM is a linear-risk model, and its coefficients are risk differences."
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
        Only ``"identity"``.  A model linear in :math:`\beta` is what keeps the clever
        covariate free of :math:`\beta`.
    """

    design: Callable[[Any, Any], Any]
    terms: tuple[str, ...]
    weights: Callable[[Any, Any], Any] | None = None
    link: MSMLink = "identity"

    def __post_init__(self) -> None:
        if self.link != "identity":
            refuse_unsupported("link", str(self.link))
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
    ) -> MSM:
        r"""The usual dose-response model, without writing the design out by hand.

        ``MSM.linear(modifiers=("W1",))`` is
        :math:`m(a, V) = \beta_0 + \beta_1 a + \beta_2 W_1 + \beta_3 a W_1`, and
        ``interaction=False`` drops the last term.  With no modifiers it is a plain
        dose-response line, whose slope is the average change in counterfactual mean per
        unit of treatment under the :math:`h`-weighted projection.

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

        return cls(design=build, terms=terms, weights=weights)


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
    """

    terms: tuple[str, ...]
    design: FloatArray
    weights: FloatArray
    arms: tuple[float, ...]

    def __post_init__(self) -> None:
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
        """
        return np.einsum("ijp,ijq,ij->pq", self.design, self.design, self.weights) / max(self.n, 1)

    @property
    def weighted_design(self) -> FloatArray:
        r"""``(n, K, p)`` array of :math:`h(a, V)\,\varphi(a, V)`.

        The only thing the clever covariate needs of a working model -- it divides this by
        the mechanism and nothing else -- so it is what
        :func:`~cleverly.fluctuation.submodel.msm_submodel` is handed.  Passing the
        product rather than this object keeps every submodel builder taking plain arrays,
        which is what lets the registry dispatch on the group name alone.
        """
        return np.asarray(self.design * self.weights[:, :, None], dtype=float)

    def moment(self, means: FloatArray) -> FloatArray:
        r"""``(p,)`` vector :math:`P_n \sum_a h(a,V)\,\varphi(a,V)\,\bar Q(a, W)`.

        ``means`` is the ``(n, K)`` array of counterfactual predictions, arms in
        :attr:`arms` order.
        """
        values = np.asarray(means, dtype=float)
        if values.shape != self.design.shape[:2]:
            raise DataError(
                f"counterfactual means must have shape {self.design.shape[:2]}; got {values.shape}"
            )
        return np.einsum("ijp,ij,ij->p", self.design, self.weights, values) / max(self.n, 1)

    def coefficients(self, means: FloatArray) -> FloatArray:
        r"""The projection :math:`\hat\beta = M^{-1} P_n \sum_a h\,\varphi\,\bar Q`."""
        return np.asarray(np.linalg.solve(self.gram, self.moment(means)), dtype=float)

    def fitted(self, beta: FloatArray) -> FloatArray:
        r"""``(n, K)`` array of :math:`m(a, V_i; \beta)` -- the working model's own values."""
        return np.asarray(self.design @ np.asarray(beta, dtype=float).reshape(-1), dtype=float)

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
