r"""The projection of a conditional contrast onto a declared basis of the modifiers.

The engine side of :class:`cleverly.ConditionalContrast` with ``method="tmle"``.  For a
binary treatment, or a declared pair ``(t, r)`` of a multi-arm one, with modifiers
:math:`V \subseteq W` and a known basis :math:`\varphi(V)` of fixed dimension :math:`p`:

- **difference scale.** :math:`\beta_0 = M^{-1} E[\varphi(V)\,\tau(W)]`, with
  :math:`\tau(W) = \bar Q(t, W) - \bar Q(r, W)` and :math:`M = E[\varphi\varphi^\top]`.
  It is the best linear projection of the conditional effect
  :math:`E[Y^t - Y^r \mid V]` onto :math:`\operatorname{span}(\varphi)`.
- **log-ratio scale.** :math:`\beta_0` solves
  :math:`U(\beta) = E[\varphi\{(\bar Q_t + \bar Q_r)\,\operatorname{expit}(\varphi^\top\beta)
  - \bar Q_t\}] = 0`, the minimiser over :math:`\{\varphi^\top\beta\}` of the log-CRR risk
  :math:`E[(\bar Q_t + \bar Q_r)\log(1 + e^\theta) - \bar Q_t\theta]` (van der Laan, Carone
  and Luedtke, arXiv:2402.01972v2, a preprint, Example 2).  With :math:`\varphi = 1` it is
  :math:`\log(E\bar Q_t / E\bar Q_r)`, the marginal log risk ratio.

Both are smooth functionals of the arm regressions with a known weight, so a TMLE
fluctuates :math:`\bar Q` along the clever covariate
:math:`H(A, W) = \sum_a \mathbb 1\{A = a\}\,c_a(V)/\{g(a \mid W)\,\Pi(a, W)\}` and reads
:math:`\hat\beta` off the targeted fit.  On the difference scale :math:`c_t = \varphi`,
:math:`c_r = -\varphi` and every other arm carries 0; on the log-ratio scale
:math:`c_t = \varphi\{\operatorname{expit}(\eta) - 1\}` and
:math:`c_r = \varphi\operatorname{expit}(\eta)`, which read :math:`\beta`, so the
fluctuation and the projection alternate
(:func:`~cleverly.estimators.targeting.solve_with_projection`).  The influence curves are
in :func:`~cleverly.inference.influence.effect_modification_coefficients`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.special import expit

from ._typing import FloatArray
from .bases import INTERCEPT, Modifiers, check_basis, resolve
from .data.causal_data import CausalData
from .exceptions import CapabilityError, DataError, Refusal, RefusalReason
from .msm import ProjectionFit, check_projection_rank

__all__ = [
    "COMPOSITION_GAPS",
    "SCALES",
    "EffectModification",
    "EffectModificationSet",
    "ModifierReader",
    "effect_modification_data_refusal",
    "effect_modification_ep_refusal",
    "effect_modification_gap",
    "log_ratio_scale_refusal",
    "modifier_frame",
]

#: The two scales a conditional contrast is reported on.
ContrastScale = Literal["difference", "log_ratio"]
SCALES: tuple[str, ...] = ("difference", "log_ratio")

#: The registered target of each scale.
TARGET_OF_SCALE = {"difference": "cate_projection", "log_ratio": "log_crr_projection"}

#: What each composition outside P1 is missing (R7 of the P1 plan), by the name of the axis.
COMPOSITION_GAPS: dict[str, str] = {
    "an intermediate": (
        "the controlled direct effect fixes an intermediate variable, and the projection's "
        "clever covariate with the intermediate mechanism and its study cell are not built."
    ),
    "a continuous or vector treatment": (
        "the projection contrasts two arms, and a dose has no pair of arms to contrast."
    ),
}

_COMPOSITION_ALTERNATIVE = {
    "an intermediate": "Fit ControlledDirectEffect(...) for the effect at a fixed intermediate.",
    "a continuous or vector treatment": (
        "Discretize the treatment and declare treatment= and reference= on the contrast."
    ),
}


def effect_modification_gap(axis: str) -> Refusal:
    """The R7 refusal of one composition outside P1.

    Parameters
    ----------
    axis : str
        A key of :data:`COMPOSITION_GAPS`.

    Returns
    -------
    Refusal
        An ``IMPLEMENTATION_GAP`` record with roadmap ``P1``.
    """
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"ConditionalContrast does not yet compose with {axis}: {COMPOSITION_GAPS[axis]}",
        alternative=_COMPOSITION_ALTERNATIVE[axis],
        roadmap="P1",
    )


def effect_modification_data_refusal(data: CausalData) -> Refusal | None:
    """The R7 refusal the data decide, or ``None``.

    A missing treatment and baseline strata compose with the projection (the in-sample
    composite construction, and one block of basis terms per stratum), so only a dose and
    an intermediate are refused here.

    Parameters
    ----------
    data : CausalData
        The prepared data.

    Returns
    -------
    Refusal or None
        The first composition gap the data meet.
    """
    if data.is_continuous_treatment or data.treatment_kind == "vector":
        return effect_modification_gap("a continuous or vector treatment")
    if data.has_intermediate:
        return effect_modification_gap("an intermediate")
    return None


#: R7-ep-mt of the P1 plan: the EP learner with a missing treatment.
EP_MISSING_TREATMENT_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "The EP learner cross-fits its nuisances and pools every row in one debiasing "
    "regression, and the cross-fitted composite construction of a missing treatment is "
    "open (roadmap X53).",
    alternative=(
        "Fit method='tmle' with model= declared and cross_fit=False (in sample), or drop the "
        "rows with a missing treatment if they are missing completely at random."
    ),
    roadmap="X53",
)

#: R7-ep-strata of the P1 plan: the EP learner with baseline strata.
EP_STRATA_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "The EP curve is a function of the modifiers, and a baseline stratum is one more "
    "modifier: the curve given V within stratum s is the curve given (V, S).",
    alternative=(
        "Declare the stratum column as a modifier, with an Indicators basis of it in the sieve."
    ),
)


def effect_modification_ep_refusal(data: CausalData) -> Refusal | None:
    """The refusal of an EP-learner conditional contrast the data decide, or ``None``.

    Parameters
    ----------
    data : CausalData
        The prepared data.

    Returns
    -------
    Refusal or None
        R7-ep-mt for a missing treatment, R7-ep-strata for baseline strata.
    """
    if data.has_missing_treatment:
        return EP_MISSING_TREATMENT_REFUSAL
    if data.has_strata:
        return EP_STRATA_REFUSAL
    return None


def log_ratio_scale_refusal(
    data: CausalData, scale: str, q_bounds: tuple[float, float] | None
) -> Refusal | None:
    """The R5 refusal of a log ratio on a shifted outcome scale, or ``None``.

    A ratio of means changes when the outcome is shifted, and the fit scales a continuous
    outcome by ``(y - a) / (b - a)``.  The log ratio is free of the factor ``b - a`` but
    not of the shift ``a``, so it is reported on the outcome's own scale only when
    ``a = 0``.

    Parameters
    ----------
    data : CausalData
        The prepared data.
    scale : str
        The declared scale.
    q_bounds : tuple of float or None
        The declared outcome bounds.

    Returns
    -------
    Refusal or None
        An ``INVALID_DECLARATION`` record when the lower bound is not 0.
    """
    if scale != "log_ratio" or data.family == "binomial":
        return None
    if q_bounds is not None and float(q_bounds[0]) == 0.0:
        return None
    if q_bounds is None:
        observed = data.outcome[data.observed]
        start = (
            "With q_bounds=None the lower end is read off the sample (here the observed "
            f"minimum, {float(np.min(observed)):g}), so the scale the ratio is computed on "
            "would change from one sample to the next, even when that minimum is 0."
        )
    else:
        start = f"These q_bounds start at {float(q_bounds[0]):g}."
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        "A ratio of means changes when the outcome is shifted, so the log ratio is computed "
        "on the outcome's own scale, which needs q_bounds=(0, upper) for a continuous "
        f"outcome. {start}",
        alternative=(
            "Declare q_bounds=(0, upper) for a nonnegative outcome, or use scale='difference'."
        ),
    )


_MAX_NEWTON = 100
_NEWTON_TOL = 1e-13


@dataclass(frozen=True)
class EffectModification:
    """The declared projection a TMLE fit targets: a basis, the modifiers, a scale, a pair.

    Parameters
    ----------
    basis : Basis
        A known basis of the modifiers from :mod:`cleverly.bases`.
    modifiers : Modifiers
        The declared modifier schema.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    treatment : Any
        The label of the treated arm ``t``, or ``None`` for the higher arm of a binary
        treatment.
    reference : Any
        The label of the reference arm ``r``, or ``None`` for the lower arm of a binary
        treatment.
    """

    basis: Any
    modifiers: Modifiers
    scale: ContrastScale = "difference"
    treatment: Any = None
    reference: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.modifiers, Modifiers):
            raise DataError("EffectModification(modifiers=...) takes a cleverly.bases.Modifiers")
        if self.scale not in SCALES:
            raise DataError(f"scale must be 'difference' or 'log_ratio'; got {self.scale!r}")
        check_basis(self.basis)
        basis = resolve(self.basis, self.modifiers)
        unread = [c for c in basis.columns if c not in self.modifiers.columns]
        if unread:
            raise DataError(
                f"the basis reads {unread}, which are not declared modifiers "
                f"{self.modifiers.columns}"
            )
        object.__setattr__(self, "basis", basis)

    @property
    def target(self) -> str:
        """The registered target this projection is reported under."""
        return TARGET_OF_SCALE[self.scale]

    @property
    def terms(self) -> tuple[str, ...]:
        """The basis terms, one coefficient each."""
        return tuple(self.basis.terms)


def modifier_frame(data: CausalData, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The modifier columns of ``data`` as raw values, keyed by name.

    A numeric adjustment column is read from the covariates.  A categorical one that the
    data layer expanded into indicators is rebuilt from them, so a basis reads the levels
    the caller declared rather than the encoding.

    Parameters
    ----------
    data : CausalData
        The prepared data.
    columns : tuple of str
        The modifier columns.

    Returns
    -------
    dict of str to ndarray
        One array per modifier.

    Raises
    ------
    DataError
        If a modifier is not an adjustment column of ``data``.
    """
    return ModifierReader.of(data, columns)(np.asarray(data.covariates))


@dataclass(frozen=True)
class ModifierReader:
    """Read the modifier columns off a covariate matrix laid out as one fit's covariates.

    A learned rule scores rows given as the fit's covariate matrix
    (:meth:`cleverly.interventions._rule_learners.FoldRule.score`), and an EP curve reads
    the modifiers by name; this record holds the layout that maps one to the other.

    Parameters
    ----------
    names : tuple of str
        The covariate names of the fit, in matrix column order.
    encodings : tuple
        The fit's categorical encodings (:class:`~cleverly.data.CategoricalEncoding`).
    columns : tuple of str
        The modifier columns to read.
    """

    names: tuple[str, ...]
    encodings: tuple[Any, ...]
    columns: tuple[str, ...]

    @classmethod
    def of(cls, data: CausalData, columns: tuple[str, ...]) -> ModifierReader:
        """The reader of ``columns`` in the covariate layout of ``data``.

        Parameters
        ----------
        data : CausalData
            The prepared data.
        columns : tuple of str
            The modifier columns.

        Returns
        -------
        ModifierReader
            The reader.

        Raises
        ------
        DataError
            If a modifier is not an adjustment column of ``data``.
        """
        names = tuple(data.covariate_names)
        encodings = tuple(data.encodings)
        encoded = {encoding.column for encoding in encodings}
        for column in columns:
            if column not in names and column not in encoded:
                raise DataError(
                    f"the modifier {column!r} is not an adjustment column of this fit; the "
                    "conditional effect given V is identified from the outcome regression "
                    "given W only when V is part of W"
                )
        return cls(names, encodings, tuple(columns))

    def __call__(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        """The modifier values of every row of ``matrix``, keyed by name.

        Parameters
        ----------
        matrix : ndarray
            ``(m, p)`` rows in the fit's covariate layout.

        Returns
        -------
        dict of str to ndarray
            One array per modifier.
        """
        covariates = np.asarray(matrix)
        encodings = {encoding.column: encoding for encoding in self.encodings}
        out: dict[str, np.ndarray] = {}
        for column in self.columns:
            if column in self.names:
                out[column] = np.asarray(covariates[:, self.names.index(column)])
                continue
            encoding = encodings[column]
            values = np.empty(covariates.shape[0], dtype=object)
            values[:] = encoding.dropped_level
            kept = [level for level in encoding.levels if level != encoding.dropped_level]
            for level, generated in zip(kept, encoding.generated, strict=False):
                indicator = covariates[:, self.names.index(generated)] > 0.5
                values[indicator] = level
            out[column] = values
        return out


@dataclass(frozen=True)
class EffectModificationSet:
    r"""A declared projection evaluated on the data: arrays and scalars only.

    The same rule :class:`cleverly.msm.MSMSet` follows: everything a retarget reads is
    already evaluated, so a loaded result needs no callable.

    Parameters
    ----------
    terms : tuple of str
        One name per coefficient.
    design : ndarray
        ``(n, p)`` basis :math:`\varphi(V_i)`.
    arms : tuple of float
        Every arm code of the data, ascending: the arms the submodel is keyed by.
    pair : tuple of float
        The codes ``(t, r)`` of the contrasted arms.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    spec : EffectModification or None
        The declaration the design was evaluated from, which a curve reads to evaluate
        the basis at new rows.
    """

    terms: tuple[str, ...]
    design: FloatArray
    arms: tuple[float, ...]
    pair: tuple[float, float]
    scale: ContrastScale = "difference"
    spec: EffectModification | None = None

    def __post_init__(self) -> None:
        design = np.asarray(self.design, dtype=float)
        if design.ndim != 2 or design.shape[1] != len(self.terms):
            raise DataError(
                f"the basis design must have shape (n, {len(self.terms)}); got {design.shape}"
            )
        if not np.all(np.isfinite(design)):
            raise DataError("the basis design contains a non-finite value")
        if self.pair[0] == self.pair[1] or any(code not in self.arms for code in self.pair):
            raise DataError(f"the contrasted pair {self.pair} must be two arms of {self.arms}")
        object.__setattr__(self, "design", design)

    @classmethod
    def evaluate(cls, spec: EffectModification, data: CausalData) -> EffectModificationSet:
        """Evaluate ``spec``'s basis at the rows of ``data`` and resolve its pair.

        Parameters
        ----------
        spec : EffectModification
            The declared projection.
        data : CausalData
            The prepared data.

        Returns
        -------
        EffectModificationSet
            The evaluated design.

        Raises
        ------
        CapabilityError
            If the basis is not known, or the treatment is not a discrete one.
        DataError
            If a modifier is not an adjustment column, a row breaks its declaration, the
            pair is not two arms, or the design is collinear.
        """
        refusal = effect_modification_data_refusal(data)
        if refusal is not None:
            raise CapabilityError.from_refusal(refusal)
        basis = spec.basis
        frame = modifier_frame(data, tuple(spec.modifiers.columns))
        design = np.asarray(basis.evaluate(frame, spec.modifiers), dtype=float)
        pair = _resolve_pair(spec, data)
        terms = tuple(basis.terms)
        gram = design.T @ (design * np.asarray(data.weights, dtype=float)[:, None]) / data.n
        check_projection_rank(gram, terms, axis="rows of the modifiers")
        return cls(terms, design, tuple(float(c) for c in data.arm_codes), pair, spec.scale, spec)

    # --------------------------------------------------------------- access

    @property
    def n(self) -> int:
        """The number of rows."""
        return int(self.design.shape[0])

    @property
    def n_terms(self) -> int:
        """The number of coefficients."""
        return len(self.terms)

    @property
    def codes(self) -> tuple[float, ...]:
        """Coefficient codes ``(0.0, ..., p - 1.0)``, which the parameters are keyed by."""
        return tuple(float(j) for j in range(self.n_terms))

    @property
    def labels(self) -> dict[float, str]:
        """Coefficient code to term name."""
        return {float(j): term for j, term in enumerate(self.terms)}

    @property
    def needs_alternation(self) -> bool:
        """Whether the clever covariate reads ``beta``: the log-ratio scale."""
        return self.scale == "log_ratio"

    @property
    def intercept_columns(self) -> tuple[int, ...]:
        """The columns that are the intercept: named so, or one at every row."""
        out = []
        for j, term in enumerate(self.terms):
            column = self.design[:, j]
            if term == INTERCEPT or (column.size and np.allclose(column, 1.0)):
                out.append(j)
        return tuple(out)

    def _arm_index(self) -> tuple[int, int]:
        return self.arms.index(self.pair[0]), self.arms.index(self.pair[1])

    def subset(self, index: Any) -> EffectModificationSet:
        """The same projection on a row subset.

        Parameters
        ----------
        index : array-like
            Row indices or a boolean mask.

        Returns
        -------
        EffectModificationSet
            The sliced design.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(self, design=self.design[idx])

    def coefficients_at(self, beta: FloatArray | None) -> FloatArray:
        r"""``(n, K, p)`` clever-covariate numerators :math:`c_a(V)`, one block per arm.

        Parameters
        ----------
        beta : ndarray or None
            The coefficients; required on the log-ratio scale, ignored otherwise.

        Returns
        -------
        ndarray
            :math:`c_a(V_i)` at every arm, 0 at an arm outside the pair.

        Raises
        ------
        DataError
            If the scale is the log ratio and ``beta`` is ``None``.
        """
        n, p = self.design.shape
        out = np.zeros((n, len(self.arms), p))
        t, r = self._arm_index()
        if self.scale == "difference":
            out[:, t, :] = self.design
            out[:, r, :] = -self.design
            return out
        if beta is None:
            raise DataError(
                "the log-ratio projection's clever covariate reads beta, so the coefficients "
                "are solved before it is built (solve_with_projection)"
            )
        e = expit(self.design @ np.asarray(beta, dtype=float))
        out[:, t, :] = self.design * (e - 1.0)[:, None]
        out[:, r, :] = self.design * e[:, None]
        return out

    def project(self, predictions: FloatArray, weights: FloatArray) -> ProjectionFit:
        r"""Solve the projection of the arm predictions on the outcome's own scale.

        On the difference scale the closed form
        :math:`\hat\beta = M_n^{-1} P_n[\varphi\,(\bar Q_t - \bar Q_r)]`, with
        ``jacobian`` :math:`M_n`.  On the log-ratio scale damped Newton on the strictly
        convex risk, with ``jacobian``
        :math:`J_n = P_n[\varphi\varphi^\top(\bar Q_t + \bar Q_r) e (1 - e)]`, the derivative
        of :math:`U_n`.

        Parameters
        ----------
        predictions : ndarray
            ``(n, K)`` arm predictions, arms in :attr:`arms` order, on the outcome scale.
        weights : ndarray
            ``(n,)`` observation weights.

        Returns
        -------
        ProjectionFit
            The coefficients, the matrix the influence curve inverts, and the solve.

        Raises
        ------
        DataError
            If the log-ratio projection meets a negative or all-zero prediction.
        """
        q = np.asarray(predictions, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        mass = float(w.sum())
        phi = self.design
        t, r = self._arm_index()
        if self.scale == "difference":
            tau = q[:, t] - q[:, r]
            gram = (phi * w[:, None]).T @ phi / mass
            moment = (phi * w[:, None]).T @ tau / mass
            beta = np.asarray(np.linalg.solve(gram, moment), dtype=float)
            residual = (phi * w[:, None]).T @ (tau - phi @ beta) / mass
            scale = (np.abs(phi) * w[:, None]).T @ np.ones(self.n) / mass
            relative = float(np.max(np.abs(residual) / np.where(scale > 0, scale, 1.0)))
            return ProjectionFit(beta, gram, converged=True, n_iter=0, score=relative)
        qt, qr = q[:, t], q[:, r]
        if np.any(qt < 0.0) or np.any(qr < 0.0):
            raise DataError(
                "the log ratio of two means needs non-negative arm predictions; a prediction "
                "below zero means the outcome scale starts below zero (R5)"
            )
        total = qt + qr
        if float(np.sum(w * total)) <= 0.0:
            raise DataError("the log ratio is undefined: both arms predict zero everywhere")
        return _newton_log_ratio(phi, qt, total, w, mass)


def _newton_log_ratio(
    phi: FloatArray, qt: FloatArray, total: FloatArray, w: FloatArray, mass: float
) -> ProjectionFit:
    """Minimise the log-CRR risk over ``phi' beta`` by damped Newton."""

    def risk(beta: FloatArray) -> float:
        eta = phi @ beta
        return float(np.sum(w * (total * np.logaddexp(0.0, eta) - qt * eta)) / mass)

    def state(beta: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        e = expit(phi @ beta)
        u = (phi * (w * (total * e - qt))[:, None]).sum(axis=0) / mass
        jac = (phi * (w * total * e * (1.0 - e))[:, None]).T @ phi / mass
        scale = (np.abs(phi) * (w * total)[:, None]).sum(axis=0) / mass
        relative = float(np.max(np.abs(u) / np.where(scale > 0, scale, 1.0)))
        return u, jac, relative

    beta = np.zeros(phi.shape[1])
    u, jac, relative = state(beta)
    current = risk(beta)
    n_iter = 0
    while n_iter < _MAX_NEWTON and relative > _NEWTON_TOL:
        try:
            step = np.linalg.solve(jac, u)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, u, rcond=None)[0]
        length = 1.0
        while True:
            candidate = beta - length * step
            trial = risk(candidate)
            if trial <= current + 1e-15 * (1.0 + abs(current)) or length < 1e-10:
                break
            length *= 0.5
        if trial > current + 1e-15 * (1.0 + abs(current)):
            break
        beta, current = candidate, trial
        u, jac, relative = state(beta)
        n_iter += 1
    return ProjectionFit(
        beta, jac, converged=bool(relative <= 1e-10), n_iter=n_iter, score=float(relative)
    )


def _resolve_pair(spec: EffectModification, data: CausalData) -> tuple[float, float]:
    """The arm codes of ``(t, r)``: declared labels, or the two arms of a binary treatment."""
    levels = tuple(data.treatment_levels)
    if spec.treatment is None and spec.reference is None:
        if len(levels) != 2:
            raise DataError(
                f"the treatment has {len(levels)} arms {list(levels)}; a conditional contrast "
                "compares two of them, so declare treatment= and reference="
            )
        return 1.0, 0.0

    def code(label: Any, name: str) -> float:
        if label not in levels:
            raise DataError(
                f"{name}={label!r} is not a treatment level; choose from {list(levels)}"
            )
        return float(levels.index(label))

    if spec.treatment is None or spec.reference is None:
        if len(levels) != 2:
            raise DataError(
                "a conditional contrast on a multi-arm treatment needs both treatment= and "
                "reference="
            )
        if spec.treatment is not None:
            t = code(spec.treatment, "treatment")
            return t, 1.0 - t
        r = code(spec.reference, "reference")
        return 1.0 - r, r
    t, r = code(spec.treatment, "treatment"), code(spec.reference, "reference")
    if t == r:
        raise DataError("treatment= and reference= name the same arm")
    return t, r
