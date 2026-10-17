r"""The second-order bias of the plug-in :math:`\nu^2`.

The plug-in estimate :math:`\hat\nu^2 = E_n[w\,\hat\alpha^2]` is one coordinate of the root
:math:`\hat\eta` of a stacked estimating equation :math:`\sum_c Z_c(\eta) = 0`, summed over
units :math:`c` (rows, or clusters under ``id=``).  The base stack is

=====================  =========================================================  ==========
block                  row :math:`i`                                              when
=====================  =========================================================  ==========
:math:`\theta_k`       :math:`w_i\,(e_{A_i} - g_k(x_i))_{-r} \otimes x_i`,        every fold
                       on the training rows of fold :math:`k`
:math:`p`              :math:`w_i\,(1\{A_i = c\} - p)`                            ATT, ATC
:math:`\nu`            :math:`w_i\,(\alpha_i(\theta_{k(i)}, p)^2 - \nu)`,          always
                       with :math:`k(i)` the fold whose test rows hold :math:`i`
=====================  =========================================================  ==========

A fixed L2 penalty adds the deterministic :math:`-P\theta_k/C` to the :math:`\theta_k`
block.  The weights are mean one, so the :math:`\nu` and :math:`p` rows are the ratio
(Hajek) forms, and the weight mean needs no row of its own: a stack in the raw weights has
the same root and the same bias.

**The bias.**  For independent units with :math:`E Z_c(\eta^*) = 0`, Rilstone,
Srivastava and Ullah (1996, Proposition 3.2) and Newey and Smith (2004, Theorem 4.1) give
:math:`E\hat\eta - \eta^* = \beta + o(1/n)`, with

.. math::

    \beta = Q\Big\{\sum_c \nabla Z_c\, Q Z_c + \tfrac12 \sum_c \nabla^2 Z_c[V]
            + \mathrm{pen}(\eta)\Big\},\qquad
    Q = -\Big(\sum_c \nabla Z_c\Big)^{-1},\quad V = Q S Q^\top,\quad
    S = \sum_c Z_c Z_c^\top.

The sign of the curvature term is :math:`+\frac12`, which the scalar checks
:math:`\hat\theta = 1/\bar y` and :math:`\hat\theta = \log\bar y` fix.  The
:math:`\nu` coordinate of :math:`\beta` is the bias of :math:`\hat\nu^2`.  It holds the
three terms of the plan's hand derivation, :math:`D^\top\beta_\theta`,
:math:`\frac12\mathrm{tr}(GV)` and, in sample, the covariance term :math:`c`, and it holds the
share and Hajek terms too.  A cross-fitted :math:`\nu` row of a test row carries no
:math:`\theta_{k(i)}` score row, so the held-out form has no covariance term.

**The curve.**  The curve of :math:`\hat\nu^2` is :math:`\psi_i = \gamma^\top \tilde u_i`,
where :math:`\gamma = -n A^{-\top} e_\nu` is the :math:`\nu` row of the inverse Jacobian
and :math:`\tilde u_i` carries the penalty's row allocation.  The engine returns the curve
and no bias of its variance: the bound reports the plain influence variance of the curve.

**Derivatives.**  The base rows, their Jacobian-vector products and their second
directional derivatives are closed forms
(:func:`~cleverly.sensitivity._mechanism_stack.representer_g_derivative` holds the
representer's).  Each is complex-analytic in its parameters.

:class:`PluginStack` is the engine.  ``mass`` and ``strata`` let a test evaluate the
expected sums of a fixed design: each row then stands for ``mass`` independent units, and
each stratum's rows are centred before the products that need :math:`E Z_c = 0` per unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray

__all__ = ["PluginStack", "SecondOrderBias", "StackFold", "representer_derivatives"]

#: Eigenvalues of a sandwich below this fraction of the largest carry no curvature.
_EIGEN_CUTOFF = 1e-13


@dataclass(frozen=True)
class StackFold:
    """One fold's logistic mechanism, as the engine reads it.

    Parameters
    ----------
    design : ndarray
        ``(n, d)`` design of every row, intercept first when the model fits one.
    theta : ndarray
        ``(K - 1, d)`` coefficients against the first arm.
    penalty : ndarray
        ``(r, r)`` penalty matrix ``P / C`` with ``r = (K - 1) d``.
    train : ndarray
        ``(n,)`` mask of the rows the fold's model was fitted on.
    test : ndarray
        ``(n,)`` mask of the rows whose mechanism the fold predicts.
    null : ndarray or None
        ``(r, m)`` orthonormal directions of ``theta`` that the design does not identify,
        such as the difference of two equal columns; ``None`` when there are none.
    """

    design: FloatArray
    theta: FloatArray
    penalty: FloatArray
    train: BoolArray
    test: BoolArray
    null: FloatArray | None = None


@dataclass(frozen=True)
class SecondOrderBias:
    r"""The engine's output.

    Parameters
    ----------
    point : float
        :math:`\hat B`, the second-order bias of :math:`\hat\nu^2`.
    curve : ndarray
        ``(n,)`` curve :math:`\psi_i = \gamma^\top \tilde u_i` of the stack.
    root : ndarray
        ``(P,)`` second-order bias of every coordinate of the base root
        :math:`(\theta_1, \dots, \theta_F, p, \nu)`.
    """

    point: float
    curve: FloatArray
    root: FloatArray


def representer_derivatives(
    group: str, arm: int, versus: int | None, onehot: Any, g: Any, share: Any
) -> tuple[Any, Any, Any]:
    r"""The representer and its first and second derivatives in the mechanism's columns.

    Parameters
    ----------
    group : str
        ``"mean"``, ``"att"`` or ``"atc"``.
    arm : int
        Column of the parameter's arm.
    versus : int or None
        Column of the arm it is contrasted with, or ``None`` for a mean.
    onehot : ndarray
        ``(n, K)`` indicators of each row's arm.
    g : ndarray
        ``(n, K)`` mechanism, real or complex.
    share : float or complex or None
        The share of the conditioning arm, for ``"att"`` and ``"atc"``.

    Returns
    -------
    tuple of ndarray
        ``(n,)`` values of :math:`\alpha`, ``(n, K)`` values of
        :math:`b_c = \partial\alpha/\partial g_c` and ``(n, K, K)`` values of
        :math:`e_{cd} = \partial^2\alpha/\partial g_c\partial g_d`.  For the ATE contrast
        :math:`1\{A=a\}/g_a - 1\{A=b\}/g_b`, :math:`e_{aa} = 2\cdot 1\{A=a\}/g_a^3` and
        :math:`e_{bb} = -2\cdot 1\{A=b\}/g_b^3`.
    """
    n, k = g.shape
    alpha = np.zeros(n, dtype=g.dtype)
    first = np.zeros((n, k), dtype=g.dtype)
    second = np.zeros((n, k, k), dtype=g.dtype)
    if group == "mean":
        terms = [(arm, 1.0)] if versus is None else [(arm, 1.0), (versus, -1.0)]
        for j, sign in terms:
            on = onehot[:, j]
            alpha = alpha + sign * on / g[:, j]
            first[:, j] += -sign * on / g[:, j] ** 2
            second[:, j, j] += 2.0 * sign * on / g[:, j] ** 3
        return alpha, first, second
    assert versus is not None and share is not None
    i, r = arm, versus
    on_a, on_r = onehot[:, i], onehot[:, r]
    if group == "att":
        alpha = (on_a - on_r * g[:, i] / g[:, r]) / share
        first[:, i] = -on_r / (g[:, r] * share)
        first[:, r] = on_r * g[:, i] / (g[:, r] ** 2 * share)
        cross = on_r / (g[:, r] ** 2 * share)
        second[:, i, r] = cross
        second[:, r, i] = cross
        second[:, r, r] = -2.0 * on_r * g[:, i] / (g[:, r] ** 3 * share)
        return alpha, first, second
    alpha = (on_a * g[:, r] / g[:, i] - on_r) / share
    first[:, r] = on_a / (g[:, i] * share)
    first[:, i] = -on_a * g[:, r] / (g[:, i] ** 2 * share)
    cross = -on_a / (g[:, i] ** 2 * share)
    second[:, i, r] = cross
    second[:, r, i] = cross
    second[:, i, i] = 2.0 * on_a * g[:, r] / (g[:, i] ** 3 * share)
    return alpha, first, second


def _softmax(design: Any, theta: Any) -> Any:
    """``(n, K)`` probabilities with the reference column first; complex-analytic."""
    eta = np.concatenate([np.zeros((design.shape[0], 1)), design @ theta.T], axis=1)
    eta = eta - eta.real.max(axis=1, keepdims=True)
    expd = np.exp(eta)
    return expd / expd.sum(axis=1, keepdims=True)


def _training_mask(fold: StackFold) -> BoolArray:
    """The rows whose score enters the fold's block: its training rows."""
    return fold.train


def _stacks_share(conditional: bool) -> bool:
    """Whether the conditioning share is a stacked parameter."""
    return conditional


def _sandwich(q: FloatArray, meat: FloatArray, layout: Any) -> FloatArray:
    """``Q S Q'``, the variance of the stacked root in sum units."""
    return np.asarray(q @ meat @ q.T, dtype=float)


def _assemble(
    q: FloatArray, first: FloatArray, curvature: FloatArray, penalty: FloatArray, layout: Any
) -> FloatArray:
    """``Q {first + curvature / 2 + penalty}``, the Rilstone-Srivastava-Ullah bias."""
    return np.asarray(q @ (first + 0.5 * curvature + penalty), dtype=float)


@dataclass(frozen=True)
class _Layout:
    """Where each block of the base stack sits."""

    folds: int
    width: int  # r = (K - 1) d
    share: int | None
    nu: int
    size: int

    def theta(self, k: int) -> slice:
        return slice(k * self.width, (k + 1) * self.width)

    @property
    def thetas(self) -> slice:
        return slice(0, self.folds * self.width)


@dataclass
class _State:
    """The mechanism and the representer at one parameter value."""

    fold_g: list[Any]
    own: Any
    alpha: Any
    b: Any
    e: Any
    share: Any
    nu: Any


class PluginStack:
    r"""The stacked estimating equation of the plug-in :math:`\nu^2`, and its bias.

    Parameters
    ----------
    folds : sequence of StackFold
        The fitted mechanism, one entry per fold; empty for a declared known mechanism.
    treatment_index : ndarray
        ``(n,)`` column of each row's arm.
    n_arms : int
        ``K``.
    weights : ndarray
        ``(n,)`` mean-one weights.
    group : str
        ``"mean"``, ``"att"`` or ``"atc"``.
    arm : int
        Column of the parameter's arm.
    versus : int or None
        Column of the arm it is contrasted with.
    conditioning : int or None
        Column of the conditioning arm of an ATT or ATC.
    known : ndarray or None
        ``(n, K)`` declared mechanism when ``folds`` is empty.
    mass : ndarray or None
        ``(n,)`` count of units each row stands for; ``None`` for a sample.
    cluster : ndarray or None
        ``(n,)`` cluster codes; ``None`` for independent rows.
    strata : ndarray or None
        ``(n,)`` design-point labels of a fixed design, within which units are centred.
    """

    def __init__(
        self,
        folds: Any,
        treatment_index: Any,
        n_arms: int,
        weights: Any,
        group: str,
        arm: int,
        versus: int | None,
        conditioning: int | None,
        *,
        known: Any = None,
        mass: Any = None,
        cluster: Any = None,
        strata: Any = None,
    ) -> None:
        self.folds = tuple(folds)
        self.index = np.asarray(treatment_index, dtype=int)
        self.k = int(n_arms)
        self.weights = np.asarray(weights, dtype=float)
        self.n = self.index.shape[0]
        self.group = group
        self.arm = int(arm)
        self.versus = None if versus is None else int(versus)
        self.conditioning = None if conditioning is None else int(conditioning)
        self.known = None if known is None else np.asarray(known, dtype=float)
        self.mass = np.ones(self.n) if mass is None else np.asarray(mass, dtype=float)
        if cluster is not None and mass is not None:
            raise ValueError("a clustered stack has unit masses")
        self.cluster = cluster
        if cluster is None:
            self.codes = np.arange(self.n)
            self.units = self.n
        else:
            _, codes = np.unique(np.asarray(cluster), return_inverse=True)
            self.codes = codes.reshape(-1)
            self.units = int(self.codes.max()) + 1
        # A row of a fixed design stands for ``mass`` units; a cluster is one unit.
        self.unit_mass = self.mass if cluster is None else np.ones(self.units)
        self.strata = None if strata is None else np.asarray(strata)
        self.onehot = np.eye(self.k)[self.index]
        self.indicator = (
            None if self.conditioning is None else self.onehot[:, self.conditioning].copy()
        )
        d = self.folds[0].design.shape[1] if self.folds else 0
        share = _stacks_share(self.conditioning is not None)
        width = (self.k - 1) * d
        base = len(self.folds) * width
        self.layout = _Layout(
            folds=len(self.folds),
            width=width,
            share=base if share else None,
            nu=base + (1 if share else 0),
            size=base + (2 if share else 1),
        )
        self.d = d
        self.train = [np.asarray(_training_mask(f), dtype=float) for f in self.folds]
        self.test = [np.asarray(f.test, dtype=bool) for f in self.folds]
        self.weight_mass = float(self.mass @ self.weights)
        # The fixed share when it is not stacked.
        self.fixed_share = (
            None
            if self.indicator is None
            else float(self.mass @ (self.weights * self.indicator)) / self.weight_mass
        )
        self._cache: dict[Any, _State] = {}
        self.root = self._root()

    # ------------------------------------------------------------------ the base stack

    def _root(self) -> FloatArray:
        layout = self.layout
        eta = np.zeros(layout.size)
        for k, fold in enumerate(self.folds):
            eta[layout.theta(k)] = np.asarray(fold.theta, dtype=float).reshape(-1)
        if layout.share is not None:
            eta[layout.share] = self.fixed_share
        state = self._state(eta, nu_free=True)
        eta[layout.nu] = float(self.mass @ (self.weights * state.alpha**2)) / self.weight_mass
        return eta

    def _state(self, eta: Any, *, nu_free: bool = False) -> _State:
        key = (eta.dtype.str, eta.tobytes(), nu_free)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._compute_state(eta, nu_free=nu_free)
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = cached
        return cached

    def _compute_state(self, eta: Any, *, nu_free: bool) -> _State:
        layout = self.layout
        fold_g = [
            _softmax(fold.design, eta[layout.theta(k)].reshape(self.k - 1, self.d))
            for k, fold in enumerate(self.folds)
        ]
        if self.known is not None:
            own: Any = self.known.astype(eta.dtype)
        else:
            own = np.zeros((self.n, self.k), dtype=eta.dtype)
            for g, test in zip(fold_g, self.test, strict=True):
                own[test] = g[test]
        share = None
        if self.indicator is not None:
            share = eta[layout.share] if layout.share is not None else self.fixed_share
        alpha, b, e = representer_derivatives(
            self.group, self.arm, self.versus, self.onehot, own, share
        )
        nu = 0.0 if nu_free else eta[layout.nu]
        return _State(fold_g=fold_g, own=own, alpha=alpha, b=b, e=e, share=share, nu=nu)

    def _share_parts(self, state: _State) -> tuple[Any, Any, Any]:
        """``d alpha / dp``, ``d^2 alpha / dp^2`` and ``d b / dp`` for a stacked share."""
        if self.layout.share is None:
            zero = np.zeros_like(state.alpha)
            return zero, zero, np.zeros_like(state.b)
        p = state.share
        return -state.alpha / p, 2.0 * state.alpha / p**2, -state.b / p

    def rows(self, eta: Any) -> Any:
        """``(n, P)`` base rows at ``eta``; complex-analytic."""
        layout = self.layout
        state = self._state(eta)
        w = self.weights
        out = np.zeros((self.n, layout.size), dtype=eta.dtype)
        for k, (fold, g) in enumerate(zip(self.folds, state.fold_g, strict=True)):
            residual = (self.onehot - g)[:, 1:]
            score = (residual[:, :, None] * fold.design[:, None, :]).reshape(self.n, -1)
            out[:, layout.theta(k)] = (w * self.train[k])[:, None] * score
        if layout.share is not None:
            assert self.indicator is not None
            out[:, layout.share] = w * (self.indicator - state.share)
        out[:, layout.nu] = w * (state.alpha**2 - state.nu)
        return out

    def _directions(self, a: Any, k: int) -> Any:
        """``(n, K)`` linear-predictor directions of fold ``k``, reference column zero."""
        block = np.broadcast_to(a[..., self.layout.theta(k)], (self.n, self.layout.width))
        t = np.einsum("imj,ij->im", block.reshape(self.n, self.k - 1, self.d), self.folds[k].design)
        return np.concatenate([np.zeros((self.n, 1), dtype=t.dtype), t], axis=1)

    def _own(self, per_fold: list[Any], shape: tuple[int, ...], dtype: Any) -> Any:
        own = np.zeros(shape, dtype=dtype)
        for value, test in zip(per_fold, self.test, strict=True):
            own[test] = value[test]
        return own

    def jvp(self, eta: Any, a: Any) -> Any:
        """``(n, P)`` products ``grad u_i a_i``, for one direction or one per row."""
        layout = self.layout
        state = self._state(eta)
        a = np.asarray(a)
        w = self.weights
        dtype = np.result_type(eta.dtype, a.dtype)
        out = np.zeros((self.n, layout.size), dtype=dtype)
        dgs = []
        for k, (fold, g) in enumerate(zip(self.folds, state.fold_g, strict=True)):
            tf = self._directions(a, k)
            dg = g * (tf - (g * tf).sum(axis=1, keepdims=True))
            dgs.append(dg)
            block = (dg[:, 1:, None] * fold.design[:, None, :]).reshape(self.n, -1)
            out[:, layout.theta(k)] = -(w * self.train[k])[:, None] * block
        dg_own = self._own(dgs, (self.n, self.k), dtype)
        alpha_p, _, _ = self._share_parts(state)
        d_alpha = (state.b * dg_own).sum(axis=1)
        if layout.share is not None:
            a_p = np.broadcast_to(a[..., layout.share], (self.n,))
            d_alpha = d_alpha + alpha_p * a_p
            out[:, layout.share] = -w * a_p
        a_nu = np.broadcast_to(a[..., layout.nu], (self.n,))
        out[:, layout.nu] = w * (2.0 * state.alpha * d_alpha - a_nu)
        return out

    def second_directional(self, eta: Any, a: Any) -> Any:
        """``(n, P)`` second derivatives ``grad^2 u_i [a_i, a_i]``, in closed form."""
        layout = self.layout
        state = self._state(eta)
        a = np.asarray(a)
        w = self.weights
        dtype = np.result_type(eta.dtype, a.dtype)
        out = np.zeros((self.n, layout.size), dtype=dtype)
        dgs, d2gs = [], []
        for k, (fold, g) in enumerate(zip(self.folds, state.fold_g, strict=True)):
            tf = self._directions(a, k)
            centred = tf - (g * tf).sum(axis=1, keepdims=True)
            dg = g * centred
            d2g = g * centred**2 - g * (g * centred * tf).sum(axis=1, keepdims=True)
            dgs.append(dg)
            d2gs.append(d2g)
            block = (d2g[:, 1:, None] * fold.design[:, None, :]).reshape(self.n, -1)
            out[:, layout.theta(k)] = -(w * self.train[k])[:, None] * block
        dg_own = self._own(dgs, (self.n, self.k), dtype)
        d2g_own = self._own(d2gs, (self.n, self.k), dtype)
        alpha_p, alpha_pp, b_p = self._share_parts(state)
        d_alpha = (state.b * dg_own).sum(axis=1)
        d2_alpha = (state.b * d2g_own).sum(axis=1) + np.einsum(
            "ic,icd,id->i", dg_own, state.e, dg_own
        )
        if layout.share is not None:
            a_p = np.broadcast_to(a[..., layout.share], (self.n,))
            d_alpha = d_alpha + alpha_p * a_p
            d2_alpha = d2_alpha + 2.0 * a_p * (b_p * dg_own).sum(axis=1) + alpha_pp * a_p**2
        out[:, layout.nu] = w * (2.0 * d_alpha**2 + 2.0 * state.alpha * d2_alpha)
        return out

    def penalty(self, eta: Any) -> Any:
        """``(P,)`` deterministic penalty term ``-P theta_k`` of the summed equation."""
        out = np.zeros(self.layout.size, dtype=eta.dtype)
        for k, fold in enumerate(self.folds):
            out[self.layout.theta(k)] = -fold.penalty @ eta[self.layout.theta(k)]
        return out

    def penalty_jacobian(self) -> FloatArray:
        out = np.zeros((self.layout.size, self.layout.size))
        for k, fold in enumerate(self.folds):
            out[self.layout.theta(k), self.layout.theta(k)] = -fold.penalty
        return out

    # ----------------------------------------------------------------------- the sums

    def _units(self, rows: Any) -> Any:
        if self.cluster is None:
            return rows
        out = np.zeros((self.units, *rows.shape[1:]), dtype=rows.dtype)
        np.add.at(out, self.codes, rows)
        return out

    def _centred(self, units: Any) -> Any:
        """Units minus their stratum's mean, for a fixed design; unchanged otherwise."""
        if self.strata is None:
            return units
        out = np.array(units, copy=True)
        for label in np.unique(self.strata):
            chosen = self.strata == label
            weight = self.mass[chosen]
            out[chosen] -= (weight @ units[chosen]) / weight.sum()
        return out

    def _summed(self, rows: Any) -> Any:
        """``sum_i mass_i rows_i``."""
        return self.mass @ rows

    def information(self, eta: Any) -> FloatArray:
        """``A = sum_c grad Z_c + grad pen``, in closed form."""
        size = self.layout.size
        out = np.zeros((size, size))
        for j in range(size):
            out[:, j] = self._summed(self.jvp(eta, np.eye(size)[j])).real
        return out + self.penalty_jacobian()

    def _null_projector(self) -> FloatArray:
        """The projector onto the unidentified directions of the stacked blocks."""
        size = self.layout.size
        out = np.zeros((size, size))
        for k, fold in enumerate(self.folds):
            if fold.null is None or fold.null.size == 0:
                continue
            block = self.layout.theta(k)
            out[block, block] = fold.null @ fold.null.T
        return out

    def _q(self, information: FloatArray) -> FloatArray:
        """``-A^+``: the inverse on the identified directions, zero on the others.

        The rows do not move along an unidentified direction, so that direction is a right
        and a left null space of ``A``, and ``(A - Pi)^{-1} + Pi`` is its pseudo-inverse.
        """
        projector = self._null_projector()
        return np.asarray(-(np.linalg.inv(information - projector) + projector), dtype=float)

    def _eigen(self, variance: FloatArray) -> list[tuple[float, FloatArray]]:
        values, vectors = np.linalg.eigh((variance + variance.T) / 2.0)
        top = float(values.max()) if values.size else 0.0
        return [
            (float(value), vectors[:, j])
            for j, value in enumerate(values)
            if value > _EIGEN_CUTOFF * max(top, 0.0)
        ]

    def _allocation(self, eta: Any) -> Any:
        """``(n, P)`` row allocation of the penalty, ``(w_i / sum w) P theta``."""
        return np.outer(self.weights / self.weight_mass, self.penalty(eta))

    # ------------------------------------------------------------------------ the bias

    def bias(self) -> SecondOrderBias:
        """The second-order bias of the point estimate.

        Returns
        -------
        SecondOrderBias
            The bias, the stack's curve and the bias of every root coordinate.
        """
        layout = self.layout
        size = layout.size
        eta = self.root
        base_units = self._units(self.rows(eta))
        information = self.information(eta)
        q = self._q(information)
        centred = self._centred(base_units)
        meat = (centred * self.unit_mass[:, None]).T @ centred
        sandwich = _sandwich(q, meat, layout)
        directions = (centred @ q.T)[self.codes]
        first = self._summed(self.jvp(eta, directions)).real
        curvature = np.zeros(size)
        for value, vector in self._eigen(sandwich):
            curvature += value * self._summed(self.second_directional(eta, vector)).real
        beta = _assemble(q, first, curvature, self.penalty(eta), layout)
        gamma = self.weight_mass * (q.T @ np.eye(size)[layout.nu])
        curve = (self.rows(eta) + self._allocation(eta)) @ gamma
        return SecondOrderBias(float(beta[layout.nu]), np.asarray(curve), beta)

    def root_variance(self) -> FloatArray:
        """``(P, P)`` sandwich variance of the base root, in sum units."""
        eta = self.root
        q = self._q(self.information(eta))
        centred = self._centred(self._units(self.rows(eta)))
        meat = (centred * self.unit_mass[:, None]).T @ centred
        return _sandwich(q, meat, self.layout)
