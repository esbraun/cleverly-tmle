r"""Clever covariates and the parametric submodels they index.

The targeting step moves the initial outcome regression :math:`\bar Q^0` along a
one-dimensional (or here, low-dimensional) parametric submodel whose score at
:math:`\epsilon = 0` equals the efficient influence function of the target
parameter.  Solving for :math:`\epsilon` therefore makes the estimator solve the
estimated efficient score equation :math:`P_n D^* = 0`.

That is what the double-robustness and efficiency arguments are *built on*, not a
guarantee they hold: those follow from the second-order remainder of the von Mises
expansion, which is a product of the two nuisance errors and so vanishes when either
one does.  Solving the score equation is the step that makes the remainder the only
thing left over -- see :mod:`cleverly.estimators.tmle` for the conditions each
guarantee needs, and ``tests/unit/test_remainder.py`` for the product form checked
exactly.

For a logistic fluctuation the submodel is

.. math::

    \operatorname{logit} \bar Q^*_\epsilon(a, W)
      = \operatorname{logit} \bar Q^0(a, W) + \epsilon^\top h(a, W)

where :math:`h` is the *clever covariate*.  Its form depends on the target:

``mean`` (used for ``EY1``, ``EY0``, ``ATE``, ``RR``, ``OR``)
    One column per arm, :math:`h_a(A, W) = \mathbb 1\{A = a\} / (g_a(W)\,\pi_a(W))` --
    two of them for a binary treatment, ``K`` for a ``K``-armed one.  Fitting every
    coefficient solves the score equation for each counterfactual mean separately, which
    is what makes the risk ratio and odds ratio available in addition to the difference,
    and what makes any contrast across ``K`` arms a functional of one targeted
    distribution rather than a fluctuation of its own.

``att`` / ``atc``
    A single column contrasting the arms, with the control arm reweighted by the
    propensity odds -- see :func:`att_submodel`.

``regime``
    One column per *intervention*, :math:`h_r(A, W) = g^\star_r(A \mid W) / g(A \mid W)`
    -- a treatment rule or a stochastic assignment rather than a constant arm.  It is the
    same object as ``mean`` when every regime is static: one column of ``mean`` is one
    column of ``regime`` with :math:`g^\star_r = \mathbb 1\{a = v\}`.  The two are kept
    apart because the *parameters* move from the arms to the regimes -- see
    :func:`regime_submodel` for why that separation cannot be expressed by re-keying.

Here :math:`\pi_a(W) = P(\Delta = 1 \mid A = a, W)` is the probability that the
outcome is observed; with no missingness it is one and drops out.  Note the
:math:`\Delta` indicator itself does *not* appear in :math:`h`: it enters by
restricting the fluctuation regression to rows with an observed outcome.  That is
not an approximation but an identity, since

.. math::

    \sum_{i:\,\Delta_i = 1} h_i\,(Y_i - \bar Q^*_i)
      \;=\; \sum_i \Delta_i\, h_i\,(Y_i - \bar Q^*_i),

which is why :func:`~cleverly.fluctuation.iterative._score` averages over all ``n``
rows rather than over the observed ones: an unobserved row contributes a genuine zero,
not a missing value.

**What double robustness means once :math:`\pi` is in the denominator.**  The obvious
generalisation -- "consistent if any one of the three nuisances is right" -- is false.
Only the *product* :math:`g_a \pi_a` appears in the estimating equation, so the
remainder of the von Mises expansion is

.. math::

    R_2 = \int \Bigl(\frac{g_0 \pi_0}{\hat g \hat\pi} - 1\Bigr)
              (\bar Q_0 - \bar Q)\, dP_0 ,

and the guarantee is: consistent if :math:`\bar Q` is right, **or** if the product
:math:`g\,\pi` is right.  A correct propensity buys nothing on its own when the
missingness model is wrong; conversely, errors in the two mechanisms can cancel
exactly.  ``tests/unit/test_remainder_mar.py`` checks both statements at machine
precision, and ``tests/unit/test_influence_gateaux_mar.py`` checks that the influence
curve above is the efficient one for the observed-data model.

A controlled direct effect puts a third factor, :math:`q_z(a, W) = P(Z = z \mid A = a,
W)`, in the same denominator, and the statement generalises the same way rather than
gaining a third independent half: consistent if :math:`\bar Q` is right, **or** if the
product :math:`g\,q_z\,\pi` is right.  Everything above holds with :math:`\pi` replaced
by :math:`q_z\,\pi` throughout; :mod:`cleverly.estimators.direct_effect` derives it,
along with the parameter and the assumptions that identify it.

**Truncation.**  ``g``, :math:`\pi` and the intermediate density are all bounded away
from zero before they enter :math:`h`.  This regularises estimation; it does not
redefine the target.  The plug-in is an average of targeted predictions and contains no
mechanism at all, so no bound can move :math:`\Psi` -- what a binding bound moves is
:math:`R_2`, by exactly the formula above evaluated at the truncated value.  The trade
is variance for second-order bias, and ``res.sensitivity.truncation_curve()`` (with
``mechanism=True`` for :math:`\pi`) is how to see how much of it you are paying.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray

__all__ = [
    "SUBMODEL_BUILDERS",
    "Submodel",
    "SubmodelBuilder",
    "TargetGroup",
    "atc_submodel",
    "att_submodel",
    "check_arms",
    "mean_submodel",
    "mtp_submodel",
    "regime_submodel",
    "register_submodel",
    "restrict",
    "submodel_for",
]

#: Which fluctuation a target's score equation needs.  A plain ``str`` rather than a
#: ``Literal``, because the set is a registry the caller can extend -- see
#: :func:`register_submodel`.  It is validated against :data:`SUBMODEL_BUILDERS` at the
#: two places it matters, building a submodel and registering a target.
TargetGroup = str


def check_arms(observed: FloatArray, arms: Mapping[float, FloatArray], what: str) -> None:
    """Validate an arm mapping against the observed array it accompanies.

    Shared by :class:`Submodel` and :class:`~cleverly.fluctuation.iterative.InitialFit`,
    which key their counterfactual quantities the same way and so can get them wrong the
    same way.  The key check is not pedantry: ``arms[1]`` and ``arms[1.0]`` are different
    dictionary entries, so an integer key would silently create a second arm that no
    lookup ever finds.
    """
    if not arms:
        raise ValueError(f"a {what} needs at least one counterfactual arm")
    for level, values in arms.items():
        if not isinstance(level, float):
            raise TypeError(
                f"{what} arm keys must be floats -- the treatment level the arm sets -- "
                f"but got {level!r} of type {type(level).__name__}"
            )
        if np.asarray(values).shape != np.asarray(observed).shape:
            raise ValueError(
                f"{what} arm {level} has shape {np.asarray(values).shape}, but its observed "
                f"array has shape {np.asarray(observed).shape}"
            )


@dataclass(frozen=True)
class Submodel:
    """Clever covariates evaluated at the observed treatment and at each arm.

    Attributes
    ----------
    observed:
        ``(n, k)`` covariate at the treatment each unit actually received; this is
        what the fluctuation regression uses.
    arms:
        Maps a treatment level to the ``(n, k)`` covariate obtained by setting the
        treatment to that level for everybody, so a binary treatment carries
        ``{0.0: ..., 1.0: ...}``.  Applying the fitted ``epsilon`` to these gives the
        targeted counterfactual predictions.

        Keyed rather than named as two fields so that every routine which moves or
        subsets a submodel -- :meth:`map_arms`, :func:`restrict`, :func:`weighted_form` --
        is written once and does not count arms.  The mapping is not copied
        defensively, on the same terms as the arrays it holds.
    names:
        Column labels, for reporting ``epsilon`` back to the user.  Their order is the
        column order, which :attr:`arm_columns` refers to.
    group:
        Which estimand family this submodel targets; a key of :data:`SUBMODEL_BUILDERS`.
    arm_columns:
        For a submodel that fits *one column per arm*, the column of ``observed`` whose
        coefficient targets each arm -- ``{0.0: 0, 1.0: 1}`` for :func:`mean_submodel`.

        **Empty for a contrast submodel** (``att``, ``atc``), where a single column
        targets a difference of arms and no column belongs to one of them.  That
        distinction is why this is a mapping rather than an assumed column order: it is
        what lets the influence curves index a column by the arm it targets instead of
        by the literal ``0`` and ``1`` that only a two-arm submodel has.
    """

    observed: FloatArray
    arms: dict[float, FloatArray]
    names: tuple[str, ...]
    group: TargetGroup
    arm_columns: dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_arms(self.observed, self.arms, "submodel")
        if self.observed.shape[1] != len(self.names):
            raise ValueError(
                f"{self.observed.shape[1]} covariate column(s) but {len(self.names)} name(s)"
            )
        for level, column in self.arm_columns.items():
            if level not in self.arms:
                raise ValueError(
                    f"arm_columns names level {level}, which is not one of the submodel's "
                    f"arms {sorted(self.arms)}"
                )
            if not 0 <= column < self.observed.shape[1]:
                raise ValueError(
                    f"arm_columns maps level {level} to column {column}, outside the "
                    f"{self.observed.shape[1]} column(s) this submodel has"
                )

    def map_arms(self, fn: Callable[[FloatArray], FloatArray]) -> Submodel:
        """Apply ``fn`` to the observed covariate and to every arm's, keys preserved."""
        return Submodel(
            fn(self.observed),
            {level: fn(values) for level, values in self.arms.items()},
            self.names,
            self.group,
            dict(self.arm_columns),
        )

    def column_for(self, level: float) -> FloatArray:
        """The observed covariate column whose coefficient targets ``level``.

        Raises for a contrast submodel, where no single column belongs to one arm; the
        influence curves for those estimands index the sole column directly.
        """
        try:
            column = self.arm_columns[level]
        except KeyError:
            raise KeyError(
                f"the {self.group!r} submodel has no column dedicated to arm {level}; it "
                f"has arm columns for {sorted(self.arm_columns)}"
            ) from None
        return np.asarray(self.observed[:, column], dtype=float)

    @property
    def levels(self) -> tuple[float, ...]:
        """The arm levels, ascending."""
        return tuple(sorted(self.arms))

    @property
    def n(self) -> int:
        return int(self.observed.shape[0])

    @property
    def dim(self) -> int:
        return int(self.observed.shape[1])

    @property
    def max_abs(self) -> float:
        """Largest absolute clever-covariate value.

        A large value is the signature of a practical positivity violation: a
        single unit whose inverse propensity weight dominates the estimating
        equation.  :mod:`cleverly.sensitivity.positivity` reports it.
        """
        if self.observed.size == 0:
            return 0.0
        return float(np.max(np.abs(self.observed)))


def _arm_matrix(n: int, k: int, probabilities: FloatArray | None, label: str) -> FloatArray:
    """Validate an ``(n, K)`` arm-indexed probability array, or ones when absent.

    Ones rather than ``None`` so the callers multiply unconditionally: a mechanism that
    does not apply contributes a factor of one to every arm's denominator, which is an
    identity rather than a special case.
    """
    if probabilities is None:
        return np.ones((n, k))
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (n, k):
        raise ValueError(f"{label} must have shape ({n}, {k}); got {probs.shape}")
    if np.any(probs <= 0):
        raise ValueError(f"{label} must be strictly positive after bounding")
    return probs


def _selection_indicator(n: int, selection: FloatArray | None) -> FloatArray:
    """``1`` when the unit's realised intermediate equals the targeted value."""
    if selection is None:
        return np.ones(n)
    indicator = np.asarray(selection, dtype=float).reshape(-1)
    if indicator.shape[0] != n:
        raise ValueError(f"selection has length {indicator.shape[0]}, expected {n}")
    return indicator


def mean_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
) -> Submodel:
    r"""One column per arm, targeting every counterfactual mean at once.

    .. math::

        h_a(A, W) = \frac{\mathbb 1\{A = a\}}
                         {g_a(W)\,\pi_a(W)\,q_a(W)}

    Fitting all ``K`` coefficients solves the score equation for each counterfactual mean
    separately, which is what makes any contrast of them -- a difference against a
    reference arm, a risk ratio, a linear combination across a dose -- available from one
    targeted distribution rather than one fluctuation each.

    Parameters
    ----------
    treatment:
        Arm codes, length ``n``; see
        :attr:`~cleverly.data.CausalData.arm_codes`.
    propensity:
        ``(n, K)`` array of ``g(a | W) = P(A = a | W)``, columns in ``arms`` order and
        already truncated away from zero.  Rows need **not** sum to one: with more than
        two arms the truncation is applied arm by arm and deliberately not renormalised,
        for the reasons :meth:`~cleverly.estimators._nuisance.Propensity.bounded` sets
        out.  Only ``g_a`` enters arm ``a``'s column, so the sum never appears.
    arms:
        The arm codes ``propensity``'s columns are keyed by, ascending.  Defaults to the
        two-arm case so that the many places constructing a binary submodel directly need
        not repeat it.
    treated_fraction:
        Accepted and ignored.  Every builder in :data:`SUBMODEL_BUILDERS` takes the same
        keyword-only signature so the registry can dispatch without knowing which
        arguments each one happens to need; the counterfactual means do not condition on
        an arm and so have no use for the treated share.
    missingness:
        Optional ``(n, K)`` array of ``P(Delta = 1 | A = a, W)`` per arm.
    intermediate_density, selection:
        For a controlled direct effect at intermediate value ``z``:
        ``P(Z = z | A = a, W)`` per arm, and the indicator ``1{Z_i = z}``.  The
        indicator multiplies only the *observed* covariate -- the counterfactual
        columns are already evaluated at ``Z = z`` by construction.
    """
    del treated_fraction, regimes, shifts  # see the parameters' docstrings
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)
    g = np.asarray(propensity, dtype=float)
    if g.ndim == 1:
        # The two-arm convenience form: a bare ``g1`` vector, with arm 0 the complement.
        # Kept because the sensitivity code, the C-TMLE search and the tests all build a
        # binary submodel from one vector, and spelling out both columns at every one of
        # those call sites would be noise.
        if k != 2:
            raise ValueError(
                f"propensity was given as a single vector but there are {k} arms {list(arms)}; "
                "supply the (n, K) mechanism"
            )
        g = np.column_stack([1.0 - g.reshape(-1), g.reshape(-1)])
    if g.shape != (n, k):
        raise ValueError(f"propensity must have shape ({n}, {k}); got {g.shape}")
    if np.any(g <= 0) or np.any(g >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")

    pi = _arm_matrix(n, k, missingness, "missingness probabilities")
    pz = _arm_matrix(n, k, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)

    inverse = [1.0 / (g[:, j] * pi[:, j] * pz[:, j]) for j in range(k)]
    zeros = np.zeros(n)
    return Submodel(
        np.column_stack([(a == arm) * keep * inverse[j] for j, arm in enumerate(arms)]),
        {
            arm: np.column_stack(
                [inverse[j] if i == j else zeros for i in range(k)],
            )
            for j, arm in enumerate(arms)
        },
        tuple(f"h{arm:g}" for arm in arms),
        "mean",
        # One column per arm, in the order ``names`` gives.
        {arm: j for j, arm in enumerate(arms)},
    )


def _regime_densities(n: int, k: int, regimes: FloatArray | None) -> FloatArray:
    """Validate the ``(n, K, R)`` regime densities the ``regime`` submodel needs.

    Shape and non-negativity only.  That each row of each regime sums to one is checked
    where the regime is *built* --
    :meth:`cleverly.interventions.RegimeSet.evaluate` -- rather than repeated here: it is
    a property of the intervention, the tolerance belongs with it, and this runs once per
    truncation bound in a sensitivity sweep.
    """
    if regimes is None:
        raise ValueError(
            "the 'regime' submodel needs regimes=: an (n, K, R) array of g*(a | W) per "
            "regime. Build one with cleverly.interventions.RegimeSet.evaluate."
        )
    values = np.asarray(regimes, dtype=float)
    if values.ndim != 3 or values.shape[:2] != (n, k):
        raise ValueError(
            f"regimes must have shape ({n}, {k}, R) -- rows, arms, regimes -- got {values.shape}"
        )
    if values.shape[2] == 0:
        raise ValueError("regimes must contain at least one regime")
    if np.any(values < 0.0):
        raise ValueError("regime densities must be non-negative")
    return values


def regime_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
) -> Submodel:
    r"""One column per *regime*, targeting :math:`E[Y^{g^\star_r}]` for each.

    .. math::

        h_r(A, W) = \frac{g^\star_r(A \mid W)}
                         {g(A \mid W)\,\pi(W)\,q(W)}
                  = \sum_a \mathbb 1\{A = a\}\,
                    \frac{g^\star_r(a \mid W)}{g_a(W)\,\pi_a(W)\,q_a(W)}

    the Riesz representer of :math:`\Psi_r(P) = E_W \sum_a g^\star_r(a \mid W)
    \bar Q(a, W)`.  With a :class:`~cleverly.interventions.Static` regime
    :math:`g^\star_r(a \mid W) = \mathbb 1\{a = v\}` and this is one column of
    :func:`mean_submodel` -- which is the sense in which the arm-keyed path was always a
    special case, and what ``tests/unit/test_regime_submodel.py`` checks against it
    entry by entry.

    **Why the columns are regimes and the arms are still arms.**  The fluctuation has to
    update :math:`\bar Q(a, W)` at every arm, because the plug-in evaluates a *mixture*
    over them; but the score equations being solved are one per regime.  So
    :attr:`Submodel.arms` is keyed by arm and each of its entries has ``R`` columns --
    the covariate that arm's prediction is fluctuated by -- while the parameters live on
    the columns.  :attr:`Submodel.arm_columns` is left empty for exactly the reason
    :func:`att_submodel` leaves it empty: no column belongs to a single arm.

    Parameters
    ----------
    regimes:
        ``(n, K, R)``: ``regimes[i, j, r]`` is :math:`g^\star_r(\text{arms}[j] \mid W_i)`.
    treated_fraction:
        Accepted and ignored; see :func:`mean_submodel`.
    """
    del treated_fraction, shifts  # see the parameter's docstring
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)
    g = np.asarray(propensity, dtype=float)
    if g.ndim == 1:
        if k != 2:
            raise ValueError(
                f"propensity was given as a single vector but there are {k} arms {list(arms)}; "
                "supply the (n, K) mechanism"
            )
        g = np.column_stack([1.0 - g.reshape(-1), g.reshape(-1)])
    if g.shape != (n, k):
        raise ValueError(f"propensity must have shape ({n}, {k}); got {g.shape}")
    if np.any(g <= 0) or np.any(g >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")

    star = _regime_densities(n, k, regimes)
    pi = _arm_matrix(n, k, missingness, "missingness probabilities")
    pz = _arm_matrix(n, k, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)

    # (n, K): the denominator arm by arm, exactly as mean_submodel builds it.
    inverse = 1.0 / (g * pi * pz)
    # (n, K, R) -> (n, R) per arm: the covariate that arm's prediction is fluctuated by.
    weighted = star * inverse[:, :, None]
    counterfactual = {float(arm): weighted[:, j, :] for j, arm in enumerate(arms)}
    indicator = np.column_stack([(a == arm) for arm in arms]).astype(float)
    observed = keep[:, None] * np.einsum("ij,ijr->ir", indicator, weighted)

    return Submodel(
        observed,
        counterfactual,
        tuple(f"h_regime{r}" for r in range(star.shape[2])),
        "regime",
        # No arm_columns: a column targets a regime, which is a distribution over the
        # arms rather than one of them.
    )


def mtp_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
) -> Submodel:
    r"""One column per *shift*, targeting :math:`E[\bar Q(d_\delta(A, W), W)]` for each.

    .. math::

        h_r(a, W) = \frac{g(a - \delta_r \mid W)}{g(a \mid W)}
                    + \mathbb 1\{a > u_r - \delta_r\}

    -- see :mod:`cleverly.interventions.shift` for the derivation, the sanity checks it
    satisfies, and why this is not the ``regime`` fluctuation at the induced density.

    ``shifts`` is the ``(n, S, S)`` array
    :math:`h_r(d_s(A_i, W_i), W_i)` **stacked with** the ``(n, S)`` covariate at the
    observed treatment; both are read off
    :class:`~cleverly.interventions.shift.ShiftSet`, which computed them from one stored
    conditional density.  ``propensity`` is ignored: a continuous treatment has no
    per-arm mechanism, and its ``(n, 0)`` propensity carries no information.

    Unlike ``regime``, ``arm_columns`` is **populated**: column ``s`` really does target
    one parameter, the mean under shift ``s``, so
    :meth:`~Submodel.column_for` can answer and
    :func:`~cleverly.inference.influence.shift_means` reads it.
    """
    del propensity, arms, treated_fraction, missingness, intermediate_density, regimes
    if shifts is None:
        raise ValueError(
            "the 'mtp' submodel needs shifts=: the clever covariate evaluated at the "
            "observed treatment and at each shifted one. Build one with "
            "cleverly.interventions.ShiftSet.evaluate."
        )
    stacked = np.asarray(shifts, dtype=float)
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    if stacked.ndim != 3 or stacked.shape[0] != n or stacked.shape[1] != stacked.shape[2] + 1:
        raise ValueError(
            "the 'mtp' submodel needs shifts= of shape (n, S + 1, S): the covariate at "
            f"the observed treatment stacked above the one at each shifted treatment. "
            f"Got {stacked.shape} for {n} rows."
        )
    keep = _selection_indicator(n, selection)
    observed = keep[:, None] * stacked[:, 0, :]
    counterfactual = {
        float(index): keep[:, None] * stacked[:, index + 1, :] for index in range(stacked.shape[2])
    }
    return Submodel(
        observed,
        counterfactual,
        tuple(f"h_shift{r}" for r in range(stacked.shape[2])),
        "mtp",
        {float(index): index for index in range(stacked.shape[2])},
    )


def _binary_margin(
    propensity: FloatArray, arms: tuple[float, ...], group: str
) -> tuple[FloatArray, FloatArray]:
    r"""``(g_0, g_1)`` for a submodel that is defined only against a single contrast.

    The conditional-effect submodels reweight one arm by the propensity *odds*
    :math:`g_1 / g_0`, and an odds needs exactly two arms to be an odds.  With more, "the
    effect among the treated" does not name one parameter -- there is a separate
    conditional effect for each non-reference arm, each with its own contrast submodel --
    so this refuses rather than silently taking arms 0 and 1 and reporting an answer for
    a subset of the data nobody asked about.
    """
    if len(arms) != 2:
        raise ValueError(
            f"the {group!r} submodel needs a binary treatment; got {len(arms)} arms "
            f"{list(arms)}. It reweights one arm by the propensity odds g1 / g0, which is "
            "only an odds with two arms. Estimate the counterfactual means and take the "
            "contrast you want with result.contrast()."
        )
    g = np.asarray(propensity, dtype=float)
    one = g.reshape(-1) if g.ndim == 1 else g[:, arms.index(1.0)]
    if np.any(one <= 0) or np.any(one >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")
    return 1.0 - one, one


def att_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
) -> Submodel:
    r"""One-column submodel targeting the ATT.

    .. math::

        h(a, W) = \frac{1}{P(A = 1)}
                  \left( \frac{\mathbb 1\{a = 1\}}{\pi_1(W)}
                       - \frac{\mathbb 1\{a = 0\}}{\pi_0(W)}
                         \frac{g_1(W)}{g_0(W)} \right)

    The treated arm needs no reweighting -- the ATT conditions on ``A = 1``, so the
    treated outcomes are already drawn from the target population.  Control units
    are reweighted by the propensity odds ``g_1 / g_0`` to make them resemble the
    treated, which is why the ATT is far more sensitive than the ATE to small
    ``g_0(W)``; it is also why ``g_bounds="auto"`` uses a more conservative bound
    for this estimand.
    """
    del regimes, shifts  # accepted and ignored; the ATT conditions on an arm, not a regime
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    share = _required_treated_fraction(treated_fraction, "att")
    g0, g1 = _binary_margin(propensity, arms, "att")
    pi = _arm_matrix(n, 2, missingness, "missingness probabilities")
    pz = _arm_matrix(n, 2, intermediate_density, "intermediate probabilities")
    pi0, pi1 = pi[:, 0], pi[:, 1]
    pz0, pz1 = pz[:, 0], pz[:, 1]
    keep = _selection_indicator(n, selection)

    treated_term = 1.0 / (share * pi1 * pz1)
    control_term = (g1 / g0) / (share * pi0 * pz0)

    return Submodel(
        (keep * (a * treated_term - (1.0 - a) * control_term)).reshape(-1, 1),
        {1.0: treated_term.reshape(-1, 1), 0.0: (-control_term).reshape(-1, 1)},
        ("h_att",),
        "att",
        # No arm_columns: the single column targets a contrast of the arms, so no column
        # belongs to one of them.
    )


def atc_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
) -> Submodel:
    """One-column submodel targeting the ATC -- the mirror image of the ATT."""
    del regimes, shifts  # accepted and ignored, as in att_submodel
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    # In (0, 1) because the treated share is, which the helper has already enforced --
    # so there is no second range check here.
    control_fraction = 1.0 - _required_treated_fraction(treated_fraction, "atc")
    g0, g1 = _binary_margin(propensity, arms, "atc")
    pi = _arm_matrix(n, 2, missingness, "missingness probabilities")
    pz = _arm_matrix(n, 2, intermediate_density, "intermediate probabilities")
    pi0, pi1 = pi[:, 0], pi[:, 1]
    pz0, pz1 = pz[:, 0], pz[:, 1]
    keep = _selection_indicator(n, selection)

    control_term = 1.0 / (control_fraction * pi0 * pz0)
    treated_term = (g0 / g1) / (control_fraction * pi1 * pz1)

    return Submodel(
        (keep * (a * treated_term - (1.0 - a) * control_term)).reshape(-1, 1),
        {1.0: treated_term.reshape(-1, 1), 0.0: (-control_term).reshape(-1, 1)},
        ("h_atc",),
        "atc",
    )


def _required_treated_fraction(treated_fraction: float | None, group: str) -> float:
    """The treated share, for a builder that cannot work without one.

    The uniform builder signature makes ``treated_fraction`` optional at the type level
    even though the conditional-effect submodels require it, so the requirement is
    enforced here instead of by the dispatcher.  That is deliberate: the dispatcher no
    longer knows which builders need what, and a builder that silently substituted a
    default would report an ATT against a population nobody specified.
    """
    if treated_fraction is None:
        raise ValueError(f"the {group!r} submodel needs treated_fraction")
    if not 0.0 < treated_fraction < 1.0:
        raise ValueError(f"treated_fraction must lie in (0, 1); got {treated_fraction}")
    return float(treated_fraction)


#: What a submodel builder looks like from the registry's side.  Every builder takes the
#: treatment, the truncated per-arm propensity, and the same five keyword arguments --
#: ignoring the ones it has no use for -- so that :func:`submodel_for` can dispatch on the
#: group name alone.  ``arms`` joined that signature when the treatment stopped being
#: binary: a builder cannot key its output by arm without being told which arms there are,
#: and inferring them from the observed treatment would go wrong on exactly the subsample
#: that is missing one.
SubmodelBuilder = Callable[..., Submodel]

#: The registered fluctuations, in report order.  A group is a *score equation*, not an
#: estimand: several targets share one (see :mod:`cleverly.targets`), and adding a target
#: to the registry there does not add a fluctuation here.
SUBMODEL_BUILDERS: dict[str, SubmodelBuilder] = {}


def register_submodel(
    group: str, builder: SubmodelBuilder, *, replace: bool = False
) -> SubmodelBuilder:
    """Add a clever-covariate builder to the registry.

    Mirrors :func:`cleverly.targets.register`, including the refusal to shadow an
    existing entry without saying so: replacing the ``"mean"`` fluctuation silently would
    change what most of the built-in estimands report.

    A ``builder`` must accept the keyword arguments in :data:`SubmodelBuilder` and return
    a :class:`Submodel` whose ``group`` equals ``group`` -- the latter is checked, since a
    mismatch would send the influence curves looking for the wrong estimand family.
    """
    if group in SUBMODEL_BUILDERS and not replace:
        raise ValueError(
            f"a submodel builder for group {group!r} is already registered; pass "
            "replace=True to override it deliberately"
        )
    SUBMODEL_BUILDERS[group] = builder
    return builder


register_submodel("mean", mean_submodel)
register_submodel("att", att_submodel)
register_submodel("atc", atc_submodel)
register_submodel("regime", regime_submodel)
register_submodel("mtp", mtp_submodel)


#: Keyword arguments that joined :data:`SubmodelBuilder`'s signature after it was first
#: documented, and what to tell the author of a builder that predates each.  A missing one
#: surfaces as a bare ``unexpected keyword argument`` from deep inside the dispatcher,
#: which says nothing about the fix; these do.
_SIGNATURE_ADDITIONS: dict[str, str] = {
    "arms": (
        "Every submodel builder now takes the arm codes its columns are keyed by, because "
        "a treatment may have more than two arms; add 'arms=(0.0, 1.0)' to its "
        "keyword-only parameters and index the (n, K) propensity with it."
    ),
    "regimes": (
        "Every submodel builder now takes the regime densities, because an intervention "
        "may be a rule or a stochastic assignment rather than a constant arm; add "
        "'regimes=None' to its keyword-only parameters. A builder that targets arms "
        "rather than regimes should accept and ignore it, as mean_submodel does."
    ),
    "shifts": (
        "Every submodel builder now takes the shift clever covariates, because a "
        "treatment may be continuous and its intervention a modified treatment policy "
        "rather than anything indexed by an arm; add 'shifts=None' to its keyword-only "
        "parameters. A builder that targets arms or regimes should accept and ignore it, "
        "as mean_submodel does."
    ),
}


def submodel_for(
    group: TargetGroup,
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
) -> Submodel:
    """Build the clever covariate for an estimand family, by registry lookup.

    Replaces an ``if/elif`` chain that hardcoded the three built-in groups, and with it
    the last place a new fluctuation had to be threaded through by hand.  A group the
    registry does not know is an error naming what it does know, which is strictly more
    useful than the previous fixed list.
    """
    try:
        builder = SUBMODEL_BUILDERS[group]
    except KeyError:
        raise ValueError(
            f"unknown target group {group!r}; registered groups are "
            f"{sorted(SUBMODEL_BUILDERS)}. Use register_submodel() to add one."
        ) from None
    try:
        submodel = builder(
            treatment,
            propensity,
            arms=arms,
            treated_fraction=treated_fraction,
            missingness=missingness,
            intermediate_density=intermediate_density,
            selection=selection,
            regimes=regimes,
            shifts=shifts,
        )
    except TypeError as error:
        # Some keywords are newer than the extension point, so a builder written against
        # an older signature fails here with a bare "unexpected keyword argument". Saying
        # what to add is worth the branch: the builder is user code the library cannot fix.
        message = str(error)
        for keyword, fix in _SIGNATURE_ADDITIONS.items():
            if keyword in message:
                raise TypeError(
                    f"the builder registered for group {group!r} does not accept {keyword!r}. {fix}"
                ) from error
        raise
    if submodel.group != group:
        raise ValueError(
            f"the builder registered for group {group!r} returned a submodel labelled "
            f"{submodel.group!r}; the label decides which influence curve is used, so the "
            "two must agree"
        )
    return submodel


def weighted_form(submodel: Submodel, weights: FloatArray) -> tuple[Submodel, FloatArray]:
    r"""Recast a fluctuation in *weighted* form.

    R's ``tmle`` calls this ``target.gwt``: instead of using :math:`h` as a
    regression covariate, move its magnitude into the observation weights and keep
    only its sign.  The submodel becomes
    :math:`\operatorname{logit}\bar Q^*_\epsilon = \operatorname{logit}\bar Q^0
    + \epsilon^\top \operatorname{sign}(h)`, and because

    .. math:: \sum_i (w_i |h_i|)\, \operatorname{sign}(h_i)\,(Y_i - \bar Q^*_i)
              = \sum_i w_i\, h_i\, (Y_i - \bar Q^*_i)

    the *estimating equation being solved is identical*.  What changes is the
    design matrix, which no longer contains the extreme values that make the fit
    numerically fragile under weak overlap.  For the two-column ``mean`` submodel
    the columns have disjoint support (one per arm), so a single weight vector
    serves both.
    """
    magnitude = np.abs(submodel.observed).sum(axis=1)
    return submodel.map_arms(np.sign), np.asarray(weights, dtype=float) * magnitude


def restrict(submodel: Submodel, mask: BoolArray | IntArray) -> Submodel:
    """Row-subset a submodel, by boolean mask or by integer index.

    Used to cut a submodel down to one validation fold for the cross-validated targeting
    step.  *Not* used for missing outcomes: those are handled by the ``observed`` mask
    threaded into the fluctuation, which keeps the score's denominator at the full ``n``
    where the estimating equation needs it.
    """
    index = np.asarray(mask)
    return submodel.map_arms(lambda values: values[index])
