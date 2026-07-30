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
    Two columns, :math:`h_1(a, W) = \mathbb 1\{a = 1\} / (g_1(W)\,\pi_1(W))` and
    :math:`h_0(a, W) = \mathbb 1\{a = 0\} / (g_0(W)\,\pi_0(W))`.  Fitting both
    coefficients solves the score equation for each counterfactual mean
    separately, which is what makes the risk ratio and odds ratio available in
    addition to the difference.

``att`` / ``atc``
    A single column contrasting the arms, with the control arm reweighted by the
    propensity odds -- see :func:`att_submodel`.

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


def _arm_columns(
    n: int, probabilities: FloatArray | None, label: str
) -> tuple[FloatArray, FloatArray]:
    """Split an ``(n, 2)`` arm-indexed probability array into its two columns."""
    if probabilities is None:
        return np.ones(n), np.ones(n)
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (n, 2):
        raise ValueError(f"{label} must have shape ({n}, 2); got {probs.shape}")
    if np.any(probs <= 0):
        raise ValueError(f"{label} must be strictly positive after bounding")
    return probs[:, 0], probs[:, 1]


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
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
) -> Submodel:
    """Two-column submodel targeting both counterfactual means.

    Parameters
    ----------
    treatment:
        Binary treatment indicator, length ``n``.
    propensity:
        ``g(W) = P(A = 1 | W)``, already truncated away from 0 and 1.
    treated_fraction:
        Accepted and ignored.  Every builder in :data:`SUBMODEL_BUILDERS` takes the same
        keyword-only signature so the registry can dispatch without knowing which
        arguments each one happens to need; the counterfactual means do not condition on
        an arm and so have no use for the treated share.
    missingness:
        Optional ``(n, 2)`` array of ``P(Delta = 1 | A = a, W)`` for ``a = 0, 1``.
    intermediate_density, selection:
        For a controlled direct effect at intermediate value ``z``:
        ``P(Z = z | A = a, W)`` per arm, and the indicator ``1{Z_i = z}``.  The
        indicator multiplies only the *observed* covariate -- the counterfactual
        columns are already evaluated at ``Z = z`` by construction.
    """
    del treated_fraction  # see the parameter's docstring
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g1 = np.asarray(propensity, dtype=float).reshape(-1)
    n = a.shape[0]
    if g1.shape[0] != n:
        raise ValueError(f"propensity has length {g1.shape[0]}, expected {n}")
    if np.any(g1 <= 0) or np.any(g1 >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")
    g0 = 1.0 - g1
    pi0, pi1 = _arm_columns(n, missingness, "missingness probabilities")
    pz0, pz1 = _arm_columns(n, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)

    inv_one = 1.0 / (g1 * pi1 * pz1)
    inv_zero = 1.0 / (g0 * pi0 * pz0)

    zeros = np.zeros(n)
    return Submodel(
        np.column_stack([(1.0 - a) * keep * inv_zero, a * keep * inv_one]),
        {
            0.0: np.column_stack([inv_zero, zeros]),
            1.0: np.column_stack([zeros, inv_one]),
        },
        ("h0", "h1"),
        "mean",
        # One column per arm, in the order ``names`` gives: h0 is column 0, h1 column 1.
        {0.0: 0, 1.0: 1},
    )


def att_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
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
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g1 = np.asarray(propensity, dtype=float).reshape(-1)
    n = a.shape[0]
    share = _required_treated_fraction(treated_fraction, "att")
    if np.any(g1 <= 0) or np.any(g1 >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")
    g0 = 1.0 - g1
    pi0, pi1 = _arm_columns(n, missingness, "missingness probabilities")
    pz0, pz1 = _arm_columns(n, intermediate_density, "intermediate probabilities")
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
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
) -> Submodel:
    """One-column submodel targeting the ATC -- the mirror image of the ATT."""
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g1 = np.asarray(propensity, dtype=float).reshape(-1)
    n = a.shape[0]
    # In (0, 1) because the treated share is, which the helper has already enforced --
    # so there is no second range check here.
    control_fraction = 1.0 - _required_treated_fraction(treated_fraction, "atc")
    if np.any(g1 <= 0) or np.any(g1 >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")
    g0 = 1.0 - g1
    pi0, pi1 = _arm_columns(n, missingness, "missingness probabilities")
    pz0, pz1 = _arm_columns(n, intermediate_density, "intermediate probabilities")
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
#: treatment, the truncated propensity, and the same four keyword arguments -- ignoring
#: the ones it has no use for -- so that :func:`submodel_for` can dispatch on the group
#: name alone.
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
    change what five of the seven built-in estimands report.

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


def submodel_for(
    group: TargetGroup,
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
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
    submodel = builder(
        treatment,
        propensity,
        treated_fraction=treated_fraction,
        missingness=missingness,
        intermediate_density=intermediate_density,
        selection=selection,
    )
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
