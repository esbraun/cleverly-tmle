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
    One column per *non-reference arm*, contrasting that arm with the reference and
    reweighting one of the two by the propensity odds :math:`g_a / g_r` -- a single
    column for a binary treatment, which is the classic ATT.  Unlike ``mean``, the
    columns are not disjoint: the reference arm loads every one of them, since it is the
    arm every contrast is taken against.  See :func:`att_submodel`.

``regime``
    One column per *intervention*, :math:`h_r(A, W) = g^\star_r(A \mid W) / g(A \mid W)`
    -- a treatment rule or a stochastic assignment rather than a constant arm.  It is the
    same object as ``mean`` when every regime is static: one column of ``mean`` is one
    column of ``regime`` with :math:`g^\star_r = \mathbb 1\{a = v\}`.  The two are kept
    apart because the *parameters* move from the arms to the regimes -- see
    :func:`regime_submodel` for why that separation cannot be expressed by re-keying.

``ipsi``
    One column per *tilt of the mechanism*,
    :math:`h_r(A, W) = q_{\delta_r}(A \mid W) / (g(A \mid W)\,\pi(A, W))`, where
    :math:`q_\delta` multiplies the odds of treatment by :math:`\delta`.  Entry for entry
    this is what ``regime`` builds at the density :math:`q_\delta`, and the two are
    nonetheless separate groups: :math:`q_\delta` is a functional of :math:`P`, so the
    influence curve carries a further term and the estimator must fluctuate the mechanism
    as well as :math:`\bar Q`.  A group is a score equation, and this one has two --
    see :func:`ipsi_submodel` and :mod:`cleverly.fluctuation.mechanism`.

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

``ipsi`` is the exception to the paragraph above, and in the strict direction.  There
:math:`g` is inside the *estimand* rather than only in the estimating equation, so the
remainder keeps a term in :math:`(\hat g - g_0)^2` that no :math:`\bar Q` and no
:math:`\hat\pi` can reach, and the guarantee becomes: consistent if :math:`\hat g` is
right **and** one of :math:`\hat\pi`, :math:`\bar Q` is.  In particular the two mechanisms
cannot trade off against each other the way they do above -- nuisances wrong everywhere
whose product :math:`\hat g\hat\pi` is right drive the remainder to zero for ``mean`` and
not for ``ipsi``.  ``tests/unit/test_remainder_ipsi_mar.py`` states that as an equality and
keeps the failed cancellation as a negative control.

A controlled direct effect puts a third factor, :math:`q_z(a, W) = P(Z = z \mid A = a,
W)`, in the same denominator, and the statement generalises the same way rather than
gaining a third independent half: consistent if :math:`\bar Q` is right, **or** if the
product :math:`g\,q_z\,\pi` is right.  Everything above holds with :math:`\pi` replaced
by :math:`q_z\,\pi` throughout; :mod:`cleverly.estimators.direct_effect` derives it,
along with the parameter and the assumptions that identify it.

**Truncation.**  ``g``, :math:`\pi` and the intermediate density are all bounded away
from zero before they enter :math:`h` -- except that on ``ipsi`` the bound on ``g`` is
withheld, because there truncating it would move :math:`\Psi(\delta)` rather than
regularise a denominator, and none is needed: see :func:`ipsi_submodel`.  :math:`\pi` is
bounded there on the ordinary terms.  This regularises estimation; it does not
redefine the target.  The plug-in is an average of targeted predictions and contains no
mechanism at all, so no bound can move :math:`\Psi` -- what a binding bound moves is
:math:`R_2`, by exactly the formula above evaluated at the truncated value.  The trade
is variance for second-order bias, and ``res.sensitivity.truncation_curve()`` (with
``mechanism=True`` for :math:`\pi`) is how to see how much of it you are paying.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

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
    "ipsi_submodel",
    "mean_submodel",
    "msm_submodel",
    "mtp_submodel",
    "regime_submodel",
    "register_submodel",
    "restrict",
    "stitch",
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

        **Empty for a contrast submodel** (``att``, ``atc``), where a column targets a
        difference of arms and no column belongs to one of them.  That distinction is why
        this is a mapping rather than an assumed column order: it is what lets the
        influence curves index a column by the arm it targets instead of by the literal
        ``0`` and ``1`` that only a two-arm submodel has.
    contrast_columns:
        The mirror of :attr:`arm_columns` for a contrast submodel: the column whose
        coefficient targets the contrast *of* that arm against the reference.  Populated
        by :func:`att_submodel` and :func:`atc_submodel`, which fit one column per
        non-reference arm -- a single column for a binary treatment, and ``K - 1`` for a
        ``K``-armed one.  Empty for every per-arm submodel.

        Two mappings rather than one with a wider meaning, because they answer different
        questions: ``arm_columns[a]`` says which column *updates* arm ``a``, and
        ``contrast_columns[a]`` says which column carries the parameter ``a`` is contrasted
        under.  A contrast submodel's reference arm appears in every column and belongs to
        none, which is exactly what makes ``column_for`` inapplicable there.
    """

    observed: FloatArray
    arms: dict[float, FloatArray]
    names: tuple[str, ...]
    group: TargetGroup
    arm_columns: dict[float, int] = field(default_factory=dict)
    contrast_columns: dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_arms(self.observed, self.arms, "submodel")
        if self.observed.shape[1] != len(self.names):
            raise ValueError(
                f"{self.observed.shape[1]} covariate column(s) but {len(self.names)} name(s)"
            )
        for label, mapping in (
            ("arm_columns", self.arm_columns),
            ("contrast_columns", self.contrast_columns),
        ):
            for level, column in mapping.items():
                if level not in self.arms:
                    raise ValueError(
                        f"{label} names level {level}, which is not one of the submodel's "
                        f"arms {sorted(self.arms)}"
                    )
                if not 0 <= column < self.observed.shape[1]:
                    raise ValueError(
                        f"{label} maps level {level} to column {column}, outside the "
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
            dict(self.contrast_columns),
        )

    def column_for(self, level: float) -> FloatArray:
        """The observed covariate column whose coefficient targets ``level``.

        Raises for a contrast submodel, where no single column belongs to one arm; the
        influence curves for those estimands read :meth:`contrast_column_for` instead.
        """
        try:
            column = self.arm_columns[level]
        except KeyError:
            raise KeyError(
                f"the {self.group!r} submodel has no column dedicated to arm {level}; it "
                f"has arm columns for {sorted(self.arm_columns)}"
            ) from None
        return np.asarray(self.observed[:, column], dtype=float)

    def contrast_column_for(self, level: float) -> FloatArray:
        """The observed covariate column carrying the contrast of ``level`` and the reference.

        The counterpart of :meth:`column_for` for the conditional-effect submodels, and
        the reason the influence curves never index column ``0``: with more than two arms
        there is one such column per non-reference arm, and which is which is a statement
        about arm *codes* rather than about position.
        """
        try:
            column = self.contrast_columns[level]
        except KeyError:
            raise KeyError(
                f"the {self.group!r} submodel has no column carrying a contrast of arm "
                f"{level}; it has contrast columns for {sorted(self.contrast_columns)}"
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
    """Validate an ``(n, K)`` treatment-indexed probability array, or ones when absent.

    Ones rather than ``None`` so the callers multiply unconditionally: a mechanism that
    does not apply contributes a factor of one to every arm's denominator, which is an
    identity rather than a special case.

    ``k`` is the number of treatment values the mechanism was evaluated at, which is the
    arm count on the arm path and ``S + 1`` on a shift path -- the observed dose and each
    shifted one.  Nothing here reads the arms themselves, so the two share the check.
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


def _arm_mechanism(propensity: FloatArray, n: int, k: int, arms: tuple[float, ...]) -> FloatArray:
    """The ``(n, K)`` mechanism, validated, accepting the two-arm convenience form.

    One definition rather than three, because every builder that divides by ``g`` needs
    exactly this and had been repeating it: the vector form matters for the binary
    regression surface (see ``CLAUDE.md``) and a copy of it that drifted would move the
    two-arm numbers.
    """
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
    return g


def mean_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
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
    arm_fractions, reference:
        Accepted and ignored.  Every builder in :data:`SUBMODEL_BUILDERS` takes the same
        keyword-only signature so the registry can dispatch without knowing which
        arguments each one happens to need; the counterfactual means condition on no arm
        and are contrasted by the *estimand* rather than by the fluctuation, so they have
        use for neither the arm shares nor the reference.
    missingness:
        Optional ``(n, K)`` array of ``P(Delta = 1 | A = a, W)`` per arm.
    intermediate_density, selection:
        For a controlled direct effect at intermediate value ``z``:
        ``P(Z = z | A = a, W)`` per arm, and the indicator ``1{Z_i = z}``.  The
        indicator multiplies only the *observed* covariate -- the counterfactual
        columns are already evaluated at ``Z = z`` by construction.
    """
    del arm_fractions, reference, regimes, shifts, msm, incremental  # see the docstrings
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)
    g = _arm_mechanism(propensity, n, k, arms)

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
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
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
    arm_fractions, reference:
        Accepted and ignored; see :func:`mean_submodel`.
    """
    del arm_fractions, reference, shifts, msm, incremental  # see the parameter's docstring
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)
    g = _arm_mechanism(propensity, n, k, arms)

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


def _tilt_weights(n: int, k: int, incremental: FloatArray | None) -> FloatArray:
    """Validate the ``(n, K, R)`` clever covariates the ``ipsi`` submodel needs.

    Shape and finiteness only, on the same terms as :func:`_regime_densities`: that the
    tilted density is normalised and that the covariate is bounded by ``delta`` are
    properties of the *intervention*, established where it is built
    (:meth:`cleverly.interventions.IPSISet.evaluate`).
    """
    if incremental is None:
        raise ValueError(
            "the 'ipsi' submodel needs incremental=: an (n, K, R) array of "
            "q_delta(a | W) / g(a | W) per tilt. Build one with "
            "cleverly.interventions.IPSISet.evaluate."
        )
    values = np.asarray(incremental, dtype=float)
    if values.ndim != 3 or values.shape[:2] != (n, k):
        raise ValueError(
            f"incremental must have shape ({n}, {k}, R) -- rows, arms, tilts -- got {values.shape}"
        )
    if values.shape[2] == 0:
        raise ValueError("incremental must contain at least one tilt")
    if not np.all(np.isfinite(values)):
        raise ValueError("incremental clever covariates must be finite")
    return values


def ipsi_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
) -> Submodel:
    r"""One column per *tilt*, targeting :math:`E[Y^{q_{\delta_r}}]` for each.

    .. math::

        h_r(A, W) = \frac{q_{\delta_r}(A \mid W)}{g(A \mid W)\,\pi(A, W)}
                  = \frac{\delta_r A + 1 - A}{\bigl(\delta_r g(W) + 1 - g(W)\bigr)\,\pi(A, W)}

    with :math:`\pi(A, W) = P(\Delta = 1 \mid A, W)`, one where no outcome is missing.

    Column-for-column this is what :func:`regime_submodel` would build at the density
    :math:`q_{\delta_r}`, and that is deliberate: the two estimands genuinely share a
    score equation for :math:`\bar Q`.  What they do **not** share is the influence
    curve, because :math:`q_\delta` depends on :math:`P` and a regime's does not.  The
    group is therefore separate, which is what makes
    :func:`~cleverly.inference.influence.ipsi_means` rather than
    :func:`~cleverly.inference.influence.regime_means` the curve this submodel's estimates
    are read with -- :func:`submodel_for` checks the label for exactly that reason.

    **This is only half of the targeting.** Fluctuating :math:`\bar Q` along these columns
    solves the first of two score equations; the second lives in the tangent space of the
    treatment mechanism and is solved by
    :func:`~cleverly.fluctuation.mechanism.fit_mechanism`, whose covariate is the blip
    weighted by :attr:`~cleverly.interventions.IPSISet.derivative`.  See
    :func:`~cleverly.fluctuation.mechanism.solve_ipsi` for the alternation between them.

    Parameters
    ----------
    incremental:
        ``(n, K, R)``: ``incremental[i, j, r]`` is
        :math:`q_{\delta_r}(\text{arms}[j] \mid W_i) / g(\text{arms}[j] \mid W_i)`,
        precomputed by :class:`~cleverly.interventions.IPSISet` in the form where the
        mechanism has cancelled.
    propensity:
        Accepted and **ignored**, unlike in every other builder here.  ``build_submodel``
        hands over a *truncated* mechanism, and truncating :math:`g` on this axis would
        move the estimand rather than the estimator -- :math:`g` appears in
        :math:`\Psi(\delta)` itself.  Nothing here divides by :math:`g`, so there is
        nothing for a bound to protect.
    missingness:
        ``(n, K)`` of :math:`P(\Delta = 1 \mid A = \text{arms}[j], W_i)`, dividing the
        covariate exactly as it does in :func:`mean_submodel` and :func:`regime_submodel`.
        Unlike ``propensity`` above this one *is* a denominator to protect, and
        ``nuisance_bound=`` protects it in the ordinary way: :math:`\pi` is not inside
        :math:`\Psi(\delta)`, so bounding it regularises the estimator rather than moving
        the estimand -- the exact thing bounding :math:`g` here would do.

        That the composition is this and no more was not assumed.  The efficient influence
        function of :math:`\Psi(\delta)` under missingness at random is this covariate's
        residual term plus Kennedy's mechanism term *unchanged*, because :math:`q_\delta`
        is a functional of :math:`P(A \mid W)` and both :math:`A` and :math:`W` are
        recorded for every row.  ``tests/unit/test_influence_gateaux_ipsi_mar.py`` checks
        that against a complex-step Gateaux derivative on ``tests/discrete_law_mar.py``.
    arm_fractions, reference, intermediate_density, selection:
        Accepted and ignored.  A fit that declares ``incremental=`` still refuses
        ``intermediate=`` in :meth:`~cleverly.estimators.TMLE._validate_settings`, for
        want of a written-down parameter rather than for want of a factor -- so the last
        two are never anything but trivial by the time they arrive.
    """
    del propensity, arm_fractions, reference, intermediate_density, selection
    del regimes, shifts, msm  # see the parameters' docstrings
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)

    weights = (
        _tilt_weights(n, k, incremental)
        / _arm_matrix(n, k, missingness, "missingness probabilities")[:, :, None]
    )
    counterfactual = {float(arm): weights[:, j, :] for j, arm in enumerate(arms)}
    indicator = np.column_stack([(a == arm) for arm in arms]).astype(float)
    observed = np.einsum("ij,ijr->ir", indicator, weights)

    return Submodel(
        observed,
        counterfactual,
        tuple(f"h_ipsi{r}" for r in range(weights.shape[2])),
        "ipsi",
        # No arm_columns: a column targets a tilt, which spreads over the arms rather
        # than naming one -- the same reason regime_submodel leaves it empty.
    )


def _weighted_design(n: int, k: int, msm: FloatArray | None) -> FloatArray:
    r"""Validate the ``(n, K, p)`` array :math:`h(a, V)\varphi(a, V)` the ``msm`` submodel needs.

    Shape and finiteness only.  That the weights are non-negative and that the design is
    not collinear are properties of the *working model*, checked where it is built
    (:class:`cleverly.msm.MSMSet`) rather than repeated here: this runs once per truncation
    bound in a sensitivity sweep, and neither property can change between them.
    """
    if msm is None:
        raise ValueError(
            "the 'msm' submodel needs msm=: an (n, K, p) array of h(a, V) * phi(a, V) per "
            "arm and term. Build one with cleverly.msm.MSMSet.evaluate(...).weighted_design."
        )
    values = np.asarray(msm, dtype=float)
    if values.ndim != 3 or values.shape[:2] != (n, k):
        raise ValueError(
            f"msm must have shape ({n}, {k}, p) -- rows, arms, terms -- got {values.shape}"
        )
    if values.shape[2] == 0:
        raise ValueError("a working model must have at least one term")
    if not np.all(np.isfinite(values)):
        raise ValueError("the working model's weighted design contains a non-finite value")
    return values


def msm_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
) -> Submodel:
    r"""One column per *coefficient* of a working model, targeting its projection.

    .. math::

        H(A, W) = \frac{h(A, V)\,\varphi(A, V)}
                       {g(A \mid W)\,\pi(W)\,q(W)}
                = \sum_a \mathbb 1\{A = a\}\,
                  \frac{h(a, V)\,\varphi(a, V)}{g_a(W)\,\pi_a(W)\,q_a(W)}

    a ``p``-column design, one column per term.  Fluctuating along it solves

    .. math::

        P_n\Big[\frac{h(A,V)\,\varphi(A,V)}{g(A \mid W)}\,\big(Y - \bar Q^*(A, W)\big)\Big] = 0 ,

    which is the first of the two terms of the efficient influence function for
    :math:`\beta`; the second is solved *exactly* by the weighted least squares that reads
    :math:`\hat\beta` off the targeted fit, so the pair is zero by construction rather than
    by iteration.  See :mod:`cleverly.msm` for the estimand and
    :func:`~cleverly.inference.influence.msm_coefficients` for the curve.

    **Why the columns are coefficients and the arms are still arms.**  Exactly the
    reasoning :func:`regime_submodel` sets out, one axis further along.  The fluctuation
    updates :math:`\bar Q(a, W)` at every arm, because the projection reads the
    counterfactual mean at all of them; but the score equations are one per *term*.  So
    :attr:`Submodel.arms` is keyed by arm and each entry carries ``p`` columns, while the
    parameters live on the columns.  :attr:`Submodel.arm_columns` is left empty for the
    same reason :func:`att_submodel` and :func:`regime_submodel` leave it empty: no column
    belongs to a single arm.

    Note that with a **saturated** working model -- one indicator column per arm, uniform
    weights -- this is :func:`mean_submodel` entry for entry, which is what
    ``tests/unit/test_msm_submodel.py`` checks.  That is the sense in which the arm-keyed
    path is a special case here too.

    Parameters
    ----------
    msm:
        ``(n, K, p)``: ``msm[i, j, :]`` is
        :math:`h(\text{arms}[j], V_i)\,\varphi(\text{arms}[j], V_i)`, from
        :attr:`cleverly.msm.MSMSet.weighted_design`.  The weight is folded in there
        because the covariate needs only the product, and passing plain arrays is what
        lets the registry dispatch on the group name alone.
    arm_fractions, reference:
        Accepted and ignored; see :func:`mean_submodel`.
    """
    del arm_fractions, reference, regimes, shifts, incremental  # see the parameters' docstrings
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)
    g = _arm_mechanism(propensity, n, k, arms)

    weighted = _weighted_design(n, k, msm)
    pi = _arm_matrix(n, k, missingness, "missingness probabilities")
    pz = _arm_matrix(n, k, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)

    # (n, K): the denominator arm by arm, exactly as mean_submodel builds it, so a fit
    # with delta= or intermediate= composes rather than needing its own derivation.
    inverse = 1.0 / (g * pi * pz)
    # (n, K, p) -> (n, p) per arm: the covariate that arm's prediction is fluctuated by.
    covariate = weighted * inverse[:, :, None]
    counterfactual = {float(arm): covariate[:, j, :] for j, arm in enumerate(arms)}
    indicator = np.column_stack([(a == arm) for arm in arms]).astype(float)
    observed = keep[:, None] * np.einsum("ij,ijp->ip", indicator, covariate)

    return Submodel(
        observed,
        counterfactual,
        tuple(f"h_msm{j}" for j in range(weighted.shape[2])),
        "msm",
        # No arm_columns: a column targets a coefficient of the working model, which
        # summarises every arm rather than naming one.
    )


def mtp_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
) -> Submodel:
    r"""One column per *shift*, targeting :math:`E[\bar Q(d_\delta(A, W), W)]` for each.

    .. math::

        h_r(a, W) = \frac{g(a - \delta_r \mid W)}{g(a \mid W)}
                    + \mathbb 1\{a > u_r - \delta_r\}

    -- see :mod:`cleverly.interventions.shift` for the derivation, the sanity checks it
    satisfies, and why this is not the ``regime`` fluctuation at the induced density.

    with a further factor per mechanism that stands between the dose and a recorded
    outcome:

    .. math::

        H_r(a, W) = \frac{h_r(a, W)}{\pi(a, W)\, q_z(a, W)}

    ``shifts`` is the ``(n, S, S)`` array
    :math:`h_r(d_s(A_i, W_i), W_i)` **stacked with** the ``(n, S)`` covariate at the
    observed treatment; both are read off
    :class:`~cleverly.interventions.shift.ShiftSet`, which computed them from one stored
    conditional density.  ``propensity`` is ignored: a continuous treatment has no
    per-arm mechanism, and its ``(n, 0)`` propensity carries no information.

    ``missingness`` and ``intermediate_density`` are **not** ignored, and the axis they
    are indexed by is the thing to get right.  Both are ``(n, S + 1)`` -- the mechanism at
    the observed dose in column ``0`` and at :math:`d_s(A, W)` in column ``s + 1``, which
    is ``shifts``' first axis exactly -- so block ``j`` of the covariate is divided by
    column ``j`` of each.  Dividing every block by column ``0`` would be the arm path's
    mistake with the indicator removed: :math:`\bar Q^*(d_s(A,W), W)` is the fluctuation
    read *at the shifted dose*, so the mechanism has to be the one that holds there.  It is
    a silent error wherever the mechanism does not happen to depend on the dose, which is
    why ``tests/discrete_law_shift_cde.py`` makes both depend on it -- and it is invisible
    to a Gateaux check on an exact law, where ``epsilon`` is zero and no counterfactual
    block is read at all.  ``tests/unit/test_shift_submodel.py`` pins the blocks
    structurally and ``tests/unit/test_shift_fit.py`` pins the plug-in they move; neither
    is redundant with the other.

    ``selection`` multiplies only the *observed* covariate, exactly as it does in
    :func:`mean_submodel`: the counterfactual blocks are already evaluated at ``Z = z`` by
    construction, and zeroing them would leave every row whose intermediate took the other
    level with an **un-updated** prediction in the plug-in.

    Unlike ``regime``, ``arm_columns`` is **populated**: column ``s`` really does target
    one parameter, the mean under shift ``s``, so
    :meth:`~Submodel.column_for` can answer and
    :func:`~cleverly.inference.influence.shift_means` reads it.
    """
    del propensity, arms, arm_fractions, reference, regimes
    del msm
    del incremental
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
    evaluated = stacked.shape[1]
    pi = _arm_matrix(n, evaluated, missingness, "missingness probabilities")
    pz = _arm_matrix(n, evaluated, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)
    covariate = stacked / (pi * pz)[:, :, None]
    observed = keep[:, None] * covariate[:, 0, :]
    counterfactual = {float(index): covariate[:, index + 1, :] for index in range(stacked.shape[2])}
    return Submodel(
        observed,
        counterfactual,
        tuple(f"h_shift{r}" for r in range(stacked.shape[2])),
        "mtp",
        {float(index): index for index in range(stacked.shape[2])},
    )


def _reference_index(reference: float | None, arms: tuple[float, ...], group: str) -> int:
    """Which arm every conditional contrast is taken against, as a column index.

    ``None`` is the lowest arm, which is the rule ``reference=`` follows everywhere else
    and which for a binary treatment is the control -- so a two-armed conditional effect
    means exactly what it always did without the caller saying so.
    """
    if reference is None:
        return 0
    level = float(reference)
    if level not in arms:
        raise ValueError(
            f"the {group!r} submodel was given reference={reference!r}, which is not one "
            f"of the arms {list(arms)}"
        )
    return arms.index(level)


def att_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
) -> Submodel:
    r"""One column per non-reference arm, targeting the effect among that arm's units.

    .. math::

        h_a(A, W) = \frac{1}{P(A = a)}
                  \left( \frac{\mathbb 1\{A = a\}}{\pi_a(W)}
                       - \frac{\mathbb 1\{A = r\}}{\pi_r(W)}
                         \frac{g_a(W)}{g_r(W)} \right)

    with ``r`` the reference arm.  Arm ``a`` needs no reweighting -- the parameter
    conditions on ``A = a``, so those outcomes are already drawn from the target
    population.  Reference units are reweighted by the propensity odds ``g_a / g_r`` to
    make them resemble arm ``a``'s, which is why this estimand is far more sensitive than
    the ATE to small ``g_r(W)``; it is also why ``g_bounds="auto"`` uses a more
    conservative bound for it.

    With two arms this is the classic ATT: one column, ``r = 0``, ``a = 1``, and the
    arrays are bit for bit what they were before the estimand was defined for more arms.
    With ``K`` arms there are ``K - 1`` columns and ``K - 1`` score equations solved in
    **one** fluctuation, because they share a targeted ``Qbar``: the reference arm's
    prediction is updated by every column, since ``r`` appears in every contrast.  That
    also makes the Hessian non-diagonal here, unlike :func:`mean_submodel`'s.

    ``arm_fractions`` is ``P(A = a)`` per arm, in ``arms`` order, and may be given as a
    bare ``P(A = 1)`` for a binary treatment on the same terms as ``propensity`` may be
    given as a bare ``g_1``.
    """
    del regimes, shifts, msm, incremental  # accepted and ignored; this conditions on an arm
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)
    shares = _required_arm_fractions(arm_fractions, arms, "att")
    g = _arm_mechanism(propensity, n, k, arms)
    pi = _arm_matrix(n, k, missingness, "missingness probabilities")
    pz = _arm_matrix(n, k, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)
    r = _reference_index(reference, arms, "att")

    contrasts = [j for j in range(k) if j != r]
    # The conditioning arm's own term, and the reference units reweighted to resemble it.
    # One pair per contrast: the parameter is "the effect among those who received a",
    # which is a different population for each a.
    own = [1.0 / (shares[j] * pi[:, j] * pz[:, j]) for j in contrasts]
    against = [(g[:, j] / g[:, r]) / (shares[j] * pi[:, r] * pz[:, r]) for j in contrasts]

    return _contrast_submodel(a, arms, keep, r, contrasts, own, against, "att")


def atc_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
) -> Submodel:
    r"""The mirror image of :func:`att_submodel`: the effect among the *reference* arm.

    .. math::

        h_a(A, W) = \frac{1}{P(A = r)}
                  \left( \frac{\mathbb 1\{A = a\}}{\pi_a(W)}\frac{g_r(W)}{g_a(W)}
                       - \frac{\mathbb 1\{A = r\}}{\pi_r(W)} \right)

    Every column conditions on the *same* population, ``A = r``, which is the sense in
    which "the controls" generalises: with two arms the reference is the control arm and
    this is the classic ATC, and with more there is one such effect per non-reference arm,
    all averaged over the reference arm's covariate distribution.
    """
    del regimes, shifts, msm, incremental  # accepted and ignored, as in att_submodel
    a = np.asarray(treatment, dtype=float).reshape(-1)
    n = a.shape[0]
    k = len(arms)
    shares = _required_arm_fractions(arm_fractions, arms, "atc")
    g = _arm_mechanism(propensity, n, k, arms)
    pi = _arm_matrix(n, k, missingness, "missingness probabilities")
    pz = _arm_matrix(n, k, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)
    r = _reference_index(reference, arms, "atc")

    contrasts = [j for j in range(k) if j != r]
    # The reference arm's share throughout, since every column conditions on A = r.
    own = [(g[:, r] / g[:, j]) / (shares[r] * pi[:, j] * pz[:, j]) for j in contrasts]
    against = [1.0 / (shares[r] * pi[:, r] * pz[:, r]) for _ in contrasts]

    return _contrast_submodel(a, arms, keep, r, contrasts, own, against, "atc")


def _contrast_submodel(
    treatment: FloatArray,
    arms: tuple[float, ...],
    keep: FloatArray,
    reference: int,
    contrasts: list[int],
    own: list[FloatArray],
    against: list[FloatArray],
    group: str,
) -> Submodel:
    """Assemble a conditional-effect submodel from its two per-contrast terms.

    ``own[c]`` is the covariate at the contrast arm itself and ``against[c]`` the one at
    the reference arm, both already carrying whichever conditioning share the group uses.
    Shared by the two builders because only those terms differ between them, and the
    column bookkeeping -- which arm loads which column, and that the reference loads all
    of them -- is the part that a copy could get subtly wrong.
    """
    n = treatment.shape[0]
    zeros = np.zeros(n)
    observed = np.column_stack(
        [
            keep * ((treatment == arms[j]) * own[c] - (treatment == arms[reference]) * against[c])
            for c, j in enumerate(contrasts)
        ]
    )
    counterfactual = {
        # The reference arm's prediction is updated by every column: it is the arm every
        # contrast is taken against, so it belongs to none of them and appears in all.
        arms[reference]: np.column_stack([-term for term in against]),
        **{
            arms[j]: np.column_stack(
                [own[c] if c == other else zeros for other in range(len(contrasts))]
            )
            for c, j in enumerate(contrasts)
        },
    }
    return Submodel(
        observed,
        counterfactual,
        # The bare name on a binary treatment, where there is one contrast and nothing to
        # distinguish -- the same rule reported parameter names follow, and the reason a
        # two-armed fit's score diagnostics read exactly as they did.
        (f"h_{group}",)
        if len(contrasts) == 1
        else tuple(f"h_{group}[{arms[j]:g}]" for j in contrasts),
        group,
        # No arm_columns: a column targets a *contrast* of two arms, so none belongs to
        # one of them. Which contrast it carries is `contrast_columns`.
        {},
        {arms[j]: c for c, j in enumerate(contrasts)},
    )


def _required_arm_fractions(
    arm_fractions: FloatArray | float | None, arms: tuple[float, ...], group: str
) -> FloatArray:
    """``P(A = a)`` per arm, for a builder that cannot work without them.

    The uniform builder signature makes ``arm_fractions`` optional at the type level even
    though the conditional-effect submodels require it, so the requirement is enforced
    here instead of by the dispatcher.  That is deliberate: the dispatcher no longer knows
    which builders need what, and a builder that silently substituted a default would
    report an ATT against a population nobody specified.

    A scalar is read as ``P(A = 1)`` on a binary treatment, exactly as
    :func:`_arm_mechanism` reads a bare vector as ``g_1`` -- the complement is taken
    rather than a second number accepted, which is what keeps the two-arm arithmetic
    identical to what it was.
    """
    if arm_fractions is None:
        raise ValueError(f"the {group!r} submodel needs arm_fractions")
    if isinstance(arm_fractions, (int, float)) and not isinstance(arm_fractions, bool):
        if len(arms) != 2:
            raise ValueError(
                f"arm_fractions was given as a single share but there are {len(arms)} arms "
                f"{list(arms)}; supply one share per arm"
            )
        share = float(arm_fractions)
        shares = np.array([1.0 - share, share])
    else:
        shares = np.asarray(arm_fractions, dtype=float).reshape(-1)
    if shares.shape != (len(arms),):
        raise ValueError(
            f"arm_fractions must have one share per arm {list(arms)}; got shape {shares.shape}"
        )
    if np.any(shares <= 0.0) or np.any(shares >= 1.0):
        raise ValueError(f"arm_fractions must lie in (0, 1); got {list(shares)}")
    return shares


#: What a submodel builder looks like from the registry's side.  Every builder takes the
#: treatment, the truncated per-arm propensity, and the same five keyword arguments --
#: ignoring the ones it has no use for -- so that :func:`submodel_for` can dispatch on the
#: group name alone.  ``arms`` joined that signature when the treatment stopped being
#: binary: a builder cannot key its output by arm without being told which arms there are,
#: and inferring them from the observed treatment would go wrong on exactly the subsample
#: that is missing one.  ``reference`` joined it for the same reason one step on: a
#: conditional effect is one parameter per *non-reference* arm, so a builder that targets
#: contrasts has to be told which arm they are taken against.
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
register_submodel("ipsi", ipsi_submodel)
register_submodel("mtp", mtp_submodel)
register_submodel("msm", msm_submodel)


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
    "msm": (
        "Every submodel builder now takes the working model's weighted design, because a "
        "fit's parameters may be the coefficients of a marginal structural model rather "
        "than anything indexed by an arm; add 'msm=None' to its keyword-only parameters. "
        "A builder that targets arms, regimes or shifts should accept and ignore it, as "
        "mean_submodel does."
    ),
    "arm_fractions": (
        "'treated_fraction' is now 'arm_fractions': the share of the sample in *each* "
        "arm, in 'arms' order, because a conditional effect on a multi-valued treatment "
        "conditions on an arm that need not be arm 1. Rename the keyword; a builder that "
        "wants only the treated share can still be handed one, since a scalar is read as "
        "P(A = 1) on a binary treatment."
    ),
    "reference": (
        "Every submodel builder now takes the arm every contrast is taken against, "
        "because a conditional effect on a multi-valued treatment is one parameter per "
        "non-reference arm and the fluctuation has to know which arm that is; add "
        "'reference=None' to its keyword-only parameters. A builder whose columns target "
        "arms rather than contrasts should accept and ignore it, as mean_submodel does."
    ),
    "incremental": (
        "Every submodel builder now takes the incremental interventions' clever "
        "covariates, because a fit's intervention may be a tilt of the estimated "
        "mechanism rather than anything known in advance; add 'incremental=None' to its "
        "keyword-only parameters. A builder that targets arms, regimes, shifts or a "
        "working model should accept and ignore it, as mean_submodel does."
    ),
}


def submodel_for(
    group: TargetGroup,
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    arms: tuple[float, ...] = (0.0, 1.0),
    arm_fractions: FloatArray | float | None = None,
    reference: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
    regimes: FloatArray | None = None,
    shifts: FloatArray | None = None,
    msm: FloatArray | None = None,
    incremental: FloatArray | None = None,
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
            arm_fractions=arm_fractions,
            reference=reference,
            missingness=missingness,
            intermediate_density=intermediate_density,
            selection=selection,
            regimes=regimes,
            shifts=shifts,
            msm=msm,
            incremental=incremental,
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


def stitch(pieces: Sequence[tuple[IntArray, Submodel]], n: int) -> Submodel:
    """Reassemble one submodel from per-fold ones, each covering its own rows.

    :func:`restrict`'s inverse, and needed for the same reason the targeted predictions
    are stitched back together: under fold-wise targeting each fold gets its own
    covariate, and the *pooled* score has to be taken against the covariate each row was
    actually fluctuated by.  Where the covariate does not depend on anything fold-specific
    -- every group but a linked ``msm`` -- restricting and stitching returns the array it
    started with, value for value, which is why this can sit on the common path.

    Every piece must agree about the group, the column names and which column belongs to
    which arm; only the rows differ.  The folds partition the sample, so each row is
    written exactly once and no row is left unwritten.
    """
    if not pieces:
        raise ValueError("stitching needs at least one submodel")
    first = pieces[0][1]
    written = np.zeros(n, dtype=bool)
    observed = np.empty((n, first.dim))
    arms = {level: np.empty((n, first.dim)) for level in first.arms}
    for index, piece in pieces:
        if piece.group != first.group or piece.names != first.names:
            raise ValueError(
                f"stitching submodels that describe different fluctuations: "
                f"{first.group!r} {list(first.names)} against {piece.group!r} "
                f"{list(piece.names)}"
            )
        rows = np.asarray(index)
        observed[rows] = piece.observed
        for level, values in piece.arms.items():
            arms[level][rows] = values
        written[rows] = True
    if not bool(np.all(written)):
        raise ValueError(
            f"the pieces cover {int(written.sum())} of {n} rows; a stitched submodel "
            "needs a partition of the sample, which is what the validation folds are"
        )
    return replace(first, observed=observed, arms=arms)
