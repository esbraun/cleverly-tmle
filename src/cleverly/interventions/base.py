r"""Interventions: what "counterfactual" means for a particular estimand.

Until now an intervention was implicit.  Every counterfactual quantity in the package
was keyed by *arm* -- :attr:`~cleverly.fluctuation.iterative.InitialFit.arms` holds
:math:`\bar Q(a, W)`, :func:`~cleverly.fluctuation.submodel.mean_submodel` builds
:math:`\mathbb 1\{A = a\} / g_a(W)` -- which silently identifies "the intervention" with
"set :math:`A` to the constant :math:`a`".  That is one intervention among many, and the
identification it hides is the reason a rule :math:`d(W)` or a stochastic assignment
:math:`g^\star(\cdot \mid W)` had no way to be expressed.

A **regime** here is a conditional density over the arms,

.. math::

    g^\star(a \mid W), \qquad \sum_a g^\star(a \mid W) = 1 ,

evaluated at every row: an ``(n, K)`` matrix.  All three supported kinds are that one
object.  :class:`Static` puts all its mass on one arm, :class:`Rule` puts it on
:math:`d(W)`, and :class:`Stochastic` spreads it.  Writing them as one representation is
what lets a single clever covariate

.. math::

    h(A, W) = \frac{g^\star(A \mid W)}{g(A \mid W)}

cover the three, and collapse to :math:`\mathbb 1\{A = a\}/g_a(W)` exactly when the
regime is :class:`Static`.

**What is deliberately not here.**  Both are about the *influence function*, not about
effort -- and both are implemented, elsewhere, under keywords of their own:

- An **incremental propensity-score intervention** tilts the estimated mechanism,
  :math:`g^\star_\delta(1 \mid W) = \delta g_1 / (\delta g_1 + 1 - g_1)`.  Its
  :math:`g^\star` is a functional of :math:`P`, so the efficient influence function
  carries a further term for the pathwise derivative through :math:`g` (Kennedy, 2019)
  that none of the regimes here need, and the estimator has to fluctuate the mechanism
  as well as :math:`\bar Q`.  Neither this Protocol -- whose ``density`` sees only the
  data -- nor the influence curve below can express that, which is why it is a parameter
  axis of its own: :mod:`cleverly.interventions.incremental` and ``TMLE(incremental=)``.
  The paragraph stays here rather than being deleted, because the thing to stop a reader
  doing is writing one as a :class:`Stochastic`.
- A **modified treatment policy** shifting a continuous treatment needs
  :math:`g^\star` and :math:`g` as conditional *densities* on a continuum, which the
  learner layer does not estimate -- there is no ``predict_density``.

:func:`refuse_unsupported` states both, by name, where a user would meet them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._typing import FloatArray
from ..data.causal_data import CausalData
from ..exceptions import DataError

__all__ = [
    "Intervention",
    "RegimeSet",
    "Rule",
    "Static",
    "Stochastic",
    "as_interventions",
    "refuse_unsupported",
]

#: How close to one a supplied stochastic density's row sums must be.  Loose enough for
#: a user's arithmetic in float32, tight enough that a genuine normalisation mistake --
#: forgetting an arm, or handing over unnormalised weights -- is caught.
_SIMPLEX_TOLERANCE = 1e-8


@runtime_checkable
class Intervention(Protocol):
    """A conditional distribution over the treatment arms.

    Implement :meth:`density` and carry a :attr:`name`; everything else -- the clever
    covariate, the influence curve, the positivity report, the parameter names -- is
    written against the ``(n, K)`` matrix and needs to know nothing about which kind of
    intervention produced it.

    Parameters
    ----------
    *args, **kwargs
        Present because :func:`typing.runtime_checkable` gives a protocol a synthetic
        constructor.  A protocol is implemented, not instantiated.

    Attributes
    ----------
    name : str
    """

    @property
    def name(self) -> str:
        """What this regime is called in reported parameter names: ``ey[treat_all]``.

        A read-only property rather than a mutable attribute so that the frozen
        dataclasses below satisfy the protocol: an intervention that could be renamed
        after a fit had used it would put the reported parameter names out of step with
        the densities they came from.
        """
        ...

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` array of :math:`g^\\star(a \\mid W_i)`, columns in arm-code order.
        """
        ...


# ------------------------------------------------------------------ level lookup


def _code_for(data: CausalData, level: Any) -> float:
    """The arm code for a user-facing treatment level.

    Levels are compared as the user wrote them, so ``Static("high")`` works on a string
    treatment and ``Static(1)`` on a numeric one.  A level the data does not declare is
    an error naming the ones it does: the alternative -- an all-zero density column --
    would be read downstream as a perfectly well-formed regime that never treats anyone.
    """
    for index, declared in enumerate(data.treatment_levels):
        if declared == level or (
            isinstance(level, (int, float, np.integer, np.floating))
            and isinstance(declared, (int, float, np.integer, np.floating))
            and float(declared) == float(level)
        ):
            return float(index)
    raise DataError(
        f"{level!r} is not a level of {data.treatment_name}; its levels are "
        f"{list(data.treatment_levels)}"
    )


def _one_hot(codes: FloatArray, n_arms: int) -> FloatArray:
    """Degenerate density: all mass on the arm each row names."""
    arm_codes = np.arange(n_arms, dtype=float)
    return np.asarray(codes, dtype=float).reshape(-1, 1) == arm_codes.reshape(1, -1)


def _covariate_frame(data: CausalData) -> Any:
    """The covariates as a dataframe in the caller's backend, for a user's rule to read.

    Covariates only, deliberately.  A rule that reads the outcome is not an intervention,
    and one that reads the observed treatment is a different object again -- a regime
    depending on :math:`A` is not a function of the history a point-treatment parameter
    conditions on.  Restricting the frame is how that is enforced rather than documented.

    The columns are the *encoded* covariates, so a categorical column appears as the
    indicators :meth:`~cleverly.data.CausalData.from_frame` expanded it into
    (``region__west``), not under its original name.  ``data.covariate_names`` is the
    list a rule should be written against.
    """
    return data.frame_like(
        {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
    )


def _as_array(values: Any) -> FloatArray:
    """A numpy array from whatever a user's callable returned (Series, list, array)."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values)


# ------------------------------------------------------------------- the kinds


@dataclass(frozen=True)
class Static:
    """Set the treatment to one level for everybody: :math:`g^\\star = \\mathbb 1\\{a = v\\}`.

    The degenerate regime, and the one the arm-keyed path has always estimated.  It is
    here for two reasons beyond completeness: it is the reference a rule is usually
    contrasted against, and a fit whose regimes are all :class:`Static` must reproduce
    the ordinary arm fit exactly, which is what ``tests/unit/test_regimes.py`` asserts.

    Parameters
    ----------
    level : Any
        The treatment level to assign to every row, as the caller spells it.
    name : str
        Label used in reported parameter names.  Empty builds ``"always <level>"``.
    """

    level: Any
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"always {self.level}")

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` density, columns in arm-code order.
        """
        code = _code_for(data, self.level)
        return _one_hot(np.full(data.n, code), data.n_arms).astype(float)


@dataclass(frozen=True)
class Rule:
    """A dynamic regime: assign :math:`d(W)`, a deterministic function of the covariates.

    ``rule`` is handed a dataframe of the covariates (see :func:`_covariate_frame`) in
    the backend the data arrived in, and returns the *level* -- the user's own label, not
    an internal code -- to assign to each row.

    ``` python
    Rule(lambda w: np.where(w["age"] > 65, 1, 0), name="treat the elderly")
    ```

    Every returned level is checked against the declared support before it becomes a
    density, so a rule with a typo or an off-by-one fails naming the levels that exist
    rather than producing a regime nobody asked for.

    Parameters
    ----------
    rule : callable
        Maps the covariate frame to the treatment level assigned to each row.
    name : str
        Label used in reported parameter names.
    """

    rule: Callable[[Any], Any]
    name: str

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` density, columns in arm-code order.
        """
        assigned = _as_array(self.rule(_covariate_frame(data)))
        if assigned.shape[0] != data.n or assigned.ndim > 1:
            raise DataError(
                f"rule {self.name!r} returned {assigned.shape} assignments for {data.n} "
                "rows; it must return one treatment level per row"
            )
        codes = np.array([_code_for(data, level) for level in assigned.tolist()], dtype=float)
        return _one_hot(codes, data.n_arms).astype(float)


@dataclass(frozen=True)
class Stochastic:
    """A known stochastic regime: assign arm :math:`a` with probability :math:`g^\\star(a \\mid W)`.

    ``density`` is handed the covariate frame and returns an ``(n, K)`` array whose
    columns are in :attr:`~cleverly.data.CausalData.arm_codes` order and whose rows sum
    to one.

    *Known* is the load-bearing word.  :math:`g^\\star` must be a fixed function of
    :math:`W`, not one derived from the estimated mechanism: the influence curve this
    package reports for a regime has no term for :math:`g^\\star` depending on
    :math:`P`.  See the module docstring, and :func:`refuse_unsupported`.

    Parameters
    ----------
    density_fn : callable
        Maps the covariate frame to an ``(n, K)`` array of arm probabilities.
    name : str
        Label used in reported parameter names.
    """

    density_fn: Callable[[Any], Any]
    name: str

    def density(self, data: CausalData) -> FloatArray:
        """Evaluate this regime's arm probabilities for every row.

        Parameters
        ----------
        data : CausalData
            Validated study data, which supplies the covariates and the arm order.

        Returns
        -------
        ndarray
            ``(n, K)`` density, columns in arm-code order.
        """
        values = np.asarray(_as_array(self.density_fn(_covariate_frame(data))), dtype=float)
        check_regime_density(
            values, label=f"stochastic regime {self.name!r}", shape=(data.n, data.n_arms)
        )
        return values


def check_regime_density(
    values: FloatArray, *, label: str, shape: tuple[int, ...] | None = None
) -> None:
    """Check finite probability simplexes with treatment arms on the second axis."""
    if values.ndim < 2 or (shape is not None and values.shape != shape):
        raise DataError(f"{label} returned shape {values.shape}; expected {shape}")
    if not np.all(np.isfinite(values)):
        raise DataError(f"{label} contains a non-finite probability")
    if np.any(values < 0.0):
        raise DataError(f"{label} returned a negative probability")
    sums = values.sum(axis=1)
    worst = float(np.max(np.abs(sums - 1.0))) if sums.size else 0.0
    if worst > _SIMPLEX_TOLERANCE:
        raise DataError(
            f"{label} has rows summing to as far as {worst:.3g} from one; "
            "a regime is a distribution over the arms, so its rows must be normalised"
        )


# ----------------------------------------------------------------- the refusals


def refuse_unsupported(kind: str) -> None:
    """Raise for an intervention that is not a regime, and say where it went.

    Called from :func:`as_interventions`, which is where a ``Shift`` or an ``Incremental``
    handed to ``interventions=`` arrives.  Both kinds here are implemented under keywords
    of their own, so both messages name that keyword; the ``ValueError`` rather than
    ``NotImplementedError`` says the difference is one of API rather than of derivation.
    """
    if kind == "ipsi":
        raise ValueError(
            "incremental propensity-score interventions are implemented, but not as an "
            "intervention. Their g*(a | W) = delta*g / (delta*g + 1 - g) is a functional "
            "of P, so the efficient influence function carries a further term for the "
            "dependence on g-hat (Kennedy 2019) and the estimator fluctuates the "
            "mechanism as well as the outcome regression -- neither of which the regime "
            "path can express. Declare one with cleverly.interventions.Incremental and "
            "pass it to TMLE(incremental=...). Building one by hand as a Stochastic "
            "regime would report a standard error for a different functional, and would "
            "be too small: the term it omits is orthogonal to the rest of the curve."
        )
    if kind in {"mtp", "shift"}:
        raise ValueError(
            "modified treatment policies are implemented, but not as an intervention. A "
            "shift reads the dose a unit actually received and moves it, so it is not a "
            "conditional distribution over arms the way a regime is; declare one with "
            "cleverly.interventions.Shift and pass it to TMLE(shifts=...). A shift of a "
            "*discrete* treatment can be written as a Rule."
        )
    raise ValueError(f"unknown intervention kind {kind!r}")


# ------------------------------------------------------------------ regime sets


@dataclass(frozen=True)
class RegimeSet:
    """The regimes a fit targets, and their evaluated densities.

    Keyed by an internal float code exactly as the arms are, with the labels carried
    alongside -- the same convention, and for the same reason: every array that is per
    regime (the clever covariate's columns, the counterfactual predictions, the influence
    curves) can then be handled by code that does not count regimes or know what they
    mean, while every reported name uses what the user called them.

    Parameters
    ----------
    names : tuple of str
        Regime labels in code order.
    values : ndarray
        ``(n, K, R)`` evaluated densities.
    reference : float
        Code of the regime contrasts are taken against.

    Attributes
    ----------
    names:
        Regime labels in code order, so ``names[r]`` is the label of code ``float(r)``.
    values:
        ``(n, K, R)`` densities: ``values[i, a, r]`` is :math:`g^\\star_r(a \\mid W_i)`.
    reference:
        The regime code contrasts are taken against, defaulting to the first supplied.
    """

    names: tuple[str, ...]
    values: FloatArray
    reference: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 3 or values.shape[2] != len(self.names):
            raise DataError(
                f"regime densities must be (n, K, {len(self.names)}) for regimes "
                f"{list(self.names)}; got shape {values.shape}"
            )
        if len(set(self.names)) != len(self.names):
            raise DataError(f"regime names must be distinct; got {list(self.names)}")
        if self.reference not in self.codes:
            raise DataError(
                f"reference regime code {self.reference} is not one of {list(self.codes)}"
            )
        object.__setattr__(self, "values", values)

    # ------------------------------------------------------------------ build

    @classmethod
    def evaluate(
        cls,
        interventions: Sequence[Intervention],
        data: CausalData,
        *,
        reference: str | None = None,
    ) -> RegimeSet:
        """Evaluate every intervention on ``data`` and assemble the set.

        Parameters
        ----------
        interventions : sequence of Intervention
            The regimes to evaluate, in the order their codes will follow.
        data : CausalData
            Validated study data to evaluate them on.
        reference : str or None
            Label of the regime contrasts are taken against.  ``None`` uses the first one
            supplied, mirroring the arm convention where the reference is the lowest code.

        Returns
        -------
        RegimeSet
            The evaluated densities, keyed by code.
        """
        if len(interventions) < 1:
            raise DataError("at least one intervention is required")
        names = tuple(str(intervention.name) for intervention in interventions)
        stacked = np.stack(
            [np.asarray(intervention.density(data), dtype=float) for intervention in interventions],
            axis=2,
        )
        code = 0.0
        if reference is not None:
            if reference not in names:
                raise DataError(f"reference={reference!r} is not one of the regimes {list(names)}")
            code = float(names.index(reference))
        return cls(names, stacked, code)

    # ------------------------------------------------------------------ access

    @property
    def n(self) -> int:
        """Return the number of observations."""
        return int(self.values.shape[0])

    @property
    def n_arms(self) -> int:
        """Return the number of treatment arms."""
        return int(self.values.shape[1])

    @property
    def n_regimes(self) -> int:
        """Return the number of regimens."""
        return len(self.names)

    @property
    def codes(self) -> tuple[float, ...]:
        """Regime codes, ``(0.0, ..., R-1.0)``, in the order the regimes were supplied."""
        return tuple(float(r) for r in range(self.n_regimes))

    @property
    def labels(self) -> dict[float, str]:
        """Code to label, which is what :func:`~cleverly.targets.parameter_name` is given."""
        return {float(r): name for r, name in enumerate(self.names)}

    def label(self, code: float) -> str:
        """Return the label a regime code was supplied under.

        Parameters
        ----------
        code : float
            Regime code.

        Returns
        -------
        str
            The label, as the caller spelled it.
        """
        return self.labels[float(code)]

    def column(self, code: float) -> FloatArray:
        """Return one regime's evaluated density.

        Parameters
        ----------
        code : float
            Regime code.

        Returns
        -------
        ndarray
            ``(n, K)`` density for that regime alone.
        """
        return np.asarray(self.values[:, :, round(float(code))], dtype=float)

    def subset(self, index: Any) -> RegimeSet:
        """The same regimes on a row subset -- a bootstrap resample, a validation fold.

        The densities are sliced rather than re-evaluated.  For a regime that is a
        function of :math:`W` alone the two agree exactly, and slicing is what keeps a
        loaded result (which carries the evaluated densities but not the callables that
        made them) usable everywhere a fitted one is.

        Parameters
        ----------
        index : array_like
            Row positions or a boolean mask.

        Returns
        -------
        RegimeSet
            The same regimes over the selected rows.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return replace(self, values=self.values[idx])

    @property
    def is_static(self) -> bool:
        """Whether every regime puts all its mass on one arm, the same one for every row.

        The condition under which the regime path is estimating exactly what the arm path
        estimates, and so the one the equivalence test checks.
        """
        for r in range(self.n_regimes):
            column = self.values[:, :, r]
            if column.size and not bool(np.all(column == column[:1])):
                return False
            if not bool(np.all(np.isin(column, (0.0, 1.0)))):
                return False
        return True


def as_interventions(value: Any) -> tuple[Intervention, ...]:
    """Normalise the ``interventions=`` argument into a tuple.

    A bare level is read as :class:`Static` on it, so ``interventions=(1, 0)`` means what
    it looks like it means; anything implementing :class:`Intervention` is taken as is.

    A :class:`~cleverly.interventions.Shift` or
    :class:`~cleverly.interventions.Incremental` is neither, and is sent to
    :func:`refuse_unsupported` rather than falling through to ``Static``.  Both are
    implemented, under keywords of their own, and the ``Static`` fallthrough would wrap
    the object as though it were a treatment *level* -- giving a regime named
    ``"always Shift(delta=0.5, ...)"`` and an error much further downstream, about
    something else.
    """
    from .incremental import Incremental
    from .shift import Shift

    if value is None:
        return ()
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    out: list[Intervention] = []
    for item in items:
        if isinstance(item, Shift):
            refuse_unsupported("shift")
        if isinstance(item, Incremental):
            refuse_unsupported("ipsi")
        if isinstance(item, Intervention) and not isinstance(item, (str, bytes)):
            out.append(item)
        else:
            out.append(Static(item))
    return tuple(out)
