r"""Declared, stateless bases of effect modifiers, and the schema the modifiers obey.

A conditional contrast asks how the effect of a treatment changes with a few baseline
columns, the *modifiers* :math:`V`.  Its finite working model is a basis
:math:`\varphi(v) = (\varphi_1(v), \ldots, \varphi_p(v))` of known functions of
:math:`V`, and the projection of the conditional effect onto :math:`\operatorname{span}
(\varphi)` is a :math:`p`-vector with an interval (:class:`cleverly.ConditionalContrast`).

**Every basis here is declared and stateless.**  No basis has a ``fit``.  ``evaluate`` is
a pure function of the constructor arguments, the declared :class:`Modifiers` and the rows
it is handed.  A domain, a knot, a level set or a component is never read off the sample:
a basis built from the rows is a functional of the observed law, and its influence curve
would need a term for estimating it, which no estimator here computes.  A basis that needs
a domain (:class:`Cosine`, :class:`BSpline`) or a level set (:class:`Indicators`) reads it
from its own arguments or from :class:`Modifiers`, and refuses when neither declares it.

The module imports :mod:`numpy` and :mod:`cleverly.exceptions` only, so code outside the
estimator stack can build a design from it.

Every shipped basis but :class:`Constant` carries an intercept column by default
(``intercept=True``), and :class:`Additive` shares one intercept across its components,
so ``Linear(["W1"])`` has the two terms ``(intercept)`` and ``W1``.  Pass
``intercept=False`` to drop it.

A basis also declares a *size*, the number of functions each component contributes
beside the intercept, and :meth:`Basis.truncate` keeps the first ``k`` of them.  For
:class:`Cosine` the truncated spans are nested.  :class:`Linear`, :class:`Indicators` and
:class:`Constant` have one size and truncate to themselves.  A :class:`BSpline` truncated
to ``k`` interior knots has equally spaced knots again, and its span is nested only when
``k + 1`` divides the larger count plus one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Protocol

import numpy as np

from .exceptions import CapabilityError, DataError, Refusal, RefusalReason

__all__ = [
    "INTERCEPT",
    "Additive",
    "BSpline",
    "Basis",
    "BasisKind",
    "Constant",
    "Cosine",
    "Indicators",
    "Interactions",
    "Linear",
    "Modifiers",
    "Sieve",
    "Tensor",
    "UserBasis",
    "check_basis",
    "interactions",
    "resolve",
]

#: What a basis declares about its functions.  ``"known"`` is a fixed function of the
#: modifiers chosen without reading the data; ``"estimated"`` is one built from the rows.
BasisKind = Literal["known", "estimated"]

#: The name the shared intercept column reports under.
INTERCEPT = "(intercept)"

#: Why a basis computed from the sample is refused (R3 of the P1 plan).
SAMPLE_BASIS_REFUSAL = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "The working model must be a known function of the modifiers. A domain, knot, level "
    "set or component read off the sample makes the basis a functional of the observed "
    "law, and the influence curve would need a term for estimating it. For the EP sieve, "
    "EPL's Theorem 2 also assumes a deterministic sieve (conditions C3a and C3b).",
    alternative=(
        "Declare each needed domain and level set in cleverly.bases.Modifiers, and build "
        "the model from cleverly.bases."
    ),
)


def _refuse_sample_basis(detail: str) -> CapabilityError:
    """The R3 error, with the sentence that names what is missing."""
    return CapabilityError(
        f"{detail} {SAMPLE_BASIS_REFUSAL.condition}",
        reason=SAMPLE_BASIS_REFUSAL.reason,
        alternative=SAMPLE_BASIS_REFUSAL.alternative,
    )


# ----------------------------------------------------------------- reading a frame


def _read_column(frame: Any, name: str) -> np.ndarray:
    """One column of ``frame`` as a numpy array, from pandas, polars or a mapping."""
    try:
        values = frame[name]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        raise DataError(
            f"the modifier column {name!r} is not in the rows given; a basis reads each "
            "modifier by its declared name"
        ) from error
    except Exception as error:  # polars raises its own ColumnNotFoundError
        raise DataError(
            f"the modifier column {name!r} is not in the rows given; a basis reads each "
            "modifier by its declared name"
        ) from error
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values).reshape(-1)


def _numeric(name: str, values: np.ndarray) -> np.ndarray:
    """``values`` as finite floats, or a refusal naming the column."""
    try:
        numbers = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise DataError(
            f"the modifier {name!r} is read as a continuous value and holds a value that is "
            "not a number; declare its levels in Modifiers(levels=...) to read it as a "
            "category"
        ) from error
    if not np.all(np.isfinite(numbers)):
        raise DataError(f"the modifier {name!r} holds a missing or non-finite value")
    return numbers


def _as_domain(name: str, value: Any) -> tuple[float, float]:
    try:
        low, high = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError) as error:
        raise DataError(
            f"the domain of {name!r} must be a pair (low, high); got {value!r}"
        ) from error
    if len(value) != 2 or not (np.isfinite(low) and np.isfinite(high)) or not low < high:
        raise DataError(
            f"the domain of {name!r} must be two finite numbers with low < high; got {value!r}"
        )
    return low, high


def _as_levels(name: str, value: Any) -> tuple[Hashable, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DataError(f"the levels of {name!r} must be a sequence such as a tuple; got {value!r}")
    levels = tuple(value)
    if len(levels) < 2:
        raise DataError(f"a categorical modifier needs at least two levels; {name!r} has {levels}")
    if len(set(levels)) != len(levels):
        raise DataError(f"the levels of {name!r} must be distinct; got {levels}")
    return levels


def _as_dict(pairs: Any) -> dict[str, Any]:
    """A normalized declaration (a tuple of pairs) as a dict."""
    return dict(pairs)


def _pairs(
    name: str, mapping: Mapping[str, Any] | Sequence[tuple[str, Any]] | None, convert: Any
) -> tuple[tuple[str, Any], ...]:
    """A declaration mapping, as a tuple of ``(column, value)`` pairs, converted."""
    if mapping is None:
        return ()
    items = mapping.items() if isinstance(mapping, Mapping) else tuple(mapping)
    out: list[tuple[str, Any]] = []
    for key, value in items:
        if not isinstance(key, str):
            raise DataError(f"{name} must be keyed by column name; got the key {key!r}")
        out.append((key, convert(key, value)))
    if len({key for key, _ in out}) != len(out):
        raise DataError(f"{name} names a column twice")
    return tuple(out)


# --------------------------------------------------------------------- the schema


@dataclass(frozen=True)
class Modifiers:
    """The modifier columns, with the domain or the level set each one is declared on.

    A continuous modifier may carry a domain ``(low, high)``, which a basis that reads a
    domain needs and which the rows are checked against.  A categorical modifier carries
    its declared levels, in the order an :class:`Indicators` basis reports them; the first
    level is the reference.  No column has both.

    ``outside`` says what happens to a row whose continuous modifier lies outside its
    domain, or whose categorical modifier holds an undeclared level.  ``"refuse"``
    raises.  ``"clip"`` maps a continuous value to the nearest end of its domain, a known
    fixed transform, so the working model is then a function of the clipped modifier;
    an undeclared level still raises.  ``"clip"`` needs a domain for every continuous
    modifier.

    The schema checks rows against what it declares.  It does not know the adjustment
    set: :class:`cleverly.ConditionalContrast` checks that each modifier is an
    adjustment column.

    Parameters
    ----------
    columns : sequence of str
        The modifier columns, in the order a basis reads them.
    domains : mapping of str to (float, float), optional
        The declared domain of a continuous modifier.
    levels : mapping of str to sequence, optional
        The declared levels of a categorical modifier, reference level first.
    outside : {"refuse", "clip"}
        What a continuous value outside its domain does.

    See Also
    --------
    cleverly.ConditionalContrast : The estimand that declares the modifiers.
    Linear : A basis that needs no domain.

    Examples
    --------
    >>> from cleverly.bases import Modifiers
    >>> modifiers = Modifiers(["age", "sex"], domains={"age": (18, 90)}, levels={"sex": ("f", "m")})
    >>> modifiers.domain("age"), modifiers.levels_of("sex")
    ((18.0, 90.0), ('f', 'm'))
    >>> modifiers.continuous
    ('age',)
    """

    columns: Sequence[str]
    domains: Mapping[str, tuple[float, float]] | tuple[tuple[str, tuple[float, float]], ...] = ()
    levels: Mapping[str, Sequence[Any]] | tuple[tuple[str, tuple[Any, ...]], ...] = ()
    outside: Literal["refuse", "clip"] = "refuse"

    def __post_init__(self) -> None:
        if isinstance(self.columns, str) or not isinstance(self.columns, Sequence):
            raise DataError(
                f"Modifiers(columns=...) takes a sequence of column names; got {self.columns!r}"
            )
        columns = tuple(self.columns)
        if not columns:
            raise DataError("Modifiers needs at least one column")
        if not all(isinstance(column, str) and column for column in columns):
            raise DataError(f"modifier columns must be non-empty strings; got {columns}")
        if len(set(columns)) != len(columns):
            raise DataError(f"modifier columns must be distinct; got {columns}")
        domains = _pairs("domains", self.domains, _as_domain)
        levels = _pairs("levels", self.levels, _as_levels)
        for label, pairs in (("domains", domains), ("levels", levels)):
            unknown = [key for key, _ in pairs if key not in columns]
            if unknown:
                raise DataError(f"{label} names {unknown}, which are not modifier columns")
        both = sorted({key for key, _ in domains} & {key for key, _ in levels})
        if both:
            raise DataError(
                f"{both} declare both a domain and levels; a modifier is either continuous "
                "(a domain) or categorical (levels)"
            )
        if self.outside not in ("refuse", "clip"):
            raise DataError(f"outside must be 'refuse' or 'clip'; got {self.outside!r}")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "levels", levels)
        if self.outside == "clip":
            undeclared = [c for c in self.continuous if c not in dict(domains)]
            if undeclared:
                raise _refuse_sample_basis(
                    f"outside='clip' maps a value to the nearest end of its domain, and "
                    f"{undeclared} declare no domain, so the ends would be read off the "
                    "sample."
                )

    @property
    def continuous(self) -> tuple[str, ...]:
        """The modifiers with no declared level set, in column order."""
        categorical = set(_as_dict(self.levels))
        return tuple(column for column in self.columns if column not in categorical)

    @property
    def categorical(self) -> tuple[str, ...]:
        """The modifiers with a declared level set, in column order."""
        categorical = set(_as_dict(self.levels))
        return tuple(column for column in self.columns if column in categorical)

    def domain(self, column: str) -> tuple[float, float] | None:
        """The declared domain of ``column``, or ``None``.

        Parameters
        ----------
        column : str
            A modifier column.

        Returns
        -------
        tuple of float or None
            ``(low, high)``, or ``None`` when no domain is declared.
        """
        domain: tuple[float, float] | None = _as_dict(self.domains).get(column)
        return domain

    def levels_of(self, column: str) -> tuple[Any, ...] | None:
        """The declared levels of ``column``, or ``None``.

        Parameters
        ----------
        column : str
            A modifier column.

        Returns
        -------
        tuple or None
            The levels, reference first, or ``None`` when none are declared.
        """
        levels: tuple[Any, ...] | None = _as_dict(self.levels).get(column)
        return levels

    def values(self, frame: Any, columns: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """Read and check the modifier columns of ``frame``.

        A continuous modifier comes back as floats, clipped under ``outside="clip"``; a
        categorical one as its raw values, each checked against the declared levels.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns, in pandas, polars or a mapping of arrays.
        columns : sequence of str, optional
            The modifiers to read.  Every declared modifier when omitted.

        Returns
        -------
        dict of str to ndarray
            One array per modifier.

        Raises
        ------
        DataError
            If a column is missing, a continuous value is not a finite number, a value
            lies outside its domain under ``outside="refuse"``, or a categorical value is
            not a declared level.
        """
        wanted = self.columns if columns is None else tuple(columns)
        out: dict[str, np.ndarray] = {}
        for column in wanted:
            if column not in self.columns:
                raise DataError(f"{column!r} is not one of the declared modifiers {self.columns}")
            raw = _read_column(frame, column)
            levels = self.levels_of(column)
            if levels is not None:
                _check_levels(column, raw, levels)
                out[column] = raw
                continue
            values = _numeric(column, raw)
            domain = self.domain(column)
            if domain is not None:
                values = _check_domain(column, values, domain, clip=self.outside == "clip")
            out[column] = values
        return out


def _check_levels(column: str, raw: np.ndarray, levels: tuple[Any, ...]) -> None:
    known = np.zeros(raw.shape[0], dtype=bool)
    for level in levels:
        known |= _equals(raw, level)
    if not np.all(known):
        unknown = sorted({repr(value) for value in raw[~known].tolist()})[:5]
        raise DataError(
            f"the modifier {column!r} holds values {', '.join(unknown)} outside its declared "
            f"levels {levels}. A level set is declared before the fit; add the level to "
            "Modifiers(levels=...) or drop the rows."
        )


def _equals(raw: np.ndarray, level: Any) -> np.ndarray:
    """Elementwise ``raw == level`` that treats ``0`` and ``0.0`` as one level."""
    try:
        return np.asarray(raw == level, dtype=bool).reshape(-1)
    except (TypeError, ValueError):  # pragma: no cover - mixed object arrays
        return np.array([value == level for value in raw.tolist()], dtype=bool)


def _check_domain(
    column: str, values: np.ndarray, domain: tuple[float, float], *, clip: bool
) -> np.ndarray:
    low, high = domain
    outside = (values < low) | (values > high)
    if not np.any(outside):
        return values
    if clip:
        return np.clip(values, low, high)
    raise DataError(
        f"{int(np.count_nonzero(outside))} row(s) of the modifier {column!r} lie outside its "
        f"declared domain [{low:g}, {high:g}] (from {float(values.min()):g} to "
        f"{float(values.max()):g}). Widen the declared domain, drop the rows, or declare "
        "Modifiers(outside='clip') to map them to the nearest end."
    )


# ----------------------------------------------------------------------- the bases


class Basis(Protocol):  # numpydoc ignore=PR01
    """What a working model of the modifiers provides.

    The protocol is not runtime-checkable: :func:`check_basis` decides what a working
    model may be, by its concrete type and its declaration.

    Attributes
    ----------
    terms : tuple of str
    n_terms : int
    max_size : int
    basis_kind : {"known", "estimated"}
    columns : tuple of str
    """

    @property
    def terms(self) -> tuple[str, ...]:  # pragma: no cover - protocol
        """One name per design column."""
        ...

    @property
    def n_terms(self) -> int:  # pragma: no cover - protocol
        """The number of design columns."""
        ...

    @property
    def max_size(self) -> int:  # pragma: no cover - protocol
        """The largest size :meth:`truncate` accepts."""
        ...

    @property
    def basis_kind(self) -> str:  # pragma: no cover - protocol
        """``"known"`` for a basis fixed before the fit."""
        ...

    @property
    def columns(self) -> tuple[str, ...]:  # pragma: no cover - protocol
        """The modifier columns the basis reads."""
        ...

    def evaluate(
        self, frame: Any, modifiers: Modifiers | None = None
    ) -> np.ndarray:  # pragma: no cover - protocol
        """The ``(n, p)`` design at the rows of ``frame``.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.
        modifiers : Modifiers or None
            The declared domains and levels, when the basis reads them.

        Returns
        -------
        ndarray
            One column per term.
        """
        ...

    def truncate(self, k: int) -> Basis:  # pragma: no cover - protocol
        """The same basis with each component cut to its first ``k`` functions.

        Parameters
        ----------
        k : int
            The number of functions to keep in each component.

        Returns
        -------
        Basis
            The truncated basis.
        """
        ...


class _ShippedBasis:
    """Shared behaviour of the shipped bases: names, the intercept, and the block form.

    A subclass supplies :meth:`_block`, the ``(n, q)`` non-intercept columns and their
    names, and :meth:`truncate`.  ``evaluate`` prepends the intercept when the basis
    declares one.
    """

    intercept: bool = True

    @property
    def basis_kind(self) -> str:
        """Always ``"known"``: a shipped basis is fixed by its arguments."""
        return "known"

    #: The modifier columns the basis reads: a field or a property of each subclass.
    columns: Any

    @property
    def max_size(self) -> int:
        """The largest size :meth:`truncate` accepts."""
        return 1

    def _block_names(self) -> tuple[str, ...]:
        raise NotImplementedError  # pragma: no cover - every subclass overrides

    def _block(self, values: Mapping[str, np.ndarray], modifiers: Modifiers | None) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover - every subclass overrides

    @property
    def terms(self) -> tuple[str, ...]:
        """One name per design column, intercept first when there is one."""
        names = self._block_names()
        return (INTERCEPT, *names) if self.intercept else names

    @property
    def n_terms(self) -> int:
        """The number of design columns."""
        return len(self.terms)

    def evaluate(self, frame: Any, modifiers: Modifiers | None = None) -> np.ndarray:
        """The ``(n, p)`` design at the rows of ``frame``.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.
        modifiers : Modifiers, optional
            The declared schema, which supplies a domain or a level set the basis does
            not carry itself, and checks every row.

        Returns
        -------
        ndarray
            The design, one row per row of ``frame`` and one column per term.

        Raises
        ------
        CapabilityError
            If the basis needs a domain or a level set that neither it nor ``modifiers``
            declares (a basis computed from the sample, R3).
        DataError
            If a column is missing or a row breaks its declaration.
        """
        values = _values_for(self.columns, frame, modifiers, self)
        n = _length(values, frame)
        block = self._block(values, modifiers).reshape(n, -1)
        if self.intercept:
            block = np.column_stack([np.ones(n), block]) if block.size else np.ones((n, 1))
        return np.asarray(block, dtype=float)


def _length(values: Mapping[str, np.ndarray], frame: Any) -> int:
    if values:
        return int(next(iter(values.values())).shape[0])
    if hasattr(frame, "shape"):
        return int(frame.shape[0])
    if isinstance(frame, Mapping) and frame:
        return int(np.asarray(next(iter(frame.values()))).reshape(-1).shape[0])
    return len(frame)


def _values_for(
    columns: tuple[str, ...], frame: Any, modifiers: Modifiers | None, basis: Any
) -> dict[str, np.ndarray]:
    """The raw (checked, possibly clipped) values of the columns a basis reads."""
    if modifiers is not None:
        missing = [column for column in columns if column not in modifiers.columns]
        if missing:
            raise DataError(
                f"the basis reads {missing}, which are not declared in Modifiers "
                f"{modifiers.columns}"
            )
        return modifiers.values(frame, columns)
    out: dict[str, np.ndarray] = {}
    for column in columns:
        raw = _read_column(frame, column)
        levels = getattr(basis, "level_set", lambda _c: None)(column)
        if levels is not None:
            _check_levels(column, raw, levels)
            out[column] = raw
            continue
        values = _numeric(column, raw)
        domain = getattr(basis, "domain_of", lambda _c: None)(column)
        if domain is not None:
            values = _check_domain(column, values, domain, clip=False)
        out[column] = values
    return out


def _as_columns(columns: Any, owner: str) -> tuple[str, ...]:
    if isinstance(columns, str):
        columns = (columns,)
    if not isinstance(columns, Sequence) or not columns:
        raise DataError(f"{owner} takes a non-empty sequence of column names; got {columns!r}")
    names = tuple(columns)
    if not all(isinstance(name, str) and name for name in names):
        raise DataError(f"{owner} columns must be non-empty strings; got {names}")
    if len(set(names)) != len(names):
        raise DataError(f"{owner} columns must be distinct; got {names}")
    return names


def _as_size(value: Any, owner: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise DataError(f"{owner} needs an integer size of at least {minimum}; got {value!r}")
    return int(value)


@dataclass(frozen=True)
class Constant(_ShippedBasis):
    """The intercept alone: the projection onto it is the average effect.

    Parameters
    ----------
    intercept : bool
        Always ``True``; a constant basis is its intercept.
    """

    intercept: bool = True

    def __post_init__(self) -> None:
        if self.intercept is not True:
            raise DataError("Constant() is the intercept column; intercept=False leaves nothing")

    @property
    def columns(self) -> tuple[str, ...]:
        """No modifier column is read."""
        return ()

    @property
    def max_size(self) -> int:
        """``0``: the constant has no size to truncate."""
        return 0

    def _block_names(self) -> tuple[str, ...]:
        return ()

    def _block(self, values: Mapping[str, np.ndarray], modifiers: Modifiers | None) -> np.ndarray:
        return np.zeros((0,))

    def evaluate(self, frame: Any, modifiers: Modifiers | None = None) -> np.ndarray:
        """A column of ones, one row per row of ``frame``.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows; only their number is read.
        modifiers : Modifiers, optional
            Ignored, accepted for the common signature.

        Returns
        -------
        ndarray
            ``(n, 1)`` ones.
        """
        del modifiers
        return np.ones((_length({}, frame), 1))

    def truncate(self, k: int) -> Constant:
        """Return this basis; a constant has nothing to truncate.

        Parameters
        ----------
        k : int
            Any non-negative size.

        Returns
        -------
        Constant
            This basis.
        """
        _as_size(k, "Constant.truncate", minimum=0)
        return self


@dataclass(frozen=True)
class Linear(_ShippedBasis):
    """Each modifier as it is, beside an intercept.  Needs no domain.

    Parameters
    ----------
    columns : sequence of str
        The continuous (or 0/1) modifiers, one design column each.
    intercept : bool
        Whether to prepend the intercept column.

    Examples
    --------
    >>> from cleverly.bases import Linear
    >>> basis = Linear(["W1", "W3"])
    >>> basis.terms
    ('(intercept)', 'W1', 'W3')
    >>> basis.evaluate({"W1": [0.5, -1.0], "W3": [2.0, 0.0]}).tolist()
    [[1.0, 0.5, 2.0], [1.0, -1.0, 0.0]]
    """

    columns: Sequence[str]
    intercept: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", _as_columns(self.columns, "Linear"))

    def _block_names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def _block(self, values: Mapping[str, np.ndarray], modifiers: Modifiers | None) -> np.ndarray:
        return np.column_stack([_numeric(name, values[name]) for name in self.columns])

    def truncate(self, k: int) -> Linear:
        """Return this basis: a linear term has one size.

        Parameters
        ----------
        k : int
            A size of at least one.

        Returns
        -------
        Linear
            This basis.
        """
        _as_size(k, "Linear.truncate")
        return self


def _domain_for(
    column: str,
    declared: Mapping[str, tuple[float, float]],
    modifiers: Modifiers | None,
    owner: str,
) -> tuple[float, float]:
    if column in declared:
        return declared[column]
    domain = None if modifiers is None else modifiers.domain(column)
    if domain is None:
        raise _refuse_sample_basis(
            f"{owner} reads the domain of {column!r}, and neither the basis nor Modifiers "
            "declares one."
        )
    return domain


@dataclass(frozen=True)
class Cosine(_ShippedBasis):
    r"""Cosines of each modifier on its declared domain.

    Column :math:`j = 1, \ldots, \text{size}` of modifier :math:`v` on the domain
    :math:`[l, h]` is :math:`\sqrt 2 \cos(j \pi (v - l)/(h - l))`, orthonormal under the
    uniform law on the domain.  The domain comes from ``domains`` or from the declared
    :class:`Modifiers`, never from the rows.

    Parameters
    ----------
    columns : sequence of str
        The continuous modifiers.
    size : int
        The number of cosines per modifier.
    domains : mapping of str to (float, float), optional
        Domains carried by the basis itself, which take precedence over the schema's.
    intercept : bool
        Whether to prepend the intercept column.
    """

    columns: Sequence[str]
    size: int = 1
    domains: Mapping[str, tuple[float, float]] | tuple[tuple[str, tuple[float, float]], ...] = ()
    intercept: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", _as_columns(self.columns, "Cosine"))
        object.__setattr__(self, "size", _as_size(self.size, "Cosine"))
        object.__setattr__(self, "domains", _pairs("domains", self.domains, _as_domain))

    def domain_of(self, column: str) -> tuple[float, float] | None:
        """The domain this basis carries for ``column``, or ``None``.

        Parameters
        ----------
        column : str
            A modifier column.

        Returns
        -------
        tuple of float or None
            ``(low, high)``, or ``None``.
        """
        return dict(self.domains).get(column)

    @property
    def max_size(self) -> int:
        """The declared size."""
        return int(self.size)

    def _block_names(self) -> tuple[str, ...]:
        return tuple(f"cos{j}({c})" for c in self.columns for j in range(1, self.size + 1))

    def _block(self, values: Mapping[str, np.ndarray], modifiers: Modifiers | None) -> np.ndarray:
        blocks: list[np.ndarray] = []
        declared = _as_dict(self.domains)
        for column in self.columns:
            low, high = _domain_for(column, declared, modifiers, "Cosine")
            u = (_numeric(column, values[column]) - low) / (high - low)
            blocks.extend(math.sqrt(2.0) * np.cos(j * math.pi * u) for j in range(1, self.size + 1))
        return np.column_stack(blocks)

    def truncate(self, k: int) -> Cosine:
        """The first ``k`` cosines of each modifier.

        Parameters
        ----------
        k : int
            A size between one and :attr:`max_size`.

        Returns
        -------
        Cosine
            The truncated basis, whose span lies inside this one's.
        """
        k = _as_size(k, "Cosine.truncate")
        if k > self.size:
            raise DataError(f"Cosine has size {self.size}; it cannot be truncated to {k}")
        return Cosine(self.columns, k, self.domains, self.intercept)


def _bspline_columns(u: np.ndarray, n_interior: int, degree: int) -> np.ndarray:
    """B-spline basis on ``[0, 1]`` with equally spaced interior knots (Cox--de Boor)."""
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    n_basis = n_interior + degree + 1
    # Degree 0: indicator of each knot span; the last span is closed at 1.
    spans = len(knots) - 1
    basis = np.zeros((u.shape[0], spans))
    for i in range(spans):
        left, right = knots[i], knots[i + 1]
        if right > left:
            basis[:, i] = (u >= left) & (u < right)
    last = max(i for i in range(spans) if knots[i + 1] > knots[i])
    basis[u >= 1.0, last] = 1.0
    for d in range(1, degree + 1):
        nxt = np.zeros((u.shape[0], spans - d))
        for i in range(spans - d):
            left_den = knots[i + d] - knots[i]
            right_den = knots[i + d + 1] - knots[i + 1]
            term = np.zeros(u.shape[0])
            if left_den > 0:
                term += (u - knots[i]) / left_den * basis[:, i]
            if right_den > 0:
                term += (knots[i + d + 1] - u) / right_den * basis[:, i + 1]
            nxt[:, i] = term
        basis = nxt
    return basis[:, :n_basis]


@dataclass(frozen=True)
class BSpline(_ShippedBasis):
    """B-splines of each modifier with equally spaced knots on its declared domain.

    The knots divide the declared domain into ``n_interior_knots + 1`` equal spans; they
    are never sample quantiles.  The B-splines of one modifier sum to one, so under an
    intercept the first of them is dropped to keep the design of full rank.

    Parameters
    ----------
    columns : sequence of str
        The continuous modifiers.
    n_interior_knots : int
        The number of interior knots per modifier.
    degree : int
        The polynomial degree, 3 for cubic splines.
    domains : mapping of str to (float, float), optional
        Domains carried by the basis itself, which take precedence over the schema's.
    intercept : bool
        Whether to prepend the intercept column.
    """

    columns: Sequence[str]
    n_interior_knots: int = 3
    degree: int = 3
    domains: Mapping[str, tuple[float, float]] | tuple[tuple[str, tuple[float, float]], ...] = ()
    intercept: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", _as_columns(self.columns, "BSpline"))
        object.__setattr__(
            self, "n_interior_knots", _as_size(self.n_interior_knots, "BSpline", minimum=0)
        )
        object.__setattr__(self, "degree", _as_size(self.degree, "BSpline degree"))
        object.__setattr__(self, "domains", _pairs("domains", self.domains, _as_domain))

    def domain_of(self, column: str) -> tuple[float, float] | None:
        """The domain this basis carries for ``column``, or ``None``.

        Parameters
        ----------
        column : str
            A modifier column.

        Returns
        -------
        tuple of float or None
            ``(low, high)``, or ``None``.
        """
        return dict(self.domains).get(column)

    @property
    def max_size(self) -> int:
        """The declared number of interior knots."""
        return int(self.n_interior_knots)

    def _per_column(self) -> int:
        count = self.n_interior_knots + self.degree + 1
        return count - 1 if self.intercept else count

    def _block_names(self) -> tuple[str, ...]:
        start = 2 if self.intercept else 1
        return tuple(
            f"bs{j}({c})" for c in self.columns for j in range(start, start + self._per_column())
        )

    def _block(self, values: Mapping[str, np.ndarray], modifiers: Modifiers | None) -> np.ndarray:
        blocks = []
        declared = dict(self.domains)
        for column in self.columns:
            low, high = _domain_for(column, declared, modifiers, "BSpline")
            u = (_numeric(column, values[column]) - low) / (high - low)
            full = _bspline_columns(u, self.n_interior_knots, self.degree)
            blocks.append(full[:, 1:] if self.intercept else full)
        return np.column_stack(blocks)

    def truncate(self, k: int) -> BSpline:
        """The same splines with ``k`` equally spaced interior knots.

        Parameters
        ----------
        k : int
            A knot count between zero and :attr:`max_size`.

        Returns
        -------
        BSpline
            The basis with ``k`` interior knots.
        """
        k = _as_size(k, "BSpline.truncate", minimum=0)
        if k > self.n_interior_knots:
            raise DataError(
                f"BSpline has {self.n_interior_knots} interior knots; it cannot be truncated to {k}"
            )
        return BSpline(self.columns, k, self.degree, self.domains, self.intercept)


@dataclass(frozen=True)
class Indicators(_ShippedBasis):
    """Indicators of the declared levels of one categorical modifier.

    Under an intercept the reference level, the first declared one, is dropped.

    Parameters
    ----------
    column : str
        The categorical modifier.
    levels : sequence, optional
        The levels carried by the basis itself, which take precedence over the schema's.
    intercept : bool
        Whether to prepend the intercept column.
    """

    column: str
    levels: Sequence[Any] = ()
    intercept: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise DataError(f"Indicators takes one column name; got {self.column!r}")
        levels = tuple(self.levels)
        if levels:
            levels = _as_levels(self.column, levels)
        object.__setattr__(self, "levels", levels)

    @property
    def columns(self) -> tuple[str, ...]:
        """The one modifier column."""
        return (self.column,)

    def level_set(self, column: str) -> tuple[Any, ...] | None:
        """The levels this basis carries for ``column``, or ``None``.

        Parameters
        ----------
        column : str
            A modifier column.

        Returns
        -------
        tuple or None
            The levels, or ``None`` when the basis carries none.
        """
        return tuple(self.levels) if column == self.column and self.levels else None

    def _levels(self, modifiers: Modifiers | None) -> tuple[Any, ...]:
        if self.levels:
            return tuple(self.levels)
        levels = None if modifiers is None else modifiers.levels_of(self.column)
        if levels is None:
            raise _refuse_sample_basis(
                f"Indicators reads the levels of {self.column!r}, and neither the basis nor "
                "Modifiers declares them."
            )
        return levels

    def _block_names(self) -> tuple[str, ...]:
        if not self.levels:
            raise DataError(
                f"Indicators({self.column!r}) reads its levels from Modifiers, so its terms "
                "are named only once they are resolved; call cleverly.bases.resolve(basis, "
                "modifiers) first"
            )
        kept = self.levels[1:] if self.intercept else self.levels
        return tuple(f"{self.column}[{level}]" for level in kept)

    def named(self, modifiers: Modifiers | None) -> Indicators:
        """This basis with its levels resolved from ``modifiers``.

        Parameters
        ----------
        modifiers : Modifiers or None
            The declared schema.

        Returns
        -------
        Indicators
            A basis that carries its levels, so its terms are named.
        """
        return Indicators(self.column, self._levels(modifiers), self.intercept)

    def _block(self, values: Mapping[str, np.ndarray], modifiers: Modifiers | None) -> np.ndarray:
        levels = self._levels(modifiers)
        raw = values[self.column]
        _check_levels(self.column, raw, levels)
        kept = levels[1:] if self.intercept else levels
        return np.column_stack([_equals(raw, level).astype(float) for level in kept])

    def truncate(self, k: int) -> Indicators:
        """Return this basis: a categorical modifier enters whole at every size.

        Parameters
        ----------
        k : int
            A size of at least one.

        Returns
        -------
        Indicators
            This basis.
        """
        _as_size(k, "Indicators.truncate")
        return self


def _components(bases: Sequence[Any], owner: str) -> tuple[Any, ...]:
    out = tuple(bases)
    if not out:
        raise DataError(f"{owner} needs at least one basis")
    for basis in out:
        check_basis(basis)
    return out


def _non_intercept(basis: Any, frame: Any, modifiers: Modifiers | None) -> np.ndarray:
    design = np.asarray(basis.evaluate(frame, modifiers), dtype=float)
    terms = tuple(basis.terms) if basis.terms else ()
    if terms and terms[0] == INTERCEPT:
        return design[:, 1:]
    return design


def resolve(basis: Any, modifiers: Modifiers | None) -> Any:
    """``basis`` with every level set it reads taken from ``modifiers``.

    An :class:`Indicators` basis that carries no levels reads them from the schema, and
    its terms are named only once they are resolved.  Every other basis comes back as it
    is, with its components resolved.

    Parameters
    ----------
    basis : Basis
        A declared basis.
    modifiers : Modifiers or None
        The declared schema.

    Returns
    -------
    Basis
        The resolved basis.

    Raises
    ------
    CapabilityError
        If an :class:`Indicators` basis has levels in neither place (R3).
    """
    if isinstance(basis, Indicators):
        return basis.named(modifiers)
    if isinstance(basis, (Additive, Tensor, Interactions)):
        return basis.named(modifiers)
    return basis


_resolved = resolve


def _block_terms(basis: Any) -> tuple[str, ...]:
    terms = tuple(basis.terms)
    return terms[1:] if terms and terms[0] == INTERCEPT else terms


@dataclass(frozen=True, init=False)
class Additive:
    """Several bases side by side, sharing one intercept.

    Parameters
    ----------
    *bases : Basis
        The components.  Each component's own intercept is dropped.
    intercept : bool
        Whether to prepend the shared intercept column.

    Examples
    --------
    >>> from cleverly.bases import Additive, Cosine, Indicators, Modifiers
    >>> modifiers = Modifiers(["W1", "W2"], domains={"W1": (-1, 1)}, levels={"W2": (0, 1)})
    >>> basis = Additive(Cosine(["W1"], 2), Indicators("W2")).named(modifiers)
    >>> basis.terms
    ('(intercept)', 'cos1(W1)', 'cos2(W1)', 'W2[1]')
    """

    bases: tuple[Any, ...]
    intercept: bool

    def __init__(self, *bases: Any, intercept: bool = True) -> None:
        object.__setattr__(self, "bases", _components(bases, "Additive"))
        object.__setattr__(self, "intercept", intercept)

    @property
    def basis_kind(self) -> str:
        """``"known"`` when every component is."""
        return "known" if all(b.basis_kind == "known" for b in self.bases) else "estimated"

    @property
    def columns(self) -> tuple[str, ...]:
        """The modifier columns the components read, each once, in order."""
        seen: list[str] = []
        for basis in self.bases:
            seen.extend(c for c in basis.columns if c not in seen)
        return tuple(seen)

    @property
    def max_size(self) -> int:
        """The largest size of a component."""
        return max(int(b.max_size) for b in self.bases)

    @property
    def terms(self) -> tuple[str, ...]:
        """The shared intercept, then each component's non-intercept terms."""
        names = tuple(t for b in self.bases for t in _block_terms(b))
        return (INTERCEPT, *names) if self.intercept else names

    @property
    def n_terms(self) -> int:
        """The number of design columns."""
        return len(self.terms)

    def named(self, modifiers: Modifiers | None) -> Additive:
        """This basis with every level set resolved from ``modifiers``.

        Parameters
        ----------
        modifiers : Modifiers or None
            The declared schema.

        Returns
        -------
        Additive
            A basis whose terms carry the level names.
        """
        return Additive(*(_resolved(b, modifiers) for b in self.bases), intercept=self.intercept)

    def evaluate(self, frame: Any, modifiers: Modifiers | None = None) -> np.ndarray:
        """The ``(n, p)`` design: the intercept, then each component's columns.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.
        modifiers : Modifiers, optional
            The declared schema.

        Returns
        -------
        ndarray
            The design.
        """
        blocks = [_non_intercept(b, frame, modifiers) for b in self.bases]
        n = blocks[0].shape[0]
        if self.intercept:
            blocks.insert(0, np.ones((n, 1)))
        return np.column_stack(blocks)

    def truncate(self, k: int) -> Additive:
        """Each component cut to ``min(k, its size)``.

        Parameters
        ----------
        k : int
            A size between one and :attr:`max_size`.

        Returns
        -------
        Additive
            The truncated basis.
        """
        k = _as_size(k, "Additive.truncate")
        return Additive(
            *(b.truncate(min(k, b.max_size)) for b in self.bases), intercept=self.intercept
        )


@dataclass(frozen=True)
class Tensor:
    """The tensor product of two bases: every product of a column of each.

    With both intercepts kept, the product carries the intercept, each basis's own terms,
    and their interactions.  ``Tensor(basis, Linear(["A"]))`` is the basis, the column
    ``A`` and ``A`` times every term of the basis.

    Parameters
    ----------
    left : Basis
        The first factor.
    right : Basis
        The second factor.
    """

    left: Any
    right: Any

    def __post_init__(self) -> None:
        check_basis(self.left)
        check_basis(self.right)

    @property
    def basis_kind(self) -> str:
        """``"known"`` when both factors are."""
        return "known" if self.left.basis_kind == self.right.basis_kind == "known" else "estimated"

    @property
    def columns(self) -> tuple[str, ...]:
        """The modifier columns either factor reads."""
        left = tuple(self.left.columns)
        return left + tuple(c for c in self.right.columns if c not in left)

    @property
    def max_size(self) -> int:
        """The larger of the factors' sizes."""
        return max(int(self.left.max_size), int(self.right.max_size))

    @property
    def terms(self) -> tuple[str, ...]:
        """``a:b`` for every pair of terms, the intercept absorbed."""
        out = []
        for a in self.left.terms:
            for b in self.right.terms:
                if a == INTERCEPT:
                    out.append(b)
                elif b == INTERCEPT:
                    out.append(a)
                else:
                    out.append(f"{a}:{b}")
        return tuple(out)

    @property
    def n_terms(self) -> int:
        """The number of design columns."""
        return len(self.terms)

    def named(self, modifiers: Modifiers | None) -> Tensor:
        """This basis with every level set resolved from ``modifiers``.

        Parameters
        ----------
        modifiers : Modifiers or None
            The declared schema.

        Returns
        -------
        Tensor
            A basis whose terms carry the level names.
        """
        return Tensor(_resolved(self.left, modifiers), _resolved(self.right, modifiers))

    def evaluate(self, frame: Any, modifiers: Modifiers | None = None) -> np.ndarray:
        """The ``(n, p_left * p_right)`` design of products.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.
        modifiers : Modifiers, optional
            The declared schema.

        Returns
        -------
        ndarray
            The design.
        """
        a = np.asarray(self.left.evaluate(frame, modifiers), dtype=float)
        b = np.asarray(self.right.evaluate(frame, modifiers), dtype=float)
        return np.einsum("ip,iq->ipq", a, b).reshape(a.shape[0], -1)

    def truncate(self, k: int) -> Tensor:
        """Both factors cut to ``min(k, their size)``.

        Parameters
        ----------
        k : int
            A size of at least one.

        Returns
        -------
        Tensor
            The truncated basis.
        """
        k = _as_size(k, "Tensor.truncate")
        return Tensor(
            self.left.truncate(min(k, self.left.max_size)),
            self.right.truncate(min(k, self.right.max_size)),
        )


@dataclass(frozen=True)
class Interactions:
    """An additive basis with the products of its components up to an interaction order.

    Order 1 is the additive basis.  Order 2 adds, for every pair of components, the
    products of their non-intercept columns; order 3 every triple; and so on.

    Parameters
    ----------
    basis : Additive
        The additive basis whose components interact.
    order : int
        The largest number of components in one product.
    """

    basis: Additive
    order: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.basis, Additive):
            raise DataError("interactions() takes an Additive basis of components")
        order = _as_size(self.order, "interactions order")
        if order > len(self.basis.bases):
            raise DataError(
                f"an interaction order of {order} needs at least {order} components; the "
                f"basis has {len(self.basis.bases)}"
            )
        object.__setattr__(self, "order", order)

    @property
    def basis_kind(self) -> str:
        """The additive basis's declaration."""
        return self.basis.basis_kind

    @property
    def columns(self) -> tuple[str, ...]:
        """The modifier columns the components read."""
        return self.basis.columns

    @property
    def max_size(self) -> int:
        """The additive basis's size."""
        return self.basis.max_size

    def _groups(self) -> list[tuple[int, ...]]:
        count = len(self.basis.bases)
        return [g for s in range(2, self.order + 1) for g in combinations(range(count), s)]

    @property
    def terms(self) -> tuple[str, ...]:
        """The additive terms, then one name per product column."""
        blocks = [_block_terms(b) for b in self.basis.bases]
        out = list(self.basis.terms)
        for group in self._groups():
            names: list[str] = [""]
            for index in group:
                names = [f"{a}:{b}" if a else b for a in names for b in blocks[index]]
            out.extend(names)
        return tuple(out)

    @property
    def n_terms(self) -> int:
        """The number of design columns."""
        return len(self.terms)

    def named(self, modifiers: Modifiers | None) -> Interactions:
        """This basis with every level set resolved from ``modifiers``.

        Parameters
        ----------
        modifiers : Modifiers or None
            The declared schema.

        Returns
        -------
        Interactions
            A basis whose terms carry the level names.
        """
        return Interactions(self.basis.named(modifiers), self.order)

    def evaluate(self, frame: Any, modifiers: Modifiers | None = None) -> np.ndarray:
        """The additive design, then the product columns.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.
        modifiers : Modifiers, optional
            The declared schema.

        Returns
        -------
        ndarray
            The design.
        """
        design = [self.basis.evaluate(frame, modifiers)]
        blocks = [_non_intercept(b, frame, modifiers) for b in self.basis.bases]
        n = design[0].shape[0]
        for group in self._groups():
            product = np.ones((n, 1))
            for index in group:
                product = np.einsum("ip,iq->ipq", product, blocks[index]).reshape(n, -1)
            design.append(product)
        return np.column_stack(design)

    def truncate(self, k: int) -> Interactions:
        """The additive basis cut to ``k``, with the same order.

        Parameters
        ----------
        k : int
            A size of at least one.

        Returns
        -------
        Interactions
            The truncated basis.
        """
        return Interactions(self.basis.truncate(k), self.order)


def interactions(basis: Additive, order: int) -> Additive | Interactions:
    """An additive basis with its components' products up to ``order``.

    Parameters
    ----------
    basis : Additive
        The additive basis.
    order : int
        The interaction order; ``1`` returns ``basis`` itself.

    Returns
    -------
    Additive or Interactions
        The basis with interactions.
    """
    if _as_size(order, "interactions order") == 1:
        return basis
    return Interactions(basis, order)


@dataclass(frozen=True)
class UserBasis:
    """A basis written as a function of the modifier rows, with its declaration.

    ``function(frame)`` returns the ``(n, p)`` design.  A callable can close over any
    estimate and no code can inspect a closure, so whether it is a known function is a
    declaration: ``basis_kind="known"`` states that the function was fixed without reading
    this study's rows.  ``None`` and ``"estimated"`` are refused when a model is declared.

    Parameters
    ----------
    function : callable
        ``frame -> (n, p)`` array.
    terms : tuple of str
        One name per column.
    columns : tuple of str
        The modifier columns the function reads.
    basis_kind : {"known", "estimated"} or None
        The declaration.
    """

    function: Callable[[Any], Any]
    terms: tuple[str, ...]
    columns: tuple[str, ...]
    basis_kind: BasisKind | None = None

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise DataError("UserBasis(function=...) takes a callable frame -> (n, p) array")
        terms = tuple(str(t) for t in self.terms)
        if not terms or len(set(terms)) != len(terms):
            raise DataError(f"UserBasis terms must be distinct and non-empty; got {terms}")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "columns", _as_columns(self.columns, "UserBasis"))
        if self.basis_kind not in (None, "known", "estimated"):
            raise DataError(
                f"basis_kind must be 'known', 'estimated' or None; got {self.basis_kind!r}"
            )

    @property
    def n_terms(self) -> int:
        """The number of design columns."""
        return len(self.terms)

    @property
    def max_size(self) -> int:
        """``1``: a written basis has one size."""
        return 1

    def evaluate(self, frame: Any, modifiers: Modifiers | None = None) -> np.ndarray:
        """``function`` at the checked modifier rows.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.
        modifiers : Modifiers, optional
            The declared schema; the rows are checked against it first.

        Returns
        -------
        ndarray
            The design.
        """
        values = _values_for(tuple(self.columns), frame, modifiers, self)
        design = self.function(values)
        if hasattr(design, "to_numpy"):
            design = design.to_numpy()
        design = np.asarray(design, dtype=float)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        n = _length(values, frame)
        if design.shape != (n, self.n_terms):
            raise DataError(
                f"UserBasis returned shape {design.shape}; expected ({n}, {self.n_terms})"
            )
        if not np.all(np.isfinite(design)):
            raise DataError("UserBasis returned a non-finite value")
        return design

    def truncate(self, k: int) -> UserBasis:
        """Return this basis: a written basis has one size.

        Parameters
        ----------
        k : int
            A size of at least one.

        Returns
        -------
        UserBasis
            This basis.
        """
        _as_size(k, "UserBasis.truncate")
        return self


_SHIPPED = (Constant, Linear, Cosine, BSpline, Indicators, Additive, Tensor, Interactions)


def check_basis(basis: Any) -> None:
    """Refuse a working model that is not a declared, known basis (R3).

    A shipped basis is known by its type.  A :class:`UserBasis` must declare
    ``basis_kind="known"``.  Any other object is refused, because nothing shows that its
    functions were not built from the rows.

    Parameters
    ----------
    basis : Any
        The declared working model.

    Raises
    ------
    CapabilityError
        If the basis is not a shipped basis or a ``UserBasis`` declared known.
    """
    if isinstance(basis, _SHIPPED):
        return
    if isinstance(basis, UserBasis):
        if basis.basis_kind == "known":
            return
        detail = (
            "UserBasis(basis_kind=None) does not declare whether its function is known."
            if basis.basis_kind is None
            else "UserBasis(basis_kind='estimated') declares a basis built from the sample."
        )
        raise _refuse_sample_basis(detail)
    raise _refuse_sample_basis(
        f"{type(basis).__name__} is not a basis from cleverly.bases, so nothing shows that "
        "its functions were fixed before the fit; wrap a known function in "
        "UserBasis(..., basis_kind='known')."
    )


#: How a sieve's size is chosen: a fixed integer, cross-validation, or a declared schedule.
SieveSize = int | Literal["cv"] | Callable[[int], int] | Mapping[int, int]


@dataclass(frozen=True)
class Sieve:
    """A nested family of bases and the rule that picks its size.

    ``size`` is an integer (a fixed size), ``"cv"`` (chosen by cross-validation, which
    makes the target data-adaptive), or a declared schedule: a callable ``n -> size`` or
    a mapping from sample sizes to sizes, read at the largest key at most ``n``.  A fixed
    size or a schedule resolves to one known basis before the fit.

    Parameters
    ----------
    basis : Basis
        The largest member of the family; smaller members are its truncations.
    size : int, "cv", callable or mapping
        How the member is chosen.
    """

    basis: Any
    size: Any = "cv"

    def __post_init__(self) -> None:
        check_basis(self.basis)
        size = self.size
        if isinstance(size, str):
            if size != "cv":
                raise DataError(f"Sieve size must be an int, 'cv' or a schedule; got {size!r}")
        elif isinstance(size, Mapping):
            if not size:
                raise DataError("a Sieve schedule mapping needs at least one entry")
            object.__setattr__(
                self, "size", tuple(sorted((int(k), int(v)) for k, v in size.items()))
            )
        elif callable(size) or isinstance(size, tuple):
            pass
        else:
            _as_size(size, "Sieve", minimum=0)

    @property
    def chosen_by_cv(self) -> bool:
        """Whether the size is chosen by cross-validation."""
        return isinstance(self.size, str)

    def size_at(self, n: int) -> int:
        """The size this sieve declares for a sample of ``n`` rows.

        Parameters
        ----------
        n : int
            The sample size.

        Returns
        -------
        int
            The declared size.

        Raises
        ------
        DataError
            If the size is chosen by cross-validation, or a schedule gives no size at ``n``.
        """
        if isinstance(self.size, str):
            raise DataError("a Sieve with size='cv' has no declared size")
        if isinstance(self.size, tuple):
            eligible = [value for key, value in self.size if key <= n]
            if not eligible:
                raise DataError(f"the Sieve schedule declares no size at n = {n}")
            return int(eligible[-1])
        if callable(self.size):
            return _as_size(self.size(int(n)), "Sieve schedule", minimum=0)
        return int(self.size)

    def at(self, n: int) -> Any:
        """The basis this sieve declares for a sample of ``n`` rows.

        Parameters
        ----------
        n : int
            The sample size.

        Returns
        -------
        Basis
            The truncated basis.
        """
        size = self.size_at(n)
        if size == 0:
            return Constant()
        return self.basis.truncate(min(size, int(self.basis.max_size)))

    @classmethod
    def default(cls, modifiers: Modifiers, *, max_size: int = 6, size: Any = "cv") -> Sieve:
        """Additive cosines of the continuous modifiers and indicators of the categorical.

        Parameters
        ----------
        modifiers : Modifiers
            The declared schema; every continuous modifier needs a domain.
        max_size : int
            The largest number of cosines per continuous modifier.
        size : int, "cv", callable or mapping
            How the member is chosen.

        Returns
        -------
        Sieve
            The default sieve.
        """
        components: list[Any] = []
        if modifiers.continuous:
            components.append(Cosine(modifiers.continuous, max_size))
        components.extend(Indicators(column) for column in modifiers.categorical)
        return cls(Additive(*components).named(modifiers), size)
