r"""The band-rank placebo: a dose whose declared support reads the covariates or the history.

A declared support ``Support(lower, upper)`` with a callable bound gives each unit its own
band ``[L(h), U(h)]``. A plain permutation of the doses ignores the receiving unit's band, so a
placebo dose can leave it and the refit stops on the declared-support check. This module
permutes each dose's **position within its own band** and maps it into the receiving unit's
band (X82 part J, ``X82-J-plan.md`` sections 3.1 to 3.3):

======================================  =================================================
form of the band on every recording row  position ``u`` and placebo dose ``A'``
======================================  =================================================
both bounds finite (``scale``)          ``u = (A - L) / (U - L)``, ``A' = L' + u (U' - L')``
only the lower bound finite (``lower``)  ``u = A - L``, ``A' = L' + u``
only the upper bound finite (``upper``)  ``u = U - A``, ``A' = U' - u``
======================================  =================================================

The primed bounds are the receiving unit's, read on the frame the refit reads. A point dose
reads its covariates. A component of a point vector reads its chain-rule frame: the covariates,
the donor's categorical components and the placebo doses of the components before it. A
longitudinal node reads the placebo policy frame ``H'_t``, which holds the placebo doses of the
earlier nodes.

**Exact null.** The drawn positions are a permutation, an i.i.d. draw or a Markov-kernel draw of
the position column, so they are independent of every non-treatment variable given the data. The
placebo dose is a function of the receiving unit's own frame and of its drawn position. So
``E[Y | A', W] = E[Y | W]`` on the pooled placebo law, and by induction over the nodes every
sequential regression is free of the placebo treatment. Every policy contrast is exactly 0.

**Refusals.** The draws below refuse, before any refit, each case they do not implement
(``IMPLEMENTATION_GAP``, roadmap ``X82``) and each declaration that fails on a placebo
history (``INVALID_DECLARATION``):

=====================================================  ===================================
case                                                   reason
=====================================================  ===================================
a bound finite on some rows and infinite on others      gap: a draw stratified by the bound
                                                       pattern is exact but is not built
a declared atom outside some recording row's band       gap: a draw stratified by the atoms
                                                       each band holds is exact but is not
                                                       built
a held longitudinal node with a callable band          gap: the copied dose can leave a
                                                       later band
an atom or a zero-width band on a vector component      gap
or on a longitudinal node
a callable bound that is not a row-wise function of     invalid: it can read the observed
the frame                                              dose and leak it into the placebo
an inverted, ``NaN`` or wrongly infinite band on a      invalid
placebo history
an empty cluster band on a placebo history              gap
=====================================================  ===================================
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import narwhals as nw
import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import CapabilityError, RefusalReason
from ..interventions.policy import JointComponent, Support, _atom_index, _component_frame
from ..utils.frames import frame_from_dict
from ._markov_placebo import MarkovKernel

__all__ = [
    "BAND_SENTENCE",
    "NEAREST_DONORS",
    "PanelBands",
    "PointBands",
    "is_banded",
    "placebo_bands",
]

#: The donors a position node draws among (``X82-J-plan.md`` section 3.2, review M1).
NEAREST_DONORS = 5

#: The sentence a banded placebo adds to its reading (``X82-J-plan.md`` section 4).
BAND_SENTENCE = (
    "The placebo dose keeps each unit's declared band. It moves each dose's position within its "
    "band, so the placebo dose depends on the covariates through the band. A rejection can also "
    "show a treatment model that cannot follow the band"
)

Form = Literal["scale", "lower", "upper"]

_GAP_ALTERNATIVE = "The random_common_cause, subset and bootstrap_stability tests run on this fit."


def _gap(condition: str) -> CapabilityError:
    return CapabilityError(
        condition,
        reason=RefusalReason.IMPLEMENTATION_GAP,
        alternative=_GAP_ALTERNATIVE,
        roadmap="X82",
    )


def _invalid(condition: str) -> CapabilityError:
    return CapabilityError(condition, reason=RefusalReason.INVALID_DECLARATION)


def is_banded(support: Any) -> bool:
    """Whether ``support`` is a :class:`Support` with a callable bound.

    Parameters
    ----------
    support : object
        A declared support, or ``None``.

    Returns
    -------
    bool
        ``True`` when a bound is a function of the frame.
    """
    return isinstance(support, Support) and (callable(support.lower) or callable(support.upper))


def _permuted_frame(frame: Any, order: IntArray) -> Any:
    """``frame`` with its rows in ``order``, in the same backend."""
    native = nw.from_native(frame, eager_only=True)
    columns = {name: native[name].to_numpy()[order] for name in native.columns}
    return frame_from_dict(columns, backend=nw.get_native_namespace(native).__name__)


def check_row_wise(support: Support, frame: Callable[[], Any], n: int, *, node: str) -> None:
    """Refuse a callable bound that is not a row-wise function of the frame (review m2).

    The bounds are read on the frame with its rows shifted by one, and must equal the shifted
    bounds. A function of a whole covariate column, such as a rank, passes. A closure over a
    per-row array outside the frame, such as the observed dose, fails.

    Parameters
    ----------
    support : Support
        The declared support.
    frame : callable
        Returns the frame the bounds read.
    n : int
        The number of rows.
    node : str
        The node the refusal names.

    Raises
    ------
    CapabilityError
        If the shifted frame does not give the shifted bounds.
    """
    if n < 2:
        return
    order = np.roll(np.arange(n), 1)
    lower, upper = support.bounds(frame, n)
    shifted = _permuted_frame(frame(), order)
    moved_lower, moved_upper = support.bounds(lambda: shifted, n)
    for expected, found in ((lower[order], moved_lower), (upper[order], moved_upper)):
        scale = max(1.0, float(np.nanmax(np.abs(np.where(np.isfinite(expected), expected, 0.0)))))
        if not np.allclose(found, expected, rtol=1e-9, atol=1e-12 * scale, equal_nan=True):
            raise _invalid(
                f"the declared support of node {node!r} is not a row-wise function of its frame: "
                "on the frame with its rows shifted by one, the bounds are not the shifted "
                "bounds. The band-rank placebo needs each band to be a function of its own "
                "row's frame, because a bound that reads the observed dose or another per-row "
                "array outside the frame leaks the treatment into the placebo."
            )


def band_form(lower: FloatArray, upper: FloatArray, rows: BoolArray, *, node: str) -> Form:
    """The form of a band on the rows that record a dose.

    Parameters
    ----------
    lower, upper : ndarray
        ``(n,)`` bounds.
    rows : ndarray
        ``(n,)`` the rows that record a dose.
    node : str
        The node the refusal names.

    Returns
    -------
    {"scale", "lower", "upper"}
        Both bounds finite on every row, only the lower one, or only the upper one.

    Raises
    ------
    CapabilityError
        If a bound is finite on some rows and infinite on others, or both bounds are
        infinite on every row.
    """
    low = np.isfinite(lower[rows])
    high = np.isfinite(upper[rows])
    if low.all() and high.all():
        return "scale"
    if low.all() and not high.any():
        return "lower"
    if high.all() and not low.any():
        return "upper"
    if not low.any() and not high.any():
        raise _gap(
            f"the declared support of node {node!r} is a function with both bounds infinite on "
            "every row. The band-rank placebo has no position to move in an unbounded band; "
            "declare the support with constant bounds instead."
        )
    raise _gap(
        f"the declared support of node {node!r} has a bound that is finite on some rows and "
        "infinite on others. Not implemented: a band-rank draw stratified by the bound pattern "
        "is exact, but it changes the dose law by row."
    )


def positions(dose: FloatArray, lower: FloatArray, upper: FloatArray, form: Form) -> FloatArray:
    """Each dose's position within its band, ``nan`` on a zero-width band.

    Parameters
    ----------
    dose, lower, upper : ndarray
        ``(n,)`` doses and bounds.
    form : {"scale", "lower", "upper"}
        The band's form.

    Returns
    -------
    ndarray
        ``(n,)`` positions in ``[0, 1]`` for ``"scale"``, and offsets from the finite bound
        otherwise.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        if form == "lower":
            return dose - lower
        if form == "upper":
            return upper - dose
        width = upper - lower
        u = np.where(width > 0, (dose - lower) / np.where(width > 0, width, 1.0), np.nan)
    return np.clip(u, 0.0, 1.0)


def mapped(u: FloatArray, lower: FloatArray, upper: FloatArray, form: Form) -> FloatArray:
    """The dose at position ``u`` of each band, kept inside it against rounding.

    Parameters
    ----------
    u, lower, upper : ndarray
        ``(n,)`` positions and bounds.
    form : {"scale", "lower", "upper"}
        The band's form.

    Returns
    -------
    ndarray
        ``(n,)`` doses. A zero-width band gives its bound.
    """
    if form == "lower":
        return np.maximum(lower + u, lower)
    if form == "upper":
        return np.minimum(upper - u, upper)
    width = upper - lower
    dose = np.where(width > 0, lower + u * width, lower)
    return np.minimum(np.maximum(dose, lower), upper)


def check_placebo_band(
    lower: FloatArray, upper: FloatArray, rows: BoolArray, form: Form, *, node: str, where: str
) -> None:
    """Refuse a band that a placebo history makes inverted, ``NaN`` or of another form.

    Parameters
    ----------
    lower, upper : ndarray
        ``(n,)`` bounds on the placebo frame.
    rows : ndarray
        ``(n,)`` the rows that take a placebo dose.
    form : {"scale", "lower", "upper"}
        The form the observed bands have.
    node : str
        The node the refusal names.
    where : str
        ``"row"`` or ``"cluster"``, the unit the refusal names.

    Raises
    ------
    CapabilityError
        If any such row's band is invalid.
    """
    low = np.isfinite(lower)
    high = np.isfinite(upper)
    expected_low = form in ("scale", "lower")
    expected_high = form in ("scale", "upper")
    with np.errstate(invalid="ignore"):
        bad = rows & (
            np.isnan(lower)
            | np.isnan(upper)
            | (low != expected_low)
            | (high != expected_high)
            | (lower > upper)
        )
    if not bad.any():
        return
    first = int(np.flatnonzero(bad)[0])
    if where == "cluster" and lower[first] > upper[first]:
        raise _gap(
            f"on a placebo history the bands of the rows of cluster {first} at node {node!r} "
            f"share no dose (their intersection is [{float(lower[first]):g}, "
            f"{float(upper[first]):g}]), so the cluster has no common placebo dose. Not "
            "implemented: a cluster draw that conditions on a non-empty intersection."
        )
    raise _invalid(
        f"on a placebo history the declared support of node {node!r} gives {where} {first} the "
        f"band [{float(lower[first]):g}, {float(upper[first]):g}], which is inverted, NaN or "
        f"not of the {form!r} form the observed bands have. The band-rank placebo needs a valid "
        "band on every history it draws."
    )


# ------------------------------------------------------------------ point treatment


def _covariate_frame(data: Any) -> Callable[[], Any]:
    from ..interventions.policy import lazy_frame

    return lazy_frame(
        lambda: data.frame_like(
            {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        )
    )


@dataclass(frozen=True)
class _ComponentBand:
    """One banded continuous component of a point vector."""

    index: int
    support: Support
    form: Form
    position: FloatArray


@dataclass(frozen=True)
class PointBands:
    """The band-rank draw of a point-treatment dose whose declared support is a function.

    Parameters
    ----------
    vector : bool
        Whether the treatment is a vector.
    form : {"scale", "lower", "upper"} or None
        The band's form; ``None`` on a vector.
    lower, upper : ndarray or None
        ``(n,)`` bounds of a dose of one column.
    pool : ndarray
        The rows whose position enters the draw: on a dose of one column, the rows that
        record a dose on a band of positive width; on a vector, every row.
    kinds : ndarray
        Per pool row, the index of the declared atom it records, or ``-1``.
    position : ndarray
        Per pool row, the position of its dose, ``nan`` for an atom.
    atoms : tuple of float
        The declared atoms of a dose of one column.
    components : tuple of JointComponent
        The components of a vector, ``()`` otherwise.
    bands : tuple of _ComponentBand
        The banded continuous components of a vector, in column order.
    """

    vector: bool
    form: Form | None
    lower: FloatArray | None
    upper: FloatArray | None
    pool: IntArray
    kinds: IntArray
    position: FloatArray
    atoms: tuple[float, ...]
    components: tuple[JointComponent, ...] = ()
    bands: tuple[_ComponentBand, ...] = ()

    @classmethod
    def prepare(cls, data: Any) -> PointBands | None:
        """Read the bands of ``data``, or return ``None`` when no bound is a function.

        Parameters
        ----------
        data : CausalData
            The fit's data.

        Returns
        -------
        PointBands or None
            The draw, or ``None`` for the plain permutation.

        Raises
        ------
        CapabilityError
            On a case the module docstring refuses.
        """
        if getattr(data, "treatment_kind", None) == "vector":
            return cls._prepare_vector(data)
        support = getattr(data, "treatment_support", None)
        if not is_banded(support):
            return None
        assert isinstance(support, Support)
        node = str(data.treatment_name)
        dose = np.asarray(data.treatment, dtype=float)
        n = dose.size
        recorded = (
            np.asarray(data.treatment_recorded, dtype=bool)
            if data.has_missing_treatment
            else np.ones(n, dtype=bool)
        )
        frame = _covariate_frame(data)
        check_row_wise(support, frame, n, node=node)
        lower, upper = support.bounds(frame, n)
        form = band_form(lower, upper, recorded, node=node)
        zero = recorded & (upper == lower) if form == "scale" else np.zeros(n, dtype=bool)
        pool = np.flatnonzero(recorded & ~zero)
        atoms = tuple(float(atom) for atom in data.dose_atoms)
        for atom in atoms:
            outside = ~((lower[pool] <= atom) & (atom <= upper[pool]))
            if outside.any():
                row = int(pool[np.flatnonzero(outside)[0]])
                raise _gap(
                    f"the declared atom {atom:g} of node {node!r} lies outside the band "
                    f"[{float(lower[row]):g}, {float(upper[row]):g}] of row {row}. Not "
                    "implemented: a band-rank draw stratified by which atoms each band holds is "
                    "exact, but it changes the atom share by row."
                )
        kinds = _atom_index(dose[pool], atoms) if atoms else np.full(pool.size, -1, dtype=np.int64)
        position = positions(dose[pool], lower[pool], upper[pool], form)
        position = np.where(kinds >= 0, np.nan, position)
        return cls(False, form, lower, upper, pool, kinds, position, atoms)

    @classmethod
    def _prepare_vector(cls, data: Any) -> PointBands | None:
        supports: Mapping[str, Any] = data.component_supports
        components = tuple(
            JointComponent(name, levels)
            for name, levels in zip(data.component_names, data.component_levels, strict=True)
        )
        if not any(
            component.continuous and is_banded(supports.get(component.name))
            for component in components
        ):
            return None
        natural = np.asarray(data.treatment_components, dtype=float)
        n = natural.shape[0]
        everyone = np.ones(n, dtype=bool)
        frame_with = _vector_frame_with(data)
        atoms = data.component_atoms
        bands: list[_ComponentBand] = []
        for index, component in enumerate(components):
            support = supports.get(component.name)
            if not component.continuous or not is_banded(support):
                continue
            assert isinstance(support, Support)
            name = component.name
            if atoms.get(name):
                raise _gap(
                    f"component {name!r} declares atoms and a support whose bound is a "
                    "function. Not implemented: the band-rank placebo of a vector carries no "
                    "atom."
                )
            frame = _component_frame(components, index, natural, frame_with)
            check_row_wise(support, frame, n, node=name)
            lower, upper = support.bounds(frame, n)
            form = band_form(lower, upper, everyone, node=name)
            if form == "scale" and np.any(upper == lower):
                row = int(np.flatnonzero(upper == lower)[0])
                raise _gap(
                    f"component {name!r} has a zero-width band on row {row}. Not implemented: "
                    "a vector draw moves whole rows, so the row's component has no position to "
                    "give."
                )
            bands.append(
                _ComponentBand(
                    index, support, form, positions(natural[:, index], lower, upper, form)
                )
            )
        return cls(
            True,
            None,
            None,
            None,
            np.arange(n),
            np.full(n, -1, dtype=np.int64),
            np.full(n, np.nan),
            (),
            components,
            tuple(bands),
        )

    def draw(self, data: Any, rng: np.random.Generator, mode: str | None) -> FloatArray:
        """One placebo treatment: the band positions permuted or drawn, then mapped.

        Parameters
        ----------
        data : CausalData
            The fit's data.
        rng : numpy.random.Generator
            The generator.
        mode : {"permute", "draw"} or None
            ``"draw"`` draws the donors i.i.d.; otherwise they are a permutation.

        Returns
        -------
        ndarray
            ``(n,)`` doses, or ``(n, d)`` component values of a vector.

        Raises
        ------
        CapabilityError
            If a placebo frame gives a vector component an invalid band.
        """
        size = int(self.pool.size)
        donors = rng.integers(0, size, size=size) if mode == "draw" else rng.permutation(size)
        if self.vector:
            return self._draw_vector(data, donors)
        assert self.lower is not None and self.upper is not None and self.form is not None
        out = np.asarray(data.treatment, dtype=float).copy()
        rows = self.pool
        doses = mapped(self.position[donors], self.lower[rows], self.upper[rows], self.form)
        kinds = self.kinds[donors]
        if self.atoms:
            doses = np.where(kinds >= 0, np.asarray(self.atoms)[np.maximum(kinds, 0)], doses)
        out[rows] = doses
        return out

    def _draw_vector(self, data: Any, donors: IntArray) -> FloatArray:
        natural = np.asarray(data.treatment_components, dtype=float)
        placebo = natural[donors].copy()
        n = placebo.shape[0]
        everyone = np.ones(n, dtype=bool)
        frame_with = _vector_frame_with(data)
        for band in self.bands:
            name = self.components[band.index].name
            frame = _component_frame(self.components, band.index, placebo, frame_with)
            lower, upper = band.support.bounds(frame, n)
            check_placebo_band(lower, upper, everyone, band.form, node=name, where="row")
            placebo[:, band.index] = mapped(band.position[donors], lower, upper, band.form)
        return placebo

    def extremes(self) -> tuple[FloatArray, FloatArray] | None:
        """The doses at the pooled smallest and largest positions, on a dose of one column.

        Returns
        -------
        tuple of ndarray or None
            ``(n,)`` doses with every pool row at the smallest, then the largest, position.
            ``None`` on a vector or when no row holds a continuous position.
        """
        if self.vector or not np.isfinite(self.position).any():
            return None
        assert self.lower is not None and self.upper is not None and self.form is not None
        out: list[FloatArray] = []
        rows = self.pool
        for value in (np.nanmin(self.position), np.nanmax(self.position)):
            doses = np.full(self.lower.size, np.nan)
            doses[rows] = mapped(
                np.full(rows.size, value), self.lower[rows], self.upper[rows], self.form
            )
            out.append(doses)
        return out[0], out[1]


def _vector_frame_with(data: Any) -> Callable[[Mapping[str, Any]], Any]:
    """The frame a point vector's component reads, as :func:`_vector_setup` builds it."""

    def frame_with(extra: Mapping[str, Any]) -> Any:
        payload = {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
        payload.update(extra)
        return data.frame_like(payload)

    return frame_with


# ------------------------------------------------------------------ longitudinal


@dataclass(frozen=True)
class PanelBands:
    """The band-rank draw of a longitudinal panel with a node whose support is a function.

    Parameters
    ----------
    kernel : MarkovKernel
        The kernel fitted to the positions at the banded nodes and to the treatments at the
        others.
    forms : tuple
        Per node, the band's form, or ``None`` at a node the draw does not map.
    """

    kernel: MarkovKernel
    forms: tuple[Form | None, ...]

    @property
    def banded(self) -> tuple[bool, ...]:
        """Whether each node is mapped from a position."""
        return tuple(form is not None for form in self.forms)

    @classmethod
    def prepare(cls, data: Any) -> PanelBands | None:
        """Read the banded nodes of ``data``, or return ``None`` when no bound is a function.

        Parameters
        ----------
        data : LongitudinalData
            The fit's panel.

        Returns
        -------
        PanelBands or None
            The draw, or ``None`` for the plain Markov kernel.

        Raises
        ------
        CapabilityError
            On a case the module docstring refuses.
        """
        supports = tuple(getattr(data, "dose_supports", ()) or ())
        if not any(is_banded(support) for support in supports):
            return None
        base = MarkovKernel.fit(data)
        n_times = int(data.n_times)
        treatment = np.asarray(data.treatment, dtype=float)
        trajectories = base.trajectories.copy()
        forms: list[Form | None] = []
        for time in range(1, n_times + 1):
            support = data.node_support(time)
            if data.node_kind(time) != "continuous" or not is_banded(support):
                forms.append(None)
                continue
            assert isinstance(support, Support)
            name = str(data.treatment_names[time - 1])
            if data.treatment_held:
                raise _gap(
                    f"the held node {name!r} declares a support whose bound is a function. Not "
                    "implemented: the held dose is copied to every node, where it can leave a "
                    "band that reads the later history."
                )
            if data.node_atoms(time):
                raise _gap(
                    f"node {name!r} declares atoms and a support whose bound is a function. Not "
                    "implemented: the longitudinal band-rank kernel carries no atom."
                )
            at_risk = ~np.isnan(treatment[:, time - 1])
            frame = _policy_frame(data, time)
            check_row_wise(support, frame, int(data.n), node=name)
            lower, upper = support.bounds(frame, int(data.n))
            form = band_form(lower, upper, at_risk, node=name)
            unit_lower, unit_upper = _unit_bounds(base, lower, upper, at_risk)
            trajectories[:, time - 1] = positions(
                base.trajectories[:, time - 1], unit_lower, unit_upper, form
            )
            forms.append(form)
        nearest = tuple(
            NEAREST_DONORS if forms[time - 1] is not None and time >= 2 else 1
            for time in range(1, n_times + 1)
        )
        kernel = replace(base, trajectories=trajectories, nearest=nearest)
        return cls(kernel, tuple(forms))

    def placebo(self, data: Any, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        """One masked placebo treatment matrix, and the kernel draw it was mapped from.

        Parameters
        ----------
        data : LongitudinalData
            The panel the draw was prepared on.
        rng : numpy.random.Generator
            The generator.

        Returns
        -------
        doses : ndarray
            ``(n, T)`` placebo treatment, ``nan`` exactly where the observed one is.
        drawn : ndarray
            ``(n, T)`` the masked kernel draw: positions at the banded nodes, the treatment
            elsewhere. The known mechanism of a categorical node reads it.

        Raises
        ------
        CapabilityError
            If a placebo history gives a band that is invalid, or a cluster no common dose.
        """
        kernel = self.kernel
        units = kernel.trajectories.shape[0]
        drawn = kernel.draw(rng, units)
        rows = drawn if kernel.cluster_codes is None else drawn[kernel.cluster_codes]
        observed = np.asarray(data.treatment, dtype=float)
        masked = np.where(np.isnan(observed), np.nan, rows)
        return self._mapped(data, masked, drawn, strict=True), masked

    def extremes(self, data: Any) -> tuple[FloatArray, FloatArray]:
        """The placebo matrices with every banded node at its pooled extreme position.

        Each banded node takes the smallest (then the largest) observed position on every unit,
        mapped on the placebo history those extremes give. The other nodes keep the observed
        treatment. A row whose band is invalid on that history holds ``nan``.

        Parameters
        ----------
        data : LongitudinalData
            The panel.

        Returns
        -------
        tuple of ndarray
            The two ``(n, T)`` matrices.
        """
        kernel = self.kernel
        observed = np.asarray(data.treatment, dtype=float)
        out: list[FloatArray] = []
        for pick in (np.nanmin, np.nanmax):
            units = kernel.trajectories.copy()
            for time, form in enumerate(self.forms, start=1):
                if form is None:
                    continue
                column = kernel.trajectories[:, time - 1]
                # Every unit takes the extreme, so a row at risk on the placebo history
                # whose observed band had zero width is checked as well.
                units[:, time - 1] = float(pick(column)) if np.isfinite(column).any() else 0.0
            rows = units if kernel.cluster_codes is None else units[kernel.cluster_codes]
            masked = np.where(np.isnan(observed), np.nan, np.where(np.isnan(rows), 0.0, rows))
            for time, form in enumerate(self.forms, start=1):
                if form is None:
                    masked[:, time - 1] = observed[:, time - 1]
            out.append(self._mapped(data, masked, units, strict=False))
        return out[0], out[1]

    def _mapped(
        self, data: Any, masked: FloatArray, drawn: FloatArray, *, strict: bool
    ) -> FloatArray:
        """Map the positions of ``masked`` node by node on the placebo history."""
        kernel = self.kernel
        observed = np.asarray(data.treatment, dtype=float)
        matrix = masked.copy()
        n = int(data.n)
        for time, form in enumerate(self.forms, start=1):
            if form is None:
                continue
            support = data.node_support(time)
            name = str(data.treatment_names[time - 1])
            at_risk = ~np.isnan(observed[:, time - 1])
            partial = replace(data, treatment=matrix)
            frame = _policy_frame(partial, time)
            lower, upper = support.bounds(frame, n)
            codes = kernel.cluster_codes
            if codes is None:
                unit_lower, unit_upper, unit_rows = lower, upper, at_risk
                unit_position = drawn[:, time - 1]
                where = "row"
            else:
                unit_lower, unit_upper = _cluster_bounds(
                    codes, lower, upper, at_risk, drawn.shape[0]
                )
                unit_rows = np.zeros(drawn.shape[0], dtype=bool)
                unit_rows[codes[at_risk]] = True
                unit_position = drawn[:, time - 1]
                where = "cluster"
            if strict:
                check_placebo_band(unit_lower, unit_upper, unit_rows, form, node=name, where=where)
                width = unit_upper - unit_lower if form == "scale" else np.ones_like(unit_lower)
                missing = unit_rows & np.isnan(unit_position) & (width > 0)
                if missing.any():
                    raise _gap(
                        f"no unit at risk at node {name!r} records a dose on a band of positive "
                        "width, so the band-rank placebo has no position to give a placebo "
                        "history whose band has positive width."
                    )
                valid = unit_rows
            else:
                with np.errstate(invalid="ignore"):
                    valid = unit_rows & ~(
                        np.isnan(unit_lower) | np.isnan(unit_upper) | (unit_lower > unit_upper)
                    )
            unit_dose = np.where(
                valid,
                mapped(np.nan_to_num(unit_position, nan=0.0), unit_lower, unit_upper, form),
                np.nan,
            )
            dose = unit_dose if codes is None else unit_dose[codes]
            matrix[:, time - 1] = np.where(at_risk, dose, np.nan)
        return matrix


def _policy_frame(data: Any, time: int) -> Callable[[], Any]:
    from ..interventions.policy import lazy_frame

    return lazy_frame(lambda: data.policy_frame(time))


def _cluster_bounds(
    codes: IntArray, lower: FloatArray, upper: FloatArray, rows: BoolArray, units: int
) -> tuple[FloatArray, FloatArray]:
    """The intersection of the bands of each cluster's rows in ``rows``."""
    unit_lower = np.full(units, -np.inf)
    unit_upper = np.full(units, np.inf)
    np.maximum.at(unit_lower, codes[rows], lower[rows])
    np.minimum.at(unit_upper, codes[rows], upper[rows])
    return unit_lower, unit_upper


def _unit_bounds(
    kernel: MarkovKernel, lower: FloatArray, upper: FloatArray, rows: BoolArray
) -> tuple[FloatArray, FloatArray]:
    """The bands of the kernel's sampling units: the rows, or each cluster's intersection."""
    if kernel.cluster_codes is None:
        return lower, upper
    return _cluster_bounds(
        kernel.cluster_codes, lower, upper, rows, int(kernel.trajectories.shape[0])
    )


def placebo_bands(result: Any) -> PointBands | PanelBands | None:
    """The band-rank draw of ``result``'s data, or ``None`` for the plain placebo.

    Parameters
    ----------
    result : object
        A fitted result.

    Returns
    -------
    PointBands, PanelBands or None
        The draw.

    Raises
    ------
    CapabilityError
        On a case the module docstring refuses.
    """
    from .refute import _is_panel

    data = result.data
    if _is_panel(result):
        return PanelBands.prepare(data)
    return PointBands.prepare(data)
