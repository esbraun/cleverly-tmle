r"""The parametric g-formula: forward simulation of the fitted longitudinal laws.

The estimator fits one parametric model per time-varying covariate column at nodes
:math:`2, \ldots, T`, one outcome model (or one hazard model per node on a survival outcome),
and, for a modified treatment policy or an odds tilt, one natural-treatment model per node.  It
then replicates every unit :math:`m` times, draws each copy's covariates forward from the fitted
models with the treatment set by the regimen, and averages the outcome functional over the
copies (Robins 1986; McGrath et al. 2020).  The outcome of a copy is the predicted outcome mean
at the last node, or on a survival outcome the discrete-time cumulative incidence
:math:`F_j(k) = \sum_{t \le k} S(t - 1)\, h_{jt}` along the copy's hazards.  A drawn outcome
never enters.

**Random numbers.**  Every draw is an inverse-CDF transform of a uniform
:math:`U(\text{stream}, \text{unit}, \text{copy}, \text{node}, \text{column})`, a pure function
of its five indices (a counter-based Philox4x32-10 generator).  Nothing about a draw depends on
the fitted parameters, so a regimen reads the same uniforms as every other regimen (common
random numbers).  Stream 0 gives the copies behind :math:`\hat\Phi_i`.  Stream 1 gives the
copies behind the parameter derivatives, so the derivative error is independent of the estimate.

**The curve.**  Augment each observation :math:`O_i` with its uniforms :math:`U_i`.  The pairs
are independent and identically distributed, and
:math:`\hat\psi = P_n[w\,\bar f(O, U; \hat\theta)] / P_n w` is a two-step estimator whose models
solve their own score equations.  The Jacobian of the stacked equations is block diagonal, and the
curve is

.. math::

    \mathrm{IF}_i = \frac{w_i}{\bar w}(\hat\Phi_i - \hat\psi)
        + \sum_k D_k^\top M_k^{-1} S_{k,i},

with :math:`D_k = \partial\psi / \partial\theta_k`, :math:`M_k` the information over :math:`n`
and :math:`S_{k,i}` the score of model :math:`k` at unit :math:`i`.  A Gaussian model adds the
variance equation :math:`P_n[w R (r^2 - \sigma^2)] = 0` with the maximum-likelihood divisor, so
the cross block vanishes.  The curve uses :math:`\hat\Phi_i` and not :math:`\Phi_i`, so its
variance is the statistical variance plus the simulation variance
:math:`E[w^2 \sigma^2] / (m \bar w^2)`.

**Derivatives.**  The outcome mean and the hazards are not drawn: their derivative is pathwise.
A drawn variable's derivative is the score-function identity
:math:`\partial_\theta E_\theta[f] = E_\theta[(f - b)\,\partial_\theta \log p_\theta]`, with
:math:`b` the leave-one-copy-out mean of the unit's other copies.  The baseline is independent of
the copy's own draw, so the identity holds exactly.

The estimate extrapolates every model along the simulated histories.  It exists without
positivity, so the mechanism diagnostics of the fit stay the check on overlap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
from scipy.special import ndtri

from .._typing import BoolArray, FloatArray, IntArray, Learner
from ..data.validate import arm_indicators
from ..exceptions import CapabilityError, DataError, RefusalReason
from ..interventions.base import Stochastic, _as_array, check_regime_density
from ..interventions.incremental import Incremental
from ..interventions.policy import (
    POLICY_TYPES,
    discrete_assignments,
    lazy_frame,
    policy_branches,
)
from ..learners._fitting import (
    as_target,
    clone_learner,
    fit_learner,
    predict_mean,
    predict_probabilities,
)
from ..learners.glm import GLM
from ..utils.bounds import OutcomeScaler
from .baselines import _solve
from .data import LongitudinalData
from .regimen import DynamicRegimen, Plan, Regimen, _frame_by_level
from .sequential import (
    _FILLER,
    FitSimulation,
    Mechanism,
    RegimenFit,
    SequentialStep,
    _clever_covariate,
)

__all__ = [
    "COVARIATE_KINDS",
    "ExactTerms",
    "GFormulaReport",
    "GFormulaSettings",
    "ModelBook",
    "ModelRecord",
    "SupportRow",
    "Variable",
    "counter_uniforms",
    "has_fixed_dose",
    "monte_carlo_share",
    "normal_learner_refusal",
    "philox4x32",
    "plan_node",
    "resolve_covariates",
    "run_gformula",
]

#: The covariate types ``covariate_types=`` accepts.
COVARIATE_KINDS: tuple[str, ...] = ("binary", "categorical", "normal")

#: Stream 0 gives the copies behind the estimate, stream 1 the copies behind the derivatives.
STREAM_ESTIMATE = 0
STREAM_DERIVATIVE = 1

#: Column indices of the draws that are not a covariate.  A covariate column takes its position
#: in the node's variable order, so these never collide with it.
_COL_TREATMENT = 60000
_COL_NATURAL = 60001
_COL_BRANCH = 60002

#: The most elements one chunk of simulated paths holds in its tables and score rows.
_CHUNK_ELEMENTS = 20_000_000

#: A fixed dose further than this many fitted standard deviations from the fitted mean dose of a
#: simulated history lies off the support the model has seen.
SUPPORT_Z = 2.0

_MASK32 = 0xFFFFFFFF
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85


# ------------------------------------------------------------------------------- random numbers


def philox4x32(
    counter: Sequence[Any], key: Sequence[int], rounds: int = 10
) -> tuple[Any, Any, Any, Any]:
    """The Philox4x32 block function on arrays of counters (Salmon et al. 2011).

    Parameters
    ----------
    counter : sequence of 4 array-like
        The four 32-bit counter words.  Scalars broadcast against arrays.
    key : sequence of 2 int
        The two 32-bit key words.
    rounds : int
        The number of rounds.  Ten is the published generator.

    Returns
    -------
    tuple
        The four 32-bit output words, as ``uint32`` arrays.
    """
    c0, c1, c2, c3 = (np.asarray(word, dtype=np.uint32) for word in np.broadcast_arrays(*counter))
    k0, k1 = int(key[0]) & _MASK32, int(key[1]) & _MASK32
    multiplier0, multiplier1 = np.uint64(_PHILOX_M0), np.uint64(_PHILOX_M1)
    shift = np.uint64(32)
    for _ in range(rounds):
        product0 = multiplier0 * c0.astype(np.uint64)
        product1 = multiplier1 * c2.astype(np.uint64)
        high0 = (product0 >> shift).astype(np.uint32)
        low0 = product0.astype(np.uint32)
        high1 = (product1 >> shift).astype(np.uint32)
        low1 = product1.astype(np.uint32)
        c0, c1, c2, c3 = (
            high1 ^ c1 ^ np.uint32(k0),
            low1,
            high0 ^ c3 ^ np.uint32(k1),
            low0,
        )
        k0 = (k0 + _PHILOX_W0) & _MASK32
        k1 = (k1 + _PHILOX_W1) & _MASK32
    return c0, c1, c2, c3


def counter_uniforms(
    seed: int,
    stream: int,
    units: Any,
    copies: Any,
    node: int,
    column: int,
) -> FloatArray:
    """Uniforms on the open interval, each a pure function of its five indices.

    Parameters
    ----------
    seed : int
        The simulation seed, below ``2**64``.
    stream : int
        0 for the copies behind the estimate and 1 for the copies behind the derivatives.
    units : array-like of int
        The unit index of each draw.
    copies : array-like of int
        The copy index of each draw.
    node : int
        The node, counted from one.
    column : int
        The draw's column within the node.

    Returns
    -------
    FloatArray
        One uniform in ``(0, 1)`` per draw, from 53 random bits.
    """
    packed = np.uint32(((int(node) & 0xFFFF) << 16) | (int(column) & 0xFFFF))
    words = philox4x32(
        (
            np.asarray(units, dtype=np.uint32),
            np.asarray(copies, dtype=np.uint32),
            packed,
            np.uint32(int(stream) & _MASK32),
        ),
        (int(seed) & _MASK32, (int(seed) >> 32) & _MASK32),
    )
    high = (words[0] >> np.uint32(5)).astype(np.float64)
    low = (words[1] >> np.uint32(6)).astype(np.float64)
    return np.asarray((high * 67108864.0 + low + 0.5) / 9007199254740992.0, dtype=float)


# ---------------------------------------------------------------------------- covariate typing


@dataclass(frozen=True)
class Variable:
    """One modelled unit of a node's covariate block: a column, or a one-hot group.

    Parameters
    ----------
    node : int
        The node whose block holds the variable, counted from one.
    position : int
        The variable's place in the node's draw order.
    name : str
        The column name, or the group's names joined by ``+``.
    columns : tuple of int
        The positions of the source columns within the node's block.
    column_names : tuple of str
        The names of the source columns.
    kind : {"binary", "categorical", "normal", "raw"}
        The type.  ``"raw"`` is a node-1 column that enters designs as itself and is never
        modelled.
    levels : tuple of float
        A single categorical column's observed values in code order.  A group's levels are
        ``0.0, 1.0, ...``, one per column.  Empty otherwise.
    group : bool
        Whether the variable is a one-hot group of columns.
    """

    node: int
    position: int
    name: str
    columns: tuple[int, ...]
    column_names: tuple[str, ...]
    kind: str
    levels: tuple[float, ...] = ()
    group: bool = False

    @property
    def modelled(self) -> bool:
        """Whether a model is fitted for the variable."""
        return self.node >= 2 and self.kind != "raw"

    @property
    def width(self) -> int:
        """The design columns the variable occupies."""
        return len(self.levels) - 1 if self.kind == "categorical" else 1

    def observed_codes(self, block: FloatArray) -> IntArray:
        """The category code of each row of a categorical variable.

        Parameters
        ----------
        block : FloatArray
            The node's ``(n, p)`` covariate block.

        Returns
        -------
        IntArray
            The code of each row; a row with no observed value reads code zero.
        """
        if self.group:
            sub = np.nan_to_num(block[:, list(self.columns)], nan=0.0)
            return np.asarray(np.argmax(sub, axis=1), dtype=np.int64)
        values = np.nan_to_num(block[:, self.columns[0]], nan=self.levels[0])
        distance = np.abs(values[:, None] - np.asarray(self.levels)[None, :])
        return np.asarray(np.argmin(distance, axis=1), dtype=np.int64)

    def observed_values(self, block: FloatArray) -> FloatArray:
        """The value of each row of a binary, normal or raw variable."""
        return np.nan_to_num(np.asarray(block[:, self.columns[0]], dtype=float), nan=0.0)

    def frame_values(self, payload: Any) -> dict[str, FloatArray]:
        """The source columns' values for a simulated ``payload``: codes, or values.

        Parameters
        ----------
        payload : array-like
            Category codes of a categorical variable, else the values.

        Returns
        -------
        dict of str to FloatArray
            One array per source column, in the coding the data came in.
        """
        array = np.asarray(payload)
        if self.kind != "categorical":
            return {self.column_names[0]: np.asarray(array, dtype=float)}
        codes = array.astype(np.int64)
        if self.group:
            return {
                name: (codes == index).astype(float) for index, name in enumerate(self.column_names)
            }
        return {self.column_names[0]: np.asarray(self.levels, dtype=float)[codes]}


def _declared_kinds(
    covariate_types: Mapping[Any, str] | None, names: Sequence[Sequence[str]]
) -> dict[tuple[str, ...], str]:
    """The declared types keyed by the tuple of source columns, validated against the blocks."""
    if not covariate_types:
        return {}
    present = {name: index for index, block in enumerate(names) for name in block}
    declared: dict[tuple[str, ...], str] = {}
    seen: dict[str, tuple[str, ...]] = {}
    for key, kind in covariate_types.items():
        columns = (key,) if isinstance(key, str) else tuple(key)
        if kind not in COVARIATE_KINDS:
            raise ValueError(
                f"covariate_types= maps {key!r} to {kind!r}; a type is one of "
                f"{list(COVARIATE_KINDS)}"
            )
        missing = [name for name in columns if name not in present]
        if not columns or missing:
            raise DataError(
                f"covariate_types= names {missing or list(columns)} which is not a time-varying "
                "covariate column. The time-varying columns are "
                f"{sorted(present)}. A column that the covariate screen dropped is neither "
                "modelled nor simulated, so it takes no type"
            )
        if len(columns) > 1 and kind != "categorical":
            raise DataError(
                f"covariate_types= gives the one-hot group {list(columns)} the type {kind!r}. A "
                "tuple of columns is one categorical variable, drawn once"
            )
        if len({present[name] for name in columns}) != 1:
            raise DataError(
                f"covariate_types= groups {list(columns)} across nodes. A one-hot group lies "
                "inside one node's block"
            )
        for name in columns:
            if name in seen:
                raise DataError(
                    f"covariate_types= types the column {name!r} twice: in {list(seen[name])} "
                    f"and in {list(columns)}"
                )
            seen[name] = columns
        declared[columns] = kind
    return declared


def resolve_covariates(
    data: LongitudinalData, covariate_types: Mapping[Any, str] | None
) -> tuple[tuple[Variable, ...], ...]:
    """The variables of every node's covariate block, in the node's draw order.

    Only a binary column is inferred: its observed values are a subset of ``{0, 1}``.  Any other
    undeclared column at nodes ``2..T`` is refused before any learner runs.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    covariate_types : mapping or None
        A column name, or a tuple of one-hot column names, mapped to ``"binary"``,
        ``"categorical"`` or ``"normal"``.

    Returns
    -------
    tuple of tuple of Variable
        One tuple per node.

    Raises
    ------
    CapabilityError
        When a column at nodes ``2..T`` is neither binary nor declared.
    DataError
        When a declaration names a column that is not a covariate, or a declared type does
        not hold the observed values.
    """
    declared = _declared_kinds(covariate_types, data.time_varying_names)
    grouped = {name: columns for columns in declared if len(columns) > 1 for name in columns}
    blocks: list[tuple[Variable, ...]] = []
    for node in range(1, data.n_times + 1):
        names = data.time_varying_names[node - 1]
        block = data.time_varying[node - 1]
        rows = data.uncensored_through(node - 1) & data.event_free_through(node - 1)
        index = {name: position for position, name in enumerate(names)}
        variables: list[Variable] = []
        for name in names:
            if name in grouped:
                columns = grouped[name]
                if name != min(columns, key=lambda other: index[other]):
                    continue
                positions = tuple(index[other] for other in columns)
                observed = np.nan_to_num(block[rows][:, list(positions)], nan=0.0)
                if observed.size and not np.all(
                    np.isin(observed, (0.0, 1.0)) & (observed.sum(axis=1, keepdims=True) == 1.0)
                ):
                    raise DataError(
                        f"covariate_types= declares {list(columns)} a one-hot group, but a row "
                        "does not hold exactly one 1 among them"
                    )
                variables.append(
                    Variable(
                        node,
                        len(variables),
                        "+".join(columns),
                        positions,
                        tuple(columns),
                        "categorical",
                        tuple(float(code) for code in range(len(columns))),
                        group=True,
                    )
                )
                continue
            kind = declared.get((name,))
            values = block[rows, index[name]]
            finite = values[np.isfinite(values)]
            distinct = np.unique(finite)
            if kind is None and node >= 2:
                if distinct.size and np.all(np.isin(distinct, (0.0, 1.0))):
                    kind = "binary"
                else:
                    raise CapabilityError(
                        f"the time-varying column {name!r} at node {node} is not binary, and "
                        "covariate_types= does not declare it. Only a column coded 0/1 is "
                        "inferred. Each modelled covariate needs a conditional law to draw "
                        "from.",
                        reason=RefusalReason.INVALID_DECLARATION,
                        alternative=(
                            f"Pass covariate_types={{{name!r}: 'normal'}} for a Gaussian column, "
                            "'categorical' for a column of integer categories, or a tuple of "
                            "one-hot columns mapped to 'categorical'."
                        ),
                    )
            if kind == "binary" and distinct.size and not np.all(np.isin(distinct, (0.0, 1.0))):
                raise DataError(
                    f"covariate_types= declares {name!r} binary, but its observed values are "
                    f"{distinct[:6].tolist()}. A binary column is coded 0/1"
                )
            if kind == "categorical":
                if distinct.size < 2:
                    raise DataError(
                        f"covariate_types= declares {name!r} categorical, but it holds fewer "
                        "than two observed values"
                    )
                variables.append(
                    Variable(
                        node,
                        len(variables),
                        name,
                        (index[name],),
                        (name,),
                        "categorical",
                        tuple(float(value) for value in distinct),
                    )
                )
                continue
            variables.append(
                Variable(
                    node,
                    len(variables),
                    name,
                    (index[name],),
                    (name,),
                    "raw" if node == 1 else str(kind),
                )
            )
        blocks.append(tuple(variables))
    return tuple(blocks)


# ------------------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class Layout:
    """Where each part of a history sits in the design table.

    The table runs ``[W, L_1, A_1, L_2, A_2, ..., L_T, A_T]``.  Every model reads a prefix of
    it, because every model conditions on the history before its own variable: the covariate at
    position ``j`` of node ``t`` reads everything before its slot, the natural treatment reads
    the columns before ``A_t``, a hazard reads through ``A_t`` and the outcome reads all of it.

    Parameters
    ----------
    n_baseline : int
        The number of baseline columns.
    variable_start : dict
        The first table column of each variable, keyed by ``(node, position)``.
    treatment_start : dict
        The first table column of each node's treatment block, keyed by node.  A node of a held
        design after the first has none.
    treatment_stop : dict
        The column after each treatment block.
    node_stop : dict
        The column after everything node ``t`` adds.
    width : int
        The number of table columns.
    """

    n_baseline: int
    variable_start: dict[tuple[int, int], int]
    treatment_start: dict[int, int]
    treatment_stop: dict[int, int]
    node_stop: dict[int, int]
    width: int

    @classmethod
    def build(cls, data: LongitudinalData, variables: Sequence[Sequence[Variable]]) -> Layout:
        """The layout of a panel's design table.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        variables : sequence of sequence of Variable
            The variables of each node.

        Returns
        -------
        Layout
            The column map.
        """
        column = int(data.baseline.shape[1])
        variable_start: dict[tuple[int, int], int] = {}
        treatment_start: dict[int, int] = {}
        treatment_stop: dict[int, int] = {}
        node_stop: dict[int, int] = {}
        for node in range(1, data.n_times + 1):
            for variable in variables[node - 1]:
                variable_start[(node, variable.position)] = column
                column += variable.width
            if not data.is_held_node(node):
                treatment_start[node] = column
                column += (
                    1 if data.is_continuous_node(node) else len(data.treatment_levels[node - 1]) - 1
                )
                treatment_stop[node] = column
            node_stop[node] = column
        return cls(
            int(data.baseline.shape[1]),
            variable_start,
            treatment_start,
            treatment_stop,
            node_stop,
            column,
        )

    def natural_prefix(self, node: int) -> int:
        """The columns before ``A_t``: the history a treatment model at node ``t`` reads."""
        return self.treatment_start.get(node, self.node_stop[node])


def write_variable(
    table: FloatArray, layout: Layout, variable: Variable, payload: Any, rows: Any = slice(None)
) -> None:
    """Write a variable's design columns: indicators of the codes, or the values themselves."""
    start = layout.variable_start[(variable.node, variable.position)]
    if variable.kind == "categorical":
        table[rows, start : start + variable.width] = arm_indicators(
            np.asarray(payload), len(variable.levels)
        )
    else:
        table[rows, start] = np.asarray(payload, dtype=float)


def write_treatment(
    table: FloatArray, layout: Layout, data: LongitudinalData, node: int, payload: Any
) -> None:
    """Write node ``t``'s treatment columns: indicators of the codes, or the dose."""
    if node not in layout.treatment_start:
        return
    start = layout.treatment_start[node]
    values = np.nan_to_num(np.asarray(payload, dtype=float), nan=0.0)
    if data.is_continuous_node(node):
        table[:, start] = values
    else:
        stop = layout.treatment_stop[node]
        table[:, start:stop] = arm_indicators(values, len(data.treatment_levels[node - 1]))


def observed_table(
    data: LongitudinalData, variables: Sequence[Sequence[Variable]], layout: Layout
) -> FloatArray:
    """The design table of the observed histories, filled with zeros where a row has none."""
    table = np.zeros((data.n, layout.width))
    table[:, : layout.n_baseline] = np.nan_to_num(data.baseline, nan=0.0)
    for node in range(1, data.n_times + 1):
        block = data.time_varying[node - 1]
        for variable in variables[node - 1]:
            payload = (
                variable.observed_codes(block)
                if variable.kind == "categorical"
                else variable.observed_values(block)
            )
            write_variable(table, layout, variable, payload)
        write_treatment(table, layout, data, node, data.treatment[:, node - 1])
    return table


# -------------------------------------------------------------------------------------- models


@dataclass
class FittedModel:
    """One fitted conditional model, with the rows and weights it was fitted on.

    Parameters
    ----------
    key : str
        A unique name, such as ``"covariate:2:L2"``.
    label : str
        The name a message uses.
    role : {"covariate", "outcome", "hazard", "treatment"}
        What the model describes.
    response : {"binary", "multi", "normal", "mean"}
        A binary or categorical response gives class probabilities, a Gaussian one a mean and a
        residual variance, and an outcome a mean.
    prefix : int
        The columns of the design table the model reads.
    n_classes : int
        The number of classes of a binary or categorical response.
    learner : object
        The fitted learner, or ``None`` for a response with one observed class.
    present : tuple of int
        The classes the fit rows hold.
    sigma2 : float
        The residual variance of a Gaussian response, with the maximum-likelihood divisor.
    rows : BoolArray
        The rows the model was fitted on.
    weights : FloatArray
        The observation weights on those rows, zero elsewhere.
    target : object
        The observed target as the learner saw it, filled outside the fit rows.
    stack : bool
        Whether the estimate reads the model, so that its block joins the stacked curve.
    task : {"regression", "classification"}
        How a non-GLM outcome learner predicts.
    constant : FloatArray or None
        The class probabilities of a response with one observed class.
    """

    key: str
    label: str
    role: str
    response: str
    prefix: int
    n_classes: int
    learner: Any
    present: tuple[int, ...]
    sigma2: float
    rows: BoolArray
    weights: FloatArray
    target: Any
    stack: bool = True
    task: str = "regression"
    constant: FloatArray | None = None

    @property
    def is_glm(self) -> bool:
        """Whether the fitted learner is a :class:`~cleverly.learners.GLM`."""
        return isinstance(self.learner, GLM)

    @property
    def parametric(self) -> bool:
        """Whether the stacked equations read the model: a GLM, or a constant."""
        return self.constant is not None or self.is_glm

    @property
    def aliased(self) -> bool:
        """Whether the GLM aliased a design column on its fit rows."""
        return self.is_glm and not bool(np.all(self.learner.active_))

    def outside_span(self, x: FloatArray) -> BoolArray:
        """Rows whose design departs from the relation an aliased column held on the fit rows."""
        if not self.aliased:
            return np.zeros(x.shape[0], dtype=bool)
        return np.asarray(self.learner.outside_span(x), dtype=bool)

    def class_probabilities(self, x: FloatArray) -> FloatArray:
        """``(N, K)`` class probabilities, zero for a class the fit rows never held."""
        n = x.shape[0]
        if self.constant is not None:
            return np.tile(self.constant, (n, 1))
        out = np.zeros((n, self.n_classes))
        if self.is_glm:
            glm = self.learner
            if glm.family_ == "binomial":
                positive = glm.predict(x)
                columns = [1.0 - positive, positive]
            else:
                columns = list(glm.predict_proba(x).T)
            for class_index, column in zip(self.present, columns, strict=True):
                out[:, class_index] = column
            return out
        probabilities = predict_probabilities(self.learner, x, [float(c) for c in self.present])
        for column, class_index in enumerate(self.present):
            out[:, class_index] = probabilities[:, column]
        return out

    def mean(self, x: FloatArray) -> FloatArray:
        """``(N,)`` fitted mean of a Gaussian or outcome response."""
        if self.is_glm:
            return np.asarray(self.learner.predict(x), dtype=float)
        return predict_mean(self.learner, x, self.task)  # type: ignore[arg-type]

    @property
    def n_parameters(self) -> int:
        """The length of the model's parameter vector in the stacked equations."""
        if not self.is_glm:
            return 0
        return int(self.learner.n_parameters) + (1 if self.response == "normal" else 0)

    def mean_jacobian(self, x: FloatArray) -> FloatArray:
        """``(N, P)`` derivative of the mean in the parameters (zero in the variance column)."""
        jacobian = np.asarray(self.learner.mean_jacobian(x), dtype=float)
        if self.response == "normal":
            jacobian = np.column_stack([jacobian, np.zeros(jacobian.shape[0])])
        return jacobian

    def class_jacobian(self, x: FloatArray) -> FloatArray:
        """``(N, K, P)`` derivative of every class probability in the parameters."""
        glm = self.learner
        raw = np.asarray(glm.mean_jacobian(x), dtype=float)
        n = x.shape[0]
        out = np.zeros((n, self.n_classes, self.n_parameters))
        if glm.family_ == "binomial":
            out[:, self.present[1], :] = raw
            out[:, self.present[0], :] = -raw
        else:
            for column, class_index in enumerate(self.present):
                out[:, class_index, :] = raw[:, column, :]
        return out

    def draw_score(self, x: FloatArray, drawn: FloatArray, tilt: float | None = None) -> FloatArray:
        """``(N, P)`` derivative of the log density of ``drawn`` in the parameters.

        Parameters
        ----------
        x : FloatArray
            The design table prefix the draw was made from.
        drawn : FloatArray
            The drawn class index, or the drawn Gaussian value.
        tilt : float or None
            The odds multiplier of an odds tilt.  The draw is then from
            ``expit(eta + log(tilt))`` and the score is ``x (a - q)``.

        Returns
        -------
        FloatArray
            One score row per path.
        """
        glm = self.learner
        if self.response == "normal":
            residual = drawn - glm.predict(x)
            beta = np.asarray(glm.score_contributions(x, drawn), dtype=float) / self.sigma2
            variance = (residual * residual - self.sigma2) / (2.0 * self.sigma2**2)
            return np.column_stack([beta, variance])
        if tilt is not None:
            eta = glm.linear_predictor(x) + np.log(tilt)
            tilted = 1.0 / (1.0 + np.exp(-eta))
            return np.asarray(glm.design(x) * (drawn - tilted)[:, None], dtype=float)
        return np.asarray(glm.score_contributions(x, drawn.astype(np.int64)), dtype=float)

    def projection(self, x: FloatArray, n: int) -> FloatArray:
        """``(n, P)`` scores times the inverse information over ``n``: the curve's model term.

        The curve adds ``projection @ D`` for the model's derivative ``D``.  The scores sum to
        zero over the fit rows at the maximum likelihood estimate, so the term has mean zero.
        """
        glm = self.learner
        information = np.asarray(glm.information(x, self.weights), dtype=float) / n
        scores = np.asarray(glm.score_contributions(x, self.target, self.weights), dtype=float)
        projected = _solve(information, scores.T).T
        if self.response != "normal":
            return np.asarray(projected, dtype=float)
        residual = np.asarray(self.target, dtype=float) - glm.predict(x)
        variance = self.weights * (residual * residual - self.sigma2)
        mass = float(self.weights.sum()) / n
        return np.column_stack([projected, variance / mass])


def _instance(template: Learner | None, family: str) -> Learner:
    """A fresh learner for a response of ``family``: the template's GLM settings, or a copy."""
    if template is None:
        return GLM(family=family)  # type: ignore[arg-type]
    if isinstance(template, GLM):
        return GLM(
            family=family,  # type: ignore[arg-type]
            interactions=template.interactions,
            max_iter=template.max_iter,
            tol=template.tol,
        )
    return clone_learner(template)


def _fit_response(
    *,
    key: str,
    label: str,
    role: str,
    response: str,
    template: Learner | None,
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray,
    rows: BoolArray,
    n_classes: int = 0,
    task: str = "regression",
    stack: bool = True,
    outcome_family: str = "gaussian",
) -> FittedModel:
    """Fit one conditional model on ``rows`` and keep what the curve and the draws read."""
    use = rows & (weights > 0.0) & np.isfinite(y)
    if not np.any(use):
        raise DataError(f"{label}: no fit row has a positive weight and an observed value")
    fit_weights = np.where(use, weights, 0.0)
    xs, ys, ws = x[use], y[use], weights[use]
    prefix = int(x.shape[1])
    common: dict[str, Any] = {
        "key": key,
        "label": label,
        "role": role,
        "response": response,
        "prefix": prefix,
        "n_classes": n_classes,
        "rows": use,
        "weights": fit_weights,
        "stack": stack,
        "task": task,
    }
    try:
        if response in ("binary", "multi"):
            labels = cast(FloatArray, ys.astype(np.int64))
            present = tuple(int(value) for value in np.unique(labels))
            target = np.where(use, y, 0.0).astype(np.int64)
            if len(present) == 1:
                constant = np.zeros(n_classes)
                constant[present[0]] = 1.0
                return FittedModel(
                    **common,
                    learner=None,
                    present=present,
                    sigma2=0.0,
                    target=target,
                    constant=constant,
                )
            family = "binomial" if len(present) == 2 else "multinomial"
            learner = _instance(template, family)
            if isinstance(learner, GLM):
                learner.fit(xs, labels, ws)
            else:
                learner = fit_learner(learner, xs, labels, ws, copy=False)
            return FittedModel(
                **common, learner=learner, present=present, sigma2=0.0, target=target
            )
        if response == "normal":
            learner = _instance(template, "gaussian")
            assert isinstance(learner, GLM)
            learner.fit(xs, ys.astype(float), ws)
            residual = ys - learner.predict(xs)
            sigma2 = float(np.sum(ws * residual * residual) / np.sum(ws))
            if not sigma2 > 0.0:
                raise DataError("the residual variance is zero, so the Gaussian law is degenerate")
            return FittedModel(
                **common,
                learner=learner,
                present=(),
                sigma2=sigma2,
                target=np.where(use, y, 0.0),
            )
        learner = _instance(
            template, template.family if isinstance(template, GLM) else outcome_family
        )
        values = np.where(use, y, 0.0)
        if task == "classification":
            values = as_target(values, "classification")
        if isinstance(learner, GLM):
            learner.fit(xs, values[use], ws)
        else:
            learner = fit_learner(learner, xs, values[use], ws, copy=False)
        return FittedModel(**common, learner=learner, present=(), sigma2=0.0, target=values)
    except DataError as error:
        raise DataError(f"{label}: {error}") from error


@dataclass(frozen=True)
class GFormulaSettings:
    """What the forward simulation needs beyond the data and the regimens.

    Parameters
    ----------
    n_draws : int
        The copies of each unit behind the estimate.
    n_derivative_draws : int
        The copies of each unit behind the parameter derivatives, at least two.
    covariate_types : mapping or None
        The declared covariate types.
    covariate_learner : object or None
        The template of every covariate model.  ``None`` is a main-terms GLM.
    outcome_learner : object or None
        The outcome model.
    hazard_learner : object or None
        The template of every hazard model.
    treatment_learner : object or None
        The template of every natural-treatment model.
    seed : int
        The simulation seed.
    exact : callable or None
        A test hook.  It is called with the :class:`ModelBook`, the plan and the cells, and
        returns the exact :math:`\\Phi_i` and derivatives (:class:`ExactTerms`) in place of the
        simulation.
    """

    n_draws: int
    n_derivative_draws: int
    covariate_types: Mapping[Any, str] | None
    covariate_learner: Learner | None
    outcome_learner: Learner | None
    hazard_learner: Learner | None
    treatment_learner: Learner | None
    seed: int
    exact: Callable[..., ExactTerms] | None = None


@dataclass(frozen=True)
class ExactTerms:
    """Exact simulation terms for one plan, from a test's enumeration.

    Parameters
    ----------
    phi : FloatArray
        ``(n, C)`` exact regimen means per unit and cell.
    derivatives : dict of str to FloatArray
        By model key, the ``(C, P)`` derivative of the estimate in the model's parameters.
    """

    phi: FloatArray
    derivatives: dict[str, FloatArray]


@dataclass(frozen=True)
class Cell:
    """One reported parameter of a plan: a cause at a horizon.

    Parameters
    ----------
    cause : str or None
        The absorbing cause, or ``None``.
    horizon : int
        The node the parameter is indexed by.
    cause_index : int
        The zero-based cause column, zero when there is one event.
    """

    cause: str | None
    horizon: int
    cause_index: int


def plan_node(regimen: Regimen | DynamicRegimen, time: int) -> tuple[str, Any]:
    """The kind and the declaration of a regimen's node.

    Parameters
    ----------
    regimen : Regimen or DynamicRegimen
        The regimen.
    time : int
        The node, counted from one.

    Returns
    -------
    tuple
        One of ``"label"``, ``"rule"``, ``"policy"``, ``"mtp"``, ``"tilt"`` and the node.
    """
    if isinstance(regimen, Regimen):
        return "label", regimen.at(time)
    node = regimen.plan[time - 1]
    if isinstance(node, Incremental):
        return "tilt", node
    if isinstance(node, POLICY_TYPES):
        return "mtp", node
    if isinstance(node, Stochastic):
        return "policy", node
    if callable(node):
        return "rule", node
    return "label", node


@dataclass
class ModelBook:
    """Every model of a fit, with the observed design table they were fitted on.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    variables : tuple of tuple of Variable
        The variables of each node.
    layout : Layout
        Where each part of a history sits in the design table.
    table : FloatArray
        The ``(n, width)`` design table of the observed histories.
    scaler : OutcomeScaler
        The outcome transformation.
    covariate : dict
        The covariate models, keyed by ``(node, position)``.
    outcome : FittedModel or None
        The outcome model of an end-of-study outcome.
    hazard : dict
        The hazard model of each node of a survival outcome.
    treatment : dict
        The natural-treatment model of each node that needs one.
    """

    data: LongitudinalData
    variables: tuple[tuple[Variable, ...], ...]
    layout: Layout
    table: FloatArray
    scaler: OutcomeScaler
    covariate: dict[tuple[int, int], FittedModel]
    outcome: FittedModel | None
    hazard: dict[int, FittedModel]
    treatment: dict[int, FittedModel]
    _projection: dict[str, FloatArray] = field(default_factory=dict)

    @property
    def models(self) -> list[FittedModel]:
        """Every model in stack order: covariates, treatment, outcome, hazards."""
        out = [self.covariate[key] for key in sorted(self.covariate)]
        out += [self.treatment[node] for node in sorted(self.treatment)]
        if self.outcome is not None:
            out.append(self.outcome)
        out += [self.hazard[node] for node in sorted(self.hazard)]
        return out

    @property
    def stacked(self) -> list[FittedModel]:
        """The models whose block joins the stacked curve: read by the estimate and a GLM."""
        return [model for model in self.models if model.stack and model.is_glm]

    @property
    def parametric(self) -> bool:
        """Whether every model the estimate reads is a GLM or a constant."""
        return all(model.parametric for model in self.models if model.stack)

    def projection(self, model: FittedModel) -> FloatArray:
        """The cached ``(n, P)`` curve term of a stacked model."""
        if model.key not in self._projection:
            x = self.table[:, : model.prefix]
            self._projection[model.key] = model.projection(x, self.data.n)
        return self._projection[model.key]

    def design(self, model: FittedModel) -> FloatArray:
        """The observed design a model was fitted on."""
        return self.table[:, : model.prefix]


def fit_models(
    data: LongitudinalData,
    plans: Sequence[Plan],
    settings: GFormulaSettings,
    scaler: OutcomeScaler,
    variables: tuple[tuple[Variable, ...], ...],
) -> ModelBook:
    """Fit every conditional model the plans read.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans, which decide the natural-treatment models.
    settings : GFormulaSettings
        The learners.
    scaler : OutcomeScaler
        The outcome transformation.
    variables : tuple of tuple of Variable
        The typed covariates.

    Returns
    -------
    ModelBook
        The fitted models.
    """
    layout = Layout.build(data, variables)
    table = observed_table(data, variables, layout)
    weights = np.asarray(data.weights, dtype=float)
    covariate: dict[tuple[int, int], FittedModel] = {}
    for node in range(2, data.n_times + 1):
        rows = data.uncensored_through(node - 1) & data.event_free_through(node - 1)
        block = data.time_varying[node - 1]
        for variable in variables[node - 1]:
            if not variable.modelled:
                continue
            start = layout.variable_start[(node, variable.position)]
            if variable.kind == "categorical":
                response, y, n_classes = (
                    "multi",
                    variable.observed_codes(block).astype(float),
                    len(variable.levels),
                )
            elif variable.kind == "binary":
                response, y, n_classes = "binary", variable.observed_values(block), 2
            else:
                response, y, n_classes = "normal", variable.observed_values(block), 0
            if response == "normal" and not (
                settings.covariate_learner is None or isinstance(settings.covariate_learner, GLM)
            ):
                raise normal_learner_refusal("covariate_learner", variable.name)
            covariate[(node, variable.position)] = _fit_response(
                key=f"covariate:{node}:{variable.name}",
                label=f"the model of {variable.name!r} at node {node}",
                role="covariate",
                response=response,
                template=settings.covariate_learner,
                x=table[:, :start],
                y=np.where(rows, y, np.nan),
                weights=weights,
                rows=rows,
                n_classes=n_classes,
            )
    hazard: dict[int, FittedModel] = {}
    outcome: FittedModel | None = None
    if data.is_survival:
        n_causes = max(len(data.cause_labels), 1)
        for node in range(1, data.n_times + 1):
            rows = data.uncensored_through(node) & data.event_free_through(node - 1)
            if data.cause_event is None:
                codes = data.event[:, node - 1].astype(float)  # type: ignore[index]
            else:
                happened = data.cause_event[:, node - 1, :]
                codes = np.where(happened.any(axis=1), happened.argmax(axis=1) + 1.0, 0.0)
            hazard[node] = _fit_response(
                key=f"hazard:{node}",
                label=f"the hazard model at node {node}",
                role="hazard",
                response="multi",
                template=settings.hazard_learner,
                x=table[:, : layout.node_stop[node]],
                y=np.where(rows, codes, np.nan),
                weights=weights,
                rows=rows,
                n_classes=n_causes + 1,
            )
    else:
        rows = data.uncensored_through(data.n_times)
        scaled = scaler.scale(np.nan_to_num(data.outcome, nan=0.0))
        task = "classification" if data.family == "binomial" else "regression"
        outcome = _fit_response(
            key="outcome",
            label=f"the outcome model of {data.outcome_name!r}",
            role="outcome",
            response="mean",
            template=settings.outcome_learner,
            x=table,
            y=np.where(rows & np.isfinite(data.outcome), scaled, np.nan),
            weights=weights,
            rows=rows,
            task=task,
            outcome_family="binomial" if data.family == "binomial" else "gaussian",
        )
    treatment: dict[int, FittedModel] = {}
    needs: dict[int, bool] = {}
    for plan in plans:
        for node in range(1, data.n_times + 1):
            if data.is_held_node(node):
                continue
            kind, _ = plan_node(plan.regimen, node)
            if kind == "tilt" or (kind == "mtp" and node >= 2):
                needs[node] = True
            elif kind == "label" and data.is_continuous_node(node):
                needs.setdefault(node, False)
    for node, stacked in sorted(needs.items()):
        rows = (
            data.uncensored_through(node - 1)
            & data.event_free_through(node - 1)
            & np.isfinite(data.treatment[:, node - 1])
        )
        observed = data.treatment[:, node - 1]
        continuous = data.is_continuous_node(node)
        n_levels = 0 if continuous else len(data.treatment_levels[node - 1])
        template = settings.treatment_learner if stacked else None
        if continuous and stacked and not (template is None or isinstance(template, GLM)):
            raise normal_learner_refusal("treatment_learner", data.treatment_names[node - 1])
        treatment[node] = _fit_response(
            key=f"treatment:{node}",
            label=f"the natural-treatment model of {data.treatment_names[node - 1]!r}",
            role="treatment",
            response="normal" if continuous else ("binary" if n_levels == 2 else "multi"),
            template=template,
            x=table[:, : layout.natural_prefix(node)],
            y=np.where(rows, observed, np.nan),
            weights=weights,
            rows=rows,
            n_classes=n_levels,
            stack=stacked,
        )
    return ModelBook(data, variables, layout, table, scaler, covariate, outcome, hazard, treatment)


def normal_learner_refusal(argument: str, name: str) -> CapabilityError:
    return CapabilityError(
        f"{argument}= is not a GLM, and {name!r} is Gaussian. The forward draw of a Gaussian "
        "column needs the residual variance of the fit, and a learner other than a GLM gives "
        "only an in-sample residual that understates it.",
        reason=RefusalReason.INVALID_DECLARATION,
        alternative=f"Leave {argument}= at its GLM default, or pass a GLM.",
    )


# ----------------------------------------------------------------------------------- simulation


@dataclass
class _Draw:
    """One drawn quantity of a chunk: its model, its value, and an odds tilt's multiplier."""

    model: FittedModel
    value: FloatArray
    tilt: float | None = None


@dataclass
class _Paths:
    """The simulated histories of a chunk of units under one plan."""

    units: IntArray
    copies: IntArray
    table: FloatArray
    values: dict[str, FloatArray]
    labels: dict[int, Any] = field(default_factory=dict)
    draws: list[_Draw] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.units.size)


def _categorical(probabilities: FloatArray, uniforms: FloatArray) -> FloatArray:
    """Inverse-CDF draw of a class index: the first class whose cumulative mass exceeds ``u``."""
    cumulative = np.cumsum(probabilities, axis=1)
    index = (uniforms[:, None] >= cumulative).sum(axis=1)
    return np.asarray(np.minimum(index, probabilities.shape[1] - 1), dtype=float)


@dataclass(frozen=True)
class SupportRow:
    """The support of a fixed dose at one continuous node.

    Parameters
    ----------
    plan : str
        The regimen label.
    node : int
        The node, counted from one.
    name : str
        The treatment column.
    dose : float
        The fixed dose.
    off_support : float
        The weighted share of simulated histories whose fitted mean dose lies more than
        :data:`SUPPORT_Z` fitted standard deviations from the dose.
    window : float
        The weighted share of observed doses within :data:`SUPPORT_Z` fitted standard deviations
        of the dose.
    """

    plan: str
    node: int
    name: str
    dose: float
    off_support: float
    window: float


class _Simulator:
    """Forward simulation of one plan from the fitted models, on one stream."""

    def __init__(
        self,
        book: ModelBook,
        plan: Plan,
        seed: int,
        stream: int,
        *,
        check_span: bool = False,
    ) -> None:
        self.book = book
        self.data = book.data
        self.layout = book.layout
        self.plan = plan
        self.regimen = plan.regimen
        self.seed = seed
        self.stream = stream
        self.check_span = check_span
        self.leaving: dict[str, int] = {}
        self.off_support: dict[int, float] = {}
        data = self.data
        self._observed: dict[tuple[int, int], FloatArray] = {}
        for node in range(1, data.n_times + 1):
            block = data.time_varying[node - 1]
            for variable in book.variables[node - 1]:
                if node == 1 or not variable.modelled:
                    self._observed[(node, variable.position)] = (
                        variable.observed_codes(block).astype(float)
                        if variable.kind == "categorical"
                        else variable.observed_values(block)
                    )
        self._levels = [
            self._level_array(node) if not data.is_continuous_node(node) else None
            for node in range(1, data.n_times + 1)
        ]

    def _level_array(self, node: int) -> Any:
        levels = self.data.treatment_levels[node - 1]
        array = np.empty(len(levels), dtype=object)
        for index, level in enumerate(levels):
            array[index] = level
        return array

    # -------------------------------------------------------------------------- one chunk

    def simulate(self, units: IntArray, copies: int, path_weight: FloatArray) -> _Paths:
        """Simulate ``copies`` histories of every unit in ``units``."""
        data, layout, book = self.data, self.layout, self.book
        rows = np.repeat(units, copies)
        copy_index = np.tile(np.arange(copies), units.size)
        table = np.zeros((rows.size, layout.width))
        table[:, : layout.n_baseline] = book.table[rows, : layout.n_baseline]
        baseline = np.nan_to_num(data.baseline[rows], nan=0.0)
        values: dict[str, FloatArray] = {
            name: baseline[:, index] for index, name in enumerate(data.baseline_names)
        }
        paths = _Paths(rows, copy_index, table, values)
        for node in range(1, data.n_times + 1):
            for variable in book.variables[node - 1]:
                start = layout.variable_start[(node, variable.position)]
                if (node, variable.position) in self._observed:
                    payload = self._observed[(node, variable.position)][rows]
                else:
                    payload = self._draw_covariate(paths, node, variable, start)
                write_variable(table, layout, variable, payload)
                values.update(variable.frame_values(payload))
            self._set_treatment(paths, node, path_weight)
        return paths

    def _uniforms(self, paths: _Paths, node: int, column: int) -> FloatArray:
        return counter_uniforms(self.seed, self.stream, paths.units, paths.copies, node, column)

    def _span(self, model: FittedModel, x: FloatArray) -> None:
        """Count the simulated rows that leave the span an aliased column held at the fit."""
        if not self.check_span or not model.aliased:
            return
        count = int(np.count_nonzero(model.outside_span(x)))
        if count:
            self.leaving[model.key] = self.leaving.get(model.key, 0) + count

    def _draw_covariate(
        self, paths: _Paths, node: int, variable: Variable, start: int
    ) -> FloatArray:
        model = self.book.covariate[(node, variable.position)]
        x = paths.table[:, :start]
        self._span(model, x)
        uniforms = self._uniforms(paths, node, variable.position)
        if model.response == "binary":
            drawn = (uniforms < model.class_probabilities(x)[:, 1]).astype(float)
        elif model.response == "multi":
            drawn = _categorical(model.class_probabilities(x), uniforms)
        else:
            drawn = model.mean(x) + np.sqrt(model.sigma2) * ndtri(uniforms)
        paths.draws.append(_Draw(model, drawn))
        return drawn

    # ------------------------------------------------------------------------- treatment

    def _history_frame(self, paths: _Paths, node: int) -> Any:
        names = self.data.history_names(node)
        return self.data.frame_like({name: paths.values[name] for name in names})

    def _policy_frame(self, paths: _Paths, node: int) -> Any:
        data = self.data
        payload = {name: paths.values[name] for name in data.history_names(node)}
        for earlier in range(1, min(node, data.n_decisions + 1)):
            payload[data.treatment_names[earlier - 1]] = paths.labels[earlier]
        return data.frame_like(payload)

    def _code_of(self, node: int, label: object) -> int:
        levels = self.data.treatment_levels[node - 1]
        for code, level in enumerate(levels):
            if bool(level == label):
                return code
        raise DataError(
            f"regimen {self.plan.label!r} assigns {label!r} at time {node}; treatment column "
            f"{self.data.decision_name(node)!r} has levels {list(levels)!r}"
        )

    def _codes_of(self, node: int, arms: Any) -> FloatArray:
        levels = self.data.treatment_levels[node - 1]
        raw = np.asarray(arms, dtype=object)
        codes = np.zeros(raw.size)
        matched = np.zeros(raw.size, dtype=bool)
        for code, level in enumerate(levels):
            if isinstance(level, tuple):
                here = np.fromiter((value == level for value in raw), bool, raw.size)
            else:
                here = np.asarray(raw == level, dtype=bool)
            codes[here & ~matched] = float(code)
            matched |= here
        if not np.all(matched):
            value = raw[int(np.flatnonzero(~matched)[0])]
            raise DataError(
                f"regimen {self.plan.label!r} assigns {value!r} at time {node} on a simulated "
                f"history; treatment column {self.data.decision_name(node)!r} has levels "
                f"{list(levels)!r}"
            )
        return codes

    def _set_treatment(self, paths: _Paths, node: int, path_weight: FloatArray) -> None:
        data, layout = self.data, self.layout
        if data.is_held_node(node):
            paths.labels[node] = paths.labels[1]
            return
        kind, declared = plan_node(self.regimen, node)
        continuous = data.is_continuous_node(node)
        n = paths.n
        if kind == "label":
            if continuous:
                value = np.full(n, float(declared))
                self._record_support(paths, node, float(declared), path_weight)
            else:
                value = np.full(n, float(self._code_of(node, declared)))
        elif kind == "rule":
            returned = declared(self._history_frame(paths, node))
            arms = np.asarray(_as_array(returned), dtype=object)
            if arms.ndim != 1 or arms.shape[0] != n:
                raise DataError(
                    f"the rule at time {node} of regimen {self.plan.label!r} returned "
                    f"{arms.shape} assignments for {n} simulated histories; it must return one "
                    "arm per row"
                )
            value = self._codes_of(node, arms)
        elif kind == "policy":
            value = _categorical(
                self._policy_density(paths, node, declared),
                self._uniforms(paths, node, _COL_TREATMENT),
            )
        elif kind == "tilt":
            value = self._tilted(paths, node, float(declared.delta))
        else:
            value = self._modified(paths, node, declared, continuous)
        write_treatment(paths.table, layout, data, node, value)
        if continuous:
            paths.labels[node] = value
        else:
            levels = self._levels[node - 1]
            assert levels is not None
            paths.labels[node] = levels[value.astype(np.int64)]

    def _policy_density(self, paths: _Paths, node: int, declared: Stochastic) -> FloatArray:
        data = self.data
        label = f"the policy at time {node} of regimen {self.plan.label!r}"
        levels = tuple(data.treatment_levels[node - 1])
        returned = declared.density_fn(self._policy_frame(paths, node))
        if hasattr(returned, "columns") and not isinstance(returned, np.ndarray):
            returned = _frame_by_level(
                returned, levels, label=label, name=data.treatment_names[node - 1]
            )
        density = np.asarray(_as_array(returned), dtype=float)
        if density.shape != (paths.n, len(levels)):
            raise DataError(
                f"{label} returned shape {density.shape}; expected {(paths.n, len(levels))} "
                "on the simulated histories"
            )
        check_regime_density(density, label=label)
        return density

    def _tilted(self, paths: _Paths, node: int, delta: float) -> FloatArray:
        model = self.book.treatment[node]
        x = paths.table[:, : model.prefix]
        self._span(model, x)
        natural = model.class_probabilities(x)[:, 1]
        tilted = delta * natural / (delta * natural + 1.0 - natural)
        drawn = (self._uniforms(paths, node, _COL_TREATMENT) < tilted).astype(float)
        paths.draws.append(_Draw(model, drawn, tilt=delta))
        return drawn

    def _modified(self, paths: _Paths, node: int, declared: Any, continuous: bool) -> FloatArray:
        data = self.data
        n = paths.n
        if node == 1:
            natural = np.nan_to_num(data.treatment[paths.units, 0], nan=0.0)
        else:
            model = self.book.treatment[node]
            x = paths.table[:, : model.prefix]
            self._span(model, x)
            uniforms = self._uniforms(paths, node, _COL_NATURAL)
            if continuous:
                natural = model.mean(x) + np.sqrt(model.sigma2) * ndtri(uniforms)
            else:
                natural = _categorical(model.class_probabilities(x), uniforms)
            paths.draws.append(_Draw(model, natural))
        frame = lazy_frame(lambda: self._policy_frame(paths, node))
        branches = policy_branches(declared)
        probabilities = np.array([probability for probability, _ in branches])
        if len(branches) > 1:
            branch = _categorical(
                np.tile(probabilities, (n, 1)), self._uniforms(paths, node, _COL_BRANCH)
            ).astype(np.int64)
        else:
            branch = np.zeros(n, dtype=np.int64)
        value = np.zeros(n)
        if continuous:
            for index, (_, member) in enumerate(branches):
                assigned = np.asarray(member.assign(natural, frame), dtype=float).reshape(-1)
                value = np.where(branch == index, assigned, value)
            return value
        where = f"node {data.treatment_names[node - 1]!r} at time {node}"
        maps = discrete_assignments(
            declared, data.treatment_levels[node - 1], frame, np.ones(n, dtype=bool), where=where
        )
        rows = np.arange(n)
        for index, (_, codes) in enumerate(maps):
            value = np.where(branch == index, codes[rows, natural.astype(np.int64)], value)
        return value

    def _record_support(
        self, paths: _Paths, node: int, dose: float, path_weight: FloatArray
    ) -> None:
        """Accumulate the weighted share of simulated histories off the fixed dose's support."""
        model = self.book.treatment.get(node)
        if model is None or not self.check_span:
            return
        mean = model.mean(paths.table[:, : model.prefix])
        off = np.abs(dose - mean) / np.sqrt(model.sigma2) > SUPPORT_Z
        self.off_support[node] = self.off_support.get(node, 0.0) + float(path_weight[off].sum())


# ----------------------------------------------------------------------------- the functional


def _cells(data: LongitudinalData, horizons: Sequence[int]) -> list[Cell]:
    causes: tuple[str | None, ...] = data.cause_labels or (None,)
    return [
        Cell(cause, int(horizon), index)
        for index, cause in enumerate(causes)
        for horizon in horizons
    ]


class _Functional:
    """The outcome functional of a simulated history, and its pathwise derivatives."""

    def __init__(self, book: ModelBook, cells: Sequence[Cell]) -> None:
        self.book = book
        self.cells = list(cells)

    def zero_derivatives(self) -> dict[str, FloatArray]:
        """Zeroed ``(C, P)`` accumulators for every stacked GLM block."""
        return {
            model.key: np.zeros((len(self.cells), model.n_parameters))
            for model in self.book.stacked
        }

    def evaluate(
        self,
        table: FloatArray,
        *,
        path_weight: FloatArray | None = None,
        derivatives: dict[str, FloatArray] | None = None,
        span: Callable[[FittedModel, FloatArray], None] | None = None,
    ) -> FloatArray:
        """``(N, C)`` functional values, adding pathwise derivatives into ``derivatives``."""
        book = self.book
        if book.outcome is not None:
            model = book.outcome
            x = table[:, : model.prefix]
            if span is not None:
                span(model, x)
            if derivatives is not None and model.stack and model.is_glm:
                assert path_weight is not None
                derivatives[model.key] += (path_weight @ model.mean_jacobian(x))[None, :]
            return model.mean(x).reshape(-1, 1)
        return self._survival(table, path_weight, derivatives, span)

    def _survival(
        self,
        table: FloatArray,
        path_weight: FloatArray | None,
        derivatives: dict[str, FloatArray] | None,
        span: Callable[[FittedModel, FloatArray], None] | None,
    ) -> FloatArray:
        book = self.book
        data = book.data
        n_times = data.n_times
        n_causes = max(len(data.cause_labels), 1)
        hazards = np.zeros((table.shape[0], n_times, n_causes))
        for node in range(1, n_times + 1):
            model = book.hazard[node]
            x = table[:, : model.prefix]
            if span is not None:
                span(model, x)
            hazards[:, node - 1, :] = model.class_probabilities(x)[:, 1:]
        total = hazards.sum(axis=2)
        survival = np.ones((table.shape[0], n_times + 1))
        survival[:, 1:] = np.cumprod(1.0 - total, axis=1)
        incidence = np.cumsum(survival[:, :n_times, None] * hazards, axis=1)
        values = np.column_stack(
            [incidence[:, cell.horizon - 1, cell.cause_index] for cell in self.cells]
        )
        if derivatives is None:
            return np.asarray(values, dtype=float)
        assert path_weight is not None
        for node in range(1, n_times + 1):
            model = book.hazard[node]
            if not (model.stack and model.is_glm):
                continue
            jacobian = model.class_jacobian(table[:, : model.prefix])
            tails = {node: np.zeros((table.shape[0], n_causes))}
            exclusive = survival[:, node - 1]
            running = tails[node]
            for later in range(node + 1, n_times + 1):
                running = running + exclusive[:, None] * hazards[:, later - 1, :]
                tails[later] = running
                exclusive = exclusive * (1.0 - total[:, later - 1])
            for index, cell in enumerate(self.cells):
                if cell.horizon < node:
                    continue
                term = (
                    survival[:, node - 1][:, None] * jacobian[:, cell.cause_index + 1, :]
                    + tails[cell.horizon][:, cell.cause_index][:, None] * jacobian[:, 0, :]
                )
                derivatives[model.key][index] += path_weight @ term
        return np.asarray(values, dtype=float)


# --------------------------------------------------------------------------------- plan result


@dataclass
class _PlanResult:
    """One plan's simulation: the unit means, the derivatives and the stream-1 copies."""

    phi: FloatArray
    derivatives: dict[str, FloatArray]
    copy_values: FloatArray
    support: list[SupportRow]
    exact: bool = False


def _chunk_units(n: int, copies: int, width: int) -> int:
    """Units per chunk so a chunk's tables stay near :data:`_CHUNK_ELEMENTS`."""
    return max(1, min(n, _CHUNK_ELEMENTS // max(1, copies * (width + 60))))


def _refuse_aliased(plan: Plan, book: ModelBook, simulator: _Simulator) -> None:
    """Refuse a regimen whose simulated histories leave the span of an aliased GLM column."""
    if not simulator.leaving:
        return
    by_key = {model.key: model for model in book.models}
    key = next(iter(simulator.leaving))
    raise DataError(
        f"regimen {plan.label!r} is not identified by the parametric g-formula at "
        f"{by_key[key].label}: the GLM's fit rows hold a covariate column that is constant or a "
        "combination of other columns, so it has no coefficient, and "
        f"{simulator.leaving[key]} simulated row(s) depart from that relation. No unit with "
        "those histories follows the regimen, so its mean would assume the column has no "
        "effect. Remove the regimen, or remove the column from the design"
    )


def _simulate_plan(
    book: ModelBook,
    plan: Plan,
    cells: Sequence[Cell],
    settings: GFormulaSettings,
) -> _PlanResult:
    """Simulate one plan on both streams, or take the exact terms a test supplies."""
    data = book.data
    n, n_cells = data.n, len(cells)
    weights = np.asarray(data.weights, dtype=float)
    functional = _Functional(book, cells)
    if settings.exact is not None:
        terms = settings.exact(book, plan, list(cells))
        return _PlanResult(
            np.asarray(terms.phi, dtype=float),
            {key: np.asarray(value, dtype=float) for key, value in terms.derivatives.items()},
            np.zeros((n, settings.n_derivative_draws, n_cells)),
            [],
            exact=True,
        )
    # Stream 0: the copies behind the unit means.
    m = settings.n_draws
    estimate = _Simulator(book, plan, settings.seed, STREAM_ESTIMATE, check_span=True)
    phi = np.zeros((n, n_cells))
    step = _chunk_units(n, m, book.layout.width)
    for first in range(0, n, step):
        units = np.arange(first, min(first + step, n), dtype=np.int64)
        path_weight = np.repeat(weights[units], m) / (n * m)
        paths = estimate.simulate(units, m, path_weight)
        values = functional.evaluate(paths.table, span=estimate._span)
        phi[units] = values.reshape(units.size, m, n_cells).mean(axis=1)
    _refuse_aliased(plan, book, estimate)
    # Stream 1: the copies behind the derivatives, with the leave-one-copy-out baseline.
    m_d = settings.n_derivative_draws
    derivative = _Simulator(book, plan, settings.seed, STREAM_DERIVATIVE)
    derivatives = functional.zero_derivatives()
    copy_values = np.zeros((n, m_d, n_cells))
    step = _chunk_units(n, m_d, book.layout.width)
    for first in range(0, n, step):
        units = np.arange(first, min(first + step, n), dtype=np.int64)
        path_weight = np.repeat(weights[units], m_d) / (n * m_d)
        paths = derivative.simulate(units, m_d, path_weight)
        values = functional.evaluate(paths.table, path_weight=path_weight, derivatives=derivatives)
        per_unit = values.reshape(units.size, m_d, n_cells)
        copy_values[units] = per_unit
        totals = np.repeat(per_unit.sum(axis=1), m_d, axis=0)
        adjusted = (m_d * values - totals) / (m_d - 1)
        for draw in paths.draws:
            model = draw.model
            if not (model.stack and model.is_glm):
                continue
            score = model.draw_score(paths.table[:, : model.prefix], draw.value, draw.tilt)
            derivatives[model.key] += np.einsum("nc,np->cp", adjusted * path_weight[:, None], score)
    support = _support_rows(book, plan, estimate)
    return _PlanResult(phi, derivatives, copy_values, support)


def _support_rows(book: ModelBook, plan: Plan, simulator: _Simulator) -> list[SupportRow]:
    """The support diagnostic of each fixed dose of a plan."""
    data = book.data
    rows: list[SupportRow] = []
    for node, off_support in sorted(simulator.off_support.items()):
        model = book.treatment[node]
        _, declared = plan_node(plan.regimen, node)
        dose = float(declared)
        observed = data.treatment[:, node - 1]
        fit = model.rows & np.isfinite(observed)
        near = np.abs(np.where(fit, observed, 0.0) - dose) <= SUPPORT_Z * np.sqrt(model.sigma2)
        weights = np.where(fit, data.weights, 0.0)
        window = float(np.sum(weights * near) / np.sum(weights))
        rows.append(
            SupportRow(
                plan.label,
                node,
                data.treatment_names[node - 1],
                dose,
                float(off_support),
                window,
            )
        )
    return rows


# ------------------------------------------------------------------------------------ report


@dataclass(frozen=True)
class ModelRecord:
    """One fitted model, as the report lists it.

    Parameters
    ----------
    key : str
        The model's unique name.
    label : str
        The name a message uses.
    role : str
        ``"covariate"``, ``"outcome"``, ``"hazard"`` or ``"treatment"``.
    response : str
        ``"binary"``, ``"multi"``, ``"normal"`` or ``"mean"``.
    learner : str
        The learner's class name, or ``"constant"`` for a response with one observed class.
    stacked : bool
        Whether the estimate reads the model.
    parametric : bool
        Whether the stacked equations read the model.
    n_rows : int
        The rows the model was fitted on.
    n_parameters : int
        The length of its parameter vector in the stacked equations.
    """

    key: str
    label: str
    role: str
    response: str
    learner: str
    stacked: bool
    parametric: bool
    n_rows: int
    n_parameters: int


@dataclass(frozen=True)
class GFormulaReport:
    """What a parametric g-formula fit simulated, and how much of the interval it is.

    Parameters
    ----------
    n_draws : int
        The simulated histories per unit behind each estimate.
    n_derivative_draws : int
        The simulated histories per unit behind each derivative.
    seed : int
        The simulation seed.
    covariate_types : tuple of tuple
        ``(node, variable, type)`` of every modelled covariate.
    orders : tuple of tuple of str
        The draw order of the variables of each node.
    models : tuple of ModelRecord
        Every fitted model.
    parametric : bool
        Whether every model the estimate reads is a GLM.
    model_reliance : str or None
        The statement that a fixed continuous dose makes, ``None`` without one.
    support : tuple of SupportRow
        The support of each fixed dose.
    monte_carlo_share : dict of str to float
        By parameter name, the share of the curve's variance that the simulation contributes.
    """

    n_draws: int
    n_derivative_draws: int
    seed: int
    covariate_types: tuple[tuple[int, str, str], ...]
    orders: tuple[tuple[str, ...], ...]
    models: tuple[ModelRecord, ...]
    parametric: bool
    model_reliance: str | None
    support: tuple[SupportRow, ...]
    monte_carlo_share: dict[str, float] = field(default_factory=dict)

    @property
    def max_share(self) -> float:
        """The largest Monte Carlo share over the reported parameters."""
        shares = [value for value in self.monte_carlo_share.values() if np.isfinite(value)]
        return max(shares) if shares else float("nan")

    def describe(self) -> list[str]:
        """The settings lines of ``LongitudinalResult.summary()``.

        Returns
        -------
        list of str
            One line per setting.
        """
        lines = [
            f"simulation: {self.n_draws} simulated histories per unit for the estimate and "
            f"{self.n_derivative_draws} for the derivatives (Philox streams 0 and 1, seed "
            f"{self.seed}); the interval includes the simulation error"
        ]
        typed = [
            f"{variable} ({kind}, node {node})" for node, variable, kind in self.covariate_types
        ]
        if typed:
            lines.append("covariate models: " + ", ".join(typed))
            lines.append(
                "  the draw order within a node defines non-nested models, so at most one order "
                "is correctly specified in general"
            )
        if self.monte_carlo_share:
            worst = max(self.monte_carlo_share, key=lambda name: self.monte_carlo_share[name])
            lines.append(
                f"Monte Carlo share of the variance: at most {self.max_share:.1%} ({worst}); "
                "raise n_draws to lower it"
            )
        if self.model_reliance is not None:
            lines.append(f"model reliance: {self.model_reliance}")
        for row in self.support:
            lines.append(
                f"  support of {row.plan} at {row.name!r}, dose {row.dose:g}: "
                f"{row.off_support:.1%} of simulated histories lie more than {SUPPORT_Z:g} "
                f"fitted SD from the fitted mean dose; {row.window:.1%} of observed doses lie "
                "in the same window"
            )
        return lines


def monte_carlo_share(
    data: LongitudinalData, fits: Sequence[RegimenFit], curve: FloatArray
) -> float:
    """The share of a parameter's curve variance that the simulation contributes.

    The within-unit variance of the stream-1 copies, divided by the number of copies behind the
    estimate, is the simulation variance of each unit's mean.  The units are independent given
    the data, so the simulation variance of the estimate adds over units.  A contrast reads the
    difference of the two regimens' copies, which share their uniforms.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    fits : sequence of RegimenFit
        The fit behind a level, or the fit and its reference behind a contrast.
    curve : FloatArray
        The parameter's curve, on the scale of the fits.

    Returns
    -------
    float
        ``nan`` when a fit holds no simulation, else a share in ``[0, 1]``.
    """
    simulations = [fit.simulation for fit in fits]
    if any(simulation is None for simulation in simulations):
        return float("nan")
    first = simulations[0]
    assert first is not None
    copies = np.asarray(first.copy_values, dtype=float)
    if len(fits) == 2:
        second = simulations[1]
        assert second is not None
        copies = copies - np.asarray(second.copy_values, dtype=float)
    per_unit = copies.var(axis=1, ddof=1) / first.n_draws
    weights = np.asarray(data.weights, dtype=float)
    weights = weights / weights.mean()
    simulated = float(np.sum(weights**2 * per_unit))
    if data.cluster is None:
        total = float(np.sum(curve**2))
    else:
        total = float(np.sum(np.bincount(data.cluster, weights=curve) ** 2))
    return float(min(1.0, simulated / total)) if total > 0.0 else float("nan")


# ------------------------------------------------------------------------------------ driver


def _regimen_fit(
    data: LongitudinalData,
    plan: Plan,
    mechanism: Mechanism,
    g_bounds: tuple[float, float],
    cell: Cell,
    psi: float,
    curve: FloatArray,
    simulation: FitSimulation,
) -> RegimenFit:
    """One cell's :class:`RegimenFit`: the estimate and curve, with the diagnostic steps."""
    n = data.n
    masks = plan.masks(data)
    numerator = plan.cumulative_numerator(data)
    unbounded, bounded = mechanism.cumulative_with_unbounded(data, plan, g_bounds)
    steps = []
    for time in range(1, cell.horizon + 1):
        at_risk, following = masks.at_risk(time), masks.following(time)
        _, clever = _clever_covariate(
            at_risk,
            following,
            bounded,
            time,
            numerator=None if numerator is None else numerator[:, time - 1],
        )
        steps.append(
            SequentialStep(
                time=time,
                trained_on=following,
                at_risk=at_risk,
                pseudo_outcome=np.zeros(n),
                initial=np.full(n, _FILLER),
                targeted=np.full(n, _FILLER),
                clever=clever,
                fluctuation=None,
            )
        )
    return RegimenFit(
        regimen=plan.regimen,
        psi_scaled=psi,
        influence_curve_scaled=curve,
        horizon=cell.horizon,
        cause=cell.cause,
        steps=tuple(steps),
        cumulative_unbounded=unbounded,
        cumulative=bounded,
        assignment=np.asarray(plan.values),
        obs_weights=np.asarray(data.weights, dtype=float),
        policy=plan.policy,
        cumulative_numerator=numerator,
        constant_terms=(),
        simulation=simulation,
    )


def _report(
    book: ModelBook,
    settings: GFormulaSettings,
    support: Sequence[SupportRow],
    fixed_doses: bool,
) -> GFormulaReport:
    models = tuple(
        ModelRecord(
            model.key,
            model.label,
            model.role,
            model.response,
            "constant" if model.learner is None else type(model.learner).__name__,
            model.stack,
            model.parametric,
            int(np.count_nonzero(model.rows)),
            model.n_parameters,
        )
        for model in book.models
    )
    reliance = (
        "a fixed continuous dose is not pathwise differentiable without a model. The estimate "
        "and its interval hold inside the fitted parametric laws, which are extrapolated to "
        "the fixed dose at every node. The interval does not cover a misspecified model."
        if fixed_doses
        else None
    )
    return GFormulaReport(
        settings.n_draws,
        settings.n_derivative_draws,
        settings.seed,
        tuple(
            (variable.node, variable.name, variable.kind)
            for block in book.variables
            for variable in block
            if variable.modelled
        ),
        tuple(tuple(variable.name for variable in block) for block in book.variables),
        models,
        book.parametric,
        reliance,
        tuple(support),
    )


def has_fixed_dose(data: LongitudinalData, plans: Sequence[Plan]) -> bool:
    """Whether any plan assigns a fixed dose at a continuous treatment node.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans.

    Returns
    -------
    bool
        ``True`` when a plan labels a continuous, non-held node with a number.
    """
    return any(
        data.is_continuous_node(node)
        and not data.is_held_node(node)
        and plan_node(plan.regimen, node)[0] == "label"
        for plan in plans
        for node in range(1, data.n_times + 1)
    )


def run_gformula(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    *,
    settings: GFormulaSettings,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    horizons: Sequence[int],
) -> tuple[dict[tuple[str, str | None, int], RegimenFit], GFormulaReport]:
    """Fit the models once, simulate every plan, and build one fit per reported cell.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    plans : sequence of Plan
        The resolved plans.
    mechanism : Mechanism
        The diagnostic mechanism, which the estimate does not read.
    settings : GFormulaSettings
        The draws, learners and seed.
    scaler : OutcomeScaler
        The outcome transformation.
    g_bounds : tuple of float
        The cumulative bounds of the diagnostic clever covariate.
    horizons : sequence of int
        The nodes a parameter is reported at.

    Returns
    -------
    tuple
        The fits keyed by ``(label, cause, horizon)`` in plan, cause, horizon order, and the
        report.
    """
    variables = resolve_covariates(data, settings.covariate_types)
    book = fit_models(data, plans, settings, scaler, variables)
    cells = _cells(data, horizons)
    weights = np.asarray(data.weights, dtype=float)
    mean_weight = float(np.mean(weights))
    projections = {model.key: book.projection(model) for model in book.stacked}
    fits: dict[tuple[str, str | None, int], RegimenFit] = {}
    support: list[SupportRow] = []
    fixed_doses = has_fixed_dose(data, plans)
    for plan in plans:
        result = _simulate_plan(book, plan, cells, settings)
        support.extend(result.support)
        psi = (weights @ result.phi) / weights.sum()
        for index, cell in enumerate(cells):
            curve = weights / mean_weight * (result.phi[:, index] - psi[index])
            for key, derivative in result.derivatives.items():
                curve = curve + projections[key] @ derivative[index]
            simulation = FitSimulation(
                n_draws=settings.n_draws,
                n_derivative_draws=settings.n_derivative_draws,
                copy_values=result.copy_values[:, :, index],
                exact=result.exact,
            )
            fits[(plan.label, cell.cause, cell.horizon)] = _regimen_fit(
                data, plan, mechanism, g_bounds, cell, float(psi[index]), curve, simulation
            )
    report = _report(book, settings, support, fixed_doses)
    for fit_key, fit in fits.items():
        assert fit.simulation is not None
        fits[fit_key] = replace(fit, simulation=replace(fit.simulation, report=report))
    return fits, report
