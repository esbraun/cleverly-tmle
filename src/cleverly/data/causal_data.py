"""The :class:`CausalData` container.

A single validated, numpy-backed view of a causal dataset that every estimator
consumes.  Users hand in a pandas or polars dataframe (or raw arrays); the
container records which backend it came from so results can be returned in the
same one.

The roles mirror the arguments of R's ``tmle()``:

===================  ==========  ==========================================
role                 R name      meaning
===================  ==========  ==========================================
``outcome``          ``Y``       outcome, possibly missing where ``delta=0``
``treatment``        ``A``       binary treatment indicator
``covariates``       ``W``       baseline confounders
``delta``            ``Delta``   1 when the outcome is observed
``weights``          ``obsWeights``  observation weights
``cluster``          ``id``      independent unit for variance estimation
``intermediate``     ``Z``       binary intermediate, for controlled direct effects
===================  ==========  ==========================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import narwhals as nw
import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import DataError
from ..utils.frames import as_frame, frame_from_dict, is_dataframe, matrix_from_columns
from .validate import (
    check_covariates,
    check_delta,
    check_outcome,
    check_weights,
    encode_binary,
    encode_clusters,
    infer_family,
)

__all__ = ["CategoricalEncoding", "CausalData"]

_MIN_OBSERVATIONS = 10


@dataclass(frozen=True)
class CategoricalEncoding:
    """Record of how one non-numeric covariate was expanded into indicators.

    Stored so a fit is reproducible and so the same expansion can be replayed on
    new data (for example inside a refutation test that perturbs one column).
    """

    column: str
    levels: tuple[Any, ...]
    dropped_level: Any
    generated: tuple[str, ...]


@dataclass(frozen=True)
class CausalData:
    """Validated inputs for a point-treatment causal estimator.

    Build one with :meth:`from_frame` (pandas/polars) or :meth:`from_arrays`
    (numpy).  All attributes are plain numpy arrays; ``weights`` is normalised to
    mean one so weighted and unweighted variances are on the same scale.
    """

    outcome: FloatArray
    treatment: FloatArray
    covariates: FloatArray
    covariate_names: tuple[str, ...]
    weights: FloatArray
    observed: BoolArray
    family: str
    outcome_name: str = "Y"
    treatment_name: str = "A"
    treatment_levels: tuple[Any, Any] = (0, 1)
    delta_name: str | None = None
    cluster: IntArray | None = None
    cluster_name: str | None = None
    intermediate: FloatArray | None = None
    intermediate_name: str | None = None
    weights_name: str | None = None
    dropped_covariates: tuple[str, ...] = ()
    encodings: tuple[CategoricalEncoding, ...] = ()
    _template: Any = None

    # ------------------------------------------------------------------ build

    @classmethod
    def from_frame(
        cls,
        data: Any,
        *,
        outcome: str,
        treatment: str,
        covariates: Sequence[str] | None = None,
        delta: str | None = None,
        weights: str | None = None,
        id: str | None = None,
        intermediate: str | None = None,
        family: str = "auto",
    ) -> CausalData:
        """Build from a pandas or polars dataframe by column name.

        ``covariates=None`` uses every column that has not been claimed by
        another role.  Non-numeric covariate columns are one-hot encoded with the
        first (sorted) level dropped, which keeps the design matrix full rank for
        linear learners; the encoding is recorded on :attr:`encodings`.
        """
        if not is_dataframe(data):
            raise DataError(
                "from_frame expects a pandas or polars DataFrame; "
                "use CausalData.from_arrays for numpy input"
            )
        frame = as_frame(data)
        columns = list(frame.columns)

        roles = {"outcome": outcome, "treatment": treatment}
        for role, name in (
            ("delta", delta),
            ("weights", weights),
            ("id", id),
            ("intermediate", intermediate),
        ):
            if name is not None:
                roles[role] = name
        missing = {role: name for role, name in roles.items() if name not in columns}
        if missing:
            raise DataError(f"columns not found in the frame: {missing}; available: {columns}")

        claimed = set(roles.values())
        if covariates is None:
            covariate_names = [name for name in columns if name not in claimed]
            if not covariate_names:
                raise DataError(
                    "no covariate columns left after assigning roles; pass covariates=[...]"
                )
        else:
            covariate_names = list(covariates)
            overlap = claimed.intersection(covariate_names)
            if overlap:
                raise DataError(
                    f"columns {sorted(overlap)} are used both as a role and a covariate"
                )
            absent = [name for name in covariate_names if name not in columns]
            if absent:
                raise DataError(f"covariate columns not found: {absent}; available: {columns}")

        w_matrix, w_names, encodings = _encode_covariates(frame, covariate_names)

        return cls._build(
            outcome=frame[outcome].to_numpy(),
            treatment=frame[treatment].to_numpy(),
            covariates=w_matrix,
            covariate_names=w_names,
            delta=frame[delta].to_numpy() if delta is not None else None,
            weights=frame[weights].to_numpy() if weights is not None else None,
            cluster=frame[id].to_numpy() if id is not None else None,
            intermediate=(frame[intermediate].to_numpy() if intermediate is not None else None),
            family=family,
            outcome_name=outcome,
            treatment_name=treatment,
            delta_name=delta,
            weights_name=weights,
            cluster_name=id,
            intermediate_name=intermediate,
            encodings=encodings,
            template=frame,
        )

    @classmethod
    def from_arrays(
        cls,
        outcome: Any,
        treatment: Any,
        covariates: Any,
        *,
        covariate_names: Sequence[str] | None = None,
        delta: Any = None,
        weights: Any = None,
        id: Any = None,
        intermediate: Any = None,
        family: str = "auto",
        outcome_name: str = "Y",
        treatment_name: str = "A",
    ) -> CausalData:
        """Build from numpy arrays, mirroring ``tmle(Y, A, W, ...)`` in R."""
        w = np.asarray(covariates, dtype=float)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        if covariate_names is None:
            names = [f"W{j + 1}" for j in range(w.shape[1])]
        else:
            names = list(covariate_names)
        return cls._build(
            outcome=np.asarray(outcome),
            treatment=np.asarray(treatment),
            covariates=w,
            covariate_names=names,
            delta=None if delta is None else np.asarray(delta),
            weights=None if weights is None else np.asarray(weights),
            cluster=None if id is None else np.asarray(id),
            intermediate=None if intermediate is None else np.asarray(intermediate),
            family=family,
            outcome_name=outcome_name,
            treatment_name=treatment_name,
            delta_name="Delta" if delta is not None else None,
            weights_name="weights" if weights is not None else None,
            cluster_name="id" if id is not None else None,
            intermediate_name="Z" if intermediate is not None else None,
            encodings=(),
            template=None,
        )

    @classmethod
    def _build(
        cls,
        *,
        outcome: np.ndarray,
        treatment: np.ndarray,
        covariates: FloatArray,
        covariate_names: Sequence[str],
        delta: np.ndarray | None,
        weights: np.ndarray | None,
        cluster: np.ndarray | None,
        intermediate: np.ndarray | None,
        family: str,
        outcome_name: str,
        treatment_name: str,
        delta_name: str | None,
        weights_name: str | None,
        cluster_name: str | None,
        intermediate_name: str | None,
        encodings: Sequence[CategoricalEncoding],
        template: Any,
    ) -> CausalData:
        n = len(outcome)
        if n < _MIN_OBSERVATIONS:
            raise DataError(f"need at least {_MIN_OBSERVATIONS} observations; got {n}")
        for label, arr in (
            (treatment_name, treatment),
            ("covariates", covariates),
        ):
            if len(arr) != n:
                raise DataError(f"{label} has length {len(arr)}, expected {n}")

        a, levels = encode_binary(treatment, treatment_name)
        observed = (
            np.ones(n, dtype=bool) if delta is None else check_delta(delta, delta_name or "delta")
        )
        y = check_outcome(outcome, outcome_name, None if delta is None else observed)
        resolved_family = infer_family(y, observed) if family == "auto" else family
        if resolved_family not in ("binomial", "gaussian"):
            raise DataError(f"family must be 'binomial', 'gaussian' or 'auto'; got {family!r}")
        if resolved_family == "binomial":
            observed_values = np.unique(y[observed])
            if not np.all(np.isin(observed_values, (0.0, 1.0))):
                raise DataError(
                    "family='binomial' requires a 0/1 outcome; observed values "
                    f"{observed_values[:6].tolist()}"
                )

        w, w_names, dropped = check_covariates(covariates, list(covariate_names))
        obs_weights = check_weights(weights, n, weights_name or "weights")
        codes = None if cluster is None else encode_clusters(cluster, cluster_name or "id")
        z = None
        if intermediate is not None:
            z, _ = encode_binary(intermediate, intermediate_name or "Z")

        kept = set(w_names)
        retained_encodings = tuple(
            enc for enc in encodings if any(name in kept for name in enc.generated)
        )

        return cls(
            outcome=y,
            treatment=a,
            covariates=w,
            covariate_names=tuple(w_names),
            weights=obs_weights,
            observed=observed,
            family=resolved_family,
            outcome_name=outcome_name,
            treatment_name=treatment_name,
            treatment_levels=levels,
            delta_name=delta_name,
            cluster=codes,
            cluster_name=cluster_name,
            intermediate=z,
            intermediate_name=intermediate_name,
            weights_name=weights_name,
            dropped_covariates=tuple(dropped),
            encodings=retained_encodings,
            _template=template,
        )

    # ------------------------------------------------------------- properties

    @property
    def n(self) -> int:
        """Number of observations."""
        return int(self.outcome.shape[0])

    @property
    def n_covariates(self) -> int:
        return int(self.covariates.shape[1])

    @property
    def n_clusters(self) -> int:
        """Number of independent units: clusters if given, else observations."""
        if self.cluster is None:
            return self.n
        return int(np.unique(self.cluster).size)

    @property
    def has_missing_outcome(self) -> bool:
        return bool(not np.all(self.observed))

    @property
    def has_intermediate(self) -> bool:
        return self.intermediate is not None

    @property
    def is_weighted(self) -> bool:
        return bool(not np.allclose(self.weights, 1.0))

    @property
    def treated_fraction(self) -> float:
        """Weighted ``P(A = 1)``, the denominator of the ATT."""
        return float(np.average(self.treatment, weights=self.weights))

    @property
    def backend(self) -> str | None:
        """Name of the dataframe backend the data came from, if any."""
        if self._template is None:
            return None
        return str(nw.get_native_namespace(self._template).__name__)

    # ----------------------------------------------------------------- design

    def treatment_design(self, *, include_intermediate: bool = False) -> FloatArray:
        """Design matrix for a model of the outcome: ``[A, W]``.

        With ``include_intermediate=True`` the intermediate variable is appended,
        which is what a controlled-direct-effect ``Q`` model conditions on.
        """
        blocks = [self.treatment.reshape(-1, 1), self.covariates]
        if include_intermediate:
            if self.intermediate is None:
                raise DataError("no intermediate variable was supplied")
            blocks.append(self.intermediate.reshape(-1, 1))
        return np.hstack(blocks)

    def counterfactual_design(
        self,
        treatment_value: float,
        *,
        intermediate_value: float | None = None,
    ) -> FloatArray:
        """``[a, W]`` with the treatment (and optionally ``Z``) set to a constant."""
        a = np.full((self.n, 1), float(treatment_value))
        blocks = [a, self.covariates]
        if intermediate_value is not None:
            blocks.append(np.full((self.n, 1), float(intermediate_value)))
        return np.hstack(blocks)

    def missingness_design(self) -> FloatArray:
        """``[A, W]`` -- the conditioning set for ``P(Delta = 1 | A, W)``."""
        return self.treatment_design()

    # -------------------------------------------------------------- reshaping

    def subset(self, index: Any) -> CausalData:
        """A copy holding only the selected rows.

        Used by the data-subset refutation test and the bootstrap.  Weights are
        re-normalised for the subset, and cluster codes are re-derived so they
        stay contiguous.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        if idx.size < _MIN_OBSERVATIONS:
            raise DataError(f"subset has {idx.size} rows; need at least {_MIN_OBSERVATIONS}")
        cluster = (
            None if self.cluster is None else np.unique(self.cluster[idx], return_inverse=True)[1]
        )
        return replace(
            self,
            outcome=self.outcome[idx],
            treatment=self.treatment[idx],
            covariates=self.covariates[idx],
            weights=check_weights(self.weights[idx], idx.size),
            observed=self.observed[idx],
            cluster=None if cluster is None else np.asarray(cluster, dtype=np.int64),
            intermediate=None if self.intermediate is None else self.intermediate[idx],
            _template=None if self._template is None else self._template[:1],
        )

    def with_treatment(self, treatment: FloatArray) -> CausalData:
        """A copy with the treatment replaced (used by the placebo refuter)."""
        a, _ = encode_binary(np.asarray(treatment), self.treatment_name)
        if a.size != self.n:
            raise DataError(f"replacement treatment has length {a.size}, expected {self.n}")
        return replace(self, treatment=a)

    def with_extra_covariate(self, values: FloatArray, name: str) -> CausalData:
        """A copy with one extra covariate column appended."""
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        if column.shape[0] != self.n:
            raise DataError(f"extra covariate has length {column.shape[0]}, expected {self.n}")
        if name in self.covariate_names:
            raise DataError(f"covariate {name!r} already exists")
        return replace(
            self,
            covariates=np.hstack([self.covariates, column]),
            covariate_names=(*self.covariate_names, name),
        )

    def without_covariates(self, names: Sequence[str]) -> CausalData:
        """A copy with the named covariates removed (used by benchmarking)."""
        drop = set(names)
        unknown = drop.difference(self.covariate_names)
        if unknown:
            raise DataError(f"unknown covariates {sorted(unknown)}")
        keep = [j for j, name in enumerate(self.covariate_names) if name not in drop]
        if not keep:
            raise DataError("cannot drop every covariate")
        return replace(
            self,
            covariates=self.covariates[:, keep],
            covariate_names=tuple(self.covariate_names[j] for j in keep),
        )

    # ------------------------------------------------------------------ output

    def frame_like(self, payload: dict[str, Any]) -> Any:
        """Build a dataframe of ``payload`` in the backend this data came from.

        Every result object routes its tabular output through here, which is what
        makes "pandas in, pandas out; polars in, polars out" hold throughout.
        """
        return frame_from_dict(payload, like=self._template)

    def to_frame(self) -> Any:
        """Round-trip the validated data back into a dataframe.

        The frame uses the backend the data arrived in (pandas when it arrived as
        numpy arrays), which makes it easy to inspect exactly what the estimator
        saw after encoding and validation.
        """
        payload: dict[str, Any] = {
            self.outcome_name: self.outcome,
            self.treatment_name: self.treatment,
        }
        for j, name in enumerate(self.covariate_names):
            payload[name] = self.covariates[:, j]
        if self.delta_name is not None or self.has_missing_outcome:
            payload[self.delta_name or "Delta"] = self.observed.astype(float)
        if self.is_weighted or self.weights_name is not None:
            payload[self.weights_name or "weights"] = self.weights
        if self.cluster is not None:
            payload[self.cluster_name or "id"] = self.cluster
        if self.intermediate is not None:
            payload[self.intermediate_name or "Z"] = self.intermediate
        return frame_from_dict(payload, like=self._template)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = [
            f"n={self.n}",
            f"family={self.family!r}",
            f"covariates={self.n_covariates}",
            f"P(A=1)={self.treated_fraction:.3f}",
        ]
        if self.has_missing_outcome:
            parts.append(f"observed={float(self.observed.mean()):.3f}")
        if self.cluster is not None:
            parts.append(f"clusters={self.n_clusters}")
        if self.has_intermediate:
            parts.append("intermediate=yes")
        if self.is_weighted:
            parts.append("weighted=yes")
        return f"CausalData({', '.join(parts)})"


def _encode_covariates(
    frame: nw.DataFrame[Any], names: Sequence[str]
) -> tuple[FloatArray, list[str], tuple[CategoricalEncoding, ...]]:
    """Expand non-numeric covariate columns into 0/1 indicators."""
    numeric: list[str] = []
    blocks: list[FloatArray] = []
    out_names: list[str] = []
    encodings: list[CategoricalEncoding] = []

    schema = frame.schema
    for name in names:
        dtype = schema[name]
        if dtype.is_numeric():
            numeric.append(name)
            continue
        values = frame[name].to_numpy()
        if values.dtype == bool:
            blocks.append(np.asarray(values, dtype=float).reshape(-1, 1))
            out_names.append(name)
            continue
        levels = tuple(np.unique(values).tolist())
        if len(levels) < 2:
            raise DataError(f"covariate {name!r} is constant")
        if len(levels) > 50:
            raise DataError(
                f"covariate {name!r} has {len(levels)} levels; encode it yourself "
                "(target/frequency encoding) before handing it to CausalData"
            )
        dropped, kept = levels[0], levels[1:]
        generated = tuple(f"{name}__{level}" for level in kept)
        indicators = np.column_stack([np.asarray(values == level, dtype=float) for level in kept])
        blocks.append(indicators)
        out_names.extend(generated)
        encodings.append(
            CategoricalEncoding(
                column=name, levels=levels, dropped_level=dropped, generated=generated
            )
        )

    if numeric:
        blocks.insert(0, matrix_from_columns(frame, numeric))
        out_names = numeric + out_names

    if not blocks:
        raise DataError("no usable covariate columns")
    return np.hstack(blocks), out_names, tuple(encodings)
