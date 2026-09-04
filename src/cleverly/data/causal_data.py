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
``treatment``        ``A``       treatment: arms, or a dose on a continuum
``covariates``       ``W``       baseline confounders
``delta``            ``Delta``   1 when the outcome is observed
``weights``          ``obsWeights``  observation weights
``cluster``          ``id``      independent unit for variance estimation
``intermediate``     ``Z``       binary intermediate, for controlled direct effects
===================  ==========  ==========================================

``treatment_kind`` decides how the treatment column is read, and it is *declared* rather
than inferred.  ``"discrete"`` codes it into arms; ``"continuous"`` keeps its own values,
leaves :attr:`~CausalData.treatment_levels` empty, and hands the mechanism over to a
conditional density.  Guessing from the number of distinct values would silently redefine
the estimand the day a new batch of data added one more dose.

Supplying ``weights`` changes the parameter being estimated, not just its weighting:
the estimand becomes the causal parameter in the weight-tilted population.
:mod:`cleverly.data.weighting` states that parameter, the influence function that goes
with it, and the readings of "weight" -- frequency counts, replicate weights -- that this
container refuses rather than approximates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import narwhals as nw
import numpy as np

from .._typing import BoolArray, FloatArray, IntArray
from ..exceptions import DataError
from ..utils.frames import (
    as_frame,
    backend_of,
    column_array,
    frame_from_dict,
    has_nulls,
    is_dataframe,
    matrix_from_columns,
)
from .validate import (
    MIN_OBSERVATIONS,
    arm_indicators,
    check_covariates,
    check_delta,
    check_outcome,
    check_weights,
    encode_binary,
    encode_clusters,
    encode_continuous_treatment,
    encode_treatment,
    infer_family,
    resolve_family,
)
from .weighting import (
    WeightReport,
    WeightSpec,
    _prepare_weights,
    describe_weights,
    effective_sample_size,
)

__all__ = ["CategoricalEncoding", "CausalData", "TreatmentKind"]

#: How the treatment column is to be read.  ``"discrete"`` codes it into arms; there is
#: no third reading, and the two are not points on a scale -- they select different
#: mechanisms (a distribution over arms versus a conditional density) and therefore
#: different estimands.
TreatmentKind = Literal["discrete", "continuous"]


def arm_share(
    treatment: FloatArray,
    weights: FloatArray,
    arm: float,
    mask: Any = None,
) -> float:
    """Weighted ``P(A = arm)`` on the rows ``mask`` keeps.

    One rule, stated once.  The same weighted share was open-coded at the summary
    header, at the cross-fitted and the stratified clever covariates, and at the
    simulated-confounding population fraction.  Written out four times it is four
    chances for a fold, a stratum and a report to disagree about what an arm's share
    of a population is.

    :attr:`CausalData.arm_fractions` deliberately does **not** route its two-arm case
    through this function.  There it takes the complement of
    :attr:`CausalData.treated_fraction`, because ``1 - E_w[A]`` and ``E_w[1 - A]``
    need not agree in the last bit, and a binary fit's ATC arithmetic is pinned to the
    complement.

    Parameters
    ----------
    treatment : ndarray
        Arm codes, as :attr:`CausalData.treatment` holds them.
    weights : ndarray
        Row masses over the same rows as ``treatment``.
    arm : float
        The arm code whose share is wanted.
    mask : ndarray or None
        Boolean mask or index array selecting the population.  ``None`` takes every row.

    Returns
    -------
    float
        The weighted fraction of the selected rows in ``arm``.
    """
    if mask is not None:
        treatment = treatment[mask]
        weights = weights[mask]
    return float(np.average(treatment == arm, weights=weights))


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
    mean one so weighted and unweighted variances are on the same scale, with
    :attr:`weight_spec` recording how those weights are to be read.
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
    treatment_levels: tuple[Any, ...] = (0, 1)
    #: Whether :attr:`treatment` holds arm codes or the treatment's own numeric values.
    #: ``"continuous"`` means there are no arms at all: :attr:`treatment_levels` is empty,
    #: :attr:`n_arms` is zero, and the mechanism is a conditional *density* rather than a
    #: distribution over a finite set.  Defaulted so every existing construction, and
    #: every ``dataclasses.replace`` of one, stays on the arm-coded path unchanged.
    treatment_kind: TreatmentKind = "discrete"
    delta_name: str | None = None
    cluster: IntArray | None = None
    cluster_name: str | None = None
    intermediate: FloatArray | None = None
    intermediate_name: str | None = None
    weights_name: str | None = None
    weight_spec: WeightSpec = field(default_factory=WeightSpec)
    dropped_covariates: tuple[str, ...] = ()
    encodings: tuple[CategoricalEncoding, ...] = ()
    #: Optional finite baseline partition used for conditional target parameters.  Codes
    #: are ``0..S-1`` and :attr:`strata_levels` maps them back to the caller's labels.
    #: The raw columns remain in the adjustment set; this is target metadata, not a
    #: replacement for confounding control.
    strata: IntArray | None = None
    strata_names: tuple[str, ...] = ()
    strata_levels: tuple[tuple[Any, ...], ...] = ()
    #: Name of the dataframe backend the data arrived in, or ``None`` for numpy input.
    #: A *name* and not the frame it came from: this used to hold the whole input
    #: frame, which pinned it in memory for the life of every result derived from the
    #: fit even though the only thing ever read off it was its namespace -- and which
    #: :func:`cleverly.load` had no way to restore, so a saved polars fit came back
    #: emitting pandas.
    backend: str | None = None

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
        weights_type: str = "probability",
        weights_estimated: bool = False,
        id: str | None = None,
        intermediate: str | None = None,
        strata: Sequence[str] | None = None,
        family: str = "auto",
        treatment_kind: TreatmentKind = "discrete",
    ) -> CausalData:
        """Build from a pandas or polars dataframe by column name.

        ``covariates=None`` uses every column that has not been claimed by
        another role.  Non-numeric covariate columns are one-hot encoded with the
        first (sorted) level dropped, which keeps the design matrix full rank for
        linear learners; the encoding is recorded on :attr:`encodings`.

        Parameters
        ----------
        weights_type:
            How to read ``weights``.  ``"probability"`` (also ``"sampling"``,
            ``"survey"``, ``"design"``) is the supported reading: the weights encode a
            tilt of the population, and the sample size is still the number of rows.
            ``"frequency"`` -- counts of identical units -- is a different experiment and
            is refused with instructions rather than silently mis-analysed.
        weights_estimated:
            Declare that the weights came out of a fitted model.  Changes no number; it
            makes the reports state that the intervals condition on the estimated
            weights.  See :mod:`cleverly.data.weighting`.
        treatment_kind:
            ``"continuous"`` reads the treatment as a quantity on a continuum rather than
            as a set of arms.  It is a declaration rather than something inferred: a dose
            recorded at fifteen distinct values could reasonably be either, and guessing
            from the number of levels would silently change the estimand when a new batch
            of data happened to add a sixteenth.
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
        strata_names = tuple(strata or ())
        absent_strata = [name for name in strata_names if name not in columns]
        if absent_strata:
            raise DataError(f"strata columns not found: {absent_strata}; available: {columns}")
        if len(set(strata_names)) != len(strata_names):
            raise DataError(f"strata= contains duplicate columns: {list(strata_names)}")
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

        outside = [name for name in strata_names if name not in covariate_names]
        if outside:
            raise DataError(
                f"baseline strata must also be adjustment covariates; add {outside} to "
                "covariates= (or leave covariates=None)"
            )

        strata_codes, strata_levels = _encode_strata(frame, strata_names)

        w_matrix, w_names, encodings = _encode_covariates(frame, covariate_names)

        for role, name in roles.items():
            _reject_null_labels(frame, name, role)

        return cls._build(
            outcome=column_array(frame, outcome),
            # Not cast: `encode_treatment` reads the dtype kind to tell a numeric arm
            # from a categorical one, and a cast to float would erase that distinction.
            treatment=frame[treatment].to_numpy(),
            covariates=w_matrix,
            covariate_names=w_names,
            delta=column_array(frame, delta) if delta is not None else None,
            weights=column_array(frame, weights) if weights is not None else None,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            cluster=frame[id].to_numpy() if id is not None else None,
            intermediate=(frame[intermediate].to_numpy() if intermediate is not None else None),
            family=family,
            treatment_kind=treatment_kind,
            outcome_name=outcome,
            treatment_name=treatment,
            delta_name=delta,
            weights_name=weights,
            cluster_name=id,
            intermediate_name=intermediate,
            strata=strata_codes,
            strata_names=strata_names,
            strata_levels=strata_levels,
            encodings=encodings,
            backend=backend_of(frame),
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
        weights_type: str = "probability",
        weights_estimated: bool = False,
        id: Any = None,
        intermediate: Any = None,
        family: str = "auto",
        treatment_kind: TreatmentKind = "discrete",
        outcome_name: str = "Y",
        treatment_name: str = "A",
        strata: np.ndarray | None = None,
        strata_names: Sequence[str] | None = None,
    ) -> CausalData:
        """Build from numpy arrays, mirroring ``tmle(Y, A, W, ...)`` in R.

        See :meth:`from_frame` for ``weights_type``, ``weights_estimated`` and
        ``treatment_kind``.
        """
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
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            cluster=None if id is None else np.asarray(id),
            intermediate=None if intermediate is None else np.asarray(intermediate),
            family=family,
            treatment_kind=treatment_kind,
            outcome_name=outcome_name,
            treatment_name=treatment_name,
            delta_name="Delta" if delta is not None else None,
            weights_name="weights" if weights is not None else None,
            cluster_name="id" if id is not None else None,
            intermediate_name="Z" if intermediate is not None else None,
            strata=None if strata is None else np.asarray(strata),
            strata_names=tuple(strata_names or ()),
            strata_levels=(),
            encodings=(),
            backend=None,
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
        weights_type: str,
        weights_estimated: bool,
        cluster: np.ndarray | None,
        intermediate: np.ndarray | None,
        family: str,
        treatment_kind: TreatmentKind = "discrete",
        outcome_name: str,
        treatment_name: str,
        delta_name: str | None,
        weights_name: str | None,
        cluster_name: str | None,
        intermediate_name: str | None,
        encodings: Sequence[CategoricalEncoding],
        backend: str | None,
        strata: np.ndarray | None = None,
        strata_names: Sequence[str] = (),
        strata_levels: Sequence[tuple[Any, ...]] = (),
    ) -> CausalData:
        n = len(outcome)
        if n < MIN_OBSERVATIONS:
            raise DataError(f"need at least {MIN_OBSERVATIONS} observations; got {n}")
        for label, arr in (
            (treatment_name, treatment),
            ("covariates", covariates),
        ):
            if len(arr) != n:
                raise DataError(f"{label} has length {len(arr)}, expected {n}")

        if treatment_kind == "continuous":
            a = encode_continuous_treatment(treatment, treatment_name)
            levels: tuple[object, ...] = ()
        elif treatment_kind == "discrete":
            a, levels = encode_treatment(treatment, treatment_name)
        else:
            raise DataError(
                f"treatment_kind must be 'discrete' or 'continuous'; got {treatment_kind!r}"
            )
        observed = (
            np.ones(n, dtype=bool) if delta is None else check_delta(delta, delta_name or "delta")
        )
        y = check_outcome(outcome, outcome_name, None if delta is None else observed)
        resolved_family = resolve_family(y, observed, family)

        w, w_names, dropped = check_covariates(covariates, list(covariate_names))
        obs_weights, spec = _prepare_weights(
            weights,
            n,
            weights_type=weights_type,
            weights_estimated=weights_estimated,
            weights_name=weights_name,
        )
        codes = None if cluster is None else encode_clusters(cluster, cluster_name or "id")
        z = None
        if intermediate is not None:
            z, _ = encode_binary(intermediate, intermediate_name or "Z")

        strata_codes: IntArray | None = None
        resolved_strata_levels: tuple[tuple[Any, ...], ...] = tuple(strata_levels)
        if strata is not None:
            raw = np.asarray(strata)
            if raw.ndim == 0 or raw.shape[0] != n:
                actual = 1 if raw.ndim == 0 else raw.shape[0]
                raise DataError(f"strata has length {actual}, expected {n}")
            if resolved_strata_levels:
                codes_array = np.asarray(raw, dtype=np.int64).reshape(-1)
                expected = np.arange(len(resolved_strata_levels), dtype=np.int64)
                if not np.array_equal(np.unique(codes_array), expected):
                    raise DataError(
                        "encoded strata must use every code 0..S-1 named by strata_levels"
                    )
                strata_codes = codes_array
            else:
                matrix = raw.reshape(n, -1)
                rows = [tuple(_python_scalar(value) for value in row) for row in matrix]
                strata_codes, resolved_strata_levels = _codes_for_rows(rows)
            if len(resolved_strata_levels) < 2:
                raise DataError("strata defines only one baseline stratum")
            if not strata_names:
                strata_names = tuple(f"V{j + 1}" for j in range(len(resolved_strata_levels[0])))
            if len(set(strata_names)) != len(tuple(strata_names)):
                raise DataError("strata_names must be distinct")
            if len(tuple(strata_names)) != len(resolved_strata_levels[0]):
                raise DataError("strata_names must have one entry per stratum-defining column")
            zero_mass = [
                resolved_strata_levels[code]
                for code in range(len(resolved_strata_levels))
                if float(obs_weights[strata_codes == code].sum()) <= 0.0
            ]
            if zero_mass:
                raise DataError(
                    f"baseline strata {zero_mass} have zero observation-weight mass in "
                    "the target population"
                )
        elif strata_names or resolved_strata_levels:
            raise DataError("strata_names/strata_levels require strata values")

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
            treatment_kind=treatment_kind,
            delta_name=delta_name,
            cluster=codes,
            cluster_name=cluster_name,
            intermediate=z,
            intermediate_name=intermediate_name,
            weights_name=weights_name,
            weight_spec=spec,
            dropped_covariates=tuple(dropped),
            encodings=retained_encodings,
            strata=strata_codes,
            strata_names=tuple(strata_names),
            strata_levels=resolved_strata_levels,
            backend=backend,
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
    def has_strata(self) -> bool:
        """Whether conditional target parameters were requested."""
        return self.strata is not None

    @property
    def n_strata(self) -> int:
        return len(self.strata_levels)

    def stratum_label(self, code: int) -> str:
        """A stable, human-readable label for one baseline stratum."""
        if self.strata is None or not 0 <= int(code) < self.n_strata:
            raise DataError(f"{code!r} is not one of this dataset's baseline strata")
        values = self.strata_levels[int(code)]
        return ", ".join(
            f"{name}={value!r}" for name, value in zip(self.strata_names, values, strict=True)
        )

    @property
    def is_weighted(self) -> bool:
        return bool(not np.allclose(self.weights, 1.0))

    @property
    def effective_n(self) -> float:
        """Kish effective sample size of the observation weights, ``(sum w)^2 / sum w^2``.

        The size of the unweighted sample that would carry the same information.  Equal
        to :attr:`n` when the weights are constant, and smaller otherwise -- by the
        design effect, which is the factor the weighting inflates the variance by.

        Not only a diagnostic: this is the sample size ``g_bounds="auto"`` is resolved
        at, since the rule is a bias-variance compromise and this is the number the
        variance side is really working from.
        """
        return effective_sample_size(self.weights)

    def weight_report(self) -> WeightReport:
        """Effective sample size, weight concentration and the estimand statement.

        See :mod:`cleverly.data.weighting`; ``print(data.weight_report().summary())``.
        """
        return describe_weights(self.weights, self.weight_spec)

    @property
    def treated_fraction(self) -> float:
        """Weighted ``P(A = 1)``, the denominator of a binary fit's ATT."""
        if self.is_continuous_treatment:
            raise DataError(
                f"{self.treatment_name} is continuous, so 'the treated fraction' names no "
                "quantity: there is no treated arm to take the share of. The ATT and ATC "
                "are undefined here for the same reason."
            )
        return float(np.average(self.treatment, weights=self.weights))

    @property
    def arm_fractions(self) -> FloatArray:
        """Weighted ``P(A = a)`` for every arm, in :attr:`arm_codes` order.

        The denominator of a conditional effect, which on a multi-valued treatment is one
        share per arm rather than one number: ``att[a vs r]`` averages over the units that
        received ``a``, and that is a different population for each ``a``.

        **Two arms take the complement of** :attr:`treated_fraction` rather than
        recomputing the share, for the reason
        :meth:`~cleverly.estimators._nuisance.Propensity.bounded` takes the complement of
        ``g1``: it is what keeps a binary fit's ATC arithmetic bit for bit what it was,
        since ``1 - E_w[A]`` and ``E_w[1 - A]`` need not agree in the last bit.  Every
        other arm count reads its share from :func:`arm_share`.
        """
        share = self.treated_fraction  # raises on a continuous treatment, as it should
        if self.n_arms == 2:
            return np.array([1.0 - share, share])
        return np.array([arm_share(self.treatment, self.weights, code) for code in self.arm_codes])

    # ------------------------------------------------------------------- arms

    @property
    def n_arms(self) -> int:
        """How many treatment levels the *declared support* carries.

        Read off :attr:`treatment_levels` rather than counted in
        :attr:`treatment`, so a subset or a bootstrap resample that happens to
        miss an arm still describes the same estimand -- and fails loudly at the
        point the missing arm is needed, instead of silently becoming a
        lower-dimensional problem.
        """
        return len(self.treatment_levels)

    @property
    def arm_codes(self) -> tuple[float, ...]:
        """The internal codes for the arms, ``(0.0, ..., K-1.0)``, ascending.

        These are what :class:`~cleverly.fluctuation.submodel.Submodel` and
        :class:`~cleverly.fluctuation.iterative.InitialFit` key their per-arm arrays by.
        :meth:`arm_label` maps one back to the level the user supplied.
        """
        return tuple(float(i) for i in range(self.n_arms))

    def arm_label(self, code: float) -> Any:
        """The user's original level for an internal arm code.

        Everything a reader sees -- parameter names, positivity tables, error
        messages -- goes through here, so a fit on ``{"low", "high"}`` is never
        reported in terms of ``1.0``.
        """
        if self.is_continuous_treatment:
            raise DataError(
                f"{self.treatment_name} is continuous and has no arms, so {float(code)!r} "
                "has no label. A continuous treatment's parameters are indexed by "
                "intervention -- a shift -- rather than by arm."
            )
        index = round(float(code))
        if index < 0 or index >= self.n_arms or float(index) != float(code):
            raise DataError(
                f"{float(code)!r} is not an arm of {self.treatment_name}; its levels are "
                f"{list(self.treatment_levels)} with codes {list(self.arm_codes)}"
            )
        return self.treatment_levels[index]

    @property
    def is_continuous_treatment(self) -> bool:
        """Whether the treatment lives on a continuum rather than a finite set of arms.

        Read off the declaration, not counted in the data: a continuous column that
        happens to visit only a few values is still continuous, and an arm-coded one is
        still arm-coded however many levels it has.
        """
        return self.treatment_kind == "continuous"

    @property
    def is_binary_treatment(self) -> bool:
        """Whether there are exactly two arms.

        A handful of estimands and analyses name one of exactly two arms -- ``ey1`` and
        ``ey0``, the incremental estimands, ``CTMLE`` -- and check this rather than
        assuming it.  Not the ATT, the omitted-variable bound, the E-value or the MNAR
        tilt, which are one parameter per contrast and read the arms off the parameter;
        this docstring named all four until they were, which is the shape of claim to
        check against the code rather than inherit.
        """
        return self.n_arms == 2

    # ----------------------------------------------------------------- design

    def treatment_block(self, codes: FloatArray) -> FloatArray:
        r"""Drop-first indicators for the treatment: ``(n, K-1)``.

        The point-treatment half of the shared arm-encoding rule; the encoding itself and
        the argument for it live in :func:`~cleverly.data.validate.arm_indicators`, which
        :meth:`~cleverly.longitudinal.data.LongitudinalData.history_design` also calls, so
        a design that conditions on an arm is coded the same way wherever it is built.
        With two arms that is a single column holding the 0/1 code itself, so a two-arm
        fit is unchanged bit for bit -- ``tests/unit/test_causal_data.py`` asserts that
        equality rather than leaving it to be read off this docstring.

        For a **continuous** treatment this is the single numeric column itself, which is
        why the branch is here rather than in the shared helper: there are no arms to
        indicate.  The objection to one numeric column -- that it imposes a linear
        dose-response -- is not answerable by indicators here; it is answered by the
        learner instead.  That is why the default library's splines and boosting matter
        more for a continuous treatment than for an arm-coded one, and why
        a linear regression on a continuous dose really does fit a straight line in the
        exposure.
        """
        c = np.asarray(codes, dtype=float).reshape(-1)
        if self.is_continuous_treatment:
            return c.reshape(-1, 1)
        return arm_indicators(c, self.n_arms)

    def treatment_design(self, *, include_intermediate: bool = False) -> FloatArray:
        """Design matrix for a model of the outcome: ``[A, W]``.

        ``A`` occupies :meth:`treatment_block` -- one column for a binary treatment,
        ``K-1`` indicator columns for a ``K``-armed one.

        With ``include_intermediate=True`` the intermediate variable is appended,
        which is what a controlled-direct-effect ``Q`` model conditions on.
        """
        blocks = [self.treatment_block(self.treatment), self.covariates]
        if include_intermediate:
            if self.intermediate is None:
                raise DataError("no intermediate variable was supplied")
            blocks.append(self.intermediate.reshape(-1, 1))
        return np.hstack(blocks)

    def counterfactual_design(
        self,
        treatment_value: float | FloatArray,
        *,
        intermediate_value: float | None = None,
    ) -> FloatArray:
        """``[a, W]`` with the treatment (and optionally ``Z``) set as asked.

        For an arm-coded treatment ``treatment_value`` is an *arm code* -- see
        :attr:`arm_codes` -- and is validated against the declared support, so a typo asks
        for an arm that does not exist rather than quietly producing an all-zero indicator
        block that the model reads as the dropped first arm.

        For a **continuous** treatment it is any float, or an ``(n,)`` array giving a
        value *per row*.  The per-row form is what a modified treatment policy needs:
        :math:`d(a, w) = a + \\delta` sets a different value for every unit, so the
        design cannot be built by broadcasting one number.  There is no declared support
        to validate against; how far outside the observed range a shifted value falls is
        a positivity question, reported by
        :func:`~cleverly.interventions.support.check_support` rather than raised here.

        The intermediate column is appended only when the data actually carries one, so
        that this and :meth:`treatment_design` cannot disagree about the width of the
        design.  Asking for a level the data has no column for is an error rather than a
        silently wider matrix than the model was trained on.
        """
        if self.is_continuous_treatment:
            values = np.asarray(treatment_value, dtype=float).reshape(-1)
            if values.size == 1:
                values = np.full(self.n, float(values[0]))
            elif values.size != self.n:
                raise DataError(
                    f"counterfactual_design was given {values.size} treatment values for "
                    f"{self.n} rows; pass one value for everybody or one per row"
                )
            if not np.all(np.isfinite(values)):
                raise DataError("counterfactual treatment values must all be finite")
        else:
            code = float(np.asarray(treatment_value, dtype=float).reshape(()))
            self.arm_label(code)  # validates the code against the declared support
            values = np.full(self.n, code)
        a = self.treatment_block(values)
        blocks = [a, self.covariates]
        if intermediate_value is not None:
            if self.intermediate is None:
                raise DataError(
                    f"counterfactual_design was given intermediate_value="
                    f"{intermediate_value!r} but the data has no intermediate variable; "
                    "the design would be one column wider than the outcome model's."
                )
            blocks.append(np.full((self.n, 1), float(intermediate_value)))
        return np.hstack(blocks)

    def missingness_design(self) -> FloatArray:
        """``[A, W]`` -- the conditioning set for ``P(Delta = 1 | A, W)``.

        Missingness at random is assumed to hold given treatment and baseline
        covariates: among units with the same ``(A, W)``, whether the outcome was
        recorded carries no information about what it would have been.
        :func:`~cleverly.sensitivity.missingness_tilt` makes that a dial rather than a
        premise.

        Note the intermediate variable is deliberately *not* in this design, and that is
        a modelling assumption rather than an oversight.  Conditioning on ``Z`` would be
        right if missingness were a consequence of the intermediate, but then the
        estimand would need a sequential (longitudinal) factorisation that this
        point-treatment estimator does not implement.  As it stands, combining
        ``delta=`` with ``intermediate=`` assumes ``Delta`` is not caused by ``Z`` --
        equivalently that ``Delta`` is independent of ``Z`` given ``(A, W)``, which
        holds in particular when the outcome's recording is decided before ``Z`` is
        realised.  Where that is implausible, the estimand belongs to an ``ltmle``-style
        longitudinal analysis; see :class:`cleverly.longitudinal.LTMLE` and its recipes in
        ``docs/user-guide/longitudinal.md``.

        :mod:`cleverly.estimators.direct_effect` states the rest of the assumptions this
        one belongs to, and derives the influence function they identify.
        """
        return self.treatment_design()

    # -------------------------------------------------------------- reshaping

    def subset(self, index: Any) -> CausalData:
        """A copy holding only the selected rows.

        Used by the data-subset refutation test and the bootstrap.  Weights are
        re-normalised for the subset, and cluster codes are re-derived so they
        stay contiguous.

        Re-normalising is what makes a bootstrap replicate estimate the same parameter
        as the original fit: the tilt is defined by the weights *relative to the
        resample's own mean*, so a replicate that happens to draw the heavy rows must
        renormalise or it would target a different tilt.  The scale factor is folded into
        :attr:`weight_spec` so the supplied weights stay recoverable.
        """
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        if idx.size < MIN_OBSERVATIONS:
            raise DataError(f"subset has {idx.size} rows; need at least {MIN_OBSERVATIONS}")
        cluster = (
            None if self.cluster is None else np.unique(self.cluster[idx], return_inverse=True)[1]
        )
        selected = self.weights[idx]
        mean = float(selected.mean())
        return replace(
            self,
            outcome=self.outcome[idx],
            treatment=self.treatment[idx],
            covariates=self.covariates[idx],
            weights=check_weights(selected, idx.size),
            weight_spec=self.weight_spec.rescaled(self.weight_spec.scale * mean),
            observed=self.observed[idx],
            cluster=None if cluster is None else np.asarray(cluster, dtype=np.int64),
            intermediate=None if self.intermediate is None else self.intermediate[idx],
            strata=None if self.strata is None else self.strata[idx],
        )

    def with_treatment(self, treatment: FloatArray) -> CausalData:
        """A copy with the treatment replaced (used by the placebo refuter).

        The replacement is in *arm codes* and is validated against the declared support,
        which is stricter than re-encoding it would be -- and deliberately so.  The
        placebo refuter permutes the existing treatment, and a permutation must keep the
        arms it permutes: re-encoding would silently accept a replacement that dropped an
        arm, and the refuted fit would then estimate a different parameter from the one
        it is supposed to be a null for.
        """
        a = np.asarray(treatment, dtype=float).reshape(-1)
        if a.size != self.n:
            raise DataError(f"replacement treatment has length {a.size}, expected {self.n}")
        if self.is_continuous_treatment:
            # No declared support to keep, so the arm check below has nothing to check.
            # A permutation of a continuous column keeps its marginal distribution, which
            # is what the placebo refuter needs.
            if not np.all(np.isfinite(a)):
                raise DataError("replacement treatment contains non-finite values")
            return replace(self, treatment=a)
        found = np.unique(a)
        if not np.array_equal(found, np.asarray(self.arm_codes, dtype=float)):
            raise DataError(
                f"replacement {self.treatment_name} has arm codes {found.tolist()}, but the "
                f"data declares {list(self.arm_codes)} (levels {list(self.treatment_levels)}). "
                "A replacement treatment must keep every arm, or the refitted estimate is "
                "not a null for the same parameter."
            )
        return replace(self, treatment=a)

    def with_outcome(
        self, outcome: Any, *, family: str = "auto", name: str = "replacement outcome"
    ) -> CausalData:
        """Return a copy with a validated replacement outcome.

        Parameters
        ----------
        outcome : array-like
            One replacement value per analysis row. Missing or non-finite values are
            allowed only on rows already marked as unobserved.
        family : {"auto", "gaussian", "binomial"}
            Outcome family for the replacement. ``"auto"`` infers it from observed
            replacement values.
        name : str
            What the caller calls this outcome. Every refusal names it, so a caller that
            passes a named argument can report the argument rather than the role.

        Returns
        -------
        CausalData
            A replacement that preserves every role except outcome and family.

        Raises
        ------
        DataError
            If length, observed values, family, or family support is invalid.
        """
        values = np.asarray(outcome)
        if values.ndim == 0 or values.reshape(-1).size != self.n:
            actual = 1 if values.ndim == 0 else values.reshape(-1).size
            raise DataError(f"{name} has length {actual}, expected {self.n}")
        cleaned = check_outcome(values, name, self.observed)
        resolved = infer_family(cleaned, self.observed) if family == "auto" else family
        if resolved not in ("binomial", "gaussian"):
            raise DataError(
                f"{name} family must be 'binomial', 'gaussian' or 'auto'; got {family!r}"
            )
        if resolved == "binomial":
            observed_values = np.unique(cleaned[self.observed])
            if not np.all(np.isin(observed_values, (0.0, 1.0))):
                raise DataError(
                    f"{name} with family='binomial' requires 0/1 observed "
                    f"values; observed {observed_values[:6].tolist()}"
                )
        return replace(self, outcome=cleaned, family=resolved)

    def with_covariates(
        self, covariates: Any, *, name: str = "replacement covariates"
    ) -> CausalData:
        """Return a copy with a validated complete covariate design.

        An array is read by position against :attr:`covariate_names`.  A pandas or polars
        frame is matched on its column names instead, and is reordered into the fitted
        order; a frame that does not name exactly this design is refused rather than read
        positionally, because a permuted frame would move every estimate with no error.  A
        row index that is not ``0..n-1`` is refused for the same reason: this method reads
        rows by position and never aligns on an index.

        Each recorded :class:`CategoricalEncoding` is checked against the replacement, so
        a block that is no longer a drop-first indicator block is refused here.  What is
        preserved is every other role, every covariate name, and every encoding
        declaration: this method replaces the values of the design and nothing else.  The
        replacement is copied, so a caller that reuses one buffer across replicates cannot
        mutate the returned data.

        Parameters
        ----------
        covariates : array-like or DataFrame
            Complete encoded design with the same shape as the fitted design.  A dataframe
            must carry exactly the columns named by :attr:`covariate_names`.
        name : str
            Name used in validation errors.

        Returns
        -------
        CausalData
            A copy holding a private copy of the replacement design.

        Raises
        ------
        DataError
            If the frame's columns or row index, the shape, the finiteness, or a recorded
            categorical block is invalid.
        """
        if is_dataframe(covariates):
            values = self._covariates_from_frame(covariates, name)
        else:
            values = np.array(covariates, dtype=float, copy=True)
        if values.shape != self.covariates.shape:
            raise DataError(f"{name} has shape {values.shape}, expected {self.covariates.shape}")
        if not np.all(np.isfinite(values)):
            raise DataError(f"{name} contains non-finite values")
        self._check_encoded_blocks(values, name)
        return replace(self, covariates=values)

    def _covariates_from_frame(self, covariates: Any, name: str) -> FloatArray:
        """Read a replacement design out of a dataframe by column name.

        Parameters
        ----------
        covariates : DataFrame
            Replacement design, as a pandas or polars frame.
        name : str
            Name used in validation errors.

        Returns
        -------
        numpy.ndarray
            The frame's columns in :attr:`covariate_names` order.
        """
        expected = list(self.covariate_names)
        if len(set(expected)) != len(expected):
            raise DataError(
                f"{name} was given as a dataframe, but this design repeats a covariate "
                "name, so a name does not identify one column. Pass a numpy array in "
                f"covariate_names order instead; the order is {expected}."
            )
        frame = as_frame(covariates)
        columns = list(frame.columns)
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        if duplicates:
            raise DataError(f"{name} repeats the columns {duplicates}")
        if set(columns) != set(expected):
            missing = [column for column in expected if column not in columns]
            unexpected = [column for column in columns if column not in expected]
            raise DataError(
                f"{name} must carry exactly the fitted covariate columns. Missing "
                f"{missing}; unexpected {unexpected}. The expected columns are {expected}."
            )
        index = nw.maybe_get_index(frame)
        if index is not None:
            labels = np.asarray(index)
            rows = int(frame.shape[0])
            if labels.shape != (rows,) or not np.array_equal(labels, np.arange(rows)):
                raise DataError(
                    f"{name} has a row index that is not 0..n-1. This method reads rows by "
                    "position and does not align on an index. Call reset_index(drop=True) "
                    "on the frame first, or pass a numpy array."
                )
        return matrix_from_columns(frame, expected)

    def _check_encoded_blocks(self, values: FloatArray, name: str) -> None:
        """Refuse a replacement that breaks a recorded drop-first indicator block.

        Parameters
        ----------
        values : numpy.ndarray
            Replacement design, already aligned to :attr:`covariate_names`.
        name : str
            Name used in validation errors.
        """
        position = {column: j for j, column in enumerate(self.covariate_names)}
        for encoding in self.encodings:
            # An encoding survives on `encodings` when duplicate-column removal kept only
            # part of its block, so only the retained indicators can be checked.  Any
            # subset of a valid drop-first block is itself one, so the weaker check on a
            # partial block is the strongest true statement available here.
            columns = [position[item] for item in encoding.generated if item in position]
            if not columns:
                continue
            block = values[:, columns]
            names = [self.covariate_names[j] for j in columns]
            if not bool(np.all(np.isin(block, (0.0, 1.0)))):
                raise DataError(
                    f"{name} does not encode covariate {encoding.column!r} as indicators: "
                    f"columns {names} must hold 0 or 1. This data records a drop-first "
                    f"encoding of {encoding.column!r} over levels "
                    f"{list(encoding.levels)}, and every estimate reads it that way."
                )
            active = np.count_nonzero(block, axis=1)
            if int(np.max(active, initial=0)) > 1:
                rows = int(np.count_nonzero(active > 1))
                raise DataError(
                    f"{name} sets more than one indicator of covariate "
                    f"{encoding.column!r} on {rows} of {values.shape[0]} rows; columns "
                    f"{names} are a drop-first block, so at most one is active. The "
                    f"dropped level {encoding.dropped_level!r} is the all-zero row."
                )

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
        return frame_from_dict(payload, backend=self.backend)

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
        return frame_from_dict(payload, backend=self.backend)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = [
            f"n={self.n}",
            f"family={self.family!r}",
            f"covariates={self.n_covariates}",
        ]
        if self.is_continuous_treatment:
            low, high = float(self.treatment.min()), float(self.treatment.max())
            parts.append(f"{self.treatment_name} in [{low:.3g}, {high:.3g}]")
        else:
            parts.append(f"P(A=1)={self.treated_fraction:.3f}")
        if self.has_missing_outcome:
            parts.append(f"observed={float(self.observed.mean()):.3f}")
        if self.cluster is not None:
            parts.append(f"clusters={self.n_clusters}")
        if self.has_intermediate:
            parts.append("intermediate=yes")
        if self.is_weighted:
            parts.append(f"weighted=yes (effective n={self.effective_n:.0f})")
        return f"CausalData({', '.join(parts)})"


def _reject_null_labels(frame: nw.DataFrame[Any], name: str, role: str) -> None:
    """Refuse a null in a column that will not be cast to float.

    A numeric column can carry one: :func:`~cleverly.utils.frames.column_array` casts
    it and the null arrives downstream as ``nan``, where :mod:`cleverly.data.validate`
    either rejects it with a message about *that* role or -- for the outcome -- reads it
    against ``delta``.  A non-numeric one cannot: ``to_numpy`` on a nullable or
    arrow-backed column hands back an ``object`` array carrying ``pd.NA``, and the first
    comparison made on it raises ``TypeError: boolean value of NA is ambiguous`` from
    inside numpy rather than anything a caller can act on.  A boolean column is the easy
    way to reach that, because narwhals does not count ``Boolean`` as numeric and
    ``dtype_backend="pyarrow"`` makes every such column nullable by construction.

    The branch is on the *column's* logical type, read through narwhals -- not on which
    dataframe library produced it, which stays something this package never asks.
    """
    if frame.schema[name].is_numeric() or not has_nulls(frame, name):
        return
    raise DataError(
        f"{role} column {name!r} contains missing values and is not numeric. "
        "Impute them, drop those rows, or encode the column yourself before "
        "handing it to CausalData."
    )


def _python_scalar(value: Any) -> Any:
    """Turn numpy scalar labels into serialisable Python values."""
    return value.item() if isinstance(value, np.generic) else value


def _codes_for_rows(
    rows: Sequence[tuple[Any, ...]],
) -> tuple[IntArray, tuple[tuple[Any, ...], ...]]:
    """Encode a finite row partition in stable first-appearance order."""
    lookup: dict[tuple[Any, ...], int] = {}
    levels: list[tuple[Any, ...]] = []
    codes = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        try:
            code = lookup.get(row)
        except TypeError as exc:  # an array/list-valued dataframe cell
            raise DataError("strata columns must contain scalar, hashable values") from exc
        if code is None:
            code = len(levels)
            lookup[row] = code
            levels.append(row)
        codes[i] = code
    return codes, tuple(levels)


def _encode_strata(
    frame: nw.DataFrame[Any], names: Sequence[str]
) -> tuple[IntArray | None, tuple[tuple[Any, ...], ...]]:
    """Encode the requested baseline columns without changing their covariate encoding."""
    if not names:
        return None, ()
    for name in names:
        if has_nulls(frame, name):
            raise DataError(f"strata column {name!r} contains missing values")
    columns = [frame[name].to_numpy() for name in names]
    rows = [tuple(_python_scalar(column[i]) for column in columns) for i in range(len(columns[0]))]
    codes, levels = _codes_for_rows(rows)
    if len(levels) < 2:
        raise DataError("strata= defines only one baseline stratum")
    return codes, levels


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
        _reject_null_labels(frame, name, "covariate")
        # Read off the declared schema rather than off whatever dtype ``to_numpy`` chose.
        # The two agree for every spelling of a boolean column that reaches here -- that
        # was measured, not assumed, and ``test_a_boolean_covariate_stays_one_column_under_
        # every_spelling`` says so rather than claiming to guard a bug -- but the schema is
        # the thing actually being asked about, and not depending on narwhals' dtype
        # inference is the whole point of the cast in ``column_array``.
        if dtype == nw.Boolean:
            blocks.append(column_array(frame, name).reshape(-1, 1))
            out_names.append(name)
            encodings.append(
                CategoricalEncoding(
                    column=name,
                    levels=(False, True),
                    dropped_level=False,
                    generated=(name,),
                )
            )
            continue
        values = frame[name].to_numpy()
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
