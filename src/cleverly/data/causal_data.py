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
    check_covariates,
    check_delta,
    check_outcome,
    check_weights,
    encode_binary,
    encode_clusters,
    encode_continuous_treatment,
    encode_treatment,
    infer_family,
)
from .weighting import (
    WeightReport,
    WeightSpec,
    describe_weights,
    effective_sample_size,
    resolve_weight_kind,
    warn_if_concentrated,
    warn_if_counts,
)

__all__ = ["CategoricalEncoding", "CausalData", "TreatmentKind"]

#: How the treatment column is to be read.  ``"discrete"`` codes it into arms; there is
#: no third reading, and the two are not points on a scale -- they select different
#: mechanisms (a distribution over arms versus a conditional density) and therefore
#: different estimands.
TreatmentKind = Literal["discrete", "continuous"]

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
        label = weights_name or "weights"
        kind = resolve_weight_kind(weights_type, n)
        obs_weights = check_weights(weights, n, label)
        if weights is None:
            spec = WeightSpec(kind=kind, estimated=weights_estimated)
        else:
            warn_if_counts(np.asarray(weights, dtype=float), label)
            warn_if_concentrated(obs_weights, label)
            spec = WeightSpec(
                kind=kind,
                estimated=weights_estimated,
                name=label,
                scale=float(np.mean(np.asarray(weights, dtype=float))),
            )
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
        since ``1 - E_w[A]`` and ``E_w[1 - A]`` need not agree in the last bit.
        """
        share = self.treated_fraction  # raises on a continuous treatment, as it should
        if self.n_arms == 2:
            return np.array([1.0 - share, share])
        return np.array(
            [
                float(np.average(self.treatment == code, weights=self.weights))
                for code in self.arm_codes
            ]
        )

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

        With two arms this is a single column holding the 0/1 code itself, which is
        exactly the design the binary estimator has always used -- so a two-arm fit
        is unchanged, bit for bit.  ``tests/unit/test_causal_data.py`` asserts that
        equality rather than leaving it to be read off this docstring.

        With more than two arms a *single numeric column* would be wrong, not merely
        crude: it would impose a linear dose-response on the outcome regression,
        forcing :math:`\bar Q(2, W) - \bar Q(1, W) = \bar Q(1, W) - \bar Q(0, W)` for
        any learner linear in its design, and so shrink the very contrasts the fit
        exists to estimate.  Indicators leave the arms unconstrained.

        The first arm is dropped rather than one-hot encoding all ``K``, so an
        unregularised model with an intercept has a full-rank design.  Which arm is
        dropped is a property of the design only and does not privilege any arm in the
        estimand: the counterfactual means are all evaluated by prediction, and the
        reference used for *contrasts* is a separate, caller-chosen thing.

        For a **continuous** treatment this is the single numeric column itself.  The
        objection above -- that one column imposes a linear dose-response -- is not
        answerable by indicators here, because there are no arms to indicate; it is
        answered by the learner instead.  That is why the default library's splines and
        boosting matter more for a continuous treatment than for an arm-coded one, and
        why ``library="glm"`` on a continuous dose really does fit a straight line in the
        exposure.
        """
        c = np.asarray(codes, dtype=float).reshape(-1)
        if self.is_continuous_treatment:
            return c.reshape(-1, 1)
        return np.column_stack([(c == level).astype(float) for level in self.arm_codes[1:]])

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
        longitudinal analysis; see ``docs/roadmap.md``.

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
        if idx.size < _MIN_OBSERVATIONS:
            raise DataError(f"subset has {idx.size} rows; need at least {_MIN_OBSERVATIONS}")
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
