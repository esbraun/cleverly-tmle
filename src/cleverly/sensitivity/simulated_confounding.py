"""Qualitative stress surfaces for a simulated unobserved common cause.

The surface perturbs treatment and outcome through one shared latent vector, then
refits the complete estimator at each declared strength pair. It describes movement
of the fitted point estimate. It does not provide a bound, corrected estimate, test,
or sensitivity-adjusted interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Real
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..exceptions import CapabilityError
from ..utils.frames import emit_frame
from ..utils.random import resolve_assessment_seed
from ..utils.text import format_table
from ..validation.simulation import ReplicationFailure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = [
    "ConfounderStrengthGrid",
    "ObservedConfounderCalibration",
    "SimulatedConfoundingCell",
    "SimulatedConfoundingResult",
    "simulated_confounding",
]


@dataclass(frozen=True)
class ConfounderStrengthGrid:
    """Declare treatment and outcome perturbation strengths.

    Parameters
    ----------
    treatment : tuple of float
        Treatment perturbation strengths. Continuous-treatment values are signed linear
        coefficients. Binary-treatment values are checked as flip probabilities from zero
        through 0.5 when the surface runs.
    outcome : tuple of float
        Outcome perturbation strengths. Gaussian strengths are signed. Binomial
        strengths are checked as flip probabilities when the surface runs.

    See Also
    --------
    simulated_confounding : Evaluate the declared strength pairs.
    """

    treatment: tuple[float, ...]
    outcome: tuple[float, ...]

    def __post_init__(self) -> None:
        treatment = _numeric_strengths(self.treatment, "treatment")
        outcome = _numeric_strengths(self.outcome, "outcome")
        if 0.0 not in treatment or 0.0 not in outcome:
            raise ValueError("treatment and outcome strengths must each contain zero")
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)


@dataclass(frozen=True)
class ObservedConfounderCalibration:
    """Record one model-dependent observed-covariate calibration.

    Parameters
    ----------
    covariate : str
        Numeric adjustment covariate used for the comparison.
    role : {"treatment", "outcome"}
        Perturbed variable whose source-style scale is reported.
    family : {"binomial", "gaussian"}
        Variable family that determines the calibration rule.
    strength : float
        Prediction-change fraction or signed standardized marginal coefficient.
    method : str
        Finite-sample rule used to calculate ``strength``.
    """

    covariate: str
    role: Literal["treatment", "outcome"]
    family: Literal["binomial", "gaussian"]
    strength: float
    method: str


@dataclass(frozen=True)
class SimulatedConfoundingCell:
    """Store one strength pair's estimate or structured failure.

    Parameters
    ----------
    treatment_strength : float
        Declared binary flip probability or continuous linear coefficient.
    outcome_strength : float
        Declared outcome perturbation strength.
    estimate : float or None
        Refitted estimate. ``None`` when the cell failed.
    displacement : float or None
        Estimate minus the original estimate. ``None`` when the cell failed.
    induced_treatment_association : float or None
        Pearson correlation between the shared latent vector and the treatment of this
        cell, measured on the analysis data. The anchor cell reports the original
        treatment, which gives the null level of the same data. Every other cell reports
        the perturbed treatment. The value is ``None`` when that treatment has zero
        standard deviation, and when a cell failed before the surface built it.
    failure : ReplicationFailure or None
        Structured refit or replacement failure retained for this cell.
    """

    treatment_strength: float
    outcome_strength: float
    estimate: float | None
    displacement: float | None
    induced_treatment_association: float | None = None
    failure: ReplicationFailure | None = None


@dataclass(frozen=True)
class SimulatedConfoundingResult:
    """Store a qualitative simulated-confounding stress surface.

    Parameters
    ----------
    estimand : str
        Marginal binary-treatment ATE alias, named modified-policy mean alias, or named
        modified-policy contrast alias.
    original_estimate : float
        Point estimate from the unperturbed fitted result.
    grid : ConfounderStrengthGrid
        Explicit strength pairs evaluated by the surface.
    cells : tuple of SimulatedConfoundingCell
        Successful estimates and retained failures in grid order.
    calibrations : tuple of ObservedConfounderCalibration
        Optional source-style comparisons against numeric observed covariates.
    root_seed : int
        Seed resolved for the complete operation.
    latent_seed : int
        Tagged child seed used to draw the shared latent vector.
    refit_seed : int
        Seed reused by every estimator refit. It equals ``root_seed``. A cell that leaves
        the data unchanged then reproduces the zero-strength anchor exactly, provided the
        root seed equals the seed of the original fit.
    treatment_family : str
        Treatment family covered by the perturbation law.
    outcome_family : str
        Original outcome family.
    treatment_law : str
        Human-readable treatment perturbation rule.
    outcome_law : str
        Human-readable outcome perturbation rule.
    backend : str or None
        Dataframe backend used by frame methods.

    See Also
    --------
    simulated_confounding : Build this result from a fitted estimator.
    """

    estimand: str
    original_estimate: float
    grid: ConfounderStrengthGrid
    cells: tuple[SimulatedConfoundingCell, ...]
    calibrations: tuple[ObservedConfounderCalibration, ...]
    root_seed: int
    latent_seed: int
    refit_seed: int
    treatment_family: str
    outcome_family: str
    treatment_law: str
    outcome_law: str
    backend: str | None = None

    @property
    def complete(self) -> bool:
        """Return whether every declared cell produced an estimate."""
        return all(cell.failure is None for cell in self.cells)

    @property
    def failures(self) -> tuple[ReplicationFailure, ...]:
        """Return structured failures in grid order."""
        return tuple(cell.failure for cell in self.cells if cell.failure is not None)

    @property
    def successful_cells(self) -> tuple[SimulatedConfoundingCell, ...]:
        """Return cells that produced estimates in grid order."""
        return tuple(cell for cell in self.cells if cell.failure is None)

    def to_frame(self) -> Any:
        """Return one backend-native row per declared strength pair.

        Returns
        -------
        dataframe
            Estimates, displacements, induced treatment associations, and retained
            failure details.
        """
        payload = {
            "treatment_strength": [cell.treatment_strength for cell in self.cells],
            "outcome_strength": [cell.outcome_strength for cell in self.cells],
            "estimate": [cell.estimate for cell in self.cells],
            "displacement": [cell.displacement for cell in self.cells],
            "induced_treatment_association": [
                cell.induced_treatment_association for cell in self.cells
            ],
            "error_type": [
                None if cell.failure is None else cell.failure.error_type for cell in self.cells
            ],
            "message": [
                None if cell.failure is None else cell.failure.message for cell in self.cells
            ],
        }
        return emit_frame(payload, backend=self.backend)

    def calibration_frame(self) -> Any:
        """Return model-dependent observed-covariate comparisons.

        Returns
        -------
        dataframe
            One row per covariate and perturbed-variable role.
        """
        payload = {
            "covariate": [row.covariate for row in self.calibrations],
            "role": [row.role for row in self.calibrations],
            "family": [row.family for row in self.calibrations],
            "strength": [row.strength for row in self.calibrations],
            "method": [row.method for row in self.calibrations],
        }
        return emit_frame(payload, backend=self.backend)

    def summary(self) -> str:
        """Return a qualitative text summary without an inferential verdict.

        Returns
        -------
        str
            Original estimate, seeds, perturbation laws, cell movements, and the
            induced treatment association of each cell.
        """
        rows = []
        for cell in self.cells:
            association = cell.induced_treatment_association
            reported = "n/a" if association is None else f"{association:+.4f}"
            if cell.failure is None:
                rows.append(
                    [
                        f"{cell.treatment_strength:.4g}",
                        f"{cell.outcome_strength:.4g}",
                        f"{cell.estimate:.5g}",
                        f"{cell.displacement:+.5g}",
                        reported,
                    ]
                )
            else:
                rows.append(
                    [
                        f"{cell.treatment_strength:.4g}",
                        f"{cell.outcome_strength:.4g}",
                        "failed",
                        cell.failure.error_type,
                        reported,
                    ]
                )
        return "\n".join(
            [
                f"Simulated common-cause stress surface for {self.estimand!r}",
                f"original estimate: {self.original_estimate:.5g}",
                (
                    f"seeds: root={self.root_seed}, latent={self.latent_seed}, "
                    f"refit={self.refit_seed}"
                ),
                self.treatment_law,
                self.outcome_law,
                "",
                format_table(
                    [
                        "treatment strength",
                        "outcome strength",
                        "estimate",
                        "movement",
                        "induced association",
                    ],
                    rows,
                ),
                "",
                "The induced association is the correlation between the latent vector and the "
                "treatment of the cell.",
                *self._reading_guard(),
                "This surface is qualitative. It is not a bound or sensitivity-adjusted inference.",
            ]
        )

    def _reading_guard(self) -> tuple[str, ...]:
        """Return the lines that say which cells carry a confounding path."""
        if self.treatment_family == "binary":
            return (
                "An association near zero says the treatment axis moved the estimate by "
                "misclassification and not by confounding.",
            )
        outcome_axis = (
            "Its movement reports the outcome level shift alone. A policy mean keeps that "
            "level shift, and a contrast removes most of it. A small residual stays, "
            "because each cell refits the outcome regression."
            if self.outcome_family == "gaussian"
            else "Its movement reports the outcome perturbation alone. The tail "
            "perturbation attenuates the fitted outcome regression. A policy mean and a "
            "contrast both move with that attenuation."
        )
        return (
            "The association reports what the continuous linear perturbation achieved on "
            "these data. A cell whose outcome strength is zero has no confounding path, "
            "whatever its association, and reports dose perturbation alone.",
            f"A cell whose treatment strength is zero also has no confounding path. {outcome_axis}",
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _numeric_strengths(values: Any, role: str) -> tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError(f"{role} strengths must be a non-empty sequence") from error
    if not raw:
        raise ValueError(f"{role} strengths must not be empty")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw):
        raise TypeError(f"{role} strengths must contain only numeric values")
    strengths = tuple(float(value) for value in raw)
    if not all(np.isfinite(strengths)):
        raise ValueError(f"{role} strengths must be finite")
    if len(set(strengths)) != len(strengths):
        raise ValueError(f"{role} strengths must not contain duplicates")
    return strengths


@dataclass(frozen=True)
class _ValidatedRequest:
    estimator: Any
    calibration_names: tuple[str, ...]
    treatment_family: Literal["binary", "continuous"]


def _validate_continuous_policy_state(
    result: Any,
    estimand: str,
    key: Any,
    identified: Any,
    functional: Any,
    estimator: Any,
) -> None:
    """Require one coherent modified-policy request across every stored layer."""
    from ..interventions.shift import Shift
    from ..study import ModifiedTreatmentPolicy, ModifiedTreatmentPolicyEffect
    from ..targets import TARGETS
    from ..targets.base import parameter_name

    target = key.estimand
    typed_estimand = identified.estimand
    typed_type = {
        "ey_shift": ModifiedTreatmentPolicy,
        "ate_shift": ModifiedTreatmentPolicyEffect,
    }.get(target)
    if typed_type is None or type(typed_estimand) is not typed_type:
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent registered modified-policy "
            "identification provenance"
        )
    typed_state: Any = typed_estimand
    typed_policies = tuple(typed_state.shifts)
    typed_reference = typed_state.reference

    declared_policies = tuple(functional.interventions)
    replay_policies = tuple(estimator.shifts)
    if any(type(shift) is not Shift for shift in (*declared_policies, *replay_policies)):
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent structured shift metadata"
        )
    declared_names = tuple(shift.name for shift in declared_policies)
    declared_deltas = tuple(float(shift.delta) for shift in declared_policies)
    declared_reference = declared_names[0] if functional.reference is None else functional.reference
    fitted_shifts = result.nuisance.shifts
    fitted_names = () if fitted_shifts is None else tuple(fitted_shifts.names)
    fitted_deltas = () if fitted_shifts is None else tuple(fitted_shifts.deltas)
    fitted_reference = None if fitted_shifts is None else fitted_names[int(fitted_shifts.reference)]
    expected_alias = parameter_name(
        target,
        arm=key.value,
        versus=key.reference if target == "ate_shift" else None,
    )
    expected_shifted = np.column_stack(
        [shift.apply(result.data.treatment)[0] for shift in declared_policies]
    )
    expected_capped = np.column_stack(
        [shift.apply(result.data.treatment)[1] for shift in declared_policies]
    )
    expected_reference = fitted_reference if target == "ate_shift" else None
    if (
        fitted_shifts is None
        or typed_policies != declared_policies
        or typed_reference != functional.reference
        or replay_policies != declared_policies
        or estimator.reference != functional.reference
        or fitted_names != declared_names
        or fitted_deltas != declared_deltas
        or fitted_reference != declared_reference
        or not np.array_equal(fitted_shifts.shifted, expected_shifted)
        or not np.array_equal(fitted_shifts.capped, expected_capped)
        or key.alias != estimand
        or key.value not in fitted_names
        or key.reference != expected_reference
        or expected_alias != estimand
    ):
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent structured shift metadata"
        )

    registered = TARGETS.get(target)
    declared = getattr(typed_estimand, "name", None)
    if (
        functional.target != target
        or declared != target
        or registered is None
        or identified.identification != registered.identification
    ):
        raise CapabilityError(
            "continuous simulated_confounding found inconsistent registered modified-policy "
            "identification provenance"
        )

    # A zero-delta shift maps every dose to itself, so its policy mean is E[Y] and its
    # counterfactual treatment has no dependence on the dose a common cause would move.
    # The treatment axis of such a surface is identically zero, and the outcome axis
    # reports the level shift ``Y' = Y - k_Y U`` alone.  An ``ate_shift`` contrast that
    # uses the same policy as its reference keeps treatment dependence, so it stays.
    if target == "ey_shift":
        selected = declared_policies[declared_names.index(key.value)]
        if float(selected.delta) == 0.0:
            raise CapabilityError(
                f"continuous simulated_confounding refuses the policy mean {estimand!r}; a "
                "zero-delta policy is the natural course, its mean is E[Y], and it carries no "
                "counterfactual treatment dependence for a simulated common cause to move. "
                "Select a nonzero-delta ey_shift[...] mean, or an ate_shift[...] contrast "
                "that uses the natural course as its reference"
            )


def _zero_delta_policy_means(functional: Any) -> frozenset[str]:
    """Name every ``ey_shift`` alias whose policy is the zero-delta natural course.

    Parameters
    ----------
    functional : BackdoorMeanContrast
        Identification functional of the fitted result.

    Returns
    -------
    frozenset of str
        Aliases the selection message must not advertise, because each one names a
        policy mean the surface refuses.
    """
    aliases = set()
    for policy in functional.interventions:
        delta = getattr(policy, "delta", None)
        name = getattr(policy, "name", None)
        if isinstance(name, str) and isinstance(delta, Real) and float(delta) == 0.0:
            aliases.add(f"ey_shift[{name}]")
    return frozenset(aliases)


def _validate_request(
    result: Any,
    estimand: str,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: Any,
) -> _ValidatedRequest:
    """Validate the complete supported boundary before a refit or random draw."""
    from ..estimators.base import TMLEResult
    from ..estimators.ctmle import CTMLE
    from ..estimators.drtmle import DRTMLE
    from ..estimators.tmle import TMLE
    from ..study import (
        BackdoorMeanContrast,
        ExplicitAdjustmentProvider,
        ParameterKey,
    )
    from ..targets import TARGETS

    if type(result) is not TMLEResult:
        raise CapabilityError(
            "simulated_confounding supports point-treatment TMLEResult objects only; "
            "no longitudinal perturbation law is implemented"
        )
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError(
            "simulated_confounding needs a replayable fitted estimator; this restored or "
            "legacy result has no estimator configuration"
        )
    data = result.data
    if data.is_continuous_treatment:
        treatment_family: Literal["binary", "continuous"] = "continuous"
    elif data.is_binary_treatment:
        treatment_family = "binary"
    else:
        raise CapabilityError(
            "simulated_confounding has no category-valued perturbation law for a multi-arm "
            "treatment"
        )
    if treatment_family == "binary" and type(estimator) not in {TMLE, CTMLE, DRTMLE}:
        raise CapabilityError(
            "simulated_confounding supports ordinary TMLE, collaborative TMLE, and "
            f"complete-outcome DR-TMLE; got {type(estimator).__name__}"
        )
    if treatment_family == "continuous" and type(estimator) is not TMLE:
        raise CapabilityError(
            "continuous simulated_confounding supports exact ordinary TMLE only; "
            f"got {type(estimator).__name__}"
        )
    if treatment_family == "binary" and any(value < 0.0 or value > 0.5 for value in grid.treatment):
        raise ValueError("binary treatment strengths must be between 0 and 0.5")
    if data.family not in {"gaussian", "binomial"}:
        raise CapabilityError(
            f"simulated_confounding has no perturbation law for outcome family={data.family!r}"
        )
    if data.family == "binomial" and any(value < 0.0 or value > 0.5 for value in grid.outcome):
        raise ValueError("binomial outcome strengths must be between 0 and 0.5")
    if data.has_missing_outcome:
        raise CapabilityError(
            "simulated_confounding has no missing-outcome perturbation and observation refit"
        )
    if data.has_intermediate or result.intermediate_value is not None:
        raise CapabilityError(
            "simulated_confounding has no controlled-direct-effect or intermediate-variable law"
        )
    if data.weights_name is not None or data.is_weighted:
        raise CapabilityError(
            "simulated_confounding does not support observation-weighted target populations"
        )
    if data.cluster is not None:
        raise CapabilityError("simulated_confounding does not support clustered fits")
    if result.n_repeats > 1:
        raise CapabilityError(
            "simulated_confounding does not support repeated cross-fitting; fit one split"
        )

    identified = result.identified_effect
    if identified is None:
        raise CapabilityError(
            "simulated_confounding needs identification metadata for a backdoor contrast; "
            "this legacy fit records none"
        )
    functional = identified.functional
    if type(functional) is not BackdoorMeanContrast:
        raise CapabilityError(
            "simulated_confounding supports a backdoor-identified marginal ATE; "
            f"got {type(functional).__name__}"
        )
    provider = getattr(identified, "provider", None)
    if type(provider) is not ExplicitAdjustmentProvider:
        raise CapabilityError(
            "simulated_confounding needs registered explicit-adjustment backdoor provenance"
        )
    if treatment_family == "continuous" and estimand == "ate":
        vacuous = _zero_delta_policy_means(functional)
        admissible = [
            name
            for name in result.estimates
            if name.startswith(("ey_shift[", "ate_shift[")) and name not in vacuous
        ]
        detail = f"choose one of {admissible}" if admissible else "this fit reports none"
        raise ValueError(
            "continuous simulated_confounding requires an explicit ey_shift[...] policy mean "
            f"or ate_shift[...] contrast alias; {detail}"
        )
    if estimand not in result.estimates:
        # A continuous fit can report the zero-delta natural-course mean, which the next
        # call refuses.  Advertising it here hands the caller a refused alias.
        vacuous = (
            _zero_delta_policy_means(functional)
            if treatment_family == "continuous"
            else frozenset()
        )
        admissible = [name for name in result.estimates if name not in vacuous]
        detail = f"choose one of {admissible}" if admissible else "this fit reports none"
        raise ValueError(f"estimand {estimand!r} is unavailable; {detail}")
    key = result.parameter_keys.get(estimand)
    if type(key) is not ParameterKey:
        raise CapabilityError(
            f"simulated_confounding needs a structured parameter key for {estimand!r}"
        )
    declared = getattr(getattr(identified, "estimand", None), "name", None)
    if treatment_family == "binary":
        if (
            functional.longitudinal
            or functional.axis != "arm"
            or functional.interventions
            or functional.msm is not None
        ):
            raise CapabilityError(
                "simulated_confounding supports a marginal arm-indexed ATE, not a regimen, "
                "stochastic, incremental, modified-policy, or MSM parameter"
            )
        if key.estimand != "ate" or key.axis != "arm" or key.stratum is not None:
            raise CapabilityError(
                "simulated_confounding supports only a marginal ATE; ATT, ATC, means, ratios, "
                "conditional strata, and other parameters are outside its source boundary"
            )
        registered = TARGETS.get("ate")
        if (
            functional.target != "ate"
            or declared != "ate"
            or registered is None
            or identified.identification != registered.identification
        ):
            raise CapabilityError(
                "simulated_confounding found inconsistent registered ATE identification provenance"
            )
    else:
        if (
            functional.longitudinal
            or functional.axis != "shift"
            or not functional.interventions
            or functional.msm is not None
        ):
            raise CapabilityError(
                "continuous simulated_confounding supports a marginal modified-treatment-policy "
                "parameter, not an arm, regimen, stochastic, incremental, or MSM parameter"
            )
        if (
            key.estimand not in {"ey_shift", "ate_shift"}
            or key.axis != "shift"
            or key.stratum is not None
        ):
            raise CapabilityError(
                "continuous simulated_confounding supports only a marginal ey_shift policy mean "
                "or ate_shift contrast; conditional strata and other parameters are outside its "
                "source boundary"
            )
        _validate_continuous_policy_state(result, estimand, key, identified, functional, estimator)

    names = tuple(
        [benchmark_covariates] if isinstance(benchmark_covariates, str) else benchmark_covariates
    )
    if len(set(names)) != len(names):
        raise ValueError("benchmark_covariates contains duplicates")
    categorical = {
        name for encoding in data.encodings for name in (encoding.column, *encoding.generated)
    }
    for name in names:
        if not isinstance(name, str):
            raise TypeError("benchmark_covariates must contain only column names")
        if name in categorical:
            raise CapabilityError(
                f"simulated_confounding cannot calibrate categorical covariate {name!r}; "
                "zeroing one encoded column does not define a logical-covariate benchmark"
            )
        if name not in data.covariate_names:
            raise ValueError(
                f"benchmark covariate {name!r} is unavailable; numeric adjustment columns are "
                f"{[name for name in data.covariate_names if name not in categorical]}"
            )
        column = data.covariates[:, data.covariate_names.index(name)]
        if float(np.std(column)) == 0.0:
            raise CapabilityError(
                f"simulated_confounding cannot calibrate constant covariate {name!r}"
            )
    return _ValidatedRequest(estimator, names, treatment_family)


_LATENT_SEED_TAG = 3


def _latent_child_seed(root_seed: int) -> int:
    """Derive the latent-draw seed from a tagged child of the root seed.

    The tag keeps this stream apart from every other child stream built on the same
    root. A plain ``SeedSequence(root_seed).spawn(...)`` reproduces the bootstrap
    replicate seeds in :mod:`cleverly.inference.bootstrap`, and the tagged form in
    :func:`cleverly.validation.refute._generated_child_seeds` uses tags one and two.

    Parameters
    ----------
    root_seed : int
        Seed resolved for the complete operation.

    Returns
    -------
    int
        Seed for the shared latent vector.
    """
    sequence = np.random.SeedSequence([root_seed, _LATENT_SEED_TAG])
    return int(sequence.generate_state(1)[0])


def _flip_mask(latent: np.ndarray[Any, Any], strength: float) -> np.ndarray[Any, Any]:
    if strength == 0.0:
        return np.zeros(latent.shape, dtype=bool)
    threshold = -NormalDist().inv_cdf(strength)
    return latent >= threshold


def _flip_binary(values: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.where(mask, 1.0 - values, values)


def _linear_treatment(
    values: np.ndarray[Any, Any], latent: np.ndarray[Any, Any], strength: float
) -> np.ndarray[Any, Any]:
    """Apply the source-backed continuous treatment perturbation ``A' = A + k_A U``."""
    return values + strength * latent


def _perturb_treatment(
    values: np.ndarray[Any, Any],
    latent: np.ndarray[Any, Any],
    strength: float,
    family: Literal["binary", "continuous"],
) -> np.ndarray[Any, Any]:
    if family == "continuous":
        return _linear_treatment(values, latent, strength)
    return _flip_binary(values, _flip_mask(latent, strength))


def _treatment_association(
    latent: np.ndarray[Any, Any], treatment: np.ndarray[Any, Any]
) -> float | None:
    """Report the realised correlation between the latent vector and one treatment.

    For a binary tail flip, the association depends on the treated fraction. A balanced
    design reaches zero association, where the treatment axis moves the estimate by
    misclassification alone. For a continuous linear perturbation, this is the achieved
    association between ``U`` and ``A'``.

    Parameters
    ----------
    latent : ndarray
        Shared latent vector drawn for the complete surface.
    treatment : ndarray
        Original or perturbed treatment of one cell.

    Returns
    -------
    float or None
        Pearson correlation, or ``None`` when the treatment has zero standard deviation.
    """
    if float(np.std(treatment)) == 0.0:
        return None
    return float(np.corrcoef(latent, treatment)[0, 1])


def _gaussian_outcome(
    values: np.ndarray[Any, Any], latent: np.ndarray[Any, Any], strength: float
) -> np.ndarray[Any, Any]:
    return values - strength * latent


def _binary_calibration(
    design: np.ndarray[Any, Any], target: np.ndarray[Any, Any], index: int
) -> float:
    model = LogisticRegression(max_iter=1000)
    model.fit(design, target)
    baseline = model.predict(design)
    removed = design.copy()
    removed[:, index] = 0.0
    return float(np.mean(model.predict(removed) != baseline))


def _continuous_calibration(covariate: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> float:
    return float(np.corrcoef(covariate, target)[0, 1] * np.std(target))


def _calibrate(result: Any, names: tuple[str, ...]) -> tuple[ObservedConfounderCalibration, ...]:
    if not names:
        return ()
    data = result.data
    design = StandardScaler().fit_transform(data.covariates)
    rows: list[ObservedConfounderCalibration] = []
    for name in names:
        index = data.covariate_names.index(name)
        if data.is_continuous_treatment:
            treatment_strength = _continuous_calibration(data.covariates[:, index], data.treatment)
            treatment_family: Literal["binomial", "gaussian"] = "gaussian"
            treatment_method = "signed standardized marginal coefficient"
        else:
            treatment_strength = _binary_calibration(design, data.treatment, index)
            treatment_family = "binomial"
            treatment_method = "logistic class-prediction change fraction"
        rows.append(
            ObservedConfounderCalibration(
                covariate=name,
                role="treatment",
                family=treatment_family,
                strength=treatment_strength,
                method=treatment_method,
            )
        )
        if data.family == "binomial":
            outcome_strength = _binary_calibration(design, data.outcome, index)
            method = "logistic class-prediction change fraction"
        else:
            outcome_strength = _continuous_calibration(data.covariates[:, index], data.outcome)
            method = "signed standardized marginal coefficient"
        rows.append(
            ObservedConfounderCalibration(
                covariate=name,
                role="outcome",
                family=data.family,
                strength=outcome_strength,
                method=method,
            )
        )
    return tuple(rows)


def simulated_confounding(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    grid: ConfounderStrengthGrid,
    benchmark_covariates: tuple[str, ...] = (),
    random_state: int | None = None,
) -> SimulatedConfoundingResult:
    """Refit an additive parameter across a simulated common-cause strength grid.

    Parameters
    ----------
    result : TMLEResult
        Replayable backdoor-identified marginal binary-treatment ATE, continuous
        modified-treatment-policy mean, or continuous modified-policy effect fit.
    estimand : str
        Additive parameter alias to report. A continuous fit requires an explicit
        ``ey_shift[...]`` alias of a nonzero-delta policy, or an ``ate_shift[...]`` alias.
    grid : ConfounderStrengthGrid
        Explicit treatment and outcome perturbation strengths.
    benchmark_covariates : tuple of str
        Numeric observed covariates for optional model-dependent calibration.
    random_state : int or None
        Root seed. ``None`` uses the fitted estimator's seed or draws a recorded seed.

    Returns
    -------
    SimulatedConfoundingResult
        Point-estimate movements, the induced treatment association of each cell, and
        retained cell failures. The result has no verdict or sensitivity-adjusted
        inference.

    See Also
    --------
    ConfounderStrengthGrid : The explicit strength declaration.
    cleverly.sensitivity.omitted_variable_bounds : A non-refit bias-bound analysis.

    Notes
    -----
    Every cell refits under the resolved root seed, and the zero-strength anchor is the
    original fit itself. The two agree on the cross-fitting folds when the root seed
    equals the seed of the original fit, which is what ``random_state=None`` resolves to.
    When the original fit declared no ``random_state``, the surface cannot reproduce the
    folds of that fit, so movement near the anchor can still carry a fold artifact. An
    explicit ``random_state`` other than the seed of the fit has the same effect.

    Each cell reports ``induced_treatment_association``. It is the correlation between the
    shared latent vector and the treatment of that cell. For binary treatment, the flip is
    non-differential misclassification. Its induced association depends on the treated
    fraction. A balanced design reports an association near zero. For continuous
    treatment, the operation applies ``A' = A + k_A U`` and reports the correlation that
    linear perturbation achieves on the analysis data. That association grows with the
    treatment strength by construction, so it is not on its own evidence of confounding.
    A confounding path also needs the latent vector in the outcome, which happens only at
    a nonzero outcome strength. A cell whose outcome strength is zero therefore has no
    confounding path, whatever its association, and its movement reports dose perturbation
    alone.

    The zero treatment-strength column carries no confounding path either, because the
    latent vector never reaches the treatment there. For a Gaussian outcome that column
    reports the level shift ``Y' = Y - k_Y U`` alone. The level shift largely cancels in
    an ``ate_shift`` contrast. A policy mean keeps it, so read the zero treatment-strength
    column of an ``ey_shift`` surface as an artifact of the outcome law.

    A zero-delta shift is the natural course, its policy mean is ``E[Y]``, and it has no
    counterfactual treatment dependence. ``simulated_confounding`` refuses that mean before
    it draws the latent vector. It still accepts an ``ate_shift`` contrast that uses the
    natural course as its reference.
    """
    if type(grid) is not ConfounderStrengthGrid:
        raise TypeError("grid must be an exact ConfounderStrengthGrid declaration")
    request = _validate_request(result, estimand, grid, benchmark_covariates)
    calibrations = _calibrate(result, request.calibration_names)
    root_seed = resolve_assessment_seed(result, random_state)
    # Every cell refits under the root seed, not a spawned child.  The zero-strength
    # anchor is the original fit, which ran under the estimator's own ``random_state``;
    # a child seed would give every other cell different cross-fitting folds and charge
    # the fold change to the perturbation.  ``TMLE.refit`` returns the original fit
    # unchanged when the seed it gets equals the estimator's own, so an unperturbed cell
    # reproduces the anchor exactly.  Same convention as ``cleverly.validation.refute``
    # and ``cleverly.sensitivity.omitted_variable``.  The latent draw keeps its own
    # tagged child seed, so the shared latent vector stays independent of the folds.
    refit_seed = root_seed
    latent_seed = _latent_child_seed(root_seed)
    latent = np.random.default_rng(latent_seed).normal(size=result.data.n)
    original = float(result[estimand].psi)
    cells: list[SimulatedConfoundingCell] = []

    for cell_index, (treatment_strength, outcome_strength) in enumerate(
        product(grid.treatment, grid.outcome)
    ):
        if treatment_strength == 0.0 and outcome_strength == 0.0:
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    estimate=original,
                    displacement=0.0,
                    induced_treatment_association=_treatment_association(
                        latent, result.data.treatment
                    ),
                )
            )
            continue
        # A cell reports the association of the treatment the surface built for it, even
        # when the cell later fails.  An arm-loss cell reaches the zero-variance guard and
        # reports ``None``, because a constant treatment has no correlation.
        association: float | None = None
        try:
            treatment = _perturb_treatment(
                result.data.treatment,
                latent,
                treatment_strength,
                request.treatment_family,
            )
            association = _treatment_association(latent, treatment)
            replacement = result.data.with_treatment(treatment)
            if result.data.family == "gaussian":
                outcome = _gaussian_outcome(result.data.outcome, latent, outcome_strength)
            else:
                outcome = _flip_binary(result.data.outcome, _flip_mask(latent, outcome_strength))
            replacement = replacement.with_outcome(
                outcome,
                family=result.data.family,
                name="simulated-confounding outcome",
            )
            refitted = request.estimator.refit(replacement, random_state=refit_seed)
            estimate = float(refitted[estimand].psi)
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    estimate=estimate,
                    displacement=estimate - original,
                    induced_treatment_association=association,
                )
            )
        except Exception as error:
            cells.append(
                SimulatedConfoundingCell(
                    treatment_strength=treatment_strength,
                    outcome_strength=outcome_strength,
                    estimate=None,
                    displacement=None,
                    induced_treatment_association=association,
                    failure=ReplicationFailure(
                        replicate=cell_index,
                        seed=root_seed,
                        error_type=type(error).__name__,
                        message=str(error),
                    ),
                )
            )

    return SimulatedConfoundingResult(
        estimand=estimand,
        original_estimate=original,
        grid=grid,
        cells=tuple(cells),
        calibrations=calibrations,
        root_seed=root_seed,
        latent_seed=latent_seed,
        refit_seed=refit_seed,
        treatment_family=request.treatment_family,
        outcome_family=result.data.family,
        treatment_law=(
            "Treatment is flipped in the declared upper latent-normal tail."
            if request.treatment_family == "binary"
            else "Continuous treatment adds signed strength times the shared latent value."
        ),
        outcome_law=(
            "Gaussian outcome subtracts signed strength times the shared latent value."
            if result.data.family == "gaussian"
            else "Binomial outcome is flipped in the declared upper latent-normal tail."
        ),
        backend=result.data.backend,
    )
