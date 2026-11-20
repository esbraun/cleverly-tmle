r"""Refutation of a continuous-time survival fit (X101 part (a)).

:func:`cleverly.validation.refute` routes a
:class:`~cleverly.survival.ContinuousSurvivalResult` here. Each test refits the estimator the
result stores, :meth:`~cleverly.survival.ContinuousSurvivalTMLE.refit`, on a perturbed copy of
the stored :class:`~cleverly.survival.data.ContinuousSurvivalData`:

- ``placebo`` permutes the recorded treatment among the units (``"permute"``, the default) or
  draws it from its empirical marginal (``"draw"``). Under that law the identified risk
  difference is exactly 0 and the risk ratio exactly 1, whether or not censoring is at random
  given the treatment there; the self-normalized ``t`` of the refits reads the mean of the
  placebo estimates against that null. An arm's risk has no fixed null and is refused.
- ``random_common_cause`` appends one standard-normal baseline column, which leaves the
  estimand unchanged; each refit keeps the fit's seed and so its folds.
- ``subset`` refits on ``min(n, max(20, int(fraction * n)))`` units drawn without replacement,
  each keeping its original outer cross-fitting label, so the scatter ratio ``rho`` reads the
  sampling spread alone. Every draw is checked against the engine's two event and follow-up
  guards before the first subset refit.

The readings are those of every family (:mod:`cleverly.validation._readings`), and their
records come from the builders the point loop uses. The placebo ``t`` and the subset ``rho`` on
a risk difference print a verdict inside the scope that ``refuter-calibration-continuous-survival``
measured (``VERDICT_EVIDENCE`` in :mod:`cleverly.validation._interpretation`); every other
reading is descriptive: ``passed`` is ``None``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from .._refusal import Refusal, RefusalReason
from ..exceptions import CapabilityError, CleverlyError
from ..inference.influence import spread_name
from ._readings import paired_shift, placebo_reading

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .refute import RefutationResult, _RefuteRequest

__all__ = ["continuous_survival_refute", "continuous_survival_rules"]

#: What runs on a continuous-time fit when a test is refused for want of an adapter.
_RUNS: Final = "The placebo, random_common_cause and subset tests run on this fit."

#: The roadmap item of the continuous-time refutations that have no adapter yet.
_ROADMAP: Final = "X105"

_NEGATIVE_CONTROL: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no negative-control adapter is implemented for a continuous-time survival fit: a "
    "control is a time-to-event pair with its own censoring and target times, and the refit "
    "needs a declaration of both.",
    alternative=_RUNS,
    roadmap=_ROADMAP,
)

_GENERATED: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no generated event-time process is declared for a continuous-time survival fit: the "
    "survival processes draw per-node indicators on a time grid, and this fit has none.",
    alternative=_RUNS,
    roadmap=_ROADMAP,
)

_RESAMPLING: Final = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no resampling refit is implemented for a continuous-time survival fit: a unit drawn "
    "twice must share a cross-fitting fold, and the engine draws its folds from the data. "
    "The full-refit bootstrap of X6 (d) needs the same fold rule.",
    alternative=_RUNS,
    roadmap=_ROADMAP,
)

_FUTURE_TREATMENT: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "the future_treatment test reads a later treatment node, and a continuous-time survival "
    "fit has one baseline treatment.",
)

#: The refusal of each test with no continuous-time adapter, by test name.
_TEST_REFUSALS: Final[dict[str, Refusal]] = {
    "negative_control_outcome": _NEGATIVE_CONTROL,
    "dummy_outcome": _GENERATED,
    "simulated_outcome": _GENERATED,
    "plasmode": _GENERATED,
    "bootstrap_stability": _RESAMPLING,
    "bootstrap_measurement_error": _RESAMPLING,
    "future_treatment": _FUTURE_TREATMENT,
}

_MARKOV: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "placebo_mode='markov' draws treatment trajectories of a longitudinal panel; a "
    "continuous-time survival fit has one baseline treatment, which the placebo permutes "
    "('permute') or draws ('draw').",
)

_NOISE_NODES: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "noise_nodes=True appends a noise column to every node of a longitudinal fit; a "
    "continuous-time survival fit has baseline covariates only. Leave noise_nodes=False: "
    "random_common_cause appends one baseline column.",
)

_MISSING: Final = Refusal(
    RefusalReason.INVALID_DECLARATION,
    "refute needs the fitted estimator and the data that produced the result",
)

#: The no-effect value of each contrast scale: a difference reads 0, a ratio reads 1 (on the
#: log scale).
_NULLS: Final[dict[str, float]] = {"difference": 0.0, "ratio": 1.0}


def _level_placebo(estimand: str) -> Refusal:
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"the placebo has no fixed null for {estimand!r}: under the placebo law both arms' "
        "risks equal one functional of the observed hazards, which is estimated from the same "
        "data and equals the marginal risk only when censoring is independent of the event "
        "time given the covariates alone.",
        alternative=(
            "The placebo runs on ate_regimen (null 0) and rr_regimen (null 1); "
            "random_common_cause and subset run on this estimand."
        ),
    )


def continuous_survival_rules(result: Any, request: _RefuteRequest) -> Refusal | None:
    """The first refusal a refutation request meets on a continuous-time fit, or ``None``.

    The rules run in one order: a result with no stored estimator or data, an estimand the
    fit did not report, a test with no continuous-time adapter (in request order), an option
    of a longitudinal panel, and the placebo on an estimate with no fixed null.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fit.
    request : _RefuteRequest
        The validated request.

    Returns
    -------
    Refusal or None
        The record :func:`~cleverly.validation.refute` raises and the ``refute`` capability
        row quotes, or ``None`` when every requested test runs.
    """
    if result.estimator is None or result.data is None:
        return _MISSING
    if request.estimand not in result.estimates:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            f"estimand {request.estimand!r} was not requested in this fit",
        )
    for name in request.tests:
        if name in _TEST_REFUSALS:
            return _TEST_REFUSALS[name]
    if "placebo" in request.tests and request.placebo_mode == "markov":
        return _MARKOV
    if "random_common_cause" in request.tests and request.noise_nodes:
        return _NOISE_NODES
    if "placebo" in request.tests and str(result[request.estimand].scale) not in _NULLS:
        return _level_placebo(request.estimand)
    return None


def _guard_subset(
    estimator: Any, data: Any, draws: list[np.ndarray], count: int
) -> CapabilityError | None:
    """The refusal of the first subset draw that fails one of the engine's two guards.

    The engine refuses a fit whose arm has no event of some cause by the first target time,
    or whose arm is followed only to before the last target time. Every draw is read here
    before the first subset refit, so a draw that would stop the test part way through
    refuses it first.
    """
    arm_labels, A = estimator._arms(data.treatment)
    declared = estimator.causes if estimator.causes is not None else data.causes
    causes = estimator._causes(data.event, declared)
    T = np.asarray(data.time, dtype=float)
    D = np.asarray(data.event, dtype=int)
    first, last = estimator.horizons[0], estimator.horizons[-1]
    alternative = "Raise subset_fraction, or drop subset from tests=."
    for k, rows in enumerate(draws, start=1):
        t, d, a_ = T[rows], D[rows], A[rows]
        for a in (0, 1):
            for code, label in causes.items():
                times = t[(a_ == a) & (d == code)]
                if times.size == 0 or times.min() > first:
                    return CapabilityError(
                        f"subset draw {k} of {count} leaves arm {arm_labels[a]!r} with no event "
                        f"of cause {label!r} by the first target time.",
                        reason=RefusalReason.INVALID_DECLARATION,
                        alternative=alternative,
                    )
        for a in (0, 1):
            followed = float(t[a_ == a].max())
            if followed < last:
                return CapabilityError(
                    f"subset draw {k} of {count} leaves arm {arm_labels[a]!r} followed only to "
                    f"{followed:g}, before the last target time.",
                    reason=RefusalReason.INVALID_DECLARATION,
                    alternative=alternative,
                )
    return None


def _frozen_folds(estimator: Any, rows: np.ndarray) -> np.ndarray:
    """The fit's outer label of each drawn row, compacted as the point subset does."""
    labels = np.asarray(estimator._outer_folds)[rows]
    if not estimator.cross_fit:
        return np.zeros(rows.size, dtype=int)
    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(int)


def _placebo_treatment(treatment: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    """The placebo treatment column: a permutation, or i.i.d. draws with both arms present."""
    if mode != "draw":
        return np.asarray(rng.permutation(treatment))
    arms = np.unique(treatment)
    for _ in range(1000):
        drawn = rng.choice(treatment, size=treatment.size, replace=True)
        if np.unique(drawn).size == arms.size:
            return np.asarray(drawn)
    raise RuntimeError(  # pragma: no cover - a 1000-fold failure needs an arm share near zero
        "the placebo draw dropped an arm of the treatment 1000 times"
    )


def continuous_survival_refute(
    result: Any,
    request: _RefuteRequest,
    *,
    seed: int,
    subset_fraction: float,
    seed_spread: bool,
) -> RefutationResult:
    """Run the admitted refutation tests of ``request`` on a continuous-time fit.

    :func:`continuous_survival_rules` has admitted the request, so every test named runs.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fit, with its stored estimator and data.
    request : _RefuteRequest
        The validated request.
    seed : int
        The root seed of the draws, and the seed of the placebo and subset refits.
    subset_fraction : float
        The share of units each subset refit reads.
    seed_spread : bool
        Whether ``random_common_cause`` also refits the data at fresh seeds and prints the
        spread of those refits.

    Returns
    -------
    RefutationResult
        One descriptive record per test.
    """
    from .refute import (
        _DEFAULT_NOISE_REPLICATES,
        _DEFAULT_PERTURBATION_REPLICATES,
        RefutationResult,
        _generated_child_seeds,
        _no_effect_difference,
        _noise_record,
        _on_draw,
        _placebo_detail,
        _placebo_record,
        _read_configuration,
        _resolve,
        _subset_record,
        _subset_scatter,
    )

    estimator = result.estimator
    data = result.data
    estimand = request.estimand
    estimate = result[estimand]
    original = float(estimate.psi)
    is_ratio = str(estimate.scale) == "ratio"
    std_error = float(estimate.plugin_std_error)
    status = estimate.inference
    unit = spread_name("standard errors", status)
    fit = _read_configuration(result, estimand)
    mean_kind: Literal["arithmetic", "geometric"] = "geometric" if is_ratio else "arithmetic"
    rng = np.random.default_rng(seed)
    fit_seed = int(estimator.random_state)
    n = int(data.n)
    outcomes = []

    def refit(
        replacement: Any, label: str, *, at: int = seed, folds: Any = None
    ) -> tuple[float, float]:
        try:
            refitted = estimator._for_estimate_refits().refit(
                replacement, random_state=at, _frozen_folds=folds
            )
        except CleverlyError as error:
            raise _on_draw(error, label) from error
        value = refitted[estimand]
        return float(value.psi), float(value.plugin_std_error)

    def on_scale(value: float) -> float:
        return float(np.log(value)) if is_ratio else float(value)

    def replicates(name: str) -> int:
        if request.n_replicates is not None:
            return request.n_replicates
        if name == "random_common_cause":
            return _DEFAULT_NOISE_REPLICATES
        return _DEFAULT_PERTURBATION_REPLICATES

    # Draw and check the subsets before any refit, so a draw the engine cannot fit is refused
    # before the placebo and noise refits spend their time.
    subset_draws: list[np.ndarray] = []
    subset_size = 0
    if "subset" in request.tests:
        subset_count = replicates("subset")
        subset_size = min(n, max(20, int(subset_fraction * n)))
        subset_draws = [
            np.sort(rng.choice(n, size=subset_size, replace=False)) for _ in range(subset_count)
        ]
        refused = _guard_subset(estimator, data, subset_draws, subset_count)
        if refused is not None:
            raise refused

    for name in request.tests:
        count = replicates(name)
        if name == "placebo":
            null = _NULLS[str(estimate.scale)]
            mode = request.placebo_mode or "permute"
            treatment = np.asarray(data.treatment)
            fits = []
            for index in range(1, count + 1):
                placebo = _placebo_treatment(treatment, rng, mode)
                fits.append(
                    refit(
                        replace(data, treatment=placebo),
                        f"on placebo refit {index} of {count}",
                    )
                )
            values = [psi for psi, _ in fits]
            reading = (
                None
                if count < 2
                else placebo_reading(
                    [_no_effect_difference(psi, null) for psi in values],
                    std_error=std_error,
                    plugin_std_errors=[se for _, se in fits],
                    scale=original,
                )
            )
            mean = float(np.exp(np.mean(np.log(values)))) if is_ratio else float(np.mean(values))
            detail = _placebo_detail(mean_kind, mean, null, reading, count)
            plain_length = len(detail)
            detail += (
                ". Each refit reruns the continuous-time engine with the treatment column "
                + (
                    "drawn from its empirical marginal"
                    if mode == "draw"
                    else "permuted among the units"
                )
            )
            _, coded = estimator._arms(treatment)
            treated = int(coded.sum())
            if mode == "permute" and 2 * treated != n:
                detail += (
                    f". The arms differ in size ({n - treated} and {treated}), so the mean "
                    "over permutations carries a bias of order 1/n"
                )
            outcomes.append(
                _placebo_record(
                    estimand=estimand,
                    original=original,
                    fits=fits,
                    null=null,
                    reading=reading,
                    resolution=_resolve("placebo", "t", fit, count, mode),
                    mean_kind=mean_kind,
                    detail=detail,
                    plain_length=plain_length,
                    withheld="" if reading is not None else "one shuffled copy has no spread",
                    status=status,
                )
            )
        elif name == "random_common_cause":
            # Each noise refit keeps the fit's seed, so its folds and every learner's draws,
            # and the paired shift carries no fold or learner-seed noise.
            noise_fits = []
            for index in range(count):
                noisy = replace(
                    data,
                    baseline=np.column_stack([data.baseline, rng.normal(size=n)]),
                    names=(*data.names, f"_noise_{index}"),
                )
                noise_fits.append(
                    refit(noisy, f"on noise refit {index + 1} of {count}", at=fit_seed)
                )
            shift, shift_sd, _ = paired_shift(
                [on_scale(psi) for psi, _ in noise_fits],
                [on_scale(original)] * count,
                std_error=std_error,
            )
            extras = [("spread of the paired differences (SE units)", shift_sd)]
            if seed_spread:
                child = _generated_child_seeds(seed, name, count)
                base = [
                    refit(data, f"on seed refit {index + 1} of {count}", at=child[index])
                    for index in range(count)
                ]
                _, _, seed_only = paired_shift(
                    [on_scale(psi) for psi, _ in base],
                    [on_scale(psi) for psi, _ in base],
                    std_error=std_error,
                )
                extras.append(("spread across seeds with no noise column (SE units)", seed_only))
            outcomes.append(
                _noise_record(
                    estimand=estimand,
                    original=original,
                    fits=noise_fits,
                    shift=shift,
                    extras=extras,
                    mean_kind=mean_kind,
                    unit=unit,
                    is_ratio=is_ratio,
                )
            )
        else:  # subset; the rules admit no other test
            draws, size = subset_draws, subset_size
            fits = [
                refit(
                    _rows(data, rows),
                    f"on subset refit {index} of {count}",
                    folds=_frozen_folds(estimator, rows),
                )
                for index, rows in enumerate(draws, start=1)
            ]
            outcomes.append(
                _subset_record(
                    estimand=estimand,
                    original=original,
                    fits=fits,
                    reading=_subset_scatter(fits, std_error, n, size, is_ratio),
                    resolution=_resolve("subset", "rho", fit, count),
                    is_ratio=is_ratio,
                    units=n,
                    size=size,
                    status=status,
                )
            )
    return RefutationResult(
        tests=tuple(replace(test, inference=status) for test in outcomes),
        estimand=estimand,
        backend=data.backend,
        random_state=seed,
    )


def _rows(data: Any, rows: np.ndarray) -> Any:
    """``data`` restricted to ``rows``, in order."""
    return replace(
        data,
        time=np.asarray(data.time)[rows],
        event=np.asarray(data.event)[rows],
        treatment=np.asarray(data.treatment)[rows],
        baseline=np.asarray(data.baseline)[rows],
    )
