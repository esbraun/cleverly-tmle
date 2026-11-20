"""How many refits and learner fits a ``refute()`` call would make, counted before any fit.

:func:`planned_refits` takes exactly :func:`~cleverly.validation.refute`'s keywords, runs the
checks ``refute`` runs before its first draw (one shared preflight), and prices every test with
the refit preview each engine defines beside its ``refit`` (X106 parts I and G2). It fits
nothing.

A test whose refits read data that are known before any draw, at seeds known before any draw,
is priced exactly: the noise refits, the negative control and the future-treatment fit. Every
other test refits drawn data, whose constant folds and class counts are not known ahead, so it
is priced at the learners' nominal counts with no constant-fold skip and reads as an upper
bound with a floor (``min_fits``) that holds when no draw fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .._planned_fits import PlannedFits
from .._summary import Summary

__all__ = ["PlannedRefitTest", "PlannedRefits", "planned_refits"]

Bound = Literal["exact", "upper"]

#: What every record says, whatever its tests.
_COMMON_NOTES = (
    "A refit that raises after some of its learners fitted is kept as a failure, so a test "
    "with failed draws can make fewer fits than min_fits.",
)
_DRAW_NOTE = (
    "A subset draw, a band-rank placebo draw and any refit can still refuse the call: the "
    "preview checks the request, not the draws that follow other tests' draws."
)
_BUILD_NOTE = (
    "A longitudinal plasmode builds its law and truth inside refute(), before the first draw; "
    "a regimen or density that fails on simulated histories stops the call there."
)
_CONTROL_CHECK_NOTE = (
    "The check that the control is not nearly a function of the adjustment set fits two "
    "least-squares regressions before any refit. The preview runs that check too, so it "
    "makes those two fits itself; they are counted under preparation."
)
_SEED_NOTE = (
    "Neither random_state nor the fit records a seed, so refute() draws one; tests whose "
    "refits read that seed are counted at nominal counts."
)


@dataclass(frozen=True)
class PlannedRefitTest:
    """The refits one test of a ``refute()`` call would make.

    Parameters
    ----------
    name : str
        The test, as ``tests=`` names it.
    draws : int
        The draws the test makes: the resolved replicate count, or 1 for a one-off fit.
    refits_per_draw : int
        2 for ``random_common_cause`` with ``seed_spread=True``, else 1.
    refits : int
        The refits the test runs. 0 where it runs none, such as a longitudinal placebo for a
        regimen its kernel never draws.
    unit : PlannedFits or None
        The preview of one refit; ``None`` where no refit runs.
    learner_fits : int
        Top-level learner fits over every refit, plus :attr:`preparation`. For an upper test,
        the most the test can make.
    candidate_fits : int
        Fits of base estimators inside those fits, plus :attr:`preparation`.
    min_fits : int
        The fewest top-level fits the test makes when no draw fails.
    preparation : int
        Fits made once before the draws, such as the frozen models of a plasmode law.
    bound : {"exact", "upper"}
        ``"exact"`` when the counts are the ones the test makes, assuming no fit raises.
    notes : tuple of str
        What the counts of this test assume.
    """

    name: str
    draws: int
    refits_per_draw: int
    refits: int
    unit: PlannedFits | None
    learner_fits: int
    candidate_fits: int
    min_fits: int
    preparation: int = 0
    bound: Bound = "exact"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedRefits:
    """The refits and learner fits a ``refute()`` call would make, counted before any fit.

    :func:`cleverly.validation.planned_refits` returns one.

    Parameters
    ----------
    tests : tuple of PlannedRefitTest
        One record per requested test, in the order ``refute()`` runs them.
    estimand : str
        The estimand the tests refit.
    random_state : int or None
        The seed ``refute()`` would use, or ``None`` when it draws one at the call.
    notes : tuple of str
        Plain sentences about what the counts assume.

    Attributes
    ----------
    total_refits : int
    total_learner_fits : int
    total_candidate_fits : int
    min_fits : int
    exact : bool
    """

    tests: tuple[PlannedRefitTest, ...]
    estimand: str
    random_state: int | None
    notes: tuple[str, ...] = ()

    @property
    def total_refits(self) -> int:
        """The refits over every test."""
        return sum(test.refits for test in self.tests)

    @property
    def total_learner_fits(self) -> int:
        """The top-level learner fits over every test, an upper bound when not :attr:`exact`."""
        return sum(test.learner_fits for test in self.tests)

    @property
    def total_candidate_fits(self) -> int:
        """The base-estimator fits over every test."""
        return sum(test.candidate_fits for test in self.tests)

    @property
    def min_fits(self) -> int:
        """The fewest top-level fits over every test when no draw fails."""
        return sum(test.min_fits for test in self.tests)

    @property
    def exact(self) -> bool:
        """Whether every test is exact."""
        return all(test.bound == "exact" for test in self.tests)

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Summary:
        """Return the printed table of the planned refits.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            ``"full"`` adds each test's notes and the record's notes.

        Returns
        -------
        Summary
            A ``str`` whose text is the printed table; a notebook renders it as HTML.
        """
        from .._summary._setup import planned_refits_summary

        return planned_refits_summary(self, detail)

    def to_frame(self, backend: str | None = None) -> Any:
        """Return one row per test.

        Parameters
        ----------
        backend : {"pandas", "polars", "pyarrow"} or None, default=None
            Dataframe backend to return. ``None`` uses pandas when installed, then the
            first available backend.

        Returns
        -------
        dataframe
            Columns ``test``, ``draws``, ``refits_per_draw``, ``refits``, ``learner_fits``,
            ``candidate_fits``, ``min_fits``, ``preparation`` and ``bound``.
        """
        from ..utils.frames import frame_from_dict

        return frame_from_dict(
            {
                "test": [test.name for test in self.tests],
                "draws": [test.draws for test in self.tests],
                "refits_per_draw": [test.refits_per_draw for test in self.tests],
                "refits": [test.refits for test in self.tests],
                "learner_fits": [test.learner_fits for test in self.tests],
                "candidate_fits": [test.candidate_fits for test in self.tests],
                "min_fits": [test.min_fits for test in self.tests],
                "preparation": [test.preparation for test in self.tests],
                "bound": [test.bound for test in self.tests],
            },
            backend=backend,
        )

    def __repr__(self) -> str:
        from .._summary import short_repr

        fits = self.total_learner_fits
        return short_repr(
            "PlannedRefits",
            (
                ("refits", self.total_refits),
                ("learner fits", fits if self.exact else f"at most {fits}"),
                ("", "exact" if self.exact else "upper bound"),
            ),
        )

    def _repr_html_(self) -> str:
        return self.summary().as_html()


# ----------------------------------------------------------------- pricing


class _Test:
    """Collects the refit previews of one test."""

    def __init__(self, name: str, draws: int, per_draw: int = 1) -> None:
        self.name = name
        self.draws = draws
        self.per_draw = per_draw
        self.parts: list[tuple[PlannedFits, int]] = []
        self.preparation = 0
        self.upper = False
        self.notes: list[str] = []

    def add(self, plan: PlannedFits, times: int = 1) -> None:
        self.parts.append((plan, int(times)))

    def build(self) -> PlannedRefitTest:
        refits = sum(times for _, times in self.parts)
        exact = not self.upper and all(plan.exact for plan, _ in self.parts)
        return PlannedRefitTest(
            name=self.name,
            draws=self.draws,
            refits_per_draw=self.per_draw,
            refits=refits,
            unit=self.parts[0][0] if self.parts else None,
            learner_fits=sum(plan.total_fits * times for plan, times in self.parts)
            + self.preparation,
            candidate_fits=sum(plan.total_candidate_fits * times for plan, times in self.parts)
            + self.preparation,
            min_fits=sum(plan.min_fits * times for plan, times in self.parts) + self.preparation,
            preparation=self.preparation,
            bound="exact" if exact else "upper",
            notes=tuple(self.notes),
        )


def _preview(estimator: Any, data: Any, **options: Any) -> PlannedFits:
    """``estimator._planned_refit(data, **options)``, refusing an estimator with none."""
    from ..exceptions import CapabilityError, RefusalReason

    planned = getattr(estimator, "_planned_refit", None)
    if planned is None:
        raise CapabilityError(
            f"{type(estimator).__name__} has no preview of its refit, so the refits of a "
            "refutation cannot be counted before they run",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Run refute() and read its refits, or fit an engine that "
            "CausalStudy dispatches to.",
            roadmap="X106",
        )
    plan: PlannedFits = planned(data, **options)
    return plan


def _family_data(data: Any, family: str) -> Any:
    """``data`` with an outcome of ``family``, for a nominal count, whose values it never reads
    beyond its classes."""
    if str(data.family) == family:
        return data
    n = int(data.n)
    stand_in = (np.arange(n) % 2).astype(float) if family == "binomial" else np.linspace(0, 1, n)
    return data.with_outcome(stand_in, family=family)


def _point_plasmode_preparation(data: Any) -> int:
    """The models :meth:`PlasmodeLaw.fit` fits on a point fit's data."""
    fits = 0 if data.known_treatment is not None else 1
    fits += int(bool(data.has_missing_outcome))
    fits += int(bool(data.has_missing_treatment))
    fits += int(bool(data.has_intermediate))
    return fits


def _longitudinal_plasmode_preparation(data: Any) -> int:
    """The models :meth:`LongitudinalPlasmodeLaw.fit` fits on a panel."""
    from ._longitudinal_plasmode import _binary

    def logistic(values: np.ndarray) -> int:
        values = np.asarray(values, dtype=float)
        return int(values.size > 0 and float(np.mean(values)) not in (0.0, 1.0))

    treatment = np.asarray(data.treatment, dtype=float)
    uncensored = np.asarray(data.uncensored, dtype=bool)
    blocks = [np.asarray(block, dtype=float) for block in data.time_varying]
    repeated = tuple(data.repeated_names)
    fits = 0
    for node in range(int(data.n_times)):
        rows = ~np.isnan(treatment[:, node])
        names = data.time_varying_names[node]
        for column in range(blocks[node].shape[1]):
            if repeated and node and names[column] == repeated[node - 1]:
                continue
            values = blocks[node][rows, column]
            fits += logistic(values) if _binary(values) else 1
        fits += logistic(treatment[rows, node])
        if data.censoring_names:
            fits += logistic(uncensored[rows, node].astype(float))
    return fits


def planned_refits(result: Any, **kwargs: Any) -> PlannedRefits:
    """Count the refits and learner fits :func:`~cleverly.validation.refute` would make.

    Takes exactly ``refute``'s keywords and runs every check ``refute`` runs before its first
    draw, so a request ``refute`` refuses raises the same record here. It fits nothing.

    Parameters
    ----------
    result : TMLEResult or another fitted result that ``refute`` accepts
        The fit.
    **kwargs : Any
        ``refute``'s keywords: ``estimand``, ``tests``, ``n_replicates``, ``seed_spread``
        and the rest. Only ``random_state`` and the arguments that decide what runs change
        the counts.

    Returns
    -------
    PlannedRefits
        One record per requested test.

    Raises
    ------
    TypeError
        If a keyword is not one ``refute`` takes.
    ValueError
        If an argument is malformed, as ``refute`` raises it.
    CapabilityError
        If ``refute`` refuses the request, with the same record, or if the fit's estimator
        has no refit preview.

    Notes
    -----
    The noise refits, the negative control and the future-treatment fit read data and seeds
    known before any draw, so they are counted exactly. Every other test refits drawn data and
    is counted at the learners' nominal counts, as an upper bound with a floor that holds when
    no draw fails. A draw that a check refuses part way through (a subset draw, a band-rank
    placebo draw on a fit whose seed is drawn at the call) can still stop ``refute``.
    """
    import inspect

    from .refute import refute

    bound = inspect.signature(refute).bind(result, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("result")
    return _planned(result, arguments)


def _planned(result: Any, a: dict[str, Any]) -> PlannedRefits:
    import importlib

    module: Any = importlib.import_module("cleverly.validation.refute")
    from .._continuous_survival_family import is_continuous_survival
    from ._band_placebo import placebo_bands
    from ._structural_refute import (
        is_structural,
        structural_fit_seeds,
        structural_frozen_folds,
    )

    request = module._refute_preflight(
        result,
        estimand=a["estimand"],
        tests=a["tests"],
        n_replicates=a["n_replicates"],
        placebo_mode=a["placebo_mode"],
        noise_nodes=a["noise_nodes"],
        negative_control_outcome=a["negative_control_outcome"],
        negative_control_kind=a["negative_control_kind"],
        negative_control_floor=a["negative_control_floor"],
        future_treatment_control=a["future_treatment_control"],
        future_treatment_node=a["future_treatment_node"],
        dummy_outcome=a["dummy_outcome"],
        simulated_outcome=a["simulated_outcome"],
        outcome_bands=a["outcome_bands"],
        bootstrap_measurement_error=a["bootstrap_measurement_error"],
        measurement_error_rule=a["measurement_error_rule"],
    )
    estimator = result.estimator
    random_state = a["random_state"]
    if random_state is not None:
        seed: int | None = int(random_state)
    elif getattr(estimator, "random_state", None) is not None:
        seed = int(estimator.random_state)
    else:
        seed = None
    known = seed is not None
    root = 0 if seed is None else seed
    notes = list(_COMMON_NOTES)
    if not known:
        notes.append(_SEED_NOTE)
    n_replicates = a["n_replicates"]
    seed_spread = bool(a["seed_spread"])
    data = result.data
    tests: list[PlannedRefitTest] = []

    def count(name: str) -> int:
        if n_replicates is not None:
            return int(n_replicates)
        if name in module._GENERATED_TESTS:
            return int(module.DEFAULT_OUTCOME_REPLICATES)
        if name in ("bootstrap_measurement_error", "bootstrap_stability"):
            return int(module._DEFAULT_MEASUREMENT_REPLICATES)
        if name == "random_common_cause":
            return int(module._DEFAULT_NOISE_REPLICATES)
        return int(module._DEFAULT_PERTURBATION_REPLICATES)

    if is_continuous_survival(result):
        unit_estimator = estimator._for_estimate_refits()
        survival_seed = int(estimator.random_state)
        for name in request.tests:
            draws = count(name)
            if name == "random_common_cause":
                test = _Test(name, draws, 2 if seed_spread else 1)
                test.add(_preview(unit_estimator, data, random_state=survival_seed), draws)
                if seed_spread:
                    for child in module._generated_child_seeds(root, name, draws):
                        test.add(
                            _preview(unit_estimator, data, random_state=child, nominal=not known)
                        )
            else:
                test = _Test(name, draws)
                options: dict[str, Any] = {"random_state": root, "nominal": True}
                if name == "subset":
                    options["_frozen_folds"] = np.asarray(estimator._outer_folds)
                test.add(_preview(unit_estimator, data, **options), draws)
                test.upper = True
            tests.append(test.build())
        notes.append(_DRAW_NOTE)
        return PlannedRefits(
            tests=tuple(tests), estimand=request.estimand, random_state=seed, notes=tuple(notes)
        )

    longitudinal = module._is_longitudinal(result)
    structural = is_structural(result)
    panel = longitudinal or structural
    intermediate = module._intermediate_option(result)
    held = module._frozen_rule(result)
    rows = np.arange(int(data.n))

    def held_fitter() -> Any:
        if held is None:
            return estimator
        from ..interventions.learned import frozen_rule_replay

        return frozen_rule_replay(estimator, held, None)

    requested = request.tests
    if "placebo" in requested:
        bands = (
            request.placebo_bands["bands"]
            if "bands" in request.placebo_bands
            else placebo_bands(result)
        )
        if bands is not None:
            if known:
                # The band-rank sets are the first draws of refute's generator, so this replay
                # of its per-draw checks is exact.
                module._band_placebo_states(
                    result,
                    bands,
                    np.random.default_rng(root),
                    a["placebo_mode"],
                    count("placebo"),
                )
            else:
                notes.append(_DRAW_NOTE)
    for name in requested:
        draws = count(name)
        if name == "placebo":
            test = _Test(name, draws)
            test.upper = True
            gap = None
            if longitudinal:
                from ._markov_placebo import MarkovKernel

                gap = module._placebo_positivity_gap(result, MarkovKernel.fit(data))
            if gap is not None:
                test.notes.append(gap[0].upper() + gap[1:] + ".")
                test.upper = False
            else:
                test.add(
                    _preview(estimator, data, random_state=root, nominal=True, **intermediate),
                    draws,
                )
        elif name == "random_common_cause":
            test = _Test(name, draws, 2 if seed_spread else 1)
            repeats = getattr(result, "repeats", None)
            fit_seed = repeats[0].seed if repeats else seed
            repeat_seeds = [item.seed for item in repeats] if repeats else None
            options = dict(intermediate)
            if structural:
                options["frozen_folds"] = structural_frozen_folds(result)
                repeat_seeds = list(structural_fit_seeds(result) or ()) or None
                fit_seed = repeat_seeds[0] if repeat_seeds else fit_seed
            if repeat_seeds is not None:
                options["frozen_seeds"] = repeat_seeds
            # ``refit(random_state=None)`` keeps the estimator's own seed, as refute's does.
            own = getattr(estimator, "random_state", None) is not None
            fixed = (fit_seed is not None or own) and (
                repeat_seeds is None or own or all(item is not None for item in repeat_seeds)
            )
            noisy = module._noise_data(data, np.random.default_rng(0), 0, request.noise_nodes)
            fitter = held_fitter()
            test.add(
                _preview(
                    fitter,
                    noisy,
                    random_state=fit_seed,
                    nominal=not fixed,
                    **options,
                ),
                draws,
            )
            if seed_spread:
                for child in module._generated_child_seeds(root, name, draws):
                    test.add(
                        _preview(
                            fitter, data, random_state=child, nominal=not known, **intermediate
                        )
                    )
        elif name == "subset":
            test = _Test(name, draws)
            test.upper = True
            test.add(
                _preview(
                    held_fitter(),
                    data,
                    random_state=root,
                    nominal=True,
                    frozen_folds=module._sliced_folds(result, rows),
                    **intermediate,
                ),
                draws,
            )
        elif name == "negative_control_outcome":
            test = _Test(name, 1)
            if a["negative_control_kind"] == "pre_treatment":
                matrix, _, kept = module._adjustment_set(data, longitudinal)
                values = np.asarray(a["negative_control_outcome"], dtype=float).reshape(-1)
                test.preparation = module._residual_check_fits(values[kept], matrix[kept])
                test.notes.append(_CONTROL_CHECK_NOTE)
            replacement = data.with_outcome(
                a["negative_control_outcome"], name="negative_control_outcome"
            )
            if structural:
                test.add(
                    _preview(
                        estimator,
                        replacement,
                        random_state=root,
                        frozen_folds=structural_frozen_folds(result),
                        frozen_seeds=structural_fit_seeds(result),
                        nominal=False,
                    )
                )
            else:
                control, _ = module._control_estimator(
                    estimator,
                    str(replacement.family),
                    module._control_values(replacement, longitudinal),
                )
                test.add(
                    _preview(
                        control, replacement, random_state=root, nominal=not known, **intermediate
                    )
                )
        elif name == "future_treatment":
            test = _Test(name, 1)
            fitter, point, *_ = module._future_treatment_fit(
                result,
                control=a["future_treatment_control"],
                node=int(a["future_treatment_node"]) + 1,
                seed=root,
            )
            test.add(_preview(fitter, point, random_state=root, nominal=not known))
            node = int(a["future_treatment_node"]) + 1
            present = data.node_present(node)
            history, _ = module._future_history(data, node)
            control = np.asarray(a["future_treatment_control"], dtype=float).reshape(-1)
            test.preparation = module._residual_check_fits(control[present], history[present])
            test.notes.append(_CONTROL_CHECK_NOTE)
        elif name in module._GENERATED_TESTS:
            test = _Test(name, draws)
            test.upper = True
            process = request.process(name)
            family = str(getattr(process, "family", data.family))
            target = data if panel else _family_data(data, family)
            seeds = module._generated_child_seeds(root, name, draws)
            test.add(
                _preview(estimator, target, random_state=seeds[0], nominal=True, **intermediate),
                draws,
            )
            if name == "plasmode":
                test.preparation = (
                    _longitudinal_plasmode_preparation(data)
                    if longitudinal
                    else _point_plasmode_preparation(data)
                )
                if longitudinal:
                    notes.append(_BUILD_NOTE)
        else:  # the bootstrap tests
            test = _Test(name, draws)
            test.upper = True
            options = dict(intermediate)
            if structural:
                options["frozen_seeds"] = structural_fit_seeds(result)
            test.add(
                _preview(
                    held_fitter(),
                    data,
                    random_state=root,
                    nominal=True,
                    frozen_folds=module._sliced_folds(result, rows),
                    **options,
                ),
                draws,
            )
        tests.append(test.build())
    if ("subset" in requested or "placebo" in requested) and _DRAW_NOTE not in notes:
        notes.append(_DRAW_NOTE)
    return PlannedRefits(
        tests=tuple(tests), estimand=request.estimand, random_state=seed, notes=tuple(notes)
    )
