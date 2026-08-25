"""What a method-evidence study declares, and the register of the studies that exist.

A study is a *declaration*: a law to sample from, an estimator configuration, the estimands
it reports, the margins its verdicts are bounded by, and where its artefacts and its
documentation row live.  Everything else -- the summaries, the verdicts, the negative
controls, the manifest, the documentation gate -- is shared machinery that reads this record.
Registering a new method is therefore a matter of writing one of these, not of copying a
study module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from tests.studies.evidence.inference import se_ratio_for_coverage

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Margins:
    """Pre-declared acceptance margins.

    Every field is a statement about the estimator that does not depend on how many
    replications were run, which is what lets more Monte Carlo make a verdict easier rather
    than harder.  ``coverage_floor`` is the binding validity claim; ``se_ratio_sanity`` is a
    wider screen on the reported standard error, and :meth:`validate` refuses a pair where
    the screen would bind first and quietly become the real gate.
    """

    confidence_level: float = 0.99
    alpha: float = 0.05
    bootstrap_replicates: int = 10_000
    #: Bias must lie within this many empirical standard deviations of zero.
    standardized_bias: float = 0.25
    #: The lower bound of the exact coverage interval must clear this.
    coverage_floor: float = 0.90
    #: Reported only: coverage above this is conservative rather than invalid.
    over_coverage_ceiling: float = 0.99
    #: Two-sided sanity band on mean reported SE over empirical SD.
    se_ratio_sanity: tuple[float, float] = (0.80, 1.20)
    #: Two-sided *calibration* band, for the one property cell where both nuisances are
    #: correctly specified.  Deliberately tighter than the coverage floor implies, which
    #: :meth:`__post_init__` forbids for ``se_ratio_sanity`` -- and the two are different
    #: instruments rather than the same one declared twice.  ``se_ratio_sanity`` is a screen
    #: standing behind a one-sided validity gate, applied to every estimand of every law
    #: including the ones whose influence-curve standard error is only conservative; here the
    #: study separately tests calibration against the sampling spread, so a departure is a
    #: defect rather than a regime.  Both remain equivalence-shaped, so both still become
    #: easier to satisfy as replications are added.
    calibration_se_ratio: tuple[float, float] = (0.93, 1.07)
    #: The same claim on the coverage scale, also two-sided.  Asking whether the exact
    #: interval lies inside this band is answerable and gets easier with replications; asking
    #: whether coverage *is* 0.95 is the point test the coverage floor exists to avoid.
    calibration_coverage: tuple[float, float] = (0.92, 0.98)
    #: How far above nominal a test's size may be established to sit.  One-sided: a test that
    #: under-rejects is conservative, and a power cell is what stops an inert one passing.
    type_i_margin: float = 0.05
    #: Paired mean difference, in pooled empirical standard deviations.
    paired_difference: float = 0.15
    rmse_noninferiority: float = 1.10
    coverage_noninferiority: float = -0.025
    calibration_noninferiority: float = 0.05

    def __post_init__(self) -> None:
        implied = se_ratio_for_coverage(self.coverage_floor, alpha=self.alpha)
        if self.se_ratio_sanity[0] > implied:
            raise ValueError(
                f"the SE sanity band's lower limit {self.se_ratio_sanity[0]} is tighter than "
                f"the coverage floor {self.coverage_floor} implies ({implied:.4f}), so the "
                f"screen would bind before the validity gate does"
            )
        low, high = self.calibration_se_ratio
        if low < self.se_ratio_sanity[0] or high > self.se_ratio_sanity[1]:
            raise ValueError(
                f"the calibration band {self.calibration_se_ratio} is not inside the sanity "
                f"band {self.se_ratio_sanity}, so it adds no claim the screen does not already "
                f"make"
            )
        if not low < 1.0 < high:
            raise ValueError(
                f"the calibration band {self.calibration_se_ratio} excludes a correctly scaled "
                f"standard error, so no valid estimator could satisfy it"
            )
        nominal = 1.0 - self.alpha
        if not self.calibration_coverage[0] < nominal < self.calibration_coverage[1]:
            raise ValueError(
                f"the calibration coverage band {self.calibration_coverage} excludes the "
                f"nominal rate {nominal}, so no valid interval could satisfy it"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1); got {self.confidence_level}")

    @property
    def coverage_at_se_floor(self) -> float:
        """The coverage the sanity band's lower limit corresponds to, for the record."""
        from tests.studies.evidence.inference import coverage_for_se_ratio

        return coverage_for_se_ratio(self.se_ratio_sanity[0], alpha=self.alpha)

    def as_json(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "alpha": self.alpha,
            "bootstrap_replicates": self.bootstrap_replicates,
            "standardized_bias_margin": self.standardized_bias,
            "coverage_floor": self.coverage_floor,
            "over_coverage_ceiling": self.over_coverage_ceiling,
            "se_ratio_sanity_band": list(self.se_ratio_sanity),
            "calibration_se_ratio_band": list(self.calibration_se_ratio),
            "calibration_coverage_band": list(self.calibration_coverage),
            "type_i_margin": self.type_i_margin,
            "paired_difference_margin_sd": self.paired_difference,
            "rmse_ratio_noninferiority_margin": self.rmse_noninferiority,
            "coverage_difference_noninferiority_margin": self.coverage_noninferiority,
            "calibration_excess_noninferiority_margin": self.calibration_noninferiority,
        }


@dataclass(frozen=True)
class StudyRecord:
    """One row of the implementation validation grid, and all the machinery needs to check it."""

    #: The name the documentation grid's first cell must carry, verbatim.
    name: str
    slug: str
    #: Directory holding the committed artefacts.
    artifacts: Path
    #: Reader-facing document and the heading anchor its section lives under.
    document: str
    anchor: str
    #: Scenario name -> the estimands that scenario reports.
    scenarios: Mapping[str, tuple[str, ...]]
    replicates: int
    n: int
    seed: int
    margins: Margins = field(default_factory=Margins)
    implementation: str = "cleverly"
    #: The comparison implementation, or ``None`` for a study with no canonical comparator.
    reference: str | None = None
    #: Why this study keeps a comparator that fails its *own* independent truth gates.
    #:
    #: :mod:`tests.studies.evidence.comparison` already separates the two questions: whether
    #: the subject is similar to and no worse than the reference, and whether either is any
    #: good on its own.  A reference that degrades is reported in ``reference_valid`` rather
    #: than turning the subject's row red.  The regeneration driver nonetheless refused any
    #: run whose reference failed, which made deleting the comparator the only way to publish
    #: -- and deleting it throws away the paired similarity and non-inferiority evidence,
    #: which is the strongest statement a study can make about an implementation it did not
    #: write.
    #:
    #: Setting this string keeps the comparator and records the reason, in the spirit of the
    #: ``accepted:`` lines in ``tests/prose-report.md``: the exception is allowed and the
    #: recorded reason is the point.  It relaxes nothing about the subject.  ``passed`` still
    #: gates on similarity and non-inferiority, and ``subject_valid`` is still required.
    accepted_reference_failure: str = ""
    #: Estimands whose reference reports its standard error on a different scale, so a raw
    #: SE comparison would compare two different reported quantities.
    incomparable_se: frozenset[str] = frozenset()
    #: Repository-relative modules whose content the manifest records.
    modules: tuple[str, ...] = ()
    #: Import paths for the study-specific sampling/fitting and property adapters.
    runner_module: str = "tests.studies.canonical_tmle"
    properties_module: str = "tests.studies.canonical_properties"
    #: Property name -> the cells the committed property summary must contain.
    property_cells: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def implementations(self) -> tuple[str, ...]:
        return (
            (self.implementation,)
            if self.reference is None
            else (self.implementation, self.reference)
        )

    @property
    def estimands(self) -> frozenset[str]:
        return frozenset(name for names in self.scenarios.values() for name in names)

    @property
    def cells(self) -> int:
        """Scenario-estimand cells, which is what the paired verdict table has one row per."""
        return sum(len(names) for names in self.scenarios.values())

    def artifact(self, name: str) -> Path:
        return self.artifacts / name

    @property
    def document_path(self) -> Path:
        return ROOT / self.document

    def runner(self) -> Any:
        """The study-specific module that draws samples and refits the subject."""
        return import_module(self.runner_module)

    def properties(self) -> Any:
        """The study-specific module that generates and summarizes property cells."""
        return import_module(self.properties_module)


def registered() -> tuple[StudyRecord, ...]:
    """Every registered study, in documentation-grid order.

    Imported lazily so the framework modules stay importable without pulling in a study's
    estimator configuration, and so a study module can import the framework.
    """
    from tests.studies.canonical_ctmle_oat import STUDY as CANONICAL_CTMLE_OAT
    from tests.studies.canonical_ctmle_selector import STUDY as CANONICAL_CTMLE_SELECTOR
    from tests.studies.canonical_cvtmle import STUDY as CANONICAL_CVTMLE
    from tests.studies.canonical_ltmle import STUDY as CANONICAL_LTMLE
    from tests.studies.canonical_ltmle_crossfit import STUDY as CANONICAL_LTMLE_CROSSFIT
    from tests.studies.canonical_ltmle_survival import STUDY as CANONICAL_LTMLE_SURVIVAL
    from tests.studies.canonical_ltmle_survival_crossfit import (
        STUDY as CANONICAL_LTMLE_SURVIVAL_CROSSFIT,
    )
    from tests.studies.canonical_tmle import STUDY as CANONICAL_TMLE
    from tests.studies.fold_evaluated_cvtmle import STUDY as FOLD_EVALUATED_CVTMLE

    return (
        CANONICAL_TMLE,
        CANONICAL_CVTMLE,
        FOLD_EVALUATED_CVTMLE,
        CANONICAL_CTMLE_SELECTOR,
        CANONICAL_CTMLE_OAT,
        CANONICAL_LTMLE,
        CANONICAL_LTMLE_CROSSFIT,
        CANONICAL_LTMLE_SURVIVAL,
        CANONICAL_LTMLE_SURVIVAL_CROSSFIT,
    )
