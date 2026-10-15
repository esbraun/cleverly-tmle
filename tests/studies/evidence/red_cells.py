"""The red-cell ledger: every red verdict a registered study publishes, and the ask that owns it.

A ``reporting`` study publishes its red verdicts instead of refusing to publish, which is the
whole point of the policy -- and also why a red cell can sit on a study page for months with no
record of who is meant to close it.  RM18's prose named most of them and missed one outright:
the paired ``ate_policy[+0.25 vs natural course]`` row of ``shift-policies`` was red and no
roadmap row mentioned it.  A hand-kept list drifts the same way every hand-typed number does.

So the ledger is generated.  :func:`red_rows` reads each study's committed verdict tables,
:data:`OWNERS` declares which roadmap ask owns each red row, and :func:`findings` states every
way the two can disagree.  ``tests/unit/test_red_cell_ledger.py`` asserts there are none and that
the published page is what :func:`fill` renders.

Three kinds of row can be red, and each is read the way the regeneration gates it.

* A **truth** row of ``performance-tests.csv`` that fails.  The comparator's own row is left
  out when the study declares an accepted reference failure, as ``regenerate.py`` does.
* A **paired** row of ``equivalence.csv`` that fails.  The verdict is *recomputed* from the
  committed leg endpoints rather than read off ``passed``, and a disagreement is refused.  A
  ledger that trusted the column would inherit any hand edit to it.
* A **property** row of ``properties.csv`` that fails its own rule *or* its family's joint
  clause, which is the rule ``claims.property_cells_passed`` counts by.  A diagnostic row
  states no verdict and is never red.
* An **accounting** row of ``failure-accounting.csv``: a hook audit that the study's recorded
  fitting failures leave red.  A truth or paired cell that the accounting leaves red stays a
  truth or paired row, and its ``fails by`` names the accounting.

A study module may also define a ``scientific_failures`` hook, which the regeneration driver
adds to the failures it gates or reports.  Those hooks audit fits rather than publish verdict
cells, and the ledger does not read them.  :data:`EXCLUDED_HOOKS` names every study that has
one and what the hook reports, and ``tests/unit/test_red_cell_ledger.py`` fails when a new hook
appears, and when a ``gated`` study's hook reports a failure in its committed artefacts.
``claims.load`` is never called either: it reads the multi-megabyte replication archives, and
no verdict here needs them.

Run it after ``python -m tests.studies.evidence.document``, which also calls :func:`fill`::

    python -m tests.studies.evidence.red_cells
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tests.documents import pipe_table
from tests.studies.evidence import failures, property_verdicts
from tests.studies.evidence.comparison import ComparisonVerdict, comparison_verdict
from tests.studies.evidence.document import _CLOSE, _OPEN, _table, measured, render
from tests.studies.evidence.manifest import write_lines
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord, registered

#: The published ledger.
PAGE = ROOT / "docs" / "technical-reference" / "method-evidence" / "red-cells.md"

#: The roadmap, and the header of its "Red-cell owners" table, whose ``id`` column names every
#: owner.
ROADMAP = ROOT / "docs" / "roadmap.md"
OWNER_TABLE = ("id", "work", "acceptance")

#: Where a reader finds each owner.  ``F18``, ``F19`` and ``F27`` have sections of their own;
#: every other owner is a row of the roadmap's "Red-cell owners" table.
_OWNERS_SECTION = "red-cell-owners"
_OWNER_ANCHORS = {
    "F18": "f18-selector-path-c-tmle-inference",
    "F19": "f19-outcome-adaptive-c-tmle-generated-design-inference",
    # The owner of every red cell of the reporting study ``learned-rule-cvtmle-boundary``.
    "F27": "f27-learned-policy-value-outside-the-published-conditions",
}

#: The kinds of red row, in the order the ledger prints them.  An ``accounting`` row is a hook
#: audit that a recorded fitting failure leaves red (``failure-accounting.csv``).
KINDS = ("truth", "paired", "property", "accounting")

#: The three generated blocks of :data:`PAGE`.
BLOCKS = ("red-cell-summary", "red-cells", "reporting-without-red")

SUMMARY_COLUMNS = ("owner", "red rows", "studies")
RED_COLUMNS = ("study", "kind", "row", "role", "fails by", "measured", "owner")
QUIET_COLUMNS = ("study", "verdicts it publishes", "red rows")


@dataclass(frozen=True, order=True)
class RedKey:
    """Which committed verdict a ledger row is.

    Parameters
    ----------
    slug : str
        The study's registered slug.
    kind : str
        One of :data:`KINDS`.
    key : str
        ``property/cell`` for a property row, ``scenario/estimand`` for a paired row,
        ``implementation/scenario/estimand`` for a truth row, and ``table/leg`` for an
        accounting row.
    """

    slug: str
    kind: str
    key: str


@dataclass(frozen=True)
class RedRow:
    """One red verdict, with what it failed and the endpoint it failed on.

    Parameters
    ----------
    key : RedKey
        The verdict this row is.
    role : str
        The property role, ``paired``, or the implementation a truth row judges.
    fails_by : str
        Which rule, clause or leg failed.
    measured : str
        The endpoints the failed verdict was read from.
    """

    key: RedKey
    role: str
    fails_by: str
    measured: str


def _keys(slug: str, kind: str, *keys: str) -> tuple[RedKey, ...]:
    return tuple(RedKey(slug, kind, key) for key in keys)


#: The roadmap ask that owns each red row, by its ``id`` in the "Red-cell owners" table.
#:
#: Written owner-first so a reader sees each ask's cells together, and flattened into
#: :data:`OWNERS` below, which refuses a row claimed twice.  Every entry must be red and every
#: red row must have an entry; :func:`findings` states both directions.
CLAIMS: dict[str, tuple[RedKey, ...]] = {
    "F18": (
        *_keys(
            "canonical-ctmle-selector",
            "property",
            "selector_necessity/collaborative",
            "selector_necessity/empty_control",
            "type_i_error/sharp_null",
        ),
        *_keys(
            "canonical-multi-arm-ctmle-selector",
            "property",
            "selector_necessity/greedy",
            "selector_necessity/ordered",
            "selector_necessity/discrete",
            "selector_necessity/empty_control",
            "type_i_error/sharp_null",
            "interval_calibration/correctly_specified",
        ),
    ),
    "F19": (
        *_keys(
            "canonical-multi-arm-ctmle-oat",
            "property",
            "generated_design/estimated",
            "generated_design/oracle_design",
        ),
        # The per-arm design's revert cells and every primary row they withhold.  Both
        # implementations miss coverage alike, because R ``drtmle`` with ``adapt_g`` computes
        # the same curve, and the two oracle-design rows are red by the joint clause only.
        *_keys(
            "ctmle-oat-per-arm",
            "truth",
            "cleverly-ctmle-oat-per-arm/binary_active/ate",
            "cleverly-ctmle-oat-per-arm/binary_active/ey0",
            "cleverly-ctmle-oat-per-arm/binary_active/ey1",
            "cleverly-ctmle-oat-per-arm/binary_active/or",
            "cleverly-ctmle-oat-per-arm/binary_active/paf",
            "cleverly-ctmle-oat-per-arm/binary_active/par",
            "cleverly-ctmle-oat-per-arm/binary_active/rr",
            "cleverly-ctmle-oat-per-arm/three_arm_active/ate[1 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_active/ate[2 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_active/ey[1]",
            "cleverly-ctmle-oat-per-arm/three_arm_active/ey[2]",
            "cleverly-ctmle-oat-per-arm/three_arm_active/or[1 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_active/or[2 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_active/rr[1 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_active/rr[2 vs 0]",
            "drtmle-r-oat-per-arm/binary_active/ate",
            "drtmle-r-oat-per-arm/binary_active/ey0",
            "drtmle-r-oat-per-arm/binary_active/ey1",
            "drtmle-r-oat-per-arm/binary_active/or",
            "drtmle-r-oat-per-arm/binary_active/paf",
            "drtmle-r-oat-per-arm/binary_active/par",
            "drtmle-r-oat-per-arm/binary_active/rr",
            "drtmle-r-oat-per-arm/three_arm_active/ate[1 vs 0]",
            "drtmle-r-oat-per-arm/three_arm_active/ate[2 vs 0]",
            "drtmle-r-oat-per-arm/three_arm_active/ey[1]",
            "drtmle-r-oat-per-arm/three_arm_active/ey[2]",
            "drtmle-r-oat-per-arm/three_arm_active/or[1 vs 0]",
            "drtmle-r-oat-per-arm/three_arm_active/or[2 vs 0]",
            "drtmle-r-oat-per-arm/three_arm_active/rr[1 vs 0]",
            "drtmle-r-oat-per-arm/three_arm_active/rr[2 vs 0]",
        ),
        *_keys(
            "ctmle-oat-per-arm",
            "property",
            "generated_design/binary_active__estimated",
            "generated_design/binary_active__oracle_design",
            "generated_design/three_arm_active__estimated",
            "generated_design/three_arm_active__oracle_design",
            "interval_calibration/binary_active__correctly_specified",
            "interval_calibration/binary_repeats__correctly_specified",
            "interval_calibration/binary_weighted__correctly_specified",
            "interval_calibration/three_arm_active__correctly_specified",
            "root_n_and_efficiency/n_2000",
            "root_n_and_efficiency/n_500",
            "root_n_and_efficiency/n_8000",
            "simultaneous_coverage/three_arm_active__simultaneous_band",
            "type_i_error/sharp_null",
        ),
        # The strata and missing-data extension: every red truth row of both implementations is
        # short on its standard error with an equivalent bias, the route its reading rule declared.
        *_keys(
            "ctmle-oat-per-arm-strata-missing",
            "truth",
            "cleverly-ctmle-oat-per-arm/missing_outcome/ate",
            "cleverly-ctmle-oat-per-arm/missing_outcome/ey0",
            "cleverly-ctmle-oat-per-arm/missing_outcome/ey1",
            "cleverly-ctmle-oat-per-arm/missing_outcome/or",
            "cleverly-ctmle-oat-per-arm/missing_outcome/rr",
            "cleverly-ctmle-oat-per-arm/missing_treatment/ate",
            "cleverly-ctmle-oat-per-arm/missing_treatment/ey0",
            "cleverly-ctmle-oat-per-arm/missing_treatment/ey1",
            "cleverly-ctmle-oat-per-arm/missing_treatment/or",
            "cleverly-ctmle-oat-per-arm/missing_treatment/rr",
            "cleverly-ctmle-oat-per-arm/strata/ate",
            "cleverly-ctmle-oat-per-arm/strata/ate[S=0]",
            "cleverly-ctmle-oat-per-arm/strata/ey0",
            "cleverly-ctmle-oat-per-arm/strata/ey1",
            "cleverly-ctmle-oat-per-arm/three_arm_missing/ate[1 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_missing/ate[2 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_missing/ey[1]",
            "cleverly-ctmle-oat-per-arm/three_arm_missing/ey[2]",
            "cleverly-ctmle-oat-per-arm/three_arm_missing/or[1 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_missing/or[2 vs 0]",
            "cleverly-ctmle-oat-per-arm/three_arm_missing/rr[1 vs 0]",
            "drtmle-r-oat-per-arm-strata-missing/missing_outcome/ate",
            "drtmle-r-oat-per-arm-strata-missing/missing_outcome/ey0",
            "drtmle-r-oat-per-arm-strata-missing/missing_outcome/ey1",
            "drtmle-r-oat-per-arm-strata-missing/missing_outcome/or",
            "drtmle-r-oat-per-arm-strata-missing/missing_outcome/rr",
            "drtmle-r-oat-per-arm-strata-missing/missing_treatment/ate",
            "drtmle-r-oat-per-arm-strata-missing/missing_treatment/ey0",
            "drtmle-r-oat-per-arm-strata-missing/missing_treatment/ey1",
            "drtmle-r-oat-per-arm-strata-missing/missing_treatment/or",
            "drtmle-r-oat-per-arm-strata-missing/missing_treatment/rr",
            "drtmle-r-oat-per-arm-strata-missing/strata/ate",
            "drtmle-r-oat-per-arm-strata-missing/strata/ate[S=0]",
            "drtmle-r-oat-per-arm-strata-missing/strata/ey0",
            "drtmle-r-oat-per-arm-strata-missing/strata/ey1",
            "drtmle-r-oat-per-arm-strata-missing/three_arm_missing/ate[1 vs 0]",
            "drtmle-r-oat-per-arm-strata-missing/three_arm_missing/ate[2 vs 0]",
            "drtmle-r-oat-per-arm-strata-missing/three_arm_missing/ey[1]",
            "drtmle-r-oat-per-arm-strata-missing/three_arm_missing/ey[2]",
            "drtmle-r-oat-per-arm-strata-missing/three_arm_missing/or[1 vs 0]",
            "drtmle-r-oat-per-arm-strata-missing/three_arm_missing/or[2 vs 0]",
            "drtmle-r-oat-per-arm-strata-missing/three_arm_missing/rr[1 vs 0]",
        ),
        *_keys(
            "ctmle-oat-per-arm-strata-missing",
            "property",
            "root_n_and_efficiency/n_2000",
            "root_n_and_efficiency/n_4000",
            "root_n_and_efficiency/n_8000",
        ),
    ),
    "RM18-fixed-weights": (
        *_keys(
            "weighted-ltmle-crossfit",
            "property",
            "interval_calibration/static__correctly_specified",
        ),
        *_keys(
            "weighted-ltmle-crossfit",
            "paired",
            "selected_censored_end_of_study/ey_regimen[never]",
        ),
    ),
    "RM18-boundary": (
        # RM18-n500 read this cell's endpoint as a boundary resolution, as it read the
        # multi-arm DR-TMLE ``n_500`` cell below.  F18 keeps the selector's interval claims.
        *_keys(
            "canonical-multi-arm-ctmle-selector",
            "property",
            "root_n_and_efficiency/n_500",
        ),
        *_keys(
            "canonical-drtmle",
            "property",
            "double_robust_contraction/treatment_correct_n1500",
        ),
        *_keys(
            "canonical-multi-arm-drtmle",
            "property",
            "double_robust_contraction/outcome_correct_n4000",
            "interval_calibration/correctly_specified",
            "root_n_and_efficiency/n_500",
        ),
        *_keys(
            "weighted-ltmle-crossfit",
            "property",
            "double_robustness/static__both_wrong",
        ),
        *_keys(
            "weighted-ltmle-crossfit",
            "paired",
            "selected_censored_end_of_study/ey_regimen[always]",
            "selected_censored_end_of_study/ey_regimen[treat then continue if l2 positive]",
        ),
    ),
    "RM18-one-sided-bias": (
        *_keys(
            "canonical-drtmle",
            "property",
            "double_robustness/outcome_correct",
            "double_robustness/treatment_correct",
        ),
        *_keys(
            "canonical-multi-arm-drtmle",
            "property",
            "double_robustness/treatment_correct",
        ),
    ),
    "RM18-ordinary-weighted": _keys(
        "weighted-ltmle",
        "property",
        "interval_calibration/static__correctly_specified",
        "type_i_error/static__sharp_null",
        "targeting_necessity/dynamic__targeted",
        "targeting_necessity/dynamic__untargeted",
        "targeting_necessity/static__targeted",
        "targeting_necessity/static__untargeted",
    ),
    "RM18-comparator-density": _keys(
        "shift-policies",
        "paired",
        "continuous_modified_policy/ate_policy[+0.25 vs natural course]",
    ),
    # The arm mean near 0.94 in the smallest stratum, and the two bands that include it.
    "strata-boundary-mean": (
        *_keys(
            "canonical-stratified-tmle",
            "truth",
            "cleverly-stratified-tmle/stratified_binary/ey[1][V=2]",
            "tmle3-stratified/stratified_binary/ey[1][V=2]",
        ),
        *_keys(
            "canonical-stratified-tmle",
            "property",
            "interval_calibration/v2_ey1__correctly_specified",
            "simultaneous_coverage/strata__simultaneous_band",
            "simultaneous_coverage/crossfit_strata__simultaneous_band",
        ),
    ),
    # Four default bands and the cluster multiplier band of the cross-fitted clustered LTMLE,
    # which cover just under the band, read as finite-sample by
    # ``tests/unit/test_band_shortfall_reading.py``.
    "band-finite-sample": (
        *_keys(
            "clustered-cross-fitted-ltmle",
            "property",
            "simultaneous_coverage/clustered_regimens__simultaneous_band",
        ),
        *_keys(
            "canonical-ltmle-survival",
            "property",
            "simultaneous_coverage/all_reported__simultaneous_band",
        ),
        *_keys(
            "default-simultaneous-bands",
            "property",
            "simultaneous_coverage/categorical_ltmle__simultaneous_band",
            "simultaneous_coverage/categorical_ltmle_crossfit__simultaneous_band",
            "simultaneous_coverage/cde_z0__simultaneous_band",
        ),
        *_keys(
            "longitudinal-sdr",
            "property",
            "simultaneous_coverage/six_regimens__simultaneous_band",
        ),
    ),
    # The calibration draws of the multi-arm missing-outcome DR-TMLE study, and the band read
    # off the same fits.  ``tests/unit/test_band_shortfall_reading.py`` rebuilds the reading.
    "F4-calibration-draws": _keys(
        "multi-arm-mar-drtmle",
        "property",
        "interval_calibration/ate__correctly_specified",
        "simultaneous_coverage/arms__simultaneous_band",
    ),
    # The logit-MSM slope in the two smaller strata of the stratified incremental and MSM
    # study.  ``tests/unit/test_band_shortfall_reading.py`` rebuilds the reading.
    "X8-logit-small-stratum": _keys(
        "canonical-stratified-incremental-msm",
        "property",
        "interval_calibration/logit_v1_a__correctly_specified",
        "interval_calibration/logit_v2_a__correctly_specified",
    ),
    # Stratum 2 of the identity-MSM study: a calibration cell and the paired W row that is
    # inconclusive on its calibration leg.  ``tests/unit/test_band_shortfall_reading.py``.
    "X8-identity-small-stratum": (
        *_keys(
            "canonical-stratified-msm-identity",
            "paired",
            "stratified_msm_identity/msm[W][V=2]",
        ),
        *_keys(
            "canonical-stratified-msm-identity",
            "property",
            "interval_calibration/identity_v2_a__correctly_specified",
        ),
    ),
    # The treatment-correct cells of the stratified DR-TMLE study, read by
    # ``tests/diagnostics/x8_drtmle_treatment_correct``: the one-sided bias of RM18 at stratum
    # size, shared by the shipped unstratified fit and by R drtmle.
    "X8-drtmle-one-sided-bias": _keys(
        "canonical-stratified-drtmle",
        "property",
        "double_robustness/marginal_ate__treatment_correct",
        "double_robustness/v0_ate__treatment_correct",
        "double_robustness/v1_ate__treatment_correct",
        "double_robustness/v2_ate__treatment_correct",
    ),
    # The smallest stratum of the stratified DR-TMLE study, red in both implementations alike.
    "X8-drtmle-small-stratum": (
        *_keys(
            "canonical-stratified-drtmle",
            "truth",
            "cleverly-stratified-drtmle/stratified_both_correct/ey[1][V=2]",
            "drtmle-r-stratified/stratified_both_correct/ey[1][V=2]",
        ),
        *_keys(
            "canonical-stratified-drtmle",
            "property",
            "interval_calibration/v2_ate__correctly_specified",
        ),
    ),
    # The two longitudinal bootstrap kinds the X20 study measured red.  Rule 3 of the
    # study keeps each kind out of ``LICENSED_BOOTSTRAP_DESIGNS``.
    "X20-bootstrap": _keys(
        "full-refit-bootstrap-and-derived-contrasts",
        "property",
        "interval_calibration/boot_ey_crossfit__correctly_specified",
        "interval_calibration/boot_ate_crossfit__correctly_specified",
        "interval_calibration/boot_ate_clustered__correctly_specified",
    ),
    # The declared negative control of ``sequential-baselines``: law ``X`` makes every
    # main-terms ``GLM`` wrong, so the calibration rule must fail.  Nothing is owed.
    "X41-negative-control": _keys(
        "sequential-baselines",
        "property",
        "misspecification_control/hajek_x_contrast__main_terms",
        "misspecification_control/ht_x_contrast__main_terms",
        "misspecification_control/ice_x_contrast__main_terms",
    ),
    # The smallest-composite arm of the composite study, red in both implementations alike.
    "composite-high-arm": _keys(
        "composite-missing-drtmle",
        "truth",
        "cleverly-composite-drtmle/three_arm_mar_outcome_and_treatment/ey[high]",
        "drtmle-r-composite/three_arm_mar_outcome_and_treatment/ey[high]",
    ),
    # The two laws of the RM30 reporting study.  F27 states, row by row, that the
    # ``exceptional`` cell has no published result and that the ``weak_blip`` cell has one.
    "F27": _keys(
        "learned-rule-cvtmle-boundary",
        "truth",
        "cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]",
        "cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]",
    ),
    # The finite-sample limits of clustered intervals: the fold-evaluated covariate pair of the
    # unequal-size study, and the few-cluster study's unequal-size fold-evaluated bias cells and
    # DR-TMLE IID controls.
    "F28": (
        *_keys(
            "clustered-unequal-cvtmle",
            "property",
            "clustered_inference/cluster_robust_fold_evaluated_cluster_covariate",
            "clustered_inference/iid_control_fold_evaluated_cluster_covariate",
        ),
        *_keys(
            "clustered-few-cluster-tmle",
            "property",
            "few_cluster_reference/drtmle_crossfit__equal10__j10__iid_t_control",
            "few_cluster_reference/drtmle_crossfit__equal10__j20__iid_t_control",
            "few_cluster_reference/drtmle_crossfit__equal10__j30__iid_t_control",
            "few_cluster_reference/drtmle_crossfit__unequal_informative__j10__iid_t_control",
            "few_cluster_reference/tmle_cv_evaluation__unequal_informative__j10__t_reference",
            "few_cluster_reference/tmle_cv_evaluation__unequal_informative__j20__t_reference",
            "few_cluster_reference/tmle_cv_evaluation__unequal_informative__j30__t_reference",
        ),
    ),
    "F1-power-design": (
        *_keys("stochastic-categorical-ltmle", "property", "power/mix__alternative"),
    ),
    "mtp-longitudinal-limits": (
        *_keys(
            "longitudinal-mtp",
            "property",
            "interval_calibration/categorical_mtp__correctly_specified",
        ),
    ),
    "mtp-point-calibration": (
        *_keys("policy-point-mtp", "property", "interval_calibration/halve__correctly_specified"),
    ),
    "X13-finite-sample": (
        *_keys(
            "point-treatment-survival",
            "property",
            "interval_calibration/weighted_t5__correctly_specified",
        ),
        *_keys(
            "point-treatment-survival-crossfit",
            "property",
            "interval_calibration/three_arm_t3__correctly_specified",
        ),
    ),
    # ``tests/unit/test_sdr_shortfall_reading.py`` rebuilds the reading.
    "X4-finite-sample": (
        *_keys(
            "longitudinal-sdr",
            "property",
            "sequential_double_robustness/both_correct__n_2000",
        ),
    ),
}


def owners(claims: Mapping[str, Sequence[RedKey]]) -> dict[RedKey, str]:
    """Flatten owner-first claims into one owner per red row.

    Parameters
    ----------
    claims : Mapping
        Owner ``id`` to the red rows it claims.

    Returns
    -------
    dict
        Each claimed row, keyed to its one owner.

    Raises
    ------
    ValueError
        If two owners, or one owner twice, claim the same row.
    """
    flat: dict[RedKey, str] = {}
    for owner, keys in claims.items():
        for key in keys:
            if key in flat:
                raise ValueError(f"{key} is claimed by {flat[key]!r} and again by {owner!r}")
            flat[key] = owner
    return flat


OWNERS = owners(CLAIMS)

#: Every study whose module defines a ``scientific_failures`` hook, and what that hook reports.
#:
#: The regeneration driver adds each hook's frames to the failures it gates or reports, and
#: the ledger reads none of them, because none is a verdict cell.  A ``gated`` study refuses to
#: publish when its hook reports a row, so its committed artefacts carry none; the test checks
#: that.  A ``reporting`` study publishes the rows, and its evidence page states them.
EXCLUDED_HOOKS: dict[str, str] = {
    "canonical-drtmle": "the score audit of both implementations and the subject's solver flag",
    "canonical-multi-arm-drtmle": (
        "the score audit of both implementations and the subject's solver flag"
    ),
    "mar-natural-course-tmle": "the exact-equality probe of the scale workaround",
    "mar-attributable-tmle": "the exact-equality probe of the scale workaround on both R paths",
    "stacked-mar-arm-indexed-cvtmle": "the exact-equality probe of the scale workaround",
}

#: The artefacts :func:`red_rows` reads, by the short names ``claims.ARTIFACTS`` uses.
FRAMES = {
    "performance": "performance-tests.csv",
    "equivalence": "equivalence.csv",
    "properties": "properties.csv",
}

#: The verdict columns of each table.  ``bool(nan)`` is ``True``, so a missing verdict read as
#: an object or float column would pass silently; :func:`check_verdict_columns` refuses it.
VERDICT_COLUMNS = {
    "performance": ("passed",),
    "equivalence": ("passed", "se_comparable", "reference_valid"),
    "properties": ("passed", "property_passed"),
}


def check_verdict_columns(tables: Mapping[str, pd.DataFrame]) -> None:
    """Refuse a non-empty verdict table whose verdict column is not boolean.

    Parameters
    ----------
    tables : Mapping
        Verdict tables, keyed as :data:`FRAMES` names them.

    Raises
    ------
    TypeError
        If a non-empty table lacks a verdict column, or holds one that is not ``bool``.
    """
    for name, columns in VERDICT_COLUMNS.items():
        frame = tables[name]
        if frame.empty:
            continue
        for column in columns:
            if column not in frame.columns:
                raise TypeError(f"the {name} table has no {column!r} column")
            if frame[column].dtype != bool:
                raise TypeError(
                    f"the {name} table's {column!r} column is {frame[column].dtype}, not bool; "
                    f"a missing verdict cannot be read as a pass or a failure"
                )


def frames(record: StudyRecord) -> dict[str, pd.DataFrame]:
    """Read the three committed verdict tables of one study.

    Parameters
    ----------
    record : StudyRecord
        The study to read.

    Returns
    -------
    dict
        Each verdict table, keyed by its short name in :data:`FRAMES`, and the failure
        accounting under ``accounting``, empty when the study recorded no failure.

    Raises
    ------
    TypeError
        If a verdict column is missing or not boolean, as :func:`check_verdict_columns` says.
    """
    tables = {name: pd.read_csv(record.artifact(filename)) for name, filename in FRAMES.items()}
    check_verdict_columns(tables)
    path = record.artifact(failures.ACCOUNTING)
    tables["accounting"] = (
        pd.read_csv(path, keep_default_na=False, na_values=[""])
        if path.exists()
        else pd.DataFrame(columns=list(failures.ACCOUNTING_COLUMNS))
    )
    return tables


def paired_legs(row: Any, margins: Margins, subject_valid: bool) -> ComparisonVerdict:
    """Rebuild a paired verdict from its committed endpoints.

    Each leg is the comparison :func:`tests.studies.evidence.comparison.equivalence` makes,
    against the study's declared margins, and the conclusion comes from the same
    :func:`~tests.studies.evidence.comparison.comparison_conclusion`.  Only the endpoints are
    read from the row.  The verdict columns are what this rebuilds, so they are not inputs.

    Parameters
    ----------
    row : Any
        One ``equivalence.csv`` row, as ``itertuples`` yields it.
    margins : Margins
        The study's declared margins.
    subject_valid : bool
        Whether the subject passes its own truth gate on this scenario and estimand.

    Returns
    -------
    ComparisonVerdict
        Every leg, and the verdict they produce.
    """
    committed = {
        "confidence_level": float(row.confidence_level),
        "rmse_noninferiority": float(row.rmse_noninferiority_margin),
        "coverage_noninferiority": float(row.coverage_noninferiority_margin),
    }
    current = {
        "confidence_level": margins.confidence_level,
        "rmse_noninferiority": margins.rmse_noninferiority,
        "coverage_noninferiority": margins.coverage_noninferiority,
    }
    if bool(row.se_comparable):
        committed["calibration_noninferiority"] = float(row.calibration_noninferiority_margin)
        current["calibration_noninferiority"] = margins.calibration_noninferiority
    drift = [
        name
        for name in committed
        if not math.isclose(committed[name], current[name], rel_tol=0.0, abs_tol=1e-15)
    ]
    if drift:
        raise ValueError(
            f"the committed comparison margins disagree with the registry for {', '.join(drift)}"
        )
    return comparison_verdict(
        paired_ci_lower=float(row.paired_ci_lower),
        paired_ci_upper=float(row.paired_ci_upper),
        mean_margin=float(row.mean_margin),
        rmse_ratio_upper=float(row.rmse_ratio_upper),
        rmse_noninferiority_margin=float(row.rmse_noninferiority_margin),
        coverage_difference_lower=float(row.coverage_difference_lower),
        coverage_noninferiority_margin=float(row.coverage_noninferiority_margin),
        se_comparable=bool(row.se_comparable),
        calibration_excess_upper=float(row.calibration_excess_upper),
        calibration_excess_resolution=float(row.calibration_excess_resolution),
        calibration_noninferiority_margin=float(row.calibration_noninferiority_margin),
        subject_valid=subject_valid,
    )


def _paired_measured(row: Any) -> str:
    text = (
        f"difference {render(float(row.paired_ci_lower))} to "
        f"{render(float(row.paired_ci_upper))} within {render(float(row.mean_margin))}, "
        f"RMSE ratio bound {render(float(row.rmse_ratio_upper))} "
        f"vs {render(float(row.rmse_noninferiority_margin))}, "
        f"coverage difference bound {render(float(row.coverage_difference_lower))} "
        f"vs {render(float(row.coverage_noninferiority_margin))}"
    )
    if bool(row.se_comparable):
        text += (
            f", calibration excess bound {render(float(row.calibration_excess_upper))} "
            f"vs {render(float(row.calibration_noninferiority_margin))}, "
            f"resolution {render(float(row.calibration_excess_resolution))}"
        )
    return text


def _truth_measured(row: Any) -> str:
    return (
        f"bias {render(float(row.bias_ci_lower))} to {render(float(row.bias_ci_upper))}, "
        f"coverage {render(float(row.coverage))}, SE ratio {render(float(row.se_ratio))}"
    )


def _failures_measured(accounting: pd.DataFrame, table: str, key: tuple[str, ...]) -> str:
    """``, failures k of R (<reason>)`` for an accounted cell, else nothing."""
    frame = accounting.loc[accounting["table"] == table]
    columns = (
        ["implementation", "scenario", "estimand"]
        if table == "performance"
        else [
            "scenario",
            "estimand",
        ]
    )
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(columns, key, strict=True):
        mask &= frame[column].astype(str) == value
    rows = frame.loc[mask]
    if rows.empty:
        return ""
    first = rows.iloc[0]
    return f", failures {int(first['failed'])} of {int(first['requested'])} ({first['reason']})"


def red_rows(record: StudyRecord, tables: Mapping[str, pd.DataFrame]) -> list[RedRow]:
    """Every red verdict one study's committed tables publish.

    Parameters
    ----------
    record : StudyRecord
        The study the tables belong to.
    tables : Mapping
        Its verdict tables, keyed as :func:`frames` returns them.

    Returns
    -------
    list of RedRow
        One row per red verdict, truth rows first, then paired, property, and accounting.

    Raises
    ------
    ValueError
        If a committed paired verdict disagrees with the verdict its legs produce.
    """
    out: list[RedRow] = []
    accounting = tables.get("accounting")
    if accounting is None:
        accounting = pd.DataFrame(columns=list(failures.ACCOUNTING_COLUMNS))
    truth_gates = failures.cell_gates(accounting, "performance")
    paired_gates = failures.cell_gates(accounting, "equivalence")
    performance = tables["performance"]
    for row in performance.itertuples():
        if bool(row.passed):
            continue
        if record.accepted_reference_failure and row.implementation == record.reference:
            continue
        key = (str(row.implementation), str(row.scenario), str(row.estimand))
        gate = truth_gates.get(key, (True, ""))
        # A cell red on its own legs is red by its truth gate, whatever its accounting says.
        own = all(
            bool(getattr(row, leg, True))
            for leg in ("bias_equivalent", "coverage_valid", "se_calibrated")
        )
        out.append(
            RedRow(
                RedKey(record.slug, "truth", "/".join(key)),
                str(row.implementation),
                "its truth gate" if gate[0] or not own else f"failure accounting: {gate[1]}",
                _truth_measured(row) + _failures_measured(accounting, "performance", key),
            )
        )

    valid = performance.set_index(["implementation", "scenario", "estimand"])["passed"]
    for row in tables["equivalence"].itertuples():
        subject_valid = bool(valid.loc[(record.implementation, row.scenario, row.estimand)])
        legs = paired_legs(row, record.margins, subject_valid)
        key = (str(row.scenario), str(row.estimand))
        gate = paired_gates.get(key, (True, failures.WITHIN_BOUND))
        # The accounting applies to the verdict the legs give, as ``failures.correct_paired``
        # does when it writes the row.
        conclusion, passed = failures.accounted_conclusion(
            legs.conclusion,
            equivalent=legs.similar and legs.rmse and legs.coverage and legs.calibration,
            gate=gate,
        )
        if (passed, conclusion) != (bool(row.passed), str(row.comparison_conclusion)):
            raise ValueError(
                f"{record.slug} commits {row.comparison_conclusion!r} (passed={row.passed}) for "
                f"{row.scenario}/{row.estimand}, but its legs conclude {conclusion!r} "
                f"(passed={passed})"
            )
        if passed:
            continue
        reasons = list(legs.failed())
        if legs.passed:
            reasons.append(f"failure accounting: {gate[1]}")
        out.append(
            RedRow(
                RedKey(record.slug, "paired", "/".join(key)),
                "paired",
                f"{conclusion}: {', '.join(reasons)}",
                _paired_measured(row) + _failures_measured(accounting, "equivalence", key),
            )
        )

    for row in tables["properties"].itertuples():
        if str(row.role) == property_verdicts.DIAGNOSTIC_ROLE:
            continue
        own, joint = bool(row.passed), bool(row.property_passed)
        if own and joint:
            continue
        out.append(
            RedRow(
                RedKey(record.slug, "property", f"{row.property}/{row.cell}"),
                str(row.role),
                "its own rule" if not own else "the family's joint clause",
                measured(row),
            )
        )

    hooks = accounting.loc[accounting["table"] == "hook"]
    for row in hooks.itertuples():
        if bool(row.gate_passed):
            continue
        out.append(
            RedRow(
                RedKey(record.slug, "accounting", f"hook/{row.leg}"),
                str(row.implementation),
                f"failure accounting: {row.reason}",
                # A hook with no ``account_failures`` counts no failing row, so the count is NaN.
                (
                    f"failing rows {render(float(row.complete_case))}, "
                    if math.isfinite(float(row.complete_case))
                    else "failing rows not counted, "
                )
                + f"failures {int(row.failed)} of {int(row.requested)}",
            )
        )
    return out


def verdict_count(tables: Mapping[str, pd.DataFrame]) -> int:
    """How many verdicts one study publishes, counted the way :func:`red_rows` reads them.

    Parameters
    ----------
    tables : Mapping
        The study's verdict tables, keyed as :func:`frames` returns them.

    Returns
    -------
    int
        Truth rows, paired rows, and property rows other than diagnostics.
    """
    properties = tables["properties"]
    scored = properties.loc[properties["role"] != property_verdicts.DIAGNOSTIC_ROLE]
    return len(tables["performance"]) + len(tables["equivalence"]) + len(scored)


def owner_ids(roadmap: Path = ROADMAP) -> list[str]:
    """The owner names the roadmap's "Red-cell owners" table declares.

    Parameters
    ----------
    roadmap : Path
        The roadmap to read.

    Returns
    -------
    list of str
        Each ``id`` cell, with its code quotes removed, in table order.
    """
    return [row["id"].strip("`") for row in pipe_table(roadmap, OWNER_TABLE)]


def findings(
    rows: Iterable[RedRow],
    owned: Mapping[RedKey, str],
    policies: Mapping[str, str],
    ids: Iterable[str],
) -> list[str]:
    """Every way the red rows, their owners, the roadmap and the registry disagree.

    Pure, so a test can hand it a deliberately broken input and watch it object.

    Parameters
    ----------
    rows : Iterable of RedRow
        The red rows the committed tables publish.
    owned : Mapping
        The declared owner of each red row.
    policies : Mapping
        Each study slug's registered publication policy.
    ids : Iterable of str
        The owner names the roadmap declares.

    Returns
    -------
    list of str
        One sentence per disagreement.  Empty when the ledger is consistent.
    """
    rows = list(rows)
    known = set(ids)
    red = {row.key for row in rows}
    out = [f"{key} is red and has no owner" for key in sorted(red - set(owned))]
    out += [f"{key} has an owner but is not red" for key in sorted(set(owned) - red)]
    out += [
        f"{key} is owned by {owner!r}, which the roadmap's id column does not list"
        for key, owner in sorted(owned.items())
        if owner not in known
    ]
    out += [
        f"{row.key} is red in a study registered as {policies.get(row.key.slug)!r}"
        for row in rows
        if policies.get(row.key.slug) != "reporting"
    ]
    return out


def _markdown_link(text: str, target: str) -> str:
    # Joined from parts because ``test_documentation_links`` reads every bracket-parenthesis
    # pair in a source file as a link, and a template's placeholder names no file.
    return "".join(("[", text, "]", "(", target, ")"))


def _link(record: StudyRecord) -> str:
    return _markdown_link(record.name, Path(record.document).name)


def _owner(owner: str) -> str:
    anchor = _OWNER_ANCHORS.get(owner, _OWNERS_SECTION)
    return _markdown_link(f"`{owner}`", f"../../roadmap.md#{anchor}")


def _ordered(
    rows: Iterable[RedRow], records: Sequence[StudyRecord]
) -> list[tuple[StudyRecord, RedRow]]:
    position = {record.slug: index for index, record in enumerate(records)}
    by_slug = {record.slug: record for record in records}
    return [
        (by_slug[row.key.slug], row)
        for row in sorted(
            rows,
            key=lambda row: (position[row.key.slug], KINDS.index(row.key.kind), row.key.key),
        )
    ]


def render_summary(
    rows: Sequence[RedRow],
    owned: Mapping[RedKey, str],
    records: Sequence[StudyRecord],
    ids: Sequence[str],
) -> list[str]:
    """One row per owner that holds a red row, in roadmap order, and a total.

    Parameters
    ----------
    rows : Sequence of RedRow
        The red rows.
    owned : Mapping
        The owner of each red row.
    records : Sequence of StudyRecord
        The registered studies, in grid order.
    ids : Sequence of str
        The roadmap's owner names, in table order.

    Returns
    -------
    list of str
        The table's lines.
    """
    ordered = _ordered(rows, records)
    table: list[tuple[str, str, str]] = []
    for owner in ids:
        held = [(record, row) for record, row in ordered if owned.get(row.key) == owner]
        if not held:
            continue
        names = dict.fromkeys(record.name for record, _ in held)
        table.append((_owner(owner), str(len(held)), ", ".join(names)))
    slugs = {row.key.slug for row in rows}
    table.append(("total", str(len(rows)), f"{len(slugs)} studies"))
    return _table(SUMMARY_COLUMNS, table)


def render_red_cells(
    rows: Sequence[RedRow], owned: Mapping[RedKey, str], records: Sequence[StudyRecord]
) -> list[str]:
    """One row per red verdict, in grid order.

    Parameters
    ----------
    rows : Sequence of RedRow
        The red rows.
    owned : Mapping
        The owner of each red row.
    records : Sequence of StudyRecord
        The registered studies, in grid order.

    Returns
    -------
    list of str
        The table's lines.
    """
    table = [
        (
            _link(record),
            row.key.kind,
            f"`{row.key.key}`",
            row.role,
            row.fails_by,
            row.measured,
            _owner(owned[row.key]) if row.key in owned else "**none**",
        )
        for record, row in _ordered(rows, records)
    ]
    return _table(RED_COLUMNS, table)


def render_reporting_without_red(
    rows: Sequence[RedRow],
    records: Sequence[StudyRecord],
    counts: Mapping[str, int],
) -> list[str]:
    """Every ``reporting`` study that publishes no red row.

    The policy check reads one way: a red row needs ``reporting``, and ``reporting`` needs no
    red row.  Listing these studies keeps that asymmetry visible instead of implied.

    Parameters
    ----------
    rows : Sequence of RedRow
        The red rows.
    records : Sequence of StudyRecord
        The registered studies, in grid order.
    counts : Mapping
        Each study slug's verdict count, from :func:`verdict_count`.

    Returns
    -------
    list of str
        The table's lines.
    """
    red = {row.key.slug for row in rows}
    table = [
        (_link(record), str(counts[record.slug]), "0")
        for record in records
        if record.publication_policy == "reporting" and record.slug not in red
    ]
    return _table(QUIET_COLUMNS, table)


def rendered(records: Sequence[StudyRecord] | None = None) -> dict[str, list[str]]:
    """Render all three blocks from the committed tables.

    Parameters
    ----------
    records : Sequence of StudyRecord, optional
        The studies to read.  Defaults to every registered study.

    Returns
    -------
    dict
        Each block's lines, keyed by its name in :data:`BLOCKS`.
    """
    records = registered() if records is None else records
    rows: list[RedRow] = []
    counts: dict[str, int] = {}
    for record in records:
        tables = frames(record)
        rows += red_rows(record, tables)
        counts[record.slug] = verdict_count(tables)
    ids = owner_ids()
    return {
        "red-cell-summary": render_summary(rows, OWNERS, records, ids),
        "red-cells": render_red_cells(rows, OWNERS, records),
        "reporting-without-red": render_reporting_without_red(rows, records, counts),
    }


def published(page: Path = PAGE) -> dict[str, list[list[str]]]:
    """Every generated block the page carries, by name, as each occurrence's lines.

    Parameters
    ----------
    page : Path
        The ledger page.

    Returns
    -------
    dict
        Block name to one list of lines per occurrence of that block.

    Raises
    ------
    LookupError
        If a block is opened and never closed.
    """
    lines = page.read_text(encoding="utf-8").splitlines()
    out: dict[str, list[list[str]]] = {}
    for name in BLOCKS:
        opening = _OPEN.format(name=name)
        for head, line in enumerate(lines):
            if line.strip() != opening:
                continue
            tail = next(
                (index for index in range(head + 1, len(lines)) if lines[index].strip() == _CLOSE),
                None,
            )
            if tail is None:
                raise LookupError(f"{opening} is never closed by {_CLOSE}")
            out.setdefault(name, []).append(lines[head + 1 : tail])
    return out


def fill(page: Path = PAGE) -> list[str]:
    """Rewrite the ledger page's generated blocks from the committed tables.

    Parameters
    ----------
    page : Path
        The ledger page.

    Returns
    -------
    list of str
        One line per block whose content changed.

    Raises
    ------
    LookupError
        If the page does not carry each block exactly once.
    """
    blocks = rendered()
    lines = page.read_text(encoding="utf-8").splitlines(keepends=True)
    changed: list[str] = []
    for name, block in blocks.items():
        opening = _OPEN.format(name=name)
        heads = [index for index, line in enumerate(lines) if line.strip() == opening]
        if len(heads) != 1:
            raise LookupError(f"{page.name} carries {opening} {len(heads)} times, not once")
        head = heads[0]
        tail = next(
            (index for index in range(head + 1, len(lines)) if lines[index].strip() == _CLOSE),
            None,
        )
        if tail is None:
            raise LookupError(f"{opening} is never closed by {_CLOSE}")
        replacement = [line + "\n" for line in block]
        if lines[head + 1 : tail] != replacement:
            changed.append(f"{name}: {len(block)} line(s)")
        lines[head + 1 : tail] = replacement
    write_lines(page, "".join(lines))
    return changed


def report() -> None:
    """Rewrite the ledger page and print what changed."""
    changed = fill()
    print(f"red-cell ledger: {len(changed)} block(s) updated in {PAGE.relative_to(ROOT)}")
    for line in changed:
        print(f"  {line}")


def main() -> None:
    """Parse no arguments, so ``--help`` describes the module, and rewrite the page."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()


if __name__ == "__main__":
    main()
