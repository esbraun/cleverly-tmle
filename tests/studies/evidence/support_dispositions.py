"""The disposition of each positive red row of the validation ledger (plan section 6).

A support row licenses a kind of inference only where its study cells are green. A positive red
row is a cell that fails: a property row with the role ``positive``, a truth row of the
``cleverly`` implementation, or a paired row. Each such row needs one decision, because the
row that its study licenses cannot claim the kind that the red cell measures.

This module holds the decision table and builds its draft. ``DISPOSITIONS`` holds one decision
per positive red row, and ``DECISIONS`` holds the owner's answer for each flagged line.
``python -m tests.studies.evidence.support_dispositions --draft`` prints the draft.

Examples
--------
    python -m tests.studies.evidence.support_dispositions --draft
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.studies.evidence.red_cells import RedKey, RedRow

__all__ = [
    "DECISIONS",
    "DISPOSITIONS",
    "NO_SHORTFALL",
    "ROW_STUDY",
    "Disposition",
    "DraftEntry",
    "PositiveRow",
    "draft_entries",
    "kinds_of",
    "main",
    "positive_red_rows",
    "render_draft",
]

Choice = Literal["supported with a stated limit", "experimental", "withheld (status)"]

SUPPORTED_WITH_LIMIT: Choice = "supported with a stated limit"
EXPERIMENTAL: Choice = "experimental"
WITHHELD: Choice = "withheld (status)"

#: The owners whose red rows default to a stated limit: the owner reads a finite-sample effect
#: and no defect, with the measured numbers (plan section 6.2).
LIMIT_OWNERS = frozenset(
    {
        "band-finite-sample",
        "strata-boundary-mean",
        "X13-finite-sample",
        "mtp-point-calibration",
        "mtp-longitudinal-limits",
        "F1-power-design",
        "X4-finite-sample",
        "F4-calibration-draws",
        "composite-high-arm",
        "RM18-ordinary-weighted",
        "RM18-fixed-weights",
        "RM18-boundary",
        "X8-identity-small-stratum",
        "X8-drtmle-small-stratum",
        "X16-plugin-variance",
        "RM18-comparator-density",
        "X19-competing-calibration",
        "X45-finite-sample",
        "mtp-pricing-learned-density",
        "X26-plugin-conservative",
        "X26-refusal-limited",
        "adaptive-iptw-finite-sample",
        "adaptive-iptw-composite-l3",
    }
)
#: The owners whose red rows default to experimental, and the owner decides (flagged).
FLAGGED_EXPERIMENTAL = {
    "X8-logit-small-stratum": "the logit slope at a small stratum reads an SE ratio of 0.78 to 0.94",
    "X8-drtmle-one-sided-bias": "the one-sided bias of the DR-TMLE cell",
    "RM18-one-sided-bias": "the one-sided bias of the DR-TMLE cell",
    "F28": "the informative-size t cells read a bias of 0.11 to 0.13 and an SE ratio of 0.83",
    "F27": "the boundary truth rows read a coverage of 0.894 and an SE ratio of 0.82",
    "mtp-pricing-calibration": (
        "X40: the five-node price_down rows read a coverage of 0.916 to 0.923 and an SE ratio of "
        "0.93 to 0.95"
    ),
    "X40-sdr-heavy-ratios": (
        "X40: SDR under the two-branch plan p4 reads an SE ratio of 0.9408 and a coverage of 0.9357 "
        "at n = 4,000"
    ),
}
#: The studies whose interval calibration cells use a t reference, so a Wald row there is ``wald_t``.
T_REFERENCE_SLUGS = frozenset({"clustered-few-cluster-tmle", "few-cluster-cross-fitted-ltmle"})

_POINT_FAMILIES = frozenset(
    {
        "bias",
        "double_robustness",
        "double_robust_contraction",
        "sequential_double_robustness",
        "targeting_necessity",
        "selector_necessity",
        "generated_design",
    }
)
_THEORY_LEGS = {"bias_equivalent": "point", "coverage_valid": "wald", "se_calibrated": "wald"}


@dataclass(frozen=True)
class Disposition:
    """The decision about one positive red row.

    Parameters
    ----------
    row : str
        The support row the decision applies to. The study of the red row is in its evidence.
    kinds : tuple of str
        The kinds of inference the red row measures, as :data:`cleverly._support.Kind`.
    disposition : {"supported with a stated limit", "experimental", "withheld (status)"}
        What the row licenses for those kinds.
    limit : str
        The limit printed in the row note. Empty unless the disposition is a stated limit.
    flag : str, default ""
        When non-empty, the owner decides the line (Q6).
    """

    row: str
    kinds: tuple[str, ...]
    disposition: Choice
    limit: str
    flag: str = ""


@dataclass(frozen=True)
class PositiveRow:
    """A positive red row with what the committed tables say about its failed legs.

    Parameters
    ----------
    red : RedRow
        The red verdict.
    owner : str
        The owner of the red row in the red-cell ledger, or an empty text.
    failed_legs : tuple of str, default ()
        For a truth row, the legs of the performance table that failed.
    """

    red: RedRow
    owner: str = ""
    failed_legs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DraftEntry:
    """One line of the draft: the default decision and the rows it could apply to.

    Parameters
    ----------
    key : RedKey
        The red row.
    kinds : tuple of str
        The kinds the red row measures.
    disposition : str
        The default disposition.
    limit : str
        The default limit text.
    flag : str
        The reason the owner decides, or an empty text.
    candidates : tuple of str
        The rows whose evidence names the study and whose kinds include the red row's.
    owner : str
        The owner of the red row.
    """

    key: RedKey
    kinds: tuple[str, ...]
    disposition: str
    limit: str
    flag: str
    candidates: tuple[str, ...]
    owner: str


#: One line per positive red row: study, kind of red row, key, the support row the decision
#: applies to, the kinds of that row it concerns, the disposition, the limit, and the owner flag.
#: The owner picked each row by hand from the rows that name the study. A red row whose kind no
#: row of the study lists gets ``experimental``, because a fit that reads that kind from the row
#: is already experimental (``support_for``), and the line names the nearest row.
LIMIT = SUPPORTED_WITH_LIMIT
_TABLE: tuple[tuple[str, str, str, str, tuple[str, ...], Choice, str, str], ...] = (
    (
        "canonical-stratified-tmle",
        "truth",
        "cleverly-stratified-tmle/stratified_binary/ey[1][V=2]",
        "T13",
        ("wald",),
        LIMIT,
        "strata-boundary-mean: bias -0.0027 to 0.000813, coverage 0.9170, SE ratio 0.9850",
        "",
    ),
    (
        "canonical-stratified-tmle",
        "property",
        "interval_calibration/v2_ey1__correctly_specified",
        "T13",
        ("wald",),
        LIMIT,
        "strata-boundary-mean: coverage 0.9056 to 0.9344, SE ratio 0.9315 to 1.0094, empirical efficiency ratio 0.9753 to 1.0552, reported efficiency ratio 0.9742 to 0.9931",
        "",
    ),
    (
        "canonical-stratified-tmle",
        "property",
        "simultaneous_coverage/crossfit_strata__simultaneous_band",
        "T13",
        ("band",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "canonical-stratified-tmle",
        "property",
        "simultaneous_coverage/strata__simultaneous_band",
        "T13",
        ("band",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "canonical-stratified-incremental-msm",
        "property",
        "interval_calibration/logit_v1_a__correctly_specified",
        "T16.logit_binary",
        ("wald",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-stratified-incremental-msm",
        "property",
        "interval_calibration/logit_v2_a__correctly_specified",
        "T16.logit_binary",
        ("wald",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-stratified-msm-identity",
        "paired",
        "stratified_msm_identity/msm[W][V=2]",
        "T17",
        ("wald",),
        LIMIT,
        "X8-identity-small-stratum: difference -0.000049 to 0.000037 within 0.000694, RMSE ratio bound 1.0047 vs 1.1000, coverage difference bound -0.0220 vs -0.0250, calibration excess bound 0.0523 vs 0.0500, resolution 0.0476",
        "",
    ),
    (
        "canonical-stratified-msm-identity",
        "property",
        "interval_calibration/identity_v2_a__correctly_specified",
        "T17",
        ("wald",),
        LIMIT,
        "X8-identity-small-stratum: coverage 0.9233 to 0.9515, SE ratio 0.9264 to 1.0042, empirical efficiency ratio 0.9925 to 1.0766, reported efficiency ratio 0.9937 to 1.0005",
        "",
    ),
    (
        "canonical-stratified-drtmle",
        "truth",
        "cleverly-stratified-drtmle/stratified_both_correct/ey[1][V=2]",
        "D03",
        ("wald",),
        LIMIT,
        "X8-drtmle-small-stratum: bias 0.0023 to 0.0112, coverage 0.9000, SE ratio 0.9388",
        "",
    ),
    (
        "canonical-stratified-drtmle",
        "property",
        "double_robustness/marginal_ate__treatment_correct",
        "D03",
        ("point",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-stratified-drtmle",
        "property",
        "double_robustness/v0_ate__treatment_correct",
        "D03",
        ("point",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-stratified-drtmle",
        "property",
        "double_robustness/v1_ate__treatment_correct",
        "D03",
        ("point",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-stratified-drtmle",
        "property",
        "double_robustness/v2_ate__treatment_correct",
        "D03",
        ("point",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-stratified-drtmle",
        "property",
        "interval_calibration/v2_ate__correctly_specified",
        "D03",
        ("wald",),
        LIMIT,
        "X8-drtmle-small-stratum: coverage 0.9069 to 0.9380, SE ratio 0.9294 to 1.0112, empirical efficiency ratio 1.0348 to 1.1237, reported efficiency ratio 1.0296 to 1.0653",
        "",
    ),
    (
        "clustered-unequal-cvtmle",
        "property",
        "clustered_inference/cluster_robust_fold_evaluated_cluster_covariate",
        "T10.fold_evaluated",
        ("wald",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "clustered-few-cluster-tmle",
        "property",
        "few_cluster_reference/tmle_cv_evaluation__unequal_informative__j10__t_reference",
        "T11.fold_evaluated",
        ("wald_t",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "clustered-few-cluster-tmle",
        "property",
        "few_cluster_reference/tmle_cv_evaluation__unequal_informative__j20__t_reference",
        "T11.fold_evaluated",
        ("wald_t",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "clustered-few-cluster-tmle",
        "property",
        "few_cluster_reference/tmle_cv_evaluation__unequal_informative__j30__t_reference",
        "T11.fold_evaluated",
        ("wald_t",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "clustered-cross-fitted-ltmle",
        "property",
        "simultaneous_coverage/clustered_regimens__simultaneous_band",
        "G05.cross_fitted",
        ("band",),
        LIMIT,
        "band-finite-sample: joint coverage 0.9191 to 0.9458",
        "",
    ),
    (
        "canonical-ctmle-selector",
        "property",
        "selector_necessity/collaborative",
        "C01.greedy",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-ctmle-selector",
        "property",
        "type_i_error/sharp_null",
        "C01.greedy",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-multi-arm-ctmle-selector",
        "property",
        "interval_calibration/correctly_specified",
        "C02.greedy",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-multi-arm-ctmle-selector",
        "property",
        "root_n_and_efficiency/n_500",
        "C02.greedy",
        ("point",),
        LIMIT,
        "RM18-boundary: bias 0.0010, coverage 0.8965 to 0.9627, SE ratio 0.9892",
        "",
    ),
    (
        "canonical-multi-arm-ctmle-selector",
        "property",
        "selector_necessity/discrete",
        "C02.greedy",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-multi-arm-ctmle-selector",
        "property",
        "selector_necessity/greedy",
        "C02.greedy",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-multi-arm-ctmle-selector",
        "property",
        "selector_necessity/ordered",
        "C02.greedy",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-multi-arm-ctmle-selector",
        "property",
        "type_i_error/sharp_null",
        "C02.greedy",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-ctmle-sequences",
        "truth",
        "cleverly-ctmle-sequences/binary_lasso/ate",
        "C09.lasso",
        ("wald",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "canonical-ctmle-sequences",
        "property",
        "selector_necessity/lasso_collaborative",
        "C09.lasso",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-ctmle-sequences",
        "property",
        "selector_necessity/learners_collaborative",
        "C09.lasso",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-multi-arm-ctmle-oat",
        "property",
        "generated_design/estimated",
        "C04",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-multi-arm-ctmle-oat",
        "property",
        "generated_design/oracle_design",
        "C04",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "canonical-drtmle",
        "property",
        "double_robust_contraction/treatment_correct_n1500",
        "D01",
        ("point",),
        LIMIT,
        "RM18-boundary: coverage 0.8864 to 0.9385, bias 0.0115",
        "",
    ),
    (
        "canonical-drtmle",
        "property",
        "double_robustness/outcome_correct",
        "D01",
        ("point",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-drtmle",
        "property",
        "double_robustness/treatment_correct",
        "D01",
        ("point",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-multi-arm-drtmle",
        "property",
        "double_robust_contraction/outcome_correct_n4000",
        "D02",
        ("point",),
        LIMIT,
        "RM18-boundary: coverage 0.8988 to 0.9542, bias 0.000900",
        "",
    ),
    (
        "canonical-multi-arm-drtmle",
        "property",
        "double_robustness/treatment_correct",
        "D02",
        ("point",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "canonical-multi-arm-drtmle",
        "property",
        "interval_calibration/correctly_specified",
        "D02",
        ("wald",),
        LIMIT,
        "RM18-boundary: coverage 0.9300 to 0.9597, SE ratio 0.9258 to 1.0123",
        "",
    ),
    (
        "canonical-multi-arm-drtmle",
        "property",
        "root_n_and_efficiency/n_500",
        "D02",
        ("point", "wald"),
        LIMIT,
        "RM18-boundary: bias -0.000902, coverage 0.8965 to 0.9627, SE ratio 0.9888",
        "",
    ),
    (
        "multi-arm-mar-drtmle",
        "property",
        "interval_calibration/ate__correctly_specified",
        "D09",
        ("wald",),
        LIMIT,
        "F4-calibration-draws: coverage 0.9287 to 0.9537, SE ratio 0.9294 to 1.0000, empirical efficiency ratio 0.9994 to 1.0751, reported efficiency ratio 0.9980 to 1.0008",
        "",
    ),
    (
        "multi-arm-mar-drtmle",
        "property",
        "simultaneous_coverage/arms__simultaneous_band",
        "D09",
        ("band",),
        LIMIT,
        "F4-calibration-draws: joint coverage 0.9196 to 0.9462",
        "",
    ),
    (
        "composite-missing-drtmle",
        "truth",
        "cleverly-composite-drtmle/three_arm_mar_outcome_and_treatment/ey[high]",
        "D10.outcome",
        ("wald",),
        LIMIT,
        "composite-high-arm: bias -0.0054 to 0.0013, coverage 0.9225, SE ratio 0.9342",
        "",
    ),
    (
        "learned-rule-cvtmle-boundary",
        "truth",
        "cleverly-learned-rule-cvtmle/exceptional/ey_learned_rule[learned rule]",
        "T24",
        ("wald",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "learned-rule-cvtmle-boundary",
        "truth",
        "cleverly-learned-rule-cvtmle/weak_blip/ey_learned_rule[learned rule]",
        "T24",
        ("wald",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "shift-policies",
        "paired",
        "continuous_modified_policy/ate_policy[+0.25 vs natural course]",
        "T21",
        ("wald",),
        LIMIT,
        "RM18-comparator-density: difference 0.000061 to 0.000361 within 0.000770, RMSE ratio bound 1.0587 vs 1.1000, coverage difference bound -0.0050 vs -0.0250, calibration excess bound 0.0648 vs 0.0500, resolution 0.0439",
        "",
    ),
    (
        "policy-point-mtp",
        "property",
        "interval_calibration/halve__correctly_specified",
        "T22",
        ("wald",),
        LIMIT,
        "mtp-point-calibration: coverage 0.9479 to 0.9709, SE ratio 1.0011 to 1.0881, empirical efficiency ratio 0.9208 to 1.0003, reported efficiency ratio 1.0003 to 1.0034",
        "",
    ),
    (
        "weighted-ltmle",
        "property",
        "interval_calibration/static__correctly_specified",
        "G03",
        ("wald",),
        LIMIT,
        "RM18-ordinary-weighted: coverage 0.9187 to 0.9454, SE ratio 0.9263 to 0.9985, empirical efficiency ratio 0.9855 to 1.0615, reported efficiency ratio 0.9783 to 0.9899",
        "",
    ),
    (
        "weighted-ltmle",
        "property",
        "targeting_necessity/dynamic__targeted",
        "G03",
        ("point",),
        LIMIT,
        "RM18-ordinary-weighted: bias -0.0025 to 0.0019, margin 0.0075",
        "",
    ),
    (
        "weighted-ltmle",
        "property",
        "targeting_necessity/static__targeted",
        "G03",
        ("point",),
        LIMIT,
        "RM18-ordinary-weighted: bias -0.0027 to 0.0075, margin 0.0172",
        "",
    ),
    (
        "weighted-ltmle",
        "property",
        "type_i_error/static__sharp_null",
        "G03",
        ("wald",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "stochastic-categorical-ltmle",
        "property",
        "power/mix__alternative",
        "G08.in_sample",
        ("wald",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "longitudinal-mtp",
        "property",
        "interval_calibration/categorical_mtp__correctly_specified",
        "G10.continuous",
        ("wald",),
        LIMIT,
        "mtp-longitudinal-limits: coverage 0.9289 to 0.9560, SE ratio 0.9432 to 1.0214, empirical efficiency ratio 1.0746 to 1.1643, reported efficiency ratio 1.0948 to 1.1006",
        "",
    ),
    (
        "longitudinal-sdr",
        "property",
        "sequential_double_robustness/both_correct__n_2000",
        "G23.in_sample",
        ("point",),
        LIMIT,
        "X4-finite-sample: bias 0.000011 to 0.0112, margin 0.0172, coverage 0.8853 to 0.9327, SE ratio 0.9459",
        "",
    ),
    (
        "longitudinal-sdr",
        "property",
        "simultaneous_coverage/six_regimens__simultaneous_band",
        "G23.in_sample",
        ("band",),
        LIMIT,
        "band-finite-sample: joint coverage 0.9085 to 0.9394",
        "",
    ),
    (
        "weighted-ltmle-crossfit",
        "paired",
        "selected_censored_end_of_study/ey_regimen[always]",
        "G04",
        ("wald",),
        LIMIT,
        "RM18-boundary: difference -0.000150 to 0.000202 within 0.0031, RMSE ratio bound 1.0162 vs 1.1000, coverage difference bound -0.0475 vs -0.0250, calibration excess bound 0.1050 vs 0.0500, resolution 0.1121",
        "",
    ),
    (
        "weighted-ltmle-crossfit",
        "paired",
        "selected_censored_end_of_study/ey_regimen[never]",
        "G04",
        ("wald",),
        LIMIT,
        "RM18-fixed-weights: difference -0.000350 to 0.000263 within 0.0047, RMSE ratio bound 1.0245 vs 1.1000, coverage difference bound -0.0300 vs -0.0250, calibration excess bound 0.0465 vs 0.0500, resolution 0.0662",
        "",
    ),
    (
        "weighted-ltmle-crossfit",
        "paired",
        "selected_censored_end_of_study/ey_regimen[treat then continue if l2 positive]",
        "G04",
        ("wald",),
        LIMIT,
        "RM18-boundary: difference -0.000118 to 0.000400 within 0.0032, RMSE ratio bound 1.0175 vs 1.1000, coverage difference bound -0.0425 vs -0.0250, calibration excess bound 0.0382 vs 0.0500, resolution 0.1119",
        "",
    ),
    (
        "weighted-ltmle-crossfit",
        "property",
        "interval_calibration/static__correctly_specified",
        "G04",
        ("wald",),
        LIMIT,
        "RM18-fixed-weights: coverage 0.9232 to 0.9492, SE ratio 0.9430 to 1.0203, empirical efficiency ratio 1.0195 to 1.1006, reported efficiency ratio 1.0333 to 1.0442",
        "",
    ),
    (
        "canonical-ltmle-survival",
        "property",
        "simultaneous_coverage/all_reported__simultaneous_band",
        "G11",
        ("band",),
        LIMIT,
        "band-finite-sample: joint coverage 0.9140 to 0.9283",
        "",
    ),
    (
        "point-treatment-survival",
        "property",
        "interval_calibration/weighted_t5__correctly_specified",
        "G14.survival",
        ("wald",),
        LIMIT,
        "X13-finite-sample: coverage 0.9421 to 0.9689, SE ratio 0.9807 to 1.0762, empirical efficiency ratio 0.9300 to 1.0204, reported efficiency ratio 1.0000 to 1.0015",
        "",
    ),
    (
        "point-treatment-survival-crossfit",
        "property",
        "interval_calibration/three_arm_t3__correctly_specified",
        "G17.survival",
        ("wald",),
        LIMIT,
        "X13-finite-sample: coverage 0.9421 to 0.9689, SE ratio 0.9650 to 1.0560, empirical efficiency ratio 1.0101 to 1.1085, reported efficiency ratio 1.0592 to 1.0764",
        "",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/binary_active/ate",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/binary_active/ey0",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/binary_active/ey1",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/binary_active/or",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/binary_active/paf",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/binary_active/par",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/binary_active/rr",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/ate[1 vs 0]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/ate[2 vs 0]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/ey[1]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/ey[2]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/or[1 vs 0]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/or[2 vs 0]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/rr[1 vs 0]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_active/rr[2 vs 0]",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "generated_design/binary_active__estimated",
        "C05.binary",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "generated_design/binary_active__oracle_design",
        "C05.binary",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "generated_design/three_arm_active__estimated",
        "C05.binary",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "generated_design/three_arm_active__oracle_design",
        "C05.binary",
        ("point",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "interval_calibration/binary_active__correctly_specified",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "interval_calibration/binary_repeats__correctly_specified",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "interval_calibration/binary_weighted__correctly_specified",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "interval_calibration/three_arm_active__correctly_specified",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "root_n_and_efficiency/n_2000",
        "C05.binary",
        ("point", "wald"),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "root_n_and_efficiency/n_500",
        "C05.binary",
        ("point", "wald"),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "root_n_and_efficiency/n_8000",
        "C05.binary",
        ("point", "wald"),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "simultaneous_coverage/three_arm_active__simultaneous_band",
        "C05.binary",
        ("band",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm",
        "property",
        "type_i_error/sharp_null",
        "C05.binary",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_outcome/ate",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_outcome/ey0",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_outcome/ey1",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_outcome/or",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_outcome/rr",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_treatment/ate",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_treatment/ey0",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_treatment/ey1",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_treatment/or",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/missing_treatment/rr",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/strata/ate",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/strata/ate[S=0]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/strata/ey0",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/strata/ey1",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_missing/ate[1 vs 0]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_missing/ate[2 vs 0]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_missing/ey[1]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_missing/ey[2]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_missing/or[1 vs 0]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_missing/or[2 vs 0]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "truth",
        "cleverly-ctmle-oat-per-arm/three_arm_missing/rr[1 vs 0]",
        "C06.strata",
        ("wald",),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "property",
        "root_n_and_efficiency/n_2000",
        "C06.strata",
        ("point", "wald"),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "property",
        "root_n_and_efficiency/n_4000",
        "C06.strata",
        ("point", "wald"),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "ctmle-oat-per-arm-strata-missing",
        "property",
        "root_n_and_efficiency/n_8000",
        "C06.strata",
        ("point", "wald"),
        WITHHELD,
        "",
        "Q1",
    ),
    (
        "default-simultaneous-bands",
        "property",
        "simultaneous_coverage/categorical_ltmle__simultaneous_band",
        "T02.binary",
        ("band",),
        LIMIT,
        "band-finite-sample: joint coverage 0.9101 to 0.9382",
        "",
    ),
    (
        "default-simultaneous-bands",
        "property",
        "simultaneous_coverage/categorical_ltmle_crossfit__simultaneous_band",
        "T02.binary",
        ("band",),
        LIMIT,
        "band-finite-sample: joint coverage 0.9173 to 0.9443",
        "",
    ),
    (
        "default-simultaneous-bands",
        "property",
        "simultaneous_coverage/cde_z0__simultaneous_band",
        "T02.binary",
        ("band",),
        LIMIT,
        "band-finite-sample: joint coverage 0.9015 to 0.9309",
        "",
    ),
    (
        "full-refit-bootstrap-and-derived-contrasts",
        "property",
        "interval_calibration/boot_ate_clustered__correctly_specified",
        "T01.binary",
        ("bootstrap",),
        EXPERIMENTAL,
        "",
        "Q6",
    ),
    (
        "full-refit-bootstrap-and-derived-contrasts",
        "property",
        "interval_calibration/boot_ate_crossfit__correctly_specified",
        "T01.binary",
        ("bootstrap",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "full-refit-bootstrap-and-derived-contrasts",
        "property",
        "interval_calibration/boot_ey_crossfit__correctly_specified",
        "T01.binary",
        ("bootstrap",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "parametric-gformula",
        "property",
        "interval_calibration/nice_g1_treat_all__correctly_specified",
        "G29.g1",
        ("wald",),
        LIMIT,
        "X45-finite-sample: coverage 0.9278 to 0.9529, SE ratio 0.9259 to 1.0000",
        "",
    ),
    (
        "longitudinal-incremental-dropout",
        "property",
        "interval_calibration/competing_contrast_x2_h3__correctly_specified",
        "G34.competing",
        ("wald",),
        LIMIT,
        "X19-competing-calibration: coverage 0.9491 to 0.9681, SE ratio 1.0124 to 1.0830, empirical efficiency ratio 0.9239 to 0.9884, reported efficiency ratio 1.0005 to 1.0007",
        "",
    ),
)

#: X40 (continuous-policy validation). The owner chose ``experimental`` with the measured number
#: (the RM42b measured-shortfall standard) for the four red price_down cells of the five-node
#: price study on 2026-10-16. To flip it to a stated
#: limit, set this to ``LIMIT``. Then move each sentence of
#: ``SHORTFALLS`` for ``G39.ltmle.tau5`` and ``G39.sdr.tau5`` in ``src/cleverly/_support_rows.py``
#: to ``MEASURED_LIMITS`` under the same row id, with the text of ``_PRICE_DOWN_LIMIT``.
_X40_PRICE_DOWN: Choice = EXPERIMENTAL
_PRICE_DOWN_LIMIT = {
    "G39.ltmle.tau5": "mtp-pricing-calibration: coverage 0.9160 to 0.9170, SE ratio 0.9310 to 0.9479",
    "G39.sdr.tau5": "mtp-pricing-calibration: coverage 0.9190 to 0.9230, SE ratio 0.9339 to 0.9518",
}
_X40_PRICING: tuple[tuple[str, str, str, str, tuple[str, ...], Choice, str, str], ...] = (
    *(
        (
            "mtp-pricing",
            "truth",
            f"cleverly-mtp-pricing/{scenario}/{estimand}",
            row,
            ("wald",),
            _X40_PRICE_DOWN,
            _PRICE_DOWN_LIMIT[row] if _X40_PRICE_DOWN == LIMIT else "",
            "X40",
        )
        for scenario, row in (
            ("pricing_ltmle_tau5", "G39.ltmle.tau5"),
            ("pricing_sdr_tau5", "G39.sdr.tau5"),
        )
        for estimand in ("ate_regimen[price_down vs natural_prices]", "ey_regimen[price_down]")
    ),
    # The learned-density arms read conservative intervals and a coverage above nominal, so the
    # decision is a stated limit.
    (
        "mtp-pricing",
        "property",
        "interval_calibration/up_learned_density_tau2__correctly_specified",
        "G10.continuous",
        ("wald",),
        LIMIT,
        "mtp-pricing-learned-density: coverage 0.9530 to 0.9747, SE ratio 1.0265 to 1.1162",
        "",
    ),
    (
        "mtp-pricing",
        "property",
        "interval_calibration/up_learned_density_tau3__correctly_specified",
        "G40.tau3",
        ("wald",),
        LIMIT,
        "mtp-pricing-learned-density: coverage 0.9473 to 0.9704, SE ratio 1.0493 to 1.1408",
        "",
    ),
    (
        "mtp-pricing",
        "property",
        "interval_calibration/up_learned_density_tau5__correctly_specified",
        "G40.tau5",
        ("wald",),
        LIMIT,
        "mtp-pricing-learned-density: coverage 0.9485 to 0.9713, SE ratio 1.0428 to 1.1323",
        "",
    ),
)

#: X40 S1 (``mtp-boundaries``). The ``p4_sdr`` calibration cell is a declared ``reporting`` arm
#: at n = 4,000. The ``boundaries_ltmle`` p4 truth rows pass at the n = 5,000 primary, so they
#: have no line here.
_X40_BOUNDARIES: tuple[tuple[str, str, str, str, tuple[str, ...], Choice, str, str], ...] = (
    (
        "mtp-boundaries",
        "property",
        "interval_calibration/p4_sdr__correctly_specified",
        "G24.mtp.in_sample",
        ("wald",),
        EXPERIMENTAL,
        "",
        "X40",
    ),
)

#: X26 (``omitted-variable-bound-plugin-se``). Both red cells are lower bounds on the cross-fitted
#: three-arm fit. The owners read a finite-sample effect and no defect: ``X26-plugin-conservative``
#: reads an interval above nominal, and ``X26-refusal-limited`` reads an interval inside the band
#: that only the worst-case fill of 32 lost replicates puts outside it. Each decision is a stated
#: limit with the measured number.
_X26: tuple[tuple[str, str, str, str, tuple[str, ...], Choice, str, str], ...] = (
    (
        "omitted-variable-bound-plugin-se",
        "property",
        "interval_calibration/multinomial_logistic_crossfit_ate_2v0_lower__correctly_specified",
        "T36.three_arm_stacked",
        ("wald",),
        LIMIT,
        "X26-plugin-conservative: coverage 0.9556 to 0.9657, SE ratio 1.0332 to 1.0723, "
        "failures 32 of 10000 (complete case fails)",
        "",
    ),
    (
        "omitted-variable-bound-plugin-se",
        "property",
        "interval_calibration/multinomial_logistic_crossfit_att_2v0_lower__correctly_specified",
        "T36.three_arm_stacked",
        ("wald",),
        LIMIT,
        "X26-refusal-limited: coverage 0.9568 to 0.9667, SE ratio 1.0288 to 1.0683, "
        "failures 32 of 10000 (worst case fails)",
        "",
    ),
)

#: The red cells of ``adaptive-iptw`` (X21). The three flexible-learner cells are experimental. A
#: gradient-boosting mechanism is outside the configuration of the row: the key reads the learner
#: family, so a boosted fit matches no row and no shortfall labels the row (``NO_SHORTFALL``). The
#: other cells read finite-sample or draw-specific effects with no defect found, so each leaves its
#: row supported with a stated limit.
_ADAPTIVE_IPTW: tuple[tuple[str, str, str, str, tuple[str, ...], Choice, str, str], ...] = (
    (
        "adaptive-iptw",
        "property",
        "bias/ate__flexible",
        "AI.points.cross_fit",
        ("point",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "bias/ate__rate",
        "AI.points.cross_fit",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias 0.000882 to 0.0046, margin 0.0044",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "bias/ey0__flexible",
        "AI.points.cross_fit",
        ("point",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "bias/ey1__flexible",
        "AI.points.cross_fit",
        ("point",),
        EXPERIMENTAL,
        "",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "bias/ey_low__composite_l3",
        "AI.points_composite_three_arm.cross_fit",
        ("point",),
        LIMIT,
        "adaptive-iptw-composite-l3: bias -0.0075 to -0.000620, margin 0.0066",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "interval_calibration/ey0__known_clustered__correctly_specified",
        "AI.known_clustered",
        ("wald",),
        LIMIT,
        "adaptive-iptw-finite-sample: coverage 0.9394 to 0.9643, SE ratio 0.9902 to 1.0722",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "targeting_necessity/cn_ey_obs__targeted",
        "AI.points_missing_outcome.cross_fit",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias -0.0031 to 0.0014, margin 0.0043",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "targeting_necessity/p_ate__targeted",
        "AI.points.cross_fit",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias 0.000882 to 0.0046, margin 0.0044",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "targeting_necessity/p_ey0__targeted",
        "AI.points.cross_fit",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias -0.0032 to -0.000466, margin 0.0033",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "targeting_necessity/p_ey1__targeted",
        "AI.points.cross_fit",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias -0.000589 to 0.0024, margin 0.0035",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "targeting_necessity/pd_d__targeted",
        "AI.points_regime",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias -0.000687 to 0.0037, margin 0.0043",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "targeting_necessity/pd_s__targeted",
        "AI.points_regime",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias -0.000669 to 0.0029, margin 0.0034",
        "",
    ),
    (
        "adaptive-iptw",
        "property",
        "targeting_necessity/rm_d__targeted",
        "AI.points_regime_missing",
        ("point",),
        LIMIT,
        "adaptive-iptw-finite-sample: bias -0.0025 to 0.0010, margin 0.0048",
        "",
    ),
)

#: The committed decisions, one per positive red row.
DISPOSITIONS: Mapping[RedKey, Disposition] = {
    RedKey(slug, kind, key): Disposition(row, kinds, disposition, limit, flag)
    for slug, kind, key, row, kinds, disposition, limit, flag in (
        *_TABLE,
        *_X40_PRICING,
        *_X40_BOUNDARIES,
        *_X26,
        *_ADAPTIVE_IPTW,
    )
}

#: The owner's answer for each flagged key (RM42b-progress.md, "Eric's answers", 2026-10-15).
DECISIONS: Mapping[RedKey, str] = {
    key: (
        "Q1: keep the withheld status; the row prints its reason and a point estimate only"
        if entry.flag == "Q1"
        else (
            "X40: experimental, with the measured number in the reason (the owner's recommended "
            "default, pending his confirmation)"
            if entry.flag == "X40"
            else "Q6: experimental, with the measured number in the reason"
            + (" (F27: supported inside C3, refused outside)" if "learned-rule" in key.slug else "")
        )
    )
    for key, entry in DISPOSITIONS.items()
    if entry.flag
}

#: The experimental lines whose licensing row has no ``shortfalls`` sentence, each with the
#: reason. A shortfall labels every fit of its row. These cells lie where the key cannot see
#: them, or outside the configuration of the row, so a row-wide label would be wrong.
NO_SHORTFALL: Mapping[RedKey, str] = {
    **{
        key: "the cell fixes one inconsistent nuisance, which the key of the fit cannot read"
        for key, entry in DISPOSITIONS.items()
        if entry.disposition == EXPERIMENTAL and entry.row.startswith("D0")
    },
    **{
        key: "the exceptional and weak-blip laws are laws of the data, which the key cannot read"
        for key, entry in DISPOSITIONS.items()
        if entry.row == "T24"
    },
    **{
        key: "the cell is a cross-fitted or clustered bootstrap, outside the configuration of the row"
        for key, entry in DISPOSITIONS.items()
        if entry.row == "T01.binary" and entry.disposition == EXPERIMENTAL
    },
    **{
        key: "the cell is the two-branch plan p4 under SDR with heavy cumulative ratios, which "
        "the key of the fit cannot read; the SDR caveat of longitudinal-tmle.md states the measured reading"
        for key, entry in DISPOSITIONS.items()
        if key.slug == "mtp-boundaries" and key.kind == "property"
    },
    **{
        key: "the cell fits a gradient-boosting mechanism, outside the configuration of the row: "
        "the key reads the mechanism learner family, and a boosted fit matches no row"
        for key, entry in DISPOSITIONS.items()
        if key.slug == "adaptive-iptw" and entry.disposition == EXPERIMENTAL
    },
    **{
        key: "the cell measures a hypothesis test, and the label has no test kind; the row's "
        "coverage cells carry its interval limit"
        for key, entry in DISPOSITIONS.items()
        if entry.disposition == EXPERIMENTAL and key.key.startswith(("type_i_error/", "power/"))
    },
}

#: The study that names the row where the red row's own study is exempt from the support rows.
ROW_STUDY: Mapping[str, str] = {"learned-rule-cvtmle-boundary": "learned-rule-cvtmle"}


def kinds_of(positive: PositiveRow) -> tuple[str, ...]:
    """The kinds of inference one positive red row measures (plan section 6.2).

    Parameters
    ----------
    positive : PositiveRow
        The red row.

    Returns
    -------
    tuple of str
        Kinds in the order point, wald, wald_t, band, bootstrap. A property family with
        no rule gives an empty tuple, so the draft shows the gap.
    """
    key = positive.red.key
    if key.kind == "paired":
        return ("wald",)
    if key.kind == "truth":
        legs = {_THEORY_LEGS[leg] for leg in positive.failed_legs if leg in _THEORY_LEGS}
        return tuple(kind for kind in ("point", "wald") if kind in legs) or ("point", "wald")
    family, _, cell = key.key.partition("/")
    if family == "interval_calibration":
        if cell.startswith("boot_"):
            return ("bootstrap",)
        return ("wald_t",) if key.slug in T_REFERENCE_SLUGS else ("wald",)
    if family in ("few_cluster_reference", "clustered_inference"):
        return ("wald_t", "wald")
    if family == "simultaneous_coverage":
        return ("band",)
    if family == "root_n_and_efficiency":
        return ("point", "wald")
    if family in _POINT_FAMILIES:
        return ("point",)
    if family in ("type_i_error", "power"):
        return ("wald",)
    return ()


def _default(positive: PositiveRow) -> tuple[str, str, str]:
    """The default disposition, limit and flag of one positive red row."""
    owner, key = positive.owner, positive.red.key
    if owner in ("F18", "F19"):
        return WITHHELD, "", "Q1"
    if owner in LIMIT_OWNERS:
        return SUPPORTED_WITH_LIMIT, f"{owner}: {positive.red.measured}", ""
    if owner == "X20-bootstrap":
        if "clustered" in key.key:
            return EXPERIMENTAL, "", "X20 clustered cell: coverage 0.912 to 0.934"
        return EXPERIMENTAL, "", ""
    if owner in FLAGGED_EXPERIMENTAL:
        return EXPERIMENTAL, "", f"{owner}: {FLAGGED_EXPERIMENTAL[owner]}"
    return EXPERIMENTAL, "", f"no default for owner {owner or 'none'}"


def draft_entries(
    positives: Iterable[PositiveRow], rows: Iterable[tuple[str, frozenset[str], Sequence[str]]]
) -> list[DraftEntry]:
    """The default decision of each positive red row.

    Parameters
    ----------
    positives : iterable of PositiveRow
        The positive red rows.
    rows : iterable of tuple
        Each support row as ``(id, kinds, evidence)``.

    Returns
    -------
    list of DraftEntry
        One entry per red row, in the order given. The candidates are the rows that name the
        study and list every kind the red row measures. A withheld status applies to any row of
        the study, so its candidates ignore the kinds.
    """
    table = list(rows)
    entries = []
    for positive in positives:
        kinds = kinds_of(positive)
        disposition, limit, flag = _default(positive)
        if not kinds:
            flag = (flag + "; " if flag else "") + "no kind rule for this property family"
        slug = positive.red.key.slug
        candidates = tuple(
            row_id
            for row_id, row_kinds, evidence in table
            if slug in evidence and (disposition == WITHHELD or set(kinds) <= row_kinds)
        )
        if not candidates:
            flag = (flag + "; " if flag else "") + "no row of the study lists these kinds"
        entries.append(
            DraftEntry(
                positive.red.key, kinds, disposition, limit, flag, candidates, positive.owner
            )
        )
    return entries


def positive_red_rows() -> list[PositiveRow]:
    """Every positive red row of the committed verdict tables.

    Returns
    -------
    list of PositiveRow
        Property rows with the role ``positive``, truth rows of an implementation that starts
        with ``cleverly``, and paired rows, across the registered studies.
    """
    from cleverly._support_rows import EXEMPT_STUDIES
    from tests.studies.evidence.red_cells import OWNERS, frames, red_rows
    from tests.studies.evidence.registry import registered

    found: list[PositiveRow] = []
    for record in registered():
        if record.slug in EXEMPT_STUDIES and record.slug not in ROW_STUDY:
            # An exempt study licenses no support row, so its red rows bear on no row's
            # disposition. ``ROW_STUDY`` names the exempt studies whose rows still do.
            continue
        tables = frames(record)
        performance = tables["performance"].set_index(["implementation", "scenario", "estimand"])
        for red in red_rows(record, tables):
            kind = red.key.kind
            positive = (
                (kind == "property" and red.role == "positive")
                or (kind == "truth" and red.role.startswith("cleverly"))
                or kind == "paired"
            )
            if not positive:
                continue
            legs: tuple[str, ...] = ()
            if kind == "truth":
                row = performance.loc[tuple(red.key.key.split("/"))]
                legs = tuple(leg for leg in _THEORY_LEGS if leg in row and not bool(row[leg]))
            found.append(PositiveRow(red, OWNERS.get(red.key, ""), legs))
    return found


def render_draft(entries: Sequence[DraftEntry]) -> str:
    """The draft as Markdown, with the flagged lines marked.

    Parameters
    ----------
    entries : sequence of DraftEntry
        The draft entries.

    Returns
    -------
    str
        A counts block and one table row per entry. A flagged line starts with ``FLAG``.
    """
    by_disposition = Counter(entry.disposition for entry in entries)
    flagged = [entry for entry in entries if entry.flag]
    lines = [
        "# B4 draft: dispositions of the positive red rows",
        "",
        f"Entries: {len(entries)}. Flagged: {len(flagged)}. "
        f"With no candidate row: {sum(1 for entry in entries if not entry.candidates)}.",
        "",
        *(f"- {name}: {count}" for name, count in sorted(by_disposition.items())),
        "",
        "| flag | study | row key | kinds | owner | disposition | candidate rows | limit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        cells = (
            f"FLAG: {entry.flag}" if entry.flag else "",
            entry.key.slug,
            f"{entry.key.kind}: {entry.key.key}",
            ", ".join(entry.kinds) or "none",
            entry.owner,
            entry.disposition,
            ", ".join(entry.candidates) or "none",
            entry.limit,
        )
        lines.append("| " + " | ".join(re.sub(r"[|\n]", "/", cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser of ``python -m tests.studies.evidence.support_dispositions``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tests.studies.evidence.support_dispositions",
        description=(
            "Print the draft disposition of each positive red row of the validation ledger. "
            "The script reads the committed verdict tables and the support rows. It runs no "
            "study."
        ),
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        required=True,
        help="print the draft entries, with the flagged lines marked",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the draft to this file (default: standard out)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the draft.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The arguments. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The exit status, ``0``.
    """
    args = build_parser().parse_args(argv)
    from cleverly._support import registered_rows

    rows = [(row.id, row.kinds, row.evidence) for row in registered_rows()]
    text = render_draft(draft_entries(positive_red_rows(), rows))
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
