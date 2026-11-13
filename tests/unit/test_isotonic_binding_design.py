"""The design of ``isotonic-binding`` (Fix-8 plan sections 5.0 and 5.8), recomputed.

Every declared number of :mod:`tests.studies.canonical_isotonic_binding` and
:mod:`tests.studies.isotonic_binding_properties` is checked here: the quadrature truths, the
efficiency bound that sizes the noise arm, the support row a study fit lands on, the replication
count and the witnesses' pass probabilities, the pooled pre-read
(``tests/studies/isotonic_binding_pilot.csv``), the tolerance rule on the committed refits
(``tests/studies/isotonic_binding_tolerance.csv``), and the declared red-cell reading.
"""

from __future__ import annotations

import functools
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binom

from tests.studies import _isotonic_binding_law as law
from tests.studies import canonical_isotonic_binding as study
from tests.studies import isotonic_binding_pilot as pilot
from tests.studies import isotonic_binding_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.registry import ROOT, Margins, registered

MARGINS = Margins()
#: Plan 5.0: the single-cell product floor, at which a perfectly calibrated cell passes with
#: probability 0.977.
PRODUCT_FLOOR = 2_400
#: Plan 5.0 (rev 2.1, 2.2.1): a calibration cell whose pre-read coverage is below this, or whose
#: predicted pass probability at the declared R is below :data:`PREDICTED_PASS`, is diagnosed
#: before declaration.
PRE_READ_COVERAGE = 0.937
PREDICTED_PASS = 0.90
#: The witnesses' pass probability at their pre-read shares, and at the plan's lower shares.
WITNESS_PASS = 0.9999
#: Plan 5.8: the tolerance rule.
MOVE_AGREEMENT = 1e-4
DEAD_BAND = (1e-9, 1e-5)
TOLERANCE_DRAWS = 300
#: The tolerance-table draws refitted in the fast tier (a deviation from plan 5.8's 300 refits,
#: recorded in the study README): a never level that does not move, one that moves just under
#: the material threshold, and one that moves just over it.
REFIT_DRAWS = {1: (False, False), 0: (True, False), 9: (True, True)}
#: The plan's quoted truths (5.8), to five places.
PLAN_TRUTHS = {
    ("never", 1): 0.26074,
    ("never", 2): 0.26493,
    ("always", 1): 0.19717,
    ("always", 2): 0.20106,
}


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(pilot.SUMMARY)


@pytest.fixture(scope="module")
def tolerance() -> pd.DataFrame:
    return pd.read_csv(pilot.TOLERANCE)


def reading(summary: pd.DataFrame, law_name: str, regimen: str, horizon: int) -> pd.Series:
    rows = summary.loc[
        (summary["law"] == law_name)
        & (summary["regimen"] == regimen)
        & (summary["horizon"] == horizon)
    ]
    assert len(rows) == 1, (law_name, regimen, horizon)
    return rows.iloc[0]


def calibration(summary: pd.DataFrame) -> pd.Series:
    return reading(summary, law.LAW, "never", 1)


def share_pass(share: float, floor: float, replicates: int) -> float:
    """The exact probability that the 99% Clopper-Pearson lower limit reaches ``floor``."""
    tail = (1.0 - MARGINS.confidence_level) / 2.0
    counts = np.arange(1, replicates + 1)
    lower = beta.ppf(tail, counts, replicates - counts + 1)
    accepted = counts[lower >= floor]
    return float(binom.sf(int(accepted[0]) - 1, replicates, share))


@functools.cache
def fitted(seed: int, tol: float) -> dict[tuple[str, int], law.Level]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return law.levels(law.fit(law.draw(pilot.N, seed), tol))


class TestTheDeclaration:
    def test_the_constants(self) -> None:
        assert study.SLUG == "isotonic-binding"
        assert (study.SEED, study.RESAMPLING_SEED) == (20267741, 20267742)
        assert study.STUDY.publication_policy == "reporting"
        assert study.STUDY.reference is None
        assert (study.PRIMARY_N, study.PRIMARY_REPLICATES) == (2_000, 1_600)
        assert study.PRIMARY_ESTIMANDS == (law.ESTIMAND,) == ("risk_regimen[never @ t=1]",)
        assert (law.G_BOUNDS, law.MAX_ITER, law.TOL) == ((1e-8, 1.0), 100, 1e-10)
        assert (law.LOGISTIC_TOL, law.REFIT_TOL) == (1e-13, 1e-10)
        assert properties.N == pilot.N == 2_000
        assert (properties.BINDING_SE, properties.MATERIAL_SE) == (1e-6, 0.1)
        assert (pilot.BINDING_SE, pilot.MATERIAL_SE) == (1e-6, 0.1)
        assert properties.SHARE_FLOORS == {"binding_witness": 0.25, "material_move": 0.10}

    def test_the_learner_is_the_pinned_class(self) -> None:
        learner = law.logistic()
        assert learner.get_params()["C"] == np.inf
        assert learner.get_params()["tol"] == 1e-13
        assert learner.get_params()["max_iter"] == 100_000

    def test_the_seeds_are_the_study_s_own(self) -> None:
        seeds = {s for record in registered() for s in (record.seed, record.resampling_seed)}
        assert not {study.SEED, study.RESAMPLING_SEED, pilot.PILOT_SEED} & seeds

    def test_declared_and_not_registered(self) -> None:
        assert study.SLUG not in {record.slug for record in registered()}
        from tests.unit.test_method_evidence import UNCOMMITTED_MODULES

        assert properties.__name__ in UNCOMMITTED_MODULES

    def test_the_record_declares_the_cells_the_module_builds(self) -> None:
        built: dict[str, set[str]] = {}
        for cell in properties.declared_cells():
            built.setdefault(cell.property, set()).add(cell.cell)
        assert built == {family: set(cells) for family, cells in study.PROPERTY_CELLS.items()}

    def test_the_roles_and_streams(self) -> None:
        cells = {(c.property, c.cell): c for c in properties.declared_cells()}
        for key, cell in cells.items():
            assert cell.replicates == properties.REPLICATES
            assert cell.n == properties.N
            if key[0] in properties.REPORTED_FAMILIES:
                assert cell.role == property_verdicts.DIAGNOSTIC_ROLE
            elif key[1].endswith("_control"):
                assert cell.role == "control"
            else:
                assert cell.role == "positive"
            stream = cell.fit_kwargs["stream"]
            assert cell.dgp == properties.STREAMS[stream][0]
            assert (stream == "lg2_gs2") == (key[0] == "reported_large_move")

    def test_every_cell_of_a_stream_reads_the_same_draw(self) -> None:
        seeds: dict[str, list[int]] = {}
        for ((_, stream, _, _, _, seed, _),) in properties.payloads(budget=3):
            seeds.setdefault(stream, []).append(seed)
        assert set(seeds) == set(properties.STREAMS)
        assert seeds["lg2"] != seeds["lg2_gs2"]


class TestTheLaw:
    def test_the_truths_are_the_plan_s(self) -> None:
        for (regimen, horizon), value in PLAN_TRUTHS.items():
            assert law.TRUTH[law.name(regimen, horizon)] == pytest.approx(value, abs=5e-6)

    def test_the_quadrature_has_converged(self) -> None:
        finer = law.truths(200)
        for key, value in law.TRUTH.items():
            assert finer[key] == pytest.approx(value, abs=1e-10)

    def test_the_risk_is_strictly_increasing(self) -> None:
        for regimen in law.REGIMENS:
            first, second = (law.TRUTH[law.name(regimen, h)] for h in law.HORIZONS)
            assert second - first > 0.003

    def test_the_draws_have_the_law_s_moments(self) -> None:
        frame = law.draw(400_000, 5)
        w = frame["W"]
        assert (frame["A1"] - law.p_treatment_1(w)).mean() == pytest.approx(0.0, abs=3e-3)
        assert (frame["C1"] - law.p_retention_1(w)).mean() == pytest.approx(0.0, abs=3e-3)
        seen = frame["Y1"].notna()
        residual = frame.loc[seen, "Y1"] - law.p_event_1(w[seen], frame.loc[seen, "A1"])
        assert residual.mean() == pytest.approx(0.0, abs=3e-3)
        assert frame.loc[frame["Y1"] == 1, "L2"].isna().all()

    def test_the_efficiency_bound(self) -> None:
        assert law.efficiency_sd() == pytest.approx(
            properties.EFFICIENCY_SD["projected_t1"], abs=1e-6
        )
        frame = law.draw(1_000_000, 9)
        w = frame["W"].to_numpy()
        follow = ((frame["A1"] == 0) & (frame["C1"] == 1)).to_numpy()
        p = law.p_event_1(w, 0.0)
        weight = 1.0 / ((1.0 - law.p_treatment_1(w)) * law.p_retention_1(w))
        y = np.nan_to_num(frame["Y1"].to_numpy())
        curve = np.where(follow, weight * (y - p), 0.0) + p - law.TRUTH[law.ESTIMAND]
        assert float(np.std(curve)) == pytest.approx(law.efficiency_sd(), rel=0.02)

    def test_a_study_fit_lands_on_the_isotonic_row(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = law.fit(law.draw(pilot.N, 3))
        support = result.estimates[law.ESTIMAND].support
        assert support.row == "G31.isotonic"
        assert support.label == "supported"


class TestTheSizes:
    def test_the_pooled_predictive_rule(self, summary: pd.DataFrame) -> None:
        row = calibration(summary)
        pooled = sizing.calibration_replicates(
            MARGINS,
            {"projected_t1__correctly_specified": float(row["se_ratio"])},
            int(row["draws"]),
            lambda labels: np.ones((len(labels), len(labels))),
        )
        assert max(PRODUCT_FLOOR, pooled) == properties.REPLICATES

    def test_the_product_floor(self) -> None:
        assert sizing.calibration_pass(MARGINS, PRODUCT_FLOOR, np.array([1.0]))[0] == (
            pytest.approx(0.977, abs=2e-3)
        )

    def test_the_derived_controls_fail_with_certainty(self) -> None:
        r = properties.REPLICATES
        for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / np.sqrt(2.0)):
            assert sizing.calibration_control_probability(MARGINS, r, ratio) > 0.9999

    def test_the_witnesses_pass_at_the_pre_read(self, summary: pd.DataFrame) -> None:
        row = calibration(summary)
        for family, column in (
            ("binding_witness", "binding_share"),
            ("material_move", "material_share"),
        ):
            floor = properties.SHARE_FLOORS[family]
            share = float(row[column])
            count = round(share * int(row["draws"]))
            lower = float(beta.ppf(0.005, count, int(row["draws"]) - count + 1))
            assert share_pass(lower, floor, properties.REPLICATES) > WITNESS_PASS

    def test_the_witnesses_pass_at_the_plan_s_lower_shares(self) -> None:
        assert share_pass(0.30, 0.25, properties.REPLICATES) > 0.998
        assert share_pass(0.14, 0.10, properties.REPLICATES) > 0.9997


class TestThePreRead:
    def test_the_pre_read_pools_the_plan_s_draws_with_fresh_ones(
        self, summary: pd.DataFrame
    ) -> None:
        row = calibration(summary)
        assert (int(row["plan_draws"]), int(row["pilot_draws"])) == (
            len(pilot.PLAN_SEEDS),
            pilot.DRAWS,
        )
        assert int(row["draws"]) >= 2_000
        large = reading(summary, law.LARGE_MOVE_LAW, "never", 1)
        assert int(large["pilot_draws"]) == pilot.LARGE_MOVE_DRAWS

    def test_the_calibration_cell_reads_above_the_pre_read_line(
        self, summary: pd.DataFrame
    ) -> None:
        row = calibration(summary)
        assert float(row["coverage"]) >= PRE_READ_COVERAGE
        predicted = sizing.calibration_pass(
            MARGINS, properties.REPLICATES, np.array([float(row["se_ratio"])])
        )[0]
        assert predicted >= PREDICTED_PASS

    def test_the_projection_binds_materially(self, summary: pd.DataFrame) -> None:
        row = calibration(summary)
        assert float(row["binding_share"]) > 0.3
        assert float(row["material_share"]) > 0.15
        assert 0.3 < float(row["max_move"]) < 1.0

    def test_the_two_intervals_rarely_disagree(self, summary: pd.DataFrame) -> None:
        assert float(calibration(summary)["disagree_share"]) < 0.01


class TestTheTolerance:
    def test_the_table_holds_every_level_of_the_first_draws(self, tolerance: pd.DataFrame) -> None:
        assert sorted(tolerance["draw"].unique()) == list(range(TOLERANCE_DRAWS))
        assert len(tolerance) == 4 * TOLERANCE_DRAWS

    def test_the_indicators_agree(self, tolerance: pd.DataFrame) -> None:
        pinned, refit = tolerance["move_pinned"], tolerance["move_refit"]
        assert ((pinned > properties.BINDING_SE) == (refit > properties.BINDING_SE)).all()
        assert ((pinned >= properties.MATERIAL_SE) == (refit >= properties.MATERIAL_SE)).all()

    def test_the_moves_agree(self, tolerance: pd.DataFrame) -> None:
        moved = tolerance["move_pinned"] > 0.0
        pinned = tolerance.loc[moved, "move_pinned"]
        refit = tolerance.loc[moved, "move_refit"]
        assert ((refit - pinned).abs() / pinned).max() <= MOVE_AGREEMENT
        assert (tolerance.loc[~moved, "move_refit"] == 0.0).all()

    def test_no_move_falls_in_the_dead_band(self, tolerance: pd.DataFrame) -> None:
        for column in ("move_pinned", "move_refit"):
            values = tolerance[column]
            assert not values.between(*DEAD_BAND, inclusive="neither").any()

    def test_every_logistic_fit_converged(self, tolerance: pd.DataFrame) -> None:
        assert (tolerance["stalled_pinned"] == 0).all()
        assert (tolerance["stalled_refit"] == 0).all()

    @pytest.mark.parametrize("draw", sorted(REFIT_DRAWS))
    def test_the_table_reproduces(self, tolerance: pd.DataFrame, draw: int) -> None:
        rows = tolerance.loc[tolerance["draw"] == draw]
        for tol, column in ((law.LOGISTIC_TOL, "move_pinned"), (law.REFIT_TOL, "move_refit")):
            levels = fitted(pilot.pilot_seed(draw), tol)
            for _, row in rows.iterrows():
                move = levels[(row["regimen"], int(row["horizon"]))].move
                assert move == pytest.approx(float(row[column]), rel=1e-6, abs=1e-12)
                assert properties.binds(move) == properties.binds(float(row[column]))
                assert properties.moves_materially(move) == properties.moves_materially(
                    float(row[column])
                )
            never = levels[("never", 1)].move
            assert (properties.binds(never), properties.moves_materially(never)) == REFIT_DRAWS[
                draw
            ]


#: The calibration cells, whose red verdict follows the shared reading: a red control is a
#: defect suspect, and a red positive is admissible only as an owned finite-sample red.
CALIBRATION_GATED = (
    "interval_calibration/projected_t1__correctly_specified",
    "interval_calibration/projected_t1__shrunken_se_control",
    "interval_calibration/projected_t1__noise_control",
)
#: The witnesses, whose red verdict is a shortfall of ``G31.isotonic`` (plan 5.8): with the
#: law, learners and tolerance fixed, a red share is a measured fact the row states.
WITNESSES = ("binding_witness/never_t1", "material_move/never_t1")
GATED = (*CALIBRATION_GATED, *WITNESSES)
#: The support row the study backs.
ROW = "G31.isotonic"


def red_witnesses(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Each red witness of committed tables, with its measured share."""
    out = {}
    for row in tables["properties"].itertuples():
        name = f"{row.property}/{row.cell}"
        if name in WITNESSES and not (bool(row.passed) and bool(row.property_passed)):
            out[name] = float(row.coverage)
    return out


def unstated_witnesses(red: dict[str, float], shortfalls: tuple[tuple[str, str], ...]) -> list[str]:
    """The red witnesses whose share no shortfall sentence of the row states (to 3 places)."""
    texts = [text for _, text in shortfalls]
    return [
        name
        for name, share in red.items()
        if not any(name.split("/")[0] in text and f"{share:.3f}" in text for text in texts)
    ]


class TestTheRedCellReading:
    def test_the_gated_cells_are_every_cell_that_is_not_reported(self) -> None:
        assert set(GATED) == set(properties.gated_cells())

    def test_no_calibration_cell_reads_an_inadmissible_red(self) -> None:
        """The shared red-cell reading, applied mechanically to the calibration cells.

        Empty until the run commits its tables; at registration a red control, or a red
        positive without its finite-sample owner, fails here.  The primary's truth rows gate
        here too.
        """
        from tests.studies.mtp_study_tools import gating_findings

        assert (
            gating_findings(study.STUDY, CALIBRATION_GATED, truth_implementations=("cleverly",))
            == []
        )

    def test_every_red_witness_is_a_shortfall_of_the_row(self) -> None:
        """A red witness labels ``G31.isotonic`` experimental with its measured share.

        Empty until the run commits its tables.
        """
        from cleverly._support import registered_rows
        from tests.studies.evidence import red_cells

        if not study.STUDY.artifact(red_cells.FRAMES["properties"]).exists():
            return
        tables = red_cells.frames(study.STUDY)
        assert set(WITNESSES) <= {
            f"{row.property}/{row.cell}" for row in tables["properties"].itertuples()
        }
        (row,) = (row for row in registered_rows() if row.id == ROW)
        assert unstated_witnesses(red_witnesses(tables), row.shortfalls) == []

    def test_a_red_witness_without_its_shortfall_fails_the_check(self) -> None:
        """The control of the check above, on a synthetic red witness."""
        table = pd.DataFrame(
            {
                "property": ["binding_witness", "material_move"],
                "cell": ["never_t1", "never_t1"],
                "passed": [False, True],
                "property_passed": [False, True],
                "coverage": [0.2412, 0.19],
            }
        )
        red = red_witnesses({"properties": table})
        assert red == {"binding_witness/never_t1": 0.2412}
        assert unstated_witnesses(red, ()) == ["binding_witness/never_t1"]
        stated = (("wald", "The binding_witness share read 0.241, below its floor 0.25."),)
        assert unstated_witnesses(red, stated) == []


#: A cited summary value on a method-evidence page.
CITED = re.compile(r"properties\[(?P<family>\w+)/(?P<cell>\w+)\]:(?P<column>\w+)")


def placeholder_citations(text: str) -> list[str]:
    """The citations of a share cell's placeholder-derived columns in ``text``."""
    shares = set(properties.SHARE_CELLS)
    return [
        match.group(0)
        for match in CITED.finditer(text)
        if (match.group("family"), match.group("cell")) in shares
        and match.group("column") not in properties.SHARE_COLUMNS
    ]


class TestThePlaceholderColumns:
    def test_every_share_cell_is_declared(self) -> None:
        declared = {
            (family, cell) for family, cells in study.PROPERTY_CELLS.items() for cell in cells
        }
        assert set(properties.SHARE_CELLS) <= declared

    def test_the_evidence_page_cites_no_placeholder_column(self) -> None:
        page = ROOT / study.STUDY.document
        if page.exists():
            assert placeholder_citations(page.read_text(encoding="utf-8")) == []

    def test_a_placeholder_citation_is_found(self) -> None:
        text = (
            "`properties[binding_witness/never_t1]:coverage_ci_lower` and "
            "`properties[binding_witness/never_t1]:se_ratio` and "
            "`properties[interval_calibration/projected_t1__correctly_specified]:se_ratio`"
        )
        assert placeholder_citations(text) == ["properties[binding_witness/never_t1]:se_ratio"]
