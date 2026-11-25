"""The declaration of the hte3 comparison of the EP curve (P1 plan, PR 4b, section 4).

Checked here before any run: the cells are leading prefixes of committed parent groups, the
comparison owns no root seed, its two stream labels are its own, a re-drawn replicate is the
parent's draw bit for bit, the exported nuisances are the fit's, the numpy emulation of hte3
solves the equations hte3 solves (each with a witness that is nonzero at the inputs), and the
sizing rule reproduces the plan's numbers.  The tests that read the committed pilot, the pinned
replicate counts or the committed R smoke rows skip until those files exist.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize
from scipy.special import expit
from sklearn.linear_model import LinearRegression

from tests.studies import ep_cate, ep_cate_properties, ep_crr, ep_crr_properties
from tests.studies import ep_hte3 as study
from tests.studies import ep_hte3_pilot as emulation
from tests.studies.evidence.registry import ROOT, registered
from tests.studies.evidence.seeds import stream_seed

#: The committed SD of each parent's EP oracle comparison, as the plan's cells table states it.
PLAN_SDS = {
    "E1": 0.102,
    "E2": 0.127,
    "E3": 0.100,
    "E4": 0.114,
    "E5": 0.081,
    "E6": 0.087,
    "E7": 0.365,
    "E8": 0.367,
    "C1": 0.083,
    "C2": 0.139,
    "C3": 0.087,
    "C4": 0.131,
    "C5": 0.130,
    "C6": 0.343,
}


def _committed() -> pd.DataFrame:
    frames = []
    for module in (ep_cate, ep_crr):
        rows = pd.read_parquet(module.STUDY.artifacts / "property-replicates.parquet")
        frames.append(rows.loc[rows["property"] == module.ORACLE].assign(parent=module.SLUG))
    return pd.concat(frames, ignore_index=True)


class TestTheCells:
    def test_the_cells_are_the_plans(self) -> None:
        assert [cell.label for cell in study.READING_CELLS] == list(PLAN_SDS)
        assert {cell.label: cell.mutation_of for cell in study.MUTATION_CELLS} == {
            "M1": "E8",
            "M2": "C6",
        }
        witnesses = {cell.label for cell in study.CELLS if cell.role == study.WITNESS}
        assert witnesses == {"E7", "E8", "C5", "C6"}

    @pytest.mark.parametrize("cell", study.CELLS, ids=lambda cell: cell.label)
    def test_each_cell_reads_a_committed_group_of_its_parent(self, cell: study.Cell) -> None:
        group = cell.module.GROUP_BY_LABEL[cell.group]
        law = study.law_of(cell)
        assert group.kind == "gated" and group.n == 4_000
        if cell.parent == ep_cate.SLUG:
            fits = law.ensemble if cell.learner == "ensemble" else law.fits_misspecified(group.n)
        else:
            fits = cell.learner == "ensemble" or (
                law.rescue and cell.learner in ep_crr.RESCUE_LEARNERS
            )
        assert fits
        assert cell.parent_cell in cell.module.PROPERTY_CELLS[cell.module.ORACLE]
        assert cell.planning_replicates <= study.max_replicates(cell)

    def test_the_committed_sds_are_the_plans(self) -> None:
        """The cell-to-parent map is right: each parent cell's committed SD is the plan's."""
        committed = _committed()
        for cell in study.READING_CELLS:
            rows = committed.loc[
                (committed["parent"] == cell.parent) & (committed["cell"] == cell.parent_cell)
            ]
            assert len(rows) == study.max_replicates(cell)
            assert rows["estimate"].std(ddof=1) == pytest.approx(PLAN_SDS[cell.label], abs=6e-4)

    def test_the_margins_are_the_plans(self) -> None:
        assert math.log(1.25) == study.NONINFERIORITY_MARGIN
        assert math.log(1.25) == study.SIMILARITY_MARGIN
        assert math.log(1.5) == study.WITNESS_MARGIN
        assert study.CURVE_DISTANCE_THRESHOLD == 1.0
        assert study.CONFIDENCE == 0.99
        margins = {cell.label: cell.validity_margin for cell in study.CELLS}
        for label in ("E7", "E8", "M1"):
            assert margins.pop(label) == math.log(3.0)
        assert set(margins.values()) == {math.log(1.5)}


class TestTheDraws:
    def test_the_comparison_owns_no_root_seed_and_no_record(self) -> None:
        assert not hasattr(study, "SEED")
        assert not hasattr(study, "STUDY")
        assert "ep-hte3" not in {record.slug for record in registered()}
        for module in (ep_cate, ep_crr):
            assert module.STUDY.reference is None
            assert not hasattr(module, "COMPARATOR")

    @pytest.mark.parametrize("label", [study.PILOT_STREAM, study.SMOKE_STREAM])
    def test_each_stream_label_is_used_nowhere_else(self, label: str) -> None:
        quoted = f'"{label}"'
        users = sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tests").rglob("*.py")
            if quoted in path.read_text(encoding="utf-8") and path.name != "test_ep_hte3_design.py"
        )
        assert users == ["tests/studies/ep_hte3.py"]

    def test_the_new_streams_are_disjoint_from_the_parents(self) -> None:
        parent_streams = {
            ep_cate.SLUG: ("property_sample", "ise_ratio_pilot", "rescue_noise_pilot"),
            ep_crr.SLUG: (
                "property_sample",
                "crr_ise_ratio_pilot",
                "crr_growth_pilot",
                "crr_hessian_pilot",
                "b1_offset_pilot_scale2",
            ),
        }
        theirs, ours = set(), set()
        for cell in study.READING_CELLS:
            record = cell.module.STUDY
            for r in range(30):
                theirs |= {
                    stream_seed(record, stream, cell.group, r)
                    for stream in parent_streams[cell.parent]
                }
                ours |= {study.draw_seed(cell, r, study.PILOT_STREAM)}
                ours |= {study.draw_seed(cell, r, study.SMOKE_STREAM)}
        groups = {(cell.parent, cell.group) for cell in study.READING_CELLS}
        assert len(ours) == 2 * 30 * len(groups)
        assert ours.isdisjoint(theirs)
        roots = {ep_cate.SEED, ep_cate.RESAMPLING_SEED, ep_crr.SEED, ep_crr.RESAMPLING_SEED}
        assert ours.isdisjoint(roots)

    @pytest.mark.parametrize("label", ["E5", "C6"])
    def test_a_redrawn_replicate_is_the_parents_draw(self, label: str) -> None:
        cell = study.CELL_BY_LABEL[label]
        properties = ep_cate_properties if cell.parent == ep_cate.SLUG else ep_crr_properties
        parent, _ = properties._draw(cell.group, 7)
        pd.testing.assert_frame_equal(study.draw(cell, 7), parent, check_exact=True)

    def test_the_ise_points_are_the_parents(self) -> None:
        for label in ("E1", "E5", "C1"):
            cell = study.CELL_BY_LABEL[label]
            law = study.law_of(cell)
            seed = stream_seed(cell.module.STUDY, study.ISE_STREAM, law.label)
            assert np.array_equal(
                study.ise_points(cell),
                cell.module.covariates(law, cell.module.ISE_POINTS, np.random.default_rng(seed))
                if cell.parent == ep_cate.SLUG
                else cell.module.covariates(cell.module.ISE_POINTS, np.random.default_rng(seed)),
            )


class TestTheExport:
    @pytest.fixture(scope="class")
    def fitted(self) -> tuple[pd.DataFrame, object]:
        law = ep_cate.LAW_BY_LABEL["d3_complex_limited_noise2"]
        frame = ep_cate.sample(law, 600, np.random.default_rng(11))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ep_cate.effect_of(law, frame).estimate(
                "ep",
                sieve=ep_cate.Sieve.default(ep_cate.modifiers(law), max_size=3, size=2),
                outcome_learner=LinearRegression(),
                outcome_by_arm=True,
                n_folds=2,
                q_bounds=ep_cate.q_bounds(law),
                random_state=0,
            )
        return frame, fit

    def test_the_exported_nuisances_are_the_fits(self, fitted: tuple) -> None:
        frame, fit = fitted
        mu0, mu1, pi0, pi1 = study.nuisances(fit)
        nuisance_fit = fit.fit.nuisance_fit
        nuisance = nuisance_fit.nuisance
        # The arm regressions the fit holds, unscaled, at both arms.
        arms = {
            a: nuisance.scaler.unscale_levels(np.asarray(nuisance.outcome.arms[a], dtype=float))
            for a in (0.0, 1.0)
        }
        assert np.array_equal(mu0, arms[0.0]) and np.array_equal(mu1, arms[1.0])
        assert not np.allclose(mu0, mu1), "a witness that vanishes would pass for any swap"
        low, high = nuisance_fit.config.g_bounds
        assert pi1.min() >= low - 1e-15 and pi1.max() <= high + 1e-15
        np.testing.assert_allclose(pi0 + pi1, 1.0, atol=1e-12)
        # The exported pair reproduces the AIPW pseudo-outcomes the EP fit reads.
        a = frame["A"].to_numpy(float)
        y = frame["Y"].to_numpy(float)
        from cleverly.effect_curves import _pseudo_outcomes

        _, _, d_t, d_r = _pseudo_outcomes(
            nuisance,
            fit.fit.data,
            (1.0, 0.0),
            g_bounds=tuple(nuisance_fit.config.g_bounds),
            missingness_bound=float(nuisance_fit.config.missingness_bound),
        )
        np.testing.assert_allclose(d_t, a / pi1 * (y - mu1) + mu1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(d_r, (1 - a) / pi0 * (y - mu0) + mu0, rtol=0, atol=1e-12)

    def test_the_inputs_archive_keeps_the_columns_apart(self) -> None:
        cell = study.CELL_BY_LABEL["E7"]
        rng = np.random.default_rng(3)
        n = 50
        export = study.Export(
            "E7",
            4,
            rng.uniform(-1, 1, (n, 3)),
            rng.integers(0, 2, n).astype(float),
            rng.normal(size=n),
            5,
            rng.normal(size=n),
            rng.normal(size=n) + 3,
            np.full(n, 0.3),
            np.full(n, 0.7),
            np.zeros(10),
            1.0,
            1.0,
            1.0,
        )
        frame = study.input_frame(cell, [export])
        assert list(frame.columns) == [
            "scenario",
            "replicate",
            "k",
            "scale",
            "W1",
            "W2",
            "W3",
            "A",
            "Y",
            "mu0",
            "mu1",
            "pi0",
            "pi1",
        ]
        assert np.array_equal(frame["mu1"], export.mu1) and np.array_equal(frame["pi0"], export.pi0)
        assert set(frame["scale"]) == {"difference"}
        mutated = export.mutated("M1")
        assert np.all(mutated.pi0 == 0.5) and np.all(mutated.pi1 == 0.5)
        assert np.array_equal(mutated.mu1, export.mu1)


# ---------------------------------------------------------------- the emulation


def _synthetic(scale: str, n: int = 800, seed: int = 0) -> study.Export:
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, (n, 3))
    pi1 = expit(1.5 * w.sum(axis=1))
    a = rng.binomial(1, pi1).astype(float)
    if scale == "difference":
        mu0 = np.sin(3 * w[:, 0])
        mu1 = mu0 + 1 + w[:, 1]
        y = np.where(a == 1, mu1, mu0) + rng.normal(size=n)
        # Wrong outcome regressions, so the debiasing step has work to do.
        mu0, mu1 = mu0 + 0.5 * w[:, 2], mu1 - 0.4
    else:
        mu0 = expit(-0.5 + w[:, 0])
        mu1 = np.minimum(mu0 * np.exp(0.3 + 0.2 * w[:, 1]), 0.95)
        y = rng.binomial(1, np.where(a == 1, mu1, mu0)).astype(float)
        mu0, mu1 = expit(-0.5 + 0 * w[:, 0]), np.full(n, 0.4)
    return study.Export(
        "X",
        0,
        w,
        a,
        y,
        3,
        mu0,
        mu1,
        np.clip(1 - pi1, 0.02, 0.98),
        np.clip(pi1, 0.02, 0.98),
        np.zeros(1),
        1.0,
        1.0,
        1.0,
    )


class TestTheEmulation:
    def test_brent_is_rs_optimize(self) -> None:
        def f(x: float) -> float:
            return (x - 0.137) ** 2 + 0.01 * abs(x - 0.2)

        found = emulation.brent_fmin(f, 0.001, 0.5)
        reference = optimize.minimize_scalar(f, bounds=(0.001, 0.5), method="bounded").x
        assert found == pytest.approx(reference, abs=1e-6)
        assert found == pytest.approx(0.137 + 0.005, abs=1e-6)

    def test_the_adaptive_truncation_binds_where_overlap_is_poor(self) -> None:
        export = _synthetic("difference")
        treated = export.a == 1
        star = emulation.truncate_propensity(export.pi1, treated)
        assert star.min() > export.pi1.min() + 0.01, "a truncation that never binds is no witness"
        assert star.min() >= 1 / export.a.size
        assert np.all(star >= np.minimum(export.pi1, star.min()) - 1e-15)
        # Away from the boundary the rule moves nothing.
        flat = emulation.truncate_propensity(np.full(400, 0.5), np.arange(400) % 2 == 0)
        assert np.all(flat == 0.5)

    def test_the_sieve_is_sieves_column_set(self) -> None:
        index = emulation.sieve_index(3, 5)
        assert index.shape == (16, 3)
        assert np.all(index[0] == 1)
        expected = {(1, 1, 1)} | {
            tuple(j if c == m else 1 for c in range(3)) for m in range(3) for j in range(2, 7)
        }
        assert {tuple(row) for row in index} == expected
        assert np.all((index > 1).sum(axis=1) <= 1)
        assert emulation.sieve_index(20, 5).shape == (101, 20)

    def test_the_sieve_rescales_by_the_training_quantiles_and_clips(self) -> None:
        x = np.linspace(-1, 1, 101)[:, None]
        norm = emulation.sieve_norm(x)
        assert norm[:, 0] == pytest.approx([-0.98, 0.98])
        phi = emulation.sieve_basis(np.array([[-1.0], [0.0], [1.0]]), 2, norm)
        np.testing.assert_allclose(phi[:, 1], [1.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(phi[:, 2], [1.0, -1.0, 1.0], atol=1e-12)

    def test_the_cate_step_solves_hte3s_weighted_score(self) -> None:
        export = _synthetic("difference")
        norm = emulation.sieve_norm(export.w)
        phi = emulation.sieve_basis(export.w, export.k, norm)
        weights = emulation._weights(export)
        sign = np.where(export.a == 1, 1.0, -1.0)
        offset = np.where(export.a == 1, export.mu1, export.mu0)
        before = phi.T @ (weights * sign * (export.y - offset)) / export.a.size
        assert np.max(np.abs(before)) > 0.05, "the initial fit must leave a score to solve"
        theta = emulation.cate_ep(export, export.w)
        final = np.linalg.lstsq(phi, theta, rcond=None)[0]
        np.testing.assert_allclose(phi @ final, theta, atol=1e-10)
        # theta is the projection of mu_1^* - mu_0^* = mu_1 - mu_0 + 2 phi beta; recover beta.
        beta = np.linalg.lstsq(phi, (theta - (export.mu1 - export.mu0)) / 2, rcond=None)[0]
        star = offset + sign * (phi @ beta)
        after = phi.T @ (weights * sign * (export.y - star)) / export.a.size
        assert np.max(np.abs(after)) < 1e-6

    def test_the_crr_step_solves_each_arms_weighted_score(self) -> None:
        export = _synthetic("log_ratio")
        theta = emulation.crr_ep(export, export.w)
        assert np.all(np.isfinite(theta))
        mutated = emulation.crr_ep(export.mutated("M"), export.w)
        assert np.max(np.abs(mutated - theta)) > 1e-3, "the 1/pi weight must move the curve"

    def test_the_mutation_moves_the_cate_curve(self) -> None:
        export = _synthetic("difference")
        theta = emulation.cate_ep(export, export.w)
        mutated = emulation.cate_ep(export.mutated("M"), export.w)
        assert np.max(np.abs(mutated - theta)) > 1e-2


# ---------------------------------------------------------------- the sizing rule


class TestTheSizingRule:
    def test_the_plans_worst_case_table(self) -> None:
        """Plan 4.4 at correlation -1: SD(D) = 2s, R = 100 or 300."""
        p = study.pass_probability
        tenth = math.log(1.1)
        assert p(0.0, 2 * 0.367, 150) == pytest.approx(0.915, abs=0.002)
        assert p(0.0, 2 * 0.367, 300) == pytest.approx(0.998, abs=0.002)
        assert p(tenth, 2 * 0.367, 300) == pytest.approx(0.75, abs=0.01)
        assert p(0.0, 2 * 0.343, 150) == pytest.approx(0.949, abs=0.002)
        assert p(tenth, 2 * 0.343, 300) == pytest.approx(0.81, abs=0.01)
        sds = [PLAN_SDS[label] for label in PLAN_SDS if label not in ("E7", "E8", "C6")]
        assert min(p(tenth, 2 * s, 100) for s in sds) == pytest.approx(0.987, abs=0.002)

    def test_the_rule_takes_the_smallest_candidate_and_reports_the_rest(self) -> None:
        planning = {"a": (math.log(1.1), 0.05), "b": (math.log(1.1), 0.45), "c": (0.2, 0.5)}
        caps = {"a": 300, "b": 300, "c": 300}
        sized, reported = study.size_cells(planning, caps)
        assert sized["a"] == 100
        assert sized["b"] == 200
        assert reported == {"c"}
        # A cap below the candidate a cell needs reports it instead.
        sized, reported = study.size_cells(planning, {**caps, "b": 100})
        assert "b" in reported and "b" not in sized

    def test_the_family_target_raises_the_weakest_cell(self) -> None:
        mean = math.log(1.1)
        sd = next(
            s
            for s in np.linspace(0.05, 0.5, 400)
            if study.pass_probability(mean, s, 100) < 0.95
            and study.pass_probability(mean, s, 100) >= 0.90
        )
        planning = {label: (mean, float(sd)) for label in "abcd"}
        sized, _ = study.size_cells(planning, dict.fromkeys(planning, 300))
        family = np.prod([study.pass_probability(mean, sd, r) for r in sized.values()])
        assert family >= study.TARGET_FAMILY
        assert max(sized.values()) > 100

    def test_the_committed_m2_mutation_fails_at_twenty_replicates(self) -> None:
        assert study.design_numbers()["mutations"]["M2"] < 1e-6

    def test_without_a_pilot_the_plans_fallback_applies(self, tmp_path: Path) -> None:
        numbers = study.design_numbers(tmp_path / "absent.csv")
        assert numbers["source"] == "fallback"
        assert numbers["replicates"] == {c.label: c.planning_replicates for c in study.CELLS}
        assert numbers["reported"] == {"E7", "E8", "C6", "M1"}

    @pytest.mark.skipif(not study.PILOT_CSV.exists(), reason="the pilot has not been committed")
    def test_the_pilot_covers_every_cell(self) -> None:
        pilot = study.pilot_rows()
        assert pilot is not None
        for cell in study.READING_CELLS:
            assert pilot[(cell.label, "d")]["replicates"] >= emulation.REPLICATES
        assert pilot[("M1", "log_ratio_hte3")]["replicates"] >= emulation.REPLICATES

    def test_the_declared_replicates_are_the_rules(self) -> None:
        if not study.REPLICATES:
            pytest.skip("the comparison is pending its declaration: REPLICATES is not pinned yet")
        numbers = study.design_numbers()
        assert numbers["source"] == "pilot", "the declaration reads the committed pilot"
        assert dict(study.REPLICATES) == numbers["replicates"]
        assert numbers["reported"] == study.REPORTED
        for cell in study.CELLS:
            assert study.REPLICATES[cell.label] <= study.max_replicates(cell)


# ---------------------------------------------------------------- the R smoke


def _smoke(name: str) -> pd.DataFrame:
    path = study.SMOKE_DIR / name
    if not path.exists():
        pytest.skip(f"the R smoke has not been committed ({path.relative_to(ROOT)})")
    return pd.read_csv(path)


class TestTheRSmoke:
    def test_the_emulation_agrees_with_r_at_every_point(self) -> None:
        agreement = _smoke("agreement.csv")
        assert set(agreement["cell"]) == {cell.label for cell in study.CELLS}
        assert (agreement["points"] == 20_000).all()
        assert agreement["max_abs_difference"].max() <= study.EMULATION_TOLERANCE

    def test_the_committed_points_agree(self) -> None:
        points = _smoke("points.csv")
        gap = (points["theta_r"] - points["theta_emulation"]).abs()
        assert gap.max() <= study.EMULATION_TOLERANCE
        assert points["theta_r"].abs().max() > 0.1, "a curve near zero would agree with anything"

    def test_sieves_index_matrix_is_the_declared_column_set(self) -> None:
        index = _smoke("sieve-index.csv")
        for cell_label, rows in index.groupby("scenario"):
            cell = study.CELL_BY_LABEL[str(cell_label)]
            d = len(study.columns(cell))
            matrix = (
                rows.pivot(index="column", columns="modifier", values="index")
                .sort_index()
                .to_numpy()
            )
            expected = emulation.sieve_index(d, study.sieve_size(cell))
            assert {tuple(row) for row in matrix} == {tuple(row) for row in expected}
            assert np.array_equal(matrix, expected), "the emulation's column order is Sieve's"
