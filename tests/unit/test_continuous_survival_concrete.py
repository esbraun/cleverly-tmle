"""The ``concrete`` comparator phase of ``continuous-time-survival`` (roadmap X6 part f).

CI installs no R, so the first group reads the pinned image and the R runner as text and checks
what they transcribe against :mod:`tests.studies.continuous_survival_concrete`, and checks the
Python side's hazard construction and fit settings.  The second group replays a committed run:
it skips until ``run_phase.py --publish`` has written ``manifest.json``.  Then it gates the
image, the runner and the harness against the bytes that ran, recomputes every table from the
committed rows, and refits the first draw of each cell.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from cleverly.survival import CoxHazard
from tests.documents import pipe_table
from tests.studies import continuous_survival_concrete as phase
from tests.studies import continuous_survival_law as law
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import ROOT

DOCKERFILE = ROOT / phase.REFERENCE_SOURCES[0]
RUNNER = ROOT / phase.REFERENCE_SOURCES[1]
MANIFEST = phase.ARTIFACTS / "manifest.json"
REVISIONS = ROOT / "tests" / "canonical" / "provenance-revisions.md"


# --------------------------------------------------------------------- the pin, as text


def test_the_image_installs_the_archive_tarball_checked_against_its_digest() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^FROM rocker/r-ver:[0-9.]+@sha256:[0-9a-f]{64}$", text, re.M)
    assert f"ARG CONCRETE_VERSION={phase.VERSION}" in text
    assert f"ARG CONCRETE_SHA256={phase.TARBALL_SHA256}" in text
    assert re.fullmatch(r"[0-9a-f]{64}", phase.TARBALL_SHA256)
    url = phase.TARBALL_URL.replace(phase.VERSION, "${CONCRETE_VERSION}")
    assert url in text
    checked = text.index("sha256sum -c -")
    assert text.index(url) < checked < text.index("R CMD INSTALL /tmp/concrete.tar.gz")
    assert "packageVersion('concrete')) == '${CONCRETE_VERSION}'" in text
    for commit in phase.NEAREST_COMMITS:
        assert commit in text


def _r_vector(source: str, name: str) -> list[str]:
    match = re.search(rf"^{name} <- c\((?P<body>[^)]*)\)", source, re.M)
    assert match is not None, name
    return [item.strip() for item in match.group("body").replace("\n", " ").split(",")]


def test_the_runner_transcribes_the_declared_times_formulas_and_cells() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert [float(value) for value in _r_vector(source, "times")] == list(law.TIMES)
    assert _r_vector(source, "causes") == [
        f'"{code}" = "{label}"' for code, label in law.CAUSES.items()
    ]
    for name, formula in phase.FORMULAS.items():
        assert f'{name} = "{formula}"' in source
    for cell, (hazards, censoring, treatment, folds) in phase.CONCRETE_CELLS.items():
        block = source[source.index(f"  {cell} = list(") :]
        block = block[: block.index("folds = ") + 20]

        def quoted(values: tuple[str, ...]) -> str:
            return "c(" + ", ".join(f'"{value}"' for value in values) + ")"

        assert f"hazards = {quoted(hazards)}" in block, cell
        assert f"censoring = {quoted(censoring)}" in block, cell
        assert f"treatment = {quoted(treatment)}" in block, cell
        assert f"folds = {folds}L" in block, cell
    # concrete's update defaults and its positivity bound, which cleverly's "auto" equals.
    assert "MaxUpdateIter = 500, OneStepEps = 0.1, MinNuisance = bound" in source
    assert "bound <- 5 / sqrt(n) / log(n)" in source
    assert "RenameCovs = FALSE" in source
    assert "Simultaneous = FALSE" in source


def test_the_runner_names_each_target_as_the_engine_does() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'sprintf("cif_regimen[%s, %s @ t=%g]", arm, cause, time)' in source
    assert 'sprintf("ate_regimen[treated vs control, %s @ t=%g]", cause, time)' in source
    assert law.level_name("treated", "cause1", 1.12) == "cif_regimen[treated, cause1 @ t=1.12]"
    assert law.difference_name("cause2", 0.46) == (
        "ate_regimen[treated vs control, cause2 @ t=0.46]"
    )
    assert 'arms <- c(treated = "A=1", control = "A=0")' in source


def test_the_runner_writes_the_phase_columns() -> None:
    from tests.unit.test_canonical_runner_parity import published_keys

    assert published_keys(RUNNER.read_text(encoding="utf-8")) == phase.ROW_COLUMNS


# -------------------------------------------------------------- the hazard construction


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return law.sample(600, 20261119)


def _cause1(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = frame[["Z", *law.BASELINE]].to_numpy(dtype=float)
    return X, frame["time"].to_numpy(dtype=float), (frame["event"] == 1).to_numpy(float)


def test_the_concrete_hazard_of_the_treatment_model_is_breslow(frame: pd.DataFrame) -> None:
    X, time, event = _cause1(frame)
    grid = np.linspace(0.05, 2.32, 40)
    ours = phase.ConcreteCoxHazard("treatment").fit(X, time, event).cumulative_hazard(X, grid)
    breslow = CoxHazard("treatment").fit(X, time, event).cumulative_hazard(X, grid)
    np.testing.assert_allclose(ours, breslow, rtol=1e-10)


def test_the_concrete_hazard_takes_the_treatment_models_baseline(frame: pd.DataFrame) -> None:
    """A nonzero witness: with covariates the construction is not the selected model's Breslow."""
    X, time, event = _cause1(frame)
    grid = np.linspace(0.05, 2.32, 40)
    model = phase.ConcreteCoxHazard("interaction").fit(X, time, event)
    breslow = CoxHazard("interaction").fit(X, time, event).cumulative_hazard(X, grid)
    ours = model.cumulative_hazard(X, grid)
    assert np.max(np.abs(ours / breslow - 1.0)) > 0.05
    # The construction written out: the treatment-only baseline at Z = 0 (d over the sum of
    # exp(gamma Z) in the risk set) times exp((x - m) beta), m zero on every 0/1 column.
    gamma = CoxHazard("treatment").fit(X, time, event).coef_[0]
    events = np.sort(time[event > 0])
    jumps = np.array([1.0 / np.exp(gamma * X[time >= u, 0]).sum() for u in events])
    base = np.array([jumps[events <= t].sum() for t in grid])
    design = np.hstack([X, X[:, :1] * X[:, 1:]])
    means = design.mean(axis=0)
    uncentered = [0, 1, 2, 4, 5, 6, 8]  # Z, X1, X2, X4 and their products with Z
    assert np.all(np.isin(design[:, uncentered], (0.0, 1.0)))
    means[uncentered] = 0.0
    risk = np.exp((design - means) @ model.model_.coef_)
    np.testing.assert_allclose(ours, risk[:, None] * base[None, :], rtol=1e-9)
    assert means[3] != 0.0 and means[7] != 0.0, "X3 and Z X3 are centred"


def test_the_concrete_hazard_clones_as_the_engine_clones_a_learner() -> None:
    from cleverly.survival.cox import _clone

    clone = _clone(phase.ConcreteCoxHazard("interaction"))
    assert isinstance(clone, phase.ConcreteCoxHazard) and clone.terms == "interaction"


# ------------------------------------------------------------------------- the settings


def test_the_parity_fit_is_concretes_form() -> None:
    subject = phase.engine("parity")
    assert subject.tilt == "factual"
    assert subject.norm == "euclidean"
    assert subject.cross_fit is False
    assert subject.tol == pytest.approx(1.0 / (3.0 * np.sqrt(phase.N) * np.log(phase.N)))
    assert subject.variance == "influence_curve"
    assert subject.max_iter == 500 and subject.step_size == 0.1
    assert subject.g_bounds == "auto"
    assert [type(m) for m in subject.hazard_library] == [phase.ConcreteCoxHazard]
    assert [m.terms for m in subject.hazard_library] == ["interaction"]
    assert [m.terms for m in subject.censoring_library] == ["main"]


def test_the_wrong_hazard_cell_differs_only_in_the_tilt_and_the_baseline() -> None:
    factual = phase.engine("hazard_wrong_factual")
    default = phase.engine("hazard_wrong")
    assert (factual.tilt, default.tilt) == ("factual", "all")
    assert default.norm == "covariance" and default.tol is None
    assert [m.terms for m in factual.hazard_library] == ["treatment"]
    assert [repr(m) for m in default.hazard_library] == ["CoxHazard('treatment')"]
    assert [repr(m) for m in default.censoring_library] == ["CoxHazard('main')"]


def test_the_treatment_library_is_the_logistic_alone_inside_a_parity_fit() -> None:
    small = law.sample(300, 7)
    result = phase.fit(small, "hazard_wrong")
    assert result.record.propensity.names == ("logistic",)
    default = phase.registered.engine(cross_fit=False)
    assert default.treatment_learner is None


def test_the_draws_and_seeds_are_the_phases_own() -> None:
    seeds = {phase.sample_seed(cell, r) for cell in phase.CELLS for r in range(50)}
    assert len(seeds) == 150
    registered = {phase.registered.replicate_seed(phase.STUDY, "both_cf_1k", r) for r in range(50)}
    assert not seeds & registered
    assert phase.seed_rows(3)["seed"].between(0, 2**31 - 1).all()
    pd.testing.assert_frame_equal(phase.draw("p1_parity", 4, 50), phase.draw("p1_parity", 4, 50))


# ------------------------------------------------------------------- the reading tables


def _synthetic_rows() -> pd.DataFrame:
    """Rows of a tiny run where concrete equals cleverly's factual fit plus a known offset."""
    records = []
    for cell in phase.CELLS:
        for replicate in range(200):
            for name in phase.ESTIMANDS:
                truth = float(law.TRUTH[name])
                base = truth + 0.02 * (-1) ** replicate  # mean zero over the draws
                implementations = [
                    (impl, base if impl == phase.SUBJECT_FACTUAL else base + 0.001)
                    for impl, _ in phase.FITS[cell]
                ]
                shift = 0.03 if cell == "p3_hazard_wrong" and name == phase.DECLARED_TARGET else 0
                implementations.append((phase.REFERENCE, base + 0.0005 + shift))
                for implementation, value in implementations:
                    records.append(
                        {
                            "implementation": implementation,
                            "scenario": cell,
                            "replicate": replicate,
                            "n": phase.N,
                            "estimand": name,
                            "truth": truth,
                            "estimate": value,
                            "inference_estimate": value,
                            "std_error": 0.02,
                            "ci_lower": value - 0.04,
                            "ci_upper": value + 0.04,
                            "inference_scale": "identity",
                            "covered": int(abs(value - truth) <= 0.04),
                            "initial_estimate": value,
                            "ic_std_error": 0.02,
                            "steps": 3.0,
                            "converged": 1.0,
                            "seconds": 1.0,
                        }
                    )
    return pd.DataFrame(records, columns=list(phase.ROW_COLUMNS))


def test_the_tables_read_the_declared_outcomes() -> None:
    tables = phase.tables(_synthetic_rows())
    parity = tables["parity.csv"]
    declared = parity[parity["cell"] == "p1_parity"]
    np.testing.assert_allclose(declared["max_abs_difference"], 0.0005, rtol=1e-6)
    assert declared["within_tolerance"].all() and (declared["fits"] == 200).all()
    verdicts = tables["verdicts.csv"].set_index("reading")["verdict"]
    assert verdicts.to_dict() == {
        "p1_estimates_agree": "pass",
        "p3_concrete_bias_above_margin": "pass",
        "p3_cleverly_bias_within_margin": "pass",
    }


def test_a_parity_gap_above_the_tolerance_fails_its_reading() -> None:
    """Mutation control: an offset of 0.005 on one target turns the P1 reading red."""
    rows = _synthetic_rows()
    moved = (
        (rows["implementation"] == phase.REFERENCE)
        & (rows["scenario"] == "p1_parity")
        & (rows["estimand"] == phase.ESTIMANDS[0])
        & (rows["replicate"] == 7)
    )
    rows.loc[moved, "estimate"] += 0.005
    verdicts = phase.tables(rows)["verdicts.csv"].set_index("reading")["verdict"]
    assert verdicts["p1_estimates_agree"] == "fail"


def test_a_concrete_without_bias_fails_the_robustness_reading() -> None:
    rows = _synthetic_rows()
    target = (
        (rows["implementation"] == phase.REFERENCE)
        & (rows["scenario"] == "p3_hazard_wrong")
        & (rows["estimand"] == phase.DECLARED_TARGET)
    )
    rows.loc[target, "estimate"] -= 0.03
    verdicts = phase.tables(rows)["verdicts.csv"].set_index("reading")["verdict"]
    assert verdicts["p3_concrete_bias_above_margin"] == "fail"


def test_the_reference_reader_keeps_every_float(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = _synthetic_rows()
    rows = rows[rows["implementation"] == phase.REFERENCE].copy()
    rows["estimate"] = rows["estimate"] + 1e-17 * np.arange(len(rows))
    path = tmp_path / "concrete.csv"
    rows.to_csv(path, index=False)
    read = phase.read_reference(path)
    np.testing.assert_array_equal(read["estimate"].to_numpy(), rows["estimate"].to_numpy())


# ------------------------------------------------------------- a committed run, replayed

committed = pytest.mark.skipif(
    not MANIFEST.exists(), reason="the concrete phase has not run; run_phase.py --publish writes it"
)


def _manifest() -> dict:  # type: ignore[type-arg]
    return json.loads(MANIFEST.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _judged() -> dict[tuple[str, str], str]:
    """``(source, recorded) -> current`` of each result-neutral declaration in the ledger."""
    columns = ("source", "recorded", "current", "judgement")
    return {
        (row["source"].strip("`"), row["recorded"].strip("`")): row["current"].strip("`")
        for row in pipe_table(REVISIONS, columns)
        if row["judgement"].startswith("result-neutral:")
    }


@committed
def test_the_image_runner_and_harness_are_the_bytes_that_ran() -> None:
    """The container and R-runner hashes are gated, as every R comparator's are."""
    recorded = _manifest()["reference_sha256"]
    assert set(recorded) == set(phase.REFERENCE_SOURCES)
    judged = _judged()
    for name, digest in recorded.items():
        current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert current in (digest, judged.get((name, digest))), (
            f"{name} changed after the concrete phase ran. A result-determining edit reruns the "
            f"phase; a result-neutral one is declared in {REVISIONS.name}"
        )


@committed
def test_the_manifest_records_the_declared_run() -> None:
    manifest = _manifest()
    assert manifest["declaration"] == json.loads(json.dumps(phase.declaration()))
    for name, digest in manifest["sha256"].items():
        path = phase.ARTIFACTS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


@committed
def test_every_table_is_a_function_of_the_committed_rows() -> None:
    rows = read_rows(phase.ARTIFACTS / "replicates.parquet")
    for name, table in phase.tables(rows).items():
        committed_table = pd.read_csv(phase.ARTIFACTS / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(
            table.reset_index(drop=True), committed_table, check_dtype=False, rtol=1e-12
        )


@committed
@pytest.mark.parametrize("cell", phase.CELLS)
def test_the_first_draw_of_each_cell_refits_to_its_committed_rows(cell: str) -> None:
    rows = read_rows(phase.ARTIFACTS / "replicates.parquet")
    implementations = [implementation for implementation, _ in phase.FITS[cell]]
    held = rows[
        (rows["scenario"] == cell)
        & (rows["replicate"] == 0)
        & rows["implementation"].isin(implementations)
    ]
    if held.empty:
        pytest.skip(f"the first draw of {cell} failed in the committed run")
    refit = pd.DataFrame(phase.fit_replicate(cell, 0))
    keys = ["implementation", "estimand"]
    joined = held.merge(refit, on=keys, suffixes=("", "_refit"))
    assert len(joined) == len(held)
    for column in ("estimate", "std_error", "initial_estimate"):
        np.testing.assert_allclose(joined[column], joined[f"{column}_refit"], rtol=1e-9)
