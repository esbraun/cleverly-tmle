"""The declared design of ``parametric-gformula`` (X45).

Every number here was fixed before any registered run.  The design pilot (about 200 to 1,200
replicates per label on an independent stream, see ``DESIGN_PILOT``) supplied the sampling
facts; the laws supply the rest.  Each test below proves one item of the design.

* **Specification.**  The main-terms models are the law.  A fit on the exact support of the
  law (a weighted population fit) recovers every coefficient of ``G1``, ``G3`` and ``G4``.  On
  ``G2``, whose Gaussian support is not finite, a large sample sits within the sampling error
  of the law.  A deliberate mutation of a coefficient is detected by the same check.
* **Truths.**  Binary and categorical paths are enumerated, each Gaussian variable is
  integrated by Gauss-Hermite, and ``G4`` has a closed form.  A forward simulation, sized so
  its standard error is at most 5% of the bias tolerance (a quarter of a sampling SD at the
  calibration size), agrees with every truth.
* **Existence of the MLE.**  The probability that every GLM of a fit set exists, from
  the pinned probe, is at least 0.999 at the first rung of the shared ladder.
* **The D-terms.**  Each kind of derivative term (covariate, outcome, treatment and censoring
  models) carries a share of the standard error at the calibration size.
* **The ICE limit.**  The main-terms ICE fit converges to a value away from the truth on
  ``G2``, by an amount the paired fit set detects with power at least 0.95.
* **The Monte Carlo control.**  The ``n_draws`` 1 control predicts an SE ratio below the
  calibration band wherever the simulation term carries a large share of the variance.
* **Sizing.**  The replicates of each calibration label give a pass probability of at least
  0.90 under the SE-ratio rule.

Seeds: ``SEED`` and ``RESAMPLING_SEED`` are new, and no other registered study uses either.
"""

from __future__ import annotations

import importlib
import math
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, norm

from cleverly.learners import GLM
from cleverly.longitudinal import ParametricGFormula
from tests.studies import canonical_parametric_gformula as study
from tests.studies import parametric_gformula_laws as laws
from tests.studies import parametric_gformula_properties as properties
from tests.studies.canonical_full_refit_bootstrap import calibration_pass_probability
from tests.studies.evidence import descriptions
from tests.studies.evidence.properties import TRUTH_ABSOLUTE_FLOOR
from tests.studies.evidence.registry import Margins, registered

gf: Any = importlib.import_module("cleverly.longitudinal.gformula")

#: The bias tolerance is a quarter of a sampling SD (the shared ``bias_equivalent`` margin).
BIAS_MARGIN = Margins().standardized_bias


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert record.margins == Margins()
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N) == (1_000, 8_000)
    assert (study.SEED, study.RESAMPLING_SEED) == (20264501, 2026450101)
    assert study.LADDER == (2_000, 4_000, 8_000)
    assert (study.N_DRAWS, study.N_DRAWS_ONE, study.M_PRIME) == (10, 1, 500)
    assert study.CALIBRATION_N == 2_000
    assert record.calibration_efficiency_ratio is False
    assert record.reference is None
    assert (record.pending_reference, record.comparison) == (
        "pygformula",
        "matching implementation",
    )
    assert record.failure_bound == {study.IMPLEMENTATION: 0.01, study.REFERENCE: 0.01}
    assert properties.SHRUNKEN_SE_FACTOR == 0.70
    assert properties.LADDER_REPLICATES == 1_000
    assert properties.PAIRED_N == {"G1": 2_000, "G2": 32_000}
    assert study.ESTIMANDS == {
        "G1": (
            "ey_regimen[treat_all]",
            "ey_regimen[treat_none]",
            "ey_regimen[rule]",
            "ey_regimen[odds x2]",
            "ate_regimen[treat_all vs treat_none]",
        ),
        "G2": (
            "ey_regimen[treat_all]",
            "ey_regimen[treat_none]",
            "ey_regimen[policy_b]",
            "ate_regimen[treat_all vs treat_none]",
        ),
        "G3": (
            "cif_regimen[treat_all_4, relapse @ t=2]",
            "cif_regimen[treat_all_4, relapse @ t=4]",
            "cif_regimen[treat_all_4, death @ t=4]",
            "ate_regimen[treat_all_4 vs treat_none_4, relapse @ t=4]",
        ),
        "G4": (
            "ey_regimen[shift_half]",
            "ey_regimen[dose_half]",
            "ate_regimen[shift_half vs natural course]",
        ),
    }


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = properties.generate_property_rows(n_jobs=1, budget=2)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert int(rows["failed_replicates"].max()) == 0
    for (family, name), cell in declared.items():
        selected = rows.loc[(rows["property"] == family) & (rows["cell"] == name)]
        np.testing.assert_allclose(
            selected["truth"],
            cell.dgp.truth()[cell.estimand],
            rtol=1e-12,
            atol=TRUTH_ABSOLUTE_FLOOR,
        )
        assert set(selected["n"]) == {cell.n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = properties.summarize_properties(rows)
    assert {(row.property, row.cell) for row in summary.itertuples()} == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }


def test_no_two_families_share_a_declared_stream() -> None:
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in streams.values())


def test_the_study_draws_from_its_own_seeds() -> None:
    mine = {study.STUDY.seed, study.STUDY.resampling_seed}
    assert len(mine) == 2
    others = {
        seed
        for record in registered()
        if record.slug != study.STUDY.slug
        for seed in (record.seed, record.resampling_seed)
    }
    assert not mine & others


def test_every_key_the_study_publishes_has_a_description() -> None:
    descriptions.implementation(study.IMPLEMENTATION)
    for scenario, estimands in study.SCENARIOS.items():
        descriptions.scenario(scenario)
        for name in estimands:
            descriptions.estimand(name)
    for name, _ in properties.TARGETS.values():
        descriptions.estimand(name)
    for family, cells in study.PROPERTY_CELLS.items():
        descriptions.claim(family)
        for cell in cells:
            descriptions.cell(family, cell, role="positive")


def test_the_pending_comparator_declares_the_sources_the_manifest_records() -> None:
    folder = study.STUDY.artifacts
    for name in ("Dockerfile", "requirements.txt", "run_pygformula.py", "regenerate.py"):
        assert (folder / name).is_file(), name
    pinned = (folder / "requirements.txt").read_text()
    assert f"pygformula=={study.PYGFORMULA}" in pinned
    assert re.search(
        r"^FROM python:3\.10-slim-bookworm@sha256:[0-9a-f]{64}$",
        (folder / "Dockerfile").read_text(),
        re.M,
    )
    assert f"ARG PYGFORMULA={study.PYGFORMULA}" in (folder / "Dockerfile").read_text()
    assert study.STUDY.comparator_state == "pending"


# ------------------------------------------------------------------- (a) specification


def _fit_book(law: str, frame: pd.DataFrame, weights: str | None = None) -> Any:
    """The fitted model book of the study's ``ParametricGFormula`` on ``frame``.

    The hook returns zero exact terms, so the call stops after the models are fitted and the
    book is the one the production estimator would simulate from.
    """
    store: dict[str, Any] = {}

    def hook(book: Any, plan: Any, cells: Any) -> Any:
        store["book"] = book
        return gf.ExactTerms(
            np.zeros((book.data.n, len(cells))),
            {m.key: np.zeros((len(cells), m.n_parameters)) for m in book.stacked},
        )

    target = study.law_of(law)
    options: dict[str, Any] = {
        "reference": study.REFERENCES[law],
        "covariate_types": study.COVARIATE_TYPES[law],
        "horizons": study.HORIZONS.get(law),
        "n_draws": 2,
        "treatment_learner": GLM(),
        "simultaneous": False,
        "random_state": 1,
    }
    if target.has_censoring:
        options["censoring_learner"] = GLM()
    estimator = ParametricGFormula(study.regimens(law), **options)
    estimator._exact = hook
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        columns: dict[str, Any] = target.fit_columns(continuous_treatment=law in ("G4", "G4Q"))
        if weights:
            columns["weights"] = weights
        estimator.fit(frame, **columns)
    return store["book"]


def _variable(model: Any) -> str:
    """The law variable a fitted model stands for, from its key."""
    parts = model.key.split(":")
    if parts[0] == "covariate":
        return parts[2]
    if parts[0] == "treatment":
        return f"A{parts[1]}"
    if parts[0] == "outcome":
        return "Y"
    return f"E{parts[1]}"


def _original_scale(book: Any, model: Any, coef: np.ndarray) -> np.ndarray:
    """The outcome model's coefficients on the outcome's own scale (fitted on ``(Y - a) / r``)."""
    if model.role != "outcome":
        return coef
    out = np.asarray(coef, dtype=float) * book.scaler.range
    out[..., 0] += book.scaler.lower
    return out


def _distance(target: laws.Law, book: Any) -> dict[str, float]:
    """The largest gap, per model, between the fitted coefficients and the law's.

    A Gaussian model also compares its variance with the law's squared SD.  A coefficient the
    fit leaves out (an inactive column) must be zero in the law.
    """
    out: dict[str, float] = {}
    for model in book.models:
        name = _variable(model)
        truth = target.coefficients(name)
        glm = model.learner
        active = np.asarray(glm.active_)
        coef = _original_scale(book, model, np.asarray(glm.coef_, dtype=float))
        reference = truth[:, active] if model.response == "multi" else truth[active]
        gap = np.abs(coef.reshape(reference.shape) - reference).max()
        if (~active).any():
            gap = max(gap, float(np.abs(truth[..., ~active]).max()))
        if model.response == "normal":
            gap = max(gap, abs(model.sigma2 - target.var(name).sd ** 2))
        out[model.key] = float(gap)
    return out


@pytest.mark.parametrize(("law", "order"), [("G1", 12), ("G3", 12), ("G4", 3)])
def test_an_exact_support_fit_recovers_the_law(law: str, order: int) -> None:
    """The main-terms models are the law: a weighted fit on its support returns every term."""
    target = study.law_of(law)
    support = target.support(order)
    frame = support.drop(columns="prob").assign(w=support["prob"])
    gaps = _distance(target, _fit_book(law, frame, "w"))
    expected = {
        "G1": 6,
        "G3": 7,
        "G4": 6,
    }[law]
    # Covariate models of nodes 2 to T, the natural-treatment models, the outcome or the
    # hazards.  Censoring is handled by the risk set, so it has no model.
    assert len(gaps) == expected
    assert max(gaps.values()) < 1e-6, gaps


def test_a_large_sample_fit_of_g2_sits_within_its_sampling_error() -> None:
    """``G2`` has Gaussian covariates at every node, so its support is read from a sample."""
    target = study.law_of("G2")
    book = _fit_book("G2", target.sample(200_000, 11))
    worst = 0.0
    for model in book.models:
        name = _variable(model)
        glm = model.learner
        covariance = np.linalg.inv(glm.information(book.design(model), model.weights))
        if model.response == "normal":
            covariance = covariance * model.sigma2
        error = np.sqrt(np.diag(covariance))
        active = np.asarray(glm.active_)
        coef = _original_scale(book, model, np.asarray(glm.coef_, dtype=float)).reshape(-1)
        if model.role == "outcome":
            error = error * book.scaler.range
        z = (coef - target.coefficients(name)[active]) / error
        worst = max(worst, float(np.abs(z).max()))
    assert worst < 4.5


def test_the_specification_check_detects_a_wrong_law() -> None:
    """Two deliberate mutations, both caught: a wrong coefficient and a missing square."""
    target = study.law_of("G1")
    support = target.support(12)
    frame = support.drop(columns="prob").assign(w=support["prob"])
    book = _fit_book("G1", frame, "w")
    outcome = target.var("Y")
    changed = dict(outcome.coef)
    key = next(iter(changed))
    changed[key] = changed[key] + 0.2
    mutated = target.with_variable("Y", coef=changed)
    assert max(_distance(mutated, book).values()) > 0.1
    # G4Q carries a squared dose; the main-terms fit cannot recover it.
    quad = study.law_of("G4Q")
    support = quad.support(3)
    frame = support.drop(columns="prob").assign(w=support["prob"])
    gaps = _distance(quad, _fit_book("G4Q", frame, "w"))
    assert max(gaps.values()) > 0.05


# ------------------------------------------------------------------------- (b) truths

#: The plan labels whose truth the forward simulation re-derives, by law.
SIMULATED_PLANS = {
    "G1": ("treat_all", "treat_none", "rule", "odds x2"),
    "G2": ("treat_all", "treat_none", "policy_b"),
    "G3": ("treat_all_4", "treat_none_4"),
    "G4": ("natural course", "shift_half", "dose_half"),
    "G4Q": ("dose_half",),
}
#: The bias tolerance of each law, from its smallest calibration SD at the calibration size.
BATCH = 500_000


def tolerance(law: str) -> float:
    """The bias tolerance: a quarter of the smallest sampling SD of the law's labels at n = 2,000."""
    sds = [
        properties.NOISE_SD[label]
        for label in study.CALIBRATION_LABELS
        if label.split("_")[1] == law.lower()[:2]
    ]
    return BIAS_MARGIN * min(sds) / math.sqrt(study.CALIBRATION_N)


def _simulated_plan(law: str, label: str) -> dict[int, laws.NodePlan]:
    nodes = range(1, study.law_of(law).n_times + 1)
    if law in ("G4", "G4Q"):
        if label == "dose_half":
            return {t: laws.Fixed(0.5) for t in nodes}
        return {t: laws.ShiftPlan(0.5 if label == "shift_half" else 0.0) for t in nodes}
    return study._node_plans(law, label)


def _truth_name(law: str, label: str, cause: str = "relapse", horizon: int = 4) -> str:
    if law == "G3":
        return f"cif_regimen[{label}, {cause} @ t={horizon}]"
    return f"ey_regimen[{label}]"


@pytest.mark.parametrize("law", ["G1", "G2", "G3", "G4", "G4Q"])
def test_every_law_names_only_earlier_columns(law: str) -> None:
    study.law_of(law).check()


def test_the_gauss_hermite_truths_have_converged() -> None:
    """Order 12 is within 1% of the bias tolerance of a higher order, on every integrated law."""
    for law, label in (
        ("G1", "rule"),
        ("G1", "odds x2"),
        ("G3", "treat_all_4"),
        ("G2", "policy_b"),
    ):
        target = study.law_of(law)
        plan = _simulated_plan(law, label)
        horizons = study.HORIZONS.get(law)
        chunks = 12 if law == "G2" else 1
        low = target.g_formula(plan, order=12, chunks=chunks, horizons=horizons)
        high = target.g_formula(plan, order=16, chunks=16 if law == "G2" else 1, horizons=horizons)
        pairs = [(low[k], high[k]) for k in low] if isinstance(low, dict) else [(low, high)]
        for a, b in pairs:
            assert abs(a - b) < 0.01 * tolerance(law), (law, label, a, b)


@pytest.mark.parametrize("law", ["G4", "G4Q"])
def test_the_closed_forms_are_the_enumerated_means(law: str) -> None:
    """A fixed dose has the closed form of ``linear_means`` and the integrated form agree."""
    target = study.law_of(law)
    plan = {t: laws.Fixed(0.5) for t in range(1, 4)}
    integrated = target.g_formula(plan, order=6)
    closed = study.truths(law)["ey_regimen[dose_half]"]
    assert integrated == pytest.approx(closed, abs=1e-9)


@pytest.mark.parametrize("law", ["G1", "G2", "G3", "G4", "G4Q"])
def test_a_forward_simulation_agrees_with_every_truth(law: str) -> None:
    """The simulation is a different program from the sums, so a sign or a mask shows here.

    Its standard error is at most 5% of the bias tolerance, so a truth that is wrong by
    a fifth of the tolerance is four standard errors off.
    """
    target = study.law_of(law)
    truth = study.truths(law)
    for label in SIMULATED_PLANS[law]:
        plan = _simulated_plan(law, label)
        horizons = study.HORIZONS.get(law)
        values: dict[Any, list[float]] = {}
        errors: dict[Any, list[float]] = {}
        batches = 0
        while True:
            batches += 1
            got = target.monte_carlo(plan, BATCH, 7_000 + batches, horizons=horizons)
            for key, (value, error) in (got if isinstance(got, dict) else {None: got}).items():
                values.setdefault(key, []).append(value)
                errors.setdefault(key, []).append(error)
            spread = max(math.sqrt(float(np.sum(np.square(e)))) / batches for e in errors.values())
            if spread <= 0.05 * tolerance(law) or batches >= 400:
                break
        for key, batch_values in values.items():
            mean = float(np.mean(batch_values))
            error = math.sqrt(float(np.sum(np.square(errors[key])))) / batches
            name = _truth_name(law, label, *key) if key is not None else _truth_name(law, label)
            assert error <= 0.05 * tolerance(law), (law, label, key, error)
            assert abs(mean - truth[name]) < 4.0 * error, (law, label, key, mean, truth[name])


# ------------------------------------------------------------------------ the design pilot

#: ``label -> (replicates, mean SE over empirical SD, relative SD of the SE, sqrt(n) times the
#: empirical SD)`` of each calibration label at n = 2,000, from 1,200 replicates of the pilot on
#: the ``design_pilot`` stream of ``calibration_g1`` to ``calibration_g4`` (offset simulation
#: seeds, so no published cell reads them).  ``tests.studies.parametric_gformula_properties.
#: NOISE_SD`` is the last column.
DESIGN_PILOT: dict[str, tuple[int, float, float, float]] = {
    "nice_g1_treat_all": (1200, 1.0167, 0.0233, 0.815),
    "nice_g1_treat_none": (1200, 1.0190, 0.0417, 1.244),
    "nice_g1_rule": (1200, 0.9871, 0.0168, 0.640),
    "nice_g1_odds_x2": (1200, 0.9954, 0.0128, 0.635),
    "nice_g1_contrast": (1200, 1.0324, 0.0359, 1.672),
    "nice_g2_treat_all": (1200, 1.0153, 0.0184, 7.586),
    "nice_g2_treat_none": (1200, 0.9640, 0.0209, 8.963),
    "nice_g2_policy_b": (1200, 1.0013, 0.0128, 5.929),
    "nice_g2_contrast": (1200, 0.9637, 0.0248, 10.740),
    "nice_g3_relapse_h2": (1200, 0.9637, 0.0398, 0.583),
    "nice_g3_relapse_h4": (1200, 0.9854, 0.0423, 0.832),
    "nice_g3_death_h4": (1200, 0.9961, 0.0782, 0.645),
    "nice_g3_contrast_h4": (1200, 0.9691, 0.0314, 1.725),
    "nice_g4_shift": (1200, 1.0095, 0.0170, 3.889),
    "nice_g4_dose": (1200, 1.0220, 0.0233, 1.760),
    "nice_g4_contrast": (1200, 1.0318, 0.0267, 1.168),
}


def test_the_noise_scale_is_the_pilot_spread() -> None:
    assert set(DESIGN_PILOT) == set(study.CALIBRATION_LABELS) == set(properties.NOISE_SD)
    for label, (_, _, _, spread) in DESIGN_PILOT.items():
        assert properties.NOISE_SD[label] == pytest.approx(spread, abs=1e-3), label


def _shrunk_ratios() -> dict[str, float]:
    """Each label's SE ratio shrunk toward the pooled mean, by the pilot's own spread.

    The ratios of 16 labels spread more than their sampling errors say (``tau`` below), so each
    label is pulled toward the precision-weighted mean by the share of its variance that is
    sampling error.  This is the empirical-Bayes estimate of the true ratio of the label.
    """
    labels = list(DESIGN_PILOT)
    ratio = np.array([DESIGN_PILOT[x][1] for x in labels])
    count = np.array([DESIGN_PILOT[x][0] for x in labels], dtype=float)
    noise = np.array([DESIGN_PILOT[x][2] for x in labels])
    sampling = ratio**2 * (1.0 / (2.0 * (count - 1.0)) + noise**2 / count)
    pooled = float(np.average(ratio, weights=1.0 / sampling))
    tau2 = max(0.0, float(np.var(ratio, ddof=1)) - float(sampling.mean()))
    weight = tau2 / (tau2 + sampling)
    return dict(zip(labels, pooled + weight * (ratio - pooled), strict=True))


def _pass_probability(replicates: int, label: str, ratio: float) -> float:
    margins = study.STUDY.margins
    return calibration_pass_probability(
        replicates,
        true_se_ratio=ratio,
        se_noise=DESIGN_PILOT[label][2],
        confidence_level=margins.confidence_level,
        se_band=margins.calibration_se_ratio,
        coverage_band=margins.calibration_coverage,
    )


def test_each_calibration_label_is_sized_for_its_pass_probability() -> None:
    """A label passes the SE-ratio rule with probability at least 0.90 at its law's replicates.

    The true ratio of each label is its empirical-Bayes estimate from the pilot, and the SE
    noise is the pilot's.  The next smaller multiple of 400 fails somewhere, so the size is not
    padded.
    """
    shrunk = _shrunk_ratios()
    worst = {}
    for label in study.CALIBRATION_LABELS:
        law = label.split("_")[1].upper()
        replicates = properties.CALIBRATION_REPLICATES[law]
        probability = _pass_probability(replicates, label, shrunk[label])
        assert probability >= 0.90, (label, replicates, probability)
        worst[label] = _pass_probability(replicates - 400, label, shrunk[label])
    assert min(worst.values()) < 0.90
    assert set(properties.CALIBRATION_REPLICATES.values()) == {2_400}


def test_the_pilot_shows_no_calibration_cell_outside_the_band() -> None:
    for label, (_, ratio, _, _) in DESIGN_PILOT.items():
        low, high = study.STUDY.margins.calibration_se_ratio
        assert low + 0.02 < ratio < high - 0.02, label
    assert max(abs(r - 1.0) for r in _shrunk_ratios().values()) < 0.02


# ------------------------------------------------------------------------- (c) the D-terms

#: The largest mean SE share of each kind of derivative term over the labels of a law, from 40
#: design draws of n = 2,000.  The share of a kind is one minus the SE of the stacked curve
#: without the kind over the SE of the curve.
TERM_SHARES: dict[str, dict[str, float]] = {
    "G1": {
        "plug-in": 0.039,
        "covariate/binary": 0.026,
        "treatment/binary": 0.005,
        "outcome": 0.737,
    },
    "G2": {
        "plug-in": 0.357,
        "covariate/normal": 0.294,
        "covariate/binary": 0.035,
        "outcome": 0.054,
    },
    "G3": {"plug-in": 0.013, "covariate/binary": 0.007, "hazard": 0.949},
    "G4": {
        "plug-in": 0.520,
        "covariate/normal": 0.061,
        "treatment/normal": 0.040,
        "outcome": 0.225,
    },
}
TERM_FLOOR = 0.03
#: A kind whose share stays below the floor in every law is covered by an exact-law witness
#: (``TestExactTermsLawB`` of ``tests/unit/test_parametric_gformula.py``), not by the SE.
WITNESS_ONLY = {"treatment/binary"}


def test_every_derivative_term_kind_carries_a_share_or_a_named_witness() -> None:
    best: dict[str, float] = {}
    for kinds in TERM_SHARES.values():
        for kind, share in kinds.items():
            best[kind] = max(best.get(kind, 0.0), share)
    assert set(best) == {
        "plug-in",
        "covariate/binary",
        "covariate/normal",
        "treatment/binary",
        "treatment/normal",
        "outcome",
        "hazard",
    }
    for kind, share in best.items():
        if kind in WITNESS_ONLY:
            assert share < TERM_FLOOR, kind
        else:
            assert share >= TERM_FLOOR, (kind, share)


def _curve_blocks(
    law: str, frame: pd.DataFrame, seed: int
) -> dict[tuple[Any, ...], dict[str, np.ndarray]]:
    """Each estimand's influence curve split into the plug-in and one block per model kind."""
    captured: list[tuple[Any, Any, list[Any], Any]] = []
    original = gf._simulate_plan

    def spy(book: Any, plan: Any, cells: Any, settings: Any) -> Any:
        result = original(book, plan, cells, settings)
        captured.append((book, plan, list(cells), result))
        return result

    gf._simulate_plan = spy
    try:
        study.fit("nice", law, frame, seed=seed)
    finally:
        gf._simulate_plan = original
    out: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
    for book, plan, cells, result in captured:
        weights = np.asarray(book.data.weights, dtype=float)
        psi = (weights @ result.phi) / weights.sum()
        kinds = {
            m.key: f"{m.role}/{m.response}" if m.role in ("covariate", "treatment") else m.role
            for m in book.models
        }
        for index, cell in enumerate(cells):
            block = {"plug-in": weights / weights.mean() * (result.phi[:, index] - psi[index])}
            for key, derivative in result.derivatives.items():
                model = next(m for m in book.models if m.key == key)
                block[kinds[key]] = block.get(kinds[key], 0.0) + (
                    book.projection(model) @ derivative[index]
                )
            out[(plan.label, cell.cause, cell.horizon)] = block
    return out


def _shares(blocks: dict[str, np.ndarray]) -> dict[str, float]:
    total = sum(blocks.values())
    se = np.sqrt(np.mean(total**2))
    return {
        kind: 1.0 - float(np.sqrt(np.mean((total - block) ** 2))) / float(se)
        for kind, block in blocks.items()
    }


@pytest.mark.parametrize("law", ["G1", "G2", "G3", "G4"])
def test_a_recomputed_draw_shows_the_pinned_term_shares(law: str) -> None:
    """Four fresh draws give each kind's largest share within 0.07 of the 40-draw table."""
    draws = 4
    largest: dict[str, list[float]] = {}
    for r in range(draws):
        frame = properties.draw(law, study.CALIBRATION_N, 90_000 + r)
        per_kind: dict[str, float] = {}
        for blocks in _curve_blocks(law, frame, seed=r).values():
            for kind, share in _shares(blocks).items():
                per_kind[kind] = max(per_kind.get(kind, 0.0), share)
        for kind, share in per_kind.items():
            largest.setdefault(kind, []).append(share)
    assert set(largest) == set(TERM_SHARES[law])
    for kind, values in largest.items():
        assert float(np.mean(values)) == pytest.approx(TERM_SHARES[law][kind], abs=0.07), kind


# ------------------------------------------------------------------- (d) the ICE limit

#: The population limit of the main-terms ICE fit minus the truth, from a weighted fit on the
#: exact support (Gauss-Hermite order 8 on ``G2``, 12 on ``G1``).  The limit is a fixed amount,
#: so its standardized size grows with the square root of n.
ICE_BIAS: dict[str, dict[str, float]] = {
    "G1": {
        "ey_regimen[treat_all]": 0.00063,
        "ey_regimen[treat_none]": 0.00033,
        "ate_regimen[treat_all vs treat_none]": 0.00030,
    },
    "G2": {
        "ey_regimen[treat_all]": 0.07493,
        "ey_regimen[treat_none]": 0.04284,
        "ate_regimen[treat_all vs treat_none]": 0.03209,
    },
}
#: ``label -> (NICE, ICE)`` empirical SD times ``sqrt(n)`` of the ``G2`` paired pilot at
#: n = 8,000 (300 replicates).
ICE_PILOT_SD = {
    "pair_g2_treat_all": (7.309, 8.679),
    "pair_g2_treat_none": (8.618, 13.483),
    "pair_g2_contrast": (10.393, 14.310),
}
ICE_ESTIMAND = {
    "pair_g2_treat_all": "ey_regimen[treat_all]",
    "pair_g2_treat_none": "ey_regimen[treat_none]",
    "pair_g2_contrast": "ate_regimen[treat_all vs treat_none]",
}


def _ice_limit(law: str, order: int) -> dict[str, float]:
    """``limit - truth`` of the static ICE fit on the weighted support at ``order``."""
    from cleverly.longitudinal import SequentialGComputation

    target = study.law_of(law)
    support = target.support(order)
    frame = support.drop(columns="prob").assign(w=support["prob"])
    options: dict[str, Any] = {
        "treatment_learner": GLM(),
        "outcome_learner": GLM(family="gaussian") if law == "G2" else GLM(),
        "simultaneous": False,
    }
    if target.has_censoring:
        options["censoring_learner"] = GLM()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SequentialGComputation(
            study.regimens(law, static_only=True), reference=study.REFERENCES[law], **options
        ).fit(frame, weights="w", **target.fit_columns())
    nodes = range(1, target.n_times + 1)
    levels = {
        "treat_all": target.g_formula(
            dict.fromkeys(nodes, laws.Fixed(1.0)), order=order, chunks=6 if law == "G2" else 1
        ),
        "treat_none": target.g_formula(
            dict.fromkeys(nodes, laws.Fixed(0.0)), order=order, chunks=6 if law == "G2" else 1
        ),
    }
    out = {f"ey_regimen[{k}]": fitted[f"ey_regimen[{k}]"].psi - v for k, v in levels.items()}
    out["ate_regimen[treat_all vs treat_none]"] = (
        out["ey_regimen[treat_all]"] - out["ey_regimen[treat_none]"]
    )
    return out


@pytest.mark.parametrize(("law", "order"), [("G1", 12), ("G2", 6)])
def test_the_main_terms_ice_limit_is_the_pinned_bias(law: str, order: int) -> None:
    """The population limit of ICE; ``G2`` at order 6 sits within 0.002 of the order-8 pin."""
    found = _ice_limit(law, order)
    for name, pinned in ICE_BIAS[law].items():
        assert found[name] == pytest.approx(pinned, abs=0.002 if law == "G2" else 2e-4), name


def test_the_g1_ice_limit_is_negligible_so_its_pairs_report() -> None:
    """Under a tenth of the bias tolerance, so no G1 pair can be gated."""
    for label in ("treat_all", "treat_none", "contrast"):
        name = {
            "treat_all": "ey_regimen[treat_all]",
            "treat_none": "ey_regimen[treat_none]",
            "contrast": "ate_regimen[treat_all vs treat_none]",
        }[label]
        sd = properties.NOISE_SD[f"nice_g1_{label}"] / math.sqrt(study.CALIBRATION_N)
        assert abs(ICE_BIAS["G1"][name]) < 0.2 * BIAS_MARGIN * sd, label
    assert not [x for x in study.ICE_GATED if x.startswith("pair_g1")]


def ice_power(label: str, n: int, replicates: int) -> tuple[float, float]:
    """The chances that the ICE arm is discriminated and that the pair clears the displacement.

    The standardized bias of the ICE fit is ``bias sqrt(n) / sd_ice``.  Its 99% interval leaves
    the equivalence margin when the bias exceeds the margin by ``z`` standard errors, where
    ``se = sqrt((1 + b^2 / 2) / R)``.  The displacement is the bias over the NICE spread, with
    the paired difference of spread ``sqrt(sd_nice^2 + sd_ice^2)`` (no covariance credit).
    """
    nice_sd, ice_sd = ICE_PILOT_SD[label]
    bias = abs(ICE_BIAS["G2"][ICE_ESTIMAND[label]])
    b = bias * math.sqrt(n) / ice_sd
    se_b = math.sqrt((1.0 + b**2 / 2.0) / replicates)
    z = float(norm.ppf(0.5 + study.STUDY.margins.confidence_level / 2.0))
    discriminated = float(norm.cdf((b - BIAS_MARGIN - z * se_b) / se_b))
    shift = bias * math.sqrt(n) / nice_sd
    se_shift = math.sqrt((nice_sd**2 + ice_sd**2) / replicates) / nice_sd
    displaced = float(norm.cdf((shift - properties.ICE_DISPLACEMENT) / se_shift))
    return discriminated, displaced


@pytest.mark.parametrize("label", study.ICE_GATED)
def test_each_gated_ice_label_has_predictive_power_at_the_paired_size(label: str) -> None:
    n = properties.PAIRED_N["G2"]
    replicates = properties.PAIRED_REPLICATES["G2"]
    discriminated, displaced = ice_power(label, n, replicates)
    assert discriminated >= 0.95, (label, discriminated)
    assert displaced >= 0.95, (label, displaced)


def test_the_contrast_is_unpowered_at_the_ladder_top_so_the_paired_size_is_larger() -> None:
    """At n = 8,000 the contrast cannot be discriminated, which is why the set runs larger."""
    discriminated, _ = ice_power("pair_g2_contrast", 8_000, properties.PAIRED_REPLICATES["G2"])
    assert discriminated < 0.5
    assert properties.PAIRED_N["G2"] == 32_000
    assert set(study.ICE_GATED) == set(ICE_PILOT_SD)
    assert properties.ICE_NECESSITY.labels == study.ICE_GATED
    assert properties.ICE_NECESSITY.threshold == properties.ICE_DISPLACEMENT == 0.35


# ------------------------------------------------------------- (e) the Monte Carlo control

#: The mean reported Monte Carlo share of the variance at ``n_draws`` 1 (200 design draws).
MC_SHARE = {
    "nice_g1_treat_all": 0.0114,
    "nice_g1_treat_none": 0.0123,
    "nice_g2_treat_all": 0.2018,
    "nice_g2_treat_none": 0.1643,
    "nice_g4_shift": 0.1260,
    "nice_g4_dose": 0.0862,
}
#: The accounting pilot at n = 2,000 (400 replicates): the SE ratio of the ``n_draws`` 1 fit and
#: of the ``mc_removed`` control that takes the SE of the independent ``m'`` fit.
ACCOUNTING_PILOT = {
    "nice_g1_treat_all": (1.0338, 1.0278),
    "nice_g1_treat_none": (1.0130, 1.0066),
    "nice_g2_treat_all": (0.9549, 0.8538),
    "nice_g2_treat_none": (1.0457, 0.9570),
    "nice_g4_shift": (0.9865, 0.9218),
    "nice_g4_dose": (0.9824, 0.9396),
}


def predicted_ratio(label: str, base: float) -> float:
    """The SE ratio of the control: the fit's ratio times the root of what is left of the variance."""
    share = MC_SHARE[label]
    return base * math.sqrt(1.0 - share * (1.0 - 1.0 / study.M_PRIME))


def control_power(label: str, replicates: int) -> float:
    """The chance that the whole 99% SE-ratio interval of ``mc_removed`` lies below the band."""
    base = _shrunk_ratios()[label]
    ratio = predicted_ratio(label, base)
    noise = DESIGN_PILOT[label][2]
    spread = ratio * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
    z = float(norm.ppf(0.5 + study.STUDY.margins.confidence_level / 2.0))
    low_edge = study.STUDY.margins.calibration_se_ratio[0]
    return float(norm.cdf((low_edge - z * spread - ratio) / spread))


def test_the_pilot_ratios_follow_the_predicted_ratio_of_ratios() -> None:
    """``mc_removed / n_draws_1`` is ``sqrt(1 - share (1 - 1/m'))``, to within the pilot error."""
    for label, (base, removed) in ACCOUNTING_PILOT.items():
        assert removed / base == pytest.approx(predicted_ratio(label, 1.0), abs=0.012), label


def test_only_labels_with_power_at_least_95_percent_are_gated() -> None:
    gated = {
        cell.removesuffix("__mc_removed")
        for cell, side in properties.GATED_DIRECTIONS.items()
        if side == "below"
    }
    assert set(properties.GATED_DIRECTIONS.values()) <= {"below"}
    for label in MC_SHARE:
        law = label.split("_")[1].upper()
        power = control_power(label, properties.ACCOUNTING_REPLICATES[law])
        assert (label in gated) == (power >= 0.95), (label, power)
    assert gated == {"nice_g2_treat_all"}
    assert tuple(MC_SHARE) == study.MC_LABELS


def test_the_gated_control_is_powered_only_at_the_larger_size() -> None:
    """The interval is too wide to clear the band at 2,400 replicates, and at 6,000 by a hair."""
    assert control_power("nice_g2_treat_all", 2_400) < 0.8
    assert (
        control_power("nice_g2_treat_all", 6_000)
        < 0.95
        <= control_power("nice_g2_treat_all", 6_400)
    )
    assert properties.ACCOUNTING_REPLICATES == {"G1": 2_400, "G2": 6_400, "G4": 2_400}


def test_the_one_draw_positive_cell_is_sized_like_a_calibration_cell() -> None:
    shrunk = _shrunk_ratios()
    for label in MC_SHARE:
        law = label.split("_")[1].upper()
        assert (
            _pass_probability(properties.ACCOUNTING_REPLICATES[law], label, shrunk[label]) >= 0.90
        )


# ------------------------------------------------------------- (f) existence of the MLE

#: ``(law, n) -> (draws, failures)``: draws of the law on which a model of the NICE book raised
#: the GLM's ``DataError`` (a probability within ``1e-13`` of 0 or 1, or a singular information
#: matrix).  Only the models are fitted.  ``G4Q`` is the quadratic-dose law fitted main-terms.
MLE_PROBE: dict[tuple[str, int], tuple[int, int]] = {
    ("G1", 2_000): (4_000, 0),
    ("G2", 2_000): (4_000, 0),
    ("G3", 2_000): (4_000, 0),
    ("G4", 2_000): (4_000, 0),
    ("G4Q", 2_000): (4_000, 0),
    ("G1", 4_000): (2_000, 0),
    ("G2", 4_000): (2_000, 0),
    ("G3", 4_000): (2_000, 0),
    ("G4", 4_000): (2_000, 0),
    ("G1", 8_000): (2_000, 0),
    ("G2", 8_000): (2_000, 0),
    ("G3", 8_000): (2_000, 0),
    ("G4", 8_000): (2_000, 0),
    ("G2", 32_000): (400, 0),
}


def test_the_mle_exists_with_probability_at_least_0_999_at_every_rung() -> None:
    """The ladder starts at 2,000: the 95% upper bound of the failure rate there is under 0.001.

    The larger rungs have fewer draws, so they show a point estimate at or below 0.001.  A
    failure is a recordable error that the accounting counts against the 1% bound, never a
    redraw.
    """
    for (law, n), (draws, failures) in MLE_PROBE.items():
        assert failures / draws <= 0.001, (law, n)
        if n == study.LADDER[0]:
            upper = float(beta.ppf(0.95, failures + 1, draws - failures))
            assert upper <= 0.001, (law, upper)
    assert study.LADDER[0] == 2_000
    assert {n for _, n in MLE_PROBE if n in study.LADDER} == set(study.LADDER)
    assert {law for law, _ in MLE_PROBE} == {"G1", "G2", "G3", "G4", "G4Q"}


def test_the_probe_can_fail_on_a_separated_draw() -> None:
    """A deliberate separation raises the error the probe counts."""
    from cleverly.exceptions import DataError

    frame = properties.draw("G1", 400, 3)
    frame["L1"] = frame["A1"]  # the node-1 treatment is a function of the covariate
    frame["L2"] = frame["A2"]
    with pytest.raises(DataError), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit_book("G1", frame)


# ----------------------------------------------------------------- (g) sizes and the budget


def test_the_pilot_at_the_paired_size_shows_the_predicted_ice_bias() -> None:
    """The n = 32,000 pilot (100 replicates) sits near ``bias sqrt(n) / sd_ice`` of each pair."""
    observed = {  # standardized ICE bias, from the pilot
        "pair_g2_treat_all": 1.624,
        "pair_g2_treat_none": 0.592,
        "pair_g2_contrast": 0.454,
    }
    n = properties.PAIRED_N["G2"]
    for label, value in observed.items():
        _, ice_sd = ICE_PILOT_SD[label]
        predicted = abs(ICE_BIAS["G2"][ICE_ESTIMAND[label]]) * math.sqrt(n) / ice_sd
        assert value == pytest.approx(predicted, abs=0.12), label


def test_the_declared_replicates() -> None:
    assert dict.fromkeys(("G1", "G2", "G3", "G4"), 2_400) == properties.CALIBRATION_REPLICATES
    assert properties.ACCOUNTING_REPLICATES == {"G1": 2_400, "G2": 6_400, "G4": 2_400}
    assert properties.PAIRED_REPLICATES == {"G1": 1_200, "G2": 1_200}
    assert properties.LADDER_REPLICATES == 1_000
    assert properties.EXTRAPOLATION_REPLICATES == 1_000
    assert properties.SHARED_N == {"G1": 2_000, "G2": 8_000}
    assert set(properties.FIT_SETS) == {
        *(f"calibration_g{i}" for i in (1, 2, 3, 4)),
        *(f"accounting_g{i}" for i in (1, 2, 4)),
        *(f"ladder_g{i}_n_{n}" for i in (1, 2, 3, 4) for n in study.LADDER),
        "paired_g1",
        "paired_g2",
        "shared_g1",
        "shared_g2",
        "extrapolation_g4q",
    }


#: Wall-clock seconds per replicate on four workers (the best 50-replicate chunk of the pilot,
#: divided by 50), by fit set.  The primary phase fits the four laws at n = 8,000, one
#: replicate of every scenario at a time, at 3.9 seconds on four workers.
SECONDS_PER_REPLICATE: dict[str, float] = {
    "calibration_g1": 0.056,
    "calibration_g2": 0.062,
    "calibration_g3": 0.116,
    "calibration_g4": 0.034,
    "accounting_g1": 0.56,
    "accounting_g2": 1.23,
    "accounting_g4": 1.14,
    "ladder_g1_n_2000": 0.048,
    "ladder_g1_n_4000": 0.102,
    "ladder_g1_n_8000": 0.22,
    "ladder_g2_n_2000": 0.06,
    "ladder_g2_n_4000": 0.126,
    "ladder_g2_n_8000": 0.29,
    "ladder_g3_n_2000": 0.12,
    "ladder_g3_n_4000": 0.25,
    "ladder_g3_n_8000": 0.54,
    "ladder_g4_n_2000": 0.034,
    "ladder_g4_n_4000": 0.078,
    "ladder_g4_n_8000": 0.17,
    "paired_g1": 0.28,
    "paired_g2": 1.64,
    "shared_g1": 0.23,
    "shared_g2": 0.354,
    "extrapolation_g4q": 0.034,
}
PRIMARY_SECONDS_PER_REPLICATE = 3.9


def test_the_budget_of_the_declared_run() -> None:
    """About 6.0 hours on four vCPUs, 24 CPU-hours, if the best-case chunk times hold."""
    assert set(SECONDS_PER_REPLICATE) == set(properties.FIT_SETS)
    seconds = (
        sum(
            SECONDS_PER_REPLICATE[name] * replicates
            for name, (_, _, replicates) in properties.FIT_SETS.items()
        )
        + PRIMARY_SECONDS_PER_REPLICATE * study.PRIMARY_REPLICATES
    )
    assert seconds / 3600.0 == pytest.approx(6.0, abs=0.1)
    assert 4 * seconds / 3600.0 == pytest.approx(24.0, abs=0.4)
