"""The declared design of ``parametric-gformula`` (X45).

Every number here was fixed before any registered run.  The design pilot (about 200 to 1,200
replicates per label on an independent stream, see ``DESIGN_PILOT``) supplied the sampling
facts; the laws supply the rest.  Each test below proves one item of the design.

* **Specification.**  The main-terms models are the law.  A fit on the exact support of the
  law (a weighted population fit) recovers every coefficient of ``G1``, ``G3``, ``G4`` and
  ``G5``.  On ``G2``, whose Gaussian support is not finite, a large sample sits within the
  sampling error of the law.  A deliberate mutation of a coefficient is detected by the same
  check.
* **Truths.**  Binary and categorical paths are enumerated, each Gaussian variable is
  integrated by Gauss-Hermite, and ``G4`` has a closed form.  A forward simulation, sized so
  its standard error is at most 5% of the bias tolerance (a quarter of a sampling SD at the
  calibration size), agrees with every truth.
* **Existence of the MLE.**  The probability that every model of the NICE book exists, from
  the pinned probe, is at least 0.999 at the first rung of the shared ladder.  A second probe
  covers the whole fit set of each reported family, whose ``LTMLE`` and ``ICE`` fits can
  separate on a draw; its rate stays under the declared failure bound.
* **The D-terms.**  Each kind of derivative term (covariate, outcome, hazard, treatment and
  plug-in) carries a share of the standard error in some calibration label (contrasts
  included).  Dropping the term from that label's curve fails the calibration rule with chance
  at least 0.95, with the posterior uncertainty of the label's base SE ratio counted.
* **The ICE limit.**  The main-terms ICE fit converges to a value away from the truth on
  ``G2``, by an amount the paired fit set detects with power at least 0.95.
* **The Monte Carlo accounting.**  The positive ``n_draws_1`` cells are calibration cells.  A
  pinned catch table gives the chance that each fails without the simulation term.  The
  ``mc_removed`` cells report, because none has a predictive power of 0.95.
* **Sizing.**  The replicates of each calibration label give a pass probability of at least
  0.95 at the shrunk SE ratio under the SE-ratio rule.  With the empirical-Bayes posterior SD
  of the ratio (about 0.0085) integrated, the pass probability is 0.925 to 0.95, above the
  floor of 0.90 (``PREDICTIVE_PASS`` pins each label).
* **The red-cell reading.**  The expected count of finite-sample reds is computed at the
  shrunk SE ratios, with the gated derived controls counted, and declared.

Pilot scripts and outputs: ``cleverly-beta-plans/reviews/x45/design-pilot/`` (the first design
pilot), its ``review-probes/`` (the pre-queue review) and ``design-remediation/`` (the pilots of
the labels added after the review: ``pilot.py``, ``pilot_n1.py``, ``term_shares_labels.py``,
``mle_probe.py`` and ``mle_models.py``).

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
    assert record.comparator_state == "none"
    assert record.failure_bound == {study.IMPLEMENTATION: 0.01}
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
            "ey_regimen[rule_dose]",
            "ate_regimen[shift_half vs natural course]",
        ),
        "G5": (
            "ey_regimen[treat_all]",
            "ey_regimen[treat_none]",
            "ey_regimen[odds x2]",
            "ate_regimen[treat_all vs treat_none]",
        ),
    }


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # One replicate per fit set suffices: the cells a run publishes are fixed by the fit set,
        # and test_method_evidence.py checks the summary, which needs two, on the committed rows.
        rows = properties.generate_property_rows(n_jobs=1, budget=1)
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


def test_the_fixture_comparator_pins_its_image() -> None:
    folder = study.STUDY.artifacts
    for name in ("Dockerfile", "requirements.txt", "run_pygformula.py", "compare_pygformula.py"):
        assert (folder / name).is_file(), name
    pinned = (folder / "requirements.txt").read_text()
    assert f"pygformula=={study.PYGFORMULA}" in pinned
    assert re.search(
        r"^FROM python:3\.10-slim-bookworm@sha256:[0-9a-f]{64}$",
        (folder / "Dockerfile").read_text(),
        re.M,
    )
    assert f"ARG PYGFORMULA={study.PYGFORMULA}" in (folder / "Dockerfile").read_text()
    assert study.STUDY.comparator_state == "none"


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


@pytest.mark.parametrize(("law", "order"), [("G1", 12), ("G3", 12), ("G4", 3), ("G5", 12)])
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
        "G5": 8,
    }[law]
    # Covariate models of nodes 2 to T (a categorical covariate is one multinomial model), the
    # natural-treatment models, the outcome or the hazards.  Censoring is handled by the risk
    # set, so it has no model.
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
    "G4": ("natural course", "shift_half", "dose_half", "rule_dose"),
    "G4Q": ("dose_half",),
    "G5": ("treat_all", "treat_none", "odds x2"),
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
        if label == "rule_dose":
            intercept, slope = laws.G4_RULE_DOSE

            def rule_at(t: int) -> laws.Rule:
                return laws.Rule(lambda table: intercept + slope * table[f"L{t}"])

            return {t: rule_at(t) for t in nodes}
        return {t: laws.ShiftPlan(0.5 if label == "shift_half" else 0.0) for t in nodes}
    return study._node_plans(law, label)


def _truth_name(law: str, label: str, cause: str = "relapse", horizon: int = 4) -> str:
    if law == "G3":
        return f"cif_regimen[{label}, {cause} @ t={horizon}]"
    return f"ey_regimen[{label}]"


@pytest.mark.parametrize("law", ["G1", "G2", "G3", "G4", "G4Q", "G5"])
def test_every_law_names_only_earlier_columns(law: str) -> None:
    study.law_of(law).check()


def test_the_gauss_hermite_truths_have_converged() -> None:
    """Order 12 is within 1% of the bias tolerance of a higher order, on every integrated law."""
    for law, label in (
        ("G1", "rule"),
        ("G1", "odds x2"),
        ("G3", "treat_all_4"),
        ("G2", "policy_b"),
        ("G5", "treat_all"),
        ("G5", "odds x2"),
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


@pytest.mark.parametrize("law", ["G1", "G2", "G3", "G4", "G4Q", "G5"])
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
    "nice_g4_rule_dose": (1200, 1.0242, 0.0192, 2.491),
    "nice_g4_contrast": (1200, 1.0318, 0.0267, 1.168),
    "nice_g5_treat_all": (1200, 1.0170, 0.0245, 2.281),
    "nice_g5_treat_none": (1200, 1.0196, 0.0228, 3.175),
    "nice_g5_odds_x2": (1200, 1.0126, 0.0177, 2.509),
    "nice_g5_contrast": (1200, 1.0087, 0.0220, 3.873),
}


def test_the_noise_scale_is_the_pilot_spread() -> None:
    assert set(DESIGN_PILOT) == set(study.CALIBRATION_LABELS) == set(properties.NOISE_SD)
    for label, (_, _, _, spread) in DESIGN_PILOT.items():
        assert properties.NOISE_SD[label] == pytest.approx(spread, abs=1e-3), label


def _shrink(table: dict[str, tuple[int, float, float]]) -> dict[str, float]:
    """Each label's SE ratio shrunk toward the pooled mean, by the pilot's own spread."""
    return _shrink_with_sd(table)[0]


def _shrink_with_sd(
    table: dict[str, tuple[int, float, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Each label's shrunk SE ratio and its empirical-Bayes posterior SD.

    ``table`` maps a label to ``(replicates, SE ratio, relative SD of the SE)``.  The ratios of
    several labels spread more than their sampling errors say (``tau`` below), so each label is
    pulled toward the precision-weighted mean by the share of its variance that is sampling
    error.  This is the empirical-Bayes estimate of the true ratio of the label.  Its posterior
    variance is the weight times the sampling variance of the pilot ratio.
    """
    labels = list(table)
    ratio = np.array([table[x][1] for x in labels])
    count = np.array([table[x][0] for x in labels], dtype=float)
    noise = np.array([table[x][2] for x in labels])
    sampling = ratio**2 * (1.0 / (2.0 * (count - 1.0)) + noise**2 / count)
    pooled = float(np.average(ratio, weights=1.0 / sampling))
    tau2 = max(0.0, float(np.var(ratio, ddof=1)) - float(sampling.mean()))
    weight = tau2 / (tau2 + sampling)
    shrunk = pooled + weight * (ratio - pooled)
    sd = np.sqrt(weight * sampling)
    return dict(zip(labels, shrunk, strict=True)), dict(zip(labels, sd, strict=True))


def _shrunk_ratios() -> dict[str, float]:
    """The shrunk SE ratio of each calibration label (``n_draws`` 10)."""
    return _shrink({x: (v[0], v[1], v[2]) for x, v in DESIGN_PILOT.items()})


def _posterior_sds() -> dict[str, float]:
    """The empirical-Bayes posterior SD of the true SE ratio of each calibration label."""
    return _shrink_with_sd({x: (v[0], v[1], v[2]) for x, v in DESIGN_PILOT.items()})[1]


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


#: The floor of the brief for the pass probability of a calibration label.
PASS_FLOOR = 0.90


def _predictive_pass_probability(replicates: int, label: str) -> float:
    """The pass probability with the posterior uncertainty of the label's true ratio integrated."""
    shrunk, sd = _shrunk_ratios()[label], _posterior_sds()[label]
    return sum(
        float(w) * _pass_probability(replicates, label, shrunk + sd * float(x))
        for x, w in zip(_BASE_NODES, _BASE_WEIGHTS, strict=True)
    )


#: ``label -> posterior-predictive pass probability`` at the declared replicates, with the
#: empirical-Bayes posterior SD (about 0.0085) of the true ratio integrated.  The labels named
#: here sit at 0.925 to 0.94; every other label is higher.  The pass probability at the shrunk
#: point alone is at least 0.958.
PREDICTIVE_PASS: dict[str, float] = {
    "nice_g1_treat_all": 0.9391,
    "nice_g1_treat_none": 0.9369,
    "nice_g1_rule": 0.9498,
    "nice_g1_odds_x2": 0.9491,
    "nice_g1_contrast": 0.9252,
    "nice_g2_treat_all": 0.9402,
    "nice_g2_treat_none": 0.9417,
    "nice_g2_policy_b": 0.9474,
    "nice_g2_contrast": 0.9414,
    "nice_g3_relapse_h2": 0.941,
    "nice_g3_relapse_h4": 0.9491,
    "nice_g3_death_h4": 0.9462,
    "nice_g3_contrast_h4": 0.9446,
    "nice_g4_shift": 0.9437,
    "nice_g4_dose": 0.9351,
    "nice_g4_rule_dose": 0.9334,
    "nice_g4_contrast": 0.9261,
    "nice_g5_treat_all": 0.9389,
    "nice_g5_treat_none": 0.937,
    "nice_g5_odds_x2": 0.9419,
    "nice_g5_contrast": 0.9441,
}


def test_each_calibration_label_is_sized_for_its_pass_probability() -> None:
    """A label passes the SE-ratio rule with probability at least 0.95 at the shrunk ratio.

    The true ratio of each label is its empirical-Bayes estimate from the pilot, and the SE
    noise is the pilot's.  With the posterior SD of that ratio integrated, the pass probability
    is 0.925 to 0.94 at the weakest labels, above the floor of 0.90 of the brief.  The next
    smaller multiple of 400 clears 0.90 only by 0.005 at its worst label at the shrunk point and
    falls under 0.95 at every label, so the declared size is 2,400.
    """
    shrunk = _shrunk_ratios()
    worst = {}
    predictive = {}
    for label in study.CALIBRATION_LABELS:
        law = label.split("_")[1].upper()
        replicates = properties.CALIBRATION_REPLICATES[law]
        probability = _pass_probability(replicates, label, shrunk[label])
        assert probability >= 0.95, (label, replicates, probability)
        predictive[label] = _predictive_pass_probability(replicates, label)
        assert predictive[label] >= PASS_FLOOR, (label, replicates, predictive[label])
        worst[label] = _pass_probability(replicates - 400, label, shrunk[label])
    assert max(worst.values()) < 0.95
    assert set(properties.CALIBRATION_REPLICATES.values()) == {2_400}
    assert set(PREDICTIVE_PASS) == set(predictive)
    for label, pinned in PREDICTIVE_PASS.items():
        assert predictive[label] == pytest.approx(pinned, abs=0.003), label


def test_the_pilot_shows_no_calibration_cell_outside_the_band() -> None:
    for label, (_, ratio, _, _) in DESIGN_PILOT.items():
        low, high = study.STUDY.margins.calibration_se_ratio
        assert low + 0.02 < ratio < high - 0.02, label
    assert max(abs(r - 1.0) for r in _shrunk_ratios().values()) < 0.02


# ------------------------------------------------------------------------- (c) the D-terms

# A derivative term is evidenced by a calibration label when dropping it moves the label's SE
# ratio out of the calibration band with chance at least 0.95 at the label's replicates.  The
# share of a kind is one minus the SE of the stacked curve without the kind over the SE of the
# curve, so dropping the kind scales the SE ratio by one minus the share.

#: The chance that dropping a term kind must be caught with, at the label's replicates.
CATCH_FLOOR = 0.95
#: The nodes of the quadrature that integrates the uncertainty of a base SE ratio.
_BASE_NODES, _BASE_WEIGHTS = np.polynomial.hermite_e.hermegauss(21)
_BASE_WEIGHTS = _BASE_WEIGHTS / _BASE_WEIGHTS.sum()


def _se_leg_pass(replicates: int, ratio: float, noise: float) -> float:
    """The chance that the 99% interval of the SE ratio lies inside the calibration band."""
    low, high = study.STUDY.margins.calibration_se_ratio
    z = float(norm.ppf(0.5 + study.STUDY.margins.confidence_level / 2.0))
    spread = ratio * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
    inside = norm.cdf((high - z * spread - ratio) / spread) - norm.cdf(
        (low + z * spread - ratio) / spread
    )
    return float(max(inside, 0.0))


def _base_sd(label: str) -> float:
    """The posterior SD of the label's true SE ratio: the uncertainty of its base ratio.

    The catch is centred on the shrunk ratio, so it spreads by the posterior SD of that ratio,
    not by the raw sampling SD of the pilot.
    """
    return _posterior_sds()[label]


def _catch(replicates: int, factor: float, centre: float, spread: float, noise: float) -> float:
    """The chance that a curve scaled by ``factor`` fails the SE-ratio leg of the rule.

    The true ratio of the label is not known: it is ``centre`` with the sampling SD ``spread`` of
    its pilot.  The catch integrates over that uncertainty, so a label whose base ratio could sit
    on the other side of a band edge does not look better powered than it is.
    """
    passed = sum(
        float(w) * _se_leg_pass(replicates, factor * (centre + spread * float(x)), noise)
        for x, w in zip(_BASE_NODES, _BASE_WEIGHTS, strict=True)
    )
    return 1.0 - passed


def mutant_catch(label: str, replicates: int, factor: float) -> float:
    """The chance that a calibration label fails the rule when its curve is scaled by ``factor``."""
    return _catch(
        replicates, factor, _shrunk_ratios()[label], _base_sd(label), DESIGN_PILOT[label][2]
    )


def term_floor(label: str) -> float:
    """The smallest share whose removal the label's calibration rule catches at 0.95.

    Found by bisection on the share ``s`` of ``mutant_catch(label, R, 1 - s)``.
    """
    replicates = properties.CALIBRATION_REPLICATES[label.split("_")[1].upper()]
    low, high = 0.0, 0.5
    for _ in range(40):
        mid = 0.5 * (low + high)
        if mutant_catch(label, replicates, 1.0 - mid) >= CATCH_FLOOR:
            high = mid
        else:
            low = mid
    return high


#: The mean SE share of each kind of derivative term in the curve of each calibration label, from
#: 40 design draws of n = 2,000 per law.  A contrast label is the difference of two plan curves.
#: ``tests/unit/test_parametric_gformula_design.py::label_shares`` computes it.
TERM_SHARES: dict[str, dict[str, float]] = {
    "nice_g1_treat_all": {
        "covariate/binary": 0.015,
        "outcome": 0.727,
        "plug-in": 0.024,
        "treatment/binary": 0.0,
    },
    "nice_g1_treat_none": {
        "covariate/binary": 0.026,
        "outcome": 0.721,
        "plug-in": 0.013,
        "treatment/binary": 0.0,
    },
    "nice_g1_rule": {
        "covariate/binary": 0.003,
        "outcome": 0.739,
        "plug-in": 0.031,
        "treatment/binary": 0.0,
    },
    "nice_g1_odds_x2": {
        "covariate/binary": 0.015,
        "outcome": 0.66,
        "plug-in": 0.039,
        "treatment/binary": 0.005,
    },
    "nice_g1_contrast": {
        "covariate/binary": 0.02,
        "outcome": 0.798,
        "plug-in": 0.0,
        "treatment/binary": 0.0,
    },
    "nice_g2_treat_all": {
        "covariate/binary": 0.033,
        "covariate/normal": 0.262,
        "outcome": 0.032,
        "plug-in": 0.238,
    },
    "nice_g2_treat_none": {
        "covariate/binary": 0.034,
        "covariate/normal": 0.296,
        "outcome": 0.051,
        "plug-in": 0.18,
    },
    "nice_g2_policy_b": {
        "covariate/binary": 0.022,
        "covariate/normal": 0.183,
        "outcome": 0.016,
        "plug-in": 0.361,
    },
    "nice_g2_contrast": {
        "covariate/binary": 0.05,
        "covariate/normal": 0.491,
        "outcome": 0.085,
        "plug-in": 0.0,
    },
    "nice_g3_relapse_h2": {"covariate/binary": 0.002, "hazard": 0.831, "plug-in": 0.011},
    "nice_g3_relapse_h4": {"covariate/binary": 0.007, "hazard": 0.818, "plug-in": 0.011},
    "nice_g3_death_h4": {"covariate/binary": 0.001, "hazard": 0.937, "plug-in": 0.001},
    "nice_g3_contrast_h4": {"covariate/binary": 0.006, "hazard": 0.889, "plug-in": 0.0},
    "nice_g4_shift": {
        "covariate/normal": 0.056,
        "outcome": 0.058,
        "plug-in": 0.451,
        "treatment/normal": 0.041,
    },
    "nice_g4_dose": {
        "covariate/normal": 0.061,
        "outcome": 0.22,
        "plug-in": 0.287,
        "treatment/normal": 0.0,
    },
    "nice_g4_rule_dose": {
        "covariate/normal": 0.099,
        "outcome": 0.112,
        "plug-in": 0.369,
        "treatment/normal": 0.0,
    },
    "nice_g4_contrast": {
        "covariate/normal": 0.15,
        "outcome": 0.303,
        "plug-in": -0.0,
        "treatment/normal": 0.109,
    },
    "nice_g5_treat_all": {
        "covariate/binary": 0.163,
        "covariate/multi": 0.184,
        "outcome": 0.061,
        "plug-in": 0.133,
        "treatment/binary": 0.0,
    },
    "nice_g5_treat_none": {
        "covariate/binary": 0.238,
        "covariate/multi": 0.147,
        "outcome": 0.066,
        "plug-in": 0.095,
        "treatment/binary": 0.0,
    },
    "nice_g5_odds_x2": {
        "covariate/binary": 0.096,
        "covariate/multi": 0.09,
        "outcome": 0.024,
        "plug-in": 0.209,
        "treatment/binary": 0.119,
    },
    "nice_g5_contrast": {
        "covariate/binary": 0.256,
        "covariate/multi": 0.199,
        "outcome": 0.094,
        "plug-in": 0.009,
        "treatment/binary": 0.0,
    },
}

#: The kinds an exact-law witness covers alone: none, because every kind below reaches the floor.
WITNESS_ONLY: set[str] = set()


def _blocks_of(
    law: str, frame: pd.DataFrame, seed: int
) -> dict[tuple[Any, ...], dict[str, np.ndarray]]:
    """Each plan curve of a fit split into the plug-in and one block per model kind."""
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


def _label_blocks(law: str, frame: pd.DataFrame, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """The curve blocks of every calibration label of a law, a contrast as a difference."""
    plans = _blocks_of(law, frame, seed)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, estimand in study.KEYS[law].items():
        level = re.fullmatch(r"(?:ey|cif)_regimen\[([^,\]]+?)(?:, (\w+) @ t=(\d+))?\]", estimand)
        pair = re.fullmatch(r"ate_regimen\[(.+?) vs (.+?)(?:, (\w+) @ t=(\d+))?\]", estimand)
        if level:
            cause = level.group(2)
            horizon = int(level.group(3)) if level.group(3) else None
            if law != "G3":
                horizon = study.law_of(law).n_times
            out[study.label_of(law, key)] = plans[(level.group(1), cause, horizon)]
        else:
            assert pair is not None, estimand
            cause = pair.group(3)
            horizon = int(pair.group(4)) if pair.group(4) else study.law_of(law).n_times
            first = plans[(pair.group(1), cause, horizon)]
            second = plans[(pair.group(2), cause, horizon)]
            difference: dict[str, np.ndarray] = {}
            for kind in {*first, *second}:
                difference[kind] = np.asarray(first.get(kind, 0.0) - second.get(kind, 0.0))
            out[study.label_of(law, key)] = difference
    return out


def _shares(blocks: dict[str, np.ndarray]) -> dict[str, float]:
    total = sum(blocks.values())
    se = np.sqrt(np.mean(total**2))
    return {
        kind: 1.0 - float(np.sqrt(np.mean((total - block) ** 2))) / float(se)
        for kind, block in blocks.items()
    }


def label_shares(law: str, draws: int, first_seed: int) -> dict[str, dict[str, float]]:
    """The mean share of every kind in every calibration label of ``law`` over ``draws`` draws."""
    totals: dict[str, dict[str, list[float]]] = {}
    for r in range(draws):
        frame = properties.draw(law, study.CALIBRATION_N, first_seed + r)
        for label, blocks in _label_blocks(law, frame, seed=r).items():
            for kind, share in _shares(blocks).items():
                totals.setdefault(label, {}).setdefault(kind, []).append(share)
    return {
        label: {kind: float(np.mean(v)) for kind, v in kinds.items()}
        for label, kinds in totals.items()
    }


def _best_evidence() -> dict[str, tuple[str, float]]:
    """For each term kind, the label whose pinned share clears its floor by the most."""
    best: dict[str, tuple[str, float]] = {}
    margin: dict[str, float] = {}
    for label, kinds in TERM_SHARES.items():
        floor = term_floor(label)
        for kind, share in kinds.items():
            if share - floor > margin.get(kind, -1.0):
                margin[kind] = share - floor
                best[kind] = (label, share)
    return best


def test_every_derivative_term_kind_is_evidenced_by_a_label_that_would_catch_its_removal() -> None:
    best = _best_evidence()
    assert set(best) == {
        "plug-in",
        "covariate/binary",
        "covariate/multi",
        "covariate/normal",
        "treatment/binary",
        "treatment/normal",
        "outcome",
        "hazard",
    }
    assert not WITNESS_ONLY
    for kind, (label, share) in best.items():
        law = label.split("_")[1].upper()
        replicates = properties.CALIBRATION_REPLICATES[law]
        assert share >= term_floor(label), (kind, label, share, term_floor(label))
        assert mutant_catch(label, replicates, 1.0 - share) >= CATCH_FLOOR, (kind, label)


def test_the_term_floor_is_the_share_the_rule_can_see() -> None:
    """At 2,400 replicates a label with an ordinary pilot spread needs a share near 0.06 to 0.07."""
    floors = {label: term_floor(label) for label in TERM_SHARES}
    assert min(floors.values()) > 0.04 and max(floors.values()) < 0.09, floors
    # A share of 0.03 (the former floor) is caught with a chance well under 0.95.
    assert mutant_catch("nice_g2_treat_all", 2_400, 1.0 - 0.03) < 0.6


@pytest.mark.parametrize("law", ["G1", "G2", "G3", "G4", "G5"])
def test_a_recomputed_draw_shows_the_pinned_term_shares(law: str) -> None:
    """Four fresh draws give each kind's share in each label within 0.07 of the 40-draw table.

    The tolerance is wider than the margin of the weakest evidence over its floor (0.119 against
    0.076 for the treatment tilt), so it cannot guard the claim.  Each share that is the best
    evidence for its kind must also clear the computed floor of its label directly.
    """
    found = label_shares(law, 4, 90_000)
    pinned = {x: TERM_SHARES[x] for x in TERM_SHARES if x.split("_")[1].upper() == law}
    assert set(found) == set(pinned)
    for label, kinds in found.items():
        assert set(kinds) == set(pinned[label]), label
        for kind, share in kinds.items():
            assert share == pytest.approx(pinned[label][kind], abs=0.07), (label, kind)
    for kind, (label, _) in _best_evidence().items():
        if label in found:
            assert found[label][kind] >= term_floor(label), (kind, label, found[label][kind])


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


# ------------------------------------------------------------- (e) the Monte Carlo accounting

#: The ``n_draws`` 1 pilot at n = 2,000 (1,200 draws on the ``design_pilot_n1`` stream): per
#: positive ``n_draws_1`` label ``(replicates, SE ratio, relative SD of the SE, mean reported
#: Monte Carlo share of the variance)``.  The share is the term the accounting cell keeps in the
#: curve.  ``G1`` has no row: its share is 1% to 1.2%, so its cell measures nothing for the term.
N1_PILOT: dict[str, tuple[int, float, float, float]] = {
    "nice_g2_treat_all": (1200, 0.9729, 0.0181, 0.2013),
    "nice_g2_treat_none": (1200, 1.0329, 0.0196, 0.1649),
    "nice_g4_shift": (1200, 1.0270, 0.0174, 0.1268),
    "nice_g4_dose": (1200, 1.0072, 0.0238, 0.0861),
    "nice_g5_treat_all": (1200, 0.9797, 0.0241, 0.2388),
    "nice_g5_treat_none": (1200, 1.0153, 0.0225, 0.1851),
    "nice_g5_odds_x2": (1200, 0.9776, 0.0186, 0.3823),
}
#: The accounting pilot at n = 2,000 (400 draws): the SE ratio of the ``n_draws`` 1 fit and of the
#: ``mc_removed`` cell, which takes the SE of the independent ``m'`` fit.  The cell is reported.
ACCOUNTING_PILOT = {
    "nice_g2_treat_all": (0.9549, 0.8538),
    "nice_g2_treat_none": (1.0457, 0.9570),
    "nice_g4_shift": (0.9865, 0.9218),
    "nice_g4_dose": (0.9824, 0.9396),
}


def _n1_shrunk() -> dict[str, float]:
    """The shrunk SE ratio of each ``n_draws_1`` cell, pooled with the calibration labels.

    All cells are SE ratios of the same estimator on the same laws, so the pool shares one
    spread ``tau``; the ``n_draws_1`` cells keep their own pilot points.
    """
    table = {x: (v[0], v[1], v[2]) for x, v in DESIGN_PILOT.items()}
    table |= {f"n1:{x}": (v[0], v[1], v[2]) for x, v in N1_PILOT.items()}
    shrunk = _shrink(table)
    return {x: shrunk[f"n1:{x}"] for x in N1_PILOT}


def _n1_posterior_sds() -> dict[str, float]:
    """The empirical-Bayes posterior SD of the true ratio of each ``n_draws_1`` cell."""
    table = {x: (v[0], v[1], v[2]) for x, v in DESIGN_PILOT.items()}
    table |= {f"n1:{x}": (v[0], v[1], v[2]) for x, v in N1_PILOT.items()}
    sd = _shrink_with_sd(table)[1]
    return {x: sd[f"n1:{x}"] for x in N1_PILOT}


def _n1_pass_probability(replicates: int, label: str) -> float:
    margins = study.STUDY.margins
    return calibration_pass_probability(
        replicates,
        true_se_ratio=_n1_shrunk()[label],
        se_noise=N1_PILOT[label][2],
        confidence_level=margins.confidence_level,
        se_band=margins.calibration_se_ratio,
        coverage_band=margins.calibration_coverage,
    )


def mc_catch(label: str, replicates: int) -> float:
    """The chance that the positive ``n_draws_1`` cell fails when its curve lacks the simulation term.

    Removing a share ``s`` of the variance scales the SE ratio by ``sqrt(1 - s)``.  The base ratio
    carries the posterior SD of its shrunk ratio, which the catch integrates over (:func:`_catch`).
    """
    _, _, noise, share = N1_PILOT[label]
    return _catch(
        replicates, math.sqrt(1.0 - share), _n1_shrunk()[label], _n1_posterior_sds()[label], noise
    )


def predicted_ratio(label: str, base: float) -> float:
    """The SE ratio of the control: the fit's ratio times the root of what is left of the variance."""
    share = N1_PILOT[label][3]
    return base * math.sqrt(1.0 - share * (1.0 - 1.0 / study.M_PRIME))


def control_power(label: str, replicates: int) -> float:
    """The predictive chance that the whole 99% SE-ratio interval of ``mc_removed`` is below the band.

    The base ratio of the label is uncertain by the posterior SD of its shrunk ratio, so the power
    averages over it.  Treating the shrunk ratio as known would overstate the power.
    """
    noise = N1_PILOT[label][2]
    base_sd = _n1_posterior_sds()[label]
    z = float(norm.ppf(0.5 + study.STUDY.margins.confidence_level / 2.0))
    low_edge = study.STUDY.margins.calibration_se_ratio[0]
    total = 0.0
    for x, w in zip(_BASE_NODES, _BASE_WEIGHTS, strict=True):
        mean = predicted_ratio(label, _n1_shrunk()[label] + base_sd * float(x))
        spread = mean * math.sqrt(1.0 / (2.0 * (replicates - 1)) + noise**2 / replicates)
        total += float(w) * float(norm.cdf((low_edge - z * spread - mean) / spread))
    return total


#: The chance that each positive ``n_draws_1`` cell fails without the simulation term, at the
#: declared replicates, with the posterior uncertainty of its base ratio counted.  A label is evidence for the
#: term where the catch is at least :data:`CATCH_FLOOR`.
MC_CATCH: dict[str, float] = {
    "nice_g2_treat_all": 1.0,
    "nice_g2_treat_none": 0.9977,
    "nice_g4_shift": 0.9433,
    "nice_g4_dose": 0.6771,
    "nice_g5_treat_all": 1.0,
    "nice_g5_treat_none": 0.9999,
    "nice_g5_odds_x2": 1.0,
}
#: The labels whose catch reaches the floor.  The two ``G4`` cells are positive calibration cells at
#: ``n_draws`` 1 and are not evidence for the term: the simulation share is 13% and 9%.
MC_EVIDENCE = {
    "nice_g2_treat_all",
    "nice_g2_treat_none",
    "nice_g5_treat_all",
    "nice_g5_treat_none",
    "nice_g5_odds_x2",
}


def test_the_pilot_ratios_follow_the_predicted_ratio_of_ratios() -> None:
    """``mc_removed / n_draws_1`` is ``sqrt(1 - share (1 - 1/m'))``, to within the pilot error."""
    for label, (base, removed) in ACCOUNTING_PILOT.items():
        assert removed / base == pytest.approx(predicted_ratio(label, 1.0), abs=0.012), label


def test_the_mc_removed_cells_report_because_none_is_powered() -> None:
    """With the base ratio uncertain, no ``mc_removed`` control reaches a predictive power of 0.95.

    The control is a property of the law, so its red would be a finite-sample red.  Every cell
    reports and none gates.
    """
    for label in N1_PILOT:
        law = label.split("_")[1].upper()
        if law not in study.MC_REMOVED_LAWS:
            continue
        assert control_power(label, properties.ACCOUNTING_REPLICATES[law]) < CATCH_FLOOR, label
    assert {label.split("_")[1].upper() for label in N1_PILOT} - set(study.MC_REMOVED_LAWS) == {
        "G5"
    }
    assert not hasattr(properties, "GATED_DIRECTIONS")


def test_the_mutation_catch_table_of_the_n_draws_1_cells() -> None:
    """The positive cells that catch a curve without the Monte Carlo term, and the ones that do not."""
    found = {
        label: mc_catch(label, properties.ACCOUNTING_REPLICATES[label.split("_")[1].upper()])
        for label in N1_PILOT
    }
    for label, pinned in MC_CATCH.items():
        assert found[label] == pytest.approx(pinned, abs=0.01), label
    assert set(found) == set(MC_CATCH)
    assert {x for x, v in found.items() if v >= CATCH_FLOOR} == MC_EVIDENCE
    assert tuple(N1_PILOT) == study.MC_LABELS
    # A G1 accounting cell would remove 1.1% to 1.2% of the variance, which moves the ratio by 0.6%
    # and is caught with a chance of about 0.03, so the fit set is dropped.
    assert math.sqrt(1.0 - 0.0123) > 0.99


def test_the_one_draw_positive_cell_is_sized_like_a_calibration_cell() -> None:
    """Every positive ``n_draws_1`` cell passes with chance at least 0.95 at the declared size.

    The smaller size of 2,000 passes at 0.90 to 0.93, which is the bare floor, so the size is the
    next multiple of 400 that clears 0.95 with room for the spread of the pilot.
    """
    for label in N1_PILOT:
        law = label.split("_")[1].upper()
        assert _n1_pass_probability(properties.ACCOUNTING_REPLICATES[law], label) >= 0.95, label
    assert set(properties.ACCOUNTING_REPLICATES.values()) == {2_400}


# ------------------------------------------------------------- (f) existence of the MLE

#: ``(law, n) -> (draws, failures)``: draws of the law on which a model of the NICE book raised
#: the GLM's ``DataError`` (a probability within ``1e-13`` of 0 or 1, or a singular information
#: matrix).  Only the NICE models are fitted.  ``G4Q`` is the quadratic-dose law fitted main-terms.
MLE_PROBE: dict[tuple[str, int], tuple[int, int]] = {
    ("G1", 2_000): (4_000, 0),
    ("G2", 2_000): (4_000, 0),
    ("G3", 2_000): (4_000, 0),
    ("G4", 2_000): (4_000, 0),
    ("G4Q", 2_000): (4_000, 0),
    ("G5", 2_000): (4_000, 0),
    ("G1", 4_000): (2_000, 0),
    ("G2", 4_000): (2_000, 0),
    ("G3", 4_000): (2_000, 0),
    ("G4", 4_000): (2_000, 0),
    ("G1", 8_000): (2_000, 0),
    ("G2", 8_000): (2_000, 0),
    ("G3", 8_000): (2_000, 0),
    ("G4", 8_000): (2_000, 0),
    ("G5", 8_000): (1_000, 0),
    ("G2", 32_000): (400, 0),
}
#: ``fit set -> (draws, failures)``: draws on which any fit of the whole fit set raised a
#: recordable error (``DataError``).  These are the fit sets that fit a method beside NICE: the
#: ``LTMLE`` and ``IPSI`` fits of the shared-target sets and the ICE fit of the paired sets.  The
#: ``G1`` shared-target failures are the ``LTMLE`` fit, whose ``treat_none`` follower regression
#: separates on about 0.7% of draws (``arm_failures.py``).
FIT_SET_PROBE: dict[str, tuple[int, int]] = {
    "shared_g1": (3_000, 20),
    "paired_g1": (3_000, 0),
    "shared_g2": (800, 0),
    "paired_g2": (200, 0),
}


def test_the_mle_exists_with_probability_at_least_0_999_at_every_rung() -> None:
    """The ladder starts at 2,000: the 95% upper bound of the failure rate there is under 0.001.

    The larger rungs have fewer draws, so they show a point estimate at or below 0.001.  A
    failure is a recordable error that the accounting counts against the 1% bound, never a
    redraw.  The probe fits the models of the NICE book only.
    """
    for (law, n), (draws, failures) in MLE_PROBE.items():
        assert failures / draws <= 0.001, (law, n)
        if n == study.LADDER[0]:
            upper = float(beta.ppf(0.95, failures + 1, draws - failures))
            assert upper <= 0.001, (law, upper)
    assert study.LADDER[0] == 2_000
    assert {n for _, n in MLE_PROBE if n in study.LADDER} == set(study.LADDER)
    assert {law for law, _ in MLE_PROBE} == {"G1", "G2", "G3", "G4", "G4Q", "G5"}


def test_the_baseline_fits_of_the_reported_sets_fail_only_where_nothing_is_gated() -> None:
    """The ``LTMLE`` and ICE fits can separate; the sets they sit in report or stay under the bound.

    ``shared_g1`` fails on 0.67% of draws, from the ``LTMLE`` fit.  Every cell of that fit set is
    a diagnostic row, which the failure accounting does not score, so no gated cell loses a
    replicate to it.  A fit set with a gated cell (``paired_g2``) shows no failure.
    """
    assert set(FIT_SET_PROBE) == {
        x for x in properties.FIT_SETS if x.split("_")[0] in ("shared", "paired")
    }
    gated_sets = {f"paired_{label.split('_')[1]}" for label in study.ICE_GATED}
    for name, (draws, failures) in FIT_SET_PROBE.items():
        rate = failures / draws
        if name in gated_sets:
            assert failures == 0, name
        else:
            assert rate <= study.FAILURE_BOUND, name
        cells = properties.FIT_SET_CELLS[name]
        scored = [c for c in cells if c.split("/")[0] in ("ice_necessity",)]
        assert bool(scored) == (name in gated_sets), name
    assert FIT_SET_PROBE["shared_g1"][1] / FIT_SET_PROBE["shared_g1"][0] == pytest.approx(
        0.0067, abs=1e-4
    )


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
    assert dict.fromkeys(("G1", "G2", "G3", "G4", "G5"), 2_400) == properties.CALIBRATION_REPLICATES
    assert properties.ACCOUNTING_REPLICATES == {"G2": 2_400, "G4": 2_400, "G5": 2_400}
    assert properties.PAIRED_REPLICATES == {"G1": 1_200, "G2": 1_200}
    assert properties.LADDER_REPLICATES == 1_000
    assert properties.EXTRAPOLATION_REPLICATES == 1_000
    assert properties.SHARED_N == {"G1": 2_000, "G2": 8_000}
    assert set(properties.FIT_SETS) == {
        *(f"calibration_g{i}" for i in (1, 2, 3, 4, 5)),
        *(f"accounting_g{i}" for i in (2, 4, 5)),
        *(f"ladder_g{i}_n_{n}" for i in (1, 2, 3, 4) for n in study.LADDER),
        "paired_g1",
        "paired_g2",
        "shared_g1",
        "shared_g2",
        "extrapolation_g4q",
    }


#: Wall-clock seconds per replicate on four workers (the best 50-replicate chunk of the pilot,
#: divided by 50), by fit set.  ``accounting_g5`` fits the one-history curve only.  The primary
#: phase fits the five laws at n = 8,000, one replicate of every scenario at a time, at 1.71
#: seconds on four workers (60 replicates of ``draw_and_fit``).
SECONDS_PER_REPLICATE: dict[str, float] = {
    "calibration_g1": 0.056,
    "calibration_g2": 0.062,
    "calibration_g3": 0.116,
    "calibration_g4": 0.044,
    "calibration_g5": 0.076,
    "accounting_g2": 1.23,
    "accounting_g4": 1.316,
    "accounting_g5": 0.07,
    "ladder_g1_n_2000": 0.048,
    "ladder_g1_n_4000": 0.102,
    "ladder_g1_n_8000": 0.22,
    "ladder_g2_n_2000": 0.06,
    "ladder_g2_n_4000": 0.126,
    "ladder_g2_n_8000": 0.29,
    "ladder_g3_n_2000": 0.12,
    "ladder_g3_n_4000": 0.25,
    "ladder_g3_n_8000": 0.54,
    "ladder_g4_n_2000": 0.042,
    "ladder_g4_n_4000": 0.088,
    "ladder_g4_n_8000": 0.198,
    "paired_g1": 0.28,
    "paired_g2": 1.64,
    "shared_g1": 0.23,
    "shared_g2": 0.354,
    "extrapolation_g4q": 0.034,
}
PRIMARY_SECONDS_PER_REPLICATE = 1.71
#: The budget of the declared run after the review: wall-clock hours on four workers.
BUDGET_HOURS = 3.9
#: The budget of the declared run before this design review: 6.0 hours on four workers.
BUDGET_BEFORE_REVIEW_HOURS = 6.0


def _budget_hours() -> float:
    seconds = (
        sum(
            SECONDS_PER_REPLICATE[name] * replicates
            for name, (_, _, replicates) in properties.FIT_SETS.items()
        )
        + PRIMARY_SECONDS_PER_REPLICATE * study.PRIMARY_REPLICATES
    )
    return seconds / 3600.0


def test_the_budget_of_the_declared_run() -> None:
    """BUDGET_HOURS on four vCPUs, four times that in CPU-hours, if the best-case chunk times hold."""
    assert set(SECONDS_PER_REPLICATE) == set(properties.FIT_SETS)
    assert _budget_hours() == pytest.approx(BUDGET_HOURS, abs=0.1)
    assert _budget_hours() < BUDGET_BEFORE_REVIEW_HOURS


# ------------------------------------------------------------ (h) the declared red-cell reading


def expected_reds() -> tuple[float, float, int]:
    """The expected count of finite-sample reds under a perfectly calibrated estimator.

    Returns ``(expected count, chance of at least one, cells counted)``.  Each positive calibration
    label and each positive ``n_draws_1`` cell passes with the chance of its replicates at its
    empirical-Bayes shrunk SE ratio and its own SE noise.  Each ``ice_necessity`` label passes
    with its two powers.  Each derived control (a shrunken SE and added noise, on every
    calibration label) passes when its curve is caught, with the base ratio uncertain.  The
    ladder rungs and rate cells (20) and the primary rows (21) each pass at 0.99 or better by
    their tolerance rules, so they count at 0.99 here, which overstates their share of reds.
    """
    margins = study.STUDY.margins
    passes: list[float] = []
    shrunk = _shrunk_ratios()
    for label, (_, _, noise, _) in DESIGN_PILOT.items():
        replicates = properties.CALIBRATION_REPLICATES[label.split("_")[1].upper()]
        passes.append(
            calibration_pass_probability(
                replicates,
                true_se_ratio=shrunk[label],
                se_noise=noise,
                confidence_level=margins.confidence_level,
                se_band=margins.calibration_se_ratio,
                coverage_band=margins.calibration_coverage,
            )
        )
    n1_shrunk = _n1_shrunk()
    for label, (_, _, noise, _) in N1_PILOT.items():
        replicates = properties.ACCOUNTING_REPLICATES[label.split("_")[1].upper()]
        passes.append(
            calibration_pass_probability(
                replicates,
                true_se_ratio=n1_shrunk[label],
                se_noise=noise,
                confidence_level=margins.confidence_level,
                se_band=margins.calibration_se_ratio,
                coverage_band=margins.calibration_coverage,
            )
        )
    for label in study.ICE_GATED:
        discriminated, displaced = ice_power(
            label, properties.PAIRED_N["G2"], properties.PAIRED_REPLICATES["G2"]
        )
        passes.append(discriminated * displaced)
    passes += [0.99] * (20 + sum(len(x) for x in study.ESTIMANDS.values()))
    for label in DESIGN_PILOT:
        replicates = properties.CALIBRATION_REPLICATES[label.split("_")[1].upper()]
        for factor in (properties.SHRUNKEN_SE_FACTOR, 1.0 / math.sqrt(2.0)):
            passes.append(mutant_catch(label, replicates, factor))
    expected = sum(1.0 - x for x in passes)
    return expected, 1.0 - float(np.prod(passes)), len(passes)


def test_the_declared_count_of_expected_reds() -> None:
    """The study module declares the expected reds under perfect calibration before any run."""
    expected, at_least_one, cells = expected_reds()
    assert (expected, at_least_one) == pytest.approx(
        (study.EXPECTED_RED_COUNT, study.PROBABILITY_OF_A_RED), abs=0.02
    )
    assert cells == study.GATED_CELLS
    assert f"{study.EXPECTED_RED_COUNT:.2f}" in str(study.__doc__)
    assert f"{study.PROBABILITY_OF_A_RED:.2f}" in str(study.__doc__)
    # The reading is taken at the shrunk ratios and counts the 42 gated derived controls.
    assert (study.EXPECTED_RED_COUNT, study.PROBABILITY_OF_A_RED) == (1.26, 0.72)
    assert cells == 72 + 2 * len(DESIGN_PILOT)


def test_the_two_derived_controls_are_caught_with_certainty() -> None:
    """A shrunken SE (ratio 0.7) and added noise (ratio 0.71) leave the band for every label."""
    for label, (_, _, noise, _) in DESIGN_PILOT.items():
        replicates = properties.CALIBRATION_REPLICATES[label.split("_")[1].upper()]
        for ratio in (properties.SHRUNKEN_SE_FACTOR, 1.0 / math.sqrt(2.0)):
            assert 1.0 - _se_leg_pass(replicates, ratio, noise) > 0.999, label
