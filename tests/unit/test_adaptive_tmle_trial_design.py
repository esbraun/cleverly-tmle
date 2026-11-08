"""The declared constants of the ``adaptive-tmle-trial`` study, recomputed without a run.

The law checks call the exact-law quadrature of ``tests.studies._atmle_laws`` and, for the
standard deviation of an influence function, one large draw of the law with the true nuisances
(never an estimator run). The control checks compute each control's power from those exact
quantities before any declared run. The pilot readings and the sizing rule that read them are
pinned. The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from tests.studies import _atmle_laws as laws
from tests.studies import adaptive_tmle_selective as selective
from tests.studies import adaptive_tmle_trial as study
from tests.studies import adaptive_tmle_trial_properties as properties
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed
from tests.unit.test_study_provenance import _required_from

pytestmark = pytest.mark.xdist_group("adaptive_tmle_trial_design")

N = study.PRIMARY_N
R = study.PRIMARY_REPLICATES
MARGINS = study.STUDY.margins
POWER = 0.90


def _coverage(bias_over_sd: float, se_ratio: float = 1.0) -> float:
    """The coverage of a 95% Wald interval with bias ``b`` SDs and reported SE ``r`` SDs."""
    z = norm.ppf(0.975) * se_ratio
    return float(norm.cdf(z - bias_over_sd) - norm.cdf(-z - bias_over_sd))


def _fails_floor_probability(coverage: float, replicates: int) -> float:
    """P(the exact interval of the observed coverage lies wholly below the floor)."""
    floor = MARGINS.coverage_floor
    level = MARGINS.confidence_level
    counts = [
        k
        for k in range(replicates + 1)
        if clopper_pearson(k, replicates, confidence_level=level).high < floor
    ]
    return float(binom.cdf(max(counts), replicates, coverage)) if counts else 0.0


def _passes_floor_probability(coverage: float, replicates: int) -> float:
    floor = MARGINS.coverage_floor
    level = MARGINS.confidence_level
    counts = [
        k
        for k in range(replicates + 1)
        if clopper_pearson(k, replicates, confidence_level=level).low >= floor
    ]
    return float(binom.sf(min(counts) - 1, replicates, coverage)) if counts else 0.0


def _model(law: laws.Law, basis: object, n: int = 6_000) -> SimpleNamespace:
    """A stored selection that keeps every column of ``basis`` fitted on one draw (no estimator)."""
    from cleverly.estimators._working_basis import WorkingDictionary

    frame = laws.draw(law, n, 3)
    w = frame[law.columns()].to_numpy()
    a, s = frame["A"].to_numpy(float), frame["S"].to_numpy(float)
    dictionary = WorkingDictionary.fit(
        basis, w, a, s, np.arange(n), controls_only=law.controls_only
    )
    return SimpleNamespace(
        dictionary=dictionary,
        effect=SimpleNamespace(selected=np.arange(dictionary.phi_keep.size)),
        bias=SimpleNamespace(selected=np.arange(dictionary.psi_keep.size)),
    )


# ----------------------------------------------------------------------------- the truths


@pytest.mark.parametrize(
    ("name", "psi", "psi_2"),
    [
        ("a0", 1.5, 1.5),
        ("a1a", 1.5, 1.5),
        ("a1b", 1.5, 1.5),
        ("a2c", 4.2, 4.2),
        ("a2d", 4.2, 4.2),
        ("a3", 1.5, 1.5),
        ("a4", 1.5, 1.5),
        ("a5", 1.5, 1.5),
        ("e1", 1.5, 1.5),
    ],
)
def test_the_published_effects_come_out_of_the_quadrature(
    name: str, psi: float, psi_2: float
) -> None:
    """Plan 5: Psi = Psi_2 = 1.5 on (a), (b) and the positivity law and 4.2 on (c), (d)."""
    targets = study.exact(name)
    assert targets["psi"] == pytest.approx(psi, abs=1e-6)
    assert targets["psi_2"] == pytest.approx(psi_2, abs=1e-6)


def test_the_binomial_law_risks_and_effect() -> None:
    """B1y. The plan says the external risk "stays in [0.1, 0.9]". That holds under control
    (0.8 Q + 0.1) and not under treatment, where the external risk equals the trial's
    expit(.) of an unbounded normal index: the declared law keeps the plan's formula and the
    test pins the share of covariate mass outside [0.05, 0.95]. The risk difference is read by
    quadrature."""
    law = laws.LAWS["b1y"]
    w, wt = laws.grid(law)
    control = law.q0(w, 0.0)
    assert control.min() >= 0.1 and control.max() <= 0.9
    treated = law.q0(w, 1.0)
    outside = float(np.sum(wt * ((treated < 0.05) | (treated > 0.95))))
    assert outside == pytest.approx(0.0, abs=0.02)
    assert study.exact("b1y")["psi"] == pytest.approx(0.183022, abs=1e-6)


@pytest.mark.parametrize("name", ["a1b", "e1", "h2", "b1y"])
def test_the_quadrature_is_converged_to_a_hundredth_of_a_standard_error(name: str) -> None:
    """Doubling the nodes per panel moves psi and psi_2 by under 0.01 SE (n3)."""
    law = laws.LAWS[name]
    coarse = laws.exact_targets(law, laws.PANEL_NODES)
    fine = laws.exact_targets(law, 2 * laws.PANEL_NODES)
    standard_error = 4.0 / np.sqrt(N)  # a generous bound on sd(D) / sqrt(n) of every cell
    assert abs(coarse["psi"] - fine["psi"]) < 0.01 * standard_error
    assert abs(coarse["psi_2"] - fine["psi_2"]) < 0.01 * standard_error


def test_the_projection_quadrature_is_converged_on_a_fitted_dictionary() -> None:
    """The stored fold transformer is applied to the nodes, never refitted there, and the
    projection moves by under 0.01 SE when the nodes per panel double."""
    from cleverly.estimators._working_basis import WorkingBasis

    law = laws.LAWS["h2"]
    model = _model(law, WorkingBasis())
    coarse = laws.projection(law, model, "pooled", laws.PANEL_NODES).value
    fine = laws.projection(law, model, "pooled", 2 * laws.PANEL_NODES).value
    assert abs(coarse - fine) < 0.01 * 0.07
    trial = laws.projection(law, model, "trial", laws.PANEL_NODES).value
    assert abs(trial - laws.projection(law, model, "trial", 2 * laws.PANEL_NODES).value) < 0.0007


def test_a_saturated_dictionary_projects_onto_the_effect() -> None:
    """On a law whose effect and bias are linear in W, the default dictionary's projection is
    the truth (Lemma 3): the quadrature solves the same Gram systems the estimator does."""
    from cleverly.estimators._working_basis import WorkingBasis

    law = laws.LAWS["a1b"]
    model = _model(law, WorkingBasis())
    assert laws.projection(law, model, "pooled").value == pytest.approx(1.5, abs=2e-3)


def test_the_unpenalized_projection_is_not_the_truth_on_the_separating_law() -> None:
    """A nonzero witness for the projection code: a main-terms dictionary on E1 misses by far."""
    law = laws.LAWS["e1"]
    value = laws.projection(law, _model(law, laws.MainTerms()), "pooled").value
    assert abs(value - 1.5) > 0.3


# ----------------------------------------------------------------------------- the laws


def test_the_separating_law_separates_by_three_standard_errors() -> None:
    """E1 (B1): inf pi(W) >= 0.02, and the K2 main-terms projection is >= 3 SE from Psi at the
    study's n, with sd(D_w) at P_0 from the exact-law influence function."""
    law = laws.LAWS["e1"]
    w, _ = laws.grid(law)
    assert law.s(w).min() >= 0.02
    model = _model(law, laws.MainTerms())
    truth, _, table = laws.oracle_parts(law, model, "pooled", n=400_000, seed=11)
    standard_error = float(table["total"].std()) / np.sqrt(N)
    assert (truth.value - 1.5) / standard_error >= 3.0
    assert truth.value - 1.5 == pytest.approx(0.3208, abs=1e-3)
    assert standard_error == pytest.approx(0.0768, abs=2e-3)


def test_the_heterogeneous_law_separates_psi_and_psi_2_by_three_standard_errors() -> None:
    """H2 (B2): gamma is set so that |Psi - Psi_2| >= 3 SE of the pooled interval at n."""
    targets = study.exact("h2")
    gap = targets["psi_2"] - targets["psi"]
    assert gap == pytest.approx(0.232532, abs=1e-5)
    law = laws.LAWS["h2"]
    from cleverly.estimators._working_basis import WorkingBasis

    _, _, table = laws.oracle_parts(law, _model(law, WorkingBasis()), "pooled", n=400_000, seed=12)
    assert gap / (float(table["total"].std()) / np.sqrt(N)) >= 3.0


def test_every_cell_names_a_law_and_a_gated_target() -> None:
    assert set(study.CELLS) == set(study.SCENARIOS)
    for name, cell in study.CELLS.items():
        assert cell.law in laws.LAWS, name
        assert cell.estimand in {"psi", "psi_2", "projection"}
        assert cell.form in {"cv", "pooled"}
    # The projection gates only where the plan says Psi and the projection coincide: a2c, a2d,
    # E1 and the K2 ladder.
    assert {n for n, c in study.CELLS.items() if c.estimand == "projection"} == {
        "a2c",
        "a2d",
        "e1",
        "e1_k2_n1000",
        "e1_k2_n2000",
    }


# ----------------------------------------------------------------------------- the controls


def test_k1_naive_pooled_tmle_fails_psi_coverage_with_power() -> None:
    """K1: the naive pooled TMLE of (b) misses Psi by Psi# = 0.056, 0.9 of its SDs; its Psi
    coverage falls below the floor with probability >= 0.90 at the study's replications."""
    oracle = laws.pooled_ate_oracle(laws.LAWS["a1b"], n=300_000, seed=21)
    sd = oracle["sd"] / np.sqrt(N)
    bias = oracle["naive"] - 1.5
    assert bias == pytest.approx(0.0563, abs=1e-3)
    assert _fails_floor_probability(_coverage(bias / sd), R) >= POWER


def test_k2_main_terms_model_passes_its_projection_and_fails_psi() -> None:
    """K2: on E1 the unpenalized main-terms working model covers its own projection (power of
    passing >= 0.90) and misses Psi (power of failing >= 0.90)."""
    law = laws.LAWS["e1"]
    truth, _, table = laws.oracle_parts(
        law, _model(law, laws.MainTerms()), "pooled", n=300_000, seed=22
    )
    sd = float(table["total"].std()) / np.sqrt(N)
    assert _fails_floor_probability(_coverage((truth.value - 1.5) / sd), R) >= POWER
    assert _passes_floor_probability(_coverage(0.0), R) >= POWER


def test_k3_the_pi_step_removes_a_bias_the_untargeted_fit_keeps() -> None:
    """K3 (plan fallback): with Pi a main-terms logistic and g, Qt correct, the untargeted
    plug-in converges to Psi + E[C (Pi_lim - Pi)] = Psi - 0.023, 1.7 of its SDs, and fails the
    floor with probability >= 0.90. The SE-only mutation on the correct-Pi law has no power:
    a correct logistic Pi already solves the Pi score, so epsilon is 0 to three decimals."""
    bias = laws.pi_limit_bias(laws.LAWS["a3"])
    assert bias == pytest.approx(-0.02309, abs=2e-4)
    oracle_sd = 0.0137  # sd(D) / sqrt(n) at P_0, from oracle_parts on a3 (T3 pilot)
    assert _fails_floor_probability(_coverage(bias / oracle_sd), R) >= POWER


def test_k5_the_pooled_population_misses_psi_2_with_power() -> None:
    """K5: the pooled-population fit (S/p replaced by 1) reads Psi, 0.2325 below Psi_2."""
    gap = study.exact("h2")["psi_2"] - study.exact("h2")["psi"]
    law = laws.LAWS["h2"]
    from cleverly.estimators._working_basis import WorkingBasis

    _, _, table = laws.oracle_parts(law, _model(law, WorkingBasis()), "pooled", n=300_000, seed=23)
    sd = float(table["total"].std()) / np.sqrt(N)
    assert _fails_floor_probability(_coverage(gap / sd), R) >= POWER


# ----------------------------------------------------------------------------- the hooks


def _controls_frame() -> pd.DataFrame:
    """Synthetic primary rows that make every verdict read the way a correct run reads: the
    mutated controls sit 0.5 off the truth they are read against, the positives sit on it."""
    rng = np.random.default_rng(0)
    rows = []
    for scenario, cell in study.CELLS.items():
        for k in range(300):
            row = dict.fromkeys(study.EXTRA_COLUMNS, np.nan)
            row.update(
                scenario=scenario,
                estimate=1.5 + rng.normal(0, 0.07),
                std_error=0.07,
                truth=1.5,
                psi_truth=1.5,
                psi_2_truth=1.5,
                projection_truth=1.5,
                covered_psi=1,
                covered_psi_2=1,
                covered_projection=1,
                converged=1,
                replicate=k,
            )
            if "comparator" in cell.controls:
                row["comparator_estimate"] = 1.5 + rng.normal(0, 0.12)
            if cell.controls:
                own = scenario == "e1"  # K2 reads its own projection (1.82), not Psi
                truth = 1.82 if own else 1.5
                miss = 0.0 if own else 0.5
                estimate = truth + miss + rng.normal(0, 0.07)
                row.update(
                    control_estimate=estimate,
                    control_std_error=0.07,
                    control_truth=truth,
                    control_ci_lower=estimate - 0.137,
                    control_ci_upper=estimate + 0.137,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_the_control_verdicts_read_the_rows_and_a_broken_control_is_found() -> None:
    """Every verdict passes on rows that read as a correct run, and a deliberate mutation of
    the K1 rows (the control moved onto Psi) is reported by the gate hook."""
    frame = _controls_frame()
    table = study.controls_table(frame)
    assert table.set_index("control")["passed"].all(), table
    assert {
        "efficiency",
        "k1_naive_pooled",
        "k2_main_terms_projection",
        "k2_main_terms_psi",
        "k3_targeted",
        "k3_no_pi_step",
        "k5_pooled_for_psi_2",
        "k4_ladder",
    } <= set(table["control"])
    hook = {study.CONTROLS_ARTIFACT: table, study.REPORTED_ARTIFACT: pd.DataFrame()}
    assert next(iter(study.scientific_failures(hook).values())).empty
    broken = frame.copy()
    shift = broken["scenario"] == "a1b"
    broken.loc[shift, "control_estimate"] = broken.loc[shift, "psi_truth"]
    broken.loc[shift, "control_ci_lower"] = broken.loc[shift, "psi_truth"] - 0.137
    broken.loc[shift, "control_ci_upper"] = broken.loc[shift, "psi_truth"] + 0.137
    hook[study.CONTROLS_ARTIFACT] = study.controls_table(broken)
    assert set(next(iter(study.scientific_failures(hook).values()))["control"]) == {
        "k1_naive_pooled"
    }


def test_the_efficiency_gate_reads_the_upper_bound_of_the_ratio() -> None:
    """The efficiency contrast passes only when the bootstrap upper bound is below one."""
    rng = np.random.default_rng(1)
    base = pd.DataFrame(
        {
            "scenario": "a0",
            "psi_truth": 1.5,
            "estimate": 1.5 + rng.normal(0, 0.05, 600),
            "comparator_estimate": 1.5 + rng.normal(0, 0.07, 600),
        }
    )
    better = study.efficiency_row(base, study.STUDY, seed=1)
    assert better["passed"] and better["upper"] < 1.0
    worse = base.assign(estimate=1.5 + rng.normal(0, 0.09, 600))
    assert not study.efficiency_row(worse, study.STUDY, seed=1)["passed"]
    reported = study.efficiency_row(worse, study.STUDY, seed=1, role="reporting")
    assert reported["passed"] and reported["upper"] > 1.0
    assert study.EFFICIENCY_CELLS == ("a0",) and study.EFFICIENCY_REPORTED == ("a1a", "a1b")


def test_the_a0_efficiency_gain_is_the_theory_and_a_forced_arm_column_removes_it() -> None:
    """At ``P_0`` on ``a0`` the bias model's oracle selection is its constant. There the SD of
    ``D_w`` is 0.48 of the trial-only TMLE's efficient SD, ``sqrt((1/0.67 + 1/0.33) / 1,000)``
    (unit noise), so the gated ratio has a wide margin. With ``A`` forced into the bias model
    the external arms get free mean shifts and the ratio is 1.00: no first-order gain on any
    law with a constant trial randomization and constant noise variance, which is why the
    engine penalizes ``A``."""
    from cleverly.estimators._working_basis import WorkingBasis

    law = laws.LAWS["a0"]
    model = _model(law, WorkingBasis())
    assert model.dictionary.psi_names[:2] == ("1", "A")
    trial_only = np.sqrt((1 / 0.67 + 1 / 0.33) / 1_000)
    ratios = {}
    for bias in ([0], [0, 1]):
        selection = SimpleNamespace(
            dictionary=model.dictionary,
            effect=SimpleNamespace(selected=np.array([0])),
            bias=SimpleNamespace(selected=np.array(bias)),
        )
        truth, _, table = laws.oracle_parts(law, selection, "pooled", n=300_000, seed=24)
        assert truth.value == pytest.approx(1.5, abs=1e-6)
        ratios[len(bias)] = float(table["total"].std()) / np.sqrt(N) / trial_only
    assert ratios[1] == pytest.approx(0.484, abs=0.01)
    assert ratios[2] == pytest.approx(1.0, abs=0.01)


def test_an_arm_product_without_the_arm_column_misses_the_effect_by_a_sample_mean() -> None:
    """The witness of ``atmle.arm_hierarchy``. On ``a3`` the external treated rows' bias is
    ``2.1 W1`` and ``E[W1] = 0``, so ``{1, A*W1}`` holds the truth in population-centred
    columns. The dictionary centres ``W1`` at a sample mean, and there that selection's
    projection misses ``Psi`` by more than half the standard error of ``D_w`` at the study's
    ``n`` (1.15 on this draw), a bias of the sample mean's order at every ``n``. ``A`` added
    restores ``Psi``, and so does centring at the population mean."""
    from cleverly.estimators._working_basis import WorkingBasis

    law = laws.LAWS["a3"]

    def selection(population_centred: bool, columns: list[str]) -> SimpleNamespace:
        model = _model(law, WorkingBasis())
        if population_centred:
            basis = model.dictionary.basis
            for j, knots in basis.knots_.items():
                basis.knots_[j] = knots + basis.mean_[j] / basis.scale_[j]
            basis.mean_ = np.zeros_like(basis.mean_)
        names = list(model.dictionary.psi_names)
        model.bias = SimpleNamespace(selected=np.array([names.index(c) for c in columns]))
        return model

    psi = study.exact("a3")["psi"]
    sparse = selection(False, ["1", "A*x0"])
    _, _, table = laws.oracle_parts(law, sparse, "pooled", n=300_000, seed=24)
    se = float(table["total"].std()) / np.sqrt(N)
    assert abs(laws.projection(law, sparse, "pooled").value - psi) > 0.5 * se
    for centred, columns in ((False, ["1", "A", "A*x0"]), (True, ["1", "A*x0"])):
        value = laws.projection(law, selection(centred, columns), "pooled").value
        assert value == pytest.approx(psi, abs=0.02 * se)


def test_a_cheap_cell_fits_and_every_extra_column_is_present() -> None:
    """Fit one small cell through the public workflow and read every column of its row."""
    frame, _ = study.draw_from_seed("b1y", 900, 5)
    values = study.fit_cell(frame, "b1y", 5)
    assert set(study.EXTRA_COLUMNS) <= set(values)
    assert values["estimand"] == "psi"
    assert values["truth"] == pytest.approx(study.exact("b1y")["psi"])
    assert values["ci_lower"] < values["estimate"] < values["ci_upper"]
    assert values["converged"] in (0, 1)


def test_the_k2_and_k3_mutations_change_the_fit() -> None:
    """The study driver's two engine options take effect: K2 keeps every main-terms column with
    no penalty, and K3 leaves Pi at its initial fit (epsilon 0)."""
    law = laws.LAWS["a3"]
    frame = laws.draw(law, 1_200, 4)
    plain = laws.fit_trial(frame, law, seed=1, form="pooled")
    off = laws.fit_trial(
        frame, law, seed=1, form="pooled", mutation=laws.Mutation(no_source_step=True)
    )
    assert off.fit.epsilon == (0.0,)
    assert plain.fit.epsilon != (0.0,)
    e1 = laws.LAWS["e1"]
    frame = laws.draw(e1, 1_000, 4)
    fitted = laws.fit_trial(
        frame, e1, seed=1, basis=laws.MainTerms(), mutation=laws.Mutation(unpenalized=True)
    )
    model = fitted.working_models[0]
    assert model.effect.selected.size == 4  # [1, W1, W2, W3]
    assert model.effect.penalty == 0.0
    # The engine's own functions are restored after the fit.
    from cleverly.estimators import atmle

    assert atmle.select_working_model.__name__ == "select_working_model"
    assert atmle.MAX_PASSES == 50


# ----------------------------------------------------------------------------- the record


def test_the_declared_seeds_meet_no_other_study_and_no_pilot() -> None:
    """The record's sample seeds equal no registered study's, and the pilot's differ."""
    mine = {replicate_seed(study.STUDY, s, r) for s in study.STUDY.scenarios for r in range(50)}
    for other in registered():
        if other.slug == study.STUDY.slug:
            continue
        theirs = {replicate_seed(other, s, r) for s in other.scenarios for r in range(50)}
        assert not mine & theirs, other.slug
        assert other.seed != study.SEED and other.resampling_seed != study.RESAMPLING_SEED
    from tests.canonical.adaptive_tmle_trial.pilot import PILOT_SEEDS

    pilots = {20290399, 20290397, 20290393, *(s for pair in PILOT_SEEDS.values() for s in pair)}
    declared = {study.SEED, study.RESAMPLING_SEED, selective.SEED, selective.RESAMPLING_SEED}
    assert not pilots & declared
    assert not pilots & {
        seed
        for other in registered()
        if other.slug != study.STUDY.slug
        for seed in (other.seed, other.resampling_seed)
    }
    assert len(
        {stream_seed(study.STUDY, "fold_partition", s, 0) for s in study.STUDY.scenarios}
    ) == len(study.STUDY.scenarios)


def test_the_record_lists_every_study_specific_module() -> None:
    """The manifest hashes the modules ``STUDY.modules`` lists; the walk of
    ``test_study_provenance`` must find nothing else."""
    required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
    assert required <= set(study.STUDY.modules)


def test_the_record_declares_what_the_plan_declares() -> None:
    record = study.STUDY
    assert record.n == 4_000 and record.truth_varies_by_replicate
    assert record.publication_policy == "gated" and record.reference is None
    assert record.scenario_n == study.LADDER_N
    assert properties.FAILURE_BOUND == 0.01 and record.failure_bound == {study.IMPLEMENTATION: 0.01}
    assert study.COMPARATOR["commit"] == "845280889d2fb7d929adb494759b1588ab776f43"
    assert study.COMPARATOR["comparison"] == "matching implementation"
    assert sizing.BUDGET_STEP == 400 and R % 400 == 0


def test_a_refused_replicate_is_recorded_as_a_failure_not_a_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan M2: a fit the method refuses counts as failed, non-covering and reported. The
    driver stops on a CapabilityError and records a DataError, so the study translates it."""
    from cleverly._refusal import RefusalReason
    from cleverly.exceptions import CapabilityError, DataError
    from cleverly.validation._failures import recordable

    def refuse(*args: object, **kwargs: object) -> None:
        raise CapabilityError(
            "the working-model fit of fold 2 is rank deficient",
            reason=RefusalReason.INVALID_DECLARATION,
        )

    monkeypatch.setattr(laws, "fit_trial", refuse)
    frame, _ = study.draw_from_seed("b1y", 300, 1)
    with pytest.raises(DataError) as caught:
        study.fit_cell(frame, "b1y", 1)
    assert recordable(caught.value) and "fold 2" in str(caught.value)


# ------------------------------------------------------------------- the selective-enrollment study


def test_the_selective_law_constants_and_overlap() -> None:
    """``S6_SD0`` is the SD of the unscaled wiggle under A = 0, and about 1.5% of the covariate
    mass has ``P(S = 1 | W)`` outside [0.05, 0.95] (the paper's 1.4% to rounding)."""
    w = np.random.default_rng(11).normal(size=(2_000_000, 3))
    raw = np.sin(2 * w[:, 0]) + np.cos(3 * w[:, 1]) + np.sin(2 * w[:, 2])
    assert abs(raw.std() - laws.S6_SD0) < 0.003
    p = laws.LAWS["s6"].s(w)
    outside = float(np.mean((p < 0.05) | (p > 0.95)))
    assert 0.013 < outside < 0.016


def test_the_selective_record_declares_its_own_seeds_and_size() -> None:
    record = selective.STUDY
    assert record.n == selective.PRIMARY_N == 3_200 and record.truth_varies_by_replicate
    assert selective.PRIMARY_REPLICATES % sizing.BUDGET_STEP == 0
    seeds = {record.seed, record.resampling_seed}
    for other in [*registered(), study.STUDY]:
        assert not seeds & {other.seed, other.resampling_seed}, other.slug
        mine = {replicate_seed(record, s, r) for s in record.scenarios for r in range(50)}
        theirs = {replicate_seed(other, s, r) for s in other.scenarios for r in range(50)}
        assert not mine & theirs
    required = _required_from(record.runner_module, record.properties_module)
    assert required <= set(record.modules)


def test_the_k6_row_reports_and_never_gates() -> None:
    """A pilot found main-terms nuisances cover at 0.93 on S6, so K6 is a reporting row."""
    rng = np.random.default_rng(3)
    n = 400
    rows = pd.DataFrame(
        {
            "scenario": "s6_cv",
            "control_truth": 1.5,
            "control_ci_lower": 1.5 - 0.5 + rng.normal(0, 0.01, n),
            "control_ci_upper": 1.5 + 0.5,
        }
    )
    table = selective.controls_table(rows)
    assert list(table["role"]) == ["reporting"] and bool(table["passed"].all())
    assert table["value"].iloc[0] == 1.0


# ------------------------------------------------------------------------------------ the sizing

#: The sizing pilot (``python -m tests.canonical.adaptive_tmle_trial.pilot``, commit 685171b2, run
#: on the 16-core laptop on 2026-11-06): 800 replicates per scenario on the pilot seeds, 0 lost
#: replicates, every control passed (efficiency ``a0`` RMSE ratio 0.703 [0.654, 0.754];
#: ``k3_targeted`` 0.941). These are the covered counts of each gated scenario's interval.
#: ``b1y`` is a reported stress row (``study.REPORTED``): the pilot read 708 of 800, and the rule
#: reads only gated cells, because at that count no budget up to its cap meets it.
PILOT_REPLICATES = 800
PILOT_COVERED = {
    "a0": 754,
    "a1a": 759,
    "a1b": 746,
    "a2c": 764,
    "a2d": 765,
    "a3": 760,
    "a3_pi": 753,
    "a4": 756,
    "a5": 759,
    "e1": 759,
    "e1_k2_n1000": 767,
    "e1_k2_n2000": 756,
    "f1a": 751,
    "f1b": 755,
    "h2_pooled": 761,
    "h2_trial": 752,
}


def test_b1y_is_a_reported_stress_row_and_every_other_cell_is_gated() -> None:
    """``b1y`` is declared in ``reported_scenarios`` and is the only such scenario, and the
    sizing reads exactly the gated cells."""
    assert study.REPORTED == frozenset({"b1y"}) == study.STUDY.reported_scenarios
    assert set(PILOT_COVERED) == set(study.SCENARIOS) - study.REPORTED


def test_the_replicate_count_is_the_primary_rule_at_the_pilot() -> None:
    """``R`` is ``repeated_sizing.primary_replicates`` at the pilot's covered counts.

    The rule prices each cell at the one-sided 80% lower bound of its pilot coverage (clause
    c), so a pilot that happens to read high cannot under-size the run (method-benchmarking:
    "Size the budget against the binding confidence bound, not the most favorable point
    estimate"). It also needs the bias clause and the per-cell clause at coverage 0.93, which
    set its floor of 2,400."""
    assert set(PILOT_COVERED) <= set(study.SCENARIOS)
    assert (
        sizing.primary_replicates(MARGINS, PILOT_COVERED, PILOT_REPLICATES, implementations=1) == R
    )


#: The selective study's pilot, in the same form: 1,200 replicates per scenario at commit 4486fc95
#: on the 16-core local queue (2026-11-07, 0 lost replicates, 545.46 core-seconds per replicate
#: set). The rule prices them at 2,400, its floor.
SELECTIVE_PILOT_REPLICATES = 1200
SELECTIVE_PILOT_COVERED = {"s6_cv": 1159, "s6_pooled": 1126}


def test_the_selective_replicate_count_is_the_primary_rule_at_its_pilot() -> None:
    assert set(SELECTIVE_PILOT_COVERED) <= set(selective.SCENARIOS)
    assert (
        sizing.primary_replicates(
            MARGINS, SELECTIVE_PILOT_COVERED, SELECTIVE_PILOT_REPLICATES, implementations=1
        )
        == selective.PRIMARY_REPLICATES
    )
