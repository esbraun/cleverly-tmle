"""The declared design of the AdaptiveIPTW pairing with ``drtmle::adaptive_iptw``.

The runner (``tests/canonical/adaptive_iptw_pairing/run_study.R``) carries its own copies of
the study's tables, so a test reads them back and compares.  The fold rule is checked against
the split ``adaptive_iptw`` performs.  The agreement bounds are checked with mutation controls:
a numpy transcription of drtmle's loop (``fluctuateG`` and ``estimateQrn`` at drtmle 538a3a2)
agrees with the package inside each bound, and each mutation of it that a defect would cause
moves past the bound of the scenario class built to see it.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import expit, logit

from tests.studies import adaptive_iptw_laws as laws
from tests.studies import canonical_adaptive_iptw as gate
from tests.studies import canonical_adaptive_iptw_pairing as study
from tests.studies.evidence.registry import ROOT

pytestmark = [
    pytest.mark.xdist_group("adaptive_iptw_pairing_design"),
    # A mutated loop (a flipped tilt) can drive the reduced design near-singular; the emulation
    # reads the fit regardless, and the assertions say whether it moved.
    pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning"),
]

RUNNER = ROOT / "tests" / "canonical" / "adaptive_iptw_pairing" / "run_study.R"


def _r_vector(source: str, name: str) -> str:
    match = re.search(rf"^{name} <- c\(([^\n]*)\)$", source, flags=re.M)
    if match is None:
        match = re.search(rf"^{name} <- (?:c|list)\(\n(.*?)^\)", source, flags=re.S | re.M)
    assert match is not None, f"run_study.R has no {name}"
    return match.group(1)


def _r_strings(body: str) -> list[str]:
    return re.findall(r'"([^"]+)"', body)


def _r_scalar(source: str, name: str) -> float:
    match = re.search(rf"^{name} <- ([0-9.eL+-]+)$", source, flags=re.M)
    assert match is not None, f"run_study.R has no {name}"
    return float(match.group(1).rstrip("L"))


def test_the_declared_numbers() -> None:
    record = study.STUDY
    assert record.publication_policy == "reporting"
    assert (study.SEED, study.RESAMPLING_SEED) == (20262103, 2026210301)
    assert (study.PRIMARY_REPLICATES, study.PRIMARY_N, study.N_FOLDS) == (1_000, 2_000, 10)
    assert (study.AGREEMENT_EXACT, study.AGREEMENT_CROSSFIT) == (1e-4, 0.05)
    assert (study.AGREEMENT_QUANTILE, study.INITIAL_AGREEMENT) == (0.99, 1e-10)
    assert (study.WITNESS_MULTIPLE, study.NUISANCE_GATE) == (10.0, 1e-8)


def test_the_runner_carries_the_module_tables() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    degrees = dict(re.findall(r"(\w+) = (\d+)L", _r_vector(source, "DEGREE")))
    assert {name: int(value) for name, value in degrees.items()} == {
        name: gate.DEGREE[study.DEGREE_KEY[study.DRAWS[scenario.draw].law]]
        for name, scenario in study.SCENARIOS.items()
    }
    assert set(_r_strings(_r_vector(source, "ONE_STEP"))) == set(study.ONE_STEP)
    assert set(_r_strings(_r_vector(source, "CROSS_FITTED"))) == {
        name
        for name, scenario in study.SCENARIOS.items()
        if study.DRAWS[scenario.draw].folds != "none"
    }
    labels = re.findall(r"(\w+) = c\(([^)]*)\)", _r_vector(source, "ARM_LABELS"))
    assert {name: tuple(_r_strings(body)) for name, body in labels} == study.ARM_LABELS
    assert _r_scalar(source, "NUISANCE_GATE") == study.NUISANCE_GATE
    assert _r_scalar(source, "N_FOLDS") == study.N_FOLDS
    for name in study.SCENARIOS:
        assert f'"{name}"' in source or f"{name} = " in source, f"the runner never names {name}"


def test_every_one_step_scenario_fits_the_one_step_form() -> None:
    for name in study.SCENARIOS:
        assert ("one_step" in name) == (name in study.ONE_STEP)


def _drtmle_folds(permutation: np.ndarray, n_folds: int) -> np.ndarray:
    # split(perm, rep(1:K, length = n)): the i-th entry of perm goes to fold ((i - 1) mod K) + 1.
    folds = np.empty(permutation.size, dtype=np.int64)
    folds[permutation] = np.arange(permutation.size) % n_folds
    return folds


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_dealt_folds_are_the_ones_drtmle_splits(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 2_003
    permutation = rng.permutation(n)
    assignment = study.deal(permutation, study.N_FOLDS)
    np.testing.assert_array_equal(assignment, _drtmle_folds(permutation, study.N_FOLDS))
    # The runner hands drtmle order(position) for sample(seq_len(n)).
    position = np.empty(n)
    position[permutation] = np.arange(1, n + 1)
    np.testing.assert_array_equal(np.argsort(position, kind="stable"), permutation)


def test_one_permutation_serves_every_scenario_of_a_draw() -> None:
    for name, scenario in study.SCENARIOS.items():
        owner = study.SEED_OWNERS.get(name, name)
        assert study.SCENARIOS[owner].draw == scenario.draw


# ---------------------------------------------------------------------- mutation controls


def _qr(g: np.ndarray, ind: np.ndarray, y: np.ndarray, folds: np.ndarray, degree: int) -> Any:
    """``estimateQrn``'s ``glm_Qr`` branch: per fold, the regression of ``Y`` on the basis in
    ``qlogis(g)`` among the training rows with ``I = 1``, predicted at the fold's rows."""
    out = np.empty_like(g)
    for fold in np.unique(folds):
        valid = folds == fold
        train = (~valid if (~valid).any() else valid) & (ind == 1)
        model = laws.DesignBasisGLM(degree).fit(g[train][:, None], y[train])
        out[valid] = model.predict(g[valid][:, None])
    return out


def _drtmle(
    g: np.ndarray,
    ind: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    degree: int,
    lower: float,
    *,
    one_step: bool,
    sign: float = 1.0,
    max_iter: int = 100,
    refit_folds: np.ndarray | None = None,
) -> float:
    """drtmle's adaptive_iptw at one key, with the fluctuation solved exactly."""
    fit_folds = folds if refit_folds is None else refit_folds
    qr = _qr(g, ind, y, fit_folds, degree)
    if one_step:
        return float(np.mean(ind / g * y - qr / g * (ind - g)))
    g_star, qr_star, score, rounds = g.copy(), qr, np.inf, 0
    while abs(score) > 1e-8 and rounds < max_iter:
        rounds += 1
        covariate, offset = qr_star / g_star, logit(g_star)

        def equation(epsilon: float, covariate: Any = covariate, offset: Any = offset) -> float:
            return float(np.sum(covariate * (ind - expit(offset + epsilon * covariate))))

        try:
            epsilon = brentq(equation, -50.0, 50.0, xtol=1e-14)
        except ValueError:
            return float("inf")
        g_star = np.maximum(expit(offset + sign * epsilon * covariate), lower)
        qr_star = _qr(g_star, ind, y, fit_folds, degree)
        score = float(np.mean(qr_star / g_star * (ind - g_star)))
    return float(np.mean(ind / g_star * y))


def _effective(g: np.ndarray, ind: np.ndarray, degree: int) -> int:
    distinct = len(np.unique(np.round(logit(g[ind == 1]), 10)))
    return max(1, min(degree, distinct - 1))


Mutation = dict[str, Any]


def _gaps(scenario: str, replicate: int, mutations: dict[str, Mutation]) -> pd.DataFrame:
    """``|cleverly - emulation| / SE`` for each arm key of one replicate, by mutation."""
    key = (scenario, replicate, study.STUDY.n_of(scenario))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample, _, rows = study._replicate(key, study._draw(key))
    reported = pd.DataFrame(rows).set_index("estimand")
    n = len(sample)
    cross = bool(sample["position"].notna().all())
    folds = (
        _drtmle_folds(np.argsort(sample["position"].to_numpy()), study.N_FOLDS)
        if cross
        else np.zeros(n, dtype=np.int64)
    )
    treatment, outcome = sample["A"].to_numpy(), sample["Y"].to_numpy()
    recorded = ~np.isnan(outcome) & ~np.isnan(treatment)
    declared = gate.DEGREE[study.DEGREE_KEY[study.DRAWS[study.SCENARIOS[scenario].draw].law]]
    out = []
    for arm in (0, 1):
        g = sample[f"gn_a{arm}"].to_numpy()
        lower = float(sample[f"lb_a{arm}"].iloc[0])
        ind = (recorded & (treatment == arm)).astype(float)
        y = np.nan_to_num(outcome)
        row = reported.loc[f"ey{arm}"]
        for name, mutation in mutations.items():
            settings = {"folds": folds, "degree": _effective(g, ind, declared), **mutation}
            if "shuffle" in settings:
                rng = np.random.default_rng(settings.pop("shuffle"))
                settings["folds"] = _drtmle_folds(rng.permutation(n), study.N_FOLDS)
            if settings.pop("in_sample_qr", False):
                settings["refit_folds"] = np.zeros(n, dtype=np.int64)
            value = _drtmle(
                g,
                ind,
                y,
                settings.pop("folds"),
                settings.pop("degree"),
                lower,
                one_step=scenario in study.ONE_STEP,
                **settings,
            )
            gap = abs(value - row["estimate"]) / row["std_error"]
            out.append({"arm": arm, "mutation": name, "gap": gap})
    return pd.DataFrame(out)


def _smallest(gaps: pd.DataFrame) -> Callable[[str], float]:
    return lambda name: float(gaps.loc[gaps["mutation"] == name, "gap"].min())


def test_in_sample_targeting_defects_cross_the_exact_bound() -> None:
    # The rate mechanism leaves targeting work in sample, so a dropped refit or a flipped tilt
    # moves the estimate far past AGREEMENT_EXACT, while drtmle's loop itself agrees.
    gaps = _gaps(
        "p_rate_insample",
        0,
        {"faithful": {}, "flipped_tilt": {"sign": -1.0}, "one_round": {"max_iter": 1}},
    )
    smallest = _smallest(gaps)
    assert gaps.loc[gaps["mutation"] == "faithful", "gap"].max() <= study.AGREEMENT_EXACT
    assert smallest("flipped_tilt") > 100 * study.AGREEMENT_EXACT
    assert smallest("one_round") > 10 * study.AGREEMENT_EXACT


@pytest.mark.parametrize(
    ("scenario", "in_sample_multiple"),
    [("pd_known_arms_one_step", 10.0), ("c_l1_crossfit_one_step", 1.0)],
)
def test_fold_defects_cross_the_exact_bound_of_the_one_step_form(
    scenario: str, in_sample_multiple: float
) -> None:
    # The one-step form reads the cross-fitted Qr once, so the folds alone decide it: other
    # folds, or Qr trained on its own fold, move it past AGREEMENT_EXACT.  On law C l1 the
    # composite takes few values, so Qr trained in sample moves the estimate least there.
    gaps = _gaps(
        scenario,
        0,
        {"faithful": {}, "shuffled_folds": {"shuffle": 7}, "in_sample_qr": {"in_sample_qr": True}},
    )
    smallest = _smallest(gaps)
    assert gaps.loc[gaps["mutation"] == "faithful", "gap"].max() <= 1e-10
    assert smallest("shuffled_folds") > 10 * study.AGREEMENT_EXACT
    assert smallest("in_sample_qr") > in_sample_multiple * study.AGREEMENT_EXACT


def test_the_cross_fitted_targeted_loop_agrees_inside_its_bound() -> None:
    # drtmle tilts each row along its own fold's covariate and the package tilts each fold's
    # slab (D9), so the two differ at order 1/n after the first round; a flipped tilt does not.
    gaps = _gaps("pd_known_arms", 0, {"faithful": {}, "flipped_tilt": {"sign": -1.0}})
    assert gaps.loc[gaps["mutation"] == "faithful", "gap"].max() <= study.AGREEMENT_CROSSFIT
    assert _smallest(gaps)("flipped_tilt") > 100 * study.AGREEMENT_CROSSFIT


# ------------------------------------------------------------------------ agreement rule


def _pair(gap: float, *, scenario: str = "p_insample", replicates: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    estimate = 0.5 + 0.01 * rng.standard_normal(replicates)
    base = {
        "scenario": scenario,
        "replicate": np.arange(replicates),
        "estimand": "ey0",
        "std_error": 0.01,
        "initial_estimate": estimate + 0.01,
    }
    subject = pd.DataFrame({**base, "implementation": study.IMPLEMENTATION, "estimate": estimate})
    reference = pd.DataFrame(
        {**base, "implementation": study.REFERENCE, "estimate": estimate + gap * 0.01}
    )
    rows = pd.concat([subject, reference], ignore_index=True)
    rows["inference_estimate"] = rows["estimate"]
    return rows


def test_the_agreement_rule_reads_its_bound() -> None:
    inside = study.agreement(_pair(0.5 * study.AGREEMENT_EXACT))
    outside = study.agreement(_pair(2.0 * study.AGREEMENT_EXACT))
    assert bool(inside["passed"].item()) and not bool(outside["passed"].item())
    assert bool(inside["witnessed"].item())
    crossfit = study.agreement(_pair(0.5 * study.AGREEMENT_CROSSFIT, scenario="pd_known_arms"))
    assert bool(crossfit["passed"].item())
    assert study.agreement_bound("pd_known_arms_one_step") == study.AGREEMENT_EXACT


def test_a_lost_replicate_leaves_the_quantiles_and_the_accounting_counts_it() -> None:
    rows = _pair(0.0)
    rows = rows.drop(rows.index[(rows["implementation"] == study.REFERENCE)][:1])
    table = study.agreement(rows)
    assert bool(table["passed"].item()) and int(table["paired"].item()) == 199
    audit = study.account_failures(None, {"agreement.csv": table})  # type: ignore[arg-type]
    assert bool(audit["passed"].item()) and int(audit["own"].item()) == 0
    failing = study.agreement(_pair(2.0 * study.AGREEMENT_EXACT))
    audit = study.account_failures(None, {"agreement.csv": failing})  # type: ignore[arg-type]
    assert not bool(audit["passed"].item()) and int(audit["own"].item()) == 1


def test_a_descriptive_scenario_reports_its_gap_and_its_one_step_form_carries_the_verdict() -> None:
    reported = study.agreement(_pair(10 * study.AGREEMENT_CROSSFIT, scenario="c_l3_crossfit"))
    assert bool(reported["passed"].item()) and not bool(reported["verdict"].item())
    for name in study.DESCRIPTIVE:
        assert f"{name}_one_step" in study.ONE_STEP
        assert study.SCENARIOS[f"{name}_one_step"].draw == study.SCENARIOS[name].draw


def _slab(
    g: np.ndarray,
    ind: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    degree: int,
    lower: float,
    *,
    own_fold_tilt: bool = True,
) -> float:
    """The package's cross-fitted loop: fold ``k``'s design is ``g`` tilted along fold ``k``'s
    own ``Qr_k / g`` by one pooled step.  ``own_fold_tilt=False`` tilts every fold's design along
    each row's own fold covariate instead, which is drtmle's rule (the mutation)."""
    n_folds = int(folds.max()) + 1
    rows = np.arange(g.size)
    slab = np.tile(g, (n_folds, 1))

    def fit(slab: np.ndarray) -> np.ndarray:
        out = np.empty_like(slab)
        for fold in range(n_folds):
            train = (folds != fold) & (ind == 1)
            model = laws.DesignBasisGLM(degree).fit(slab[fold][train][:, None], y[train])
            out[fold] = model.predict(slab[fold][:, None])
        return out

    qr = fit(slab)
    for _ in range(100):
        g_own, qr_own = slab[folds, rows], qr[folds, rows]
        if abs(np.mean(qr_own / g_own * (ind - g_own))) <= 1e-12:
            break
        covariate, offset = qr_own / g_own, logit(g_own)

        def equation(epsilon: float, covariate: Any = covariate, offset: Any = offset) -> float:
            return float(np.sum(covariate * (ind - expit(offset + epsilon * covariate))))

        epsilon = brentq(equation, -50.0, 50.0, xtol=1e-14)
        direction = qr / slab if own_fold_tilt else np.tile(covariate, (n_folds, 1))
        slab = np.maximum(expit(logit(slab) + epsilon * direction), lower)
        qr = fit(slab)
    return float(np.mean(ind / slab[folds, rows] * y))


def test_the_package_tilts_each_fold_slab_along_its_own_covariate() -> None:
    # The witness behind DESCRIPTIVE: the package's composite cross-fitted loop is the slab loop
    # (D9), not drtmle's per-row rule, so the gap drtmle shows there is the declared difference.
    scenario = "c_l3_crossfit"
    key = (scenario, 0, study.STUDY.n_of(scenario))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample, _, rows = study._replicate(key, study._draw(key))
        reported = pd.DataFrame(rows).set_index("estimand")
        folds = _drtmle_folds(np.argsort(sample["position"].to_numpy()), study.N_FOLDS)
        treatment, outcome = sample["A"].to_numpy(), sample["Y"].to_numpy()
        recorded = ~np.isnan(outcome) & ~np.isnan(treatment)
        for code, label in enumerate(study.ARM_LABELS[scenario]):
            g = sample[f"gn_a{code}"].to_numpy()
            ind = (recorded & (treatment == code)).astype(float)
            y = np.nan_to_num(outcome)
            degree = _effective(g, ind, gate.DEGREE["c"])
            lower = float(sample[f"lb_a{code}"].iloc[0])
            row = reported.loc[f"ey[{label}]"]
            slab = _slab(g, ind, y, folds, degree, lower)
            per_row = _slab(g, ind, y, folds, degree, lower, own_fold_tilt=False)
            assert abs(slab - row["estimate"]) / row["std_error"] <= 1e-4
            assert abs(per_row - row["estimate"]) / row["std_error"] > 1e-4


def test_the_configuration_names_no_key_the_record_owns() -> None:
    # write_manifest refuses such a key only at the end of a run (queue row 49's smoke).
    from tests.studies.evidence.manifest import RECORD_CONFIGURATION_KEYS

    assert not RECORD_CONFIGURATION_KEYS & set(study.CONFIGURATION)
    assert not RECORD_CONFIGURATION_KEYS & set(study.REFERENCE_METADATA)
