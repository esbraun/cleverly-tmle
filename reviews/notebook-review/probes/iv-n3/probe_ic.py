"""IV-N3: decompose the cross-fitted shift SE inflation on make_shift_dose(n=3000).

Per seed and configuration, records the reported SE, an independent recomputation of the
influence curve from the fit's own targeted predictions and trimmed ratios, the one-step
estimator built from the initial fit, the IC with the exact density ratio, and the
moments of the held-out ratio per fold.

usage: python probe_ic.py FIRST COUNT WORKERS OUT
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHIFTS = (
    (0.0, None, "current practice"),
    (0.5, 5.0, "+0.5 capped at 5"),
    (0.5, None, "+0.5 uncapped"),
    (1.0, None, "+1.0 uncapped"),
)
KEYS = {
    "one": ("ate_shift[+1.0 uncapped vs current practice]", 3, 1.0),
    "unc": ("ate_shift[+0.5 uncapped vs current practice]", 2, 0.5),
}
CONFIGS = ("xfit_boost", "xfit_boost_notrim", "insample_boost", "xfit_oracle", "xfit_rboost")
if os.environ.get("IVN3_CONFIGS"):
    CONFIGS = tuple(os.environ["IVN3_CONFIGS"].split(","))


def mu(w):
    return 2.0 + 0.7 * w[:, 0] - 0.3 * w[:, 1]


def q0(a, w):
    return 1.0 + 0.5 * a + 0.25 * a**2 + w[:, 0] - 0.5 * w[:, 1] + 0.2 * w[:, 2]


def one_seed(seed):
    sys.path.insert(0, str(ROOT))
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    from cleverly import (
        CausalStudy,
        CrossFitting,
        ModelSpec,
        ModifiedTreatmentPolicyEffect,
        PointTreatment,
        Runtime,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import make_shift_dose, shift_dgp
    from cleverly.interventions import Shift
    from cleverly.learners.density import bin_edges
    from tests.studies.canonical_shift_policies import OracleShiftDensity

    frame, truth = make_shift_dose(n=3000, seed=seed, shifts=SHIFTS)
    a = np.asarray(frame["A"], dtype=float)
    y = np.asarray(frame["Y"], dtype=float)
    w = np.column_stack([np.asarray(frame[c], dtype=float) for c in ("W1", "W2", "W3")])
    def edges_for(nb):
        return tuple(float(v) for v in bin_edges(a, nb))
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), treatment_kind="continuous"
        ),
    ).identify(ModifiedTreatmentPolicyEffect(tuple(Shift(d, cap=c, name=n) for d, c, n in SHIFTS)))

    def rboost():
        return HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=seed
        )

    rows = []
    for name in CONFIGS:
        xfit = name.startswith("xfit")
        nb = 320 if "320" in name else 40
        density = (
            OracleShiftDensity(shift_dgp(), edges_for(nb))
            if "oracle" in name
            else rboost()
            if "rboost" in name
            else HistGradientBoostingClassifier(random_state=seed)
        )
        if "quad" in name:
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures

            outcome = make_pipeline(PolynomialFeatures(2), LinearRegression())
        else:
            outcome = HistGradientBoostingRegressor(random_state=seed)
        targeting = Targeting(q_bounds=(-30.0, 40.0), shift_trim=1.0 if "notrim" in name else 0.999)
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=outcome,
                treatment_learner=density,
                density_bins=nb,
            ),
            cross_fitting=CrossFitting(n_folds=3) if xfit else CrossFitting(enabled=False),
            targeting=targeting,
            runtime=Runtime(random_state=seed, n_jobs=1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = effect.estimate(method=method)
        nu = res.nuisance
        sc = nu.scaler
        fl = res.fluctuations["mtp"]
        ss = nu.shifts
        design = ss.design  # (n, S+1, S), trimmed
        lo, rng = float(sc.lower), float(sc.upper) - float(sc.lower)
        qs_obs = lo + rng * np.asarray(fl.targeted.observed)
        qi_obs = lo + rng * np.asarray(nu.outcome.observed)
        fold_of = np.full(a.size, -1)
        for v, (_, test) in enumerate(nu.folds):
            fold_of[test] = v
        row = {"seed": seed, "config": name}
        for lab, (key, r, delta) in KEYS.items():
            p = res.estimates[key]
            tr = truth[key]
            row[f"{lab}_psi"] = p.psi
            row[f"{lab}_se"] = p.std_error
            row[f"{lab}_truth"] = tr
            row[f"{lab}_cover"] = bool(p.ci[0] <= tr <= p.ci[1])
            # independent IC: D_r - D_0 from trimmed h at the observed dose and Q*
            qs_r = lo + rng * np.asarray(fl.targeted.arms[float(r)])
            qs_0 = lo + rng * np.asarray(fl.targeted.arms[0.0])
            h_r = design[:, 0, r]
            h_0 = design[:, 0, 0]
            psi_re = qs_r.mean() - qs_0.mean()
            ic_re = (h_r - h_0) * (y - qs_obs) + (qs_r - qs_0) - psi_re
            row[f"{lab}_psi_re_diff"] = psi_re - p.psi
            row[f"{lab}_ic_maxdiff"] = float(np.max(np.abs(ic_re - p.influence_curve)))
            # one-step from the initial fit, trimmed and untrimmed h
            qi_r = lo + rng * np.asarray(nu.outcome.arms[float(r)])
            qi_0 = lo + rng * np.asarray(nu.outcome.arms[0.0])
            plug = qi_r.mean() - qi_0.mean()
            resid = y - qi_obs
            corr_trim = float(np.mean((h_r - h_0) * resid))
            corr_raw = float(np.mean((ss.ratio[:, r] - ss.ratio[:, 0]) * resid))
            row[f"{lab}_plugin"] = plug
            row[f"{lab}_onestep"] = plug + corr_trim
            row[f"{lab}_onestep_raw"] = plug + corr_raw
            row[f"{lab}_tmle_move"] = p.psi - plug
            row[f"{lab}_os_move"] = corr_trim
            ic_os = (h_r - h_0) * resid + (qi_r - qi_0) - (plug + corr_trim)
            row[f"{lab}_se_os"] = float(np.std(ic_os, ddof=1) / np.sqrt(a.size))
            # exact density ratio (continuous normal law), same Q*
            h_true = np.exp(delta * (a - mu(w)) - 0.5 * delta**2)
            ic_true_h = (h_true - 1.0) * (y - qs_obs) + (qs_r - qs_0)
            row[f"{lab}_se_trueh"] = float(np.std(ic_true_h, ddof=1) / np.sqrt(a.size))
            ic_true_h_init = (h_true - 1.0) * resid + (qi_r - qi_0)
            row[f"{lab}_se_trueh_init"] = float(np.std(ic_true_h_init, ddof=1) / np.sqrt(a.size))
            top = h_r >= np.quantile(h_r, 0.99)
            row[f"{lab}_qmove_top1pct"] = float(np.mean(np.abs(qs_obs - qi_obs)[top]))
            row[f"{lab}_qmove_rest"] = float(np.mean(np.abs(qs_obs - qi_obs)[~top]))
            row[f"{lab}_resid_init_sd"] = float(np.std(resid))
            row[f"{lab}_resid_star_sd"] = float(np.std(y - qs_obs))
            row[f"{lab}_resid_init_top_sd"] = float(np.sqrt(np.mean(resid[top] ** 2)))
            row[f"{lab}_resid_star_top_sd"] = float(np.sqrt(np.mean((y - qs_obs)[top] ** 2)))
            row[f"{lab}_kappa"] = float(np.sum(h_true * h_r) / np.sum(h_r**2))
            # efficient IC: true h and true Q
            eff = (h_true - 1.0) * (y - q0(a, w)) + q0(a + delta, w) - q0(a, w)
            row[f"{lab}_se_eff"] = float(np.std(eff, ddof=1) / np.sqrt(a.size))
            # concentration of the reported IC
            d2 = np.sort((p.influence_curve - p.influence_curve.mean()) ** 2)[::-1]
            row[f"{lab}_ic_top10_share"] = float(d2[:10].sum() / d2.sum())
            row[f"{lab}_ic_top1pct_share"] = float(d2[: a.size // 100].sum() / d2.sum())
            # residual term vs plug-in term of the reported IC
            row[f"{lab}_var_resid"] = float(np.var((h_r - h_0) * (y - qs_obs)))
            row[f"{lab}_var_plug"] = float(np.var(qs_r - qs_0))
            row[f"{lab}_var_resid_trueh"] = float(np.var((h_true - 1.0) * (y - qs_obs)))
            # ratio moments: estimated vs exact, overall and per fold
            raw = ss.ratio[:, r]
            row[f"{lab}_mean_h"] = float(raw.mean())
            row[f"{lab}_mean_h2"] = float(np.mean(raw**2))
            row[f"{lab}_mean_htrue2"] = float(np.mean(h_true**2))
            row[f"{lab}_max_h"] = float(raw.max())
            row[f"{lab}_ceiling"] = float(ss.ceiling[r]) if ss.ceiling is not None else np.nan
            row[f"{lab}_mse_h"] = float(np.mean((raw - h_true) ** 2))
            row[f"{lab}_corr_logh"] = float(np.corrcoef(np.log(np.maximum(raw, 1e-12)), np.log(h_true))[0, 1])
            hd = ss.ratio_at[:, r, r]  # h_r at d_r(A): what the plug-in step reads
            row[f"{lab}_mean_h_at_d"] = float(hd.mean())
            for v in range(3 if xfit else 0):
                m = fold_of == v
                row[f"{lab}_f{v}_mean_h"] = float(raw[m].mean())
                row[f"{lab}_f{v}_mean_h2"] = float(np.mean(raw[m] ** 2))
        row["eps_one"] = float(fl.epsilon[3])
        row["eps_unc"] = float(fl.epsilon[2])
        row["q_rmse"] = float(np.sqrt(np.mean((qi_obs - q0(a, w)) ** 2)))
        rows.append(row)
    return rows


if __name__ == "__main__":
    first, count, workers = (int(v) for v in sys.argv[1:4])
    out = Path(sys.argv[4])
    sys.path.insert(0, str(ROOT))
    import cleverly

    assert Path(cleverly.__file__).resolve().is_relative_to(ROOT / "src"), cleverly.__file__
    print(cleverly.__file__, flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for chunk in pool.map(one_seed, range(first, first + count)) for r in chunk]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))
