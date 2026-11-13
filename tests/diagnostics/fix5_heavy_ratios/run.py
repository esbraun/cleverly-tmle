r"""Heavy inverse weights in the pricing and boundaries studies: the readings, the tail sizes, the probe.

Three readings, written to this directory:

``reading.csv``
    The committed replications of the five-node pricing rows and the boundaries ``p4`` cells
    (``tests/canonical/mtp_pricing/replicates.parquet``,
    ``tests/canonical/mtp_boundaries/{replicates,property-replicates}.parquet``): coverage, the SE
    ratio ``mean(SE) / SD(estimate)``, the RMS ratio ``sqrt(mean(SE^2)) / SD(estimate)``, the
    correlation of the error with the SE, the misses on each side, the coverage by SE quartile and
    the CV of the SE.  Coverage reads each replication's committed ``covered`` flag.  The RMS ratio reads the bias of the variance estimator; the SE ratio adds
    the Jensen gap of a dispersed SE, about ``1 - CV^2 / 2``.  On the pricing rows the misses are
    one-sided and the error correlates with the SE at about -0.75: the draws that hold none of the
    rare rows with large cumulative ratios read a small SE.  :func:`committed_reading` rebuilds it
    from the parquets and nothing else.

``quadrature.csv``
    The exact-law moments of the boundaries law's efficient influence function
    (``tests/studies/mtp_tail_size_common.py``, :func:`law_b_moments`) at orders 40 and 80, for
    ``p1`` to ``p4`` on law B and ``p5`` on law B', with each tail size.

``diagonal.csv``
    The diagonal form of the policy-weighted variance with the true regressions
    (:func:`~tests.studies.mtp_tail_size_common.diagonal_variance`) beside the second moment
    of the curve on the same grid, for each plan with itself and with the natural plan.  The
    two agree to quadrature precision: the change-of-measure identity, checked by integration.

``probe_variance.csv``
    The policy-weighted variance of ``ey_regimen[price_down]`` (and its covariance with the
    natural plan) on the first 24 draws of the ``pricing_ltmle_tau5`` and ``pricing_sdr_tau5``
    scenarios at n = 5,000 with the exact ratio, from the design probe that chose the variance
    learner.  :func:`check_probe` refits each draw with ``variance="policy_weighted"`` and checks
    the shipped code reproduces each number to 1e-10 (the fast tier checks two draws,
    ``tests/unit/test_policy_variance.py``).

``law_p_moments.csv`` (``--law-p``)
    The tail sizes of the pricing law's own curves (:func:`~tests.studies.mtp_tail_size_common.law_p_moments`):
    for each ``tau``, each fit of ``LAW_P_FITS`` and each mean and contrast, the SD, skewness and
    kurtosis of the curve on :math:`10^8` fresh draws with the limit nuisances of a
    :math:`10^7`-row fit, the 1% and 99% chunk-bootstrap quantiles of the kurtosis, and the tail
    size declared at the 99% bound.  About two CPU-hours; ``--checkpoint`` keeps each fit and
    each chunk, so a stopped run resumes.

``law_p_quadrature.csv`` (``--law-p``)
    The same limit curves at ``tau = 2`` integrated on a grid at two orders, the check of the
    Monte Carlo where the law admits one.

``law_b_own_moments.csv`` (``--law-b``)
    The tail sizes of the boundaries fits' own curves
    (:func:`~tests.studies.mtp_tail_size_common.law_b_own_moments`): the primary learners and
    the calibration cells' learners, ``LTMLE`` and ``SDR``, on law B and B', with the limit
    nuisances from the population normal equations on the quadrature grid, at two orders.
    About an hour.

``probe1.csv`` ... ``screen.csv`` (``--probes NAME ... --checkpoint DIR``)
    The design probes the documentation cites, rebuilt from the shipped code by
    :mod:`~tests.diagnostics.fix5_heavy_ratios.probes`, which describes each.  Each writes one
    row per draw (``NAME_rows.csv.gz``) beside its summary.  About seven CPU-hours in all.

``tests/unit/test_heavy_ratio_reading.py`` rebuilds ``reading.csv`` and checks the other
files against the code that produced them; ``tests/unit/test_mtp_tail_size.py`` reads the
law-P and law-B files and ``tests/unit/test_heavy_ratio_probes.py`` the probe summaries.

    python -m tests.diagnostics.fix5_heavy_ratios.run [--probe] [--law-p] [--law-b]
        [--probes NAME ...] [--checkpoint DIR]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import LTMLE, SDR
from tests.diagnostics.fix5_heavy_ratios import probes
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_pricing_common as common
from tests.studies import mtp_tail_size_common as tail
from tests.studies.fractional_glm import QuasiBinomialGLM

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parents[1] / "canonical"
CRITICAL = 1.959964

#: The committed cells this diagnostic reads: ``(source, selector, estimand)``.
PRICING_ROWS = tuple(
    (scenario, estimand)
    for scenario in ("pricing_ltmle_tau5", "pricing_sdr_tau5")
    for estimand in (
        "ey_regimen[price_down]",
        "ate_regimen[price_down vs natural_prices]",
        "ey_regimen[price_up]",
    )
)
BOUNDARIES_CELLS = ("p4_sdr__correctly_specified", "p4__correctly_specified")
BOUNDARIES_ROWS = tuple(
    (scenario, estimand)
    for scenario in ("boundaries_ltmle", "boundaries_sdr")
    for estimand in ("ey_regimen[p4]", "ate_regimen[p4 vs natural]")
)
QUADRATURE = tuple(
    (plan, quantity, order)
    for plan in ("p1", "p2", "p3", "p4", "p5")
    for quantity in ("ey", "ate")
    for order in (40, 80)
)


def describe(frame: pd.DataFrame, source: str, cell: str, estimand: str) -> dict[str, object]:
    """One row of ``reading.csv``."""
    error = frame["estimate"].to_numpy() - frame["truth"].to_numpy()
    se = frame["std_error"].to_numpy()
    covered = frame["covered"].to_numpy(dtype=bool)
    quartile = pd.qcut(se, 4, labels=False)
    sd = float(np.std(error, ddof=1))
    row: dict[str, object] = {
        "source": source,
        "cell": cell,
        "estimand": estimand,
        "replicates": len(frame),
        "coverage": float(np.mean(covered)),
        "se_ratio": float(np.mean(se)) / sd,
        "rms_ratio": float(np.sqrt(np.mean(se**2))) / sd,
        "error_se_correlation": float(np.corrcoef(error, se)[0, 1]),
        "miss_above": float(np.mean(error > CRITICAL * se)),
        "miss_below": float(np.mean(error < -CRITICAL * se)),
        "se_cv": float(np.std(se, ddof=1) / np.mean(se)),
    }
    for k in range(4):
        row[f"coverage_q{k + 1}"] = float(np.mean(covered[quartile == k]))
    return row


def committed_reading() -> pd.DataFrame:
    """``reading.csv`` from the committed replication files."""
    rows = []
    priced = pd.read_parquet(CANONICAL / "mtp_pricing" / "replicates.parquet")
    priced = priced[priced["implementation"] == "cleverly-mtp-pricing"]
    for scenario, estimand in PRICING_ROWS:
        chosen = priced[(priced["scenario"] == scenario) & (priced["estimand"] == estimand)]
        rows.append(describe(chosen, "mtp_pricing/replicates", scenario, estimand))
    cells = pd.read_parquet(CANONICAL / "mtp_boundaries" / "property-replicates.parquet")
    for cell in BOUNDARIES_CELLS:
        chosen = cells[cells["cell"] == cell]
        # A calibration cell records no estimand column; it names its property.
        estimand = str(chosen["property"].iloc[0])
        rows.append(describe(chosen, "mtp_boundaries/property-replicates", cell, estimand))
    bounded = pd.read_parquet(CANONICAL / "mtp_boundaries" / "replicates.parquet")
    bounded = bounded[bounded["implementation"] != "lmtp"]
    for scenario, estimand in BOUNDARIES_ROWS:
        chosen = bounded[(bounded["scenario"] == scenario) & (bounded["estimand"] == estimand)]
        rows.append(describe(chosen, "mtp_boundaries/replicates", scenario, estimand))
    return pd.DataFrame(rows)


def quadrature() -> pd.DataFrame:
    """``quadrature.csv``: the law-B moments at two orders."""
    rows = []
    for plan, quantity, order in QUADRATURE:
        moments = tail.law_b_moments(plan, quantity, order)  # type: ignore[arg-type]
        rows.append(
            {
                "plan": plan,
                "quantity": quantity,
                "order": order,
                "mass": moments.mass,
                "sd": moments.sd,
                "skewness": moments.skewness,
                "kappa": moments.kappa,
                "n_kappa": moments.n_kappa,
            }
        )
    return pd.DataFrame(rows)


def diagonal() -> pd.DataFrame:
    """``diagonal.csv``: the diagonal form against the curve's second moment, ``p1`` to ``p4``."""
    rows = []
    for plan in ("p1", "p2", "p3", "p4"):
        for right in (plan, "natural"):
            rows.append(
                {
                    "plan": plan,
                    "right": right,
                    "form": tail.diagonal_variance(plan, right),
                    "moments": tail.moments_covariance(plan, right),
                }
            )
    return pd.DataFrame(rows)


def probe_fit(estimator: str, rep: int, n: int) -> tuple[float, float, float]:
    """``(psi, n Var, n Cov)`` of ``ey_regimen[price_down]`` on one probe draw."""
    labels = ("natural_prices", "price_down")
    frame, _ = pricing.draw_scenario(f"pricing_{estimator}_tau5", n, rep)
    settings = {
        "reference": "natural_prices",
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": common.OracleRatio(),
        "n_folds": 1,
        "g_bounds": pricing.G_BOUNDS,
        "ratio": "classifier",
        "simultaneous": False,
        "random_state": 0,
        "variance": "policy_weighted",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if estimator == "ltmle":
            model: LTMLE | SDR = LTMLE(
                common.regimens(labels, 5),
                max_iter=100,
                tol=1e-10,
                alpha=pricing.PRIMARY_ALPHA,
                **settings,
            )
        else:
            model = SDR(common.regimens(labels, 5), **settings)
        result = model.fit(frame, outcome="Y", **pricing.roles(5))
    estimate = result["ey_regimen[price_down]"]
    covariance = estimate.policy.covariance
    names = covariance.names
    down, natural = names.index("ey_regimen[price_down]"), names.index("ey_regimen[natural_prices]")
    matrix = covariance.matrix * n
    return float(estimate.psi), float(matrix[down, down]), float(matrix[down, natural])


def check_probe() -> float:
    """The largest relative deviation of the shipped code from ``probe_variance.csv``."""
    reference = pd.read_csv(HERE / "probe_variance.csv")
    worst = 0.0
    for row in reference.itertuples():
        psi, variance, covariance = probe_fit(str(row.est), int(row.rep), int(row.n))
        for got, want in ((psi, row.psi), (variance, row.variance), (covariance, row.covariance)):
            worst = max(worst, abs(got / float(want) - 1.0))
    return worst


#: The quadrature orders of ``law_p_quadrature.csv``.
LAW_P_ORDERS = (24, 40)


def law_p(checkpoint: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``law_p_moments.csv`` and ``law_p_quadrature.csv``."""
    rows, grid = [], []
    for tau in (2, 3, 5):
        for (learners, estimator, quantity), reading in tail.law_p_moments(
            tau, checkpoint=checkpoint
        ).items():
            rows.append(
                {
                    "tau": tau,
                    "learners": learners,
                    "estimator": estimator,
                    "quantity": quantity,
                    "draws": reading.draws,
                    "sd": reading.sd,
                    "skewness": reading.skewness,
                    "kappa": reading.kappa,
                    "kappa_lower": reading.kappa_lower,
                    "kappa_upper": reading.kappa_upper,
                    "n_kappa": reading.n_kappa,
                }
            )
    for (learners, estimator), curve in tail.law_p_curves(2, checkpoint=checkpoint).items():
        for order in LAW_P_ORDERS:
            for quantity, moments in tail.law_p_quadrature(curve, order).items():
                grid.append(
                    {
                        "tau": 2,
                        "learners": learners,
                        "estimator": estimator,
                        "quantity": quantity,
                        "order": order,
                        "mass": moments.mass,
                        "sd": moments.sd,
                        "skewness": moments.skewness,
                        "kappa": moments.kappa,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(grid)


#: The quadrature orders of ``law_b_own_moments.csv``.
LAW_B_ORDERS = (24, 40)


def law_b() -> pd.DataFrame:
    """``law_b_own_moments.csv``."""
    rows = []
    for (label_law, learners, estimator), _ in tail.LAW_B_FITS.items():
        for order in LAW_B_ORDERS:
            moments = tail.law_b_own_moments(label_law, learners, estimator, order)
            for quantity, reading in moments.items():
                rows.append(
                    {
                        "law": label_law,
                        "learners": learners,
                        "estimator": estimator,
                        "quantity": quantity,
                        "order": order,
                        "mass": reading.mass,
                        "sd": reading.sd,
                        "skewness": reading.skewness,
                        "kappa": reading.kappa,
                        "n_kappa": reading.n_kappa,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="refit the 48 probe draws")
    parser.add_argument("--law-p", action="store_true", help="the law-P own-curve tail sizes")
    parser.add_argument("--checkpoint", type=Path, help="keep the law-P fits and chunks here")
    parser.add_argument("--law-b", action="store_true", help="the law-B own-curve tail sizes")
    parser.add_argument(
        "--probes", nargs="+", choices=probes.PROBES, help="rebuild these design probes"
    )
    args = parser.parse_args()
    if args.probes:
        if args.checkpoint is None:
            parser.error("--probes needs --checkpoint")
        for name in args.probes:
            probes.run(name, args.checkpoint / name)
        return
    if args.law_b:
        law_b().to_csv(HERE / "law_b_own_moments.csv", index=False, float_format="%.12g")
        return
    if args.law_p:
        if args.checkpoint is not None:
            args.checkpoint.mkdir(parents=True, exist_ok=True)
        moments, grid = law_p(args.checkpoint)
        moments.to_csv(HERE / "law_p_moments.csv", index=False, float_format="%.10g")
        grid.to_csv(HERE / "law_p_quadrature.csv", index=False, float_format="%.12g")
        return
    committed_reading().to_csv(HERE / "reading.csv", index=False, float_format="%.6g")
    quadrature().to_csv(HERE / "quadrature.csv", index=False, float_format="%.12g")
    diagonal().to_csv(HERE / "diagonal.csv", index=False, float_format="%.15g")
    if args.probe:
        worst = check_probe()
        print(f"largest relative deviation from the probe: {worst:.3g}")
        if worst > 1e-10:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
