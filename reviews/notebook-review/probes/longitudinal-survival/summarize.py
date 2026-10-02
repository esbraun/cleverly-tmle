import json
import sys

import numpy as np
import pandas as pd

T = json.load(open(".tmp/notebook-review/longitudinal-survival/truth.json"))
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ".tmp/notebook-review/longitudinal-survival/sweep.csv")
n = len(d)
print("seeds:", n)


def block(prefix_truth):
    print(f"{'param':32s} {'truth':>7s} {'mean':>7s} {'bias':>8s} {'emp_sd':>7s} {'mean_se':>7s} {'se/sd':>6s} {'bias/sd':>7s} {'mean_z':>7s} {'cover':>6s}")
    for name, tv in prefix_truth:
        x = d[name]
        se = d[name + "_se"]
        z = (x - tv) / se
        covcol = [c for c in d.columns if c.startswith(name + "_cov")]
        cov = ((x - 1.96 * se <= tv) & (tv <= x + 1.96 * se)).mean()
        print(f"{name:32s} {tv:7.4f} {x.mean():7.4f} {x.mean()-tv:8.4f} {x.std():7.4f} {se.mean():7.4f} {se.mean()/x.std():6.2f} {(x.mean()-tv)/x.std():7.2f} {z.mean():7.2f} {cov:6.3f}")


pairs = []
for p in ("always", "never"):
    for h in (1, 2):
        pairs.append((f"exit_{p}_t{h}", T[f"surv {p} t{h}"]))
for h in (1, 2):
    pairs.append((f"exit_diff_t{h}", T[f"diff surv a-n t{h}"]))
print("\n== retention fit (clustered) ==")
block(pairs)
print("band all-cover fraction:", d["band_all_cover"].mean())
pairs = []
key = {"readmission": "relapse", "death": "death"}
for p in ("always", "never"):
    for c in ("readmission", "death"):
        for h in (1, 2):
            pairs.append((f"cif_{p}_{c}_t{h}", T[f"cif {key[c]} {p} t{h}"]))
for c in ("readmission", "death"):
    for h in (1, 2):
        pairs.append((f"cifd_{c}_t{h}", T[f"diff cif {key[c]} a-n t{h}"]))
print("\n== competing fit ==")
block(pairs)
print("\n== death-as-censoring fit vs CDE functional (death precedes readmission) ==")
pairs = [(f"el_{p}_t{h}", T[f"cde_dfirst {p} t{h}"]) for p in ("always", "never") for h in (1, 2)]
pairs.append(("el_diff_t2", T["diff cde_dfirst a-n t2"]))
block(pairs)
print("== same fit vs total-effect CIF ==")
pairs = [(f"el_{p}_t{h}", T[f"cif relapse {p} t{h}"]) for p in ("always", "never") for h in (1, 2)]
pairs.append(("el_diff_t2", T["diff cif relapse a-n t2"]))
block(pairs)

print("\n== claim fractions ==")
f = lambda s, m: print(f"{m:80s} {s.mean():.3f}  ({int(s.sum())}/{n})")
f((d.naive_t1 > T["diff surv a-n t1"]) & (d.naive_t1 < 0), "naive t1 understates benefit (pop < naive < 0)")
f((d.naive_t2 > T["diff surv a-n t2"]) & (d.naive_t2 < 0), "naive t2 understates benefit (pop < naive < 0)")
f((d.exit_diff_t1 < 0) & (d.exit_diff_t2 < d.exit_diff_t1), "exit diffs both negative and t2 reduction larger")
f(d.exit_diff_t1_cov & d.exit_diff_t2_cov, "both exit diff intervals cover")
f((d.cifd_readmission_t1 < 0) & (d.cifd_readmission_t2.abs() < d.cifd_readmission_t1.abs()), "readmission diff negative at t1 and shrinks toward zero at t2")
f((d.cifd_death_t1 < 0) & (d.cifd_death_t2 < d.cifd_death_t1), "death diff negative both, larger at t2")
f(d.excess_max == 0, "excess 0 every row")
f(d.eps_max_row == "never/death/2/2", "largest |epsilon| on never/death/t2/node2")
cover12 = [c for c in d.columns if (c.startswith("cif_") or c.startswith("cifd_")) and c.endswith("_cov")]
misses = (~d[cover12].astype(bool)).sum(axis=1)
print("competing: mean misses of 12 intervals per draw:", misses.mean(), " P(>=2 misses):", (misses >= 2).mean())
f(d.el_diff_t2 < d.cifd_readmission_t2 - 0.01, "censored diff < total readmission diff - 0.01 (notebook claim 'larger reduction')")
rn = d.el_never_t2 - d.cif_never_readmission_t2
ra = d.el_always_t2 - d.cif_always_readmission_t2
f((rn > ra + 0.01) & (ra > 0), "raised never > raised always + 0.01 and raised always > 0")
f(d.el_diff_t2_cov_cde, "censored diff CI covers CDE functional")
f(d.el_diff_t2_cov_total, "censored diff CI covers total-effect truth")
f(d.el_diff_t2_cov_cde & d.el_diff_t2_cov_total, "censored diff CI covers BOTH")
f((d.el_diff_t2 + 1.96 * d.el_diff_t2_se) < 0, "censored diff CI excludes zero")
f((d.cifd_readmission_t2 - 1.96 * d.cifd_readmission_t2_se < 0) & (d.cifd_readmission_t2 + 1.96 * d.cifd_readmission_t2_se > 0), "total readmission t2 CI includes zero")
print("min ESS ratio retention range:", d.exit_min_ess_ratio.min(), d.exit_min_ess_ratio.max(), "max trunc", d.exit_max_trunc.max())
