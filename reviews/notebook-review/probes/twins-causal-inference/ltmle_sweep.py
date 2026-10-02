"""Independent LTMLE truth and a sweep over the synthetic draw (sim_rng seed)."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, CrossFitting, Inference, LongitudinalTreatment, ModelSpec, RegimeContrast, Runtime, TMLEMethod

D = "C:/Users/erics/Documents/Projects/cleverly-tmle/.claude/worktrees/bridge-cse_01BbvRGwt6wTxoguDudAmLTi/.tmp/notebook-review/twins-causal-inference/"
SEED = 2026
X = pd.read_csv(D + "twin_pairs_X_3years_samesex.csv")
rows = X.sample(n=6000, random_state=SEED).index.sort_values()
X = X.loc[rows].reset_index(drop=True)
w1 = X["dmar"].fillna(0).to_numpy(float)
w2 = np.maximum(X["diabetes"].fillna(0).to_numpy(float), X["chyper"].fillna(0).to_numpy(float))


def ex(v):
    return 1 / (1 + np.exp(-v))


# Independent truth: enumerate L2 per row, and separately Monte Carlo the intervened system.
def truth_enum(a1, a2):
    tot = 0.0
    for l2 in (0, 1):
        pl = ex(-0.7 + 0.9 * a1 + 0.8 * w2)
        pl = pl if l2 == 1 else 1 - pl
        tot = tot + pl * ex(-2.2 - 0.35 * a1 - 0.55 * a2 + 0.9 * l2 + 0.5 * w2)
    return tot.mean()


mc_rng = np.random.default_rng(99)
def truth_mc(a1, a2, reps=200):
    out = []
    for _ in range(reps):
        l2 = mc_rng.binomial(1, ex(-0.7 + 0.9 * a1 + 0.8 * w2))
        y = mc_rng.binomial(1, ex(-2.2 - 0.35 * a1 - 0.55 * a2 + 0.9 * l2 + 0.5 * w2))
        out.append(y.mean())
    return np.mean(out)


te = truth_enum(1, 1) - truth_enum(0, 0)
tm = truth_mc(1, 1) - truth_mc(0, 0)
print(f"truth enum {te:.6f}  MC(1.2M) {tm:.6f}  notebook asserts -0.06464265")
print(f"share W2=1 {w2.mean():.4f}, W1=1 {w1.mean():.4f}")
sys.stdout.flush()

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
recs = []
for k in range(n_seeds):
    rng = np.random.default_rng(SEED + 1 + k)
    a1 = rng.binomial(1, ex(-0.2 + 0.8 * w1 - 0.6 * w2))
    l2 = rng.binomial(1, ex(-0.7 + 0.9 * a1 + 0.8 * w2))
    a2 = rng.binomial(1, ex(-0.3 + 0.5 * a1 + l2 + 0.4 * w1))
    y = rng.binomial(1, ex(-2.2 - 0.35 * a1 - 0.55 * a2 + 0.9 * l2 + 0.5 * w2))
    semi = pd.DataFrame({"sim_W1": w1, "sim_W2": w2, "sim_A1": a1, "sim_L2": l2, "sim_A2": a2, "sim_Y": y})
    study = CausalStudy(semi, design=LongitudinalTreatment(outcome="sim_Y", treatment=("sim_A1", "sim_A2"),
                        baseline=("sim_W1", "sim_W2"), time_varying=((), ("sim_L2",)), outcome_family="binomial"))
    eff = study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never"))
    m = TMLEMethod(models=ModelSpec(outcome_learner=LogisticRegression(max_iter=3000, random_state=SEED),
                                    pseudo_learner=LinearRegression(),
                                    treatment_learner=LogisticRegression(max_iter=3000, random_state=SEED)),
                   cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
                   inference=Inference(alpha=0.05, simultaneous=False), runtime=Runtime(random_state=SEED, n_jobs=1))
    est = eff.estimate(method=m)["ate_regimen[always vs never]"]
    rr = semi.groupby(["sim_A1", "sim_A2"])["sim_Y"].mean()
    naive = rr.loc[(1, 1)] - rr.loc[(0, 0)]
    lo, hi = est.ci
    recs.append(dict(seed=k, psi=est.psi, se=est.std_error, lo=lo, hi=hi, naive=naive,
                     cov_truth=lo <= te <= hi, cov_naive=lo <= naive <= hi,
                     closer=abs(est.psi - te) < abs(naive - te)))
    print(recs[-1]); sys.stdout.flush()
df = pd.DataFrame(recs)
df.to_csv(D + "ltmle_sweep.csv", index=False)
print("N", len(df))
print("coverage of truth", df.cov_truth.mean(), " covers naive", df.cov_naive.mean(), " LTMLE closer than naive", df.closer.mean())
print("mean psi - truth", (df.psi - te).mean(), " mean naive - truth", (df.naive - te).mean(), " sd psi", df.psi.std(), " mean se", df.se.mean())
