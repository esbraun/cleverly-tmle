"""Gate N3 probe: size of the ordinary remainder on the bounded law at n=2000 with the page's primary nuisances."""
import os
os.environ["OMP_NUM_THREADS"]="1"
import sys; sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from cleverly.datasets import navigation_data
from cleverly.datasets.synthetic import nonlinear_bounded_dgp
COV=["discharge_risk","prior_utilization","medication_burden","age"]
dgp=nonlinear_bounded_dgp()
rows=[]
for seed in range(8000,8020):
    frame,truth=navigation_data(n=2000,seed=seed)
    W=frame[COV].to_numpy(float); A=frame["transition_navigation"].to_numpy(float); Y=frame["transition_score"].to_numpy(float)
    g0=dgp.propensity(W); Q1=dgp.outcome_mean(W,1.0,None); Q0=dgp.outcome_mean(W,0.0,None)
    gh=np.empty(2000); q1=np.empty(2000); q0=np.empty(2000)
    X=np.column_stack([A,W])
    for tr,te in KFold(3,shuffle=True,random_state=seed).split(W):
        gh[te]=LogisticRegression(max_iter=1000).fit(W[tr],A[tr]).predict_proba(W[te])[:,1]
        m=HistGradientBoostingRegressor(random_state=seed).fit(X[tr],Y[tr])
        q1[te]=m.predict(np.column_stack([np.ones(len(te)),W[te]])); q0[te]=m.predict(np.column_stack([np.zeros(len(te)),W[te]]))
    r1=np.mean((gh-g0)/gh*(q1-Q1)); r0=np.mean(((1-gh)-(1-g0))/(1-gh)*(q0-Q0))
    rows.append(dict(seed=seed, rms_q1=np.sqrt(np.mean((q1-Q1)**2)), rms_q0=np.sqrt(np.mean((q0-Q0)**2)),
        mean_abs_g_err=np.mean(np.abs(gh-g0)), rem_1=r1, rem_0=r0, rem_ate=r1-r0,
        rms_g0_g=np.sqrt(np.mean((gh-g0)**2))))
d=pd.DataFrame(rows); pd.set_option("display.width",200)
print(d.round(4).to_string())
print(d.drop(columns="seed").agg(["mean","std"]).round(4).to_string())
