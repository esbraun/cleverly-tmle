"""RM40b smoke pilot (plan 5.6): R = 200 per block on fresh seeds; positive verdicts never read.

Command: `PYTHONPATH=. python reviews/rm40/rm40b_pilot.py 200 3` with one BLAS thread. The
fresh seeds replace the record's seeds through a forked pool. The output is `rm40b_pilot.log`.
Only control and reported cells print their statistics; positive cells print exits and seconds.
"""
import dataclasses, sys, time
import pandas as pd
from tests.studies import drtmle_one_wrong_properties as p
p.STUDY = dataclasses.replace(p.STUDY, seed=99_400_001, resampling_seed=99_400_002)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
t = time.time()
import joblib
with joblib.parallel_config(backend="multiprocessing"):  # fork keeps the fresh seeds
    rows = p.generate_property_rows(n_jobs=int(sys.argv[2]), budget=int(sys.argv[1]))
print("wall secs", round(time.time() - t), "fits rows", len(rows))
rows.to_csv("/tmp/rm40b_pilot_rows.csv.gz", index=False)
s = p.summarize_properties(rows)
ctl = s["role"] != "positive"
print(s.loc[ctl, ["cell", "replicates", "standardized_bias", "se_ratio", "coverage", "growth", "growth_ci_lower", "passed", "remainder_proxy_over_sd", "nonconverged_fits", "seconds_per_fit"]].to_string())
print(s.loc[~ctl, ["cell", "replicates", "nonconverged_fits", "seconds_per_fit"]].to_string())
