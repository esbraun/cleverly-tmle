# CV-TMLE of the fold-local learned-rule value

This directory holds the gated repeated-sampling study of the learned-rule value that roadmap row
RM30 adds. The subject is
`TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`.
The target is the average, over the ten outer folds, of the value of the plug-in rule that each
fold learns on its training rows. Van der Laan and Luedtke (2015), Section 7 and Appendix B, define
the target and its inference.

The RM30 section of `docs/roadmap.md` declares every quantity of this study before any run: the
laws, the learners, the sizes, the budgets, the seeds, the fold-partition seeds, the truth rule,
the failed-fit rule and the run form. Its subsection "The two studies, declared before they run"
is the governing text.

| part | declaration |
| --- | --- |
| scenarios | `non_exceptional` (blip `0.1 + W1`) and `misspecified_limit` (blip `0.8 W1 + W1^2 - 0.3`), 6,000 replications each at n = 2,000 |
| truth | each replication refits each fold's outcome learner on the fit's training rows, requires the fit's rule on every validation row, and integrates the rules' values by the trapezoid rule on 4,001 points of `W1` for each `W2` |
| properties | `interval_calibration` (17,000), `root_n_and_efficiency` (6,000 at each of 500, 2,000 and 8,000), `targeting_necessity` (2,655), and `fold_locality` (2,655) |
| verdicts | the framework defaults, read on the error `estimate - truth` of each row, because the record sets `truth_varies_by_replicate` |
| comparator | none. `equivalence.csv` is empty and schema-valid |

The artefacts are the six the framework writes, a `manifest.json`, and `harness.csv.gz`. The last
file holds the truth, the oracle standard error, the rule rows checked and the solver warnings of
each primary replication. The property rows carry the same three harness columns.

## Run

Use the main checkout's Python 3.13.7 with `PYTHONPATH` naming this tree's `src` and root, and one
thread per numerical library. Run nothing else on the machine.

A declared run takes `--output` and `--jobs` only. `--output` names an empty scratch directory
outside the repository. The run refuses to start unless the tree is clean, `HEAD` equals its
upstream, `cleverly` imports from this tree's `src`, the runtime is the one rule L7 declares,
and every thread variable is 1. It writes the artefacts, the manifest and `run.log` to the
scratch directory, then copies them here. It copies them also when a gated verdict fails.
Run this command:

```bash
python -u -m tests.canonical.learned_rule_cvtmle.regenerate --jobs 16 --output <empty scratch directory>
```

A smoke run is any other `--replicates`. It draws from the throwaway seeds of rule L9, writes to a
scratch `--output` outside the repository, and skips the property study:

```bash
python -u -m tests.canonical.learned_rule_cvtmle.regenerate --replicates 4 --jobs 16 --allow-failures --output <scratch>
```

`--property-harness-check CAP` adds a smoke-only check of the property path. It fits at most `CAP`
throwaway draws of each property cell and prints the rule agreement, the solver warnings and the
wall time.

The declarations are in `tests/studies/learned_rule_cvtmle.py`,
`tests/studies/learned_rule_cvtmle_properties.py` and `tests/studies/_learned_rule_law.py`. The
run form is in `tests/canonical/learned_rule_run.py`.
