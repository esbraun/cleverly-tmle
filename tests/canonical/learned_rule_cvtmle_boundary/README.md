# CV-TMLE of the fold-local learned-rule value at exceptional and weak-blip laws

This directory holds the reporting study of the learned-rule value that roadmap row RM30 declares.
The estimator, the learners, the size, the truth rule and the oracle standard error are those of
the gated study in `../learned_rule_cvtmle/`. Only the laws differ.

| scenario | blip | condition C3 |
| --- | --- | --- |
| `exceptional` | `0` | fails. Every rule has the value 0.512179, and the fold rules have no fixed limit |
| `weak_blip` | `0.15 W1` | holds. A red cell is a finite-sample limit at n = 2,000 |

Each scenario runs 6,000 replications at n = 2,000. The verdicts are the gated study's primary
verdicts, read on the error `estimate - truth` of each row, under `publication_policy="reporting"`.
The study has no property cell. Each red cell enters the red-cell ledger with the owner F27.

`reading.csv` applies the declared reading table to each law's 99% Clopper-Pearson coverage
interval, from the top.

| reading | condition |
| --- | --- |
| `under-covers at the <law> law` | the interval's upper end is below 0.95 |
| `no under-coverage resolved at the declared budget` | the interval's lower end is at or above 0.90 |
| `unresolved` | otherwise |

A smoke run writes `smoke run, not the declared budget` in place of each reading.
`harness.csv.gz` holds the truth, the oracle standard error, the rule rows checked and the solver
warnings of each replication.

## Run

Run this study after the gated study, alone on the machine, with the main checkout's Python 3.13.7.
A declared run refuses to start unless the tree is clean, `HEAD` equals its upstream, and
`cleverly` imports from this tree's `src`:

```bash
python -u -m tests.canonical.learned_rule_cvtmle_boundary.regenerate --jobs 16
```

A smoke run draws from the throwaway seeds of rule L9 and writes to a scratch `--output` outside the
repository:

```bash
python -u -m tests.canonical.learned_rule_cvtmle_boundary.regenerate --replicates 4 --jobs 16 --output <scratch>
```

The declarations are in `tests/studies/learned_rule_cvtmle_boundary.py` and
`tests/studies/_learned_rule_law.py`. The run form is in `tests/canonical/learned_rule_run.py`.
