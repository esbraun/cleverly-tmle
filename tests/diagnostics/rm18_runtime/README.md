# RM18 runtime isolation

This directory holds the diagnostic that RM18 of `docs/roadmap.md` declares in "The runtime
isolation, declared before it runs". That subsection fixes the arms, the preconditions, the
statistics and the reading rule. The code here applies them and changes no study. The committed
`isolation.csv`, `row-drift.csv`, `run.log` and `arms/` hold the recorded run. RM18's "What the
runtime isolation found" reads them.

| file | what it does |
| --- | --- |
| `row_drift.py` | reads committed rows at two commits with `git show`. It reproduces "What the committed history already separates" and the single-fold control table in "What the pooled update found" |
| `compare.py` | reads the four scratch arms of each study against the committed rows. It writes `isolation.csv` |
| `row-drift.csv`, `isolation.csv` | the long-format output of each module, one statistic per row |
| `run.log` | the commands, preflight, `pip freeze` records, wall times and exit codes of the runs |
| `arms/<study>-<arm>-manifest.json` | the manifest each arm wrote, copied with LF line endings |

Both modules read git history. Run them in a full clone. CI checks out a shallow clone, so the
fast tier does not run them.

## Arms

Each arm pins one code state and one runtime. Pin the code with a detached worktree, and run from
that worktree with `PYTHONPATH=<worktree>/src`. Check `cleverly.__file__` before each arm. An
editable install in the interpreter can otherwise bind a different source tree.

| arm | code | runtime |
| --- | --- | --- |
| `F-R11` | `7d5485a` | Python 3.11.13, SciPy 1.17.1 |
| `F-R13` | `7d5485a` | Python 3.13.7, SciPy 1.18.0 |
| `P-R11` | `56100ce` | Python 3.11.13, SciPy 1.17.1 |
| `P-R13` | `56100ce` | Python 3.13.7, SciPy 1.18.0 |

Give each arm an empty `output` and `cache` directory under `<root>/<study>/<arm>/`. Always pass
`--output`, because its default is the committed study directory. Always pass `--cache`, because
the driver otherwise deletes `samples.csv.gz`, which `compare.py` reads.

```bash
# weighted study: the first arm runs lmtp in Docker
python -m tests.canonical.weighted_lmtp_ltmle.regenerate --output <root>/weighted_lmtp_ltmle/F-R11/output --cache <root>/weighted_lmtp_ltmle/F-R11/cache
# every later weighted arm reuses that lmtp result
cp <root>/weighted_lmtp_ltmle/F-R11/cache/reference-results.csv <root>/weighted_lmtp_ltmle/<arm>/cache/
# end-of-study study: every arm reads the committed lmtp rows
python -m tests.canonical.lmtp_ltmle.regenerate --skip-reference --output <root>/lmtp_ltmle/<arm>/output --cache <root>/lmtp_ltmle/<arm>/cache
```

The weighted study refuses `--skip-reference`, because it writes `reference-inference.csv.gz`.

## Reading

Write each output to a scratch path, and compare it with the committed file. Both modules require
`--output`, and `compare.py` also requires `--publish-manifests`. A bare run therefore cannot
overwrite the committed record.

```bash
python -m tests.diagnostics.rm18_runtime.row_drift --output <scratch>/row-drift.csv
python -m tests.diagnostics.rm18_runtime.compare --arms <root> --output <scratch>/isolation.csv --publish-manifests <scratch>/arms
cmp <scratch>/row-drift.csv tests/diagnostics/rm18_runtime/row-drift.csv
cmp <scratch>/isolation.csv tests/diagnostics/rm18_runtime/isolation.csv
```

`compare.py` marks a study "harness not validated, no attribution" when a precondition fails. It
still records the count of rows outside the 1e-9 tolerance and the largest difference. It also
refuses to validate a study when an arm's manifest breaks a declared fixed condition. Each arm must
record its pinned code commit, a clean worktree, and its runtime's Python and SciPy versions. The
four arms must record one numpy, one pandas and one scikit-learn version. The module prints each
broken condition.

## A standard-error pathology in the weighted study

The largest weighted code-axis change in a standard error in `isolation.csv` is 49,702. It does
not measure the code change. In each `learner_weight_necessity` cell, 13 of 1,200 replicates
report a standard error above 1 at both code states, and in all four arms. Replicate 1099 of the
control reports 49,757.77 at `7d5485a` and 55.32 at `0e03a15`, with the same estimate.

Those replicates set the published SE ratio of both cells in
`tests/canonical/weighted_lmtp_ltmle/properties.csv`. It is 2,758 and 2,825 at `0e03a15`, and
6,772 and 6,925 at `f0110bc`. The verdicts of both cells read the bias intervals and the paired
displacement, and not the reported standard error. No roadmap item owns this pathology.
