# Ratio trimming on LTMLE and SDR (`option-ratio-trim`)

This study measures `LTMLE(ratio_trim=q)` and `SDR(ratio_trim=q)`, which cap every per-node
density ratio at the quantile `q` of all per-node ratios, on a two-node law with weak support
(Fix-8 plan sections 2.5 and 5.6). The trimmed fit withholds its interval, so the study reads
the point estimate only. It has no comparator: every cell is read against the exact truth.

| file | role |
| --- | --- |
| `tests/studies/canonical_option_ratio_trim.py` | The record and the primary phase. |
| `tests/studies/option_ratio_trim_properties.py` | The property families and their rules. |
| `tests/studies/_option_trim_law.py` | The law, learners, fits, trim quantiles and exact-law limits. |
| `tests/studies/option_ratio_trim_pilot.py` | The pre-read pilot and its committed summary. |
| `tests/unit/test_option_ratio_trim_design.py` | The design test: every declared number, recomputed. |
| `tests/unit/test_option_ratio_trim_witness.py` | The fast-tier exact-law witnesses and mutation controls. |

## State of this directory

Declared, not run. No artifact is committed, so the record is not in `registered()`. The four
`G56` support rows (the keys of `G01`, `G02`, `G23.in_sample` and `G23.cross_fitted` with
`ratio_trim=True`, kind `point` only) and the evidence page land with the registration, after
the run. The rows are to set `learners=` to `LinearRegression` for the outcome and
pseudo-outcome regressions, so a trimmed fit with any other outcome learner stays
experimental. Their limit text is to say that the trimmed fit is consistent when every outcome
regression is, that the trim removes double robustness on the trimmed share (cell
`trim_necessity/ltmle_trimmed`), that it was measured only with a linear outcome regression,
and that no interval is given (F30). No preflight condition exists for `ratio_trim`; the
support key alone keeps a trimmed fit experimental until then.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_ratio_trim/regenerate.py --replicates 2 --output /tmp/option-ratio-trim-smoke
```

Run the declared study:

```bash
python tests/canonical/option_ratio_trim/regenerate.py --output <empty scratch directory>
```
