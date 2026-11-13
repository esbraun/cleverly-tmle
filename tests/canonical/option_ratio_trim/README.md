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
experimental. Their limit text is to say that the trimmed point is consistent only when every iterated
outcome regression is linear in the history, since the trim removes double robustness on the
trimmed share (cell `trim_necessity/ltmle_trimmed`). Pinning `LinearRegression` fixes the
learner class; it does not make a linear regression correct, so the text states the linearity
condition rather than implying the pinned learner guarantees it. It is also to say that no
interval is given (F30) and that the plug-in diagnostic is not a standard error: in the pilot
it was 0.57 of the spread of the trimmed point, with coverage 0.73. No preflight condition
exists for `ratio_trim`; the support key alone keeps a trimmed fit experimental until then.

The evidence page is to narrate the default untrimmed fit's under-coverage in the pilot, 0.894
for LTMLE and 0.903 for SDR, with standard-error ratios 0.936 and 0.949. It is a property of
the default path on this weak-support law and turns no cell red. The page may call it a
finite-sample effect of the large weights only after a larger-n run or a canonical `ltmle`
comparison confirms that.

The primary row is point-only, so it passes on the bias leg alone. A smoke run cannot pass it:
at 200 replicates the gate (`gating_findings`) reports it red with standardized bias 0.156,
because the 99% interval half-width is about 0.18 there. At the pilot's bias of 0.0026 spreads,
the probability that the row is bias-equivalent is 0.66 at 200 replicates and above 0.9999 at
the declared 1600 (`repeated_sizing.bias_equivalent_probability`).

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/option_ratio_trim/regenerate.py --replicates 2 --output /tmp/option-ratio-trim-smoke
```

Run the declared study:

```bash
python tests/canonical/option_ratio_trim/regenerate.py --output <empty scratch directory>
```
