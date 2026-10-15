# Cross-cutting findings from the point-treatment tutorial revision

The point-treatment thread found these while revising `docs/examples/point-treatment-tmle.ipynb`.
Each one touches files that the thread does not own.

| id | finding | evidence | files to change | owner |
| --- | --- | --- | --- | --- |
| PTX-1 | The bound refuses `cf_y = 1`, so a benchmark whose gain is clipped at 1 calibrates no bound. `BenchmarkResult` also hides the unclipped gain. The gains themselves are the published proxies for the bound's strengths, so the notebook passes them unchanged. A share conversion `gain / (1 + gain)` would contradict the source | `reviews/notebook-review/verification/point-treatment-tmle.md` PT-04 (arXiv:2112.13398v6, Appendix E.5, Remark 8). On the point-treatment fit, `discharge_risk` has a raw gain of 1.316, printed as 1.0000 | `src/cleverly/sensitivity/omitted_variable.py` and its documentation | the "Benchmark gains vs bound shares" thread. When it merges, the notebook should bound at the `discharge_risk` benchmark too |
| PTX-2 | Other notebooks link into this notebook by heading anchor | `step-4-write-the-protocol`, `step-5-design-and-identification`, `step-6-estimate-the-ate`, `step-10-sensitivity-what-the-fit-cannot-check` | none. The revision keeps all four headings | none |
