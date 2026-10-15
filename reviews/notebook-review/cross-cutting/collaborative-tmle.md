# Cross-cutting findings from the collaborative TMLE tutorial revision

The revision of `docs/examples/collaborative-tmle.ipynb` found these items outside its own files.
Each needs one owner before anyone edits the shared file.

| item | file | finding | proposed change |
| --- | --- | --- | --- |
| 1 | `docs/examples/index.md`, row of the collaborative TMLE tutorial | the row describes the old page: a queue lottery, a design-based TMLE, and C-TMLE as a check | question: "how to estimate the ATE when a strong assignment predictor has an uncertain role". Failure mode: "a staffing variable that may be an instrument makes the ordinary TMLE weights extreme. OAT C-TMLE fits g on the outcome prediction, so its weights stay moderate, and its interval needs a correct outcome model" |
| 2 | `docs/examples/index.md`, "death strategy on each page" row | the collaborative TMLE page now uses a binary success outcome and the composite strategy (death counts as an unsuccessful transition) | drop the collaborative TMLE page from the list of pages that use the hypothetical strategy |
| 3 | `docs/examples/survey-nonresponse.ipynb`, title cell | "handles death ... by the hypothetical strategy, as collaborative TMLE does" is now false | drop "as collaborative TMLE does" |
| 4 | `docs/development/example-notebooks.md`, sensitivity table, C-TMLE row | the row asks for `simulated_confounding()` on the collaborative fit. That surface is qualitative and not a bound. The revised page reads the omitted-variable bound on the ordinary TMLE fit of the declared set only | allow the bound on the ordinary TMLE fit alone, or state why the surface is still required |
| 5 | `src/cleverly/methods.py`, `CollaborativeTMLEMethod` docstring of `strategy` | says `"oat"` reports no interval on either design. Since X58a, the in-sample per-arm fit with `ParametricOutcome` and `ParametricMechanism` reports one | name the admitted configuration and `per_arm_parametric_support` |
| 6 | the X58a support predicate | a binary ATE fit has one estimand, yet the interval still needs `Inference(simultaneous=False)`; the default fit takes `generated_design_plugin` | for X58 part (b): admit `simultaneous=True` when the fit has one estimand, since no band is built |
| 7 | `result.assess()` on an admitted per-arm parametric fit | the `evalue` row reads `unavailable`, although the fit reports an interval | check whether the E-value status keys on the method rather than the inference status |

Repeated-sampling numbers on the page come from the review probe
`reviews/notebook-review/probes/collaborative-tmle-oat/` (2,000 draws, `summary.log`) and from the
registered study `ctmle-oat-per-arm-parametric`. The old probe directory
`reviews/notebook-review/probes/collaborative-tmle-final/` no longer backs any claim on the page.

Probe note: a `ParametricOutcome(features="main_terms")` outcome model is misspecified on this law
but nearly unbiased for the ATE (mean error +0.0009 for both fits), because the arm-specific need
terms cancel in the average. It is no control for outcome-model misspecification. The page cites
the registered `pseudo_true` cell instead.
