# Settings reference

Every setting a fit takes, its default, and the kind of fit that uses it. Look a setting up here
when a guide page names it, or before you change one. [Methods and learners](methods-learners.md)
explains how to choose among them.

You can pass a setting in two ways. Give `estimate` the keyword in the "keyword" column, or build
the group object and pass it inside a method object. The two lines below fit the same way:

```python
from sklearn.linear_model import LinearRegression

from cleverly import CrossFitting, ModelSpec, TMLEMethod

by_keyword = dict(n_folds=5, cross_fit=True, outcome_learner=LinearRegression())
by_object = TMLEMethod(
    models=ModelSpec(outcome_learner=LinearRegression()),
    cross_fitting=CrossFitting(n_folds=5, enabled=True),
)
```

Pass `**by_keyword` or `method=by_object` to `effect.estimate`. Four keywords differ from the
name of the setting they set: `cross_fit`, `stratify_folds`, `targeting` and `cv_evaluation`.

The "used on" column names the fits whose engine reads the setting: *point* is a single treatment
decision (`PointTreatment`), *over time* a longitudinal design or a time-to-event design on a grid
of check times, *exact times* a `TimeToEvent(grid="continuous")` fit, and *trial* a trial with
external data. A fit refuses a setting it does not read when you change it from its default,
rather than ignoring it. Some methods change a default; the last section lists each case.

## Models and learners

Every learner slot takes an unfitted scikit-learn estimator. `None` uses the default library, a
Super Learner over boosting, a random forest and a lasso
([learner choices](methods-learners.md)).

| setting | keyword | default | used on | what it changes |
| --- | --- | --- | --- | --- |
| `outcome_learner` | `outcome_learner` | `None` | point, over time, trial | the model of the outcome given the treatment and the covariates; on a grid survival fit, the model of the event (or of its hazard under `survival_model="hazard"`) |
| `treatment_learner` | `treatment_learner` | `None` | point, over time, exact times, trial | the model of who gets each treatment option, at each time over time |
| `missingness_learner` | `missingness_learner` | `None` | point, trial | the model of whose outcome is observed, and of whose treatment is observed when a missing treatment is declared |
| `intermediate_learner` | `intermediate_learner` | `None` | point | the model of the variable after treatment that a controlled direct effect holds fixed |
| `pseudo_learner` | `pseudo_learner` | `None` | over time | the regression that works backward through the times, with a real-valued target |
| `censoring_learner` | `censoring_learner` | `None` | over time | the model of who stays in follow-up |
| `competing_learner` | `competing_learner` | `None` | over time | the model of the competing event that `eliminate=` prevents. `None` uses the censoring learner |
| `density_bins` | `density_bins` | `None` | point, over time | the number of core bins of a continuous-dose density. `None` grows with the sample |
| `ratio` | `ratio` | `None` | point, over time | how a dose policy's density ratio is estimated: `"density"` or `"classifier"`. `None` chooses per fit |
| `screen_treatment` | `screen_treatment` | `False` | point | drop covariates from the treatment model whose correlation with the treatment is below the threshold |
| `screen_threshold` | `screen_threshold` | `0.1` | point | the absolute correlation that screening keeps |
| `min_retain` | `min_retain` | `None` | point | the fewest covariates screening keeps |
| `hazard_pooling` | `hazard_pooling` | `'node'` | over time | one censoring model per time (`"node"`), or one pooled over every time (`"pooled"`) |
| `outcome_by_arm` | `outcome_by_arm` | `False` | point | one outcome model per treatment option, each fitted on that option's rows |
| `survival_model` | `survival_model` | `None` | over time | how a survival curve is modelled: `"iterated_mean"` regressions or `"hazard"` targeting. `None` is the method's own choice |

Screening keeps each covariate whose absolute correlation with the treatment reaches
`screen_threshold`, and at least `min_retain` of them, before the treatment model is fitted. It
narrows the adjustment set by the data, so report it. A dose policy's density ratio divides two
fitted densities under `ratio="density"`, and reads the odds of a classifier that tells observed
from shifted doses under `"classifier"`; the [density-ratio route](../technical-reference/scope-and-refusals.md#density-ratio-route)
states when `None` picks each. An exact-times fit chooses its hazard models on the engine
(`ContinuousSurvivalTMLE(hazard_library=...)`), not through `ModelSpec`.

## Folds

| setting | keyword | default | used on | what it changes |
| --- | --- | --- | --- | --- |
| `enabled` | `cross_fit` | `True` | point, over time, exact times, trial | whether each row's predictions come from models trained without that row |
| `n_folds` | `n_folds` | `10` | point, over time, exact times, trial | the number of outer folds |
| `learner_folds` | `learner_folds` | `5` | point, over time, exact times, trial | the folds a Super Learner uses inside each training set to weigh its candidates |
| `repeats` | `repeats` | `1` | point, over time | independent fold draws, combined by their median; a learned-rule value combines them by their mean |
| `stratify_by` | `stratify_folds` | `'none'` | point | the fold policy. Only `"none"` is accepted when folds are drawn |
| `targeting_scheme` | `targeting_scheme` | `'pooled'` | point | one correction step over all folds (`"pooled"`), or one per fold (`"fold"`) |
| `fold_evaluation` | `cv_evaluation` | `False` | point | also report the estimate evaluated fold by fold |
| `split_plan` | `split_plan` | `None` | point, trial | reuse the outer folds of an earlier fit |

[Two fold layers](methods-learners.md) explains the two layers and the refusals, and
[CV-TMLE and cross-fitting](../technical-reference/cv-tmle.md) states the contracts.

## The correction step

After the models are fitted, a correction step uses the treatment model to remove the bias the
outcome model leaves. These settings control it.

| setting | keyword | default | used on | what it changes |
| --- | --- | --- | --- | --- |
| `fluctuation` | `fluctuation` | `'logistic'` | point | the form of the correction: `"logistic"` keeps predictions inside the outcome's range, `"linear"` does not |
| `algorithm` | `targeting` | `'iterative'` | point, over time | repeat the correction until the score equation is solved (`"iterative"`), or take small steps along one path (`"one_step"`) |
| `g_bounds` | `g_bounds` | `'auto'` | point, over time, exact times, trial | the cap on extreme treatment probabilities. `"auto"` chooses it from the target |
| `q_bounds` | `q_bounds` | `None` | point, over time | the known range of a continuous outcome. A cross-fitted fit of a continuous outcome needs it |
| `nuisance_bound` | `nuisance_bound` | `0.01` | point | the lower bound of auxiliary regressions |
| `submodel_alpha` | `submodel_alpha` | `0.9995` | point, over time | how far inside the range the logistic correction keeps predictions. Not the interval's `alpha` |
| `target_weights` | `target_weights` | `False` | point | whether observation weights enter the correction |
| `step_size` | `step_size` | `0.001` | point, over time, exact times | the step of `algorithm="one_step"` |
| `max_iter` | `max_iter` | `20` | point, over time, exact times | the most correction rounds |
| `tol` | `tol` | `1e-10` | point, over time, exact times | the tolerance of the default stopping rule |
| `stopping` | `stopping` | `'relative'` | point, over time | the rule that decides the correction is done: `"relative"`, or the `tmle3` rules `"scaled_var"` and `"sample_size"` |
| `step_bound` | `step_bound` | `None` | point, over time | the longest step of one round. `None` takes the full step |
| `step_search` | `step_search` | `True` | point, over time | under `step_bound`, search for the best step length |
| `iterate` | `iterate` | `'last'` | point, over time | report the last round (`"last"`) or the round with the smallest score (`"best"`) |
| `conditional_form` | `conditional_form` | `'arm_mean'` | point | how the ATT and ATC average after the correction; `"known_score"` needs declared treatment chances |
| `isotonic` | `isotonic` | `False` | over time | keep a survival curve and its band from decreasing |
| `ratio_trim` | `ratio_trim` | `None` | over time | trim per-time density ratios at this quantile. A trim that moves a ratio removes the interval |
| `node_bounds` | `node_bounds` | `None` | over time | a declared outcome range at chosen times |

[Targeting and bounds](methods-learners.md) describes the main choices, and
[stopping rules, step bound and best iterate](../technical-reference/point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate)
states their contracts.

## Intervals and bands

| setting | keyword | default | used on | what it changes |
| --- | --- | --- | --- | --- |
| `alpha` | `alpha` | `0.05` | point, over time, exact times, trial | one minus the confidence level: `0.05` gives 95% intervals |
| `n_bootstrap` | `n_bootstrap` | `0` | point, over time, exact times | full-refit bootstrap replicates. `0` uses the standard error from each row's contribution |
| `bootstrap_resampling` | `bootstrap_resampling` | `'auto'` | point, over time | resample rows (`"iid"`) or clusters (`"cluster"`). `"auto"` follows the design |
| `simultaneous` | `simultaneous` | `None` | point, over time, exact times | whether a fit with several estimates adds a band that covers all of them at once. `None` is the method's own choice |
| `n_multiplier` | `n_multiplier` | `10000` | point, over time, exact times | the random draws behind the band |
| `multiplier_kind` | `multiplier_kind` | `'mammen'` | point, over time, exact times | the distribution of those draws |
| `band_scale` | `band_scale` | `'logit'` | point, over time, exact times | build the band of a probability on the logit scale, so it stays inside 0 to 1, or on the estimate's own scale (`"identity"`) |
| `band_reference` | `band_reference` | `'normal'` | point, over time | the reference distribution of the band: normal, or a Student t for each estimate (`"t"`) |

With `simultaneous=None` a fit with several estimates reports the band, except two risk-curve fits:
collaborative TMLE and the drift-corrected DR-TMLE on a time-to-event design, which report
pointwise intervals only. Each bootstrap replicate refits every model, so `n_bootstrap=200`
multiplies the run time by about 200; [count the fits](methods-learners.md) before a long run.
`band_scale="identity"` gives a symmetric band; `band_reference="t"` widens the band of an
estimate with few effective rows. [How every method reports uncertainty](../technical-reference/inference.md)
states when each holds.

## Seeds, run labels and parallel jobs

| setting | keyword | default | used on | what it changes |
| --- | --- | --- | --- | --- |
| `random_state` | `random_state` | `None` | point, over time, exact times, trial | the seed of the folds, the learners and the resampling |
| `run_id` | `run_id` | `None` | point, over time, exact times, trial | a label of your own, stored with the result |
| `n_jobs` | `n_jobs` | `1` | point, over time, exact times | the most parallel jobs the fit starts |

`n_jobs` does not set the threads each model uses inside a job;
[`cleverly.learners.set_thread_limit`](../api/learners.md) does.

## Method options

Each method object takes the five groups above and the options below. `TMLEMethod`, `SDRMethod`,
`IPSIMethod` and `GComputationMethod` take no options of their own.

### Collaborative TMLE

`CollaborativeTMLEMethod` chooses the treatment model with the data.

| setting | default | what it changes |
| --- | --- | --- |
| `strategy` | `'greedy'` | how candidate treatment models are built and chosen ([methods and learners](methods-learners.md) compares them) |
| `preorder` | `None` | a rule that orders the covariates before an ordered search |
| `ordering` | `None` | an explicit covariate order |
| `candidates` | `None` | explicit nested adjustment sets for a discrete search |
| `selection_folds` | `5` | the folds that score each candidate |
| `selection_inner_folds` | `2` | the inner folds a learner uses during the selection |
| `loss` | `'auto'` | the loss that scores candidates |
| `penalty` | `True` | whether a larger candidate pays a complexity penalty |
| `selection_estimand` | `'ate'` | the quantity the selection targets |
| `oat_design` | `None` | the treatment design of `strategy="oat"`; `None` fits one model per treatment option |
| `lambdas` | `None` | the penalty grid of `strategy="lasso"`, or the penalty pair of `strategy="derivative"` |
| `treatment_candidates` | `None` | the declared learners of `strategy="learners"`, least to most flexible |
| `selection_split` | `None` | fold labels that replace the random selection split |
| `patience` | `None` | stop scoring after this many candidates fail to beat the best |

### DR-TMLE

`DRTMLEMethod` adds equations that keep the interval valid when one of the two models is wrong.

| setting | default | what it changes |
| --- | --- | --- |
| `guard` | `('Q', 'g')` | which extra equations to solve: against a wrong outcome model, a wrong treatment model, or both |
| `reduction` | `'univariate'` | the form of the reduced regressions |
| `reduced_outcome_learner` | `None` | the reduced regression of the outcome |
| `reduced_treatment_learner` | `None` | the reduced regression of the treatment |
| `reduced_mechanism_residual_learner` | `None` | the reduced regression of the treatment model's residual |
| `reduced_crossfit` | `'foldwise'` | keep each fold's reduced regressions apart from its data (`"foldwise"`), or R `drtmle`'s pooled form |
| `update_order` | `'drtmle'` | the order of the extra updates |
| `evaluation` | `None` | separate data for a companion fit |
| `randomized` | `False` | whether treatment was randomized, for a fit with missing outcomes |

[Doubly-robust inference](../technical-reference/dr-tmle/index.md) states each option's contract.

### Adaptive-propensity IPTW

| setting | default | what it changes |
| --- | --- | --- |
| `form` | `'targeted'` | report the weighted mean at the adjusted treatment model (`"targeted"`) or the one-step correction (`"one_step"`) |
| `reduced_outcome_learner` | `None` | the regression of the outcome on the fitted treatment probability; `None` takes `outcome_learner` |

[Adaptive-propensity IPTW](../technical-reference/adaptive-iptw.md) states the contract.

### Adaptive TMLE for a trial with external data

| setting | default | what it changes |
| --- | --- | --- |
| `form` | `'cv'` | choose the working models on the other folds (`"cv"`), or on every row (`"pooled"`) |
| `working_basis` | `None` | the covariate terms the working models choose from |
| `source_learner` | `None` | the model of who is in the trial; `None` takes `treatment_learner` |
| `force_arm` | `False` | leave the treatment column of the bias model unpenalized, as the `atmle` package does |

[A small trial with external data](trial-with-external-data.md) shows the options in use.

### EP learner

| setting | default | what it changes |
| --- | --- | --- |
| `sieve` | `None` | the smooth functions that correct the curve; `None` chooses their number by cross-validation |
| `final_learner` | `None` | the model of the curve on the effect-modifier columns |

### Sequential IPW

| setting | default | what it changes |
| --- | --- | --- |
| `normalization` | `'hajek'` | divide the weighted sum by the sum of the weights (`"hajek"`) or by the number of rows (`"horvitz_thompson"`) |

### Parametric g-formula

| setting | default | what it changes |
| --- | --- | --- |
| `n_draws` | `10` | the simulated histories per row behind the estimate |
| `n_derivative_draws` | `None` | the simulated histories behind the derivatives; `None` takes the larger of `n_draws` and 10 |
| `covariate_types` | `None` | the type of each time-varying column |
| `covariate_learner` | `None` | the model of every time-varying covariate; `None` is a main-terms GLM |
| `hazard_learner` | `None` | the model of every hazard of a survival outcome; `None` is a GLM |

## Where a method changes a default

| setting | where | default there |
| --- | --- | --- |
| `n_folds` | `AdaptiveTMLEMethod` | `5` |
| `enabled` | `GComputationMethod` | `False` |
| `enabled` | `IPWMethod` | `False` |
| `enabled` | `ParametricGFormulaMethod` | `False` |
| `simultaneous` | `CollaborativeTMLEMethod` over time | `False` |
| `survival_model` | `CollaborativeTMLEMethod` over time | `'hazard'` |
| `simultaneous` | `DRTMLEMethod` over time | `False` |
| `survival_model` | `DRTMLEMethod` over time | `'hazard'` |
| `max_iter` | `DRTMLEMethod` over time | `200` |
| `g_bounds` | over time | `(0.01, 1.0)` |
| `max_iter` | exact times | `500` |
| `step_size` | exact times | `0.1` |
| `tol` | exact times | `None` |

The three reference estimators for analyses over time also fit a main-terms GLM for every learner
left at `None`, in place of the default library ([report the sequential baselines](longitudinal.md)).
