# Parametric g-formula

This study validates `ParametricGFormula` (NICE), the parametric g-formula of Robins (1986). It
simulates forward in time from node-specific main-terms models and averages the simulated outcome
under each plan. The page
[Parametric g-formula](../parametric-gformula.md) states the estimator, its stacked interval, and
the way it differs from `gformula` and `pygformula`.

The study draws from five exact laws, `G1` to `G5`, and a sixth law, `G4Q`, for the report of a
misspecified fit. Every law of the primary phase is the main-terms model that the estimator fits,
so the estimator is correctly specified on it. Every truth is a sum over the branches of the law, with Gauss-Hermite nodes for a Gaussian
variable, or a closed form. `tests/studies/parametric_gformula_laws.py` states each one. The study module
`tests/studies/canonical_parametric_gformula.py` declares the laws, the plans and the reading
rule before the run.

The comparator is `pygformula` 1.2.0. The comparator runs are pending, so this study carries the
label "Python property evidence, with comparator runs pending". A follow-up PR from X45 registers
the runs. The comparison is not acceptance evidence for any claim until that PR merges.

## What was compared

| setting | `cleverly` | `pygformula` 1.2.0 |
| --- | --- | --- |
| status | run and published on this page | pending. No reference process ran, and `equivalence.csv` has no rows |
| datasets | 1,000 samples of 8,000 rows of each of five laws | the identical rows, once the runs exist |
| models | one main-terms `GLM` for each covariate, treatment, censoring and outcome node, in the full-history design | bare column names with a pooled residual SD over time, so only laws where pooled and node-specific models agree can pair |
| simulation | `n_draws = 10` histories for each unit, Philox streams keyed by scenario and replicate | `n_simul = n`, with at least 20 seeds for the Monte Carlo SE |
| intervals | stacked estimating equation with the chain rule and the simulation term, pointwise 95% Wald | none exists. A pending fixture compares the estimate against four combined Monte Carlo SE |

The pending runs use their own environment (`tests/canonical/parametric_gformula_study/Dockerfile`
and `requirements.txt`), and never a package extra. The README of that directory lists the fixtures.

The study has two phases. The primary phase fits each law once for every replicate and reads the
bias, the coverage and the SE ratio against the exact truth. The property phase reads
calibration, the root-n ladder, the Monte Carlo accounting, and the necessity of the
parametric g-formula against a main-terms ICE fit.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law G1: three nodes, a binary covariate, censoring and a binary end-of-study outcome | `ate_regimen[treat_all vs treat_none]` | difference in mean outcome between the plans "treat at each of the three times" against "treat at none of the three times" | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0014 to 0.0018 | 0.9580 | 0.9941 | pass |
| law G1: three nodes, a binary covariate, censoring and a binary end-of-study outcome | `ey_regimen[odds x2]` | mean outcome under the plan multiply the treatment odds by two | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000833 to 0.000317 | 0.9540 | 1.0030 | pass |
| law G1: three nodes, a binary covariate, censoring and a binary end-of-study outcome | `ey_regimen[rule]` | mean outcome under the plan follow the covariate-dependent rule | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000893 to 0.000307 | 0.9360 | 0.9596 | pass |
| law G1: three nodes, a binary covariate, censoring and a binary end-of-study outcome | `ey_regimen[treat_all]` | mean outcome under the plan treat at each of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000954 to 0.000519 | 0.9570 | 1.0256 | pass |
| law G1: three nodes, a binary covariate, censoring and a binary end-of-study outcome | `ey_regimen[treat_none]` | mean outcome under the plan treat at none of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0016 to 0.000803 | 0.9380 | 0.9614 | pass |
| law G2: three nodes, a Gaussian then a binary covariate per node and a continuous outcome | `ate_regimen[treat_all vs treat_none]` | difference in mean outcome between the plans "treat at each of the three times" against "treat at none of the three times" | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0144 to 0.0047 | 0.9470 | 0.9872 | pass |
| law G2: three nodes, a Gaussian then a binary covariate per node and a continuous outcome | `ey_regimen[policy_b]` | mean outcome under the plan treat with probability 0.3 + 0.4 B_t at each of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000406 to 0.0104 | 0.9460 | 0.9998 | pass |
| law G2: three nodes, a Gaussian then a binary covariate per node and a continuous outcome | `ey_regimen[treat_all]` | mean outcome under the plan treat at each of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0044 to 0.0098 | 0.9420 | 0.9912 | pass |
| law G2: three nodes, a Gaussian then a binary covariate per node and a continuous outcome | `ey_regimen[treat_none]` | mean outcome under the plan treat at none of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000333 to 0.0155 | 0.9510 | 0.9978 | pass |
| law G3: four nodes, a binary covariate, censoring and two competing causes | `ate_regimen[treat_all_4 vs treat_none_4, relapse @ t=4]` | difference in cumulative incidence of relapse between the plans "treat at each of the four times" against "treat at none of the four times" at horizon t = 4 | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0013 to 0.0017 | 0.9520 | 1.0197 | pass |
| law G3: four nodes, a binary covariate, censoring and two competing causes | `cif_regimen[treat_all_4, death @ t=4]` | cumulative incidence of death under the plan treat at each of the four times at horizon t = 4 | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000597 to 0.000555 | 0.9480 | 1.0118 | pass |
| law G3: four nodes, a binary covariate, censoring and two competing causes | `cif_regimen[treat_all_4, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at each of the four times at horizon t = 2 | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000410 to 0.000615 | 0.9460 | 1.0020 | pass |
| law G3: four nodes, a binary covariate, censoring and two competing causes | `cif_regimen[treat_all_4, relapse @ t=4]` | cumulative incidence of relapse under the plan treat at each of the four times at horizon t = 4 | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000695 to 0.000793 | 0.9420 | 1.0033 | pass |
| law G4: three nodes, a Gaussian covariate and a continuous dose per node, and a continuous outcome | `ate_regimen[shift_half vs natural course]` | difference in mean outcome between the plans "add 0.5 to the dose at each of the three times" against "leave the observed treatment mechanism unchanged" | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000706 to 0.0015 | 0.9520 | 0.9908 | pass |
| law G4: three nodes, a Gaussian covariate and a continuous dose per node, and a continuous outcome | `ey_regimen[dose_half]` | mean outcome under the plan assign the fixed dose 0.5 at each of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0021 to 0.0012 | 0.9530 | 0.9929 | pass |
| law G4: three nodes, a Gaussian covariate and a continuous dose per node, and a continuous outcome | `ey_regimen[rule_dose]` | mean outcome under the plan assign the dose 0.5 + 0.5 L_t at each of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0032 to 0.0015 | 0.9500 | 0.9948 | pass |
| law G4: three nodes, a Gaussian covariate and a continuous dose per node, and a continuous outcome | `ey_regimen[shift_half]` | mean outcome under the plan add 0.5 to the dose at each of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0052 to 0.0019 | 0.9570 | 1.0089 | pass |
| law G5: three nodes, a binary and a three-level categorical covariate, a binary treatment and a continuous outcome | `ate_regimen[treat_all vs treat_none]` | difference in mean outcome between the plans "treat at each of the three times" against "treat at none of the three times" | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.000804 to 0.0064 | 0.9450 | 0.9950 | pass |
| law G5: three nodes, a binary and a three-level categorical covariate, a binary treatment and a continuous outcome | `ey_regimen[odds x2]` | mean outcome under the plan multiply the treatment odds by two | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0033 to 0.0014 | 0.9470 | 0.9929 | pass |
| law G5: three nodes, a binary and a three-level categorical covariate, a binary treatment and a continuous outcome | `ey_regimen[treat_all]` | mean outcome under the plan treat at each of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0019 to 0.0024 | 0.9530 | 0.9903 | pass |
| law G5: three nodes, a binary and a three-level categorical covariate, a binary treatment and a continuous outcome | `ey_regimen[treat_none]` | mean outcome under the plan treat at none of the three times | `cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs | -0.0056 to 0.000534 | 0.9400 | 0.9600 | pass |
<!-- /generated -->

All 21 implementation-estimand tests pass. They read the four or five estimands of each law at
n = 8,000. No primary replicate was lost to a fitting failure.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `extrapolation_report` | `g4q_dose__main_terms` | diagnostic | parametric g-formula with main-terms models: law G4Q, mean under the fixed dose 0.5, where the outcome carries a quadratic dose term: a main-terms model of a law with a quadratic dose term, fitted and extrapolated to a fixed dose | reported only | bias 1.4840 to 1.4970, coverage 0 to 0.0053, SE ratio 0.9796 | reported |
| `ice_necessity` | `pair_g2_contrast__ice` | control | parametric g-formula against main-terms ICE: law G2, treatment at all three times against none: the main-terms ICE fit on the same draws | bias outside the margin, and displaced from the parametric g-formula by the declared threshold | bias 0.0258 to 0.0374, margin 0.0195, coverage 0.9088 to 0.9476, SE ratio 1.0181 | pass |
| `ice_necessity` | `pair_g2_contrast__nice` | positive | parametric g-formula against main-terms ICE: law G2, treatment at all three times against none: the parametric g-formula with main-terms models | bias inside the margin | bias -0.0039 to 0.0045, margin 0.0141, coverage 0.9402 to 0.9711, SE ratio 1.0243 | pass |
| `ice_necessity` | `pair_g2_treat_all__ice` | control | parametric g-formula against main-terms ICE: law G2, mean under treatment at all three times: the main-terms ICE fit on the same draws | bias outside the margin, and displaced from the parametric g-formula by the declared threshold | bias 0.0706 to 0.0779, margin 0.0123, coverage 0.6204 to 0.6917, SE ratio 0.9885 | pass |
| `ice_necessity` | `pair_g2_treat_all__nice` | positive | parametric g-formula against main-terms ICE: law G2, mean under treatment at all three times: the parametric g-formula with main-terms models | bias inside the margin | bias -0.0038 to 0.0026, margin 0.0107, coverage 0.9421 to 0.9725, SE ratio 1.0026 | pass |
| `ice_necessity` | `pair_g2_treat_none__ice` | control | parametric g-formula against main-terms ICE: law G2, mean under no treatment: the main-terms ICE fit on the same draws | bias outside the margin, and displaced from the parametric g-formula by the declared threshold | bias 0.0373 to 0.0481, margin 0.0182, coverage 0.8830 to 0.9271, SE ratio 1.0133 | pass |
| `ice_necessity` | `pair_g2_treat_none__nice` | positive | parametric g-formula against main-terms ICE: law G2, mean under no treatment: the parametric g-formula with main-terms models | bias inside the margin | bias -0.0045 to 0.0028, margin 0.0123, coverage 0.9306 to 0.9641, SE ratio 0.9852 | pass |
| `ice_reported` | `pair_g1_contrast__ice` | diagnostic | parametric g-formula against main-terms ICE: law G1, treatment at all three times against none: the main-terms ICE fit on the same draws | reported only | bias 0.000863 to 0.0093, coverage 0.9124 to 0.9505, SE ratio 0.9619 | reported |
| `ice_reported` | `pair_g1_contrast__nice` | diagnostic | parametric g-formula against main-terms ICE: law G1, treatment at all three times against none: the parametric g-formula with main-terms models | reported only | bias -0.0012 to 0.0048, coverage 0.9162 to 0.9534, SE ratio 0.9711 | reported |
| `ice_reported` | `pair_g1_treat_all__ice` | diagnostic | parametric g-formula against main-terms ICE: law G1, mean under treatment at all three times: the main-terms ICE fit on the same draws | reported only | bias -0.000408 to 0.0029, coverage 0.9238 to 0.9591, SE ratio 0.9553 | reported |
| `ice_reported` | `pair_g1_treat_all__nice` | diagnostic | parametric g-formula against main-terms ICE: law G1, mean under treatment at all three times: the parametric g-formula with main-terms models | reported only | bias -0.0010 to 0.0018, coverage 0.9228 to 0.9584, SE ratio 0.9701 | reported |
| `ice_reported` | `pair_g1_treat_none__ice` | diagnostic | parametric g-formula against main-terms ICE: law G1, mean under no treatment: the main-terms ICE fit on the same draws | reported only | bias -0.0078 to 0.000093, coverage 0.9031 to 0.9432, SE ratio 0.9591 | reported |
| `ice_reported` | `pair_g1_treat_none__nice` | diagnostic | parametric g-formula against main-terms ICE: law G1, mean under no treatment: the parametric g-formula with main-terms models | reported only | bias -0.0035 to 0.000751, coverage 0.9200 to 0.9562, SE ratio 0.9856 | reported |
| `interval_calibration` | `nice_g1_contrast__correctly_specified` | positive | parametric g-formula: law G1, treatment at all three times against none: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9264 to 0.9518, SE ratio 0.9451 to 1.0165 | pass |
| `interval_calibration` | `nice_g1_contrast__noise_control` | control | parametric g-formula: law G1, treatment at all three times against none: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8146 to 0.8540, SE ratio 0.6861 to 0.7379 | pass |
| `interval_calibration` | `nice_g1_contrast__shrunken_se_control` | control | parametric g-formula: law G1, treatment at all three times against none: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7981 to 0.8390, SE ratio 0.6605 to 0.7115 | pass |
| `interval_calibration` | `nice_g1_odds_x2__correctly_specified` | positive | parametric g-formula: law G1, mean under the odds tilt of two: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9319 to 0.9563, SE ratio 0.9347 to 1.0127 | pass |
| `interval_calibration` | `nice_g1_odds_x2__noise_control` | control | parametric g-formula: law G1, mean under the odds tilt of two: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8172 to 0.8564, SE ratio 0.6831 to 0.7368 | pass |
| `interval_calibration` | `nice_g1_odds_x2__shrunken_se_control` | control | parametric g-formula: law G1, mean under the odds tilt of two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7998 to 0.8406, SE ratio 0.6548 to 0.7071 | pass |
| `interval_calibration` | `nice_g1_rule__correctly_specified` | positive | parametric g-formula: law G1, mean under the rule A_t = L_t: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9328 to 0.9570, SE ratio 0.9567 to 1.0320 | pass |
| `interval_calibration` | `nice_g1_rule__noise_control` | control | parametric g-formula: law G1, mean under the rule A_t = L_t: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8090 to 0.8489, SE ratio 0.6738 to 0.7249 | pass |
| `interval_calibration` | `nice_g1_rule__shrunken_se_control` | control | parametric g-formula: law G1, mean under the rule A_t = L_t: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8059 to 0.8461, SE ratio 0.6699 to 0.7217 | pass |
| `interval_calibration` | `nice_g1_treat_all__correctly_specified` | positive | parametric g-formula: law G1, mean under treatment at all three times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9278 to 0.9529, SE ratio 0.9259 to 1.0000 | **fail** |
| `interval_calibration` | `nice_g1_treat_all__noise_control` | control | parametric g-formula: law G1, mean under treatment at all three times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8203 to 0.8592, SE ratio 0.6762 to 0.7292 | pass |
| `interval_calibration` | `nice_g1_treat_all__shrunken_se_control` | control | parametric g-formula: law G1, mean under treatment at all three times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7959 to 0.8370, SE ratio 0.6482 to 0.6991 | pass |
| `interval_calibration` | `nice_g1_treat_none__correctly_specified` | positive | parametric g-formula: law G1, mean under no treatment: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9305 to 0.9552, SE ratio 0.9554 to 1.0311 | pass |
| `interval_calibration` | `nice_g1_treat_none__noise_control` | control | parametric g-formula: law G1, mean under no treatment: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8164 to 0.8556, SE ratio 0.6671 to 0.7205 | pass |
| `interval_calibration` | `nice_g1_treat_none__shrunken_se_control` | control | parametric g-formula: law G1, mean under no treatment: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8007 to 0.8413, SE ratio 0.6700 to 0.7218 | pass |
| `interval_calibration` | `nice_g2_contrast__correctly_specified` | positive | parametric g-formula: law G2, treatment at all three times against none: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9351 to 0.9589, SE ratio 0.9651 to 1.0384 | pass |
| `interval_calibration` | `nice_g2_contrast__noise_control` | control | parametric g-formula: law G2, treatment at all three times against none: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8159 to 0.8552, SE ratio 0.6829 to 0.7356 | pass |
| `interval_calibration` | `nice_g2_contrast__shrunken_se_control` | control | parametric g-formula: law G2, treatment at all three times against none: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8150 to 0.8544, SE ratio 0.6747 to 0.7275 | pass |
| `interval_calibration` | `nice_g2_policy_b__correctly_specified` | positive | parametric g-formula: law G2, mean under the known policy that treats with probability 0.3 + 0.4 B_t: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9319 to 0.9563, SE ratio 0.9503 to 1.0258 | pass |
| `interval_calibration` | `nice_g2_policy_b__noise_control` | control | parametric g-formula: law G2, mean under the known policy that treats with probability 0.3 + 0.4 B_t: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8185 to 0.8576, SE ratio 0.6915 to 0.7445 | pass |
| `interval_calibration` | `nice_g2_policy_b__shrunken_se_control` | control | parametric g-formula: law G2, mean under the known policy that treats with probability 0.3 + 0.4 B_t: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8085 to 0.8485, SE ratio 0.6654 to 0.7189 | pass |
| `interval_calibration` | `nice_g2_treat_all__correctly_specified` | positive | parametric g-formula: law G2, mean under treatment at all three times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9365 to 0.9600, SE ratio 0.9622 to 1.0434 | pass |
| `interval_calibration` | `nice_g2_treat_all__noise_control` | control | parametric g-formula: law G2, mean under treatment at all three times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8238 to 0.8623, SE ratio 0.6922 to 0.7500 | pass |
| `interval_calibration` | `nice_g2_treat_all__shrunken_se_control` | control | parametric g-formula: law G2, mean under treatment at all three times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8120 to 0.8517, SE ratio 0.6743 to 0.7303 | pass |
| `interval_calibration` | `nice_g2_treat_none__correctly_specified` | positive | parametric g-formula: law G2, mean under no treatment: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9383 to 0.9615, SE ratio 0.9497 to 1.0237 | pass |
| `interval_calibration` | `nice_g2_treat_none__noise_control` | control | parametric g-formula: law G2, mean under no treatment: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8029 to 0.8433, SE ratio 0.6618 to 0.7131 | pass |
| `interval_calibration` | `nice_g2_treat_none__shrunken_se_control` | control | parametric g-formula: law G2, mean under no treatment: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8037 to 0.8441, SE ratio 0.6665 to 0.7172 | pass |
| `interval_calibration` | `nice_g3_contrast_h4__correctly_specified` | positive | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9337 to 0.9578, SE ratio 0.9743 to 1.0479 | pass |
| `interval_calibration` | `nice_g3_contrast_h4__noise_control` | control | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8037 to 0.8441, SE ratio 0.6713 to 0.7230 | pass |
| `interval_calibration` | `nice_g3_contrast_h4__shrunken_se_control` | control | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8116 to 0.8513, SE ratio 0.6816 to 0.7328 | pass |
| `interval_calibration` | `nice_g3_death_h4__correctly_specified` | positive | parametric g-formula: law G3, death incidence by t = 4 under treatment at all four times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9374 to 0.9608, SE ratio 0.9533 to 1.0291 | pass |
| `interval_calibration` | `nice_g3_death_h4__noise_control` | control | parametric g-formula: law G3, death incidence by t = 4 under treatment at all four times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.7968 to 0.8378, SE ratio 0.6643 to 0.7143 | pass |
| `interval_calibration` | `nice_g3_death_h4__shrunken_se_control` | control | parametric g-formula: law G3, death incidence by t = 4 under treatment at all four times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7990 to 0.8398, SE ratio 0.6676 to 0.7202 | pass |
| `interval_calibration` | `nice_g3_relapse_h2__correctly_specified` | positive | parametric g-formula: law G3, relapse incidence by t = 2 under treatment at all four times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9369 to 0.9604, SE ratio 0.9691 to 1.0409 | pass |
| `interval_calibration` | `nice_g3_relapse_h2__noise_control` | control | parametric g-formula: law G3, relapse incidence by t = 2 under treatment at all four times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8055 to 0.8457, SE ratio 0.6721 to 0.7251 | pass |
| `interval_calibration` | `nice_g3_relapse_h2__shrunken_se_control` | control | parametric g-formula: law G3, relapse incidence by t = 2 under treatment at all four times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8024 to 0.8429, SE ratio 0.6787 to 0.7284 | pass |
| `interval_calibration` | `nice_g3_relapse_h4__correctly_specified` | positive | parametric g-formula: law G3, relapse incidence by t = 4 under treatment at all four times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9416 to 0.9641, SE ratio 0.9886 to 1.0647 | pass |
| `interval_calibration` | `nice_g3_relapse_h4__noise_control` | control | parametric g-formula: law G3, relapse incidence by t = 4 under treatment at all four times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8216 to 0.8604, SE ratio 0.6802 to 0.7332 | pass |
| `interval_calibration` | `nice_g3_relapse_h4__shrunken_se_control` | control | parametric g-formula: law G3, relapse incidence by t = 4 under treatment at all four times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8116 to 0.8513, SE ratio 0.6912 to 0.7455 | pass |
| `interval_calibration` | `nice_g4_contrast__correctly_specified` | positive | parametric g-formula: law G4, the raised dose against the natural course: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9333 to 0.9574, SE ratio 0.9735 to 1.0501 | pass |
| `interval_calibration` | `nice_g4_contrast__noise_control` | control | parametric g-formula: law G4, the raised dose against the natural course: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8177 to 0.8568, SE ratio 0.6897 to 0.7445 | pass |
| `interval_calibration` | `nice_g4_contrast__shrunken_se_control` | control | parametric g-formula: law G4, the raised dose against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8098 to 0.8497, SE ratio 0.6813 to 0.7361 | pass |
| `interval_calibration` | `nice_g4_dose__correctly_specified` | positive | parametric g-formula: law G4, mean under the fixed dose 0.5 at each of the three times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9333 to 0.9574, SE ratio 0.9655 to 1.0425 | pass |
| `interval_calibration` | `nice_g4_dose__noise_control` | control | parametric g-formula: law G4, mean under the fixed dose 0.5 at each of the three times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8142 to 0.8536, SE ratio 0.6785 to 0.7343 | pass |
| `interval_calibration` | `nice_g4_dose__shrunken_se_control` | control | parametric g-formula: law G4, mean under the fixed dose 0.5 at each of the three times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8103 to 0.8501, SE ratio 0.6762 to 0.7292 | pass |
| `interval_calibration` | `nice_g4_rule_dose__correctly_specified` | positive | parametric g-formula: law G4, mean under the dose 0.5 + 0.5 L_t set by a rule at each time: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9342 to 0.9582, SE ratio 0.9622 to 1.0378 | pass |
| `interval_calibration` | `nice_g4_rule_dose__noise_control` | control | parametric g-formula: law G4, mean under the dose 0.5 + 0.5 L_t set by a rule at each time: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8120 to 0.8517, SE ratio 0.6915 to 0.7433 | pass |
| `interval_calibration` | `nice_g4_rule_dose__shrunken_se_control` | control | parametric g-formula: law G4, mean under the dose 0.5 + 0.5 L_t set by a rule at each time: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8137 to 0.8532, SE ratio 0.6730 to 0.7262 | pass |
| `interval_calibration` | `nice_g4_shift__correctly_specified` | positive | parametric g-formula: law G4, mean under a dose raised by 0.5 at each of the three times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9379 to 0.9611, SE ratio 0.9719 to 1.0468 | pass |
| `interval_calibration` | `nice_g4_shift__noise_control` | control | parametric g-formula: law G4, mean under a dose raised by 0.5 at each of the three times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8116 to 0.8513, SE ratio 0.6706 to 0.7224 | pass |
| `interval_calibration` | `nice_g4_shift__shrunken_se_control` | control | parametric g-formula: law G4, mean under a dose raised by 0.5 at each of the three times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8085 to 0.8485, SE ratio 0.6812 to 0.7331 | pass |
| `interval_calibration` | `nice_g5_contrast__correctly_specified` | positive | parametric g-formula: law G5, treatment at all three times against none: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9622, SE ratio 0.9760 to 1.0523 | pass |
| `interval_calibration` | `nice_g5_contrast__noise_control` | control | parametric g-formula: law G5, treatment at all three times against none: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8281 to 0.8663, SE ratio 0.7054 to 0.7608 | pass |
| `interval_calibration` | `nice_g5_contrast__shrunken_se_control` | control | parametric g-formula: law G5, treatment at all three times against none: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8177 to 0.8568, SE ratio 0.6833 to 0.7363 | pass |
| `interval_calibration` | `nice_g5_odds_x2__correctly_specified` | positive | parametric g-formula: law G5, mean under the odds tilt of two: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9346 to 0.9585, SE ratio 0.9651 to 1.0402 | pass |
| `interval_calibration` | `nice_g5_odds_x2__noise_control` | control | parametric g-formula: law G5, mean under the odds tilt of two: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8068 to 0.8469, SE ratio 0.6822 to 0.7333 | pass |
| `interval_calibration` | `nice_g5_odds_x2__shrunken_se_control` | control | parametric g-formula: law G5, mean under the odds tilt of two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8150 to 0.8544, SE ratio 0.6764 to 0.7279 | pass |
| `interval_calibration` | `nice_g5_treat_all__correctly_specified` | positive | parametric g-formula: law G5, mean under treatment at all three times: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9388 to 0.9619, SE ratio 0.9780 to 1.0537 | pass |
| `interval_calibration` | `nice_g5_treat_all__noise_control` | control | parametric g-formula: law G5, mean under treatment at all three times: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8242 to 0.8627, SE ratio 0.6927 to 0.7457 | pass |
| `interval_calibration` | `nice_g5_treat_all__shrunken_se_control` | control | parametric g-formula: law G5, mean under treatment at all three times: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8164 to 0.8556, SE ratio 0.6845 to 0.7395 | pass |
| `interval_calibration` | `nice_g5_treat_none__correctly_specified` | positive | parametric g-formula: law G5, mean under no treatment: all three required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9328 to 0.9570, SE ratio 0.9638 to 1.0379 | pass |
| `interval_calibration` | `nice_g5_treat_none__noise_control` | control | parametric g-formula: law G5, mean under no treatment: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8242 to 0.8627, SE ratio 0.6994 to 0.7521 | pass |
| `interval_calibration` | `nice_g5_treat_none__shrunken_se_control` | control | parametric g-formula: law G5, mean under no treatment: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8050 to 0.8453, SE ratio 0.6747 to 0.7261 | pass |
| `monte_carlo_accounting` | `nice_g2_treat_all__mc_removed` | diagnostic | parametric g-formula: law G2, mean under treatment at all three times: the estimate of the one-history fit, with the standard error of an independent fit of 500 histories per unit, so the simulation term is absent from the curve | reported only: the SE-ratio interval, with no verdict | SE ratio 0.8658 to 0.9309 | reported |
| `monte_carlo_accounting` | `nice_g2_treat_all__n_draws_1` | positive | parametric g-formula: law G2, mean under treatment at all three times: one simulated history per unit behind the estimate | bias inside the margin, SE-ratio and coverage intervals inside their calibration bands | SE ratio 0.9688 to 1.0415 | pass |
| `monte_carlo_accounting` | `nice_g2_treat_none__mc_removed` | diagnostic | parametric g-formula: law G2, mean under no treatment: the estimate of the one-history fit, with the standard error of an independent fit of 500 histories per unit, so the simulation term is absent from the curve | reported only: the SE-ratio interval, with no verdict | SE ratio 0.8764 to 0.9448 | reported |
| `monte_carlo_accounting` | `nice_g2_treat_none__n_draws_1` | positive | parametric g-formula: law G2, mean under no treatment: one simulated history per unit behind the estimate | bias inside the margin, SE-ratio and coverage intervals inside their calibration bands | SE ratio 0.9585 to 1.0332 | pass |
| `monte_carlo_accounting` | `nice_g4_dose__mc_removed` | diagnostic | parametric g-formula: law G4, mean under the fixed dose 0.5 at each of the three times: the estimate of the one-history fit, with the standard error of an independent fit of 500 histories per unit, so the simulation term is absent from the curve | reported only: the SE-ratio interval, with no verdict | SE ratio 0.9292 to 1.0026 | reported |
| `monte_carlo_accounting` | `nice_g4_dose__n_draws_1` | positive | parametric g-formula: law G4, mean under the fixed dose 0.5 at each of the three times: one simulated history per unit behind the estimate | bias inside the margin, SE-ratio and coverage intervals inside their calibration bands | SE ratio 0.9730 to 1.0492 | pass |
| `monte_carlo_accounting` | `nice_g4_shift__mc_removed` | diagnostic | parametric g-formula: law G4, mean under a dose raised by 0.5 at each of the three times: the estimate of the one-history fit, with the standard error of an independent fit of 500 histories per unit, so the simulation term is absent from the curve | reported only: the SE-ratio interval, with no verdict | SE ratio 0.9116 to 0.9797 | reported |
| `monte_carlo_accounting` | `nice_g4_shift__n_draws_1` | positive | parametric g-formula: law G4, mean under a dose raised by 0.5 at each of the three times: one simulated history per unit behind the estimate | bias inside the margin, SE-ratio and coverage intervals inside their calibration bands | SE ratio 0.9738 to 1.0470 | pass |
| `monte_carlo_accounting` | `nice_g5_odds_x2__n_draws_1` | positive | parametric g-formula: law G5, mean under the odds tilt of two: one simulated history per unit behind the estimate | bias inside the margin, SE-ratio and coverage intervals inside their calibration bands | SE ratio 0.9488 to 1.0214 | pass |
| `monte_carlo_accounting` | `nice_g5_treat_all__n_draws_1` | positive | parametric g-formula: law G5, mean under treatment at all three times: one simulated history per unit behind the estimate | bias inside the margin, SE-ratio and coverage intervals inside their calibration bands | SE ratio 0.9492 to 1.0237 | pass |
| `monte_carlo_accounting` | `nice_g5_treat_none__n_draws_1` | positive | parametric g-formula: law G5, mean under no treatment: one simulated history per unit behind the estimate | bias inside the margin, SE-ratio and coverage intervals inside their calibration bands | SE ratio 0.9649 to 1.0401 | pass |
| `root_n_and_efficiency` | `nice_g1_contrast__n_2000` | positive | parametric g-formula: law G1, treatment at all three times against none: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0014, coverage 0.9342 to 0.9694, SE ratio 1.0207 | pass |
| `root_n_and_efficiency` | `nice_g1_contrast__n_4000` | positive | parametric g-formula: law G1, treatment at all three times against none: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000402, coverage 0.9249 to 0.9627, SE ratio 0.9860 | pass |
| `root_n_and_efficiency` | `nice_g1_contrast__n_8000` | positive | parametric g-formula: law G1, treatment at all three times against none: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000653, coverage 0.9283 to 0.9652, SE ratio 1.0058 | pass |
| `root_n_and_efficiency` | `nice_g2_contrast__n_2000` | positive | parametric g-formula: law G2, treatment at all three times against none: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0075, coverage 0.9249 to 0.9627, SE ratio 0.9824 | pass |
| `root_n_and_efficiency` | `nice_g2_contrast__n_4000` | positive | parametric g-formula: law G2, treatment at all three times against none: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0016, coverage 0.9295 to 0.9661, SE ratio 1.0042 | pass |
| `root_n_and_efficiency` | `nice_g2_contrast__n_8000` | positive | parametric g-formula: law G2, treatment at all three times against none: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0044, coverage 0.9400 to 0.9734, SE ratio 1.0320 | pass |
| `root_n_and_efficiency` | `nice_g3_contrast_h4__n_2000` | positive | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000326, coverage 0.9307 to 0.9669, SE ratio 0.9928 | pass |
| `root_n_and_efficiency` | `nice_g3_contrast_h4__n_4000` | positive | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000556, coverage 0.9249 to 0.9627, SE ratio 0.9944 | pass |
| `root_n_and_efficiency` | `nice_g3_contrast_h4__n_8000` | positive | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000263, coverage 0.9365 to 0.9710, SE ratio 0.9963 | pass |
| `root_n_and_efficiency` | `nice_g4_contrast__n_2000` | positive | parametric g-formula: law G4, the raised dose against the natural course: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0013, coverage 0.9318 to 0.9677, SE ratio 1.0097 | pass |
| `root_n_and_efficiency` | `nice_g4_contrast__n_4000` | positive | parametric g-formula: law G4, the raised dose against the natural course: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000864, coverage 0.9283 to 0.9652, SE ratio 1.0130 | pass |
| `root_n_and_efficiency` | `nice_g4_contrast__n_8000` | positive | parametric g-formula: law G4, the raised dose against the natural course: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000069, coverage 0.9295 to 0.9661, SE ratio 0.9952 | pass |
| `root_n_rate` | `nice_g1_contrast__empirical_sd` | positive | parametric g-formula: law G1, treatment at all three times against none: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5483 to -0.4295 | pass |
| `root_n_rate` | `nice_g1_contrast__reported_se` | positive | parametric g-formula: law G1, treatment at all three times against none: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5014 to -0.4966 | pass |
| `root_n_rate` | `nice_g2_contrast__empirical_sd` | positive | parametric g-formula: law G2, treatment at all three times against none: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5913 to -0.4772 | pass |
| `root_n_rate` | `nice_g2_contrast__reported_se` | positive | parametric g-formula: law G2, treatment at all three times against none: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5013 to -0.4980 | pass |
| `root_n_rate` | `nice_g3_contrast_h4__empirical_sd` | positive | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5625 to -0.4442 | pass |
| `root_n_rate` | `nice_g3_contrast_h4__reported_se` | positive | parametric g-formula: law G3, relapse incidence by t = 4, treatment at all four times against none: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5009 to -0.4970 | pass |
| `root_n_rate` | `nice_g4_contrast__empirical_sd` | positive | parametric g-formula: law G4, the raised dose against the natural course: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5489 to -0.4293 | pass |
| `root_n_rate` | `nice_g4_contrast__reported_se` | positive | parametric g-formula: law G4, the raised dose against the natural course: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5013 to -0.4978 | pass |
| `shared_target` | `target_g1_contrast__ltmle` | diagnostic | parametric g-formula beside the targeted estimator: law G1, treatment at all three times against none: `LTMLE` with GLM nuisances on the same draws, which targets the same parameter | reported only | bias -0.0025 to 0.0067, coverage 0.9007 to 0.9414, SE ratio 0.9405 | reported |
| `shared_target` | `target_g1_contrast__nice` | diagnostic | parametric g-formula beside the targeted estimator: law G1, treatment at all three times against none: the parametric g-formula with main-terms models | reported only | bias -0.0028 to 0.0031, coverage 0.9224 to 0.9582, SE ratio 0.9820 | reported |
| `shared_target` | `target_g1_odds_x2__ipsi` | diagnostic | parametric g-formula beside the targeted estimator: law G1, mean under the odds tilt of two: `IPSI` with GLM nuisances on the same draws, which targets the same parameter | reported only | bias -0.0015 to 0.000614, coverage 0.9339 to 0.9667, SE ratio 1.0184 | reported |
| `shared_target` | `target_g1_odds_x2__nice` | diagnostic | parametric g-formula beside the targeted estimator: law G1, mean under the odds tilt of two: the parametric g-formula with main-terms models | reported only | bias -0.0014 to 0.000733, coverage 0.9320 to 0.9653, SE ratio 1.0118 | reported |
| `shared_target` | `target_g1_rule__ltmle` | diagnostic | parametric g-formula beside the targeted estimator: law G1, mean under the rule A_t = L_t: `LTMLE` with GLM nuisances on the same draws, which targets the same parameter | reported only | bias -0.0017 to 0.0025, coverage 0.9301 to 0.9639, SE ratio 0.9872 | reported |
| `shared_target` | `target_g1_rule__nice` | diagnostic | parametric g-formula beside the targeted estimator: law G1, mean under the rule A_t = L_t: the parametric g-formula with main-terms models | reported only | bias -0.0014 to 0.000654, coverage 0.9398 to 0.9709, SE ratio 1.0166 | reported |
| `shared_target` | `target_g1_treat_all__ltmle` | diagnostic | parametric g-formula beside the targeted estimator: law G1, mean under treatment at all three times: `LTMLE` with GLM nuisances on the same draws, which targets the same parameter | reported only | bias -0.0018 to 0.0015, coverage 0.9320 to 0.9653, SE ratio 1.0005 | reported |
| `shared_target` | `target_g1_treat_all__nice` | diagnostic | parametric g-formula beside the targeted estimator: law G1, mean under treatment at all three times: the parametric g-formula with main-terms models | reported only | bias -0.0018 to 0.000915, coverage 0.9301 to 0.9639, SE ratio 1.0022 | reported |
| `shared_target` | `target_g2_contrast__ltmle` | diagnostic | parametric g-formula beside the targeted estimator: law G2, treatment at all three times against none: `LTMLE` with GLM nuisances on the same draws, which targets the same parameter | reported only | bias -0.0298 to -0.0050, coverage 0.8866 to 0.9300, SE ratio 0.8754 | reported |
| `shared_target` | `target_g2_contrast__nice` | diagnostic | parametric g-formula beside the targeted estimator: law G2, treatment at all three times against none: the parametric g-formula with main-terms models | reported only | bias -0.0066 to 0.0108, coverage 0.9220 to 0.9577, SE ratio 0.9908 | reported |
| `shared_target` | `target_g2_treat_all__ltmle` | diagnostic | parametric g-formula beside the targeted estimator: law G2, mean under treatment at all three times: `LTMLE` with GLM nuisances on the same draws, which targets the same parameter | reported only | bias -0.0088 to 0.0066, coverage 0.9258 to 0.9606, SE ratio 0.9741 | reported |
| `shared_target` | `target_g2_treat_all__nice` | diagnostic | parametric g-formula beside the targeted estimator: law G2, mean under treatment at all three times: the parametric g-formula with main-terms models | reported only | bias -0.0053 to 0.0080, coverage 0.9154 to 0.9527, SE ratio 0.9662 | reported |
<!-- /generated -->

The study has 119 property cells. Of these, 96 carry a verdict and 23 report a value only. 95 of
the 96 pass, and one is red. `properties.csv` has 119 rows.

| family | cells | what it reads | control that must fail |
| --- | --- | --- | --- |
| `interval_calibration` | 21 labels at n = 2,000 and 2,400 replications on `G1` to `G5` | the SE ratio inside 0.93 to 1.07, the coverage inside 0.92 to 0.98, and the bias inside its margin | the standard error scaled by 0.7, and added noise of one pilot sampling SD |
| `root_n_and_efficiency` | four contrasts at n = 2,000, 4,000 and 8,000 | the bias, the coverage floor, and the SE ratio screen | none. The next family reads the rate |
| `root_n_rate` | the empirical SD and the reported SE of the same four contrasts | the slope of each against n, in the band for a root-n rate | none. The slope interval must exclude -0.25 |
| `monte_carlo_accounting` | seven cells at `n_draws = 1`, and four `mc_removed` cells | the interval that includes the simulation term passes the calibration rule | the `mc_removed` cells report the same estimate with the standard error of 500 histories, and carry no verdict |
| `ice_necessity` | three plans on `G2` at n = 32,000 | the NICE bias inside its margin | the main-terms ICE fit on the same draws, which must leave the margin |
| `ice_reported` | the paired ICE and NICE fits on `G1` | none. The cells report | none |
| `shared_target` | `LTMLE` and `IPSI` beside NICE on `G1` and `G2` | none. The cells report | none |
| `extrapolation_report` | the main-terms fit of `G4Q` at a fixed dose | none. The cell reports a bias of 1.49 | none |

**One red cell.** The cell `interval_calibration/nice_g1_treat_all__correctly_specified`
(n = 2,000, 2,400 replications) has an SE ratio of 0.9619. Its 99% interval, 0.9259 to 1.0000,
leaves the band 0.93 to 1.07. Its coverage is 0.9413, with a 99% interval of 0.9278 to 0.9529
inside the band 0.92 to 0.98. Its bias is 0.000158 with a standard error of 0.000393.

The study declares the reading of a red cell before the run. A calibration cell reads
finite-sample only when three conditions hold. Its 99% SE-ratio interval contains 1. Its coverage
is consistent with 0.95. The diagnosis finds no defect.

The study expects 1.26 such reds over 114 gated cells, and the chance of at least one is 0.72.
This cell meets the first two conditions. The witnesses below support the third.

| witness | SE ratio | replications |
| --- | --- | --- |
| the same law, plan and size in `shared_target/target_g1_treat_all__nice` | 1.0022 | 1,192 |
| the same law, plan and size in `ice_reported/pair_g1_treat_all__nice` | 0.9701 | 1,198 |
| the other `G1` calibration cells | 0.972 to 0.993 | 2,400 each |
| the `G1` contrast ladder at n = 2,000, 4,000 and 8,000 | 1.021, 0.986 and 1.006 | the ladder size |

The independent draws of the same law, plan and size read 1.0022 and 0.9701. Every other `G1`
calibration cell passes, and the ladder reads close to one. No defect shows, so the cell reads as
a finite-sample red. The roadmap owner `X45-finite-sample` holds it, and the
[red-cell ledger](red-cells.md) lists it.

**Fitting failures.** The property phase recorded ten fitting failures in `property-failures.csv`.
Two of 1,200 replicates were lost in the paired `G1` fit set, and eight of 1,200 in the shared
`G1` fit set. Each failure is a GLM with a singular information matrix. Both rates are under the
declared bound of 0.01.

A rerun of `paired_g1/293` and `shared_g1/248` fits `ParametricGFormula` without error. The
failing fit is the sequential outcome GLM of the comparator in the same fit set. The comparator is
`SequentialGComputation` with `method="ice"` in the paired set and `LTMLE` in the shared set. These
cells carry no verdict, so no gate reads the lost replicates. `property-failure-accounting.csv`
has no rows.

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications per scenario |
| `n` | 8000 | observations per primary replication |
| `independent_tests_total` | 21 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 21 | of those, passing |
| `property_cells_total` | 96 | repeated-sampling property cells |
| `property_cells_passed` | 95 | cells whose own and family verdicts pass |
| `property_cells_reported` | 23 | cells that report a value and carry no verdict |
| `max_standardized_bias` | 0.0782 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9360 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9088 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0865 | highest bootstrap primary SE-ratio endpoint |
| `subject_failures` | 0 | primary replicates of `cleverly` lost to a recorded fitting failure |
| `margin:subject_failure_bound` | 0.0100 | share of replicates one primary cell may lose |
| `margin:subject_failures_allowed` | 10 | replicates one primary cell may lose |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed |

## Limitations

| limit | what it means for use |
| --- | --- |
| the comparator runs are pending | no `pygformula` output is compared. A follow-up PR from X45 registers the runs, and this page is Python property evidence until then |
| correctly specified laws | every law but `G4Q` is the main-terms model that the estimator fits. The study does not show coverage under misspecification. The `G4Q` cell reports a bias of 1.49 |
| one red cell | the `G1` treat-all calibration cell reads as finite-sample under a declared reading. A repeat run under a fresh declaration would test the reading |
| GLM learners only | the estimator ships with `GLM` models. A flexible learner is outside the evidence |
| node-specific models | `pygformula` pools residual SD over time, so its fixtures use laws where pooled and node-specific models agree. The pending pairs do not cover node-specific laws |
| no published variance | no published analytic variance exists for the simulation form. The interval is a natural extension, and its calibration here is the evidence for it |
| lost replicates | ten property replicates were lost to comparator fits. They carry no verdict, so no gate reads them |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/parametric_gformula_study/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. It records the comparator as pending, with the sources of the stub runner and the image.
Run `python -m tests.canonical.parametric_gformula_study.regenerate` to regenerate the artifacts.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/parametric_gformula_study/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/parametric_gformula_study/properties.csv)
carry every published row.
