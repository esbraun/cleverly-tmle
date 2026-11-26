# Start here

Begin here. The first page runs a whole analysis on simulated data and explains each step that the
other tutorials reuse. The second runs one on real records, where no true answer is known.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| a treatment given once and an outcome measured later | [one treatment, one outcome](../point-treatment-tmle.ipynb) | a simple comparison of treated and untreated patients mixes the treatment's effect with who received it. The page also shows what happens to the estimate when one or both models are wrong |
| a full analysis of real records, where no true answer is known | [smoking and twins' birth weight](../twins-causal-inference.ipynb) | adjusting for a variable that the treatment itself changes, here gestational age, removes part of the effect |

```{toctree}
:hidden:
:maxdepth: 1

One treatment, one outcome </examples/point-treatment-tmle>
Smoking and twins' birth weight </examples/twins-causal-inference>
```
