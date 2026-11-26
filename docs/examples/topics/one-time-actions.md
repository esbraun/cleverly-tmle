# One-time actions

These tutorials estimate what a treatment, offer or practice given once did to an outcome measured
later. They cover a randomized pilot, the gain over current practice, and several practices at once.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| the treatment was assigned at random with a recorded chance, as in a pilot | [a randomized pilot](../randomized-pilot.ipynb) | because the chance of the offer differed by risk group, a simple comparison of offered and other patients is confounded even though the offer was random (here -0.006 against a true 0.142). Replacing the recorded chances with a fitted model of who got the treatment gives up the one model known to be right; when that fitted model and the outcome model are both wrong, the estimate is biased |
| the gain from offering everyone over current practice, or whether the offer still helps when a later step, such as a doctor's visit within 7 days, is guaranteed for everyone | [gain over current practice](../population-impact.ipynb) | the usual average effect compares offering everyone with offering no one, so it also counts the gain current practice has already collected and overstates what offering everyone would add |
| several practices or treatments, each chosen for some patients, and you want to know which of them change the outcome | [which discharge practices matter](../ranking-exposures.ipynb) | when one candidate causes another, as a navigator books the follow-up visit, ranking each one with the others held as they were leaves out the part of its effect that runs through the other, and here makes navigation look weak. A simple comparison also mixes each practice's effect with who received it: patients referred to home health stay home less often though the referral does nothing |

## See also

- [A small pilot with outside records: adaptive TMLE](../trial-with-external-data.ipynb) combines a small randomized pilot with outside records of patients who were not randomized.

```{toctree}
:hidden:
:maxdepth: 1

A randomized pilot </examples/randomized-pilot>
Gain over current practice </examples/population-impact>
Which discharge practices matter </examples/ranking-exposures>
```
