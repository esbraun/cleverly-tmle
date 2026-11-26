# Missing, sampled or combined data

These tutorials handle data that a textbook analysis assumes away. Outcomes go missing, a sample is
drawn with unequal chances, or a small randomized trial is combined with outside records that differ
from it.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| many outcomes are missing, for example unanswered surveys | [missing survey answers](../survey-nonresponse.ipynb) | analyzing only those who answered estimates the effect for people like the respondents, who differ from the whole group |
| the outcome was measured on a sample drawn with known, unequal chances, such as an interview that picks high-risk patients more often | [an interview of a sample of patients](../sampling-weights.ipynb) | an analysis without the sampling weights estimates the effect among the sampled patients, who differ from everyone the sample stands for |
| a small randomized trial, plus outside records on the same outcome from patients who were not randomized | [a small pilot with outside records](../trial-with-external-data.ipynb) | in the records, sicker patients got the offer more often, so the simple pooled difference is off (7.58 against a true 6.05). Learning how the records differ from the pilot, a fixed offset, and removing it narrowed the interval, and the interval is valid only for the effect within the description of the difference that the fit picked |

```{toctree}
:hidden:
:maxdepth: 1

Missing survey answers </examples/survey-nonresponse>
An interview of a sample of patients </examples/sampling-weights>
A small pilot with outside records </examples/trial-with-external-data>
```
