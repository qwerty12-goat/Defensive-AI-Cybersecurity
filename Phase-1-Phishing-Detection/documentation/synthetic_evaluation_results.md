# Synthetic AI Phishing-Class Evaluation Results

## Overview

A frozen set of 500 controlled AI-generated phishing-class messages was evaluated with the project's two trained classifiers:

- TF-IDF + Logistic Regression baseline
- DistilBERT

The synthetic dataset was finalized before either classifier was evaluated on it. It contains 100 messages in each of five communication categories and an equal 250/250 split between Copilot and Gemini generation sources.

Because every synthetic sample belongs to the positive phishing class, the primary metric for this evaluation is detection rate/recall. Precision and false-positive rate are not estimated from this positive-only set.

## Traditional Validation vs. Synthetic Evaluation

| Model | Traditional validation phishing recall | Synthetic detection rate |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | 98.37% | 55.40% |
| DistilBERT | 99.09% | 36.20% |

Both models detected substantially fewer messages on the controlled synthetic evaluation set than on the traditional validation distribution.

The ordering also reversed. DistilBERT had higher recall on the traditional validation set, while the baseline detected more messages on the synthetic set.

This result is evidence of a robustness limitation under this particular synthetic distribution shift. It does not establish that either model will perform this way on all AI-generated phishing.

## Overall Synthetic Results

| Model | Detected | Missed | Detection rate | 95% Wilson CI |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 277/500 | 223/500 | 55.4% | 51.0%-59.7% |
| DistilBERT | 181/500 | 319/500 | 36.2% | 32.1%-40.5% |

The baseline detected 96 more messages than DistilBERT.

## Results by Communication Category

| Category | Baseline | DistilBERT |
| --- | ---: | ---: |
| Account/security | 99/100 (99%) | 82/100 (82%) |
| Delivery/service | 55/100 (55%) | 28/100 (28%) |
| General social engineering | 20/100 (20%) | 2/100 (2%) |
| Promotional/offer | 84/100 (84%) | 59/100 (59%) |
| Workplace/business | 19/100 (19%) | 10/100 (10%) |

Category was strongly associated with detection performance in this experiment.

Account/security-style messages were detected most frequently by both models. Workplace/business and general social-engineering messages were among the most difficult.

The category-specific 95% Wilson intervals were:

| Category | Baseline 95% CI | DistilBERT 95% CI |
| --- | ---: | ---: |
| Account/security | 94.6%-99.8% | 73.3%-88.3% |
| Delivery/service | 45.2%-64.4% | 20.1%-37.5% |
| General social engineering | 13.3%-28.9% | 0.6%-7.0% |
| Promotional/offer | 75.6%-89.9% | 49.2%-68.1% |
| Workplace/business | 12.5%-27.8% | 5.5%-17.4% |

## Results by Generation Source

| Generation source | Baseline | DistilBERT |
| --- | ---: | ---: |
| Copilot | 132/250 (52.8%) | 81/250 (32.4%) |
| Gemini | 145/250 (58.0%) | 100/250 (40.0%) |

Wilson intervals:

| Generation source | Baseline 95% CI | DistilBERT 95% CI |
| --- | ---: | ---: |
| Copilot | 46.6%-58.9% | 26.9%-38.4% |
| Gemini | 51.8%-64.0% | 34.1%-46.2% |

Both classifiers had somewhat higher detection rates on the Gemini-generated subset, but reduced detection was present for both generation sources. The experiment therefore does not indicate that the robustness weakness is isolated to one generator.

## Prediction Overlap

The two classifiers were evaluated on the exact same 500 messages.

| Paired outcome | Messages | Percent |
| --- | ---: | ---: |
| Both detected | 168 | 33.6% |
| Both missed | 210 | 42.0% |
| Baseline only detected | 109 | 21.8% |
| DistilBERT only detected | 13 | 2.6% |

DistilBERT was not completely redundant with the baseline because it uniquely detected 13 messages. However, the baseline uniquely detected 109.

By category:

| Category | Both detected | Both missed | Baseline only | DistilBERT only |
| --- | ---: | ---: | ---: | ---: |
| Account/security | 81 | 0 | 18 | 1 |
| Delivery/service | 23 | 40 | 32 | 5 |
| General social engineering | 1 | 79 | 19 | 1 |
| Promotional/offer | 56 | 13 | 28 | 3 |
| Workplace/business | 7 | 78 | 12 | 3 |

## Paired Statistical Comparison

Because both classifiers evaluated the same messages, an exact two-sided McNemar test was used.

- Both correct: 168
- Baseline correct, DistilBERT wrong: 109
- Baseline wrong, DistilBERT correct: 13
- Both wrong: 210
- Discordant pairs: 122
- Continuity-corrected chi-square: 73.9754
- Exact two-sided p-value: 4.67819866022e-20
- Reportable result: p < 0.001

Within this controlled synthetic evaluation set, the paired difference in detection performance is statistically significant.

This should not be generalized into a claim that logistic regression is universally superior to DistilBERT. The traditional validation results showed the opposite ordering.

## Message-Length Analysis

The synthetic messages were short overall:

- Median length: 193 characters
- Median length: 26 words

Detected and missed messages had very similar lengths.

| Model | Detected median | Missed median |
| --- | ---: | ---: |
| Baseline | 189 characters / 25 words | 197 / 26 |
| DistilBERT | 186 characters / 25 words | 196 / 26 |

The overlap groups were also similar:

| Outcome | Samples | Median characters | Median words |
| --- | ---: | ---: | ---: |
| Baseline only | 109 | 193 | 25 |
| Both detected | 168 | 186 | 25 |
| Both missed | 210 | 197 | 26 |
| DistilBERT only | 13 | 193 | 26 |

Simple message length therefore does not appear to explain the large synthetic detection gap.

These messages are also far shorter than the context-window sizes investigated during the earlier DistilBERT robustness analysis, making truncation an unlikely explanation for this synthetic-set result.

## Language-Pattern Analysis

Document-frequency analysis showed similar associations for both models.

Words more associated with detection included terms such as:

- account
- security
- profile
- alert
- password
- login
- promotional
- activity

Words more associated with misses included terms such as:

- schedule
- project
- meeting
- workspace
- team
- document
- internal
- planning
- revised
- updated

This pattern is consistent with the category-level results. Messages containing recognizable account/security or promotional language were detected more often, while routine organizational and workplace-style language was associated with misses.

These are associations within the synthetic dataset. They should not be interpreted as proof that any individual word caused a prediction.

The word `fictional` also appeared in the DistilBERT miss analysis. Because the dataset was generated under controlled safety constraints, this may reflect the synthetic-generation procedure rather than a real-world phishing characteristic.

## Prediction Confidence

### Baseline

Detected messages:

- Mean confidence: 73.86%
- Median confidence: 73.93%
- At least 90% confident: 46/277 (16.6%)
- At least 99% confident: 2/277 (0.7%)

Missed messages:

- Mean confidence: 77.31%
- Median confidence: 78.20%
- At least 90% confident: 51/223 (22.9%)
- At least 99% confident: 0/223

### DistilBERT

Detected messages:

- Mean confidence: 94.80%
- Median confidence: 99.41%
- At least 90% confident: 152/181 (84.0%)
- At least 99% confident: 106/181 (58.6%)

Missed messages:

- Mean confidence: 97.04%
- Median confidence: 99.92%
- At least 90% confident: 290/319 (90.9%)
- At least 99% confident: 233/319 (73.0%)

The DistilBERT result is especially important. Most synthetic messages that it missed were not borderline classifications. The model assigned the legitimate class with high confidence to many phishing-class samples.

## High-Confidence Misses by Category

A high-confidence miss was defined as a phishing-class message predicted legitimate with at least 90% confidence.

### Baseline

| Category | Total misses | High-confidence misses | Percent of misses |
| --- | ---: | ---: | ---: |
| Account/security | 1 | 0 | 0.0% |
| Delivery/service | 45 | 2 | 4.4% |
| General social engineering | 80 | 17 | 21.2% |
| Promotional/offer | 16 | 0 | 0.0% |
| Workplace/business | 81 | 32 | 39.5% |

### DistilBERT

| Category | Total misses | High-confidence misses | Percent of misses |
| --- | ---: | ---: | ---: |
| Account/security | 18 | 13 | 72.2% |
| Delivery/service | 72 | 62 | 86.1% |
| General social engineering | 98 | 95 | 96.9% |
| Promotional/offer | 41 | 34 | 82.9% |
| Workplace/business | 90 | 86 | 95.6% |

For DistilBERT, high-confidence misses were especially concentrated in general social-engineering and workplace/business messages.

## High-Confidence Misses by Generator

| Model | Copilot | Gemini |
| --- | ---: | ---: |
| Baseline | 26/118 (22.0%) | 25/105 (23.8%) |
| DistilBERT | 157/169 (92.9%) | 133/150 (88.7%) |

High-confidence DistilBERT failures occurred extensively for both generators.

## Main Finding

The advanced DistilBERT classifier achieved better performance than the TF-IDF + Logistic Regression baseline on the traditional validation distribution, including higher phishing recall and a lower false-positive rate.

That advantage did not persist on the frozen controlled synthetic evaluation set. Both models experienced large reductions in phishing detection, and the simpler baseline detected substantially more synthetic messages than DistilBERT.

The robustness analysis suggests that this result is associated more strongly with communication style and distribution shift than with simple message length. DistilBERT's failures were particularly notable because most were high-confidence legitimate predictions.

This demonstrates that strong in-distribution validation performance alone is not sufficient evidence of robustness to a substantially different synthetic email distribution.

## Limitations

These findings apply to this experiment and should not be interpreted as universal measurements of AI-generated phishing detection.

Important limitations include:

- only 500 synthetic messages were evaluated;
- all 500 messages belong to the phishing class;
- the messages were generated under safety constraints and are intentionally non-operational;
- only two generation sources were used;
- the five categories are controlled experimental groupings;
- the historical training corpus and synthetic dataset differ in time period, source, and writing process;
- category and generator comparisons are descriptive within this dataset;
- word-frequency associations are not causal explanations;
- the synthetic set was used for evaluation, not model tuning.

The frozen synthetic dataset should remain unchanged. If future model development uses these results for tuning or model selection, a new independent synthetic holdout should be created for unbiased final evaluation.
