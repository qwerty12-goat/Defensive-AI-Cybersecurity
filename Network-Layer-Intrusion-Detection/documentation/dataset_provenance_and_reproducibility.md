# Network Layer Dataset Provenance and Reproducibility Plan

## Research Scope

The Network Layer investigates the following primary research question:

> How accurately can machine-learning models distinguish malicious from benign network traffic while maintaining a low false-positive rate, and how robust are those models when evaluated under network-traffic distribution shift?

The work is limited to defensive classification of pre-recorded public network-flow data. It does not involve scanning, attacking, probing, or generating malicious traffic against real systems.

## Primary Dataset

**Dataset:** UNSW-NB15  
**Creator:** UNSW Canberra Cyber Range Lab  
**Official source:** https://research.unsw.edu.au/projects/unsw-nb15-dataset  
**Original publication:** Nour Moustafa and Jill Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," MilCIS, 2015. DOI: 10.1109/MilCIS.2015.7348942

According to the official dataset documentation, UNSW-NB15 was created using the IXIA PerfectStorm tool in the UNSW Canberra Cyber Range Lab. The dataset combines normal network activity with synthetic attack behavior. Approximately 100 GB of raw traffic was captured with tcpdump. Argus, Bro-IDS, and additional feature-generation algorithms were used to derive network-flow features.

The complete dataset contains 2,540,044 records stored across four primary CSV files. The official documentation describes 49 features with the class label.

Nine attack categories are represented:

- Fuzzers
- Analysis
- Backdoors
- DoS
- Exploits
- Generic
- Reconnaissance
- Shellcode
- Worms

The dataset also includes normal traffic.

## Official Modeling Partitions

The initial Network Layer experiment will use the official preconfigured modeling partitions:

- `UNSW_NB15_training-set.csv`: 175,341 records
- `UNSW_NB15_testing-set.csv`: 82,332 records

The official testing partition will remain isolated from model development.

A validation partition will be derived only from the official training data. The exact validation procedure, random seed, class distributions, duplicate checks, and any grouping or leakage controls will be documented before model training.

## Labels

The primary task is binary intrusion detection:

- `label = 0`: normal
- `label = 1`: attack

The `attack_cat` field identifies attack categories and will be retained for evaluation and error analysis. It will **not** be used as a predictive input for the binary classifier because it directly describes the target class and would introduce target leakage.

## Untouched-Test Policy

The official test partition is reserved for one final evaluation after:

1. preprocessing decisions are frozen;
2. feature-selection and leakage decisions are frozen;
3. candidate models are trained and compared using training/validation data;
4. hyperparameters and decision thresholds are frozen;
5. robustness experiments used for model-development decisions are complete;
6. the final evaluation protocol is documented.

After the official test partition is consumed, it will be treated permanently as consumed and will not be used for subsequent tuning, model selection, threshold adjustment, calibration fitting, or feature selection.

Any later model improvements will require a new independent holdout for unbiased final evaluation.

## Raw-Data Preservation

Original downloaded dataset files will be preserved unchanged under the local raw-data directory and excluded from ordinary source-control commits when their size makes Git storage inappropriate.

Before processing, the project will record for every downloaded source file:

- exact filename;
- byte size;
- SHA-256 hash;
- source URL;
- download date.

Cleaning and transformation will be performed through reproducible scripts rather than manual spreadsheet editing.

## Reproducibility Policy

The Network Layer will record the computational environment before the first model is trained.

At minimum, the project will preserve:

- operating system;
- Python version;
- exact package versions;
- scikit-learn version;
- model-library versions used later;
- CPU/GPU information where relevant;
- random seeds;
- preprocessing configuration;
- model hyperparameters;
- dataset hashes;
- trained-model hashes;
- evaluation-script configuration.

A dependency snapshot will be captured before modeling rather than reconstructed after the experiments.

## Seed Policy

A fixed seed of **42** will be used where deterministic randomization is required unless a later experiment explicitly studies seed sensitivity.

Every script that performs randomized splitting, sampling, or model initialization must set and document its seed.

A fixed seed improves repeatability but does not guarantee bit-for-bit reproducibility across different libraries, hardware, operating systems, or parallel implementations.

## Leakage Audit

Before model training, the project will explicitly investigate:

- exact duplicate records;
- duplicates or highly related records across partitions;
- target-derived columns;
- identifiers that may encode collection artifacts;
- attack-category leakage;
- time or session information that could make the task artificially easy;
- categorical features with suspiciously direct relationships to the target;
- preprocessing fitted outside the training partition.

Any feature removed for leakage reasons will be documented with the reason for removal.

## Evaluation Metrics

Primary binary-classification metrics will include:

- accuracy;
- precision;
- recall;
- F1 score;
- false-positive rate;
- false-negative rate;
- confusion matrix.

Attack-category-specific detection performance will also be reported where sample sizes permit meaningful interpretation.

Because intrusion-detection systems can generate operational burden through false alarms, false-positive rate will be treated as a primary metric rather than relying on accuracy alone.

## Distribution-Shift Evaluation

The official test set will provide the primary held-out evaluation for the UNSW-NB15 modeling protocol.

A separate robustness experiment will later evaluate the frozen models under a meaningfully different traffic distribution. The exact external dataset or shift protocol will be selected only after the primary dataset has been inspected and the leakage structure is understood.

Performance on such a dataset will be described as distribution-shift or external robustness evidence and will not automatically be generalized to real-world deployment.

## Known Scope Limitation

UNSW-NB15 was generated in a controlled cyber-range environment rather than collected as an unrestricted sample of modern production networks. Therefore, performance on UNSW-NB15 alone cannot establish real-world deployment performance.

This limitation is part of the motivation for the planned distribution-shift evaluation.

## Citation

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). *2015 Military Communications and Information Systems Conference (MilCIS)*. IEEE. https://doi.org/10.1109/MILCIS.2015.7348942

## Record Status

This document was created before Network Layer dataset processing or model training. File-specific hashes, exact environment versions, and preprocessing decisions will be appended only after they are observed directly.
