# AIML Amazon Challenge 2026 - Team Zero

Business Entity Resolution for the Amazon ML Challenge 2026.

## Problem

Given business records from 3 independent sources with noisy, inconsistent name and
address fields, identify which Source 2 / Source 3 records refer to the same real-world
business as each Source 1 entity. Source 1 is the deduplicated reference source; an
entity may match zero, one, or many records.

## Metric

`F_0.5 = (1.25 * P * R) / (0.25 * P + R)`, computed per Source 1 entity then
macro-averaged over all entities. Precision is weighted 2x over recall, so a borderline
candidate is better left out. Correctly predicting an empty list for a singleton scores
a full 1.0.

## Data

Datasets are **not** stored in this repository. They live in Camber Stash at
`stash://jatinteam97222687/projects/AIML_Amazon_Challenge/Dataset/` and are excluded by
`.gitignore`.

Measured sizes:

| File | Rows |
|---|---|
| train_source1.tsv | 2,206,821 |
| train_source2.tsv | 5,034,616 |
| train_source3.tsv | 5,285,603 |
| train_ground_truth.tsv | 2,206,821 |
| test_source1.tsv | 1,732,544 |
| test_source2.tsv | 4,887,273 |
| test_source3.tsv | 5,082,316 |

Test Source 1 contains 259,452 `France` entities; the training data contains none.

## Environment

Camber Jupyter, 16 CPU / 61.4 GB RAM / no GPU. Python 3.12.9.
Approach is CPU-only: string-similarity features plus gradient-boosted trees.

## Planning

See `MASTER_HACKATHON_PLAN.md` for the full competition analysis, phase roadmap,
timeline, risk register, and submission checklist.

## Rules observed

- No external databases, APIs, geocoding services, or internet data augmentation
- Final model must be MIT / Apache-2.0 licensed and <= 8B parameters
- Competition data is never committed to version control
