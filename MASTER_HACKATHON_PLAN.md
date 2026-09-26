# MASTER HACKATHON PLAN - Amazon ML Challenge 2026
## Business Entity Resolution

> **Provenance rule.** Every fact in sections A-F is quoted from `Dataset/README.md`,
> `Dataset/Documentation_template.md`, `Dataset/utils/validate_submission.py`, or
> measured directly from the TSV files. Anything else is tagged
> **UNKNOWN - NEEDS VERIFICATION**.

---

## A. Competition Summary

| Item | Value | Source |
|---|---|---|
| Task | Business Entity Resolution across 3 noisy sources | README |
| Source 1 role | Deduplicated reference source | README |
| Goal | For each S1 entity, find all matching S2/S3 records | README |
| Metric | F_0.5, macro-averaged per S1 entity | README |
| Scored file | `matching_results.tsv` only | README |
| Also required | `candidate_pairs.tsv` (not scored, audited) | README |
| Leaderboard | Public = test subset, Private = remainder, private decides | README |
| Deadline | 27 Sep 2026, 11:59 PM IST | user-supplied, not in any file |
| Submissions/day | **UNKNOWN - NEEDS VERIFICATION** | not in README |
| Team name | **UNKNOWN - NEEDS VERIFICATION** (needed for zip name) | - |

---

## B. Exact ML Task

Multi-label record linkage, not plain classification.

```
INPUT      3 TSVs: entity_id, business_name, business_address, country
PROCESS    normalise text -> block -> score candidate pairs
FEATURES   string similarity on name + address, country agreement
MODEL      binary pair classifier: P(S1_i matches S2/S3_j)
PREDICT    threshold probability -> keep or drop each candidate
SUBMIT     one row per S1 entity, comma-joined matched IDs (may be empty)
SCORE      F_0.5 per S1 entity, then mean over ALL S1 entities
```

The scoring unit is the **entity**, not the pair. An entity with 11 true matches
counts exactly as much as a singleton.

---

## C. Dataset Understanding (measured)

### Row counts

| File | Rows |
|---|---|
| train_source1.tsv | 2,206,821 |
| train_source2.tsv | 5,034,616 |
| train_source3.tsv | 5,285,603 |
| train_ground_truth.tsv | 2,206,821 |
| test_source1.tsv | 1,732,544 |
| test_source2.tsv | 4,887,273 |
| test_source3.tsv | 5,082,316 |

Ground truth is exactly 1:1 with train S1 - confirmed.

### Country distribution - the domain-shift trap

| Country | Train S1 | Test S1 |
|---|---|---|
| US | 1,323,633 | 663,106 |
| India | 883,188 | 809,986 |
| **France** | **0** | **259,452** |

**15% of the test set is a country with zero training examples.** France entities must
still appear in the submission. No `{US, India}` hard-coding; country logic must fall
back to a generic path.

### Match-count distribution (train)

| Matches | Entities | Share |
|---|---|---|
| 0 (singleton) | 123,247 | 5.6% |
| 1 | 119,157 | 5.4% |
| 2 | 375,212 | 17.0% |
| 3 | 530,841 | 24.1% |
| 4 | 484,115 | 21.9% |
| 5 | 321,957 | 14.6% |
| 6 | 164,868 | 7.5% |
| 7 | 63,968 | 2.9% |
| 8-11 | 23,456 | 1.1% |

Mean ~3.46 matches. Links: 3,693,619 S2 + 3,944,746 S3 = 7,638,365.

1. **Multi-match is the norm** - a top-1 matcher is structurally capped.
2. **Singletons are 5.6% of score mass** - each correct empty row earns a full 1.0.

### Observed noise

- Address token order varies: `1795 Westchester Drive, High Point, NC` vs
  `IA, Iowa City, 1064 Newton Rd, Unit 11` - token-set features beat positional parsing.
- Junk prefixes in names: `<< Team Ecole`.
- French addresses carry accents and region names - Unicode normalisation is mandatory.

### Scale check

Brute force on test = 1.73M x 9.97M ~= **1.7e13 pairs**. Infeasible. Blocking is not an
optimisation, it is the only way this runs, and it caps achievable recall.

---

## D. Evaluation Metric

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)    per entity, then macro-average
```

Precision weighted 2x over recall. README example: 3 predicted, 2 correct, both true
matches found -> P=0.667, R=1.0, F_0.5=**0.714**. Drop the wrong ID -> **1.0**.

**Operational rule: when a candidate is borderline, leave it out.**

---

## E. Submission Requirements

`output/matching_results.tsv` - tab-separated, header
`source1_entity_id<TAB>matched_entity_ids`

- Exactly one row per test S1 entity - all **1,732,544**
- Empty second field for singletons
- Comma-joined IDs, no quoting, no spaces
- S2-/S3- IDs only; no S1 self-matches; no duplicates within a list
- No duplicate `source1_entity_id` rows

`output/candidate_pairs.tsv` - same shape, column `candidate_entity_ids`. Must be the
**final** pre-model candidate set: exactly what the model scored, not an earlier
blocking pass. Matches must be a subset of candidates.

Final zip `<team_name>_submission.zip` contains `output/` (both TSVs),
`code/business_entity_resolution/` (src, README.md, requirements.txt), and the filled
`Documentation_template.md`.

Local gate before every upload: run `utils/validate_submission.py` with `--matching`,
`--candidate` and `--test-dir`. Exit 0 = safe. Per its docstring, `--check-ids` is
**off by default** because ID-existence checking on 1.7M entities costs several GB.
Run it once on the final file; drop `--candidate` if memory is tight.

---

## F. Rules and Constraints

**Prohibited - immediate disqualification:**
- Commercial entity-resolution APIs or services
- Government / business-registration lookups
- **Geocoding APIs for address normalisation**
- Any external internet data augmentation

**Model licence:** final model must be **MIT or Apache 2.0** and **<= 8B parameters**.
Offline Apache-2.0 sentence-transformer checkpoints satisfy the licence rule, but
weights must be pre-downloaded and used offline. A model whose licence you cannot name
is a model you cannot ship.

---

## G. Tool Stack

**Verified present:** Python 3.12, pandas, Camber Jupyter, dataset readable at
`/home/jovyan/projects/AIML_Amazon_Challenge/Dataset/`.

**Verified constraint:** the project directory is **read-only** from this kernel
(PermissionError on write). Working files go in the conversation folder; final
artefacts must be copied to Stash explicitly.

**UNKNOWN - NEEDS VERIFICATION (blocks any compute commitment):**
- Kernel RAM ceiling - decides whether 5M-row TF-IDF fits in memory
- GPU presence on this node - decides embeddings vs pure string features
- Whether rapidfuzz, lightgbm, xgboost, sentence-transformers are installed
- Camber job node sizes available on the free trial
- Whether offline model weights can be downloaded at all

Documented Camber sizes: XXSMALL 4C/16GB, SMALL 16C/64GB, LARGE 64C/256GB,
XSMALL 8C/32GB/1GPU, MEDIUM 48C/192GB/4GPU. Availability on a trial account is
**unverified**.

Default assumption until measured: **CPU-only, string-similarity features,
gradient-boosted trees** - the approach that cannot be blocked by a missing GPU.

---

## H. Model Families

| Candidate | Hypothesis | Cost | Risk | Verdict |
|---|---|---|---|---|
| **LightGBM on pair features** | Similarity features separate match/non-match; fast on millions of rows | Low | Low | **Primary** |
| XGBoost / CatBoost | Marginal gain, useful for a blend | Low | Low | Later if time |
| Logistic regression | Sanity baseline, calibrated probabilities | Trivial | Low | Baseline only |
| TF-IDF char n-gram + ANN | Blocking engine, sets recall ceiling | Medium | Memory | **Required** |
| Sentence-transformer embeddings | Semantic name matching; may help France | High | GPU + licence + offline weights | Only if GPU verified |
| Cross-encoder re-ranker | Best per-pair precision | Very high | Cannot score 10M+ pairs in 34h | **Rejected on time** |

Blocking: country-partitioned TF-IDF char n-gram on normalised name, top-K neighbours
per S1 entity, unioned with an address-token pass. K sets the recall ceiling and must be
measured against train ground truth before any modelling.

---

## I. Phase Roadmap

| # | Phase | Objective | Done when | Priority |
|---|---|---|---|---|
| 1 | Rules audit | Extract every constraint | Plan reviewed | DONE |
| 2 | Data audit | Row counts, countries, match distribution | Measured | DONE |
| 3 | Env probe | RAM, CPU, GPU, libraries | Numbers in hand | CRITICAL |
| 4 | Validation | Entity-level holdout + local F_0.5 scorer | Scorer reproduces README 0.714 | CRITICAL |
| 5 | Normalisation | Unicode, case, legal suffixes, address abbreviations | Applied uniformly | CRITICAL |
| 6 | Blocking | Candidate generation + measured recall ceiling | Recall >= 0.95 at manageable K | CRITICAL |
| 7 | Baseline | LightGBM on core features | First local F_0.5 recorded | CRITICAL |
| 8 | Threshold tuning | Optimise F_0.5, not F1 | Per-country thresholds swept | CRITICAL |
| 9 | Feature expansion | Address / token / rank features | Each change scored vs baseline | LATER |
| 10 | Full-test inference | Generate both TSVs | Validator exits 0 | CRITICAL |
| 11 | Submit + iterate | Upload, compare public vs local | Gap understood | CRITICAL |
| 12 | Package | Zip with code, README, requirements, methodology | Reproducible from scratch | CRITICAL |
| 13 | Embeddings | Semantic matching if GPU available | Beats baseline or dropped | OPTIONAL |

---

## J. 34-Hour Timeline (26 Sep 14:00 -> 27 Sep 23:59 IST)

| Window | Phase | Deliverable |
|---|---|---|
| 14:00-15:00 | Env probe + holdout split | RAM/GPU known, holdout frozen |
| 15:00-16:00 | Local F_0.5 scorer | Reproduces README 0.714 exactly |
| 16:00-18:00 | Normalisation + blocking v1 | Recall ceiling measured |
| 18:00-20:00 | Feature extraction | Pair feature matrix built |
| 20:00-22:00 | LightGBM baseline + threshold sweep | First honest local F_0.5 |
| 22:00-00:00 | Full-test inference dry run | Both TSVs, validator exit 0 |
| 00:00-01:00 | **First leaderboard submission** | Public score anchored early |
| 01:00-07:00 | Buffer / long jobs / rest | Blocking recall improvements |
| 07:00-11:00 | Feature expansion round 2 | Scored experiments only |
| 11:00-14:00 | Per-country threshold tuning | France path validated |
| 14:00-17:00 | Best-model full inference | Candidate final TSVs |
| 17:00-19:00 | Validate + submit | Second/third entry |
| 19:00-21:00 | Package zip + methodology | Reproducible bundle |
| 21:00-22:00 | **Final submission** | Uploaded and confirmed |
| 22:00-23:59 | Contingency only | No new features |

Hard rule: **final upload by 22:00, not 23:59.**

---

## K. Experiment Log Format

| Exp | Blocking | Features | Model | Threshold | Local F_0.5 | Public | Runtime | Decision |
|---|---|---|---|---|---|---|---|---|
| 000 | - | - | predict-all-empty | - | pending | - | - | floor reference |

The all-empty baseline is worth measuring first. On the train distribution it scores
about 0.056 (the singleton share) and is the floor any real model must beat.
Every row needs a stated hypothesis before it is run.

---

## L. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| **France domain shift (15% of test)** | Large silent score loss | Country-agnostic features; simulate by stripping country on a holdout slice |
| **Blocking recall ceiling** | Caps final score regardless of model | Measure recall on train GT before modelling |
| **Optimising F1 instead of F_0.5** | Systematic over-prediction | Scorer implements F_0.5 only |
| **Memory blowup on 5M-row TF-IDF** | Kernel death, hours lost | Per-country processing, sparse matrices, chunked inference |
| **Read-only project dir** | Silent write failures | Confirmed; writes go to conversation folder |
| **Format rejection** | Wasted submission slot | Run validator every time |
| **External lookup temptation** | Disqualification | Hard prohibition recorded |
| **Unknown submission limit** | Wasted attempts | Verify on Unstop before first upload |
| **Final upload at 23:59** | Total loss | Deadline moved to 22:00 |

---

## M. Final Submission Checklist

- [ ] 1,732,544 rows in matching_results.tsv, one per test S1 entity
- [ ] Header exactly source1_entity_id TAB matched_entity_ids
- [ ] Tab-separated; no stray quoting; no pandas index column
- [ ] Empty field for singletons, not nan or []
- [ ] Only S2-/S3- IDs; no S1; no duplicates in any list
- [ ] No duplicate source1_entity_id
- [ ] Every France entity present
- [ ] candidate_pairs.tsv present; matches are a subset
- [ ] validate_submission.py exits 0, including one --check-ids run
- [ ] Zip structure correct; requirements.txt pinned
- [ ] Documentation_template.md filled: methodology, blocking, model, features
- [ ] Model licence MIT/Apache-2.0, <= 8B params, stated in the doc
- [ ] No external data or API anywhere in the pipeline
- [ ] Submitted by 22:00 IST 27 Sep

---

## N. Do This Right Now

1. **Probe the environment** - RAM, CPU count, GPU, installed libraries. Every UNKNOWN
   in section G blocks the compute plan.
2. **Confirm two facts on Unstop** that no provided file contains:
   - submissions allowed per day
   - exact team name for the zip filename
3. **Build the F_0.5 scorer and freeze an entity-level holdout.** No modelling until the
   scorer reproduces the README example (0.714) exactly. Without a trustworthy local
   metric, every later decision is guesswork.

---

## Live Project Status

- **Phase:** 2 complete (data audit), 3 next (env probe)
- **Status:** IN PROGRESS
- **Best Local F_0.5:** none yet
- **Best Public:** none yet
- **Best Model:** none yet
- **Current Experiment:** none
- **Next Action:** environment probe
- **Biggest Risk:** France domain shift, 259,452 test entities with no training analogue

---

## O. Phase 3 RESOLVED - Measured Environment (supersedes section G UNKNOWNs)

Measured on this kernel, 26 Sep 2026.

| Resource | Value |
|---|---|
| Python | 3.12.9 |
| CPU cores (logical = affinity) | 16 |
| RAM total | 61.4 GB |
| RAM available | 58.3 GB |
| Disk free on /home/jovyan | 1,236.7 GB |
| **GPU** | **none - no nvidia-smi, CPU only** |

This is a **SMALL-class node (16C/64GB)**. No GPU.

### Library status

Pre-installed: numpy 2.2.6, pandas 2.3.1, scipy 1.16.0, scikit-learn 1.7.1,
pyarrow 21.0.0, networkx 3.5.

Installed in Phase 3b (pip returncode 0, all imports verified):
lightgbm 4.7.0, rapidfuzz 3.14.6, jellyfish, optuna 5.0.0.

Still absent: torch, sentence-transformers, faiss, xgboost, catboost, recordlinkage.

### Decisions forced by these numbers

1. **CPU-only string-similarity + LightGBM is the plan, not a fallback.** No GPU means
   no transformer embeddings on 10M records in the time available. Section H's
   "sentence-transformer" row is now **rejected on hardware**, not merely deferred.
2. **58 GB RAM against ~2.4 GB of TSV is comfortable** for per-country sparse TF-IDF.
   Blocking must still be chunked - the danger is the candidate-pair explosion, not
   loading the sources.
3. **1.2 TB disk** removes any need to compress intermediates. Write Parquet freely.
4. **16 cores** - set `n_jobs=16` / `num_threads=16`; rapidfuzz `cdist` parallelises.
5. **Reinstall risk:** these four packages live in the kernel session. A kernel restart
   may lose them; `requirements.txt` must pin all four for the submission zip.

---

## P. Competition Facts Confirmed by Team (26 Sep 2026)

| Item | Value | Source |
|---|---|---|
| Team name | **Zero** | user-confirmed |
| Submissions per day | **5** | user-confirmed |
| Zip filename | `Zero_submission.zip` | derived from README pattern |

Section A's two UNKNOWNs are now closed.

### Submission budget - 5 per day

Two calendar days remain (26 and 27 Sep IST), so **at most 10 uploads**, and realistically
5 on the 27th. Budget explicitly:

| Slot | Purpose |
|---|---|
| 26 Sep #1 | Format smoke test - validated pipeline output, anchors a public score early |
| 26 Sep #2-3 | Blocking / threshold variants if time allows |
| 27 Sep #1-2 | Best tuned model |
| 27 Sep #3 | Final chosen submission, uploaded by 22:00 |
| 27 Sep #4-5 | Held in reserve for a rejected-format emergency |

Never spend a slot on a file that has not passed `validate_submission.py` locally.

### Live status after Phase 3

- **Phase:** 3 complete (environment probed, libraries installed), 4 next (F_0.5 scorer + holdout)
- **Hardware:** 16 CPU / 61.4 GB RAM / no GPU / 1.2 TB disk
- **Best Local F_0.5:** none yet
- **Best Public:** none yet
- **Next Action:** build F_0.5 scorer, validate against README worked example (0.714)
- **Biggest Risk:** France domain shift, 259,452 test entities with no training analogue
