# PHASES - Amazon ML Challenge 2026 (Team Zero)

Business Entity Resolution. Status as of 26 Sep 2026, ~23:30 IST.
Legend: DONE | **IN PROGRESS** | TODO | OPTIONAL

| # | Phase | Objective | Definition of Done | Priority | Status |
|---|---|---|---|---|---|
| 1 | Rules & competition audit | Extract every rule, metric, format and constraint from the provided files | Plan written, nothing invented | CRITICAL | DONE |
| 2 | Data audit | Row counts, country split, match-count distribution, noise patterns | All 7 TSVs measured | CRITICAL | DONE |
| 3 | Environment probe | RAM, CPU, GPU, library availability | Numbers measured, libs installed | CRITICAL | DONE |
| 3b | Hardware upgrade | Move to a node that can hold the candidate set | LARGE 64C/246GB confirmed | CRITICAL | DONE |
| 4 | F_0.5 scorer | Trustworthy local metric | Reproduces README example 0.7143 | CRITICAL | DONE |
| 5 | Frozen holdout | Entity-level stratified split + all-empty floor | Split frozen to Parquet, floor measured | CRITICAL | DONE |
| 6 | Normalisation | Unicode/case/legal-suffix/address abbreviation cleanup | Applied identically to train and test | CRITICAL | DONE |
| 7 | Blocking / candidate generation | Reduce 1.7e13 pairs to a scoreable set | Recall measured: India 0.9440 / US 0.9844 at ~84 cand/entity | CRITICAL | DONE |
| 8 | Pair features | Name + address + country similarity matrix | 33 features on 8.32M labeled pairs | CRITICAL | DONE |
| 9 | LightGBM baseline | First honest local F_0.5 | **0.9315** at global th=0.70 vs 0.0558 floor | CRITICAL | DONE |
| 10 | Threshold tuning | Optimise F_0.5 (not F1), per country | India 0.675 / US 0.725 -> **0.9317** | CRITICAL | DONE |
| 11 | Full-test inference | Generate both submission TSVs | 37/37 shards, 140.9M pairs, validator **exit 0** | CRITICAL | DONE |
| 12 | Submit & iterate | Anchor a public score, compare vs local | Public/local gap understood | CRITICAL | **IN PROGRESS** |
| 13 | Final package | Zero_submission.zip + methodology | Reproducible from scratch | CRITICAL | TODO |
| 14 | Feature expansion | Extra token/rank/address features | Each change scored vs baseline | LATER | TODO |
| 15 | Ensembling | Blend / stack pair scorers | Beats single model on holdout | OPTIONAL | TODO |
| 16 | Embeddings | Semantic name matching | Rejected - no GPU on this node | OPTIONAL | REJECTED |

## Where we are

**Phase 12 - leaderboard submission.** Phases 1-11 are complete and verified.

- Local **F_0.5 = 0.9317** on the frozen 40k-entity holdout, graded against full ground
  truth including blocking misses. Floor (all-empty) 0.0558; blocking-imposed ceiling
  ~0.984, so the model captures ~95% of the reachable range.
- Thresholds locked: India 0.675, US 0.725, France 0.700 (uncalibrated global optimum -
  no French ground truth exists to tune against).
- Full test scored: **37/37 shards, 140,864,009 candidate pairs, 1,732,544 entities.**
- `matching_results.tsv` (92 MB) and `candidate_pairs.tsv` (1.8 GB) both pass the
  official `validate_submission.py` with **exit code 0**.

Phase 16 stays rejected on hardware: no GPU, so transformer embeddings over ~10M records
are not finishable in the time available.

**What actually stalled earlier:** not Phase 8 (which took 1.6 min) but Phase 11
inference, which was loading all three countries' 10.3M-row pool before filtering and
rebuilding the index on every retry. Fixed with Parquet predicate pushdown plus
resumable shard writes.

**Biggest remaining lever:** India blocking recall 0.9393 vs US 0.9841. India's 6.6-pt
F_0.5 deficit is a retrieval problem, not a classifier problem - a candidate that was
never retrieved cannot be predicted.

## Gate rules

- No modelling until the holdout is frozen (Phase 5) - otherwise every score is guesswork.
- No submission until `utils/validate_submission.py` exits 0. Budget is 5 uploads/day.
- Blocking recall (Phase 7) caps the final score regardless of model quality; measure it
  before touching LightGBM.

## Key constraints carried through every phase

- Metric is F_0.5 macro-averaged **per entity** - precision weighted 2x, so drop borderline candidates.
- 259,452 test entities are `France`, a country with **zero** training examples.
- Singletons are 5.6% of score mass; a correct empty row is worth a full 1.0.
- No external data, APIs, or geocoding services. Final model must be MIT/Apache-2.0 and <= 8B params.
