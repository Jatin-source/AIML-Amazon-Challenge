# PHASES - Amazon ML Challenge 2026 (Team Zero)

Business Entity Resolution. Status as of 26 Sep 2026, ~18:15 IST.
Legend: DONE | **IN PROGRESS** | TODO | OPTIONAL

| # | Phase | Objective | Definition of Done | Priority | Status |
|---|---|---|---|---|---|
| 1 | Rules & competition audit | Extract every rule, metric, format and constraint from the provided files | Plan written, nothing invented | CRITICAL | DONE |
| 2 | Data audit | Row counts, country split, match-count distribution, noise patterns | All 7 TSVs measured | CRITICAL | DONE |
| 3 | Environment probe | RAM, CPU, GPU, library availability | Numbers measured, libs installed | CRITICAL | DONE |
| 3b | Hardware upgrade | Move to a node that can hold the candidate set | LARGE 64C/246GB confirmed | CRITICAL | DONE |
| 4 | F_0.5 scorer | Trustworthy local metric | Reproduces README example 0.7143 | CRITICAL | DONE |
| 5 | Frozen holdout | Entity-level stratified split + all-empty floor | Split frozen to Parquet, floor measured | CRITICAL | DONE |
| 6 | Normalisation | Unicode/case/legal-suffix/address abbreviation cleanup | Applied identically to train and test | CRITICAL | **IN PROGRESS** |
| 7 | Blocking / candidate generation | Reduce 1.7e13 pairs to a scoreable set | Recall ceiling >= 0.95 measured on train GT | CRITICAL | TODO |
| 8 | Pair features | Name + address + country similarity matrix | Feature matrix built on holdout | CRITICAL | TODO |
| 9 | LightGBM baseline | First honest local F_0.5 | Score recorded, beats all-empty floor | CRITICAL | TODO |
| 10 | Threshold tuning | Optimise F_0.5 (not F1), per country | Sweep done, France path validated | CRITICAL | TODO |
| 11 | Full-test inference | Generate both submission TSVs | validate_submission.py exits 0 | CRITICAL | TODO |
| 12 | Submit & iterate | Anchor a public score, compare vs local | Public/local gap understood | CRITICAL | TODO |
| 13 | Final package | Zero_submission.zip + methodology | Reproducible from scratch | CRITICAL | TODO |
| 14 | Feature expansion | Extra token/rank/address features | Each change scored vs baseline | LATER | TODO |
| 15 | Ensembling | Blend / stack pair scorers | Beats single model on holdout | OPTIONAL | TODO |
| 16 | Embeddings | Semantic name matching | Rejected - no GPU on this node | OPTIONAL | REJECTED |

## Where we are

**Phase 6 - normalisation.** Phases 1-5 are complete and verified. Holdout is frozen at 331,024 entities; measured all-empty floor is **0.0558**, oracle **1.0000**. Phase 16 is
rejected on hardware: the node has no GPU, so transformer embeddings over ~10M
records are not finishable in the time available.

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
