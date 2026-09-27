# France Zero-Shot Plan

All numbers below are **measured** on the India holdout (4,000 queries, 200,000 candidate pairs,
K_TOP=50) using a LightGBM model fit on the cached India training matrix (750,000 pairs, 36,130 positives).

## Hard dataset constraint

France has **zero training labels**:

| split | US | India | France |
|---|---|---|---|
| train source1 | 1,323,633 | 883,188 | **0** |
| train source2 | 3,016,817 | 2,017,799 | **0** |
| train source3 | 3,170,056 | 2,115,547 | **0** |
| test source1 | 663,106 | 809,986 | 259,452 |

France = 259,452 / 1,732,544 = **14.98%** of test source1. Supervised France training is impossible.
No pseudo-labelling of France has been attempted or validated.

## Measured finding 1 — ground-truth multiplicity invalidates Hungarian

From `train_ground_truth.tsv` (2,206,821 source1 rows, 7,638,365 unique candidate ids):

- mean true matches per source1 = **3.461**, median 3, max 11
- **89.02%** of source1 entities have **more than one** true match
- only 5.40% have exactly 1; 5.58% have zero
- candidate ids claimed by more than one source1 entity: **0 (0.0000%)**

The one-source1-per-candidate constraint in `finalize_v4.py` is therefore *structurally consistent*
with the ground truth. It cannot create a violation. But it also cannot help: since no candidate is
ever shared, a correct model produces no duplicate candidate assignments to remove. Hungarian is a
no-op on correct predictions and only ever deletes *incorrect* duplicates — a marginal precision
filter, not a source of gain. Must be measured, not assumed.

## Measured finding 2 — V5 tunes the wrong objective

The cached matrices contain features + labels only, no entity ids. The sweep inside
`train_v4_resumable.py` therefore maximises **pair-level micro F0.5**. The leaderboard scores
**macro F0.5 averaged per source1 entity**. Measured divergence:

| objective | optimal threshold | micro F0.5 | macro F0.5 (has-pos) |
|---|---|---|---|
| micro (what V5 optimises) | 0.80 | **0.9671** | 0.9511 |
| macro (what is scored) | 0.60 | 0.9621 | **0.9557** |

Optimising the wrong metric costs **-0.0046 macro F0.5** and pushes the threshold 0.20 too high.

## Measured finding 3 — blocking recall is the dominant loss, by ~4x

| component | value |
|---|---|
| queries with >=1 true match among 50 candidates | 3,397 / 4,000 = **84.92%** |
| queries with ZERO true match retrieved | **603** (15.08%) |
| macro F0.5 ceiling imposed by blocking | **0.8492** |
| macro F0.5 achieved @ t=0.60 | **0.8116** |
| recoverable by better ranking/threshold | **0.0376** |
| locked away by blocking recall | **0.1508** |

603 queries score exactly 0 at every threshold. No reranker, feature, or threshold change can
recover them. **Blocking recall is worth ~4x more than every ranking improvement combined.**

## Revised priority order

1. **Blocking recall** (ceiling 0.8492 -> higher). Largest available gain by a wide margin.
   Retrieval-side changes only: normalization variants, exact-match indexes (postal+name, phone,
   email, domain), K sweep. Must be measured as *candidate recall*, not F0.5.
2. **Macro-correct threshold tuning** (+0.0046 measured). Requires persisting entity ids in the
   cache so groups are reconstructable — currently recovered via the `n_cand` block walk.
3. **Ranking improvements** (bounded by 0.0376 total). Cross-source consistency, score margin,
   joint name/address. Lower ceiling than commonly assumed.
4. **Hungarian** — measure on/off. Expected near-zero given 0% candidate sharing.
5. **Pseudo-labelling** — last, only if 1-3 are exhausted and stability is demonstrated.

## Zero-shot transfer status

India->US and US->India transfer experiments require the US cache, which is still building
(`[build] US`, PID 765). Not yet measured. No France score has been estimated and none will be
reported without a measurement.
