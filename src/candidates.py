"""Phase 7 LOCKED candidate generation: raw-IDF top-50 UNION cosine top-50.

Measured recall on the frozen holdout (2,000 queries/country vs the full S2+S3 pool):
India 0.9440, US 0.9844, ~84 candidates per entity.

Per-country indexes are built once and queried in chunks; each chunk is written to its
own Parquet shard so a kill or restart resumes instead of redoing the whole pass.
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import InvertedIndex

K_RAW = 50
K_COS = 50
CHUNK = 100_000


def _query_block(ix, pool_ids, q):
    out_s1, out_cand, out_score = [], [], []
    for eid, nm, ad in zip(q.entity_id.values, q.n_name.values, q.n_addr.values):
        raw, _ = ix.query(nm, ad, top_k=K_RAW)
        cos, cs = ix.query_cosine(nm, ad, top_k=K_COS)
        cos_rank = {int(v): i for i, v in enumerate(cos.tolist())}
        union = list(dict.fromkeys(raw.tolist() + cos.tolist()))
        for j in union:
            out_s1.append(eid)
            out_cand.append(pool_ids[j])
            out_score.append(float(cs[cos_rank[j]]) if j in cos_rank else 0.0)
    return pd.DataFrame({"source1_entity_id": out_s1,
                         "candidate_entity_id": out_cand,
                         "cos_score": np.asarray(out_score, dtype=np.float32)})


def generate(s1, pool, out_dir, chunk=CHUNK):
    """Write candidate shards for every S1 record in ``s1`` against ``pool``.

    ``s1`` / ``pool``: normalised frames with entity_id, country, n_name, n_addr.
    Returns the list of shard paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for country in sorted(s1.country.unique()):
        p = pool[pool.country == country].reset_index(drop=True)
        if p.empty:                      # unseen country (e.g. France): no same-country pool
            p = pool.reset_index(drop=True)
        ix = InvertedIndex(p.n_name.values, p.n_addr.values).build_norms()
        pool_ids = p.entity_id.values
        q_all = s1[s1.country == country].reset_index(drop=True)
        for start in range(0, len(q_all), chunk):
            shard = out_dir / f"{country}_{start:08d}.parquet"
            shards.append(shard)
            if shard.exists():
                continue
            _query_block(ix, pool_ids, q_all.iloc[start:start + chunk]).to_parquet(
                shard, index=False)
        del ix, p
        gc.collect()
    return shards
