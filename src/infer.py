"""Full-test inference: blocking -> features -> LightGBM -> threshold (Phase 11).

Per-country, chunked, and resumable: each chunk writes its own Parquet shard, so a
kill or kernel restart resumes instead of repeating the whole pass. Blocking queries
run in a thread pool - the hot path (concatenate / argsort / reduceat) is NumPy and
releases the GIL, and threads avoid the copy-on-write blow-up that killed a fork pool
on the 1M-entry token index.
"""
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import InvertedIndex
from .features import build as build_features

K_RAW = 50
K_COS = 50
CHUNK = 50_000
THREADS = 16


def _block_one(args):
    ix, nm, ad = args
    raw, _ = ix.query(nm, ad, top_k=K_RAW)
    cos, csc = ix.query_cosine(nm, ad, top_k=K_COS)
    return raw, cos, csc


def block_chunk(ix, q, threads=THREADS):
    """Return flat arrays (query_pos, pool_idx, raw_rank, cos_rank, cos_score)."""
    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = list(ex.map(_block_one,
                          ((ix, nm, ad) for nm, ad in zip(q.n_name.values, q.n_addr.values)),
                          chunksize=64))
    qi, ci, rr, cr, cs = [], [], [], [], []
    for k, (raw, cos, csc) in enumerate(res):
        rmap = {int(v): i for i, v in enumerate(raw.tolist())}
        cmap = {int(v): i for i, v in enumerate(cos.tolist())}
        for j in dict.fromkeys(raw.tolist() + cos.tolist()):
            j = int(j)
            qi.append(k); ci.append(j)
            rr.append(rmap.get(j, 99)); cr.append(cmap.get(j, 99))
            cs.append(float(csc[cmap[j]]) if j in cmap else 0.0)
    return (np.asarray(qi, dtype=np.int32), np.asarray(ci, dtype=np.int32),
            np.asarray(rr, dtype=np.int16), np.asarray(cr, dtype=np.int16),
            np.asarray(cs, dtype=np.float32))


def run_country(s1, pool, model, threshold, out_dir, country,
                chunk=CHUNK, threads=THREADS, feature_names=None):
    """Score every s1 record of one country; write one shard per chunk.

    Shards carry every candidate plus its probability, so candidate_pairs.tsv and
    matching_results.tsv are both derivable without re-running the model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = pool.reset_index(drop=True)
    ix = InvertedIndex(p.n_name.values, p.n_addr.values).build_norms()
    pool_ids = p.entity_id.values
    pool_name = p.n_name.values
    pool_addr = p.n_addr.values
    q_all = s1.reset_index(drop=True)
    shards = []
    for start in range(0, len(q_all), chunk):
        shard = out_dir / f"{country}_{start:08d}.parquet"
        shards.append(shard)
        if shard.exists():
            continue
        q = q_all.iloc[start:start + chunk]
        qi, ci, rr, cr, cs = block_chunk(ix, q, threads)
        frame = pd.DataFrame({
            "source1_entity_id": q.entity_id.values[qi],
            "candidate_entity_id": pool_ids[ci],
            "s1_name": q.n_name.values[qi], "s1_addr": q.n_addr.values[qi],
            "c_name": pool_name[ci], "c_addr": pool_addr[ci],
            "raw_rank": rr, "cos_rank": cr, "cos_score": cs,
        })
        X = build_features(frame, workers=-1)
        if feature_names is not None:
            X = X[feature_names]
        frame["p"] = model.predict(X, num_threads=64).astype(np.float32)
        frame[["source1_entity_id", "candidate_entity_id", "p"]].to_parquet(
            shard, index=False)
        del frame, X, qi, ci, rr, cr, cs
        gc.collect()
    del ix, p
    gc.collect()
    return shards
