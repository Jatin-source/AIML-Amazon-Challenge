import sys, gc, time, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.make_holdout import build as make_holdout
from src.normalize2 import core_name, norm_addr
from src.blocking3_union import Index3Union

TRAIN = Path("Dataset/student_resource/dataset/train")
COUNTRY = "India"
N_HOLD, K_BASE, K_NS = 4000, 50, 20

def extract(fp, country):
    out = [c[c["country"] == country] for c in pd.read_csv(fp, sep="\t", dtype=str, chunksize=200_000)]
    out = [c for c in out if not c.empty]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

split = make_holdout(TRAIN)
hold_ids = set(split[split.split == "holdout"].entity_id)
gt = pd.read_csv(TRAIN/"train_ground_truth.tsv", sep="\t", dtype=str)
gt_idx = {s: frozenset(x.strip() for x in str(m).split(",") if x.strip())
          for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) if pd.notna(m)}
del gt, split; gc.collect()

pool = pd.concat([extract(TRAIN/"train_source2.tsv", COUNTRY),
                  extract(TRAIN/"train_source3.tsv", COUNTRY)], ignore_index=True)
pool["c_core"] = pool["business_name"].fillna("").apply(core_name)
pool["c_addr"] = pool["business_address"].fillna("").apply(norm_addr)
print(f"pool={len(pool)}", flush=True)
t0 = time.time()
ix = Index3Union(pool.c_core.values, pool.c_addr.values)
pool_ids = pool.entity_id.values
print(f"index built in {time.time()-t0:.0f}s", flush=True)
del pool; gc.collect()

q = extract(TRAIN/"train_source1.tsv", COUNTRY)
q = q[q.entity_id.isin(hold_ids)].reset_index(drop=True)
q = q.sample(N_HOLD, random_state=42).reset_index(drop=True)
q["s1_core"] = q["business_name"].fillna("").apply(core_name)
q["s1_addr"] = q["business_address"].fillna("").apply(norm_addr)

rows = []
for tag in ["baseline", "union"]:
    t0 = time.time()
    hit, tot_true, tot_found, n_cand, only_ng = 0, 0, 0, 0, 0
    for eid, nm, ad in zip(q.entity_id.values, q.s1_core.values, q.s1_addr.values):
        truth = gt_idx.get(eid, frozenset())
        if tag == "baseline":
            ids, sc, og = ix.query(nm, ad, top_k=K_BASE)
        else:
            ids, sc, og = ix.query_union(nm, ad, k_per_ns=K_NS)
        got = set(pool_ids[ids].tolist()) if len(ids) else set()
        f = len(got & truth)
        n_cand += len(got); tot_true += len(truth); tot_found += f
        if f: hit += 1
        if tag == "union" and len(ids):
            ng_only = og[:, 1] & ~og[:, 0]
            if ng_only.any():
                only_ng += len(set(pool_ids[ids[ng_only]].tolist()) & truth)
    rows.append(dict(variant=tag, queries=len(q),
                     candidate_recall=round(hit/len(q), 4),
                     true_match_recall=round(tot_found/max(tot_true,1), 4),
                     avg_candidates=round(n_cand/len(q), 1),
                     queries_zero_recall=len(q)-hit,
                     true_matches_only_from_ngram=only_ng,
                     seconds=round(time.time()-t0, 1)))
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
Path("backup/measurements").mkdir(parents=True, exist_ok=True)
df.to_csv("backup/measurements/union-vs-weighted-candidate-recall.csv", index=False)
print(df.to_string(index=False), flush=True)
b, u = df.iloc[0], df.iloc[1]
print(f"\nDELTA candidate_recall: {u.candidate_recall - b.candidate_recall:+.4f}", flush=True)
print(f"DELTA zero-recall queries: {int(u.queries_zero_recall - b.queries_zero_recall)}", flush=True)
print(f"cost: {u.avg_candidates/b.avg_candidates:.2f}x candidates, {u.seconds/b.seconds:.2f}x time", flush=True)
print("UNION TEST COMPLETE", flush=True)
