import os, gc, json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.normalize2 import core_name, norm_addr, norm_name, legal_set
from src.blocking3 import Index3
from src.features3 import build as build_features

N_WORKERS = int(os.cpu_count() * 0.85)   # ~54 of 64 cores
K_TOP      = 50

def extract_country(filepath, country):
    chunks = []
    for chunk in pd.read_csv(filepath, sep="\t", dtype=str, chunksize=100_000):
        c = chunk[chunk["country"] == country]
        if not c.empty:
            chunks.append(c)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def query_batch(args):
    """Process a slice of queries; returns (qi, ci, cs, origins)."""
    q_slice, ix, K_TOP = args
    qi, ci, cs, origins_list = [], [], [], []
    for k, (nm, ad) in enumerate(zip(q_slice.s1_core.values, q_slice.s1_addr.values)):
        cands, scores, origins = ix.query(nm, ad, top_k=K_TOP)
        for c, s, o in zip(cands, scores, origins):
            qi.append(k); ci.append(int(c)); cs.append(float(s)); origins_list.append(o)
    return qi, ci, cs, origins_list

def run_safe():
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    test_dir = Path("Dataset/student_resource/dataset/test")

    print(f"Loading model (K_TOP={K_TOP}, workers={N_WORKERS})...", flush=True)
    model = lgb.Booster(model_file="backup/model/lgbm_v3.txt")
    with open("backup/model/thresholds_v3.json") as f:
        conf = json.load(f)
        thresholds      = conf["thresholds"]
        france_threshold = conf["france_threshold"]

    out_matching   = open("outputs/matching_results.tsv",   "w", encoding="utf-8")
    out_candidates = open("outputs/candidate_pairs.tsv",    "w", encoding="utf-8")
    out_probs      = open("outputs/predictions_raw.csv",    "w", encoding="utf-8")
    out_matching.write("source1_entity_id\tmatched_entity_ids\n")
    out_candidates.write("source1_entity_id\tcandidate_entity_ids\n")
    out_probs.write("source1_entity_id,candidate_entity_id,probability\n")

    for country in ["France", "India", "US"]:
        print(f"\n--- {country} ---", flush=True)
        t_s2 = extract_country(test_dir / "test_source2.tsv", country)
        t_s3 = extract_country(test_dir / "test_source3.tsv", country)
        p_c  = pd.concat([t_s2, t_s3], ignore_index=True)
        del t_s2, t_s3; gc.collect()

        if len(p_c) == 0:
            t_s1 = extract_country(test_dir / "test_source1.tsv", country)
            for eid in t_s1.entity_id:
                out_matching.write(f"{eid}\t\n")
                out_candidates.write(f"{eid}\t\n")
            del t_s1; gc.collect(); continue

        p_c["c_core"]  = p_c["business_name"].fillna("").apply(core_name)
        p_c["c_name"]  = p_c["business_name"].fillna("").apply(norm_name)
        p_c["c_legal"] = p_c["business_name"].fillna("").apply(legal_set)
        p_c["c_addr"]  = p_c["business_address"].fillna("").apply(norm_addr)
        ix         = Index3(p_c.c_core.values, p_c.c_addr.values)
        pool_ids   = p_c.entity_id.values
        pool_core  = p_c.c_core.values
        pool_name  = p_c.c_name.values
        pool_legal = p_c.c_legal.values
        pool_addr  = p_c.c_addr.values
        del p_c; gc.collect()

        q_c = extract_country(test_dir / "test_source1.tsv", country)
        q_c["s1_core"]  = q_c["business_name"].fillna("").apply(core_name)
        q_c["s1_name"]  = q_c["business_name"].fillna("").apply(norm_name)
        q_c["s1_legal"] = q_c["business_name"].fillna("").apply(legal_set)
        q_c["s1_addr"]  = q_c["business_address"].fillna("").apply(norm_addr)

        threshold = france_threshold if country == "France" else thresholds.get(country, 0.5)
        chunk_size = 10000

        for start in range(0, len(q_c), chunk_size):
            q = q_c.iloc[start:start+chunk_size]

            # ── parallel retrieval across N_WORKERS slices ──────────────────
            slice_size = max(1, len(q) // N_WORKERS)
            slices = [q.iloc[i:i+slice_size] for i in range(0, len(q), slice_size)]
            all_qi, all_ci, all_cs, all_orig = [], [], [], []
            offset = 0
            with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
                results = list(ex.map(query_batch, [(s, ix, K_TOP) for s in slices]))
            for (qi, ci, cs, origs), sl in zip(results, slices):
                for i in range(len(qi)):
                    all_qi.append(qi[i] + offset)
                    all_ci.append(ci[i]); all_cs.append(cs[i]); all_orig.append(origs[i])
                offset += len(sl)

            if not all_qi: continue
            qi  = np.asarray(all_qi, dtype=np.int32)
            ci  = np.asarray(all_ci, dtype=np.int32)
            cs  = np.asarray(all_cs, dtype=np.float32)
            origins_arr = np.asarray(all_orig, dtype=bool)

            df_t = pd.DataFrame({"q": qi, "s": cs})
            score_gaps = (df_t.groupby("q")["s"].transform("max") - cs).values

            frame = pd.DataFrame({
                "source1_entity_id":   q.entity_id.values[qi],
                "candidate_entity_id": pool_ids[ci],
                "s1_core":  q.s1_core.values[qi],  "s1_name":  q.s1_name.values[qi],
                "s1_legal": q.s1_legal.values[qi],  "s1_addr":  q.s1_addr.values[qi],
                "c_core":   pool_core[ci],           "c_name":   pool_name[ci],
                "c_legal":  pool_legal[ci],          "c_addr":   pool_addr[ci],
                "cos_score": cs, "score_gap": score_gaps,
                "in_word":     origins_arr[:,0].astype(int),
                "in_ngram":    origins_arr[:,1].astype(int),
                "in_addr":     origins_arr[:,2].astype(int),
                "in_phonetic": origins_arr[:,3].astype(int),
            })
            cands_dict = frame.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
            X = build_features(frame, workers=-1)
            frame["p"] = model.predict(X, num_threads=N_WORKERS).astype(np.float32)
            frame[["source1_entity_id","candidate_entity_id","p"]].to_csv(
                out_probs, mode="a", header=False, index=False)

            frame_pred = frame[frame.p >= threshold]
            preds = frame_pred.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()

            for eid in q.entity_id.values:
                pred_ids = set(preds.get(eid, []))
                cand_ids = set(cands_dict.get(eid, []))
                out_matching.write(f"{eid}\t{\',\'.join(pred_ids)}\n")
                out_candidates.write(f"{eid}\t{\',\'.join(cand_ids)}\n")

            del frame, X, origins_arr, frame_pred, cands_dict, preds; gc.collect()
            print(f"Processed up to {start+chunk_size}", flush=True)

        del ix, pool_ids, pool_core, pool_name, pool_legal, pool_addr, q_c; gc.collect()

    out_matching.close(); out_candidates.close(); out_probs.close()
    print("Done generating parallel test output.", flush=True)

if __name__ == "__main__":
    run_safe()
