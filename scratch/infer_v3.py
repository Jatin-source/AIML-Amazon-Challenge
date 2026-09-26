import os
import gc
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.normalize2 import core_name, norm_addr, norm_name, legal_set
from src.blocking3 import Index3
from src.features3 import build as build_features

def extract_country(filepath, country):
    chunks = []
    for chunk in pd.read_csv(filepath, sep="\t", dtype=str, chunksize=100_000):
        c = chunk[chunk["country"] == country]
        if not c.empty:
            chunks.append(c)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()

def run_safe():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    test_dir = Path("Dataset/student_resource/dataset/test")
    
    print("Loading model and thresholds...", flush=True)
    # Using the NEW v3 model!
    model = lgb.Booster(model_file="backup/model/lgbm_v3.txt")
    with open("backup/model/thresholds_v3.json") as f:
        conf = json.load(f)
        thresholds = conf["thresholds"]
        france_threshold = conf["france_threshold"]
    
    out_matching = open("outputs/matching_results.tsv", "w", encoding="utf-8")
    out_matching.write("source1_entity_id\tmatched_entity_ids\n")
    out_candidates = open("outputs/candidate_pairs.tsv", "w", encoding="utf-8")
    out_candidates.write("source1_entity_id\tcandidate_entity_ids\n")
    out_probs = open("outputs/predictions_raw.csv", "w", encoding="utf-8")
    out_probs.write("source1_entity_id,candidate_entity_id,probability\n")
    
    K_TOP = 25
    
    for country in ["France", "India", "US"]:
        print(f"\n--- Processing {country} ---", flush=True)
        
        t_s2 = extract_country(test_dir / "test_source2.tsv", country)
        t_s3 = extract_country(test_dir / "test_source3.tsv", country)
        
        p_c = pd.concat([t_s2, t_s3], ignore_index=True)
        del t_s2, t_s3; gc.collect()
        
        if len(p_c) == 0:
            t_s1 = extract_country(test_dir / "test_source1.tsv", country)
            for eid in t_s1.entity_id:
                out_matching.write(f"{eid}\t\n")
                out_candidates.write(f"{eid}\t\n")
            del t_s1
            gc.collect()
            continue
            
        p_c["c_core"] = p_c["business_name"].fillna("").apply(core_name)
        p_c["c_name"] = p_c["business_name"].fillna("").apply(norm_name)
        p_c["c_legal"] = p_c["business_name"].fillna("").apply(legal_set)
        p_c["c_addr"] = p_c["business_address"].fillna("").apply(norm_addr)
        
        ix = Index3(p_c.c_core.values, p_c.c_addr.values)
        pool_ids = p_c.entity_id.values
        pool_core = p_c.c_core.values
        pool_name = p_c.c_name.values
        pool_legal = p_c.c_legal.values
        pool_addr = p_c.c_addr.values
        del p_c; gc.collect()
        
        q_c = extract_country(test_dir / "test_source1.tsv", country)
        q_c["s1_core"] = q_c["business_name"].fillna("").apply(core_name)
        q_c["s1_name"] = q_c["business_name"].fillna("").apply(norm_name)
        q_c["s1_legal"] = q_c["business_name"].fillna("").apply(legal_set)
        q_c["s1_addr"] = q_c["business_address"].fillna("").apply(norm_addr)
        
        threshold = france_threshold if country == "France" else thresholds.get(country, 0.5)
        
        chunk = 10000
        for start in range(0, len(q_c), chunk):
            q = q_c.iloc[start:start+chunk]
            
            qi, ci, cs = [], [], []
            origins_list = []
            
            for k, (nm, ad) in enumerate(zip(q.s1_core.values, q.s1_addr.values)):
                cands, scores, origins = ix.query(nm, ad, top_k=K_TOP)
                for c, s, o in zip(cands, scores, origins):
                    qi.append(k)
                    ci.append(int(c))
                    cs.append(float(s))
                    origins_list.append(o)
                    
            if not qi: continue
                    
            qi = np.asarray(qi, dtype=np.int32)
            ci = np.asarray(ci, dtype=np.int32)
            cs = np.asarray(cs, dtype=np.float32)
            origins_arr = np.asarray(origins_list, dtype=bool)
            
            score_gaps = np.zeros(len(qi), dtype=np.float32)
            df_temp = pd.DataFrame({"q": qi, "s": cs})
            max_scores = df_temp.groupby("q")["s"].transform("max")
            score_gaps = (max_scores - cs).values
            
            frame = pd.DataFrame({
                "source1_entity_id": q.entity_id.values[qi],
                "candidate_entity_id": pool_ids[ci],
                "s1_core": q.s1_core.values[qi], "s1_name": q.s1_name.values[qi],
                "s1_legal": q.s1_legal.values[qi], "s1_addr": q.s1_addr.values[qi],
                "c_core": pool_core[ci], "c_name": pool_name[ci],
                "c_legal": pool_legal[ci], "c_addr": pool_addr[ci],
                "cos_score": cs,
                "score_gap": score_gaps,
                "in_word": origins_arr[:, 0].astype(int),
                "in_ngram": origins_arr[:, 1].astype(int),
                "in_addr": origins_arr[:, 2].astype(int),
                "in_phonetic": origins_arr[:, 3].astype(int)
            })
            
            cands_dict = frame.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
            
            X = build_features(frame, workers=-1)
            frame["p"] = model.predict(X, num_threads=4).astype(np.float32)
                
            frame_pred = frame[frame.p >= threshold]
            # Write probabilities for Hungarian resolution
            frame[["source1_entity_id", "candidate_entity_id", "p"]].to_csv(out_probs, mode="a", header=False, index=False)
            preds = frame_pred.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
            
            for eid in q.entity_id.values:
                pred_ids = set(preds.get(eid, []))
                cand_ids = set(cands_dict.get(eid, []))
                
                out_matching.write(f"{eid}\t{','.join(pred_ids)}\n")
                for c_id in cand_ids:
                    if c_id in cands_dict.get(eid, []):
                        # We need to map cand_ids to probs.
                        pass
                out_candidates.write(f"{eid}\t{','.join(cand_ids)}\n")
                
            del frame, qi, ci, cs, frame_pred, cands_dict, preds, X, origins_arr
            gc.collect()
            print(f"Processed up to {start+chunk}", flush=True)
            
        del ix, pool_ids, pool_core, pool_name, pool_legal, pool_addr, q_c
        gc.collect()
        
    out_matching.close()
    out_candidates.close()
    out_probs.close()
    print("Done generating memory-safe test output.", flush=True)

if __name__ == "__main__":
    run_safe()
