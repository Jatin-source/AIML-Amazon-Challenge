import os, gc, json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.make_holdout import build as make_holdout
from src.normalize2 import core_name, norm_addr, norm_name, legal_set
from src.blocking3 import Index3
from src.features3 import build as build_features

def extract_country(filepath, country):
    chunks = []
    for chunk in pd.read_csv(filepath, sep="\t", dtype=str, chunksize=100_000):
        c = chunk[chunk["country"] == country]
        if not c.empty:
            chunks.append(c)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def train_model():
    train_dir = Path("Dataset/student_resource/dataset/train")
    out_dir = Path("backup/model"); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data & holdout split...")
    split_df = make_holdout(train_dir)
    train_ids = set(split_df[split_df.split == "train"].entity_id)
    holdout_ids = set(split_df[split_df.split == "holdout"].entity_id)
    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str)

    K_TOP = 50          # Phase D: increased from 25
    N_SAMPLE = 15000    # Phase B: increased from 5000

    train_features, train_labels = [], []
    holdout_features, holdout_labels = [], []
    thresholds = {}

    for country in ["France", "India", "US"]:   # Phase B: all 3 countries
        print(f"\n--- {country} ---")
        t_s2 = extract_country(train_dir / "train_source2.tsv", country)
        t_s3 = extract_country(train_dir / "train_source3.tsv", country)
        p_c = pd.concat([t_s2, t_s3], ignore_index=True)
        del t_s2, t_s3; gc.collect()
        if len(p_c) == 0: continue

        p_c["c_core"]  = p_c["business_name"].fillna("").apply(core_name)
        p_c["c_name"]  = p_c["business_name"].fillna("").apply(norm_name)
        p_c["c_legal"] = p_c["business_name"].fillna("").apply(legal_set)
        p_c["c_addr"]  = p_c["business_address"].fillna("").apply(norm_addr)
        ix = Index3(p_c.c_core.values, p_c.c_addr.values)
        pool_ids   = p_c.entity_id.values
        pool_core  = p_c.c_core.values
        pool_name  = p_c.c_name.values
        pool_legal = p_c.c_legal.values
        pool_addr  = p_c.c_addr.values
        del p_c; gc.collect()

        q_c = extract_country(train_dir / "train_source1.tsv", country)
        q_train = q_c[q_c.entity_id.isin(train_ids)].reset_index(drop=True)
        q_hold  = q_c[q_c.entity_id.isin(holdout_ids)].reset_index(drop=True)
        if len(q_train) > N_SAMPLE:
            q_train = q_train.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
        if len(q_hold) > 3000:
            q_hold = q_hold.sample(3000, random_state=42).reset_index(drop=True)

        for split_q, feats_list, labs_list in [
                (q_train, train_features, train_labels),
                (q_hold, holdout_features, holdout_labels)]:
            if len(split_q) == 0: continue
            for col in ["business_name", "business_address"]:
                split_q[col] = split_q[col].fillna("")
            split_q["s1_core"]  = split_q["business_name"].apply(core_name)
            split_q["s1_name"]  = split_q["business_name"].apply(norm_name)
            split_q["s1_legal"] = split_q["business_name"].apply(legal_set)
            split_q["s1_addr"]  = split_q["business_address"].apply(norm_addr)

            chunk = 2500
            for start in range(0, len(split_q), chunk):
                q = split_q.iloc[start:start+chunk]
                qi, ci, cs, origins_list = [], [], [], []
                for k, (nm, ad) in enumerate(zip(q.s1_core.values, q.s1_addr.values)):
                    cands, scores, origins = ix.query(nm, ad, top_k=K_TOP)
                    for c, s, o in zip(cands, scores, origins):
                        qi.append(k); ci.append(int(c)); cs.append(float(s)); origins_list.append(o)
                if not qi: continue
                qi  = np.asarray(qi, dtype=np.int32)
                ci  = np.asarray(ci, dtype=np.int32)
                cs  = np.asarray(cs, dtype=np.float32)
                origins_arr = np.asarray(origins_list, dtype=bool)
                df_t = pd.DataFrame({"q": qi, "s": cs})
                score_gaps = (df_t.groupby("q")["s"].transform("max") - cs).values
                frame = pd.DataFrame({
                    "source1_entity_id": q.entity_id.values[qi],
                    "candidate_entity_id": pool_ids[ci],
                    "s1_core": q.s1_core.values[qi], "s1_name": q.s1_name.values[qi],
                    "s1_legal": q.s1_legal.values[qi], "s1_addr": q.s1_addr.values[qi],
                    "c_core": pool_core[ci], "c_name": pool_name[ci],
                    "c_legal": pool_legal[ci], "c_addr": pool_addr[ci],
                    "cos_score": cs, "score_gap": score_gaps,
                    "in_word": origins_arr[:,0].astype(int), "in_ngram": origins_arr[:,1].astype(int),
                    "in_addr": origins_arr[:,2].astype(int), "in_phonetic": origins_arr[:,3].astype(int)
                })
                X = build_features(frame, workers=-1)
                gt_c = gt[gt.source1_entity_id.isin(frame.source1_entity_id)].set_index("source1_entity_id")
                labels = []
                for s1_id, c_id in zip(frame.source1_entity_id, frame.candidate_entity_id):
                    true_str = gt_c.loc[s1_id, "matched_entity_ids"] if s1_id in gt_c.index else ""
                    true_ids = set(x.strip() for x in str(true_str).split(",") if x.strip() and pd.notna(true_str))
                    labels.append(1 if c_id in true_ids else 0)
                feats_list.append(X); labs_list.append(np.array(labels))
                del frame, X, origins_arr; gc.collect()
                print(f"  {split_q is q_train and 'train' or 'hold'} chunk {start+chunk}")
        del ix, pool_ids, pool_core, pool_name, pool_legal, pool_addr, q_c; gc.collect()

    X_train = pd.concat(train_features, ignore_index=True)
    y_train = np.concatenate(train_labels)
    print("\nTraining LightGBM v3 full (3 countries)...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {"objective":"binary","metric":"binary_logloss","learning_rate":0.05,
              "num_leaves":63,"bagging_fraction":0.8,"feature_fraction":0.8,"verbose":-1,"n_jobs":32}
    model = lgb.train(params, dtrain, num_boost_round=200)
    model.save_model("backup/model/lgbm_v3.txt")
    print("Model saved.")

    # Phase C: threshold tuning on holdout
    print("\nTuning F0.5 thresholds on holdout set...")
    X_hold = pd.concat(holdout_features, ignore_index=True)
    y_hold = np.concatenate(holdout_labels)
    probs  = model.predict(X_hold, num_threads=32)

    best_thresholds = {}
    best_global = 0.5
    for thresh in np.arange(0.2, 0.85, 0.02):
        preds = (probs >= thresh).astype(int)
        if preds.sum() == 0: continue
        prec = (preds * y_hold).sum() / preds.sum()
        rec  = (preds * y_hold).sum() / max(y_hold.sum(), 1)
        f05  = (1 + 0.25) * prec * rec / max(0.25 * prec + rec, 1e-9)
        if f05 > best_global:
            best_global = f05
            best_thresholds["global"] = float(thresh)
    best = best_thresholds.get("global", 0.5)
    print(f"Best global threshold: {best:.3f}  F0.5={best_global:.4f}")

    with open("backup/model/thresholds_v3.json", "w") as f:
        import json
        json.dump({"thresholds": {"India": best, "US": best}, "france_threshold": best}, f)
    print("Thresholds saved.")

if __name__ == "__main__":
    train_model()
