import os, gc, json, sys
import pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.make_holdout import build as make_holdout
from src.normalize2 import core_name, norm_addr, norm_name, legal_set
from src.blocking3 import Index3
from src.features3 import build as build_features

K_TOP, N_TRAIN, N_HOLD = 50, 15000, 4000
TRAIN_DIR = Path("Dataset/student_resource/dataset/train")
CACHE = Path("backup/cache"); CACHE.mkdir(parents=True, exist_ok=True)
COUNTRIES = ["France", "India", "US"]

def extract_country(fp, country):
    out = [c[c["country"] == country] for c in pd.read_csv(fp, sep="\t", dtype=str, chunksize=200_000)]
    out = [c for c in out if not c.empty]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def norm_pool(df, pre):
    df[pre+"core"]  = df["business_name"].fillna("").apply(core_name)
    df[pre+"name"]  = df["business_name"].fillna("").apply(norm_name)
    df[pre+"legal"] = df["business_name"].fillna("").apply(legal_set)
    df[pre+"addr"]  = df["business_address"].fillna("").apply(norm_addr)
    return df

def pairs(q, ix, pool, gt_idx):
    pool_ids, p_core, p_name, p_legal, p_addr = pool
    qi, ci, cs, og = [], [], [], []
    for k, (nm, ad) in enumerate(zip(q.s1_core.values, q.s1_addr.values)):
        c_, s_, o_ = ix.query(nm, ad, top_k=K_TOP)
        for c, s, o in zip(c_, s_, o_):
            qi.append(k); ci.append(int(c)); cs.append(float(s)); og.append(o)
    if not qi: return None, None
    qi = np.asarray(qi, np.int32); ci = np.asarray(ci, np.int32); cs = np.asarray(cs, np.float32)
    oa = np.asarray(og, bool)
    gap = (pd.DataFrame({"q": qi, "s": cs}).groupby("q")["s"].transform("max") - cs).values
    fr = pd.DataFrame({
        "source1_entity_id": q.entity_id.values[qi], "candidate_entity_id": pool_ids[ci],
        "s1_core": q.s1_core.values[qi], "s1_name": q.s1_name.values[qi],
        "s1_legal": q.s1_legal.values[qi], "s1_addr": q.s1_addr.values[qi],
        "c_core": p_core[ci], "c_name": p_name[ci], "c_legal": p_legal[ci], "c_addr": p_addr[ci],
        "cos_score": cs, "score_gap": gap,
        "in_word": oa[:,0].astype(int), "in_ngram": oa[:,1].astype(int),
        "in_addr": oa[:,2].astype(int), "in_phonetic": oa[:,3].astype(int)})
    X = build_features(fr, workers=-1)
    y = np.fromiter((1 if c in gt_idx.get(s, ()) else 0
                     for s, c in zip(fr.source1_entity_id, fr.candidate_entity_id)), np.int8, len(fr))
    return X, y

def build_country(country, train_ids, hold_ids, gt_idx):
    xt, xh = CACHE/f"{country}_train_X.parquet", CACHE/f"{country}_hold_X.parquet"
    if xt.exists() and xh.exists():
        print(f"[cache hit] {country}", flush=True); return
    print(f"[build] {country}", flush=True)
    p = pd.concat([extract_country(TRAIN_DIR/"train_source2.tsv", country),
                   extract_country(TRAIN_DIR/"train_source3.tsv", country)], ignore_index=True)
    if p.empty: return
    p = norm_pool(p, "c_")
    ix = Index3(p.c_core.values, p.c_addr.values)
    pool = (p.entity_id.values, p.c_core.values, p.c_name.values, p.c_legal.values, p.c_addr.values)
    del p; gc.collect()
    q_all = extract_country(TRAIN_DIR/"train_source1.tsv", country)
    for tag, ids, n, dest in [("train", train_ids, N_TRAIN, xt), ("hold", hold_ids, N_HOLD, xh)]:
        qs = q_all[q_all.entity_id.isin(ids)].reset_index(drop=True)
        if len(qs) > n: qs = qs.sample(n, random_state=42).reset_index(drop=True)
        if qs.empty: continue
        qs = norm_pool(qs, "s1_")
        Xs, ys = [], []
        for st in range(0, len(qs), 2500):
            X, y = pairs(qs.iloc[st:st+2500], ix, pool, gt_idx)
            if X is not None: Xs.append(X); ys.append(y)
            print(f"  {country}/{tag} {min(st+2500,len(qs))}/{len(qs)}", flush=True)
        if Xs:
            pd.concat(Xs, ignore_index=True).to_parquet(dest, index=False)
            np.save(str(dest).replace("_X.parquet", "_y.npy"), np.concatenate(ys))
        del Xs, ys; gc.collect()
    del ix, pool, q_all; gc.collect()

def f05(y, p, t):
    pr = (p >= t)
    tp = float((pr & (y == 1)).sum())
    if pr.sum() == 0 or y.sum() == 0: return 0.0
    prec, rec = tp/pr.sum(), tp/y.sum()
    return 0.0 if prec+rec == 0 else 1.25*prec*rec/(0.25*prec+rec)

def main():
    split = make_holdout(TRAIN_DIR)
    train_ids = set(split[split.split == "train"].entity_id)
    hold_ids  = set(split[split.split == "holdout"].entity_id)
    gt = pd.read_csv(TRAIN_DIR/"train_ground_truth.tsv", sep="\t", dtype=str)
    gt_idx = {s: frozenset(x.strip() for x in str(m).split(",") if x.strip())
              for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) if pd.notna(m)}
    del gt; gc.collect()
    for c in COUNTRIES:
        build_country(c, train_ids, hold_ids, gt_idx)

    Xtr = pd.concat([pd.read_parquet(CACHE/f"{c}_train_X.parquet") for c in COUNTRIES
                     if (CACHE/f"{c}_train_X.parquet").exists()], ignore_index=True)
    ytr = np.concatenate([np.load(CACHE/f"{c}_train_y.npy") for c in COUNTRIES
                          if (CACHE/f"{c}_train_y.npy").exists()])
    print(f"train matrix {Xtr.shape} positives={int(ytr.sum())}", flush=True)
    params = {"objective":"binary","metric":"binary_logloss","learning_rate":0.05,
              "num_leaves":127,"min_data_in_leaf":40,"bagging_fraction":0.8,"bagging_freq":1,
              "feature_fraction":0.8,"lambda_l2":1.0,"verbose":-1,"n_jobs":32,
              "scale_pos_weight":float((ytr==0).sum())/max(int(ytr.sum()),1)}
    model = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=400)
    model.save_model("backup/model/lgbm_v3.txt")
    del Xtr, ytr; gc.collect()

    grid = np.arange(0.05, 0.96, 0.01)
    thr, rows = {}, []
    for c in COUNTRIES:
        fx = CACHE/f"{c}_hold_X.parquet"
        if not fx.exists(): thr[c] = 0.5; continue
        Xh = pd.read_parquet(fx); yh = np.load(CACHE/f"{c}_hold_y.npy")
        ph = model.predict(Xh, num_threads=32)
        scores = [(f05(yh, ph, t), float(t)) for t in grid]
        best_f, best_t = max(scores)
        thr[c] = best_t
        rows.append({"country": c, "threshold": best_t, "f05": round(best_f, 4),
                     "f05_at_0.5": round(f05(yh, ph, 0.5), 4), "pairs": len(yh),
                     "positives": int(yh.sum())})
        del Xh, yh, ph; gc.collect()
    with open("backup/model/thresholds_v3.json", "w") as f:
        json.dump({"thresholds": thr, "france_threshold": thr.get("France", 0.5)}, f, indent=2)
    pd.DataFrame(rows).to_csv("backup/measurements/v4-holdout-f05-by-country.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("TRAIN STAGE COMPLETE", flush=True)

if __name__ == "__main__":
    main()
