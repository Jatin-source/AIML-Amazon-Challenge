import os, gc, json, sys
import pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
import multiprocessing as mp
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.normalize2 import core_name, norm_addr, norm_name, legal_set
from src.blocking3_union import Index3Union as Index3
from src.features3 import build as build_features

K_TOP = 50
CHUNK = 20000
N_PROC = max(2, int(os.cpu_count() * 0.80))
TEST_DIR = Path("Dataset/student_resource/dataset/test")
SHARDS = Path("outputs/shards_union"); SHARDS.mkdir(parents=True, exist_ok=True)
COUNTRIES = ["France", "India", "US"]

_IX = None
_CORES = None
_ADDRS = None

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

def _init(ix):
    global _IX
    _IX = ix

def _retrieve(args):
    off, cores, addrs = args
    qi, ci, cs, og = [], [], [], []
    for k, (nm, ad) in enumerate(zip(cores, addrs)):
        c_, s_, o_ = _IX.query_union(nm, ad, k_per_ns=20)
        for c, s, o in zip(c_, s_, o_):
            qi.append(off + k); ci.append(int(c)); cs.append(float(s)); og.append(o)
    return qi, ci, cs, og

def run_country(country, model, threshold):
    done = sorted(int(p.stem.split("_")[-1]) for p in SHARDS.glob(f"{country}_*.parquet"))
    q_all = extract_country(TEST_DIR / "test_source1.tsv", country)
    if q_all.empty:
        return
    pool_df = pd.concat([extract_country(TEST_DIR/"test_source2.tsv", country),
                         extract_country(TEST_DIR/"test_source3.tsv", country)], ignore_index=True)
    if pool_df.empty:
        pd.DataFrame({"source1_entity_id": q_all.entity_id.values,
                      "candidate_entity_id": "", "p": np.nan}).to_parquet(
            SHARDS / f"{country}_0.parquet", index=False)
        return
    pool_df = norm_pool(pool_df, "c_")
    ix = Index3(pool_df.c_core.values, pool_df.c_addr.values)
    pool_ids = pool_df.entity_id.values
    p_core, p_name, p_legal, p_addr = (pool_df.c_core.values, pool_df.c_name.values,
                                       pool_df.c_legal.values, pool_df.c_addr.values)
    del pool_df; gc.collect()
    q_all = norm_pool(q_all, "s1_")
    n = len(q_all)
    print(f"{country}: {n} queries, resuming after chunks {done[-1] if done else 'none'}", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(N_PROC, initializer=_init, initargs=(ix,)) as pool:
        for idx, start in enumerate(range(0, n, CHUNK)):
            if idx in done:
                continue
            q = q_all.iloc[start:start+CHUNK]
            slice_n = max(1, len(q) // N_PROC + 1)
            tasks = [(i, q.s1_core.values[i:i+slice_n], q.s1_addr.values[i:i+slice_n])
                     for i in range(0, len(q), slice_n)]
            qi, ci, cs, og = [], [], [], []
            for a, b, c, d in pool.map(_retrieve, tasks):
                qi += a; ci += b; cs += c; og += d
            if not qi:
                continue
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
            probs = model.predict(X, num_threads=N_PROC).astype(np.float32)
            keep = probs >= min(threshold, 0.30)
            pd.DataFrame({"source1_entity_id": fr.source1_entity_id.values[keep],
                          "candidate_entity_id": fr.candidate_entity_id.values[keep],
                          "p": probs[keep]}).to_parquet(SHARDS / f"{country}_{idx}.parquet", index=False)
            del fr, X, probs, oa; gc.collect()
            print(f"{country} chunk {idx} done ({min(start+CHUNK, n)}/{n})", flush=True)
    del ix; gc.collect()

def main():
    model = lgb.Booster(model_file="backup/model/lgbm_v5_union.txt")
    conf = json.load(open("backup/model/thresholds_v5_union.json"))
    thr = conf["thresholds"]
    for c in COUNTRIES:
        t = conf.get("france_threshold", 0.5) if c == "France" else thr.get(c, 0.5)
        run_country(c, model, float(t))
    print("INFER STAGE COMPLETE", flush=True)

if __name__ == "__main__":
    main()
