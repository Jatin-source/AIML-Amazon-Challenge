import json, gc, sys
import pandas as pd, numpy as np
from pathlib import Path

SHARDS = Path("outputs/shards")
TEST_DIR = Path("Dataset/student_resource/dataset/test")
COUNTRIES = ["France", "India", "US"]
USE_HUNGARIAN = "--hungarian" in sys.argv
SUFFIX = "_hungarian" if USE_HUNGARIAN else ""

def main():
    conf = json.load(open("backup/model/thresholds_v3.json"))
    thr = conf["thresholds"]
    frames = []
    for c in COUNTRIES:
        t = float(thr.get(c, conf.get("france_threshold", 0.5)))
        fs = sorted(SHARDS.glob(f"{c}_*.parquet"))
        if not fs:
            print(f"WARNING: no shards for {c}", flush=True); continue
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        d = d[d.p.notna() & (d.p >= t)]
        print(f"{c}: {len(fs)} shards -> {len(d)} pairs at t={t:.2f}", flush=True)
        frames.append(d)
    if not frames:
        raise SystemExit("no shards found")
    df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()

    if USE_HUNGARIAN:
        before = len(df)
        df = df.sort_values("p", ascending=False).drop_duplicates(
            subset=["candidate_entity_id"], keep="first")
        print(f"HUNGARIAN ON: {before} -> {len(df)} ({before-len(df)} dropped)", flush=True)
    else:
        print(f"HUNGARIAN OFF: keeping all {len(df)} assignments "
              f"(GT has 0 shared candidates and mean 3.46 matches/entity)", flush=True)

    preds = df.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
    cands = {}
    for c in COUNTRIES:
        for f in sorted(SHARDS.glob(f"{c}_*.parquet")):
            d = pd.read_parquet(f)
            for s1, grp in d[d.p.notna()].groupby("source1_entity_id")["candidate_entity_id"]:
                cands.setdefault(s1, []).extend(grp.tolist())
            del d
        gc.collect()

    ids = pd.concat([ch[["entity_id"]] for ch in pd.read_csv(
        TEST_DIR/"test_source1.tsv", sep="\t", dtype=str, usecols=["entity_id"],
        chunksize=500_000)], ignore_index=True).entity_id.values

    Path("outputs").mkdir(exist_ok=True)
    with open(f"outputs/matching_results{SUFFIX}.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in ids:
            f.write(f"{eid}\t{','.join(preds.get(eid, []))}\n")
    with open(f"outputs/candidate_pairs{SUFFIX}.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in ids:
            f.write(f"{eid}\t{','.join(cands.get(eid, []))}\n")

    matched = sum(1 for e in ids if preds.get(e))
    stats = pd.DataFrame([{"hungarian": USE_HUNGARIAN, "source1_rows": len(ids),
                           "rows_with_match": matched, "match_rate": round(matched/len(ids), 4),
                           "total_assignments": len(df),
                           "avg_matches_per_matched_row": round(len(df)/max(matched,1), 3)}])
    Path("backup/measurements").mkdir(parents=True, exist_ok=True)
    stats.to_csv(f"backup/measurements/v4-final-submission-stats{SUFFIX}.csv", index=False)
    print(stats.to_string(index=False), flush=True)
    print("FINALIZE COMPLETE", flush=True)

if __name__ == "__main__":
    main()
