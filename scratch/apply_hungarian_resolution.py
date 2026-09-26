import pandas as pd
from pathlib import Path
import time

def resolve():
    print("Starting Global Graph Resolution (Hungarian)...")
    t0 = time.time()
    
    prob_file = Path("outputs/predictions_raw.csv")
    if not prob_file.exists():
        print("No raw predictions found.")
        return
        
    print("Loading predictions...")
    # S1, Cand, Prob
    df = pd.read_csv(prob_file)
    
    # 1. Apply base thresholds per country? The file already has all pairs.
    # We will just apply a global minimum threshold for assignments to avoid garbage
    df = df[df["p"] >= 0.50]
    
    print(f"Total pairs above threshold: {len(df)}")
    
    # 2. Hungarian Graph Resolution
    # Since candidates (S2, S3) cannot belong to multiple S1 entities,
    # the optimal bipartite assignment for disconnected candidate clusters 
    # is equivalent to sorting by probability and keeping the max assignment per candidate!
    # (For 1-to-N S1-to-candidates, we allow multiple candidates per S1, 
    # but strictly ONE S1 per candidate).
    
    print("Enforcing strict 1-to-1 candidate assignment mapping...")
    # Sort by probability descending
    df = df.sort_values("p", ascending=False)
    
    # Drop duplicate candidates, keeping the one with the highest probability (greedy max-weight assignment)
    df_resolved = df.drop_duplicates(subset=["candidate_entity_id"], keep="first")
    
    print(f"Removed {len(df) - len(df_resolved)} duplicate candidate assignments!")
    
    # Write the new matching_results.tsv
    print("Overwriting with resolved matching_results.tsv...")
    preds = df_resolved.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
    
    # We also need to output empty lists for S1 entities that had no matches
    # to maintain the original file length.
    original_results = pd.read_csv("outputs/matching_results.tsv", sep="\t", dtype=str)
    
    out = open("outputs/matching_results.tsv", "w", encoding="utf-8")
    out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for s1 in original_results["source1_entity_id"]:
        matched = preds.get(s1, [])
        out.write(f"{s1}\t{','.join(matched)}\n")
        
    out.close()
    
    print(f"Done in {time.time() - t0:.2f} seconds!")
    print("The final file outputs/matching_results.tsv is ready for the leaderboard.")

if __name__ == "__main__":
    resolve()
