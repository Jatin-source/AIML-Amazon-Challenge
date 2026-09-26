"""Reproducible entity-level holdout split (Phase 5).

Splits train Source-1 entities 85/15, stratified by country x match-count bucket.
Entity-level only: an S1 entity and all of its matched S2/S3 records stay on one side.
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.15


def build(train_dir: Path):
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t",
                     usecols=["entity_id", "country"], dtype=str)
    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False, na_values=[])
    gt.columns = ["entity_id", "matched_entity_ids"]
    df = s1.merge(gt, on="entity_id", how="left", validate="one_to_one")
    ids = df["matched_entity_ids"].fillna("")
    df["n_matches"] = ids.str.count(",").add(1).where(ids != "", 0).astype("int16")
    df["bucket"] = df["country"] + "|" + df["n_matches"].clip(upper=6).astype(str)
    _, ho = train_test_split(df.index.values, test_size=TEST_SIZE,
                             stratify=df["bucket"].values, random_state=SEED)
    df["split"] = "train"
    df.loc[ho, "split"] = "holdout"
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build(Path(a.train_dir))
    df[["entity_id", "country", "n_matches", "split"]].to_parquet(
        out / "holdout_split.parquet", index=False)
    df.loc[df.split == "holdout", ["entity_id", "matched_entity_ids"]].to_csv(
        out / "holdout_ground_truth.tsv", sep="\t", index=False)
    print(df.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
