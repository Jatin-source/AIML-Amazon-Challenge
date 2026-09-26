"""Pair features for the matching model (Phase 8b).

Country-agnostic by construction: nothing is keyed on a country label, so the France
test slice (zero training examples) flows through the same code path as US and India.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

try:                                     # rapidfuzz >= 3.6: C++-level parallel cdist
    from rapidfuzz.process import cpdist as _cpdist
except ImportError:                      # pragma: no cover
    _cpdist = None

STR_SCORERS = {
    "ratio": fuzz.ratio,
    "tok_set": fuzz.token_set_ratio,
    "tok_sort": fuzz.token_sort_ratio,
    "part": fuzz.partial_ratio,
    "jw": JaroWinkler.normalized_similarity,
}


def _str_feats(left, right, prefix, workers=-1):
    """Elementwise string similarity. Uses cpdist (releases the GIL) when available."""
    out = {}
    for tag, scorer in STR_SCORERS.items():
        if _cpdist is not None:
            v = np.asarray(_cpdist(left, right, scorer=scorer, workers=workers),
                           dtype=np.float32)
        else:
            v = np.fromiter((scorer(a, b) for a, b in zip(left, right)),
                            dtype=np.float32, count=len(left))
        out[f"{prefix}_{tag}"] = v
    return out


def _token_feats(left, right, prefix):
    """Jaccard, containment, intersection size, length delta over tokens."""
    n = len(left)
    jac = np.empty(n, dtype=np.float32); cont = np.empty(n, dtype=np.float32)
    inter = np.empty(n, dtype=np.float32); ldiff = np.empty(n, dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        sa, sb = set(a.split()), set(b.split())
        k = len(sa & sb)
        jac[i] = k / (len(sa | sb) or 1)
        cont[i] = k / (min(len(sa), len(sb)) or 1)
        inter[i] = k
        ldiff[i] = abs(len(a) - len(b))
    return {f"{prefix}_jac": jac, f"{prefix}_cont": cont,
            f"{prefix}_inter": inter, f"{prefix}_lendiff": ldiff}


def _num_feats(left, right):
    """Digit-token agreement: house/building numbers and postcodes."""
    n = len(left)
    same = np.zeros(n, dtype=np.float32); jac = np.zeros(n, dtype=np.float32)
    both = np.zeros(n, dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        na = {t for t in a.split() if t.isdigit()}
        nb = {t for t in b.split() if t.isdigit()}
        both[i] = bool(na) and bool(nb)
        if na and nb:
            k = len(na & nb)
            same[i] = k > 0
            jac[i] = k / len(na | nb)
    return {"num_any_same": same, "num_jac": jac, "num_both_present": both}


def build(df, workers=-1):
    """Return a float32 feature frame for a candidate-pair frame.

    Needs columns: source1_entity_id, candidate_entity_id, s1_name, s1_addr,
    c_name, c_addr, raw_rank, cos_rank, cos_score.
    """
    feats = {}
    feats.update(_str_feats(df.s1_name.values, df.c_name.values, "name", workers))
    feats.update(_str_feats(df.s1_addr.values, df.c_addr.values, "addr", workers))
    feats.update(_token_feats(df.s1_name.values, df.c_name.values, "name"))
    feats.update(_token_feats(df.s1_addr.values, df.c_addr.values, "addr"))
    feats.update(_num_feats(df.s1_addr.values, df.c_addr.values))

    feats["raw_rank"] = df.raw_rank.values.astype(np.float32)
    feats["cos_rank"] = df.cos_rank.values.astype(np.float32)
    feats["cos_score"] = df.cos_score.values.astype(np.float32)
    feats["in_both_rankers"] = ((df.raw_rank.values < 99) &
                                (df.cos_rank.values < 99)).astype(np.float32)
    feats["is_s3"] = np.char.startswith(
        df.candidate_entity_id.values.astype(str), "S3-").astype(np.float32)

    X = pd.DataFrame(feats)

    # Per-entity relative features: this candidate versus the entity's best candidate.
    g = X.groupby(df.source1_entity_id.values)
    for col in ["cos_score", "name_tok_set", "addr_tok_set"]:
        mx = g[col].transform("max")
        X[f"{col}_rel"] = (X[col] / mx.replace(0, np.nan)).fillna(0)
        X[f"{col}_gap"] = mx - X[col]
    X["n_cand"] = g["cos_score"].transform("size")
    return X.astype(np.float32)
