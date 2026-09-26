"""Pair features v2 (Phase 13b).

New over v1, each justified by a measurement or a verified data fact:

* ``num_conflict`` / ``pin_conflict`` - v1 only signalled number AGREEMENT. Both sides
  having numbers that all disagree is a near-certain false merge, and under F_0.5 a
  false positive costs 2x a miss. v1 gave the model no way to express that.
* ``core_*`` similarities - measured: 64.4% of India blocking misses share zero name
  word-tokens, largely legal-suffix and concatenation noise. Scoring the
  legal-stripped brand separately from the full name separates brand from boilerplate.
* ``legal_same`` / ``legal_both`` - the legal form kept as its own signal rather than
  being folded into the name, so "Acme SARL" vs "Acme SAS" is visibly different.
* four-ranker rank features - the candidate set is now a union of word-raw, word-cosine,
  gram-raw and gram-cosine, so which rankers found a pair is informative.

Country-agnostic by construction: nothing keys on a country label, so France (zero
training examples) flows through the identical code path.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

try:
    from rapidfuzz.process import cpdist as _cpdist
except ImportError:  # pragma: no cover
    _cpdist = None

SCORERS = {"ratio": fuzz.ratio, "tok_set": fuzz.token_set_ratio,
           "tok_sort": fuzz.token_sort_ratio, "part": fuzz.partial_ratio,
           "jw": JaroWinkler.normalized_similarity}
PIN_MIN, PIN_MAX = 5, 6


def _str_feats(left, right, prefix, scorers, workers=-1):
    out = {}
    for tag in scorers:
        sc = SCORERS[tag]
        if _cpdist is not None:
            v = np.asarray(_cpdist(left, right, scorer=sc, workers=workers), dtype=np.float32)
        else:
            v = np.fromiter((sc(a, b) for a, b in zip(left, right)),
                            dtype=np.float32, count=len(left))
        out[f"{prefix}_{tag}"] = v
    return out


def _token_feats(left, right, prefix):
    n = len(left)
    jac = np.empty(n, np.float32); cont = np.empty(n, np.float32)
    inter = np.empty(n, np.float32); ldiff = np.empty(n, np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        sa, sb = set(a.split()), set(b.split())
        k = len(sa & sb)
        jac[i] = k / (len(sa | sb) or 1)
        cont[i] = k / (min(len(sa), len(sb)) or 1)
        inter[i] = k
        ldiff[i] = abs(len(a) - len(b))
    return {f"{prefix}_jac": jac, f"{prefix}_cont": cont,
            f"{prefix}_inter": inter, f"{prefix}_lendiff": ldiff}


def _is_pin(t):
    return t.isdigit() and PIN_MIN <= len(t) <= PIN_MAX


def _num_pin_feats(left_addr, right_addr):
    """Numeric and postcode consistency, including explicit CONFLICT signals."""
    n = len(left_addr)
    same = np.zeros(n, np.float32); conflict = np.zeros(n, np.float32)
    both = np.zeros(n, np.float32); jac = np.zeros(n, np.float32)
    pin_exact = np.zeros(n, np.float32); pin_conflict = np.zeros(n, np.float32)
    pin_both = np.zeros(n, np.float32)
    for i, (a, b) in enumerate(zip(left_addr, right_addr)):
        ta, tb = a.split(), b.split()
        # postcode = last 5-6 digit token, never index 0 (that is a house number)
        pa = next((ta[j] for j in range(len(ta) - 1, 0, -1) if _is_pin(ta[j])), "")
        pb = next((tb[j] for j in range(len(tb) - 1, 0, -1) if _is_pin(tb[j])), "")
        if pa and pb:
            pin_both[i] = 1.0
            if pa == pb:
                pin_exact[i] = 1.0
            else:
                pin_conflict[i] = 1.0
        na = {t for t in ta if t.isdigit() and t != pa}
        nb = {t for t in tb if t.isdigit() and t != pb}
        if na and nb:
            both[i] = 1.0
            k = len(na & nb)
            jac[i] = k / len(na | nb)
            if k:
                same[i] = 1.0
            else:
                conflict[i] = 1.0
    return {"num_any_same": same, "num_conflict": conflict, "num_both_present": both,
            "num_jac": jac, "pin_exact": pin_exact, "pin_conflict": pin_conflict,
            "pin_both_present": pin_both}


def _legal_feats(left, right):
    n = len(left)
    same = np.zeros(n, np.float32); both = np.zeros(n, np.float32)
    jac = np.zeros(n, np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        sa, sb = set(a.split()), set(b.split())
        if sa and sb:
            both[i] = 1.0
            k = len(sa & sb)
            jac[i] = k / len(sa | sb)
            same[i] = 1.0 if k else 0.0
    return {"legal_same": same, "legal_both": both, "legal_jac": jac}


def build(df, workers=-1):
    """Feature frame. Needs: s1_name/core/legal/addr, c_name/core/legal/addr,
    w_raw_rank, w_cos_rank, g_raw_rank, g_cos_rank, w_cos_score, g_cos_score."""
    f = {}
    f.update(_str_feats(df.s1_core.values, df.c_core.values, "core",
                        ["ratio", "tok_set", "tok_sort", "part", "jw"], workers))
    f.update(_str_feats(df.s1_name.values, df.c_name.values, "name",
                        ["tok_set", "jw"], workers))
    f.update(_str_feats(df.s1_addr.values, df.c_addr.values, "addr",
                        ["ratio", "tok_set", "tok_sort", "part", "jw"], workers))
    f.update(_token_feats(df.s1_core.values, df.c_core.values, "core"))
    f.update(_token_feats(df.s1_addr.values, df.c_addr.values, "addr"))
    f.update(_num_pin_feats(df.s1_addr.values, df.c_addr.values))
    f.update(_legal_feats(df.s1_legal.values, df.c_legal.values))

    for c in ["in_word", "in_ngram", "in_addr", "in_phonetic", "cos_score", "score_gap"]:
        if c in df.columns:
            f[c] = df[c].values.astype(np.float32)
            
    if "in_word" in df.columns:
        ranks = np.stack([df[c].values for c in ["in_word", "in_ngram", "in_addr", "in_phonetic"]])
        f["n_rankers_found"] = ranks.sum(axis=0).astype(np.float32)
        f["best_rank"] = ranks.max(axis=0).astype(np.float32)
    f["best_rank"] = ranks.min(axis=0).astype(np.float32)
    f["is_s3"] = np.char.startswith(
        df.candidate_entity_id.values.astype(str), "S3-").astype(np.float32)

    X = pd.DataFrame(f)
    g = X.groupby(df.source1_entity_id.values)
    for col in ["cos_score", "core_tok_set", "addr_tok_set"]:
        mx = g[col].transform("max")
        X[f"{col}_rel"] = (X[col] / mx.replace(0, np.nan)).fillna(0)
        X[f"{col}_gap"] = mx - X[col]
    X["n_cand"] = g["cos_score"].transform("size")
    return X.astype(np.float32)
