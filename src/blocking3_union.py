"""Blocking v3-union: independent top-K per namespace, then union of id sets.

Rationale: Index3.query() merges all four namespaces into ONE weighted score
(WEIGHTS = 1.0, 0.45, 1.0, 0.5) and then takes a single top-K. A candidate found
only by the char-ngram namespace is scaled by 0.45 and loses to clean word
matches -> score dilution. query_union() removes that competition: each
namespace contributes its own top-K, and the union survives to the ranker.
"""
import numpy as np
from src.blocking3 import Index3, tokenize, MAX_DF, NGRAM, WEIGHTS

K_PER_NS = 20


class Index3Union(Index3):
    def _per_ns(self, core, addr, max_df):
        """Return {ns: (doc_ids, cosine)} computed independently per namespace."""
        chunks, wts, nss = [], [], []
        qn = np.zeros(4, dtype=np.float64)
        for ns, t in tokenize(core, addr, self.ngram):
            tid = self.vocab.get((ns, t))
            if tid is None:
                continue
            w = self.idf[tid]
            qn[ns] += float(w) ** 2
            if self.df[tid] > max_df:
                continue
            lo, hi = self.offsets[tid], self.offsets[tid + 1]
            chunks.append(self.postings[lo:hi])
            wts.append(np.full(hi - lo, w, dtype=np.float32))
            nss.append(np.full(hi - lo, ns, dtype=np.int8))
        if not chunks:
            return {}, qn
        ids = np.concatenate(chunks)
        w = np.concatenate(wts)
        ns_arr = np.concatenate(nss)
        qn = np.sqrt(np.maximum(qn, 1e-9))

        out = {}
        for k in range(4):
            m = ns_arr == k
            if not m.any():
                continue
            sub, sw = ids[m], w[m]
            o = np.argsort(sub, kind="stable")
            sub, sw = sub[o], sw[o]
            b = np.flatnonzero(np.concatenate([[True], sub[1:] != sub[:-1]]))
            du = sub[b]
            dot = np.add.reduceat(sw, b)
            cos = (dot / (self.norms[k][du] * qn[k])).astype(np.float32)
            out[k] = (du, cos)
        return out, qn

    def query_union(self, core, addr, k_per_ns=K_PER_NS, max_df=MAX_DF,
                    weights=WEIGHTS, cap=None):
        """Independent top-k per namespace, unioned. Same return signature as query()."""
        per_ns, _ = self._per_ns(core, addr, max_df)
        if not per_ns:
            return (np.empty(0, np.int32), np.empty(0, np.float32),
                    np.empty((0, 4), bool))

        picked = []
        tops = {}
        for k, (du, cos) in per_ns.items():
            if len(du) > k_per_ns:
                sel = np.argpartition(-cos, k_per_ns)[:k_per_ns]
            else:
                sel = np.arange(len(du))
            tops[k] = set(du[sel].tolist())
            picked.append(du[sel])

        uniq = np.unique(np.concatenate(picked))
        scores = np.zeros(len(uniq), dtype=np.float32)
        origins = np.zeros((len(uniq), 4), dtype=bool)

        for k, (du, cos) in per_ns.items():
            keep = np.isin(du, uniq)
            if not keep.any():
                continue
            pos = np.searchsorted(uniq, du[keep])
            scores[pos] += weights[k] * cos[keep]
            origins[pos, k] = True

        if cap is not None and len(uniq) > cap:
            sel = np.argpartition(-scores, cap)[:cap]
            uniq, scores, origins = uniq[sel], scores[sel], origins[sel]

        o = np.argsort(-scores)
        return uniq[o], scores[o], origins[o]
