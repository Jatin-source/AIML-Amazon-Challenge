"""Memory-safe inverted-index blocking (Phase 7c).

Single-process by design. The Phase 7c kernel death was caused by handing a large
Python token index to a 64-worker fork pool: copy-on-write duplicated it per worker
and crossed the 252.3 GB cgroup limit. Here the index is a CSR-style trio of NumPy
arrays built once in-process, so there is no per-worker duplication.
"""
import numpy as np

MAX_DF = 50_000   # query-time skip: tokens above this are near-zero IDF and cost a lot
TOP_K = 50


def tokenize(n_name, n_addr):
    """Namespaced token list: name and address token spaces stay distinct."""
    return ["n:" + t for t in n_name.split()] + ["a:" + t for t in n_addr.split()]


class InvertedIndex:
    """CSR inverted index over a record pool, with IDF weights."""

    def __init__(self, names, addrs):
        self.n_docs = len(names)
        vocab, rows, cols = {}, [], []
        for i, (nm, ad) in enumerate(zip(names, addrs)):
            for t in set(tokenize(nm, ad)):
                tid = vocab.get(t)
                if tid is None:
                    tid = vocab[t] = len(vocab)
                rows.append(tid)
                cols.append(i)
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        order = np.argsort(rows, kind="stable")
        self.postings = cols[order]
        counts = np.bincount(rows, minlength=len(vocab)).astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        self.df = counts
        self.idf = np.log(1.0 + self.n_docs / np.maximum(counts, 1)).astype(np.float32)
        self.vocab = vocab

    def nbytes(self):
        return int(self.postings.nbytes + self.offsets.nbytes + self.idf.nbytes)

    def query(self, n_name, n_addr, top_k=TOP_K, max_df=MAX_DF):
        """Return (record_indices, idf_overlap_scores) for the top_k best candidates."""
        chunks, weights = [], []
        for t in set(tokenize(n_name, n_addr)):
            tid = self.vocab.get(t)
            if tid is None or self.df[tid] > max_df:
                continue
            lo, hi = self.offsets[tid], self.offsets[tid + 1]
            chunks.append(self.postings[lo:hi])
            weights.append(np.full(hi - lo, self.idf[tid], dtype=np.float32))
        if not chunks:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        ids = np.concatenate(chunks)
        w = np.concatenate(weights)
        order = np.argsort(ids, kind="stable")
        ids, w = ids[order], w[order]
        bounds = np.flatnonzero(np.concatenate([[True], ids[1:] != ids[:-1]]))
        uniq = ids[bounds]
        scores = np.add.reduceat(w, bounds)
        if len(uniq) > top_k:
            sel = np.argpartition(-scores, top_k)[:top_k]
            uniq, scores = uniq[sel], scores[sel]
        order = np.argsort(-scores)
        return uniq[order], scores[order]

    def build_norms(self):
        """L2 norm of each doc's IDF vector, for cosine scoring."""
        rep = np.diff(self.offsets)
        w = np.repeat(self.idf.astype(np.float64) ** 2, rep)
        sq = np.bincount(self.postings, weights=w, minlength=self.n_docs)
        self.norms = np.sqrt(np.maximum(sq, 1e-9)).astype(np.float32)
        return self

    def query_cosine(self, n_name, n_addr, top_k=TOP_K, max_df=MAX_DF):
        """Cosine-normalised retrieval: divides IDF overlap by the doc's L2 norm.

        Raw IDF-sum ranking is length-biased - a verbose address accumulates more
        matching tokens than a terse true match, which buried India's gold at p99
        rank ~19,700. Dividing by the doc norm removes that bias.
        """
        if not hasattr(self, "norms"):
            self.build_norms()
        chunks, weights = [], []
        qnorm = 0.0
        for t in set(tokenize(n_name, n_addr)):
            tid = self.vocab.get(t)
            if tid is None:
                continue
            qnorm += float(self.idf[tid]) ** 2
            if self.df[tid] > max_df:
                continue
            lo, hi = self.offsets[tid], self.offsets[tid + 1]
            chunks.append(self.postings[lo:hi])
            weights.append(np.full(hi - lo, self.idf[tid], dtype=np.float32))
        if not chunks:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        ids = np.concatenate(chunks)
        w = np.concatenate(weights)
        order = np.argsort(ids, kind="stable")
        ids, w = ids[order], w[order]
        bounds = np.flatnonzero(np.concatenate([[True], ids[1:] != ids[:-1]]))
        uniq = ids[bounds]
        dot = np.add.reduceat(w, bounds)
        scores = dot / (self.norms[uniq] * np.sqrt(max(qnorm, 1e-9)))
        if len(uniq) > top_k:
            sel = np.argpartition(-scores, top_k)[:top_k]
            uniq, scores = uniq[sel], scores[sel]
        order = np.argsort(-scores)
        return uniq[order], scores[order]
