"""Blocking v2: namespace-weighted IDF retrieval (word / char-ngram / address).

Two measured facts drive the design:

1. 64.4% of India blocking misses share ZERO name word-tokens
   ("vadodara minerals co" vs "vadodaraminerals com"). Word tokens cannot bridge a
   concatenation or an internal typo, so char n-grams of the core name are needed.
2. Dumping n-grams into ONE flat index DESTROYS recall (-10 to -13 pts, measured).
   A name emits ~20 n-grams against ~6 address tokens, so n-grams swamp the score,
   and address is the strongest single signal (95.6% of true pairs share an address
   token).

The fix is a per-namespace cosine with explicit weights: the three token spaces are
scored independently and then blended, instead of competing on raw token count.
"""
import numpy as np

MAX_DF = 50_000
TOP_K = 20
NGRAM = 4
WEIGHTS = (1.0, 0.45, 1.0)


def ngrams(core, n=NGRAM):
    c = core.replace(" ", "")
    if len(c) < n:
        return [c] if c else []
    return [c[i:i + n] for i in range(len(c) - n + 1)]


def tokenize(core, addr, n=NGRAM):
    """(namespace, token) pairs. 0 = core word, 1 = core char n-gram, 2 = address word."""
    out = [(0, t) for t in set(core.split())]
    out += [(1, g) for g in set(ngrams(core, n))]
    out += [(2, t) for t in set(addr.split())]
    return out


class Index2:
    """CSR inverted index with per-namespace IDF weights and per-namespace doc norms."""

    def __init__(self, cores, addrs, ngram=NGRAM):
        self.ngram = ngram
        self.n_docs = len(cores)
        vocab, rows, cols, tns = {}, [], [], []
        for i, (c, a) in enumerate(zip(cores, addrs)):
            for ns, t in tokenize(c, a, ngram):
                key = (ns, t)
                tid = vocab.get(key)
                if tid is None:
                    tid = vocab[key] = len(vocab)
                    tns.append(ns)
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
        self.token_ns = np.asarray(tns, dtype=np.int8)

        rep = np.diff(self.offsets)
        post_ns = np.repeat(self.token_ns, rep)
        post_w2 = np.repeat(self.idf.astype(np.float64) ** 2, rep)
        self.norms = np.empty((3, self.n_docs), dtype=np.float32)
        for k in range(3):
            m = post_ns == k
            sq = np.bincount(self.postings[m], weights=post_w2[m], minlength=self.n_docs)
            self.norms[k] = np.sqrt(np.maximum(sq, 1e-9))

    def nbytes(self):
        return int(self.postings.nbytes + self.offsets.nbytes +
                   self.idf.nbytes + self.norms.nbytes)

    def query(self, core, addr, top_k=TOP_K, max_df=MAX_DF, weights=WEIGHTS):
        """Weighted sum of per-namespace cosine similarities."""
        chunks, wts, nss = [], [], []
        qn = np.zeros(3, dtype=np.float64)
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
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        ids = np.concatenate(chunks)
        w = np.concatenate(wts)
        ns_arr = np.concatenate(nss)
        uniq = np.unique(ids)
        qn = np.sqrt(np.maximum(qn, 1e-9))
        scores = np.zeros(len(uniq), dtype=np.float32)
        for k in range(3):
            m = ns_arr == k
            if not m.any():
                continue
            sub, sw = ids[m], w[m]
            o = np.argsort(sub, kind="stable")
            sub, sw = sub[o], sw[o]
            b = np.flatnonzero(np.concatenate([[True], sub[1:] != sub[:-1]]))
            du = sub[b]
            dot = np.add.reduceat(sw, b)
            cos = dot / (self.norms[k][du] * qn[k])
            scores[np.searchsorted(uniq, du)] += weights[k] * cos.astype(np.float32)
        if len(uniq) > top_k:
            sel = np.argpartition(-scores, top_k)[:top_k]
            uniq, scores = uniq[sel], scores[sel]
        o = np.argsort(-scores)
        return uniq[o], scores[o]
