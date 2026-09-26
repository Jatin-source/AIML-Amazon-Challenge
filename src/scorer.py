"""Macro-averaged F_0.5 scorer for the Business Entity Resolution challenge."""

BETA2 = 0.25  # beta^2 with beta = 0.5


def entity_f_beta(pred_ids, true_ids):
    """F_0.5 for one Source-1 entity. Empty/empty scores 1.0."""
    pred, true = set(pred_ids), set(true_ids)
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    return (1 + BETA2) * precision * recall / (BETA2 * precision + recall)


def macro_f_beta(pred_map, truth_map):
    """Mean entity F_0.5 over every entity in truth_map (missing preds score 0/1.0)."""
    if not truth_map:
        return 0.0
    total = 0.0
    for eid, true_ids in truth_map.items():
        total += entity_f_beta(pred_map.get(eid, ()), true_ids)
    return total / len(truth_map)


def parse_id_list(value):
    """Split a comma-joined matched_entity_ids cell into a list of IDs."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    return [tok for tok in (t.strip() for t in s.split(",")) if tok]


def load_id_map(path, sep="\t"):
    """Read a two-column TSV into {source1_entity_id: [ids]}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(sep)
            out[parts[0].strip()] = parse_id_list(parts[1] if len(parts) > 1 else "")
    return out
