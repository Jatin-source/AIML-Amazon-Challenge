"""Normalisation v2: legal-form stripping + multi-country address tokens (Phase 12)."""
import re, unicodedata

# Legal forms are STRIPPED from core_name, not renamed. An identity map like
# {"sarl": "sarl"} changes nothing, so "Acme SARL" vs "Acme" stays unmatchable.
LEGAL_FORMS = {
    "corp", "corporation", "inc", "incorporated", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "llc", "plc", "holdings", "holding", "group", "intl",
    "international", "enterprises", "enterprise", "industries", "trust", "society",
    "sarl", "sas", "sasu", "eurl", "sci", "snc", "gie", "sa", "scop", "sem",
}
NAME_MAP = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd",
    "private": "pvt", "and": "&", "international": "intl",
}
ADDR = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd", "bd": "blvd",
    "bld": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
    "square": "sq", "highway": "hwy", "suite": "ste", "apartment": "apt",
    "floor": "fl", "building": "bldg", "block": "blk", "sector": "sec", "phase": "ph",
    "north": "n", "south": "s", "east": "e", "west": "w", "saint": "st", "sainte": "st",
    "avenida": "ave", "number": "no", "num": "no", "opposite": "opp",
    # France
    "rue": "r", "chemin": "che", "route": "rte", "impasse": "imp", "allee": "all",
    "passage": "psg", "faubourg": "fbg", "quai": "qu", "zone": "za",
    "cours": "crs", "villa": "vla", "residence": "res", "batiment": "bat",
    # India
    "nagar": "ngr", "colony": "col", "marg": "mg", "cross": "crs", "main": "mn",
    "extension": "ext", "layout": "lyt", "puram": "prm", "vihar": "vhr", "gali": "gli",
    "near": "nr", "behind": "bhd", "post": "po", "village": "vlg", "district": "dist",
}
# Single-token states. Multi-word states are handled by a bigram pass in norm_addr,
# because "madhya pradesh" arrives as two tokens and an unspaced key never fires.
STATE1 = {
    "maharashtra": "mh", "gujarat": "gj", "karnataka": "ka", "telangana": "tg",
    "rajasthan": "rj", "haryana": "hr", "punjab": "pb", "kerala": "kl",
    "odisha": "od", "orissa": "od", "bihar": "br", "jharkhand": "jh", "assam": "as",
    "chhattisgarh": "cg", "uttarakhand": "uk", "delhi": "dl", "goa": "ga",
    "manipur": "mn", "meghalaya": "ml", "tripura": "tr", "nagaland": "nl",
    "sikkim": "sk", "mizoram": "mz", "puducherry": "py", "chandigarh": "ch",
}
STATE2 = {
    ("madhya", "pradesh"): "mp", ("uttar", "pradesh"): "up",
    ("west", "bengal"): "wb", ("andhra", "pradesh"): "ap",
    ("himachal", "pradesh"): "hp", ("arunachal", "pradesh"): "ar",
    ("tamil", "nadu"): "tn", ("jammu", "kashmir"): "jk",
    ("new", "delhi"): "dl", ("uttar", "khand"): "uk",
}
JUNK = re.compile(r"^[^0-9a-z]+|[^0-9a-z]+$")
NONALNUM = re.compile(r"[^0-9a-z&]+")
WS = re.compile(r"\s+")
PIN = re.compile(r"^\d{5,6}$")


def fold(text):
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.encode("ascii", "ignore").decode("ascii").lower()


def _toks(text):
    return [t for t in NONALNUM.sub(" ", JUNK.sub("", fold(text))).split() if t]


def norm_name(raw):
    """Full canonical name: abbreviations unified, legal forms kept."""
    return WS.sub(" ", " ".join(NAME_MAP.get(t, t) for t in _toks(raw))).strip()


def core_name(raw):
    """Brand text with legal forms REMOVED. Falls back to the full name if empty."""
    toks = [NAME_MAP.get(t, t) for t in _toks(raw)]
    core = [t for t in toks if t not in LEGAL_FORMS]
    return " ".join(core) if core else " ".join(toks)


def legal_set(raw):
    """The legal-form tokens themselves, as a separate comparable signal."""
    return " ".join(sorted({t for t in _toks(raw) if t in LEGAL_FORMS}))


def norm_addr(raw):
    """Canonical address: bigram state codes, abbreviations, zero-stripped digits."""
    toks = _toks(raw)
    out, i = [], 0
    while i < len(toks):
        if i + 1 < len(toks) and (toks[i], toks[i + 1]) in STATE2:
            out.append(STATE2[(toks[i], toks[i + 1])])
            i += 2
            continue
        t = toks[i]
        t = STATE1.get(t, ADDR.get(t, t))
        if t.isdigit() and len(t) > 1 and not PIN.match(t):
            t = t.lstrip("0") or "0"
        out.append(t)
        i += 1
    return WS.sub(" ", " ".join(out)).strip()


def pincode(norm_address):
    """Postcode: scan from the END, skip index 0 (a leading run is a house number)."""
    toks = norm_address.split()
    for i in range(len(toks) - 1, 0, -1):
        if PIN.match(toks[i]):
            return toks[i]
    return ""
